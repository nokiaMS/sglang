# 模型权重检查器，用于快照、重置、比较和校验模型权重
# 支持FP8量化权重的反量化比较，检测权重意外修改和分布式训练中的不一致
import hashlib  # 导入哈希计算模块
import logging  # 导入日志记录模块
import time  # 导入时间模块
from typing import Dict, Iterable, Optional, Set, Tuple  # 导入类型注解

import torch  # 导入PyTorch张量库
import torch.distributed as dist  # 导入PyTorch分布式通信模块
from pydantic import BaseModel, ConfigDict  # 导入Pydantic模型基类和配置

from sglang.srt.layers.quantization.fp8_utils import (  # 导入FP8量化工具
    block_quant_dequant,  # 块量化反量化函数
    inverse_transform_scale_ue8m0,  # UE8M0缩放因子逆变换函数
)
from sglang.srt.managers.mm_utils import tensor_hash  # 导入张量哈希计算工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class _StrictBaseModel(BaseModel):  # 严格Pydantic模型基类，禁止额外字段
    model_config = ConfigDict(extra="forbid")  # 配置禁止额外字段


class ParallelismInfo(_StrictBaseModel):  # 并行信息模型
    tp_rank: int  # 张量并行秩
    tp_size: int  # 张量并行大小
    dp_rank: int  # 数据并行秩
    dp_size: int  # 数据并行大小
    pp_rank: int  # 流水线并行秩
    pp_size: int  # 流水线并行大小
    rank: int  # 全局秩
    size: int  # 全局大小


class ChecksumInfo(_StrictBaseModel):  # 校验和信息模型
    checksums: Dict[str, str]  # 各权重张量的校验和
    per_gpu_checksum: str  # 单GPU整体校验和
    parallelism_info: ParallelismInfo  # 并行信息


_NON_PERSISTENT_BUFFER_PATTERNS = (  # 非持久缓冲区名称模式（这些缓冲区在权重加载后重新计算）
    "cos_sin_cache",  # 余弦正弦缓存
    "inv_freq",  # 逆频率
    "freqs_cis",  # 频率复数
    "_weight_fp32",  # FP32权重副本
)


def _is_non_persistent_buffer_name(name: str) -> bool:  # 判断名称是否为非持久缓冲区
    return any(pat in name for pat in _NON_PERSISTENT_BUFFER_PATTERNS)  # 检查名称是否包含任一模式


class WeightChecker:  # 权重检查器类
    def __init__(self, model_runner):  # 初始化权重检查器
        self._model_runner = model_runner  # 保存模型运行器引用
        self._snapshot_tensors = None  # 快照张量字典（初始为None）

    def handle(self, action: str) -> Optional[Dict]:  # 处理权重检查操作
        logger.info(f"[WeightChecker] handle action={action}")  # 记录操作类型
        if action == "snapshot":  # 快照操作
            return self._snapshot()  # 执行快照
        elif action == "reset_tensors":  # 重置张量操作
            return self._reset_tensors()  # 执行重置
        elif action == "compare":  # 比较操作
            return self._compare()  # 执行比较
        elif action == "checksum":  # 校验和操作
            return self._compute_checksum()  # 计算校验和
        else:  # 不支持的操作
            raise Exception(f"Unsupported {action=}")  # 抛出异常

    def _snapshot(self):  # 快照当前模型权重
        named_tensors = [  # 收集所有命名张量
            (name, param.data.detach().cpu()) for name, param in self._model_state()  # 分离并移到CPU
        ]
        self._snapshot_tensors = dict(named_tensors)  # 转为字典保存
        assert len(self._snapshot_tensors) == len(  # 确保没有重复的名称
            named_tensors
        ), f"should not have duplicated tensor name"  # 不应有重复的张量名称

    def _reset_tensors(self):  # 用随机值重置模型权重
        for name, param in self._model_state():  # 遍历所有命名参数
            if _is_non_persistent_buffer_name(name):  # 跳过非持久缓冲区
                continue  # 不重置非持久缓冲区
            param.copy_(_random_like(param))  # 用随机值替换参数

    def _compare(self):  # 比较当前权重与快照权重
        assert self._snapshot_tensors is not None  # 确保已有快照

        skip_compare_names = {  # 收集跳过比较的参数名
            name
            for name, param in self._model_state()
            if getattr(param, "_skip_weight_check", False)  # 检查是否标记跳过权重检查
        }
        _check_tensors(  # 检查快照和当前权重是否一致
            expect_tensors=_postprocess_tensors(  # 后处理快照张量
                self._snapshot_tensors, skip_compare_names  # 传入快照和跳过名称
            ),
            actual_tensors=_postprocess_tensors(  # 后处理当前张量
                dict(self._model_state()), skip_compare_names  # 传入当前状态和跳过名称
            ),
        )

    def _compute_checksum(self) -> Dict:  # 计算模型权重的校验和
        torch.cuda.synchronize()  # 同步CUDA操作
        start = time.perf_counter()  # 记录开始时间

        skip_compare_names = {  # 收集跳过比较的参数名
            name
            for name, param in self._model_state()
            if getattr(param, "_skip_weight_check", False)  # 检查是否标记跳过权重检查
        }

        # Reuse the snapshot/compare postprocess pipeline so fp8 weights are  # 复用快照/比较的后处理管道，使fp8权重
        # dequantized to bf16 before hashing — two (qweight, scale) pairs that  # 在哈希前反量化为bf16——两个产生相同bf16的
        # produce the same bf16 must produce the same checksum.  # (qweight, scale)对必须产生相同的校验和
        checksums = {  # 计算各张量的校验和
            name: _hash_tensor(tensor.data)  # 计算张量哈希
            for name, should_compare, tensor in _postprocess_tensors(  # 后处理张量
                dict(self._model_state()), skip_compare_names  # 传入当前状态和跳过名称
            )
            if should_compare  # 仅计算需要比较的张量
        }

        h = hashlib.sha256()  # 创建SHA256哈希对象
        for name in sorted(checksums):  # 按名称排序遍历
            h.update(name.encode())  # 更新名称到哈希
            h.update(checksums[name].encode())  # 更新校验和到哈希
        overall = h.hexdigest()  # 获取整体校验和

        torch.cuda.synchronize()  # 同步CUDA操作
        elapsed = time.perf_counter() - start  # 计算耗时
        logger.info(  # 记录校验和计算完成
            f"[WeightChecker] checksum computed for {len(checksums)} tensors in {elapsed:.3f}s"  # 显示张量数量和耗时
        )

        info = ChecksumInfo(  # 创建校验和信息
            checksums=checksums,  # 各张量校验和
            per_gpu_checksum=overall,  # 整体校验和
            parallelism_info=self._parallelism_info(),  # 并行信息
        )
        return info.model_dump()  # 返回模型转储的字典

    def _parallelism_info(self) -> ParallelismInfo:  # 获取当前并行信息
        mr = self._model_runner  # 获取模型运行器
        return ParallelismInfo(  # 返回并行信息
            tp_rank=mr.tp_rank,  # 张量并行秩
            tp_size=mr.tp_size,  # 张量并行大小
            dp_rank=mr.dp_rank if mr.dp_rank is not None else 0,  # 数据并行秩（默认0）
            dp_size=mr.dp_size,  # 数据并行大小
            pp_rank=mr.pp_rank,  # 流水线并行秩
            pp_size=mr.pp_size,  # 流水线并行大小
            rank=dist.get_rank() if dist.is_initialized() else 0,  # 全局秩
            size=dist.get_world_size() if dist.is_initialized() else 1,  # 全局大小
        )

    def _model_state(self):  # 迭代模型的所有命名参数和缓冲区
        yield from self._model_runner.model.named_parameters()  # 产出命名参数
        yield from self._model_runner.model.named_buffers()  # 产出命名缓冲区


def _hash_tensor(t: torch.Tensor) -> str:  # 计算张量的哈希值
    return f"{tensor_hash(t):016x}"  # 返回16位十六进制哈希字符串


def _check_tensors(  # 检查两组张量是否一致
    expect_tensors: Iterable[Tuple[str, bool, torch.Tensor]],  # 期望张量迭代器
    actual_tensors: Iterable[Tuple[str, bool, torch.Tensor]],  # 实际张量迭代器
):
    from sglang.srt.debug_utils.dumper import get_tensor_info  # 导入张量信息获取工具

    good_names = []  # 一致的张量名称列表
    error_messages = []  # 错误消息列表
    info_messages = []  # 信息消息列表

    for (expect_name, expect_should_compare, expect), (  # 遍历期望和实际张量对
        actual_name,
        actual_should_compare,
        actual,
    ) in zip(expect_tensors, actual_tensors, strict=True):  # 严格模式配对
        assert expect_name == actual_name, f"{expect_name=} {actual_name=}"  # 确保名称一致
        assert (
            expect_should_compare == actual_should_compare
        ), f"{expect_should_compare=} {actual_should_compare=}"  # 确保比较标志一致
        name = expect_name  # 获取名称
        should_compare = expect_should_compare  # 获取是否应比较标志

        expect = expect.cuda()  # 移到GPU
        actual = actual.cuda()  # 移到GPU

        if torch.all(expect == actual):  # 如果完全相等
            good_names.append(name)  # 添加到一致列表
        else:  # 如果不相等
            abs_diff = (actual.float() - expect.float()).abs()  # 计算绝对差值
            msg = (  # 构造错误消息
                f"name={name} "
                f"max_abs_err={abs_diff.max()} "  # 最大绝对误差
                f"mean_abs_err={abs_diff.mean()} "  # 平均绝对误差
                f"{get_tensor_info(expect)=} "  # 期望张量信息
                f"{get_tensor_info(actual)=} "  # 实际张量信息
            )
            (error_messages if should_compare else info_messages).append(msg)  # 根据是否应比较分类

    logger.info(f"[check_tensors] equal tensors: {good_names}")  # 记录一致的张量
    if len(info_messages) > 0:  # 如果有信息消息
        logger.info(f"[check_tensors] info: {info_messages}")  # 记录信息消息
    if len(error_messages) > 0:  # 如果有错误消息
        raise Exception(f"check tensor equality failed:\n" + "\n".join(error_messages))  # 抛出异常


def _random_like(t: torch.Tensor):  # 创建与给定张量形状和类型相同的随机张量
    device = t.device  # 获取设备
    shape = t.shape  # 获取形状
    dtype = t.dtype  # 获取数据类型

    if dtype.is_floating_point:  # 如果是浮点类型
        return torch.rand(shape, device=device, dtype=torch.float32).to(dtype)  # 生成随机浮点值

    if dtype == torch.bool:  # 如果是布尔类型
        return torch.rand(shape, device=device) > 0.5  # 随机生成True/False

    info = torch.iinfo(dtype)  # 获取整数类型信息
    return torch.randint(  # 生成随机整数值
        low=int(info.min), high=int(info.max), size=shape, device=device, dtype=dtype  # 在类型范围内随机
    )


def _postprocess_tensors(  # 后处理张量：跳过非持久缓冲区和量化权重，反量化FP8权重
    raw: Dict[str, torch.Tensor],  # 原始张量字典
    skip_compare_names: Set[str],  # 跳过比较的名称集合
) -> Iterable[Tuple[str, bool, torch.Tensor]]:  # 返回（名称，是否比较，张量）元组迭代器
    from sglang.srt.debug_utils.dumper import get_tensor_info  # 导入张量信息获取工具

    skip_compare_names = set(skip_compare_names)  # 复制跳过名称集合

    # Skip non-persistent buffers (registered with persistent=False; recomputed  # 跳过非持久缓冲区（以persistent=False注册；在权重加载后
    # after weight load and not part of the synced payload).  # 重新计算，不属于同步载荷）
    for name in raw:  # 遍历所有张量名称
        if _is_non_persistent_buffer_name(name):  # 如果是非持久缓冲区
            skip_compare_names.add(name)  # 添加到跳过集合
            logger.info(f"[check_tensors] Skipping non-persistent buffer: {name}")  # 记录跳过信息

    # dequant fp8  # 反量化FP8权重
    quant_names = [  # 收集量化权重名称
        name
        for name in raw
        # Match: `something.weight`, `something.experts.w2_weight`  # 匹配：`something.weight`、`something.experts.w2_weight`
        if name.endswith("weight") and name.replace("weight", "weight_scale_inv") in raw  # 如果有权重且有对应缩放因子
    ]
    quant_scale_names = [  # 收集量化缩放因子名称
        name.replace("weight", "weight_scale_inv") for name in quant_names  # 替换weight为weight_scale_inv
    ]
    skip_compare_names.update(quant_names)  # 跳过原始量化权重
    skip_compare_names.update(quant_scale_names)  # 跳过缩放因子
    for name in quant_names:  # 遍历量化权重名称
        w_q = raw[name]  # 获取量化权重
        w_s = raw[name.replace("weight", "weight_scale_inv")]  # 获取缩放因子

        try:  # 尝试反量化
            if w_s.dtype == torch.int32:  # 如果缩放因子是int32类型
                # UE8M0 packed format (Blackwell DeepGEMM)  # UE8M0打包格式（Blackwell DeepGEMM）
                w_s_for_dequant = inverse_transform_scale_ue8m0(w_s, mn=w_q.shape[-2])  # 逆变换缩放因子
            else:  # 其他类型
                w_s_for_dequant = w_s  # 直接使用缩放因子

            w_dequant = block_quant_dequant(  # 块量化反量化
                w_q,  # 量化权重
                w_s_for_dequant,  # 缩放因子
                # TODO do not hardcode  # 待办：不要硬编码
                block_size=[128, 128],  # 块大小128x128
                dtype=torch.bfloat16,  # 反量化为bfloat16
            )
            yield name, True, w_dequant  # 产出反量化后的权重（标记为需要比较）
        except Exception as e:  # 如果反量化失败
            e.add_note(  # 添加错误备注
                f"when handling {name=} {get_tensor_info(w_q)=} {get_tensor_info(w_s)=}"  # 显示张量信息
            )
            raise  # 重新抛出异常

    for name in raw:  # 遍历所有张量
        should_compare = name not in skip_compare_names  # 检查是否应比较
        yield name, should_compare, raw[name]  # 产出名称、比较标志和张量
