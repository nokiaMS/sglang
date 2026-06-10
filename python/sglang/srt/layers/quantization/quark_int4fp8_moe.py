# Quark INT4-FP8 MoE 量化配置与方法实现，支持在线将 BF16/FP16 权重量化为 INT4 并以 FP8 进行推理计算
import logging  # 导入日志模块
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # 导入类型提示

import torch  # 导入 PyTorch
from tqdm import tqdm  # 导入进度条库
from tqdm.std import EMA  # 导入指数移动平均类

from sglang.srt.distributed import get_tensor_model_parallel_rank  # 导入获取张量并行排名函数
from sglang.srt.layers.int4fp8_utils import (  # 导入 INT4-FP8 工具函数
    pack_int4_to_int32,  # 将 INT4 打包为 INT32
    quantize_fp8_scale_tensorwise,  # 逐张量 FP8 缩放量化
    quantize_int4_scale_columnwise,  # 逐列 INT4 缩放量化
)
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig  # 导入 MoE 运行器相关类
from sglang.srt.layers.quantization.base_config import (  # 导入量化基础配置类
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod  # 导入 FP8 线性方法
from sglang.srt.utils import BAR_FORMAT, is_hip, set_weight_attrs  # 导入工具函数

if TYPE_CHECKING:  # 仅用于类型检查
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutput  # 导入分发输出类型

_is_hip = is_hip()  # 判断当前是否为 AMD HIP 平台


if _is_hip:  # 如果是 HIP 平台
    from aiter.ops.shuffle import shuffle_weight  # 导入权重重排函数

    ON_GFX950 = "gfx950" in torch.cuda.get_device_properties("cuda").gcnArchName  # 检测是否为 gfx950 GPU

logger = logging.getLogger(__name__)  # 初始化日志记录器


def tqdm_reset_no_print(tqdm_bar: tqdm, total=None):  # 重置进度条但不打印
    tqdm_bar.n = 0  # 重置当前计数
    if total is not None:  # 如果指定了总数
        tqdm_bar.total = total  # 更新总数
    if tqdm_bar.disable:  # 如果进度条被禁用
        return  # 直接返回
    tqdm_bar.last_print_n = 0  # 重置上次打印计数
    tqdm_bar.last_print_t = tqdm_bar.start_t = tqdm_bar._time()  # 重置上次打印时间和开始时间
    tqdm_bar._ema_dn = EMA(tqdm_bar.smoothing)  # 重置指数移动平均
    tqdm_bar._ema_dt = EMA(tqdm_bar.smoothing)  # 重置指数移动平均
    tqdm_bar._ema_miniters = EMA(tqdm_bar.smoothing)  # 重置最小迭代指数移动平均


class QuarkInt4Fp8Config(QuantizationConfig):  # Quark INT4-FP8 量化配置类
    """Config class for Quark Quantization.  # Quark 量化配置类

    - Weight: static, per-channel, symmetric  # 权重：静态，逐通道，对称
    - Activation: dynamic, per-token, symmetric  # 激活：动态，逐 token，对称
    """

    def __init__(  # 初始化方法
        self,
        is_checkpoint_fp8_serialized: bool = False,  # 检查点是否以 FP8 格式序列化
        activation_scheme: str = "dynamic",  # 激活量化方案
    ):
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized  # 保存 FP8 序列化标志
        self.activation_scheme = activation_scheme  # 保存激活量化方案

        if activation_scheme != "dynamic":  # 如果激活方案不是动态
            raise NotImplementedError(  # 抛出未实现错误
                "QuarkInt4Fp8Config only supports activation_scheme='dynamic'."
            )  # QuarkInt4Fp8Config 仅支持动态激活方案

        self.weight_block_size = None  # 权重块大小，未使用

        self.num_quant_layers = 0  # 量化层数计数器

        tp_rank = get_tensor_model_parallel_rank()  # 获取当前张量并行排名

        # The weight iterator already has a progress bar on rank=0, account for that.  # 权重迭代器在 rank=0 已有进度条，需适配
        position = 1 + tqdm._get_free_pos()  # 计算进度条位置
        self.online_quant_progress_bar = tqdm(  # 创建在线量化进度条
            total=0,  # 初始总数为 0
            desc=f"Online quark_int4fp8_moe quantization on rank={tp_rank}",  # 进度条描述
            position=position,  # 进度条位置
            bar_format=BAR_FORMAT,  # 进度条格式
            mininterval=2.0,  # 最小更新间隔
        )

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:  # 获取支持的激活数据类型
        return [torch.float16, torch.bfloat16]  # 支持 float16 和 bfloat16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低 GPU 计算能力要求
        return 70  # 最低计算能力为 70

    @classmethod
    def get_name(self) -> str:  # 获取量化方法名称
        return "quark_int4fp8_moe"  # 返回 quark_int4fp8_moe

    @classmethod
    def get_config_filenames(cls) -> List[str]:  # 获取配置文件名列表
        return []  # 返回空列表

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuarkInt4Fp8Config":  # 从配置字典创建实例
        return cls()  # 返回默认配置实例

    def get_quant_method(  # 获取量化方法
        self,
        layer: torch.nn.Module,  # 目标层
        prefix: str,  # 层名前缀
    ) -> Optional["QuantizeMethodBase"]:
        # TODO: fix circular imports issues in sglang forcing us to import here instead of at
        # the top of file.  # TODO: 修复 sglang 中的循环导入问题，迫使我们在此处而非文件顶部导入
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合 MoE 层

        if isinstance(layer, LinearBase):  # 如果是线性层
            return Fp8LinearMethod(self)  # 返回 FP8 线性方法
        elif isinstance(layer, FusedMoE):  # 如果是融合 MoE 层
            return QuarkInt4Fp8MoEMethod(self)  # 返回 Quark INT4-FP8 MoE 方法

        return None  # 其他层返回 None

    def get_scaled_act_names(self) -> List[str]:  # 获取需要缩放的激活名称列表
        return []  # 返回空列表


class QuarkInt4Fp8MoEMethod(FusedMoEMethodBase):  # Quark INT4-FP8 MoE 方法
    """MoE method for INT4FP8.  # INT4FP8 MoE 方法

    Supports loading BF16/FP16 checkpoints, quantizing down to INT4, and dequantizing to FP8 during inference.  # 支持加载 BF16/FP16 检查点，量化为 INT4，并在推理时反量化为 FP8

    Args:  # 参数
        quant_config: The quantization config.  # 量化配置
    """

    def __init__(self, quant_config):  # 初始化方法
        self.quant_config = quant_config  # 保存量化配置

        self.online_quant_progress_bar = self.quant_config.online_quant_progress_bar  # 保存在线量化进度条

        self.tp_rank = get_tensor_model_parallel_rank()  # 获取当前张量并行排名

        if not _is_hip:  # 如果不是 HIP 平台
            raise NotImplementedError(  # 抛出未实现错误
                "The quark_int4fp8_moe online quantization scheme is only supported on AMD GPUs."
            )  # quark_int4fp8_moe 在线量化方案仅在 AMD GPU 上支持

    def get_weight_loader(self, layer, original_weight_loader):  # 获取在线 INT4-FP8 权重加载器
        def online_int4_fp8_weight_loader(  # 在线 INT4-FP8 权重加载函数
            param: torch.nn.Parameter,  # 目标参数
            loaded_weight: torch.Tensor,  # 加载的权重
            weight_name: str,  # 权重名称
            shard_id: str,  # 分片标识（w1/w2/w3）
            expert_id: int,  # 专家 ID
        ):
            if shard_id in ["w1", "w3"]:  # 如果是 w1 或 w3 分片
                shard_size = self.w13_shard_size  # 使用 w13 分片大小
            else:  # w2 分片
                shard_size = self.w2_shard_size  # 使用 w2 分片大小

            original_use_presharded_weights = layer.use_presharded_weights  # 保存原始预分片标志

            if not layer.use_presharded_weights:  # 如果模型未预分片
                # In case the model is not pre-sharded (most checkpoints on HF Hub),
                # we shard the model here in order to run online quantization on
                # already sharded weights.  # 如果模型未预分片（HF Hub 上大多数检查点），在此处分片以在已分片权重上运行在线量化
                # Some models as `lmzheng/grok-1` are already be sharded.  # 一些模型如 grok-1 已经分片
                layer.use_presharded_weights = True  # 设置为预分片模式

                if shard_id in ["w1", "w3"]:  # w1/w3 在第 0 维分片
                    shard_dim = 0  # 分片维度为 0
                    loaded_weight = loaded_weight.narrow(  # 截取当前 rank 对应的分片
                        shard_dim, shard_size * self.tp_rank, shard_size
                    )
                else:  # w2 在第 1 维分片
                    shard_dim = 1  # 分片维度为 1
                    loaded_weight = loaded_weight.narrow(  # 截取当前 rank 对应的分片
                        shard_dim, shard_size * self.tp_rank, shard_size
                    )

            # We want to run online quantization on-device for speed purposes.  # 我们希望在设备上运行在线量化以提高速度
            loaded_weight = loaded_weight.to(param.device)  # 将权重移至目标设备

            _, fp8_scale = quantize_fp8_scale_tensorwise(loaded_weight)  # 计算逐张量 FP8 缩放因子

            int4_w, int4_scale = quantize_int4_scale_columnwise(loaded_weight)  # 逐列量化为 INT4 并获取缩放因子

            int4_w = pack_int4_to_int32(int4_w)  # 将 INT4 权重打包为 INT32
            int4_scale /= fp8_scale  # 将 INT4 缩放因子除以 FP8 缩放因子，得到相对缩放

            if shard_id in ["w1", "w3"]:  # 如果是 w1 或 w3 分片
                if shard_id == "w1":  # w1 分片
                    shard_slice = slice(0, shard_size)  # 切片范围为前半部分
                    idx = 0  # 索引为 0
                else:  # w3 分片
                    shard_slice = slice(shard_size, 2 * shard_size)  # 切片范围为后半部分
                    idx = 1  # 索引为 1

                assert param[expert_id][shard_slice].dtype == int4_w.dtype  # 断言数据类型一致

                assert (  # 断言 INT4 缩放形状一致
                    layer.w13_int4_scale[expert_id][shard_slice].shape
                    == int4_scale.shape
                )
                assert (  # 断言 INT4 缩放数据类型一致
                    layer.w13_int4_scale[expert_id][shard_slice].dtype
                    == int4_scale.dtype
                )

                layer.w13_int4_scale[expert_id][shard_slice].copy_(int4_scale)  # 复制 INT4 缩放到层参数

                assert layer.w13_fp8_scale[expert_id][idx].shape == fp8_scale.shape  # 断言 FP8 缩放形状一致
                assert layer.w13_fp8_scale[expert_id][idx].dtype == fp8_scale.dtype  # 断言 FP8 缩放数据类型一致

                layer.w13_fp8_scale[expert_id][idx].copy_(fp8_scale)  # 复制 FP8 缩放到层参数
            else:  # w2 分片
                assert param[expert_id].dtype == int4_w.dtype  # 断言数据类型一致
                assert param[expert_id].shape == int4_w.shape  # 断言形状一致

                assert layer.w2_int4_scale[expert_id].shape == int4_scale.shape  # 断言 INT4 缩放形状一致
                assert layer.w2_int4_scale[expert_id].dtype == int4_scale.dtype  # 断言 INT4 缩放数据类型一致

                layer.w2_int4_scale[expert_id].copy_(int4_scale)  # 复制 INT4 缩放到层参数

                assert layer.w2_fp8_scale[expert_id].shape == fp8_scale.shape  # 断言 FP8 缩放形状一致
                assert layer.w2_fp8_scale[expert_id].dtype == fp8_scale.dtype  # 断言 FP8 缩放数据类型一致

                layer.w2_fp8_scale[expert_id].copy_(fp8_scale)  # 复制 FP8 缩放到层参数

            original_weight_loader(  # 调用原始权重加载器
                param,
                int4_w,  # 传入打包后的 INT4 权重
                shard_id=shard_id,  # 分片标识
                weight_name=weight_name,  # 权重名称
                expert_id=expert_id,  # 专家 ID
            )

            # Reset `use_presharded_weights` as the same layer may load several different weights.  # 重置预分片标志，因为同一层可能加载多个不同权重
            layer.use_presharded_weights = original_use_presharded_weights  # 恢复原始预分片标志

            self.online_quant_progress_bar.update(1)  # 更新在线量化进度条

        return online_int4_fp8_weight_loader  # 返回在线权重加载函数

    def create_weights(  # 创建权重参数
        self,
        layer: torch.nn.Module,  # 目标层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        # TODO: fix circular imports issues in sglang forcing us to import here instead of at
        # the top of file.  # TODO: 修复 sglang 中的循环导入问题，迫使我们在此处而非文件顶部导入
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入 MoE 权重缩放支持枚举

        # print("intermediate_size_per_partition", intermediate_size_per_partition)
        # fused moe logic already hands TP logic.  # 融合 MoE 逻辑已处理张量并行逻辑
        self.w13_shard_size = intermediate_size_per_partition  # w13 分片大小
        self.w2_shard_size = intermediate_size_per_partition  # w2 分片大小

        assert "weight_loader" in extra_weight_attrs  # 断言存在权重加载器
        original_weight_loader = extra_weight_attrs.get("weight_loader")  # 获取原始权重加载器

        online_int4fp8_weight_loader = self.get_weight_loader(  # 获取在线 INT4-FP8 权重加载器
            layer, original_weight_loader
        )
        extra_weight_attrs["weight_loader"] = online_int4fp8_weight_loader  # 替换为在线权重加载器

        params_dtype = torch.uint32  # 设置参数数据类型为 uint32（用于 INT4 打包）
        # WEIGHTS  # 权重
        # INT4 MoE weight - INT32 packed  # INT4 MoE 权重 - 以 INT32 打包
        w13_weight = torch.nn.Parameter(  # 创建 w13 权重参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                2 * intermediate_size_per_partition,  # w1 和 w3 的中间层大小之和
                hidden_size // 8,  # 隐藏层大小除以 8（INT4 打包）
                dtype=params_dtype,  # 数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        w2_weight = torch.nn.Parameter(  # 创建 w2 权重参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小
                intermediate_size_per_partition // 8,  # 中间层大小除以 8（INT4 打包）
                dtype=params_dtype,  # 数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight)  # 注册 w13 权重参数
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置权重属性

        layer.register_parameter("w2_weight", w2_weight)  # 注册 w2 权重参数
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置权重属性

        # Allocate 2 scales for w1 and w3 respectively.  # 分别为 w1 和 w3 分配 2 个缩放因子
        # They will be combined to a single scale after weight loading.  # 它们将在权重加载后合并为单个缩放因子
        w13_fp8_scale = torch.nn.Parameter(  # 创建 w13 FP8 缩放参数
            torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False  # 每个专家 2 个缩放值
        )
        w2_fp8_scale = torch.nn.Parameter(  # 创建 w2 FP8 缩放参数
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False  # 每个专家 1 个缩放值
        )
        layer.register_parameter("w13_fp8_scale", w13_fp8_scale)  # 注册 w13 FP8 缩放参数
        layer.register_parameter("w2_fp8_scale", w2_fp8_scale)  # 注册 w2 FP8 缩放参数

        if _is_hip:  # 如果是 HIP 平台
            w13_int4_scale = torch.nn.Parameter(  # 创建 w13 INT4 缩放参数
                torch.ones(  # 分配全 1 张量
                    num_experts,  # 专家数量维度
                    2 * intermediate_size_per_partition,  # w1 和 w3 的中间层大小之和
                    dtype=torch.float32,  # 数据类型为 float32
                ),
                requires_grad=False,  # 不需要梯度
            )
            w2_int4_scale = torch.nn.Parameter(  # 创建 w2 INT4 缩放参数
                torch.ones(num_experts, hidden_size, dtype=torch.float32),  # 每个专家每个输出通道一个缩放值
                requires_grad=False,  # 不需要梯度
            )
            layer.register_parameter("w13_int4_scale", w13_int4_scale)  # 注册 w13 INT4 缩放参数
            layer.register_parameter("w2_int4_scale", w2_int4_scale)  # 注册 w2 INT4 缩放参数

        extra_weight_attrs.update(  # 更新额外权重属性
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}  # 设置量化方法为逐张量
        )

        set_weight_attrs(w13_fp8_scale, extra_weight_attrs)  # 设置 w13 FP8 缩放属性
        set_weight_attrs(w2_fp8_scale, extra_weight_attrs)  # 设置 w2 FP8 缩放属性

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly  # 添加使用的量化方法（逐张量/分组/通道），以确保权重缩放正确加载
        extra_weight_attrs.update(  # 更新额外权重属性
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}  # 设置量化方法为逐通道
        )

        set_weight_attrs(w13_int4_scale, extra_weight_attrs)  # 设置 w13 INT4 缩放属性
        set_weight_attrs(w2_int4_scale, extra_weight_attrs)  # 设置 w2 INT4 缩放属性

        w13_input_scale = None  # w13 输入缩放设为 None
        layer.register_parameter("w13_input_scale", w13_input_scale)  # 注册 w13 输入缩放参数

        w2_input_scale = None  # w2 输入缩放设为 None
        layer.register_parameter("w2_input_scale", w2_input_scale)  # 注册 w2 输入缩放参数

        # Loading from the checkpoint w1, w2, w3 times the number of experts.  # 从检查点加载 w1、w2、w3 乘以专家数量
        total = self.online_quant_progress_bar.total + num_experts * 3  # 计算总进度
        tqdm_reset_no_print(self.online_quant_progress_bar, total=total)  # 重置进度条总数

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理
        if _is_hip and not ON_GFX950:  # 如果是 HIP 平台但不是 gfx950
            # CDNA3 does not support OCP FP8E4M3FN, but uses FP8E4M3FNUZ.  # CDNA3 不支持 OCP FP8E4M3FN，但使用 FP8E4M3FNUZ
            # CDNA4 supports OCP FP8E4M3FN.  # CDNA4 支持 OCP FP8E4M3FN
            layer.w13_int4_scale *= 0.5  # INT4 缩放乘以 0.5 以适配 FNUZ 格式
            layer.w2_int4_scale *= 0.5  # INT4 缩放乘以 0.5 以适配 FNUZ 格式

            layer.w13_fp8_scale *= 2.0  # FP8 缩放乘以 2.0 以补偿 INT4 缩放的调整
            layer.w2_fp8_scale *= 2.0  # FP8 缩放乘以 2.0 以补偿 INT4 缩放的调整

        # TODO: and use_aiter_moe: add after triton kernel added  # TODO: 在添加 triton 内核后增加 use_aiter_moe 检查
        # INT4-FP8 (INT4 MoE Weight, FP8 Compute)  # INT4-FP8（INT4 MoE 权重，FP8 计算）
        # Weight Permutation  # 权重重排
        layer.w13_weight = torch.nn.Parameter(  # 对 w13 权重进行重排
            shuffle_weight(layer.w13_weight.data, (16, 16)),  # 以 16x16 块为单位重排
            requires_grad=False,  # 不需要梯度
        )
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        layer.w2_weight = torch.nn.Parameter(  # 对 w2 权重进行重排
            shuffle_weight(layer.w2_weight.data, (16, 16)),  # 以 16x16 块为单位重排
            requires_grad=False,  # 不需要梯度
        )
        torch.cuda.empty_cache()  # 清空 CUDA 缓存

        # INT4-FP8 : offset INT4 w13_int4_scale to single w13_fp8_scale  # INT4-FP8：将 INT4 w13_int4_scale 偏移到单个 w13_fp8_scale
        # Fp8 moe kernel needs single fp8 w13_fp8_scale for w13 per expert.  # FP8 MoE 内核每个专家需要单个 w13_fp8_scale
        # We won't do requant each expert's fp8 weight (not direct available),  # 我们不会对每个专家的 FP8 权重重新量化（不可直接获取）
        # instead we adjust half of INT4 w13_int4_scale numbers  # 而是调整一半的 INT4 w13_int4_scale 数值
        assert layer.w13_fp8_scale is not None  # 断言 w13_fp8_scale 不为 None
        shard_size = layer.intermediate_size_per_partition  # 获取分片大小
        max_w13_scales = layer.w13_fp8_scale.max(dim=1).values  # 获取每个专家的最大 FP8 缩放值
        for expert_id in range(layer.num_experts):  # 遍历每个专家
            start = 0  # 起始索引
            max_w13_scale_fp8 = max_w13_scales[expert_id]  # 当前专家的最大 FP8 缩放值
            for shard_id in range(2):  # 遍历 w1 和 w3 两个分片
                if layer.w13_fp8_scale[expert_id][shard_id] != max_w13_scale_fp8:  # 如果当前分片缩放不等于最大值
                    int4_rescale = (  # 计算重缩放因子
                        layer.w13_fp8_scale[expert_id][shard_id] / max_w13_scale_fp8
                    )  # 重缩放因子 = 当前缩放 / 最大缩放
                    layer.w13_int4_scale[expert_id][  # 调整对应分片的 INT4 缩放
                        start : start + shard_size
                    ] *= int4_rescale  # 乘以重缩放因子
                start += shard_size  # 更新起始索引

        layer.w13_fp8_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)  # 用最大值替换 w13 FP8 缩放

        # special hack to asm_moe, which takes (weight_int4_scale * weight_scale) as post GEMM scaling
        # optimal design - shall apply per-column weight_int4_scale before GEMM, and weight_scale post
        # 针对 asm_moe 的特殊处理，将 (weight_int4_scale * weight_scale) 作为 GEMM 后缩放
        # 最优设计应在 GEMM 前应用逐列 weight_int4_scale，GEMM 后应用 weight_scale
        for expert_id in range(layer.num_experts):  # 遍历每个专家
            layer.w13_int4_scale[expert_id] *= max_w13_scales[expert_id]  # w13 INT4 缩放乘以最大 FP8 缩放
            layer.w2_int4_scale[expert_id] *= layer.w2_fp8_scale[expert_id]  # w2 INT4 缩放乘以 FP8 缩放

    def create_moe_runner(  # 创建 MoE 运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 目标层和 MoE 运行器配置
    ):
        from sglang.srt.layers.moe.utils import (  # 导入 MoE 工具函数
            get_moe_a2a_backend,  # 获取 MoE All-to-All 后端
            get_moe_runner_backend,  # 获取 MoE 运行器后端
        )

        self.moe_runner_config = moe_runner_config  # 保存 MoE 运行器配置
        moe_runner_backend = get_moe_runner_backend()  # 获取 MoE 运行器后端
        if moe_runner_backend.is_auto() and get_moe_a2a_backend().supports_aiter():  # 如果后端为自动且支持 aiter
            moe_runner_backend = MoeRunnerBackend.AITER  # 设为 AITER 后端

        if moe_runner_backend.is_aiter():  # 如果使用 AITER 后端
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)  # 创建 MoE 运行器
        else:  # 其他后端
            # TODO(cwan): refactor other backends  # TODO(cwan): 重构其他后端
            pass  # 暂不处理

    def apply(  # 应用 MoE 计算方法
        self,
        layer: torch.nn.Module,  # 目标层
        dispatch_output: "DispatchOutput",  # 分发输出
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.moe_runner.aiter import (  # 导入 Aiter MoE 量化信息类
            AiterMoeQuantInfo,  # Aiter MoE 量化信息
            AiterQuantType,  # Aiter 量化类型
        )

        moe_runner_config = self.moe_runner_config  # 获取 MoE 运行器配置

        # TODO: add triton kernel and add check get_bool_env_var("CK_MOE")  # TODO: 添加 triton 内核并添加 CK_MOE 环境变量检查
        assert (  # 断言不支持 no_combine 模式
            not moe_runner_config.no_combine
        ), f"no_combine={moe_runner_config.no_combine} is not supported."  # 不支持 no_combine 模式

        quant_info = AiterMoeQuantInfo(  # 创建 Aiter MoE 量化信息
            w13_weight=layer.w13_weight,  # w13 权重
            w2_weight=layer.w2_weight,  # w2 权重
            quant_type=AiterQuantType.PER_TOKEN,  # 量化类型为逐 token
            w13_scale=layer.w13_int4_scale,  # w13 INT4 缩放
            w2_scale=layer.w2_int4_scale,  # w2 INT4 缩放
        )
        return self.runner.run(dispatch_output, quant_info)  # 运行 MoE 并返回结果
