# 压缩张量WNA16 MoE量化方案
# 本文件实现了WNA16量化在MoE（混合专家）层的应用，
# 包括Marlin格式转换、Triton MoE实现和NPU平台INT4动态量化支持。

from __future__ import annotations  # 启用延迟类型注解评估

import enum  # 导入枚举模块
import logging  # 导入日志模块
from enum import Enum  # 导入枚举基类
from typing import TYPE_CHECKING  # 导入类型检查常量

import torch  # 导入PyTorch深度学习框架
from compressed_tensors import CompressionFormat  # 导入压缩格式枚举

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (  # 导入NPU W4A16 INT4动态MoE方法
    NPUW4A16Int4DynamicMoEMethod,  # NPU W4A16 INT4动态MoE方法类
)
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig  # 导入MoE运行器相关类
from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # 导入压缩张量方案
    WNA16_SUPPORTED_BITS,  # WNA16支持的位宽列表
    CompressedTensorsMoEScheme,  # 压缩张量MoE方案基类
)
from sglang.srt.layers.quantization.gptq import gptq_marlin_moe_repack  # 导入GPTQ Marlin MoE重打包函数
from sglang.srt.layers.quantization.marlin_utils import (  # 导入Marlin工具函数
    marlin_make_workspace,  # 创建Marlin工作空间
    marlin_moe_permute_scales,  # Marlin MoE缩放因子置换
)
from sglang.srt.layers.quantization.utils import replace_parameter  # 导入参数替换工具函数
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip, set_weight_attrs  # 导入工具函数

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入token分发器类型
        CombineInput,  # 合并输入类型
        StandardDispatchOutput,  # 标准分发输出类型
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (  # 导入压缩张量配置类型
        CompressedTensorsConfig,  # 压缩张量配置类
    )


__all__ = [  # 模块公开导出列表
    "CompressedTensorsWNA16MoE",  # Marlin格式的WNA16 MoE方案
    "CompressedTensorsWNA16TritonMoE",  # Triton格式的WNA16 MoE方案
    "NPUCompressedTensorsW4A16Int4DynamicMoE",  # NPU W4A16 INT4动态MoE方案
]

_is_hip = is_hip()  # 检测当前是否为HIP（AMD GPU）环境
_is_cuda = is_cuda()  # 检测当前是否为CUDA环境

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITER（仅HIP环境）

if _use_aiter:  # 如果使用AITER
    pass  # 暂无操作，预留扩展


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class GPTQMarlinState(Enum):  # GPTQ Marlin状态枚举
    """GPTQ Marlin权重状态枚举，用于跟踪权重的打包/就绪状态。"""
    REPACK = enum.auto()  # 需要重打包状态
    READY = enum.auto()  # 就绪状态


class CompressedTensorsWNA16MoE(CompressedTensorsMoEScheme):
    """压缩张量WNA16 MoE量化方案，使用Marlin内核进行高效推理。"""

    def __init__(self, quant_config: CompressedTensorsConfig, num_gpu_experts=-1):  # 初始化方法
        """初始化WNA16 MoE量化方案，从配置中解析量化参数。"""
        self.quant_config = quant_config  # 保存量化配置
        config = self.quant_config.target_scheme_map["Linear"].get("weights")  # 获取线性层权重配置
        self.num_bits = config.num_bits  # 量化位宽
        self.packed_factor = 32 // config.num_bits  # 打包因子（32位中可打包的量化值数量）
        self.strategy = config.strategy  # 量化策略
        self.group_size = config.group_size  # 分组大小
        self.actorder = config.actorder  # 激活排序方式
        assert config.symmetric, "Only symmetric quantization is supported for MoE"  # 断言仅支持对称量化

        if not (  # 如果不满足以下条件
            self.quant_config.quant_format == CompressionFormat.pack_quantized.value  # 量化格式为打包量化
            and self.num_bits in WNA16_SUPPORTED_BITS  # 位宽在支持列表中
        ):
            raise ValueError(  # 抛出值错误
                "For Fused MoE layers, only ",  # 融合MoE层仅支持
                f"{CompressionFormat.pack_quantized.value} ",  # 打包量化格式
                "is supported for the following bits: ",  # 用于以下位宽：
                f"{WNA16_SUPPORTED_BITS}",  # 支持的位宽列表
            )
        self.num_gpu_experts = num_gpu_experts  # 保存GPU专家数量

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低GPU算力要求
        """获取运行此量化方案所需的最低GPU计算能力版本。"""
        # ampere and up  # Ampere架构及以上
        return 80  # 返回SM 8.0（Ampere架构）

    def create_weights(  # 创建权重参数方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外的权重属性关键字参数
    ):
        """为WNA16 MoE层创建并注册所有权重参数（打包权重、缩放因子、分组索引等）。"""
        # Will transpose the loaded weight along the  # 将沿中间维度和隐藏维度转置加载的权重
        # intermediate and hidden dim sizes. Will  # 将
        # shard for TP along the transposed dims  # 沿转置维度进行张量并行分片
        extra_weight_attrs.update(  # 更新额外权重属性
            {"is_transposed": True, "quant_method": self.strategy}  # 设置转置标志和量化方法
        )
        w13_weight = torch.nn.Parameter(  # 创建w13（门控+上投影）打包权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size // self.packed_factor,  # 隐藏层大小除以打包因子
                2 * intermediate_size_per_partition,  # 2倍中间层大小（门控+上投影）
                dtype=torch.int32,  # 使用int32类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_packed", w13_weight)  # 将w13打包权重注册到层中
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置w13权重的额外属性

        w2_weight = torch.nn.Parameter(  # 创建w2（下投影）打包权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                intermediate_size_per_partition // self.packed_factor,  # 中间层大小除以打包因子
                hidden_size,  # 隐藏层大小维度
                dtype=torch.int32,  # 使用int32类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight_packed", w2_weight)  # 将w2打包权重注册到层中
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置w2权重的额外属性

        # In the case where we have actorder/g_idx,  # 当有激活排序/分组索引时，
        # we do not partition the w2 scales  # 不对w2缩放因子进行分片
        load_full_w2 = self.actorder and self.group_size != -1  # 是否加载完整w2缩放因子

        if load_full_w2:  # 如果需要加载完整w2
            w2_scales_size = intermediate_size_per_partition * layer.moe_tp_size  # w2缩放大小乘以TP大小
        else:  # 否则
            w2_scales_size = intermediate_size_per_partition  # w2缩放大小等于分区中间层大小

        self.is_k_full = (not self.actorder) or layer.moe_tp_size == 1  # K维度是否完整

        if self.strategy == "channel":  # 如果是通道级量化策略
            num_groups_w2 = num_groups_w13 = 1  # 通道级量化时组数为1
            self.group_size = -1  # 设置group_size为-1表示通道级
        else:  # 否则为分组量化
            num_groups_w2 = w2_scales_size // self.group_size  # w2的组数
            num_groups_w13 = hidden_size // self.group_size  # w13的组数

        w13_scale = torch.nn.Parameter(  # 创建w13权重缩放因子参数
            torch.ones(  # 创建全1张量
                num_experts,  # 专家数量维度
                num_groups_w13,  # w13组数
                2 * intermediate_size_per_partition,  # 2倍中间层大小
                dtype=params_dtype,  # 参数数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_scale", w13_scale)  # 将w13缩放因子注册到层中
        set_weight_attrs(w13_scale, extra_weight_attrs)  # 设置w13缩放因子的额外属性

        w2_scale = torch.nn.Parameter(  # 创建w2权重缩放因子参数
            torch.ones(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),  # 全1张量
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight_scale", w2_scale)  # 将w2缩放因子注册到层中
        set_weight_attrs(w2_scale, extra_weight_attrs)  # 设置w2缩放因子的额外属性
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})  # 设置w2是否加载完整

        w2_weight_shape = torch.nn.Parameter(  # 创建w2权重形状参数
            torch.empty(num_experts, 2), requires_grad=False  # 空张量，2个元素
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)  # 将w2权重形状注册到层中
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)  # 设置w2权重形状的额外属性
        w13_weight_shape = torch.nn.Parameter(  # 创建w13权重形状参数
            torch.empty(num_experts, 2), requires_grad=False  # 空张量，2个元素
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)  # 将w13权重形状注册到层中
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)  # 设置w13权重形状的额外属性

        w13_g_idx = torch.nn.Parameter(  # 创建w13分组索引参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小维度
                dtype=torch.int32,  # 使用int32类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)  # 将w13分组索引注册到层中
        set_weight_attrs(w13_g_idx, extra_weight_attrs)  # 设置w13分组索引的额外属性

        w2_g_idx = torch.nn.Parameter(  # 创建w2分组索引参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                intermediate_size_per_partition,  # 中间层大小维度
                dtype=torch.int32,  # 使用int32类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)  # 将w2分组索引注册到层中
        set_weight_attrs(w2_g_idx, extra_weight_attrs)  # 设置w2分组索引的额外属性

        w13_g_idx_sort_indices = torch.nn.Parameter(  # 创建w13分组索引排序结果参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小维度
                dtype=torch.int32,  # 使用int32类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)  # 将w13排序索引注册到层中
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)  # 设置w13排序索引的额外属性

        w2_g_idx_sort_indices = torch.nn.Parameter(  # 创建w2分组索引排序结果参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                intermediate_size_per_partition,  # 中间层大小维度
                dtype=torch.int32,  # 使用int32类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)  # 将w2排序索引注册到层中
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)  # 设置w2排序索引的额外属性

        layer.a13_scale = None  # 激活缩放因子设为None
        layer.a2_scale = None  # 激活缩放因子设为None
        layer.marlin_state = GPTQMarlinState.REPACK  # 设置Marlin状态为需要重打包

        if not hasattr(layer, "_original_shapes"):  # 如果层没有原始形状记录
            layer._original_shapes = {}  # 创建原始形状字典

        # Force record: these are the target GPTQ shapes for rollback.  # 强制记录：这些是回滚时的目标GPTQ形状
        layer._original_shapes["w13_weight_packed"] = tuple(w13_weight.shape)  # 记录w13打包权重的原始形状
        layer._original_shapes["w2_weight_packed"] = tuple(w2_weight.shape)  # 记录w2打包权重的原始形状

        # Also record the shapes of the scales.  # 同时记录缩放因子的形状
        layer._original_shapes["w2_weight_scale"] = tuple(w2_scale.shape)  # 记录w2缩放因子的原始形状
        layer._original_shapes["w13_weight_scale"] = tuple(w13_scale.shape)  # 记录w13缩放因子的原始形状

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的后处理方法
        """权重加载后处理：将权重重打包为Marlin格式，处理分组索引和缩放因子。"""

        # Skip if the layer is already converted to Marlin format to prevent double-packing.  # 如果层已经转换为Marlin格式则跳过，防止重复打包
        if getattr(layer, "is_marlin_converted", False):  # 检查是否已转换
            return  # 已转换则直接返回

        if not hasattr(layer, "_original_shapes"):  # 如果层没有原始形状记录
            layer._original_shapes = {}  # 创建原始形状字典

        def replace_tensor(name, new_t):  # 定义替换张量的内部函数
            """内部函数：用新张量替换层中的指定参数张量，保留原始形状记录。"""
            target_attr = getattr(layer, name)  # 获取目标属性

            # Only save if the key doesn't exist to prevent overwriting with Marlin shapes.  # 仅在键不存在时保存，防止用Marlin形状覆盖
            if name not in layer._original_shapes:  # 如果名称不在原始形状字典中
                # This is a safety check; `create_weights` usually handles this already.  # 安全检查；`create_weights`通常已经处理了
                layer._original_shapes[name] = tuple(target_attr.shape)  # 记录原始形状

            # It is important to use resize_() here since it ensures  # 在此使用resize_()很重要，因为它确保
            # the same buffer is reused  # 复用相同的缓冲区
            target_attr.resize_(new_t.shape)  # 调整目标属性大小为新张量的形状
            target_attr.copy_(new_t)  # 将新张量数据复制到目标属性
            del new_t  # 删除新张量引用以释放内存

        num_experts = layer.w13_weight_g_idx.shape[0]  # 获取专家数量
        device = layer.w13_weight_g_idx.device  # 获取设备类型

        # when running models with grouped act order,  # 当运行带有分组激活排序的模型时，
        # resort to g_idx values provided in checkpoint  # 使用检查点中提供的分组索引值
        if self.actorder == "group":  # 如果激活排序为分组模式
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)  # 创建w13排序索引空张量
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)  # 创建w2排序索引空张量
            w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)  # 创建w13排序后分组索引空张量
            w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)  # 创建w2排序后分组索引空张量

            for e in range(num_experts):  # 遍历每个专家
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_weight_g_idx[e]).to(  # 对w13分组索引排序并转为
                    torch.int32  # int32类型
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_weight_g_idx[e]).to(  # 对w2分组索引排序并转为
                    torch.int32  # int32类型
                )
                w13_sorted_g_idx[e] = layer.w13_weight_g_idx[e][  # 按排序索引重排w13分组索引
                    w13_g_idx_sort_indices[e]  # 使用w13排序索引
                ]
                w2_sorted_g_idx[e] = layer.w2_weight_g_idx[e][w2_g_idx_sort_indices[e]]  # 按排序索引重排w2分组索引

            replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)  # 替换w13分组索引为排序后的版本
            replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)  # 替换w2分组索引为排序后的版本
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)  # 替换w13排序索引
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)  # 替换w2排序索引

        else:  # 如果不是分组激活排序
            layer.w13_weight_g_idx = torch.nn.Parameter(  # 设置w13分组索引为空
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),  # 空张量
                requires_grad=False,  # 不需要梯度
            )
            layer.w2_weight_g_idx = torch.nn.Parameter(  # 设置w2分组索引为空
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),  # 空张量
                requires_grad=False,  # 不需要梯度
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(  # 设置w13排序索引为空
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),  # 空张量
                requires_grad=False,  # 不需要梯度
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(  # 设置w2排序索引为空
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),  # 空张量
                requires_grad=False,  # 不需要梯度
            )

        marlin_w13_qweight = gptq_marlin_moe_repack(  # 将w13权重重打包为Marlin格式
            layer.w13_weight_packed,  # w13打包权重
            layer.w13_g_idx_sort_indices,  # w13排序索引
            layer.w13_weight_packed.shape[1] * self.packed_factor,  # 原始K维度大小
            layer.w13_weight_packed.shape[2],  # N维度大小
            self.num_bits,  # 量化位宽
        )
        replace_tensor("w13_weight_packed", marlin_w13_qweight)  # 替换w13打包权重
        marlin_w2_qweight = gptq_marlin_moe_repack(  # 将w2权重重打包为Marlin格式
            layer.w2_weight_packed,  # w2打包权重
            layer.w2_g_idx_sort_indices,  # w2排序索引
            layer.w2_weight_packed.shape[1] * self.packed_factor,  # 原始K维度大小
            layer.w2_weight_packed.shape[2],  # N维度大小
            self.num_bits,  # 量化位宽
        )
        replace_tensor("w2_weight_packed", marlin_w2_qweight)  # 替换w2打包权重
        # Repack scales  # 重打包缩放因子
        marlin_w13_scales = marlin_moe_permute_scales(  # 将w13缩放因子置换为Marlin格式
            layer.w13_weight_scale,  # w13缩放因子
            layer.w13_weight_packed.shape[2],  # N维度大小
            layer.w13_weight_scale.shape[2],  # 缩放因子的最后一个维度
            self.group_size,  # 分组大小
        )
        replace_tensor("w13_weight_scale", marlin_w13_scales)  # 替换w13缩放因子

        marlin_w2_scales = marlin_moe_permute_scales(  # 将w2缩放因子置换为Marlin格式
            layer.w2_weight_scale,  # w2缩放因子
            layer.w2_weight_scale.shape[1]  # 缩放因子第二维度
            * (self.group_size if self.group_size != -1 else self.packed_factor),  # 乘以分组大小或打包因子
            layer.w2_weight_scale.shape[2],  # 缩放因子第三维度
            self.group_size,  # 分组大小
        )
        replace_tensor("w2_weight_scale", marlin_w2_scales)  # 替换w2缩放因子

        layer.workspace = marlin_make_workspace(layer.w13_weight_packed.device, 4)  # 创建Marlin工作空间，4字节对齐
        layer.is_marlin_converted = True  # 标记层已转换为Marlin格式

    def restore_weights_before_loading(self, layer: torch.nn.Module):  # 恢复权重到加载前形状的方法
        """Forcibly resize parameters back to their original shapes (e.g., GPTQ format) before loading weights."""  # 强制将参数调整回原始形状（如GPTQ格式）以便加载权重

        if not hasattr(layer, "_original_shapes"):  # 如果层没有原始形状记录
            return  # 直接返回

        for name, orig_shape in layer._original_shapes.items():  # 遍历所有原始形状
            param = getattr(layer, name, None)  # 获取参数

            if param is not None and param.shape != orig_shape:  # 如果参数存在且形状不同
                param.resize_(orig_shape)  # 调整参数大小为原始形状

        layer.is_marlin_converted = False  # 重置Marlin转换标志

    def create_moe_runner(  # 创建MoE运行器方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 目标层和MoE运行器配置
    ):
        """创建Marlin后端的MoE运行器。"""
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置
        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)  # 创建Marlin后端的MoE运行器

    def get_marlin_quant_info(self, layer):  # 获取Marlin量化信息方法
        """从层中提取Marlin量化信息，用于MoE运行器执行推理。"""
        from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo  # 导入Marlin MoE量化信息类

        return MarlinMoeQuantInfo(  # 返回Marlin MoE量化信息对象
            w13_qweight=layer.w13_weight_packed,  # w13打包量化权重
            w2_qweight=layer.w2_weight_packed,  # w2打包量化权重
            w13_scales=layer.w13_weight_scale,  # w13缩放因子
            w2_scales=layer.w2_weight_scale,  # w2缩放因子
            w13_g_idx_sort_indices=getattr(layer, "w13_g_idx_sort_indices", None),  # w13分组索引排序结果
            w2_g_idx_sort_indices=getattr(layer, "w2_g_idx_sort_indices", None),  # w2分组索引排序结果
            weight_bits=self.num_bits,  # 权重量化位宽
            w13_g_idx=getattr(layer, "w13_weight_g_idx", None),  # w13分组索引
            w2_g_idx=getattr(layer, "w2_weight_g_idx", None),  # w2分组索引
            is_k_full=self.is_k_full,  # K维度是否完整
        )

    def apply_weights(  # 应用权重进行前向计算方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:  # 返回合并输入
        """应用WNA16量化权重执行MoE前向计算（使用Marlin内核）。"""
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (  # 导入融合Marlin MoE函数
            fused_marlin_moe,  # 融合Marlin MoE函数
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入类

        assert (  # 断言
            self.moe_runner_config.activation == "silu"  # 仅支持SiLU激活函数
        ), "Only SiLU activation is supported."  # 仅支持SiLU激活函数

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取topk输出

        topk_weights, topk_ids, router_logits = topk_output  # 解包topk输出

        # Get expert_map for EP support  # 获取专家映射以支持专家并行
        expert_map = None  # 初始化专家映射为None
        global_num_experts = -1  # 初始化全局专家数量为-1
        if hasattr(layer, "dispatcher") and hasattr(  # 如果层有分发器且分发器有
            layer.dispatcher, "local_expert_mapping"  # 本地专家映射
        ):
            expert_map = layer.dispatcher.local_expert_mapping  # 获取本地专家映射
            if expert_map is not None:  # 如果专家映射不为空
                global_num_experts = self.moe_runner_config.num_experts  # 获取全局专家数量

        output = fused_marlin_moe(  # 调用融合Marlin MoE函数
            x,  # 隐藏状态
            layer.w13_weight_packed,  # w13打包权重
            layer.w2_weight_packed,  # w2打包权重
            layer.w13_weight_scale,  # w13缩放因子
            layer.w2_weight_scale,  # w2缩放因子
            router_logits,  # 路由器logits
            topk_weights,  # topk权重
            topk_ids,  # topk ID
            global_num_experts=global_num_experts,  # 全局专家数量
            expert_map=expert_map,  # 专家映射
            g_idx1=layer.w13_weight_g_idx,  # w13分组索引
            g_idx2=layer.w2_weight_g_idx,  # w2分组索引
            sort_indices1=layer.w13_g_idx_sort_indices,  # w13排序索引
            sort_indices2=layer.w2_g_idx_sort_indices,  # w2排序索引
            num_bits=self.num_bits,  # 量化位宽
            is_k_full=self.is_k_full,  # K维度是否完整
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,  # 路由缩放因子
            workspace=layer.workspace,  # Marlin工作空间
        )
        return StandardCombineInput(hidden_states=output)  # 返回标准合并输入


class CompressedTensorsWNA16TritonMoE(CompressedTensorsWNA16MoE):
    """ROCm/HIP-compatible W4A16 MoE method using Triton kernels instead of Marlin.  # ROCm/HIP兼容的W4A16 MoE方法，使用Triton内核而非Marlin

    Inherits weight creation from CompressedTensorsWNA16MoE but converts  # 继承CompressedTensorsWNA16MoE的权重创建，但将
    weights to the uint8-packed format expected by the Triton fused MoE kernel  # 权重转换为Triton融合MoE内核所需的uint8打包格式
    instead of the Marlin-specific format.  # 而非Marlin专用格式。
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的后处理方法
        """权重加载后处理：将权重转换为Triton内核所需的uint8打包格式。"""
        if getattr(layer, "is_triton_converted", False):  # 如果已转换为Triton格式
            return  # 直接返回

        num_experts = layer.w13_weight_packed.shape[0]  # 获取专家数量

        # Convert w13 weights: [E, K//8, N] int32 -> [E, N, K//2] uint8  # 转换w13权重：[E, K//8, N] int32 -> [E, N, K//2] uint8
        w13 = layer.w13_weight_packed.data  # 获取w13权重数据
        w13 = w13.transpose(1, 2).contiguous().view(torch.uint8)  # 转置并转为uint8视图
        layer.w13_weight_packed = torch.nn.Parameter(w13, requires_grad=False)  # 替换为uint8参数

        # Convert w2 weights: [E, K//8, N] int32 -> [E, N, K//2] uint8  # 转换w2权重：[E, K//8, N] int32 -> [E, N, K//2] uint8
        w2 = layer.w2_weight_packed.data  # 获取w2权重数据
        w2 = w2.transpose(1, 2).contiguous().view(torch.uint8)  # 转置并转为uint8视图
        layer.w2_weight_packed = torch.nn.Parameter(w2, requires_grad=False)  # 替换为uint8参数

        # Convert w13 scales: [E, K//group_size, N] -> [E, N, K//group_size]  # 转换w13缩放因子：[E, K//group_size, N] -> [E, N, K//group_size]
        w13_scale = layer.w13_weight_scale.data  # 获取w13缩放因子数据
        w13_scale = w13_scale.transpose(1, 2).contiguous()  # 转置缩放因子
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)  # 替换缩放因子参数

        # Convert w2 scales: [E, K//group_size, N] -> [E, N, K//group_size]  # 转换w2缩放因子：[E, K//group_size, N] -> [E, N, K//group_size]
        w2_scale = layer.w2_weight_scale.data  # 获取w2缩放因子数据
        w2_scale = w2_scale.transpose(1, 2).contiguous()  # 转置缩放因子
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)  # 替换缩放因子参数

        layer.is_triton_converted = True  # 标记层已转换为Triton格式

    def create_moe_runner(  # 创建MoE运行器方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 目标层和MoE运行器配置
    ):
        """创建Triton后端的MoE运行器。"""
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)  # 创建Triton后端的MoE运行器

    def get_triton_quant_info(self, layer):  # 获取Triton量化信息方法
        """从层中提取Triton量化信息，用于MoE运行器执行推理。"""
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo  # 导入Triton MoE量化信息类

        return TritonMoeQuantInfo(  # 返回Triton MoE量化信息对象
            w13_weight=layer.w13_weight_packed,  # w13打包权重
            w2_weight=layer.w2_weight_packed,  # w2打包权重
            use_int4_w4a16=True,  # 使用INT4 W4A16模式
            w13_scale=layer.w13_weight_scale,  # w13缩放因子
            w2_scale=layer.w2_weight_scale,  # w2缩放因子
            block_shape=[0, self.group_size],  # 块形状（0表示不限制行，group_size限制列）
        )

    def apply_weights(  # 应用权重进行前向计算方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        dispatch_output: "StandardDispatchOutput",  # 标准分发输出
    ) -> "CombineInput":  # 返回合并输入
        """应用WNA16量化权重执行MoE前向计算（使用Triton内核）。"""
        assert (  # 断言
            self.moe_runner_config.activation == "silu"  # 仅支持SiLU激活函数
        ), "Only SiLU activation is supported."  # 仅支持SiLU激活函数

        quant_info = self.get_triton_quant_info(layer)  # 获取Triton量化信息
        return self.runner.run(dispatch_output, quant_info)  # 调用Triton运行器执行推理


class NPUCompressedTensorsW4A16Int4DynamicMoE(CompressedTensorsMoEScheme):
    """NPU平台上的W4A16 INT4动态量化MoE方案。"""

    def __init__(self, quantization_config) -> None:  # 初始化方法
        """初始化NPU W4A16 INT4动态量化MoE方案，从配置中解析量化参数。"""
        self.pack_factor = 8  # weight dtype is int4,  but use int32 to create  # 权重数据类型为int4，但使用int32创建（32/4=8）
        target = (  # 确定目标层类型
            "MoEGMM" if "MoEGMM" in quantization_config.target_scheme_map else "Linear"  # 优先使用MoEGMM，否则使用Linear
        )
        if target in quantization_config.target_scheme_map:  # 如果目标类型在配置方案映射中
            self.group_size = quantization_config.target_scheme_map[target][  # 从配置中获取分组大小
                "weights"  # 权重配置
            ].group_size  # 获取分组大小
        else:  # 否则
            self.group_size = 128  # 默认分组大小为128

        self.kernel = NPUW4A16Int4DynamicMoEMethod()  # 创建NPU W4A16 INT4动态MoE内核实例

    # TODO: See if we can merge this method's logic  # TODO: 看看是否可以合并此方法的逻辑
    # with CompressedTensorsWNA16MoE. Need more models and tests.  # 与CompressedTensorsWNA16MoE。需要更多模型和测试。
    # @OrangeRedeng @TamirBaydasov  # 相关开发者
    def create_weights(  # 创建权重参数方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外的权重属性关键字参数
    ) -> None:
        """为NPU W4A16 INT4 MoE层创建并注册所有权重参数（权重、缩放因子、偏移等）。"""
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入融合MoE权重缩放支持枚举

        self.num_experts = num_experts  # 保存专家数量
        if (  # 判断量化方法
            extra_weight_attrs.get(  # 获取中间层大小
                "moe_intermediate_size", intermediate_size_per_partition  # 如果没有则使用分区中间层大小
            )
            // intermediate_size_per_partition  # 除以分区中间层大小
            > 1  # 如果大于1（表示有TP分片）
        ):
            quant_method = FusedMoeWeightScaleSupported.GROUP.value  # 使用分组量化方法
        else:  # 否则
            quant_method = FusedMoeWeightScaleSupported.CHANNEL.value  # 使用通道级量化方法
        extra_weight_attrs.update({"quant_method": quant_method})  # 更新量化方法属性
        # weight  # 权重
        w13_weight = torch.nn.Parameter(  # 创建w13权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                2 * intermediate_size_per_partition,  # 2倍中间层大小
                hidden_size // self.pack_factor,  # 隐藏层大小除以打包因子
                dtype=torch.int32,  # 使用int32类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight)  # 将w13权重注册到层中
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置w13权重的额外属性
        w2_weight = torch.nn.Parameter(  # 创建w2权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小维度
                intermediate_size_per_partition // self.pack_factor,  # 中间层大小除以打包因子
                dtype=torch.int32,  # 使用int32类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight", w2_weight)  # 将w2权重注册到层中
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置w2权重的额外属性

        # scale  # 缩放因子
        weight_scale_dtype = torch.bfloat16  # 权重缩放因子使用bfloat16类型
        w13_weight_scale = torch.nn.Parameter(  # 创建w13权重缩放因子参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                2 * intermediate_size_per_partition,  # 2倍中间层大小
                hidden_size // self.group_size,  # 隐藏层大小除以分组大小
                dtype=weight_scale_dtype,  # 使用bfloat16类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)  # 将w13缩放因子注册到层中
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)  # 设置w13缩放因子的额外属性
        w2_weight_scale = torch.nn.Parameter(  # 创建w2权重缩放因子参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小维度
                intermediate_size_per_partition // self.group_size,  # 中间层大小除以分组大小
                dtype=weight_scale_dtype,  # 使用bfloat16类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)  # 将w2缩放因子注册到层中
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)  # 设置w2缩放因子的额外属性

        # offset  # 偏移量
        w13_weight_offset = torch.nn.Parameter(  # 创建w13权重偏移量参数
            torch.zeros(  # 创建全零张量
                num_experts,  # 专家数量维度
                2 * intermediate_size_per_partition,  # 2倍中间层大小
                hidden_size // self.group_size,  # 隐藏层大小除以分组大小
                dtype=weight_scale_dtype,  # 使用bfloat16类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)  # 将w13偏移量注册到层中
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)  # 设置w13偏移量的额外属性

        w2_weight_offset = torch.nn.Parameter(  # 创建w2权重偏移量参数
            torch.zeros(  # 创建全零张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小维度
                intermediate_size_per_partition // self.group_size,  # 中间层大小除以分组大小
                dtype=weight_scale_dtype,  # 使用bfloat16类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)  # 将w2偏移量注册到层中
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)  # 设置w2偏移量的额外属性

        w13_weight_shape = torch.nn.Parameter(  # 创建w13权重形状参数
            torch.empty(num_experts, 2), requires_grad=False  # 空张量，2个元素
        )
        layer.register_parameter("w13_weight_shape", w13_weight_shape)  # 将w13权重形状注册到层中
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)  # 设置w13权重形状的额外属性

        w2_weight_shape = torch.nn.Parameter(  # 创建w2权重形状参数
            torch.empty(num_experts, 2), requires_grad=False  # 空张量，2个元素
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)  # 将w2权重形状注册到层中
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)  # 设置w2权重形状的额外属性

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的后处理方法
        """权重加载后的后处理，委托给NPU内核实现。"""
        self.kernel.process_weights_after_loading(layer)  # 调用NPU内核的权重后处理方法

    def create_moe_runner(  # 创建MoE运行器方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 目标层和MoE运行器配置
    ):
        """创建MoE运行器，保存运行器配置。"""
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置

    def apply_weights(  # 应用权重进行前向计算方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:  # 返回合并输入
        """应用量化权重进行MoE前向计算，委托给NPU内核实现。"""

        return self.kernel.apply(layer, dispatch_output)  # 调用NPU内核的apply方法执行前向计算

    def apply_without_routing_weights(  # 不带路由的应用权重方法
        self,
        layer,  # 目标层
        hidden_states,  # 隐藏状态
        hidden_states_scale,  # 隐藏状态缩放因子
        group_list_type,  # 分组列表类型
        group_list,  # 分组列表
        output_dtype,  # 输出数据类型
    ):
        """不带路由权重的MoE前向计算方法，委托给NPU内核实现。"""
        return self.kernel.apply_without_routing_weights(  # 调用NPU内核的不带路由权重方法
            layer,  # 目标层
            hidden_states,  # 隐藏状态
            hidden_states_scale,  # 隐藏状态缩放因子
            group_list_type,  # 分组列表类型
            group_list,  # 分组列表
            output_dtype,  # 输出数据类型
        )
