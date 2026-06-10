# 压缩张量W4A8 INT8 MoE量化方案实现
# 本文件实现了CompressedTensors框架下W4A8（权重4比特、激活8比特）INT8量化的MoE（混合专家）层方案
# 主要用于NPU（神经网络处理器）上的MoE模型量化推理

from __future__ import annotations # 启用延迟注解评估

import logging # 导入日志模块
from typing import TYPE_CHECKING # 导入类型检查常量

import torch # 导入PyTorch库

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import ( # 从NPU量化模块导入W4A8 INT8动态MoE方法
    NPUW4A8Int8DynamicMoEMethod,
)
from sglang.srt.layers.moe import MoeRunnerConfig # 导入MoE运行器配置类
from sglang.srt.layers.quantization.compressed_tensors.schemes import ( # 导入压缩张量MoE方案基类
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs # 导入权重属性设置工具函数

if TYPE_CHECKING: # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import ( # 从token分发器模块导入类型
        CombineInput, # 合并输入类型
        StandardDispatchOutput, # 标准分发输出类型
    )

__all__ = ["NPUCompressedTensorsW4A8Int8DynamicMoE"] # 模块公开接口列表


logger = logging.getLogger(__name__) # 获取当前模块的日志记录器


class NPUCompressedTensorsW4A8Int8DynamicMoE(CompressedTensorsMoEScheme):
    """NPU平台上的压缩张量W4A8 INT8动态量化MoE方案类"""

    ### TODO: Get rid of code duplication with python/sglang/srt/modelslim/modelslim_moe.py @OrangeRedeng @TamirBaydasov
    ### 待办：消除与 python/sglang/srt/modelslim/modelslim_moe.py 的代码重复 @OrangeRedeng @TamirBaydasov
    def __init__(self, quantization_config) -> None: # 初始化方法，接收量化配置
        self.group_size = 0 # 分组大小，0表示不分组（逐通道量化）
        self.is_per_channel_weight = self.group_size == 0 # 是否为逐通道权重量化
        self.tp_size = 1 # 张量并行大小
        self.activation_use_clip = ( # 是否使用激活裁剪
            quantization_config.get("config_groups", {}) # 从配置中获取config_groups
            .get("group_1", {}) # 获取group_1配置
            .get("activation_use_clip", False) # 获取activation_use_clip标志，默认为False
        )
        self.kernel = NPUW4A8Int8DynamicMoEMethod() # 初始化NPU W4A8 INT8动态MoE计算内核

    def create_weights( # 创建权重参数方法
        self,
        layer: torch.nn.Module, # 目标神经网络层
        num_experts: int, # 专家数量
        hidden_size: int, # 隐藏层大小
        intermediate_size_per_partition: int, # 每个分区的中间层大小
        params_dtype: torch.dtype, # 参数数据类型
        **extra_weight_attrs, # 额外的权重属性关键字参数
    ) -> None:
        """创建并注册MoE层所需的权重参数，包括权重、缩放因子和偏移量"""
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported # 导入融合MoE权重缩放支持枚举

        self.num_experts = num_experts # 保存专家数量
        extra_weight_attrs.update( # 更新额外权重属性，设置量化方法为逐通道
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        # >> weight # >> 权重
        w13_output_size = intermediate_size_per_partition # w13（gate+up）的输出大小
        w2_output_size = hidden_size // 2 # w2（down）的输出大小
        w13_weight = torch.nn.Parameter( # 创建w13权重参数
            torch.empty(num_experts, w13_output_size, hidden_size, dtype=torch.int8), # 形状为[专家数, 输出大小, 隐藏大小]，int8类型
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight) # 在层中注册w13权重参数
        set_weight_attrs(w13_weight, extra_weight_attrs) # 设置w13权重的额外属性
        w2_weight = torch.nn.Parameter( # 创建w2权重参数
            torch.empty( # 创建空张量
                num_experts, # 专家数量维度
                w2_output_size, # 输出大小维度
                intermediate_size_per_partition, # 中间层大小维度
                dtype=torch.int8, # int8数据类型
            ),
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w2_weight", w2_weight) # 在层中注册w2权重参数
        set_weight_attrs(w2_weight, extra_weight_attrs) # 设置w2权重的额外属性

        # >> scale # >> 缩放因子
        weight_scale_dtype = torch.int64 if self.activation_use_clip else torch.float32 # 使用激活裁剪时缩放因子为int64，否则为float32
        w13_weight_scale = torch.nn.Parameter( # 创建w13权重缩放因子参数
            torch.empty( # 创建空张量
                num_experts, # 专家数量维度
                2 * intermediate_size_per_partition, # 2倍中间层大小（gate和up）
                1, # 最后一维为1（广播）
                dtype=weight_scale_dtype, # 缩放因子数据类型
            ),
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale) # 在层中注册w13权重缩放因子
        set_weight_attrs(w13_weight_scale, extra_weight_attrs) # 设置w13缩放因子的额外属性

        w2_weight_scale = torch.nn.Parameter( # 创建w2权重缩放因子参数
            torch.empty(num_experts, hidden_size, 1, dtype=weight_scale_dtype), # 形状为[专家数, 隐藏大小, 1]
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale) # 在层中注册w2权重缩放因子
        set_weight_attrs(w2_weight_scale, extra_weight_attrs) # 设置w2缩放因子的额外属性

        # >> offset # >> 偏移量
        w13_weight_offset = torch.nn.Parameter( # 创建w13权重偏移量参数
            torch.empty( # 创建空张量
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32 # 形状为[专家数, 2倍中间层大小, 1]，float32类型
            ),
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset) # 在层中注册w13权重偏移量
        set_weight_attrs(w13_weight_offset, extra_weight_attrs) # 设置w13偏移量的额外属性

        w2_weight_offset = torch.nn.Parameter( # 创建w2权重偏移量参数
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32), # 形状为[专家数, 隐藏大小, 1]，float32类型
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset) # 在层中注册w2权重偏移量
        set_weight_attrs(w2_weight_offset, extra_weight_attrs) # 设置w2偏移量的额外属性

        # >>> special param for w4a8 # >>> W4A8专用参数
        if self.activation_use_clip: # 如果使用激活裁剪
            self._init_activation_clip_params( # 初始化激活裁剪参数
                layer, # 目标层
                num_experts, # 专家数量
                hidden_size, # 隐藏层大小
                intermediate_size_per_partition, # 每个分区的中间层大小
                extra_weight_attrs, # 额外权重属性
            )
        else: # 否则（不使用激活裁剪）
            self._init_extra_scale_params( # 初始化额外缩放参数
                layer, # 目标层
                num_experts, # 专家数量
                hidden_size, # 隐藏层大小
                intermediate_size_per_partition, # 每个分区的中间层大小
                extra_weight_attrs, # 额外权重属性
            )

    def _init_activation_clip_params( # 初始化激活裁剪参数的私有方法
        self,
        layer: torch.nn.Module, # 目标神经网络层
        num_experts: int, # 专家数量
        hidden_size: int, # 隐藏层大小
        intermediate_size_per_partition: int, # 每个分区的中间层大小
        extra_weight_attrs: dict, # 额外权重属性字典
    ) -> None:
        """
        Initializes bias and alpha parameters for quantization schemes that use activation clipping.
        初始化使用激活裁剪的量化方案所需的偏置和alpha参数。

        This helper registers `w13_bias`, `w2_bias`, and `w2_alpha`, which are required to
        此辅助方法注册 `w13_bias`、`w2_bias` 和 `w2_alpha`，这些参数用于

        shift and scale the activations or outputs to compensate for the precision loss
        对激活或输出进行偏移和缩放，以补偿

        introduced by clamping activations.
        由激活裁剪引入的精度损失。
        """
        w13_bias = torch.nn.Parameter( # 创建w13偏置参数
            torch.ones( # 创建全1张量
                num_experts, 2 * intermediate_size_per_partition, dtype=torch.float # 形状为[专家数, 2倍中间层大小]，float类型
            ),
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w13_bias", w13_bias) # 在层中注册w13偏置参数
        set_weight_attrs(w13_bias, extra_weight_attrs) # 设置w13偏置的额外属性

        w2_bias = torch.nn.Parameter( # 创建w2偏置参数
            torch.ones(num_experts, hidden_size, dtype=torch.float), # 形状为[专家数, 隐藏大小]，float类型
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w2_bias", w2_bias) # 在层中注册w2偏置参数
        set_weight_attrs(w2_bias, extra_weight_attrs) # 设置w2偏置的额外属性

        w2_alpha = torch.nn.Parameter( # 创建w2 alpha参数
            torch.ones(num_experts, dtype=torch.float), requires_grad=False # 形状为[专家数]，float类型，不需要梯度
        )
        layer.register_parameter("w2_alpha", w2_alpha) # 在层中注册w2 alpha参数
        set_weight_attrs(w2_alpha, extra_weight_attrs) # 设置w2 alpha的额外属性

    def _init_extra_scale_params( # 初始化额外缩放参数的私有方法
        self,
        layer: torch.nn.Module, # 目标神经网络层
        num_experts: int, # 专家数量
        hidden_size: int, # 隐藏层大小
        intermediate_size_per_partition: int, # 每个分区的中间层大小
        extra_weight_attrs: dict, # 额外权重属性字典
    ) -> None:
        """
        Initializes additional scaling, offset, and bias parameters for quantization schemes without activation clipping.
        初始化不使用激活裁剪的量化方案所需的额外缩放、偏移和偏置参数。

        This method registers the following parameters:
        此方法注册以下参数：
        1. Scale Biases: `w13_scale_bias` and `w2_scale_bias`.
        1. 缩放偏置：`w13_scale_bias` 和 `w2_scale_bias`。
        2. Secondary Quantization Params (initialized only for grouped quantization):
        2. 二级量化参数（仅在分组量化时初始化）：
            `w13_weight_scale_second`, `w13_weight_offset_second`,
            `w13_weight_scale_second`、`w13_weight_offset_second`、
            `w2_weight_scale_second`, and `w2_weight_offset_second`.
            `w2_weight_scale_second` 和 `w2_weight_offset_second`。
        """
        if not self.is_per_channel_weight: # 如果不是逐通道权重（即分组量化）
            w13_weight_scale_second = torch.nn.Parameter( # 创建w13二级权重缩放因子参数
                torch.empty( # 创建空张量
                    num_experts, # 专家数量维度
                    2 * intermediate_size_per_partition, # 2倍中间层大小维度
                    hidden_size // self.group_size, # 按分组大小划分的维度
                    dtype=torch.float32, # float32数据类型
                ),
                requires_grad=False, # 不需要梯度
            )
            layer.register_parameter("w13_weight_scale_second", w13_weight_scale_second) # 在层中注册w13二级缩放因子
            set_weight_attrs(w13_weight_scale_second, extra_weight_attrs) # 设置w13二级缩放因子的额外属性

            w13_weight_offset_second = torch.nn.Parameter( # 创建w13二级权重偏移量参数
                torch.empty( # 创建空张量
                    num_experts, # 专家数量维度
                    2 * intermediate_size_per_partition, # 2倍中间层大小维度
                    hidden_size // self.group_size, # 按分组大小划分的维度
                    dtype=torch.float32, # float32数据类型
                ),
                requires_grad=False, # 不需要梯度
            )
            layer.register_parameter( # 在层中注册w13二级偏移量
                "w13_weight_offset_second", w13_weight_offset_second # 参数名和参数对象
            )
            set_weight_attrs(w13_weight_offset_second, extra_weight_attrs) # 设置w13二级偏移量的额外属性

            w2_weight_scale_second = torch.nn.Parameter( # 创建w2二级权重缩放因子参数
                torch.empty( # 创建空张量
                    num_experts, # 专家数量维度
                    hidden_size, # 隐藏大小维度
                    intermediate_size_per_partition // self.group_size, # 按分组大小划分的中间层维度
                    dtype=torch.float32, # float32数据类型
                ),
                requires_grad=False, # 不需要梯度
            )
            layer.register_parameter("w2_weight_scale_second", w2_weight_scale_second) # 在层中注册w2二级缩放因子
            set_weight_attrs(w2_weight_scale_second, extra_weight_attrs) # 设置w2二级缩放因子的额外属性

            w2_weight_offset_second = torch.nn.Parameter( # 创建w2二级权重偏移量参数
                torch.empty( # 创建空张量
                    num_experts, # 专家数量维度
                    hidden_size, # 隐藏大小维度
                    intermediate_size_per_partition // self.group_size, # 按分组大小划分的中间层维度
                    dtype=torch.float32, # float32数据类型
                ),
                requires_grad=False, # 不需要梯度
            )
            layer.register_parameter("w2_weight_offset_second", w2_weight_offset_second) # 在层中注册w2二级偏移量
            set_weight_attrs(w2_weight_offset_second, extra_weight_attrs) # 设置w2二级偏移量的额外属性

        w13_scale_bias = torch.nn.Parameter( # 创建w13缩放偏置参数
            torch.empty( # 创建空张量
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32 # 形状为[专家数, 2倍中间层大小, 1]，float32类型
            ),
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w13_scale_bias", w13_scale_bias) # 在层中注册w13缩放偏置
        set_weight_attrs(w13_scale_bias, extra_weight_attrs) # 设置w13缩放偏置的额外属性

        w2_scale_bias = torch.nn.Parameter( # 创建w2缩放偏置参数
            torch.empty( # 创建空张量
                num_experts, hidden_size, 16 // self.tp_size, dtype=torch.float32 # 形状为[专家数, 隐藏大小, 16/tp大小]，float32类型
            ),
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w2_scale_bias", w2_scale_bias) # 在层中注册w2缩放偏置
        set_weight_attrs(w2_scale_bias, extra_weight_attrs) # 设置w2缩放偏置的额外属性

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: # 权重加载后处理方法
        """权重加载后的后处理，委托给内核执行"""
        self.kernel.process_weights_after_loading( # 调用内核的权重后处理方法
            layer, self.is_per_channel_weight, self.activation_use_clip # 传入层、是否逐通道权重、是否使用激活裁剪
        )

    def create_moe_runner( # 创建MoE运行器方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig # 接收层和MoE运行器配置
    ):
        """创建MoE运行器，保存运行器配置"""
        self.moe_runner_config = moe_runner_config # 保存MoE运行器配置

    def apply_weights( # 应用权重方法，执行MoE推理
        self,
        layer: torch.nn.Module, # 目标神经网络层
        dispatch_output: StandardDispatchOutput, # 标准分发输出
    ) -> CombineInput: # 返回合并输入

        return self.kernel.apply(layer, dispatch_output) # 委托给内核执行权重应用

    def apply_weights_with_router_logits( # 使用路由器logits应用权重方法
        self,
        layer, # 目标层
        hidden_states, # 隐藏状态
        hidden_states_scale, # 隐藏状态缩放因子
        group_list_type, # 分组列表类型
        group_list, # 分组列表
        output_dtype, # 输出数据类型
    ):
        """使用路由器logits应用权重，执行不带路由权重的计算"""
        return self.kernel.apply_without_routing_weights( # 委托给内核执行不带路由权重的应用
            layer, # 目标层
            hidden_states, # 隐藏状态
            hidden_states_scale, # 隐藏状态缩放因子
            group_list_type, # 分组列表类型
            group_list, # 分组列表
            output_dtype, # 输出数据类型
        )
