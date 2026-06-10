# 文件说明：GPTQ量化在Intel CPU上的AMX实现，包含线性层和MoE两种GPTQ量化方法
from __future__ import annotations  # 启用延迟类型注解求值

from typing import TYPE_CHECKING, List, Optional  # 导入类型检查、列表和可选类型

import torch  # 导入PyTorch深度学习框架

from sglang.srt.layers.moe import (  # 导入MoE运行器配置类
    MoeRunnerConfig,
)
from sglang.srt.layers.parameter import (  # 导入量化参数类
    ChannelQuantScaleParameter,  # 通道级量化缩放参数
    GroupQuantScaleParameter,  # 分组级量化缩放参数
    PackedColumnParameter,  # 打包列参数
    PackedvLLMParameter,  # vLLM打包参数
    RowvLLMParameter,  # vLLM行参数
)
from sglang.srt.layers.quantization.base_config import (  # 导入量化方法基类
    FusedMoEMethodBase,  # MoE方法基类
    LinearMethodBase,  # 线性方法基类
)

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入标准分发输出类型
        StandardDispatchOutput,
    )

from sglang.srt.layers.amx_utils import (  # 导入AMX工具函数
    CPUQuantMethod,  # CPU量化方法枚举
    _amx_process_weight_after_loading,  # AMX权重加载后处理函数
)

from .gptq import GPTQConfig  # 导入GPTQ配置类


class CPUGPTQConfig(GPTQConfig):  # CPU GPTQ配置类，继承自GPTQ配置类
    """CPU Config class for AWQ, inherit from AWQConfig"""  # CPU上AWQ的配置类，继承自AWQConfig（注意：实际继承自GPTQConfig）

    @classmethod  # 类方法装饰器
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:  # 获取支持的激活值数据类型
        return [torch.half, torch.bfloat16]  # 支持半精度和BF16数据类型

    def get_quant_method(  # 获取量化方法
        self, layer: torch.nn.Module, prefix: str  # 目标层和前缀字符串
    ) -> Optional[LinearMethodBase]:  # 返回可选的线性方法基类实例
        # Delay the import to avoid circular dependency  # 延迟导入以避免循环依赖
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE层

        if isinstance(layer, FusedMoE):  # 如果是MoE层
            return GPTQMoEIntelAMXMethod(self)  # 返回GPTQ MoE Intel AMX方法实例

        if isinstance(layer, LinearBase):  # 如果是线性层
            return GPTQLinearIntelAMXMethod(self)  # 返回GPTQ线性Intel AMX方法实例


class GPTQLinearIntelAMXMethod(LinearMethodBase):  # GPTQ线性层Intel AMX方法类，继承自线性方法基类
    """Linear method for GPTQ on Intel CPU with AMX."""  # 在Intel CPU上使用AMX的GPTQ线性方法

    def __init__(self, quant_config: GPTQConfig):  # 初始化方法，接收GPTQ量化配置
        self.quant_config = quant_config  # 保存量化配置
        # GPTQ v1 and v2 format deals with zero points differently  # GPTQ v1和v2格式处理零点的方式不同
        self.use_v2_format = quant_config.checkpoint_format == "gptq_v2"  # 判断是否使用v2格式

    def create_weights(  # 创建量化权重参数
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: list[int],  # 输出分区大小列表
        input_size: int,  # 完整输入大小
        output_size: int,  # 完整输出大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        del output_size  # Unused.  # 未使用，删除该参数
        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器
        if input_size_per_partition % self.quant_config.group_size != 0:  # 检查输入大小是否与分组大小对齐
            raise ValueError(  # 抛出值错误
                "The input size is not aligned with the quantized "  # 输入大小与量化权重形状不对齐
                "weight shape. This can be caused by too large "  # 这可能是由于张量并行大小过大导致
                "tensor parallel size."
            )
        output_size_per_partition = sum(output_partition_sizes)  # 计算分区输出总大小
        if output_size_per_partition % self.quant_config.pack_factor.numerator != 0:  # 检查输出大小是否与打包因子对齐
            raise ValueError(  # 抛出值错误
                "The output size is not aligned with the quantized "  # 输出大小与量化权重形状不对齐
                "weight shape. This can be caused by too large "  # 这可能是由于张量并行大小过大导致
                "tensor parallel size."
            )

        if self.quant_config.desc_act and not (  # 如果使用激活值降序排列，但不同时满足以下条件
            self.quant_config.true_sequential and self.quant_config.static_groups
        ):
            raise ValueError(  # 抛出值错误
                "Currently, desc_act (True) is only supported with sequential and static group on CPU with AMX."  # 当前CPU AMX仅支持顺序和静态分组下的desc_act
            )
        if self.quant_config.weight_bits != 4:  # 检查权重位数是否为4位
            raise ValueError("Currently, only 4bits is supported on CPU with AMX.")  # 当前CPU AMX仅支持4位量化
        if self.use_v2_format:  # 检查是否使用v2格式
            raise ValueError("Currently, gptq_v2 is not supported on CPU with AMX.")  # 当前CPU AMX不支持gptq_v2格式

        if self.quant_config.group_size != -1:  # 如果分组大小不为-1
            group_size = self.quant_config.group_size  # 使用配置的分组大小
        else:
            group_size = input_size  # 否则使用输入大小作为分组大小

        scale_and_zero_size = input_size_per_partition // group_size  # 计算缩放因子和零点的大小
        scale_and_zero_input_dim = 0  # 缩放因子和零点的输入维度为0

        qweight = PackedvLLMParameter(  # 创建打包量化权重参数
            data=torch.empty(  # 分配空张量
                input_size_per_partition // self.quant_config.pack_factor,  # 打包后的输入维度大小
                output_size_per_partition,  # 输出维度大小
                dtype=torch.int32,  # 32位整数类型
            ),
            input_dim=0,  # 输入维度索引
            output_dim=1,  # 输出维度索引
            packed_dim=0,  # 打包维度索引
            packed_factor=self.quant_config.pack_factor,  # 打包因子
            weight_loader=weight_loader,  # 权重加载器
        )

        g_idx = RowvLLMParameter(  # 创建分组索引参数
            data=torch.tensor(  # 创建分组索引张量
                [
                    i // self.quant_config.group_size  # 每个元素除以分组大小得到组号
                    for i in range(input_size_per_partition)  # 遍历输入分区大小
                ],
                dtype=torch.int32,  # 32位整数类型
            ),
            input_dim=0,  # 输入维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        qzeros_args = {  # 量化零点参数字典
            "data": torch.empty(  # 分配空张量
                scale_and_zero_size,  # 缩放和零点大小
                output_size_per_partition // self.quant_config.pack_factor,  # 打包后的输出维度大小
                dtype=torch.int32,  # 32位整数类型
            ),
            "weight_loader": weight_loader,  # 权重加载器
        }
        weight_scale_args = {  # 权重缩放因子参数字典
            "data": torch.empty(  # 分配空张量
                scale_and_zero_size,  # 缩放和零点大小
                output_size_per_partition,  # 输出维度大小
                dtype=params_dtype,  # 参数数据类型
            ),
            "weight_loader": weight_loader,  # 权重加载器
        }
        if scale_and_zero_input_dim is None:  # 如果缩放因子输入维度为None（通道级量化）
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)  # 创建通道级缩放参数
            qzeros = PackedColumnParameter(  # 创建打包列零点参数
                output_dim=1,  # 输出维度索引
                packed_dim=1,  # 打包维度索引
                packed_factor=self.quant_config.pack_factor,  # 打包因子
                **qzeros_args,  # 零点参数
            )

        else:  # 否则使用分组级量化
            scales = GroupQuantScaleParameter(  # 创建分组级缩放参数
                output_dim=1, input_dim=0, **weight_scale_args  # 指定输出和输入维度
            )
            qzeros = PackedvLLMParameter(  # 创建vLLM打包零点参数
                input_dim=0,  # 输入维度索引
                output_dim=1,  # 输出维度索引
                packed_dim=1,  # 打包维度索引
                packed_factor=self.quant_config.pack_factor,  # 打包因子
                **qzeros_args,  # 零点参数
            )

        layer.register_parameter("qweight", qweight)  # 注册量化权重参数到层
        layer.register_parameter("g_idx", g_idx)  # 注册分组索引参数到层
        layer.register_parameter("qzeros", qzeros)  # 注册量化零点参数到层
        layer.register_parameter("scales", scales)  # 注册缩放因子参数到层

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的处理方法，使用AMX格式转换
        _amx_process_weight_after_loading(  # 调用AMX权重加载后处理函数
            layer, ["qweight", "qzeros", "scales"], None, "gptq"  # 处理量化权重、零点和缩放因子
        )

    def apply(  # 应用量化权重进行前向计算
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 可选偏置张量
    ) -> torch.Tensor:
        return torch.ops.sgl_kernel.int4_scaled_mm_cpu(  # 调用int4量化矩阵乘法CPU算子
            x,  # 输入张量
            layer.qweight,  # 量化权重
            layer.qzeros,  # 量化零点
            layer.scales,  # 缩放因子
            bias,  # 偏置
        )


class GPTQMoEIntelAMXMethod(FusedMoEMethodBase):  # GPTQ MoE Intel AMX方法类，继承自MoE方法基类
    """MoE method for GPTQ on Intel CPU with AMX."""  # 在Intel CPU上使用AMX的GPTQ MoE方法

    def __init__(self, quant_config: GPTQConfig):  # 初始化方法，接收GPTQ量化配置
        super().__init__()  # 调用父类初始化
        self.quant_config = quant_config  # 保存量化配置
        self.use_v2_format = quant_config.checkpoint_format == "gptq_v2"  # 判断是否使用v2格式
        self.moe_runner_config: Optional[MoeRunnerConfig] = None  # MoE运行器配置，初始为None

    def create_weights(  # 创建MoE量化权重参数
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        if self.quant_config.desc_act and not (  # 如果使用激活值降序排列，但不同时满足以下条件
            self.quant_config.true_sequential and self.quant_config.static_groups
        ):
            raise ValueError(  # 抛出值错误
                "Currently, desc_act (True) is only supported with sequential and static group on CPU with AMX."  # 当前CPU AMX仅支持顺序和静态分组下的desc_act
            )
        if self.quant_config.weight_bits != 4:  # 检查权重位数是否为4位
            raise ValueError("Currently, only 4bits is supported on CPU with AMX.")  # 当前CPU AMX仅支持4位量化
        if self.use_v2_format:  # 检查是否使用v2格式
            raise ValueError("Currently, gptq_v2 is not supported on CPU with AMX.")  # 当前CPU AMX不支持gptq_v2格式
        # Delay the import to avoid circular dependency  # 延迟导入以避免循环依赖
        from sglang.srt.layers.linear import set_weight_attrs  # 导入设置权重属性的工具函数
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入MoE权重缩放支持枚举

        if self.quant_config.group_size != -1:  # 如果分组大小不为-1（使用分组量化）
            scales_size13 = hidden_size // self.quant_config.group_size  # 计算w13缩放大小
            w2_scales_size = intermediate_size_per_partition  # w2缩放大小为分区中间层大小
            scales_size2 = w2_scales_size // self.quant_config.group_size  # 计算w2缩放大小
            strategy = FusedMoeWeightScaleSupported.GROUP.value  # 量化策略为分组量化
        else:  # 使用通道级量化
            scales_size13 = 1  # 通道级量化缩放大小为1
            scales_size2 = 1  # 通道级量化缩放大小为1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value  # 量化策略为通道量化

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})  # 更新额外权重属性
        # Fused gate_up_proj (column parallel)  # 融合gate_up投影（列并行）
        w13_qweight = torch.nn.Parameter(  # 创建w13量化权重参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                hidden_size // self.quant_config.pack_factor,  # 打包后的隐藏层大小
                2 * intermediate_size_per_partition,  # gate和up拼接的中间层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_qweight", w13_qweight)  # 注册w13量化权重参数
        set_weight_attrs(w13_qweight, extra_weight_attrs)  # 设置w13量化权重的额外属性
        # down_proj (row parallel)  # 下投影（行并行）
        w2_qweight = torch.nn.Parameter(  # 创建w2量化权重参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                intermediate_size_per_partition // self.quant_config.pack_factor,  # 打包后的中间层大小
                hidden_size,  # 隐藏层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_qweight", w2_qweight)  # 注册w2量化权重参数
        set_weight_attrs(w2_qweight, extra_weight_attrs)  # 设置w2量化权重的额外属性
        # up_proj scales  # 上投影缩放因子
        w13_scales = torch.nn.Parameter(  # 创建w13缩放因子参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                scales_size13,  # w13缩放大小
                2 * intermediate_size_per_partition,  # gate和up拼接的中间层大小
                dtype=params_dtype,  # 参数数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_scales", w13_scales)  # 注册w13缩放因子参数
        set_weight_attrs(w13_scales, extra_weight_attrs)  # 设置w13缩放因子的额外属性
        # down_proj scales  # 下投影缩放因子
        w2_scales = torch.nn.Parameter(  # 创建w2缩放因子参数
            torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),  # 分配空张量
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_scales", w2_scales)  # 注册w2缩放因子参数
        set_weight_attrs(w2_scales, extra_weight_attrs)  # 设置w2缩放因子的额外属性
        # dont shard the w2 scales when running act order  # 使用激活值降序排列时不切分w2缩放因子
        set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})  # 如果使用desc_act则需要加载完整w2
        # up_proj scales  # 上投影缩放因子（此处实际为零点）
        w13_qzeros = torch.nn.Parameter(  # 创建w13量化零点参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                scales_size13,  # w13缩放大小
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,  # 打包后的中间层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)  # 注册w13量化零点参数
        set_weight_attrs(w13_qzeros, extra_weight_attrs)  # 设置w13量化零点的额外属性
        # down_proj scales  # 下投影缩放因子（此处实际为零点）
        w2_qzeros = torch.nn.Parameter(  # 创建w2量化零点参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                scales_size2,  # w2缩放大小
                hidden_size // self.quant_config.pack_factor,  # 打包后的隐藏层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)  # 注册w2量化零点参数
        set_weight_attrs(w2_qzeros, extra_weight_attrs)  # 设置w2量化零点的额外属性
        # dont shard the w2 scales when running act order  # 使用激活值降序排列时不切分w2零点
        set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})  # 如果使用desc_act则需要加载完整w2
        w13_g_idx = torch.nn.Parameter(  # 创建w13分组索引参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)  # 注册w13分组索引参数
        set_weight_attrs(w13_g_idx, extra_weight_attrs)  # 设置w13分组索引的额外属性
        w2_g_idx = torch.nn.Parameter(  # 创建w2分组索引参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                intermediate_size_per_partition,  # 每个分区的中间层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)  # 注册w2分组索引参数
        set_weight_attrs(w2_g_idx, extra_weight_attrs)  # 设置w2分组索引的额外属性

    def create_moe_runner(  # 创建MoE运行器
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        moe_runner_config: MoeRunnerConfig,  # MoE运行器配置
        **extra_weight_attrs,  # 额外权重属性
    ):
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的处理方法，使用AMX格式转换
        _amx_process_weight_after_loading(  # 处理w13权重
            layer, ["w13_qweight", "w13_qzeros", "w13_scales"], None, "gptq"  # 处理w13量化权重、零点和缩放因子
        )
        _amx_process_weight_after_loading(  # 处理w2权重
            layer, ["w2_qweight", "w2_qzeros", "w2_scales"], None, "gptq"  # 处理w2量化权重、零点和缩放因子
        )

    def apply(  # 应用量化权重进行MoE前向计算
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入类型

        assert (  # 断言
            self.moe_runner_config.activation == "silu"  # MoE运行器配置的激活函数必须是SiLU
        ), "Only SiLU activation is supported."  # 仅支持SiLU激活函数

        x = dispatch_output.hidden_states  # 获取分发输出的隐藏状态
        topk_output = dispatch_output.topk_output  # 获取分发输出的topk结果
        topk_weights, topk_ids, _ = topk_output  # 解包topk权重、ID
        output = torch.ops.sgl_kernel.fused_experts_cpu(  # 调用CPU融合专家计算算子
            x,  # 输入张量
            layer.w13_qweight,  # w13量化权重
            layer.w2_qweight,  # w2量化权重
            topk_weights,  # topk权重
            topk_ids,  # topk ID
            False,  # inplace See [Note] inplace should be False in fused_experts.  # 原地操作标志，参见注释：fused_experts中inplace应为False
            CPUQuantMethod.INT4_W4A8,  # CPU量化方法为INT4 W4A8
            layer.w13_scales,  # w1_scale  # w1缩放因子
            layer.w2_scales,  # w2_scale  # w2缩放因子
            layer.w13_qzeros,  # w13量化零点
            layer.w2_qzeros,  # w2量化零点
            None,  # block_size  # 块大小，不使用
            None,  # w1 bias  # w1偏置，不使用
            None,  # w3 bias  # w3偏置，不使用
            None,  # alpha  # alpha参数，不使用
            None,  # limit  # limit参数，不使用
            True,  # is_vnni  # 是否使用VNNI格式
        )
        return StandardCombineInput(hidden_states=output)  # 返回标准合并输入，包含输出隐藏状态
