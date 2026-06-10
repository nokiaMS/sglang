# W8A8 FP8量化方法实现文件
# 实现8位权重/8位激活的FP8量化配置和方法
# 包括W8A8Fp8Config配置类、W8A8Fp8LinearMethod线性层量化方法和
# W8A8FP8MoEMethod混合专家量化方法
# 支持离线量化的FP8检查点加载和在线量化两种模式
from __future__ import annotations  # 启用延迟类型注解求值

from typing import TYPE_CHECKING, Any, Dict, List, Optional  # 导入类型提示

import torch  # 导入PyTorch深度学习框架
from torch.nn.parameter import Parameter  # 导入参数类

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig  # 导入MoE相关类
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo  # 导入Triton MoE量化信息类
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter  # 导入参数子类
from sglang.srt.layers.quantization.base_config import (  # 导入量化基础配置类
    FusedMoEMethodBase,  # 融合MoE方法基类
    LinearMethodBase,  # 线性方法基类
    QuantizationConfig,  # 量化配置基类
    QuantizeMethodBase,  # 量化方法基类
)
from sglang.srt.layers.quantization.fp8_kernel import (  # 导入FP8内核函数
    fp8_dtype,  # FP8数据类型
    is_fp8_fnuz,  # 判断是否为FNUZ格式
    per_token_group_quant_fp8,  # 逐token组FP8量化
)
from sglang.srt.layers.quantization.fp8_utils import (  # 导入FP8工具函数
    apply_fp8_linear,  # 应用FP8线性变换
    cutlass_fp8_supported,  # 判断是否支持CUTLASS FP8
    input_to_float8,  # 输入转FP8
    normalize_e4m3fn_to_e4m3fnuz,  # 将E4M3FN归一化为E4M3FNUZ
)
from sglang.srt.utils import set_weight_attrs  # 导入权重属性设置函数

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入token分发器类型
        CombineInput,  # 合并输入
        StandardDispatchOutput,  # 标准分发输出
    )

_is_fp8_fnuz = is_fp8_fnuz()  # 判断当前FP8是否为FNUZ格式


class W8A8Fp8Config(QuantizationConfig):  # W8A8 FP8量化配置类
    """Config class for W8A8 FP8 Quantization.
    # W8A8 FP8量化配置类

    Weight Quantization:
    # 权重量化:
    - Method: Static quantization
    # 方法: 静态量化
    - Granularity: Per-channel
    # 粒度: 逐通道
    - Type: Symmetric
    # 类型: 对称

    Activation Quantization:
    # 激活量化:
    - Method: Dynamic quantization
    # 方法: 动态量化
    - Granularity: Per-token
    # 粒度: 逐token
    - Type: Symmetric
    # 类型: 对称

    Note:
    # 注意:
    - For models without offline quantization, weights will be quantized during model loading
    # 对于没有离线量化的模型，权重将在模型加载期间被量化
    - If CUTLASS is supported: Per-channel weight quantization is used
    # 如果支持CUTLASS: 使用逐通道权重量化
    - If CUTLASS is not supported: Falls back to per-tensor weight quantization
    # 如果不支持CUTLASS: 回退到逐张量权重量化
    """

    def __init__(self, is_checkpoint_fp8_serialized: bool = False):  # 初始化方法
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized  # 保存FP8序列化标志

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:  # 获取支持的激活数据类型
        return [torch.float16, torch.bfloat16]  # 支持float16和bfloat16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低GPU计算能力要求
        return 89  # 需要计算能力8.9(SM89/L40)

    @classmethod
    def get_name(self) -> str:  # 获取量化方法名称
        return "w8a8_fp8"  # 返回名称

    @classmethod
    def get_config_filenames(cls) -> List[str]:  # 获取配置文件名列表
        return []  # 无配置文件

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W8A8Fp8Config:  # 从配置字典创建配置实例
        quant_method = cls.get_from_keys(config, ["quant_method"])  # 获取量化方法名
        is_checkpoint_fp8_serialized = (  # 判断检查点是否以FP8序列化
            "compressed-tensors" in quant_method or "w8a8_fp8" in quant_method  # compressed-tensors或w8a8_fp8
        )
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized)  # 返回配置实例

    def get_quant_method(  # 获取量化方法
        self,
        layer: torch.nn.Module,  # 网络层
        prefix: str,  # 层前缀
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase  # 导入线性基类
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE类

        if isinstance(layer, LinearBase):  # 如果是线性层
            return W8A8Fp8LinearMethod(self)  # 返回W8A8 FP8线性方法
        elif isinstance(layer, FusedMoE):  # 如果是融合MoE层
            return W8A8FP8MoEMethod(self)  # 返回W8A8 FP8 MoE方法
        return None  # 其他情况返回None

    def get_scaled_act_names(self) -> List[str]:  # 获取需要缩放的激活名称
        return []  # 无需缩放的激活


class W8A8Fp8LinearMethod(LinearMethodBase):  # W8A8 FP8线性层量化方法类

    def __init__(self, quantization_config: W8A8Fp8Config):  # 初始化方法
        self.cutlass_fp8_supported = cutlass_fp8_supported()  # 检查是否支持CUTLASS FP8
        self.quantization_config = quantization_config  # 保存量化配置

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理
        weight = layer.weight  # 获取权重

        if self.quantization_config.is_checkpoint_fp8_serialized:  # 如果检查点已以FP8序列化
            weight_scale = layer.weight_scale.detach()  # 获取权重缩放因子(分离梯度)
            # If checkpoint offline quantized with w8a8_fp8, load the weight and weight_scale directly.
            # 如果检查点已用w8a8_fp8离线量化，直接加载权重和权重缩放因子。
            if _is_fp8_fnuz:  # 如果是FNUZ格式
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(  # 归一化为FNUZ格式
                    weight=weight, weight_scale=weight_scale  # 传入权重和缩放因子
                )

            layer.weight = Parameter(weight.t(), requires_grad=False)  # 转置权重并转为参数
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)  # 转缩放因子为参数
        else:  # 检查点未以FP8序列化
            # If checkpoint not offline quantized, quantize the weights with per-channel quantization.
            # 如果检查点未离线量化，使用逐通道量化对权重进行量化。
            if self.cutlass_fp8_supported:  # 如果支持CUTLASS
                # if cutlass supported, we use cutlass_scaled_mm
                # which requires per-channel quantization on weight
                # 如果支持cutlass，我们使用cutlass_scaled_mm
                # 这需要对权重进行逐通道量化
                qweight, weight_scale = per_token_group_quant_fp8(  # 逐token组FP8量化
                    layer.weight, layer.weight.shape[-1]  # 权重和组大小
                )
                weight_scale = weight_scale.t().contiguous()  # 转置缩放因子并使内存连续
            else:  # 不支持CUTLASS
                # if cutlass not supported, we fall back to use torch._scaled_mm
                # which requires per tensor quantization on weight
                # 如果不支持cutlass，我们回退到使用torch._scaled_mm
                # 这需要对权重进行逐张量量化
                qweight, weight_scale = input_to_float8(layer.weight, dtype=fp8_dtype)  # 转为FP8

            # Update the layer with the new values.
            # 用新值更新网络层。
            layer.weight = Parameter(qweight.t(), requires_grad=False)  # 转置量化权重并转为参数
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)  # 转缩放因子为参数
            layer.input_scale = None  # 清除输入缩放因子

    def create_weights(  # 创建线性层权重
        self,
        layer: torch.nn.Module,  # 目标网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: List[int],  # 输出分区大小列表
        input_size: int,  # 输入总大小
        output_size: int,  # 输出总大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        weight_dtype = (  # 确定权重数据类型
            torch.float8_e4m3fn  # FP8 E4M3
            if self.quantization_config.is_checkpoint_fp8_serialized  # 如果检查点已FP8序列化
            else params_dtype  # 否则使用参数数据类型
        )

        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器
        self.logical_widths = output_partition_sizes  # 保存逻辑宽度

        weight = ModelWeightParameter(  # 创建模型权重参数
            data=torch.empty(  # 创建空张量
                sum(output_partition_sizes),  # 输出维度大小之和
                input_size_per_partition,  # 输入分区大小
                dtype=weight_dtype,  # 权重数据类型
            ),
            input_dim=1,  # 输入维度索引
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("weight", weight)  # 注册权重参数

        if self.quantization_config.is_checkpoint_fp8_serialized:  # 如果检查点已FP8序列化
            weight_scale = ChannelQuantScaleParameter(  # 创建逐通道量化缩放参数
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),  # 每通道1个缩放值
                output_dim=0,  # 输出维度索引
                weight_loader=weight_loader,  # 权重加载器
            )
            layer.register_parameter("weight_scale", weight_scale)  # 注册权重缩放参数
        else:  # 检查点未FP8序列化
            layer.weight_scale = None  # 缩放因子设为None(将在加载后计算)

    def apply(  # 应用FP8线性变换
        self,
        layer: torch.nn.Module,  # 目标网络层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置项(可选)
    ):
        return apply_fp8_linear(  # 调用FP8线性变换函数
            x,  # 输入
            layer.weight,  # 权重
            layer.weight_scale,  # 权重缩放因子
            bias=bias,  # 偏置
            cutlass_fp8_supported=self.cutlass_fp8_supported,  # CUTLASS FP8支持标志
        )


class W8A8FP8MoEMethod(FusedMoEMethodBase):  # W8A8 FP8 MoE量化方法类
    """MoE method for FP8.
    # FP8的MoE方法。
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.
    # 支持加载具有静态权重缩放和动态/静态激活缩放的FP8检查点。
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    # 也支持加载具有动态激活缩放的量化FP16/BF16模型检查点。
    权重缩放因子将在模型权重加载后初始化。
    Args:
        quant_config: The quantization config.
    # 参数:
    #     quant_config: 量化配置。
    """

    def __init__(self, quant_config: W8A8Fp8Config):  # 初始化方法
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建MoE权重
        self,
        layer: torch.nn.Module,  # 目标网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入MoE权重缩放支持枚举

        # WEIGHTS
        # 权重
        w13_weight = torch.nn.Parameter(  # 创建w13权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量
                2 * intermediate_size_per_partition,  # 2倍中间层大小(门控)
                hidden_size,  # 隐藏层大小
                dtype=fp8_dtype,  # FP8数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight)  # 注册w13权重
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置额外属性

        w2_weight = torch.nn.Parameter(  # 创建w2权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量
                hidden_size,  # 隐藏层大小
                intermediate_size_per_partition,  # 中间层分区大小
                dtype=fp8_dtype,  # FP8数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight", w2_weight)  # 注册w2权重
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置额外属性

        w13_weight_scale = torch.nn.Parameter(  # 创建w13权重缩放因子参数
            torch.ones(  # 创建全1张量
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32  # 每通道1个缩放值
            ),
            requires_grad=False,  # 不需要梯度
        )
        w2_weight_scale = torch.nn.Parameter(  # 创建w2权重缩放因子参数
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),  # 每通道1个缩放值
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)  # 注册w13缩放因子
        layer.register_parameter("w2_weight_scale", w2_weight_scale)  # 注册w2缩放因子

        extra_weight_attrs.update(  # 更新额外属性
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}  # 设置量化方法为通道级
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)  # 设置w13缩放因子额外属性
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)  # 设置w2缩放因子额外属性

        w13_input_scale = None  # w13输入缩放因子初始化为None
        layer.register_parameter("w13_input_scale", w13_input_scale)  # 注册w13输入缩放因子

        w2_input_scale = None  # w2输入缩放因子初始化为None
        layer.register_parameter("w2_input_scale", w2_input_scale)  # 注册w2输入缩放因子

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理
        layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)  # 确保w13权重为参数
        layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)  # 确保w2权重为参数
        layer.w13_weight_scale = Parameter(  # 确保w13缩放因子为参数
            layer.w13_weight_scale.data, requires_grad=False  # 不需要梯度
        )
        layer.w2_weight_scale = Parameter(  # 确保w2缩放因子为参数
            layer.w2_weight_scale.data, requires_grad=False  # 不需要梯度
        )

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 层和配置
    ):
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)  # 创建Triton后端MoE运行器

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:  # 获取Triton量化信息
        return TritonMoeQuantInfo(  # 返回Triton量化信息对象
            w13_weight=layer.w13_weight,  # w13权重
            w2_weight=layer.w2_weight,  # w2权重
            use_fp8_w8a8=True,  # 使用FP8 W8A8
            per_channel_quant=True,  # 使用逐通道量化
            w13_scale=layer.w13_weight_scale,  # w13缩放因子
            w2_scale=layer.w2_weight_scale,  # w2缩放因子
            a13_scale=layer.w13_input_scale,  # w13输入缩放因子
            a2_scale=layer.w2_input_scale,  # w2输入缩放因子
        )

    def apply(  # 应用MoE方法
        self,
        layer: torch.nn.Module,  # 目标网络层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:

        quant_info = self.get_triton_quant_info(layer)  # 获取量化信息
        return self.runner.run(dispatch_output, quant_info)  # 运行MoE并返回结果
