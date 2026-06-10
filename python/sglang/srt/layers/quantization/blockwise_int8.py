# 分块 INT8 量化模块
# 本文件实现了分块 INT8 量化方法，支持静态权重缩放和动态激活缩放。
# 包含配置类 BlockInt8Config、线性层量化方法 BlockInt8LinearMethod
# 和 MoE 层量化方法 BlockInt8MoEMethod。

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py

from __future__ import annotations  # 启用延迟注解评估

import logging  # 导入日志模块
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # 导入类型注解工具

import torch  # 导入 PyTorch 深度学习框架
from torch.nn import Module  # 导入神经网络模块基类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小函数
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig  # 导入 MoE 运行器相关类
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo  # 导入 Triton MoE 量化信息类
from sglang.srt.layers.parameter import BlockQuantScaleParameter, ModelWeightParameter  # 导入自定义参数类
from sglang.srt.layers.quantization.base_config import (  # 导入量化配置基类
    FusedMoEMethodBase,  # 融合 MoE 方法基类
    LinearMethodBase,  # 线性方法基类
    QuantizationConfig,  # 量化配置基类
    QuantizeMethodBase,  # 量化方法基类
)
from sglang.srt.layers.quantization.int8_utils import apply_w8a8_block_int8_linear  # 导入分块 INT8 线性计算函数
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 导入未量化线性方法
from sglang.srt.layers.quantization.utils import is_layer_skipped  # 导入层跳过检查函数
from sglang.srt.utils import set_weight_attrs  # 导入权重属性设置工具

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import (  # MoE 令牌分发器类型
        CombineInput,  # 合并输入
        StandardDispatchOutput,  # 标准分发输出
    )

ACTIVATION_SCHEMES = ["static", "dynamic"]  # 支持的激活量化方案列表

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


class BlockInt8Config(QuantizationConfig):  # 分块 INT8 量化配置类
    """Config class for INT8."""  # INT8 配置类

    def __init__(  # 初始化方法
        self,
        is_checkpoint_int8_serialized: bool = False,  # 检查点是否以 INT8 序列化，默认 False
        activation_scheme: str = "dynamic",  # 激活量化方案，默认动态
        ignored_layers: Optional[List[str]] = None,  # 忽略量化的层列表
        weight_block_size: List[int] = None,  # 权重量化块大小
    ) -> None:
        self.is_checkpoint_int8_serialized = is_checkpoint_int8_serialized  # 保存 INT8 序列化标志
        if is_checkpoint_int8_serialized:  # 如果检查点已 INT8 序列化
            logger.warning(  # 记录警告
                "Detected int8 checkpoint. Please note that the "  # 检测到 int8 检查点。请注意
                "format is experimental and subject to change."  # 该格式是实验性的，可能会变更
            )
        if activation_scheme not in ACTIVATION_SCHEMES:  # 检查激活方案是否支持
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")  # 不支持时抛出错误
        self.activation_scheme = activation_scheme  # 保存激活方案
        self.ignored_layers = ignored_layers or []  # 保存忽略层列表
        if weight_block_size is not None:  # 如果指定了权重块大小
            if not is_checkpoint_int8_serialized:  # 检查点未 INT8 序列化时报错
                raise ValueError(
                    f"The block-wise quantization only supports int8-serialized checkpoint for now."  # 分块量化目前仅支持 int8 序列化检查点
                )
            if len(weight_block_size) != 2:  # 块大小维度必须为 2
                raise ValueError(
                    f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions."  # 权重量化块大小必须有 2 个维度
                )
            if activation_scheme != "dynamic":  # 块量化目前仅支持动态激活方案
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."  # 分块量化目前仅支持动态激活方案
                )
        self.weight_block_size = weight_block_size  # 保存权重块大小

    @classmethod
    def get_name(cls) -> str:  # 获取量化方法名称
        return "blockwise_int8"  # 返回名称

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:  # 获取支持的激活数据类型
        return [torch.bfloat16, torch.half]  # 支持 bfloat16 和 float16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低 GPU 计算能力要求
        return 80  # 最低需要计算能力 8.0

    @classmethod
    def get_config_filenames(cls) -> List[str]:  # 获取配置文件名列表
        return []  # 无配置文件

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BlockInt8Config:  # 从配置字典创建配置对象
        quant_method = cls.get_from_keys(config, ["quant_method"])  # 获取量化方法
        is_checkpoint_int8_serialized = "int8" in quant_method  # 判断是否为 INT8 序列化
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])  # 获取激活方案
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)  # 获取忽略层列表
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)  # 获取权重块大小
        return cls(  # 返回新创建的配置对象
            is_checkpoint_int8_serialized=is_checkpoint_int8_serialized,  # INT8 序列化标志
            activation_scheme=activation_scheme,  # 激活方案
            ignored_layers=ignored_layers,  # 忽略层列表
            weight_block_size=weight_block_size,  # 权重块大小
        )

    def get_quant_method(  # 获取适用于指定层的量化方法
        self, layer: torch.nn.Module, prefix: str  # 目标层和层前缀
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合 MoE 层

        if isinstance(layer, LinearBase):  # 如果是线性层
            if is_layer_skipped(prefix, self.ignored_layers):  # 检查是否跳过该层
                return UnquantizedLinearMethod()  # 返回未量化的线性方法
            return BlockInt8LinearMethod(self)  # 返回分块 INT8 线性方法
        elif isinstance(layer, FusedMoE):  # 如果是融合 MoE 层
            return BlockInt8MoEMethod(self)  # 返回分块 INT8 MoE 方法
        return None  # 不支持的层返回 None

    def get_scaled_act_names(self) -> List[str]:  # 获取需要后缩放的激活函数名列表
        return []  # 分块 INT8 不需要后缩放


class BlockInt8LinearMethod(LinearMethodBase):  # 分块 INT8 线性层量化方法
    """Linear method for INT8.  # INT8 线性方法
    Supports loading INT8 checkpoints with static weight scale and  # 支持加载具有静态权重缩放和
    dynamic activation scale.  # 动态激活缩放的 INT8 检查点

    Limitations:  # 限制
    Only support block-wise int8 quantization and int8 checkpoint  # 仅支持分块 int8 量化和 int8 检查点

    Args:  # 参数
        quant_config: The quantization config.  # 量化配置
    """

    def __init__(self, quant_config: BlockInt8Config):  # 初始化方法
        self.quant_config = quant_config  # 保存量化配置
        assert self.quant_config.weight_block_size is not None  # 断言权重块大小不为空
        assert self.quant_config.is_checkpoint_int8_serialized  # 断言检查点已 INT8 序列化

    def create_weights(  # 创建线性层权重
        self,
        layer: torch.nn.Module,  # 目标层
        input_size_per_partition: int,  # 当前分区的输入维度大小
        output_partition_sizes: List[int],  # 各逻辑权重的输出维度大小列表
        input_size: int,  # 跨所有秩的输入维度总大小
        output_size: int,  # 跨所有秩的输出维度总大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        output_size_per_partition = sum(output_partition_sizes)  # 计算分区总输出大小
        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器

        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小

        block_n, block_k = (  # 获取量化块的 N 和 K 维度
            self.quant_config.weight_block_size[0],  # 块的 N 维度（输出方向）
            self.quant_config.weight_block_size[1],  # 块的 K 维度（输入方向）
        )
        # Required by row parallel  # 行并行所需
        if tp_size > 1 and input_size // input_size_per_partition == tp_size:  # 如果使用行并行
            if input_size_per_partition % block_k != 0:  # 检查输入维度是否能被 block_k 整除
                raise ValueError(  # 不能整除时报错
                    f"Weight input_size_per_partition = "  # 权重输入维度
                    f"{input_size_per_partition} is not divisible by "  # 不能被整除
                    f"weight quantization block_k = {block_k}."  # 权重量化块 K 维度
                )
        # Required by column parallel or enabling merged weights  # 列并行或启用合并权重所需
        if (tp_size > 1 and output_size // output_size_per_partition == tp_size) or len(  # 列并行或合并权重
            output_partition_sizes
        ) > 1:
            for output_partition_size in output_partition_sizes:  # 遍历每个输出分区大小
                if output_partition_size % block_n != 0:  # 检查是否能被 block_n 整除
                    raise ValueError(  # 不能整除时报错
                        f"Weight output_partition_size = "  # 权重输出分区大小
                        f"{output_partition_size} is not divisible by "  # 不能被整除
                        f"weight quantization block_n = {block_n}."  # 权重量化块 N 维度
                    )

        layer.logical_widths = output_partition_sizes  # 保存逻辑宽度列表

        layer.input_size_per_partition = input_size_per_partition  # 保存分区输入大小
        layer.output_size_per_partition = output_size_per_partition  # 保存分区输出大小
        layer.orig_dtype = params_dtype  # 保存原始数据类型

        # WEIGHT  # 权重
        weight_dtype = (  # 确定权重数据类型
            torch.int8  # 如果 INT8 序列化则使用 int8
            if self.quant_config.is_checkpoint_int8_serialized  # 判断是否 INT8 序列化
            else params_dtype  # 否则使用原始数据类型
        )

        weight = ModelWeightParameter(  # 创建模型权重参数
            data=torch.empty(  # 创建空张量
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype  # 输出 x 输入维度
            ),
            input_dim=1,  # 输入维度索引
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("weight", weight)  # 注册权重参数

        # WEIGHT SCALE  # 权重缩放因子

        scale = BlockQuantScaleParameter(  # 创建分块量化缩放参数
            data=torch.empty(  # 创建空张量
                (output_size_per_partition + block_n - 1) // block_n,  # N 方向的块数（向上取整）
                (input_size_per_partition + block_k - 1) // block_k,  # K 方向的块数（向上取整）
                dtype=torch.float32,  # 数据类型为 float32
            ),
            input_dim=1,  # 输入维度索引
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        scale[:] = torch.finfo(torch.float32).min  # 初始化为 float32 最小值
        layer.register_parameter("weight_scale_inv", scale)  # 注册权重缩放参数

        # INPUT ACTIVATION SCALE  # 输入激活缩放因子
        assert self.quant_config.activation_scheme == "dynamic"  # 断言使用动态激活方案
        layer.register_parameter("input_scale", None)  # 注册空输入缩放参数

    def process_weights_after_loading(self, layer: Module) -> None:  # 权重加载后处理
        # Block quant doesn't need to process weights after loading  # 分块量化不需要加载后处理权重
        # Use torch Parameter to avoid cuda graph capturing issue  # 使用 torch Parameter 以避免 CUDA 图捕获问题
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)  # 将权重转为不可训练参数
        layer.weight_scale_inv = torch.nn.Parameter(  # 将缩放因子转为不可训练参数
            layer.weight_scale_inv.data, requires_grad=False  # 不需要梯度
        )

    def apply(  # 应用量化方法进行前向计算
        self,
        layer: torch.nn.Module,  # 目标层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置张量，可选
    ) -> torch.Tensor:
        return apply_w8a8_block_int8_linear(  # 调用分块 INT8 线性计算函数
            input=x,  # 输入张量
            weight=layer.weight,  # 权重
            block_size=self.quant_config.weight_block_size,  # 量化块大小
            weight_scale=layer.weight_scale_inv,  # 权重缩放因子
            input_scale=None,  # 输入缩放因子（动态模式下为 None）
            bias=bias,  # 偏置
        )


class BlockInt8MoEMethod(FusedMoEMethodBase):  # 分块 INT8 MoE 层量化方法
    """MoE method for INT8.  # INT8 的 MoE 方法
    Supports loading INT8 checkpoints with static weight scale and  # 支持加载具有静态权重缩放和
    dynamic activation scale.  # 动态激活缩放的 INT8 检查点

    Limitations:  # 限制
    Only support block-wise int8 quantization and int8 checkpoint  # 仅支持分块 int8 量化和 int8 检查点

    Args:  # 参数
        quant_config: The quantization config.  # 量化配置
    """

    def __init__(self, quant_config: BlockInt8Config):  # 初始化方法
        self.quant_config = quant_config  # 保存量化配置
        assert self.quant_config.weight_block_size is not None  # 断言权重块大小不为空
        assert self.quant_config.is_checkpoint_int8_serialized  # 断言检查点已 INT8 序列化

    def create_weights(  # 创建 MoE 层权重
        self,
        layer: Module,  # 目标层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 当前分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入 MoE 权重缩放支持枚举

        if self.quant_config.is_checkpoint_int8_serialized:  # 如果 INT8 序列化
            params_dtype = torch.int8  # 使用 int8 数据类型
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小

        block_n, block_k = (  # 获取量化块的 N 和 K 维度
            self.quant_config.weight_block_size[0],  # 块的 N 维度（输出方向）
            self.quant_config.weight_block_size[1],  # 块的 K 维度（输入方向）
        )
        # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.  # 注意(HandH1998): 为确保分块量化缩放的正确对齐，gate 和 up 层权重的 output_size 必须能被 block_n 整除
        # Required by column parallel or enabling merged weights  # 列并行或启用合并权重所需
        if intermediate_size_per_partition % block_n != 0:  # 检查中间层大小是否能被 block_n 整除
            raise ValueError(  # 不能整除时报错
                f"The output_size of gate's and up's weight = "  # gate 和 up 权重的 output_size
                f"{intermediate_size_per_partition} is not divisible by "  # 不能被整除
                f"weight quantization block_n = {block_n}."  # 权重量化块 N 维度
            )
        if tp_size > 1:  # 如果使用张量并行
            # Required by row parallel  # 行并行所需
            if intermediate_size_per_partition % block_k != 0:  # 检查中间层大小是否能被 block_k 整除
                raise ValueError(  # 不能整除时报错
                    f"The input_size of down's weight = "  # down 权重的 input_size
                    f"{intermediate_size_per_partition} is not divisible by "  # 不能被整除
                    f"weight quantization block_k = {block_k}."  # 权重量化块 K 维度
                )

        # WEIGHTS  # 权重
        w13_weight = torch.nn.Parameter(  # 创建 w13（gate_up）权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                2 * intermediate_size_per_partition,  # 2 倍中间层大小
                hidden_size,  # 隐藏层大小
                dtype=params_dtype,  # 数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight)  # 注册 w13 权重参数
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置额外属性

        w2_weight = torch.nn.Parameter(  # 创建 w2（down）权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小
                intermediate_size_per_partition,  # 中间层大小
                dtype=params_dtype,  # 数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight", w2_weight)  # 注册 w2 权重参数
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置额外属性

        # WEIGHT_SCALES  # 权重缩放因子
        w13_weight_scale = torch.nn.Parameter(  # 创建 w13 权重缩放参数
            torch.ones(  # 创建全 1 张量
                num_experts,  # 专家数量维度
                2 * ((intermediate_size_per_partition + block_n - 1) // block_n),  # 2 倍 N 方向块数
                (hidden_size + block_k - 1) // block_k,  # K 方向块数
                dtype=torch.float32,  # 数据类型为 float32
            ),
            requires_grad=False,  # 不需要梯度
        )
        w2_weight_scale = torch.nn.Parameter(  # 创建 w2 权重缩放参数
            torch.ones(  # 创建全 1 张量
                num_experts,  # 专家数量维度
                (hidden_size + block_n - 1) // block_n,  # N 方向块数
                (intermediate_size_per_partition + block_k - 1) // block_k,  # K 方向块数
                dtype=torch.float32,  # 数据类型为 float32
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)  # 注册 w13 缩放参数
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)  # 注册 w2 缩放参数

        extra_weight_attrs.update(  # 更新额外权重属性
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}  # 设置量化方法为分块量化
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)  # 设置 w13 缩放额外属性
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)  # 设置 w2 缩放额外属性

        # INPUT_SCALES  # 输入缩放因子
        assert self.quant_config.activation_scheme == "dynamic"  # 断言使用动态激活方案
        layer.w13_input_scale = None  # w13 输入缩放为空（动态模式）
        layer.w2_input_scale = None  # w2 输入缩放为空（动态模式）

    def process_weights_after_loading(self, layer: Module) -> None:  # 权重加载后处理
        # Block quant doesn't need to process weights after loading  # 分块量化不需要加载后处理权重
        return  # 直接返回

    def create_moe_runner(  # 创建 MoE 运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 目标层和 MoE 运行器配置
    ):
        self.moe_runner_config = moe_runner_config  # 保存 MoE 运行器配置
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)  # 创建 Triton 后端 MoE 运行器

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:  # 获取 Triton MoE 量化信息
        return TritonMoeQuantInfo(  # 返回 Triton MoE 量化信息对象
            w13_weight=layer.w13_weight,  # w13 权重
            w2_weight=layer.w2_weight,  # w2 权重
            use_int8_w8a8=True,  # 标记使用 INT8 W8A8
            w13_scale=layer.w13_weight_scale_inv,  # w13 权重缩放
            w2_scale=layer.w2_weight_scale_inv,  # w2 权重缩放
            a13_scale=layer.w13_input_scale,  # w13 激活缩放
            a2_scale=layer.w2_input_scale,  # w2 激活缩放
            block_shape=self.quant_config.weight_block_size,  # 量化块形状
        )

    def apply(  # 应用 MoE 量化方法
        self,
        layer: torch.nn.Module,  # 目标层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:

        quant_info = self.get_triton_quant_info(layer)  # 获取量化信息

        return self.runner.run(dispatch_output, quant_info)  # 运行 MoE 并返回结果
