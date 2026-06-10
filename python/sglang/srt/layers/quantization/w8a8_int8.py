# W8A8 INT8量化方法实现文件
# 实现8位权重/8位激活的INT8量化配置和方法
# 包括W8A8Int8Config配置类、W8A8Int8LinearMethod线性层量化方法和
# W8A8Int8MoEMethod混合专家量化方法
# 支持CPU(AMX/ARM64)和GPU(CUDA)平台
from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 导入日志模块
from types import MappingProxyType  # 导入映射代理类型(只读字典)
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, cast  # 导入类型提示

import torch  # 导入PyTorch深度学习框架
from torch.nn.parameter import Parameter  # 导入参数类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小函数
from sglang.srt.layers.amx_utils import (  # 导入AMX相关工具
    CPUQuantMethod,  # CPU量化方法枚举
    _amx_process_weight_after_loading,  # AMX权重加载后处理函数
)
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig  # 导入MoE相关类
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo  # 导入Triton MoE量化信息类
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter  # 导入参数子类
from sglang.srt.layers.quantization.base_config import (  # 导入量化基础配置类
    FusedMoEMethodBase,  # 融合MoE方法基类
    LinearMethodBase,  # 线性方法基类
    QuantizationConfig,  # 量化配置基类
    QuantizeMethodBase,  # 量化方法基类
)
from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer  # 导入层忽略判断函数
from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8  # 导入逐token INT8量化函数
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 导入未量化线性方法
from sglang.srt.utils import (  # 导入工具函数
    cpu_has_amx_support,  # 检查CPU是否支持AMX
    is_cpu,  # 判断是否为CPU平台
    is_cuda,  # 判断是否为CUDA平台
    is_host_cpu_arm64,  # 判断是否为ARM64主机CPU
    set_weight_attrs,  # 设置权重属性
    use_intel_amx_backend,  # 判断是否使用Intel AMX后端
)
from sglang.srt.utils.patch_torch import register_fake_if_exists  # 导入torch补丁注册函数

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput  # 导入标准分发输出类型

_is_cuda = is_cuda()  # 是否为CUDA平台
_is_cpu_amx_available = cpu_has_amx_support()  # CPU AMX是否可用
_is_cpu = is_cpu()  # 是否为CPU平台
_is_cpu_arm64 = is_host_cpu_arm64()  # 是否为ARM64 CPU

if _is_cuda:  # 如果是CUDA平台
    from sgl_kernel import int8_scaled_mm  # 导入INT8缩放矩阵乘法

    @register_fake_if_exists("sgl_kernel::int8_scaled_mm")  # 注册抽象实现(如果存在)
    def _int8_scaled_mm_abstract(  # INT8缩放矩阵乘法的抽象实现
        mat_a,  # 矩阵A
        mat_b,  # 矩阵B
        scales_a,  # 矩阵A的缩放因子
        scales_b,  # 矩阵B的缩放因子
        out_dtype,  # 输出数据类型
        bias=None,  # 偏置(可选)
    ):
        M = mat_a.shape[-2]  # 获取M维度
        N = mat_b.shape[-1]  # 获取N维度
        return mat_a.new_empty((M, N), dtype=out_dtype)  # 返回指定形状和类型的空张量


logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


class W8A8Int8Config(QuantizationConfig):  # W8A8 INT8量化配置类
    """Config class for W8A8 Quantization.
    # W8A8量化配置类

    - Weight: static, per-channel, symmetric
    # 权重: 静态、逐通道、对称
    - Activation: dynamic, per-token, symmetric
    # 激活: 动态、逐token、对称
    """

    def __init__(self, quant_config: Dict[str, Any] = {}):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.quant_description = quant_config  # 保存量化描述
        self.is_dynamic = quant_config.get("is_dynamic", False)  # 是否为动态量化
        ignore = cast(List[str], quant_config.get("ignore", []))  # 获取忽略层列表
        self.ignore = ignore if ignore is not None else []  # 保存忽略层列表
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})  # 获取打包模块映射
        self.packed_modules_mapping = (  # 保存打包模块映射
            packed_modules_mapping if packed_modules_mapping is not None else {}  # 确保不为None
        )

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:  # 获取支持的激活数据类型
        return [torch.float16, torch.bfloat16]  # 支持float16和bfloat16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低GPU计算能力要求
        return 75  # 需要计算能力7.5(SM75/Turing)

    @classmethod
    def get_name(self) -> str:  # 获取量化方法名称
        return "w8a8_int8"  # 返回名称

    @classmethod
    def get_config_filenames(cls) -> List[str]:  # 获取配置文件名列表
        filenames = []  # 空列表
        return filenames  # 返回空列表

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W8A8Int8Config:  # 从配置字典创建配置实例
        return cls(config)  # 返回配置实例

    def get_quant_method(  # 获取量化方法
        self,
        layer: torch.nn.Module,  # 网络层
        prefix: str,  # 层前缀
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase  # 导入线性基类
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE类

        if should_ignore_layer(  # 检查层是否应该被忽略
            prefix, ignore=self.ignore, fused_mapping=self.packed_modules_mapping  # 传入前缀、忽略列表和融合映射
        ):
            return UnquantizedLinearMethod()  # 返回未量化方法
        if isinstance(layer, LinearBase):  # 如果是线性层
            return W8A8Int8LinearMethod(self)  # 返回W8A8 INT8线性方法
        elif isinstance(layer, FusedMoE):  # 如果是融合MoE层
            return W8A8Int8MoEMethod(self)  # 返回W8A8 INT8 MoE方法
        return None  # 其他情况返回None

    def is_layer_skipped(  # 判断层是否被跳过
        self, prefix: str, fused_mapping: Mapping[str, List[str]] = MappingProxyType({})  # 前缀和融合映射
    ):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        # 改编自vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]  # 获取投影名称
        if proj_name in fused_mapping:  # 如果在融合映射中
            shard_prefixes = [  # 生成每个分片的前缀列表
                prefix.replace(proj_name, shard_proj_name)  # 将投影名称替换为分片名称
                for shard_proj_name in fused_mapping[proj_name]  # 遍历每个分片名称
            ]

            is_skipped = None  # 初始化跳过标志
            for shard_prefix in shard_prefixes:  # 遍历每个分片前缀
                is_shard_skipped = (  # 检查分片是否被跳过
                    self.quant_description[shard_prefix + ".weight"] == "FLOAT"  # 权重类型为FLOAT则跳过
                )

                if is_skipped is None:  # 如果是第一个分片
                    is_skipped = is_shard_skipped  # 设置初始值
                elif is_shard_skipped != is_skipped:  # 如果分片之间不一致
                    raise ValueError(  # 抛出错误
                        f"Detected some but not all shards of {prefix} "  # 检测到部分分片被跳过
                        "are quantized. All shards of fused layers "  # 融合层的所有分片
                        "to have the same precision."  # 必须具有相同的精度
                    )
        else:  # 非融合层
            is_skipped = self.quant_description[prefix + ".weight"] == "FLOAT"  # 直接检查权重类型

        assert is_skipped is not None  # 断言跳过标志已设置
        return is_skipped  # 返回是否跳过

    def get_scaled_act_names(self) -> List[str]:  # 获取需要缩放的激活名称
        return []  # 无需缩放的激活


class W8A8Int8LinearMethod(LinearMethodBase):  # W8A8 INT8线性层量化方法类

    def __init__(self, quantization_config: W8A8Int8Config):  # 初始化方法
        self.quantization_config = quantization_config  # 保存量化配置

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理
        if _is_cpu:  # 如果是CPU平台
            if _is_cpu_amx_available:  # 如果AMX可用
                _amx_process_weight_after_loading(layer, ["weight"])  # 对权重进行AMX打包处理
            elif _is_cpu_arm64:  # 如果是ARM64平台
                layer.weight = Parameter(layer.weight.data, requires_grad=False)  # 直接转为参数
            else:  # 其他CPU平台
                assert False, "W8A8Int8LinearMethod on CPU only works on AMX or Arm64"  # 不支持
        else:  # 非CPU平台(GPU)
            layer.weight = Parameter(layer.weight.t(), requires_grad=False)  # 转置权重并转为参数
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)  # 转缩放因子为参数

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

        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器
        self.logical_widths = output_partition_sizes  # 保存逻辑宽度

        weight = ModelWeightParameter(  # 创建模型权重参数
            data=torch.empty(  # 创建空张量
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8  # int8类型
            ),
            input_dim=1,  # 输入维度索引
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("weight", weight)  # 注册权重参数

        weight_scale = ChannelQuantScaleParameter(  # 创建逐通道量化缩放参数
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),  # 每通道1个缩放值
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("weight_scale", weight_scale)  # 注册权重缩放参数

    def apply(  # 应用INT8线性变换
        self,
        layer: torch.nn.Module,  # 目标网络层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置项(可选)
    ):
        if use_intel_amx_backend(layer) or _is_cpu_arm64:  # 如果使用AMX后端或ARM64
            return torch.ops.sgl_kernel.int8_scaled_mm_with_quant(  # 调用INT8缩放矩阵乘法(带量化)
                x,  # 输入
                layer.weight,  # 权重
                layer.weight_scale,  # 权重缩放因子
                bias,  # 偏置
                x.dtype,  # 输出数据类型
                True,  # is_vnni # 使用VNNI格式
            )
        x_q, x_scale = per_token_quant_int8(x)  # 对输入进行逐token INT8量化

        x_q_2d = x_q.view(-1, x_q.shape[-1])  # 将量化输入展平为2维
        x_scale_2d = x_scale.view(-1, x_scale.shape[-1])  # 将缩放因子展平为2维
        output_shape = [*x_q.shape[:-1], layer.weight.shape[1]]  # 计算输出形状

        output = int8_scaled_mm(  # 调用INT8缩放矩阵乘法
            x_q_2d,  # 量化输入(2维)
            layer.weight,  # 权重
            x_scale_2d,  # 输入缩放因子(2维)
            layer.weight_scale,  # 权重缩放因子
            out_dtype=x.dtype,  # 输出数据类型
            bias=bias,  # 偏置
        )

        return output.view(output_shape)  # 恢复原始形状并返回


class W8A8Int8MoEMethod(FusedMoEMethodBase):  # W8A8 INT8 MoE量化方法类
    """MoE method for INT8.
    # INT8的MoE方法。
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    # 支持加载具有静态权重缩放和动态/静态激活缩放的INT8检查点。
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

    def __init__(self, quant_config: W8A8Int8Config):  # 初始化方法
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

        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小

        # WEIGHTS
        # 权重
        w13_weight = torch.nn.Parameter(  # 创建w13权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量
                2 * intermediate_size_per_partition,  # 2倍中间层大小(门控)
                hidden_size,  # 隐藏层大小
                dtype=torch.int8,  # int8类型
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
                dtype=torch.int8,  # int8类型
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
        if _is_cpu_amx_available:  # 如果CPU AMX可用
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])  # 对权重进行AMX打包处理
        else:  # AMX不可用
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
            use_int8_w8a8=True,  # 使用INT8 W8A8
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
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入类

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取topk输出

        if use_intel_amx_backend(layer) or _is_cpu_arm64:  # 如果使用AMX后端或ARM64
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu  # 导入CPU topk权重应用函数

            topk_weights, topk_ids, _ = topk_output  # 解包topk输出
            topk_ids = topk_ids.int()  # 转为int类型
            x, topk_weights = apply_topk_weights_cpu(  # 在CPU上应用topk权重
                self.moe_runner_config.apply_router_weight_on_input, topk_weights, x  # 传入配置和权重
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(  # 调用CPU融合专家计算
                x,  # 输入
                layer.w13_weight,  # w13权重
                layer.w2_weight,  # w2权重
                topk_weights,  # topk权重
                topk_ids,  # topk专家ID
                False,  # inplace See [Note] inplace should be False in fused_experts. # 原地操作 见[注] fused_experts中inplace应为False
                CPUQuantMethod.INT8_W8A8,  # CPU量化方法: INT8 W8A8
                layer.w13_weight_scale,  # w1_scale # w1缩放因子
                layer.w2_weight_scale,  # w2_scale # w2缩放因子
                None,  # w1_zp # w1零点
                None,  # w2_zp # w2零点
                None,  # block_size # 块大小
                None,  # w1 bias # w1偏置
                None,  # w3 bias # w3偏置
                None,  # alpha # alpha参数
                None,  # limit # 限制值
                True,  # is_vnni # 使用VNNI格式
            )
            return StandardCombineInput(hidden_states=output)  # 返回标准合并输入

        quant_info = self.get_triton_quant_info(layer)  # 获取Triton量化信息
        return self.runner.run(dispatch_output, quant_info)  # 使用Triton运行器执行MoE
