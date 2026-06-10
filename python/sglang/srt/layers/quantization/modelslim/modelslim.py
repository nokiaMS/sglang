# 文件说明：ModelSlim量化框架的配置和方法实现，支持NPU上的W4A4、W8A8等多种量化方案，包含线性层和MoE层的量化方法
from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 导入日志模块
from types import MappingProxyType  # 导入映射代理类型（不可变映射）
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union, cast  # 导入各种类型注解

import torch  # 导入PyTorch深度学习框架

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (  # 导入NPU线性方法基类
    _NPULinearMethodBase,
)
from sglang.srt.layers.quantization.base_config import (  # 导入量化配置基类
    FusedMoEMethodBase,  # MoE方法基类
    QuantizationConfig,  # 量化配置基类
)
from sglang.srt.layers.quantization.modelslim.schemes import (  # 导入ModelSlim量化方案
    ModelSlimW4A4Int4,  # W4A4 INT4量化方案
    ModelSlimW4A4Int4MoE,  # W4A4 INT4 MoE量化方案
    ModelSlimW4A8Int8MoE,  # W4A8 INT8 MoE量化方案
    ModelSlimW8A8Int8,  # W8A8 INT8量化方案
    ModelSlimW8A8Int8MoE,  # W8A8 INT8 MoE量化方案
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 导入未量化线性方法
from sglang.srt.utils import apply_module_patch  # 导入模块补丁工具函数

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置类
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入MoE分发器相关类型
        CombineInput,  # 合并输入类型
        StandardDispatchOutput,  # 标准分发输出类型
    )
    from sglang.srt.layers.quantization.base_config import QuantizeMethodBase  # 导入量化方法基类
    from sglang.srt.layers.quantization.modelslim.schemes import (  # 导入ModelSlim方案类型
        ModelSlimLinearScheme,  # ModelSlim线性方案类型
        ModelSlimMoEScheme,  # ModelSlim MoE方案类型
    )

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


# func refers to RMSNorm.__init__  # func指向RMSNorm.__init__方法
def npu_wrapper_rmsnorm_init(func):  # NPU RMSNorm初始化包装器函数
    def init(self, hidden_size: int, **extra_args) -> None:  # 包装后的初始化方法
        func(self, hidden_size, **extra_args)  # 调用原始初始化方法
        self.ignore_anti = True  # 设置忽略反量化标志
        # The Ascend w8a8_int8 quantization requires adding a bias in rmsnorm  # Ascend w8a8_int8量化需要在rmsnorm中添加偏置
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)  # 创建零偏置参数

    return init  # 返回包装后的初始化方法


# func refers to RMSNorm.forward_oot  # func指向RMSNorm.forward_oot方法
def npu_wrapper_rmsnorm_forward(func):  # NPU RMSNorm前向传播包装器函数
    def _rmsnorm_forward_oot(  # 包装后的前向传播方法
        self,
        x: torch.Tensor,  # 输入张量
        residual: Optional[torch.Tensor] = None,  # 可选残差张量
        post_residual_addition: Optional[torch.Tensor] = None,  # 可选残差后加法张量
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 返回张量或张量元组
        if not x.is_contiguous():  # 如果输入不是连续张量
            x = x.contiguous()  # 转为连续张量
        if residual is not None:  # 如果有残差输入
            if post_residual_addition is not None:  # 如果有残差后加法
                residual = residual + post_residual_addition  # 将残差后加法加到残差上
            from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias  # 导入带偏置的RMSNorm加法核

            out, residual_out = add_rmsnorm_bias(  # 调用带偏置的RMSNorm加法核
                x,  # 输入张量
                residual,  # 残差张量
                self.weight.data,  # RMSNorm权重
                self.bias,  # 偏置
                self.variance_epsilon,  # 方差epsilon
            )
            return out.to(x.dtype), residual_out  # 返回输出和残差

        out = torch.ops.npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]  # 调用NPU RMSNorm算子
        out = out + self.bias  # 加上偏置
        return out.to(x.dtype)  # 返回转换为输入数据类型的输出

    return _rmsnorm_forward_oot  # 返回包装后的前向传播方法


class ModelSlimConfig(QuantizationConfig):  # ModelSlim量化配置类，继承自量化配置基类
    """
    Config class for ModelSlim Quantization, a NPU-specific quantization type.
    ModelSlim量化配置类，一种NPU专用的量化类型。
    """

    def __init__(self, quant_config: Dict[str, Any] = {}):  # 初始化方法，接收量化配置字典
        super().__init__()  # 调用父类初始化
        self.quant_description = quant_config  # 保存量化描述信息
        ignore = cast(List[str], quant_config.get("ignore", []))  # 获取忽略的层列表
        self.ignore = ignore if ignore is not None else []  # 如果忽略列表为None则设为空列表
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})  # 获取打包模块映射
        self.packed_modules_mapping = (  # 保存打包模块映射
            packed_modules_mapping if packed_modules_mapping is not None else {}  # 如果映射为None则设为空字典
        )

        for name in self.quant_description.keys():  # 遍历量化描述的所有键
            if "norm.bias" in name:  # 如果键名包含"norm.bias"
                apply_module_patch(  # 对RMSNorm的__init__方法应用补丁
                    "sglang.srt.layers.layernorm.RMSNorm",  # RMSNorm类路径
                    "__init__",  # 方法名
                    [npu_wrapper_rmsnorm_init],  # 使用NPU RMSNorm初始化包装器
                )
                apply_module_patch(  # 对RMSNorm的forward_npu方法应用补丁
                    "sglang.srt.layers.layernorm.RMSNorm",  # RMSNorm类路径
                    "forward_npu",  # 方法名
                    [npu_wrapper_rmsnorm_forward],  # 使用NPU RMSNorm前向传播包装器
                )

    def update_packed_modules_mapping(self, mapping: Dict[str, List[str]]) -> None:  # 更新打包模块映射
        self.packed_modules_mapping.update(mapping)  # 用新映射更新现有映射

    def get_linear_method(self) -> ModelSlimLinearMethod:  # 获取线性量化方法
        return ModelSlimLinearMethod(self)  # 返回ModelSlim线性方法实例

    @classmethod  # 类方法装饰器
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:  # 获取支持的激活值数据类型
        return [torch.int8, torch.float16, torch.bfloat16]  # 支持INT8、FP16和BF16

    @classmethod  # 类方法装饰器
    def get_min_capability(cls) -> int:  # 获取最低硬件能力要求
        return 0  # 最低能力要求为0（无限制）

    @classmethod  # 类方法装饰器
    def get_name(cls) -> str:  # 获取量化方法名称
        return "modelslim"  # 返回名称"modelslim"

    @classmethod  # 类方法装饰器
    def get_config_filenames(cls) -> List[str]:  # 获取配置文件名列表
        filenames = ["quant_model_description.json"]  # 配置文件名为quant_model_description.json
        return filenames  # 返回文件名列表

    @classmethod  # 类方法装饰器
    def from_config(cls, config: Dict[str, Any]) -> ModelSlimConfig:  # 从配置字典创建ModelSlimConfig实例
        return cls(config)  # 用配置字典创建实例并返回

    def get_quant_method(  # 获取量化方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        prefix: str,  # 层名称前缀
    ) -> Optional[QuantizeMethodBase]:  # 返回可选的量化方法实例
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE层

        if isinstance(layer, LinearBase):  # 如果是线性层
            # TODO: we should remove this code and switch to the packed_modules_mapping declared inside the modeling files  # TODO：我们应该移除此代码，转而使用模型文件中声明的packed_modules_mapping
            key = "model"  # 默认键为"model"
            if "vision_model" in prefix:  # 如果前缀包含"vision_model"
                key = "vision_model"  # 键设为"vision_model"
            elif "visual" in prefix:  # 如果前缀包含"visual"
                key = "visual"  # 键设为"visual"
            if "vision_tower" in prefix or "mm_projector" in prefix:  # 如果前缀包含视觉塔或多模态投影器
                prefix = prefix.replace(r"attn.qkv_proj", r"wqkv")  # 替换注意力qkv投影名
                prefix = prefix.replace(r"attn.proj", r"wo")  # 替换注意力输出投影名
            packed_modules_mapping_subset = self.packed_modules_mapping.get(key, {})  # 获取对应键的打包模块映射子集
            prefix_in_quant_config = prefix  # 量化配置中的前缀
            proj_name = prefix.split(".")[-1]  # 获取前缀最后一部分作为投影名
            if proj_name in packed_modules_mapping_subset:  # 如果投影名在打包模块映射中
                prefix_in_quant_config = prefix.replace(  # 替换前缀中的投影名
                    proj_name, packed_modules_mapping_subset[proj_name][0]  # 使用映射中的第一个名
                )
            if self.is_layer_skipped(  # 如果该层在子集映射中被跳过
                prefix, packed_modules_mapping_subset
            ) or self.is_layer_skipped(prefix, self.packed_modules_mapping):  # 或在完整映射中被跳过
                return UnquantizedLinearMethod()  # 返回未量化线性方法
            layer.scheme = self.get_linear_scheme(layer, prefix_in_quant_config)  # 设置层的量化方案
            return ModelSlimLinearMethod(self)  # 返回ModelSlim线性方法实例
        elif isinstance(layer, FusedMoE):  # 如果是MoE层
            layer.scheme = self.get_moe_scheme(layer, prefix)  # 设置层的MoE量化方案
            return ModelSlimFusedMoEMethod(self)  # 返回ModelSlim MoE方法实例
        return None  # 其他层返回None

    def get_linear_scheme(  # 获取线性层量化方案
        self, layer: torch.nn.Module, prefix: Optional[str] = None  # 层和可选前缀
    ) -> Optional[ModelSlimLinearScheme]:  # 返回可选的ModelSlim线性方案
        """
        get_scheme method adjusted for modelslim, taken from
        python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py
        为modelslim调整的get_scheme方法，参考自compressed_tensors.py
        """

        linear_quant_schemes = [  # 线性量化方案列表
            ("W4A4_DYNAMIC", ModelSlimW4A4Int4),  # W4A4动态量化方案
            ("W8A8", ModelSlimW8A8Int8),  # W8A8静态量化方案
            ("W8A8_DYNAMIC", ModelSlimW8A8Int8),  # W8A8动态量化方案
        ]

        quant_schemes = [self.quant_description.get(prefix + ".weight", "")]  # 从量化描述中获取当前层的量化方案

        for scheme_name, scheme_class in linear_quant_schemes:  # 遍历所有线性量化方案
            if any(s == scheme_name for s in quant_schemes):  # 如果当前层匹配任一方案
                logger.info_once(f"Using {scheme_class.__name__}")  # 记录使用的方案名称（仅一次）
                return scheme_class(quant_config=self.quant_description, prefix=prefix)  # 返回方案实例

        logger.warning(  # 记录警告
            f"Unsupported Linear modelslim scheme: "  # 不支持的ModelSlim线性方案
            f"{quant_schemes} in layer: {prefix}"  # 在层中的方案
        )
        return None  # 返回None表示无匹配方案

    def get_moe_scheme(  # 获取MoE层量化方案
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        prefix: str,  # 层名称前缀
    ) -> Optional[ModelSlimMoEScheme]:  # 返回可选的ModelSlim MoE方案
        moe_quant_schemes = [  # MoE量化方案列表
            ("W4A4_DYNAMIC", ModelSlimW4A4Int4MoE),  # W4A4动态量化MoE方案
            ("W4A8_DYNAMIC", ModelSlimW4A8Int8MoE),  # W4A8动态量化MoE方案
            ("W8A8_DYNAMIC", ModelSlimW8A8Int8MoE),  # W8A8动态量化MoE方案
        ]

        moe_weight_suffixes = [".0.gate_proj.weight", ".0.w2.weight"]  # MoE权重后缀列表
        quant_schemes = [  # 获取当前层的量化方案
            self.quant_description.get(prefix + suffix, "")  # 从量化描述中获取
            for suffix in moe_weight_suffixes  # 遍历MoE权重后缀
        ]

        for scheme_name, scheme_class in moe_quant_schemes:  # 遍历所有MoE量化方案
            if any(s == scheme_name for s in quant_schemes):  # 如果当前层匹配任一方案
                logger.info_once(f"Using {scheme_class.__name__}")  # 记录使用的方案名称（仅一次）
                return scheme_class(self)  # 返回方案实例

        logger.warning(  # 记录警告
            f"Unsupported FusedMoe modelslim scheme: "  # 不支持的ModelSlim MoE方案
            f"{quant_schemes} in layer: {prefix}"  # 在层中的方案
        )
        return None  # 返回None表示无匹配方案

    def is_layer_skipped(  # 判断层是否应跳过量化
        self, prefix: str, fused_mapping: Mapping[str, List[str]] = MappingProxyType({})  # 前缀和融合映射
    ):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped  # 适配自vLLM的is_layer_skipped方法
        proj_name = prefix.split(".")[-1]  # 获取前缀最后一部分作为投影名
        if proj_name in fused_mapping:  # 如果投影名在融合映射中
            shard_prefixes = [  # 构建分片前缀列表
                prefix.replace(proj_name, shard_proj_name)  # 替换投影名为分片投影名
                for shard_proj_name in fused_mapping[proj_name]  # 遍历融合映射中的分片投影名
            ]

            is_skipped = None  # 初始化跳过标志
            for shard_prefix in shard_prefixes:  # 遍历分片前缀
                is_shard_skipped = (  # 判断分片是否跳过
                    self.quant_description.get(shard_prefix + ".weight", "") == "FLOAT"  # 如果量化描述中标记为FLOAT则跳过
                )

                if is_skipped is None:  # 如果跳过标志尚未设置
                    is_skipped = is_shard_skipped  # 设置为当前分片的跳过状态
                elif is_shard_skipped != is_skipped:  # 如果当前分片的跳过状态与之前不一致
                    raise ValueError(  # 抛出值错误
                        f"Detected some but not all shards of {prefix} "  # 检测到部分分片被量化
                        "are quantized. All shards of fused layers "  # 融合层的所有分片
                        "to have the same precision."  # 必须具有相同的精度
                    )
        else:  # 如果投影名不在融合映射中
            is_skipped = self.quant_description.get(prefix + ".weight", "") == "FLOAT"  # 直接判断该层是否标记为FLOAT

        assert is_skipped is not None  # 断言跳过标志已设置
        return is_skipped  # 返回跳过标志

    def get_scaled_act_names(self) -> List[str]:  # 获取需要缩放的激活名列表
        return []  # 返回空列表（无缩放激活）


class ModelSlimLinearMethod(_NPULinearMethodBase):  # ModelSlim线性量化方法类，继承自NPU线性方法基类

    def __init__(self, quantization_config: ModelSlimConfig):  # 初始化方法，接收ModelSlim配置
        self.quantization_config = quantization_config  # 保存量化配置

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的处理方法
        layer.scheme.process_weights_after_loading(layer)  # 调用层的量化方案进行权重后处理

    def create_weights(  # 创建线性层量化权重参数
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: List[int],  # 输出分区大小列表
        input_size: int,  # 完整输入大小
        output_size: int,  # 完整输出大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        """
        Use the ModelSlimLinearScheme associated with the layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        使用与层关联的ModelSlimLinearScheme创建层所需的参数。参数详情参见LinearMethodBase
        """
        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器
        layer.scheme.create_weights(  # 调用层的量化方案创建权重
            layer=layer,  # 目标层
            input_size=input_size,  # 完整输入大小
            input_size_per_partition=input_size_per_partition,  # 每个分区的输入大小
            output_partition_sizes=output_partition_sizes,  # 输出分区大小列表
            output_size=output_size,  # 完整输出大小
            params_dtype=params_dtype,  # 参数数据类型
            weight_loader=weight_loader,  # 权重加载器
        )

    def apply(  # 应用量化权重进行前向计算
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 可选偏置张量
    ):
        """
        Use the output of create_weights and the ModelSlimLinearScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details
        使用create_weights的输出和与层关联的ModelSlimLinearScheme对层输入执行前向传播。参数详情参见LinearMethodBase

        """

        scheme = layer.scheme  # 获取层的量化方案
        if scheme is None:  # 如果方案为None
            raise ValueError("A scheme must be defined for each layer")  # 抛出值错误，每层必须定义方案
        return scheme.apply_weights(layer, x, bias=bias)  # 调用方案的apply_weights方法执行前向计算


class ModelSlimFusedMoEMethod(FusedMoEMethodBase):  # ModelSlim融合MoE量化方法类，继承自MoE方法基类

    def __init__(self, quantization_config: ModelSlimConfig):  # 初始化方法，接收ModelSlim配置
        self.quantization_config = quantization_config  # 保存量化配置

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的处理方法
        layer.scheme.process_weights_after_loading(layer)  # 调用层的量化方案进行权重后处理

    def create_weights(  # 创建MoE量化权重参数
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        """
        Use the ModelSlimMoEScheme associated with the layer to create
        the necessary parameters for the layer. See FusedMoEMethodBase for param
        details
        使用与层关联的ModelSlimMoEScheme创建层所需的参数。参数详情参见FusedMoEMethodBase
        """
        layer.scheme.create_weights(  # 调用层的MoE量化方案创建权重
            layer=layer,  # 目标层
            num_experts=num_experts,  # 专家数量
            hidden_size=hidden_size,  # 隐藏层大小
            intermediate_size_per_partition=intermediate_size_per_partition,  # 每个分区的中间层大小
            params_dtype=params_dtype,  # 参数数据类型
            **extra_weight_attrs,  # 额外权重属性
        )

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 目标层和MoE运行器配置
    ):
        return layer.scheme.create_moe_runner(layer, moe_runner_config)  # 调用层的量化方案创建MoE运行器

    def apply(  # 应用量化权重进行MoE前向计算
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:  # 返回合并输入
        """
        Use the output of create_weights and the ModelSlimMoEScheme
        associated with the layer to apply the forward pass with the
        layer input.  See FusedMoEMethodBase for param details
        使用create_weights的输出和与层关联的ModelSlimMoEScheme对层输入执行前向传播。参数详情参见FusedMoEMethodBase

        """
        scheme = layer.scheme  # 获取层的量化方案
        if scheme is None:  # 如果方案为None
            raise ValueError("A scheme must be defined for each layer")  # 抛出值错误，每层必须定义方案
        return scheme.apply_weights(layer, dispatch_output)  # 调用方案的apply_weights方法执行MoE前向计算

    def apply_without_routing_weights(  # 不使用路由权重的应用方法
        self,
        layer,  # 目标层
        hidden_states,  # 隐藏状态
        hidden_states_scale,  # 隐藏状态缩放因子
        group_list_type,  # 分组列表类型
        group_list,  # 分组列表
        output_dtype,  # 输出数据类型
    ):
        return layer.scheme.apply_without_routing_weights(  # 调用方案的不使用路由权重方法
            layer,  # 目标层
            hidden_states,  # 隐藏状态
            hidden_states_scale,  # 隐藏状态缩放因子
            group_list_type,  # 分组列表类型
            group_list,  # 分组列表
            output_dtype,  # 输出数据类型
        )
