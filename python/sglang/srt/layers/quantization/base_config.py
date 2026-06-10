# 量化配置基类模块
# 本文件定义了量化方法的基础抽象类，包括 QuantizeMethodBase（量化方法基类）、
# LinearMethodBase（线性层量化方法基类）、FusedMoEMethodBase（融合 MoE 量化方法基类）
# 以及 QuantizationConfig（量化配置基类），为各种量化方案提供统一的接口规范。

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/base_config.py
from __future__ import annotations  # 启用延迟注解评估

import inspect  # 导入检查模块，用于检查类属性
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type  # 导入类型注解工具

import torch  # 导入 PyTorch 深度学习框架
from torch import nn  # 导入 PyTorch 神经网络模块

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig  # MoE 运行器配置
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo  # Triton MoE 量化信息
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput  # MoE 令牌分发器类型
    from sglang.srt.models.utils import WeightsMapper  # 权重映射器


class QuantizeMethodBase(ABC):  # 量化方法基类，所有量化方法的抽象基类
    """Base class for different quantized methods."""  # 不同量化方法的基类

    def create_weights(  # 创建权重方法
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs  # 参数：层、权重参数和额外属性
    ):
        """Create weights for a layer.  # 为层创建权重

        The weights will be set as attributes of the layer."""  # 权重将被设置为层的属性
        raise NotImplementedError()  # 需子类实现

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:  # 应用量化方法进行前向计算
        """Apply the weights in layer to the input tensor.  # 将层中的权重应用到输入张量

        Expects create_weights to have been called before on the layer."""  # 要求在此之前已调用 create_weights
        raise NotImplementedError()  # 需子类实现

    def process_weights_after_loading(self, layer: nn.Module) -> None:  # 权重加载后的处理方法
        """Process the weight after loading.  # 加载后处理权重

        This can be used for example, to transpose weights for computation.  # 例如，可以用于转置权重以便计算
        """
        return  # 默认无操作


class LinearMethodBase(QuantizeMethodBase):  # 线性层量化方法基类
    """Base class for different (maybe quantized) linear methods."""  # 不同（可能量化）线性方法的基类

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
        """Create weights for a linear layer.  # 为线性层创建权重
           The weights will be set as attributes of the layer.  # 权重将被设置为层的属性

        Args:  # 参数说明
            layer: The layer that is using the LinearMethodBase factory.  # 使用 LinearMethodBase 工厂的层
            input_size_per_partition: Size of the weight input dim on rank X.  # 秩 X 上权重输入维度的大小
            output_partition_sizes: Sizes of the output dim of each logical  # 各逻辑权重输出维度的大小
                weight on rank X. E.g., output_partition_sizes for QKVLinear  # 秩 X 上的权重。例如 QKVLinear 的
                is a list contains the width of Wq, Wk, Wv on rank X.  # 是包含秩 X 上 Wq, Wk, Wv 宽度的列表
            input_size: Size of the input dim of the weight across all ranks.  # 跨所有秩的权重输入维度大小
            output_size: Size of the output dim of the weight across all ranks.  # 跨所有秩的权重输出维度大小
            params_dtype: Datatype of the parameters.  # 参数的数据类型
        """
        raise NotImplementedError()  # 需子类实现

    @abstractmethod
    def apply(  # 应用线性层量化方法
        self,
        layer: torch.nn.Module,  # 目标层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置张量，可选
    ) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.  # 将层中的权重应用到输入张量
        Expects create_weights to have been called before on the layer."""  # 要求在此之前已调用 create_weights
        raise NotImplementedError()  # 需子类实现


class FusedMoEMethodBase(QuantizeMethodBase):  # 融合 MoE 量化方法基类

    def create_weights(  # 创建 MoE 层权重
        self,
        layer: torch.nn.Module,  # 目标层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 当前分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        raise NotImplementedError  # 需子类实现

    def create_moe_runner(  # 创建 MoE 运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 目标层和 MoE 运行器配置
    ):
        raise NotImplementedError  # 需子类实现

    @abstractmethod
    def apply(  # 应用 MoE 量化方法
        self,
        layer: torch.nn.Module,  # 目标层
        dispatch_output: DispatchOutput,  # 分发输出
    ) -> CombineInput:
        raise NotImplementedError  # 需子类实现

    def get_triton_quant_info(self, layer: torch.nn.Module) -> "TritonMoeQuantInfo":  # 获取 Triton MoE 量化信息
        """Return a ``TritonMoeQuantInfo`` describing the quantisation state  # 返回描述层上量化状态的 TritonMoeQuantInfo
        stored on *layer*.  # 存储在 *layer* 上的

        The LoRA MoE runner calls this so that ``invoke_fused_moe_kernel``  # LoRA MoE 运行器调用此方法，以便 invoke_fused_moe_kernel
        receives the correct flags / scales / block-shape for the base  # 接收基础权重的正确标志/缩放/块形状
        weights.  Each quantisation method must override this with the  # 每种量化方法必须使用
        same construction it already uses inside ``apply()``.  # 与 apply() 中相同的构造来覆盖此方法
        """
        raise NotImplementedError(  # 需子类实现
            f"{type(self).__name__} must implement get_triton_quant_info()"  # 类名必须实现 get_triton_quant_info
        )


class QuantizationConfig(ABC):  # 量化配置基类，所有量化配置的抽象基类
    """Base class for quantization configs."""  # 量化配置的基类

    def __init__(self):  # 初始化方法
        super().__init__()  # 调用父类初始化
        # mapping is updated by models as they initialize  # 映射在模型初始化时更新
        self.packed_modules_mapping: Dict[str, List[str]] = dict()  # 打包模块映射字典

    def update_packed_modules_mapping(self, mapping: Dict[str, List[str]]) -> None:  # 更新打包模块映射
        self.packed_modules_mapping = mapping  # 设置映射

    @abstractmethod
    def get_name(self) -> str:  # 获取量化方法名称
        """Name of the quantization method."""  # 量化方法的名称
        raise NotImplementedError()  # 需子类实现

    @abstractmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:  # 获取支持的激活数据类型列表
        """List of supported activation dtypes."""  # 支持的激活数据类型列表
        raise NotImplementedError()  # 需子类实现

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:  # 获取最低 GPU 计算能力要求
        """Minimum GPU capability to support the quantization method.  # 支持该量化方法的最低 GPU 计算能力

        E.g., 70 for Volta, 75 for Turing, 80 for Ampere.  # 例如，70 对应 Volta，75 对应 Turing，80 对应 Ampere
        This requirement is due to the custom CUDA kernels used by the  # 此要求是因为量化方法使用的自定义 CUDA 内核
        quantization method.  # 量化方法
        """
        raise NotImplementedError()  # 需子类实现

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> List[str]:  # 获取配置文件名列表
        """List of filenames to search for in the model directory."""  # 在模型目录中搜索的文件名列表
        raise NotImplementedError()  # 需子类实现

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":  # 从配置字典创建量化配置对象
        """Create a config class from the model's quantization config."""  # 从模型的量化配置创建配置类
        raise NotImplementedError()  # 需子类实现

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:  # 覆盖量化方法检测
        """
        Detects if this quantization method can support a given checkpoint  # 检测此量化方法是否支持给定的检查点格式
        format by overriding the user specified quantization method --  # 通过覆盖用户指定的量化方法
        this method should only be overwritten by subclasses in exceptional  # 此方法仅应在特殊情况下被子类覆盖
        circumstances  # 情况
        """
        return None  # 默认不覆盖

    @classmethod
    def _modelopt_override_quantization_method(  # ModelOpt 量化方法覆盖逻辑
        cls, hf_quant_config, user_quant  # HuggingFace 量化配置和用户指定量化方法
    ) -> Optional[str]:
        """Shared ModelOpt quantization method override logic."""  # 共享的 ModelOpt 量化方法覆盖逻辑
        if hf_quant_config is None:  # 如果没有 HuggingFace 量化配置
            return None  # 返回 None

        # Check if this is a ModelOpt config  # 检查是否为 ModelOpt 配置
        quant_algo = hf_quant_config.get("quant_algo", "").upper()  # 获取量化算法并转为大写

        # If user specified generic "modelopt", auto-detect the specific method  # 如果用户指定了通用的 "modelopt"，自动检测具体方法
        if user_quant == "modelopt":  # 如果用户指定了 modelopt
            if "FP8" in quant_algo:  # 如果量化算法包含 FP8
                return "modelopt_fp8"  # 返回 modelopt_fp8
            elif "NVFP4" in quant_algo or "FP4" in quant_algo:  # 如果包含 NVFP4 或 FP4
                return "modelopt_fp4"  # 返回 modelopt_fp4

        # The hf_quant_config may be a parsed quant config, so we need to check the  # hf_quant_config 可能是已解析的量化配置，因此需要检查
        # quant_method.  # quant_method 字段
        if hf_quant_config.get("quant_method", "") == "modelopt_fp8":  # 如果量化方法为 modelopt_fp8
            return "modelopt_fp8"  # 返回 modelopt_fp8
        elif hf_quant_config.get("quant_method", "") == "modelopt_fp4":  # 如果量化方法为 modelopt_fp4
            return "modelopt_fp4"  # 返回 modelopt_fp4

        return None  # 不匹配则返回 None

    @staticmethod
    def get_from_keys(config: Dict[str, Any], keys: List[str]) -> Any:  # 从配置字典中按键列表获取值
        """Get a value from the model's quantization config."""  # 从模型量化配置中获取值
        for key in keys:  # 遍历键列表
            if key in config:  # 如果键存在于配置中
                return config[key]  # 返回对应的值
        raise ValueError(  # 键不存在时抛出错误
            f"Cannot find any of {keys} in the model's " "quantization config."  # 在模型量化配置中找不到任何指定键
        )

    @staticmethod
    def get_from_keys_or(config: Dict[str, Any], keys: List[str], default: Any) -> Any:  # 从配置字典中按键列表获取值，支持默认值
        """Get a optional value from the model's quantization config."""  # 从模型量化配置中获取可选值
        try:  # 尝试获取值
            return QuantizationConfig.get_from_keys(config, keys)  # 调用 get_from_keys
        except ValueError:  # 如果键不存在
            return default  # 返回默认值

    @abstractmethod
    def get_quant_method(  # 获取适用于指定层的量化方法
        self, layer: torch.nn.Module, prefix: str  # 目标层和层前缀
    ) -> Optional[QuantizeMethodBase]:
        """Get the quantize method to use for the quantized layer.  # 获取量化层要使用的量化方法

        Args:  # 参数说明
            layer: The layer for the quant method.  # 需要量化的层
            prefix: The full name of the layer in the state dict  # 层在状态字典中的完整名称
        Returns:  # 返回值
            The quantize method. None if the given layer doesn't support quant  # 量化方法。如果层不支持量化则返回 None
            method.  # 方法
        """
        raise NotImplementedError()  # 需子类实现

    @abstractmethod
    def get_scaled_act_names(self) -> List[str]:  # 获取需要后缩放的激活函数名列表
        """Returns the activation function names that should be post-scaled.  # 返回需要后缩放的激活函数名

        For now, this is only used by AWQ.  # 目前仅 AWQ 使用
        """
        raise NotImplementedError()  # 需子类实现

    def apply_weight_name_mapper(  # 应用权重名称映射器
        self, hf_to_sglang_mapper: "WeightsMapper"  # HuggingFace 到 SGLang 的权重映射器
    ):  # noqa: B027
        """
        Interface for models to update module names referenced in  # 模型更新量化配置中引用的模块名称的接口
        quantization configs in order to reflect the sglang model structure  # 以反映 sglang 模型结构
        :param hf_to_sglang_mapper: maps from hf model structure (the assumed  # 参数: 从 HuggingFace 模型结构（假设的
            structure of the qconfig) to sglang model structure  # 量化配置结构）映射到 sglang 模型结构
        """
        pass  # 默认无操作


def method_has_implemented_embedding(method_class: Type[QuantizeMethodBase]) -> bool:  # 检查量化方法是否实现了嵌入方法
    """
    Not all quant methods have embedding implemented, so we need to check that  # 并非所有量化方法都实现了嵌入，因此需要检查
    it exists for our given method. We check this by making sure the function  # 给定方法是否存在嵌入实现。通过确认函数
    has been changed from the base implementation.  # 已从基类实现中被修改来检查
    """
    base_embedding = inspect.getattr_static(QuantizeMethodBase, "embedding", None)  # 获取基类的 embedding 属性
    class_embedding = inspect.getattr_static(method_class, "embedding", None)  # 获取子类的 embedding 属性

    return class_embedding is not None and class_embedding is not base_embedding  # 返回子类是否有不同于基类的 embedding 实现
