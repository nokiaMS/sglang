# Compressed Tensors量化方案实现
# 本文件实现了基于Neural Magic compressed-tensors格式的量化配置和方法，支持多种量化方案（W8A8、W4A16、FP8、FP4等）
# 适配自vLLM项目的compressed-tensors量化实现
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
from contextlib import suppress  # 导入上下文管理器工具
from typing import (  # 导入类型提示工具
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    cast,
)

import torch  # 导入PyTorch库
from compressed_tensors.config import (  # 从compressed_tensors库导入压缩配置类
    CompressionFormat,
    SparsityCompressionConfig,
    SparsityStructure,
)
from compressed_tensors.quantization import (  # 从compressed_tensors库导入量化相关类
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from pydantic import BaseModel  # 导入Pydantic基础模型

from sglang.srt.layers.moe import MoeRunnerConfig, get_moe_runner_backend  # 导入MoE运行器相关组件
from sglang.srt.layers.quantization.base_config import (  # 导入量化基础配置类
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # 导入各种压缩张量量化方案
    WNA16_SUPPORTED_BITS,
    CompressedTensorsLinearScheme,
    CompressedTensorsMoEScheme,
    CompressedTensorsMxInt4MoE,
    CompressedTensorsW4A4Fp4,
    CompressedTensorsW4A4Nvfp4MoE,
    CompressedTensorsW8A8Fp8,
    CompressedTensorsW8A8Fp8MoE,
    CompressedTensorsW8A8Int8,
    CompressedTensorsW8A16Fp8,
    CompressedTensorsWNA16,
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
    NPUCompressedTensorsW4A8Int8DynamicMoE,
    NPUCompressedTensorsW4A16Int4DynamicMoE,
    NPUCompressedTensorsW8A8Int8,
    NPUCompressedTensorsW8A8Int8DynamicMoE,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (  # 导入压缩张量工具函数
    find_matched_target,
    is_activation_quantization_format,
    should_ignore_layer,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod  # 导入FP8线性方法
from sglang.srt.layers.quantization.unquant import (  # 导入未量化方法
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu  # 导入硬件平台检测工具

_is_cuda = is_cuda()  # 检测是否为CUDA平台
_is_npu = is_npu()  # 检测是否为NPU平台
_is_hip = is_hip()  # 检测是否为HIP(ROCm)平台

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入MoE分发器类型
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.models.utils import WeightsMapper  # 导入权重映射器类型

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

__all__ = ["CompressedTensorsLinearMethod"]  # 模块公开接口

SPARSITY_CONFIG_NAME: Literal["sparsity_config"] = "sparsity_config"  # 稀疏配置名称常量
QUANTIZATION_SCHEME_MAP_TYPE = Dict[str, Optional[Dict[str, QuantizationArgs]]]  # 量化方案映射类型别名


class DeviceCapability(NamedTuple):  # 设备能力命名元组，表示GPU计算能力版本
    major: int  # 主版本号
    minor: int  # 次版本号

    def as_version_str(self) -> str:  # 将设备能力转换为版本字符串
        return f"{self.major}.{self.minor}"  # 返回"主版本.次版本"格式字符串

    def to_int(self) -> int:  # 将设备能力转换为整数表示
        """
        Express device capability as an integer ``<major><minor>``.
        将设备能力表示为整数``<主版本><次版本>``。

        It is assumed that the minor version is always a single digit.
        假设次版本号总是单个数字。
        """
        assert 0 <= self.minor < 10  # 断言次版本号在0-9之间
        return self.major * 10 + self.minor  # 返回整数形式的设备能力


class CompressedTensorsConfig(QuantizationConfig):  # Compressed Tensors量化配置类，继承自QuantizationConfig
    def __init__(  # 初始化方法
        self,
        target_scheme_map: Dict[str, Any],  # 目标层到量化方案的映射
        ignore: List[str],  # 忽略量化的层名列表
        quant_format: str,  # 量化格式
        sparsity_scheme_map: Dict[str, SparsityCompressionConfig],  # 目标层到稀疏方案的映射
        sparsity_ignore_list: List[str],  # 忽略稀疏的层名列表
        kv_cache_scheme: Optional[Dict[str, Any]] = None,  # KV缓存量化方案
        config: Optional[Dict[str, Any]] = None,  # 原始配置字典
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,  # 打包模块映射
        linear_fp8_config: Optional[Any] = None,  # 线性层FP8配置（混合量化场景）
    ):
        super().__init__()  # 调用父类初始化
        self.ignore = ignore  # 保存忽略列表
        self.quant_format = quant_format  # 保存量化格式
        # Map from [target -> scheme]
        # 从[目标层 -> 方案]的映射
        self.target_scheme_map = target_scheme_map  # 保存目标方案映射
        self.kv_cache_scheme = kv_cache_scheme  # 保存KV缓存方案
        self.sparsity_scheme_map = sparsity_scheme_map  # 保存稀疏方案映射
        self.sparsity_ignore_list = sparsity_ignore_list  # 保存稀疏忽略列表
        self.config = config  # 保存原始配置
        self.packed_modules_mapping = packed_modules_mapping or {}  # 保存打包模块映射，默认为空字典
        self.linear_fp8_config = linear_fp8_config  # 保存线性层FP8配置

    def get_linear_method(self) -> CompressedTensorsLinearMethod:  # 获取线性层量化方法
        return CompressedTensorsLinearMethod(self)  # 返回CompressedTensorsLinearMethod实例

    def get_supported_act_dtypes(cls) -> List[torch.dtype]:  # 获取支持的激活数据类型列表
        return [torch.float16, torch.bfloat16]  # 支持float16和bfloat16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低设备计算能力要求
        return 70  # 最低要求计算能力7.0

    def get_name(self) -> str:  # 获取量化方法名称
        return "compressed_tensors"  # 返回方法名称

    def get_scaled_act_names(self) -> List[str]:  # 获取需要缩放的激活名称列表
        return []  # 无需缩放的激活

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: "WeightsMapper"):  # 应用权重名称映射器，将HuggingFace名称映射为SGLang名称
        self.target_scheme_map = hf_to_sglang_mapper.apply_dict(self.target_scheme_map)  # 映射目标方案中的层名
        self.ignore = hf_to_sglang_mapper.apply_list(self.ignore)  # 映射忽略列表中的层名
        self.sparsity_scheme_map = hf_to_sglang_mapper.apply_dict(
            self.sparsity_scheme_map
        )  # 映射稀疏方案中的层名
        self.sparsity_ignore_list = hf_to_sglang_mapper.apply_list(
            self.sparsity_ignore_list
        )  # 映射稀疏忽略列表中的层名
        if self.kv_cache_scheme is not None:  # 如果KV缓存方案存在
            self.kv_cache_scheme = hf_to_sglang_mapper.apply_dict(self.kv_cache_scheme)  # 映射KV缓存方案中的层名

    def get_quant_method(  # 获取指定层的量化方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        prefix: str,  # 层名前缀
    ) -> Optional[QuantizeMethodBase]:  # 返回量化方法或None
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类

        if isinstance(layer, LinearBase):  # 如果是线性层
            # If linear_fp8_config is set, use FP8 for linear layers
            # This allows mixed quantization: experts with int4, linear layers with fp8
            # 如果设置了linear_fp8_config，则对线性层使用FP8
            # 这允许混合量化：专家层使用int4，线性层使用fp8
            if self.linear_fp8_config is not None:  # 如果存在FP8配置
                return Fp8LinearMethod(self.linear_fp8_config)  # 返回FP8线性方法
            scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)  # 获取线性层量化方案
            if scheme is None:  # 如果方案为空
                return UnquantizedLinearMethod()  # 返回未量化的线性方法
            layer.scheme = scheme  # 将方案保存到层上
            return CompressedTensorsLinearMethod(self)  # 返回压缩张量线性方法
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入FusedMoE层

        if isinstance(layer, FusedMoE):  # 如果是FusedMoE层
            layer.scheme = self.get_moe_scheme(layer=layer, layer_name=prefix)  # 获取MoE量化方案
            if layer.scheme is None:  # ignored layer  # 如果方案为空（被忽略的层）
                use_triton_kernels = get_moe_runner_backend().is_triton_kernels()  # 检测是否使用Triton内核
                use_flashinfer_trtllm_moe = (
                    get_moe_runner_backend().is_flashinfer_trtllm()
                )  # 检测是否使用FlashInfer TRT-LLM
                use_deep_gemm = get_moe_runner_backend().is_deep_gemm()  # 检测是否使用DeepGEMM
                return UnquantizedFusedMoEMethod(
                    use_triton_kernels, use_flashinfer_trtllm_moe, use_deep_gemm
                )  # 返回未量化的MoE方法
            return CompressedTensorsFusedMoEMethod(self)  # 返回压缩张量MoE方法
        return None  # 不支持的层类型返回None

    def _add_fused_moe_to_target_scheme_map(self):  # 将FusedMoE添加到目标方案映射中
        """
        Helper function to update target_scheme_map
        since linear layers get fused into FusedMoE
        targeting 'Linear' needs to also match
        FusedMoE modules.
        辅助函数，用于更新target_scheme_map
        因为线性层会被融合为FusedMoE
        所以针对'Linear'的配置也需要匹配
        FusedMoE模块。
        """
        if (
            "Linear" not in self.target_scheme_map
            or "FusedMoE" in self.target_scheme_map
        ):  # 如果映射中没有Linear或已有FusedMoE则无需添加
            return  # 直接返回
        self.target_scheme_map["FusedMoE"] = self.target_scheme_map["Linear"]  # 将Linear方案复制给FusedMoE
        self.target_scheme_map["DeepEPMoE"] = self.target_scheme_map["Linear"]  # 将Linear方案复制给DeepEPMoE

    @property
    def weight_block_size(self) -> Optional[List[int]]:  # 获取权重块大小属性
        """Get the weight block size from the quantization config."""  # 从量化配置中获取权重块大小。
        if "Linear" in self.target_scheme_map:  # 如果映射中包含Linear
            weights_config = self.target_scheme_map["Linear"].get("weights")  # 获取权重配置
            if weights_config and hasattr(weights_config, "block_structure"):  # 如果权重配置存在且有block_structure属性
                return weights_config.block_structure  # 返回块结构
        return None  # 否则返回None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> CompressedTensorsConfig:  # 从配置字典创建CompressedTensorsConfig实例
        ignore: List[str] = cast(List[str], config.get("ignore", []))  # 获取忽略列表
        quant_format = cast(str, config.get("format"))  # 获取量化格式
        target_scheme_map = cls._quantization_scheme_map_from_config(config=config)  # 从配置解析量化方案映射
        sparsity_scheme_map, sparsity_ignore_list = cls._parse_sparsity_config(
            config=config
        )  # 解析稀疏配置
        packed_modules_mapping = config.get("packed_modules_mapping", {})  # 获取打包模块映射

        # Parse linear_fp8_config if present (for mixed quantization scenarios)
        # Format: {"activation_scheme": "dynamic", "fmt": "e4m3",
        #          "quant_method": "fp8", "weight_block_size": [128, 128]}
        # 如果存在linear_fp8_config则解析（用于混合量化场景）
        # 格式：{"activation_scheme": "dynamic", "fmt": "e4m3",
        #        "quant_method": "fp8", "weight_block_size": [128, 128]}
        linear_fp8_config = None  # 初始化FP8配置为None
        if "linear_fp8_config" in config:  # 如果配置中包含linear_fp8_config
            from sglang.srt.layers.quantization.fp8 import Fp8Config  # 导入FP8配置类

            fp8_cfg = config["linear_fp8_config"]  # 获取FP8配置字典
            # Check if it's fp8 format based on quant_method field
            # 根据quant_method字段检查是否为fp8格式
            is_fp8 = fp8_cfg.get("quant_method") == "fp8"  # 判断量化方法是否为fp8
            linear_fp8_config = Fp8Config(  # 创建FP8配置实例
                is_checkpoint_fp8_serialized=is_fp8,  # 检查点是否为FP8序列化
                activation_scheme=fp8_cfg.get("activation_scheme", "dynamic"),  # 激活量化方案
                ignored_layers=fp8_cfg.get("ignored_layers"),  # 忽略的层
                weight_block_size=fp8_cfg.get("weight_block_size"),  # 权重块大小
            )

        return cls(  # 创建并返回CompressedTensorsConfig实例
            target_scheme_map=target_scheme_map,
            ignore=ignore,
            quant_format=quant_format,
            sparsity_scheme_map=sparsity_scheme_map,
            sparsity_ignore_list=sparsity_ignore_list,
            config=config,
            packed_modules_mapping=packed_modules_mapping,
            linear_fp8_config=linear_fp8_config,
        )

    @classmethod
    def _parse_sparsity_config(  # 解析稀疏配置方法
        cls, config: Dict[str, Any]
    ) -> Tuple[Dict[str, SparsityCompressionConfig], List[str]]:
        """
        :param config: The `quantization_config` dictionary from config.json
        :param config: 来自config.json的`quantization_config`字典
        :return: A tuple with two elements
        :return: 包含两个元素的元组
            1. A dictionary mapping target layer names to their corresponding
                sparsity_config
            1. 将目标层名称映射到对应稀疏配置的字典
            2. A list of layer names to ignore for sparsity
            2. 稀疏忽略的层名称列表
        """
        if not (sparsity_config := config.get(SPARSITY_CONFIG_NAME)):  # 如果没有稀疏配置
            return dict(), []  # 返回空字典和空列表

        sparsity_config = SparsityCompressionConfig.model_validate(sparsity_config)  # 验证稀疏配置
        sparse_scheme_map: Dict[str, SparsityCompressionConfig] = {
            target: sparsity_config for target in sparsity_config.targets or list()
        }  # 为每个目标创建稀疏方案映射
        sparsity_ignore_list = sparsity_config.ignore or list()  # 获取稀疏忽略列表
        return sparse_scheme_map, sparsity_ignore_list  # 返回稀疏方案映射和忽略列表

    @classmethod
    def _quantization_scheme_map_from_config(  # 从配置字典解析量化方案映射
        cls, config: Dict[str, Any]
    ) -> QUANTIZATION_SCHEME_MAP_TYPE:
        """
        :param config: The `quantization_config` dictionary from config.json
        :param config: 来自config.json的`quantization_config`字典
        :return: A dictionary mapping target layer names to their corresponding
            quantization_args for weights and input activations
        :return: 将目标层名称映射到权重和输入激活的量化参数的字典
        """
        target_scheme_map: Dict[str, Any] = dict()  # 初始化目标方案映射
        quant_format = cast(str, config.get("format"))  # 获取量化格式

        # The quant_config has multiple config_groups, each containing
        # an input_activations key with details about how the activations are
        # quantized, a weights key indicating how the weights are quantized,
        # and a list of targets under the `targets` key, dictating which
        # layers are impacted by the quantization details. The quantization
        # details follow the structure defined by the QuantizationArgs
        # pydantic model, which is used to verify the structure of the
        # quant_config and also store the details for later use.
        # quant_config包含多个config_groups，每个组包含
        # input_activations键（描述激活如何量化）、weights键（描述权重如何量化）、
        # 以及targets键下的目标列表（指示哪些层受量化影响）。
        # 量化细节遵循QuantizationArgs pydantic模型定义的结构，
        # 用于验证quant_config的结构并存储细节供后续使用。

        config_groups = config.get("config_groups", dict())  # 获取配置组
        for _, quant_config in config_groups.items():  # 遍历每个配置组
            targets = quant_config.get("targets")  # 获取目标列表
            for target in targets:  # 遍历每个目标
                target_scheme_map[target] = {}  # 初始化目标的方案字典
                target_scheme_map[target]["weights"] = QuantizationArgs.model_validate(
                    quant_config.get("weights")
                )  # 验证并保存权重量化参数

                target_scheme_map[target]["input_activations"] = None  # 初始化输入激活量化参数为None
                if is_activation_quantization_format(quant_format):  # 如果是激活量化格式
                    input_activations = quant_config.get("input_activations")  # 获取输入激活配置
                    # The only case where we have activation quant supported
                    # but no input_activations provided in the config
                    # should be w8a16fp8 w8a16fp8 can also run for cases where
                    # there is an input_quant but it is ignored
                    # 唯一支持激活量化但配置中没有提供input_activations的情况
                    # 应该是w8a16fp8，w8a16fp8也可以在有input_quant但被忽略的情况下运行
                    if not input_activations:  # 如果没有输入激活配置
                        assert (
                            target_scheme_map[target]["weights"].type
                            == QuantizationType.FLOAT
                        )  # 断言权重类型必须为FLOAT
                    else:
                        target_scheme_map[target]["input_activations"] = (
                            QuantizationArgs.model_validate(  # noqa: E501
                                quant_config.get("input_activations")
                            )
                        )  # 验证并保存输入激活量化参数
        return target_scheme_map  # 返回目标方案映射

    @classmethod
    def get_config_filenames(cls) -> List[str]:  # 获取配置文件名列表
        return []  # 返回空列表（不使用单独的配置文件）

    def _check_scheme_supported(self, min_capability: int, error: bool = True) -> bool:  # 检查当前设备是否支持该量化方案
        capability_tuple = DeviceCapability(*torch.cuda.get_device_capability())  # 获取设备计算能力

        if capability_tuple is not None:  # 如果获取到设备能力
            capability = capability_tuple.to_int()  # 转换为整数表示
            supported = capability >= min_capability  # 检查是否满足最低要求
            if error and not supported:  # 如果不满足且需要报错
                raise RuntimeError(
                    "Quantization scheme is not supported for ",
                    f"the current GPU. Min capability: {min_capability}. ",
                    f"Current capability: {capability}.",
                )  # 抛出运行时错误
            return supported  # 返回是否支持
        else:
            return False  # 无法获取设备能力则返回不支持

    def _is_dynamic_token_w4a8(  # 判断是否为动态token级W4A8量化方案
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> bool:
        is_weight_4_bits = weight_quant.num_bits == 4  # 权重是否为4位
        is_activation_8_bits = input_quant.num_bits == 8  # 激活是否为8位
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.GROUP.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        )  # 权重策略是否为GROUP或CHANNEL
        is_token = (
            weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
        )  # 是否为token级激活量化
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic  # 权重静态且激活动态

        return (
            is_weight_4_bits
            and is_activation_8_bits
            and is_token
            and weight_quant.symmetric
            and is_dynamic
        )  # 返回是否为动态token级W4A8方案

    def _is_static_tensor_w8a8(  # 判断是否为静态tensor级W8A8量化方案
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8  # 权重和激活是否都为8位
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        )  # 权重策略是否为TENSOR或CHANNEL
        is_tensor = (
            weight_strategy
            and input_quant.strategy == QuantizationStrategy.TENSOR.value
        )  # 是否为tensor级激活量化
        is_static = not weight_quant.dynamic and not input_quant.dynamic  # 权重和激活都为静态

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        # 支持对称和非对称输入量化。
        # 仅支持对称权重量化。
        return is_8_bits and is_tensor and weight_quant.symmetric and is_static  # 返回是否为静态tensor级W8A8方案

    def _is_dynamic_token_w8a8(  # 判断是否为动态token级W8A8量化方案
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8  # 权重和激活是否都为8位
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        )  # 权重策略是否为TENSOR或CHANNEL
        is_token = (
            weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
        )  # 是否为token级激活量化
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic  # 权重静态且激活动态

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        # 支持对称和非对称输入量化。
        # 仅支持对称权重量化。
        return is_8_bits and is_token and weight_quant.symmetric and is_dynamic  # 返回是否为动态token级W8A8方案

    def _is_fp8_w8a8(  # 判断是否为FP8 W8A8量化方案
        self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        # Confirm weights and activations quantized.
        # 确认权重和激活都已量化。
        if weight_quant is None or input_quant is None:  # 如果权重或激活未量化
            return False  # 返回False

        # Confirm weight scheme is supported.
        # 确认权重方案受支持。
        is_floating_point = (
            weight_quant.type == QuantizationType.FLOAT
            and input_quant.type == QuantizationType.FLOAT
        )  # 权重和激活是否都为浮点类型
        is_symmetric_weight = weight_quant.symmetric  # 权重是否为对称量化
        is_static_weight = not weight_quant.dynamic  # 权重是否为静态
        is_tensor_or_channel_or_block_weight = weight_quant.strategy in [
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
            QuantizationStrategy.BLOCK,
        ]  # 权重策略是否为TENSOR/CHANNEL/BLOCK
        if not (
            is_floating_point
            and is_symmetric_weight
            and is_static_weight
            and is_tensor_or_channel_or_block_weight
        ):  # 如果不满足权重方案条件
            return False  # 返回False

        # Dynamic quantization is always supported if weights supported.
        # 如果权重受支持，动态量化始终受支持。
        if input_quant.dynamic:  # 如果激活为动态量化
            return True  # 返回True

        # Confirm activation scheme is supported.
        # 确认激活方案受支持。
        is_symmetric_activation = input_quant.symmetric  # 激活是否为对称量化
        is_per_tensor_activation = input_quant.strategy == QuantizationStrategy.TENSOR  # 激活策略是否为TENSOR
        return is_symmetric_activation and is_per_tensor_activation  # 返回激活方案是否受支持

    def _is_fp8_w8a16(self, weight_quant: BaseModel, input_quant: BaseModel) -> bool:  # 判断是否为FP8 W8A16量化方案
        # Confirm weights quantized.
        # 确认权重已量化。
        if weight_quant is None:  # 如果权重未量化
            return False  # 返回False

        # Confirm we have floating points.
        # 确认为浮点类型。
        if weight_quant.type != QuantizationType.FLOAT:  # 如果权重不是浮点类型
            return False  # 返回False

        # Confirm weight scheme is supported.
        # 确认权重方案受支持。
        is_symmetric_weight = weight_quant.symmetric  # 权重是否为对称量化
        is_static_weight = not weight_quant.dynamic  # 权重是否为静态
        is_per_tensor_or_channel_weight = weight_quant.strategy in [
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
        ]  # 权重策略是否为TENSOR或CHANNEL
        if not (
            is_symmetric_weight
            and is_static_weight  # noqa: SIM103
            and is_per_tensor_or_channel_weight
        ):  # 如果不满足权重方案条件
            return False  # 返回False

        # All conditions satisfied.
        # 所有条件满足。
        return True  # 返回True

    def _is_fp4a4_nvfp4(  # 判断是否为NVFP4 W4A4量化方案
        self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ):
        if weight_quant is None or input_quant is None:  # 如果权重或激活未量化
            return False  # 返回False

        is_tensor_group_quant = (
            weight_quant.strategy == QuantizationStrategy.TENSOR_GROUP.value
            and input_quant.strategy == QuantizationStrategy.TENSOR_GROUP.value
        )  # 权重和激活策略是否都为TENSOR_GROUP
        is_symmetric = weight_quant.symmetric and input_quant.symmetric  # 权重和激活是否都为对称

        is_group_size_16 = (
            weight_quant.group_size == 16 and input_quant.group_size == 16
        )  # 权重和激活分组大小是否都为16
        is_float_type = (
            weight_quant.type == QuantizationType.FLOAT
            and input_quant.type == QuantizationType.FLOAT
        )  # 权重和激活是否都为浮点类型
        is_4_bits = weight_quant.num_bits == 4 and input_quant.num_bits == 4  # 权重和激活是否都为4位

        return (
            is_tensor_group_quant
            and is_float_type
            and is_4_bits
            and is_group_size_16
            and is_symmetric
        )  # 返回是否为NVFP4 W4A4方案

    def _is_wNa16_group_channel(  # 判断是否为WNA16分组/通道量化方案（权重N位，激活16位）
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> bool:
        input_quant_none = input_quant is None  # 输入激活量化是否为空
        is_symmetric = weight_quant.symmetric  # 权重是否为对称量化
        is_channel_group = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value
        )  # 权重策略是否为CHANNEL或GROUP
        is_static = not weight_quant.dynamic  # 权重是否为静态

        return is_channel_group and input_quant_none and is_symmetric and is_static  # 返回是否为WNA16分组/通道方案

    def _is_mxint4a16(self, weight_quant: BaseModel, input_quant: BaseModel) -> bool:  # 判断是否为MX INT4 W4A16量化方案
        input_quant_none = input_quant is None  # 输入激活量化是否为空
        is_symmetric = weight_quant.symmetric  # 权重是否为对称量化
        is_mxint4 = (
            weight_quant.num_bits == 4
            and weight_quant.type == QuantizationType.INT
            and weight_quant.strategy == QuantizationStrategy.GROUP.value
            and weight_quant.group_size == 32
        )  # 是否满足MX INT4条件：4位、INT类型、GROUP策略、分组大小32
        is_static = not weight_quant.dynamic  # 权重是否为静态

        return is_mxint4 and input_quant_none and is_symmetric and is_static  # 返回是否为MX INT4 W4A16方案

    def _is_dynamic_token_w4(  # 判断是否为动态token级W4量化方案
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> bool:
        is_w4 = weight_quant.num_bits == 4  # 权重是否为4位
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value
        )  # 权重策略是否为TENSOR/CHANNEL/GROUP
        if input_quant is not None:  # 如果存在输入激活量化
            is_token = (
                weight_strategy
                and input_quant.strategy == QuantizationStrategy.TOKEN.value
            )  # 是否为token级激活量化
            is_dynamic = not weight_quant.dynamic and input_quant.dynamic  # 权重静态且激活动态
        else:  # 如果没有输入激活量化
            is_token = weight_strategy  # 使用权重策略判断
            is_dynamic = not weight_quant.dynamic  # 仅检查权重是否静态

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        # 支持对称和非对称输入量化。
        # 仅支持对称权重量化。
        return is_w4 and weight_quant.symmetric and is_token and is_dynamic  # 返回是否为动态token级W4方案

    def _get_scheme_from_parts(  # 根据权重和激活量化参数选择对应的量化方案
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> CompressedTensorsLinearScheme:

        # Detect If Mixed Precision
        # 检测是否为混合精度
        if self._is_wNa16_group_channel(weight_quant, input_quant):  # 如果是WNA16分组/通道方案
            if (
                self.quant_format == CompressionFormat.pack_quantized.value
                and weight_quant.num_bits in WNA16_SUPPORTED_BITS
            ):  # 如果是打包量化格式且位数受支持
                return CompressedTensorsWNA16(  # 返回WNA16方案
                    num_bits=weight_quant.num_bits,
                    strategy=weight_quant.strategy,
                    group_size=weight_quant.group_size,
                    actorder=weight_quant.actorder,
                )
            else:  # 其他格式不受支持
                raise ImportError(
                    "Other method (CompressedTensorsW4A16Sparse24) is not supported now"
                )  # 抛出导入错误

        if is_activation_quantization_format(self.quant_format):  # 如果是激活量化格式
            if self._is_fp4a4_nvfp4(weight_quant, input_quant):  # 如果是NVFP4 W4A4方案
                is_fp4a4_nvfp4_supported = self._check_scheme_supported(
                    CompressedTensorsW4A4Fp4.get_min_capability(), error=False
                )  # 检查设备是否支持NVFP4
                if is_fp4a4_nvfp4_supported:  # 如果支持
                    return CompressedTensorsW4A4Fp4()  # 返回W4A4 FP4方案
                else:  # 如果不支持
                    raise NotImplementedError(
                        "Current platform does not support w4a4 nvfp4 quantization."
                    )  # 抛出未实现错误

            if self._is_fp8_w8a8(weight_quant, input_quant):  # 如果是FP8 W8A8方案
                is_fp8_w8a8_supported = self._check_scheme_supported(
                    CompressedTensorsW8A8Fp8.get_min_capability(), error=False
                )  # 检查设备是否支持FP8 W8A8
                if is_fp8_w8a8_supported:  # 如果支持
                    return CompressedTensorsW8A8Fp8(  # 返回W8A8 FP8方案
                        weight_quant=weight_quant,
                        is_static_input_scheme=(
                            input_quant and not input_quant.dynamic
                        ),
                    )
                else:  # 如果不支持
                    # note: input_quant will be present for converted models;
                    # will be ignored during inference post loading
                    # 注意：input_quant在转换后的模型中会存在；
                    # 在推理加载后将被忽略
                    return CompressedTensorsW8A16Fp8(  # 降级返回W8A16 FP8方案
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=not input_quant.dynamic,
                    )

            # note: input_quant can be None
            # 注意：input_quant可能为None
            if self._is_fp8_w8a16(weight_quant, input_quant):  # 如果是FP8 W8A16方案
                is_static_input_scheme = input_quant and not input_quant.dynamic  # 判断输入是否为静态方案
                return CompressedTensorsW8A16Fp8(  # 返回W8A16 FP8方案
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=is_static_input_scheme,
                )

            if self._is_static_tensor_w8a8(weight_quant, input_quant):  # 如果是静态tensor级W8A8方案
                if not _is_npu:  # 如果不是NPU平台
                    return CompressedTensorsW8A8Int8(  # 返回GPU版W8A8 INT8方案
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=True,
                        input_symmetric=input_quant.symmetric,
                    )
                else:  # 如果是NPU平台
                    return NPUCompressedTensorsW8A8Int8(  # 返回NPU版W8A8 INT8方案
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=True,
                        input_symmetric=input_quant.symmetric,
                    )

            if self._is_dynamic_token_w8a8(weight_quant, input_quant):  # 如果是动态token级W8A8方案
                if not _is_npu:  # 如果不是NPU平台
                    return CompressedTensorsW8A8Int8(  # 返回GPU版W8A8 INT8方案（动态）
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=False,
                        input_symmetric=input_quant.symmetric,
                    )
                else:  # 如果是NPU平台
                    return NPUCompressedTensorsW8A8Int8(  # 返回NPU版W8A8 INT8方案（动态）
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=False,
                        input_symmetric=input_quant.symmetric,
                    )

        raise NotImplementedError("No compressed-tensors compatible scheme was found.")  # 抛出未实现错误

    def get_moe_scheme(  # 获取MoE层量化方案
        self, layer: torch.nn.Module, layer_name: Optional[str] = None
    ) -> Optional[CompressedTensorsMoEScheme]:
        """
        compressed-tensors supports non uniform in the following way:
        compressed-tensors通过以下方式支持非均匀量化：

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.
        config_groups的目标：可以有N个config_groups，每个都有量化方案。
            每个config_group有一个目标列表，可以是完整的层名、层名的正则表达式，
            或nn.Module名称。

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsMoEScheme used for infernece.
        检测layer_name是否在任何目标中找到，
        并使用匹配目标对应的量化方案
        来选择用于推理的CompressedTensorsMoEScheme。
        """

        # FusedMoE was made by combining multiple Linears so need to
        # make sure quantization config for Linear can target it
        # FusedMoE由多个Linear组合而成，因此需要
        # 确保Linear的量化配置也能匹配它
        self._add_fused_moe_to_target_scheme_map()  # 将FusedMoE添加到目标方案映射
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]  # 构建未融合的投影层名称列表
        # TODO: refactor this to use expert_mapping and check all layer numbers
        # TODO: 重构此部分以使用expert_mapping并检查所有层编号
        all_scheme_dicts = [self.get_scheme_dict(layer, name) for name in unfused_names]  # 获取所有投影层的方案字典
        scheme_dict = all_scheme_dicts[0] if all_scheme_dicts else None  # 取第一个方案字典

        # multiple schemes found
        # 发现多个方案
        if not all(d == scheme_dict for d in all_scheme_dicts):  # 如果方案不一致
            raise ValueError(
                "All MoE projections need to have same "
                "quantization scheme but found multiple"
            )  # 抛出值错误

        if scheme_dict is None:  # ignored layer  # 被忽略的层
            return None  # 返回None

        weight_quant = scheme_dict.get("weights")  # 获取权重量化参数
        input_quant = scheme_dict.get("input_activations")  # 获取输入激活量化参数

        if self._is_wNa16_group_channel(weight_quant, input_quant):  # 如果是WNA16分组/通道方案
            if not _is_npu:  # 如果不是NPU平台
                if (
                    self._is_mxint4a16(weight_quant, input_quant)
                    and get_moe_runner_backend().is_flashinfer_trtllm()
                ):  # 如果是MX INT4且使用flashinfer_trtllm后端
                    logger.info_once(
                        "Using CompressedTensorsMxInt4MoE with flashinfer_trtllm backend"
                    )  # 记录使用MX INT4 MoE方案
                    return CompressedTensorsMxInt4MoE(self)  # 返回MX INT4 MoE方案
                elif _is_hip:  # 如果是HIP(ROCm)平台
                    logger.info_once("Using CompressedTensorsWNA16TritonMoE (ROCm)")  # 记录使用Triton MoE方案
                    return CompressedTensorsWNA16TritonMoE(self)  # 返回Triton MoE方案
                else:  # 其他GPU平台
                    moe_backend = get_moe_runner_backend()  # 获取MoE运行器后端
                    if moe_backend.is_triton():  # 如果使用Triton后端
                        logger.info_once(
                            "Using CompressedTensorsWNA16TritonMoE "
                            "(moe_runner_backend=triton)"
                        )  # 记录使用Triton MoE方案
                        return CompressedTensorsWNA16TritonMoE(self)  # 返回Triton MoE方案
                    logger.info_once("Using CompressedTensorsWNA16MarlinMoEMethod")  # 记录使用Marlin MoE方案
                    return CompressedTensorsWNA16MoE(self)  # 返回Marlin MoE方案
            else:  # NPU平台
                if (
                    self._is_dynamic_token_w4(weight_quant, input_quant)
                    and input_quant is None
                ):  # 如果是动态token级W4且无输入量化
                    logger.info_once("Using NPUCompressedTensorsW4A16Int4DynamicMoE")  # 记录使用NPU W4A16方案
                    return NPUCompressedTensorsW4A16Int4DynamicMoE(self)  # 返回NPU W4A16动态MoE方案
        elif self._is_fp4a4_nvfp4(weight_quant, input_quant):  # 如果是NVFP4 W4A4方案
            logger.info_once("Using CompressedTensorsW4A4Nvfp4MoE")  # 记录使用NVFP4 MoE方案
            return CompressedTensorsW4A4Nvfp4MoE()  # 返回NVFP4 MoE方案
        elif self._is_fp8_w8a8(weight_quant, input_quant):  # 如果是FP8 W8A8方案
            logger.info_once("Using CompressedTensorsW8A8Fp8MoE")  # 记录使用FP8 MoE方案
            return CompressedTensorsW8A8Fp8MoE(weight_quant, input_quant)  # 返回FP8 MoE方案
        elif self._is_dynamic_token_w8a8(weight_quant, input_quant):  # 如果是动态token级W8A8方案
            if _is_npu:  # 如果是NPU平台
                logger.info_once("Using NPUCompressedTensorsW8A8Int8DynamicMoE")  # 记录使用NPU W8A8方案
                return NPUCompressedTensorsW8A8Int8DynamicMoE(weight_quant, input_quant)  # 返回NPU W8A8动态MoE方案
            else:  # 非NPU平台
                raise NotImplementedError(
                    f"The W8A8Int8 Fused MoE scheme is implemented only for NPU for now."
                )  # 抛出未实现错误，当前仅NPU支持
        elif self._is_dynamic_token_w4a8(weight_quant, input_quant):  # 如果是动态token级W4A8方案
            if _is_npu:  # 如果是NPU平台
                logger.info_once("Using NPUCompressedTensorsW4A8Int8DynamicMoE")  # 记录使用NPU W4A8方案
                return NPUCompressedTensorsW4A8Int8DynamicMoE(self)  # 返回NPU W4A8动态MoE方案
            else:  # 非NPU平台
                raise NotImplementedError(
                    f"The W4A8Int8 Fused MoE scheme is implemented only for NPU for now."
                )  # 抛出未实现错误，当前仅NPU支持
        else:  # 其他不支持的方案
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )  # 抛出运行时错误

    def get_linear_scheme(  # 获取线性层量化方案
        self, layer: torch.nn.Module, layer_name: Optional[str] = None
    ) -> Optional[CompressedTensorsLinearScheme]:
        """
        compressed-tensors supports non uniform in the following way:
        compressed-tensors通过以下方式支持非均匀量化：

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.
        config_groups的目标：可以有N个config_groups，每个都有量化方案。
            每个config_group有一个目标列表，可以是完整的层名、层名的正则表达式，
            或nn.Module名称。

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for infernece.
        检测layer_name是否在任何目标中找到，
        并使用匹配目标对应的量化方案
        来选择用于推理的CompressedTensorsScheme。
        """

        # Find the "target" in the compressed-tensors config
        # that our layer conforms to.
        # 在compressed-tensors配置中查找
        # 与我们层匹配的"target"。
        # TODO : add compressed-tensors as dep
        # so we do not have to re-write these functions
        # need to make accelerate optional in ct to do this
        # TODO：将compressed-tensors添加为依赖
        # 这样就不必重写这些函数
        # 需要将accelerate在ct中设为可选才能做到这一点

        # Use the new get_scheme_dict method to extract QuantizationArgs
        # 使用新的get_scheme_dict方法提取QuantizationArgs
        scheme_dict = self.get_scheme_dict(layer, layer_name)  # 获取方案字典
        weight_quant = None  # 初始化权重量化参数
        input_quant = None  # 初始化输入激活量化参数
        if scheme_dict:  # 如果方案字典存在
            weight_quant = scheme_dict.get("weights")  # 获取权重量化参数
            input_quant = scheme_dict.get("input_activations")  # 获取输入激活量化参数

        # Find the sparsity scheme of the layer
        # assume that fused layers inerhit first component's sparsity scheme
        # 查找层的稀疏方案
        # 假设融合层继承第一个组件的稀疏方案
        sparsity_targets = self.sparsity_scheme_map.keys() - set(
            self.sparsity_ignore_list
        )  # 计算有效稀疏目标
        sparsity_scheme: Optional[SparsityCompressionConfig] = None  # 初始化稀疏方案
        with suppress(ValueError):  # 抑制ValueError
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=sparsity_targets,
                fused_mapping=self.packed_modules_mapping,
            )  # 查找匹配的稀疏目标
            sparsity_scheme = self.sparsity_scheme_map[matched_target]  # 获取稀疏方案

        if self.supports_cutlass_24(
            weight_quant=weight_quant,
            input_quant=input_quant,
            sparsity_scheme=sparsity_scheme,
        ):  # 如果支持Cutlass 2:4稀疏
            raise ImportError("CompressedTensors24 is not supported now")  # 抛出导入错误
        elif weight_quant is None:  # 如果没有权重量化参数
            logger.warning_once(
                "Acceleration for non-quantized schemes is "
                "not supported by Compressed Tensors. "
                "Falling back to UnquantizedLinearMethod"
            )  # 记录警告：非量化方案不支持加速
            return None  # 返回None

        else:  # 存在权重量化参数
            # Find the quant_scheme
            # 查找量化方案
            scheme = self._get_scheme_from_parts(  # type: ignore
                weight_quant=weight_quant,
                input_quant=input_quant,
            )  # 从量化参数获取方案

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        # 如果设备不支持该方案则抛出错误
        # （例如fp8需要Ada Lovelace架构）
        # Note: NPU devices do not support min_capability function
        # 注意：NPU设备不支持min_capability函数
        if not _is_npu:  # 如果不是NPU平台
            self._check_scheme_supported(scheme.get_min_capability())  # 检查设备是否支持该方案
        logger.debug("Using scheme: %s for %s", scheme.__class__.__name__, layer_name)  # 记录使用的方案
        return scheme  # 返回量化方案

    def get_scheme_dict(  # 获取指定层的量化方案字典
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> dict[str, QuantizationArgs | str | None] | None:
        """
        Extract the QuantizationArgs for a given layer.
        提取给定层的量化参数。

        Returns:
        返回：
            dict with {
            包含以下键的字典 {
                "weights": QuantizationArgs,
                "input_activations": QuantizationArgs | None,
                "format": str | None
            } | None
        """
        if should_ignore_layer(
            layer_name, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
        ):  # 如果该层应被忽略
            return None  # 返回None

        # Will be empty for models with only sparsity
        # 对于仅使用稀疏的模型将为空
        if self.target_scheme_map:  # 如果目标方案映射非空
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping,
            )  # 查找匹配的目标

            return self.target_scheme_map[matched_target]  # 返回匹配目标的方案字典

        return None  # 目标方案映射为空则返回None

    def get_cache_scale(self, name: str) -> Optional[str]:  # 获取KV缓存缩放参数名
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM
        检查参数名是否匹配compressed-tensors中k/v缓存缩放的格式。
        如果匹配，返回vLLM期望的等效参数名

        :param name: param name
        :param name: 参数名
        :return: matching param name for KV cache scale in vLLM
        :return: vLLM中KV缓存缩放的匹配参数名
        """
        if name.endswith(".output_scale") and ".k_proj" in name:  # 如果是k_proj的输出缩放
            return name.replace(".k_proj.output_scale", ".attn.k_scale")  # 替换为注意力k缩放名
        if name.endswith(".output_scale") and ".v_proj" in name:  # 如果是v_proj的输出缩放
            return name.replace(".v_proj.output_scale", ".attn.v_scale")  # 替换为注意力v缩放名
        # If no matches, return None
        # 如果没有匹配，返回None
        return None  # 无匹配返回None

    @staticmethod
    def supports_cutlass_24(  # 静态方法，检查是否支持Cutlass 2:4稀疏内核
        weight_quant: Optional[QuantizationArgs],  # 权重量化参数
        input_quant: Optional[QuantizationArgs],  # 输入激活量化参数
        sparsity_scheme: Optional[SparsityCompressionConfig] = None,  # 稀疏方案配置
    ) -> bool:
        """
        Check if the layer is supported by the Cutlass 2:4 Kernel
        Conditions:
            - Overarching condition: Sparsity Structure is 2:4
            - Unquantized cases are supported
            - Weight only quantization is not-supported
            - Supported weight quantization strategies are TENSOR and CHANNEL
            - Supported input quantization strategies are TENSOR and TOKEN
            - Only 8 bit quantization is supported
        检查层是否受Cutlass 2:4内核支持
        条件：
            - 总体条件：稀疏结构为2:4
            - 支持未量化的情况
            - 不支持仅权重量化
            - 支持的权重量化策略为TENSOR和CHANNEL
            - 支持的输入量化策略为TENSOR和TOKEN
            - 仅支持8位量化

        :return: True if the layer is supported by the Cutlass 2:4 Kernel
            False otherwise
        :return: 如果层受Cutlass 2:4内核支持则返回True，否则返回False
        """
        if sparsity_scheme is None:  # 如果没有稀疏方案
            return False  # 返回False

        is_valid_sparsity_structure: bool = (
            sparsity_scheme.sparsity_structure == SparsityStructure.TWO_FOUR.value
        )  # 检查稀疏结构是否为2:4

        valid_compressors = {
            CompressionFormat.dense.value,
            CompressionFormat.sparse_24_bitmask.value,
        }  # 有效的压缩格式集合

        is_valid_sparsity = (
            is_valid_sparsity_structure and sparsity_scheme.format in valid_compressors
        )  # 检查稀疏结构是否有效

        if not is_valid_sparsity:  # 如果稀疏无效
            return False  # 返回False

        # Unquantized cases are supported
        # 支持未量化的情况
        if weight_quant is None and input_quant is None:  # 如果权重和激活都未量化
            return True  # 返回True

        # Weight only quantization is not-supported
        # 不支持仅权重量化
        if weight_quant is not None and input_quant is None:  # 如果有权重量化但无激活量化
            return False  # 返回False

        supported_weight_quant_strategies = [
            QuantizationStrategy.TENSOR.value,
            QuantizationStrategy.CHANNEL.value,
        ]  # 支持的权重量化策略列表

        assert weight_quant is not None  # 断言权重量化参数不为空
        assert input_quant is not None  # 断言输入激活量化参数不为空
        if weight_quant.strategy not in supported_weight_quant_strategies:  # 如果权重量化策略不受支持
            return False  # 返回False

        supported_input_quant_strategies = [
            QuantizationStrategy.TENSOR.value,
            QuantizationStrategy.TOKEN.value,
        ]  # 支持的输入量化策略列表

        if input_quant.strategy not in supported_input_quant_strategies:  # 如果输入量化策略不受支持
            return False  # 返回False

        return weight_quant.num_bits == input_quant.num_bits == 8  # 返回是否都为8位量化


class CompressedTensorsLinearMethod(LinearMethodBase):  # Compressed Tensors线性层量化方法类，继承自LinearMethodBase

    def __init__(self, quantization_config: CompressedTensorsConfig):  # 初始化方法
        self.quantization_config = quantization_config  # 保存量化配置
        self.quant_config = quantization_config  # 保存量化配置（别名）

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理方法
        layer.scheme.process_weights_after_loading(layer)  # 调用层的方案的后处理方法

    def create_weights(  # 创建量化权重参数方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: List[int],  # 输出分区大小列表
        input_size: int,  # 输入总大小
        output_size: int,  # 输出总大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        使用与每层关联的CompressedTensorsScheme创建
        该层所需的参数。参数详情参见LinearMethodBase
        """
        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器
        layer.scheme.create_weights(  # 调用层的方案创建权重
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def apply(  # 应用量化权重进行前向计算方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置张量
    ):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details
        使用create_weights的输出和与层关联的CompressedTensorsScheme
        对层输入应用前向传播。参数详情参见LinearMethodBase

        """

        scheme = layer.scheme  # 获取层的量化方案
        if scheme is None:  # 如果方案为空
            raise ValueError("A scheme must be defined for each layer")  # 抛出值错误
        return scheme.apply_weights(layer, x, bias=bias)  # 调用方案的apply_weights方法


class CompressedTensorsFusedMoEMethod(FusedMoEMethodBase):  # Compressed Tensors融合MoE量化方法类，继承自FusedMoEMethodBase

    def __init__(self, quantization_config: CompressedTensorsConfig):  # 初始化方法
        self.quantization_config = quantization_config  # 保存量化配置
        self.quant_config = quantization_config  # 保存量化配置（别名）

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理方法
        layer.scheme.process_weights_after_loading(layer)  # 调用层的方案的后处理方法

    def create_weights(  # 创建量化权重参数方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        使用与每层关联的CompressedTensorsScheme创建
        该层所需的参数。参数详情参见LinearMethodBase
        """
        layer.scheme.create_weights(  # 调用层的方案创建权重
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(  # 创建MoE运行器方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        return layer.scheme.create_moe_runner(layer, moe_runner_config)  # 调用层的方案创建MoE运行器

    def get_triton_quant_info(self, layer: torch.nn.Module):  # 获取Triton量化信息方法
        return layer.scheme.get_triton_quant_info(layer)  # 调用层的方案获取Triton量化信息

    def get_marlin_quant_info(self, layer: torch.nn.Module):  # 获取Marlin量化信息方法
        return layer.scheme.get_marlin_quant_info(layer)  # 调用层的方案获取Marlin量化信息

    def apply(  # 应用量化权重进行前向计算方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        dispatch_output: StandardDispatchOutput,  # 分发输出
    ) -> CombineInput:  # 返回合并输入
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details
        使用create_weights的输出和与层关联的CompressedTensorsScheme
        对层输入应用前向传播。参数详情参见LinearMethodBase

        """
        scheme = layer.scheme  # 获取层的量化方案
        if scheme is None:  # 如果方案为空
            raise ValueError("A scheme must be defined for each layer")  # 抛出值错误
        return scheme.apply_weights(layer, dispatch_output)  # 调用方案的apply_weights方法

    def apply_weights_with_router_logits(  # 带路由器logits的权重应用方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        dispatch_output: StandardDispatchOutput,  # 分发输出
    ) -> torch.Tensor:  # 返回张量
        scheme = layer.scheme  # 获取层的量化方案
        if scheme is None:  # 如果方案为空
            raise ValueError("A scheme must be defined for each layer")  # 抛出值错误
        return scheme.apply_weights_with_router_logits(layer, dispatch_output)  # 调用方案的apply_weights_with_router_logits方法

    def apply_without_routing_weights(  # 无路由权重的应用方法
        self,
        layer,  # 目标层
        hidden_states,  # 隐藏状态
        hidden_states_scale,  # 隐藏状态缩放
        group_list_type,  # 分组列表类型
        group_list,  # 分组列表
        output_dtype,  # 输出数据类型
    ):
        return layer.scheme.apply_without_routing_weights(  # 调用层的方案的apply_without_routing_weights方法
            layer,
            hidden_states,
            hidden_states_scale,
            group_list_type,
            group_list,
            output_dtype,
        )
