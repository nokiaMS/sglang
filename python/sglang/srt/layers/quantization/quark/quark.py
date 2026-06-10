# Quark 量化框架核心实现
# 本文件实现了 Quark 量化框架的核心配置和执行逻辑，
# 包括 QuarkConfig（量化配置）、QuarkLinearMethod（线性层量化方法）、
# QuarkFusedMoEMethod（融合 MoE 量化方法）和 QuarkKVCacheMethod（KV 缓存量化方法），
# 支持 FP8 W8A8 和 MX-FP4 W4A4 等量化方案。

# SPDX-License-Identifier: Apache-2.0

import fnmatch  # 导入文件名模式匹配模块 # import filename pattern matching module
import logging  # 导入日志模块 # import logging module
from typing import TYPE_CHECKING, Any, List, Optional, cast  # 导入类型提示工具 # import type hints

import torch  # 导入 PyTorch 深度学习框架 # import PyTorch framework

from sglang.srt.layers.linear import LinearBase  # 导入线性层基类 # import linear layer base class
from sglang.srt.layers.moe import MoeRunnerConfig  # 导入 MoE 运行器配置 # import MoE runner config
from sglang.srt.layers.quantization.base_config import (  # noqa: E501 导入量化基础配置类 # import quantization base config classes
    FusedMoEMethodBase,  # 融合 MoE 方法基类 # fused MoE method base class
    LinearMethodBase,  # 线性方法基类 # linear method base class
    QuantizationConfig,  # 量化配置基类 # quantization config base class
    QuantizeMethodBase,  # 量化方法基类 # quantize method base class
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod  # 导入 KV 缓存方法基类 # import KV cache method base class
from sglang.srt.layers.quantization.quark.schemes import (  # 导入 Quark 量化方案 # import Quark quantization schemes
    QuarkLinearScheme,  # Quark 线性层量化方案基类 # Quark linear scheme base class
    QuarkMoEScheme,  # Quark MoE 量化方案基类 # Quark MoE scheme base class
    QuarkW4A4MXFP4,  # Quark W4A4 MX-FP4 量化方案 # Quark W4A4 MX-FP4 scheme
    QuarkW4A4MXFp4MoE,  # Quark W4A4 MX-FP4 MoE 量化方案 # Quark W4A4 MX-FP4 MoE scheme
    QuarkW8A8Fp8,  # Quark W8A8 FP8 量化方案 # Quark W8A8 FP8 scheme
    QuarkW8A8FP8MoE,  # Quark W8A8 FP8 MoE 量化方案 # Quark W8A8 FP8 MoE scheme
)
from sglang.srt.layers.quantization.quark.utils import deep_compare, should_ignore_layer  # 导入工具函数 # import utility functions
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 导入未量化线性方法 # import unquantized linear method
from sglang.srt.layers.radix_attention import RadixAttention  # 导入 Radix 注意力层 # import Radix attention layer
from sglang.srt.utils import get_device_capability  # 导入设备能力查询工具 # import device capability query utility

if TYPE_CHECKING:  # 类型检查时才导入 # import only during type checking
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput  # 标准分发输出类型 # standard dispatch output type

__all__ = ["QuarkLinearMethod", "QuarkFusedMoEMethod"]  # 模块公开接口 # module public interface

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器 # get logger for current module


class QuarkConfig(QuantizationConfig):  # Quark 量化配置类，继承自 QuantizationConfig # Quark quantization config class

    def __init__(  # 初始化方法 # initializer
        self,
        quant_config: dict[str, Any],  # 量化配置字典 # quantization config dict
        kv_cache_group: Optional[list[str]] = None,  # KV 缓存分组列表 # KV cache group list
        kv_cache_config: Optional[dict[str, Any]] = None,  # KV 缓存配置 # KV cache config
        pack_method: str = "reorder",  # 打包方法，默认为重排序 # pack method, default reorder
    ):
        super().__init__()  # 调用父类初始化 # call parent class initializer
        if kv_cache_group is None:  # 如果 KV 缓存分组为空 # if KV cache group is None
            kv_cache_group = []  # 初始化为空列表 # initialize as empty list
        self.quant_config = quant_config  # 保存量化配置 # save quantization config
        self.kv_cache_group = kv_cache_group  # 保存 KV 缓存分组 # save KV cache group
        self.kv_cache_config = kv_cache_config  # 保存 KV 缓存配置 # save KV cache config
        self.pack_method = pack_method  # 保存打包方法 # save pack method
        self.exclude_layers = cast(list[str], self.quant_config.get("exclude", []))  # 获取排除层列表 # get excluded layers list

        self.packed_modules_mapping = self.quant_config["packed_modules_mapping"]  # 获取打包模块映射 # get packed modules mapping

    def get_linear_method(self) -> "QuarkLinearMethod":  # 获取线性层量化方法 # get linear quantization method
        return QuarkLinearMethod(self)  # 返回 QuarkLinearMethod 实例 # return QuarkLinearMethod instance

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:  # 获取支持的激活数据类型 # get supported activation dtypes
        return [torch.float16, torch.bfloat16]  # 支持 float16 和 bfloat16 # support float16 and bfloat16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低设备算力要求 # get minimum device capability
        return 70  # 最低算力为 7.0 # minimum capability 7.0

    def get_name(self) -> str:  # 获取量化方法名称 # get quantization method name
        return "quark"  # 返回名称 "quark" # return name "quark"

    def apply_weight_name_mapper(self, hf_to_sglang_mapper):  # 应用权重名称映射器 # apply weight name mapper
        mapped = hf_to_sglang_mapper.apply_list(self.exclude_layers)  # 映射排除层名称 # map excluded layer names
        expanded = []  # 扩展后的列表 # expanded list
        for name in mapped:  # 遍历映射后的名称 # iterate over mapped names
            expanded.append(name)  # 添加原始名称 # add original name
            if name.startswith("language_model."):  # 如果名称以 "language_model." 开头 # if name starts with "language_model."
                expanded.append(name.removeprefix("language_model."))  # 添加去掉前缀后的名称 # add name without prefix
        self.exclude_layers = list(dict.fromkeys(expanded))  # 去重并保持顺序 # deduplicate while preserving order

    def get_quant_method(  # 获取层的量化方法 # get quantization method for layer
        self, layer: torch.nn.Module, prefix: str  # 神经网络层和前缀 # neural network layer and prefix
    ) -> Optional["QuantizeMethodBase"]:  # 返回量化方法或 None # return quantization method or None
        # Check if the layer is skipped for quantization.
        # 检查该层是否跳过量化。
        if should_ignore_layer(  # 判断是否应忽略该层 # check if layer should be ignored
            prefix,  # 层名前缀 # layer name prefix
            ignore=self.exclude_layers,  # 排除层列表 # excluded layers list
            fused_mapping=self.packed_modules_mapping,  # 打包模块映射 # packed modules mapping
        ):
            if isinstance(layer, LinearBase):  # 如果是线性层 # if linear layer
                return UnquantizedLinearMethod()  # 返回未量化方法 # return unquantized method
            elif isinstance(layer, RadixAttention):  # 如果是 Radix 注意力层 # if Radix attention layer
                return QuarkKVCacheMethod(self)  # 返回 KV 缓存量化方法 # return KV cache quantization method
            return None  # 其他情况返回 None # return None for other cases

        if isinstance(layer, LinearBase):  # 如果是线性层 # if linear layer
            scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)  # 获取线性层量化方案 # get linear quantization scheme
            layer.scheme = scheme  # 将方案绑定到层 # bind scheme to layer
            return QuarkLinearMethod(self)  # 返回 Quark 线性方法 # return Quark linear method

        if isinstance(layer, RadixAttention):  # 如果是 Radix 注意力层 # if Radix attention layer
            return QuarkKVCacheMethod(self)  # 返回 KV 缓存量化方法 # return KV cache quantization method

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合 MoE 层 # import fused MoE layer

        if isinstance(layer, FusedMoE):  # 如果是融合 MoE 层 # if fused MoE layer
            layer.scheme = self.get_moe_scheme(layer, prefix)  # 获取 MoE 量化方案并绑定到层 # get MoE scheme and bind to layer
            return QuarkFusedMoEMethod(self)  # 返回 Quark 融合 MoE 方法 # return Quark fused MoE method

        return None  # 其他情况返回 None # return None for other cases

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuarkConfig":  # 从配置字典创建 QuarkConfig 实例 # create QuarkConfig from config dict
        export_config = config.get("export")  # 获取导出配置 # get export config
        if export_config is None:  # 如果导出配置不存在 # if export config is None
            raise ValueError(  # 抛出值错误 # raise value error
                "The export key should be included in "
                "the configurations of Quark quantized model"
            )  # Quark 量化模型的配置中应包含 export 键 # export key should be in Quark quantized model config

        kv_cache_group = cast(list[str], export_config.get("kv_cache_group"))  # 获取 KV 缓存分组 # get KV cache group
        pack_method = cast(str, export_config.get("pack_method"))  # 获取打包方法 # get pack method

        # In the export model of quark, the quantization configuration
        # of kv_cache is stored in layer_quant_config. First, it is
        # judged whether kv_cache_group exists, and then it is judged
        # whether layer_quant_config has a quantization configuration
        # that matches kv_cache.
        # 在 Quark 导出模型中，kv_cache 的量化配置存储在 layer_quant_config 中。
        # 首先判断 kv_cache_group 是否存在，然后判断 layer_quant_config 中
        # 是否有与 kv_cache 匹配的量化配置。
        if len(kv_cache_group) == 0:  # 如果 KV 缓存分组为空 # if KV cache group is empty
            kv_cache_config = None  # KV 缓存配置设为 None # set KV cache config to None
        else:  # 否则 # otherwise
            kv_cache_set = set(kv_cache_group)  # 将分组转为集合 # convert group to set
            layer_quant_config = cast(dict[str, Any], config.get("layer_quant_config"))  # 获取层量化配置 # get layer quant config
            layer_quant_names = list(layer_quant_config.keys())  # 获取层量化名称列表 # get layer quant names list
            layer_quant_set = set(layer_quant_names)  # 将名称转为集合 # convert names to set

            if not kv_cache_set.issubset(layer_quant_set):  # 检查 KV 缓存分组是否为层量化的子集 # check if KV cache group is subset of layer quant
                raise ValueError(  # 抛出值错误 # raise value error
                    "The Quark quantized model has the "
                    "kv_cache_group parameter setting, "
                    "but no kv_cache quantization settings "
                    "were found in the quantization "
                    "configuration."
                )  # Quark 量化模型设置了 kv_cache_group 参数，但未找到对应的量化配置 # Quark model has kv_cache_group but no matching quant config

            q_configs = [  # 获取各 KV 缓存层的量化配置 # get quant configs for each KV cache layer
                cast(dict[str, Any], layer_quant_config.get(name))  # 获取单个层的量化配置 # get single layer quant config
                for name in kv_cache_group  # 遍历 KV 缓存分组名称 # iterate over KV cache group names
            ]
            if not all(deep_compare(q_config, q_configs[0]) for q_config in q_configs):  # 检查所有配置是否一致 # check if all configs are consistent
                raise ValueError(  # 抛出值错误 # raise value error
                    "The quantization method used for kv_cache should "
                    "be the same, but the quantization method for the "
                    "kv_cache layer in the config is different."
                )  # kv_cache 使用的量化方法应相同，但配置中的方法不一致 # kv_cache quant methods should be same but differ in config
            kv_cache_config = q_configs[0].get("output_tensors")  # 获取 output_tensors 作为 KV 缓存配置 # get output_tensors as KV cache config
            if kv_cache_config is None:  # 如果配置为空 # if config is None
                raise ValueError("The kv_cache quantization configuration is empty.")  # 抛出配置为空的错误 # raise empty config error

            # Since we have already set kv_cache quantization configurations,
            # we will remove the quantization configuration for the
            # output_tensors corresponding to the kv_cache layer.
            # 由于已经设置了 kv_cache 量化配置，
            # 将移除 kv_cache 层对应的 output_tensors 量化配置。
            for q_config in q_configs:  # 遍历所有 KV 缓存量化配置 # iterate over all KV cache quant configs
                q_config["output_tensors"] = None  # 将 output_tensors 设为 None # set output_tensors to None

            # In case q_proj output is also quantized, remove the configuration
            # to keep qkv consistency.
            # 如果 q_proj 输出也被量化，则移除该配置以保持 qkv 一致性。
            q_proj_q_config = cast(dict[str, Any], layer_quant_config.get("*q_proj"))  # 获取 q_proj 量化配置 # get q_proj quant config
            if q_proj_q_config is not None:  # 如果 q_proj 配置存在 # if q_proj config exists
                q_proj_q_config["output_tensors"] = None  # 将 output_tensors 设为 None # set output_tensors to None

        return cls(  # 创建并返回 QuarkConfig 实例 # create and return QuarkConfig instance
            quant_config=config,  # 量化配置 # quantization config
            kv_cache_group=kv_cache_group,  # KV 缓存分组 # KV cache group
            kv_cache_config=kv_cache_config,  # KV 缓存配置 # KV cache config
            pack_method=pack_method,  # 打包方法 # pack method
        )

    @classmethod
    def get_config_filenames(cls) -> list[str]:  # 获取配置文件名列表 # get config filenames list
        return []  # 返回空列表 # return empty list

    def _check_scheme_supported(self, min_capability: int, error: bool = True) -> bool:  # 检查设备是否支持该量化方案 # check if device supports the quant scheme
        capability_tuple = get_device_capability()  # 获取设备算力元组 # get device capability tuple

        if capability_tuple is not None:  # 如果设备算力信息存在 # if device capability info exists
            assert 0 <= capability_tuple[1] < 10  # 断言次版本号在 0-9 范围内 # assert minor version in range 0-9
            capability = capability_tuple[0] * 10 + capability_tuple[1]  # 计算算力数值 # compute capability value

            supported = capability >= min_capability  # 判断是否满足最低算力要求 # check if meets minimum capability
            if error and not supported:  # 如果需要报错且不支持 # if error flag set and not supported
                raise RuntimeError(  # 抛出运行时错误 # raise runtime error
                    "Quantization scheme is not supported for ",
                    f"the current GPU. Min capability: {min_capability}. ",
                    f"Current capability: {capability}.",
                )  # 当前 GPU 不支持该量化方案 # current GPU does not support the quant scheme
            return supported  # 返回是否支持 # return support status
        else:  # 设备算力信息不存在 # device capability info not available
            return False  # 返回不支持 # return not supported

    def _is_fp8_w8a8(  # 判断是否为 FP8 W8A8 量化方案 # check if FP8 W8A8 quant scheme
        self,
        weight_quant: Optional[dict[str, Any]],  # 权重量化配置 # weight quant config
        input_quant: Optional[dict[str, Any]],  # 输入量化配置 # input quant config
    ) -> bool:  # 返回布尔判断结果 # return boolean result
        # Confirm weights and input quantized.
        # 确认权重和输入都已量化。
        if weight_quant is None or input_quant is None:  # 如果权重或输入未量化 # if weight or input not quantized
            return False  # 返回 False # return False

        # Confirm weight scheme is supported
        # 确认权重量化方案受支持
        is_fp8_dtype = (  # 检查数据类型是否为 fp8_e4m3 # check if dtype is fp8_e4m3
            weight_quant.get("dtype") == "fp8_e4m3"  # 权重数据类型 # weight dtype
            and input_quant.get("dtype") == "fp8_e4m3"  # 输入数据类型 # input dtype
        )
        is_static_weight = not weight_quant.get("is_dynamic")  # 检查权重是否为静态量化 # check if weight is static quantized
        is_per_tensor_or_channel_weight = weight_quant.get("qscheme") in [  # 检查权重量化方案是否为逐张量或逐通道 # check if weight qscheme is per_tensor or per_channel
            "per_tensor",  # 逐张量 # per tensor
            "per_channel",  # 逐通道 # per channel
        ]

        if not (is_fp8_dtype and is_static_weight and is_per_tensor_or_channel_weight):  # 如果不满足所有条件 # if not all conditions met
            return False  # 返回 False # return False

        # Dynamic quantization is always supported if weights supported.
        # 如果权重受支持，动态量化总是受支持。
        if input_quant.get("is_dynamic"):  # 如果输入使用动态量化 # if input uses dynamic quantization
            return True  # 返回 True # return True

        # Confirm activation scheme is supported.
        # 确认激活量化方案受支持。
        is_per_tensor_activation = input_quant.get("qscheme") == "per_tensor"  # 检查激活是否为逐张量量化 # check if activation is per_tensor quantized
        return is_per_tensor_activation  # 返回激活量化方案是否为逐张量 # return if activation is per_tensor

    def _is_mx_fp4(  # 判断是否为 MX-FP4 量化方案 # check if MX-FP4 quant scheme
        self,
        weight_quant: Optional[dict[str, Any]],  # 权重量化配置 # weight quant config
        input_quant: Optional[dict[str, Any]],  # 输入量化配置 # input quant config
    ) -> bool:  # 返回布尔判断结果 # return boolean result
        # Confirm weights and input quantized.
        # 确认权重和输入都已量化。
        if weight_quant is None or input_quant is None:  # 如果权重或输入未量化 # if weight or input not quantized
            logger.debug(  # 记录调试日志 # log debug message
                "Quark model is not in MX-FP4 format: "
                "weight_quant or input_quant not set"
            )  # Quark 模型不是 MX-FP4 格式：weight_quant 或 input_quant 未设置 # Quark model not in MX-FP4 format: weight_quant or input_quant not set
            return False  # 返回 False # return False

        # Input and weight dtype needs to be fp4.
        # 输入和权重的数据类型需要为 fp4。
        if weight_quant.get("dtype") != "fp4" or input_quant.get("dtype") != "fp4":  # 检查数据类型 # check dtype
            logger.debug("Quark model is not in MX-FP4 format: dtype not fp4")  # 记录调试日志：数据类型不是 fp4 # log debug: dtype not fp4
            return False  # 返回 False # return False

        # Input and weight qscheme needs to be per group.
        # 输入和权重的量化方案需要为逐组。
        if (  # 检查量化方案 # check quant scheme
            weight_quant.get("qscheme") != "per_group"  # 权重量化方案不是逐组 # weight qscheme is not per_group
            or input_quant.get("qscheme") != "per_group"  # 输入量化方案不是逐组 # input qscheme is not per_group
        ):
            logger.debug("Quark model is not in MX-FP4 format: not per_group")  # 记录调试日志：不是逐组量化 # log debug: not per_group
            return False  # 返回 False # return False

        # Input and weight group size needs to be 32.
        # 输入和权重的分组大小需要为 32。
        if weight_quant.get("group_size") != 32 or input_quant.get("group_size") != 32:  # 检查分组大小 # check group size
            logger.debug("Quark model is not in MX-FP4 format: not group_size=32")  # 记录调试日志：分组大小不是 32 # log debug: group_size not 32
            return False  # 返回 False # return False

        # Weights need to use static quantization.
        # 权重需要使用静态量化。
        if weight_quant.get("is_dynamic") is True:  # 如果权重使用动态量化 # if weight uses dynamic quantization
            logger.debug("Quark model is not in MX-FP4 format: not weight static")  # 记录调试日志：权重不是静态量化 # log debug: weight not static
            return False  # 返回 False # return False

        # Activations need to use dynamic quantization.
        # 激活需要使用动态量化。
        if input_quant.get("is_dynamic") is False:  # 如果激活不使用动态量化 # if activation does not use dynamic quantization
            logger.debug("Quark model is not in MX-FP4 format: not activation dynamic")  # 记录调试日志：激活不是动态量化 # log debug: activation not dynamic
            return False  # 返回 False # return False

        # Activations and weight scales need to be in e8m0 format.
        # 激活和权重的缩放因子需要使用 e8m0 格式。
        if (  # 检查缩放因子格式 # check scale format
            weight_quant.get("scale_format") != "e8m0"  # 权重缩放格式不是 e8m0 # weight scale format not e8m0
            or input_quant.get("scale_format") != "e8m0"  # 输入缩放格式不是 e8m0 # input scale format not e8m0
        ):
            logger.debug("Quark model is not in MX-FP4 format: not scale_format e8m0")  # 记录调试日志：缩放格式不是 e8m0 # log debug: scale format not e8m0
            return False  # 返回 False # return False

        return True  # 所有条件满足，返回 True # all conditions met, return True

    def _find_matched_config(  # 查找与层名匹配的量化配置 # find matched quant config for layer name
        self, layer_name: str, module: torch.nn.Module  # 层名和模块 # layer name and module
    ) -> dict[str, Any]:  # 返回匹配的量化配置 # return matched quant config

        proj_name = layer_name.split(".")[-1]  # 获取投影层名称（最后一部分） # get projection name (last part)
        if proj_name in self.packed_modules_mapping:  # 如果投影名称在打包模块映射中 # if proj name in packed modules mapping
            shard_proj_names = self.packed_modules_mapping[proj_name]  # 获取分片投影名称列表 # get shard projection names

            # Convert fused_name --> [shard_names]
            # 将融合名称转换为分片名称列表
            shard_names = [  # 生成分片层名 # generate shard layer names
                layer_name.replace(proj_name, shard_proj_name)  # 替换投影名称为分片名称 # replace proj name with shard name
                for shard_proj_name in shard_proj_names  # 遍历分片投影名称 # iterate over shard proj names
            ]
            shard_configs = [  # 获取各分片的量化配置 # get quant configs for each shard
                self._find_matched_config(shard_name, module)  # 递归查找分片配置 # recursively find shard config
                for shard_name in shard_names  # 遍历分片名称 # iterate over shard names
            ]
            if not all(  # 检查所有分片配置是否一致 # check if all shard configs are consistent
                deep_compare(q_config, shard_configs[0]) for q_config in shard_configs  # 深度比较各配置 # deep compare each config
            ):
                raise ValueError(  # 抛出值错误 # raise value error
                    f"Found a different quantization configuration for "
                    f"{shard_proj_names} in {layer_name}. vLLM "
                    "requires all to use the same scheme."
                )  # 发现不同的量化配置，要求所有分片使用相同方案 # found different quant configs, all shards must use same scheme
            return shard_configs[0]  # 返回第一个分片的配置 # return first shard config
        else:  # 非打包模块 # non-packed module
            layer_quant_config = cast(  # 获取层量化配置 # get layer quant config
                dict[str, Any], self.quant_config.get("layer_quant_config")  # 从量化配置中获取 # get from quant config
            )
            for name_pattern in layer_quant_config:  # 遍历层量化配置的模式 # iterate over layer quant config patterns
                if fnmatch.fnmatch(layer_name, name_pattern):  # 如果层名匹配模式 # if layer name matches pattern
                    return layer_quant_config[name_pattern]  # 返回匹配的配置 # return matched config

            layer_type = type(module).__name__  # 获取模块类型名称 # get module type name
            layer_type_quant_config = cast(  # 获取层类型量化配置 # get layer type quant config
                dict[str, Any], self.quant_config.get("layer_type_quant_config")  # 从量化配置中获取 # get from quant config
            )
            if layer_type in layer_type_quant_config:  # 如果层类型在配置中 # if layer type in config
                return layer_type_quant_config[layer_type]  # 返回类型匹配的配置 # return type-matched config

            global_quant_config = cast(  # 获取全局量化配置 # get global quant config
                dict[str, Any], self.quant_config.get("global_quant_config")  # 从量化配置中获取 # get from quant config
            )
            return global_quant_config  # 返回全局量化配置 # return global quant config

    def _get_scheme_from_config(self, config: dict[str, Any]) -> "QuarkLinearScheme":  # 从配置获取量化方案 # get quant scheme from config
        if config.get("output_tensors") or config.get("bias"):  # 如果配置包含 output_tensors 或 bias # if config has output_tensors or bias
            raise NotImplementedError(  # 抛出未实现错误 # raise not implemented error
                "Currently, Quark models with output_tensors "
                "and bias quantized are not supported"
            )  # 当前不支持 output_tensors 和 bias 量化的 Quark 模型 # Quark models with output_tensors and bias quantized not supported
        weight_config = cast(dict[str, Any], config.get("weight"))  # 获取权重量化配置 # get weight quant config
        input_config = cast(dict[str, Any], config.get("input_tensors"))  # 获取输入量化配置 # get input quant config

        if self._is_mx_fp4(weight_config, input_config):  # 如果是 MX-FP4 方案 # if MX-FP4 scheme
            return QuarkW4A4MXFP4(weight_config, input_config)  # 返回 W4A4 MX-FP4 方案实例 # return W4A4 MX-FP4 scheme instance
        if self._is_fp8_w8a8(weight_config, input_config):  # 如果是 FP8 W8A8 方案 # if FP8 W8A8 scheme
            is_fp8_w8a8_supported = self._check_scheme_supported(  # 检查 FP8 W8A8 是否受设备支持 # check if FP8 W8A8 is supported by device
                QuarkW8A8Fp8.get_min_capability(), error=False  # 不抛出错误，仅检查 # don't raise error, just check
            )
            if is_fp8_w8a8_supported:  # 如果设备支持 FP8 W8A8 # if device supports FP8 W8A8
                return QuarkW8A8Fp8(weight_config, input_config)  # 返回 W8A8 FP8 方案实例 # return W8A8 FP8 scheme instance

        raise NotImplementedError(  # 抛出未实现错误 # raise not implemented error
            "No quark compatible scheme was found. "
            f"Weight config: {weight_config}, "
            f"Input config: {input_config}"
        )  # 未找到兼容的 Quark 量化方案 # no compatible Quark quant scheme found

    def get_linear_scheme(  # 获取线性层量化方案 # get linear layer quant scheme
        self, layer: torch.nn.Module, layer_name: str  # 神经网络层和层名 # neural network layer and layer name
    ) -> "QuarkLinearScheme":  # 返回 Quark 线性量化方案 # return Quark linear scheme

        layer_quant_config = self._find_matched_config(layer_name, layer)  # 查找匹配的层量化配置 # find matched layer quant config

        # Find the quant_scheme
        # 查找量化方案
        scheme = self._get_scheme_from_config(layer_quant_config)  # 从配置获取量化方案 # get quant scheme from config

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        # 如果设备不支持该方案则抛出错误（例如 fp8 需要 Ada Lovelace 架构）
        self._check_scheme_supported(scheme.get_min_capability())  # 检查方案是否受设备支持 # check if scheme is supported by device

        return scheme  # 返回量化方案 # return quant scheme

    def get_moe_scheme(  # 获取 MoE 量化方案 # get MoE quant scheme
        self,
        module: torch.nn.Module,  # 神经网络模块 # neural network module
        layer_name: str,  # 层名 # layer name
    ) -> "QuarkMoEScheme":  # 返回 Quark MoE 量化方案 # return Quark MoE scheme
        layer_quant_config = self._find_matched_config(layer_name, module)  # 查找匹配的层量化配置 # find matched layer quant config

        if layer_quant_config.get("output_tensors") or layer_quant_config.get("bias"):  # 如果配置包含 output_tensors 或 bias # if config has output_tensors or bias
            raise NotImplementedError(  # 抛出未实现错误 # raise not implemented error
                "Currently, Quark models with "
                "output_tensors and bias "
                "quantized are not supported"
            )  # 当前不支持 output_tensors 和 bias 量化的 Quark 模型 # Quark models with output_tensors and bias quantized not supported
        weight_config = layer_quant_config.get("weight")  # 获取权重量化配置 # get weight quant config
        input_config = layer_quant_config.get("input_tensors")  # 获取输入量化配置 # get input quant config

        if self._is_mx_fp4(weight_config, input_config):  # 如果是 MX-FP4 方案 # if MX-FP4 scheme
            return QuarkW4A4MXFp4MoE(weight_config, input_config)  # 返回 W4A4 MX-FP4 MoE 方案实例 # return W4A4 MX-FP4 MoE scheme instance
        elif self._is_fp8_w8a8(weight_config, input_config):  # 如果是 FP8 W8A8 方案 # if FP8 W8A8 scheme
            return QuarkW8A8FP8MoE(weight_config, input_config)  # 返回 W8A8 FP8 MoE 方案实例 # return W8A8 FP8 MoE scheme instance
        else:  # 不支持的方案 # unsupported scheme
            raise RuntimeError("Unsupported FusedMoe scheme")  # 抛出不支持的融合 MoE 方案错误 # raise unsupported fused MoE scheme error

    def get_scaled_act_names(self) -> List[str]:  # 获取需要缩放的激活名称列表 # get scaled activation names list
        return []  # 返回空列表 # return empty list


class QuarkLinearMethod(LinearMethodBase):  # Quark 线性层量化方法类，继承自 LinearMethodBase # Quark linear method class

    def __init__(self, quantization_config: QuarkConfig):  # 初始化方法 # initializer
        self.quantization_config = quantization_config  # 保存量化配置 # save quantization config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后处理 # process weights after loading
        layer.scheme.process_weights_after_loading(layer)  # 调用层绑定的方案的后处理方法 # call scheme's post-processing method

    def create_weights(  # 创建线性层量化权重 # create linear layer quantized weights
        self,
        layer: torch.nn.Module,  # 目标神经网络层 # target neural network layer
        input_size_per_partition: int,  # 每个分区的输入大小 # input size per partition
        output_partition_sizes: list[int],  # 输出分区大小列表 # output partition sizes list
        input_size: int,  # 总输入大小 # total input size
        output_size: int,  # 总输出大小 # total output size
        params_dtype: torch.dtype,  # 参数数据类型 # parameter data type
        **extra_weight_attrs,  # 额外权重属性 # extra weight attributes
    ):
        """
        Use the QuarkLinearScheme associated with the layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        使用与层关联的 QuarkLinearScheme 创建该层所需的参数。
        参数详情参见 LinearMethodBase
        """
        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器 # get weight loader
        layer.scheme.create_weights(  # 调用层绑定的方案创建权重 # call scheme's create_weights method
            layer=layer,  # 目标层 # target layer
            input_size=input_size,  # 总输入大小 # total input size
            input_size_per_partition=input_size_per_partition,  # 每分区输入大小 # input size per partition
            output_partition_sizes=output_partition_sizes,  # 输出分区大小 # output partition sizes
            output_size=output_size,  # 总输出大小 # total output size
            params_dtype=params_dtype,  # 参数数据类型 # parameter data type
            weight_loader=weight_loader,  # 权重加载器 # weight loader
        )

    def apply(  # 应用量化权重进行前向计算 # apply quantized weights for forward computation
        self,
        layer: torch.nn.Module,  # 目标神经网络层 # target neural network layer
        x: torch.Tensor,  # 输入张量 # input tensor
        bias: Optional[torch.Tensor] = None,  # 偏置张量（可选） # bias tensor (optional)
    ):
        """
        Use the output of create_weights and the QuarkLinearScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        使用 create_weights 的输出和与层关联的 QuarkLinearScheme
        对层输入执行前向传播。参数详情参见 LinearMethodBase
        """
        scheme = layer.scheme  # 获取层绑定的量化方案 # get scheme bound to layer
        if scheme is None:  # 如果方案未定义 # if scheme is not defined
            raise ValueError("A scheme must be defined for each layer")  # 抛出方案必须定义的错误 # raise error: scheme must be defined
        return scheme.apply_weights(layer, x, bias=bias)  # 调用方案的 apply_weights 方法 # call scheme's apply_weights method


class QuarkFusedMoEMethod(FusedMoEMethodBase):  # Quark 融合 MoE 量化方法类，继承自 FusedMoEMethodBase # Quark fused MoE method class

    def __init__(self, quantization_config: QuarkConfig):  # 初始化方法 # initializer
        self.quantization_config = quantization_config  # 保存量化配置 # save quantization config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后处理 # process weights after loading
        layer.scheme.process_weights_after_loading(layer)  # 调用层绑定的方案的后处理方法 # call scheme's post-processing method

    def create_weights(  # 创建 MoE 层量化权重 # create MoE layer quantized weights
        self,
        layer: torch.nn.Module,  # 目标神经网络层 # target neural network layer
        num_experts: int,  # 专家数量 # number of experts
        hidden_size: int,  # 隐藏层大小 # hidden size
        intermediate_size_per_partition: int,  # 每个分区的中间层大小 # intermediate size per partition
        params_dtype: torch.dtype,  # 参数数据类型 # parameter data type
        **extra_weight_attrs,  # 额外权重属性 # extra weight attributes
    ):
        """
        Use the QuarkMoEScheme associated with the layer to create
        the necessary parameters for the layer. See FusedMoEMethodBase for param
        details
        使用与层关联的 QuarkMoEScheme 创建该层所需的参数。
        参数详情参见 FusedMoEMethodBase
        """
        layer.scheme.create_weights(  # 调用层绑定的方案创建权重 # call scheme's create_weights method
            layer=layer,  # 目标层 # target layer
            num_experts=num_experts,  # 专家数量 # number of experts
            hidden_size=hidden_size,  # 隐藏层大小 # hidden size
            intermediate_size_per_partition=intermediate_size_per_partition,  # 每分区中间层大小 # intermediate size per partition
            params_dtype=params_dtype,  # 参数数据类型 # parameter data type
            **extra_weight_attrs,  # 额外权重属性 # extra weight attributes
        )

    def create_moe_runner(  # 创建 MoE 运行器 # create MoE runner
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 目标层和 MoE 运行器配置 # target layer and MoE runner config
    ):
        layer.scheme.create_moe_runner(layer, moe_runner_config)  # 调用层绑定的方案创建 MoE 运行器 # call scheme's create_moe_runner method

    def apply(  # 应用量化权重进行 MoE 前向计算 # apply quantized weights for MoE forward computation
        self,
        layer: torch.nn.Module,  # 目标神经网络层 # target neural network layer
        dispatch_output: "StandardDispatchOutput",  # 标准分发输出 # standard dispatch output
    ):
        """
        Use the output of create_weights and the QuarkMoEScheme
        associated with the layer to apply the forward pass with the
        fused MoE layer. See FusedMoEMethodBase for param details

        使用 create_weights 的输出和与层关联的 QuarkMoEScheme
        对融合 MoE 层执行前向传播。参数详情参见 FusedMoEMethodBase
        """
        scheme = layer.scheme  # 获取层绑定的量化方案 # get scheme bound to layer
        if scheme is None:  # 如果方案未定义 # if scheme is not defined
            raise ValueError("A scheme must be defined for each layer")  # 抛出方案必须定义的错误 # raise error: scheme must be defined
        return scheme.apply_weights(layer, dispatch_output)  # 调用方案的 apply_weights 方法 # call scheme's apply_weights method


class QuarkKVCacheMethod(BaseKVCacheMethod):  # Quark KV 缓存量化方法类，继承自 BaseKVCacheMethod # Quark KV cache method class
    """
    Supports loading kv-cache scaling factors from quark checkpoints.
    支持从 Quark 检查点加载 KV 缓存缩放因子。
    """

    def __init__(self, quant_config: QuarkConfig):  # 初始化方法 # initializer
        self.validate_kv_cache_config(quant_config.kv_cache_config)  # 验证 KV 缓存配置 # validate KV cache config
        super().__init__(quant_config)  # 调用父类初始化 # call parent class initializer

    @staticmethod
    def validate_kv_cache_config(kv_cache_config: Optional[dict[str, Any]]):  # 验证 KV 缓存配置 # validate KV cache config
        """
        Validator for the kv cache configuration. Useful for controlling the
        kv cache quantization schemes, that are being supported in vLLM
        :param kv_cache_config: the quark kv cache scheme
        KV 缓存配置验证器。用于控制 vLLM 中支持的 KV 缓存量化方案
        :param kv_cache_config: Quark KV 缓存方案
        """
        if kv_cache_config is None:  # 如果配置为空 # if config is None
            return  # 直接返回 # return directly

        dtype = kv_cache_config.get("dtype")  # 获取数据类型 # get dtype
        if dtype != "fp8_e4m3":  # 如果数据类型不是 fp8_e4m3 # if dtype is not fp8_e4m3
            raise NotImplementedError(  # 抛出未实现错误 # raise not implemented error
                "Currently supported kv cache quantization is "
                f"dtype=fp8_e4m3, however received {dtype}"
            )  # 当前仅支持 fp8_e4m3 类型的 KV 缓存量化 # only fp8_e4m3 KV cache quantization is supported

        qscheme = kv_cache_config.get("qscheme")  # 获取量化方案 # get quant scheme
        if qscheme != "per_tensor":  # 如果量化方案不是逐张量 # if qscheme is not per_tensor
            raise NotImplementedError(  # 抛出未实现错误 # raise not implemented error
                "Only support per-tensor scaling factor "
                "for quark KV cache. "
                f"Expected qscheme: per_tensor, found qscheme: {qscheme}"
            )  # Quark KV 缓存仅支持逐张量缩放因子 # Quark KV cache only supports per-tensor scaling factor
