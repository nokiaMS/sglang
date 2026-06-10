# AutoRound 量化配置模块
# 本文件实现了 AutoRound 量化方法的配置类，支持 GPTQ 和 AWQ 后端，
# 可根据模型层的类型和配置自动选择合适的量化方案。
# 参考: https://arxiv.org/pdf/2309.05516

# SPDX-License-Identifier: Apache-2.0

import logging  # 导入日志模块
import re  # 导入正则表达式模块
from fractions import Fraction  # 导入分数类型，用于精确计算打包因子
from typing import Any, Optional, Union  # 导入类型注解工具

import torch  # 导入 PyTorch 深度学习框架

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

from sglang.srt.layers.quantization.utils import get_scalar_types  # 导入标量类型获取工具

ScalarType, scalar_types = get_scalar_types()  # 获取标量类型及其注册表

from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.utils import is_npu  # 导入 NPU 检测工具

_is_npu = is_npu()  # 检测当前是否在 NPU 上运行


class AutoRoundConfig(QuantizationConfig):  # AutoRound 量化配置类，继承自 QuantizationConfig
    """Config class for AutoRound.  # AutoRound 配置类
    Reference: https://arxiv.org/pdf/2309.05516  # 参考论文链接
    """  # 类文档字符串结束

    SUPPORTED_BITS = {2, 3, 4, 8}  # 支持的权重量化位数
    SUPPORTED_DTYPES = {"int"}  # 支持的数据类型
    SUPPORTED_FORMATS = {"auto_round:auto_gptq", "auto_round:auto_awq"}  # 支持的打包格式
    SUPPORTED_BACKENDS = {"auto", "gptq", "gptq:marlin", "awq", "awq:marlin", "marlin"}  # 支持的后端

    def __init__(  # 初始化方法
        self,
        weight_bits: int,  # 权重量化位数
        group_size: int,  # 量化分组大小
        sym: bool = True,  # 是否使用对称量化，默认为 True
        packing_format: str = "auto_round:auto_gptq",  # 打包格式，默认为 auto_gptq
        block_name_to_quantize: Optional[Union[str, list[str]]] = None,  # 需要量化的块名称
        extra_config: Optional[dict[str, Any]] = None,  # 额外配置字典
        data_type: str = "int",  # 数据类型，默认为 int
        backend: str = "auto",  # 量化后端，默认为 auto
    ) -> None:
        super().__init__()  # 调用父类初始化
        if weight_bits not in self.SUPPORTED_BITS:  # 检查量化位数是否支持
            raise ValueError(  # 抛出不支持的位数错误
                f"Unsupported weight_bits: {weight_bits}, "  # 不支持的权重量化位数
                f"currently only support  {self.SUPPORTED_BITS}"  # 当前仅支持
            )
        if data_type not in self.SUPPORTED_DTYPES:  # 检查数据类型是否支持
            raise ValueError(  # 抛出不支持的数据类型错误
                f"Unsupported data_type: {data_type},"  # 不支持的数据类型
                f" currently only support  {self.SUPPORTED_DTYPES}"  # 当前仅支持
            )
        if packing_format not in self.SUPPORTED_FORMATS:  # 检查打包格式是否支持
            raise ValueError(  # 抛出不支持的打包格式错误
                f"Unsupported packing_format: {packing_format}, "  # 不支持的打包格式
                f"currently only support  {self.SUPPORTED_FORMATS}"  # 当前仅支持
            )
        if backend not in self.SUPPORTED_BACKENDS:  # 检查后端是否支持
            raise ValueError(  # 抛出不支持的后端错误
                f"Unsupported backend: {backend},  "  # 不支持的后端
                f"currently only support  {self.SUPPORTED_BACKENDS}"  # 当前仅支持
            )

        self.weight_bits = weight_bits  # 保存权重量化位数
        self.group_size = group_size  # 保存量化分组大小
        self.sym = sym  # 保存是否对称量化标志
        self.packing_format = packing_format  # 保存打包格式
        self.block_name_to_quantize = (  # 处理需要量化的块名称
            block_name_to_quantize.split(",")  # 如果是字符串，按逗号分割
            if isinstance(block_name_to_quantize, str)  # 判断是否为字符串
            else block_name_to_quantize  # 否则直接使用列表
        )
        self.extra_config = extra_config  # 保存额外配置
        self.data_type = data_type  # 保存数据类型
        self.backend = backend  # 保存量化后端
        self.pack_factor = Fraction(32, weight_bits)  # 计算打包因子，32位除以量化位数

    def __repr__(self) -> str:  # 返回配置对象的字符串表示
        return (  # 返回格式化字符串
            f"AutoRoundConfig(weight_bits={self.weight_bits}, "  # 权重位数
            f"group_size={self.group_size}, sym={self.sym})"  # 分组大小和对称标志
        )

    @classmethod
    def get_name(cls):  # 获取量化方法名称
        return "auto-round"  # 返回名称

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:  # 获取支持的激活数据类型
        return [torch.half, torch.bfloat16]  # 支持 float16 和 bfloat16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低 GPU 计算能力要求
        return 60  # 最低需要计算能力 6.0

    @classmethod
    def get_config_filenames(cls) -> list[str]:  # 获取配置文件名列表
        return ["quantization_config.json"]  # 量化配置文件名

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoRoundConfig":  # 从配置字典创建配置对象
        return cls(  # 返回新创建的配置对象
            weight_bits=cls.get_from_keys(config, ["bits"]),  # 获取量化位数
            group_size=cls.get_from_keys(config, ["group_size"]),  # 获取分组大小
            sym=cls.get_from_keys(config, ["sym"]),  # 获取是否对称量化
            packing_format=cls.get_from_keys_or(  # 获取打包格式
                config,
                ["packing_format"],
                "auto_round:auto_gptq",  # 默认值
            ),
            block_name_to_quantize=cls.get_from_keys_or(  # 获取需量化的块名称
                config, ["block_name_to_quantize", "to_quant_block_names"], None  # 支持两个键名
            ),
            extra_config=cls.get_from_keys_or(config, ["extra_config"], None),  # 获取额外配置
            data_type=cls.get_from_keys_or(config, ["data_type"], "int"),  # 获取数据类型
            backend=cls.get_from_keys_or(  # 获取量化后端
                config, ["backend", "vllm_backend", "sglang_backend"], "auto"  # 支持多个键名
            ),
        )

    def get_scaled_act_names(self) -> list[str]:  # 获取需要后缩放的激活函数名
        """Returns the activation function names that should be post-scaled.  # 返回需要后缩放的激活函数名

        For now, this is only used by AWQ.  # 目前仅 AWQ 使用
        """
        raise NotImplementedError  # 尚未实现

    def get_layer_config(self, layer, layer_name: str):  # 获取指定层的量化配置
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行词嵌入层

        def get_config(name: str, quantized: bool = True):  # 内部函数：根据名称获取配置
            if not self.extra_config:  # 如果没有额外配置
                return (  # 返回默认配置
                    self.weight_bits if quantized else 16,  # 量化位数或16位
                    self.group_size if quantized else -1,  # 分组大小或-1
                    self.sym if quantized else True,  # 对称标志或True
                )

            # Exact match first  # 先精确匹配
            if name in self.extra_config:  # 如果名称在额外配置中
                cfg = self.extra_config[name]  # 获取对应配置
                return (  # 返回匹配的配置
                    cfg.get("bits", self.weight_bits if quantized else 16),  # 位数
                    cfg.get("group_size", self.group_size if quantized else -1),  # 分组大小
                    cfg.get("sym", self.sym if quantized else True),  # 对称标志
                )

            REGEX_SPECIAL_CHARS = set(r"*+?^$()[]{}|\\")  # 正则表达式特殊字符集合
            for pattern, cfg in self.extra_config.items():  # 遍历额外配置项
                if not isinstance(pattern, str) or not any(  # 如果不是字符串或不包含
                    c in REGEX_SPECIAL_CHARS for c in pattern  # 正则特殊字符
                ):
                    continue  # 跳过非正则模式

                try:  # 尝试正则匹配
                    if re.fullmatch(pattern, name):  # 如果完全匹配
                        return (  # 返回正则匹配的配置
                            cfg.get("bits", self.weight_bits if quantized else 16),  # 位数
                            cfg.get("group_size", self.group_size if quantized else -1),  # 分组大小
                            cfg.get("sym", self.sym if quantized else True),  # 对称标志
                        )
                except re.error:  # 捕获正则表达式错误
                    # Invalid regex, ignore.  # 无效正则，忽略
                    continue  # 继续下一个

            return (  # 返回默认配置
                self.weight_bits if quantized else 16,  # 量化位数或16位
                self.group_size if quantized else -1,  # 分组大小或-1
                self.sym if quantized else True,  # 对称标志或True
            )

        # 1. Exact match from config  # 1. 从配置中精确匹配
        if self.extra_config and layer_name in self.extra_config:  # 如果额外配置中存在该层名
            return get_config(layer_name)  # 返回精确匹配的配置

        # 2. Determine whether layer should be quantized  # 2. 判断该层是否需要量化
        quantized = not isinstance(layer, ParallelLMHead)  # 默认非词嵌入层需要量化
        if self.block_name_to_quantize:  # 如果指定了需要量化的块名
            quantized = any(  # 判断层名是否以任一指定块名开头
                layer_name.startswith(name) for name in self.block_name_to_quantize
            )

        # 3. Handle fused MoE  # 3. 处理融合 MoE 层
        if self.extra_config and "fusedmoe" in layer.__class__.__name__.lower():  # 如果是融合MoE层且有额外配置
            moe_configs = [  # 收集所有子层的配置
                get_config(name, quantized)  # 获取子层配置
                for name in self.extra_config  # 遍历额外配置
                if name.startswith(layer_name)  # 匹配以当前层名开头的配置
            ]
            if moe_configs:  # 如果找到了子层配置
                if len(set(moe_configs)) == 1:  # 如果所有子层配置一致
                    return moe_configs[0]  # 返回统一的配置
                raise ValueError(  # 子层配置不一致时报错
                    f"Fused MoE layer '{layer_name}' requires "  # 融合MoE层需要
                    f"consistent quant config for all sub-layers"  # 所有子层一致的量化配置
                )

        # 4. Handle fused QKV or other patterns  # 4. 处理融合 QKV 或其他融合模式
        if self.extra_config:  # 如果有额外配置
            for fusion_key, sub_keys in self.packed_modules_mapping.items():  # 遍历打包模块映射
                if fusion_key in layer_name and layer_name.count(fusion_key) == 1:  # 如果层名包含融合键且仅出现一次
                    sub_names = [  # 生成子模块名称列表
                        layer_name.replace(fusion_key, sub_key) for sub_key in sub_keys  # 替换融合键为子键
                    ]
                    sub_configs = [get_config(name, quantized) for name in sub_names]  # 获取子模块配置
                    if len(set(sub_configs)) == 1:  # 如果所有子模块配置一致
                        return sub_configs[0]  # 返回统一配置
                    raise ValueError(  # 子模块配置不一致时报错
                        f"Fused module '{layer_name}' requires "  # 融合模块需要
                        f"consistent quant config for {sub_names}"  # 子模块一致的量化配置
                    )

        # 5. Fallback or try a regular expression match  # 5. 回退或尝试正则匹配
        return get_config(layer_name, quantized)  # 返回层配置

    def check_quantized(self, weight_bits: int) -> bool:  # 检查权重是否已量化
        return weight_bits < 16  # 位数小于16即为已量化

    def apply_awq_quant_layer(self, layer, prefix: str, backend: str = "auto"):  # 对层应用 AWQ 量化方法
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合 MoE 层
        from sglang.srt.layers.quantization.marlin_utils import (  # 导入 Marlin 工具
            check_marlin_supported,  # 检查 Marlin 支持的函数
            check_moe_marlin_supports_layer,  # 检查 MoE Marlin 支持的函数
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 导入未量化线性方法
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行词嵌入层

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)  # 获取当前层的量化配置
        if not self.check_quantized(weight_bits):  # 如果层不需要量化
            if isinstance(layer, (LinearBase, ParallelLMHead)):  # 如果是线性层或词嵌入层
                return UnquantizedLinearMethod()  # 返回未量化的线性方法
            else:
                return None  # 其他层返回 None
        logger.debug(  # 记录调试日志
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",  # 日志格式
            prefix,  # 层前缀
            layer.__class__.__name__,  # 层类型名
            weight_bits,  # 量化位数
            group_size,  # 分组大小
            sym,  # 对称标志
        )
        if backend == "auto" or "marlin" in backend:  # 自动选择后端或使用 Marlin
            AWQ_TYPE_MAP = {  # AWQ 类型映射
                4: scalar_types.uint4,  # 4位对应 uint4
                8: scalar_types.uint8,  # 8位对应 uint8
            }
            use_marlin = (weight_bits in AWQ_TYPE_MAP) and check_marlin_supported(  # 检查是否可以使用 Marlin
                AWQ_TYPE_MAP[weight_bits], group_size, not sym  # 传入类型、分组大小和零点标志
            )
            if isinstance(layer, FusedMoE):  # 如果是融合 MoE 层
                use_marlin = use_marlin and check_moe_marlin_supports_layer(  # 额外检查 MoE Marlin 支持
                    layer, group_size  # 传入层和分组大小
                )

        else:
            use_marlin = False  # 不使用 Marlin 后端
        if use_marlin:  # 如果使用 Marlin 后端
            from sglang.srt.layers.quantization.awq import (  # 导入 AWQ 相关类
                AWQLinearMethod,  # AWQ 线性方法
                AWQMarlinConfig,  # AWQ Marlin 配置
                AWQMoEMethod,  # AWQ MoE 方法
            )

            quant_args_marlin = AWQMarlinConfig(  # 创建 AWQ Marlin 配置
                weight_bits=weight_bits,  # 权重位数
                group_size=group_size,  # 分组大小
                zero_point=not sym,  # 零点标志（非对称时有零点）
                lm_head_quantized=False,  # 语言模型头不量化
                full_config={},  # 完整配置为空
                modules_to_not_convert=[],  # 不转换的模块为空
            )
        else:  # 不使用 Marlin 时
            from sglang.srt.layers.quantization.awq import AWQConfig, AWQLinearMethod  # 导入 AWQ 配置和线性方法

            quant_args = AWQConfig(  # 创建 AWQ 配置
                weight_bits=weight_bits,  # 权重位数
                group_size=group_size,  # 分组大小
                zero_point=not sym,  # 零点标志
            )

        if isinstance(layer, FusedMoE):  # 如果是融合 MoE 层
            if use_marlin:  # 使用 Marlin 后端
                layer.scheme = quant_args_marlin.get_moe_scheme(layer)  # 设置 MoE 量化方案
                return AWQMoEMethod(quant_args_marlin)  # 返回 AWQ MoE 方法
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config  # 导入 MoE WNA16 配置

            config = {  # 构建 MoE 量化配置字典
                "quant_method": "awq",  # 量化方法为 AWQ
                "bits": weight_bits,  # 量化位数
                "group_size": group_size,  # 分组大小
                "zero_point": not sym,  # 零点标志
                "lm_head": False,  # 语言模型头不量化
            }
            return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)  # 通过 MoeWNA16 配置获取量化方法

        if isinstance(layer, (LinearBase, ParallelLMHead)):  # 如果是线性层或词嵌入层
            if use_marlin:  # 使用 Marlin 后端
                layer.scheme = quant_args_marlin.get_linear_scheme(layer)  # 设置线性层量化方案
                return AWQLinearMethod(quant_args_marlin)  # 返回 AWQ Marlin 线性方法
            else:  # 不使用 Marlin
                layer.scheme = quant_args.get_linear_scheme(layer)  # 设置线性层量化方案
                return AWQLinearMethod(quant_args)  # 返回 AWQ 线性方法
        return None  # 不支持的层类型返回 None

    def apply_gptq_quant_layer(self, layer, prefix: str, backend: str = "auto"):  # 对层应用 GPTQ 量化方法
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合 MoE 层
        from sglang.srt.layers.quantization.gptq import (  # 导入 GPTQ 相关类
            GPTQAscendConfig,  # GPTQ Ascend 配置
            GPTQLinearAscendMethod,  # GPTQ Ascend 线性方法
            GPTQMoEAscendMethod,  # GPTQ Ascend MoE 方法
        )
        from sglang.srt.layers.quantization.marlin_utils import (  # 导入 Marlin 工具
            check_marlin_supported,  # 检查 Marlin 支持
            check_moe_marlin_supports_layer,  # 检查 MoE Marlin 支持
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 导入未量化线性方法
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行词嵌入层

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)  # 获取当前层的量化配置
        if not self.check_quantized(weight_bits):  # 如果层不需要量化
            if isinstance(layer, (LinearBase, ParallelLMHead)):  # 如果是线性层或词嵌入层
                return UnquantizedLinearMethod()  # 返回未量化的线性方法
            else:
                return None  # 其他层返回 None

        logger.debug(  # 记录调试日志
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",  # 日志格式
            prefix,  # 层前缀
            layer.__class__.__name__,  # 层类型名
            weight_bits,  # 量化位数
            group_size,  # 分组大小
            sym,  # 对称标志
        )
        if _is_npu:  # 如果运行在 NPU 上
            quant_args = GPTQAscendConfig(  # 创建 GPTQ Ascend 配置
                weight_bits=weight_bits,  # 权重位数
                group_size=group_size,  # 分组大小
                lm_head_quantized=False,  # 语言模型头不量化
                desc_act=False,  # 不使用描述符激活
                dynamic={},  # 动态配置为空
            )
            quant_args.sym = sym  # 设置对称标志

            if isinstance(layer, FusedMoE):  # 如果是融合 MoE 层
                return GPTQMoEAscendMethod(quant_args)  # 返回 GPTQ Ascend MoE 方法

            if isinstance(layer, (LinearBase, ParallelLMHead)):  # 如果是线性层或词嵌入层
                return GPTQLinearAscendMethod(quant_args)  # 返回 GPTQ Ascend 线性方法

            return None  # 不支持的层返回 None

        if backend == "auto" or "marlin" in backend:  # 自动选择后端或使用 Marlin
            GPTQ_TYPE_MAP = {  # GPTQ 类型映射
                (4, True): scalar_types.uint4b8,  # 4位对称对应 uint4b8
                (8, True): scalar_types.uint8b128,  # 8位对称对应 uint8b128
            }
            use_marlin = (weight_bits, sym) in GPTQ_TYPE_MAP and check_marlin_supported(  # 检查是否可以使用 Marlin
                GPTQ_TYPE_MAP[(weight_bits, sym)], group_size, has_zp=not sym  # 传入类型、分组大小和零点标志
            )
            if isinstance(layer, FusedMoE):  # 如果是融合 MoE 层
                use_marlin = use_marlin and check_moe_marlin_supports_layer(  # 额外检查 MoE Marlin 支持
                    layer, group_size  # 传入层和分组大小
                )
        else:
            use_marlin = False  # 不使用 Marlin 后端
        if use_marlin:  # 如果使用 Marlin 后端
            from sglang.srt.layers.quantization.gptq import (  # 导入 GPTQ Marlin 相关类
                GPTQMarlinConfig,  # GPTQ Marlin 配置
                GPTQMarlinLinearMethod,  # GPTQ Marlin 线性方法
                GPTQMarlinMoEMethod,  # GPTQ Marlin MoE 方法
            )

            quant_args_marlin = GPTQMarlinConfig(  # 创建 GPTQ Marlin 配置
                weight_bits=weight_bits,  # 权重位数
                group_size=group_size,  # 分组大小
                is_sym=sym,  # 是否对称量化
                lm_head_quantized=False,  # 语言模型头不量化
                desc_act=False,  # 不使用描述符激活
                dynamic={},  # 动态配置为空
                full_config={},  # 完整配置为空
            )
        else:  # 不使用 Marlin 时
            from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQLinearMethod  # 导入 GPTQ 配置和线性方法

            quant_args = GPTQConfig(  # 创建 GPTQ 配置
                weight_bits=weight_bits,  # 权重位数
                group_size=group_size,  # 分组大小
                lm_head_quantized=False,  # 语言模型头不量化
                desc_act=False,  # 不使用描述符激活
                dynamic={},  # 动态配置为空
            )

        if isinstance(layer, FusedMoE):  # 如果是融合 MoE 层
            if use_marlin:  # 使用 Marlin 后端
                from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config  # 导入 MoE WNA16 配置

                config = {  # 构建 MoE 量化配置字典
                    "quant_method": "gptq",  # 量化方法为 GPTQ
                    "bits": weight_bits,  # 量化位数
                    "group_size": group_size,  # 分组大小
                    "sym": sym,  # 对称标志
                    "lm_head": False,  # 语言模型头不量化
                }
                return MoeWNA16Config.from_config(config).get_quant_method(  # 通过 MoeWNA16 配置获取量化方法
                    layer, prefix  # 传入层和前缀
                )
            return GPTQMarlinMoEMethod(quant_args_marlin)  # 返回 GPTQ Marlin MoE 方法

        if isinstance(layer, (LinearBase, ParallelLMHead)):  # 如果是线性层或词嵌入层
            if use_marlin:  # 使用 Marlin 后端
                return GPTQMarlinLinearMethod(quant_args_marlin)  # 返回 GPTQ Marlin 线性方法
            else:  # 不使用 Marlin
                return GPTQLinearMethod(quant_args)  # 返回 GPTQ 线性方法

        return None  # 不支持的层类型返回 None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):  # 获取适用于给定层的量化方法
        # TODO enable CPU quant method later  # TODO 后续启用 CPU 量化方法
        if "gptq" in self.packing_format or "gptq" in self.backend:  # 如果打包格式或后端包含 gptq
            return self.apply_gptq_quant_layer(layer, prefix)  # 应用 GPTQ 量化
        if "awq" in self.packing_format or "awq" in self.backend:  # 如果打包格式或后端包含 awq
            return self.apply_awq_quant_layer(layer, prefix)  # 应用 AWQ 量化
