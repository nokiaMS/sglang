# AWQ量化配置与方法实现模块
# 实现了AWQ、AWQ-Marlin等量化配置类及对应的线性层/MoE量化方法
# 支持CUDA、HIP、XPU和NPU等多种硬件平台

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 日志模块
import warnings  # 警告模块
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # 类型提示工具

import torch  # PyTorch深度学习框架

from sglang.srt.layers.linear import LinearBase  # 线性层基类
from sglang.srt.layers.moe import MoeRunnerConfig  # MoE运行器配置
from sglang.srt.layers.quantization.base_config import (  # 量化基础配置类
    FusedMoEMethodBase,  # 融合MoE方法基类
    LinearMethodBase,  # 线性方法基类
    QuantizationConfig,  # 量化配置基类
    QuantizeMethodBase,  # 量化方法基类
)
from sglang.srt.layers.quantization.marlin_utils import (  # Marlin工具函数
    check_marlin_supported,  # 检查是否支持Marlin
    check_marlin_supports_layer,  # 检查层是否被Marlin支持
    check_moe_marlin_supports_layer,  # 检查MoE层是否被Marlin支持
    verify_marlin_supported,  # 验证Marlin支持
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 未量化的线性方法
from sglang.srt.layers.quantization.utils import get_scalar_types  # 获取标量类型工具
from sglang.srt.utils.patch_torch import register_fake_if_exists  # 注册torch.compile的fake实现

from .schemes import (  # 从schemes子模块导入各种量化方案
    AWQAscendLinearScheme,  # 昇腾平台AWQ线性层方案
    AWQAscendMoEScheme,  # 昇腾平台AWQ MoE方案
    AWQIntelAMXLinearScheme,  # Intel AMX平台AWQ线性层方案
    AWQIntelAMXMoEScheme,  # Intel AMX平台AWQ MoE方案
    AWQLinearScheme,  # 通用AWQ线性层方案
    AWQMarlinLinearScheme,  # Marlin后端AWQ线性层方案
    AWQMoEScheme,  # 通用AWQ MoE方案
)

if TYPE_CHECKING:  # 仅在类型检查时导入，运行时不导入
    from sglang.srt.layers.moe.token_dispatcher import (  # MoE token分发器
        CombineInput,  # 合并输入类型
        StandardDispatchOutput,  # 标准分发输出类型
    )

from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu  # 硬件平台检测工具

_is_cuda = is_cuda()  # 检测是否为CUDA平台
_is_hip = is_hip()  # 检测是否为HIP（AMD ROCm）平台
_is_xpu = is_xpu()  # 检测是否为XPU（Intel GPU）平台
_is_npu = is_npu()  # 检测是否为NPU（昇腾）平台

if not (_is_cuda or _is_hip or _is_xpu or _is_npu):  # 如果不在支持的平台上
    warnings.warn(f"Only CUDA, HIP and XPU support AWQ currently.")  # 发出警告：仅CUDA、HIP和XPU当前支持AWQ

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


ScalarType, scalar_types = get_scalar_types()  # 获取标量类型及其注册表


def is_layer_skipped_awq(prefix: str, modules_to_not_convert: List[str]):  # 判断某层是否应跳过AWQ量化
    return any(module_name in prefix for module_name in modules_to_not_convert)  # 如果层名前缀包含在不转换模块列表中，则跳过


class AWQConfig(QuantizationConfig):  # AWQ量化配置类
    """Config class for AWQ.  # AWQ的配置类

    Reference: https://arxiv.org/abs/2306.00978  # 参考论文：AWQ激活感知权重量化
    """

    def __init__(  # 初始化方法
        self,
        weight_bits: int,  # 权重位宽
        group_size: int,  # 量化分组大小
        zero_point: bool,  # 是否使用零点
        modules_to_not_convert: Optional[List[str]] = None,  # 不进行量化的模块名列表
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.weight_bits = weight_bits  # 保存权重位宽
        self.group_size = group_size  # 保存分组大小
        self.zero_point = zero_point  # 保存零点标志
        self.modules_to_not_convert = modules_to_not_convert or []  # 保存不转换模块列表，默认为空

        if self.weight_bits != 4:  # 如果权重位宽不是4位
            raise ValueError(  # 抛出异常
                "Currently, only 4-bit weight quantization is supported for "  # 当前仅支持4位权重量化
                f"AWQ, but got {self.weight_bits} bits."  # 但获取到了其他位宽
            )
        self.pack_factor = 32 // self.weight_bits  # 计算打包因子：32位除以权重位宽

    def __repr__(self) -> str:  # 返回配置的字符串表示
        return (
            f"AWQConfig(weight_bits={self.weight_bits}, "  # 权重位宽
            f"group_size={self.group_size}, "  # 分组大小
            f"zero_point={self.zero_point}, "  # 零点标志
            f"modules_to_not_convert={self.modules_to_not_convert})"  # 不转换模块列表
        )

    def get_scaled_act_names(self) -> List[str]:  # 获取需要缩放的激活名称列表
        return []  # AWQ不需要缩放激活，返回空列表

    def get_name(self) -> str:  # 获取量化方法名称
        return "awq"  # 返回"awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:  # 获取支持的激活数据类型
        return [torch.float16] if not _is_npu else [torch.float16, torch.bfloat16]  # NPU额外支持bfloat16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低GPU计算能力要求
        # The AWQ kernel only supports Turing or newer GPUs.  # AWQ内核仅支持Turing或更新GPU
        if _is_npu:  # 如果是NPU平台
            raise NotImplementedError(  # 抛出未实现异常
                'NPU hardware does not support "get_min_capability" feature.'  # NPU硬件不支持"get_min_capability"特性
            )
        else:  # 非NPU平台
            return 75  # 返回计算能力7.5（Turing架构）

    @staticmethod
    def get_config_filenames() -> List[str]:  # 获取配置文件名列表
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq  # 量化配置文件名，例如casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq  # 例如abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",  # 另一种量化配置文件名
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> AWQConfig:  # 从配置字典创建AWQConfig实例
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])  # 从配置中获取权重位宽
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])  # 从配置中获取分组大小
        zero_point = cls.get_from_keys(config, ["zero_point"])  # 从配置中获取零点标志
        modules_to_not_convert = cls.get_from_keys_or(  # 从配置中获取不转换模块列表
            config, ["modules_to_not_convert"], None  # 默认值为None
        )
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)  # 创建并返回AWQConfig实例

    def get_quant_method(  # 获取层的量化方法
        self, layer: torch.nn.Module, prefix: str  # 层模块和层名前缀
    ) -> Optional[LinearMethodBase]:  # 返回线性方法基类或None
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE层

        if _is_npu:  # 如果是NPU平台
            if isinstance(layer, LinearBase):  # 如果是线性层
                if is_layer_skipped_awq(prefix, self.modules_to_not_convert):  # 如果该层被跳过
                    return UnquantizedLinearMethod()  # 返回未量化方法
                layer.scheme = self.get_linear_scheme(layer)  # 获取并设置线性层方案
                return AWQLinearMethod(self)  # 返回AWQ线性方法
            elif isinstance(layer, FusedMoE):  # 如果是融合MoE层
                layer.scheme = self.get_moe_scheme(layer)  # 获取并设置MoE方案
                return AWQMoEMethod(self)  # 返回AWQ MoE方法
            return None  # 其他层类型返回None

        if isinstance(layer, LinearBase):  # 如果是线性层
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):  # 如果该层被跳过
                return UnquantizedLinearMethod()  # 返回未量化方法
            layer.scheme = self.get_linear_scheme(layer)  # 获取并设置线性层方案
            return AWQLinearMethod(self)  # 返回AWQ线性方法
        return None  # 非线性层返回None

    def get_linear_scheme(self, layer: torch.nn.Module):  # 获取线性层量化方案
        assert isinstance(layer, LinearBase)  # 断言层必须是线性层
        # TODO: move platform-specific AWQ scheme selection into the platform  # 待办：将平台特定的AWQ方案选择移至平台
        # plugin factory once quantization hooks are available there.  # 一旦量化钩子在那里可用，就移至插件工厂
        if _is_npu:  # 如果是NPU平台
            return AWQAscendLinearScheme(self)  # 返回昇腾平台AWQ线性层方案
        return AWQLinearScheme(self)  # 返回通用AWQ线性层方案

    def get_moe_scheme(self, layer: torch.nn.Module):  # 获取MoE量化方案
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE层

        assert isinstance(layer, FusedMoE)  # 断言层必须是融合MoE层
        # This is currently only reached by the NPU path in get_quant_method.  # 当前仅在get_quant_method的NPU路径中可达
        if _is_npu:  # 如果是NPU平台
            return AWQAscendMoEScheme(self)  # 返回昇腾平台AWQ MoE方案
        raise NotImplementedError("AWQConfig only supports MoE scheme on NPU.")  # 非NPU平台抛出未实现异常


class AWQCPUConfig(AWQConfig):  # AWQ CPU配置类，继承自AWQConfig
    """CPU Config class for AWQ, inherit from AWQConfig"""  # AWQ的CPU配置类，继承自AWQConfig

    def get_supported_act_dtypes(self) -> List[torch.dtype]:  # 获取CPU支持的激活数据类型
        return [torch.float16, torch.bfloat16]  # CPU支持float16和bfloat16

    def get_quant_method(  # 获取CPU平台的量化方法
        self, layer: torch.nn.Module, prefix: str  # 层模块和层名前缀
    ) -> Optional[LinearMethodBase]:  # 返回线性方法基类或None
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE层

        if isinstance(layer, LinearBase):  # 如果是线性层
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):  # 如果该层被跳过
                return UnquantizedLinearMethod()  # 返回未量化方法
            layer.scheme = self.get_linear_scheme(layer)  # 获取并设置线性层方案
            return AWQLinearMethod(self)  # 返回AWQ线性方法
        elif isinstance(layer, FusedMoE):  # 如果是融合MoE层
            layer.scheme = self.get_moe_scheme(layer)  # 获取并设置MoE方案
            return AWQMoEMethod(self)  # 返回AWQ MoE方法
        return None  # 其他层类型返回None

    def get_linear_scheme(self, layer: torch.nn.Module):  # 获取CPU平台线性层量化方案
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类

        assert isinstance(layer, LinearBase)  # 断言层必须是线性层
        return AWQIntelAMXLinearScheme(self)  # 返回Intel AMX平台AWQ线性层方案

    def get_moe_scheme(self, layer: torch.nn.Module):  # 获取CPU平台MoE量化方案
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE层

        assert isinstance(layer, FusedMoE)  # 断言层必须是融合MoE层
        return AWQIntelAMXMoEScheme(self)  # 返回Intel AMX平台AWQ MoE方案


class AWQMarlinConfig(QuantizationConfig):  # AWQ Marlin量化配置类
    """Config class for AWQ Marlin"""  # AWQ Marlin的配置类

    # num_bits -> type  # 位数到类型的映射
    TYPE_MAP = {
        4: scalar_types.uint4,  # 4位对应uint4
        8: scalar_types.uint8,  # 8位对应uint8
    }

    def __init__(  # 初始化方法
        self,
        weight_bits: int,  # 权重位宽
        group_size: int,  # 量化分组大小
        zero_point: bool,  # 是否使用零点
        lm_head_quantized: bool,  # LM头是否量化
        modules_to_not_convert: Optional[list[str]],  # 不进行量化的模块名列表
        full_config: dict[str, Any],  # 完整配置字典
    ) -> None:
        super().__init__()  # 调用父类初始化
        if _is_hip:  # 如果是HIP平台
            warnings.warn(f"HIP does not support fused_marlin_moe currently.")  # 警告：HIP当前不支持融合Marlin MoE
        self.pack_factor = 32 // weight_bits  # packed into int32  # 打包因子：32位除以权重位宽，打包为int32
        self.group_size = group_size  # 保存分组大小
        self.zero_point = zero_point  # 保存零点标志
        self.lm_head_quantized = lm_head_quantized  # 保存LM头量化标志
        self.weight_bits = weight_bits  # 保存权重位宽
        self.modules_to_not_convert = modules_to_not_convert or []  # 保存不转换模块列表，默认为空
        self.full_config = full_config  # 保存完整配置字典

        if self.weight_bits not in self.TYPE_MAP:  # 如果权重位宽不在类型映射中
            raise ValueError(  # 抛出异常
                f"Unsupported num_bits = {self.weight_bits}. "  # 不支持的位数
                f"Supported num_bits = {self.TYPE_MAP.keys()}"  # 支持的位数列表
            )

        self.quant_type = self.TYPE_MAP[self.weight_bits]  # 根据权重位宽获取量化类型

        verify_marlin_supported(  # 验证Marlin是否支持当前配置
            self.quant_type, group_size=self.group_size, has_zp=self.zero_point  # 传入量化类型、分组大小和零点标志
        )

    def __repr__(self) -> str:  # 返回配置的字符串表示
        return (
            f"AWQMarlinConfig(quant_type={self.quant_type}, "  # 量化类型
            f"group_size={self.group_size}, "  # 分组大小
            f"zero_point={self.zero_point}, "  # 零点标志
            f"lm_head_quantized={self.lm_head_quantized}, "  # LM头量化标志
            f"modules_to_not_convert={self.modules_to_not_convert})"  # 不转换模块列表
        )

    def get_scaled_act_names(self) -> List[str]:  # 获取需要缩放的激活名称列表
        return []  # AWQ Marlin不需要缩放激活，返回空列表

    @classmethod
    def get_name(cls) -> str:  # 获取量化方法名称
        return "awq_marlin"  # 返回"awq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:  # 获取支持的激活数据类型
        return [torch.half, torch.bfloat16]  # 支持float16和bfloat16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低GPU计算能力要求
        return 80  # 返回计算能力8.0（Ampere架构）

    @classmethod
    def get_config_filenames(cls) -> list[str]:  # 获取配置文件名列表
        return ["quantize_config.json"]  # 返回量化配置文件名

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AWQMarlinConfig:  # 从配置字典创建AWQMarlinConfig实例
        weight_bits = cls.get_from_keys(config, ["bits"])  # 从配置中获取权重位宽
        group_size = cls.get_from_keys(config, ["group_size"])  # 从配置中获取分组大小
        zero_point = cls.get_from_keys(config, ["zero_point"])  # 从配置中获取零点标志
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)  # 从配置中获取LM头量化标志，默认False
        modules_to_not_convert = cls.get_from_keys_or(  # 从配置中获取不转换模块列表
            config, ["modules_to_not_convert"], None  # 默认值为None
        )
        return cls(  # 创建并返回AWQMarlinConfig实例
            weight_bits,  # 权重位宽
            group_size,  # 分组大小
            zero_point,  # 零点标志
            lm_head_quantized,  # LM头量化标志
            modules_to_not_convert,  # 不转换模块列表
            config,  # 完整配置字典
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:  # 覆盖量化方法，判断是否可转为awq_marlin
        can_convert = cls.is_awq_marlin_compatible(hf_quant_cfg)  # 检查是否与AWQ Marlin兼容
        is_valid_user_quant = (  # 检查用户指定的量化方法是否有效
            user_quant is None or user_quant == "marlin" or user_quant == "awq_marlin"  # 未指定或指定为marlin/awq_marlin
        )

        if can_convert and is_valid_user_quant:  # 如果兼容且用户量化方法有效
            msg = (  # 构建日志消息
                "The model is convertible to {} during runtime."  # 模型可在运行时转换为{}
                " Using {} kernel.".format(cls.get_name(), cls.get_name())  # 使用{}内核
            )
            logger.info(msg)  # 记录信息日志
            return cls.get_name()  # 返回awq_marlin方法名

        if can_convert and user_quant == "awq":  # 如果兼容但用户显式指定了awq
            logger.info(  # 记录信息日志
                "Detected that the model can run with awq_marlin"  # 检测到模型可使用awq_marlin运行
                ", however you specified quantization=awq explicitly,"  # 但您显式指定了quantization=awq
                " so forcing awq. Use quantization=awq_marlin for"  # 因此强制使用awq。使用quantization=awq_marlin以获得
                " faster inference"  # 更快的推理
            )
        return None  # 返回None表示不覆盖

    def get_quant_method(  # 获取层的量化方法
        self, layer: torch.nn.Module, prefix: str  # 层模块和层名前缀
    ) -> Optional[QuantizeMethodBase]:  # 返回量化方法基类或None
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE层
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行LM头

        if isinstance(layer, LinearBase) or (  # 如果是线性层或
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized  # 是并行LM头且LM头已量化
        ):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):  # 如果该层被跳过
                return UnquantizedLinearMethod()  # 返回未量化方法
            # Check if the layer is supported by AWQMarlin.  # 检查该层是否被AWQMarlin支持
            if not check_marlin_supports_layer(layer, self.group_size):  # 如果层不被Marlin支持
                logger.warning_once(  # 记录一次警告
                    "Layer '%s' is not supported by AWQMarlin. Falling back to unoptimized AWQ kernels.",  # noqa: E501  # 层'%s'不被AWQMarlin支持，回退到未优化的AWQ内核
                    prefix,  # 层名前缀
                )
                return AWQConfig.from_config(self.full_config).get_quant_method(  # 回退到普通AWQ配置的量化方法
                    layer, prefix  # 传入层和前缀
                )
            layer.scheme = self.get_linear_scheme(layer)  # 获取并设置线性层方案
            return AWQLinearMethod(self)  # 返回AWQ线性方法
        elif isinstance(layer, FusedMoE):  # 如果是融合MoE层
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config  # 导入MoE WNA16配置

            if not check_moe_marlin_supports_layer(layer, self.group_size):  # 如果MoE层不被Marlin支持
                logger.warning_once(  # 记录一次警告
                    f"Layer '{prefix}' is not supported by AWQMoeMarlin. "  # 层'{prefix}'不被AWQMoeMarlin支持
                    "Falling back to Moe WNA16 kernels."  # 回退到MoE WNA16内核
                )
                return MoeWNA16Config.from_config(self.full_config).get_quant_method(  # 回退到MoeWNA16配置的量化方法
                    layer, prefix  # 传入层和前缀
                )
            layer.scheme = self.get_moe_scheme(layer)  # 获取并设置MoE方案
            return AWQMoEMethod(self)  # 返回AWQ MoE方法
        return None  # 其他层类型返回None

    def get_linear_scheme(self, layer: torch.nn.Module):  # 获取Marlin线性层量化方案
        return AWQMarlinLinearScheme(self)  # 返回Marlin后端AWQ线性层方案

    def get_moe_scheme(self, layer: torch.nn.Module):  # 获取Marlin MoE量化方案
        return AWQMoEScheme(self)  # 返回通用AWQ MoE方案

    @classmethod
    def is_awq_marlin_compatible(cls, quant_config: dict[str, Any]):  # 判断配置是否与AWQ Marlin兼容
        # Extract data from quant config.  # 从量化配置中提取数据
        quant_method = quant_config.get("quant_method", "").lower()  # 获取量化方法名并转为小写
        num_bits = quant_config.get("bits")  # 获取量化位数
        group_size = quant_config.get("group_size")  # 获取分组大小
        zero_point = quant_config.get("zero_point")  # 获取零点标志

        if not _is_cuda:  # 如果不是CUDA平台
            return False  # 不兼容

        if quant_method != "awq":  # 如果量化方法不是awq
            return False  # 不兼容

        # If we cannot find the info needed in the config, cannot convert.  # 如果在配置中找不到所需信息，则无法转换
        if num_bits is None or group_size is None or zero_point is None:  # 如果任一关键参数缺失
            return False  # 不兼容

        if num_bits not in cls.TYPE_MAP:  # 如果位数不在类型映射中
            return False  # 不兼容

        return check_marlin_supported(  # 检查Marlin是否支持该配置
            quant_type=cls.TYPE_MAP[num_bits], group_size=group_size, has_zp=zero_point  # 传入量化类型、分组大小和零点标志
        )


class AWQLinearMethod(LinearMethodBase):  # AWQ线性层量化方法类
    """Linear method for AWQ.  # AWQ的线性方法

    Args:  # 参数
        quant_config: The AWQ quantization config.  # quant_config: AWQ量化配置
    """

    def __init__(self, quant_config: AWQConfig):  # 初始化方法
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建量化权重
        self,
        layer: torch.nn.Module,  # 目标层模块
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: List[int],  # 输出分区大小列表
        input_size: int,  # 总输入大小
        output_size: int,  # 总输出大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器
        layer.scheme.create_weights(  # 委托给方案创建权重
            layer=layer,  # 目标层
            input_size_per_partition=input_size_per_partition,  # 每个分区的输入大小
            output_partition_sizes=output_partition_sizes,  # 输出分区大小列表
            input_size=input_size,  # 总输入大小
            output_size=output_size,  # 总输出大小
            params_dtype=params_dtype,  # 参数数据类型
            weight_loader=weight_loader,  # 权重加载器
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的后处理
        layer.scheme.process_weights_after_loading(layer)  # 委托给方案进行后处理

    def apply(  # 应用量化线性计算
        self,
        layer: torch.nn.Module,  # 目标层模块
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置张量，可选
    ) -> torch.Tensor:  # 返回输出张量
        return layer.scheme.apply_weights(layer, x, bias)  # 委托给方案应用权重计算


class AWQMoEMethod(FusedMoEMethodBase):  # AWQ MoE量化方法类

    def __init__(self, quant_config: AWQMarlinConfig):  # 初始化方法
        self.quant_config = quant_config  # 保存量化配置
        self.quant_type = scalar_types.uint4  # 量化类型设为uint4
        if self.quant_config.weight_bits != 4:  # 如果权重位宽不是4位
            raise ValueError("AWQMoEMethod only supports 4bit now.")  # 抛出异常：AWQMoEMethod当前仅支持4位

    def create_weights(  # 创建MoE量化权重
        self,
        layer: torch.nn.Module,  # 目标层模块
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        layer.scheme.create_weights(  # 委托给方案创建权重
            layer=layer,  # 目标层
            num_experts=num_experts,  # 专家数量
            hidden_size=hidden_size,  # 隐藏层大小
            intermediate_size_per_partition=intermediate_size_per_partition,  # 每个分区的中间层大小
            params_dtype=params_dtype,  # 参数数据类型
            **extra_weight_attrs,  # 额外权重属性
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的后处理
        layer.scheme.process_weights_after_loading(layer)  # 委托给方案进行后处理

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 层模块和MoE运行器配置
    ):
        layer.scheme.create_moe_runner(layer, moe_runner_config)  # 委托给方案创建MoE运行器

    def apply(  # 应用MoE量化计算
        self,
        layer: torch.nn.Module,  # 目标层模块
        dispatch_output: StandardDispatchOutput,  # 分发输出
    ) -> CombineInput:  # 返回合并输入
        return layer.scheme.apply_weights(layer, dispatch_output)  # 委托给方案应用权重计算


# Register fake implementations for torch.compile support  # 为torch.compile支持注册fake实现
if _is_cuda:  # 如果是CUDA平台

    @register_fake_if_exists("sgl_kernel::awq_marlin_repack")  # 注册awq_marlin_repack的fake实现
    def _(b_q_weight, size_k, size_n, num_bits):  # fake函数：模拟AWQ Marlin重打包操作
        return b_q_weight.new_empty(  # 返回新的空张量
            (size_k // 16, size_n * (num_bits // 2)), dtype=b_q_weight.dtype  # 形状为(size_k//16, size_n*(num_bits//2))，数据类型与输入一致
        )
