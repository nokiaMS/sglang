# GPTQ量化配置与方法模块：定义GPTQ、GPTQAscend、GPTQMarlin的配置类及线性/MoE推理方法，支持权重创建、后处理和推理
from __future__ import annotations  # 启用延迟注解评估，支持类型提示中的前向引用

import logging  # 导入日志模块
from fractions import Fraction  # 导入分数类，用于精确表示打包因子
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union  # 导入类型提示工具

import torch  # 导入PyTorch张量库

from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置
from sglang.srt.layers.quantization.base_config import (  # 导入量化基类
    FusedMoEMethodBase,  # MoE融合方法基类
    LinearMethodBase,  # 线性方法基类
    QuantizationConfig,  # 量化配置基类
    QuantizeMethodBase,  # 量化方法基类
)
from sglang.srt.layers.quantization.marlin_utils import check_marlin_supported  # 导入Marlin支持检查函数
from sglang.srt.layers.quantization.utils import (  # 导入量化工具函数
    get_linear_quant_method,  # 获取线性量化方法
    get_scalar_types,  # 获取标量类型
)
from sglang.srt.utils.patch_torch import register_fake_if_exists  # 导入torch.compile伪实现注册函数

from .schemes import (  # 从schemes子包导入各种量化方案
    GPTQAscendLinearScheme,  # GPTQ Ascend NPU线性方案
    GPTQLinearScheme,  # GPTQ线性方案
    GPTQMarlinLinearScheme,  # GPTQ Marlin线性方案
    GPTQMarlinMoEScheme,  # GPTQ Marlin MoE方案
    GPTQMoEAscendScheme,  # GPTQ Ascend NPU MoE方案
)

if TYPE_CHECKING:  # 仅在类型检查时导入，避免运行时循环依赖
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入MoE token分发器类型
        CombineInput,  # 合并输入类型
        StandardDispatchOutput,  # 标准分发输出类型
    )

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器
_, scalar_types = get_scalar_types()  # 获取标量类型集合（忽略第一个返回值）


def check_marlin_format(hf_quant_cfg: Dict[str, Any]) -> bool:  # 检查HuggingFace量化配置是否为Marlin格式
    # compat: gptqmodel and autogptq (eol) main use checkpoint_format: str
    # 兼容：gptqmodel和autogptq（已停止维护）主要使用checkpoint_format: str
    # compat: autogptq <=0.7.1 is_marlin_format: bool
    # 兼容：autogptq <=0.7.1使用is_marlin_format: bool
    return hf_quant_cfg.get("checkpoint_format") == "marlin" or hf_quant_cfg.get(  # 检查checkpoint_format是否为"marlin"
        "is_marlin_format", False  # 或检查is_marlin_format是否为True
    )


class GPTQConfig(QuantizationConfig):  # GPTQ量化配置类，继承自量化配置基类
    """Config class for GPTQ.
    GPTQ配置类

    Reference: https://arxiv.org/abs/2210.17323  # 参考论文
    """

    def __init__(  # 初始化方法
        self,
        weight_bits: int,  # 权重量化位数
        group_size: int,  # 量化组大小，-1表示每通道量化
        desc_act: bool,  # 是否启用描述符激活（激活顺序重排）
        lm_head_quantized: bool,  # 语言模型头是否量化
        dynamic: Dict[str, Dict[str, Union[int, bool]]],  # 动态量化配置字典
        checkpoint_format: str = "",  # 检查点格式，默认为空
        true_sequential: bool = False,  # 是否使用真正的顺序量化，默认否
        static_groups: bool = False,  # 是否使用静态组，默认否
    ) -> None:
        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # GPTQModel使用`dynamic`配置属性来允许每个模块的量化配置，使每个模块可以单独优化。
        # Format is Dict[str, Dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # 格式为Dict[str, Dict]，其中键是正则表达式字符串，可以执行正匹配（"+:"前缀）或负匹配（"-:"前缀）。
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # 默认为正匹配，覆盖基础量化配置模式，如果不使用前缀。值是字段键和覆盖值的字典格式。
        # Negative matching will skip quantization init for this module
        # entirely:
        # 负匹配将完全跳过此模块的量化初始化：
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # 非量化推理。更多详细信息和量化示例可在https://github.com/ModelCloud/GPTQModel找到。
        # Example:  # 示例
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # 最后1/2的层10-21使用8bit，而0-9使用4bit
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        #  # 最后1/4的层16-21使用8bit且group_size为64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix  # `.*\.`匹配层节点前缀
        #  # positive match layer 10-15  # 正匹配层10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21  # 正匹配层16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers  # 负匹配（跳过）所有`moe`层
        # }
        super().__init__()  # 调用父类初始化
        self.dynamic = dynamic  # 保存动态量化配置

        self.weight_bits = weight_bits  # 保存权重量化位数
        self.group_size = group_size  # 保存量化组大小
        self.desc_act = desc_act  # 保存是否启用描述符激活
        self.lm_head_quantized = lm_head_quantized  # 保存语言模型头是否量化
        self.pack_factor = Fraction(32, self.weight_bits)  # 计算打包因子（32位除以量化位数）
        # GPTQ v1 and v2 format deals with zero points differently.
        # GPTQ v1和v2格式对零点的处理方式不同。
        # Currently GPTQModel stores v1 format checkpoints by default,
        # 当前GPTQModel默认存储v1格式检查点，
        # but provides the option to set `format="gptq_v2"` in `QuantizeConfig`.
        # 但提供了在`QuantizeConfig`中设置`format="gptq_v2"`的选项。
        self.checkpoint_format = checkpoint_format  # 保存检查点格式
        self.true_sequential = true_sequential  # 保存是否使用真正的顺序量化
        self.static_groups = static_groups  # 保存是否使用静态组
        if self.weight_bits not in [2, 3, 4, 8]:  # 检查权重位数是否支持
            raise ValueError(  # 抛出值错误
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {self.weight_bits} bits."  # 当前GPTQ仅支持2/3/4/8位权重量化
            )

    def __repr__(self) -> str:  # 返回配置的字符串表示
        return (
            f"GPTQConfig(weight_bits={self.weight_bits}, "  # 权重位数
            f"group_size={self.group_size}, "  # 组大小
            f"desc_act={self.desc_act}),"  # 描述符激活
            f"lm_head_quantized={self.lm_head_quantized}), "  # 语言模型头量化
            f"dynamic={self.dynamic},"  # 动态配置
            f"checkpoint_format={self.checkpoint_format})"  # 检查点格式
        )

    def get_scaled_act_names(self) -> List[str]:  # 获取需要后缩放的激活函数名列表
        """Returns the activation function names that should be post-scaled.
        返回需要后缩放的激活函数名称

        For now, this is only used by AWQ.
        目前，此方法仅被AWQ使用。
        """
        raise NotImplementedError  # GPTQ不支持此方法，抛出未实现异常

    @classmethod
    def get_name(cls) -> str:  # 获取量化方法名称
        return "gptq"  # 返回"gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:  # 获取支持的激活数据类型列表
        return [torch.half]  # 仅支持float16

    @classmethod
    # Need to figure it out  # 需要进一步确认
    def get_min_capability(cls) -> int:  # 获取最低GPU计算能力要求
        return 60  # 最低要求为SM 6.0

    @classmethod
    def get_config_filenames(cls) -> List[str]:  # 获取配置文件名列表
        return ["quantize_config.json"]  # 配置文件名为quantize_config.json

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> GPTQConfig:  # 从配置字典创建GPTQConfig实例
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})  # 获取动态配置，默认为空字典
        dynamic = {} if dynamic is None else dynamic  # 如果动态配置为None则设为空字典

        weight_bits = cls.get_from_keys(config, ["bits"])  # 获取权重位数
        group_size = cls.get_from_keys(config, ["group_size"])  # 获取组大小
        desc_act = cls.get_from_keys(config, ["desc_act"])  # 获取描述符激活标志
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)  # 获取语言模型头是否量化，默认否
        checkpoint_format = cls.get_from_keys_or(  # 获取检查点格式
            config, ["checkpoint_format"], default=""  # 默认为空字符串
        )
        true_sequential = cls.get_from_keys_or(  # 获取是否使用真正的顺序量化
            config, ["true_sequential"], default=False  # 默认为否
        )
        static_groups = cls.get_from_keys_or(config, ["static_groups"], default=False)  # 获取是否使用静态组，默认否
        return cls(  # 创建并返回GPTQConfig实例
            weight_bits,  # 权重位数
            group_size,  # 组大小
            desc_act,  # 描述符激活
            lm_head_quantized,  # 语言模型头量化
            dynamic,  # 动态配置
            checkpoint_format,  # 检查点格式
            true_sequential,  # 顺序量化
            static_groups,  # 静态组
        )

    def get_quant_method(  # 获取层的量化方法
        self, layer: torch.nn.Module, prefix: str  # 神经网络层和参数前缀
    ) -> Optional[LinearMethodBase]:  # 返回线性方法或None
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入MoE层类型

        if isinstance(layer, FusedMoE):  # 如果层是MoE层
            raise TypeError("GPTQ Method does not support MoE, please use gptq_marlin")  # GPTQ不支持MoE，请使用gptq_marlin
        return get_linear_quant_method(  # 获取线性量化方法
            self, layer, prefix=prefix, linear_method_cls=GPTQLinearMethod  # 使用GPTQLinearMethod类
        )

    def get_linear_scheme(self, layer: torch.nn.Module):  # 获取线性量化方案
        return GPTQLinearScheme(self)  # 返回GPTQ线性方案实例

    def get_moe_scheme(self, layer: torch.nn.Module):  # 获取MoE量化方案
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入MoE层类型

        assert isinstance(layer, FusedMoE)  # 断言层为MoE层
        raise NotImplementedError("GPTQConfig does not support MoE.")  # GPTQ不支持MoE，抛出未实现异常


class GPTQAscendConfig(GPTQConfig):  # GPTQ Ascend NPU配置类，继承自GPTQConfig
    """Config class for GPTQ on Ascend NPU."""  # Ascend NPU上的GPTQ配置类

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:  # 获取支持的激活数据类型列表
        return [torch.half, torch.bfloat16]  # 支持float16和bfloat16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低计算能力要求
        raise NotImplementedError(  # NPU不支持此功能，抛出未实现异常
            'NPU hardware does not support "get_min_capability" feature.'  # NPU硬件不支持"get_min_capability"功能
        )

    def get_quant_method(  # 获取层的量化方法（Ascend NPU版本）
        self, layer: torch.nn.Module, prefix: str  # 神经网络层和参数前缀
    ) -> Optional[LinearMethodBase]:  # 返回线性方法或None
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入MoE层类型

        if isinstance(layer, FusedMoE):  # 如果层是MoE层
            layer.scheme = self.get_moe_scheme(layer)  # 设置层的MoE量化方案
            return GPTQMoEAscendMethod(self)  # 返回GPTQ Ascend MoE方法
        if isinstance(layer, LinearBase):  # 如果层是线性层
            layer.scheme = self.get_linear_scheme(layer)  # 设置层的线性量化方案
            return GPTQLinearAscendMethod(self)  # 返回GPTQ Ascend线性方法
        return None  # 其他类型层返回None

    def get_linear_scheme(self, layer: torch.nn.Module):  # 获取线性量化方案（Ascend NPU版本）
        return GPTQAscendLinearScheme(self)  # 返回GPTQ Ascend线性方案实例

    def get_moe_scheme(self, layer: torch.nn.Module):  # 获取MoE量化方案（Ascend NPU版本）
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入MoE层类型

        assert isinstance(layer, FusedMoE)  # 断言层为MoE层
        return GPTQMoEAscendScheme(self)  # 返回GPTQ Ascend MoE方案实例


class GPTQMarlinConfig(QuantizationConfig):  # GPTQ Marlin量化配置类，继承自量化配置基类
    """Config class for GPTQ Marlin"""  # GPTQ Marlin配置类

    # (num_bits, is_sym) -> quant_type  # （位数，是否对称）-> 量化类型的映射
    TYPE_MAP = {  # 类型映射表
        (4, True): scalar_types.uint4b8,  # 4位对称量化对应uint4b8
        (8, True): scalar_types.uint8b128,  # 8位对称量化对应uint8b128
    }

    def __init__(  # 初始化方法
        self,
        weight_bits: int,  # 权重量化位数
        group_size: int,  # 量化组大小
        desc_act: bool,  # 是否启用描述符激活
        is_sym: bool,  # 是否为对称量化
        lm_head_quantized: bool,  # 语言模型头是否量化
        dynamic: Dict[str, Dict[str, Union[int, bool]]],  # 动态量化配置字典
        full_config: Dict[str, Any],  # 完整配置字典
    ) -> None:
        super().__init__()  # 调用父类初始化
        if desc_act and group_size == -1:  # 如果启用desc_act且组大小为-1
            # In this case, act_order == True is the same as act_order == False
            # 在这种情况下，act_order == True与act_order == False相同
            # (since we have only one group per output channel)
            # （因为每个输出通道只有一个组）
            desc_act = False  # 将desc_act设为False

        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # GPTQModel使用`dynamic`配置属性来允许每个模块的量化配置，使每个模块可以单独优化。
        # Format is Dict[str, Dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # 格式为Dict[str, Dict]，其中键是正则表达式字符串，可以执行正匹配（"+:"前缀）或负匹配（"-:"前缀）。
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # 默认为正匹配，覆盖基础量化配置模式，如果不使用前缀。值是字段键和覆盖值的字典格式。
        # Negative matching will skip quantization init for this module
        # entirely:
        # 负匹配将完全跳过此模块的量化初始化：
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # 非量化推理。更多详细信息和量化示例可在https://github.com/ModelCloud/GPTQModel找到。
        # Example:  # 示例
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # 最后1/2的层10-21使用8bit，而0-9使用4bit
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        #  # 最后1/4的层16-21使用8bit且group_size为64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix  # `.*\.`匹配层节点前缀
        #  # positive match layer 10-15  # 正匹配层10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21  # 正匹配层16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers  # 负匹配（跳过）所有`moe`层
        # }
        self.dynamic = dynamic  # 保存动态量化配置

        self.weight_bits = weight_bits  # 保存权重量化位数
        self.is_sym = is_sym  # 保存是否为对称量化

        self.pack_factor = 32 // weight_bits  # packed into int32  # 打包到int32中，打包因子为32除以位数
        self.group_size = group_size  # 保存量化组大小
        self.desc_act = desc_act  # 保存是否启用描述符激活
        self.lm_head_quantized = lm_head_quantized  # 保存语言模型头是否量化
        self.full_config = full_config  # 保存完整配置

        if (weight_bits, is_sym) not in self.TYPE_MAP:  # 检查量化配置是否受支持
            raise ValueError(  # 抛出值错误
                "Unsupported quantization config: " f"bits={weight_bits}, sym={is_sym}"  # 不支持的量化配置
            )

        # (num_bits, is_sym) -> quant_type  # （位数，是否对称）-> 量化类型
        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]  # 从映射表获取量化类型

    def __repr__(self) -> str:  # 返回配置的字符串表示
        return (
            f"GPTQMarlinConfig(quant_type={self.quant_type}, "  # 量化类型
            f"group_size={self.group_size}, "  # 组大小
            f"desc_act={self.desc_act}, "  # 描述符激活
            f"lm_head_quantized={self.lm_head_quantized}), "  # 语言模型头量化
            f"dynamic={self.dynamic}"  # 动态配置
        )

    def get_scaled_act_names(self) -> List[str]:  # 获取需要后缩放的激活函数名列表
        """Returns the activation function names that should be post-scaled.
        返回需要后缩放的激活函数名称

        For now, this is only used by AWQ.
        目前，此方法仅被AWQ使用。
        """
        raise NotImplementedError  # Marlin不支持此方法，抛出未实现异常

    @classmethod
    def get_name(cls) -> str:  # 获取量化方法名称
        return "gptq_marlin"  # 返回"gptq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:  # 获取支持的激活数据类型列表
        return [torch.half, torch.bfloat16]  # 支持float16和bfloat16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低GPU计算能力要求
        return 80  # 最低要求为SM 8.0

    @classmethod
    def get_config_filenames(cls) -> List[str]:  # 获取配置文件名列表
        return ["quantize_config.json"]  # 配置文件名为quantize_config.json

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> GPTQMarlinConfig:  # 从配置字典创建GPTQMarlinConfig实例
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})  # 获取动态配置，默认为空字典
        dynamic = {} if dynamic is None else dynamic  # 如果动态配置为None则设为空字典

        weight_bits = cls.get_from_keys(config, ["bits"])  # 获取权重位数
        group_size = cls.get_from_keys(config, ["group_size"])  # 获取组大小
        desc_act = cls.get_from_keys(config, ["desc_act"])  # 获取描述符激活标志
        is_sym = cls.get_from_keys(config, ["sym"])  # 获取是否为对称量化
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)  # 获取语言模型头是否量化，默认否
        return cls(  # 创建并返回GPTQMarlinConfig实例
            weight_bits,  # 权重位数
            group_size,  # 组大小
            desc_act,  # 描述符激活
            is_sym,  # 对称量化
            lm_head_quantized,  # 语言模型头量化
            dynamic,  # 动态配置
            config,  # 完整配置
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:  # 覆写量化方法，判断是否可转换为Marlin
        is_marlin_format = check_marlin_format(hf_quant_cfg)  # 检查是否已经是Marlin格式

        can_convert = cls.is_gptq_marlin_compatible(hf_quant_cfg)  # 检查是否可以转换为Marlin

        is_valid_user_quant = (  # 检查用户指定的量化方法是否有效
            user_quant is None or user_quant == "marlin" or user_quant == "gptq_marlin"  # None、marlin或gptq_marlin均有效
        )

        if not is_marlin_format and can_convert and is_valid_user_quant:  # 非Marlin格式、可转换、用户量化方法有效
            msg = (  # 构建提示信息
                "The model is convertible to {} during runtime."  # 模型可在运行时转换为{}
                " Using {} kernel.".format(cls.get_name(), cls.get_name())  # 使用{}内核
            )
            logger.info(msg)  # 记录信息日志
            return cls.get_name()  # 返回量化方法名称，表示使用Marlin

        if not is_marlin_format and can_convert and user_quant == "gptq":  # 可转换但用户显式指定gptq
            logger.info(  # 记录信息日志
                "Detected that the model can run with gptq_marlin"  # 检测到模型可以使用gptq_marlin运行
                ", however you specified quantization=gptq explicitly,"  # 但您显式指定了quantization=gptq
                " so forcing gptq. Use quantization=gptq_marlin for"  # 因此强制使用gptq。使用quantization=gptq_marlin以获得
                " faster inference"  # 更快的推理速度
            )
        return None  # 不覆写，返回None

    def get_quant_method(  # 获取层的量化方法（Marlin版本）
        self, layer: torch.nn.Module, prefix: str  # 神经网络层和参数前缀
    ) -> Optional[QuantizeMethodBase]:  # 返回量化方法或None
        # Delay the import to avoid circular dependency  # 延迟导入以避免循环依赖
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入MoE层类型

        if isinstance(layer, FusedMoE):  # 如果层是MoE层
            return GPTQMarlinMoEMethod(self)  # 返回GPTQ Marlin MoE方法
        return get_linear_quant_method(  # 获取线性量化方法
            self, layer, prefix=prefix, linear_method_cls=GPTQMarlinLinearMethod  # 使用GPTQMarlinLinearMethod类
        )

    def get_linear_scheme(self, layer: torch.nn.Module):  # 获取线性量化方案（Marlin版本）
        return GPTQMarlinLinearScheme(self)  # 返回GPTQ Marlin线性方案实例

    def get_moe_scheme(self, layer: torch.nn.Module):  # 获取MoE量化方案（Marlin版本）
        return GPTQMarlinMoEScheme(self)  # 返回GPTQ Marlin MoE方案实例

    @classmethod
    def is_gptq_marlin_compatible(cls, quant_config: Dict[str, Any]):  # 检查GPTQ配置是否与Marlin兼容
        quant_method = quant_config.get("quant_method", "").lower()  # 获取量化方法名并转小写
        num_bits = quant_config.get("bits")  # 获取权重位数
        group_size = quant_config.get("group_size")  # 获取组大小
        sym = quant_config.get("sym")  # 获取是否对称量化
        desc_act = quant_config.get("desc_act")  # 获取是否启用描述符激活

        if quant_method != "gptq":  # 如果量化方法不是gptq
            return False  # 不兼容

        # Marlin conversion is only valid if required properties are found
        # 仅当找到所需属性时，Marlin转换才有效
        if num_bits is None or group_size is None or sym is None or desc_act is None:  # 检查必要属性是否存在
            return False  # 缺少必要属性，不兼容

        if (num_bits, sym) not in cls.TYPE_MAP:  # 检查位数和对称性组合是否受支持
            return False  # 不受支持的组合，不兼容

        try:  # 尝试检查Marlin是否支持
            return check_marlin_supported(  # 调用Marlin支持检查
                quant_type=cls.TYPE_MAP[(num_bits, sym)], group_size=group_size  # 传入量化类型和组大小
            )
        except Exception:  # 检查过程中出现异常
            return False  # 不兼容


class GPTQLinearMethod(LinearMethodBase):  # GPTQ线性推理方法类，继承自线性方法基类
    """Linear method for GPTQ.
    GPTQ的线性方法

    Args:  # 参数
        quant_config: The GPTQ quantization config.  # GPTQ量化配置
    """

    def __init__(self, quant_config: GPTQConfig):  # 初始化方法，接收GPTQ配置
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建量化权重参数
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: list[int],  # 输出分区大小列表
        input_size: int,  # 总输入大小
        output_size: int,  # 总输出大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 其他权重属性
    ):  # 创建量化权重参数
        if not hasattr(layer, "scheme"):  # 如果层没有scheme属性
            layer.scheme = self.quant_config.get_linear_scheme(layer)  # 从配置获取并设置线性方案
        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器
        layer.scheme.create_weights(  # 委托方案创建权重
            layer=layer,  # 目标层
            input_size_per_partition=input_size_per_partition,  # 分区输入大小
            output_partition_sizes=output_partition_sizes,  # 输出分区大小
            input_size=input_size,  # 总输入大小
            output_size=output_size,  # 总输出大小
            params_dtype=params_dtype,  # 参数数据类型
            weight_loader=weight_loader,  # 权重加载器
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的处理
        layer.scheme.process_weights_after_loading(layer)  # 委托方案处理

    def apply(  # 应用推理计算
        self,
        layer: torch.nn.Module,  # 目标层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置张量，可选
    ) -> torch.Tensor:  # 返回输出张量
        return layer.scheme.apply_weights(layer, x, bias)  # 委托方案应用权重


class GPTQMoEAscendMethod(FusedMoEMethodBase):  # GPTQ Ascend NPU MoE推理方法类，继承自MoE方法基类

    def __init__(self, quant_config: GPTQConfig):  # 初始化方法，接收GPTQ配置
        super().__init__()  # 调用父类初始化
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建MoE量化权重参数
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 其他权重属性
    ):  # 创建MoE量化权重参数
        if not hasattr(layer, "scheme"):  # 如果层没有scheme属性
            layer.scheme = self.quant_config.get_moe_scheme(layer)  # 从配置获取并设置MoE方案
        layer.scheme.create_weights(  # 委托方案创建权重
            layer=layer,  # 目标层
            num_experts=num_experts,  # 专家数量
            hidden_size=hidden_size,  # 隐藏层大小
            intermediate_size_per_partition=intermediate_size_per_partition,  # 分区中间层大小
            params_dtype=params_dtype,  # 参数数据类型
            **extra_weight_attrs,  # 其他权重属性
        )

    def create_moe_runner(  # 创建MoE运行器
        self,
        layer: torch.nn.Module,  # 目标层
        moe_runner_config: MoeRunnerConfig,  # MoE运行器配置
        **extra_weight_attrs,  # 其他权重属性
    ):  # 创建MoE运行器
        layer.scheme.create_moe_runner(layer, moe_runner_config)  # 委托方案创建MoE运行器

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的处理
        layer.scheme.process_weights_after_loading(layer)  # 委托方案处理

    def apply(  # 应用MoE推理计算
        self,
        layer: torch.nn.Module,  # 目标层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> torch.Tensor:  # 返回输出张量
        return layer.scheme.apply_weights(layer, dispatch_output)  # 委托方案应用权重


class GPTQMarlinLinearMethod(LinearMethodBase):  # GPTQ Marlin线性推理方法类，继承自线性方法基类
    """Linear method for GPTQ Marlin.
    GPTQ Marlin的线性方法

    Args:  # 参数
        quant_config: The GPTQ Marlin quantization config.  # GPTQ Marlin量化配置
    """

    _kernel_backends_being_used: set[str] = set()  # 当前正在使用的内核后端集合

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:  # 初始化方法，接收Marlin配置
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建量化权重参数（Marlin版本）
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: list[int],  # 输出分区大小列表
        input_size: int,  # 总输入大小
        output_size: int,  # 总输出大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 其他权重属性
    ) -> None:
        if not hasattr(layer, "scheme"):  # 如果层没有scheme属性
            layer.scheme = self.quant_config.get_linear_scheme(layer)  # 从配置获取并设置线性方案
        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器
        layer.scheme.create_weights(  # 委托方案创建权重
            layer=layer,  # 目标层
            input_size_per_partition=input_size_per_partition,  # 分区输入大小
            output_partition_sizes=output_partition_sizes,  # 输出分区大小
            input_size=input_size,  # 总输入大小
            output_size=output_size,  # 总输出大小
            params_dtype=params_dtype,  # 参数数据类型
            weight_loader=weight_loader,  # 权重加载器
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的处理
        layer.scheme.process_weights_after_loading(layer)  # 委托方案处理

    def apply(  # 应用推理计算（Marlin版本）
        self,
        layer: torch.nn.Module,  # 目标层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置张量，可选
    ) -> torch.Tensor:  # 返回输出张量
        return layer.scheme.apply_weights(layer, x, bias)  # 委托方案应用权重


class GPTQLinearAscendMethod(GPTQLinearMethod):  # GPTQ Ascend NPU线性推理方法类，继承自GPTQLinearMethod
    """Linear method for GPTQ on Ascend NPU."""  # Ascend NPU上的GPTQ线性方法


class GPTQMarlinMoEMethod(FusedMoEMethodBase):  # GPTQ Marlin MoE推理方法类，继承自MoE方法基类
    """MoE Marlin method with quantization."""  # 带量化的MoE Marlin方法

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:  # 初始化方法，接收Marlin配置
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建MoE量化权重参数（Marlin版本）
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 其他权重属性
    ):  # 创建MoE量化权重参数
        if not hasattr(layer, "scheme"):  # 如果层没有scheme属性
            layer.scheme = self.quant_config.get_moe_scheme(layer)  # 从配置获取并设置MoE方案
        layer.scheme.create_weights(  # 委托方案创建权重
            layer=layer,  # 目标层
            num_experts=num_experts,  # 专家数量
            hidden_size=hidden_size,  # 隐藏层大小
            intermediate_size_per_partition=intermediate_size_per_partition,  # 分区中间层大小
            params_dtype=params_dtype,  # 参数数据类型
            **extra_weight_attrs,  # 其他权重属性
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的处理
        layer.scheme.process_weights_after_loading(layer)  # 委托方案处理

    def create_moe_runner(  # 创建MoE运行器（Marlin版本）
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 目标层和MoE运行器配置
    ):  # 创建MoE运行器
        layer.scheme.create_moe_runner(layer, moe_runner_config)  # 委托方案创建MoE运行器

    def apply(  # 应用MoE推理计算（Marlin版本）
        self,
        layer: torch.nn.Module,  # 目标层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:  # 返回合并输入
        return layer.scheme.apply_weights(layer, dispatch_output)  # 委托方案应用权重


# Register fake implementations for torch.compile support. The decorator is a
# no-op when the custom op is unavailable on the current platform.
# 注册伪实现以支持torch.compile。当自定义算子在当前平台不可用时，装饰器为空操作。
@register_fake_if_exists("sgl_kernel::gptq_gemm")  # 为sgl_kernel::gptq_gemm注册伪实现
def _(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit):  # GPTQ GEMM伪实现函数
    return a.new_empty((a.shape[0], b_q_weight.shape[-1]), dtype=a.dtype)  # 返回与输入同类型的空张量


@register_fake_if_exists("sgl_kernel::gptq_marlin_repack")  # 为sgl_kernel::gptq_marlin_repack注册伪实现
def _(b_q_weight, perm, size_k, size_n, num_bits):  # GPTQ Marlin重打包伪实现函数
    return b_q_weight.new_empty(  # 返回与输入同类型的空张量
        (size_k // 16, size_n * (num_bits // 2)), dtype=b_q_weight.dtype  # 形状为(size_k//16, size_n*(num_bits//2))
    )


@register_fake_if_exists("sgl_kernel::gptq_shuffle")  # 为sgl_kernel::gptq_shuffle注册伪实现
def _(q_weight, q_perm, bit):  # GPTQ shuffle伪实现函数
    return  # 无返回值
