# MoE（混合专家模型）token分发器模块的初始化文件
# 导出所有分发器相关的类和数据结构，供外部统一引用
from sglang.srt.layers.moe.token_dispatcher.base import (  # 从基础模块导入基类和通用类型
    BaseDispatcher,  # 分发器基类
    BaseDispatcherConfig,  # 分发器配置基类
    CombineInput,  # 合并输入协议
    CombineInputChecker,  # 合并输入格式检查器
    CombineInputFormat,  # 合并输入格式枚举
    DispatchOutput,  # 分发输出协议
    DispatchOutputChecker,  # 分发输出格式检查器
    DispatchOutputFormat,  # 分发输出格式枚举
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (  # 从DeepEP模块导入DeepEP相关类
    DeepEPConfig,  # DeepEP配置类
    DeepEPDispatcher,  # DeepEP分发器
    DeepEPLLCombineInput,  # DeepEP低延迟合并输入
    DeepEPLLDispatchOutput,  # DeepEP低延迟分发输出
    DeepEPNormalCombineInput,  # DeepEP普通模式合并输入
    DeepEPNormalDispatchOutput,  # DeepEP普通模式分发输出
)
from sglang.srt.layers.moe.token_dispatcher.flashinfer import (  # 从FlashInfer模块导入FlashInfer相关类
    FlashinferDispatcher,  # FlashInfer分发器
    FlashinferDispatchOutput,  # FlashInfer分发输出
)
from sglang.srt.layers.moe.token_dispatcher.mooncake import (  # 从Mooncake模块导入Mooncake相关类
    MooncakeCombineInput,  # Mooncake合并输入
    MooncakeDispatchOutput,  # Mooncake分发输出
    MooncakeEPDispatcher,  # Mooncake EP分发器
)
from sglang.srt.layers.moe.token_dispatcher.moriep import (  # 从MoriEP模块导入MoriEP相关类
    MoriEPDispatcher,  # MoriEP分发器
    MoriEPLLCombineInput,  # MoriEP低延迟合并输入
    MoriEPLLDispatchOutput,  # MoriEP低延迟分发输出
    MoriEPNormalCombineInput,  # MoriEP普通模式合并输入
    MoriEPNormalDispatchOutput,  # MoriEP普通模式分发输出
)
from sglang.srt.layers.moe.token_dispatcher.nixl import (  # 从Nixl模块导入Nixl相关类
    NixlEPCombineInput,  # Nixl合并输入
    NixlEPDispatcher,  # Nixl分发器
    NixlEPDispatchOutput,  # Nixl分发输出
)
from sglang.srt.layers.moe.token_dispatcher.standard import (  # 从标准模块导入标准分发器相关类
    StandardCombineInput,  # 标准合并输入
    StandardDispatcher,  # 标准分发器
    StandardDispatchOutput,  # 标准分发输出
)

__all__ = [  # 模块公开导出的符号列表
    "BaseDispatcher",  # 分发器基类
    "BaseDispatcherConfig",  # 分发器配置基类
    "CombineInput",  # 合并输入协议
    "CombineInputChecker",  # 合并输入格式检查器
    "CombineInputFormat",  # 合并输入格式枚举
    "DispatchOutput",  # 分发输出协议
    "DispatchOutputFormat",  # 分发输出格式枚举
    "DispatchOutputChecker",  # 分发输出格式检查器
    "FlashinferDispatchOutput",  # FlashInfer分发输出
    "FlashinferDispatcher",  # FlashInfer分发器
    "MooncakeCombineInput",  # Mooncake合并输入
    "MooncakeDispatchOutput",  # Mooncake分发输出
    "MooncakeEPDispatcher",  # Mooncake EP分发器
    "MoriEPNormalDispatchOutput",  # MoriEP普通模式分发输出
    "MoriEPNormalCombineInput",  # MoriEP普通模式合并输入
    "MoriEPLLDispatchOutput",  # MoriEP低延迟分发输出
    "MoriEPLLCombineInput",  # MoriEP低延迟合并输入
    "MoriEPDispatcher",  # MoriEP分发器
    "NixlEPCombineInput",  # Nixl合并输入
    "NixlEPDispatchOutput",  # Nixl分发输出
    "NixlEPDispatcher",  # Nixl分发器
    "StandardDispatcher",  # 标准分发器
    "StandardDispatchOutput",  # 标准分发输出
    "StandardCombineInput",  # 标准合并输入
    "DeepEPConfig",  # DeepEP配置类
    "DeepEPDispatcher",  # DeepEP分发器
    "DeepEPNormalDispatchOutput",  # DeepEP普通模式分发输出
    "DeepEPLLDispatchOutput",  # DeepEP低延迟分发输出
    "DeepEPLLCombineInput",  # DeepEP低延迟合并输入
    "DeepEPNormalCombineInput",  # DeepEP普通模式合并输入
]
