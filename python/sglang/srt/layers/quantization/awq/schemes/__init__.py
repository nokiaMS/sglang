# AWQ量化方案（schemes）子模块的初始化文件
# 导出各种平台和后端的AWQ线性层/MoE量化方案类

# SPDX-License-Identifier: Apache-2.0

from .awq_cpu import AWQIntelAMXLinearScheme, AWQIntelAMXMoEScheme  # 从CPU方案模块导入Intel AMX平台方案
from .awq_linear import AWQAscendLinearScheme, AWQLinearScheme  # 从线性层方案模块导入通用和昇腾平台线性层方案
from .awq_marlin import AWQMarlinLinearScheme  # 从Marlin方案模块导入Marlin后端线性层方案
from .awq_moe import AWQAscendMoEScheme, AWQMoEScheme  # 从MoE方案模块导入通用和昇腾平台MoE方案
from .awq_scheme import AWQLinearSchemeBase, AWQMoESchemeBase  # 从基类模块导入AWQ线性层和MoE方案基类

__all__ = [  # 模块公开导出的符号列表
    "AWQLinearSchemeBase",  # AWQ线性层方案基类
    "AWQMoESchemeBase",  # AWQ MoE方案基类
    "AWQLinearScheme",  # 通用AWQ线性层方案
    "AWQAscendLinearScheme",  # 昇腾平台AWQ线性层方案
    "AWQIntelAMXLinearScheme",  # Intel AMX平台AWQ线性层方案
    "AWQMarlinLinearScheme",  # Marlin后端AWQ线性层方案
    "AWQMoEScheme",  # 通用AWQ MoE方案
    "AWQAscendMoEScheme",  # 昇腾平台AWQ MoE方案
    "AWQIntelAMXMoEScheme",  # Intel AMX平台AWQ MoE方案
]
