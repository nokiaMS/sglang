# ModelSlim量化方案子模块的初始化文件，负责导出所有量化方案类
# SPDX-License-Identifier: Apache-2.0

from .modelslim_scheme import ModelSlimLinearScheme, ModelSlimMoEScheme  # 导入ModelSlim线性层和MoE层的基础抽象方案类
from .modelslim_w4a4_int4 import ModelSlimW4A4Int4  # 导入W4A4 Int4量化方案类
from .modelslim_w4a4_int4_moe import ModelSlimW4A4Int4MoE  # 导入W4A4 Int4 MoE量化方案类
from .modelslim_w4a8_int8_moe import ModelSlimW4A8Int8MoE  # 导入W4A8 Int8 MoE量化方案类
from .modelslim_w8a8_int8 import ModelSlimW8A8Int8  # 导入W8A8 Int8量化方案类
from .modelslim_w8a8_int8_moe import ModelSlimW8A8Int8MoE  # 导入W8A8 Int8 MoE量化方案类

__all__ = [  # 定义模块的公开导出列表
    "ModelSlimLinearScheme",  # ModelSlim线性层基础方案
    "ModelSlimMoEScheme",  # ModelSlim MoE层基础方案
    "ModelSlimW8A8Int8",  # W8A8 Int8量化方案
    "ModelSlimW4A4Int4",  # W4A4 Int4量化方案
    "ModelSlimW4A4Int4MoE",  # W4A4 Int4 MoE量化方案
    "ModelSlimW4A8Int8MoE",  # W4A8 Int8 MoE量化方案
    "ModelSlimW8A8Int8MoE",  # W8A8 Int8 MoE量化方案
]
