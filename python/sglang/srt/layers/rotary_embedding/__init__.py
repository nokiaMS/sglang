# 旋转位置编码（RoPE）模块的公共API入口，提供RotaryEmbedding、MRotaryEmbedding等类的统一导入
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # vLLM项目贡献者版权声明
# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.6.6.post1/vllm/model_executor/layers/rotary_embedding.py  # 改编自vLLM项目
"""Rotary Positional Embeddings - public API (drop-in replacement for rotary_embedding.py).  # 旋转位置编码 - 公共API（rotary_embedding.py的直接替换）"""  # 旋转位置编码公共API

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding  # 导入基础旋转位置编码类 # 导入基础RoPE类
from sglang.srt.layers.rotary_embedding.factory import get_rope, get_rope_wrapper  # 导入RoPE工厂函数 # 导入工厂函数
from sglang.srt.layers.rotary_embedding.mrope import (  # 导入多模态旋转位置编码类 # 导入MRoPE相关类
    Ernie4_5_VLRotaryEmbedding,
    MRotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_pos_emb  # 导入旋转位置编码应用函数 # 导入应用函数
from sglang.srt.layers.rotary_embedding.yarn import (  # 导入YaRN相关函数 # 导入YaRN缩放函数
    yarn_find_correction_range,
    yarn_get_mscale_simple,
    yarn_linear_ramp_mask,
)

_yarn_find_correction_range = yarn_find_correction_range  # YaRN校正范围查找的别名 # YaRN校正范围查找别名
_yarn_get_mscale = yarn_get_mscale_simple  # YaRN缩放因子的别名 # YaRN缩放因子别名
_yarn_linear_ramp_mask = yarn_linear_ramp_mask  # YaRN线性斜坡掩码的别名 # YaRN线性斜坡掩码别名

__all__ = [  # 模块公开导出列表 # 模块导出列表
    "RotaryEmbedding",
    "get_rope",
    "get_rope_wrapper",
    "MRotaryEmbedding",
    "Ernie4_5_VLRotaryEmbedding",
    "apply_rotary_pos_emb",
    "_yarn_find_correction_range",
    "_yarn_get_mscale",
    "_yarn_linear_ramp_mask",
]
