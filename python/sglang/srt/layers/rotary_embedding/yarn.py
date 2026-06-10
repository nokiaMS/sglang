# YaRN缩放旋转位置编码模块
# 实现YaRN（Yet another RoPE extensioN method）缩放方法及相关辅助函数
"""YaRNScalingRotaryEmbedding + YaRN helper functions."""  # YaRN缩放旋转位置编码及YaRN辅助函数

from __future__ import annotations  # 启用延迟注解评估

import math  # 数学运算模块
from typing import Tuple  # 类型提示：元组类型

import torch  # PyTorch深度学习框架

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding  # 导入基础旋转位置编码类


# Inverse dim formula to find dim based on number of rotations  # 逆维度公式，根据旋转次数查找维度
def yarn_find_correction_dim(  # 根据旋转次数查找校正维度
    num_rotations: int,  # 旋转次数
    dim: int,  # 维度大小
    base: float = 10000,  # 频率基数，默认为10000
    max_position_embeddings: int = 2048,  # 最大位置嵌入数，默认为2048
) -> float:  # 返回校正维度值
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (  # 逆维度公式计算
        2 * math.log(base)  # 除以2倍的基数对数
    )


# Find dim range bounds based on rotations  # 根据旋转次数查找维度范围边界
def yarn_find_correction_range(  # 查找YaRN校正范围
    low_rot: int,  # 低旋转次数
    high_rot: int,  # 高旋转次数
    dim: int,  # 维度大小
    base: float = 10000,  # 频率基数，默认为10000
    max_position_embeddings: int = 2048,  # 最大位置嵌入数，默认为2048
    truncate: bool = True,  # 是否截断为整数，默认为True
) -> Tuple[int, int]:  # 返回校正范围的下界和上界
    low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)  # 计算低旋转次数对应的校正维度
    high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)  # 计算高旋转次数对应的校正维度
    if truncate:  # 如果需要截断
        low = math.floor(low)  # 下界取整
        high = math.ceil(high)  # 上界取整
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case  # 钳制值以防越界


def yarn_linear_ramp_mask(  # 生成YaRN线性斜坡掩码
    low: float, high: float, dim: int, dtype: torch.dtype, device: torch.device = None  # 下界、上界、维度、数据类型、设备
) -> torch.Tensor:  # 返回斜坡掩码张量
    if low == high:  # 如果下界等于上界
        high += 0.001  # Prevent singularity  # 防止奇异性（除以零）

    linear_func = (torch.arange(dim, dtype=dtype, device=device) - low) / (high - low)  # 计算线性函数值
    ramp_func = torch.clamp(linear_func, 0, 1)  # 钳制到[0, 1]范围，形成斜坡
    return ramp_func  # 返回斜坡掩码


def yarn_get_mscale_simple(scale: float = 1) -> float:  # 获取YaRN简单幅度缩放因子
    if scale <= 1:  # 如果缩放比例不超过1
        return 1.0  # 返回1.0（不缩放）
    return 0.1 * math.log(scale) + 1.0  # 对数缩放公式


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:  # 获取YaRN幅度缩放因子（带mscale参数）
    if scale <= 1:  # 如果缩放比例不超过1
        return 1.0  # 返回1.0（不缩放）
    return 0.1 * mscale * math.log(scale) + 1.0  # 带mscale参数的对数缩放公式


class YaRNScalingRotaryEmbedding(RotaryEmbedding):  # YaRN缩放旋转位置编码类
    """RotaryEmbedding extended with YaRN method.  # 使用YaRN方法扩展的旋转位置编码

    Credits to Peng et al. github.com/jquesnelle/yarn  # 致谢Peng等人 github.com/jquesnelle/yarn
    """

    def __init__(  # 初始化YaRN缩放旋转位置编码
        self,
        head_size: int,  # 注意力头的大小
        rotary_dim: int,  # 旋转维度
        max_position_embeddings: int,  # 最大位置嵌入数
        base: int,  # 频率基数
        is_neox_style: bool,  # 是否使用NeoX风格
        scaling_factor: float,  # 缩放因子
        dtype: torch.dtype,  # 数据类型
        *,
        extrapolation_factor: float = 1,  # 外推因子，默认为1
        attn_factor: float = 1,  # 注意力因子，默认为1
        beta_fast: int = 32,  # 快速beta参数，默认为32
        beta_slow: int = 1,  # 慢速beta参数，默认为1
        truncate: bool = True,  # 是否截断，默认为True
    ) -> None:
        self.scaling_factor = scaling_factor  # 保存缩放因子
        self.extrapolation_factor = extrapolation_factor  # 保存外推因子
        self.attn_factor = attn_factor  # 保存注意力因子
        self.beta_fast = beta_fast  # 保存快速beta参数
        self.beta_slow = beta_slow  # 保存慢速beta参数
        self.truncate = truncate  # 保存截断标志
        # Get n-d magnitude scaling corrected for interpolation  # 获取经插值校正的n维幅度缩放
        self.mscale = float(yarn_get_mscale_simple(self.scaling_factor) * attn_factor)  # 计算综合幅度缩放因子
        super().__init__(  # 调用父类初始化
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:  # 计算带YaRN缩放的逆频率
        pos_freqs = self.base ** (  # 计算位置频率
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim  # 生成0到rotary_dim步长为2的序列并归一化
        )
        inv_freq_extrapolation = 1.0 / pos_freqs  # 外推逆频率
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)  # 插值逆频率（除以缩放因子）

        low, high = yarn_find_correction_range(  # 查找校正范围
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
            self.truncate,
        )
        # Get n-d rotational scaling corrected for extrapolation  # 获取经外推校正的n维旋转缩放
        inv_freq_mask = (  # 计算逆频率掩码
            1
            - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor  # 乘以外推因子
        inv_freq = (  # 混合插值和外推的逆频率
            inv_freq_interpolation * (1 - inv_freq_mask)  # 插值部分（权重为1-掩码）
            + inv_freq_extrapolation * inv_freq_mask  # 外推部分（权重为掩码）
        )
        return inv_freq  # 返回混合逆频率

    def _compute_cos_sin_cache(self) -> torch.Tensor:  # 计算YaRN缩放的余弦正弦缓存
        inv_freq = self._compute_inv_freq(self.scaling_factor)  # 计算带缩放的逆频率
        t = torch.arange(  # 生成扩展的位置序列
            self.max_position_embeddings * self.scaling_factor, dtype=torch.float32  # 扩展最大位置数
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 计算位置-频率矩阵
        cos = freqs.cos() * self.mscale  # 计算余弦值并乘以幅度缩放因子
        sin = freqs.sin() * self.mscale  # 计算正弦值并乘以幅度缩放因子
        cache = torch.cat((cos, sin), dim=-1)  # 拼接余弦和正弦值
        return cache  # 返回缓存
