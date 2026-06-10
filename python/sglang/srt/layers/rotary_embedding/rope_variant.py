# RoPE缩放变体模块
# 实现了多种旋转位置编码的缩放变体：Phi3LongRoPE、FourierRoPE、DeepseekScaling、Llama3、Llama4Vision、DynamicNTK、DynamicNTKAlpha、DualChunkRotaryEmbedding、Gemma4
"""RoPE scaling variants: Phi3LongRoPE, FourierRoPE, DeepseekScaling, Llama3,
Llama4Vision, DynamicNTK, DynamicNTKAlpha, DualChunkRotaryEmbedding."""  # RoPE缩放变体：Phi3LongRoPE、FourierRoPE、DeepseekScaling、Llama3、Llama4Vision、DynamicNTK、DynamicNTKAlpha、DualChunkRotaryEmbedding

from __future__ import annotations  # 启用延迟注解评估

import math  # 数学运算模块
from typing import List, Optional, Tuple, Union  # 类型提示工具

import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding  # 导入基础旋转位置编码类
from sglang.srt.layers.rotary_embedding.utils import (  # 导入旋转位置编码工具函数
    apply_rotary_pos_emb_native,  # 原生旋转位置编码应用函数
    rotate_gptj,  # GPT-J风格旋转函数
    rotate_neox,  # NeoX风格旋转函数
)
from sglang.srt.layers.rotary_embedding.yarn import (  # 导入YaRN相关工具函数
    yarn_find_correction_range,  # YaRN校正范围查找函数
    yarn_get_mscale,  # YaRN幅度缩放获取函数
    yarn_linear_ramp_mask,  # YaRN线性斜坡掩码函数
)
from sglang.srt.layers.utils import MultiPlatformOp  # 导入多平台操作基类
from sglang.srt.utils import cpu_has_amx_support, get_device, is_cuda, is_hip, is_npu  # 导入平台检测工具函数

_is_cuda = is_cuda()  # 检测是否为CUDA平台
_is_hip = is_hip()  # 检测是否为HIP(AMD)平台
_is_npu = is_npu()  # 检测是否为NPU(华为昇腾)平台
_is_cpu_amx_available = cpu_has_amx_support()  # 检测CPU是否支持AMX指令集

if _is_npu:  # 如果是NPU平台
    import torch_npu  # 导入华为NPU扩展模块


class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):  # Phi3系列模型的缩放旋转位置编码
    """Phi3 family of models scaled rotary embedding."""  # Phi3系列模型的缩放旋转位置编码

    def __init__(  # 初始化Phi3长RoPE缩放旋转位置编码
        self,
        head_size: int,  # 注意力头的大小
        rotary_dim: int,  # 旋转维度
        max_position_embeddings: int,  # 最大位置嵌入数
        original_max_position_embeddings: int,  # 原始最大位置嵌入数
        base: int,  # 频率基数
        is_neox_style: bool,  # 是否使用NeoX风格
        dtype: torch.dtype,  # 数据类型
        short_factor: List[float],  # 短序列缩放因子
        long_factor: List[float],  # 长序列缩放因子
        short_mscale: Optional[float] = None,  # 短序列幅度缩放因子
        long_mscale: Optional[float] = None,  # 长序列幅度缩放因子
    ):
        super().__init__()  # 调用父类初始化

        if is_neox_style is False:  # 如果不是NeoX风格
            raise ValueError(  # 抛出值错误
                "`Phi3LongRoPEScaledRotaryEmbedding` only supports neox_style."  # Phi3LongRoPE仅支持neox_style
            )

        self.rotary_dim = rotary_dim  # 保存旋转维度
        self.head_size = head_size  # 保存注意力头大小
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置嵌入数
        self.original_max_position_embeddings = original_max_position_embeddings  # 保存原始最大位置嵌入数
        self.base = base  # 保存频率基数
        self.short_factor = short_factor  # 保存短序列缩放因子
        self.long_factor = long_factor  # 保存长序列缩放因子

        scale = self.max_position_embeddings / self.original_max_position_embeddings  # 计算缩放比例
        if scale <= 1.0:  # 如果缩放比例不超过1
            scaling_factor = 1.0  # 缩放因子为1
        else:
            scaling_factor = math.sqrt(  # 计算动态缩放因子
                1 + math.log(scale) / math.log(self.original_max_position_embeddings)  # 基于对数的缩放公式
            )
        if short_mscale is None:  # 如果未提供短序列幅度缩放因子
            short_mscale = scaling_factor  # 使用动态缩放因子
        if long_mscale is None:  # 如果未提供长序列幅度缩放因子
            long_mscale = scaling_factor  # 使用动态缩放因子

        self.short_mscale = short_mscale  # 保存短序列幅度缩放因子
        self.long_mscale = long_mscale  # 保存长序列幅度缩放因子

        short_cache = self._compute_cos_sin_cache(  # 计算短序列的余弦正弦缓存
            original_max_position_embeddings, short_factor, short_mscale  # 使用原始最大位置和短序列参数
        )
        short_cache = short_cache.to(dtype)  # 转换数据类型
        self.register_buffer("short_cos_sin_cache", short_cache, persistent=False)  # 注册短序列缓存为非持久化缓冲区

        long_cache = self._compute_cos_sin_cache(  # 计算长序列的余弦正弦缓存
            max_position_embeddings, long_factor, long_mscale  # 使用最大位置和长序列参数
        )
        long_cache = long_cache.to(dtype)  # 转换数据类型
        self.register_buffer("long_cos_sin_cache", long_cache, persistent=False)  # 注册长序列缓存为非持久化缓冲区

        long_short_cache = torch.cat(  # 拼接长序列和短序列缓存
            [self.short_cos_sin_cache, self.long_cos_sin_cache], dim=0  # 在第0维拼接
        )
        self.register_buffer(  # 注册长-短拼接缓存为非持久化缓冲区
            "long_short_cos_sin_cache", long_short_cache, persistent=False
        )

    def _compute_inv_freq(self, rescale_factors: List[float]) -> torch.Tensor:  # 计算逆频率
        rescale_factors = torch.tensor(rescale_factors, dtype=torch.float32)  # 将缩放因子转为张量
        inv_freq = 1.0 / (  # 计算逆频率
            rescale_factors  # 乘以缩放因子
            * (
                self.base  # 频率基数
                ** (
                    torch.arange(0, self.rotary_dim, 2, dtype=torch.float)  # 生成0到rotary_dim步长为2的序列
                    / self.rotary_dim  # 除以旋转维度
                )
            )
        )
        return inv_freq  # 返回逆频率

    def _compute_cos_sin_cache(  # 计算余弦正弦缓存
        self,
        max_position_embeddings: int,  # 最大位置嵌入数
        rescale_factors: List[float],  # 缩放因子列表
        mscale: float,  # 幅度缩放因子
    ) -> torch.Tensor:  # 返回缓存张量
        inv_freq = self._compute_inv_freq(rescale_factors)  # 计算逆频率
        t = torch.arange(max_position_embeddings, dtype=torch.float)  # 生成位置序列
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 计算位置-频率矩阵（外积）
        cos = freqs.cos() * mscale  # 计算余弦值并乘以幅度缩放因子
        sin = freqs.sin() * mscale  # 计算正弦值并乘以幅度缩放因子
        cache = torch.cat((cos, sin), dim=-1)  # 拼接余弦和正弦值
        return cache  # 返回缓存

    def forward(  # 前向传播，应用Phi3长RoPE缩放旋转位置编码
        self,
        positions: torch.Tensor,  # 位置张量
        query: torch.Tensor,  # 查询张量
        key: torch.Tensor,  # 键张量
        offsets: Optional[torch.Tensor] = None,  # 偏移量张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回旋转后的查询和键张量
        query = query.unflatten(1, (-1, self.head_size))  # 将查询重塑为(num_heads, head_size)
        key = key.unflatten(1, (-1, self.head_size))  # 将键重塑为(num_heads, head_size)

        k = self.original_max_position_embeddings  # 获取原始最大位置嵌入数
        long_prompt_offset = (  # 计算长提示的偏移量
            torch.any(positions > k).float() * torch.full_like(positions, k)  # 如果任何位置超过k，则偏移k
        ).long()  # 转为长整型
        idx = (  # 计算索引
            torch.add(positions, long_prompt_offset)  # 位置加上长提示偏移
            if long_prompt_offset is not None  # 如果偏移量不为空
            else positions  # 否则直接使用位置
        )
        self.long_short_cos_sin_cache: torch.Tensor = self.long_short_cos_sin_cache.to(  # 将缓存转移到索引所在设备
            idx.device
        )
        idx = torch.add(idx, offsets) if offsets is not None else idx  # 如果有偏移量则加上偏移
        cos_sin = torch.index_select(self.long_short_cos_sin_cache, 0, idx)  # 根据索引从缓存中选取余弦正弦值

        cos, sin = cos_sin.chunk(2, dim=-1)  # 将缓存分为余弦和正弦两部分
        cos = cos.repeat(1, 2).unsqueeze(-2)  # 重复余弦值并增加维度
        sin = sin.repeat(1, 2).unsqueeze(-2)  # 重复正弦值并增加维度

        query_rot = query[..., : self.rotary_dim]  # 获取查询中需要旋转的部分
        query_pass = query[..., self.rotary_dim :]  # 获取查询中不旋转的部分
        query_rot = query_rot * cos + rotate_neox(query_rot) * sin  # 应用NeoX风格旋转
        query = torch.cat((query_rot, query_pass), dim=-1)  # 拼接旋转和不旋转的部分

        key_rot = key[..., : self.rotary_dim]  # 获取键中需要旋转的部分
        key_pass = key[..., self.rotary_dim :]  # 获取键中不旋转的部分
        key_rot = key_rot * cos + rotate_neox(key_rot) * sin  # 应用NeoX风格旋转
        key = torch.cat((key_rot, key_pass), dim=-1)  # 拼接旋转和不旋转的部分

        return query.flatten(-2), key.flatten(-2)  # 返回展平后的查询和键


class FourierRotaryEmbedding(nn.Module):  # 傅里叶旋转位置编码
    """Fourier RotaryEmbedding extended."""  # 傅里叶旋转位置编码扩展

    def __init__(  # 初始化傅里叶旋转位置编码
        self,
        head_size: int,  # 注意力头的大小
        rotary_dim: int,  # 旋转维度
        max_position_embeddings: int,  # 最大位置嵌入数
        base: int,  # 频率基数
        is_neox_style: bool,  # 是否使用NeoX风格
        dtype: torch.dtype,  # 数据类型
        num_kv_heads: int,  # KV头的数量
        *,
        fope_init_factor: float = 0.1,  # 傅里叶RoPE初始化因子
        fope_sep_head: bool = True,  # 是否为每个头使用独立的傅里叶参数
        num_inv_freq: int = None,  # 逆频率的数量
        device: Optional[str] = "cuda",  # 设备
    ) -> None:
        self.fope_init_factor = fope_init_factor  # 保存傅里叶RoPE初始化因子
        self.fope_sep_head = fope_sep_head  # 保存是否使用独立头的标志
        self.num_inv_freq = num_inv_freq  # 保存逆频率数量
        self.num_kv_heads = num_kv_heads  # 保存KV头数量
        self.device = device  # 保存设备

        super().__init__()  # 调用父类初始化
        self.head_size = head_size  # 保存注意力头大小
        self.rotary_dim = rotary_dim  # 保存旋转维度
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置嵌入数
        self.base = base  # 保存频率基数
        self.is_neox_style = is_neox_style  # 保存NeoX风格标志
        self.dtype = dtype  # 保存数据类型

        self.inv_freq: torch.Tensor  # 逆频率张量
        self.register_buffer(  # 注册逆频率为缓冲区
            "inv_freq", self._compute_inv_freq(self.base), persistent=False
        )
        self.input_dim = self.inv_freq.shape[-1]  # 逆频率的输入维度
        self.output_dim = self.inv_freq.shape[-1]  # 逆频率的输出维度
        self.cos_coef = nn.Parameter(  # 余弦系数参数
            torch.empty(
                self.num_kv_heads, self.input_dim, self.output_dim, dtype=torch.float32  # 形状为(num_kv_heads, input_dim, output_dim)
            ),
            requires_grad=False,  # 不计算梯度
        )
        self.sin_coef = nn.Parameter(  # 正弦系数参数
            torch.empty(
                self.num_kv_heads, self.input_dim, self.output_dim, dtype=torch.float32  # 形状为(num_kv_heads, input_dim, output_dim)
            ),
            requires_grad=False,  # 不计算梯度
        )
        self.cos_sin_cache: torch.Tensor  # 余弦正弦缓存张量
        self.register_buffer(  # 注册余弦正弦缓存为缓冲区
            "cos_sin_cache", self._compute_cos_sin_cache(), persistent=False
        )
        self.update_buffer = False  # 缓冲区更新标志，初始化为False

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:  # 计算逆频率
        inv_freq = 1.0 / (  # 计算逆频率
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.int64).to(  # 生成0到rotary_dim步长为2的序列
                    device=self.device, dtype=torch.float  # 转换设备和数据类型
                )
                / self.rotary_dim  # 除以旋转维度
            )
        )
        assert (  # 断言逆频率为递减序列
            inv_freq[:-1] > inv_freq[1:]
        ), "Expected inv_freq to be in decreasing order"  # 期望逆频率为递减顺序
        inv_freq_idx_selected = torch.ones_like(inv_freq, dtype=torch.bool)  # 初始化全True的选择掩码
        if self.num_inv_freq is not None:  # 如果指定了逆频率数量
            inv_freq_idx_selected[self.num_inv_freq :] = False  # 只保留前num_inv_freq个
        else:  # 未指定逆频率数量
            inv_freq_idx_selected = inv_freq > (  # 选择大于阈值的逆频率
                2.0 * torch.pi / self.max_position_embeddings  # 阈值为2π/最大位置数
            )
        inv_freq = inv_freq[inv_freq_idx_selected]  # 根据选择掩码过滤逆频率
        return inv_freq  # 返回过滤后的逆频率

    def _compute_cos_sin_cache(self) -> torch.Tensor:  # 计算余弦正弦缓存
        t = torch.arange(  # 生成位置序列
            self.max_position_embeddings, dtype=torch.float, device=self.device
        )
        freqs = torch.einsum("i,j -> ij", t, self.inv_freq)  # 计算位置-频率矩阵
        if self.fope_sep_head:  # 如果使用独立头
            pos_cos = freqs.cos().unsqueeze(0).expand(self.num_kv_heads, -1, -1)  # 扩展余弦值到所有头
            pos_sin = freqs.sin().unsqueeze(0).expand(self.num_kv_heads, -1, -1)  # 扩展正弦值到所有头
        else:  # 不使用独立头
            pos_cos = freqs.cos()  # 直接使用余弦值
            pos_sin = freqs.sin()  # 直接使用正弦值
        if self.fope_sep_head:  # 如果使用独立头
            sin = torch.einsum("htD, hDd -> thd", pos_sin, self.sin_coef.float())  # 计算加权的正弦值
            cos = torch.einsum("htD, hDd -> thd", pos_cos, self.cos_coef.float())  # 计算加权的余弦值
        else:  # 不使用独立头
            sin = torch.einsum("tD, Dd -> td", pos_sin, self.sin_coef.float())  # 计算加权的正弦值
            cos = torch.einsum("tD, Dd -> td", pos_cos, self.cos_coef.float())  # 计算加权的余弦值
        sin = F.pad(  # 对正弦值进行填充
            input=sin,
            pad=(0, self.head_size // 2 - sin.size(-1)),  # 填充到head_size/2的长度
            mode="constant",  # 使用常数填充
            value=1,  # 填充值1（乘法恒等）
        )
        cos = F.pad(  # 对余弦值进行填充
            input=cos,
            pad=(0, self.head_size // 2 - cos.size(-1)),  # 填充到head_size/2的长度
            mode="constant",  # 使用常数填充
            value=1,  # 填充值1（乘法恒等）
        )
        sin = torch.cat((sin, sin), dim=-1)  # 重复正弦值以匹配完整的旋转维度
        cos = torch.cat((cos, cos), dim=-1)  # 重复余弦值以匹配完整的旋转维度
        cache = torch.cat((cos, sin), dim=-1)  # 拼接余弦和正弦值
        return cache  # 返回缓存

    def forward(  # 前向传播，应用傅里叶旋转位置编码
        self,
        positions: torch.Tensor,  # 位置张量
        query: torch.Tensor,  # 查询张量
        key: torch.Tensor,  # 键张量
        offsets: Optional[torch.Tensor] = None,  # 偏移量张量
        **kwargs,  # 其他关键字参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回旋转后的查询和键张量
        if not self.update_buffer:  # 如果缓冲区需要更新
            self.cos_sin_cache = self._compute_cos_sin_cache()  # 重新计算余弦正弦缓存
            self.update_buffer = True  # 标记缓冲区已更新
        query = query.unflatten(-1, (-1, self.head_size))  # 将查询重塑为(num_heads, head_size)
        key = key.unflatten(-1, (-1, self.head_size))  # 将键重塑为(num_heads, head_size)
        positions_with_offsets = (  # 计算带偏移的位置
            torch.add(positions, offsets) if offsets is not None else positions  # 如果有偏移则加上
        )
        cos_sin = torch.index_select(self.cos_sin_cache, 0, positions_with_offsets).to(  # 根据位置从缓存中选取
            dtype=query.dtype  # 转换为查询的数据类型
        )
        cos, sin = cos_sin.chunk(2, dim=-1)  # 将缓存分为余弦和正弦两部分
        assert (  # 断言查询和键的维度
            query.dim() == key.dim() == 3
        ), "Expected query key (seq_len, heads, head_dim)"  # 期望查询键的形状为(序列长度, 头数, 头维度)
        assert cos.dim() <= 3 and sin.dim() <= 3  # 断言余弦和正弦的维度不超过3
        need_reshape = False  # 是否需要重塑形状的标志
        if cos.dim() == 3:  # 如果余弦维度为3（使用独立头）
            need_reshape = True  # 需要重塑形状
            query_shape = query.shape  # 保存查询的原始形状
            key_shape = key.shape  # 保存键的原始形状
            cos = cos.flatten(0, 1)  # 将前两个维度展平
            sin = sin.flatten(0, 1)  # 将前两个维度展平
            seq_len = cos.size(0)  # 获取序列长度
            query = query.reshape(seq_len, -1, query.size(-1))  # 重塑查询形状
            key = key.reshape(seq_len, -1, key.size(-1))  # 重塑键形状
        query, key = apply_rotary_pos_emb_native(query, key, cos, sin)  # 应用原生旋转位置编码
        if need_reshape:  # 如果需要重塑形状
            query = query.reshape(query_shape)  # 恢复查询的原始形状
            key = key.reshape(key_shape)  # 恢复键的原始形状
        return query.flatten(-2), key.flatten(-2)  # 返回展平后的查询和键

    def extra_repr(self) -> str:  # 返回模块的额外表示字符串
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"  # 头大小和旋转维度
        s += f", max_position_embeddings={self.max_position_embeddings}"  # 最大位置嵌入数
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"  # 频率基数和风格标志
        s += f", fope_init_factor={self.fope_init_factor}, fope_sep_head={self.fope_sep_head}"  # 傅里叶RoPE参数
        s += f", num_inv_freq={self.num_inv_freq}, num_kv_heads={self.num_kv_heads}"  # 逆频率数量和KV头数
        return s  # 返回表示字符串


class DeepseekScalingRotaryEmbedding(RotaryEmbedding):  # Deepseek缩放旋转位置编码（基于YaRN方法）
    """RotaryEmbedding extended with YaRN method.  # 使用YaRN方法扩展的旋转位置编码

    Credits to Peng et al. github.com/jquesnelle/yarn  # 致谢Peng等人 github.com/jquesnelle/yarn
    """

    def __init__(  # 初始化Deepseek缩放旋转位置编码
        self,
        head_size: int,  # 注意力头的大小
        rotary_dim: int,  # 旋转维度
        max_position_embeddings: int,  # 最大位置嵌入数
        base: int,  # 频率基数
        is_neox_style: bool,  # 是否使用NeoX风格
        scaling_factor: float,  # 缩放因子
        dtype: torch.dtype,  # 数据类型
        *,
        extrapolation_factor: float = 1,  # 外推因子
        attn_factor: float = 1,  # 注意力因子
        beta_fast: int = 32,  # 快速beta参数
        beta_slow: int = 1,  # 慢速beta参数
        mscale: float = 1,  # 幅度缩放因子
        mscale_all_dim: float = 0,  # 所有维度的幅度缩放因子
        device: Optional[str] = None,  # 设备
    ) -> None:
        self.scaling_factor = scaling_factor  # 保存缩放因子
        self.extrapolation_factor = extrapolation_factor  # 保存外推因子
        self.attn_factor = attn_factor  # 保存注意力因子
        self.beta_fast = beta_fast  # 保存快速beta参数
        self.beta_slow = beta_slow  # 保存慢速beta参数
        self.mscale = float(  # 计算综合幅度缩放因子
            yarn_get_mscale(self.scaling_factor, float(mscale))  # 获取指定mscale的缩放值
            / yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))  # 除以所有维度的缩放值
            * attn_factor  # 乘以注意力因子
        )
        self.cos_cached_total = None  # 总余弦缓存，初始化为None
        self.sin_cached_total = None  # 总正弦缓存，初始化为None
        self.cos_cached = None  # 当前余弦缓存，初始化为None
        self.sin_cached = None  # 当前正弦缓存，初始化为None
        self.device = device if device is not None else get_device()  # 保存设备
        super().__init__(  # 调用父类初始化
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        if _is_hip:  # 如果是HIP平台
            self._forward_method = self.forward_native  # 使用原生前向方法

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:  # 计算带缩放的逆频率
        pos_freqs = self.base ** (  # 计算位置频率
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=self.device)  # 生成0到rotary_dim步长为2的序列
            / self.rotary_dim  # 除以旋转维度
        )
        inv_freq_extrapolation = 1.0 / pos_freqs  # 外推逆频率
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)  # 插值逆频率
        low, high = yarn_find_correction_range(  # 查找YaRN校正范围
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = (  # 计算逆频率掩码
            1
            - yarn_linear_ramp_mask(
                low, high, self.rotary_dim // 2, dtype=torch.float, device=self.device
            )
        ) * self.extrapolation_factor  # 乘以外推因子
        inv_freq = (  # 混合插值和外推的逆频率
            inv_freq_interpolation * (1 - inv_freq_mask)  # 插值部分
            + inv_freq_extrapolation * inv_freq_mask  # 外推部分
        )
        return inv_freq  # 返回混合逆频率

    def _compute_cos_sin_cache(self) -> torch.Tensor:  # 计算余弦正弦缓存
        inv_freq = self._compute_inv_freq(self.scaling_factor)  # 计算带缩放的逆频率
        t = torch.arange(  # 生成扩展的位置序列
            self.max_position_embeddings * self.scaling_factor,  # 扩展最大位置数
            device=self.device,  # 设备
            dtype=torch.float32,  # 数据类型
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 计算位置-频率矩阵
        cos = freqs.cos() * self.mscale  # 计算余弦值并乘以幅度缩放因子
        sin = freqs.sin() * self.mscale  # 计算正弦值并乘以幅度缩放因子
        cache = torch.cat((cos, sin), dim=-1)  # 拼接余弦和正弦值
        if _is_npu:  # 如果是NPU平台
            emb = torch.cat((freqs, freqs), dim=-1)  # 拼接频率
            self.cos_cached_total = torch.cos(emb) * self.mscale  # 计算总余弦缓存
            self.sin_cached_total = torch.sin(emb) * self.mscale  # 计算总正弦缓存
        return cache  # 返回缓存

    def get_cos_cached_total(self):  # 获取总余弦缓存
        return self.cos_cached_total  # 返回总余弦缓存

    def get_sin_cached_total(self):  # 获取总正弦缓存
        return self.sin_cached_total  # 返回总正弦缓存

    def get_cos_sin_cache(  # 根据位置获取余弦和正弦缓存
        self, positions, dtype, offsets: Optional[torch.Tensor] = None  # 位置、数据类型、偏移量
    ):
        self.cos_cached = (  # 根据位置索引获取余弦缓存
            self.cos_cached_total[
                torch.add(positions, offsets) if offsets is not None else positions  # 如果有偏移则加上偏移
            ]
            .unsqueeze(-2)  # 增加两个维度
            .unsqueeze(-2)
            .to(dtype)  # 转换数据类型
        )
        self.sin_cached = (  # 根据位置索引获取正弦缓存
            self.sin_cached_total[
                torch.add(positions, offsets) if offsets is not None else positions  # 如果有偏移则加上偏移
            ]
            .unsqueeze(-2)  # 增加两个维度
            .unsqueeze(-2)
            .to(dtype)  # 转换数据类型
        )
        cos = self.cos_cached.to(positions.device)  # 将余弦缓存转移到位置所在设备
        sin = self.sin_cached.to(positions.device)  # 将正弦缓存转移到位置所在设备
        return cos, sin  # 返回余弦和正弦缓存

    def forward_native(  # PyTorch原生实现的前向传播
        self,
        positions: torch.Tensor,  # 位置张量
        query: torch.Tensor,  # 查询张量
        key: torch.Tensor,  # 键张量
        offsets: Optional[torch.Tensor] = None,  # 偏移量张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回旋转后的查询和键张量
        """PyTorch-native implementation equivalent to forward()."""  # 等价于forward()的PyTorch原生实现
        dtype = query.dtype  # 保存查询的原始数据类型
        query_rot = query[..., : self.rotary_dim]  # 获取查询中需要旋转的部分
        key_rot = key[..., : self.rotary_dim]  # 获取键中需要旋转的部分
        if self.rotary_dim < self.head_size:  # 如果旋转维度小于头大小
            query_pass = query[..., self.rotary_dim :]  # 获取查询中不旋转的部分
            key_pass = key[..., self.rotary_dim :]  # 获取键中不旋转的部分
        cos_sin = self.cos_sin_cache[  # 根据位置从缓存中选取余弦正弦值
            torch.add(positions, offsets) if offsets is not None else positions  # 如果有偏移则加上偏移
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)  # 将缓存分为余弦和正弦两部分
        if self.is_neox_style:  # 如果是NeoX风格
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)  # 重复余弦值并增加维度
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)  # 重复正弦值并增加维度
        else:  # GPT-J风格
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)  # 交错重复余弦值并增加维度
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)  # 交错重复正弦值并增加维度
        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj  # 选择旋转函数
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin  # 应用旋转位置编码到查询
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin  # 应用旋转位置编码到键
        if self.rotary_dim < self.head_size:  # 如果旋转维度小于头大小
            query = torch.cat((query_rot, query_pass), dim=-1)  # 拼接旋转和不旋转的查询
            key = torch.cat((key_rot, key_pass), dim=-1)  # 拼接旋转和不旋转的键
        else:  # 旋转维度等于头大小
            query = query_rot  # 直接使用旋转后的查询
            key = key_rot  # 直接使用旋转后的键
        return query.to(dtype), key.to(dtype)  # 转换数据类型后返回

    def forward_npu(  # NPU平台的前向传播
        self,
        positions: torch.Tensor,  # 位置张量
        query: torch.Tensor,  # 查询张量
        key: torch.Tensor,  # 键张量
        offsets: Optional[torch.Tensor] = None,  # 偏移量张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回旋转后的查询和键张量
        num_tokens, num_q_heads, _ = query.shape  # 获取token数和查询头数
        num_k_heads = key.shape[1]  # 获取键头数
        cos, sin = self.get_cos_sin_cache(positions, query.dtype, offsets)  # 获取余弦和正弦缓存
        query_rot = query[..., : self.rotary_dim]  # 获取查询中需要旋转的部分
        key_rot = key[..., : self.rotary_dim]  # 获取键中需要旋转的部分
        if self.rotary_dim < self.head_size:  # 如果旋转维度小于头大小
            query_pass = query[..., self.rotary_dim :]  # 获取查询中不旋转的部分
            key_pass = key[..., self.rotary_dim :]  # 获取键中不旋转的部分
        query_rot = torch_npu.npu_interleave_rope(  # 使用NPU交错旋转操作
            query_rot.reshape(num_tokens, num_q_heads, 1, self.rotary_dim),  # 重塑查询形状
            cos,
            sin,
        )
        key_rot = torch_npu.npu_interleave_rope(  # 使用NPU交错旋转操作
            key_rot.reshape(num_tokens, num_k_heads, 1, self.rotary_dim),  # 重塑键形状
            cos,
            sin,
        )
        query_rot = query_rot.reshape(num_tokens, -1, self.rotary_dim)  # 恢复查询形状
        key_rot = key_rot.reshape(num_tokens, -1, self.rotary_dim)  # 恢复键形状
        if self.rotary_dim < self.head_size:  # 如果旋转维度小于头大小
            query = torch.cat((query_rot, query_pass), dim=-1)  # 拼接旋转和不旋转的查询
            key = torch.cat((key_rot, key_pass), dim=-1)  # 拼接旋转和不旋转的键
        else:  # 旋转维度等于头大小
            query = query_rot  # 直接使用旋转后的查询
            key = key_rot  # 直接使用旋转后的键
        return query, key  # 返回查询和键

    def forward_cpu(  # CPU平台的前向传播
        self,
        positions: torch.Tensor,  # 位置张量
        query: torch.Tensor,  # 查询张量
        key: torch.Tensor,  # 键张量
        offsets: Optional[torch.Tensor] = None,  # 偏移量张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回旋转后的查询和键张量
        positions = torch.add(positions, offsets) if offsets is not None else positions  # 如果有偏移则加上偏移
        if _is_cpu_amx_available:  # 如果CPU支持AMX指令集
            return torch.ops.sgl_kernel.rotary_embedding_cpu(  # 使用CPU优化的旋转嵌入操作
                positions, query, key, self.head_size, self.cos_sin_cache, False
            )
        else:  # CPU不支持AMX
            return self.forward_native(positions, query, key, offsets)  # 使用原生实现


class Llama3RotaryEmbedding(RotaryEmbedding):  # Llama3旋转位置编码

    def __init__(  # 初始化Llama3旋转位置编码
        self,
        head_size: int,  # 注意力头的大小
        rotary_dim: int,  # 旋转维度
        max_position_embeddings: int,  # 最大位置嵌入数
        base: int,  # 频率基数
        is_neox_style: bool,  # 是否使用NeoX风格
        dtype: torch.dtype,  # 数据类型
        scaling_factor: float,  # 缩放因子
        low_freq_factor: float,  # 低频因子
        high_freq_factor: float,  # 高频因子
        orig_max_position: int,  # 原始最大位置数
    ) -> None:
        self.scaling_factor = scaling_factor  # 保存缩放因子
        self.low_freq_factor = low_freq_factor  # 保存低频因子
        self.high_freq_factor = high_freq_factor  # 保存高频因子
        self.orig_max_position = orig_max_position  # 保存原始最大位置数
        super().__init__(  # 调用父类初始化
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:  # 计算Llama3的逆频率
        inv_freqs = super()._compute_inv_freq(base)  # 获取基础逆频率
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor  # 计算低频波长阈值
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor  # 计算高频波长阈值
        wave_len = 2 * math.pi / inv_freqs  # 计算每个频率的波长
        if self.low_freq_factor != self.high_freq_factor:  # 如果低频因子和高频因子不同
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (  # 计算平滑因子
                self.high_freq_factor - self.low_freq_factor
            )
        else:  # 低频因子等于高频因子
            smooth = 0  # 平滑因子为0
        new_freqs = torch.where(  # 根据波长应用不同的缩放策略
            wave_len < high_freq_wavelen,  # 高频（短波长）不缩放
            inv_freqs,
            torch.where(
                wave_len > low_freq_wavelen,  # 低频（长波长）除以缩放因子
                inv_freqs / self.scaling_factor,
                (1 - smooth) * inv_freqs / self.scaling_factor + smooth * inv_freqs,  # 中间频率平滑过渡
            ),
        )
        return new_freqs  # 返回新的逆频率


class Llama4VisionRotaryEmbedding(RotaryEmbedding):  # Llama4视觉旋转位置编码

    def __init__(  # 初始化Llama4视觉旋转位置编码
        self,
        head_size: int,  # 注意力头的大小
        rotary_dim: int,  # 旋转维度
        max_position_embeddings: int,  # 最大位置嵌入数
        base: int,  # 频率基数
        is_neox_style: bool,  # 是否使用NeoX风格
        dtype: torch.dtype,  # 数据类型
    ):
        super().__init__(  # 调用父类初始化
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:  # 计算Llama4视觉的逆频率
        inv_freqs = super()._compute_inv_freq(base)  # 获取基础逆频率
        inv_freqs = inv_freqs[: (self.rotary_dim // 2)]  # 只取前rotary_dim/2个逆频率
        return inv_freqs  # 返回截断后的逆频率

    def _compute_cos_sin_cache(self) -> torch.Tensor:  # 计算Llama4视觉的余弦正弦缓存
        inv_freq = self._compute_inv_freq(self.base)  # 计算逆频率
        num_patches = self.max_position_embeddings  # 补丁数量等于最大位置数
        img_idx = torch.arange(num_patches, dtype=torch.int32).reshape(num_patches, 1)  # 生成图像索引
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)  # 在末尾添加CLS token索引
        img_idx[-1, -1] = -2  # set to ID_CLS_TOKEN  # 设置为CLS_TOKEN的ID
        num_patches_single_dim = int(math.sqrt(num_patches))  # 计算单维度的补丁数（假设为正方形）
        frequencies_x = img_idx % num_patches_single_dim  # 计算x方向的频率
        frequencies_y = img_idx // num_patches_single_dim  # 计算y方向的频率
        freqs_x = (  # 计算x方向的频率矩阵
            (frequencies_x + 1)[..., None] * inv_freq[None, None, :]  # 频率乘以逆频率
        ).repeat_interleave(2, dim=-1)  # 交错重复
        freqs_y = (  # 计算y方向的频率矩阵
            (frequencies_y + 1)[..., None] * inv_freq[None, None, :]  # 频率乘以逆频率
        ).repeat_interleave(2, dim=-1)  # 交错重复
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]  # 拼接并取偶数索引
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)  # 将CLS token位置填充为0
        cache = torch.view_as_complex(  # 将频率转换为复数形式的缓存
            torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # 堆叠余弦和正弦
        )
        return cache  # 返回缓存

    def forward(  # 前向传播，应用Llama4视觉旋转位置编码
        self,
        query: torch.Tensor,  # 查询张量
        key: torch.Tensor,  # 键张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回旋转后的查询和键张量
        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(query.device)  # 将缓存转移到查询所在设备
        query_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))  # 将查询视为复数
        key_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))  # 将键视为复数
        broadcast_shape = [  # 计算广播形状
            d if i == 1 or i == (query_.ndim - 1) else 1  # 只在第1维和最后1维保持原始大小
            for i, d in enumerate(query_.shape)
        ]
        freqs_ci = self.cos_sin_cache.view(*broadcast_shape)  # 将缓存重塑为广播形状
        query_out = torch.view_as_real(query_ * freqs_ci).flatten(3)  # 复数乘法后转为实数
        key_out = torch.view_as_real(key_ * freqs_ci).flatten(3)  # 复数乘法后转为实数
        return query_out.type_as(query), key_out.type_as(key)  # 转换数据类型后返回


class DynamicNTKAlphaRotaryEmbedding(RotaryEmbedding):  # 带Alpha的动态NTK缩放旋转位置编码
    """RotaryEmbedding extended with Dynamic NTK scaling.  # 使用动态NTK缩放扩展的旋转位置编码

    Credits to the Reddit users /u/bloc97 and /u/emozilla  # 致谢Reddit用户/u/bloc97和/u/emozilla
    """

    def __init__(  # 初始化动态NTK-Alpha缩放旋转位置编码
        self,
        head_size: int,  # 注意力头的大小
        rotary_dim: int,  # 旋转维度
        max_position_embeddings: int,  # 最大位置嵌入数
        base: int,  # 频率基数
        is_neox_style: bool,  # 是否使用NeoX风格
        scaling_alpha: float,  # 缩放Alpha参数
        dtype: torch.dtype,  # 数据类型
    ) -> None:
        self.scaling_alpha = scaling_alpha  # 保存缩放Alpha参数
        super().__init__(  # 调用父类初始化
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:  # 计算动态NTK-Alpha的余弦正弦缓存
        max_len = self.max_position_embeddings  # 获取最大长度
        base = self.base * self.scaling_alpha ** (  # 计算调整后的基数
            self.rotary_dim / (self.rotary_dim - 2)  # Alpha缩放公式
        )
        inv_freq = self._compute_inv_freq(base)  # 使用调整后的基数计算逆频率
        t = torch.arange(max_len, dtype=torch.float)  # 生成位置序列
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 计算位置-频率矩阵
        cos = freqs.cos()  # 计算余弦值
        sin = freqs.sin()  # 计算正弦值
        cache = torch.cat((cos, sin), dim=-1)  # 拼接余弦和正弦值
        return cache  # 返回缓存


class DualChunkRotaryEmbedding(MultiPlatformOp):  # 双块旋转位置编码（用于Dual Chunk Attention）
    """Rotary positional embedding for Dual Chunk Attention."""  # 双块注意力机制的旋转位置编码

    def __init__(  # 初始化双块旋转位置编码
        self,
        head_size: int,  # 注意力头的大小
        rotary_dim: int,  # 旋转维度
        max_position_embeddings: int,  # 最大位置嵌入数
        base: int,  # 频率基数
        is_neox_style: bool,  # 是否使用NeoX风格
        dtype: torch.dtype,  # 数据类型
        chunk_size: int,  # 块大小
        local_size: int,  # 局部大小
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.head_size = head_size  # 保存注意力头大小
        self.rotary_dim = rotary_dim  # 保存旋转维度
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置嵌入数
        self.base = base  # 保存频率基数
        self.is_neox_style = is_neox_style  # 保存NeoX风格标志
        self.chunk_size = chunk_size  # 保存块大小
        self.local_size = local_size  # 保存局部大小
        self.dtype = dtype  # 保存数据类型
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")  # 设置CUDA设备
        q_cache, qc_cache, k_cache, qc_no_clamp_cache, q_inter_cache = (  # 计算五组余弦正弦缓存
            self._compute_cos_sin_cache()
        )
        self.register_buffer("cos_sin_q_cache", q_cache, persistent=False)  # 注册查询缓存
        self.register_buffer("cos_sin_qc_cache", qc_cache, persistent=False)  # 注册查询后续块缓存
        self.register_buffer("cos_sin_k_cache", k_cache, persistent=False)  # 注册键缓存
        self.register_buffer(  # 注册无钳位的查询后续块缓存
            "cos_sin_qc_no_clamp_cache", qc_no_clamp_cache, persistent=False
        )
        self.register_buffer("cos_sin_q_inter_cache", q_inter_cache, persistent=False)  # 注册查询跨块缓存

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:  # 计算逆频率
        inv_freq = 1.0 / (  # 计算逆频率
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim  # 生成频率序列
            )
        )
        return inv_freq  # 返回逆频率

    def _compute_cos_sin_cache(self) -> torch.Tensor:  # 计算双块旋转位置编码的余弦正弦缓存
        inv_freq = self._compute_inv_freq(self.base)  # 计算逆频率
        chunk_len = self.chunk_size - self.local_size  # 计算块的有效长度
        q_t = torch.arange(chunk_len, dtype=torch.float)  # 查询的时间序列
        qc_t = (torch.arange(chunk_len, dtype=torch.float) + chunk_len).clamp(  # 查询后续块的时间序列（钳位到chunk_size）
            max=self.chunk_size
        )
        k_t = torch.arange(self.max_position_embeddings, dtype=torch.float) % chunk_len  # 键的时间序列（取模）
        qc_no_clamp_t = torch.arange(chunk_len, dtype=torch.float) + chunk_len  # 无钳位的查询后续块时间序列
        q_inter_t = torch.arange(chunk_len, dtype=torch.float) + self.chunk_size  # 跨块查询的时间序列

        q_freqs = torch.outer(q_t, inv_freq)  # 查询的频率矩阵
        qc_freqs = torch.outer(qc_t, inv_freq)  # 查询后续块的频率矩阵
        k_freqs = torch.outer(k_t, inv_freq)  # 键的频率矩阵
        qc_no_clamp_freqs = torch.outer(qc_no_clamp_t, inv_freq)  # 无钳位查询后续块的频率矩阵
        q_inter_freqs = torch.outer(q_inter_t, inv_freq)  # 跨块查询的频率矩阵

        q_cache = torch.cat((q_freqs.cos(), q_freqs.sin()), dim=-1).to(  # 查询缓存
            dtype=self.dtype, device=self.device
        )
        qc_cache = torch.cat((qc_freqs.cos(), qc_freqs.sin()), dim=-1).to(  # 查询后续块缓存
            dtype=self.dtype, device=self.device
        )
        k_cache = torch.cat((k_freqs.cos(), k_freqs.sin()), dim=-1).to(  # 键缓存
            dtype=self.dtype, device=self.device
        )
        qc_no_clamp_cache = torch.cat(  # 无钳位查询后续块缓存
            (qc_no_clamp_freqs.cos(), qc_no_clamp_freqs.sin()), dim=-1
        ).to(dtype=self.dtype, device=self.device)
        q_inter_cache = torch.cat(  # 跨块查询缓存
            (q_inter_freqs.cos(), q_inter_freqs.sin()), dim=-1
        ).to(dtype=self.dtype, device=self.device)
        return q_cache, qc_cache, k_cache, qc_no_clamp_cache, q_inter_cache  # 返回五组缓存

    def forward(  # 前向传播，应用双块旋转位置编码
        self,
        positions: torch.Tensor,  # 位置张量
        query: torch.Tensor,  # 查询张量
        key: torch.Tensor,  # 键张量
        offsets: Optional[torch.Tensor] = None,  # 偏移量张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回旋转后的查询和键张量
        query = query.view(*query.shape[:-1], -1, self.head_size)  # 重塑查询形状
        key = key.view(*key.shape[:-1], -1, self.head_size)  # 重塑键形状
        query_rot = query[..., : self.rotary_dim]  # 获取查询中需要旋转的部分
        key_rot = key[..., : self.rotary_dim]  # 获取键中需要旋转的部分
        if self.rotary_dim < self.head_size:  # 如果旋转维度小于头大小
            query_pass = query[..., self.rotary_dim :]  # 获取查询中不旋转的部分
            key_pass = key[..., self.rotary_dim :]  # 获取键中不旋转的部分
        else:  # 旋转维度等于头大小
            query_pass = None  # 无需保留不旋转部分
            key_pass = None  # 无需保留不旋转部分

        positions_with_offsets = (  # 计算带偏移的位置
            torch.add(positions, offsets) if offsets is not None else positions  # 如果有偏移则加上
        )
        key = self._apply_rotary_embedding(  # 对键应用旋转编码
            self.cos_sin_k_cache[positions_with_offsets], key_rot, key_pass
        )
        chunk_len = self.chunk_size - self.local_size  # 计算块的有效长度
        query = self._apply_rotary_embedding(  # 对查询应用块内旋转编码
            self.cos_sin_q_cache[positions_with_offsets % chunk_len],
            query_rot,
            query_pass,
        )
        query_succ = self._apply_rotary_embedding(  # 对查询应用后续块旋转编码
            self.cos_sin_qc_cache[positions_with_offsets % chunk_len],
            query_rot,
            query_pass,
        )
        query_inter = self._apply_rotary_embedding(  # 对查询应用跨块旋转编码（使用最后一个块位置）
            self.cos_sin_qc_cache[chunk_len - 1].repeat(positions.shape[0], 1),
            query_rot,
            query_pass,
        )
        query_succ_critical = self._apply_rotary_embedding(  # 对查询应用无钳位后续块旋转编码
            self.cos_sin_qc_no_clamp_cache[positions_with_offsets % chunk_len],
            query_rot,
            query_pass,
        )
        query_inter_critical = self._apply_rotary_embedding(  # 对查询应用跨块关键旋转编码
            self.cos_sin_q_inter_cache[positions_with_offsets % chunk_len],
            query_rot,
            query_pass,
        )
        query = torch.cat(  # 拼接所有查询变体
            (query, query_succ, query_inter, query_succ_critical, query_inter_critical),
            dim=-1,
        )
        return query, key  # 返回查询和键

    def _apply_rotary_embedding(self, cos_sin, hidden_rot, hidden_pass):  # 应用旋转位置编码到隐藏状态
        cos, sin = cos_sin.chunk(2, dim=-1)  # 将缓存分为余弦和正弦两部分
        if self.is_neox_style:  # 如果是NeoX风格
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)  # 重复余弦值并增加维度
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)  # 重复正弦值并增加维度
        else:  # GPT-J风格
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)  # 交错重复余弦值并增加维度
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)  # 交错重复正弦值并增加维度
        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj  # 选择旋转函数
        hidden_rot = hidden_rot * cos + rotate_fn(hidden_rot) * sin  # 应用旋转位置编码
        if self.rotary_dim < self.head_size:  # 如果旋转维度小于头大小
            hidden = torch.cat((hidden_rot, hidden_pass), dim=-1)  # 拼接旋转和不旋转的部分
        else:  # 旋转维度等于头大小
            hidden = hidden_rot  # 直接使用旋转后的结果
        return hidden.flatten(-2).squeeze(0)  # 展平并去除批维度

    def extra_repr(self) -> str:  # 返回模块的额外表示字符串
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"  # 头大小和旋转维度
        s += f", max_position_embeddings={self.max_position_embeddings}"  # 最大位置嵌入数
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"  # 频率基数和风格标志
        s += f", chunk_size={self.chunk_size}, local_size={self.local_size}"  # 块大小和局部大小
        return s  # 返回表示字符串


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):  # 动态NTK缩放旋转位置编码
    """RotaryEmbedding extended with Dynamic NTK scaling.  # 使用动态NTK缩放扩展的旋转位置编码

    Credits to the Reddit users /u/bloc97 and /u/emozilla  # 致谢Reddit用户/u/bloc97和/u/emozilla
    """

    def __init__(  # 初始化动态NTK缩放旋转位置编码
        self,
        head_size: int,  # 注意力头的大小
        rotary_dim: int,  # 旋转维度
        max_position_embeddings: int,  # 最大位置嵌入数
        base: int,  # 频率基数
        is_neox_style: bool,  # 是否使用NeoX风格
        scaling_factor: float,  # 缩放因子
        dtype: torch.dtype,  # 数据类型
    ) -> None:
        self.scaling_factor = scaling_factor  # 保存缩放因子
        super().__init__(  # 调用父类初始化
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:  # 计算动态NTK缩放的余弦正弦缓存
        max_len = self.max_position_embeddings * self.scaling_factor  # 计算缩放后的最大长度
        base = self.base * (  # 计算调整后的基数
            (self.scaling_factor * max_len / self.max_position_embeddings)  # 缩放比例
            - (self.scaling_factor - 1)  # 减去缩放因子偏移
        ) ** (self.rotary_dim / (self.rotary_dim - 2))  # 动态NTK基数调整公式
        inv_freq = self._compute_inv_freq(base)  # 使用调整后的基数计算逆频率
        t = torch.arange(max_len, dtype=torch.float)  # 生成位置序列
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 计算位置-频率矩阵
        cos = freqs.cos()  # 计算余弦值
        sin = freqs.sin()  # 计算正弦值
        cache = torch.cat((cos, sin), dim=-1)  # 拼接余弦和正弦值
        return cache  # 返回缓存


class Gemma4RotaryEmbedding(RotaryEmbedding):  # Gemma4专用旋转位置编码（带交叉混合）
    """Gemma4-specific RoPE with cross-mixing.  # Gemma4专用的带交叉混合的RoPE

    Instead of rotating the first `rotary_dim` dimensions contiguously,  # 不是连续旋转前rotary_dim个维度
    splits the head into two halves and applies rotation across both.  # 而是将头分成两半并在两半之间交叉应用旋转

    For a head_dim of D and rotary_dim of R:  # 对于head_dim为D、rotary_dim为R的情况：
    - Standard RoPE rotates: [0, R)  # 标准RoPE旋转：[0, R)
    - Gemma4 RoPE rotates: [0, R/2) cross-mixed with [D/2, D/2 + R/2)  # Gemma4 RoPE旋转：[0, R/2)与[D/2, D/2 + R/2)交叉混合
    """

    def __init__(  # 初始化Gemma4旋转位置编码
        self,
        head_size: int,  # 注意力头的大小
        rotary_dim: int,  # 旋转维度
        max_position_embeddings: int,  # 最大位置嵌入数
        base: float,  # 频率基数
        is_neox_style: bool,  # 是否使用NeoX风格
        dtype: torch.dtype,  # 数据类型
    ) -> None:
        # Store angles before calling super().__init__  # 在调用super().__init__之前存储角度
        # rotary_dim is already scaled by partial_rotary_factor in get_rope  # rotary_dim已在get_rope中被partial_rotary_factor缩放
        # For Gemma4: head_size=512, partial_rotary_factor=0.25 -> rotary_dim=128  # 对于Gemma4：head_size=512, partial_rotary_factor=0.25 -> rotary_dim=128
        self.rope_angles = rotary_dim // 2  # Number of rotation angles per half  # 每半部分的旋转角度数
        self.nope_angles = (head_size // 2) - self.rope_angles  # Non-rotated per half  # 每半部分的非旋转角度数

        super().__init__(  # 调用父类初始化，注意rotary_dim被设为head_size
            head_size,
            head_size,  # 使用head_size作为rotary_dim（因为交叉混合需要完整维度）
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:  # 计算Gemma4的逆频率
        """Compute frequencies only for the rotated dimensions.  # 仅计算旋转维度的频率

        Non-rotated dims are padded with 0.0 to produce identity rotation.  # 非旋转维度用0.0填充以产生恒等旋转
        """
        freq_exponents = (  # 计算频率指数
            torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.float) / self.head_size  # 步长为2，除以head_size
        )
        inv_freq = 1.0 / (base**freq_exponents)  # 计算逆频率

        # Zero-pad for non-rotated dims (identity rotation: cos=1, sin=0)  # 为非旋转维度零填充（恒等旋转：cos=1, sin=0）
        if self.nope_angles > 0:  # 如果存在非旋转角度
            inv_freq = torch.cat(  # 拼接逆频率和零填充
                [
                    inv_freq,
                    torch.zeros(self.nope_angles, dtype=torch.float),  # 零填充
                ]
            )
        return inv_freq  # 返回逆频率

    def extra_repr(self) -> str:  # 返回模块的额外表示字符串
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"  # 头大小和旋转维度
        s += f", rope_angles={self.rope_angles}, nope_angles={self.nope_angles}"  # 旋转角度数和非旋转角度数
        s += f", max_position_embeddings={self.max_position_embeddings}"  # 最大位置嵌入数
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"  # 频率基数和风格标志
        return s  # 返回表示字符串
