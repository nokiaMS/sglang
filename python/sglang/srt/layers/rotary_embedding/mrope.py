# 多模态旋转位置编码（MRoPE）模块，实现MRotaryEmbedding、YaRNScalingMRotaryEmbedding、Ernie4_5_VLRotaryEmbedding及交错RoPE核函数
"""MRotaryEmbedding, YaRNScalingMRotaryEmbedding, Ernie4_5_VLRotaryEmbedding,  # MRotaryEmbedding、YaRNScalingMRotaryEmbedding、Ernie4_5_VLRotaryEmbedding，
apply_interleaved_rope for multimodal RoPE.  # apply_interleaved_rope用于多模态RoPE。"""

from __future__ import annotations  # 启用延迟注解求值 # 启用延迟注解

from typing import List, Optional, Tuple  # 导入类型注解 # 导入类型提示

import torch  # 导入PyTorch # 导入PyTorch框架

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding  # 导入基础旋转位置编码类 # 导入基础RoPE类
from sglang.srt.layers.rotary_embedding.triton_kernels import (  # 导入Triton核函数 # 导入Triton核函数
    triton_ernie45_rope_fused_inplace,
    triton_mrope_fused,
)
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb  # 导入旋转编码应用工具 # 导入旋转编码应用函数
from sglang.srt.layers.rotary_embedding.yarn import (  # 导入YaRN相关函数 # 导入YaRN相关函数
    yarn_find_correction_range,
    yarn_get_mscale_simple,
    yarn_linear_ramp_mask,
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数 # 导入全局服务器参数
from sglang.srt.utils import cpu_has_amx_support, is_cuda, is_npu, support_triton  # 导入平台检测工具 # 导入平台检测函数

_is_cuda = is_cuda()  # 是否为CUDA平台 # 是否为CUDA平台
_is_npu = is_npu()  # 是否为NPU平台 # 是否为NPU平台
_is_cpu_amx_available = cpu_has_amx_support()  # CPU是否支持AMX指令集 # CPU是否支持AMX

if _is_cuda:  # CUDA平台导入JIT核函数 # CUDA平台导入JIT核函数
    from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace  # 导入CUDA原地RoPE核函数 # 导入CUDA原地RoPE核函数

if _is_npu:  # NPU平台导入扩展 # NPU平台导入扩展
    import torch_npu  # 导入华为NPU扩展 # 导入torch_npu


import triton  # 导入Triton # 导入Triton框架
import triton.language as tl  # 导入Triton语言 # 导入Triton语言


@triton.jit  # Triton JIT编译的交错RoPE核函数 # Triton JIT编译的交错RoPE核函数
def apply_interleaved_rope_kernel(
    x_ptr,  # 输入指针 # 输入指针
    out_ptr,  # 输出指针 # 输出指针
    S: tl.constexpr,  # 序列长度（编译时常量） # 序列长度
    D: tl.constexpr,  # 维度大小（编译时常量） # 维度大小
    stride_x_m,  # x的模态维步长 # x模态维步长
    stride_x_s,  # x的序列维步长 # x序列维步长
    stride_out_s,  # 输出的序列维步长 # 输出序列维步长
    section_1_end,  # 第1段结束位置 # 第1段结束位置
    section_2_end,  # 第2段结束位置 # 第2段结束位置
    BLOCK_S: tl.constexpr,  # 序列维块大小（编译时常量） # 序列维块大小
    BLOCK_SIZE: tl.constexpr,  # 维度块大小（编译时常量） # 维度块大小
):
    start_s = tl.program_id(0) * BLOCK_S  # 计算序列维起始偏移 # 计算序列维起始偏移
    s_offsets = start_s + tl.arange(0, BLOCK_S)  # 生成序列维偏移 # 生成序列维偏移

    dim_offset = tl.program_id(1) * BLOCK_SIZE  # 计算维度起始偏移 # 计算维度起始偏移
    dim_indices = dim_offset + tl.arange(0, BLOCK_SIZE)  # 生成维度索引 # 生成维度索引

    mask_s = s_offsets < S  # 序列维掩码 # 序列维掩码
    mask_d = dim_indices < D  # 维度掩码 # 维度掩码
    mask = mask_s[:, None] & mask_d[None, :]  # 组合掩码 # 组合掩码

    val_ptr = (  # 计算模态0的输入指针 # 模态0的输入指针
        x_ptr + 0 * stride_x_m + s_offsets[:, None] * stride_x_s + dim_indices[None, :]
    )
    val = tl.load(val_ptr, mask=mask, other=0.0)  # 加载模态0的值 # 加载模态0的值

    cond_a = (dim_indices[None, :] % 3 == 1) & (  # 判断是否属于模态1的位置 # 判断是否属于模态1
        dim_indices[None, :] < section_1_end * 3
    )
    val_a_ptr = (  # 计算模态1的输入指针 # 模态1的输入指针
        x_ptr + 1 * stride_x_m + s_offsets[:, None] * stride_x_s + dim_indices[None, :]
    )
    val_a = tl.load(val_a_ptr, mask=mask & cond_a, other=0.0)  # 加载模态1的值 # 加载模态1的值

    cond_b = (dim_indices[None, :] % 3 == 2) & (  # 判断是否属于模态2的位置 # 判断是否属于模态2
        dim_indices[None, :] < section_2_end * 3
    )
    val_b_ptr = (  # 计算模态2的输入指针 # 模态2的输入指针
        x_ptr + 2 * stride_x_m + s_offsets[:, None] * stride_x_s + dim_indices[None, :]
    )
    val_b = tl.load(val_b_ptr, mask=mask & cond_b, other=0.0)  # 加载模态2的值 # 加载模态2的值

    val = tl.where(cond_a, val_a, val)  # 根据条件选择模态1的值 # 根据条件选择模态1
    val = tl.where(cond_b, val_b, val)  # 根据条件选择模态2的值 # 根据条件选择模态2

    out_ptr = out_ptr + s_offsets[:, None] * stride_out_s + dim_indices[None, :]  # 计算输出指针 # 计算输出指针
    tl.store(out_ptr, val, mask=mask)  # 存储结果 # 存储结果


def apply_interleaved_rope_triton(x: torch.Tensor, mrope_section: list) -> torch.Tensor:  # 使用Triton核函数应用交错RoPE # 使用Triton核函数应用交错RoPE
    x = x.contiguous()  # 确保内存连续 # 确保内存连续
    M, S, D = x.shape  # 解包形状：模态数、序列长度、维度 # 解包形状

    out = torch.empty((S, D), dtype=x.dtype, device=x.device)  # 分配输出张量 # 分配输出张量

    BLOCK_S = 64  # 序列维块大小 # 序列维块大小
    BLOCK_SIZE = 128  # 维度块大小 # 维度块大小

    grid = (triton.cdiv(S, BLOCK_S), triton.cdiv(D, BLOCK_SIZE))  # 计算网格大小 # 计算网格大小

    section_1_end = mrope_section[1]  # 第1段结束位置 # 第1段结束位置
    section_2_end = mrope_section[2]  # 第2段结束位置 # 第2段结束位置

    apply_interleaved_rope_kernel[grid](  # 启动Triton核函数 # 启动Triton核函数
        x,
        out,
        S,
        D,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        section_1_end,
        section_2_end,
        BLOCK_S=BLOCK_S,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out  # 返回输出 # 返回输出


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:  # PyTorch原生实现交错RoPE # PyTorch原生实现交错RoPE
    x_t = x[0].clone()  # 克隆模态0作为基础 # 克隆模态0
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]  # 用模态1替换对应位置 # 替换模态1位置
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]  # 用模态2替换对应位置 # 替换模态2位置
    return x_t  # 返回交错后的张量 # 返回交错结果


class MRotaryEmbedding(RotaryEmbedding):  # 多模态旋转位置编码类，继承RotaryEmbedding # 多模态旋转位置编码类
    """Rotary Embedding with Multimodal Sections.  # 带多模态段的旋转位置编码。"""

    def __init__(  # 初始化多模态旋转位置编码 # 初始化方法
        self,
        head_size: int,  # 注意力头大小 # 注意力头维度
        rotary_dim: int,  # 旋转维度 # 旋转编码维度
        max_position_embeddings: int,  # 最大位置编码数 # 最大位置嵌入数
        base: int,  # 旋转基频 # 旋转基频
        is_neox_style: bool,  # 是否为NeoX风格 # 是否NeoX风格
        dtype: torch.dtype,  # 数据类型 # 数据类型
        mrope_section: Optional[List[int]] = None,  # 多模态RoPE段配置 # 多模态RoPE段配置
        mrope_interleaved: bool = False,  # 是否使用交错模式 # 是否交错模式
        mrope_interleaved_glm: bool = False,  # 是否使用GLM交错模式 # 是否GLM交错模式
    ) -> None:
        super().__init__(  # 调用父类初始化 # 调用父类初始化
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        self.mrope_section = mrope_section  # 保存多模态RoPE段配置 # 保存段配置
        self.mrope_interleaved = mrope_interleaved  # 保存交错模式标志 # 保存交错模式标志
        self.mrope_interleaved_glm = mrope_interleaved_glm  # 保存GLM交错模式标志 # 保存GLM交错标志
        if self.mrope_section:  # 验证并修正mrope_section # 验证并修正段配置
            expected_sum = rotary_dim // 2  # 期望的段总和 # 期望的段总和
            actual_sum = sum(self.mrope_section)  # 实际的段总和 # 实际的段总和
            if actual_sum != expected_sum:  # 总和不匹配 # 总和不匹配
                print(
                    f"MRoPE section sum mismatch: expected {expected_sum}, got {actual_sum}. "
                    f"Adjusting mrope_section to match rotary_dim // 2 = {expected_sum}"
                )
                if actual_sum > 0:  # 实际总和大于0时按比例缩放 # 实际总和大于0时按比例缩放
                    scale_factor = expected_sum / actual_sum  # 计算缩放因子 # 计算缩放因子
                    self.mrope_section = [
                        max(1, int(section * scale_factor))  # 按比例缩放每段，最小为1 # 按比例缩放
                        for section in self.mrope_section
                    ]
                    current_sum = sum(self.mrope_section)  # 缩放后的总和 # 缩放后总和
                    if current_sum != expected_sum:  # 缩放后仍不匹配则调整最后一段 # 缩放后不匹配调整最后一段
                        self.mrope_section[-1] += expected_sum - current_sum
                else:  # 实际总和为0时均分 # 总和为0时均分
                    self.mrope_section = [
                        expected_sum // len(self.mrope_section)  # 平均分配 # 平均分配
                    ] * len(self.mrope_section)
                    remainder = expected_sum % len(self.mrope_section)  # 余数 # 余数
                    for i in range(remainder):  # 将余数分配到前几段 # 分配余数
                        self.mrope_section[i] += 1
                print(
                    f"Corrected mrope_section: {self.mrope_section} (sum={sum(self.mrope_section)})"
                )

        # MRoPE axis_map interleaving pattern depends on mrope_section sizes.  # MRoPE axis_map交错模式取决于mrope_section大小。
        # The algorithm cycles through axes [0(T), 1(H), 2(W)] round-robin,  # 算法以轮询方式循环遍历轴[0(T), 1(H), 2(W)]，
        # skipping any axis that has exhausted its allocated pairs.  # 跳过已耗尽分配对数的轴。
        #
        # For GLM-V (mrope_section=[8,12,12]):  # 对于GLM-V (mrope_section=[8,12,12])：
        #   T(8) < H(12) = W(12), so T exhausts first at pair 24.  # T(8) < H(12) = W(12)，因此T在第24对首先耗尽。
        #   Result: [0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 1,1,2, 1,1,2, 2,2]  # 结果：[0,1,2,...1,1,2,1,1,2,2,2]
        #   After T runs out, only H and W fill the remaining slots.  # T耗尽后，只有H和W填充剩余槽位。
        #
        # For Qwen3-VL (mrope_section=[24,20,20]):  # 对于Qwen3-VL (mrope_section=[24,20,20])：
        #   T(24) > H(20) = W(20), so H and W exhaust first near the tail.  # T(24) > H(20) = W(20)，因此H和W在尾部首先耗尽。
        #   Result: [0,1,2, 0,1,2, ...repeated evenly..., 0,1, 0,1, 0,0]  # 结果：[0,1,2, 0,1,2, ...均匀重复..., 0,1, 0,1, 0,0]
        #   After H/W run out, T fills the remaining slots.  # H/W耗尽后，T填充剩余槽位。

        if self.mrope_interleaved_glm:  # GLM交错模式构建轴映射 # GLM交错模式构建轴映射
            num_pairs = rotary_dim // 2  # 旋转对数 # 旋转对数
            axis_map = torch.empty(num_pairs, dtype=torch.long)  # 分配轴映射张量 # 分配轴映射
            assert sum(self.mrope_section) == num_pairs  # 断言段总和等于旋转对数 # 断言段总和
            counts = [0, 0, 0]  # 各轴已分配计数 # 各轴已分配计数
            current_ax = 0  # 当前轴 # 当前轴

            for i in range(num_pairs):  # 遍历每对 # 遍历每对
                current_ax = i % 3  # 轮询选择轴 # 轮询选择轴
                while counts[current_ax] >= self.mrope_section[current_ax]:  # 跳过已耗尽的轴 # 跳过已耗尽的轴
                    current_ax = (current_ax + 1) % 3

                axis_map[i] = current_ax  # 记录轴映射 # 记录轴映射
                counts[current_ax] += 1  # 递增计数 # 递增计数
            self.register_buffer("axis_map", axis_map, persistent=False)  # 注册轴映射为缓冲区 # 注册为缓冲区
        else:
            self.axis_map = None  # 非GLM交错模式不使用轴映射 # 非GLM交错模式不使用轴映射
        if get_global_server_args().rl_on_policy_target is not None:  # RL训练模式 # RL训练模式
            self._forward_method = self.forward_native  # 使用原生前向方法 # 使用原生前向方法

    def get_cos_sin_with_position(self, positions):  # 根据位置获取多模态cos/sin值 # 根据位置获取cos/sin
        if positions.ndim == 1:  # 一维位置使用父类方法 # 一维位置使用父类方法
            return super().get_cos_sin_with_position(positions)
        assert positions.ndim == 2  # 断言为二维位置 # 断言为二维位置
        assert self.mrope_section  # 断言mrope_section已设置 # 断言段配置已设置
        cos_sin = self.cos_sin_cache[positions]  # 按位置索引获取cos/sin # 按位置索引获取
        last_dim = cos_sin.size()[-1]  # 最后一维大小 # 最后一维大小
        cos, sin = cos_sin.chunk(2, dim=-1)  # 分离cos和sin # 分离cos和sin
        if self.mrope_interleaved:  # 交错模式 # 交错模式
            if support_triton(get_global_server_args().attention_backend):  # 支持Triton时使用Triton核函数 # 支持Triton使用Triton核函数
                cos = apply_interleaved_rope_triton(cos, self.mrope_section)  # Triton交错RoPE # Triton交错RoPE
                sin = apply_interleaved_rope_triton(sin, self.mrope_section)  # Triton交错RoPE # Triton交错RoPE
            else:
                cos = apply_interleaved_rope(cos, self.mrope_section)  # 原生交错RoPE # 原生交错RoPE
                sin = apply_interleaved_rope(sin, self.mrope_section)  # 原生交错RoPE # 原生交错RoPE
        else:  # 非交错模式按段拼接 # 非交错模式按段拼接
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],  # 按段选择对应模态的cos # 按段选择对应模态
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],  # 按段选择对应模态的sin # 按段选择对应模态
                dim=-1,
            )
        self.position_cos = cos.repeat(1, 2).view(-1, 1, 1, last_dim).contiguous()  # 保存位置cos # 保存位置cos
        self.position_sin = sin.repeat(1, 2).view(-1, 1, 1, last_dim).contiguous()  # 保存位置sin # 保存位置sin

    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> None:  # 匹配cos/sin缓存的数据类型和设备与query一致 # 匹配缓存数据类型和设备
        if (  # 如果设备或数据类型不匹配 # 如果设备或数据类型不匹配
            self.cos_sin_cache.device != query.device  # 设备不匹配 # 设备不同
            or self.cos_sin_cache.dtype != query.dtype  # 数据类型不匹配 # 数据类型不同
        ):
            self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)  # 转换缓存 # 转换缓存

    def forward_native(  # PyTorch原生实现的多模态RoPE前向方法 # 原生PyTorch前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor,  # 键张量 # 键张量
        fused_set_kv_buffer_arg=None,  # 融合KV缓冲区参数 # 融合KV缓冲区参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (  # 断言不支持融合KV缓冲区 # 断言不支持融合KV缓冲区
            fused_set_kv_buffer_arg is None
        ), "save kv cache is not supported for MRotaryEmbedding."
        assert positions.ndim == 1 or positions.ndim == 2  # 断言位置为1D或2D # 断言位置维度

        cos_sin = self.cos_sin_cache[positions]  # 按位置索引获取cos/sin # 按位置索引获取
        cos, sin = cos_sin.chunk(2, dim=-1)  # 分离cos和sin # 分离cos和sin
        if positions.ndim == 2:  # 二维位置（多模态） # 二维位置（多模态）
            assert self.mrope_section  # 断言mrope_section已设置 # 断言段配置已设置
            if self.mrope_interleaved:  # 交错模式 # 交错模式
                cos = apply_interleaved_rope(cos, self.mrope_section)  # 应用交错RoPE # 应用交错RoPE
                sin = apply_interleaved_rope(sin, self.mrope_section)  # 应用交错RoPE # 应用交错RoPE
            else:  # 非交错模式按段拼接 # 非交错模式按段拼接
                cos = torch.cat(
                    [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],  # 按段选择对应模态 # 按段选择
                    dim=-1,
                )
                sin = torch.cat(
                    [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],  # 按段选择对应模态 # 按段选择
                    dim=-1,
                )

        seq_len_q = query.shape[0]  # query序列长度 # query序列长度
        query_shape = query.shape  # 保存query原始形状 # 保存原始形状
        query = query.view(seq_len_q, -1, self.head_size)  # 重塑query为3D # 重塑query
        query_rot = query[..., : self.rotary_dim]  # 提取旋转部分 # 提取旋转部分
        query_pass = query[..., self.rotary_dim :]  # 提取不旋转部分 # 提取不旋转部分
        query_rot = apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)  # 应用旋转编码 # 应用旋转编码
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)  # 拼接并恢复形状 # 拼接恢复形状

        seq_len_k = key.shape[0]  # key序列长度 # key序列长度
        key_shape = key.shape  # 保存key原始形状 # 保存原始形状
        key = key.view(seq_len_k, -1, self.head_size)  # 重塑key为3D # 重塑key
        key_rot = key[..., : self.rotary_dim]  # 提取旋转部分 # 提取旋转部分
        key_pass = key[..., self.rotary_dim :]  # 提取不旋转部分 # 提取不旋转部分
        key_rot = apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)  # 应用旋转编码 # 应用旋转编码
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)  # 拼接并恢复形状 # 拼接恢复形状
        return query, key  # 返回旋转后的query和key # 返回结果

    def forward_cpu(  # CPU平台的多模态RoPE前向方法 # CPU平台前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor,  # 键张量 # 键张量
        fused_set_kv_buffer_arg=None,  # 融合KV缓冲区参数 # 融合KV缓冲区参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if _is_cpu_amx_available:  # CPU支持AMX指令集 # CPU支持AMX
            return torch.ops.sgl_kernel.multimodal_rotary_embedding_cpu(  # 使用CPU多模态RoPE算子 # 使用CPU多模态RoPE算子
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.mrope_section if self.mrope_section else None,
                self.mrope_interleaved,
                self.is_neox_style,
            )
        return self.forward_native(positions, query, key, fused_set_kv_buffer_arg)  # 不支持AMX则使用原生方法 # 不支持AMX回退原生

    def forward_cuda(  # CUDA平台的多模态RoPE前向方法 # CUDA平台前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor,  # 键张量 # 键张量
        fused_set_kv_buffer_arg=None,  # 融合KV缓冲区参数 # 融合KV缓冲区参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert positions.ndim == 1 or positions.ndim == 2  # 断言位置为1D或2D # 断言位置维度
        if positions.ndim == 2 and self.mrope_section:  # 二维位置且有mrope_section配置 # 二维位置且有段配置
            return self.forward_triton(positions, query, key)  # 使用Triton核函数 # 使用Triton核函数
        return self.forward_native(positions, query, key, fused_set_kv_buffer_arg)  # 一维位置使用原生方法 # 一维位置使用原生方法

    def forward_triton(  # 使用Triton核函数的多模态RoPE前向方法 # Triton核函数前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor,  # 键张量 # 键张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.mrope_section  # 断言mrope_section已设置 # 断言段配置已设置
        self._match_cos_sin_cache_dtype(query)  # 匹配缓存数据类型 # 匹配缓存数据类型
        triton_mrope_fused(  # 调用Triton融合MRoPE核函数 # 调用Triton融合MRoPE
            query,
            key,
            self.cos_sin_cache,
            positions,
            self.mrope_section,
            self.head_size,
            self.rotary_dim,
            self.mrope_interleaved,
            self.mrope_interleaved_glm,
            self.is_neox_style,
            self.axis_map,
        )
        return query, key  # 返回旋转后的query和key # 返回结果

    def forward_npu(  # NPU平台的多模态RoPE前向方法 # NPU平台前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor,  # 键张量 # 键张量
        fused_set_kv_buffer_arg=None,  # 融合KV缓冲区参数 # 融合KV缓冲区参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (  # 断言不支持融合KV缓冲区 # 断言不支持融合KV缓冲区
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for npu implementation"
        if query.shape[1] > 4096:  # 头数较大时回退到原生方法 # 头数较大回退原生方法
            return self.forward_native(positions, query, key, fused_set_kv_buffer_arg)
        rotary_mode = "half" if self.is_neox_style else "interleave"  # 设置旋转模式 # 设置旋转模式
        mrope_section = [0, 0, 0]  # 多模态段设为0 # 多模态段设为0
        query_out, key_out = torch_npu.npu_mrope(  # 调用NPU多模态RoPE算子 # 调用NPU多模态RoPE
            positions,
            query,
            key,
            self.cos_sin_cache,
            self.head_size,
            mrope_section=mrope_section,
            rotary_mode=rotary_mode,
        )
        return query_out, key_out  # 返回旋转后的query和key # 返回结果

    def forward_xpu(  # XPU平台的多模态RoPE前向方法 # XPU平台前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor,  # 键张量 # 键张量
        fused_set_kv_buffer_arg=None,  # 融合KV缓冲区参数 # 融合KV缓冲区参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert positions.ndim in (1, 2)  # 断言位置为1D或2D # 断言位置维度
        if positions.ndim == 2 and self.mrope_section:  # 二维位置且有mrope_section配置 # 二维位置且有段配置
            return self.forward_triton(positions, query, key)  # 使用Triton核函数 # 使用Triton核函数
        return self.forward_native(positions, query, key, fused_set_kv_buffer_arg)  # 一维位置使用原生方法 # 一维位置使用原生方法

    @staticmethod
    def get_rope_index(  # 获取MRoPE位置索引（静态方法） # 获取MRoPE位置索引
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        model_type,
        tokens_per_second=None,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        from sglang.srt.layers.rotary_embedding.mrope_rope_index import get_rope_index  # 延迟导入 # 延迟导入

        return get_rope_index(  # 委托给mrope_rope_index模块 # 委托给具体实现
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            model_type,
            tokens_per_second,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            **kwargs,
        )

    @staticmethod
    def get_rope_index_qwen3_omni(  # 获取Qwen3-Omni的MRoPE位置索引 # 获取Qwen3-Omni MRoPE位置索引
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        tokens_per_second=None,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        from sglang.srt.layers.rotary_embedding.mrope_rope_index import (  # 延迟导入 # 延迟导入
            get_rope_index_qwen3_omni,
        )

        return get_rope_index_qwen3_omni(  # 委托给具体实现 # 委托给具体实现
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            tokens_per_second,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            **kwargs,
        )

    @staticmethod
    def get_rope_index_glm4v(  # 获取GLM-4V的MRoPE位置索引 # 获取GLM-4V MRoPE位置索引
        input_ids, hf_config, image_grid_thw, video_grid_thw, attention_mask, **kwargs
    ):
        from sglang.srt.layers.rotary_embedding.mrope_rope_index import (  # 延迟导入 # 延迟导入
            get_rope_index_glm4v,
        )

        return get_rope_index_glm4v(  # 委托给具体实现 # 委托给具体实现
            input_ids,
            hf_config,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
            **kwargs,
        )

    @staticmethod
    def get_rope_index_ernie45(  # 获取Ernie4.5的MRoPE位置索引 # 获取Ernie4.5 MRoPE位置索引
        input_ids, hf_config, image_grid_thw, video_grid_thw, **kwargs
    ):
        from sglang.srt.layers.rotary_embedding.mrope_rope_index import (  # 延迟导入 # 延迟导入
            get_rope_index_ernie45,
        )

        return get_rope_index_ernie45(  # 委托给具体实现 # 委托给具体实现
            input_ids, hf_config, image_grid_thw, video_grid_thw, **kwargs
        )


class YaRNScalingMRotaryEmbedding(MRotaryEmbedding):  # 带YaRN缩放的多模态旋转位置编码 # YaRN缩放多模态旋转位置编码
    """MRoPE-enabled rotary embedding with YaRN context scaling.  # 支持MRoPE的YaRN上下文缩放旋转位置编码。"""

    def __init__(  # 初始化YaRN缩放多模态旋转位置编码 # 初始化方法
        self,
        head_size: int,  # 注意力头大小 # 注意力头维度
        rotary_dim: int,  # 旋转维度 # 旋转编码维度
        max_position_embeddings: int,  # 最大位置编码数 # 最大位置嵌入数
        base: int,  # 旋转基频 # 旋转基频
        is_neox_style: bool,  # 是否为NeoX风格 # 是否NeoX风格
        scaling_factor: float,  # 缩放因子 # 缩放因子
        dtype: torch.dtype,  # 数据类型 # 数据类型
        *,
        mrope_section: Optional[List[int]] = None,  # 多模态RoPE段配置 # 多模态RoPE段配置
        mrope_interleaved: bool = False,  # 是否使用交错模式 # 是否交错模式
        extrapolation_factor: float = 1,  # 外推因子 # 外推因子
        attn_factor: float = 1,  # 注意力因子 # 注意力因子
        beta_fast: int = 32,  # 快速beta参数 # 快速beta参数
        beta_slow: int = 1,  # 慢速beta参数 # 慢速beta参数
        truncate: bool = True,  # 是否截断 # 是否截断
    ) -> None:
        self.scaling_factor = scaling_factor  # 保存缩放因子 # 保存缩放因子
        self.extrapolation_factor = extrapolation_factor  # 保存外推因子 # 保存外推因子
        self.attn_factor = attn_factor  # 保存注意力因子 # 保存注意力因子
        self.beta_fast = beta_fast  # 保存快速beta # 保存快速beta
        self.beta_slow = beta_slow  # 保存慢速beta # 保存慢速beta
        self.truncate = truncate  # 保存截断标志 # 保存截断标志
        self.mscale = float(yarn_get_mscale_simple(self.scaling_factor) * attn_factor)  # 计算缩放乘数 # 计算缩放乘数
        super().__init__(  # 调用父类初始化 # 调用父类初始化
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:  # 计算YaRN缩放后的逆频率 # 计算YaRN缩放逆频率
        pos_freqs = self.base ** (  # 计算位置频率 # 计算位置频率
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs  # 外推逆频率 # 外推逆频率
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)  # 插值逆频率 # 插值逆频率
        low, high = yarn_find_correction_range(  # 查找校正范围 # 查找校正范围
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
            self.truncate,
        )
        inv_freq_mask = (  # 计算逆频率掩码 # 计算逆频率掩码
            1
            - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor
        inv_freq = (  # 混合插值和外推逆频率 # 混合插值和外推
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq  # 返回混合后的逆频率 # 返回混合逆频率

    def _compute_cos_sin_cache(self) -> torch.Tensor:  # 计算YaRN缩放后的cos/sin缓存 # 计算YaRN缩放cos/sin缓存
        inv_freq = self._compute_inv_freq(self.scaling_factor)  # 计算逆频率 # 计算逆频率
        t = torch.arange(  # 生成缩放后的位置序列 # 生成缩放后位置序列
            self.max_position_embeddings * self.scaling_factor, dtype=torch.float32
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 计算外积 # 计算外积
        cos = freqs.cos() * self.mscale  # 计算cos并乘以缩放乘数 # 计算cos乘以mscale
        sin = freqs.sin() * self.mscale  # 计算sin并乘以缩放乘数 # 计算sin乘以mscale
        cache = torch.cat((cos, sin), dim=-1)  # 拼接cos和sin # 拼接cos和sin
        return cache  # 返回缓存 # 返回缓存


class Ernie4_5_VLRotaryEmbedding(MRotaryEmbedding):  # Ernie4.5视觉语言模型的3D旋转位置编码 # Ernie4.5 VL 3D旋转位置编码
    """3D rotary positional embedding. [h w h w h w h w... t t t...]  # 3D旋转位置编码。[h w h w h w h w... t t t...]"""

    def __init__(  # 初始化Ernie4.5 VL旋转位置编码 # 初始化方法
        self,
        head_size: int,  # 注意力头大小 # 注意力头维度
        rotary_dim: int,  # 旋转维度 # 旋转编码维度
        max_position_embeddings: int,  # 最大位置编码数 # 最大位置嵌入数
        base: int,  # 旋转基频 # 旋转基频
        is_neox_style: bool,  # 是否为NeoX风格 # 是否NeoX风格
        dtype: torch.dtype,  # 数据类型 # 数据类型
        mrope_section: Optional[List[int]] = None,  # 多模态RoPE段配置 # 多模态RoPE段配置
        mrope_interleaved: bool = False,  # 是否使用交错模式 # 是否交错模式
    ) -> None:
        super().__init__(  # 调用父类初始化 # 调用父类初始化
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
        )
        self._apply_rotary_emb_wrapped = torch.compile(dynamic=True)(apply_rotary_emb)  # 编译优化旋转编码函数 # 编译优化旋转编码

    def forward_native(  # PyTorch原生实现的Ernie4.5 VL前向方法 # 原生PyTorch前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor = None,  # 键张量 # 键张量
    ):
        assert positions.ndim == 1 or positions.ndim == 2  # 断言位置为1D或2D # 断言位置维度
        assert key is not None  # 断言key不为None # 断言key不为None

        num_tokens = positions.shape[-1]  # token数量 # token数量
        cos_sin = self.cos_sin_cache[positions]  # 按位置索引获取cos/sin # 按位置索引获取
        cos, sin = cos_sin.chunk(2, dim=-1)  # 分离cos和sin # 分离cos和sin
        if positions.ndim == 2:  # 二维位置（多模态） # 二维位置
            assert self.mrope_section  # 断言mrope_section已设置 # 断言段配置已设置
            section_h = self.mrope_section[0]  # 高度段大小 # 高度段大小
            section_w = self.mrope_section[1]  # 宽度段大小 # 宽度段大小
            section_t = self.mrope_section[2]  # 时间段大小 # 时间段大小
            assert section_h == section_w  # 断言高度和宽度段大小相同 # 断言h和w段大小相同
            section_cos_t = cos[..., -section_t:]  # 提取时间段的cos # 提取时间段的cos
            section_cos_h = cos[..., : section_h + section_w : 2]  # 提取高度段的cos（交错） # 提取高度段的cos
            section_cos_w = cos[..., 1 : section_h + section_w : 2]  # 提取宽度段的cos（交错） # 提取宽度段的cos
            cos_t, cos_h, cos_w = section_cos_t[0], section_cos_h[1], section_cos_w[2]  # 分别获取各模态的cos # 分别获取各模态cos
            cos_hw = torch.stack([cos_h, cos_w], dim=-1).reshape(  # 交错拼接h和w的cos # 交错拼接h和w
                cos_h.shape[:-1] + (cos_h.shape[-1] * 2,)
            )
            cos = torch.cat([cos_hw, cos_t], dim=-1)  # 拼接hw和t的cos # 拼接hw和t的cos
            section_sin_t = sin[..., -section_t:]  # 提取时间段的sin # 提取时间段的sin
            section_sin_h = sin[..., : section_h + section_w : 2]  # 提取高度段的sin（交错） # 提取高度段的sin
            section_sin_w = sin[..., 1 : section_h + section_w : 2]  # 提取宽度段的sin（交错） # 提取宽度段的sin
            sin_t, sin_h, sin_w = section_sin_t[0], section_sin_h[1], section_sin_w[2]  # 分别获取各模态的sin # 分别获取各模态sin
            sin_hw = torch.stack([sin_h, sin_w], dim=-1).reshape(  # 交错拼接h和w的sin # 交错拼接h和w
                sin_h.shape[:-1] + (sin_h.shape[-1] * 2,)
            )
            sin = torch.cat([sin_hw, sin_t], dim=-1)  # 拼接hw和t的sin # 拼接hw和t的sin

        query_shape = query.shape  # 保存query原始形状 # 保存原始形状
        query = query.view(num_tokens, -1, self.head_size)  # 重塑query为3D # 重塑query
        query_rot = query[..., : self.rotary_dim]  # 提取旋转部分 # 提取旋转部分
        query_pass = query[..., self.rotary_dim :]  # 提取不旋转部分 # 提取不旋转部分
        query_rot = self._apply_rotary_emb_wrapped(  # 应用旋转编码 # 应用旋转编码
            query_rot, cos, sin, self.is_neox_style
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)  # 拼接并恢复形状 # 拼接恢复形状

        key_shape = key.shape  # 保存key原始形状 # 保存原始形状
        key = key.view(num_tokens, -1, self.head_size)  # 重塑key为3D # 重塑key
        key_rot = key[..., : self.rotary_dim]  # 提取旋转部分 # 提取旋转部分
        key_pass = key[..., self.rotary_dim :]  # 提取不旋转部分 # 提取不旋转部分
        key_rot = self._apply_rotary_emb_wrapped(key_rot, cos, sin, self.is_neox_style)  # 应用旋转编码 # 应用旋转编码
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)  # 拼接并恢复形状 # 拼接恢复形状
        return query, key  # 返回旋转后的query和key # 返回结果

    def forward_cuda(  # CUDA平台的Ernie4.5 VL前向方法 # CUDA平台前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor = None,  # 键张量 # 键张量
    ):
        assert key is not None  # 断言key不为None # 断言key不为None
        assert positions.ndim in (1, 2)  # 断言位置为1D或2D # 断言位置维度
        self._match_cos_sin_cache_dtype(query)  # 匹配缓存数据类型 # 匹配缓存数据类型

        if positions.ndim == 2:  # 二维位置（多模态） # 二维位置
            assert self.mrope_section is not None  # 断言mrope_section已设置 # 断言段配置已设置
            triton_ernie45_rope_fused_inplace(  # 调用Ernie4.5专用Triton融合RoPE # 调用Ernie4.5 Triton融合RoPE
                q=query,
                k=key,
                cos_sin_cache=self.cos_sin_cache,
                positions=positions,
                mrope_section=self.mrope_section,
                head_size=self.head_size,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
            )
            return query, key  # 返回旋转后的query和key # 返回结果

        if _is_cuda and (apply_rope_with_cos_sin_cache_inplace is not None):  # CUDA平台且有一维RoPE核函数 # CUDA平台且有一维RoPE核函数
            apply_rope_with_cos_sin_cache_inplace(  # 调用一维RoPE核函数 # 调用一维RoPE核函数
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_size,
                cos_sin_cache=self.cos_sin_cache,
                is_neox=self.is_neox_style,
            )
            return query, key  # 返回旋转后的query和key # 返回结果

        return self.forward_native(positions, query, key)  # 回退到原生方法 # 回退到原生方法

    def forward(  # Ernie4.5 VL的前向方法总入口 # 前向方法总入口
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor,  # 键张量 # 键张量
        fused_set_kv_buffer_arg=None,  # 融合KV缓冲区参数 # 融合KV缓冲区参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert positions.ndim == 1 or positions.ndim == 2  # 断言位置为1D或2D # 断言位置维度
        return self.forward_cuda(positions, query, key)  # 委托给CUDA前向方法 # 委托给CUDA前向方法
