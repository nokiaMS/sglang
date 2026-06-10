# 文件说明：ROCm平台MLA解码注意力与RoPE融合的Triton内核实现
# 本文件实现了在AMD ROCm平台上，带旋转位置编码(RoPE)融合的MLA解码注意力机制
# 基于lightllm的gqa_flash_decoding_stage1/stage2改编，支持DeepSeek2的多头潜在注意力(MLA)

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding. # 解码阶段的内存高效注意力
It supports page size = 1. # 支持page size = 1
"""

# Adapted from # 改编自
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import triton  # 导入Triton库 # GPU内核编写框架
import triton.language as tl  # 导入Triton语言并别名为tl # Triton编程语言

from sglang.srt.layers.attention.triton_ops.decode_attention import (  # 导入解码注意力的softmax+reduceV函数 # 导入stage2内核
    _decode_softmax_reducev_fwd,
)


def is_hip():  # 检测当前是否为HIP(AMD)平台 # 判断是否运行在AMD ROCm平台
    return triton.runtime.driver.active.get_current_target().backend == "hip"  # 检查后端是否为hip # 比较后端类型


_is_hip = is_hip()  # 缓存HIP平台检测结果 # HIP平台标志


@triton.jit  # Triton JIT编译装饰器 # 将函数编译为GPU内核
def tanh(x):  # 双曲正切函数的Triton实现 # 使用缩放的sigmoid实现tanh
    # Tanh is just a scaled sigmoid # Tanh只是缩放后的sigmoid # 基于sigmoid实现tanh
    return 2 * tl.sigmoid(2 * x) - 1  # 2*sigmoid(2x)-1 = tanh(x) # 缩放sigmoid公式


@triton.jit  # Triton JIT编译装饰器 # 将函数编译为GPU内核
def _fwd_grouped_kernel_stage1_rope(  # 带RoPE融合的分组注意力stage1内核 # 解码注意力第一阶段：分块计算QK和AV
    Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r) # 查询张量，包含非旋转和旋转部分
    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r) # 键缓冲区，包含潜在表示和位置编码
    V_buffer,  # Holds [KV], b*s x (c) # 值缓冲区，包含潜在表示
    cos_sin_cache,  # max_seq_len x (rotary_dim * 2) # 余弦正弦缓存，用于RoPE计算
    positions,  # sequence positions # 序列位置索引
    sm_scale,  # softmax缩放因子 # softmax缩放系数
    kv_indptr,  # KV索引指针数组 # 每个batch的KV起始位置
    kv_indices,  # KV索引数组 # KV缓存中的实际位置索引
    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank + 1) # 注意力中间输出
    k_pe_t_out,  # k_pe转置输出 # 应用RoPE后的k_pe输出
    stride_qb,  # Q的batch步长 # Q张量在batch维度的步长
    stride_qh,  # Q的head步长 # Q张量在head维度的步长
    stride_buf_kbs,  # K缓冲区的batch步长 # K缓冲区在batch维度的步长
    stride_buf_vbs,  # V缓冲区的batch步长 # V缓冲区在batch维度的步长
    stride_mid_ob,  # 中间输出的batch步长 # 中间输出在batch维度的步长
    stride_mid_oh,  # 中间输出的head步长 # 中间输出在head维度的步长
    stride_mid_os,  # 中间输出的split步长 # 中间输出在split维度的步长
    stride_kpe_tokens_out_b,  # k_pe输出的batch步长 # k_pe输出在batch维度的步长
    stride_cos_sin_cache_s,  # cos_sin缓存的序列步长 # cos_sin缓存序列维度的步长
    stride_positions_b,  # positions的batch步长 # 位置索引在batch维度的步长
    rotary_dim: tl.constexpr,  # 旋转维度大小 # RoPE应用的维度
    kv_lora_rank: tl.constexpr,  # KV的LoRA秩 # MLA中KV压缩的维度
    qk_rope_head_dim: tl.constexpr,  # QK旋转部分的头维度 # 旋转位置编码的维度
    kv_group_num: tl.constexpr,  # KV分组数 # GQA中的组数
    q_head_num: tl.constexpr,  # 查询头总数 # 查询头的数量
    BLOCK_C: tl.constexpr,  # C维度（kv_lora_rank）的块大小 # 压缩维度的分块大小
    BLOCK_R: tl.constexpr,  # R维度（rope维度）的块大小 # 旋转维度的分块大小
    BLOCK_N: tl.constexpr,  # N维度（KV序列）的块大小 # KV序列的分块大小
    BLOCK_H: tl.constexpr,  # H维度（头）的块大小 # 头维度的分块大小
    NUM_KV_SPLITS: tl.constexpr,  # KV分割数 # KV序列的分割数量
    logit_cap: tl.constexpr,  # logit上限值 # 注意力logit的截断值
    USE_ROPE: tl.constexpr,  # 是否使用RoPE # 旋转位置编码标志
    IS_NEOX_STYLE: tl.constexpr,  # 是否使用Neox风格的RoPE # RoPE风格标志
):

    cur_batch = tl.program_id(0)  # 获取当前batch索引 # 当前batch ID
    cur_head_id = tl.program_id(1)  # 获取当前头块索引 # 当前头块ID
    split_kv_id = tl.program_id(2)  # 获取当前KV分割索引 # 当前KV分割ID

    if BLOCK_H < kv_group_num:  # 如果块大小小于KV分组数 # 判断头块是否能容纳所有KV组
        VALID_BLOCK_H: tl.constexpr = BLOCK_H  # 使用BLOCK_H作为有效头数 # 受限于块大小
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num  # 使用KV分组数作为有效头数 # 受限于分组数
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)  # 计算当前处理的所有头索引 # 头索引范围
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H  # 头有效性掩码（前半部分） # 头范围下界掩码
    mask_h = mask_h & (cur_head < q_head_num)  # 头有效性掩码（不超过总头数） # 完整的头有效性掩码

    offs_c = tl.arange(0, BLOCK_C)  # C维度（kv_lora_rank）的索引范围 # 压缩维度索引
    offs_qk_r = tl.arange(kv_lora_rank, kv_lora_rank + BLOCK_R)  # to get the k_pe # R维度（rope部分）的索引范围 # 旋转维度索引

    off_q_pe = (  # Q的旋转部分偏移 # Q_PE的内存地址偏移
        cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_qk_r[None, :]  # batch+head+rope维度偏移 # 各维度偏移组合
    )
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]  # Q的非旋转部分偏移 # Q_NOPE的内存地址偏移

    mask_c = offs_c < kv_lora_rank  # C维度有效性掩码 # 压缩维度有效性掩码
    mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)  # R维度有效性掩码 # 旋转维度有效性掩码

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)  # 加载当前batch的KV起始索引 # 读取KV起始位置
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx  # 加载当前batch的序列长度 # 读取序列长度

    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_c[None, :]), other=0.0)  # 加载Q的非旋转部分 # 读取Q_NOPE
    q_pe = tl.load(  # 加载Q的旋转部分 # 读取Q_PE
        Q + off_q_pe, mask=(mask_h[:, None]) & (mask_qk_r[None, :]), other=0.0  # 应用头和维度掩码 # 应用有效性掩码
    )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)  # 每个KV分割的长度 # 每个分割处理的KV数量
    split_kv_start = kv_len_per_split * split_kv_id  # 当前分割的起始位置 # 当前分割的KV起始索引
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)  # 当前分割的结束位置 # 当前分割的KV结束索引

    # apply rotary embedding for q_pe, and k_pe (last token per batch of K_PE) # 对q_pe和k_pe应用旋转位置编码（每个batch K_PE的最后一个token）
    LAST_SPLIT = split_kv_end == cur_batch_seq_len  # 判断是否为最后一个分割 # 最后一个分割标志
    k_pe_last_token = tl.zeros([BLOCK_R], dtype=q.dtype)  # 初始化最后一个token的k_pe # 用于存储RoPE后的k_pe

    if USE_ROPE:  # 如果使用RoPE # RoPE处理
        if IS_NEOX_STYLE:  # Neox风格的RoPE # GPT-NeoX风格旋转
            # [BLOCK_ROTARY // 2, BLOCK_ROTARY // 2 + 1, BLOCK_ROTARY // 2 + 2, ..., 0, 1, 2, ..., BLOCK_ROTARY // 2 - 1, pass:]
            # 旋转维度的重排索引 # Neox风格的索引重排
            offs_qk_rot_r = kv_lora_rank + (  # 计算旋转后的索引偏移 # 旋转索引
                (tl.arange(0, BLOCK_R) + (rotary_dim // 2)) % rotary_dim  # 取模实现循环移位 # 循环移位索引
            )
            # Which elements to flip # 哪些元素需要取反 # 需要取反的元素掩码
            mask_rotate = tl.arange(0, BLOCK_R) < (rotary_dim // 2)  # 前半部分需要取反 # 前半部分掩码
            # [0 , 1, 2, ..., rotary_dim // 2 - 1, 0 , 1, 2, ..., rotary_dim // 2 - 1]
            offs_rotary = tl.arange(0, BLOCK_R) % (rotary_dim // 2)  # cos/sin表的索引 # cos/sin查找索引
        else:  # 非Neox风格的RoPE # 交替式旋转（GPT-J风格）
            # [1, 0, 3, 2, 5, 4, ..., BLOCK_R, BLOCK_R - 1]
            offs_qk_rot_r = (  # 计算旋转后的索引偏移 # 旋转索引
                kv_lora_rank  # 加上kv_lora_rank偏移 # 基础偏移
                + (((tl.arange(0, BLOCK_R) + 1) % 2) * 2)  # 交替选择 # 偶数/奇数交替
                - 1  # 偏移调整 # 索引调整
                + tl.arange(0, BLOCK_R)  # 加上原始索引 # 原始索引
            )
            mask_rotate = tl.arange(0, BLOCK_R) % 2 < 1  # 偶数位置需要取反 # 偶数位置掩码
            # [0, 0, 1, 1, ..., rotary_dim // 2 - 1, rotary_dim // 2 - 1]
            offs_rotary = tl.arange(0, BLOCK_R) // 2  # cos/sin表的索引 # cos/sin查找索引

        if qk_rope_head_dim > rotary_dim:  # 如果rope头维度大于旋转维度 # 维度不匹配处理
            offs_qk_rot_r = tl.where(  # 超出旋转维度的部分保持原索引 # 限制旋转索引范围
                tl.arange(0, BLOCK_R) < rotary_dim, offs_qk_rot_r, tl.arange(0, BLOCK_R)  # 在旋转维度内用旋转索引，否则用原始索引 # 条件选择
            )
            offs_rotary = tl.where(  # 超出旋转维度的部分保持原索引 # 限制cos/sin索引范围
                tl.arange(0, BLOCK_R) < rotary_dim, offs_rotary, tl.arange(0, BLOCK_R)  # 在旋转维度内用旋转索引，否则用原始索引 # 条件选择
            )

        mask_rotary = tl.arange(0, BLOCK_R) < rotary_dim  # 旋转维度有效性掩码 # RoPE有效维度掩码

        pos = tl.load(positions + cur_batch * stride_positions_b)  # 加载当前位置索引 # 读取序列位置
        cos = tl.load(  # 加载余弦值 # 读取cos值
            cos_sin_cache + pos * stride_cos_sin_cache_s + offs_rotary,  # 计算cos的地址偏移 # cos地址偏移
            mask=mask_rotary,  # 应用旋转维度掩码 # 有效性掩码
            other=1.0,  # 超出范围填充1.0 # 无效位置填充1
        )
        sin = tl.load(  # 加载正弦值 # 读取sin值
            cos_sin_cache  # cos_sin缓存基地址 # 缓存基地址
            + pos * stride_cos_sin_cache_s  # 序列位置偏移 # 位置偏移
            + offs_rotary  # 旋转维度偏移 # 旋转索引偏移
            + rotary_dim // 2,  # sin在缓存中的偏移（后半部分） # sin在缓存中的起始偏移
            mask_rotary,  # 应用旋转维度掩码 # 有效性掩码
            other=0.0,  # 超出范围填充0.0 # 无效位置填充0
        )

        off_q_pe_rot = (  # 旋转后的Q_PE偏移 # 旋转后的Q_PE地址偏移
            cur_batch * stride_qb  # batch偏移 # batch步长偏移
            + cur_head[:, None] * stride_qh  # head偏移 # head步长偏移
            + offs_qk_rot_r[None, :]  # 旋转索引偏移 # 旋转后的维度偏移
        )
        mask_qk_rot_r = offs_qk_rot_r < (kv_lora_rank + qk_rope_head_dim)  # 旋转索引有效性掩码 # 旋转索引范围掩码

        # 0, 2, 4,.... 1, 3, 5... # 偶数位和奇数位的元素 # 交替排列的索引
        q_pe_rot = tl.load(  # 加载旋转配对的Q_PE元素 # 读取旋转配对的元素
            Q + off_q_pe_rot,  # 旋转后的Q_PE地址 # 旋转后的地址
            mask=(mask_h[:, None]) & (mask_qk_rot_r[None, :]),  # 应用头和维度掩码 # 有效性掩码
            other=0.0,  # 无效位置填充0 # 填充值
        )
        q_pe_rot = tl.where(mask_rotate[None, :], -q_pe_rot, q_pe_rot)  # 对需要取反的元素取反 # 旋转：部分元素取反

        q_pe = q_pe * cos + q_pe_rot * sin  # 应用RoPE：x*cos + x_rot*sin # RoPE旋转公式

        # we only apply to the last token in the K_PE # 仅对K_PE中的最后一个token应用RoPE # 只对最新token的k_pe应用RoPE
        if LAST_SPLIT:  # 如果是最后一个分割 # 仅最后一个分割处理k_pe
            # debug assert # 调试断言 # 开发调试用
            if (cur_batch == 0 and cur_head == 0) and split_kv_id < NUM_KV_SPLITS - 1:  # 非最后一个分割不应计算k_pe # 验证只有最后一个分割计算k_pe
                tl.device_assert(False, "Only last split should compute k_pe")  # 触发设备断言 # 断言失败

            kv_loc = tl.load(  # 加载最后一个token的KV位置 # 读取最后token的缓存位置
                kv_indices + cur_batch_kv_start_idx + cur_batch_seq_len - 1  # 最后一个token的索引 # 序列末尾位置
            )
            offs_buf_k_pe_last_token = kv_loc * stride_buf_kbs + offs_qk_r  # 最后一个token的k_pe偏移 # k_pe原始地址偏移
            offs_buf_k_pe_rot_last_token = kv_loc * stride_buf_kbs + offs_qk_rot_r  # 最后一个token的旋转k_pe偏移 # k_pe旋转地址偏移
            k_pe_last_token = tl.load(K_Buffer + offs_buf_k_pe_last_token)  # 加载最后一个token的k_pe # 读取k_pe

            k_pe_rot_last_token = tl.load(K_Buffer + offs_buf_k_pe_rot_last_token)  # 加载旋转配对的k_pe # 读取旋转配对元素
            k_pe_rot_last_token = tl.where(  # 对需要取反的元素取反 # 旋转：部分元素取反
                mask_rotate, -k_pe_rot_last_token, k_pe_rot_last_token  # 条件取反 # 取反或保持
            )

            k_pe_last_token = k_pe_last_token * cos + k_pe_rot_last_token * sin  # 应用RoPE # RoPE旋转公式

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")  # 初始化行最大值为负无穷 # 在线softmax最大值
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)  # 初始化行求和为0 # 在线softmax归一化因子
    acc = tl.zeros([BLOCK_H, BLOCK_C], dtype=tl.float32)  # 初始化累加器为0 # 注意力加权累加器

    if split_kv_end > split_kv_start:  # 如果当前分割非空 # 检查分割是否有数据
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):  # 遍历KV序列块 # 按块遍历KV序列
            offs_n = start_n + tl.arange(0, BLOCK_N)  # 当前块的KV序列索引 # 当前KV块索引
            kv_loc = tl.load(  # 加载KV位置索引 # 读取KV缓存位置
                kv_indices + cur_batch_kv_start_idx + offs_n,  # KV索引地址偏移 # 索引地址
                mask=offs_n < split_kv_end,  # 有效性掩码 # 遮蔽超出分割范围的索引
                other=0,  # 无效位置填充0 # 填充值
            )

            offs_buf_kv = kv_loc[None, :] * stride_buf_kbs + offs_c[:, None]  # KV压缩部分的偏移 # KV潜在表示地址偏移
            offs_buf_k_pe = kv_loc[None, :] * stride_buf_kbs + offs_qk_r[:, None]  # K_PE部分的偏移 # K_PE地址偏移

            k_pe = tl.load(  # 加载k_pe数据 # 读取键的位置编码
                K_Buffer + offs_buf_k_pe,  # K_PE地址 # K_PE数据地址
                mask=(offs_n[None, :] < split_kv_end) & (mask_qk_r[:, None]),  # 有效性掩码 # 有效性检查
                other=0.0,  # 无效位置填充0 # 填充值
            )  # positional embedding part of keys # 键的位置编码部分 # 键的旋转位置编码

            if (USE_ROPE and LAST_SPLIT) and start_n >= cur_batch_seq_len - BLOCK_N:  # 如果使用RoPE且是最后一块 # 对最后一个token的k_pe应用RoPE
                k_pe = tl.where(  # 替换最后一个token的k_pe为RoPE后的值 # 条件替换
                    offs_n[None, :] != (split_kv_end - 1),  # 不是最后一个token # 判断是否为最后一个token
                    k_pe,  # 保持原值 # 保持原值
                    k_pe_last_token[:, None],  # 使用RoPE后的值 # 使用RoPE处理后的值
                )

            # (16, 64) x (64, 32) # 矩阵乘法维度 # 矩阵乘法形状
            # dot product of rope parts # 旋转部分的点积 # Q_PE和K_PE的点积
            qk = tl.dot(q_pe, k_pe.to(q_pe.dtype))  # 计算旋转部分的QK点积 # Q_PE @ K_PE

            kv = tl.load(  # 加载KV压缩部分 # 读取KV潜在表示
                K_Buffer + offs_buf_kv,  # KV地址 # KV数据地址
                mask=(offs_n[None, :] < split_kv_end) & (mask_c[:, None]),  # 有效性掩码 # 有效性检查
                other=0.0,  # 无效位置填充0 # 填充值
            )  # the shared latent tensor for keys and values # 键和值的共享潜在张量 # MLA中KV的压缩表示

            # (16, 512) x (512, 32) # 矩阵乘法维度 # 矩阵乘法形状
            # dot product of nope parts # 非旋转部分的点积 # Q_NOPE和KV的点积
            qk += tl.dot(q, kv)  # 累加非旋转部分的QK点积 # Q_NOPE @ KV

            qk *= sm_scale  # 应用softmax缩放 # 缩放QK点积

            if logit_cap > 0:  # 如果设置了logit上限 # logit截断处理
                qk = logit_cap * tanh(qk / logit_cap)  # 使用tanh截断logit # 截断公式

            qk = tl.where(  # 应用有效性掩码 # 遮蔽无效位置
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")  # 有效位置保留，无效设为负无穷 # 条件掩码
            )

            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + offs_c[None, :]  # V的偏移地址 # V数据地址偏移
            v = tl.load(  # 加载V数据 # 读取值数据
                V_buffer + offs_buf_v,  # V地址 # V数据地址
                mask=(offs_n[:, None] < split_kv_end) & (mask_c[None, :]),  # 有效性掩码 # 有效性检查
                other=0.0,  # 无效位置填充0 # 填充值
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)  # 更新行最大值 # 取当前块最大值和全局最大值的较大者
            re_scale = tl.exp(e_max - n_e_max)  # 计算旧累加器的重缩放因子 # 历史值的缩放因子
            p = tl.exp(qk - n_e_max[:, None])  # 计算指数概率 # 减去最大值后取指数
            acc *= re_scale[:, None]  # 重缩放累加器 # 缩放历史累加结果
            # (16, 32) x (32, 512) # 矩阵乘法维度 # 矩阵乘法形状
            acc += tl.dot(p.to(v.dtype), v)  # 累加加权V值 # 概率加权求和

            e_sum = e_sum * re_scale + tl.sum(p, 1)  # 更新行求和 # 更新归一化因子
            e_max = n_e_max  # 更新行最大值 # 更新全局最大值

        offs_mid_o = (  # 中间输出的偏移地址 # 中间输出的内存地址偏移
            cur_batch * stride_mid_ob  # batch偏移 # batch步长偏移
            + cur_head[:, None] * stride_mid_oh  # head偏移 # head步长偏移
            + split_kv_id * stride_mid_os  # split偏移 # split步长偏移
            + offs_c[None, :]  # dim偏移 # 维度偏移
        )

        if USE_ROPE:  # 如果使用RoPE # RoPE输出处理
            if LAST_SPLIT:  # 如果是最后一个分割 # 最后一个分割输出k_pe
                k_pe_last_token_ptrs = (  # k_pe输出的指针 # k_pe输出地址
                    k_pe_t_out  # k_pe输出基地址 # 输出基地址
                    + cur_batch * stride_kpe_tokens_out_b  # batch偏移 # batch步长偏移
                    + tl.arange(0, BLOCK_R)  # 维度偏移 # 维度索引
                )
                tl.store(k_pe_last_token_ptrs, k_pe_last_token, mask=mask_qk_r)  # 存储RoPE后的k_pe # 写入k_pe

        tl.store(  # 存储中间输出（归一化后的注意力值） # 写入中间注意力输出
            Att_Out + offs_mid_o,  # 中间输出地址 # 输出地址
            acc / e_sum[:, None],  # 归一化后的累加结果 # 归一化结果
            mask=(mask_h[:, None]) & (mask_c[None, :]),  # 有效性掩码 # 有效性检查
        )

        offs_mid_o_1 = (  # lse输出的偏移地址 # log-sum-exp输出地址偏移
            cur_batch * stride_mid_ob  # batch偏移 # batch步长偏移
            + cur_head * stride_mid_oh  # head偏移 # head步长偏移
            + split_kv_id * stride_mid_os  # split偏移 # split步长偏移
            + kv_lora_rank  # lse存储在kv_lora_rank位置 # lse在最后一个位置
        )

        tl.store(  # 存储log-sum-exp值 # 写入lse值
            Att_Out + offs_mid_o_1,  # lse输出地址 # lse地址
            e_max + tl.log(e_sum),  # log-sum-exp值 # 计算lse
            mask=mask_h,  # 头有效性掩码 # 头掩码
        )


# TODO rope offset # 待办：RoPE偏移 # 需要添加RoPE偏移支持
def _decode_grouped_att_m_fwd_rope(  # 解码分组注意力前向计算（带RoPE） # 解码阶段MLA注意力的stage1入口函数
    q,  # 查询张量 # Query
    k_buffer,  # 键缓冲区 # Key缓冲区
    v_buffer,  # 值缓冲区 # Value缓冲区
    att_out,  # 注意力中间输出 # 注意力中间结果
    k_pe_tokens_out,  # k_pe输出张量 # RoPE后的k_pe输出
    kv_lora_rank,  # c # KV的LoRA秩（压缩维度）
    cos_sin_cache,  # 余弦正弦缓存 # RoPE的cos/sin缓存
    positions,  # 序列位置索引 # 位置信息
    rotary_dim,  # 旋转维度 # RoPE应用的维度
    kv_indptr,  # KV索引指针 # 每个batch的KV起始位置
    kv_indices,  # KV索引数组 # KV缓存位置索引
    num_kv_splits,  # KV分割数 # KV序列分割数量
    sm_scale,  # softmax缩放因子 # softmax缩放系数
    logit_cap,  # logit上限值 # 注意力logit截断值
    use_rope,  # 是否使用RoPE # RoPE标志
    is_neox_style=True,  # 是否使用Neox风格RoPE # RoPE风格
):
    if use_rope:  # 如果使用RoPE # RoPE启用检查
        assert (  # 断言检查 # 验证输出缓冲区
            k_pe_tokens_out is not None  # k_pe输出必须非空 # 必须提供k_pe输出缓冲区
        ), "We must output the k_pe tokens with rope applied if rope fusion enabled."  # 启用RoPE融合时必须输出k_pe # 错误信息

    BLOCK = 32  # KV序列分块大小 # KV序列块大小

    # # [TODO] work around shmem limit on MI3xx # 待办：解决MI3xx上的共享内存限制 # AMD MI300系列GPU的共享内存限制
    # if _is_hip and kv_lora_rank >= 576: # 如果是HIP平台且kv_lora_rank较大 # 针对大维度的特殊处理
    #     BLOCK = 16 # 使用更小的块大小 # 减小块大小以适应共享内存

    qk_rope_head_dim = k_buffer.shape[-1] - kv_lora_rank  # 计算rope头维度 # K_PE的维度
    batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]  # 获取batch大小和头数 # 批次大小和头数
    kv_group_num = q.shape[1] // k_buffer.shape[1]  # 计算KV分组数 # GQA分组数

    BLOCK_C = triton.next_power_of_2(kv_lora_rank)  # C维度块大小（2的幂） # 压缩维度分块大小
    BLOCK_R = triton.next_power_of_2(qk_rope_head_dim)  # R维度块大小（2的幂） # 旋转维度分块大小

    BLOCK_H = 16  # 头维度块大小 # 头维度分块大小
    NUM_KV_SPLITS = num_kv_splits  # KV分割数 # KV序列分割数量
    grid = (  # 设置内核启动网格 # 内核网格大小
        batch,  # batch维度 # 批次数
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),  # head块维度 # 头分块数
        NUM_KV_SPLITS,  # KV分割维度 # 分割数
    )

    extra_kargs = {}  # 额外内核参数 # 额外内核参数字典
    num_stages = 2  # 流水线阶段数 # 内核流水线级数
    if _is_hip:  # 如果是HIP平台 # AMD平台特殊配置
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}  # HIP特有参数 # AMD ROCm优化参数
        num_stages = 1  # HIP平台使用1个流水线阶段 # 减少流水线级数

    _fwd_grouped_kernel_stage1_rope[grid](  # 启动stage1内核 # 调用带RoPE的stage1内核
        q,  # 查询张量 # Query
        k_buffer,  # 键缓冲区 # Key缓冲区
        v_buffer,  # 值缓冲区 # Value缓冲区
        cos_sin_cache,  # 余弦正弦缓存 # cos/sin缓存
        positions,  # 序列位置索引 # 位置信息
        sm_scale,  # softmax缩放因子 # 缩放系数
        kv_indptr,  # KV索引指针 # KV起始位置
        kv_indices,  # KV索引数组 # KV位置索引
        att_out,  # 注意力中间输出 # 中间结果
        k_pe_tokens_out,  # k_pe输出 # k_pe输出
        q.stride(0),  # Q的batch步长 # Q batch步长
        q.stride(1),  # Q的head步长 # Q head步长
        k_buffer.stride(0),  # K的batch步长 # K batch步长
        v_buffer.stride(0),  # V的batch步长 # V batch步长
        att_out.stride(0),  # 中间输出的batch步长 # 输出batch步长
        att_out.stride(1),  # 中间输出的head步长 # 输出head步长
        att_out.stride(2),  # 中间输出的split步长 # 输出split步长
        k_pe_tokens_out.stride(0) if use_rope else 0,  # k_pe输出的batch步长 # k_pe输出步长
        cos_sin_cache.stride(0) if use_rope else 0,  # cos_sin缓存的序列步长 # cos/sin缓存步长
        positions.stride(0) if use_rope else 0,  # positions的batch步长 # 位置步长
        rotary_dim,  # 旋转维度 # RoPE维度
        kv_lora_rank,  # KV的LoRA秩 # 压缩维度
        qk_rope_head_dim,  # rope头维度 # 旋转维度
        kv_group_num=kv_group_num,  # KV分组数 # GQA分组数
        q_head_num=head_num,  # 查询头数 # 查询头数量
        BLOCK_C=BLOCK_C,  # C维度块大小 # 压缩维度分块
        BLOCK_R=BLOCK_R,  # R维度块大小 # 旋转维度分块
        BLOCK_N=BLOCK,  # N维度块大小 # KV序列分块
        BLOCK_H=BLOCK_H,  # H维度块大小 # 头维度分块
        NUM_KV_SPLITS=NUM_KV_SPLITS,  # KV分割数 # 分割数量
        logit_cap=logit_cap,  # logit上限值 # 截断值
        USE_ROPE=use_rope,  # 是否使用RoPE # RoPE标志
        IS_NEOX_STYLE=is_neox_style,  # 是否Neox风格 # RoPE风格
        num_warps=4,  # warp数量 # 线程束数量
        num_stages=num_stages,  # 流水线阶段数 # 流水线级数
        **extra_kargs,  # 额外参数 # 额外内核参数
    )


def decode_attention_fwd_grouped_rope(  # 解码注意力前向计算（带RoPE）的主入口函数 # MLA解码注意力的完整前向计算
    q,  # 查询张量 # Query
    k_buffer,  # 键缓冲区 # Key缓冲区
    v_buffer,  # 值缓冲区 # Value缓冲区
    o,  # 输出张量 # 注意力输出
    kv_indptr,  # KV索引指针 # 每个batch的KV起始位置
    kv_indices,  # KV索引数组 # KV缓存位置索引
    k_pe_tokens,  # k_pe token张量 # K_PE输出缓冲区
    kv_lora_rank,  # KV的LoRA秩 # 压缩维度
    rotary_dim,  # 旋转维度 # RoPE维度
    cos_sin_cache,  # 余弦正弦缓存 # cos/sin缓存
    positions,  # 序列位置索引 # 位置信息
    attn_logits,  # 注意力logits # 注意力中间结果缓冲区
    num_kv_splits,  # KV分割数 # 分割数量
    sm_scale,  # softmax缩放因子 # 缩放系数
    logit_cap=0.0,  # logit上限值（默认0表示不截断） # 截断值
    use_rope=False,  # 是否使用RoPE（默认否） # RoPE标志
    is_neox_style=False,  # 是否Neox风格RoPE（默认否） # RoPE风格
):
    _decode_grouped_att_m_fwd_rope(  # 调用stage1：分块计算QK和AV # 执行第一阶段
        q,  # 查询张量 # Query
        k_buffer,  # 键缓冲区 # Key缓冲区
        v_buffer,  # 值缓冲区 # Value缓冲区
        attn_logits,  # 注意力中间输出 # 中间结果
        k_pe_tokens,  # k_pe输出 # k_pe输出
        kv_lora_rank,  # KV的LoRA秩 # 压缩维度
        cos_sin_cache,  # 余弦正弦缓存 # cos/sin缓存
        positions,  # 序列位置索引 # 位置信息
        rotary_dim,  # 旋转维度 # RoPE维度
        kv_indptr,  # KV索引指针 # KV起始位置
        kv_indices,  # KV索引数组 # KV位置索引
        num_kv_splits,  # KV分割数 # 分割数量
        sm_scale,  # softmax缩放因子 # 缩放系数
        logit_cap,  # logit上限值 # 截断值
        use_rope,  # 是否使用RoPE # RoPE标志
        is_neox_style,  # 是否Neox风格 # RoPE风格
    )
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, kv_indptr, num_kv_splits)  # 调用stage2：softmax归约和V加权 # 执行第二阶段
