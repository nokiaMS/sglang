# 解码阶段高效内存注意力Triton内核文件
# 本文件实现了解码阶段的内存高效注意力计算，
# 支持页大小=1的分页注意力机制。
# 包含两阶段计算：
# - 阶段1：分块计算QKV注意力分数和中间结果
# - 阶段2：跨块归约softmax并生成最终输出
# 支持MHA/GQA/MQA/MLA等多种注意力模式，
# 以及logit截断、XAI温度调节、注意力汇聚(sink)等特性。

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
Memory-efficient attention for decoding.
It supports page size = 1.

解码阶段的内存高效注意力。
支持页大小 = 1。
"""

# Adapted from
# 改编自
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import logging  # 导入日志模块

import triton  # 导入Triton GPU编程框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.utils import is_hip  # 导入HIP平台检测函数

_is_hip = is_hip()  # 检测当前是否为HIP(AMD ROCm)平台

logger = logging.getLogger(__name__)  # 创建日志记录器


_MIN_BLOCK_KV = 32  # KV维度最小块大小


@triton.jit  # Triton JIT编译装饰器
def tanh(x):  # Triton实现的tanh函数
    # Tanh is just a scaled sigmoid
    # Tanh只是缩放的sigmoid
    return 2 * tl.sigmoid(2 * x) - 1  # tanh(x) = 2*sigmoid(2x) - 1


@triton.jit  # Triton JIT编译装饰器
def _fwd_kernel_stage1(  # 解码注意力阶段1内核（MHA模式）
    Q,  # Query张量指针
    K_Buffer,  # Key缓存指针
    V_Buffer,  # Value缓存指针
    sm_scale_withk,  # 包含K缩放的softmax缩放因子
    kv_indptr,  # KV索引偏移指针
    kv_indices,  # KV索引指针
    Att_Out,  # 注意力中间输出指针
    Att_Lse,  # 注意力对数求和指数指针
    num_kv_splits,  # KV分块数指针
    stride_qbs,  # Query批次步长
    stride_qh,  # Query头步长
    stride_buf_kbs,  # Key缓存批次步长
    stride_buf_kh,  # Key缓存头步长
    stride_buf_vbs,  # Value缓存批次步长
    stride_buf_vh,  # Value缓存头步长
    stride_mid_ob,  # 中间输出批次步长
    stride_mid_oh,  # 中间输出头步长
    stride_mid_os,  # 中间输出分块步长
    kv_group_num: tl.constexpr,  # KV组数（编译时常量，用于GQA）
    BLOCK_DMODEL: tl.constexpr,  # 模型维度块大小（编译时常量）
    BLOCK_DV: tl.constexpr,  # Value维度块大小（编译时常量）
    BLOCK_N: tl.constexpr,  # N维度块大小（编译时常量）
    MIN_BLOCK_KV: tl.constexpr,  # KV最小块大小（编译时常量）
    logit_cap: tl.constexpr,  # logit截断值（编译时常量）
    Lk: tl.constexpr,  # Key头维度（编译时常量）
    Lv: tl.constexpr,  # Value头维度（编译时常量）
    xai_temperature_len: tl.constexpr,  # XAI温度长度（编译时常量）
):  # 解码注意力阶段1：分块计算QKV注意力分数
    """解码注意力阶段1内核，分块计算QKV注意力并存储中间结果"""
    cur_batch = tl.program_id(0)  # 获取当前批次索引
    cur_head = tl.program_id(1)  # 获取当前头索引
    split_kv_id = tl.program_id(2)  # 获取当前KV分块索引

    cur_kv_head = cur_head // kv_group_num  # 计算当前KV头索引（GQA共享）

    offs_d = tl.arange(0, BLOCK_DMODEL)  # 模型维度偏移
    offs_dv = tl.arange(0, BLOCK_DV)  # Value维度偏移
    mask_d = offs_d < Lk  # Key维度掩码
    mask_dv = offs_dv < Lv  # Value维度掩码

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)  # 加载当前批次KV起始索引
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx  # 计算当前序列KV长度
    kv_splits = tl.load(num_kv_splits + cur_batch)  # 加载当前批次的KV分块数

    if xai_temperature_len > 0:  # 如果启用XAI温度调节
        offs_qidx = cur_batch_seq_len - 1  # 当前query的位置索引
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))  # 温度缩放系数
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale  # 计算温度值
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)  # 超过温度长度时应用缩放

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d  # 计算Query偏移

    kv_len_per_split = (  # 计算每个KV分块的长度
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV  # 对齐到MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id  # 当前分块起始位置
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)  # 当前分块结束位置

    e_max = -float("inf")  # 初始化最大指数值为负无穷
    e_sum = 0.0  # 初始化指数求和为0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)  # 初始化Value累加器

    if split_kv_end > split_kv_start:  # 如果当前分块非空
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)  # 加载Query向量
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):  # 沿KV维度分块迭代
            offs_n = start_n + tl.arange(0, BLOCK_N)  # 计算N维度偏移
            kv_loc = tl.load(  # 加载KV位置索引
                kv_indices + cur_batch_kv_start_idx + offs_n,  # 从索引数组获取
                mask=offs_n < split_kv_end,  # 掩码
                other=0,  # 越界填充值
            )
            offs_buf_k = (  # 计算Key缓存偏移
                kv_loc[:, None] * stride_buf_kbs  # token位置偏移
                + cur_kv_head * stride_buf_kh  # 头偏移
                + offs_d[None, :]  # 维度偏移
            )
            k = tl.load(  # 加载Key向量
                K_Buffer + offs_buf_k,  # Key缓存地址
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),  # 掩码
                other=0.0,  # 越界填充值
            )
            qk = tl.sum(q[None, :] * k, 1)  # 计算QK点积
            qk *= sm_scale_withk  # 乘以softmax缩放因子

            if logit_cap > 0:  # 如果启用logit截断
                qk = logit_cap * tanh(qk / logit_cap)  # 应用logit截断

            if xai_temperature_len > 0:  # 如果启用XAI温度调节
                qk *= xai_temperature_reg  # 乘以温度调节系数

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))  # 越界位置设为负无穷

            offs_buf_v = (  # 计算Value缓存偏移
                kv_loc[:, None] * stride_buf_vbs  # token位置偏移
                + cur_kv_head * stride_buf_vh  # 头偏移
                + offs_dv[None, :]  # 维度偏移
            )
            v = tl.load(  # 加载Value向量
                V_Buffer + offs_buf_v,  # Value缓存地址
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),  # 掩码
                other=0.0,  # 越界填充值
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)  # 更新最大指数值
            re_scale = tl.exp(e_max - n_e_max)  # 计算重缩放因子
            p = tl.exp(qk - n_e_max)  # 计算softmax概率
            acc *= re_scale  # 重缩放累加器
            acc += tl.sum(p[:, None] * v, 0)  # 累加加权Value

            e_sum = e_sum * re_scale + tl.sum(p, 0)  # 更新指数求和
            e_max = n_e_max  # 更新最大指数值

        offs_mid_o = (  # 计算中间输出偏移
            cur_batch * stride_mid_ob  # 批次偏移
            + cur_head * stride_mid_oh  # 头偏移
            + split_kv_id * stride_mid_os  # 分块偏移
            + offs_dv  # 维度偏移
        )

        tl.store(  # 存储中间输出（归一化后的注意力值）
            Att_Out + offs_mid_o,  # 中间输出地址
            acc / e_sum,  # 归一化结果
            mask=(mask_dv),  # 维度掩码
        )

        offs_mid_o_1 = (  # 计算LSE存储偏移
            cur_batch * stride_mid_ob  # 批次偏移
            + cur_head * stride_mid_oh  # 头偏移
            + split_kv_id * stride_mid_os  # 分块偏移
        ) // Lv  # 除以Value维度

        tl.store(  # 存储对数求和指数
            Att_Lse + offs_mid_o_1,  # LSE地址
            e_max + tl.log(e_sum),  # log(sum(exp)) = e_max + log(e_sum)
        )


def _decode_att_m_fwd(  # 解码注意力MHA前向传播（阶段1）
    q,  # Query张量
    k_buffer,  # Key缓存
    v_buffer,  # Value缓存
    att_out,  # 注意力中间输出
    att_lse,  # 注意力LSE
    kv_indptr,  # KV索引偏移
    kv_indices,  # KV索引
    num_kv_splits,  # KV分块数
    max_kv_splits,  # 最大KV分块数
    sm_scale_withk,  # 包含K缩放的softmax缩放因子
    logit_cap,  # logit截断值
    xai_temperature_len=-1,  # XAI温度长度（默认-1，不启用）
):  # 解码注意力MHA模式阶段1：分块计算QKV注意力
    """解码注意力MHA模式阶段1前向传播，启动阶段1内核计算"""
    BLOCK = 64  # 默认块大小
    # [TODO] work around SGPR limit on MI3xx
    # [待办] 解决MI3xx上的SGPR限制
    if _is_hip:  # HIP平台
        BLOCK = 8  # 减小块大小以规避SGPR限制
    MAX_KV_SPLITS = max_kv_splits  # 最大KV分块数
    Lk = k_buffer.shape[-1]  # Key头维度
    Lv = v_buffer.shape[-1]  # Value头维度

    batch, head_num = q.shape[0], q.shape[1]  # 获取批次大小和头数

    grid = (batch, head_num, MAX_KV_SPLITS)  # 定义3D网格
    kv_group_num = q.shape[1] // k_buffer.shape[1]  # 计算KV组数（GQA）

    if kv_group_num == 1:  # MHA模式
        num_warps = 4  # 使用4个warp
    else:  # GQA/MQA模式
        num_warps = 2  # 使用2个warp
        if _is_hip:  # HIP平台
            num_warps = 1  # 使用1个warp

    BLOCK_DMODEL = triton.next_power_of_2(Lk)  # Key维度对齐到2的幂
    BLOCK_DV = triton.next_power_of_2(Lv)  # Value维度对齐到2的幂

    _fwd_kernel_stage1[grid](  # 启动阶段1内核
        q,  # Query张量
        k_buffer,  # Key缓存
        v_buffer,  # Value缓存
        sm_scale_withk,  # softmax缩放因子
        kv_indptr,  # KV索引偏移
        kv_indices,  # KV索引
        att_out,  # 中间输出
        att_lse,  # LSE
        num_kv_splits,  # KV分块数
        q.stride(0),  # Query批次步长
        q.stride(1),  # Query头步长
        k_buffer.stride(0),  # Key缓存批次步长
        k_buffer.stride(1),  # Key缓存头步长
        v_buffer.stride(0),  # Value缓存批次步长
        v_buffer.stride(1),  # Value缓存头步长
        att_out.stride(0),  # 中间输出批次步长
        att_out.stride(1),  # 中间输出头步长
        att_out.stride(2),  # 中间输出分块步长
        kv_group_num=kv_group_num,  # KV组数
        BLOCK_DMODEL=BLOCK_DMODEL,  # Key维度块大小
        BLOCK_DV=BLOCK_DV,  # Value维度块大小
        BLOCK_N=BLOCK,  # N维度块大小
        MIN_BLOCK_KV=_MIN_BLOCK_KV,  # KV最小块大小
        logit_cap=logit_cap,  # logit截断值
        xai_temperature_len=xai_temperature_len,  # XAI温度长度
        num_warps=num_warps,  # warp数量
        num_stages=2,  # 流水线阶段数
        Lk=Lk,  # Key头维度
        Lv=Lv,  # Value头维度
    )


@triton.jit  # Triton JIT编译装饰器
def _fwd_grouped_kernel_stage1(  # 解码注意力阶段1分组内核（GQA/MQA/MLA模式）
    Q,  # Query张量指针
    K_Buffer,  # Key缓存指针
    V_Buffer,  # Value缓存指针
    sm_scale_withk,  # 包含K缩放的softmax缩放因子
    kv_indptr,  # KV索引偏移指针
    kv_indices,  # KV索引指针
    Att_Out,  # 注意力中间输出指针
    Att_Lse,  # 注意力对数求和指数指针
    num_kv_splits,  # KV分块数指针
    stride_qbs,  # Query批次步长
    stride_qh,  # Query头步长
    stride_buf_kbs,  # Key缓存批次步长
    stride_buf_kh,  # Key缓存头步长
    stride_buf_vbs,  # Value缓存批次步长
    stride_buf_vh,  # Value缓存头步长
    stride_mid_ob,  # 中间输出批次步长
    stride_mid_oh,  # 中间输出头步长
    stride_mid_os,  # 中间输出分块步长
    kv_group_num: tl.constexpr,  # KV组数（编译时常量）
    q_head_num: tl.constexpr,  # Query头数（编译时常量）
    BLOCK_DMODEL: tl.constexpr,  # 模型维度块大小（编译时常量）
    BLOCK_DPE: tl.constexpr,  # 位置编码维度块大小（编译时常量，MLA用）
    BLOCK_DV: tl.constexpr,  # Value维度块大小（编译时常量）
    BLOCK_N: tl.constexpr,  # N维度块大小（编译时常量）
    BLOCK_H: tl.constexpr,  # 头维度块大小（编译时常量）
    MIN_BLOCK_KV: tl.constexpr,  # KV最小块大小（编译时常量）
    logit_cap: tl.constexpr,  # logit截断值（编译时常量）
    xai_temperature_len: tl.constexpr,  # XAI温度长度（编译时常量）
    Lk: tl.constexpr,  # Key头维度（编译时常量）
    Lv: tl.constexpr,  # Value头维度（编译时常量）
    HAS_MLA: tl.constexpr = False,  # 是否为MLA模式（编译时常量，默认False）
    USE_PDL: tl.constexpr = False,  # 是否使用PDL（编译时常量，默认False）
):  # 解码注意力阶段1分组模式：支持多Query头并行处理
    """解码注意力阶段1分组内核，支持GQA/MQA/MLA模式，多Query头并行"""
    cur_batch = tl.program_id(0)  # 获取当前批次索引
    cur_head_id = tl.program_id(1)  # 获取当前头块索引
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)  # 计算当前KV头索引
    split_kv_id = tl.program_id(2)  # 获取当前KV分块索引

    if BLOCK_H < kv_group_num:  # 如果块头数小于KV组数
        VALID_BLOCK_H: tl.constexpr = BLOCK_H  # 有效块头数为BLOCK_H
    else:  # 块头数不小于KV组数
        VALID_BLOCK_H: tl.constexpr = kv_group_num  # 有效块头数为KV组数
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)  # 计算当前处理的所有头索引
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H  # 头索引掩码（块内有效头）
    mask_h = mask_h & (cur_head < q_head_num)  # 头索引掩码（不超过总头数）

    offs_d = tl.arange(0, BLOCK_DMODEL)  # 模型维度偏移
    offs_dv = tl.arange(0, BLOCK_DV)  # Value维度偏移
    mask_d = offs_d < Lk  # Key维度掩码
    mask_dv = offs_dv < Lv  # Value维度掩码

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)  # 加载当前批次KV起始索引
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx  # 计算序列KV长度
    kv_splits = tl.load(num_kv_splits + cur_batch)  # 加载KV分块数

    if xai_temperature_len > 0:  # 如果启用XAI温度调节
        offs_qidx = cur_batch_seq_len - 1  # 当前query位置索引
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))  # 温度缩放系数
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale  # 计算温度值
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)  # 超过温度长度时应用缩放

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]  # 计算Query偏移

    if BLOCK_DPE > 0:  # 如果有位置编码维度（MLA模式）
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)  # 位置编码维度偏移
        mask_dpe = offs_dpe < Lk  # 位置编码维度掩码
        off_qpe = (  # 计算Query位置编码偏移
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]  # 批次+头+位置编码维度
        )

    kv_len_per_split = (  # 计算每个KV分块的长度
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV  # 对齐到MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id  # 当前分块起始位置
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)  # 当前分块结束位置

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")  # 初始化最大指数值（每头一个）
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)  # 初始化指数求和（每头一个）
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)  # 初始化Value累加器（多头并行）

    # Hoist loop-invariant base offsets
    # 提升循环不变的基础偏移
    base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]  # Key缓存基础偏移（不含token位置）
    if BLOCK_DPE > 0:  # MLA模式
        base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]  # Key位置编码基础偏移
    if not HAS_MLA:  # 非MLA模式
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]  # Value缓存基础偏移

    if split_kv_end > split_kv_start:  # 如果当前分块非空
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)  # 加载Query向量（多头）
        q_k = q.to(K_Buffer.dtype.element_ty)  # 转换为Key的数据类型（用于dot乘法）
        if BLOCK_DPE > 0:  # MLA模式
            qpe = tl.load(  # 加载Query位置编码
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0  # 带掩码加载
            )
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):  # 沿KV维度分块迭代（带流水线）
            offs_n = start_n + tl.arange(0, BLOCK_N)  # 计算N维度偏移
            kv_loc = tl.load(  # 加载KV位置索引
                kv_indices + cur_batch_kv_start_idx + offs_n,  # 从索引数组获取
                mask=offs_n < split_kv_end,  # 掩码
                other=0,  # 越界填充值
            )
            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k  # 计算Key缓存完整偏移
            k = tl.load(  # 加载Key向量
                K_Buffer + offs_buf_k,  # Key缓存地址
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),  # 掩码
                other=0.0,  # 越界填充值
            )
            qk = tl.dot(q_k, k)  # 使用dot乘法计算QK点积（更高效）
            if BLOCK_DPE > 0:  # MLA模式：额外计算位置编码的QK
                offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + base_offs_kpe  # Key位置编码完整偏移
                kpe = tl.load(  # 加载Key位置编码
                    K_Buffer + offs_buf_kpe,  # Key缓存地址
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),  # 掩码
                    other=0.0,  # 越界填充值
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))  # 累加位置编码的QK点积
            qk *= sm_scale_withk  # 乘以softmax缩放因子

            if logit_cap > 0:  # 如果启用logit截断
                qk = logit_cap * tanh(qk / logit_cap)  # 应用logit截断

            if xai_temperature_len > 0:  # 如果启用XAI温度调节
                qk *= xai_temperature_reg[:, None]  # 乘以温度调节系数

            qk = tl.where(  # 应用掩码
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")  # 无效位置设为负无穷
            )
            if HAS_MLA:  # MLA模式：V = K的转置
                v = tl.trans(k)  # Key转置作为Value
            else:  # 非MLA模式
                offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v  # 计算Value缓存完整偏移
                v = tl.load(  # 加载Value向量
                    V_Buffer + offs_buf_v,  # Value缓存地址
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),  # 掩码
                    other=0.0,  # 越界填充值
                )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)  # 更新最大指数值（沿KV维度取最大）
            re_scale = tl.exp(e_max - n_e_max)  # 计算重缩放因子
            p = tl.exp(qk - n_e_max[:, None])  # 计算softmax概率
            acc *= re_scale[:, None]  # 重缩放累加器
            acc += tl.dot(p.to(v.dtype), v)  # 使用dot乘法累加加权Value

            e_sum = e_sum * re_scale + tl.sum(p, 1)  # 更新指数求和
            e_max = n_e_max  # 更新最大指数值

        offs_mid_o = (  # 计算中间输出偏移
            cur_batch * stride_mid_ob  # 批次偏移
            + cur_head[:, None] * stride_mid_oh  # 头偏移
            + split_kv_id * stride_mid_os  # 分块偏移
            + offs_dv[None, :]  # 维度偏移
        )

        tl.store(  # 存储中间输出（归一化后的注意力值）
            Att_Out + offs_mid_o,  # 中间输出地址
            acc / e_sum[:, None],  # 归一化结果
            mask=(mask_h[:, None]) & (mask_dv[None, :]),  # 掩码
        )

        offs_mid_o_1 = (  # 计算LSE存储偏移
            cur_batch * stride_mid_ob  # 批次偏移
            + cur_head * stride_mid_oh  # 头偏移
            + split_kv_id * stride_mid_os  # 分块偏移
        ) // Lv  # 除以Value维度

        tl.store(  # 存储对数求和指数
            Att_Lse + offs_mid_o_1,  # LSE地址
            e_max + tl.log(e_sum),  # log(sum(exp))
            mask=mask_h,  # 头掩码
        )

    if USE_PDL:  # 如果使用PDL（程序依赖启动）
        tl.extra.cuda.gdc_launch_dependents()  # 启动依赖的内核


def _decode_grouped_att_m_fwd(  # 解码注意力GQA/MQA/MLA前向传播（阶段1）
    q,  # Query张量
    k_buffer,  # Key缓存
    v_buffer,  # Value缓存
    att_out,  # 注意力中间输出
    att_lse,  # 注意力LSE
    kv_indptr,  # KV索引偏移
    kv_indices,  # KV索引
    num_kv_splits,  # KV分块数
    max_kv_splits,  # 最大KV分块数
    sm_scale_withk,  # 包含K缩放的softmax缩放因子
    logit_cap,  # logit截断值
    xai_temperature_len=-1,  # XAI温度长度（默认-1）
    has_mla=False,  # 是否为MLA模式（默认False）
    use_pdl=False,  # 是否使用PDL（默认False）
):  # 解码注意力GQA/MQA/MLA模式阶段1前向传播
    """解码注意力GQA/MQA/MLA模式阶段1前向传播，启动分组内核"""
    BLOCK = 32  # 默认块大小
    Lk = k_buffer.shape[-1]  # Key头维度
    Lv = v_buffer.shape[-1]  # Value头维度

    # [TODO] work around shmem limit on MI3xx
    # [待办] 解决MI3xx上的共享内存限制
    if _is_hip and Lk >= 576:  # HIP平台且Key维度>=576
        BLOCK = 16  # 减小块大小以规避共享内存限制

    if Lk == 576:  # MLA 576维
        BLOCK_DMODEL = 512  # 模型维度块大小
        BLOCK_DPE = 64  # 位置编码维度块大小
    elif Lk == 288:  # MLA 288维
        BLOCK_DMODEL = 256  # 模型维度块大小
        BLOCK_DPE = 32  # 位置编码维度块大小
    else:  # 其他维度
        BLOCK_DMODEL = triton.next_power_of_2(Lk)  # 对齐到2的幂
        BLOCK_DPE = 0  # 无位置编码维度
    BLOCK_DV = triton.next_power_of_2(Lv)  # Value维度对齐到2的幂

    batch, head_num = q.shape[0], q.shape[1]  # 获取批次大小和头数
    kv_group_num = q.shape[1] // k_buffer.shape[1]  # 计算KV组数

    BLOCK_H = 16  # 头维度块大小
    MAX_KV_SPLITS = max_kv_splits  # 最大KV分块数
    grid = (  # 定义3D网格
        batch,  # 批次维度
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),  # 头维度（按组对齐）
        MAX_KV_SPLITS,  # KV分块维度
    )

    extra_kargs = {}  # 额外内核参数
    num_stages = 2  # 流水线阶段数
    if _is_hip:  # HIP平台额外参数
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}  # ROCm优化参数
        num_stages = 1  # HIP平台使用1个流水线阶段

    _fwd_grouped_kernel_stage1[grid](  # 启动分组阶段1内核
        q,  # Query张量
        k_buffer,  # Key缓存
        v_buffer,  # Value缓存
        sm_scale_withk,  # softmax缩放因子
        kv_indptr,  # KV索引偏移
        kv_indices,  # KV索引
        att_out,  # 中间输出
        att_lse,  # LSE
        num_kv_splits,  # KV分块数
        q.stride(0),  # Query批次步长
        q.stride(1),  # Query头步长
        k_buffer.stride(0),  # Key缓存批次步长
        k_buffer.stride(1),  # Key缓存头步长
        v_buffer.stride(0),  # Value缓存批次步长
        v_buffer.stride(1),  # Value缓存头步长
        att_out.stride(0),  # 中间输出批次步长
        att_out.stride(1),  # 中间输出头步长
        att_out.stride(2),  # 中间输出分块步长
        kv_group_num=kv_group_num,  # KV组数
        q_head_num=head_num,  # Query头数
        BLOCK_DMODEL=BLOCK_DMODEL,  # 模型维度块大小
        BLOCK_DPE=BLOCK_DPE,  # 位置编码维度块大小
        BLOCK_DV=BLOCK_DV,  # Value维度块大小
        BLOCK_N=BLOCK,  # N维度块大小
        BLOCK_H=BLOCK_H,  # 头维度块大小
        MIN_BLOCK_KV=_MIN_BLOCK_KV,  # KV最小块大小
        logit_cap=logit_cap,  # logit截断值
        xai_temperature_len=xai_temperature_len,  # XAI温度长度
        num_warps=4,  # warp数量
        num_stages=num_stages,  # 流水线阶段数
        Lk=Lk,  # Key头维度
        Lv=Lv,  # Value头维度
        HAS_MLA=has_mla,  # 是否MLA模式
        USE_PDL=use_pdl,  # 是否使用PDL
        **extra_kargs,  # 额外参数
    )


@triton.jit  # Triton JIT编译装饰器
def _fwd_kernel_stage2(  # 解码注意力阶段2内核（跨块归约softmax）
    Mid_O,  # 阶段1中间输出指针
    Mid_O_1,  # 阶段1中间LSE指针
    O,  # 最终输出指针
    v_scale,  # Value缩放因子
    kv_indptr,  # KV索引偏移指针
    num_kv_splits,  # KV分块数指针
    sink_ptr,  # 注意力汇聚指针
    stride_mid_ob,  # 中间输出批次步长
    stride_mid_oh,  # 中间输出头步长
    stride_mid_os,  # 中间输出分块步长
    stride_obs,  # 输出批次步长
    stride_oh,  # 输出头步长
    MAX_KV_SPLITS: tl.constexpr,  # 最大KV分块数（编译时常量）
    MIN_BLOCK_KV: tl.constexpr,  # KV最小块大小（编译时常量）
    BLOCK_DV: tl.constexpr,  # Value维度块大小（编译时常量）
    Lv: tl.constexpr,  # Value头维度（编译时常量）
    HAS_SINK: tl.constexpr,  # 是否有注意力汇聚（编译时常量）
    USE_PDL: tl.constexpr = False,  # 是否使用PDL（编译时常量，默认False）
):  # 解码注意力阶段2：跨KV分块归约softmax并生成最终输出
    """解码注意力阶段2内核，跨KV分块归约softmax，支持注意力汇聚"""
    cur_batch = tl.program_id(0)  # 获取当前批次索引
    cur_head = tl.program_id(1)  # 获取当前头索引

    if USE_PDL:  # 如果使用PDL
        tl.extra.cuda.gdc_wait()  # 等待依赖的内核完成

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(  # 计算序列KV长度
        kv_indptr + cur_batch  # 结束索引减起始索引
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)  # 加载KV分块数

    offs_d = tl.arange(0, BLOCK_DV)  # Value维度偏移
    mask_d = offs_d < Lv  # Value维度掩码

    e_sum = 0.0  # 初始化指数求和
    e_max = -float("inf")  # 初始化最大指数值
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)  # 初始化Value累加器

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d  # 中间输出偏移
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv  # LSE偏移
    kv_len_per_split = (  # 计算每个KV分块的长度
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV  # 对齐到MIN_BLOCK_KV
    )

    for split_kv_id in tl.range(0, MAX_KV_SPLITS, num_stages=2):  # 遍历所有KV分块（带流水线）
        split_kv_start = kv_len_per_split * split_kv_id  # 当前分块起始位置
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)  # 当前分块结束位置

        if split_kv_end > split_kv_start:  # 如果当前分块非空
            tv = tl.load(  # 加载中间输出值
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0  # 带掩码加载
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)  # 加载中间LSE
            n_e_max = tl.maximum(tlogic, e_max)  # 更新最大指数值

            old_scale = tl.exp(e_max - n_e_max)  # 计算旧值的重缩放因子
            acc *= old_scale  # 重缩放累加器
            exp_logic = tl.exp(tlogic - n_e_max)  # 计算当前分块的指数值
            acc += exp_logic * tv  # 累加加权中间输出

            e_sum = e_sum * old_scale + exp_logic  # 更新指数求和
            e_max = n_e_max  # 更新最大指数值

    if HAS_SINK:  # 如果有注意力汇聚
        cur_sink = tl.load(sink_ptr + cur_head)  # 加载汇聚值
        e_sum += tl.exp(cur_sink - e_max)  # 将汇聚值加入指数求和

    tl.store(  # 存储最终输出
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,  # 输出地址
        acc / e_sum * v_scale,  # 归一化后乘以Value缩放因子
        mask=mask_d,  # 掩码
    )


def _decode_softmax_reducev_fwd(  # 解码注意力阶段2：softmax归约Value前向传播
    logits,  # 阶段1中间输出
    lse,  # 阶段1中间LSE
    q,  # Query张量（用于获取形状）
    o,  # 最终输出
    v_scale,  # Value缩放因子
    v_buffer,  # Value缓存（用于获取形状）
    kv_indptr,  # KV索引偏移
    num_kv_splits,  # KV分块数
    max_kv_splits,  # 最大KV分块数
    sinks=None,  # 注意力汇聚（可选）
    use_pdl=False,  # 是否使用PDL（默认False）
):  # 跨KV分块归约softmax并生成最终输出
    """解码注意力阶段2前向传播，跨KV分块归约softmax"""
    batch, head_num = q.shape[0], q.shape[1]  # 获取批次大小和头数
    Lv = v_buffer.shape[-1]  # Value头维度
    BLOCK_DV = triton.next_power_of_2(Lv)  # Value维度对齐到2的幂

    MAX_KV_SPLITS = max_kv_splits  # 最大KV分块数
    HAS_SINK = sinks is not None  # 是否有注意力汇聚

    extra_kargs = {}  # 额外内核参数
    if _is_hip:  # HIP平台额外参数
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}  # ROCm优化参数

    grid = (batch, head_num)  # 定义2D网格
    _fwd_kernel_stage2[grid](  # 启动阶段2内核
        logits,  # 中间输出
        lse,  # 中间LSE
        o,  # 最终输出
        v_scale,  # Value缩放因子
        kv_indptr,  # KV索引偏移
        num_kv_splits,  # KV分块数
        sinks,  # 注意力汇聚
        logits.stride(0),  # 中间输出批次步长
        logits.stride(1),  # 中间输出头步长
        logits.stride(2),  # 中间输出分块步长
        o.stride(0),  # 输出批次步长
        o.stride(1),  # 输出头步长
        MAX_KV_SPLITS=MAX_KV_SPLITS,  # 最大KV分块数
        MIN_BLOCK_KV=_MIN_BLOCK_KV,  # KV最小块大小
        BLOCK_DV=BLOCK_DV,  # Value维度块大小
        Lv=Lv,  # Value头维度
        HAS_SINK=HAS_SINK,  # 是否有注意力汇聚
        USE_PDL=use_pdl,  # 是否使用PDL
        num_warps=4,  # warp数量
        num_stages=2,  # 流水线阶段数
        **({"launch_pdl": True} if use_pdl else {}),  # PDL启动参数
        **extra_kargs,  # 额外参数
    )


def decode_attention_fwd_normal(  # 解码注意力MHA模式前向传播
    q,  # Query张量
    k_buffer,  # Key缓存
    v_buffer,  # Value缓存
    o,  # 输出张量
    kv_indptr,  # KV索引偏移
    kv_indices,  # KV索引
    attn_logits,  # 注意力中间输出
    attn_lse,  # 注意力LSE
    num_kv_splits,  # KV分块数
    max_kv_splits,  # 最大KV分块数
    sm_scale_withk,  # 包含K缩放的softmax缩放因子
    v_scale,  # Value缩放因子
    logit_cap=0.0,  # logit截断值（默认0，不截断）
    sinks=None,  # 注意力汇聚（可选）
    xai_temperature_len=-1,  # XAI温度长度（默认-1）
):  # MHA模式解码注意力前向传播，依次执行阶段1和阶段2
    """MHA模式解码注意力前向传播，依次执行阶段1和阶段2"""
    _decode_att_m_fwd(  # 执行阶段1：分块计算QKV注意力
        q,  # Query张量
        k_buffer,  # Key缓存
        v_buffer,  # Value缓存
        attn_logits,  # 中间输出
        attn_lse,  # LSE
        kv_indptr,  # KV索引偏移
        kv_indices,  # KV索引
        num_kv_splits,  # KV分块数
        max_kv_splits,  # 最大KV分块数
        sm_scale_withk,  # softmax缩放因子
        logit_cap,  # logit截断值
        xai_temperature_len,  # XAI温度长度
    )
    _decode_softmax_reducev_fwd(  # 执行阶段2：跨块归约softmax
        attn_logits,  # 中间输出
        attn_lse,  # LSE
        q,  # Query张量
        o,  # 最终输出
        v_scale,  # Value缩放因子
        v_buffer,  # Value缓存
        kv_indptr,  # KV索引偏移
        num_kv_splits,  # KV分块数
        max_kv_splits,  # 最大KV分块数
        sinks,  # 注意力汇聚
    )


def decode_attention_fwd_grouped(  # 解码注意力GQA/MQA/MLA模式前向传播
    q,  # Query张量
    k_buffer,  # Key缓存
    v_buffer,  # Value缓存
    o,  # 输出张量
    kv_indptr,  # KV索引偏移
    kv_indices,  # KV索引
    attn_logits,  # 注意力中间输出
    attn_lse,  # 注意力LSE
    num_kv_splits,  # KV分块数
    max_kv_splits,  # 最大KV分块数
    sm_scale_withk,  # 包含K缩放的softmax缩放因子
    v_scale,  # Value缩放因子
    logit_cap=0.0,  # logit截断值（默认0）
    sinks=None,  # 注意力汇聚（可选）
    xai_temperature_len=-1,  # XAI温度长度（默认-1）
    has_mla=False,  # 是否为MLA模式（默认False）
    use_pdl=False,  # 是否使用PDL（默认False）
):  # GQA/MQA/MLA模式解码注意力前向传播，依次执行阶段1和阶段2
    """GQA/MQA/MLA模式解码注意力前向传播，依次执行阶段1和阶段2"""
    _decode_grouped_att_m_fwd(  # 执行阶段1：分组模式分块计算QKV注意力
        q,  # Query张量
        k_buffer,  # Key缓存
        v_buffer,  # Value缓存
        attn_logits,  # 中间输出
        attn_lse,  # LSE
        kv_indptr,  # KV索引偏移
        kv_indices,  # KV索引
        num_kv_splits,  # KV分块数
        max_kv_splits,  # 最大KV分块数
        sm_scale_withk,  # softmax缩放因子
        logit_cap,  # logit截断值
        xai_temperature_len,  # XAI温度长度
        has_mla=has_mla,  # 是否MLA模式
        use_pdl=use_pdl,  # 是否使用PDL
    )
    _decode_softmax_reducev_fwd(  # 执行阶段2：跨块归约softmax
        attn_logits,  # 中间输出
        attn_lse,  # LSE
        q,  # Query张量
        o,  # 最终输出
        v_scale,  # Value缩放因子
        v_buffer,  # Value缓存
        kv_indptr,  # KV索引偏移
        num_kv_splits,  # KV分块数
        max_kv_splits,  # 最大KV分块数
        sinks,  # 注意力汇聚
        use_pdl=use_pdl,  # 是否使用PDL
    )


def decode_attention_fwd(  # 解码注意力前向传播统一入口函数
    q,  # Query张量
    k_buffer,  # Key缓存
    v_buffer,  # Value缓存
    o,  # 输出张量
    kv_indptr,  # KV索引偏移
    kv_indices,  # KV索引
    attn_logits,  # 注意力中间输出
    attn_lse,  # 注意力LSE
    num_kv_splits,  # KV分块数
    max_kv_splits,  # 最大KV分块数
    sm_scale,  # softmax缩放因子
    k_scale,  # Key缩放因子
    v_scale,  # Value缩放因子
    logit_cap=0.0,  # logit截断值（默认0）
    sinks=None,  # 注意力汇聚（可选）
    xai_temperature_len=-1,  # XAI温度长度（默认-1）
    has_mla=False,  # 是否为MLA模式（默认False）
    use_pdl=False,  # 是否使用PDL（默认False）
):  # 根据KV组数自动选择MHA或GQA/MQA/MLA模式
    """解码注意力前向传播统一入口，根据KV组数自动选择MHA或GQA/MQA/MLA模式"""
    assert max_kv_splits == attn_logits.shape[2]  # 断言：最大KV分块数与中间输出维度一致
    assert q.shape[0] <= kv_indptr.shape[0] - 1  # 断言：Query批次不超过KV索引范围
    assert q.shape[0] <= attn_logits.shape[0]  # 断言：Query批次不超过中间输出维度

    kv_group_num = q.shape[1] // v_buffer.shape[1]  # 计算KV组数

    if kv_group_num == 1:  # MHA模式
        # MHA
        # MHA（多头注意力）
        decode_attention_fwd_normal(  # 调用MHA模式
            q,  # Query张量
            k_buffer,  # Key缓存
            v_buffer,  # Value缓存
            o,  # 输出张量
            kv_indptr,  # KV索引偏移
            kv_indices,  # KV索引
            attn_logits,  # 中间输出
            attn_lse,  # LSE
            num_kv_splits,  # KV分块数
            max_kv_splits,  # 最大KV分块数
            sm_scale * k_scale,  # 合并softmax和Key缩放因子
            v_scale,  # Value缩放因子
            logit_cap=logit_cap,  # logit截断值
            sinks=sinks,  # 注意力汇聚
            xai_temperature_len=xai_temperature_len,  # XAI温度长度
        )
    else:  # GQA/MQA/MLA模式
        # GQA/MQA/MLA
        # GQA（分组查询注意力）/MQA（多查询注意力）/MLA（多线性注意力）
        decode_attention_fwd_grouped(  # 调用GQA/MQA/MLA模式
            q,  # Query张量
            k_buffer,  # Key缓存
            v_buffer,  # Value缓存
            o,  # 输出张量
            kv_indptr,  # KV索引偏移
            kv_indices,  # KV索引
            attn_logits,  # 中间输出
            attn_lse,  # LSE
            num_kv_splits,  # KV分块数
            max_kv_splits,  # 最大KV分块数
            sm_scale * k_scale,  # 合并softmax和Key缩放因子
            v_scale,  # Value缩放因子
            logit_cap=logit_cap,  # logit截断值
            sinks=sinks,  # 注意力汇聚
            xai_temperature_len=xai_temperature_len,  # XAI温度长度
            has_mla=has_mla,  # 是否MLA模式
            use_pdl=use_pdl,  # 是否使用PDL
        )
