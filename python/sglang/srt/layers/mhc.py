# 多头混合(MHC)层模块，实现基于TileLang的MHC前处理、后处理和融合内核，用于DSV4模型的路由混合计算
# SPDX-License-Identifier: Apache-2.0
import functools  # 导入偏函数工具
import math  # 导入数学函数
from typing import Tuple  # 导入元组类型

import tilelang  # 导入TileLang编译器
import tilelang.language as T  # 导入TileLang语言
import torch  # 导入PyTorch核心库

from sglang.jit_kernel.utils import is_arch_support_pdl  # 导入PDL架构支持检测
from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_round_robin_split  # 导入DSA预填充检测
from sglang.srt.layers.utils.common import strict_contiguous  # 导入严格连续性检查

tilelang.set_log_level("WARNING")  # 设置TileLang日志级别为警告

pass_configs = {  # TileLang通道配置
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,  # 禁用warp特化
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,  # 禁用TMA降低
}

FP8 = "float8_e4m3"  # FP8数据类型字符串
BF16 = "bfloat16"  # BF16数据类型字符串
FP32 = "float32"  # FP32数据类型字符串
INT32 = "int32"  # INT32数据类型字符串


@tilelang.jit(pass_configs=pass_configs)  # TileLang JIT编译，使用通道配置
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):  # HC分割Sinkhorn内核
    n = T.symbolic("n")  # 符号化token数量
    mix_hc = (2 + hc) * hc  # 混合维度：(2+hc)*hc
    threads = 64  # 线程数

    ENABLE_PDL = is_arch_support_pdl()  # 检测是否支持PDL

    @T.prim_func  # TileLang原语函数
    def hc_split_sinkhorn_kernel_(  # HC分割Sinkhorn内核实现
        mixes: T.Tensor[(n, mix_hc), FP32],  # 混合输入张量
        hc_scale: T.Tensor[(3,), T.float32],  # HC缩放因子
        hc_base: T.Tensor[(mix_hc,), T.float32],  # HC偏置基底
        pre: T.Tensor[(n, hc), FP32],  # 前混合输出
        post: T.Tensor[(n, hc), FP32],  # 后混合输出
        comb: T.Tensor[(n, hc, hc), FP32],  # 组合混合输出
    ):
        with T.Kernel(n, threads=threads) as i:  # 启动n个CTA，每个64线程
            if ENABLE_PDL:  # 如果支持PDL
                T.pdl_sync()  # PDL同步

            mixes_shared = T.alloc_shared(mix_hc, FP32)  # 分配共享内存
            comb_frag = T.alloc_fragment((hc, hc), FP32)  # 分配组合片段寄存器
            T.copy(mixes[i, :], mixes_shared)  # 从全局内存拷贝到共享内存

            for j in T.Parallel(hc):  # 并行计算前混合
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps  # sigmoid+eps
            for j in T.Parallel(hc):  # 并行计算后混合
                post[i, j] = 2 * T.sigmoid(  # 2*sigmoid
                    mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc]
                )
            for j, k in T.Parallel(hc, hc):  # 并行计算组合logits
                comb_frag[j, k] = (  # 线性变换
                    mixes_shared[j * hc + k + hc * 2] * hc_scale[2]
                    + hc_base[j * hc + k + hc * 2]
                )

            row_sum = T.alloc_fragment(hc, FP32)  # 行求和片段
            col_sum = T.alloc_fragment(hc, FP32)  # 列求和片段

            row_max = T.alloc_fragment(hc, FP32)  # 行最大值片段
            T.reduce_max(comb_frag, row_max, dim=1)  # 计算每行最大值
            for j, k in T.Parallel(hc, hc):  # 减去行最大值并取指数
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)  # 计算行求和
            for j, k in T.Parallel(hc, hc):  # 行归一化
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            T.reduce_sum(comb_frag, col_sum, dim=0)  # 计算列求和
            for j, k in T.Parallel(hc, hc):  # 列归一化
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):  # Sinkhorn迭代
                T.reduce_sum(comb_frag, row_sum, dim=1)  # 行求和
                for j, k in T.Parallel(hc, hc):  # 行归一化
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)  # 列求和
                for j, k in T.Parallel(hc, hc):  # 列归一化
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])  # 将结果写回全局内存
            if ENABLE_PDL:  # 如果支持PDL
                T.pdl_trigger()  # PDL触发

    return hc_split_sinkhorn_kernel_  # 返回内核函数


def hc_split_sinkhorn(  # HC分割Sinkhorn主机函数
    mixes: torch.Tensor,  # 混合输入张量
    hc_scale: torch.Tensor,  # HC缩放因子
    hc_base: torch.Tensor,  # HC偏置基底
    hc_mult: int = 4,  # HC乘数，默认4
    sinkhorn_iters: int = 20,  # Sinkhorn迭代次数，默认20
    eps: float = 1e-6,  # epsilon值，默认1e-6
):
    b, s, _ = mixes.size()  # 获取批次大小、序列长度和混合维度
    pre = mixes.new_empty(b, s, hc_mult)  # 分配前混合输出
    post = mixes.new_empty(b, s, hc_mult)  # 分配后混合输出
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)  # 分配组合混合输出
    kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)  # 编译内核
    kernel(  # 启动内核
        mixes.view(-1, (2 + hc_mult) * hc_mult),  # 展平并重塑mixes
        hc_scale,  # 缩放因子
        hc_base,  # 偏置基底
        pre.view(-1, hc_mult),  # 前混合输出视图
        post.view(-1, hc_mult),  # 后混合输出视图
        comb.view(-1, hc_mult, hc_mult),  # 组合输出视图
    )
    return pre, post, comb  # 返回前混合、后混合和组合结果


@tilelang.jit(  # TileLang JIT编译
    pass_configs={  # 通道配置
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,  # 禁用warp特化
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,  # 禁用TMA降低
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,  # PTXAS寄存器使用级别
    },
)
def mhc_pre_big_fuse_tilelang(  # MHC前处理大融合TileLang内核
    gemm_out_mul,  # GEMM输出乘积累加
    gemm_out_sqrsum,  # GEMM输出平方和
    hc_scale,  # HC缩放因子
    hc_base,  # HC偏置基底
    residual,  # 残差输入
    post_mix,  # 后混合输出
    comb_mix,  # 组合混合输出
    layer_input,  # 层输入输出
    hidden_size: int,  # 隐藏层大小
    rms_eps: float,  # RMS epsilon
    hc_pre_eps: float,  # HC前混合epsilon
    hc_sinkhorn_eps: float,  # HC Sinkhorn epsilon
    hc_post_mult_value: float,  # HC后混合乘数值
    sinkhorn_repeat: int,  # Sinkhorn重复次数
    n_splits: int = 16,  # 分割数，默认16
    hc_mult: int = 4,  # HC乘数，默认4
    gemm_last_dim: int = -1,  # GEMM最后维度，默认-1
):
    num_tokens = T.dynamic("num_tokens")  # 动态token数量
    hc_mult3 = hc_mult * (2 + hc_mult)  # 混合维度
    if gemm_last_dim < 0:  # 如果未指定最后维度
        gemm_last_dim = hc_mult3  # 使用混合维度
    hidden_block = math.gcd(512, hidden_size)  # 计算隐藏块大小

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]  # GEMM乘积累加张量
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # GEMM平方和张量
    hc_scale: T.Tensor[[3], T.float32]  # 缩放因子张量
    hc_base: T.Tensor[[hc_mult3], T.float32]  # 偏置基底张量
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # 残差张量
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # 后混合输出张量
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # 组合混合输出张量
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # 层输入输出张量

    ENABLE_PDL = is_arch_support_pdl()  # 检测PDL支持
    with T.Kernel(num_tokens, threads=96) as i:  # 启动num_tokens个CTA，每个96线程
        rms = T.alloc_fragment(1, T.float32)  # 分配RMS片段
        mixes = T.alloc_fragment(hc_mult3, T.float32)  # 分配混合片段
        T.clear(mixes)  # 清零混合片段
        rms[0] = 0  # 初始化RMS

        if ENABLE_PDL:  # 如果支持PDL
            T.pdl_sync()  # PDL同步

        for i_split in T.serial(n_splits):  # 累加所有分割的平方和
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)  # 计算rsqrt归一化
        for j in T.Parallel(hc_mult3):  # 累加所有分割的GEMM结果并归一化
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]  # 乘以rsqrt
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)  # 分配共享内存
        T.copy(mixes, mixes_shared)  # 拷贝到共享内存

        if T.get_thread_binding() < 32:  # 前32个线程处理post和comb
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)  # 分配组合矩阵片段
            for j in T.Parallel(hc_mult):  # 计算后混合
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value  # 乘以后混合乘数
                )
            for j, k in T.Parallel(hc_mult, hc_mult):  # 计算组合logits
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)  # 行求和
            col_sum = T.alloc_fragment(hc_mult, T.float32)  # 列求和

            row_max = T.alloc_fragment(hc_mult, T.float32)  # 行最大值
            T.reduce_max(cm, row_max, dim=1)  # 计算行最大值
            for j, k in T.Parallel(hc_mult, hc_mult):  # 减去最大值并取指数
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)  # 行求和
            for j, k in T.Parallel(hc_mult, hc_mult):  # 行归一化+eps
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)  # 列求和
            for j, k in T.Parallel(hc_mult, hc_mult):  # 列归一化
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):  # Sinkhorn迭代
                T.reduce_sum(cm, row_sum, dim=1)  # 行求和
                for j, k in T.Parallel(hc_mult, hc_mult):  # 行归一化
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)  # 列求和
                for j, k in T.Parallel(hc_mult, hc_mult):  # 列归一化
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):  # 写入组合混合结果
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:  # 后64个线程处理前混合和层输入
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)  # 分配前混合共享内存
            for j in T.Parallel(hc_mult):  # 计算前混合
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps  # 加上前混合epsilon
                )
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):  # 流水线处理隐藏维度
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)  # 分配共享内存
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)  # 分配寄存器片段
                T.copy(residual[i, 0, i0_h * hidden_block], xs)  # 加载残差
                T.copy(xs, xl)  # 拷贝到寄存器

                ol = T.alloc_fragment(hidden_block, T.float32)  # 分配输出片段
                T.clear(ol)  # 清零

                for i_hc in T.serial(hc_mult):  # 对每个HC路由累加
                    pre = pre_mix_shared[i_hc]  # 获取前混合系数
                    for i1_h in T.Parallel(hidden_block):  # 加权求和
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                T.copy(ol, layer_input[i, i0_h * hidden_block])  # 写入层输入

        if ENABLE_PDL:  # 如果支持PDL
            T.pdl_trigger()  # PDL触发


@tilelang.jit  # TileLang JIT编译
def mhc_pre_gemm_sqrsum_tilelang(  # MHC前处理GEMM+平方和TileLang内核
    x,  # 输入张量
    fn,  # 函数权重
    out,  # GEMM输出
    sqrsum,  # 平方和输出
    hc_mult3: int,  # 混合维度
    hc_hidden_size: int,  # HC隐藏层大小
    token_block: int = 32,  # token块大小，默认32
    hidden_block: int = 256,  # 隐藏块大小，默认256
) -> tilelang.JITKernel:  # 返回JIT内核
    assert hc_mult3 <= 32  # 断言混合维度不超过32
    num_tokens = T.dynamic("num_tokens")  # 动态token数量
    assert hc_hidden_size % hidden_block == 0  # 断言隐藏层可被块整除

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)  # 输入张量声明
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)  # 函数权重声明
    out: T.Tensor((num_tokens, hc_mult3), T.float32)  # GEMM输出声明
    sqrsum: T.Tensor((num_tokens), T.float32)  # 平方和输出声明

    ENABLE_PDL = is_arch_support_pdl()  # 检测PDL支持
    with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:  # 按token块启动内核
        out_frag = T.alloc_fragment((token_block, 32), T.float32)  # 分配输出片段
        sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)  # 分配平方和部分片段
        T.clear(out_frag)  # 清零输出
        T.clear(sqrsum_part)  # 清零平方和
        if ENABLE_PDL:  # 如果支持PDL
            T.pdl_sync()  # PDL同步
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=2):  # 流水线处理隐藏维度
            x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)  # 分配x共享内存
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)  # 分配fn共享内存

            T.annotate_layout(  # 注解布局
                {x_smem_16: tilelang.layout.make_swizzled_layout(x_smem_16)}  # 交错布局
            )

            T.copy(x[px * token_block, pz * hidden_block], x_smem_16)  # 加载x数据
            T.copy(fn[0, pz * hidden_block], fn_smem)  # 加载fn数据

            x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)  # 分配bf16片段
            T.copy(x_smem_16, x_frag_16)  # 拷贝到寄存器
            x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)  # 分配fp32片段
            T.copy(x_frag_16, x_frag)  # 转换为fp32

            for jj in T.serial(hidden_block // 4):  # 计算4路平方和
                for i, j in T.Parallel(token_block, 4):
                    sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]

            T.gemm(  # 执行GEMM
                x_frag,  # 矩阵A
                fn_smem,  # 矩阵B
                out_frag,  # 矩阵C（累加）
                transpose_A=False,  # A不转置
                transpose_B=True,  # B转置
                wg_wait=0,  # warp组等待0
                clear_accum=False,  # 不清零累加器
            )
        sqrsum_l = T.alloc_fragment(token_block, T.float32)  # 分配平方和片段
        T.reduce_sum(sqrsum_part, sqrsum_l)  # 归约平方和
        for i in T.Parallel(token_block):  # 写入平方和
            sqrsum[px * token_block + i] = sqrsum_l[i]
        for i, j in T.Parallel(token_block, 32):  # 写入GEMM输出
            if j < hc_mult3:  # 只写有效列
                out[px * token_block + i, j] = out_frag[i, j]
        if ENABLE_PDL:  # 如果支持PDL
            T.pdl_trigger()  # PDL触发


@functools.cache  # 缓存装饰器，避免重复编译
def mhc_pre_gemm_sqrsum_splitk_kernel(  # MHC前处理GEMM+平方和SplitK内核
    hc_mult3: int,  # 混合维度
    hc_hidden_size: int,  # HC隐藏层大小
    split_k: int,  # SplitK因子
    token_block: int = 32,  # token块大小
    hidden_block: int = 256,  # 隐藏块大小
    threads: int = 128,  # 线程数
) -> Tuple[tilelang.JITKernel, tilelang.JITKernel]:  # 返回两个阶段的内核
    assert hc_mult3 <= 32  # 断言混合维度不超过32
    assert hc_hidden_size % hidden_block == 0  # 断言隐藏层可被块整除
    assert hc_hidden_size % split_k == 0  # 断言隐藏层可被SplitK整除
    split_size = hc_hidden_size // split_k  # 计算每个分割的大小
    assert split_size % hidden_block == 0  # 断言分割大小可被块整除

    num_tokens = T.dynamic("num_tokens")  # 动态token数量

    ENABLE_PDL = is_arch_support_pdl()  # 检测PDL支持

    @tilelang.jit  # 第0阶段：部分GEMM和平方和
    def mhc_pre_gemm_sqrsum_splitk_stage_0(  # SplitK第0阶段内核
        x: T.Tensor[(num_tokens, hc_hidden_size), T.bfloat16],  # 输入张量
        fn: T.Tensor[(hc_mult3, hc_hidden_size), T.float32],  # 函数权重
        out_partial: T.Tensor[(split_k, num_tokens, 32), T.float32],  # 部分GEMM输出
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],  # 部分平方和
    ):
        with T.Kernel(T.ceildiv(num_tokens, token_block), split_k, threads=threads) as (  # 启动内核
            px,  # token块索引
            bz,  # SplitK索引
        ):
            out_frag = T.alloc_fragment((token_block, 32), T.float32)  # 分配输出片段
            sq_part4 = T.alloc_fragment((token_block, 4), T.float32)  # 分配平方和片段
            T.clear(out_frag)  # 清零输出
            T.clear(sq_part4)  # 清零平方和

            k_base = bz * split_size  # 计算当前分割的K起始位置

            if ENABLE_PDL:  # 如果支持PDL
                T.pdl_sync()  # PDL同步

            for pz in T.Pipelined(split_size // hidden_block, num_stages=2):  # 流水线处理
                x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)  # x共享内存
                fn_smem = T.alloc_shared((32, hidden_block), T.float32)  # fn共享内存

                T.annotate_layout(  # 注解布局
                    {x_smem: tilelang.layout.make_swizzled_layout(x_smem)}  # 交错布局
                )

                T.copy(x[px * token_block, k_base + pz * hidden_block], x_smem)  # 加载x
                T.copy(fn[0, k_base + pz * hidden_block], fn_smem)  # 加载fn

                x_f16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)  # bf16片段
                T.copy(x_smem, x_f16)  # 拷贝
                x_f = T.alloc_fragment((token_block, hidden_block), T.float32)  # fp32片段
                T.copy(x_f16, x_f)  # 转换为fp32

                for jj in T.serial(hidden_block // 4):  # 计算4路平方和
                    for i, j in T.Parallel(token_block, 4):
                        v = x_f[i, jj * 4 + j]
                        sq_part4[i, j] += v * v

                T.gemm(  # 执行GEMM
                    x_f,  # 矩阵A
                    fn_smem,  # 矩阵B
                    out_frag,  # 矩阵C
                    transpose_A=False,  # A不转置
                    transpose_B=True,  # B转置
                    wg_wait=0,  # warp组等待
                    clear_accum=False,  # 不清零
                )

            sq_l = T.alloc_fragment((token_block,), T.float32)  # 平方和归约片段
            T.reduce_sum(sq_part4, sq_l)  # 归约

            for i in T.Parallel(token_block):  # 写入平方和
                t = px * token_block + i
                if t < num_tokens:  # 边界检查
                    sqrsum_partial[bz, t] = sq_l[i]

            for i, j in T.Parallel(token_block, 32):  # 写入GEMM输出
                t = px * token_block + i
                if t < num_tokens:  # 边界检查
                    out_partial[bz, t, j] = out_frag[i, j]

            if ENABLE_PDL:  # 如果支持PDL
                T.pdl_trigger()  # PDL触发

    @tilelang.jit  # 第1阶段：归约
    def mhc_pre_gemm_sqrsum_splitk_stage_1(  # SplitK第1阶段内核
        out_partial: T.Tensor[(split_k, num_tokens, 32), T.float32],  # 部分GEMM输出
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],  # 部分平方和
        out: T.Tensor[(num_tokens, hc_mult3), T.float32],  # 最终GEMM输出
        sqrsum: T.Tensor[(num_tokens,), T.float32],  # 最终平方和
    ):
        warps_per_cta = threads // 32  # 每个CTA的warp数
        num_reduce = T.ceildiv(split_k, 32)  # 归约轮数
        with T.Kernel(T.ceildiv(num_tokens, warps_per_cta), threads=threads) as (px,):  # 启动内核
            tx = T.get_thread_binding()  # 获取线程绑定
            warp = tx // 32  # 计算warp索引
            lane = tx % 32  # 计算lane索引
            t = px * warps_per_cta + warp  # 计算token索引
            s = T.alloc_local((1,), T.float32)  # 分配本地平方和
            acc = T.alloc_local((1,), T.float32)  # 分配本地累加器
            s[0] = 0  # 初始化平方和
            acc[0] = 0  # 初始化累加器
            if ENABLE_PDL:  # 如果支持PDL
                T.pdl_sync()  # PDL同步

            if t < num_tokens:  # 边界检查
                for r in T.serial(num_reduce):  # 归约平方和
                    bz = r * 32 + lane  # 计算SplitK索引
                    s[0] += T.if_then_else(bz < split_k, sqrsum_partial[bz, t], 0.0)  # 条件累加
                sqrsum[t] = T.warp_reduce_sum(s[0])  # warp内归约
                if lane < hc_mult3:  # 只在有效lane上归约GEMM
                    for bz in T.serial(split_k):  # 遍历所有分割
                        acc[0] += out_partial[bz, t, lane]  # 累加
                    out[t, lane] = acc[0]  # 写入结果

            if ENABLE_PDL:  # 如果支持PDL
                T.pdl_trigger()  # PDL触发

    return (  # 返回两个阶段的内核
        mhc_pre_gemm_sqrsum_splitk_stage_0,  # 第0阶段
        mhc_pre_gemm_sqrsum_splitk_stage_1,  # 第1阶段
    )


def _compute_num_split_for_mhc_pre(num_tokens: int, hc_hidden_size: int) -> int:  # 计算MHC前处理的分割数
    block_m, block_k = 64, 64  # GEMM块大小
    grid_size = (num_tokens + block_m - 1) // block_m  # 计算网格大小
    num_block_k = (hc_hidden_size + block_k - 1) // block_k  # 计算K维度块数
    n_sms = torch.cuda.get_device_properties(0).multi_processor_count  # 获取SM数量
    return max(1, min(n_sms // max(grid_size, 1), num_block_k // 4))  # 返回最优分割数


@tilelang.jit(  # TileLang JIT编译
    pass_configs={  # 通道配置
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,  # 禁用warp特化
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,  # 禁用TMA降低
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,  # PTXAS寄存器使用级别
    },
)
def mhc_pre_big_fuse_with_norm_tilelang(  # 带归一化的MHC前处理大融合内核
    gemm_out_mul,  # GEMM乘积累加
    gemm_out_sqrsum,  # GEMM平方和
    hc_scale,  # HC缩放因子
    hc_base,  # HC偏置基底
    residual,  # 残差输入
    post_mix,  # 后混合输出
    comb_mix,  # 组合混合输出
    layer_input,  # 层输入输出
    norm_weight,  # 归一化权重
    hidden_size: int,  # 隐藏层大小
    rms_eps: float,  # RMS epsilon
    hc_pre_eps: float,  # HC前混合epsilon
    hc_sinkhorn_eps: float,  # HC Sinkhorn epsilon
    hc_post_mult_value: float,  # HC后混合乘数值
    sinkhorn_repeat: int,  # Sinkhorn重复次数
    norm_eps: float,  # 归一化epsilon
    n_splits: int = 16,  # 分割数
    hc_mult: int = 4,  # HC乘数
    gemm_last_dim: int = -1,  # GEMM最后维度
):
    """Fused mhc_pre big_fuse + RMSNorm of layer_input.
    融合mhc_pre big_fuse + layer_input的RMS归一化。

    Identical to mhc_pre_big_fuse_tilelang for the (post_mix, comb_mix) path.
    For the layer_input path, the weighted-sum result is stashed in shared
    memory while accumulating sum_sq, then a second pipelined sweep applies
    rsqrt(sum_sq/D + norm_eps) * norm_weight before writing to HBM.
    对于(post_mix, comb_mix)路径与mhc_pre_big_fuse_tilelang相同。
    对于layer_input路径，加权求和结果暂存到共享内存并累加平方和，
    然后第二次流水线扫描应用rsqrt(sum_sq/D + norm_eps) * norm_weight后写入HBM。
    """
    num_tokens = T.dynamic("num_tokens")  # 动态token数量
    hc_mult3 = hc_mult * (2 + hc_mult)  # 混合维度
    if gemm_last_dim < 0:  # 如果未指定最后维度
        gemm_last_dim = hc_mult3  # 使用混合维度
    hidden_block = math.gcd(1024, hidden_size)  # 计算隐藏块大小

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]  # GEMM乘积累加张量
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # GEMM平方和张量
    hc_scale: T.Tensor[[3], T.float32]  # 缩放因子张量
    hc_base: T.Tensor[[hc_mult3], T.float32]  # 偏置基底张量
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # 残差张量
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # 后混合输出张量
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # 组合混合输出张量
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # 层输入输出张量
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]  # 归一化权重张量

    ENABLE_PDL = is_arch_support_pdl()  # 检测PDL支持
    with T.Kernel(num_tokens, threads=96) as i:  # 启动内核
        rms = T.alloc_fragment(1, T.float32)  # 分配RMS片段
        mixes = T.alloc_fragment(hc_mult3, T.float32)  # 分配混合片段
        T.clear(mixes)  # 清零
        rms[0] = 0  # 初始化

        if ENABLE_PDL:  # 如果支持PDL
            T.pdl_sync()  # PDL同步

        for i_split in T.serial(n_splits):  # 累加平方和
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)  # rsqrt归一化
        for j in T.Parallel(hc_mult3):  # 累加GEMM并归一化
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]  # 乘以rsqrt
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)  # 共享内存
        T.copy(mixes, mixes_shared)  # 拷贝到共享内存

        if T.get_thread_binding() < 32:  # 前32线程处理post和comb
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)  # 组合矩阵片段
            for j in T.Parallel(hc_mult):  # 计算后混合
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):  # 计算组合logits
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)  # 行求和
            col_sum = T.alloc_fragment(hc_mult, T.float32)  # 列求和

            row_max = T.alloc_fragment(hc_mult, T.float32)  # 行最大值
            T.reduce_max(cm, row_max, dim=1)  # 行最大值
            for j, k in T.Parallel(hc_mult, hc_mult):  # 减最大值取指数
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)  # 行求和
            for j, k in T.Parallel(hc_mult, hc_mult):  # 行归一化
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)  # 列求和
            for j, k in T.Parallel(hc_mult, hc_mult):  # 列归一化
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):  # Sinkhorn迭代
                T.reduce_sum(cm, row_sum, dim=1)  # 行求和
                for j, k in T.Parallel(hc_mult, hc_mult):  # 行归一化
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)  # 列求和
                for j, k in T.Parallel(hc_mult, hc_mult):  # 列归一化
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):  # 写入组合混合
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:  # 后64线程处理前混合和归一化层输入
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)  # 前混合共享内存
            for j in T.Parallel(hc_mult):  # 计算前混合
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )

            # Stash unnormalized weighted-sum output in shared memory as bf16
            # (matches the rounding the reference path does when RMSNorm reads bf16).
            # 将未归一化的加权求和输出以bf16暂存到共享内存
            # （匹配参考路径在RMSNorm读取bf16时的舍入行为）。
            output_shared = T.alloc_shared(hidden_size, T.bfloat16)  # 输出共享内存
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)  # 逐位置平方和
            T.clear(sumsq_per_pos)  # 清零

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=3):  # 流水线3阶段
                xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)  # 残差共享内存
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)  # 残差寄存器
                T.copy(residual[i, 0, i0_h * hidden_block], xs)  # 加载残差
                T.copy(xs, xl)  # 拷贝到寄存器

                ol = T.alloc_fragment(hidden_block, T.float32)  # 输出片段
                T.clear(ol)  # 清零

                for i_hc in T.serial(hc_mult):  # 加权求和
                    pre = pre_mix_shared[i_hc]  # 前混合系数
                    for i1_h in T.Parallel(hidden_block):  # 逐元素计算
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                for i1_h in T.Parallel(hidden_block):  # 累加平方和并暂存输出
                    sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]  # 累加平方
                    output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])  # 暂存bf16

            sumsq = T.alloc_fragment(1, T.float32)  # 总平方和
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)  # 归约
            rsqrt_norm = T.alloc_fragment(1, T.float32)  # rsqrt归一化因子
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)  # 计算rsqrt

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):  # 第二轮流水线
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)  # 权重共享内存
                w_local = T.alloc_fragment(hidden_block, T.float32)  # 权重寄存器
                T.copy(norm_weight[i0_h * hidden_block], w_shared)  # 加载权重
                T.copy(w_shared, w_local)  # 拷贝到寄存器

                ol = T.alloc_fragment(hidden_block, T.float32)  # 输出片段
                for i1_h in T.Parallel(hidden_block):  # 应用归一化
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]  # 乘以rsqrt
                        * w_local[i1_h]  # 乘以权重
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])  # 写入归一化结果

        if ENABLE_PDL:  # 如果支持PDL
            T.pdl_trigger()  # PDL触发


def mhc_pre(  # MHC前处理主机函数
    residual: torch.Tensor,  # 残差输入
    fn: torch.Tensor,  # 函数权重
    hc_scale: torch.Tensor,  # HC缩放因子
    hc_base: torch.Tensor,  # HC偏置基底
    rms_eps: float,  # RMS epsilon
    hc_pre_eps: float,  # HC前混合epsilon
    hc_sinkhorn_eps: float,  # HC Sinkhorn epsilon
    hc_post_mult_value: float,  # HC后混合乘数值
    sinkhorn_repeat: int,  # Sinkhorn重复次数
    n_splits: int = 1,  # 分割数
    n_splits_pre: int = 32,  # 预分割数
    *,  # 以下为仅关键字参数
    norm_weight: torch.Tensor | None = None,  # 归一化权重
    norm_eps: float | None = None,  # 归一化epsilon
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回后混合、组合混合和层输入

    assert residual.dtype == torch.bfloat16  # 断言残差为bf16
    assert fn.dtype == torch.float32  # 断言函数权重为fp32
    assert hc_scale.dtype == torch.float32  # 断言缩放因子为fp32
    assert hc_base.dtype == torch.float32  # 断言偏置基底为fp32

    hc_mult = residual.shape[-2]  # 获取HC乘数
    hidden_size = residual.shape[-1]  # 获取隐藏层大小
    hc_mult2 = hc_mult * hc_mult  # HC平方
    hc_mult3 = hc_mult * 2 + hc_mult2  # 混合维度

    hc_hidden_size = hc_mult * hidden_size  # HC隐藏层大小
    assert fn.shape[0] == hc_mult3  # 断言函数权重第一维正确
    assert fn.shape[1] == hc_hidden_size  # 断言函数权重第二维正确
    assert hc_scale.shape == (3,)  # 断言缩放因子形状正确
    assert hc_base.shape == (hc_mult3,)  # 断言偏置基底形状正确

    outer_shape = residual.shape[:-2]  # 获取外部形状

    residual_flat = residual.view(-1, hc_mult, hidden_size)  # 展平残差
    num_tokens = residual_flat.shape[0]  # 获取token数
    fn_flat = fn  # 函数权重（已是2D）

    post_mix = torch.empty(  # 分配后混合输出
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(  # 分配组合混合输出
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(  # 分配层输入输出
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():  # 如果启用DeepGEMM预归一化
        n_splits = _compute_num_split_for_mhc_pre(num_tokens, hc_hidden_size)  # 计算分割数

        gemm_out_mul = torch.empty(  # 分配GEMM乘积输出
            n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
        )
        gemm_out_sqrsum = torch.empty(  # 分配GEMM平方和输出
            n_splits, num_tokens, dtype=torch.float32, device=residual.device
        )

        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm  # 导入DeepGEMM

        tf32_hc_prenorm_gemm(  # 执行DeepGEMM预归一化GEMM
            residual_flat.view(num_tokens, hc_hidden_size),  # 展平残差
            fn_flat,  # 函数权重
            gemm_out_mul,  # 乘积输出
            gemm_out_sqrsum,  # 平方和输出
            n_splits,  # 分割数
        )
        gemm_last_dim = hc_mult3  # 设置GEMM最后维度
        big_fuse_n_splits = n_splits  # 设置大融合分割数
    else:  # 不使用DeepGEMM
        if num_tokens <= 2048:  # 小批量使用SplitK
            assert n_splits == 1  # 断言分割数为1
            if hc_hidden_size == 16384:  # 隐藏大小为16384
                hidden_block = 256  # 块大小256
            elif hc_hidden_size == 28672:  # 隐藏大小为28672
                hidden_block = 128  # 块大小128
            else:  # 不支持的隐藏大小
                raise NotImplementedError(  # 抛出未实现错误
                    f"mhc_pre splitk kernel only supports hc_hidden_size in {{16384, 28672}}, "
                    f"got {hc_hidden_size}"
                )
            kernel_0, _ = mhc_pre_gemm_sqrsum_splitk_kernel(  # 获取SplitK内核
                hc_mult3,
                hc_hidden_size,
                split_k=n_splits_pre,  # 预分割数
                token_block=32,
                hidden_block=hidden_block,
            )
            partial_out = torch.empty(  # 分配部分GEMM输出
                n_splits_pre,
                num_tokens,
                32,
                dtype=torch.float32,
                device=residual.device,
            )
            partial_sqrsum = torch.empty(  # 分配部分平方和
                n_splits_pre, num_tokens, dtype=torch.float32, device=residual.device
            )
            kernel_0(  # 启动第0阶段内核
                residual_flat.view(num_tokens, hc_hidden_size),
                fn_flat,
                partial_out,
                partial_sqrsum,
            )
            # Stage_1 reduction is folded into big_fuse below; skip launching it.
            # Stage_1归约已折叠到下面的big_fuse中；跳过启动。
            gemm_out_mul = partial_out  # 使用部分输出
            gemm_out_sqrsum = partial_sqrsum  # 使用部分平方和
            gemm_last_dim = 32  # GEMM最后维度为32
            big_fuse_n_splits = n_splits_pre  # 大融合分割数
        else:  # 大批量使用简单TileLang
            gemm_out_mul = torch.empty(  # 分配GEMM输出
                n_splits,
                num_tokens,
                hc_mult3,
                dtype=torch.float32,
                device=residual.device,
            )
            gemm_out_sqrsum = torch.empty(  # 分配平方和输出
                n_splits, num_tokens, dtype=torch.float32, device=residual.device
            )
            assert (  # 断言简单版本不支持SplitK
                n_splits == 1
            ), "The simple TileLang version gemm_sqrsum doesn't support split-k"
            mhc_pre_gemm_sqrsum_tilelang(  # 启动简单GEMM+平方和内核
                residual_flat.view(num_tokens, hc_mult * hidden_size),
                fn_flat,
                gemm_out_mul.squeeze(0),  # 去掉第0维
                gemm_out_sqrsum.squeeze(0),  # 去掉第0维
                hc_mult3,
                hc_mult * hidden_size,
            )
            gemm_last_dim = hc_mult3  # GEMM最后维度
            big_fuse_n_splits = n_splits  # 大融合分割数

    if norm_weight is not None:  # 如果有归一化权重
        assert norm_eps is not None, "norm_eps required when norm_weight is provided"  # 断言需要norm_eps
        assert norm_weight.shape == (  # 断言权重形状正确
            hidden_size,
        ), f"norm_weight shape {tuple(norm_weight.shape)} != (hidden_size={hidden_size},)"
        norm_weight_bf = (  # 转换为bf16
            norm_weight.bfloat16()
            if norm_weight.dtype != torch.bfloat16
            else norm_weight
        )
        if not norm_weight_bf.is_contiguous():  # 如果不连续
            norm_weight_bf = norm_weight_bf.contiguous()  # 使其连续
        mhc_pre_big_fuse_with_norm_tilelang(  # 启动带归一化的大融合内核
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight_bf,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            big_fuse_n_splits,
            hc_mult,
            gemm_last_dim,
        )
    else:  # 没有归一化权重
        mhc_pre_big_fuse_tilelang(  # 启动大融合内核（无归一化）
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            big_fuse_n_splits,
            hc_mult,
            gemm_last_dim,
        )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)  # 重塑后混合形状
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)  # 重塑组合混合形状
    layer_input = layer_input.view(*outer_shape, hidden_size)  # 重塑层输入形状

    return post_mix, comb_mix, layer_input  # 返回后混合、组合混合和层输入


@tilelang.jit(  # TileLang JIT编译
    pass_configs={  # 通道配置
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,  # 禁用warp特化
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,  # 禁用TMA降低
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,  # PTXAS寄存器使用级别
    },
)
def mhc_post_tilelang(  # MHC后处理TileLang内核
    a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
) -> tilelang.JITKernel:  # 返回JIT内核
    n = T.dynamic("num_tokens")  # 动态token数量
    h = hidden  # 隐藏层大小

    h_blk = math.gcd(hidden, h_blk)  # 计算隐藏块大小
    a: T.Tensor((n, hc, hc), T.float32)  # 组合混合系数
    b: T.Tensor((n, hc, h), T.bfloat16)  # 残差输入
    c: T.Tensor((n, hc), T.float32)  # 后混合系数
    d: T.Tensor((n, h), T.bfloat16)  # 隐藏输入
    x: T.Tensor((n, h), T.bfloat16)  # 输出

    ENABLE_PDL = is_arch_support_pdl()  # 检测PDL支持
    with T.Kernel(n, threads=n_thr) as i_n:  # 启动n个CTA
        if ENABLE_PDL:  # 如果支持PDL
            T.pdl_sync()  # PDL同步

        x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)  # x共享内存
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)  # b共享内存
        d_shared = T.alloc_shared(h_blk, T.bfloat16)  # d共享内存

        x_local = T.alloc_fragment((hc, h_blk), T.float32)  # x寄存器
        b_local = T.alloc_fragment((hc, h_blk), T.float32)  # b寄存器
        d_local = T.alloc_fragment(h_blk, T.float32)  # d寄存器

        a_local = T.alloc_fragment((hc, hc), T.float32)  # a系数寄存器
        c_local = T.alloc_fragment(hc, T.float32)  # c系数寄存器
        T.copy(a[i_n, 0, 0], a_local)  # 加载a
        T.copy(c[i_n, 0], c_local)  # 加载c

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):  # 流水线处理
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)  # 加载b
            T.copy(d[i_n, i0_h * h_blk], d_shared)  # 加载d

            T.copy(b_shared, b_local)  # 拷贝到寄存器
            T.copy(d_shared, d_local)  # 拷贝到寄存器
            for i_hco, i1_h in T.Parallel(hc, h_blk):  # 计算后处理
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]  # 后混合*隐藏输入
                for i_hci in T.serial(hc):  # 组合混合累加
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
            T.copy(x_local, x_shared)  # 拷贝到共享内存

            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])  # 写入输出

        if ENABLE_PDL:  # 如果支持PDL
            T.pdl_trigger()  # PDL触发


def mhc_post(  # MHC后处理主机函数
    x: torch.Tensor,  # 隐藏输入
    residual: torch.Tensor,  # 残差输入
    post_layer_mix: torch.Tensor,  # 后混合系数
    comb_res_mix: torch.Tensor,  # 组合混合系数
) -> torch.Tensor:  # 返回输出张量
    if is_dsa_prefill_cp_round_robin_split():  # 如果是DSA预填充循环分割
        x = strict_contiguous(x)  # 确保连续
        residual = strict_contiguous(residual)  # 确保连续
        post_layer_mix = strict_contiguous(post_layer_mix)  # 确保连续
        comb_res_mix = strict_contiguous(comb_res_mix)  # 确保连续
    out = torch.empty_like(residual)  # 分配输出
    mhc_post_tilelang(  # 启动后处理内核
        comb_res_mix,  # 组合混合
        residual,  # 残差
        post_layer_mix.squeeze(-1),  # 后混合（去掉最后一维）
        x,  # 隐藏输入
        out,  # 输出
        residual.shape[-2],  # HC乘数
        residual.shape[-1],  # 隐藏层大小
    )
    return out  # 返回输出


@tilelang.jit(  # TileLang JIT编译
    pass_configs={  # 通道配置
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,  # 禁用warp特化
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,  # 禁用TMA降低
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,  # PTXAS寄存器使用级别
    },
)
def mhc_fused_post_pre_fma_tilelang(  # MHC融合后处理+前处理FMA内核
    prev_comb_mix,  # 前一层组合混合
    prev_residual,  # 前一层残差
    prev_post_mix,  # 前一层后混合
    hidden_in,  # 隐藏输入
    pre_fn,  # 前处理函数权重
    mixes_partial_out,  # 混合部分输出
    sqrsum_partial_out,  # 平方和部分输出
    cur_residual_out,  # 当前残差输出
    hc: int,  # HC乘数
    hidden_size: int,  # 隐藏层大小
    num_mix_outputs: int,  # 混合输出数
    n_thr: int = 256,  # 线程数
    tile_mix_outputs: int = 1,  # 混合输出块大小
    split_k: int = 1,  # SplitK因子
) -> tilelang.JITKernel:  # 返回JIT内核
    num_tokens = T.dynamic("num_tokens")  # 动态token数量
    split_k = T.dynamic("split_k")  # 动态SplitK因子

    hidden_per_split = (hidden_size + split_k - 1) // split_k  # 每个分割的隐藏大小
    num_mix_output_tiles = (num_mix_outputs + tile_mix_outputs - 1) // tile_mix_outputs  # 混合输出块数

    prev_comb_mix: T.Tensor((num_tokens, hc, hc), T.float32)  # 前一层组合混合
    prev_residual: T.Tensor((num_tokens, hc, hidden_size), T.bfloat16)  # 前一层残差
    prev_post_mix: T.Tensor((num_tokens, hc), T.float32)  # 前一层后混合
    hidden_in: T.Tensor((num_tokens, hidden_size), T.bfloat16)  # 隐藏输入
    pre_fn: T.Tensor((num_mix_outputs, hc, hidden_size), T.float32)  # 前处理函数权重

    mixes_partial_out: T.Tensor((split_k, num_tokens, num_mix_outputs), T.float32)  # 混合部分输出
    sqrsum_partial_out: T.Tensor((split_k, num_tokens), T.float32)  # 平方和部分输出
    cur_residual_out: T.Tensor((num_tokens, hc, hidden_size), T.bfloat16)  # 当前残差输出

    hidden_iters_per_thread = (hidden_per_split + n_thr - 1) // n_thr  # 每线程隐藏迭代数
    num_warps = n_thr // 32  # warp数

    ENABLE_PDL = is_arch_support_pdl()  # 检测PDL支持

    # CTA assignment:
    #   token_idx           : this CTA handles one token.
    #   mix_output_tile_idx : this CTA handles a small tile of mix output columns.
    #                          For HC=4, num_mix_outputs = 24:
    #                            [0:4]   -> pre logits
    #                            [4:8]   -> post logits
    #                            [8:24]  -> comb logits
    #   hidden_split_idx    : this CTA handles one split of the hidden dimension.
    # CTA分配：
    #   token_idx           : 此CTA处理一个token。
    #   mix_output_tile_idx : 此CTA处理混合输出列的一小块。
    #                          对于HC=4，num_mix_outputs = 24：
    #                            [0:4]   -> 前logits
    #                            [4:8]   -> 后logits
    #                            [8:24]  -> 组合logits
    #   hidden_split_idx    : 此CTA处理隐藏维度的一个分割。
    #
    # Thread assignment inside one CTA:
    #   Each thread owns several hidden positions in this hidden split:
    #     hidden_idx = hidden_split_start + hidden_iter * n_thr + thread_idx
    # 一个CTA内的线程分配：
    #   每个线程拥有此隐藏分割中的若干隐藏位置：
    #     hidden_idx = hidden_split_start + hidden_iter * n_thr + thread_idx
    #
    # For each owned hidden_idx, the thread computes:
    #   1. post result: cur_residual[token, :, hidden_idx]
    #   2. sqrsum partial for pre RMS
    #   3. GEMM partial for several mix output columns
    # 对于每个拥有的hidden_idx，线程计算：
    #   1. 后处理结果：cur_residual[token, :, hidden_idx]
    #   2. 前RMS的平方和部分
    #   3. 若干混合输出列的GEMM部分
    with T.Kernel(  # 启动3D内核
        num_tokens,  # token维度
        num_mix_output_tiles,  # 混合输出块维度
        split_k,  # 隐藏分割维度
        threads=n_thr,  # 线程数
    ) as (token_idx, mix_output_tile_idx, hidden_split_idx):
        thread_idx = T.get_thread_binding()  # 线程索引
        warp_idx = T.get_warp_idx()  # warp索引
        lane_idx = T.get_lane_idx()  # lane索引

        warp_partials = T.alloc_shared((num_warps, tile_mix_outputs + 1), T.float32)  # warp部分结果
        post_mix_smem = T.alloc_shared((hc,), T.float32)  # 后混合共享内存
        comb_mix_smem = T.alloc_shared((hc, hc), T.float32)  # 组合混合共享内存

        post_mix_for_token = T.alloc_local((hc,), T.float32)  # 后混合本地存储
        comb_mix_for_token = T.alloc_local((hc, hc), T.float32)  # 组合混合本地存储

        mix_acc = T.alloc_local((tile_mix_outputs,), T.float32)  # 混合累加器
        sqrsum_acc = T.alloc_local((1,), T.float32)  # 平方和累加器
        cur_residual_values = T.alloc_local((hc,), T.float32)  # 当前残差值

        T.clear(mix_acc)  # 清零混合累加器
        T.clear(sqrsum_acc)  # 清零平方和累加器

        hidden_split_start = hidden_split_idx * hidden_per_split  # 隐藏分割起始位置

        if ENABLE_PDL:  # 如果支持PDL
            T.pdl_sync()  # PDL同步

        # Load post/comb coefficients for this token.
        # 加载此token的后/组合系数。
        #
        # PyTorch equivalent:
        #   post = prev_post_mix[token_idx]      # [HC]
        #   comb = prev_comb_mix[token_idx]      # [HC, HC]
        # PyTorch等价：
        #   post = prev_post_mix[token_idx]      # [HC]
        #   comb = prev_comb_mix[token_idx]      # [HC, HC]
        T.copy(prev_post_mix[token_idx, 0], post_mix_smem)  # 加载后混合
        T.copy(prev_comb_mix[token_idx, 0, 0], comb_mix_smem)  # 加载组合混合

        for route_idx in T.unroll(hc):  # 拷贝后混合到本地
            post_mix_for_token[route_idx] = post_mix_smem[route_idx]

        for old_route_idx in T.unroll(hc):  # 拷贝组合混合到本地
            for new_route_idx in T.unroll(hc):
                comb_mix_for_token[old_route_idx, new_route_idx] = comb_mix_smem[
                    old_route_idx, new_route_idx
                ]

        for hidden_iter in T.serial(hidden_iters_per_thread):  # 遍历隐藏位置
            hidden_idx = hidden_split_start + hidden_iter * n_thr + thread_idx  # 计算隐藏索引

            if hidden_idx < hidden_size:  # 边界检查
                # Step A: fused post.
                # 步骤A：融合后处理。
                #
                # PyTorch equivalent:
                #   cur_residual =
                #       post.unsqueeze(-1) * hidden_in.unsqueeze(1)
                #       + (
                #           comb.unsqueeze(-1)
                #           * prev_residual.unsqueeze(2)
                #         ).sum(dim=1)
                # PyTorch等价：
                #   cur_residual =
                #       post.unsqueeze(-1) * hidden_in.unsqueeze(1)
                #       + (
                #           comb.unsqueeze(-1)
                #           * prev_residual.unsqueeze(2)
                #         ).sum(dim=1)
                #
                # Scalar form for this token and this hidden position:
                #   cur_residual[j, h]
                #     = post[j] * hidden_in[h]
                #     + sum_k comb[k, j] * prev_residual[k, h]
                # 此token和此隐藏位置的标量形式：
                #   cur_residual[j, h]
                #     = post[j] * hidden_in[h]
                #     + sum_k comb[k, j] * prev_residual[k, h]
                for new_route_idx in T.unroll(hc):  # 计算后混合部分
                    cur_residual_values[new_route_idx] = (
                        post_mix_for_token[new_route_idx]
                        * hidden_in[token_idx, hidden_idx]
                    )

                    for old_route_idx in T.unroll(hc):  # 计算组合混合部分
                        cur_residual_values[new_route_idx] += (
                            comb_mix_for_token[old_route_idx, new_route_idx]
                            * prev_residual[token_idx, old_route_idx, hidden_idx]
                        )

                # Match the unfused path:
                #   mhc_post writes bf16 residual,
                #   then mhc_pre reads bf16 residual.
                # 匹配未融合路径：
                #   mhc_post写入bf16残差，
                #   然后mhc_pre读取bf16残差。
                for route_idx in T.unroll(hc):  # 转换为bf16
                    cur_residual_values[route_idx] = T.bfloat16(
                        cur_residual_values[route_idx]
                    )

                # Step B1: pre sqrsum partial.
                # 步骤B1：前处理平方和部分。
                #
                # PyTorch equivalent:
                #   x_flat = cur_residual.reshape(T, HC * H).float()
                #   sqrsum = (x_flat * x_flat).sum(dim=-1)
                # PyTorch等价：
                #   x_flat = cur_residual.reshape(T, HC * H).float()
                #   sqrsum = (x_flat * x_flat).sum(dim=-1)
                #
                # Only mix_output_tile_idx == 0 writes cur_residual and sqrsum,
                # otherwise different output-column CTAs would duplicate this work.
                # 仅mix_output_tile_idx == 0写入cur_residual和sqrsum，
                # 否则不同输出列的CTA会重复此工作。
                if mix_output_tile_idx == 0:  # 只在第一个混合输出块写入
                    for route_idx in T.unroll(hc):  # 写入残差和平方和
                        cur_residual_out[token_idx, route_idx, hidden_idx] = (
                            cur_residual_values[route_idx]
                        )
                        sqrsum_acc[0] += (
                            cur_residual_values[route_idx]
                            * cur_residual_values[route_idx]
                        )

                # Step B2: pre GEMM partial.
                # 步骤B2：前处理GEMM部分。
                #
                # PyTorch equivalent:
                #   mixes = F.linear(x_flat, fn)
                # PyTorch等价：
                #   mixes = F.linear(x_flat, fn)
                #
                # Scalar form:
                #   mixes[token, o] +=
                #       pre_fn[o, route, hidden] * cur_residual[route, hidden]
                # 标量形式：
                #   mixes[token, o] +=
                #       pre_fn[o, route, hidden] * cur_residual[route, hidden]
                #
                # This CTA computes only tile_mix_outputs columns of mixes.
                # 此CTA仅计算mixes的tile_mix_outputs列。
                for tile_col_idx in T.unroll(tile_mix_outputs):  # 遍历混合输出列
                    mix_output_idx = (
                        mix_output_tile_idx * tile_mix_outputs + tile_col_idx
                    )

                    if mix_output_idx < num_mix_outputs:  # 边界检查
                        for route_idx in T.unroll(hc):  # GEMM部分累加
                            mix_acc[tile_col_idx] += (
                                pre_fn[mix_output_idx, route_idx, hidden_idx]
                                * cur_residual_values[route_idx]
                            )

        # Reduce thread partials inside each warp.
        # 在每个warp内归约线程部分结果。
        for tile_col_idx in T.unroll(tile_mix_outputs):  # warp内归约混合
            mix_acc[tile_col_idx] = T.warp_reduce_sum(mix_acc[tile_col_idx])

        if mix_output_tile_idx == 0:  # warp内归约平方和
            sqrsum_acc[0] = T.warp_reduce_sum(sqrsum_acc[0])

        # One lane per warp writes warp-level partials to shared memory.
        # 每个warp一个lane将warp级部分结果写入共享内存。
        if lane_idx == 0:  # 每个warp的第0个lane写入
            for tile_col_idx in T.unroll(tile_mix_outputs):  # 写入混合部分结果
                warp_partials[warp_idx, tile_col_idx] = mix_acc[tile_col_idx]

            if mix_output_tile_idx == 0:  # 写入平方和部分结果
                warp_partials[warp_idx, tile_mix_outputs] = sqrsum_acc[0]

        T.sync_threads()  # 同步线程

        # Reduce across warps and write split partials.
        # 跨warp归约并写入分割部分结果。
        #
        # The full PyTorch result would be:
        #   mixes = F.linear(cur_residual.reshape(T, HC * H), fn)
        #   sqrsum = (cur_residual.float() ** 2).sum(dim=(1, 2))
        # 完整PyTorch结果为：
        #   mixes = F.linear(cur_residual.reshape(T, HC * H), fn)
        #   sqrsum = (cur_residual.float() ** 2).sum(dim=(1, 2))
        #
        # This kernel is split along hidden, so each CTA writes only:
        #   mixes_partial_out[hidden_split_idx, token, o]
        #   sqrsum_partial_out[hidden_split_idx, token]
        # 此内核沿隐藏维度分割，因此每个CTA仅写入：
        #   mixes_partial_out[hidden_split_idx, token, o]
        #   sqrsum_partial_out[hidden_split_idx, token]
        #
        # Later mhc_pre_big_fuse does:
        #   mixes = mixes_partial_out.sum(dim=0)
        #   sqrsum = sqrsum_partial_out.sum(dim=0)
        #   rms = rsqrt(sqrsum / (HC * H) + eps)
        #   mixes *= rms
        #   mixes -> pre/post/comb
        #   layer_input = sum_j pre[j] * cur_residual[j]
        # 随后mhc_pre_big_fuse执行：
        #   mixes = mixes_partial_out.sum(dim=0)
        #   sqrsum = sqrsum_partial_out.sum(dim=0)
        #   rms = rsqrt(sqrsum / (HC * H) + eps)
        #   mixes *= rms
        #   mixes -> pre/post/comb
        #   layer_input = sum_j pre[j] * cur_residual[j]
        if warp_idx == 0:  # 由warp 0负责跨warp归约
            for tile_col_idx in T.unroll(tile_mix_outputs):  # 归约混合
                mix_output_idx = mix_output_tile_idx * tile_mix_outputs + tile_col_idx

                if mix_output_idx < num_mix_outputs and lane_idx == tile_col_idx:  # 边界检查和lane分配
                    mix_output_partial = T.alloc_var(T.float32, init=0.0)  # 分配部分结果变量

                    for reduce_warp_idx in T.unroll(num_warps):  # 跨warp累加
                        mix_output_partial += warp_partials[
                            reduce_warp_idx, tile_col_idx
                        ]

                    mixes_partial_out[hidden_split_idx, token_idx, mix_output_idx] = (
                        mix_output_partial  # 写入最终部分结果
                    )

            if mix_output_tile_idx == 0 and lane_idx == 0:  # 归约平方和
                sqrsum_partial = T.alloc_var(T.float32, init=0.0)  # 分配部分结果

                for reduce_warp_idx in T.unroll(num_warps):  # 跨warp累加
                    sqrsum_partial += warp_partials[reduce_warp_idx, tile_mix_outputs]

                sqrsum_partial_out[hidden_split_idx, token_idx] = sqrsum_partial  # 写入平方和

        if ENABLE_PDL:  # 如果支持PDL
            T.pdl_trigger()  # PDL触发


def mhc_fused_post_pre(  # MHC融合后处理+前处理主机函数
    x: torch.Tensor,  # 隐藏输入
    residual: torch.Tensor,  # 残差输入
    post_layer_mix: torch.Tensor,  # 后混合系数
    comb_res_mix: torch.Tensor,  # 组合混合系数
    fn: torch.Tensor,  # 函数权重
    hc_scale: torch.Tensor,  # HC缩放因子
    hc_base: torch.Tensor,  # HC偏置基底
    rms_eps: float,  # RMS epsilon
    hc_pre_eps: float,  # HC前混合epsilon
    hc_sinkhorn_eps: float,  # HC Sinkhorn epsilon
    hc_post_mult_value: float,  # HC后混合乘数值
    sinkhorn_repeat: int,  # Sinkhorn重复次数
    n_splits: int = 1,  # 分割数
    tile_n: int = 1,  # 混合输出块大小
    *,  # 以下为仅关键字参数
    norm_weight: torch.Tensor | None = None,  # 归一化权重
    norm_eps: float | None = None,  # 归一化epsilon
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回4个张量
    """Fuse the boundary between one mHC post step and the next mHC pre step.
    融合一个MHC后处理步骤与下一个MHC前处理步骤之间的边界。

    The unfused sequence is ``mhc_post -> pre-norm GEMM -> mhc_pre big_fuse``.
    This wrapper keeps the numerically sensitive ``mhc_pre_big_fuse`` stage,
    including optional RMSNorm, but removes the separate post/pre boundary.
    Small token batches use the FMA kernel above to combine ``mhc_post`` and the
    pre-norm GEMM in one launch; larger batches keep DeepGEMM for throughput and
    only fuse the Python/model-level scheduling boundary.
    未融合序列为``mhc_post -> pre-norm GEMM -> mhc_pre big_fuse``。
    此包装器保留数值敏感的``mhc_pre_big_fuse``阶段（包括可选的RMSNorm），
    但移除了独立的后处理/前处理边界。小token批量使用上面的FMA内核将``mhc_post``
    和预归一化GEMM合并为一次启动；大批量保持DeepGEMM以获取吞吐量，
    仅融合Python/模型级别的调度边界。

    Returns:
        residual_cur: post-mapped residual, shape (..., hc_mult, hidden_size)
        residual_cur: 后映射残差，形状(..., hc_mult, hidden_size)
        post_mix_cur: shape (..., hc_mult, 1)
        post_mix_cur: 形状(..., hc_mult, 1)
        comb_mix_cur: shape (..., hc_mult, hc_mult)
        comb_mix_cur: 形状(..., hc_mult, hc_mult)
        layer_input_cur: shape (..., hidden_size)
        layer_input_cur: 形状(..., hidden_size)
    """

    assert residual.dtype == torch.bfloat16  # 断言残差为bf16
    assert x.dtype == torch.bfloat16  # 断言x为bf16
    assert post_layer_mix.dtype == torch.float32  # 断言后混合为fp32
    assert comb_res_mix.dtype == torch.float32  # 断言组合混合为fp32
    assert fn.dtype == torch.float32  # 断言函数权重为fp32
    assert hc_scale.dtype == torch.float32  # 断言缩放因子为fp32
    assert hc_base.dtype == torch.float32  # 断言偏置基底为fp32

    hc_mult = residual.shape[-2]  # 获取HC乘数
    hidden_size = residual.shape[-1]  # 获取隐藏层大小
    hc_mult2 = hc_mult * hc_mult  # HC平方
    hc_mult3 = hc_mult * 2 + hc_mult2  # 混合维度
    hc_hidden_size = hc_mult * hidden_size  # HC隐藏层大小
    outer_shape = residual.shape[:-2]  # 外部形状

    assert x.shape == (*outer_shape, hidden_size)  # 断言x形状正确
    assert post_layer_mix.shape in (  # 断言后混合形状正确
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)  # 断言组合混合形状正确
    assert fn.shape == (hc_mult3, hc_hidden_size)  # 断言函数权重形状正确
    assert hc_scale.shape == (3,)  # 断言缩放因子形状正确
    assert hc_base.shape == (hc_mult3,)  # 断言偏置基底形状正确

    residual_flat = residual.view(-1, hc_mult, hidden_size)  # 展平残差
    num_tokens = residual_flat.shape[0]  # 获取token数
    if num_tokens == 0:  # 如果没有token
        # Some DP/EP ranks can receive no tokens; return correctly typed empty
        # tensors so later fused layers keep the same contracts as mhc_pre/hc_post.
        # 某些DP/EP rank可能不接收token；返回正确类型的空张量，
        # 使后续融合层保持与mhc_pre/hc_post相同的约定。
        return (
            torch.empty_like(residual),  # 空残差
            torch.empty(  # 空后混合
                (*outer_shape, hc_mult, 1), dtype=torch.float32, device=residual.device
            ),
            torch.empty(  # 空组合混合
                (*outer_shape, hc_mult, hc_mult),
                dtype=torch.float32,
                device=residual.device,
            ),
            torch.empty(  # 空层输入
                (*outer_shape, hidden_size),
                dtype=torch.bfloat16,
                device=residual.device,
            ),
        )
    x_flat = x.view(num_tokens, hidden_size)  # 展平x

    # The scalar-FMA kernel wins only for small batches where launch
    # overhead dominates; beyond the threshold DeepGEMM's tensor-core path wins.
    # 标量FMA内核仅在小批量（启动开销占主导）时胜出；
    # 超过阈值后DeepGEMM的张量核心路径胜出。
    fma_token_threshold = 32  # FMA阈值
    if num_tokens <= fma_token_threshold:  # 小批量
        tile_n = 2 if num_tokens < 8 else 3  # 根据token数选择块大小
        n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4  # 根据token数和隐藏大小选择分割数
    else:  # 大批量
        n_splits = _compute_num_split_for_mhc_pre(num_tokens, hc_hidden_size)  # 计算分割数

    gemm_out_mul = torch.empty(  # 分配GEMM乘积输出
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(  # 分配GEMM平方和输出
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual_flat)  # 分配当前残差输出

    if num_tokens <= fma_token_threshold:  # 小批量路径
        # Small-batch path: one TileLang launch computes hc_post, the bf16
        # residual write, GEMM partials, and the RMS square-sum partials.
        # 小批量路径：一次TileLang启动计算hc_post、bf16残差写入、GEMM部分和RMS平方和部分。
        mhc_fused_post_pre_fma_tilelang(  # 启动FMA融合内核
            comb_res_mix.view(num_tokens, hc_mult, hc_mult),
            residual_flat,
            post_layer_mix.view(num_tokens, hc_mult),
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_cur,
            hc_mult,
            hidden_size,
            hc_mult3,
            tile_mix_outputs=tile_n,
            split_k=n_splits,
        )
    else:  # 大批量路径
        # Large-batch path: keep the existing high-throughput TileLang hc_post +
        # DeepGEMM pre-norm GEMM decomposition instead of replacing tensor cores.
        # 大批量路径：保持现有高吞吐TileLang hc_post + DeepGEMM预归一化GEMM分解，
        # 而非替换张量核心。
        mhc_post_tilelang(  # 启动后处理内核
            comb_res_mix.view(num_tokens, hc_mult, hc_mult),
            residual_flat,
            post_layer_mix.view(num_tokens, hc_mult),
            x_flat,
            residual_cur,
            hc_mult,
            hidden_size,
        )

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():  # 如果启用DeepGEMM预归一化
            import deep_gemm  # 导入DeepGEMM

            deep_gemm.tf32_hc_prenorm_gemm(  # 执行DeepGEMM预归一化GEMM
                residual_cur.view(num_tokens, hc_hidden_size),
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                num_splits=n_splits,
            )
        else:  # 不使用DeepGEMM
            # Fallback mirrors mhc_pre when DeepGEMM prenorm is disabled.
            # 回退方案，镜像DeepGEMM预归一化禁用时的mhc_pre。
            n_splits = 1  # 重置分割数为1
            gemm_out_mul_2d = torch.empty(  # 分配2D GEMM输出
                num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
            )
            gemm_out_sqrsum_1d = torch.empty(  # 分配1D平方和输出
                num_tokens, dtype=torch.float32, device=residual.device
            )
            mhc_pre_gemm_sqrsum_tilelang(  # 启动简单GEMM+平方和内核
                residual_cur.view(num_tokens, hc_hidden_size),
                fn,
                gemm_out_mul_2d,
                gemm_out_sqrsum_1d,
                hc_mult3,
                hc_hidden_size,
            )
            gemm_out_mul = gemm_out_mul_2d.unsqueeze(0)  # 增加第0维
            gemm_out_sqrsum = gemm_out_sqrsum_1d.unsqueeze(0)  # 增加第0维

    post_mix_cur = torch.empty(  # 分配当前后混合输出
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(  # 分配当前组合混合输出
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(  # 分配当前层输入输出
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    if norm_weight is not None:  # 如果有归一化权重
        # Final mhc_pre stage: convert GEMM partials into post/comb/layer_input
        # and fuse the following RMSNorm when the model passed a norm weight.
        # 最终mhc_pre阶段：将GEMM部分结果转换为post/comb/layer_input，
        # 并在模型提供归一化权重时融合后续的RMSNorm。
        assert norm_eps is not None  # 断言需要norm_eps
        assert norm_weight.shape == (hidden_size,)  # 断言权重形状正确
        norm_weight_bf = (  # 转换为bf16
            norm_weight.bfloat16()
            if norm_weight.dtype != torch.bfloat16
            else norm_weight
        )
        if not norm_weight_bf.is_contiguous():  # 如果不连续
            norm_weight_bf = norm_weight_bf.contiguous()  # 使其连续
        mhc_pre_big_fuse_with_norm_tilelang(  # 启动带归一化的大融合内核
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            norm_weight_bf,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
            hc_mult3,
        )
    else:  # 没有归一化权重
        # Same mhc_pre finalization without the model-layer RMSNorm.
        # 相同的mhc_pre最终化，但不包含模型层RMSNorm。
        mhc_pre_big_fuse_tilelang(  # 启动大融合内核（无归一化）
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
            hc_mult3,
        )

    return (  # 返回4个张量
        residual_cur.view(*outer_shape, hc_mult, hidden_size),  # 当前残差
        post_mix_cur.view(*outer_shape, hc_mult, 1),  # 当前后混合
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),  # 当前组合混合
        layer_input_cur.view(*outer_shape, hidden_size),  # 当前层输入
    )
