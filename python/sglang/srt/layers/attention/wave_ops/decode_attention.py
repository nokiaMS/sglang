# Wave解码注意力模块，基于Wave语言实现的高效解码注意力计算，支持page_size=1
"""
Memory-efficient attention for decoding.  # 解码阶段的高效内存注意力
It supports page size = 1.  # 支持页大小为1
"""

import functools  # 导入functools模块，用于缓存装饰器
import logging  # 导入日志模块

from wave_lang.kernel.lang.global_symbols import *  # 导入Wave语言全局符号
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile  # 导入Wave编译选项和编译函数
from wave_lang.kernel.wave.constraints import GenericDot, MMAOperand, MMAType  # 导入矩阵乘法约束相关类型
from wave_lang.kernel.wave.templates.paged_decode_attention import (  # 导入分页解码注意力模板
    get_paged_decode_attention_kernels,  # 获取分页解码注意力核函数
    get_paged_decode_intermediate_arrays_shapes,  # 获取分页解码中间数组形状
    paged_decode_attention_shape,  # 分页解码注意力形状
)
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params  # 导入默认调度参数
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config  # 导入默认运行配置设置

logger = logging.getLogger(__name__)  # 创建日志记录器
import os  # 导入操作系统模块

dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))  # 是否导出生成的MLIR，默认为0


# 获取或编译Wave解码注意力核函数（带LRU缓存，最大缓存4096个）
@functools.lru_cache(maxsize=4096)  # 使用LRU缓存，最大缓存4096项
def get_wave_kernel(
    shape: paged_decode_attention_shape,  # 分页解码注意力形状
    max_kv_splits,  # 最大KV分片数
    input_dtype,  # 输入数据类型
    output_dtype,  # 输出数据类型
    logit_cap,  # logits上限值
):
    mha = (shape.num_query_heads // shape.num_kv_heads) == 1  # 判断是否为多头注意力（查询头数等于键值头数）

    # Get the kernels (either compile or load from cache).  # 获取核函数（编译或从缓存加载）
    if mha:  # 如果是MHA模式
        mfma_variant = (  # 设置MHA的MFMA变体
            GenericDot(along_dim=MMAOperand.M, k_vec_size=4, k_mult=1),  # 沿M维度的通用点积，k向量大小4，k乘数1
            GenericDot(along_dim=MMAOperand.M, k_vec_size=1, k_mult=64),  # 沿M维度的通用点积，k向量大小1，k乘数64
        )
    else:  # 如果是GQA/MQA模式
        mfma_variant = (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)  # 设置GQA/MQA的MFMA变体

    (
        phase_0,  # 阶段0核函数（QK计算）
        phase_1,  # 阶段1核函数（SV计算）
        hyperparams_0,  # 阶段0超参数
        hyperparams_1,  # 阶段1超参数
        dynamic_symbols_0,  # 阶段0动态符号
        dynamic_symbols_1,  # 阶段1动态符号
    ) = get_paged_decode_attention_kernels(  # 获取分页解码注意力核函数
        shape,  # 注意力形状
        mfma_variant,  # MFMA变体
        max_kv_splits,  # 最大KV分片数
        input_dtype=input_dtype,  # 输入数据类型
        output_dtype=output_dtype,  # 输出数据类型
        logit_cap=logit_cap,  # logits上限
    )
    hyperparams_0.update(get_default_scheduling_params())  # 用默认调度参数更新阶段0超参数
    hyperparams_1.update(get_default_scheduling_params())  # 用默认调度参数更新阶段1超参数

    options = WaveCompileOptions(  # 创建阶段0编译选项
        subs=hyperparams_0,  # 超参数替换
        canonicalize=True,  # 启用规范化
        run_bench=False,  # 不运行基准测试
        use_buffer_ops=True,  # 使用缓冲区操作
        waves_per_eu=2,  # 每个执行单元的wave数
        dynamic_symbols=dynamic_symbols_0,  # 动态符号
        wave_runtime=True,  # 启用Wave运行时
    )
    options = set_default_run_config(options)  # 设置默认运行配置
    phase_0 = wave_compile(options, phase_0)  # 编译阶段0核函数

    options = WaveCompileOptions(  # 创建阶段1编译选项
        subs=hyperparams_1,  # 超参数替换
        canonicalize=True,  # 启用规范化
        run_bench=False,  # 不运行基准测试
        use_buffer_ops=False,  # 不使用缓冲区操作
        waves_per_eu=4,  # 每个执行单元的wave数
        dynamic_symbols=dynamic_symbols_1,  # 动态符号
        wave_runtime=True,  # 启用Wave运行时
    )
    options = set_default_run_config(options)  # 设置默认运行配置
    phase_1 = wave_compile(options, phase_1)  # 编译阶段1核函数

    return phase_0, phase_1  # 返回阶段0和阶段1的编译核函数


# 获取解码注意力中间数组的形状
def decode_attention_intermediate_arrays_shapes(
    num_seqs, head_size_kv, num_query_heads, max_kv_splits  # num_seqs: 序列数, head_size_kv: 键值头维度, num_query_heads: 查询头数, max_kv_splits: 最大KV分片数
):
    # Not all fields are used, but we need to pass them to the function  # 并非所有字段都使用，但需要传递给函数
    shape = paged_decode_attention_shape(  # 构建分页解码注意力形状
        num_query_heads=num_query_heads,  # 查询头数
        num_kv_heads=0,  # 键值头数设为0（不使用）
        head_size=0,  # 头维度设为0（不使用）
        head_size_kv=head_size_kv,  # 键值头维度
        block_size=0,  # 块大小设为0（不使用）
        num_seqs=num_seqs,  # 序列数
    )
    return get_paged_decode_intermediate_arrays_shapes(shape, max_kv_splits)  # 返回中间数组形状


# 执行Wave解码注意力计算
def decode_attention_wave(
    q,  # 查询张量
    k_buffer,  # 键缓存缓冲区
    v_buffer,  # 值缓存缓冲区
    o,  # 输出张量
    b_req_idx,  # 批次请求索引
    req_to_token,  # 请求到token的映射
    attn_logits,  # 注意力logits
    attn_logits_max,  # 注意力logits最大值
    num_kv_splits,  # KV分片数
    max_kv_splits,  # 最大KV分片数
    sm_scale,  # softmax缩放因子
    logit_cap,  # logits上限值
):
    num_seqs, num_query_heads, head_size = q.shape  # 获取查询张量的形状信息
    _, num_kv_heads, _ = k_buffer.shape  # 获取键缓存的头数信息
    _, _, head_size_kv = v_buffer.shape  # 获取值缓存的头维度信息
    block_size = 32  # 设置块大小为32
    shape = paged_decode_attention_shape(  # 构建分页解码注意力形状
        num_query_heads,  # 查询头数
        num_kv_heads,  # 键值头数
        head_size,  # 头维度
        head_size_kv,  # 键值头维度
        block_size,  # 块大小
        num_seqs,  # 序列数
    )

    phase_0, phase_1 = get_wave_kernel(  # 获取阶段0和阶段1的核函数
        shape, max_kv_splits, q.dtype, o.dtype, logit_cap  # 传入形状、分片数、数据类型和上限
    )

    mb_qk = phase_0(  # 执行阶段0（QK点积计算）
        q,  # 查询张量
        k_buffer,  # 键缓存
        v_buffer,  # 值缓存
        b_req_idx,  # 批次请求索引
        req_to_token,  # 请求到token映射
        attn_logits,  # 注意力logits输出
        attn_logits_max,  # 注意力logits最大值输出
    )
    if dump_generated_mlir:  # 如果需要导出MLIR
        filename = f"wave_decode_attention_phase0_{'x'.join(map(str, shape))}.mlir"  # 构建阶段0的MLIR文件名
        with open(filename, "w") as f:  # 打开文件写入
            f.write(mb_qk.module_op.get_asm())  # 写入MLIR汇编代码

    mb_sv = phase_1(attn_logits, attn_logits_max, b_req_idx, o)  # 执行阶段1（SV加权求和计算）
    if dump_generated_mlir:  # 如果需要导出MLIR
        filename = f"wave_decode_attention_phase1_{'x'.join(map(str, shape))}.mlir"  # 构建阶段1的MLIR文件名
        with open(filename, "w") as f:  # 打开文件写入
            f.write(mb_sv.module_op.get_asm())  # 写入MLIR汇编代码


# 解码注意力前向传播入口函数
def decode_attention_fwd(
    q,  # 查询张量
    k_buffer,  # 键缓存缓冲区
    v_buffer,  # 值缓存缓冲区
    o,  # 输出张量
    b_req_idx,  # 批次请求索引
    req_to_token,  # 请求到token的映射
    attn_logits,  # 注意力logits
    attn_logits_max,  # 注意力logits最大值
    num_kv_splits,  # KV分片数
    max_kv_splits,  # 最大KV分片数
    sm_scale,  # softmax缩放因子
    logit_cap=0.0,  # logits上限值 # logits上限值，默认为0.0
):
    decode_attention_wave(  # 调用Wave解码注意力计算
        q,  # 查询张量
        k_buffer,  # 键缓存
        v_buffer,  # 值缓存
        o,  # 输出
        b_req_idx,  # 批次请求索引
        req_to_token,  # 请求到token映射
        attn_logits,  # 注意力logits
        attn_logits_max,  # 注意力logits最大值
        num_kv_splits,  # KV分片数
        max_kv_splits,  # 最大KV分片数
        sm_scale,  # softmax缩放因子
        logit_cap,  # logits上限
    )
