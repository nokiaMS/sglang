# Wave扩展注意力模块，基于Wave语言实现的高效扩展（prefill）注意力计算，支持page_size=1
"""
Memory-efficient attention for prefill.  # 预填充阶段的高效内存注意力
It support page size = 1.  # 支持页大小为1
"""

import functools  # 导入functools模块，用于缓存装饰器
import os  # 导入操作系统模块

import torch  # 导入PyTorch
from wave_lang.kernel.lang.global_symbols import *  # 导入Wave语言全局符号
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile  # 导入Wave编译选项和编译函数
from wave_lang.kernel.wave.constraints import MMAType  # 导入矩阵乘法累加类型
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType  # 导入调度类型
from wave_lang.kernel.wave.templates.attention_common import AttentionShape  # 导入注意力形状定义
from wave_lang.kernel.wave.templates.extend_attention import get_extend_attention_kernel  # 导入扩展注意力核函数生成器
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params  # 导入默认调度参数
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config  # 导入默认运行配置设置

dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))  # 是否导出生成的MLIR，默认为0


# 获取或编译Wave扩展注意力核函数（带LRU缓存）
@functools.lru_cache  # 使用LRU缓存避免重复编译
def get_wave_kernel(
    shape: AttentionShape,  # 注意力形状
    q_shape: tuple[int],  # 查询张量形状
    k_shape: tuple[int],  # 键张量形状
    v_shape: tuple[int],  # 值张量形状
    k_cache_shape: tuple[int],  # 键缓存形状
    v_cache_shape: tuple[int],  # 值缓存形状
    o_shape: tuple[int],  # 输出张量形状
    input_dtype: torch.dtype,  # 输入数据类型
    output_dtype: torch.dtype,  # 输出数据类型
    size_dtype: torch.dtype,  # 尺寸数据类型
    is_causal: bool,  # 是否使用因果掩码
    logit_cap: float,  # logits上限值
    layer_scaling: float,  # 层缩放因子
):
    assert shape.num_query_heads % shape.num_kv_heads == 0  # 断言查询头数必须能被键值头数整除

    mfma_variant = (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16)  # 设置MFMA变体类型
    (
        extend_attention,  # 扩展注意力核函数
        hyperparams,  # 超参数
        dynamic_symbols,  # 动态符号
    ) = get_extend_attention_kernel(  # 获取扩展注意力核函数
        shape,  # 注意力形状
        mfma_variant,  # MFMA变体
        q_shape,  # 查询形状
        k_shape,  # 键形状
        v_shape,  # 值形状
        k_cache_shape,  # 键缓存形状
        v_cache_shape,  # 值缓存形状
        o_shape,  # 输出形状
        input_dtype=input_dtype,  # 输入数据类型
        output_dtype=output_dtype,  # 输出数据类型
        size_dtype=size_dtype,  # 尺寸数据类型
        is_causal=is_causal,  # 是否因果
        layer_scaling=layer_scaling,  # 层缩放因子
        logit_cap=logit_cap,  # logits上限
    )

    hyperparams.update(get_default_scheduling_params())  # 用默认调度参数更新超参数
    options = WaveCompileOptions(  # 创建Wave编译选项
        subs=hyperparams,  # 超参数替换
        canonicalize=True,  # 启用规范化
        run_bench=False,  # 不运行基准测试
        schedule=SchedulingType.NONE,  # 不使用调度
        use_scheduling_barriers=False,  # 不使用调度屏障
        dynamic_symbols=dynamic_symbols,  # 动态符号
        use_buffer_ops=True,  # 使用缓冲区操作
        waves_per_eu=2,  # 每个执行单元的wave数
        denorm_fp_math_f32="preserve-sign",  # 非规格化浮点数学保持符号
        wave_runtime=True,  # 启用Wave运行时
    )
    options = set_default_run_config(options)  # 设置默认运行配置
    extend_attention = wave_compile(options, extend_attention)  # 编译扩展注意力核函数

    return extend_attention  # 返回编译后的核函数


# 执行Wave扩展注意力计算
def extend_attention_wave(
    q_extend,  # 扩展查询张量
    k_extend,  # 扩展键张量
    v_extend,  # 扩展值张量
    k_buffer,  # 键缓存缓冲区
    v_buffer,  # 值缓存缓冲区
    qo_indptr,  # 查询/输出索引指针
    kv_indptr,  # 键值索引指针
    kv_indices,  # 键值索引
    custom_mask,  # 自定义掩码
    mask_indptr,  # 掩码索引指针
    max_seq_len,  # 最大序列长度
    output,  # 输出张量
    is_causal=True,  # 是否使用因果掩码 # 是否使用因果掩码，默认为True
    layer_scaling=None,  # 层缩放因子 # 层缩放因子，默认为None
    logit_cap=0,  # logits上限值 # logits上限值，默认为0
):
    shape = AttentionShape(  # 构建注意力形状对象
        num_query_heads=q_extend.shape[1],  # 查询头数
        num_kv_heads=k_extend.shape[1],  # 键值头数
        head_size=q_extend.shape[2],  # 查询头维度
        head_size_kv=k_extend.shape[2],  # 键值头维度
        num_seqs=kv_indptr.shape[0] - 1,  # 序列数（索引指针长度减1）
        max_seq_len=max_seq_len,  # 最大序列长度
    )

    # Run the wave kernel.  # 运行Wave核函数
    extend_attention = get_wave_kernel(  # 获取扩展注意力核函数
        shape,  # 注意力形状
        q_extend.shape,  # 查询形状
        k_extend.shape,  # 键形状
        v_extend.shape,  # 值形状
        k_buffer.shape,  # 键缓存形状
        v_buffer.shape,  # 值缓存形状
        output.shape,  # 输出形状
        input_dtype=q_extend.dtype,  # 输入数据类型
        output_dtype=output.dtype,  # 输出数据类型
        size_dtype=qo_indptr.dtype,  # 尺寸数据类型
        is_causal=is_causal,  # 是否因果
        layer_scaling=layer_scaling,  # 层缩放因子
        logit_cap=logit_cap,  # logits上限
    )

    mb = extend_attention(  # 执行扩展注意力核函数
        q_extend,  # 查询张量
        k_extend,  # 键张量
        v_extend,  # 值张量
        k_buffer,  # 键缓存
        v_buffer,  # 值缓存
        qo_indptr,  # 查询输出索引指针
        kv_indptr,  # 键值索引指针
        kv_indices,  # 键值索引
        max_seq_len,  # 最大序列长度
        output,  # 输出张量
    )

    if dump_generated_mlir:  # 如果需要导出MLIR
        shape_list = [  # 构建形状列表
            q_extend.shape[0],  # 查询token数
            q_extend.shape[1],  # 查询头数
            k_extend.shape[1],  # 键头数
            q_extend.shape[2],  # 查询头维度
            k_extend.shape[2],  # 键头维度
        ]
        filename = f"wave_prefill_attention_{'x'.join(map(str, shape_list))}.mlir"  # 构建文件名
        with open(filename, "w") as f:  # 打开文件写入
            f.write(mb.module_op.get_asm())  # 写入MLIR汇编代码
