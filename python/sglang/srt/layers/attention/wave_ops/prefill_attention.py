# Wave预填充注意力模块，基于Wave语言实现的高效预填充注意力计算，支持page_size=1
"""
Memory-efficient attention for prefill.  # 解码阶段的高效内存注意力
It support page size = 1.  # 支持页大小为1
"""

import math  # 导入数学模块
import os  # 导入操作系统模块

from wave_lang.kernel.lang.global_symbols import *  # 导入Wave语言全局符号
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile  # 导入Wave编译选项和编译函数
from wave_lang.kernel.wave.constraints import MMAType  # 导入矩阵乘法累加类型
from wave_lang.kernel.wave.templates.attention_common import AttentionShape  # 导入注意力形状定义
from wave_lang.kernel.wave.templates.prefill_attention import (  # 导入预填充注意力核函数生成器
    get_prefill_attention_kernel,  # 获取预填充注意力核函数
)
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params  # 导入默认调度参数
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config  # 导入默认运行配置设置

dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))  # 是否导出生成的MLIR，默认为0


# 执行Wave预填充注意力计算
def prefill_attention_wave(
    q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=True  # is_causal: 是否使用因果掩码 # 是否使用因果掩码，默认为True
):

    shape = AttentionShape(  # 构建注意力形状对象
        num_query_heads=q.shape[1],  # 查询头数
        num_kv_heads=k.shape[1],  # 键值头数
        head_size=q.shape[2],  # 查询头维度
        head_size_kv=k.shape[2],  # 键值头维度
        num_seqs=b_seq_len.shape[0],  # 序列数
        max_seq_len=max_seq_len,  # 最大序列长度
        total_seq_len=q.shape[0],  # 总序列长度
    )

    assert shape.num_query_heads % shape.num_kv_heads == 0  # 断言查询头数必须能被键值头数整除

    output_shape = (shape.total_seq_len, shape.num_query_heads, shape.head_size_kv)  # 计算输出形状
    # Run the wave kernel.  # 运行Wave核函数
    mfma_variant = (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)  # 设置MFMA变体类型
    prefill, hyperparams = get_prefill_attention_kernel(  # 获取预填充注意力核函数和超参数
        shape,  # 注意力形状
        mfma_variant,  # MFMA变体
        q.shape,  # 查询张量形状
        k.shape,  # 键张量形状
        v.shape,  # 值张量形状
        output_shape,  # 输出形状
        input_dtype=q.dtype,  # 输入数据类型
        output_dtype=o.dtype,  # 输出数据类型
        size_dtype=b_seq_len.dtype,  # 尺寸数据类型
    )

    hyperparams.update(get_default_scheduling_params())  # 用默认调度参数更新超参数

    log2e = 1.44269504089  # log2(e)的近似值，用于softmax计算
    dk_sqrt = math.sqrt(1.0 / shape.head_size)  # 计算缩放因子1/sqrt(d_k)

    options = WaveCompileOptions(  # 创建Wave编译选项
        subs=hyperparams,  # 超参数替换
        canonicalize=True,  # 启用规范化
        run_bench=False,  # 不运行基准测试
        use_scheduling_barriers=False,  # 不使用调度屏障
    )
    options = set_default_run_config(options)  # 设置默认运行配置
    prefill = wave_compile(options, prefill)  # 编译预填充核函数

    mb = prefill(  # 执行预填充核函数
        q * dk_sqrt * log2e,  # 对查询张量应用缩放和log2e变换
        k,  # 键张量
        v,  # 值张量
        b_start_loc,  # 批次起始位置
        b_seq_len,  # 批次序列长度
        o,  # 输出张量
    )
    if dump_generated_mlir:  # 如果需要导出MLIR
        shape_list = [q.shape[0], q.shape[1], k.shape[1], q.shape[2], k.shape[2]]  # 构建形状列表
        filename = f"wave_prefill_attention_{'x'.join(map(str, shape_list))}.mlir"  # 构建文件名
        with open(filename, "w") as f:  # 打开文件写入
            f.write(mb.module_op.get_asm())  # 写入MLIR汇编代码
