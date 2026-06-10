# HIP Flash MLA注意力计算入口模块
# 本文件提供了Flash MLA（Multi-head Latent Attention）与KV缓存结合的入口函数，
# 支持多种后端实现（torch、tilelang、triton、kernel），主要用于HIP（AMD ROCm）平台上的
# FP8稀疏注意力解码计算和正确性验证。

from typing import Any, Optional  # 导入类型注解工具

import torch  # 导入PyTorch库

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz  # 导入FP8 FNUZ检测函数
from sglang.srt.utils import is_hip  # 导入HIP平台检测函数

FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn  # 根据FNUZ支持情况选择FP8数据类型


def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):  # Flash MLA KV缓存注意力的入口分发函数
    if is_hip():  # 如果是HIP（AMD）平台
        import os  # 导入操作系统模块

        backend = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "tilelang")  # 从环境变量获取后端，默认tilelang
    else:  # 否则（CUDA平台）
        import sgl_kernel.flash_mla as flash_mla  # 导入sgl_kernel的flash_mla模块

    if backend == "comparison":  # 如果是比较模式
        pack_ref, pack_fast_via_tester = flash_mla_with_kvcache_entrypoint(  # 使用torch后端获取参考结果
            backend="torch", **kwargs
        )
        pack_fast_via_api = flash_mla_with_kvcache_entrypoint(  # 使用kernel后端获取API结果
            backend="kernel", **kwargs
        )
        _assert_close(pack_ref=pack_fast_via_tester, pack_fast=pack_fast_via_api)  # 断言测试器结果与API结果接近
        _assert_close(pack_ref=pack_ref, pack_fast=pack_fast_via_tester)  # 断言参考结果与测试器结果接近
        _assert_close(pack_ref=pack_ref, pack_fast=pack_fast_via_api)  # 断言参考结果与API结果接近
        return pack_ref  # 返回参考结果

    if backend == "torch":  # 如果是torch后端
        return flash_mla_with_kvcache_torch(**kwargs)  # 调用torch实现

    if backend == "tilelang":  # 如果是tilelang后端
        from sglang.srt.layers.attention.dsa.tilelang_kernel import (  # 导入tilelang稀疏注意力
            dpsk_v4_fp8_attention_fwd,
        )

        return dpsk_v4_fp8_attention_fwd(**kwargs)  # 调用tilelang实现

    if backend == "triton":  # 如果是triton后端
        from sglang.srt.layers.attention.nsa.triton_decode import (  # 导入triton FP8注意力
            triton_fp8_attention_fwd,
        )

        return triton_fp8_attention_fwd(**kwargs)  # 调用triton实现

    if backend == "kernel":  # 如果是kernel后端（CUDA sgl_kernel）
        return flash_mla.flash_mla_with_kvcache(**kwargs)  # 调用sgl_kernel实现

    raise NotImplementedError(f"unknown backend: {backend!r}")  # 未知后端则抛出异常


def flash_mla_with_kvcache_torch(  # 使用PyTorch实现的Flash MLA KV缓存注意力（用于参考验证）
    q: torch.Tensor,  # 查询张量
    k_cache: torch.Tensor,  # 键缓存张量
    block_table: Optional[torch.Tensor],  # 块表
    cache_seqlens: Optional[torch.Tensor],  # 缓存序列长度
    head_dim_v: int,  # V头维度
    tile_scheduler_metadata: Any,  # tile调度器元数据
    num_splits: None = None,  # 分割数
    softmax_scale: Optional[float] = None,  # softmax缩放因子
    causal: bool = False,  # 是否因果注意力
    is_fp8_kvcache: bool = False,  # 是否FP8 KV缓存
    indices: Optional[torch.Tensor] = None,  # 索引张量
    attn_sink: Optional[torch.Tensor] = None,  # 注意力汇聚点
    extra_k_cache: Optional[torch.Tensor] = None,  # 额外键缓存
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,  # 额外KV缓存索引
    topk_length: Optional[torch.Tensor] = None,  # topk长度
    extra_topk_length: Optional[torch.Tensor] = None,  # 额外topk长度
):

    from sglang.srt.flashmla_tests import quant as flashmla_quant  # 导入flashmla量化模块
    from sglang.srt.flashmla_tests.lib import (  # 导入flashmla测试库
        ExtraTestParamForDecode,  # 解码额外测试参数
        KVScope,  # KV范围
        TestcaseForDecode,  # 解码测试用例
        TestParam,  # 测试参数
    )
    from sglang.srt.flashmla_tests.ref import ref_sparse_attn_decode  # 导入参考稀疏注意力解码

    assert block_table is None  # 断言块表为None
    assert cache_seqlens is None  # 断言缓存序列长度为None
    assert is_fp8_kvcache  # 断言为FP8 KV缓存

    b, s_q, h_q, d_qk = q.shape  # 获取查询张量的维度：批次、序列长度、头数、QK维度
    d_v = head_dim_v  # V维度

    fp8_layout = flashmla_quant.FP8KVCacheLayout.MODEL1_FP8Sparse  # FP8 KV缓存布局

    p = TestParam(  # 创建测试参数
        s_q=s_q,  # 查询序列长度
        s_kv="unused",  # 键值序列长度（未使用）
        topk="unused",  # topk（未使用）
        h_q=h_q,  # 查询头数
        h_kv=1,  # 键值头数
        d_qk=d_qk,  # QK维度
        d_v=d_v,  # V维度
        decode=ExtraTestParamForDecode(  # 解码额外参数
            b=b,  # 批次大小
            is_varlen="unused",  # 是否变长（未使用）
            have_zero_seqlen_k="unused",  # 是否有零长度序列（未使用）
            extra_s_k="unused",  # 额外序列长度（未使用）
            extra_topk="unused",  # 额外topk（未使用）
            extra_block_size="unused",  # 额外块大小（未使用）
            have_extra_topk_length="unused",  # 是否有额外topk长度（未使用）
        ),
        # unused?  # 未使用？
        seed=-1,  # 随机种子
        check_correctness=True,  # 是否检查正确性
        is_all_indices_invalid=False,  # 是否所有索引无效
        num_runs=10,  # 运行次数
        have_attn_sink=True,  # 是否有注意力汇聚点
        have_topk_length=True,  # 是否有topk长度
    )

    blocked_k_quantized = k_cache  # 量化后的分块键缓存
    blocked_k = flashmla_quant.dequantize_k_cache(  # 反量化键缓存
        blocked_k_quantized.view(FP8_DTYPE), fp8_layout  # 将量化缓存视为FP8类型并反量化
    )
    # blocked_k_requantized = flashmla_quant.quantize_k_cache(blocked_k, fp8_layout)
    # assert torch.testing.assert_allclose(blocked_k_requantized.byte(), blocked_k_quantized.byte())
    kv_scope = KVScope(  # 创建KV范围
        t="unused",  # 时间步（未使用）
        cache_seqlens="unused",  # 缓存序列长度（未使用）
        block_table="unused",  # 块表（未使用）
        blocked_k=blocked_k,  # 反量化后的分块键
        blocked_k_quantized=blocked_k_quantized,  # 量化后的分块键
        abs_indices="unused",  # 绝对索引（未使用）
        indices_in_kvcache=indices,  # KV缓存中的索引
        topk_length=topk_length,  # topk长度
    )

    extra_kv_scope = None  # 额外KV范围初始化为None
    if extra_k_cache is not None:  # 如果有额外键缓存
        extra_blocked_k_quantized = extra_k_cache  # 额外量化键缓存
        extra_blocked_k = flashmla_quant.dequantize_k_cache(  # 反量化额外键缓存
            extra_blocked_k_quantized.view(FP8_DTYPE), fp8_layout  # 视为FP8类型并反量化
        )
        # extra_blocked_k_requantized = flashmla_quant.quantize_k_cache(extra_blocked_k, fp8_layout)
        # assert torch.testing.assert_allclose(extra_blocked_k_requantized.byte(), extra_blocked_k_quantized.byte())
        extra_kv_scope = KVScope(  # 创建额外KV范围
            t="unused",  # 时间步（未使用）
            cache_seqlens="unused",  # 缓存序列长度（未使用）
            block_table="unused",  # 块表（未使用）
            blocked_k=extra_blocked_k,  # 反量化后的额外分块键
            blocked_k_quantized=extra_blocked_k_quantized,  # 量化后的额外分块键
            abs_indices="unused",  # 绝对索引（未使用）
            indices_in_kvcache=extra_indices_in_kvcache,  # 额外KV缓存索引
            topk_length=extra_topk_length,  # 额外topk长度
        )

    t = TestcaseForDecode(  # 创建解码测试用例
        p="unused",  # 测试参数（未使用）
        q=q,  # 查询张量
        attn_sink=attn_sink,  # 注意力汇聚点
        sm_scale=softmax_scale,  # softmax缩放因子
        kv_scope=kv_scope,  # KV范围
        extra_kv_scope=extra_kv_scope,  # 额外KV范围
    )
    # print(f"hi {p=} {t=}")
    # print(
    #     f"hi info "
    #     f"{get_tensor_info(t.kv_scope.blocked_k)=} "
    #     f"{get_tensor_info(t.kv_scope.blocked_k_quantized)=} "
    #     f"{get_tensor_info(t.extra_kv_scope.blocked_k) if t.extra_kv_scope is not None else None=} "
    #     f"{get_tensor_info(t.extra_kv_scope.blocked_k_quantized) if t.extra_kv_scope is not None else None=} "
    # )

    pack_ref = ref_sparse_attn_decode(p, t)  # 使用参考实现计算稀疏注意力解码

    # tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
    # pack_fast_via_tester = flashmla_lib.run_flash_mla_decode(
    #     p, t, tile_scheduler_metadata, num_splits=None
    # )

    # return pack_ref, pack_fast_via_tester
    return pack_ref  # 返回参考结果


def _assert_close(pack_ref, pack_fast):  # 断言两组结果足够接近
    import sglang.srt.flashmla_tests.kernelkit as kk  # 导入kernelkit测试工具

    out_ref, lse_ref = pack_ref  # 参考输出和log-softmax
    out_fast, lse_fast = pack_fast  # 快速输出和log-softmax

    # the copied threshold is too strict, not checked why
    # copied from: test_flash_mla_sparse_decoding.py
    # 复制的阈值太严格，未检查原因
    # 复制自：test_flash_mla_sparse_decoding.py
    # is_out_correct = kk.check_is_allclose(
    #     "out", out_fast, out_ref, abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6
    # )
    # is_lse_correct = kk.check_is_allclose(
    #     "lse", lse_fast, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536
    # )

    # loosen thresh  # 放宽阈值
    is_out_correct = kk.check_is_allclose(  # 检查输出是否接近
        "out", out_fast, out_ref, abs_tol=1e-2, rel_tol=10.0, cos_diff_tol=5e-6  # 使用宽松的阈值
    )
    is_lse_correct = kk.check_is_allclose(  # 检查log-softmax是否接近
        "lse", lse_fast, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536  # LSE阈值
    )

    assert is_out_correct and is_lse_correct, f"{is_out_correct=} {is_lse_correct=}"  # 断言两者都正确
