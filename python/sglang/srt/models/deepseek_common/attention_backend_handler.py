# 注意力后端调度处理器
# 本文件负责根据不同的注意力后端（如 flashinfer、fa3、flashmla、triton 等）
# 和前向批次状态（prefill/decode/speculative 等），选择合适的注意力前向方法
# （MHA、MLA、MHA_CHUNKED_KV、MHA_ONE_SHOT 等）。

from sglang.srt.compilation.piecewise_cuda_graph_manager import is_in_piecewise_cuda_graph
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.utils.cp_utils import mla_use_prefill_cp
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_common.utils import _is_hip
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import use_intel_amx_backend

# 支持 MHA_ONE_SHOT 模式的后端列表
MHA_ONE_SHOT_SUPPORTED_BACKENDS = ["fa3", "flashinfer", "flashmla"]


# 注意力后端注册表，将后端名称映射到对应的处理函数
class AttentionBackendRegistry:
    _handlers = {}

    # 注册一个后端名称及其处理函数
    @classmethod
    def register(cls, backend_name, handler_func):
        cls._handlers[backend_name] = handler_func

    # 获取指定后端的处理函数，若不存在则返回 triton 后端的处理函数
    @classmethod
    def get_handler(cls, backend_name):
        return cls._handlers.get(backend_name, cls._handlers.get("triton"))


# 根据 MLA 子类型分派前向方法
# - ROCm 平台：解码阶段使用融合 RoPE 的 MLA，否则使用标准 MLA
# - 其他平台：若使用 Intel AMX 后端且具有 fused_qkv_a_proj_with_mqa，使用 CPU 融合 RoPE MLA，否则使用标准 MLA
def _dispatch_mla_subtype(attn, forward_batch):
    if _is_hip:
        if attn.rocm_fused_decode_mla and forward_batch.forward_mode.is_decode():
            return AttnForwardMethod.MLA_FUSED_ROPE_ROCM
        else:
            return AttnForwardMethod.MLA
    else:
        if hasattr(attn, "fused_qkv_a_proj_with_mqa") and use_intel_amx_backend(attn):
            return AttnForwardMethod.MLA_FUSED_ROPE_CPU
        else:
            return AttnForwardMethod.MLA


# 昇腾（Ascend/NPU）后端的注意力处理函数
# - 扩展模式（非投机验证/草稿扩展）：有 indexer 时使用 DSA_NPU，否则使用 MHA_NPU
# - 其他模式（解码等）：有 indexer 时使用 DSA_NPU，否则使用 MLA_NPU
def handle_attention_ascend(attn, forward_batch):
    if (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_target_verify()
        and not forward_batch.forward_mode.is_draft_extend()
        and not forward_batch.forward_mode.is_draft_extend_v2()
    ):
        if hasattr(attn, "indexer"):
            return AttnForwardMethod.DSA_NPU
        else:
            return AttnForwardMethod.MHA_NPU
    else:
        if hasattr(attn, "indexer"):
            return AttnForwardMethod.DSA_NPU
        else:
            return AttnForwardMethod.MLA_NPU


# 获取扩展前缀长度之和
def _get_sum_extend_prefix_lens(forward_batch):
    return (
        sum(forward_batch.extend_prefix_lens_cpu)
        if forward_batch.extend_prefix_lens_cpu is not None
        else 0
    )


# 判断是否支持 MHA_ONE_SHOT 模式
# 需要后端支持且序列总长度不超过最大块容量
def _support_mha_one_shot(attn, forward_batch, backend_name):
    attn_supported = backend_name in MHA_ONE_SHOT_SUPPORTED_BACKENDS
    sum_seq_lens = (
        sum(forward_batch.seq_lens_cpu) if forward_batch.seq_lens_cpu is not None else 0
    )
    return attn_supported and sum_seq_lens <= forward_batch.get_max_chunk_capacity()


# 通用注意力后端处理核心逻辑
# 1. 分段 CUDA 图模式下始终使用 MLA
# 2. MLA prefill CP 模式下使用 absorbed MLA
# 3. 非投机扩展模式下：
#    - 前缀长度超过阈值或为0时，支持 MHA_ONE_SHOT 则使用，否则使用 MHA_CHUNKED_KV
# 4. 其他情况分派到 MLA 子类型
def _handle_attention_backend(attn, forward_batch, backend_name):
    if is_in_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    # MLA prefill CP forces absorbed MLA regardless of prefix length: the
    # CP path gathers latent KV via rebuild_cp_kv_cache and feeds the
    # backend's absorbed-MLA kernel.
    if mla_use_prefill_cp(forward_batch):
        return _dispatch_mla_subtype(attn, forward_batch)

    sum_extend_prefix_lens = _get_sum_extend_prefix_lens(forward_batch)
    disable_ragged = (
        backend_name in ["flashinfer", "flashmla"]
    ) and attn.flashinfer_mla_disable_ragged

    if (
        not disable_ragged
        and forward_batch.forward_mode.is_extend_without_speculative()
        and (
            (
                sum_extend_prefix_lens >= attn.chunked_prefix_cache_threshold
                and not attn.disable_chunked_prefix_cache
            )
            or sum_extend_prefix_lens == 0
        )
    ):
        if _support_mha_one_shot(attn, forward_batch, backend_name):
            return AttnForwardMethod.MHA_ONE_SHOT
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


# FlashInfer 后端处理函数
def handle_attention_flashinfer(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "flashinfer")


# FlashAttention 3 后端处理函数
# 确定性推理启用时使用 MLA，否则走通用处理逻辑
def handle_attention_fa3(attn, forward_batch):
    # when deterministic inference is enabled, use MLA
    if get_global_server_args().enable_deterministic_inference:
        return _dispatch_mla_subtype(attn, forward_batch)
    else:
        return _handle_attention_backend(attn, forward_batch, "fa3")


# FlashMLA 后端处理函数
def handle_attention_flashmla(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "flashmla")


# CUTLASS MLA 后端处理函数
def handle_attention_cutlass_mla(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "cutlass_mla")


# FlashAttention 4 后端处理函数
# TODO(cicirori): 目前 DeepSeekV3 使用 FA4 MHA
def handle_attention_fa4(attn, forward_batch):
    # TODO(cicirori): use FA4 MHA for DeepSeekV3 for now
    return AttnForwardMethod.MHA_CHUNKED_KV


# TRT-LLM MLA 后端处理函数
# 分段 CUDA 图模式下使用 MLA
# 扩展模式且未禁用分块前缀缓存时使用 MHA_CHUNKED_KV，否则分派 MLA 子类型
def handle_attention_trtllm_mla(attn, forward_batch):
    if is_in_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    sum_extend_prefix_lens = _get_sum_extend_prefix_lens(forward_batch)
    if forward_batch.forward_mode.is_extend_without_speculative() and (
        not attn.disable_chunked_prefix_cache or sum_extend_prefix_lens == 0
    ):
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


# TokenSpeed MLA 后端处理函数
# 与 TRT-LLM MLA 共享相同的分派模式：纯 prefill 使用 MHA_CHUNKED_KV，投机解码/解码使用 MLA
def handle_attention_tokenspeed_mla(attn, forward_batch):
    # tokenspeed_mla shares the trtllm_mla dispatch pattern: pure prefill goes
    # via MHA chunked KV (TRT-LLM ragged), spec decode / decode goes via MLA.
    return handle_attention_trtllm_mla(attn, forward_batch)


# AITER 后端处理函数
# 扩展模式使用 MHA，解码模式使用 MLA
def handle_attention_aiter(attn, forward_batch):
    if forward_batch.forward_mode.is_extend_without_speculative():
        return AttnForwardMethod.MHA
    else:
        return AttnForwardMethod.MLA


# DSA（Deepseek Sparse Attention）后端处理函数
# 分派逻辑集中在 DeepseekSparseAttnBackend.set_dsa_prefill_impl 中，
# 并在 init_forward_metadata 中执行。此处从 backend.use_mha 读取决策。
def handle_attention_dsa(attn, forward_batch):
    """
    Dispatch logic is centralized in DeepseekSparseAttnBackend.set_dsa_prefill_impl and executed
    in init_forward_metadata. Read the decision from backend.use_mha.
    """

    backend = get_attn_backend()
    if isinstance(backend, TboAttnBackend):  # 启用 TBO 时，获取主后端
        backend = backend.primary
    if hasattr(backend, "use_mha") and backend.use_mha:
        return AttnForwardMethod.MHA_ONE_SHOT
    return AttnForwardMethod.MLA


# Triton 后端处理函数
# 分段 CUDA 图模式下使用 MLA
# 确定性推理启用时使用 MLA
# 无前缀扩展时使用 MHA，否则分派 MLA 子类型
def handle_attention_triton(attn, forward_batch):
    if is_in_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    # when deterministic inference is enabled, use MLA
    if get_global_server_args().enable_deterministic_inference:
        return _dispatch_mla_subtype(attn, forward_batch)

    if (
        forward_batch.forward_mode.is_extend_without_speculative()
        and sum(forward_batch.extend_prefix_lens_cpu) == 0
    ):
        return AttnForwardMethod.MHA
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


# Intel XPU 后端处理函数
def handle_attention_intel_xpu(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "intel_xpu")


# 注册所有后端及其处理函数
AttentionBackendRegistry.register("ascend", handle_attention_ascend)
AttentionBackendRegistry.register("flashinfer", handle_attention_flashinfer)
AttentionBackendRegistry.register("fa3", handle_attention_fa3)
AttentionBackendRegistry.register("flashmla", handle_attention_flashmla)
AttentionBackendRegistry.register("cutlass_mla", handle_attention_cutlass_mla)
AttentionBackendRegistry.register("fa4", handle_attention_fa4)
AttentionBackendRegistry.register("trtllm_mla", handle_attention_trtllm_mla)
AttentionBackendRegistry.register("tokenspeed_mla", handle_attention_tokenspeed_mla)
AttentionBackendRegistry.register("aiter", handle_attention_aiter)
AttentionBackendRegistry.register("dsa", handle_attention_dsa)
AttentionBackendRegistry.register(
    "nsa", handle_attention_dsa
)  # Deprecated alias; use "dsa"
AttentionBackendRegistry.register("triton", handle_attention_triton)
AttentionBackendRegistry.register("intel_xpu", handle_attention_intel_xpu)
