# ROCm 线性层工具模块
# 本文件提供了 AMD ROCm 平台上的线性层相关工具函数，
# 包括融合 QK/ROPE 拼接、MLA 缓存操作以及 DeepSeek V3 路由 GEMM 等。
import torch  # 导入 PyTorch 库
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla  # 导入融合 QK ROPE 拼接与 MLA 缓存函数
from aiter.ops.triton.fused_qk_concat import fused_qk_rope_cat  # 导入融合 QK ROPE 拼接函数
from aiter.tuned_gemm import tgemm  # 导入 aiter 调优 GEMM 模块

__all__ = ["fused_qk_rope_cat", "fused_qk_rope_cat_and_cache_mla"]  # 模块公开接口列表


def aiter_dsv3_router_gemm(  # DeepSeek V3 路由 GEMM 函数，使用 aiter 调优的 GEMM 调度器
    hidden_states: torch.Tensor,  # 隐藏状态张量
    weight: torch.Tensor,  # 权重张量
):
    """Use aiter tuned GEMM dispatcher (tgemm.mm) to automatically select the GEMM kernel."""  # 使用 aiter 调优 GEMM 调度器 (tgemm.mm) 自动选择 GEMM 内核
    return tgemm.mm(hidden_states, weight.detach(), otype=hidden_states.dtype)  # 调用 tgemm.mm 执行矩阵乘法，输出类型与输入一致


def get_dsv3_gemm_output_zero_allocator_size(  # 获取 DeepSeek V3 GEMM 输出零分配器大小
    n_routed_experts: int, num_moe_layers: int, allocate_size: int, embedding_dim: int  # 路由专家数、MoE 层数、分配大小、嵌入维度
):
    if embedding_dim != 7168 or n_routed_experts != 256:  # 如果嵌入维度不是 7168 或路由专家数不是 256
        return 0  # 返回 0，表示不需要分配

    per_layer_size = 256 * (allocate_size + n_routed_experts)  # 计算每层的大小

    return num_moe_layers * per_layer_size  # 返回总分配大小 = MoE 层数 × 每层大小
