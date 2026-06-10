# LoRA Triton算子模块的初始化文件，导出所有Triton实现的LoRA前向计算函数
# 包括分块Embedding查找、分块SGMV扩展/收缩、QKV/门控LoRA、融合MoE LoRA等算子
from .chunked_embedding_lora_a import chunked_embedding_lora_a_forward  # 导入分块Embedding LoRA A前向函数
from .chunked_sgmv_expand import chunked_sgmv_lora_expand_forward  # 导入分块SGMV LoRA扩展前向函数
from .chunked_sgmv_shrink import chunked_sgmv_lora_shrink_forward  # 导入分块SGMV LoRA收缩前向函数
from .embedding_lora_a import embedding_lora_a_fwd  # 导入Embedding LoRA A前向函数
from .fused_moe_lora_kernel import fused_moe_lora  # 导入融合MoE LoRA内核函数
from .gate_up_lora_b import gate_up_lora_b_fwd  # 导入门控LoRA B前向函数
from .kv_b_lora_absorbed import (  # 导入KV LoRA吸收后的步骤函数
    step_a_q_fwd,  # Q的步骤A前向函数
    step_a_v_fwd,  # V的步骤A前向函数
    step_b_q_fwd,  # Q的步骤B前向函数
    step_b_v_fwd,  # V的步骤B前向函数
)
from .qkv_lora_b import qkv_lora_b_fwd  # 导入QKV LoRA B前向函数
from .sgemm_lora_a import sgemm_lora_a_fwd  # 导入SGEMM LoRA A前向函数
from .sgemm_lora_b import sgemm_lora_b_fwd  # 导入SGEMM LoRA B前向函数
from .virtual_experts import merged_experts_fused_moe_lora_add  # 导入合并专家融合MoE LoRA加法函数

__all__ = [  # 模块公开接口列表
    "gate_up_lora_b_fwd",  # 门控LoRA B前向函数
    "qkv_lora_b_fwd",  # QKV LoRA B前向函数
    "sgemm_lora_a_fwd",  # SGEMM LoRA A前向函数
    "sgemm_lora_b_fwd",  # SGEMM LoRA B前向函数
    "chunked_sgmv_lora_shrink_forward",  # 分块SGMV LoRA收缩前向函数
    "chunked_sgmv_lora_expand_forward",  # 分块SGMV LoRA扩展前向函数
    "fused_moe_lora",  # 融合MoE LoRA函数
    "chunked_embedding_lora_a_forward",  # 分块Embedding LoRA A前向函数
    "embedding_lora_a_fwd",  # Embedding LoRA A前向函数
    "merged_experts_fused_moe_lora_add",  # 合并专家融合MoE LoRA加法函数
    "step_a_q_fwd",  # Q的步骤A前向函数
    "step_a_v_fwd",  # V的步骤A前向函数
    "step_b_q_fwd",  # Q的步骤B前向函数
    "step_b_v_fwd",  # V的步骤B前向函数
]
