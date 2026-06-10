# 文件说明：DeepSeek MLA（多头潜在注意力）在CPU平台上的融合RoPE前向传播方法
# 本文件实现了基于Intel AMX指令集的CPU端MLA注意力融合计算，
# 包括权重打包初始化、QKV投影融合RoPE的预处理以及核心注意力计算。

from __future__ import annotations  # 启用延迟类型注解评估

from typing import TYPE_CHECKING  # 导入类型检查工具

import torch  # 导入PyTorch深度学习框架

from sglang.srt.layers.amx_utils import PackWeightMethod  # 导入AMX权重打包方法
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息类
from sglang.srt.models.deepseek_common.utils import (  # 导入DeepSeek通用工具函数
    _is_cpu,  # 判断是否在CPU上运行
    _is_cpu_amx_available,  # 判断CPU AMX指令集是否可用
)
from sglang.srt.utils import BumpAllocator, use_intel_amx_backend  # 导入内存分配器和AMX后端判断工具

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA  # 导入DeepSeek V2 MLA注意力类


# DeepSeek MLA CPU前向传播混入类，提供CPU平台上MLA注意力的融合RoPE前向计算
class DeepseekMLACpuForwardMixin:

    # 初始化MLA融合RoPE CPU前向传播，配置权重打包和量化参数
    def init_mla_fused_rope_cpu_forward(self: DeepseekV2AttentionMLA):
        assert hasattr(self, "has_fused_proj") and hasattr(self, "is_packed_weight")  # 断言必须具有fused_proj和packed_weight属性

        # If we have self.fused_qkv_a_proj_with_mqa and we're running on CPU, we will choose the torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight kernel
        # 如果存在fused_qkv_a_proj_with_mqa且在CPU上运行，将选择torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight内核
        # which requires self.w_kc and self.w_vc to be packed.
        # 该内核要求self.w_kc和self.w_vc被打包。
        # If not, we will use torch.bmm and weight shouldn't be packed in this case
        # 否则，将使用torch.bmm，此时权重不应被打包
        if self.has_fused_proj and _is_cpu and _is_cpu_amx_available:  # 如果有融合投影且在CPU上且AMX可用
            self.quant_method = PackWeightMethod(  # 设置量化方法为权重打包方法
                weight_names=["w_kc", "w_vc"], transpose_dims=[[1, 2], [1, 2]]  # 指定需要打包的权重名和转置维度
            )

        self.qkv_proj_with_rope_is_int8 = (  # 判断QKV融合RoPE投影是否为int8量化
            self.has_fused_proj  # 是否有融合投影
            and not self.is_packed_weight  # 且权重未被打包
            and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.int8  # 且权重数据类型为int8
        )
        self.qkv_proj_with_rope_is_fp8 = (  # 判断QKV融合RoPE投影是否为fp8量化
            self.has_fused_proj  # 是否有融合投影
            and not self.is_packed_weight  # 且权重未被打包
            and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.float8_e4m3fn  # 且权重数据类型为fp8
        )

        self.weight_block_size = None  # 初始化权重块大小为None
        if self.qkv_proj_with_rope_is_fp8 and _is_cpu and _is_cpu_amx_available:  # 如果fp8量化且CPU且AMX可用
            assert getattr(  # 断言融合投影和q_b_proj的块量化配置一致
                self.fused_qkv_a_proj_with_mqa.quant_method, "block_quant", False  # 获取融合投影的块量化标志
            ) == getattr(self.q_b_proj.quant_method, "block_quant", False)  # 与q_b_proj的块量化标志比较
            use_block_quant = getattr(  # 获取是否使用块量化
                self.fused_qkv_a_proj_with_mqa.quant_method, "block_quant", False  # 从融合投影量化方法中获取
            )

            if use_block_quant:  # 如果使用块量化
                assert (  # 断言融合投影和q_b_proj的权重块大小一致
                    self.fused_qkv_a_proj_with_mqa.quant_method.quant_config.weight_block_size  # 融合投影的权重块大小
                    == self.q_b_proj.quant_method.quant_config.weight_block_size  # q_b_proj的权重块大小
                )
                self.weight_block_size = (  # 设置权重块大小
                    self.fused_qkv_a_proj_with_mqa.quant_method.quant_config.weight_block_size  # 从融合投影量化配置中获取
                )

    # MLA融合RoPE CPU前向传播的预处理阶段，执行QKV投影融合RoPE计算
    def forward_absorb_fused_mla_rope_cpu_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
        zero_allocator: BumpAllocator,  # 零值内存分配器
    ):
        assert self.q_lora_rank is not None and use_intel_amx_backend(  # 断言q_lora_rank存在且使用Intel AMX后端
            self
        ), "forward_absorb_fused_mla_rope_cpu_prepare requires q_lora_rank is not None and use_intel_amx_backend"  # 错误提示信息

        q_input, k_input, v_input = (  # 调用融合QKV投影RoPE内核，获取Q/K/V输入
            torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight(  # 调用SGL内核的融合QKV投影RoPE算子
                hidden_states,  # 隐藏状态
                self.fused_qkv_a_proj_with_mqa.weight,  # 融合QKV投影权重
                self.q_b_proj.weight,  # Q的b投影权重
                self.w_kc,  # K的压缩权重
                self.q_a_layernorm.weight,  # Q的a层归一化权重
                self.kv_a_layernorm.weight,  # KV的a层归一化权重
                positions,  # 位置编码
                self.rotary_emb.cos_sin_cache,  # 旋转嵌入的余弦正弦缓存
                self.kv_a_layernorm.variance_epsilon,  # KV层归一化的方差epsilon
                self.qkv_proj_with_rope_is_int8,  # 是否为int8量化
                self.qkv_proj_with_rope_is_fp8,  # 是否为fp8量化
                (  # 融合投影的权重缩放因子
                    self.fused_qkv_a_proj_with_mqa.weight_scale  # int8时的权重缩放
                    if self.qkv_proj_with_rope_is_int8
                    else (  # 否则检查fp8
                        self.fused_qkv_a_proj_with_mqa.weight_scale_inv  # fp8时的权重缩放逆
                        if self.qkv_proj_with_rope_is_fp8
                        else None  # 非量化时为None
                    )
                ),
                (  # q_b_proj的权重缩放因子
                    self.q_b_proj.weight_scale  # int8时的权重缩放
                    if self.qkv_proj_with_rope_is_int8
                    else (  # 否则检查fp8
                        self.q_b_proj.weight_scale_inv  # fp8时的权重缩放逆
                        if self.qkv_proj_with_rope_is_fp8
                        else None  # 非量化时为None
                    )
                ),
                self.w_scale if self.qkv_proj_with_rope_is_fp8 else None,  # fp8时的w缩放因子
                True,  # is_vnni # 是否为VNNI格式
                self.weight_block_size,  # 权重块大小
                self.q_lora_rank,  # Q的LoRA秩
                self.kv_lora_rank,  # KV的LoRA秩
                self.qk_rope_head_dim,  # QK旋转位置编码头维度
            )
        )
        return (q_input, k_input, v_input, forward_batch, zero_allocator)  # 返回Q/K/V输入、前向批次和分配器

    # MLA融合RoPE CPU前向传播的核心计算阶段，执行MQA注意力和价值投影
    def forward_absorb_fused_mla_rope_cpu_core(
        self: DeepseekV2AttentionMLA,
        q_input,  # Q输入张量
        k_input,  # K输入张量
        v_input,  # V输入张量
        forward_batch,  # 前向批次信息
        zero_allocator,  # 零值内存分配器
    ):
        assert self.q_lora_rank is not None and use_intel_amx_backend(  # 断言q_lora_rank存在且使用Intel AMX后端
            self
        ), "forward_absorb_fused_mla_rope_cpu_core requires q_lora_rank is not None and use_intel_amx_backend"  # 错误提示信息

        attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)  # 执行MQA注意力计算
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)  # 重塑注意力输出形状

        # [Note] Align shapes of bmm inputs.
        # [注意] 对齐bmm输入的形状。
        # Shapes of inputs:
        # 输入的形状：
        #   q_nope: [M, B, K]
        #   q_nope（Q的非旋转部分）: [M, B, K]
        #   original self.w_kc: [B, K, N]
        #   原始self.w_kc: [B, K, N]
        #   current self.w_kc (which has been converted in PackWeightMethod): [B, N, K]
        #   当前self.w_kc（已在PackWeightMethod中转换）: [B, N, K]

        # Shapes of inputs to sgl_kernel.cpu.bmm:
        # sgl_kernel.cpu.bmm输入的形状：
        #   out: [B, M, N]
        #   输出: [B, M, N]
        #   mat1: [B, M, K]
        #   矩阵1: [B, M, K]
        #   mat2: [B, N, K]
        #   矩阵2: [B, N, K]
        B = self.w_vc.size(0)  # 获取批次维度B
        N = self.w_vc.size(1)  # 获取输出维度N
        M = attn_output.size(0)  # 获取序列长度M
        output = torch.empty([M, int(B * N)], dtype=attn_output.dtype)  # 分配输出张量
        attn_bmm_output = output.view([M, B, N]).transpose_(0, 1)  # 重塑并转置为bmm输出视图
        torch.ops.sgl_kernel.bmm_cpu(  # 调用CPU批量矩阵乘法内核
            attn_bmm_output,  # 输出张量
            attn_output.transpose(0, 1),  # 注意力输出（转置后）
            self.w_vc,  # 价值压缩权重
            True,  # is_vnni # 是否为VNNI格式
            self.w_scale if self.qkv_proj_with_rope_is_fp8 else None,  # scale # fp8时的缩放因子
        )
        attn_output = output  # 将bmm输出赋值给attn_output
        output, _ = self.o_proj(attn_output)  # 执行输出投影

        return output  # 返回最终输出
