# 文件说明：DeepSeek MLA（多头潜在注意力）在ROCm平台上的融合RoPE前向传播方法
# 本文件实现了基于AMD ROCm的MLA注意力融合计算，包括RoPE融合的解码注意力、
# FP8/BF16混合精度矩阵乘法，以及ROCm特有的融合解码MLA支持。

from __future__ import annotations  # 启用延迟类型注解评估

import os  # 导入操作系统模块
from typing import TYPE_CHECKING  # 导入类型检查工具

import torch  # 导入PyTorch深度学习框架

from sglang.srt.layers.quantization.fp8_kernel import per_tensor_quant_mla_fp8  # 导入FP8逐张量量化内核
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息类
from sglang.srt.model_executor.forward_context import (  # 导入前向上下文工具
    get_attn_backend,  # 获取注意力后端
    get_token_to_kv_pool,  # 获取KV缓存池
)
from sglang.srt.models.deepseek_common.utils import (  # 导入DeepSeek通用工具函数
    _is_cuda,  # 判断是否在CUDA上运行
    _is_hip,  # 判断是否在HIP（ROCm）上运行
)
from sglang.srt.utils import BumpAllocator, get_bool_env_var  # 导入内存分配器和环境变量工具

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA  # 导入DeepSeek V2 MLA注意力类

if _is_cuda:  # 如果在CUDA平台上
    from sgl_kernel import bmm_fp8  # 导入FP8批量矩阵乘法内核

if _is_hip:  # 如果在HIP（ROCm）平台上
    from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (  # 导入ROCm MLA解码RoPE算子
        decode_attention_fwd_grouped_rope,  # 分组RoPE解码注意力前向传播
    )


# DeepSeek MLA ROCm前向传播混入类，提供ROCm平台上MLA注意力的融合RoPE前向计算
class DeepseekMLARocmForwardMixin:

    # 初始化MLA融合RoPE ROCm前向传播，配置ROCm融合解码MLA开关
    def init_mla_fused_rope_rocm_forward(self: DeepseekV2AttentionMLA):
        self.rocm_fused_decode_mla = get_bool_env_var(  # 从环境变量读取ROCm融合解码MLA开关
            "SGLANG_ROCM_FUSED_DECODE_MLA", "false"  # 默认关闭
        )

    # MLA融合RoPE前向传播的预处理阶段，计算Q/K/V输入并准备注意力计算所需数据
    def forward_absorb_fused_mla_rope_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
        zero_allocator: BumpAllocator,  # 零值内存分配器
    ):
        enable_rope_fusion = (  # 检查是否启用RoPE融合
            os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"  # 默认启用
        )
        # NOTE: hidden_states can be a tuple for some quantization paths.
        # 注意：对于某些量化路径，hidden_states可能是元组。
        # For shape/device/dtype, use the first tensor; still pass the original
        # 对于shape/device/dtype，使用第一个张量；仍然传递原始的
        # hidden_states through linear ops which may accept tuple inputs.
        # hidden_states通过可能接受元组输入的线性操作。
        hidden_states_tensor = (  # 获取隐藏状态张量（处理元组情况）
            hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states  # 如果是元组取第一个，否则直接使用
        )

        q_len = hidden_states_tensor.shape[0]  # 获取查询序列长度
        q_input = hidden_states_tensor.new_empty(  # 分配Q输入张量
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim  # 形状：[序列长度, 头数, KV秩+RoPE维度]
        )
        if self.q_lora_rank is not None:  # 如果使用LoRA秩（MLA模式）
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(  # 融合QKV投影并分割为Q和潜在缓存
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1  # 按Q秩和KV秩+RoPE维度分割
            )
            q = self.q_a_layernorm(q)  # 对Q进行a层归一化
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)  # Q的b投影并重塑形状
        else:  # 非MLA模式
            q = self.q_proj(hidden_states)[0].view(  # 直接Q投影并重塑形状
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]  # 获取KV潜在缓存
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)  # 将Q分为非旋转部分和旋转部分

        if _is_hip:  # 如果在HIP（ROCm）平台上
            # TODO(haishaw): add bmm_fp8 to ROCm
            # 待办(haishaw)：为ROCm添加bmm_fp8
            q_nope_out = torch.bmm(  # 使用BF16批量矩阵乘法计算Q_nope与w_kc的乘积
                q_nope.to(torch.bfloat16).transpose(0, 1),  # Q非旋转部分转为BF16并转置
                self.w_kc.to(torch.bfloat16) * self.w_scale,  # K压缩权重转为BF16并乘以缩放因子
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:  # 如果w_kc为FP8类型
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(  # 对Q非旋转部分进行FP8量化
                q_nope.transpose(0, 1),  # 转置Q非旋转部分
                zero_allocator.allocate(1),  # 分配缩放因子空间
                dtype=torch.float8_e4m3fn,  # FP8数据类型
            )
            q_nope_out = bmm_fp8(  # 使用FP8批量矩阵乘法
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16  # 输入值、权重、缩放因子、输出类型
            )
        else:  # 其他精度
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)  # 使用标准批量矩阵乘法
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)  # 将Q非旋转输出填入Q输入前部
        v_input = latent_cache[..., : self.kv_lora_rank]  # 从潜在缓存中提取V输入
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)  # 对V输入进行归一化并增加维度
        k_input = latent_cache.unsqueeze(1)  # 将潜在缓存作为K输入并增加维度
        k_input[..., : self.kv_lora_rank] = v_input  # 将归一化后的V填入K输入前部

        if not enable_rope_fusion:  # 如果不启用RoPE融合
            k_pe = k_input[..., self.kv_lora_rank :]  # 提取K的旋转位置编码部分
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)  # 应用旋转位置编码
            q_input[..., self.kv_lora_rank :] = q_pe  # 将旋转后的Q填入Q输入后部
            k_input[..., self.kv_lora_rank :] = k_pe  # 将旋转后的K填入K输入后部
            k_pe_output = None  # 无需输出k_pe
        else:  # 启用RoPE融合时
            k_pe_output = torch.empty_like(k_input[..., self.kv_lora_rank :])  # 分配k_pe输出空间

        q_input[..., self.kv_lora_rank :] = q_pe  # 将Q的旋转部分填入Q输入后部

        # attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        # 注意力输出 = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        # Use Fused ROPE with use_rope=OFF.
        # 使用融合ROPE，use_rope=OFF。
        attn_output = torch.empty(  # 分配注意力输出张量
            (q_len, self.num_local_heads, self.kv_lora_rank),  # 形状：[序列长度, 头数, KV秩]
            dtype=q.dtype,  # 数据类型与Q一致
            device=q.device,  # 设备与Q一致
        )
        attn_logits, _, kv_indptr, kv_indices, _, _, _ = (  # 获取注意力后端的前向元数据
            get_attn_backend().forward_metadata
        )
        cos_sin_cache = self.rotary_emb.cos_sin_cache  # 获取旋转嵌入的余弦正弦缓存
        num_kv_split = get_attn_backend().num_kv_splits  # 获取KV分片数
        sm_scale = self.attn_mqa.scaling  # 获取注意力缩放因子
        if attn_logits is None:  # 如果注意力logits未分配
            attn_logits = torch.empty(  # 分配注意力logits张量
                (
                    forward_batch.batch_size,  # 批次大小
                    self.num_local_heads,  # 头数
                    num_kv_split,  # KV分片数
                    self.kv_lora_rank + 1,  # KV秩+1（用于max值）
                ),
                dtype=torch.float32,  # 使用float32精度
                device=q.device,  # 设备与Q一致
            )

        # save current latent cache.
        # 保存当前潜在缓存。
        get_token_to_kv_pool().set_kv_buffer(  # 将当前K输入写入KV缓存
            self.attn_mqa, forward_batch.out_cache_loc, k_input, None  # 注意力模块、输出缓存位置、K输入、无附加信息
        )
        key_cache_buf = get_token_to_kv_pool().get_key_buffer(self.attn_mqa.layer_id)  # 获取键缓存缓冲区
        val_cache_buf = key_cache_buf[..., : self.kv_lora_rank]  # 值缓存为键缓存的前KV秩部分

        return (  # 返回所有预处理结果
            q_input,  # Q输入
            key_cache_buf,  # 键缓存缓冲区
            val_cache_buf,  # 值缓存缓冲区
            attn_output,  # 注意力输出（空张量）
            kv_indptr,  # KV索引指针
            kv_indices,  # KV索引
            k_pe_output,  # K旋转位置编码输出
            cos_sin_cache,  # 余弦正弦缓存
            positions,  # 位置编码
            attn_logits,  # 注意力logits
            num_kv_split,  # KV分片数
            sm_scale,  # 缩放因子
            enable_rope_fusion,  # 是否启用RoPE融合
            k_input,  # K输入
            forward_batch,  # 前向批次信息
            zero_allocator,  # 零值内存分配器
        )

    # MLA融合RoPE前向传播的核心计算阶段，执行分组RoPE解码注意力和价值投影
    def forward_absorb_fused_mla_rope_core(
        self: DeepseekV2AttentionMLA,
        q_input,  # Q输入张量
        key_cache_buf,  # 键缓存缓冲区
        val_cache_buf,  # 值缓存缓冲区
        attn_output,  # 注意力输出张量
        kv_indptr,  # KV索引指针
        kv_indices,  # KV索引
        k_pe_output,  # K旋转位置编码输出
        cos_sin_cache,  # 余弦正弦缓存
        positions,  # 位置编码
        attn_logits,  # 注意力logits
        num_kv_split,  # KV分片数
        sm_scale,  # 缩放因子
        enable_rope_fusion,  # 是否启用RoPE融合
        k_input,  # K输入
        forward_batch,  # 前向批次信息
        zero_allocator,  # 零值内存分配器
    ):
        decode_attention_fwd_grouped_rope(  # 调用分组RoPE解码注意力前向传播
            q_input,  # Q输入
            key_cache_buf,  # 键缓存缓冲区
            val_cache_buf,  # 值缓存缓冲区
            attn_output,  # 注意力输出
            kv_indptr,  # KV索引指针
            kv_indices,  # KV索引
            k_pe_output,  # K旋转位置编码输出
            self.kv_lora_rank,  # KV的LoRA秩
            self.rotary_emb.rotary_dim,  # 旋转维度
            cos_sin_cache,  # 余弦正弦缓存
            positions,  # 位置编码
            attn_logits,  # 注意力logits
            num_kv_split,  # KV分片数
            sm_scale,  # 缩放因子
            logit_cap=self.attn_mqa.logit_cap,  # logit上限
            use_rope=enable_rope_fusion,  # 是否使用RoPE融合
            is_neox_style=self.rotary_emb.is_neox_style,  # 是否为Neox风格旋转
        )

        if enable_rope_fusion:  # 如果启用了RoPE融合
            k_input[..., self.kv_lora_rank :] = k_pe_output  # 将融合RoPE后的K旋转部分填入K输入
            get_token_to_kv_pool().set_kv_buffer(  # 更新KV缓存
                self.attn_mqa, forward_batch.out_cache_loc, k_input, None  # 注意力模块、输出缓存位置、K输入、无附加信息
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)  # 重塑注意力输出形状

        if _is_hip:  # 如果在HIP（ROCm）平台上
            # TODO(haishaw): add bmm_fp8 to ROCm
            # 待办(haishaw)：为ROCm添加bmm_fp8
            attn_bmm_output = torch.bmm(  # 使用BF16批量矩阵乘法计算注意力输出与w_vc的乘积
                attn_output.to(torch.bfloat16).transpose(0, 1),  # 注意力输出转为BF16并转置
                self.w_vc.to(torch.bfloat16) * self.w_scale,  # 价值压缩权重转为BF16并乘以缩放因子
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:  # 如果w_vc为FP8类型
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(  # 对注意力输出进行FP8量化
                attn_output.transpose(0, 1),  # 转置注意力输出
                zero_allocator.allocate(1),  # 分配缩放因子空间
                dtype=torch.float8_e4m3fn,  # FP8数据类型
            )
            attn_bmm_output = bmm_fp8(  # 使用FP8批量矩阵乘法
                attn_output_val,  # 量化后的注意力输出
                self.w_vc,  # 价值压缩权重
                attn_output_scale,  # 注意力输出缩放因子
                self.w_scale,  # 权重缩放因子
                torch.bfloat16,  # 输出类型
            )
        else:  # 其他精度
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)  # 使用标准批量矩阵乘法
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)  # 转置并展平bmm输出
        output, _ = self.o_proj(attn_output)  # 执行输出投影

        return output  # 返回最终输出
