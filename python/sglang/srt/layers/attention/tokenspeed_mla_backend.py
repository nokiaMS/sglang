# 文件说明：tokenspeed-mla CuTe DSL注意力后端实现
# 该模块为Blackwell架构SM100上基于CuTe DSL的MLA注意力内核提供后端，
# 继承TRTLLMMLABackend，仅重写解码和预填充内核调用，其余元数据、
# KV缓存布局、CUDA图管理、FP8量化/RoPE、draft-extend填充和
# 分块前缀分发等逻辑均从父类继承。

# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations  # 启用延迟注解评估 # 启用延迟类型注解

"""Attention backend for the tokenspeed-mla CuTe DSL kernels on Blackwell.
# Blackwell架构上tokenspeed-mla CuTe DSL内核的注意力后端。

Subclasses :class:`TRTLLMMLABackend` and overrides only ``_run_decode_kernel``
and ``_run_prefill_kernel``. All metadata, KV-cache layout, CUDA-graph
plumbing, FP8 quantize/rope, draft-extend padding, and chunked-prefix
dispatch are inherited unchanged from the parent.
# 子类化TRTLLMMLABackend，仅重写_run_decode_kernel和_run_prefill_kernel。
# 所有元数据、KV缓存布局、CUDA图管理、FP8量化/RoPE、draft-extend填充
# 和分块前缀分发均从父类不变继承。
"""

import logging  # 导入日志模块 # 导入日志模块
from typing import TYPE_CHECKING, Optional  # 导入类型提示 # 导入类型检查和可选类型

import torch  # 导入PyTorch # 导入PyTorch框架

from sglang.jit_kernel.fp8_quantize import fp8_quantize  # 导入FP8量化内核 # 导入FP8量化函数
from sglang.jit_kernel.mla_kv_pack_quantize_fp8 import mla_kv_pack_quantize_fp8  # 导入MLA KV打包FP8量化内核 # 导入MLA KV打包量化函数
from sglang.jit_kernel.utils import is_arch_support_pdl  # 导入架构PDL支持检测 # 导入PDL架构支持检测
from sglang.srt.layers.attention.trtllm_mla_backend import (  # 导入TRT-LLM MLA后端 # 导入TRTLLM MLA后端类
    TRTLLMMLABackend,  # TRT-LLM MLA后端类 # TRTLLM MLA后端类
    TRTLLMMLAMultiStepDraftBackend,  # TRT-LLM MLA多步草稿后端类 # TRTLLM MLA多步草稿后端类
)
from sglang.srt.utils import is_flashinfer_available, is_tokenspeed_mla_available  # 导入可用性检测工具 # 导入可用性检测函数

if is_flashinfer_available():  # 如果FlashInfer可用 # 检查FlashInfer是否可用
    import flashinfer.rope as _flashinfer_rope  # 导入FlashInfer RoPE模块 # 导入FlashInfer旋转位置编码

if is_tokenspeed_mla_available():  # 如果tokenspeed_mla可用 # 检查tokenspeed_mla是否可用
    import tokenspeed_mla  # 导入tokenspeed_mla库 # 导入tokenspeed_mla库

if TYPE_CHECKING:  # 如果是类型检查阶段 # 类型检查时才导入
    from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力类 # 导入基数注意力类
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # 导入前向批次信息类
    from sglang.srt.model_executor.model_runner import ModelRunner  # 导入模型运行器 # 导入模型运行器类
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA  # 导入DeepseekV2 MLA注意力类 # 导入DeepseekV2 MLA注意力类

logger = logging.getLogger(__name__)  # 获取当前模块的日志器 # 获取模块日志器


# Workspace upper bound for tokenspeed_mla_decode:
#   num_sms * num_heads * max_q_len * (kv_lora_rank + 1) * sizeof(float32)
# MAX_Q_LEN=8 covers EAGLE3 num_draft_tokens=4 plus headroom.
# tokenspeed_mla_decode工作空间上限：
#   SM数量 * 注意力头数 * 最大查询长度 * (kv_lora_rank + 1) * float32字节数
# MAX_Q_LEN=8覆盖EAGLE3的num_draft_tokens=4并留有余量。
_TOKENSPEED_MAX_Q_LEN = 8  # tokenspeed最大查询长度常量 # tokenspeed最大查询长度

_g_tokenspeed_workspace: dict[torch.device, torch.Tensor] = {}  # 全局工作空间缓存字典 # 全局工作空间缓存


def _get_tokenspeed_workspace(  # 获取tokenspeed工作空间缓冲区 # 获取tokenspeed工作空间
    device: torch.device, num_heads: int, kv_lora_rank: int  # 设备、头数、KV LoRA秩 # 设备、注意力头数、KV LoRA秩
) -> torch.Tensor:  # 返回工作空间张量 # 返回工作空间张量
    needed = (  # 计算所需空间大小 # 计算所需字节数
        tokenspeed_mla.get_num_sm(device)  # 获取设备SM数量 # 获取SM数量
        * num_heads  # 乘以注意力头数 # 乘以头数
        * _TOKENSPEED_MAX_Q_LEN  # 乘以最大查询长度 # 乘以最大查询长度
        * (kv_lora_rank + 1)  # 乘以(kv_lora_rank + 1) # 乘以LoRA秩加1
        * 4  # 乘以float32字节数 # 乘以4字节
    )
    existing = _g_tokenspeed_workspace.get(device)  # 获取现有工作空间 # 获取已有工作空间
    if existing is None or existing.numel() < needed:  # 如果不存在或空间不足 # 检查是否需要重新分配
        _g_tokenspeed_workspace[device] = torch.empty(  # 分配新的工作空间 # 分配新工作空间
            needed, dtype=torch.int8, device=device  # 使用int8类型 # 使用int8类型
        )
    return _g_tokenspeed_workspace[device]  # 返回工作空间张量 # 返回工作空间


# TODO(Qiaolin-Yu): Merge this attention backend into trtllm_mla_backend.py
# once the same CuteDSL kernels in flashinfer_trtllm are stable
# and there is no performance gap compared to this backend.
# 待办(Qiaolin-Yu)：一旦flashinfer_trtllm中的相同CuTeDSL内核稳定
# 且与此后端无性能差距，将此后端合并到trtllm_mla_backend.py中。
class TokenspeedMLABackend(TRTLLMMLABackend):  # tokenspeed-mla CuTe DSL注意力后端类 # tokenspeed MLA注意力后端类
    """tokenspeed-mla CuTe DSL attention backend (Blackwell SM100, FP8 KV)."""
    # tokenspeed-mla CuTe DSL注意力后端（Blackwell SM100，FP8 KV缓存）。

    def __init__(  # 初始化方法 # 初始化方法
        self,
        model_runner: "ModelRunner",  # 模型运行器 # 模型运行器实例
        skip_prefill: bool = False,  # 是否跳过预填充 # 是否跳过预填充阶段
        kv_indptr_buf: Optional[torch.Tensor] = None,  # KV索引指针缓冲区 # KV索引指针缓冲区（可选）
        q_indptr_decode_buf: Optional[torch.Tensor] = None,  # 解码查询索引指针缓冲区 # 解码查询索引指针缓冲区（可选）
    ):
        super().__init__(  # 调用父类初始化 # 调用父类初始化
            model_runner,  # 模型运行器 # 传入模型运行器
            skip_prefill,  # 是否跳过预填充 # 传入跳过预填充标志
            kv_indptr_buf,  # KV索引指针缓冲区 # 传入KV索引指针缓冲区
            q_indptr_decode_buf,  # 解码查询索引指针缓冲区 # 传入解码查询索引指针缓冲区
        )

        if self.data_type != torch.float8_e4m3fn:  # 如果数据类型不是FP8 # 检查数据类型是否为FP8
            raise ValueError(  # 抛出数值错误 # 抛出异常
                "tokenspeed_mla backend requires --kv-cache-dtype fp8_e4m3, "  # 错误提示 # 错误提示
                f"got data_type={self.data_type}."  # 实际数据类型 # 显示实际类型
            )
        if self.page_size not in (32, 64):  # 如果页大小不在支持范围内 # 检查页大小
            raise ValueError(  # 抛出数值错误 # 抛出异常
                "tokenspeed_mla backend requires page_size in {32, 64}, "  # 错误提示 # 错误提示
                f"got page_size={self.page_size}."  # 实际页大小 # 显示实际页大小
            )

        self._tokenspeed_workspace: Optional[torch.Tensor] = None  # 初始化工作空间为空 # 初始化工作空间
        if is_tokenspeed_mla_available():  # 如果tokenspeed_mla可用 # 检查tokenspeed_mla可用性
            self._tokenspeed_workspace = _get_tokenspeed_workspace(  # 获取工作空间 # 获取工作空间
                self.device, self.num_q_heads, self.kv_lora_rank  # 传入设备、查询头数、KV LoRA秩 # 传入参数
            )

            # Pre-JIT the prefill kernel variants. Each cute.compile takes 1-2
            # min; without warm-up the first request trips the 300 s scheduler
            # watchdog.
            # 预先JIT编译预填充内核变体。每次cute.compile需要1-2分钟；
            # 不预热的话，第一个请求会触发300秒调度器看门狗。
            _compile_prefill_kernel = tokenspeed_mla.mla_prefill._compile_prefill_kernel  # 获取编译函数 # 获取预填充内核编译函数
            _compiled_kernels = tokenspeed_mla.mla_prefill._compiled_kernels  # 获取已编译内核缓存 # 获取已编译内核字典
            head_dim_qk = self.qk_nope_head_dim + self.qk_rope_head_dim  # 计算QK头维度 # 计算QK总头维度
            enable_ex2_emulation = tokenspeed_mla.mla_prefill._enable_ex2_emulation()  # 检查是否启用ex2仿真 # 检查ex2仿真
            use_pdl = is_arch_support_pdl()  # 检查是否支持PDL # 检查PDL支持
            for is_causal in (True, False):  # 遍历因果/非因果模式 # 遍历因果标志
                for return_lse in (True, False):  # 遍历是否返回LSE # 遍历返回LSE标志
                    # Non-causal is only entered from the chunked-prefix
                    # branch, which always asks for the LSE.
                    # 非因果模式仅从分块前缀分支进入，该分支总是需要LSE。
                    if is_causal is False and return_lse is False:  # 如果非因果且不返回LSE # 跳过无效组合
                        continue  # 跳过 # 跳过
                    # Runtime feeds fp8_e4m3fn q/k/v
                    # 运行时输入FP8 E4M3格式的Q/K/V
                    config = (  # 构建内核配置元组 # 构建配置
                        torch.float8_e4m3fn,  # FP8数据类型 # FP8数据类型
                        head_dim_qk,  # QK头维度 # QK头维度
                        self.v_head_dim,  # V头维度 # V头维度
                        is_causal,  # 因果标志 # 因果标志
                        return_lse,  # 是否返回LSE # 是否返回LSE
                        use_pdl,  # 是否使用PDL # 是否使用PDL
                        enable_ex2_emulation,  # 是否启用ex2仿真 # 是否启用ex2仿真
                    )
                    if config in _compiled_kernels:  # 如果配置已编译 # 检查是否已编译
                        continue  # 跳过 # 跳过
                    _compiled_kernels[config] = _compile_prefill_kernel(  # 编译并缓存内核 # 编译预填充内核
                        torch.float8_e4m3fn,  # FP8数据类型 # FP8数据类型
                        head_dim_qk,  # QK头维度 # QK头维度
                        self.v_head_dim,  # V头维度 # V头维度
                        is_causal,  # 因果标志 # 因果标志
                        return_lse,  # 是否返回LSE # 是否返回LSE
                        use_pdl=use_pdl,  # 是否使用PDL # PDL标志
                        enable_ex2_emulation=enable_ex2_emulation,  # 是否启用ex2仿真 # ex2仿真标志
                    )

    def _fused_rope_fp8_quantize(  # 融合RoPE和FP8量化方法 # 融合旋转位置编码与FP8量化
        self,
        q_nope: torch.Tensor,  # 查询非位置编码部分 # 查询非RoPE部分
        q_pe: torch.Tensor,  # 查询位置编码部分 # 查询RoPE部分
        k_nope: torch.Tensor,  # 键非位置编码部分 # 键非RoPE部分
        k_pe: torch.Tensor,  # 键位置编码部分 # 键RoPE部分
        cos_sin_cache: torch.Tensor,  # 余弦正弦缓存 # 余弦正弦缓存
        positions: torch.Tensor,  # 位置索引 # 位置索引
        is_neox: bool,  # 是否为Neox风格 # 是否Neox风格
        qk_nope_head_dim: int,  # QK非位置编码头维度 # QK非RoPE头维度
        qk_rope_head_dim: int,  # QK位置编码头维度 # QK RoPE头维度
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回FP8量化的Q和K元组 # 返回FP8量化的Q和K
        """Fused RoPE + FP8 quantize that also packs nope+pe along the last
        dim, so FMHA consumes contig FP8 Q/K without an extra concat or cast.
        """
        # 融合RoPE + FP8量化，同时沿最后一维打包nope+pe，
        # 使FMHA无需额外拼接或类型转换即可消费连续的FP8 Q/K。
        num_heads = q_nope.shape[1]  # 获取注意力头数 # 获取头数
        seq_len = q_nope.shape[0]  # 获取序列长度 # 获取序列长度
        q_fp8 = torch.empty(  # 分配FP8查询张量 # 分配FP8查询输出
            (seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim),  # 形状 # 张量形状
            dtype=torch.float8_e4m3fn,  # FP8数据类型 # FP8数据类型
            device=q_nope.device,  # 设备 # 设备
        )
        k_fp8 = torch.empty(  # 分配FP8键张量 # 分配FP8键输出
            (seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim),  # 形状 # 张量形状
            dtype=torch.float8_e4m3fn,  # FP8数据类型 # FP8数据类型
            device=k_nope.device,  # 设备 # 设备
        )
        if seq_len == 0:  # 如果序列长度为0 # 检查空序列
            return q_fp8, k_fp8  # 返回空张量 # 返回空张量

        # Broadcast the shared latent k_pe across heads — RoPE is position-only
        # so per-head outputs are identical, and the cache write below reuses
        # head 0.
        # 将共享的潜在k_pe广播到所有头——RoPE仅依赖位置，
        # 因此每个头的输出相同，下面的缓存写入复用头0。
        if k_pe.dim() == 3 and k_pe.shape[1] == 1:  # 如果k_pe是3D且第1维为1 # 检查是否需要广播
            k_pe_expanded = k_pe.expand(-1, num_heads, -1)  # 扩展k_pe到所有头 # 广播到所有头
        else:  # 否则 # 否则
            k_pe_expanded = k_pe  # 直接使用原始k_pe # 直接使用

        _flashinfer_rope.mla_rope_quantize_fp8(  # 调用融合RoPE+FP8量化内核 # 调用融合RoPE量化内核
            q_rope=q_pe,  # 查询RoPE部分 # 查询RoPE输入
            k_rope=k_pe_expanded,  # 键RoPE部分（已广播） # 键RoPE输入
            q_nope=q_nope,  # 查询非RoPE部分 # 查询非RoPE输入
            k_nope=k_nope,  # 键非RoPE部分 # 键非RoPE输入
            cos_sin_cache=cos_sin_cache,  # 余弦正弦缓存 # 余弦正弦缓存
            pos_ids=positions,  # 位置ID # 位置ID
            is_neox=is_neox,  # 是否Neox风格 # Neox风格标志
            quantize_dtype=torch.float8_e4m3fn,  # 量化数据类型 # 量化目标类型
            q_rope_out=q_fp8[..., qk_nope_head_dim:],  # 查询RoPE输出切片 # 查询RoPE输出
            k_rope_out=k_fp8[..., qk_nope_head_dim:],  # 键RoPE输出切片 # 键RoPE输出
            q_nope_out=q_fp8[..., :qk_nope_head_dim],  # 查询非RoPE输出切片 # 查询非RoPE输出
            k_nope_out=k_fp8[..., :qk_nope_head_dim],  # 键非RoPE输出切片 # 键非RoPE输出
            quant_scale_q=1.0,  # 查询量化缩放因子 # 查询量化比例
            quant_scale_kv=1.0,  # KV量化缩放因子 # KV量化比例
            enable_pdl=is_arch_support_pdl(),  # 是否启用PDL # PDL支持标志
        )
        return q_fp8, k_fp8  # 返回FP8量化的Q和K # 返回FP8量化的Q和K

    def prepare_prefill_qkv(  # 准备预填充阶段的FP8 Q/K/V # 准备预填充QKV
        self,
        *,  # 强制关键字参数 # 强制关键字参数
        q: torch.Tensor,  # 查询张量 # 查询张量
        q_pe: torch.Tensor,  # 查询位置编码 # 查询RoPE部分
        kv_a: torch.Tensor,  # KV压缩激活值 # KV压缩激活
        k_pe: torch.Tensor,  # 键位置编码 # 键RoPE部分
        positions: torch.Tensor,  # 位置索引 # 位置索引
        layer: "DeepseekV2AttentionMLA",  # MLA注意力层 # MLA注意力层
        forward_batch: "ForwardBatch",  # 前向批次 # 前向批次信息
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回FP8的(Q, K, V)元组 # 返回FP8 QKV
        """Build FP8 (Q, K, V) for the FMHA kernel and write FP8 KV cache."""
        # 为FMHA内核构建FP8格式的(Q, K, V)并写入FP8 KV缓存。
        kv = layer.kv_b_proj(kv_a)[0]  # 通过kv_b_proj投影KV # KV投影
        kv = kv.view(  # 重塑KV形状 # 重塑KV形状
            -1, layer.num_local_heads, layer.qk_nope_head_dim + layer.v_head_dim  # 形状参数 # 形状
        )
        k_nope = kv[..., : layer.qk_nope_head_dim]  # 提取键非RoPE部分 # 提取键非RoPE部分
        v_bf16 = kv[..., layer.qk_nope_head_dim :]  # 提取值（BF16格式） # 提取值
        q_nope = q[..., : layer.qk_nope_head_dim]  # 提取查询非RoPE部分 # 提取查询非RoPE部分

        q_fp8, k_fp8 = self._fused_rope_fp8_quantize(  # 执行融合RoPE+FP8量化 # 融合RoPE和FP8量化
            q_nope=q_nope,  # 查询非RoPE # 查询非RoPE
            q_pe=q_pe,  # 查询RoPE # 查询RoPE
            k_nope=k_nope,  # 键非RoPE # 键非RoPE
            k_pe=k_pe,  # 键RoPE # 键RoPE
            cos_sin_cache=layer.rotary_emb.cos_sin_cache,  # 余弦正弦缓存 # 余弦正弦缓存
            positions=positions,  # 位置索引 # 位置索引
            is_neox=getattr(layer.rotary_emb, "is_neox_style", True),  # 是否Neox风格 # Neox风格标志
            qk_nope_head_dim=layer.qk_nope_head_dim,  # QK非RoPE头维度 # QK非RoPE头维度
            qk_rope_head_dim=layer.qk_rope_head_dim,  # QK RoPE头维度 # QK RoPE头维度
        )
        v_fp8 = fp8_quantize(v_bf16, enable_pdl=is_arch_support_pdl())  # 对V进行FP8量化 # 对V进行FP8量化

        # k_pe is shared across heads (RoPE is position-only), so head 0
        # reproduces the original [tokens, 1, qk_rope] latent layout.
        # k_pe在所有头之间共享（RoPE仅依赖位置），因此头0复现了
        # 原始的[tokens, 1, qk_rope]潜在布局。
        kv_a_fp8 = fp8_quantize(kv_a, enable_pdl=is_arch_support_pdl())  # 对KV压缩激活进行FP8量化 # 对KV压缩激活进行FP8量化
        k_pe_fp8 = k_fp8[:, 0:1, layer.qk_nope_head_dim :]  # 提取头0的键RoPE部分 # 提取头0的键RoPE FP8
        self.token_to_kv_pool.set_mla_kv_buffer(  # 写入MLA KV缓存 # 写入MLA KV缓存
            layer.attn_mha,  # 注意力层 # 注意力层
            forward_batch.out_cache_loc,  # 输出缓存位置 # 输出缓存位置
            kv_a_fp8.unsqueeze(1),  # KV压缩激活（增加维度） # KV压缩激活（扩展维度）
            k_pe_fp8,  # 键RoPE FP8 # 键RoPE FP8
        )
        return q_fp8, k_fp8, v_fp8  # 返回FP8 Q/K/V # 返回FP8 QKV

    def pack_prefix_chunk_kv(  # 打包前缀分块的KV # 打包前缀分块KV
        self,
        k_nope: torch.Tensor,  # 键非RoPE部分 # 键非RoPE部分
        k_pe: torch.Tensor,  # 键RoPE部分 # 键RoPE部分
        v: torch.Tensor,  # 值 # 值
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回打包后的FP8 K和V元组 # 返回打包后的FP8 KV
        """Pack strided ``k_nope``+``k_pe`` into contig FP8 K and quantize
        strided ``v`` into contig FP8 V in a single kernel.
        """
        # 在单个内核中将步进式的k_nope+k_pe打包为连续FP8 K，
        # 并将步进式的v量化为连续FP8 V。
        return mla_kv_pack_quantize_fp8(  # 调用MLA KV打包量化内核 # 调用MLA KV打包量化
            k_nope, k_pe, v, enable_pdl=is_arch_support_pdl()  # 传入参数 # 传入参数
        )

    def _run_decode_kernel(  # 运行解码内核 # 运行解码内核
        self,
        query: torch.Tensor,  # 查询张量 # 查询张量
        kv_cache: torch.Tensor,  # KV缓存张量 # KV缓存张量
        block_tables: torch.Tensor,  # 块表张量 # 块表张量
        seq_lens: torch.Tensor,  # 序列长度张量 # 序列长度张量
        max_seq_len: int,  # 最大序列长度 # 最大序列长度
        layer: "RadixAttention",  # 注意力层 # 注意力层
    ) -> torch.Tensor:  # 返回输出张量 # 返回输出张量
        k_scale = getattr(layer, "k_scale_float", None)  # 获取键缩放因子 # 获取键缩放因子
        if k_scale is None:  # 如果缩放因子为空 # 检查缩放因子
            k_scale = 1.0  # 默认缩放因子为1.0 # 默认值1.0
        softmax_scale = float(layer.scaling) * float(k_scale)  # 计算softmax缩放因子 # 计算softmax缩放
        output_scale = float(k_scale)  # 计算输出缩放因子 # 输出缩放

        seq_lens_i32 = (  # 将序列长度转换为int32 # 转换序列长度类型
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)  # 如果已是int32则直接使用 # 条件转换
        )
        return tokenspeed_mla.tokenspeed_mla_decode(  # 调用tokenspeed MLA解码内核 # 调用tokenspeed MLA解码内核
            query=query,  # 查询张量 # 查询
            kv_cache=kv_cache,  # KV缓存 # KV缓存
            workspace_buffer=self._tokenspeed_workspace,  # 工作空间缓冲区 # 工作空间
            kv_lora_rank=self.kv_lora_rank,  # KV LoRA秩 # KV LoRA秩
            qk_rope_head_dim=self.qk_rope_head_dim,  # QK RoPE头维度 # QK RoPE头维度
            block_tables=block_tables,  # 块表 # 块表
            seq_lens=seq_lens_i32,  # 序列长度（int32） # 序列长度
            max_seq_len=int(max_seq_len),  # 最大序列长度 # 最大序列长度
            softmax_scale=softmax_scale,  # softmax缩放因子 # softmax缩放
            output_scale=output_scale,  # 输出缩放因子 # 输出缩放
            enable_pdl=is_arch_support_pdl(),  # 是否启用PDL # PDL标志
        )

    def _run_prefill_kernel(  # 运行预填充内核 # 运行预填充内核
        self,
        q: torch.Tensor,  # 查询张量 # 查询张量
        k: torch.Tensor,  # 键张量 # 键张量
        v: torch.Tensor,  # 值张量 # 值张量
        layer: "RadixAttention",  # 注意力层 # 注意力层
        batch_size: int,  # 批次大小 # 批次大小
        cum_seq_lens_q: torch.Tensor,  # 查询累积序列长度 # 查询累积序列长度
        max_q_len: int,  # 最大查询长度 # 最大查询长度
        seq_lens_kv: torch.Tensor,  # KV序列长度 # KV序列长度
        cum_seq_lens_kv: torch.Tensor,  # KV累积序列长度 # KV累积序列长度
        max_kv_len: int,  # 最大KV长度 # 最大KV长度
        is_causal: bool,  # 是否因果注意力 # 是否因果
        return_lse: bool,  # 是否返回LSE # 是否返回LSE
        out_buffer: torch.Tensor,  # 输出缓冲区 # 输出缓冲区
        o_sf_scale: float = 1.0,  # 输出缩放因子 # 输出缩放因子
    ):  # Q/K/V arrive already in FP8 via the model-side fused path
        # (prepare_prefill_qkv / pack_prefix_chunk_kv); no quantize here.
        # Q/K/V已通过模型侧融合路径（prepare_prefill_qkv / pack_prefix_chunk_kv）
        # 以FP8格式传入；此处无需量化。
        return tokenspeed_mla.tokenspeed_mla_prefill(  # 调用tokenspeed MLA预填充内核 # 调用tokenspeed MLA预填充内核
            query=q,  # 查询张量 # 查询
            key=k,  # 键张量 # 键
            value=v,  # 值张量 # 值
            seq_lens=seq_lens_kv,  # KV序列长度 # KV序列长度
            cum_seq_lens=cum_seq_lens_kv,  # KV累积序列长度 # KV累积序列长度
            max_seq_len=int(max_kv_len),  # 最大KV长度 # 最大KV长度
            batch_size=int(batch_size),  # 批次大小 # 批次大小
            softmax_scale=float(layer.scaling),  # softmax缩放因子 # softmax缩放
            is_causal=is_causal,  # 是否因果 # 因果标志
            return_lse=return_lse,  # 是否返回LSE # 返回LSE标志
            cum_seq_lens_q=cum_seq_lens_q,  # 查询累积序列长度 # 查询累积序列长度
            max_seq_len_q=int(max_q_len),  # 最大查询长度 # 最大查询长度
            enable_pdl=is_arch_support_pdl(),  # 是否启用PDL # PDL标志
        )


class TokenspeedMLAMultiStepDraftBackend(TRTLLMMLAMultiStepDraftBackend):  # tokenspeed MLA多步草稿后端类 # tokenspeed MLA多步草稿后端
    """Multi-step draft backend for tokenspeed_mla used by EAGLE."""
    # EAGLE使用的tokenspeed_mla多步草稿后端。

    def __init__(  # 初始化方法 # 初始化方法
        self, model_runner: "ModelRunner", topk: int, speculative_num_steps: int  # 模型运行器、topk值、投机步数 # 模型运行器、topk值、投机步数
    ):
        super().__init__(model_runner, topk, speculative_num_steps)  # 调用父类初始化 # 调用父类初始化
        # Parent populates self.attn_backends with TRT-LLM instances; replace
        # them with tokenspeed instances sharing the parent's index buffers.
        # 父类用TRT-LLM实例填充self.attn_backends；将其替换为共享父类索引缓冲区的tokenspeed实例。
        for i in range(self.speculative_num_steps - 1):  # 遍历投机步数 # 遍历投机步数
            self.attn_backends[i] = TokenspeedMLABackend(  # 替换为tokenspeed后端实例 # 替换为tokenspeed后端
                model_runner,  # 模型运行器 # 模型运行器
                skip_prefill=True,  # 跳过预填充 # 跳过预填充
                kv_indptr_buf=self.kv_indptr[i],  # KV索引指针缓冲区 # KV索引指针缓冲区
                q_indptr_decode_buf=self.q_indptr_decode,  # 解码查询索引指针缓冲区 # 解码查询索引指针缓冲区
            )
