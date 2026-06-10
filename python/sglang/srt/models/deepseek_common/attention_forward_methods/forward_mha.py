# DeepSeek模型MHA（多头注意力）前向计算方法模块
# 实现了DeepSeek V3/R1模型中MHA模式下的多种前向计算策略，包括：
# 1. 普通MHA前向（无KV缓存前缀）
# 2. 单次MHA前向（短KV前缀，一次完成）
# 3. 分块KV的MHA前向（长KV前缀，分块处理后合并）
# 同时包含KV缓存读写、RoPE位置编码、FP8/MXFP4量化融合等辅助功能

from __future__ import annotations  # 启用延迟类型注解求值，允许在类型提示中使用尚未定义的类型

from typing import TYPE_CHECKING  # 导入类型检查常量

import torch  # 导入PyTorch深度学习框架

from sglang.srt.environ import envs  # 从sglang运行时环境模块导入环境变量配置
from sglang.srt.layers.attention.dsa.dequant_k_cache import dequantize_k_cache_paged  # 导入分页式FP8 KV缓存反量化函数
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend  # 导入TBO（Token级批处理优化）注意力后端
from sglang.srt.layers.attention.utils import concat_and_cast_mha_k_triton  # 导入Triton实现的MHA K拼接与类型转换工具
from sglang.srt.layers.communicator import get_attn_tp_context  # 导入注意力张量并行上下文获取函数
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息类
from sglang.srt.model_executor.forward_context import (  # 导入前向上下文相关函数
    get_attn_backend,  # 获取当前注意力后端
    get_token_to_kv_pool,  # 获取token到KV缓存池的映射
)
from sglang.srt.models.deepseek_common.utils import (  # 导入DeepSeek通用工具函数
    _is_cuda,  # 判断是否为CUDA平台
    _is_hip,  # 判断是否为HIP(AMD ROCm)平台
    _is_musa,  # 判断是否为Moore Threads MUSA平台
    _is_npu,  # 判断是否为华为NPU平台
    _use_aiter_gfx95,  # 判断是否使用aiter gfx95优化
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数
from sglang.srt.utils import BumpAllocator, get_bool_env_var, next_power_of_2  # 导入凸分配器、布尔环境变量获取和2的幂计算工具

_use_fp8_prefill_attn = (  # 是否使用FP8预填充注意力
    get_bool_env_var("SGLANG_AITER_FP8_PREFILL_ATTN", "True") and _use_aiter_gfx95  # 默认启用且平台为aiter gfx95
)

if TYPE_CHECKING:  # 仅在类型检查时导入，运行时不导入
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA  # 导入DeepSeek V2 MLA注意力类，用于类型提示

if _is_cuda:  # 如果是CUDA平台
    from sgl_kernel import merge_state_v2  # 导入sgl_kernel中的状态合并v2算子

    from sglang.jit_kernel.concat_mla import concat_mla_k  # 导入JIT编译的MLA K拼接内核
elif _is_musa:  # 如果是MUSA平台
    from sgl_kernel import concat_mla_k  # 从sgl_kernel导入MLA K拼接算子

if _use_aiter_gfx95:  # 如果使用aiter gfx95优化
    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant  # 导入融合RMS归一化+FP8分组量化算子

    from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype  # 导入FP8数据类型定义
    from sglang.srt.layers.quantization.rocm_mxfp4_utils import fused_rms_mxfp4_quant  # 导入融合RMS归一化+MXFP4量化算子


def _resolve_attn_backend(forward_batch: ForwardBatch):  # 解析实际使用的注意力后端，处理TBO包装情况
    backend = get_attn_backend()  # 获取当前注意力后端
    if isinstance(backend, TboAttnBackend):  # 如果后端是TBO包装器
        backend = backend.primary  # 获取其内部的主要后端
    return backend  # 返回解析后的后端


# Configs for DeepSeek-V3:  # DeepSeek-V3的配置参数：
# num_local_heads = 128  # 本地注意力头数为128
# qk_nope_head_dim = 128  # Q/K非位置编码头维度为128
# qk_rope_head_dim = 64  # Q/K旋转位置编码头维度为64
# qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192  # Q/K总头维度为192
# v_head_dim = 128  # V头维度为128

# Configs for kv chunking strategy:  # KV分块策略的配置参数：
# sum_prefix_length:  # 前缀总长度：
#   Total number of tokens to be fetched from kv cache for current batch.  # 当前批次需要从KV缓存中获取的token总数
#   e.g: For batch with 2 sequences, seq_lens_kv = [1024, 2048], seq_lens_q = [512, 1024], then sum_prefix_length = (1024 - 512) + (2048 - 1024) = 1536  # 例如：2个序列的批次，KV长度=[1024,2048]，Q长度=[512,1024]，前缀总长度=1536
# sum_extended_length:  # 扩展部分总长度：
#   Total number of tokens in the extended part of the current batch. (=sum(seq_lens_q))  # 当前批次扩展部分的token总数（等于Q长度之和）
# chunked_prefix_cache_threshold:  # 分块前缀缓存阈值：
#   The minimum sum_prefix_length to enable mha with kv chunking, 8192 by default (can be changed with SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD)  # 启用KV分块MHA的最小前缀总长度，默认8192
#   For batches with smaller sum_prefix_length > 0, MLA kernel with absorption will be used instead.  # 对于前缀总长度较小(>0)的批次，将使用带吸收的MLA内核
# max_kv_chunk_capacity:  # 最大KV分块容量：
#   The maximum number of tokens in each kv chunk, 128 * 1024 by default (can be changed with SGLANG_MAX_KV_CHUNK_CAPACITY, or get with forward_batch.get_max_chunk_capacity())  # 每个KV分块的最大token数，默认128*1024

# The forward methods for MHA in DeepSeek models:  # DeepSeek模型中MHA的前向方法：
#
# 1. forward_normal: AttnForwardMethod.MHA  # 1. 普通前向：MHA方法
#    use multi-head attention with empty kv cache (the first batch of chunked prefill, prefix lens = 0)  # 使用空KV缓存的多头注意力（分块预填充的第一个批次，前缀长度=0）
#    q: [sum_extended_length, num_local_heads, qk_head_dim]  # Q张量形状
#    k: [sum_extended_length, num_local_heads, qk_head_dim]  # K张量形状
#    v: [sum_extended_length, num_local_heads, v_head_dim]  # V张量形状
#
# 2. forward_normal_one_shot: AttnForwardMethod.MHA_ONE_SHOT  # 2. 单次前向：MHA一次完成方法
#    use multi-head attention with short kv prefix length (chunked_prefix_cache_threshold <= sum_prefix_lens <= max_kv_chunk_capacity)  # 使用短KV前缀长度的多头注意力（阈值 <= 前缀长度 <= 最大容量）
#    the kv latent vectors are fetched from memory pool, with combined kv_indices of prefix part and extended part  # 从内存池获取KV潜向量，使用前缀和扩展部分的合并KV索引
#    q: [batch_size, num_local_heads, qk_head_dim]  # Q张量形状
#    k: [sum_extended_length + sum_prefix_length, num_local_heads, qk_head_dim]  # K张量形状（含前缀）
#    v: [sum_extended_length + sum_prefix_length, num_local_heads, v_head_dim]  # V张量形状（含前缀）
#
# 3. forward_normal_chunked_kv: AttnForwardMethod.MHA_CHUNKED_KV  # 3. 分块KV前向：MHA分块KV方法
#    multiple phases of multi-head attention with chunked kv cache (sum_prefix_length > max_kv_chunk_capacity)  # 多阶段分块KV缓存多头注意力（前缀长度 > 最大容量）
#    For the first phase, it will execute normal forward method, and returns output o_1 and lse_1,  # 第一阶段执行普通前向，返回输出o_1和lse_1
#       q_1: [sum_extended_length, num_local_heads, qk_head_dim],  # 第一阶段Q
#       k_1: [sum_extended_length, num_local_heads, qk_head_dim],  # 第一阶段K
#       v_1: [sum_extended_length, num_local_heads, v_head_dim],  # 第一阶段V
#       acc_o_1, acc_lse_1 = o_1, lse_1  # 累积输出和lse
#    For i in range(2, n), (n-1 is the number of prefix chunks), kv latent vectors are fetched from memory pool with prefix kv indices  # 对于第2到n-1阶段（n-1为前缀分块数），使用前缀KV索引从内存池获取KV潜向量
#       q_i: [sum_extended_length, num_local_heads, qk_head_dim],  # 第i阶段Q
#       k_i: [chunk_size, num_local_heads, qk_head_dim],  # 第i阶段K
#       v_i: [chunk_size, num_local_heads, v_head_dim],  # 第i阶段V
#       acc_o_i, acc_lse_i = merge_state(acc_o_{i-1}, acc_lse_{i-1}, o_i, lse_i)  # 合并累积状态
#       The final output is the accumulated output acc_o_n  # 最终输出为累积输出acc_o_n


class DeepseekMHAForwardMixin:  # DeepSeek MHA前向计算混入类，提供MHA模式下的各种前向方法

    def init_mha_forward(self: DeepseekV2AttentionMLA):  # 初始化MHA前向计算相关参数
        self.disable_chunked_prefix_cache = (  # 是否禁用分块前缀缓存
            get_global_server_args().disable_chunked_prefix_cache  # 从全局服务器参数获取
        )

        # TODO: Design a finer way to determine the threshold  # TODO：设计更精细的阈值确定方式
        self.chunked_prefix_cache_threshold = (  # 分块前缀缓存的阈值
            envs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD.get()  # 从环境变量获取
        )

    def forward_normal_prepare(  # 普通MHA前向的准备工作，计算Q/K/V张量
        self: DeepseekV2AttentionMLA,  # 类型标注为DeepseekV2AttentionMLA
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
        zero_allocator: BumpAllocator,  # 零分配器
    ):
        if self.q_lora_rank is not None:  # 如果使用Q的LoRA秩（MLA模式）
            q, latent_cache = (  # 从注意力TP上下文获取QKV潜向量并分割
                get_attn_tp_context()  # 获取注意力张量并行上下文
                .fetch_qkv_latent()  # 获取QKV潜向量
                .split(  # 按维度分割
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],  # 分割点：Q LoRA维度 和 KV LoRA+RoPE维度
                    dim=-1,  # 在最后一维分割
                )
            )

            # DSA Indexer: cache quantized keys, auto-skip topk for sequences <= dsa_index_topk  # DSA索引器：缓存量化键，对短序列自动跳过topk

            if self.use_dsa:  # 如果使用动态稀疏注意力(DSA)
                # DSA requires unquantized q_lora for the indexer. When q_b_proj is FP8  # DSA需要未量化的q_lora用于索引器。当q_b_proj为FP8时
                # on gfx95, we can still use fused RMSNorm+FP8 quant, but MUST request  # 在gfx95上，仍可使用融合RMSNorm+FP8量化，但必须请求
                # the unquantized output for q_lora; otherwise q_lora becomes the (fp8,scale)  # q_lora的未量化输出；否则q_lora变成(fp8,scale)元组
                # tuple.
                if (  # 如果使用aiter gfx95且Q投影权重为FP8类型
                    _use_aiter_gfx95
                    and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                ):
                    q_quanted, q_lora, _, _ = fused_rms_fp8_group_quant(  # 执行融合RMS归一化+FP8分组量化，同时获取未量化的q_lora
                        q,  # 输入Q潜向量
                        self.q_a_layernorm.weight,  # Q的a层归一化权重
                        self.q_a_layernorm.variance_epsilon,  # 归一化的方差epsilon
                        None,  # 无残差输入
                        None,  # 无额外缩放
                        None,  # 无额外偏移
                        group_size=128,  # 量化分组大小
                        dtype_quant=torch.float8_e4m3fn,  # 量化目标数据类型为FP8
                        res1=None,  # 无残差1
                        output_unquantized_inp1=True,  # 输出未量化的inp1（q_lora），DSA需要
                    )
                    q = self.q_b_proj(q_quanted)[0].view(  # 对量化后的Q进行b投影并重塑形状
                        -1, self.num_local_heads, self.qk_head_dim  # 形状为[token数, 头数, 头维度]
                    )
                else:  # 不使用aiter gfx95+FP8的普通路径
                    q_lora = self.q_a_layernorm(q)  # 对Q潜向量进行a层归一化
                    q = self.q_b_proj(q_lora)[0].view(  # 对归一化后的Q进行b投影并重塑形状
                        -1, self.num_local_heads, self.qk_head_dim  # 形状为[token数, 头数, 头维度]
                    )
                _ = self.indexer(  # 调用DSA索引器进行索引计算
                    x=hidden_states,  # 输入隐藏状态
                    q_lora=q_lora,  # Q的LoRA表示，用于索引器
                    positions=positions,  # 位置编码
                    forward_batch=forward_batch,  # 前向批次信息
                    layer_id=self.layer_id,  # 当前层ID
                    return_indices=False,  # 不返回索引结果
                )
            elif _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:  # 如果使用aiter gfx95且权重为MXFP4(uint8)类型
                # MXFP4: fused RMSNorm + quant  # MXFP4：融合RMS归一化+量化
                q, _, _, _ = fused_rms_mxfp4_quant(  # 执行融合RMS归一化+MXFP4量化
                    q,  # 输入Q潜向量
                    self.q_a_layernorm.weight,  # Q的a层归一化权重
                    self.q_a_layernorm.variance_epsilon,  # 归一化的方差epsilon
                    None,  # 无残差输入
                    None,  # 无额外缩放
                    None,  # 无额外偏移
                )
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)  # MXFP4量化后投影并重塑形状
            elif _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.float8_e4m3fn:  # 如果使用aiter gfx95且权重为FP8类型（非DSA路径）

                q, _, _, _ = fused_rms_fp8_group_quant(  # 执行融合RMS归一化+FP8分组量化
                    q,  # 输入Q潜向量
                    self.q_a_layernorm.weight,  # Q的a层归一化权重
                    self.q_a_layernorm.variance_epsilon,  # 归一化的方差epsilon
                    None,  # 无残差输入
                    None,  # 无额外缩放
                    None,  # 无额外偏移
                    group_size=128,  # 量化分组大小
                    dtype_quant=torch.float8_e4m3fn,  # 量化目标数据类型为FP8
                    res1=None,  # 无残差1
                    output_unquantized_inp1=False,  # 不输出未量化的inp1
                )
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)  # FP8量化后投影并重塑形状
            else:  # 普通路径（不使用量化融合）
                q = self.q_a_layernorm(q)  # 对Q潜向量进行a层归一化
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)  # 投影并重塑形状

        else:  # 不使用Q LoRA秩（标准MHA模式）
            q = self.q_proj(hidden_states)[0].view(  # 直接对隐藏状态进行Q投影并重塑形状
                -1, self.num_local_heads, self.qk_head_dim  # 形状为[token数, 头数, 头维度]
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]  # 使用MQA的KV投影获取潜缓存

        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)  # 将Q分割为非位置编码部分和旋转位置编码部分
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)  # 将潜缓存分割为KV LoRA部分和RoPE部分
        latent_cache = latent_cache.unsqueeze(1)  # 在第1维增加维度，用于后续KV缓存存储

        if _use_aiter_gfx95 and self.kv_b_proj.weight.dtype == torch.float8_e4m3fn:  # 如果使用aiter gfx95且KV投影权重为FP8类型

            kv_a_quanted, kv_a, _, _ = fused_rms_fp8_group_quant(  # 对KV LoRA执行融合RMS归一化+FP8量化，同时保留未量化版本
                kv_a,  # 输入KV LoRA向量
                self.kv_a_layernorm.weight,  # KV的a层归一化权重
                self.kv_a_layernorm.variance_epsilon,  # 归一化的方差epsilon
                None,  # 无残差输入
                None,  # 无额外缩放
                None,  # 无额外偏移
                group_size=128,  # 量化分组大小
                dtype_quant=torch.float8_e4m3fn,  # 量化目标数据类型为FP8
                res1=None,  # 无残差1
                output_unquantized_inp1=True,  # return unqaunt kv_a  # 返回未量化的kv_a，后续one-shot路径需要
            )

        else:  # 普通路径（不使用FP8量化融合）
            kv_a = self.kv_a_layernorm(kv_a)  # 对KV LoRA向量进行a层归一化

        k_pe = latent_cache[:, :, self.kv_lora_rank :]  # 从潜缓存中提取K的旋转位置编码部分

        # Backend prefill hook: the backend owns the BF16->FP8 transition  # 后端预填充钩子：后端负责BF16到FP8的转换
        # (fused RoPE + quantize for Q/K, direct FP8 KV-cache write) and  # （融合RoPE + Q/K量化，直接写入FP8 KV缓存）
        # returns FP8 tensors ready for its kernel. Backends without the  # 返回准备好的FP8张量供其内核使用
        # hook fall through to the BF16 path below.  # 没有此钩子的后端回退到下面的BF16路径
        backend = _resolve_attn_backend(forward_batch)  # 解析实际使用的注意力后端
        if hasattr(backend, "prepare_prefill_qkv"):  # 如果后端支持预填充QKV准备钩子
            q_out, k_out, v_out = backend.prepare_prefill_qkv(  # 调用后端的预填充QKV准备函数
                q=q,  # Q张量
                q_pe=q_pe,  # Q的旋转位置编码
                kv_a=kv_a,  # KV LoRA向量
                k_pe=k_pe,  # K的旋转位置编码
                positions=positions,  # 位置编码
                layer=self,  # 当前注意力层
                forward_batch=forward_batch,  # 前向批次信息
            )
            return q_out, k_out, v_out, forward_batch  # 返回后端准备的QKV和批次信息

        if self.rotary_emb is not None:  # 如果存在旋转位置编码模块
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)  # 应用旋转位置编码到Q和K的RoPE部分
        q[..., self.qk_nope_head_dim :] = q_pe  # 将编码后的q_pe放回Q的RoPE位置

        self._set_mla_kv_buffer(latent_cache, kv_a, k_pe, forward_batch)  # 将KV潜缓存写入KV缓存池
        if (  # 检查是否为MHA单次模式且有前缀
            forward_batch.mha_one_shot  # 是MHA单次模式
            and sum(forward_batch.extend_prefix_lens_cpu) != 0  # 且存在前缀长度
        ):
            if (  # 如果使用DSA且KV缓存为FP8格式
                self.use_dsa
                and self.kv_cache_dtype == "fp8_e4m3"
                and (
                    not get_global_server_args().dsa_decode_backend == "trtllm"  # DSA解码后端不是TRT-LLM
                    or not get_global_server_args().dsa_prefill_backend == "trtllm"  # 或DSA预填充后端不是TRT-LLM
                )
            ):
                # FP8 path: dequantize DSA-specific FP8 format to BF16  # FP8路径：将DSA专用FP8格式反量化为BF16
                kv_a, k_pe = self._get_mla_kv_buffer_from_fp8_for_dsa(forward_batch)  # 从FP8缓存反量化获取KV
            else:
                # BF16/FP16 path: directly fetch from cache  # BF16/FP16路径：直接从缓存获取
                kv_a, k_pe = self._get_mla_kv_buffer(  # 从KV缓存池获取KV潜向量
                    forward_batch.fetch_mha_one_shot_kv_indices(),  # 获取单次模式的KV索引
                    q.dtype,  # 目标数据类型
                    forward_batch,  # 前向批次信息
                )
        if _use_fp8_prefill_attn and self.kv_b_proj.weight.dtype == torch.uint8:  # 如果使用FP8预填充注意力且KV投影权重为MXFP4(uint8)类型
            # MXFP4 weights + FP8 prefill: fuse GEMM, nope/v split, and k_pe cat  # MXFP4权重 + FP8预填充：融合GEMM、nope/v分割和k_pe拼接
            # into a single kernel (fused_gemm_afp4wfp4_split_cat) that writes k and v  # 到单个内核中，直接以FP8格式写入k和v
            # directly in FP8, avoiding a separate elementwise cast  # 避免单独的逐元素类型转换
            k, v = self.kv_b_proj(  # 调用KV投影，使用融合内核直接输出FP8的K和V
                (  # 传入融合内核所需参数元组
                    kv_a,  # KV LoRA向量
                    k_pe.expand(-1, self.num_local_heads, -1),  # 扩展k_pe到所有头
                    self.qk_nope_head_dim,  # QK非位置编码头维度
                    self.v_head_dim,  # V头维度
                    fp8_dtype,  # FP8数据类型
                )
            )[0]
        else:  # 普通KV投影路径
            if _use_aiter_gfx95 and self.kv_b_proj.weight.dtype == torch.float8_e4m3fn:  # 如果使用aiter gfx95且权重为FP8
                kv = self.kv_b_proj(kv_a_quanted)[0]  # 使用量化后的KV LoRA进行投影
            else:  # 普通路径
                kv = self.kv_b_proj(kv_a)[0]  # 使用未量化的KV LoRA进行投影
            kv = kv.view(  # 重塑KV投影结果的形状
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim  # 形状为[token数, 头数, nope维度+v维度]
            )
            k_nope = kv[..., : self.qk_nope_head_dim]  # 提取K的非位置编码部分
            v = kv[..., self.qk_nope_head_dim :]  # 提取V部分

            k = self._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)  # 拼接K的nope和RoPE部分
        return q, k, v, forward_batch  # 返回Q、K、V张量和前向批次信息

    def forward_normal_core(  # 普通MHA前向的核心计算，执行注意力计算和输出投影
        self: DeepseekV2AttentionMLA,  # 类型标注为DeepseekV2AttentionMLA
        q: torch.Tensor,  # Q张量
        k: torch.Tensor,  # K张量
        v: torch.Tensor,  # V张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:  # 返回输出张量
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)  # 执行多头注意力计算，不保存KV缓存
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)  # 重塑注意力输出形状为[token数, 头数*V维度]
        output, _ = self.o_proj(attn_output)  # 执行输出投影
        return output  # 返回输出张量

    def forward_normal_chunked_kv_prepare(  # 分块KV的MHA前向准备工作，委托给普通前向准备
        self: DeepseekV2AttentionMLA,  # 类型标注为DeepseekV2AttentionMLA
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
        zero_allocator: BumpAllocator,  # 零分配器
    ):
        # In normal mha, the k and v tensors will become overly large when the prefix length is long.  # 在普通MHA中，当前缀长度很长时，K和V张量会变得过大
        # To avoid this, we split the kv cache into chunks and process them one after another.  # 为避免此问题，我们将KV缓存分块并逐个处理
        # Since mha is compute friendly, the for loop induced here will not introduce significant overhead.  # 由于MHA计算友好，此处引入的循环不会带来显著开销
        # The top comments in https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py  # vLLM仓库中MLA后端的顶部注释
        # will be helpful for understanding the purpose of this function.  # 有助于理解此函数的目的

        # First do normal mha forward to get output for extended part  # 首先执行普通MHA前向以获取扩展部分的输出
        return self.forward_normal_prepare(  # 委托给普通前向准备函数
            positions, hidden_states, forward_batch, zero_allocator  # 传递所有参数
        )

    def forward_normal_chunked_kv_core(  # 分块KV的MHA前向核心计算，处理扩展部分和分块前缀的注意力
        self: DeepseekV2AttentionMLA,  # 类型标注为DeepseekV2AttentionMLA
        q: torch.Tensor,  # Q张量
        k: torch.Tensor,  # K张量
        v: torch.Tensor,  # V张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:  # 返回输出张量
        has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(  # 检查是否存在扩展前缀
            forward_batch.extend_prefix_lens_cpu  # 前缀长度列表
        )
        # Only initialize the info once  # 仅初始化一次分块信息
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:  # 如果有前缀但分块信息未初始化
            forward_batch.prepare_chunked_prefix_cache_info(q.device)  # 准备分块前缀缓存信息
            if hasattr(get_attn_backend(), "init_mha_chunk_metadata"):  # 如果后端支持MHA分块元数据初始化
                get_attn_backend().init_mha_chunk_metadata(forward_batch)  # 初始化MHA分块元数据

        forward_batch.mha_return_lse = has_extend_prefix  # 设置是否返回对数softmax指数(lse)
        # Do mha for extended part without prefix  # 对无前缀的扩展部分执行MHA
        forward_batch.set_attn_attend_prefix_cache(False)  # 设置不关注前缀缓存
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)  # 扩展部分的MHA注意力计算

        # Do mha attention with chunked prefix cache if there are any sequence with prefix  # 如果存在有前缀的序列，则对分块前缀缓存执行MHA注意力
        if has_extend_prefix:  # 如果有扩展前缀
            attn_output, lse = attn_output  # 解构获取注意力输出和lse
            forward_batch.set_attn_attend_prefix_cache(True)  # 设置关注前缀缓存
            attn_output = self._chunked_prefix_attn_mha(  # 执行分块前缀MHA注意力并合并结果
                q=q,  # Q张量
                accum_output=attn_output,  # 累积的注意力输出
                accum_lse=lse,  # 累积的lse
                forward_batch=forward_batch,  # 前向批次信息
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)  # 重塑注意力输出形状
        output, _ = self.o_proj(attn_output)  # 执行输出投影
        return output  # 返回输出张量

    def forward_normal_one_shot_prepare(  # 单次MHA前向的准备工作，设置单次模式标志后委托给普通前向准备
        self: DeepseekV2AttentionMLA,  # 类型标注为DeepseekV2AttentionMLA
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
        zero_allocator: BumpAllocator,  # 零分配器
    ):
        forward_batch.mha_one_shot = True  # 设置MHA单次模式标志
        return self.forward_normal_prepare(  # 委托给普通前向准备函数
            positions, hidden_states, forward_batch, zero_allocator  # 传递所有参数
        )

    def forward_normal_one_shot_core(  # 单次MHA前向的核心计算，在一次注意力操作中处理扩展部分和前缀
        self: DeepseekV2AttentionMLA,  # 类型标注为DeepseekV2AttentionMLA
        q: torch.Tensor,  # Q张量
        k: torch.Tensor,  # K张量
        v: torch.Tensor,  # V张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:  # 返回输出张量
        has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)  # 检查是否存在任何扩展前缀
        # Only initialize the info once  # 仅初始化一次
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:  # 如果有前缀但分块信息未初始化
            forward_batch.num_prefix_chunks = 0  # 单次模式不需要分块，设为0
            if hasattr(get_attn_backend(), "init_mha_chunk_metadata"):  # 如果后端支持MHA分块元数据初始化
                get_attn_backend().init_mha_chunk_metadata(forward_batch)  # 初始化MHA分块元数据
        forward_batch.mha_return_lse = False  # 单次模式不需要返回lse
        # Do mha for extended part without prefix  # 对无前缀的扩展部分执行MHA
        forward_batch.set_attn_attend_prefix_cache(False)  # 设置不关注前缀缓存
        return self.forward_normal_core(q, k, v, forward_batch)  # 委托给普通核心计算

    def _chunked_prefix_attn_mha(  # 分块前缀MHA注意力，逐块处理前缀KV缓存并合并结果
        self: DeepseekV2AttentionMLA,  # 类型标注为DeepseekV2AttentionMLA
        q: torch.Tensor,  # Q张量
        accum_output: torch.Tensor,  # 累积的注意力输出
        accum_lse: torch.Tensor,  # 累积的对数softmax指数
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:  # 返回合并后的注意力输出

        # kv_b_proj needs BF16 input, but legacy q.dtype was BF16 by accident.  # kv_b_proj需要BF16输入，但旧代码中q.dtype意外为BF16
        backend = _resolve_attn_backend(forward_batch)  # 解析实际使用的注意力后端
        pack_fn = getattr(backend, "pack_prefix_chunk_kv", None)  # 获取后端的KV打包函数（如果支持）
        kv_a_dtype = torch.bfloat16 if pack_fn is not None else q.dtype  # 如果有打包函数则使用BF16，否则使用Q的数据类型

        assert forward_batch.num_prefix_chunks is not None  # 断言前缀分块数已初始化
        for i in range(forward_batch.num_prefix_chunks):  # 遍历每个前缀分块
            forward_batch.set_prefix_chunk_idx(i)  # 设置当前前缀分块索引

            kv_indices = forward_batch.prefix_chunk_kv_indices[i]  # 获取当前分块的KV索引
            # Fetch latent cache from memory pool with precomputed chunked kv indices  # 使用预计算的分块KV索引从内存池获取潜缓存
            kv_a_normed, k_pe = self._get_mla_kv_buffer(  # 从KV缓存池获取归一化后的KV LoRA和K的RoPE
                kv_indices, kv_a_dtype, forward_batch  # 传入KV索引、数据类型和批次信息
            )
            kv = self.kv_b_proj(kv_a_normed)[0]  # 对KV LoRA进行b投影
            kv = kv.view(  # 重塑KV投影结果形状
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim  # 形状为[token数, 头数, nope维度+v维度]
            )
            v = kv[..., self.qk_nope_head_dim :]  # 提取V部分
            k_nope = kv[..., : self.qk_nope_head_dim]  # 提取K的非位置编码部分

            if pack_fn is not None:  # 如果后端支持KV打包函数
                k, v = pack_fn(k_nope, k_pe, v)  # 使用后端打包函数合并K和V
            else:  # 普通路径，手动拼接
                k = torch.empty(  # 创建空的K张量
                    (  # 形状元组
                        k_nope.shape[0],  # token数
                        self.num_local_heads,  # 头数
                        self.qk_nope_head_dim + self.qk_rope_head_dim,  # 总K维度（nope + RoPE）
                    ),
                    dtype=v.dtype,  # 数据类型与V相同
                    device=v.device,  # 设备与V相同
                )
                k[..., : self.qk_nope_head_dim] = k_nope  # 填充K的nope部分
                k[..., self.qk_nope_head_dim :] = k_pe  # 填充K的RoPE部分

            output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)  # 对当前分块执行MHA注意力计算
            tmp_output = torch.empty_like(accum_output)  # 创建临时输出张量
            tmp_lse = torch.empty_like(accum_lse)  # 创建临时lse张量
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)  # 使用merge_state_v2合并新旧注意力状态
            accum_output, accum_lse = tmp_output, tmp_lse  # 更新累积输出和lse
            del kv, k, v, output, lse, tmp_output, tmp_lse  # 释放中间变量占用的显存

        return accum_output  # 返回合并后的最终注意力输出

    def _set_mla_kv_buffer(  # 将KV潜缓存写入KV缓存池
        self: DeepseekV2AttentionMLA,  # 类型标注为DeepseekV2AttentionMLA
        latent_cache: torch.Tensor,  # 潜缓存张量
        kv_a: torch.Tensor,  # KV LoRA向量（归一化后）
        k_pe: torch.Tensor,  # K的旋转位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
    ):
        if _is_cuda or _use_aiter_gfx95:  # CUDA平台或aiter gfx95平台
            # Save latent cache  # 保存潜缓存
            get_token_to_kv_pool().set_mla_kv_buffer(  # 使用MLA专用接口写入KV缓存
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe  # 传入注意力模块、输出缓存位置、KV LoRA和K RoPE
            )
        elif _is_npu:  # 华为NPU平台
            # To reduce a time-costing split operation  # 为减少耗时的分割操作
            get_token_to_kv_pool().set_kv_buffer(  # 使用通用KV缓存接口写入
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe  # 传入注意力模块、输出缓存位置、KV LoRA和K RoPE
            )
        else:  # 其他平台（如通用CPU/GPU）
            latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)  # 将KV LoRA写入潜缓存前半部分
            latent_cache[:, :, self.kv_lora_rank :] = k_pe.clone()  # 将K RoPE写入潜缓存后半部分（克隆以避免原地修改）

            # Save latent cache  # 保存潜缓存
            get_token_to_kv_pool().set_kv_buffer(  # 使用通用KV缓存接口写入
                self.attn_mha, forward_batch.out_cache_loc, latent_cache, None  # 传入注意力模块、输出缓存位置、潜缓存和无额外值
            )

    def _get_mla_kv_buffer(  # 从KV缓存池读取KV潜向量
        self: DeepseekV2AttentionMLA,  # 类型标注为DeepseekV2AttentionMLA
        kv_indices: torch.Tensor,  # KV索引张量
        dst_dtype: torch.dtype,  # 目标数据类型
        forward_batch: ForwardBatch,  # 前向批次信息
    ):
        if _is_cuda or _use_aiter_gfx95:  # CUDA平台或aiter gfx95平台
            kv_a, k_pe = get_token_to_kv_pool().get_mla_kv_buffer(  # 使用MLA专用接口读取KV缓存
                self.attn_mha, kv_indices, dst_dtype  # 传入注意力模块、KV索引和目标数据类型
            )
            kv_a = kv_a.squeeze(1)  # 去除KV LoRA的第1维
        else:  # 其他平台
            latent_cache_buf = get_token_to_kv_pool().get_key_buffer(  # 获取键缓存缓冲区
                self.attn_mha.layer_id  # 传入当前层ID
            )
            latent_cache = latent_cache_buf[kv_indices].contiguous().to(dst_dtype)  # 按索引获取潜缓存并转为目标类型

            kv_a, k_pe = latent_cache.split(  # 将潜缓存分割为KV LoRA和K RoPE
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1  # 分割点：KV LoRA维度 和 RoPE维度
            )
            kv_a = kv_a.squeeze(1).contiguous()  # 去除KV LoRA的第1维并确保内存连续
        return kv_a, k_pe  # 返回KV LoRA和K RoPE

    def _get_mla_kv_buffer_from_fp8_for_dsa(  # 从FP8 KV缓存反量化获取KV潜向量，用于DSA注意力
        self: DeepseekV2AttentionMLA,  # 类型标注为DeepseekV2AttentionMLA
        forward_batch: ForwardBatch,  # 前向批次信息
    ):
        """
        Dequantize FP8 KV cache to BF16 for MLA attention (DSA-specific format).  # 将FP8 KV缓存反量化为BF16，用于MLA注意力（DSA专用格式）

        Returns: (kv_a, k_pe) both in BF16  # 返回：(kv_a, k_pe)，均为BF16格式
        """
        backend = get_attn_backend()  # 获取当前注意力后端
        if isinstance(backend, TboAttnBackend):  # if enable tbo, get primary backend  # 如果启用TBO，获取主要后端
            backend = backend.primary  # 获取TBO包装器内的主要后端
        kv_indices = backend.forward_metadata.page_table_1_flattened  # 从后端元数据获取页表索引
        assert (  # 断言索引不为空
            kv_indices is not None
        ), "page_table_1_flattened should have been generated for FP8 MHA path"  # page_table_1_flattened应在FP8 MHA路径中已生成

        kv_cache_fp8 = get_token_to_kv_pool().get_key_buffer(self.attn_mha.layer_id)  # 获取FP8格式的键缓存

        kv_latent_bf16 = dequantize_k_cache_paged(kv_cache_fp8, kv_indices)  # 分页反量化FP8键缓存为BF16

        kv_a = kv_latent_bf16[:, :, : self.kv_lora_rank].squeeze(1).contiguous()  # 提取KV LoRA部分并去除第1维
        k_pe = kv_latent_bf16[:, :, self.kv_lora_rank :]  # 提取K RoPE部分

        return kv_a, k_pe  # 返回BF16格式的KV LoRA和K RoPE

    def _concat_and_cast_mha_k(  # 拼接K的nope和RoPE部分，并根据平台选择最优的拼接实现
        self: DeepseekV2AttentionMLA,  # 类型标注为DeepseekV2AttentionMLA
        k_nope: torch.Tensor,  # K的非位置编码部分
        k_pe: torch.Tensor,  # K的旋转位置编码部分
        forward_batch: ForwardBatch,  # 前向批次信息
    ):
        # Temporary for DeepSeek V3/R1 only, but can generalize if needed  # 目前仅用于DeepSeek V3/R1，如有需要可泛化
        k_shape = (k_nope.shape[0], self.num_local_heads, self.qk_head_dim)  # 目标K张量形状
        if (  # CUDA/MUSA平台且维度为DeepSeek V3标准配置
            (_is_cuda or _is_musa)
            and (self.num_local_heads == 128)  # 128个头
            and (self.qk_nope_head_dim == 128)  # nope维度128
            and (self.qk_rope_head_dim == 64)  # RoPE维度64
        ):
            k = k_nope.new_empty(*k_shape)  # 创建空K张量
            concat_mla_k(k=k, k_nope=k_nope, k_rope=k_pe)  # 使用JIT/sgl_kernel的concat_mla_k拼接
        elif (  # CUDA平台且维度为2的幂（支持Triton拼接）
            _is_cuda
            and next_power_of_2(self.num_local_heads) == self.num_local_heads  # 头数是2的幂
            and next_power_of_2(self.qk_nope_head_dim) == self.qk_nope_head_dim  # nope维度是2的幂
            and next_power_of_2(self.qk_rope_head_dim) == self.qk_rope_head_dim  # RoPE维度是2的幂
        ):
            # fa3 mha support fp8 inputs  # FlashAttention3 MHA支持FP8输入
            if (  # 如果使用FA3且KV缓存为非auto类型（量化格式）
                self.current_attention_backend == "fa3"
                and self.kv_cache_dtype != "auto"
            ):
                attn_dtype = get_token_to_kv_pool().dtype  # 使用KV缓存池的数据类型
            else:  # 普通路径
                attn_dtype = k_nope.dtype  # 使用K_nope的数据类型
            k = k_nope.new_empty(*k_shape, dtype=attn_dtype)  # 创建指定数据类型的空K张量
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)  # 使用Triton内核拼接K
        elif _is_hip and self.current_attention_backend == "aiter":  # HIP(AMD)平台且使用aiter后端
            k = k_nope.new_empty(*k_shape)  # 创建空K张量
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)  # 使用Triton内核拼接K
        else:  # 通用路径：手动拼接
            k = k_nope.new_empty(*k_shape)  # 创建空K张量
            k[..., : self.qk_nope_head_dim] = k_nope  # 填充K的nope部分
            k[..., self.qk_nope_head_dim :] = k_pe  # 填充K的RoPE部分
        return k  # 返回拼接后的K张量
