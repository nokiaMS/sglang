# TRT-LLM MLA注意力后端模块，提供基于FlashInfer TRT-LLM MLA核的注意力计算后端实现
# 支持解码（decode）、预填充（prefill）、推测验证（target_verify）和草稿扩展（draft_extend）模式
from __future__ import annotations  # 启用延迟注解评估

"""
Support attention backend for TRTLLM MLA kernels from flashinfer.  # 支持来自FlashInfer的TRT-LLM MLA核的注意力后端
"""

import logging  # 导入日志模块
import math  # 导入数学模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Optional, Union  # 导入类型提示相关

import torch  # 导入PyTorch
import triton  # 导入Triton
import triton.language as tl  # 导入Triton语言

from sglang.jit_kernel.fixup_zero_kv import fixup_zero_kv_rows  # 导入零KV行修复函数
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph  # 导入分段CUDA图判断函数
from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.layers.attention.flashinfer_mla_backend import (  # 导入FlashInfer MLA后端
    FlashInferMLAAttnBackend,  # FlashInfer MLA注意力后端基类
    FlashInferMLAMultiStepDraftBackend,  # FlashInfer MLA多步草稿后端
)
from sglang.srt.layers.attention.utils import (  # 导入注意力工具函数
    concat_mla_absorb_q_general,  # MLA吸收模式查询拼接
    create_flashmla_kv_indices_triton,  # 使用Triton创建FlashMLA KV索引
    get_num_page_per_block_flashmla,  # 获取FlashMLA每块的页数
    mla_quantize_and_rope_for_fp8,  # FP8量化与RoPE处理
)
from sglang.srt.layers.dp_attention import get_attention_tp_size  # 导入注意力张量并行大小获取函数
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant  # 导入FP8缩放量化的核函数
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode  # 导入前向批次信息和模式
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import is_flashinfer_available, is_float4_e2m1fn_x2  # 导入FlashInfer可用性检查和float4格式检查

if is_flashinfer_available():  # 如果FlashInfer可用
    import flashinfer  # 导入FlashInfer

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力类
    from sglang.srt.model_executor.model_runner import ModelRunner  # 导入模型运行器
    from sglang.srt.speculative.spec_info import SpecInput  # 导入推测输入信息

logger = logging.getLogger(__name__)  # 创建日志记录器

# Constants  # 常量定义
DEFAULT_WORKSPACE_SIZE_MB = 150  # Memory workspace size in MB  # 内存工作空间大小，单位MB

# Block constraint from flashinfer requirements  # 来自FlashInfer需求的块约束
# From flashinfer.decode._check_trtllm_gen_mla_shape:  # 来自flashinfer.decode._check_trtllm_gen_mla_shape:
#   block_num % (128 / block_size) == 0  # 块数必须能被(128/块大小)整除
# This imposes that the total number of blocks must be divisible by  # 这要求总块数必须能被
# (128 / block_size). We capture the 128 constant here so we can  # (128/块大小)整除。我们在此捕获128常量以便
# compute the LCM with other padding constraints.  # 与其他填充约束计算最小公倍数
TRTLLM_BLOCK_CONSTRAINT = 128  # TRT-LLM块约束常量


# Triton JIT核函数：对草稿扩展查询张量进行填充，支持并行化头和维度处理
@triton.jit
def pad_draft_extend_query_kernel(
    q_ptr,  # Input query tensor [total_seq_len, num_heads, head_dim]  # 输入查询张量[总序列长度, 头数, 头维度]
    padded_q_ptr,  # Output padded query tensor [batch_size, max_seq_len, num_heads, head_dim]  # 输出填充后的查询张量[批大小, 最大序列长度, 头数, 头维度]
    seq_lens_q_ptr,  # Sequence lengths for each sequence [batch_size]  # 每个序列的序列长度[批大小]
    cumsum_ptr,  # Cumulative sum of sequence lengths [batch_size + 1]  # 序列长度的累加和[批大小+1]
    batch_size,  # 批大小
    max_seq_len,  # 最大序列长度
    num_heads,  # 头数
    head_dim,  # 头维度
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
):
    """Triton kernel for padding draft extended query tensor with parallelized head and dim processing."""  # Triton核函数，用于填充草稿扩展查询张量，支持并行化头和维度处理
    # Use 3D program IDs: (batch_seq, head_block, dim_block)  # 使用3D程序ID：(批次序列, 头块, 维度块)
    batch_seq_pid = tl.program_id(0)  # 获取批次序列维度程序ID
    head_pid = tl.program_id(1)  # 获取头维度程序ID
    dim_pid = tl.program_id(2)  # 获取维度程序ID

    batch_id = batch_seq_pid // max_seq_len  # 计算批次ID
    seq_pos = batch_seq_pid % max_seq_len  # 计算序列内位置

    if batch_id >= batch_size:  # 如果批次ID超出范围
        return  # 直接返回

    # Load sequence length for this batch  # 加载该批次的序列长度
    seq_len = tl.load(seq_lens_q_ptr + batch_id)  # 从全局内存加载序列长度

    if seq_pos >= seq_len:  # 如果位置超出序列长度
        return  # 直接返回

    # Load cumulative sum to get start position in input tensor  # 加载累加和以获取输入张量中的起始位置
    input_start = tl.load(cumsum_ptr + batch_id)  # 加载累加和起始位置
    input_pos = input_start + seq_pos  # 计算输入位置

    # Calculate head and dim block ranges  # 计算头和维度的块范围
    head_start = head_pid * BLOCK_SIZE  # 头维度块起始
    head_end = tl.minimum(head_start + BLOCK_SIZE, num_heads)  # 头维度块结束
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)  # 头维度掩码

    dim_start = dim_pid * BLOCK_SIZE  # 维度块起始
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, head_dim)  # 维度块结束
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)  # 维度掩码

    # Calculate input offset  # 计算输入偏移量
    input_offset = (  # 输入偏移量由批次、头和维度组成
        input_pos * num_heads * head_dim  # 批次偏移
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim  # 头偏移
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]  # 维度偏移
    )

    # Load data  # 加载数据
    data = tl.load(  # 从全局内存加载数据
        q_ptr + input_offset,  # 数据地址
        mask=head_mask[:, None] & dim_mask[None, :],  # 有效数据掩码
        other=0.0,  # 掩码外的填充值
    )

    # Calculate output offset  # 计算输出偏移量
    output_offset = (  # 输出偏移量由批次、序列位置、头和维度组成
        batch_id * max_seq_len * num_heads * head_dim  # 批次偏移
        + seq_pos * num_heads * head_dim  # 序列位置偏移
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim  # 头偏移
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]  # 维度偏移
    )

    # Store data  # 存储数据
    tl.store(  # 将数据存储到全局内存
        padded_q_ptr + output_offset,  # 输出地址
        data,  # 数据
        mask=head_mask[:, None] & dim_mask[None, :],  # 有效数据掩码
    )


# Triton JIT核函数：对草稿扩展输出张量进行反填充，支持并行化头和维度处理
@triton.jit
def unpad_draft_extend_output_kernel(
    raw_out_ptr,  # Input raw output tensor (batch_size, token_per_batch, tp_q_head_num, v_head_dim)  # 输入原始输出张量(批大小, 每批token数, 查询头数, 值头维度)
    output_ptr,  # Output tensor (-1, tp_q_head_num, v_head_dim)  # 输出张量(-1, 查询头数, 值头维度)
    num_accept_tokens_ptr,  # Accept lengths for each sequence [batch_size]  # 每个序列的接受长度[批大小]
    cumsum_ptr,  # Cumulative sum of accept lengths [batch_size + 1]  # 接受长度的累加和[批大小+1]
    batch_size,  # 批大小
    token_per_batch,  # 每批token数
    tp_q_head_num,  # 张量并行查询头数
    v_head_dim,  # 值头维度
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
):
    """Triton kernel for unpadding draft extended output tensor with parallelized head and dim processing."""  # Triton核函数，用于反填充草稿扩展输出张量，支持并行化头和维度处理
    batch_seq_pid = tl.program_id(0)  # 获取批次序列维度程序ID
    head_pid = tl.program_id(1)  # 获取头维度程序ID
    dim_pid = tl.program_id(2)  # 获取维度程序ID

    batch_id = batch_seq_pid // token_per_batch  # 计算批次ID
    seq_pos = batch_seq_pid % token_per_batch  # 计算序列内位置

    if batch_id >= batch_size:  # 如果批次ID超出范围
        return  # 直接返回

    # Load accept length for this batch  # 加载该批次的接受长度
    accept_len = tl.load(num_accept_tokens_ptr + batch_id)  # 从全局内存加载接受长度

    if seq_pos >= accept_len:  # 如果位置超出接受长度
        return  # 直接返回

    # Load cumulative sum to get start position in output tensor  # 加载累加和以获取输出张量中的起始位置
    output_start = tl.load(cumsum_ptr + batch_id)  # 加载累加和起始位置
    output_pos = output_start + seq_pos  # 计算输出位置

    # Calculate head and dim block ranges  # 计算头和维度的块范围
    head_start = head_pid * BLOCK_SIZE  # 头维度块起始
    head_end = tl.minimum(head_start + BLOCK_SIZE, tp_q_head_num)  # 头维度块结束
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)  # 头维度掩码

    dim_start = dim_pid * BLOCK_SIZE  # 维度块起始
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, v_head_dim)  # 维度块结束
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)  # 维度掩码

    # Calculate input offset: (batch_id, seq_pos, head_id, dim_id)  # 计算输入偏移量：(批次ID, 序列位置, 头ID, 维度ID)
    input_offset = (  # 输入偏移量由批次、序列位置、头和维度组成
        batch_id * token_per_batch * tp_q_head_num * v_head_dim  # 批次偏移
        + seq_pos * tp_q_head_num * v_head_dim  # 序列位置偏移
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim  # 头偏移
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]  # 维度偏移
    )

    # Load data  # 加载数据
    data = tl.load(  # 从全局内存加载数据
        raw_out_ptr + input_offset,  # 数据地址
        mask=head_mask[:, None] & dim_mask[None, :],  # 有效数据掩码
        other=0.0,  # 掩码外的填充值
    )

    output_offset = (  # 输出偏移量由序列位置、头和维度组成
        output_pos * tp_q_head_num * v_head_dim  # 序列位置偏移
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim  # 头偏移
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]  # 维度偏移
    )

    # Store data  # 存储数据
    tl.store(  # 将数据存储到全局内存
        output_ptr + output_offset,  # 输出地址
        data,  # 数据
        mask=head_mask[:, None] & dim_mask[None, :],  # 有效数据掩码
    )


# 对QKV张量进行FP8量化
def _quantize_fp8_qkv(q, k, v, layer):
    q = q.to(torch.float8_e4m3fn)  # 将查询张量转换为FP8格式

    k_scale = getattr(layer, "k_scale_float", None)  # 获取键缩放因子
    if k_scale is None:  # 如果没有缩放因子
        k_scale = 1.0  # 默认缩放为1.0
    if k_scale != 1.0:  # 如果缩放因子不为1.0
        assert hasattr(layer, "k_scale"), "k_scale is not set"  # 断言k_scale属性存在
        k_2d, _ = scaled_fp8_quant(  # 使用FP8缩放量化
            k.reshape(-1, k.shape[-1]).contiguous(), layer.k_scale  # 将K展平为2D并量化
        )
        k = k_2d.reshape(k.shape)  # 恢复K的原始形状
    else:  # 如果缩放因子为1.0
        k = k.to(torch.float8_e4m3fn)  # 直接转换为FP8格式

    v_scale = getattr(layer, "v_scale_float", None)  # 获取值缩放因子
    if v_scale is None:  # 如果没有缩放因子
        v_scale = 1.0  # 默认缩放为1.0
    if v_scale != 1.0:  # 如果缩放因子不为1.0
        assert hasattr(layer, "v_scale"), "v_scale is not set"  # 断言v_scale属性存在
        v_2d, _ = scaled_fp8_quant(  # 使用FP8缩放量化
            v.reshape(-1, v.shape[-1]).contiguous(), layer.v_scale  # 将V展平为2D并量化
        )
        v = v_2d.reshape(v.shape)  # 恢复V的原始形状
    else:  # 如果缩放因子为1.0
        v = v.to(torch.float8_e4m3fn)  # 直接转换为FP8格式

    return q, k, v, k_scale, v_scale  # 返回量化后的QKV和缩放因子


global_zero_init_workspace_buffer = None  # 全局零初始化工作空间缓冲区
# cute-dsl needs its own workspace: it overwrites the buffer with split-KV  # cute-dsl需要自己的工作空间：它用split-KV
# partials, which corrupts the trtllm-gen multiCtasKv counters that rely on the  # 部分结果覆盖缓冲区，这会破坏依赖
# zero-init buffer (they share it under attention-backend=cutedsl_mla, where  # 零初始化缓冲区的trtllm-gen multiCtasKv计数器
# draft-extend falls back to trtllm-gen) and deadlocks the reduction.  # （在attention-backend=cutedsl_mla下它们共享，草稿扩展回退到trtllm-gen）并导致规约死锁
global_cute_dsl_workspace_buffer = None  # 全局cute-dsl工作空间缓冲区


# TRT-LLM MLA预填充元数据
@dataclass
class TRTLLMMLAPrefillMetadata:
    """Metadata for TRTLLM MLA prefill operations."""  # TRT-LLM MLA预填充操作的元数据

    max_seq_len: int  # 最大序列长度
    cum_seq_lens: torch.Tensor  # 累积序列长度张量
    seq_lens: torch.Tensor  # 序列长度张量
    fallback_to_flashinfer_impl: bool = False  # 是否回退到FlashInfer实现 # 是否回退到FlashInfer实现，默认为False


# TRT-LLM MLA解码元数据
@dataclass
class TRTLLMMLADecodeMetadata:
    """Metadata for TRTLLM MLA decode operations."""  # TRT-LLM MLA解码操作的元数据

    block_kv_indices: Optional[torch.Tensor] = None  # 块KV索引张量 # 块KV索引张量，默认为None
    max_seq_len_k: Optional[int] = None  # 键最大序列长度 # 键最大序列长度，默认为None
    max_seq_len_q: Optional[int] = None  # 查询最大序列长度 # 查询最大序列长度，默认为None
    sum_seq_lens_q: Optional[int] = None  # 查询序列长度总和 # 查询序列长度总和，默认为None
    cu_seqlens_q: Optional[torch.Tensor] = None  # 查询累积序列长度 # 查询累积序列长度，默认为None
    seq_lens_q: Optional[torch.Tensor] = None  # 查询序列长度 # 查询序列长度，默认为None
    seq_lens_k: Optional[torch.Tensor] = None  # 键序列长度 # 键序列长度，默认为None


# TRT-LLM MLA注意力后端，继承自FlashInfer MLA注意力后端
class TRTLLMMLABackend(FlashInferMLAAttnBackend):
    """TRTLLM MLA attention kernel from flashinfer."""  # 来自FlashInfer的TRT-LLM MLA注意力核

    # trtllm-gen kernels rebuild metadata from preallocated buffers and never  # trtllm-gen核从预分配缓冲区重建元数据，从不
    # read seq_lens_cpu / seq_lens_sum; opt out of the D2H sync.  # 读取seq_lens_cpu/seq_lens_sum；选择退出D2H同步
    needs_cpu_seq_lens: bool = False  # 是否需要CPU序列长度 # 不需要CPU序列长度同步

    def __init__(
        self,
        model_runner: ModelRunner,  # 模型运行器
        skip_prefill: bool = False,  # 是否跳过预填充 # 是否跳过预填充，默认为False
        kv_indptr_buf: Optional[torch.Tensor] = None,  # KV索引指针缓冲区 # KV索引指针缓冲区，默认为None
        q_indptr_decode_buf: Optional[torch.Tensor] = None,  # 解码查询索引指针缓冲区 # 解码查询索引指针缓冲区，默认为None
        backend: str = "trtllm-gen",  # 后端类型 # 后端类型，默认为trtllm-gen
    ):
        super().__init__(  # 调用父类初始化
            model_runner,  # 模型运行器
            skip_prefill,  # 是否跳过预填充
            kv_indptr_buf,  # KV索引指针缓冲区
            q_indptr_decode_buf,  # 解码查询索引指针缓冲区
        )

        config = model_runner.model_config  # 获取模型配置

        # Model parameters  # 模型参数
        self.num_q_heads = config.num_attention_heads // get_attention_tp_size()  # 查询头数（考虑张量并行）
        self.num_kv_heads = config.get_num_kv_heads(get_attention_tp_size())  # 键值头数（考虑张量并行）
        self.num_local_heads = config.num_attention_heads // get_attention_tp_size()  # 本地头数（考虑张量并行）

        # MLA-specific dimensions  # MLA特定维度
        self.kv_lora_rank = config.kv_lora_rank  # KV LoRA秩
        self.qk_nope_head_dim = config.qk_nope_head_dim  # QK非旋转头维度
        self.qk_rope_head_dim = config.qk_rope_head_dim  # QK旋转位置编码头维度
        self.v_head_dim = config.v_head_dim  # 值头维度
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim  # KV缓存维度

        # Runtime parameters  # 运行时参数
        self.backend = backend  # 后端类型
        self.scaling = config.scaling  # 缩放因子
        self.data_type = model_runner.kv_cache_dtype  # KV缓存数据类型
        self.q_data_type = model_runner.dtype  # 查询数据类型
        self.page_size = model_runner.page_size  # 页大小
        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 请求到token映射

        # Workspace allocation  # 工作空间分配
        self.workspace_size = DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024  # 计算工作空间大小（字节）
        if self.backend == "cute-dsl":  # 如果使用cute-dsl后端
            # Separate buffer from trtllm-gen (see note above); safe to share  # 与trtllm-gen分开的缓冲区（见上面注释）；可安全共享
            # among cute-dsl instances.  # 在cute-dsl实例之间
            global global_cute_dsl_workspace_buffer  # 声明全局cute-dsl工作空间缓冲区
            if global_cute_dsl_workspace_buffer is None:  # 如果全局缓冲区尚未创建
                global_cute_dsl_workspace_buffer = torch.zeros(  # 创建零初始化的工作空间
                    self.workspace_size,  # 工作空间大小
                    dtype=torch.int8,  # 数据类型为int8
                    device=model_runner.device,  # 设备
                )
            self.workspace_buffer = global_cute_dsl_workspace_buffer  # 使用全局cute-dsl工作空间
        else:  # 使用trtllm-gen后端
            global global_zero_init_workspace_buffer  # 声明全局零初始化工作空间缓冲区
            if global_zero_init_workspace_buffer is None:  # 如果全局缓冲区尚未创建
                global_zero_init_workspace_buffer = torch.zeros(  # 创建零初始化的工作空间
                    self.workspace_size,  # 工作空间大小
                    dtype=torch.int8,  # 数据类型为int8
                    device=model_runner.device,  # 设备
                )
            self.workspace_buffer = global_zero_init_workspace_buffer  # 使用全局零初始化工作空间

        # CUDA graph state  # CUDA图状态
        self.decode_cuda_graph_metadata = {}  # 解码CUDA图元数据字典
        self.decode_cuda_graph_kv_indices = None  # 解码CUDA图KV索引
        self.padded_q_buffer = None  # 填充查询缓冲区
        self.unpad_output_buffer = None  # 反填充输出缓冲区
        self.forward_prefill_metadata: Optional[TRTLLMMLAPrefillMetadata] = None  # 前向预填充元数据
        self.forward_decode_metadata: Union[TRTLLMMLADecodeMetadata, None] = None  # 前向解码元数据

        self.disable_chunked_prefix_cache = (  # 是否禁用分块前缀缓存
            get_global_server_args().disable_chunked_prefix_cache  # 从全局服务器参数获取
        )

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens  # 推测解码草稿token数
        self.cuda_graph_custom_mask = None  # CUDA图自定义掩码

    # 计算满足TRT-LLM和Triton约束的填充块数
    def _calc_padded_blocks(self, max_seq_len: int) -> int:
        """
        Calculate padded block count that satisfies both TRT-LLM and Triton constraints.  # 计算满足TRT-LLM和Triton约束的填充块数

        Args:  # 参数：
            max_seq_len: Maximum sequence length in tokens  # 最大序列长度（以token为单位）

        Returns:  # 返回：
            Number of blocks padded to satisfy all constraints  # 满足所有约束的填充块数
        """
        blocks = triton.cdiv(max_seq_len, self.page_size)  # 计算需要的块数（向上取整除法）

        # Apply dual constraints (take LCM to satisfy both):  # 应用双重约束（取最小公倍数满足两者）：
        # 1. TRT-LLM: block_num % (128 / page_size) == 0  # 1. TRT-LLM约束：块数能被(128/页大小)整除
        # 2. Triton: number of pages per block  # 2. Triton约束：每块的页数
        trtllm_constraint = TRTLLM_BLOCK_CONSTRAINT // self.page_size  # TRT-LLM约束值
        triton_constraint = get_num_page_per_block_flashmla(self.page_size)  # Triton约束值
        constraint_lcm = math.lcm(trtllm_constraint, triton_constraint)  # 计算两个约束的最小公倍数

        if blocks % constraint_lcm != 0:  # 如果块数不满足约束
            blocks = triton.cdiv(blocks, constraint_lcm) * constraint_lcm  # 向上取整到约束的倍数
        return blocks  # 返回填充后的块数

    # 使用Triton核函数创建块KV索引张量
    def _create_block_kv_indices(
        self,
        batch_size: int,  # 批大小
        max_blocks: int,  # 每个序列的最大块数
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        device: torch.device,  # 目标设备
    ) -> torch.Tensor:
        """
        Create block KV indices tensor using Triton kernel.  # 使用Triton核函数创建块KV索引张量

        Args:  # 参数：
            batch_size: Batch size  # 批大小
            max_blocks: Maximum number of blocks per sequence  # 每个序列的最大块数
            req_pool_indices: Request pool indices  # 请求池索引
            seq_lens: Sequence lengths  # 序列长度
            device: Target device  # 目标设备

        Returns:  # 返回：
            Block KV indices tensor  # 块KV索引张量
        """
        block_kv_indices = torch.full(  # 创建填充为-1的块KV索引张量
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device  # 形状为(批大小, 最大块数)，类型为int32
        )

        create_flashmla_kv_indices_triton[(batch_size,)](  # 调用Triton核函数创建KV索引
            self.req_to_token,  # 请求到token映射
            req_pool_indices,  # 请求池索引
            seq_lens,  # 序列长度
            None,  # 前缀长度（未使用）
            block_kv_indices,  # 输出的块KV索引
            self.req_to_token.stride(0),  # 步长
            max_blocks,  # 最大块数
            PAGED_SIZE=self.page_size,  # 页大小
        )

        return block_kv_indices  # 返回块KV索引

    # 初始化TRTLLM MLA的CUDA图状态
    def init_cuda_graph_state(
        self,
        max_bs: int,  # 最大批大小
        max_num_tokens: int,  # 最大token数
        kv_indices_buf: Optional[torch.Tensor] = None,  # KV索引缓冲区 # KV索引缓冲区，默认为None
    ):
        """Initialize CUDA graph state for TRTLLM MLA."""  # 初始化TRTLLM MLA的CUDA图状态

        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)  # 计算每个序列的最大填充块数

        self.decode_cuda_graph_kv_indices = torch.full(  # 创建填充为-1的解码CUDA图KV索引
            (max_bs, max_blocks_per_seq), -1, dtype=torch.int32, device=self.device  # 形状为(最大批大小, 最大块数)
        )
        num_tokens_per_bs = max_num_tokens // max_bs  # 计算每个批次的token数

        if is_float4_e2m1fn_x2(self.data_type):  # 如果数据类型是float4格式
            # Buffer for padded query: (max_bs, max_draft_tokens, num_q_heads, v_head_dim)  # 填充查询缓冲区：(最大批大小, 最大草稿token数, 查询头数, 值头维度)
            self.store_dtype = torch.uint8  # 存储数据类型为uint8
            self.padded_q_buffer = torch.zeros(  # 创建填充查询缓冲区
                (max_bs, num_tokens_per_bs // 2, self.num_q_heads, self.kv_cache_dim),  # 形状注意除以2（float4打包）
                dtype=self.store_dtype,  # 数据类型
                device=self.device,  # 设备
            )

            # Buffer for unpadded output: (max_num_tokens, num_q_heads, v_head_dim)  # 反填充输出缓冲区：(最大token数, 查询头数, 值头维度)
            self.unpad_output_buffer = torch.zeros(  # 创建反填充输出缓冲区
                (max_num_tokens // 2, self.num_q_heads, 512),  # 形状注意除以2（float4打包），最后一维512
                dtype=self.store_dtype,  # 数据类型
                device=self.device,  # 设备
            )
        else:  # 非float4格式
            # Buffer for padded query: (max_bs, max_draft_tokens, num_q_heads, v_head_dim)  # 填充查询缓冲区：(最大批大小, 最大草稿token数, 查询头数, 值头维度)
            self.padded_q_buffer = torch.zeros(  # 创建填充查询缓冲区
                (max_bs, num_tokens_per_bs, self.num_q_heads, self.kv_cache_dim),  # 标准形状
                dtype=self.data_type,  # 数据类型
                device=self.device,  # 设备
            )

            # Buffer for unpadded output: (max_num_tokens, num_q_heads, v_head_dim)  # 反填充输出缓冲区：(最大token数, 查询头数, 值头维度)
            self.unpad_output_buffer = torch.zeros(  # 创建反填充输出缓冲区
                (max_num_tokens, self.num_q_heads, 512),  # 标准形状，最后一维512
                dtype=self.data_type,  # 数据类型
                device=self.device,  # 设备
            )

        if self.num_draft_tokens and not self.skip_prefill:  # 如果有草稿token且不跳过预填充
            # Worst-case FULL_MASK tree-mask scratch (bool); build_tree writes it  # 最坏情况FULL_MASK树掩码暂存区(bool)；build_tree原地写入
            # in-place so the gpu_only path needs no seq_lens_sum.  # 所以gpu_only路径不需要seq_lens_sum
            self.cuda_graph_custom_mask = torch.zeros(  # 创建CUDA图自定义掩码
                max_num_tokens * (self.max_context_len + self.num_draft_tokens),  # 大小为最大token数*(最大上下文长度+草稿token数)
                dtype=torch.bool,  # 布尔类型
                device=self.device,  # 设备
            )

        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)  # 调用父类初始化CUDA图状态

    # 获取草稿后需要填充的验证缓冲区
    def get_verify_buffers_to_fill_after_draft(self):
        return [self.cuda_graph_custom_mask, None]  # 返回CUDA图自定义掩码和None

    # 为CUDA图捕获分配持久元数据缓冲区
    def _init_cuda_graph_metadata(
        self,
        bs: int,  # 批大小
        num_tokens: int,  # token数
        forward_mode: ForwardMode,  # 前向模式
        seq_lens: torch.Tensor,  # 序列长度
        device: torch.device,  # 设备
    ):
        """Allocate persistent metadata buffers for CUDA graph capture."""  # 为CUDA图捕获分配持久元数据缓冲区
        metadata = TRTLLMMLADecodeMetadata()  # 创建解码元数据

        if forward_mode.is_target_verify():  # 如果是目标验证模式
            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)  # 初始化键序列长度为零
        elif forward_mode.is_draft_extend(include_v2=True):  # 如果是草稿扩展模式
            num_tokens_per_bs = num_tokens // bs  # 计算每批次token数
            metadata.max_seq_len_q = num_tokens_per_bs  # 设置查询最大序列长度
            metadata.sum_seq_lens_q = num_tokens_per_bs * bs  # 设置查询序列长度总和
            metadata.cu_seqlens_q = torch.arange(  # 创建查询累积序列长度
                0,  # 起始值
                bs * num_tokens_per_bs + 1,  # 结束值
                num_tokens_per_bs,  # 步长
                dtype=torch.int32,  # 数据类型
                device=device,  # 设备
            )
            metadata.seq_lens_q = torch.full(  # 创建填充的查询序列长度
                (bs,), num_tokens_per_bs, dtype=torch.int32, device=device  # 所有序列长度相同
            )
            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)  # 初始化键序列长度为零

        # Capture with full width so future longer sequences are safe during replay.  # 使用全宽度捕获，以便重放时更长的序列安全
        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)  # 计算最大填充块数
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks_per_seq]  # 截取对应批大小的KV索引
        metadata.block_kv_indices = block_kv_indices  # 设置块KV索引
        metadata.max_seq_len_k = self.max_context_len  # 设置键最大序列长度

        self.decode_cuda_graph_metadata[bs] = metadata  # 保存到CUDA图元数据字典
        self.forward_decode_metadata = metadata  # 更新当前解码元数据

    # 为CUDA图捕获初始化元数据
    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,  # 批大小
        num_tokens: int,  # token数
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        forward_mode: ForwardMode,  # 前向模式
        spec_info: Optional[SpecInput],  # 推测信息
    ):
        """Initialize metadata for CUDA graph capture."""  # 为CUDA图捕获初始化元数据

        # Delegate to parent for non-decode modes.  # 对于非解码模式委托给父类
        if (
            not forward_mode.is_decode_or_idle()  # 不是解码或空闲模式
            and not forward_mode.is_target_verify()  # 不是目标验证模式
            and not forward_mode.is_draft_extend(include_v2=True)  # 不是草稿扩展模式
        ):
            return super().init_forward_metadata_capture_cuda_graph(  # 委托给父类处理
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

        self._init_cuda_graph_metadata(  # 初始化CUDA图元数据
            bs, num_tokens, forward_mode, seq_lens, seq_lens.device
        )
        self.init_forward_metadata_replay_cuda_graph(  # 同时调用重放初始化
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=None,  # 序列长度总和为None
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens.cpu(),  # 将序列长度移到CPU
        )

    # 使用新输入重放CUDA图
    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,  # 批大小
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        seq_lens_sum: int,  # 序列长度总和
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        forward_mode: ForwardMode,  # 前向模式
        spec_info: Optional[SpecInput],  # 推测信息
        seq_lens_cpu: Optional[torch.Tensor],  # CPU上的序列长度
    ):
        """Replay CUDA graph with new inputs."""  # 使用新输入重放CUDA图
        # Delegate to parent for non-decode modes.  # 对于非解码模式委托给父类
        if (
            not forward_mode.is_decode_or_idle()  # 不是解码或空闲模式
            and not forward_mode.is_target_verify()  # 不是目标验证模式
            and not forward_mode.is_draft_extend(include_v2=True)  # 不是草稿扩展模式
        ):
            return super().init_forward_metadata_replay_cuda_graph(  # 委托给父类处理
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

        metadata = self.decode_cuda_graph_metadata[bs]  # 获取对应批大小的CUDA图元数据

        if forward_mode.is_target_verify():  # 如果是目标验证模式
            seq_lens = seq_lens[:bs] + self.num_draft_tokens  # 序列长度加上草稿token数
            metadata.seq_lens_k.copy_(seq_lens.to(dtype=torch.int32))  # 拷贝键序列长度
            del seq_lens_sum  # not handle "num_draft_tokens" but we do not need it  # 不处理"num_draft_tokens"但我们不需要它
        elif forward_mode.is_draft_extend(include_v2=True):  # 如果是草稿扩展模式
            num_tokens_per_bs = self.num_draft_tokens  # 每批次token数等于草稿token数
            metadata.max_seq_len_q = num_tokens_per_bs  # 更新查询最大序列长度
            metadata.sum_seq_lens_q = num_tokens_per_bs * bs  # 更新查询序列长度总和
            metadata.cu_seqlens_q[: bs + 1].copy_(  # 更新查询累积序列长度
                torch.arange(  # 创建等差数列
                    0,  # 起始值
                    bs * num_tokens_per_bs + 1,  # 结束值
                    step=num_tokens_per_bs,  # 步长
                    dtype=torch.int32,  # 数据类型
                    device=seq_lens.device,  # 设备
                )
            )
            metadata.seq_lens_q[:bs].fill_(num_tokens_per_bs)  # 填充查询序列长度
            # see NOTE(draft_extend seq_len handling)  # 参见注释(草稿扩展序列长度处理)
            seq_lens = seq_lens[:bs] - metadata.seq_lens_q[:bs] + metadata.max_seq_len_q  # 调整序列长度
            metadata.seq_lens_k.copy_(seq_lens.to(torch.int32))  # 拷贝键序列长度

        # Update block indices for new sequences.  # 更新新序列的块索引
        create_flashmla_kv_indices_triton[(bs,)](  # 调用Triton核函数创建KV索引
            self.req_to_token,  # 请求到token映射
            req_pool_indices[:bs],  # 请求池索引（截取对应批大小）
            seq_lens,  # 序列长度
            None,  # 前缀长度（未使用）
            metadata.block_kv_indices,  # 输出的块KV索引
            self.req_to_token.stride(0),  # 步长
            metadata.block_kv_indices.shape[1],  # 块数
            PAGED_SIZE=self.page_size,  # 页大小
        )

    # 获取CUDA图中序列长度的填充值
    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""  # 获取CUDA图中序列长度的填充值
        return 1  # 返回1

    # 初始化MHA分块元数据（重载版本1）
    def init_mha_chunk_metadata(self, forward_batch: "ForwardBatch") -> None:
        has_prefix = any(forward_batch.extend_prefix_lens_cpu)  # 检查是否有前缀
        fallback_to_flashinfer_impl = (  # 判断是否需要回退到FlashInfer实现
            self.disable_chunked_prefix_cache and has_prefix  # 禁用分块前缀缓存且有前缀
        ) or is_in_piecewise_cuda_graph()  # 或在分段CUDA图中
        if fallback_to_flashinfer_impl:  # 如果需要回退
            super().init_mha_chunk_metadata(forward_batch)  # 调用父类初始化

    # 初始化前向传播的元数据
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""  # 初始化前向传播的元数据
        # Delegate to parent for non-decode modes.  # 对于非解码模式委托给父类
        if (
            forward_batch.forward_mode.is_extend()  # 如果是扩展模式
            and not forward_batch.forward_mode.is_target_verify()  # 且不是目标验证
            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)  # 且不是草稿扩展
        ):
            # For extend batch with prefix length > 0, fallback to ragged kernel implemented in flashinfer MLA backend  # 对于前缀长度>0的扩展批次，回退到flashinfer MLA后端实现的ragged核
            # when chunked prefix cache is disabled.  # 当禁用分块前缀缓存时
            # Also fallback to flashinfer MLA backend when in piecewise cuda graph, since it only supports MLA forward mode.  # 在分段CUDA图中也回退到flashinfer MLA后端，因为它只支持MLA前向模式
            has_prefix = any(forward_batch.extend_prefix_lens_cpu)  # 检查是否有前缀
            fallback_to_flashinfer_impl = (  # 判断是否需要回退到FlashInfer实现
                self.disable_chunked_prefix_cache and has_prefix  # 禁用分块前缀缓存且有前缀
            ) or is_in_piecewise_cuda_graph()  # 或在分段CUDA图中
            if fallback_to_flashinfer_impl:  # 如果需要回退
                super().init_forward_metadata(forward_batch)  # 调用父类初始化

            seq_lens = forward_batch.seq_lens - forward_batch.extend_prefix_lens  # 计算实际序列长度（减去前缀长度）
            cum_seq_lens_q = torch.cat(  # 计算查询累积序列长度
                (
                    torch.zeros(  # 起始零值
                        1, dtype=torch.int32, device=forward_batch.seq_lens.device  # 形状1，int32类型
                    ),
                    torch.cumsum(seq_lens, dim=0),  # 累积求和
                )
            ).int()  # 转换为int32
            max_seq_len = max(forward_batch.extend_seq_lens_cpu)  # 获取最大扩展序列长度
            self.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(  # 创建预填充元数据
                max_seq_len,  # 最大序列长度
                cum_seq_lens_q,  # 累积序列长度
                seq_lens,  # 序列长度
                fallback_to_flashinfer_impl,  # 是否回退到FlashInfer
            )
        elif (
            forward_batch.forward_mode.is_decode_or_idle()  # 解码或空闲模式
            or forward_batch.forward_mode.is_target_verify()  # 或目标验证模式
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)  # 或草稿扩展模式
        ):
            bs = forward_batch.batch_size  # 获取批大小
            self.forward_decode_metadata = TRTLLMMLADecodeMetadata()  # 创建解码元数据
            # This is necessary because the backend instance persists across forward passes,  # 这是必要的，因为后端实例在前向传播之间持久存在
            # and forward_prefill_metadata from a previous regular extend call could still be set.  # 并且之前常规扩展调用的forward_prefill_metadata可能仍然被设置
            if (
                forward_batch.forward_mode.is_target_verify()  # 如果是目标验证模式
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)  # 或草稿扩展模式
            ):
                self.forward_prefill_metadata = None  # 清除预填充元数据
            # Get maximum sequence length.  # 获取最大序列长度
            if getattr(forward_batch, "seq_lens_cpu", None) is not None:  # 如果有CPU上的序列长度
                max_seq = forward_batch.seq_lens_cpu.max().item()  # 从CPU版本获取最大值
            else:  # 否则从GPU版本获取
                max_seq = forward_batch.seq_lens.max().item()  # 从GPU版本获取最大值

            seq_lens = forward_batch.seq_lens  # 获取序列长度

            if forward_batch.forward_mode.is_target_verify():  # 如果是目标验证模式
                max_seq = max_seq + self.num_draft_tokens  # 最大序列长度加上草稿token数
                seq_lens = seq_lens + self.num_draft_tokens  # 序列长度加上草稿token数
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)  # 保存键序列长度
            elif forward_batch.forward_mode.is_draft_extend(include_v2=True):  # 如果是草稿扩展模式
                sum_seq_lens_q = sum(forward_batch.extend_seq_lens_cpu)  # 计算查询序列长度总和
                max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)  # 获取查询最大序列长度
                cu_seqlens_q = torch.nn.functional.pad(  # 计算查询累积序列长度并填充
                    torch.cumsum(  # 累积求和
                        forward_batch.extend_seq_lens, dim=0, dtype=torch.int32  # 沿维度0累积
                    ),
                    (1, 0),  # 在左侧填充一个零
                )
                # see NOTE(draft_extend seq_len handling)  # 参见注释(草稿扩展序列长度处理)
                seq_lens = seq_lens - forward_batch.extend_seq_lens + max_seq_len_q  # 调整序列长度

                self.forward_decode_metadata.max_seq_len_q = max_seq_len_q  # 保存查询最大序列长度
                self.forward_decode_metadata.sum_seq_lens_q = sum_seq_lens_q  # 保存查询序列长度总和
                self.forward_decode_metadata.cu_seqlens_q = cu_seqlens_q  # 保存查询累积序列长度
                self.forward_decode_metadata.seq_lens_q = forward_batch.extend_seq_lens  # 保存查询序列长度
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)  # 保存键序列长度

            max_seqlen_pad = self._calc_padded_blocks(max_seq)  # 计算填充后的块数
            block_kv_indices = self._create_block_kv_indices(  # 创建块KV索引
                bs,  # 批大小
                max_seqlen_pad,  # 最大填充块数
                forward_batch.req_pool_indices,  # 请求池索引
                seq_lens,  # 序列长度
                seq_lens.device,  # 设备
            )

            self.forward_decode_metadata.block_kv_indices = block_kv_indices  # 保存块KV索引
            self.forward_decode_metadata.max_seq_len_k = int(max_seq)  # 保存键最大序列长度
            self.forward_decode_metadata.batch_size = bs  # 保存批大小

            forward_batch.decode_trtllm_mla_metadata = self.forward_decode_metadata  # 将元数据附加到前向批次
        else:  # 其他模式
            return super().init_forward_metadata(forward_batch)  # 委托给父类

    # 初始化MHA分块元数据（重载版本2，禁用FlashInfer ragged）
    def init_mha_chunk_metadata(self, forward_batch: ForwardBatch):
        super().init_mha_chunk_metadata(forward_batch, disable_flashinfer_ragged=True)  # 调用父类，禁用FlashInfer ragged

    # 使用Triton核函数填充草稿扩展查询
    def pad_draft_extend_query(
        self,
        q: torch.Tensor,  # 查询张量
        padded_q: torch.Tensor,  # 填充后的查询张量
        seq_lens_q: torch.Tensor,  # 查询序列长度
        cu_seqlens_q: torch.Tensor,  # 查询累积序列长度
    ) -> torch.Tensor:
        """Pad draft extended query using Triton kernel."""  # 使用Triton核函数填充草稿扩展查询
        batch_size = cu_seqlens_q.shape[0] - 1  # 计算批大小
        max_seq_len_q = padded_q.shape[1]  # 获取查询最大序列长度
        num_heads = padded_q.shape[2]  # 获取头数
        head_dim = padded_q.shape[3]  # 获取头维度

        # Launch Triton kernel with 3D grid for parallelized head and dim processing  # 使用3D网格启动Triton核函数，并行处理头和维度
        BLOCK_SIZE = 64  # 设置块大小为64
        num_head_blocks = triton.cdiv(num_heads, BLOCK_SIZE)  # 计算头维度块数
        num_dim_blocks = triton.cdiv(head_dim, BLOCK_SIZE)  # 计算维度块数
        grid = (batch_size * max_seq_len_q, num_head_blocks, num_dim_blocks)  # 构建3D网格

        pad_draft_extend_query_kernel[grid](  # 启动填充核函数
            q_ptr=q,  # 查询张量指针
            padded_q_ptr=padded_q,  # 输出填充查询指针
            seq_lens_q_ptr=seq_lens_q,  # 序列长度指针
            cumsum_ptr=cu_seqlens_q,  # 累积和指针
            batch_size=batch_size,  # 批大小
            max_seq_len=max_seq_len_q,  # 最大序列长度
            num_heads=num_heads,  # 头数
            head_dim=head_dim,  # 头维度
            BLOCK_SIZE=BLOCK_SIZE,  # 块大小
        )
        return padded_q  # 返回填充后的查询

    # 使用Triton核函数反填充草稿扩展输出
    def unpad_draft_extend_output(
        self,
        raw_out: torch.Tensor,  # 原始输出张量
        cu_seqlens_q: torch.Tensor,  # 查询累积序列长度
        seq_lens_q: torch.Tensor,  # 查询序列长度
        sum_seq_lens_q: int,  # 查询序列长度总和
    ) -> torch.Tensor:
        """Unpad draft extended output using Triton kernel."""  # 使用Triton核函数反填充草稿扩展输出
        # raw_out: (batch_size, token_per_batch, layer.tp_q_head_num, layer.v_head_dim)  # 原始输出形状：(批大小, 每批token数, 查询头数, 值头维度)
        batch_size = seq_lens_q.shape[0]  # 获取批大小
        token_per_batch = raw_out.shape[1]  # max_seq_len  # 每批token数，即最大序列长度
        tp_q_head_num = raw_out.shape[2]  # num_heads  # 查询头数
        v_head_dim = raw_out.shape[3]  # head_dim  # 值头维度
        total_tokens = sum_seq_lens_q  # 总token数

        # Check if we're in CUDA graph mode (buffers are pre-allocated)  # 检查是否处于CUDA图模式（缓冲区已预分配）
        if self.unpad_output_buffer is not None:  # 如果有预分配的缓冲区
            # Use pre-allocated buffer for CUDA graph compatibility  # 使用预分配缓冲区以兼容CUDA图
            output = self.unpad_output_buffer[:total_tokens, :, :].to(  # 截取并转换数据类型
                dtype=raw_out.dtype  # 转换为原始输出的数据类型
            )
        else:  # 非CUDA图模式
            # Dynamic allocation for non-CUDA graph mode  # 非CUDA图模式的动态分配
            output = torch.empty(  # 创建空输出张量
                (total_tokens, tp_q_head_num, v_head_dim),  # 形状
                dtype=raw_out.dtype,  # 数据类型
                device=raw_out.device,  # 设备
            )

        # Launch Triton kernel with 3D grid for parallelized head and dim processing  # 使用3D网格启动Triton核函数，并行处理头和维度
        BLOCK_SIZE = 64  # 设置块大小为64
        num_head_blocks = triton.cdiv(tp_q_head_num, BLOCK_SIZE)  # 计算头维度块数
        num_dim_blocks = triton.cdiv(v_head_dim, BLOCK_SIZE)  # 计算维度块数
        grid = (batch_size * token_per_batch, num_head_blocks, num_dim_blocks)  # 构建3D网格

        unpad_draft_extend_output_kernel[grid](  # 启动反填充核函数
            raw_out_ptr=raw_out,  # 原始输出指针
            output_ptr=output,  # 输出指针
            num_accept_tokens_ptr=seq_lens_q,  # 接受token数指针
            cumsum_ptr=cu_seqlens_q,  # 累积和指针
            batch_size=batch_size,  # 批大小
            token_per_batch=token_per_batch,  # 每批token数
            tp_q_head_num=tp_q_head_num,  # 查询头数
            v_head_dim=v_head_dim,  # 值头维度
            BLOCK_SIZE=BLOCK_SIZE,  # 块大小
        )
        return output[:total_tokens, :, :]  # 返回截取后的输出

    # 计算解码BMM1的缩放因子
    def _compute_decode_bmm1_scale(self, layer: RadixAttention) -> float:
        """BMM1 scale ``q_scale * k_scale * softmax_scale``. k_scale only  # BMM1缩放为``q_scale * k_scale * softmax_scale``。k_scale仅
        applies when the KV cache stores FP8."""  # 在KV缓存存储FP8时适用
        q_scale = 1.0  # 查询缩放因子设为1.0
        if self.data_type == torch.float8_e4m3fn:  # 如果KV缓存数据类型是FP8
            k_scale = (  # 获取键缩放因子
                layer.k_scale_float  # 从层获取k_scale_float
                if getattr(layer, "k_scale_float", None) is not None  # 如果存在
                else 1.0  # 否则默认1.0
            )
        else:  # 如果KV缓存不是FP8
            if getattr(layer, "k_scale_float", None) is not None:  # 如果层有k_scale_float属性
                logger.warning_once(  # 输出一次性警告
                    "Checkpoint has k_scale but KV cache dtype is not FP8. "  # 检查点有k_scale但KV缓存类型不是FP8
                    "Ignoring k_scale for BMM1 (k_scale=%.4f, kv_dtype=%s).",  # 忽略BMM1的k_scale
                    layer.k_scale_float,  # 键缩放因子值
                    self.data_type,  # KV缓存数据类型
                )
            k_scale = 1.0  # 键缩放因子设为1.0
        return q_scale * k_scale * layer.scaling  # 返回BMM1缩放因子

    # 运行解码核函数（子类可重写以替换核函数）
    def _run_decode_kernel(
        self,
        query: torch.Tensor,  # 查询张量
        kv_cache: torch.Tensor,  # KV缓存张量
        block_tables: torch.Tensor,  # 块表
        seq_lens: torch.Tensor,  # 序列长度
        max_seq_len: int,  # 最大序列长度
        layer: RadixAttention,  # 注意力层
    ) -> torch.Tensor:
        """Hook for subclasses to swap the decode/spec-verify kernel."""  # 子类可重写的钩子，用于替换解码/推测验证核函数

        # Scale computation for TRTLLM MLA kernel BMM1 operation:  # TRTLLM MLA核BMM1操作的缩放计算：
        # The final BMM1 scale is computed as: q_scale * k_scale * softmax_scale  # 最终BMM1缩放计算为：q_scale * k_scale * softmax_scale
        # Scale components:  # 缩放组件：
        # - q_scale: Query scaling factor (set to 1.0 for both FP16/FP8 paths)  # - q_scale：查询缩放因子（FP16/FP8路径均设为1.0）
        # - k_scale: Key scaling factor from model checkpoint. Only applied when KV cache  # - k_scale：来自模型检查点的键缩放因子。仅在KV缓存
        #   stores FP8-quantized values, to compensate for the quantization scaling.  #   存储FP8量化值时应用，以补偿量化缩放
        #   For BF16/FP16 KV cache, k_scale must be 1.0 since values are unscaled.  #   对于BF16/FP16 KV缓存，k_scale必须为1.0，因为值未缩放
        # - softmax_scale: Attention softmax scaling = 1/sqrt(head_dim), pre-computed as layer.scaling  # - softmax_scale：注意力softmax缩放=1/sqrt(head_dim)，预计算为layer.scaling
        bmm1_scale = self._compute_decode_bmm1_scale(layer)  # 计算BMM1缩放因子
        seq_lens_i32 = (  # 确保序列长度为int32类型
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)  # 如果不是int32则转换
        )
        extra_kwargs = {"backend": self.backend} if self.backend != "trtllm-gen" else {}  # 非默认后端时添加backend参数
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(  # 调用FlashInfer的TRTLLM MLA批量解码
            query=query,  # 查询张量
            kv_cache=kv_cache,  # KV缓存
            workspace_buffer=self.workspace_buffer,  # 工作空间缓冲区
            qk_nope_head_dim=self.qk_nope_head_dim,  # QK非旋转头维度
            kv_lora_rank=self.kv_lora_rank,  # KV LoRA秩
            qk_rope_head_dim=self.qk_rope_head_dim,  # QK旋转位置编码头维度
            block_tables=block_tables,  # 块表
            seq_lens=seq_lens_i32,  # 序列长度（int32）
            max_seq_len=max_seq_len,  # 最大序列长度
            bmm1_scale=bmm1_scale,  # BMM1缩放因子
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),  # 跳过softmax的阈值缩放因子
            **extra_kwargs,  # 额外参数
        )

    # 运行预填充核函数（子类可重写以替换ragged预填充核函数）
    def _run_prefill_kernel(
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 注意力层
        batch_size: int,  # 批大小
        cum_seq_lens_q: torch.Tensor,  # 查询累积序列长度
        max_q_len: int,  # 最大查询长度
        seq_lens_kv: torch.Tensor,  # 键值序列长度
        cum_seq_lens_kv: torch.Tensor,  # 键值累积序列长度
        max_kv_len: int,  # 最大键值长度
        is_causal: bool,  # 是否因果
        return_lse: bool,  # 是否返回log-sum-exp
        out_buffer: torch.Tensor,  # 输出缓冲区
        o_sf_scale: float = 1.0,  # 输出缩放因子 # 输出缩放因子，默认为1.0
    ):
        """Hook for subclasses to swap the ragged prefill kernel. Q/K/V arrive  # 子类可重写的钩子，用于替换ragged预填充核函数。Q/K/V到达
        in model-native dtype; subclasses do any kernel-specific quantization.  # 时为模型原生数据类型；子类执行任何核特定的量化
        Returns the output tensor or ``(output, lse)`` if ``return_lse``."""  # 返回输出张量，或当return_lse为True时返回``(output, lse)``
        q_scale = k_scale = v_scale = 1.0  # 初始化QKV缩放因子为1.0
        if self.data_type == torch.float8_e4m3fn:  # 如果KV缓存数据类型是FP8
            q, k, v, k_scale, v_scale = _quantize_fp8_qkv(q, k, v, layer)  # 对QKV进行FP8量化
        return flashinfer.prefill.trtllm_ragged_attention_deepseek(  # 调用FlashInfer的TRTLLM ragged注意力（DeepSeek专用）
            query=q,  # 查询张量
            key=k,  # 键张量
            value=v,  # 值张量
            workspace_buffer=self.workspace_buffer,  # 工作空间缓冲区
            batch_size=batch_size,  # 批大小
            window_left=-1,  # 滑动窗口左边界（-1表示不限制）
            enable_pdl=False,  # 不启用PDL
            max_q_len=max_q_len,  # 最大查询长度
            bmm1_scale=q_scale * k_scale * layer.scaling,  # BMM1缩放因子
            bmm2_scale=v_scale,  # BMM2缩放因子
            cum_seq_lens_q=cum_seq_lens_q,  # 查询累积序列长度
            cum_seq_lens_kv=cum_seq_lens_kv,  # 键值累积序列长度
            seq_lens=seq_lens_kv,  # 键值序列长度
            max_kv_len=max_kv_len,  # 最大键值长度
            is_causal=is_causal,  # 是否因果
            return_lse=return_lse,  # 是否返回log-sum-exp
            o_sf_scale=o_sf_scale,  # 输出缩放因子
            out=out_buffer,  # 输出缓冲区
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),  # 跳过softmax的阈值缩放因子
        )

    # 使用TRTLLM MLA核运行解码前向传播
    def forward_decode(
        self,
        q: torch.Tensor,  # q_nope  # 非旋转部分查询
        k: torch.Tensor,  # k_nope  # 非旋转部分键
        v: torch.Tensor,  # not used in this backend  # 本后端不使用
        layer: RadixAttention,  # 注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache: bool = True,  # 是否保存KV缓存 # 是否保存KV缓存，默认为True
        q_rope: Optional[torch.Tensor] = None,  # 旋转部分查询 # 旋转位置编码部分查询，默认为None
        k_rope: Optional[torch.Tensor] = None,  # 旋转部分键 # 旋转位置编码部分键，默认为None
        cos_sin_cache: Optional[torch.Tensor] = None,  # 余弦正弦缓存 # 余弦正弦缓存，默认为None
        is_neox: Optional[bool] = False,  # 是否Neox风格 # 是否Neox风格旋转位置编码，默认为False
        llama_4_scaling: Optional[torch.Tensor] = None,  # Llama4缩放因子 # Llama4缩放因子，默认为None
    ) -> torch.Tensor:
        """Run forward for decode using TRTLLM MLA kernel."""  # 使用TRTLLM MLA核运行解码前向传播
        merge_query = q_rope is not None  # 判断是否需要合并查询（有旋转部分时需要合并）
        if self.data_type == torch.float8_e4m3fn:  # 如果KV缓存数据类型是FP8
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor  # 对于FP8路径，我们量化查询和旋转部分并合并为单个张量
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend  # 注意：此trtllm_mla后端的FP8解码路径跳过了deepseek_v2.py:forward_absorb_prepare中的RoPE应用
            assert all(  # 断言所有必要参数都不为None
                x is not None for x in [q_rope, k_rope, cos_sin_cache]  # q_rope、k_rope和cos_sin_cache必须存在
            ), "For FP8 path and using flashinfer.rope.mla_rope_quantize we need all of q_rope, k_rope and cos_sin_cache to be not None."  # FP8路径使用mla_rope_quantize时所有参数必须不为None
            q, k, k_rope = mla_quantize_and_rope_for_fp8(  # 对MLA进行FP8量化和RoPE处理
                q,  # 非旋转部分查询
                q_rope,  # 旋转部分查询
                k.squeeze(1),  # 非旋转部分键（去除维度1）
                k_rope.squeeze(1),  # 旋转部分键（去除维度1）
                forward_batch.positions,  # 位置索引
                cos_sin_cache,  # 余弦正弦缓存
                is_neox,  # 是否Neox风格
                self.kv_lora_rank,  # KV LoRA秩
                self.qk_rope_head_dim,  # QK旋转头维度
            )
            merge_query = False  # FP8路径已经合并，不需要再次合并

        # Save KV cache if requested  # 如果请求则保存KV缓存
        if save_kv_cache:  # 如果需要保存KV缓存
            assert (  # 断言k和k_rope都不为None
                k is not None and k_rope is not None  # k和k_rope必须存在
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."  # 填充trtllm_mla KV缓存时k_nope和k_rope都必须不为None
            self.token_to_kv_pool.set_mla_kv_buffer(  # 设置MLA KV缓冲区
                layer, forward_batch.out_cache_loc, k, k_rope  # 层、输出缓存位置、非旋转键和旋转键
            )

        # Prepare query tensor inline  # 内联准备查询张量
        if merge_query:  # 如果需要合并查询
            # For FP16 path, we merge the query and rope parts into a single tensor  # 对于FP16路径，将查询和旋转部分合并为单个张量
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)  # 重塑非旋转部分查询
            q_rope_reshaped = q_rope.view(  # 重塑旋转部分查询
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim  # 形状
            )
            query = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)  # 拼接非旋转和旋转部分
        else:  # 不需要合并
            # For FP8 path, we already have the query and rope parts merged because of the quantize_and_rope_for_fp8 function  # 对于FP8路径，由于quantize_and_rope_for_fp8函数已经合并了查询和旋转部分
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)  # 直接重塑查询

        # Apply llama 4 scaling if provided  # 如果提供了Llama4缩放则应用
        if llama_4_scaling is not None:  # 如果有Llama4缩放因子
            query = query.to(self.q_data_type) * llama_4_scaling  # 转换数据类型并乘以缩放因子
            query = query.to(self.data_type)  # 转换回KV缓存数据类型

        # Ensure query has shape [bs, acc_q_len, num_q_heads, head_dim] when seq_len 1  # 当seq_len为1时确保查询形状为[bs, acc_q_len, num_q_heads, head_dim]
        if query.dim() == 3:  # 如果查询是3维
            query = query.unsqueeze(1)  # 在第1维添加维度

        # Prepare KV cache inline  # 内联准备KV缓存
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)  # 获取键缓存
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)  # 重塑并添加维度

        # Get metadata  # 获取元数据
        metadata = (  # 从前向批次或自身获取解码元数据
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)  # 优先从批次获取
            or self.forward_decode_metadata  # 否则使用自身的
        )

        # Ensure batch_size is sufficient, the batch size increase due to the padding from the forward batch  # 确保批大小足够，批大小因前向批次的填充而增加
        # FIXME(@rainj-me), refactor the skip_attn_backend_init, init_forward_metadata for attn backends  # FIXME(@rainj-me)，重构skip_attn_backend_init和init_forward_metadata
        # and padding logic in prepare_mlp_sync_batch to avoid this  # 以及prepare_mlp_sync_batch中的填充逻辑以避免此问题
        batch_size = getattr(metadata, "batch_size", None)  # 获取元数据中的批大小
        if batch_size is not None and batch_size < forward_batch.batch_size:  # 如果元数据批大小不足
            self.init_forward_metadata(forward_batch)  # 重新初始化前向元数据
            metadata = forward_batch.decode_trtllm_mla_metadata  # 更新元数据

        raw_out = self._run_decode_kernel(  # 运行解码核函数
            query=query,  # 查询张量
            kv_cache=kv_cache,  # KV缓存
            block_tables=metadata.block_kv_indices,  # 块表（KV索引）
            seq_lens=forward_batch.seq_lens,  # 序列长度
            max_seq_len=metadata.max_seq_len_k,  # 最大序列长度
            layer=layer,  # 注意力层
        )

        # Reshape output directly without slicing  # 直接重塑输出，无需切片
        output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)  # 重塑为2D输出
        return output  # 返回输出

    # 使用TRTLLM MLA核运行扩展（预填充）前向传播
    def forward_extend(
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache: bool = True,  # 是否保存KV缓存 # 是否保存KV缓存，默认为True
        q_rope: Optional[torch.Tensor] = None,  # 旋转部分查询 # 旋转位置编码部分查询，默认为None
        k_rope: Optional[torch.Tensor] = None,  # 旋转部分键 # 旋转位置编码部分键，默认为None
        cos_sin_cache: Optional[torch.Tensor] = None,  # 余弦正弦缓存 # 余弦正弦缓存，默认为None
        is_neox: Optional[bool] = False,  # 是否Neox风格 # 是否Neox风格旋转位置编码，默认为False
        llama_4_scaling: Optional[torch.Tensor] = None,  # Llama4缩放因子 # Llama4缩放因子，默认为None
    ) -> torch.Tensor:

        if (
            self.forward_prefill_metadata is not None  # 如果预填充元数据存在
            and self.forward_prefill_metadata.fallback_to_flashinfer_impl  # 且需要回退到FlashInfer实现
        ):
            return super().forward_extend(  # 委托给父类的扩展方法
                q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope  # 传递所有参数
            )

        # TODO refactor to avoid code duplication  # TODO重构以避免代码重复
        merge_query = q_rope is not None  # 判断是否需要合并查询
        if (
            self.data_type == torch.float8_e4m3fn  # 如果KV缓存数据类型是FP8
        ) and forward_batch.forward_mode.is_target_verify():  # 且是目标验证模式
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor  # 对于FP8路径，量化查询和旋转部分并合并为单个张量
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend  # 注意：此trtllm_mla后端的FP8解码路径跳过了deepseek_v2.py:forward_absorb_prepare中的RoPE应用
            assert all(  # 断言所有必要参数都不为None
                x is not None for x in [q_rope, k_rope, cos_sin_cache]  # q_rope、k_rope和cos_sin_cache必须存在
            ), "For FP8 path and using flashinfer.rope.mla_rope_quantize we need all of q_rope, k_rope and cos_sin_cache to be not None."  # FP8路径使用mla_rope_quantize时所有参数必须不为None
            q, k, k_rope = mla_quantize_and_rope_for_fp8(  # 对MLA进行FP8量化和RoPE处理
                q,  # 非旋转部分查询
                q_rope,  # 旋转部分查询
                k.squeeze(1),  # 非旋转部分键（去除维度1）
                k_rope.squeeze(1),  # 旋转部分键（去除维度1）
                forward_batch.positions,  # 位置索引
                cos_sin_cache,  # 余弦正弦缓存
                is_neox,  # 是否Neox风格
                self.kv_lora_rank,  # KV LoRA秩
                self.qk_rope_head_dim,  # QK旋转头维度
            )
            merge_query = False  # FP8路径已经合并，不需要再次合并

        # Save KV cache if requested  # 如果请求则保存KV缓存
        if save_kv_cache:  # 如果需要保存KV缓存
            assert (  # 断言k和k_rope都不为None
                k is not None and k_rope is not None  # k和k_rope必须存在
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."  # 填充trtllm_mla KV缓存时k_nope和k_rope都必须不为None
            self.token_to_kv_pool.set_mla_kv_buffer(  # 设置MLA KV缓冲区
                layer, forward_batch.out_cache_loc, k, k_rope  # 层、输出缓存位置、非旋转键和旋转键
            )

        # TODO refactor to avoid code duplication  # TODO重构以避免代码重复
        # Prepare query tensor inline  # 内联准备查询张量
        if merge_query:  # 如果需要合并查询
            # For FP16 path, we merge the query and rope parts into a single tensor  # 对于FP16路径，将查询和旋转部分合并为单个张量
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)  # 重塑非旋转部分查询
            q_rope_reshaped = q_rope.view(  # 重塑旋转部分查询
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim  # 形状
            )
            q = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)  # 拼接非旋转和旋转部分

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)  # 重塑查询张量

        # Apply llama 4 scaling if provided  # 如果提供了Llama4缩放则应用
        if llama_4_scaling is not None:  # 如果有Llama4缩放因子
            q = q.to(self.q_data_type) * llama_4_scaling  # 转换数据类型并乘以缩放因子
            q = q.to(self.data_type)  # 转换回KV缓存数据类型

        if (
            forward_batch.forward_mode.is_target_verify()  # 如果是目标验证模式
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)  # 或草稿扩展模式
        ):
            metadata = (  # 获取解码元数据
                getattr(forward_batch, "decode_trtllm_mla_metadata", None)  # 优先从批次获取
                or self.forward_decode_metadata  # 否则使用自身的
            )

            # Ensure batch_size is sufficient, the batch size increase due to the padding from the forward batch  # 确保批大小足够，批大小因前向批次的填充而增加
            # FIXME(@rainj-me), refactor the skip_attn_backend_init, init_forward_metadata for attn backends  # FIXME(@rainj-me)，重构skip_attn_backend_init和init_forward_metadata
            # and padding logic in prepare_mlp_sync_batch to avoid this  # 以及prepare_mlp_sync_batch中的填充逻辑以避免此问题
            batch_size = getattr(metadata, "batch_size", None)  # 获取元数据中的批大小
            if batch_size is not None and batch_size < forward_batch.batch_size:  # 如果元数据批大小不足
                self.init_forward_metadata(forward_batch)  # 重新初始化前向元数据
                metadata = forward_batch.decode_trtllm_mla_metadata  # 更新元数据

            # Ensure query has shape [bs, num_draft_tokens, num_q_heads, head_dim]  # 确保查询形状为[bs, num_draft_tokens, num_q_heads, head_dim]
            bs = forward_batch.batch_size  # 获取批大小

            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)  # 获取键缓存
            kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)  # 重塑并添加维度

            q = q.to(self.data_type)  # 将查询转换为KV缓存数据类型

            if forward_batch.forward_mode.is_target_verify():  # 如果是目标验证模式
                max_seq_len = (  # 计算最大序列长度
                    metadata.max_seq_len_k + forward_batch.spec_info.draft_token_num  # 键最大序列长度+草稿token数
                )
                # For target_verify, all sequences have the same number of draft tokens  # 对于target_verify，所有序列有相同数量的草稿token
                q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)  # 重塑查询为4D
                needs_unpad = False  # 不需要反填充
            else:  # 草稿扩展模式
                # draft_extend: handle varying num_correct_drafts_per_req. If total_tokens % bs == 0,  # 草稿扩展：处理变化的num_correct_drafts_per_req。如果total_tokens % bs == 0，
                # we can directly reshape q; otherwise, pad to max_seq_len_q.  # 可以直接重塑q；否则填充到max_seq_len_q
                total_tokens = q.shape[0]  # 获取总token数
                tokens_per_seq = total_tokens // bs if bs > 0 else 0  # 计算每序列token数
                can_direct_view = bs > 0 and (total_tokens % bs == 0)  # 判断是否可以直接重塑

                if can_direct_view:  # 如果可以直接重塑
                    max_seq_len = metadata.max_seq_len_k + tokens_per_seq  # 计算最大序列长度
                    q = q.view(bs, tokens_per_seq, layer.tp_q_head_num, layer.head_dim)  # 重塑查询为4D
                    needs_unpad = False  # 不需要反填充
                else:  # 不能直接重塑，需要填充
                    # Varying lengths: pad q to (bs, max_seq_len_q, ...)  # 变长情况：填充q到(bs, max_seq_len_q, ...)
                    actual_seq_lens_q = forward_batch.extend_seq_lens  # 实际查询序列长度
                    actual_max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)  # 实际最大查询序列长度
                    max_seq_len = metadata.max_seq_len_k + actual_max_seq_len_q  # 计算最大序列长度

                    actual_cu_seqlens_q = torch.nn.functional.pad(  # 计算实际累积序列长度并填充
                        torch.cumsum(actual_seq_lens_q, dim=0, dtype=torch.int32),  # 累积求和
                        (1, 0),  # 在左侧填充一个零
                    )

                    if self.padded_q_buffer is not None:  # 如果有预分配的填充查询缓冲区
                        padded_q = self.padded_q_buffer[  # 使用预分配缓冲区
                            :bs, :actual_max_seq_len_q, :, :  # 截取对应大小
                        ].to(dtype=q.dtype)  # 转换数据类型
                        padded_q.zero_()  # 清零
                    else:  # 没有预分配缓冲区
                        padded_q = torch.zeros(  # 动态创建填充查询张量
                            (
                                bs,  # 批大小
                                actual_max_seq_len_q,  # 最大查询序列长度
                                layer.tp_q_head_num,  # 查询头数
                                layer.head_dim,  # 头维度
                            ),
                            dtype=q.dtype,  # 数据类型
                            device=q.device,  # 设备
                        )

                    q = self.pad_draft_extend_query(  # 调用填充方法
                        q, padded_q, actual_seq_lens_q, actual_cu_seqlens_q  # 传递参数
                    )
                    needs_unpad = True  # 需要反填充
                    unpad_seq_lens_q = actual_seq_lens_q  # 保存反填充用的序列长度
                    unpad_cu_seqlens_q = actual_cu_seqlens_q  # 保存反填充用的累积序列长度
                    unpad_sum_seq_lens_q = total_tokens  # 保存反填充用的总token数

            assert kv_cache.dtype == self.data_type  # 断言KV缓存数据类型匹配

            raw_out = self._run_decode_kernel(  # 运行解码核函数
                query=q,  # 查询张量
                kv_cache=kv_cache,  # KV缓存
                block_tables=metadata.block_kv_indices,  # 块表
                seq_lens=metadata.seq_lens_k,  # 键序列长度
                max_seq_len=max_seq_len,  # 最大序列长度
                layer=layer,  # 注意力层
            )

            if needs_unpad:  # 如果需要反填充
                # Unpad the output for draft_extend mode with varying lengths  # 对变长的草稿扩展模式进行反填充输出
                # Use the actual values computed during padding, not from metadata  # 使用填充期间计算的实际值，而非元数据中的值
                output = self.unpad_draft_extend_output(  # 调用反填充方法
                    raw_out,  # 原始输出
                    unpad_cu_seqlens_q,  # 累积序列长度
                    unpad_seq_lens_q,  # 序列长度
                    unpad_sum_seq_lens_q,  # 总token数
                )
                output = output.view(-1, layer.tp_q_head_num * layer.v_head_dim)  # 重塑为2D
            else:  # 不需要反填充
                output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)  # 直接重塑为2D
            return output  # 返回输出

        if k_rope is not None:  # 如果有旋转部分键
            k = torch.cat([k, k_rope], dim=-1)  # 将非旋转和旋转部分键拼接
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)  # 重塑键张量
        v = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)  # 重塑值张量

        # When chunked prefix cache is enabled, dispatch to different path for ragged attention.  # 启用分块前缀缓存时，分派到不同的ragged注意力路径
        if forward_batch.attn_attend_prefix_cache:  # 如果需要处理分块前缀缓存
            # MHA for chunked prefix kv cache when running model with MLA  # MLA模型运行时分块前缀KV缓存的MHA处理
            assert forward_batch.prefix_chunk_idx is not None  # 断言前缀分块索引存在
            assert forward_batch.prefix_chunk_cu_seq_lens is not None  # 断言前缀分块累积序列长度存在
            assert q_rope is None  # 断言旋转部分查询为None
            assert k_rope is None  # 断言旋转部分键为None
            chunk_idx = forward_batch.prefix_chunk_idx  # 获取当前分块索引

            out = torch.empty(  # 创建空输出张量
                q.shape[0],  # 查询token数
                layer.tp_q_head_num,  # 查询头数
                layer.v_head_dim,  # 值头维度
                dtype=self.q_data_type,  # 查询数据类型
                device=q.device,  # 设备
            )
            result = self._run_prefill_kernel(  # 运行预填充核函数
                q=q,  # 查询张量
                k=k,  # 键张量
                v=v,  # 值张量
                layer=layer,  # 注意力层
                batch_size=forward_batch.batch_size,  # 批大小
                cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,  # 查询累积序列长度
                max_q_len=self.forward_prefill_metadata.max_seq_len,  # 最大查询长度
                seq_lens_kv=forward_batch.prefix_chunk_seq_lens[chunk_idx],  # 分块键值序列长度
                cum_seq_lens_kv=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],  # 分块键值累积序列长度
                max_kv_len=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],  # 分块最大键值长度
                is_causal=False,  # 分块前缀缓存非因果
                return_lse=True,  # 返回log-sum-exp
                out_buffer=out,  # 输出缓冲区
                o_sf_scale=-1.0,  # 输出缩放因子设为-1.0
            )

            # The TRT-LLM ragged attention cubin kernel does not correctly  # TRT-LLM ragged注意力cubin核不能正确
            # handle rows with kv_len == 0: it leaves stale data in the  # 处理kv_len == 0的行：它在
            # workspace softmaxStats buffer and may produce non-zero output  # 工作空间softmaxStats缓冲区中留下陈旧数据并可能产生非零输出
            # for those rows.  Fix up by forcing out=0 and lse=-inf for  # 对于这些行。通过强制out=0和lse=-inf来修复
            # zero-KV rows so that downstream merge_state ignores them.  # 零KV行，以便下游merge_state忽略它们
            # Skip entirely when this chunk has no zero-KV rows (pure CPU  # 当此分块没有零KV行时完全跳过（纯CPU
            # check, precomputed in prepare_chunked_prefix_cache_info).  # 检查，在prepare_chunked_prefix_cache_info中预计算）
            if forward_batch.prefix_chunk_has_zero_kv[chunk_idx]:  # 如果当前分块有零KV行
                out_tensor, lse_tensor = result  # 解包输出和log-sum-exp
                fixup_zero_kv_rows(  # 修复零KV行
                    out_tensor,  # 输出张量
                    lse_tensor,  # log-sum-exp张量
                    forward_batch.prefix_chunk_seq_lens[chunk_idx],  # 分块键值序列长度
                    self.forward_prefill_metadata.cum_seq_lens,  # 查询累积序列长度
                    self.forward_prefill_metadata.max_seq_len,  # 最大查询长度
                )

            return result  # 返回结果
        else:  # 不需要处理分块前缀缓存
            out = torch.empty(  # 创建空输出张量
                q.shape[0],  # 查询token数
                q.shape[1],  # 查询头数
                v.shape[2],  # 值头维度
                device=q.device,  # 设备
                dtype=self.q_data_type,  # 查询数据类型
            )
            return self._run_prefill_kernel(  # 运行预填充核函数
                q=q,  # 查询张量
                k=k,  # 键张量
                v=v,  # 值张量
                layer=layer,  # 注意力层
                batch_size=forward_batch.batch_size,  # 批大小
                cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,  # 查询累积序列长度
                max_q_len=self.forward_prefill_metadata.max_seq_len,  # 最大查询长度
                seq_lens_kv=self.forward_prefill_metadata.seq_lens,  # 键值序列长度
                cum_seq_lens_kv=self.forward_prefill_metadata.cum_seq_lens,  # 键值累积序列长度
                max_kv_len=self.forward_prefill_metadata.max_seq_len,  # 最大键值长度
                is_causal=True,  # 因果注意力
                return_lse=forward_batch.mha_return_lse,  # 是否返回log-sum-exp
                out_buffer=out,  # 输出缓冲区
                o_sf_scale=1.0,  # 输出缩放因子设为1.0
            )


# TRT-LLM MLA多步草稿后端，用于EAGLE推测解码
class TRTLLMMLAMultiStepDraftBackend(FlashInferMLAMultiStepDraftBackend):
    """Multi-step draft backend for TRT-LLM MLA used by EAGLE."""  # EAGLE使用的TRT-LLM MLA多步草稿后端

    # Per-step draft decode never reads seq_lens_cpu / seq_lens_sum; opt out so  # 每步草稿解码从不读取seq_lens_cpu/seq_lens_sum；选择退出以便
    # decide_needs_cpu_seq_lens' OR over the backends stays False.  # decide_needs_cpu_seq_lens在各后端上的OR运算保持为False
    needs_cpu_seq_lens: bool = False  # 是否需要CPU序列长度 # 不需要CPU序列长度同步

    def __init__(
        self,
        model_runner: "ModelRunner",  # 模型运行器
        topk: int,  # Top-K值
        speculative_num_steps: int,  # 推测步数
        backend: str = "trtllm-gen",  # 后端类型 # 后端类型，默认为trtllm-gen
    ):
        super().__init__(model_runner, topk, speculative_num_steps)  # 调用父类初始化

        for i in range(self.speculative_num_steps - 1):  # 为每一步（除最后一步外）创建注意力后端
            self.attn_backends[i] = TRTLLMMLABackend(  # 使用TRTLLM MLA后端
                model_runner,  # 模型运行器
                skip_prefill=True,  # 跳过预填充
                kv_indptr_buf=self.kv_indptr[i],  # KV索引指针缓冲区
                q_indptr_decode_buf=self.q_indptr_decode,  # 解码查询索引指针缓冲区
                backend=backend,  # 后端类型
            )

    # 初始化前向传播的元数据
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):  # 为每一步初始化元数据
            self.attn_backends[i].init_forward_metadata(forward_batch)  # 调用每步后端的初始化

    # 使用新输入重放CUDA图
    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int  # 前向批次和批大小
    ):
        for i in range(self.speculative_num_steps - 1):  # 为每一步重放CUDA图
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(  # 调用每步后端的重放方法
                bs,  # 批大小
                forward_batch.req_pool_indices,  # 请求池索引
                forward_batch.seq_lens,  # 序列长度
                seq_lens_sum=None,  # 序列长度总和为None
                encoder_lens=None,  # 编码器长度为None
                forward_mode=ForwardMode.DECODE,  # 前向模式为解码
                spec_info=forward_batch.spec_info,  # 推测信息
                seq_lens_cpu=forward_batch.seq_lens_cpu,  # CPU上的序列长度
            )  # 结束init_forward_metadata_replay_cuda_graph调用
