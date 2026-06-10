# Wave注意力后端模块
# 实现基于Wave语言的注意力后端，用于AMD GPU上的高效注意力计算，
# 包含Triton内核计算KV分片数、前向元数据管理以及扩展/解码注意力前向传播

from __future__ import annotations  # 启用延迟注解评估

import logging  # 日志模块
from dataclasses import dataclass  # 数据类装饰器
from typing import TYPE_CHECKING, Optional  # 类型提示

import torch  # PyTorch核心库
import triton  # Triton编译器
import triton.language as tl  # Triton语言

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend  # 注意力后端基类
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton  # 创建KV索引的Triton内核
from sglang.srt.layers.dp_attention import get_attention_tp_size  # 获取张量并行大小
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode  # 前向批次信息
from sglang.srt.utils import get_bool_env_var, get_device_core_count  # 环境变量和设备核心数工具

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.layers.radix_attention import RadixAttention  # 基数注意力
    from sglang.srt.model_executor.model_runner import ModelRunner  # 模型运行器
    from sglang.srt.speculative.spec_info import SpecInput  # 推测解码输入

logger = logging.getLogger(__name__)  # 获取模块日志器


@triton.jit  # Triton JIT编译的KV分片数计算内核
def get_num_kv_splits_triton(  # 计算每个序列的KV分片数
    num_kv_splits_ptr,  # KV分片数输出指针
    seq_lens_ptr,  # 序列长度输入指针
    num_seq,  # 序列数
    num_group,  # 组数
    num_head,  # 头数
    num_kv_head,  # KV头数
    max_kv_splits,  # 最大KV分片数
    device_core_count,  # 设备核心数
    MAX_NUM_SEQ: tl.constexpr,  # 最大序列数（编译时常量）
):
    # TODO: this method is tunable, we need more online serving data to tune it
    # TODO: 此方法可调优，需要更多在线服务数据来调优
    offs_seq = tl.arange(0, MAX_NUM_SEQ)  # 序列偏移量
    mask_seq = offs_seq < num_seq  # 有效序列掩码

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)  # 加载序列长度
    max_seq_len = tl.max(seq_lens)  # 最大序列长度
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)  # 重新加载（无效位置用最大值填充）
    min_seq_len = tl.min(seq_lens)  # 最小序列长度
    if max_seq_len * 8 < min_seq_len * 10:  # 如果最大和最小相近
        min_seq_len = max_seq_len  # 使用最大值作为最小值
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)  # 策略1：基于序列长度比
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)  # 策略1的块大小

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    # 注意：这是一个技巧，让num_kv_split随序列长度逐渐增长
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0  # 扩展序列长度
    ext_device_core_count = tl.cast(  # 扩展设备核心数
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32  # 核心数 * log2(序列长度)
    )
    block_h, num_kv_group = 16, num_head // num_kv_head  # 头块大小和KV组数
    if num_kv_group == 1:  # 如果是MQA
        token_grid = num_seq * num_group * num_head  # token网格大小
    else:  # GQA
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        # 来自triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        block_h = tl.minimum(block_h, num_kv_group)  # 限制块大小
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)  # token网格大小
    max_kv_splits_2 = tl.minimum(  # 策略2：基于设备核心数
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits  # 核心数/token网格
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)  # 策略2的块大小

    num_kv_splits = tl.maximum(  # 取两种策略的最大值
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)  # 两种策略的分片数
    )

    offs_token = offs_seq * num_group  # token偏移
    mask_token = offs_token < num_seq * num_group  # 有效token掩码
    for i in range(0, num_group):  # 遍历每个组
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)  # 存储KV分片数


@dataclass  # 前向元数据数据类，保存注意力计算所需的中间变量
class ForwardMetadata:
    attn_logits: torch.Tensor  # 注意力logits
    attn_lse: torch.Tensor  # 注意力log-softmax值
    max_extend_len: int  # 最大扩展长度
    num_kv_splits: torch.Tensor  # KV分片数
    kv_indptr: torch.Tensor  # KV索引指针
    kv_indices: torch.Tensor  # KV索引
    qo_indptr: torch.Tensor  # 查询输出索引指针
    custom_mask: torch.Tensor  # 自定义掩码
    mask_indptr: torch.Tensor  # 掩码索引指针


class WaveAttnBackend(AttentionBackend):  # Wave注意力后端，基于Wave语言实现
    def __init__(  # 初始化方法
        self,
        model_runner: ModelRunner,  # 模型运行器
        skip_prefill: bool = False,  # 是否跳过预填充
        kv_indptr_buf: Optional[torch.Tensor] = None,  # KV索引指针缓冲区
    ):
        # Lazy import to avoid the initialization of cuda context
        # 延迟导入以避免CUDA上下文的初始化
        from sglang.srt.layers.attention.wave_ops.decode_attention import (
            decode_attention_fwd,  # 解码注意力前向
        )
        from sglang.srt.layers.attention.wave_ops.extend_attention import (
            extend_attention_wave,  # 扩展注意力前向
        )

        super().__init__()  # 调用父类初始化

        # Set unique cache dir for each process to avoid cache write races
        # 为每个进程设置唯一的缓存目录以避免缓存写入竞争
        import wave_lang.kernel.wave.cache as cache  # Wave缓存模块

        base_cache_dir = cache.CACHE_BASE_DIR  # 基础缓存目录
        new_dir = base_cache_dir / f"worker_{model_runner.tp_rank}"  # 按TP秩创建子目录
        logger.info(f"Setting Wave cache dir: {new_dir}")  # 记录缓存目录
        cache.CACHE_BASE_DIR = new_dir  # 设置新缓存目录

        self.decode_attention_fwd = decode_attention_fwd  # 保存解码注意力前向函数
        self.extend_attention_fwd = extend_attention_wave  # 保存扩展注意力前向函数

        self.skip_prefill = skip_prefill  # 是否跳过预填充

        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        # 池引用 - 在构造时捕获，以便在ForwardBatch字段删除后仍能存活。
        self.req_to_token_pool = model_runner.req_to_token_pool  # 请求到token的映射池
        self.token_to_kv_pool = model_runner.token_to_kv_pool  # token到KV的映射池

        max_bs = model_runner.req_to_token_pool.size  # 最大批量大小

        if kv_indptr_buf is None:  # 如果没有提供KV索引指针缓冲区
            self.kv_indptr = torch.zeros(  # 创建零张量
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device  # 形状为[max_bs+1]
            )
        else:  # 使用提供的缓冲区
            self.kv_indptr = kv_indptr_buf  # 使用外部缓冲区

        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 请求到token映射表

        if not self.skip_prefill:  # 如果不跳过预填充
            self.qo_indptr = torch.zeros(  # 查询输出索引指针
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device  # 形状为[max_bs+1]
            )

            self.mask_indptr = torch.zeros(  # 掩码索引指针
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device  # int64类型
            )

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens  # 推测解码草稿token数

        self.num_head = (  # 本地注意力头数
            model_runner.model_config.num_attention_heads // get_attention_tp_size()  # 总头数/TP大小
        )
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(  # 本地KV头数
            get_attention_tp_size()  # TP大小
        )

        self.static_kv_splits = get_bool_env_var(  # 是否使用静态KV分片
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"  # 默认为false
        )
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits  # 最大KV分片数
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]  # V头维度

        self.forward_metadata: ForwardMetadata = None  # 前向元数据

        self.max_context_len = model_runner.model_config.context_len  # 最大上下文长度

        self.device = model_runner.device  # 设备
        self.device_core_count = get_device_core_count(model_runner.gpu_id)  # 设备核心数

    def get_num_kv_splits(  # 计算KV分片数
        self,
        num_kv_splits: torch.Tensor,  # KV分片数输出张量
        seq_lens: torch.Tensor,  # 序列长度张量
    ):
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]  # token数和序列数
        num_group = num_token // num_seq  # 每个序列的组数

        assert (  # 断言token数可被序列数整除
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"  # 序列数与token数不匹配

        if self.static_kv_splits or self.device_core_count <= 0:  # 静态分片或无核心信息
            num_kv_splits.fill_(self.max_kv_splits)  # 使用最大分片数
            return  # 直接返回

        if num_seq < 256:  # 小批量
            SCHEDULE_SEQ = 256  # 使用256
        else:  # 大批量
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)  # 使用最近的2的幂

        get_num_kv_splits_triton[(1,)](  # 调用Triton内核计算分片数
            num_kv_splits,  # 输出分片数
            seq_lens,  # 序列长度
            num_seq,  # 序列数
            num_group,  # 组数
            self.num_head,  # 头数
            self.num_kv_head,  # KV头数
            self.max_kv_splits,  # 最大分片数
            self.device_core_count,  # 设备核心数
            MAX_NUM_SEQ=SCHEDULE_SEQ,  # 调度序列数
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):  # 初始化Wave注意力后端的前向元数据
        """Init auxiliary variables for wave attention backend.
        初始化Wave注意力后端的辅助变量。"""

        bs = forward_batch.batch_size  # 批量大小
        kv_indptr = self.kv_indptr  # KV索引指针
        spec_info = forward_batch.spec_info  # 推测解码信息

        if forward_batch.forward_mode.is_decode_or_idle():  # 解码或空闲模式
            if spec_info is None:  # 无推测解码
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)  # 计算KV索引指针
                kv_indptr = kv_indptr[: bs + 1]  # 截取有效部分
                kv_indices = torch.empty(  # 创建KV索引
                    forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device  # 大小为序列长度总和
                )
                create_flashinfer_kv_indices_triton[(bs,)](  # Triton内核创建KV索引
                    self.req_to_token,  # 请求到token映射
                    forward_batch.req_pool_indices,  # 请求池索引
                    forward_batch.seq_lens,  # 序列长度
                    kv_indptr,  # KV索引指针
                    None,  # 无编码器长度
                    kv_indices,  # 输出KV索引
                    self.req_to_token.stride(0),  # 步长
                )
            else:  # 有推测解码
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices  # 使用spec_info的索引
                bs = kv_indptr.shape[0] - 1  # 更新批量大小

            from sglang.srt.layers.attention.wave_ops.decode_attention import (
                decode_attention_intermediate_arrays_shapes,  # 解码注意力中间数组形状
            )

            attn_logits_shape, attn_logits_max_shape = (  # 获取中间数组形状
                decode_attention_intermediate_arrays_shapes(
                    bs, self.v_head_dim, self.num_head, self.max_kv_splits  # 批量、头维度、头数、分片数
                )
            )
            attn_logits = torch.empty(  # 创建注意力logits
                attn_logits_shape,  # 形状
                dtype=torch.float32,  # float32
                device=self.device,  # 设备
            )
            attn_lse = torch.empty(  # 创建注意力LSE
                attn_logits_max_shape,  # 形状
                dtype=torch.float32,  # float32
                device=self.device,  # 设备
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)  # KV分片数

            self.get_num_kv_splits(num_kv_splits, forward_batch.seq_lens)  # 计算KV分片数

            qo_indptr = None  # 查询输出索引指针
            custom_mask = None  # 自定义掩码
            mask_indptr = None  # 掩码索引指针
            max_extend_len = None  # 最大扩展长度
        elif forward_batch.forward_mode.is_target_verify():  # 目标验证模式
            bs = len(forward_batch.req_pool_indices)  # 批量大小
            qo_indptr = torch.arange(  # 创建查询输出索引指针
                0,  # 起始
                (1 + bs) * self.num_draft_tokens,  # 结束
                step=self.num_draft_tokens,  # 步长为草稿token数
                dtype=torch.int32,  # int32
                device=self.device,  # 设备
            )
            # Different with flashinfer kv_indptr and kv_indices construction
            # 与flashinfer的kv_indptr和kv_indices构造不同
            kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)  # 计算KV索引指针
            kv_indptr = kv_indptr[: bs + 1]  # 截取有效部分
            kv_indices = torch.empty(  # 创建KV索引
                kv_indptr[-1], dtype=torch.int32, device=self.device  # 大小为最后一个指针值
            )
            create_flashinfer_kv_indices_triton[(bs,)](  # Triton内核创建KV索引
                self.req_to_token,  # 请求到token映射
                forward_batch.req_pool_indices,  # 请求池索引
                forward_batch.seq_lens,  # 序列长度
                kv_indptr,  # KV索引指针
                None,  # 无编码器长度
                kv_indices,  # 输出KV索引
                self.req_to_token.stride(0),  # 步长
            )

            custom_mask = spec_info.custom_mask  # 获取自定义掩码
            seq_mask_len = self.num_draft_tokens * (  # 序列掩码长度
                forward_batch.seq_lens + self.num_draft_tokens  # (序列长度 + 草稿数) * 草稿数
            )
            mask_indptr = self.mask_indptr  # 掩码索引指针
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)  # 计算掩码索引指针
            mask_indptr = mask_indptr[: bs + 1]  # 截取有效部分
            max_extend_len = self.num_draft_tokens  # 最大扩展长度为草稿token数
            num_kv_splits = None  # 无KV分片数
            attn_logits = None  # 无注意力logits
            attn_lse = None  # 无注意力LSE
        elif forward_batch.forward_mode.is_draft_extend():  # 草稿扩展模式
            kv_indices, kv_indptr, qo_indptr, custom_mask = (  # 生成注意力参数
                spec_info.generate_attn_arg_prefill(  # 生成预填充注意力参数
                    forward_batch.req_pool_indices,  # 请求池索引
                    forward_batch.seq_lens,  # 序列长度
                    None,  # 无编码器长度
                    self.req_to_token,  # 请求到token映射
                )
            )
            mask_indptr = None  # 无掩码索引指针
            # TODO(FIXME): This will trigger an invalid Eagle tree when using
            # `max(spec_info.num_accept_tokens_cpu)`.
            # It might have been forgotten to update somewhere.
            # TODO(FIXME): 使用`max(spec_info.num_accept_tokens_cpu)`时会触发无效的Eagle树。
            # 可能某处忘记更新了。
            max_extend_len = torch.max(spec_info.num_accept_tokens).item()  # 最大扩展长度
            num_kv_splits = None  # 无KV分片数
            attn_logits = None  # 无注意力logits
            attn_lse = None  # 无注意力LSE
        else:  # 标准扩展模式
            kv_indptr[1 : bs + 1] = torch.cumsum(  # 计算KV索引指针
                forward_batch.extend_prefix_lens, dim=0  # 使用扩展前缀长度
            )
            kv_indptr = kv_indptr[: bs + 1]  # 截取有效部分
            kv_indices = torch.empty(  # 创建KV索引
                forward_batch.extend_prefix_lens.sum().item(),  # 大小为前缀长度总和
                dtype=torch.int32,  # int32
                device=self.device,  # 设备
            )
            create_flashinfer_kv_indices_triton[(bs,)](  # Triton内核创建KV索引
                self.req_to_token,  # 请求到token映射
                forward_batch.req_pool_indices,  # 请求池索引
                forward_batch.extend_prefix_lens,  # 扩展前缀长度
                kv_indptr,  # KV索引指针
                None,  # 无编码器长度
                kv_indices,  # 输出KV索引
                self.req_to_token.stride(0),  # 步长
            )

            qo_indptr = self.qo_indptr  # 查询输出索引指针
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)  # 计算指针
            qo_indptr = qo_indptr[: bs + 1]  # 截取有效部分
            custom_mask = None  # 无自定义掩码
            mask_indptr = None  # 无掩码索引指针
            attn_logits = None  # 无注意力logits
            attn_lse = None  # 无注意力LSE
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()  # 最大扩展长度
            num_kv_splits = None  # 无KV分片数

        self.forward_metadata = ForwardMetadata(  # 构造前向元数据
            attn_logits,  # 注意力logits
            attn_lse,  # 注意力LSE
            max_extend_len,  # 最大扩展长度
            num_kv_splits,  # KV分片数
            kv_indptr,  # KV索引指针
            kv_indices,  # KV索引
            qo_indptr,  # 查询输出索引指针
            custom_mask,  # 自定义掩码
            mask_indptr,  # 掩码索引指针
        )

    def init_cuda_graph_state(  # 初始化CUDA图状态
        self,
        max_bs: int,  # 最大批量大小
        max_num_tokens: int,  # 最大token数
        kv_indices_buf: Optional[torch.Tensor] = None,  # KV索引缓冲区
    ):
        from sglang.srt.layers.attention.wave_ops.decode_attention import (
            decode_attention_intermediate_arrays_shapes,  # 解码注意力中间数组形状
        )

        attn_logits_shape, attn_logits_max_shape = (  # 获取中间数组形状
            decode_attention_intermediate_arrays_shapes(
                max_bs, self.v_head_dim, self.num_head, self.max_kv_splits  # 最大批量、头维度、头数、分片数
            )
        )
        self.cuda_graph_attn_logits = torch.zeros(  # CUDA图注意力logits
            attn_logits_shape,  # 形状
            dtype=torch.float32,  # float32
            device=self.device,  # 设备
        )
        self.cuda_graph_attn_lse = torch.zeros(  # CUDA图注意力LSE
            attn_logits_max_shape,  # 形状
            dtype=torch.float32,  # float32
            device=self.device,  # 设备
        )
        self.cuda_graph_num_kv_splits = torch.full(  # CUDA图KV分片数
            (max_bs,), self.max_kv_splits, dtype=torch.int32, device=self.device  # 全部填充最大分片数
        )
        if kv_indices_buf is None:  # 如果没有提供KV索引缓冲区
            self.cuda_graph_kv_indices = torch.zeros(  # 创建CUDA图KV索引
                (max_bs * self.max_context_len),  # 大小为最大批量*最大上下文长度
                dtype=torch.int32,  # int32
                device=self.device,  # 设备
            )
        else:  # 使用提供的缓冲区
            self.cuda_graph_kv_indices = kv_indices_buf  # 使用外部缓冲区

        if not self.skip_prefill:  # 如果不跳过预填充
            self.cuda_graph_custom_mask = torch.zeros(  # CUDA图自定义掩码
                (max_bs * self.max_context_len),  # 大小为最大批量*最大上下文长度
                dtype=torch.uint8,  # uint8
                device=self.device,  # 设备
            )

    def _build_cuda_graph_forward_metadata(  # 构建CUDA图前向元数据
        self,
        bs: int,  # 批量大小
        forward_mode: ForwardMode,  # 前向模式
        spec_info: Optional[SpecInput],  # 推测解码输入
    ) -> ForwardMetadata:  # 返回前向元数据
        if forward_mode.is_decode_or_idle():  # 解码或空闲模式
            return ForwardMetadata(  # 返回解码模式元数据
                attn_logits=self.cuda_graph_attn_logits,  # CUDA图logits
                attn_lse=self.cuda_graph_attn_lse,  # CUDA图LSE
                max_extend_len=None,  # 无最大扩展长度
                num_kv_splits=self.cuda_graph_num_kv_splits,  # CUDA图KV分片数
                kv_indptr=self.kv_indptr[: bs + 1],  # KV索引指针
                kv_indices=self.cuda_graph_kv_indices,  # CUDA图KV索引
                qo_indptr=None,  # 无查询输出指针
                custom_mask=None,  # 无自定义掩码
                mask_indptr=None,  # 无掩码索引指针
            )
        elif forward_mode.is_target_verify():  # 目标验证模式
            return ForwardMetadata(  # 返回验证模式元数据
                attn_logits=None,  # 无注意力logits
                attn_lse=None,  # 无注意力LSE
                max_extend_len=self.num_draft_tokens,  # 最大扩展长度为草稿token数
                num_kv_splits=None,  # 无KV分片数
                kv_indptr=self.kv_indptr[: bs + 1],  # KV索引指针
                kv_indices=self.cuda_graph_kv_indices,  # CUDA图KV索引
                qo_indptr=self.qo_indptr[: bs + 1],  # 查询输出指针
                custom_mask=self.cuda_graph_custom_mask,  # CUDA图自定义掩码
                mask_indptr=self.mask_indptr[: bs + 1],  # 掩码索引指针
            )
        else:  # 无效模式
            raise ValueError(f"Invalid forward mode: {forward_mode=} for CUDA Graph.")  # CUDA图的前向模式无效

    def init_forward_metadata_capture_cuda_graph(  # 捕获CUDA图时初始化前向元数据
        self,
        bs: int,  # 批量大小
        num_tokens: int,  # token数
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        forward_mode: ForwardMode,  # 前向模式
        spec_info: Optional[SpecInput],  # 推测解码输入
    ):
        assert encoder_lens is None, "Not supported"  # 不支持编码器长度

        # Multi-step speculative decode: kv buffers come from spec_info rather than
        # the cuda-graph pool, so replay is not involved for this path.
        # 多步推测解码：KV缓冲区来自spec_info而非CUDA图池，因此此路径不涉及重放。
        if forward_mode.is_decode_or_idle() and spec_info is not None:  # 带推测的解码模式
            self.forward_metadata = ForwardMetadata(  # 使用spec_info的元数据
                attn_logits=self.cuda_graph_attn_logits,  # CUDA图logits
                attn_lse=self.cuda_graph_attn_lse,  # CUDA图LSE
                max_extend_len=None,  # 无最大扩展长度
                num_kv_splits=self.cuda_graph_num_kv_splits,  # CUDA图KV分片数
                kv_indptr=spec_info.kv_indptr,  # spec_info的KV索引指针
                kv_indices=spec_info.kv_indices,  # spec_info的KV索引
                qo_indptr=None,  # 无查询输出指针
                custom_mask=None,  # 无自定义掩码
                mask_indptr=None,  # 无掩码索引指针
            )
            return  # 直接返回

        self.init_forward_metadata_replay_cuda_graph(  # 初始化重放元数据
            bs=bs,  # 批量大小
            req_pool_indices=req_pool_indices,  # 请求池索引
            seq_lens=seq_lens,  # 序列长度
            seq_lens_sum=None,  # 无序列长度总和
            encoder_lens=encoder_lens,  # 编码器长度
            forward_mode=forward_mode,  # 前向模式
            spec_info=spec_info,  # 推测解码输入
            seq_lens_cpu=None,  # 无CPU序列长度
        )
        self.forward_metadata = self._build_cuda_graph_forward_metadata(  # 构建CUDA图元数据
            bs, forward_mode, spec_info  # 批量大小、前向模式、推测输入
        )

    def init_forward_metadata_replay_cuda_graph(  # 重放CUDA图时初始化前向元数据
        self,
        bs: int,  # 批量大小
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        seq_lens_sum: int,  # 序列长度总和
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        forward_mode: ForwardMode,  # 前向模式
        spec_info: Optional[SpecInput],  # 推测解码输入
        seq_lens_cpu: Optional[torch.Tensor],  # CPU上的序列长度
    ):
        if forward_mode.is_decode_or_idle():  # 解码或空闲模式
            kv_indptr = self.kv_indptr  # KV索引指针
            kv_indices = self.cuda_graph_kv_indices  # CUDA图KV索引
            num_kv_splits = self.cuda_graph_num_kv_splits  # CUDA图KV分片数
            if spec_info is None:  # 无推测解码
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)  # 计算KV索引指针
                kv_indptr = kv_indptr[: bs + 1]  # 截取有效部分
                create_flashinfer_kv_indices_triton[(bs,)](  # Triton内核创建KV索引
                    self.req_to_token,  # 请求到token映射
                    req_pool_indices[:bs],  # 请求池索引
                    seq_lens[:bs],  # 序列长度
                    kv_indptr,  # KV索引指针
                    None,  # 无编码器长度
                    kv_indices,  # 输出KV索引
                    self.req_to_token.stride(0),  # 步长
                )
                num_token = bs  # token数等于批量大小
            else:  # 有推测解码
                kv_indptr[: spec_info.kv_indptr.shape[0]] = spec_info.kv_indptr  # 复制spec_info的指针
                kv_indices[: spec_info.kv_indices.shape[0]] = spec_info.kv_indices  # 复制spec_info的索引
                num_token = spec_info.kv_indptr.shape[0] - 1  # token数
            self.get_num_kv_splits(num_kv_splits[:num_token], seq_lens[:bs])  # 计算KV分片数
        elif forward_mode.is_target_verify():  # 目标验证模式
            bs = len(req_pool_indices)  # 批量大小
            qo_indptr = self.qo_indptr[: bs + 1]  # 查询输出指针
            qo_indptr[: bs + 1] = torch.arange(  # 构造等差序列
                0,  # 起始
                (1 + bs) * self.num_draft_tokens,  # 结束
                step=self.num_draft_tokens,  # 步长为草稿token数
                dtype=torch.int32,  # int32
                device=self.device,  # 设备
            )
            kv_indptr = self.kv_indptr[: bs + 1]  # KV索引指针
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)  # 计算指针
            kv_indices = self.cuda_graph_kv_indices  # CUDA图KV索引
            create_flashinfer_kv_indices_triton[(bs,)](  # Triton内核创建KV索引
                self.req_to_token,  # 请求到token映射
                req_pool_indices,  # 请求池索引
                seq_lens,  # 序列长度
                kv_indptr,  # KV索引指针
                None,  # 无编码器长度
                kv_indices,  # 输出KV索引
                self.req_to_token.stride(0),  # 步长
            )
            custom_mask = self.cuda_graph_custom_mask  # CUDA图自定义掩码
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask  # 复制掩码
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)  # 序列掩码长度
            mask_indptr = self.mask_indptr[: bs + 1]  # 掩码索引指针
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)  # 计算指针
        else:  # 无效模式
            raise ValueError(  # 抛出错误
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."  # CUDA图重放的前向模式无效
            )

    def get_cuda_graph_seq_len_fill_value(self):  # 获取CUDA图中序列长度的填充值
        return 1  # 返回1

    def forward_extend(  # 扩展注意力前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存
    ):
        # TODO: reuse the buffer across layers  # TODO: 在层间复用缓冲区
        if layer.qk_head_dim != layer.v_head_dim:  # 如果QK头维度不等于V头维度
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))  # 按V头维度分配输出
        else:  # 维度相同
            o = torch.empty_like(q)  # 分配与q相同形状的输出

        if save_kv_cache:  # 如果需要保存KV缓存
            self.token_to_kv_pool.set_kv_buffer(  # 设置KV缓存
                layer, forward_batch.out_cache_loc, k, v  # 层、缓存位置、K、V
            )

        max_extend_len = self.forward_metadata.max_extend_len  # 获取最大扩展长度
        computed_max_ext_seq_len = torch.max(forward_batch.extend_seq_lens)  # 计算最大扩展序列长度
        if computed_max_ext_seq_len != max_extend_len:  # 如果不匹配
            assert len(forward_batch.extend_seq_lens) == 1  # 只支持单序列
            forward_batch.extend_seq_lens[0] = max_extend_len  # 使用元数据中的值
            forward_batch.seq_lens = max_extend_len  # 更新序列长度

        self.extend_attention_fwd(  # 调用扩展注意力前向
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),  # 重塑Q
            k.contiguous(),  # 连续化K
            v.contiguous(),  # 连续化V
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),  # K缓存
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),  # V缓存
            self.forward_metadata.qo_indptr,  # 查询输出指针
            self.forward_metadata.kv_indptr,  # KV索引指针
            self.forward_metadata.kv_indices,  # KV索引
            self.forward_metadata.custom_mask,  # 自定义掩码
            self.forward_metadata.mask_indptr,  # 掩码索引指针
            self.forward_metadata.max_extend_len,  # 最大扩展长度
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),  # 重塑输出
            is_causal=True,  # 因果注意力
            layer_scaling=layer.scaling,  # 层缩放
            logit_cap=layer.logit_cap,  # logits上限
        )
        return o  # 返回输出

    def forward_decode(  # 解码注意力前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        # 在torch.compile期间，rotary_emb存在一个bug，导致输出值为3D张量形状。此处正确重塑输出。
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)  # 重塑Q为2D

        # TODO: reuse the buffer across layers  # TODO: 在层间复用缓冲区
        if layer.qk_head_dim != layer.v_head_dim:  # 如果QK头维度不等于V头维度
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))  # 按V头维度分配输出
        else:  # 维度相同
            o = torch.empty_like(q)  # 分配与q相同形状的输出

        if save_kv_cache:  # 如果需要保存KV缓存
            self.token_to_kv_pool.set_kv_buffer(  # 设置KV缓存
                layer, forward_batch.out_cache_loc, k, v  # 层、缓存位置、K、V
            )

        self.decode_attention_fwd(  # 调用解码注意力前向
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),  # 重塑Q
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),  # K缓存
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),  # V缓存
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),  # 重塑输出
            self.forward_metadata.kv_indptr,  # KV索引指针
            self.forward_metadata.kv_indices,  # KV索引
            self.forward_metadata.attn_logits,  # 注意力logits
            self.forward_metadata.attn_lse,  # 注意力LSE
            self.forward_metadata.num_kv_splits,  # KV分片数
            self.max_kv_splits,  # 最大KV分片数
            layer.scaling,  # 层缩放
            layer.logit_cap,  # logits上限
        )
        return o  # 返回输出
