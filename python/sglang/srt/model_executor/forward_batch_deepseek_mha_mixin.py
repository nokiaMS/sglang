# DeepSeek MHA前向批处理混入类，用于管理分块前缀缓存的元数据
# Mixin class for metadata management of Deepseek MHA forward (chunked prefix cache)  # 用于DeepSeek MHA前向（分块前缀缓存）元数据管理的混入类
# More details can be found in python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py  # 更多详情可在forward_mha.py中找到

from typing import List, Optional  # 导入类型提示

import torch  # 导入PyTorch框架
import triton  # 导入Triton框架
import triton.language as tl  # 导入Triton语言

from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton  # 导入FlashInfer KV索引创建函数
from sglang.srt.model_executor.forward_context import (  # 导入前向上下文工具
    get_req_to_token_pool,  # 获取请求到token映射池
    get_token_to_kv_pool,  # 获取token到KV映射池
)


class ForwardBatchDeepSeekMHAMixin:  # DeepSeek MHA前向批处理混入类，管理分块前缀缓存元数据
    # For MLA chunked prefix cache used in chunked prefill  # 用于分块预填充中的MLA分块前缀缓存
    # Tell attention backend whether the kv cache needs to be attended in current pass  # 告知注意力后端当前轮次是否需要处理KV缓存
    attn_attend_prefix_cache: Optional[bool] = None  # 是否需要在当前轮次处理前缀KV缓存
    # Number of prefix cache chunks  # 前缀缓存分块数量
    num_prefix_chunks: Optional[int] = None  # 前缀缓存分块数
    # Index of current chunk, used by attention backend  # 当前分块索引，由注意力后端使用
    prefix_chunk_idx: Optional[int] = None  # 当前前缀缓存分块索引
    # Maximum number of tokens in each chunk per sequence. Computed from maximum chunk capacity  # 每个序列每个分块的最大token数，根据最大分块容量计算
    prefix_chunk_len: Optional[int] = None  # 每个分块的最大token数
    # Start positions of prefix cache for each chunk, (num_prefix_chunks, batch_size)  # 每个分块的前缀缓存起始位置，形状为(num_prefix_chunks, batch_size)
    prefix_chunk_starts: Optional[torch.Tensor] = None  # 前缀缓存分块起始位置张量
    # Lengths of prefix cache for each chunk, (num_prefix_chunks, batch_size)  # 每个分块的前缀缓存长度，形状为(num_prefix_chunks, batch_size)
    prefix_chunk_seq_lens: Optional[torch.Tensor] = None  # 前缀缓存分块序列长度张量
    # Accumulated lengths of prefix cache for each chunk, (num_prefix_chunks, batch_size + 1)  # 每个分块的前缀缓存累积长度，形状为(num_prefix_chunks, batch_size + 1)
    prefix_chunk_cu_seq_lens: Optional[torch.Tensor] = None  # 前缀缓存分块累积序列长度张量
    # Max lengths of prefix cache for each chunk, (num_prefix_chunks,)  # 每个分块的前缀缓存最大长度，形状为(num_prefix_chunks,)
    prefix_chunk_max_seq_lens: Optional[List[int]] = None  # 前缀缓存分块最大序列长度列表
    # Per-chunk flag: True if any sequence has kv_len==0 in that chunk.  # 每分块标志：如果某个分块中有任何序列的kv_len==0则为True
    # Precomputed on CPU to avoid GPU-CPU sync in the hot path.  # 在CPU上预计算以避免热路径中的GPU-CPU同步
    prefix_chunk_has_zero_kv: Optional[List[bool]] = None  # 前缀缓存分块是否有零KV长度的标志列表
    # Number of tokens in each prefix cache chunk, (num_prefix_chunks,)  # 每个前缀缓存分块的token数量，形状为(num_prefix_chunks,)
    prefix_chunk_num_tokens: Optional[List[int]] = None  # 前缀缓存分块token数量列表
    # KV Indices for each chunk  # 每个分块的KV索引
    prefix_chunk_kv_indices: Optional[List[torch.Tensor]] = None  # 前缀缓存分块KV索引列表
    # For MLA chunked prefix cache used in chunked prefill  # 用于分块预填充中的MLA分块前缀缓存
    # Tell attention backend whether lse needs to be returned  # 告知注意力后端是否需要返回lse
    mha_return_lse: Optional[bool] = None  # 是否返回对数softmax值
    # Whether to apply MHA_ONE_SHOT forward method  # 是否应用MHA_ONE_SHOT前向方法
    mha_one_shot: Optional[bool] = None  # 是否使用一次性MHA前向
    # KV Indices for MHA_ONE_SHOT forward method  # MHA_ONE_SHOT前向方法的KV索引
    mha_one_shot_kv_indices: Optional[torch.Tensor] = None  # 一次性MHA前向的KV索引张量

    def get_max_chunk_capacity(self):  # 获取最大分块容量
        return envs.SGLANG_MAX_KV_CHUNK_CAPACITY.get()  # 从环境变量获取最大KV分块容量

    def set_prefix_chunk_idx(self, idx: int):  # 设置当前前缀缓存分块索引
        self.prefix_chunk_idx = idx  # 更新分块索引

    def set_attn_attend_prefix_cache(self, attn_attend_prefix_cache: bool):  # 设置是否在当前轮次处理前缀KV缓存
        self.attn_attend_prefix_cache = attn_attend_prefix_cache  # 更新标志

    def prepare_chunked_kv_indices(self, device: torch.device):  # 为每个分块准备KV索引
        self.prefix_chunk_kv_indices = []  # 初始化KV索引列表
        req_to_token = get_req_to_token_pool().req_to_token  # 获取请求到token映射表
        for idx in range(self.num_prefix_chunks):  # 遍历每个分块
            chunk_starts = self.prefix_chunk_starts[idx]  # 获取当前分块起始位置
            chunk_seq_lens = self.prefix_chunk_seq_lens[idx]  # 获取当前分块序列长度
            chunk_cu_seq_lens = self.prefix_chunk_cu_seq_lens[idx]  # 获取当前分块累积序列长度
            num_chunk_tokens = self.prefix_chunk_num_tokens[idx]  # 获取当前分块token数

            chunk_kv_indices = torch.empty(  # 创建当前分块的KV索引张量
                num_chunk_tokens, dtype=torch.int32, device=device
            )

            create_chunked_prefix_cache_kv_indices[(self.batch_size,)](  # 使用Triton内核创建分块前缀缓存KV索引
                req_to_token,  # 请求到token映射表
                self.req_pool_indices,  # 请求池索引
                chunk_starts,  # 分块起始位置
                chunk_seq_lens,  # 分块序列长度
                chunk_cu_seq_lens,  # 分块累积序列长度
                chunk_kv_indices,  # 输出：分块KV索引
                req_to_token.shape[1],  # 映射表第二维度大小
            )
            self.prefix_chunk_kv_indices.append(chunk_kv_indices)  # 将当前分块KV索引添加到列表

    # Here we suppose the length of each chunk is equal  # 这里假设每个分块的长度相等
    # For example, if we have 4 sequences with prefix length [256, 512, 768, 1024], prefix_chunk_len = 256  # 例如，如果有4个序列，前缀长度为[256, 512, 768, 1024]，则prefix_chunk_len = 256
    # num_prefix_chunks = cdiv(1024, 256) = 4  # 分块数量 = 向上取整(1024/256) = 4
    # prefix_chunk_starts = [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512], [768, 768, 768, 768]]  # 各分块起始位置
    # prefix_chunk_ends = [[256, 256, 256, 256], [256, 512, 512, 512], [256, 512, 768, 768], [256, 512, 768, 1024]]  # 各分块结束位置
    # prefix_chunk_seq_lens = [[256, 256, 256, 256], [0, 256, 256, 256], [0, 0, 256, 256], [0, 0, 0, 256]]  # 各分块序列长度
    # TODO: Implement a better way to allocate chunk lengths that uses memory spaces more efficiently.  # TODO：实现更好的分块长度分配方式，更高效地使用内存空间
    def get_prefix_chunk_seq_lens(  # 计算每个分块的起始位置和序列长度
        self, prefix_lens: torch.Tensor, num_prefix_chunks: int, prefix_chunk_len: int
    ):
        device = prefix_lens.device  # 获取设备
        prefix_chunk_starts = (  # 计算每个分块的起始位置
            torch.arange(num_prefix_chunks, device=device, dtype=torch.int32)
            .unsqueeze(1)
            .expand(-1, self.batch_size)
            * prefix_chunk_len
        )
        prefix_chunk_ends = torch.min(  # 计算每个分块的结束位置（取前缀长度和起始+分块长度的较小值）
            prefix_lens.unsqueeze(0),
            prefix_chunk_starts + prefix_chunk_len,
        ).to(torch.int32)

        prefix_chunk_seq_lens = (  # 计算每个分块的实际序列长度（结束-起始，最小为0）
            (prefix_chunk_ends - prefix_chunk_starts).clamp(min=0).to(torch.int32)
        )

        return prefix_chunk_starts, prefix_chunk_seq_lens  # 返回分块起始位置和序列长度

    # Called before each attention module if using chunked kv cache for prefill  # 如果使用分块KV缓存进行预填充，在每个注意力模块之前调用
    # Some of the codes are adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py  # 部分代码改编自vLLM项目
    def prepare_chunked_prefix_cache_info(self, device: torch.device):  # 准备分块前缀缓存信息

        from sglang.srt.mem_cache.memory_pool import (  # 导入内存池相关类
            HybridLinearKVPool,  # 混合线性KV池
            MLATokenToKVPool,  # MLA Token到KV池
        )

        token_to_kv_pool = get_token_to_kv_pool()  # 获取token到KV映射池
        assert isinstance(token_to_kv_pool, MLATokenToKVPool) or (  # 断言池类型为MLATokenToKVPool或HybridLinearKVPool
            isinstance(token_to_kv_pool, HybridLinearKVPool)
            and isinstance(token_to_kv_pool.full_kv_pool, MLATokenToKVPool)
        ), "Currently chunked prefix cache can only be used by Deepseek models"  # 当前分块前缀缓存仅支持DeepSeek模型

        if not any(self.extend_prefix_lens_cpu):  # 如果没有前缀长度
            self.num_prefix_chunks = 0  # 设置分块数为0
            return  # 直接返回

        if self.prefix_chunk_len is not None:  # 如果分块长度已设置
            # Chunked kv cache info already prepared by prior modules  # 分块KV缓存信息已由先前模块准备
            return  # 直接返回

        self.prefix_chunk_idx = -1  # 初始化分块索引为-1

        # chunk_capacity is the maximum number of tokens in each chunk  # chunk_capacity是每个分块的最大token数
        chunk_capacity = self.get_max_chunk_capacity()  # 获取最大分块容量
        self.prefix_chunk_len = chunk_capacity // self.batch_size  # 每个分块的长度 = 容量 / 批大小

        self.num_prefix_chunks = (  # 计算分块数量
            max(self.extend_prefix_lens_cpu) + self.prefix_chunk_len - 1
        ) // self.prefix_chunk_len  # 向上取整

        # Here we compute chunk lens twice to avoid stream sync, once on gpu and once on cpu.  # 这里计算两次分块长度以避免流同步，一次在GPU上，一次在CPU上
        prefix_chunk_starts_cuda, prefix_chunk_seq_lens_cuda = (  # GPU上计算分块信息
            self.get_prefix_chunk_seq_lens(
                self.extend_prefix_lens,
                self.num_prefix_chunks,
                self.prefix_chunk_len,
            )
        )
        _, prefix_chunk_seq_lens_cpu = self.get_prefix_chunk_seq_lens(  # CPU上计算分块信息
            torch.tensor(self.extend_prefix_lens_cpu),
            self.num_prefix_chunks,
            self.prefix_chunk_len,
        )
        self.prefix_chunk_starts = prefix_chunk_starts_cuda  # 保存GPU上的分块起始位置
        self.prefix_chunk_seq_lens = prefix_chunk_seq_lens_cuda  # 保存GPU上的分块序列长度

        # Metadata for attention backend  # 注意力后端的元数据
        self.prefix_chunk_cu_seq_lens = torch.zeros(  # 创建累积序列长度张量
            self.num_prefix_chunks,
            self.batch_size + 1,
            device=device,
            dtype=torch.int32,
        )
        self.prefix_chunk_cu_seq_lens[:, 1:] = prefix_chunk_seq_lens_cuda.cumsum(  # 计算累积和
            dim=1
        ).to(torch.int32)
        self.prefix_chunk_max_seq_lens = prefix_chunk_seq_lens_cpu.max(  # 计算每个分块的最大序列长度
            dim=1
        ).values.tolist()

        self.prefix_chunk_num_tokens = prefix_chunk_seq_lens_cpu.sum(dim=1).tolist()  # 计算每个分块的总token数
        assert max(self.prefix_chunk_num_tokens) <= self.get_max_chunk_capacity()  # 断言不超过最大分块容量

        # Per-chunk flag: does any sequence have kv_len == 0?  # 每分块标志：是否有任何序列的kv_len==0？
        # Pure CPU check (prefix_chunk_seq_lens_cpu is on CPU), no GPU sync.  # 纯CPU检查（prefix_chunk_seq_lens_cpu在CPU上），无需GPU同步
        self.prefix_chunk_has_zero_kv = [  # 计算每个分块是否有零KV长度的序列
            bool((prefix_chunk_seq_lens_cpu[i] == 0).any())
            for i in range(self.num_prefix_chunks)
        ]

        # Precompute the kv indices for each chunk  # 预计算每个分块的KV索引
        self.prepare_chunked_kv_indices(device)

    def fetch_mha_one_shot_kv_indices(self):  # 获取MHA一次性前向方法的KV索引
        if self.mha_one_shot_kv_indices is not None:  # 如果已存在缓存
            return self.mha_one_shot_kv_indices  # 直接返回缓存
        batch_size = self.batch_size  # 批大小
        paged_kernel_lens_sum = sum(self.seq_lens_cpu)  # 所有序列长度之和
        kv_indices = torch.empty(  # 创建KV索引张量
            paged_kernel_lens_sum,
            dtype=torch.int32,
            device=self.req_pool_indices.device,
        )
        kv_indptr = torch.zeros(  # 创建KV间接指针张量
            batch_size + 1,
            dtype=torch.int32,
            device=self.req_pool_indices.device,
        )
        kv_indptr[1:] = torch.cumsum(self.seq_lens, dim=0)  # 计算累积和
        req_to_token = get_req_to_token_pool().req_to_token  # 获取请求到token映射表
        create_flashinfer_kv_indices_triton[(self.batch_size,)](  # 使用Triton内核创建FlashInfer KV索引
            req_to_token,  # 请求到token映射表
            self.req_pool_indices,  # 请求池索引
            self.seq_lens,  # 序列长度
            kv_indptr,  # KV间接指针
            None,  # 无额外参数
            kv_indices,  # 输出：KV索引
            req_to_token.shape[1],  # 映射表第二维度大小
        )
        self.mha_one_shot_kv_indices = kv_indices  # 缓存KV索引
        return kv_indices  # 返回KV索引


@triton.jit
def create_chunked_prefix_cache_kv_indices(  # Triton内核：创建分块前缀缓存的KV索引
    req_to_token_ptr,  # (max_batch, max_context_len,)  # 请求到token映射指针，形状为(最大批次数, 最大上下文长度)
    req_pool_indices_ptr,  # (batch_size,)  # 请求池索引指针，形状为(批大小,)
    chunk_start_idx_ptr,  # (batch_size,)  # 分块起始索引指针，形状为(批大小,)
    chunk_seq_lens_ptr,  # (batch_size,)  # 分块序列长度指针，形状为(批大小,)
    chunk_cu_seq_lens_ptr,  # (batch_size + 1,)  # 分块累积序列长度指针，形状为(批大小+1,)
    chunk_kv_indices_ptr,  # (num_chunk_tokens,)  # 分块KV索引输出指针，形状为(分块token数,)
    req_to_token_ptr_stride: tl.constexpr,  # 请求到token映射的步长（编译时常量）
):
    BLOCK_SIZE: tl.constexpr = 512  # 分块大小为512（编译时常量）
    pid = tl.program_id(axis=0)  # 获取当前程序的ID（对应批次中的序列索引）

    # find the req pool idx, this is for batch to token  # 查找请求池索引，用于批次到token的映射
    req_pool_index = tl.load(req_pool_indices_ptr + pid)  # 加载当前序列的请求池索引
    chunk_kv_indices_offset = tl.load(chunk_cu_seq_lens_ptr + pid)  # 加载当前序列在KV索引中的偏移量

    # get the token positions of current chunk  # 获取当前分块的token位置
    chunk_start_pos = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)  # 加载当前分块的起始位置
    chunk_seq_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)  # 加载当前分块的序列长度

    num_loop = tl.cdiv(chunk_seq_len, BLOCK_SIZE)  # 计算需要循环的次数
    for i in range(num_loop):  # 遍历每个分块
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE  # 计算当前分块的偏移量
        mask = offset < chunk_seq_len  # 创建掩码，只处理有效位置
        data = tl.load(  # 从映射表中加载数据
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + chunk_start_pos
            + offset,
            mask=mask,  # 应用掩码
        )
        tl.store(  # 将数据存储到KV索引输出中
            chunk_kv_indices_ptr + chunk_kv_indices_offset + offset, data, mask=mask
        )
