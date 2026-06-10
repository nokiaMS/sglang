# LoRA后端基类实现
# 定义了所有LoRA后端必须实现的接口方法，包括LoRA A/B矩阵运算、
# QKV/Gate-Up投影的LoRA计算、CUDA Graph批量信息初始化和MoE LoRA支持
from typing import Tuple, Union  # 导入类型提示工具

import torch  # 导入PyTorch库
import triton  # 导入Triton编译器
import triton.language as tl  # 导入Triton语言

from sglang.srt.lora.backend.lmhead_mixing import LoRABackendLmHeadMixing  # 导入LM Head混合类
from sglang.srt.lora.utils import LoRABatchInfo, MoELoRABatchInfo  # 导入LoRA批量信息数据类
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批量信息类


class BaseLoRABackend(LoRABackendLmHeadMixing):  # LoRA后端基类，继承LM Head混合类
    """Base class for different Lora backends.
       Each backend has its own implementation of Lora kernels.
    不同LoRA后端的基类。
       每个后端有自己的LoRA算子实现。

    Args:
        max_loras_per_batch: maximum number of different lora weights
                             that can be applied in a single forward batch.
        max_loras_per_batch: 单次前向批次中可应用的最大不同LoRA权重数量。
        device: the device where the backend runs.
        device: 后端运行的设备。
    """

    def __init__(self, max_loras_per_batch: int, device: torch.device):  # 初始化方法
        self.max_loras_per_batch = max_loras_per_batch  # 保存每批次最大LoRA数量
        self.device = device  # 保存运行设备
        self.init_lm_head_config()  # 初始化LM Head配置
        self._is_moe_lora = False  # 初始化MoE LoRA标志为False

    def run_lora_a_embedding(  # 运行LoRA A嵌入查找（支持CUDA Graph）
        self,
        input_ids: torch.Tensor,  # token ID张量，形状为(s,)
        weights: torch.Tensor,  # LoRA A嵌入权重，形状为(num_loras, rank, vocab_size)
        vocab_size: int,  # 基础词表大小（>=vocab_size的为额外token）
        extra_embeddings: torch.Tensor = None,  # 额外token嵌入，形状为(num_loras, num_extra_tokens, rank)
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run LoRA A embedding lookup with CUDA graph support.
        运行支持CUDA Graph的LoRA A嵌入查找。

        Args:
            input_ids: token IDs with shape (s,), where s is the sum of all sequence lengths
            input_ids: token ID，形状为(s,)，s为所有序列长度之和
            weights: LoRA A embedding weights with shape (num_loras, rank, vocab_size)
            weights: LoRA A嵌入权重，形状为(num_loras, rank, vocab_size)
            vocab_size: base vocabulary size (tokens >= vocab_size are extra tokens)
            vocab_size: 基础词表大小（>=vocab_size的token为额外token）
            extra_embeddings: extra token embeddings with shape (num_loras, num_extra_tokens, rank)
            extra_embeddings: 额外token嵌入，形状为(num_loras, num_extra_tokens, rank)
            Only needed if there are added tokens beyond base vocabulary.
            仅当基础词表之外有新增token时需要。

        Returns:
            result with shape (s, rank)
            结果，形状为(s, rank)
        """
        pass  # 由子类实现

    def run_extra_token_embedding(  # 运行额外token嵌入查找
        self,
        input_ids: torch.Tensor,  # token ID张量，形状为(s,)
        output: torch.Tensor,  # 输出张量，形状为(s, embed_dim)，将被就地修改
        extra_embeddings: torch.Tensor,  # 额外嵌入，形状为(num_loras, num_extra_tokens, embed_dim)
        vocab_size: int,  # 基础词表大小
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply extra token embeddings to output in-place.
        就地将额外token嵌入应用到输出。

        Args:
            input_ids: (s,) token IDs
            input_ids: (s,) token ID
            output: (s, embed_dim) output tensor to be modified
            output: (s, embed_dim) 待修改的输出张量
            extra_embeddings: (num_loras, num_extra_tokens, embed_dim) extra embeddings
            extra_embeddings: (num_loras, num_extra_tokens, embed_dim) 额外嵌入
            vocab_size: base vocabulary size
            vocab_size: 基础词表大小

        Returns:
            output: modified output tensor
            output: 修改后的输出张量
        """
        raise NotImplementedError  # 由子类实现

    def run_lora_a_sgemm(  # 运行LoRA A模块的分段GEMM（shrink阶段）
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs  # 输入矩阵和权重
    ) -> torch.Tensor:
        """Run segment Gemm of lora a modules with current backend.
        The definition of segment Gemm can be referred to https://docs.flashinfer.ai/api/gemm.html.
        使用当前后端运行LoRA A模块的分段GEMM。
        分段GEMM的定义可参考 https://docs.flashinfer.ai/api/gemm.html。

        Args:
             x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
             x: 输入矩阵，形状为(s, input_dim)，s为所有序列长度之和
             weights: a set of lora weights with shape (num_lora, c * r, input_dim),
                      here r is lora rank, c is a multiplier for stacked modules (e.g., c=3 for qkv_proj, c=2 for gate_up_proj)
                      usually input_dim is much larger than r
             weights: LoRA权重集合，形状为(num_lora, c * r, input_dim)，
                      r为LoRA秩，c为堆叠模块乘数（如qkv_proj的c=3，gate_up_proj的c=2），
                      通常input_dim远大于r
        Returns:
             result with shape (s, c * r)
             结果，形状为(s, c * r)
        """
        pass  # 由子类实现

    def run_lora_b_sgemm(  # 运行LoRA B模块的分段GEMM（expand阶段）
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs  # 输入矩阵和权重
    ) -> torch.Tensor:
        """Run segment Gemm of lora b modules with current backend.
        The definition of segment Gemm can be referred to https://docs.flashinfer.ai/api/gemm.html.
        使用当前后端运行LoRA B模块的分段GEMM。
        分段GEMM的定义可参考 https://docs.flashinfer.ai/api/gemm.html。

        Args:
             x: input matrix with shape (s, r), here s is the sum of all sequence lengths, r is lora rank
             x: 输入矩阵，形状为(s, r)，s为所有序列长度之和，r为LoRA秩
             weights: a set of lora weights with shape (num_lora, output_dim, r)
                      usually output_dim is much larger than r
             weights: LoRA权重集合，形状为(num_lora, output_dim, r)，
                      通常output_dim远大于r
        Returns:
             result with shape (s, output_dim)
             结果，形状为(s, output_dim)
        """
        pass  # 由子类实现

    def run_qkv_lora(  # 运行QKV层的LoRA前向传播
        self,
        x: torch.Tensor,  # 输入矩阵，形状为(s, input_dim)
        qkv_lora_a: Union[torch.Tensor, Tuple[torch.Tensor]],  # QKV的LoRA A权重
        qkv_lora_b: Union[torch.Tensor, Tuple[torch.Tensor]],  # QKV的LoRA B权重
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run the lora pass for QKV Layer.
        运行QKV层的LoRA前向传播。

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            x: 输入矩阵，形状为(s, input_dim)，s为所有序列长度之和
            qkv_lora_a: lora_a module for qkv, with shape (num_lora, 3 * r, input_dim)
            qkv_lora_a: QKV的lora_a模块，形状为(num_lora, 3 * r, input_dim)
            qkv_lora_b: lora_b module for qkv.
            qkv_lora_b: QKV的lora_b模块。
                        If passed in as a tensor, its shape should be (num_lora,output_dim_q + 2 * output_dim_kv, r)
                        如果作为张量传入，形状应为(num_lora, output_dim_q + 2 * output_dim_kv, r)
                        If passed in as a tuple of two tensors, it should contain:
                        如果作为两个张量的元组传入，应包含：
                           a lora_b module for q, with shape (1, num_lora, output_dim_q, r)
                           Q的lora_b模块，形状为(1, num_lora, output_dim_q, r)
                           and a combined lora_b module for kv, with shape (2, num_lora, output_dim_kv, r)
                           和KV的合并lora_b模块，形状为(2, num_lora, output_dim_kv, r)
        Returns:
            result with shape (s, output_dim_q + 2 * output_dim_kv)
            结果，形状为(s, output_dim_q + 2 * output_dim_kv)
        """
        pass  # 由子类实现

    def run_gate_up_lora(  # 运行gate_up_proj的LoRA前向传播，通常附加到MergedColumnParallelLayer
        self,
        x: torch.Tensor,  # 输入矩阵，形状为(s, input_dim)
        gate_up_lora_a: Union[torch.Tensor, Tuple[torch.Tensor]],  # gate_up的LoRA A权重
        gate_up_lora_b: Union[torch.Tensor, Tuple[torch.Tensor]],  # gate_up的LoRA B权重
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run the lora pass for gate_up_proj, usually attached to MergedColumnParallelLayer.
        运行gate_up_proj的LoRA前向传播，通常附加到MergedColumnParallelLayer。

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            x: 输入矩阵，形状为(s, input_dim)，s为所有序列长度之和
            gate_up_lora_a: lora_a module for gate_up_proj, with shape (num_lora, 2 * r, input_dim)
            gate_up_lora_a: gate_up_proj的lora_a模块，形状为(num_lora, 2 * r, input_dim)
            gate_up_lora_b: lora_b module for qkv.
            gate_up_lora_b: gate_up的lora_b模块。
                        If passed in as a tensor, its shape should be (num_lora, 2 * output_dim, r)
                        如果作为张量传入，形状应为(num_lora, 2 * output_dim, r)
                        If passed in as a tuple, it should contain two tensors with shape (num_lora, output_dim, r)
                        如果作为元组传入，应包含两个形状为(num_lora, output_dim, r)的张量
        Returns:
            result with shape (s, 2 * output_dim)
            结果，形状为(s, 2 * output_dim)
        """
        pass  # 由子类实现

    def init_cuda_graph_batch_info(  # 初始化CUDA Graph模式的LoRA批量元数据
        self,
        max_bs_in_cuda_graph: int,  # CUDA Graph模式的最大批量大小
        num_tokens_per_bs: int,  # 每个序列的token数量（解码为1，target_verify>1）
    ):
        """Phase 2 of LoRA CUDA graph init: dense LoRA batch metadata.
        LoRA CUDA图初始化第2阶段：密集LoRA批量元数据。

        Called during CudaGraphRunner.__init__(), after init_memory_pool().
        在CudaGraphRunner.__init__()中，init_memory_pool()之后调用。

        Args:
            max_bs_in_cuda_graph: maximum batch size for CUDA Graph mode
            max_bs_in_cuda_graph: CUDA Graph模式的最大批量大小
            num_tokens_per_bs: number of tokens per sequence (1 for decoding, >1 for target_verify)
            num_tokens_per_bs: 每个序列的token数量（解码为1，target_verify大于1）
        """
        pass  # 由子类实现

    @property
    def is_moe_lora(self) -> bool:  # 是否启用MoE LoRA的属性
        return self._is_moe_lora  # 返回MoE LoRA标志

    @is_moe_lora.setter
    def is_moe_lora(self, value: bool):  # 设置MoE LoRA标志
        self._is_moe_lora = value  # 更新MoE LoRA标志

    def init_cuda_graph_moe_buffers(  # 初始化CUDA Graph模式下MoE LoRA的中间缓冲区
        self,
        max_bs: int,  # 最大批量大小
        max_loras: int,  # 最大LoRA数量
        compute_dtype: torch.dtype,  # 计算数据类型
        moe_layer,  # MoE层对象
    ):
        """Phase 1 of LoRA CUDA graph init: MoE intermediate buffers.
        LoRA CUDA图初始化第1阶段：MoE中间缓冲区。

        Called once before init_memory_pool() with a representative MoE layer
        to extract dimensions.  All FusedMoEWithLoRA layers share the same
        buffers since they execute sequentially during forward.
        在init_memory_pool()之前用代表性MoE层调用一次以提取维度。
        所有FusedMoEWithLoRA层共享相同的缓冲区，因为它们在前向传播中顺序执行。

        This is backend-agnostic because MoE LoRA always uses the same
        fused Triton kernel (TritonRunnerCoreWithLoRA) regardless of which
        dense LoRA backend is selected.
        这是后端无关的，因为MoE LoRA始终使用相同的融合Triton算子
        (TritonRunnerCoreWithLoRA)，无论选择哪个密集LoRA后端。
        """
        base = moe_layer.base_layer  # 获取MoE基础层
        top_k = base.top_k  # 获取top-k值
        qinfo = moe_layer._quant_info  # 获取量化信息
        E, N, _ = qinfo.w13_weight.shape  # 获取专家数、中间维度
        hidden_dim = qinfo.w2_weight.shape[1]  # 获取隐藏维度
        device = qinfo.w13_weight.device  # 获取设备
        dtype = compute_dtype  # 计算数据类型
        num_experts = base.num_experts  # 获取专家数量

        block_size_m = 64  # 块大小
        max_num_tokens_padded = max_bs * top_k + num_experts * (block_size_m - 1)  # 计算填充后的最大token数
        max_num_tokens_padded = (  # 向上对齐到块大小
            (max_num_tokens_padded + block_size_m - 1) // block_size_m
        ) * block_size_m
        max_num_m_blocks = (max_num_tokens_padded + block_size_m - 1) // block_size_m  # 计算最大块数

        self.moe_cg_buffers = {  # 创建MoE CUDA Graph缓冲区字典
            "intermediate_cache1": torch.empty(  # 中间缓存1，形状为(max_bs, top_k, N)
                (max_bs, top_k, N), device=device, dtype=dtype
            ),
            "intermediate_cache2": torch.empty(  # 中间缓存2，形状为(max_bs * top_k, N // 2)
                (max_bs * top_k, N // 2), device=device, dtype=dtype
            ),
            "intermediate_cache3": torch.empty(  # 中间缓存3，形状为(max_bs, top_k, hidden_dim)
                (max_bs, top_k, hidden_dim), device=device, dtype=dtype
            ),
            "out_hidden_states": torch.empty(  # 输出隐藏状态，形状为(max_bs, hidden_dim)
                (max_bs, hidden_dim), device=device, dtype=dtype
            ),
            "sorted_token_ids_lora": torch.empty(  # 排序后的token ID（LoRA用）
                (max_loras * max_num_tokens_padded,),
                device=device,
                dtype=torch.int32,
            ),
            "expert_ids_lora": torch.empty(  # 专家ID（LoRA用）
                (max_loras * max_num_m_blocks,),
                device=device,
                dtype=torch.int32,
            ),
            "num_tokens_post_padded_lora": torch.empty(  # 填充后的token数量（LoRA用）
                (max_loras,), device=device, dtype=torch.int32
            ),
            "adapter_enabled": torch.zeros(max_loras, dtype=torch.int32, device=device),  # 适配器启用标志
            # int64 copy of weight_indices for index_fill_(), which requires
            # LongTensor.  weight_indices itself must stay int32 because the
            # CUDA moe_lora_align kernel casts it to int32_t*.
            # weight_indices的int64副本，用于index_fill_()，该函数需要LongTensor。
            # weight_indices本身必须保持int32，因为CUDA的moe_lora_align算子将其转换为int32_t*。
            "weight_indices_long": torch.zeros(  # int64版本的权重索引
                max_bs, dtype=torch.int64, device=device
            ),
            "lora_ids": torch.arange(max_loras, dtype=torch.int32, device=device),  # LoRA ID数组
            "cumsum_buffer": torch.zeros(  # 累积和缓冲区
                max_loras * (num_experts + 1),
                dtype=torch.int32,
                device=device,
            ),
            "token_mask": torch.empty(  # token掩码
                (max_loras * max_bs * top_k,),
                dtype=torch.int32,
                device=device,
            ),
            "max_num_tokens_padded": max_num_tokens_padded,  # 填充后的最大token数
            "max_num_m_blocks": max_num_m_blocks,  # 最大块数
            "token_lora_mapping": torch.full(  # token到LoRA的映射，初始为-1
                (max_bs,), -1, dtype=torch.int32, device=device
            ),
        }

    def _add_moe_lora_info(  # 向批量信息中添加MoE LoRA信息
        self, forward_batch: ForwardBatch, batch_info: LoRABatchInfo  # 前向批次和批量信息
    ) -> LoRABatchInfo:
        if not self.is_moe_lora:  # 如果未启用MoE LoRA
            return batch_info  # 直接返回原始批量信息

        if batch_info.use_cuda_graph:  # 如果使用CUDA Graph
            adapter_enabled = self.moe_cg_buffers["adapter_enabled"]  # 获取预分配的适配器启用张量
            token_lora_mapping = self.moe_cg_buffers["token_lora_mapping"]  # 获取预分配的token映射张量
        else:  # 非图模式
            adapter_enabled = None  # 不使用预分配张量
            token_lora_mapping = None  # 不使用预分配张量

        num_tokens = (  # 计算token总数
            sum(forward_batch.extend_seq_lens_cpu)  # 扩展模式取扩展长度之和
            if forward_batch.forward_mode.is_extend()  # 如果是扩展模式
            else forward_batch.batch_size  # 解码模式取批量大小
        )
        max_len = (  # 计算最大序列长度
            max(forward_batch.extend_seq_lens_cpu)  # 扩展模式取最大扩展长度
            if forward_batch.forward_mode.is_extend()  # 如果是扩展模式
            else 1  # 解码模式长度为1
        )

        if (  # 如果有请求级别的分段信息
            batch_info.req_seg_indptr is not None
            or batch_info.req_weight_indices is not None
        ):
            assert batch_info.req_seg_indptr is not None  # 确保请求分段指针不为空
            assert batch_info.req_weight_indices is not None  # 确保请求权重索引不为空
            num_moe_segments = batch_info.bs  # MoE段数等于批量大小
            seg_indptr = batch_info.req_seg_indptr[: num_moe_segments + 1]  # 使用请求级分段指针
            req_to_lora = batch_info.req_weight_indices[:num_moe_segments]  # 使用请求级权重索引
        else:  # 没有请求级别的分段信息
            num_moe_segments = batch_info.num_segments  # MoE段数等于批量段数
            seg_indptr = batch_info.seg_indptr[: num_moe_segments + 1]  # 使用分段指针
            req_to_lora = batch_info.weight_indices[:num_moe_segments]  # 使用权重索引

        adapter_enabled, token_lora_mapping = _compute_moe_lora_info(  # 计算MoE LoRA信息
            num_tokens,  # token总数
            seg_indptr,  # 分段索引指针
            batch_info.lora_ranks,  # LoRA秩
            req_to_lora,  # 请求到LoRA的映射
            adapter_enabled,  # 适配器启用张量
            token_lora_mapping,  # token映射张量
            max_len=max_len,  # 最大序列长度
        )

        batch_info.moe_lora_info = MoELoRABatchInfo(  # 创建MoE LoRA批量信息
            seg_indptr=seg_indptr,  # 分段索引指针
            req_to_lora=req_to_lora,  # 请求到LoRA的映射
            adapter_enabled=adapter_enabled,  # 适配器启用标志
            token_lora_mapping=token_lora_mapping,  # token到LoRA的映射
        )

        return batch_info  # 返回更新后的批量信息

    def prepare_lora_batch(  # 准备当前前向批次的LoRA权重和批量信息
        self,
        forward_batch: ForwardBatch,  # 当前前向批次
        weight_indices: list[int],  # LoRA权重索引列表
        lora_ranks: list[int],  # LoRA秩列表
        scalings: list[float],  # 缩放因子列表
        use_cuda_graph: bool,  # 是否使用CUDA Graph
    ):
        """Prepare the lora weights and batch info for current forward batch.
        为当前前向批次准备LoRA权重和批量信息。

        This method provides a hook for each backend to conduct its own preparation
        logic for each forward batch.
        此方法为每个后端提供了执行自己前向批次准备逻辑的钩子。

        Args:
            forward_batch: the ForwardBatch object for current forward pass
            forward_batch: 当前前向传播的ForwardBatch对象
            weight_indices: list of indices of lora weights to be applied for current batch
            weight_indices: 当前批次要应用的LoRA权重索引列表
            lora_ranks: list of lora ranks corresponding to weight_indices
            lora_ranks: 对应weight_indices的LoRA秩列表
            scalings: list of scaling factors corresponding to weight_indices
            scalings: 对应weight_indices的缩放因子列表
            use_cuda_graph: whether to use CUDA Graph for this batch
            use_cuda_graph: 此批次是否使用CUDA Graph
        """
        pass  # 由子类实现


@triton.jit  # Triton JIT编译装饰器
def _compute_moe_lora_info_kernel(  # 计算MoE LoRA信息的Triton核函数
    seg_indptr_ptr,  # 分段索引指针
    lora_ranks_ptr,  # LoRA秩指针
    weight_indices_ptr,  # 权重索引指针
    adapter_enabled_ptr,  # 适配器启用标志指针
    token_lora_mapping_ptr,  # token到LoRA映射指针
    num_segments,  # 分段数量
    max_len,  # 最大序列长度
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
):
    pid = tl.program_id(0)  # 获取当前程序ID
    num_pid_m = tl.cdiv(max_len, BLOCK_SIZE)  # 计算每段的块数

    pid_seg = pid // num_pid_m  # 计算当前分段ID
    pid_m = pid % num_pid_m  # 计算当前段内块ID
    seg_start = tl.load(seg_indptr_ptr + pid_seg)  # 加载当前段起始位置
    seg_end = tl.load(seg_indptr_ptr + pid_seg + 1)  # 加载当前段结束位置
    seg_len = seg_end - seg_start  # 计算段长度

    offs = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算块内偏移
    valid = offs < seg_len  # 计算有效掩码
    lora_id = tl.load(weight_indices_ptr + pid_seg)  # 加载当前段的LoRA ID
    lora_rank = tl.load(lora_ranks_ptr + lora_id)  # 加载对应LoRA的秩
    tl.store(  # 存储适配器启用标志
        adapter_enabled_ptr + lora_id,  # 目标地址
        (lora_rank > 0).to(tl.int32),  # 如果秩大于0则启用
        mask=pid_m == 0,  # 仅在第一个块时写入（避免重复写入）
    )
    tl.store(token_lora_mapping_ptr + seg_start + offs, lora_id, mask=valid)  # 存储token到LoRA的映射


def _compute_moe_lora_info(  # 计算MoE LoRA信息的主机函数
    num_tokens: int,  # token总数
    seg_indptr: torch.Tensor,  # 分段索引指针
    lora_ranks: torch.Tensor,  # LoRA秩
    weight_indices: torch.Tensor,  # 权重索引
    adapter_enabled: torch.Tensor | None,  # 适配器启用标志（可选）
    token_lora_mapping: torch.Tensor | None,  # token到LoRA的映射（可选）
    max_len: int,  # 最大序列长度
) -> tuple[torch.Tensor, torch.Tensor]:  # 返回(适配器启用标志, token映射)
    if token_lora_mapping is not None:  # 如果提供了预分配的映射张量
        assert (
            num_tokens <= token_lora_mapping.shape[0]
        ), "num_tokens must be less than or equal to the shape of token_lora_mapping"  # 确保token数不超过映射大小
        token_lora_mapping = token_lora_mapping[:num_tokens]  # 截取到实际token数
    else:  # 未提供预分配张量
        token_lora_mapping = torch.empty(  # 创建新的映射张量
            (num_tokens,), dtype=torch.int32, device=seg_indptr.device  # 形状为(num_tokens,)
        )

    if adapter_enabled is not None:  # 如果提供了预分配的适配器张量
        assert (
            len(lora_ranks) <= adapter_enabled.shape[0]
        ), "lora_ranks must be less than or equal to the shape of adapter_enabled"  # 确保LoRA秩数不超过适配器大小
    else:  # 未提供预分配张量
        adapter_enabled = torch.empty(  # 创建新的适配器张量
            len(lora_ranks), dtype=torch.int32, device=lora_ranks.device  # 形状为(LoRA数,)
        )

    adapter_enabled.zero_()  # 将适配器启用标志清零

    has_segments = weight_indices.numel() != 0  # 检查是否有分段
    use_cuda_kernel = (  # 判断是否使用CUDA核函数
        num_tokens != 0 and has_segments and seg_indptr.device.type == "cuda"  # 非空、有分段且在CUDA上
    )
    if use_cuda_kernel:  # 使用CUDA核函数
        block_size = 256  # 块大小
        tiles_per_segment = triton.cdiv(max_len, block_size)  # 每段的块数
        grid_size = tiles_per_segment * weight_indices.numel()  # 总网格大小
        assert grid_size * block_size >= num_tokens, (  # 确保覆盖所有token
            f"MoE LoRA token-mapping launch under-covers tokens: "  # MoE LoRA token映射启动覆盖不足
            f"{grid_size=} {block_size=} {num_tokens=}"
        )
        _compute_moe_lora_info_kernel[(grid_size,)](  # 启动Triton核函数
            seg_indptr,  # 分段索引指针
            lora_ranks,  # LoRA秩
            weight_indices,  # 权重索引
            adapter_enabled,  # 适配器启用标志
            token_lora_mapping,  # token映射
            weight_indices.numel(),  # 分段数量
            max_len,  # 最大序列长度
            BLOCK_SIZE=block_size,  # 块大小
        )
        return adapter_enabled, token_lora_mapping  # 返回结果

    if has_segments:  # 有分段但不使用CUDA核函数（CPU回退路径）
        active_ranks = lora_ranks[weight_indices.long()]  # 获取活跃LoRA的秩
        adapter_enabled.scatter_(  # 散射写入适配器启用标志
            0, weight_indices.long(), (active_ranks > 0).to(torch.int32)  # 秩大于0的适配器启用
        )
    if num_tokens == 0:  # 如果没有token
        return adapter_enabled, token_lora_mapping  # 直接返回
    if not has_segments:  # 如果没有分段
        token_lora_mapping.fill_(-1)  # 所有token映射为-1（无LoRA）
        return adapter_enabled, token_lora_mapping  # 返回结果

    token_positions = torch.arange(  # 创建token位置张量
        num_tokens, device=seg_indptr.device, dtype=torch.int32  # 形状为(num_tokens,)
    )
    # There is a torch.compile bug so we can't use seg_indptr[1:] here.
    # Instead we pass seg_indptr and then subtract 1 from the result.
    # This works because seg_indptr[0] == 0.
    # 存在torch.compile的bug，所以不能使用seg_indptr[1:]。
    # 改为传入seg_indptr然后从结果中减1。
    # 这之所以有效是因为seg_indptr[0] == 0。
    req_indices = (  # 计算每个token所属的请求索引
        torch.searchsorted(seg_indptr.to(torch.int32), token_positions, right=True) - 1  # 二分搜索后减1
    )

    token_lora_mapping = torch.index_select(  # 根据请求索引从权重索引中选取
        weight_indices.to(torch.int32), 0, req_indices, out=token_lora_mapping  # 输出到token映射
    )

    return adapter_enabled, token_lora_mapping  # 返回适配器启用标志和token映射
