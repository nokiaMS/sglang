# 分段式SGMV LoRA后端实现
# 基于Punica论文的SGMV算法，将输入序列分割为固定大小的块(chunk)，
# 以减少LoRA分布不均时的过多核函数启动，提升GPU利用率
import dataclasses  # 导入数据类工具
from typing import List, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch库

from sglang.srt.lora.backend.base_backend import BaseLoRABackend  # 导入LoRA后端基类
from sglang.srt.lora.triton_ops import (  # 导入分段式LoRA Triton算子
    chunked_embedding_lora_a_forward,  # 分段式嵌入LoRA A前向
    chunked_sgmv_lora_expand_forward,  # 分段式SGMV LoRA expand前向
    chunked_sgmv_lora_shrink_forward,  # 分段式SGMV LoRA shrink前向
)
from sglang.srt.lora.utils import (  # 导入LoRA工具函数
    LoRABatchInfo,  # LoRA批量信息数据类
    generate_sequence_lengths,  # 生成序列长度
    get_lm_head_pruned_lens,  # 获取LM Head裁剪长度
    merge_and_chunk_segments,  # 合并和分块分段
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批量信息类
from sglang.srt.server_args import ServerArgs  # 导入服务器参数类

MIN_CHUNK_SIZE = 16  # 最小块大小常量


class ChunkedSgmvLoRABackend(BaseLoRABackend):  # 分段式SGMV LoRA后端类，继承自基类
    """
    Chunked LoRA backend using segmented matrix-vector multiplication.
    分段式LoRA后端，使用分段矩阵向量乘法。

    This backend is largely based on the SGMV (Segmented Gather Matrix-Vector multiplication) algorithm
    introduced in the Punica paper (https://arxiv.org/pdf/2310.18547). One main variation made here is to
    segment the input sequences into fixed-size chunks, which reduces excessive kernel launches especially
    when the LoRA distribution is skewed.
    此后端主要基于Punica论文(https://arxiv.org/pdf/2310.18547)中提出的SGMV
    (分段收集矩阵向量乘法)算法。此处的一个主要变体是将输入序列分割为
    固定大小的块，以减少LoRA分布不均时过多的核函数启动。
    """

    name = "csgmv"  # 后端名称标识

    def __init__(  # 初始化方法
        self,
        max_loras_per_batch: int,  # 每批次最大LoRA数量
        device: torch.device,  # 运行设备
        server_args: ServerArgs,  # 服务器参数
    ):
        super().__init__(max_loras_per_batch, device)  # 调用父类初始化
        self.max_chunk_size = server_args.max_lora_chunk_size  # 从服务器参数获取最大块大小

    def run_lora_a_embedding(  # 运行LoRA A嵌入查找
        self,
        input_ids: torch.Tensor,  # token ID张量
        weights: torch.Tensor,  # LoRA A嵌入权重
        vocab_size: int,  # 基础词表大小
        extra_embeddings: torch.Tensor = None,  # 额外token嵌入
        *args,
        **kwargs,
    ) -> torch.Tensor:
        assert (  # 断言不支持额外嵌入
            extra_embeddings is None
        ), "Extra embeddings for lora a is not supported yet in chunked backend"  # 分段后端尚不支持LoRA A的额外嵌入
        return chunked_embedding_lora_a_forward(  # 调用分段式嵌入LoRA A前向算子
            input_ids=input_ids,  # 输入token ID
            weights=weights,  # 嵌入权重
            batch_info=self.batch_info,  # 批量信息
            vocab_size=vocab_size,  # 词表大小
        )

    def run_lora_a_sgemm(  # 运行LoRA A矩阵的分段GEMM（shrink阶段）
        self,
        x: torch.Tensor,  # 输入张量
        weights: torch.Tensor,  # LoRA A权重
        pruned_batch_info: LoRABatchInfo = None,  # 裁剪后的批量信息（可选）
        stack_num: int = 1,  # 堆叠数量（如QKV为3，gate_up为2）
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_info = (  # 选择使用的批量信息
            pruned_batch_info if pruned_batch_info is not None else self.batch_info  # 优先使用裁剪后的
        )
        return chunked_sgmv_lora_shrink_forward(  # 调用分段式SGMV shrink前向算子
            x=x,  # 输入张量
            weights=weights,  # LoRA A权重
            batch_info=batch_info,  # 批量信息
            num_slices=stack_num,  # 切片数量
        )

    def run_lora_b_sgemm(  # 运行LoRA B矩阵的分段GEMM（expand阶段）
        self,
        x: torch.Tensor,  # 输入张量（LoRA A的输出）
        weights: torch.Tensor,  # LoRA B权重
        output_offset: torch.Tensor,  # 输出偏移量
        base_output: torch.Tensor = None,  # 基础模型输出（可选）
        pruned_batch_info: LoRABatchInfo = None,  # 裁剪后的批量信息（可选）
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # For simple lora B, we use slice offsets [0, output_dim]
        # 对于简单LoRA B，使用切片偏移[0, output_dim]
        output_dim = weights.shape[-2]  # 获取输出维度
        max_slice_size = output_dim  # 最大切片大小等于输出维度
        batch_info = (  # 选择使用的批量信息
            pruned_batch_info if pruned_batch_info is not None else self.batch_info  # 优先使用裁剪后的
        )
        return chunked_sgmv_lora_expand_forward(  # 调用分段式SGMV expand前向算子
            x=x,  # 输入张量
            weights=weights,  # LoRA B权重
            batch_info=batch_info,  # 批量信息
            slice_offsets=output_offset,  # 切片偏移
            max_slice_size=max_slice_size,  # 最大切片大小
            base_output=base_output,  # 基础输出
        )

    def run_qkv_lora(  # 运行QKV投影的LoRA计算
        self,
        x: torch.Tensor,  # 输入张量
        qkv_lora_a: torch.Tensor,  # QKV的LoRA A权重
        qkv_lora_b: torch.Tensor,  # QKV的LoRA B权重
        output_offset: torch.Tensor,  # 输出偏移量
        max_qkv_out_dim: int,  # QKV最大输出维度
        base_output: torch.Tensor = None,  # 基础模型输出（可选）
        n_slices: int = 3,  # 切片数量（默认3，对应Q/K/V）
        *args,
        **kwargs,
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # x: (s, 输入维度)
        # qkv_lora_a: (num_lora, n_slices * r, input_dim)
        # qkv_lora_a: (LoRA数量, 切片数 * 秩, 输入维度)
        # qkv_lora_b: (num_lora, total_output_dim, r)
        # qkv_lora_b: (LoRA数量, 总输出维度, 秩)
        assert isinstance(qkv_lora_b, torch.Tensor)  # 确保qkv_lora_b是张量类型

        lora_a_output = chunked_sgmv_lora_shrink_forward(  # 计算LoRA A的shrink前向
            x=x,  # 输入张量
            weights=qkv_lora_a,  # QKV LoRA A权重
            batch_info=self.batch_info,  # 批量信息
            num_slices=n_slices,  # 切片数量
        )
        lora_output = chunked_sgmv_lora_expand_forward(  # 计算LoRA B的expand前向
            x=lora_a_output,  # LoRA A的输出
            weights=qkv_lora_b,  # QKV LoRA B权重
            batch_info=self.batch_info,  # 批量信息
            slice_offsets=output_offset,  # 切片偏移
            max_slice_size=max_qkv_out_dim,  # 最大切片大小
            base_output=base_output,  # 基础输出
        )
        return lora_output  # 返回LoRA输出

    def run_gate_up_lora(  # 运行Gate-Up投影的LoRA计算
        self,
        x: torch.Tensor,  # 输入张量
        gate_up_lora_a: torch.Tensor,  # Gate-Up的LoRA A权重
        gate_up_lora_b: torch.Tensor,  # Gate-Up的LoRA B权重
        output_offset: torch.Tensor,  # 输出偏移量
        base_output: torch.Tensor = None,  # 基础模型输出（可选）
        *args,
        **kwargs,
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # x: (s, 输入维度)
        # gate_up_lora_a: (num_lora, 2 * r, input_dim)
        # gate_up_lora_a: (LoRA数量, 2 * 秩, 输入维度)
        # gate_up_lora_b: (num_lora, 2 * output_dim, r)
        # gate_up_lora_b: (LoRA数量, 2 * 输出维度, 秩)
        assert isinstance(gate_up_lora_b, torch.Tensor)  # 确保gate_up_lora_b是张量类型
        output_dim = gate_up_lora_b.shape[-2] // 2  # 计算每个切片的输出维度

        # lora_a_output: (s, 2 * r)
        # lora_a_output: (s, 2 * 秩)
        lora_a_output = chunked_sgmv_lora_shrink_forward(  # 计算LoRA A的shrink前向
            x=x,  # 输入张量
            weights=gate_up_lora_a,  # Gate-Up LoRA A权重
            batch_info=self.batch_info,  # 批量信息
            num_slices=2,  # gate_up有2个切片
        )
        lora_output = chunked_sgmv_lora_expand_forward(  # 计算LoRA B的expand前向
            x=lora_a_output,  # LoRA A的输出
            weights=gate_up_lora_b,  # Gate-Up LoRA B权重
            batch_info=self.batch_info,  # 批量信息
            slice_offsets=output_offset,  # 切片偏移
            max_slice_size=output_dim,  # 最大切片大小
            base_output=base_output,  # 基础输出
        )
        return lora_output  # 返回LoRA输出

    def _determine_chunk_size(self, forward_batch: ForwardBatch) -> int:  # 根据批量中token数量启发式确定块大小
        """
        Heuristically determine the chunk size based on token token number in a batch.
        根据批量中token数量启发式确定块大小。

        Args:
            forward_batch (ForwardBatch): The batch information containing sequence lengths.
            forward_batch (ForwardBatch): 包含序列长度的批量信息。

        Returns:
            The determined chunk size
            确定的块大小
        """
        num_tokens = (  # 计算token数量
            forward_batch.extend_num_tokens  # 扩展模式使用扩展token数
            if forward_batch.forward_mode.is_extend()  # 如果是扩展模式
            else forward_batch.batch_size  # 解码模式使用批量大小
        )
        return self._determine_chunk_size_for_tokens(num_tokens)  # 根据token数确定块大小

    def _determine_chunk_size_for_tokens(self, num_tokens: int) -> int:  # 根据token数直接确定块大小
        """Determine chunk size given a token count directly.
        给定token数量直接确定块大小。"""
        if self.max_chunk_size <= MIN_CHUNK_SIZE:  # 如果最大块大小不超过最小块大小
            return MIN_CHUNK_SIZE  # 返回最小块大小

        if num_tokens >= 256:  # token数大于等于256
            chunk_size = 128  # 使用128的块大小
        elif num_tokens >= 64:  # token数大于等于64
            chunk_size = 32  # 使用32的块大小
        else:  # num_tokens < 64  # token数小于64
            chunk_size = 16  # 使用16的块大小
        return min(self.max_chunk_size, chunk_size)  # 不超过最大块大小

    @staticmethod
    def _build_req_seg_indptr(forward_batch: ForwardBatch) -> torch.Tensor:  # 在CPU上构建每请求的累积token边界（固定内存）
        """Build per-request cumulative token boundaries on CPU (pinned).
        在CPU上构建每请求的累积token边界（固定内存）。"""
        bs = forward_batch.batch_size  # 获取批量大小
        if forward_batch.forward_mode.is_decode():  # 如果是解码模式
            indptr = torch.arange(bs + 1, dtype=torch.int32, pin_memory=True)  # 解码模式每请求1个token
        else:  # 扩展模式
            seg_lens = generate_sequence_lengths(forward_batch, device="cpu")  # 在CPU上生成序列长度
            indptr = torch.zeros(bs + 1, dtype=torch.int32, pin_memory=True)  # 初始化索引指针
            torch.cumsum(seg_lens, dim=0, out=indptr[1:])  # 计算累积和
        return indptr  # 返回索引指针

    def init_cuda_graph_batch_info(  # 初始化CUDA Graph模式的批量信息
        self,
        max_bs_in_cuda_graph: int,  # 图模式下的最大批量大小
        num_tokens_per_bs: int,  # 每个序列的token数量
    ):
        max_num_segments = (  # 计算最大分段数
            (num_tokens_per_bs + MIN_CHUNK_SIZE - 1) // MIN_CHUNK_SIZE  # 每个序列的分段数
        ) * max_bs_in_cuda_graph  # 乘以批量大小
        max_num_tokens = max_bs_in_cuda_graph * num_tokens_per_bs  # 计算最大token数
        with torch.device("cuda"):  # 在CUDA设备上下文中创建张量
            self.cuda_graph_batch_info = LoRABatchInfo(  # 创建CUDA Graph批量信息
                bs=max_bs_in_cuda_graph,  # 批量大小
                use_cuda_graph=True,  # 启用图模式
                seg_lens=torch.zeros(max_num_segments, dtype=torch.int32),  # 分段长度
                seg_indptr=torch.zeros(max_num_segments + 1, dtype=torch.int32),  # 分段索引指针
                weight_indices=torch.zeros(max_num_segments, dtype=torch.int32),  # 权重索引
                permutation=torch.zeros(max_num_tokens, dtype=torch.int32),  # 排列索引
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32),  # LoRA秩
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float),  # 缩放因子
                num_segments=None,  # Set per batch  # 分段数量（每批次设置）
                max_len=None,  # Not used in CSGMV backend  # 最大长度（CSGMV后端不使用）
                req_seg_indptr=torch.zeros(max_bs_in_cuda_graph + 1, dtype=torch.int32),  # 请求级分段指针
                req_weight_indices=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32),  # 请求级权重索引
            )

    def prepare_lora_batch(  # 准备当前前向批次的LoRA权重和批量信息
        self,
        forward_batch: ForwardBatch,  # 当前前向批次
        weight_indices: list[int],  # LoRA权重索引列表
        lora_ranks: list[int],  # LoRA秩列表
        scalings: list[float],  # 缩放因子列表
        use_cuda_graph: bool,  # 是否使用CUDA Graph
    ):
        chunk_size = self._determine_chunk_size(forward_batch)  # 确定块大小

        permutation, weight_indices_reordered = ChunkedSgmvLoRABackend._get_permutation(  # 计算排列和重排后的权重索引
            seq_weight_indices=weight_indices,  # 序列级权重索引
            forward_batch=forward_batch,  # 前向批次信息
        )

        seg_weight_indices, seg_indptr = self._get_segments_info(  # 计算分段信息
            weights_reordered=weight_indices_reordered,  # 重排后的权重索引
            chunk_size=chunk_size,  # 块大小
        )
        num_segments = len(seg_weight_indices)  # 获取分段数量

        lora_ranks_tensor = torch.tensor(  # 创建LoRA秩张量
            lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"  # 使用固定内存
        )
        scalings_tensor = torch.tensor(  # 创建缩放因子张量
            scalings, dtype=torch.float, pin_memory=True, device="cpu"  # 使用固定内存
        )

        bs = forward_batch.batch_size  # 获取批量大小
        req_wi_tensor = torch.tensor(  # 创建请求级权重索引张量
            weight_indices, dtype=torch.int32, pin_memory=True, device="cpu"  # 使用固定内存
        )
        req_seg_indptr_cpu = self._build_req_seg_indptr(forward_batch)  # 构建请求级分段指针

        if not use_cuda_graph:  # 非图模式
            batch_info = LoRABatchInfo(  # 创建新的批量信息
                bs=bs,  # 批量大小
                num_segments=num_segments,  # 分段数量
                max_len=chunk_size,  # 块大小
                use_cuda_graph=False,  # 不使用图模式
                seg_indptr=torch.empty(  # 分段索引指针
                    (num_segments + 1,), dtype=torch.int32, device=self.device  # 形状为(分段数+1,)
                ),
                weight_indices=torch.empty(  # 权重索引
                    (num_segments,), dtype=torch.int32, device=self.device  # 形状为(分段数,)
                ),
                lora_ranks=torch.empty(  # LoRA秩
                    (self.max_loras_per_batch,), dtype=torch.int32, device=self.device  # 形状为(最大LoRA数,)
                ),
                scalings=torch.empty(  # 缩放因子
                    (self.max_loras_per_batch,), dtype=torch.float, device=self.device  # 形状为(最大LoRA数,)
                ),
                permutation=torch.empty(  # 排列索引
                    (len(permutation),), dtype=torch.int32, device=self.device  # 形状为(token数,)
                ),
                seg_lens=None,  # 分段长度（CSGMV后端不使用）
                req_seg_indptr=torch.empty(  # 请求级分段指针
                    (bs + 1,), dtype=torch.int32, device=self.device  # 形状为(批量大小+1,)
                ),
                req_weight_indices=torch.empty(  # 请求级权重索引
                    (bs,), dtype=torch.int32, device=self.device  # 形状为(批量大小,)
                ),
            )
        else:  # 图模式
            batch_info = self.cuda_graph_batch_info  # 使用预分配的CUDA Graph批量信息
            batch_info.bs = bs  # 更新批量大小
            batch_info.num_segments = num_segments  # 更新分段数量
            batch_info.max_len = chunk_size  # 更新块大小

        # Copy to device asynchronously
        # 异步复制到设备
        batch_info.lora_ranks[: self.max_loras_per_batch].copy_(  # 复制LoRA秩
            lora_ranks_tensor, non_blocking=True  # 非阻塞复制
        )
        batch_info.scalings[: self.max_loras_per_batch].copy_(  # 复制缩放因子
            scalings_tensor, non_blocking=True  # 非阻塞复制
        )
        batch_info.weight_indices[:num_segments].copy_(  # 复制权重索引
            seg_weight_indices, non_blocking=True  # 非阻塞复制
        )
        batch_info.seg_indptr[: num_segments + 1].copy_(seg_indptr, non_blocking=True)  # 复制分段指针（非阻塞）
        batch_info.permutation[: len(permutation)].copy_(permutation, non_blocking=True)  # 复制排列索引（非阻塞）
        batch_info.req_seg_indptr[: bs + 1].copy_(req_seg_indptr_cpu, non_blocking=True)  # 复制请求级分段指针（非阻塞）
        batch_info.req_weight_indices[:bs].copy_(req_wi_tensor, non_blocking=True)  # 复制请求级权重索引（非阻塞）

        batch_info = self._add_moe_lora_info(forward_batch, batch_info)  # 添加MoE LoRA信息

        self.batch_info = batch_info  # 保存当前批次信息
        self.lm_head_batch_info, self.lm_head_pass_batch_infos = (  # 准备LM Head批量信息
            self._prepare_lm_head_batch_info(forward_batch, weight_indices, batch_info)  # 调用准备方法
        )

    def _prepare_lm_head_batch_info(  # 准备LM Head的批量信息
        self,
        forward_batch: ForwardBatch,  # 当前前向批次
        weight_indices: list[int],  # LoRA权重索引列表
        batch_info: LoRABatchInfo,  # 当前批量信息
    ) -> Tuple[Optional[LoRABatchInfo], Optional[List[LoRABatchInfo]]]:  # 返回(LM Head批量信息, 逐Pass批量信息列表)

        # Precompute lm_head_batch_info for pruned lm_head LoRA
        # 为裁剪后的lm_head LoRA预计算lm_head_batch_info
        pruned_lens = get_lm_head_pruned_lens(forward_batch)  # 获取LM Head裁剪长度
        lm_head_batch_info = None  # 初始化LM Head批量信息
        lm_head_pass_batch_infos = None  # 初始化逐Pass批量信息

        if pruned_lens is not None:  # 如果有裁剪长度
            pruned_total = sum(pruned_lens)  # 计算裁剪后的总token数
            chunk_size = self._determine_chunk_size_for_tokens(pruned_total)  # 确定块大小
            lm_head_segments = merge_and_chunk_segments(  # 合并和分块分段
                weight_indices, pruned_lens, chunk_size=chunk_size  # 权重索引、裁剪长度和块大小
            )
            lm_head_batch_info = self._build_lm_head_batch_info(  # 构建LM Head批量信息
                lm_head_segments, batch_info, chunk_size, pruned_total  # 分段信息、批量信息、块大小和总token数
            )

            # Precompute per-pass batch_infos for logprobs chunking
            # 为logprobs分块预计算每pass的batch_info
            pass_segments = self._get_lm_head_pass_segments(weight_indices, pruned_lens)  # 获取每Pass分段信息
            if pass_segments is not None:  # 如果有分段信息
                lm_head_pass_batch_infos = []  # 初始化列表
                for seg_wi, seg_lens_list in pass_segments:  # 遍历每个Pass
                    pass_total = sum(seg_lens_list)  # 计算当前Pass的总token数
                    pass_chunk_size = self._determine_chunk_size_for_tokens(pass_total)  # 确定当前Pass的块大小
                    chunked_segments = merge_and_chunk_segments(  # 合并和分块分段
                        seg_wi, seg_lens_list, chunk_size=pass_chunk_size  # 权重索引、长度列表和块大小
                    )
                    lm_head_pass_batch_infos.append(  # 添加到列表
                        self._build_lm_head_batch_info(  # 构建LM Head批量信息
                            chunked_segments,  # 分块分段
                            batch_info,  # 批量信息
                            pass_chunk_size,  # 块大小
                            pass_total,  # 总token数
                        )
                    )

        return lm_head_batch_info, lm_head_pass_batch_infos  # 返回LM Head批量信息和逐Pass批量信息

    def _build_lm_head_batch_info(  # 构建裁剪后LM Head输入的LoRABatchInfo
        self,
        lm_head_segments: Tuple[List[int], List[int]],  # LM Head分段信息（权重索引列表, 长度列表）
        batch_info: LoRABatchInfo,  # 原始批量信息
        chunk_size: int,  # 块大小
        expected_tokens: int,  # 预期token数
    ) -> LoRABatchInfo:
        seg_weight_indices_cpu, seg_lens_cpu = lm_head_segments  # 解包分段信息
        pruned_total = sum(seg_lens_cpu)  # 计算裁剪后的总token数
        num_segments = len(seg_weight_indices_cpu)  # 获取分段数量

        weight_indices = torch.tensor(  # 创建权重索引张量
            seg_weight_indices_cpu, dtype=torch.int32, device=self.device  # 在设备上创建
        )
        seg_lens = torch.tensor(seg_lens_cpu, dtype=torch.int32, device=self.device)  # 创建分段长度张量
        seg_indptr = torch.zeros(  # 初始化分段索引指针
            (num_segments + 1,), dtype=torch.int32, device=self.device  # 形状为(分段数+1,)
        )
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)  # 计算累积和

        # Identity permutation (lm_head tokens are in original order)
        # 恒等排列（lm_head token保持原始顺序）
        permutation = torch.arange(pruned_total, dtype=torch.int32, device=self.device)  # 创建恒等排列

        return dataclasses.replace(  # 创建替换字段的批量信息副本
            batch_info,  # 原始批量信息
            num_segments=num_segments,  # 分段数量
            max_len=chunk_size,  # 块大小
            seg_indptr=seg_indptr,  # 分段索引指针
            weight_indices=weight_indices,  # 权重索引
            permutation=permutation,  # 排列索引
            expected_tokens=expected_tokens,  # 预期token数
        )

    @staticmethod
    def _get_permutation(seq_weight_indices, forward_batch: ForwardBatch):  # 计算按LoRA适配器分组token的排列索引
        """
        Computes permutation indices for reordering tokens by their LoRA adapter assignments.
        计算按LoRA适配器分配重排token的排列索引。

        This function implements the "gather" step in Chunked Segmented Gather Matrix Vector
        multiplication by creating a permutation that groups tokens by their LoRA adapter.
        Tokens using the same LoRA adapter are placed together to enable efficient batched
        computation.
        此函数实现了分段式分段收集矩阵向量乘法中的"聚集"步骤，
        通过创建将token按LoRA适配器分组的排列。使用相同LoRA适配器
        的token被放在一起以实现高效的批量计算。

        Example:
            seq_weight_indices = [0, 1, 0]  # 3 sequences using adapters [0, 1, 0]
            seq_weight_indices = [0, 1, 0]  # 3个序列使用适配器[0, 1, 0]
            extend_seq_lens = [2, 1, 3]     # sequence lengths [2, 1, 3 tokens]
            extend_seq_lens = [2, 1, 3]     # 序列长度[2, 1, 3个token]

            # Creates row_weight_indices: [0, 0, 1, 0, 0, 0] (6 tokens total)
            # 创建row_weight_indices: [0, 0, 1, 0, 0, 0]（共6个token）
            # Returns permutation: [0, 1, 3, 4, 5, 2] (groups adapter 0 tokens together)
            # 返回排列: [0, 1, 3, 4, 5, 2]（将适配器0的token分组在一起）
            # weights_reordered: [0, 0, 0, 0, 0, 1] (sorted by adapter)
            # 重排权重: [0, 0, 0, 0, 0, 1]（按适配器排序）

        Args:
            seq_weight_indices: List of LoRA adapter indices for each sequence
            seq_weight_indices: 每个序列的LoRA适配器索引列表
            forward_batch (ForwardBatch): Batch information containing sequence lengths
            forward_batch (ForwardBatch): 包含序列长度的批量信息

        Returns:
            tuple: (permutation, weights_reordered) where:
            tuple: (排列, 重排权重) 其中：
                - permutation: Token reordering indices to group by adapter
                - permutation: 按适配器分组的token重排索引
                - weights_reordered: Sorted adapter indices for each token
                - weights_reordered: 每个token排序后的适配器索引
        """
        with torch.device("cpu"):  # 在CPU上执行
            seq_weight_indices = torch.tensor(seq_weight_indices, dtype=torch.int32)  # 转换为张量
            seg_lens_cpu = generate_sequence_lengths(forward_batch)  # 生成序列长度

            row_weight_indices = torch.repeat_interleave(  # 将序列级权重索引扩展到行级
                seq_weight_indices, seg_lens_cpu  # 按序列长度重复
            )
            permutation = torch.empty(  # 创建排列张量
                (len(row_weight_indices),), dtype=torch.long, pin_memory=True  # 使用固定内存
            )
            torch.argsort(row_weight_indices, stable=True, out=permutation)  # 稳定排序获取排列索引
            weights_reordered = row_weight_indices[permutation]  # 按排列重排权重索引

            return permutation, weights_reordered  # 返回排列和重排后的权重索引

    def _get_segments_info(self, weights_reordered: torch.Tensor, chunk_size: int):  # 计算分段式SGMV操作的分段信息
        """
        Computes segment information for chunked SGMV operations.
        计算分段式SGMV操作的分段信息。

        This function takes the reordered weight indices and creates segments of fixed size
        (self.segment_size) for efficient kernel execution. Each segment contains tokens
        that use the same LoRA adapter, enabling vectorized computation.
        此函数接收重排后的权重索引，创建固定大小(self.segment_size)的分段
        以实现高效的核函数执行。每个分段包含使用相同LoRA适配器的token，
        以实现向量化计算。

        The segmentation is necessary because:
        分段是必要的，因为：
        1. GPU kernels work efficiently on fixed-size blocks
        1. GPU核函数在固定大小块上高效运行
        2. Large groups of tokens using the same adapter are split into manageable chunks
        2. 使用相同适配器的大token组被分割为可管理的块
        3. Each segment can be processed independently in parallel
        3. 每个分段可以独立并行处理

        Example:
            weights_reordered = [0, 0, 0, 0, 0, 1]  # 5 tokens with adapter 0, 1 with adapter 1
            weights_reordered = [0, 0, 0, 0, 0, 1]  # 5个token使用适配器0，1个使用适配器1
            segment_size = 3
            segment_size = 3

            # Creates segments:
            # 创建分段：
            # Segment 0: tokens 0-2 (adapter 0), length=3
            # 分段0: token 0-2（适配器0），长度=3
            # Segment 1: tokens 3-4 (adapter 0), length=2
            # 分段1: token 3-4（适配器0），长度=2
            # Segment 2: token 5 (adapter 1), length=1
            # 分段2: token 5（适配器1），长度=1

            # Returns:
            # 返回：
            # weight_indices_list: [0, 0, 1] (adapter for each segment)
            # weight_indices_list: [0, 0, 1]（每个分段的适配器）
            # seg_indptr: [0, 3, 5, 6] (cumulative segment boundaries)
            # seg_indptr: [0, 3, 5, 6]（累积分段边界）

        Args:
            weights_reordered (torch.Tensor): Sorted adapter indices for each token
            weights_reordered (torch.Tensor): 每个token排序后的适配器索引
            chunk_size (int): Fixed size for each segment
            chunk_size (int): 每个分段的固定大小

        Returns:
            tuple: (weight_indices_list, seg_indptr) where:
            tuple: (权重索引列表, 分段指针) 其中：
                - weight_indices_list: LoRA adapter index for each segment
                - weight_indices_list: 每个分段的LoRA适配器索引
                - seg_indptr: Cumulative segment boundaries (CSR-style indptr)
                - seg_indptr: 累积分段边界（CSR风格的indptr）
        """
        with torch.device("cpu"):  # 在CPU上执行
            unique_weights, counts = torch.unique_consecutive(  # 找出连续的唯一权重和计数
                weights_reordered, return_counts=True  # 返回计数值
            )

            weight_indices_list = []  # 初始化权重索引列表
            seg_lens_list = []  # 初始化分段长度列表

            for weight_idx, group_len in zip(unique_weights, counts):  # 遍历每个唯一权重组
                group_len = group_len.item()  # 转换为Python整数
                num_segs = (group_len + chunk_size - 1) // chunk_size  # 计算该组需要的分段数（向上取整）

                weight_indices_list.extend([weight_idx.item()] * num_segs)  # 添加权重索引
                seg_lens_list.extend([chunk_size] * (num_segs - 1))  # 添加完整块的分段长度
                seg_lens_list.append(group_len - (num_segs - 1) * chunk_size)  # 最后一个块的长度可能不足chunk_size

            seg_lens = torch.tensor(seg_lens_list, dtype=torch.int32)  # 创建分段长度张量

            weight_indices_list = torch.tensor(  # 创建权重索引张量
                weight_indices_list, dtype=torch.int32, pin_memory=True  # 使用固定内存
            )

            seg_indptr = torch.empty(  # 创建分段索引指针
                (len(seg_lens) + 1,), dtype=torch.int32, pin_memory=True  # 使用固定内存
            )
            seg_indptr[0] = 0  # 第一个元素为0
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)  # 后续元素为累积和

            return weight_indices_list, seg_indptr  # 返回权重索引列表和分段索引指针
