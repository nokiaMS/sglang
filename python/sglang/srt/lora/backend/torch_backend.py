# 基于PyTorch原生算子的LoRA后端实现
# 提供TorchNativeLoRABackend类，使用纯PyTorch/Sgemm算子执行LoRA的前向计算
# 支持CUDA Graph模式和普通模式，包括LoRA A/B矩阵乘法及QKV/Gate-Up LoRA
from dataclasses import dataclass # 导入数据类装饰器
from typing import Optional # 导入可选类型

import torch # 导入PyTorch库

from sglang.srt.lora.backend.base_backend import BaseLoRABackend # 导入LoRA后端基类
from sglang.srt.lora.torch_ops import ( # 导入Torch实现的LoRA Sgemm算子
    sgemm_lora_a_embedding_fwd, # LoRA A矩阵嵌入查找前向
    sgemm_lora_a_fwd, # LoRA A矩阵Sgemm前向
    sgemm_lora_b_fwd, # LoRA B矩阵Sgemm前向
)
from sglang.srt.lora.utils import LoRABatchInfo, generate_sequence_lengths # 导入LoRA批处理信息和序列长度生成工具
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批次信息


@dataclass # 数据类装饰器
class TorchNativeLoRABatchInfo(LoRABatchInfo): # Torch原生LoRA批处理信息，继承自LoRABatchInfo
    # ranks of each lora adapter, in shape (lora_num,) placed on cpu device
    # 每个LoRA适配器的秩，形状为(lora_num,)，存放于CPU设备上
    lora_ranks_cpu: Optional[torch.Tensor] = None # LoRA秩的CPU张量

    # Indice pointers of each segment in shape (num_segments + 1, ) placed on cpu device
    # 每个段的索引指针，形状为(num_segments + 1,)，存放于CPU设备上
    seg_indptr_cpu: Optional[torch.Tensor] = None # 段索引指针的CPU张量

    # Lengths of each segments in shape (num_segments,) placed on cpu device
    # 每个段的长度，形状为(num_segments,)，存放于CPU设备上
    seg_lens_cpu: Optional[torch.Tensor] = None # 段长度的CPU张量

    # The index of lora adapter used by each segment, in shape (num_segments,) placed on cpu device
    # 每个段使用的LoRA适配器索引，形状为(num_segments,)，存放于CPU设备上
    weight_indices_cpu: Optional[torch.Tensor] = None # 权重索引的CPU张量

    # Scaling factors for each lora adapter, in shape (lora_num,) placed on cpu device
    # 每个LoRA适配器的缩放因子，形状为(lora_num,)，存放于CPU设备上
    scalings_cpu: Optional[torch.Tensor] = None # 缩放因子的CPU张量


class TorchNativeLoRABackend(BaseLoRABackend): # Torch原生LoRA后端类，继承自BaseLoRABackend
    name = "torch_native" # 后端名称标识

    def __init__( # 初始化方法
        self,
        max_loras_per_batch: int, # 每批最大LoRA数量
        device: torch.device, # 计算设备
        **kwargs, # 其他关键字参数
    ):
        super().__init__(max_loras_per_batch, device) # 调用父类初始化

    def run_lora_a_embedding( # 运行LoRA A矩阵嵌入查找
        self,
        input_ids: torch.Tensor, # 输入token ID张量
        weights: torch.Tensor, # LoRA权重张量
        vocab_size: int, # 词表大小
        extra_embeddings: torch.Tensor = None, # 额外嵌入张量
        *args, # 位置参数
        **kwargs, # 关键字参数
    ) -> torch.Tensor: # 返回输出张量
        assert (
            extra_embeddings is None
        ), "Extra embeddings for lora a is not supported yet in chunked backend" # 断言不支持额外嵌入
        output_tensor = sgemm_lora_a_embedding_fwd( # 调用LoRA A嵌入前向算子
            inputs=input_ids, # 输入token ID
            weights=weights, # LoRA权重
            batch_info=self.batch_info, # 批处理信息
            vocab_size=vocab_size, # 词表大小
        )

        return output_tensor # 返回输出张量

    def run_lora_a_sgemm( # 运行LoRA A矩阵Sgemm乘法
        self,
        x: torch.Tensor, # 输入张量
        weights: torch.Tensor, # LoRA A权重张量
        stack_num: int = 1, # 堆叠数量（切片数）
        *args, # 位置参数
        **kwargs, # 关键字参数
    ) -> torch.Tensor: # 返回输出张量
        output_tensor = sgemm_lora_a_fwd( # 调用LoRA A Sgemm前向算子
            inputs=x, # 输入张量
            weights=weights, # LoRA A权重
            batch_info=self.batch_info, # 批处理信息
            num_slices=stack_num, # 切片数量
        )

        return output_tensor # 返回输出张量

    def run_lora_b_sgemm( # 运行LoRA B矩阵Sgemm乘法
        self,
        x: torch.Tensor, # 输入张量
        weights: torch.Tensor, # LoRA B权重张量
        output_offset_cpu: torch.Tensor, # 输出偏移量的CPU张量
        base_output: torch.Tensor = None, # 基础输出张量（用于残差加法）
        *args, # 位置参数
        **kwargs, # 关键字参数
    ) -> torch.Tensor: # 返回输出张量
        _, weight_out_dim, _ = weights.shape # 解构权重形状，获取输出维度

        output_tensor = sgemm_lora_b_fwd( # 调用LoRA B Sgemm前向算子
            inputs=x, # 输入张量
            weights=weights, # LoRA B权重
            batch_info=self.batch_info, # 批处理信息
            slice_offsets=output_offset_cpu, # 切片偏移量
            base_output=base_output, # 基础输出
        )

        return output_tensor # 返回输出张量

    def run_qkv_lora( # 运行QKV LoRA计算（A和B矩阵乘法串联）
        self,
        x: torch.Tensor, # 输入张量
        qkv_lora_a: torch.Tensor, # QKV LoRA A权重
        qkv_lora_b: torch.Tensor, # QKV LoRA B权重
        output_offset: torch.Tensor, # 输出偏移量
        output_offset_cpu: torch.Tensor, # 输出偏移量的CPU张量
        max_qkv_out_dim: int, # 最大QKV输出维度
        base_output: torch.Tensor = None, # 基础输出张量
        n_slices: int = 3, # 切片数量（Q/K/V三部分）
        *args, # 位置参数
        **kwargs, # 关键字参数
    ) -> torch.Tensor: # 返回输出张量
        lora_a_output = sgemm_lora_a_fwd( # 先执行LoRA A矩阵乘法
            inputs=x, # 输入张量
            weights=qkv_lora_a, # QKV LoRA A权重
            batch_info=self.batch_info, # 批处理信息
            num_slices=n_slices, # 切片数量
        )

        output_tensor = sgemm_lora_b_fwd( # 再执行LoRA B矩阵乘法
            inputs=lora_a_output, # LoRA A的输出作为输入
            weights=qkv_lora_b, # QKV LoRA B权重
            batch_info=self.batch_info, # 批处理信息
            slice_offsets=output_offset_cpu, # 切片偏移量
            base_output=base_output, # 基础输出
        )

        return output_tensor # 返回输出张量

    def run_gate_up_lora( # 运行Gate-Up LoRA计算
        self,
        x: torch.Tensor, # 输入张量
        gate_up_lora_a: torch.Tensor, # Gate-Up LoRA A权重
        gate_up_lora_b: torch.Tensor, # Gate-Up LoRA B权重
        output_offset_cpu: torch.Tensor, # 输出偏移量的CPU张量
        base_output: torch.Tensor = None, # 基础输出张量
        *args, # 位置参数
        **kwargs, # 关键字参数
    ) -> torch.Tensor: # 返回输出张量
        num_slices = len(output_offset_cpu) - 1 # 从偏移量计算切片数量
        _, weight_out_dim, _ = gate_up_lora_b.shape # 解构权重形状，获取输出维度

        lora_a_output = sgemm_lora_a_fwd( # 先执行LoRA A矩阵乘法
            inputs=x, # 输入张量
            weights=gate_up_lora_a, # Gate-Up LoRA A权重
            batch_info=self.batch_info, # 批处理信息
            num_slices=num_slices, # 切片数量
        )

        output_tensor = sgemm_lora_b_fwd( # 再执行LoRA B矩阵乘法
            inputs=lora_a_output, # LoRA A的输出作为输入
            weights=gate_up_lora_b, # Gate-Up LoRA B权重
            batch_info=self.batch_info, # 批处理信息
            slice_offsets=output_offset_cpu, # 切片偏移量
            base_output=base_output, # 基础输出
        )

        return output_tensor # 返回输出张量

    def init_cuda_graph_batch_info( # 初始化CUDA Graph模式下的批处理信息
        self,
        max_bs_in_cuda_graph: int, # CUDA Graph中的最大批大小
        num_tokens_per_bs: int, # 每个批大小的token数量
    ):
        with torch.device("cuda"): # 在CUDA设备上下文中创建张量
            self.cuda_graph_batch_info = TorchNativeLoRABatchInfo( # 创建Torch原生LoRA批处理信息
                use_cuda_graph=True, # 启用CUDA Graph模式
                bs=max_bs_in_cuda_graph, # 批大小
                num_segments=self.max_loras_per_batch, # 段数量等于最大LoRA数量
                seg_lens=torch.full(
                    (max_bs_in_cuda_graph,), num_tokens_per_bs, dtype=torch.int32
                ), # 段长度，填充为每个批的token数
                seg_indptr=torch.zeros(max_bs_in_cuda_graph + 1, dtype=torch.int32), # 段索引指针，初始化为零
                weight_indices=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32), # 权重索引，初始化为零
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32), # LoRA秩，初始化为零
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float), # 缩放因子，初始化为零
                permutation=None, # 排列信息为空
                max_len=num_tokens_per_bs, # 最大长度为每个批的token数
            )

            # Initialize seg_indptr for CUDA graph as they remain constant
            # across batches.
            # 初始化CUDA Graph的seg_indptr，因为它们在不同批次间保持不变
            torch.cumsum( # 计算段长度的累加和，用于构建索引指针
                self.cuda_graph_batch_info.seg_lens[:max_bs_in_cuda_graph],
                dim=0, # 沿第0维累加
                out=self.cuda_graph_batch_info.seg_indptr[1 : max_bs_in_cuda_graph + 1], # 输出到seg_indptr的[1:]位置
            )

    def prepare_lora_batch( # 准备LoRA批处理数据
        self,
        forward_batch: ForwardBatch, # 前向批次信息
        weight_indices: list[int], # 权重索引列表
        lora_ranks: list[int], # LoRA秩列表
        scalings: list[float], # 缩放因子列表
        use_cuda_graph: bool, # 是否使用CUDA Graph
    ):
        # Do not use merge optimization for graph mode
        # Use pinned memory to avoid synchronizations during host-to-device transfer
        # 图模式不使用合并优化
        # 使用固定内存避免主机到设备传输时的同步
        original_seq_lens_cpu = generate_sequence_lengths(forward_batch, device="cpu") # 生成CPU上的原始序列长度
        if not use_cuda_graph: # 非CUDA Graph模式
            original_weight_indices_tensor = torch.tensor(
                weight_indices, dtype=torch.int32, device="cpu"
            ) # 将权重索引转为CPU张量

            unique_weight_indices_tensor, inverse_weight_indices_tensor = (
                torch.unique_consecutive(
                    original_weight_indices_tensor, return_inverse=True
                )
            ) # 计算连续唯一权重索引及其逆映射

            seg_lens_cpu = ( # 计算每个段的长度
                torch.zeros_like(
                    unique_weight_indices_tensor, dtype=torch.int32, device="cpu"
                ) # 创建与唯一索引同形状的零张量
                .scatter_add_( # 使用散射加法聚合段长度
                    0, # 沿第0维散射
                    inverse_weight_indices_tensor, # 逆映射索引
                    original_seq_lens_cpu, # 原始序列长度
                )
                .pin_memory() # 固定内存以加速H2D传输
            )

            weight_indices_tensor = unique_weight_indices_tensor.pin_memory() # 固定唯一权重索引内存
        else: # CUDA Graph模式
            weight_indices_tensor = torch.repeat_interleave(
                torch.tensor(weight_indices, dtype=torch.int32, device="cpu"),
                original_seq_lens_cpu,
            ).pin_memory() # 按序列长度重复权重索引并固定内存
            seg_lens_cpu = torch.ones_like(weight_indices_tensor).pin_memory() # 每段长度为1，固定内存

        seg_indptr_cpu = torch.zeros(
            (len(seg_lens_cpu) + 1,), dtype=torch.int32, pin_memory=True
        ) # 创建段索引指针，大小为段数+1
        seg_indptr_cpu[1:] = torch.cumsum(seg_lens_cpu, dim=0) # 计算累加和填充索引指针
        lora_ranks_tensor = torch.tensor(
            lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"
        ) # 将LoRA秩列表转为固定内存张量
        scalings_tensor = torch.tensor(
            scalings, dtype=torch.float, pin_memory=True, device="cpu"
        ) # 将缩放因子列表转为固定内存张量

        bs = forward_batch.batch_size # 获取批大小
        num_segments = len(weight_indices_tensor) # 获取段数量

        if use_cuda_graph: # CUDA Graph模式
            assert (
                self.cuda_graph_batch_info is not None
            ), "CUDA Graph batch info is not initialized." # 断言CUDA Graph批信息已初始化
            batch_info = self.cuda_graph_batch_info # 使用预分配的CUDA Graph批信息
            batch_info.bs = forward_batch.batch_size # 更新批大小
            batch_info.num_segments = num_segments # 更新段数量
        else: # 非CUDA Graph模式
            max_len = max(seg_lens_cpu) # 计算最大段长度

            batch_info = TorchNativeLoRABatchInfo( # 创建新的Torch原生LoRA批处理信息
                bs=forward_batch.batch_size, # 批大小
                num_segments=num_segments, # 段数量
                max_len=max_len, # 最大段长度
                use_cuda_graph=False, # 不使用CUDA Graph
                seg_lens=torch.empty((bs,), dtype=torch.int32, device=self.device), # 在GPU上分配段长度张量
                seg_indptr=torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                ), # 在GPU上分配段索引指针张量
                weight_indices=torch.empty(
                    (bs,), dtype=torch.int32, device=self.device
                ), # 在GPU上分配权重索引张量
                lora_ranks=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.int32, device=self.device
                ), # 在GPU上分配LoRA秩张量
                scalings=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.float, device=self.device
                ), # 在GPU上分配缩放因子张量
                permutation=None, # 排列信息为空
            )

        # Copy to device asynchronously
        # 异步拷贝到设备
        batch_info.lora_ranks[: self.max_loras_per_batch].copy_(
            lora_ranks_tensor, non_blocking=True
        ) # 异步拷贝LoRA秩到GPU
        batch_info.scalings[: self.max_loras_per_batch].copy_(
            scalings_tensor, non_blocking=True
        ) # 异步拷贝缩放因子到GPU
        batch_info.weight_indices[:num_segments].copy_(
            weight_indices_tensor, non_blocking=True
        ) # 异步拷贝权重索引到GPU
        batch_info.seg_indptr[: len(seg_indptr_cpu)].copy_(
            seg_indptr_cpu, non_blocking=True
        ) # 异步拷贝段索引指针到GPU
        batch_info.seg_lens[: len(seg_lens_cpu)].copy_(seg_lens_cpu, non_blocking=True) # 异步拷贝段长度到GPU

        batch_info.lora_ranks_cpu = lora_ranks_tensor # 保存CPU上的LoRA秩张量
        batch_info.seg_indptr_cpu = seg_indptr_cpu # 保存CPU上的段索引指针张量
        batch_info.seg_lens_cpu = seg_lens_cpu # 保存CPU上的段长度张量
        batch_info.weight_indices_cpu = weight_indices_tensor # 保存CPU上的权重索引张量
        batch_info.scalings_cpu = scalings_tensor # 保存CPU上的缩放因子张量

        batch_info = self._add_moe_lora_info(forward_batch, batch_info) # 添加MoE LoRA信息
        self.batch_info = batch_info # 保存批处理信息到实例