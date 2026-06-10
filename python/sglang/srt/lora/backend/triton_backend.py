# 基于Triton算子的LoRA后端实现
# 提供TritonLoRABackend类，使用Triton内核执行LoRA的前向计算
# 支持段合并优化、CUDA Graph模式、LM Head批处理信息预计算等功能
import dataclasses # 导入数据类模块
from typing import List, Optional, Tuple # 导入类型提示

import torch # 导入PyTorch库

from sglang.srt.lora.backend.base_backend import BaseLoRABackend # 导入LoRA后端基类
from sglang.srt.lora.triton_ops import ( # 导入Triton实现的LoRA算子
    embedding_lora_a_fwd, # 嵌入LoRA A前向
    gate_up_lora_b_fwd, # Gate-Up LoRA B前向
    qkv_lora_b_fwd, # QKV LoRA B前向
    sgemm_lora_a_fwd, # LoRA A Sgemm前向
    sgemm_lora_b_fwd, # LoRA B Sgemm前向
)
from sglang.srt.lora.utils import ( # 导入LoRA工具函数
    LoRABatchInfo, # LoRA批处理信息
    get_lm_head_pruned_lens, # 获取LM Head修剪后的长度
    merge_and_chunk_segments, # 合并和分块段
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批次信息


class TritonLoRABackend(BaseLoRABackend): # Triton LoRA后端类，继承自BaseLoRABackend
    name = "triton" # 后端名称标识

    def __init__( # 初始化方法
        self,
        max_loras_per_batch: int, # 每批最大LoRA数量
        device: torch.device, # 计算设备
        **kwargs, # 其他关键字参数
    ):
        super().__init__(max_loras_per_batch, device) # 调用父类初始化

    def run_lora_a_embedding( # 运行LoRA A嵌入查找
        self,
        input_ids: torch.Tensor, # 输入token ID张量
        weights: torch.Tensor, # LoRA权重张量
        vocab_size: int, # 词表大小
        extra_embeddings: torch.Tensor = None, # 额外嵌入张量
        *args, # 位置参数
        **kwargs, # 关键字参数
    ) -> torch.Tensor: # 返回输出张量
        """Run LoRA A embedding lookup using Triton kernel.""" # 使用Triton内核运行LoRA A嵌入查找
        return embedding_lora_a_fwd( # 调用Triton嵌入LoRA A前向算子
            input_ids=input_ids, # 输入token ID
            weights=weights, # LoRA权重
            batch_info=self.batch_info, # 批处理信息
            vocab_size=vocab_size, # 词表大小
            extra_embeddings=extra_embeddings, # 额外嵌入
        )

    def _sgemm_info(self, pruned_batch_info=None): # 获取Sgemm批处理信息（支持修剪后的批信息）
        """Return the sgemm batch_info (merged segments when available).""" # 返回sgemm批处理信息（如果可用则使用合并段）
        if pruned_batch_info is not None: # 如果提供了修剪后的批信息
            return pruned_batch_info # 直接返回修剪后的批信息
        return getattr(self, "sgemm_batch_info", None) or self.batch_info # 返回sgemm批信息或默认批信息

    def run_lora_a_sgemm( # 运行LoRA A矩阵Sgemm乘法
        self,
        x: torch.Tensor, # 输入张量
        weights: torch.Tensor, # LoRA A权重张量
        pruned_batch_info: LoRABatchInfo = None, # 修剪后的批处理信息
        stack_num: int = 1, # 堆叠数量（切片数）
        *args, # 位置参数
        **kwargs, # 关键字参数
    ) -> torch.Tensor: # 返回输出张量
        return sgemm_lora_a_fwd( # 调用Triton LoRA A Sgemm前向算子
            x, weights, self._sgemm_info(pruned_batch_info), stack_num=stack_num # 传入输入、权重、批信息和切片数
        )

    def run_lora_b_sgemm( # 运行LoRA B矩阵Sgemm乘法
        self,
        x: torch.Tensor, # 输入张量
        weights: torch.Tensor, # LoRA B权重张量
        base_output: torch.Tensor = None, # 基础输出张量
        pruned_batch_info: LoRABatchInfo = None, # 修剪后的批处理信息
        *args, # 位置参数
        **kwargs, # 关键字参数
    ) -> torch.Tensor: # 返回输出张量
        return sgemm_lora_b_fwd( # 调用Triton LoRA B Sgemm前向算子
            x, weights, self._sgemm_info(pruned_batch_info), base_output # 传入输入、权重、批信息和基础输出
        )

    def run_qkv_lora( # 运行QKV LoRA计算
        self,
        x: torch.Tensor, # 输入张量
        qkv_lora_a: torch.Tensor, # QKV LoRA A权重
        qkv_lora_b: torch.Tensor, # QKV LoRA B权重
        output_offset: torch.Tensor, # 输出偏移量
        max_qkv_out_dim: int, # 最大QKV输出维度
        base_output: torch.Tensor = None, # 基础输出张量
        n_slices: int = 3, # 切片数量（Q/K/V三部分）
        *args, # 位置参数
        **kwargs, # 关键字参数
    ) -> torch.Tensor: # 返回输出张量

        # x: (s, input_dim)
        # qkv_lora_a: (num_lora, n_slices * r, input_dim)
        # qkv_lora_b: (num_lora, total_output_dim, r)
        # x: (序列长度, 输入维度)
        # qkv_lora_a: (LoRA数量, 切片数*秩, 输入维度)
        # qkv_lora_b: (LoRA数量, 总输出维度, 秩)
        assert isinstance(qkv_lora_b, torch.Tensor) # 断言qkv_lora_b是张量类型

        sgemm_info = self._sgemm_info() # 获取Sgemm批处理信息
        lora_a_output = sgemm_lora_a_fwd(x, qkv_lora_a, sgemm_info, stack_num=n_slices) # 执行LoRA A Sgemm
        lora_output = qkv_lora_b_fwd( # 执行QKV LoRA B前向
            lora_a_output, # LoRA A输出
            qkv_lora_b, # QKV LoRA B权重
            sgemm_info, # Sgemm批信息
            output_offset, # 输出偏移量
            max_qkv_out_dim, # 最大QKV输出维度
            base_output, # 基础输出
            n_slices=n_slices, # 切片数量
        )
        return lora_output # 返回LoRA输出

    def run_gate_up_lora( # 运行Gate-Up LoRA计算
        self,
        x: torch.Tensor, # 输入张量
        gate_up_lora_a: torch.Tensor, # Gate-Up LoRA A权重
        gate_up_lora_b: torch.Tensor, # Gate-Up LoRA B权重
        base_output: torch.Tensor = None, # 基础输出张量
        *args, # 位置参数
        **kwargs, # 关键字参数
    ) -> torch.Tensor: # 返回输出张量

        # x: (s, input_dim)
        # gate_up_lora_a: (num_lora, 2 * r, input_dim)
        # gate_up_lora_b: (num_lora, 2 * output_dim, r)
        # x: (序列长度, 输入维度)
        # gate_up_lora_a: (LoRA数量, 2*秩, 输入维度)
        # gate_up_lora_b: (LoRA数量, 2*输出维度, 秩)
        assert isinstance(gate_up_lora_b, torch.Tensor) # 断言gate_up_lora_b是张量类型
        output_dim = gate_up_lora_b.shape[-2] // 2 # 计算单个输出维度（gate或up）

        sgemm_info = self._sgemm_info() # 获取Sgemm批处理信息
        # lora_a_output: (s, 2 * r)
        # lora_a_output: (序列长度, 2*秩)
        lora_a_output = sgemm_lora_a_fwd(x, gate_up_lora_a, sgemm_info, stack_num=2) # 执行LoRA A Sgemm，切片数为2
        lora_output = gate_up_lora_b_fwd( # 执行Gate-Up LoRA B前向
            lora_a_output, # LoRA A输出
            gate_up_lora_b, # Gate-Up LoRA B权重
            sgemm_info, # Sgemm批信息
            output_dim, # 输出维度
            base_output, # 基础输出
        )
        return lora_output # 返回LoRA输出

    def init_cuda_graph_batch_info( # 初始化CUDA Graph模式下的批处理信息
        self,
        max_bs_in_cuda_graph: int, # CUDA Graph中的最大批大小
        num_tokens_per_bs: int, # 每个批大小的token数量
    ):
        max_tokens = max_bs_in_cuda_graph * num_tokens_per_bs # 计算最大token数
        mlpb = self.max_loras_per_batch # 每批最大LoRA数量
        with torch.device("cuda"): # 在CUDA设备上下文中创建张量
            self.cuda_graph_batch_info = LoRABatchInfo( # 创建LoRA批处理信息
                bs=max_bs_in_cuda_graph, # 批大小
                use_cuda_graph=True, # 启用CUDA Graph
                num_segments=None, # 段数量为空
                seg_lens=torch.full(
                    (max_bs_in_cuda_graph,), num_tokens_per_bs, dtype=torch.int32
                ), # 段长度，填充为每个批的token数
                seg_indptr=torch.zeros(max_bs_in_cuda_graph + 1, dtype=torch.int32), # 段索引指针，初始化为零
                max_len=num_tokens_per_bs, # 最大长度
                weight_indices=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32), # 权重索引，初始化为零
                lora_ranks=torch.zeros(mlpb, dtype=torch.int32), # LoRA秩，初始化为零
                scalings=torch.zeros(mlpb, dtype=torch.float), # 缩放因子，初始化为零
                permutation=None, # 排列信息为空
            )

            torch.cumsum( # 计算段长度的累加和
                self.cuda_graph_batch_info.seg_lens[:max_bs_in_cuda_graph],
                dim=0, # 沿第0维累加
                out=self.cuda_graph_batch_info.seg_indptr[1 : max_bs_in_cuda_graph + 1], # 输出到seg_indptr[1:]
            )

            # Sgemm batch_info with segments merged by adapter.
            # Updated each batch by compute_sgemm_routing().
            # Sgemm批处理信息，段按适配器合并。
            # 每次批处理由compute_sgemm_routing()更新。
            self.cuda_graph_sgemm_batch_info = LoRABatchInfo( # 创建Sgemm批处理信息（按适配器合并段）
                bs=mlpb, # 批大小等于最大LoRA数量
                use_cuda_graph=True, # 启用CUDA Graph
                num_segments=mlpb, # 段数量等于最大LoRA数量
                seg_lens=torch.zeros(mlpb, dtype=torch.int32), # 段长度，初始化为零
                seg_indptr=torch.zeros(mlpb + 1, dtype=torch.int32), # 段索引指针，初始化为零
                max_len=max_tokens, # 最大长度为最大token数
                weight_indices=torch.arange(mlpb, dtype=torch.int32), # 权重索引为0到mlpb-1
                lora_ranks=torch.zeros(mlpb, dtype=torch.int32), # LoRA秩，初始化为零
                scalings=torch.zeros(mlpb, dtype=torch.float), # 缩放因子，初始化为零
                permutation=torch.zeros(max_tokens, dtype=torch.int32), # 排列张量，初始化为零
            )

    def compute_sgemm_routing(self, use_cuda_graph: bool): # 计算Sgemm路由，按适配器排序并合并段
        """Sort tokens by adapter and build merged segments for sgemm LoRA.""" # 按适配器排序token并为sgemm LoRA构建合并段
        bi = self.batch_info # 获取批处理信息
        bs = bi.bs # 批大小
        mlpb = self.max_loras_per_batch # 每批最大LoRA数量
        wi = bi.weight_indices[:bs] # 获取前bs个权重索引

        perm = torch.argsort(wi, stable=True).to(torch.int32) # 稳定排序权重索引，得到排列
        sorted_wi = wi[perm] # 按排列获取排序后的权重索引
        adapter_ids = torch.arange(mlpb, device=wi.device, dtype=torch.int32) # 创建适配器ID范围
        seg_starts = torch.searchsorted(sorted_wi, adapter_ids) # 搜索每个适配器在排序数组中的起始位置
        seg_ends = torch.searchsorted(sorted_wi, adapter_ids, right=True) # 搜索每个适配器在排序数组中的结束位置
        seg_lens = seg_ends - seg_starts # 计算每个段的长度

        if use_cuda_graph: # CUDA Graph模式
            sgemm = getattr(self, "cuda_graph_sgemm_batch_info", None) # 获取预分配的sgemm批信息
            if sgemm is None: # 如果不存在
                return # 直接返回
            sgemm.permutation[:bs] = perm # 更新排列
            sgemm.seg_lens[:] = seg_lens # 更新段长度
            sgemm.seg_indptr[0:1].zero_() # 将seg_indptr[0]置零
            torch.cumsum(sgemm.seg_lens, dim=0, out=sgemm.seg_indptr[1:]) # 计算累加和填充seg_indptr[1:]
            sgemm.max_len = bs # 更新最大长度
            sgemm.lora_ranks[:mlpb] = bi.lora_ranks[:mlpb] # 拷贝LoRA秩
            sgemm.scalings[:mlpb] = bi.scalings[:mlpb] # 拷贝缩放因子
        else: # 非CUDA Graph模式
            seg_indptr = torch.zeros(mlpb + 1, dtype=torch.int32, device=wi.device) # 创建段索引指针
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0) # 计算累加和填充seg_indptr[1:]
            sgemm = LoRABatchInfo( # 创建新的Sgemm批处理信息
                bs=mlpb, # 批大小等于最大LoRA数量
                use_cuda_graph=False, # 不使用CUDA Graph
                num_segments=mlpb, # 段数量
                seg_lens=seg_lens, # 段长度
                seg_indptr=seg_indptr, # 段索引指针
                max_len=bs, # 最大长度
                weight_indices=adapter_ids, # 权重索引
                lora_ranks=bi.lora_ranks[:mlpb].clone(), # 克隆LoRA秩
                scalings=bi.scalings[:mlpb].clone(), # 克隆缩放因子
                permutation=perm, # 排列
            )

        self.sgemm_batch_info = sgemm # 保存sgemm批信息

    def prepare_lora_batch( # 准备LoRA批处理数据
        self,
        forward_batch: ForwardBatch, # 前向批次信息
        weight_indices: list[int], # 权重索引列表
        lora_ranks: list[int], # LoRA秩列表
        scalings: list[float], # 缩放因子列表
        use_cuda_graph: bool, # 是否使用CUDA Graph
    ):
        # Use pinned memory to avoid synchronizations during host-to-device transfer
        # 使用固定内存避免主机到设备传输时的同步
        weight_indices_tensor = torch.tensor(
            weight_indices, dtype=torch.int32, pin_memory=True, device="cpu"
        ) # 将权重索引转为固定内存CPU张量
        lora_ranks_tensor = torch.tensor(
            lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"
        ) # 将LoRA秩转为固定内存CPU张量
        scalings_tensor = torch.tensor(
            scalings, dtype=torch.float, pin_memory=True, device="cpu"
        ) # 将缩放因子转为固定内存CPU张量

        bs = forward_batch.batch_size # 获取批大小

        if use_cuda_graph: # CUDA Graph模式
            assert (
                self.cuda_graph_batch_info is not None
            ), "CUDA Graph batch info is not initialized." # 断言CUDA Graph批信息已初始化
            batch_info = self.cuda_graph_batch_info # 使用预分配的CUDA Graph批信息
            batch_info.bs = forward_batch.batch_size # 更新批大小
            batch_info.num_segments = forward_batch.batch_size # 更新段数量
        else: # 非CUDA Graph模式
            max_len = ( # 计算最大段长度
                # Calculate max_len from the CPU copy to avoid D2H transfer.
                # 从CPU副本计算max_len以避免设备到主机传输。
                max(forward_batch.extend_seq_lens_cpu) # 取扩展序列长度的最大值
                if forward_batch.forward_mode.is_extend() # 如果是扩展模式
                else 1 # 否则为1（解码模式每个token一段）
            )
            seg_lens = ( # 计算段长度
                forward_batch.extend_seq_lens # 扩展模式使用扩展序列长度
                if forward_batch.forward_mode.is_extend() # 如果是扩展模式
                else torch.ones(bs, dtype=torch.int32, device=self.device) # 解码模式每段长度为1
            )
            seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device) # 创建段索引指针
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0) # 计算累加和填充seg_indptr[1:]

            batch_info = LoRABatchInfo( # 创建LoRA批处理信息
                bs=forward_batch.batch_size, # 批大小
                num_segments=forward_batch.batch_size, # 段数量
                max_len=max_len, # 最大段长度
                use_cuda_graph=False, # 不使用CUDA Graph
                seg_lens=seg_lens, # 段长度
                seg_indptr=seg_indptr, # 段索引指针
                weight_indices=torch.empty(
                    (bs,), dtype=torch.int32, device=self.device
                ), # 在GPU上分配权重索引张量
                lora_ranks=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.int64, device=self.device
                ), # 在GPU上分配LoRA秩张量（int64）
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
        batch_info.weight_indices[:bs].copy_(weight_indices_tensor, non_blocking=True) # 异步拷贝权重索引到GPU

        batch_info = self._add_moe_lora_info(forward_batch, batch_info) # 添加MoE LoRA信息
        self.batch_info = batch_info # 保存批处理信息到实例

        # Biggest win is in decode.
        # 在解码模式下收益最大。
        is_decode = not forward_batch.forward_mode.is_extend() # 判断是否为解码模式
        if is_decode: # 解码模式
            self.compute_sgemm_routing(use_cuda_graph) # 计算Sgemm路由（合并段优化）
        else: # 扩展模式
            self.sgemm_batch_info = None # 不使用Sgemm路由

        self.lm_head_batch_info, self.lm_head_pass_batch_infos = (
            self._prepare_lm_head_batch_info(forward_batch, weight_indices, batch_info)
        ) # 预计算LM Head批处理信息

    def _prepare_lm_head_batch_info( # 准备LM Head的批处理信息
        self,
        forward_batch: ForwardBatch, # 前向批次信息
        weight_indices: list[int], # 权重索引列表
        batch_info: LoRABatchInfo, # LoRA批处理信息
    ) -> Tuple[Optional[LoRABatchInfo], Optional[List[LoRABatchInfo]]]: # 返回LM Head批信息和每轮批信息

        # Precompute lm_head_batch_info for pruned lm_head LoRA
        # 预计算修剪后lm_head LoRA的批处理信息
        pruned_lens = get_lm_head_pruned_lens(forward_batch) # 获取LM Head修剪后的长度
        lm_head_batch_info = None # 初始化LM Head批信息为空
        lm_head_pass_batch_infos = None # 初始化每轮批信息为空

        if pruned_lens is not None: # 如果存在修剪后的长度
            pruned_total = sum(pruned_lens) # 计算修剪后的总token数
            lm_head_segments = merge_and_chunk_segments(
                weight_indices, pruned_lens, chunk_size=pruned_total
            ) # 合并和分块LM Head段
            lm_head_batch_info = self._build_lm_head_batch_info(
                lm_head_segments, batch_info, pruned_total
            ) # 构建LM Head批处理信息

            # Precompute per-pass batch_infos for logprobs chunking
            # 预计算每轮的批处理信息用于logprobs分块
            pass_segments = self._get_lm_head_pass_segments(weight_indices, pruned_lens) # 获取LM Head每轮段信息
            if pass_segments is not None: # 如果存在每轮段信息
                lm_head_pass_batch_infos = [] # 初始化每轮批信息列表
                for seg_wi, seg_lens_list in pass_segments: # 遍历每轮段
                    pass_total = sum(seg_lens_list) # 计算每轮总token数
                    merged_segments = merge_and_chunk_segments(
                        seg_wi, seg_lens_list, chunk_size=pass_total
                    ) # 合并和分块每轮段
                    self.lm_head_pass_batch_infos.append(
                        self._build_lm_head_batch_info(
                            merged_segments, batch_info, pass_total
                        )
                    ) # 构建并添加每轮LM Head批信息

        return lm_head_batch_info, lm_head_pass_batch_infos # 返回LM Head批信息和每轮批信息

    def _build_lm_head_batch_info( # 构建LM Head批处理信息
        self,
        lm_head_segments: Tuple[List[int], List[int]], # LM Head段（权重索引和段长度）
        batch_info: LoRABatchInfo, # 原始LoRA批处理信息
        expected_tokens: int, # 预期token数
    ) -> LoRABatchInfo: # 返回构建的LM Head批处理信息
        seg_weight_indices_cpu, seg_lens_cpu = lm_head_segments # 解构段权重索引和段长度
        num_segments = len(seg_weight_indices_cpu) # 获取段数量

        seg_lens = torch.tensor(seg_lens_cpu, dtype=torch.int32, device=self.device) # 将段长度转为GPU张量
        seg_indptr = torch.zeros(
            (num_segments + 1,), dtype=torch.int32, device=self.device
        ) # 创建段索引指针
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0) # 计算累加和填充seg_indptr[1:]

        return dataclasses.replace( # 使用dataclasses.replace创建新的批处理信息
            batch_info, # 基于原始批信息
            bs=num_segments, # 更新批大小为段数量
            num_segments=num_segments, # 更新段数量
            max_len=max(seg_lens_cpu), # 更新最大段长度
            seg_lens=seg_lens, # 更新段长度
            seg_indptr=seg_indptr, # 更新段索引指针
            weight_indices=torch.tensor(
                seg_weight_indices_cpu, dtype=torch.int32, device=self.device
            ), # 将段权重索引转为GPU张量
            expected_tokens=expected_tokens, # 设置预期token数
        )