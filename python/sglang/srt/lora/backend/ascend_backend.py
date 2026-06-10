# 昇腾(NPU)平台上的LoRA后端实现
# 提供基于华为Ascend NPU的LoRA矩阵运算，包括shrink/expand sgmv操作
# 以及QKV和Gate-Up投影的分段LoRA计算
import torch  # 导入PyTorch库

from sglang.srt.lora.backend.base_backend import BaseLoRABackend  # 导入LoRA后端基类
from sglang.srt.lora.utils import LoRABatchInfo  # 导入LoRA批量信息数据类
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批量信息类
from sglang.srt.utils import is_npu  # 导入NPU环境检测工具函数

if is_npu():  # 如果当前运行环境为NPU
    import sgl_kernel_npu  # noqa: F401  # 导入SGL NPU算子库
    import torch_npu  # noqa: F401  # 导入PyTorch NPU后端库


class AscendLoRABackend(BaseLoRABackend):  # 昇腾LoRA后端类，继承自基类
    name = "ascend"  # 后端名称标识

    def __init__(  # 初始化方法
        self,
        max_loras_per_batch: int,  # 每批次最大LoRA数量
        device: torch.device,  # 运行设备
        **kwargs,  # 其他关键字参数
    ):
        super().__init__(max_loras_per_batch, device)  # 调用父类初始化

    def run_lora_a_sgemm(  # 运行LoRA A矩阵的分段GEMM（shrink阶段）
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs  # 输入张量和权重
    ) -> torch.Tensor:  # 返回输出张量

        total_seq_len, _ = x.shape  # 获取总序列长度
        _, weight_out_dim, _ = weights.shape  # 获取权重输出维度

        output_tensor = torch.zeros(  # 初始化输出张量为零
            (total_seq_len, weight_out_dim), dtype=x.dtype, device=x.device  # 形状为(序列长度, 输出维度)
        )
        torch.ops.npu.sgmv_shrink(  # 调用NPU的sgmv shrink算子
            x,  # 输入张量
            weights,  # LoRA A权重
            self.batch_info.weight_indices,  # 权重索引
            self.batch_info.seg_lens,  # 分段长度
            output_tensor,  # 输出张量
            1.0,  # 缩放因子
        )
        scaling = (  # 计算缩放因子
            self.batch_info.scalings.gather(0, self.batch_info.weight_indices)  # 根据权重索引收集缩放值
            .repeat_interleave(self.batch_info.seg_lens, output_size=total_seq_len)  # 按分段长度重复
            .unsqueeze(-1)  # 在最后增加一个维度以便广播
        )
        output_tensor *= scaling  # 将缩放因子应用到输出

        return output_tensor  # 返回缩放后的输出

    def run_lora_b_sgemm(  # 运行LoRA B矩阵的分段GEMM（expand阶段）
        self,
        x: torch.Tensor,  # 输入张量（LoRA A的输出）
        weights: torch.Tensor,  # LoRA B权重
        base_output: torch.Tensor = None,  # 基础模型输出（可选，用于累加）
        *args,
        **kwargs,
    ) -> torch.Tensor:
        total_seq_len, _ = x.shape  # 获取总序列长度
        _, weight_out_dim, _ = weights.shape  # 获取权重输出维度

        if base_output is None:  # 如果没有提供基础输出
            output_tensor = torch.zeros(  # 创建零张量作为输出
                (total_seq_len, weight_out_dim), device=x.device, dtype=x.dtype  # 形状为(序列长度, 输出维度)
            )
        else:  # 如果提供了基础输出
            output_tensor = base_output  # 直接在基础输出上累加

        torch.ops.npu.sgmv_expand(  # 调用NPU的sgmv expand算子
            x,  # 输入张量
            weights,  # LoRA B权重
            self.batch_info.weight_indices,  # 权重索引
            self.batch_info.seg_lens,  # 分段长度
            output_tensor,  # 输出张量
            0,  # 起始偏移
            weight_out_dim,  # 输出维度大小
        )

        return output_tensor  # 返回输出张量

    def run_qkv_lora(  # 运行QKV投影的LoRA计算
        self,
        x: torch.Tensor,  # 输入张量
        qkv_lora_a: torch.Tensor,  # QKV的LoRA A权重
        qkv_lora_b: torch.Tensor,  # QKV的LoRA B权重
        output_offset: torch.Tensor,  # 输出偏移量
        output_offset_cpu: torch.Tensor,  # CPU上的输出偏移量
        max_qkv_out_dim: int,  # QKV最大输出维度
        base_output: torch.Tensor = None,  # 基础模型输出（可选）
        n_slices: int = 3,  # 切片数量（默认3，对应Q/K/V）
        *args,
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(qkv_lora_b, torch.Tensor)  # 确保qkv_lora_b是张量类型

        total_seq_len, _ = x.shape  # 获取总序列长度
        _, weight_intermediate_dim, _ = qkv_lora_a.shape  # 获取A权重的中间维度
        _, weight_out_dim, _ = qkv_lora_b.shape  # 获取B权重的输出维度
        max_rank = weight_intermediate_dim // n_slices  # 计算每个切片的最大秩

        if base_output is None:  # 如果没有提供基础输出
            output_tensor = torch.zeros(  # 创建零张量作为输出
                (total_seq_len, weight_out_dim), device=x.device, dtype=x.dtype  # 形状为(序列长度, 输出维度)
            )
        else:  # 如果提供了基础输出
            output_tensor = base_output  # 直接在基础输出上累加

        lora_a_output = torch.zeros(  # 初始化LoRA A的输出
            total_seq_len, weight_intermediate_dim, dtype=x.dtype, device=x.device  # 形状为(序列长度, 中间维度)
        )
        torch.ops.npu.sgmv_shrink(  # 调用NPU的sgmv shrink算子计算LoRA A
            x,  # 输入张量
            qkv_lora_a,  # QKV LoRA A权重
            self.batch_info.weight_indices,  # 权重索引
            self.batch_info.seg_lens,  # 分段长度
            lora_a_output,  # LoRA A输出
            1.0,  # 缩放因子
        )

        scaling = (  # 计算缩放因子
            self.batch_info.scalings.gather(0, self.batch_info.weight_indices)  # 根据权重索引收集缩放值
            .repeat_interleave(self.batch_info.seg_lens, output_size=total_seq_len)  # 按分段长度重复
            .unsqueeze(-1)  # 在最后增加一个维度以便广播
        )
        lora_a_output *= scaling  # 将缩放因子应用到LoRA A输出

        for slice_id in range(n_slices):  # 遍历每个切片（Q/K/V）
            slice_offset = output_offset_cpu[slice_id]  # 获取当前切片的输出偏移
            slice_offset_next = output_offset_cpu[slice_id + 1]  # 获取下一个切片的输出偏移
            slice_size = slice_offset_next - slice_offset  # 计算当前切片的大小
            torch.ops.npu.sgmv_expand(  # 调用NPU的sgmv expand算子计算LoRA B
                lora_a_output[:, (max_rank * slice_id) : (max_rank * (slice_id + 1))],  # 当前切片的LoRA A输出
                qkv_lora_b[:, slice_offset:slice_offset_next],  # 当前切片的LoRA B权重
                self.batch_info.weight_indices,  # 权重索引
                self.batch_info.seg_lens,  # 分段长度
                output_tensor,  # 输出张量
                slice_offset,  # 起始偏移
                slice_size,  # 切片大小
            )

        return output_tensor  # 返回输出张量

    def run_gate_up_lora(  # 运行Gate-Up投影的LoRA计算
        self,
        x: torch.Tensor,  # 输入张量
        gate_up_lora_a: torch.Tensor,  # Gate-Up的LoRA A权重
        gate_up_lora_b: torch.Tensor,  # Gate-Up的LoRA B权重
        base_output: torch.Tensor = None,  # 基础模型输出（可选）
        *args,
        **kwargs,
    ) -> torch.Tensor:

        num_slices = 2  # Gate-Up有2个切片（gate和up）
        assert isinstance(gate_up_lora_b, torch.Tensor)  # 确保gate_up_lora_b是张量类型

        total_seq_len, _ = x.shape  # 获取总序列长度
        _, weight_intermediate_dim, _ = gate_up_lora_a.shape  # 获取A权重的中间维度
        _, weight_out_dim, _ = gate_up_lora_b.shape  # 获取B权重的输出维度
        slice_size = weight_out_dim // num_slices  # 计算每个切片的大小
        max_rank = weight_intermediate_dim // num_slices  # 计算每个切片的最大秩

        if base_output is None:  # 如果没有提供基础输出
            output_tensor = torch.zeros(  # 创建零张量作为输出
                (total_seq_len, weight_out_dim), device=x.device, dtype=x.dtype  # 形状为(序列长度, 输出维度)
            )
        else:  # 如果提供了基础输出
            output_tensor = base_output  # 直接在基础输出上累加

        lora_a_output = torch.zeros(  # 初始化LoRA A的输出
            total_seq_len, weight_intermediate_dim, dtype=x.dtype, device=x.device  # 形状为(序列长度, 中间维度)
        )

        torch.ops.npu.sgmv_shrink(  # 调用NPU的sgmv shrink算子计算LoRA A
            x,  # 输入张量
            gate_up_lora_a,  # Gate-Up LoRA A权重
            self.batch_info.weight_indices,  # 权重索引
            self.batch_info.seg_lens,  # 分段长度
            lora_a_output,  # LoRA A输出
            1.0,  # 缩放因子
        )

        scaling = (  # 计算缩放因子
            self.batch_info.scalings.gather(0, self.batch_info.weight_indices)  # 根据权重索引收集缩放值
            .repeat_interleave(self.batch_info.seg_lens, output_size=total_seq_len)  # 按分段长度重复
            .unsqueeze(-1)  # 在最后增加一个维度以便广播
        )
        lora_a_output *= scaling  # 将缩放因子应用到LoRA A输出

        slice_offset = 0  # 初始化切片偏移为0
        for slice_id in range(num_slices):  # 遍历每个切片（gate和up）
            torch.ops.npu.sgmv_expand(  # 调用NPU的sgmv expand算子计算LoRA B
                lora_a_output[:, (max_rank * slice_id) : (max_rank * (slice_id + 1))],  # 当前切片的LoRA A输出
                gate_up_lora_b[:, slice_offset : slice_offset + slice_size],  # 当前切片的LoRA B权重
                self.batch_info.weight_indices,  # 权重索引
                self.batch_info.seg_lens,  # 分段长度
                output_tensor,  # 输出张量
                slice_offset,  # 起始偏移
                slice_size,  # 切片大小
            )
            slice_offset += slice_size  # 更新切片偏移

        return output_tensor  # 返回输出张量

    def init_cuda_graph_batch_info(  # 初始化NPU图模式的批量信息（与CUDA Graph类似）
        self,
        max_bs_in_cuda_graph: int,  # 图模式下的最大批量大小
        num_tokens_per_bs: int,  # 每个序列的token数量
    ):
        with torch.device("npu"):  # 在NPU设备上下文中创建张量
            self.npu_graph_batch_info = LoRABatchInfo(  # 创建NPU图的批量信息
                bs=max_bs_in_cuda_graph,  # 批量大小
                use_cuda_graph=True,  # 启用图模式
                num_segments=None,  # 分段数量（每批次设置）
                seg_lens=torch.full(  # 分段长度张量
                    (max_bs_in_cuda_graph,), num_tokens_per_bs, dtype=torch.int32  # 每个序列的token数
                ),
                seg_indptr=torch.empty(max_bs_in_cuda_graph + 1, dtype=torch.int32),  # 分段索引指针
                max_len=num_tokens_per_bs,  # 最大序列长度
                weight_indices=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32),  # 权重索引
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32),  # LoRA秩
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float),  # 缩放因子
                permutation=None,  # 排列索引（NPU后端不需要）
            )

            # Initialize seg_indptr for NPU graph as they remain constant
            # across batches.
            # 初始化NPU图的seg_indptr，因为它们在不同批次间保持不变
            torch.cumsum(  # 计算分段长度的累积和
                self.npu_graph_batch_info.seg_lens[:max_bs_in_cuda_graph],  # 取前max_bs个分段长度
                dim=0,  # 沿第0维累加
                out=self.npu_graph_batch_info.seg_indptr[1 : max_bs_in_cuda_graph + 1],  # 输出到seg_indptr[1:]
            )

    def prepare_lora_batch(  # 准备当前前向批次的LoRA权重和批量信息
        self,
        forward_batch: ForwardBatch,  # 当前前向批次
        weight_indices: list[int],  # LoRA权重索引列表
        lora_ranks: list[int],  # LoRA秩列表
        scalings: list[float],  # 缩放因子列表
        use_cuda_graph: bool,  # 是否使用CUDA/NPU图模式
    ):
        # Use pinned memory to avoid synchronizations during host-to-device transfer
        # 使用固定内存以避免主机到设备传输时的同步等待
        weight_indices_tensor = torch.tensor(  # 创建权重索引张量
            weight_indices, dtype=torch.int32, pin_memory=True, device="cpu"  # 使用固定内存
        )
        lora_ranks_tensor = torch.tensor(  # 创建LoRA秩张量
            lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"  # 使用固定内存
        )
        scalings_tensor = torch.tensor(  # 创建缩放因子张量
            scalings, dtype=torch.float, pin_memory=True, device="cpu"  # 使用固定内存
        )

        bs = forward_batch.batch_size  # 获取批次大小

        if use_cuda_graph:  # 如果使用图模式
            assert (
                self.npu_graph_batch_info is not None
            ), "NPU Graph batch info is not initialized."  # 确保NPU图批量信息已初始化
            batch_info = self.npu_graph_batch_info  # 使用预分配的NPU图批量信息
            batch_info.bs = forward_batch.batch_size  # 更新批次大小
            batch_info.num_segments = forward_batch.batch_size  # 更新分段数量
        else:  # 非图模式
            max_len = (  # 计算最大序列长度
                # Calculate max_len from the CPU copy to avoid D2H transfer.
                # 从CPU副本计算max_len以避免设备到主机的传输
                max(forward_batch.extend_seq_lens_cpu)  # 扩展模式取最大扩展长度
                if forward_batch.forward_mode.is_extend()  # 如果是扩展模式
                else 1  # 解码模式每个序列只有1个token
            )
            seg_lens = (  # 计算分段长度
                forward_batch.extend_seq_lens  # 扩展模式使用扩展序列长度
                if forward_batch.forward_mode.is_extend()  # 如果是扩展模式
                else torch.ones(bs, dtype=torch.int32, device=self.device)  # 解码模式每段长度为1
            )
            seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)  # 初始化分段索引指针
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)  # 计算累积和作为索引指针

            batch_info = LoRABatchInfo(  # 创建新的批量信息
                bs=forward_batch.batch_size,  # 批次大小
                num_segments=forward_batch.batch_size,  # 分段数量
                max_len=max_len,  # 最大序列长度
                use_cuda_graph=False,  # 不使用图模式
                seg_lens=seg_lens,  # 分段长度
                seg_indptr=seg_indptr,  # 分段索引指针
                weight_indices=torch.empty(  # 权重索引张量
                    (bs,), dtype=torch.int32, device=self.device  # 形状为(批次大小,)
                ),
                lora_ranks=torch.empty(  # LoRA秩张量
                    (self.max_loras_per_batch,), dtype=torch.int32, device=self.device  # 形状为(最大LoRA数,)
                ),
                scalings=torch.empty(  # 缩放因子张量
                    (self.max_loras_per_batch,), dtype=torch.float, device=self.device  # 形状为(最大LoRA数,)
                ),
                permutation=None,  # 排列索引（NPU后端不需要）
            )

        # Copy to device asynchronously
        # 异步复制到设备
        batch_info.lora_ranks[: self.max_loras_per_batch].copy_(  # 复制LoRA秩到设备
            lora_ranks_tensor, non_blocking=True  # 非阻塞复制
        )
        batch_info.scalings[: self.max_loras_per_batch].copy_(  # 复制缩放因子到设备
            scalings_tensor, non_blocking=True  # 非阻塞复制
        )
        batch_info.weight_indices[:bs].copy_(weight_indices_tensor, non_blocking=True)  # 复制权重索引到设备（非阻塞）

        batch_info = self._add_moe_lora_info(forward_batch, batch_info)  # 添加MoE LoRA信息
        self.batch_info = batch_info  # 保存当前批次信息
