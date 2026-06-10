# LM Head LoRA混合类实现
# 为LoRA后端提供LM Head相关的批量信息预计算功能，
# 支持logprobs分块处理时每个Pass使用独立的批量信息
from typing import List, Optional, Tuple  # 导入类型提示

from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.lora.utils import LoRABatchInfo, build_lm_head_pass_segments  # 导入LoRA工具函数
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批量信息类


class LoRABackendLmHeadMixing:  # LM Head LoRA混合类，为后端提供LM Head相关功能
    def init_lm_head_config(self):  # 初始化LM Head配置
        self.lm_head_batch_info = None  # LM Head批量信息
        # Precomputed per-pass lm_head batch_infos.  When the logits processor
        # calls lm_head in multiple passes (chunked logprobs), each pass gets
        # its own batch_info from this list.
        # 预计算的每Pass的lm_head batch_info。当logits处理器
        # 在多个Pass中调用lm_head（分块logprobs）时，每个Pass从
        # 此列表中获取自己的batch_info。
        self.lm_head_pass_batch_infos = None  # 逐Pass的LM Head批量信息列表
        # Current pass index.  When set, apply_lora uses
        # lm_head_pass_batch_infos[idx] instead of lm_head_batch_info.
        # 当前Pass索引。设置后，apply_lora使用
        # lm_head_pass_batch_infos[idx]而非lm_head_batch_info。
        self._lm_head_pass_idx = None  # 当前LM Head Pass索引

    def _get_lm_head_pass_segments(  # 获取LM Head LoRA logprobs分块的每Pass分段信息
        self,
        weight_indices: list[int],  # LoRA权重索引列表
        pruned_lens: List[int],  # 裁剪后的长度列表
    ) -> Optional[List[Tuple[List[int], List[int]]]]:  # 返回分段信息列表或None
        """Compute per-pass segment info for lm_head LoRA logprobs chunking.
        计算LM Head LoRA logprobs分块的每Pass分段信息。

        When LogitsProcessor splits pruned states into fixed-size passes,
        each pass needs its own segmentation so that lm_head LoRA operates
        on the correct adapter assignments.  This method returns the generic
        per-pass (seg_weight_indices, seg_lens) tuples; each backend is
        responsible for converting them into backend-specific LoRABatchInfo.
        当LogitsProcessor将裁剪后的状态分割为固定大小的Pass时，
        每个Pass需要自己的分段，以便lm_head LoRA在正确的适配器
        分配上操作。此方法返回通用的每Pass(分段权重索引, 分段长度)
        元组；每个后端负责将它们转换为后端特定的LoRABatchInfo。

        Returns None if logprobs chunking is disabled or the pruned token
        count does not exceed the logprobs chunk size.
        如果logprobs分块被禁用或裁剪后的token数不超过logprobs块大小，则返回None。
        """
        logprobs_chunk_size = envs.SGLANG_LOGITS_PROCESSER_CHUNK_SIZE.get()  # 获取logprobs块大小
        enable_logprobs_chunk = envs.SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK.get()  # 获取是否启用logprobs分块
        pruned_total = sum(pruned_lens)  # 计算裁剪后的总token数

        if not enable_logprobs_chunk or pruned_total <= logprobs_chunk_size:  # 如果未启用或token数不超过块大小
            return None  # 返回None

        return build_lm_head_pass_segments(  # 构建LM Head每Pass分段信息
            weight_indices, pruned_lens, logprobs_chunk_size  # 权重索引、裁剪长度和块大小
        )

    def _prepare_lm_head_batch_info(  # 准备当前前向批次的LM Head批量信息
        self,
        forward_batch: ForwardBatch,  # 当前前向批次
        weight_indices: list[int],  # LoRA权重索引列表
        batch_info: LoRABatchInfo,  # 当前批量信息
    ) -> Tuple[Optional[LoRABatchInfo], Optional[List[LoRABatchInfo]]]:  # 返回(LM Head批量信息, 逐Pass批量信息)
        """Prepare the lm_head batch info for the current forward batch.
        为当前前向批次准备lm_head批量信息。"""
        """It returns a tuple of (lm_head_batch_info, lm_head_pass_batch_infos).
        返回一个元组(lm_head_batch_info, lm_head_pass_batch_infos)。"""
        pass  # 由子类实现

    def _build_lm_head_batch_info(  # 构建裁剪后LM Head输入的LoRABatchInfo
        self,
        lm_head_segments: Tuple[List[int], List[int]],  # LM Head分段信息
        batch_info: LoRABatchInfo,  # 原始批量信息
        chunk_size: int,  # 块大小
        expected_tokens: int,  # 预期token数
    ) -> LoRABatchInfo:
        """Build a LoRABatchInfo for pruned lm_head input.
        为裁剪后的lm_head输入构建LoRABatchInfo。"""
        pass  # 由子类实现
