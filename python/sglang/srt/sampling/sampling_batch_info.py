# 本文件定义了采样批处理信息类 SamplingBatchInfo，用于在 SGLang 推理引擎中
# 管理一批请求的采样参数（如温度、top_p、top_k、min_p 等），并提供了批级别的
# 惩罚器（penalty）、自定义 logit 处理器、语法约束掩码、logit 偏置等功能。
# 支持批次的过滤（filter）、合并（merge）以及为前向传播准备副本等操作。

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

import sglang.srt.sampling.penaltylib as penaltylib
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SamplingBatchInfo:
    """采样批处理信息类，用于存储和管理一批请求的采样参数及相关的惩罚器和掩码。"""

    # Basic batched sampling params
    # 基础批量采样参数
    temperatures: torch.Tensor  # 温度张量，控制生成的随机性
    top_ps: torch.Tensor  # top-p（核采样）阈值张量
    top_ks: torch.Tensor  # top-k 采样阈值张量
    min_ps: torch.Tensor  # min-p 采样阈值张量

    # Whether all requests use greedy sampling
    # 是否所有请求都使用贪心采样（温度为0或top_k<=1）
    is_all_greedy: bool

    # Whether any requests use top_p sampling
    # 是否有请求使用 top_p 采样
    need_top_p_sampling: bool

    # Whether any requests use top_k sampling
    # 是否有请求使用 top_k 采样
    need_top_k_sampling: bool

    # Whether any request needs min_p sampling
    # 是否有请求使用 min_p 采样
    need_min_p_sampling: bool

    # Masking tensors for grammar-guided structured outputs
    # 语法引导结构化输出的掩码张量
    vocab_size: int  # 词表大小
    grammars: Optional[List] = None  # 语法约束列表
    rids_int: Optional[torch.Tensor] = None  # 请求ID的整数表示
    bootstrap_room_ids_int: Optional[torch.Tensor] = None  # 引导房间ID的整数表示
    vocab_mask: Optional[torch.Tensor] = None  # 词表掩码，用于限制生成词的范围
    apply_mask_func: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None  # 应用掩码的函数

    # Penalizer
    # 惩罚器编排器，管理频率/存在/重复等惩罚
    penalizer_orchestrator: Optional[penaltylib.BatchedPenalizerOrchestrator] = None
    acc_additive_penalties: Optional[torch.Tensor] = None  # Used in the overlap mode
    # 在重叠模式下累积的加法惩罚
    acc_scaling_penalties: Optional[torch.Tensor] = (
        None  # Used in the overlap mode for repetition penalty
        # 在重叠模式下累积的缩放惩罚（用于重复惩罚）
    )

    # Whether any request has custom logit processor
    # 是否有请求使用自定义 logit 处理器
    has_custom_logit_processor: bool = False
    # Custom parameters
    # 自定义参数列表
    custom_params: Optional[List[Optional[Dict[str, Any]]]] = None
    # Custom logit processor
    # 自定义 logit 处理器字典，键为处理器哈希，值为(处理器对象, 布尔掩码张量)
    custom_logit_processor: Optional[
        Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]
    ] = None

    # Used for deterministic sampling
    # 用于确定性采样的种子张量
    sampling_seed: Optional[torch.Tensor] = None

    # Device
    # 运行设备
    device: str = "cuda"

    # Handle logit bias
    # logit 偏置张量，用于调整特定 token 的生成概率
    logit_bias: Optional[torch.Tensor] = None

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
        """从调度批次对象构建 SamplingBatchInfo 实例。"""
        global_server_args = get_global_server_args()
        enable_deterministic = global_server_args.enable_deterministic_inference

        reqs = batch.reqs  # 获取批次中的所有请求
        device = batch.device  # 获取设备类型

        # 从请求中提取温度参数并构建张量，view(-1, 1)用于后续广播
        temperatures = torch.tensor(
            [r.sampling_params.temperature for r in reqs],
            dtype=torch.float,
            device=device,
        ).view(-1, 1)
        # 提取 top_p 参数
        top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs], dtype=torch.float, device=device
        )
        # 提取 top_k 参数
        top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs], dtype=torch.int32, device=device
        )
        # 提取 min_p 参数
        min_ps = torch.tensor(
            [r.sampling_params.min_p for r in reqs], dtype=torch.float, device=device
        )
        # 如果启用确定性推理，则提取采样种子
        sampling_seed = (
            torch.tensor(
                [
                    (
                        r.sampling_params.sampling_seed
                        if r.sampling_params.sampling_seed is not None
                        else 42  # 默认种子值为42
                    )
                    for r in reqs
                ],
                dtype=torch.int64,
                device=device,
            )
            if enable_deterministic
            else None
        )

        # 构建 logit_bias 张量：如果任何请求有 logit_bias，则创建全零张量并填充
        logit_bias = None
        if any(r.sampling_params.logit_bias is not None for r in reqs):
            logit_bias = torch.zeros(len(reqs), vocab_size, device=device)
            for i, r in enumerate(reqs):
                if r.sampling_params.logit_bias is not None:
                    for key, value in r.sampling_params.logit_bias.items():
                        logit_bias[i, int(key)] = value  # 将偏置值设置到对应 token 位置

        # Check if any request has custom logit processor
        # 检查是否有请求使用自定义 logit 处理器
        has_custom_logit_processor = (
            global_server_args.enable_custom_logit_processor
            and any(r.custom_logit_processor for r in reqs)  # check the flag first.
        )  # then check the requests.
        # 先检查全局开关，再检查各请求的标志

        if has_custom_logit_processor:
            # Merge the same type of custom logit processors together
            # 将相同类型的自定义 logit 处理器合并在一起
            processor_dict = {}
            for i, r in enumerate(reqs):
                if r.custom_logit_processor is None:
                    continue
                processor_str = r.custom_logit_processor
                if processor_str not in processor_dict:
                    processor_dict[processor_str] = []
                processor_dict[processor_str].append(i)  # 记录使用同一处理器的请求索引

            # 为每种处理器创建处理器对象和对应的布尔掩码张量
            merged_custom_logit_processor = {
                hash(processor_str): (
                    # The deserialized custom logit processor object
                    # 反序列化的自定义 logit 处理器对象
                    CustomLogitProcessor.from_str(processor_str),
                    # The mask tensor for the requests that use this custom logit processor
                    # 使用该处理器的请求对应的布尔掩码张量
                    torch.zeros(len(reqs), dtype=torch.bool)
                    .scatter_(0, torch.tensor(true_indices), True)
                    .to(device, non_blocking=True),
                )
                for processor_str, true_indices in processor_dict.items()
            }
            custom_params = [r.sampling_params.custom_params for r in reqs]
        else:
            merged_custom_logit_processor = None
            custom_params = None

        # Each penalizers will do nothing if they evaluate themselves as not required by looking at
        # the sampling_params of the requests (See {_is_required()} of each penalizers). So this
        # should not add hefty computation overhead other than simple checks.
        # 每个惩罚器会通过检查请求的采样参数自行判断是否需要激活，
        # 因此不会带来额外的计算开销。
        #
        # While we can choose not to even create the class instances if they are not required, this
        # could add additional complexity to the {ScheduleBatch} class, especially we need to
        # handle {filter_batch()} and {merge_batch()} cases as well.
        # 虽然可以在不需要时不创建实例，但这会增加 ScheduleBatch 类的复杂度，
        # 特别是需要处理 filter_batch() 和 merge_batch() 的情况。
        penalizer_orchestrator = penaltylib.BatchedPenalizerOrchestrator(
            vocab_size=vocab_size,
            batch=batch,
            penalizers={
                penaltylib.BatchedFrequencyPenalizer,  # 频率惩罚器
                penaltylib.BatchedMinNewTokensPenalizer,  # 最小新token惩罚器
                penaltylib.BatchedPresencePenalizer,  # 存在惩罚器
                penaltylib.BatchedRepetitionPenalizer,  # 重复惩罚器
            },
        )

        # 构建 SamplingBatchInfo 实例
        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            sampling_seed=sampling_seed,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),  # 所有请求都是贪心采样
            need_top_p_sampling=any(r.sampling_params.top_p != 1.0 for r in reqs),  # 有请求需要 top_p 采样
            need_top_k_sampling=any(r.sampling_params.top_k != TOP_K_ALL for r in reqs),  # 有请求需要 top_k 采样
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),  # 有请求需要 min_p 采样
            vocab_size=vocab_size,
            penalizer_orchestrator=penalizer_orchestrator,
            has_custom_logit_processor=has_custom_logit_processor,
            custom_params=custom_params,
            custom_logit_processor=merged_custom_logit_processor,
            device=device,
            logit_bias=logit_bias,
        )
        ret.adjusted_from_schedule_batch(batch, vocab_size)
        return ret

    # placeholder for override
    # 预留的子类重写方法，用于在构建后进行额外调整
    def adjusted_from_schedule_batch(self, batch: ScheduleBatch, vocab_size: int):
        pass

    # placeholder for override
    # 预留的子类重写方法，用于合并批次时的额外调整
    def adjusted_merge_batch(self, other: "SamplingBatchInfo"):
        pass

    # placeholder for override
    # 预留的子类重写方法，用于过滤批次时的额外调整
    def adjusted_filter_batch(
        self, keep_indices: List[int], keep_indices_device: torch.Tensor
    ):
        pass

    def __len__(self):
        """返回批次中的请求数量。"""
        return len(self.temperatures)

    def update_regex_vocab_mask(self):
        """更新语法约束的词表掩码，用于结构化输出（如 JSON/正则约束）。"""
        if not self.grammars:
            self.vocab_mask = None
            self.apply_mask_func = None
            return

        # Find a grammar from the list
        # 从列表中找到第一个有效的语法对象
        first_grammar = next(grammar for grammar in self.grammars if grammar)

        # TODO(lianmin): Maybe we can reuse the existing mask?
        # 分配词表掩码空间
        self.vocab_mask = first_grammar.allocate_vocab_mask(
            vocab_size=self.vocab_size,
            batch_size=len(self.temperatures),
            device=self.device,
        )
        self.apply_mask_func = (
            first_grammar.apply_vocab_mask
        )  # force to use static method
        # 强制使用静态方法来应用掩码

        # Apply the mask
        # 对每个语法约束，填充对应行的词表掩码
        for i, grammar in enumerate(self.grammars):
            if grammar and not grammar.finished and not grammar.is_terminated():
                grammar.fill_vocab_mask(self.vocab_mask, i)

        # Move the mask to the device if needed
        # 将掩码移动到目标设备（如GPU）
        self.vocab_mask = first_grammar.move_vocab_mask(self.vocab_mask, self.device)

    def update_penalties(self):
        """更新累积惩罚值（加法惩罚和缩放惩罚），用于重叠模式下的惩罚计算。"""
        if self.penalizer_orchestrator.is_required:
            # 初始化加法惩罚为零张量
            self.acc_additive_penalties = torch.zeros(
                (len(self.temperatures), self.vocab_size),
                dtype=torch.float32,
                device=self.temperatures.device,
            )
            # 累积加法惩罚（如频率惩罚、存在惩罚）
            self.penalizer_orchestrator.accumulate_additive_penalties(
                self.acc_additive_penalties
            )
            # 累积缩放惩罚（如重复惩罚）
            self.acc_scaling_penalties = (
                self.penalizer_orchestrator.accumulate_scaling_penalties()
            )
        else:
            self.acc_additive_penalties = None
            self.acc_scaling_penalties = None

    def apply_logits_bias(self, logits: torch.Tensor):
        """对 logits 应用各种偏置和惩罚，包括加法惩罚、缩放惩罚、语法掩码和 logit 偏置。"""
        if self.acc_additive_penalties is not None:
            # Used in the overlap mode
            # 在重叠模式下应用累积的加法惩罚
            logits.add_(self.acc_additive_penalties)

        if self.acc_scaling_penalties is not None:
            # Used in the overlap mode
            # 在重叠模式下应用累积的缩放惩罚
            apply_scaling_penalties(logits, self.acc_scaling_penalties)

        if self.penalizer_orchestrator and self.penalizer_orchestrator.is_required:
            # Used in the non-overlap mode
            # 在非重叠模式下直接通过惩罚器编排器应用惩罚
            self.penalizer_orchestrator.apply(logits)

        if self.vocab_mask is not None:
            # 应用语法约束的词表掩码，将不允许的 token logits 设为负无穷
            self.apply_mask_func(logits=logits, vocab_mask=self.vocab_mask)

        if self.logit_bias is not None:
            # 应用 logit 偏置，调整特定 token 的生成概率
            logits.add_(self.logit_bias)

    def filter_batch(self, keep_indices: List[int], keep_indices_device: torch.Tensor):
        """根据保留索引过滤批次，移除不需要的请求，保留指定的请求。"""
        # 过滤惩罚器编排器
        self.penalizer_orchestrator.filter(keep_indices_device)

        # 如果有自定义 logit 处理器，也需要过滤
        if self.has_custom_logit_processor:
            self._filter_batch_custom_logit_processor(keep_indices, keep_indices_device)

        # 过滤基础采样参数张量
        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "sampling_seed",
        ]:
            value = getattr(self, item, None)
            if value is not None:
                setattr(self, item, value[keep_indices_device])  # 按保留索引选择

        # 过滤 logit 偏置
        if self.logit_bias is not None:
            self.logit_bias = self.logit_bias[keep_indices_device]

        self.adjusted_filter_batch(keep_indices, keep_indices_device)

    def _filter_batch_custom_logit_processor(
        self, keep_indices: List[int], keep_indices_device: torch.Tensor
    ):
        """Filter the custom logit processor and custom params"""
        # 过滤自定义 logit 处理器和自定义参数
        self.custom_logit_processor = {
            k: (p, mask[keep_indices_device])
            for k, (p, mask) in self.custom_logit_processor.items()
            if torch.any(
                mask[keep_indices_device]
            )  # ignore the custom logit processor whose mask is all False
            # 忽略掩码全为 False 的处理器（即过滤后没有请求使用该处理器）
        }
        self.custom_params = [self.custom_params[i] for i in keep_indices]

        # If the custom logit processor is an empty dict, set the flag to False,
        # and set the custom logit processor and custom params to None.
        # 如果过滤后没有自定义处理器，重置相关标志和字段
        if len(self.custom_logit_processor) == 0:
            self.custom_logit_processor = None
            self.custom_params = None
            self.has_custom_logit_processor = False

    @staticmethod
    def merge_custom_logit_processor(
        lhs: Optional[Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]],
        rhs: Optional[Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]],
        bs1: int,
        bs2: int,
        device: str,
    ):
        """合并两个批次的自定义 logit 处理器，包括处理器对象和对应的掩码张量。"""
        if lhs is None and rhs is None:
            return None
        lhs, rhs = lhs or {}, rhs or {}

        # 获取两个字典中所有处理器的键的并集
        keys = set(lhs.keys()).union(set(rhs.keys()))
        merged_dict = {}

        for k in keys:
            # Get the logit processor object
            # 获取 logit 处理器对象（同一个哈希对应同一个处理器）
            processor = lhs[k][0] if k in lhs else rhs[k][0]
            # Get and merge the mask tensors from the two dicts
            # 获取并合并两个字典中的掩码张量
            left_mask = (
                lhs[k][1]
                if k in lhs
                else torch.zeros(bs1, dtype=torch.bool, device=device)  # 左侧没有则用全零掩码
            )
            right_mask = (
                rhs[k][1]
                if k in rhs
                else torch.zeros(bs2, dtype=torch.bool, device=device)  # 右侧没有则用全零掩码
            )
            # 拼接左右掩码
            merged_dict[k] = (processor, torch.cat([left_mask, right_mask]))

            # 断言合并后的掩码长度等于两个批次大小之和
            assert merged_dict[k][1].shape[0] == bs1 + bs2, (
                f"The batch size of merged mask ({merged_dict[k][1].shape[0]}) does not match "
                f"the sum of the batch sizes of the two masks ({bs1 + bs2})"
                f"\n{left_mask=}\n{right_mask=}\n{bs1=}\n{bs2=}"
                f"\n{lhs=}\n{rhs=}"
            )

        return merged_dict

    def merge_batch(self, other: "SamplingBatchInfo"):
        """将另一个 SamplingBatchInfo 合并到当前实例中，用于批处理请求的合并。"""
        # 合并惩罚器编排器
        self.penalizer_orchestrator.merge(other.penalizer_orchestrator)

        # Merge the custom logit processors and custom params lists
        # 合并自定义 logit 处理器和自定义参数列表
        if self.has_custom_logit_processor or other.has_custom_logit_processor:
            # Merge the custom logit processors
            # 合并自定义 logit 处理器
            self.custom_logit_processor = (
                SamplingBatchInfo.merge_custom_logit_processor(
                    self.custom_logit_processor,
                    other.custom_logit_processor,
                    len(self),
                    len(other),
                    self.device,
                )
            )
            # Merge the custom params lists
            # 合并自定义参数列表，没有的用 None 填充
            self.custom_params = self.custom_params or [None] * len(self)
            other.custom_params = other.custom_params or [None] * len(other)
            self.custom_params.extend(other.custom_params)

            # Set the flag to True if any of the two has custom logit processor
            # 只要任一批次有自定义处理器，标志就设为 True
            self.has_custom_logit_processor = True

        # Merge logit bias - note this has to come before the temperatures tensor update! Otherwise will cause crashes.
        # 合并 logit 偏置 - 注意：必须在温度张量更新之前执行，否则会导致崩溃。
        # See note below on len(self) and len(other).
        self.logit_bias = merge_bias_tensor(
            self.logit_bias, other.logit_bias, len(self), len(other), self.device, 0.0
        )

        # Note: because the __len()__ operator is defined on the temperatures tensor,
        # please make sure any merge operation with len(self) or len(other) is done before
        # the merge operation of the temperatures tensor below.
        # 注意：由于 __len__() 基于 temperatures 张量定义，
        # 必须在合并 temperatures 之前完成所有依赖 len(self)/len(other) 的操作。
        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "sampling_seed",
        ]:
            self_val = getattr(self, item, None)
            other_val = getattr(other, item, None)
            if self_val is not None and other_val is not None:
                setattr(self, item, torch.cat([self_val, other_val]))  # 拼接张量

        # 合并布尔标志：贪心用与，采样需求用或
        self.is_all_greedy &= other.is_all_greedy  # 只有全部贪心才是贪心
        self.need_top_p_sampling |= other.need_top_p_sampling  # 有任一需要即可
        self.need_top_k_sampling |= other.need_top_k_sampling
        self.need_min_p_sampling |= other.need_min_p_sampling

        self.adjusted_merge_batch(other)

    def copy_for_forward(self):
        """为前向传播创建副本，预先计算惩罚值并移除惩罚器编排器的依赖。"""
        # Accumulate the penalty into a pre-allocated buffer to get rid of the dependency of `penalizer_orchestrator` later
        # 预先累积惩罚到缓冲区，以便后续不再依赖 penalizer_orchestrator
        self.update_penalties()
        return dataclasses.replace(self, penalizer_orchestrator=None)  # 创建副本并移除编排器


def merge_bias_tensor(
    lhs: Optional[torch.Tensor],
    rhs: Optional[torch.Tensor],
    bs1: int,
    bs2: int,
    device: str,
    default: float,
):
    """Merge two bias tensors for batch merging.
    合并两个偏置张量，用于批次合并操作。

    Args:
        lhs: Left-hand side tensor
            左侧偏置张量
        rhs: Right-hand side tensor
            右侧偏置张量
        bs1: Batch size of left-hand side tensor
            左侧张量的批次大小
        bs2: Batch size of right-hand side tensor
            右侧张量的批次大小
        device: Device to place the merged tensor on
            合并张量放置的设备
        default: Default value for missing tensor elements
            缺失张量元素的默认填充值

    Returns:
        Merged tensor or None if both inputs are None
        合并后的张量，如果两者都为 None 则返回 None
    """
    if lhs is None and rhs is None:
        return None

    if lhs is not None and rhs is not None:
        # 两侧都有偏置张量，直接拼接
        return torch.cat([lhs, rhs])
    else:
        # 只有一侧有偏置张量，需要为另一侧创建默认值填充的张量
        if lhs is not None:
            shape, dtype = lhs.shape[1:], lhs.dtype  # 获取非词表维度的形状和数据类型
        else:
            shape, dtype = rhs.shape[1:], rhs.dtype

        if lhs is None:
            # 为左侧创建用默认值填充的张量
            lhs = torch.empty((bs1, *shape), device=device, dtype=dtype).fill_(default)
        if rhs is None:
            # 为右侧创建用默认值填充的张量
            rhs = torch.empty((bs2, *shape), device=device, dtype=dtype).fill_(default)
        return torch.cat([lhs, rhs])  # 拼接并返回
