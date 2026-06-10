# 该文件实现了批处理惩罚编排器及惩罚器抽象基类
# 编排器负责管理多个惩罚器的生命周期，协调它们的准备、累积、应用、过滤和合并操作
# 支持加性惩罚和乘性惩罚两种类型，并提供推测解码下的惩罚扩展机制

from __future__ import annotations  # 启用延迟注解评估

import abc  # 抽象基类模块
import weakref  # 弱引用模块
from typing import TYPE_CHECKING, Optional, Set, Type  # 类型注解

import torch  # PyTorch张量库

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.managers.schedule_batch import ScheduleBatch  # 调度批次类


class BatchedPenalizerOrchestrator:  # 批处理惩罚编排器
    def __init__(
        self,
        vocab_size: int,  # 词表大小
        batch: ScheduleBatch,  # 调度批次
        penalizers: Set[Type["_BatchedPenalizer"]],  # 惩罚器类型集合
    ):  # 初始化编排器
        """初始化惩罚编排器，创建并准备所有惩罚器"""
        self.vocab_size = vocab_size  # 词表大小
        self._batch_ref = weakref.ref(batch)  # 使用弱引用引用批次对象
        self.device = batch.device  # 设备
        self.penalizers = {Penalizer: Penalizer(self) for Penalizer in penalizers}  # 实例化所有惩罚器

        is_required = False  # 是否需要惩罚的标志
        for penalizer in self.penalizers.values():  # 遍历所有惩罚器
            pen_is_required = penalizer.prepare_if_required()  # 检查并准备惩罚器
            is_required |= pen_is_required  # 更新需要标志
        self.is_required = is_required  # 保存是否需要惩罚

    @property
    def batch(self) -> ScheduleBatch | None:  # 获取当前批次
        """通过弱引用获取当前调度批次"""
        return self._batch_ref()  # 返回批次对象

    @batch.setter
    def batch(self, value: Optional[ScheduleBatch]):  # 设置当前批次
        """设置调度批次，使用弱引用避免循环引用"""
        if value is None:  # 如果值为None
            self._batch_ref = lambda: None  # 设置返回None的lambda
        else:
            self._batch_ref = weakref.ref(value)  # 使用弱引用引用新批次

    def reqs(self):  # 获取当前批次的请求列表
        """获取当前批次的请求列表"""
        return self.batch.reqs  # 返回请求列表

    def cumulate_output_tokens(self, output_ids: torch.Tensor):  # 累计输出token
        """
        Feed the output tokens to the penalizers.

        Args:
            output_ids (torch.Tensor): The output tokens.
        """  # 将输出token传递给所有惩罚器进行累计
        for penalizer in self.penalizers.values():  # 遍历所有惩罚器
            penalizer.cumulate_output_tokens(output_ids=output_ids)  # 传递输出token

    def apply(self, logits: torch.Tensor, repeat: Optional[int] = None):  # 应用所有惩罚器到logits
        """
        Apply all penalizers to the logits in-place.

        Args:
            logits: The logits tensor to apply penalties to.
            repeat: If set (speculative decoding), per-request penalties are
                expanded via repeat_interleave to match the draft token layout.
                Additive penalties are captured into a zeros tensor, expanded,
                then added; scaling penalties are accumulated, expanded, then
                applied directly.
        """  # 将所有惩罚器应用到logits上（原地修改）
        if repeat is None:  # 如果不需要重复（非推测解码）
            for penalizer in self.penalizers.values():  # 遍历所有惩罚器
                penalizer.apply(logits)  # 直接应用惩罚器
        else:
            # Additive: capture into zeros, expand, add  # 加性惩罚：捕获到零张量，扩展，相加
            bs = logits.shape[0] // repeat  # 计算实际批次大小
            additive = torch.zeros(
                (bs, logits.shape[1]), dtype=torch.float32, device=logits.device  # 创建零张量
            )
            self.accumulate_additive_penalties(additive)  # 累积加性惩罚
            logits.add_(torch.repeat_interleave(additive, repeat, dim=0))  # 扩展后加到logits上
            # Scaling: accumulate, expand, apply  # 乘性惩罚：累积，扩展，应用
            accumulated = self.accumulate_scaling_penalties()  # 累积乘性惩罚
            if accumulated is not None:  # 如果有乘性惩罚
                from sglang.srt.sampling.penaltylib.repetition_penalty import (
                    apply_scaling_penalties,  # 导入缩放惩罚应用函数
                )

                expanded = torch.repeat_interleave(accumulated, repeat, dim=0)  # 扩展乘性惩罚
                apply_scaling_penalties(logits, expanded)  # 应用缩放惩罚

    def accumulate_additive_penalties(self, logits: torch.Tensor):  # 仅累积加性惩罚
        """Apply only additive (non-multiplicative) penalizers."""  # 仅应用加性（非乘性）惩罚器
        for penalizer in self.penalizers.values():  # 遍历所有惩罚器
            if not penalizer.is_multiplicative:  # 如果是加性惩罚器
                penalizer.apply(logits)  # 应用该惩罚器

    def accumulate_scaling_penalties(self) -> Optional[torch.Tensor]:  # 累积乘性惩罚
        """Accumulate all multiplicative penalty tensors into one, or None if none active."""  # 将所有乘性惩罚张量累积为一个，若无活跃的则返回None
        result = None  # 结果张量
        for penalizer in self.penalizers.values():  # 遍历所有惩罚器
            if not penalizer._is_prepared or not penalizer.is_multiplicative:  # 如果未准备或非乘性
                continue  # 跳过
            if result is None:  # 如果结果为空
                result = penalizer.get_scaling_penalties().clone()  # 克隆第一个乘性惩罚
            else:
                result *= penalizer.get_scaling_penalties()  # 乘以后续的乘性惩罚
        return result  # 返回累积结果

    def filter(self, keep_indices: torch.Tensor):  # 根据保留索引过滤惩罚器
        """
        Filter the penalizers based on the indices to keep in the batch.

        Args:
            keep_indices (torch.Tensor): Tensor of indices to keep in the batch.
        """  # 根据批次中保留的索引过滤惩罚器
        if not self.is_required:  # 如果不需要惩罚
            return  # 直接返回

        if len(keep_indices) == 0:  # 如果没有保留的索引
            # No requests left in the batch, fully release orchestrator resources  # 批次中没有剩余请求，完全释放编排器资源
            self.release()  # 释放资源
            return

        is_required = False  # 是否需要的标志
        for penalizer in self.penalizers.values():  # 遍历所有惩罚器
            tmp_is_required = penalizer.is_required()  # 检查惩罚器是否仍然需要
            is_required |= tmp_is_required  # 更新标志
            if tmp_is_required:  # 如果仍然需要
                penalizer.filter(keep_indices=keep_indices)  # 过滤惩罚器
            else:
                penalizer.teardown()  # 拆卸不需要的惩罚器
        self.is_required = is_required  # 更新是否需要的标志

    # Resource management helpers  # 资源管理辅助方法
    def release(self) -> None:  # 释放所有惩罚器资源
        """Release all penalizers and break references so GC can reclaim promptly."""  # 释放所有惩罚器并断开引用以便GC及时回收
        for penalizer in self.penalizers.values():  # 遍历所有惩罚器
            penalizer.teardown()  # 拆卸惩罚器
        self.penalizers.clear()  # 清空惩罚器字典
        # Break reference to ScheduleBatch  # 断开对ScheduleBatch的引用
        self._batch_ref = None  # 清空弱引用
        self.is_required = False  # 标记为不需要

    # Context manager support  # 上下文管理器支持
    def __enter__(self) -> "BatchedPenalizerOrchestrator":  # 进入上下文
        """进入上下文管理器"""
        return self  # 返回自身

    def __exit__(self, exc_type, exc, tb) -> None:  # 退出上下文
        """退出上下文管理器时释放资源"""
        self.release()  # 释放资源

    def merge(self, their: "BatchedPenalizerOrchestrator"):  # 合并另一个编排器
        """
        Merge the penalizers of another orchestrator into this one.

        Note that this function **must** be called _before_ self.batch.reqs is updated (filtered).
        Each unprepared penalizers would have to be prepared (creating tensors, etc.) first before merging.
        This step requires the original batch.reqs, before it gets merged with other batch.reqs.

        Args:
            their (BatchedPenalizerOrchestrator): The orchestrator to merge into this one.
        """  # 将另一个编排器的惩罚器合并到当前编排器
        if not self.is_required and not their.is_required:  # 如果两者都不需要惩罚
            return  # 直接返回

        self.is_required = True  # 标记为需要惩罚
        for penalizer, their_penalizer in their.penalizers.items():  # 遍历另一个编排器的惩罚器
            self.penalizers[penalizer].merge(their_penalizer)  # 合并惩罚器


class _BatchedPenalizer(abc.ABC):  # 批处理惩罚器抽象基类
    """
    An abstract class for a batched penalizer.
    """  # 批处理惩罚器的抽象类

    is_multiplicative: bool = False  # 是否为乘性惩罚，默认为加性

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):  # 初始化惩罚器
        """初始化惩罚器，保存编排器的弱引用"""
        self._orchestrator_ref: weakref.ReferenceType[BatchedPenalizerOrchestrator] = (
            weakref.ref(orchestrator)  # 使用弱引用引用编排器
        )
        self._is_prepared = False  # 是否已准备标志

    @property
    def orchestrator(self) -> BatchedPenalizerOrchestrator:  # 获取编排器
        """通过弱引用获取编排器实例"""
        orch: Optional[BatchedPenalizerOrchestrator] = self._orchestrator_ref()  # 获取弱引用对象
        # This should never happen, but we need to handle it gracefully  # 这不应该发生，但需要优雅处理
        if orch is None:  # 如果编排器已被垃圾回收
            raise RuntimeError(
                "BatchedPenalizerOrchestrator has been garbage-collected"  # 抛出运行时错误
            )
        return orch  # 返回编排器

    def is_prepared(self) -> bool:  # 检查是否已准备
        """检查惩罚器是否已准备"""
        return self._is_prepared  # 返回准备状态

    def is_required(self) -> bool:  # 检查是否需要
        """检查惩罚器是否需要被使用"""
        return self._is_required()  # 调用子类实现

    def prepare(self):  # 准备惩罚器
        """准备惩罚器，初始化所需张量"""
        if not self._is_prepared:  # 如果未准备
            self._prepare()  # 调用子类的准备方法
            self._is_prepared = True  # 标记为已准备

    def prepare_if_required(self):  # 如果需要则准备
        """如果需要则准备惩罚器，返回是否需要"""
        if self._is_required():  # 如果需要
            self.prepare()  # 准备惩罚器
            return True  # 返回True
        else:
            return False  # 返回False

    def teardown(self):  # 拆卸惩罚器
        """拆卸惩罚器，释放资源"""
        self._teardown()  # 调用子类的拆卸方法
        self._is_prepared = False  # 标记为未准备

    def cumulate_output_tokens(self, output_ids: torch.Tensor):  # 累计输出token
        """将输出token传递给惩罚器进行累计"""
        if not self._is_prepared:  # 如果未准备
            return  # 直接返回

        self._cumulate_output_tokens(output_ids=output_ids)  # 调用子类的累计方法

    def apply(self, logits: torch.Tensor) -> torch.Tensor:  # 应用惩罚
        """将惩罚应用到logits上"""
        if not self._is_prepared:  # 如果未准备
            return  # 直接返回

        self._apply(logits=logits)  # 调用子类的应用方法

    def filter(self, keep_indices: torch.Tensor):  # 过滤惩罚器
        """根据保留索引过滤惩罚器的内部张量"""
        if not self._is_prepared:  # 如果未准备
            return  # 直接返回

        self._filter(keep_indices=keep_indices)  # 调用子类的过滤方法

    def merge(self, their: "_BatchedPenalizer"):  # 合并另一个惩罚器
        """将另一个惩罚器合并到当前惩罚器"""
        if not self._is_prepared and not their._is_prepared:  # 如果两者都未准备
            return  # 直接返回

        self.prepare()  # 准备当前惩罚器
        their.prepare()  # 准备另一个惩罚器
        self._merge(their)  # 调用子类的合并方法

    @abc.abstractmethod
    def _is_required(self) -> bool:  # 检查是否需要的抽象方法
        """
        Check if the penalizer is required to be prepared.
        """  # 检查惩罚器是否需要准备
        pass  # 抽象方法，子类必须实现

    @abc.abstractmethod
    def _prepare(self):  # 准备的抽象方法
        """
        Prepare the penalizer.
        Usually, this is where the penalizer initializes its tensors.
        """  # 准备惩罚器，通常在此初始化张量
        pass  # 抽象方法，子类必须实现

    @abc.abstractmethod
    def _cumulate_output_tokens(self, output_ids: torch.Tensor):  # 累计输出token的抽象方法
        """
        Cumulate the output tokens.
        Orchestrator will call this function to feed the output tokens to the penalizer.
        """  # 累计输出token，编排器调用此函数将输出token传递给惩罚器
        pass  # 抽象方法，子类必须实现

    @abc.abstractmethod
    def _apply(self, logits: torch.Tensor) -> torch.Tensor:  # 应用惩罚的抽象方法
        """
        Apply the penalizer to the logits.
        Penalizers can modify the logits in-place if needed.
        """  # 将惩罚器应用到logits，如果需要可以原地修改
        pass  # 抽象方法，子类必须实现

    def get_scaling_penalties(self) -> torch.Tensor:  # 获取缩放惩罚张量
        """
        Return the accumulated scaling penalty tensor for multiplicative penalizers.
        Only meaningful when is_multiplicative is True. Subclasses should override.
        """  # 返回乘性惩罚器的累积缩放惩罚张量，仅在is_multiplicative为True时有意义，子类应重写
        raise NotImplementedError  # 未实现异常

    @abc.abstractmethod
    def _filter(self, keep_indices: torch.Tensor):  # 过滤的抽象方法
        """
        Filter the penalizer (tensors or underlying data) based on the indices to keep in the batch.
        """  # 根据批次中保留的索引过滤惩罚器（张量或底层数据）
        pass  # 抽象方法，子类必须实现

    @abc.abstractmethod
    def _merge(self, their: "_BatchedPenalizer"):  # 合并的抽象方法
        """
        Merge the penalizer with another penalizer.
        """  # 将惩罚器与另一个惩罚器合并
        pass  # 抽象方法，子类必须实现

    @abc.abstractmethod
    def _teardown(self):  # 拆卸的抽象方法
        """
        Teardown the penalizer.
        """  # 拆卸惩罚器
        pass  # 抽象方法，子类必须实现
