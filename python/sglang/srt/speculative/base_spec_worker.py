# 推测解码（Speculative Decoding）的基础工作器抽象类定义。
# 本文件定义了推测解码框架中草稿模型工作器（BaseDraftWorker）和
# 推测工作器（BaseSpecWorker）的抽象基类，为具体的推测解码实现
# 提供统一的接口规范，包括草稿生成、草稿扩展、验证完成回调等核心方法。

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker


# 草稿模型工作器的抽象基类，定义草稿生成和扩展的接口
class BaseDraftWorker(ABC):
    # 草稿生成方法：由草稿模型生成候选token序列
    @abstractmethod
    def draft():
        pass

    # 草稿扩展方法：在已有草稿序列基础上继续扩展生成更多候选token
    @abstractmethod
    def draft_extend():
        pass


# 推测解码工作器的抽象基类，协调目标模型和草稿模型的交互
class BaseSpecWorker(ABC):
    # 目标模型工作器属性：返回用于验证的目标模型（大模型）工作器实例
    @property
    @abstractmethod
    def target_worker(self) -> TpModelWorker:
        pass

    # 草稿模型工作器属性：返回用于快速生成候选的草稿模型（小模型）工作器实例
    @property
    @abstractmethod
    def draft_worker(self) -> BaseDraftWorker:
        pass

    # spec_v2 所涉及的注意力后端元组，用于 decide_needs_cpu_seq_lens 进行按位或运算
    # 默认只返回目标模型的后端；子类可扩展以包含草稿模型的后端
    @property
    def spec_v2_attn_backends(self) -> tuple:
        """Attn backends touched by spec_v2 forward; OR-ed by decide_needs_cpu_seq_lens.
        Default returns target only; subclasses extend with draft backends."""
        return (self.target_worker.model_runner.attn_backend,)

    # 清理缓存池的抽象方法，用于释放推测解码过程中的临时缓存资源
    @abstractmethod
    def clear_cache_pool(self):
        # TODO: move this abstract method to BaseTpWorker and call through self.model_runner
        pass

    # 验证完成后的CPU端回调钩子，在验证完成且接受计数已传回CPU时调用。
    # 默认为空操作；支持自适应推测的工作器可重写此方法，向控制器反馈信息，
    # 而无需在工作器热路径中强制进行 GPU→CPU 同步。
    def on_verify_complete_cpu(self, num_correct_drafts_per_req: list[int]) -> None:
        """Hook called after verify finishes and accept counts are on CPU.

        Default no-op. Adaptive-aware workers override this to feed the
        controller without forcing a GPU→CPU sync in the worker hot path.
        """
        # num_correct_drafts_per_req: 每个请求中被目标模型接受的草稿token数量列表
        pass
