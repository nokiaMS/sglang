# 自适应投机解码的运行时状态管理模块
# 定义投机解码各阶段（draft/verify/extend）的运行时资源状态，
# 以及自适应控制器，用于根据验证结果动态切换运行时状态。
import logging  # 导入日志模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Protocol  # 导入类型检查和协议基类

from sglang.srt.speculative.adaptive_spec_params import AdaptiveSpeculativeParams  # 导入自适应投机参数类

if TYPE_CHECKING:  # 类型检查时才导入
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend  # 注意力后端基类
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner  # CPU图运行器
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner  # CUDA图运行器
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (  # EAGLE draft CUDA图运行器
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (  # EAGLE draft extend CUDA图运行器
        EAGLEDraftExtendCudaGraphRunner,
    )

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@dataclass  # 数据类装饰器
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.
    # 绑定到特定投机解码配置的完整运行时资源集合。

    Each decode round runs three stages — draft, verify, extend — and every
    stage has shape-dependent resources (attention backends and CUDA graphs)
    that must match the current configuration.  Switching adaptive steps
    means swapping the entire state atomically.
    # 每个解码轮次运行三个阶段——draft、verify、extend——每个阶段
    # 都有与形状相关的资源（注意力后端和CUDA图），必须与当前配置匹配。
    # 切换自适应步数意味着原子性地交换整个状态。
    """

    # -- Configuration (determines shapes for all stages) --
    # -- 配置（决定所有阶段的形状）--
    speculative_num_steps: int  # 投机步数
    speculative_num_draft_tokens: int  # 投机草稿token数

    # -- Draft stage: draft model multi-step autoregressive generation --
    # -- Draft阶段：draft模型多步自回归生成 --
    draft_attn_backend: "AttentionBackend | None"  # draft阶段的注意力后端
    cuda_graph_runner: "EAGLEDraftCudaGraphRunner | None"  # draft阶段的CUDA图运行器

    # -- Verify stage: target model one-pass tree verification --
    # -- Verify阶段：目标模型单次树验证 --
    target_attn_backend: "AttentionBackend"  # verify阶段的注意力后端
    target_graph_runner: "CudaGraphRunner | CPUGraphRunner | None"  # verify阶段的CUDA/CPU图运行器

    # -- Extend stage: draft model KV cache catch-up after verify --
    # -- Extend阶段：验证后draft模型KV缓存追赶 --
    draft_extend_attn_backend: "AttentionBackend | None"  # extend阶段的注意力后端
    cuda_graph_runner_for_draft_extend: "EAGLEDraftExtendCudaGraphRunner | None"  # extend阶段的CUDA图运行器


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""
    # 使用AdaptiveController的worker必须实现的协议接口

    speculative_num_steps: int  # 投机步数属性

    def build_adaptive_runtime_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ) -> SpecRuntimeState: ...  # 构建自适应运行时状态的方法

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...  # 应用运行时状态的方法


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.
    # 拥有自适应决策和运行时状态切换的门面类。

    Works with any worker that implements ``AdaptiveSpecWorker`` protocol:
    # 与任何实现AdaptiveSpecWorker协议的worker配合使用：
      - ``build_adaptive_runtime_state(steps, draft_tokens)`` → runtime state
      - ``apply_runtime_state(state)`` → apply it to the worker
      # 构建运行时状态 → 应用到worker

    The worker only needs to:
    # worker只需：
      1. Call ``register()`` for the initial state, then ``init_states()``
         once during startup.
      # 1. 在启动时调用register()注册初始状态，然后调用init_states()
      2. Call ``on_verify_complete(num_correct_drafts_per_req)`` after each decode verify.
      # 2. 每次解码验证后调用on_verify_complete()
    """

    def __init__(self, worker: AdaptiveSpecWorker, config_path: str | None = None):
        # 初始化自适应控制器
        self.worker = worker  # 保存worker引用
        self.params = AdaptiveSpeculativeParams(  # 创建自适应投机参数
            initial_steps=worker.speculative_num_steps,  # 使用worker的初始步数
            cfg_path=config_path,  # 配置文件路径
        )
        self._states: dict[int, SpecRuntimeState] = {}  # 步数到运行时状态的映射

    @property  # 属性装饰器
    def candidate_steps(self) -> list[int]:
        # 获取候选步数列表
        return self.params.candidate_steps  # 返回参数中的候选步数

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.
        # 注册一个预构建的运行时状态。

        *steps* defaults to ``state.speculative_num_steps`` when not given.
        # 未指定steps时，默认使用state.speculative_num_steps
        """
        key = steps if steps is not None else state.speculative_num_steps  # 确定注册的键
        self._states[key] = state  # 将状态存入映射

    def init_states(self) -> None:
        """Build and register runtime states for all candidate steps."""
        # 为所有候选步数构建并注册运行时状态。
        for steps in self.params.candidate_steps:  # 遍历所有候选步数
            if steps in self._states:  # 如果该步数已有状态则跳过
                continue
            state = self.worker.build_adaptive_runtime_state(  # 构建运行时状态
                speculative_num_steps=steps,  # 指定步数
                speculative_num_draft_tokens=steps + 1,  # 草稿token数为步数+1
            )
            self._states[steps] = state  # 注册状态
        self._activate(self.params.current_steps)  # 激活当前步数对应的状态

    def on_verify_complete(self, num_correct_drafts_per_req: list[int]) -> None:
        """Feed verify results; switch runtime state if EMA warrants it."""
        # 输入验证结果；如果EMA（指数移动平均）表明需要，切换运行时状态。
        if self.params.update(num_correct_drafts_per_req):  # 更新参数，判断是否需要切换
            self._activate(self.params.current_steps)  # 激活新的步数对应的状态

    def _activate(self, speculative_num_steps: int) -> None:
        # 激活指定步数对应的运行时状态
        state = self._states.get(speculative_num_steps)  # 获取对应状态
        if state is None:  # 如果状态不存在
            raise ValueError(  # 抛出异常
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)  # 将状态应用到worker
