# 自适应投机解码参数模块
# 根据观察到的接受长度，在运行时调整投机步数(speculative_num_steps)。
"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""

from __future__ import annotations  # 启用延迟注解求值

import json  # 导入JSON解析模块
import logging  # 导入日志模块
from typing import TYPE_CHECKING  # 导入类型检查常量

from sglang.srt.utils import log_info_on_rank0  # 导入rank0日志输出工具

if TYPE_CHECKING:  # 类型检查时才导入
    from sglang.srt.server_args import ServerArgs  # 服务器参数类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    """Return why adaptive spec cannot run under the given server args, or None if supported."""
    # 返回自适应投机解码在给定服务器参数下无法运行的原因，如支持则返回None。
    if server_args.speculative_algorithm not in ("EAGLE", "EAGLE3"):  # 检查算法是否为EAGLE或EAGLE3
        return (
            f"speculative_algorithm={server_args.speculative_algorithm} "
            "(only EAGLE/EAGLE3 are supported)"  # 仅支持EAGLE/EAGLE3
        )
    if server_args.speculative_eagle_topk != 1:  # 检查topk是否为1
        return (
            f"speculative_eagle_topk={server_args.speculative_eagle_topk} "
            "(only topk=1 is supported)"  # 仅支持topk=1
        )
    if server_args.enable_dp_attention:  # 检查是否启用了DP注意力
        return (
            "enable_dp_attention=True is not supported "
            "(adaptive tier decisions are not synchronized across DP ranks)"  # 自适应层级决策在DP rank间不同步
        )
    if server_args.enable_multi_layer_eagle:  # 检查是否启用了多层EAGLE
        return (
            "enable_multi_layer_eagle=True is not supported "
            "(MultiLayerEagleWorker does not implement adaptive)"  # MultiLayerEagleWorker未实现自适应
        )
    if server_args.enable_two_batch_overlap:  # 检查是否启用了双批次重叠
        return (
            "enable_two_batch_overlap=True is not supported "
            "(adaptive state swap would discard the TboAttnBackend wrapper)"  # 自适应状态交换会丢弃TboAttnBackend包装器
        )
    if server_args.enable_pdmux:  # 检查是否启用了PDMUX
        return (
            "enable_pdmux=True is not supported "
            "(adaptive state swap does not update decode_attn_backend_group)"  # 自适应状态交换不更新decode_attn_backend_group
        )
    return None  # 支持自适应，返回None


def load_adaptive_config(path: str | None) -> dict[str, object]:
    """Load adaptive speculative config from a JSON file.
    # 从JSON文件加载自适应投机解码配置。

    The file may contain any subset of the following keys:
    # 文件可以包含以下键的任意子集：
        ema_alpha, update_interval, warmup_batches,
        down_hysteresis, up_hysteresis, candidate_steps

    Returns an empty dict when *path* is ``None``.
    # 当path为None时返回空字典。
    """
    if path is None:  # 路径为空则返回空字典
        return {}
    with open(path) as f:  # 打开配置文件
        cfg = json.load(f)  # 加载JSON内容
    if not isinstance(cfg, dict):  # 检查是否为字典类型
        raise ValueError(
            "speculative_adaptive_config must be a JSON object, "
            f"got {type(cfg).__name__}"  # 配置必须是JSON对象
        )
    return cfg  # 返回配置字典


def _resolve_candidate_steps(initial_steps: int, cfg: dict[str, object]) -> list[int]:
    """Return sorted, deduplicated candidate steps; inserts *initial_steps* when missing."""
    # 返回排序去重的候选步数；当initial_steps缺失时插入。
    raw = cfg.get("candidate_steps") or (1, 3, 7)  # 从配置获取或使用默认值
    candidates: set[int] = set(raw)  # 去重

    # Ensure the worker's initial speculative_num_steps is itself a candidate.
    # 确保worker的初始speculative_num_steps本身是一个候选值。
    # Otherwise AdaptiveController.register() would store the worker's pre-built
    # runtime state under a key that _activate() never queries, leaking that
    # state's draft attn backend and cuda graph buffers for the process lifetime.
    # 否则AdaptiveController.register()会将worker预构建的运行时状态
    # 存储在_activate()从不查询的键下，导致该状态的draft注意力后端
    # 和CUDA图缓存在进程生命周期内泄漏。
    if initial_steps not in candidates:  # 如果初始步数不在候选集中
        log_info_on_rank0(
            logger,
            f"Adding initial speculative_num_steps={initial_steps} to "
            f"candidate_steps={sorted(candidates)} so the pre-built "
            f"runtime state is reused.",  # 将初始步数添加到候选集中以复用预构建状态
        )
        candidates.add(initial_steps)  # 添加初始步数

    return sorted(candidates)  # 返回排序后的候选步数列表


def resolve_candidate_steps_from_config(
    initial_steps: int, cfg_path: str | None
) -> list[int]:
    """Load adaptive config and resolve candidate steps."""
    # 加载自适应配置并解析候选步数。
    cfg = load_adaptive_config(cfg_path)  # 加载配置
    return _resolve_candidate_steps(initial_steps, cfg)  # 解析候选步数


class AdaptiveSpeculativeParams:
    """Tracks acceptance rate via EMA and adapts num_steps accordingly.
    # 通过EMA跟踪接受率并相应调整num_steps。

    The core idea: if drafts are consistently accepted, try more steps;
    # 核心思想：如果草稿持续被接受，尝试更多步数；
    if drafts are consistently rejected early, reduce steps to avoid waste.
    # 如果草稿持续被早期拒绝，减少步数以避免浪费。

    Formula: target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)
    # 公式：target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)
    - Probes one step beyond observed acceptance
    # 探测超出观察接受度一步
    - EMA smoothing prevents oscillation
    # EMA平滑防止振荡
    - Only updates every `update_interval` batches for stability
    # 仅每隔update_interval批次更新一次以保证稳定性
    """

    def __init__(
        self,
        initial_steps: int,  # 初始步数
        cfg_path: str | None = None,  # 配置文件路径
    ):
        cfg = load_adaptive_config(cfg_path)  # 加载配置
        # TODO: Wider range of candidate_steps (once lazy init is supported).
        # TODO: 更大范围的candidate_steps（一旦支持懒初始化）。
        self.candidate_steps = _resolve_candidate_steps(initial_steps, cfg)  # 解析候选步数
        assert (
            len(self.candidate_steps) >= 2
        ), "candidate_steps must have at least 2 distinct values"  # 候选步数至少需要2个不同值

        self.ema_alpha = cfg.get("ema_alpha", 0.2)  # EMA平滑系数，默认0.2
        self.update_interval = cfg.get("update_interval", 5)  # 更新间隔，默认5个批次
        self.warmup_batches = cfg.get("warmup_batches", 10)  # 预热批次数，默认10
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)  # 下降迟滞，默认-0.25
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)  # 上升迟滞，默认0.0

        self.current_steps = initial_steps  # 当前步数

        # Initialize EMA at current steps - 1 (neutral starting point)
        # 将EMA初始化为当前步数-1（中性起始点）
        self.ema_accept_len = float(self.current_steps - 1)  # EMA接受长度
        self._batch_count = 0  # 批次计数器

        log_info_on_rank0(
            logger,
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={self.current_steps}, candidate_steps={self.candidate_steps}",  # 记录初始化信息
        )

    def update(self, num_correct_drafts_per_req: list[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.
        # 用观察到的接受长度更新EMA。如果参数发生变化则返回True。

        Args:
            num_correct_drafts_per_req: Per-request accepted draft token counts from last verify.
            # 每个请求在上次验证中被接受的草稿token数。
        """
        if not num_correct_drafts_per_req:  # 如果列表为空
            return False  # 不更新

        batch_avg = sum(num_correct_drafts_per_req) / len(num_correct_drafts_per_req)  # 计算批次平均接受长度
        self.ema_accept_len = (  # 更新EMA值
            1 - self.ema_alpha
        ) * self.ema_accept_len + self.ema_alpha * batch_avg  # EMA公式

        self._batch_count += 1  # 递增批次计数
        if self._batch_count <= self.warmup_batches:  # 预热期内不更新
            return False

        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:  # 不在更新间隔点上
            return False

        return self._recompute_params()  # 重新计算参数

    def _recompute_params(self) -> bool:
        """Recompute steps from EMA. Returns True if params changed."""
        # 根据EMA重新计算步数。如果参数变化则返回True。
        old_steps = self.current_steps  # 保存旧步数
        current_idx = self.candidate_steps.index(old_steps)  # 获取当前步数在候选列表中的索引

        # TODO: Consider limiting step changes to avoid overshooting.
        # TODO: 考虑限制步数变化以避免过冲。
        while current_idx > 0:  # 向下调整步数
            prev_step = self.candidate_steps[current_idx - 1]  # 前一个候选步数
            drop_threshold = prev_step - 0.5 + self.down_hysteresis  # 下降阈值
            if self.ema_accept_len <= drop_threshold:  # EMA低于下降阈值
                current_idx -= 1  # 降低步数
            else:
                break  # 不再下降

        while current_idx < len(self.candidate_steps) - 1:  # 向上调整步数
            current_step = self.candidate_steps[current_idx]  # 当前候选步数
            rise_threshold = current_step - 0.5 + self.up_hysteresis  # 上升阈值
            if self.ema_accept_len > rise_threshold:  # EMA高于上升阈值
                current_idx += 1  # 增加步数
            else:
                break  # 不再上升

        target = self.candidate_steps[current_idx]  # 目标步数

        if target != old_steps:  # 如果步数发生变化
            self.current_steps = target  # 更新当前步数
            log_info_on_rank0(
                logger,
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})",  # 记录步数变化
            )
            return True  # 参数已变化
        return False  # 参数未变化
