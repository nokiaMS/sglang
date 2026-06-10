# Frozen-KV MTP 工具函数模块
# 提供Frozen-KV MTP（多token预测）的上下文管理器、位置设置、
# top-k展开、隐藏状态选择和种子捕获等工具函数。
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations  # 启用延迟注解求值

from contextlib import contextmanager  # 导入上下文管理器装饰器
from typing import TYPE_CHECKING, Tuple  # 导入类型注解

import torch  # 导入PyTorch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # 导入logits处理器输出
from sglang.srt.managers.schedule_batch import ScheduleBatch  # 导入调度批次
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次
from sglang.srt.speculative.frozen_kv_mtp_info import (  # 导入Frozen-KV MTP信息类
    FrozenKVMTPContext,
    FrozenKVMTPDraftExtendInput,
    FrozenKVMTPDraftInput,
)
from sglang.srt.speculative.spec_utils import fast_topk  # 导入快速topk工具

if TYPE_CHECKING:  # 类型检查时才导入
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


@contextmanager  # 上下文管理器装饰器
def frozen_kv_target_view(
    forward_batch: ForwardBatch,  # 前向批次
    kv_context: FrozenKVMTPContext,  # Frozen-KV MTP上下文
    draft_attn_backend: "AttentionBackend",  # 草稿注意力后端
):
    # 构建针对已提交目标前缀几何的注意力元数据
    """Build attention metadata against committed target-prefix geometry.
    # 根据已提交目标前缀几何构建注意力元数据。

    Swaps ``draft_attn_backend.token_to_kv_pool`` to the frozen target pool
    so any helper that reads ``get_token_to_kv_pool()`` during metadata init
    sees the frozen target pool. Pool refs are derived from
    ``get_attn_backend().token_to_kv_pool`` — the single backend-attribute
    swap is seen by both readers (``get_token_to_kv_pool()`` and the
    backend's own ``self.token_to_kv_pool``).
    # 将draft_attn_backend.token_to_kv_pool交换到冻结的目标池，
    # 以便在元数据初始化期间读取get_token_to_kv_pool()的辅助函数
    # 能看到冻结的目标池。池引用从get_attn_backend().token_to_kv_pool派生——
    # 单个后端属性交换被两个读取者（get_token_to_kv_pool()和后端自身的
    # self.token_to_kv_pool）共同看到。
    """
    if kv_context is None:  # 上下文未绑定
        raise RuntimeError(
            "Frozen-KV MTP target view called before the model was bound; "
            "bind the frozen KV context first."
        )
    saved_spec_info = forward_batch.spec_info  # 保存原spec_info
    forward_batch.spec_info = None  # 临时清除spec_info
    saved_backend_pool = draft_attn_backend.token_to_kv_pool  # 保存原KV池
    draft_attn_backend.token_to_kv_pool = kv_context.target_token_to_kv_pool  # 交换到目标池
    try:
        yield  # 在此上下文中执行
    finally:
        forward_batch.spec_info = saved_spec_info  # 恢复spec_info
        draft_attn_backend.token_to_kv_pool = saved_backend_pool  # 恢复原KV池


@contextmanager  # 上下文管理器装饰器
def target_kv_pool_view(
    forward_batch: ForwardBatch,  # 前向批次
    kv_context: FrozenKVMTPContext,  # Frozen-KV MTP上下文
    draft_attn_backend: "AttentionBackend",  # 草稿注意力后端
):
    # 使用目标的冻结KV池运行草稿模型前向
    """Run the draft model's forward with the target's frozen KV pool.
    # 使用目标的冻结KV池运行草稿模型的前向。

    Swaps ``draft_attn_backend.token_to_kv_pool`` to the frozen target pool.
    The single backend-attribute swap is seen by both readers —
    ``get_token_to_kv_pool()`` (because it resolves through
    ``get_attn_backend()``) and the backend's own ``self.token_to_kv_pool``
    reads (because ``self is draft_attn_backend``).
    # 将draft_attn_backend.token_to_kv_pool交换到冻结的目标池。
    # 单个后端属性交换被两个读取者共同看到——get_token_to_kv_pool()
    # （因为它通过get_attn_backend()解析）和后端自身的
    # self.token_to_kv_pool读取（因为self is draft_attn_backend）。
    """
    if kv_context is None:  # 上下文未绑定
        raise RuntimeError(
            "Frozen-KV MTP target KV pool view called before the model was bound; "
            "bind the frozen KV context first."
        )
    saved_backend_pool = draft_attn_backend.token_to_kv_pool  # 保存原KV池
    draft_attn_backend.token_to_kv_pool = kv_context.target_token_to_kv_pool  # 交换到目标池
    try:
        yield  # 在此上下文中执行
    finally:
        draft_attn_backend.token_to_kv_pool = saved_backend_pool  # 恢复原KV池


def set_frozen_kv_positions(forward_batch: ForwardBatch, topk: int) -> None:
    # 设置Frozen-KV的位置：RoPE阶段为最后写入的目标槽位，不随草稿步进推进
    """Rope phase = last written target slot, not advanced per draft step."""
    seq_lens = forward_batch.seq_lens  # 序列长度
    positions = torch.clamp(seq_lens - 1, min=0).to(torch.int64)  # 位置 = seq_len - 1（最小0）
    if (  # topk>1时需要扩展
        topk > 1
        and forward_batch.positions is not None
        and forward_batch.positions.numel() == positions.numel() * topk
    ):
        positions = positions.repeat_interleave(topk, dim=0)  # 每个请求复制topk次
    if forward_batch.positions is None:  # 位置未初始化
        forward_batch.positions = positions  # 直接设置
    else:  # 位置已初始化
        if forward_batch.positions.shape == positions.shape:  # 形状匹配
            forward_batch.positions.copy_(positions)  # 原地复制
        else:  # 形状不匹配
            forward_batch.positions = positions  # 替换


def expand_for_topk_draft(forward_batch: ForwardBatch, topk: int) -> None:
    # 为top-k草稿扩展已提交前缀的元数据
    """Repeat committed-prefix metadata for the active ``B * topk`` frontier."""
    # 为活跃的B * topk前沿重复已提交前缀的元数据。
    if topk == 1 or forward_batch.batch_size == 0:  # topk=1或空批次无需扩展
        return

    if forward_batch.batch_size != forward_batch.seq_lens.shape[0]:  # 批次已展开
        raise RuntimeError(
            "Frozen-KV MTP topk expansion expects an unexpanded forward "
            "batch where batch_size == len(seq_lens)."
        )

    forward_batch.batch_size *= topk  # 扩展批次大小
    forward_batch.req_pool_indices = forward_batch.req_pool_indices.repeat_interleave(
        topk, dim=0
    )  # 重复请求池索引
    forward_batch.seq_lens = forward_batch.seq_lens.repeat_interleave(topk, dim=0)  # 重复序列长度
    if forward_batch.seq_lens_cpu is not None:  # CPU序列长度
        forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu.repeat_interleave(
            topk, dim=0
        )  # 重复
        forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()  # 重新求和
    else:  # 无CPU序列长度
        forward_batch.seq_lens_sum = torch.sum(forward_batch.seq_lens).item()  # 从GPU求和

    positions = torch.clamp(forward_batch.seq_lens - 1, min=0).to(torch.int64)  # 计算位置
    forward_batch.positions = positions  # 设置位置
    forward_batch.num_token_non_padded_cpu = positions.numel()  # 非填充token数
    if forward_batch.num_token_non_padded is not None:  # 有GPU非填充计数
        forward_batch.num_token_non_padded.fill_(positions.numel())  # 填充
    if (  # 有多维RoPE位置且可以扩展
        forward_batch.mrope_positions is not None
        and forward_batch.mrope_positions.shape[-1] * topk == positions.numel()
    ):
        forward_batch.mrope_positions = forward_batch.mrope_positions.repeat_interleave(
            topk, dim=-1
        )  # 重复多维RoPE位置


def position_for_batch(batch: ScheduleBatch) -> torch.Tensor:
    # 为批次计算位置（序列长度-1，最小0）
    return torch.clamp(batch.seq_lens - 1, min=0).to(torch.int64)


def select_last_extend_hidden(
    batch: ScheduleBatch, hidden_states: torch.Tensor
) -> torch.Tensor:
    # 从扩展隐藏状态中选择每个请求的最后一个token的隐藏状态
    if hidden_states.shape[0] == batch.batch_size():  # 已经是每请求一个
        return hidden_states  # 直接返回
    lens = torch.tensor(batch.extend_lens, device=hidden_states.device)  # 扩展长度
    last_indices = torch.cumsum(lens, dim=0) - 1  # 最后token的索引
    return hidden_states[last_indices.to(torch.long)]  # 返回最后token的隐藏状态


def select_last_verified_seed(
    draft_input: FrozenKVMTPDraftExtendInput,  # Frozen-KV MTP草稿扩展输入
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 从验证后的扩展输入中选择每个请求最后接受的token的种子
    counts = draft_input.num_accept_tokens.to(torch.long)  # 每请求接受的token数
    last_indices = torch.cumsum(counts, dim=0) - 1  # 最后接受token的索引
    return (
        draft_input.input_ids[last_indices],  # 最后接受的input_ids
        draft_input.hidden_states[last_indices],  # 最后接受的隐藏状态
    )


def capture_for_decode(
    logits_output: LogitsProcessorOutput, draft_input: FrozenKVMTPDraftInput, topk: int
) -> None:
    # 从解码logits中捕获下一步草稿的种子（top-k概率和索引以及隐藏状态）
    probs = torch.softmax(logits_output.next_token_logits, dim=-1)  # 计算概率
    draft_input.topk_p, draft_input.topk_index = fast_topk(probs, topk, dim=-1)  # 计算top-k
    draft_input.hidden_states = logits_output.hidden_states  # 保存隐藏状态
