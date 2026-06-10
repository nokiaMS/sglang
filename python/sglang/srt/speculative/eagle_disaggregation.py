# EAGLE投机解码 disaggregation 模块
# 在 disaggregation（分离式部署）模式下构建EAGLE草稿输入，
# 从远程请求中获取 top-k 概率和索引以及隐藏状态。
from __future__ import annotations  # 启用延迟注解求值

from typing import TYPE_CHECKING  # 导入类型检查常量

import torch  # 导入PyTorch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode  # 导入隐藏状态捕获模式
from sglang.srt.speculative.eagle_info import EagleDraftInput  # 导入EAGLE草稿输入类

if TYPE_CHECKING:  # 类型检查时才导入
    from sglang.srt.managers.overlap_utils import FutureMap  # Future映射类
    from sglang.srt.managers.schedule_batch import ScheduleBatch  # 调度批次类
    from sglang.srt.server_args import ServerArgs  # 服务器参数类


def build_eagle_disagg_draft_input(
    batch: ScheduleBatch,  # 调度批次
    server_args: ServerArgs,  # 服务器参数
    last_tokens_tensor: torch.Tensor,  # 最后token张量
    future_map: FutureMap,  # Future映射
) -> EagleDraftInput:
    # 在disaggregation模式下构建EAGLE草稿输入
    num_states = server_args.speculative_eagle_topk  # 状态数 = topk
    if server_args.enable_multi_layer_eagle:  # 启用多层EAGLE
        num_states *= server_args.speculative_num_steps  # 状态数 = topk * 步数

    topk_p = torch.stack(  # 堆叠top-k概率
        [
            torch.as_tensor(
                req.output_topk_p[:num_states],  # 取前num_states个概率
                device=batch.device,
                dtype=torch.float32,
            )
            for req in batch.reqs  # 遍历每个请求
        ],
        dim=0,
    )
    topk_index = torch.stack(  # 堆叠top-k索引
        [
            torch.as_tensor(
                req.output_topk_index[:num_states],  # 取前num_states个索引
                device=batch.device,
                dtype=torch.int64,
            )
            for req in batch.reqs  # 遍历每个请求
        ],
        dim=0,
    )

    hidden_states = torch.stack(  # 堆叠隐藏状态
        [req.hidden_states_tensor for req in batch.reqs], dim=0
    ).to(batch.device)  # 移到批次设备

    spec_info = EagleDraftInput(  # 创建草稿输入
        topk_p=topk_p,
        topk_index=topk_index,
        hidden_states=hidden_states,
        bonus_tokens=last_tokens_tensor,
    )
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST  # 只捕获最后一层的隐藏状态

    if batch.enable_overlap:  # 启用重叠模式
        spec_info.future_indices = batch.req_pool_indices  # 设置Future索引
        future_map.publish(spec_info.future_indices, batch.seq_lens)  # 发布Future
        future_map.stash(spec_info.future_indices, spec_info)  # 暂存草稿输入

    return spec_info  # 返回草稿输入
