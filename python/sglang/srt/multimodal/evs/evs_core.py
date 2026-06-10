# 本文件实现 EVS（Efficient Video Sampling）的核心算法，包括计算保留 token 数量、
# 计算保留掩码、以及根据剪枝结果调整 input_ids 中的占位符等功能
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/evs.py

import torch  # 导入 PyTorch 深度学习框架


def compute_retained_tokens_count(
    tokens_per_frame: int, num_frames: int, q: float
) -> int:  # 返回保留的 token 总数
    """
    Compute the number of retained tokens for a given video.
    Method ensures that we retain all the tokens from the first frame
    regardless of the pruning rate.
    计算给定视频保留的 token 数量，确保第一帧的所有 token 始终保留。

    Args:
        tokens_per_frame: The number of tokens per frame.
        num_frames: The total number of frames.
        q: The pruning rate.

    Returns:
        The number of retained tokens.
    """
    total_tokens = tokens_per_frame * num_frames  # 计算 token 总数
    evs_num_tokens = int(total_tokens * (1 - q))  # 根据剪枝率计算保留的 token 数
    min_num_tokens = tokens_per_frame  # 最少保留一帧的 token 数
    return max(min_num_tokens, evs_num_tokens)  # 返回保留 token 数，确保不少于第一帧的 token 数


def compute_retention_mask(
    video_embeds: torch.Tensor,  # 视频嵌入张量
    video_size_thw: torch.LongTensor | tuple[int, int, int],  # 视频的 (时间, 高度, 宽度) 尺寸
    spatial_merge_size: int,  # 空间合并尺寸
    q: float,  # 剪枝率
) -> torch.Tensor:  # 返回保留掩码
    """
    Computes the retention mask for input video embeddings.
    计算输入视频嵌入的保留掩码，基于帧间余弦相似度进行时间冗余剪枝。

    Args:
        video_embeds (`torch.Tensor`): The input video embeddings
            of shape `(T * H * W // spatial_merge_size ^ 2, hidden_size)`
        video_size_thw (`torch.LongTensor` of shape `(3)`):
            The temporal, height and width of video.
        spatial_merge_size: Size reduction for rows & cols dimensions.
        q: (`float`): Pruning rate factor [0,1)

    Returns:
        `torch.Tensor`: The retention mask for the video embeddings of
            `(T * H * W // spatial_merge_size ^ 2)` shape.
    """
    T, H, W = map(int, video_size_thw)  # 解包视频尺寸

    # Use reshape instead of einops to avoid graph breaks
    video_embeds = video_embeds.reshape(  # 重塑为四维张量 [T, H', W', D]
        T,
        H // spatial_merge_size,
        W // spatial_merge_size,
        video_embeds.size(-1),
    )
    tokens_per_frame = (H // spatial_merge_size) * (W // spatial_merge_size)  # 计算每帧的 token 数
    # Core EVS
    similarity = torch.nn.functional.cosine_similarity(  # 计算相邻帧间的余弦相似度
        video_embeds[1:, ...], video_embeds[:-1, ...], dim=-1
    )
    dissimilarity = 1 - similarity  # 计算差异性（1 - 相似度）

    # Always ensure we include all tokens from the first frame
    dissimilarity = torch.cat(  # 第一帧的差异性设为最大值，确保其全部保留
        [255 * torch.ones_like(video_embeds[:1, :, :, 0]), dissimilarity], dim=0
    )

    dissimilarity_flat = dissimilarity.view(-1)  # 展平为一维
    order = torch.argsort(dissimilarity_flat, dim=-1, descending=True, stable=True)  # 按差异性降序排序
    retain_num_tokens = compute_retained_tokens_count(  # 计算保留 token 数量
        tokens_per_frame=tokens_per_frame, num_frames=T, q=q
    )
    topk_indices = order[:retain_num_tokens]  # 取差异性最大的前 N 个位置

    retention_mask = torch.zeros_like(dissimilarity_flat, dtype=torch.bool)  # 初始化全 False 掩码
    retention_mask[topk_indices] = True  # 将保留位置设为 True
    retention_mask = retention_mask.reshape(dissimilarity.size())  # 恢复原始形状

    mask = retention_mask.view(-1)  # "T H W -> (T H W)" 展平为一维布尔掩码
    return mask  # 返回保留掩码


# ▲ End of VLLM code


def tokens_per_frame(
    *,  # 以下参数必须以关键字参数形式传入
    q: float,  # 剪枝率
    num_frames: int,  # 帧数
    frame_num_tokens: int,  # 每帧 token 数
) -> list[int]:  # 返回每帧的 token 数列表
    """
    Before EVS pruning, we want to pre-reduce input_ids to be the same length that will be retained of embeddings due to EVS pruning, so the forward batch metadata will be correct post EVS.
    We don't know the exact number of tokens per frame after EVS pruning, but we know the *total* number of tokens that will be retained.
    So, we create a bogus tokens_per_frame list that sums to the total number of tokens that will be retained, and use it for placeholder spans, later to replaced, see `replace_offsets_with_tokens_per_frame` below.
    在 EVS 剪枝前，生成一个伪造的每帧 token 数列表，使总和等于剪枝后保留的 token 总数，
    用于预调整 input_ids 长度，后续会被实际值替换。
    """
    retained = compute_retained_tokens_count(  # 计算保留的 token 总数
        tokens_per_frame=frame_num_tokens, num_frames=num_frames, q=q
    )
    base = retained // num_frames  # 每帧的基础 token 数
    rem = retained % num_frames  # 余数
    tpf = [base] * (num_frames - 1) + [base + rem]  # 将余数加到最后一帧
    assert sum(tpf) == retained  # 确保总和等于保留的 token 数
    return tpf  # 返回每帧 token 数列表


def replace_offsets_with_tokens_per_frame(
    *,  # 以下参数必须以关键字参数形式传入
    pre_chunked_input_ids: list[int],  # 原始的分块 input_ids
    num_tokens_per_frame: list[int],  # 每帧新的 token 数
    frame_offsets_inclusive: list[tuple[int, int]],  # 每帧在 input_ids 中的起止偏移（含端点）
    filler_token_id: int,  # 填充 token 的 ID
) -> list[int]:  # 返回修改后的 input_ids
    """
    Given a single video, after EVS pruning of redundant tokens, we have a new `num_tokens_per_frame`, therefore the existing input_ids and offsets are stale.
    We need to replace all stale offsets with new offsets that reflect the new `num_tokens_per_frame`, respectively.
    在 EVS 剪枝后，用新的每帧 token 数替换 input_ids 中过时的占位符偏移区间。

    Returns:
        Modified input_ids with offsets replaced with new offsets.

    Examples:
    >>> assert replace_offsets_with_tokens_per_frame(
    ...     pre_chunked_input_ids=[1, 0, 0, 4, 5, 0, 0, 0, 9, 10, 0, 0, 12, 13],
    ...     frame_offsets_inclusive=[(1, 2), (5, 7), (10, 11)],
    ...     num_tokens_per_frame=[1, 4, 2],
    ...     filler_token_id=0,
    ... ) ==                      [1, 0, 4, 5, 0, 0, 0, 0, 9, 10, 0, 0, 12, 13]

    >>> assert replace_offsets_with_tokens_per_frame(
    ...     pre_chunked_input_ids=[1, 0, 0, 4, 5, 9, 10, 0, 0, 0],
    ...     frame_offsets_inclusive=[(1, 2), (7, 9)],
    ...     num_tokens_per_frame=[1, 4],
    ...     filler_token_id=0,
    ... ) ==                      [1, 0, 4, 5, 9, 10, 0, 0, 0, 0]

    >>> assert replace_offsets_with_tokens_per_frame(
    ...     pre_chunked_input_ids=[0, 0, 1, 4, 0, 0, 0, 5, 9, 10],
    ...     frame_offsets_inclusive=[(0, 1), (4, 6)],
    ...     num_tokens_per_frame=[1, 4],
    ...     filler_token_id=0,
    ... ) ==                      [0, 1, 4, 0, 0, 0, 0, 5, 9, 10]
    """
    assert isinstance(pre_chunked_input_ids, list)  # 确保 input_ids 是列表
    ids = pre_chunked_input_ids  # 别名引用

    if len(frame_offsets_inclusive) == 1:  # 如果只有一个帧偏移区间
        """There might be no frame separators, in which case there will be one contiguous span of tokens"""
        final = ids[0 : frame_offsets_inclusive[0][0]]  # 保留偏移前的部分
        frames = [filler_token_id] * sum(num_tokens_per_frame)  # 生成新的填充 token
        final.extend(frames)  # 添加填充 token
    else:  # 多个帧偏移区间
        cursor = 0  # 游标，跟踪当前处理位置
        final = []  # 存储最终结果
        for (start, end), num_tokens in zip(  # 遍历每帧的偏移和 token 数
            frame_offsets_inclusive, num_tokens_per_frame, strict=True
        ):
            final.extend(ids[cursor:start])  # 保留帧间隔前的非帧 token
            final.extend([filler_token_id] * num_tokens)  # 用新的填充 token 替换原帧区间
            cursor = end + 1  # 移动游标到帧区间之后
    final.extend(ids[frame_offsets_inclusive[-1][1] + 1 :])  # 保留最后一个帧偏移之后的部分
    return final  # 返回修改后的 input_ids
