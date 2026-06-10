# 对数概率(Logprob)工具模块
# 提供对数概率的提取、Top-K选择、指定token ID概率查询等功能，
# 支持预填充和解码阶段，以及投机解码(Speculative Decoding)的logprob处理。

from __future__ import annotations  # 启用延迟类型注解求值

import dataclasses  # 导入数据类模块
from enum import Enum, auto  # 导入枚举类型
from typing import TYPE_CHECKING, List, Optional, Union  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.environ import envs  # 导入环境变量配置

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput  # 导入logits元数据和输出类型
    from sglang.srt.managers.schedule_batch import ScheduleBatch  # 导入调度批次类型
    from sglang.srt.speculative.eagle_info import EagleVerifyOutput  # 导入Eagle验证输出类型
    from sglang.srt.speculative.ngram_info import NgramVerifyInput  # 导入Ngram验证输入类型


class LogprobStage(Enum):  # 对数概率计算阶段枚举
    PREFILL = auto()  # 预填充阶段
    DECODE = auto()  # 解码阶段


@dataclasses.dataclass
class InputLogprobsResult:  # 输入token对数概率结果数据类
    input_token_logprobs: torch.Tensor  # 输入token的对数概率张量
    input_top_logprobs_val: Optional[List] = None  # 输入top-k对数概率值
    input_top_logprobs_idx: Optional[List] = None  # 输入top-k对数概率索引
    input_token_ids_logprobs_val: Optional[List] = None  # 输入指定token ID的对数概率值
    input_token_ids_logprobs_idx: Optional[List] = None  # 输入指定token ID的索引


def get_top_logprobs_raw(  # 获取原始Top-K对数概率（支持预填充和解码阶段）
    logprobs: torch.Tensor,  # 对数概率张量
    top_logprobs_nums: List[int],  # 每个请求的Top-K数量列表
    stage: LogprobStage,  # 计算阶段（预填充/解码）
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,  # 预填充阶段每个请求的裁剪长度
    no_copy_to_cpu: bool = False,  # 是否不拷贝到CPU
):
    max_k = max(top_logprobs_nums)  # 取最大的K值
    values, indices = logprobs.topk(max_k, dim=-1)  # 取前max_k个最大值和索引
    if not no_copy_to_cpu:  # 如果需要拷贝到CPU
        values = values.tolist()  # 将值转为Python列表
        indices = indices.tolist()  # 将索引转为Python列表

    top_logprobs_val = []  # Top-K对数概率值列表
    top_logprobs_idx = []  # Top-K对数概率索引列表

    if stage == LogprobStage.DECODE:  # 解码阶段
        for i, k in enumerate(top_logprobs_nums):  # 遍历每个请求
            top_logprobs_val.append(values[i][:k])  # 取前k个值
            top_logprobs_idx.append(indices[i][:k])  # 取前k个索引
    else:  # 预填充阶段
        pt = 0  # 指针，跟踪当前在展平列表中的位置
        for k, pruned_len in zip(top_logprobs_nums, extend_logprob_pruned_lens_cpu):  # 遍历每个请求
            if pruned_len <= 0:  # 如果裁剪长度为0或负
                top_logprobs_val.append([])  # 添加空列表
                top_logprobs_idx.append([])  # 添加空列表
                continue  # 跳过

            top_logprobs_val.append([values[pt + j][:k] for j in range(pruned_len)])  # 取每个token的前k个值
            top_logprobs_idx.append([indices[pt + j][:k] for j in range(pruned_len)])  # 取每个token的前k个索引
            pt += pruned_len  # 移动指针

    return top_logprobs_val, top_logprobs_idx  # 返回Top-K值和索引


def get_top_logprobs_prefill(  # 获取预填充阶段的Top-K对数概率
    all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata  # 全部对数概率和元数据
):
    return get_top_logprobs_raw(  # 调用原始Top-K函数
        all_logprobs,  # 对数概率
        logits_metadata.top_logprobs_nums,  # Top-K数量
        stage=LogprobStage.PREFILL,  # 预填充阶段
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,  # 裁剪长度
    )


def get_top_logprobs(  # 获取解码阶段的Top-K对数概率
    logprobs: torch.Tensor,  # 对数概率张量
    top_logprobs_nums: List[int],  # Top-K数量列表
    no_copy_to_cpu: bool = False,  # 是否不拷贝到CPU
):
    return get_top_logprobs_raw(  # 调用原始Top-K函数
        logprobs,  # 对数概率
        top_logprobs_nums,  # Top-K数量
        stage=LogprobStage.DECODE,  # 解码阶段
        no_copy_to_cpu=no_copy_to_cpu,  # 是否不拷贝到CPU
    )


def get_token_ids_logprobs_raw(  # 获取指定token ID的原始对数概率（支持预填充和解码阶段）
    logprobs: torch.Tensor,  # 对数概率张量
    token_ids_logprobs_list: List[Optional[List[int]]],  # 每个请求要查询的token ID列表
    stage: LogprobStage,  # 计算阶段
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,  # 预填充阶段的裁剪长度
    no_copy_to_cpu: bool = False,  # 是否不拷贝到CPU
):
    vals, idxs = [], []  # 初始化值和索引列表
    if stage == LogprobStage.DECODE:  # 解码阶段
        for i, token_ids in enumerate(token_ids_logprobs_list):  # 遍历每个请求
            if token_ids is None:  # 如果未指定token ID
                vals.append([])  # 添加空列表
                idxs.append([])  # 添加空列表
            else:  # 否则查询指定token的概率
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(  # 构建token ID张量
                    logprobs.device, non_blocking=True  # 异步拷贝到对应设备
                )
                row = logprobs[i, token_ids_tensor]  # 取该行中指定token的对数概率
                vals.append(row if no_copy_to_cpu else row.tolist())  # 添加值（按需转列表）
                idxs.append(token_ids)  # 添加索引
    else:  # prefill  # 预填充阶段
        pt = 0  # 指针
        for i, (token_ids, pruned_len) in enumerate(  # 遍历每个请求
            zip(token_ids_logprobs_list, extend_logprob_pruned_lens_cpu)
        ):
            if pruned_len <= 0:  # 如果裁剪长度为0或负
                vals.append([])  # 添加空列表
                idxs.append([])  # 添加空列表
                continue  # 跳过
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(  # 构建token ID张量
                logprobs.device, non_blocking=True  # 异步拷贝到对应设备
            )
            pos_logprobs = logprobs[pt : pt + pruned_len, token_ids_tensor]  # 取裁剪范围内指定token的概率
            vals.append(pos_logprobs if no_copy_to_cpu else pos_logprobs.tolist())  # 添加值（按需转列表）
            idxs.append([token_ids for _ in range(pruned_len)])  # 每个位置都记录相同的token ID列表
            pt += pruned_len  # 移动指针
    return vals, idxs  # 返回值和索引


def get_token_ids_logprobs_prefill(  # 获取预填充阶段指定token ID的对数概率
    all_logprobs, logits_metadata: LogitsMetadata, no_copy_to_cpu=False  # 全部对数概率、元数据和是否不拷贝到CPU
):
    return get_token_ids_logprobs_raw(  # 调用原始函数
        all_logprobs,  # 对数概率
        logits_metadata.token_ids_logprobs,  # token ID列表
        stage=LogprobStage.PREFILL,  # 预填充阶段
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,  # 裁剪长度
        no_copy_to_cpu=no_copy_to_cpu,  # 是否不拷贝到CPU
    )


def get_token_ids_logprobs(logprobs, token_ids_logprobs, no_copy_to_cpu=False):  # 获取解码阶段指定token ID的对数概率
    return get_token_ids_logprobs_raw(  # 调用原始函数
        logprobs,  # 对数概率
        token_ids_logprobs,  # token ID列表
        stage=LogprobStage.DECODE,  # 解码阶段
        no_copy_to_cpu=no_copy_to_cpu,  # 是否不拷贝到CPU
    )


def get_top_logprobs_chunk(  # 分块获取Top-K对数概率，处理跨块边界情况
    logprobs: torch.Tensor,  # 对数概率张量
    logits_metadata: LogitsMetadata,  # logits元数据
    top_k_nums: List[int],  # 每个序列的Top-K数量
    pruned_lens: List[int],  # 每个序列的裁剪长度
    input_top_logprobs_val: List,  # 存储Top-K对数概率值的列表
    input_top_logprobs_idx: List,  # 存储Top-K token索引的列表
    split_pruned_len: int,  # 上一块中剩余的裁剪token长度
) -> int:
    """Get top-k logprobs for each sequence in the chunk.
    # 获取分块中每个序列的top-k对数概率。

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size]
        # 对数概率张量，形状为 [seq_len, vocab_size]
        logits_metadata: Metadata containing top-k and pruned length info
        # 包含top-k和裁剪长度信息的元数据
        top_k_nums: List of top-k numbers for each sequence
        # 每个序列的top-k数量列表
        pruned_lens: List of pruned lengths for each sequence
        # 每个序列的裁剪长度列表
        input_top_logprobs_val: List to store top-k logprob values
        # 存储top-k对数概率值的列表
        input_top_logprobs_idx: List to store top-k token indices
        # 存储top-k token索引的列表
        split_pruned_len: Length of pruned tokens from previous chunk
        # 上一块中裁剪token的长度

    Returns:
        int: Number of remaining tokens to process in next chunk
        # 整数：下一个块中需要处理的剩余token数
    """
    # No sequences in the chunk
    # 分块中没有序列
    if logprobs.shape[0] == 0:  # 如果对数概率张量为空
        return 0  # 返回0

    max_k = max(logits_metadata.top_logprobs_nums)  # 取最大的K值
    ret = logprobs.topk(max_k, dim=1)  # 取前max_k个最大值
    values = ret.values.tolist()  # 转为Python列表
    indices = ret.indices.tolist()  # 转为Python列表

    pt = 0  # 指针，跟踪当前在展平列表中的位置
    next_split_pruned_len = 0  # 下一个块的剩余裁剪长度
    for n, (k, pruned_len) in enumerate(zip(top_k_nums, pruned_lens)):  # 遍历每个序列
        if n == 0:  # 对于第一个序列
            # For the first sequence, adjust the pruned length
            # 对于第一个序列，调整裁剪长度
            pruned_len -= split_pruned_len  # 减去上一块中剩余的裁剪长度
        else:  # 后续序列
            # After the first sequence, no split in the middle
            # 第一个序列之后，不存在中间切分
            split_pruned_len = 0  # 重置为0

        if pruned_len <= 0:  # 如果调整后裁剪长度为0或负
            # if pruned length is less than or equal to 0,
            # there is no top-k logprobs to process
            # 如果裁剪长度小于等于0，则没有top-k对数概率需要处理
            input_top_logprobs_val.append([])  # 添加空列表
            input_top_logprobs_idx.append([])  # 添加空列表
            continue  # 跳过

        # Get the top-k logprobs
        # 获取top-k对数概率
        val = []  # 当前序列的值列表
        idx = []  # 当前序列的索引列表
        for j in range(pruned_len):  # 遍历裁剪范围内的每个token
            # Handle remaining tokens in next chunk if any
            # 如果有剩余token则在下一个块中处理
            if pt + j >= len(values):  # 如果超出当前块的范围
                next_split_pruned_len = split_pruned_len + j  # 记录下一个块的剩余裁剪长度
                break  # 跳出循环
            # Append the top-k logprobs
            # 追加top-k对数概率
            val.append(values[pt + j][:k])  # 添加前k个值
            idx.append(indices[pt + j][:k])  # 添加前k个索引

        # Append or extend based on whether the sequence was split across chunks
        # 根据序列是否跨块切分来决定追加还是扩展
        if len(val) > 0:  # 如果有值
            if split_pruned_len > 0:  # 如果是跨块延续的序列
                input_top_logprobs_val[-1].extend(val)  # 扩展到上一个条目
                input_top_logprobs_idx[-1].extend(idx)  # 扩展到上一个条目
            else:  # 新序列
                input_top_logprobs_val.append(val)  # 追加新条目
                input_top_logprobs_idx.append(idx)  # 追加新条目

        pt += pruned_len  # 移动指针
    return next_split_pruned_len  # 返回下一个块的剩余裁剪长度


def get_token_ids_logprobs_chunk(  # 分块获取指定token ID的对数概率，处理跨块边界情况
    logprobs: torch.Tensor,  # 对数概率张量
    token_ids_logprobs: List[int],  # 每个序列要查询的token ID列表
    pruned_lens: List[int],  # 每个序列的裁剪长度
    input_token_ids_logprobs_val: List,  # 存储token ID对数概率值的列表
    input_token_ids_logprobs_idx: List,  # 存储token ID索引的列表
    split_pruned_len: int = 0,  # 上一块中剩余的裁剪token长度
):
    """Get token_ids logprobs for each sequence in the chunk.
    # 获取分块中每个序列的token_ids对数概率。

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size]
        # 对数概率张量，形状为 [seq_len, vocab_size]
        logits_metadata: Metadata containing token IDs and pruned length info
        # 包含token ID和裁剪长度信息的元数据
        token_ids_logprobs: List of token IDs for each sequence
        # 每个序列的token ID列表
        pruned_lens: List of pruned lengths for each sequence
        # 每个序列的裁剪长度列表
        input_token_ids_logprobs_val: List to store token logprob values
        # 存储token对数概率值的列表
        input_token_ids_logprobs_idx: List to store token indices
        # 存储token索引的列表
        split_pruned_len: Length of pruned tokens from previous chunk
        # 上一块中裁剪token的长度

    Returns:
        int: Number of remaining tokens to process in next chunk
        # 整数：下一个块中需要处理的剩余token数
    """

    # No sequences in the chunk
    # 分块中没有序列
    if logprobs.shape[0] == 0:  # 如果对数概率张量为空
        return 0  # 返回0

    pt = 0  # 指针
    next_split_pruned_len = 0  # 下一个块的剩余裁剪长度
    for n, (token_ids, pruned_len) in enumerate(  # 遍历每个序列
        zip(
            token_ids_logprobs,  # token ID列表
            pruned_lens,  # 裁剪长度列表
        )
    ):
        # Adjust pruned length for first sequence
        # 对第一个序列调整裁剪长度
        if n == 0:  # 第一个序列
            pruned_len -= split_pruned_len  # 减去上一块的剩余裁剪长度
        else:  # 后续序列
            split_pruned_len = 0  # 重置为0

        if pruned_len <= 0:  # 如果裁剪长度为0或负
            # if pruned length is less than or equal to 0,
            # there is no token ids logprobs to process
            # 如果裁剪长度小于等于0，则没有token ID对数概率需要处理
            input_token_ids_logprobs_val.append([])  # 添加空列表
            input_token_ids_logprobs_idx.append([])  # 添加空列表
            continue  # 跳过

        # Get the token ids logprobs
        # 获取token ID对数概率
        val = []  # 当前序列的值列表
        idx = []  # 当前序列的索引列表
        for j in range(pruned_len):  # 遍历裁剪范围内的每个token
            # Handle remaining tokens in next chunk if any
            # 如果有剩余token则在下一个块中处理
            if pt + j >= logprobs.shape[0]:  # 如果超出当前块的范围
                next_split_pruned_len = split_pruned_len + j  # 记录下一个块的剩余裁剪长度
                break  # 跳出循环
            if token_ids is not None:  # 如果指定了token ID
                val.append(logprobs[pt + j, token_ids].tolist())  # 取指定token的概率并转列表
                idx.append(token_ids)  # 记录token ID

        # Append or extend based on whether the sequence was split across chunks
        # 根据序列是否跨块切分来决定追加还是扩展
        if len(val) > 0:  # 如果有值
            if split_pruned_len > 0:  # 如果是跨块延续的序列
                input_token_ids_logprobs_val[-1].extend(val)  # 扩展到上一个条目
                input_token_ids_logprobs_idx[-1].extend(idx)  # 扩展到上一个条目
            else:  # 新序列
                input_token_ids_logprobs_val.append(val)  # 追加新条目
                input_token_ids_logprobs_idx.append(idx)  # 追加新条目

        pt += pruned_len  # 移动指针
    return next_split_pruned_len  # 返回下一个块的剩余裁剪长度


def add_output_logprobs_for_spec_v1(  # 为投机解码v1添加输出对数概率
    batch: ScheduleBatch,  # 调度批次
    res: Union[EagleVerifyOutput, NgramVerifyInput],  # 验证结果
    logits_output: Optional[LogitsProcessorOutput] = None,  # logits输出（可选）
):
    # Extract args
    # 提取参数
    if logits_output is None:  # 如果未提供logits输出
        logits_output = res.logits_output  # 从验证结果中获取

    if hasattr(res, "num_correct_drafts_per_req_cpu"):  # 如果结果有每个请求的正确草稿数
        num_correct_drafts_per_req_cpu = res.num_correct_drafts_per_req_cpu  # 直接获取
    else:  # 否则
        # FIXME: Get a NgramVerifyOutput class and use that instead of this hack.
        # FIXME: 获取NgramVerifyOutput类并使用它代替这个hack。
        num_correct_drafts_per_req_cpu = res.num_correct_drafts.tolist()  # 从张量转为列表

    top_logprobs_nums = batch.top_logprobs_nums  # 获取每个请求的Top-K数量
    token_ids_logprobs = batch.token_ids_logprobs  # 获取每个请求要查询的token ID
    accept_indices = res.accept_indices  # 获取接受索引
    assert len(accept_indices) == len(logits_output.next_token_logits)  # 断言接受索引和logits长度一致

    temperatures = batch.sampling_info.temperatures  # 获取温度参数
    num_draft_tokens = batch.spec_info.draft_token_num  # 获取草稿token数量
    # acceptance indices are the indices in a "flattened" batch.
    # dividing it to num_draft_tokens will yield the actual batch index.
    # 接受索引是"展平"批次中的索引。
    # 除以num_draft_tokens将得到实际的批次索引。
    temperatures = temperatures[accept_indices // num_draft_tokens]  # 按接受索引映射温度
    if envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():  # 如果要求返回原始对数概率
        logprobs = torch.nn.functional.log_softmax(  # 不应用温度
            logits_output.next_token_logits, dim=-1
        )
    else:  # 否则
        logprobs = torch.nn.functional.log_softmax(  # 应用温度缩放
            logits_output.next_token_logits / temperatures, dim=-1
        )
    batch_next_token_ids = res.accept_tokens  # 获取接受的token ID
    num_tokens_per_req = [accept + 1 for accept in num_correct_drafts_per_req_cpu]  # 每个请求的token数 = 正确草稿数 + 1

    # We should repeat top_logprobs_nums to match num_tokens_per_req.
    # 我们应该重复top_logprobs_nums以匹配num_tokens_per_req。
    top_logprobs_nums_repeat_interleaved = [  # 交错重复Top-K数量
        num
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req)  # 对每个请求
        for _ in range(num_tokens)  # 重复num_tokens次
    ]

    token_ids_logprobs_repeat_interleaved = [  # 交错重复token ID对数概率
        token_ids
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req)  # 对每个请求
        for _ in range(num_tokens)  # 重复num_tokens次
    ]

    # Extract logprobs
    # 提取对数概率
    should_top_logprobs = any(x > 0 for x in top_logprobs_nums)  # 是否需要Top-K对数概率
    should_token_ids_logprobs = any(x is not None for x in token_ids_logprobs)  # 是否需要指定token ID对数概率
    if should_top_logprobs:  # 如果需要Top-K对数概率
        (
            logits_output.next_token_top_logprobs_val,  # Top-K对数概率值
            logits_output.next_token_top_logprobs_idx,  # Top-K对数概率索引
        ) = get_top_logprobs(
            logprobs,  # 对数概率
            top_logprobs_nums_repeat_interleaved,  # 交错重复的Top-K数量
        )

    if should_token_ids_logprobs:  # 如果需要指定token ID对数概率
        (
            logits_output.next_token_token_ids_logprobs_val,  # token ID对数概率值
            logits_output.next_token_token_ids_logprobs_idx,  # token ID对数概率索引
        ) = get_token_ids_logprobs(
            logprobs,  # 对数概率
            token_ids_logprobs_repeat_interleaved,  # 交错重复的token ID列表
        )

    logits_output.next_token_logprobs = logprobs[  # 提取接受token的对数概率
        torch.arange(len(batch_next_token_ids), device=batch.sampling_info.device),  # 行索引
        batch_next_token_ids,  # 列索引（接受token ID）
    ]

    # Add output logprobs to the request
    # 将输出对数概率添加到请求中
    pt = 0  # 指针
    next_token_logprobs = logits_output.next_token_logprobs.tolist()  # 转为Python列表
    accept_tokens_list = batch_next_token_ids.tolist()  # 接受token列表
    token_top_logprobs_val = logits_output.next_token_top_logprobs_val  # Top-K对数概率值
    token_top_logprobs_idx = logits_output.next_token_top_logprobs_idx  # Top-K对数概率索引
    token_ids_logprobs_val = logits_output.next_token_token_ids_logprobs_val  # token ID对数概率值
    token_ids_logprobs_idx = logits_output.next_token_token_ids_logprobs_idx  # token ID对数概率索引
    for req, num_tokens in zip(batch.reqs, num_tokens_per_req, strict=True):  # 遍历每个请求
        for _ in range(num_tokens):  # 遍历每个接受的token
            if req.return_logprob:  # 如果请求要求返回对数概率
                req.logprob.output_token_logprobs_val.append(next_token_logprobs[pt])  # 添加token对数概率值
                req.logprob.output_token_logprobs_idx.append(accept_tokens_list[pt])  # 添加token ID
                if req.logprob.top_logprobs_num > 0:  # 如果请求Top-K对数概率
                    assert (  # 断言应该计算Top-K
                        should_top_logprobs
                    ), "Inconsistent state: should_top_logprobs is False"  # 不一致状态：should_top_logprobs为False
                    req.logprob.output_top_logprobs_val.append(  # 添加Top-K对数概率值
                        token_top_logprobs_val[pt]
                    )
                    req.logprob.output_top_logprobs_idx.append(  # 添加Top-K对数概率索引
                        token_top_logprobs_idx[pt]
                    )
                if (  # 如果请求指定token ID对数概率且应该计算
                    req.logprob.token_ids_logprob is not None
                    and should_token_ids_logprobs
                ):
                    req.logprob.output_token_ids_logprobs_val.append(  # 添加token ID对数概率值
                        token_ids_logprobs_val[pt]
                    )
                    req.logprob.output_token_ids_logprobs_idx.append(  # 添加token ID对数概率索引
                        token_ids_logprobs_idx[pt]
                    )
            pt += 1  # 移动指针


def compute_spec_v2_logprobs(  # 计算投机解码v2的对数概率
    batch,  # 批次
    logits_output,  # logits输出
    predict: torch.Tensor,  # 预测token张量
    accept_index: torch.Tensor,  # 接受索引张量
    speculative_num_steps: int,  # 投机步数
):
    """Compute logprobs for accepted tokens after spec v2 verify sampling.
    # 在投机解码v2验证采样后计算接受token的对数概率。

    Gathers logits at accepted positions, applies log_softmax (temperature-scaled
    if not greedy), and populates logits_output.next_token_logprobs (plus optional
    top-k / token-ids logprobs) so they flow through copy_to_cpu().
    # 在接受位置收集logits，应用log_softmax（非贪心模式下进行温度缩放），
    # 并填充 logits_output.next_token_logprobs（以及可选的 top-k / token-ids 对数概率），
    # 以便它们通过 copy_to_cpu() 流转。
    """
    bs = len(batch.seq_lens)  # 批次大小
    max_accept = speculative_num_steps + 1  # 最大接受token数 = 投机步数 + 1
    device = predict.device  # 设备

    flat_accept_idx = accept_index.long().reshape(-1)  # 展平接受索引
    gathered_logits = logits_output.next_token_logits[flat_accept_idx]  # 按接受索引收集logits

    if batch.sampling_info.is_all_greedy or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():  # 贪心模式或要求原始概率
        gathered_logprobs = torch.nn.functional.log_softmax(gathered_logits, dim=-1)  # 不应用温度
    else:  # 否则应用温度缩放
        temperatures = torch.repeat_interleave(  # 重复温度以匹配max_accept
            batch.sampling_info.temperatures,
            max_accept,
            dim=0,
        )
        gathered_logprobs = torch.nn.functional.log_softmax(  # 应用温度缩放的log_softmax
            gathered_logits / temperatures, dim=-1
        )
    gathered_logprobs.clamp_(min=torch.finfo(gathered_logprobs.dtype).min)  # 钳制最小值以避免数值问题

    accepted_token_ids = predict[flat_accept_idx]  # 按接受索引获取预测token ID
    token_logprobs = gathered_logprobs[  # 提取接受token的对数概率
        torch.arange(bs * max_accept, device=device),  # 行索引
        accepted_token_ids.long(),  # 列索引（token ID）
    ]
    logits_output.next_token_logprobs = token_logprobs.reshape(bs, max_accept)  # 重塑为 [bs, max_accept]

    if batch.top_logprobs_nums and any(x > 0 for x in batch.top_logprobs_nums):  # 如果需要Top-K对数概率
        top_logprobs_nums_expanded = [  # 扩展Top-K数量以匹配每个接受token
            num for num in batch.top_logprobs_nums for _ in range(max_accept)
        ]
        (
            logits_output.next_token_top_logprobs_val,  # Top-K对数概率值
            logits_output.next_token_top_logprobs_idx,  # Top-K对数概率索引
        ) = get_top_logprobs(
            gathered_logprobs, top_logprobs_nums_expanded, no_copy_to_cpu=True  # 不拷贝到CPU，保持GPU张量
        )

    if batch.token_ids_logprobs and any(  # 如果需要指定token ID对数概率
        x is not None for x in batch.token_ids_logprobs
    ):
        token_ids_logprobs_expanded = [  # 扩展token ID列表以匹配每个接受token
            ids for ids in batch.token_ids_logprobs for _ in range(max_accept)
        ]
        (
            logits_output.next_token_token_ids_logprobs_val,  # token ID对数概率值
            logits_output.next_token_token_ids_logprobs_idx,  # token ID对数概率索引
        ) = get_token_ids_logprobs(
            gathered_logprobs, token_ids_logprobs_expanded, no_copy_to_cpu=True  # 不拷贝到CPU，保持GPU张量
        )
