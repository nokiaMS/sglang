# 上下文并行(Context Parallel)工具模块
# 提供上下文并行计算所需的元数据结构、数据切分/重组、全聚集通信等功能，
# 支持预填充阶段的序列切分(zigzag/round-robin)和KV缓存的跨rank同步。

from dataclasses import dataclass  # 导入数据类装饰器
from itertools import accumulate  # 导入累加函数
from typing import Callable, List  # 导入类型提示

import torch  # 导入PyTorch
import torch.nn.functional as F  # 导入PyTorch函数式模块

from sglang.srt.distributed.device_communicators.pynccl_allocator import (  # 导入对称内存使用函数
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (  # 导入DP注意力相关的CP工具函数
    attn_cp_all_gather_into_tensor,
    get_attention_cp_group,
    get_attention_cp_rank,
    get_attention_cp_size,
    is_allocation_symmetric,
)
from sglang.srt.layers.moe import get_moe_a2a_backend  # 导入MoE全互联后端获取函数
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool  # 导入KV池获取函数
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数


@dataclass
class ContextParallelMetadata:  # 上下文并行元数据类，存储CP切分和重组所需的所有索引与张量
    # Layout lists have length bs * cp_segment_num (= bs * 2 * cp_size).
    # 布局列表长度为 bs * cp_segment_num（即 bs * 2 * cp_size）
    split_list: List[int] = None  # 每个块的token数量列表
    zigzag_index: List[int] = None  # zigzag排列索引，选取当前rank拥有的块
    cp_reverse_index: List[int] = None  # 全聚集后的逆排列索引，恢复原始顺序
    reverse_split_len: List[int] = None  # 全聚集后各rank的切分长度列表

    # Per-rank-aggregate lists have length cp_size.
    # 每个rank聚合列表长度为 cp_size
    # max_rank_len is a list of cp_size copies of max(per_rank_actual_token),
    # kept as a list for torch.split() bucket sizes.
    # max_rank_len 是 cp_size 个 max(per_rank_actual_token) 的拷贝，
    # 保留为列表以便 torch.split() 直接用作分桶大小。
    per_rank_actual_token: List[int] = None  # 每个rank实际拥有的token数量
    max_rank_len: List[int] = None  # 每个rank的最大长度（用于填充对齐）

    # Per-sequence FlashAttention tensors (shape [bs] or [bs+1]).
    # 每个序列的FlashAttention张量（形状 [bs] 或 [bs+1]）
    kv_len_prev_tensor: torch.Tensor = None  # [bs] int32 CUDA  # 前半部分KV长度张量
    kv_len_next_tensor: torch.Tensor = None  # [bs] int32 CUDA  # 后半部分KV长度张量
    actual_seq_q_prev_tensor: torch.Tensor = None  # [bs] int32 CUDA  # 前半部分实际查询序列长度
    actual_seq_q_next_tensor: torch.Tensor = None  # [bs] int32 CUDA  # 后半部分实际查询序列长度
    cu_seqlens_q_prev_tensor: torch.Tensor = None  # [bs+1] int32 CUDA  # 前半部分累积序列长度
    cu_seqlens_q_next_tensor: torch.Tensor = None  # [bs+1] int32 CUDA  # 后半部分累积序列长度

    # Scalars derived from the per-sequence lists above.
    # 从上述每序列列表派生的标量
    total_q_prev_tokens: int = 0  # 前半部分总查询token数
    total_q_next_tokens: int = 0  # 后半部分总查询token数
    max_seqlen_q_prev: int = 0  # 前半部分最大查询序列长度
    max_seqlen_q_next: int = 0  # 后半部分最大查询序列长度

    # Per-seq CPU lists (useful for NSA indexer and diagnostics).
    # 每序列CPU列表（用于NSA索引器和诊断）
    kv_len_prev_list: List[int] = None  # 前半部分KV长度的CPU列表
    kv_len_next_list: List[int] = None  # 后半部分KV长度的CPU列表
    actual_seq_q_prev_list: List[int] = None  # 前半部分实际查询序列长度的CPU列表
    actual_seq_q_next_list: List[int] = None  # 后半部分实际查询序列长度的CPU列表

    # Aggregate sum of extend_seq_lens across the batch.
    # 批次中所有扩展序列长度的总和
    total_seq_lens: int = 0  # 总序列长度
    bs: int = 1  # 批次大小


def is_prefill_context_parallel_enabled():  # 判断预填充上下文并行是否启用
    return get_global_server_args().enable_prefill_context_parallel  # 返回全局服务器参数中的启用标志


def is_prefill_cp_in_seq_split():  # 判断预填充CP是否为序列内切分模式
    return (  # 返回判断结果
        is_prefill_context_parallel_enabled()  # 预填充CP已启用
        and get_global_server_args().prefill_cp_mode == "in-seq-split"  # 且模式为序列内切分
    )


def is_mla_prefill_cp_enabled() -> bool:  # 判断MLA预填充上下文并行是否启用
    sa = get_global_server_args()  # 获取全局服务器参数
    return sa.enable_prefill_context_parallel and sa.use_mla_backend  # 返回CP启用且使用MLA后端的判断结果


def mla_use_prefill_cp(forward_batch, mla_enable_prefill_cp=None):  # 判断当前前向批次是否应使用MLA预填充CP
    if mla_enable_prefill_cp is None:  # 如果未显式指定MLA CP启用标志
        mla_enable_prefill_cp = is_mla_prefill_cp_enabled()  # 则从全局参数获取
    return (  # 返回判断结果
        forward_batch.attn_cp_metadata is not None  # CP元数据存在
        and mla_enable_prefill_cp  # MLA CP已启用
        and forward_batch.forward_mode.is_context_parallel_extend()  # 且当前为CP扩展模式
    )


def can_cp_split(seq_len: int, cp_size: int, forward_batch):  # 判断给定序列是否可以进行CP切分
    # Base conditions: CP must be enabled, size > 1, and this must be a
    # CP-extend (prefill) step. The seq_len // (cp_size * 2) check ensures
    # the load-balancing split into 2 * cp_size blocks is non-degenerate.
    # 基本条件：CP必须启用、size > 1、且必须是CP扩展（预填充）步骤。
    # seq_len // (cp_size * 2) 检查确保负载均衡切分为 2*cp_size 个块是有效的。
    from sglang.srt.model_executor.forward_batch_info import ForwardMode  # 导入前向模式枚举

    cur_cp_seq_len = seq_len // (cp_size * 2)  # 计算切分后每个块的序列长度
    if not (  # 如果不满足以下所有条件
        cur_cp_seq_len != 0  # 切分后块长度不为0
        and cp_size > 1  # CP大小大于1
        # prepare_context_parallel_metadata hard-codes bs_per_cp_group = 1;
        # guard explicitly to avoid silent mis-partitioning under continuous batching.
        # prepare_context_parallel_metadata 硬编码 bs_per_cp_group = 1；
        # 显式保护以避免连续批处理下的静默错误分区。
        and forward_batch.forward_mode.is_context_parallel_extend()  # 当前为CP扩展模式
        # is_context_parallel_extend() returns True for MIXED (prefill+decode
        # in one step), but the zigzag split only makes sense on pure extend.
        # is_context_parallel_extend() 对 MIXED（一步中同时预填充和解码）也返回 True，
        # 但 zigzag 切分仅在纯扩展时才有意义。
        and forward_batch.forward_mode != ForwardMode.MIXED  # 排除混合模式
        and is_prefill_context_parallel_enabled()  # 预填充CP已启用
    ):
        return False  # 不满足条件则返回False

    # Per-sequence guards for bs > 1. Every sequence must be long enough for
    # the 2*cp_size-way split. A sub-threshold request reaching this point
    # means the scheduler failed to filter it out and a silent non-CP
    # fallback would have masked the bug -- raise instead. Per-sequence
    # radix-cache prefix is supported: prefix is baked into kv_len_prev/next
    # via prefix_offsets[s] inside prepare_context_parallel_metadata.
    # 批次大小 > 1 时的每序列保护。每个序列必须足够长以支持 2*cp_size 路切分。
    # 低于阈值的请求到达此处意味着调度器未能过滤掉它，静默的非CP回退会掩盖bug。
    # 支持每序列的radix缓存前缀：前缀通过 prepare_context_parallel_metadata
    # 中的 prefix_offsets[s] 被编入 kv_len_prev/next。
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)  # 获取扩展序列长度列表
    if extend_lens is None:  # 如果没有扩展序列长度信息
        return True  # 默认允许切分

    cp_min = cp_size * 2  # 最小可切分长度为 2*cp_size
    for L in extend_lens:  # 遍历每个序列的扩展长度
        if L < cp_min:  # 如果序列长度小于最小可切分长度
            # A sub-threshold request cannot be zigzag-split into 2*cp_size
            # blocks; fall back to a normal (non-CP) prefill for this batch
            # instead of failing. Happens e.g. when a radix-cache prefix hit
            # leaves only a few unique extend tokens.
            # 低于阈值的请求无法进行 zigzag 切分为 2*cp_size 个块；
            # 对此批次回退到普通（非CP）预填充而不是报错。
            # 例如当 radix 缓存前缀命中后仅剩下少量唯一扩展token时会发生。
            return False  # 回退到非CP模式

    return True  # 所有序列均满足切分条件


def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):  # 按CP元数据切分并重组输入数据
    from sglang.srt.layers.attention.dsa.utils import (  # 导入DSA相关的CP切分工具
        dsa_cp_round_robin_split_data,
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():  # 如果使用DSA轮询切分模式
        cp_size = get_attention_cp_size()  # 获取CP大小
        assert (  # 断言输入的第一维可被CP大小整除
            input_.shape[0] % cp_size == 0
        ), f"Expect input shape 0 can divided by cp size, but got input shape {input_.shape}, cp size {cp_size}"  # 期望输入第0维可被CP大小整除，但得到...
        return dsa_cp_round_robin_split_data(input_)  # 返回DSA轮询切分结果

    input_list = list(  # 按split_list切分输入
        torch.split(input_, forward_batch.attn_cp_metadata.split_list, dim=0)
    )
    result = torch.cat(  # 按zigzag_index重组切分后的块
        [input_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index], dim=0
    ).view(-1, input_.shape[-1])  # 展平为二维张量
    return result  # 返回重组结果


def cp_split_and_rebuild_position(forward_batch, positions: torch.Tensor):  # 按CP元数据切分并重组位置编码
    from sglang.srt.layers.attention.dsa.utils import (  # 导入DSA相关的CP切分工具
        dsa_cp_round_robin_split_data,
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():  # 如果使用DSA轮询切分模式
        cp_size = get_attention_cp_size()  # 获取CP大小
        assert positions.shape[0] % cp_size == 0, (  # 断言位置编码长度可被CP大小整除
            f"Expect positions shape 0 can divided by cp size, but got positions shape {positions.shape}, "
            f"cp size {cp_size}"  # 期望位置编码第0维可被CP大小整除，但得到...
        )
        return dsa_cp_round_robin_split_data(positions)  # 返回DSA轮询切分结果

    position_id_list = list(  # 按split_list切分位置编码
        torch.split(positions, forward_batch.attn_cp_metadata.split_list, dim=-1)
    )
    positions = torch.cat(  # 按zigzag_index重组切分后的位置块
        [position_id_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index],
        dim=-1,
    )
    return positions  # 返回重组后的位置编码


def cp_round_robin_input_ids(input_ids):  # 对输入ID进行轮询式重排，分配给各CP rank
    """
    input input_ids:
    rank0~7: 0,1,2,3,4,5,...
    # 输入 input_ids:
    # rank0~7: 0,1,2,3,4,5,...

    output input_ids:
    a2a none:
    rank0~7: 0,8,16,...,1,9,17,...,2,10,18,...
    # 输出 input_ids（a2a none）:
    # rank0~7: 0,8,16,...,1,9,17,...,2,10,18,...

    not a2a none:
    rank0: 0,8,16,...
    rank1: 1,9,17,...
    rank2: 2,10,18,...
    ...
    # 非a2a none时:
    # rank0: 0,8,16,...
    # rank1: 1,9,17,...
    # rank2: 2,10,18,...
    # ...
    """
    cp_size = get_attention_cp_size()  # 获取CP大小
    cp_rank = get_attention_cp_rank()  # 获取当前CP rank
    if get_moe_a2a_backend().is_none():  # 如果MoE全互联后端为none
        input_ids = input_ids.reshape(-1, cp_size).T.flatten()  # 重塑后转置再展平，实现轮询重排
    else:  # 否则使用步长采样方式
        input_ids = input_ids[cp_rank::cp_size].contiguous()  # 按rank步长采样并确保连续内存
    return input_ids  # 返回重排后的输入ID


def cp_all_gather_reorganized_into_tensor(input_tensor, cp_size, forward_batch, stream):  # 对2D张量执行全聚集并按实际token重组
    """
    Allgather communication for context_parallel(kv_cache, index_k, hidden_states).
    This implementation mainly consists of three parts:
    Step 1, padding the input shape to unify the shape for allgather communication (the shape must be the same).
    Step 2, allgather communication(async).
    Step 3, removing the padding and reassembling the data according to the actual tokens.
    # 上下文并行的全聚集通信（用于kv_cache, index_k, hidden_states）。
    # 该实现主要包含三个步骤：
    # 第1步，填充输入形状以统一全聚集通信的形状（形状必须相同）。
    # 第2步，全聚集通信（异步）。
    # 第3步，移除填充并按实际token重组数据。
    """
    max_len = forward_batch.attn_cp_metadata.max_rank_len[0]  # 获取最大rank长度
    pad_size = max_len - input_tensor.shape[0]  # 计算需要填充的大小
    if pad_size > 0:  # 如果需要填充
        input_tensor = F.pad(  # 对输入张量进行常量填充
            input_tensor, (0, 0, 0, pad_size), mode="constant", value=0
        )
    with use_symmetric_memory(  # 使用对称内存上下文
        get_attention_cp_group(), disabled=not is_allocation_symmetric()
    ):
        input_tensor_full = torch.empty(  # 创建全聚集输出缓冲区
            max_len * cp_size,  # 总长度为最大rank长度乘以CP大小
            input_tensor.shape[1],  # 特征维度不变
            device=input_tensor.device,  # 设备不变
            dtype=input_tensor.dtype,  # 数据类型不变
        )

    get_attention_cp_group().cp_all_gather_into_tensor_async(  # 异步执行全聚集
        input_tensor_full, input_tensor, stream
    )

    outputs_list_max = list(  # 按max_rank_len切分全聚集结果
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(  # 按实际token数量截取并拼接
        [
            outputs_list_max[index][:per_rank_len]  # 截取每个rank实际长度
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )

    return outputs  # 返回重组后的输出


def cp_all_gather_reorganized_into_tensor_kv_cache(  # 对多维KV缓存张量执行全聚集并按实际token重组
    input_tensor, cp_size, forward_batch, stream
):
    """
    Allgather communication for context_parallel KV cache.
    Handles multi-dimensional tensors (e.g., [seq_len, num_heads, head_dim]).
    # 上下文并行KV缓存的全聚集通信。
    # 处理多维张量（如 [seq_len, num_heads, head_dim]）。
    """
    max_len = forward_batch.attn_cp_metadata.max_rank_len[0]  # 获取最大rank长度
    pad_size = max_len - input_tensor.shape[0]  # 计算需要填充的大小
    if pad_size > 0:  # 如果需要填充
        # Pad the first dimension (seq_len). F.pad expects padding in reverse dimension order.
        # For n dimensional tensor, we need 2*n values: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
        # To pad only the first dimension: [0, 0] * (ndim - 1) + [0, pad_size]
        # 填充第一维（seq_len）。F.pad 期望按维度逆序填充。
        # 对于n维张量，需要2*n个值：(最后维左, 最后维右, ..., 第一维左, 第一维右)
        # 仅填充第一维: [0, 0] * (ndim - 1) + [0, pad_size]
        padding = [0, 0] * (input_tensor.ndim - 1) + [0, pad_size]  # 构造填充参数
        input_tensor = F.pad(input_tensor, padding, mode="constant", value=0)  # 执行填充

    # Create output tensor with proper shape for all dimensions
    # 创建具有正确形状的输出张量以容纳所有维度
    with use_symmetric_memory(  # 使用对称内存上下文
        get_attention_cp_group(), disabled=not is_allocation_symmetric()
    ):
        input_tensor_full = torch.empty(  # 创建全聚集输出缓冲区
            max_len * cp_size,  # 总长度
            *input_tensor.shape[1:],  # 其余维度保持不变
            device=input_tensor.device,  # 设备不变
            dtype=input_tensor.dtype,  # 数据类型不变
        )

    get_attention_cp_group().cp_all_gather_into_tensor_async(  # 异步执行全聚集
        input_tensor_full, input_tensor, stream
    )

    outputs_list_max = list(  # 按max_rank_len切分全聚集结果
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(  # 按实际token数量截取并拼接
        [
            outputs_list_max[index][:per_rank_len]  # 截取每个rank实际长度
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )

    return outputs  # 返回重组后的输出


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):  # 全聚集后重排输出张量，恢复原始序列顺序
    """
    # for in-seq-split
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+
    # 序列内切分模式:
    # 全聚集前: 各rank拥有对称的块对 (block0,block7), (block1,block6) 等
    # 重排前: 块按rank顺序排列
    # 重排后: 块按原始序列顺序排列

    # for round-robin-split
    |   +-----------before allgather------------+|
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |
    |   +--------------result-------------------+
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |   +-------------------------+
    # 轮询切分模式:
    # 全聚集前: 各rank按步长拥有token
    # 重排后: token按原始顺序排列
    """
    from sglang.srt.layers.attention.dsa.utils import (  # 导入DSA相关判断函数
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():  # 如果使用DSA轮询切分模式
        with use_symmetric_memory(  # 使用对称内存上下文
            get_attention_cp_group(), disabled=not is_allocation_symmetric()
        ):
            output_tensor = input_tensor.new_empty(  # 创建输出张量
                (input_tensor.shape[0] * cp_size, *input_tensor.shape[1:]),  # 大小为输入的cp_size倍
            )
        attn_cp_all_gather_into_tensor(  # 执行全聚集
            output_tensor,
            input_tensor,
        )
        out_shape = output_tensor.shape  # 获取输出形状
        output_tensor = (  # 转置重排：将 [cp_size, seq_len, ...] 转为 [seq_len, cp_size, ...] 再展平
            output_tensor.view(cp_size, -1, *out_shape[1:])  # 重塑为 [cp_size, seq_per_rank, ...]
            .transpose(0, 1)  # 转置前两维
            .reshape(out_shape)  # 重塑回原始形状
        )
        return output_tensor  # 返回重排后的张量

    # TODO: Do we need to remove the padding here?
    # TODO: 这里是否需要移除填充？
    bs_seq_len, hidden_size = input_tensor.shape  # 获取序列长度和隐藏维度
    output_tensor = cp_all_gather_reorganized_into_tensor(  # 执行全聚集并重组
        input_tensor,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(  # 按reverse_split_len切分输出
        torch.split(
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(  # 按cp_reverse_index逆排列恢复原始顺序
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )
    output_tensor = output_tensor.view(-1, hidden_size)  # 展平为二维张量
    return output_tensor  # 返回重排后的张量


def cp_all_gather_rerange_kv_cache(input_tensor, cp_size, forward_batch, stream):  # 全聚集后重排KV缓存，恢复原始序列顺序
    """
    Allgather and reorganize KV cache from all ranks in context parallel group.
    # 从上下文并行组的所有rank全聚集并重组KV缓存。

    # for in-seq-split
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+
    # 序列内切分模式下的KV缓存重排，恢复原始块顺序
    """
    output_tensor = cp_all_gather_reorganized_into_tensor_kv_cache(  # 执行多维全聚集并重组
        input_tensor,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(  # 按reverse_split_len切分输出
        torch.split(
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(  # 按cp_reverse_index逆排列恢复原始顺序
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )
    # No need to reshape - output_tensor already has the correct shape [seq_len, ...]
    # 无需重塑 - output_tensor 已具有正确形状 [seq_len, ...]
    return output_tensor  # 返回重排后的KV缓存


def cp_allgather_and_save_kv_cache(forward_batch, layer, k, v, cp_size):  # 全聚集KV缓存并保存到各rank的本地内存池
    """
    Allgather KV cache from all CP ranks and write the full result
    into each rank's local memory pool.
    # 从所有CP rank全聚集KV缓存，并将完整结果写入每个rank的本地内存池。
    """
    cache_loc = (  # 获取缓存位置
        forward_batch.out_cache_loc  # 非交叉注意力时使用out_cache_loc
        if not layer.is_cross_attention  # 如果不是交叉注意力
        else forward_batch.encoder_out_cache_loc  # 否则使用encoder_out_cache_loc
    )

    k = k.contiguous()  # 确保key张量内存连续
    v = v.contiguous()  # 确保value张量内存连续

    key_cache_full = cp_all_gather_rerange_kv_cache(  # 全聚集并重排key缓存
        k, cp_size, forward_batch, torch.cuda.current_stream()
    )
    value_cache_full = cp_all_gather_rerange_kv_cache(  # 全聚集并重排value缓存
        v, cp_size, forward_batch, torch.cuda.current_stream()
    )

    get_token_to_kv_pool().set_kv_buffer(  # 将完整的KV缓存写入内存池
        layer,  # 层对象
        cache_loc,  # 缓存位置
        key_cache_full,  # 完整的key缓存
        value_cache_full,  # 完整的value缓存
        layer.k_scale,  # key缩放因子
        layer.v_scale,  # value缩放因子
    )


def cp_attn_forward_extend(  # CP注意力前向扩展：将查询切分为前后两半，分别计算注意力后拼接
    forward_batch,
    q: torch.Tensor,
    device: torch.device,
    attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """
    Split q into prev/next zigzag halves based on CP metadata, call the
    backend-specific attention function twice with appropriate per-half
    metadata, and concatenate the results.
    # 根据CP元数据将q切分为前/后zigzag两半，使用各自的元数据分别调用
    # 后端特定的注意力函数，然后拼接结果。

    For bs > 1, q is laid out as [all_prev_tokens_across_seqs,
    all_next_tokens_across_seqs]; the split point is total_q_prev_tokens.
    cu_seqlens_q_prev/next tensors have shape [bs+1] and carry the
    per-sequence boundaries through FlashAttention's variable-length API.
    # 对于 bs > 1，q的布局为 [所有序列的prev_token, 所有序列的next_token]；
    # 切分点为 total_q_prev_tokens。
    # cu_seqlens_q_prev/next 张量形状为 [bs+1]，通过FlashAttention的变长API
    # 传递每序列的边界信息。

    attn_fn signature:
        attn_fn(q, cu_seqlens_q, cache_seqlens, max_seqlen_q) -> result
    where only these four CP-varying parameters differ between halves.
    All other backend-specific args should be captured in the closure.
    # attn_fn 签名:
    #     attn_fn(q, cu_seqlens_q, cache_seqlens, max_seqlen_q) -> result
    # 其中只有这四个CP相关参数在前后两半之间不同。
    # 所有其他后端特定参数应在闭包中捕获。
    """
    cp_meta = forward_batch.attn_cp_metadata  # 获取CP元数据

    q_prev = q[: cp_meta.total_q_prev_tokens]  # 提取前半部分查询
    q_next = q[cp_meta.total_q_prev_tokens :]  # 提取后半部分查询

    result_prev = attn_fn(  # 计算前半部分注意力
        q_prev,  # 前半部分查询
        cp_meta.cu_seqlens_q_prev_tensor,  # 前半部分累积序列长度
        cp_meta.kv_len_prev_tensor,  # 前半部分KV长度
        cp_meta.max_seqlen_q_prev,  # 前半部分最大查询序列长度
    )
    result_next = attn_fn(  # 计算后半部分注意力
        q_next,  # 后半部分查询
        cp_meta.cu_seqlens_q_next_tensor,  # 后半部分累积序列长度
        cp_meta.kv_len_next_tensor,  # 后半部分KV长度
        cp_meta.max_seqlen_q_next,  # 后半部分最大查询序列长度
    )

    return torch.concat([result_prev, result_next], dim=0)  # 拼接前后两部分的注意力结果


def prepare_context_parallel_metadata(  # 准备上下文并行元数据，计算所有切分、索引和FlashAttention参数
    kv_len,  # KV总长度
    cp_rank,  # 当前CP rank
    cp_size,  # CP大小
    seqs_len,  # 序列长度列表
    extend_seqs_len=None,  # 扩展序列长度列表
    device="cuda",  # 设备类型
):
    from sglang.srt.layers.attention.dsa.utils import (  # 导入DSA相关判断函数
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():  # 如果使用DSA轮询切分模式
        return ContextParallelMetadata()  # 返回空元数据（轮询模式不需要zigzag元数据）

    """prepare_input_dp_with_cp_dsa-zigzag index
    Example (DP_ATTENT_TP == CP_SIZE == 4, single sequence):
        block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7
        rank 0: block0, block7
        rank 1: block1, block6
        rank 2: block2, block5
        rank 3: block3, block4
    For bs > 1, each sequence is split into cp_segment_num = 2 * cp_size
    blocks independently; per-rank layout becomes:
        [s0.block_r, s1.block_r, ..., s_{bs-1}.block_r,
         s0.block_{2*cp_size-1-r}, ..., s_{bs-1}.block_{2*cp_size-1-r}]
    i.e. all prev blocks first, then all next blocks -- so torch.split at
    total_q_prev_tokens cleanly separates them.
    # 准备DP+CP DSA zigzag索引
    # 示例 (DP_ATTENT_TP == CP_SIZE == 4, 单序列):
    #     block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7
    #     rank 0: block0, block7
    #     rank 1: block1, block6
    #     rank 2: block2, block5
    #     rank 3: block3, block4
    # 对于 bs > 1，每个序列独立切分为 cp_segment_num = 2 * cp_size 个块；
    # 每个rank的布局变为：
    #     [s0.block_r, s1.block_r, ..., s_{bs-1}.block_r,
    #      s0.block_{2*cp_size-1-r}, ..., s_{bs-1}.block_{2*cp_size-1-r}]
    # 即所有前半块在前，所有后半块在后——因此 torch.split 在
    # total_q_prev_tokens 处可以干净地分离它们。
    """
    assert extend_seqs_len is not None  # 断言扩展序列长度列表不为空
    extend_seqs_len = [int(x) for x in extend_seqs_len]  # 将扩展序列长度转为整数列表

    # Update the extend_seqs_len to the padded length.
    # 将扩展序列长度更新为填充后的长度。
    pad_len = int(kv_len) - sum(extend_seqs_len)  # 计算填充长度
    if pad_len > 0:  # 如果需要填充
        extend_seqs_len[-1] += pad_len  # 将填充长度加到最后一个序列
        if seqs_len is not None and len(seqs_len) == len(extend_seqs_len):  # 如果序列长度列表有效
            seqs_len = list(seqs_len)  # 转为列表（避免修改原始数据）
            seqs_len[-1] += pad_len  # 同样更新最后一个序列长度

    bs = len(extend_seqs_len)  # 批次大小
    cp_segment_num = cp_size * 2  # CP段数 = 2 * CP大小

    # Prefix offset (radix cache hit length) per sequence. For non-NSA
    # (FlashAttention) the prefix is baked into kv_len_prev/next via
    # prefix_offsets[s] below, so cache_seqlens correctly covers the cached
    # prefix. NSA leaves bare cumulatives so its indexer can re-add the
    # offset itself.
    # 每个序列的前缀偏移量（radix缓存命中长度）。对于非NSA
    # (FlashAttention)，前缀通过下方的prefix_offsets[s]被编入
    # kv_len_prev/next，因此cache_seqlens正确覆盖了缓存前缀。
    # NSA保留裸累积值，以便其索引器可以自行重新添加偏移。
    if seqs_len is not None and len(seqs_len) == bs:  # 如果序列长度列表有效
        prefix_offsets = [  # 计算每个序列的前缀偏移量
            max(int(seqs_len[s]) - extend_seqs_len[s], 0) for s in range(bs)  # 取序列长度减去扩展长度与0的最大值
        ]
    else:  # 否则
        prefix_offsets = [0] * bs  # 前缀偏移量全部为0

    # Per-sequence block sizes: first (L % cp_segment_num) blocks get +1.
    # 每序列的块大小：前 (L % cp_segment_num) 个块各多分配1个token。
    per_seq_block_sizes: List[List[int]] = []  # 每序列的块大小列表
    split_list: List[int] = []  # 所有块的切分大小列表
    for s in range(bs):  # 遍历每个序列
        L = extend_seqs_len[s]  # 当前序列的扩展长度
        base = L // cp_segment_num  # 基础块大小
        rem = L % cp_segment_num  # 余数：前rem个块各多1个token
        blk = [base + 1 if i < rem else base for i in range(cp_segment_num)]  # 构建块大小列表
        per_seq_block_sizes.append(blk)  # 添加到每序列块大小列表
        split_list.extend(blk)  # 展开到总切分列表

    # Per-rank aggregate: this rank owns block r and block (2*cp_size-1-r)
    # of every sequence.
    # 每个rank的聚合：该rank拥有每个序列的第r块和第(2*cp_size-1-r)块。
    per_rank_actual_token = [0] * cp_size  # 初始化每个rank的实际token数
    for r in range(cp_size):  # 遍历每个rank
        total = 0  # 初始化token总数
        for s in range(bs):  # 遍历每个序列
            total += (  # 累加该rank在该序列中的token数
                per_seq_block_sizes[s][r]  # 前半块（第r块）
                + per_seq_block_sizes[s][cp_segment_num - 1 - r]  # 后半块（对称块）
            )
        per_rank_actual_token[r] = total  # 记录该rank的总token数
    max_single_rank = max(per_rank_actual_token) if per_rank_actual_token else 0  # 最大rank token数
    # Kept as cp_size copies so downstream torch.split(x, max_rank_len) still
    # works directly. All entries intentionally identical.
    # 保留为 cp_size 个拷贝，以便下游 torch.split(x, max_rank_len) 仍可直接使用。
    # 所有条目故意设为相同值。
    max_rank_len = [max_single_rank] * cp_size  # 每个rank的最大长度列表

    # Zigzag index selecting which of split_list's bs * cp_segment_num pieces
    # this rank owns, in the order [all_prevs, all_nexts].
    # Zigzag索引：从split_list的 bs*cp_segment_num 个块中选取当前rank拥有的块，
    # 顺序为 [所有前半块, 所有后半块]。
    zigzag_index = list(  # 前半部分索引：每个序列的第cp_rank块
        range(cp_rank, cp_rank + bs * cp_segment_num, cp_segment_num)
    ) + list(  # 后半部分索引：每个序列的第(cp_segment_num-cp_rank-1)块
        range(
            cp_segment_num - cp_rank - 1,
            bs * cp_segment_num,
            cp_segment_num,
        )
    )

    # Reverse index: given the post-allgather concatenation
    #   [rank0_prevs_all_seqs, rank0_nexts_all_seqs,
    #    rank1_prevs_all_seqs, rank1_nexts_all_seqs, ...]
    # produce a permutation that restores [s0_b0..s0_bN, s1_b0..s1_bN, ...].
    # 逆索引：给定全聚集后的拼接顺序
    #   [rank0_所有序列的prev, rank0_所有序列的next,
    #    rank1_所有序列的prev, rank1_所有序列的next, ...]
    # 生成一个排列以恢复 [s0_b0..s0_bN, s1_b0..s1_bN, ...] 的原始顺序。
    cp_reverse_index: List[int] = []  # 逆排列索引列表
    for batch_id in range(bs):  # 遍历每个序列
        cp_reverse_index.extend(  # 添加前向索引（偶数位置块）
            list(range(batch_id, cp_segment_num * bs, 2 * bs))  # 前半块索引
            + list(  # 添加反向索引（奇数位置块，从后向前）
                range(
                    (cp_segment_num - 1) * bs + batch_id,
                    0,
                    -2 * bs,
                )
            )
        )

    # Split sizes matching the post-allgather concatenation order above.
    # 与上述全聚集后拼接顺序匹配的切分大小。
    reverse_split_len: List[int] = []  # 逆向切分长度列表
    for r in range(cp_size):  # 遍历每个rank
        for s in range(bs):  # 遍历每个序列的前半块
            reverse_split_len.append(per_seq_block_sizes[s][r])  # 添加前半块大小
        for s in range(bs):  # 遍历每个序列的后半块
            reverse_split_len.append(per_seq_block_sizes[s][cp_segment_num - 1 - r])  # 添加后半块大小

    # Per-sequence cumulatives used for FA cache_seqlens.
    #   kv_len_prev[s] = sum of seq s's blocks [0..cp_rank] (inclusive).
    #   kv_len_next[s] = sum of seq s's blocks [0..cp_segment_num-cp_rank] (inclusive).
    # 用于FA cache_seqlens的每序列累积值。
    #   kv_len_prev[s] = 序列s的块 [0..cp_rank]（含）之和。
    #   kv_len_next[s] = 序列s的块 [0..cp_segment_num-cp_rank]（含）之和。
    from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp  # 导入DSA预填充CP判断函数

    nsa_mode = is_dsa_enable_prefill_cp()  # 判断是否启用NSA模式
    kv_len_prev_list: List[int] = []  # 前半部分KV长度列表
    kv_len_next_list: List[int] = []  # 后半部分KV长度列表
    actual_seq_q_prev_list: List[int] = []  # 前半部分实际查询序列长度列表
    actual_seq_q_next_list: List[int] = []  # 后半部分实际查询序列长度列表
    for s in range(bs):  # 遍历每个序列
        blk = per_seq_block_sizes[s]  # 当前序列的块大小列表
        cum_prev = sum(blk[: cp_rank + 1])  # 前半部分累积（块0到块cp_rank）
        cum_next = sum(blk[: cp_segment_num - cp_rank])  # 后半部分累积（块0到块cp_segment_num-cp_rank-1）
        # NSA indexer re-adds prefix offset itself; leave bare cumulative.
        # For non-NSA (FlashAttention), bake prefix into cache_seqlens.
        # NSA索引器自行重新添加前缀偏移；保留裸累积值。
        # 对于非NSA (FlashAttention)，将前缀编入cache_seqlens。
        if nsa_mode:  # 如果是NSA模式
            kv_len_prev_list.append(cum_prev)  # 仅添加裸累积值
            kv_len_next_list.append(cum_next)  # 仅添加裸累积值
        else:  # 否则（FlashAttention模式）
            kv_len_prev_list.append(prefix_offsets[s] + cum_prev)  # 添加前缀偏移 + 累积值
            kv_len_next_list.append(prefix_offsets[s] + cum_next)  # 添加前缀偏移 + 累积值
        actual_seq_q_prev_list.append(blk[cp_rank])  # 前半部分查询长度为当前rank的前半块大小
        actual_seq_q_next_list.append(blk[cp_segment_num - cp_rank - 1])  # 后半部分查询长度为当前rank的后半块大小

    # FlashAttention CUDA tensors (device parameterized for unit tests).
    # FlashAttention CUDA张量（设备参数化以支持单元测试）。
    kv_len_prev_tensor = torch.tensor(  # 构建前半部分KV长度的CUDA张量
        kv_len_prev_list, device=device, dtype=torch.int32
    )
    kv_len_next_tensor = torch.tensor(  # 构建后半部分KV长度的CUDA张量
        kv_len_next_list, device=device, dtype=torch.int32
    )
    actual_seq_q_prev_tensor = torch.tensor(  # 构建前半部分实际查询序列长度的CUDA张量
        actual_seq_q_prev_list, device=device, dtype=torch.int32
    )
    actual_seq_q_next_tensor = torch.tensor(  # 构建后半部分实际查询序列长度的CUDA张量
        actual_seq_q_next_list, device=device, dtype=torch.int32
    )
    cu_prev = [0] + list(accumulate(actual_seq_q_prev_list))  # 前半部分累积序列长度（首元素为0）
    cu_next = [0] + list(accumulate(actual_seq_q_next_list))  # 后半部分累积序列长度（首元素为0）
    cu_seqlens_q_prev_tensor = torch.tensor(cu_prev, device=device, dtype=torch.int32)  # 前半部分cu_seqlens张量
    cu_seqlens_q_next_tensor = torch.tensor(cu_next, device=device, dtype=torch.int32)  # 后半部分cu_seqlens张量

    total_q_prev_tokens = cu_prev[-1]  # 前半部分总查询token数
    total_q_next_tokens = cu_next[-1]  # 后半部分总查询token数
    max_seqlen_q_prev = max(actual_seq_q_prev_list) if actual_seq_q_prev_list else 0  # 前半部分最大查询序列长度
    max_seqlen_q_next = max(actual_seq_q_next_list) if actual_seq_q_next_list else 0  # 后半部分最大查询序列长度
    total_seq_lens = sum(extend_seqs_len)  # 总序列长度

    # Cheap invariants: metadata must be a valid permutation spec.
    # - split_list has bs * cp_segment_num pieces (all blocks, all seqs).
    # - zigzag_index has 2 * bs entries (this rank's prev + next per seq).
    # - cp_reverse_index has bs * cp_segment_num entries (reorders the
    #   full allgathered stream back to per-seq-original order).
    # 廉价不变量：元数据必须是有效的排列规范。
    # - split_list 有 bs * cp_segment_num 个元素（所有块，所有序列）。
    # - zigzag_index 有 2 * bs 个元素（该rank每个序列的prev + next）。
    # - cp_reverse_index 有 bs * cp_segment_num 个元素（将完整的全聚集流
    #   重新排列回每序列的原始顺序）。
    assert len(split_list) == bs * cp_segment_num  # 断言split_list长度正确
    assert sum(split_list) == total_seq_lens  # 断言split_list总和等于总序列长度
    assert len(zigzag_index) == 2 * bs  # 断言zigzag_index长度正确
    assert len(cp_reverse_index) == bs * cp_segment_num  # 断言cp_reverse_index长度正确
    assert sorted(cp_reverse_index) == list(range(bs * cp_segment_num))  # 断言cp_reverse_index是0..N-1的排列
    assert sum(per_rank_actual_token) == total_seq_lens  # 断言所有rank的token总数等于总序列长度

    return ContextParallelMetadata(  # 返回构建好的上下文并行元数据对象
        split_list=split_list,  # 切分列表
        zigzag_index=zigzag_index,  # zigzag索引
        cp_reverse_index=cp_reverse_index,  # 逆排列索引
        reverse_split_len=reverse_split_len,  # 逆向切分长度
        per_rank_actual_token=per_rank_actual_token,  # 每个rank实际token数
        max_rank_len=max_rank_len,  # 每个rank的最大长度
        kv_len_prev_tensor=kv_len_prev_tensor,  # 前半部分KV长度张量
        kv_len_next_tensor=kv_len_next_tensor,  # 后半部分KV长度张量
        actual_seq_q_prev_tensor=actual_seq_q_prev_tensor,  # 前半部分实际查询序列长度张量
        actual_seq_q_next_tensor=actual_seq_q_next_tensor,  # 后半部分实际查询序列长度张量
        cu_seqlens_q_prev_tensor=cu_seqlens_q_prev_tensor,  # 前半部分cu_seqlens张量
        cu_seqlens_q_next_tensor=cu_seqlens_q_next_tensor,  # 后半部分cu_seqlens张量
        total_q_prev_tokens=total_q_prev_tokens,  # 前半部分总查询token数
        total_q_next_tokens=total_q_next_tokens,  # 后半部分总查询token数
        max_seqlen_q_prev=max_seqlen_q_prev,  # 前半部分最大查询序列长度
        max_seqlen_q_next=max_seqlen_q_next,  # 后半部分最大查询序列长度
        kv_len_prev_list=kv_len_prev_list,  # 前半部分KV长度CPU列表
        kv_len_next_list=kv_len_next_list,  # 后半部分KV长度CPU列表
        actual_seq_q_prev_list=actual_seq_q_prev_list,  # 前半部分实际查询序列长度CPU列表
        actual_seq_q_next_list=actual_seq_q_next_list,  # 后半部分实际查询序列长度CPU列表
        total_seq_lens=total_seq_lens,  # 总序列长度
        bs=bs,  # 批次大小
    )
