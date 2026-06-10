# DSA（DeepSeek Attention）工具函数模块
# 本文件提供了DeepSeek稀疏注意力（DSA）相关的工具函数，包括：
# - aiter预洗牌分页MQA可用性检测
# - DSA序列长度计算和填充
# - 上下文并行（CP）轮询分割逻辑
# - CP模式下token数量的填充计算

from functools import lru_cache  # 导入LRU缓存装饰器
from typing import TYPE_CHECKING, List, Tuple, Union  # 导入类型注解工具

import torch  # 导入PyTorch库
import triton  # 导入Triton库
import triton.language as tl  # 导入Triton语言

from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.layers.dp_attention import (  # 导入数据并行注意力相关函数
    DpPaddingMode,  # DP填充模式
    get_attention_cp_rank,  # 获取注意力CP秩
    get_attention_cp_size,  # 获取注意力CP大小
    get_attention_dp_rank,  # 获取注意力DP秩
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数
from sglang.srt.utils import get_bool_env_var, is_hip  # 导入环境变量和平台检测工具
from sglang.srt.utils.common import ceil_align, ceil_div  # 导入向上对齐和向上整除工具


@lru_cache(maxsize=1)  # 使用LRU缓存，结果只计算一次
def aiter_can_use_preshuffle_paged_mqa() -> bool:  # 检测aiter预洗牌分页MQA是否可用
    """Whether aiter's preshuffle paged MQA / cache kernels can be used on this runtime.
    # aiter的预洗牌分页MQA/缓存内核是否可在当前运行时使用。

    aiter's ``deepgemm_fp8_paged_mqa_logits`` only supports ``KVBlockSize > 1`` and
    ``Preshuffle=True`` on its gluon kernel path. The gluon path is enabled when
    Triton >= 3.5.0, OR when ``AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`` is set
    (which additionally requires that the AOT gluon kernel artifacts ship inside
    the aiter wheel/image). Otherwise aiter asserts ``KVBlockSize == 1`` and
    refuses ``Preshuffle=True``.
    # aiter的``deepgemm_fp8_paged_mqa_logits``仅在其gluon内核路径上支持``KVBlockSize > 1``和
    # ``Preshuffle=True``。当Triton >= 3.5.0时，或设置了``AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1``
    # 时（还需要AOT gluon内核构件随aiter wheel/镜像一起发布），将启用gluon路径。
    # 否则aiter断言``KVBlockSize == 1``并拒绝``Preshuffle=True``。

    sglang's DSA indexer uses this single decision to pick:
    # sglang的DSA索引器使用此单一决策来选择：
      * ``page_size``: 64 (preshuffle) vs 1 (legacy) on ROCm
      * ``Preshuffle`` / ``preshuffle`` flags on the aiter MQA + cache kernels
      * ``get_page_table_64`` vs ``get_page_table_1`` on the metadata
      * whether ``GetKAndS.execute`` uses the aiter or the triton implementation
      * ``page_size``：ROCm上64（预洗牌）对1（传统）
      * aiter MQA + 缓存内核上的``Preshuffle`` / ``preshuffle``标志
      * 元数据上的``get_page_table_64``对``get_page_table_1``
      * ``GetKAndS.execute``是否使用aiter或triton实现

    The result is cached so the cost is paid once per process.
    # 结果被缓存，因此每个进程只付出一次代价。

    Set ``SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1`` to force the legacy path even when
    the gluon kernel would otherwise be available (useful for CI bisection).
    ``SGLANG_NSA_HIP_DISABLE_PRESHUFFLE`` is a deprecated alias.
    # 设置``SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1``以强制使用传统路径，即使gluon内核
    # 本来可用（适用于CI二分法）。``SGLANG_NSA_HIP_DISABLE_PRESHUFFLE``是已弃用的别名。
    """
    if not is_hip():  # 如果不是HIP平台
        return False  # 返回False
    if not get_bool_env_var("SGLANG_USE_AITER"):  # 如果未启用aiter
        return False  # 返回False
    if envs.SGLANG_DSA_HIP_DISABLE_PRESHUFFLE.get():  # 如果禁用了预洗牌
        return False  # 返回False
    if get_bool_env_var("AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS"):  # 如果启用了AOT gluon
        return True  # 返回True
    try:  # 尝试检查Triton版本
        from packaging.version import Version  # 导入版本工具

        return Version(Version(triton.__version__).base_version) >= Version("3.5.0")  # Triton版本>=3.5.0则可用
    except Exception:  # 捕获异常
        return False  # 默认不可用


if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息类


def compute_dsa_seqlens(original_seq_lens, dsa_index_topk: int):  # 计算DSA序列长度（裁剪到topk）
    return original_seq_lens.clamp(max=dsa_index_topk)  # 将序列长度裁剪到最大topk值


def is_dsa_enable_prefill_cp():  # 检查是否启用DSA prefill上下文并行
    return get_global_server_args().enable_dsa_prefill_context_parallel  # 返回全局参数中的设置


def is_dsa_prefill_cp_in_seq_split():  # 检查是否使用序列内分割的CP模式
    return (  # 返回是否启用且模式为"in-seq-split"
        is_dsa_enable_prefill_cp()  # 启用DSA prefill CP
        and get_global_server_args().dsa_prefill_cp_mode == "in-seq-split"  # 模式为序列内分割
    )


def is_dsa_prefill_cp_round_robin_split():  # 检查是否使用轮询分割的CP模式
    return (  # 返回是否启用且模式为"round-robin-split"
        is_dsa_enable_prefill_cp()  # 启用DSA prefill CP
        and get_global_server_args().dsa_prefill_cp_mode == "round-robin-split"  # 模式为轮询分割
    )


def can_dsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):  # 检查是否可以对当前批次执行轮询分割
    if not forward_batch.forward_mode.is_context_parallel_extend():  # 如果不是上下文并行扩展模式
        return False  # 返回False
    cp_size = get_attention_cp_size()  # 获取CP大小
    seq_len = sum(forward_batch.extend_seq_lens_cpu)  # 计算总序列长度
    return (  # 返回是否满足轮询分割条件
        is_dsa_prefill_cp_round_robin_split()  # 启用了轮询分割
        and seq_len > 0  # 序列长度大于0
        and seq_len >= cp_size  # 序列长度不小于CP大小
        and cp_size > 1  # CP大小大于1
    )


def dsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):  # 按轮询规则分割数据到CP秩
    """
    # for round-robin-split, split the tokens evenly according to the rule of token_idx % cp_size.
    # 对于轮询分割，按照token_idx % cp_size的规则均匀分割token。
    |   +-----------before split------------+|
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |
    |   +--------------result-------------------+
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |   +-------------------------+
    """
    cp_size = get_attention_cp_size()  # 获取CP大小
    cp_rank = get_attention_cp_rank()  # 获取当前CP秩
    if isinstance(input_, (tuple, list)):  # 如果输入是列表或元组
        indices = range(cp_rank, len(input_), cp_size)  # 计算当前秩对应的索引
        return input_[indices]  # 返回选中的元素

    tokens = len(input_)  # 获取token数量
    if tokens % cp_size != 0:  # 如果token数不能被CP大小整除
        cur_len = tokens // cp_size + (tokens % cp_size > cp_rank)  # 当前秩分配的长度
        if cur_len == 0:  # 如果当前秩没有分配到token
            return input_.new_empty(0, *input_.shape[1:])  # 返回空张量
        indices = torch.arange(cp_rank, tokens, cp_size, device=input_.device)  # 生成索引
        return input_[indices]  # 返回选中的元素

    # for torch device tensor  # 对于torch设备张量
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()  # 重塑并选择当前秩对应的数据


def cal_padded_tokens(forward_batch: "ForwardBatch"):  # 计算填充后的token数量
    # Consistent with the padding calculation logic in ForwardBatch.prepare_mlp_sync_batch,
    # calculate the actual token length after padding when attn_tp_size > 1 or in the MAX_LEN padding mode.
    # 与ForwardBatch.prepare_mlp_sync_batch中的填充计算逻辑一致，
    # 计算当attn_tp_size > 1或MAX_LEN填充模式下的实际token长度。
    global_num_tokens = forward_batch.global_num_tokens_cpu.copy()  # 复制全局token数量
    sync_group_size = len(global_num_tokens)  # 同步组大小
    attn_cp_size = get_attention_cp_size()  # 获取注意力CP大小
    for i in range(sync_group_size):  # 遍历同步组
        # Must match ForwardBatch.prepare_mlp_sync_batch, which pads to
        # attn_cp_size * 2 (tokens are split into 2 * CP chunks for load balance).
        # 必须与ForwardBatch.prepare_mlp_sync_batch匹配，后者填充到
        # attn_cp_size * 2（token被分成2 * CP块以实现负载均衡）。
        global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_cp_size * 2)  # 对齐到CP大小的两倍
    dp_padding_mode = DpPaddingMode.get_dp_padding_mode(  # 获取DP填充模式
        forward_batch.is_extend_in_batch, global_num_tokens
    )
    if dp_padding_mode.is_max_len():  # 如果是最大长度填充模式
        tokens = max(global_num_tokens)  # 使用最大token数
    elif len(global_num_tokens) > 1:  # 如果有多个同步组
        tokens = global_num_tokens[get_attention_dp_rank()]  # 使用当前DP秩的token数
    else:  # 否则
        tokens = global_num_tokens[0]  # 使用唯一组的token数
    if can_dsa_prefill_cp_round_robin_split(forward_batch):  # 如果可以轮询分割
        tokens = ceil_div(tokens, attn_cp_size)  # 整除向上取整
    return tokens  # 返回填充后的token数


def pad_dsa_cache_seqlens(forward_batch: "ForwardBatch", dsa_cache_seqlens):  # 填充DSA缓存序列长度
    attn_cp_size = get_attention_cp_size()  # 获取CP大小
    needs_cp_pad = attn_cp_size > 1 and can_dsa_prefill_cp_round_robin_split(  # 是否需要CP填充
        forward_batch
    )
    needs_dp_pad = forward_batch.global_num_tokens_cpu is not None  # 是否需要DP填充
    if not needs_cp_pad and not needs_dp_pad:  # 如果都不需要填充
        return dsa_cache_seqlens  # 直接返回
    tokens = cal_padded_tokens(forward_batch)  # 计算填充后的token数
    pad_len = tokens - dsa_cache_seqlens.shape[0]  # 计算需要填充的长度
    if pad_len > 0:  # 如果需要填充
        dsa_cache_seqlens = torch.cat(  # 拼接零填充
            [
                dsa_cache_seqlens,  # 原始序列长度
                dsa_cache_seqlens.new_zeros(pad_len, *dsa_cache_seqlens.shape[1:]),  # 零填充
            ]
        )
    return dsa_cache_seqlens  # 返回填充后的序列长度


def can_dsa_cp_split(seq_len: int, cp_size: int, use_dsa: bool, forward_batch):  # 检查是否可以进行DSA CP分割
    if is_dsa_prefill_cp_round_robin_split():  # 如果使用轮询分割
        cur_cp_seq_len = seq_len // cp_size  # 当前CP序列长度
        assert (  # 断言序列长度可被CP大小整除
            seq_len % cp_size == 0
        ), f"seq_len {seq_len} is not divisible by cp_size {cp_size} when dsa_prefill_cp_mode is round-robin-split"
    else:  # 否则使用序列内分割
        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        # Note: (self.cp_size * 2) To achieve load balancing for seq computation,
        # the seq data needs to be divided and recombined at twice the size of cp_size.
        # TODO 当前仅支持prefill batch=1且len(input_ids) > self.cp_size * 2
        # 注意：(self.cp_size * 2) 为了实现序列计算的负载均衡，
        # 序列数据需要以cp_size两倍的大小进行分割和重组。
        cur_cp_seq_len = seq_len // (cp_size * 2)  # 当前CP序列长度
    if (  # 如果满足以下所有条件
        cur_cp_seq_len != 0  # CP序列长度非零
        and cp_size > 1  # CP大小大于1
        and use_dsa  # 使用DSA
        and forward_batch.forward_mode.is_context_parallel_extend()  # 是上下文并行扩展
        and is_dsa_enable_prefill_cp()  # 启用了DSA prefill CP
        and sum(forward_batch.extend_seq_lens_cpu) >= cp_size  # 总扩展序列长度不小于CP大小
    ):
        return True  # 可以分割
    else:  # 否则
        return False  # 不可以分割


@triton.jit  # Triton JIT编译内核
def dsa_cp_round_robin_split_q_seqs_kernel(  # 轮询分割查询序列的Triton内核
    in_seqs_ptr,  # 输入序列长度指针
    out_seqs_ptr,  # 输出序列长度指针
    bs_idx_ptr,  # 批次索引指针
    tokens: tl.constexpr,  # token数量（编译时常量）
    cp_size: tl.constexpr,  # CP大小（编译时常量）
    cp_rank: tl.constexpr,  # CP秩（编译时常量）
):
    extra_seq = 0  # 额外序列计数
    bs_idx = 0  # 批次索引
    for bs in range(tokens):  # 遍历每个token
        cur_len = tl.load(in_seqs_ptr + bs)  # 加载当前序列长度
        cur_len += extra_seq  # 加上额外序列
        cur_seq = cur_len // cp_size + (cur_len % cp_size > cp_rank)  # 计算当前秩分配的长度
        if cur_seq > 0:  # 如果分配的长度大于0
            tl.store(bs_idx_ptr + bs_idx, bs)  # 存储批次索引
            tl.store(out_seqs_ptr + bs_idx, cur_seq)  # 存储分配的序列长度
            bs_idx += 1  # 递增批次索引
        extra_seq = cur_len - cur_seq * cp_size  # 更新额外序列计数


def dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs):  # CPU端轮询分割查询序列
    cp_size = get_attention_cp_size()  # 获取CP大小
    cp_rank = get_attention_cp_rank()  # 获取当前CP秩
    extra_seq = 0  # 额外序列计数
    q_seqs = []  # 查询序列列表
    for bs, cur_len in enumerate(extend_seqs):  # 遍历每个序列
        cur_len += extra_seq  # 加上额外序列
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)  # 计算当前秩分配的长度
        q_seqs.append(cur_seq)  # 添加到列表
        extra_seq = cur_len - cur_seq * cp_size  # 更新额外序列计数
    bs_idx = list([i for i, x in enumerate(q_seqs) if x > 0])  # 找出非零长度的批次索引
    q_seqs = [q_len for q_len in q_seqs if q_len > 0]  # 过滤零长度序列
    return q_seqs, bs_idx  # 返回查询序列和批次索引


def dsa_cp_round_robin_split_q_seqs(  # 轮询分割查询序列（GPU + CPU协同）
    extend_seqs_cpu, extend_seqs  # CPU端序列长度，GPU端序列长度
) -> Tuple[List, torch.Tensor, List, torch.Tensor]:  # 返回CPU/GPU端的长度和索引
    """
    round-robin-split distributes tokens across ranks based on token_idx % cp_size.
    # 轮询分割根据token_idx % cp_size将token分配到各个秩。

    Return:  # 返回值：
    ret_q_lens_cpu(List) and ret_q_lens(torch.Tensor): the partitioned length (excluding zeros) on the current cp rank
        for each sequence after distribution across cp ranks.
    # ret_q_lens_cpu(List)和ret_q_lens(torch.Tensor)：跨CP秩分配后，当前CP秩上每个序列的
    # 分区长度（排除零值）。
    bs_idx_cpu(List) and bs_idx(torch.Tensor): marks which sequences are ultimately selected,
        i.e., those with a partitioned length greater than zero.
    # bs_idx_cpu(List)和bs_idx(torch.Tensor)：标记最终被选中的序列，即分区长度大于零的序列。
    """
    cp_size = get_attention_cp_size()  # 获取CP大小
    cp_rank = get_attention_cp_rank()  # 获取当前CP秩
    # len(ret_q_lens_cpu) == len(bs_idx_cpu)  # 返回长度等于批次索引长度
    ret_q_lens_cpu, bs_idx_cpu = dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs_cpu)  # CPU端计算
    ret_q_lens = torch.empty(  # GPU端分配的长度
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty(  # GPU端批次索引
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
    )
    grid = (1,)  # 网格大小
    dsa_cp_round_robin_split_q_seqs_kernel[grid](  # 调用Triton内核
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx  # 返回CPU/GPU端结果


def dsa_use_prefill_cp(forward_batch, dsa_enable_prefill_cp=None):  # 检查当前批次是否使用prefill CP
    if dsa_enable_prefill_cp is None:  # 如果未传入参数
        dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()  # 从全局参数获取
    if (  # 如果满足以下所有条件
        forward_batch.attn_cp_metadata is not None  # CP元数据不为空
        and dsa_enable_prefill_cp  # 启用了DSA prefill CP
        and forward_batch.forward_mode.is_context_parallel_extend()  # 是上下文并行扩展
    ):
        return True  # 使用prefill CP
    else:  # 否则
        return False  # 不使用prefill CP
