# 文件说明：基于Marlin格式的融合MoE量化计算模块
# 本模块实现了使用Marlin量化格式的MoE（混合专家模型）层前向计算，
# 支持INT4/INT8量化权重、MXFP4格式，包含权重反量化标量类型判断、
# 带限幅的SwiGLU激活函数、以及完整的融合Marlin MoE前向流程

from typing import Optional  # 导入可选类型注解

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch神经网络函数模块

from sglang.srt.utils import is_cuda  # 导入CUDA环境检测函数
from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册装饰器

_is_cuda = is_cuda()  # 检测当前环境是否为CUDA

if _is_cuda:  # 如果是CUDA环境
    from sgl_kernel import moe_sum_reduce  # 导入MoE求和归约内核

    from sglang.jit_kernel.activation import silu_and_mul  # 导入SiLU激活函数内核
    from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm  # 导入Marlin量化MoE GEMM内核


def get_scalar_type(num_bit: int, has_zp: bool, scales: Optional[torch.Tensor] = None):  # 获取量化权重的标量类型
    # 根据量化位数、是否有零点、缩放因子数据类型判断Marlin量化的标量类型
    from sgl_kernel.scalar_type import scalar_types  # 导入标量类型定义

    if (  # 判断是否为MXFP4格式（E2M1浮点4位）
        not has_zp  # 没有零点
        and num_bit == 4  # 4位量化
        and scales is not None  # 缩放因子存在
        and scales.dtype == torch.float8_e8m0fnu  # 缩放因子为E8M0格式
    ):
        return scalar_types.float4_e2m1f  # 返回MXFP4 E2M1浮点类型
    if has_zp:  # 如果有零点
        assert num_bit == 4  # 断言必须为4位量化
        return scalar_types.uint4  # 返回无符号4位整数类型
    else:  # 无零点情况
        return scalar_types.uint4b8 if num_bit == 4 else scalar_types.uint8b128  # 4位返回uint4b8，8位返回uint8b128


def swiglu_limit_func(  # 带限幅的SwiGLU激活函数
    output: torch.Tensor,  # 输出张量
    input: torch.Tensor,  # first half is gate, second half is up  # 前半部分是门控，后半部分是上投影
    swiglu_limit: float = 0.0,  # SwiGLU限幅值，0表示无限幅
) -> None:  # 无返回值，结果写入output
    # 实现带限幅的SwiGLU激活：对gate和up分别限幅后计算silu(gate)*up
    d = input.shape[1] // 2  # 计算隐藏维度的一半，用于分割gate和up
    gate = input[:, :d]  # 取前半部分作为门控值
    up = input[:, d:]  # 取后半部分作为上投影值

    if swiglu_limit > 0:  # 如果限幅值大于0
        gate = torch.clamp(gate, max=swiglu_limit)  # 对gate值进行上限限幅
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)  # 对up值进行双向限幅

    output.copy_(F.silu(gate) * up)  # 计算silu(gate) * up并将结果写入output


@register_custom_op(out_shape="hidden_states")  # 注册为自定义算子，输出形状与hidden_states相同
def fused_marlin_moe(  # 融合Marlin MoE前向计算函数
    hidden_states: torch.Tensor,  # 输入隐藏状态张量
    w1: torch.Tensor,  # 第一组专家权重（门控+上投影）
    w2: torch.Tensor,  # 第二组专家权重（下投影）
    w1_scale: torch.Tensor,  # w1的量化缩放因子
    w2_scale: torch.Tensor,  # w2的量化缩放因子
    gating_output: torch.Tensor,  # 门控输出（softmax之前）
    topk_weights: torch.Tensor,  # Top-k路由权重
    topk_ids: torch.Tensor,  # Top-k专家索引
    global_num_experts: int = -1,  # 全局专家数量，-1表示使用w1中的专家数
    expert_map: Optional[torch.Tensor] = None,  # 专家映射表（用于专家并行）
    g_idx1: Optional[torch.Tensor] = None,  # w1的激活排序索引
    g_idx2: Optional[torch.Tensor] = None,  # w2的激活排序索引
    sort_indices1: Optional[torch.Tensor] = None,  # w1的激活排序输入排列
    sort_indices2: Optional[torch.Tensor] = None,  # w2的激活排序输入排列
    w1_zeros: Optional[torch.Tensor] = None,  # w1的量化零点
    w2_zeros: Optional[torch.Tensor] = None,  # w2的量化零点
    workspace: Optional[torch.Tensor] = None,  # 工作空间张量
    num_bits: int = 8,  # 量化位数
    is_k_full: bool = True,  # K维度是否完整
    inplace: bool = False,  # 是否原地操作
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子
    clamp_limit: Optional[float] = None,  # 激活限幅值
) -> torch.Tensor:  # 返回MoE计算结果张量
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.
    本函数使用两组权重w1和w2以及top-k门控机制计算MoE层。

    Parameters:
    参数：
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
      hidden_states：MoE层的输入张量。
    - w1 (torch.Tensor): The first set of expert weights.
      w1：第一组专家权重。
    - w2 (torch.Tensor): The second set of expert weights.
      w2：第二组专家权重。
    - w1_scale (torch.Tensor): Scale to be used for w1.
      w1_scale：w1的量化缩放因子。
    - w2_scale (torch.Tensor): Scale to be used for w2.
      w2_scale：w2的量化缩放因子。
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
      gating_output：门控操作输出（softmax之前）。
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
      g_idx1：第一组激活排序索引。
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
      g_idx2：第二组激活排序索引。
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
      sort_indices1：第一组激活排序输入排列。
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
      sort_indices2：第二组激活排序输入排列。
    - topk_weights (torch.Tensor): Top-k weights.
      topk_weights：Top-k路由权重。
    - topk_ids (torch.Tensor): Indices of topk-k elements.
      topk_ids：Top-k元素的索引。
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
      w1_zeros：w1的可选零点。
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
      w2_zeros：w2的可选零点。
    - num_bits (int): The number of bits in expert weights quantization.
      num_bits：专家权重量化的位数。

    Returns:
    返回：
    - torch.Tensor: The output tensor after applying the MoE layer.
      应用MoE层后的输出张量。
    """
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size  # 导入MoE块大小对齐函数

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"  # 断言令牌数匹配
    assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch w1"  # 断言隐藏维度与w1匹配
    assert hidden_states.shape[1] == w2.shape[2] // (  # 断言隐藏维度与w2匹配
        num_bits // 2  # 根据量化位数计算预期维度
    ), "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"  # 断言输入张量内存连续
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"  # 断言w1权重内存连续
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"  # 断言w2权重内存连续
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]  # 断言输入数据类型为FP16或BF16
    is_mxfp4_marlin = (  # 判断是否为MXFP4 Marlin格式
        num_bits == 4  # 4位量化
        and w1_zeros is None  # 无w1零点
        and w2_zeros is None  # 无w2零点
        and w1_scale.dtype == torch.float8_e8m0fnu  # w1缩放因子为E8M0格式
        and w2_scale.dtype == torch.float8_e8m0fnu  # w2缩放因子为E8M0格式
    )
    if is_mxfp4_marlin:  # 如果是MXFP4 Marlin格式
        assert hidden_states.dtype == torch.bfloat16, (  # 断言输入必须为BF16
            "MXFP4 Marlin with E8M0 scales is only instantiated for bfloat16 "  # MXFP4 Marlin仅支持BF16激活
            f"activations, got {hidden_states.dtype}"  # 显示实际数据类型
        )
    else:  # 非MXFP4 Marlin格式
        assert (  # 断言hidden_states与w1_scale数据类型一致
            hidden_states.dtype == w1_scale.dtype
        ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w1_scale.dtype ({w1_scale.dtype})"
        assert (  # 断言hidden_states与w2_scale数据类型一致
            hidden_states.dtype == w2_scale.dtype
        ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w2_scale.dtype ({w2_scale.dtype})"
    assert num_bits in [4, 8]  # 断言量化位数为4或8

    M, K = hidden_states.shape  # 获取令牌数M和隐藏维度K
    E = w1.shape[0]  # 获取专家数量E
    N = w2.shape[1] * 16  # 获取中间维度N（Marlin格式需乘16）
    topk = topk_ids.shape[1]  # 获取top-k值

    # M block size selection logic
    # M块大小选择逻辑
    # TODO: tune this further for specific models
    # TODO: 针对特定模型进一步调优
    for block_size_m in [8, 16, 32, 48, 64]:  # 遍历候选块大小
        if M * topk / E / block_size_m < 0.9:  # 如果平均每个块的处理量小于0.9
            break  # 选择当前块大小

    if global_num_experts == -1:  # 如果未指定全局专家数
        global_num_experts = E  # 使用w1中的专家数量
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(  # 对令牌按块大小对齐排列
        topk_ids, block_size_m, global_num_experts  # 传入topk ID、块大小和全局专家数
    )

    if workspace is None:  # 如果未提供工作空间
        max_workspace_size = (max(2 * N, K) // 64) * (  # 计算最大工作空间大小
            sorted_token_ids.size(0) // block_size_m  # 根据排序令牌数和块大小计算
        )
        device = hidden_states.device  # 获取计算设备
        sms = torch.cuda.get_device_properties(device).multi_processor_count  # 获取SM数量
        max_workspace_size = min(max_workspace_size, sms * 4)  # 限制工作空间不超过SM数的4倍
        workspace = torch.zeros(  # 创建工作空间张量
            max_workspace_size, dtype=torch.int, device=device, requires_grad=False  # 整型，不需要梯度
        )

    scalar_type1 = get_scalar_type(num_bits, w1_zeros is not None, w1_scale)  # 获取w1的标量类型
    scalar_type2 = get_scalar_type(num_bits, w2_zeros is not None, w2_scale)  # 获取w2的标量类型

    intermediate_cache2 = torch.empty(  # 创建中间缓存2（下投影输入）
        (M * topk_ids.shape[1], N),  # 形状为 [M*topk, N]
        device=hidden_states.device,  # 设备与输入相同
        dtype=hidden_states.dtype,  # 数据类型与输入相同
    )
    intermediate_cache13 = torch.empty(  # 创建中间缓存13（w1和w3输出共享的缓冲区）
        (M * topk_ids.shape[1] * max(2 * N, K),),  # 大小取2*N和K中的较大值
        device=hidden_states.device,  # 设备与输入相同
        dtype=hidden_states.dtype,  # 数据类型与输入相同
    )
    intermediate_cache1 = intermediate_cache13[: M * topk_ids.shape[1] * 2 * N]  # 从共享缓冲区中切出w1输出部分
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N)  # reshape为 [M*topk, 2*N]
    intermediate_cache3 = intermediate_cache13[: M * topk_ids.shape[1] * K]  # 从共享缓冲区中切出w3输出部分
    intermediate_cache3 = intermediate_cache3.view(-1, K)  # reshape为 [M*topk, K]

    use_atomic_add = (  # 判断是否使用原子加法
        hidden_states.dtype == torch.half  # FP16数据类型
        or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9  # 或计算能力>=9.0
    ) and (not is_mxfp4_marlin)  # 且非MXFP4 Marlin格式

    intermediate_cache1 = moe_wna16_marlin_gemm(  # 执行w1的Marlin量化GEMM计算
        hidden_states,  # 输入隐藏状态
        intermediate_cache1,  # 输出缓冲区
        w1,  # w1权重
        None,  # b_bias_or_none  # 偏置（无）
        w1_scale,  # w1缩放因子
        None,  # global_scale_or_none  # 全局缩放（无）
        w1_zeros,  # w1零点
        g_idx1,  # w1分组索引
        sort_indices1,  # w1排序索引
        workspace,  # 工作空间
        sorted_token_ids,  # 排序后的令牌ID
        expert_ids,  # 专家ID
        num_tokens_post_padded,  # 填充后的令牌数
        topk_weights,  # topk权重
        moe_block_size=block_size_m,  # MoE块大小
        top_k=topk,  # top-k值
        mul_topk_weights=False,  # 不在GEMM中乘以topk权重
        is_ep=expert_map is not None,  # 是否为专家并行模式
        b_q_type=scalar_type1,  # w1的标量类型
        size_m=M,  # M维度大小
        size_n=2 * N,  # N维度大小（gate+up拼接）
        size_k=K,  # K维度大小
        is_k_full=is_k_full,  # K维度是否完整
        use_atomic_add=use_atomic_add,  # 是否使用原子加法
        use_fp32_reduce=True,  # 使用FP32精度进行归约
        is_zp_float=False,  # 零点是否为浮点型
    )

    if clamp_limit is not None:  # 如果指定了限幅值
        swiglu_limit_func(  # 使用带限幅的SwiGLU激活函数
            intermediate_cache2,  # 输出张量
            intermediate_cache1.view(-1, 2 * N),  # 输入张量（reshape为2*N列）
            clamp_limit,  # 限幅值
        )
    else:  # 无限幅
        silu_and_mul(intermediate_cache1.view(-1, 2 * N), intermediate_cache2)  # 使用标准SiLU激活函数

    if expert_map is not None:  # 如果处于专家并行模式
        intermediate_cache3.zero_()  # 将w3输出缓冲区清零（用于原子加法累加）

    intermediate_cache3 = moe_wna16_marlin_gemm(  # 执行w2的Marlin量化GEMM计算（下投影）
        intermediate_cache2,  # 激活后的中间结果
        intermediate_cache3,  # 输出缓冲区
        w2,  # w2权重
        None,  # b_bias_or_none  # 偏置（无）
        w2_scale,  # w2缩放因子
        None,  # global_scale_or_none  # 全局缩放（无）
        w2_zeros,  # w2零点
        g_idx2,  # w2分组索引
        sort_indices2,  # w2排序索引
        workspace,  # 工作空间
        sorted_token_ids,  # 排序后的令牌ID
        expert_ids,  # 专家ID
        num_tokens_post_padded,  # 填充后的令牌数
        topk_weights,  # topk权重
        moe_block_size=block_size_m,  # MoE块大小
        top_k=1,  # top-k值为1（每个slot已确定专家）
        mul_topk_weights=True,  # 在GEMM中乘以topk权重
        is_ep=expert_map is not None,  # 是否为专家并行模式
        b_q_type=scalar_type2,  # w2的标量类型
        size_m=M * topk,  # M维度（令牌数*topk）
        size_n=K,  # N维度（隐藏维度）
        size_k=N,  # K维度（中间维度）
        is_k_full=is_k_full,  # K维度是否完整
        use_atomic_add=use_atomic_add,  # 是否使用原子加法
        use_fp32_reduce=True,  # 使用FP32精度进行归约
        is_zp_float=False,  # 零点是否为浮点型
    ).view(-1, topk, K)  # reshape为 [M, topk, K]

    output = hidden_states if inplace else torch.empty_like(hidden_states)  # 如果原地操作则复用输入，否则创建新张量

    if is_mxfp4_marlin:  # 如果是MXFP4 Marlin格式
        return torch.sum(intermediate_cache3, dim=1, out=output)  # 在topk维度上求和
    else:  # 非MXFP4 Marlin格式
        if routed_scaling_factor is None:  # 如果未指定路由缩放因子
            routed_scaling_factor = 1.0  # 默认缩放因子为1.0

        moe_sum_reduce(  # 执行MoE求和归约（应用路由缩放因子）
            intermediate_cache3,  # 输入张量
            output,  # 输出张量
            routed_scaling_factor,  # 路由缩放因子
        )
        return output  # 返回最终输出
