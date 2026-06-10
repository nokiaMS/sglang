# 文件说明：融合MoE路由器实现，包含基于CUDA Core和Tensor Core的两种Triton内核。
# CUDA Core版本适用于小批量场景，逐元素计算路由logits；Tensor Core版本适用于大批量场景，利用矩阵乘法加速。
# 还提供了路由选择器（shim）根据输入大小自动选择内核版本，以及FusedMoeRouter封装类。
from typing import Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
import triton  # 导入Triton
import triton.language as tl  # 导入Triton语言

from sglang.srt.layers.moe.topk import fused_topk  # 导入融合TopK函数
from sglang.srt.utils import is_hip  # 导入HIP平台检测

_is_hip = is_hip()  # 检测是否为HIP平台


@triton.jit  # Triton JIT编译内核
def fused_moe_router_cudacore_kernel(  # 基于CUDA Core的融合MoE路由器内核
    input_ptr,  # input (bs, hidden_dim)  # 输入指针（batch_size, hidden_dim）
    moe_router_weight_ptr,  # input (num_experts, hidden_dim)  # 路由权重指针（专家数, hidden_dim）
    topk_weights_ptr,  # output (bs, topk)  # TopK权重输出指针
    topk_ids_ptr,  # output (bs, topk)  # TopK ID输出指针
    correction_bias_ptr,  # 校正偏置指针
    is_correction_bias: tl.constexpr,  # 是否有校正偏置（编译时常量）
    num_experts: tl.constexpr,  # 专家数量（编译时常量）
    topk: tl.constexpr,  # TopK值（编译时常量）
    moe_softcapping: tl.constexpr,  # 路由软截断值（编译时常量）
    moe_renormalize: tl.constexpr,  # not supported  # 是否重归一化（不支持）
    hidden_dim: tl.constexpr,  # 隐藏维度（编译时常量）
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
):
    pid = tl.program_id(axis=0)  # 获取当前线程的程序ID

    offsets = tl.arange(0, BLOCK_SIZE)  # 生成块内偏移序列
    mask = offsets < hidden_dim  # 生成隐藏维度掩码

    # moe_router_weight is k major  # 路由权重是K主序存储
    expert_offsets = tl.arange(0, num_experts)[:, None]  # 专家偏移量
    router_mask = mask[None, :]  # 路由权重掩码
    w_router = tl.load(  # 加载路由权重
        moe_router_weight_ptr + expert_offsets * hidden_dim + offsets[None, :],  # 计算权重地址
        mask=router_mask,  # 应用掩码
        other=0.0,  # 越界填充0
    )

    x = tl.load(input_ptr + pid * hidden_dim + offsets, mask=mask, other=0.0)  # 加载输入向量

    # todo: tl.dot?  # 待办：使用tl.dot加速？
    logits = tl.sum((w_router.to(tl.float32) * x[None, :].to(tl.float32)), axis=-1)  # 计算路由logits（逐元素乘法求和）

    # logit softcap  # logit软截断
    if moe_softcapping == 0:  # 如果不使用软截断
        logits_softcapped = logits  # 直接使用原始logits
    else:  # 使用软截断
        logits_scaled = logits / moe_softcapping  # 缩放logits
        exped = tl.exp(2 * logits_scaled)  # 计算指数
        top = exped - 1  # tanh分子上半部分
        bottom = exped + 1  # tanh分母
        logits_softcapped = top / bottom * moe_softcapping  # tanh截断

    # Add bias after softcapping  # 软截断后添加偏置
    if is_correction_bias:  # 如果有校正偏置
        bias = tl.load(correction_bias_ptr + tl.arange(0, num_experts))  # 加载偏置
        logits_softcapped = logits_softcapped + bias  # 加上偏置

    # topk  # TopK选择
    # assert 1 <= topk <= num_experts  # 断言：1 <= topk <= 专家数

    # 5.38 us  # 性能基准：5.38微秒

    top1 = tl.argmax(logits_softcapped, axis=0)  # 找到top1专家索引
    tl.store(topk_ids_ptr + pid * topk + 0, top1)  # 5.63 us  # 存储top1 ID

    top1_v = tl.max(logits_softcapped, axis=0)  # top1的logit值
    invsumexp = 1.0 / tl.sum(tl.exp(logits_softcapped - top1_v), axis=0)  # softmax归一化因子

    tl.store(  # 存储top1权重
        topk_weights_ptr + pid * topk + 0,  # top1权重地址
        invsumexp,  # softmax概率
    )  # 5.73 us  # 性能基准

    if topk >= 2:  # 如果topk >= 2
        top2 = tl.argmax(  # 找到top2专家索引
            tl.where(  # 排除top1
                tl.arange(0, num_experts) != top1, logits_softcapped, float("-inf")  # top1位置设为负无穷
            ),
            axis=0,
        )
        tl.store(topk_ids_ptr + pid * topk + 1, top2)  # 存储top2 ID
        top2_v = tl.sum(logits_softcapped * (tl.arange(0, num_experts) == top2), axis=0)  # top2的logit值
        tl.store(  # 存储top2权重
            topk_weights_ptr + pid * topk + 1,  # top2权重地址
            tl.exp(top2_v - top1_v) * invsumexp,  # softmax概率
        )  # 5.95us  # 性能基准

    # probably slow  # 可能较慢
    if topk > 2:  # 如果topk > 2
        topk_mask = tl.full(logits_softcapped.shape, 1.0, dtype=logits_softcapped.dtype)  # 初始化掩码全1
        topk_mask = tl.where(  # 掩码top1
            tl.arange(0, num_experts) != top1, topk_mask, float("-inf")  # top1位置设为负无穷
        )
        topk_mask = tl.where(  # 掩码top2
            tl.arange(0, num_experts) != top2, topk_mask, float("-inf")  # top2位置设为负无穷
        )
        for i in range(2, topk):  # 逐个寻找top3及以后的专家
            topi = tl.argmax(logits_softcapped + topk_mask, axis=0)  # 找到当前最大值索引
            topk_mask = tl.where(  # 掩码当前专家
                tl.arange(0, num_experts) != topi, topk_mask, float("-inf")  # 当前专家设为负无穷
            )
            tl.store(topk_ids_ptr + pid * topk + i, topi)  # 存储topi ID
            topi_v = tl.sum(  # topi的logit值
                logits_softcapped * (tl.arange(0, num_experts) == topi), axis=0  # 提取对应位置的值
            )
            tl.store(  # 存储topi权重
                topk_weights_ptr + pid * topk + i,  # topi权重地址
                tl.exp(topi_v - top1_v) * invsumexp,  # softmax概率
            )
    # assert not moe_renormalize, "moe weight renormalization not implemented"  # 断言：不支持重归一化


def fused_moe_router_cudacore(  # CUDA Core版MoE路由器主机函数
    x: torch.Tensor,  # 输入张量（batch_size, hidden_dim）
    router_weight: torch.Tensor,  # 路由权重（num_experts, hidden_dim）
    topk: int,  # TopK值
    moe_softcapping: float,  # 软截断值
    correction_bias: Optional[torch.Tensor] = None,  # 校正偏置
):
    assert len(x.shape) == 2 and x.shape[1] == router_weight.shape[1]  # 断言输入和权重第二维匹配
    bs, hidden_dim = x.shape  # 解析batch_size和hidden_dim
    num_experts = router_weight.shape[0]  # 获取专家数

    # router_logits = torch.empty((bs, num_experts), dtype=torch.float32, device=x.device)  # 注释掉的中间变量
    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)  # 分配TopK权重
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)  # 分配TopK ID
    is_correction_bias = correction_bias is not None  # 判断是否有校正偏置

    max_warps = 16 if _is_hip else 32  # HIP平台最大16个warp，否则32
    config = {  # 内核配置
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),  # 块大小为hidden_dim的下一个2的幂
        "num_warps": max(  # warp数
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4  # 在4和max_warps之间
        ),
    }

    fused_moe_router_cudacore_kernel[(bs,)](  # 启动CUDA Core路由器内核
        x,  # 输入
        router_weight,  # 路由权重
        topk_weights,  # TopK权重
        topk_ids,  # TopK ID
        correction_bias,  # 校正偏置
        is_correction_bias=is_correction_bias,  # 是否有偏置
        num_experts=num_experts,  # 专家数
        topk=topk,  # TopK值
        moe_softcapping=moe_softcapping,  # 软截断值
        moe_renormalize=False,  # 不重归一化
        hidden_dim=hidden_dim,  # 隐藏维度
        **config,  # 额外配置
    )

    return topk_weights, topk_ids  # 返回TopK权重和ID


@triton.jit  # Triton JIT编译内核
def fused_moe_router_tensorcore_kernel(  # 基于Tensor Core的融合MoE路由器内核
    a_ptr,  # input (bs, hidden_dim)  # 输入指针（batch_size, hidden_dim）
    b_ptr,  # input (num_experts, hidden_dim)  # 路由权重指针（专家数, hidden_dim）
    topk_weights_ptr,  # output (bs, topk)  # TopK权重输出指针
    topk_ids_ptr,  # output (bs, topk)  # TopK ID输出指针
    bs,  # batch size  # 批量大小
    num_experts: tl.constexpr,  # 专家数量（编译时常量）
    topk: tl.constexpr,  # only support topk <= 2  # TopK值（仅支持topk <= 2）
    moe_softcapping: tl.constexpr,  # 软截断值（编译时常量）
    moe_renormalize: tl.constexpr,  # not supported  # 重归一化（不支持）
    correction_bias_ptr,  # 校正偏置指针
    is_correction_bias: tl.constexpr,  # 是否有校正偏置（编译时常量）
    K: tl.constexpr,  # K维度（hidden_dim，编译时常量）
    BLOCK_SIZE_M: tl.constexpr,  # M维度块大小（编译时常量）
    BLOCK_SIZE_N: tl.constexpr,  # N维度块大小（编译时常量）
    BLOCK_SIZE_K: tl.constexpr,  # K维度块大小（编译时常量）
    stride_am: tl.constexpr,  # A的行步长（编译时常量）
    stride_bn: tl.constexpr,  # B的行步长（编译时常量）
    dp_attn_workaround_flag: tl.constexpr,  # DP注意力变通标志（编译时常量）
):

    # 1. get block id  # 1. 获取块ID
    pid = tl.program_id(axis=0)  # 当前块的程序ID

    # 2. create pointers for the first block of A and B  # 2. 创建A和B第一个块的指针
    # 2.1. setup a_ptrs with offsets in m and k  # 2.1. 设置A指针的m和k偏移
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]  # M维度偏移
    bs_mask = offs_m < bs  # M维度掩码
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]  # K维度偏移
    a_ptrs = a_ptr + (offs_m * stride_am + offs_k)  # 计算A的地址

    # 2.2. setup b_ptrs with offsets in k and n.  # 2.2. 设置B指针的k和n偏移
    #      Note: b matrix is k-major.  # 注意：B矩阵是K主序存储
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]  # K维度偏移
    offs_n = tl.arange(0, BLOCK_SIZE_N)[:, None]  # N维度偏移
    expert_mask = offs_n < num_experts  # 专家维度掩码
    b_ptrs = b_ptr + (offs_n * stride_bn + offs_k)  # 计算B的地址

    # 3. Create an accumulator of float32 of size [BLOCK_SIZE_M, BLOCK_SIZE_N]  # 3. 创建float32累加器，大小为[BLOCK_SIZE_M, BLOCK_SIZE_N]
    #    3.1. iterate in K dimension  # 3.1. 沿K维度迭代
    #    3.2. transpose tile B  # 3.2. 转置B的瓦片
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 初始化累加器为零
    for k in range(0, K // BLOCK_SIZE_K):  # hidden_dim % BLOCK_SIZE_K == 0  # 沿K维度迭代
        a = tl.load(  # 加载A的瓦片
            a_ptrs,
            mask=bs_mask,  # 应用batch掩码
            other=0.0,  # 越界填充0
        ).to(tl.float32)  # 转为float32
        b = tl.load(b_ptrs, mask=expert_mask, other=0.0).to(tl.float32).T  # 加载B的瓦片并转置
        acc += tl.dot(a, b)  # 使用Tensor Core矩阵乘法累加

        # Advance the ptrs to the next K block.  # 将指针前进到下一个K块
        a_ptrs += BLOCK_SIZE_K  # A指针前移
        b_ptrs += BLOCK_SIZE_K  # B指针前移

    # 4. logit softcap  # 4. logit软截断
    if moe_softcapping == 0:  # 如果不使用软截断
        logits_softcapped = acc  # 直接使用原始logits
    else:  # 使用软截断
        logits_scaled = acc / moe_softcapping  # 缩放logits
        exped = tl.exp(2 * logits_scaled)  # 计算指数
        logits_softcapped = (exped - 1) / (exped + 1) * moe_softcapping  # tanh截断

    # Add bias after softcapping  # 软截断后添加偏置
    if is_correction_bias:  # 如果有校正偏置
        bias = tl.load(  # 加载偏置
            correction_bias_ptr + tl.arange(0, BLOCK_SIZE_N)[None, :],  # 偏置地址
            mask=expert_mask.T,  # 应用转置后的专家掩码
            other=0.0,  # 越界填充0
        )
        logits_softcapped = logits_softcapped + bias  # 加上偏置

    if dp_attn_workaround_flag:  # 如果启用DP注意力变通
        logits_softcapped = tl.where(  # 处理NaN值
            logits_softcapped != logits_softcapped, -1e9, logits_softcapped  # NaN替换为-1e9
        )

    # 5. top1  # 5. 找top1
    arange_block_size_n = tl.arange(0, BLOCK_SIZE_N)[None, :]  # N维度索引
    cond_top1 = arange_block_size_n < num_experts  # top1条件掩码
    top1 = tl.argmax(tl.where(cond_top1, logits_softcapped, float("-inf")), axis=1)  # 找到top1索引
    top1_v = tl.max(  # top1的logit值
        tl.where(cond_top1, logits_softcapped, float("-inf")), axis=1, keep_dims=True  # 保持维度
    )
    top1_invsumexp = 1.0 / tl.sum(  # softmax归一化因子
        tl.where(cond_top1, tl.exp(logits_softcapped - top1_v), 0.0), axis=1  # 条件求和
    )

    # 6. store top1 to output  # 6. 存储top1到输出
    offs_top1 = pid * topk * BLOCK_SIZE_M + topk * tl.arange(0, BLOCK_SIZE_M)  # top1偏移
    top1_mask = offs_top1 < bs * topk  # top1掩码
    tl.store(topk_ids_ptr + offs_top1, top1, mask=top1_mask)  # 存储top1 ID
    tl.store(  # 存储top1权重
        topk_weights_ptr + offs_top1,
        top1_invsumexp,  # softmax概率
        mask=top1_mask,  # 应用掩码
    )

    # 7. handle topk == 2  # 7. 处理topk == 2
    if topk == 2:
        cond_top2 = (arange_block_size_n < num_experts) & (  # top2条件掩码
            arange_block_size_n != top1[:, None]  # 排除top1
        )
        top2 = tl.argmax(  # 找到top2索引
            tl.where(cond_top2, logits_softcapped, float("-inf")),  # 条件选择
            axis=1,
            keep_dims=True,  # 保持维度
        )
        top2_v = tl.sum(  # top2的logit值
            logits_softcapped * (arange_block_size_n == top2), axis=1, keep_dims=True  # 提取对应位置
        )
        top2_invsumexp = tl.exp(top2_v - top1_v) * top1_invsumexp[:, None]  # top2的softmax概率

        # store top2  # 存储top2
        offs_top2 = (  # top2偏移
            pid * topk * BLOCK_SIZE_M + topk * tl.arange(0, BLOCK_SIZE_M)[:, None] + 1  # top2的位置
        )
        top2_mask = offs_top2 < bs * topk  # top2掩码
        tl.store(topk_ids_ptr + offs_top2, top2, mask=top2_mask)  # 存储top2 ID
        tl.store(  # 存储top2权重
            topk_weights_ptr + offs_top2,
            top2_invsumexp,  # softmax概率
            mask=top2_mask,  # 应用掩码
        )


def fused_moe_router_tensorcore(  # Tensor Core版MoE路由器主机函数
    x: torch.Tensor,  # 输入张量（batch_size, hidden_dim）
    router_weight: torch.Tensor,  # 路由权重（num_experts, hidden_dim）
    topk: int,  # TopK值
    moe_softcapping: float,  # 软截断值
    BLOCK_SIZE_M: int,  # M维度块大小
    BLOCK_SIZE_N: int,  # N维度块大小
    BLOCK_SIZE_K: int,  # K维度块大小
    correction_bias: Optional[torch.Tensor] = None,  # 校正偏置
):
    assert len(x.shape) == 2 and x.shape[1] == router_weight.shape[1]  # 断言输入和权重第二维匹配
    bs, hidden_dim = x.shape  # 解析batch_size和hidden_dim
    num_experts = router_weight.shape[0]  # 获取专家数

    assert num_experts <= BLOCK_SIZE_N  # 断言专家数不超过N块大小
    assert hidden_dim % BLOCK_SIZE_K == 0  # 断言hidden_dim可被K块大小整除
    assert topk <= 2  # 断言topk不超过2

    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)  # 分配TopK权重
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)  # 分配TopK ID
    is_correction_bias = correction_bias is not None  # 判断是否有校正偏置

    grid = (triton.cdiv(bs, BLOCK_SIZE_M) * triton.cdiv(num_experts, BLOCK_SIZE_N),)  # 计算网格大小

    # TODO(ch-wan): temporary workaround for dp attention. We should support masked  # 待办：DP注意力的临时变通方案。应支持掩码
    # router to skip padded tokens.  # 路由器以跳过填充token
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled  # 导入DP注意力检测

    dp_attn_workaround_flag = is_dp_attention_enabled()  # 检查是否启用DP注意力

    fused_moe_router_tensorcore_kernel[grid](  # 启动Tensor Core路由器内核
        a_ptr=x,  # 输入
        b_ptr=router_weight,  # 路由权重
        topk_weights_ptr=topk_weights,  # TopK权重
        topk_ids_ptr=topk_ids,  # TopK ID
        bs=bs,  # batch_size
        num_experts=num_experts,  # 专家数
        topk=topk,  # TopK值
        moe_softcapping=moe_softcapping,  # 软截断值
        moe_renormalize=False,  # 不重归一化
        K=hidden_dim,  # K维度
        correction_bias_ptr=correction_bias,  # 校正偏置
        is_correction_bias=is_correction_bias,  # 是否有偏置
        BLOCK_SIZE_M=BLOCK_SIZE_M,  # M块大小
        BLOCK_SIZE_N=BLOCK_SIZE_N,  # N块大小
        BLOCK_SIZE_K=BLOCK_SIZE_K,  # K块大小
        stride_am=hidden_dim,  # A行步长
        stride_bn=hidden_dim,  # B行步长
        dp_attn_workaround_flag=dp_attn_workaround_flag,  # DP注意力标志
    )

    return topk_weights, topk_ids  # 返回TopK权重和ID


def fused_moe_router_shim(  # MoE路由器选择器，根据输入大小自动选择内核版本
    moe_softcapping,  # 软截断值
    hidden_states,  # 隐藏状态
    gating_output,  # 门控输出（路由权重）
    topk,  # TopK值
    renormalize,  # 是否重归一化
    correction_bias: Optional[torch.Tensor] = None,  # 校正偏置
    enable_deterministic_inference: bool = False,  # 是否启用确定性推理
):
    assert not renormalize  # 断言不使用重归一化
    assert (  # 断言隐藏状态和门控输出形状匹配
        len(hidden_states.shape) == 2
        and hidden_states.shape[1] == gating_output.shape[1]
    )
    bs, hidden_dim = hidden_states.shape  # 解析batch_size和hidden_dim
    num_experts = gating_output.shape[0]  # 获取专家数

    BLOCK_SIZE_M = 32  # M维度块大小

    BLOCK_SIZE_N = max(num_experts, 16)  # N维度块大小，至少16
    BLOCK_SIZE_K = (  # K维度块大小
        256 if num_experts < 256 else 64  # 专家数少用256，多用64
    )  # if experts are large, need to use smaller k block or shared memory OOM  # 专家数大时需用更小的K块，否则共享内存溢出

    if (  # 判断是否使用Tensor Core版本
        (bs >= 512 or num_experts > 8)  # 大批量或专家数多
        and hidden_dim % BLOCK_SIZE_K == 0  # hidden_dim可被K块大小整除
        # we keep using single kernel to avoid non-deterministic behavior  # 我们保持使用单内核以避免非确定性行为
        and not enable_deterministic_inference  # 未启用确定性推理
    ):
        # if large batch size or large expert, use kernel that uses tensorcore in matmul  # 大批量或专家数多时，使用Tensor Core矩阵乘法内核
        return fused_moe_router_tensorcore(  # 返回Tensor Core版本
            x=hidden_states,  # 隐藏状态
            router_weight=gating_output,  # 路由权重
            topk=topk,  # TopK值
            moe_softcapping=moe_softcapping,  # 软截断值
            BLOCK_SIZE_M=BLOCK_SIZE_M,  # M块大小
            BLOCK_SIZE_N=BLOCK_SIZE_N,  # N块大小
            BLOCK_SIZE_K=BLOCK_SIZE_K,  # K块大小
            correction_bias=correction_bias,  # 校正偏置
        )
    else:  # 小批量场景
        # if smaller, use kernel that does not use tensorcore in matmul  # 小批量时，使用不使用Tensor Core的内核
        return fused_moe_router_cudacore(  # 返回CUDA Core版本
            x=hidden_states,  # 隐藏状态
            router_weight=gating_output,  # 路由权重
            topk=topk,  # TopK值
            moe_softcapping=moe_softcapping,  # 软截断值
            correction_bias=correction_bias,  # 校正偏置
        )


class FusedMoeRouter:  # 融合MoE路由器封装类
    def __init__(self, router_linear, topk, moe_softcapping) -> None:  # 初始化路由器
        self.router_linear = router_linear  # 路由线性层
        self.topk = topk  # TopK值
        self.moe_softcapping = moe_softcapping  # 软截断值

    def __call__(self, *args, **kwargs):  # 可调用对象接口
        return self.forward(*args, **kwargs)  # 委托给forward方法

    def forward(  # 前向传播入口
        self, x: torch.Tensor, residual: torch.Tensor  # 输入张量和残差
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.is_cuda:  # 如果在CUDA上
            return self.forward_cuda(x, residual)  # 使用CUDA前向
        else:  # 否则
            return self.forward_vllm(x, residual)  # 使用vLLM前向

    def forward_cuda(  # CUDA平台前向传播
        self, x: torch.Tensor, autotune=False  # 输入张量和自动调优标志
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return fused_moe_router_shim(  # 调用路由器选择器
            moe_softcapping=self.moe_softcapping,  # 软截断值
            hidden_states=x,  # 隐藏状态
            gating_output=self.router_linear.weight,  # 路由权重
            topk=self.topk,  # TopK值
            renormalize=False,  # 不重归一化
        )

    def forward_torch(  # PyTorch参考实现
        self,
        x: torch.Tensor,  # 输入张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = x.float() @ self.router_linear.weight.T.float()  # 计算路由logits

        g = torch.tanh(g.float() / self.moe_softcapping) * self.moe_softcapping  # tanh软截断

        return fused_topk(x, g, self.topk, False)  # 调用通用TopK函数
