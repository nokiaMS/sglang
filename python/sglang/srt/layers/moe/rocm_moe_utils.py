# Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1rc2/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py  # 改编自vLLM项目的ROCm AITER融合MoE实现
# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # SPDX版权声明：vLLM项目的贡献者
# 文件说明：ROCm平台MoE工具函数，提供AITER融合MoE内核调用和Triton实现的量化解码内核。
# 包含FP8/FP4权重量化的MoE前向计算、MXFP4反量化（upscale）内核等功能。
# 主要用于AMD GPU（ROCm/HIP）平台的混合专家模型推理加速。
from enum import IntEnum  # 导入整数枚举类
from typing import Optional  # 导入可选类型

import torch  # 导入PyTorch
import triton  # 导入Triton
import triton.language as tl  # 导入Triton语言

from sglang.srt.utils import get_bool_env_var, is_hip  # 导入环境变量和HIP检测工具
from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册

_is_hip = is_hip()  # 检测是否为HIP平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITER（仅HIP平台）


class ActivationMethod(IntEnum):  # 激活方法枚举类
    # This allows interfacing with AITER ActivationType enum  # 这允许与AITER的ActivationType枚举接口
    # without importing the ActivationType enum from AITER globally.  # 而无需全局导入AITER的ActivationType枚举
    SILU = 0  # SiLU激活
    GELU = 1  # GELU激活


# NOTE: for non _use_aiter case, use lazy registration to avoid overhead  # 注意：对于非_use_aiter情况，使用延迟注册以避免开销
# (registration may not be trigger actually, since it will not be called)  # （注册可能实际不会触发，因为不会被调用）
@register_custom_op(out_shape="hidden_states", eager=_use_aiter)  # 注册自定义算子，输出形状与hidden_states相同
def rocm_aiter_asm_moe_tkw1(  # ROCm AITER汇编MoE内核（topk权重为1）
    hidden_states: torch.Tensor,  # 隐藏状态张量
    w1: torch.Tensor,  # 第一层权重
    w2: torch.Tensor,  # 第二层权重
    topk_weights: torch.Tensor,  # TopK权重
    topk_ids: torch.Tensor,  # TopK ID
    fc1_scale: Optional[torch.Tensor] = None,  # 第一层缩放因子
    fc2_scale: Optional[torch.Tensor] = None,  # 第二层缩放因子
    fc1_smooth_scale: Optional[torch.Tensor] = None,  # 第一层平滑缩放因子
    fc2_smooth_scale: Optional[torch.Tensor] = None,  # 第二层平滑缩放因子
    a16: bool = False,  # 是否使用16位精度
    per_tensor_quant_scale: Optional[torch.Tensor] = None,  # 每张量量化缩放因子
    expert_mask: Optional[torch.Tensor] = None,  # 专家掩码
    activation_method: int = ActivationMethod.SILU.value,  # 激活方法，默认SiLU
) -> torch.Tensor:

    from aiter import ActivationType  # 导入AITER激活类型
    from aiter.fused_moe_bf16_asm import asm_moe_tkw1  # 导入AITER汇编MoE内核

    activation = ActivationType(activation_method)  # 将整数转为激活类型枚举

    return asm_moe_tkw1(  # 调用AITER汇编MoE内核
        hidden_states,  # 隐藏状态
        w1,  # 第一层权重
        w2,  # 第二层权重
        topk_weights,  # TopK权重
        topk_ids,  # TopK ID
        fc1_scale=fc1_scale,  # 第一层缩放
        fc2_scale=fc2_scale,  # 第二层缩放
        fc1_smooth_scale=fc1_smooth_scale,  # 第一层平滑缩放
        fc2_smooth_scale=fc2_smooth_scale,  # 第二层平滑缩放
        a16=a16,  # 16位精度标志
        per_tensor_quant_scale=per_tensor_quant_scale,  # 每张量缩放
        expert_mask=expert_mask,  # 专家掩码
        activation=activation,  # 激活类型
    )


def rocm_fused_experts_tkw1(  # ROCm融合专家MoE前向（topk权重为1）
    hidden_states: torch.Tensor,  # 隐藏状态张量
    w1: torch.Tensor,  # 第一层权重
    w2: torch.Tensor,  # 第二层权重
    topk_weights: torch.Tensor,  # TopK权重
    topk_ids: torch.Tensor,  # TopK ID
    activation: str = "silu",  # 激活函数，默认silu
    apply_router_weight_on_input: bool = False,  # 是否在输入上应用路由权重
    use_fp8_w8a8: bool = False,  # 是否使用FP8 W8A8量化
    per_channel_quant: bool = False,  # 是否使用逐通道量化
    w1_scale: Optional[torch.Tensor] = None,  # 第一层权重缩放因子
    w2_scale: Optional[torch.Tensor] = None,  # 第二层权重缩放因子
    a1_scale: Optional[torch.Tensor] = None,  # 第一层激活缩放因子
    a2_scale: Optional[torch.Tensor] = None,  # 第二层激活缩放因子
    block_shape: Optional[list[int]] = None,  # 块形状
) -> torch.Tensor:

    activation_method = (  # 确定激活方法枚举值
        ActivationMethod.SILU if activation == "silu" else ActivationMethod.GELU  # silu对应SILU，否则GELU
    )
    # All AITER Fused MoE kernels are expecting the following datatypes  # 所有AITER融合MoE内核期望以下数据类型
    topk_weights = topk_weights.to(torch.float32)  # 转为float32
    topk_ids = topk_ids.to(torch.int32)  # 转为int32

    # w8a8 per-channel quantization  # W8A8逐通道量化
    if per_channel_quant and apply_router_weight_on_input and use_fp8_w8a8:  # 逐通道量化且在输入上应用路由权重且使用FP8
        # AITER tkw1 kernel for FP8 models with `apply_router_weight_on_input`  # AITER tkw1内核用于启用了`apply_router_weight_on_input`的FP8模型
        # This applies topk_weights on the GEMM output of the first FC layer  # 此操作将topk权重应用于第一层FC的GEMM输出
        #  rather than the second FC.  # 而非第二层FC
        assert (  # 断言topk权重为2维
            topk_weights.dim() == 2
        ), "`topk_weights` should be in shape (num_tokens, topk)"  # `topk_weights`应为(num_tokens, topk)形状
        assert topk_weights.shape[-1] == 1, (  # 断言topk=1
            "Only support topk=1 when" " `apply_router_weight_on_input` is True"  # 当`apply_router_weight_on_input`为True时仅支持topk=1
        )

        return rocm_aiter_asm_moe_tkw1(  # 调用AITER汇编MoE内核
            hidden_states,  # 隐藏状态
            w1,  # 第一层权重
            w2,  # 第二层权重
            topk_weights,  # TopK权重
            topk_ids,  # TopK ID
            fc1_scale=w1_scale,  # 第一层缩放因子
            fc2_scale=w2_scale,  # 第二层缩放因子
            fc1_smooth_scale=None,  # 无平滑缩放
            fc2_smooth_scale=None,  # 无平滑缩放
            a16=False,  # 不使用16位
            per_tensor_quant_scale=None,  # 无每张量缩放
            expert_mask=None,  # 无专家掩码
            activation_method=activation_method,  # 激活方法
        )
    else:  # 不满足上述条件
        assert False, "This should not be called."  # 断言不应被调用


@triton.jit  # Triton JIT编译内核
def upscale_kernel(  # 反量化缩放内核（FP8/FP16->FP16/BF16）
    A_ptr,  # *fp16 / *fp32  # 输入指针（fp16/fp32）
    scale_ptr,  # *fp16 / *fp32  # 缩放因子指针（fp16/fp32）
    Out_ptr,  # *fp16 / *fp32  # 输出指针（fp16/fp32）
    M,  # 行数  # M维度
    N,  # 列数  # N维度
    recv_token_num,  # 接收的token数量  # 实际接收的token数量
    stride_am,  # A的行步长  # A的第0维步长
    stride_an,  # A的列步长  # A的第1维步长
    stride_sm,  # scale的行步长  # scale的第0维步长
    stride_sn,  # scale的列步长  # scale的第1维步长
    stride_om,  # Out的行步长  # Out的第0维步长
    stride_on,  # Out的列步长  # Out的第1维步长
    BLOCK_N: tl.constexpr,  # N维度的块大小（编译时常量）
):
    pid_m = tl.program_id(0)  # row id  # 行索引
    pid_n = tl.program_id(1)  # block id along N  # N维度的块索引

    recv_token_num_val = tl.load(recv_token_num)  # 加载实际接收的token数

    if pid_m >= recv_token_num_val:  # 如果行索引超出范围
        return  # 直接返回

    # column offsets  # 列偏移
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # 计算列偏移
    mask = offs_n < N  # 生成列掩码

    # A[m, n]  # 加载输入A的第m行
    a_ptrs = A_ptr + pid_m * stride_am + offs_n * stride_an  # 计算A的地址
    a = tl.load(a_ptrs, mask=mask, other=0.0)  # 加载A的值

    # scale index: n // 128  # 缩放因子索引：n // 128
    scale_idx = offs_n // 128  # 每128列共享一个缩放因子
    s_ptrs = scale_ptr + pid_m * stride_sm + scale_idx * stride_sn  # 计算scale的地址
    s = tl.load(s_ptrs, mask=mask, other=1.0)  # 加载缩放因子

    out = a * s  # 逐元素相乘

    out_ptrs = Out_ptr + pid_m * stride_om + offs_n * stride_on  # 计算输出地址
    tl.store(out_ptrs, out, mask=mask)  # 存储结果


def upscale(hidden_state, hidden_state_scale, recv_token_num, output_dtype):  # 反量化缩放函数
    M, N = hidden_state.shape  # 获取输入形状

    Out = torch.empty_like(hidden_state, dtype=output_dtype)  # 创建输出张量

    BLOCK_N = 256  # N维度的块大小

    grid = (M, triton.cdiv(N, BLOCK_N))  # 计算网格大小

    upscale_kernel[grid](  # 调用upscale内核
        hidden_state,  # 输入隐藏状态
        hidden_state_scale,  # 缩放因子
        Out,  # 输出
        M,  # 行数
        N,  # 列数
        recv_token_num,  # 接收token数
        hidden_state.stride(0),  # 行步长
        hidden_state.stride(1),  # 列步长
        hidden_state_scale.stride(0),  # scale行步长
        hidden_state_scale.stride(1),  # scale列步长
        Out.stride(0),  # 输出行步长
        Out.stride(1),  # 输出列步长
        BLOCK_N=BLOCK_N,  # 块大小
    )

    return Out  # 返回反量化结果


@triton.jit  # Triton JIT编译内核
def upscale_fp4x2_block32_kernel(  # MXFP4反量化内核（将FP4 E2M1解码为FP16/BF16/FP32）
    A_u8_ptr,  # *uint8  (view from float4_e2m1fn_x2)  # 输入指针（uint8视图，来自float4_e2m1fn_x2）
    S_u8_ptr,  # *uint8  (view from float8_e8m0fnu), shape (M, N_fp4/32)  # 缩放因子指针（uint8视图，来自float8_e8m0fnu）
    Out_ptr,  # *fp16/fp32/bf16, shape (M, N_fp4)  # 输出指针
    N_FP4: tl.constexpr,  # FP4元素总数（编译时常量）
    recv_token_num,  # 接收的token数量
    stride_am,  # A的行步长
    stride_an,  # A strides (in uint8 elements) for (M, packed_N)  # A的列步长（以uint8元素计）
    stride_sm,  # S的行步长
    stride_sn,  # S strides (in uint8 elements) for (M, N_FP4/32)  # S的列步长（以uint8元素计）
    stride_om,  # Out的行步长
    stride_on,  # Out strides (in output elements) for (M, N_FP4)  # Out的列步长（以输出元素计）
    BLOCK_N: tl.constexpr,  # N维度的块大小（编译时常量）
    OUT_DTYPE: tl.constexpr,  # tl.float16 / tl.float32 / tl.bfloat16  # 输出数据类型（编译时常量）
):
    pid_m = tl.program_id(0)  # 行索引
    pid_n = tl.program_id(1)  # N维度块索引

    recv_token_num_val = tl.load(recv_token_num)  # 加载实际接收token数
    if pid_m >= recv_token_num_val:  # 如果行索引超出范围
        return  # 直接返回

    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # 计算元素偏移
    mask = offs < N_FP4  # 生成掩码

    # --------------------------  # 分隔线
    # Load packed fp4x2 byte  # 加载打包的FP4x2字节
    # --------------------------  # 分隔线
    byte_idx = offs >> 1  # offs // 2  # 字节索引（两个FP4值打包在一个字节中）
    is_hi = (offs & 1) != 0  # select high nibble?  # 是否选择高4位

    a_ptrs = A_u8_ptr + pid_m * stride_am + byte_idx * stride_an  # 计算输入地址
    a_byte = tl.load(a_ptrs, mask=mask, other=0).to(tl.int32)  # 加载字节数据

    lo = a_byte & 0xF  # 提取低4位
    hi = (a_byte >> 4) & 0xF  # 提取高4位
    code = tl.where(is_hi, hi, lo).to(tl.int32)  # 0..15  # 根据高低位选择4位编码

    # --------------------------  # 分隔线
    # Decode float4_e2m1fn  # 解码float4_e2m1fn格式
    # layout: [sign|exp(2)|mant(1)]  # 布局：[符号|指数(2位)|尾数(1位)]
    # bias=1, finite-only  # 偏置=1，仅有限值
    # --------------------------  # 分隔线
    sign = (code >> 3) & 0x1  # 提取符号位
    exp = (code >> 1) & 0x3  # 提取2位指数
    mant = code & 0x1  # 提取1位尾数

    mant_f = mant.to(tl.float32) * 0.5  # 尾数转为浮点（0.0或0.5）
    is_sub = exp == 0  # 判断是否为次正规数

    # normal: 2^(exp-bias) * (1 + mant/2), bias=1  # 正规数：2^(exp-bias) * (1 + mant/2)，bias=1
    e_norm = (exp - 1).to(tl.float32)  # 正规数指数偏移
    val_norm = tl.exp2(e_norm) * (1.0 + mant_f)  # 计算正规数值

    # subnorm/zero: mant/2 * 2^(1-bias) = mant/2  # 次正规/零：mant/2 * 2^(1-bias) = mant/2
    val_sub = mant_f  # 次正规数值

    val = tl.where(is_sub, val_sub, val_norm)  # 根据是否次正规选择值
    val = tl.where(sign != 0, -val, val)  # apply sign  # 应用符号位

    # --------------------------  # 分隔线
    # Per-token block32 scale: scale_idx = offs // 32  # 每token的block32缩放因子：索引 = offs // 32
    # scale dtype: float8_e8m0fnu stored in uint8  # 缩放因子类型：float8_e8m0fnu，以uint8存储
    # decode: e==0 -> 0  # 解码：e==0 -> 0
    #         e in [1..254] -> 2^(e-127)  #         e在[1..254] -> 2^(e-127)
    #         e==255 -> clamp to 254  #         e==255 -> 截断为254
    # --------------------------  # 分隔线
    scale_idx = offs >> 5  # offs // 32  # 每32个元素共享一个缩放因子

    s_ptrs = S_u8_ptr + pid_m * stride_sm + scale_idx * stride_sn  # 计算缩放因子地址
    e = tl.load(s_ptrs, mask=mask, other=0).to(tl.int32)  # 加载指数

    e = tl.minimum(e, 254)  # clamp 255->254  # 将255截断为254
    is_zero = e == 0  # 判断是否为零缩放
    exp_s = (e - 127).to(tl.float32)  # 计算缩放因子指数
    s = tl.exp2(exp_s)  # 计算2的幂
    s = tl.where(is_zero, 0.0, s)  # 零指数对应零缩放

    out = (val * s).to(OUT_DTYPE)  # 乘以缩放因子并转为输出类型

    out_ptrs = Out_ptr + pid_m * stride_om + offs * stride_on  # 计算输出地址
    tl.store(out_ptrs, out, mask=mask)  # 存储结果


def upscale_mxfp4(hidden_state, hidden_state_scale, recv_token_num, output_dtype):  # MXFP4反量化函数
    """  # 文档字符串
    hidden_state: (M, packed_N) torch.float4_e2m1fn_x2  # 隐藏状态：(M, packed_N) float4_e2m1fn_x2格式
    hidden_state_scale: (M, packed_N*2/32) = (M, N_fp4/32) torch.float8_e8m0fnu  # 缩放因子：(M, N_fp4/32) float8_e8m0fnu格式
    output: (M, N_fp4) output_dtype  # 输出：(M, N_fp4) 指定输出类型
    """
    assert hidden_state.dtype == torch.float4_e2m1fn_x2, hidden_state.dtype  # 断言输入为FP4类型
    assert hidden_state_scale.dtype == torch.float8_e8m0fnu, hidden_state_scale.dtype  # 断言缩放因子为E8M0类型
    assert hidden_state.is_contiguous() or True  # stride-based load OK  # 步长加载可行（断言恒通过）

    M, packed_N = hidden_state.shape  # 获取打包后的形状
    N_fp4 = packed_N * 2  # 每个打包元素包含2个FP4值

    # scale second dim must be N_fp4/32  # 缩放因子第二维必须为N_fp4/32
    assert hidden_state_scale.shape[0] == M  # 断言行数匹配
    assert hidden_state_scale.shape[1] == (N_fp4 // 32), (  # 断言列数匹配
        hidden_state_scale.shape,  # 实际形状
        N_fp4,  # 期望的FP4维度
    )

    # Triton doesn't (reliably) accept torch.float4/float8 pointers directly.  # Triton不能（可靠地）直接接受torch.float4/float8指针
    # Use raw uint8 views.  # 使用原始uint8视图
    A_u8 = hidden_state.view(torch.uint8)  # 将FP4数据转为uint8视图
    S_u8 = hidden_state_scale.view(torch.uint8)  # 将缩放因子转为uint8视图

    Out = torch.empty((M, N_fp4), dtype=output_dtype, device=hidden_state.device)  # 创建输出张量

    BLOCK_N = 256  # N维度的块大小
    grid = (M, triton.cdiv(N_fp4, BLOCK_N))  # 计算网格大小

    OUT_TL = (  # 确定Triton输出类型
        tl.float16  # float16
        if output_dtype == torch.float16  # 如果输出类型为float16
        else tl.bfloat16 if output_dtype == torch.bfloat16 else tl.float32  # 否则bfloat16或float32
    )

    upscale_fp4x2_block32_kernel[grid](  # 调用MXFP4反量化内核
        A_u8,  # uint8视图的输入
        S_u8,  # uint8视图的缩放因子
        Out,  # 输出张量
        N_FP4=N_fp4,  # FP4维度
        recv_token_num=recv_token_num,  # 接收token数
        stride_am=A_u8.stride(0),  # A的行步长
        stride_an=A_u8.stride(1),  # A的列步长
        stride_sm=S_u8.stride(0),  # S的行步长
        stride_sn=S_u8.stride(1),  # S的列步长
        stride_om=Out.stride(0),  # Out的行步长
        stride_on=Out.stride(1),  # Out的列步长
        BLOCK_N=BLOCK_N,  # 块大小
        OUT_DTYPE=OUT_TL,  # 输出类型
        num_warps=4,  # warp数
    )
    return Out  # 返回反量化结果
