# FlashInfer CuteDSL MoE 实现文件
# 使用 FlashInfer 的 CuteDSL 内核实现 FP4 量化的 MoE 计算
# 支持掩码分组 GEMM 和 NVFP4 量化
from typing import Optional  # 导入可选类型 # 导入可选类型提示

import torch  # 导入 PyTorch # 导入 PyTorch 深度学习框架
from flashinfer import (  # 导入 FlashInfer 函数 # 导入 FlashInfer 库函数
    scaled_fp4_grouped_quantize,  # FP4 分组量化 # 缩放 FP4 分组量化函数
    silu_and_mul_scaled_nvfp4_experts_quantize,  # SiLU 乘法缩放 NVFP4 量化 # SiLU 激活+乘法+NVFP4 量化融合函数
)
from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked  # 导入掩码分组 GEMM # 导入 NT 格式掩码分组 GEMM 内核


def get_cute_dtype(input: torch.Tensor) -> str:  # 获取 CuteDSL 数据类型字符串 # 将 PyTorch 数据类型转换为 CuteDSL 数据类型字符串
    """将 PyTorch 数据类型转换为 CuteDSL 支持的数据类型字符串"""
    if input.dtype == torch.bfloat16:  # BF16 类型 # 判断是否为 BF16 数据类型
        return "bfloat16"  # 返回 bfloat16 # 返回 BF16 字符串
    elif input.dtype == torch.float16:  # FP16 类型 # 判断是否为 FP16 数据类型
        return "float16"  # 返回 float16 # 返回 FP16 字符串
    elif input.dtype == torch.float32:  # FP32 类型 # 判断是否为 FP32 数据类型
        return "float32"  # 返回 float32 # 返回 FP32 字符串
    else:  # 不支持的类型 # 否则
        raise ValueError(f"Unsupported cute dtype {input.dtype}")  # 抛出异常 # 抛出不支持的数据类型异常


def flashinfer_cutedsl_moe_masked(  # FlashInfer CuteDSL MoE 掩码前向计算 # 使用 FlashInfer CuteDSL 内核执行掩码 MoE 计算
    """使用 FlashInfer CuteDSL 内核执行掩码 Mixture-of-Experts 计算""",
    hidden_states: tuple[torch.Tensor, Optional[torch.Tensor]],  # 隐藏状态（可能包含量化数据） # 输入隐藏状态，可为原始或量化数据
    input_global_scale: torch.Tensor,  # 输入全局缩放因子 # 输入的全局缩放因子
    w1: torch.Tensor,  # 第一个 GEMM 权重 # gate-up 投影的 FP4 权重
    w1_blockscale: torch.Tensor,  # 第一个 GEMM 块缩放 # gate-up 权重的块缩放因子
    w1_alpha,  # 第一个 GEMM alpha # gate-up 权重的全局缩放系数
    w2: torch.Tensor,  # 第二个 GEMM 权重 # down 投影的 FP4 权重
    a2_global_scale: torch.Tensor,  # 中间层全局缩放因子 # 中间激活的全局缩放因子
    w2_blockscale: torch.Tensor,  # 第二个 GEMM 块缩放 # down 权重的块缩放因子
    w2_alpha,  # 第二个 GEMM alpha # down 权重的全局缩放系数
    masked_m: torch.Tensor,  # 掩码维度索引 # 每个专家的有效 token 数
    down_sm_count: Optional[int] = None,  # down 层 SM 数量 # 下投影层使用的 SM 数量
    down_signals: Optional[torch.Tensor] = None,  # down 层信号 # 下投影层的同步信号
    down_start_event: Optional[torch.cuda.Event] = None,  # down 层起始事件 # 下投影层的起始 CUDA 事件
):
    """
    Perform masked Mixture-of-Experts computation with FlashInfer's CuteDSL
    kernels.
    使用 FlashInfer 的 CuteDSL 内核执行掩码 Mixture-of-Experts 计算。

    Args:
        hidden_states: Either of the following case
            * tuple[torch.Tensor, None]: [num_experts, m, k], bf16, None means no quant
            * tuple[torch.Tensor, torch.Tensor]: [num_experts, m, k // 2], uint8, [num_experts, m, k // 16], float8_e4m3fn
        input_global_scale (torch.Tensor): (l,)
        w1 (torch.Tensor): fp4 weights, [l, 2 * n, k // 2], uint8
        w1_blockscale (torch.Tensor): blockscale factors, e4m3,
        w1_alpha (torch.Tensor): (l,)
        w2 (torch.Tensor): fp4 weights, [l, k, n // 2], uint8
        a2_global_scale (torch.Tensor): (l,)
        w2_blockscale (torch.Tensor): blockscale factors, e4m3,
        w2_alpha (torch.Tensor): (l,)
        masked_m (torch.Tensor): Masked dimension indices
    参数：
        hidden_states: 以下情况之一
            * tuple[torch.Tensor, None]: [专家数, m, k], bf16, None 表示未量化
            * tuple[torch.Tensor, torch.Tensor]: [专家数, m, k // 2], uint8, [专家数, m, k // 16], float8_e4m3fn
        input_global_scale (torch.Tensor): (l,)
        w1 (torch.Tensor): FP4 权重, [l, 2 * n, k // 2], uint8
        w1_blockscale (torch.Tensor): 块缩放因子, e4m3,
        w1_alpha (torch.Tensor): (l,)
        w2 (torch.Tensor): FP4 权重, [l, k, n // 2], uint8
        a2_global_scale (torch.Tensor): (l,)
        w2_blockscale (torch.Tensor): 块缩放因子, e4m3,
        w2_alpha (torch.Tensor): (l,)
        masked_m (torch.Tensor): 掩码维度索引

    Notes:
        - Assumes max(masked_m) == m.
    注意：
        - 假设 max(masked_m) == m。
    """

    # === Assertions on dtypes ===
    # === 数据类型断言 ===
    assert w1.dtype == torch.uint8, f"w1 must be uint8 (fp4 packed), got {w1.dtype}"  # 检查 w1 数据类型 # 断言 w1 为 uint8（FP4 打包）
    assert (  # 检查 w1 块缩放类型 # 断言 w1 块缩放因子为 FP8
        w1_blockscale.dtype == torch.float8_e4m3fn
    ), f"w1_blockscale must be float8_e4m3fn, got {w1_blockscale.dtype}"  # 错误信息 # 类型不匹配的错误信息
    assert (  # 检查 w1 alpha 类型 # 断言 w1 alpha 为 float32
        w1_alpha.dtype == torch.float32
    ), f"w1_alpha must be float32, got {w1_alpha.dtype}"  # 错误信息 # 类型不匹配的错误信息
    assert w2.dtype == torch.uint8, f"w2 must be uint8 (fp4 packed), got {w2.dtype}"  # 检查 w2 数据类型 # 断言 w2 为 uint8（FP4 打包）
    assert (  # 检查 a2 全局缩放类型 # 断言 a2 全局缩放为 float32
        a2_global_scale.dtype == torch.float32
    ), f"a2_global_scale must be float32, got {a2_global_scale.dtype}"  # 错误信息 # 类型不匹配的错误信息
    assert (  # 检查 w2 块缩放类型 # 断言 w2 块缩放因子为 FP8
        w2_blockscale.dtype == torch.float8_e4m3fn
    ), f"w2_blockscale must be float8_e4m3fn, got {w2_blockscale.dtype}"  # 错误信息 # 类型不匹配的错误信息
    assert (  # 检查 w2 alpha 类型 # 断言 w2 alpha 为 float32
        w2_alpha.dtype == torch.float32
    ), f"w2_alpha must be float32, got {w2_alpha.dtype}"  # 错误信息 # 类型不匹配的错误信息
    assert (  # 检查 hidden_states 元组长度 # 断言隐藏状态为长度 2 的元组
        len(hidden_states) == 2
    ), f"hidden_states must be a tuple of length 2, got {len(hidden_states)}"  # 错误信息 # 长度不匹配的错误信息

    # === Assertions on shapes ===
    # === 形状断言 ===
    n = w2.shape[-1] * 2  # intermediate dimension # 中间维度 # 计算中间维度大小

    if hidden_states[1] is not None:  # 已量化的隐藏状态 # 判断隐藏状态是否已量化

        a_q = hidden_states[0].view(torch.uint8)  # 量化数据转为 uint8 # 将量化数据视图转为 uint8
        a_q_sf = hidden_states[1].view(torch.float8_e4m3fn)  # 缩放因子转为 FP8 # 将缩放因子视图转为 FP8
        m, k_by_2, num_experts = a_q.shape  # 解包形状 # 解包量化数据的形状
        k = k_by_2 * 2  # 计算原始 K 维度 # 恢复原始 K 维度大小
    else:  # 未量化的隐藏状态 # 否则
        num_experts, m, k = hidden_states[0].shape  # 解包形状 # 解包隐藏状态的形状

        assert (  # 检查输入全局缩放类型 # 断言输入全局缩放为 float32
            input_global_scale.dtype == torch.float32
        ), f"input_global_scale must be float32, got {input_global_scale.dtype}"  # 错误信息 # 类型不匹配的错误信息
        assert input_global_scale.shape == (  # 检查输入全局缩放形状 # 断言形状为 (l,)
            num_experts,
        ), f"input_global_scale must be (l,), got {input_global_scale.shape}"  # 错误信息 # 形状不匹配的错误信息

        a_q, a_q_sf = scaled_fp4_grouped_quantize(  # 执行 FP4 分组量化 # 对隐藏状态进行 FP4 量化
            hidden_states[0],  # 隐藏状态 # 原始隐藏状态
            masked_m,  # 掩码维度 # 每个专家的有效 token 数
            input_global_scale,  # 全局缩放因子 # 输入的全局缩放因子
        )

    assert w1.shape[-2] == 2 * n, f"w1 last-2 dim must be 2*n, got {w1.shape}"  # 检查 w1 倒数第二维 # 断言 w1 倒数第二维为 2*n
    assert (  # 检查 w1 最后一维 # 断言 w1 最后一维*2 等于 k
        w1.shape[-1] * 2 == k
    ), f"w1 last dim * 2 must equal k, got {w1.shape[-1]} vs k={k}"  # 错误信息 # 维度不匹配的错误信息
    assert w2.shape[-2:] == (  # 检查 w2 最后两维 # 断言 w2 最后两维为 (k, n//2)
        k,
        n // 2,
    ), f"w2 shape mismatch, got {w2.shape[-2:]}, expected {(k, n//2)}"  # 错误信息 # 形状不匹配的错误信息
    assert w1_alpha.shape == (  # 检查 w1 alpha 形状 # 断言 w1 alpha 形状为 (l,)
        num_experts,
    ), f"w1_alpha must be (l,), got {w1_alpha.shape}"  # 错误信息 # 形状不匹配的错误信息
    assert a2_global_scale.shape == (  # 检查 a2 全局缩放形状 # 断言 a2 全局缩放形状为 (l,)
        num_experts,
    ), f"a2_global_scale must be (l,), got {a2_global_scale.shape}"  # 错误信息 # 形状不匹配的错误信息
    assert w2_alpha.shape == (  # 检查 w2 alpha 形状 # 断言 w2 alpha 形状为 (l,)
        num_experts,
    ), f"w2_alpha must be (l,), got {w2_alpha.shape}"  # 错误信息 # 形状不匹配的错误信息

    # TODO(kaixih@nvidia): dtype should be based on inputs.
    # TODO(kaixih@nvidia): 数据类型应基于输入。
    gateup_output = torch.empty(  # 分配 gateup 输出缓冲区 # 分配 gate-up 层输出缓冲区
        (num_experts, m, n * 2), dtype=torch.bfloat16, device=a_q.device  # 形状和数据类型 # BF16 类型和设备
    )
    gateup_output = gateup_output.permute(1, 2, 0)  # requirement of kernel # 内核要求 # 转置为内核要求的 [m, 2n, l] 布局
    sf_vec_size = 16  # 缩放因子向量大小 # 设置缩放因子向量大小
    assert a_q_sf.dtype == torch.float8_e4m3fn  # 检查缩放因子类型 # 断言缩放因子为 FP8 类型
    assert a_q.dtype == torch.uint8  # 检查量化数据类型 # 断言量化数据为 uint8 类型
    ab_dtype = "float4_e2m1fn"  # AB 数据类型 # 设置 AB 数据类型为 FP4
    sf_dtype = "float8_e4m3fn"  # 缩放因子数据类型 # 设置缩放因子数据类型为 FP8
    c_dtype = "bfloat16"  # 输出数据类型 # 设置输出数据类型为 BF16

    # Gemm1
    # 第一个 GEMM（gate-up 投影）
    grouped_gemm_nt_masked(  # 执行掩码分组 GEMM # 调用 NT 格式掩码分组 GEMM 内核
        (a_q, a_q_sf),  # 输入（量化数据和缩放因子） # 输入量化激活和缩放因子
        (w1.permute(1, 2, 0), w1_blockscale),  # 权重（转置和块缩放） # 权重和块缩放因子
        gateup_output,  # 输出 # gate-up 层输出
        masked_m,  # 掩码维度 # 每个专家的有效 token 数
        ab_dtype=ab_dtype,  # AB 数据类型 # AB 数据类型
        sf_dtype=sf_dtype,  # 缩放因子类型 # 缩放因子数据类型
        c_dtype=c_dtype,  # 输出数据类型 # 输出数据类型
        sf_vec_size=sf_vec_size,  # 缩放因子向量大小 # 缩放因子向量大小
        alpha=w1_alpha.view(1, 1, num_experts),  # alpha 缩放系数 # 全局缩放系数
        alpha_dtype=get_cute_dtype(w1_alpha),  # alpha 数据类型 # alpha 的 CuteDSL 数据类型
    )  # in logical [m, n, l] # 逻辑形状为 [m, n, l]

    # SILU and quantization
    # SiLU 激活和量化
    diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(  # SiLU+乘法+NVFP4 量化 # 执行 SiLU 激活、乘法和 NVFP4 量化
        gateup_output.permute(2, 0, 1),  # 转置回 [l, m, 2n] # 转置回专家优先布局
        masked_m,  # 掩码维度 # 每个专家的有效 token 数
        a2_global_scale,  # 全局缩放因子 # 中间层的全局缩放因子
    )

    if down_start_event is not None:  # 有 down 层起始事件 # 判断是否提供了起始事件
        down_start_event.record()  # 记录事件 # 记录 CUDA 事件

    # Gemm2
    # 第二个 GEMM（down 投影）
    out = torch.empty((num_experts, m, k), dtype=torch.bfloat16, device=a_q.device)  # 分配输出缓冲区 # 分配最终输出缓冲区
    out = out.permute(1, 2, 0)  # requirement of kernel # 内核要求 # 转置为内核要求的 [m, k, l] 布局
    grouped_gemm_nt_masked(  # 执行掩码分组 GEMM # 调用 NT 格式掩码分组 GEMM 内核
        (diq, diq_sf),  # 输入（量化数据和缩放因子） # 中间层量化数据和缩放因子
        (w2.permute(1, 2, 0), w2_blockscale),  # 权重（转置和块缩放） # 权重和块缩放因子
        out,  # 输出 # 最终输出
        masked_m,  # 掩码维度 # 每个专家的有效 token 数
        ab_dtype=ab_dtype,  # AB 数据类型 # AB 数据类型
        sf_dtype=sf_dtype,  # 缩放因子类型 # 缩放因子数据类型
        c_dtype=c_dtype,  # 输出数据类型 # 输出数据类型
        sf_vec_size=sf_vec_size,  # 缩放因子向量大小 # 缩放因子向量大小
        alpha=w2_alpha.view(1, 1, num_experts),  # alpha 缩放系数 # 全局缩放系数
        alpha_dtype=get_cute_dtype(w2_alpha),  # alpha 数据类型 # alpha 的 CuteDSL 数据类型
        **(  # 可选参数 # 可选的 SM 和信号参数
            dict(
                sm_count=down_sm_count,  # SM 数量 # 使用的 SM 数量
                dst_signals=down_signals,  # 同步信号 # 目标同步信号
            )
            if down_sm_count is not None or down_signals is not None  # 有可选参数 # 判断是否提供可选参数
            else {}  # 无可选参数 # 空字典
        ),
    )  # in logical [m, k, l] # 逻辑形状为 [m, k, l]
    return out.permute(2, 0, 1)  # 转置回 [l, m, k] 并返回 # 转置回专家优先布局并返回
