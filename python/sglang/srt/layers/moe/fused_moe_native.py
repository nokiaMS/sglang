# 文件说明：FusedMoE的PyTorch原生实现模块，用于torch.compile编译模式
# 本模块提供了MoE（混合专家模型）的纯PyTorch原生前向计算实现，
# 包含两种前向函数：fused_moe_forward_native（基于分派输出的融合版本）
# 和moe_forward_native（基于逐专家循环的标准版本）

"""
Torch-native implementation for FusedMoE. This is used for torch.compile.
It is based on https://github.com/pytorch-labs/gpt-fast/blob/32971d3129541c5bfb4f715abc33d1c5f408d204/mixtral-moe/model.py#L204
"""
# Torch原生FusedMoE实现，用于torch.compile。基于gpt-fast项目的mixtral-moe实现

import torch  # 导入PyTorch库
from torch.nn import functional as F  # 导入PyTorch神经网络函数模块

from sglang.srt.layers.activation import GeluAndMul, SiluAndMul  # 导入GELU和SiLU激活函数
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig  # 导入MoE运行器配置类
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (  # 导入swiglu激活函数
    swiglu_gpt_oss_sigmoid_alpha,  # GPT-OSS风格的带alpha参数的swiglu激活函数
)
from sglang.srt.layers.moe.token_dispatcher import (  # 导入MoE令牌分派器类型
    StandardCombineInput,  # 标准合并输入类型
    StandardDispatchOutput,  # 标准分派输出类型
)
from sglang.srt.layers.moe.topk import StandardTopKOutput  # 导入标准TopK输出类型


def fused_moe_forward_native(  # 融合MoE原生前向计算函数（基于分派输出的融合版本）
    layer: torch.nn.Module,  # MoE层模块
    dispatch_output: StandardDispatchOutput,  # 标准分派输出（包含输入、缩放和topk结果）
) -> StandardCombineInput:  # 返回标准合并输入
    # 融合MoE前向计算：使用einsum实现批量专家计算，避免逐专家循环

    x, x_scale, topk_output = dispatch_output  # 解包分派输出：输入张量、缩放因子、topk结果
    moe_runner_config = layer.moe_runner_config  # 获取MoE运行器配置

    if moe_runner_config.apply_router_weight_on_input:  # 如果需要在输入上应用路由权重
        raise NotImplementedError()  # 本实现不支持该功能，抛出异常

    topk_weights, topk_ids, _ = topk_output  # 解包topk输出：权重、专家ID、占位符

    w13_weights = layer.w13_weight[topk_ids]  # 根据topk专家ID索引获取w13权重 [topk, E, 2N, K]
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)  # 将w13权重拆分为w1(门控)和w3(上投影)两部分
    w2_weights = layer.w2_weight[topk_ids]  # 根据topk专家ID索引获取w2(下投影)权重
    x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)  # 计算门控投影：x @ w1
    if moe_runner_config.activation == "silu":  # 如果激活函数为SiLU
        x1 = F.silu(x1)  # 对门控结果应用SiLU激活
    elif moe_runner_config.activation == "gelu":  # 如果激活函数为GELU
        x1 = F.gelu(x1)  # 对门控结果应用GELU激活
    else:  # 其他激活函数
        raise ValueError(f"Unsupported activation: {moe_runner_config.activation=}")  # 抛出不支持的激活函数异常
    x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)  # 计算上投影：x @ w3
    expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)  # 计算下投影：激活结果 * w2
    expert_outs = torch.einsum(  # 对topk专家输出进行加权求和
        "tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype)  # 乘以路由权重后求和
    )
    return StandardCombineInput(hidden_states=expert_outs)  # 返回标准合并输入


def moe_forward_native(  # MoE原生前向计算函数（基于逐专家循环的标准版本）
    layer: torch.nn.Module,  # MoE层模块
    x: torch.Tensor,  # 输入张量
    topk_output: StandardTopKOutput,  # 标准TopK输出
    moe_runner_config: MoeRunnerConfig,  # MoE运行器配置
) -> torch.Tensor:  # 返回MoE计算结果张量
    # 标准MoE前向计算：逐专家循环处理，支持偏置和自定义激活函数

    if moe_runner_config.apply_router_weight_on_input:  # 如果需要在输入上应用路由权重
        raise NotImplementedError()  # 本实现不支持该功能，抛出异常

    topk_weights, topk_ids, _ = topk_output  # 解包topk输出：权重、专家ID、占位符

    # Ref code from https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/e0828e3cc0a03408724b80c3cc92c8e072db8d01/modeling_deepseek.py#L589
    # 参考代码来自DeepSeek-V2模型实现
    len_experts = layer.num_experts  # 获取专家数量

    cnts = topk_ids.new_zeros((topk_ids.shape[0], len_experts))  # 创建令牌-专家计数矩阵，初始化为零
    cnts.scatter_(1, topk_ids.to(torch.int64), 1)  # 在对应专家位置填充1，统计每个令牌选择的专家
    tokens_per_expert = cnts.sum(dim=0)  # 按列求和，得到每个专家处理的令牌数量
    idxs = topk_ids.view(-1).argsort()  # 对topk_id展平后排序，得到按专家分组的令牌索引

    sorted_tokens = x[idxs // topk_ids.shape[1]]  # 根据排序索引获取按专家分组的令牌
    tokens_per_expert = tokens_per_expert.cpu().numpy()  # 将每个专家的令牌数转为CPU numpy数组

    if moe_runner_config.activation == "silu":  # 如果激活函数为SiLU
        act = SiluAndMul()  # 创建SiLU激活函数实例
    elif moe_runner_config.activation == "gelu":  # 如果激活函数为GELU
        act = GeluAndMul()  # 创建GELU激活函数实例
    else:  # 其他激活函数
        raise ValueError(f"Unsupported activation: {moe_runner_config.activation=}")  # 抛出不支持的激活函数异常

    # Get bias terms if available
    # 获取偏置项（如果可用）
    w13_bias = getattr(layer, "w13_weight_bias", None)  # 获取w13权重偏置，不存在则为None
    w2_bias = getattr(layer, "w2_weight_bias", None)  # 获取w2权重偏置，不存在则为None
    outputs = []  # 初始化输出列表，用于收集每个专家的计算结果
    start_idx = 0  # 初始化起始索引，用于在排序令牌中定位当前专家的令牌范围
    for i, num_tokens in enumerate(tokens_per_expert):  # 遍历每个专家及其处理的令牌数量
        end_idx = start_idx + num_tokens  # 计算当前专家令牌的结束索引
        if num_tokens == 0:  # 如果当前专家没有需要处理的令牌
            continue  # 跳过此专家
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]  # 获取当前专家需要处理的令牌子集

        layer_w13_weight = layer.w13_weight[i]  # 获取第i个专家的w13权重
        layer_w2_weight = layer.w2_weight[i]  # 获取第i个专家的w2权重

        # Store original dtype
        # 保存原始数据类型
        original_dtype = tokens_for_this_expert.dtype  # 记录令牌的原始数据类型，用于后续恢复

        # Get bias terms if available for this expert
        # 获取当前专家的偏置项（如果可用）
        layer_w13_bias = w13_bias[i] if w13_bias is not None else None  # 获取第i个专家的w13偏置
        layer_w2_bias = w2_bias[i] if w2_bias is not None else None  # 获取第i个专家的w2偏置

        # Apply w13 linear
        # 应用w13线性变换（门控+上投影）
        gate_up = F.linear(tokens_for_this_expert, layer_w13_weight)  # 计算gate_up = x @ w13.T

        # Add bias if present (for models like GPT-OSS)
        # 如果存在偏置则添加（用于GPT-OSS等模型）
        if layer_w13_bias is not None:  # 如果w13偏置存在
            gate_up_fp32 = gate_up.float() + layer_w13_bias  # 在FP32精度下添加偏置，避免精度损失
            gate_up = gate_up_fp32.to(original_dtype)  # 转回原始数据类型

        # Apply activation
        # 应用激活函数
        if (  # 如果使用带alpha参数的swiglu激活
            moe_runner_config.activation == "silu"  # 激活函数为SiLU
            and moe_runner_config.gemm1_alpha is not None  # 且gemm1_alpha参数不为None
        ):
            assert moe_runner_config.gemm1_clamp_limit is not None  # 断言clamp_limit参数也不为None
            gate_up = swiglu_gpt_oss_sigmoid_alpha(  # 使用GPT-OSS风格的带alpha参数的swiglu激活
                gate_up,  # gate_up张量
                moe_runner_config.gemm1_alpha,  # sigmoid alpha参数
                moe_runner_config.gemm1_clamp_limit,  # clamp限制值
            )
        else:  # 使用标准激活函数
            gate_up = act(gate_up)  # 应用标准SiLU或GELU激活函数

        # Apply w2 linear
        # 应用w2线性变换（下投影）
        expert_out = F.linear(gate_up, layer_w2_weight)  # 计算expert_out = gate_up @ w2.T

        # Add bias if present (for models like GPT-OSS)
        # 如果存在偏置则添加（用于GPT-OSS等模型）
        if layer_w2_bias is not None:  # 如果w2偏置存在
            expert_out = expert_out.float() + layer_w2_bias  # 在FP32精度下添加偏置，避免精度损失
            expert_out = expert_out.to(original_dtype)  # 转回原始数据类型

        outputs.append(expert_out)  # 将当前专家的输出添加到列表中
        start_idx = end_idx  # 更新起始索引，指向下一个专家的令牌起始位置

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)  # 拼接所有专家输出，若无输出则创建空张量
    new_x = torch.empty_like(outs)  # 创建与输出同形状的空张量，用于逆排序还原

    new_x[idxs] = outs  # 使用排序索引将输出还原为原始令牌顺序
    final_out = (  # 对topk专家输出进行加权求和
        new_x.view(*topk_ids.shape, -1)  # 将输出reshape为 [num_tokens, topk, hidden_dim]
        .type(topk_weights.dtype)  # 转换为与topk权重相同的数据类型
        .mul_(topk_weights.unsqueeze(dim=-1))  # 乘以topk路由权重（扩展维度以广播）
        .sum(dim=1)  # 在topk维度上求和，得到最终输出
        .type(new_x.dtype)  # 转回原始数据类型
    )
    return final_out  # 返回MoE最终输出
