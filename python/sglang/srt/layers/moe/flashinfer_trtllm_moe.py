# FlashInfer TRT-LLM MoE 实现文件
# 使用 FlashInfer 的 TRT-LLM 风格内核实现 FP8 块级缩放和逐张量缩放的 MoE 计算
# 提供带自定义算子注册的包装函数，支持 torch.compile
from typing import Optional  # 导入可选类型 # 导入可选类型提示

import torch  # 导入 PyTorch # 导入 PyTorch 深度学习框架

from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册 # 导入自定义算子注册装饰器


def _fake_fp8_block_scale_moe(  # FP8 块级缩放 MoE 的 fake 实现 # FP8 块级缩放 MoE 的伪实现（用于 torch.compile）
    """FP8 块级缩放 MoE 的伪实现（用于 torch.compile 推导输出形状）"""
    routing_logits: torch.Tensor,  # 路由 logits # 路由器的 logits
    routing_bias: Optional[torch.Tensor],  # 路由偏置 # 路由偏置
    hidden_states: torch.Tensor,  # 隐藏状态 # 输入隐藏状态
    hidden_states_scale: torch.Tensor,  # 隐藏状态缩放因子 # 隐藏状态的量化缩放因子
    gemm1_weights: torch.Tensor,  # 第一个 GEMM 权重 # gate-up 投影权重
    gemm1_weights_scale: torch.Tensor,  # 第一个 GEMM 权重缩放 # gate-up 权重的缩放因子
    gemm2_weights: torch.Tensor,  # 第二个 GEMM 权重 # down 投影权重
    gemm2_weights_scale: torch.Tensor,  # 第二个 GEMM 权重缩放 # down 权重的缩放因子
    num_experts: int,  # 专家数量 # 专家总数
    top_k: int,  # top-k 值 # 每个token选择的专家数
    n_group: Optional[int],  # N 分组数 # N 维度的分组数
    topk_group: Optional[int],  # top-k 分组数 # 每组选择的专家数
    intermediate_size: int,  # 中间层大小 # 中间维度大小
    local_expert_offset: int,  # 本地专家偏移 # 本地专家的起始偏移
    local_num_experts: int,  # 本地专家数 # 本地专家数量
    routed_scaling_factor: Optional[float],  # 路由缩放因子 # 路由缩放因子
    routing_method_type: int = 0,  # 路由方法类型 # 路由方法类型
    use_shuffled_weight: bool = False,  # 是否使用混排权重 # 是否使用混排的权重布局
    weight_layout: int = 0,  # 权重布局 # 权重布局类型
    enable_pdl: Optional[bool] = None,  # 是否启用 PDL # 是否启用可编程数据流水线
    tune_max_num_tokens: int = 8192,  # 调优最大 token 数 # 调优时的最大 token 数
    fp8_quantization_type: Optional[int] = None,  # FP8 量化类型 # FP8 量化类型
    activation_type: Optional[int] = None,  # 激活函数类型 # 激活函数类型
) -> torch.Tensor:
    return torch.empty(  # 返回空张量 # 返回与输入同形状的空 BF16 张量
        hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device  # 形状和设备 # 保持与输入相同的形状和设备
    )


@register_custom_op(fake_impl=_fake_fp8_block_scale_moe)  # 注册自定义算子 # 注册为自定义算子并指定 fake 实现
def trtllm_fp8_block_scale_moe_wrapper(  # TRT-LLM FP8 块级缩放 MoE 包装函数 # TRT-LLM 风格的 FP8 块级缩放 MoE 计算包装器
    """TRT-LLM 风格的 FP8 块级缩放 MoE 计算包装函数""",
    routing_logits: torch.Tensor,  # 路由 logits # 路由器的 logits
    routing_bias: Optional[torch.Tensor],  # 路由偏置 # 路由偏置
    hidden_states: torch.Tensor,  # 隐藏状态 # 输入隐藏状态
    hidden_states_scale: torch.Tensor,  # 隐藏状态缩放因子 # 隐藏状态的量化缩放因子
    gemm1_weights: torch.Tensor,  # 第一个 GEMM 权重 # gate-up 投影权重
    gemm1_weights_scale: torch.Tensor,  # 第一个 GEMM 权重缩放 # gate-up 权重的缩放因子
    gemm2_weights: torch.Tensor,  # 第二个 GEMM 权重 # down 投影权重
    gemm2_weights_scale: torch.Tensor,  # 第二个 GEMM 权重缩放 # down 权重的缩放因子
    num_experts: int,  # 专家数量 # 专家总数
    top_k: int,  # top-k 值 # 每个token选择的专家数
    n_group: Optional[int],  # N 分组数 # N 维度的分组数
    topk_group: Optional[int],  # top-k 分组数 # 每组选择的专家数
    intermediate_size: int,  # 中间层大小 # 中间维度大小
    local_expert_offset: int,  # 本地专家偏移 # 本地专家的起始偏移
    local_num_experts: int,  # 本地专家数 # 本地专家数量
    routed_scaling_factor: Optional[float],  # 路由缩放因子 # 路由缩放因子
    routing_method_type: int = 0,  # 路由方法类型 # 路由方法类型
    use_shuffled_weight: bool = False,  # 是否使用混排权重 # 是否使用混排的权重布局
    weight_layout: int = 0,  # 权重布局 # 权重布局类型
    enable_pdl: Optional[bool] = None,  # 是否启用 PDL # 是否启用可编程数据流水线
    tune_max_num_tokens: int = 8192,  # 调优最大 token 数 # 调优时的最大 token 数
    fp8_quantization_type: Optional[int] = None,  # FP8 量化类型 # FP8 量化类型
    activation_type: Optional[int] = None,  # 激活函数类型 # 激活函数类型
) -> torch.Tensor:
    try:  # 尝试导入 # 尝试导入 FlashInfer 的 TRT-LLM MoE 函数
        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe  # 导入 TRT-LLM FP8 块级缩放 MoE # 导入 FlashInfer 的 TRT-LLM MoE 函数
    except ImportError as e:  # 导入失败 # 捕获导入异常
        raise ImportError(  # 抛出导入异常 # 抛出更详细的导入错误
            "Can't import trtllm_fp8_block_scale_moe from flashinfer. "  # 错误信息 # 无法导入的错误信息
            "Please check flashinfer version."  # 提示信息 # 提示检查 FlashInfer 版本
        ) from e  # 链接原始异常 # 保留原始异常链
    kwargs = {  # 构建参数字典 # 构建 FlashInfer 函数的关键字参数
        "routing_logits": routing_logits,  # 路由 logits # 路由器的 logits
        "routing_bias": routing_bias,  # 路由偏置 # 路由偏置
        "hidden_states": hidden_states,  # 隐藏状态 # 输入隐藏状态
        "hidden_states_scale": hidden_states_scale,  # 隐藏状态缩放 # 隐藏状态缩放因子
        "gemm1_weights": gemm1_weights,  # 第一个 GEMM 权重 # gate-up 投影权重
        "gemm1_weights_scale": gemm1_weights_scale,  # 第一个 GEMM 权重缩放 # gate-up 权重缩放因子
        "gemm2_weights": gemm2_weights,  # 第二个 GEMM 权重 # down 投影权重
        "gemm2_weights_scale": gemm2_weights_scale,  # 第二个 GEMM 权重缩放 # down 权重缩放因子
        "num_experts": num_experts,  # 专家数量 # 专家总数
        "top_k": top_k,  # top-k 值 # 每个token选择的专家数
        "n_group": n_group,  # N 分组数 # N 维度的分组数
        "topk_group": topk_group,  # top-k 分组数 # 每组选择的专家数
        "intermediate_size": intermediate_size,  # 中间层大小 # 中间维度大小
        "local_expert_offset": local_expert_offset,  # 本地专家偏移 # 本地专家的起始偏移
        "local_num_experts": local_num_experts,  # 本地专家数 # 本地专家数量
        "routed_scaling_factor": routed_scaling_factor,  # 路由缩放因子 # 路由缩放因子
        "routing_method_type": routing_method_type,  # 路由方法类型 # 路由方法类型
        "use_shuffled_weight": use_shuffled_weight,  # 是否使用混排权重 # 是否使用混排的权重布局
        "weight_layout": weight_layout,  # 权重布局 # 权重布局类型
        "enable_pdl": enable_pdl,  # 是否启用 PDL # 是否启用可编程数据流水线
        "tune_max_num_tokens": tune_max_num_tokens,  # 调优最大 token 数 # 调优时的最大 token 数
    }
    if fp8_quantization_type is not None:  # 有 FP8 量化类型 # 判断是否指定了 FP8 量化类型
        from flashinfer.fused_moe import Fp8QuantizationType  # 导入 FP8 量化类型枚举 # 导入 FlashInfer 的 FP8 量化类型枚举

        kwargs["fp8_quantization_type"] = Fp8QuantizationType(fp8_quantization_type)  # 设置 FP8 量化类型 # 将整数转为枚举值

    if activation_type is not None:  # 有激活函数类型 # 判断是否指定了激活函数类型
        from flashinfer.fused_moe.core import ActivationType  # 导入激活函数类型枚举 # 导入 FlashInfer 的激活函数类型枚举

        kwargs["activation_type"] = ActivationType(activation_type)  # 设置激活函数类型 # 将整数转为枚举值

    return trtllm_fp8_block_scale_moe(**kwargs)  # 调用 FlashInfer 函数 # 调用实际的 TRT-LLM MoE 函数


def _fake_fp8_block_scale_routed_moe(  # FP8 块级缩放路由 MoE 的 fake 实现 # FP8 块级缩放路由 MoE 的伪实现（用于 torch.compile）
    """FP8 块级缩放路由 MoE 的伪实现（用于 torch.compile 推导输出形状）"""
    topk_ids: torch.Tensor,  # TopK 专家 ID # 已选择的 top-k 专家 ID
    routing_bias: Optional[torch.Tensor],  # 路由偏置 # 路由偏置
    hidden_states: torch.Tensor,  # 隐藏状态 # 输入隐藏状态
    hidden_states_scale: torch.Tensor,  # 隐藏状态缩放因子 # 隐藏状态的量化缩放因子
    gemm1_weights: torch.Tensor,  # 第一个 GEMM 权重 # gate-up 投影权重
    gemm1_weights_scale: torch.Tensor,  # 第一个 GEMM 权重缩放 # gate-up 权重的缩放因子
    gemm2_weights: torch.Tensor,  # 第二个 GEMM 权重 # down 投影权重
    gemm2_weights_scale: torch.Tensor,  # 第二个 GEMM 权重缩放 # down 权重的缩放因子
    num_experts: int,  # 专家数量 # 专家总数
    top_k: int,  # top-k 值 # 每个token选择的专家数
    n_group: Optional[int],  # N 分组数 # N 维度的分组数
    topk_group: Optional[int],  # top-k 分组数 # 每组选择的专家数
    intermediate_size: int,  # 中间层大小 # 中间维度大小
    local_expert_offset: int,  # 本地专家偏移 # 本地专家的起始偏移
    local_num_experts: int,  # 本地专家数 # 本地专家数量
    routed_scaling_factor: Optional[float],  # 路由缩放因子 # 路由缩放因子
    routing_method_type: int = 0,  # 路由方法类型 # 路由方法类型
    use_shuffled_weight: bool = False,  # 是否使用混排权重 # 是否使用混排的权重布局
    weight_layout: int = 0,  # 权重布局 # 权重布局类型
    enable_pdl: Optional[bool] = None,  # 是否启用 PDL # 是否启用可编程数据流水线
    tune_max_num_tokens: int = 8192,  # 调优最大 token 数 # 调优时的最大 token 数
    fp8_quantization_type: Optional[int] = None,  # FP8 量化类型 # FP8 量化类型
    activation_type: Optional[int] = None,  # 激活函数类型 # 激活函数类型
) -> torch.Tensor:
    return torch.empty(  # 返回空张量 # 返回与输入同形状的空 BF16 张量
        hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device  # 形状和设备 # 保持与输入相同的形状和设备
    )


@register_custom_op(fake_impl=_fake_fp8_block_scale_routed_moe)  # 注册自定义算子 # 注册为自定义算子并指定 fake 实现
def trtllm_fp8_block_scale_routed_moe_wrapper(  # TRT-LLM FP8 块级缩放路由 MoE 包装函数 # TRT-LLM 风格的 FP8 块级缩放路由 MoE 计算包装器
    """TRT-LLM 风格的 FP8 块级缩放路由 MoE 计算包装函数""",
    topk_ids: torch.Tensor,  # TopK 专家 ID # 已选择的 top-k 专家 ID
    routing_bias: Optional[torch.Tensor],  # 路由偏置 # 路由偏置
    hidden_states: torch.Tensor,  # 隐藏状态 # 输入隐藏状态
    hidden_states_scale: torch.Tensor,  # 隐藏状态缩放因子 # 隐藏状态的量化缩放因子
    gemm1_weights: torch.Tensor,  # 第一个 GEMM 权重 # gate-up 投影权重
    gemm1_weights_scale: torch.Tensor,  # 第一个 GEMM 权重缩放 # gate-up 权重的缩放因子
    gemm2_weights: torch.Tensor,  # 第二个 GEMM 权重 # down 投影权重
    gemm2_weights_scale: torch.Tensor,  # 第二个 GEMM 权重缩放 # down 权重的缩放因子
    num_experts: int,  # 专家数量 # 专家总数
    top_k: int,  # top-k 值 # 每个token选择的专家数
    n_group: Optional[int],  # N 分组数 # N 维度的分组数
    topk_group: Optional[int],  # top-k 分组数 # 每组选择的专家数
    intermediate_size: int,  # 中间层大小 # 中间维度大小
    local_expert_offset: int,  # 本地专家偏移 # 本地专家的起始偏移
    local_num_experts: int,  # 本地专家数 # 本地专家数量
    routed_scaling_factor: Optional[float],  # 路由缩放因子 # 路由缩放因子
    routing_method_type: int = 0,  # 路由方法类型 # 路由方法类型
    use_shuffled_weight: bool = False,  # 是否使用混排权重 # 是否使用混排的权重布局
    weight_layout: int = 0,  # 权重布局 # 权重布局类型
    enable_pdl: Optional[bool] = None,  # 是否启用 PDL # 是否启用可编程数据流水线
    tune_max_num_tokens: int = 8192,  # 调优最大 token 数 # 调优时的最大 token 数
    fp8_quantization_type: Optional[int] = None,  # FP8 量化类型 # FP8 量化类型
    activation_type: Optional[int] = None,  # 激活函数类型 # 激活函数类型
) -> torch.Tensor:
    try:  # 尝试导入 # 尝试导入 FlashInfer 的路由 MoE 函数
        from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe  # 导入 TRT-LLM FP8 块级缩放路由 MoE # 导入 FlashInfer 的路由 MoE 函数
    except ImportError as e:  # 导入失败 # 捕获导入异常
        raise ImportError(  # 抛出导入异常 # 抛出更详细的导入错误
            "Can't import trtllm_fp8_block_scale_routed_moe from flashinfer. "  # 错误信息 # 无法导入的错误信息
            "Please check flashinfer version."  # 提示信息 # 提示检查 FlashInfer 版本
        ) from e  # 链接原始异常 # 保留原始异常链
    kwargs = {  # 构建参数字典 # 构建 FlashInfer 函数的关键字参数
        "topk_ids": topk_ids,  # TopK 专家 ID # 已选择的 top-k 专家 ID
        "routing_bias": routing_bias,  # 路由偏置 # 路由偏置
        "hidden_states": hidden_states,  # 隐藏状态 # 输入隐藏状态
        "hidden_states_scale": hidden_states_scale,  # 隐藏状态缩放 # 隐藏状态缩放因子
        "gemm1_weights": gemm1_weights,  # 第一个 GEMM 权重 # gate-up 投影权重
        "gemm1_weights_scale": gemm1_weights_scale,  # 第一个 GEMM 权重缩放 # gate-up 权重缩放因子
        "gemm2_weights": gemm2_weights,  # 第二个 GEMM 权重 # down 投影权重
        "gemm2_weights_scale": gemm2_weights_scale,  # 第二个 GEMM 权重缩放 # down 权重缩放因子
        "num_experts": num_experts,  # 专家数量 # 专家总数
        "top_k": top_k,  # top-k 值 # 每个token选择的专家数
        "n_group": n_group,  # N 分组数 # N 维度的分组数
        "topk_group": topk_group,  # top-k 分组数 # 每组选择的专家数
        "intermediate_size": intermediate_size,  # 中间层大小 # 中间维度大小
        "local_expert_offset": local_expert_offset,  # 本地专家偏移 # 本地专家的起始偏移
        "local_num_experts": local_num_experts,  # 本地专家数 # 本地专家数量
        "routed_scaling_factor": routed_scaling_factor,  # 路由缩放因子 # 路由缩放因子
        "routing_method_type": routing_method_type,  # 路由方法类型 # 路由方法类型
        "use_shuffled_weight": use_shuffled_weight,  # 是否使用混排权重 # 是否使用混排的权重布局
        "weight_layout": weight_layout,  # 权重布局 # 权重布局类型
        "enable_pdl": enable_pdl,  # 是否启用 PDL # 是否启用可编程数据流水线
        "tune_max_num_tokens": tune_max_num_tokens,  # 调优最大 token 数 # 调优时的最大 token 数
    }
    if fp8_quantization_type is not None:  # 有 FP8 量化类型 # 判断是否指定了 FP8 量化类型
        from flashinfer.fused_moe import Fp8QuantizationType  # 导入 FP8 量化类型枚举 # 导入 FlashInfer 的 FP8 量化类型枚举

        kwargs["fp8_quantization_type"] = Fp8QuantizationType(fp8_quantization_type)  # 设置 FP8 量化类型 # 将整数转为枚举值

    if activation_type is not None:  # 有激活函数类型 # 判断是否指定了激活函数类型
        from flashinfer.fused_moe.core import ActivationType  # 导入激活函数类型枚举 # 导入 FlashInfer 的激活函数类型枚举

        kwargs["activation_type"] = ActivationType(activation_type)  # 设置激活函数类型 # 将整数转为枚举值

    return trtllm_fp8_block_scale_routed_moe(**kwargs)  # 调用 FlashInfer 函数 # 调用实际的路由 MoE 函数


def _fake_fp8_per_tensor_scale_moe(  # FP8 逐张量缩放 MoE 的 fake 实现 # FP8 逐张量缩放 MoE 的伪实现（用于 torch.compile）
    """FP8 逐张量缩放 MoE 的伪实现（用于 torch.compile 推导输出形状）"""
    routing_logits: torch.Tensor,  # 路由 logits # 路由器的 logits
    routing_bias: Optional[torch.Tensor],  # 路由偏置 # 路由偏置
    hidden_states: torch.Tensor,  # 隐藏状态 # 输入隐藏状态
    gemm1_weights: torch.Tensor,  # 第一个 GEMM 权重 # gate-up 投影权重
    output1_scales_scalar: torch.Tensor,  # 第一个 GEMM 输出缩放 # gate-up 输出的逐张量缩放因子
    output1_scales_gate_scalar: torch.Tensor,  # 第一个 GEMM gate 输出缩放 # gate 分支输出的逐张量缩放因子
    gemm2_weights: torch.Tensor,  # 第二个 GEMM 权重 # down 投影权重
    output2_scales_scalar: torch.Tensor,  # 第二个 GEMM 输出缩放 # down 输出的逐张量缩放因子
    num_experts: int,  # 专家数量 # 专家总数
    top_k: int,  # top-k 值 # 每个token选择的专家数
    n_group: Optional[int],  # N 分组数 # N 维度的分组数
    topk_group: Optional[int],  # top-k 分组数 # 每组选择的专家数
    intermediate_size: int,  # 中间层大小 # 中间维度大小
    local_expert_offset: int,  # 本地专家偏移 # 本地专家的起始偏移
    local_num_experts: int,  # 本地专家数 # 本地专家数量
    routed_scaling_factor: Optional[float],  # 路由缩放因子 # 路由缩放因子
    use_routing_scales_on_input: bool,  # 是否在输入上使用路由缩放 # 是否将路由缩放应用于输入
    routing_method_type: int = 0,  # 路由方法类型 # 路由方法类型
    enable_pdl: Optional[bool] = None,  # 是否启用 PDL # 是否启用可编程数据流水线
    tune_max_num_tokens: int = 8192,  # 调优最大 token 数 # 调优时的最大 token 数
    activation_type: Optional[int] = None,  # 激活函数类型 # 激活函数类型
) -> torch.Tensor:
    return torch.empty(  # 返回空张量 # 返回与输入同形状的空 BF16 张量
        hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device  # 形状和设备 # 保持与输入相同的形状和设备
    )


@register_custom_op(fake_impl=_fake_fp8_per_tensor_scale_moe)  # 注册自定义算子 # 注册为自定义算子并指定 fake 实现
def trtllm_fp8_per_tensor_scale_moe_wrapper(  # TRT-LLM FP8 逐张量缩放 MoE 包装函数 # TRT-LLM 风格的 FP8 逐张量缩放 MoE 计算包装器
    """TRT-LLM 风格的 FP8 逐张量缩放 MoE 计算包装函数""",
    routing_logits: torch.Tensor,  # 路由 logits # 路由器的 logits
    routing_bias: Optional[torch.Tensor],  # 路由偏置 # 路由偏置
    hidden_states: torch.Tensor,  # 隐藏状态 # 输入隐藏状态
    gemm1_weights: torch.Tensor,  # 第一个 GEMM 权重 # gate-up 投影权重
    output1_scales_scalar: torch.Tensor,  # 第一个 GEMM 输出缩放 # gate-up 输出的逐张量缩放因子
    output1_scales_gate_scalar: torch.Tensor,  # 第一个 GEMM gate 输出缩放 # gate 分支输出的逐张量缩放因子
    gemm2_weights: torch.Tensor,  # 第二个 GEMM 权重 # down 投影权重
    output2_scales_scalar: torch.Tensor,  # 第二个 GEMM 输出缩放 # down 输出的逐张量缩放因子
    num_experts: int,  # 专家数量 # 专家总数
    top_k: int,  # top-k 值 # 每个token选择的专家数
    n_group: Optional[int],  # N 分组数 # N 维度的分组数
    topk_group: Optional[int],  # top-k 分组数 # 每组选择的专家数
    intermediate_size: int,  # 中间层大小 # 中间维度大小
    local_expert_offset: int,  # 本地专家偏移 # 本地专家的起始偏移
    local_num_experts: int,  # 本地专家数 # 本地专家数量
    routed_scaling_factor: Optional[float],  # 路由缩放因子 # 路由缩放因子
    use_routing_scales_on_input: bool,  # 是否在输入上使用路由缩放 # 是否将路由缩放应用于输入
    routing_method_type: int = 0,  # 路由方法类型 # 路由方法类型
    enable_pdl: Optional[bool] = None,  # 是否启用 PDL # 是否启用可编程数据流水线
    tune_max_num_tokens: int = 8192,  # 调优最大 token 数 # 调优时的最大 token 数
    activation_type: Optional[int] = None,  # 激活函数类型 # 激活函数类型
) -> torch.Tensor:
    # lazy import
    # 延迟导入
    try:  # 尝试导入 # 尝试导入 FlashInfer 的逐张量缩放 MoE 函数
        from flashinfer.fused_moe import trtllm_fp8_per_tensor_scale_moe  # 导入 TRT-LLM FP8 逐张量缩放 MoE # 导入 FlashInfer 的逐张量缩放 MoE 函数
    except ImportError as e:  # 导入失败 # 捕获导入异常
        raise ImportError(  # 抛出导入异常 # 抛出更详细的导入错误
            "Can't import trtllm_fp8_per_tensor_scale_moe from flashinfer. "  # 错误信息 # 无法导入的错误信息
            "Please check flashinfer version."  # 提示信息 # 提示检查 FlashInfer 版本
        ) from e  # 链接原始异常 # 保留原始异常链

    kwargs = {  # 构建参数字典 # 构建 FlashInfer 函数的关键字参数
        "routing_logits": routing_logits,  # 路由 logits # 路由器的 logits
        "routing_bias": routing_bias,  # 路由偏置 # 路由偏置
        "hidden_states": hidden_states,  # 隐藏状态 # 输入隐藏状态
        "gemm1_weights": gemm1_weights,  # 第一个 GEMM 权重 # gate-up 投影权重
        "output1_scales_scalar": output1_scales_scalar,  # 第一个 GEMM 输出缩放 # gate-up 输出缩放
        "output1_scales_gate_scalar": output1_scales_gate_scalar,  # 第一个 GEMM gate 缩放 # gate 分支缩放
        "gemm2_weights": gemm2_weights,  # 第二个 GEMM 权重 # down 投影权重
        "output2_scales_scalar": output2_scales_scalar,  # 第二个 GEMM 输出缩放 # down 输出缩放
        "num_experts": num_experts,  # 专家数量 # 专家总数
        "top_k": top_k,  # top-k 值 # 每个token选择的专家数
        "n_group": n_group,  # N 分组数 # N 维度的分组数
        "topk_group": topk_group,  # top-k 分组数 # 每组选择的专家数
        "intermediate_size": intermediate_size,  # 中间层大小 # 中间维度大小
        "local_expert_offset": local_expert_offset,  # 本地专家偏移 # 本地专家的起始偏移
        "local_num_experts": local_num_experts,  # 本地专家数 # 本地专家数量
        "routed_scaling_factor": routed_scaling_factor,  # 路由缩放因子 # 路由缩放因子
        "use_routing_scales_on_input": use_routing_scales_on_input,  # 在输入上使用路由缩放 # 是否在输入上应用路由缩放
        "routing_method_type": routing_method_type,  # 路由方法类型 # 路由方法类型
        "enable_pdl": enable_pdl,  # 是否启用 PDL # 是否启用可编程数据流水线
        "tune_max_num_tokens": tune_max_num_tokens,  # 调优最大 token 数 # 调优时的最大 token 数
    }

    if activation_type is not None:  # 有激活函数类型 # 判断是否指定了激活函数类型
        from flashinfer.fused_moe.core import ActivationType  # 导入激活函数类型枚举 # 导入 FlashInfer 的激活函数类型枚举

        kwargs["activation_type"] = ActivationType(activation_type)  # 设置激活函数类型 # 将整数转为枚举值

    return trtllm_fp8_per_tensor_scale_moe(**kwargs)  # 调用 FlashInfer 函数 # 调用实际的逐张量缩放 MoE 函数
