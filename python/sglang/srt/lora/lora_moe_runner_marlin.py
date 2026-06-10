# Marlin MoE运行器核心，支持LoRA注入的钩子机制
# 使用Marlin int4/int8内核执行基础MoE投影，通过钩子注入LoRA增量
# 流水线: gate_up Marlin投影 -> LoRA钩子 -> SiLU激活 -> down Marlin投影 -> LoRA钩子 -> 归约
"""Marlin MoE runner core with hook support for LoRA injection.
# Marlin MoE运行器核心，支持LoRA注入的钩子。

Uses Marlin int4/int8 kernels for the base MoE projections.
# 使用Marlin int4/int8内核执行基础MoE投影。
LoRA deltas are injected via hooks.
# LoRA增量通过钩子注入。
"""

from __future__ import annotations # 启用延迟注解评估

from typing import TYPE_CHECKING, Optional # 导入类型检查和可选类型

import torch # 导入PyTorch库

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig # 导入MoE运行器配置
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo # 导入Marlin MoE量化信息
from sglang.srt.utils import is_cuda # 导入CUDA检测工具

if TYPE_CHECKING: # 仅用于类型检查
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput, # 标准合并输入
        StandardDispatchOutput, # 标准分发输出
    )

_is_cuda = is_cuda() # 检测是否为CUDA环境

if _is_cuda: # 如果是CUDA环境
    from sgl_kernel import silu_and_mul # 导入SiLU与乘法融合内核

    from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm # 导入MoE WNA16 Marlin GEMM内核
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        get_scalar_type, # 导入标量类型获取函数
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        moe_align_block_size, # 导入MoE块大小对齐函数
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
        moe_sum_reduce_triton, # 导入MoE求和归约Triton内核
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace # 导入Marlin工作空间创建工具


_MARLIN_WORKSPACE: Optional[torch.Tensor] = None # Marlin内核的全局工作空间


class MarlinLoraRunnerCore: # Marlin LoRA运行器核心类
    """
    MoE runner using Marlin kernels for base projections, with hooks for LoRA.
    # 使用Marlin内核进行基础投影的MoE运行器，带有LoRA钩子。

    Pipeline:
    # 流水线：
      1. moe_wna16_marlin_gemm (gate_up)
      # 1. moe_wna16_marlin_gemm（gate_up投影）
      1.5. hooks.after_gate_up
      # 1.5. 钩子.after_gate_up（注入LoRA增量）
      2. silu_and_mul
      # 2. silu_and_mul（SiLU激活与乘法）
      3. moe_wna16_marlin_gemm (down)
      # 3. moe_wna16_marlin_gemm（down投影）
      3.5. hooks.after_down
      # 3.5. 钩子.after_down（注入LoRA增量）
      4. moe_sum_reduce
      # 4. moe_sum_reduce（专家输出求和归约）
    """

    def __init__(self, config: MoeRunnerConfig): # 初始化方法，接收MoE运行器配置
        self.config = config # 保存配置

    def run_from_dispatch( # 从分发输出运行MoE+LoRA流水线
        self,
        dispatch_output: StandardDispatchOutput, # 分发输出（包含hidden_states和topk信息）
        quant_info: MarlinMoeQuantInfo, # Marlin量化信息
        runner_config: MoeRunnerConfig, # 运行器配置
        hooks=None, # LoRA钩子
    ) -> StandardCombineInput: # 返回标准合并输入
        global _MARLIN_WORKSPACE # 声明使用全局Marlin工作空间
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput # 导入标准合并输入类

        assert hooks is not None, "hooks must be provided for MarlinLoraRunnerCore" # 断言必须提供钩子

        hidden_states = dispatch_output.hidden_states # 获取隐藏状态
        topk_output = dispatch_output.topk_output # 获取topk输出
        topk_weights = topk_output.topk_weights # 获取topk权重
        topk_ids = topk_output.topk_ids # 获取topk专家ID

        assert runner_config.activation == "silu", "Only SiLU activation is supported." # 断言仅支持SiLU激活
        assert (
            torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
        ), "MarlinLoraRunnerCore requires CUDA compute capability >= 9" # 断言需要CUDA计算能力>=9
        inplace = runner_config.inplace # 是否原地操作
        routed_scaling_factor = runner_config.routed_scaling_factor # 路由缩放因子

        M, K = hidden_states.shape # 获取序列长度M和隐藏维度K
        E = quant_info.w13_qweight.shape[0] # 获取专家数量E
        N = quant_info.w2_qweight.shape[1] * 16 # 计算中间维度N
        topk = topk_ids.shape[1] # 获取topk数量
        num_bits = quant_info.weight_bits # 获取权重量化位数

        for block_size_m in [8, 16, 32, 48, 64]: # 遍历候选块大小
            if M * topk / E / block_size_m < 0.9: # 如果平均每块token数不足0.9
                break # 选择当前块大小

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, E
        ) # 对齐token到块大小，获取排序后的token ID和专家ID

        if ( # 检查是否需要创建或重新创建工作空间
            _MARLIN_WORKSPACE is None # 工作空间未创建
            or _MARLIN_WORKSPACE.device != hidden_states.device # 或设备不匹配
        ):
            _MARLIN_WORKSPACE = marlin_make_workspace(
                hidden_states.device, max_blocks_per_sm=4
            ) # 创建Marlin工作空间
        workspace = _MARLIN_WORKSPACE # 使用全局工作空间

        scalar_type1 = get_scalar_type(num_bits, quant_info.w13_qzeros is not None) # 获取gate/up权重的标量类型
        scalar_type2 = get_scalar_type(num_bits, quant_info.w2_qzeros is not None) # 获取down权重的标量类型

        # Stage 1: Gate/Up (Marlin)
        # 阶段1：Gate/Up投影（Marlin）
        intermediate_cache1 = torch.empty(
            (M * topk, 2 * N), device=hidden_states.device, dtype=hidden_states.dtype
        ) # 分配gate/up中间缓存，形状为(M*topk, 2*N)
        intermediate_cache1 = moe_wna16_marlin_gemm( # 执行Marlin GEMM gate/up投影
            hidden_states, # 输入隐藏状态
            intermediate_cache1, # 输出缓存
            quant_info.w13_qweight, # gate/up量化权重
            None, # 无偏置
            quant_info.w13_scales, # gate/up缩放因子
            None, # 无偏置缩放
            quant_info.w13_qzeros, # gate/up量化零点
            quant_info.w13_g_idx, # gate/up组索引
            quant_info.w13_g_idx_sort_indices, # gate/up组索引排序索引
            workspace, # Marlin工作空间
            sorted_token_ids, # 排序后的token ID
            expert_ids, # 专家ID
            num_tokens_post_padded, # 填充后的token数
            topk_weights, # topk权重
            moe_block_size=block_size_m, # MoE块大小
            top_k=topk, # topk数量
            mul_topk_weights=False, # 不乘topk权重（在归约时乘）
            is_ep=quant_info.expert_map is not None, # 是否使用专家并行
            b_q_type=scalar_type1, # 权重标量类型
            size_m=M, # M维度
            size_n=2 * N, # N维度
            size_k=K, # K维度
            is_k_full=quant_info.is_k_full, # K是否完整
            use_atomic_add=True, # 使用原子加法
            use_fp32_reduce=True, # 使用FP32归约
            is_zp_float=False, # 零点非浮点
        )

        # Hook: after gate_up
        # 钩子：gate_up投影后
        if hooks.after_gate_up: # 如果存在gate_up后钩子
            intermediate_cache1_3d = intermediate_cache1.view(M, topk, 2 * N) # 将中间缓存重塑为3D
            hooks.after_gate_up( # 调用gate_up后钩子，注入LoRA增量
                hidden_states, intermediate_cache1_3d, topk_weights, topk_ids
            )

        # Stage 2: Activation
        # 阶段2：激活函数（SiLU和乘法）
        intermediate_cache2 = torch.empty(
            (M * topk, N), device=hidden_states.device, dtype=hidden_states.dtype
        ) # 分配激活后中间缓存，形状为(M*topk, N)
        silu_and_mul(intermediate_cache1.view(-1, 2 * N), intermediate_cache2) # 执行SiLU激活与乘法

        # Stage 3: Down (Marlin)
        # 阶段3：Down投影（Marlin）
        intermediate_cache3 = torch.empty(
            (M * topk, K), device=hidden_states.device, dtype=hidden_states.dtype
        ) # 分配down投影中间缓存，形状为(M*topk, K)
        if quant_info.expert_map is not None: # 如果使用专家并行
            intermediate_cache3.zero_() # 初始化为零（原子加法需要）

        intermediate_cache3 = moe_wna16_marlin_gemm( # 执行Marlin GEMM down投影
            intermediate_cache2, # 激活后的中间缓存作为输入
            intermediate_cache3, # 输出缓存
            quant_info.w2_qweight, # down量化权重
            None, # 无偏置
            quant_info.w2_scales, # down缩放因子
            None, # 无偏置缩放
            quant_info.w2_qzeros, # down量化零点
            quant_info.w2_g_idx, # down组索引
            quant_info.w2_g_idx_sort_indices, # down组索引排序索引
            workspace, # Marlin工作空间
            sorted_token_ids, # 排序后的token ID
            expert_ids, # 专家ID
            num_tokens_post_padded, # 填充后的token数
            topk_weights, # topk权重
            moe_block_size=block_size_m, # MoE块大小
            top_k=1, # topk=1（down投影每个token只对应一个专家）
            mul_topk_weights=True, # 乘topk权重
            is_ep=quant_info.expert_map is not None, # 是否使用专家并行
            b_q_type=scalar_type2, # 权重标量类型
            size_m=M * topk, # M维度
            size_n=K, # N维度
            size_k=N, # K维度
            is_k_full=quant_info.is_k_full, # K是否完整
            use_atomic_add=True, # 使用原子加法
            use_fp32_reduce=True, # 使用FP32归约
            is_zp_float=False, # 零点非浮点
        )
        intermediate_cache3 = intermediate_cache3.view(M, topk, K) # 将down投影输出重塑为3D

        # Hook: after down
        # 钩子：down投影后
        if hooks.after_down: # 如果存在down后钩子
            hooks.after_down( # 调用down后钩子，注入LoRA增量
                intermediate_cache2, intermediate_cache3, topk_weights, topk_ids
            )

        # Stage 4: Reduction
        # 阶段4：归约（专家输出求和）
        output = hidden_states if inplace else torch.empty_like(hidden_states) # 原地模式或新建输出
        if routed_scaling_factor is None: # 如果未设置路由缩放因子
            routed_scaling_factor = 1.0 # 默认为1.0
        # NOTE: fusion opportunity here
        # 注意：此处有融合优化机会
        moe_sum_reduce_triton(intermediate_cache3, output, routed_scaling_factor) # 执行MoE求和归约

        return StandardCombineInput(hidden_states=output) # 返回标准合并输入