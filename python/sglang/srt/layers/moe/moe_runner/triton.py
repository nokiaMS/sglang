# Triton MoE 运行器模块 - 定义了基于 Triton 后端的 MoE（混合专家）运行器核心类、输入/输出数据结构及注册的置换/融合钩子函数
from __future__ import annotations  # 启用延迟注解评估，支持前向引用类型

from dataclasses import dataclass  # 用于自动生成数据类的特殊方法
from typing import TYPE_CHECKING, Any, List, Optional  # 类型提示工具

import torch  # PyTorch 深度学习框架

from sglang.srt.layers.moe.moe_runner.base import (  # 从 MoE 运行器基类导入核心抽象
    MoeQuantInfo,  # 量化信息基类
    MoeRunnerConfig,  # 运行器配置基类
    MoeRunnerCore,  # 运行器核心基类
    RunnerInput,  # 运行器输入基类
    RunnerOutput,  # 运行器输出基类
    register_fused_func,  # 注册融合函数的装饰器
    register_post_permute,  # 注册后置换函数的装饰器
    register_pre_permute,  # 注册前置换函数的装饰器
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend  # MoE 运行器后端枚举

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher.standard import (  # 标准令牌分发器类型
        StandardCombineInput,  # 标准合并输入
        StandardDispatchOutput,  # 标准分发输出
    )


@dataclass  # Triton 运行器输入数据类
class TritonRunnerInput(RunnerInput):  # 继承运行器输入基类

    hidden_states: torch.Tensor  # 隐藏状态张量
    topk_weights: torch.Tensor  # Top-K 路由权重
    topk_ids: torch.Tensor  # Top-K 专家ID
    sorted_token_ids: torch.Tensor  # 排序后的令牌ID（按专家分组）
    expert_ids: torch.Tensor  # 专家ID列表
    num_tokens_post_padded: torch.Tensor  # 填充后的令牌数量

    @property
    def runner_backend(self) -> MoeRunnerBackend:  # 返回运行器后端类型
        return MoeRunnerBackend.TRITON  # 返回 Triton 后端标识


@dataclass  # Triton 运行器输出数据类
class TritonRunnerOutput(RunnerOutput):  # 继承运行器输出基类

    hidden_states: torch.Tensor  # 输出的隐藏状态张量

    @property
    def runner_backend(self) -> MoeRunnerBackend:  # 返回运行器后端类型
        return MoeRunnerBackend.TRITON  # 返回 Triton 后端标识


@dataclass  # Triton MoE 量化信息数据类
class TritonMoeQuantInfo(MoeQuantInfo):  # 继承量化信息基类
    w13_weight: torch.Tensor  # w1+w3 融合权重（门控+上投影）
    w2_weight: torch.Tensor  # w2 下投影权重
    b13: Optional[torch.Tensor] = None  # w1+w3 偏置（可选）
    b2: Optional[torch.Tensor] = None  # w2 偏置（可选）
    use_fp8_w8a8: bool = False  # 是否使用 FP8 权重8位激活8位量化
    use_int8_w8a8: bool = False  # 是否使用 INT8 权重8位激活8位量化
    use_int8_w8a16: bool = False  # 是否使用 INT8 权重8位激活16位量化
    use_int4_w4a16: bool = False  # 是否使用 INT4 权重4位激活16位量化
    per_channel_quant: bool = False  # 是否使用逐通道量化
    w13_scale: Optional[torch.Tensor] = None  # w1+w3 量化缩放因子（可选）
    w2_scale: Optional[torch.Tensor] = None  # w2 量化缩放因子（可选）
    w13_zp: Optional[torch.Tensor] = None  # w1+w3 量化零点（可选）
    w2_zp: Optional[torch.Tensor] = None  # w2 量化零点（可选）
    a13_scale: Optional[torch.Tensor] = None  # 激活缩放因子用于w13（可选）
    a2_scale: Optional[torch.Tensor] = None  # 激活缩放因子用于w2（可选）
    block_shape: Optional[List[int]] = None  # 块状量化的块形状（可选）


class TritonRunnerCore(MoeRunnerCore):  # Triton MoE 运行器核心类，继承运行器核心基类

    def __init__(self, config: MoeRunnerConfig):  # 初始化 Triton 运行器核心
        super().__init__(config)  # 调用父类初始化

    def run(  # 执行 MoE 前向计算
        self,
        runner_input: TritonRunnerInput,  # Triton 运行器输入
        quant_info: TritonMoeQuantInfo,  # Triton 量化信息
        running_state: dict,  # 运行时状态字典
        hooks: Optional[Any] = None,  # 可选的钩子函数
    ) -> TritonRunnerOutput:  # 返回 Triton 运行器输出
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (  # 延迟导入融合 MoE 核序列函数
            _fused_moe_kernel_sequence,
        )

        filter_expert = (  # 判断是否需要过滤专家（当配置的专家数与本地专家数不同时）
            self.config.num_experts is None  # 专家数为None时需要过滤
            or self.config.num_experts != self.config.num_local_experts  # 或专家数与本地专家数不一致时
        )

        out = _fused_moe_kernel_sequence(  # 调用融合 MoE 核序列，执行完整的 MoE 计算
            runner_input.hidden_states,  # 隐藏状态
            quant_info.w13_weight,  # w1+w3 权重
            quant_info.w2_weight,  # w2 权重
            runner_input.topk_weights,  # Top-K 权重
            runner_input.topk_ids,  # Top-K 专家ID
            runner_input.sorted_token_ids,  # 排序后的令牌ID
            runner_input.expert_ids,  # 专家ID
            runner_input.num_tokens_post_padded,  # 填充后令牌数
            running_state["config"],  # 内核配置
            running_state.get("down_config"),  # 下投影内核配置（可选）
            running_state.get("down_moe_use_tma", False),  # 下投影是否使用 TMA（可选）
            b1=quant_info.b13,  # w1+w3 偏置
            b2=quant_info.b2,  # w2 偏置
            use_fp8_w8a8=quant_info.use_fp8_w8a8,  # FP8 量化标志
            use_int8_w8a8=quant_info.use_int8_w8a8,  # INT8 W8A8 量化标志
            use_int8_w8a16=quant_info.use_int8_w8a16,  # INT8 W8A16 量化标志
            use_int4_w4a16=quant_info.use_int4_w4a16,  # INT4 W4A16 量化标志
            per_channel_quant=quant_info.per_channel_quant,  # 逐通道量化标志
            w1_scale=quant_info.w13_scale,  # w1+w3 权重缩放因子
            w2_scale=quant_info.w2_scale,  # w2 权重缩放因子
            w1_zp=quant_info.w13_zp,  # w1+w3 零点
            w2_zp=quant_info.w2_zp,  # w2 零点
            a1_scale=quant_info.a13_scale,  # w1+w3 激活缩放因子
            a2_scale=quant_info.a2_scale,  # w2 激活缩放因子
            block_shape=quant_info.block_shape,  # 块状量化形状
            activation=self.config.activation,  # 激活函数类型
            is_gated=self.config.is_gated,  # 是否为门控 MoE
            no_combine=self.config.no_combine,  # 是否跳过合并步骤
            inplace=self.config.inplace,  # 是否原地操作
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,  # 是否在输入上应用路由权重
            routed_scaling_factor=self.config.routed_scaling_factor,  # 路由缩放因子
            gemm1_alpha=self.config.gemm1_alpha,  # GEMM1 alpha 参数
            gemm1_limit=self.config.gemm1_clamp_limit,  # GEMM1 钳位限制
            filter_expert=filter_expert,  # 是否过滤专家
            hooks=hooks,  # 钩子函数
            swiglu_limit=self.config.swiglu_limit,  # SwiGLU 钳位限制
        )

        return TritonRunnerOutput(hidden_states=out)  # 将输出包装为 TritonRunnerOutput 返回

    @property
    def runner_backend(self) -> MoeRunnerBackend:  # 返回运行器后端类型
        return MoeRunnerBackend.TRITON  # 返回 Triton 后端标识


@register_fused_func("none", "triton")  # 注册融合函数：从 none 格式到 triton 后端
def fused_experts_none_to_triton(  # 将标准分发输出直接融合为 Triton MoE 计算结果
    dispatch_output: StandardDispatchOutput,  # 标准分发输出
    quant_info: TritonMoeQuantInfo,  # Triton 量化信息
    runner_config: MoeRunnerConfig,  # MoE 运行器配置
) -> StandardCombineInput:  # 返回标准合并输入
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts  # 延迟导入融合专家函数
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput  # 延迟导入标准合并输入

    output = fused_experts(  # 调用融合专家函数执行 MoE 计算
        hidden_states=dispatch_output.hidden_states,  # 隐藏状态
        w1=quant_info.w13_weight,  # w1+w3 权重
        w2=quant_info.w2_weight,  # w2 权重
        topk_output=dispatch_output.topk_output,  # Top-K 输出
        moe_runner_config=runner_config,  # 运行器配置
        b1=quant_info.b13,  # w1+w3 偏置
        b2=quant_info.b2,  # w2 偏置
        use_fp8_w8a8=quant_info.use_fp8_w8a8,  # FP8 量化标志
        use_int8_w8a8=quant_info.use_int8_w8a8,  # INT8 W8A8 量化标志
        use_int8_w8a16=quant_info.use_int8_w8a16,  # INT8 W8A16 量化标志
        use_int4_w4a16=quant_info.use_int4_w4a16,  # INT4 W4A16 量化标志
        per_channel_quant=quant_info.per_channel_quant,  # 逐通道量化标志
        w1_scale=quant_info.w13_scale,  # w1+w3 权重缩放因子
        w2_scale=quant_info.w2_scale,  # w2 权重缩放因子
        w1_zp=quant_info.w13_zp,  # w1+w3 零点
        w2_zp=quant_info.w2_zp,  # w2 零点
        a1_scale=quant_info.a13_scale,  # w1+w3 激活缩放因子
        a2_scale=quant_info.a2_scale,  # w2 激活缩放因子
        block_shape=quant_info.block_shape,  # 块状量化形状
    )

    return StandardCombineInput(  # 将输出包装为标准合并输入返回
        hidden_states=output,  # MoE 计算后的隐藏状态
    )


@register_pre_permute("standard", "triton")  # 注册前置换函数：从 standard 格式到 triton 后端
def pre_permute_standard_to_triton(  # 将标准分发输出转换为 Triton 运行器输入
    dispatch_output: StandardDispatchOutput,  # 标准分发输出
    quant_info: TritonMoeQuantInfo,  # Triton 量化信息
    runner_config: MoeRunnerConfig,  # MoE 运行器配置
    running_state: dict,  # 运行时状态字典
) -> TritonRunnerInput:  # 返回 Triton 运行器输入

    # NOTE: this is dead code as a fused func for standard format is registered.
    # 注意：这是死代码，因为标准格式的融合函数已注册。
    # This is left here for testing and examples.
    # 此处保留用于测试和示例。

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (  # 延迟导入准备融合 MoE 运行的函数
        _prepare_fused_moe_run,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker  # 延迟导入 TopK 输出检查器

    hidden_states, topk_output = (  # 从分发输出中提取隐藏状态和 TopK 输出
        dispatch_output.hidden_states,  # 隐藏状态
        dispatch_output.topk_output,  # TopK 输出
    )

    assert TopKOutputChecker.format_is_standard(topk_output)  # 断言 TopK 输出格式为标准格式

    (  # 调用准备函数，获取运行所需的配置和对齐数据
        config,  # 内核配置
        down_config,  # 下投影内核配置
        down_moe_use_tma,  # 下投影是否使用 TMA
        sorted_token_ids,  # 排序后的令牌ID
        expert_ids,  # 专家ID
        num_tokens_post_padded,  # 填充后令牌数
    ) = _prepare_fused_moe_run(  # 准备融合 MoE 运行
        hidden_states,  # 隐藏状态
        quant_info.w13_weight,  # w1+w3 权重
        quant_info.w2_weight,  # w2 权重
        topk_output.topk_ids,  # Top-K 专家ID
        use_fp8_w8a8=quant_info.use_fp8_w8a8,  # FP8 量化标志
        use_int8_w8a8=quant_info.use_int8_w8a8,  # INT8 W8A8 量化标志
        use_int8_w8a16=quant_info.use_int8_w8a16,  # INT8 W8A16 量化标志
        use_int4_w4a16=quant_info.use_int4_w4a16,  # INT4 W4A16 量化标志
        per_channel_quant=quant_info.per_channel_quant,  # 逐通道量化标志
        block_shape=quant_info.block_shape,  # 块状量化形状
    )

    running_state["config"] = config  # 保存内核配置到运行状态
    running_state["down_config"] = down_config  # 保存下投影配置到运行状态
    running_state["down_moe_use_tma"] = down_moe_use_tma  # 保存 TMA 标志到运行状态

    return TritonRunnerInput(  # 返回 Triton 运行器输入
        hidden_states=hidden_states,  # 隐藏状态
        topk_weights=topk_output.topk_weights,  # Top-K 权重
        topk_ids=topk_output.topk_ids,  # Top-K 专家ID
        sorted_token_ids=sorted_token_ids,  # 排序后的令牌ID
        expert_ids=expert_ids,  # 专家ID
        num_tokens_post_padded=num_tokens_post_padded,  # 填充后令牌数
    )


@register_post_permute("triton", "standard")  # 注册后置换函数：从 triton 后端到 standard 格式
def post_permute_triton_to_standard(  # 将 Triton 运行器输出转换为标准合并输入
    runner_output: TritonRunnerOutput,  # Triton 运行器输出
    quant_info: TritonMoeQuantInfo,  # Triton 量化信息
    runner_config: MoeRunnerConfig,  # MoE 运行器配置
    running_state: dict,  # 运行时状态字典
) -> StandardCombineInput:  # 返回标准合并输入

    # NOTE: this is dead code as a fused func for standard format is registered.
    # 注意：这是死代码，因为标准格式的融合函数已注册。
    # This is left here for testing and examples.
    # 此处保留用于测试和示例。

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput  # 延迟导入标准合并输入

    return StandardCombineInput(  # 返回标准合并输入
        hidden_states=runner_output.hidden_states,  # 使用运行器输出的隐藏状态
    )
