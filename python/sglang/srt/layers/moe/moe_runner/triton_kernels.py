# Triton Kernels MoE 运行器后端骨架 - 基于 triton_kernels 外部包实现 MoE 专家执行，包含运行器输入/输出数据类、量化信息及置换钩子函数
"""Triton kernels MoE runner backend skeleton."""  # 英文文档字符串：Triton 内核 MoE 运行器后端骨架

from __future__ import annotations  # 启用延迟注解评估，支持前向引用类型

from dataclasses import dataclass  # 用于自动生成数据类的特殊方法
from typing import TYPE_CHECKING, Any, Optional  # 类型提示工具

import torch  # PyTorch 深度学习框架

from sglang.srt.layers.moe.moe_runner.base import (  # 从 MoE 运行器基类导入核心抽象
    MoeQuantInfo,  # 量化信息基类
    MoeRunnerConfig,  # 运行器配置基类
    MoeRunnerCore,  # 运行器核心基类
    RunnerInput,  # 运行器输入基类
    RunnerOutput,  # 运行器输出基类
    register_post_permute,  # 注册后置换函数的装饰器
    register_pre_permute,  # 注册前置换函数的装饰器
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend  # MoE 运行器后端枚举

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from triton_kernels.matmul_ogs import (  # triton_kernels 包的矩阵乘法相关类型
        GatherIndx,  # 聚合索引
        PrecisionConfig,  # 精度配置
        RoutingData,  # 路由数据
        ScatterIndx,  # 散射索引
    )

    from sglang.srt.layers.moe.token_dispatcher.standard import (  # 标准令牌分发器类型
        StandardCombineInput,  # 标准合并输入
        StandardDispatchOutput,  # 标准分发输出
    )


# ---------------------------------------------------------------------------
# Runner IO dataclasses
# 运行器输入/输出数据类
# ---------------------------------------------------------------------------


@dataclass  # Triton Kernels 运行器输入数据类
class TritonKernelsRunnerInput(RunnerInput):  # 继承运行器输入基类
    """Input bundle passed to the triton-kernels runner core."""  # 传递给 triton-kernels 运行器核心的输入数据包

    hidden_states: torch.Tensor  # 隐藏状态张量
    routing_data: "RoutingData"  # 路由数据，包含专家分配信息
    gather_indx: "GatherIndx"  # 聚合索引，用于令牌收集
    scatter_indx: "ScatterIndx"  # 散射索引，用于结果分发

    @property
    def runner_backend(self) -> MoeRunnerBackend:  # 返回运行器后端类型
        return MoeRunnerBackend.TRITON_KERNELS  # 返回 TRITON_KERNELS 后端标识


@dataclass  # Triton Kernels 运行器输出数据类
class TritonKernelsRunnerOutput(RunnerOutput):  # 继承运行器输出基类
    """Output bundle returned from the triton-kernels runner core."""  # 从 triton-kernels 运行器核心返回的输出数据包

    hidden_states: torch.Tensor  # 输出的隐藏状态张量

    @property
    def runner_backend(self) -> MoeRunnerBackend:  # 返回运行器后端类型
        return MoeRunnerBackend.TRITON_KERNELS  # 返回 TRITON_KERNELS 后端标识


@dataclass  # Triton Kernels 量化信息数据类
class TritonKernelsQuantInfo(MoeQuantInfo):  # 继承量化信息基类
    """Quantization payload consumed by the triton-kernels backend."""  # triton-kernels 后端使用的量化载荷

    w13_weight: torch.Tensor  # w1+w3 融合权重（门控+上投影）
    w2_weight: torch.Tensor  # w2 下投影权重
    w13_bias: Optional[torch.Tensor] = None  # w1+w3 偏置（可选）
    w2_bias: Optional[torch.Tensor] = None  # w2 偏置（可选）
    w13_precision_config: Optional[PrecisionConfig] = None  # w1+w3 精度配置（可选）
    w2_precision_config: Optional[PrecisionConfig] = None  # w2 精度配置（可选）
    global_num_experts: int = -1  # 全局专家数量，默认-1表示未设置


# ---------------------------------------------------------------------------
# Runner core
# 运行器核心
# ---------------------------------------------------------------------------


class TritonKernelsRunnerCore(MoeRunnerCore):  # Triton Kernels MoE 运行器核心类，继承运行器核心基类
    """Execute MoE experts via the external triton_kernels package."""  # 通过外部 triton_kernels 包执行 MoE 专家

    def run(  # 执行 MoE 前向计算
        self,
        runner_input: TritonKernelsRunnerInput,  # Triton Kernels 运行器输入
        quant_info: TritonKernelsQuantInfo,  # Triton Kernels 量化信息
        running_state: dict,  # 运行时状态字典
        hooks: Optional[Any] = None,  # 可选的钩子函数
    ) -> TritonKernelsRunnerOutput:  # 返回 Triton Kernels 运行器输出
        from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (  # 延迟导入 triton_kernels 的融合专家函数
            triton_kernel_fused_experts,  # 无偏置版本
            triton_kernel_fused_experts_with_bias,  # 带偏置版本
        )

        assert (  # 断言仅支持门控 MoE
            self.config.is_gated
        ), "Only gated MoEs are supported for Triton Kernels runner"  # Triton Kernels 运行器仅支持门控 MoE

        hidden_states = runner_input.hidden_states  # 获取隐藏状态

        common_kwargs = dict(  # 构造两个融合函数共用的参数字典
            routing_data=runner_input.routing_data,  # 路由数据
            gather_indx=runner_input.gather_indx,  # 聚合索引
            scatter_indx=None if self.config.no_combine else runner_input.scatter_indx,  # 不合并时散射索引为None
            inplace=False,  # 不使用原地操作
            activation=self.config.activation,  # 激活函数类型
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,  # 是否在输入上应用路由权重
            global_num_experts=quant_info.global_num_experts,  # 全局专家数
        )

        has_bias = quant_info.w13_bias is not None or quant_info.w2_bias is not None  # 判断是否存在偏置

        if has_bias:  # 有偏置的情况
            assert (  # 断言两个偏置必须同时存在
                quant_info.w13_bias is not None and quant_info.w2_bias is not None
            ), "Bias execution requires both w13_bias and w2_bias"  # 偏置执行需要 w13_bias 和 w2_bias 同时存在
            output = triton_kernel_fused_experts_with_bias(  # 调用带偏置的融合专家函数
                hidden_states=hidden_states,  # 隐藏状态
                w1=quant_info.w13_weight,  # w1+w3 权重
                w1_pcg=quant_info.w13_precision_config,  # w1+w3 精度配置
                b1=quant_info.w13_bias,  # w1+w3 偏置
                w2=quant_info.w2_weight,  # w2 权重
                w2_pcg=quant_info.w2_precision_config,  # w2 精度配置
                b2=quant_info.w2_bias,  # w2 偏置
                gemm1_alpha=self.config.gemm1_alpha,  # GEMM1 alpha 参数
                gemm1_clamp_limit=self.config.gemm1_clamp_limit,  # GEMM1 钳位限制
                **common_kwargs,  # 展开共用参数
            )
        else:  # 无偏置的情况
            output = triton_kernel_fused_experts(  # 调用无偏置的融合专家函数
                hidden_states=hidden_states,  # 隐藏状态
                w1=quant_info.w13_weight,  # w1+w3 权重
                w2=quant_info.w2_weight,  # w2 权重
                **common_kwargs,  # 展开共用参数
            )

        if self.config.no_combine:  # 如果不需要合并结果
            tokens = runner_input.hidden_states.shape[0]  # 获取令牌数
            hidden = runner_input.hidden_states.shape[-1]  # 获取隐藏维度
            total_rows = output.shape[0]  # 获取输出总行数
            top_k = total_rows // tokens  # 计算 Top-K 值
            output = output.view(tokens, top_k, hidden)  # 重塑输出形状为 (令牌数, Top-K, 隐藏维度)

        return TritonKernelsRunnerOutput(hidden_states=output)  # 将输出包装为 TritonKernelsRunnerOutput 返回

    @property
    def runner_backend(self) -> MoeRunnerBackend:  # 返回运行器后端类型
        return MoeRunnerBackend.TRITON_KERNELS  # 返回 TRITON_KERNELS 后端标识


# ---------------------------------------------------------------------------
# Permute / fused hooks
# 置换 / 融合钩子
# ---------------------------------------------------------------------------


@register_pre_permute("standard", "triton_kernel")  # 注册前置换函数：从 standard 格式到 triton_kernel 后端
def pre_permute_standard_to_triton_kernels(  # 将标准分发输出转换为 Triton Kernels 运行器输入
    dispatch_output: "StandardDispatchOutput",  # 标准分发输出
    quant_info: TritonKernelsQuantInfo,  # Triton Kernels 量化信息
    runner_config: MoeRunnerConfig,  # MoE 运行器配置
    running_state: dict,  # 运行时状态字典
) -> TritonKernelsRunnerInput:  # 返回 Triton Kernels 运行器输入
    from sglang.srt.layers.moe.topk import TopKOutputChecker  # 延迟导入 TopK 输出检查器

    hidden_states = dispatch_output.hidden_states  # 获取隐藏状态
    topk_output = dispatch_output.topk_output  # 获取 TopK 输出

    assert TopKOutputChecker.format_is_triton_kernels(  # 断言 TopK 输出格式为 triton_kernels 格式
        topk_output
    ), "Triton-kernel runner expects TritonKernelTopKOutput"  # Triton-kernel 运行器需要 TritonKernelTopKOutput

    routing_data, gather_indx, scatter_indx = topk_output  # 解包 TopK 输出为路由数据、聚合索引和散射索引

    return TritonKernelsRunnerInput(  # 返回 Triton Kernels 运行器输入
        hidden_states=hidden_states,  # 隐藏状态
        routing_data=routing_data,  # 路由数据
        gather_indx=gather_indx,  # 聚合索引
        scatter_indx=scatter_indx,  # 散射索引
    )


@register_post_permute("triton_kernel", "standard")  # 注册后置换函数：从 triton_kernel 后端到 standard 格式
def post_permute_triton_kernels_to_standard(  # 将 Triton Kernels 运行器输出转换为标准合并输入
    runner_output: TritonKernelsRunnerOutput,  # Triton Kernels 运行器输出
    quant_info: TritonKernelsQuantInfo,  # Triton Kernels 量化信息
    runner_config: MoeRunnerConfig,  # MoE 运行器配置
    running_state: dict,  # 运行时状态字典
) -> StandardCombineInput:  # 返回标准合并输入
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput  # 延迟导入标准合并输入

    hidden_states = runner_output.hidden_states  # 获取运行器输出的隐藏状态

    if (  # 如果配置了路由缩放因子且不为1.0且不跳过合并
        runner_config.routed_scaling_factor is not None  # 路由缩放因子不为None
        and runner_config.routed_scaling_factor != 1.0  # 路由缩放因子不为1.0
        and not runner_config.no_combine  # 不是不合并模式
    ):
        hidden_states.mul_(runner_config.routed_scaling_factor)  # 原地乘以路由缩放因子

    return StandardCombineInput(hidden_states=hidden_states)  # 返回标准合并输入
