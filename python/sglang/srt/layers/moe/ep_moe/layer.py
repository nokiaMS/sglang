# DeepEP MoE 层实现文件
# 实现 Expert Parallel MoE 的核心层，基于 DeepEP 框架
# 支持正常模式和低延迟模式，以及 W4A8 量化
from __future__ import annotations  # 启用延迟注解评估 # 启用 Python 延迟注解特性

import logging  # 导入日志模块 # 导入日志记录模块
from typing import TYPE_CHECKING, Any, Dict, Optional  # 导入类型提示 # 导入类型提示工具

import torch  # 导入 PyTorch # 导入 PyTorch 深度学习框架

from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph  # 导入分段 CUDA 图检测 # 导入分段 CUDA 图上下文检测函数
from sglang.srt.environ import envs  # 导入环境变量 # 导入环境变量配置
from sglang.srt.layers import deep_gemm_wrapper  # 导入 DeepGEMM 包装器 # 导入 DeepGEMM 包装模块
from sglang.srt.layers.moe import (  # 导入 MoE 工具函数 # 导入 MoE 相关工具
    get_deepep_mode,  # 获取 DeepEP 模式 # 获取 DeepEP 模式配置
    get_moe_a2a_backend,  # 获取 MoE All-to-All 后端 # 获取 MoE All-to-All 通信后端
    get_moe_runner_backend,  # 获取 MoE 运行器后端 # 获取 MoE 运行器后端
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (  # 导入融合 MoE 层 # 导入 Triton 融合 MoE 层实现
    FusedMoE,  # 融合 MoE 基类 # 融合 MoE 层基类
    moe_forward_piecewise_cuda_graph_impl,  # 分段 CUDA 图前向实现 # 分段 CUDA 图 MoE 前向实现
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (  # 导入 DeepEP token 调度器 # 导入 DeepEP token 调度器
    DeepEPLLCombineInput,  # DeepEP 低延迟合并输入 # DeepEP 低延迟模式合并输入类
    DeepEPNormalCombineInput,  # DeepEP 正常模式合并输入 # DeepEP 正常模式合并输入类
)
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker  # 导入 TopK 输出及检查器 # 导入 TopK 输出类和检查器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类 # 导入量化配置基类
from sglang.srt.layers.quantization.fp8 import Fp8Config  # 导入 FP8 量化配置 # 导入 FP8 量化配置类
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz  # 导入 FP8 FNUZ 检测 # 导入 FP8 FNUZ 格式检测函数
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod  # 导入 W4A8 FP8 量化 # 导入 W4A8 FP8 量化配置和方法
from sglang.srt.utils import get_bool_env_var, is_hip, is_npu  # 导入工具函数 # 导入环境变量和平台检测函数

if TYPE_CHECKING:  # 类型检查时导入 # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入 token 调度器类型 # 导入 token 调度器类型定义
        DeepEPLLDispatchOutput,  # DeepEP 低延迟调度输出 # DeepEP 低延迟模式调度输出类型
        DeepEPNormalDispatchOutput,  # DeepEP 正常模式调度输出 # DeepEP 正常模式调度输出类型
        DispatchOutput,  # 调度输出基类 # 调度输出基类型
    )

_is_hip = is_hip()  # 检测是否为 HIP 平台 # 判断是否为 AMD HIP 平台
_is_npu = is_npu()  # 检测是否为 NPU 平台 # 判断是否为华为 NPU 平台
_is_fp8_fnuz = is_fp8_fnuz()  # 检测是否为 FP8 FNUZ 格式 # 判断是否使用 FP8 FNUZ 格式
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用 AITER # 判断是否在 HIP 平台启用 AITER


logger = logging.getLogger(__name__)  # 获取当前模块日志记录器 # 创建当前模块的日志记录器


class DeepEPMoE(FusedMoE):  # DeepEP MoE 层类 # 基于 DeepEP 的 Expert Parallel MoE 层实现
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    Mooncake EP shares the same class, as they expose the same interface.
    """
    # 基于 DeepEP 的 MoE Expert Parallel 实现 (https://github.com/deepseek-ai/DeepEP/tree/main)
    # Mooncake EP 共享相同的类，因为它们暴露相同的接口。

    _has_printed = False  # 是否已打印过信息 # 类级别的打印标记

    def __init__(  # 初始化方法 # DeepEPMoE 类的构造函数
        self,
        num_experts: int,  # 专家数量 # 专家总数
        top_k: int,  # top-k 值 # 每个token选择的专家数
        hidden_size: int,  # 隐藏层大小 # 隐藏维度大小
        intermediate_size: int,  # 中间层大小 # 中间维度大小
        layer_id: int,  # 层 ID # 当前层的 ID
        num_fused_shared_experts: int = 0,  # 融合共享专家数 # 融合的共享专家数量
        params_dtype: Optional[torch.dtype] = None,  # 参数数据类型 # 权重数据类型
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
        activation: str = "silu",  # 激活函数 # 激活函数类型
        routed_scaling_factor: Optional[float] = None,  # 路由缩放因子 # 路由缩放因子
        **kwargs,  # 其他关键字参数 # 其他关键字参数
    ):
        super().__init__(  # 调用父类构造函数 # 调用 FusedMoE 的构造函数
            num_experts=num_experts,  # 专家数量 # 专家总数
            top_k=top_k,  # top-k 值 # 每个token选择的专家数
            hidden_size=hidden_size,  # 隐藏层大小 # 隐藏维度大小
            intermediate_size=intermediate_size,  # 中间层大小 # 中间维度大小
            layer_id=layer_id,  # 层 ID # 当前层的 ID
            num_fused_shared_experts=num_fused_shared_experts,  # 融合共享专家数 # 融合的共享专家数量
            params_dtype=params_dtype,  # 参数数据类型 # 权重数据类型
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=prefix,  # 前缀 # 参数名前缀
            activation=activation,  # 激活函数 # 激活函数类型
            routed_scaling_factor=routed_scaling_factor,  # 路由缩放因子 # 路由缩放因子
            **kwargs,  # 其他关键字参数 # 其他关键字参数
        )
        if _use_aiter:  # 使用 AITER # 判断是否使用 AITER
            self.deprecate_flag = True  # 标记弃用 # 设置弃用标记
        elif _is_npu:  # NPU 平台 # 判断是否为 NPU 平台
            self.deprecate_flag = True  # 标记弃用 # 设置弃用标记
        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(  # 启用 DeepGEMM 且为 FP8 量化 # 判断是否启用 DeepGEMM 且使用 FP8 量化
            quant_config, Fp8Config  # FP8 量化配置 # FP8 量化配置类型
        ):
            self.deprecate_flag = True  # 标记弃用 # 设置弃用标记
        elif (  # DeepGEMM + BF16 dispatch # DeepGEMM + BF16 分发模式
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM  # 启用 DeepGEMM # 判断是否启用 DeepGEMM
            and envs.SGLANG_DEEPEP_BF16_DISPATCH.get()  # 启用 BF16 分发 # 判断是否启用 BF16 分发
        ):
            self.deprecate_flag = True  # 标记弃用 # 设置弃用标记
        elif (  # FlashInfer CuteDSL + ModelOpt FP4 # FlashInfer CuteDSL + ModelOpt FP4 量化模式
            get_moe_runner_backend().is_flashinfer_cutedsl()  # 使用 FlashInfer CuteDSL 后端 # 判断是否使用 CuteDSL 后端
            and quant_config is not None  # 有量化配置 # 判断是否有量化配置
            and quant_config.get_name() == "modelopt_fp4"  # 量化类型为 modelopt_fp4 # 判断量化类型
        ):
            self.deprecate_flag = True  # 标记弃用 # 设置弃用标记
        elif (  # 未量化 BF16 DeepEP 低延迟模式 # 未量化的 BF16 DeepEP 低延迟模式
            quant_config is None  # 无量化配置 # 判断是否无量化配置
            and self.w13_weight.dtype == torch.bfloat16  # 权重为 BF16 # 判断权重是否为 BF16
            and get_moe_runner_backend().is_deep_gemm()  # 使用 DeepGEMM 后端 # 判断是否使用 DeepGEMM 后端
            and get_moe_a2a_backend().is_deepep()  # 使用 DeepEP 后端 # 判断是否使用 DeepEP 后端
            and get_deepep_mode().enable_low_latency()  # 启用低延迟模式 # 判断是否启用低延迟模式
            and not _is_npu  # 非 NPU 平台 # 排除 NPU 平台
            and not _is_hip  # 非 HIP 平台 # 排除 HIP 平台
        ):
            assert (  # 断言 # 断言检查
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM  # 必须启用 DeepGEMM # 必须启用 DeepGEMM
            ), "Unquantized DeepEP low-latency MoE requires DeepGEMM BF16"  # 错误信息 # 未量化的 DeepEP 低延迟 MoE 需要 DeepGEMM BF16
            self.deprecate_flag = True  # 标记弃用 # 设置弃用标记
        else:  # 其他情况 # 否则
            self.deprecate_flag = False  # 不标记弃用 # 不设置弃用标记

        if self.deprecate_flag:  # 已标记弃用 # 判断是否标记弃用
            return  # 提前返回 # 提前返回，使用父类实现

        if isinstance(quant_config, Fp8Config):  # FP8 量化配置 # 判断是否为 FP8 量化配置
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)  # 是否使用块量化 # 获取块量化标志
            self.use_fp8_w8a8 = True  # 使用 FP8 W8A8 # 启用 FP8 W8A8 量化
            self.fp8_dtype = torch.float8_e4m3fn  # FP8 数据类型 # 设置 FP8 数据类型
            self.use_w4afp8 = False  # 不使用 W4A8 FP8 # 禁用 W4A8 FP8 量化
        elif isinstance(quant_config, W4AFp8Config):  # W4A8 FP8 量化配置 # 判断是否为 W4A8 FP8 量化配置
            self.use_w4afp8 = True  # 使用 W4A8 FP8 # 启用 W4A8 FP8 量化
            self.use_fp8_w8a8 = False  # 不使用 FP8 W8A8 # 禁用 FP8 W8A8 量化
            self.use_block_quant = False  # 不使用块量化 # 禁用块量化
        else:  # 无量化配置 # 否则
            self.use_w4afp8 = False  # 不使用 W4A8 FP8 # 禁用 W4A8 FP8 量化
            self.use_fp8_w8a8 = False  # 不使用 FP8 W8A8 # 禁用 FP8 W8A8 量化
            self.use_block_quant = False  # 不使用块量化 # 禁用块量化

        self.deepep_mode = get_deepep_mode()  # 获取 DeepEP 模式 # 获取 DeepEP 模式配置
        if (  # 低延迟模式且非 NPU/HIP 且有量化 # 判断低延迟模式条件
            self.deepep_mode.enable_low_latency()  # 启用低延迟模式 # 判断是否启用低延迟模式
            and not _is_npu  # 非 NPU # 排除 NPU 平台
            and not _is_hip  # 非 HIP # 排除 HIP 平台
            and quant_config is not None  # 有量化配置 # 判断是否有量化配置
        ):
            # AMD HIP and NPU support low_latency DeepEP without DeepGEMM.
            # AMD HIP 和 NPU 支持不使用 DeepGEMM 的低延迟 DeepEP。
            assert (  # 断言 # 断言检查
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM  # 必须启用 DeepGEMM # 必须启用 DeepGEMM
            ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"  # 错误信息 # DeepEP 模式需要 deep_gemm

    def forward(  # 前向传播方法 # DeepEPMoE 的前向传播入口
        self,
        hidden_states: torch.Tensor,  # 隐藏状态 # 输入隐藏状态张量
        topk_output: TopKOutput,  # TopK 输出 # TopK 路由输出
    ):
        if is_in_piecewise_cuda_graph():  # 在分段 CUDA 图中 # 判断是否在分段 CUDA 图执行
            assert TopKOutputChecker.format_is_standard(  # 断言 TopK 格式为标准格式 # 检查 TopK 输出格式
                topk_output  # TopK 输出 # TopK 路由输出
            ), "Only standard topk output is supported for piecewise cuda graph"  # 错误信息 # 分段 CUDA 图仅支持标准 TopK 输出
            return moe_forward_piecewise_cuda_graph_impl(  # 使用分段 CUDA 图实现 # 调用分段 CUDA 图 MoE 前向实现
                hidden_states,  # 隐藏状态 # 输入隐藏状态
                topk_output.topk_weights,  # TopK 权重 # TopK 选择的权重
                topk_output.topk_ids,  # TopK ID # TopK 选择的专家 ID
                topk_output.router_logits,  # 路由 logits # 路由器的 logits
                self.layer_id,  # 层 ID # 当前层的 ID
            )
        else:  # 不在分段 CUDA 图中 # 否则
            return self.forward_impl(hidden_states, topk_output)  # 调用实际前向实现 # 调用标准前向实现

    def forward_impl(  # 前向实现方法 # DeepEPMoE 的核心前向实现
        self,
        hidden_states: torch.Tensor,  # 隐藏状态 # 输入隐藏状态张量
        topk_output: TopKOutput,  # TopK 输出 # TopK 路由输出
    ):

        if self.deprecate_flag:  # 已标记弃用 # 判断是否标记弃用
            return super().forward_impl(  # 使用父类实现 # 调用 FusedMoE 的前向实现
                hidden_states,  # 隐藏状态 # 输入隐藏状态
                topk_output,  # TopK 输出 # TopK 路由输出
            )

        dispatch_output = self.dispatcher.dispatch(  # 执行分发 # 调用调度器的分发方法
            hidden_states=hidden_states, topk_output=topk_output  # 传入隐藏状态和 TopK 输出 # 传入参数
        )
        combine_input = self.run_moe_core(dispatch_output)  # 运行 MoE 核心 # 执行 MoE 核心计算
        return self.dispatcher.combine(combine_input=combine_input)  # 执行合并 # 调用调度器的合并方法

    def dispatch(  # 分发方法 # 执行 token 分发
        self,
        hidden_states: torch.Tensor,  # 隐藏状态 # 输入隐藏状态张量
        topk_output: TopKOutput,  # TopK 输出 # TopK 路由输出
    ):
        return self.dispatcher.dispatch(  # 调用调度器分发 # 委托给调度器的分发方法
            hidden_states=hidden_states,  # 隐藏状态 # 输入隐藏状态
            topk_output=topk_output,  # TopK 输出 # TopK 路由输出
        )

    def run_moe_core(  # 运行 MoE 核心计算 # 执行 MoE 的核心专家计算
        self,
        dispatch_output: DispatchOutput,  # 分发输出 # 调度器的分发输出
    ):

        if self.deprecate_flag:  # 已标记弃用 # 判断是否标记弃用
            return super().run_moe_core(dispatch_output)  # 使用父类实现 # 调用 FusedMoE 的核心计算

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker  # 导入分发输出检查器 # 导入分发输出格式检查器

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):  # DeepEP 正常模式 # 判断是否为 DeepEP 正常模式输出
            if self.quant_config is None:  # 无量化配置 # 判断是否无量化配置
                raise NotImplementedError(  # 抛出未实现异常 # 抛出异常
                    "Unquantized DeepEP MoE currently supports low_latency mode only"  # 错误信息 # 未量化的 DeepEP MoE 目前仅支持低延迟模式
                )
            elif self.use_w4afp8:  # 使用 W4A8 FP8 # 判断是否使用 W4A8 FP8 量化
                output = self.forward_cutlass_w4afp8(dispatch_output)  # CUTLASS W4A8 前向 # 调用 CUTLASS W4A8 前向计算
            else:  # 其他量化 # 否则
                assert False, "forward_deepgemm_contiguous is deprecated"  # 断言失败 # deepgemm_contiguous 已弃用
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):  # DeepEP 低延迟模式 # 判断是否为 DeepEP 低延迟模式输出
            if self.use_w4afp8:  # 使用 W4A8 FP8 # 判断是否使用 W4A8 FP8 量化
                output = self.forward_cutlass_w4afp8_masked(dispatch_output)  # CUTLASS W4A8 掩码前向 # 调用掩码模式前向计算
            else:  # 其他量化 # 否则
                assert False, "forward_deepgemm_masked is deprecated"  # 断言失败 # deepgemm_masked 已弃用

        combine_input_wrapper = (  # 合并输入包装器 # 选择合并输入的包装类
            DeepEPNormalCombineInput  # 正常模式合并输入 # DeepEP 正常模式合并输入类
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)  # 正常模式 # 判断是否为正常模式
            else DeepEPLLCombineInput  # 低延迟模式合并输入 # DeepEP 低延迟模式合并输入类
        )

        return combine_input_wrapper(  # 创建合并输入 # 创建合并输入对象
            hidden_states=output,  # 专家输出 # MoE 核心计算结果
            topk_ids=dispatch_output.topk_ids,  # TopK ID # TopK 专家 ID
            topk_weights=dispatch_output.topk_weights,  # TopK 权重 # TopK 选择权重
        )

    def combine(  # 合并方法 # 执行 token 合并
        self,
        hidden_states: torch.Tensor,  # 隐藏状态 # 专家输出隐藏状态
        topk_ids: torch.Tensor,  # TopK ID # TopK 专家 ID
        topk_weights: torch.Tensor,  # TopK 权重 # TopK 选择权重
        overlap_args: Optional[Dict[str, Any]] = None,  # 重叠参数 # 通信计算重叠参数
    ):
        return self.dispatcher.combine(  # 调用调度器合并 # 委托给调度器的合并方法
            hidden_states=hidden_states,  # 隐藏状态 # 专家输出隐藏状态
            topk_ids=topk_ids,  # TopK ID # TopK 专家 ID
            topk_weights=topk_weights,  # TopK 权重 # TopK 选择权重
            overlap_args=overlap_args,  # 重叠参数 # 通信计算重叠参数
        )

    def forward_cutlass_w4afp8(  # CUTLASS W4A8 FP8 正常模式前向 # 使用 CUTLASS W4A8 FP8 量化的正常模式前向计算
        self,
        dispatch_output: DeepEPNormalDispatchOutput,  # DeepEP 正常模式分发输出 # DeepEP 正常模式的分发输出
    ):
        assert self.moe_runner_config.activation == "silu"  # 检查激活函数 # 断言激活函数为 SiLU
        assert isinstance(self.quant_method, W4AFp8MoEMethod)  # 检查量化方法 # 断言量化方法为 W4AFp8MoEMethod
        return self.quant_method.apply_deepep_normal(  # 应用 DeepEP 正常模式 # 调用 W4A8 FP8 的 DeepEP 正常模式计算
            layer=self,  # 当前层 # 当前 MoE 层
            dispatch_output=dispatch_output,  # 分发输出 # 分发输出数据
        )

    def forward_cutlass_w4afp8_masked(  # CUTLASS W4A8 FP8 掩码模式前向 # 使用 CUTLASS W4A8 FP8 量化的低延迟掩码模式前向计算
        self,
        dispatch_output: DeepEPLLDispatchOutput,  # DeepEP 低延迟分发输出 # DeepEP 低延迟模式的分发输出
    ):
        assert self.moe_runner_config.activation == "silu"  # 检查激活函数 # 断言激活函数为 SiLU
        assert isinstance(self.quant_method, W4AFp8MoEMethod)  # 检查量化方法 # 断言量化方法为 W4AFp8MoEMethod
        return self.quant_method.apply_deepep_ll(  # 应用 DeepEP 低延迟模式 # 调用 W4A8 FP8 的 DeepEP 低延迟模式计算
            layer=self,  # 当前层 # 当前 MoE 层
            dispatch_output=dispatch_output,  # 分发输出 # 分发输出数据
        )


def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):  # 获取 MoE 实现类 # 根据配置返回合适的 MoE 实现类
    """根据量化配置和后端类型返回合适的 MoE 实现类"""
    # [TODO] kk, temporary solution
    # [TODO] kk, 临时解决方案
    if (  # 判断 All-to-All 后端类型 # 检查是否使用特定后端
        get_moe_a2a_backend().is_mori()  # Mori 后端 # 判断是否为 Mori 后端
        or get_moe_a2a_backend().is_deepep()  # DeepEP 后端 # 判断是否为 DeepEP 后端
        or get_moe_a2a_backend().is_mooncake()  # Mooncake 后端 # 判断是否为 Mooncake 后端
        or get_moe_a2a_backend().is_nixl()  # NIXL 后端 # 判断是否为 NIXL 后端
    ):
        return DeepEPMoE  # 返回 DeepEP MoE 类 # 返回 DeepEPMoE 实现
    if get_moe_a2a_backend().is_ascend_fuseep():  # Ascend FuseEP 后端 # 判断是否为 Ascend FuseEP 后端
        # ascend_fuseep bypasses dispatch/combine inside FusedMoE.forward
        # (see forward_fuseep in hardware_backend/npu/moe/fuseep.py).
        # ascend_fuseep 绕过 FusedMoE.forward 中的 dispatch/combine
        # （参见 hardware_backend/npu/moe/fuseep.py 中的 forward_fuseep）。
        return FusedMoE  # 返回融合 MoE 基类 # 返回 FusedMoE 实现

    return FusedMoE  # 默认返回融合 MoE 基类 # 默认返回 FusedMoE 实现
