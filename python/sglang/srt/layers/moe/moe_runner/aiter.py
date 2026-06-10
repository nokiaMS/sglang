# AITER MoE Runner模块：实现基于AMD AITER库的MoE计算核心、输入/输出数据结构，
# 以及与standard/deepep/mori等调度后端之间的pre-permute和post-permute转换函数。
from __future__ import annotations  # 启用延迟类型注解求值

from dataclasses import dataclass  # 数据类装饰器
from enum import Enum  # 枚举基类
from typing import TYPE_CHECKING, Any, Optional, Union  # 类型提示工具

import torch  # PyTorch深度学习框架

from sglang.srt.layers.moe.moe_runner.base import (  # 从基模块导入MoE Runner基础类
    MoeQuantInfo,  # MoE量化信息基类
    MoeRunnerConfig,  # MoE Runner配置类
    MoeRunnerCore,  # MoE Runner核心基类
    RunnerInput,  # Runner输入基类
    RunnerOutput,  # Runner输出基类
    register_post_permute,  # 注册后置换装饰器
    register_pre_permute,  # 注册前置换装饰器
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend  # MoE Runner后端枚举
from sglang.srt.utils import get_int_env_var  # 获取整型环境变量

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher.base import CombineInput  # Combine输入基类
    from sglang.srt.layers.moe.token_dispatcher.deepep import (  # DeepEP调度输出类型
        DeepEPLLDispatchOutput,  # DeepEP低延迟调度输出
        DeepEPNormalDispatchOutput,  # DeepEP普通调度输出
    )
    from sglang.srt.layers.moe.token_dispatcher.moriep import (  # MoriEP调度输出类型
        MoriEPLLDispatchOutput,  # MoriEP低延迟调度输出
        MoriEPNormalDispatchOutput,  # MoriEP普通调度输出
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (  # 标准调度类型
        StandardCombineInput,  # 标准Combine输入
        StandardDispatchOutput,  # 标准调度输出
    )


class AiterQuantType(str, Enum):  # AITER量化类型枚举
    NONE = "No"  # 无量化
    PER_TOKEN = "per_Token"  # 逐token量化
    PER_128X128 = "per_128x128"  # 每128x128块量化
    PER_1X32 = "per_1x32"  # 每1x32块量化（W4A4）


@dataclass
class AiterMoeQuantInfo(MoeQuantInfo):  # AITER MoE量化信息数据类
    w13_weight: torch.Tensor  # W1+W3门控上投影权重
    w2_weight: torch.Tensor  # W2下投影权重
    quant_type: AiterQuantType = AiterQuantType.NONE  # 量化类型，默认无量化
    w13_scale: Optional[torch.Tensor] = None  # W1+W3权重缩放因子
    w2_scale: Optional[torch.Tensor] = None  # W2权重缩放因子
    a13_scale: Optional[torch.Tensor] = None  # W1+W3激活缩放因子
    a2_scale: Optional[torch.Tensor] = None  # W2激活缩放因子
    b13: Optional[torch.Tensor] = None  # W1+W3偏置
    b2: Optional[torch.Tensor] = None  # W2偏置
    expert_mask: Optional[torch.Tensor] = None  # 专家掩码（用于屏蔽sink槽位）
    doweight_stage1: bool = False  # 是否在第一阶段进行加权
    hidden_pad: int = 0  # 隐藏状态填充大小
    intermediate_pad: int = 0  # 中间维度填充大小
    swiglu_limit: float = 0.0  # SwiGLU限幅值


@dataclass
class AiterRunnerInput(RunnerInput):  # AITER Runner输入数据类
    hidden_states: torch.Tensor  # 隐藏状态张量
    topk_ids: torch.Tensor  # int32  # Top-K专家索引
    topk_weights: torch.Tensor  # float32  # Top-K路由权重
    # Effective activation quant_type (may differ from quant_info.quant_type  # 有效激活量化类型（可能与quant_info.quant_type不同
    # after the dispatch-aware decision in mori pre_permute).  # 在mori前置换中根据调度决策后可能改变）。
    quant_type: AiterQuantType  # 激活量化类型
    # Per-token activation scale produced by an EP dispatcher (mori). Falls  # 由EP调度器（mori）生成的逐token激活缩放因子。
    # back to quant_info.a13_scale when None.  # 为None时回退到quant_info.a13_scale。
    a1_scale: Optional[torch.Tensor] = None  # 逐token激活缩放因子
    # Mori-only fused_moe kwargs.  # 仅Mori使用的fused_moe关键字参数。
    num_local_tokens: Optional[torch.Tensor] = None  # 每个专家接收的本地token数
    output_dtype: Optional[torch.dtype] = None  # 输出数据类型

    @property
    def runner_backend(self) -> MoeRunnerBackend:  # 返回AITER后端标识
        return MoeRunnerBackend.AITER


@dataclass
class AiterRunnerOutput(RunnerOutput):  # AITER Runner输出数据类
    hidden_states: torch.Tensor  # 输出隐藏状态张量

    @property
    def runner_backend(self) -> MoeRunnerBackend:  # 返回AITER后端标识
        return MoeRunnerBackend.AITER


_AITER_ACTIVATIONS = {"silu": "Silu", "swiglu": "Swiglu"}  # AITER激活函数名称映射表


def _aiter_activation(activation: str):  # 将激活函数名称转换为AITER的ActivationType枚举值
    from aiter import ActivationType  # 从AITER库导入激活类型

    return getattr(ActivationType, _AITER_ACTIVATIONS.get(activation, "Gelu"))  # 获取对应激活类型，默认Gelu


def _aiter_quant_type(quant_type: AiterQuantType):  # 将AiterQuantType转换为AITER的QuantType枚举值
    from aiter import QuantType  # 从AITER库导入量化类型

    return getattr(QuantType, quant_type.value)  # 根据枚举值获取对应QuantType


class AiterRunnerCore(MoeRunnerCore):  # AITER MoE Runner核心实现类
    def run(  # 执行AITER MoE计算
        self,
        runner_input: AiterRunnerInput,  # AITER Runner输入
        quant_info: AiterMoeQuantInfo,  # AITER量化信息
        running_state: dict,  # 运行时状态字典
        hooks: Optional[Any] = None,  # 钩子函数
    ) -> AiterRunnerOutput:  # 返回AITER Runner输出
        assert not self.config.no_combine, "no_combine=True is not supported by AITER"  # AITER不支持no_combine模式

        if runner_input.hidden_states.shape[0] == 0:  # 如果输入token数为0
            return AiterRunnerOutput(hidden_states=runner_input.hidden_states)  # 直接返回空输出

        from aiter.fused_moe import fused_moe  # 从AITER导入融合MoE算子
        from aiter.ops.flydsl.moe_common import GateMode  # 导入门控模式枚举

        from sglang.srt.environ import envs  # 导入环境变量

        a1_scale = (  # 确定输入激活缩放因子
            runner_input.a1_scale  # 优先使用Runner输入中的a1_scale
            if runner_input.a1_scale is not None  # 如果输入中提供了a1_scale
            else quant_info.a13_scale  # 否则回退到量化信息中的a13_scale
        )

        extra: dict = {}  # 额外参数字典
        if runner_input.num_local_tokens is not None:  # 如果提供了本地token数
            extra["num_local_tokens"] = runner_input.num_local_tokens  # 传递给fused_moe
        if runner_input.output_dtype is not None:  # 如果指定了输出数据类型
            extra["dtype"] = runner_input.output_dtype  # 传递给fused_moe
        if quant_info.swiglu_limit > 0:  # 如果设置了SwiGLU限幅值
            # Default (INTERLEAVE) preserves the pre-fix behavior for paths  # 默认(INTERLEAVE)保留了之前路径的行为
            # that prepare weights in the gate/up-interleaved layout. Set  # 这些路径以gate/up交错布局准备权重。设置
            # `SGLANG_USE_AITER_MOE_GU_ITLV=0` to switch to SEPARATED, which  # `SGLANG_USE_AITER_MOE_GU_ITLV=0`切换到SEPARATED，
            # matches the layout produced by `Mxfp4MoEMethod` (gpt-oss  # 这与`Mxfp4MoEMethod`（gpt-oss
            # MXFP4) and the gptoss_fp4 tuned FlyDSL kernels.  # MXFP4）和gptoss_fp4调优的FlyDSL内核生成的布局一致。
            extra["gate_mode"] = (  # 设置门控模式
                GateMode.INTERLEAVE.value  # 交错模式
                if envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()  # 如果启用了交错布局
                else GateMode.SEPARATED.value  # 否则使用分离模式
            )
            extra["swiglu_limit"] = quant_info.swiglu_limit  # 传递SwiGLU限幅值

        output = fused_moe(  # 调用AITER融合MoE算子
            hidden_states=runner_input.hidden_states,  # 输入隐藏状态
            w1=quant_info.w13_weight,  # W1+W3权重
            w2=quant_info.w2_weight,  # W2权重
            topk_weight=runner_input.topk_weights,  # Top-K路由权重
            topk_ids=runner_input.topk_ids,  # Top-K专家索引
            quant_type=_aiter_quant_type(runner_input.quant_type),  # 量化类型
            activation=_aiter_activation(self.config.activation),  # 激活函数
            w1_scale=quant_info.w13_scale,  # W1+W3权重缩放因子
            w2_scale=quant_info.w2_scale,  # W2权重缩放因子
            a1_scale=a1_scale,  # 输入激活缩放因子
            a2_scale=quant_info.a2_scale,  # W2激活缩放因子
            bias1=quant_info.b13,  # W1+W3偏置
            bias2=quant_info.b2,  # W2偏置
            expert_mask=quant_info.expert_mask,  # 专家掩码
            doweight_stage1=quant_info.doweight_stage1,  # 是否在第一阶段加权
            hidden_pad=quant_info.hidden_pad,  # 隐藏状态填充
            intermediate_pad=quant_info.intermediate_pad,  # 中间维度填充
            **extra,  # 额外参数
        )
        return AiterRunnerOutput(hidden_states=output)  # 返回AITER Runner输出

    @property
    def runner_backend(self) -> MoeRunnerBackend:  # 返回AITER后端标识
        return MoeRunnerBackend.AITER


# ---------------------------------------------------------------------------
# Pre-permute: dispatch_output -> AiterRunnerInput  # 前置换：调度输出 -> AITER Runner输入
# ---------------------------------------------------------------------------


@register_pre_permute("standard", "aiter")  # 注册standard到aiter的前置换函数
def pre_permute_standard_to_aiter(  # 将标准调度输出转换为AITER Runner输入
    dispatch_output: StandardDispatchOutput,  # 标准调度输出
    quant_info: AiterMoeQuantInfo,  # AITER量化信息
    runner_config: MoeRunnerConfig,  # Runner配置
    running_state: dict,  # 运行时状态
) -> AiterRunnerInput:  # 返回AITER Runner输入
    hidden_states = dispatch_output.hidden_states  # 获取隐藏状态
    topk_weights, topk_ids, _ = dispatch_output.topk_output  # 解包Top-K输出
    topk_weights = topk_weights.to(torch.float32)  # 转换权重为float32

    if runner_config.apply_router_weight_on_input and not quant_info.doweight_stage1:  # 如果需要在输入端应用路由权重且不支持doweight_stage1
        # Pre-scale at the Python level for kernels that don't honor doweight_stage1.  # 在Python层面预缩放，用于不支持doweight_stage1的内核。
        assert (
            topk_weights.dim() == 2 and topk_weights.shape[-1] == 1
        ), "apply_router_weight_on_input requires topk=1"  # apply_router_weight_on_input要求topk=1
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)  # 将路由权重乘到隐藏状态上
        topk_weights = torch.ones_like(topk_weights)  # 将路由权重设为全1

    return AiterRunnerInput(  # 构造并返回AITER Runner输入
        hidden_states=hidden_states,  # 隐藏状态
        topk_ids=topk_ids.to(torch.int32),  # Top-K专家索引（int32）
        topk_weights=topk_weights,  # Top-K路由权重
        quant_type=quant_info.quant_type,  # 量化类型
    )


def _is_mori_dispatch_output(dispatch_output: Any) -> bool:  # 判断调度输出是否来自MoriEP
    # MoriEP{Normal,LL}DispatchOutput carry the post-mori-permute origin_topk_*  # MoriEP{Normal,LL}DispatchOutput携带mori置换后的origin_topk_*
    # tensors that the standard DeepEP outputs lack.  # 张量，标准DeepEP输出不具备这些。
    return hasattr(dispatch_output, "origin_topk_ids")  # 通过检查origin_topk_ids属性判断


def _resolve_mori_quant_type(  # 解析Mori路径下的AITER激活量化类型
    dispatch_a1_dtype: torch.dtype,  # 调度端激活数据类型
    dispatch_scale: Optional[torch.Tensor],  # 调度端缩放因子
    weight_quant: AiterQuantType,  # 权重量化类型
) -> AiterQuantType:  # 返回解析后的量化类型
    """Pick the activation quant_type for AITER when the dispatch path may have  # 当调度路径可能已预量化隐藏状态时，选择AITER的激活量化类型。
    pre-quantized hidden_states. Mirrors the original MoriEPMoE.run_moe_core  # 镜像原始MoriEPMoE.run_moe_core
    decision tree."""  # 决策树。"""
    is_fp8_quant = weight_quant in (  # 判断权重是否为FP8量化
        AiterQuantType.PER_128X128,
        AiterQuantType.PER_TOKEN,
    )
    is_w4a4 = weight_quant == AiterQuantType.PER_1X32  # 判断是否为W4A4量化
    is_fp4_dispatch = dispatch_a1_dtype == torch.float4_e2m1fn_x2  # 判断调度端是否为FP4数据类型
    has_dispatch_scale = dispatch_scale is not None  # 判断调度端是否提供了缩放因子

    if is_w4a4:  # 如果是W4A4权重
        # W4A4 weights always run as per_1x32; FP8 dispatch is upscaled to BF16  # W4A4权重始终以per_1x32运行；FP8调度在此点前已上采样为BF16，
        # before this point so dispatch_scale won't conflict.  # 所以dispatch_scale不会冲突。
        return AiterQuantType.PER_1X32  # 返回per_1x32量化类型
    if is_fp8_quant:  # 如果是FP8量化权重
        return weight_quant  # 直接使用权重量化类型
    # BF16 weights: lift to the dispatch-side quant type when scales are provided.  # BF16权重：当提供缩放因子时，提升到调度端量化类型。
    if has_dispatch_scale and is_fp4_dispatch:  # 有缩放因子且调度端为FP4
        return AiterQuantType.PER_1X32  # 返回per_1x32
    if has_dispatch_scale and not is_fp4_dispatch:  # 有缩放因子且调度端非FP4
        return AiterQuantType.PER_128X128  # 返回per_128x128
    return AiterQuantType.NONE  # 无缩放因子，不量化


def _pre_permute_deepep_to_aiter(  # 将DeepEP/MoriEP调度输出转换为AITER Runner输入
    dispatch_output: Union[  # 调度输出（支持多种类型）
        DeepEPNormalDispatchOutput,  # DeepEP普通调度输出
        DeepEPLLDispatchOutput,  # DeepEP低延迟调度输出
        MoriEPNormalDispatchOutput,  # MoriEP普通调度输出
        MoriEPLLDispatchOutput,  # MoriEP低延迟调度输出
    ],
    quant_info: AiterMoeQuantInfo,  # AITER量化信息
    runner_config: MoeRunnerConfig,  # Runner配置
    running_state: dict,  # 运行时状态
) -> AiterRunnerInput:  # 返回AITER Runner输入
    is_mori = _is_mori_dispatch_output(dispatch_output)  # 判断是否为Mori调度输出

    hidden_states = dispatch_output.hidden_states  # 获取隐藏状态
    topk_ids = dispatch_output.topk_ids.to(torch.int32)  # 转换Top-K索引为int32
    topk_weights = dispatch_output.topk_weights.to(torch.float32)  # 转换Top-K权重为float32
    a1_scale: Optional[torch.Tensor] = None  # 初始化激活缩放因子
    num_local_tokens: Optional[torch.Tensor] = None  # 初始化本地token数
    output_dtype: Optional[torch.dtype] = None  # 初始化输出数据类型
    quant_type = quant_info.quant_type  # 获取量化类型

    if is_mori:  # 如果是Mori调度输出
        from sglang.srt.layers.moe.rocm_moe_utils import upscale, upscale_mxfp4  # 导入上采样工具

        a1_scale = dispatch_output.hidden_states_scale  # 获取调度端隐藏状态缩放因子
        num_local_tokens = dispatch_output.num_recv_tokens_per_expert  # 获取每个专家接收的token数
        output_dtype = dispatch_output.out_dtype  # 获取输出数据类型

        # Truncate dispatch tensors to the configured cap; mori combine only  # 将调度张量截断到配置的上限；mori combine只
        # reads [0, totalRecvTokenNum), so the truncated result needs no  # 读取[0, totalRecvTokenNum)，因此截断结果不需要
        # padding back.  # 回填。
        mori_max = get_int_env_var("SGLANG_MORI_MOE_MAX_INPUT_TOKENS", 0)  # 获取Mori MoE最大输入token数
        if mori_max > 0:  # 如果设置了最大值
            hidden_states = hidden_states[:mori_max]  # 截断隐藏状态
            if a1_scale is not None:  # 如果缩放因子存在
                a1_scale = a1_scale[:mori_max]  # 截断缩放因子
            topk_ids = topk_ids[:mori_max]  # 截断Top-K索引
            topk_weights = topk_weights[:mori_max]  # 截断Top-K权重

        # Upscale dispatched activations when there is no AITER kernel for the  # 当没有AITER内核支持
        # weight/activation dtype pair.  # 权重/激活数据类型对时，上采样调度激活值。
        weight_quant = quant_info.quant_type  # 获取权重量化类型
        is_fp8_quant = weight_quant in (  # 判断是否为FP8量化
            AiterQuantType.PER_128X128,
            AiterQuantType.PER_TOKEN,
        )
        is_w4a4 = weight_quant == AiterQuantType.PER_1X32  # 判断是否为W4A4
        is_fp4_dispatch = hidden_states.dtype == torch.float4_e2m1fn_x2  # 判断隐藏状态是否为FP4

        if is_w4a4 and a1_scale is not None and not is_fp4_dispatch:  # W4A4权重+FP8调度：需要反量化为BF16
            # W4A4 weights with FP8 dispatch: dequant FP8->BF16 first; the  # W4A4权重搭配FP8调度：先将FP8反量化为BF16；
            # FP4 per_1x32 path needs BF16 input.  # FP4 per_1x32路径需要BF16输入。
            hidden_states = upscale(  # 将FP8上采样为BF16
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None  # 清空缩放因子（已应用）
        elif is_fp8_quant and is_fp4_dispatch and a1_scale is not None:  # FP8权重+FP4调度：无对应内核，需反量化
            # FP8 weights + FP4 dispatch: no kernel for the fp4x2/fp8 pair;  # FP8权重 + FP4调度：无fp4x2/fp8对的内核；
            # dequant FP4->BF16 and let fused_moe re-quantize to FP8.  # 将FP4反量化为BF16，由fused_moe重新量化为FP8。
            hidden_states = upscale_mxfp4(  # 将FP4上采样为BF16
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None  # 清空缩放因子（已应用）

        quant_type = _resolve_mori_quant_type(  # 解析Mori路径下的量化类型
            hidden_states.dtype, a1_scale, weight_quant
        )

        running_state["aiter_combine_topk_ids"] = dispatch_output.origin_topk_ids  # 保存Mori原始Top-K索引
        running_state["aiter_combine_topk_weights"] = (  # 保存Mori原始Top-K权重
            dispatch_output.origin_topk_weights
        )
    else:  # 非Mori（标准DeepEP）路径
        # DeepEP marks invalid topk slots with idx == -1; AITER cannot accept  # DeepEP用idx==-1标记无效的topk槽位；AITER不能接受
        # negative ids, so reroute them to the sink slot at index  # 负数id，因此将它们重路由到索引为
        # num_local_experts (masked off by quant_info.expert_mask which has  # num_local_experts的sink槽位（由具有
        # shape (num_local_experts + 1,)).  # shape(num_local_experts+1)的quant_info.expert_mask屏蔽）。
        topk_ids = torch.where(  # 将-1替换为sink专家索引
            topk_ids == -1,  # 无效槽位
            torch.full_like(topk_ids, runner_config.num_local_experts),  # 替换为sink专家
            topk_ids,  # 有效槽位保持不变
        )
        running_state["aiter_combine_topk_ids"] = dispatch_output.topk_ids  # 保存DeepEP Top-K索引
        running_state["aiter_combine_topk_weights"] = dispatch_output.topk_weights  # 保存DeepEP Top-K权重

    running_state["aiter_combine_is_mori"] = is_mori  # 记录是否为Mori路径

    return AiterRunnerInput(  # 构造并返回AITER Runner输入
        hidden_states=hidden_states,  # 隐藏状态
        topk_ids=topk_ids,  # Top-K专家索引
        topk_weights=topk_weights,  # Top-K路由权重
        quant_type=quant_type,  # 量化类型
        a1_scale=a1_scale,  # 激活缩放因子
        num_local_tokens=num_local_tokens,  # 每个专家的本地token数
        output_dtype=output_dtype,  # 输出数据类型
    )


register_pre_permute("deepep_normal", "aiter")(_pre_permute_deepep_to_aiter)  # 注册deepep_normal到aiter的前置换
register_pre_permute("deepep_ll", "aiter")(_pre_permute_deepep_to_aiter)  # 注册deepep_ll到aiter的前置换


# ---------------------------------------------------------------------------
# Post-permute: AiterRunnerOutput -> CombineInput  # 后置换：AITER Runner输出 -> Combine输入
# ---------------------------------------------------------------------------


@register_post_permute("aiter", "standard")  # 注册aiter到standard的后置换函数
def post_permute_aiter_to_standard(  # 将AITER Runner输出转换为标准Combine输入
    runner_output: AiterRunnerOutput,  # AITER Runner输出
    quant_info: AiterMoeQuantInfo,  # AITER量化信息
    runner_config: MoeRunnerConfig,  # Runner配置
    running_state: dict,  # 运行时状态
) -> StandardCombineInput:  # 返回标准Combine输入
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput  # 导入标准Combine输入类

    return StandardCombineInput(hidden_states=runner_output.hidden_states)  # 直接包装隐藏状态返回


def _post_permute_aiter_to_deepep(  # 将AITER Runner输出转换为DeepEP/MoriEP Combine输入
    runner_output: AiterRunnerOutput,  # AITER Runner输出
    quant_info: AiterMoeQuantInfo,  # AITER量化信息
    runner_config: MoeRunnerConfig,  # Runner配置
    running_state: dict,  # 运行时状态
    is_normal: bool,  # 是否为普通模式（否则为低延迟模式）
) -> CombineInput:  # 返回Combine输入
    if running_state.get("aiter_combine_is_mori"):  # 如果是Mori路径
        from sglang.srt.layers.moe.token_dispatcher.moriep import (  # 导入MoriEP Combine输入类型
            MoriEPLLCombineInput,  # MoriEP低延迟Combine输入
            MoriEPNormalCombineInput,  # MoriEP普通Combine输入
        )

        cls = MoriEPNormalCombineInput if is_normal else MoriEPLLCombineInput  # 根据模式选择类
    else:  # 非Mori路径
        from sglang.srt.layers.moe.token_dispatcher.deepep import (  # 导入DeepEP Combine输入类型
            DeepEPLLCombineInput,  # DeepEP低延迟Combine输入
            DeepEPNormalCombineInput,  # DeepEP普通Combine输入
        )

        cls = DeepEPNormalCombineInput if is_normal else DeepEPLLCombineInput  # 根据模式选择类

    return cls(  # 构造并返回Combine输入
        hidden_states=runner_output.hidden_states,  # 隐藏状态
        topk_ids=running_state["aiter_combine_topk_ids"],  # Top-K专家索引
        topk_weights=running_state["aiter_combine_topk_weights"],  # Top-K路由权重
    )


@register_post_permute("aiter", "deepep_normal")  # 注册aiter到deepep_normal的后置换
def post_permute_aiter_to_deepep_normal(  # 将AITER Runner输出转换为DeepEP普通模式Combine输入
    runner_output: AiterRunnerOutput,  # AITER Runner输出
    quant_info: AiterMoeQuantInfo,  # AITER量化信息
    runner_config: MoeRunnerConfig,  # Runner配置
    running_state: dict,  # 运行时状态
) -> CombineInput:  # 返回Combine输入
    return _post_permute_aiter_to_deepep(  # 调用通用后置换函数，指定普通模式
        runner_output, quant_info, runner_config, running_state, is_normal=True
    )


@register_post_permute("aiter", "deepep_ll")  # 注册aiter到deepep_ll的后置换
def post_permute_aiter_to_deepep_ll(  # 将AITER Runner输出转换为DeepEP低延迟模式Combine输入
    runner_output: AiterRunnerOutput,  # AITER Runner输出
    quant_info: AiterMoeQuantInfo,  # AITER量化信息
    runner_config: MoeRunnerConfig,  # Runner配置
    running_state: dict,  # 运行时状态
) -> CombineInput:  # 返回Combine输入
    return _post_permute_aiter_to_deepep(  # 调用通用后置换函数，指定低延迟模式
        runner_output, quant_info, runner_config, running_state, is_normal=False
    )