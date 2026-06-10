# Copyright 2023-2024 SGLang Team  # 版权所有 2023-2024 SGLang团队
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证2.0版授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache许可证URL
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的保证
# See the License for the specific language governing permissions and  # 请参阅许可证以了解管理权限和
# limitations under the License.  # 限制的特定语言
# ==============================================================================  # 分隔线
# 文件说明：Mega-MoE前向路径和专家权重预处理，供DeepSeek V2/V4共享使用。
# 利用DeepGEMM库实现高效的大规模MoE计算，支持FP8/FP4混合精度和SwiGLU激活。
# 包含对称缓冲区管理、预调度、权重变换和共享专家与路由专家的重叠计算。
"""Mega-MoE forward path and expert-weight prep shared by Deepseek V2/V4."""  # Mega-MoE前向路径和专家权重预处理，供Deepseek V2/V4共享

from __future__ import annotations  # 启用延迟类型注解评估

import os  # 导入操作系统模块
from contextlib import nullcontext  # 导入空上下文管理器
from typing import TYPE_CHECKING, Optional  # 导入类型提示

import torch  # 导入PyTorch

from sglang.jit_kernel.dsv4 import mega_moe_pre_dispatch  # 导入Mega MoE预调度JIT内核
from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo  # 导入专家位置调度信息
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens  # 导入DP注意力全局token数获取
from sglang.srt.layers.moe.utils import get_moe_a2a_backend  # 导入MoE全互连后端获取
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode  # 导入CUDA图捕获模式检测

if TYPE_CHECKING:  # 仅用于类型检查时导入
    from deep_gemm import SymmBuffer  # DeepGEMM对称缓冲区

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 前向批次信息
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE  # DeepSeek V2 MoE模型


_MEGA_MOE_SYMM_BUFFER: dict = {}  # Mega MoE对称缓冲区缓存字典
_MEGA_MOE_DG_ENV_APPLIED = False  # DeepGEMM环境变量是否已应用的标志


def _apply_mega_moe_dg_env() -> None:  # 将SGLang的FP4/MXF4优化标志转发到DeepGEMM环境变量
    """Forward sglang's FP4/MXF4 opt-in flags to DeepGEMM via env vars.  # 将SGLang的FP4/MXF4选择标志通过环境变量转发到DeepGEMM

    DeepGEMM reads `DG_USE_FP4_ACTS` (and `DG_USE_MXF4_KIND`) at host-function  # DeepGEMM在主机函数调用时读取`DG_USE_FP4_ACTS`（和`DG_USE_MXF4_KIND`）
    call time — both `get_symm_buffer_for_mega_moe` and `fp8_fp4_mega_moe`.  # 在`get_symm_buffer_for_mega_moe`和`fp8_fp4_mega_moe`中
    Forwarding once at first use is sufficient (these are static config  # 首次使用时转发一次即可（这些是静态配置
    flags, not per-request state) and matches the `setdefault` pattern so  # 标志，非每请求状态），且匹配`setdefault`模式，因此
    explicit `DG_USE_*` overrides from outside still win.  # 外部显式`DG_USE_*`覆盖仍然优先
    """
    global _MEGA_MOE_DG_ENV_APPLIED  # 声明全局变量
    if _MEGA_MOE_DG_ENV_APPLIED:  # 如果已应用
        return  # 直接返回
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get():  # 如果启用FP4激活
        os.environ.setdefault("DG_USE_FP4_ACTS", "1")  # 设置DeepGEMM FP4激活环境变量
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND.get():  # 如果启用MXF4类型
        os.environ.setdefault("DG_USE_MXF4_KIND", "1")  # 设置DeepGEMM MXF4类型环境变量
    _MEGA_MOE_DG_ENV_APPLIED = True  # 标记已应用


def _get_mega_moe_symm_buffer(  # 获取或创建Mega MoE对称缓冲区
    group,  # 通信组
    num_experts: int,  # 专家数
    num_max_tokens_per_rank: int,  # 每个rank的最大token数
    num_topk: int,  # TopK数
    hidden: int,  # 隐藏维度
    intermediate_hidden: int,  # 中间隐藏维度
) -> SymmBuffer:
    import deep_gemm  # 导入DeepGEMM库

    _apply_mega_moe_dg_env()  # 应用DeepGEMM环境变量

    key = (  # 构建缓存键
        id(group),  # 通信组ID
        num_max_tokens_per_rank,  # 每rank最大token数
        num_experts,  # 专家数
        num_topk,  # TopK数
        hidden,  # 隐藏维度
        intermediate_hidden,  # 中间隐藏维度
    )
    buf = _MEGA_MOE_SYMM_BUFFER.get(key)  # 从缓存中查找
    if buf is None:  # 如果缓存未命中
        buf = deep_gemm.get_symm_buffer_for_mega_moe(  # 创建新的对称缓冲区
            group,  # 通信组
            num_experts,  # 专家数
            num_max_tokens_per_rank,  # 每rank最大token数
            num_topk,  # TopK数
            hidden,  # 隐藏维度
            intermediate_hidden,  # 中间隐藏维度
            use_fp8_dispatch=True,  # 使用FP8调度
            activation="swiglu",  # 激活函数为SwiGLU
        )
        _MEGA_MOE_SYMM_BUFFER[key] = buf  # 存入缓存
    return buf  # 返回缓冲区


def should_use_mega_moe(moe: "DeepseekV2MoE", hidden_states: torch.Tensor) -> bool:  # 判断是否应使用Mega MoE
    if not get_moe_a2a_backend().is_megamoe():  # 如果后端不支持Mega MoE
        return False  # 不使用
    if not getattr(moe.experts, "_mega_moe_weights_built", False):  # 如果权重未构建
        return False  # 不使用
    if get_is_capture_mode():  # 如果处于CUDA图捕获模式
        return True  # 始终使用

    global_num_tokens = get_dp_global_num_tokens()  # 获取DP全局token数
    if global_num_tokens:  # 如果有全局token数
        max_tokens_per_rank = max(global_num_tokens)  # 取最大值
    else:  # 否则
        max_tokens_per_rank = hidden_states.shape[0]  # 使用隐藏状态的batch维度
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()  # 获取token数上限
    return max_tokens_per_rank <= cap  # 如果token数不超过上限则使用


def forward_mega_moe(  # Mega MoE前向传播主入口
    moe: "DeepseekV2MoE",  # MoE模型实例
    hidden_states: torch.Tensor,  # 隐藏状态
    forward_batch: Optional["ForwardBatch"] = None,  # 前向批次信息
    input_ids_global: Optional[torch.Tensor] = None,  # 全局输入token ID
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]  # 获取token数量

    sbo_overlap_flag = (  # 判断是否启用共享-路由专家重叠
        moe.alt_stream is not None  # 有备用CUDA流
        and moe.num_fused_shared_experts == 0  # 没有融合共享专家（共享专家单独运行）
        and num_tokens > 0  # 有token
        and get_is_capture_mode()  # 处于CUDA图捕获模式
    )

    if sbo_overlap_flag:  # 如果启用重叠
        current_stream = torch.cuda.current_stream()  # 获取当前CUDA流
        moe.alt_stream.wait_stream(current_stream)  # 备用流等待当前流
        shared_output = moe._forward_shared_experts(hidden_states)  # 在当前流计算共享专家
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)  # 使用备用流上下文
    else:  # 不启用重叠
        shared_output = moe._forward_shared_experts(hidden_states)  # 计算共享专家
        mega_stream_ctx = nullcontext()  # 使用空上下文

    with mega_stream_ctx:  # 在指定流上下文中执行路由专家
        y = _run_mega_routed(  # 运行路由专家计算
            moe, hidden_states, forward_batch, input_ids_global, num_tokens  # 传入所有参数
        )

    if sbo_overlap_flag:  # 如果启用了重叠
        current_stream.wait_stream(moe.alt_stream)  # 当前流等待备用流完成

    if shared_output is not None:  # 如果有共享专家输出
        y.add_(shared_output)  # 将共享专家输出加到路由专家输出上
    return y  # 返回最终输出


def _run_mega_routed(  # 运行Mega MoE路由专家计算
    moe: "DeepseekV2MoE",  # MoE模型实例
    hidden_states: torch.Tensor,  # 隐藏状态
    forward_batch: Optional["ForwardBatch"],  # 前向批次信息
    input_ids_global: Optional[torch.Tensor],  # 全局输入token ID
    num_tokens: int,  # token数量
) -> torch.Tensor:
    import deep_gemm  # 导入DeepGEMM库

    from sglang.srt.distributed.parallel_state import get_moe_ep_group  # 导入MoE专家并行组

    hidden_size = moe.config.hidden_size  # 获取隐藏维度

    if num_tokens > 0:  # 如果有token
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)  # 计算路由logits
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}  # 哈希路由需要input_ids
        topk_output = moe.topk(  # 执行TopK选择
            hidden_states,  # 隐藏状态
            router_logits,  # 路由logits
            num_token_non_padded=(  # 非填充token数
                forward_batch.num_token_non_padded  # 从前向批次获取
                if forward_batch is not None  # 如果有前向批次
                else None  # 否则为None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(  # 专家位置调度信息
                layer_id=moe.layer_id,  # 层ID
            ),
            **topk_kwargs,  # 传递额外参数
        )
        topk_ids = topk_output.topk_ids  # 获取TopK ID
        topk_weights = topk_output.topk_weights  # 获取TopK权重
    else:  # 没有token
        topk_ids = None  # ID为None
        topk_weights = None  # 权重为None

    ep_group = get_moe_ep_group().device_group  # 获取专家并行设备组
    num_experts = moe.experts.num_experts  # 获取专家数
    top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts  # TopK数加融合共享专家数
    intermediate_size = moe.config.moe_intermediate_size  # 获取MoE中间维度
    num_max_tokens_per_rank = (  # 获取每rank最大token数
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()  # 从环境变量获取
    )
    assert num_tokens <= num_max_tokens_per_rank, (  # 断言token数不超过上限
        f"mega MoE: num_tokens={num_tokens} exceeds cap "  # token数超过上限
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="  # 环境变量值
        f"{num_max_tokens_per_rank}; raise the env var or shrink "  # 请增大环境变量或缩小
        f"cuda_graph_max_bs / chunked_prefill_size accordingly"  # cuda_graph_max_bs / chunked_prefill_size
    )

    buf = _get_mega_moe_symm_buffer(  # 获取对称缓冲区
        ep_group,  # 专家并行组
        num_experts=num_experts,  # 专家数
        num_max_tokens_per_rank=num_max_tokens_per_rank,  # 每rank最大token数
        num_topk=top_k,  # TopK数
        hidden=hidden_size,  # 隐藏维度
        intermediate_hidden=intermediate_size,  # 中间隐藏维度
    )

    if num_tokens > 0:  # 如果有token
        topk_ids_in = topk_ids.to(torch.int32)  # 转为int32
        topk_weights_in = topk_weights.to(torch.float32)  # 转为float32
    else:  # 没有token
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int32)  # 创建空ID张量
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)  # 创建空权重张量

    use_fp4_acts = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get()  # 是否使用FP4激活
    if use_fp4_acts:  # 如果使用FP4
        # FP4 path goes through DeepGEMM's mega_moe_pre_dispatch which  # FP4路径通过DeepGEMM的mega_moe_pre_dispatch
        # handles the E2M1 packing variant. The jit implementation  # 处理E2M1打包变体。JIT实现
        # only emits FP8.  # 仅输出FP8
        deep_gemm.mega_moe_pre_dispatch(  # 调用DeepGEMM预调度
            hidden_states,  # 隐藏状态
            topk_ids_in,  # TopK ID
            topk_weights_in,  # TopK权重
            buf.x,  # 缓冲区输入
            buf.x_sf,  # 缓冲区缩放因子
            buf.topk_idx,  # 缓冲区TopK索引
            buf.topk_weights,  # 缓冲区TopK权重
            num_tokens=num_tokens,  # token数量
            group_size=32,  # 量化组大小
            use_fp4_acts=True,  # 使用FP4激活
        )
    else:  # 使用FP8
        mega_moe_pre_dispatch(  # 调用JIT预调度内核
            hidden_states,  # 隐藏状态
            topk_ids_in,  # TopK ID
            topk_weights_in,  # TopK权重
            buf.x,  # 缓冲区输入
            buf.x_sf,  # 缓冲区缩放因子
            buf.topk_idx,  # 缓冲区TopK索引
            buf.topk_weights,  # 缓冲区TopK权重
            quant_group_size=32,  # 量化组大小
        )

    # Allocate at least one row so y has a non-null CUDA data_ptr;  # 分配至少一行，使y具有非空CUDA数据指针
    # the DeepGEMM tvm-ffi binding rejects nullptr in convert_to_torch_tensor().  # DeepGEMM的tvm-ffi绑定在convert_to_torch_tensor()中拒绝空指针
    y = torch.empty(  # 创建输出张量
        (max(num_tokens, 1), hidden_size),  # 至少1行，hidden_size列
        dtype=torch.bfloat16,  # bfloat16精度
        device=hidden_states.device,  # 相同设备
    )
    swiglu_limit = getattr(moe.config, "swiglu_limit", None)  # 获取SwiGLU截断限制
    deep_gemm.fp8_fp4_mega_moe(  # 调用DeepGEMM Mega MoE内核
        y,  # 输出张量
        moe.experts.mega_l1_weights,  # L1权重
        moe.experts.mega_l2_weights,  # L2权重
        buf,  # 对称缓冲区
        recipe=(1, 1, 32),  # 量化recipe
        activation="swiglu",  # SwiGLU激活
        activation_clamp=swiglu_limit,  # 激活截断值
        fast_math=True,  # 启用快速数学
    )
    y = y[:num_tokens]  # 截取有效token行

    if not moe.experts.should_fuse_routed_scaling_factor_in_topk:  # 如果TopK中未融合路由缩放因子
        y.mul_(moe.routed_scaling_factor)  # 手动乘以路由缩放因子
    return y  # 返回输出


def build_mega_moe_experts_weights(experts) -> None:  # 构建Mega MoE专家权重（预变换）
    from deep_gemm import (  # 导入DeepGEMM权重变换函数
        transform_sf_into_required_layout,  # 缩放因子布局变换
        transform_weights_for_mega_moe,  # Mega MoE权重变换
    )
    from deep_gemm.mega import _interleave_l1_weights, _transpose_sf_for_utccp  # 导入L1权重交错和UTCCP转置

    if getattr(experts, "_mega_moe_weights_built", False):  # 如果权重已构建
        return  # 直接返回

    w13 = experts.w13_weight.data  # 获取gate+up投影权重
    w13_sf_fp32 = experts.w13_weight_scale_inv.data  # 获取gate+up缩放因子
    w2 = experts.w2_weight.data  # 获取down投影权重
    w2_sf_fp32 = experts.w2_weight_scale_inv.data  # 获取down缩放因子

    num_groups, n1, half_k1 = w13.shape  # 解析w13形状：组数、输出维度、半输入维度
    k1 = half_k1 * 2  # 完整输入维度
    _, n2, half_k2 = w2.shape  # 解析w2形状
    k2 = half_k2 * 2  # 完整输入维度

    w13_sf = transform_sf_into_required_layout(  # 变换w13缩放因子布局
        w13_sf_fp32,  # 原始缩放因子
        mn=n1,  # M*N维度
        k=k1,  # K维度
        recipe=(1, 32),  # 量化recipe
        num_groups=num_groups,  # 组数
        disable_ue8m0_cast=False,  # 不禁用UE8M0转换
    )
    w2_sf = transform_sf_into_required_layout(  # 变换w2缩放因子布局
        w2_sf_fp32,  # 原始缩放因子
        mn=n2,  # M*N维度
        k=k2,  # K维度
        recipe=(1, 32),  # 量化recipe
        num_groups=num_groups,  # 组数
        disable_ue8m0_cast=False,  # 不禁用UE8M0转换
    )

    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():  # 如果启用内存修复优化
        # Build the interleaved L1 weight + scale once; share the weight buffer  # 一次性构建交错L1权重和缩放因子；共享权重缓冲区
        # between `w13_weight.data` (normal deep-ep path) and `mega_l1_weights[0]`  # 在`w13_weight.data`（普通deep-ep路径）和`mega_l1_weights[0]`之间
        # (mega moe path). Mega moe additionally needs a UTCCP-transposed scale;  # （mega moe路径）。Mega moe还需要UTCCP转置的缩放因子；
        # the deep-ep path consumes the non-transposed interleaved scale and a  # deep-ep路径使用非转置的交错缩放因子和
        # swizzle-aware activation kernel. L2 weight is untouched by the mega  # 支持swizzle的激活内核。L2权重不受mega变换影响
        # transform, so the existing `w2_weight.data` is shared directly.  # 因此现有的`w2_weight.data`直接共享
        w13_interleaved, w13_sf_interleaved = _interleave_l1_weights((w13, w13_sf))  # 交错L1权重
        w13_sf_utccp = _transpose_sf_for_utccp(w13_sf_interleaved)  # UTCCP转置w13缩放因子
        w2_sf_utccp = _transpose_sf_for_utccp(w2_sf)  # UTCCP转置w2缩放因子

        experts.w13_weight.data = w13_interleaved  # 替换w13权重为交错版本
        experts.w13_weight_scale_inv.data = w13_sf_interleaved  # 替换w13缩放因子为交错版本
        experts.w2_weight.data = w2_sf  # 保持w2权重不变
        experts.w13_weight_scale_inv.format_ue8m0 = True  # 标记w13缩放因子为UE8M0格式
        experts.w2_weight_scale_inv.format_ue8m0 = True  # 标记w2缩放因子为UE8M0格式

        experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)  # 设置mega L1权重（共享+UTCCP缩放）
        experts.mega_l2_weights = (experts.w2_weight.data, w2_sf_utccp)  # 设置mega L2权重（共享+UTCCP缩放）
    else:  # 不使用内存修复优化
        l1_pair, l2_pair = transform_weights_for_mega_moe((w13, w13_sf), (w2, w2_sf))  # 完整变换权重

        experts.mega_l1_weights = l1_pair  # 设置mega L1权重
        experts.mega_l2_weights = l2_pair  # 设置mega L2权重

    experts._mega_moe_weights_built = True  # 标记权重已构建
