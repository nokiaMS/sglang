# 归一化层的融合算子实现
# 本文件实现了多种归一化层的融合算子，包括：
# - RMSNorm：根均值平方归一化
# - LayerNorm：层归一化
# - GemmaRMSNorm：Gemma风格的RMSNorm（权重+1）
# - Gemma3RMSNorm：Gemma3风格的RMSNorm
# - Gemma4RMSNorm：Gemma4风格的RMSNorm（支持scale_shift和with_scale）
# - RMSNormWithoutScale：无缩放权重的RMSNorm
# 每个类均支持多平台前向传播（CUDA、HIP、CPU、NPU、XPU等），
# 并支持allreduce融合以优化分布式训练/推理性能。

# Copyright 2023-2024 SGLang Team  # 版权所有 2023-2024 SGLang团队
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证2.0版授权
# you may not use this file except in compliance with the License.  # 您不得在未遵守许可证的情况下使用此文件
# You may obtain a copy of the License at  # 您可以在以下地址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意，否则根据许可证分发的软件
# distributed under the License is distributed on an "AS IS" BASIS,  # 是按"原样"分发的，不附带任何明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不附带任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and  # 请参阅许可证以了解管理权限和限制的特定语言
# limitations under the License.  # 限制
# ==============================================================================
"""Fused operators for normalization layers."""  # 归一化层的融合算子

import logging  # 导入日志模块
from typing import Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数式接口

from sglang.srt.batch_invariant_ops import (  # 导入批量不变操作
    is_batch_invariant_mode_enabled,  # 批量不变模式是否启用
    rms_norm_batch_invariant,  # 批量不变RMS归一化
)
from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.layers.utils import MultiPlatformOp  # 导入多平台操作基类
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import (  # 导入工具函数
    cpu_has_amx_support,  # CPU是否支持AMX指令集
    get_bool_env_var,  # 获取布尔环境变量
    is_cpu,  # 是否为CPU平台
    is_cuda,  # 是否为CUDA平台
    is_flashinfer_available,  # FlashInfer是否可用
    is_hip,  # 是否为HIP(AMD ROCm)平台
    is_musa,  # 是否为摩尔线程MUSA平台
    is_npu,  # 是否为华为NPU平台
    is_xpu,  # 是否为Intel XPU平台
)

_is_cuda = is_cuda()  # 检测当前是否为CUDA平台
_is_flashinfer_available = is_flashinfer_available()  # 检测FlashInfer是否可用
_is_hip = is_hip()  # 检测当前是否为HIP平台
_is_musa = is_musa()  # 检测当前是否为MUSA平台
_is_npu = is_npu()  # 检测当前是否为NPU平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITER（仅HIP平台）
_is_cpu_amx_available = cpu_has_amx_support()  # 检测CPU是否支持AMX
_is_cpu = is_cpu()  # 检测当前是否为CPU平台
_is_xpu = is_xpu()  # 检测当前是否为XPU平台
_flashinfer_layernorm_available = False  # FlashInfer LayerNorm是否可用的标志

if _is_cuda or _is_xpu or _is_musa:  # 如果是CUDA、XPU或MUSA平台
    if _is_flashinfer_available:  # 如果FlashInfer可用
        try:  # 尝试导入FlashInfer归一化模块
            import flashinfer.norm  # 导入FlashInfer归一化

            from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册

            def _layernorm_fake_impl(  # LayerNorm假实现函数（用于torch.compile追踪）
                input: torch.Tensor,  # 输入张量
                gamma: torch.Tensor,  # 缩放权重
                beta: torch.Tensor,  # 偏置
                eps: float = 1e-6,  # epsilon值
            ) -> torch.Tensor:  # 返回与输入相同形状的空张量
                return torch.empty_like(input)  # 返回空张量

            @register_custom_op(fake_impl=_layernorm_fake_impl)  # 注册为自定义算子
            def layernorm(  # LayerNorm函数（注册为自定义算子）
                input: torch.Tensor,  # 输入张量
                gamma: torch.Tensor,  # 缩放权重
                beta: torch.Tensor,  # 偏置
                eps: float = 1e-6,  # epsilon值
            ) -> torch.Tensor:  # 返回归一化后的张量
                return flashinfer.norm.layernorm(input, gamma, beta, eps)  # 调用FlashInfer的LayerNorm

            _flashinfer_layernorm_available = True  # 标记FlashInfer LayerNorm可用
        except (ImportError, AttributeError):  # 捕获导入错误
            _flashinfer_layernorm_available = False  # 标记FlashInfer LayerNorm不可用
    else:  # 否则FlashInfer不可用
        _flashinfer_layernorm_available = False  # 标记不可用

    from sgl_kernel import (  # 从sgl_kernel导入融合归一化算子
        fused_add_rmsnorm,  # 融合加法+RMS归一化
        gemma_fused_add_rmsnorm,  # Gemma融合加法+RMS归一化
        gemma_rmsnorm,  # Gemma RMS归一化
        rmsnorm,  # RMS归一化
    )
_has_aiter_layer_norm = False  # AITER LayerNorm是否可用的标志
_has_vllm_rms_norm = False  # vLLM RMS归一化是否可用的标志
if _use_aiter:  # 如果使用AITER
    from aiter import layernorm2d_fwd as layer_norm  # 导入AITER LayerNorm
    from aiter import rmsnorm2d_fwd as rms_norm  # 导入AITER RMS归一化
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm  # 导入AITER融合加法+RMS归一化

    _has_aiter_layer_norm = True  # aiter provides the layer_norm functions  # AITER提供LayerNorm函数
    _has_vllm_rms_norm = True  # aiter provides the rms_norm functions  # AITER提供RMS归一化函数
elif _is_hip:  # 否则如果是HIP平台
    try:  # 尝试从vLLM导入
        from vllm._custom_ops import fused_add_rms_norm, rms_norm  # 导入vLLM融合归一化算子

        _has_vllm_rms_norm = True  # 标记vLLM RMS归一化可用
    except ImportError:  # 捕获导入错误
        # Fallback: vllm not available, will use forward_native  # 回退：vLLM不可用，将使用forward_native
        _has_vllm_rms_norm = False  # 标记不可用

if _is_cuda:  # 如果是CUDA平台
    # HF-semantics RMSNorm kernel (JIT-compiled).  Used when `cast_x_before_out_mul=True`  # HF语义RMSNorm内核（JIT编译）。当cast_x_before_out_mul=True时使用
    # (the transformers backend path) to produce outputs that are numerically identical  # （transformers后端路径），以产生数值上完全一致的输出
    # to HuggingFace `LlamaRMSNorm`: the cast from fp32 to the activation dtype happens  # 与HuggingFace LlamaRMSNorm一致：从fp32到激活数据类型的转换发生在权重乘法之前
    # BEFORE the weight multiply, so the multiply is done in the narrow dtype.  # 因此乘法在较窄的数据类型中完成
    _jit_rmsnorm_hf_available = False  # JIT HF RMSNorm是否可用的标志
    try:  # 尝试导入JIT HF RMSNorm
        from sglang.jit_kernel.rmsnorm_hf import (  # 从JIT内核模块导入
            is_supported_rmsnorm_hf_hidden_size,  # 检测是否支持给定隐藏大小的HF RMSNorm
        )  # 导入JIT HF RMSNorm内核
        from sglang.jit_kernel.rmsnorm_hf import rmsnorm_hf as _jit_rmsnorm_hf

        _jit_rmsnorm_hf_available = True  # 捕获导入错误
    except ImportError:

        def is_supported_rmsnorm_hf_hidden_size(d: int) -> bool:  # 返回False
            return False

        _jit_rmsnorm_hf = None

    from sglang.jit_kernel.norm import fused_add_rmsnorm as _jit_fused_add_rmsnorm  # 从JIT内核模块导入
    from sglang.jit_kernel.norm import (  # 检测是否支持给定隐藏大小的JIT融合加法+RMSNorm
        is_supported_jit_fused_add_rmsnorm_hidden_size,
    )


logger = logging.getLogger(__name__)  # 创建模块级日志记录器

if _is_npu:  # 如果是NPU平台
    import torch_npu  # 导入华为NPU扩展
    from sgl_kernel_npu.norm.add_rmsnorm_bias import add_gemma_rms_norm  # 导入NPU Gemma RMS归一化


def _forward_with_allreduce_fusion(  # Allreduce融合RMS归一化共享逻辑函数
    norm_module,  # 归一化模块
    x: torch.Tensor,  # 输入张量
    residual: Optional[torch.Tensor],  # 残差张量（可选）
    post_residual_addition: Optional[torch.Tensor],  # 残差后加法张量（可选）
    weight: torch.Tensor,  # 归一化权重
    use_attn_tp_group: bool = True,  # 是否使用注意力TP组（默认True）
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 返回归一化结果或(结果, 残差)元组
    """Shared allreduce-fused RMSNorm logic usable by any norm."""  # 任何归一化模块可共享的allreduce融合RMSNorm逻辑
    if residual is not None:  # 如果有残差
        from sglang.srt.distributed import (  # 导入分布式通信函数
            get_attn_tensor_model_parallel_world_size,  # 获取注意力TP世界大小
            get_moe_expert_parallel_world_size,  # 获取MoE专家并行世界大小
            get_moe_tensor_parallel_world_size,  # 获取MoE张量并行世界大小
            tensor_model_parallel_all_reduce,  # 张量模型并行allreduce
            tensor_model_parallel_fused_allreduce_rmsnorm,  # 融合allreduce+RMS归一化
        )
        from sglang.srt.layers.flashinfer_comm_fusion import (  # 导入FlashInfer通信融合
            flashinfer_allreduce_residual_rmsnorm,  # FlashInfer allreduce+残差+RMS归一化
        )

        if use_attn_tp_group:  # 如果使用注意力TP组
            world_size = get_attn_tensor_model_parallel_world_size()  # 获取注意力TP世界大小
        else:  # 否则
            if get_moe_expert_parallel_world_size() > 1:  # 如果MoE专家并行大于1
                world_size = get_moe_expert_parallel_world_size()  # 使用专家并行世界大小
            else:  # 否则
                world_size = get_moe_tensor_parallel_world_size()  # 使用MoE张量并行世界大小

        if world_size > 1:  # 如果世界大小大于1（需要通信）
            if post_residual_addition is not None:  # 如果有残差后加法
                residual = residual + post_residual_addition  # 将后加法加到残差上

            # Prefer AITER fused AR+RMSNorm when enabled on AMD.  # 在AMD平台上启用时优先使用AITER融合AR+RMSNorm
            if _use_aiter:  # 如果使用AITER
                fused_result = tensor_model_parallel_fused_allreduce_rmsnorm(  # 尝试AITER融合allreduce+RMSNorm
                    x, residual, weight, norm_module.variance_epsilon  # 传入输入、残差、权重、epsilon
                )
                if fused_result is not None:  # 如果融合路径成功
                    return fused_result  # 返回融合结果
            else:  # 否则使用FlashInfer融合路径
                fused_result = flashinfer_allreduce_residual_rmsnorm(  # 尝试FlashInfer融合
                    input_tensor=x,  # 输入张量
                    residual=residual,  # 残差
                    weight=weight,  # 权重
                    eps=norm_module.variance_epsilon,  # epsilon值
                    max_token_num=max(x.shape[0], 2048),  # 最大token数
                    use_attn_tp_group=use_attn_tp_group,  # 是否使用注意力TP组
                )
                if fused_result[0] is not None:  # 如果融合路径成功
                    return fused_result  # 返回融合结果

            # For AITER route, preserve correctness when fused path is unavailable.  # 对于AITER路径，当融合路径不可用时保持正确性
            if _use_aiter and get_global_server_args().enable_aiter_allreduce_fusion:  # 如果AITER融合allreduce启用
                x = tensor_model_parallel_all_reduce(x)  # 执行标准allreduce
                return norm_module.forward(x, residual, None)  # 然后执行归一化前向

    return norm_module.forward(x, residual, post_residual_addition)  # 默认：执行标准归一化前向


class RMSNorm(MultiPlatformOp):  # RMS归一化模块，继承自多平台操作基类
    def __init__(  # 初始化函数
        self,  # 隐藏层大小
        hidden_size: int,  # epsilon值，默认1e-6
        eps: float = 1e-6,  # 方差计算用的隐藏大小（可选）
        var_hidden_size: Optional[int] = None,  # 是否在乘权重前转换数据类型
        cast_x_before_out_mul: bool = False,  # 是否使用FP32残差
        fp32_residual: bool = False,  # 是否有权重（默认True）
        has_weight: bool = True,  # 权重数据类型（可选）
        weight_dtype: Optional = None,  # 覆盖原始数据类型（可选）
        override_orig_dtype: Optional = None,
    ) -> None:  # 调用父类初始化
        super().__init__()  # 保存是否有权重标志
        self.has_weight = has_weight  # 保存类型转换标志
        self.cast_x_before_out_mul = cast_x_before_out_mul  # 保存FP32残差标志
        self.fp32_residual = fp32_residual  # 保存覆盖数据类型
        self.override_orig_dtype = override_orig_dtype  # 如果有权重
        if self.has_weight:  # 创建可训练权重参数
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))  # 否则无权重
        else:  # 创建固定为1的权重（不参与训练）
            self.weight = torch.ones(hidden_size, dtype=weight_dtype)  # 保存epsilon值
        self.variance_epsilon = eps  # 保存隐藏大小
        self.hidden_size = hidden_size  # 计算方差大小覆盖值
        self.variance_size_override = (  # 如果与隐藏大小相同则为None
            None if var_hidden_size == hidden_size else var_hidden_size  # 如果使用AITER
        )  # 设置前向方法为AITER版本
        if _use_aiter:
            self._forward_method = self.forward_aiter

    def forward_cuda(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 如果输入为空
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 如果有残差
        if x.numel() == 0:  # 如果有残差后加法
            if residual is not None:  # 加到残差上
                if post_residual_addition is not None:  # 返回空输入和残差
                    residual = residual + post_residual_addition  # 返回空输入
                return x, residual  # sgl_kernel rmsnorm需要2D输入；重塑高维张量
            return x  # 是否需要重塑形状
        # sgl_kernel rmsnorm requires 2D input; reshape higher-rank tensors  # 如果需要重塑
        needs_reshape = x.dim() != 2 and residual is None  # 保存原始形状
        if needs_reshape:  # 重塑为2D
            original_shape = x.shape  # 如果有方差大小覆盖
            x = x.contiguous().reshape(-1, original_shape[-1])  # 使用原生实现
        if self.variance_size_override is not None:  # 如果启用了批量不变模式
            return self.forward_native(x, residual, post_residual_addition)  # 有残差/需类型转换/FSDP目标时使用原生实现
        if is_batch_invariant_mode_enabled():
            if (  # 使用原生实现
                residual is not None  # 使用批量不变RMS归一化
                or self.cast_x_before_out_mul  # 输入
                or get_global_server_args().rl_on_policy_target == "fsdp"  # 权重
            ):  # epsilon值
                return self.forward_native(x, residual, post_residual_addition)
            return rms_norm_batch_invariant(  # 如果需要在乘权重前转换且无残差
                x,  # 使用HF语义内核（在权重乘法前转换到目标数据类型）
                self.weight.data,  # 如果满足JIT HF RMSNorm使用条件
                self.variance_epsilon,  # JIT HF RMSNorm可用
            )  # 输入为FP16或BF16
        if self.cast_x_before_out_mul and residual is None:  # 权重与输入类型相同
            # Use HF-semantics kernel (cast to dtype before weight multiply).  # 隐藏大小受支持
            if (
                _jit_rmsnorm_hf_available  # 使用JIT HF RMSNorm内核
                and x.dtype in (torch.float16, torch.bfloat16)  # 传入连续输入、权重、epsilon
                and self.weight.data.dtype == x.dtype
                and is_supported_rmsnorm_hf_hidden_size(x.shape[-1])  # 回退：纯Python HF语义（已在forward_native中实现）
            ):  # 使用原生实现
                out = _jit_rmsnorm_hf(  # 如果需要重塑回原形状
                    x.contiguous(), self.weight.data, self.variance_epsilon  # 重塑回原始形状
                )  # 返回输出
            else:  # 如果有残差
                # Fallback: pure-Python HF semantics (already implemented in forward_native).  # 如果需要在乘权重前转换类型
                out = self.forward_native(x, None, None)  # 如果满足JIT融合加法+RMSNorm条件
            if needs_reshape:  # 输入为FP16或BF16
                out = out.reshape(original_shape)  # 权重与输入类型相同
            return out  # 无残差后加法或后加法类型与输入相同
        if residual is not None:  # 隐藏大小受JIT支持
            if self.cast_x_before_out_mul:
                if (  # 如果有残差后加法
                    x.dtype in (torch.float16, torch.bfloat16)  # 加到残差上
                    and self.weight.data.dtype == x.dtype  # 使用JIT融合加法+RMSNorm内核
                    and (  # 输入
                        post_residual_addition is None  # 残差
                        or post_residual_addition.dtype == x.dtype  # 权重
                    )  # epsilon值
                    and is_supported_jit_fused_add_rmsnorm_hidden_size(x.shape[-1])  # 传入类型转换标志
                ):  # 返回输入（原地修改）和残差
                    if post_residual_addition is not None:  # 否则使用原生实现
                        residual = residual + post_residual_addition  # TODO: 理想情况(hidden_states+residual)+post_residual_addition
                    _jit_fused_add_rmsnorm(  # 但目前只能hidden_states+(residual+post_residual_addition)
                        x,  # 两者不等，可能需要给fused_add_rmsnorm添加参数
                        residual,  # 如果有残差后加法
                        self.weight.data,  # 加到残差上
                        self.variance_epsilon,  # 调用融合加法+RMSNorm
                        cast_x_before_out_mul=self.cast_x_before_out_mul,  # 返回输入（原地修改）和残差
                    )  # 无残差时调用RMSNorm
                    return x, residual  # 如果需要重塑回原形状
                return self.forward_native(x, residual, post_residual_addition)  # 重塑回原始形状
            # TODO: Ideally we want to have (hidden_states+residual)+post_residual_addition.  # 返回输出
            # but right now we can only have hidden_states+(residual+post_residual_addition).
            # (hidden_states+residual)+post_residual_addition != hidden_states+(residual+post_residual_addition),
            # we probably need to add another parameter to fused_add_rmsnorm
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        if needs_reshape:
            out = out.reshape(original_shape)
        return out

    def forward_npu(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 如果有残差
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 如果有残差后加法
        if residual is not None:  # 加到残差上
            if post_residual_addition is not None:  # 调用NPU融合加法+RMSNorm
                residual = residual + post_residual_addition  # 传入残差、输入、权重、epsilon
            out, _, residual_out = torch_npu.npu_add_rms_norm(  # 返回归一化结果和残差输出
                residual, x, self.weight.data, self.variance_epsilon  # 无残差时调用NPU RMSNorm
            )
            return out, residual_out
        return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]

    def forward_aiter(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 修复dsv4 dp attention问题
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 症状是torch.AcceleratorError: HIP错误：无效配置参数
        # Fix dsv4 dp attenton issue  # 如果输入行数为0
        # the symptom is torch.AcceleratorError: HIP error: invalid configuration argument  # 如果有残差
        if x.shape[0] == 0:  # 返回空输入和残差
            if residual is not None:  # 返回空输入
                return x, residual  # AITER的RMSNorm内核需要2D连续输入
            return x  # 已安全的布局作为零拷贝路径，仅对跨步或高维视图进行归一化
        # Aiter's RMSNorm kernels expect 2D contiguous inputs. Keep the  # 是否需要重塑形状
        # already-safe layout as a zero-copy path, and only normalize strided or  # 如果需要重塑
        # higher-rank views such as Q/K slices from packed QKV projections.  # 保存原始形状
        needs_reshape = x.dim() != 2 and residual is None  # 重塑为2D
        if needs_reshape:  # 否则如果不连续
            original_shape = x.shape  # 转为连续
            x = x.contiguous().reshape(-1, original_shape[-1])  # 如果有残差
        elif not x.is_contiguous():  # 分配残差输出张量
            x = x.contiguous()  # 分配输出张量
        if residual is not None:  # 如果有残差后加法
            residual_out = torch.empty_like(x)  # 加到残差上
            output = torch.empty_like(x)  # 调用AITER融合加法+RMSNorm
            if post_residual_addition is not None:  # 输出张量
                residual = residual + post_residual_addition  # 输入张量
            fused_add_rms_norm(  # 残差
                output,  # 残差输出
                x,  # 权重
                residual,  # epsilon值
                residual_out,  # 返回输出和残差输出
                self.weight.data,  # 无残差时调用AITER RMSNorm
                self.variance_epsilon,  # 如果需要重塑回原形状
            )  # 重塑回原始形状
            return output, residual_out  # 返回输出
        output = rms_norm(x, self.weight.data, self.variance_epsilon)
        if needs_reshape:
            output = output.reshape(original_shape)
        return output

    def forward_hip(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 如果vLLM不可用，回退到原生实现
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 如果vLLM RMS归一化不可用
        # Fallback to native implementation if vllm is not available  # 使用原生实现
        if not _has_vllm_rms_norm:
            return self.forward_native(x, residual, post_residual_addition)  # 如果输入不连续

        if not x.is_contiguous():  # 转为连续
            # NOTE: Remove this if aiter kernel supports discontinuous input  # 如果有残差
            x = x.contiguous()  # 分配输出张量
        if residual is not None:  # 分配残差输出张量
            out = torch.empty_like(x)  # 如果有残差后加法
            residual_out = torch.empty_like(x)  # 加到残差上
            if post_residual_addition is not None:  # 调用vLLM融合加法+RMSNorm
                residual = residual + post_residual_addition  # 传入参数
            fused_add_rms_norm(  # 返回输出和残差输出
                out, x, residual_out, residual, self.weight.data, self.variance_epsilon  # 分配输出张量
            )  # 调用vLLM RMSNorm
            return out, residual_out  # 返回输出
        out = torch.empty_like(x)
        rms_norm(out, x, self.weight.data, self.variance_epsilon)
        return out

    def forward_musa(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 如果输入不连续
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 转为连续
        if not x.is_contiguous():
            x = x.contiguous()  # 如果有残差

        if residual is not None:  # 加到残差上
            if post_residual_addition is not None:  # 调用融合加法+RMSNorm
                residual = residual + post_residual_addition  # 返回输入（原地修改）和残差
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual  # 无残差时使用PyTorch内置RMSNorm

        out = nn.functional.rms_norm(  # 返回输出
            x, (self.hidden_size,), self.weight.data, self.variance_epsilon
        )
        return out

    def forward_native(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 如果输入不连续
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 转为连续
        if not x.is_contiguous():  # 获取原始数据类型
            x = x.contiguous()  # 转换为FP32进行计算
        orig_dtype = self.override_orig_dtype or x.dtype  # 如果有残差
        x = x.to(torch.float32)  # 加上残差
        if residual is not None:  # 如果有残差后加法
            x = x + residual.to(torch.float32)  # 加上后加法
            if post_residual_addition is not None:  # 如果需要FP32残差
                x = x + post_residual_addition.to(torch.float32)  # 克隆FP32残差
            if self.fp32_residual:  # 残差转回原始类型
                residual = x.clone()
            else:  # 获取隐藏大小
                residual = x.to(orig_dtype)  # 如果隐藏大小不匹配

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(  # 如果无方差大小覆盖
                "Expected hidden_size to be "  # 使用完整x计算方差
                f"{self.hidden_size}, but found: {hidden_size}"  # 如果隐藏大小小于覆盖值
            )  # 抛出值错误

        if self.variance_size_override is None:
            x_var = x  # 使用前N列计算方差
        else:
            if hidden_size < self.variance_size_override:  # 计算方差（均值平方）
                raise ValueError(  # 应用RMSNorm
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"  # 如果需要在乘权重前转换类型
                )  # 先转回原始类型再乘权重

            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)  # 返回归一化结果
        x = x * torch.rsqrt(variance + self.variance_epsilon)  # 返回归一化结果和残差

        if self.cast_x_before_out_mul:
            x = self.weight * x.to(orig_dtype)
        else:
            x = (x * self.weight).to(orig_dtype)

        if residual is None:
            return x
        else:
            return x, residual

    def forward_cpu(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 如果CPU支持AMX
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 如果有残差
        if _is_cpu_amx_available:  # 如果有残差后加法
            if residual is not None:  # 加到残差上
                if post_residual_addition is not None:  # 调用CPU融合加法+RMSNorm
                    residual = residual + post_residual_addition  # 传入参数
                torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(  # 返回输入（原地修改）和残差
                    x, residual, self.weight.data, self.variance_epsilon  # 调用CPU RMSNorm
                )  # 传入参数
                return x, residual  # 否则CPU不支持AMX
            return torch.ops.sgl_kernel.rmsnorm_cpu(  # 使用原生实现
                x, self.weight.data, self.variance_epsilon
            )
        else:
            return self.forward_native(x, residual, post_residual_addition)

    def forward_xpu(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 如果有方差大小覆盖
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 使用原生实现
        if self.variance_size_override is not None:  # 如果启用了批量不变模式
            return self.forward_native(x, residual, post_residual_addition)  # 有残差或FSDP目标时使用原生实现
        if is_batch_invariant_mode_enabled():
            if (  # 使用原生实现
                residual is not None  # 使用批量不变RMS归一化
                or get_global_server_args().rl_on_policy_target == "fsdp"  # 输入
            ):  # 权重
                return self.forward_native(x, residual, post_residual_addition)  # epsilon值
            return rms_norm_batch_invariant(
                x,  # 如果有残差
                self.weight.data,  # 如果有残差后加法
                self.variance_epsilon,  # 加到残差上
            )  # 调用融合加法+RMSNorm
        if residual is not None:  # 返回输入（原地修改）和残差
            if post_residual_addition is not None:  # 无残差时调用RMSNorm
                residual = residual + post_residual_addition  # 返回输出
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_with_allreduce_fusion(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 是否使用注意力TP组（默认True）
        post_residual_addition: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        use_attn_tp_group: bool = True,  # 带allreduce融合的前向传播，优先使用flashinfer融合操作
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 调用共享融合逻辑
        """Forward with allreduce fusion, prioritizing flashinfer fused operations."""  # 传入模块、输入、残差、后加法、权重、TP组标志
        return _forward_with_allreduce_fusion(
            self, x, residual, post_residual_addition, self.weight, use_attn_tp_group
        )


class LayerNorm(MultiPlatformOp):  # 初始化函数
    def __init__(  # 隐藏层大小
        self,  # epsilon值，默认1e-6
        hidden_size: int,  # 是否使用逐元素仿射变换（默认True）
        eps: float = 1e-6,  # 是否使用偏置（默认True）
        elementwise_affine: bool = True,  # 数据类型（默认FP32）
        bias: bool = True,
        dtype: torch.dtype = torch.float32,  # 调用父类初始化
    ) -> None:  # 保存隐藏大小
        super().__init__()  # 保存epsilon值
        self.hidden_size = hidden_size  # 保存仿射变换标志
        self.variance_epsilon = eps  # 保存偏置标志
        self.elementwise_affine = elementwise_affine  # 保存数据类型
        self.use_bias = bias
        self.dtype = dtype  # 创建偏置参数（初始化为0）

        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=self.dtype))
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=self.dtype))

    def forward_cuda(  # 输入张量
        self,  # 返回归一化后的张量
        x: torch.Tensor,  # 如果满足FlashInfer LayerNorm条件
    ) -> torch.Tensor:  # FlashInfer LayerNorm可用
        if (  # 输入为BF16
            _flashinfer_layernorm_available  # 权重类型为FP32
            and x.dtype == torch.bfloat16
            and self.dtype == torch.float32  # 使用FlashInfer LayerNorm
        ):  # 否则
            return layernorm(x, self.weight, self.bias, self.variance_epsilon)  # 使用原生实现
        else:
            return self.forward_native(x)

    def forward_native(  # 输入张量
        self,  # 返回归一化后的张量
        x: torch.Tensor,  # 如果无仿射变换则权重为None
    ) -> torch.Tensor:  # 如果无偏置则为None
        weight = self.weight if self.elementwise_affine else None  # 保存原始数据类型
        bias = self.bias if self.use_bias else None  # 转换为目标数据类型
        orig_dtype = x.dtype  # 调用PyTorch内置LayerNorm
        x = x.to(self.dtype)  # 转回原始数据类型
        return F.layer_norm(
            x,
            (self.hidden_size,),
            weight=weight,
            bias=bias,
            eps=self.variance_epsilon,
        ).to(orig_dtype)

    def forward_hip(  # 输入张量
        self,  # 返回归一化后的张量
        x: torch.Tensor,  # 如果满足AITER LayerNorm条件
    ) -> torch.Tensor:  # AITER LayerNorm可用
        if (  # 输入为BF16或FP16
            _has_aiter_layer_norm  # 输入类型与权重类型相同
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.dtype == self.dtype  # 保存原始形状
        ):  # 重塑为2D
            orig_shape = x.shape  # 使用AITER LayerNorm并恢复形状
            x = x.reshape(-1, self.hidden_size)
            return layer_norm(x, self.weight, self.bias, self.variance_epsilon).view(  # 否则
                orig_shape  # 使用原生实现
            )
        else:
            return self.forward_native(x)

    def forward_npu(  # 输入张量
        self,  # 使用原生实现
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_cpu(  # 输入张量
        self,  # 如果CPU支持AMX
        x: torch.Tensor,  # 获取偏置数据（如果不使用偏置则为None）
    ) -> torch.Tensor:  # 调用CPU LayerNorm
        if _is_cpu_amx_available:  # 否则
            bias_data = self.bias.data if self.use_bias else None  # 使用原生实现
            return torch.ops.sgl_kernel.layernorm_cpu(
                x, self.weight.data, bias_data, self.variance_epsilon
            )
        else:
            return self.forward_native(x)


class GemmaRMSNorm(MultiPlatformOp):  # 初始化函数
    def __init__(  # 隐藏层大小
        self,  # epsilon值，默认1e-6
        hidden_size: int,
        eps: float = 1e-6,  # 调用父类初始化
    ) -> None:  # 权重初始化为0（Gemma风格，实际使用时会加1）
        super().__init__()  # 保存epsilon值
        self.weight = nn.Parameter(torch.zeros(hidden_size))  # 注册gemma_weight缓冲区（权重+1）
        self.variance_epsilon = eps  # 不持久化保存
        self.register_buffer(
            "gemma_weight", torch.ones_like(self.weight), persistent=False  # Gemma权重 = 标准权重 + 1。预计算一次
        )  # 如果TRTLLM allreduce融合将来提供gemma风格归一化则可移除
        # (Chen-0210) Gemma weight = standard_weight + 1. Precompute once.  # 设置权重加载器
        # If TRTLLM allreduce fusion ever provides gemma-style norm
        # natively, this can be removed.
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:  # 断言参数大小匹配
        assert param.size() == loaded_weight.size()  # 复制加载的权重
        param.data.copy_(loaded_weight)  # 保持存储稳定，用于CUDA图或捕获此缓冲区的融合路径
        # Keep storage stable for CUDA graphs or fused paths that capture this buffer.  # 预计算gemma_weight = weight + 1
        torch.add(param.data, 1.0, out=self.gemma_weight)

    def _forward_impl(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 是否需要重塑形状
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 如果需要重塑
        needs_reshape = x.dim() != 2 and residual is None  # 保存原始形状
        if needs_reshape:  # 重塑为2D
            original_shape = x.shape  # 如果有残差
            x = x.contiguous().reshape(-1, original_shape[-1])  # 如果有残差后加法
        if residual is not None:  # 加到残差上
            if post_residual_addition is not None:  # 调用Gemma融合加法+RMSNorm
                residual = residual + post_residual_addition  # 返回输入（原地修改）和残差
            gemma_fused_add_rmsnorm(  # 无残差时调用Gemma RMSNorm
                x, residual, self.weight.data, self.variance_epsilon  # 如果需要重塑回原形状
            )  # 重塑回原始形状
            return x, residual  # 返回输出
        out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
        if needs_reshape:
            out = out.reshape(original_shape)
        return out

    def forward_native(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 保存原始数据类型
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 如果有残差
        orig_dtype = x.dtype  # 如果有残差后加法
        if residual is not None:  # 加到残差上
            if post_residual_addition is not None:  # 输入加残差
                residual = residual + post_residual_addition  # 更新残差
            x = x + residual
            residual = x  # 转换为FP32

        x = x.float()  # 应用RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)  # Gemma权重 = 1.0 + weight
        x = x * torch.rsqrt(variance + self.variance_epsilon)  # 转回原始数据类型
        x = x * (1.0 + self.weight.float())  # 无残差返回x，有残差返回(x, residual)
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_cuda(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 调用内部前向实现
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._forward_impl(x, residual, post_residual_addition)

    def forward_hip(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 如果vLLM RMS归一化不可用
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 使用原生实现
        if not _has_vllm_rms_norm:
            return self.forward_native(x, residual, post_residual_addition)  # 获取gemma_weight（weight+1）

        w = self.gemma_weight  # aiter API说明
        if _use_aiter:  # 如果有残差
            # aiter API: rms_norm(input, weight, eps) -> output  # 分配输出张量
            #            fused_add_rms_norm(output, input, residual, residual_out, weight, eps)  # 分配残差输出张量
            if residual is not None:  # 如果有残差后加法
                output = torch.empty_like(x)  # 加到残差上
                residual_out = torch.empty_like(x)  # 调用AITER融合加法+RMSNorm
                if post_residual_addition is not None:  # 返回输出和残差输出
                    residual = residual + post_residual_addition  # 无残差时调用AITER RMSNorm
                fused_add_rms_norm(  # 否则使用vLLM API
                    output, x, residual, residual_out, w, self.variance_epsilon  # vllm API说明
                )  # 如果输入不连续
                return output, residual_out  # 转为连续
            return rms_norm(x, w, self.variance_epsilon)  # 如果有残差
        else:  # 分配输出张量
            # vllm API: rms_norm(out, input, weight, eps) -> None (in-place)  # 分配残差输出张量
            #           fused_add_rms_norm(out, input, residual_out, residual, weight, eps)  # 如果有残差后加法
            if not x.is_contiguous():  # 加到残差上
                x = x.contiguous()  # 调用vLLM融合加法+RMSNorm
            if residual is not None:  # 返回输出和残差输出
                out = torch.empty_like(x)  # 分配输出张量
                residual_out = torch.empty_like(x)  # 调用vLLM RMSNorm
                if post_residual_addition is not None:  # 返回输出
                    residual = residual + post_residual_addition
                fused_add_rms_norm(
                    out, x, residual_out, residual, w, self.variance_epsilon
                )
                return out, residual_out
            out = torch.empty_like(x)
            rms_norm(out, x, w, self.variance_epsilon)
            return out

    def forward_cpu(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 如果CPU支持AMX
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 如果有残差
        if _is_cpu_amx_available:  # 如果有残差后加法
            if residual is not None:  # 加到残差上
                if post_residual_addition is not None:  # 调用CPU Gemma融合加法+RMSNorm
                    residual = residual + post_residual_addition  # 返回输入和残差
                torch.ops.sgl_kernel.gemma_fused_add_rmsnorm_cpu(  # 调用CPU Gemma RMSNorm
                    x, residual, self.weight.data, self.variance_epsilon  # 使用原生实现
                )
                return x, residual
            return torch.ops.sgl_kernel.gemma_rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        return self.forward_native(x, residual, post_residual_addition)

    def forward_npu(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 如果配置了使用原生Gemma RMSNorm
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 使用原生实现
        if envs.SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM.get():  # 如果有残差
            return self.forward_native(x, residual)  # 如果有残差后加法
        if residual is not None:  # 加到残差上
            if post_residual_addition is not None:  # 调用NPU Gemma融合加法+RMSNorm
                residual = residual + post_residual_addition  # 返回归一化结果和残差
            norm_out, residual = add_gemma_rms_norm(
                x, self.weight, residual, self.variance_epsilon  # 无残差时调用NPU Gemma RMSNorm
            )  # 返回归一化结果
            return norm_out, residual

        x, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)
        return x

    def forward_xpu(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        post_residual_addition: Optional[torch.Tensor] = None,  # 调用内部前向实现
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._forward_impl(x, residual, post_residual_addition)

    def forward_with_allreduce_fusion(  # 输入张量
        self,  # 残差张量（可选）
        x: torch.Tensor,  # 残差后加法（可选）
        residual: Optional[torch.Tensor] = None,  # 是否使用注意力TP组（默认True）
        post_residual_addition: Optional[torch.Tensor] = None,  # 返回结果或(结果,残差)
        use_attn_tp_group: bool = True,  # 带allreduce融合的前向传播；融合内核使用1+weight
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 调用共享融合逻辑，传入gemma_weight（weight+1）
        """Forward with allreduce fusion; uses 1 + weight for fused kernels."""
        return _forward_with_allreduce_fusion(
            self,
            x,
            residual,
            post_residual_addition,
            self.gemma_weight,
            use_attn_tp_group=True,
        )


class Gemma3RMSNorm(MultiPlatformOp):  # 初始化函数
    def __init__(self, dim: int, eps: float = 1e-6):  # 维度大小
        super().__init__()  # epsilon值
        self.eps = eps  # 调用父类初始化
        self.weight = nn.Parameter(torch.zeros(dim))  # 保存epsilon值
        # Re-dispatch  # 权重初始化为0（Gemma3风格）

    def _norm(self, x):  # 归一化辅助函数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 计算RMSNorm：x * rsqrt(mean(x^2) + eps)

    def forward_native(self, x):  # 在FP32下进行归一化
        output = self._norm(x.float())  # Llama执行x.to(float16)*w，而Gemma3执行(x*w).to(float16)
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)  # Gemma权重 = 1.0 + weight
        # See https://github.com/huggingface/transformers/pull/29402  # 转回原始数据类型
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def forward_cpu(self, x):  # 如果CPU支持AMX且最后一维连续
        if _is_cpu_amx_available and x.stride(-1) == 1:  # 调用CPU Gemma3 RMSNorm
            return torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x, self.weight, self.eps)  # 使用原生实现
        return self.forward_native(x)

    def forward_cuda(self, x):  # 使用原生实现
        return self.forward_native(x)

    def forward_npu(self, x):  # 调用NPU Gemma RMSNorm
        output, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.eps)
        return output

    def extra_repr(self):  # 返回维度和eps信息
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma4RMSNorm(MultiPlatformOp):  # 初始化函数
    def __init__(  # 维度大小
        self,  # epsilon值，默认1e-6
        dim: int,  # 缩放偏移值，默认0.0
        eps: float = 1e-6,  # 是否带缩放权重，默认True
        scale_shift: float = 0.0,
        with_scale: bool = True,  # 调用父类初始化
    ):  # 保存是否带缩放标志
        super().__init__()
        self.with_scale = with_scale  # 如果带缩放

        if self.with_scale:  # 否则
            self.weight = nn.Parameter(torch.ones(dim))  # 注册为不持久化的缓冲区（固定为1，不参与训练）
        else:
            self.register_buffer("weight", torch.ones(dim), persistent=False)  # 保存epsilon值

        self.eps = eps
        self.scale_shift = scale_shift

    def __repr__(self):  # 获取维度大小
        dim = self.weight.shape[0]  # 返回类名和参数信息
        return (
            f"{self.__class__.__name__}(dim={dim}, eps={self.eps}, "  # 返回格式字符串第二行
            f"with_scale={self.with_scale}, scale_shift={self.scale_shift})"
        )

    def _norm(self, x):  # 计算均值平方 + eps
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps  # 返回x * pow(mean_squared, -0.5)
        return x * torch.pow(mean_squared, -0.5)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:  # 在FP32下进行归一化
        normed_output = self._norm(x.float())  # 如果带缩放
        if self.with_scale:  # 应用权重和缩放偏移：output * (weight + scale_shift)
            normed_output = normed_output * (self.weight.float() + self.scale_shift)  # 转回原始数据类型
        return normed_output.type_as(x)

    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:  # 如果CPU支持AMX
        if _is_cpu_amx_available:  # 调用CPU Gemma4 RMSNorm
            return torch.ops.sgl_kernel.gemma4_rmsnorm_cpu(  # 传入输入、权重、eps、scale_shift、with_scale
                x, self.weight.data, self.eps, self.scale_shift, self.with_scale  # 使用原生实现
            )
        return self.forward_native(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:  # 如果输入为空
        if x.numel() == 0:  # 返回空输入
            return x  # 是否需要重塑为2D
        needs_reshape = x.dim() != 2  # 如果需要重塑
        if needs_reshape:  # 保存原始形状
            original_shape = x.shape  # 重塑为2D
            x = x.contiguous().reshape(-1, original_shape[-1])  # 如果带缩放且缩放偏移为1.0
        if self.with_scale and self.scale_shift == 1.0:  # gemma_rmsnorm：norm(x) * (1 + weight)
            # gemma_rmsnorm: norm(x) * (1 + weight)  # 调用Gemma RMSNorm
            out = gemma_rmsnorm(x, self.weight.data, self.eps)  # 否则
        else:  # rmsnorm：norm(x) * weight
            # rmsnorm: norm(x) * weight  # with_scale=False -> weight为1 -> norm(x)*1
            # with_scale=False → weight is ones → norm(x) * 1 = norm(x)  # scale_shift=0.0 -> 无+1偏移的标准RMSNorm
            # scale_shift=0.0 → standard RMSNorm without +1 shift  # 调用标准RMSNorm
            out = rmsnorm(x, self.weight.data, self.eps)

        if needs_reshape:  # 重塑回原始形状
            out = out.reshape(original_shape)  # 返回输出
        return out

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:  # sgl_kernel的gemma_rmsnorm在ROCm上不可用
        # sgl_kernel's gemma_rmsnorm is not available on ROCm;  # 委托给纯PyTorch实现
        # delegate to the pure-PyTorch implementation.  # 使用原生实现
        return self.forward_native(x)


class RMSNormWithoutScale(MultiPlatformOp):  # 初始化函数
    def __init__(self, hidden_size: int, eps=1e-6):  # 隐藏层大小
        super().__init__()  # epsilon值
        self.hidden_size = hidden_size  # 调用父类初始化
        self.eps = eps  # 保存隐藏大小

    def _norm(self, x):  # 归一化辅助函数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 计算RMSNorm：x * rsqrt(mean(x^2) + eps)

    def forward_native(self, x):  # 保存原始数据类型
        orig_dtype = x.dtype  # 转换为FP32
        x = x.float()  # 计算方差
        variance = x.pow(2).mean(dim=-1, keepdim=True)  # 应用RMSNorm
        x = x * torch.rsqrt(variance + self.eps)  # 转回原始数据类型
        return x.to(orig_dtype)

    def forward_cuda(self, x):  # 使用原生实现
        return self.forward_native(x)

    def extra_repr(self):  # 返回隐藏大小和eps信息
        return f"{self.hidden_size}, eps={self.eps}"