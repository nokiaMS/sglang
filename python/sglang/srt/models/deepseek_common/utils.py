# DeepSeek 模型通用工具函数
# 本文件提供 DeepSeek V2/V3 系列模型的通用工具函数和平台检测变量，
# 包括设备类型检测（CUDA/HIP/NPU/MUSA/XPU/CPU）、量化相关工具函数、
# AWQ 反量化、YaRN 缩放、LLaMA-4 位置缩放等。

# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import math
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton.layer import get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_musa,
    is_npu,
    is_nvidia_cublas_version_ge_12_9,
    is_xpu,
)

# 平台检测变量：在模块加载时确定运行时平台
_is_hip = is_hip()  # AMD ROCm 平台
_is_cuda = is_cuda()  # NVIDIA CUDA 平台
_is_npu = is_npu()  # 华为昇腾 NPU 平台
_is_musa = is_musa()  # 摩尔线程 MUSA 平台
_is_fp8_fnuz = is_fp8_fnuz()  # FP8 FNUZ 格式检测
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用 AMD AITER 库
_is_cpu_amx_available = cpu_has_amx_support()  # CPU 是否支持 AMX 指令集
_is_cpu = is_cpu()  # 是否运行在 CPU 上
_is_xpu = is_xpu()  # Intel XPU 平台
_device_sm = get_device_sm()  # GPU 计算能力（如 80, 90 等）
_is_gfx95_supported = is_gfx95_supported()  # AMD gfx95 架构是否支持
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported  # 是否使用 AITER gfx95 优化


_is_cublas_ge_129 = is_nvidia_cublas_version_ge_12_9()  # cuBLAS 版本是否 >= 12.9

logger = logging.getLogger(__name__)

# NVFP4 检查点中需要 FP8 量化的注意力模块
NVFP4_CKPT_FP8_ATTN_QUANT_MODULES = ["q_b_proj"]

# 支持 absorbed core attention 的后端列表
# 这些后端在 forward_absorb_core 中使用 q_nope_out（已吸收 kv_b_proj 权重的 Q）
# 而非原始的 q/k 向量
FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = [
    "fa3",
    "dsa",
    "nsa",  # Deprecated alias for "dsa"
    "flashinfer",
    "cutlass_mla",
    "trtllm_mla",
    "cutedsl_mla",
    "tokenspeed_mla",
    "ascend",
    "intel_xpu",
]


# 获取当前设备对应的 AWQ 反量化函数
# CUDA: 使用 sgl_kernel 的 awq_dequantize
# HIP(ROCm): 使用 Triton 实现的 awq_dequantize_triton
# NPU: 使用 awq_dequantize_decomposition
# 其他平台: 返回 None
def awq_dequantize_func():
    """
    Get the AWQ dequantize function for the current device

    Return:
        - The AWQ dequantize function for the current device.
        - None if the current device is not supported.
    """
    if _is_cuda:
        from sgl_kernel import awq_dequantize

        return awq_dequantize
    elif _is_hip:
        from sglang.kernel_api_logging import debug_kernel_api
        from sglang.srt.layers.quantization.awq.awq_triton import (
            awq_dequantize_triton as awq_dequantize,
        )

        return debug_kernel_api(awq_dequantize, op_name="DeepseekCommon.awq_dequantize")
    elif _is_npu:
        from sglang.kernel_api_logging import debug_kernel_api
        from sglang.srt.layers.quantization.awq.awq_triton import (
            awq_dequantize_decomposition as awq_dequantize,
        )

        return debug_kernel_api(awq_dequantize, op_name="DeepseekCommon.awq_dequantize")
    else:
        return None


# 判断是否启用 NextN MoE 的 bf16 到 fp8 转换
# 需要：环境变量启用 + 量化配置为 modelopt_fp4 + MoE 运行后端为 DeepGEMM
def enable_nextn_moe_bf16_cast_to_fp8(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    return (
        envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get()
        and quant_config is not None
        and quant_config.get_name() == "modelopt_fp4"
        and get_moe_runner_backend().is_deep_gemm()
    )


# YaRN（Yet another RoPE extensioN）的 mscale 计算函数
# 用于在扩展上下文长度时对注意力分数进行缩放
def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# LLaMA-4 风格的位置缩放计算
# 当位置超过原始最大位置编码时，应用对数缩放因子
def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings)
    )
    return scaling[..., None, None]
