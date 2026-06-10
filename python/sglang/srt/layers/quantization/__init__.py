# 量化配置注册与分发模块
# 定义所有支持的量化方法（FP8、AWQ、GPTQ、BitsAndBytes、GGUF等），
# 提供量化配置查询接口，支持CPU、CUDA、NPU、MPS等不同平台的量化方法选择。

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
# 改编自 https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
from __future__ import annotations  # 启用延迟类型注解求值

import builtins  # 导入内置模块（用于保存原始isinstance）
import inspect  # 导入检查模块
from typing import TYPE_CHECKING, Dict, Optional, Type  # 导入类型提示

import torch  # 导入PyTorch


# Define empty classes as placeholders when vllm is not available
# 定义空类作为占位符，当vllm不可用时使用
class DummyConfig:  # 虚拟配置类，用作占位符
    def override_quantization_method(self, *args, **kwargs):  # 覆盖量化方法（空实现）
        return None  # 返回None


CompressedTensorsConfig = DummyConfig  # 临时将CompressedTensorsConfig设为占位符

from sglang.srt.layers.quantization.auto_round import AutoRoundConfig  # 导入AutoRound量化配置
from sglang.srt.layers.quantization.awq import AWQConfig, AWQCPUConfig, AWQMarlinConfig  # 导入AWQ量化配置
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.quantization.bitsandbytes import BitsAndBytesConfig  # 导入BitsAndBytes量化配置
from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config  # 导入块级INT8量化配置
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (  # 导入压缩张量量化配置（覆盖占位符）
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config  # 导入FP8量化配置
from sglang.srt.layers.quantization.fpgemm_fp8 import FBGEMMFp8Config  # 导入FBGEMM FP8量化配置
from sglang.srt.layers.quantization.gguf import GGUFConfig  # 导入GGUF量化配置
from sglang.srt.layers.quantization.gptq import (  # 导入GPTQ量化配置
    GPTQAscendConfig,  # GPTQ Ascend配置
    GPTQConfig,  # GPTQ通用配置
    GPTQMarlinConfig,  # GPTQ Marlin配置
)
from sglang.srt.layers.quantization.gptq_cpu import CPUGPTQConfig  # 导入CPU GPTQ量化配置
from sglang.srt.layers.quantization.mlx import MlxQuantizationConfig  # 导入MLX量化配置
from sglang.srt.layers.quantization.modelopt_quant import (  # 导入ModelOpt量化配置
    ModelOptFp4Config,  # ModelOpt FP4配置
    ModelOptFp8Config,  # ModelOpt FP8配置
    ModelOptMixedPrecisionConfig,  # ModelOpt混合精度配置
)
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig  # 导入ModelSlim量化配置
from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config  # 导入MoE WNA16量化配置
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config  # 导入MXFP4量化配置
from sglang.srt.layers.quantization.petit import PetitNvFp4Config  # 导入Petit NVFP4量化配置
from sglang.srt.layers.quantization.qoq import QoQConfig  # 导入QoQ量化配置
from sglang.srt.layers.quantization.quark.quark import QuarkConfig  # 导入Quark量化配置
from sglang.srt.layers.quantization.quark_int4fp8_moe import QuarkInt4Fp8Config  # 导入Quark INT4/FP8 MoE量化配置
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config  # 导入W4A FP8量化配置
from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config  # 导入W8A8 FP8量化配置
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config  # 导入W8A8 INT8量化配置
from sglang.srt.utils import (  # 导入平台检测工具函数
    cpu_has_amx_support,  # CPU AMX支持检测
    is_cpu,  # CPU平台检测
    is_cuda,  # CUDA平台检测
    is_hip,  # HIP平台检测
    is_mps,  # MPS平台检测
    is_npu,  # NPU平台检测
    mxfp_supported,  # MXFP支持检测
)

_is_mxfp_supported = mxfp_supported()  # 检测MXFP是否受支持

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.topk import TopKOutput  # 导入TopK输出类型

# Base quantization methods
# 基础量化方法
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {  # 量化方法名称到配置类的映射
    "fp8": Fp8Config,  # FP8量化
    "mxfp8": Fp8Config,  # MXFP8量化（使用FP8配置）
    "blockwise_int8": BlockInt8Config,  # 块级INT8量化
    "modelopt": ModelOptFp8Config,  # Auto-detect, defaults to FP8  # 自动检测，默认为FP8
    "modelopt_fp8": ModelOptFp8Config,  # ModelOpt FP8量化
    "modelopt_fp4": ModelOptFp4Config,  # ModelOpt FP4量化
    "modelopt_mixed": ModelOptMixedPrecisionConfig,  # ModelOpt混合精度量化
    "w8a8_int8": W8A8Int8Config,  # W8A8 INT8量化
    "w8a8_fp8": W8A8Fp8Config,  # W8A8 FP8量化
    "awq": AWQConfig,  # AWQ量化
    "awq_marlin": AWQMarlinConfig,  # AWQ Marlin量化
    "bitsandbytes": BitsAndBytesConfig,  # BitsAndBytes量化
    "gguf": GGUFConfig,  # GGUF量化
    "gptq": GPTQConfig,  # GPTQ量化
    "gptq_marlin": GPTQMarlinConfig,  # GPTQ Marlin量化
    "moe_wna16": MoeWNA16Config,  # MoE WNA16量化
    "compressed-tensors": CompressedTensorsConfig,  # 压缩张量量化
    "qoq": QoQConfig,  # QoQ量化
    "w4afp8": W4AFp8Config,  # W4A FP8量化
    "petit_nvfp4": PetitNvFp4Config,  # Petit NVFP4量化
    "fbgemm_fp8": FBGEMMFp8Config,  # FBGEMM FP8量化
    "quark": QuarkConfig,  # Quark量化
    "auto-round": AutoRoundConfig,  # AutoRound量化
    "modelslim": ModelSlimConfig,  # ModelSlim量化
    "quark_int4fp8_moe": QuarkInt4Fp8Config,  # Quark INT4/FP8 MoE量化
}


if is_cpu() or is_cuda() or (_is_mxfp_supported and is_hip()):  # CPU、CUDA或MXFP支持的HIP平台
    BASE_QUANTIZATION_METHODS.update(  # 添加MXFP4支持
        {
            "mxfp4": Mxfp4Config,  # MXFP4量化配置
        }
    )


if is_npu():  # NPU平台
    BASE_QUANTIZATION_METHODS.update(  # 使用Ascend版GPTQ配置覆盖通用GPTQ
        {
            "gptq": GPTQAscendConfig,  # GPTQ Ascend量化配置
        }
    )


if is_mps():  # MPS平台（Apple Silicon）
    BASE_QUANTIZATION_METHODS.update(  # 添加MLX量化方法
        {
            "mlx_q4": MlxQuantizationConfig,  # MLX 4位量化
            "mlx_q8": MlxQuantizationConfig,  # MLX 8位量化
        }
    )

# subset of above quant methods, supported on CPU
# 上述量化方法的子集，在CPU上受支持
CPU_QUANTIZATION_METHODS = {  # CPU支持的量化方法映射
    "fp8": Fp8Config,  # FP8量化
    "w8a8_int8": W8A8Int8Config,  # W8A8 INT8量化
    "compressed-tensors": CompressedTensorsConfig,  # 压缩张量量化
    "awq": AWQCPUConfig,  # AWQ CPU量化
    "gptq": CPUGPTQConfig,  # CPU GPTQ量化
    "mxfp4": Mxfp4Config,  # MXFP4量化
}

QUANTIZATION_METHODS = {**BASE_QUANTIZATION_METHODS}  # 全局量化方法字典（基于基础方法）


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:  # 根据量化方法名获取对应的配置类
    if quantization not in QUANTIZATION_METHODS:  # 如果量化方法不存在
        raise ValueError(  # 抛出异常
            f"Invalid quantization method: {quantization}. "  # 无效的量化方法
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"  # 可用方法列表
        )
    from sglang.srt.utils import is_cpu  # 导入CPU平台检测函数

    if is_cpu() and cpu_has_amx_support():  # 如果是CPU平台且支持AMX指令
        if quantization not in CPU_QUANTIZATION_METHODS:  # 如果该量化方法在CPU上不支持
            raise ValueError(  # 抛出异常
                f"Invalid quantization method on CPU: {quantization}. "  # CPU上无效的量化方法
                f"Available methods on CPU: {list(QUANTIZATION_METHODS.keys())}"  # CPU上可用方法列表
            )
        else:  # 否则
            return CPU_QUANTIZATION_METHODS[quantization]  # 返回CPU特定的量化配置

    return QUANTIZATION_METHODS[quantization]  # 返回通用量化配置


original_isinstance = builtins.isinstance  # 保存Python内置的isinstance函数（用于后续可能的monkey-patch）
