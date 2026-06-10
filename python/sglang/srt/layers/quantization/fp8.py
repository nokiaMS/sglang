# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py
#
# FP8 量化配置与实现模块
# 本文件实现了 FP8（8位浮点）量化方案，包括：
# - Fp8Config: FP8 量化配置类，支持 per-tensor、per-channel 和 block-wise 量化
# - Fp8LinearMethod: FP8 线性层量化方法，支持权重和激活的 FP8 量化推理
# - Fp8MoEMethod: FP8 MoE（混合专家）层量化方法，支持多专家模型的 FP8 推理
# - Fp8KVCacheMethod: FP8 KV 缓存量化方法
# 支持 FP8 E4M3 和 E5M2 格式，兼容 NVIDIA、AMD ROCm、Intel CPU (AMX) 等多种硬件平台

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

# 分布式训练相关导入

# 分布式训练相关导入
from sglang.srt.distributed import get_tensor_model_parallel_world_size, get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmFp8MoeQuantInfo,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    get_moe_a2a_backend,
    get_moe_padding_size,
    get_moe_runner_backend,
    get_moe_weight_sizes,
)
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
    scaled_fp8_quant,
)
from sglang.srt.layers.quantization.fp8_utils import (
    _use_aiter_bpreshuffle_gfx95,
    apply_fp8_linear,
    can_auto_enable_marlin_fp8,
    cutlass_fp8_supported,
    dispatch_w8a8_block_fp8_linear,
    dispatch_w8a8_mxfp8_linear,
    get_fp8_gemm_runner_backend,
    input_to_float8,
    mxfp8_group_quantize,
    normalize_e4m3fn_to_e4m3fnuz,
    requant_weight_ue8m0_inplace,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.quantization.utils import (
    all_close_1d,
    convert_to_channelwise,
    is_layer_skipped,
    per_tensor_dequantize,
    requantize_with_max_scale,
)
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_musa,
    is_npu,
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
    log_info_on_rank0,
    print_warning_once,
    set_weight_attrs,
    use_intel_amx_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.aiter import AiterMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config
    from sglang.srt.models.utils import WeightsMapper

_is_hip = is_hip()  # 是否为 AMD ROCm 平台
_is_cuda = is_cuda()  # 是否为 NVIDIA CUDA 平台
_is_musa = is_musa()  # 是否为摩尔线程 MUSA 平台
_is_npu = is_npu()  # 是否为华为 NPU 平台
_is_cpu_amx_available = cpu_has_amx_support()  # CPU 是否支持 AMX 指令集
_is_cpu = is_cpu()  # 是否为 CPU 平台
_is_fp8_fnuz = is_fp8_fnuz()  # 是否使用 FP8 FNUZ 格式（AMD MI300x 硬件格式）
_use_hip_int4 = get_bool_env_var("SGLANG_INT4_WEIGHT") and _is_hip  # 是否在 ROCm 上使用 INT4 权重
_use_aiter = envs.SGLANG_USE_AITER.get() and _is_hip  # 是否使用 AMD AITER 加速库
_is_shuffle_moe_mxfp4 = is_gfx95_supported()  # 是否支持 GFX95 平台（用于 MXFP4 权重重排）


def _require_fp4_dtype():
    """检查并返回 FP4 数据类型，如果不支持则抛出异常"""
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        raise RuntimeError(
            "DeepSeek-V4 FP4 experts require torch.float4_e2m1fn_x2 support."
        )
    return fp4_dtype


if _use_aiter or _use_hip_int4:
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )

if _use_aiter:
    from sglang.srt.layers.quantization.fp8_utils import (
        aiter_w8a8_block_fp8_linear,
        use_aiter_triton_gemm_w8a8_tuned_gfx950,
    )


ACTIVATION_SCHEMES = ["static", "dynamic"]  # 支持的激活量化方案：静态（预先计算缩放因子）和动态（运行时计算）

logger = logging.getLogger(__name__)


class Fp8Config(QuantizationConfig):
    """FP8 量化配置类。
    
    管理 FP8 量化的各项参数，包括：
    - 检查点格式（FP8 序列化 vs FP16/BF16）
    - 激活量化方案（静态/动态）
    - 忽略量化的层
    - 权重分块大小（用于 block-wise 量化）
    - 是否使用 MXFP8 格式
    - 是否为 FP4 专家（DeepSeek-V4 特性）
    """

    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        use_mxfp8: bool = False,
        is_fp4_experts: bool = False,
    ) -> None:
        super().__init__()
        # DSV4 mxfp4-packed (True) vs converted FP8 (False); injected by
        # model_loader from ModelConfig. Default False off the DSV4 path.
        # DSV4 的 mxfp4-packed 格式 (True) 与转换后的 FP8 格式 (False)；
        # 由 model_loader 从 ModelConfig 注入，非 DSV4 路径默认为 False
        self.is_fp4_experts = is_fp4_experts
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized  # 检查点是否为 FP8 序列化格式
        if is_checkpoint_fp8_serialized:
            log_info_on_rank0(logger, "Detected fp8 checkpoint.")  # 检测到 FP8 检查点
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")  # 不支持的激活量化方案
        self.activation_scheme = activation_scheme  # 激活量化方案：static 或 dynamic
        self.ignored_layers = ignored_layers or []  # 不进行量化的层列表
        if ignored_layers_str := envs.SGLANG_FP8_IGNORED_LAYERS.get():  # 从环境变量追加忽略的层
            self.ignored_layers.extend(
                [
                    layer.strip()
                    for layer in ignored_layers_str.split(",")
                    if layer.strip()
                ]
            )
        self.packed_modules_mapping = packed_modules_mapping or {}  # 打包模块映射（如 QKV 打包）
        self.use_mxfp8 = use_mxfp8  # 是否使用 MXFP8 格式（Microscaling FP8）
        if weight_block_size is not None:  # 分块量化参数校验
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    f"The block-wise quantization only supports fp8-serialized checkpoint for now."
                )  # 分块量化目前仅支持 FP8 序列化检查点
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions."
                )  # 权重量化分块大小必须为 2 维
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."
                )  # 分块量化目前仅支持动态激活方案
        if self.use_mxfp8:  # MXFP8 的分块大小固定为 [1, 32]
            if weight_block_size is None:
                weight_block_size = [1, 32]
            elif weight_block_size != [1, 32]:
                raise ValueError("MXFP8 requires weight_block_size=[1, 32].")
        self.weight_block_size = weight_block_size  # 权重分块大小，None 表示不使用分块量化

    def get_name(self) -> str:
        """返回量化方法名称，MXFP8 或 FP8"""
        return "mxfp8" if self.use_mxfp8 else "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        """返回支持的激活数据类型：bfloat16 和 float16"""
        return [torch.bfloat16, torch.half]

    def get_min_capability(self) -> int:
        """返回最低硬件计算能力要求。MXFP8 需要 SM100，普通 FP8 需要 SM80"""
        if _is_musa:
            return 31

        return 100 if self.use_mxfp8 else 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """返回配置文件名列表（空列表表示不从文件加载）"""
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Fp8Config:
        """从配置字典创建 Fp8Config 实例"""
        quant_method = cls.get_from_keys(config, ["quant_method"])  # 获取量化方法名
        use_mxfp8 = "mxfp8" in quant_method  # 检查是否为 MXFP8 方法
        is_checkpoint_fp8_serialized = ("fp8" in quant_method) or use_mxfp8  # 检查检查点是否为 FP8 格式
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])  # 获取激活量化方案
        packed_modules_mapping = (
            cls.get_from_keys_or(config, ["packed_modules_mapping"], {}) or {}
        )
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        if ignored_layers:
            # Keep both "model." and non-"model." variants for robust prefix matching.
            # 保留带 "model." 前缀和不带前缀的两种变体，以确保前缀匹配的鲁棒性
            normalized = []
            for layer in ignored_layers:
                base = layer.removeprefix("model.")
                normalized.append(base)
                normalized.append(f"model.{base}")
            ignored_layers = normalized
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        if use_mxfp8 and weight_block_size is not None:
            logger.warning(
                "MXFP8 ignoring incoming weight_block_size in config.json; it is fixed to [1, 32]."
            )
            weight_block_size = [1, 32]
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            packed_modules_mapping=packed_modules_mapping,
            use_mxfp8=use_mxfp8,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """根据层类型返回对应的量化方法。
        
        - LinearBase 层 -> Fp8LinearMethod
        - FusedMoE 层 -> Fp8MoEMethod（或 FP4 变体）
        - RadixAttention 层 -> Fp8KVCacheMethod
        - 其他层 -> None（不量化）
        """
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.radix_attention import RadixAttention

        if isinstance(layer, LinearBase):
            if is_layer_skipped(  # 检查该层是否在忽略列表中
                prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()  # 不量化的线性方法
            return Fp8LinearMethod(self)  # FP8 线性层量化方法
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(  # 检查该 MoE 层是否在忽略列表中
                prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedFusedMoEMethod(  # 不量化的 MoE 方法
                    layer.use_triton_kernels, layer.use_flashinfer_trtllm_moe
                )

            fp8_method = Fp8MoEMethod(self)  # FP8 MoE 量化方法

            if self.is_fp4_experts and get_moe_runner_backend().is_marlin():  # FP4 专家 + Marlin 后端
                from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
                    Mxfp4MarlinMoEMethod,
                )

                return Mxfp4MarlinMoEMethod(fp8_method, prefix=prefix)

            if self.is_fp4_experts and get_moe_runner_backend().is_flashinfer_mxfp4():
                # SM100 (Blackwell) -> trtllm-gen path.
                # SM100 (Blackwell) -> 使用 trtllm-gen 路径
                # SM90  (Hopper)    -> cutlass mixed-input path (FlashInfer #3084).
                # SM90  (Hopper)    -> 使用 cutlass 混合输入路径 (FlashInfer #3084)
                if is_sm90_supported() and not is_sm100_supported():
                    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
                        Mxfp4FlashinferCutlassMoEMethod,
                    )

                    return Mxfp4FlashinferCutlassMoEMethod(fp8_method, prefix=prefix)

                from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                    Mxfp4FlashinferTrtllmMoEMethod,
                )

                return Mxfp4FlashinferTrtllmMoEMethod(fp8_method, prefix=prefix)
            return fp8_method
        elif isinstance(layer, RadixAttention):
            return Fp8KVCacheMethod(self)  # FP8 KV 缓存量化方法
        return None

    def get_scaled_act_names(self) -> List[str]:
        """返回需要缩放的激活名称列表（FP8 不需要特殊缩放）"""
        return []

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: "WeightsMapper"):
        """应用权重名称映射，将 HuggingFace 名称映射为 SGLang 名称（用于忽略层列表）"""
        if self.ignored_layers:
            self.ignored_layers = list(
                dict.fromkeys(hf_to_sglang_mapper.apply_list(self.ignored_layers))
            )


class Fp8LinearMethod(LinearMethodBase):
    """FP8 线性层量化方法。

    支持以下量化方案：
    - 逐通道权重量化 + 逐 token 激活量化
    - 逐张量权重量化 + 逐张量激活量化
    - 分块权重量化 + 分块激活量化

    支持以下检查点格式：
    - FP8 检查点
    - FP16/BF16 检查点（在这种情况下，权重将在加载时被量化为 FP8）

    注意：
    - 激活量化方案可以是静态或动态的，动态激活量化更常用
    - 在 NV 平台上，如果未启用分块量化，则默认使用逐通道权重量化

    Args:
        quant_config: 量化配置对象
    """

    """Linear method for FP8.

    It supports the following quantization schemes:
    - Per-channel weight quantization + per-token activation quantization
    - Per-tensor weight quantization + per-tensor activation quantization
    - Blockwise weight quantization + blockwise activation quantization

    It supports the following checkpoint formats:
    - FP8 checkpoint
    - FP16/BF16 checkpoint. In this case, the weights will be quantized to FP8 during the weight loading.

    Notes:
    - The activation quantization scheme can be static or dynamic. The dynamic activation quantization is more commonly used.
    - On NV platforms, the per-channel weight quantization is used by default, if block quantization is not enabled.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Union[Fp8Config, W4AFp8Config]):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()  # 检查是否支持 CUTLASS FP8 内核

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        # 对于缺乏 FP8 硬件支持的 GPU，可以利用 Marlin 内核进行快速仅权重 FP8 量化
        self.use_marlin = False
        if _is_cuda:
            force_marlin = get_bool_env_var("SGLANG_FORCE_FP8_MARLIN")  # 是否强制使用 Marlin
            auto_enable = can_auto_enable_marlin_fp8()  # 是否可以自动启用 Marlin
            self.use_marlin = force_marlin or auto_enable

        self.use_mxfp8 = getattr(self.quant_config, "use_mxfp8", False)  # 是否使用 MXFP8 格式
        self.block_quant = (  # 是否使用分块量化（MXFP8 或设置了 weight_block_size）
            self.use_mxfp8 or self.quant_config.weight_block_size is not None
        )
        self.w8a8_block_fp8_linear = None  # W8A8 分块 FP8 线性计算函数
        self.w8a8_mxfp8_linear = None  # W8A8 MXFP8 线性计算函数
        if self.use_mxfp8:
            self.w8a8_mxfp8_linear = dispatch_w8a8_mxfp8_linear()  # 分发 MXFP8 线性计算实现
        else:
            self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()  # 分发分块 FP8 线性计算实现
        self.is_checkpoint_fp8_serialized = (  # 检查点是否为 FP8 序列化
            self.quant_config.is_checkpoint_fp8_serialized
        )
        self.use_aiter_fp8_per_token = envs.SGLANG_USE_AITER_FP8_PER_TOKEN.get()  # AITER 是否使用逐 token FP8
        self.use_per_token_if_dynamic = False  # 动态模式下是否使用逐 token 量化

    def validate_block_quant_shapes(
        self,
        input_size: int,
        input_size_per_partition: int,
        output_size: int,
        output_size_per_partition: int,
        output_partition_sizes: List[int],
        skip_block_quant_check: bool = False,
    ):
        """验证分块量化在张量并行下的形状兼容性。
        
        确保分区后的权重尺寸能被量化分块大小整除，
        包括行并行（对 block_k）和列并行（对 block_n）的校验。
        """
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行度
        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )

        if skip_block_quant_check:
            print_warning_once(
                "Skipping block quantization checks for weight partition."
            )
        else:
            # Required by row parallel
            # 行并行要求：输入分区大小必须能被 block_k 整除
            if tp_size > 1 and input_size // input_size_per_partition == tp_size:
                if input_size_per_partition % block_k != 0:
                    raise ValueError(
                        f"Weight input_size_per_partition = "
                        f"{input_size_per_partition} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )
            # Required by column parallel or enabling merged weights
            # 列并行或合并权重要求：输出分区大小必须能被 block_n 整除
            if (
                tp_size > 1 and output_size // output_size_per_partition == tp_size
            ) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}."
                        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        skip_block_quant_check: bool = False,
        **extra_weight_attrs,
    ):
        """为线性层创建 FP8 量化所需的权重参数和缩放因子参数。
        
        根据是否使用分块量化，创建不同类型的缩放因子参数：
        - 分块量化：BlockQuantScaleParameter（二维缩放矩阵）
        - 非分块量化：PerTensorScaleParameter（逐张量缩放因子）
        同时根据激活方案（静态/动态）决定是否创建输入缩放因子。
        """
        # Copy the layer attributes
        # 复制层属性
        output_size_per_partition = sum(output_partition_sizes)  # 输出分区总大小
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")

        if self.block_quant:
            block_n, block_k = self.quant_config.weight_block_size
            self.validate_block_quant_shapes(
                input_size,
                input_size_per_partition,
                output_size,
                output_size_per_partition,
                output_partition_sizes,
                skip_block_quant_check,
            )

        # Create the weight
        # 创建权重参数，根据检查点格式选择数据类型
        weight_dtype = (
            torch.float8_e4m3fn if self.is_checkpoint_fp8_serialized else params_dtype
        )
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        # 如果检查点已是 FP8 序列化的，则直接加载缩放因子；
        # 否则等到 process_weights_after_loading 阶段再量化
        if self.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            # 权重缩放因子
            if self.block_quant:
                if hasattr(self.quant_config, "activation_scheme"):
                    assert self.quant_config.activation_scheme == "dynamic"
                elif hasattr(self.quant_config, "linear_activation_scheme"):
                    assert self.quant_config.linear_activation_scheme == "dynamic"
                if self.use_mxfp8 and not self.is_checkpoint_fp8_serialized:
                    raise ValueError(
                        "MXFP8 requires fp8-serialized checkpoint for linear layers."
                    )
                scale_dtype = torch.uint8 if self.use_mxfp8 else torch.float32  # MXFP8 使用 uint8（UE8M0），普通 FP8 使用 float32
                scale_init = torch.zeros if scale_dtype == torch.uint8 else torch.empty  # MXFP8 用零初始化，FP8 用空初始化
                scale = BlockQuantScaleParameter(
                    data=scale_init(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=scale_dtype,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale.format_ue8m0 = self.use_mxfp8  # 标记缩放因子是否为 UE8M0 格式
                if scale_dtype != torch.uint8:
                    scale[:] = torch.finfo(torch.float32).min  # 用 float32 最小值初始化（后续会被实际值覆盖）
                layer.register_parameter("weight_scale_inv", scale)  # 注册权重缩放因子（逆缩放）
            else:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)

            # INPUT ACTIVATION SCALE
            # 输入激活缩放因子（仅静态量化方案需要）
            if (
                hasattr(self.quant_config, "activation_scheme")
                and self.quant_config.activation_scheme == "static"
            ) or (
                hasattr(self.quant_config, "linear_activation_scheme")
                and self.quant_config.linear_activation_scheme == "static"
            ):
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading_block_quant(self, layer: Module) -> None:
        """分块量化模式下的权重后处理。
        
        根据不同平台和配置进行：
        - ROCm (FNUZ)：归一化权重和缩放因子到 e4m3fnuz 格式
        - CPU (AMX)：处理权重以适配 AMX 指令集
        - MXFP8：处理缩放因子格式和重排
        - 普通 FP8 分块：可能需要将缩放因子重量化为 UE8M0 格式
        """
        # If ROCm, normalize the weights and scales to e4m3fnuz
        # 如果是 ROCm 平台，将权重和缩放因子归一化为 e4m3fnuz 格式
        if _is_fp8_fnuz:
            # activation_scheme: dynamic
            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
            )
            layer.input_scale = None
        elif _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "Fp8LinearMethod on CPU requires that CPU has AMX support"  # CPU 上 FP8 需要 AMX 支持
            _amx_process_weight_after_loading(layer, ["weight"])  # 使用 AMX 处理权重
            layer.weight_scale_inv = torch.nn.Parameter(
                layer.weight_scale_inv.data, requires_grad=False
            )
            return
        elif self.use_mxfp8:
            if not self.is_checkpoint_fp8_serialized:
                self._quantize_mxfp8_weights(layer)  # 在线量化权重为 MXFP8 格式
                return
            # MXFP8 scales are stored as UE8M0 uint8; no requantization here.
            # MXFP8 缩放因子以 UE8M0 uint8 格式存储，此处无需重量化
            # Keep parameter object to preserve weight_loader attrs for hot reload.
            # 保留参数对象以保持 weight_loader 属性，支持热重载
            layer.weight_scale_inv.requires_grad_(False)
            layer.weight_scale_inv.format_ue8m0 = True
            self._process_mxfp8_linear_weight_scale(layer)  # 处理 MXFP8 线性层权重缩放因子
            return
        else:
            # For fp8 linear weights run with deepgemm, the weights and scales need be requantized to ue8m0
            # 对于使用 DeepGEMM 运行的 FP8 线性权重，需要将权重和缩放因子重量化为 UE8M0 格式
            from sglang.srt.layers.quantization.fp8_utils import (
                deepgemm_w8a8_block_fp8_linear_with_fallback,
            )
            from sglang.srt.model_loader.utils import (
                should_deepgemm_weight_requant_ue8m0,
            )

            if (
                should_deepgemm_weight_requant_ue8m0(
                    weight_block_size=getattr(
                        self.quant_config, "weight_block_size", None
                    ),
                )
                and (
                    self.w8a8_block_fp8_linear
                    is deepgemm_w8a8_block_fp8_linear_with_fallback
                )
                and (not layer.weight_scale_inv.format_ue8m0)
            ):
                requant_weight_ue8m0_inplace(  # 原地将缩放因子重量化为 UE8M0 格式
                    layer.weight,
                    layer.weight_scale_inv,
                    self.quant_config.weight_block_size,
                )
                layer.weight_scale_inv.format_ue8m0 = True
            weight, weight_scale = layer.weight.data, layer.weight_scale_inv.data

        layer.weight.data = weight.data
        layer.weight_scale_inv.data = weight_scale.data

        if (
            _use_aiter_bpreshuffle_gfx95
            and self.w8a8_block_fp8_linear is aiter_w8a8_block_fp8_linear
        ):
            n, k = layer.weight.shape
            if not use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k):
                # TODO(1am9trash), to deal with case that this branch chance
                # drops as use_aiter_triton_gemm_w8a8_tuned_gfx950() expands
                t = shuffle_weight(layer.weight, (16, 16))
                layer.weight.copy_(t)
                del t

    def _process_mxfp8_linear_weight_scale(self, layer: Module) -> None:
        """处理 MXFP8 线性层的权重缩放因子，根据 GEMM 后端进行重排/交织。
        
        - FlashInfer TRT-LLM 后端：对权重和缩放因子进行矩阵重排
        - FlashInfer CUTLASS 后端：对缩放因子进行块级交织（swizzle）
        - Triton 后端：直接使用标准 2D UE8M0 缩放因子，无需处理
        """
        if not self.use_mxfp8:
            return

        if get_fp8_gemm_runner_backend().is_flashinfer_trtllm():
            # FlashInfer TRT-LLM 路径：对权重和缩放因子进行矩阵重排
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight.data
            scale_u8 = layer.weight_scale_inv.data
            n, k = weight.shape
            epilogue_tile_m = 128  # epilogue 分块大小为 128

            copy_or_rebind_param(
                layer,
                "weight",
                shuffle_matrix_a(
                    weight.contiguous().view(torch.uint8), epilogue_tile_m
                ).view(torch.float8_e4m3fn),
            )
            copy_or_rebind_param(
                layer,
                "weight_scale_inv",
                shuffle_matrix_sf_a(
                    scale_u8.contiguous().view(torch.uint8).reshape(n, k // 32),
                    epilogue_tile_m,
                    num_elts_per_sf=32,
                )
                .reshape_as(scale_u8)
                .contiguous(),
            )
        elif get_fp8_gemm_runner_backend().is_flashinfer_cutlass():
            # FlashInfer CUTLASS 路径：对缩放因子进行块级交织
            from flashinfer import block_scale_interleave

            scale_u8 = layer.weight_scale_inv.data
            # block_scale_interleave may pad and/or reshape scales,
            # so store swizzled scales separately to keep weight update working
            # block_scale_interleave 可能会填充和/或重塑缩放因子，
            # 因此单独存储交织后的缩放因子以保持权重更新正常工作
            copy_or_rebind_param(
                layer,
                "weight_scale_inv_swizzled",
                block_scale_interleave(scale_u8.contiguous()).contiguous(),
            )
        else:
            # Triton path consumes canonical 2D UE8M0 scales directly.
            # Triton 路径直接使用标准 2D UE8M0 缩放因子，无需额外处理
            return

    def _quantize_mxfp8_weights(self, layer: Module) -> None:
        """将权重在线量化为 MXFP8 格式。
        
        对非 FP8 序列化的检查点，在加载后将 FP16/BF16 权重量化为 MXFP8，
        并设置对应的缩放因子参数。
        """
        weight = layer.weight.data
        qweight, weight_scale = mxfp8_group_quantize(weight)  # 使用 MXFP8 分组量化
        # Keep parameter objects to preserve weight_loader attrs for hot reload.
        # 保留参数对象以保持 weight_loader 属性，支持热重载
        layer.weight.data = qweight
        layer.weight.requires_grad_(False)
        if hasattr(layer, "weight_scale_inv") and layer.weight_scale_inv is not None:
            layer.weight_scale_inv.data = weight_scale  # 更新已有缩放因子参数
            layer.weight_scale_inv.requires_grad_(False)
        else:
            # First-time online MXFP8 quantization (no serialized scales).
            # 首次在线 MXFP8 量化（无序列化的缩放因子）
            layer.register_parameter(
                "weight_scale_inv", Parameter(weight_scale, requires_grad=False)
            )
        layer.weight_scale_inv.format_ue8m0 = True  # 标记缩放因子为 UE8M0 格式
        self._process_mxfp8_linear_weight_scale(layer)
        layer.input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        """权重加载后的处理，根据量化配置对权重进行转换和缩放因子处理。
        
        分块量化路径：
        - 调用 process_weights_after_loading_block_quant
        非分块量化路径：
        - FP16/BF16 检查点：在线量化为 FP8
        - FP8 检查点：转换为逐通道或逐张量缩放格式
        最后，如果使用 Marlin 后端，则准备 Marlin 所需的参数。
        """
        if self.block_quant:
            self.process_weights_after_loading_block_quant(layer)
        else:
            layer.weight = Parameter(layer.weight.data, requires_grad=False)

            # If checkpoint not serialized fp8, quantize the weights.
            # 如果检查点不是 FP8 序列化的，则将权重量化为 FP8
            if not self.is_checkpoint_fp8_serialized:
                if (
                    self.cutlass_fp8_supported
                    or self.use_marlin
                    or (_use_aiter and self.use_aiter_fp8_per_token)
                ):
                    # apply per-channel quantization default as
                    # cutlass sgl-kernel and marlin only support per-channel scale
                    # 默认应用逐通道量化，因为 cutlass sgl-kernel 和 marlin 仅支持逐通道缩放
                    qweight, weight_scale = per_token_group_quant_fp8(  # 逐 token 组 FP8 量化
                        layer.weight, layer.weight.shape[-1]
                    )
                    weight_scale = weight_scale.t().contiguous()  # 转置并保证连续内存
                    if _use_aiter and self.use_aiter_fp8_per_token:
                        self.use_per_token_if_dynamic = True  # AITER 启用逐 token 量化
                        qweight = shuffle_weight(qweight.contiguous(), (16, 16))  # AITER 权重重排
                else:
                    # per-tensor quantization
                    # 逐张量量化
                    qweight, weight_scale = input_to_float8(layer.weight)

                # Update the layer with the new values.
                # 使用量化后的值更新层参数
                layer.weight = Parameter(qweight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                layer.input_scale = None

            # If checkpoint is fp8, handle that there are N scales for N
            # shards in a fused module
            # 如果检查点是 FP8 的，需要处理融合模块中 N 个分片对应 N 个缩放因子的情况
            else:
                layer.weight_scale = Parameter(
                    layer.weight_scale.data, requires_grad=False
                )
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.data, requires_grad=False
                    )

                # cutlass sgl-kernel and marlin only support per-channel scale; aiter supports per-channel scale
                # cutlass sgl-kernel 和 marlin 仅支持逐通道缩放；aiter 支持逐通道缩放
                if (
                    self.cutlass_fp8_supported
                    or self.use_marlin
                    or (_use_aiter and self.use_aiter_fp8_per_token)
                ):
                    weight = layer.weight
                    weight_scale = convert_to_channelwise(  # 将逐张量缩放转换为逐通道缩放
                        layer.weight_scale, layer.logical_widths
                    )
                    if _use_aiter and self.use_aiter_fp8_per_token:
                        # Otherwise, by default, aiter only uses per-tensor quantization
                        # 默认情况下，aiter 仅使用逐张量量化
                        self.use_per_token_if_dynamic = True
                        if _is_fp8_fnuz:
                            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                weight=weight,
                                weight_scale=weight_scale,
                            )
                        weight = shuffle_weight(weight.contiguous(), (16, 16))
                else:
                    # Dequant -> Quant with max scale so we can run per tensor.
                    # 反量化 -> 使用最大缩放因子重新量化，以便运行逐张量推理
                    weight = layer.weight
                    weight_scale = layer.weight_scale
                    # If ROCm, normalize the weights and scales to e4m3fnuz
        # 如果是 ROCm 平台，将权重和缩放因子归一化为 e4m3fnuz 格式
        if _is_fp8_fnuz:
                    if _is_fp8_fnuz:
                        weight, weight_scale, input_scale = (
                            normalize_e4m3fn_to_e4m3fnuz(
                                weight=weight,
                                weight_scale=weight_scale,
                                input_scale=layer.input_scale,
                            )
                        )
                        if input_scale is not None:
                            layer.input_scale = Parameter(
                                input_scale, requires_grad=False
                            )

                    weight_scale, weight = requantize_with_max_scale(  # 使用最大缩放因子重新量化权重
                        weight=weight,
                        weight_scale=weight_scale,
                        logical_widths=layer.logical_widths,
                    )

                # Update layer with new values.
                # 使用新值更新层参数
                layer.weight = Parameter(weight.t(), requires_grad=False)  # 转置权重以匹配计算格式
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.max(), requires_grad=False
                    )

        if self.use_marlin:
            if self.block_quant:
                layer.weight_block_size = self.quant_config.weight_block_size
            prepare_fp8_layer_for_marlin(layer, not self.block_quant)  # 为 Marlin 内核准备 FP8 层参数
            # Activations not quantized for marlin.
            # Marlin 不量化激活，删除输入缩放因子
            del layer.input_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """应用 FP8 量化线性计算。
        
        根据配置选择不同的计算路径：
        - Marlin 后端：使用 Marlin FP8 线性计算
        - MXFP8：使用 MXFP8 W8A8 线性计算
        - 分块量化：使用分块 FP8 W8A8 线性计算（支持 CPU AMX 和 GPU）
        - 非分块量化：使用通用 FP8 线性计算
        
        输入 x 可以是张量（动态量化）或元组 (张量, 缩放因子)（静态量化）。
        """
        if self.use_marlin:
            return torch.ops.sglang.apply_fp8_marlin_linear(  # 使用 Marlin FP8 线性算子
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        if self.use_mxfp8:
            # MXFP8 路径：选择 swizzled 或标准缩放因子
            if get_fp8_gemm_runner_backend().is_flashinfer_cutlass():
                weight_scale = layer.weight_scale_inv_swizzled  # 使用交织后的缩放因子
            else:
                weight_scale = layer.weight_scale_inv  # 使用标准缩放因子
            if isinstance(x, tuple):  # 静态量化：输入包含预计算的缩放因子
                return self.w8a8_mxfp8_linear(
                    input=x[0],
                    weight=layer.weight,
                    weight_scale=weight_scale,
                    input_scale=x[1],
                    bias=bias,
                )
            return self.w8a8_mxfp8_linear(
                input=x,
                weight=layer.weight,
                weight_scale=weight_scale,
                input_scale=None,
                bias=bias,
            )

        if self.block_quant:
            if use_intel_amx_backend(layer):  # CPU AMX 后端
                return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
                    x,
                    layer.weight,
                    layer.weight_scale_inv,
                    self.quant_config.weight_block_size,
                    bias,
                    x.dtype,
                    True,  # is_vnni 是否使用 VNNI 格式
                )

            if isinstance(x, tuple):  # 静态量化：输入包含预计算的缩放因子
                return self.w8a8_block_fp8_linear(
                    input=x[0],
                    weight=layer.weight,
                    block_size=self.quant_config.weight_block_size,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=x[1],
                    bias=bias,
                )

            return self.w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
                bias=bias,
            )

        return apply_fp8_linear(  # 非分块量化的通用 FP8 线性计算
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
        )


class Fp8MoEMethod(FusedMoEMethodBase):
    """FP8 MoE（混合专家）量化方法。
    
    支持加载具有静态权重缩放因子和动态/静态激活缩放因子的 FP8 检查点。
    也支持加载量化的 FP16/BF16 模型检查点，配合动态激活缩放。
    权重缩放因子将在模型权重加载后初始化。

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: 量化配置对象
    """

    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.use_mxfp8 = getattr(self.quant_config, "use_mxfp8", False)  # 是否使用 MXFP8
        self.block_quant = (  # 是否使用分块量化
            self.use_mxfp8 or self.quant_config.weight_block_size is not None
        )
        self.is_fp4_expert = self.quant_config.is_fp4_experts  # 是否为 FP4 专家权重
        self.with_bias = False  # 是否有偏置
        if get_moe_runner_backend().is_cutlass():  # CUTLASS FP8 MoE 的硬件要求校验
            assert (
                cutlass_fp8_supported()
            ), "cutlass_fp8 MoE requires CUDA 12.0+ with SM90 or CUDA 12.4+ with SM89"
            assert self.block_quant, "cutlass_fp8 MoE requires block quantization"
            assert (
                is_sm100_supported() or is_sm90_supported() or is_sm120_supported()
            ), "cutlass_fp8 MoE requires SM90, SM100, or SM120 GPUs"

    @staticmethod
    def is_deepgemm_moe_runner_backend_enabled() -> bool:
        """检查 MoE 是否会实际使用 DeepGEMM 运行器进行 FP8 计算"""
        """Check if MoE will actually use DeepGEMM runner for FP8."""
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_deep_gemm():
            return True
        if moe_runner_backend.is_auto():
            return deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and (
                get_moe_a2a_backend().is_deepep()
                or get_moe_a2a_backend().is_mooncake()
                or get_moe_a2a_backend().is_nixl()
            )
        return False

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        """为 MoE 层创建 FP8 量化所需的权重和缩放因子参数。
        
        创建以下参数：
        - w13_weight / w2_weight：门控和下投影权重
        - w13_weight_scale_inv / w2_weight_scale_inv：分块量化的权重缩放因子
        - w13_weight_scale / w2_weight_scale：非分块量化的权重缩放因子
        - w13_input_scale / w2_input_scale：静态激活缩放因子
        - w13_weight_bias / w2_weight_bias：可选偏置
        """
        self.with_bias = with_bias
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.uint32 if _use_hip_int4 else torch.float8_e4m3fn  # INT4 使用 uint32 打包，FP8 使用 e4m3fn
        tp_size = get_tensor_model_parallel_world_size()  # 张量并行度

        w13_up_dim, w2_up_dim, weight_padded = get_moe_weight_sizes(  # 获取 MoE 权重尺寸（考虑 AITER 对齐和填充）
            intermediate_size_per_partition,
            is_aiter_moe=_use_aiter,
            is_concat=True,
            is_packed=False,
        )

        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )

            padding_size = get_moe_padding_size(_use_aiter)
            if not (_use_aiter and padding_size == block_n == block_k):
                # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
                # Required by column parallel or enabling merged weights
                if intermediate_size_per_partition % block_n != 0:
                    raise ValueError(
                        f"The output_size of gate's and up's weight = "
                        f"{intermediate_size_per_partition} is not divisible by "
                        f"weight quantization block_n = {block_n}."
                    )
                if tp_size > 1:
                    # Required by row parallel
                    if intermediate_size_per_partition % block_k != 0:
                        raise ValueError(
                            f"The input_size of down's weight = "
                            f"{intermediate_size_per_partition} is not divisible by "
                            f"weight quantization block_k = {block_k}."
                        )

        # WEIGHTS
        # 创建专家权重参数
        if self.is_fp4_expert:  # FP4 专家权重：每个元素 4 位，用 int8 打包（2 个 FP4 元素共享一个 int8）
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // 2,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // 2,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
        elif _is_hip and _use_hip_int4:
            # INT4 MoE weight - INT32 packed
            # INT4 MoE 权重 - 以 INT32 打包（8 个 INT4 元素共享一个 int32）
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // 8,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // 8,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
        else:
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    w13_up_dim,
                    hidden_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    w2_up_dim,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )

        extra_weight_attrs.update(
            {"weight_padded": weight_padded},
        )

        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIAS (optional, e.g. GPT-OSS)
        # 偏置（可选，例如 GPT-OSS 模型）
        if self.with_bias:
            w13_up_dim = (
                2 * intermediate_size_per_partition
                if layer.moe_runner_config.is_gated
                else intermediate_size_per_partition
            )
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

        # WEIGHT_SCALES
        # 创建权重缩放因子参数
        if self.is_fp4_expert:  # FP4 专家缩放因子
            fp4_block_k = 32  # FP4 的分块 K 维度为 32
            fp4_scale_dtype = torch.float8_e8m0fnu if _use_aiter else torch.float32  # AITER 使用 E8M0 格式，否则使用 float32
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // fp4_block_k,
                    dtype=fp4_scale_dtype,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // fp4_block_k,
                    dtype=fp4_scale_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        elif self.block_quant:
            scale_dtype = torch.uint8 if self.use_mxfp8 else torch.float32
            scale_init = torch.zeros if scale_dtype == torch.uint8 else torch.ones
            w13_weight_scale = torch.nn.Parameter(
                scale_init(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                scale_init(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            # w13_weight and w2_weight are always requanted together
            # w13_weight 和 w2_weight 总是一起被重量化
            w13_weight_scale.format_ue8m0 = self.use_mxfp8  # 标记缩放因子格式
            w2_weight_scale.format_ue8m0 = self.use_mxfp8
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"
            if get_moe_runner_backend().is_cutlass():
                self._ensure_cutlass_buffers_initialized(layer)  # 初始化 CUTLASS 所需的缓冲区

        else:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            # 为 w1 和 w3 分别分配 2 个缩放因子，权重加载后合并为单个缩放因子
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            if _is_hip:  # _use_aiter: TODO: add check back after triton kernel
                # ROCm - using column scaling, duplicate scaling numbers in case per tensor scaling
                # ROCm 平台 - 使用列缩放，复制缩放数值以兼容逐张量缩放
                w13_weight_scale1 = torch.nn.Parameter(
                    torch.ones(
                        num_experts,
                        2 * intermediate_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                w2_weight_scale1 = torch.nn.Parameter(
                    torch.ones(num_experts, hidden_size, dtype=torch.float32),
                    requires_grad=False,
                )
                layer.register_parameter("w13_weight_scale1", w13_weight_scale1)
                layer.register_parameter("w2_weight_scale1", w2_weight_scale1)

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        # 添加使用的量化方法（逐张量/分组/逐通道），以确保权重缩放因子正确加载
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        # 如果加载 FP8 检查点，传递 weight_loader；
        # 如果加载 FP16 检查点，则不传递（将在 process_weights_after_loading 中量化）
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

            if _is_hip and _use_hip_int4:
                extra_weight_attrs.update(
                    {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
                )
                set_weight_attrs(w13_weight_scale1, extra_weight_attrs)
                set_weight_attrs(w2_weight_scale1, extra_weight_attrs)

        # INPUT_SCALES
        # 输入激活缩放因子（仅静态量化方案需要）
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading_block_quant(self, layer: Module) -> None:
        """分块量化模式下 MoE 权重的后处理。
        
        根据平台和配置进行：
        - AMD AITER FP4 专家：权重填充对齐和重排
        - ROCm FNUZ：归一化为 e4m3fnuz 格式
        - AITER：权重预重排
        - CPU AMX：AMX 权重处理
        - MXFP8：MXFP8 MoE 权重处理
        - DeepGEMM：重量化为 UE8M0 格式
        """
        # AMD FP4 experts: use aiter's native MXFP4 MoE path
        # AMD FP4 专家：使用 aiter 的原生 MXFP4 MoE 路径
        if _use_aiter and self.is_fp4_expert:
            fp4_weight_dtype = _require_fp4_dtype()

            # CK FP4 MoE kernel requires K_packed divisible by 128
            # (i.e., K_logical divisible by 256).
            # Pad intermediate_size_per_partition if needed.
            # CK FP4 MoE 内核要求 K_packed 能被 128 整除（即 K_logical 能被 256 整除）。
            # 如果需要，填充 intermediate_size_per_partition。
            fp4_k_align = 256  # FP4 对齐要求为 256
            E, w13_N, w13_K_packed = layer.w13_weight.shape
            _, w2_N, w2_K_packed = layer.w2_weight.shape
            inter_per_part = w13_N // 2
            padded_inter = (
                (inter_per_part + fp4_k_align - 1) // fp4_k_align * fp4_k_align
            )
            if padded_inter != inter_per_part:
                pad_amount = padded_inter - inter_per_part
                fp4_block_k = 32

                # Pad w13_weight: (E, 2*inter, K_packed) → (E, 2*padded, K_packed)
                # 填充 w13_weight: (E, 2*inter, K_packed) → (E, 2*padded, K_packed)
                old_w13 = layer.w13_weight.data
                new_w13 = torch.zeros(
                    E,
                    2 * padded_inter,
                    w13_K_packed,
                    dtype=old_w13.dtype,
                    device=old_w13.device,
                )
                new_w13[:, :inter_per_part, :] = old_w13[:, :inter_per_part, :]
                new_w13[:, padded_inter : padded_inter + inter_per_part, :] = old_w13[
                    :, inter_per_part:, :
                ]
                layer.w13_weight = torch.nn.Parameter(new_w13, requires_grad=False)

                # Pad w2_weight: (E, N, inter_packed) → (E, N, padded_packed)
                # 填充 w2_weight: (E, N, inter_packed) → (E, N, padded_packed)
                old_w2 = layer.w2_weight.data
                new_w2 = torch.zeros(
                    E,
                    w2_N,
                    padded_inter // 2,
                    dtype=old_w2.dtype,
                    device=old_w2.device,
                )
                new_w2[:, :, :w2_K_packed] = old_w2
                layer.w2_weight = torch.nn.Parameter(new_w2, requires_grad=False)

                # Pad w13 scale: (E, 2*inter, K/block_k) → (E, 2*padded, K/block_k)
                # 填充 w13 缩放因子: (E, 2*inter, K/block_k) → (E, 2*padded, K/block_k)
                old_s13 = layer.w13_weight_scale_inv.data
                _, _, s13_K = old_s13.shape
                new_s13 = torch.zeros(
                    E,
                    2 * padded_inter,
                    s13_K,
                    dtype=old_s13.dtype,
                    device=old_s13.device,
                )
                new_s13[:, :inter_per_part, :] = old_s13[:, :inter_per_part, :]
                new_s13[:, padded_inter : padded_inter + inter_per_part, :] = old_s13[
                    :, inter_per_part:, :
                ]
                layer.w13_weight_scale_inv = torch.nn.Parameter(
                    new_s13, requires_grad=False
                )

                # Pad w2 scale: (E, N, inter/block_k) → (E, N, padded/block_k)
                # 填充 w2 缩放因子: (E, N, inter/block_k) → (E, N, padded/block_k)
                old_s2 = layer.w2_weight_scale_inv.data
                new_s2 = torch.zeros(
                    E,
                    w2_N,
                    padded_inter // fp4_block_k,
                    dtype=old_s2.dtype,
                    device=old_s2.device,
                )
                new_s2[:, :, : old_s2.shape[2]] = old_s2
                layer.w2_weight_scale_inv = torch.nn.Parameter(
                    new_s2, requires_grad=False
                )

            for scale_name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
                scale = getattr(layer, scale_name)
                num_experts, num_rows, _ = scale.shape
                # a8w4: aiter flydsl scale layout
                # a8w4: aiter flydsl 缩放因子布局
                is_w13_scale = scale_name == "w13_weight_scale_inv"
                scale.data = shuffle_scale_a16w4(
                    scale.view(num_experts * num_rows, -1), num_experts, is_w13_scale
                )

            layer.w13_weight.data = layer.w13_weight.data.view(fp4_weight_dtype)  # 将 int8 视图转为 FP4 数据类型
            layer.w2_weight.data = layer.w2_weight.data.view(fp4_weight_dtype)

            is_shuffled = _is_shuffle_moe_mxfp4
            if is_shuffled:
                # a8w4: aiter flydsl weight layout
                # a8w4: aiter flydsl 权重布局
                layer.w13_weight.data = shuffle_weight_a16w4(  # 对 w13 权重进行 A16W4 重排
                    layer.w13_weight.contiguous(), 16, True
                )
                layer.w2_weight.data = shuffle_weight_a16w4(
                    layer.w2_weight.contiguous(), 16, False
                )
            layer.w13_weight.is_shuffled = is_shuffled
            layer.w2_weight.is_shuffled = is_shuffled
            return

        # If ROCm, normalize the weights and scales to e4m3fnuz
        # 如果是 ROCm 平台，将权重和缩放因子归一化为 e4m3fnuz 格式
            # activation_scheme: dynamic
            w13_weight, w13_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.w13_weight,
                weight_scale=layer.w13_weight_scale_inv,
                input_scale=None,
            )
            w2_weight, w2_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.w2_weight,
                weight_scale=layer.w2_weight_scale_inv,
                input_scale=None,
            )
            # Reset the parameter
            # 重置参数
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale_inv = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            layer.w13_input_scale = None
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale_inv = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            layer.w2_input_scale = None
            if _use_aiter:
                layer.w13_weight.data = shuffle_weight(
                    layer.w13_weight.contiguous(), (16, 16)
                )
                layer.w2_weight.data = shuffle_weight(
                    layer.w2_weight.contiguous(), (16, 16)
                )
        elif _use_aiter:
            # Pre-shuffle weights
            # 预重排权重（AITER 加速所需的数据重排）
            t = shuffle_weight(layer.w13_weight, (16, 16))
            layer.w13_weight.copy_(t)
            del t
            t = shuffle_weight(layer.w2_weight, (16, 16))
            layer.w2_weight.copy_(t)
            del t
        elif _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "Fp8MoEMethod on CPU requires that CPU has AMX support"  # CPU 上的 FP8 MoE 需要 AMX 支持
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])  # 使用 AMX 处理 MoE 权重
        elif self.use_mxfp8:
            self._process_mxfp8_moe_weights(  # 处理 MXFP8 MoE 权重
                layer, quantize=not self.quant_config.is_checkpoint_fp8_serialized
            )
        else:
            # For fp8 moe run with deepgemm, the expert weights and scales need be requantized to ue8m0
            # 对于使用 DeepGEMM 运行的 FP8 MoE，专家权重和缩放因子需要重量化为 UE8M0 格式
            from sglang.srt.layers import deep_gemm_wrapper
            from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
            from sglang.srt.model_loader.utils import (
                should_deepgemm_weight_requant_ue8m0,
            )

            # Check if MoE will actually use DeepGEMM runner
            # 检查 MoE 是否会实际使用 DeepGEMM 运行器
            will_use_deepgemm = self.is_deepgemm_moe_runner_backend_enabled()

            if self.is_fp4_expert:
                if get_moe_runner_backend().is_marlin():
                    layer.w13_weight.data = layer.w13_weight.data.view(torch.int8)
                    layer.w2_weight.data = layer.w2_weight.data.view(torch.int8)
                    return

                fp4_weight_dtype = _require_fp4_dtype() if _use_aiter else torch.int8
                layer.w13_weight.data = layer.w13_weight.data.view(fp4_weight_dtype)
                layer.w2_weight.data = layer.w2_weight.data.view(fp4_weight_dtype)

                if get_moe_a2a_backend().is_megamoe():
                    from sglang.srt.layers.moe.mega_moe import (
                        build_mega_moe_experts_weights,
                    )

                    build_mega_moe_experts_weights(layer)
                    return

                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 and will_use_deepgemm:
                    # DeepGEMM UE8M0 缩放因子格式转换
                    from deep_gemm import transform_sf_into_required_layout

                    for scale_param, weight_param in [
                        (layer.w13_weight_scale_inv, layer.w13_weight),
                        (layer.w2_weight_scale_inv, layer.w2_weight),
                    ]:
                        num_experts, n, _ = scale_param.data.shape
                        k = weight_param.shape[2] * 2
                        scale_param.data = transform_sf_into_required_layout(
                            scale_param.data,
                            mn=n,
                            k=k,
                            recipe=(1, 32),
                            num_groups=num_experts,
                            disable_ue8m0_cast=False,
                        )
                    layer.w13_weight_scale_inv.format_ue8m0 = True
                    layer.w2_weight_scale_inv.format_ue8m0 = True

            if (
                not self.is_fp4_expert
                and should_deepgemm_weight_requant_ue8m0(
                    weight_block_size=getattr(
                        self.quant_config, "weight_block_size", None
                    ),
                )
                and will_use_deepgemm
                and not layer.w13_weight_scale_inv.format_ue8m0
            ):
                assert isinstance(
                    layer, DeepEPMoE
                ), "DeepGemm MoE is only supported with DeepEPMoE"
                weight_block_size = self.quant_config.weight_block_size
                requant_weight_ue8m0_inplace(
                    layer.w13_weight, layer.w13_weight_scale_inv, weight_block_size
                )
                requant_weight_ue8m0_inplace(
                    layer.w2_weight, layer.w2_weight_scale_inv, weight_block_size
                )
                layer.w13_weight_scale_inv.format_ue8m0 = True
                layer.w2_weight_scale_inv.format_ue8m0 = True

    def _process_mxfp8_moe_weights(self, layer: Module, quantize: bool = True) -> None:
        """处理 MXFP8 MoE 权重，包括量化和缩放因子重排。
        
        根据 MoE 运行器后端选择不同的量化路径：
        - CUTLASS：使用 SM100 ES 内核进行量化和重排
        - FlashInfer TRT-LLM：使用 FlashInfer 进行量化
        - Triton：使用 Triton 内核进行量化和重排
        如果 quantize=False（已序列化的检查点），仅对缩放因子进行重排。
        """

        if not (_is_cuda and is_sm100_supported()):
            raise RuntimeError("MXFP8 MoE quantization requires SM100.")  # MXFP8 MoE 量化需要 SM100（Blackwell）GPU

        def _quantize_and_swizzle_with_cutlass_es_kernel(weight: torch.Tensor):
            """使用 CUTLASS ES 内核在 SM100 上进行 MXFP8 分块量化和重排"""
            from sgl_kernel import es_sm100_mxfp8_blockscaled_grouped_quant

            weight = weight.contiguous()
            num_experts, m, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"

            weight_flat = weight.view(-1, k).contiguous()
            problem_sizes = torch.empty(
                (num_experts, 3), dtype=torch.int32, device=weight.device
            )
            problem_sizes[:, 0] = m
            problem_sizes[:, 1] = 0
            problem_sizes[:, 2] = k
            expert_offsets = torch.arange(
                0, num_experts * m, m, dtype=torch.int32, device=weight.device
            )
            aligned_m = ((m + 127) // 128) * 128
            blockscale_offsets = torch.arange(
                0,
                num_experts * aligned_m,
                aligned_m,
                dtype=torch.int32,
                device=weight.device,
            )
            qweight = torch.empty_like(weight_flat, dtype=torch.float8_e4m3fn)
            scale = torch.empty(
                (num_experts * aligned_m, k // 32),
                dtype=torch.uint8,
                device=weight.device,
            )
            es_sm100_mxfp8_blockscaled_grouped_quant(
                weight_flat,
                problem_sizes,
                expert_offsets,
                blockscale_offsets,
                qweight,
                scale,
            )
            qweight = qweight.view_as(weight)
            scale = scale.view(num_experts, aligned_m, k // 32)
            if aligned_m != m:
                scale = scale[:, :m, :]
            return qweight, scale

        def _swizzle_mxfp8_sf(scale, num_warps):
            """使用 Triton 内核对 MXFP8 缩放因子进行重排（swizzle）"""
            from triton_kernels.tensor import convert_layout, wrap_torch_tensor
            from triton_kernels.tensor_details import layout

            scale_layout, scale_layout_opts = (
                layout.make_default_matmul_mxfp4_w_scale_layout(
                    mx_axis=1, num_warps=num_warps
                )
            )
            scale = scale.transpose(-2, -1)
            scale = convert_layout(
                wrap_torch_tensor(scale), scale_layout, **scale_layout_opts
            )
            return scale

        def _swizzle_with_triton_kernel(
            weight_shape: tuple[int, int, int], scale: torch.Tensor
        ):
            """使用 Triton 内核对已有的 MXFP8 缩放因子进行重排"""
            num_experts, m, k = weight_shape
            aligned_m = ((m + 127) // 128) * 128
            scale = scale.view(num_experts, aligned_m, k // 32)
            num_warps = 8
            scale = _swizzle_mxfp8_sf(scale, num_warps)
            scale = scale.data.view(num_experts, aligned_m, k // 32)
            return scale

        def _quantize_and_swizzle_with_triton_kernel(weight: torch.Tensor):
            """使用 Triton 内核进行 MXFP8 量化和缩放因子重排"""

            weight = weight.contiguous()
            _, _, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"

            weight_flat = weight.view(-1, k).contiguous()
            qweight, scale = mxfp8_group_quantize(weight_flat)
            qweight = qweight.view_as(weight)
            scale = _swizzle_with_triton_kernel(weight.shape, scale)
            return qweight, scale

        def _quantize_with_flashinfer_trtllm(weight: torch.Tensor):
            """使用 FlashInfer 进行 MXFP8 量化（标准缩放布局，不重排）"""
            weight = weight.contiguous()
            num_experts, m, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"
            from flashinfer import mxfp8_quantize

            weight_flat = weight.view(-1, k).contiguous()
            qweight, scale = mxfp8_quantize(weight_flat, False)
            scale_u8 = (
                scale.view(torch.uint8).contiguous().view(num_experts, m, k // 32)
            )
            return qweight.view_as(weight), scale_u8

        if quantize:
            # 需要在线量化权重
            if get_moe_runner_backend().is_cutlass():  # CUTLASS 路径：使用 ES 内核量化和重排
                w13_q, w13_s = _quantize_and_swizzle_with_cutlass_es_kernel(
                    layer.w13_weight.data
                )
                w2_q, w2_s = _quantize_and_swizzle_with_cutlass_es_kernel(
                    layer.w2_weight.data
                )
            elif (
                get_moe_runner_backend().is_flashinfer_trtllm()
                or get_moe_runner_backend().is_flashinfer_trtllm_routed()
            ):
                # Match FlashInfer TRT-LLM MoE test contracts:
                # 1) quantize in canonical (non-swizzled) scale layout, and
                # 2) do row/layout shuffling in align_mxfp8_moe_weights_for_flashinfer_trtllm.
                # 匹配 FlashInfer TRT-LLM MoE 测试约定：
                # 1) 以标准（非重排）缩放布局进行量化，然后
                # 2) 在 align_mxfp8_moe_weights_for_flashinfer_trtllm 中进行行/布局重排
                w13_q, w13_s = _quantize_with_flashinfer_trtllm(layer.w13_weight.data)
                w2_q, w2_s = _quantize_with_flashinfer_trtllm(layer.w2_weight.data)
            else:
                # Triton 路径：量化和重排
                w13_q, w13_s = _quantize_and_swizzle_with_triton_kernel(
                    layer.w13_weight.data
                )
                w2_q, w2_s = _quantize_and_swizzle_with_triton_kernel(
                    layer.w2_weight.data
                )
        else:
            # 不需要量化，仅对缩放因子进行重排（已序列化的 FP8 检查点）
            if (
                get_moe_runner_backend().is_flashinfer_trtllm()
                or get_moe_runner_backend().is_flashinfer_trtllm_routed()
            ):
                w13_q = layer.w13_weight.data
                w2_q = layer.w2_weight.data
                w13_s = layer.w13_weight_scale_inv.data
                w2_s = layer.w2_weight_scale_inv.data
            else:
                w13_q = layer.w13_weight.data
                w2_q = layer.w2_weight.data
                w13_s = _swizzle_with_triton_kernel(
                    layer.w13_weight.data.shape, layer.w13_weight_scale_inv.data
                )
                w2_s = _swizzle_with_triton_kernel(
                    layer.w2_weight.data.shape, layer.w2_weight_scale_inv.data
                )

        # Keep parameter objects to preserve weight_loader attrs for hot reload.
        # Prefer in-place copy; rebind only when shape/dtype changes (online quantize).
        # 保留参数对象以保持 weight_loader 属性，支持热重载。
        # 优先就地复制；仅当形状/数据类型变化时重新绑定（在线量化）。
        def _copy_or_rebind(param: Parameter, new_value: torch.Tensor) -> None:
            if (
                param.data.shape == new_value.shape
                and param.data.dtype == new_value.dtype
            ):
                param.data.copy_(new_value)
            else:
                param.data = new_value

        _copy_or_rebind(layer.w13_weight, w13_q)
        _copy_or_rebind(layer.w2_weight, w2_q)
        _copy_or_rebind(layer.w13_weight_scale_inv, w13_s)
        _copy_or_rebind(layer.w2_weight_scale_inv, w2_s)
        layer.w13_weight.requires_grad_(False)
        layer.w2_weight.requires_grad_(False)
        layer.w13_weight_scale_inv.requires_grad_(False)
        layer.w2_weight_scale_inv.requires_grad_(False)
        layer.w13_weight_scale_inv.format_ue8m0 = True
        layer.w2_weight_scale_inv.format_ue8m0 = True
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        if (
            get_moe_runner_backend().is_flashinfer_trtllm()
            or get_moe_runner_backend().is_flashinfer_trtllm_routed()
        ):
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                align_mxfp8_moe_weights_for_flashinfer_trtllm,
            )

            align_mxfp8_moe_weights_for_flashinfer_trtllm(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        """MoE 权重加载后的处理。
        
        根据不同情况：
        - ROCm INT4：处理 INT4 权重重排和缩放因子
        - 分块量化：调用 process_weights_after_loading_block_quant
        - FP16/BF16 检查点：在线量化为 FP8
        - FP8 检查点：合并缩放因子、处理静态激活缩放
        """
        if _is_hip and _use_hip_int4:  # ROCm INT4 权重处理
            self.process_weights_hip_int4(layer)

        elif self.block_quant:
            # Block quant doesn't need to process weights after loading
            # 分块量化：调用分块量化的权重后处理方法
            self.process_weights_after_loading_block_quant(layer)

        # If checkpoint is fp16 or bfloat16, quantize in place.
        # 如果检查点是 fp16 或 bfloat16，就地量化为 FP8
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            # If ROCm, fp8_dtype will be float8_e4m3fnuz (MI300x HW)
            # 如果是 ROCm 平台，fp8_dtype 将是 float8_e4m3fnuz（MI300x 硬件格式）
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)  # 创建 FP8 权重缓冲区
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            # 重新初始化 w13_scale，因为直接量化合并的 w13 权重并生成单个缩放因子
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_local_experts,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )
            for expert in range(layer.num_local_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w13_weight.data[expert, :, :])  # 逐专家量化 w13 权重
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w2_weight.data[expert, :, :])  # 逐专家量化 w2 权重
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

            if _is_hip:
                self.process_weights_hip_scale_padding(layer)  # ROCm 平台：处理缩放因子填充

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        # 如果检查点是 FP8 的，需要处理 MoE 内核要求每个专家的 w13 有单一激活缩放和权重缩放
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            # FP8 MoE 内核需要单一激活缩放因子。
            # 如果各专家的缩放因子不同，取最大值。
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                    layer.w2_input_scale
                ):
                    print_warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer. "
                        # 发现 FP8 MoE 层的输入缩放因子不一致，使用各专家的最大值
                    )
                layer.w13_input_scale = torch.nn.Parameter(
                    layer.w13_input_scale.max(), requires_grad=False
                )
                layer.w2_input_scale = torch.nn.Parameter(
                    layer.w2_input_scale.max(), requires_grad=False
                )

            # If ROCm, normalize the weights and scales to e4m3fnuz
            # 如果是 ROCm 平台，将权重和缩放因子归一化为 e4m3fnuz 格式
            if _is_fp8_fnuz:
                # Normalize the weights and scales
                w13_weight, w13_weight_scale, w13_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                    )
                )
                w2_weight, w2_weight_scale, w2_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
                    )
                )
                # Reset the parameter
                # 重置参数
                layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False
                )
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False
                    )
                layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_weight_scale, requires_grad=False
                )
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False
                    )
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            # FP8 MoE 内核需要每个专家的 w13 有单一权重缩放因子。
            # 取最大值，然后反量化并重新量化每个专家。
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values  # 取每个专家 w13 权重缩放因子的最大值
            for expert_id in range(layer.num_local_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(  # 反量化每个分片的权重
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    (
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        _,
                    ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])  # 使用最大缩放因子重新量化
                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

            if _is_hip:
                self.process_weights_hip_scale_padding(layer)  # ROCm 平台：处理缩放因子填充

            # Align FP8 weights to FlashInfer per-tensor kernel layout if enabled
            # 如果启用了 FlashInfer 逐张量内核，将 FP8 权重对齐到其布局
            if (
                get_moe_runner_backend().is_flashinfer_trtllm()
                or get_moe_runner_backend().is_flashinfer_trtllm_routed()
            ):
                from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                    align_fp8_moe_weights_for_flashinfer_trtllm,
                )

                align_fp8_moe_weights_for_flashinfer_trtllm(layer)

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"weight_dtype": layer.w13_weight.dtype})  # 设置分发器的量化配置

    def process_weights_hip_int4(self, layer: Module):
        """处理 ROCm 平台 INT4 权重的重排和缩放因子合并。
        
        将 INT4 权重进行重排以适配 AITER 内核，
        并合并 w13 的两个缩放因子为单一缩放因子。
        """
        # TODO: _use_aiter: add after triton kernel added
        # INT4-FP8 (INT4 MoE Weight, FP8 Compute)
        # Weight Permutation
        # INT4-FP8（INT4 MoE 权重，FP8 计算）
        # 权重重排：将 INT4 权重按 (16,16) 分块进行重排
        layer.w13_weight = torch.nn.Parameter(
            shuffle_weight(layer.w13_weight.data, (16, 16)),
            requires_grad=False,
        )
        torch.cuda.empty_cache()  # 释放 GPU 缓存
        layer.w2_weight = torch.nn.Parameter(
            shuffle_weight(layer.w2_weight.data, (16, 16)),
            requires_grad=False,
        )
        torch.cuda.empty_cache()

        # INT4-FP8 : offset INT4 w13_weight_scale1 to single w13_weight_scale
        # Fp8 moe kernel needs single fp8 w13_weight_scale for w13 per expert.
        # We won't do requant each expert's fp8 weight (not direct available),
        # instead we adjust half of INT4 w13_weight_scale1 numbers
        # INT4-FP8：将 INT4 的 w13_weight_scale1 偏移为单一 w13_weight_scale
        # FP8 MoE 内核需要每个专家的 w13 有单一 FP8 缩放因子。
        # 不对每个专家的 FP8 权重进行重量化（直接不可用），
        # 而是调整 INT4 w13_weight_scale1 的一半数值
        assert layer.w13_weight_scale is not None
        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = layer.w13_weight_scale.max(dim=1).values  # 取每个专家 w13 缩放因子的最大值
        for expert_id in range(layer.num_local_experts):
            start = 0
            max_w13_scale_fp8 = max_w13_scales[expert_id]
            for shard_id in range(2):
                if layer.w13_weight_scale[expert_id][shard_id] != max_w13_scale_fp8:
                    int4_rescale = (  # 计算 INT4 重缩放比例
                        layer.w13_weight_scale[expert_id][shard_id] / max_w13_scale_fp8
                    )
                    layer.w13_weight_scale1[expert_id][
                        start : start + shard_size
                    ] *= int4_rescale  # 调整列缩放因子
                start += shard_size

        layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)

        # special hack to asm_moe, which takes (weight_scale1 * weight_scale) as post GEMM scaling
        # optimal design - shall apply per-column weight_scale1 before GEMM, and weight_scale post
        # asm_moe 的特殊处理：将 (weight_scale1 * weight_scale) 作为 GEMM 后缩放
        # 最优设计 - 应在 GEMM 前应用逐列 weight_scale1，在 GEMM 后应用 weight_scale
        for expert_id in range(layer.num_local_experts):
            layer.w13_weight_scale1[expert_id] *= max_w13_scales[expert_id]
            layer.w2_weight_scale1[expert_id] *= layer.w2_weight_scale[expert_id]

    def process_weights_hip_scale_padding(self, layer: Module):
        """ROCm 平台的 MoE 权重缩放因子填充处理。
        
        对于 AITER 后端：对权重进行重排并合并列缩放因子。
        对于普通 ROCm 后端：如果设置了 SGLANG_MOE_PADDING，则对权重进行填充以减少内存通道竞争。
        """
        padding_size = get_moe_padding_size(_use_aiter)
        if _use_aiter:
            layer.w13_weight = torch.nn.Parameter(
                shuffle_weight(layer.w13_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                shuffle_weight(layer.w2_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()

            # ROCm (_use_aiter): using column-wise scaling
            # ROCm (_use_aiter)：使用列缩放，将全局缩放因子合并到列缩放因子中
            layer.w13_weight_scale1 *= layer.w13_weight_scale.unsqueeze(-1)
            layer.w2_weight_scale1 *= layer.w2_weight_scale.unsqueeze(-1)
        elif get_bool_env_var("SGLANG_MOE_PADDING"):
            # If ROCm, apply weight padding (min. Mem channel contention) only if set
            # 如果是 ROCm 平台且设置了 SGLANG_MOE_PADDING，则对权重进行填充以减少内存通道竞争
            layer.w13_weight = torch.nn.Parameter(
                F.pad(layer.w13_weight.data, (0, padding_size), "constant", 0),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                F.pad(layer.w2_weight.data, (0, padding_size), "constant", 0),
                requires_grad=False,
            )
            torch.cuda.empty_cache()

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        """创建 MoE 运行器实例。
        
        根据后端配置选择合适的 MoE 运行器：
        - AUTO：自动选择（DeepGEMM > AITER > Triton）
        - DeepGEMM/Triton/AITER/FlashInfer TRT-LLM：创建对应的 MoeRunner
        """
        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()

        if moe_runner_backend.is_auto():  # 自动选择最佳后端
            if self.is_deepgemm_moe_runner_backend_enabled():  # 优先使用 DeepGEMM
                moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
            elif (
                _is_hip
                and (_use_aiter or _use_hip_int4)
                and get_moe_a2a_backend().supports_aiter()
            ):
                moe_runner_backend = MoeRunnerBackend.AITER  # ROCm + AITER/INT4 -> AITER 后端
            else:
                moe_runner_backend = MoeRunnerBackend.TRITON  # 默认使用 Triton 后端

        if (
            moe_runner_backend.is_deep_gemm()
            or moe_runner_backend.is_triton()
            or moe_runner_backend.is_aiter()
            or moe_runner_backend.is_flashinfer_trtllm()
            or moe_runner_backend.is_flashinfer_trtllm_routed()
        ):
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
        else:
            # TODO(cwan): refactor other backends
            # TODO(cwan): 重构其他后端
            pass

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        """获取 Triton MoE 运行器所需的量化信息"""
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            b13=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
            use_fp8_w8a8=True,
            w13_scale=(
                layer.w13_weight_scale_inv
                if self.block_quant
                else layer.w13_weight_scale
            ),
            w2_scale=(
                layer.w2_weight_scale_inv if self.block_quant else layer.w2_weight_scale
            ),
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=self.quant_config.weight_block_size,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        """应用 FP8 量化 MoE 计算。
        
        根据不同后端选择计算路径：
        - CPU AMX：使用 CPU 融合专家计算
        - AITER (ROCm)：使用 AMD AITER 加速
        - CUTLASS：使用 NVIDIA CUTLASS FP8 分组 GEMM
        - DeepGEMM：使用 DeepGEMM FP8 MoE
        - FlashInfer TRT-LLM：使用 FlashInfer TRT-LLM MoE
        - Triton：使用 Triton MoE
        """

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        moe_runner_config = self.moe_runner_config

        if use_intel_amx_backend(layer):  # Intel AMX CPU 后端
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = dispatch_output.topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )

            output = torch.ops.sgl_kernel.fused_experts_cpu(  # 使用 CPU 融合专家算子
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace See [Note] inplace should be False in fused_experts.
                # inplace: 参见 [Note] fused_experts 中 inplace 应为 False
                CPUQuantMethod.FP8_W8A16,
                layer.w13_weight_scale_inv,  # w1_scale
                layer.w2_weight_scale_inv,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                self.quant_config.weight_block_size,  # block_size
                None,  # w1 bias
                None,  # w3 bias
                None,  # alpha
                None,  # limit
                True,  # is_vnni 是否使用 VNNI 格式
            )
            return StandardCombineInput(hidden_states=output)

        if (
            _is_hip
            and getattr(self, "runner", None) is not None
            and self.runner.runner_backend.is_aiter()
        ):  # AITER (ROCm) 后端
            quant_info = self.maybe_get_hip_aiter_quant_info(
                layer,
                moe_runner_config.no_combine,
            )
            if quant_info is not None:
                return self.runner.run(dispatch_output, quant_info)

        if get_moe_runner_backend().is_cutlass():  # CUTLASS FP8 分组 GEMM 后端
            from sglang.srt.layers.moe.cutlass_moe import cutlass_fused_experts_fp8

            with use_symmetric_memory(  # 使用对称内存（如果分配对称）
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                symm_output = torch.empty_like(x)

            topk_weights, topk_ids, _ = dispatch_output.topk_output
            use_mxfp8 = getattr(self.quant_config, "use_mxfp8", False)
            output = cutlass_fused_experts_fp8(
                x,
                layer.w13_weight.transpose(1, 2),
                layer.w2_weight.transpose(1, 2),
                layer.w13_weight_scale_inv.transpose(1, 2),
                layer.w2_weight_scale_inv.transpose(1, 2),
                topk_weights,
                topk_ids,
                self.ab_strides1,
                self.c_strides1,
                self.ab_strides2,
                self.c_strides2,
                self.workspace,
                self.a_ptr,
                self.b_ptr,
                self.out_ptr,
                self.a_scales_ptr,
                self.b_scales_ptr,
                self.expert_offsets,
                self.problem_sizes1,
                self.problem_sizes2,
                use_fp8_blockscale=True,
                use_mxfp8=use_mxfp8,
                output=symm_output,
                enable_es=(use_mxfp8, use_mxfp8),
            )
            return StandardCombineInput(hidden_states=output)

        if self.runner.runner_backend.is_deep_gemm():  # DeepGEMM 后端

            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight

            if self.block_quant:
                block_shape = self.quant_config.weight_block_size
                w13_scale = layer.w13_weight_scale_inv  # 分块量化使用逆缩放因子
                w2_scale = layer.w2_weight_scale_inv
            else:
                # Convert per-tensor quant to per-block quant by repeating scales for forward_deepgemm
                # 将逐张量缩放转换为逐块缩放，通过重复缩放因子适配 DeepGEMM 前向计算
                scale_block_size = 128  # 逐张量转逐块的分块大小
                block_shape = [scale_block_size, scale_block_size]
                w13_scale_n = (w13_weight.shape[1] - 1) // scale_block_size + 1  # w13 缩放因子 N 维度
                w13_scale_k = (w13_weight.shape[2] - 1) // scale_block_size + 1  # w13 缩放因子 K 维度
                w13_scale = (  # 通过重复插值将逐张量缩放扩展为逐块缩放
                    layer.w13_weight_scale.unsqueeze(1)
                    .repeat_interleave(w13_scale_n, dim=1)
                    .unsqueeze(2)
                    .repeat_interleave(w13_scale_k, dim=2)
                )
                w2_scale_n = (w2_weight.shape[1] - 1) // scale_block_size + 1  # w2 缩放因子 N 维度
                w2_scale_k = (w2_weight.shape[2] - 1) // scale_block_size + 1  # w2 缩放因子 K 维度
                w2_scale = (
                    layer.w2_weight_scale.unsqueeze(1)
                    .repeat_interleave(w2_scale_n, dim=1)
                    .unsqueeze(2)
                    .repeat_interleave(w2_scale_k, dim=2)
                )
            quant_info = DeepGemmMoeQuantInfo(  # 构建 DeepGEMM MoE 量化信息
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                use_fp8=True,
                w13_scale=w13_scale,
                w2_scale=w2_scale,
                block_shape=block_shape,
                is_fp4_experts=self.is_fp4_expert,
            )
        elif (
            self.runner.runner_backend.is_flashinfer_trtllm()
            or self.runner.runner_backend.is_flashinfer_trtllm_routed()
        ):  # FlashInfer TRT-LLM 后端
            # FlashInfer TRT-LLM backend only supports fused execution and consumes
            # router logits directly (no separate apply_with_router_logits needed).
            # FlashInfer TRT-LLM 后端仅支持融合执行，直接使用路由 logits
            # （不需要单独的 apply_with_router_logits）。
            # FlashInfer TRT-LLM routed backend consumes SGLang-computed
            # top-k ids/weights (packed into int32) instead of router logits.
            # FlashInfer TRT-LLM routed 后端使用 SGLang 计算的 top-k id/权重
            #（打包为 int32）而非路由 logits。
            global_num_experts = int(getattr(layer, "num_experts"))  # 全局专家数
            num_local_experts = int(getattr(layer, "num_local_experts"))  # 本地专家数
            moe_ep_rank = int(getattr(layer, "moe_ep_rank"))  # 专家并行秩

            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                get_activation_type,
            )

            activation_type = get_activation_type(  # 获取激活函数类型
                self.moe_runner_config.activation,
                is_gated=self.moe_runner_config.is_gated,
            )

            quant_info = FlashInferTrtllmFp8MoeQuantInfo(  # 构建 FlashInfer TRT-LLM FP8 MoE 量化信息
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                global_num_experts=global_num_experts,
                local_expert_offset=moe_ep_rank * num_local_experts,
                local_num_experts=num_local_experts,
                intermediate_size=layer.w2_weight.shape[2],
                routing_method_type=int(
                    getattr(layer, "routing_method_type", None)
                    or RoutingMethodType.DeepSeekV3
                ),
                block_quant=self.block_quant,
                use_mxfp8=getattr(self.quant_config, "use_mxfp8", False),
                weight_block_k=(
                    None
                    if self.quant_config.weight_block_size is None
                    else self.quant_config.weight_block_size[1]
                ),
                w13_weight_scale_inv=(
                    layer.w13_weight_scale_inv if self.block_quant else None
                ),
                w2_weight_scale_inv=(
                    layer.w2_weight_scale_inv if self.block_quant else None
                ),
                w13_input_scale=layer.w13_input_scale if not self.block_quant else None,
                output1_scales_scalar=(
                    getattr(layer, "output1_scales_scalar", None)
                    if not self.block_quant
                    else None
                ),
                output1_scales_gate_scalar=(
                    getattr(layer, "output1_scales_gate_scalar", None)
                    if not self.block_quant
                    else None
                ),
                output2_scales_scalar=(
                    getattr(layer, "output2_scales_scalar", None)
                    if not self.block_quant
                    else None
                ),
                activation_type=activation_type,
            )
        elif self.runner.runner_backend.is_triton():  # Triton 后端
            quant_info = self.get_triton_quant_info(layer)
        else:
            raise NotImplementedError(
                "Unsupported runner backend: %s" % self.runner.runner_backend
            )

        return self.runner.run(dispatch_output, quant_info)

    def _ensure_cutlass_buffers_initialized(self, layer: Module) -> None:
        """初始化 CUTLASS FP8 MoE 所需的缓冲区和指针。
        
        预分配 CUTLASS 分组 GEMM 所需的各种工作空间、步长、偏移量等缓冲区，
        以避免运行时重复分配。
        """
        if getattr(self, "_cutlass_buffers_ready", False):
            return

        device = layer.w13_weight.device
        num_experts = layer.w13_weight.shape[0]
        hidden_size = layer.w2_weight.shape[1]
        intermediate_size_per_partition = layer.intermediate_size_per_partition

        self.ab_strides1 = torch.full(  # w13 权重/激活的步长
            (num_experts,), hidden_size, device=device, dtype=torch.int64
        )
        self.c_strides1 = torch.full(  # w13 输出的步长
            (num_experts,),
            2 * intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.ab_strides2 = torch.full(  # w2 权重/激活的步长
            (num_experts,),
            intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides2 = torch.full(  # w2 输出的步长
            (num_experts,), hidden_size, device=device, dtype=torch.int64
        )
        self.workspace = torch.empty(90000, device=device, dtype=torch.uint8)  # CUTLASS 工作空间
        self.a_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)  # 激活指针
        self.b_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)  # 权重指针
        self.out_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)  # 输出指针
        self.a_scales_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)  # 激活缩放因子指针
        self.b_scales_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)  # 权重缩放因子指针
        self.expert_offsets = torch.empty(  # 专家偏移量
            num_experts + 1, device=device, dtype=torch.int32
        )
        self.problem_sizes1 = torch.empty(  # w13 问题尺寸
            num_experts, 3, device=device, dtype=torch.int32
        )
        self.problem_sizes2 = torch.empty(  # w2 问题尺寸
            num_experts, 3, device=device, dtype=torch.int32
        )

        self._cutlass_buffers_ready = True  # 标记 CUTLASS 缓冲区已初始化

    def maybe_get_hip_aiter_quant_info(
        self,
        layer: torch.nn.Module,
        no_combine: bool = False,
    ) -> Optional["AiterMoeQuantInfo"]:
        """尝试获取 AMD AITER MoE 量化信息。
        
        如果当前不使用 AITER 或 INT4，返回 None。
        否则构建 AiterMoeQuantInfo，包括权重、缩放因子和量化类型。
        """
        if not (_use_aiter or _use_hip_int4):
            return None
        assert not no_combine, f"{no_combine=} is not supported."  # no_combine 模式不被支持

        from sglang.srt.layers.moe.moe_runner.aiter import (
            AiterMoeQuantInfo,
            AiterQuantType,
        )

        w13_weight = layer.w13_weight
        w2_weight = layer.w2_weight

        if self.block_quant:
            quant_type = (  # 根据是否为 FP4 专家选择量化类型
                AiterQuantType.PER_1X32  # FP4 使用 1x32 分块量化
                if self.is_fp4_expert
                else AiterQuantType.PER_128X128  # FP8 使用 128x128 分块量化
            )

            if self.is_fp4_expert:
                fp4_weight_dtype = _require_fp4_dtype()
                w13_weight = w13_weight.view(fp4_weight_dtype)
                w2_weight = w2_weight.view(fp4_weight_dtype)
                if getattr(layer.w13_weight, "is_shuffled", False):
                    w13_weight.is_shuffled = True
                    w2_weight.is_shuffled = True
            w13_scale = layer.w13_weight_scale_inv  # 分块量化使用逆缩放因子
            w2_scale = layer.w2_weight_scale_inv
        else:
            quant_type = AiterQuantType.PER_TOKEN  # 非分块量化使用逐 token 量化
            w13_scale = layer.w13_weight_scale1  # 非分块量化使用列缩放因子
            w2_scale = layer.w2_weight_scale1
        return AiterMoeQuantInfo(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            quant_type=quant_type,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            expert_mask=layer.dispatcher.expert_mask_gpu if _use_aiter else None,  # 专家掩码（AITER 用）
            swiglu_limit=self.moe_runner_config.swiglu_limit or 0.0,  # SwiGLU 截断限制
        )


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """FP8 KV 缓存量化方法。
    
    支持从 FP8 检查点加载 KV 缓存缩放因子。
    """

    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)  # 调用父类 BaseKVCacheMethod 的初始化
