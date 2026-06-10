# MoE（混合专家模型）工具模块：定义MoE All-to-All后端、MoE Runner后端、DeepEP模式等枚举类型，
# 以及全局配置初始化、查询函数和辅助工具函数，供MoE层各组件共用。
from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 日志模块
import os  # 操作系统接口
from contextlib import contextmanager  # 上下文管理器工具
from enum import Enum, IntEnum  # 枚举类型基类
from typing import TYPE_CHECKING, Optional  # 类型提示工具

import torch  # PyTorch深度学习框架

from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size  # 获取MoE专家并行世界大小
from sglang.srt.environ import envs  # 环境变量配置
from sglang.srt.layers.dp_attention import (  # 数据并行注意力相关工具
    get_attention_dp_size,  # 获取注意力DP大小
    is_dp_attention_enabled,  # 判断是否启用DP注意力
)
from sglang.srt.utils import is_npu  # 判断是否为NPU设备

_is_npu = is_npu()  # 全局标记：当前是否为NPU环境

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.server_args import ServerArgs  # 服务器参数类型

from sglang.srt.server_args import get_global_server_args  # 获取全局服务器参数

logger = logging.getLogger(__name__)  # 当前模块日志器


class MoeA2ABackend(Enum):  # MoE All-to-All通信后端枚举

    NONE = "none"  # 无A2A后端
    DEEPEP = "deepep"  # DeepEP后端
    MOONCAKE = "mooncake"  # Mooncake后端
    NIXL = "nixl"  # NIXL后端
    MORI = "mori"  # Mori后端
    ASCEND_FUSEEP = "ascend_fuseep"  # 昇腾FuseEP后端
    FLASHINFER = "flashinfer"  # FlashInfer后端
    MEGAMOE = "megamoe"  # MegaMoE后端
    CUSTOMIZED = "customized"  # 自定义后端

    @classmethod
    def _missing_(cls, value):  # 枚举值缺失时的回退处理
        if value is None:  # 如果值为None，返回NONE成员
            return cls.NONE
        for member in cls:  # 遍历所有成员进行值匹配
            if value == member.value:
                return member
        raise ValueError(f"No {cls.__name__} member for value {value}")  # 无匹配则抛出异常

    def is_none(self):  # 判断是否为NONE后端
        return self == MoeA2ABackend.NONE

    def is_deepep(self):  # 判断是否为DeepEP后端
        return self == MoeA2ABackend.DEEPEP

    def is_mooncake(self):  # 判断是否为Mooncake后端
        return self == MoeA2ABackend.MOONCAKE

    def is_nixl(self):  # 判断是否为NIXL后端
        return self == MoeA2ABackend.NIXL

    def is_flashinfer(self):  # 判断是否为FlashInfer后端
        return self == MoeA2ABackend.FLASHINFER

    def is_ascend_fuseep(self):  # 判断是否为昇腾FuseEP后端
        return self == MoeA2ABackend.ASCEND_FUSEEP

    def is_mori(self):  # 判断是否为Mori后端
        return self == MoeA2ABackend.MORI

    def is_megamoe(self):  # 判断是否为MegaMoE后端
        return self == MoeA2ABackend.MEGAMOE

    def is_customized(self):  # 判断是否为自定义后端
        return self == MoeA2ABackend.CUSTOMIZED

    def supports_aiter(self) -> bool:  # 判断该后端是否支持AITER加速
        return self in (  # 以下后端支持AITER
            MoeA2ABackend.NONE,
            MoeA2ABackend.DEEPEP,
            MoeA2ABackend.MOONCAKE,
            MoeA2ABackend.NIXL,
            MoeA2ABackend.MORI,
        )


class MoeRunnerBackend(Enum):  # MoE计算Runner后端枚举

    AUTO = "auto"  # 自动选择
    DEEP_GEMM = "deep_gemm"  # DeepGEMM后端
    TRITON = "triton"  # Triton后端
    TRITON_KERNELS = "triton_kernel"  # Triton Kernels后端
    FLASHINFER_TRTLLM = "flashinfer_trtllm"  # FlashInfer+TensorRT-LLM后端
    FLASHINFER_TRTLLM_ROUTED = "flashinfer_trtllm_routed"  # FlashInfer+TensorRT-LLM路由后端
    FLASHINFER_CUTLASS = "flashinfer_cutlass"  # FlashInfer+Cutlass后端
    FLASHINFER_MXFP4 = "flashinfer_mxfp4"  # FlashInfer+MXFP4后端
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"  # FlashInfer+CuteDSL后端
    CUTLASS = "cutlass"  # Cutlass后端
    MARLIN = "marlin"  # Marlin后端
    AITER = "aiter"  # AITER后端

    def is_auto(self):  # 判断是否为自动模式
        return self == MoeRunnerBackend.AUTO

    def is_deep_gemm(self):  # 判断是否为DeepGEMM后端
        return self == MoeRunnerBackend.DEEP_GEMM

    def is_triton(self):  # 判断是否为Triton后端
        return self == MoeRunnerBackend.TRITON

    def is_triton_kernels(self):  # 判断是否为Triton Kernels后端
        return self == MoeRunnerBackend.TRITON_KERNELS

    def is_flashinfer_trtllm(self):  # 判断是否为FlashInfer+TRT-LLM后端
        return self == MoeRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer_trtllm_routed(self):  # 判断是否为FlashInfer+TRT-LLM路由后端
        return self == MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED

    def is_flashinfer_cutlass(self):  # 判断是否为FlashInfer+Cutlass后端
        return self == MoeRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_cutedsl(self):  # 判断是否为FlashInfer+CuteDSL后端
        return self == MoeRunnerBackend.FLASHINFER_CUTEDSL

    def is_flashinfer_mxfp4(self):  # 判断是否为FlashInfer+MXFP4后端
        return self == MoeRunnerBackend.FLASHINFER_MXFP4

    def is_cutlass(self):  # 判断是否为Cutlass后端
        return self == MoeRunnerBackend.CUTLASS

    def is_marlin(self):  # 判断是否为Marlin后端
        return self == MoeRunnerBackend.MARLIN

    def is_aiter(self):  # 判断是否为AITER后端
        return self == MoeRunnerBackend.AITER


class DeepEPMode(Enum):  # DeepEP运行模式枚举

    NORMAL = "normal"  # 普通模式（高吞吐）
    LOW_LATENCY = "low_latency"  # 低延迟模式
    AUTO = "auto"  # 自动选择模式

    def enable_normal(self) -> bool:  # 判断是否启用普通模式
        return self in [DeepEPMode.NORMAL, DeepEPMode.AUTO]

    def enable_low_latency(self) -> bool:  # 判断是否启用低延迟模式
        return self in [DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO]

    def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:  # 根据批次中是否有extend请求解析实际模式
        if self != DeepEPMode.AUTO:  # 非AUTO模式直接返回自身
            return self

        if is_extend_in_batch:  # 批次中包含extend请求时使用普通模式
            return DeepEPMode.NORMAL
        else:  # 否则使用低延迟模式
            return DeepEPMode.LOW_LATENCY

    def is_normal(self) -> bool:  # 判断是否为普通模式
        return self == DeepEPMode.NORMAL

    def is_low_latency(self) -> bool:  # 判断是否为低延迟模式
        return self == DeepEPMode.LOW_LATENCY

    def is_auto(self) -> bool:  # 判断是否为自动模式
        return self == DeepEPMode.AUTO


class DeepEPOutputDtype(Enum):  # DeepEP调度输出数据类型枚举
    """
    Describes the dispatch output data type for DeepEP.  # 描述DeepEP调度输出的数据类型。

    - BF16: dispatch hidden states in bf16  # BF16：以bf16格式调度隐藏状态
    - FP8: dispatch hidden states in fp8  # FP8：以fp8格式调度隐藏状态
    - INT8: dispatch hidden states in int8  # INT8：以int8格式调度隐藏状态
    - NVFP4: dispatch hidden states in nvfp4  # NVFP4：以nvfp4格式调度隐藏状态
    """

    BF16 = "bf16"  # BF16数据类型
    FP8 = "fp8"  # FP8数据类型
    INT8 = "int8"  # INT8数据类型
    NVFP4 = "nvfp4"  # NVFP4数据类型


def get_deepep_output_dtype(self) -> DeepEPOutputDtype:  # 自动选择DeepEP调度输出数据类型
    """
    Automatically choose the dispatch output dtype for DeepEP.  # 自动选择DeepEP的调度输出数据类型。

    The decision follows several checks in priority order:  # 决策按以下优先级顺序进行：
    0. Parse server argument.  # 0. 解析服务器参数。
    1. Parse deprecated environment variables.  # 1. 解析已弃用的环境变量。
    2. If quant_config contains input_global_scale → NVFP4 path.  # 2. 若quant_config含input_global_scale → 走NVFP4路径。
    3. Parse quant config  # 3. 解析量化配置
    4. If flashinfer_cutedsl or is_cutlass backend is active → BF16 (it quantizes hidden_states internally).  # 4. 若flashinfer_cutedsl或cutlass后端活跃 → BF16（内部自行量化）。
    5. Otherwise default for NPU → BF16 (the default for NPU).  # 5. NPU默认 → BF16。
    6. Otherwise → FP8 (the default for most models like DeepSeek-V3).  # 6. 其他 → FP8（DeepSeek-V3等多数模型的默认值）。
    """

    # 0. Parse server argument.  # 0. 解析服务器参数。
    server_args = get_global_server_args()  # 获取全局服务器参数
    if server_args and server_args.deepep_dispatcher_output_dtype != "auto":  # 如果指定了非auto的数据类型
        return DeepEPOutputDtype(server_args.deepep_dispatcher_output_dtype)  # 直接返回指定类型

    # 1. Parse deprecated environment variables.  # 1. 解析已弃用的环境变量。
    if envs.SGLANG_DEEPEP_BF16_DISPATCH.get():  # 如果设置了已弃用的BF16调度环境变量
        logger.warning_once(  # 发出弃用警告
            "Warning: The env variable SGLANG_DEEPEP_BF16_DISPATCH deprecated "  # 警告：环境变量SGLANG_DEEPEP_BF16_DISPATCH已弃用
            "and will be removed in future releases. Please use a new "  # 将在未来版本中移除，请使用新的
            "`--deepep-dispatcher-output-dtype bf16` argument instead."  # `--deepep-dispatcher-output-dtype bf16`参数代替。
        )
        return DeepEPOutputDtype.BF16  # 返回BF16类型

    # 2. NVFP4 is detected inside dispatch_a / _dispatch_core via quant_config; no need to infer here.  # 2. NVFP4在dispatch_a/_dispatch_core中通过quant_config检测；此处无需推断。
    if self.quant_config is not None:  # 如果存在量化配置
        input_global_scale = self.quant_config.get("input_global_scale", None)  # 获取input_global_scale
        if input_global_scale is not None:  # 如果存在input_global_scale
            return DeepEPOutputDtype.NVFP4  # 返回NVFP4类型

        # 3. Parse quant config to determine the output dtype of dispatcher  # 3. 解析量化配置以确定调度器输出数据类型
        dispatcher_output_dtype = self.quant_config.get("dispatcher_output_dtype", None)  # 获取dispatcher_output_dtype
        if dispatcher_output_dtype is not None:  # 如果配置了dispatcher_output_dtype
            return DeepEPOutputDtype(dispatcher_output_dtype)  # 返回对应类型

    # 4. flashinfer_cutedsl and is_cutlass expects BF16 dispatch  # 4. flashinfer_cutedsl和cutlass后端需要BF16调度
    if (
        get_moe_runner_backend().is_flashinfer_cutedsl()  # 如果是flashinfer_cutedsl后端
        or get_moe_runner_backend().is_cutlass()  # 或者是cutlass后端
    ):
        return DeepEPOutputDtype.BF16  # 返回BF16类型

    # 5. Default on NPU → BF16  # 5. NPU默认 → BF16
    if _is_npu:  # 如果是NPU环境
        return DeepEPOutputDtype.BF16  # 返回BF16类型

    # 6. Default → FP8  # 6. 默认 → FP8
    return DeepEPOutputDtype.FP8  # 返回FP8类型


MOE_A2A_BACKEND: Optional[MoeA2ABackend] = None  # 全局MoE A2A后端配置，初始化为None
MOE_RUNNER_BACKEND: Optional[MoeRunnerBackend] = None  # 全局MoE Runner后端配置，初始化为None
SPECULATIVE_MOE_RUNNER_BACKEND: Optional[MoeRunnerBackend] = None  # 推测解码MoE Runner后端配置
SPECULATIVE_MOE_A2A_BACKEND: Optional[MoeA2ABackend] = None  # 推测解码MoE A2A后端配置
DEEPEP_MODE: Optional[DeepEPMode] = None  # DeepEP运行模式配置
IS_TBO_ENABLED: Optional[bool] = None  # 是否启用Two-Batch Overlap（双批次重叠）
IS_SBO_ENABLED: Optional[bool] = None  # 是否启用Single-Batch Overlap（单批次重叠）
TBO_TOKEN_DISTRIBUTION_THRESHOLD: Optional[float] = None  # TBO令牌分布阈值
DEEPEP_CONFIG: Optional[str] = None  # DeepEP配置字符串
DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER: Optional[bool] = None  # 是否禁用flashinfer cutlass MoE FP4 all-gather
MOE_QUANTIZATION: Optional[str] = None  # MoE量化方式


def initialize_moe_config(server_args: ServerArgs):  # 根据服务器参数初始化MoE全局配置
    global MOE_A2A_BACKEND  # 声明全局变量
    global MOE_RUNNER_BACKEND  # 声明全局变量
    global SPECULATIVE_MOE_RUNNER_BACKEND  # 声明全局变量
    global SPECULATIVE_MOE_A2A_BACKEND  # 声明全局变量
    global DEEPEP_MODE  # 声明全局变量
    global DEEPEP_CONFIG  # 声明全局变量
    global IS_TBO_ENABLED  # 声明全局变量
    global IS_SBO_ENABLED  # 声明全局变量
    global TBO_TOKEN_DISTRIBUTION_THRESHOLD  # 声明全局变量
    global DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER  # 声明全局变量
    global MOE_QUANTIZATION  # 声明全局变量

    MOE_A2A_BACKEND = MoeA2ABackend(server_args.moe_a2a_backend)  # 从服务器参数解析A2A后端
    MOE_RUNNER_BACKEND = MoeRunnerBackend(server_args.moe_runner_backend)  # 从服务器参数解析Runner后端
    SPECULATIVE_MOE_RUNNER_BACKEND = (  # 解析推测解码Runner后端
        MoeRunnerBackend(server_args.speculative_moe_runner_backend)  # 如果指定了推测解码后端
        if server_args.speculative_moe_runner_backend is not None  # 则使用指定值
        else MOE_RUNNER_BACKEND  # 否则回退到普通Runner后端
    )
    SPECULATIVE_MOE_A2A_BACKEND = (  # 解析推测解码A2A后端
        MoeA2ABackend(server_args.speculative_moe_a2a_backend)  # 如果指定了推测解码A2A后端
        if server_args.speculative_moe_a2a_backend is not None  # 则使用指定值
        else MOE_A2A_BACKEND  # 否则回退到普通A2A后端
    )
    DEEPEP_MODE = DeepEPMode(server_args.deepep_mode)  # 从服务器参数解析DeepEP模式
    DEEPEP_CONFIG = server_args.deepep_config or ""  # DeepEP配置，默认空字符串
    IS_TBO_ENABLED = server_args.enable_two_batch_overlap  # 是否启用TBO
    IS_SBO_ENABLED = server_args.enable_single_batch_overlap  # 是否启用SBO
    if IS_SBO_ENABLED and torch.cuda.is_available():  # 如果启用SBO且CUDA可用
        if torch.cuda.get_device_capability()[0] == 9:  # 检查是否为SM90架构
            raise ValueError(  # SBO在SM90 GPU上不支持
                "SBO (single batch overlap) is not supported on SM90 GPUs with latest sgl-deep-gemm wheel. Please try removing --enable-single-batch-overlap argument."  # 请移除--enable-single-batch-overlap参数
            )
    TBO_TOKEN_DISTRIBUTION_THRESHOLD = server_args.tbo_token_distribution_threshold  # TBO令牌分布阈值
    DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER = (  # 是否禁用flashinfer cutlass MoE FP4 all-gather
        server_args.disable_flashinfer_cutlass_moe_fp4_allgather
    )
    MOE_QUANTIZATION = server_args.quantization  # MoE量化方式


def get_moe_a2a_backend() -> MoeA2ABackend:  # 获取当前MoE A2A后端，未初始化时默认NONE
    global MOE_A2A_BACKEND
    if MOE_A2A_BACKEND is None:  # 如果未初始化
        MOE_A2A_BACKEND = MoeA2ABackend.NONE  # 默认为NONE
    return MOE_A2A_BACKEND


def get_moe_runner_backend() -> MoeRunnerBackend:  # 获取当前MoE Runner后端，未初始化时默认AUTO
    global MOE_RUNNER_BACKEND
    if MOE_RUNNER_BACKEND is None:  # 如果未初始化
        MOE_RUNNER_BACKEND = MoeRunnerBackend.AUTO  # 默认为AUTO
    return MOE_RUNNER_BACKEND


def get_speculative_moe_runner_backend() -> MoeRunnerBackend:  # 获取推测解码MoE Runner后端
    global SPECULATIVE_MOE_RUNNER_BACKEND
    if SPECULATIVE_MOE_RUNNER_BACKEND is None:  # 如果未初始化
        logger.warning(  # 发出警告
            "SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend"  # 未初始化，使用auto后端
        )
        SPECULATIVE_MOE_RUNNER_BACKEND = MoeRunnerBackend.AUTO  # 默认为AUTO
    return SPECULATIVE_MOE_RUNNER_BACKEND


def get_speculative_moe_a2a_backend() -> MoeA2ABackend:  # 获取推测解码MoE A2A后端
    global SPECULATIVE_MOE_A2A_BACKEND
    if SPECULATIVE_MOE_A2A_BACKEND is None:  # 如果未初始化
        logger.warning(  # 发出警告
            "SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend"  # 未初始化，使用none后端
        )
        SPECULATIVE_MOE_A2A_BACKEND = MoeA2ABackend.NONE  # 默认为NONE
    return SPECULATIVE_MOE_A2A_BACKEND


def get_deepep_mode() -> DeepEPMode:  # 获取DeepEP运行模式
    global DEEPEP_MODE
    if DEEPEP_MODE is None:  # 如果未初始化
        logger.warning("DEEPEP_MODE is not initialized, using auto mode")  # 未初始化，使用auto模式
        DEEPEP_MODE = DeepEPMode.AUTO  # 默认为AUTO
    return DEEPEP_MODE


def get_deepep_config() -> str:  # 获取DeepEP配置字符串
    global DEEPEP_CONFIG
    if DEEPEP_CONFIG is None:  # 如果未初始化
        logger.warning("DEEPEP_CONFIG is not initialized, using default config")  # 未初始化，使用默认配置
        DEEPEP_CONFIG = ""  # 默认空字符串
    return DEEPEP_CONFIG


def is_tbo_enabled() -> bool:  # 判断是否启用了TBO（双批次重叠）
    global IS_TBO_ENABLED
    if IS_TBO_ENABLED is None:  # 如果未初始化
        IS_TBO_ENABLED = False  # 默认不启用
    return IS_TBO_ENABLED


def is_sbo_enabled() -> bool:  # 判断是否启用了SBO（单批次重叠）
    global IS_SBO_ENABLED
    if IS_SBO_ENABLED is None:  # 如果未初始化
        IS_SBO_ENABLED = False  # 默认不启用
    return IS_SBO_ENABLED


def is_deepep_class_backend() -> bool:  # 判断当前MoE后端是否属于DeepEP家族（DeepEP、Mooncake或Mori）
    """Check if the MoE backend is DeepEP-family (DeepEP, Mooncake, or Mori)."""  # 检查MoE后端是否为DeepEP家族（DeepEP、Mooncake或Mori）。
    b = get_moe_a2a_backend()  # 获取当前A2A后端
    return b.is_deepep() or b.is_mooncake() or b.is_mori()  # 三者之一即为DeepEP家族


def is_flashinfer_cutedsl_v1_path() -> bool:  # 判断是否为CuteDSL v1 + DeepEP低延迟路径
    """CuteDSL v1 + DeepEP low-latency path (no MoeRunner, no autotune)."""  # CuteDSL v1 + DeepEP低延迟路径（无MoeRunner，无自动调优）。
    return (
        get_moe_runner_backend().is_flashinfer_cutedsl()  # Runner后端为flashinfer_cutedsl
        and get_moe_a2a_backend().is_deepep()  # A2A后端为deepep
    )


def get_tbo_token_distribution_threshold() -> float:  # 获取TBO令牌分布阈值
    global TBO_TOKEN_DISTRIBUTION_THRESHOLD
    if TBO_TOKEN_DISTRIBUTION_THRESHOLD is None:  # 如果未初始化
        logger.warning(  # 发出警告
            "TBO_TOKEN_DISTRIBUTION_THRESHOLD is not initialized, using 0.48"  # 未初始化，使用0.48
        )
        TBO_TOKEN_DISTRIBUTION_THRESHOLD = 0.48  # 默认阈值为0.48
    return TBO_TOKEN_DISTRIBUTION_THRESHOLD


def filter_moe_weight_param_global_expert(name, x, num_local_experts):  # 过滤需要全局专家的MoE参数
    """
    Filter out for MoE expert parameters that requires global expert.  # 过滤出需要全局专家的MoE专家参数。
    """
    return (
        not getattr(x, "_sglang_require_global_experts", False)  # 参数未标记为需要全局专家
        and x.data.ndim > 0  # 参数维度大于0（非标量）
        and x.data.shape[0] == num_local_experts  # 第一维等于本地专家数
    )


def should_use_flashinfer_cutlass_moe_fp4_allgather():  # 判断是否应在flashinfer cutlass MoE中使用FP4 all-gather以减少通信开销
    """
    Perform FP4 quantize before all-gather for flashinfer cutlass moe to reduce communication cost for high-throughput serving.  # 在all-gather之前进行FP4量化，以减少高吞吐服务中的通信成本。
    """
    return (
        not DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER  # 未禁用FP4 all-gather
        and get_moe_a2a_backend().is_none()  # A2A后端为none
        and get_moe_runner_backend().is_flashinfer_cutlass()  # Runner后端为flashinfer_cutlass
        and is_dp_attention_enabled()  # 启用了DP注意力
        and MOE_QUANTIZATION == "modelopt_fp4"  # 量化方式为modelopt_fp4
        and get_moe_expert_parallel_world_size() == get_attention_dp_size()  # 专家并行大小等于注意力DP大小
    )


def should_use_dp_reduce_scatterv():  # 判断是否应在标准调度器的combine中使用reduce_scatterv代替all-reduce+dp_scatter
    """
    Use reduce_scatterv in the standard dispatcher's combine() for DP attention  # 在标准调度器的combine()中使用reduce_scatterv进行DP注意力
    with EP, replacing the default all-reduce + dp_scatter path.  # 结合EP，替代默认的all-reduce + dp_scatter路径。
    Only changes the combine (post-kernel) communication; dispatch is unchanged.  # 仅改变combine（后内核）通信；dispatch不变。
    """
    return (
        not should_use_flashinfer_cutlass_moe_fp4_allgather()  # 不使用FP4 all-gather
        and get_moe_a2a_backend().is_none()  # A2A后端为none
        and is_dp_attention_enabled()  # 启用了DP注意力
        and get_attention_dp_size() > 1  # 注意力DP大小大于1
        and get_moe_expert_parallel_world_size() == get_attention_dp_size()  # 专家并行大小等于注意力DP大小
    )


def should_skip_post_experts_all_reduce(  # 判断是否应跳过专家后的all-reduce（EP或TP）
    *,
    is_tp_path: bool,  # 是否为TP路径
    use_reduce_scatter: bool = False,  # 是否使用reduce_scatter
    should_allreduce_fusion: bool = False,  # 是否进行all-reduce融合
) -> bool:
    """Whether to skip the post-experts all-reduce (EP or TP) because a  # 是否跳过专家后的all-reduce（EP或TP），因为
    downstream component will fuse, replace, or absorb it.  # 下游组件将融合、替换或吸收它。

    Skip reasons, in order:  # 跳过原因，按顺序：
      - ``should_allreduce_fusion``: LayerCommunicator will fuse the all-reduce  # ``should_allreduce_fusion``：LayerCommunicator将融合all-reduce
        with the next layer's residual all-reduce.  # 与下一层的残差all-reduce。
      - ``use_reduce_scatter``: LayerCommunicator's post-attention scatter will  # ``use_reduce_scatter``：LayerCommunicator的注意力后scatter将
        do reduce-scatter, which would double-reduce on top of an all-reduce.  # 执行reduce-scatter，与all-reduce会产生双重规约。
      - ``should_use_dp_reduce_scatterv()``: the standard dispatcher's combine  # ``should_use_dp_reduce_scatterv()``：标准调度器的combine
        path replaces the all-reduce with a reduce-scatterv.  # 路径用reduce-scatterv替代all-reduce。
      - ``should_use_flashinfer_cutlass_moe_fp4_allgather()`` (TP path only):  # ``should_use_flashinfer_cutlass_moe_fp4_allgather()``（仅TP路径）：
        the flashinfer cutlass FP4 kernel performs an all-gather that absorbs  # flashinfer cutlass FP4内核执行all-gather以吸收
        the post-experts TP all-reduce. Not relevant to the EP all-reduce.  # 专家后的TP all-reduce。与EP all-reduce无关。

    The first two args are layer-context flags from ``LayerCommunicator`` and  # 前两个参数是来自``LayerCommunicator``的层上下文标志，
    default to ``False`` for models that don't use it. Pass ``is_tp_path=True``  # 对于不使用它的模型默认为``False``。传入``is_tp_path=True``
    for the post-experts TP all-reduce, ``False`` for the EP all-reduce.  # 表示专家后的TP all-reduce，``False``表示EP all-reduce。
    """
    if should_allreduce_fusion or use_reduce_scatter:  # 如果需要融合all-reduce或使用reduce_scatter
        return True  # 跳过all-reduce
    if should_use_dp_reduce_scatterv():  # 如果应使用DP reduce_scatterv
        return True  # 跳过all-reduce
    if is_tp_path and should_use_flashinfer_cutlass_moe_fp4_allgather():  # 如果是TP路径且应使用FP4 all-gather
        return True  # 跳过all-reduce
    return False  # 不跳过


@contextmanager
def speculative_moe_backend_context():  # 上下文管理器：临时切换到推测解码MoE Runner后端
    """
    Context manager to temporarily use the speculative MoE backend for draft model operations.  # 上下文管理器：临时使用推测MoE后端进行草稿模型操作。
    This ensures that draft models in speculative decoding use the configured speculative backend.  # 确保推测解码中的草稿模型使用配置的推测后端。
    """
    global MOE_RUNNER_BACKEND  # 声明全局变量
    original_backend = MOE_RUNNER_BACKEND  # 保存原始后端
    try:
        MOE_RUNNER_BACKEND = get_speculative_moe_runner_backend()  # 切换到推测解码后端
        yield  # 执行上下文内代码
    finally:
        MOE_RUNNER_BACKEND = original_backend  # 恢复原始后端


@contextmanager
def speculative_moe_a2a_backend_context():  # 上下文管理器：临时切换到推测解码MoE A2A后端
    """
    Context manager to temporarily use the speculative MoE A2A backend for draft model operations.  # 上下文管理器：临时使用推测MoE A2A后端进行草稿模型操作。
    This ensures that draft models in speculative decoding use the configured speculative A2A backend.  # 确保推测解码中的草稿模型使用配置的推测A2A后端。
    """
    global MOE_A2A_BACKEND  # 声明全局变量
    global DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER  # 声明全局变量
    original_backend = MOE_A2A_BACKEND  # 保存原始A2A后端
    original_disable_flashinfer_cutlass_moe_fp4_allgather = (  # 保存原始FP4 all-gather禁用状态
        DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER
    )
    try:
        MOE_A2A_BACKEND = get_speculative_moe_a2a_backend()  # 切换到推测解码A2A后端
        # Disable FP4 allgather for spec decode since MTP layers are unquantized  # 禁用推测解码的FP4 allgather，因为MTP层未量化
        DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER = True  # 强制禁用FP4 allgather
        yield  # 执行上下文内代码
    finally:
        MOE_A2A_BACKEND = original_backend  # 恢复原始A2A后端
        DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER = (  # 恢复原始FP4 all-gather禁用状态
            original_disable_flashinfer_cutlass_moe_fp4_allgather
        )


# The type of method in top-K routing, for use in torch custom op  # Top-K路由中的方法类型，用于torch自定义算子
# Please keep this in sync with the counterpart defined in https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/trtllm/fused_moe/runner.h  # 请与flashinfer仓库中定义的对应项保持同步
class RoutingMethodType(IntEnum):  # 路由方法类型枚举（整型枚举）
    # Default: Softmax -> TopK  # 默认：Softmax -> TopK
    Default = (0,)
    # Renormalize: TopK -> Softmax  # 重归一化：TopK -> Softmax
    Renormalize = (1,)
    # DeepSeekV3: Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts from the Top4 groups  # DeepSeekV3：Sigmoid -> 路由偏置加法 -> 组内Top2 -> Top4组 -> 从Top4组选Top8专家
    DeepSeekV3 = (2,)
    # Llama4: Top1 -> Sigmoid  # Llama4：Top1 -> Sigmoid
    Llama4 = (3,)
    # Qwen3: Softmax -> TopK -> Renormalize  # Qwen3：Softmax -> TopK -> 重归一化
    RenormalizeNaive = (4,)
    # TopK only (no softmax)  # 仅TopK（无softmax）
    TopK = (5,)
    # Unspecified  # 未指定
    Unspecified = 6


AITER_PADDING_SIZE = 128  # AITER后端对齐填充大小
TRITON_PADDING_SIZE = 128  # Triton后端对齐填充大小


# Unit of padding - context dependent  # 填充单位——视上下文而定
def get_moe_padding_size(is_aiter_moe):  # 获取MoE权重填充大小
    if is_aiter_moe:  # 如果是AITER后端
        return AITER_PADDING_SIZE  # 返回AITER填充大小
    else:  # 否则
        return (  # 根据环境变量决定是否使用Triton填充
            TRITON_PADDING_SIZE  # Triton填充大小
            if bool(int(os.getenv("SGLANG_MOE_PADDING", "0")))  # 如果设置了SGLANG_MOE_PADDING环境变量
            else 0  # 否则不填充
        )


def get_moe_weight_sizes(inter_dim, is_concat, is_packed, is_aiter_moe):  # 计算MoE权重张量的维度
    """
    Calculate dimensions for MoE weight tensors.  # 计算MoE权重张量的维度。

    Args:  # 参数：
        inter_dim: Base intermediate dimension.  # inter_dim：基础中间维度。
        is_concat: If True, fusions W1 (gate) and W3 (up) projections.  # is_concat：如果为True，融合W1（gate）和W3（up）投影。
        is_packed: If True, uses 4-bit quantization (two FP4 elements per byte).  # is_packed：如果为True，使用4位量化（每字节两个FP4元素）。
        is_aiter_moe: If True, applies Aiter-specific kernel padding alignment.  # is_aiter_moe：如果为True，应用AITER特定的内核填充对齐。
    """
    # w2_down_dim is the packing rank, but w13_up_dim not (of matrix to matmul)  # w2_down_dim是打包维度，但w13_up_dim不是（矩阵乘法中的矩阵）
    w13_up_dim = 2 * inter_dim if is_concat else inter_dim  # W13上投影维度：拼接时为2倍，否则为1倍
    w2_down_dim = inter_dim // 2 if is_packed else inter_dim  # W2下投影维度：打包时减半，否则不变

    if is_aiter_moe:  # 如果是AITER后端，需要特殊对齐
        padding_size = get_moe_padding_size(True)  # 获取AITER填充大小
        align_aiter = lambda n: ((n + padding_size - 1) // padding_size) * padding_size  # AITER对齐函数
        is_padded = (w2_down_dim % padding_size) > 0  # 判断w2_down_dim是否需要填充
        if is_padded:  # 如果需要填充
            # w2_down_dim, padding & aligned, unit: parameter dtype  # w2_down_dim，填充并对齐，单位：参数数据类型
            w2_down_dim = align_aiter(w2_down_dim)  # 对齐w2_down_dim
        # up proj + gate fusion : 2x  # 上投影 + gate融合：2倍
        if is_concat:  # 如果是拼接模式
            w13_up_dim = w2_down_dim * 2  # W13维度为对齐后w2维度的2倍
        # packed  # 打包模式
        if hasattr(torch, "float4_e2m1fn_x2") and is_packed:  # 如果支持FP4类型且为打包模式
            # w13_up_dim (row rank of matmul matrix) is not packing dim, *2 to recover  # w13_up_dim（矩阵乘法的行秩）不是打包维度，*2恢复
            w13_up_dim *= 2  # 恢复打包导致的维度减半

    return (w13_up_dim, w2_down_dim, False if not is_aiter_moe else is_padded)  # 返回(W13维度, W2维度, 是否填充)