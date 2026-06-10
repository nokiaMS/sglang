# FlashInfer注意力后端实现模块
# 该模块实现了基于FlashInfer库的注意力机制后端，是SGLang中最重要的注意力后端之一。
# 支持两种操作：extend（即带缓存前缀的预填充）和decode（逐token解码）。
# 包含FlashInferAttnBackend（主后端）、FlashInferIndicesUpdaterDecode/Preffill（索引更新器）、
# FlashInferMultiStepDraftBackend（多步草稿后端）等核心类。
from __future__ import annotations  # 启用延迟类型注解评估

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""
# 支持不同的注意力后端。
# 目前有两个后端：FlashInfer和Triton。
# FlashInfer更快，Triton更容易定制。
# 每个后端支持两种操作：extend（即带缓存前缀的预填充）和decode。

import logging  # 导入日志模块
import os  # 导入操作系统模块
from dataclasses import dataclass  # 导入数据类装饰器
from enum import Enum, auto  # 导入枚举类和自动值生成器
from functools import partial  # 导入偏函数工具
from typing import TYPE_CHECKING, Callable, List, Optional, Union  # 导入类型提示

import torch  # 导入PyTorch框架

from sglang.kernel_api_logging import debug_kernel_api  # 导入内核API调试装饰器
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph  # 导入分段CUDA图检测函数
from sglang.srt.dllm.config import DllmConfig  # 导入DLLM配置类
from sglang.srt.environ import envs  # 导入环境变量模块
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend  # 导入注意力后端基类
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton  # 导入KV索引创建Triton核函数
from sglang.srt.layers.dp_attention import get_attention_tp_size  # 导入注意力张量并行大小获取函数
from sglang.srt.layers.radix_attention import AttentionType  # 导入注意力类型枚举
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator  # 导入滑动窗口KV池分配器
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode  # 导入前向批次信息和模式
from sglang.srt.speculative.spec_info import SpecInput  # 导入投机解码输入规格
from sglang.srt.utils import (  # 导入工具函数
    get_int_env_var,  # 获取整数环境变量
    is_flashinfer_available,  # 检查FlashInfer是否可用
    is_sm100_supported,  # 检查SM100架构是否支持
    next_power_of_2,  # 计算下一个2的幂
)

if TYPE_CHECKING:  # 类型检查阶段才导入，避免运行时循环依赖
    from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力层类
    from sglang.srt.model_executor.model_runner import ModelRunner  # 导入模型运行器类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

if envs.SGLANG_ENABLE_TORCH_COMPILE.get():  # 如果启用了Torch编译
    torch._logging.set_logs(dynamo=logging.ERROR)  # 将dynamo日志级别设为ERROR，减少噪音
    torch._dynamo.config.suppress_errors = True  # 抑制dynamo编译错误


if is_flashinfer_available():  # 如果FlashInfer库可用
    from flashinfer import (  # 导入FlashInfer核心模块
        BatchDecodeWithPagedKVCacheWrapper,  # 分页KV缓存解码包装器
        BatchPrefillWithPagedKVCacheWrapper,  # 分页KV缓存预填充包装器
        BatchPrefillWithRaggedKVCacheWrapper,  # 不规则KV缓存预填充包装器
        fast_decode_plan,  # 快速解码规划函数
    )
    from flashinfer.cascade import merge_state  # 导入状态合并函数

    from sglang.srt.layers.attention.triton_ops.merge_state import merge_state_triton  # 导入Triton状态合并实现

    # FlashInfer's MergeState CUDA kernel uses blockDim = (head_dim/vec_size, num_heads).
    # When num_heads is large (e.g. with DP attention where attention_tp_size=1), the
    # total threads per block can exceed CUDA's limit of 1024 and the kernel launch fails
    # with `invalid configuration argument`. Fall back to the in-tree Triton implementation,
    # which uses (token, head) as the launch grid and is therefore unaffected.
    # FlashInfer的MergeState CUDA核函数使用blockDim = (head_dim/vec_size, num_heads)。
    # 当num_heads较大时（例如在DP注意力中attention_tp_size=1），
    # 每个块的总线程数可能超过CUDA的1024限制，核函数启动会失败并报
    # `invalid configuration argument`错误。回退到内置的Triton实现，
    # 它使用(token, head)作为启动网格，因此不受影响。
    _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK = 1024  # CUDA每个块最大线程数常量

    def _merge_state_max_safe_num_heads(head_dim: int, element_size: int) -> int:  # 计算MergeState最大安全头数
        # Mirrors flashinfer's vec_size selection in include/flashinfer/attention/cascade.cuh.
        # 镜像FlashInfer在include/flashinfer/attention/cascade.cuh中的vec_size选择逻辑。
        vec_size = max(16 // element_size, head_dim // 32)  # 计算向量化大小
        bdx = head_dim // vec_size  # 计算块维度x大小
        if bdx <= 0:  # 如果块维度非正
            return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK  # 返回最大线程数作为上限
        return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK // bdx  # 返回安全的最大头数

    def _safe_merge_state(  # 安全的状态合并函数，自动选择CUDA或Triton实现
        v_a: torch.Tensor,  # 第一个值状态
        s_a: torch.Tensor,  # 第一个分数状态
        v_b: torch.Tensor,  # 第二个值状态
        s_b: torch.Tensor,  # 第二个分数状态
    ):
        num_heads = v_a.shape[1]  # 获取头数
        head_dim = v_a.shape[2]  # 获取头维度
        max_heads = _merge_state_max_safe_num_heads(head_dim, v_a.element_size())  # 计算最大安全头数
        if num_heads <= max_heads:  # 如果头数在安全范围内
            return merge_state(v_a, s_a, v_b, s_b)  # 使用FlashInfer的CUDA实现
        return merge_state_triton(v_a, s_a, v_b, s_b)  # 否则使用Triton实现


class WrapperDispatch(Enum):  # 包装器分发枚举，决定使用哪种注意力模式
    SLIDING_WINDOW = auto()  # 滑动窗口注意力模式
    CROSS_ATTENTION = auto()  # 交叉注意力模式


@dataclass  # 数据类装饰器
class MultiItemScoringParams:  # 多项评分参数数据类
    """Parameters for multi-item scoring in attention computation.
    # 注意力计算中多项评分的参数

    Used when processing sequences with multiple items separated by delimiters,
    where each item needs specific attention patterns that respect item boundaries.
    # 用于处理由分隔符分隔的多个项的序列，
    # 其中每个项需要遵守项边界的特定注意力模式。

    Attributes:
        prefix_len_ptr: A uint32 1D tensor indicating the prefix length of each prompt.
                        The tensor size is equal to the batch size.
        # prefix_len_ptr: uint32 1D张量，表示每个提示的前缀长度。
        #                 张量大小等于批次大小。
        token_pos_in_items_ptr: A uint16 1D tensor indicating the token position of each item
                                starting from 0 (delimiter) for each item. For batch size > 1,
                                sequences are concatenated with zero padding to ensure same length.
        # token_pos_in_items_ptr: uint16 1D张量，表示每个项中每个token的位置，
        #                        从0（分隔符）开始。对于批次大小>1，
        #                        序列用零填充连接以确保相同长度。
        token_pos_in_items_len: Zero padding length for token_pos_in_items_ptr to handle
                                batch_size > 1 case. Defines the padded length for each sequence.
        # token_pos_in_items_len: token_pos_in_items_ptr的零填充长度，用于处理
        #                        batch_size > 1的情况。定义每个序列的填充长度。
        max_item_len_ptr: A uint16 tensor containing the max token length of all items
                         for each prompt in the batch.
        # max_item_len_ptr: uint16张量，包含批次中每个提示的所有项的最大token长度。
    """

    prefix_len_ptr: Optional[torch.Tensor] = None  # 前缀长度指针，默认为None
    token_pos_in_items_ptr: Optional[torch.Tensor] = None  # 项内token位置指针，默认为None
    token_pos_in_items_len: int = 0  # 项内token位置长度，默认为0
    max_item_len_ptr: Optional[torch.Tensor] = None  # 最大项长度指针，默认为None

    def is_enabled(self) -> bool:  # 检查多项评分是否已启用
        """Check if multi-item scoring is enabled."""  # 检查多项评分是否已启用
        return self.prefix_len_ptr is not None  # 如果前缀长度指针不为None则已启用


@dataclass  # 数据类装饰器
class DecodeMetadata:  # 解码元数据类
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]  # 解码包装器列表


@dataclass  # 数据类装饰器
class PrefillMetadata:  # 预填充元数据类
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]  # 预填充包装器列表
    use_ragged: bool  # 是否使用不规则（ragged）模式
    extend_no_prefix: bool  # 扩展是否无前缀
    multi_item_params: Optional[MultiItemScoringParams] = None  # 多项评分参数，默认为None


# Reuse this workspace buffer across all flashinfer wrappers
# 在所有FlashInfer包装器之间重用此工作区缓冲区
global_workspace_buffer = None  # 全局工作区缓冲区，初始化为None

# Use as a fast path to override the indptr in flashinfer's plan function
# This is used to remove some host-to-device copy overhead.
# 用作快速路径覆盖FlashInfer plan函数中的indptr
# 用于减少一些主机到设备的拷贝开销。
global_override_indptr_cpu = None  # 全局覆盖indptr CPU缓冲区，初始化为None


class FlashInferAttnBackend(AttentionBackend):  # FlashInfer注意力后端类，继承自注意力后端基类
    """Flashinfer attention kernels."""  # FlashInfer注意力核函数

    def __init__(  # 初始化方法
        self,
        model_runner: ModelRunner,  # 模型运行器实例
        skip_prefill: bool = False,  # 是否跳过预填充，默认为False
        kv_indptr_buf: Optional[torch.Tensor] = None,  # KV索引指针缓冲区（可选）
        kv_last_page_len_buf: Optional[torch.Tensor] = None,  # KV最后页长度缓冲区（可选）
        init_new_workspace: bool = False,  # 是否初始化新的工作区，默认为False
    ):
        super().__init__()  # 调用父类初始化
        self.prefill_backend = "fa2"  # 预填充后端设为fa2（FlashAttention 2）
        self.decode_backend = "fa2"  # 解码后端设为fa2

        self.req_to_token_pool = model_runner.req_to_token_pool  # 保存请求到token池引用
        self.token_to_kv_pool = model_runner.token_to_kv_pool  # 保存token到KV池引用
        self.enable_mis = model_runner.server_args.enable_mis  # 保存是否启用多项评分标志

        # FIXME: remove dllm workarounds from flashinfer
        # 待修复：从flashinfer中移除dllm的变通方案
        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)  # 从服务器参数创建DLLM配置
        self.is_dllm_model = self.dllm_config is not None  # 判断是否为DLLM模型

        # Parse constants
        # 解析常量
        self.decode_use_tensor_cores = should_use_tensor_core(  # 判断解码是否使用Tensor Core
            kv_cache_dtype=model_runner.kv_cache_dtype,  # KV缓存数据类型
            num_attention_heads=model_runner.model_config.num_attention_heads  # 注意力头数
            // get_attention_tp_size(),  # 除以注意力TP大小
            num_kv_heads=model_runner.model_config.get_num_kv_heads(  # KV头数
                get_attention_tp_size()  # 注意力TP大小
            ),
        )
        self.max_context_len = model_runner.model_config.context_len  # 保存最大上下文长度
        self.skip_prefill = skip_prefill  # 保存是否跳过预填充标志
        self.is_multimodal = model_runner.model_config.is_multimodal  # 保存是否为多模态模型
        assert not (  # 断言检查
            model_runner.sliding_window_size is not None  # 滑动窗口大小存在
            and model_runner.model_config.is_encoder_decoder  # 且是编码器-解码器模型
        ), "Sliding window and cross attention are not supported together"  # 错误提示：滑动窗口和交叉注意力不能同时使用

        if model_runner.sliding_window_size is not None:  # 如果启用滑动窗口
            self.num_wrappers = 2  # 需要两个包装器
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW  # 分发原因为滑动窗口
        elif model_runner.model_config.is_encoder_decoder:  # 如果是编码器-解码器模型
            self.num_wrappers = 2  # 需要两个包装器
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION  # 分发原因为交叉注意力
        else:  # 否则
            self.num_wrappers = 1  # 只需要一个包装器
            self.dispatch_reason = None  # 无分发原因

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        # Qwen2/Qwen3模型需要更大的FlashInfer工作区大小
        if (  # 如果模型架构属于以下之一
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures  # Qwen2
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures  # Qwen3
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures  # MiMo
            or "Qwen3VLForConditionalGeneration"  # Qwen3视觉语言
            in model_runner.model_config.hf_config.architectures
            or "Qwen3VLMoeForConditionalGeneration"  # Qwen3视觉语言MoE
            in model_runner.model_config.hf_config.architectures
        ):
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(512 * 1024 * 1024)  # 设置工作区大小为512MB

        # When deterministic inference is enabled, tensor cores should be used for decode
        # Also set split tile sizes for prefill and decode from environment variables, and disable kv split for cuda graph
        # More information can be found here: https://github.com/flashinfer-ai/flashinfer/pull/1675
        # 当启用确定性推理时，解码应使用Tensor Core
        # 同时从环境变量设置预填充和解码的分片瓦片大小，并禁用CUDA图的KV分片
        # 更多信息请参阅：https://github.com/flashinfer-ai/flashinfer/pull/1675
        self.enable_deterministic = (  # 保存是否启用确定性推理
            model_runner.server_args.enable_deterministic_inference  # 从服务器参数获取
        )
        self.prefill_split_tile_size = None  # 预填充分片瓦片大小，默认为None
        self.decode_split_tile_size = None  # 解码分片瓦片大小，默认为None
        self.disable_cuda_graph_kv_split = False  # 是否禁用CUDA图KV分片，默认为False
        if self.enable_deterministic:  # 如果启用确定性推理
            self.decode_use_tensor_cores = True  # 解码强制使用Tensor Core
            self.prefill_split_tile_size = get_int_env_var(  # 获取预填充分片瓦片大小
                "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096  # 环境变量名和默认值
            )
            self.decode_split_tile_size = get_int_env_var(  # 获取解码分片瓦片大小
                "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE", 2048  # 环境变量名和默认值
            )
            self.disable_cuda_graph_kv_split = True  # 禁用CUDA图KV分片
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(2048 * 1024 * 1024)  # 设置工作区大小为2GB

        self.use_paged = envs.SGLANG_FLASHINFER_USE_PAGED.get()  # 获取是否使用分页模式

        # Allocate buffers
        # 分配缓冲区
        global global_workspace_buffer  # 声明全局工作区缓冲区
        if global_workspace_buffer is None:  # 如果全局工作区缓冲区未初始化
            # different from flashinfer zero_init_global_workspace_buffer
            # 与FlashInfer的zero_init_global_workspace_buffer不同
            global_workspace_size = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()  # 获取工作区大小
            global_workspace_buffer = torch.empty(  # 创建空的工作区缓冲区
                global_workspace_size,  # 大小
                dtype=torch.uint8,  # 数据类型为uint8
                device=model_runner.device,  # 设备
            )
        if init_new_workspace:  # 如果需要初始化新的工作区
            self.workspace_buffer = torch.empty(  # 创建新的工作区缓冲区
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),  # 大小
                dtype=torch.uint8,  # 数据类型为uint8
                device=model_runner.device,  # 设备
            )
        else:  # 否则
            self.workspace_buffer = global_workspace_buffer  # 使用全局工作区缓冲区
        max_bs = model_runner.req_to_token_pool.size  # 获取最大批次大小
        if kv_indptr_buf is None:  # 如果未提供KV索引指针缓冲区
            self.kv_indptr = [  # 创建KV索引指针列表
                torch.zeros(  # 创建零张量
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device  # 形状、类型、设备
                )
                for _ in range(self.num_wrappers)  # 为每个包装器创建一个
            ]
        else:  # 否则
            assert self.num_wrappers == 1  # 断言只有一个包装器
            self.kv_indptr = [kv_indptr_buf]  # 使用提供的缓冲区

        if kv_last_page_len_buf is None:  # 如果未提供KV最后页长度缓冲区
            self.kv_last_page_len = torch.ones(  # 创建全1的KV最后页长度张量
                (max_bs,), dtype=torch.int32, device=model_runner.device  # 形状、类型、设备
            )
        else:  # 否则
            assert self.num_wrappers == 1  # 断言只有一个包装器
            self.kv_last_page_len = kv_last_page_len_buf  # 使用提供的缓冲区

        if not self.skip_prefill:  # 如果不跳过预填充
            self.qo_indptr = [  # 创建查询输出索引指针列表
                torch.zeros(  # 创建零张量
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device  # 形状、类型、设备
                )
                for _ in range(self.num_wrappers)  # 为每个包装器创建一个
            ]

        fmha_backend = "auto"  # 初始化FMHA后端为auto
        if is_sm100_supported():  # 如果支持SM100架构
            if not model_runner.server_args.disable_piecewise_cuda_graph:  # 如果未禁用分段CUDA图
                logger.info(  # 记录信息
                    "CUTLASS backend is disabled when piecewise cuda graph is enabled "
                    "due to TMA descriptor initialization issues on SM100 GPUs. "
                    "Using auto backend instead for stability."
                    # 启用分段CUDA图时，由于SM100 GPU上的TMA描述符初始化问题，
                    # CUTLASS后端被禁用。改用auto后端以确保稳定性。
                )
            else:  # 否则
                fmha_backend = "cutlass"  # 使用CUTLASS后端
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(  # 创建不规则预填充包装器
            self.workspace_buffer, "NHD", backend=fmha_backend  # 工作区、布局、后端
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        # 两个包装器：一个用于滑动窗口注意力，一个用于全注意力。
        # 在当前PR中使用两个包装器是不必要的，但为未来的PR做了准备
        self.prefill_wrappers_paged = []  # 分页预填充包装器列表
        self.prefill_wrappers_verify = []  # 验证预填充包装器列表
        self.decode_wrappers = []  # 解码包装器列表
        for _ in range(self.num_wrappers):  # 为每个包装器创建实例
            if not skip_prefill:  # 如果不跳过预填充
                self.prefill_wrappers_paged.append(  # 添加分页预填充包装器
                    BatchPrefillWithPagedKVCacheWrapper(  # 创建分页预填充包装器实例
                        self.workspace_buffer,  # 工作区缓冲区
                        "NHD",  # 布局
                        backend=self.prefill_backend,  # 后端
                    )
                )
                self.prefill_wrappers_verify.append(  # 添加验证预填充包装器
                    BatchPrefillWithPagedKVCacheWrapper(  # 创建分页预填充包装器实例
                        self.workspace_buffer,  # 工作区缓冲区
                        "NHD",  # 布局
                        backend=self.prefill_backend,  # 后端
                    )
                )
            self.decode_wrappers.append(  # 添加解码包装器
                BatchDecodeWithPagedKVCacheWrapper(  # 创建分页解码包装器实例
                    self.workspace_buffer,  # 工作区缓冲区
                    "NHD",  # 布局
                    backend=self.decode_backend,  # 后端
                    use_tensor_cores=self.decode_use_tensor_cores,  # 是否使用Tensor Core
                )
            )

        # Create indices updater
        # 创建索引更新器
        if not skip_prefill:  # 如果不跳过预填充
            self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(  # 创建预填充索引更新器
                model_runner, self  # 传入模型运行器和当前后端实例
            )  # for verify  # 用于验证
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)  # 创建解码索引更新器

        # Other metadata
        # 其他元数据
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None  # 前向元数据，初始化为None

        self.decode_cuda_graph_metadata = {}  # 解码CUDA图元数据字典
        self.prefill_cuda_graph_metadata = {}  # 预填充CUDA图元数据字典（用于验证）
        self.draft_extend_cuda_graph_metadata = {}  # 草稿扩展CUDA图元数据字典

    def _process_multi_item_scoring(  # 处理多项评分张量（内部方法）
        self, forward_batch: ForwardBatch  # 前向批次
    ) -> MultiItemScoringParams:  # 返回多项评分参数
        """Process multi-item scoring tensors for FlashInfer attention.
        # 处理FlashInfer注意力的多项评分张量

        This method handles sequences containing multiple "items" separated by delimiter tokens,
        where each item needs specific attention patterns that respect item boundaries.
        # 此方法处理包含由分隔符token分隔的多个"项"的序列，
        # 其中每个项需要遵守项边界的特定注意力模式。

        The method produces four key tensors for FlashInfer:
        - prefix_len_ptr: uint32 tensor with prefix length for each prompt in batch
        - token_pos_in_items_ptr: uint16 tensor with token positions starting from 0 at delimiters
        - token_pos_in_items_len: padding length for batch processing
        - max_item_len_ptr: uint16 tensor with max item length for each prompt
        # 此方法为FlashInfer生成四个关键张量：
        # - prefix_len_ptr: uint32张量，包含批次中每个提示的前缀长度
        # - token_pos_in_items_ptr: uint16张量，包含从0开始的分隔符处token位置
        # - token_pos_in_items_len: 批处理的填充长度
        # - max_item_len_ptr: uint16张量，包含每个提示的最大项长度

        Args:
            forward_batch: The forward batch containing input sequences and delimiter info
        # 参数：
        #     forward_batch: 包含输入序列和分隔符信息的前向批次

        Returns:
            MultiItemScoringParams: The processed multi-item scoring parameters
        # 返回：
        #     MultiItemScoringParams: 处理后的多项评分参数

        Examples:
            Following FlashInfer definition: for 3 items of length 3, 2, 4 respectively:
            token_pos_in_items_ptr = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]
        # 示例：
        #     按照FlashInfer定义：对于3个长度分别为3、2、4的项：
        #     token_pos_in_items_ptr = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]

            Case 1: Single sequence
            Text: "What is the capital of France? <delim> London <delim> Paris <delim> Berlin <delim>"
            Tokens: [What, is, the, capital, of, France, ?, <delim>, London, <delim>, Paris, <delim>, Berlin, <delim>]
            Indices: [ 0,   1,  2,   3,      4,  5,     6,   7,     8,      9,     10,    11,    12,     13]
            - prefix_len_ptr: [7] (query length before first delimiter)
            - token_pos_in_items_ptr: [0, 1, 0, 1, 0, 1, 0] (delim=0, London=1, delim=0, Paris=1, delim=0, Berlin=1, delim=0)
            - token_pos_in_items_len: 7 (actual length)
            - max_item_len_ptr: [1] (max item length is 1 token - all options are single tokens)
        # 情况1：单个序列
        # 文本："What is the capital of France? <delim> London <delim> Paris <delim> Berlin <delim>"
        # Token: [What, is, the, capital, of, France, ?, <delim>, London, <delim>, Paris, <delim>, Berlin, <delim>]
        # 索引: [ 0,   1,  2,   3,      4,  5,     6,   7,     8,      9,     10,    11,    12,     13]
        # - prefix_len_ptr: [7]（第一个分隔符前的查询长度）
        # - token_pos_in_items_ptr: [0, 1, 0, 1, 0, 1, 0]（分隔符=0, London=1, 分隔符=0, Paris=1, 分隔符=0, Berlin=1, 分隔符=0）
        # - token_pos_in_items_len: 7（实际长度）
        # - max_item_len_ptr: [1]（最大项长度为1个token - 所有选项都是单个token）

            Case 2: Batch processing (batch_size=2)
            Sequence 1: 2 items of length 2, 1 → [0, 1, 2, 0, 1, 0] (6 elements)
            Sequence 2: 3 items of length 1, 3, 2 → [0, 1, 0, 1, 2, 3, 0, 1, 2, 0] (10 elements)
            After padding both to length 10:
            - token_pos_in_items_ptr: [0, 1, 2, 0, 1, 0, 0, 0, 0, 0,    0, 1, 0, 1, 2, 3, 0, 1, 2, 0]
            - token_pos_in_items_len: 10 (padded length for batch processing)
            - max_item_len_ptr: [2, 3] (max lengths per sequence)
        # 情况2：批处理（batch_size=2）
        # 序列1：2个长度为2、1的项 → [0, 1, 2, 0, 1, 0]（6个元素）
        # 序列2：3个长度为1、3、2的项 → [0, 1, 0, 1, 2, 3, 0, 1, 2, 0]（10个元素）
        # 两者填充到长度10后：
        # - token_pos_in_items_ptr: [0, 1, 2, 0, 1, 0, 0, 0, 0, 0,    0, 1, 0, 1, 2, 3, 0, 1, 2, 0]
        # - token_pos_in_items_len: 10（批处理的填充长度）
        # - max_item_len_ptr: [2, 3]（每个序列的最大长度）
        """

        if not self.enable_mis or forward_batch.forward_mode == ForwardMode.DECODE:  # 如果未启用MIS或为解码模式
            return MultiItemScoringParams()  # 返回空的参数

        precomputed_indices = forward_batch.multi_item_delimiter_indices  # 获取预计算的分隔符索引
        if precomputed_indices is None:  # 如果预计算索引为空
            return MultiItemScoringParams()  # 返回空的参数

        prefix_cache_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)  # 获取前缀缓存长度
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)  # 获取扩展序列长度
        prefix_len_ptr, token_pos_in_items_ptr = [], []  # 初始化前缀长度列表和项内位置列表
        token_pos_in_items_len = 0  # 初始化项内位置长度
        device = forward_batch.input_ids.device  # 获取设备

        # If no extend_seq_lens, treat whole batch as one sequence
        # 如果没有extend_seq_lens，将整个批次视为一个序列
        if extend_seq_lens is None or len(extend_seq_lens) <= 1:  # 如果扩展序列长度为空或长度<=1
            extend_seq_lens = [forward_batch.input_ids.size(0)]  # 使用输入ID的总token数

        seq_start = 0  # 初始化序列起始位置
        for i, seq_len in enumerate(extend_seq_lens):  # 遍历每个序列
            seq_end = seq_start + seq_len  # 计算序列结束位置
            delimiter_indices_cpu = precomputed_indices[i]  # 获取当前序列的分隔符索引
            if len(delimiter_indices_cpu) == 0:  # 如果没有分隔符
                seq_start = seq_end  # 更新起始位置
                continue  # 跳过

            first_delim = delimiter_indices_cpu[0].item()  # CPU .item(), no GPU sync  # 获取第一个分隔符位置（CPU操作，无GPU同步）
            delimiter_indices = delimiter_indices_cpu.to(device, non_blocking=True)  # 将分隔符索引转移到设备（非阻塞）
            prefix_len = first_delim + (  # 计算前缀长度
                prefix_cache_lens[i] if prefix_cache_lens is not None else 0  # 加上前缀缓存长度（如有）
            )
            prefix_len_ptr.append(prefix_len)  # 添加前缀长度

            # Compute relative positions within items using searchsorted (no GPU sync).
            #   suffix_range      = [0, 1, 2, 3, 4, ...]
            #   searchsorted      = bucket index for each position
            #   last_delim        = delimiter offset at start of current bucket
            #   pos_within_item   = suffix_range - last_delim
            # 使用searchsorted计算项内相对位置（无GPU同步）。
            #   suffix_range      = [0, 1, 2, 3, 4, ...]
            #   searchsorted      = 每个位置的桶索引
            #   last_delim        = 当前桶起始的分隔符偏移
            #   pos_within_item   = suffix_range - last_delim
            suffix_len = seq_len - first_delim  # 计算后缀长度
            relative_positions = delimiter_indices - first_delim  # 计算相对位置

            suffix_range = torch.arange(suffix_len, dtype=torch.int64, device=device)  # 创建后缀范围
            bucket_idx = torch.searchsorted(  # 使用searchsorted查找桶索引
                relative_positions, suffix_range, right=True  # 右侧搜索
            )
            last_delim = relative_positions[torch.clamp(bucket_idx - 1, min=0)]  # 获取最后一个分隔符位置
            pos_within_item = suffix_range - last_delim  # 计算项内位置

            token_pos_in_items_ptr.append(pos_within_item.to(torch.uint16))  # 添加项内位置（转换为uint16）

            forward_batch.positions[seq_start + first_delim : seq_end] = (  # 更新位置信息
                prefix_len + pos_within_item - 1  # 前缀长度加项内位置减1
            )

            seq_start = seq_end  # 更新序列起始位置

        # Pad token_pos_in_items_ptr for batch processing
        # 对token_pos_in_items_ptr进行填充以支持批处理
        if token_pos_in_items_ptr:  # 如果有项内位置数据
            token_pos_in_items_len = max(t.numel() for t in token_pos_in_items_ptr)  # 获取最大元素数
            token_pos_in_items_ptr = [  # 对每个张量进行填充
                torch.cat(  # 拼接张量
                    [
                        t,  # 原始张量
                        torch.zeros(  # 零填充
                            token_pos_in_items_len - t.numel(),  # 填充长度
                            dtype=torch.uint16,  # 数据类型
                            device=device,  # 设备
                        ),
                    ]
                )
                for t in token_pos_in_items_ptr  # 遍历每个张量
            ]

        if not prefix_len_ptr or not token_pos_in_items_ptr:  # 如果没有前缀长度或项内位置数据
            return MultiItemScoringParams()  # 返回空的参数

        return MultiItemScoringParams(  # 返回多项评分参数
            prefix_len_ptr=torch.tensor(  # 创建前缀长度张量
                prefix_len_ptr, dtype=torch.uint32, device=device  # 数据和类型
            ),
            token_pos_in_items_ptr=torch.cat(token_pos_in_items_ptr, dim=0),  # 拼接项内位置
            token_pos_in_items_len=token_pos_in_items_len & 0xFFFFFFFF,  # 长度截断为32位
            max_item_len_ptr=torch.stack(  # 创建最大项长度张量
                [
                    t.to(torch.int32).max().to(torch.uint16)  # 获取每个张量的最大值
                    for t in token_pos_in_items_ptr  # 遍历每个张量
                ],
                dim=0,  # 沿第0维堆叠
            ),
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):  # 初始化前向元数据
        if forward_batch.forward_mode.is_decode_or_idle():  # 如果是解码或空闲模式
            self.indices_updater_decode.update(  # 更新解码索引
                forward_batch.req_pool_indices,  # 请求池索引
                forward_batch.seq_lens,  # 序列长度
                forward_batch.seq_lens_cpu,  # CPU序列长度
                forward_batch.seq_lens_sum,  # 序列长度总和
                decode_wrappers=self.decode_wrappers,  # 解码包装器
                encoder_lens=forward_batch.encoder_lens,  # 编码器长度
                spec_info=forward_batch.spec_info,  # 投机解码信息
                fixed_split_size=self.decode_split_tile_size,  # 固定分片大小
                disable_split_kv=False,  # 是否禁用KV分片
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrappers)  # 设置解码元数据
        elif forward_batch.forward_mode.is_draft_extend():  # 如果是草稿扩展模式
            self.indices_updater_prefill.update(  # 更新预填充索引
                forward_batch.req_pool_indices,  # 请求池索引
                forward_batch.seq_lens,  # 序列长度
                forward_batch.seq_lens_cpu,  # CPU序列长度
                forward_batch.seq_lens_sum,  # 序列长度总和
                prefix_lens=None,  # 前缀长度为None
                prefill_wrappers=self.prefill_wrappers_paged,  # 预填充包装器
                use_ragged=False,  # 不使用ragged模式
                encoder_lens=forward_batch.encoder_lens,  # 编码器长度
                spec_info=forward_batch.spec_info,  # 投机解码信息
            )
            self.forward_metadata = PrefillMetadata(  # 设置预填充元数据
                self.prefill_wrappers_paged, False, False  # 包装器、不使用ragged、无前缀
            )
        elif forward_batch.forward_mode.is_target_verify():  # 如果是目标验证模式
            self.indices_updater_prefill.update(  # 更新预填充索引
                forward_batch.req_pool_indices,  # 请求池索引
                forward_batch.seq_lens,  # 序列长度
                forward_batch.seq_lens_cpu,  # CPU序列长度
                forward_batch.seq_lens_sum,  # 序列长度总和
                prefix_lens=None,  # 前缀长度为None
                prefill_wrappers=self.prefill_wrappers_verify,  # 验证预填充包装器
                use_ragged=False,  # 不使用ragged模式
                encoder_lens=forward_batch.encoder_lens,  # 编码器长度
                spec_info=forward_batch.spec_info,  # 投机解码信息
            )
            self.forward_metadata = PrefillMetadata(  # 设置预填充元数据
                self.prefill_wrappers_verify, False, False  # 包装器、不使用ragged、无前缀
            )
        else:  # 否则为普通扩展模式
            prefix_lens = forward_batch.extend_prefix_lens  # 获取扩展前缀长度

            # Disable ragged wrapper and ensure prefix handling for multimodal and multi-item scoring
            # 禁用ragged包装器，确保多模态和多项评分的前缀处理
            if self.is_multimodal or self.enable_mis:  # 如果是多模态或启用MIS
                # use_ragged = False: Multi-item scoring requires the paged wrapper because:
                # 1. Ragged wrapper doesn't support the specialized multi-item parameters
                #    (prefix_len_ptr, token_pos_in_items_ptr, etc.)
                # 2. Paged wrapper provides better control over attention masking needed
                #    for respecting item boundaries in multi-item sequences
                # 3. Custom masking logic conflicts with ragged wrapper's assumptions
                # use_ragged = False: 多项评分需要分页包装器，因为：
                # 1. Ragged包装器不支持专用的多项参数
                #    （prefix_len_ptr、token_pos_in_items_ptr等）
                # 2. 分页包装器为遵守多项序列中项边界所需的注意力掩码提供更好的控制
                # 3. 自定义掩码逻辑与ragged包装器的假设冲突
                use_ragged = False  # 不使用ragged模式
                extend_no_prefix = False  # 不使用无前缀扩展
            else:  # 否则
                use_ragged = (  # 判断是否使用ragged模式
                    not self.enable_deterministic  # 未启用确定性推理
                    and not is_in_piecewise_cuda_graph()  # 不在分段CUDA图中
                    and not self.use_paged  # 未使用分页模式
                )
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)  # 是否无前缀扩展

            # Process multi-item scoring in attention backend instead of ForwardBatch
            # 在注意力后端而非ForwardBatch中处理多项评分
            multi_item_params = MultiItemScoringParams()  # 创建空的多项评分参数
            if self.enable_mis:  # 如果启用MIS
                # Use new backend-specific implementation
                # 使用新的后端特定实现
                multi_item_params = self._process_multi_item_scoring(forward_batch)  # 处理多项评分

            self.indices_updater_prefill.update(  # 更新预填充索引
                forward_batch.req_pool_indices,  # 请求池索引
                forward_batch.seq_lens,  # 序列长度
                forward_batch.seq_lens_cpu,  # CPU序列长度
                forward_batch.seq_lens_sum,  # 序列长度总和
                prefix_lens,  # 前缀长度
                prefill_wrappers=self.prefill_wrappers_paged,  # 预填充包装器
                use_ragged=use_ragged,  # 是否使用ragged模式
                encoder_lens=forward_batch.encoder_lens,  # 编码器长度
                spec_info=None,  # 无投机解码信息
                fixed_split_size=self.prefill_split_tile_size,  # 固定分片大小
                multi_item_params=multi_item_params,  # 多项评分参数
                cross_attention_custom_mask=forward_batch.cross_attention_custom_mask,  # 交叉注意力自定义掩码
            )
            self.forward_metadata = PrefillMetadata(  # 设置预填充元数据
                self.prefill_wrappers_paged,  # 预填充包装器
                use_ragged,  # 是否使用ragged模式
                extend_no_prefix,  # 是否无前缀扩展
                multi_item_params,  # 多项评分参数
            )

    def init_cuda_graph_state(  # 初始化CUDA图状态
        self,
        max_bs: int,  # 最大批次大小
        max_num_tokens: int,  # 最大token数
        kv_indices_buf: Optional[torch.Tensor] = None,  # KV索引缓冲区（可选）
    ):
        if kv_indices_buf is None:  # 如果未提供KV索引缓冲区
            cuda_graph_kv_indices = torch.zeros(  # 创建零张量作为CUDA图KV索引
                (max_num_tokens * self.max_context_len,),  # 形状
                dtype=torch.int32,  # 数据类型
                device="cuda",  # 设备
            )
        else:  # 否则
            cuda_graph_kv_indices = kv_indices_buf  # 使用提供的缓冲区

        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [  # 创建CUDA图KV索引列表
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)  # 为额外包装器克隆
        ]

        # Ensure tensors are properly allocated
        # 确保张量已正确分配
        for i in range(self.num_wrappers):  # 遍历每个包装器
            # Force allocation by performing a small operation
            # 通过执行小操作强制分配
            if len(self.cuda_graph_kv_indices[i]) > 0:  # 如果索引长度大于0
                self.cuda_graph_kv_indices[i][0] = 0  # 写入0以强制分配

        if not self.skip_prefill:  # 如果不跳过预填充
            self.cuda_graph_custom_mask = torch.zeros(  # 创建CUDA图自定义掩码
                (max_num_tokens * self.max_context_len),  # 形状
                dtype=torch.uint8,  # 数据类型
                device="cuda",  # 设备
            )
            self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]  # 克隆KV索引指针作为QK索引指针
            self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]  # 克隆KV索引指针作为QO索引指针

    def _create_decode_wrappers(self, bs: int, num_tokens: int) -> list:  # 创建解码包装器（内部方法）
        return [  # 返回解码包装器列表
            BatchDecodeWithPagedKVCacheWrapper(  # 创建分页解码包装器
                self.workspace_buffer,  # 工作区缓冲区
                "NHD",  # 布局
                backend=self.decode_backend,  # 后端
                use_cuda_graph=True,  # 启用CUDA图
                use_tensor_cores=self.decode_use_tensor_cores,  # 是否使用Tensor Core
                paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],  # 分页KV索引指针缓冲区
                paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],  # 分页KV索引缓冲区
                paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],  # 分页KV最后页长度缓冲区
            )
            for i in range(self.num_wrappers)  # 为每个包装器创建实例
        ]

    def _create_prefill_wrappers(self, bs: int, use_custom_mask: bool = False) -> list:  # 创建预填充包装器（内部方法）
        # FlashInfer's prefill wrapper decides mask mode based on whether
        # `custom_mask_buf` is initialized (not whether a custom mask is provided).
        # For cases like DFLASH draft (ENCODER_ONLY / non-causal) we do NOT use a
        # custom mask, so we must avoid initializing `custom_mask_buf`, otherwise
        # FlashInfer will treat the (zero) buffer as a real mask and block attention.
        # FlashInfer的预填充包装器根据`custom_mask_buf`是否已初始化（而非是否提供了自定义掩码）
        # 来决定掩码模式。对于DFLASH草稿（ENCODER_ONLY / 非因果）等情况，
        # 我们不使用自定义掩码，因此必须避免初始化`custom_mask_buf`，
        # 否则FlashInfer会将（零）缓冲区视为真实掩码并阻止注意力。
        wrappers = []  # 初始化包装器列表
        for i in range(self.num_wrappers):  # 遍历每个包装器
            extra = (  # 计算额外参数
                {
                    "custom_mask_buf": self.cuda_graph_custom_mask,  # 自定义掩码缓冲区
                    "mask_indptr_buf": self.cuda_graph_qk_indptr[i][: bs + 1],  # 掩码索引指针缓冲区
                }
                if use_custom_mask  # 如果使用自定义掩码
                else {}  # 否则为空
            )
            wrappers.append(  # 添加预填充包装器
                BatchPrefillWithPagedKVCacheWrapper(  # 创建分页预填充包装器
                    self.workspace_buffer,  # 工作区缓冲区
                    "NHD",  # 布局
                    use_cuda_graph=True,  # 启用CUDA图
                    backend=self.prefill_backend,  # 后端
                    qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],  # QO索引指针缓冲区
                    paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],  # 分页KV索引指针缓冲区
                    paged_kv_indices_buf=self.cuda_graph_kv_indices[i],  # 分页KV索引缓冲区
                    paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],  # 分页KV最后页长度缓冲区
                    **extra,  # 额外参数
                )
            )
        return wrappers  # 返回包装器列表

    def _prepare_cuda_graph_metadata(  # 准备CUDA图元数据（内部方法）
        self,
        bs: int,  # 批次大小
        num_tokens: int,  # token数
        forward_mode: ForwardMode,  # 前向模式
        spec_info: Optional[SpecInput],  # 投机解码信息
    ) -> None:
        if forward_mode.is_decode_or_idle():  # 如果是解码或空闲模式
            decode_wrappers = self._create_decode_wrappers(bs, num_tokens)  # 创建解码包装器
            self.decode_cuda_graph_metadata[bs] = decode_wrappers  # 保存解码CUDA图元数据
            self.forward_metadata = DecodeMetadata(decode_wrappers)  # 设置解码元数据
        elif (  # 否则如果是以下模式之一
            forward_mode.is_target_verify()  # 目标验证
            or forward_mode.is_draft_extend()  # 草稿扩展
            or forward_mode.is_dllm_extend()  # DLLM扩展
        ):
            use_custom_mask = (  # 判断是否使用自定义掩码
                forward_mode.is_target_verify()  # 是目标验证模式
                and spec_info is not None  # 投机信息存在
                and getattr(spec_info, "custom_mask", None) is not None  # 且有自定义掩码
            )
            prefill_wrappers = self._create_prefill_wrappers(bs, use_custom_mask)  # 创建预填充包装器
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers  # 保存预填充CUDA图元数据
            self.forward_metadata = PrefillMetadata(  # 设置预填充元数据
                prefill_wrappers, forward_mode.is_dllm_extend(), False  # 包装器、是否DLLM扩展、无前缀
            )
        else:  # 否则
            raise ValueError(f"Invalid mode: {forward_mode=}")  # 抛出无效模式异常

    def init_forward_metadata_capture_cuda_graph(  # 初始化CUDA图捕获时的前向元数据
        self,
        bs: int,  # 批次大小
        num_tokens: int,  # token数
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        forward_mode: ForwardMode,  # 前向模式
        spec_info: Optional[SpecInput],  # 投机解码信息
    ):
        seq_lens_sum = seq_lens.sum().item()  # 计算序列长度总和
        seq_lens_cpu = seq_lens.cpu()  # 将序列长度转移到CPU
        self._prepare_cuda_graph_metadata(bs, num_tokens, forward_mode, spec_info)  # 准备CUDA图元数据
        self.init_forward_metadata_replay_cuda_graph(  # 调用重放初始化方法
            bs=bs,  # 批次大小
            req_pool_indices=req_pool_indices,  # 请求池索引
            seq_lens=seq_lens,  # 序列长度
            seq_lens_sum=seq_lens_sum,  # 序列长度总和
            encoder_lens=encoder_lens,  # 编码器长度
            forward_mode=forward_mode,  # 前向模式
            spec_info=spec_info,  # 投机解码信息
            seq_lens_cpu=seq_lens_cpu,  # CPU序列长度
        )
        # fast_decode_plan requires _cached_module set by the initial full
        # begin_forward call above; install it only after that first plan runs.
        # fast_decode_plan需要由上面初始完整begin_forward调用设置的_cached_module；
        # 仅在第一次plan运行后才安装它。
        if forward_mode.is_decode_or_idle():  # 如果是解码或空闲模式
            for w in self.decode_cuda_graph_metadata[bs]:  # 遍历解码CUDA图元数据中的包装器
                w.begin_forward = partial(fast_decode_plan, w)  # 将begin_forward替换为fast_decode_plan的偏函数

    def init_forward_metadata_replay_cuda_graph(  # 初始化CUDA图重放时的前向元数据
        self,
        bs: int,  # 批次大小
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        seq_lens_sum: int,  # 序列长度总和
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        forward_mode: ForwardMode,  # 前向模式
        spec_info: Optional[SpecInput],  # 投机解码信息
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度
    ):
        if forward_mode.is_decode_or_idle():  # 如果是解码或空闲模式
            self.indices_updater_decode.update(  # 更新解码索引
                req_pool_indices[:bs],  # 截取请求池索引
                seq_lens[:bs],  # 截取序列长度
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,  # 截取CPU序列长度
                seq_lens_sum,  # 序列长度总和
                decode_wrappers=self.decode_cuda_graph_metadata[bs],  # 解码CUDA图元数据
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,  # 截取编码器长度
                spec_info=spec_info,  # 投机解码信息
                fixed_split_size=None,  # 无固定分片大小
                disable_split_kv=self.disable_cuda_graph_kv_split,  # 是否禁用KV分片
            )
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():  # 目标验证或草稿扩展模式
            self.indices_updater_prefill.update(  # 更新预填充索引
                req_pool_indices[:bs],  # 截取请求池索引
                seq_lens[:bs],  # 截取序列长度
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,  # 截取CPU序列长度
                seq_lens_sum,  # 序列长度总和
                prefix_lens=None,  # 前缀长度为None
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],  # 预填充CUDA图元数据
                use_ragged=False,  # 不使用ragged模式
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,  # 截取编码器长度
                spec_info=spec_info,  # 投机解码信息
            )
        elif forward_mode.is_dllm_extend():  # DLLM扩展模式
            self.indices_updater_prefill.update(  # 更新预填充索引
                req_pool_indices[:bs],  # 截取请求池索引
                seq_lens[:bs],  # 截取序列长度
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,  # 截取CPU序列长度
                seq_lens_sum,  # 序列长度总和
                prefix_lens=seq_lens - self.dllm_config.block_size,  # 前缀长度为序列长度减块大小
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],  # 预填充CUDA图元数据
                use_ragged=not self.use_paged,  # 是否使用ragged（非分页模式时使用）
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,  # 截取编码器长度
                spec_info=None,  # 无投机解码信息
            )
        else:  # 否则
            raise ValueError("Invalid forward mode")  # 抛出无效前向模式异常

    def get_cuda_graph_seq_len_fill_value(self):  # 获取CUDA图序列长度填充值
        return 1  # 返回1（解码模式下填充为1）

    @debug_kernel_api  # 内核API调试装饰器
    def forward_extend(  # 扩展（预填充）前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 基数注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存，默认为True
    ):
        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[  # 获取分页预填充包装器
            self._get_wrapper_idx(layer)  # 根据层类型获取包装器索引
        ]
        cache_loc = (  # 确定缓存位置
            forward_batch.out_cache_loc  # 非交叉注意力使用输出缓存位置
            if not layer.is_cross_attention  # 如果不是交叉注意力
            else forward_batch.encoder_out_cache_loc  # 否则使用编码器输出缓存位置
        )

        logits_soft_cap = layer.logit_cap  # 获取logits上限值

        q = q.contiguous()  # 确保查询张量是连续的
        if not self.forward_metadata.use_ragged:  # 如果不使用ragged模式
            if k is not None:  # 如果键张量不为空
                assert v is not None  # 断言值张量也不为空
                if save_kv_cache:  # 如果需要保存KV缓存
                    self.token_to_kv_pool.set_kv_buffer(  # 设置KV缓存
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale  # 层、缓存位置、K、V、缩放因子
                    )

            causal = (  # 判断是否使用因果注意力
                not layer.is_cross_attention  # 非交叉注意力
                and layer.attn_type != AttentionType.ENCODER_ONLY  # 且非仅编码器类型
            )
            o = prefill_wrapper_paged.forward(  # 调用分页预填充前向传播
                q.view(-1, layer.tp_q_head_num, layer.head_dim),  # 重塑查询张量形状
                self.token_to_kv_pool.get_kv_buffer(layer.layer_id),  # 获取KV缓存
                causal=causal,  # 是否因果
                sm_scale=layer.scaling,  # 缩放因子
                # Disable sliding window attention for multi-item scoring:
                # - Sliding window could cut across item boundaries, breaking semantic coherence
                # - Multi-item sequences need full attention to properly handle delimiter tokens
                # - Specialized multi-item parameters (prefix_len_ptr, token_pos_in_items_ptr)
                #   provide more precise attention control than simple sliding windows
                # - Item-aware masking takes precedence over window-based masking
                # 禁用多项评分的滑动窗口注意力：
                # - 滑动窗口可能跨越项边界，破坏语义连贯性
                # - 多项序列需要全注意力以正确处理分隔符token
                # - 专用的多项参数（prefix_len_ptr、token_pos_in_items_ptr）
                #   比简单的滑动窗口提供更精确的注意力控制
                # - 项感知掩码优先于基于窗口的掩码
                window_left=(  # 窗口左边界
                    layer.sliding_window_size  # 使用层的滑动窗口大小
                    if not (  # 如果不满足以下条件
                        self.forward_metadata.multi_item_params  # 有多项评分参数
                        and self.forward_metadata.multi_item_params.is_enabled()  # 且已启用
                    )
                    else -1  # 否则设为-1（禁用滑动窗口）
                ),
                logits_soft_cap=logits_soft_cap,  # logits上限
                # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                # 必须使用_float以避免破坏CUDA图捕获的设备到主机拷贝。
                k_scale=layer.k_scale_float,  # 键缩放因子（浮点）
                v_scale=layer.v_scale_float,  # 值缩放因子（浮点）
            )
        else:  # 否则使用ragged模式
            # If `k`/`v` are not explicitly provided, fall back to the KV cache stored in
            # `self.token_to_kv_pool` for this layer. This enables attention over
            # previously cached context without re-materializing KV tensors (e.g., the
            # IQuestLoopCoder path uses token_to_kv_pool as the KV source).
            # 如果未显式提供`k`/`v`，则回退到存储在`self.token_to_kv_pool`中
            # 此层的KV缓存。这允许对先前缓存的上下文进行注意力计算，
            # 而无需重新物化KV张量（例如，IQuestLoopCoder路径使用token_to_kv_pool作为KV源）。
            if k is None and v is None:  # 如果K和V都为空
                k = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)[0]  # 从KV池获取K
                v = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)[1]  # 从KV池获取V
            causal = True  # 默认使用因果注意力
            if (  # 如果满足以下条件
                layer.is_cross_attention  # 是交叉注意力
                or layer.attn_type == AttentionType.ENCODER_ONLY  # 或是仅编码器类型
            ):
                causal = False  # 禁用因果注意力
            if not self.is_dllm_model and layer.attn_type == AttentionType.ENCODER_ONLY:  # 非DLLM且仅编码器
                save_kv_cache = False  # 不保存KV缓存

            if self.forward_metadata.extend_no_prefix:  # 如果扩展无前缀
                # NOTE: FlashInfer currently has limitations with head_dim = 32 or other dimensions
                # The FlashInfer head_dim limitation itself is tracked here:
                # https://github.com/flashinfer-ai/flashinfer/issues/1048
                # 注意：FlashInfer目前在head_dim = 32或其他维度上有限制
                # FlashInfer的head_dim限制本身在此处跟踪：
                # https://github.com/flashinfer-ai/flashinfer/issues/1048
                o = self.prefill_wrapper_ragged.forward(  # 调用ragged预填充前向传播
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),  # 重塑查询张量
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),  # 重塑键张量
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),  # 重塑值张量
                    causal=causal,  # 是否因果
                    sm_scale=layer.scaling,  # 缩放因子
                    logits_soft_cap=logits_soft_cap,  # logits上限
                )

            else:  # 否则扩展有前缀
                swa_window_left = (  # 计算滑动窗口左边界
                    layer.sliding_window_size  # 使用层的滑动窗口大小
                    if not (  # 如果不满足以下条件
                        self.forward_metadata.multi_item_params  # 有多项评分参数
                        and self.forward_metadata.multi_item_params.is_enabled()  # 且已启用
                    )
                    else -1  # 否则设为-1（禁用滑动窗口）
                )
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(  # 调用ragged预填充前向传播（返回LSE）
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),  # 重塑查询张量
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),  # 重塑键张量
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),  # 重塑值张量
                    causal=causal,  # 是否因果
                    sm_scale=layer.scaling,  # 缩放因子
                    window_left=swa_window_left,  # 滑动窗口左边界
                    logits_soft_cap=logits_soft_cap,  # logits上限
                )
                o2, s2 = prefill_wrapper_paged.forward_return_lse(  # 调用分页预填充前向传播（返回LSE）
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),  # 重塑查询张量
                    self.token_to_kv_pool.get_kv_buffer(layer.layer_id),  # 获取KV缓存
                    causal=False,  # 非因果（已缓存部分）
                    sm_scale=layer.scaling,  # 缩放因子
                    window_left=swa_window_left,  # 滑动窗口左边界
                    logits_soft_cap=logits_soft_cap,  # logits上限
                )

                o, _ = _safe_merge_state(o1, s1, o2, s2)  # 安全合并两个状态

            if save_kv_cache:  # 如果需要保存KV缓存
                self.token_to_kv_pool.set_kv_buffer(  # 设置KV缓存
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale  # 层、缓存位置、K、V、缩放因子
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)  # 重塑输出张量并返回

    @debug_kernel_api  # 内核API调试装饰器
    def forward_decode(  # 解码前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 基数注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存，默认为True
    ):
        decode_wrapper = self.forward_metadata.decode_wrappers[  # 获取解码包装器
            self._get_wrapper_idx(layer)  # 根据层类型获取包装器索引
        ]
        cache_loc = (  # 确定缓存位置
            forward_batch.out_cache_loc  # 非交叉注意力使用输出缓存位置
            if not layer.is_cross_attention  # 如果不是交叉注意力
            else forward_batch.encoder_out_cache_loc  # 否则使用编码器输出缓存位置
        )

        if k is not None:  # 如果键张量不为空
            assert v is not None  # 断言值张量也不为空
            if save_kv_cache:  # 如果需要保存KV缓存
                self.token_to_kv_pool.set_kv_buffer(  # 设置KV缓存
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale  # 层、缓存位置、K、V、缩放因子
                )

        # Call the wrapped function
        # 调用包装函数
        o = decode_wrapper.forward(  # 调用解码前向传播
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),  # 重塑查询张量
            self.token_to_kv_pool.get_kv_buffer(layer.layer_id),  # 获取KV缓存
            sm_scale=layer.scaling,  # 缩放因子
            logits_soft_cap=layer.logit_cap,  # logits上限
            # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
            # 必须使用_float以避免破坏CUDA图捕获的设备到主机拷贝。
            k_scale=layer.k_scale_float,  # 键缩放因子（浮点）
            v_scale=layer.v_scale_float,  # 值缩放因子（浮点）
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)  # 重塑输出张量并返回

    def _get_wrapper_idx(self, layer: RadixAttention):  # 获取包装器索引（内部方法）
        if self.num_wrappers == 1:  # 如果只有一个包装器
            return 0  # 返回索引0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:  # 如果是滑动窗口分发
            return layer.sliding_window_size == -1  # 全注意力返回1，滑动窗口返回0
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:  # 如果是交叉注意力分发
            return layer.is_cross_attention  # 交叉注意力返回1，自注意力返回0

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")  # 抛出未知分发原因异常


class FlashInferIndicesUpdaterDecode:  # FlashInfer解码索引更新器类
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):  # 初始化方法
        # Parse Constants
        # 解析常量
        self.num_qo_heads = (  # 计算查询/输出头数
            model_runner.model_config.num_attention_heads // get_attention_tp_size()  # 总头数除以TP大小
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(  # 计算KV头数
            get_attention_tp_size()  # TP大小
        )
        self.head_dim = model_runner.model_config.head_dim  # 保存头维度
        self.data_type = model_runner.kv_cache_dtype  # 保存KV缓存数据类型
        self.q_data_type = model_runner.dtype  # 保存查询数据类型
        self.sliding_window_size = model_runner.sliding_window_size  # 保存滑动窗口大小
        self.attn_backend = attn_backend  # 保存注意力后端引用

        # Buffers and wrappers
        # 缓冲区和包装器
        self.kv_indptr = attn_backend.kv_indptr  # 保存KV索引指针
        self.kv_last_page_len = attn_backend.kv_last_page_len  # 保存KV最后页长度
        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 保存请求到token映射表
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator  # 保存KV池分配器

        # Dispatch the update function
        # 分发更新函数
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:  # 滑动窗口
            self.update = self.update_sliding_window  # 使用滑动窗口更新方法
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:  # 交叉注意力
            self.update = self.update_cross_attention  # 使用交叉注意力更新方法
        else:  # 否则
            assert self.attn_backend.num_wrappers == 1  # 断言只有一个包装器
            self.update = self.update_single_wrapper  # 使用单包装器更新方法

    def update(  # 更新方法（占位符，运行时被替换）
        self,
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度
        seq_lens_sum: int,  # 序列长度总和
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],  # 解码包装器
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        spec_info: Optional[SpecInput],  # 投机解码信息
        fixed_split_size: Optional[int] = None,  # 固定分片大小
        disable_split_kv: Optional[bool] = None,  # 是否禁用KV分片
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        # 保留签名用于类型检查。运行时会被赋值。
        raise NotImplementedError()  # 抛出未实现异常

    def update_single_wrapper(  # 单包装器更新方法
        self,
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度
        seq_lens_sum: int,  # 序列长度总和
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],  # 解码包装器
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        spec_info: Optional[SpecInput],  # 投机解码信息
        fixed_split_size: Optional[int] = None,  # 固定分片大小
        disable_split_kv: Optional[bool] = None,  # 是否禁用KV分片
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers  # 使用提供的或默认的包装器
        self.call_begin_forward(  # 调用begin_forward
            decode_wrappers[0],  # 第一个解码包装器
            req_pool_indices,  # 请求池索引
            seq_lens,  # 序列长度
            seq_lens_sum,  # 序列长度总和
            self.kv_indptr[0],  # KV索引指针
            None,  # 无KV起始索引
            spec_info,  # 投机解码信息
            seq_lens_cpu,  # CPU序列长度
            fixed_split_size=fixed_split_size,  # 固定分片大小
            disable_split_kv=disable_split_kv,  # 是否禁用KV分片
        )

    def update_sliding_window(  # 滑动窗口更新方法
        self,
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度
        seq_lens_sum: int,  # 序列长度总和
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],  # 解码包装器
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        spec_info: Optional[SpecInput],  # 投机解码信息
        fixed_split_size: Optional[int] = None,  # 固定分片大小
        disable_split_kv: Optional[bool] = None,  # 是否禁用KV分片
    ):
        assert self.sliding_window_size is not None  # 断言滑动窗口大小不为空
        for wrapper_id in range(2):  # 遍历两个包装器
            if wrapper_id == 0:  # 第一个包装器（滑动窗口注意力）
                # Sliding window attention
                # 滑动窗口注意力
                paged_kernel_lens_tmp = torch.clamp(  # 将序列长度钳位到窗口大小
                    seq_lens, max=self.sliding_window_size + 1  # 最大值为窗口大小+1
                )
                if seq_lens_cpu is not None:  # 如果CPU序列长度存在
                    seq_lens_cpu_tmp = torch.clamp(  # 钳位CPU序列长度
                        seq_lens_cpu, max=self.sliding_window_size + 1  # 最大值为窗口大小+1
                    )
                    paged_kernel_lens_sum_tmp = seq_lens_cpu_tmp.sum().item()  # 计算总和
                else:  # 否则
                    paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()  # 从GPU计算总和
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp  # 计算KV起始索引
            else:  # 第二个包装器（全注意力）
                # Full attention
                # 全注意力
                paged_kernel_lens_tmp = seq_lens  # 使用完整序列长度
                paged_kernel_lens_sum_tmp = seq_lens_sum  # 使用完整序列长度总和
                seq_lens_cpu_tmp = seq_lens_cpu  # 使用CPU序列长度
                kv_start_idx_tmp = None  # 无KV起始索引

            use_sliding_window_kv_pool = wrapper_id == 0 and isinstance(  # 判断是否使用滑动窗口KV池
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator  # 检查分配器类型
            )

            self.call_begin_forward(  # 调用begin_forward
                decode_wrappers[wrapper_id],  # 当前解码包装器
                req_pool_indices,  # 请求池索引
                paged_kernel_lens_tmp,  # 分页内核长度
                paged_kernel_lens_sum_tmp,  # 分页内核长度总和
                self.kv_indptr[wrapper_id],  # KV索引指针
                kv_start_idx_tmp,  # KV起始索引
                spec_info,  # 投机解码信息
                seq_lens_cpu=seq_lens_cpu_tmp,  # CPU序列长度
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,  # 是否使用滑动窗口KV池
                fixed_split_size=fixed_split_size,  # 固定分片大小
                disable_split_kv=disable_split_kv,  # 是否禁用KV分片
            )

    def update_cross_attention(  # 交叉注意力更新方法
        self,
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度
        seq_lens_sum: int,  # 序列长度总和
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],  # 解码包装器
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        spec_info: Optional[SpecInput],  # 投机解码信息
        fixed_split_size: Optional[int] = None,  # 固定分片大小
        disable_split_kv: Optional[bool] = None,  # 是否禁用KV分片
    ):
        # Cache encoder_lens on CPU to avoid GPU→CPU transfer per call
        # 将encoder_lens缓存到CPU，避免每次调用时的GPU→CPU传输
        encoder_lens_cpu = encoder_lens.cpu() if encoder_lens is not None else None  # 将编码器长度转到CPU
        for wrapper_id in range(2):  # 遍历两个包装器
            if wrapper_id == 0:  # 第一个包装器（自注意力）
                paged_kernel_lens = seq_lens  # 使用序列长度
                kv_start_idx = encoder_lens  # KV起始索引为编码器长度
                kv_lens_cpu = seq_lens_cpu  # CPU序列长度
            else:  # 第二个包装器（交叉注意力）
                # Cross-attention: attend to encoder tokens only
                # 交叉注意力：仅关注编码器token
                paged_kernel_lens = encoder_lens  # 使用编码器长度
                kv_start_idx = torch.zeros_like(encoder_lens)  # KV起始索引为零
                seq_lens_sum = encoder_lens.sum().item()  # 计算编码器长度总和
                kv_lens_cpu = encoder_lens_cpu  # 使用CPU编码器长度

            self.call_begin_forward(  # 调用begin_forward
                decode_wrappers[wrapper_id],  # 当前解码包装器
                req_pool_indices,  # 请求池索引
                paged_kernel_lens,  # 分页内核长度
                seq_lens_sum,  # 序列长度总和
                self.kv_indptr[wrapper_id],  # KV索引指针
                kv_start_idx,  # KV起始索引
                spec_info,  # 投机解码信息
                seq_lens_cpu=kv_lens_cpu,  # CPU KV长度
                fixed_split_size=fixed_split_size,  # 固定分片大小
                disable_split_kv=disable_split_kv,  # 是否禁用KV分片
            )

    def call_begin_forward(  # 调用begin_forward的核心方法
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,  # 解码包装器
        req_pool_indices: torch.Tensor,  # 请求池索引
        paged_kernel_lens: torch.Tensor,  # 分页内核长度
        paged_kernel_lens_sum: int,  # 分页内核长度总和
        kv_indptr: torch.Tensor,  # KV索引指针
        kv_start_idx: torch.Tensor,  # KV起始索引
        spec_info: Optional[SpecInput],  # 投机解码信息
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度
        use_sliding_window_kv_pool: bool = False,  # 是否使用滑动窗口KV池
        fixed_split_size: Optional[int] = None,  # 固定分片大小
        disable_split_kv: Optional[bool] = None,  # 是否禁用KV分片
    ):
        if spec_info is None:  # 如果没有投机信息
            bs = len(req_pool_indices)  # 批次大小等于请求池索引长度
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)  # 计算累积和作为索引指针
            kv_indptr = kv_indptr[: bs + 1]  # 截取当前批次大小的索引指针

            if wrapper.is_cuda_graph_enabled:  # 如果启用了CUDA图
                # Directly write to the cuda graph input buffer
                # 直接写入CUDA图输入缓冲区
                kv_indices = wrapper._paged_kv_indices_buf  # 使用CUDA图的KV索引缓冲区
            else:  # 否则
                kv_indices = torch.empty(  # 创建空的KV索引张量
                    paged_kernel_lens_sum, dtype=torch.int32, device="cuda"  # 形状、类型、设备
                )

            create_flashinfer_kv_indices_triton[(bs,)](  # 调用Triton核函数创建KV索引
                self.req_to_token,  # 请求到token映射表
                req_pool_indices,  # 请求池索引
                paged_kernel_lens,  # 分页内核长度
                kv_indptr,  # KV索引指针
                kv_start_idx,  # KV起始索引
                kv_indices,  # KV索引输出
                self.req_to_token.shape[1],  # 映射表列数
            )
        else:  # 否则有投机信息
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices  # 从投机信息获取
            bs = kv_indptr.shape[0] - 1  # 从索引指针推导批次大小

        if use_sliding_window_kv_pool:  # 如果使用滑动窗口KV池
            kv_last_index = kv_indptr[-1]  # 获取最后一个索引
            kv_indices[:kv_last_index] = (  # 转换KV索引
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(  # 从全量位置转换为SWA位置
                    kv_indices[:kv_last_index]  # 需要转换的索引
                )
            )

        global global_override_indptr_cpu  # 声明全局覆盖indptr CPU缓冲区
        locally_override = False  # 本地覆盖标志初始化为False
        if seq_lens_cpu is not None and global_override_indptr_cpu is None:  # 如果CPU序列长度存在且全局覆盖未初始化
            locally_override = True  # 设置本地覆盖标志
            global_override_indptr_cpu = torch.empty_like(kv_indptr, device="cpu")  # 创建CPU上的indptr缓冲区
            global_override_indptr_cpu[0] = 0  # 第一个元素设为0
            global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(seq_lens_cpu, dim=0)  # 计算CPU累积和

        # Check if this specific wrapper's begin_forward has been replaced with fast_decode_plan
        # by checking if it's a partial function with fast_decode_plan as the func
        # 检查此特定包装器的begin_forward是否已被替换为fast_decode_plan
        # 通过检查它是否是以fast_decode_plan为func的偏函数
        wrapper_uses_fast_decode_plan = (  # 判断包装器是否使用fast_decode_plan
            hasattr(wrapper.begin_forward, "func")  # 检查是否有func属性
            and wrapper.begin_forward.func == fast_decode_plan  # 且func为fast_decode_plan
        )

        if wrapper_uses_fast_decode_plan:  # 如果使用fast_decode_plan
            # When begin_forward is replaced with fast_decode_plan, pass global_override_indptr_cpu
            # 当begin_forward被替换为fast_decode_plan时，传入global_override_indptr_cpu
            wrapper.begin_forward(  # 调用fast_decode_plan
                kv_indptr,  # KV索引指针
                kv_indices,  # KV索引
                self.kv_last_page_len[:bs],  # KV最后页长度
                self.num_qo_heads,  # 查询/输出头数
                self.num_kv_heads,  # KV头数
                self.head_dim,  # 头维度
                1,  # 页大小
                data_type=self.data_type,  # 数据类型
                q_data_type=self.q_data_type,  # 查询数据类型
                non_blocking=True,  # 非阻塞传输
                fixed_split_size=fixed_split_size,  # 固定分片大小
                disable_split_kv=(  # 是否禁用KV分片
                    disable_split_kv if disable_split_kv is not None else False  # 使用传入值或默认False
                ),
                global_override_indptr_cpu=global_override_indptr_cpu,  # 全局覆盖indptr CPU缓冲区
            )
        else:  # 否则
            # When using original begin_forward, don't pass global_override_indptr_cpu
            # 使用原始begin_forward时，不传入global_override_indptr_cpu
            wrapper.begin_forward(  # 调用原始begin_forward
                kv_indptr,  # KV索引指针
                kv_indices,  # KV索引
                self.kv_last_page_len[:bs],  # KV最后页长度
                self.num_qo_heads,  # 查询/输出头数
                self.num_kv_heads,  # KV头数
                self.head_dim,  # 头维度
                1,  # 页大小
                data_type=self.data_type,  # 数据类型
                q_data_type=self.q_data_type,  # 查询数据类型
                non_blocking=True,  # 非阻塞传输
                fixed_split_size=fixed_split_size,  # 固定分片大小
                disable_split_kv=(  # 是否禁用KV分片
                    disable_split_kv if disable_split_kv is not None else False  # 使用传入值或默认False
                ),
            )

        if locally_override:  # 如果使用了本地覆盖
            global_override_indptr_cpu = None  # 清除全局覆盖缓冲区


class FlashInferIndicesUpdaterPrefill:  # FlashInfer预填充索引更新器类
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):  # 初始化方法
        # Parse Constants
        # 解析常量
        self.num_qo_heads = (  # 计算查询/输出头数
            model_runner.model_config.num_attention_heads // get_attention_tp_size()  # 总头数除以TP大小
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(  # 计算KV头数
            get_attention_tp_size()  # TP大小
        )
        self.head_dim = model_runner.model_config.head_dim  # 保存头维度
        self.data_type = model_runner.kv_cache_dtype  # 保存KV缓存数据类型
        self.q_data_type = model_runner.dtype  # 保存查询数据类型
        self.sliding_window_size = model_runner.sliding_window_size  # 保存滑动窗口大小
        self.attn_backend = attn_backend  # 保存注意力后端引用
        # Buffers and wrappers
        # 缓冲区和包装器
        self.kv_indptr = attn_backend.kv_indptr  # 保存KV索引指针
        self.kv_last_page_len = attn_backend.kv_last_page_len  # 保存KV最后页长度
        self.qo_indptr = attn_backend.qo_indptr  # 保存QO索引指针
        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 保存请求到token映射表
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator  # 保存KV池分配器
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged  # 保存ragged预填充包装器

        # Dispatch the update function
        # 分发更新函数
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:  # 滑动窗口
            self.update = self.update_sliding_window  # 使用滑动窗口更新方法
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:  # 交叉注意力
            self.update = self.update_cross_attention  # 使用交叉注意力更新方法
        else:  # 否则
            assert self.attn_backend.num_wrappers == 1  # 断言只有一个包装器
            self.update = self.update_single_wrapper  # 使用单包装器更新方法

    def update(  # 更新方法（占位符，运行时被替换）
        self,
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度
        seq_lens_sum: int,  # 序列长度总和
        prefix_lens: torch.Tensor,  # 前缀长度
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],  # 预填充包装器
        use_ragged: bool,  # 是否使用ragged模式
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        spec_info: Optional[SpecInput],  # 投机解码信息
        fixed_split_size: Optional[int] = None,  # 固定分片大小
        multi_item_params: Optional[MultiItemScoringParams] = None,  # 多项评分参数
        cross_attention_custom_mask: Optional[torch.Tensor] = None,  # 交叉注意力自定义掩码
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        # 保留签名用于类型检查。运行时会被赋值。
        raise NotImplementedError()  # 抛出未实现异常

    def update_single_wrapper(  # 单包装器更新方法
        self,
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度
        seq_lens_sum: int,  # 序列长度总和
        prefix_lens: torch.Tensor,  # 前缀长度
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],  # 预填充包装器
        use_ragged: bool,  # 是否使用ragged模式
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        spec_info: Optional[SpecInput],  # 投机解码信息
        fixed_split_size: Optional[int] = None,  # 固定分片大小
        multi_item_params: Optional[MultiItemScoringParams] = None,  # 多项评分参数
        cross_attention_custom_mask: Optional[torch.Tensor] = None,  # 交叉注意力自定义掩码
    ):
        if use_ragged:  # 如果使用ragged模式
            # TODO: remove this device sync, we can use forward_batch.extend_prefix_lens_cpu
            # and forward_batch.extend_seq_lens_cpu
            # 待办：移除此设备同步，可以使用forward_batch.extend_prefix_lens_cpu
            # 和forward_batch.extend_seq_lens_cpu
            paged_kernel_lens = prefix_lens  # 分页内核长度为前缀长度
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()  # 计算总和（触发GPU→CPU同步）
        else:  # 否则
            paged_kernel_lens = seq_lens  # 分页内核长度为序列长度
            paged_kernel_lens_sum = seq_lens_sum  # 使用序列长度总和

        self.call_begin_forward(  # 调用begin_forward
            self.prefill_wrapper_ragged,  # ragged预填充包装器
            prefill_wrappers[0],  # 第一个分页预填充包装器
            req_pool_indices,  # 请求池索引
            paged_kernel_lens,  # 分页内核长度
            paged_kernel_lens_sum,  # 分页内核长度总和
            seq_lens,  # 序列长度
            prefix_lens,  # 前缀长度
            None,  # 无KV起始索引
            self.kv_indptr[0],  # KV索引指针
            self.qo_indptr[0],  # QO索引指针
            use_ragged,  # 是否使用ragged模式
            spec_info,  # 投机解码信息
            fixed_split_size=fixed_split_size,  # 固定分片大小
            multi_item_params=multi_item_params,  # 多项评分参数
        )

    def update_sliding_window(  # 滑动窗口更新方法
        self,
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度
        seq_lens_sum: int,  # 序列长度总和
        prefix_lens: torch.Tensor,  # 前缀长度
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],  # 预填充包装器
        use_ragged: bool,  # 是否使用ragged模式
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        spec_info: Optional[SpecInput],  # 投机解码信息
        fixed_split_size: Optional[int] = None,  # 固定分片大小
        multi_item_params: Optional[MultiItemScoringParams] = None,  # 多项评分参数
        cross_attention_custom_mask: Optional[torch.Tensor] = None,  # 交叉注意力自定义掩码
    ):
        for wrapper_id in range(2):  # 遍历两个包装器
            swa_paged_custom_mask = None  # SWA分页自定义掩码初始化为None
            if wrapper_id == 0:  # 第一个包装器（滑动窗口注意力）
                if use_ragged:  # 如果使用ragged模式
                    # K for extend tokens is written after the paged wrapper runs, so
                    # the paged wrapper sees prefix-only. Trim to the last `window` tokens
                    # (required for SWATokenToKVPoolAllocator; also keeps mask O(window)).
                    # 扩展token的K在分页包装器运行后写入，因此
                    # 分页包装器只看到前缀部分。裁剪到最后`window`个token
                    # （SWATokenToKVPoolAllocator需要；也保持掩码O(window)复杂度）。
                    effective_start = torch.clamp(  # 计算有效起始位置
                        prefix_lens - self.sliding_window_size, min=0  # 钳位最小为0
                    )
                    paged_kernel_lens = prefix_lens - effective_start  # 分页内核长度为前缀长度减有效起始
                    paged_kernel_lens_sum = paged_kernel_lens.sum().item()  # 计算总和
                    kv_start_idx = effective_start  # KV起始索引为有效起始位置
                    swa_paged_custom_mask = self._build_swa_prefix_custom_mask(  # 构建SWA前缀自定义掩码
                        prefix_lens, seq_lens, effective_start  # 前缀长度、序列长度、有效起始
                    )
                else:  # 否则非ragged模式
                    # window attention use paged only
                    # 窗口注意力仅使用分页模式
                    paged_kernel_lens = torch.minimum(  # 计算分页内核长度
                        seq_lens,  # 序列长度
                        torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens,  # 窗口大小加扩展长度
                    )
                    paged_kernel_lens_sum = paged_kernel_lens.sum().item()  # 计算总和
                    kv_start_idx = seq_lens - paged_kernel_lens  # KV起始索引
            else:  # 第二个包装器（全注意力）
                # full attention
                # 全注意力
                paged_kernel_lens = seq_lens  # 使用完整序列长度
                paged_kernel_lens_sum = seq_lens_sum  # 使用完整序列长度总和
                kv_start_idx = seq_lens - paged_kernel_lens  # KV起始索引
            use_sliding_window_kv_pool = wrapper_id == 0 and isinstance(  # 判断是否使用滑动窗口KV池
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator  # 检查分配器类型
            )

            self.call_begin_forward(  # 调用begin_forward
                self.prefill_wrapper_ragged,  # ragged预填充包装器
                prefill_wrappers[wrapper_id],  # 当前分页预填充包装器
                req_pool_indices,  # 请求池索引
                paged_kernel_lens,  # 分页内核长度
                paged_kernel_lens_sum,  # 分页内核长度总和
                seq_lens,  # 序列长度
                prefix_lens,  # 前缀长度
                kv_start_idx,  # KV起始索引
                self.kv_indptr[wrapper_id],  # KV索引指针
                self.qo_indptr[wrapper_id],  # QO索引指针
                use_ragged,  # 是否使用ragged模式
                spec_info,  # 投机解码信息
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,  # 是否使用滑动窗口KV池
                fixed_split_size=fixed_split_size,  # 固定分片大小
                multi_item_params=multi_item_params,  # 多项评分参数
                cross_attention_custom_mask=swa_paged_custom_mask,  # 交叉注意力自定义掩码
            )

    def _build_swa_prefix_custom_mask(  # 构建SWA前缀自定义掩码（内部方法）
        self,
        prefix_lens: torch.Tensor,  # 前缀长度
        seq_lens: torch.Tensor,  # 序列长度
        kv_start_idx: torch.Tensor,  # KV起始索引
    ) -> Optional[torch.Tensor]:  # 返回自定义掩码或None
        """Custom SWA mask for the paged wrapper in the ragged merge_state EXTEND path.
        # ragged merge_state扩展路径中分页包装器的自定义SWA掩码

        Paged KV covers absolute positions [kv_start_idx[i], prefix_lens[i]).
        Returns None when every key is in-window for every extend query.
        # 分页KV覆盖绝对位置[kv_start_idx[i], prefix_lens[i])。
        # 当每个键都在每个扩展查询的窗口内时返回None。
        """
        window = self.sliding_window_size  # 获取滑动窗口大小
        if window is None or window < 0:  # 如果窗口大小无效
            return None  # 返回None

        prefix_lens_cpu = prefix_lens.detach().cpu().tolist()  # 将前缀长度转到CPU
        extend_lens_cpu = (seq_lens - prefix_lens).detach().cpu().tolist()  # 计算扩展长度并转到CPU
        kv_start_cpu = kv_start_idx.detach().cpu().tolist()  # 将KV起始索引转到CPU
        if all(p == 0 for p in prefix_lens_cpu):  # 如果所有前缀长度都为0
            return None  # 返回None

        device = prefix_lens.device  # 获取设备
        mask_parts: List[torch.Tensor] = []  # 初始化掩码部分列表
        need_mask = False  # 是否需要掩码标志
        for prefix_len, extend_len, kv_start in zip(  # 遍历每个序列
            prefix_lens_cpu, extend_lens_cpu, kv_start_cpu  # 前缀长度、扩展长度、KV起始
        ):
            paged_len = int(prefix_len - kv_start)  # = min(prefix_len, window)  # 计算分页长度
            if paged_len == 0 or extend_len == 0:  # 如果分页长度或扩展长度为0
                continue  # 跳过
            q_abs = torch.arange(extend_len, device=device).view(-1, 1) + prefix_len  # 计算查询绝对位置
            k_abs = torch.arange(paged_len, device=device).view(1, -1) + kv_start  # 计算键绝对位置
            block = (k_abs >= (q_abs - window)).to(torch.uint8)  # 生成掩码块
            if not bool(block.all()):  # 如果不是全部为True
                need_mask = True  # 需要掩码
            mask_parts.append(block.view(-1))  # 将掩码块展平并添加到列表

        if not need_mask or not mask_parts:  # 如果不需要掩码或掩码部分为空
            return None  # 返回None
        return torch.cat(mask_parts)  # 拼接所有掩码部分并返回

    def update_cross_attention(  # 交叉注意力更新方法
        self,
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度
        seq_lens_sum: int,  # 序列长度总和
        prefix_lens: torch.Tensor,  # 前缀长度
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],  # 预填充包装器
        use_ragged: bool,  # 是否使用ragged模式
        encoder_lens: Optional[torch.Tensor],  # 编码器长度
        spec_info: Optional[SpecInput],  # 投机解码信息
        fixed_split_size: Optional[int] = None,  # 固定分片大小
        multi_item_params: Optional[MultiItemScoringParams] = None,  # 多项评分参数
        cross_attention_custom_mask: Optional[torch.Tensor] = None,  # 交叉注意力自定义掩码
    ):
        for wrapper_id in range(2):  # 遍历两个包装器
            if wrapper_id == 0:  # 第一个包装器（自注意力）
                # normal attention
                # 正常注意力
                paged_kernel_lens = seq_lens  # 使用序列长度
                kv_start_idx = encoder_lens  # KV起始索引为编码器长度
                paged_kernel_lens_sum = seq_lens_sum  # 使用序列长度总和
            else:  # 第二个包装器（交叉注意力）
                # cross attention
                # 交叉注意力
                paged_kernel_lens = encoder_lens  # 使用编码器长度
                kv_start_idx = torch.zeros_like(encoder_lens)  # KV起始索引为零
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()  # 计算编码器长度总和

            self.call_begin_forward(  # 调用begin_forward
                self.prefill_wrapper_ragged,  # ragged预填充包装器
                prefill_wrappers[wrapper_id],  # 当前分页预填充包装器
                req_pool_indices,  # 请求池索引
                paged_kernel_lens,  # 分页内核长度
                paged_kernel_lens_sum,  # 分页内核长度总和
                seq_lens,  # 序列长度
                prefix_lens,  # 前缀长度
                kv_start_idx,  # KV起始索引
                self.kv_indptr[wrapper_id],  # KV索引指针
                self.qo_indptr[wrapper_id],  # QO索引指针
                use_ragged,  # 是否使用ragged模式
                spec_info,  # 投机解码信息
                fixed_split_size=fixed_split_size,  # 固定分片大小
                multi_item_params=multi_item_params,  # 多项评分参数
                cross_attention_custom_mask=(  # 交叉注意力自定义掩码
                    cross_attention_custom_mask if wrapper_id == 1 else None  # 仅第二个包装器使用
                ),
            )

    def call_begin_forward(  # 调用begin_forward的核心方法
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,  # ragged预填充包装器
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,  # 分页预填充包装器
        req_pool_indices: torch.Tensor,  # 请求池索引
        paged_kernel_lens: torch.Tensor,  # 分页内核长度
        paged_kernel_lens_sum: int,  # 分页内核长度总和
        seq_lens: torch.Tensor,  # 序列长度
        prefix_lens: torch.Tensor,  # 前缀长度
        kv_start_idx: torch.Tensor,  # KV起始索引
        kv_indptr: torch.Tensor,  # KV索引指针
        qo_indptr: torch.Tensor,  # QO索引指针
        use_ragged: bool,  # 是否使用ragged模式
        spec_info: Optional[SpecInput],  # 投机解码信息
        use_sliding_window_kv_pool: bool = False,  # 是否使用滑动窗口KV池
        fixed_split_size: Optional[int] = None,  # 固定分片大小
        multi_item_params: Optional[MultiItemScoringParams] = None,  # 多项评分参数
        cross_attention_custom_mask: Optional[torch.Tensor] = None,  # 交叉注意力自定义掩码
    ):
        bs = len(seq_lens)  # 获取批次大小
        if spec_info is None:  # 如果没有投机信息
            assert len(seq_lens) == len(req_pool_indices)  # 断言序列长度和请求池索引长度一致
            # Normal extend
            # 正常扩展
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)  # 计算累积和作为索引指针
            kv_indptr = kv_indptr[: bs + 1]  # 截取当前批次大小的索引指针
            kv_indices = torch.empty(  # 创建空的KV索引张量
                paged_kernel_lens_sum + 256,  # 大小加256（额外余量）
                dtype=torch.int32,  # 数据类型
                device=req_pool_indices.device,  # 设备
            )
            create_flashinfer_kv_indices_triton[(bs,)](  # 调用Triton核函数创建KV索引
                self.req_to_token,  # 请求到token映射表
                req_pool_indices,  # 请求池索引
                paged_kernel_lens,  # 分页内核长度
                kv_indptr,  # KV索引指针
                kv_start_idx,  # KV起始索引
                kv_indices,  # KV索引输出
                self.req_to_token.shape[1],  # 映射表列数
            )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)  # 计算QO索引指针
            qo_indptr = qo_indptr[: bs + 1]  # 截取当前批次大小的QO索引指针

            custom_mask = cross_attention_custom_mask  # 使用交叉注意力自定义掩码
        else:  # 否则有投机信息
            assert isinstance(spec_info, SpecInput)  # 断言投机信息为SpecInput类型
            kv_indices, kv_indptr, qo_indptr, custom_mask = (  # 从投机信息生成注意力参数
                spec_info.generate_attn_arg_prefill(  # 调用注意力参数生成函数
                    req_pool_indices,  # 请求池索引
                    paged_kernel_lens,  # 分页内核长度
                    paged_kernel_lens_sum,  # 分页内核长度总和
                    self.req_to_token,  # 请求到token映射表
                )
            )

        # extend part
        # 扩展部分
        if use_ragged:  # 如果使用ragged模式
            wrapper_ragged.begin_forward(  # 初始化ragged包装器
                qo_indptr,  # QO索引指针
                qo_indptr,  # KV索引指针（与QO相同）
                self.num_qo_heads,  # 查询/输出头数
                self.num_kv_heads,  # KV头数
                self.head_dim,  # 头维度
                q_data_type=self.q_data_type,  # 查询数据类型
            )

        if use_sliding_window_kv_pool:  # 如果使用滑动窗口KV池
            kv_last_index = kv_indptr[-1]  # 获取最后一个索引
            kv_indices[:kv_last_index] = (  # 转换KV索引
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(  # 从全量位置转换为SWA位置
                    kv_indices[:kv_last_index]  # 需要转换的索引
                )
            )

        # cached part
        # 缓存部分
        # Conditionally set multi-item parameters
        # 条件性地设置多项参数
        if multi_item_params is not None and multi_item_params.is_enabled():  # 如果多项评分已启用
            # Multi-item scoring is active - use specialized parameters and disable generic custom_mask
            # 多项评分已激活 - 使用专用参数并禁用通用custom_mask
            use_custom_mask = None  # 不使用通用自定义掩码
            prefix_len_ptr = multi_item_params.prefix_len_ptr  # 前缀长度指针
            token_pos_in_items_ptr = multi_item_params.token_pos_in_items_ptr  # 项内token位置指针
            token_pos_in_items_len = multi_item_params.token_pos_in_items_len  # 项内token位置长度
            max_item_len_ptr = multi_item_params.max_item_len_ptr  # 最大项长度指针
        else:  # 否则
            # No multi-item scoring - use standard parameters
            # 无多项评分 - 使用标准参数
            use_custom_mask = custom_mask  # 使用标准自定义掩码
            prefix_len_ptr = None  # 无前缀长度指针
            token_pos_in_items_ptr = None  # 无项内token位置指针
            token_pos_in_items_len = 0  # 项内token位置长度为0
            max_item_len_ptr = None  # 无最大项长度指针

        wrapper_paged.begin_forward(  # 初始化分页包装器
            qo_indptr,  # QO索引指针
            kv_indptr,  # KV索引指针
            kv_indices,  # KV索引
            self.kv_last_page_len[:bs],  # KV最后页长度
            self.num_qo_heads,  # 查询/输出头数
            self.num_kv_heads,  # KV头数
            self.head_dim,  # 头维度
            1,  # 页大小
            q_data_type=self.q_data_type,  # 查询数据类型
            kv_data_type=self.data_type,  # KV数据类型
            custom_mask=use_custom_mask,  # 自定义掩码
            non_blocking=True,  # 非阻塞传输
            fixed_split_size=fixed_split_size,  # 固定分片大小
            prefix_len_ptr=prefix_len_ptr,  # 前缀长度指针
            token_pos_in_items_ptr=token_pos_in_items_ptr,  # 项内token位置指针
            token_pos_in_items_len=token_pos_in_items_len,  # 项内token位置长度
            max_item_len_ptr=max_item_len_ptr,  # 最大项长度指针
        )


class FlashInferMultiStepDraftBackend:  # FlashInfer多步草稿后端类
    """
    Wrap multiple flashinfer attention backends as one for multiple consecutive
    draft decoding steps.
    # 将多个FlashInfer注意力后端包装为一个，用于多个连续的草稿解码步骤。
    """

    def __init__(  # 初始化方法
        self,
        model_runner: ModelRunner,  # 模型运行器
        topk: int,  # top-k值
        speculative_num_steps: int,  # 投机步数
    ):
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices  # 导入草稿解码KV索引生成函数

        self.topk = topk  # 保存top-k值
        self.speculative_num_steps = speculative_num_steps  # 保存投机步数
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices  # 保存KV索引生成函数引用
        self.page_size = model_runner.page_size  # 保存页大小

        max_bs = model_runner.req_to_token_pool.size * self.topk  # 计算最大批次大小
        self.kv_indptr = torch.zeros(  # 创建KV索引指针张量
            (  # 形状
                self.speculative_num_steps,  # 投机步数维度
                max_bs + 1,  # 批次大小维度
            ),
            dtype=torch.int32,  # 数据类型
            device=model_runner.device,  # 设备
        )
        self.kv_last_page_len = torch.ones(  # 创建KV最后页长度张量
            (max_bs,), dtype=torch.int32, device=model_runner.device  # 形状、类型、设备
        )
        self.attn_backends: List[FlashInferAttnBackend] = []  # 初始化注意力后端列表
        for i in range(self.speculative_num_steps - 1):  # 为每个投机步创建后端
            self.attn_backends.append(  # 添加注意力后端
                FlashInferAttnBackend(  # 创建FlashInfer注意力后端
                    model_runner,  # 模型运行器
                    skip_prefill=True,  # 跳过预填充
                    kv_indptr_buf=self.kv_indptr[i],  # KV索引指针缓冲区
                    kv_last_page_len_buf=self.kv_last_page_len,  # KV最后页长度缓冲区
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len  # 保存最大上下文长度

        # Cached variables for generate_draft_decode_kv_indices
        # 为generate_draft_decode_kv_indices缓存的变量
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]  # 保存池长度
        self.req_to_token_pool = model_runner.req_to_token_pool  # 保存请求到token池引用

    def common_template(  # 通用模板方法
        self,
        forward_batch: ForwardBatch,  # 前向批次
        kv_indices_buffer: torch.Tensor,  # KV索引缓冲区
        call_fn: Callable,  # 回调函数
    ):
        num_seqs = forward_batch.batch_size  # 获取序列数
        bs = self.topk * num_seqs  # 计算总批次大小
        seq_lens_sum = forward_batch.seq_lens_sum  # 获取序列长度总和

        self.generate_draft_decode_kv_indices[  # 生成草稿解码KV索引
            (self.speculative_num_steps, num_seqs, self.topk)  # 启动网格大小
        ](
            forward_batch.req_pool_indices,  # 请求池索引
            self.req_to_token_pool.req_to_token,  # 请求到token映射表
            forward_batch.seq_lens,  # 序列长度
            kv_indices_buffer,  # KV索引缓冲区
            self.kv_indptr,  # KV索引指针
            forward_batch.positions,  # 位置信息
            self.pool_len,  # 池长度
            kv_indices_buffer.shape[1],  # KV索引缓冲区列数
            self.kv_indptr.shape[1],  # KV索引指针列数
            next_power_of_2(num_seqs),  # 序列数的下一个2的幂
            next_power_of_2(self.speculative_num_steps),  # 投机步数的下一个2的幂
            next_power_of_2(bs),  # 批次大小的下一个2的幂
            self.page_size,  # 页大小
        )

        assert forward_batch.spec_info is not None  # 断言投机信息存在
        assert forward_batch.spec_info.is_draft_input()  # 断言为草稿输入

        # Copy the kv_indptr once to avoid multiple device-to-host copies in flashinfer's plan.
        # 一次性复制kv_indptr以避免flashinfer的plan中多次设备到主机拷贝。
        indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()  # 将KV索引指针复制到CPU
        global global_override_indptr_cpu  # 声明全局覆盖indptr CPU缓冲区

        for i in range(self.speculative_num_steps - 1):  # 遍历每个投机步
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]  # 设置当前步的KV索引指针
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][  # 设置当前步的KV索引
                : seq_lens_sum * self.topk + bs * (i + 1)  # 截取有效范围
            ]
            global_override_indptr_cpu = indptr_cpu_whole[i]  # 设置全局覆盖indptr
            call_fn(i, forward_batch)  # 调用回调函数

        global_override_indptr_cpu = None  # 清除全局覆盖缓冲区

    def init_forward_metadata(self, forward_batch: ForwardBatch):  # 初始化前向元数据
        kv_indices = torch.empty(  # 创建空的KV索引张量
            (  # 形状
                self.speculative_num_steps,  # 投机步数
                forward_batch.batch_size * self.topk * self.max_context_len,  # 每步最大索引数
            ),
            dtype=torch.int32,  # 数据类型
            device="cuda",  # 设备
        )

        def call_fn(i, forward_batch):  # 回调函数定义
            forward_batch.spec_info.kv_indptr = (  # 克隆KV索引指针
                forward_batch.spec_info.kv_indptr.clone()  # 克隆以避免共享缓冲区
            )
            forward_batch.spec_info.kv_indices = (  # 克隆KV索引
                forward_batch.spec_info.kv_indices.clone()  # 克隆以避免共享缓冲区
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)  # 初始化当前步的前向元数据

        self.common_template(forward_batch, kv_indices, call_fn)  # 调用通用模板

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):  # 初始化CUDA图状态
        self.cuda_graph_kv_indices = torch.zeros(  # 创建CUDA图KV索引张量
            (self.speculative_num_steps, max_bs * self.max_context_len),  # 形状
            dtype=torch.int32,  # 数据类型
            device="cuda",  # 设备
        )

        for i in range(self.speculative_num_steps - 1):  # 遍历每个投机步
            self.attn_backends[i].init_cuda_graph_state(  # 初始化当前步的CUDA图状态
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]  # 传入KV索引缓冲区
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):  # 初始化CUDA图捕获时的前向元数据
        def call_fn(i, forward_batch):  # 回调函数定义
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(  # 初始化当前步的CUDA图捕获
                forward_batch.batch_size,  # 批次大小
                forward_batch.batch_size * self.topk,  # token数
                forward_batch.req_pool_indices,  # 请求池索引
                forward_batch.seq_lens,  # 序列长度
                encoder_lens=None,  # 无编码器长度
                forward_mode=ForwardMode.DECODE,  # 前向模式为解码
                spec_info=forward_batch.spec_info,  # 投机解码信息
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)  # 调用通用模板

    def init_forward_metadata_replay_cuda_graph(  # 初始化CUDA图重放时的前向元数据
        self, forward_batch: ForwardBatch, bs: int  # 前向批次和批次大小
    ):
        def call_fn(i, forward_batch):  # 回调函数定义
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(  # 初始化当前步的CUDA图重放
                bs,  # 批次大小
                forward_batch.req_pool_indices,  # 请求池索引
                forward_batch.seq_lens,  # 序列长度
                seq_lens_sum=-1,  # 序列长度总和（-1表示未使用）
                encoder_lens=None,  # 无编码器长度
                forward_mode=ForwardMode.DECODE,  # 前向模式为解码
                spec_info=forward_batch.spec_info,  # 投机解码信息
                seq_lens_cpu=forward_batch.seq_lens_cpu,  # CPU序列长度
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)  # 调用通用模板


def should_use_tensor_core(  # 判断是否应使用Tensor Core进行注意力计算
    kv_cache_dtype: torch.dtype,  # KV缓存数据类型
    num_attention_heads: int,  # 注意力头数
    num_kv_heads: int,  # KV头数
) -> bool:  # 返回布尔值
    """
    Determine whether to use tensor cores for attention computation.
    # 判断是否应使用Tensor Core进行注意力计算

    Args:
        kv_cache_dtype: Data type of the KV cache
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads
    # 参数：
    #     kv_cache_dtype: KV缓存的数据类型
    #     num_attention_heads: 注意力头数
    #     num_kv_heads: 键/值头数

    Returns:
        bool: Whether to use tensor cores
    # 返回：
    #     bool: 是否使用Tensor Core
    """
    # Try to use environment variable first
    # 首先尝试使用环境变量
    env_override = os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")  # 获取环境变量
    if env_override is not None:  # 如果环境变量存在
        return env_override.lower() == "true"  # 根据环境变量值返回

    # Try to use _grouped_size_compiled_for_decode_kernels if available
    # This is for flashinfer <=0.1.6. Otherwise, there is an accuracy bug
    # 尝试使用_grouped_size_compiled_for_decode_kernels（如果可用）
    # 这适用于flashinfer <=0.1.6。否则存在精度bug
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels  # 导入分组大小编译检测函数

        if not _grouped_size_compiled_for_decode_kernels(  # 如果当前配置未编译分组大小
            num_attention_heads,  # 注意力头数
            num_kv_heads,  # KV头数
        ):
            return True  # 需要使用Tensor Core
        else:  # 否则
            return False  # 不需要使用Tensor Core
    except (ImportError, AttributeError):  # 导入失败或属性不存在
        pass  # 跳过

    # Calculate GQA group size
    # 计算GQA组大小
    gqa_group_size = num_attention_heads // num_kv_heads  # 注意力头数除以KV头数

    # For Flashinfer, a GQA group size of at least 4 is needed to efficiently
    # use Tensor Cores, as it fuses the head group with the token dimension in MMA.
    # 对于FlashInfer，GQA组大小至少为4才能有效使用Tensor Core，
    # 因为它在MMA中将头组与token维度融合。
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):  # 如果是FP8数据类型
        return True  # 始终使用Tensor Core
    elif kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):  # 如果是FP16/BF16数据类型
        return gqa_group_size >= 4  # GQA组大小>=4时使用Tensor Core
    else:  # 否则
        return False  # 不使用Tensor Core
