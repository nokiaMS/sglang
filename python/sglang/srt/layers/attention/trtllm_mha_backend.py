# 文件说明：TRTLLM MHA注意力后端实现
# 本文件实现了基于flashinfer的TRTLLM MHA注意力内核后端
# 支持sm100架构，具备滑动窗口和注意力汇聚(sink)功能
# 包含CUDA图捕获与回放、推测解码、FP8融合量化等高级特性

from __future__ import annotations  # 启用延迟注解求值 # 允许前向引用类型

"""
Support attention backend for TRTLLM MHA kernels from flashinfer. # 支持来自flashinfer的TRTLLM MHA内核的注意力后端
The kernel supports sm100 only, with sliding window and attention sink features. # 内核仅支持sm100，具备滑动窗口和注意力汇聚特性
"""

import logging  # 导入日志模块 # 日志记录
from dataclasses import dataclass  # 导入数据类装饰器 # 数据类支持
from typing import TYPE_CHECKING, Optional  # 导入类型检查和可选类型 # 类型提示

import torch  # 导入PyTorch库 # 深度学习框架

from sglang.srt.environ import envs  # 导入环境变量模块 # 环境变量
from sglang.srt.layers.attention.flashinfer_backend import (  # 导入FlashInfer注意力后端 # FlashInfer后端基类
    FlashInferAttnBackend,  # FlashInfer注意力后端类 # 基础后端
    FlashInferMultiStepDraftBackend,  # FlashInfer多步草稿后端类 # 多步草稿后端
)
from sglang.srt.layers.attention.triton_ops.trtllm_fp8_kv_kernel import (  # 导入FP8融合内核 # FP8融合量化内核
    fused_fp8_set_kv_buffer,  # 融合FP8 KV缓存写入函数 # FP8量化+写入
)
from sglang.srt.layers.attention.utils import canonicalize_stride  # 导入步长规范化工具 # 步长规范化
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool  # 导入滑动窗口KV池 # 滑动窗口KV内存池
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode  # 导入前向批次信息和模式 # 前向计算信息
from sglang.srt.utils import is_flashinfer_available  # 导入flashinfer可用性检测 # flashinfer检测
from sglang.srt.utils.common import is_sm90_supported, is_sm120_supported  # 导入GPU算力检测 # GPU算力检测

logger = logging.getLogger(__name__)  # 创建模块级日志器 # 模块日志记录器

if is_flashinfer_available():  # 如果flashinfer可用 # 检查flashinfer
    import flashinfer  # 导入flashinfer库 # flashinfer内核库

if TYPE_CHECKING:  # 类型检查时的导入 # 仅类型检查时导入
    from sglang.srt.layers.radix_attention import RadixAttention  # 基数注意力类 # Radix注意力
    from sglang.srt.model_executor.model_runner import ModelRunner  # 模型运行器 # 模型执行器
    from sglang.srt.speculative.spec_info import SpecInput  # 推测输入信息 # 推测解码输入

# Constants # 常量 # 常量定义
# Default workspace size in MB for TRTLLM MHA # TRTLLM MHA的默认工作空间大小（MB）
# Can be configured via SGLANG_FLASHINFER_WORKSPACE_SIZE environment variable # 可通过SGLANG_FLASHINFER_WORKSPACE_SIZE环境变量配置
DEFAULT_WORKSPACE_SIZE_MB = 512  # 默认工作空间大小512MB # 默认工作区大小

# Reuse this workspace buffer across all TRTLLM MHA wrappers # 在所有TRTLLM MHA包装器间复用此工作空间缓冲区
global_zero_init_workspace_buffer = None  # 全局零初始化工作空间缓冲区 # 全局工作区


@dataclass  # 数据类装饰器 # 自动生成__init__等方法
class TRTLLMMHAMetadata:  # TRTLLM MHA元数据类 # 存储TRTLLM MHA前向计算所需元数据
    # Sequence lengths for the forward batch # 前向批次的序列长度
    cache_seqlens_int32: torch.Tensor = None  # 缓存序列长度（int32） # KV缓存序列长度
    # Maximum sequence length for query # 查询的最大序列长度
    max_seq_len_q: int = 1  # 最大查询序列长度 # Q最大长度
    # Maximum sequence length for key # 键的最大序列长度
    max_seq_len_k: int = 0  # 最大键序列长度 # KV最大长度
    # Cumulative sequence lengths for `query # 查询的累积序列长度
    cu_seqlens_q: torch.Tensor = None  # Q累积序列长度 # Q的cumsum
    # Cumulative sequence lengths for key # 键的累积序列长度
    cu_seqlens_k: torch.Tensor = None  # KV累积序列长度 # KV的cumsum
    # Page table, the index of KV Cache Tables/Blocks # 页表，KV缓存表/块的索引
    page_table: torch.Tensor = None  # 页表 # KV页表
    # Page table for SWA layers (translated from full pool indices to SWA pool indices) # SWA层的页表（从全池索引转换到SWA池索引）
    swa_page_table: torch.Tensor = None  # SWA页表 # 滑动窗口页表


class TRTLLMHAAttnBackend(FlashInferAttnBackend):  # TRTLLM MHA注意力后端类 # 继承FlashInfer后端
    """TRTLLM MHA attention kernel from flashinfer.""" # 来自flashinfer的TRTLLM MHA注意力内核 # TRTLLM MHA注意力内核

    def __init__(  # 初始化方法 # 构造函数
        self,
        model_runner: ModelRunner,  # 模型运行器 # 模型执行器
        skip_prefill: bool = False,  # 是否跳过预填充 # 跳过prefill标志
        kv_indptr_buf: Optional[torch.Tensor] = None,  # KV索引指针缓冲区 # KV indptr
        kv_last_page_len_buf: Optional[torch.Tensor] = None,  # KV最后一页长度缓冲区 # KV最后页长度
        speculative_step_id: int = 0,  # 推测步ID # 推测解码步骤ID
    ):
        # Capture workspace size before super().__init__() to preserve user's # 在super().__init__()之前捕获工作空间大小，以保留用户的
        # SGLANG_FLASHINFER_WORKSPACE_SIZE setting (may be overridden by parent) # SGLANG_FLASHINFER_WORKSPACE_SIZE设置（可能被父类覆盖）
        env_var = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE  # 获取工作空间大小环境变量 # 读取环境变量
        workspace_size_bytes = (  # 计算工作空间字节数 # 工作区大小
            env_var.get()  # 获取环境变量值 # 读取设置值
            if env_var.is_set()  # 如果环境变量已设置 # 检查是否设置
            else DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024  # 否则使用默认值 # 使用默认512MB
        )

        super().__init__(  # 调用父类初始化 # 初始化FlashInfer后端
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        config = model_runner.model_config  # 获取模型配置 # 模型配置

        # MHA-specific dimensions # MHA特定维度 # MHA维度参数
        self.max_context_len = model_runner.model_config.context_len  # 最大上下文长度 # 最大上下文长度
        self.hidden_size = config.hidden_size  # 隐藏层大小 # 隐藏维度

        # Runtime parameters # 运行时参数 # 运行时参数
        self.data_type = model_runner.kv_cache_dtype  # KV缓存数据类型 # KV缓存dtype
        self.q_data_type = model_runner.dtype  # 查询数据类型 # 查询dtype
        self.page_size = model_runner.page_size  # 页大小 # 每页token数
        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 请求到token的映射表 # 请求-token映射
        self.device = model_runner.device  # 计算设备 # GPU设备

        # Workspace allocation # 工作空间分配 # 工作区分配
        self.workspace_size = workspace_size_bytes  # 工作空间大小 # 工作区字节数
        # Allocate buffers # 分配缓冲区 # 缓冲区分配
        global global_zero_init_workspace_buffer  # 声明全局变量 # 全局变量
        if global_zero_init_workspace_buffer is None:  # 如果全局缓冲区未初始化 # 检查是否已分配
            global_zero_init_workspace_buffer = torch.zeros(  # 创建零初始化工作空间 # 分配零初始化工作区
                self.workspace_size,  # 工作空间大小 # 大小
                dtype=torch.uint8,  # 数据类型uint8 # uint8类型
                device=model_runner.device,  # 计算设备 # GPU设备
            )
        self.workspace_buffer = global_zero_init_workspace_buffer  # 引用全局缓冲区 # 使用全局工作区

        # CUDA graph state # CUDA图状态 # CUDA图捕获状态
        self.decode_cuda_graph_metadata = {}  # 解码CUDA图元数据字典 # decode图元数据

        # Speculative decoding # 推测解码 # 推测解码配置
        # Only support topk <= 1 for now. # 目前仅支持topk <= 1
        self.topk = model_runner.server_args.speculative_eagle_topk or 0  # EAGLE topk值 # EAGLE topk
        self.speculative_step_id = speculative_step_id  # 推测步ID # 推测步骤ID
        self.target_verify_metadata = {}  # 目标验证元数据 # 验证元数据

        self.speculative_num_draft_tokens = (  # 推测草稿token数量 # 草稿token数
            model_runner.server_args.speculative_num_draft_tokens
        )

        # Sliding Window Attention(SWA) hybrid model support. # 滑动窗口注意力(SWA)混合模型支持
        # For hybrid SWA models, the KV cache is split into two pools (full and SWA) # 对于混合SWA模型，KV缓存分为两个池（全量和SWA）
        # with separate index spaces. We maintain a translated page_table for SWA # 具有独立的索引空间。我们为SWA维护一个转换后的page_table
        # layers so the trtllm kernel reads from the correct pool. # 层，以便trtllm内核从正确的池读取
        kv_pool = model_runner.token_to_kv_pool  # 获取KV池 # token到KV的池
        self.use_sliding_window_kv_pool = isinstance(kv_pool, SWAKVPool)  # 是否使用滑动窗口KV池 # SWA标志
        self._swa_kv_pool: Optional[SWAKVPool] = (  # SWA KV池引用 # SWA池
            kv_pool if self.use_sliding_window_kv_pool else None  # 如果使用SWA则保存引用 # 条件赋值
        )

        # Forward metadata # 前向元数据 # 前向计算元数据
        self.forward_metadata: Optional[TRTLLMMHAMetadata] = None  # 前向元数据 # 当前前向元数据

        # Init backend (XQA or TRTLLM-GEN) # 初始化后端（XQA或TRTLLM-GEN） # 选择后端实现
        # We need to specify q_type and out_type for different backend # 需要为不同后端指定q_type和out_type
        # XQA: (q_type must be bf16) # XQA：（q_type必须为bf16）
        #   KV bf16: q_type = bf16, out_type=model_runner.dtype #   KV bf16：q_type = bf16, out_type=model_runner.dtype
        #   KV fp8: q_type = bf16, out_type=model_runner.dtype #   KV fp8：q_type = bf16, out_type=model_runner.dtype
        # TRTLLM-GEN: # TRTLLM-GEN：
        #   KV bf16: q_type = bf16, out_type=model_runner.dtype #   KV bf16：q_type = bf16, out_type=model_runner.dtype
        #   KV fp8: q_type = fp8, out_type=model_runner.dtype #   KV fp8：q_type = fp8, out_type=model_runner.dtype
        self.is_xqa_impl = is_sm90_supported() or is_sm120_supported()  # 判断是否使用XQA实现 # XQA实现标志

    def _maybe_translate_swa(  # 可能转换SWA索引的函数 # 将全池索引转换为SWA池索引
        self, token_indices: torch.Tensor  # token索引张量 # 输入索引
    ) -> Optional[torch.Tensor]:  # 返回转换后的索引或None # 转换结果
        """Translate full-pool token indices to SWA-pool indices, or return None.""" # 将全池token索引转换为SWA池索引，或返回None """全池索引转SWA池索引"""
        if not self.use_sliding_window_kv_pool:  # 如果不使用SWA # 非SWA模型
            return None  # 返回None # 无需转换
        shape = token_indices.shape  # 保存原始形状 # 原始形状
        return self._swa_kv_pool.translate_loc_from_full_to_swa(  # 调用SWA池的索引转换 # 执行索引转换
            token_indices.reshape(-1)  # 展平为一维 # 展平
        ).reshape(shape)  # 恢复原始形状 # 恢复形状

    def _alloc_swa_page_table(  # 分配SWA页表缓冲区 # 分配滑动窗口页表
        self, max_bs: int, max_num_pages: int  # 最大batch大小和最大页数 # 批次大小和页数
    ) -> Optional[torch.Tensor]:  # 返回页表张量或None # 页表缓冲区
        """Allocate a SWA page_table buffer, or return None for non-SWA models.""" # 分配SWA页表缓冲区，非SWA模型返回None """分配SWA页表"""
        if not self.use_sliding_window_kv_pool:  # 如果不使用SWA # 非SWA模型
            return None  # 返回None # 无需分配
        return torch.zeros(max_bs, max_num_pages, dtype=torch.int32, device=self.device)  # 创建零初始化页表 # 分配页表

    def _copy_swa_page_table(  # 复制SWA页表数据 # 将转换后的SWA索引写入页表
        self,
        metadata: TRTLLMMHAMetadata,  # 元数据 # MHA元数据
        page_indices: torch.Tensor,  # 页索引 # 原始页索引
        num_pages: int,  # 页数量 # 有效页数
    ):
        """Translate and copy SWA page indices into metadata. No-op for non-SWA.""" # 转换并复制SWA页索引到元数据。非SWA模型无操作 """复制SWA页索引"""
        if metadata.swa_page_table is None:  # 如果没有SWA页表 # 无SWA页表
            return  # 直接返回 # 无操作
        swa_indices = self._maybe_translate_swa(page_indices)  # 转换页索引 # 索引转换
        metadata.swa_page_table[:, :num_pages].copy_(swa_indices // self.page_size)  # 复制转换后的页索引 # 写入SWA页表

    def _get_layer_cache_loc(  # 获取层的缓存位置 # 返回指定层的缓存位置索引
        self,
        layer: RadixAttention,  # 注意力层 # 当前注意力层
        forward_batch: ForwardBatch,  # 前向批次 # 前向计算批次
    ) -> torch.Tensor:  # 返回缓存位置张量 # 缓存位置
        """Return cache locations in the correct index space for the given layer.""" # 返回给定层的正确索引空间中的缓存位置 """获取层的缓存位置"""
        if self.use_sliding_window_kv_pool:  # 如果使用SWA # SWA模型
            _, is_swa = self._swa_kv_pool.layers_mapping[layer.layer_id]  # 检查当前层是否为SWA层 # 判断层类型
            if is_swa:  # 如果是SWA层 # SWA层
                return self._swa_kv_pool.translate_loc_from_full_to_swa(  # 转换为SWA索引空间 # 索引转换
                    forward_batch.out_cache_loc  # 原始缓存位置 # 原始位置
                )
        return forward_batch.out_cache_loc  # 返回原始缓存位置 # 全池位置

    def _bind_swa_page_table(  # 绑定SWA页表到元数据 # 将预分配的SWA页表绑定到元数据
        self, metadata: TRTLLMMHAMetadata, source: dict, key: str, bs: int  # 元数据、源字典、键、批次大小 # 参数
    ):
        """Bind a pre-allocated SWA page_table slice to metadata for CUDA graph.""" # 将预分配的SWA页表切片绑定到元数据，用于CUDA图 """绑定SWA页表"""
        buf = source.get(key)  # 从源字典获取缓冲区 # 读取缓冲区
        if buf is not None:  # 如果缓冲区存在 # 检查缓冲区
            metadata.swa_page_table = buf[:bs, :]  # 切片绑定到元数据 # 切片绑定

    def _get_layer_page_table(  # 获取层的页表 # 返回指定层的正确页表（SWA或全量）
        self, layer: RadixAttention, forward_batch: ForwardBatch  # 注意力层和前向批次 # 参数
    ) -> torch.Tensor:  # 返回页表张量 # 页表
        """Return the correct page_table for the given layer (SWA or full).""" # 返回给定层的正确页表（SWA或全量） """获取层页表"""
        swa_pt = self.forward_metadata.swa_page_table  # 获取SWA页表 # SWA页表
        if swa_pt is not None:  # 如果SWA页表存在 # 有SWA页表
            _, is_swa = self._swa_kv_pool.layers_mapping[layer.layer_id]  # 检查当前层是否为SWA层 # 判断层类型
            if is_swa:  # 如果是SWA层 # SWA层
                return swa_pt  # 返回SWA页表 # 使用SWA页表
        return self.forward_metadata.page_table  # 返回全量页表 # 使用全量页表

    def init_cuda_graph_state(  # 初始化CUDA图状态 # 为TRTLLM MHA初始化CUDA图所需缓冲区
        self,
        max_bs: int,  # 最大batch大小 # 最大批次大小
        max_num_tokens: int,  # 最大token数量 # 最大token数
        kv_indices_buf: Optional[torch.Tensor] = None,  # KV索引缓冲区（可选） # KV索引
    ):
        """Initialize CUDA graph state for TRTLLM MHA.""" # 为TRTLLM MHA初始化CUDA图状态 """初始化CUDA图状态"""
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size  # 计算最大页数 # 最大页数
        self.decode_cuda_graph_metadata = {  # 初始化解码CUDA图元数据 # decode图元数据字典
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),  # 缓存序列长度 # 序列长度缓冲区
            "page_table": torch.zeros(  # 页表 # 全量页表
                max_bs,  # batch维度 # 批次大小
                max_num_pages,  # 页维度 # 页数
                dtype=torch.int32,  # 数据类型int32 # int32类型
                device=self.device,  # 设备 # GPU设备
            ),
            "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),  # SWA页表 # SWA页表
            "strided_indices": torch.arange(  # 步幅索引 # 步幅索引数组
                0, self.max_context_len, self.page_size, device=self.device  # 按page_size步进 # 按页大小步进
            ),
        }

        if (  # 如果启用了推测解码 # 推测解码配置
            self.speculative_num_draft_tokens is not None  # 草稿token数已设置 # 检查草稿token数
            and self.speculative_num_draft_tokens > 0  # 草稿token数大于0 # 验证有效性
        ):
            self.decode_cuda_graph_metadata["cu_seqlens_q"] = torch.arange(  # Q累积序列长度 # Q的cumsum
                0, max_bs + 1, dtype=torch.int32, device=self.device  # 0到max_bs+1 # 范围
            )
            self.decode_cuda_graph_metadata["cu_seqlens_k"] = torch.zeros(  # KV累积序列长度 # KV的cumsum
                max_bs + 1, dtype=torch.int32, device=self.device  # 零初始化 # 零值
            )
            self.decode_cuda_graph_metadata["page_table_draft_decode"] = torch.zeros(  # 草稿解码页表 # draft decode页表
                max_bs,  # batch维度 # 批次大小
                max_num_pages,  # 页维度 # 页数
                dtype=torch.int32,  # 数据类型 # int32
                device=self.device,  # 设备 # GPU
            )
            self.decode_cuda_graph_metadata["swa_page_table_draft_decode"] = (  # 草稿解码SWA页表 # draft decode SWA页表
                self._alloc_swa_page_table(max_bs, max_num_pages)  # 分配SWA页表 # 分配
            )

            self.target_verify_metadata = {  # 目标验证元数据 # verify元数据
                "cache_seqlens": torch.zeros(  # 缓存序列长度 # 序列长度
                    max_bs, dtype=torch.int32, device=self.device  # 零初始化 # 零值
                ),
                "cu_seqlens_q": torch.arange(  # Q累积序列长度 # Q的cumsum
                    0,  # 起始值 # 起点
                    max_bs * self.speculative_num_draft_tokens + 1,  # 终止值 # 终点
                    step=self.speculative_num_draft_tokens,  # 步长 # 步长
                    dtype=torch.int32,  # 数据类型 # int32
                    device=self.device,  # 设备 # GPU
                ),
                "cu_seqlens_k": torch.zeros(  # KV累积序列长度 # KV的cumsum
                    max_bs + 1, dtype=torch.int32, device=self.device  # 零初始化 # 零值
                ),
                "page_table": torch.zeros(  # 页表 # 全量页表
                    max_bs,  # batch维度 # 批次大小
                    max_num_pages,  # 页维度 # 页数
                    dtype=torch.int32,  # 数据类型 # int32
                    device=self.device,  # 设备 # GPU
                ),
                "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),  # SWA页表 # SWA页表
                "strided_indices": torch.arange(  # 步幅索引 # 步幅索引数组
                    0, self.max_context_len, self.page_size, device=self.device  # 按page_size步进 # 按页大小步进
                ),
            }

            self.draft_extend_metadata = {  # 草稿扩展元数据 # draft extend元数据
                "cache_seqlens": torch.zeros(  # 缓存序列长度 # 序列长度
                    max_bs, dtype=torch.int32, device=self.device  # 零初始化 # 零值
                ),
                "cu_seqlens_q": torch.zeros(  # Q累积序列长度 # Q的cumsum
                    max_bs + 1,  # 大小 # 长度
                    dtype=torch.int32,  # 数据类型 # int32
                    device=self.device,  # 设备 # GPU
                ),
                "cu_seqlens_k": torch.zeros(  # KV累积序列长度 # KV的cumsum
                    max_bs + 1, dtype=torch.int32, device=self.device  # 零初始化 # 零值
                ),
                "page_table": torch.zeros(  # 页表 # 全量页表
                    max_bs,  # batch维度 # 批次大小
                    max_num_pages,  # 页维度 # 页数
                    dtype=torch.int32,  # 数据类型 # int32
                    device=self.device,  # 设备 # GPU
                ),
                "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),  # SWA页表 # SWA页表
                "strided_indices": torch.arange(  # 步幅索引 # 步幅索引数组
                    0, self.max_context_len, self.page_size, device=self.device  # 按page_size步进 # 按页大小步进
                ),
            }

    def _build_cuda_graph_metadata(  # 构建CUDA图元数据 # 创建预分配缓冲区切片引用的元数据
        self,
        bs: int,  # batch大小 # 当前批次大小
        num_tokens: int,  # token数量 # 当前token数
        forward_mode: ForwardMode,  # 前向模式 # 前向计算模式
        spec_info,  # 推测信息 # 推测解码信息
        device: torch.device,  # 计算设备 # GPU设备
    ) -> "TRTLLMMHAMetadata":  # 返回TRTLLM MHA元数据 # 元数据
        """Create TRTLLMMHAMetadata with pre-allocated buffer slice refs, stored in the dict.""" # 创建带有预分配缓冲区切片引用的TRTLLMMHAMetadata，存储在字典中 """构建CUDA图元数据"""
        metadata = TRTLLMMHAMetadata()  # 创建元数据实例 # 新建元数据

        if forward_mode.is_decode_or_idle():  # 解码或空闲模式 # decode模式
            if spec_info is not None:  # 如果有推测信息 # draft decode
                # Draft Decode (topk = 1) # 草稿解码（topk = 1）
                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[  # 缓存序列长度 # 序列长度
                    "cache_seqlens"  # 键名 # 键
                ][:bs]  # 切片到当前batch大小 # 取前bs个
                metadata.cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][  # Q累积序列长度 # Q的cumsum
                    : bs + 1  # 切片 # 取前bs+1个
                ]
                metadata.cu_seqlens_k = self.decode_cuda_graph_metadata["cu_seqlens_k"][  # KV累积序列长度 # KV的cumsum
                    : bs + 1  # 切片 # 取前bs+1个
                ]
                metadata.page_table = self.decode_cuda_graph_metadata[  # 页表 # 页表
                    "page_table_draft_decode"  # 草稿解码页表键名 # draft decode页表
                ][:bs, :]  # 切片 # 取前bs行
                self._bind_swa_page_table(  # 绑定SWA页表 # 绑定SWA页表
                    metadata,  # 元数据 # 元数据
                    self.decode_cuda_graph_metadata,  # 源字典 # 源字典
                    "swa_page_table_draft_decode",  # 键名 # 键
                    bs,  # batch大小 # 批次大小
                )
                self.decode_cuda_graph_metadata[bs] = metadata  # 缓存元数据 # 缓存
            else:  # 普通解码 # normal decode
                # Normal Decode # 普通解码
                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[  # 缓存序列长度 # 序列长度
                    "cache_seqlens"  # 键名 # 键
                ][:bs]  # 切片 # 取前bs个
                metadata.cu_seqlens_q = torch.arange(  # Q累积序列长度 # Q的cumsum
                    0, bs + 1, dtype=torch.int32, device=device  # 0到bs+1 # 范围
                )
                metadata.cu_seqlens_k = torch.zeros(  # KV累积序列长度 # KV的cumsum
                    bs + 1, dtype=torch.int32, device=device  # 零初始化 # 零值
                )
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][  # 页表 # 页表
                    :bs, :  # 切片 # 取前bs行
                ]
                self._bind_swa_page_table(  # 绑定SWA页表 # 绑定SWA页表
                    metadata,  # 元数据 # 元数据
                    self.decode_cuda_graph_metadata,  # 源字典 # 源字典
                    "swa_page_table",  # 键名 # 键
                    bs,  # batch大小 # 批次大小
                )
                self.decode_cuda_graph_metadata[bs] = metadata  # 缓存元数据 # 缓存
        elif forward_mode.is_target_verify():  # 目标验证模式 # target verify
            # Target Verify (topk = 1) # 目标验证（topk = 1）
            tokens_per_req = num_tokens // bs  # 每个请求的token数 # 每请求token数
            metadata.cache_seqlens_int32 = self.target_verify_metadata["cache_seqlens"][  # 缓存序列长度 # 序列长度
                :bs  # 切片 # 取前bs个
            ]
            metadata.cu_seqlens_q = self.target_verify_metadata["cu_seqlens_q"][  # Q累积序列长度 # Q的cumsum
                : bs + 1  # 切片 # 取前bs+1个
            ]
            metadata.cu_seqlens_k = self.target_verify_metadata["cu_seqlens_k"][  # KV累积序列长度 # KV的cumsum
                : bs + 1  # 切片 # 取前bs+1个
            ]
            metadata.max_seq_len_q = tokens_per_req  # 最大查询序列长度 # Q最大长度
            metadata.page_table = self.target_verify_metadata["page_table"][:bs, :]  # 页表 # 页表
            self._bind_swa_page_table(  # 绑定SWA页表 # 绑定SWA页表
                metadata,  # 元数据 # 元数据
                self.target_verify_metadata,  # 源字典 # 源字典
                "swa_page_table",  # 键名 # 键
                bs,  # batch大小 # 批次大小
            )
            self.target_verify_metadata[bs] = metadata  # 缓存元数据 # 缓存
        elif forward_mode.is_draft_extend():  # 草稿扩展模式 # draft extend
            num_tokens_per_bs = num_tokens // bs  # 每个请求的token数 # 每请求token数
            metadata.cache_seqlens_int32 = self.draft_extend_metadata["cache_seqlens"][  # 缓存序列长度 # 序列长度
                :bs  # 切片 # 取前bs个
            ]
            metadata.cu_seqlens_q = self.draft_extend_metadata["cu_seqlens_q"][: bs + 1]  # Q累积序列长度 # Q的cumsum
            metadata.cu_seqlens_k = self.draft_extend_metadata["cu_seqlens_k"][: bs + 1]  # KV累积序列长度 # KV的cumsum
            metadata.max_seq_len_q = num_tokens_per_bs  # 最大查询序列长度 # Q最大长度
            metadata.page_table = self.draft_extend_metadata["page_table"][:bs, :]  # 页表 # 页表
            self._bind_swa_page_table(  # 绑定SWA页表 # 绑定SWA页表
                metadata,  # 元数据 # 元数据
                self.draft_extend_metadata,  # 源字典 # 源字典
                "swa_page_table",  # 键名 # 键
                bs,  # batch大小 # 批次大小
            )
            self.draft_extend_metadata[bs] = metadata  # 缓存元数据 # 缓存

        return metadata  # 返回元数据 # 返回

    def init_forward_metadata_capture_cuda_graph(  # 初始化CUDA图捕获时的前向元数据 # 为CUDA图捕获准备元数据
        self,
        bs: int,  # batch大小 # 批次大小
        num_tokens: int,  # token数量 # token数
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求索引
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度
        encoder_lens: Optional[torch.Tensor],  # 编码器长度 # 编码器长度
        forward_mode: ForwardMode,  # 前向模式 # 前向模式
        spec_info: Optional[SpecInput],  # 推测信息 # 推测解码信息
    ):
        """Initialize metadata for CUDA graph capture.""" # 为CUDA图捕获初始化元数据 """初始化捕获元数据"""
        seq_lens_cpu = seq_lens.cpu()  # 将序列长度转到CPU # 序列长度CPU副本
        self._build_cuda_graph_metadata(  # 构建CUDA图元数据 # 构建元数据
            bs, num_tokens, forward_mode, spec_info, seq_lens.device  # 参数 # 传参
        )
        self.init_forward_metadata_replay_cuda_graph(  # 初始化回放元数据 # 初始化回放
            bs=bs,  # batch大小 # 批次大小
            req_pool_indices=req_pool_indices,  # 请求池索引 # 请求索引
            seq_lens=seq_lens,  # 序列长度 # 序列长度
            seq_lens_sum=None,  # 序列长度总和 # 总和
            encoder_lens=encoder_lens,  # 编码器长度 # 编码器长度
            forward_mode=forward_mode,  # 前向模式 # 模式
            spec_info=spec_info,  # 推测信息 # 推测信息
            seq_lens_cpu=seq_lens_cpu,  # CPU序列长度 # CPU序列长度
        )
        if forward_mode.is_draft_extend():  # 如果是草稿扩展模式 # draft extend模式
            # CUDA graph bakes max_seq_len_q as a constant.  replay() sets it to # CUDA图将max_seq_len_q烘焙为常量。replay()将其设置为
            # max(num_accept_tokens_cpu) which is None/empty at capture time, # max(num_accept_tokens_cpu)，这在捕获时为None/空，
            # falling back to 1.  Restore the correct upper bound so the kernel # 回退为1。恢复正确的上限，以便内核
            # sees num_tokens_per_bs (not 1) for all replays of this graph. # 在该图的所有回放中看到num_tokens_per_bs（而非1）。
            self.forward_metadata.max_seq_len_q = num_tokens // bs  # 恢复正确的max_seq_len_q # 修正Q最大长度

    def init_forward_metadata_replay_cuda_graph(  # 初始化CUDA图回放时的前向元数据 # 使用新输入回放CUDA图
        self,
        bs: int,  # batch大小 # 批次大小
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求索引
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度
        seq_lens_sum: int,  # 序列长度总和 # 序列长度和
        encoder_lens: Optional[torch.Tensor],  # 编码器长度 # 编码器长度
        forward_mode: ForwardMode,  # 前向模式 # 前向模式
        spec_info: Optional[SpecInput],  # 推测信息 # 推测信息
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度 # CPU端序列长度
    ):
        """Replay CUDA graph with new inputs.""" # 使用新输入回放CUDA图 """回放CUDA图"""
        seq_lens = seq_lens[:bs]  # 截取到当前batch大小 # 截取序列长度
        seq_lens_cpu = seq_lens_cpu[:bs]  # 截取CPU序列长度 # 截取CPU副本
        req_pool_indices = req_pool_indices[:bs]  # 截取请求池索引 # 截取请求索引
        metadata = None  # 元数据初始化 # 元数据
        if forward_mode.is_decode_or_idle():  # 解码或空闲模式 # decode模式
            if spec_info is not None:  # 如果有推测信息 # draft decode
                # Draft Decode # 草稿解码
                # Here we only support topk = 1 for now. # 目前仅支持topk = 1
                metadata = self.decode_cuda_graph_metadata[bs]  # 获取缓存的元数据 # 读取元数据
                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[  # 缓存序列长度 # 序列长度
                    "cache_seqlens"  # 键名 # 键
                ][:bs]  # 切片 # 取前bs个
                metadata.cache_seqlens_int32.copy_(  # 复制序列长度+推测步偏移 # 写入序列长度
                    seq_lens + self.speculative_step_id + 1  # 加上推测步偏移 # 加偏移
                )
                metadata.max_seq_len_k = seq_lens.max().item() + (  # 最大KV序列长度 # KV最大长度
                    self.speculative_step_id + 1  # 加上推测步偏移 # 加偏移
                )

                max_seq_pages = (  # 计算最大序列页数 # 最大页数
                    metadata.max_seq_len_k + self.page_size - 1  # 向上取整计算 # 向上取整
                ) // self.page_size  # 整除页大小 # 除以页大小
            else:  # 普通解码 # normal decode
                # Normal Decode # 普通解码
                metadata = self.decode_cuda_graph_metadata[bs]  # 获取缓存的元数据 # 读取元数据
                max_len = seq_lens_cpu.max().item()  # 获取最大序列长度 # CPU端最大长度
                max_seq_pages = (max_len + self.page_size - 1) // self.page_size  # 计算最大页数 # 最大页数
                metadata.max_seq_len_k = max_len  # 设置最大KV序列长度 # KV最大长度

                metadata.cache_seqlens_int32.copy_(seq_lens)  # 复制序列长度 # 写入序列长度

            metadata.cu_seqlens_k[1:].copy_(  # 更新KV累积序列长度 # 更新cumsum
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)  # 计算累积和 # cumsum
            )
            page_indices = self.req_to_token[  # 获取页索引 # 读取token到页的映射
                req_pool_indices[:, None],  # batch维度索引 # 请求索引
                self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages][  # 步幅索引 # 步幅索引
                    None, :  # 增加维度 # 增加维度
                ],
            ]
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)  # 复制页表（整除页大小） # 写入页表
            self._copy_swa_page_table(metadata, page_indices, max_seq_pages)  # 复制SWA页表 # 写入SWA页表
        elif forward_mode.is_target_verify():  # 目标验证模式 # target verify
            # Here we only support topk = 1 for now. # 目前仅支持topk = 1
            metadata = self.target_verify_metadata[bs]  # 获取缓存的元数据 # 读取元数据
            metadata.cache_seqlens_int32.copy_(seq_lens + metadata.max_seq_len_q)  # 复制序列长度+Q长度 # 写入序列长度

            metadata.max_seq_len_k = seq_lens_cpu.max().item() + metadata.max_seq_len_q  # 最大KV序列长度 # KV最大长度
            max_len = seq_lens_cpu.max().item()  # CPU端最大长度 # 最大长度
            metadata.cu_seqlens_k[1:].copy_(  # 更新KV累积序列长度 # 更新cumsum
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)  # 计算累积和 # cumsum
            )
            max_seq_pages = (  # 计算最大页数 # 最大页数
                metadata.max_seq_len_k + self.page_size - 1  # 向上取整 # 向上取整
            ) // self.page_size  # 整除页大小 # 除以页大小
            page_indices = self.req_to_token[  # 获取页索引 # 读取映射
                req_pool_indices[:, None],  # batch维度索引 # 请求索引
                self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages],  # 步幅索引 # 步幅索引
            ]
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)  # 复制页表 # 写入页表
            self._copy_swa_page_table(metadata, page_indices, max_seq_pages)  # 复制SWA页表 # 写入SWA页表
        elif forward_mode.is_draft_extend():  # 草稿扩展模式 # draft extend
            metadata = self.draft_extend_metadata[bs]  # 获取缓存的元数据 # 读取元数据
            metadata.cache_seqlens_int32.copy_(seq_lens)  # 复制序列长度 # 写入序列长度

            metadata.max_seq_len_k = seq_lens_cpu.max().item()  # 最大KV序列长度 # KV最大长度
            max_len = seq_lens_cpu.max().item()  # CPU端最大长度 # 最大长度
            metadata.cu_seqlens_k[1:].copy_(  # 更新KV累积序列长度 # 更新cumsum
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)  # 计算累积和 # cumsum
            )
            extend_lens = spec_info.num_accept_tokens[:bs]  # 获取扩展长度 # 接受token数
            if spec_info.num_accept_tokens_cpu:  # 如果有CPU端的接受token数 # CPU端数据
                metadata.max_seq_len_q = max(spec_info.num_accept_tokens_cpu)  # 使用CPU端最大值 # CPU最大值
            else:  # 无CPU端数据 # 无数据
                metadata.max_seq_len_q = 1  # 默认为1 # 默认值

            metadata.cu_seqlens_q[1:].copy_(  # 更新Q累积序列长度 # 更新Q的cumsum
                torch.cumsum(extend_lens, dim=0, dtype=torch.int32)  # 计算累积和 # cumsum
            )

            max_seq_pages = (  # 计算最大页数 # 最大页数
                metadata.max_seq_len_k + self.page_size - 1  # 向上取整 # 向上取整
            ) // self.page_size  # 整除页大小 # 除以页大小
            page_indices = self.req_to_token[  # 获取页索引 # 读取映射
                req_pool_indices[:, None],  # batch维度索引 # 请求索引
                self.draft_extend_metadata["strided_indices"][:max_seq_pages],  # 步幅索引 # 步幅索引
            ]
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)  # 复制页表 # 写入页表
            self._copy_swa_page_table(metadata, page_indices, max_seq_pages)  # 复制SWA页表 # 写入SWA页表
        self.forward_metadata = metadata  # 保存元数据 # 更新前向元数据

    def update_verify_buffers_to_fill_after_draft(  # 草稿后更新验证缓冲区 # 预留接口（空实现）
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]  # 推测信息和CUDA图batch大小 # 参数
    ):
        pass  # 空实现 # 预留

    def get_cuda_graph_seq_len_fill_value(self) -> int:  # 获取CUDA图序列长度填充值 # 返回序列长度填充值
        """Get the fill value for sequence lengths in CUDA graph.""" # 获取CUDA图中序列长度的填充值 """序列长度填充值"""
        return 1  # 填充值1 # 返回1

    def _should_use_fused_fp8_path(self, save_kv_cache: bool, k: torch.Tensor) -> bool:  # 判断是否使用FP8融合路径 # 检查FP8融合条件
        """Check if we should use the fused FP8 KV cache write path.""" # 检查是否应使用融合FP8 KV缓存写入路径 """FP8融合路径判断"""
        return save_kv_cache and k is not None and self.data_type == torch.float8_e4m3fn  # 条件：保存KV缓存、K非空、FP8数据类型 # FP8条件

    def _fused_fp8_set_kv_buffer(  # 融合FP8量化并写入KV缓存 # 量化K/V并写入FP8缓存
        self,
        q: torch.Tensor,  # 查询张量 # Query
        k: torch.Tensor,  # 键张量 # Key
        v: torch.Tensor,  # 值张量 # Value
        layer: RadixAttention,  # 注意力层 # 当前层
        forward_batch: ForwardBatch,  # 前向批次 # 前向计算批次
        **kwargs,  # 额外参数 # 其他参数
    ):
        """Fused FP8 quantization and KV cache write.""" # 融合FP8量化和KV缓存写入 """FP8量化+缓存写入"""
        cache_loc = self._get_layer_cache_loc(layer, forward_batch)  # 获取层的缓存位置 # 读取缓存位置

        # Get K/V cache buffers from token_to_kv_pool # 从token_to_kv_pool获取K/V缓存缓冲区
        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)  # 获取K/V缓存 # 读取缓存

        fused_fp8_set_kv_buffer(  # 调用融合FP8写入 # 执行FP8写入
            k=k,  # Key张量 # Key
            v=v,  # Value张量 # Value
            k_cache=k_cache,  # Key缓存 # K缓存
            v_cache=v_cache,  # Value缓存 # V缓存
            cache_loc=cache_loc,  # 缓存位置 # 缓存位置
            k_scale=layer.k_scale,  # May be None # K缩放因子（可能为None）
            v_scale=layer.v_scale,  # May be None # V缩放因子（可能为None）
            page_size=self.page_size,  # 页大小 # 页大小
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):  # 初始化前向元数据 # 为前向计算准备元数据
        """Initialize the metadata for a forward pass.""" # 为前向传播初始化元数据 """初始化前向元数据"""

        metadata = TRTLLMMHAMetadata()  # 创建元数据实例 # 新建元数据
        seqlens_in_batch = forward_batch.seq_lens  # 获取批次序列长度 # 批次序列长度
        batch_size = forward_batch.batch_size  # 获取batch大小 # 批次大小
        device = seqlens_in_batch.device  # 获取设备 # 计算设备

        if forward_batch.forward_mode.is_decode_or_idle():  # 解码或空闲模式 # decode模式
            if forward_batch.spec_info is not None:  # 如果有推测信息 # draft decode
                # Draft Decode # 草稿解码
                # Here we only support topk = 1 for now. # 目前仅支持topk = 1
                metadata.cache_seqlens_int32 = (  # 缓存序列长度 # 序列长度
                    seqlens_in_batch + (self.speculative_step_id + 1)  # 加上推测步偏移 # 加偏移
                ).to(torch.int32)  # 转为int32 # 类型转换
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (  # 最大KV序列长度 # KV最大长度
                    self.speculative_step_id + 1  # 加上推测步偏移 # 加偏移
                )
                metadata.cu_seqlens_q = torch.arange(  # Q累积序列长度 # Q的cumsum
                    0, batch_size + 1, dtype=torch.int32, device=device  # 0到batch_size+1 # 范围
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(  # KV累积序列长度（左填充0） # KV的cumsum
                    torch.cumsum(  # 计算累积和 # cumsum
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32  # 按batch累积 # 沿batch维度
                    ),
                    (1, 0),  # 左填充1个0 # 左填充
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[  # 页表 # 页表
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k  # 请求索引和序列长度 # 索引范围
                ]
            else:  # 普通解码 # normal decode
                # Normal Decode # 普通解码
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)  # 缓存序列长度转int32 # 序列长度
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()  # 最大KV序列长度 # KV最大长度
                metadata.cu_seqlens_q = torch.arange(  # Q累积序列长度 # Q的cumsum
                    0, batch_size + 1, dtype=torch.int32, device=device  # 0到batch_size+1 # 范围
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(  # KV累积序列长度 # KV的cumsum
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)  # 累积和+左填充 # cumsum+pad
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[  # 页表 # 页表
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k  # 请求索引和序列长度 # 索引范围
                ]
        elif forward_batch.forward_mode.is_target_verify():  # 目标验证模式 # target verify
            # Only support topk = 1 for now. # 目前仅支持topk = 1
            tokens_per_req = forward_batch.input_ids.shape[0] // batch_size  # 每个请求的token数 # 每请求token数
            metadata.cache_seqlens_int32 = (forward_batch.seq_lens + tokens_per_req).to(  # 缓存序列长度 # 序列长度
                torch.int32  # 转为int32 # 类型转换
            )
            metadata.max_seq_len_q = tokens_per_req  # 最大查询序列长度 # Q最大长度
            metadata.max_seq_len_k = (  # 最大KV序列长度 # KV最大长度
                forward_batch.seq_lens_cpu.max().item() + tokens_per_req  # CPU最大值+token数 # 加上请求数
            )
            metadata.cu_seqlens_q = torch.arange(  # Q累积序列长度 # Q的cumsum
                0,  # 起始值 # 起点
                batch_size * tokens_per_req + 1,  # 终止值 # 终点
                tokens_per_req,  # 步长 # 步长
                dtype=torch.int32,  # 数据类型 # int32
                device=device,  # 设备 # GPU
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(  # KV累积序列长度 # KV的cumsum
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32),  # 累积和 # cumsum
                (1, 0),  # 左填充 # 左填充1个0
            )
            metadata.page_table = self.req_to_token_pool.req_to_token[  # 页表 # 页表
                forward_batch.req_pool_indices, : metadata.max_seq_len_k  # 请求索引和序列长度 # 索引范围
            ]

        else:  # 扩展模式（prefill/extend） # extend模式
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)  # 缓存序列长度转int32 # 序列长度
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()  # 最大KV序列长度 # KV最大长度
            metadata.cu_seqlens_k = torch.nn.functional.pad(  # KV累积序列长度 # KV的cumsum
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)  # 累积和+左填充 # cumsum+pad
            )
            metadata.page_table = self.req_to_token_pool.req_to_token[  # 页表 # 页表
                forward_batch.req_pool_indices, : metadata.max_seq_len_k  # 请求索引和序列长度 # 索引范围
            ]

            if any(  # 如果有前缀长度或是草稿扩展 # 条件判断
                forward_batch.extend_prefix_lens_cpu
            ) or forward_batch.forward_mode.is_draft_extend(include_v2=True):  # 有前缀或草稿扩展 # extend条件
                extend_seq_lens = forward_batch.extend_seq_lens  # 获取扩展序列长度 # 扩展长度
                # NOTE: in piecewise CUDA graph warmup, extend_seq_lens_cpu is a torch.Tensor; # 注意：在分片CUDA图预热中，extend_seq_lens_cpu是torch.Tensor；
                # Python's max() returns a 0-d tensor, but flashinfer expects an int. # Python的max()返回0维张量，但flashinfer期望int。
                max_q = max(forward_batch.extend_seq_lens_cpu)  # 获取最大扩展长度 # 最大Q长度
                metadata.max_seq_len_q = (  # 最大查询序列长度 # Q最大长度
                    int(max_q.item()) if isinstance(max_q, torch.Tensor) else int(max_q)  # 确保返回int # 类型转换
                )
                metadata.cu_seqlens_q = torch.nn.functional.pad(  # Q累积序列长度 # Q的cumsum
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)  # 累积和+左填充 # cumsum+pad
                )
            else:  # 无前缀的扩展 # 无前缀
                metadata.max_seq_len_q = metadata.max_seq_len_k  # Q最大长度等于KV最大长度 # 等于KV长度
                metadata.cu_seqlens_q = metadata.cu_seqlens_k  # Q累积序列等于KV累积序列 # 等于KV cumsum

        # Compute SWA page table (None for non-SWA models) # 计算SWA页表（非SWA模型为None） # SWA页表计算
        metadata.swa_page_table = self._maybe_translate_swa(metadata.page_table)  # 转换SWA页表索引 # SWA索引转换

        # Convert the page tables to a strided format # 将页表转换为步幅格式 # 步幅格式转换
        if self.page_size > 1:  # 如果页大小大于1 # 分页处理
            self.strided_indices = torch.arange(  # 创建步幅索引 # 步幅索引
                0, metadata.page_table.shape[1], self.page_size, device=self.device  # 按page_size步进 # 按页大小步进
            )
            metadata.page_table = (  # 转换页表为步幅格式 # 步幅格式页表
                metadata.page_table[:, self.strided_indices] // self.page_size  # 按步幅索引取值并整除页大小 # 步幅+整除
            )
            if metadata.swa_page_table is not None:  # 如果有SWA页表 # SWA页表处理
                metadata.swa_page_table = (  # 转换SWA页表为步幅格式 # 步幅格式SWA页表
                    metadata.swa_page_table[:, self.strided_indices] // self.page_size  # 按步幅索引取值并整除页大小 # 步幅+整除
                )

        self.forward_metadata = metadata  # 保存元数据 # 更新前向元数据

    def forward_decode(  # 解码前向计算 # 使用TRTLLM MHA内核执行解码前向计算
        self,
        q: torch.Tensor,  # 查询张量 # Query
        k: torch.Tensor,  # 键张量 # Key
        v: torch.Tensor,  # 值张量 # Value
        layer: RadixAttention,  # 注意力层 # 当前层
        forward_batch: ForwardBatch,  # 前向批次 # 前向计算批次
        save_kv_cache: bool = True,  # 是否保存KV缓存 # 保存缓存标志
        **kwargs,  # 额外参数 # 其他参数
    ) -> torch.Tensor:  # 返回输出张量 # 注意力输出
        """Run forward for decode using TRTLLM MHA kernel.""" # 使用TRTLLM MHA内核运行解码前向计算 """解码前向"""
        cache_loc = forward_batch.out_cache_loc  # 获取缓存位置 # 缓存位置

        use_fused_fp8_path = self._should_use_fused_fp8_path(save_kv_cache, k)  # 判断是否使用FP8融合路径 # FP8路径检查

        if use_fused_fp8_path:  # 使用FP8融合路径 # FP8路径
            # Use fused FP8 quantization + KV cache write path # 使用融合FP8量化+KV缓存写入路径
            self._fused_fp8_set_kv_buffer(  # 调用融合FP8写入 # FP8写入
                q=q,  # 查询 # Query
                k=k,  # 键 # Key
                v=v,  # 值 # Value
                layer=layer,  # 层 # 当前层
                forward_batch=forward_batch,  # 批次 # 前向批次
            )
            k = None  # 置空K，避免重复写入 # K已写入缓存
            v = None  # 置空V，避免重复写入 # V已写入缓存
        else:  # 使用原始路径 # 原始路径
            # Use original set_kv_buffer path # 使用原始set_kv_buffer路径
            if save_kv_cache and k is not None:  # 如果需要保存KV缓存且K非空 # 保存检查
                self.token_to_kv_pool.set_kv_buffer(  # 调用原始KV缓存写入 # 写入KV缓存
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale  # 参数 # 传参
                )

        # For XQA, q_dtype should be bf16 # 对于XQA，q_dtype应为bf16 # XQA查询类型
        if self.data_type == torch.float8_e4m3fn and (not self.is_xqa_impl):  # 如果是FP8且非XQA # FP8非XQA处理
            q = q.to(torch.float8_e4m3fn)  # 转换Q为FP8 # Q转FP8
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)  # 重塑Q形状 # Q形状重塑
        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)  # 获取KV缓存 # 读取缓存
        # shape conversion: # 形状转换：
        # [num_pages, page_size, num_kv_heads, head_dim] -> [num_pages, num_kv_heads, page_size, head_dim] # 页内维度重排
        k_cache = k_cache.view(  # 重塑K缓存形状 # K缓存reshape
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim  # 新形状 # 目标形状
        ).permute(0, 2, 1, 3)  # 维度重排 # 维度置换
        v_cache = v_cache.view(  # 重塑V缓存形状 # V缓存reshape
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim  # 新形状 # 目标形状
        ).permute(0, 2, 1, 3)  # 维度重排 # 维度置换

        if layer.tp_k_head_num == 1:  # 如果K头数为1（MQA） # MQA检查
            k_cache = canonicalize_stride(k_cache)  # 规范化K缓存步长 # 步长规范化
        if layer.tp_v_head_num == 1:  # 如果V头数为1（MQA） # MQA检查
            v_cache = canonicalize_stride(v_cache)  # 规范化V缓存步长 # 步长规范化

        kv_cache = (k_cache, v_cache)  # 组合KV缓存元组 # KV缓存元组

        # TODO: add support for quantization # 待办：添加量化支持 # 量化支持
        q_scale = 1.0  # Q缩放因子默认1.0 # Q缩放
        k_scale = (  # K缩放因子 # K缩放
            layer.k_scale_float  # 使用层的K缩放因子 # 层K缩放
            if getattr(layer, "k_scale_float", None) is not None  # 如果属性存在 # 检查属性
            else 1.0  # 否则默认1.0 # 默认值
        )
        bmm1_scale = q_scale * k_scale * layer.scaling  # BMM1缩放 = Q缩放 * K缩放 * 层缩放 # 注意力缩放
        bmm2_scale = 1.0  # BMM2缩放默认1.0 # BMM2缩放
        # sink: additional value per head in the denominator of the softmax. # sink：softmax分母中每个头的附加值
        attention_sink = kwargs.get("sinks", None)  # 获取注意力汇聚值 # 注意力sink

        page_table = self._get_layer_page_table(layer, forward_batch)  # 获取层的页表 # 读取页表

        # Call TRT-LLM kernel # 调用TRT-LLM内核 # 调用内核
        # raw_out: like q, [bs, acc_q_len, num_q_heads, head_dim] but with output dtype # 原始输出：类似q，[bs, acc_q_len, num_q_heads, head_dim]但为输出dtype
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(  # 调用TRTLLM解码内核 # TRTLLM decode
            query=q,  # 查询张量 # Query
            kv_cache=kv_cache,  # KV缓存 # KV缓存
            workspace_buffer=self.workspace_buffer,  # 工作空间缓冲区 # 工作区
            block_tables=page_table,  # 页表 # 页表
            seq_lens=self.forward_metadata.cache_seqlens_int32,  # 序列长度 # 序列长度
            max_seq_len=self.max_context_len,  # 最大序列长度 # 最大长度
            bmm1_scale=bmm1_scale,  # BMM1缩放因子 # 缩放
            bmm2_scale=bmm2_scale,  # BMM2缩放因子 # 缩放
            window_left=layer.sliding_window_size,  # 滑动窗口大小 # 滑动窗口
            sinks=attention_sink,  # 注意力汇聚 # sink
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),  # 跳过softmax阈值 # softmax阈值
            out_dtype=self.q_data_type,  # model_runner.dtype # 输出数据类型
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)  # 重塑输出并返回 # 展平输出

    def forward_extend(  # 扩展前向计算 # 执行预填充/扩展前向计算
        self,
        q: torch.Tensor,  # 查询张量 # Query
        k: torch.Tensor,  # 键张量 # Key
        v: torch.Tensor,  # 值张量 # Value
        layer: RadixAttention,  # 注意力层 # 当前层
        forward_batch: ForwardBatch,  # 前向批次 # 前向计算批次
        save_kv_cache=True,  # 是否保存KV缓存 # 保存缓存标志
        **kwargs,  # 额外参数 # 其他参数
    ):
        cache_loc = forward_batch.out_cache_loc  # 获取缓存位置 # 缓存位置

        use_fused_fp8_path = self._should_use_fused_fp8_path(save_kv_cache, k)  # 判断是否使用FP8融合路径 # FP8路径检查

        if use_fused_fp8_path:  # 使用FP8融合路径 # FP8路径
            # Use fused FP8 quantization + KV cache write path # 使用融合FP8量化+KV缓存写入路径
            self._fused_fp8_set_kv_buffer(  # 调用融合FP8写入 # FP8写入
                q=q,  # 查询 # Query
                k=k,  # 键 # Key
                v=v,  # 值 # Value
                layer=layer,  # 层 # 当前层
                forward_batch=forward_batch,  # 批次 # 前向批次
            )
            k = None  # 置空K，避免重复写入 # K已写入缓存
            v = None  # 置空V，避免重复写入 # V已写入缓存
        else:  # 使用原始路径 # 原始路径
            # Use original set_kv_buffer path # 使用原始set_kv_buffer路径
            if save_kv_cache and k is not None:  # 如果需要保存KV缓存且K非空 # 保存检查
                self.token_to_kv_pool.set_kv_buffer(  # 调用原始KV缓存写入 # 写入KV缓存
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale  # 参数 # 传参
                )

        if self.data_type == torch.float8_e4m3fn:  # 如果KV缓存为FP8 # FP8处理
            q = q.to(torch.float8_e4m3fn)  # 转换Q为FP8 # Q转FP8
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)  # 重塑Q形状 # Q形状重塑
        # [num_pages, page_size, num_kv_heads, head_dim] -> [num_pages, num_kv_heads, page_size, head_dim] # 页内维度重排
        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)  # 获取KV缓存 # 读取缓存
        k_cache = k_cache.view(  # 重塑K缓存形状 # K缓存reshape
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim  # 新形状 # 目标形状
        ).permute(0, 2, 1, 3)  # 维度重排 # 维度置换
        v_cache = v_cache.view(  # 重塑V缓存形状 # V缓存reshape
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim  # 新形状 # 目标形状
        ).permute(0, 2, 1, 3)  # 维度重排 # 维度置换

        if layer.tp_k_head_num == 1:  # 如果K头数为1（MQA） # MQA检查
            k_cache = canonicalize_stride(k_cache)  # 规范化K缓存步长 # 步长规范化
        if layer.tp_v_head_num == 1:  # 如果V头数为1（MQA） # MQA检查
            v_cache = canonicalize_stride(v_cache)  # 规范化V缓存步长 # 步长规范化

        kv_cache = (k_cache, v_cache)  # 组合KV缓存元组 # KV缓存元组

        # sink: additional value per head in the denominator of the softmax. # sink：softmax分母中每个头的附加值
        attention_sink = kwargs.get("sinks", None)  # 获取注意力汇聚值 # 注意力sink
        # TODO: add support for quantization # 待办：添加量化支持 # 量化支持
        q_scale = 1.0  # Q缩放因子默认1.0 # Q缩放
        k_scale = (  # K缩放因子 # K缩放
            layer.k_scale_float  # 使用层的K缩放因子 # 层K缩放
            if getattr(layer, "k_scale_float", None) is not None  # 如果属性存在 # 检查属性
            else 1.0  # 否则默认1.0 # 默认值
        )
        bmm1_scale = q_scale * k_scale * layer.scaling  # BMM1缩放 # 注意力缩放
        bmm2_scale = 1.0  # BMM2缩放 # BMM2缩放

        page_table = self._get_layer_page_table(layer, forward_batch)  # 获取层的页表 # 读取页表

        if forward_batch.forward_mode.is_target_verify():  # 目标验证模式 # target verify
            o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(  # 调用TRTLLM解码内核（验证模式） # decode内核
                query=q,  # 查询张量 # Query
                kv_cache=kv_cache,  # KV缓存 # KV缓存
                workspace_buffer=self.workspace_buffer,  # 工作空间缓冲区 # 工作区
                block_tables=page_table,  # 页表 # 页表
                seq_lens=self.forward_metadata.cache_seqlens_int32,  # 序列长度 # 序列长度
                max_seq_len=self.max_context_len,  # 最大序列长度 # 最大长度
                bmm1_scale=bmm1_scale,  # BMM1缩放因子 # 缩放
                bmm2_scale=bmm2_scale,  # BMM2缩放因子 # 缩放
                window_left=layer.sliding_window_size,  # 滑动窗口大小 # 滑动窗口
                sinks=attention_sink,  # 注意力汇聚 # sink
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),  # 跳过softmax阈值 # softmax阈值
                out_dtype=self.q_data_type,  # model_runner.dtype # 输出数据类型
                q_len_per_req=self.forward_metadata.max_seq_len_q,  # 每请求Q长度 # Q长度
            )
        else:  # 预填充/扩展模式 # prefill模式
            o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(  # 调用TRTLLM上下文内核 # context内核
                query=q,  # 查询张量 # Query
                kv_cache=kv_cache,  # KV缓存 # KV缓存
                workspace_buffer=self.workspace_buffer,  # 工作空间缓冲区 # 工作区
                block_tables=page_table,  # 页表 # 页表
                seq_lens=self.forward_metadata.cache_seqlens_int32,  # 序列长度 # 序列长度
                max_q_len=self.forward_metadata.max_seq_len_q,  # 最大Q长度 # Q最大长度
                max_kv_len=self.max_context_len,  # 最大KV长度 # KV最大长度
                bmm1_scale=bmm1_scale,  # BMM1缩放因子 # 缩放
                bmm2_scale=bmm2_scale,  # BMM2缩放因子 # 缩放
                batch_size=self.forward_metadata.cu_seqlens_q.shape[0] - 1,  # batch大小 # 批次大小
                cum_seq_lens_q=self.forward_metadata.cu_seqlens_q,  # Q累积序列长度 # Q cumsum
                cum_seq_lens_kv=self.forward_metadata.cu_seqlens_k,  # KV累积序列长度 # KV cumsum
                window_left=layer.sliding_window_size,  # 滑动窗口大小 # 滑动窗口
                sinks=attention_sink,  # 注意力汇聚 # sink
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),  # 跳过softmax阈值 # softmax阈值
                out_dtype=self.q_data_type,  # model_runner.dtype # 输出数据类型
            )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)  # 重塑输出并返回 # 展平输出


class TRTLLMHAAttnMultiStepDraftBackend(FlashInferMultiStepDraftBackend):  # TRTLLM MHA多步草稿后端类 # EAGLE多步草稿解码后端
    """Multi-step TRTLLM MHA attention kernel used by EAGLE.""" # EAGLE使用的多步TRTLLM MHA注意力内核 # EAGLE多步注意力

    def __init__(  # 初始化方法 # 构造函数
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int  # 模型运行器、topk值、推测步数 # 参数
    ):
        super().__init__(model_runner, topk, speculative_num_steps)  # 调用父类初始化 # 初始化父类
        for i in range(self.speculative_num_steps - 1):  # 为每个推测步创建后端 # 循环创建后端
            self.attn_backends[i] = TRTLLMHAAttnBackend(  # 创建TRTLLM MHA后端实例 # 新建后端
                model_runner,  # 模型运行器 # 模型运行器
                skip_prefill=True,  # 跳过预填充 # 跳过prefill
                kv_indptr_buf=self.kv_indptr[i],  # KV索引指针 # KV indptr
                kv_last_page_len_buf=self.kv_last_page_len,  # KV最后一页长度 # 最后页长度
                speculative_step_id=i,  # 推测步ID # 步骤ID
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):  # 初始化前向元数据 # 为所有步初始化元数据
        for i in range(self.speculative_num_steps - 1):  # 遍历所有推测步 # 循环
            self.attn_backends[i].init_forward_metadata(forward_batch)  # 初始化每步的元数据 # 每步初始化

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):  # 初始化CUDA图状态 # 为所有步初始化CUDA图
        for i in range(self.speculative_num_steps - 1):  # 遍历所有推测步 # 循环
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)  # 初始化每步的CUDA图状态 # 每步初始化

    def init_forward_metadata_capture_cuda_graph(  # 初始化CUDA图捕获元数据 # 为CUDA图捕获准备元数据
        self,
        forward_batch: ForwardBatch,  # 前向批次 # 前向计算批次
    ):
        assert forward_batch.spec_info is not None  # 断言推测信息存在 # 验证spec_info
        assert forward_batch.spec_info.is_draft_input()  # 断言为草稿输入 # 验证draft模式

        for i in range(self.speculative_num_steps - 1):  # 遍历所有推测步 # 循环
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(  # 初始化每步的捕获元数据 # 每步捕获
                forward_batch.batch_size,  # batch大小 # 批次大小
                forward_batch.batch_size * self.topk,  # token数量（batch*topk） # token数
                forward_batch.req_pool_indices,  # 请求池索引 # 请求索引
                forward_batch.seq_lens,  # 序列长度 # 序列长度
                encoder_lens=forward_batch.encoder_lens,  # 编码器长度 # 编码器长度
                forward_mode=ForwardMode.DECODE,  # 前向模式为解码 # 解码模式
                spec_info=forward_batch.spec_info,  # 推测信息 # 推测信息
            )

    def init_forward_metadata_replay_cuda_graph(  # 初始化CUDA图回放元数据 # 使用新输入回放CUDA图
        self, forward_batch: ForwardBatch, bs: int  # 前向批次和batch大小 # 参数
    ):
        assert forward_batch.spec_info is not None  # 断言推测信息存在 # 验证spec_info
        assert forward_batch.spec_info.is_draft_input()  # 断言为草稿输入 # 验证draft模式

        for i in range(self.speculative_num_steps - 1):  # 遍历所有推测步 # 循环

            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(  # 初始化每步的回放元数据 # 每步回放
                bs,  # batch大小 # 批次大小
                forward_batch.req_pool_indices,  # 请求池索引 # 请求索引
                forward_batch.seq_lens,  # 序列长度 # 序列长度
                forward_batch.seq_lens_sum,  # 序列长度总和 # 总和
                encoder_lens=forward_batch.encoder_lens,  # 编码器长度 # 编码器长度
                forward_mode=ForwardMode.DECODE,  # 前向模式为解码 # 解码模式
                spec_info=forward_batch.spec_info,  # 推测信息 # 推测信息
                seq_lens_cpu=forward_batch.seq_lens_cpu,  # CPU序列长度 # CPU端序列长度
            )
