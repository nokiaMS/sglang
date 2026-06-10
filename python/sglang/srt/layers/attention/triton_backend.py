# 文件说明：Triton注意力后端实现
# 该模块实现了基于Triton内核的注意力机制后端，是SGLang中最重要的注意力后端之一。
# 支持解码（decode）、扩展（extend/prefill）、投机解码（speculative decoding）、
# 滑动窗口注意力（SWA）、CUDA图捕获与重放、确定性推理等功能。
# 包含TritonAttnBackend（主后端）和TritonMultiStepDraftBackend（多步草稿后端）两个核心类。

from __future__ import annotations  # 启用延迟注解评估 # 启用延迟类型注解

from dataclasses import dataclass  # 导入数据类装饰器 # 导入数据类
from typing import TYPE_CHECKING, List, Optional  # 导入类型提示 # 导入类型检查、列表和可选类型

import torch  # 导入PyTorch # 导入PyTorch框架
import triton  # 导入Triton编译器 # 导入Triton
import triton.language as tl  # 导入Triton语言 # 导入Triton语言

from sglang.srt.configs.model_config import AttentionArch  # 导入注意力架构枚举 # 导入注意力架构枚举
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend  # 导入注意力后端基类 # 导入注意力后端基类
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton  # 导入KV索引创建工具 # 导入KV索引创建函数
from sglang.srt.layers.dp_attention import get_attention_tp_size  # 导入注意力张量并行大小获取 # 导入注意力TP大小
from sglang.srt.layers.radix_attention import AttentionType  # 导入注意力类型枚举 # 导入注意力类型枚举
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool  # 导入滑动窗口KV内存池 # 导入SWA KV池
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode  # 导入前向批次信息和模式 # 导入前向批次信息
from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices  # 导入草稿解码KV索引生成 # 导入草稿解码KV索引生成
from sglang.srt.utils import (  # 导入工具函数 # 导入工具函数
    get_bool_env_var,  # 获取布尔环境变量 # 获取布尔环境变量
    get_device_core_count,  # 获取设备核心数 # 获取设备核心数
    get_int_env_var,  # 获取整数环境变量 # 获取整数环境变量
    is_cuda,  # 检查是否为CUDA设备 # 是否CUDA
    next_power_of_2,  # 计算下一个2的幂 # 下一个2的幂
)

_is_cuda = is_cuda()  # 检查当前是否为CUDA设备 # 检查CUDA可用性

if _is_cuda:  # 如果是CUDA设备 # CUDA条件导入
    from sgl_kernel.utils import is_arch_support_pdl  # 导入PDL架构支持检测 # 导入PDL支持检测

if TYPE_CHECKING:  # 如果是类型检查阶段 # 类型检查时才导入
    from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力类 # 导入基数注意力类
    from sglang.srt.model_executor.model_runner import ModelRunner  # 导入模型运行器 # 导入模型运行器类
    from sglang.srt.speculative.spec_info import SpecInput  # 导入投机解码规格信息 # 导入投机解码输入规格类


_MLA_DECODE_MIN_BLOCK_KV = 32  # MLA解码最小KV块大小常量 # MLA解码最小KV块大小


def _mla_decode_kv_splits_cap(  # 计算MLA解码KV分片上限 # MLA解码KV分片上限计算
    base_max_kv_splits: int, sm_count: int, max_context_len: int  # 基础最大分片数、SM数量、最大上下文长度 # 基础最大分片数、SM数、最大上下文长度
) -> int:  # 返回KV分片上限 # 返回上限值
    if sm_count <= 0:  # 如果SM数量小于等于0 # 检查SM数量
        return base_max_kv_splits  # 返回基础最大分片数 # 返回基础值
    sm_cap = next_power_of_2(sm_count)  # 计算SM数量的2的幂上限 # SM数量2的幂上限
    ctx_cap = next_power_of_2(triton.cdiv(max_context_len, _MLA_DECODE_MIN_BLOCK_KV))  # 计算上下文长度的2的幂上限 # 上下文2的幂上限
    return max(base_max_kv_splits, min(sm_cap, ctx_cap))  # 返回SM上限和上下文上限中的较小者与基础值的较大者 # 取最大值


def logit_capping_mod(logit_capping_method, logit_cap):  # logits上限修饰函数 # logits上限修饰函数
    # positive logit_cap -> tanh cap
    # 正的logit_cap -> tanh上限
    if logit_capping_method == "tanh":  # 如果使用tanh上限方法 # 检查tanh方法
        return logit_cap  # 返回logit上限值 # 返回上限值
    else:  # 否则 # 其他方法
        raise ValueError()  # 抛出数值错误 # 抛出异常


@dataclass  # 数据类装饰器 # 数据类
class ForwardMetadata:  # 前向元数据类 # 前向元数据
    attn_logits: torch.Tensor  # 注意力logits张量 # 注意力logits
    attn_lse: torch.Tensor  # 注意力对数求和指数张量 # 注意力LSE
    max_extend_len: int  # 最大扩展长度 # 最大扩展长度
    num_kv_splits: torch.Tensor  # KV分片数张量 # KV分片数
    kv_indptr: torch.Tensor  # KV索引指针张量 # KV索引指针
    kv_indices: torch.Tensor  # KV索引张量 # KV索引
    qo_indptr: torch.Tensor  # 查询输出索引指针张量 # QO索引指针
    custom_mask: torch.Tensor  # 自定义掩码张量 # 自定义掩码
    mask_indptr: torch.Tensor  # 掩码索引指针张量 # 掩码索引指针
    # Sliding window
    # 滑动窗口
    window_kv_indptr: torch.Tensor  # 窗口KV索引指针张量 # 窗口KV索引指针
    window_kv_indices: torch.Tensor  # 窗口KV索引张量 # 窗口KV索引
    window_num_kv_splits: torch.Tensor  # 窗口KV分片数张量 # 窗口KV分片数
    window_kv_offsets: torch.Tensor  # 窗口KV偏移量张量 # 窗口KV偏移量
    # Separate attn_logits for SWA layers when v_head_dim differs
    # 当v_head_dim不同时，SWA层使用的独立attn_logits
    swa_attn_logits: Optional[torch.Tensor] = None  # SWA注意力logits，默认为空 # SWA注意力logits


class TritonAttnBackend(AttentionBackend):  # Triton注意力后端类，继承自注意力后端基类 # Triton注意力后端类
    # CUDA-graph replay rebuilds metadata from preallocated kv_indptr/kv_indices
    # buffers; it never reads seq_lens_cpu / seq_lens_sum.
    # CUDA图重放从预分配的kv_indptr/kv_indices缓冲区重建元数据；
    # 它从不读取seq_lens_cpu / seq_lens_sum。
    needs_cpu_seq_lens: bool = False  # 不需要CPU序列长度 # 不需要CPU序列长度

    def __init__(  # 初始化方法 # 初始化方法
        self,
        model_runner: ModelRunner,  # 模型运行器 # 模型运行器实例
        skip_prefill: bool = False,  # 是否跳过预填充 # 是否跳过预填充阶段
        kv_indptr_buf: Optional[torch.Tensor] = None,  # KV索引指针缓冲区 # KV索引指针缓冲区（可选）
    ):
        # Lazy import to avoid the initialization of cuda context
        # 延迟导入以避免CUDA上下文的初始化
        from sglang.srt.layers.attention.triton_ops.decode_attention import (  # 导入解码注意力前向 # 导入解码注意力
            decode_attention_fwd,  # 解码注意力前向函数 # 解码注意力前向
        )
        from sglang.srt.layers.attention.triton_ops.extend_attention import (  # 导入扩展注意力相关函数 # 导入扩展注意力
            build_unified_kv_indices,  # 构建统一KV索引函数 # 构建统一KV索引
            extend_attention_fwd,  # 扩展注意力前向函数 # 扩展注意力前向
            extend_attention_fwd_unified,  # 统一扩展注意力前向函数 # 统一扩展注意力前向
        )

        super().__init__()  # 调用父类初始化 # 调用基类初始化

        self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)  # 禁用编译器对解码注意力的跟踪 # 禁用编译器跟踪解码注意力
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)  # 禁用编译器对扩展注意力的跟踪 # 禁用编译器跟踪扩展注意力
        self.extend_attention_fwd_unified = torch.compiler.disable(  # 禁用编译器对统一扩展注意力的跟踪 # 禁用编译器跟踪统一扩展注意力
            extend_attention_fwd_unified  # 统一扩展注意力前向函数 # 统一扩展注意力前向
        )
        self.build_unified_kv_indices = torch.compiler.disable(build_unified_kv_indices)  # 禁用编译器对统一KV索引构建的跟踪 # 禁用编译器跟踪统一KV索引

        # Parse args
        # 解析参数
        self.skip_prefill = skip_prefill  # 保存是否跳过预填充标志 # 保存跳过预填充标志
        max_bs = model_runner.req_to_token_pool.size  # 获取最大批次大小 # 最大批次大小
        self.sliding_window_size = model_runner.sliding_window_size  # 保存滑动窗口大小 # 保存滑动窗口大小
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        # 池引用——在构造时捕获，以便在删除对应的ForwardBatch字段后仍能存活。
        self.req_to_token_pool = model_runner.req_to_token_pool  # 保存请求到token池 # 请求-token映射池
        self.token_to_kv_pool = model_runner.token_to_kv_pool  # 保存token到KV池 # KV缓存池
        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 保存请求到token映射表 # 请求-token映射表
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator  # 保存KV池分配器 # KV池分配器
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens  # 保存草稿token数 # 草稿token数
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps  # 保存投机步数 # 投机步数
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA  # 判断是否使用MLA # 是否MLA
        self.num_head = (  # 计算注意力头数 # 注意力头数
            model_runner.model_config.num_attention_heads // get_attention_tp_size()  # 总头数除以TP大小 # 总头数/TP大小
        )
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(  # 计算KV头数 # KV头数
            get_attention_tp_size()  # 注意力TP大小 # TP大小
        )
        # The decode triton kernel derives attn_lse offsets from attn_logits
        # strides via integer division by v_head_dim (the "// Lv" trick in
        # _fwd_kernel_stage1/stage2), so attn_logits.shape[-1] must exactly
        # match the layer's v_head_dim. For hybrid SWA models where SWA and
        # full-attention layers use different v_head_dim (e.g. Gemma 4:
        # swa=256, full=512), we allocate a second buffer for SWA layers.
        # 解码Triton内核通过attn_logits步幅除以v_head_dim的整数除法（即
        # _fwd_kernel_stage1/stage2中的"// Lv"技巧）推导attn_lse偏移量，
        # 因此attn_logits.shape[-1]必须精确匹配层的v_head_dim。对于SWA
        # 和全注意力层使用不同v_head_dim的混合SWA模型（如Gemma 4:
        # swa=256, full=512），我们为SWA层分配第二个缓冲区。
        full_v_head_dim = model_runner.model_config.v_head_dim  # 获取全注意力V头维度 # 全注意力V头维度
        swa_v_head_dim = model_runner.model_config.swa_v_head_dim  # 获取SWA V头维度 # SWA V头维度
        if self.sliding_window_size is not None and swa_v_head_dim != full_v_head_dim:  # 如果启用滑动窗口且V头维度不同 # 检查SWA维度差异
            self.v_head_dim = full_v_head_dim  # 保存全注意力V头维度 # 全注意力V头维度
            self.swa_v_head_dim = swa_v_head_dim  # 保存SWA V头维度 # SWA V头维度
        elif (  # 否则如果 # 检查混合线性模型
            model_runner.hybrid_gdn_config is not None  # 混合GDN配置存在 # GDN配置
            or model_runner.kimi_linear_config is not None  # Kimi线性配置存在 # Kimi线性配置
            or model_runner.linear_attn_model_spec is not None  # 线性注意力模型规格存在 # 线性注意力规格
        ):
            # For hybrid linear models, layer_id = 0 may not be full attention
            # 对于混合线性模型，layer_id = 0可能不是全注意力
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()  # 从KV池获取V头维度 # 从池获取V头维度
            self.swa_v_head_dim = None  # SWA V头维度为空 # 无SWA维度
        else:  # 否则 # 其他情况
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[  # 从KV池获取V头维度 # 从值缓冲区获取V头维度
                -1  # 最后一维 # 最后一维
            ]
            self.swa_v_head_dim = None  # SWA V头维度为空 # 无SWA维度
        self.max_context_len = model_runner.model_config.context_len  # 保存最大上下文长度 # 最大上下文长度
        self.device = model_runner.device  # 保存设备信息 # 设备
        self.device_core_count = get_device_core_count(model_runner.gpu_id)  # 获取设备核心数 # 设备核心数
        self.static_kv_splits = get_bool_env_var(  # 获取静态KV分片环境变量 # 获取静态KV分片标志
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"  # 环境变量名和默认值 # 环境变量名和默认值
        )
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits  # 保存最大KV分片数 # 最大KV分片数
        if self.use_mla:  # 如果使用MLA # 检查MLA
            self.max_kv_splits = _mla_decode_kv_splits_cap(  # 计算MLA KV分片上限 # 计算MLA KV分片上限
                self.max_kv_splits,  # 基础最大分片数 # 基础最大分片数
                self.device_core_count,  # 设备核心数 # 设备核心数
                self.max_context_len,  # 最大上下文长度 # 最大上下文长度
            )
        if _is_cuda:  # 如果是CUDA设备 # CUDA条件检查
            self.use_pdl = is_arch_support_pdl()  # 检查是否支持PDL # 检查PDL支持
        else:  # 否则 # 非CUDA
            self.use_pdl = False  # 不使用PDL # 不使用PDL

        self.allow_bidirectional_attention_in_extend = (  # 是否允许扩展阶段双向注意力 # 扩展阶段双向注意力标志
            model_runner.server_args.disable_cuda_graph  # 禁用CUDA图 # 禁用CUDA图
            and (model_runner.server_args.chunked_prefill_size == -1)  # 且分块预填充大小为-1 # 且不分块预填充
        )

        # Decide whether enable deterministic inference with batch-invariant operations
        # 决定是否启用批次不变操作的确定性推理
        self.enable_deterministic = (  # 是否启用确定性推理 # 确定性推理标志
            model_runner.server_args.enable_deterministic_inference  # 从服务器参数获取 # 从参数获取
        )

        # Configure deterministic inference settings
        # 配置确定性推理设置
        if self.enable_deterministic:  # 如果启用确定性推理 # 检查确定性标志
            # Use fixed split tile size for batch invariance
            # 使用固定分片瓦片大小以实现批次不变性
            self.split_tile_size = get_int_env_var(  # 从环境变量获取分片瓦片大小 # 获取分片瓦片大小
                "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256  # 环境变量名和默认值 # 环境变量名和默认值
            )
            # Set static_kv_splits to False to use deterministic logic instead
            # 将static_kv_splits设为False以使用确定性逻辑
            self.static_kv_splits = False  # 禁用静态KV分片 # 禁用静态KV分片
        else:  # 否则 # 非确定性模式
            self.split_tile_size = (  # 从服务器参数获取分片瓦片大小 # 获取分片瓦片大小
                model_runner.server_args.triton_attention_split_tile_size  # 服务器参数 # 服务器参数
            )

        if self.split_tile_size is not None:  # 如果分片瓦片大小存在 # 检查瓦片大小
            self.max_kv_splits = (  # 根据瓦片大小重新计算最大KV分片数 # 重新计算最大KV分片数
                self.max_context_len + self.split_tile_size - 1  # 上下文长度加瓦片大小减1 # 计算公式
            ) // self.split_tile_size  # 整除瓦片大小 # 整除

        # Check arguments
        # 检查参数
        assert not (  # 断言检查 # 断言
            model_runner.sliding_window_size is not None  # 滑动窗口大小存在 # 滑动窗口
            and model_runner.model_config.is_encoder_decoder  # 且是编码器-解码器模型 # 编码器-解码器
        ), "Sliding window and cross attention are not supported together"  # 错误提示 # 错误提示

        # Initialize buffers
        # 初始化缓冲区
        # TODO(Jianan Ji): Make sure it behaves as expected when kv_indptr_buf is provided and sliding window is enabled
        # 待办(Jianan Ji)：确保在提供kv_indptr_buf且启用滑动窗口时行为符合预期
        if kv_indptr_buf is None:  # 如果未提供KV索引指针缓冲区 # 检查缓冲区
            self.kv_indptr = torch.zeros(  # 创建零填充的KV索引指针 # 创建KV索引指针
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device  # 形状、类型、设备 # 形状、类型、设备
            )
        else:  # 否则 # 提供了缓冲区
            self.kv_indptr = kv_indptr_buf  # 使用提供的缓冲区 # 使用提供的缓冲区

        # If sliding window is enabled, we might need two sets of buffers
        # because of interleaved attention types (e.g. for Gemma3)
        # 如果启用滑动窗口，可能需要两套缓冲区，因为存在交错注意力类型（如Gemma3）
        self.window_kv_indptr = None  # 窗口KV索引指针初始化为空 # 窗口KV索引指针
        if self.sliding_window_size is not None and self.sliding_window_size > 0:  # 如果启用滑动窗口 # 检查滑动窗口
            if kv_indptr_buf is None:  # 如果未提供KV索引指针缓冲区 # 检查缓冲区
                self.window_kv_indptr = torch.zeros(  # 创建零填充的窗口KV索引指针 # 创建窗口KV索引指针
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device  # 形状、类型、设备 # 形状、类型、设备
                )
            else:  # 否则 # 提供了缓冲区
                # When provided a buffer, create a clone for the second buffer
                # 提供缓冲区时，为第二个缓冲区创建克隆
                self.window_kv_indptr = torch.zeros_like(kv_indptr_buf)  # 创建与提供缓冲区相同的零张量 # 创建同类零张量

        if not self.skip_prefill:  # 如果不跳过预填充 # 检查预填充
            self.qo_indptr = torch.zeros(  # 创建零填充的查询输出索引指针 # 创建QO索引指针
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device  # 形状、类型、设备 # 形状、类型、设备
            )

            self.mask_indptr = torch.zeros(  # 创建零填充的掩码索引指针 # 创建掩码索引指针
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device  # 形状、类型、设备 # 形状、类型、设备
            )

        # Initialize forward metadata
        # 初始化前向元数据
        self.forward_metadata: ForwardMetadata = None  # 前向元数据初始化为空 # 前向元数据

        self.cuda_graph_custom_mask = None  # CUDA图自定义掩码初始化为空 # CUDA图自定义掩码

    def get_num_kv_splits(  # 获取KV分片数 # 获取KV分片数
        self,
        num_kv_splits: torch.Tensor,  # KV分片数张量 # KV分片数输出张量
        seq_lens: torch.Tensor,  # 序列长度张量 # 序列长度
    ):
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]  # 获取token数和序列数 # token数和序列数
        # NOTE(alcanderian): Considering speculative_decodeing,
        # num_kv_splits.shape[0] will be topk * real_num_token.
        # And the real_num_token is num_seq in decoding phase.
        # 注意(alcanderian)：考虑投机解码，num_kv_splits.shape[0]将是topk * 实际token数。
        # 实际token数在解码阶段就是num_seq。
        num_group = num_token // num_seq  # 计算每组token数 # 每组token数

        assert (  # 断言检查 # 断言
            num_group * num_seq == num_token  # 组数乘以序列数应等于token数 # 验证一致性
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"  # 错误提示 # 错误提示

        # Legacy dynamic splitting logic (non-deterministic)
        # 传统动态分片逻辑（非确定性）
        if (  # 如果 # 检查条件
            self.static_kv_splits or self.device_core_count <= 0  # 静态KV分片或设备核心数<=0 # 静态分片或无核心
        ) and not self.enable_deterministic:  # 且未启用确定性推理 # 且非确定性
            num_kv_splits.fill_(self.max_kv_splits)  # 填充为最大KV分片数 # 填充最大值
            return  # 返回 # 返回

        # deterministic
        # 确定性模式
        if self.split_tile_size is not None and self.enable_deterministic:  # 如果有分片瓦片大小且启用确定性 # 确定性条件
            # expand seq_lens to match num_token
            # 扩展seq_lens以匹配num_token
            if num_group > 1:  # 如果组数大于1 # 多组情况
                expanded_seq_lens = seq_lens.repeat_interleave(num_group)  # 重复序列长度以匹配token数 # 重复序列长度
            else:  # 否则 # 单组情况
                expanded_seq_lens = seq_lens  # 直接使用序列长度 # 直接使用

            num_kv_splits[:] = (  # 计算每个token的KV分片数 # 计算KV分片数
                expanded_seq_lens + self.split_tile_size - 1  # 序列长度加瓦片大小减1 # 计算公式
            ) // self.split_tile_size  # 整除瓦片大小 # 整除
            return  # 返回 # 返回

        if num_seq < 256:  # 如果序列数小于256 # 小批次
            SCHEDULE_SEQ = 256  # 调度序列数设为256 # 调度序列数
        else:  # 否则 # 大批次
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)  # 调度序列数为序列数的下一个2的幂 # 2的幂

        get_num_kv_splits_triton[(1,)](  # 调用Triton内核计算KV分片数 # 调用Triton内核
            num_kv_splits,  # KV分片数输出 # KV分片数
            seq_lens,  # 序列长度 # 序列长度
            num_seq,  # 序列数 # 序列数
            num_group,  # 每组token数 # 组大小
            self.num_head,  # 注意力头数 # 头数
            self.num_kv_head,  # KV头数 # KV头数
            self.max_kv_splits,  # 最大KV分片数 # 最大分片数
            self.device_core_count,  # 设备核心数 # 核心数
            MAX_NUM_SEQ=SCHEDULE_SEQ,  # 最大序列数编译时常量 # 编译时常量
        )

    def _fill_kv_indptr_and_indices(  # 填充KV索引指针和索引（内部方法） # 填充KV索引指针和索引
        self,
        bs: int,  # 批次大小 # 批次大小
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求池索引
        kv_indices: torch.Tensor,  # KV索引输出 # KV索引输出
    ) -> torch.Tensor:  # 返回KV索引指针 # 返回KV索引指针
        kv_indptr = self.kv_indptr[: bs + 1]  # 截取当前批次大小的KV索引指针 # 截取KV索引指针
        kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)  # 计算累积和作为索引指针 # 计算累积和
        create_flashinfer_kv_indices_triton[(bs,)](  # 调用Triton内核创建KV索引 # 调用Triton内核
            self.req_to_token,  # 请求到token映射表 # 请求-token映射
            req_pool_indices,  # 请求池索引 # 请求池索引
            seq_lens,  # 序列长度 # 序列长度
            kv_indptr,  # KV索引指针 # KV索引指针
            None,  # 起始索引（None表示从0开始） # 起始索引
            kv_indices,  # KV索引输出 # KV索引输出
            self.req_to_token.stride(0),  # 映射表步幅 # 步幅
        )
        return kv_indptr  # 返回KV索引指针 # 返回

    def _update_decode_kv_buffers(  # 更新解码KV缓冲区（内部方法） # 更新解码KV缓冲区
        self,
        bs: int,  # 批次大小 # 批次大小
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求池索引
    ):
        """Fill KV (and SWA) cuda-graph buffers for decode/idle mode.
        # 填充解码/空闲模式的KV（和SWA）CUDA图缓冲区。

        Returns ``(kv_indptr, window_kv_indptr, window_kv_lens)`` where
        ``window_kv_lens`` is ``None`` when sliding-window is disabled.
        返回``(kv_indptr, window_kv_indptr, window_kv_lens)``，
        当滑动窗口禁用时``window_kv_lens``为``None``。
        """
        seq_lens = seq_lens[:bs]  # 截取当前批次大小的序列长度 # 截取序列长度
        req_pool_indices = req_pool_indices[:bs]  # 截取当前批次大小的请求池索引 # 截取请求池索引
        kv_indptr = self._fill_kv_indptr_and_indices(  # 填充KV索引指针和索引 # 填充KV索引
            bs, seq_lens, req_pool_indices, self.cuda_graph_kv_indices  # 传入参数 # 传入参数
        )
        window_kv_indptr = self.window_kv_indptr  # 获取窗口KV索引指针 # 窗口KV索引指针
        window_kv_lens = None  # 窗口KV长度初始化为空 # 窗口KV长度
        if self.sliding_window_size is not None and self.sliding_window_size > 0:  # 如果启用滑动窗口 # 检查滑动窗口
            window_kv_indptr, _, window_kv_lens, _ = update_sliding_window_buffer(  # 更新滑动窗口缓冲区 # 更新滑动窗口缓冲区
                self.window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
                self.req_to_token,  # 请求到token映射表 # 请求-token映射
                self.sliding_window_size,  # 滑动窗口大小 # 滑动窗口大小
                seq_lens,  # 序列长度 # 序列长度
                req_pool_indices,  # 请求池索引 # 请求池索引
                bs,  # 批次大小 # 批次大小
                token_to_kv_pool=self.token_to_kv_pool,  # token到KV池 # KV池
                window_kv_indices=self.cuda_graph_window_kv_indices,  # 窗口KV索引 # 窗口KV索引
            )
        return kv_indptr, window_kv_indptr, window_kv_lens  # 返回KV索引指针、窗口KV索引指针、窗口KV长度 # 返回

    def _update_target_verify_buffers(  # 更新目标验证缓冲区（内部方法） # 更新目标验证缓冲区
        self,
        bs: int,  # 批次大小 # 批次大小
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求池索引
        spec_info,  # 投机解码信息 # 投机解码信息
    ):
        """Fill all cuda-graph buffers for target_verify mode.
        # 填充target_verify模式的所有CUDA图缓冲区。

        Returns the ForwardMetadata components:
        ``(qo_indptr, kv_indptr, custom_mask, mask_indptr,
          window_kv_indptr, window_kv_indices, window_num_kv_splits, window_kv_offsets)``
        返回ForwardMetadata组件：
        ``(qo_indptr, kv_indptr, custom_mask, mask_indptr,
          window_kv_indptr, window_kv_indices, window_num_kv_splits, window_kv_offsets)``
        """
        qo_indptr = self.qo_indptr[: bs + 1]  # 截取当前批次大小的QO索引指针 # 截取QO索引指针
        qo_indptr[: bs + 1] = torch.arange(  # 填充为等差序列 # 填充等差序列
            0,  # 起始值 # 起始值
            (1 + bs) * self.num_draft_tokens,  # 结束值 # 结束值
            step=self.num_draft_tokens,  # 步长为草稿token数 # 步长
            dtype=torch.int32,  # 数据类型 # 数据类型
            device=self.device,  # 设备 # 设备
        )
        kv_indptr = self._fill_kv_indptr_and_indices(  # 填充KV索引指针和索引 # 填充KV索引
            bs, seq_lens, req_pool_indices, self.cuda_graph_kv_indices  # 传入参数 # 传入参数
        )
        window_kv_indptr = self.window_kv_indptr  # 获取窗口KV索引指针 # 窗口KV索引指针
        window_kv_indices = None  # 窗口KV索引初始化为空 # 窗口KV索引
        window_num_kv_splits = None  # 窗口KV分片数初始化为空 # 窗口KV分片数
        window_kv_offsets = None  # 窗口KV偏移量初始化为空 # 窗口KV偏移量
        if self.sliding_window_size is not None and self.sliding_window_size > 0:  # 如果启用滑动窗口 # 检查滑动窗口
            window_kv_indices = self.cuda_graph_window_kv_indices  # 使用CUDA图窗口KV索引 # 使用CUDA图窗口KV索引
            window_num_kv_splits = self.cuda_graph_window_num_kv_splits  # 使用CUDA图窗口KV分片数 # 使用CUDA图窗口KV分片数
            window_kv_offsets = self.cuda_graph_window_kv_offsets  # 使用CUDA图窗口KV偏移量 # 使用CUDA图窗口KV偏移量
            window_kv_indptr, window_kv_indices, _, window_kv_offsets[:bs] = (  # 更新滑动窗口缓冲区 # 更新滑动窗口缓冲区
                update_sliding_window_buffer(  # 调用滑动窗口缓冲区更新函数 # 调用更新函数
                    self.window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
                    self.req_to_token,  # 请求到token映射表 # 请求-token映射
                    self.sliding_window_size,  # 滑动窗口大小 # 滑动窗口大小
                    seq_lens[:bs],  # 序列长度（截取） # 序列长度
                    req_pool_indices,  # 请求池索引 # 请求池索引
                    bs,  # 批次大小 # 批次大小
                    token_to_kv_pool=self.token_to_kv_pool,  # token到KV池 # KV池
                    window_kv_indices=window_kv_indices,  # 窗口KV索引 # 窗口KV索引
                )
            )
        custom_mask = self.cuda_graph_custom_mask  # 获取CUDA图自定义掩码 # CUDA图自定义掩码
        if (  # 如果 # 检查自定义掩码
            spec_info is not None  # 投机解码信息存在 # 投机信息存在
            and getattr(spec_info, "custom_mask", None) is not None  # 且有自定义掩码 # 有自定义掩码
        ):
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask  # 填充自定义掩码 # 填充自定义掩码
        else:  # 否则 # 无自定义掩码
            custom_mask = None  # 掩码为空 # 掩码为空
        seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)  # 计算序列掩码长度 # 序列掩码长度
        mask_indptr = self.mask_indptr[: bs + 1]  # 截取当前批次大小的掩码索引指针 # 截取掩码索引指针
        mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)  # 计算累积和作为掩码索引指针 # 计算累积和
        return (  # 返回ForwardMetadata组件 # 返回
            qo_indptr,  # QO索引指针 # QO索引指针
            kv_indptr,  # KV索引指针 # KV索引指针
            custom_mask,  # 自定义掩码 # 自定义掩码
            mask_indptr,  # 掩码索引指针 # 掩码索引指针
            window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
            window_kv_indices,  # 窗口KV索引 # 窗口KV索引
            window_num_kv_splits,  # 窗口KV分片数 # 窗口KV分片数
            window_kv_offsets,  # 窗口KV偏移量 # 窗口KV偏移量
        )

    def _update_draft_extend_buffers(  # 更新草稿扩展缓冲区（内部方法） # 更新草稿扩展缓冲区
        self,
        bs: int,  # 批次大小 # 批次大小
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求池索引
        forward_mode: ForwardMode,  # 前向模式 # 前向模式
        spec_info: Optional[SpecInput],  # 投机解码信息 # 投机解码信息
    ):
        """Fill QO + KV cuda-graph buffers for draft_extend mode.
        # 填充draft_extend模式的QO + KV CUDA图缓冲区。

        Returns ``(qo_indptr, kv_indptr, num_tokens_per_bs)``.
        返回``(qo_indptr, kv_indptr, num_tokens_per_bs)``。
        """
        seq_lens = seq_lens[:bs]  # 截取当前批次大小的序列长度 # 截取序列长度
        num_tokens_per_bs = self.speculative_num_steps + 1  # 每个批次的token数 # 每批次token数
        qo_indptr = self.qo_indptr[: bs + 1]  # 截取当前批次大小的QO索引指针 # 截取QO索引指针
        qo_indptr[: bs + 1] = torch.arange(  # 填充为等差序列 # 填充等差序列
            0,  # 起始值 # 起始值
            bs * num_tokens_per_bs + 1,  # 结束值 # 结束值
            step=num_tokens_per_bs,  # 步长 # 步长
            dtype=torch.int32,  # 数据类型 # 数据类型
            device=self.device,  # 设备 # 设备
        )
        if forward_mode.is_draft_extend_v2():  # 如果是DRAFT_EXTEND_V2模式 # 检查V2模式
            # DRAFT_EXTEND_V2: seq_lens = prefix + extend (bumped by eagle_info_v2).
            # Triton extend kernel receives extend K/V as separate tensors, so
            # kv_indptr/kv_indices must cover only the prefix portion.
            # extend_seq_lens_tensor is only attached to spec_info at real
            # replay (eagle_draft_extend_cuda_graph_runner.replay); during the
            # capture-time warmup it's absent, so fall back to zeros (matches
            # the pre-unification capture path in #26651). Clamp at 0 because
            # padded rows (raw_bs..bs) leave seq_lens at the fill value (1)
            # while extend_seq_lens stays at num_tokens_per_bs, which would
            # otherwise produce negative kv_lens; padded rows reference
            # reserved req-pool slot 0 and their output is discarded.
            # DRAFT_EXTEND_V2：seq_lens = prefix + extend（由eagle_info_v2增加）。
            # Triton extend内核接收扩展K/V作为独立张量，因此kv_indptr/kv_indices
            # 必须仅覆盖前缀部分。extend_seq_lens_tensor仅在真实重放时附加到
            # spec_info（eagle_draft_extend_cuda_graph_runner.replay）；在捕获时
            # 预热期间它不存在，因此回退到零（与#26651中的预统一捕获路径匹配）。
            # 钳位为0，因为填充行（raw_bs..bs）的seq_lens保持填充值（1），
            # 而extend_seq_lens保持num_tokens_per_bs，否则会产生负的kv_lens；
            # 填充行引用保留的请求池槽0，其输出被丢弃。
            if (  # 如果 # 检查扩展序列长度
                spec_info is not None  # 投机信息存在 # 投机信息存在
                and getattr(spec_info, "extend_seq_lens_tensor", None) is not None  # 且扩展序列长度存在 # 扩展序列长度存在
            ):
                extend_seq_lens = spec_info.extend_seq_lens_tensor[:bs].to(torch.int32)  # 获取扩展序列长度 # 获取扩展序列长度
            else:  # 否则 # 无扩展序列长度
                extend_seq_lens = torch.zeros(  # 创建零填充的扩展序列长度 # 创建零填充
                    bs, dtype=torch.int32, device=seq_lens.device  # 形状、类型、设备 # 形状、类型、设备
                )
            kv_lens = torch.clamp(seq_lens - extend_seq_lens, min=0).to(torch.int32)  # 计算KV长度，钳位最小为0 # 计算KV长度
        else:  # 否则 # V1模式
            # DRAFT_EXTEND_V1: seq_lens = prefix only.
            # DRAFT_EXTEND_V1：seq_lens = 仅前缀。
            kv_lens = seq_lens  # KV长度等于序列长度 # KV长度
        kv_indptr = self._fill_kv_indptr_and_indices(  # 填充KV索引指针和索引 # 填充KV索引
            bs, kv_lens, req_pool_indices, self.cuda_graph_kv_indices  # 传入参数 # 传入参数
        )
        return qo_indptr, kv_indptr, num_tokens_per_bs  # 返回QO索引指针、KV索引指针、每批次token数 # 返回

    def init_forward_metadata(self, forward_batch: ForwardBatch):  # 初始化前向元数据 # 初始化前向元数据
        """Init auxiliary variables for triton attention backend."""
        # 初始化Triton注意力后端的辅助变量。

        bs = forward_batch.batch_size  # 获取批次大小 # 批次大小
        window_kv_indptr = self.window_kv_indptr  # 获取窗口KV索引指针 # 窗口KV索引指针
        window_kv_indices = None  # 窗口KV索引初始化为空 # 窗口KV索引
        window_num_kv_splits = None  # 窗口KV分片数初始化为空 # 窗口KV分片数
        window_kv_offsets = None  # 窗口KV偏移量初始化为空 # 窗口KV偏移量
        swa_attn_logits = None  # SWA注意力logits初始化为空 # SWA注意力logits
        spec_info = forward_batch.spec_info  # 获取投机解码信息 # 投机解码信息

        if forward_batch.forward_mode.is_decode_or_idle():  # 如果是解码或空闲模式 # 检查解码模式
            if spec_info is None:  # 如果没有投机信息 # 检查投机信息
                kv_indices = torch.empty(  # 创建空的KV索引张量 # 创建KV索引
                    forward_batch.seq_lens_sum, dtype=torch.int64, device=self.device  # 形状、类型、设备 # 形状、类型、设备
                )
                kv_indptr = self._fill_kv_indptr_and_indices(  # 填充KV索引指针和索引 # 填充KV索引
                    bs,  # 批次大小 # 批次大小
                    forward_batch.seq_lens,  # 序列长度 # 序列长度
                    forward_batch.req_pool_indices,  # 请求池索引 # 请求池索引
                    kv_indices,  # KV索引 # KV索引
                )
                # Sliding window
                # 滑动窗口
                if (  # 如果 # 检查滑动窗口
                    self.sliding_window_size is not None  # 滑动窗口大小存在 # 滑动窗口
                    and self.sliding_window_size > 0  # 且大于0 # 且有效
                ):
                    window_kv_indptr, window_kv_indices, window_kv_lens, _ = (  # 更新滑动窗口缓冲区 # 更新滑动窗口缓冲区
                        update_sliding_window_buffer(  # 调用滑动窗口缓冲区更新函数 # 调用更新函数
                            self.window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
                            self.req_to_token,  # 请求到token映射表 # 请求-token映射
                            self.sliding_window_size,  # 滑动窗口大小 # 滑动窗口大小
                            forward_batch.seq_lens,  # 序列长度 # 序列长度
                            forward_batch.req_pool_indices,  # 请求池索引 # 请求池索引
                            bs,  # 批次大小 # 批次大小
                            self.device,  # 设备 # 设备
                            self.token_to_kv_pool,  # token到KV池 # KV池
                        )
                    )
                    window_num_kv_splits = torch.empty(  # 创建空的窗口KV分片数 # 创建窗口KV分片数
                        (bs,), dtype=torch.int32, device=self.device  # 形状、类型、设备 # 形状、类型、设备
                    )
                    self.get_num_kv_splits(window_num_kv_splits, window_kv_lens)  # 计算窗口KV分片数 # 计算窗口KV分片数
            else:  # 否则（有投机信息） # 有投机信息
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices  # 从投机信息获取KV索引 # 从投机信息获取
                bs = kv_indptr.shape[0] - 1  # 从索引指针推导批次大小 # 推导批次大小

            attn_logits = torch.empty(  # 创建空的注意力logits张量 # 创建注意力logits
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),  # 形状 # 形状
                dtype=torch.float32,  # 数据类型 # 数据类型
                device=self.device,  # 设备 # 设备
            )
            if self.swa_v_head_dim is not None:  # 如果SWA V头维度存在 # 检查SWA V头维度
                swa_attn_logits = torch.empty(  # 创建空的SWA注意力logits张量 # 创建SWA注意力logits
                    (bs, self.num_head, self.max_kv_splits, self.swa_v_head_dim),  # 形状 # 形状
                    dtype=torch.float32,  # 数据类型 # 数据类型
                    device=self.device,  # 设备 # 设备
                )
            else:  # 否则 # 无SWA V头维度
                swa_attn_logits = None  # SWA注意力logits为空 # SWA注意力logits为空
            attn_lse = torch.empty(  # 创建空的注意力LSE张量 # 创建注意力LSE
                (bs, self.num_head, self.max_kv_splits),  # 形状 # 形状
                dtype=torch.float32,  # 数据类型 # 数据类型
                device=self.device,  # 设备 # 设备
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)  # 创建空的KV分片数 # 创建KV分片数
            self.get_num_kv_splits(num_kv_splits, forward_batch.seq_lens)  # 计算KV分片数 # 计算KV分片数

            qo_indptr = None  # QO索引指针为空 # QO索引指针
            custom_mask = None  # 自定义掩码为空 # 自定义掩码
            mask_indptr = None  # 掩码索引指针为空 # 掩码索引指针
            max_extend_len = None  # 最大扩展长度为空 # 最大扩展长度
        elif forward_batch.forward_mode.is_target_verify():  # 如果是目标验证模式 # 检查目标验证模式
            bs = len(forward_batch.req_pool_indices)  # 从请求池索引获取批次大小 # 获取批次大小
            qo_indptr = torch.arange(  # 创建等差序列作为QO索引指针 # 创建QO索引指针
                0,  # 起始值 # 起始值
                (1 + bs) * self.num_draft_tokens,  # 结束值 # 结束值
                step=self.num_draft_tokens,  # 步长 # 步长
                dtype=torch.int32,  # 数据类型 # 数据类型
                device=self.device,  # 设备 # 设备
            )
            # Different with flashinfer kv_indptr and kv_indices construction.
            # gpu_only: seq_lens_sum may be None; ub-allocate is safe (ragged write).
            # 与FlashInfer的kv_indptr和kv_indices构建不同。
            # gpu_only：seq_lens_sum可能为None；上界分配是安全的（不规则写入）。
            seq_lens_sum = forward_batch.seq_lens_sum  # 获取序列长度总和 # 序列长度总和
            if seq_lens_sum is None:  # 如果序列长度总和为空 # 检查序列长度总和
                seq_lens_sum = bs * self.max_context_len  # 使用上界估计 # 上界估计
            kv_indices = torch.empty(  # 创建空的KV索引张量 # 创建KV索引
                seq_lens_sum, dtype=torch.int64, device=self.device  # 形状、类型、设备 # 形状、类型、设备
            )
            kv_indptr = self._fill_kv_indptr_and_indices(  # 填充KV索引指针和索引 # 填充KV索引
                bs,  # 批次大小 # 批次大小
                forward_batch.seq_lens,  # 序列长度 # 序列长度
                forward_batch.req_pool_indices,  # 请求池索引 # 请求池索引
                kv_indices,  # KV索引 # KV索引
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:  # 如果启用滑动窗口 # 检查滑动窗口
                # window_kv_offsets is used to calculate the start position in custom mask
                # window_kv_offsets用于计算自定义掩码中的起始位置
                (  # 更新滑动窗口缓冲区 # 更新滑动窗口缓冲区
                    window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
                    window_kv_indices,  # 窗口KV索引 # 窗口KV索引
                    window_kv_lens,  # 窗口KV长度 # 窗口KV长度
                    window_kv_offsets,  # 窗口KV偏移量 # 窗口KV偏移量
                ) = update_sliding_window_buffer(  # 调用滑动窗口缓冲区更新函数 # 调用更新函数
                    self.window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
                    self.req_to_token,  # 请求到token映射表 # 请求-token映射
                    self.sliding_window_size,  # 滑动窗口大小 # 滑动窗口大小
                    forward_batch.seq_lens,  # 序列长度 # 序列长度
                    forward_batch.req_pool_indices,  # 请求池索引 # 请求池索引
                    bs,  # 批次大小 # 批次大小
                    self.device,  # 设备 # 设备
                    self.token_to_kv_pool,  # token到KV池 # KV池
                )

            custom_mask = spec_info.custom_mask  # 获取自定义掩码 # 自定义掩码
            seq_mask_len = self.num_draft_tokens * (  # 计算序列掩码长度 # 序列掩码长度
                forward_batch.seq_lens + self.num_draft_tokens  # 序列长度加草稿token数 # 计算公式
            )
            mask_indptr = self.mask_indptr  # 获取掩码索引指针 # 掩码索引指针
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)  # 计算累积和 # 计算累积和
            mask_indptr = mask_indptr[: bs + 1]  # 截取当前批次大小 # 截取
            max_extend_len = self.num_draft_tokens  # 最大扩展长度为草稿token数 # 最大扩展长度
            num_kv_splits = None  # KV分片数为空 # KV分片数
            attn_logits = None  # 注意力logits为空 # 注意力logits
            attn_lse = None  # 注意力LSE为空 # 注意力LSE

        elif forward_batch.forward_mode.is_draft_extend():  # 如果是草稿扩展模式 # 检查草稿扩展模式
            # Eager only (CG replay bypasses init); explicit D2H here instead of
            # letting torch.empty inside generate_attn_arg_prefill .item() on a
            # GPU cumsum tensor.
            # 仅即时模式（CG重放跳过初始化）；此处显式D2H，而不是让
            # generate_attn_arg_prefill内部的torch.empty对GPU cumsum张量执行.item()。
            seq_lens_sum = (  # 计算序列长度总和 # 计算序列长度总和
                forward_batch.seq_lens_sum  # 如果序列长度总和存在 # 使用已有值
                if forward_batch.seq_lens_sum is not None  # 检查是否存在 # 检查
                else int(forward_batch.seq_lens.sum())  # 否则计算总和 # 计算总和
            )
            kv_indices, kv_indptr, qo_indptr, custom_mask = (  # 从投机信息生成注意力参数 # 生成注意力参数
                spec_info.generate_attn_arg_prefill(  # 调用注意力参数生成函数 # 调用生成函数
                    forward_batch.req_pool_indices,  # 请求池索引 # 请求池索引
                    forward_batch.seq_lens,  # 序列长度 # 序列长度
                    seq_lens_sum,  # 序列长度总和 # 序列长度总和
                    self.req_to_token,  # 请求到token映射表 # 请求-token映射
                )
            )
            kv_indices = kv_indices.to(torch.int64)  # 转换KV索引为int64 # 转换类型
            mask_indptr = None  # 掩码索引指针为空 # 掩码索引指针
            # TODO(FIXME): This will trigger an invalid Eagle tree when using
            # `max(spec_info.num_accept_tokens_cpu)`.
            # It might have been forgotten to update somewhere.
            # 待办(修复)：使用`max(spec_info.num_accept_tokens_cpu)`时会触发
            # 无效的Eagle树。可能某处忘记更新了。
            max_extend_len = torch.max(spec_info.num_accept_tokens).item()  # 获取最大接受token数 # 最大扩展长度
            num_kv_splits = None  # KV分片数为空 # KV分片数
            attn_logits = None  # 注意力logits为空 # 注意力logits
            attn_lse = None  # 注意力LSE为空 # 注意力LSE
        else:  # 否则（普通扩展模式） # 普通扩展模式
            # gpu_only leaves _cpu unset; ub-allocate is safe (ragged write
            # from GPU tensor, extra tail unused).
            # gpu_only使_cpu未设置；上界分配是安全的（从GPU张量不规则写入，多余尾部未使用）。
            if forward_batch.extend_prefix_lens_cpu is not None:  # 如果CPU扩展前缀长度存在 # 检查CPU前缀长度
                kv_indices_len = sum(forward_batch.extend_prefix_lens_cpu)  # 计算KV索引长度 # 计算KV索引长度
            else:  # 否则 # 无CPU前缀长度
                kv_indices_len = bs * self.max_context_len  # 使用上界估计 # 上界估计
            kv_indices = torch.empty(  # 创建空的KV索引张量 # 创建KV索引
                kv_indices_len,  # 长度 # 长度
                dtype=torch.int64,  # 数据类型 # 数据类型
                device=self.device,  # 设备 # 设备
            )
            kv_indptr = self._fill_kv_indptr_and_indices(  # 填充KV索引指针和索引 # 填充KV索引
                bs,  # 批次大小 # 批次大小
                forward_batch.extend_prefix_lens,  # 扩展前缀长度 # 扩展前缀长度
                forward_batch.req_pool_indices,  # 请求池索引 # 请求池索引
                kv_indices,  # KV索引 # KV索引
            )
            # Sliding window
            # 滑动窗口
            if self.sliding_window_size is not None and self.sliding_window_size > 0:  # 如果启用滑动窗口 # 检查滑动窗口
                (  # 更新滑动窗口缓冲区 # 更新滑动窗口缓冲区
                    window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
                    window_kv_indices,  # 窗口KV索引 # 窗口KV索引
                    window_kv_lens,  # 窗口KV长度 # 窗口KV长度
                    window_kv_offsets,  # 窗口KV偏移量 # 窗口KV偏移量
                ) = update_sliding_window_buffer(  # 调用滑动窗口缓冲区更新函数 # 调用更新函数
                    self.window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
                    self.req_to_token,  # 请求到token映射表 # 请求-token映射
                    self.sliding_window_size,  # 滑动窗口大小 # 滑动窗口大小
                    forward_batch.extend_prefix_lens,  # 扩展前缀长度 # 扩展前缀长度
                    forward_batch.req_pool_indices,  # 请求池索引 # 请求池索引
                    bs,  # 批次大小 # 批次大小
                    self.device,  # 设备 # 设备
                    self.token_to_kv_pool,  # token到KV池 # KV池
                )

            qo_indptr = self.qo_indptr  # 获取QO索引指针 # QO索引指针
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)  # 计算累积和 # 计算累积和
            qo_indptr = qo_indptr[: bs + 1]  # 截取当前批次大小 # 截取
            custom_mask = None  # 自定义掩码为空 # 自定义掩码
            mask_indptr = None  # 掩码索引指针为空 # 掩码索引指针
            attn_logits = None  # 注意力logits为空 # 注意力logits
            attn_lse = None  # 注意力LSE为空 # 注意力LSE
            # Caller usually supplies extend_seq_lens_cpu (eagle_info gpu_only
            # sets host-constant mirror); defensive GPU-max fallback if not.
            # 调用者通常提供extend_seq_lens_cpu（eagle_info gpu_only设置主机常量镜像）；
            # 如果没有则防御性地使用GPU最大值回退。
            if forward_batch.extend_seq_lens_cpu is not None:  # 如果CPU扩展序列长度存在 # 检查CPU序列长度
                max_extend_len = max(forward_batch.extend_seq_lens_cpu)  # 取最大值 # 取最大值
            else:  # 否则 # 无CPU序列长度
                max_extend_len = int(forward_batch.extend_seq_lens.max())  # 从GPU获取最大值 # GPU最大值
            num_kv_splits = None  # KV分片数为空 # KV分片数

        self.forward_metadata = ForwardMetadata(  # 创建ForwardMetadata实例 # 创建前向元数据
            attn_logits,  # 注意力logits # 注意力logits
            attn_lse,  # 注意力LSE # 注意力LSE
            max_extend_len,  # 最大扩展长度 # 最大扩展长度
            num_kv_splits,  # KV分片数 # KV分片数
            kv_indptr,  # KV索引指针 # KV索引指针
            kv_indices,  # KV索引 # KV索引
            qo_indptr,  # QO索引指针 # QO索引指针
            custom_mask,  # 自定义掩码 # 自定义掩码
            mask_indptr,  # 掩码索引指针 # 掩码索引指针
            window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
            window_kv_indices,  # 窗口KV索引 # 窗口KV索引
            window_num_kv_splits,  # 窗口KV分片数 # 窗口KV分片数
            window_kv_offsets,  # 窗口KV偏移量 # 窗口KV偏移量
            swa_attn_logits=swa_attn_logits,  # SWA注意力logits # SWA注意力logits
        )

    def init_cuda_graph_state(  # 初始化CUDA图状态 # 初始化CUDA图状态
        self,
        max_bs: int,  # 最大批次大小 # 最大批次大小
        max_num_tokens: int,  # 最大token数 # 最大token数
        kv_indices_buf: Optional[torch.Tensor] = None,  # KV索引缓冲区 # KV索引缓冲区（可选）
        cuda_graph_num_kv_splits_buf: Optional[torch.Tensor] = None,  # CUDA图KV分片数缓冲区 # CUDA图KV分片数缓冲区（可选）
    ):
        self.cuda_graph_attn_logits = torch.zeros(  # 创建零填充的CUDA图注意力logits # 创建CUDA图注意力logits
            (max_num_tokens, self.num_head, self.max_kv_splits, self.v_head_dim),  # 形状 # 形状
            dtype=torch.float32,  # 数据类型 # 数据类型
            device=self.device,  # 设备 # 设备
        )
        if self.swa_v_head_dim is not None:  # 如果SWA V头维度存在 # 检查SWA V头维度
            self.cuda_graph_swa_attn_logits = torch.zeros(  # 创建零填充的CUDA图SWA注意力logits # 创建CUDA图SWA注意力logits
                (  # 形状参数 # 形状
                    max_num_tokens,  # 最大token数 # token维度
                    self.num_head,  # 注意力头数 # 头维度
                    self.max_kv_splits,  # 最大KV分片数 # 分片维度
                    self.swa_v_head_dim,  # SWA V头维度 # V头维度
                ),
                dtype=torch.float32,  # 数据类型 # 数据类型
                device=self.device,  # 设备 # 设备
            )
        else:  # 否则 # 无SWA V头维度
            self.cuda_graph_swa_attn_logits = None  # CUDA图SWA注意力logits为空 # SWA logits为空
        self.cuda_graph_attn_lse = torch.zeros(  # 创建零填充的CUDA图注意力LSE # 创建CUDA图注意力LSE
            (max_num_tokens, self.num_head, self.max_kv_splits),  # 形状 # 形状
            dtype=torch.float32,  # 数据类型 # 数据类型
            device=self.device,  # 设备 # 设备
        )

        if cuda_graph_num_kv_splits_buf is None:  # 如果未提供CUDA图KV分片数缓冲区 # 检查缓冲区
            self.cuda_graph_num_kv_splits = torch.full(  # 创建填充为最大KV分片数的张量 # 创建KV分片数
                (max_num_tokens,),  # 形状 # 形状
                self.max_kv_splits,  # 填充值 # 填充值
                dtype=torch.int32,  # 数据类型 # 数据类型
                device=self.device,  # 设备 # 设备
            )
        else:  # 否则 # 提供了缓冲区
            self.cuda_graph_num_kv_splits = cuda_graph_num_kv_splits_buf  # 使用提供的缓冲区 # 使用提供的缓冲区

        if kv_indices_buf is None:  # 如果未提供KV索引缓冲区 # 检查缓冲区
            self.cuda_graph_kv_indices = torch.zeros(  # 创建零填充的CUDA图KV索引 # 创建CUDA图KV索引
                (max_num_tokens * self.max_context_len),  # 形状 # 形状
                dtype=torch.int64,  # 数据类型 # 数据类型
                device=self.device,  # 设备 # 设备
            )
        else:  # 否则 # 提供了缓冲区
            self.cuda_graph_kv_indices = kv_indices_buf  # 使用提供的缓冲区 # 使用提供的缓冲区

        if not self.skip_prefill:  # 如果不跳过预填充 # 检查预填充
            self.cuda_graph_custom_mask = torch.zeros(  # 创建零填充的CUDA图自定义掩码 # 创建CUDA图自定义掩码
                (max_num_tokens * self.max_context_len),  # 形状 # 形状
                dtype=torch.uint8,  # 数据类型 # 数据类型
                device=self.device,  # 设备 # 设备
            )

        if self.sliding_window_size is not None and self.sliding_window_size > 0:  # 如果启用滑动窗口 # 检查滑动窗口
            if kv_indices_buf is None:  # 如果未提供KV索引缓冲区 # 检查缓冲区
                self.cuda_graph_window_kv_indices = torch.zeros(  # 创建零填充的CUDA图窗口KV索引 # 创建窗口KV索引
                    (max_num_tokens * self.sliding_window_size),  # 形状 # 形状
                    dtype=torch.int64,  # 数据类型 # 数据类型
                    device=self.device,  # 设备 # 设备
                )
            else:  # 否则 # 提供了缓冲区
                self.cuda_graph_window_kv_indices = torch.zeros_like(kv_indices_buf)  # 创建与提供缓冲区相同的零张量 # 创建同类零张量

            self.cuda_graph_window_num_kv_splits = torch.full(  # 创建填充为最大KV分片数的窗口KV分片数 # 创建窗口KV分片数
                (max_num_tokens,),  # 形状 # 形状
                self.max_kv_splits,  # 填充值 # 填充值
                dtype=torch.int32,  # 数据类型 # 数据类型
                device=self.device,  # 设备 # 设备
            )

            self.cuda_graph_window_kv_offsets = torch.zeros(  # 创建零填充的窗口KV偏移量 # 创建窗口KV偏移量
                (max_bs,),  # 形状 # 形状
                dtype=torch.int32,  # 数据类型 # 数据类型
                device=self.device,  # 设备 # 设备
            )

    def _build_cuda_graph_forward_metadata(  # 构建CUDA图前向元数据（内部方法） # 构建CUDA图前向元数据
        self,
        bs: int,  # 批次大小 # 批次大小
        forward_mode: ForwardMode,  # 前向模式 # 前向模式
        spec_info: Optional[SpecInput],  # 投机解码信息 # 投机解码信息
    ) -> ForwardMetadata:  # 返回ForwardMetadata # 返回前向元数据
        """Construct ForwardMetadata from the current cuda-graph buffer state.
        # 从当前CUDA图缓冲区状态构建ForwardMetadata。

        Called by capture after the buffer-update helpers have already run
        (either via replay or directly).  All fields reference the same
        ``self.cuda_graph_*`` tensors that the captured graph kernels will
        read — the Python object is rebuilt each capture, but the underlying
        GPU memory addresses are stable.
        在缓冲区更新辅助函数运行后（通过重放或直接调用），由捕获调用。
        所有字段引用相同的``self.cuda_graph_*``张量，捕获的图内核将读取
        这些张量——Python对象在每次捕获时重建，但底层GPU内存地址是稳定的。
        """
        swa = self.sliding_window_size is not None and self.sliding_window_size > 0  # 是否启用滑动窗口 # 滑动窗口标志
        if forward_mode.is_decode_or_idle():  # 如果是解码或空闲模式 # 检查解码模式
            return ForwardMetadata(  # 返回解码模式的ForwardMetadata # 返回解码模式元数据
                attn_logits=self.cuda_graph_attn_logits,  # 注意力logits # 注意力logits
                attn_lse=self.cuda_graph_attn_lse,  # 注意力LSE # 注意力LSE
                max_extend_len=None,  # 最大扩展长度为空 # 最大扩展长度
                num_kv_splits=self.cuda_graph_num_kv_splits,  # KV分片数 # KV分片数
                kv_indptr=self.kv_indptr[: bs + 1],  # KV索引指针 # KV索引指针
                kv_indices=self.cuda_graph_kv_indices,  # KV索引 # KV索引
                qo_indptr=None,  # QO索引指针为空 # QO索引指针
                custom_mask=None,  # 自定义掩码为空 # 自定义掩码
                mask_indptr=None,  # 掩码索引指针为空 # 掩码索引指针
                window_kv_indptr=self.window_kv_indptr[: bs + 1] if swa else None,  # 窗口KV索引指针 # 窗口KV索引指针
                window_kv_indices=self.cuda_graph_window_kv_indices if swa else None,  # 窗口KV索引 # 窗口KV索引
                window_num_kv_splits=(  # 窗口KV分片数 # 窗口KV分片数
                    self.cuda_graph_window_num_kv_splits if swa else None  # 条件选择 # 条件选择
                ),
                window_kv_offsets=None,  # 窗口KV偏移量为空 # 窗口KV偏移量
                swa_attn_logits=self.cuda_graph_swa_attn_logits,  # SWA注意力logits # SWA注意力logits
            )
        elif forward_mode.is_target_verify():  # 如果是目标验证模式 # 检查目标验证模式
            custom_mask = (  # 确定自定义掩码 # 确定自定义掩码
                self.cuda_graph_custom_mask  # CUDA图自定义掩码 # CUDA图掩码
                if spec_info is not None  # 投机信息存在 # 投机信息存在
                and getattr(spec_info, "custom_mask", None) is not None  # 且有自定义掩码 # 有自定义掩码
                else None  # 否则为空 # 否则为空
            )
            return ForwardMetadata(  # 返回目标验证模式的ForwardMetadata # 返回目标验证元数据
                attn_logits=None,  # 注意力logits为空 # 注意力logits
                attn_lse=None,  # 注意力LSE为空 # 注意力LSE
                max_extend_len=self.num_draft_tokens,  # 最大扩展长度为草稿token数 # 最大扩展长度
                num_kv_splits=None,  # KV分片数为空 # KV分片数
                kv_indptr=self.kv_indptr[: bs + 1],  # KV索引指针 # KV索引指针
                kv_indices=self.cuda_graph_kv_indices,  # KV索引 # KV索引
                qo_indptr=self.qo_indptr[: bs + 1],  # QO索引指针 # QO索引指针
                custom_mask=custom_mask,  # 自定义掩码 # 自定义掩码
                mask_indptr=self.mask_indptr[: bs + 1],  # 掩码索引指针 # 掩码索引指针
                window_kv_indptr=self.window_kv_indptr[: bs + 1] if swa else None,  # 窗口KV索引指针 # 窗口KV索引指针
                window_kv_indices=self.cuda_graph_window_kv_indices if swa else None,  # 窗口KV索引 # 窗口KV索引
                window_num_kv_splits=(  # 窗口KV分片数 # 窗口KV分片数
                    self.cuda_graph_window_num_kv_splits if swa else None  # 条件选择 # 条件选择
                ),
                window_kv_offsets=self.cuda_graph_window_kv_offsets if swa else None,  # 窗口KV偏移量 # 窗口KV偏移量
            )
        elif forward_mode.is_draft_extend(include_v2=True):  # 如果是草稿扩展模式 # 检查草稿扩展模式
            return ForwardMetadata(  # 返回草稿扩展模式的ForwardMetadata # 返回草稿扩展元数据
                attn_logits=None,  # 注意力logits为空 # 注意力logits
                attn_lse=None,  # 注意力LSE为空 # 注意力LSE
                max_extend_len=self.speculative_num_steps + 1,  # 最大扩展长度 # 最大扩展长度
                num_kv_splits=None,  # KV分片数为空 # KV分片数
                kv_indptr=self.kv_indptr[: bs + 1],  # KV索引指针 # KV索引指针
                kv_indices=self.cuda_graph_kv_indices,  # KV索引 # KV索引
                qo_indptr=self.qo_indptr[: bs + 1],  # QO索引指针 # QO索引指针
                custom_mask=None,  # 自定义掩码为空 # 自定义掩码
                mask_indptr=None,  # 掩码索引指针为空 # 掩码索引指针
                window_kv_indptr=self.window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
                window_kv_indices=None,  # 窗口KV索引为空 # 窗口KV索引
                window_num_kv_splits=None,  # 窗口KV分片数为空 # 窗口KV分片数
                window_kv_offsets=None,  # 窗口KV偏移量为空 # 窗口KV偏移量
            )
        else:  # 否则 # 其他模式
            raise ValueError(f"Invalid forward mode: {forward_mode=} for CUDA Graph.")  # 抛出无效前向模式错误 # 抛出异常

    def init_forward_metadata_capture_cuda_graph(  # CUDA图捕获时初始化前向元数据 # CUDA图捕获初始化前向元数据
        self,
        bs: int,  # 批次大小 # 批次大小
        num_tokens: int,  # token数量 # token数量
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度
        encoder_lens: Optional[torch.Tensor],  # 编码器长度 # 编码器长度
        forward_mode: ForwardMode,  # 前向模式 # 前向模式
        spec_info: Optional[SpecInput],  # 投机解码信息 # 投机解码信息
    ):
        assert encoder_lens is None, "Not supported"  # 断言编码器长度不支持 # 编码器长度不支持

        # Multi-step speculative decode: kv buffers come from spec_info rather
        # than the cuda-graph pool, so replay is not involved for this path.
        # 多步投机解码：KV缓冲区来自spec_info而非CUDA图池，因此此路径不涉及重放。
        if forward_mode.is_decode_or_idle() and spec_info is not None:  # 如果是解码模式且有投机信息 # 检查多步投机
            self.forward_metadata = ForwardMetadata(  # 创建ForwardMetadata # 创建前向元数据
                attn_logits=self.cuda_graph_attn_logits,  # 注意力logits # 注意力logits
                attn_lse=self.cuda_graph_attn_lse,  # 注意力LSE # 注意力LSE
                max_extend_len=None,  # 最大扩展长度为空 # 最大扩展长度
                num_kv_splits=self.cuda_graph_num_kv_splits,  # KV分片数 # KV分片数
                kv_indptr=spec_info.kv_indptr,  # KV索引指针来自投机信息 # KV索引指针
                kv_indices=spec_info.kv_indices,  # KV索引来自投机信息 # KV索引
                qo_indptr=None,  # QO索引指针为空 # QO索引指针
                custom_mask=None,  # 自定义掩码为空 # 自定义掩码
                mask_indptr=None,  # 掩码索引指针为空 # 掩码索引指针
                window_kv_indptr=self.window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
                window_kv_indices=None,  # 窗口KV索引为空 # 窗口KV索引
                window_num_kv_splits=None,  # 窗口KV分片数为空 # 窗口KV分片数
                window_kv_offsets=None,  # 窗口KV偏移量为空 # 窗口KV偏移量
                swa_attn_logits=self.cuda_graph_swa_attn_logits,  # SWA注意力logits # SWA注意力logits
            )
            return  # 返回 # 返回

        # Run the same buffer update as replay, then freeze the result into
        # a ForwardMetadata whose tensor fields point into the cuda-graph buffers.
        # 运行与重放相同的缓冲区更新，然后将结果冻结为ForwardMetadata，
        # 其张量字段指向CUDA图缓冲区。
        self.init_forward_metadata_replay_cuda_graph(  # 调用重放初始化 # 调用重放初始化
            bs=bs,  # 批次大小 # 批次大小
            req_pool_indices=req_pool_indices,  # 请求池索引 # 请求池索引
            seq_lens=seq_lens,  # 序列长度 # 序列长度
            seq_lens_sum=None,  # 序列长度总和为空 # 序列长度总和
            encoder_lens=encoder_lens,  # 编码器长度 # 编码器长度
            forward_mode=forward_mode,  # 前向模式 # 前向模式
            spec_info=spec_info,  # 投机解码信息 # 投机解码信息
            seq_lens_cpu=None,  # CPU序列长度为空 # CPU序列长度
        )
        self.forward_metadata = self._build_cuda_graph_forward_metadata(  # 构建CUDA图前向元数据 # 构建CUDA图前向元数据
            bs, forward_mode, spec_info  # 传入参数 # 传入参数
        )

    def init_forward_metadata_replay_cuda_graph(  # CUDA图重放时初始化前向元数据 # CUDA图重放初始化前向元数据
        self,
        bs: int,  # 批次大小 # 批次大小
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度
        seq_lens_sum: int,  # 序列长度总和 # 序列长度总和
        encoder_lens: Optional[torch.Tensor],  # 编码器长度 # 编码器长度
        forward_mode: ForwardMode,  # 前向模式 # 前向模式
        spec_info: Optional[SpecInput],  # 投机解码信息 # 投机解码信息
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度 # CPU序列长度
    ):
        # NOTE: encoder_lens expected to be zeros or None
        # 注意：encoder_lens应为零或None
        if forward_mode.is_decode_or_idle():  # 如果是解码或空闲模式 # 检查解码模式
            assert spec_info is None, "Multi-step cuda graph init is not done here."  # 断言多步CUDA图初始化不在此处 # 断言
            _, _, window_kv_lens = self._update_decode_kv_buffers(  # 更新解码KV缓冲区 # 更新解码KV缓冲区
                bs, seq_lens, req_pool_indices  # 传入参数 # 传入参数
            )
            self.get_num_kv_splits(self.cuda_graph_num_kv_splits[:bs], seq_lens[:bs])  # 计算KV分片数 # 计算KV分片数
            if window_kv_lens is not None:  # 如果窗口KV长度存在 # 检查窗口KV长度
                self.get_num_kv_splits(  # 计算窗口KV分片数 # 计算窗口KV分片数
                    self.cuda_graph_window_num_kv_splits[:bs], window_kv_lens[:bs]  # 传入参数 # 传入参数
                )
        elif forward_mode.is_target_verify():  # 如果是目标验证模式 # 检查目标验证模式
            bs = len(req_pool_indices)  # 从请求池索引获取批次大小 # 获取批次大小
            self._update_target_verify_buffers(  # 更新目标验证缓冲区 # 更新目标验证缓冲区
                bs, seq_lens, req_pool_indices, spec_info  # 传入参数 # 传入参数
            )
        elif forward_mode.is_draft_extend(include_v2=True):  # 如果是草稿扩展模式 # 检查草稿扩展模式
            self._update_draft_extend_buffers(  # 更新草稿扩展缓冲区 # 更新草稿扩展缓冲区
                bs, seq_lens, req_pool_indices, forward_mode, spec_info  # 传入参数 # 传入参数
            )
        else:  # 否则 # 其他模式
            raise ValueError(  # 抛出无效前向模式错误 # 抛出异常
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."  # 错误提示 # 错误提示
            )

    def get_cuda_graph_seq_len_fill_value(self):  # 获取CUDA图序列长度填充值 # 获取CUDA图序列长度填充值
        return 1  # 返回1 # 返回1

    def get_verify_buffers_to_fill_after_draft(self):  # 获取草稿后需要填充的验证缓冲区 # 获取草稿后验证缓冲区
        """
        Return buffers for verify attention kernels that needs to be filled after draft.
        # 返回草稿后需要填充的验证注意力内核缓冲区。

        Typically, these are tree mask and position buffers.
        通常，这些是树掩码和位置缓冲区。
        """
        return [self.cuda_graph_custom_mask, None]  # 返回自定义掩码和None # 返回缓冲区列表

    def update_verify_buffers_to_fill_after_draft(  # 更新草稿后需要填充的验证缓冲区 # 更新草稿后验证缓冲区
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]  # 投机信息、CUDA图批次大小 # 投机信息和批次大小
    ):
        pass  # 空实现 # 空实现

    def forward_extend(  # 扩展阶段前向传播 # 扩展阶段前向传播
        self,
        q: torch.Tensor,  # 查询张量 # 查询
        k: torch.Tensor,  # 键张量 # 键
        v: torch.Tensor,  # 值张量 # 值
        layer: RadixAttention,  # 注意力层 # 注意力层
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存 # 是否保存KV缓存
        sinks=None,  # 注意力汇聚点 # 注意力汇聚点
    ):
        # TODO: reuse the buffer across layers
        # 待办：跨层复用缓冲区
        if layer.qk_head_dim != layer.v_head_dim:  # 如果QK头维度不等于V头维度 # 检查头维度差异
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))  # 创建不同维度的输出 # 创建适配维度输出
        else:  # 否则 # 维度相同
            o = torch.empty_like(q)  # 创建与查询相同形状的输出 # 创建同形状输出

        if k is None and v is None:  # 如果K和V都为空 # 检查K和V
            pool = self.token_to_kv_pool  # 获取KV池 # KV池
            cache_loc = forward_batch.out_cache_loc  # 获取缓存位置 # 缓存位置
            if isinstance(pool, SWAKVPool) and pool.layers_mapping[layer.layer_id][1]:  # 如果是SWA池且该层是SWA层 # 检查SWA池
                cache_loc = pool.translate_loc_from_full_to_swa(cache_loc)  # 转换缓存位置 # 转换缓存位置
            k_buffer, v_buffer = pool.get_kv_buffer(layer.layer_id)  # 获取KV缓存 # 获取KV缓存
            k = k_buffer[cache_loc]  # 从缓存获取键 # 获取键
            v = v_buffer[cache_loc]  # 从缓存获取值 # 获取值
        elif k is None or v is None:  # 如果K或V有一个为空 # 检查部分为空
            raise ValueError("Both k and v should be None or not None")  # 抛出数值错误 # 抛出异常
        else:  # 否则（K和V都不为空） # K和V都存在
            # Save KV cache first (must do this before unified kernel)
            # 先保存KV缓存（必须在统一内核之前执行）
            if save_kv_cache:  # 如果需要保存KV缓存 # 检查是否保存
                if layer.k_scale is None:  # 如果键缩放因子为空 # 检查键缩放
                    self.token_to_kv_pool.set_kv_buffer(  # 保存KV缓存 # 保存KV缓存
                        layer,  # 注意力层 # 注意力层
                        forward_batch.out_cache_loc,  # 输出缓存位置 # 缓存位置
                        k,  # 键 # 键
                        v,  # 值 # 值
                    )
                elif self.use_mla:  # 如果使用MLA # 检查MLA
                    # For MLA, scale K manually before storing since MLATokenToKVPool
                    # doesn't accept scale parameters. Clone to protect k from mutation
                    # since it's used later in the attention kernel.
                    # 对于MLA，存储前手动缩放K，因为MLATokenToKVPool不接受缩放参数。
                    # 克隆以保护K免受变异，因为它稍后在注意力内核中使用。
                    k_scaled = k.clone().div_(layer.k_scale)  # 克隆并缩放K # 克隆并缩放K
                    self.token_to_kv_pool.set_kv_buffer(  # 保存KV缓存 # 保存KV缓存
                        layer,  # 注意力层 # 注意力层
                        forward_batch.out_cache_loc,  # 输出缓存位置 # 缓存位置
                        k_scaled,  # 缩放后的键 # 缩放后的键
                        v,  # 值 # 值
                    )
                else:  # 否则（非MLA且有缩放） # 非MLA有缩放
                    self.token_to_kv_pool.set_kv_buffer(  # 保存KV缓存 # 保存KV缓存
                        layer,  # 注意力层 # 注意力层
                        forward_batch.out_cache_loc,  # 输出缓存位置 # 缓存位置
                        k.clone(),  # cloned to protect k,v from in-place mutation in set_kv_buffer # 克隆以保护k,v免受set_kv_buffer中的原地变异
                        v.clone(),  # 克隆v # 克隆v
                        layer.k_scale,  # 键缩放因子 # 键缩放因子
                        layer.v_scale,  # 值缩放因子 # 值缩放因子
                    )

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)  # 计算logits软上限 # 计算logits上限

        causal = True  # 默认使用因果注意力 # 默认因果
        if (  # 如果 # 检查非因果条件
            layer.is_cross_attention  # 交叉注意力 # 交叉注意力
            or layer.attn_type == AttentionType.ENCODER_ONLY  # 仅编码器 # 仅编码器
            or (  # 或者 # 或者
                layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL  # 双向解码器 # 双向解码器
                and self.allow_bidirectional_attention_in_extend  # 且允许双向注意力 # 且允许双向
            )
        ):
            causal = False  # 禁用因果 # 禁用因果

        # Deterministic mode: use unified 1-stage kernel
        # 确定性模式：使用统一1阶段内核
        if self.enable_deterministic:  # 如果启用确定性推理 # 检查确定性
            return self._forward_extend_unified(  # 调用统一扩展前向 # 调用统一扩展前向
                q, o, layer, forward_batch, causal, logits_soft_cap, sinks  # 传入参数 # 传入参数
            )

        # Normal mode: use original 2-stage kernel
        # 普通模式：使用原始2阶段内核
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:  # 如果启用滑动窗口 # 检查滑动窗口
            sliding_window_size = (  # 获取滑动窗口大小 # 获取滑动窗口大小
                layer.sliding_window_size  # 层的滑动窗口大小 # 滑动窗口大小
            )  # Needed for sliding window mask # 滑动窗口掩码所需
            kv_indptr = self.forward_metadata.window_kv_indptr  # 使用窗口KV索引指针 # 使用窗口KV索引指针
            kv_indices = self.forward_metadata.window_kv_indices  # 使用窗口KV索引 # 使用窗口KV索引
            window_kv_offsets = self.forward_metadata.window_kv_offsets  # 使用窗口KV偏移量 # 使用窗口KV偏移量
        else:  # 否则 # 无滑动窗口
            sliding_window_size = -1  # 滑动窗口大小设为-1 # 禁用滑动窗口
            kv_indptr = self.forward_metadata.kv_indptr  # 使用全量KV索引指针 # 使用全量KV索引指针
            kv_indices = self.forward_metadata.kv_indices  # 使用全量KV索引 # 使用全量KV索引
            window_kv_offsets = None  # 窗口KV偏移量为空 # 窗口KV偏移量

        if layer.k_scale is not None and layer.v_scale is not None:  # 如果键和值缩放因子都存在 # 检查缩放因子
            k_descale = layer.k_scale_float  # 获取键反缩放因子 # 键反缩放因子
            v_descale = layer.v_scale_float  # 获取值反缩放因子 # 值反缩放因子
        else:  # 否则 # 无缩放因子
            k_descale = 1.0  # 键反缩放因子为1.0 # 默认1.0
            v_descale = 1.0  # 值反缩放因子为1.0 # 默认1.0

        self.extend_attention_fwd(  # 调用扩展注意力前向 # 调用扩展注意力前向
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),  # 重塑查询形状 # 查询
            k.contiguous(),  # 确保键连续 # 键（连续）
            v.contiguous(),  # 确保值连续 # 值（连续）
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),  # 重塑输出形状 # 输出
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),  # 获取键缓存 # 键缓存
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),  # 获取值缓存 # 值缓存
            self.forward_metadata.qo_indptr,  # QO索引指针 # QO索引指针
            kv_indptr,  # KV索引指针 # KV索引指针
            kv_indices,  # KV索引 # KV索引
            self.forward_metadata.custom_mask,  # 自定义掩码 # 自定义掩码
            causal,  # 是否因果 # 因果标志
            self.forward_metadata.mask_indptr,  # 掩码索引指针 # 掩码索引指针
            self.forward_metadata.max_extend_len,  # 最大扩展长度 # 最大扩展长度
            k_descale,  # 键反缩放因子 # 键反缩放
            v_descale,  # 值反缩放因子 # 值反缩放
            layer.scaling,  # 层缩放因子 # 层缩放
            logit_cap=logits_soft_cap,  # logits上限 # logits上限
            sliding_window_size=sliding_window_size,  # 滑动窗口大小 # 滑动窗口大小
            sinks=sinks,  # 注意力汇聚点 # 汇聚点
            window_kv_offsets=window_kv_offsets,  # 窗口KV偏移量 # 窗口KV偏移量
            xai_temperature_len=layer.xai_temperature_len,  # XAI温度长度 # XAI温度长度
        )
        return o  # 返回输出 # 返回输出

    def _forward_extend_unified(  # 统一扩展前向传播（内部方法） # 统一1阶段扩展前向
        self,
        q: torch.Tensor,  # 查询张量 # 查询
        o: torch.Tensor,  # 输出张量 # 输出
        layer: RadixAttention,  # 注意力层 # 注意力层
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次
        causal: bool,  # 是否因果注意力 # 是否因果
        logits_soft_cap: float,  # logits软上限 # logits上限
        sinks: Optional[torch.Tensor],  # 注意力汇聚点 # 汇聚点
    ):
        """
        Unified 1-stage extend attention for deterministic inference.
        Both prefix and extend KV are accessed through unified kv_indices.
        """
        # 确定性推理的统一1阶段扩展注意力。
        # 前缀和扩展KV通过统一的kv_indices访问。
        bs = forward_batch.batch_size  # 获取批次大小 # 批次大小

        # Determine sliding window settings
        # 确定滑动窗口设置
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:  # 如果启用滑动窗口 # 检查滑动窗口
            sliding_window_size = layer.sliding_window_size  # 获取滑动窗口大小 # 滑动窗口大小
            # Note: for unified kernel, we use full kv_indptr (not window)
            # 注意：对于统一内核，使用全量kv_indptr（而非窗口）
            prefix_kv_indptr = self.forward_metadata.window_kv_indptr  # 前缀KV索引指针 # 前缀KV索引指针
            prefix_kv_indices = self.forward_metadata.window_kv_indices  # 前缀KV索引 # 前缀KV索引
            # Compute window start positions (absolute position of first key in window)
            # 计算窗口起始位置（窗口中第一个键的绝对位置）
            # window_start_pos = seq_len - window_len
            # window_start_pos = 序列长度 - 窗口长度
            window_kv_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]  # 计算窗口KV长度 # 计算窗口KV长度
            # Handle TARGET_VERIFY mode where extend_prefix_lens might not be set
            # 处理TARGET_VERIFY模式中extend_prefix_lens可能未设置的情况
            if forward_batch.extend_prefix_lens is not None:  # 如果扩展前缀长度存在 # 检查前缀长度
                window_start_pos = (  # 计算窗口起始位置 # 计算窗口起始位置
                    forward_batch.extend_prefix_lens[:bs] - window_kv_lens  # 前缀长度减去窗口KV长度 # 计算公式
                )
            else:  # 否则 # 无前缀长度
                # Infer from spec_info: prefix_len = seq_len - draft_token_num
                # 从spec_info推断：prefix_len = seq_len - draft_token_num
                if forward_batch.spec_info is not None and hasattr(  # 如果投机信息存在且有draft_token_num属性 # 检查投机信息
                    forward_batch.spec_info, "draft_token_num"  # 检查属性 # 检查属性
                ):
                    extend_prefix_lens = (  # 计算扩展前缀长度 # 计算扩展前缀长度
                        forward_batch.seq_lens[:bs]  # 序列长度 # 序列长度
                        - forward_batch.spec_info.draft_token_num  # 减去草稿token数 # 减去草稿token数
                    )
                    window_start_pos = extend_prefix_lens - window_kv_lens  # 计算窗口起始位置 # 计算窗口起始位置
                else:  # 否则 # 无法推断
                    window_start_pos = None  # 窗口起始位置为空 # 窗口起始位置为空
        else:  # 否则 # 无滑动窗口
            sliding_window_size = -1  # 滑动窗口大小设为-1 # 禁用滑动窗口
            prefix_kv_indptr = self.forward_metadata.kv_indptr  # 使用全量KV索引指针 # 全量KV索引指针
            prefix_kv_indices = self.forward_metadata.kv_indices  # 使用全量KV索引 # 全量KV索引
            window_start_pos = None  # 窗口起始位置为空 # 窗口起始位置为空

        extend_kv_indices = forward_batch.out_cache_loc  # 获取扩展KV索引 # 扩展KV索引
        pool = self.token_to_kv_pool  # 获取KV池 # KV池
        if (  # 如果 # 检查SWA条件
            layer.sliding_window_size is not None  # 滑动窗口大小存在 # 滑动窗口
            and layer.sliding_window_size > -1  # 且有效 # 且有效
            and isinstance(pool, SWAKVPool)  # 且是SWA池 # 且是SWA池
            and pool.layers_mapping[layer.layer_id][1]  # 且该层是SWA层 # 且是SWA层
        ):
            extend_kv_indices = pool.translate_loc_from_full_to_swa(extend_kv_indices)  # 转换缓存位置 # 转换缓存位置

        # Handle cases where extend_seq_lens or extend_start_loc might not be set
        # In speculative decoding, we can infer these from spec_info or compute them
        # 处理extend_seq_lens或extend_start_loc可能未设置的情况
        # 在投机解码中，我们可以从spec_info推断或计算这些值
        if forward_batch.extend_seq_lens is None:  # 如果扩展序列长度为空 # 检查扩展序列长度
            # TARGET_VERIFY mode: infer extend_seq_lens from spec_info
            # TARGET_VERIFY模式：从spec_info推断extend_seq_lens
            if forward_batch.spec_info is not None and hasattr(  # 如果投机信息存在且有draft_token_num属性 # 检查投机信息
                forward_batch.spec_info, "draft_token_num"  # 检查属性 # 检查属性
            ):
                draft_token_num = forward_batch.spec_info.draft_token_num  # 获取草稿token数 # 草稿token数
                extend_seq_lens = torch.full(  # 创建填充为草稿token数的张量 # 创建扩展序列长度
                    (bs,), draft_token_num, dtype=torch.int32, device=self.device  # 形状、填充值、类型、设备 # 形状、值、类型、设备
                )
            else:  # 否则 # 无法推断
                raise RuntimeError(  # 抛出运行时错误 # 抛出异常
                    "extend_seq_lens is None but cannot infer from spec_info. "  # 错误提示 # 错误提示
                    "This should not happen in TARGET_VERIFY mode."  # 错误提示 # 错误提示
                )
        else:  # 否则 # 扩展序列长度存在
            extend_seq_lens = forward_batch.extend_seq_lens  # 使用前向批次的扩展序列长度 # 使用已有值

        # Check extend_start_loc separately - it might be None even when extend_seq_lens is set
        # 单独检查extend_start_loc——即使extend_seq_lens已设置，它也可能为None
        if forward_batch.extend_start_loc is None:  # 如果扩展起始位置为空 # 检查扩展起始位置
            # Compute extend_start_loc from extend_seq_lens
            # 从extend_seq_lens计算extend_start_loc
            # extend_start_loc[i] = sum(extend_seq_lens[0:i])
            # extend_start_loc[i] = sum(extend_seq_lens[0:i])
            extend_start_loc = torch.cat(  # 拼接计算扩展起始位置 # 拼接计算起始位置
                [
                    torch.zeros(1, dtype=torch.int32, device=self.device),  # 首元素为0 # 首元素为0
                    torch.cumsum(extend_seq_lens[:-1], dim=0),  # 累积和（不含最后一个） # 累积和
                ]
            )
        else:  # 否则 # 扩展起始位置存在
            extend_start_loc = forward_batch.extend_start_loc  # 使用前向批次的扩展起始位置 # 使用已有值

        unified_kv_indptr, unified_kv_indices, prefix_lens = (  # 构建统一KV索引 # 构建统一KV索引
            self.build_unified_kv_indices(  # 调用统一KV索引构建函数 # 调用构建函数
                prefix_kv_indptr,  # 前缀KV索引指针 # 前缀KV索引指针
                prefix_kv_indices,  # 前缀KV索引 # 前缀KV索引
                extend_start_loc,  # 扩展起始位置 # 扩展起始位置
                extend_seq_lens,  # 扩展序列长度 # 扩展序列长度
                extend_kv_indices,  # 扩展KV索引 # 扩展KV索引
                bs,  # 批次大小 # 批次大小
            )
        )

        # Convert prefix_lens to int32 for the kernel
        # 将prefix_lens转换为int32以供内核使用
        prefix_lens = prefix_lens.to(torch.int32)  # 类型转换 # 类型转换

        if layer.k_scale is not None and layer.v_scale is not None:  # 如果键和值缩放因子都存在 # 检查缩放因子
            k_descale = layer.k_scale_float  # 获取键反缩放因子 # 键反缩放因子
            v_descale = layer.v_scale_float  # 获取值反缩放因子 # 值反缩放因子
        else:  # 否则 # 无缩放因子
            k_descale = 1.0  # 键反缩放因子为1.0 # 默认1.0
            v_descale = 1.0  # 值反缩放因子为1.0 # 默认1.0

        # Call unified kernel
        # 调用统一内核
        self.extend_attention_fwd_unified(  # 调用统一扩展注意力前向 # 调用统一扩展注意力
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),  # 重塑查询形状 # 查询
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),  # 重塑输出形状 # 输出
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),  # 获取键缓存 # 键缓存
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),  # 获取值缓存 # 值缓存
            k_descale,  # 键反缩放因子 # 键反缩放
            v_descale,  # 值反缩放因子 # 值反缩放
            self.forward_metadata.qo_indptr,  # QO索引指针 # QO索引指针
            unified_kv_indptr,  # 统一KV索引指针 # 统一KV索引指针
            unified_kv_indices,  # 统一KV索引 # 统一KV索引
            prefix_lens,  # 前缀长度 # 前缀长度
            self.forward_metadata.max_extend_len,  # 最大扩展长度 # 最大扩展长度
            custom_mask=self.forward_metadata.custom_mask,  # 自定义掩码 # 自定义掩码
            mask_indptr=self.forward_metadata.mask_indptr,  # 掩码索引指针 # 掩码索引指针
            sm_scale=layer.scaling,  # softmax缩放因子 # softmax缩放
            logit_cap=logits_soft_cap,  # logits上限 # logits上限
            is_causal=causal,  # 是否因果 # 因果标志
            sliding_window_size=sliding_window_size,  # 滑动窗口大小 # 滑动窗口大小
            sinks=sinks,  # 注意力汇聚点 # 汇聚点
            window_start_pos=window_start_pos,  # 窗口起始位置 # 窗口起始位置
            xai_temperature_len=layer.xai_temperature_len,  # XAI温度长度 # XAI温度长度
        )

        return o  # 返回输出 # 返回输出

    def forward_decode(  # 解码阶段前向传播 # 解码阶段前向传播
        self,
        q: torch.Tensor,  # 查询张量 # 查询
        k: torch.Tensor,  # 键张量 # 键
        v: torch.Tensor,  # 值张量 # 值
        layer: RadixAttention,  # 注意力层 # 注意力层
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存 # 是否保存KV缓存
        sinks=None,  # 注意力汇聚点 # 汇聚点
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        # 在torch.compile期间，rotary_emb有一个bug导致输出值为3D张量形状。此处正确重塑输出。
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)  # 重塑查询为一维token # 重塑查询

        # TODO: reuse the buffer across layers
        # 待办：跨层复用缓冲区
        if layer.qk_head_dim != layer.v_head_dim:  # 如果QK头维度不等于V头维度 # 检查头维度差异
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))  # 创建不同维度的输出 # 创建适配维度输出
        else:  # 否则 # 维度相同
            o = torch.empty_like(q)  # 创建与查询相同形状的输出 # 创建同形状输出

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)  # 计算logits软上限 # 计算logits上限

        if save_kv_cache:  # 如果需要保存KV缓存 # 检查是否保存
            if self.use_mla:  # 如果使用MLA # 检查MLA
                if layer.k_scale is not None:  # 如果键缩放因子存在 # 检查键缩放
                    # MLATokenToKVPool doesn't accept scale parameters; k is unused
                    # after this point in decode, so scale in place.
                    # MLATokenToKVPool不接受缩放参数；在解码中k此后不再使用，因此原地缩放。
                    k.div_(layer.k_scale)  # 原地缩放K # 原地缩放K
                self.token_to_kv_pool.set_kv_buffer(  # 保存KV缓存 # 保存KV缓存
                    layer,  # 注意力层 # 注意力层
                    forward_batch.out_cache_loc,  # 输出缓存位置 # 缓存位置
                    k,  # 键 # 键
                    v,  # 值 # 值
                )
            else:  # 否则（非MLA） # 非MLA
                self.token_to_kv_pool.set_kv_buffer(  # 保存KV缓存 # 保存KV缓存
                    layer,  # 注意力层 # 注意力层
                    forward_batch.out_cache_loc,  # 输出缓存位置 # 缓存位置
                    k,  # 键 # 键
                    v,  # 值 # 值
                    layer.k_scale,  # 键缩放因子 # 键缩放因子
                    layer.v_scale,  # 值缩放因子 # 值缩放因子
                )

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:  # 如果启用滑动窗口 # 检查滑动窗口
            kv_indptr = self.forward_metadata.window_kv_indptr  # 使用窗口KV索引指针 # 使用窗口KV索引指针
            kv_indices = self.forward_metadata.window_kv_indices  # 使用窗口KV索引 # 使用窗口KV索引
        else:  # 否则 # 无滑动窗口
            kv_indptr = self.forward_metadata.kv_indptr  # 使用全量KV索引指针 # 使用全量KV索引指针
            kv_indices = self.forward_metadata.kv_indices  # 使用全量KV索引 # 使用全量KV索引

        if layer.k_scale is not None and layer.v_scale is not None:  # 如果键和值缩放因子都存在 # 检查缩放因子
            k_descale = layer.k_scale_float  # 获取键反缩放因子 # 键反缩放因子
            v_descale = layer.v_scale_float  # 获取值反缩放因子 # 值反缩放因子
        else:  # 否则 # 无缩放因子
            k_descale = 1.0  # 键反缩放因子为1.0 # 默认1.0
            v_descale = 1.0  # 值反缩放因子为1.0 # 默认1.0

        # Select the correctly-sized attn_logits buffer for this layer.
        # The triton kernel's // Lv stride trick requires attn_logits.shape[-1]
        # to exactly match the layer's v_head_dim.
        # 为此层选择正确大小的attn_logits缓冲区。
        # Triton内核的// Lv步幅技巧要求attn_logits.shape[-1]精确匹配层的v_head_dim。
        attn_logits = self.forward_metadata.attn_logits  # 获取注意力logits # 注意力logits
        if (  # 如果 # 检查SWA注意力logits
            self.forward_metadata.swa_attn_logits is not None  # SWA注意力logits存在 # SWA logits存在
            and layer.v_head_dim == self.swa_v_head_dim  # 且层的V头维度等于SWA V头维度 # 维度匹配
        ):
            attn_logits = self.forward_metadata.swa_attn_logits  # 使用SWA注意力logits # 使用SWA logits

        self.decode_attention_fwd(  # 调用解码注意力前向 # 调用解码注意力前向
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),  # 重塑查询形状 # 查询
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),  # 获取键缓存 # 键缓存
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),  # 获取值缓存 # 值缓存
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),  # 重塑输出形状 # 输出
            kv_indptr,  # KV索引指针 # KV索引指针
            kv_indices,  # KV索引 # KV索引
            attn_logits,  # 注意力logits # 注意力logits
            self.forward_metadata.attn_lse,  # 注意力LSE # 注意力LSE
            self.forward_metadata.num_kv_splits,  # KV分片数 # KV分片数
            self.max_kv_splits,  # 最大KV分片数 # 最大KV分片数
            layer.scaling,  # 层缩放因子 # 层缩放
            k_descale,  # 键反缩放因子 # 键反缩放
            v_descale,  # 值反缩放因子 # 值反缩放
            logit_cap=logits_soft_cap,  # logits上限 # logits上限
            sinks=sinks,  # 注意力汇聚点 # 汇聚点
            xai_temperature_len=layer.xai_temperature_len,  # XAI温度长度 # XAI温度长度
            has_mla=self.use_mla,  # 是否使用MLA # MLA标志
            use_pdl=self.use_pdl,  # 是否使用PDL # PDL标志
        )
        return o  # 返回输出 # 返回输出


class TritonMultiStepDraftBackend:  # Triton多步草稿后端类 # Triton多步草稿后端
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """
    # 将多个Triton注意力后端包装为一个，用于多个连续的草稿解码步骤。
    # CUDA-graph replay rebuilds metadata from preallocated kv_indptr/kv_indices
    # buffers; it never reads seq_lens_cpu / seq_lens_sum.
    # CUDA图重放从预分配的kv_indptr/kv_indices缓冲区重建元数据；
    # 它从不读取seq_lens_cpu / seq_lens_sum。
    needs_cpu_seq_lens: bool = False  # 不需要CPU序列长度 # 不需要CPU序列长度

    def __init__(  # 初始化方法 # 初始化方法
        self,
        model_runner: ModelRunner,  # 模型运行器 # 模型运行器实例
        topk: int,  # topk值 # topk值
        speculative_num_steps: int,  # 投机步数 # 投机步数
    ):
        self.topk = topk  # 保存topk值 # topk值
        self.speculative_num_steps = speculative_num_steps  # 保存投机步数 # 投机步数
        max_bs = model_runner.req_to_token_pool.size * self.topk  # 计算最大批次大小 # 最大批次大小
        self.kv_indptr = torch.zeros(  # 创建零填充的KV索引指针 # 创建KV索引指针
            (  # 形状参数 # 形状
                self.speculative_num_steps,  # 投机步数 # 步数维度
                max_bs + 1,  # 批次大小加1 # 批次维度
            ),
            dtype=torch.int32,  # 数据类型 # 数据类型
            device=model_runner.device,  # 设备 # 设备
        )
        self.attn_backends: List[TritonAttnBackend] = []  # 初始化注意力后端列表 # 注意力后端列表
        for i in range(self.speculative_num_steps - 1):  # 遍历投机步数（减1） # 遍历步数
            self.attn_backends.append(  # 添加注意力后端 # 添加后端
                TritonAttnBackend(  # 创建Triton注意力后端 # 创建Triton后端
                    model_runner,  # 模型运行器 # 模型运行器
                    skip_prefill=True,  # 跳过预填充 # 跳过预填充
                    kv_indptr_buf=self.kv_indptr[i],  # KV索引指针缓冲区 # KV索引指针缓冲区
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len  # 获取最大上下文长度 # 最大上下文长度
        self.num_head = (  # 计算注意力头数 # 注意力头数
            model_runner.model_config.num_attention_heads // get_attention_tp_size()  # 总头数除以TP大小 # 总头数/TP大小
        )
        self.device = model_runner.device  # 保存设备信息 # 设备
        # Cached variables for generate_draft_decode_kv_indices
        # 用于generate_draft_decode_kv_indices的缓存变量
        self.req_to_token_pool = model_runner.req_to_token_pool  # 保存请求到token池 # 请求-token映射池
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]  # 保存池长度 # 池长度
        self.page_size = model_runner.server_args.page_size  # 保存页大小 # 页大小

    def common_template(  # 通用模板方法 # 通用模板方法
        self,
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次
        kv_indices_buffer: Optional[torch.Tensor],  # KV索引缓冲区 # KV索引缓冲区
        call_fn: int,  # 调用函数 # 回调函数
    ):
        if kv_indices_buffer is None:  # 如果KV索引缓冲区为空 # 检查缓冲区
            kv_indices_buffer = self.cuda_graph_kv_indices  # 使用CUDA图KV索引 # 使用CUDA图KV索引

        num_seqs = forward_batch.batch_size  # 获取序列数 # 序列数
        bs = self.topk * num_seqs  # 计算批次大小（topk乘以序列数） # 批次大小
        seq_lens_sum = forward_batch.seq_lens_sum  # 获取序列长度总和 # 序列长度总和
        if seq_lens_sum is None:  # 如果序列长度总和为空 # 检查序列长度总和
            # seq_lens_sum here only slice-clamps a preallocated kv_indices buffer;
            # over-estimate is safe. Use a static UB to skip the per-iter .sum().item() D2H.
            # 此处的seq_lens_sum仅用于切片限制预分配的kv_indices缓冲区；
            # 高估是安全的。使用静态上界以跳过每次迭代的.sum().item() D2H。
            seq_lens_sum = num_seqs * self.max_context_len  # 使用上界估计 # 上界估计

        generate_draft_decode_kv_indices[  # 调用草稿解码KV索引生成 # 生成草稿解码KV索引
            (self.speculative_num_steps, num_seqs, self.topk)  # 启动网格 # 启动网格
        ](
            forward_batch.req_pool_indices,  # 请求池索引 # 请求池索引
            self.req_to_token_pool.req_to_token,  # 请求到token映射表 # 请求-token映射
            forward_batch.seq_lens,  # 序列长度 # 序列长度
            kv_indices_buffer,  # KV索引缓冲区 # KV索引缓冲区
            self.kv_indptr,  # KV索引指针 # KV索引指针
            forward_batch.positions,  # 位置索引 # 位置索引
            self.pool_len,  # 池长度 # 池长度
            kv_indices_buffer.shape[1],  # KV索引缓冲区第二维大小 # 缓冲区大小
            self.kv_indptr.shape[1],  # KV索引指针第二维大小 # 指针大小
            next_power_of_2(num_seqs),  # 序列数的下一个2的幂 # 2的幂
            next_power_of_2(self.speculative_num_steps),  # 投机步数的下一个2的幂 # 2的幂
            next_power_of_2(bs),  # 批次大小的下一个2的幂 # 2的幂
            self.page_size,  # 页大小 # 页大小
        )

        if call_fn is None:  # 如果调用函数为空 # 检查回调函数
            return  # 返回 # 返回

        for i in range(self.speculative_num_steps - 1):  # 遍历投机步数 # 遍历步数
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]  # 设置投机信息的KV索引指针 # 设置KV索引指针
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][  # 设置投机信息的KV索引 # 设置KV索引
                : seq_lens_sum * self.topk + bs * (i + 1)  # 截取有效范围 # 截取范围
            ]
            call_fn(i, forward_batch)  # 调用回调函数 # 调用回调函数

    def init_forward_metadata(self, forward_batch: ForwardBatch):  # 初始化前向元数据 # 初始化前向元数据
        kv_indices = torch.empty(  # 创建空的KV索引张量 # 创建KV索引
            (  # 形状参数 # 形状
                self.speculative_num_steps,  # 投机步数 # 步数
                forward_batch.batch_size * self.topk * self.max_context_len,  # token数 # token数
            ),
            dtype=torch.int64,  # 数据类型 # 数据类型
            device=self.device,  # 设备 # 设备
        )

        def call_fn(i, forward_batch):  # 定义回调函数 # 定义回调函数
            forward_batch.spec_info.kv_indptr = (  # 克隆KV索引指针 # 克隆KV索引指针
                forward_batch.spec_info.kv_indptr.clone()  # 克隆 # 克隆
            )
            forward_batch.spec_info.kv_indices = (  # 克隆KV索引 # 克隆KV索引
                forward_batch.spec_info.kv_indices.clone()  # 克隆 # 克隆
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)  # 初始化注意力后端的前向元数据 # 初始化后端元数据

        self.common_template(forward_batch, kv_indices, call_fn)  # 调用通用模板 # 调用通用模板

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):  # 初始化CUDA图状态 # 初始化CUDA图状态
        self.cuda_graph_kv_indices = torch.zeros(  # 创建零填充的CUDA图KV索引 # 创建CUDA图KV索引
            (self.speculative_num_steps, max_num_tokens * self.max_context_len),  # 形状 # 形状
            dtype=torch.int64,  # 数据类型 # 数据类型
            device=self.device,  # 设备 # 设备
        )
        self.cuda_graph_num_kv_splits = torch.full(  # 创建填充为最大KV分片数的张量 # 创建CUDA图KV分片数
            (max_num_tokens,),  # 形状 # 形状
            self.attn_backends[0].max_kv_splits,  # 填充值 # 填充值
            dtype=torch.int32,  # 数据类型 # 数据类型
            device=self.device,  # 设备 # 设备
        )

        for i in range(self.speculative_num_steps - 1):  # 遍历投机步数 # 遍历步数
            self.attn_backends[i].init_cuda_graph_state(  # 初始化注意力后端的CUDA图状态 # 初始化后端CUDA图状态
                max_bs,  # 最大批次大小 # 最大批次大小
                max_num_tokens,  # 最大token数 # 最大token数
                kv_indices_buf=self.cuda_graph_kv_indices[i],  # KV索引缓冲区 # KV索引缓冲区
                cuda_graph_num_kv_splits_buf=self.cuda_graph_num_kv_splits,  # CUDA图KV分片数缓冲区 # KV分片数缓冲区
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):  # CUDA图捕获时初始化前向元数据 # CUDA图捕获初始化
        def call_fn(i, forward_batch):  # 定义回调函数 # 定义回调函数
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(  # 初始化注意力后端的CUDA图捕获元数据 # 初始化捕获元数据
                forward_batch.batch_size,  # 批次大小 # 批次大小
                forward_batch.batch_size * self.topk,  # 批次大小乘以topk # 批次大小*topk
                forward_batch.req_pool_indices,  # 请求池索引 # 请求池索引
                forward_batch.seq_lens,  # 序列长度 # 序列长度
                encoder_lens=None,  # 编码器长度为空 # 编码器长度
                forward_mode=ForwardMode.DECODE,  # 前向模式为解码 # 解码模式
                spec_info=forward_batch.spec_info,  # 投机解码信息 # 投机信息
            )

        self.common_template(forward_batch, None, call_fn)  # 调用通用模板 # 调用通用模板

    def init_forward_metadata_replay_cuda_graph(  # CUDA图重放时初始化前向元数据 # CUDA图重放初始化
        self, forward_batch: ForwardBatch, bs: int  # 前向批次、批次大小 # 前向批次和批次大小
    ):
        self.common_template(forward_batch, None, None)  # 调用通用模板（不执行回调） # 调用通用模板

        # NOTE: Multi-step's attention backends use the slice of
        # - kv_indptr buffer (cuda graph and non-cuda graph)
        # - kv_indices buffer (cuda graph only)
        # So we don't need to assign the KV indices inside the attention backend.
        # 注意：多步注意力后端使用以下缓冲区的切片：
        # - kv_indptr缓冲区（CUDA图和非CUDA图）
        # - kv_indices缓冲区（仅CUDA图）
        # 因此我们不需要在注意力后端内部分配KV索引。

        # Compute num_kv_splits only once
        # 仅计算一次num_kv_splits
        num_token = forward_batch.batch_size * self.topk  # 计算token数量 # token数量
        self.attn_backends[-1].get_num_kv_splits(  # 计算KV分片数 # 计算KV分片数
            self.attn_backends[-1].cuda_graph_num_kv_splits[:num_token],  # KV分片数缓冲区 # KV分片数缓冲区
            forward_batch.seq_lens[:bs],  # 序列长度 # 序列长度
        )


@triton.jit  # Triton JIT编译装饰器 # Triton JIT编译
def get_num_kv_splits_triton(  # Triton内核：计算KV分片数 # Triton内核计算KV分片数
    num_kv_splits_ptr,  # KV分片数输出指针 # KV分片数输出指针
    seq_lens_ptr,  # 序列长度指针 # 序列长度指针
    num_seq,  # 序列数 # 序列数
    num_group,  # 每组token数 # 每组token数
    num_head,  # 注意力头数 # 注意力头数
    num_kv_head,  # KV头数 # KV头数
    max_kv_splits,  # 最大KV分片数 # 最大KV分片数
    device_core_count,  # 设备核心数 # 设备核心数
    MAX_NUM_SEQ: tl.constexpr,  # 最大序列数（编译时常量） # 最大序列数常量
):
    # TODO: this method is tunable, we need more online serving data to tune it
    # 待办：此方法可调优，我们需要更多在线服务数据来调优
    offs_seq = tl.arange(0, MAX_NUM_SEQ)  # 生成序列偏移量 # 生成序列偏移
    mask_seq = offs_seq < num_seq  # 创建有效序列掩码 # 创建掩码

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)  # 加载序列长度 # 加载序列长度
    max_seq_len = tl.max(seq_lens)  # 计算最大序列长度 # 最大序列长度
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)  # 重新加载序列长度（无效位置填充最大值） # 重新加载
    min_seq_len = tl.min(seq_lens)  # 计算最小序列长度 # 最小序列长度
    if max_seq_len * 8 < min_seq_len * 10:  # 如果序列长度差异不大 # 检查差异
        min_seq_len = max_seq_len  # 最小序列长度设为最大值 # 设为最大值
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)  # 计算第一种KV分片数上限 # 第一种上限
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)  # 计算第一种KV块大小 # 第一种块大小

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    # 注意：这是一个技巧，让num_kv_split随序列长度逐渐增长
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0  # 计算扩展序列长度 # 扩展序列长度
    ext_device_core_count = tl.cast(  # 计算扩展设备核心数 # 扩展核心数
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32  # 设备核心数乘以log2 # 计算公式
    )
    block_h, num_kv_group = 16, num_head // num_kv_head  # 块头数和KV组数 # 块头数和KV组数
    if num_kv_group == 1:  # 如果KV组数为1（MQA） # MQA情况
        token_grid = num_seq * num_group * num_head  # token网格大小 # token网格
    else:  # 否则（GQA） # GQA情况
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        # 来自triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        block_h = tl.minimum(block_h, num_kv_group)  # 限制块头数 # 限制块头数
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)  # token网格大小 # token网格
    max_kv_splits_2 = tl.minimum(  # 计算第二种KV分片数上限 # 第二种上限
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits  # 扩展核心数除以token网格 # 计算公式
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)  # 计算第二种KV块大小 # 第二种块大小

    num_kv_splits = tl.maximum(  # 取两种分片数的最大值 # 取最大值
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)  # 两种分片数 # 两种分片数
    )

    offs_token = offs_seq * num_group  # 计算token偏移量 # token偏移量
    mask_token = offs_token < num_seq * num_group  # 创建有效token掩码 # token掩码
    for i in range(0, num_group):  # 遍历每组 # 遍历组
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)  # 存储KV分片数 # 存储结果


def update_sliding_window_buffer(  # 更新滑动窗口缓冲区 # 更新滑动窗口缓冲区
    window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
    req_to_token,  # 请求到token映射表 # 请求-token映射
    sliding_window_size,  # 滑动窗口大小 # 滑动窗口大小
    seq_lens,  # 序列长度 # 序列长度
    req_pool_indices,  # 请求池索引 # 请求池索引
    bs,  # 批次大小 # 批次大小
    device=None,  # 设备（可选） # 设备
    token_to_kv_pool=None,  # token到KV池（可选） # KV池
    window_kv_indices=None,  # 窗口KV索引（可选） # 窗口KV索引
):
    """Fill window KV buffers for sliding-window attention.
    # 填充滑动窗口注意力的窗口KV缓冲区。

    Pass ``window_kv_indices`` to write into a pre-allocated buffer (CUDA-graph
    path); omit it (or pass ``None``) to allocate a fresh tensor (eager path,
    requires ``device``).
    传入``window_kv_indices``以写入预分配缓冲区（CUDA图路径）；
    省略（或传入``None``）以分配新张量（即时路径，需要``device``）。
    """
    window_kv_lens = torch.minimum(  # 计算窗口KV长度（取序列长度和窗口大小的较小值） # 计算窗口KV长度
        seq_lens,  # 序列长度 # 序列长度
        torch.tensor(sliding_window_size),  # 滑动窗口大小 # 滑动窗口大小
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)  # 计算累积和作为索引指针 # 计算累积和
    window_kv_indptr = window_kv_indptr[: bs + 1]  # 截取当前批次大小 # 截取
    if window_kv_indices is None:  # 如果窗口KV索引为空 # 检查索引
        window_kv_indices = torch.empty(  # 创建空的窗口KV索引 # 创建窗口KV索引
            window_kv_indptr[-1], dtype=torch.int64, device=device  # 形状、类型、设备 # 形状、类型、设备
        )
    window_kv_start_idx = seq_lens - window_kv_lens  # 计算窗口KV起始索引 # 计算起始索引
    create_flashinfer_kv_indices_triton[(bs,)](  # 调用Triton内核创建KV索引 # 调用Triton内核
        req_to_token,  # 请求到token映射表 # 请求-token映射
        req_pool_indices,  # 请求池索引 # 请求池索引
        window_kv_lens,  # 窗口KV长度 # 窗口KV长度
        window_kv_indptr,  # 窗口KV索引指针 # 窗口KV索引指针
        window_kv_start_idx,  # 窗口KV起始索引 # 窗口KV起始索引
        window_kv_indices,  # 窗口KV索引输出 # 窗口KV索引输出
        req_to_token.stride(0),  # 映射表步幅 # 步幅
    )
    if hasattr(token_to_kv_pool, "translate_loc_from_full_to_swa"):  # 如果KV池有位置转换方法 # 检查转换方法
        kv_last_index = window_kv_indptr[-1]  # 获取最后一个KV索引 # 最后索引
        # Flush before+after: window_kv_indices is a different tensor than out_cache_loc.
        # 刷新前后：window_kv_indices是与out_cache_loc不同的张量。
        token_to_kv_pool.invalidate_loc_cache()  # 使位置缓存失效 # 失效缓存
        window_kv_indices[:kv_last_index] = (  # 转换窗口KV索引 # 转换索引
            token_to_kv_pool.translate_loc_from_full_to_swa(  # 从全量位置转换为SWA位置 # 转换位置
                window_kv_indices[:kv_last_index]  # 截取有效范围 # 截取
            )
        )
        token_to_kv_pool.invalidate_loc_cache()  # 再次使位置缓存失效 # 失效缓存
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx  # 返回窗口KV相关张量 # 返回
