# DFlash投机解码Worker模块
# 实现DFlash投机解码的Worker类，包含草稿模型初始化、KV缓存管理、
# 贪心采样、验证和前向传播等核心逻辑。
import logging  # 导入日志模块
import math  # 导入数学模块
from copy import deepcopy  # 导入深拷贝
from typing import Optional  # 导入可选类型

import torch  # 导入PyTorch

from sglang.srt.distributed import get_tp_group  # 导入张量并行组
from sglang.srt.managers.schedule_batch import ScheduleBatch  # 导入调度批次
from sglang.srt.managers.scheduler import GenerationBatchResult  # 导入生成批次结果
from sglang.srt.managers.tp_worker import TpModelWorker  # 导入TP模型Worker
from sglang.srt.mem_cache.common import get_last_loc  # 导入获取最后位置函数
from sglang.srt.model_executor.forward_batch_info import (  # 导入前向批次信息
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import (  # 导入服务器参数
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput  # 导入DFlash输入类
from sglang.srt.speculative.dflash_utils import (  # 导入DFlash工具函数
    can_dflash_use_fused_qkv_proj,
    is_dflash_sampling_verify_available,
    parse_dflash_draft_config,
    resolve_dflash_verify_mask_policy,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # 导入投机算法枚举
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func  # 导入请求到token池映射函数
from sglang.srt.utils import is_cuda, is_npu  # 导入设备检测

_is_npu = is_npu()  # 是否为NPU设备


logger = logging.getLogger(__name__)  # 获取日志记录器

_FusedKVMaterializeHelper = None  # 融合KV物化助手类（延迟加载）


def _get_fused_kv_materialize_helper():
    # 延迟加载融合KV物化助手类
    global _FusedKVMaterializeHelper
    if _FusedKVMaterializeHelper is None:  # 尚未加载
        from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
            FusedKVMaterializeHelper,
        )

        _FusedKVMaterializeHelper = FusedKVMaterializeHelper  # 缓存类引用
    return _FusedKVMaterializeHelper  # 返回类


class DFlashWorker:
    """DFlash speculative decoding worker (spec-v1, tp>=1/pp=1)."""
    # DFlash投机解码Worker（spec-v1，tp>=1/pp=1）。

    def __init__(
        self,
        server_args: ServerArgs,  # 服务器参数
        gpu_id: int,  # GPU ID
        tp_rank: int,  # 张量并行排名
        dp_rank: Optional[int],  # 数据并行排名
        moe_ep_rank: int,  # MoE专家并行排名
        attn_cp_rank: int,  # 注意力上下文并行排名
        moe_dp_rank: int,  # MoE数据并行排名
        nccl_port: int,  # NCCL端口
        target_worker: TpModelWorker,  # 目标Worker
    ):
        # 初始化DFlash Worker
        self.server_args = server_args  # 保存服务器参数
        self.gpu_id = gpu_id  # GPU ID
        self.tp_rank = tp_rank  # 张量并行排名
        self.dp_rank = dp_rank  # 数据并行排名
        self.moe_ep_rank = moe_ep_rank  # MoE专家并行排名
        self.attn_cp_rank = attn_cp_rank  # 注意力上下文并行排名
        self.moe_dp_rank = moe_dp_rank  # MoE数据并行排名
        self.nccl_port = nccl_port  # NCCL端口
        self.target_worker = target_worker  # 目标Worker
        self.model_runner = target_worker.model_runner  # 目标模型运行器
        self.page_size = server_args.page_size  # 页大小
        # Normalized in arg_groups.speculative_hook.handle_speculative_decoding.
        # 在arg_groups.speculative_hook.handle_speculative_decoding中归一化。
        self.draft_window_size: Optional[int] = (  # 草稿窗口大小
            server_args.speculative_draft_window_size
        )
        self.use_compact_draft_cache = self.draft_window_size is not None  # 是否使用紧凑草稿缓存
        self.device = target_worker.device  # 设备

        self._warned_sampling_fallback = False  # 是否已警告采样回退
        self._logged_first_verify = False  # 是否已记录首次验证

        # Draft runner (separate KV cache + attention backend).
        # 草稿运行器（独立的KV缓存+注意力后端）。
        # Without draft windowing, the draft worker aliases the target request->token
        # mapping and allocation state. With draft windowing enabled, the draft worker
        # keeps a private compact req->token table over the same global KV index space,
        # so radix-cache/prefix-hit KV remains reusable while draft attention sees only
        # the recent window.
        # 无草稿窗口时，草稿worker别名目标请求->token映射和分配状态。
        # 启用草稿窗口时，草稿worker在相同的全局KV索引空间上维护私有紧凑req->token表，
        # 因此radix-cache/前缀命中KV仍然可复用，而草稿注意力只看到最近的窗口。
        target_req_to_token_pool, target_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()  # 获取目标内存池
        )
        shared_req_to_token_pool = (  # 共享请求到token池
            None if self.use_compact_draft_cache else target_req_to_token_pool  # 紧凑模式不共享
        )
        draft_server_args = deepcopy(server_args)  # 深拷贝服务器参数
        draft_server_args.skip_tokenizer_init = True  # 跳过tokenizer初始化
        draft_backend = draft_server_args.speculative_draft_attention_backend  # 获取草稿注意力后端
        supported_draft_backends = ("flashinfer", "fa3", "fa4", "triton", "ascend")  # 支持的草稿后端
        if draft_backend is None:  # 未指定后端
            draft_backend, _ = draft_server_args.get_attention_backends()  # 自动选择
        if draft_backend is None:  # 仍然无法确定
            # Use triton on ROCm (no FlashInfer), flashinfer on CUDA
            # 在ROCm上使用triton（无FlashInfer），在CUDA上使用flashinfer
            import torch as _torch

            draft_backend = "triton" if _torch.version.hip else "flashinfer"  # 根据平台选择
        elif draft_backend == "trtllm_mha":  # 不支持trtllm_mha
            import torch as _torch

            _fb = "triton" if _torch.version.hip else "flashinfer"  # 回退后端
            logger.warning(
                "DFLASH draft worker does not support 'trtllm_mha' because the "
                "draft path requires non-causal attention. Falling back to "
                "'%s'.",
                _fb,
            )
            draft_backend = _fb  # 使用回退后端
        elif draft_backend not in supported_draft_backends:  # 不支持的后端
            import torch as _torch

            _fb = "triton" if _torch.version.hip else "flashinfer"  # 回退后端
            logger.warning(
                "DFLASH draft worker only supports attention_backend in %s for now, "
                "but got %r. Falling back to '%s'.",
                supported_draft_backends,
                draft_backend,
                _fb,
            )
            draft_backend = _fb  # 使用回退后端
        # Make the draft worker backend explicit and self-contained (no further overrides).
        # 使草稿worker后端显式且自包含（不再覆盖）。
        draft_server_args.speculative_draft_attention_backend = None  # 清除
        draft_server_args.prefill_attention_backend = None  # 清除
        draft_server_args.decode_attention_backend = None  # 清除
        draft_server_args.attention_backend = draft_backend  # 设置统一后端
        # Keep draft context length aligned with the target.
        # 保持草稿上下文长度与目标对齐。
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        saved_server_args = get_global_server_args()  # 保存全局服务器参数
        self.draft_worker = TpModelWorker(  # 创建草稿Worker
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=shared_req_to_token_pool,
            token_to_kv_pool_allocator=target_token_to_kv_pool_allocator,
            memory_pool_config=target_worker.model_runner.memory_pool_config,
        )
        set_global_server_args_for_scheduler(saved_server_args)  # 恢复全局服务器参数
        self.draft_model_runner = self.draft_worker.model_runner  # 草稿模型运行器
        self.draft_model = self.draft_model_runner.model  # 草稿模型
        draft_config = parse_dflash_draft_config(  # 解析DFlash草稿配置
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        if server_args.speculative_num_draft_tokens is None:  # 未指定草稿token数
            # Should not happen (ServerArgs should have inferred it), but keep a fallback.
            # 不应发生（ServerArgs应该已推断），但保留回退。
            self.block_size = int(draft_config.resolve_block_size(default=16))  # 使用配置或默认16
        else:  # 已指定
            self.block_size = int(server_args.speculative_num_draft_tokens)  # 使用指定值
            model_block_size = draft_config.block_size  # 模型配置中的块大小
            if model_block_size is None:  # 配置中未设置
                model_block_size = getattr(self.draft_model, "block_size", None)  # 尝试从模型获取
            if model_block_size is not None and int(model_block_size) != int(  # 块大小不匹配
                self.block_size
            ):
                logger.warning(
                    "DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s.",
                    self.block_size,
                    model_block_size,
                )

        self._mask_token = draft_config.mask_token  # 掩码token字符串
        self._mask_token_id_override = draft_config.mask_token_id  # 掩码token ID覆盖
        self._mask_token_id = self._resolve_mask_token_id(  # 解析掩码token ID
            mask_token=self._mask_token,
            mask_token_id=self._mask_token_id_override,
        )
        if self.tp_rank == 0:  # 主rank打印信息
            logger.info(
                "Initialized DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s, draft_window_size=%s, compact_cache=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.draft_model.__class__.__name__,
                self.block_size,
                self.draft_window_size,
                self.use_compact_draft_cache,
            )
            logger.info(
                "DFLASH draft runner ready. mask_token=%s, mask_token_id=%s, mask_token_id_override=%s",
                self._mask_token,
                self._mask_token_id,
                self._mask_token_id_override,
            )

        self._block_pos_offsets = torch.arange(  # 块内位置偏移
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None  # [cap_bs, block_size] 草稿块ID缓冲区
        self._draft_block_positions_buf: Optional[torch.Tensor] = (  # 草稿块位置缓冲区
            None  # [cap_bs, block_size]
        )
        self._draft_block_tokens_buf: Optional[torch.Tensor] = (  # 草稿块token缓冲区
            None  # [cap_bs, block_size]
        )
        self._draft_block_end_buf: Optional[torch.Tensor] = None  # [cap_bs] 草稿块结束缓冲区
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None  # [cap_bs] CPU上的草稿序列长度缓冲区
        self._draft_block_spec_info = DFlashVerifyInput(  # 草稿块验证信息
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._draft_greedy_gathered_max_buf: Optional[torch.Tensor] = None  # 贪心采样收集最大值缓冲区
        self._draft_greedy_gathered_ids_buf: Optional[torch.Tensor] = None  # 贪心采样收集ID缓冲区
        self._draft_greedy_gather_cap: int = 0  # 贪心采样收集容量
        self._draft_greedy_best_rank_buf: Optional[torch.Tensor] = None  # 最佳排名缓冲区
        self._draft_greedy_rank_index_buf: Optional[torch.Tensor] = None  # 排名索引缓冲区
        self._draft_greedy_selected_ids_buf: Optional[torch.Tensor] = None  # 选中ID缓冲区
        self._draft_greedy_index_cap: int = 0  # 贪心索引容量

        self._use_fused_kv_materialize = is_cuda()  # 是否使用融合KV物化
        self._fused_kv_helper: Optional[object] = None  # 融合KV助手
        if self._use_fused_kv_materialize:  # CUDA上初始化融合KV
            self._init_fused_kv_helper()

    def _init_fused_kv_helper(self) -> None:
        """Initialize the fused KV materialization helper with pre-stacked weights."""
        # 使用预堆叠权重初始化融合KV物化助手。
        try:
            layers = self.draft_model.layers  # 获取模型层
            fused_disable_reason: Optional[str] = None  # 禁用原因

            if len(layers) == 0:  # 没有层
                fused_disable_reason = "no layers found"  # 设置禁用原因

            for layer_idx, layer in enumerate(layers):  # 遍历每一层
                attn = layer.self_attn  # 获取自注意力模块
                eligible, reason = can_dflash_use_fused_qkv_proj(attn.qkv_proj)  # 检查是否可用融合QKV
                if not eligible:  # 不可用
                    fused_disable_reason = f"{reason}: layer={layer_idx}"  # 记录原因
                    break  # 停止检查

                # Keep semantics aligned with set_kv_buffer scaling behavior.
                # 保持语义与set_kv_buffer缩放行为一致。
                k_scale = getattr(attn.attn, "k_scale", None)  # K缩放因子
                v_scale = getattr(attn.attn, "v_scale", None)  # V缩放因子
                if k_scale is not None and not math.isclose(float(k_scale), 1.0):  # K缩放不为1
                    fused_disable_reason = (
                        "non-unit k_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, k_scale={k_scale}"
                    )
                    break  # 停止检查
                if v_scale is not None and not math.isclose(float(v_scale), 1.0):  # V缩放不为1
                    fused_disable_reason = (
                        "non-unit v_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, v_scale={v_scale}"
                    )
                    break  # 停止检查

                rope_is_neox_style = bool(  # 检查RoPE是否为neox风格
                    getattr(attn.rotary_emb, "is_neox_style", True)
                )
                if not rope_is_neox_style:  # 非neox风格
                    fused_disable_reason = (
                        "non-neox RoPE is not supported for fused KV path: "
                        f"layer={layer_idx}, rope_is_neox_style={rope_is_neox_style}"
                    )
                    break  # 停止检查

            if fused_disable_reason is not None:  # 有禁用原因
                if self.tp_rank == 0:  # 主rank打印
                    logger.info(
                        "DFLASH fused KV materialization disabled: %s",
                        fused_disable_reason,
                    )
                self._use_fused_kv_materialize = False  # 禁用融合KV
                self._fused_kv_helper = None  # 清除助手
                return  # 返回

            FusedKVMaterializeHelper = _get_fused_kv_materialize_helper()  # 获取助手类
            first_attn = layers[0].self_attn  # 第一层的注意力模块
            rotary_emb = first_attn.rotary_emb  # 旋转位置编码

            self._fused_kv_helper = FusedKVMaterializeHelper(  # 创建助手实例
                layers=layers,
                rotary_emb=rotary_emb,
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
            )
            if self.tp_rank == 0:  # 主rank打印
                logger.info(
                    "DFLASH fused KV materialization enabled. "
                    "n_layers=%d, num_kv_heads=%d, head_dim=%d",
                    len(layers),
                    first_attn.num_kv_heads,
                    first_attn.head_dim,
                )
        except Exception as e:  # 初始化失败
            logger.warning(
                "DFLASH fused KV initialization failed, falling back to sequential path: %s",
                e,
            )
            self._use_fused_kv_materialize = False  # 回退到顺序路径
            self._fused_kv_helper = None  # 清除助手

    def _ensure_draft_block_buffers(self, bs: int) -> None:
        # 确保草稿块缓冲区大小足够
        cap = (  # 当前容量
            0
            if self._draft_block_ids_buf is None
            else int(self._draft_block_ids_buf.shape[0])
        )
        if cap >= int(bs):  # 容量足够
            return  # 无需扩展

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))  # 新容量
        device = self.device  # 设备
        block_size = int(self.block_size)  # 块大小
        self._draft_block_ids_buf = torch.empty(  # 分配块ID缓冲区
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_positions_buf = torch.empty(  # 分配块位置缓冲区
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        self._draft_block_tokens_buf = torch.empty(  # 分配块token缓冲区
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_end_buf = torch.empty(  # 分配块结束缓冲区
            (new_cap,), dtype=torch.int32, device=device
        )
        self._draft_seq_lens_cpu_buf = torch.empty(  # 分配CPU序列长度缓冲区
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker.
        # 将未实现的方法委托给目标Worker。
        return getattr(self.target_worker, name)  # 从目标Worker获取属性

    def clear_cache_pool(self):
        # The target worker owns the shared KV allocator/cache. For the compact
        # sliding-window path, the draft req->token view is rebuilt from committed
        # target state before each draft forward, so there is nothing persistent
        # to flush here.
        # 目标Worker拥有共享KV分配器/缓存。对于紧凑滑动窗口路径，
        # 草稿req->token视图在每次草稿前向之前从已提交的目标状态重建，
        # 因此这里没有需要刷新的持久数据。
        pass  # 不做任何操作

    def _gather_req_to_token_masked(
        self,
        *,
        req_to_token: torch.Tensor,  # 请求到token映射表
        req_pool_indices: torch.Tensor,  # 请求池索引
        pos2d: torch.Tensor,  # 二维位置
        mask: torch.Tensor,  # 掩码
        context: str,  # 上下文描述（用于错误信息）
    ) -> torch.Tensor:
        # 根据掩码从req_to_token表中收集token位置
        if pos2d.ndim != 2:  # 位置必须是2D
            raise RuntimeError(
                f"{context} expected 2D positions, got shape={tuple(pos2d.shape)}."
            )
        if mask.shape != pos2d.shape:  # 掩码和位置形状必须一致
            raise RuntimeError(
                f"{context} mask/position shape mismatch: {tuple(mask.shape)} vs {tuple(pos2d.shape)}."
            )

        if req_pool_indices.dtype != torch.int64:  # 确保索引类型为int64
            req_pool_indices = req_pool_indices.to(torch.int64)
        if mask.dtype != torch.bool:  # 确保掩码类型为bool
            mask = mask.to(torch.bool)

        table_width = int(req_to_token.shape[1])  # 表宽度
        if table_width <= 0:  # 表为空
            if bool(mask.any().item()):  # 但掩码非空
                raise RuntimeError(
                    f"{context} req_to_token table is empty but gather mask is non-empty."
                )
            return torch.empty((0,), dtype=torch.int64, device=self.device)  # 返回空张量

        # Only the masked-off rectangular padding can be out of range in the normal
        # ragged-batch case. Replace those don't-care columns with a valid in-range
        # position before the gather so the kernel only sees real positions.
        # 在正常的ragged-batch情况下，只有被掩码遮蔽的矩形填充可能超出范围。
        # 在收集之前将这些无关列替换为有效范围内的位置，使内核只看到真实位置。
        safe_pos2d = pos2d.masked_fill(~mask, 0)  # 将无效位置替换为0
        return req_to_token[req_pool_indices[:, None], safe_pos2d][mask].to(torch.int64)  # 收集并返回

    def _gather_req_to_token_segments(
        self,
        *,
        req_to_token: torch.Tensor,  # 请求到token映射表
        req_pool_indices: torch.Tensor,  # 请求池索引
        start: torch.Tensor | None,  # 起始位置
        lengths: torch.Tensor,  # 段长度
    ) -> torch.Tensor:
        # 从req_to_token表中收集指定段的token位置
        lengths = lengths.to(torch.int64)  # 转为int64
        if lengths.numel() == 0:  # 无段
            return torch.empty((0,), dtype=torch.int64, device=self.device)  # 返回空
        max_len = int(lengths.max().item())  # 最大段长度
        if max_len <= 0:  # 最大长度为0
            return torch.empty((0,), dtype=torch.int64, device=self.device)  # 返回空

        if req_pool_indices.dtype != torch.int64:  # 确保索引类型
            req_pool_indices = req_pool_indices.to(torch.int64)
        offsets = torch.arange(  # 偏移量
            max_len, device=self.device, dtype=torch.int64
        ).unsqueeze(0)
        if start is None:  # 无起始位置
            pos2d = offsets.expand(req_pool_indices.shape[0], -1)  # 从0开始
        else:  # 有起始位置
            pos2d = start.to(torch.int64).unsqueeze(1) + offsets  # 加上起始偏移
        mask = offsets < lengths.unsqueeze(1)  # 有效位置掩码
        return self._gather_req_to_token_masked(  # 使用掩码收集
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            pos2d=pos2d,
            mask=mask,
            context="DFLASH req_to_token segment gather",
        )

    def _compute_compact_draft_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        # 计算紧凑草稿序列长度（受窗口大小限制）
        assert self.draft_window_size is not None  # 必须设置了窗口大小
        visible_lens = torch.clamp(  # 可见长度不超过窗口大小
            seq_lens.to(dtype=torch.int32, device=self.device),
            max=int(self.draft_window_size),
        )
        if self.page_size <= 1:  # 非分页模式
            return visible_lens  # 直接返回

        # Paged FA backends derive the page table from local token positions, so the
        # compact suffix must start on a page boundary. Keep up to page_size - 1 extra
        # tokens on the left to preserve valid local page structure.
        # 分页FA后端从本地token位置推导页表，因此紧凑后缀必须从页边界开始。
        # 保留最多page_size-1个额外token在左侧以保持有效的本地页结构。
        seq_lens_i64 = seq_lens.to(torch.int64)  # 转为int64
        visible_lens_i64 = visible_lens.to(torch.int64)  # 转为int64
        visible_start = seq_lens_i64 - visible_lens_i64  # 可见起始位置
        aligned_start = visible_start - torch.remainder(visible_start, self.page_size)  # 对齐到页边界
        return (seq_lens_i64 - aligned_start).to(torch.int32)  # 返回对齐后的可见长度

    def _resolve_mask_token_id(
        self, *, mask_token: str, mask_token_id: Optional[int] = None
    ) -> int:
        # 解析掩码token ID
        if not isinstance(mask_token, str) or not mask_token:  # 必须是非空字符串
            raise ValueError(
                f"DFLASH mask_token must be a non-empty string, got {mask_token!r}."
            )

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)  # 词表大小
        if mask_token_id is not None:  # 已指定ID
            resolved_id = int(mask_token_id)  # 转为整数
            if resolved_id >= vocab_size:  # 超出词表范围
                raise ValueError(
                    "DFLASH mask_token_id is outside the target vocab size. "
                    f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                    f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                    "SGLang does not support resizing target embeddings for DFLASH yet."
                )

            tokenizer = getattr(self.target_worker, "tokenizer", None)  # 获取tokenizer
            if tokenizer is not None:  # tokenizer可用
                token_id_from_vocab = tokenizer.get_vocab().get(mask_token, None)  # 从词表获取ID
                if (  # ID不一致
                    token_id_from_vocab is not None
                    and int(token_id_from_vocab) != resolved_id
                ):
                    raise ValueError(
                        "DFLASH config mismatch: dflash_config.mask_token_id conflicts with tokenizer vocab id "
                        f"for dflash_config.mask_token. mask_token={mask_token!r}, "
                        f"mask_token_id={resolved_id}, tokenizer_vocab_id={int(token_id_from_vocab)}."
                    )
            return resolved_id  # 返回解析的ID

        tokenizer = getattr(self.target_worker, "tokenizer", None)  # 获取tokenizer
        if tokenizer is None:  # tokenizer不可用
            raise RuntimeError(
                "DFLASH requires tokenizer initialization when dflash_config.mask_token_id is not set "
                "(skip_tokenizer_init is not supported in this mode)."
            )

        resolved_id = None  # 解析结果
        if getattr(tokenizer, "mask_token", None) == mask_token:  # tokenizer有mask_token属性且匹配
            resolved_id = getattr(tokenizer, "mask_token_id", None)  # 使用tokenizer的mask_token_id

        if resolved_id is None:  # 仍然未解析
            # Prefer checking the explicit vocab mapping first.
            # 优先检查显式词表映射。
            vocab = tokenizer.get_vocab()  # 获取词表
            resolved_id = vocab.get(mask_token, None)  # 从词表查找

        if resolved_id is None:  # 词表中也没有
            # Mirror the reference DFlash HF demo by adding the mask token to the tokenizer.
            # 参照DFlash HF演示，将掩码token添加到tokenizer。
            # This is safe only when the resulting id stays within the target model vocab size.
            # 仅当结果ID在目标模型词表大小范围内时安全。
            added = tokenizer.add_special_tokens({"mask_token": mask_token})  # 添加特殊token
            resolved_id = getattr(tokenizer, "mask_token_id", None)  # 获取ID
            if resolved_id is None:  # 仍未获取到
                resolved_id = tokenizer.convert_tokens_to_ids(mask_token)  # 转换获取

            if added and self.tp_rank == 0:  # 如果添加了新token且是主rank
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, tokenizer_len=%s, model_vocab_size=%s",
                    mask_token,
                    resolved_id,
                    len(tokenizer),
                    vocab_size,
                )

        if resolved_id is None or int(resolved_id) < 0:  # 解析失败
            raise ValueError(
                "DFLASH requires resolving a mask token id, but it could not be resolved. "
                f"mask_token={mask_token!r}."
            )

        if resolved_id >= vocab_size:  # 超出词表范围
            raise ValueError(
                "DFLASH mask_token_id is outside the target vocab size. "
                f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                "SGLang does not support resizing target embeddings for DFLASH yet."
            )

        return int(resolved_id)  # 返回解析的ID

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ):
        # 准备投机解码：将目标隐藏状态追加到草稿KV缓存，然后运行草稿模型生成候选块
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():  # 非解码模式
            return  # 直接返回

        if batch.has_grammar:  # 有语法约束
            raise RuntimeError(
                "Invariant broken: DFLASH batch has grammar constraints, but scheduler should have rejected this request."
            )
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:  # 非贪心采样
            if (  # 采样验证不可用且未警告过
                not is_dflash_sampling_verify_available()
                and not self._warned_sampling_fallback
                and self.tp_rank == 0
            ):
                logger.warning(
                    "DFLASH non-greedy verification is unavailable on this build/device; "
                    "falling back to greedy argmax verification."
                )
                self._warned_sampling_fallback = True  # 标记已警告

        bs = batch.batch_size()  # 批次大小

        # --- 1) Append any newly committed tokens into the draft KV cache.
        # --- 1) 将任何新提交的token追加到草稿KV缓存。
        self._append_target_hidden_to_draft_kv(batch, draft_input)

        target_model = self.target_worker.model_runner.model  # 目标模型
        embed_module = target_model.get_input_embeddings()  # 嵌入层
        lm_head = getattr(target_model, "lm_head", None)  # 语言模型头
        if (  # lm_head必须有权重和分片索引
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        # --- 2) Draft a non-causal block with the draft model.
        # --- 2) 使用草稿模型生成非因果块。
        self._ensure_draft_block_buffers(bs)  # 确保缓冲区足够
        assert self._draft_block_ids_buf is not None  # 断言缓冲区已分配
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]  # 当前批次的块ID
        block_ids.fill_(int(self._mask_token_id))  # 用掩码token填充
        block_ids[:, 0].copy_(draft_input.bonus_tokens.to(torch.long))  # 第一个位置用bonus token

        noise_embedding = embed_module(block_ids)  # 通过嵌入层获取噪声嵌入
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])  # 重塑为2D

        # For spec-v1, the draft KV cache is always materialized before drafting the
        # next block. `target_prefix_lens` stay absolute for RoPE; `draft_prefix_lens`
        # are the logical resident lengths in the draft-local cache.
        # 对于spec-v1，草稿KV缓存在草拟下一个块之前总是已物化。
        # target_prefix_lens保持绝对值用于RoPE；draft_prefix_lens是草稿本地缓存中的逻辑驻留长度。
        target_prefix_lens = batch.seq_lens  # int32, device 目标前缀长度
        draft_prefix_lens = draft_input.draft_seq_lens  # 草稿前缀长度
        if draft_prefix_lens.dtype != torch.int32:  # 确保类型正确
            draft_prefix_lens = draft_prefix_lens.to(torch.int32)
        if draft_prefix_lens.device != self.device:  # 确保设备正确
            draft_prefix_lens = draft_prefix_lens.to(self.device, non_blocking=True)

        positions_2d = self._draft_block_positions_buf[:bs]  # 二维位置缓冲区
        torch.add(
            target_prefix_lens.unsqueeze(1), self._block_pos_offsets, out=positions_2d  # 计算块内位置
        )
        positions = positions_2d.reshape(-1)  # 展平

        block_start = draft_prefix_lens  # 块起始位置
        block_end = self._draft_block_end_buf[:bs]  # 块结束位置缓冲区
        torch.add(block_start, int(self.block_size), out=block_end)  # 计算结束位置

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]  # CPU序列长度缓冲区
        seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))  # 复制到CPU
        allocator = self.draft_model_runner.token_to_kv_pool_allocator  # KV池分配器
        token_to_kv_pool_state_backup = allocator.backup_state()  # 备份分配器状态
        try:
            if self.page_size == 1:  # 非分页模式
                block_cache_loc = allocator.alloc(bs * self.block_size)  # 分配缓存位置
            else:  # 分页模式
                block_end_cpu = seq_lens_cpu + int(self.block_size)  # CPU上的结束位置
                last_loc = get_last_loc(  # 获取最后位置
                    self.draft_model_runner.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    block_start,
                )
                block_cache_loc = allocator.alloc_extend(  # 分配扩展缓存位置
                    block_start,
                    seq_lens_cpu,
                    block_end,
                    block_end_cpu,
                    last_loc,
                    bs * self.block_size,
                )
            if block_cache_loc is None:  # 分配失败（OOM）
                raise RuntimeError(
                    f"DFLASH draft OOM when allocating {bs * self.block_size} block tokens."
                )

            assign_req_to_token_pool_func(  # 更新req_to_token映射
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                block_start,
                block_end,
                block_cache_loc,
                bs,
            )

            # Use TARGET_VERIFY mode (cuda-graphable) to run a fixed-size draft block.
            # 使用TARGET_VERIFY模式（可CUDA图捕获）运行固定大小的草稿块。
            # In this mode, `seq_lens` stores the prefix lengths; attention backends
            # derive kv_len by adding `draft_token_num`.
            # 在此模式下，seq_lens存储前缀长度；注意力后端通过加上draft_token_num推导kv_len。
            draft_spec_info = self._draft_block_spec_info  # 草稿验证信息
            seq_lens = draft_prefix_lens  # 序列长度
            seq_lens_sum = int(draft_prefix_lens.sum().item())  # 序列长度总和
            forward_batch = ForwardBatch(  # 创建前向批次
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                input_ids=block_ids.flatten(),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=block_cache_loc,
                seq_lens_sum=seq_lens_sum,
                seq_lens_cpu=seq_lens_cpu,
                positions=positions,
                input_embeds=input_embeds,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=draft_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            with torch.inference_mode():  # 推理模式
                draft_logits_output = self.draft_model_runner.forward(
                    forward_batch
                ).logits_output  # 运行草稿模型前向
        finally:
            # Drop the speculative block from the shared allocator (EAGLE3-style).
            # 从共享分配器中丢弃投机块（EAGLE3风格）。
            allocator.restore_state(token_to_kv_pool_state_backup)  # 恢复分配器状态

        draft_hidden = draft_logits_output.hidden_states  # 草稿隐藏状态
        if draft_hidden is None:  # 隐藏状态为空
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, self.block_size, -1)  # 重塑形状
        draft_next = self._greedy_sample_from_vocab_parallel_head(  # 从LM头贪心采样
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, self.block_size - 1)  # 重塑形状
        draft_tokens = self._draft_block_tokens_buf[:bs]  # 草稿token缓冲区
        draft_tokens[:, 0].copy_(block_ids[:, 0])  # 第一个token是bonus token
        draft_tokens[:, 1:].copy_(draft_next)  # 后续token是草稿模型采样结果
        positions = positions_2d.reshape(-1)  # 展平位置

        verify_input = DFlashVerifyInput(  # 创建验证输入
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=self.block_size,
        )
        _, build_custom_mask = resolve_dflash_verify_mask_policy(  # 解析掩码策略
            self.model_runner.attn_backend
        )
        verify_input.prepare_for_verify(  # 准备验证
            batch,
            self.page_size,
            build_custom_mask=build_custom_mask,
        )

        batch.forward_mode = (  # 更新前向模式
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input  # 设置验证信息
        batch.return_hidden_states = False  # 不返回隐藏状态

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,  # 隐藏状态
        lm_head,  # 语言模型头
        chunk_size: int = 256,  # 分块大小
    ) -> torch.Tensor:
        """Greedy argmax over the target LM head in a TP-safe way.
        # 以TP安全的方式在目标LM头上进行贪心argmax。

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        # 无法高效地为大词表物化完整logits，且TP>1时每个rank只拥有LM头权重的一个分片。
        # 此方法计算每个rank的最大值，跨TP rank收集候选，并选择全局最大值。
        """

        if hidden_states.numel() == 0:  # 空输入
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)  # 返回空

        tp_group = get_tp_group()  # 获取TP组
        tp_size = int(tp_group.world_size)  # TP大小

        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):  # 检查属性
            raise RuntimeError(
                "DFLASH greedy sampling requires a vocab-parallel head with `weight` and `shard_indices`."
            )

        shard = lm_head.shard_indices  # 分片索引
        weight = lm_head.weight  # [local_vocab_padded, hidden] 权重
        weight_dtype = weight.dtype  # 权重数据类型

        # Valid ranges in the local shard (excluding padding):
        # 本地分片中的有效范围（排除填充）：
        #   base vocab:  [0, num_org)
        #   added vocab: [num_org_padded, num_org_padded + num_added)
        num_org = int(shard.num_org_elements)  # 原始词表元素数
        num_org_padded = int(shard.num_org_elements_padded)  # 填充后的原始词表元素数
        num_added = int(shard.num_added_elements)  # 新增词表元素数
        org_vocab_start = int(shard.org_vocab_start_index)  # 原始词表起始索引
        added_vocab_start = int(shard.added_vocab_start_index)  # 新增词表起始索引

        num_tokens = int(hidden_states.shape[0])  # token数
        out_tokens = torch.empty(  # 输出token缓冲区
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            # 将隐藏状态转换为与权重相同的数据类型
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        # Fast path (common): single-rank greedy sampling over the base vocab shard.
        # 快速路径（常见）：单rank基础词表分片的贪心采样。
        # Avoids extra max/id bookkeeping that is only needed for TP sync or added vocab.
        # 避免仅在TP同步或新增词表时需要的额外max/id记账。
        if tp_size == 1 and num_added == 0:  # 单rank且无新增词表
            for start in range(0, num_tokens, int(chunk_size)):  # 分块处理
                end = min(num_tokens, start + int(chunk_size))  # 块结束
                hs = _cast_hs(hidden_states[start:end])  # 转换类型
                if num_org > 0:  # 有原始词表
                    base_logits = torch.matmul(hs, weight[:num_org].T)  # 计算基础logits
                    out_tokens[start:end] = (
                        torch.argmax(base_logits, dim=-1).to(torch.long)
                        + org_vocab_start  # 加上起始索引
                    )
                else:  # 无原始词表
                    out_tokens[start:end] = 0  # 填0
            return out_tokens  # 返回结果

        for start in range(0, num_tokens, int(chunk_size)):  # 分块处理
            end = min(num_tokens, start + int(chunk_size))  # 块结束
            hs = _cast_hs(hidden_states[start:end])  # 转换类型
            chunk_len = int(hs.shape[0])  # 块长度

            # Base vocab logits.
            # 基础词表logits。
            if num_org > 0:  # 有原始词表
                base_logits = torch.matmul(hs, weight[:num_org].T)  # 计算基础logits
                local_max, local_arg = torch.max(base_logits, dim=-1)  # 局部最大值和索引
            else:  # 无原始词表
                local_max = torch.full(  # 填充极小值
                    (chunk_len,),
                    torch.finfo(weight_dtype).min,
                    dtype=weight_dtype,
                    device=hs.device,
                )
                local_arg = torch.zeros(  # 填充0
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )

            # Added vocab logits (e.g., LoRA-added embeddings), if present.
            # 新增词表logits（如LoRA新增嵌入），如果存在。
            if num_added > 0:  # 有新增词表
                added_slice_start = num_org_padded  # 新增词表起始
                added_slice_end = num_org_padded + num_added  # 新增词表结束
                added_logits = torch.matmul(  # 计算新增logits
                    hs, weight[added_slice_start:added_slice_end].T
                )
                added_max, added_arg = torch.max(added_logits, dim=-1)  # 新增最大值和索引
                use_added = added_max > local_max  # 是否使用新增
                local_max = torch.where(use_added, added_max, local_max)  # 选择较大值
                # For base/added conversion below, keep local_arg expressed in the full local
                # weight index space (base + padding + added), matching `lm_head.weight`.
                # 为下面的base/added转换，保持local_arg在完整本地权重索引空间中
                # （base + padding + added），与lm_head.weight匹配。
                local_arg = torch.where(
                    use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg  # 新增索引加上填充偏移
                )

            # Convert local argmax indices to global token ids.
            # 将本地argmax索引转换为全局token ID。
            if num_added == 0:  # 无新增词表
                local_arg.add_(org_vocab_start)  # 加上起始索引
                global_ids = local_arg  # 即为全局ID
            else:  # 有新增词表
                global_ids = torch.empty(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )
                is_base = local_arg < num_org  # 是否为基础词表
                global_ids[is_base] = org_vocab_start + local_arg[is_base]  # 基础词表全局ID
                global_ids[~is_base] = added_vocab_start + (  # 新增词表全局ID
                    local_arg[~is_base] - num_org_padded
                )

            if tp_size == 1:  # 单rank无需同步
                out_tokens[start:end] = global_ids.to(torch.long)  # 直接使用
                continue  # 继续下一个块

            # Gather per-rank maxima and associated global ids, then select the global max.
            # 收集每个rank的最大值和关联的全局ID，然后选择全局最大值。
            needed = tp_size * chunk_len  # 需要的总大小
            chunk_cap = int(chunk_size)  # 块容量
            if (  # 需要重新分配缓冲区
                self._draft_greedy_gather_cap < needed
                or self._draft_greedy_gathered_max_buf is None
                or self._draft_greedy_gathered_ids_buf is None
                or self._draft_greedy_gathered_max_buf.dtype != local_max.dtype
                or self._draft_greedy_gathered_max_buf.device != hs.device
            ):
                # Allocate enough space for the max chunk size to avoid reallocations.
                # 为最大块大小分配足够空间以避免重新分配。
                cap = tp_size * chunk_cap  # 容量
                self._draft_greedy_gathered_max_buf = torch.empty(  # 分配最大值缓冲区
                    (cap,), dtype=local_max.dtype, device=hs.device
                )
                self._draft_greedy_gathered_ids_buf = torch.empty(  # 分配ID缓冲区
                    (cap,), dtype=global_ids.dtype, device=hs.device
                )
                self._draft_greedy_gather_cap = cap  # 更新容量

            if (  # 需要重新分配索引缓冲区
                self._draft_greedy_index_cap < chunk_len
                or self._draft_greedy_best_rank_buf is None
                or self._draft_greedy_rank_index_buf is None
                or self._draft_greedy_selected_ids_buf is None
                or self._draft_greedy_best_rank_buf.device != hs.device
                or self._draft_greedy_selected_ids_buf.device != hs.device
            ):
                self._draft_greedy_best_rank_buf = torch.empty(  # 分配最佳rank缓冲区
                    (chunk_cap,), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_rank_index_buf = torch.empty(  # 分配rank索引缓冲区
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_selected_ids_buf = torch.empty(  # 分配选中ID缓冲区
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_index_cap = chunk_cap  # 更新容量

            gathered_max = self._draft_greedy_gathered_max_buf[:needed]  # 切片最大值缓冲区
            gathered_ids = self._draft_greedy_gathered_ids_buf[:needed]  # 切片ID缓冲区

            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())  # 收集所有rank的最大值
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())  # 收集所有rank的ID
            gathered_max = gathered_max.view(tp_size, chunk_len)  # 重塑形状
            gathered_ids = gathered_ids.view(tp_size, chunk_len)  # 重塑形状

            best_rank = self._draft_greedy_best_rank_buf[:chunk_len]  # 最佳rank缓冲区
            torch.argmax(gathered_max, dim=0, out=best_rank)  # 找到每个token的最佳rank

            rank_index = self._draft_greedy_rank_index_buf[:, :chunk_len]  # rank索引
            rank_index[0].copy_(best_rank)  # 复制最佳rank
            selected_ids = self._draft_greedy_selected_ids_buf[:, :chunk_len]  # 选中ID
            torch.gather(gathered_ids, 0, rank_index, out=selected_ids)  # 收集选中ID
            out_tokens[start:end].copy_(selected_ids.view(-1))  # 复制到输出

        return out_tokens  # 返回结果

    def _append_target_hidden_to_draft_kv(
        self,
        batch: ScheduleBatch,  # 调度批次
        draft_input: DFlashDraftInput,  # 草稿输入
    ) -> None:
        """Materialize the target hidden-state features into the draft KV cache.
        # 将目标隐藏状态特征物化到草稿KV缓存中。

        This must be run before exposing new tokens to radix cache (prefix hits), otherwise
        another request could reuse target KV indices without having draft KV values.
        # 这必须在将新token暴露给radix缓存（前缀命中）之前运行，否则另一个请求
        # 可能复用目标KV索引而没有草稿KV值。
        """

        bs = batch.batch_size()  # 批次大小
        device = self.model_runner.device  # 设备

        if draft_input.target_hidden is None:  # 缺少目标隐藏状态
            raise RuntimeError(
                "DFLASH draft state missing target_hidden context features."
            )
        if draft_input.ctx_lens.numel() != bs:  # ctx_lens长度不匹配
            raise RuntimeError(
                f"DFLASH ctx_lens length mismatch: got {draft_input.ctx_lens.numel()} for bs={bs}."
            )
        if draft_input.draft_seq_lens.numel() != bs:  # draft_seq_lens长度不匹配
            raise RuntimeError(
                f"DFLASH draft_seq_lens length mismatch: got {draft_input.draft_seq_lens.numel()} for bs={bs}."
            )

        total_ctx = int(draft_input.target_hidden.shape[0])  # 总上下文token数
        if total_ctx <= 0:  # 无上下文
            draft_input.ctx_lens = torch.zeros_like(draft_input.ctx_lens)  # 清零
            draft_input.target_hidden = draft_input.target_hidden[:0]  # 清空
            return  # 返回

        target_req_to_token = batch.req_to_token_pool.req_to_token  # 目标req_to_token表
        draft_req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token  # 草稿req_to_token表

        req_pool_indices = batch.req_pool_indices  # 请求池索引
        if req_pool_indices.dtype != torch.int64:  # 确保类型正确
            req_pool_indices = req_pool_indices.to(torch.int64)

        ctx_lens = draft_input.ctx_lens  # 上下文长度
        if ctx_lens.dtype != torch.int32:  # 确保类型正确
            ctx_lens = ctx_lens.to(torch.int32)
        if ctx_lens.device != device:  # 确保设备正确
            ctx_lens = ctx_lens.to(device, non_blocking=True)
        ctx_start = batch.seq_lens.to(torch.int64) - ctx_lens.to(torch.int64)  # 上下文起始位置

        if bs == 1:  # 单请求快速路径
            # Fast path for single request.
            max_ctx = int(total_ctx)  # 最大上下文长度
            if max_ctx <= self._block_pos_offsets.numel():  # 不超过块大小
                r = self._block_pos_offsets[:max_ctx]  # 使用预分配偏移
            else:  # 超过块大小
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)  # 创建新偏移
            pos2d = ctx_start[:, None] + r[None, :]  # [1, ctx] 二维位置
            cache2d = target_req_to_token[req_pool_indices[:, None], pos2d]  # [1, ctx] 缓存位置
            ctx_cache_loc = cache2d.reshape(-1).to(torch.int64)  # [ctx] 展平
            ctx_positions = pos2d.reshape(-1)  # [ctx] 展平位置
        else:  # 多请求路径
            # In decode mode, ctx_lens <= block_size so we can skip the .item() sync.
            # 解码模式下，ctx_lens <= block_size，可以跳过.item()同步。
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:  # 扩展模式
                max_ctx = int(ctx_lens.max().item())  # 需要D2H同步
            else:  # 解码模式
                max_ctx = int(self.block_size)  # 直接使用块大小
            if max_ctx <= 0:  # 无效值
                raise RuntimeError(f"DFLASH invalid max_ctx={max_ctx} for KV append.")

            if max_ctx <= self._block_pos_offsets.numel():  # 不超过块大小
                r = self._block_pos_offsets[:max_ctx]  # 使用预分配偏移
            else:  # 超过块大小
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)  # 创建新偏移
            r = r[None, :]  # [1, max_ctx] 增加维度
            pos2d = ctx_start[:, None] + r  # [bs, max_ctx] 二维位置
            mask = r < ctx_lens[:, None]  # 有效位置掩码

            # Batched gather of cache locations and positions.
            # 批量收集缓存位置和位置。
            ctx_cache_loc = self._gather_req_to_token_masked(  # 使用掩码收集
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                pos2d=pos2d,
                mask=mask,
                context="DFLASH target hidden KV append",
            )  # [sum(ctx_lens)]
            ctx_positions = pos2d[mask]  # [sum(ctx_lens)] 有效位置

        with torch.inference_mode():  # 推理模式
            ctx_hidden = self.draft_model.project_target_hidden(
                draft_input.target_hidden
            )  # [sum(ctx), hidden] 投影目标隐藏状态
            if ctx_hidden.shape[0] != ctx_cache_loc.numel():  # 维度不匹配
                raise RuntimeError(
                    f"DFLASH ctx_hidden/cache_loc mismatch: {ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                )

            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:  # 使用融合KV
                try:
                    self._append_target_hidden_fused(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
                except Exception as e:  # 融合路径失败
                    logger.warning(
                        "DFLASH fused KV append failed; falling back to sequential path: %s",
                        e,
                    )
                    self._use_fused_kv_materialize = False  # 禁用融合
                    self._fused_kv_helper = None  # 清除助手
                    self._append_target_hidden_sequential(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
            else:  # 使用顺序路径
                self._append_target_hidden_sequential(
                    ctx_hidden, ctx_positions, ctx_cache_loc
                )

        if self.use_compact_draft_cache:  # 紧凑草稿缓存模式
            new_draft_seq_lens = self._compute_compact_draft_seq_lens(batch.seq_lens)  # 计算新序列长度
            suffix_start = batch.seq_lens.to(torch.int64) - new_draft_seq_lens.to(
                torch.int64
            )  # 后缀起始位置
            suffix_cache_loc = self._gather_req_to_token_segments(  # 收集后缀缓存位置
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                start=suffix_start,
                lengths=new_draft_seq_lens,
            )
            assign_req_to_token_pool_func(  # 更新草稿req_to_token映射
                batch.req_pool_indices,
                draft_req_to_token,
                torch.zeros_like(new_draft_seq_lens),
                new_draft_seq_lens,
                suffix_cache_loc,
                bs,
            )
            draft_input.draft_seq_lens = new_draft_seq_lens  # 更新草稿序列长度
        else:  # 非紧凑模式
            draft_input.draft_seq_lens = batch.seq_lens.to(dtype=torch.int32)  # 使用目标序列长度
        draft_input.ctx_lens = torch.zeros_like(ctx_lens)  # 清零上下文长度
        draft_input.target_hidden = draft_input.target_hidden[:0]  # 清空目标隐藏状态

    def _append_target_hidden_sequential(
        self,
        ctx_hidden: torch.Tensor,  # 上下文隐藏状态
        ctx_positions: torch.Tensor,  # 上下文位置
        ctx_cache_loc: torch.Tensor,  # 上下文缓存位置
    ) -> None:
        # 顺序地将目标隐藏状态追加到草稿KV缓存（逐层处理）
        for layer in self.draft_model.layers:  # 遍历每一层
            attn = layer.self_attn  # 获取注意力模块
            if _is_npu:  # NPU设备
                _, k, v = attn.forward_prepare_npu(ctx_positions, ctx_hidden)  # NPU专用前向
            else:  # CUDA设备
                k, v = attn.kv_proj_only(ctx_hidden)  # 仅KV投影
                k = attn.apply_k_norm(k)  # 应用K归一化
                k = attn.apply_k_rope(ctx_positions, k)  # 应用RoPE
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)  # 重塑K形状
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)  # 重塑V形状
            self.draft_model_runner.token_to_kv_pool.set_kv_buffer(  # 写入KV缓存
                attn.attn,
                ctx_cache_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )

    def _append_target_hidden_fused(
        self,
        ctx_hidden: torch.Tensor,  # 上下文隐藏状态
        ctx_positions: torch.Tensor,  # 上下文位置
        ctx_cache_loc: torch.Tensor,  # 上下文缓存位置
    ) -> None:
        """Fused KV materialization using batched projection + Triton kernel."""
        # 使用批量投影+Triton内核的融合KV物化。
        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool  # KV缓存池
        layers = self.draft_model.layers  # 模型层

        def _write_layer_kv(
            layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor
        ) -> None:
            # 将指定层的KV写入缓存
            attn = layers[layer_idx].self_attn.attn  # 获取注意力
            token_to_kv_pool.set_kv_buffer(
                attn,
                ctx_cache_loc,
                cache_k,
                cache_v,
                attn.k_scale,
                attn.v_scale,
            )

        self._fused_kv_helper.materialize(  # 融合物化
            ctx_hidden=ctx_hidden,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

    def _update_target_mamba_state_after_verify(
        self,
        *,
        batch: ScheduleBatch,  # 调度批次
        seq_lens_pre_verify: torch.Tensor,  # 验证前序列长度
        commit_lens: torch.Tensor,  # 提交长度
    ) -> None:
        """Commit Mamba intermediate states for accepted verify steps.
        # 为已接受的验证步骤提交Mamba中间状态。

        During TARGET_VERIFY, Mamba kernels run with `disable_state_update=True` and
        cache per-step intermediate states. After acceptance, we need to commit the
        state corresponding to each request's last accepted step.
        # 在TARGET_VERIFY期间，Mamba内核以disable_state_update=True运行
        # 并缓存每步的中间状态。接受后，需要提交每个请求最后接受步骤对应的状态。
        """
        attn_backend = self.target_worker.model_runner.attn_backend  # 注意力后端
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):  # 不支持Mamba更新
            return  # 直接返回

        last_correct_step_indices = commit_lens.to(torch.int64) - 1  # 最后正确步骤索引
        mamba_steps_to_track = None  # 需要跟踪的Mamba步骤

        if batch.mamba_track_indices is not None:  # 有Mamba跟踪索引
            mamba_track_interval = self.server_args.mamba_track_interval  # 跟踪间隔
            to_track_mask = (  # 需要跟踪的掩码
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (  # 跟踪点
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)  # 跟踪步骤索引
            can_track_mask = to_track_mask & (  # 可以跟踪的掩码
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(  # 需要跟踪的步骤
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),  # -1表示不跟踪
            )

        attn_backend.update_mamba_state_after_mtp_verify(  # 更新Mamba状态
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def forward_batch_generation(
        self, batch: ScheduleBatch, **kwargs
    ) -> GenerationBatchResult:
        # DFlash前向批次生成：处理预填充和解码两个阶段
        if getattr(batch, "return_logprob", False):  # 不支持logprob
            raise RuntimeError(
                "Invariant broken: DFLASH batch requested return_logprob, but scheduler should have rejected this request."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:  # 预填充/扩展模式
            batch.capture_hidden_mode = CaptureHiddenMode.FULL  # 捕获完整隐藏状态
            batch_result = self.target_worker.forward_batch_generation(batch, **kwargs)  # 运行目标模型
            logits_output, next_token_ids = (  # 获取输出
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:  # 缺少隐藏状态
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if batch.extend_lens is None or batch.prefix_lens is None:  # 缺少扩展/前缀长度
                raise RuntimeError(
                    "DFLASH expected extend_lens / prefix_lens to be populated in extend mode, but got None."
                )

            # Materialize the prompt tokens into the draft KV cache immediately. This is required
            # for radix cache support, since the scheduler may update radix after prefill returns.
            # 立即将提示token物化到草稿KV缓存。这是radix缓存支持所必需的，
            # 因为调度器可能在预填充返回后更新radix。
            device = next_token_ids.device  # 设备

            def _to_int32_device_tensor(x, *, device=device):
                # 将输入转为设备上的int32张量
                if isinstance(x, torch.Tensor):  # 已经是张量
                    if x.device != device:  # 设备不匹配
                        x = x.to(device, non_blocking=True)  # 转换设备
                    return x if x.dtype == torch.int32 else x.to(torch.int32)  # 转换类型
                return torch.tensor(x, dtype=torch.int32, device=device)  # 创建新张量

            extend_seq_lens = _to_int32_device_tensor(batch.extend_lens)  # 扩展序列长度
            draft_input = DFlashDraftInput(  # 创建草稿输入
                bonus_tokens=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens=extend_seq_lens,
                draft_seq_lens=(
                    torch.zeros_like(extend_seq_lens)
                    if self.use_compact_draft_cache  # 紧凑模式从0开始
                    else _to_int32_device_tensor(batch.prefix_lens)  # 否则使用前缀长度
                ),
            )
            self._append_target_hidden_to_draft_kv(batch, draft_input)  # 追加到草稿KV
            batch.spec_info = draft_input  # 设置草稿输入

            return GenerationBatchResult(  # 返回结果
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_correct_drafts=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        # Decode / target-verify stage.
        # 解码/目标验证阶段。
        draft_input = batch.spec_info  # 获取草稿输入
        if not isinstance(draft_input, DFlashDraftInput):  # 类型检查
            raise RuntimeError(
                "DFLASH decode requires DFlashDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        self._prepare_for_speculative_decoding(batch, draft_input)  # 准备投机解码

        assert batch.forward_mode.is_target_verify()  # 断言是目标验证模式
        verify_input = batch.spec_info  # 获取验证输入
        assert isinstance(verify_input, DFlashVerifyInput)  # 类型检查
        need_mamba_verify_commit = hasattr(  # 是否需要Mamba验证提交
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (  # 验证前序列长度
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )

        batch_result = self.target_worker.forward_batch_generation(  # 运行目标模型验证
            batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (  # 获取输出
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        (
            new_bonus_tokens,
            commit_lens,
            next_target_hidden,
            num_correct_drafts_per_req_cpu,
        ) = verify_input.verify(  # 验证草稿
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
        )
        if need_mamba_verify_commit:  # 需要Mamba提交
            assert seq_lens_pre_verify is not None  # 断言序列长度存在
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        # Update draft state for the next iteration. Also materialize the committed verify tokens
        # into the draft KV cache immediately so radix cache entries are safe to reuse.
        # 更新下一步迭代的草稿状态。同时立即将已提交的验证token物化到草稿KV缓存，
        # 以便radix缓存条目可以安全复用。
        draft_input.bonus_tokens = new_bonus_tokens  # 更新奖励token
        draft_input.target_hidden = next_target_hidden  # 更新目标隐藏状态
        draft_input.ctx_lens = commit_lens  # 更新上下文长度
        self._append_target_hidden_to_draft_kv(batch, draft_input)  # 追加到草稿KV
        batch.spec_info = draft_input  # 更新草稿输入
        batch.forward_mode = ForwardMode.DECODE  # 恢复解码模式

        num_correct_drafts = sum(num_correct_drafts_per_req_cpu)  # 总正确草稿数
        if not self._logged_first_verify and self.tp_rank == 0:  # 首次验证
            logger.info(
                "DFLASH verify completed. num_correct_drafts_per_req=%s",
                num_correct_drafts_per_req_cpu,
            )
            self._logged_first_verify = True  # 标记已记录

        return GenerationBatchResult(  # 返回结果
            logits_output=logits_output,
            next_token_ids=new_bonus_tokens,
            num_correct_drafts=num_correct_drafts,
            num_correct_drafts_per_req_cpu=num_correct_drafts_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
