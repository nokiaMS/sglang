# 内存池配置器：用于分析和计算 KV 缓存池的大小。
# 每个模型架构都有自己的配置器，通过统一的 coeff+bias 模型从可用 GPU 内存计算池大小：
#     available_bytes = max_tokens * coeff + bias
#     max_tokens = (available_bytes - bias) / coeff
# 两个入口点，相同的核心计算逻辑：
# - calculate_pool_sizes(available_bytes, page_size)：性能分析路径
# - calculate_pool_sizes_from_max_tokens(max_tokens, page_size)：约束路径

"""Memory pool configurators for profiling and sizing KV cache pools.

Each model architecture has its own configurator that computes pool sizes
from available GPU memory using a unified coeff+bias model:

    available_bytes = max_tokens * coeff + bias
    max_tokens = (available_bytes - bias) / coeff

Two entry points, same core computation:
- calculate_pool_sizes(available_bytes, page_size): profiling path
- calculate_pool_sizes_from_max_tokens(max_tokens, page_size): constraint path
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import (
    get_dsa_index_head_dim,
    is_deepseek_dsa,
    is_deepseek_v4,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.deepseek_v4_memory_pool import get_compress_state_ring_size
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.utils.common import is_float4_e2m1fn_x2


# 内存池配置数据类，存储已解析的内存池配置信息，在 target 和 draft worker 之间共享
@dataclass
class MemoryPoolConfig:
    """Resolved memory pool config, shared between target and draft workers."""

    # 最大总 token 数（KV 缓存可容纳的 token 总量）
    max_total_num_tokens: int
    # 最大并发运行请求数
    max_running_requests: Optional[int] = None
    # 全注意力层的最大总 token 数
    full_max_total_num_tokens: Optional[int] = None
    # 滑动窗口注意力层的最大总 token 数
    swa_max_total_num_tokens: Optional[int] = None

    # DSV4 压缩注意力池大小（仅 target 使用；draft worker 保持为 0）
    # DSV4 compressed-attention pool sizes (target only; draft workers leave at 0).
    # c4 压缩注意力池的最大 token 数
    c4_max_total_num_tokens: int = 0
    # c128 压缩注意力池的最大 token 数
    c128_max_total_num_tokens: int = 0
    # c4 状态池大小
    c4_state_pool_size: int = 0
    # c128 状态池大小
    c128_state_pool_size: int = 0

    # 静态内存比例（用于错误提示）
    mem_fraction_static: Optional[float] = None

    def __post_init__(self):
        # 如果最大 token 数小于等于 0，说明内存不足，抛出异常
        if self.max_total_num_tokens <= 0:
            msg = "Not enough memory. Please try to increase --mem-fraction-static."
            if self.mem_fraction_static is not None:
                msg += f" Current value: mem_fraction_static={self.mem_fraction_static}"
            raise RuntimeError(msg)


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


# 内存池配置器基类，子类通过 coeff+bias 模型计算各自架构的池大小
class MemoryPoolConfigurator:
    """Base class for memory pool configurators.

    Subclasses compute pool sizes for their architecture via coeff+bias model.
    Both entry points return MemoryPoolConfig (with max_running_requests=None,
    to be filled by the consumer).
    """

    # 性能分析路径：从可用字节数计算池大小
    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        """Profiling path: compute pool sizes from available bytes."""
        raise NotImplementedError

    # 约束路径：从受限的 max_tokens 重新计算池大小
    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        """Constraint path: recalculate pool sizes from a constrained max_tokens."""
        raise NotImplementedError


# 标准模型的配置器：支持 MHA、MLA、DSA、FP4 等架构
# coeff = cell_size（每个 token 在所有层中占用的字节数），bias = 0
class DefaultPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for standard models: MHA, MLA, DSA, FP4.

    coeff = cell_size (bytes per token across all layers)
    bias = 0
    """

    def __init__(self, mr: ModelRunner):
        # 确定用于 KV 缓存的有效层数
        # Determine effective number of layers for KV cache
        if mambaish := mr.mambaish_config:
            # 对于 mambaish 模型，只计算全注意力层的层数
            effective_layer_ids = [
                i
                for i in mambaish.full_attention_layer_ids
                if mr.start_layer <= i < mr.end_layer
            ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = mr.num_effective_layers

        # 计算每个 token 的 KV 缓存单元大小（字节）
        self._cell_size = self._compute_cell_size(mr, num_layers)

        # DFLASH：缩放 cell_size 以考虑 draft 模型的 KV 缓存
        # DFLASH: scale cell_size to account for draft model KV cache
        if mr.spec_algorithm.is_dflash() and not mr.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import (
                scale_kv_cell_size_per_token_for_dflash,
            )

            draft_num_layers = mr.dflash_draft_num_layers
            if (
                draft_num_layers is not None
                and int(draft_num_layers) > 0
                and int(num_layers) > 0
            ):
                # 按 target+draft 的比例缩放单元大小，为 draft 模型预留 KV 缓存空间
                self._cell_size = scale_kv_cell_size_per_token_for_dflash(
                    target_cell_size_per_token=self._cell_size,
                    target_num_layers=int(num_layers),
                    draft_num_layers=int(draft_num_layers),
                )

    # 计算每个 token 的 KV 缓存成本（字节），子类可重写
    def _compute_cell_size(self, mr: ModelRunner, num_layers: int) -> int:
        """Compute per-token KV cache cost in bytes. Subclasses can override."""
        # args to config cell size
        model_config = mr.model_config
        kv_cache_dtype = mr.kv_cache_dtype

        # KV 缓存数据类型每个元素的字节数
        kv_size = torch._utils._element_size(kv_cache_dtype)
        tp_size = get_attention_tp_size()

        if mr.use_mla_backend:
            # MLA 后端：cell_size = (kv_lora_rank + qk_rope_head_dim) * num_layers * kv_size
            cell_size = (
                (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                * num_layers
                * kv_size
            )
            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer：FP4 量化需要额外的缩放因子缓冲区
                # kv_scale_buffer
                scale_block_size = 16
                # FP4 数据占原大小一半，加上缩放因子的开销
                cell_size = (cell_size // 2) + (
                    (
                        (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                        // scale_block_size
                    )
                    * num_layers
                    * kv_size
                )

            # 为 DSA 模型（DeepSeek V3.2）添加索引器 KV 缓存开销
            # Add indexer KV cache overhead for DSA models (DeepSeek V3.2)
            if is_deepseek_dsa(model_config.hf_config):
                index_head_dim = get_dsa_index_head_dim(model_config.hf_config)
                # 索引器每个 token 的大小 = index_head_dim + 量化缩放因子
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // DSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    DSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                # 累加索引器的内存开销
                cell_size += indexer_size_per_token * num_layers * element_size
        else:
            # 标准 MHA 后端：cell_size = num_kv_heads * (head_dim + v_head_dim) * num_layers * kv_size
            cell_size = (
                model_config.get_num_kv_heads(tp_size)
                * (model_config.head_dim + model_config.v_head_dim)
                * num_layers
                * kv_size
            )

            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer：FP4 量化需要额外的缩放因子缓冲区
                # kv_scale_buffer
                scale_block_size = 16
                n = model_config.get_num_kv_heads(tp_size)
                k = model_config.head_dim
                # FP4 数据占原大小一半，加上缩放因子的开销
                cell_size = (cell_size // 2) + (
                    (n * k * num_layers * 2 * kv_size) // scale_block_size
                )

        return cell_size

    # 性能分析路径：从可用字节数计算池大小
    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        # 按单元大小计算最大 token 数
        max_total_num_tokens = available_bytes // self._cell_size
        # 向下对齐到 page_size 的整数倍
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)

    # 约束路径：从受限的 max_tokens 重新计算池大小
    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        # 向下对齐到 page_size 的整数倍
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)


# 混合滑动窗口注意力模型的配置器（Gemma2、Command-R、MiMo 等）
# 将可用内存在全注意力池和 SWA 池之间分配
# 不继承 DefaultPoolConfigurator，因为使用不同的 coeff 模型
class HybridSWAPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for hybrid sliding window attention models (Gemma2, Command-R, MiMo).

    Splits available memory between full attention and SWA pools.
    Does NOT inherit DefaultPoolConfigurator — different coeff model.
    """

    def __init__(self, mr: ModelRunner):
        model_config = mr.model_config
        kv_cache_dtype = mr.kv_cache_dtype
        kv_size = torch._utils._element_size(kv_cache_dtype)
        tp_size = get_attention_tp_size()

        # 全注意力层数量和滑动窗口注意力层数量
        self._full_layers_num = len(model_config.full_attention_layer_ids)
        self._swa_layers_num = len(model_config.swa_attention_layer_ids)
        assert (
            self._swa_layers_num > 0
        ), "Hybrid SWA model must have at least one SWA layer"

        # SWA token 数与 full token 数的比例
        self._swa_full_tokens_ratio = mr.server_args.swa_full_tokens_ratio

        # 全注意力层每个 token 的内存（字节）
        # Full layer per-token memory (bytes)
        self._full_per_token = (
            model_config.get_num_kv_heads(tp_size)
            * (model_config.head_dim + model_config.v_head_dim)
            * kv_size
        )

        # SWA 层每个 token 的内存（字节）
        # SWA layer per-token memory (bytes)
        self._swa_per_token = (
            model_config.get_swa_num_kv_heads(tp_size)
            * (model_config.swa_head_dim + model_config.swa_v_head_dim)
            * kv_size
        )

        # 每个 max_total_num_tokens token 对应的字节数
        # Bytes per token of max_total_num_tokens.
        #
        # 混合模式（full_layers > 0）：max_total = full_tokens，因此 cell_size 需要考虑
        # 两个池：F*nf + r*S*ns（其中 swa_tokens = full_tokens * r）
        # Hybrid (full_layers > 0): max_total = full_tokens, so cell_size accounts
        # for both pools: F*nf + r*S*ns (where swa_tokens = full_tokens * r).
        #
        # 全 SWA 模式（full_layers == 0）：max_total 直接等于 swa_tokens。比例
        # 在此没有意义——没有全注意力池可关联，且超出滑动窗口的每个 token 都可被驱逐。
        # 因此 cell_size = S*ns，不应用比例因子。
        # All-SWA (full_layers == 0): max_total = swa_tokens directly. The ratio
        # is meaningless here -- there is no full pool to relate to, and every
        # token beyond the sliding window can be evicted. So cell_size = S*ns,
        # with no ratio factor applied.
        if self._full_layers_num == 0:
            # 全 SWA 模式：cell_size 只包含 SWA 层
            self._cell_size = self._swa_per_token * self._swa_layers_num
        else:
            # 混合模式：cell_size 包含全注意力层和 SWA 层（SWA 部分乘以比例因子）
            self._cell_size = (
                self._full_per_token * self._full_layers_num
                + self._swa_full_tokens_ratio
                * self._swa_per_token
                * self._swa_layers_num
            )

    # 核心计算：将 max_total_num_tokens 分割为全注意力/SWA 池大小
    def _solve_pool_sizes(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        """Core computation: split max_total_num_tokens into full/swa pool sizes."""

        def align_page_size(x: int) -> int:
            # 向下对齐到 page_size 的整数倍
            return (x // page_size) * page_size

        if self._full_layers_num == 0:
            # 全 SWA 模式：没有全注意力池，max_total 直接作为 SWA 池大小
            # All-SWA: no full pool, max_total = actual SWA pool size.
            # Ratio is not applied -- see __init__ comment.
            swa_tokens = align_page_size(max_total_num_tokens)
            logger.info(
                f"Use sliding window memory pool (all SWA). "
                f"swa_layer_tokens={swa_tokens}"
            )
            return MemoryPoolConfig(
                max_total_num_tokens=swa_tokens,
                full_max_total_num_tokens=0,
                swa_max_total_num_tokens=swa_tokens,
            )

        # 混合模式：full_tokens = max_total_num_tokens，swa_tokens = full_tokens * ratio
        # Hybrid: full_tokens = max_total_num_tokens, swa_tokens = full_tokens * ratio
        full_tokens = align_page_size(max_total_num_tokens)
        swa_tokens = align_page_size(int(full_tokens * self._swa_full_tokens_ratio))

        logger.info(
            f"Use sliding window memory pool. "
            f"full_layer_tokens={full_tokens}, swa_layer_tokens={swa_tokens}"
        )

        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
        )

    # 性能分析路径：从可用字节数计算池大小
    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        # 按单元大小计算最大 token 数
        max_total_num_tokens = int(available_bytes // self._cell_size)
        return self._solve_pool_sizes(max_total_num_tokens, page_size)

    # 约束路径：从受限的 max_tokens 重新计算池大小
    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        return self._solve_pool_sizes(max_total_num_tokens, page_size)


# DSV4 各池大小的中间数据类
@dataclass
class _DSV4PoolSizes:
    full_max_total_num_tokens: int
    swa_max_total_num_tokens: int
    c4_max_total_num_tokens: int
    c128_max_total_num_tokens: int
    c4_state_pool_size: int
    c128_state_pool_size: int


# DSV4 压缩注意力模型的配置器
# 将可用内存分配到 full / swa / c4 / c128 + c4_state / c128_state 池中
# coeff 为 bytes_per_full_token（当投机解码预留 draft worker 时，按 (T+D)/T 膨胀，
# 类似 dflash 的 cell_size 缩放方式）；bias = 0
class DSV4PoolConfigurator(MemoryPoolConfigurator):
    """Configurator for DSV4 compressed-attention models.

    Splits available memory across full / swa / c4 / c128 + c4_state / c128_state
    pools. coeff is bytes_per_full_token (inflated by (T+D)/T when speculative
    decode reserves a draft worker, mirroring dflash's cell_size scaling); bias = 0.
    """

    def __init__(self, mr: ModelRunner):
        cfg = mr.model_config
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.indexer_head_dim = cfg.index_head_dim
        # PP 局部切片；与 DeepSeekV4TokenToKVPool 的 stage_ratios 一致
        # PP-local slice; matches DeepSeekV4TokenToKVPool's stage_ratios.
        self.compression_ratios = cfg.compress_ratios[mr.start_layer : mr.end_layer]
        if mr.pp_size > 1:
            logger.info(
                f"DSV4 pool PP slice: rank={mr.pp_group.rank_in_group} "
                f"layers=[{mr.start_layer},{mr.end_layer}) "
                f"local={len(self.compression_ratios)}/{len(cfg.compress_ratios)}"
            )
        self.swa_page_size = cfg.window_size
        self.swa_ratio = mr.server_args.swa_full_tokens_ratio
        # 是否启用投机解码
        self.is_speculative = mr.server_args.speculative_algorithm is not None
        if mr.enable_hisparse:
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            # HiSparse 配置的 host-to-device 比率，用于缩小 c4 池大小
            self.c4_shrink_factor = parse_hisparse_config(
                mr.server_args
            ).host_to_device_ratio
        else:
            self.c4_shrink_factor = 1
        assert self.c4_shrink_factor >= 1
        if self.c4_shrink_factor > 1:
            logger.info(f"HiSparse c4 host-to-device ratio = {self.c4_shrink_factor}")

        # 压缩状态环形缓冲区大小
        self.c4_ring_size = get_compress_state_ring_size(4, self.is_speculative)
        self.c128_ring_size = get_compress_state_ring_size(128, self.is_speculative)

        # 各类型层数量统计
        self.num_layers_total = len(self.compression_ratios)
        self.num_layers_ca4 = sum(1 for r in self.compression_ratios if r == 4)
        self.num_layers_ca128 = sum(1 for r in self.compression_ratios if r == 128)

        # 计算每个 full token 对应的字节数
        self.bytes_per_full_token = self._get_bytes_per_full_token()
        if self.is_speculative:
            # 投机解码时，按 (target+draft)/target 比例膨胀每个 token 的字节数，
            # 为 draft worker 预留内存。等价于 dflash 的
            # scale_kv_cell_size_per_token_for_dflash，但作用于 bytes_per_full_token：
            # tokens = avail / (bpft * (T+D)/T)
            # Reserve memory for the speculative draft worker by inflating
            # per-token bytes by (target+draft)/target. Equivalent to dflash's
            # scale_kv_cell_size_per_token_for_dflash but applied to
            # bytes_per_full_token: tokens = avail / (bpft * (T+D)/T).
            draft_layers = 1
            target_layers = self.num_layers_total
            self.bytes_per_full_token *= (target_layers + draft_layers) / target_layers

        # 在线 c128 压缩每个索引只保留一个进行中的 (max, sum, kv) 状态，
        # 并假设严格的前向调度。投机解码 (MTP) 需要 draft 和 verify 之间的
        # 回滚/重放，在线路径尚不支持。
        # Online c128 keeps a single in-progress (max, sum, kv) state per index
        # and assumes a strict forward-only schedule. Speculative decode (MTP)
        # would need rollback / replay across draft and verify, which the
        # online path doesn't support yet.
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            assert (
                mr.spec_algorithm.is_none()
            ), "SGLANG_OPT_USE_ONLINE_COMPRESS does not support speculative decode (MTP) yet"
            logger.info("DSV4 compressed attention: online c128 enabled (ring_size=1)")

    # 计算每个 full token 对应的总字节数（包含所有池的加权开销）
    def _get_bytes_per_full_token(self) -> float:
        # KV 缓存每个 token 的字节数：nope_head_dim + 2*rope_head_dim + 8（额外开销）
        kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8

        quant_block_size = 128
        # 索引器每个 token 的字节数：indexer_head_dim + 量化缩放因子
        indexer_bytes = (
            self.indexer_head_dim + self.indexer_head_dim // quant_block_size * 4
        )

        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        state_dtype_size = 4
        # c4 状态字节数：2*2*head_dim*4（max 和 sum，各两个 head_dim）
        c4_state_bytes = 2 * 2 * attn_head_dim * state_dtype_size
        # 在线 c128 每个 slot 存储 (max, sum, kv) 即 3*head_dim，而不是
        # 原始的 (kv, score) 即 2*head_dim。结合 ring_size=1 这仍然
        # 带来大幅缩减（~3/256x），但每个 slot 的字节数会增加。
        # Online c128 stores (max, sum, kv) per slot (3*head_dim) instead of
        # raw (kv, score) (2*head_dim). Combined with ring_size=1 this still
        # nets a large reduction (~3/256x) but the per-slot bytes go up.
        c128_online = envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
        # c128 状态字节数：在线模式为 3*head_dim*4，离线模式为 2*1*head_dim*4
        c128_state_bytes = (
            (3 if c128_online else 2 * 1) * attn_head_dim * state_dtype_size
        )
        # c4 索引器状态字节数
        c4_indexer_state_bytes = 2 * 2 * self.indexer_head_dim * state_dtype_size

        # 压缩状态与 SWA 页大小的比例
        c4_state_ratio = self.c4_ring_size / self.swa_page_size
        c128_state_ratio = self.c128_ring_size / self.swa_page_size

        # c4 压缩比例因子（考虑 HiSparse 缩小因子）
        c4_frac = 1 / (4 * self.c4_shrink_factor)
        # 综合计算每个 full token 的字节数：
        # = SWA部分的KV开销 + c4部分的KV开销 + c128部分的KV开销
        #   + c4部分的索引器开销 + c4状态开销 + c128状态开销 + c4索引器状态开销
        return (
            self.swa_ratio * kv_bytes * self.num_layers_total
            + c4_frac * kv_bytes * self.num_layers_ca4
            + 1 / 128 * kv_bytes * self.num_layers_ca128
            + 1 / 4 * indexer_bytes * self.num_layers_ca4
            + self.swa_ratio * c4_state_ratio * c4_state_bytes * self.num_layers_ca4
            + self.swa_ratio
            * c128_state_ratio
            * c128_state_bytes
            * self.num_layers_ca128
            + self.swa_ratio
            * c4_state_ratio
            * c4_indexer_state_bytes
            * self.num_layers_ca4
        )

    # 根据 full_token 数量计算 DSV4 各池的具体大小
    def _compute_dsv4_sizes(self, full_token: int, page_size: int) -> _DSV4PoolSizes:
        # 对齐 full_token 到 page_size 的整数倍
        full_token = full_token // page_size * page_size
        # SWA token 数 = full_token * swa_ratio，对齐到 page_size
        swa_tokens = int(full_token * self.swa_ratio) // page_size * page_size
        return _DSV4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
            # c4 池大小 = full_token / (4 * c4_shrink_factor)
            c4_max_total_num_tokens=full_token // (4 * self.c4_shrink_factor),
            # c128 池大小 = full_token / 128
            c128_max_total_num_tokens=full_token // 128,
            # c4 状态池大小 = SWA 页数 * c4 环形缓冲区大小
            c4_state_pool_size=swa_tokens // self.swa_page_size * self.c4_ring_size,
            # c128 状态池大小 = SWA 页数 * c128 环形缓冲区大小
            c128_state_pool_size=swa_tokens // self.swa_page_size * self.c128_ring_size,
        )

    # 将 _DSV4PoolSizes 转换为 MemoryPoolConfig 并记录日志
    def _to_config(self, sizes: _DSV4PoolSizes) -> MemoryPoolConfig:
        full = sizes.full_max_total_num_tokens
        swa = sizes.swa_max_total_num_tokens
        logger.info(
            f"DSV4 pool sizes: full={full}, swa={swa}, "
            f"c4={sizes.c4_max_total_num_tokens}, "
            f"c128={sizes.c128_max_total_num_tokens}, "
            f"c4_state={sizes.c4_state_pool_size}, "
            f"c128_state={sizes.c128_state_pool_size}"
        )
        return MemoryPoolConfig(
            max_total_num_tokens=full,
            full_max_total_num_tokens=full,
            swa_max_total_num_tokens=swa,
            c4_max_total_num_tokens=sizes.c4_max_total_num_tokens,
            c128_max_total_num_tokens=sizes.c128_max_total_num_tokens,
            c4_state_pool_size=sizes.c4_state_pool_size,
            c128_state_pool_size=sizes.c128_state_pool_size,
        )

    # 性能分析路径：从可用字节数计算 DSV4 池大小
    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        # 压缩注意力要求 page_size 是 128 的整数倍
        assert (
            page_size % 128 == 0
        ), "page_size must be multiple of 128 for compressed attention"

        # 按每个 full token 的字节数计算 full_token 数量
        full_token = int(available_bytes / self.bytes_per_full_token)
        sizes = self._compute_dsv4_sizes(full_token, page_size)
        logger.info(
            f"DSV4 memory calculation: "
            f"bytes_per_full_token={self.bytes_per_full_token:.2f}, "
            f"available_bytes={available_bytes / (1 << 30):.2f} GB, "
            f"full_token={sizes.full_max_total_num_tokens}"
        )
        return self._to_config(sizes)

    # 约束路径：从受限的 max_tokens 重新计算 DSV4 池大小
    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        # 压缩注意力要求 page_size 是 128 的整数倍
        assert (
            page_size % 128 == 0
        ), "page_size must be multiple of 128 for compressed attention"
        sizes = self._compute_dsv4_sizes(max_total_num_tokens, page_size)
        return self._to_config(sizes)


# 工厂函数：根据模型架构选择合适的内存池配置器
def create_memory_pool_configurator(
    mr: ModelRunner,
) -> MemoryPoolConfigurator:
    """Factory: select the right configurator for the model architecture."""
    # DeepSeek V4 + 混合 SWA 架构使用 DSV4 配置器
    if is_deepseek_v4(mr.model_config.hf_config) and mr.is_hybrid_swa:
        return DSV4PoolConfigurator(mr)
    # 其他混合 SWA 架构使用 HybridSWA 配置器
    if mr.is_hybrid_swa:
        return HybridSWAPoolConfigurator(mr)
    # 未来：MambaPoolConfigurator
    # Future: MambaPoolConfigurator
    # 默认使用标准配置器
    return DefaultPoolConfigurator(mr)
