# Copyright 2023-2024 SGLang Team
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

# Integrates "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
# and "Punica: Multi-Tenant LoRA Serving"

# LoRA 管理器模块：负责 LoRA 适配器的加载、卸载、内存池管理及批次调度，
# 集成了 S-LoRA 和 Punia 的多租户 LoRA 服务方案，
# 支持同时服务数千个并发 LoRA 适配器。

import logging
from typing import Dict, Iterable, List, Optional

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.backend.lora_registry import get_backend_from_name
from sglang.srt.lora.layers import BaseLayerWithLoRA, FusedMoEWithLoRA, get_lora_layer
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.lora.utils import (
    EMBEDDING_NAMES,
    LoRAType,
    auto_detect_lora_target_modules,
    get_normalized_target_modules,
    get_target_module_name,
)
from sglang.srt.managers.io_struct import LoRAUpdateOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import replace_submodule
from sglang.srt.utils.hf_transformers_utils import AutoConfig

logger = logging.getLogger(__name__)


# LoRA 管理器类：管理 LoRA 适配器的完整生命周期，包括初始化、加载、卸载、
# 内存池调度和批次准备，是 LoRA 服务的核心调度组件。
class LoRAManager:
    # 初始化 LoRA 管理器，设置基础模型、LoRA 后端和可变内部状态
    def __init__(
        self,
        base_model: torch.nn.Module,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        load_config: LoadConfig,
        dtype: torch.dtype,
        server_args: ServerArgs,
        lora_backend: str = "triton",
        tp_size: int = 1,
        tp_rank: int = 0,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        lora_paths: Optional[List[LoRARef]] = None,
    ):
        # 保存基础模型引用
        self.base_model: torch.nn.Module = base_model
        # 对于多模态模型，获取文本配置
        if hasattr(base_hf_config, "get_text_config"):
            self.base_hf_config: AutoConfig = base_hf_config.get_text_config()
        else:
            self.base_hf_config: AutoConfig = base_hf_config
        self.max_loras_per_batch: int = max_loras_per_batch
        self.load_config: LoadConfig = load_config
        self.dtype: torch.dtype = dtype
        # 从模型参数获取设备类型
        self.device: torch.device = next(self.base_model.parameters()).device
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.lora_added_tokens_size: Optional[int] = None
        self.enable_lora_overlap_loading: Optional[bool] = (
            server_args.enable_lora_overlap_loading
        )

        # LoRA 驱逐策略（当内存池满时如何淘汰旧适配器）
        self.eviction_policy = server_args.lora_eviction_policy
        # 是否覆盖专家共享外层 LoRA 的自动检测结果
        self._experts_shared_outer_override: Optional[bool] = (
            server_args.experts_shared_outer_loras
        )
        # 是否使用虚拟专家（MoE 场景下）
        self.lora_use_virtual_experts: bool = server_args.lora_use_virtual_experts
        # 是否严格加载 LoRA 权重
        self.lora_strict_loading: bool = getattr(
            server_args, "lora_strict_loading", False
        )

        # LoRA backend for running sgemm kernels
        # 初始化 LoRA 计算后端（如 triton），用于执行 LoRA 的 GEMM 内核
        logger.info(f"Using {lora_backend} as backend of LoRA kernels.")
        backend_type = get_backend_from_name(lora_backend)
        self.lora_backend: BaseLoRABackend = backend_type(
            max_loras_per_batch=max_loras_per_batch,
            device=self.device,
            server_args=server_args,
        )

        # Initialize mutable internal state of the LoRAManager.
        # 初始化可变内部状态，包括适配器、形状推断、模块和内存池
        self.init_state(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
            lora_paths=lora_paths,
        )

    # CUDA 图批次信息初始化（第二阶段）：为 CUDA 图捕获准备 LoRA 批次元数据
    def init_cuda_graph_batch_info(
        self, max_bs_in_cuda_graph: int, num_tokens_per_bs: int
    ):
        """Phase 2 of LoRA CUDA graph init: dense LoRA batch metadata.

        Called during CudaGraphRunner.__init__(), after init_memory_pool().
        Phase 1 (MoE buffers) is handled earlier via init_cuda_graph_moe_buffers().
        """
        self.max_bs_in_cuda_graph = max_bs_in_cuda_graph
        self.lora_backend.init_cuda_graph_batch_info(
            max_bs_in_cuda_graph=max_bs_in_cuda_graph,
            num_tokens_per_bs=num_tokens_per_bs,
        )

    # CUDA 图 MoE 缓冲区初始化（第一阶段）：为 MoE 层准备中间缓冲区
    def init_cuda_graph_moe_buffers(
        self, max_bs: int, max_loras: int, compute_dtype, moe_layer
    ):
        """Phase 1 of LoRA CUDA graph init: MoE intermediate buffers.

        Called before init_memory_pool() so memory profiling accounts for them.
        Phase 2 (dense batch metadata) is handled later via init_cuda_graph_batch_info().
        """
        self.lora_backend.init_cuda_graph_moe_buffers(
            max_bs=max_bs,
            max_loras=max_loras,
            compute_dtype=compute_dtype,
            moe_layer=moe_layer,
        )

    # 创建 LoRA 更新结果对象，包含成功/失败状态和已加载适配器列表
    def create_lora_update_result(
        self, success: bool, error_message: str = ""
    ) -> LoRAUpdateOutput:
        return LoRAUpdateOutput(
            success=success,
            error_message=error_message,
            loaded_adapters={
                lora_ref.lora_name: lora_ref.lora_path
                for lora_ref in self.lora_refs.values()
            },
        )

    # 从指定路径加载单个 LoRA 适配器，包括配置验证和权重加载
    def load_lora_adapter(self, lora_ref: LoRARef) -> LoRAUpdateOutput:
        """
        Load a single LoRA adapter from the specified path.

        Args:
            lora_ref (LoRARef): The LoRARef object containing the LoRA name, path, and ID.
        """
        # 确保 LoRA 引用的名称和路径都已设置
        assert (
            lora_ref.lora_name is not None and lora_ref.lora_path is not None
        ), "LoRARef must have both lora_name and lora_path set for loading."
        # 确保该适配器尚未加载
        assert (
            lora_ref.lora_id not in self.loras
        ), f"LoRA adapter with ID {lora_ref.lora_id} is already loaded. This should have been verified before request is sent to the backend."

        try:
            # load configs
            # 加载 LoRA 配置文件
            new_adapter = LoRAConfig(
                lora_ref.lora_path,
                base_vocab_size=self.base_hf_config.vocab_size,
            )
            # 验证新适配器是否与当前环境兼容
            self.validate_new_adapter(new_adapter, lora_ref)
            self.configs[lora_ref.lora_id] = new_adapter

            # load weights
            # 加载 LoRA 权重到 CPU 内存
            self.load_lora_weights(lora_ref)

            # keep metadata for displayed messages
            # 保存适配器引用元数据
            self.lora_refs[lora_ref.lora_id] = lora_ref
            # 统计固定（pinned）适配器数量
            self.num_pinned_loras += int(lora_ref.pinned)
        except Exception as e:
            return self.create_lora_update_result(
                success=False,
                error_message=str(e),
            )

        return self.create_lora_update_result(success=True)

    # 验证新适配器是否可加载：检查词汇表扩展、DoRA 支持和内存池兼容性
    def validate_new_adapter(self, lora_config: LoRAConfig, lora_ref: LoRARef):
        """
        Validate if an adapter can be loaded into the current LoRA memory pool and generate error if it is incompatible.
        """
        # 当前不支持添加新词元的适配器
        if lora_config.lora_added_tokens_size > 0:
            raise ValueError(
                f"Failed to load {lora_ref.lora_name} because LoRA serving currently doesn't support adapters that add tokens to the vocabulary"
            )

        # 当前不支持 DoRA 适配器
        if lora_config.use_dora:
            raise ValueError(
                f"Failed to load {lora_ref.lora_name} because LoRA serving currently doesn't support DoRA adapters"
            )

        # Check if this LoRA adapter is already loaded
        # 检查同名的适配器是否已加载
        for existing_lora_ref in self.lora_refs.values():
            if lora_ref.lora_name == existing_lora_ref.lora_name:
                raise ValueError(
                    f"Failed to load LoRA adapter {lora_ref.lora_name} because it is already loaded"
                )

            # 同一路径的适配器已用不同名称加载，发出警告
            if lora_ref.lora_path == existing_lora_ref.lora_path:
                logger.warning(
                    f"{lora_ref.lora_path} is already loaded with name: {existing_lora_ref.lora_name}, "
                    f"but another copy is being loaded with name: {lora_ref.lora_name}"
                )

        # Check if the LoRA adapter shape is compatible with the current LoRA memory pool configuration.
        # 检查适配器形状是否与当前内存池配置兼容（rank 和目标模块）
        memory_pool = getattr(self, "memory_pool", None)
        incompatible = memory_pool and not memory_pool.can_support(lora_config)
        if incompatible:
            raise ValueError(
                f"LoRA adapter {lora_ref.lora_name} with rank {lora_config.r} is incompatible with the current "
                "LoRA memory pool configuration. Please ensure that the LoRA adapter's rank is within the configured "
                "`--max-lora-rank` and that the target modules are included in `--lora-target-modules`."
            )

        # Ensure pinned LoRA adapters does not exceed maximal limit or cause starvation.
        # 确保固定适配器不超过最大限制，避免非固定适配器和基础模型饿死
        if lora_ref.pinned and self.num_pinned_loras >= self.max_loras_per_batch - 1:
            raise ValueError(
                f"Failed to load LoRA adapter {lora_ref.lora_name} as a pinned adapter. It is not allowed to pin all slots "
                "in the LoRA memory pool to avoid starvation for unpinned adapters and base models. Please increase your "
                "`--max-loras-per-batch` or load it as unpinned LoRA adapters."
            )

    # 卸载指定 LoRA 适配器，从内存池和配置中移除
    def unload_lora_adapter(self, lora_ref: LoRARef) -> LoRAUpdateOutput:
        """
        Unload LoRA adapters by their names. This will remove the adapters from the memory pool and
        delete the corresponding LoRA modules.
        """

        # 获取待卸载适配器的配置和引用
        adapter = self.configs.get(lora_ref.lora_id)
        lora_ref = self.lora_refs.get(lora_ref.lora_id)
        assert (
            adapter is not None and lora_ref is not None
        ), f"LoRA adapter with ID {lora_ref.lora_id} is not loaded. This should have been verified before request is sent to the backend."

        try:
            # 从配置、权重和引用字典中删除适配器
            del self.configs[lora_ref.lora_id]
            del self.loras[lora_ref.lora_id]
            del self.lora_refs[lora_ref.lora_id]
            # 更新固定适配器计数
            self.num_pinned_loras -= int(lora_ref.pinned)
        except Exception as e:
            return self.create_lora_update_result(
                success=False,
                error_message=str(e),
            )

        return self.create_lora_update_result(success=True)

    # 验证批次中的 LoRA ID 是否可被当前内存池容纳
    def validate_lora_batch(self, lora_ids: set[Optional[str]]) -> bool:
        """
        Validate if the LoRA IDs in the batch can be loaded into the current LoRA memory pool.
        """
        # 批次中的 LoRA 数量不能超过每批次最大限制
        if len(lora_ids) > self.max_loras_per_batch:
            return False

        # skip pinned LoRA check if no pinned LoRA adapters are loaded.
        # 如果没有固定适配器，则无需额外检查
        if self.num_pinned_loras == 0:
            return True

        # counting the number of pinned LoRA adapters in the batch.
        # 统计批次中固定适配器的数量
        pinned_loras_in_batch = 0
        for lora_id in lora_ids:
            if lora_id is not None:
                lora_ref = self.lora_refs.get(lora_id)
                assert (
                    lora_ref is not None
                ), f"LoRA ID {lora_id} not found in lora_refs."
                pinned_loras_in_batch += int(lora_ref.pinned)

        assert pinned_loras_in_batch <= self.num_pinned_loras, (
            f"Number of pinned LoRA adapters in the batch ({pinned_loras_in_batch}) exceeds the total number of pinned adapters "
            f"({self.num_pinned_loras}). This indicates a bug in the LoRA loading logic."
        )

        # 计算非固定适配器所需槽位数
        required_slots = len(lora_ids) - pinned_loras_in_batch
        # 计算内存池中非固定适配器可用的空位
        mem_pool_vacancy = self.memory_pool.max_loras_per_batch - self.num_pinned_loras

        return required_slots <= mem_pool_vacancy

    # 将新的活跃 LoRA 适配器加载到内存池中
    def fetch_new_loras(
        self, new_loras: set[Optional[str]], running_loras: set[Optional[str]] = set()
    ):
        # Load active loras into lora memory pool
        # 合并新请求和正在运行的 LoRA ID
        cur_uids = new_loras | running_loras

        assert len(cur_uids) <= self.max_loras_per_batch
        self.memory_pool.prepare_lora_batch(
            cur_uids=cur_uids,
            lora_adapters=self.loras,
            lora_modules=self.lora_modules,
            lora_refs=self.lora_refs.copy(),  # copy snapshot of current lora_refs to avoid mutation during the batch preparation.
            lora_embed_tokens_module=self.embed_tokens_module,  # merge into embedding or lora module
            lora_lm_head_module=self.lm_head_module,  # merge into embedding or lora module
        )

    # 为当前前向批次准备 LoRA 批次信息，包括权重索引、rank 和缩放因子
    def prepare_lora_batch(self, forward_batch: ForwardBatch):
        # set up batch info shared by all lora modules
        # 获取批次大小
        bs = forward_batch.batch_size

        # 判断当前批次是否可以使用 CUDA 图
        use_cuda_graph = (
            hasattr(self, "max_bs_in_cuda_graph")
            and bs <= self.max_bs_in_cuda_graph
            and forward_batch.forward_mode.is_cuda_graph()
        )

        # 初始化权重索引、LoRA rank 和缩放因子数组
        weight_indices = [0] * len(forward_batch.lora_ids)
        lora_ranks = [0] * self.max_loras_per_batch
        scalings = [0] * self.max_loras_per_batch
        for i, uid in enumerate(forward_batch.lora_ids):
            if uid not in self.memory_pool.uid_to_buffer_id:
                continue
            # 获取该适配器在内存池中的缓冲区 ID
            weight_indices[i] = self.memory_pool.get_buffer_id(uid)
            if uid is not None:
                lora = self.loras[uid]
                # 填充对应缓冲区的 rank 和缩放因子
                lora_ranks[weight_indices[i]] = lora.config.r
                scalings[weight_indices[i]] = lora.scaling
        # Do in-place updates when CUDA graph is enabled and the batch forward mode
        # could use CUDA graph.
        # 调用后端准备批次信息，支持 CUDA 图就地更新
        self.lora_backend.prepare_lora_batch(
            forward_batch=forward_batch,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            use_cuda_graph=use_cuda_graph,
        )
        # 标记当前批次是否有活跃的 LoRA 适配器
        self.lora_backend.batch_info.has_active_lora = any(
            lora_ranks[wi] > 0 for wi in weight_indices
        )

    # 更新所有 LoRA 模块，使其关联到最新的内存缓冲区
    def update_lora_info(self):
        """
        Update all LoRA modules to associate them with the latest memory buffer.
        """
        for layer_id, layer_modules in enumerate(self.lora_modules):
            for module_name, module in layer_modules.items():
                # Hack for FusedMoE layer
                # 处理 FusedMoE 层的特殊情况：需要同时设置 gate_up 和 down 投影的 LoRA 信息
                if isinstance(module, FusedMoEWithLoRA) and all(
                    x in self.target_modules for x in ["gate_up_proj", "down_proj"]
                ):
                    # 根据内存池中是否存在 _moe 后缀来选择正确的键名
                    gate_up_key = (
                        "gate_up_proj_moe"
                        if "gate_up_proj_moe" in self.memory_pool.A_buffer
                        else "gate_up_proj"
                    )
                    down_key = (
                        "down_proj_moe"
                        if "down_proj_moe" in self.memory_pool.A_buffer
                        else "down_proj"
                    )
                    # 获取 gate_up 投影的 LoRA A 和 B 权重张量
                    gate_up_a = self.memory_pool.get_tensor(
                        target_module=gate_up_key,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_A,
                    )
                    gate_up_b = self.memory_pool.get_tensor(
                        target_module=gate_up_key,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_B,
                    )
                    # 获取 down 投影的 LoRA A 和 B 权重张量
                    down_a = self.memory_pool.get_tensor(
                        target_module=down_key,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_A,
                    )
                    down_b = self.memory_pool.get_tensor(
                        target_module=down_key,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_B,
                    )

                    # 为 MoE 模块设置 LoRA 权重信息
                    module.set_lora_info(
                        gate_up_lora_a_weights=gate_up_a,
                        gate_up_lora_b_weights=gate_up_b,
                        down_lora_a_weights=down_a,
                        down_lora_b_weights=down_b,
                    )
                    continue

                # 获取标准模块的目标模块名称
                target_module = get_target_module_name(
                    module_name, self.memory_pool.target_modules
                )

                # 为标准模块设置 LoRA A 和 B 权重
                module.set_lora_info(
                    self.memory_pool.get_tensor(
                        target_module=target_module,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_A,
                    ),
                    self.memory_pool.get_tensor(
                        target_module=target_module,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_B,
                    ),
                )

        # Update embedding layer if present - gotta merge (refer to PR codebase)
        # 更新嵌入层（embed_tokens）的 LoRA 权重信息
        if self.embed_tokens_module is not None:
            self.embed_tokens_module.set_lora_info(
                self.memory_pool.get_embedding_tensor("added_tokens", LoRAType.LORA_A),
                self.memory_pool.get_embedding_tensor("embed_tokens", LoRAType.LORA_A),
                self.memory_pool.get_embedding_tensor("embed_tokens", LoRAType.LORA_B),
            )

        # Update lm_head layer if present
        # 更新语言模型头部（lm_head）的 LoRA 权重信息
        if self.lm_head_module is not None:
            self.lm_head_module.set_lora_info(
                self.memory_pool.get_embedding_tensor("lm_head", LoRAType.LORA_A),
                self.memory_pool.get_embedding_tensor("lm_head", LoRAType.LORA_B),
            )

    # 初始化 LoRA 管理器的可变内部状态，依次初始化适配器、形状、模块和内存池
    def init_state(
        self,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        lora_paths: Optional[List[LoRARef]] = None,
    ):
        """
        Initialize the internal (mutable) state of the LoRAManager.

        When `lora_paths` is provided and not empty, it might be used for inferring LoRA shape info such as
        the target modules and max_lora_rank.
        """

        # 必须提供 lora_paths 或者同时提供 max_lora_rank 和 target_modules
        assert lora_paths or (
            max_lora_rank is not None and target_modules is not None
        ), "When no initial --lora-paths is provided, you need to specify both --max-lora-rank and --lora-target-modules for LoRA initialization."

        # 依次初始化各组件
        self.init_lora_adapters(lora_paths)
        self.init_lora_shapes(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
        )

        # 确定专家共享外层 LoRA 模式：使用用户覆盖值或自动检测结果
        if self._experts_shared_outer_override is not None:
            self.experts_shared_outer_loras = self._experts_shared_outer_override
        else:
            self.experts_shared_outer_loras = self._detect_shared_outer_loras()
        if self.experts_shared_outer_loras:
            logger.info(
                "Shared outer LoRA mode enabled: gate_up lora_A and "
                "down lora_B will be shared across experts (expert_dim=1)."
            )

        self.init_lora_modules()
        self.init_memory_pool()
        self.update_lora_info()

    # 初始化 LoRA 适配器字典，加载初始路径中的适配器
    def init_lora_adapters(self, lora_paths: Optional[List[LoRARef]] = None):
        # Configs of all active LoRA adapters, indexed by LoRA ID.
        # 所有活跃 LoRA 适配器的配置，以 LoRA ID 为索引
        self.configs: Dict[str, LoRAConfig] = {}

        # LoRA adapter weights cached in CPU memory, indexed by LoRA ID.
        # 缓存在 CPU 内存中的 LoRA 适配器权重，以 LoRA ID 为索引
        self.loras: Dict[str, LoRAAdapter] = {}

        # Mapping from LoRA ID to LoRARef object.
        # LoRA ID 到 LoRARef 对象的映射
        self.lora_refs: Dict[str, LoRARef] = {}

        # Count of pinned LoRA adapters.
        # 固定（不可驱逐）LoRA 适配器的计数
        self.num_pinned_loras: int = 0

        # 加载初始路径中的所有适配器
        if lora_paths:
            for lora_ref in lora_paths:
                result = self.load_lora_adapter(lora_ref)
                if not result.success:
                    raise RuntimeError(
                        f"Failed to load LoRA adapter {lora_ref.lora_name}: {result.error_message}"
                    )

    # 自动检测已加载适配器是否使用共享外层 LoRA 格式（MoE 场景）
    def _detect_shared_outer_loras(self) -> bool:
        """Auto-detect shared outer LoRA format from loaded adapter weights.

        MoE adapters with shared outer experts store 3D tensors where
        dim[0]=1 indicates weights shared across all experts, while
        dim[0]=num_experts indicates per-expert weights.
        Returns True if gate_up lora_A has expert_dim=1 (shared).

        All loaded adapters that expose a 3D gate_up lora_A must agree;
        mixed formats raise RuntimeError.
        """
        shared_outer: Optional[bool] = None
        for adapter_id, adapter in self.loras.items():
            found = False
            for layer in adapter.layers:
                for name, weight in layer.weights.items():
                    # 查找 gate_up_proj 的 lora_A 权重，检查是否为 3D 张量
                    if (
                        "gate_up_proj" in name
                        and "lora_A" in name
                        and weight.dim() == 3
                    ):
                        # 如果第一维为 1，则表示专家间共享
                        is_shared = weight.shape[0] == 1
                        if shared_outer is None:
                            shared_outer = is_shared
                        elif shared_outer != is_shared:
                            # 不同适配器的共享格式不一致，抛出异常
                            raise RuntimeError(
                                "Mixed shared-outer LoRA formats detected across "
                                f"loaded adapters (conflict in adapter '{adapter_id}'). "
                                "All MoE adapters must either all use shared outer "
                                "experts (expert_dim=1) or all use per-expert weights."
                            )
                        found = True
                        break
                if found:
                    break
        return bool(shared_outer) if shared_outer is not None else False

    # 推断 LoRA 目标模块和最大 rank，支持从适配器配置中自动推断
    def init_lora_shapes(
        self,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
    ):
        """Infer LoRA target modules and max_lora_rank from loaded adapters if not provided."""

        # 处理 target_modules="all" 的情况，自动检测所有可应用 LoRA 的模块
        if target_modules and target_modules == {"all"}:
            self.target_modules = auto_detect_lora_target_modules(self.base_model)
            self.target_modules.update(EMBEDDING_NAMES)
            logger.info(
                "CLI --lora-target-modules='all' resolved to %s "
                "by inspecting the base model.",
                sorted(self.target_modules),
            )
            target_modules = self.target_modules
        elif target_modules:
            # 标准化用户指定的目标模块名称
            self.target_modules = get_normalized_target_modules(target_modules)
        else:
            self.target_modules = set()

        for lora_id, config in self.configs.items():
            # Handle PEFT shorthand strings like "all-linear" or "all".
            # 处理 PEFT 简写字符串，如 "all-linear" 或 "all"
            if isinstance(config.target_modules, str):
                if config.target_modules in ("all-linear", "all"):
                    if target_modules is not None:
                        # CLI --lora-target-modules already provided; skip
                        # per-adapter inference for this adapter.
                        # 已通过 CLI 指定目标模块，跳过该适配器的自动推断
                        continue
                    else:
                        # Resolve by scanning the base model for all
                        # LoRA-compatible linear modules.
                        # 通过扫描基础模型推断所有 LoRA 兼容的线性模块
                        adapter_target_modules = auto_detect_lora_target_modules(
                            self.base_model
                        )
                        logger.info(
                            "LoRA adapter '%s' uses target_modules='%s'. "
                            "Resolved to %s by inspecting the base model.",
                            self.lora_refs[lora_id].lora_name,
                            config.target_modules,
                            sorted(adapter_target_modules),
                        )
                        self.target_modules.update(adapter_target_modules)
                        continue
                else:
                    raise ValueError(
                        f"SGLang does not recognize target_modules="
                        f"'{config.target_modules}'. Please use a list of module "
                        "name suffixes in the adapter's PEFT config, or explicitly "
                        "specify --lora-target-modules during server startup."
                    )

            if not isinstance(config.target_modules, list):
                raise ValueError(
                    f"SGLang currently only supports inferring LoRA target modules when a list of "
                    "suffixes is provided in `target_modules` field of PEFT config. Please explicitly "
                    "specify `--lora-target-modules` during server startup. You can specify `all` to "
                    "enable all support modules types. "
                )

            # 标准化适配器的目标模块名称
            adapter_target_modules = get_normalized_target_modules(
                config.target_modules
            )

            if target_modules is not None:
                # When `--lora-target-modules` is provided, validate adapter target modules is a subset of the specified target modules.
                # 验证适配器的目标模块是否为 CLI 指定目标模块的子集
                if not adapter_target_modules.issubset(self.target_modules):
                    unsupported_modules = adapter_target_modules - self.target_modules
                    lora_name = self.lora_refs[lora_id].lora_name
                    raise ValueError(
                        f"LoRA adapter '{lora_name}' contains target modules {sorted(unsupported_modules)} "
                        f"that are not included in the specified --lora-target-modules {sorted(self.target_modules)}. "
                        f"Please update --lora-target-modules to include all required modules: "
                        f"{sorted(self.target_modules | adapter_target_modules)}, or use 'all' to enable all supported modules."
                    )
            else:
                # Otherwise, infer target_modules from adapter configs.
                # 从适配器配置中推断目标模块
                self.target_modules.update(adapter_target_modules)

        # 确定最大 LoRA rank：使用用户指定值或从适配器配置中取最大值
        if max_lora_rank is not None:
            self.max_lora_rank = max_lora_rank
        else:
            self.max_lora_rank = max(
                [x.r for x in self.configs.values()],
                default=0,
            )

        # Auto-infer self.lora_added_vocab_size from loaded LoRA configs
        # This happens automatically without requiring user input
        # if self.lora_added_vocab_size is None:
        # 自动从已加载的 LoRA 配置推断新增词元数量
        if self.lora_added_tokens_size is None:
            inferred_extra_vocab_size = next(
                (
                    x.lora_added_tokens_size
                    for x in self.configs.values()
                    if x.lora_added_tokens_size > 0
                ),
                0,
            )
            if inferred_extra_vocab_size > 0:
                logger.info(
                    f"self.lora_added_tokens_size={inferred_extra_vocab_size} from LoRA adapters."
                )
            self.lora_added_tokens_size = inferred_extra_vocab_size

    # 从磁盘加载 LoRA 适配器权重到 CPU 内存，并进行加载后验证
    def load_lora_weights(self, lora_ref: LoRARef):
        """
        Load the weights of a LoRA adapter to CPU memory and conducts post-loading validation.
        """
        lora_adapter = LoRAAdapter(
            lora_ref.lora_id,
            self.configs[lora_ref.lora_id],
            self.base_hf_config,
            self.load_config,
            self.lora_backend,
            base_model=self.base_model,
        )
        # 初始化权重（从磁盘加载）
        lora_adapter.initialize_weights()

        self.loras[lora_ref.lora_id] = lora_adapter

    # 从张量字典加载 LoRA 适配器权重到 CPU 内存
    def load_lora_weights_from_tensors(
        self, lora_ref: LoRARef, tensors: Dict[str, torch.Tensor]
    ):
        """
        Load the weights of a LoRA adapter from tensors to CPU memory.
        """
        lora_adapter = LoRAAdapter(
            lora_ref.lora_id,
            self.configs[lora_ref.lora_id],
            self.base_hf_config,
            self.load_config,
            self.lora_backend,
            base_model=self.base_model,
        )
        # 从张量字典初始化权重
        lora_adapter.initialize_weights_from_tensors(tensors)
        self.loras[lora_ref.lora_id] = lora_adapter

    # 从张量和配置字典加载单个 LoRA 适配器（无需磁盘 I/O）
    def load_lora_adapter_from_tensors(
        self,
        lora_ref: LoRARef,
        tensors: Dict[str, torch.Tensor],
        config_dict: Dict,
        added_tokens_config: Optional[Dict] = None,
    ) -> LoRAUpdateOutput:
        """
        Load a single LoRA adapter from tensors and config dict.
        """
        # 确保 LoRA 引用的名称和路径都已设置
        assert (
            lora_ref.lora_name is not None and lora_ref.lora_path is not None
        ), "LoRARef must have both lora_name and lora_path set for loading."
        # 确保该适配器尚未加载
        assert (
            lora_ref.lora_id not in self.loras
        ), f"LoRA adapter with ID {lora_ref.lora_id} is already loaded. This should have been verified before request is sent to the backend."

        try:
            # 从配置字典创建 LoRAConfig 对象
            new_adapter = LoRAConfig.from_dict(
                config_dict,
                added_tokens_config,
                base_vocab_size=self.base_hf_config.vocab_size,
            )
            # 验证新适配器是否兼容
            self.validate_new_adapter(new_adapter, lora_ref)
            self.configs[lora_ref.lora_id] = new_adapter

            # 从张量加载权重
            self.load_lora_weights_from_tensors(lora_ref, tensors)

            # 保存适配器引用和更新固定适配器计数
            self.lora_refs[lora_ref.lora_id] = lora_ref
            self.num_pinned_loras += int(lora_ref.pinned)
        except Exception as e:
            return self.create_lora_update_result(
                success=False,
                error_message=str(e),
            )

        return self.create_lora_update_result(success=True)

    # （重新）初始化 LoRA 内存池，基于当前配置分配 GPU 缓冲区
    def init_memory_pool(self):
        """(Re)initialize the LoRA memory pool based on the current configurations."""
        self.memory_pool = LoRAMemoryPool(
            base_hf_config=self.base_hf_config,
            max_loras_per_batch=self.max_loras_per_batch,
            dtype=self.dtype,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_lora_rank=self.max_lora_rank,
            target_modules=self.target_modules,
            base_model=self.base_model,
            eviction_policy=self.eviction_policy,
            lora_added_tokens_size=self.lora_added_tokens_size,
            experts_shared_outer_loras=self.experts_shared_outer_loras,
            strict_loading=self.lora_strict_loading,
            enable_lora_overlap_loading=self.enable_lora_overlap_loading,
        )

        # Initializing memory pool with base model
        # 用基础模型初始化内存池（加载 None 适配器对应的基础权重）
        self.fetch_new_loras({None})

    # 将指定模块包装为支持 LoRA 的模块，并替换模型中的原始模块
    def set_lora_module(self, module_name, module):
        """Wrap any module (standard or MoE) with LoRA support."""
        # 获取适合该模块类型的 LoRA 层包装器
        lora_module = get_lora_layer(module, self.lora_backend)
        # 用 LoRA 模块替换原始子模块
        replace_submodule(self.base_model, module_name, lora_module)
        return lora_module

    # 初始化所有 LoRA 模块：扫描基础模型，将目标模块替换为支持 LoRA 的版本
    def init_lora_modules(self):
        # Look-up table that essentially maps (layer_index, module_name) to the corresponding LoRA module.
        # 查找表：将 (层索引, 模块名) 映射到对应的 LoRA 模块
        self.lora_modules: List[Dict[str, BaseLayerWithLoRA]] = [
            {} for _ in range(self.base_hf_config.num_hidden_layers)
        ]

        # 嵌入层和语言模型头部的 LoRA 模块引用
        self.embed_tokens_module: Optional[BaseLayerWithLoRA] = None
        self.lm_head_module: Optional[BaseLayerWithLoRA] = None

        # When tie_word_embeddings=True, lm_head is the same Python object as
        # embed_tokens. PyTorch's named_modules() deduplicates by object identity,
        # so lm_head will not appear as a separate entry in the scan below,
        # preventing LoRA from wrapping it. To fix this, we create a new
        # ParallelLMHead that shares the same base weight tensor (no extra GPU
        # memory) so that named_modules() yields it as an independent module.
        # 当 tie_word_embeddings=True 时，lm_head 和 embed_tokens 是同一个 Python 对象，
        # PyTorch 的 named_modules() 会按对象身份去重，导致 lm_head 不会作为独立条目出现。
        # 为解决此问题，创建一个新的 ParallelLMHead 共享相同的基础权重张量（不额外占用 GPU 内存），
        # 使 named_modules() 能将其识别为独立模块。
        if "lm_head" in self.target_modules:
            lm_head = getattr(self.base_model, "lm_head", None)
            embed_tokens = None
            for name, mod in self.base_model.named_modules():
                if name.endswith("embed_tokens"):
                    embed_tokens = mod
                    break
            # 如果 lm_head 和 embed_tokens 是同一个对象，需要解绑
            if (
                lm_head is not None
                and embed_tokens is not None
                and lm_head is embed_tokens
            ):
                logger.info(
                    "lm_head is tied with embed_tokens. Creating a separate "
                    "ParallelLMHead that shares the base weight for LoRA support."
                )
                untied_lm_head = ParallelLMHead(
                    num_embeddings=embed_tokens.org_vocab_size,
                    embedding_dim=embed_tokens.embedding_dim,
                    params_dtype=embed_tokens.weight.dtype,
                    org_num_embeddings=embed_tokens.org_vocab_size,
                )
                # Share the base weight tensor — no additional GPU memory.
                # 共享基础权重张量，不额外占用 GPU 内存
                untied_lm_head.weight = embed_tokens.weight
                # Replace the model attribute so named_modules() sees it
                # independently.
                # 替换模型属性，使 named_modules() 能识别为独立模块
                self.base_model.lm_head = untied_lm_head

        for module_name, module in self.base_model.named_modules():
            # Handle embed_tokens and lm_head before the should_apply_lora gate,
            # since VL models' should_apply_lora patterns only match language
            # model layers and would incorrectly skip these.
            # Handle embed_tokens
            # 在 should_apply_lora 检查之前处理 embed_tokens 和 lm_head，
            # 因为视觉语言模型的 should_apply_lora 模式只匹配语言模型层，会错误地跳过这些模块
            if "embed_tokens" in module_name and "embed_tokens" in self.target_modules:
                if isinstance(module, VocabParallelEmbedding) and not isinstance(
                    module, BaseLayerWithLoRA
                ):
                    lora_module = self.set_lora_module(module_name, module)
                    self.embed_tokens_module = lora_module
                    continue
            # Handle lm_head
            # 处理语言模型头部 lm_head 的 LoRA 包装
            if "lm_head" in module_name and "lm_head" in self.target_modules:
                if isinstance(module, ParallelLMHead) and not isinstance(
                    module, BaseLayerWithLoRA
                ):
                    lora_module = self.set_lora_module(module_name, module)
                    self.lm_head_module = lora_module
                    continue

            # Handle DeepSeek MLA fused projection: set the boundary
            # between q_a and kv_a output partitions so the LoRA layer
            # can apply separate B projections for each.
            # 处理 DeepSeek MLA 融合投影：设置 q_a 和 kv_a 输出分区之间的边界，
            # 使 LoRA 层可以为每个分区应用独立的 B 投影
            if (
                "fused_qkv_a_proj_with_mqa" in self.target_modules
                and module_name.endswith("fused_qkv_a_proj_with_mqa")
            ):
                from sglang.srt.lora.layers import ReplicatedLinearWithLoRA

                layer_id = get_layer_id(module_name)
                if layer_id is None:
                    continue
                lora_module = self.set_lora_module(module_name, module)
                if isinstance(lora_module, ReplicatedLinearWithLoRA):
                    # 设置 q_a 的输出维度作为分区边界
                    q_lora_rank = getattr(self.base_hf_config, "q_lora_rank", None) or 0
                    lora_module.first_output_dim = q_lora_rank
                self.lora_modules[layer_id][module_name] = lora_module
                continue

            # The module should be converted if it is included in target_names
            # 如果模块名在目标模块集合中，将其替换为 LoRA 模块
            if module_name.split(".")[-1] in self.target_modules:
                layer_id = get_layer_id(module_name)
                if layer_id is None:
                    continue
                self.lora_modules[layer_id][module_name] = self.set_lora_module(
                    module_name, module
                )
                continue

            # 处理 FusedMoE 层：当 gate_up_proj 和 down_proj 都在目标模块中时
            if isinstance(module, FusedMoE) and all(
                x in self.target_modules for x in ["gate_up_proj", "down_proj"]
            ):
                layer_id = get_layer_id(module_name)
                if layer_id is None:
                    # FusedMoE submodules outside the decoder layer hierarchy
                    # (e.g. nested helpers under non-".layers." prefixes) have
                    # no resolvable layer id; skip them so we don't index
                    # `self.lora_modules` with `None`.
                    # 跳过不在解码器层层级中的 FusedMoE 子模块
                    continue
                lora_module = self.set_lora_module(module_name, module)
                # 设置 MoE 专家的共享外层 LoRA 和虚拟专家选项
                lora_module.experts_shared_outer_loras = self.experts_shared_outer_loras
                lora_module.lora_use_virtual_experts = self.lora_use_virtual_experts
                self.lora_modules[layer_id][module_name] = lora_module
