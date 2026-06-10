# Qwen3-VL 视觉语言模型推理实现，兼容 HuggingFace 权重格式
# 本文件实现了 Qwen3-VL 模型的完整推理流程，包括视觉编码器（ViT）、
# 补丁嵌入、位置编码插值、视觉特征合并以及条件生成模型等核心组件。
# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
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
"""Inference-only Qwen3-VL model compatible with HuggingFace weights."""

import logging
import re
from collections import defaultdict
from functools import lru_cache, partial
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN

from sglang.srt.configs.qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import (
    BATCH_BUCKETS,
    FLASHINFER_MAX_SEQLEN_BUCKETS,
    FLASHINFER_WORKSPACE_SIZE_BYTES,
    VisionAttention,
)
from sglang.srt.layers.conv import Conv3dLayer
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.models.utils import (
    RotaryPosMixin,
    WeightsMapper,
    compute_cu_seqlens_from_grid_numpy,
)
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    cpu_has_amx_support,
    is_cpu,
    is_npu,
    round_up,
)
from sglang.srt.utils.hf_transformers_utils import get_processor

# 判断是否为NPU设备
_is_npu = is_npu()
# 默认使用ViT CUDA图运行器，按设备类型映射不同的图运行器
graph_runners_dict = defaultdict(lambda: ViTCudaGraphRunner)
if _is_npu:
    from sglang.srt.hardware_backend.npu.graph_runner.vit_npu_graph_runner import (
        ViTNpuGraphRunner,
    )

    # NPU设备使用专用的NPU图运行器
    graph_runners_dict["npu"] = ViTNpuGraphRunner


logger = logging.getLogger(__name__)

# 检测CPU是否支持AMX指令集
_is_cpu_amx_available = cpu_has_amx_support()
# 判断是否为CPU设备
_is_cpu = is_cpu()


# 视觉MLP模块，用于视觉编码器中的前馈网络
class Qwen3_VisionMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        hidden_act="silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        # 根据是否使用数据并行设置张量并行大小和排名
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
        self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
        # 第一个线性层：将输入维度映射到隐藏维度（列并行）
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        # 第二个线性层：将隐藏维度映射回输入维度（行并行）
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )
        # 激活函数
        self.act = ACT2FN[hidden_act]

    # 前向传播：fc1 -> 激活 -> fc2
    def forward(self, x: torch.Tensor):
        x_fc1, _ = self.linear_fc1(x)
        mlp_output, _ = self.linear_fc2(self.act(x_fc1))
        return mlp_output


# 视觉补丁嵌入层，将输入图像像素转换为补丁嵌入向量
class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # 空间补丁大小
        self.patch_size = config.patch_size
        # 时间维度补丁大小
        self.temporal_patch_size = config.temporal_patch_size
        # 输入通道数
        self.in_channels = config.in_channels
        # 嵌入维度
        self.embed_dim = config.hidden_size

        # 3D卷积核大小：[时间, 高度, 宽度]
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        # 使用3D卷积进行补丁嵌入，步长等于核大小实现非重叠切分
        self.proj = Conv3dLayer(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    # 将输入重塑为3D补丁格式并通过卷积投影
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        # 将一维输入重塑为[批次, 通道, 时间, 高度, 宽度]格式
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        # 通过3D卷积进行补丁嵌入并展平
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


# 视觉Transformer块，包含层归一化、自注意力和MLP
class Qwen3_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_dim: int,
        head_size: Optional[int] = None,
        hidden_act="silu",
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        workspace_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        # 默认使用LayerNorm归一化层
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # 注意力层前的归一化
        self.norm1 = norm_layer(dim)
        # MLP层前的归一化
        self.norm2 = norm_layer(dim)

        # 视觉注意力层
        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            head_size=head_size,
            projection_size=num_heads * head_size,
            use_qkv_parallel=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            use_data_parallel=use_data_parallel,
            use_dp_attention_reduce=is_dp_attention_enabled(),
            workspace_buffer=workspace_buffer,
        )
        # 视觉MLP层
        self.mlp = Qwen3_VisionMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel,
        )

    # 前向传播：归一化 -> 注意力 -> 残差连接 -> 归一化 -> MLP -> 残差连接
    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        output_ws: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 注意力前的层归一化
        hidden_states = self.norm1(x)
        # 重排张量维度：从(seq, batch, ...)变为(batch, seq, ...)
        hidden_states = rearrange(hidden_states, "s b ... -> b s ...")
        # 计算视觉注意力
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            output_ws=output_ws,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        # 重排回(seq, batch, ...)格式
        attn = rearrange(attn, "b s ... -> s b ...")
        # 残差连接
        x += attn
        # MLP前的层归一化
        norm2 = self.norm2(x)
        # MLP前馈计算
        mlp = self.mlp(norm2)
        # 残差连接
        x += mlp
        return x


# 视觉补丁合并器，将视觉特征映射到语言模型的隐藏维度
class Qwen3VLMoeVisionPatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        padded_context_dim: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        # 合并后的隐藏维度 = 上下文维度 * 空间合并大小的平方
        self.hidden_size = context_dim * (spatial_merge_size**2)
        # 填充后的上下文维度
        self.padded_context_dim = padded_context_dim * (spatial_merge_size**2)

        # 是否在重排后使用归一化
        self.use_postshuffle_norm = use_postshuffle_norm

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # 归一化层：根据是否使用后重排归一化选择不同的维度
        self.norm = norm_layer(
            self.hidden_size if use_postshuffle_norm else context_dim
        )
        # 根据是否使用数据并行设置张量并行参数
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
        self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
        # 第一个线性层：将合并后的维度映射到填充维度（列并行）
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.padded_context_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        # GELU激活函数
        self.act_fn = nn.GELU()
        # 第二个线性层：将填充维度映射到目标维度（行并行）
        self.linear_fc2 = RowParallelLinear(
            self.padded_context_dim,
            dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )

    # 前向传播：归一化 -> fc1 -> 激活 -> fc2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            # 后重排归一化：先展平再归一化
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            # 前重排归一化：先归一化再展平
            x = self.norm(x).view(-1, self.hidden_size)

        x_parallel, _ = self.linear_fc1(x)
        x_parallel = self.act_fn(x_parallel)
        out, _ = self.linear_fc2(x_parallel)
        return out


# 视觉编码器模型，包含补丁嵌入、位置编码、Transformer块和补丁合并器
class Qwen3VLMoeVisionModel(nn.Module, RotaryPosMixin):

    def __init__(
        self,
        vision_config: Qwen3VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        # 获取流水线并行组
        self.pp_group = get_pp_group()
        # 视觉编码器隐藏维度
        self.hidden_size = vision_config.hidden_size
        # 注意力头数
        self.num_heads = vision_config.num_heads
        # 位置编码的数量
        self.num_position_embeddings = vision_config.num_position_embeddings
        # 计算网格每边的数量（位置编码数量的平方根）
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        # 网格总数
        self.num_grid = self.num_grid_per_side * self.num_grid_per_side
        # 是否使用精确的嵌入插值（align_corners模式）
        self.align_corners = (
            get_global_server_args().enable_precise_embedding_interpolation
        )
        # 空间补丁大小
        self.patch_size = vision_config.patch_size
        # 空间合并大小
        self.spatial_merge_size = vision_config.spatial_merge_size
        # 空间合并单元大小（合并大小的平方）
        self.spatial_merge_unit = self.spatial_merge_size**2
        # 时间补丁大小
        self.temporal_patch_size = vision_config.temporal_patch_size
        # 是否使用数据并行
        self.use_data_parallel = use_data_parallel
        # layer indexes of which layer's output should be deep-stacked
        # 深度堆叠视觉索引：指定哪些层的输出需要进行深度堆叠
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        # 输出隐藏维度 = 基础维度 * (1 + 深度堆叠层数)
        self.out_hidden_size = vision_config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )
        # 补丁嵌入层
        self.patch_embed = Qwen3VLVisionPatchEmbed(config=vision_config)
        # 仅在流水线第一阶段的rank上创建位置嵌入
        if self.pp_group.is_first_rank:
            self.pos_embed = VocabParallelEmbedding(
                self.num_position_embeddings,
                self.hidden_size,
                quant_config=quant_config,
                enable_tp=not use_data_parallel,
                use_attn_tp_group=is_dp_attention_enabled() and not use_data_parallel,
                prefix=add_prefix("pos_embed", prefix),
            )
        else:
            # 非第一阶段使用占位层
            self.pos_embed = PPMissingLayer()

        # CPU + AMX支持时使用优化的LayerNorm
        if _is_cpu and _is_cpu_amx_available:
            from sglang.srt.layers.layernorm import LayerNorm

            norm_layer = partial(LayerNorm, eps=norm_eps, dtype=self.dtype)
        else:
            norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        # CPU设备上使用原始头数计算头维度
        if _is_cpu and hasattr(vision_config, "original_num_heads"):
            head_dim = self.hidden_size // vision_config.original_num_heads
        else:
            head_dim = self.hidden_size // self.num_heads
        # 旋转位置编码
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )

        # 初始化FlashInfer工作空间缓冲区
        workspace_buffer = None
        if get_global_server_args().mm_attention_backend == "flashinfer_cudnn":
            if torch.cuda.is_available() and (not _is_npu):
                ws_device = torch.device("cuda", torch.cuda.current_device())
            else:
                ws_device = self.device
            # 分配FlashInfer工作空间内存
            workspace_buffer = torch.empty(
                FLASHINFER_WORKSPACE_SIZE_BYTES,
                dtype=torch.uint8,
                device=ws_device,
            )

        # 构建视觉Transformer块列表
        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_dim=vision_config.intermediate_size,
                    head_size=head_dim,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                    use_data_parallel=use_data_parallel,
                    workspace_buffer=workspace_buffer,
                )
                for layer_idx in range(vision_config.depth)
            ]
        )
        # 主补丁合并器：将视觉特征映射到语言模型维度
        self.merger = Qwen3VLMoeVisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            padded_context_dim=self.num_heads * head_dim,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
            use_data_parallel=use_data_parallel,
        )

        # 深度堆叠合并器列表：用于合并中间层的视觉特征
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLMoeVisionPatchMerger(
                    dim=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    padded_context_dim=self.num_heads * head_dim,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"deepstack_merger_list.{layer_idx}", prefix),
                    use_data_parallel=use_data_parallel,
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
        )

        # 张量并行大小
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        # 根据设备类型初始化图运行器
        self.graph_runners = graph_runners_dict[self.device.type](self)

    # 获取模型数据类型
    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    # 获取模型设备
    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    # 计算旋转位置编码
    def rot_pos_emb(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 为每个网格计算位置ID
        pos_ids = []
        for t, h, w in grid_thw:
            # 使用混入类方法计算基础旋转位置ID
            base = self.rot_pos_ids(h, w, self.spatial_merge_size)
            # 时间维度大于1时重复位置编码
            pos_ids.append(base if t == 1 else base.repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)
        # 计算最大网格尺寸
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)

        # Use pre-computed cos_sin_cache from RotaryEmbedding
        # 使用预计算的cos/sin缓存
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        # 根据位置ID索引cos和sin值并展平
        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)

        return cos_combined, sin_combined

    # 计算单维度的连续插值索引
    def _get_interpolation_indices(self, dim_size: int) -> torch.Tensor:
        """
        Compute continuous interpolation indices for a single dimension.

        Returns continuous indices.
        """
        if self.align_corners:
            # align_corners模式：线性等间距采样
            indices = np.linspace(
                0, self.num_grid_per_side - 1, dim_size, dtype=np.float32
            )
        else:
            # 非align_corners模式：半像素偏移采样
            indices = (np.arange(dim_size, dtype=np.float32) + 0.5) * (
                self.num_grid_per_side / dim_size
            ) - 0.5
            indices = np.clip(indices, 0, self.num_grid_per_side - 1)
        return indices

    # 计算双线性插值的索引和权重
    def _calculate_indices_and_weights(self, h_idxs, w_idxs):
        """
        Compute bilinear interpolation indices and weights.

        Returns tuple of (indices, weights), each as 4 numpy arrays for the 4 corner points.
        """
        # 高度方向的向下取整和上限索引
        h_f = np.floor(h_idxs).astype(np.int64)
        h_c = np.clip(h_f + 1, 0, self.num_grid_per_side - 1)
        # 高度方向的小数部分（插值权重）
        dh = h_idxs - h_f

        # 宽度方向的向下取整和上限索引
        w_f = np.floor(w_idxs).astype(np.int64)
        w_c = np.clip(w_f + 1, 0, self.num_grid_per_side - 1)
        # 宽度方向的小数部分（插值权重）
        dw = w_idxs - w_f

        side = self.num_grid_per_side

        # 四个角点的扁平化索引
        indices = [
            (h_f[:, None] * side + w_f).flatten(),
            (h_f[:, None] * side + w_c).flatten(),
            (h_c[:, None] * side + w_f).flatten(),
            (h_c[:, None] * side + w_c).flatten(),
        ]
        # 四个角点的双线性插值权重
        weights = [
            ((1 - dh)[:, None] * (1 - dw)).flatten(),
            ((1 - dh)[:, None] * dw).flatten(),
            (dh[:, None] * (1 - dw)).flatten(),
            (dh[:, None] * dw).flatten(),
        ]
        return indices, weights

    # 将位置嵌入平铺并重组，使其与token序列对齐
    def _get_position_embedding(self, patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        """
        Tile and reorganize position embeddings to align with the token sequence.
        """
        result_parts = []
        merge_size = self.spatial_merge_size

        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            # 沿时间维度重复位置编码
            pos_embed = pos_embed.repeat(t, 1)

            # 计算合并后的高度和宽度
            h_merge = h // merge_size
            w_merge = w // merge_size

            # 重排位置编码以匹配空间合并模式
            pos_embed = (
                pos_embed.view(t, h_merge, merge_size, w_merge, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )

            result_parts.append(pos_embed)

        return torch.cat(result_parts, dim=0)

    # 在PyTorch中计算插值索引（用于CUDA图模式）
    def _torch_interp_indices(
        self, dim_size: int, device: torch.device
    ) -> torch.Tensor:
        side = self.num_grid_per_side
        if self.align_corners:
            # align_corners=True
            return torch.linspace(
                0, side - 1, dim_size, dtype=torch.float32, device=device
            )
        else:
            # align_corners=False  (match _get_interpolation_indices)
            # 匹配CPU版本的插值索引计算
            idx = (torch.arange(dim_size, dtype=torch.float32, device=device) + 0.5) * (
                side / dim_size
            ) - 0.5
            return idx.clamp_(0, side - 1)

    # 从列表快速插值位置嵌入（用于非CUDA图模式）
    def fast_pos_embed_interpolate_from_list(self, grid_thw):
        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim

        outputs = []
        for t, h, w in grid_thw:
            # 高度和宽度方向的插值索引
            h_idxs = torch.linspace(
                0, num_grid_per_side - 1, h, dtype=torch.float32, device=self.device
            )
            w_idxs = torch.linspace(
                0, num_grid_per_side - 1, w, dtype=torch.float32, device=self.device
            )

            # 计算双线性插值的四个角点索引
            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            # 计算小数部分（插值权重）
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Create meshgrid view for all h, w vars
            # 创建meshgrid以计算所有h, w组合的权重
            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            # original computation of weights
            # 原始权重计算方式
            # w00 = (1 - dh_grid) * (1 - dw_grid)
            # w01 = (1 - dh_grid) * dw_grid
            # w10 = dh_grid * (1 - dw_grid)
            # w11 = dh_grid * dw_grid
            # we reuse w11 here to avoid duplicate
            # dh_grid * dw_grid computation
            # 复用w11以避免重复计算dh_grid * dw_grid
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            # 构建四个角点的网格索引
            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * num_grid_per_side

            # 计算扁平化的索引和权重
            indices = (h_grid_idx + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=self.dtype)

            # 查找位置嵌入并加权求和
            embeds = self.pos_embed(indices)
            embeds *= weights
            combined = embeds.sum(dim=0)

            # 重排以匹配空间合并模式
            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            )
            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            # 沿时间维度重复
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    # 为FlashInfer序列长度添加批次填充
    def add_padding_to_fi_seqlens(
        self, seq: np.ndarray, batch_size: int, padding_value: int
    ) -> np.ndarray:
        # 找到大于等于当前批次大小的最小桶大小
        batch_size_padded = next(
            (b for b in BATCH_BUCKETS if b >= batch_size),
            # For large batches (> max bucket), round up to a multiple of
            # the base bucket size to avoid negative pad length.
            # 对于超过最大桶大小的批次，向上取整到基础桶大小的倍数
            round_up(batch_size, BATCH_BUCKETS[0]),
        )
        if batch_size_padded == batch_size:
            return seq
        # 填充额外的批次位置
        return np.concatenate(
            [
                seq,
                np.full(
                    (batch_size_padded - batch_size,), padding_value, dtype=seq.dtype
                ),
            ]
        )

    # 将FlashInfer最大序列长度桶化，用于cuDNN图缓存
    def bucket_flashinfer_max_seqlen(self, real_max_seqlen: int) -> int:
        if real_max_seqlen <= 0:
            return FLASHINFER_MAX_SEQLEN_BUCKETS[0]
        return next(
            (s for s in FLASHINFER_MAX_SEQLEN_BUCKETS if s >= real_max_seqlen),
            # For large sequences (> max bucket), round up to a multiple of
            # the largest bucket to avoid under-estimation.
            # 对于超过最大桶大小的序列，向上取整到最大桶大小的倍数
            round_up(real_max_seqlen, FLASHINFER_MAX_SEQLEN_BUCKETS[-1]),
        )

    # 批量插值位置嵌入（用于CUDA图模式）
    def fast_pos_embed_interpolate(self, grid_thw):
        """Interpolate position embeddings for (batch, 3) size input dimensions.

        Performs bilinear interpolation on spatial dimensions (height, width) and replicates
        along temporal dimension. The result is reorganized according to spatial_merge_size.

        Args:
            grid_thw: Tensor of shape [batch_size, 3] with (temporal, height, width) dimensions
                     in patches for each sample.

        Returns:
            Interpolated position embeddings tensor.
        """
        # 将grid_thw转移到CPU
        grid_thw_cpu = grid_thw.cpu().numpy()

        # transfer data to CPU before loop
        # 在循环前将数据转移到CPU
        temporal_dims = grid_thw_cpu[:, 0].tolist()
        height_dims = grid_thw_cpu[:, 1].tolist()
        width_dims = grid_thw_cpu[:, 2].tolist()

        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        # 计算每个样本的补丁数量
        patches_size = [h * w for h, w in zip(height_dims, width_dims)]
        total_patches = sum(patches_size)
        # 预分配索引和权重数组
        all_indices_np = np.zeros((4, total_patches), dtype=np.int64)
        all_weights_np = np.zeros((4, total_patches), dtype=np.float32)

        current_idx = 0

        # calculate indices and weights on CPU
        # 在CPU上计算索引和权重
        for t, h, w in zip(temporal_dims, height_dims, width_dims):
            # 获取高度和宽度方向的插值索引
            h_idxs = self._get_interpolation_indices(h)
            w_idxs = self._get_interpolation_indices(w)

            # 计算双线性插值的索引和权重
            indices, weights = self._calculate_indices_and_weights(h_idxs, w_idxs)

            # 将索引和权重填入预分配数组
            end_idx = current_idx + h * w
            for i in range(4):
                all_indices_np[i, current_idx:end_idx] = indices[i]
                all_weights_np[i, current_idx:end_idx] = weights[i]
            current_idx = end_idx

        # 将索引和权重从CPU转移到GPU
        idx_tensor = torch.from_numpy(all_indices_np).to(device)
        weight_tensor = torch.from_numpy(all_weights_np).to(dtype=dtype, device=device)

        # calculate interpolation
        # 计算插值结果
        pos_embeds = self.pos_embed(idx_tensor.view(-1))
        pos_embeds = pos_embeds.view(4, total_patches, -1)
        # 加权求和四个角点的位置嵌入
        patch_pos_embeds = (pos_embeds * weight_tensor.unsqueeze(-1)).sum(dim=0)
        # 按补丁数量分割
        patch_pos_embeds = patch_pos_embeds.split(patches_size)
        return self._get_position_embedding(
            patch_pos_embeds, temporal_dims, height_dims, width_dims
        )

    # 计算FlashInfer cuDNN预填充的打包元素偏移量
    def compute_flashinfer_batch_offsets_packed(
        self,
        token_cu_seqlens: np.ndarray,
        *,
        elem_per_token: int,
    ) -> np.ndarray:
        """
        Build packed *element* indptrs for FlashInfer cuDNN prefill.

        Input:
        token_cu_seqlens: (B+1,) token indptr
        elem_per_token: per-token element width on THIS TP rank
                        (usually hidden_size / attn_tp_size)

        Output:
        packed_offsets: (3 * (B_padded + 1),) int32
            [qk_indptr, v_indptr, o_indptr] concatenated,
            each indptr is (B_padded + 1,) in element units.
        """
        assert token_cu_seqlens.ndim == 1 and token_cu_seqlens.size >= 2
        B = int(token_cu_seqlens.size - 1)
        # 将批次大小桶化
        B_padded = self.bucket_flashinfer_batch_size(B)

        # token indptr -> pad to (B_padded+1,) by appending total_tokens for extra empty sequences
        # token indptr -> 通过追加总token数填充到(B_padded+1,)大小
        token_indptr = token_cu_seqlens.astype(np.int64, copy=False)  # (B+1,)
        if B_padded != B:
            # 填充额外空序列的偏移量
            pad = np.full((B_padded - B,), token_indptr[-1], dtype=token_indptr.dtype)
            token_indptr = np.concatenate([token_indptr, pad], axis=0)  # (B_padded+1,)

        # convert token indptr -> element indptr
        # 将token偏移转换为元素偏移
        elem_indptr = (token_indptr * int(elem_per_token)).astype(
            np.int32
        )  # (B_padded+1,)

        # q/k/v/o in this ViT path share the same indptr
        # q/k/v/o在此ViT路径中共享相同的indptr
        return np.concatenate([elem_indptr, elem_indptr, elem_indptr], axis=0)

    # 将批次大小桶化，用于cuDNN图缓存
    def bucket_flashinfer_batch_size(self, batch_size: int) -> int:
        """Bucketize batch size for cuDNN graph caching."""
        return next(
            (b for b in BATCH_BUCKETS if b >= batch_size),
            round_up(batch_size, BATCH_BUCKETS[0]),
        )

    # 计算填充后的序列长度
    def compute_flashinfer_sequence_lengths_padded(
        self,
        token_cu_seqlens: np.ndarray,
    ) -> np.ndarray:
        """
        token_cu_seqlens: (B+1,) token indptr
        return: (B_padded,) token lengths (padded with 0)
        """
        assert token_cu_seqlens.ndim == 1 and token_cu_seqlens.size >= 2
        B = int(token_cu_seqlens.size - 1)

        # 计算每个序列的实际长度
        seq_lens = (token_cu_seqlens[1:] - token_cu_seqlens[:-1]).astype(
            np.int32
        )  # (B,)

        # 填充到桶化的批次大小
        B_padded = self.bucket_flashinfer_batch_size(B)
        if B_padded != B:
            pad = np.zeros((B_padded - B,), dtype=np.int32)
            seq_lens = np.concatenate([seq_lens, pad], axis=0)  # (B_padded,)
        return seq_lens

    # 视觉编码器前向传播
    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # 检查是否启用CUDA图
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            if _is_npu:
                return self.forward_with_npu_graph(x, grid_thw)
            return self.forward_with_cuda_graph(x, grid_thw)

        # 将输入转移到正确的设备和数据类型
        x = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        # 补丁嵌入
        x = self.patch_embed(x)

        # 处理grid_thw格式
        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_thw = grid_thw.cpu().numpy()

        # 计算插值位置嵌入并添加到输入
        pos_embeds = self.fast_pos_embed_interpolate_from_list(grid_thw_list)
        x += pos_embeds

        # 计算旋转位置编码
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        # ---- build token indptr (B+1,) ----
        # ---- 构建token累积序列长度 (B+1,) ----
        token_cu_seqlens = np.repeat(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(axis=0, dtype=np.int32)
        # 在开头添加0
        token_cu_seqlens = np.concatenate(
            [np.zeros(1, dtype=np.int32), token_cu_seqlens]
        )

        flashinfer_max_seqlen = 0
        cu_seqlens = None
        if get_global_server_args().mm_attention_backend == "flashinfer_cudnn":
            # real token lens (B,)
            # 实际token长度 (B,)
            real_seq_lens = token_cu_seqlens[1:] - token_cu_seqlens[:-1]
            # 桶化最大序列长度
            flashinfer_max_seqlen = self.bucket_flashinfer_max_seqlen(
                int(real_seq_lens.max()) if real_seq_lens.size > 0 else 0
            )

            # (B_padded,) token lengths
            # (B_padded,) token长度
            seq_lens_padded = self.compute_flashinfer_sequence_lengths_padded(
                token_cu_seqlens
            )

            # element-per-token width on THIS ATTENTION TP rank
            # q/k/v in VisionAttention are sharded by attention TP
            # 当前注意力TP rank上每个token的元素宽度
            # VisionAttention中的q/k/v按注意力TP分片
            attn_tp_size = 1 if self.use_data_parallel else self.tp_size
            elem_per_token = (
                self.hidden_size // attn_tp_size
            )  # == heads_per_rank * head_dim

            # (3*(B_padded+1),) packed element indptrs
            # (3*(B_padded+1),) 打包的元素偏移量
            offsets_packed = self.compute_flashinfer_batch_offsets_packed(
                token_cu_seqlens,
                elem_per_token=elem_per_token,
            )

            # 构建序列长度张量，调整形状以匹配cuDNN格式
            sequence_lengths = (
                torch.from_numpy(seq_lens_padded)
                .to(device=self.device, dtype=torch.int32, non_blocking=True)
                .view(-1, 1, 1, 1)
            )  # match cuDNN test style

            cu_seqlens = torch.from_numpy(offsets_packed).to(
                device=self.device, dtype=torch.int32, non_blocking=True
            )

            max_seqlen = int(flashinfer_max_seqlen)
            sequence_lengths = sequence_lengths.to(self.device, non_blocking=True)
        else:
            sequence_lengths = None
            cu_seqlens = torch.from_numpy(token_cu_seqlens)
            if not _is_npu:
                cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
            else:
                # NPU设备上cu_seqlens保持在CPU
                cu_seqlens = cu_seqlens.to("cpu")
            max_seqlen = None

        # 添加batch维度
        x = x.unsqueeze(1)

        cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)

        # 深度堆叠特征列表
        deepstack_feature_lists = []
        num_deepstack_captured = 0

        # 逐层通过视觉Transformer块
        for layer_num, blk in enumerate(self.blocks):
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
                sequence_lengths=sequence_lengths,
            )

            # 如果当前层是深度堆叠层，收集其特征
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[num_deepstack_captured](
                    x
                )
                deepstack_feature_lists.append(deepstack_feature)
                num_deepstack_captured += 1
        # 通过主合并器处理最终特征
        x = self.merger(x)
        # 拼接主特征和深度堆叠特征
        hidden_states = torch.cat(
            [x] + deepstack_feature_lists, dim=1
        )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        return hidden_states

    # 使用NPU图运行前向传播
    def forward_with_npu_graph(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        (
            x,
            cu_seqlens,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
        ) = self._prepare_graph_inputs(x, grid_thw)

        # NPU上cu_seqlens保持在CPU
        cu_seqlens = cu_seqlens.to("cpu")
        return self.graph_runners.run(
            x=x,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            cu_seqlens=cu_seqlens,
            output_indices=None,
        )

    # 使用CUDA图运行前向传播
    def forward_with_cuda_graph(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        (
            x,
            cu_seqlens,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
        ) = self._prepare_graph_inputs(x, grid_thw)
        # 确保cu_seqlens是正确的张量格式
        if not isinstance(cu_seqlens, torch.Tensor):
            cu_seqlens = torch.tensor(cu_seqlens, device=x.device, dtype=torch.int32)
        else:
            cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32)
        # 确保cu_seqlens内存连续
        cu_seqlens = cu_seqlens.contiguous()

        return self.graph_runners.run(
            x=x,
            position_embeddings=None,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            cu_seqlens=cu_seqlens,
            cu_window_seqlens=None,
            output_indices=None,
        )

    # 加载视觉编码器权重
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # 堆叠参数映射：将q/k/v合并为qkv
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # 将分片名称替换为合并后的参数名称
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    # 准备CUDA/NPU图运行的输入数据
    def _prepare_graph_inputs(self, x: torch.Tensor, grid_thw: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # patchify
        # 补丁化
        x = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        x = self.patch_embed(x)

        # 处理grid_thw格式
        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
        else:
            grid_thw_list = grid_thw.tolist()

        # 使用批量插值方法计算位置嵌入
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        x += pos_embeds

        # rotary embedding -> (cos, sin)
        # 旋转位置编码 -> (cos, sin)
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        # compute cu_seqlens
        # 计算累积序列长度
        cu_seqlens = compute_cu_seqlens_from_grid_numpy(grid_thw)
        return x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin


# 带缓存的多模态处理器获取函数
cached_get_processor = lru_cache(get_processor)


# Qwen3语言模型，基于Qwen3Model扩展，支持深度堆叠嵌入
class Qwen3LLMModel(Qwen3Model):

    def __init__(
        self,
        *,
        config: Qwen3VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        # 验证流水线非第一阶段的起始层约束
        if not self.pp_group.is_first_rank:
            assert self.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), "start_layer should be greater than or equal to len(deepstack_visual_indexes)"

        self.hidden_size = config.hidden_size
        # 深度堆叠嵌入到解码器层的映射：前N层对应N个深度堆叠特征
        self.deepstack_embed_to_decoder_layer = range(
            len(config.vision_config.deepstack_visual_indexes)
        )

    # 获取指定层的深度堆叠嵌入
    def get_deepstack_embeds(
        self, layer_idx: int, input_deepstack_embeds: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Get deepstack embeddings for a given layer index, or None if not applicable."""
        if (
            input_deepstack_embeds is None
            or layer_idx not in self.deepstack_embed_to_decoder_layer
        ):
            return None
        # 从拼接的深度堆叠嵌入中切片出当前层的部分
        sep = self.hidden_size * layer_idx
        return input_deepstack_embeds[:, sep : sep + self.hidden_size]

    # 语言模型前向传播
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:

        if self.pp_group.is_first_rank:
            # 第一阶段：从输入ID或嵌入获取隐藏状态
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            # 非第一阶段：从流水线代理张量获取隐藏状态
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        # 收集辅助隐藏状态（用于EAGLE3等推测解码）
        aux_hidden_states = []
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer]
        ):
            layer_idx = layer_idx + self.start_layer
            # 如果当前层需要捕获，保存隐藏状态
            if layer_idx in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )

            # SGLang applies residual at the START of the next layer, not at the END like HuggingFace.
            # See: https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L549
            # To match HF behavior, deepstack must be added AFTER residual: (hidden_states + residual) + deepstack
            # The order matters because addition with different tensors is not associative in practice.
            # Deepstack for prev_layer is applied at the start of current layer via post_residual_addition.
            # SGLang在下一层的开头应用残差连接，而非HuggingFace那样在层末尾。
            # 为匹配HF行为，深度堆叠必须在残差之后添加：(hidden_states + residual) + deepstack
            # 顺序很重要，因为不同张量的加法实际上不满足结合律。
            # 前一层的深度堆叠通过post_residual_addition在当前层开头应用。
            deepstack_embeds = self.get_deepstack_embeds(
                layer_idx - 1, input_deepstack_embeds
            )
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                post_residual_addition=deepstack_embeds,
            )

        # Handle deepstack for the last processed layer if it exists.
        # 处理最后一个处理层的深度堆叠嵌入
        last_deepstack = self.get_deepstack_embeds(
            self.end_layer - 1, input_deepstack_embeds
        )

        if not self.pp_group.is_last_rank:
            # 非最后阶段：返回代理张量供下一阶段使用
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            # 最后阶段：应用层归一化
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    # 应用归一化时同时处理深度堆叠残差
                    hidden_states, _ = self.norm(
                        hidden_states, residual, post_residual_addition=last_deepstack
                    )

        # 没有辅助隐藏状态时直接返回
        if len(aux_hidden_states) == 0:
            return hidden_states

        # 返回主隐藏状态和辅助隐藏状态
        return hidden_states, aux_hidden_states


# Qwen3-VL条件生成模型，整合视觉编码器和语言模型
class Qwen3VLForConditionalGeneration(nn.Module):
    # To ensure correct weight loading and mapping.
    # 权重名称映射器，确保正确的权重加载和映射
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "attn.qkv": "attn.qkv_proj",
        },
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            # transformers v4.52+保存的检查点名称映射
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            # 原始检查点名称映射
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    )

    def __init__(
        self,
        config: Qwen3VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3LLMModel,
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.quant_config = quant_config

        # 是否为视觉编码器启用数据并行
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        # 初始化视觉编码器模型
        self.visual = Qwen3VLMoeVisionModel(
            config.vision_config,
            # NOTE: Qwen3-VL vision encoder currently supports BitsAndBytes 4-bit quantization.
            # Other quantization methods (e.g., GPTQ, AWQ) are untested and may not be supported.
            # 注意：Qwen3-VL视觉编码器目前仅支持BitsAndBytes 4位量化。
            # 其他量化方法（如GPTQ、AWQ）未经测试，可能不支持。
            quant_config=None,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            prefix=add_prefix("model.visual", prefix),
            use_data_parallel=self.use_data_parallel,
        )

        # TODO: make it more elegant
        # 根据语言模型类类型选择配置
        if language_model_cls is Qwen3LLMModel:
            self.config: Qwen3VLConfig = config  # for qwen3-vl
        else:
            # qwen3-omni / qwen3-vl-moe使用文本配置
            self.config = config.text_config  # for qwen3-omni / qwen3-vl-moe
            self.config.encoder_only = getattr(config, "encoder_only", False)
            self.config.language_only = getattr(config, "language_only", False)
            # Propagate tie_word_embeddings from parent config. In transformers
            # v5.5.3+, Qwen3VLMoeTextConfig sets tie_word_embeddings=True by
            # default but the actual model checkpoint has a separate lm_head.
            # The parent Qwen3VLMoeConfig correctly has tie_word_embeddings=False.
            # 从父配置传播tie_word_embeddings设置。
            # transformers v5.5.3+中，Qwen3VLMoeTextConfig默认设置
            # tie_word_embeddings=True，但实际模型检查点有独立的lm_head。
            # 父级Qwen3VLMoeConfig正确设置了tie_word_embeddings=False。
            if hasattr(config, "tie_word_embeddings"):
                self.config.tie_word_embeddings = config.tie_word_embeddings

        # 非仅编码器模式下初始化语言模型和lm_head
        if not hasattr(config, "encoder_only") or not config.encoder_only:
            self.model = language_model_cls(
                config=self.config,
                quant_config=quant_config,
                prefix=add_prefix("model.language_model", prefix),
            )
            if self.pp_group.is_last_rank:
                # 判断是否可以共享embed_tokens和lm_head权重
                if (
                    self.pp_group.world_size == 1
                    and self.config.tie_word_embeddings
                    and not (_is_cpu and _is_cpu_amx_available)
                ):
                    # 共享embed_tokens权重作为lm_head
                    self.lm_head = self.model.embed_tokens
                else:
                    # 使用独立的ParallelLMHead
                    self.lm_head = ParallelLMHead(
                        self.config.vocab_size,
                        self.config.hidden_size,
                        quant_config=quant_config,
                        use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                        prefix=add_prefix("lm_head", prefix),
                    )
            else:
                # 非最后阶段使用占位层
                self.lm_head = PPMissingLayer()
        else:
            # encoder_only mode: no language model, so no lm_head needed
            # 仅编码器模式：没有语言模型，因此不需要lm_head
            self.lm_head = None

        # 检查是否启用多模态旋转位置编码（mrope）
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(self.config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False
        # like {8:0, 16:1, 24:2}, which stands for the captured deepstack features on
        # 8, 16, 24 layer will be merged to 0, 1, 2 layer of decoder output hidden_states
        # 类似 {8:0, 16:1, 24:2}，表示在第8、16、24层捕获的深度堆叠特征
        # 将合并到解码器输出隐藏状态的第0、1、2层

        # deepstack
        # 深度堆叠配置
        self.deepstack_visual_indexes = config.vision_config.deepstack_visual_indexes
        self.num_deepstack_embeddings = len(self.deepstack_visual_indexes)
        # 图像和视频模态默认启用深度堆叠
        self.use_deepstack = {Modality.IMAGE: True, Modality.VIDEO: True}

        # For EAGLE3 support
        # EAGLE3推测解码支持
        self.capture_aux_hidden_states = False

    # 分离深度堆叠嵌入：将拼接的嵌入拆分为主嵌入和深度堆叠嵌入
    def separate_deepstack_embeds(self, embedding):
        assert (
            embedding.shape[-1] % (1 + self.num_deepstack_embeddings) == 0
        ), f"hidden_state of {embedding.shape} should be divisible by ({1 + self.num_deepstack_embeddings})"

        # 主嵌入部分
        separate_index = self.config.hidden_size
        input_embeds = embedding[:, :separate_index]
        # 深度堆叠嵌入部分
        input_deepstack_embeds = embedding[:, separate_index:]
        return input_embeds, input_deepstack_embeds

    # 获取模型起始层索引
    @property
    def start_layer(self) -> int:
        return getattr(getattr(self, "model", None), "start_layer", 0)

    # 获取模型结束层索引
    @property
    def end_layer(self) -> int:
        model = getattr(self, "model", None)
        end_layer = getattr(model, "end_layer", None)
        if end_layer is not None:
            return end_layer
        cfg = getattr(model, "config", None)
        return int(getattr(cfg, "num_hidden_layers", 0))

    # 使用多模态token模式填充输入ID
    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    # 提取图像特征
    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        # 在qwen-vl中，最后一维相同
        # 拼接所有图像项的像素值
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        # 拼接所有图像项的网格尺寸
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()

        if self.use_data_parallel:
            # 数据并行模式：使用分片mrope视觉模型
            return run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values,
                image_grid_thw.tolist(),
                rope_type="rope_3d",
            )
        else:
            # 标准模式：直接调用视觉编码器
            return self.visual(pixel_values, grid_thw=image_grid_thw)

    # 提取视频特征
    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        # 在qwen-vl中，最后一维相同
        # 拼接所有视频项的像素值
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        # 拼接所有视频项的网格尺寸
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        if self.use_data_parallel:
            # 数据并行模式：使用分片mrope视觉模型
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, video_grid_thw.tolist(), rope_type="rope_3d"
            )
        else:
            # 标准模式：直接调用视觉编码器
            video_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)
        return video_embeds

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # LoRA匹配模式：仅对特定层的注意力/MLP投影层应用LoRA
    _lora_pattern = re.compile(
        r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)$"
    )

    # 判断是否应对指定模块应用LoRA
    def should_apply_lora(self, module_name: str) -> bool:
        return bool(self._lora_pattern.match(module_name))

    # 模型前向传播（无梯度）
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """Run forward pass for Qwen3-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
                (Use input_metadata.mrope_positions to replace it)
        """
        # 启用mrope时使用多模态旋转位置编码
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        # 验证mrope的位置编码维度
        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        # 通用多模态嵌入处理：整合文本和视觉/视频嵌入
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            use_deepstack=self.use_deepstack,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        # 如果启用了辅助隐藏状态捕获，解包元组
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                # 计算logits
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                # 计算嵌入（用于嵌入任务）
                return self.pooler(hidden_states, forward_batch)
        else:
            # 非最后阶段直接返回隐藏状态
            return hidden_states

    # 设置DFLASH需要捕获的层
    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return
        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )
        self.capture_aux_hidden_states = True
        # 层ID需要+1，因为SGLang在下一层开头应用残差
        self.model.set_dflash_layers_to_capture([val + 1 for val in layer_ids])

    # 加载模型权重
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # 堆叠参数映射：将分片权重合并
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            # 跳过旋转嵌入的逆频率
            if "rotary_emb.inv_freq" in name:
                continue
            # 替换language_model路径前缀
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            layer_id = get_layer_id(name)

            # Only copy embed_tokens to lm_head when tie_word_embeddings=True
            # For models with tie_word_embeddings=False (e.g. 8B), lm_head has independent weights
            # 仅当tie_word_embeddings=True时复制embed_tokens权重到lm_head
            # 对于tie_word_embeddings=False的模型（如8B），lm_head有独立权重
            if (
                self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
                and self.config.tie_word_embeddings
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head_param, loaded_weight)

            is_visual = "visual" in name
            # 跳过不在当前流水线阶段的语言模型层权重
            if (
                not is_visual
                and layer_id is not None
                and hasattr(self, "model")
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # 跳过视觉模型的堆叠参数（视觉模型有自己的加载逻辑）
                if "visual" in name:
                    continue
                # 将分片名称替换为合并后的参数名称
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip loading visual/language model weights
                # 跳过不属于当前模型的视觉/语言模型权重
                if (
                    self.config.encoder_only or self.config.language_only
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # adapt to VisionAttention
                    # 适配VisionAttention的权重名称
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                    name = name.replace(r"model.visual.", r"visual.")

                try:
                    # Skip loading extra bias for GPTQ models.
                    # 跳过GPTQ模型的额外偏置加载
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name in params_dict.keys():
                        param = params_dict[name]
                    else:
                        continue

                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    # 获取嵌入层和语言模型头的权重
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    # 设置EAGLE3需要捕获辅助隐藏状态的层
    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        self.capture_aux_hidden_states = True
        self.model.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            # 默认选择第2层、中间层和倒数第3层
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            # 层ID需要+1，因为SGLang在下一层开头应用残差
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


# 模型入口类，用于SGLang框架自动发现和注册模型
EntryClass = Qwen3VLForConditionalGeneration
