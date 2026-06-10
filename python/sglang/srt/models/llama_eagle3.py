"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.utils import add_prefix

# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""

# LLaMA-EAGLE3 推测解码模型实现：EAGLE v3 版本的草稿模型，
# 相较于 v1 版本，增加了辅助隐藏状态融合、fc 归一化、
# 滑动窗口注意力以及 d2t/t2d 词汇映射等特性。

import copy
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP
from sglang.srt.server_args import get_global_server_args


# EAGLE3 解码器层：继承自 LlamaDecoderLayer，第一层输入维度翻倍以接收拼接特征
class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config=quant_config, prefix=prefix)

        # Input layer concats embeds + target_hidden before qkv (input dim 2x).
        # 第一层是输入层，需要将词嵌入和目标隐藏状态拼接，因此输入维度为 2 倍
        self.is_input_layer = layer_id == 0
        hidden_size = 2 * self.hidden_size if self.is_input_layer else self.hidden_size

        # override qkv
        # 重写 QKV 投影层以适配输入维度变化
        self.self_attn.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        # 根据模型类型选择中间层维度
        if config.model_type == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size

        # EAGLE3 独立的 MLP 层（不同于基础 Llama 的共享 MLP）
        self.mlp = LlamaMLP(
            config.hidden_size, inter_size, config.hidden_act, quant_config, prefix
        )

        # 隐藏状态归一化层
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # 解码器层前向传播：输入层和后续层的处理逻辑不同
    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.is_input_layer:
            # Input layer consumes target hidden states; no carried residual to fuse.
            # 输入层：直接将目标隐藏状态作为残差，归一化后与词嵌入拼接
            residual = hidden_states
            hidden_states = self.hidden_norm(hidden_states)
            embeds = self.input_layernorm(embeds)
            # 拼接词嵌入和归一化后的目标隐藏状态
            hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        else:
            # Fuse the previous layer's MLP residual add into hidden_norm.
            # 后续层：将前一层 MLP 的残差加融合进归一化操作
            hidden_states, residual = self.hidden_norm(hidden_states, residual)

        # Self Attention
        # 自注意力计算
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # 注意力后的层归一化
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        # MLP 前馈网络计算
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# EAGLE3 模型主体：包含词嵌入、辅助隐藏状态融合层、解码器层和最终归一化
class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        # 处理旋转位置编码缩放参数
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_scaling = rope_parameters
        else:
            rope_scaling = getattr(config, "rope_scaling", None)
        # 检测是否启用多模态旋转位置编码（mrope）
        self.is_mrope_enabled = (
            rope_scaling is not None and "mrope_section" in rope_scaling
        )
        # fix rope_scaling for qwen2.5-vl
        if self.is_mrope_enabled:
            rope_scaling["rope_type"] = "default"

        self.vocab_size = config.vocab_size
        # 词嵌入层
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # 目标模型的隐藏维度（可能与草稿模型不同）
        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        # num_aux resolution: explicit attr > eagle_config layer_ids > default 3.
        # 辅助隐藏状态数量：优先使用显式属性，其次从 eagle_config 推断，默认为 3
        self.num_aux_hidden_states = getattr(config, "num_aux_hidden_states", None)
        if self.num_aux_hidden_states is None:
            eagle_config = getattr(config, "eagle_config", None) or {}
            layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
            self.num_aux_hidden_states = len(layer_ids) if layer_ids else 3

        # 融合全连接层：将多个辅助隐藏状态拼接后映射到模型隐藏维度
        self.fc = torch.nn.Linear(
            self.hidden_size_in * self.num_aux_hidden_states,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        # Per-aux RMSNorm before fc; enabled via `fc_norm` or legacy `use_aux_norm` flag.
        # 每个 auxiliary 隐藏状态的归一化层（可选），通过 fc_norm 或 use_aux_norm 启用
        use_fc_norm = getattr(config, "fc_norm", None) or getattr(
            config, "use_aux_norm", False
        )
        if use_fc_norm:
            self.fc_norm = nn.ModuleList(
                [
                    RMSNorm(self.hidden_size_in, eps=config.rms_norm_eps)
                    for _ in range(self.num_aux_hidden_states)
                ]
            )
        else:
            self.fc_norm = None

        # 解码器层列表
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, i, quant_config, prefix)
                for i in range(config.num_hidden_layers)
            ]
        )

        # 最终层归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 是否使用归一化后的输出作为辅助隐藏状态
        self.norm_output = getattr(config, "norm_output", False)

    # 模型前向传播
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # 计算词嵌入
        if input_embeds is None:
            embeds = forward_batch.mm_input_embeds
            # 处理多模态输入：在扩展模式下拼接多模态嵌入和文本嵌入
            if (
                forward_batch.forward_mode.is_extend()
                and forward_batch.contains_mm_inputs()
                and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                assert embeds is not None
                embeds = torch.cat(
                    [embeds[:-1], self.embed_tokens(input_ids[-1].unsqueeze(0))]
                )
            if embeds is None:
                embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        # 多模态旋转位置编码
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        # 获取目标模型的辅助隐藏状态
        hidden_states = forward_batch.spec_info.hidden_states
        # 如果隐藏状态维度与词嵌入不同，需要通过 fc 层映射
        if hidden_states.shape[-1] != embeds.shape[-1]:
            # 对每个 auxiliary 隐藏状态分别归一化后再拼接
            if self.fc_norm is not None:
                chunks = hidden_states.chunk(self.num_aux_hidden_states, dim=-1)
                hidden_states = torch.cat(
                    [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                    dim=-1,
                )
            # 通过全连接层将辅助隐藏状态映射到模型隐藏维度
            hidden_states = self.fc(hidden_states)

        # idle batch
        # 空闲批次直接返回
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        residual = None
        # 逐层通过解码器
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                embeds,
                hidden_states,
                forward_batch,
                residual,
            )

        # 最终层归一化，同时返回用于 logits 计算和辅助状态的两个输出
        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # Draft decode captures pre-norm hidden by default; `norm_output` opts for normed.
        # 草稿解码默认使用归一化前的隐藏状态；norm_output 选项使用归一化后的
        aux = hidden_states_to_logits if self.norm_output else hidden_states_to_aux
        return hidden_states_to_logits, [aux]


# EAGLE3 因果语言模型：用于推测解码的草稿模型（v3 版本）
class LlamaForCausalLMEagle3(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        # Cache draft SWA size from server args once; consumed both by the post-init
        # attention patch below and by `get_attention_sliding_window_size` later.
        # 从服务参数中缓存草稿模型的滑动窗口大小
        self._draft_window_size: Optional[int] = (
            get_global_server_args().speculative_draft_window_size
        )

        # EAGLE3 模型主体
        self.model = LlamaModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        # 设置注意力滑动窗口大小
        if self._draft_window_size is not None:
            for layer in self.model.layers:
                layer.self_attn.attn.sliding_window_size = self._draft_window_size
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        # 根据配置决定是否共享词嵌入和输出头，以及是否从目标模型加载输出头
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            # 如果没有指定草稿词汇大小，则从目标模型加载输出头
            if config.draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        # 创建独立的 logits 处理器，使用草稿词汇大小
        config_ = copy.deepcopy(config)
        config_.vocab_size = (
            config_.draft_vocab_size
        )  # draft logits processor has it's own vocab size
        self.logits_processor = LogitsProcessor(config_)

        # 启用辅助隐藏状态捕获
        self.capture_aux_hidden_states = True
        # 热 token ID 映射（d2t）
        self.hot_token_id = None

    # 加载权重：处理堆叠参数映射和遗留名称兼容
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        # Define the parameter mapping for stacked parameters
        # 堆叠参数映射：将分散的权重合并到融合层中
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Legacy weight names -> new module attribute names (backwards compat).
        # 遗留权重名称到新模块属性名称的映射（向后兼容）
        legacy_name_map = {
            "midlayer": "layers.0",
            "aux_norm_low": "fc_norm.0",
            "aux_norm_mid": "fc_norm.1",
            "aux_norm_high": "fc_norm.2",
        }

        for name, loaded_weight in weights:
            # 替换遗留名称
            for legacy, new in legacy_name_map.items():
                if legacy in name:
                    name = name.replace(legacy, new)

            # d2t: 草稿到目标的 token ID 映射差值
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            # t2d: 目标到草稿的映射，跳过不加载
            if "t2d" in name:
                continue

            # 处理堆叠参数（QKV 融合、gate_up 融合）
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle regular parameters
                # 处理普通参数
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    # 获取热 token ID 映射
    def get_hot_token_id(self):
        return self.hot_token_id

    # 获取注意力滑动窗口大小
    def get_attention_sliding_window_size(self) -> Optional[int]:
        return self._draft_window_size


EntryClass = [LlamaForCausalLMEagle3]
