# LLaMA 嵌入模型实现：基于 LlamaModel 的文本嵌入提取模型，
# 通过池化层（默认取最后一个 token 的隐藏状态并归一化）生成文本嵌入向量，
# 同时兼容 Mistral 嵌入模型。

from typing import Iterable, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.model_executor.model_runner import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaModel
from sglang.srt.utils import add_prefix


# LLaMA 嵌入模型：提取文本的嵌入表示
class LlamaEmbeddingModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # 共享 LlamaModel 作为特征提取器
        self.model = LlamaModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # 池化层：取最后一个 token 的隐藏状态并进行 L2 归一化
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    # 嵌入模型前向传播
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert (
            get_embedding
        ), "LlamaEmbeddingModel / MistralModel is only used for embedding"
        # 获取模型的隐藏状态
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        # 通过池化层提取嵌入向量
        return self.pooler(hidden_states, forward_batch)

    # 加载权重：处理堆叠参数映射
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # 堆叠参数映射：将分散的 QKV 和 gate/up 投影合并
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.model.named_parameters())

        for name, loaded_weight in weights:
            # 跳过旋转位置编码的逆频率和投影器权重
            if "rotary_emb.inv_freq" in name or "projector" in name:
                return
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                # 跳过 ColossalAI 训练产生的缓存余弦/正弦值
                return
            # 跳过视觉塔中不在当前模型参数中的权重
            if name.startswith("model.vision_tower") and name not in params_dict:
                return

            # 处理堆叠参数
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                # 跳过 GPTQ 模型中不在参数字典中的额外偏置
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                # 跳过 GPTQ 模型中不在参数字典中的额外偏置
                if name.endswith(".bias") and name not in params_dict:
                    return
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


# Mistral 嵌入模型：直接复用 LlamaEmbeddingModel 的实现
class MistralModel(LlamaEmbeddingModel):
    pass


EntryClass = [LlamaEmbeddingModel, MistralModel]
