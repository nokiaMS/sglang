# RoBERTa/XLM-RoBERTa 模型推理实现文件
# 本文件实现了XLM-RoBERTa嵌入模型和序列分类模型
# 支持稀疏池化(SparsePooler)和交叉编码池化(CrossEncodingPooler)
# 包含RoBERTa嵌入层、分类头和位置ID生成等核心组件

# SPDX-License-Identifier: Apache-2.0

import os  # 导入操作系统模块
from typing import Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块

from sglang.srt.layers.pooler import CrossEncodingPooler, Pooler, PoolingType  # 导入池化层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.sparse_pooler import SparsePooler  # 导入稀疏池化
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding  # 导入词表并行嵌入
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.bert import BertEncoder  # 导入BERT编码器
from sglang.srt.utils.hf_transformers_utils import download_from_hf  # 导入HuggingFace下载工具

RobertaConfig = None  # RoBERTa配置延迟导入


# Adapted from transformers
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    """句子级分类头"""

    def __init__(self, config: RobertaConfig):
        """初始化分类头，包含全连接层和输出投影"""
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 全连接层
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)  # 输出投影

    def forward(self, features, **kwargs):
        """分类头前向传播：取CLS令牌、全连接、tanh激活、输出投影"""
        x = features[0, :]  # take <s> token (equiv. to [CLS])  取<s>令牌
        x = self.dense(x)  # 全连接
        x = torch.tanh(x)  # tanh激活
        x = self.out_proj(x)  # 输出投影
        return x


class RobertaEmbedding(nn.Module):
    """RoBERTa嵌入层，包含词嵌入、位置嵌入和令牌类型嵌入"""
    def __init__(self, config: RobertaConfig):
        """初始化RoBERTa嵌入层"""
        super().__init__()
        self.size = config.hidden_size  # 隐藏层大小
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )  # 词嵌入
        self.padding_idx = config.pad_token_id  # 填充令牌ID
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )  # 位置嵌入

        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )  # 令牌类型嵌入
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 层归一化

        self.position_ids = nn.Parameter(
            torch.empty((1, config.max_position_embeddings)),
        )  # 位置ID

        self.position_embedding_type = config.position_embedding_type  # 位置嵌入类型
        if self.position_embedding_type != "absolute":  # 仅支持绝对位置编码
            raise ValueError(
                "Only 'absolute' position_embedding_type" + " is supported"
            )

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        seq_lens: torch.Tensor,  # 序列长度
        position_ids: torch.Tensor,  # 位置ID
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """RoBERTa嵌入前向传播，生成词嵌入+位置嵌入+类型嵌入"""
        input_shape = input_ids.size()  # 输入形状
        inputs_embeds = self.word_embeddings(input_ids)  # 词嵌入

        # Adapted from vllm: https://github.com/vllm-project/vllm/commit/4a18fd14ba4a349291c798a16bf62fa8a9af0b6b/vllm/model_executor/models/roberta.py

        pos_list = []  # 位置列表
        token_list = []  # 令牌列表
        offset = 0  # 偏移量
        for seq_len in seq_lens:  # 遍历每个序列
            pos_list.append(position_ids[offset : offset + seq_len])
            token_list.append(input_ids[offset : offset + seq_len])
            offset += seq_len

        new_pos_list = []
        for positions, tokens in zip(pos_list, token_list):  # 遍历位置和令牌
            # Verify assumption that incoming position are
            # always a sequence from 0 to N.
            expected_pos = torch.arange(
                positions.size()[0], dtype=torch.long, device=inputs_embeds.device
            )
            assert torch.equal(positions, expected_pos)  # 验证位置假设
            new_pos_list.append(
                create_position_ids_from_input_ids(tokens, self.padding_idx)
            )  # 根据输入ID创建位置ID
        position_ids = torch.cat(new_pos_list)  # 拼接位置ID

        # Position embeddings.
        position_embeddings = self.position_embeddings(position_ids)  # 位置嵌入

        token_type_ids = forward_batch.token_type_ids  # 令牌类型ID
        if token_type_ids is None:  # 如果未提供
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=inputs_embeds.device
            )

        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # 令牌类型嵌入
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings  # 三种嵌入相加
        embeddings = self.LayerNorm(embeddings)  # 层归一化
        return embeddings


class XLMRobertaBaseModel(nn.Module):
    """XLM-RoBERTa基础模型，包含嵌入层和编码器"""
    def __init__(
        self,
        *,
        config: RobertaConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
        add_pooling_layer: bool = False,  # 是否添加池化层
    ):
        """初始化XLM-RoBERTa基础模型"""
        super().__init__()

        self.config = config  # 保存配置
        self.embeddings = RobertaEmbedding(config)  # 嵌入层
        self.encoder = BertEncoder(config=config, quant_config=quant_config, prefix="")  # BERT编码器
        self.pooler = (
            Pooler(pooling_type=PoolingType.CLS, normalize=True)
            if add_pooling_layer
            else None
        )  # 池化层

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
        get_embedding: bool = False,  # 是否获取嵌入
    ) -> torch.Tensor:
        """XLM-RoBERTa基础模型前向传播"""
        assert get_embedding == True
        # Your tokenized IDs

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=positions,
            seq_lens=forward_batch.seq_lens,
            forward_batch=forward_batch,
        )  # 嵌入层

        hidden_states = self.encoder(hidden_states, forward_batch=forward_batch)  # 编码器

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重"""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "query", "q"),
            ("qkv_proj", "key", "k"),
            ("qkv_proj", "value", "v"),
        ]

        params_dict = dict(self.named_parameters())  # 参数字典
        for name, loaded_weight in weights:  # 遍历权重
            name = name.replace("self", "self_attn")  # 替换名称
            if self.pooler is None and "pooler" in name:  # 跳过池化层权重
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 堆叠参数

                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)  # 替换名称
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:  # 非堆叠参数
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


# Adapted from transformers
def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=0
):
    """根据输入ID创建位置ID，跳过填充令牌"""
    mask = input_ids.ne(padding_idx).int()  # 创建非填充掩码
    incremental_indices = (
        torch.cumsum(mask, dim=0).type_as(mask) + past_key_values_length
    ) * mask  # 累积索引
    return incremental_indices.long() + padding_idx  # 返回位置ID


class XLMRobertaModel(nn.Module):
    """XLM-RoBERTa嵌入模型，支持稀疏池化"""
    def __init__(
        self,
        *,
        config: RobertaConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
        sparse_head: Optional[str] = None,  # 稀疏头名称
        model_path: Optional[str] = None,  # 模型路径
    ):
        """初始化XLM-RoBERTa模型，配置稀疏或密集池化"""
        super().__init__()
        self.roberta = XLMRobertaBaseModel(
            config=config, quant_config=quant_config, prefix=prefix
        )
        if sparse_head is not None:  # 使用稀疏池化
            self._is_sparse = True
            self._model_path = model_path
            self._sparse_head = sparse_head
            self.pooler = SparsePooler(config=config)
            # Zero out special tokens
            self._special_tokens = [
                config.bos_token_id,
                config.eos_token_id,
                config.pad_token_id,
                # self.config.unk_token_id # not available in the XLMRobertaConfig
            ]
            self._special_tokens = [t for t in self._special_tokens if t is not None]
        else:  # 使用密集池化
            self._is_sparse = False
            self.pooler = Pooler(pooling_type=PoolingType.CLS, normalize=True)

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
        get_embedding: bool = False,  # 是否获取嵌入
    ) -> torch.Tensor:
        """XLM-RoBERTa模型前向传播"""
        hidden_states = self.roberta(
            input_ids, positions, forward_batch, input_embeds, get_embedding
        )  # 基础模型前向
        embeddings = self.pooler(hidden_states, forward_batch)  # 池化

        if self._is_sparse:  # 稀疏模式
            for token_id in self._special_tokens:  # 零化特殊令牌
                embeddings.embeddings[:, token_id] = 0.0
            embeddings.embeddings = embeddings.embeddings.to_sparse()  # 转为稀疏

        return embeddings

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，包括稀疏头权重"""
        self.roberta.load_weights(weights)  # 加载基础模型权重

        if self._is_sparse:  # 加载稀疏头
            sparse_dict = XLMRobertaModel._load_sparse_linear(
                self._model_path, self._sparse_head
            )
            self.pooler.load_weights(sparse_dict)

    @staticmethod
    def _load_sparse_linear(model_path_or_dir: str, sparse_head: str) -> dict:
        """
        Load sparse_head from local dir or HF Hub.
        Returns a state_dict suitable for nn.Linear.load_state_dict().
        """
        """从本地目录或HuggingFace Hub加载稀疏头权重"""
        if os.path.isdir(model_path_or_dir):  # 本地路径
            path = os.path.join(model_path_or_dir, sparse_head)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"'{sparse_head}' not found in {model_path_or_dir}"
                )
        else:
            # remote → use SGLang HF utility
            local_dir = download_from_hf(model_path_or_dir, allow_patterns=sparse_head)  # 远程下载
            path = os.path.join(local_dir, sparse_head)

        state_dict = torch.load(path)  # 加载状态字典
        return state_dict


class XLMRobertaForSequenceClassification(nn.Module):
    """XLM-RoBERTa序列分类模型，用于重排序"""
    def __init__(
        self,
        *,
        config: RobertaConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        """初始化序列分类模型"""
        super().__init__()
        self.roberta = XLMRobertaBaseModel(
            config=config, quant_config=quant_config, prefix=prefix
        )
        self.classifier = RobertaClassificationHead(config)  # 分类头
        self.pooler = CrossEncodingPooler(config, self.classifier, self.roberta.pooler)  # 交叉编码池化

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
        get_embedding: bool = True,  # 是否获取嵌入
    ) -> torch.Tensor:
        """序列分类前向传播"""
        assert (
            get_embedding
        ), "XLMRobertaForSequenceClassification is only used for rerank"

        hidden_states = self.roberta(
            input_ids, positions, forward_batch, input_embeds, get_embedding
        )  # 基础模型前向
        return self.pooler(hidden_states, forward_batch)  # 池化输出

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，分离RoBERTa和分类头权重"""
        self_weights = []

        def weight_filter():
            """权重过滤器，分离roberta和分类头权重"""
            for name, weight in weights:
                if name.startswith("roberta."):
                    yield (name[len("roberta.") :], weight)  # 去掉roberta前缀
                else:
                    self_weights.append((name, weight))  # 分类头权重

        self.roberta.load_weights(weight_filter())  # 加载RoBERTa权重

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in self_weights:  # 加载分类头权重
            if name.startswith("classifier"):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [XLMRobertaModel, XLMRobertaForSequenceClassification]  # 模型注册入口类列表
