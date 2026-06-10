# 稀疏池化层实现：将隐藏状态转换为稀疏词汇空间嵌入，
# 通过线性变换+ReLU计算token权重，映射到词汇空间并使用最大池化聚合
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.model_executor.model_runner import ForwardBatch


# 稀疏嵌入输出数据类
@dataclass
class SparseEmbeddingOutput:
    embeddings: torch.Tensor  # [batch_size, vocab_size]


# 稀疏池化器：将隐藏状态池化为稀疏词汇空间嵌入
class SparsePooler(nn.Module):
    """A layer that pools hidden states into sparse vocabulary-space embeddings.

    This layer does the following:
    1. Applies a linear transformation + ReLU to get token-level weights
    2. Maps these weights to vocabulary positions using token IDs
    3. Aggregates weights for repeated tokens using max pooling
    4. Returns sparse embeddings in vocabulary space

    Attributes:
        config: Model configuration containing vocab_size and hidden_size
        sparse_linear: Linear layer for computing token weights
        vocab_size: Size of vocabulary for output embeddings
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        # 验证必需的配置属性
        # Validate required attributes
        if not hasattr(config, "vocab_size"):
            raise AttributeError(
                f"Config {type(config)} missing required 'vocab_size' attribute"
            )
        if not hasattr(config, "hidden_size"):
            raise AttributeError(
                f"Config {type(config)} missing required 'hidden_size' attribute"
            )

        self.vocab_size = config.vocab_size
        # 线性层：将隐藏状态映射到标量权重
        self.sparse_linear = nn.Linear(config.hidden_size, 1)
        self._weights_loaded = False

    # 前向传播：计算稀疏词汇空间嵌入
    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> SparseEmbeddingOutput:
        """
        Forward pass for sparse pooling.

        Args:
            hidden_states: Packed sequence hidden states [total_tokens, hidden_size]
            forward_batch: Batch information with sequence lengths and input_ids

        Returns:
            SparseEmbeddingOutput with embeddings of shape [batch_size, vocab_size]
        """
        if not self._weights_loaded:
            raise ValueError(
                "Sparse pooling weights not loaded. Call load_weights() first"
            )

        # 应用稀疏线性层+ReLU获取token权重
        # Apply sparse linear + ReLU to get token weights
        token_weights = F.relu(self.sparse_linear(hidden_states)).squeeze(
            -1
        )  # [total_tokens]

        # 为打包序列创建批次索引
        # Create batch indices for packed sequences
        batch_indices = torch.repeat_interleave(
            torch.arange(
                len(forward_batch.extend_seq_lens), device=hidden_states.device
            ),
            forward_batch.extend_seq_lens,
        )

        # 初始化稀疏嵌入输出
        # Initialize sparse embedding output
        sparse_embedding = torch.zeros(
            len(forward_batch.extend_seq_lens),
            self.vocab_size,
            dtype=token_weights.dtype,
            device=token_weights.device,
        )

        # 使用scatter_reduce和amax映射到词汇空间
        # Map to vocabulary space using scatter_reduce with amax
        flat_indices = batch_indices * self.vocab_size + forward_batch.input_ids
        sparse_embedding.view(-1).scatter_reduce_(
            0, flat_indices, token_weights, reduce="amax"
        )

        return SparseEmbeddingOutput(embeddings=sparse_embedding)

    # 从状态字典加载权重
    def load_weights(self, state_dict: dict):
        """Load weights from state dict (called by the model)."""
        self.sparse_linear.load_state_dict(state_dict)
        self._weights_loaded = True
