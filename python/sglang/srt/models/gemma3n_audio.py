# Gemma3n音频编码器：实现Gemma3n的音频编码器，包含累积组归一化、相对位置编码、局部注意力、子采样卷积投影和Conformer块
import math  # 导入数学模块 # import math module
from typing import Optional, Sequence, Tuple  # 导入类型提示工具 # import type hints

import torch  # 导入PyTorch库 # import PyTorch
import torch.nn as nn  # 导入神经网络模块 # import neural network module
import torch.nn.functional as F  # 导入神经网络函数模块 # import neural network functional module
from transformers import Gemma3nAudioConfig, PreTrainedModel  # 导入Gemma3n音频配置和预训练模型 # import Gemma3n audio config and pretrained model

from sglang.srt.layers.linear import (  # 导入并行线性层 # import parallel linear layers
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置 # import quantization config
from sglang.srt.models.gemma3n_causal import Gemma3nRMSNorm  # 导入Gemma3n RMS归一化层 # import Gemma3n RMS norm layer
from sglang.srt.utils import add_prefix, make_layers  # 导入工具函数 # import utility functions


class Gemma3nCumulativeGroupNorm(nn.Module):  # Gemma3n累积组归一化模块 # Gemma3n cumulative group norm module
    """Applies Group Normalization cumulatively over the time dimension.
    # 沿时间维度累积应用组归一化

    This layer normalizes the input by calculating the mean and variance
    cumulatively over the time dimension (dim 1). The statistics are computed
    over all feature dimensions (specified by `feature_dims` and `num_channels`)
    for elements marked as valid by the optional `mask`.
    # 此层通过沿时间维度(dim 1)累积计算均值和方差来归一化输入。
    # 统计量在所有特征维度上计算，仅对可选mask标记为有效的元素进行。

    If a `mask` is provided (True for valid, False for invalid/padded),
    invalid time steps do not contribute to the statistics calculation, and
    their corresponding output values are zeroed out.
    # 如果提供了mask（True为有效，False为无效/填充），无效时间步不参与统计计算，其对应输出值置零。

    Scale and bias, if enabled, are applied per-channel (last dimension).
    This behavior is similar to JAX's `GroupNormalization` with `num_groups=1`
    and `cumulative=True`.
    # 缩放和偏移（如果启用）按通道（最后维度）应用。此行为类似于JAX的GroupNormalization(num_groups=1, cumulative=True)
    """

    def __init__(  # 初始化方法 # initialization method
        self,
        num_channels: int,  # Number of channels (size of the last dimension) # 通道数（最后维度的大小）
        feature_dims: Sequence[  # Sizes of non-channel feature dimensions, e.g., (H, W) for input [B,T,H,W,C]
            int
        ],  # 非通道特征维度大小，例如输入[B,T,H,W,C]的(H, W)
        eps: float = 1e-3,  # 防止除零的小常数 # small constant to prevent division by zero
    ):
        super().__init__()  # 调用父类初始化 # call parent class init
        self.num_channels = num_channels  # 保存通道数 # save num channels
        self.feature_dims = tuple(feature_dims)  # 保存特征维度 # save feature dims
        self.eps = eps  # 保存epsilon值 # save epsilon

        # Scale parameter depends only on the channel dimension
        # 缩放参数仅依赖于通道维度
        self.weight = nn.Parameter(torch.ones(num_channels))  # 可学习缩放参数 # learnable scale parameter

        # Axes for normalization: all dimensions except Batch (0) and Time (1).
        # For input [B, T, *feature_dims, C], these are dims from 2 onwards.
        # 归一化轴：除批次(0)和时间(1)外的所有维度。对于输入[B, T, *feature_dims, C]，即从2开始的维度
        self.reduction_axes = tuple(range(2, 2 + len(self.feature_dims) + 1))  # 归约轴 # reduction axes

    def forward(  # 前向传播方法 # forward pass method
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Applies cumulative group norm, optionally using a mask.
        # 应用累积组归一化，可选使用mask

        Args:
          x: Input tensor, shape [B, T, *feature_dims, C]. # 输入张量，形状[B, T, *feature_dims, C]
          mask: Optional boolean mask, shape [B, T]. True indicates a valid
            (non-padded) time step. If None, all time steps are considered valid.
          # mask：可选布尔掩码，形状[B, T]。True表示有效（非填充）时间步。None则所有时间步视为有效

        Returns:
          Normalized tensor with the same shape as x. # 与x同形状的归一化张量
        """
        expected_input_suffix = self.feature_dims + (self.num_channels,)  # 期望的输入后缀形状 # expected input suffix shape
        if x.shape[2:] != expected_input_suffix:  # 形状不匹配 # shape mismatch
            raise ValueError(  # 抛出值错误 # raise value error
                f"Input tensor shape suffix {x.shape[2:]} does not match expected"
                f" suffix (feature_dims + num_channels) {expected_input_suffix}"
            )

        input_dtype = x.dtype  # 保存输入数据类型 # save input dtype
        # Calculations are performed in float32 for numerical stability.
        # 计算在float32下进行以确保数值稳定性
        calc_dtype = torch.float32  # 计算数据类型 # computation dtype
        x_calc = x.to(calc_dtype)  # 转换为float32 # convert to float32

        # Prepare a broadcastable mask (`mask_calc`).
        # If no mask is provided, treat all elements as valid
        # (mask_calc is all ones).
        # Otherwise, expand the [B, T] mask to [B, T, 1, ..., 1] for broadcasting.
        # 准备可广播的mask。如果未提供mask，所有元素视为有效（全1）；否则将[B, T]的mask扩展为[B, T, 1, ..., 1]
        mask_calc = torch.ones_like(x_calc, dtype=calc_dtype)  # 默认全1掩码 # default all-ones mask

        # Cumulative Statistics Calculation
        # 累积统计量计算
        # 1. Sum of values over reduction axes at each time step.
        # 1. 每个时间步在归约轴上的值之和
        sum_values_at_t = torch.sum(x_calc, dim=self.reduction_axes, keepdim=True)  # 每步求和 # sum per step
        # 2. Cumulative sum of values over time.
        # 2. 沿时间的值累积和
        cum_sum_values = torch.cumsum(sum_values_at_t, dim=1)  # 累积求和 # cumulative sum

        # 3. Count of valid elements in the normalization group at each time step.
        #    (A "group" here consists of all features at a given Batch, Time).
        # 3. 每个时间步归一化组中有效元素的计数（"组"由给定批次和时间的所有特征组成）
        elements_in_group_at_t = torch.sum(  # 每步有效元素数 # valid elements per step
            mask_calc, dim=self.reduction_axes, keepdim=True
        )
        # 4. Cumulative count of valid elements over time.
        # 4. 沿时间的有效元素累积计数
        cum_count_elements = torch.cumsum(elements_in_group_at_t, dim=1)  # 累积计数 # cumulative count
        # Avoid division by zero if all preceding elements were masked.
        # 如果所有前面元素都被mask，避免除零
        safe_cum_count_elements = torch.clamp(cum_count_elements, min=1.0)  # 安全累积计数 # safe cumulative count

        # 5. Cumulative mean.
        # 5. 累积均值
        cum_mean = cum_sum_values / safe_cum_count_elements  # 累积均值 # cumulative mean

        # 6. Sum of squared differences from the cumulative mean.
        #    Only sum for valid elements: (x_calc - cum_mean)^2 * mask_calc.
        #    Using x_calc here for the difference, as cum_mean already accounts for masking.
        # 6. 与累积均值的平方差之和。仅对有效元素求和：(x_calc - cum_mean)^2 * mask_calc
        squared_diff_from_mean = (x_calc - cum_mean).pow(2)  # 平方差 # squared difference
        sum_sq_diff_at_t = torch.sum(  # 每步平方差之和 # sum of squared diffs per step
            squared_diff_from_mean, dim=self.reduction_axes, keepdim=True
        )

        # 7. Cumulative sum of squared differences over time.
        # 7. 沿时间的平方差累积和
        cum_sum_sq_diff = torch.cumsum(sum_sq_diff_at_t, dim=1)  # 累积平方差 # cumulative squared diff

        # 8. Cumulative variance.
        # 8. 累积方差
        cum_variance = cum_sum_sq_diff / safe_cum_count_elements  # 累积方差 # cumulative variance

        # Normalize the input using the calculated cumulative statistics:
        # (x - E[x]) / sqrt(Var[x] + eps)
        # 使用计算的累积统计量归一化输入：(x - E[x]) / sqrt(Var[x] + eps)
        normalized_x = (x_calc - cum_mean) * torch.rsqrt(cum_variance + self.eps)  # 归一化 # normalize

        # Apply affine transformation (scale and bias) if enabled.
        # Scale and bias are applied per-channel (last dimension).
        # 如果启用，应用仿射变换（缩放和偏置）。缩放和偏置按通道（最后维度）应用
        scale = self.weight.to(calc_dtype)  # 缩放参数转为计算类型 # scale param to compute dtype
        # Reshape for broadcasting: [C] -> [1, ..., 1, C]
        # 重塑以广播：[C] -> [1, ..., 1, C]
        scale_view_shape = [1] * (x.dim() - 1) + [self.num_channels]  # 广播形状 # broadcast shape
        normalized_x = normalized_x * scale.view(scale_view_shape)  # 应用缩放 # apply scale

        # Zero out outputs for time steps that were originally masked (where mask_calc is 0).
        # This ensures padded/invalid positions in the input result in zero output.
        # 将原始mask为0的时间步输出置零，确保填充/无效位置的输出为零
        final_output = normalized_x * mask_calc  # 应用掩码 # apply mask

        return final_output.to(input_dtype)  # 转回原始数据类型 # convert back to original dtype


class Gemma3nAudioRelativePositionEmbedding(nn.Module):  # Gemma3n音频相对位置嵌入模块 # Gemma3n audio relative position embedding module
    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nAudioConfig,  # 音频配置 # audio config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ):
        super().__init__()  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config

        self.num_heads = self.config.conf_num_attention_heads  # 注意力头数 # number of attention heads
        self.channels = self.config.hidden_size  # 隐藏通道数 # hidden channels
        self.head_dim = self.channels // self.num_heads  # 头维度 # head dimension
        self.max_backward = max(0, self.config.conf_attention_context_left - 1)  # 最大向后上下文 # max backward context
        self.max_forward = self.config.conf_attention_context_right  # 最大向前上下文 # max forward context

        self.pos_proj = ColumnParallelLinear(  # 位置投影层 # position projection layer
            self.channels,
            self.num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("pos_proj", prefix),
        )

        min_timescale = 1.0  # 最小时间尺度 # min timescale
        max_timescale = 1.0e4  # 最大时间尺度 # max timescale
        num_timescales = self.channels // 2  # 时间尺度数量 # number of timescales
        log_timescale_increment = math.log(  # 时间尺度对数增量 # log timescale increment
            float(max_timescale) / float(min_timescale)
        ) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(  # 逆时间尺度 # inverse timescales
            torch.arange(num_timescales) * -log_timescale_increment
        )
        self.register_buffer(  # 注册逆时间尺度缓冲区 # register inverse timescales buffer
            "inv_timescales",
            inv_timescales.float().unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def _get_timing_signal_1d_pos(  # 获取1D位置时序信号 # get 1D positional timing signal
        self, position: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        assert position.ndim == 2  # 断言位置为2维 # assert position is 2D
        position = position.float().unsqueeze(-1)  # 增加最后一维 # add last dimension
        scaled_time = position * self.inv_timescales.to(  # 缩放时间 # scale time
            device=position.device, dtype=torch.float32
        )
        timing_signal = torch.cat(  # 拼接正弦和余弦 # concatenate sin and cos
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1
        )
        return timing_signal.type(dtype)  # 返回指定类型 # return in specified dtype

    def _relative_shift(  # 相对偏移方法 # relative shift method
        self,
        term_bd_before_shift: torch.Tensor,  # 偏移前的项 # term before shift
        batch_size: int,  # 批次大小 # batch size
        num_heads: int,  # 头数 # number of heads
        num_query_blocks: int,  # 查询块数 # number of query blocks
        query_block_size: int,  # 查询块大小 # query block size
        key_context_size: int,  # 键上下文大小 # key context size
        max_span_plus_1: int,  # 最大跨度加1 # max span plus 1
    ) -> torch.Tensor:
        """Performs the relative shift."""  # 执行相对偏移 # Performs the relative shift
        pad_amount_last_dim = (key_context_size + 1) - max_span_plus_1  # 最后维度填充量 # last dim padding amount
        padding_tuple = (0, pad_amount_last_dim)  # 填充元组 # padding tuple

        term_bd_padded = F.pad(term_bd_before_shift, padding_tuple)  # 填充 # pad
        term_bd_reshaped = term_bd_padded.reshape(  # 重塑 # reshape
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size * (key_context_size + 1),
            )
        )
        term_bd_sliced = term_bd_reshaped[  # 切片 # slice
            :, :, :, : query_block_size * key_context_size
        ]
        term_bd_shifted = term_bd_sliced.reshape(  # 重塑偏移结果 # reshape shifted result
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size,
                key_context_size,
            )
        )
        return term_bd_shifted  # 返回偏移结果 # return shifted result

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:  # 前向传播方法 # forward pass method
        batch_size, num_query_blocks, query_block_size, num_heads, head_dim = (  # 解包查询形状 # unpack query shape
            queries.shape
        )
        _, _, key_context_size, _, _ = keys.shape  # 解包键形状 # unpack key shape

        pos_indices = torch.arange(  # 位置索引 # position indices
            self.max_backward, -self.max_forward - 1, -1, device=queries.device
        ).unsqueeze(0)
        max_span_plus_1 = pos_indices.shape[1]  # 最大跨度加1 # max span plus 1

        sin_emb_timing_signal = self._get_timing_signal_1d_pos(  # 获取正弦嵌入时序信号 # get sin embedding timing signal
            pos_indices, dtype=queries.dtype
        )
        projected_sin_emb, _ = self.pos_proj(sin_emb_timing_signal)  # 投影正弦嵌入 # project sin embedding
        sin_emb = projected_sin_emb.reshape(  # 重塑正弦嵌入 # reshape sin embedding
            1, max_span_plus_1, self.num_heads, self.head_dim
        ).squeeze(0)

        queries_p = queries.permute(0, 3, 1, 2, 4)  # 重排查询维度 # rearrange query dimensions
        keys_p_t = keys.permute(0, 3, 1, 4, 2)  # 重排键维度并转置 # rearrange key dims and transpose
        term_ac = torch.matmul(queries_p, keys_p_t)  # 计算内容-内容项 # compute content-content term

        q_permuted = queries.permute(0, 3, 1, 2, 4)  # 重排查询 # rearrange queries
        s_permuted = sin_emb.permute(1, 2, 0)  # 重排正弦嵌入 # rearrange sin embedding
        q_reshaped = q_permuted.reshape(  # 重塑查询 # reshape queries
            batch_size, num_heads, num_query_blocks * query_block_size, head_dim
        )
        term_bd_unshifed_matmul = torch.matmul(q_reshaped, s_permuted)  # 计算内容-位置项 # compute content-position term
        term_bd_unshifed = term_bd_unshifed_matmul.reshape(  # 重塑未偏移项 # reshape unshifted term
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            max_span_plus_1,
        )

        term_bd_shifted = self._relative_shift(  # 应用相对偏移 # apply relative shift
            term_bd_unshifed,
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
            max_span_plus_1,
        )

        return term_ac + term_bd_shifted  # 返回内容-内容项加上偏移后的内容-位置项 # return content-content + shifted content-position


class Gemma3nAudioAttention(nn.Module):  # Gemma3n音频注意力模块 # Gemma3n audio attention module
    """Local dot product self-attention for audio."""  # 音频的局部点积自注意力 # Local dot product self-attention for audio

    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nAudioConfig,  # 音频配置 # audio config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ):
        super().__init__()  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config

        self.num_heads = self.config.conf_num_attention_heads  # 注意力头数 # number of attention heads
        self.hidden_size = self.config.hidden_size  # 隐藏层大小 # hidden size
        self.head_dim = self.hidden_size // self.num_heads  # 头维度 # head dimension

        self.chunk_size = self.config.conf_attention_chunk_size  # 注意力块大小 # attention chunk size
        self.max_future_horizon = self.config.conf_attention_context_right  # 最大未来视野 # max future horizon
        self.max_past_horizon = max(0, self.config.conf_attention_context_left - 1)  # 最大过去视野 # max past horizon
        self.attention_logits_soft_cap = self.config.conf_attention_logit_cap  # 注意力logit软上限 # attention logit soft cap
        self.context_size = (  # 上下文大小 # context size
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )

        self.relative_position_embedding = Gemma3nAudioRelativePositionEmbedding(  # 相对位置嵌入 # relative position embedding
            config,
            quant_config,
            prefix=add_prefix("relative_position_embedding", prefix),
        )
        self.per_dim_scale = nn.Parameter(torch.zeros((self.head_dim,)))  # 逐维度缩放参数 # per-dimension scale parameter

        self.qkv_proj = QKVParallelLinear(  # QKV并行投影层 # QKV parallel projection layer
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        q_scale = self.head_dim**-0.5  # 查询缩放因子 # query scale factor
        r_softplus_0 = 1.0 / F.softplus(torch.tensor(0.0))  # softplus(0)的倒数 # reciprocal of softplus(0)
        self.register_buffer(  # 注册缩放缓冲区 # register scale buffer
            "q_scale", (q_scale * r_softplus_0).clone().detach(), persistent=False
        )

        # Create local causal mask
        # 创建局部因果掩码
        lower_causal_mask = torch.tril(  # 下三角因果掩码 # lower triangular causal mask
            torch.ones((self.context_size, self.chunk_size), dtype=torch.bool),
            diagonal=0,
        ).T
        upper_causal_mask = torch.tril(  # 上三角因果掩码 # upper triangular causal mask
            torch.ones((self.chunk_size, self.context_size), dtype=torch.bool),
            diagonal=self.max_past_horizon + self.max_future_horizon,
        )
        local_causal_valid_mask = torch.ones(  # 局部因果有效掩码 # local causal valid mask
            (self.chunk_size, self.context_size), dtype=torch.bool
        )
        local_causal_valid_mask = (  # 组合掩码 # combine masks
            local_causal_valid_mask * lower_causal_mask * upper_causal_mask
        )
        self.register_buffer(  # 注册局部因果有效掩码 # register local causal valid mask
            "local_causal_valid_mask", local_causal_valid_mask, persistent=False
        )

        self.register_buffer(  # 注册软上限缓冲区 # register softcap buffer
            "softcap",
            torch.tensor(self.attention_logits_soft_cap).float(),
            persistent=False,
        )

    def _pad_dim1(  # 对第1维进行填充 # pad dimension 1
        self, x: torch.Tensor, dim10_val: int, dim11_val: int
    ) -> torch.Tensor:
        padding_tuple = [0] * x.ndim * 2  # 初始化填充列表 # initialize padding list
        dim_idx_from_end = x.ndim - 2  # 从末尾计的维度索引 # dimension index from end
        start_idx_for_dim = 2 * dim_idx_from_end  # 该维度的起始索引 # start index for this dim
        padding_tuple[start_idx_for_dim] = dim10_val  # 设置左填充 # set left padding
        padding_tuple[start_idx_for_dim + 1] = dim11_val  # 设置右填充 # set right padding
        return F.pad(x, tuple(padding_tuple))  # 返回填充结果 # return padded result

    def _convert_to_block(self, x: torch.Tensor) -> torch.Tensor:  # 将序列转换为不重叠块 # convert sequence to non-overlapping blocks
        """Turns a sequence to non overlapping blocks."""  # 将序列转换为不重叠块 # Turns a sequence to non overlapping blocks
        shape = x.shape  # 获取形状 # get shape
        b, t = shape[:2]  # 批次大小和序列长度 # batch size and sequence length
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size  # 计算块数 # compute number of blocks

        if (padding_len := num_blocks * self.chunk_size - t) > 0:  # 需要填充 # need padding
            x = self._pad_dim1(x, 0, padding_len)  # 填充序列 # pad sequence

        permute_dims = (b, num_blocks, self.chunk_size) + shape[2:]  # 重排维度 # permute dimensions
        x = x.reshape(permute_dims).contiguous()  # 重塑并确保连续 # reshape and ensure contiguous
        return x  # 返回块化结果 # return blocked result

    def _extract_block_context(self, x: torch.Tensor) -> torch.Tensor:  # 提取块的时序上下文 # extract temporal context for blocks
        """Extracts temporal context for every block."""  # 为每个块提取时序上下文 # Extracts temporal context for every block
        pad_left = self.max_past_horizon  # 左侧填充 # left padding
        pad_right = self.max_future_horizon + self.chunk_size - 1  # 右侧填充 # right padding
        x = self._pad_dim1(x, pad_left, pad_right)  # 填充 # pad

        frame_len = self.context_size  # 帧长度 # frame length
        frame_step = self.chunk_size  # 帧步长 # frame step

        x_unfolded = x.unfold(dimension=1, size=frame_len, step=frame_step)  # 展开提取 # unfold extraction

        if x.ndim > 2 and x_unfolded.ndim > 3:  # 高维输入 # high-dimensional input
            x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)  # 移动维度 # move dimension

        return x_unfolded.contiguous()  # 返回连续张量 # return contiguous tensor

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:  # 前向传播方法 # forward pass method
        # Project to Q, K, V
        # 投影到Q, K, V
        qkv, _ = self.qkv_proj(x)  # QKV投影 # QKV projection
        query_states, key_states, value_states = qkv.chunk(chunks=3, dim=-1)  # 分割QKV # split QKV

        # Reshape
        # 重塑
        query_states = query_states.reshape(  # 重塑查询 # reshape queries
            *x.shape[:-1], self.num_heads, self.head_dim
        ).contiguous()
        key_states = key_states.reshape(  # 重塑键 # reshape keys
            *x.shape[:-1], self.num_heads, self.head_dim
        ).contiguous()
        value_states = value_states.reshape(  # 重塑值 # reshape values
            *x.shape[:-1], self.num_heads, self.head_dim
        ).contiguous()

        # Apply per-dim scale
        # 应用逐维度缩放
        per_dim_scale_sp = F.softplus(self.per_dim_scale)  # softplus缩放 # softplus scale
        broadcast_shape = (1, 1, 1, self.head_dim)  # 广播形状 # broadcast shape
        per_dim_scale_sp_broadcast = per_dim_scale_sp.view(broadcast_shape)  # 调整形状 # adjust shape
        query_states = query_states * self.q_scale * per_dim_scale_sp_broadcast  # 应用缩放 # apply scale

        batch_size, q_time = query_states.shape[:2]  # 批次大小和查询时间 # batch size and query time

        # Convert to blocks
        # 转换为块
        query_blocks = self._convert_to_block(query_states)  # 查询分块 # query blocking
        key_blocks = self._extract_block_context(key_states)  # 键提取上下文 # key context extraction
        value_blocks = self._extract_block_context(value_states)  # 值提取上下文 # value context extraction
        num_query_blocks = query_blocks.shape[1]  # 查询块数 # number of query blocks

        # Create mask for valid positions
        # 创建有效位置掩码
        original_valid_mask = ~mask  # 有效位置掩码（取反） # valid position mask (inverted)
        extracted_valid_mask_blocks = self._extract_block_context(original_valid_mask)  # 提取有效掩码块 # extract valid mask blocks

        if (  # 检查掩码形状 # check mask shape
            extracted_valid_mask_blocks.ndim == 4
            and extracted_valid_mask_blocks.shape[0] == batch_size
            and extracted_valid_mask_blocks.shape[1] == num_query_blocks
            and extracted_valid_mask_blocks.shape[2]
            * extracted_valid_mask_blocks.shape[3]
            == self.context_size
        ):
            extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(  # 重塑掩码 # reshape mask
                batch_size, num_query_blocks, self.context_size
            )

        condition_from_input_validity = extracted_valid_mask_blocks.unsqueeze(  # 输入有效性条件 # input validity condition
            1
        ).unsqueeze(-2)
        condition_from_causality = (  # 因果性条件 # causality condition
            self.local_causal_valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )

        final_condition_for_where = torch.logical_and(  # 组合条件 # combined condition
            condition_from_input_validity,
            condition_from_causality.to(condition_from_input_validity.device),
        )

        # Compute attention scores
        # 计算注意力分数
        logits = self.relative_position_embedding(query_blocks, key_blocks)  # 带相对位置的注意力分数 # attention scores with relative position

        # Apply attention logit softcap
        # 应用注意力logit软上限
        softcap_val = self.softcap.to(logits.device)  # 软上限值 # softcap value
        logits = logits / softcap_val  # 缩放logits # scale logits
        logits = torch.tanh(logits)  # tanh截断 # tanh clipping
        logits = logits * softcap_val  # 恢复缩放 # restore scale

        # Apply the combined mask.
        # final_condition_for_where will broadcast with logits [B,N,U,W,C]
        # 应用组合掩码。final_condition_for_where将与logits [B,N,U,W,C]广播
        logits = torch.where(  # 应用掩码 # apply mask
            final_condition_for_where, logits, torch.finfo(logits.dtype).min
        )

        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(  # 计算softmax概率 # compute softmax probabilities
            dtype=value_blocks.dtype
        )

        # context_vectors is adapted from jax.numpy.einsum("BNuwc,BucNH->BuwNH", ...)
        # 上下文向量适配自jax.numpy.einsum("BNuwc,BucNH->BuwNH", ...)
        b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape  # 解包概率形状 # unpack probability shape
        h_dim = value_blocks.shape[-1]  # 值头维度 # value head dimension
        prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)  # 重排概率 # rearrange probabilities
        v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)  # 重排值 # rearrange values
        result_bmm = torch.bmm(prob_bun, v_bun)  # 批量矩阵乘法 # batch matrix multiplication
        context_vectors = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(  # 重塑上下文向量 # reshape context vectors
            0, 1, 3, 2, 4
        )
        context_vectors = context_vectors.reshape(  # 重塑为输出形状 # reshape to output shape
            (
                batch_size,
                num_query_blocks * self.chunk_size,
                self.num_heads,
                self.head_dim,
            )
        )
        context_vectors = context_vectors[:, :q_time]  # 截取到实际时间步 # truncate to actual time steps

        return context_vectors  # 返回上下文向量 # return context vectors


class Gemma3nAudioSSCPConvBlock(nn.Module):  # Gemma3n子采样卷积投影的卷积块 # Gemma3n SSCP conv block
    """A single convolution block for the SubSampleConvProjection."""  # 子采样卷积投影的单个卷积块 # A single convolution block for the SubSampleConvProjection

    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nAudioConfig,  # 音频配置 # audio config
        idx: int,  # 卷积块索引 # conv block index
        input_freq_dim: int,  # 输入频率维度 # input frequency dimension
        manual_padding: Tuple[int, int, int, int] = (0, 0, 0, 0),  # 手动填充 # manual padding
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ):
        super().__init__()  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config
        self.manual_padding = manual_padding  # 保存手动填充 # save manual padding

        in_channels = 1 if idx == 0 else self.config.sscp_conv_channel_size[idx - 1]  # 输入通道数 # input channels
        out_channels = self.config.sscp_conv_channel_size[idx]  # 输出通道数 # output channels
        kernel_h, kernel_w = self.config.sscp_conv_kernel_size[idx]  # 卷积核大小 # kernel size
        stride_h, stride_w = self.config.sscp_conv_stride_size[idx]  # 步长 # stride

        self.conv = nn.Conv2d(  # 2D卷积层 # 2D convolution layer
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=(0, 0),  # Manual padding is used # 使用手动填充
            bias=False,
        )

        f_in_padded = input_freq_dim + self.manual_padding[0] + self.manual_padding[1]  # 填充后频率维度 # padded frequency dim
        f_out_conv = (f_in_padded - kernel_w) // stride_w + 1  # 卷积后频率维度 # frequency dim after conv

        self.norm = Gemma3nCumulativeGroupNorm(  # 累积组归一化 # cumulative group norm
            num_channels=out_channels,
            feature_dims=(f_out_conv,),
            eps=self.config.sscp_conv_group_norm_eps,
        )

        self.activation = nn.ReLU()  # ReLU激活函数 # ReLU activation

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:  # 前向传播方法 # forward pass method
        audio_encodings_padded = F.pad(  # 手动填充 # manual padding
            audio_encodings, self.manual_padding, mode="constant", value=0.0
        )
        audio_encodings_conv = self.conv(audio_encodings_padded)  # 卷积 # convolution
        x_for_norm = audio_encodings_conv.permute(0, 2, 3, 1).contiguous()  # 转置用于归一化 # transpose for norm
        x_normed = self.norm(x_for_norm)  # 累积组归一化 # cumulative group norm
        audio_encodings_normed = x_normed.permute(0, 3, 1, 2).contiguous()  # 转置回来 # transpose back
        return self.activation(audio_encodings_normed)  # 返回激活后结果 # return activated result


class Gemma3nAudioSubSampleConvProjection(nn.Module):  # Gemma3n子采样卷积投影模块 # Gemma3n subsample conv projection module
    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nAudioConfig,  # 音频配置 # audio config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ):
        super().__init__()  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config

        current_f_for_block_input = config.input_feat_size  # 当前块输入的频率维度 # current block input frequency dim
        calculated_block_padding = []  # 计算的块填充 # calculated block padding
        calculated_f_out_dims = []  # 计算的输出频率维度 # calculated output frequency dims

        for i in range(2):  # Assuming 2 conv layers # 假设有2个卷积层
            kernel_h, kernel_w = config.sscp_conv_kernel_size[i]  # 卷积核大小 # kernel size
            stride_h, stride_w = config.sscp_conv_stride_size[i]  # 步长 # stride

            # Padding for Time (Height for Conv2d) - REVERSE_CAUSAL like
            # 时间维度的填充（Conv2d的高度）- 类似反向因果
            pad_t_top = 0  # 顶部填充 # top padding
            pad_t_bottom = kernel_h - 1  # 底部填充 # bottom padding

            # Frequency Padding (Width for Conv2d)
            # 频率维度填充（Conv2d的宽度）
            pad_f_left = 1  # 左侧填充 # left padding
            pad_f_right = 1  # 右侧填充 # right padding

            manual_padding_tuple = (pad_f_left, pad_f_right, pad_t_top, pad_t_bottom)  # 填充元组 # padding tuple
            calculated_block_padding.append(manual_padding_tuple)  # 添加到列表 # append to list

            f_in_padded = current_f_for_block_input + pad_f_left + pad_f_right  # 填充后输入频率维度 # padded input frequency dim
            f_out_after_conv = (f_in_padded - kernel_w) // stride_w + 1  # 卷积后频率维度 # frequency dim after conv
            calculated_f_out_dims.append(f_out_after_conv)  # 添加到列表 # append to list
            current_f_for_block_input = f_out_after_conv  # 更新当前维度 # update current dim

        self.conv_0 = Gemma3nAudioSSCPConvBlock(  # 第一个卷积块 # first conv block
            idx=0,
            input_freq_dim=config.input_feat_size,
            config=config,
            manual_padding=calculated_block_padding[0],
            quant_config=quant_config,
            prefix=add_prefix("conv_0", prefix),
        )
        self.conv_1 = Gemma3nAudioSSCPConvBlock(  # 第二个卷积块 # second conv block
            idx=1,
            input_freq_dim=calculated_f_out_dims[0],
            config=config,
            manual_padding=calculated_block_padding[1],
            quant_config=quant_config,
            prefix=add_prefix("conv_1", prefix),
        )

        final_c_out = config.sscp_conv_channel_size[-1]  # 最终输出通道数 # final output channels
        final_f_out = calculated_f_out_dims[-1]  # 最终输出频率维度 # final output frequency dim
        self.input_proj_in_features = final_c_out * final_f_out  # 输入投影特征数 # input projection features

        self.input_proj_linear = RowParallelLinear(  # 输入投影线性层 # input projection linear layer
            self.input_proj_in_features,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("input_proj_linear", prefix),
        )

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:  # 前向传播方法 # forward pass method
        audio_encodings_reshaped = audio_encodings.unsqueeze(1)  # 增加通道维 # add channel dim
        x = self.conv_0(audio_encodings_reshaped)  # 通过第一个卷积块 # through first conv block
        x = self.conv_1(x)  # 通过第二个卷积块 # through second conv block
        b, c_out, t_out, f_out = x.shape  # 解包输出形状 # unpack output shape
        x_permuted = x.permute(0, 2, 3, 1).contiguous()  # 转置 # transpose
        output_flattened = x_permuted.view(b, t_out, f_out * c_out)  # 展平频率和通道 # flatten freq and channels
        output, _ = self.input_proj_linear(output_flattened)  # 通过输入投影层 # through input projection layer
        return output  # 返回输出 # return output


class Gemma3nAudioConformerAttention(nn.Module):  # Gemma3n音频Conformer注意力模块 # Gemma3n audio Conformer attention module
    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nAudioConfig,  # 音频配置 # audio config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ):
        super().__init__()  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config

        head_dim = self.config.hidden_size // self.config.conf_num_attention_heads  # 头维度 # head dimension
        self.post_in_shape = (self.config.conf_num_attention_heads, head_dim)  # 后处理输入形状 # post-processing input shape
        self.post_in_features = self.config.hidden_size  # 后处理输入特征数 # post-processing input features

        self.register_buffer(  # 注册梯度裁剪缓冲区 # register gradient clipping buffer
            "gradient_clipping",
            torch.tensor(self.config.gradient_clipping),
            persistent=False,
        )

        self.pre_attn_norm = Gemma3nRMSNorm(self.config.hidden_size)  # 注意力前归一化 # pre-attention norm
        self.attn = Gemma3nAudioAttention(  # 音频注意力层 # audio attention layer
            config, quant_config, prefix=add_prefix("attn", prefix)
        )
        self.post = RowParallelLinear(  # 后处理线性层 # post-processing linear layer
            self.post_in_features,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("post", prefix),
        )
        self.post_norm = Gemma3nRMSNorm(self.config.hidden_size)  # 后处理归一化 # post-processing norm

    def forward(  # 前向传播方法 # forward pass method
        self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> torch.Tensor:
        audio_encodings_input_to_attn = audio_encodings  # 保存输入用于残差 # save input for residual
        audio_encodings = torch.clamp(  # 梯度裁剪 # gradient clipping
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings_norm = self.pre_attn_norm(audio_encodings)  # 注意力前归一化 # pre-attention norm
        audio_encodings_attn_out = self.attn(audio_encodings_norm, audio_mel_mask)  # 注意力计算 # attention computation

        b, t, num_heads, head_dim = audio_encodings_attn_out.shape  # 解包形状 # unpack shape
        audio_encodings_reshaped = audio_encodings_attn_out.reshape(  # 重塑注意力输出 # reshape attention output
            b, t, num_heads * head_dim
        )

        audio_encodings, _ = self.post(audio_encodings_reshaped)  # 后处理线性层 # post linear layer
        audio_encodings = torch.clamp(  # 梯度裁剪 # gradient clipping
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        return audio_encodings_input_to_attn + self.post_norm(audio_encodings)  # 残差连接 # residual connection


class Gemma3nAudioConformerFeedForward(nn.Module):  # Gemma3n音频Conformer前馈模块 # Gemma3n audio Conformer feed-forward module
    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nAudioConfig,  # 音频配置 # audio config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ):
        super().__init__()  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config

        self.register_buffer(  # 注册梯度裁剪缓冲区 # register gradient clipping buffer
            "gradient_clipping",
            torch.tensor(self.config.gradient_clipping),
            persistent=False,
        )

        self.pre_layer_norm = Gemma3nRMSNorm(self.config.hidden_size)  # 前归一化 # pre layer norm
        self.ffw_layer_1 = ColumnParallelLinear(  # 第一前馈层 # first feed-forward layer
            self.config.hidden_size,
            self.config.hidden_size * 4,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("ffw_layer_1", prefix),
        )
        self.ffw_layer_2 = RowParallelLinear(  # 第二前馈层 # second feed-forward layer
            self.config.hidden_size * 4,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("ffw_layer_2", prefix),
        )
        self.post_layer_norm = Gemma3nRMSNorm(self.config.hidden_size)  # 后归一化 # post layer norm
        self.post_layer_scale = torch.tensor(self.config.conf_residual_weight)  # 残差缩放权重 # residual scale weight

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:  # 前向传播方法 # forward pass method
        residual = audio_encodings  # 保存残差 # save residual
        audio_encodings = torch.clamp(  # 梯度裁剪 # gradient clipping
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.pre_layer_norm(audio_encodings)  # 前归一化 # pre layer norm
        audio_encodings, _ = self.ffw_layer_1(audio_encodings)  # 第一前馈层 # first FF layer
        audio_encodings = F.silu(audio_encodings)  # SiLU激活 # SiLU activation
        audio_encodings, _ = self.ffw_layer_2(audio_encodings)  # 第二前馈层 # second FF layer
        audio_encodings = torch.clamp(  # 梯度裁剪 # gradient clipping
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.post_layer_norm(audio_encodings)  # 后归一化 # post layer norm
        return residual + (audio_encodings * self.post_layer_scale)  # 残差连接加缩放 # residual connection with scale


class Gemma3nAudioConformerLightConv1d(nn.Module):  # Gemma3n音频Conformer轻量1D卷积模块 # Gemma3n audio Conformer light 1D conv module
    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nAudioConfig,  # 音频配置 # audio config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ):
        super().__init__()  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config

        self.pre_layer_norm = Gemma3nRMSNorm(  # 前归一化 # pre layer norm
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.linear_start = ColumnParallelLinear(  # 起始线性层 # start linear layer
            self.config.hidden_size,
            self.config.hidden_size * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("linear_start", prefix),
        )

        self.depthwise_conv1d = nn.Conv1d(  # 深度1D卷积 # depthwise 1D convolution
            in_channels=self.config.hidden_size,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.conf_conv_kernel_size,
            stride=1,
            padding=0,  # Manual causal padding # 手动因果填充
            groups=self.config.hidden_size,  # Depthwise # 深度可分离
            bias=False,
        )
        self.register_buffer(  # 注册梯度裁剪缓冲区 # register gradient clipping buffer
            "gradient_clipping",
            torch.tensor(self.config.gradient_clipping),
            persistent=False,
        )
        self.conv_norm = Gemma3nRMSNorm(  # 卷积后归一化 # post-conv norm
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.linear_end = RowParallelLinear(  # 结束线性层 # end linear layer
            self.config.hidden_size,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("linear_end", prefix),
        )

        self.causal_padding = self.config.conf_conv_kernel_size - 1  # 因果填充大小 # causal padding size

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:  # 前向传播方法 # forward pass method
        audio_encodings_residual = audio_encodings  # Save for residual connection # 保存用于残差连接

        audio_encodings = self.pre_layer_norm(audio_encodings)  # 前归一化 # pre layer norm
        audio_encodings, _ = self.linear_start(audio_encodings)  # 起始线性层 # start linear layer
        audio_encodings = F.glu(audio_encodings, dim=-1)  # GLU门控线性单元 # GLU gated linear unit

        # Permute for Conv1d: [B, T, D] -> [B, D, T]
        # 为Conv1d转置：[B, T, D] -> [B, D, T]
        audio_encodings_permuted = audio_encodings.permute(0, 2, 1)  # 转置 # transpose
        # Apply manual causal padding
        # 应用手动因果填充
        audio_encodings_permuted_padded = F.pad(  # 因果填充 # causal padding
            audio_encodings_permuted, (self.causal_padding, 0)
        )
        audio_encodings = self.depthwise_conv1d(audio_encodings_permuted_padded)  # 深度1D卷积 # depthwise 1D conv
        # Permute back: [B, D, T_out] -> [B, T_out, D]
        # 转置回来：[B, D, T_out] -> [B, T_out, D]
        audio_encodings = audio_encodings.permute(0, 2, 1)  # 转置回来 # transpose back
        audio_encodings = torch.clamp(  # 梯度裁剪 # gradient clipping
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.conv_norm(audio_encodings)  # 卷积后归一化 # post-conv norm
        audio_encodings = F.silu(audio_encodings)  # SiLU激活 # SiLU activation
        audio_encodings, _ = self.linear_end(audio_encodings)  # 结束线性层 # end linear layer
        output = audio_encodings + audio_encodings_residual  # 残差连接 # residual connection
        return output  # 返回输出 # return output


class Gemma3nAudioConformerBlock(nn.Module):  # Gemma3n音频Conformer块 # Gemma3n audio Conformer block
    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nAudioConfig,  # 音频配置 # audio config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ):
        super().__init__()  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config

        self.ffw_layer_start = Gemma3nAudioConformerFeedForward(  # 起始前馈层 # start feed-forward layer
            config, quant_config, prefix=add_prefix("ffw_layer_start", prefix)
        )
        self.attention = Gemma3nAudioConformerAttention(  # Conformer注意力层 # Conformer attention layer
            config, quant_config, prefix=add_prefix("attention", prefix)
        )
        self.lconv1d = Gemma3nAudioConformerLightConv1d(  # 轻量1D卷积 # light 1D conv
            config, quant_config, prefix=add_prefix("lconv1d", prefix)
        )
        self.ffw_layer_end = Gemma3nAudioConformerFeedForward(  # 结束前馈层 # end feed-forward layer
            config, quant_config, prefix=add_prefix("ffw_layer_end", prefix)
        )
        self.register_buffer(  # 注册梯度裁剪缓冲区 # register gradient clipping buffer
            "gradient_clipping",
            torch.tensor(self.config.gradient_clipping),
            persistent=False,
        )
        self.norm = Gemma3nRMSNorm(self.config.hidden_size)  # 最终归一化 # final norm

    def forward(  # 前向传播方法 # forward pass method
        self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> torch.Tensor:
        audio_encodings = self.ffw_layer_start(audio_encodings)  # 起始前馈层 # start FF layer
        audio_encodings = self.attention(audio_encodings, audio_mel_mask)  # 注意力层 # attention layer
        validity_mask_for_lconv = ~audio_mel_mask  # True for valid # True表示有效
        audio_encodings_for_lconv_input = (  # 准备卷积输入 # prepare conv input
            audio_encodings
            * validity_mask_for_lconv.unsqueeze(-1).to(audio_encodings.dtype)
        )
        audio_encodings = self.lconv1d(audio_encodings_for_lconv_input)  # 轻量1D卷积 # light 1D conv

        audio_encodings = self.ffw_layer_end(audio_encodings)  # 结束前馈层 # end FF layer
        audio_encodings = torch.clamp(  # 梯度裁剪 # gradient clipping
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        output = self.norm(audio_encodings)  # 最终归一化 # final norm
        return output  # 返回输出 # return output


class Gemma3nAudioEncoder(PreTrainedModel):  # Gemma3n音频编码器类 # Gemma3n audio encoder class
    """A Universal Speech Encoder -- https://arxiv.org/abs/2303.01037"""  # 通用语音编码器 # A Universal Speech Encoder

    config_class = Gemma3nAudioConfig  # 配置类 # config class

    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nAudioConfig,  # 音频配置 # audio config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ):
        super().__init__(config)  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config

        self.subsample_conv_projection = Gemma3nAudioSubSampleConvProjection(  # 子采样卷积投影 # subsample conv projection
            config, quant_config, prefix=add_prefix("subsample_conv_projection", prefix)
        )
        self.conformer = make_layers(  # 创建Conformer层 # create Conformer layers
            config.conf_num_hidden_layers,
            lambda idx, prefix: Gemma3nAudioConformerBlock(
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("conformer", prefix),
        )

    def forward(  # 前向传播方法 # forward pass method
        self, audio_mel: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """Encodes a batch of MELs.
        # 编码一批MEL频谱

        Args:
            audio_mel: a torch.Tensor of shape [batch, num_frames, mel_bins]. # 音频MEL张量，形状[batch, num_frames, mel_bins]
            audio_mel_mask: a torch.BoolTensor of shape [batch, num_frames]. # 音频MEL掩码，形状[batch, num_frames]

        Returns:
            audio_encodings: a torch.Tensor of shape
                `[batch_size, reduced_time_frames, hidden_size]`
            # 音频编码，形状[batch_size, reduced_time_frames, hidden_size]
            audio_mel_mask: a torch.BoolTensor of shape [batch, reduced_time_frames].
            # 音频MEL掩码，形状[batch, reduced_time_frames]
        """
        audio_encodings = self.subsample_conv_projection(
            audio_mel
        )  # audio_encodings: [B, T_sub, D]
        # 子采样卷积投影：audio_encodings: [B, T_sub, D]

        # Subsample the input audio_mel_mask to match the time dimension of audio_encodings (T_sub)
        # 对输入的audio_mel_mask进行子采样以匹配audio_encodings的时间维度(T_sub)
        t_sub = audio_encodings.shape[1]  # 子采样后时间帧数 # subsampled time frames

        time_stride_product = 1  # 时间步长乘积 # time stride product
        for stride_pair_idx in range(len(self.config.sscp_conv_stride_size)):  # 遍历步长 # iterate strides
            time_stride_product *= self.config.sscp_conv_stride_size[stride_pair_idx][0]  # 累乘时间步长 # accumulate time strides

        # Create indices for gathering from the original mask.
        # These indices map to original time steps corresponding to the start of each
        # receptive field in the subsampled output.
        # 创建从原始掩码中收集的索引。这些索引映射到子采样输出中每个感受野起始对应的原始时间步
        indices = (
            torch.arange(t_sub, device=audio_mel_mask.device) * time_stride_product
        )  # 计算索引 # compute indices
        indices = torch.clamp(indices, max=audio_mel_mask.shape[1] - 1)  # 限制索引范围 # clamp indices

        # Expand indices for batch compatibility if B > 1 and indices is 1D.
        # 如果B > 1且索引为1D，扩展索引以兼容批次
        if audio_mel_mask.ndim > 1 and indices.ndim == 1:  # 需要扩展 # need expansion
            indices = indices.unsqueeze(0).expand(
                audio_mel_mask.shape[0], -1
            )  # [B, T_sub]
        elif (
            audio_mel_mask.ndim == indices.ndim
            and audio_mel_mask.shape[0] == 1
            and indices.shape[0] != 1
            and t_sub == indices.shape[0]
        ):  # B=1但索引不为1的情况 # B=1 but indices not 1 case
            # Handle case where B=1 but indices became [T_sub] instead of [1, T_sub]
            # 处理B=1但索引变为[T_sub]而非[1, T_sub]的情况
            indices = indices.unsqueeze(0)  # 增加批次维 # add batch dim

        current_mask = torch.gather(audio_mel_mask, 1, indices)  # [B, T_sub] # 从原始掩码收集 # gather from original mask

        # Fallback: Ensure mask length matches feature length after gather.
        # 回退：确保掩码长度与gather后的特征长度匹配
        if current_mask.shape[1] != t_sub:  # 长度不匹配 # length mismatch
            if current_mask.shape[1] > t_sub:  # 掩码过长 # mask too long
                current_mask = current_mask[:, :t_sub]  # 截断 # truncate
            else:  # current_mask.shape[1] < t_sub # 掩码过短 # mask too short
                padding_needed = t_sub - current_mask.shape[1]  # 需要填充量 # padding needed
                current_mask = F.pad(
                    current_mask, (0, padding_needed), value=True
                )  # Pad with True (masked) # 用True填充（表示被掩码）

        for i, block in enumerate(self.conformer):  # 遍历Conformer块 # iterate Conformer blocks
            audio_encodings = block(
                audio_encodings, current_mask
            )  # Pass the processed mask # 传递处理后的掩码

        if self.config.conf_reduction_factor > 1:  # 需要降采样 # need downsampling
            audio_encodings = audio_encodings[:, :: self.config.conf_reduction_factor]  # 按因子降采样 # downsample by factor
            # Reduce the mask as well
            # 同时降采样掩码
            current_mask = current_mask[:, :: self.config.conf_reduction_factor]  # 降采样掩码 # downsample mask

        # Final masking of audio_encodings based on the final current_mask
        # Ensure current_mask length matches the finally reduced audio_encodings length
        # 基于最终current_mask对audio_encodings进行最终掩码处理
        # 确保current_mask长度与最终降采样后的audio_encodings长度匹配
        if current_mask.shape[1] != audio_encodings.shape[1]:  # 长度不匹配 # length mismatch
            target_len = audio_encodings.shape[1]  # 目标长度 # target length
            mask_current_len = current_mask.shape[1]  # 当前掩码长度 # current mask length
            if target_len > mask_current_len:  # 需要填充 # need padding
                padding_needed = target_len - mask_current_len  # 填充量 # padding amount
                current_mask = F.pad(current_mask, (0, padding_needed), value=True)  # 填充True # pad with True
            elif mask_current_len > target_len:  # mask is longer # 掩码更长 # mask is longer
                current_mask = current_mask[:, :target_len]  # 截断掩码 # truncate mask

        audio_encodings = audio_encodings.masked_fill(current_mask.unsqueeze(-1), 0.0)  # 掩码位置填0 # fill masked positions with 0
        return audio_encodings, current_mask  # 返回编码和掩码 # return encodings and mask
