# Copyright 2025 SGLang Team  # 版权所有2025 SGLang团队
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证2.0版授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache许可证URL
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and  # 查看许可证了解管理权限和
# limitations under the License.  # 限制的特定语言
# ==============================================================================  # 分隔线
"""SGLang-native TP-sharded audio encoder for Gemma 4.  # SGLang原生的TP分片音频编码器，用于Gemma 4

Architecture: Conformer-based USM (Universal Speech Model) with SSCP convolution
projection. Adapted from gemma3n_audio.py with Gemma 4 specific changes:  # 架构：基于Conformer的USM（通用语音模型），带有SSCP卷积投影。从gemma3n_audio.py适配，包含Gemma 4特有变更：
  - Activation clamping (clippable linears) on all conformer linears  # 所有conformer线性层上的激活裁剪（可裁剪线性层）
  - per_dim_key_scale in attention  # 注意力中的per_dim_key_scale
  - LayerNorm (not CumulativeGroupNorm) in SSCP convolution blocks  # SSCP卷积块中使用LayerNorm（非CumulativeGroupNorm）
  - Semicausal SSCP padding  # 半因果SSCP填充
  - Mask propagation through SSCP  # 通过SSCP传播掩码
  - Output projection (hidden_size -> output_proj_dims)  # 输出投影（hidden_size -> output_proj_dims）
"""

import math  # 导入数学模块
from typing import Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式神经网络模块
from transformers import Gemma4AudioConfig  # 导入Gemma4音频配置

from sglang.srt.layers.clippable_linear import (  # 导入可裁剪线性层
    ClippableColumnParallelLinear,  # 可裁剪列并行线性层
    ClippableGLUParallelLinear,  # 可裁剪GLU并行线性层
    ClippableQKVParallelLinear,  # 可裁剪QKV并行线性层
    ClippableRowParallelLinear,  # 可裁剪行并行线性层
)
from sglang.srt.layers.dp_attention import (  # 导入数据并行注意力模块
    get_attention_tp_rank,  # 获取注意力TP排名
    get_attention_tp_size,  # 获取注意力TP大小
)
from sglang.srt.layers.layernorm import Gemma4RMSNorm  # 导入Gemma4 RMSNorm
from sglang.srt.layers.linear import (  # 导入线性层模块
    ColumnParallelLinear,  # 列并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.utils import add_prefix, make_layers, set_weight_attrs  # 导入工具函数

# SSCP convolution constants (no longer in config.json, never varied across models)  # SSCP卷积常量（不再在config.json中，跨模型不变）
_SSCP_INPUT_FEAT_SIZE = 128  # SSCP输入特征大小
_SSCP_CONV_KERNEL_SIZES = ((3, 3), (3, 3))  # SSCP卷积核大小
_SSCP_CONV_STRIDE_SIZES = ((2, 2), (2, 2))  # SSCP卷积步长大小

# ---------------------------------------------------------------------------
# Relative Position Embedding  # 相对位置嵌入
# ---------------------------------------------------------------------------


class Gemma4AudioRelativePositionEmbedding(nn.Module):  # Gemma4音频相对位置嵌入类
    def __init__(  # 初始化方法
        self,
        config: Gemma4AudioConfig,  # Gemma4音频配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        tp_size = get_attention_tp_size()  # 获取注意力TP大小
        total_num_heads = config.num_attention_heads  # 总注意力头数
        self.channels = config.hidden_size  # 通道数
        self.head_dim = self.channels // total_num_heads  # 头维度
        self.num_heads = total_num_heads // tp_size  # 每个TP的头数
        self.max_backward = max(0, config.attention_context_left - 1)  # 最大后向上下文
        self.max_forward = config.attention_context_right  # 最大前向上下文

        self.pos_proj = ColumnParallelLinear(  # 位置投影层
            self.channels,  # 输入维度
            config.hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("pos_proj", prefix),  # 添加前缀
        )

        min_timescale = 1.0  # 最小时间尺度
        max_timescale = 1.0e4  # 最大时间尺度
        num_timescales = self.channels // 2  # 时间尺度数量
        log_timescale_increment = math.log(  # 对数时间尺度增量
            float(max_timescale) / float(min_timescale)  # 最大/最小时间尺度的比值取对数
        ) / max(num_timescales - 1, 1)  # 除以时间尺度数减1
        inv_timescales = min_timescale * torch.exp(  # 逆时间尺度
            torch.arange(num_timescales) * -log_timescale_increment  # 乘以负对数增量
        )
        self.register_buffer(  # 注册缓冲区
            "inv_timescales",  # 逆时间尺度
            inv_timescales.float().unsqueeze(0).unsqueeze(0),  # 增加两个维度
            persistent=False,  # 非持久化
        )

    def _get_timing_signal_1d_pos(  # 获取1D位置时序信号
        self, position: torch.Tensor, dtype: torch.dtype  # 位置张量和数据类型
    ) -> torch.Tensor:  # 返回张量
        assert position.ndim == 2  # 断言位置是2维
        position = position.float().unsqueeze(-1)  # 转为浮点并增加维度
        scaled_time = position * self.inv_timescales.to(  # 缩放时间
            device=position.device, dtype=torch.float32  # 转移到相同设备和float32
        )
        timing_signal = torch.cat(  # 拼接时序信号
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1  # 正弦和余弦
        )
        return timing_signal.type(dtype)  # 转换为指定数据类型并返回

    def _relative_shift(  # 相对偏移方法
        self,
        term_bd_before_shift: torch.Tensor,  # 偏移前的BD项张量
        batch_size: int,  # 批次大小
        num_heads: int,  # 头数
        num_query_blocks: int,  # 查询块数
        query_block_size: int,  # 查询块大小
        key_context_size: int,  # 键上下文大小
        max_span_plus_1: int,  # 最大跨度加1
    ) -> torch.Tensor:  # 返回张量
        pad_amount_last_dim = (key_context_size + 1) - max_span_plus_1  # 最后一维填充量
        padding_tuple = (0, pad_amount_last_dim)  # 填充元组

        term_bd_padded = F.pad(term_bd_before_shift, padding_tuple)  # 对BD项进行填充
        term_bd_reshaped = term_bd_padded.reshape(  # 重塑BD项形状
            (
                batch_size,  # 批次大小
                num_heads,  # 头数
                num_query_blocks,  # 查询块数
                query_block_size * (key_context_size + 1),  # 查询块大小乘以键上下文大小加1
            )
        )
        term_bd_sliced = term_bd_reshaped[  # 切片BD项
            :, :, :, : query_block_size * key_context_size  # 取前query_block_size * key_context_size列
        ]
        term_bd_shifted = term_bd_sliced.reshape(  # 重塑偏移后的BD项
            (
                batch_size,  # 批次大小
                num_heads,  # 头数
                num_query_blocks,  # 查询块数
                query_block_size,  # 查询块大小
                key_context_size,  # 键上下文大小
            )
        )
        return term_bd_shifted  # 返回偏移后的BD项

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        batch_size, num_query_blocks, query_block_size, num_heads, head_dim = (  # 解包查询形状
            queries.shape  # 查询张量形状
        )
        _, _, key_context_size, _, _ = keys.shape  # 解包键形状

        pos_indices = torch.arange(  # 位置索引
            self.max_backward, -self.max_forward - 1, -1, device=queries.device  # 从max_backward到-max_forward-1
        ).unsqueeze(0)  # 增加维度
        max_span_plus_1 = pos_indices.shape[1]  # 最大跨度加1

        sin_emb_timing_signal = self._get_timing_signal_1d_pos(  # 获取正弦嵌入时序信号
            pos_indices, dtype=queries.dtype  # 位置索引和查询数据类型
        )
        # pos_proj is a ColumnParallelLinear (no implicit dtype promotion);  # pos_proj是ColumnParallelLinear（无隐式数据类型提升）；
        # project in weight dtype, then cast back to queries' dtype for the matmuls.  # 在权重数据类型中投影，然后转换回查询的数据类型用于矩阵乘法。
        projected_sin_emb, _ = self.pos_proj(  # 投影正弦嵌入
            sin_emb_timing_signal.to(self.pos_proj.weight.dtype)  # 转换为权重数据类型
        )
        projected_sin_emb = projected_sin_emb.to(queries.dtype)  # 转换回查询数据类型
        sin_emb = projected_sin_emb.reshape(  # 重塑正弦嵌入
            1, max_span_plus_1, self.num_heads, self.head_dim  # (1, 跨度+1, 头数, 头维度)
        ).squeeze(0)  # 去除第0维

        queries_p = queries.permute(0, 3, 1, 2, 4)  # 置换查询维度
        keys_p_t = keys.permute(0, 3, 1, 4, 2)  # 置换键维度
        term_ac = torch.matmul(queries_p, keys_p_t)  # 计算AC项（查询-键乘积）

        q_permuted = queries.permute(0, 3, 1, 2, 4)  # 置换查询维度
        s_permuted = sin_emb.permute(1, 2, 0)  # 置换正弦嵌入维度
        q_reshaped = q_permuted.reshape(  # 重塑查询
            batch_size, num_heads, num_query_blocks * query_block_size, head_dim  # (批次, 头, 查询总数, 头维度)
        )
        term_bd_unshifed_matmul = torch.matmul(q_reshaped, s_permuted)  # 计算BD项矩阵乘法
        term_bd_unshifed = term_bd_unshifed_matmul.reshape(  # 重塑BD项
            batch_size,  # 批次大小
            num_heads,  # 头数
            num_query_blocks,  # 查询块数
            query_block_size,  # 查询块大小
            max_span_plus_1,  # 最大跨度加1
        )

        term_bd_shifted = self._relative_shift(  # 应用相对偏移
            term_bd_unshifed,  # 未偏移的BD项
            batch_size,  # 批次大小
            num_heads,  # 头数
            num_query_blocks,  # 查询块数
            query_block_size,  # 查询块大小
            key_context_size,  # 键上下文大小
            max_span_plus_1,  # 最大跨度加1
        )

        return term_ac + term_bd_shifted  # 返回AC项加偏移后的BD项


# ---------------------------------------------------------------------------
# Local Dot-Product Attention (with per_dim_key_scale)  # 局部点积注意力（带per_dim_key_scale）
# ---------------------------------------------------------------------------


class Gemma4AudioAttention(nn.Module):  # Gemma4音频注意力类
    def __init__(  # 初始化方法
        self,
        config: Gemma4AudioConfig,  # Gemma4音频配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        tp_size = get_attention_tp_size()  # 获取注意力TP大小
        total_num_heads = config.num_attention_heads  # 总注意力头数
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.head_dim = self.hidden_size // total_num_heads  # 头维度
        self.num_heads = total_num_heads // tp_size  # 每个TP的头数

        self.chunk_size = config.attention_chunk_size  # 注意力块大小
        self.max_future_horizon = config.attention_context_right  # 最大前向视野
        self.max_past_horizon = max(0, config.attention_context_left - 1)  # 最大后向视野
        self.attention_logits_soft_cap = config.attention_logit_cap  # 注意力logits软上限
        self.context_size = (  # 上下文大小
            self.chunk_size + self.max_past_horizon + self.max_future_horizon  # 块大小加后向加前向视野
        )

        self.relative_position_embedding = Gemma4AudioRelativePositionEmbedding(  # 相对位置嵌入
            config,  # 配置
            quant_config,  # 量化配置
            prefix=add_prefix("relative_position_embedding", prefix),  # 添加前缀
        )
        self.per_dim_scale = nn.Parameter(torch.zeros((self.head_dim,)))  # 每维度缩放参数

        self.qkv = ClippableQKVParallelLinear(  # 可裁剪QKV并行线性层
            hidden_size=self.hidden_size,  # 隐藏层大小
            head_size=self.head_dim,  # 头维度
            total_num_heads=total_num_heads,  # 总头数
            total_num_kv_heads=total_num_heads,  # 总KV头数（与Q头数相同）
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=prefix,  # 前缀
        )

        self.q_scale = (self.head_dim**-0.5) / math.log(2)  # Q缩放因子
        self.k_scale = math.log(1 + math.e) / math.log(2)  # K缩放因子

        self.register_buffer(  # 注册缓冲区
            "softcap",  # 软上限
            torch.tensor(self.attention_logits_soft_cap).float(),  # 转为浮点张量
            persistent=False,  # 非持久化
        )

    # ------ block / context helpers (identical to Gemma3n) ------------------  # 块/上下文辅助方法（与Gemma3n相同）

    def _pad_dim1(  # 填充第1维方法
        self, x: torch.Tensor, dim10_val: int, dim11_val: int  # 输入张量和填充值
    ) -> torch.Tensor:  # 返回张量
        padding_tuple = [0] * x.ndim * 2  # 初始化填充元组
        dim_idx_from_end = x.ndim - 2  # 从末尾算的维度索引
        start_idx_for_dim = 2 * dim_idx_from_end  # 该维度在填充元组中的起始索引
        padding_tuple[start_idx_for_dim] = dim10_val  # 设置左侧填充
        padding_tuple[start_idx_for_dim + 1] = dim11_val  # 设置右侧填充
        return F.pad(x, tuple(padding_tuple))  # 返回填充后的张量

    def _convert_to_block(  # 转换为块格式方法
        self, x: torch.Tensor  # 输入张量
    ) -> torch.Tensor:  # 返回张量
        shape = x.shape  # 获取形状
        b, t = shape[:2]  # 获取批次和时间维度
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size  # 计算块数
        if (padding_len := num_blocks * self.chunk_size - t) > 0:  # 如果需要填充
            x = self._pad_dim1(x, 0, padding_len)  # 填充时间维度
        permute_dims = (b, num_blocks, self.chunk_size) + shape[2:]  # 排列维度
        return x.reshape(permute_dims).contiguous()  # 重塑并返回连续张量

    def _extract_block_context(  # 提取块上下文方法
        self, x: torch.Tensor  # 输入张量
    ) -> torch.Tensor:  # 返回张量
        pad_left = self.max_past_horizon  # 左侧填充
        pad_right = self.max_future_horizon + self.chunk_size - 1  # 右侧填充
        x = self._pad_dim1(x, pad_left, pad_right)  # 填充第1维
        frame_len = self.context_size  # 帧长度
        frame_step = self.chunk_size  # 帧步长
        x_unfolded = x.unfold(dimension=1, size=frame_len, step=frame_step)  # 展开张量
        if x.ndim > 2 and x_unfolded.ndim > 3:  # 如果需要移动维度
            x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)  # 移动最后一维到第2维
        return x_unfolded.contiguous()  # 返回连续张量

    # ------ forward ---------------------------------------------------------  # 前向传播

    def forward(  # 前向传播方法
        self,
        x: torch.Tensor,  # 输入张量
        mask: torch.BoolTensor,  # 掩码张量
        causal_valid_mask: torch.BoolTensor,  # 因果有效掩码
    ) -> torch.Tensor:  # 返回张量
        q, k, v = self.qkv(x)  # 通过QKV线性层获取Q、K、V
        qkv_shape = (*x.shape[:-1], self.num_heads, self.head_dim)  # QKV形状
        query_states = q.float().reshape(qkv_shape).contiguous()  # 重塑查询状态为float
        key_states = k.float().reshape(qkv_shape).contiguous()  # 重塑键状态为float
        value_states = v.float().reshape(qkv_shape).contiguous()  # 重塑值状态为float

        per_dim_scale_sp = F.softplus(self.per_dim_scale)  # 对每维度缩放取softplus
        broadcast_shape = (1, 1, 1, self.head_dim)  # 广播形状
        query_states = (  # 缩放查询状态
            query_states * self.q_scale * per_dim_scale_sp.view(broadcast_shape)  # 乘以Q缩放和每维度缩放
        )

        key_states = key_states * self.k_scale  # 缩放键状态

        batch_size, q_time = query_states.shape[:2]  # 获取批次大小和查询时间步

        query_blocks = self._convert_to_block(query_states)  # 将查询转换为块格式
        key_blocks = self._extract_block_context(key_states)  # 提取键的块上下文
        value_blocks = self._extract_block_context(value_states)  # 提取值的块上下文
        num_query_blocks = query_blocks.shape[1]  # 查询块数

        original_valid_mask = ~mask  # 原始有效掩码（取反mask）
        extracted_valid_mask_blocks = self._extract_block_context(original_valid_mask)  # 提取有效掩码块

        if (  # 如果提取的有效掩码块维度正确
            extracted_valid_mask_blocks.ndim == 4  # 是4维
            and extracted_valid_mask_blocks.shape[0] == batch_size  # 第0维是批次大小
            and extracted_valid_mask_blocks.shape[1] == num_query_blocks  # 第1维是查询块数
            and extracted_valid_mask_blocks.shape[2]  # 第2维
            * extracted_valid_mask_blocks.shape[3]  # 乘以第3维
            == self.context_size  # 等于上下文大小
        ):
            extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(  # 重塑有效掩码块
                batch_size, num_query_blocks, self.context_size  # (批次, 查询块, 上下文)
            )

        condition_from_input_validity = extracted_valid_mask_blocks.unsqueeze(  # 输入有效性条件
            1  # 在第1维增加维度
        ).unsqueeze(-2)  # 在倒数第2维增加维度
        condition_from_causality = (  # 因果性条件
            causal_valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 增加三个维度
        )

        final_condition_for_where = torch.logical_and(  # 最终条件：逻辑与
            condition_from_input_validity,  # 输入有效性条件
            condition_from_causality.to(condition_from_input_validity.device),  # 因果性条件（转换设备）
        )

        logits = self.relative_position_embedding(query_blocks, key_blocks)  # 计算相对位置嵌入logits

        softcap_val = self.softcap.to(logits.device)  # 获取软上限值
        logits = logits / softcap_val  # 除以软上限
        logits = torch.tanh(logits)  # 应用tanh
        logits = logits * softcap_val  # 乘回软上限（实现软裁剪）

        logits = torch.where(  # 根据条件选择logits
            final_condition_for_where,  # 有效位置
            logits,  # 保留原始logits
            self.config.attention_invalid_logits_value,  # 无效位置使用配置值
        )

        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(  # 计算softmax概率
            dtype=value_blocks.dtype  # 转换为值块的数据类型
        )

        b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape  # 解包概率形状
        h_dim = value_blocks.shape[-1]  # 获取值的头维度
        prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)  # 重排概率
        v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)  # 重排值
        result_bmm = torch.bmm(prob_bun, v_bun)  # 批量矩阵乘法
        context_vectors = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(  # 重排结果
            0, 1, 3, 2, 4  # 置换维度
        )
        context_vectors = context_vectors.reshape(  # 重塑上下文向量
            batch_size,  # 批次大小
            num_query_blocks * self.chunk_size,  # 总查询长度
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
        )
        context_vectors = context_vectors[:, :q_time]  # 截取到原始查询长度
        return context_vectors  # 返回上下文向量


# ---------------------------------------------------------------------------
# SSCP (Sub-Sample Convolution Projection)  # SSCP（子采样卷积投影）
# ---------------------------------------------------------------------------


class Gemma4AudioSSCPConvBlock(nn.Module):  # Gemma4音频SSCP卷积块类
    """Single 2D conv block with LayerNorm and semicausal padding."""  # 带LayerNorm和半因果填充的2D卷积块

    def __init__(  # 初始化方法
        self,
        config: Gemma4AudioConfig,  # Gemma4音频配置
        idx: int,  # 卷积块索引
        input_freq_dim: int,  # 输入频率维度
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        conv_channels = config.subsampling_conv_channels  # 子采样卷积通道数
        in_channels = 1 if idx == 0 else conv_channels[idx - 1]  # 输入通道数
        out_channels = conv_channels[idx]  # 输出通道数
        kernel_t, kernel_f = _SSCP_CONV_KERNEL_SIZES[idx]  # 时间和频率卷积核大小
        stride_t, stride_f = _SSCP_CONV_STRIDE_SIZES[idx]  # 时间和频率步长
        self.time_stride = stride_t  # 时间步长

        # Semicausal padding (hardcoded — streaming is not supported)  # 半因果填充（硬编码——不支持流式处理）
        pad_t_top = kernel_t // 2  # 时间上方填充
        pad_t_bottom = kernel_t // 2  # 时间下方填充

        pad_f_left = 1  # 频率左侧填充
        pad_f_right = 1  # 频率右侧填充

        self.manual_padding = (pad_f_left, pad_f_right, pad_t_top, pad_t_bottom)  # 手动填充参数

        self.conv = nn.Conv2d(  # 2D卷积层
            in_channels=in_channels,  # 输入通道
            out_channels=out_channels,  # 输出通道
            kernel_size=(kernel_t, kernel_f),  # 卷积核大小
            stride=(stride_t, stride_f),  # 步长
            padding=(0, 0),  # 填充（手动填充）
            bias=False,  # 无偏置
        )

        f_in_padded = input_freq_dim + pad_f_left + pad_f_right  # 填充后的频率维度
        self.f_out_conv = (f_in_padded - kernel_f) // stride_f + 1  # 卷积后频率维度

        self.norm = nn.LayerNorm(  # LayerNorm归一化层
            [out_channels],  # 归一化通道数
            eps=config.rms_norm_eps,  # epsilon值
            elementwise_affine=True,  # 逐元素仿射
            bias=False,  # 无偏置
        )
        self.activation = nn.ReLU()  # ReLU激活函数

    def forward(  # 前向传播方法
        self, audio_encodings: torch.Tensor, audio_mel_mask: torch.Tensor  # 音频编码和mel掩码
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回元组
        mask_for_fill = audio_mel_mask.unsqueeze(1).unsqueeze(-1)  # 创建填充掩码
        audio_encodings = audio_encodings.masked_fill(mask_for_fill, 0.0)  # 将掩码位置填0

        audio_encodings_padded = F.pad(  # 手动填充
            audio_encodings, self.manual_padding, mode="constant", value=0.0  # 常量填充0
        ).to(self.conv.weight.dtype)  # 转换为卷积权重数据类型
        audio_encodings_conv = self.conv(audio_encodings_padded)  # 卷积操作

        output_mask = audio_mel_mask[:, :: self.time_stride][  # 下采样掩码
            :, : audio_encodings_conv.shape[2]  # 截取到卷积输出时间长度
        ]

        x = audio_encodings_conv.permute(0, 2, 3, 1)  # 置换维度用于LayerNorm
        x_normed = self.norm(x)  # LayerNorm归一化
        audio_encodings_normed = x_normed.permute(0, 3, 1, 2).contiguous()  # 置换回来
        return self.activation(audio_encodings_normed), output_mask  # 返回激活后的结果和掩码


class Gemma4AudioSubSampleConvProjection(nn.Module):  # Gemma4音频子采样卷积投影类
    def __init__(  # 初始化方法
        self,
        config: Gemma4AudioConfig,  # Gemma4音频配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        conv_channels = config.subsampling_conv_channels  # 子采样卷积通道数

        current_f = _SSCP_INPUT_FEAT_SIZE  # 当前频率维度
        calculated_f_out_dims = []  # 计算的输出频率维度列表

        for i in range(2):  # 遍历两个卷积块
            kernel_h, kernel_w = _SSCP_CONV_KERNEL_SIZES[i]  # 卷积核高度和宽度
            stride_h, stride_w = _SSCP_CONV_STRIDE_SIZES[i]  # 步长高度和宽度

            pad_f_left = 1  # 频率左侧填充
            pad_f_right = 1  # 频率右侧填充
            f_in_padded = current_f + pad_f_left + pad_f_right  # 填充后的输入频率维度
            f_out = (f_in_padded - kernel_w) // stride_w + 1  # 输出频率维度
            calculated_f_out_dims.append(f_out)  # 添加到列表
            current_f = f_out  # 更新当前频率维度

        self.conv_0 = Gemma4AudioSSCPConvBlock(  # 第一个SSCP卷积块
            idx=0,  # 索引为0
            input_freq_dim=_SSCP_INPUT_FEAT_SIZE,  # 输入频率维度
            config=config,  # 配置
        )
        self.conv_1 = Gemma4AudioSSCPConvBlock(  # 第二个SSCP卷积块
            idx=1,  # 索引为1
            input_freq_dim=calculated_f_out_dims[0],  # 输入频率维度为第一个块的输出
            config=config,  # 配置
        )

        final_c_out = conv_channels[-1]  # 最终输出通道数
        final_f_out = calculated_f_out_dims[-1]  # 最终输出频率维度
        self.input_proj_in_features = final_c_out * final_f_out  # 输入投影特征数

        self.input_proj_linear = RowParallelLinear(  # 输入投影线性层
            self.input_proj_in_features,  # 输入特征数
            config.hidden_size,  # 输出隐藏层大小
            bias=False,  # 无偏置
            input_is_parallel=False,  # 输入不并行
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("input_proj_linear", prefix),  # 添加前缀
        )

    def forward(  # 前向传播方法
        self, audio_encodings: torch.Tensor, audio_mel_mask: torch.Tensor  # 音频编码和mel掩码
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回元组
        audio_encodings_reshaped = audio_encodings.unsqueeze(1)  # 增加通道维度
        x, mask = self.conv_0(audio_encodings_reshaped, audio_mel_mask)  # 通过第一个卷积块
        x, mask = self.conv_1(x, mask)  # 通过第二个卷积块
        b, c_out, t_out, f_out = x.shape  # 解包形状
        x_permuted = x.permute(0, 2, 3, 1).contiguous()  # 置换维度
        output_flattened = x_permuted.reshape(b, t_out, f_out * c_out)  # 展平频率和通道维度
        output, _ = self.input_proj_linear(output_flattened)  # 通过投影线性层
        return output, mask  # 返回输出和掩码


# ---------------------------------------------------------------------------
# Conformer Blocks  # Conformer块
# ---------------------------------------------------------------------------


class Gemma4AudioConformerAttention(nn.Module):  # Gemma4音频Conformer注意力类
    def __init__(  # 初始化方法
        self,
        config: Gemma4AudioConfig,  # Gemma4音频配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.post_in_features = config.hidden_size  # 后投影输入特征数

        self.register_buffer(  # 注册缓冲区
            "gradient_clipping",  # 梯度裁剪值
            torch.tensor(config.gradient_clipping),  # 转为张量
            persistent=False,  # 非持久化
        )

        self.pre_attn_norm = Gemma4RMSNorm(config.hidden_size, scale_shift=0.0)  # 注意力前归一化
        self.attn = Gemma4AudioAttention(  # 音频注意力模块
            config, quant_config, prefix=add_prefix("attn", prefix)  # 配置和前缀
        )
        self.post = ClippableRowParallelLinear(  # 可裁剪后投影线性层
            self.post_in_features,  # 输入特征数
            config.hidden_size,  # 输出特征数
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("post", prefix),  # 添加前缀
        )
        self.post_norm = Gemma4RMSNorm(config.hidden_size, scale_shift=0.0)  # 后投影归一化

    def forward(  # 前向传播方法
        self,
        audio_encodings: torch.Tensor,  # 音频编码
        audio_mel_mask: torch.BoolTensor,  # mel掩码
        causal_valid_mask: torch.BoolTensor,  # 因果有效掩码
    ) -> torch.Tensor:  # 返回张量
        audio_encodings_input_to_attn = audio_encodings  # 保存原始输入用于残差连接
        audio_encodings = torch.clamp(  # 裁剪音频编码
            audio_encodings, -self.gradient_clipping, self.gradient_clipping  # 限制在[-梯度裁剪, 梯度裁剪]范围内
        )
        audio_encodings_norm = self.pre_attn_norm(audio_encodings)  # 注意力前归一化
        audio_encodings_attn_out = self.attn(  # 通过注意力模块
            audio_encodings_norm, audio_mel_mask, causal_valid_mask  # 归一化编码，掩码和因果掩码
        )

        b, t, num_heads, head_dim = audio_encodings_attn_out.shape  # 解包注意力输出形状
        audio_encodings_reshaped = audio_encodings_attn_out.reshape(  # 重塑注意力输出
            b, t, num_heads * head_dim  # 合并头和头维度
        ).to(dtype=audio_encodings_input_to_attn.dtype)  # 转换回原始数据类型

        audio_encodings = self.post(audio_encodings_reshaped)  # 通过后投影线性层
        audio_encodings = torch.clamp(  # 裁剪投影输出
            audio_encodings, -self.gradient_clipping, self.gradient_clipping  # 限制范围
        )
        return audio_encodings_input_to_attn + self.post_norm(audio_encodings)  # 返回残差连接结果


class Gemma4AudioConformerFeedForward(nn.Module):  # Gemma4音频Conformer前馈网络类
    def __init__(  # 初始化方法
        self,
        config: Gemma4AudioConfig,  # Gemma4音频配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        self.register_buffer(  # 注册缓冲区
            "gradient_clipping",  # 梯度裁剪值
            torch.tensor(config.gradient_clipping),  # 转为张量
            persistent=False,  # 非持久化
        )

        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, scale_shift=0.0)  # 前层归一化
        self.ffw_layer_1 = ClippableColumnParallelLinear(  # 可裁剪第一前馈层
            config.hidden_size,  # 输入维度
            config.hidden_size * 4,  # 输出维度（4倍扩展）
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("ffw_layer_1", prefix),  # 添加前缀
        )
        self.ffw_layer_2 = ClippableRowParallelLinear(  # 可裁剪第二前馈层
            config.hidden_size * 4,  # 输入维度
            config.hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("ffw_layer_2", prefix),  # 添加前缀
        )
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size, scale_shift=0.0)  # 后层归一化
        self.post_layer_scale = config.residual_weight  # 残差缩放权重

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        residual = audio_encodings  # 保存残差
        audio_encodings = torch.clamp(  # 裁剪音频编码
            audio_encodings, -self.gradient_clipping, self.gradient_clipping  # 限制范围
        )
        audio_encodings = self.pre_layer_norm(audio_encodings)  # 前层归一化
        audio_encodings = self.ffw_layer_1(audio_encodings)  # 通过第一前馈层
        audio_encodings = F.silu(audio_encodings)  # SiLU激活
        audio_encodings = self.ffw_layer_2(audio_encodings)  # 通过第二前馈层
        audio_encodings = torch.clamp(  # 裁剪输出
            audio_encodings, -self.gradient_clipping, self.gradient_clipping  # 限制范围
        )
        audio_encodings = self.post_layer_norm(audio_encodings)  # 后层归一化
        return residual + (audio_encodings * self.post_layer_scale)  # 返回带残差和缩放的结果


class Gemma4AudioConformerLightConv1d(nn.Module):  # Gemma4音频Conformer轻量1D卷积类
    def __init__(  # 初始化方法
        self,
        config: Gemma4AudioConfig,  # Gemma4音频配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.causal_padding = config.conv_kernel_size - 1  # 因果填充大小
        tp_size = get_attention_tp_size()  # 获取注意力TP大小
        hidden_per_tp = config.hidden_size // tp_size  # 每个TP的隐藏大小

        self.register_buffer(  # 注册缓冲区
            "gradient_clipping",  # 梯度裁剪值
            torch.tensor(config.gradient_clipping),  # 转为张量
            persistent=False,  # 非持久化
        )

        self.pre_layer_norm = Gemma4RMSNorm(  # 前层归一化
            config.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0  # 隐藏大小和epsilon
        )
        self.linear_start = ClippableGLUParallelLinear(  # 可裁剪GLU并行线性层
            config.hidden_size,  # 输入维度
            config.hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("linear_start", prefix),  # 添加前缀
        )
        self.depthwise_conv1d = nn.Conv1d(  # 深度1D卷积
            in_channels=hidden_per_tp,  # 输入通道数
            out_channels=hidden_per_tp,  # 输出通道数
            kernel_size=config.conv_kernel_size,  # 卷积核大小
            stride=1,  # 步长为1
            padding=0,  # 无填充（手动填充）
            groups=hidden_per_tp,  # 分组数等于通道数（深度卷积）
            bias=False,  # 无偏置
        )
        self.conv_norm = Gemma4RMSNorm(  # 卷积后归一化
            hidden_per_tp, eps=config.rms_norm_eps, scale_shift=0.0  # 每TP隐藏大小和epsilon
        )

        tp_rank = get_attention_tp_rank()  # 获取注意力TP排名

        def _shard_dim0(param, loaded_weight, _rank=tp_rank, _tp=tp_size):  # 分片第0维函数
            shard = param.shape[0]  # 获取分片大小
            loaded_weight = loaded_weight.narrow(0, _rank * shard, shard)  # 窄化权重到当前分片
            param.data.copy_(loaded_weight)  # 复制权重数据

        set_weight_attrs(self.depthwise_conv1d.weight, {"weight_loader": _shard_dim0})  # 设置深度卷积权重加载属性
        set_weight_attrs(self.conv_norm.weight, {"weight_loader": _shard_dim0})  # 设置归一化权重加载属性

        self.linear_end = ClippableRowParallelLinear(  # 可裁剪行并行线性层
            config.hidden_size,  # 输入维度
            config.hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("linear_end", prefix),  # 添加前缀
        )

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        audio_encodings_residual = audio_encodings  # 保存残差

        audio_encodings = self.pre_layer_norm(audio_encodings)  # 前层归一化
        audio_encodings = self.linear_start(audio_encodings)  # 通过起始线性层

        audio_encodings_permuted = audio_encodings.permute(0, 2, 1)  # 置换维度用于1D卷积
        audio_encodings_permuted_padded = F.pad(  # 因果填充
            audio_encodings_permuted, (self.causal_padding, 0)  # 左侧填充
        )
        audio_encodings = self.depthwise_conv1d(audio_encodings_permuted_padded)  # 深度1D卷积
        audio_encodings = audio_encodings.permute(0, 2, 1)  # 置换回原始维度
        audio_encodings = torch.clamp(  # 裁剪
            audio_encodings, -self.gradient_clipping, self.gradient_clipping  # 限制范围
        )
        audio_encodings = self.conv_norm(audio_encodings)  # 卷积后归一化
        audio_encodings = F.silu(audio_encodings)  # SiLU激活
        audio_encodings = self.linear_end(audio_encodings)  # 通过结束线性层
        return audio_encodings + audio_encodings_residual  # 返回带残差的结果


class Gemma4AudioConformerBlock(nn.Module):  # Gemma4音频Conformer块类
    def __init__(  # 初始化方法
        self,
        config: Gemma4AudioConfig,  # Gemma4音频配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        self.ffw_layer_start = Gemma4AudioConformerFeedForward(  # 起始前馈层
            config, quant_config, prefix=add_prefix("ffw_layer_start", prefix)  # 配置和前缀
        )
        self.attention = Gemma4AudioConformerAttention(  # Conformer注意力
            config, quant_config, prefix=add_prefix("attention", prefix)  # 配置和前缀
        )
        self.lconv1d = Gemma4AudioConformerLightConv1d(  # 轻量1D卷积
            config, quant_config, prefix=add_prefix("lconv1d", prefix)  # 配置和前缀
        )
        self.ffw_layer_end = Gemma4AudioConformerFeedForward(  # 结束前馈层
            config, quant_config, prefix=add_prefix("ffw_layer_end", prefix)  # 配置和前缀
        )
        self.register_buffer(  # 注册缓冲区
            "gradient_clipping",  # 梯度裁剪值
            torch.tensor(config.gradient_clipping),  # 转为张量
            persistent=False,  # 非持久化
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, scale_shift=0.0)  # 最终归一化

    def forward(  # 前向传播方法
        self,
        audio_encodings: torch.Tensor,  # 音频编码
        audio_mel_mask: torch.BoolTensor,  # mel掩码
        causal_valid_mask: torch.BoolTensor,  # 因果有效掩码
    ) -> torch.Tensor:  # 返回张量
        audio_encodings = self.ffw_layer_start(audio_encodings)  # 通过起始前馈层
        audio_encodings = self.attention(  # 通过注意力层
            audio_encodings, audio_mel_mask, causal_valid_mask  # 编码，掩码和因果掩码
        )
        validity_mask_for_lconv = ~audio_mel_mask  # 轻量卷积的有效掩码（取反mel掩码）
        audio_encodings_for_lconv_input = (  # 准备轻量卷积输入
            audio_encodings  # 音频编码
            * validity_mask_for_lconv.unsqueeze(-1).to(audio_encodings.dtype)  # 乘以有效掩码
        )
        audio_encodings = self.lconv1d(audio_encodings_for_lconv_input)  # 通过轻量1D卷积

        audio_encodings = self.ffw_layer_end(audio_encodings)  # 通过结束前馈层
        audio_encodings = torch.clamp(  # 裁剪输出
            audio_encodings, -self.gradient_clipping, self.gradient_clipping  # 限制范围
        )
        return self.norm(audio_encodings)  # 返回归一化后的结果


# ---------------------------------------------------------------------------
# Top-level Encoder  # 顶层编码器
# ---------------------------------------------------------------------------


class Gemma4AudioEncoder(nn.Module):  # Gemma4音频编码器类
    """SGLang-native TP-sharded Gemma 4 audio encoder (USM Conformer + SSCP)."""  # SGLang原生的TP分片Gemma 4音频编码器（USM Conformer + SSCP）

    def __init__(  # 初始化方法
        self,
        config: Gemma4AudioConfig,  # Gemma4音频配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(  # 子采样卷积投影
            config, quant_config, prefix=add_prefix("subsample_conv_projection", prefix)  # 配置和前缀
        )
        self.conformer = make_layers(  # 创建Conformer层
            config.num_hidden_layers,  # 隐藏层数
            lambda idx, prefix: Gemma4AudioConformerBlock(  # 每层是一个Conformer块
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                prefix=prefix,  # 前缀
            ),
            prefix=add_prefix("conformer", prefix),  # 添加前缀
        )

        if config.output_proj_dims is not None:  # 如果配置了输出投影维度
            self.output_proj = RowParallelLinear(  # 输出投影层
                config.hidden_size,  # 输入维度
                config.output_proj_dims,  # 输出维度
                bias=True,  # 有偏置
                input_is_parallel=False,  # 输入不并行
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("output_proj", prefix),  # 添加前缀
            )
        else:  # 没有配置输出投影维度
            self.output_proj = None  # 无输出投影

        # Precompute causal_valid_mask — depends only on static config values.  # 预计算因果有效掩码——仅依赖于静态配置值
        chunk_size = config.attention_chunk_size  # 注意力块大小
        max_future_horizon = config.attention_context_right  # 最大前向视野
        max_past_horizon = max(0, config.attention_context_left - 1)  # 最大后向视野
        upper_diagonal = max_past_horizon + max_future_horizon  # 上对角线偏移
        context_size = chunk_size + max_past_horizon + max_future_horizon  # 上下文大小

        lower_causal_mask = torch.tril(  # 下三角因果掩码
            torch.ones((context_size, chunk_size), dtype=torch.bool),  # 全1布尔矩阵
            diagonal=0,  # 主对角线
        ).T  # 转置
        upper_causal_mask = torch.tril(  # 上三角因果掩码
            torch.ones((chunk_size, context_size), dtype=torch.bool),  # 全1布尔矩阵
            diagonal=upper_diagonal,  # 上对角线偏移
        )
        local_causal_valid_mask = torch.ones(  # 局部因果有效掩码初始化为全1
            (chunk_size, context_size), dtype=torch.bool  # 形状和数据类型
        )
        self.register_buffer(  # 注册缓冲区
            "causal_valid_mask",  # 因果有效掩码
            local_causal_valid_mask * lower_causal_mask * upper_causal_mask,  # 三个掩码的交集
            persistent=False,  # 非持久化
        )

    @property  # 属性装饰器
    def device(self):  # 设备属性
        return next(self.parameters()).device  # 返回第一个参数的设备

    def forward(  # 前向传播方法
        self, audio_mel: torch.Tensor, audio_mel_mask: torch.BoolTensor  # mel频谱和掩码
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:  # 返回元组
        """Encode a batch of mel spectrograms.  # 编码一批mel频谱图

        Args:
            audio_mel: [batch, num_frames, mel_bins]  # 音频mel: [批次, 帧数, mel频段]
            audio_mel_mask: [batch, num_frames], True = padding  # 音频mel掩码: [批次, 帧数], True = 填充

        Returns:
            audio_encodings: [batch, reduced_frames, hidden_size/output_proj_dims]  # 音频编码: [批次, 缩减帧数, 隐藏大小/输出投影维度]
            audio_mel_mask: [batch, reduced_frames], True = padding  # 音频mel掩码: [批次, 缩减帧数], True = 填充
        """
        audio_encodings, current_mask = self.subsample_conv_projection(  # 子采样卷积投影
            audio_mel, audio_mel_mask  # mel频谱和掩码
        )

        for block in self.conformer:  # 遍历每个Conformer块
            audio_encodings = block(  # 通过Conformer块
                audio_encodings, current_mask, self.causal_valid_mask  # 编码，掩码和因果掩码
            )

        if self.output_proj is not None:  # 如果有输出投影
            audio_encodings, _ = self.output_proj(audio_encodings)  # 通过输出投影

        if current_mask.shape[1] != audio_encodings.shape[1]:  # 如果掩码和编码长度不匹配
            target_len = audio_encodings.shape[1]  # 目标长度
            if target_len > current_mask.shape[1]:  # 如果目标更长
                current_mask = F.pad(  # 填充掩码
                    current_mask, (0, target_len - current_mask.shape[1]), value=True  # 用True填充
                )
            else:  # 如果掩码更长
                current_mask = current_mask[:, :target_len]  # 截取掩码

        audio_encodings = audio_encodings.masked_fill(current_mask.unsqueeze(-1), 0.0)  # 将掩码位置填0
        return audio_encodings, current_mask  # 返回编码和掩码
