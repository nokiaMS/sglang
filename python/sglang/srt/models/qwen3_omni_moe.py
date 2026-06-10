# Qwen3 全能 MoE 多模态模型实现
# 本文件实现了 Qwen3-Omni-MoE 多模态模型，支持音频和视觉输入。
# 包含音频编码器层、正弦位置编码、音频编码器、视觉补丁合并器、视觉编码器、
# 思考器（Thinker）和完整的多模态条件生成模型，支持 MoE 专家权重融合加载。
# Copyright 2025 Qwen Team  # Qwen 团队版权
# Copyright 2025 SGLang Team  # SGLang 团队版权
# Licensed under the Apache License, Version 2.0 (the "License");  # Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 不得违反许可证使用
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 除非法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 按原样分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何担保
# See the License for the specific language governing permissions and  # 查看许可证获取权限
# limitations under the License.  # 许可证限制
# ==============================================================================
"""Inference-only Qwen3-VL model compatible with HuggingFace weights."""  # 仅推理的 Qwen3-VL 模型

import math  # 导入数学模块
from typing import Iterable, List, Optional, Tuple  # 导入类型提示

import numpy as np  # 导入 NumPy
import torch  # 导入 PyTorch 框架
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式模块
from transformers import PreTrainedModel  # 导入预训练模型基类
from transformers.activations import ACT2FN  # 导入激活函数映射
from transformers.modeling_outputs import BaseModelOutput  # 导入基础模型输出

from sglang.srt.configs.qwen3_omni import (  # 导入 Qwen3 全能配置
    Qwen3OmniMoeAudioEncoderConfig,  # 音频编码器配置
    Qwen3OmniMoeThinkerConfig,  # 思考器配置
    Qwen3OmniMoeVisionEncoderConfig,  # 视觉编码器配置
)
from sglang.srt.configs.qwen3_vl import Qwen3VLMoeConfig  # 导入 Qwen3-VL MoE 配置
from sglang.srt.distributed import (  # 导入分布式函数
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
)
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力
from sglang.srt.layers.linear import (  # 导入线性层
    ColumnParallelLinear,  # 列并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合 MoE 层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.schedule_batch import MultimodalDataItem  # 导入多模态数据项
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel  # 导入 Qwen3-VL MoE 视觉模型
from sglang.srt.models.qwen3_vl_moe import (  # 导入 Qwen3-VL MoE 组件
    Qwen3MoeLLMModel,  # Qwen3 MoE LLM 模型
    Qwen3VLMoeForConditionalGeneration,  # Qwen3-VL MoE 条件生成模型
    load_fused_expert_weights,  # 融合专家权重加载函数
)
from sglang.srt.utils import add_prefix, is_cpu, is_npu, logger  # 导入工具函数

_is_cpu = is_cpu()  # 是否为 CPU 设备


def get_head_dim_and_projection_size(
    embed_dim: int,
    num_heads: int,
    original_num_heads: Optional[int] = None,
) -> Tuple[Optional[int], int]:
    """计算头维度和投影大小，处理 CPU 上 TP 填充的情况"""
    if (not _is_cpu) or original_num_heads is None:  # 非CPU 或无原始头数
        return None, embed_dim  # 不需要特殊处理

    # On CPU, TP may pad num_heads (e.g. for tp=3/6). In that case we keep the  # CPU 上 TP 可能填充头数
    # original per-head width (from original_num_heads) and recompute projection_size  # 保持原始每头宽度并重新计算投影大小
    # with padded num_heads, so attention tensor shapes stay TP-friendly while  # 使用填充后的头数，使注意力张量形状对 TP 友好
    # preserving checkpoint semantics.  # 同时保持检查点语义
    head_dim = embed_dim // original_num_heads  # 计算头维度
    projection_size = num_heads * head_dim  # 计算投影大小
    return head_dim, projection_size  # 返回头维度和投影大小


class Qwen3OmniMoeAudioEncoderLayer(nn.Module):
    """Qwen3 全能 MoE 音频编码器层"""

    def __init__(
        self,
        config: Qwen3OmniMoeAudioEncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        """初始化音频编码器层"""
        super().__init__()  # 调用父类初始化
        embed_dim = config.d_model  # 嵌入维度
        self.embed_dim = config.d_model  # 保存嵌入维度
        head_dim, projection_size = get_head_dim_and_projection_size(  # 获取头维度和投影大小
            embed_dim=embed_dim,  # 嵌入维度
            num_heads=config.encoder_attention_heads,  # 编码器注意力头数
            original_num_heads=getattr(  # 原始头数（CPU TP 填充用）
                config, "original_encoder_attention_heads", None  # 从配置中获取
            ),
        )
        self.self_attn = VisionAttention(  # 自注意力
            embed_dim=embed_dim,  # 嵌入维度
            num_heads=config.encoder_attention_heads,  # 注意力头数
            head_dim=head_dim,  # 头维度
            projection_size=projection_size,  # 投影大小
            use_qkv_parallel=True,  # 使用 QKV 并行
            proj_bias=True,  # 投影使用偏置
            flatten_batch=True,  # 展平批次
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 自注意力层归一化
        self.dropout = config.dropout  # dropout 概率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活 dropout
        tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        use_replicated = config.encoder_ffn_dim % tp_size != 0  # 如果 FFN 维度不能被 TP 大小整除则使用复制
        fc1_cls = ReplicatedLinear if use_replicated else ColumnParallelLinear  # 选择线性层类
        fc2_cls = ReplicatedLinear if use_replicated else RowParallelLinear  # 选择线性层类
        self.fc1 = fc1_cls(  # 第一个全连接层
            self.embed_dim,  # 输入维度
            config.encoder_ffn_dim,  # 输出维度
            quant_config=quant_config,  # 量化配置
            bias=True,  # 使用偏置
            prefix=f"{prefix}.fc1",  # 参数前缀
        )
        self.fc2 = fc2_cls(  # 第二个全连接层
            config.encoder_ffn_dim,  # 输入维度
            self.embed_dim,  # 输出维度
            quant_config=quant_config,  # 量化配置
            bias=True,  # 使用偏置
            prefix=f"{prefix}.fc2",  # 参数前缀
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终层归一化

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """音频编码器层前向传播：注意力 -> 残差 -> FFN -> 残差"""
        """
        Args:  # 参数
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`  # 输入隐藏状态
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size  # 注意力头掩码
                `(encoder_attention_heads,)`.  # 编码器注意力头数大小
            output_attentions (`bool`, *optional*):  # 是否输出注意力
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under  # 是否返回所有注意力层的注意力张量
                returned tensors for more detail.  # 详见返回张量
        """
        residual = hidden_states  # 保存残差
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 自注意力层归一化
        hidden_states = self.self_attn(  # 自注意力计算
            x=hidden_states,  # 输入
            cu_seqlens=cu_seqlens,  # 变长序列边界
        )
        hidden_states = residual + hidden_states  # 残差连接
        residual = hidden_states  # 更新残差
        hidden_states = self.final_layer_norm(hidden_states)  # 最终层归一化
        hidden_states, _ = self.fc1(hidden_states)  # 第一个全连接层
        hidden_states = self.activation_fn(hidden_states)  # 激活函数
        hidden_states, _ = self.fc2(hidden_states)  # 第二个全连接层
        hidden_states = residual + hidden_states  # 残差连接

        if hidden_states.dtype == torch.float16:  # 如果是 float16
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000  # 计算裁剪值
            hidden_states = torch.clamp(  # 裁剪防止溢出
                hidden_states, min=-clamp_value, max=clamp_value  # 裁剪范围
            )

        outputs = (hidden_states,)  # 封装输出

        return outputs  # 返回输出


class SinusoidsPositionEmbedding(nn.Module):
    """正弦位置编码"""

    def __init__(self, length, channels, max_timescale=10000):
        """初始化正弦位置编码"""
        super().__init__()  # 调用父类初始化
        if channels % 2 != 0:  # 通道数必须为偶数
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")  # 抛出异常
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)  # 计算对数时间尺度增量
        inv_timescales = torch.exp(  # 计算逆时间尺度
            -log_timescale_increment * torch.arange(channels // 2).float()  # 指数衰减
        )
        scaled_time = (  # 计算缩放时间
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]  # 位置索引乘以逆时间尺度
        )
        self.register_buffer(  # 注册为缓冲区（不参与梯度计算）
            "positional_embedding",  # 缓冲区名称
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),  # 拼接正弦和余弦
            persistent=False,  # 不持久化到状态字典
        )

    def forward(self, seqlen: int):
        """获取指定长度的位置编码"""
        return self.positional_embedding[:seqlen, :]  # 返回前 seqlen 行


def _get_feat_extract_output_lengths(input_lengths):
    """计算卷积层和音频编码器的输出长度"""
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder  # 计算卷积层和音频编码器的输出长度
    """

    input_lengths_leave = input_lengths % 100  # 剩余长度（100 以内部分）
    feat_lengths = (input_lengths_leave - 1) // 2 + 1  # 特征长度
    output_lengths = (  # 输出长度
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13  # 计算最终长度
    )
    return output_lengths  # 返回输出长度


class Qwen3OmniMoeAudioEncoder(PreTrainedModel):
    """Qwen3 全能 MoE 音频编码器"""

    config: Qwen3OmniMoeAudioEncoderConfig  # 配置类型注解

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig, quant_config=None):
        """初始化音频编码器"""
        super().__init__(config)  # 调用父类初始化
        self.dropout = config.dropout  # dropout 概率

        embed_dim = config.d_model  # 嵌入维度
        self.num_mel_bins = config.num_mel_bins  # 梅尔频率箱数
        self.max_source_positions = config.max_source_positions  # 最大源位置数
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0  # 嵌入缩放因子
        self.n_window = config.n_window  # 窗口大小
        self.positional_embedding = SinusoidsPositionEmbedding(  # 正弦位置编码
            self.max_source_positions, embed_dim  # 最大位置数和嵌入维度
        )
        self.layers = nn.ModuleList(  # 编码器层列表
            [
                Qwen3OmniMoeAudioEncoderLayer(config)  # 音频编码器层
                for _ in range(config.encoder_layers)  # 遍历所有编码器层
            ]
        )
        self.ln_post = nn.LayerNorm(config.d_model)  # 后归一化层
        self.gradient_checkpointing = False  # 是否使用梯度检查点
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)  # 第一个2D卷积
        self.conv2d2 = nn.Conv2d(  # 第二个2D卷积
            config.downsample_hidden_size,  # 输入通道
            config.downsample_hidden_size,  # 输出通道
            3,  # 卷积核大小
            2,  # 步幅
            padding=1,  # 填充
        )
        self.conv2d3 = nn.Conv2d(  # 第三个2D卷积
            config.downsample_hidden_size,  # 输入通道
            config.downsample_hidden_size,  # 输出通道
            3,  # 卷积核大小
            2,  # 步幅
            padding=1,  # 填充
        )
        conv_out_dim = config.downsample_hidden_size * (  # 卷积输出维度
            (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2  # 三次下采样后的频率维度
        )
        self.conv_out = ReplicatedLinear(  # 卷积输出线性层
            conv_out_dim,  # 输入维度
            config.d_model,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
        )
        self.proj1 = ReplicatedLinear(  # 第一个投影层
            config.d_model, config.d_model, quant_config=quant_config  # 输入和输出维度
        )
        self.act = ACT2FN[config.activation_function]  # 激活函数
        self.proj2 = ReplicatedLinear(  # 第二个投影层
            config.d_model, config.output_dim, quant_config=quant_config  # 输入维度和输出维度
        )
        self.n_window_infer = self.config.n_window_infer  # 推理窗口大小
        self.conv_chunksize = self.config.conv_chunksize  # 卷积分块大小

    def _freeze_parameters(self):
        """冻结所有参数"""
        for param in self.parameters():  # 遍历所有参数
            param.requires_grad = False  # 禁用梯度
        self._requires_grad = False  # 标记不需要梯度

    def get_input_embeddings(self) -> nn.Module:
        """获取输入嵌入层"""
        return self.conv1  # 返回第一个卷积层

    def set_input_embeddings(self, value: nn.Module):
        """设置输入嵌入层"""
        self.conv1 = value  # 设置卷积层

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        """音频编码器前向传播：卷积下采样 -> 位置编码 -> 编码器层 -> 投影"""
        r"""
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):  # 特征长度
            mel length  # 梅尔长度
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):  # CNN 后长度
            mel length after cnn  # CNN 后的梅尔长度
        """
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)  # 计算 CNN 后长度
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()  # 计算分块数量

        chunk_lengths = torch.tensor(  # 分块长度
            [self.n_window * 2] * chunk_num.sum(),  # 每个分块默认长度
            dtype=torch.long,  # 数据类型
            device=feature_lens.device,  # 设备
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]  # 尾部分块索引
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)  # 设置尾部块长度
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2  # 长度为0的块设为默认长度

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)  # 按长度分割输入
        padded_feature = nn.utils.rnn.pad_sequence(  # 填充序列
            chunk_list, batch_first=True  # 批次在前
        ).transpose(1, 2)  # 转置

        # Introduce vectorized mask to avoid many small tensors  # 使用向量化掩码避免大量小张量
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)  # CNN 后特征长度
        max_len_after_cnn = (  # 最大 CNN 后长度
            int(feature_lens_after_cnn.max().item())  # 转为整数
            if feature_lens_after_cnn.numel()  # 如果有元素
            else 0  # 否则为 0
        )

        idx = torch.arange(max_len_after_cnn, device=padded_feature.device)  # 创建索引
        padded_mask_after_cnn = idx.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)  # 创建掩码

        padded_feature = padded_feature.unsqueeze(1)  # 增加通道维度

        # Add fast path + chunk normal path  # 快速路径 + 分块正常路径
        if padded_feature.size(0) <= self.conv_chunksize:  # 小批量快速路径
            padded_embed = F.gelu(self.conv2d1(padded_feature))  # 卷积1 + GELU
            padded_embed = F.gelu(self.conv2d2(padded_embed))  # 卷积2 + GELU
            padded_embed = F.gelu(self.conv2d3(padded_embed))  # 卷积3 + GELU
        else:  # 大批量分块路径
            padded_embeds = []  # 嵌入列表
            for chunk in padded_feature.split(self.conv_chunksize, dim=0):  # 分块处理
                x = F.gelu(self.conv2d1(chunk))  # 卷积1 + GELU
                x = F.gelu(self.conv2d2(x))  # 卷积2 + GELU
                x = F.gelu(self.conv2d3(x))  # 卷积3 + GELU
                padded_embeds.append(x)  # 添加到列表
            padded_embed = torch.cat(padded_embeds, dim=0)  # 拼接所有嵌入

        b, c, f, t = padded_embed.size()  # 获取嵌入形状
        padded_embed = self.conv_out(  # 卷积输出线性层
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # 重塑并转换维度
        )[0]

        positional_embedding = (  # 位置编码
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]  # 截取到当前长度
            .unsqueeze(0)  # 增加批次维度
            .to(padded_embed.dtype)  # 转换数据类型
        )
        padded_embed = padded_embed + positional_embedding  # 添加位置编码
        hidden_states = padded_embed[padded_mask_after_cnn]  # 用掩码过滤填充部分
        cu_chunk_lens = [0]  # 分块累积长度
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (  # CNN 后窗口大小
            self.n_window_infer // (self.n_window * 2)  # 推理窗口 / 编码窗口
        )
        # Use tolist() for efficient batch conversion from tensor to Python  # 使用 tolist() 高效转换
        for cnn_len in aftercnn_lens.tolist():  # 遍历每个样本的 CNN 后长度
            num_full_chunks = cnn_len // window_aftercnn  # 完整分块数
            remainder = cnn_len % window_aftercnn  # 剩余长度
            cu_chunk_lens.extend([window_aftercnn] * num_full_chunks)  # 添加完整块
            if remainder:  # 如果有剩余
                cu_chunk_lens.append(remainder)  # 添加剩余块
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(  # 累积求和
            -1, dtype=torch.int32  # int32 类型
        )
        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction  # NPU 上 cu_seqlens 必须在 CPU
        if is_npu():  # 如果是 NPU
            cu_seqlens = cu_seqlens.to("cpu")  # 移到 CPU

        for encoder_layer in self.layers:  # 遍历所有编码器层
            layer_outputs = encoder_layer(  # 通过编码器层
                hidden_states,  # 隐藏状态
                cu_seqlens,  # 变长序列边界
            )

            hidden_states = layer_outputs[0]  # 更新隐藏状态

        hidden_states = self.ln_post(hidden_states)  # 后归一化
        hidden_states = self.proj1(hidden_states)[0]  # 第一个投影
        hidden_states = self.act(hidden_states)  # 激活函数
        hidden_states = self.proj2(hidden_states)[0]  # 第二个投影
        return BaseModelOutput(last_hidden_state=hidden_states)  # 返回基础模型输出

    # Ignore copy  # 忽略复制
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """计算卷积层和音频编码器的输出长度（内部版本）"""
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder  # 计算卷积层和音频编码器的输出长度
        """
        input_lengths = (input_lengths - 1) // 2 + 1  # 第一层卷积输出
        output_lengths = (input_lengths - 2) // 2 + 1  # 第二层卷积输出
        return input_lengths, output_lengths  # 返回输入和输出长度


class Qwen3OmniMoeVisionPatchMerger(nn.Module):
    """Qwen3 全能 MoE 视觉补丁合并器"""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_postshuffle_norm=False,
    ) -> None:
        """初始化视觉补丁合并器"""
        super().__init__()  # 调用父类初始化
        self.hidden_size = context_dim * (spatial_merge_size**2)  # 合并后隐藏维度
        self.use_postshuffle_norm = use_postshuffle_norm  # 是否在重排后归一化
        self.ln_q = nn.LayerNorm(  # 归一化层
            self.hidden_size if use_postshuffle_norm else context_dim, eps=1e-6  # 根据是否后归一化选择维度
        )
        self.mlp = nn.ModuleList(  # MLP 模块列表
            [
                ColumnParallelLinear(  # 列并行全连接层
                    self.hidden_size,  # 输入维度
                    self.hidden_size,  # 输出维度
                    bias=True,  # 使用偏置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix("mlp.0", prefix),  # 参数前缀
                ),
                nn.GELU(),  # GELU 激活函数
                RowParallelLinear(  # 行并行全连接层
                    self.hidden_size,  # 输入维度
                    dim,  # 输出维度
                    bias=True,  # 使用偏置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix("mlp.2", prefix),  # 参数前缀
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """补丁合并器前向传播：归一化 -> MLP"""
        x = (  # 重塑输入
            x.view(-1, self.hidden_size)  # 后归一化模式：按合并后大小重塑
            if self.use_postshuffle_norm  # 如果使用后归一化
            else x.view(-1, x.shape[-1])  # 否则按原始维度重塑
        )
        hidden = self.ln_q(x).view(-1, self.hidden_size)  # 归一化并重塑
        for layer in self.mlp:  # 遍历 MLP 层
            if isinstance(hidden, tuple):  # 如果是元组
                hidden = hidden[0]  # 取第一个元素
            hidden = layer(hidden)  # 通过层

        if isinstance(hidden, tuple):  # 如果结果是元组
            hidden = hidden[0]  # 取第一个元素

        return hidden  # 返回输出


class Qwen3OmniMoeVisionEncoder(Qwen3VLMoeVisionModel):
    """Qwen3 全能 MoE 视觉编码器，继承自 Qwen3-VL MoE 视觉模型"""

    config: Qwen3OmniMoeVisionEncoderConfig  # 配置类型注解

    def __init__(
        self,
        config: Qwen3OmniMoeVisionEncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = None,
        **kwargs,
    ):
        """初始化视觉编码器"""
        super().__init__(  # 调用父类初始化
            vision_config=config,  # 视觉配置
            quant_config=quant_config,  # 量化配置
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),  # 归一化 epsilon
        )

        self.merger = Qwen3OmniMoeVisionPatchMerger(  # 主合并器
            dim=config.out_hidden_size,  # 输出维度
            context_dim=config.hidden_size,  # 上下文维度
            spatial_merge_size=config.spatial_merge_size,  # 空间合并大小
            quant_config=quant_config,  # 量化配置
            use_postshuffle_norm=False,  # 不使用后归一化
            prefix=add_prefix("merger", prefix),  # 参数前缀
        )
        self.merger_list = nn.ModuleList(  # 合并器列表（用于深度堆叠）
            [
                Qwen3OmniMoeVisionPatchMerger(  # 每个深度堆叠的合并器
                    dim=config.out_hidden_size,  # 输出维度
                    context_dim=config.hidden_size,  # 上下文维度
                    spatial_merge_size=config.spatial_merge_size,  # 空间合并大小
                    use_postshuffle_norm=True,  # 使用后归一化
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix("merger_list", prefix),  # 参数前缀
                )
                for _ in range(len(config.deepstack_visual_indexes))  # 按深度堆叠索引数量
            ]
        )
        del self.deepstack_merger_list  # 删除父类的深度堆叠合并器列表

    @property
    def deepstack_merger_list(self):
        """获取深度堆叠合并器列表"""
        return self.merger_list  # 返回合并器列表

    @property
    def dtype(self) -> torch.dtype:
        """获取模型数据类型"""
        return self.patch_embed.proj.weight.dtype  # 返回补丁嵌入权重的数据类型

    @property
    def device(self) -> torch.device:
        """获取模型设备"""
        return self.patch_embed.proj.weight.device  # 返回补丁嵌入权重的设备


class Qwen3OmniMoeThinkerForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
    """Qwen3 全能 MoE 思考器，整合音频编码器和视觉编码器"""

    config: Qwen3OmniMoeThinkerConfig  # 配置类型注解

    def __init__(
        self,
        config: Qwen3OmniMoeThinkerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        """初始化思考器模型"""
        super().__init__(  # 调用父类初始化
            config, quant_config, prefix, language_model_cls=Qwen3MoeLLMModel  # 使用 MoE LLM 模型类
        )
        self.audio_tower = Qwen3OmniMoeAudioEncoder(config.audio_config, quant_config)  # 音频编码器
        self.visual = Qwen3OmniMoeVisionEncoder(  # 视觉编码器
            config.vision_config,  # 视觉配置
            quant_config=quant_config,  # 量化配置
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),  # 归一化 epsilon
            prefix=add_prefix("visual", prefix),  # 参数前缀
        )
        self.pad_token_id = (  # 填充 token ID
            self.config.pad_token_id if self.config.pad_token_id is not None else -1  # 默认为 -1
        )

    def get_audio_feature(self, items: List[MultimodalDataItem]):
        """从多模态数据项中提取音频特征"""
        device = next(self.audio_tower.parameters()).device  # 获取音频编码器设备
        feature_attention_mask = (  # 拼接注意力掩码
            torch.cat([item.feature_attention_mask for item in items], dim=0)  # 拼接
            .type(torch.long)  # 转换为 long 类型
            .to(device)  # 移到设备
        )
        input_features = (  # 拼接音频特征
            torch.cat([item.feature for item in items])  # 拼接
            .type(self.audio_tower.dtype)  # 转换数据类型
            .to(next(self.audio_tower.parameters()).device)  # 移到设备
        )
        if feature_attention_mask is not None:  # 如果有掩码
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)  # 计算有效长度
            input_features = input_features.permute(0, 2, 1)[  # 转置并用掩码过滤
                feature_attention_mask.bool()  # 使用布尔掩码
            ].permute(1, 0)  # 转回原维度
        else:  # 没有掩码
            audio_feature_lengths = None  # 长度为空

        feature_lens = (  # 确定特征长度
            audio_feature_lengths  # 如果有长度
            if audio_feature_lengths is not None  # 使用音频特征长度
            else feature_attention_mask.sum(-1)  # 否则使用掩码求和
        )
        audio_outputs = self.audio_tower(  # 通过音频编码器
            input_features,  # 输入特征
            feature_lens=feature_lens,  # 特征长度
        )
        audio_features = audio_outputs.last_hidden_state  # 获取最后一层隐藏状态

        return audio_features  # 返回音频特征


class Qwen3OmniMoeForConditionalGeneration(PreTrainedModel):
    """Qwen3 全能 MoE 条件生成模型，整合思考器"""

    def __init__(
        self,
        config: Qwen3VLMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        """初始化全能 MoE 条件生成模型"""
        super().__init__(config)  # 调用父类初始化
        self.config = config  # 保存配置

        self.thinker = Qwen3OmniMoeThinkerForConditionalGeneration(  # 创建思考器
            config.thinker_config, quant_config=quant_config, prefix=prefix  # 思考器配置、量化配置和前缀
        )
        self.enable_talker = False  # 是否启用 talker（语音生成）
        self.pad_input_ids = self.thinker.pad_input_ids  # 委托思考器的填充方法
        self.forward = self.thinker.forward  # 委托思考器的前向传播

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，处理堆叠参数、专家参数和前缀映射"""
        stacked_params_mapping = [  # 堆叠参数映射表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),  # Q 映射
            (".qkv_proj", ".k_proj", "k"),  # K 映射
            (".qkv_proj", ".v_proj", "v"),  # V 映射
            ("gate_up_proj", "up_proj", 1),  # up 映射
            ("gate_up_proj", "gate_proj", 0),  # gate 映射
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 专家参数映射
            ckpt_gate_proj_name="gate_proj",  # gate 投影检查点名
            ckpt_down_proj_name="down_proj",  # down 投影检查点名
            ckpt_up_proj_name="up_proj",  # up 投影检查点名
            num_experts=self.config.num_experts,  # 专家数量
        )

        # Skip loading extra parameters for GPTQ/modelopt models.  # 跳过 GPTQ/modelopt 模型的额外参数
        ignore_suffixes = (  # 忽略的后缀列表
            ".bias",  # 偏置
            "_bias",  # 偏置（下划线格式）
            ".k_scale",  # K 缩放
            "_k_scale",  # K 缩放（下划线格式）
            ".v_scale",  # V 缩放
            "_v_scale",  # V 缩放（下划线格式）
            ".weight_scale",  # 权重缩放
            "_weight_scale",  # 权重缩放（下划线格式）
            ".input_scale",  # 输入缩放
            "_input_scale",  # 输入缩放（下划线格式）
        )

        is_fused_expert = False  # 是否为融合专家权重
        fused_expert_params_mapping = [  # 融合专家参数映射
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),  # gate_up 映射
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),  # down 映射
        ]

        num_experts = self.config.num_experts  # 专家数量

        # Pre-define `params_dict` to avoid repeated expensive traversal of model parameters.  # 预定义参数字典避免重复遍历
        params_dict = dict(self.named_parameters())  # 获取参数字典

        for name, loaded_weight in weights:  # 遍历所有权重
            name = name.replace(r"model.language_model.", r"model.")  # 替换语言模型前缀

            if ("talker" in name or "code2wav" in name) and not self.enable_talker:  # 跳过 talker 和 code2wav
                continue

            name = name.replace(".self_attn.out_proj", ".self_attn.proj")  # 替换注意力输出投影名

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:  # 如果是融合专家
                    is_fused_expert = True  # 标记为融合专家
                    expert_params_mapping = fused_expert_params_mapping  # 使用融合专家映射

                # Skip non-stacked layers and experts (experts handled below).  # 跳过非堆叠层和专家
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue
                if "visual" in name:  # 跳过视觉模块（视觉模块单独处理）
                    continue

                # We have mlp.experts[0].gate_proj in the checkpoint.  # 检查点中有 mlp.experts[0].gate_proj
                # Since we handle the experts below in expert_params_mapping,  # 因为专家在下面的 expert_params_mapping 中处理
                # we need to skip here BEFORE we update the name, otherwise  # 需要在更新名称前跳过
                # name will be updated to mlp.experts[0].gate_up_proj, which  # 否则名称会变为 gate_up_proj
                # will then be updated below in expert_params_mapping  # 然后在 expert_params_mapping 中再次更新
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.  # 导致 gate_gate_up_proj 错误
                if "mlp.experts" in name:  # 如果是专家 MLP
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                # Skip loading extra parameters for GPTQ/modelopt models.  # 跳过额外参数
                if name.endswith(ignore_suffixes) and name not in params_dict:  # 如果以忽略后缀结尾且不在字典中
                    continue
                # [TODO] Skip layers that are on other devices (check if sglang has a similar function)  # TODO 跳过其他设备上的层
                # if is_pp_missing_parameter(name, self):  # 如果是流水线缺失参数
                #     continue  # 跳过

                if name not in params_dict:  # 如果参数不在字典中
                    continue

                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break
            else:  # 非堆叠参数处理
                # Track if this is an expert weight to enable early skipping  # 跟踪是否为专家权重以提前跳过
                is_expert_weight = False  # 专家权重标志

                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping  # 解包映射
                    if weight_name not in name:  # 如果权重名不在参数名中
                        continue
                    if "visual" in name or "audio_tower" in name:  # 跳过视觉和音频塔
                        continue
                    # Anyway, this is an expert weight and should not be  # 无论如何这是专家权重
                    # attempted to load as other weights later  # 不应尝试作为其他权重加载
                    is_expert_weight = True  # 标记为专家权重
                    name_mapped = name.replace(weight_name, param_name)  # 替换为映射名称
                    if is_fused_expert:  # 如果是融合专家
                        loaded_weight = loaded_weight.transpose(-1, -2)  # no bias  # 转置（无偏置）
                        if "experts.gate_up_proj" in name:  # 如果是 gate_up_proj
                            loaded_weight = loaded_weight.chunk(2, dim=-2)  # 分成两块
                            load_fused_expert_weights(  # 加载 gate 融合权重
                                name_mapped,  # 映射名称
                                params_dict,  # 参数字典
                                loaded_weight[0],  # gate 权重
                                "w1",  # w1 标识
                                num_experts,  # 专家数量
                            )
                            load_fused_expert_weights(  # 加载 up 融合权重
                                name_mapped,  # 映射名称
                                params_dict,  # 参数字典
                                loaded_weight[1],  # up 权重
                                "w3",  # w3 标识
                                num_experts,  # 专家数量
                            )
                        else:  # down 投影
                            load_fused_expert_weights(  # 加载 down 融合权重
                                name_mapped,  # 映射名称
                                params_dict,  # 参数字典
                                loaded_weight,  # 权重
                                shard_id,  # 分片 ID
                                num_experts,  # 专家数量
                            )
                    else:  # 非融合专家
                        # Skip loading extra parameters for GPTQ/modelopt models.  # 跳过额外参数
                        if (  # 检查是否为额外参数
                            name_mapped.endswith(ignore_suffixes)  # 以忽略后缀结尾
                            and name_mapped not in params_dict  # 且不在字典中
                        ):
                            continue  # 跳过
                        if name_mapped in params_dict.keys():  # 如果映射名称在字典中
                            param = params_dict[name_mapped]  # 获取参数
                        else:  # 不在字典中
                            continue  # 跳过
                        # We should ask the weight loader to return success or  # 应该让权重加载器返回成功与否
                        # not here since otherwise we may skip experts with  # 否则可能跳过有其他副本的专家
                        # # other available replicas.  # 其他可用副本
                        weight_loader = param.weight_loader  # 获取权重加载器
                        weight_loader(  # 加载专家权重
                            param,  # 参数
                            loaded_weight,  # 加载的权重
                            name_mapped,  # 映射名称
                            shard_id=shard_id,  # 分片 ID
                            expert_id=expert_id,  # 专家 ID
                        )
                    name = name_mapped  # 更新名称
                    break
                else:  # 非专家参数处理
                    if is_expert_weight:  # 如果是专家权重但未映射到当前排名
                        # This is an expert weight but not mapped to this rank, skip all remaining processing  # 专家权重未映射到当前排名，跳过
                        continue  # 跳过
                    if "visual" in name or "audio_tower" in name:  # 视觉和音频塔参数
                        # adapt to VisionAttention  # 适配 VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")  # 替换 QKV 名
                        name = name.replace(r"model.visual.", r"visual.")  # 替换视觉前缀
                        name = name.replace(r"attn.out_proj.", r"attn.proj.")  # 替换输出投影名

                    # Skip loading extra parameters for GPTQ/modelopt models.  # 跳过额外参数
                    if name.endswith(ignore_suffixes) and name not in params_dict:  # 如果以忽略后缀结尾且不在字典中
                        continue  # 跳过

                    if name in params_dict.keys():  # 如果参数在字典中
                        param = params_dict[name]  # 获取参数
                        weight_loader = getattr(  # 获取权重加载器
                            param, "weight_loader", default_weight_loader  # 默认权重加载器
                        )
                        weight_loader(param, loaded_weight)  # 加载权重
                    else:  # 参数不在字典中
                        logger.warning(  # 警告
                            f"Loaded weight with {name=} not found in params_dict"  # 权重未找到
                        )


EntryClass = Qwen3OmniMoeForConditionalGeneration  # 模型入口类
