# Whisper语音识别模型实现
# 本文件实现了Whisper语音识别模型，包含编码器、解码器和条件生成模型。
# 支持交叉注意力、编码器KV缓存和张量并行。

from __future__ import annotations  # 启用延迟注解求值

from array import array  # 导入array模块
from typing import Any, Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
from transformers import WhisperConfig  # 导入Whisper配置

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入TP世界大小获取函数
from sglang.srt.layers.activation import get_act_fn  # 导入激活函数获取
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,  # 列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入logits处理器
from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention  # 导入注意力类型和基数注意力
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行LM头
from sglang.srt.managers.schedule_batch import MultimodalInputs  # 导入多模态输入
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器


class WhisperAttention(torch.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""  # 论文"Attention Is All You Need"中的多头注意力

    def __init__(
        self,
        embed_dim: int,  # 嵌入维度
        num_heads: int,  # 注意力头数
        bias: bool = True,  # 是否使用偏置
        layer_id: Optional[int] = None,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        is_cross_attention: bool = False,  # 是否交叉注意力
        is_encoder=False,  # 是否编码器
    ):
        super().__init__()  # 调用父类初始化
        self.total_num_heads = num_heads  # 总头数
        head_dim = embed_dim // num_heads  # 头维度
        self.is_cross_attention = is_cross_attention  # 是否交叉注意力
        self.is_encoder = is_encoder  # 是否编码器

        tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        assert (
            num_heads % tp_size == 0
        ), f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"  # 断言头数可整除
        self.num_heads = num_heads // tp_size  # TP后头数

        if (head_dim * num_heads) != embed_dim:  # 维度不匹配
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = head_dim**-0.5  # 缩放因子
        self.head_dim = head_dim  # 头维度
        self.kv_size = self.num_heads * head_dim  # KV大小

        if is_cross_attention:  # 交叉注意力
            self.q_proj = ColumnParallelLinear(  # Q投影
                embed_dim, embed_dim, quant_config=quant_config
            )
            self.kv_proj = QKVParallelLinear(  # KV投影
                hidden_size=embed_dim,
                head_size=head_dim,
                total_num_heads=0,  # 无Q头
                total_num_kv_heads=num_heads,  # KV头数
                bias=bias,
                quant_config=quant_config,
            )
        else:  # 自注意力
            self.qkv_proj = QKVParallelLinear(  # QKV投影
                embed_dim, head_dim, num_heads, quant_config=quant_config
            )
        self.out_proj = RowParallelLinear(  # 输出投影
            embed_dim, embed_dim, bias=bias, quant_config=quant_config
        )
        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,
            head_dim,
            scaling=1.0,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            is_cross_attention=is_cross_attention,
            attn_type=(
                AttentionType.ENCODER_ONLY if is_encoder else AttentionType.DECODER  # 注意力类型
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        cross_hidden_states: Optional[torch.Tensor] = None,  # 交叉注意力隐藏状态
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""  # 注意力前向传播

        if self.is_cross_attention:  # 交叉注意力
            # Cross-attention: KV cached during prefill, read from pool during decode.
            q, _ = self.q_proj(hidden_states)  # Q投影
            q = q * self.scaling  # 缩放Q
            if cross_hidden_states is not None:  # 有交叉隐藏状态
                kv, _ = self.kv_proj(cross_hidden_states)  # KV投影
                k, v = kv.split([self.kv_size, self.kv_size], dim=-1)  # 分离K和V
            else:
                k = None  # 无K
                v = None  # 无V
            attn_output = self.attn(q, k, v, forward_batch)  # 通过注意力
        else:  # 自注意力
            qkv, _ = self.qkv_proj(hidden_states)  # QKV投影
            q, k, v = qkv.chunk(chunks=3, dim=-1)  # 分离Q、K、V
            q = q * self.scaling  # 缩放Q

            if self.is_encoder:  # 编码器自注意力
                num_heads = self.attn.tp_q_head_num  # TP后Q头数
                head_dim = self.attn.head_dim  # 头维度
                batch_size, seq_len, _ = hidden_states.shape  # 获取形状

                q = q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)  # 重塑Q
                k = k.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)  # 重塑K
                v = v.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)  # 重塑V

                attn_output = torch.nn.functional.scaled_dot_product_attention(  # SDPA注意力
                    q, k, v, scale=1.0
                )
                attn_output = attn_output.permute(0, 2, 1, 3).reshape(  # 恢复形状
                    batch_size, seq_len, num_heads * head_dim
                )
            else:  # 解码器自注意力
                attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=True)  # 通过注意力并保存KV缓存

        attn_output, _ = self.out_proj(attn_output)  # 通过输出投影

        return attn_output  # 返回注意力输出


class WhisperEncoderLayer(torch.nn.Module):
    """Whisper编码器层"""

    def __init__(
        self,
        config: WhisperConfig,  # Whisper配置
        layer_id: Optional[int] = None,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
    ):
        super().__init__()  # 调用父类初始化
        self.embed_dim = config.d_model  # 嵌入维度

        self.self_attn = WhisperAttention(  # 自注意力
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            is_encoder=True,
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)  # 自注意力层归一化

        self.activation_fn = get_act_fn(  # 激活函数
            config.activation_function, quant_config=quant_config
        )

        self.fc1 = ColumnParallelLinear(self.embed_dim, config.encoder_ffn_dim)  # FFN第一层
        self.fc2 = RowParallelLinear(config.encoder_ffn_dim, self.embed_dim)  # FFN第二层
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim)  # FFN层归一化

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """编码器层前向传播"""
        residual = hidden_states  # 保存残差
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 归一化
        hidden_states = self.self_attn(hidden_states, forward_batch)  # 自注意力

        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 更新残差
        hidden_states = self.final_layer_norm(hidden_states)  # 归一化
        hidden_states, _ = self.fc1(hidden_states)  # FFN第一层
        hidden_states = self.activation_fn(hidden_states)  # 激活

        hidden_states, _ = self.fc2(hidden_states)  # FFN第二层

        hidden_states = residual + hidden_states  # 残差连接

        if hidden_states.dtype == torch.float16:  # FP16精度保护
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        return hidden_states  # 返回输出


class WhisperDecoderLayer(torch.nn.Module):
    """Whisper解码器层，包含自注意力、交叉注意力和FFN"""

    def __init__(
        self,
        config: WhisperConfig,  # Whisper配置
        layer_id: Optional[int] = None,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
    ):
        super().__init__()  # 调用父类初始化
        self.embed_dim = config.d_model  # 嵌入维度

        # Offset decoder layer IDs to avoid overlap with encoder layers
        decoder_self_attn_layer_id = config.encoder_layers + layer_id  # 解码器自注意力层ID
        decoder_cross_attn_layer_id = (  # 解码器交叉注意力层ID
            config.encoder_layers + config.decoder_layers + layer_id
        )

        self.self_attn = WhisperAttention(  # 自注意力
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            layer_id=decoder_self_attn_layer_id,
            quant_config=quant_config,
        )

        self.activation_fn = get_act_fn(  # 激活函数
            config.activation_function, quant_config=quant_config
        )

        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)  # 自注意力归一化
        self.encoder_attn = WhisperAttention(  # 交叉注意力
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            layer_id=decoder_cross_attn_layer_id,
            quant_config=quant_config,
            is_cross_attention=True,
        )
        self.encoder_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)  # 交叉注意力归一化
        self.fc1 = ColumnParallelLinear(self.embed_dim, config.decoder_ffn_dim)  # FFN第一层
        self.fc2 = RowParallelLinear(config.decoder_ffn_dim, self.embed_dim)  # FFN第二层
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim)  # FFN归一化

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,  # 解码器隐藏状态
        encoder_hidden_states: Optional[torch.Tensor],  # 编码器隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """解码器层前向传播"""
        residual = decoder_hidden_states  # 保存残差
        decoder_hidden_states = self.self_attn_layer_norm(decoder_hidden_states)  # 归一化
        decoder_hidden_states = self.self_attn(decoder_hidden_states, forward_batch)  # 自注意力
        decoder_hidden_states = residual + decoder_hidden_states  # 残差连接

        residual = decoder_hidden_states  # 更新残差
        decoder_hidden_states = self.encoder_attn_layer_norm(decoder_hidden_states)  # 归一化
        decoder_hidden_states = self.encoder_attn(  # 交叉注意力
            decoder_hidden_states, forward_batch, encoder_hidden_states
        )
        decoder_hidden_states = residual + decoder_hidden_states  # 残差连接

        residual = decoder_hidden_states  # 更新残差
        decoder_hidden_states = self.final_layer_norm(decoder_hidden_states)  # 归一化
        decoder_hidden_states, _ = self.fc1(decoder_hidden_states)  # FFN第一层
        decoder_hidden_states = self.activation_fn(decoder_hidden_states)  # 激活
        decoder_hidden_states, _ = self.fc2(decoder_hidden_states)  # FFN第二层

        decoder_hidden_states = residual + decoder_hidden_states  # 残差连接

        return decoder_hidden_states  # 返回输出


class WhisperEncoder(torch.nn.Module):
    """Whisper编码器"""

    def __init__(
        self, config: WhisperConfig, quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()  # 调用父类初始化

        embed_dim = config.d_model  # 嵌入维度
        self.embed_scale = embed_dim**-0.5 if config.scale_embedding else 1.0  # 嵌入缩放

        self.conv1 = torch.nn.Conv1d(  # 第一个卷积
            config.num_mel_bins, embed_dim, kernel_size=3, padding=1
        )
        self.conv2 = torch.nn.Conv1d(  # 第二个卷积（下采样）
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.embed_positions = torch.nn.Embedding(  # 位置嵌入
            config.max_source_positions, embed_dim
        )

        self.layers = torch.nn.ModuleList(  # 编码器层列表
            [
                WhisperEncoderLayer(config, id, quant_config)
                for id in range(config.encoder_layers)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(config.d_model)  # 最终归一化

    def forward(
        self,
        input_features: torch.Tensor,  # 输入特征
        position_ids: torch.Tensor,  # 位置ID
        forward_batch: ForwardBatch,  # 前向批次
    ):
        """编码器前向传播"""
        device = self.conv1.weight.device  # 获取设备
        input_features = input_features.to(device=device)  # 移到设备
        position_ids = position_ids.to(device=device)  # 移到设备

        inputs_embeds = torch.nn.functional.gelu(self.conv1(input_features))  # 卷积1+GELU
        inputs_embeds = torch.nn.functional.gelu(self.conv2(inputs_embeds))  # 卷积2+GELU

        inputs_embeds = inputs_embeds.mT  # 转置

        hidden_states = inputs_embeds + self.embed_positions(position_ids)  # 添加位置嵌入

        for encoder_layer in self.layers:  # 遍历编码器层
            hidden_states = encoder_layer(hidden_states, forward_batch)  # 通过当前层

        hidden_states = self.layer_norm(hidden_states)  # 最终归一化
        return hidden_states  # 返回编码器输出


class WhisperDecoder(torch.nn.Module):
    """Whisper解码器"""

    def __init__(
        self, config: WhisperConfig, quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()  # 调用父类初始化
        self.max_target_positions = config.max_target_positions  # 最大目标位置
        self.max_source_positions = config.max_source_positions  # 最大源位置
        self.embed_scale = config.d_model**-0.5 if config.scale_embedding else 1.0  # 嵌入缩放

        self.embed_tokens = torch.nn.Embedding(  # 词嵌入
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.embed_positions = torch.nn.Embedding(  # 位置嵌入
            self.max_target_positions, config.d_model
        )

        self.layers = torch.nn.ModuleList(  # 解码器层列表
            [
                WhisperDecoderLayer(config, layer_idx, quant_config)
                for layer_idx in range(config.decoder_layers)
            ]
        )

        self.layer_norm = torch.nn.LayerNorm(config.d_model)  # 最终归一化

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        encoder_hidden_states: Optional[torch.Tensor],  # 编码器隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        position_ids=None,  # 位置ID
    ):
        """解码器前向传播"""
        inputs_embeds = self.embed_tokens(input_ids)  # 词嵌入
        position_ids = position_ids.clamp(max=self.max_target_positions - 1)  # 限制位置ID
        positions = self.embed_positions(position_ids)  # 位置嵌入
        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)  # 添加位置嵌入

        for decoder_layer in self.layers:  # 遍历解码器层
            hidden_states = decoder_layer(
                hidden_states, encoder_hidden_states, forward_batch
            )

        hidden_states = self.layer_norm(hidden_states)  # 最终归一化

        return hidden_states  # 返回解码器输出


class WhisperForConditionalGeneration(torch.nn.Module):
    """Whisper条件生成模型"""

    def __init__(
        self, config: WhisperConfig, quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()  # 调用父类初始化
        self.encoder = WhisperEncoder(config, quant_config)  # 编码器
        self.decoder = WhisperDecoder(config, quant_config)  # 解码器
        self.proj_out = ParallelLMHead(  # 输出投影
            config.vocab_size, config.d_model, quant_config=quant_config
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器
        self.config = config  # 保存配置

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重"""
        stacked_params_mapping = [  # 堆叠参数映射
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),  # 自注意力Q
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),  # 自注意力K
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),  # 自注意力V
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),  # 交叉注意力K
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),  # 交叉注意力V
        ]

        params_dict = dict(self.named_parameters())  # 参数字典
        weights_dict = dict(weights)  # 权重字典

        # Whisper has no k_proj bias, create zeros
        for layer_idx in range(self.config.decoder_layers):  # 遍历解码器层
            layer_prefix = f"model.decoder.layers.{layer_idx}.encoder_attn."  # 层前缀
            k_proj_key = layer_prefix + "k_proj.weight"  # K投影权重键
            if k_proj_key in weights_dict:  # 键存在
                k_proj_weight = weights_dict[k_proj_key]  # 获取权重
                bias_key = layer_prefix + "k_proj.bias"  # 偏置键
                if bias_key not in weights_dict:  # 偏置不存在
                    weights_dict[bias_key] = torch.zeros(k_proj_weight.size(0))  # 创建零偏置

        weights_dict["proj_out.weight"] = weights_dict[  # 绑定LM头权重
            "model.decoder.embed_tokens.weight"
        ]

        for name, loaded_weight in weights_dict.items():  # 遍历权重
            name = name.replace("model.", "")  # 移除model.前缀

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠映射
                if weight_name not in name:  # 不匹配
                    continue
                name = name.replace(weight_name, param_name)  # 替换名称
                if name not in params_dict:  # 参数不存在
                    break
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break
            else:  # 非堆叠参数
                if name not in params_dict:  # 参数不存在
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取加载器
                weight_loader(param, loaded_weight)  # 加载权重

    def pad_input_ids(
        self, input_ids: array[int], mm_inputs: MultimodalInputs
    ) -> array[int]:
        """填充输入ID，添加编码器位置的虚拟token"""
        # Prepend dummy encoder tokens so that prepare_encoder_info_extend
        # correctly allocates encoder KV cache locations in the KV pool.
        # These dummy tokens are stripped before the model forward receives input_ids.
        encoder_len = self.config.max_source_positions  # 编码器长度
        mm_inputs.num_image_tokens = encoder_len  # 设置图像token数（复用字段）
        return array("q", [0]) * encoder_len + input_ids  # 前面添加虚拟token

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        """模型前向传播：编码器 -> 解码器 -> logits"""
        dtype = self.encoder.conv1.weight.dtype  # 获取数据类型

        # Run encoder for requests that haven't cached encoder output yet.
        # During decode or when encoder is already cached, encoder_hidden_states
        # is None and cross-attention reads KV from the pool via RadixAttention.
        encoder_hidden_states = None  # 编码器隐藏状态
        if not forward_batch.forward_mode.is_decode():  # 预填充阶段
            mm_inputs_list = forward_batch.mm_inputs if forward_batch.mm_inputs else []  # 多模态输入
            encoder_cached_list = (  # 编码器缓存列表
                forward_batch.encoder_cached if forward_batch.encoder_cached else []
            )

            # Collect features from all uncached requests for batched encoding
            features_to_encode = []  # 待编码特征列表
            for mm_input, cached in zip(mm_inputs_list, encoder_cached_list):  # 遍历
                if cached or mm_input is None or not mm_input.mm_items:  # 已缓存或空
                    continue
                features = mm_input.mm_items[0].feature  # 获取特征
                if features.ndim == 2:  # 2D
                    features = features.unsqueeze(0)  # 添加batch维度
                features_to_encode.append(features.to(dtype))  # 添加到列表

            if features_to_encode:  # 有待编码特征
                # Batch all features and run encoder once instead of sequentially
                features_batch = torch.cat(features_to_encode, dim=0)  # 批量拼接
                encoder_len = features_batch.shape[-1] // 2  # 编码器长度
                encoder_position_ids = torch.arange(  # 编码器位置ID
                    encoder_len, device=features_batch.device
                )

                batched_output = self.encoder(  # 批量编码
                    features_batch, encoder_position_ids, forward_batch
                )
                # Flatten [N, seq_len, dim] → [N*seq_len, dim] for cross-attention
                encoder_hidden_states = batched_output.reshape(  # 展平
                    -1, batched_output.shape[-1]
                )

        decoder_outputs = self.decoder(  # 通过解码器
            input_ids, encoder_hidden_states, forward_batch, positions
        )

        logits = self.logits_processor(  # 处理logits
            input_ids=input_ids,
            lm_head=self.proj_out,
            hidden_states=decoder_outputs,
            logits_metadata=forward_batch,
        )

        return logits  # 返回logits


EntryClass = [WhisperForConditionalGeneration]  # 入口类列表
