# ChatGLM模型实现文件 - 实现与THUDM权重兼容的ChatGLM推理模型
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

# Adapted from  # 改编自
# https://github.com/THUDM/ChatGLM2-6B  # ChatGLM2-6B的GitHub仓库
"""Inference-only ChatGLM model compatible with THUDM weights."""  # 仅推理的ChatGLM模型，兼容THUDM权重

from typing import Iterable, Optional, Tuple # 导入类型提示模块

import torch # 导入PyTorch深度学习框架
from torch import nn # 导入神经网络模块
from torch.nn import LayerNorm # 导入层归一化

from sglang.srt.configs import ChatGLMConfig # 导入ChatGLM配置
from sglang.srt.distributed import get_tensor_model_parallel_world_size # 导入获取张量并行世界大小的函数
from sglang.srt.layers.activation import SiluAndMul # 导入SiLU与乘法激活函数
from sglang.srt.layers.layernorm import RMSNorm # 导入RMS归一化层
from sglang.srt.layers.linear import ( # 导入并行线性层
    MergedColumnParallelLinear, # 合并列并行线性层
    QKVParallelLinear, # QKV并行线性层
    RowParallelLinear, # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor # 导入逻辑处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import ( # 导入词表并行嵌入层
    ParallelLMHead, # 并行语言模型头
    VocabParallelEmbedding, # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批次信息类
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.utils import add_prefix # 导入添加前缀的工具函数

LoraConfig = None # LoRA配置占位符


class GLMAttention(nn.Module): # GLM注意力层类，实现多头注意力机制
    def __init__( # 初始化GLM注意力层
        self,
        config, # 模型配置
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.hidden_size = config.hidden_size # 隐藏维度
        tp_size = get_tensor_model_parallel_world_size() # 获取张量并行大小
        self.total_num_heads = config.num_attention_heads # 总注意力头数
        assert self.total_num_heads % tp_size == 0 # 确保头数可被并行度整除
        self.num_heads = self.total_num_heads // tp_size # 每个并行分片的头数
        self.multi_query_attention = config.multi_query_attention # 是否使用多查询注意力（MQA）
        self.total_num_kv_heads = ( # 总KV头数
            config.multi_query_group_num # 使用MQA时的KV头组数
            if config.multi_query_attention # 如果启用MQA
            else config.num_attention_heads # 否则与Q头数相同
        )
        if self.total_num_kv_heads >= tp_size: # 如果KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，因此在多个张量并行GPU间分配
            # the KV heads across multiple tensor parallel GPUs.  # KV头
            assert self.total_num_kv_heads % tp_size == 0 # 确保KV头数可被TP大小整除
        else: # 否则
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，因此在多个张量并行GPU间复制
            # the KV heads across multiple tensor parallel GPUs.  # KV头
            assert tp_size % self.total_num_kv_heads == 0 # 确保TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size) # 每个并行分片的KV头数
        self.head_dim = config.hidden_size // self.total_num_heads # 每个头的维度
        self.q_size = self.num_heads * self.head_dim # Q的总维度
        self.kv_size = self.num_kv_heads * self.head_dim # KV的总维度
        self.scaling = self.head_dim**-0.5 # 缩放因子

        self.query_key_value = QKVParallelLinear( # QKV联合投影层
            self.hidden_size, # 输入维度
            self.head_dim, # 每个头的维度
            self.total_num_heads, # 总Q头数
            self.total_num_kv_heads, # 总KV头数
            bias=config.add_bias_linear or config.add_qkv_bias, # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("query_key_value", prefix), # 参数名前缀
        )
        self.dense = RowParallelLinear( # 输出投影层
            self.total_num_heads * self.head_dim, # 输入维度
            config.hidden_size, # 输出维度
            bias=config.add_bias_linear, # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("dense", prefix), # 参数名前缀
        )

        # https://huggingface.co/THUDM/chatglm3-6b-32k/blob/e210410255278dd9d74463cf396ba559c0ef801c/modeling_chatglm.py#L141  # ChatGLM3参考链接
        rope_ratio = getattr(config, "rope_ratio", 1.0) # RoPE比率，默认1.0
        max_positions = getattr(config, "seq_length", 8192) # 最大序列长度，默认8192
        self.rotary_emb = get_rope( # 旋转位置编码
            self.head_dim, # 每个头的维度
            rotary_dim=self.head_dim // 2, # 旋转维度为头维度的一半
            max_position=max_positions, # 最大位置数
            base=10000 * rope_ratio, # 基数乘以比率
            is_neox_style=False, # 非NeoX风格
        )
        self.attn = RadixAttention( # 基数注意力层
            self.num_heads, # 注意力头数
            self.head_dim, # 每个头的维度
            self.scaling, # 缩放因子
            num_kv_heads=self.num_kv_heads, # KV头数
            layer_id=layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("attn", prefix), # 参数名前缀
        )

    def forward( # 前向传播：计算QKV投影、旋转位置编码和注意力
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        position_ids: torch.Tensor, # 位置ID
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states) # 计算QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1) # 拆分QKV
        q, k = self.rotary_emb(position_ids, q, k) # 应用旋转位置编码
        context_layer = self.attn( # 执行注意力计算
            q, # 查询
            k, # 键
            v, # 值
            forward_batch, # 前向批次信息
        )
        attn_output, _ = self.dense(context_layer) # 通过输出投影层
        return attn_output # 返回注意力输出


class GLMMLP(nn.Module): # GLM MLP类，实现前馈神经网络
    """MLP.  # MLP

    MLP will take the input with h hidden state, project it to 4*h  # MLP将输入h维隐藏状态投影到4*h维
    hidden dimension, perform nonlinear transformation, and project the  # 隐藏维度，执行非线性变换，然后将
    state back into h hidden dimension.  # 状态投影回h维隐藏维度
    """

    def __init__( # 初始化GLM MLP层
        self,
        config, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化

        self.add_bias = config.add_bias_linear # 是否添加偏置

        # Project to 4h.  # 投影到4h维度
        self.dense_h_to_4h = MergedColumnParallelLinear( # 升维投影层
            config.hidden_size, # 输入维度
            [config.ffn_hidden_size] * 2, # 输出维度（gate和up各一个ffn_hidden_size）
            bias=config.add_bias_linear, # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("dense_h_to_4h", prefix), # 参数名前缀
        )

        self.activation_func = SiluAndMul() # SiLU与乘法激活函数

        # Project back to h.  # 投影回h维度
        self.dense_4h_to_h = RowParallelLinear( # 降维投影层
            config.ffn_hidden_size, # 输入维度（FFN隐藏维度）
            config.hidden_size, # 输出维度（隐藏维度）
            bias=config.add_bias_linear, # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("dense_4h_to_h", prefix), # 参数名前缀
        )

    def forward(self, hidden_states): # 前向传播：升维→激活→降维
        # [s, b, 4hp]  # [序列长度, 批次大小, 4倍隐藏维度*并行度]
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states) # 通过升维投影
        intermediate_parallel = self.activation_func(intermediate_parallel) # 通过激活函数
        # [s, b, h]  # [序列长度, 批次大小, 隐藏维度]
        output, _ = self.dense_4h_to_h(intermediate_parallel) # 通过降维投影
        return output # 返回输出


class GLMBlock(nn.Module): # GLM Transformer块类，包含注意力和MLP
    """A single transformer layer.  # 单个Transformer层

    Transformer layer takes input with size [s, b, h] and returns an  # Transformer层接收大小为[s, b, h]的输入并返回
    output of the same size.  # 相同大小的输出
    """

    def __init__( # 初始化GLM Transformer块
        self,
        config, # 模型配置
        layer_id: int, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.apply_residual_connection_post_layernorm = ( # 是否在层归一化后应用残差连接
            config.apply_residual_connection_post_layernorm
        )

        self.fp32_residual_connection = config.fp32_residual_connection # 是否使用FP32残差连接

        layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm # 根据配置选择归一化函数
        # Layernorm on the input data.  # 对输入数据进行层归一化
        self.input_layernorm = layer_norm_func( # 输入层归一化
            config.hidden_size, eps=config.layernorm_epsilon # 隐藏维度和epsilon
        )

        # Self attention.  # 自注意力层
        self.self_attention = GLMAttention(
            config, layer_id, quant_config, prefix=add_prefix("self_attention", prefix) # 配置、层ID、量化配置和前缀
        )
        self.hidden_dropout = config.hidden_dropout # 隐藏层dropout率

        # Layernorm on the attention output  # 对注意力输出进行层归一化
        self.post_attention_layernorm = layer_norm_func( # 注意力后层归一化
            config.hidden_size, eps=config.layernorm_epsilon # 隐藏维度和epsilon
        )

        # MLP  # MLP层
        self.mlp = GLMMLP(config, quant_config, prefix=add_prefix("mlp", prefix)) # MLP

    def forward( # 前向传播：层归一化→注意力→残差→层归一化→MLP→残差
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        position_ids: torch.Tensor, # 位置ID
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        # hidden_states: [num_tokens, h]  # 隐藏状态: [token数, 隐藏维度]
        # Layer norm at the beginning of the transformer layer.  # Transformer层开始时的层归一化
        layernorm_output = self.input_layernorm(hidden_states) # 输入层归一化
        # Self attention.  # 自注意力
        attention_output = self.self_attention( # 计算自注意力
            hidden_states=layernorm_output, # 归一化后的隐藏状态
            position_ids=position_ids, # 位置ID
            forward_batch=forward_batch, # 前向批次信息
        )

        # Residual connection.  # 残差连接
        if self.apply_residual_connection_post_layernorm: # 如果在层归一化后应用残差
            residual = layernorm_output # 使用归一化后的输出作为残差
        else: # 否则
            residual = hidden_states # 使用原始隐藏状态作为残差

        layernorm_input = residual + attention_output # 残差加上注意力输出

        # Layer norm post the self attention.  # 自注意力后的层归一化
        layernorm_output = self.post_attention_layernorm(layernorm_input) # 对残差结果归一化

        # Second residual connection.  # 第二个残差连接
        if self.apply_residual_connection_post_layernorm: # 如果在层归一化后应用残差
            residual = layernorm_output # 使用归一化后的输出作为残差
        else: # 否则
            residual = layernorm_input # 使用归一化前的输入作为残差

        output = self.mlp(layernorm_output) + residual # MLP输出加残差

        return output # 返回层输出


class GLMTransformer(nn.Module): # GLM Transformer类，包含多个GLMBlock
    """Transformer class."""  # Transformer类

    def __init__( # 初始化GLM Transformer
        self,
        config, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.post_layer_norm = config.post_layer_norm # 是否使用后层归一化

        # Number of layers.  # 层数
        self.num_layers = config.num_layers # 保存层数

        # Transformer layers.  # Transformer层列表
        self.layers = nn.ModuleList( # 创建模块列表
            [
                GLMBlock(
                    config, # 模型配置
                    i, # 层索引
                    quant_config, # 量化配置
                    prefix=add_prefix(f"layers.{i}", prefix), # 参数名前缀
                )
                for i in range(self.num_layers) # 遍历所有层
            ]
        )

        if self.post_layer_norm: # 如果使用后层归一化
            layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm # 选择归一化函数
            # Final layer norm before output.  # 输出前的最终层归一化
            self.final_layernorm = layer_norm_func( # 最终层归一化
                config.hidden_size, eps=config.layernorm_epsilon # 隐藏维度和epsilon
            )

    def forward( # 前向传播：依次通过所有Transformer层
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        position_ids: torch.Tensor, # 位置ID
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        for i in range(self.num_layers): # 遍历所有层
            layer = self.layers[i] # 获取当前层
            hidden_states = layer( # 通过当前层
                hidden_states=hidden_states, # 隐藏状态
                position_ids=position_ids, # 位置ID
                forward_batch=forward_batch, # 前向批次信息
            )
        # Final layer norm.  # 最终层归一化
        if self.post_layer_norm: # 如果使用后层归一化
            hidden_states = self.final_layernorm(hidden_states) # 通过最终层归一化

        return hidden_states # 返回隐藏状态


class ChatGLMM(nn.Module): # ChatGLM模型类，包含嵌入层、编码器和输出层
    def __init__( # 初始化ChatGLM模型
        self,
        config, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化

        self.embedding = VocabParallelEmbedding( # 词嵌入层
            config.padded_vocab_size, # 填充后的词表大小
            config.hidden_size, # 隐藏维度
            prefix=add_prefix("embedding", prefix), # 参数名前缀
        )

        self.num_layers = config.num_layers # Transformer层数
        self.multi_query_group_num = config.multi_query_group_num # 多查询注意力的KV头组数
        self.kv_channels = config.kv_channels # KV通道数
        self.encoder = GLMTransformer( # Transformer编码器
            config, quant_config, add_prefix("encoder", prefix) # 配置、量化配置和前缀
        )

        self.output_layer = ParallelLMHead( # 语言模型输出头
            config.padded_vocab_size, # 填充后的词表大小
            config.hidden_size, # 隐藏维度
            prefix=add_prefix("output_layer", prefix), # 参数名前缀
        )

    def forward( # 前向传播：嵌入→编码
        self,
        input_ids: torch.Tensor, # 输入token ID
        position_ids: torch.Tensor, # 位置ID
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        inputs_embeds = self.embedding(input_ids) # 计算词嵌入

        # Run encoder.  # 运行编码器
        hidden_states = self.encoder( # 通过编码器
            hidden_states=inputs_embeds, # 嵌入后的输入
            position_ids=position_ids, # 位置ID
            forward_batch=forward_batch, # 前向批次信息
        )
        return hidden_states # 返回编码器输出


class ChatGLMForCausalLM(nn.Module): # ChatGLM因果语言模型类
    packed_modules_mapping = { # 打包模块映射
        "query_key_value": ["query_key_value"], # QKV映射
        "dense_h_to_4h": ["dense_h_to_4h"], # 升维投影映射
    }
    # LoRA specific attributes  # LoRA特定属性
    supported_lora_modules = [ # 支持LoRA的模块列表
        "query_key_value", # QKV投影
        "dense", # 输出投影
        "dense_h_to_4h", # 升维投影
        "dense_4h_to_h", # 降维投影
    ]
    embedding_modules = {} # 嵌入模块映射
    embedding_padding_modules = [] # 嵌入填充模块列表

    def __init__( # 初始化ChatGLM因果语言模型
        self,
        config: ChatGLMConfig, # ChatGLM配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.config: ChatGLMConfig = config # 保存配置
        self.quant_config = quant_config # 保存量化配置
        self.max_position_embeddings = getattr(config, "max_sequence_length", 8192) # 最大位置嵌入数
        self.transformer = ChatGLMM( # ChatGLM模型
            config, quant_config, prefix=add_prefix("transformer", prefix) # 配置、量化配置和前缀
        )
        self.lm_head = self.transformer.output_layer # 语言模型头（与输出层共享）
        self.logits_processor = LogitsProcessor(config) # 逻辑处理器

    @torch.no_grad() # 禁用梯度计算
    def forward( # 前向传播：通过模型和逻辑处理器
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置索引
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, forward_batch) # 通过Transformer模型
        return self.logits_processor( # 通过逻辑处理器
            input_ids, hidden_states, self.lm_head, forward_batch # 输入ID、隐藏状态、LM头和批次信息
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载模型权重
        params_dict = dict(self.named_parameters(remove_duplicate=False)) # 获取参数字典（不去重）
        for name, loaded_weight in weights: # 遍历所有权重
            if "rotary_pos_emb.inv_freq" in name: # 如果是旋转位置编码的逆频率
                continue # 跳过（使用自定义RoPE实现）
            if "word_embeddings" in name: # 如果是词嵌入
                name = name.replace(".word_embeddings", "") # 移除word_embeddings前缀
            # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
            if name.endswith(".bias") and name not in params_dict: # 如果是偏置且不在参数字典中
                continue # 跳过
            param = params_dict[name] # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取权重加载器
            weight_loader(param, loaded_weight) # 加载权重


class ChatGLMModel(ChatGLMForCausalLM): # ChatGLM模型类，继承自ChatGLMForCausalLM
    pass # 无额外实现


EntryClass = [ChatGLMModel] # 入口类列表，用于模型注册
