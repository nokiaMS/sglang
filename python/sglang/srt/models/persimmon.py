# Persimmon模型推理实现文件
# 本文件实现了Persimmon大语言模型的推理架构
# 包含MLP、注意力层、解码器层、模型主体及因果语言模型等组件

from collections.abc import Iterable  # 导入可迭代类型
from typing import Optional  # 导入可选类型

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PersimmonConfig  # 导入Persimmon配置

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 导入分布式工具
from sglang.srt.layers.activation import get_act_fn  # 导入获取激活函数工具
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,  # 列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入logits处理器
from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer  # 导入流水线并行缺失层
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 并行词表嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix, make_layers  # 导入前缀添加和层创建工具


class PersimmonMLP(nn.Module):  # Persimmon模型的MLP模块

    def __init__(  # 初始化函数
        self, config: PersimmonConfig, quant_config: Optional[QuantizationConfig] = None  # 配置和量化配置
    ):
        super().__init__()  # 调用父类初始化
        self.dense_h_to_4h = ColumnParallelLinear(  # 从隐藏层到4倍隐藏层的列并行线性层
            config.hidden_size, config.intermediate_size, quant_config=quant_config  # 传入参数
        )
        self.dense_4h_to_h = RowParallelLinear(  # 从4倍隐藏层到隐藏层的行并行线性层
            config.intermediate_size, config.hidden_size, quant_config=quant_config  # 传入参数
        )
        self.act = get_act_fn(config.hidden_act)  # 获取激活函数

    def forward(self, hidden_states) -> torch.Tensor:  # 前向传播函数，执行MLP计算
        hidden_states, _ = self.dense_h_to_4h(hidden_states)  # 上投影
        hidden_states = self.act(hidden_states)  # 应用激活函数
        hidden_states, _ = self.dense_4h_to_h(hidden_states)  # 下投影
        return hidden_states  # 返回隐藏状态


class PersimmonAttention(nn.Module):  # Persimmon模型的注意力模块

    def __init__(  # 初始化函数
        self,
        config: PersimmonConfig,  # Persimmon配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
        layer_id: int = 0,  # 层ID
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        tensor_parallel_world_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小

        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.total_num_heads = config.num_attention_heads  # 总注意力头数
        self.num_heads = self.total_num_heads // tensor_parallel_world_size  # 每个TP秩的头数
        self.head_dim = self.hidden_size // self.total_num_heads  # 每个头的维度
        self.max_position_embeddings = config.max_position_embeddings  # 最大位置嵌入数
        self.rope_theta = config.rope_parameters["rope_theta"]  # RoPE theta
        self.partial_rotary_factor = config.partial_rotary_factor  # 部分旋转因子
        self.is_causal = True  # 是否因果注意力

        assert (self.head_dim * self.total_num_heads) == self.hidden_size  # 断言维度匹配
        assert self.total_num_heads % tensor_parallel_world_size == 0  # 断言头数可被TP大小整除

        self.query_key_value = QKVParallelLinear(  # QKV并行线性层
            self.hidden_size,  # 输入大小
            self.head_dim,  # 每个头的维度
            self.total_num_heads,  # 总头数
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
        )
        self.dense = RowParallelLinear(  # 输出投影行并行线性层
            self.total_num_heads * self.head_dim,  # 输入大小
            self.hidden_size,  # 输出大小
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
        )
        self.is_qk_layernorm = config.qk_layernorm  # 是否使用QK层归一化

        if self.is_qk_layernorm:  # 如果使用QK层归一化
            self.q_layernorm = nn.LayerNorm(self.head_dim)  # Q层归一化
            self.k_layernorm = nn.LayerNorm(self.head_dim)  # K层归一化

        self.rotary_emb = get_rope(  # 获取旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=self.max_position_embeddings,  # 最大位置
            base=self.rope_theta,  # 基础theta
            partial_rotary_factor=self.partial_rotary_factor,  # 部分旋转因子
        )
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.attn = RadixAttention(  # 基数注意力模块
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 添加前缀
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:  # 分割注意力头
        seq_length = x.shape[0]  # 序列长度
        return x.view(seq_length, self.num_heads, self.head_dim)  # 重塑为多头格式

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:  # 合并注意力头
        seq_length = x.shape[0]  # 序列长度
        return x.view(seq_length, self.num_heads * self.head_dim)  # 重塑回扁平格式

    def forward(  # 前向传播函数，执行注意力计算
        self,
        position_ids: torch.Tensor,  # 位置ID
        forward_batch: ForwardBatch,  # 前向批次信息
        hidden_states: torch.Tensor,  # 隐藏状态
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)  # 通过QKV投影
        q, k, v = qkv.chunk(chunks=3, dim=-1)  # 分割为Q、K、V

        if self.is_qk_layernorm:  # 如果使用QK层归一化
            q = self._split_heads(q)  # 分割Q头
            k = self._split_heads(k)  # 分割K头

            q = self.q_layernorm(q)  # Q层归一化
            k = self.k_layernorm(k)  # K层归一化

            q = self._merge_heads(q)  # 合并Q头
            k = self._merge_heads(k)  # 合并K头

        q, k = self.rotary_emb(position_ids, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch=forward_batch)  # 执行注意力计算
        output, _ = self.dense(attn_output)  # 通过输出投影
        return output  # 返回输出


class PersimmonDecoderLayer(nn.Module):  # Persimmon模型的解码器层

    def __init__(  # 初始化函数
        self,
        config: PersimmonConfig,  # Persimmon配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
        idx: int = 0,  # 层索引
    ):
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.self_attn = PersimmonAttention(  # 自注意力模块
            config=config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 添加前缀
            layer_id=idx,  # 层ID
        )
        self.mlp = PersimmonMLP(config, quant_config=quant_config)  # MLP模块
        self.input_layernorm = nn.LayerNorm(  # 输入层归一化
            config.hidden_size, eps=config.layer_norm_eps  # 隐藏层大小和eps
        )
        self.post_attention_layernorm = nn.LayerNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.layer_norm_eps  # 隐藏层大小和eps
        )

    def forward(  # 前向传播函数，执行解码器层计算
        self,
        position_ids: torch.Tensor,  # 位置ID
        forward_batch: ForwardBatch,  # 前向批次信息
        hidden_states: torch.Tensor,  # 隐藏状态
    ) -> torch.Tensor:
        residual = hidden_states  # 保存残差

        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化

        hidden_states = self.self_attn(  # 通过自注意力层
            position_ids=position_ids,  # 位置ID
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 保存残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后层归一化
        hidden_states = self.mlp(hidden_states)  # 通过MLP

        hidden_states = hidden_states + residual  # 残差连接

        outputs = hidden_states  # 输出
        return outputs  # 返回输出


class PersimmonModel(nn.Module):  # Persimmon模型主体

    def __init__(  # 初始化函数
        self,
        config: PersimmonConfig,  # Persimmon配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.pp_group = get_pp_group()  # 获取流水线并行组

        if self.pp_group.is_first_rank:  # 如果是第一个秩
            self.embed_tokens = VocabParallelEmbedding(  # 词表嵌入层
                config.vocab_size, config.hidden_size  # 词表大小和隐藏层大小
            )
        else:  # 否则
            self.embed_tokens = PPMissingLayer()  # 使用缺失层占位

        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,  # 隐藏层数量
            lambda idx, prefix: PersimmonDecoderLayer(  # 解码器层构造函数
                config, quant_config=quant_config, prefix=prefix, idx=idx  # 传入参数
            ),
            prefix="model.layers",  # 前缀
            pp_rank=self.pp_group.rank_in_group,  # 流水线并行秩
            pp_size=self.pp_group.world_size,  # 流水线并行大小
        )

        if self.pp_group.is_last_rank:  # 如果是最后一个秩
            self.final_layernorm = nn.LayerNorm(  # 最终层归一化
                config.hidden_size, eps=config.layer_norm_eps  # 隐藏层大小和eps
            )
        else:  # 否则
            self.final_layernorm = PPMissingLayer()  # 使用缺失层占位

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入
        return self.embed_tokens(input_ids)  # 通过词表嵌入层

    def forward(  # 前向传播函数，执行模型主体计算
        self,
        input_ids: torch.Tensor,  # 输入ID
        forward_batch: ForwardBatch,  # 前向批次信息
        positions: torch.Tensor,  # 位置
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入，可选
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:  # 如果是第一个秩
            if inputs_embeds is not None:  # 如果提供了输入嵌入
                hidden_states = inputs_embeds  # 使用输入嵌入
            else:  # 否则
                hidden_states = self.get_input_embeddings(input_ids)  # 通过词表嵌入层
        else:  # 否则
            hidden_states = forward_batch.pp_input_hidden  # 从流水线输入获取隐藏状态
        for i in range(self.start_layer, self.end_layer):  # 遍历解码器层
            layer = self.layers[i]  # 获取当前层
            hidden_states = layer(  # 通过当前层
                position_ids=positions,  # 位置ID
                forward_batch=forward_batch,  # 前向批次
                hidden_states=hidden_states,  # 隐藏状态
            )
        return self.final_layernorm(hidden_states)  # 应用最终层归一化并返回


class PersimmonForCausalLM(nn.Module):  # Persimmon因果语言模型

    def __init__(  # 初始化函数
        self,
        config: PersimmonConfig,  # Persimmon配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = PersimmonModel(  # 模型主体
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)  # 传入参数
        )
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入
        return self.model.get_input_embeddings(input_ids)  # 通过模型获取嵌入

    def forward(  # 前向传播函数，执行因果语言模型计算
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次信息
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入，可选
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(  # 通过模型主体
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向批次
            positions=positions,  # 位置
            inputs_embeds=inputs_embeds,  # 输入嵌入
        )

        return self.logits_processor(  # 通过logits处理器
            input_ids, hidden_states, self.lm_head, forward_batch  # 传入参数
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):  # 加载权重函数
        params_dict = dict(self.named_parameters())  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历权重
            if "rotary_emb.inv_freq" in name:  # 如果是旋转位置编码逆频率
                continue  # 跳过
            if name not in params_dict:  # 如果参数名不在字典中
                if name == "lm_head.weight":  # 如果是语言模型头权重
                    continue  # 跳过
                print(f"Warning: weight {name} not found in model.")  # 打印警告
                continue  # 继续
            param = params_dict[name]  # 获取参数
            if "query_key_value" in name:  # 如果是QKV权重
                output_dim = getattr(param, "output_dim", None)  # 获取输出维度
                if output_dim is not None:  # 如果输出维度存在
                    loaded_weight_shape = loaded_weight.shape  # 保存原始形状
                    num_heads = self.config.num_attention_heads  # 注意力头数
                    loaded_weight = loaded_weight.view(  # 重塑权重以分离QKV
                        loaded_weight_shape[:output_dim]  # 输出维度前的形状
                        + (num_heads, 3, -1)  # 分离为头数、3（QKV）、每头维度
                        + loaded_weight_shape[output_dim + 1 :]  # 输出维度后的形状
                    )
                    loaded_weight = loaded_weight.transpose(output_dim, output_dim + 1)  # 转置以匹配参数布局
                    loaded_weight = loaded_weight.reshape(loaded_weight_shape)  # 重塑回原始形状
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, loaded_weight)  # 加载权重


EntryClass = PersimmonForCausalLM  # 入口类为PersimmonForCausalLM
