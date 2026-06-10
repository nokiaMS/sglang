# Phi模型推理实现文件
# 本文件实现了Phi大语言模型的推理架构
# 包含注意力层、MLP、解码器层、模型主体及因果语言模型等组件

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/phi.py
from typing import Iterable, Optional  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PhiConfig  # 导入Phi配置

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 导入分布式工具
from sglang.srt.layers.activation import get_act_fn  # 导入获取激活函数工具
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,  # 列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 并行词表嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix, make_layers  # 导入前缀添加和层创建工具


class PhiAttention(nn.Module):  # Phi模型的注意力模块

    def __init__(  # 初始化函数
        self,
        config: PhiConfig,  # Phi配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
        layer_id: int = 0,  # 层ID
    ):
        super().__init__()  # 调用父类初始化
        self.total_num_heads = config.num_attention_heads  # 总注意力头数
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.head_size = self.hidden_size // self.total_num_heads  # 每个头的维度

        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        assert self.total_num_heads % tensor_model_parallel_world_size == 0  # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size  # 每个TP秩的头数

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性投影层
            self.hidden_size,  # 输入大小
            self.head_size,  # 每个头的维度
            self.total_num_heads,  # 总头数
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
        )
        self.dense = RowParallelLinear(  # 输出投影行并行线性层
            self.hidden_size,  # 输入大小
            self.hidden_size,  # 输出大小
            quant_config=quant_config,  # 量化配置
        )

        scaling = self.head_size**-0.5  # 缩放因子
        rotary_dim = int(  # 旋转维度
            config.partial_rotary_factor  # 部分旋转因子
            * (config.hidden_size // config.num_attention_heads)  # 乘以每个头的维度
        )
        assert rotary_dim % 2 == 0  # 断言旋转维度为偶数

        rope_theta = config.rope_parameters["rope_theta"]  # RoPE theta
        max_position_embeddings = getattr(config, "max_position_embeddings", 2048)  # 最大位置嵌入数
        self.rotary_emb = get_rope(  # 获取旋转位置编码
            self.head_size,  # 头维度
            rotary_dim=rotary_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置
            base=rope_theta,  # 基础theta
        )
        self.attn = RadixAttention(  # 基数注意力模块
            self.num_heads,  # 头数
            self.head_size,  # 头维度
            scaling,  # 缩放因子
            num_kv_heads=self.num_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 添加前缀
        )

    def forward(  # 前向传播函数，执行注意力计算
        self,
        position_ids: torch.Tensor,  # 位置ID
        forward_batch: ForwardBatch,  # 前向批次信息
        hidden_states: torch.Tensor,  # 隐藏状态
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影
        q, k, v = qkv.chunk(chunks=3, dim=-1)  # 分割为Q、K、V
        q, k = self.rotary_emb(position_ids, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch=forward_batch)  # 执行注意力计算
        output, _ = self.dense(attn_output)  # 通过输出投影
        return output  # 返回输出


class PhiMLP(nn.Module):  # Phi模型的MLP模块

    def __init__(  # 初始化函数
        self, config: PhiConfig, quant_config: Optional[QuantizationConfig] = None  # 配置和量化配置
    ):
        super().__init__()  # 调用父类初始化

        n_inner = getattr(config, "n_inner", None)  # 获取内层大小
        n_inner = n_inner if n_inner is not None else 4 * config.hidden_size  # 默认为4倍隐藏大小

        self.fc1 = ColumnParallelLinear(  # 第一个列并行线性层
            config.hidden_size,  # 输入大小
            n_inner,  # 输出大小
            quant_config=quant_config,  # 量化配置
        )
        self.fc2 = RowParallelLinear(  # 第二个行并行线性层
            n_inner,  # 输入大小
            config.hidden_size,  # 输出大小
            quant_config=quant_config,  # 量化配置
        )
        self.act = get_act_fn(config.hidden_act)  # 获取激活函数

    def forward(self, hidden_states):  # 前向传播函数，执行MLP计算
        hidden_states, _ = self.fc1(hidden_states)  # 通过第一个线性层
        hidden_states = self.act(hidden_states)  # 应用激活函数
        hidden_states, _ = self.fc2(hidden_states)  # 通过第二个线性层
        return hidden_states  # 返回隐藏状态


class PhiLayer(nn.Module):  # Phi模型的解码器层

    def __init__(  # 初始化函数
        self,
        config: PhiConfig,  # Phi配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
        idx: int = 0,  # 层索引
    ):
        super().__init__()  # 调用父类初始化
        self.input_layernorm = nn.LayerNorm(  # 输入层归一化
            config.hidden_size, eps=config.layer_norm_eps  # 隐藏层大小和eps
        )
        self.self_attn = PhiAttention(  # 自注意力模块
            config,  # 配置
            quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 添加前缀
            layer_id=idx,  # 层ID
        )
        self.mlp = PhiMLP(config, quant_config)  # MLP模块

    def forward(  # 前向传播函数，执行解码器层计算（并行注意力+MLP+残差）
        self,
        position_ids: torch.Tensor,  # 位置ID
        forward_batch: ForwardBatch,  # 前向批次信息
        hidden_states: torch.Tensor,  # 隐藏状态
    ) -> torch.Tensor:
        residual = hidden_states  # 保存残差
        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化
        attn_outputs = self.self_attn(  # 通过自注意力层
            position_ids=position_ids,  # 位置ID
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )
        feed_forward_hidden_states = self.mlp(hidden_states)  # 通过MLP
        hidden_states = attn_outputs + feed_forward_hidden_states + residual  # 注意力+MLP+残差
        return hidden_states  # 返回隐藏状态


class PhiModel(nn.Module):  # Phi模型主体

    def __init__(  # 初始化函数
        self,
        config: PhiConfig,  # Phi配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.embed_tokens = VocabParallelEmbedding(  # 词表嵌入层
            config.vocab_size, config.hidden_size  # 词表大小和隐藏层大小
        )

        pp_group = get_pp_group()  # 获取流水线并行组
        pp_size = pp_group.world_size  # 流水线并行大小
        pp_rank = pp_group.rank  # 流水线并行秩

        self.start_layer = pp_rank * config.num_hidden_layers // pp_size  # 起始层
        self.end_layer = (pp_rank + 1) * config.num_hidden_layers // pp_size  # 结束层

        self.layers = make_layers(  # 创建解码器层
            config.num_hidden_layers,  # 隐藏层数量
            lambda idx, prefix: PhiLayer(  # 解码器层构造函数
                config, quant_config=quant_config, prefix=prefix, idx=idx  # 传入参数
            ),
            prefix=add_prefix("layers", prefix),  # 添加前缀
        )

        self.final_layernorm = nn.LayerNorm(  # 最终层归一化
            config.hidden_size, eps=config.layer_norm_eps  # 隐藏层大小和eps
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入
        return self.embed_tokens(input_ids)  # 通过词表嵌入层

    def forward(  # 前向传播函数，执行模型主体计算
        self,
        input_ids: torch.Tensor,  # 输入ID
        forward_batch: ForwardBatch,  # 前向批次信息
        positions: torch.Tensor,  # 位置
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入，可选
    ) -> torch.Tensor:
        if inputs_embeds is not None:  # 如果提供了输入嵌入
            hidden_states = inputs_embeds  # 使用输入嵌入
        else:  # 否则
            hidden_states = self.get_input_embeddings(input_ids)  # 通过词表嵌入层
        for i in range(self.start_layer, self.end_layer):  # 遍历解码器层
            layer = self.layers[i]  # 获取当前层

            hidden_states = layer(  # 通过当前层
                position_ids=positions,  # 位置ID
                forward_batch=forward_batch,  # 前向批次
                hidden_states=hidden_states,  # 隐藏状态
            )
        hidden_states = self.final_layernorm(hidden_states)  # 应用最终层归一化
        return hidden_states  # 返回隐藏状态


class PhiForCausalLM(nn.Module):  # Phi因果语言模型
    packed_modules_mapping = {  # 打包模块映射
        "qkv_proj": [  # QKV投影
            "q_proj",  # Q投影
            "k_proj",  # K投影
            "v_proj",  # V投影
        ]
    }

    def __init__(  # 初始化函数
        self,
        config: PhiConfig,  # Phi配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = PhiModel(  # 模型主体
            config=config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("model", prefix),  # 添加前缀
        )

        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            bias=True,  # 使用偏置
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
        weights = dict(weights)  # 转换权重为字典
        loaded_keys = set()  # 已加载键集合

        for name, param in params_dict.items():  # 遍历参数
            if name in loaded_keys:  # 如果已加载
                continue  # 跳过

            # Handle packed weights  # 处理打包权重
            is_packed = False  # 是否已打包标志
            for packed_name, src_names in self.packed_modules_mapping.items():  # 遍历打包映射
                if packed_name not in name:  # 如果打包名不在参数名中
                    continue  # 继续

                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                for src_name in src_names:  # 遍历源名称
                    full_src_name = name.replace(packed_name, src_name)  # 构建完整源名称
                    if full_src_name in weights:  # 如果源名称在权重中
                        loaded_weight = weights[full_src_name]  # 获取权重
                        # The shard_id for QKVParallelLinear is 'q', 'k', 'v'.  # QKV并行线性层的分片ID
                        shard_id = src_name.split("_")[0]  # 获取分片ID
                        weight_loader(param, loaded_weight, shard_id)  # 加载权重
                        loaded_keys.add(full_src_name)  # 添加到已加载集合

                loaded_keys.add(name)  # 添加当前名称到已加载集合
                is_packed = True  # 标记为已打包
                break  # 跳出循环
            if is_packed:  # 如果已打包
                continue  # 继续下一个参数

            # Handle non-packed weights  # 处理非打包权重
            if name not in weights:  # 如果名称不在权重中
                # Redundant with the check in the loop, but good for safety  # 与循环中的检查冗余，但更安全
                continue  # 跳过

            loaded_weight = weights[name]  # 获取权重
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, loaded_weight)  # 加载权重
            loaded_keys.add(name)  # 添加到已加载集合


EntryClass = PhiForCausalLM  # 入口类为PhiForCausalLM
