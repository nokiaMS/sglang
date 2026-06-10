# 文件说明：Ernie4.5模型的推理实现，兼容baidu/ERNIE-4.5-*-PT权重
# 包含MoE门控、MoE层、解码器层、模型主体及因果语言模型等核心组件

# Copyright 2023-2025 SGLang Team
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
# limitations under the License. # 详见License限制条件
# ==============================================================================

"""Inference-only Ernie4.5 model compatible with baidu/ERNIE-4.5-*-PT weights.""" # 仅推理的Ernie4.5模型，兼容百度ERNIE-4.5-*-PT权重

from typing import Iterable, List, Optional, Tuple, Union # 导入类型提示工具

import torch # 导入PyTorch库
import torch.nn.functional as F # 导入PyTorch函数式API
from torch import nn # 导入神经网络模块
from transformers.models.ernie4_5_moe.configuration_ernie4_5_moe import (
    Ernie4_5_MoeConfig, # 导入Ernie4.5 MoE配置类
)

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size, # 获取张量并行世界大小
    tensor_model_parallel_all_reduce, # 张量并行全归约操作
)
from sglang.srt.layers.communicator import enable_moe_dense_fully_dp # 导入MoE密集全DP使能函数
from sglang.srt.layers.layernorm import RMSNorm # 导入RMS归一化层
from sglang.srt.layers.logits_processor import LogitsProcessor # 导入logits处理器
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class # 导入MoE实现类获取函数
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE # 导入融合MoE层
from sglang.srt.layers.moe.topk import TopK # 导入TopK选择层
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead, # 导入并行语言模型头
    VocabParallelEmbedding, # 导入词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批信息
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP as Ernie4MLP # 导入DeepseekV2 MLP作为Ernie4 MLP
from sglang.srt.models.llama import LlamaAttention as Ernie4Attention # 导入Llama注意力作为Ernie4注意力
from sglang.srt.utils import add_prefix, is_npu, make_layers # 导入工具函数：添加前缀、判断NPU、创建层
from sglang.srt.utils.hf_transformers_utils import get_rope_config # 导入RoPE配置获取函数

_is_npu = is_npu() # 判断当前是否为NPU环境


class MoEGate(nn.Module): # MoE门控网络模块，用于计算专家路由logits
    def __init__( # MoE门控初始化
        self,
        config, # 模型配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.weight = nn.Parameter( # 门控权重参数
            torch.empty((config.moe_num_experts, config.hidden_size)) # 形状：(专家数, 隐藏维度)
        )
        self.e_score_correction_bias = nn.Parameter( # 专家分数校正偏置
            torch.empty((1, config.moe_num_experts)) # 形状：(1, 专家数)
        )

    def forward(self, hidden_states): # MoE门控前向传播，计算路由logits
        logits = F.linear(hidden_states, self.weight, None) # 线性变换计算路由logits
        return logits # 返回路由logits


class Ernie4Moe(nn.Module): # Ernie4 MoE模块，包含门控、TopK选择和专家网络
    def __init__( # Ernie4 MoE初始化
        self,
        config: Ernie4_5_MoeConfig, # Ernie4.5 MoE配置
        layer_id: int, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.layer_id = layer_id # 保存层ID
        self.tp_size = get_tensor_model_parallel_world_size() # 获取张量并行世界大小
        self.moe_num_shared_experts = getattr(config, "moe_num_shared_experts", 0) # 获取共享专家数，默认0

        if config.hidden_act != "silu": # 检查激活函数是否为silu
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now." # 仅支持silu激活函数
            )

        self.gate = MoEGate(config=config, prefix=add_prefix("gate", prefix)) # 创建MoE门控

        correction_bias = self.gate.e_score_correction_bias # 获取校正偏置
        # npu only supports 1D, but current correction_bias is 2D # NPU仅支持1D，但当前校正偏置为2D
        if _is_npu: # 如果是NPU环境
            correction_bias = correction_bias.squeeze(0) # 将2D偏置压缩为1D
        self.topk = TopK( # 创建TopK选择器
            top_k=config.moe_k, # Top-K值
            layer_id=layer_id, # 层ID
            renormalize=True, # 启用重归一化
            use_grouped_topk=False, # 不使用分组TopK
            correction_bias=correction_bias, # 校正偏置
        )

        self.experts = get_moe_impl_class(quant_config)( # 创建专家网络实现
            num_experts=config.moe_num_experts, # 专家数量
            top_k=config.moe_k, # Top-K值
            hidden_size=config.hidden_size, # 隐藏维度
            intermediate_size=config.moe_intermediate_size, # 中间维度
            layer_id=self.layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("experts", prefix), # 参数前缀
        )

        if self.moe_num_shared_experts > 0: # 如果存在共享专家
            intermediate_size = ( # 计算共享专家中间维度
                config.moe_intermediate_size * config.moe_num_shared_experts
            )
            # disable tp for shared experts when enable deepep moe # 启用DeepEP MoE时禁用共享专家的张量并行
            self.shared_experts = Ernie4MLP( # 创建共享专家MLP
                hidden_size=config.hidden_size, # 隐藏维度
                intermediate_size=intermediate_size, # 中间维度
                hidden_act=config.hidden_act, # 激活函数
                quant_config=quant_config, # 量化配置
                reduce_results=False, # 不归约结果（因为后续还需加和）
                prefix=add_prefix("shared_experts", prefix), # 参数前缀
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: # Ernie4 MoE前向传播入口
        return self.forward_normal(hidden_states) # 调用标准前向传播

    def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor: # Ernie4 MoE标准前向传播
        shared_output = ( # 计算共享专家输出
            self.shared_experts(hidden_states)
            if self.moe_num_shared_experts > 0
            else None # 无共享专家时为None
        )
        # router_logits: (num_tokens, n_experts) # 路由logits形状：(token数, 专家数)
        router_logits = self.gate(hidden_states) # 通过门控计算路由logits
        topk_output = self.topk(hidden_states, router_logits) # TopK选择专家
        final_hidden_states = self.experts( # 计算专家网络输出
            hidden_states=hidden_states, topk_output=topk_output
        )
        if shared_output is not None: # 如果存在共享专家输出
            final_hidden_states = final_hidden_states + shared_output # 将共享专家输出加到最终结果
        if self.tp_size > 1: # 如果张量并行度大于1
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states) # 全归约同步
        return final_hidden_states # 返回最终隐藏状态


class Ernie4DecoderLayer(nn.Module): # Ernie4解码器层，包含自注意力和MLP/MoE
    """A single transformer layer. # 单个Transformer层

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size. # Transformer层接收[s, b, h]大小的输入并返回相同大小的输出
    """

    def __init__( # Ernie4解码器层初始化
        self,
        config, # 模型配置
        layer_id: int, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
        is_mtp: bool = False, # 是否为MTP（多token预测）层
    ):
        super().__init__() # 调用父类初始化
        rope_theta, rope_scaling = get_rope_config(config) # 获取RoPE配置参数
        rope_is_neox_style = getattr(config, "rope_is_neox_style", False) # 获取RoPE是否为Neox风格
        # Self attention. # 自注意力
        self.self_attn = Ernie4Attention( # 创建自注意力层
            config=config, # 模型配置
            hidden_size=config.hidden_size, # 隐藏维度
            num_heads=config.num_attention_heads, # 注意力头数
            num_kv_heads=config.num_key_value_heads, # KV头数
            layer_id=layer_id, # 层ID
            rope_theta=rope_theta, # RoPE基频
            rope_scaling=rope_scaling, # RoPE缩放
            rope_is_neox_style=rope_is_neox_style, # RoPE Neox风格
            max_position_embeddings=config.max_position_embeddings, # 最大位置编码
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("self_attn", prefix), # 参数前缀
            bias=config.use_bias, # 是否使用偏置
        )
        moe_layer_start_index = getattr( # 获取MoE层起始索引
            config, "moe_layer_start_index", config.num_hidden_layers
        )
        moe_layer_end_index = getattr( # 获取MoE层结束索引
            config, "moe_layer_end_index", config.num_hidden_layers - 1
        )
        # MLP # MLP模块
        if (not is_mtp) and ( # 非MTP层且满足MoE层条件
            moe_layer_start_index <= layer_id <= moe_layer_end_index
            and (layer_id - moe_layer_start_index) % config.moe_layer_interval == 0
        ):
            self.mlp = Ernie4Moe( # 使用MoE作为MLP
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else: # 否则使用普通MLP
            if enable_moe_dense_fully_dp(): # 启用MoE密集全DP时
                mlp_tp_rank, mlp_tp_size = 0, 1 # 设置TP秩和大小为0和1
            else:
                mlp_tp_rank, mlp_tp_size = None, None # 不指定TP秩和大小
            self.mlp = Ernie4MLP( # 使用普通MLP
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank, # TP秩
                tp_size=mlp_tp_size, # TP大小
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 输入层归一化
        self.post_attention_layernorm = RMSNorm( # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward( # Ernie4解码器层前向传播
        self,
        positions: torch.Tensor, # 位置编码
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批信息
        residual: Optional[torch.Tensor], # 残差连接
    ) -> Tuple[torch.Tensor, torch.Tensor]: # 返回隐藏状态和残差
        # Self Attention # 自注意力
        if residual is None: # 无残差（第一层）
            residual = hidden_states # 保存隐藏状态作为残差
            hidden_states = self.input_layernorm(hidden_states) # 对隐藏状态做层归一化
        else: # 有残差
            hidden_states, residual = self.input_layernorm(hidden_states, residual) # 层归一化并更新残差
        hidden_states = self.self_attn( # 自注意力计算
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected # 全连接层
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual) # 注意力后层归一化
        hidden_states = self.mlp(hidden_states) # MLP/MoE计算

        return hidden_states, residual # 返回隐藏状态和残差


class Ernie4Model(nn.Module): # Ernie4模型主体，包含嵌入层、解码器层和最终归一化
    def __init__( # Ernie4模型初始化
        self,
        config: Ernie4_5_MoeConfig, # Ernie4.5 MoE配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.embed_tokens = VocabParallelEmbedding( # 创建词表并行嵌入层
            config.vocab_size, # 词表大小
            config.hidden_size, # 隐藏维度
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("embed_tokens", prefix), # 参数前缀
        )
        self.layers = make_layers( # 创建解码器层列表
            config.num_hidden_layers, # 隐藏层数
            lambda idx, prefix: Ernie4DecoderLayer( # 每层的构造函数
                config=config, layer_id=idx, quant_config=quant_config, prefix=prefix
            ),
            prefix="model.layers", # 参数前缀
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 最终归一化层

    @torch.no_grad() # 禁用梯度计算
    def forward( # Ernie4模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批信息
        input_embeds: torch.Tensor = None, # 输入嵌入（可选）
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]: # 返回隐藏状态
        if input_embeds is None: # 无输入嵌入
            hidden_states = self.embed_tokens(input_ids) # 通过嵌入层获取隐藏状态
        else: # 有输入嵌入
            hidden_states = input_embeds # 直接使用输入嵌入
        residual = None # 初始化残差为None
        for layer in self.layers: # 遍历每一层
            hidden_states, residual = layer( # 前向传播每一层
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual) # 最终归一化

        return hidden_states # 返回隐藏状态


class Ernie4_5_ForCausalLM(nn.Module): # Ernie4.5因果语言模型，非MoE版本
    packed_modules_mapping = { # 打包模块映射
        "qkv_proj": ["q_proj", "k_proj", "v_proj"], # QKV投影打包
        "gate_up_proj": ["gate_proj", "up_proj"], # gate和up投影打包
    }
    stacked_params_mapping = [ # 堆叠参数映射
        # (param_name, weight_name, shard_id) # (参数名, 权重名, 分片ID)
        (".qkv_proj", ".q_proj", "q"), # Q投影
        (".qkv_proj", ".k_proj", "k"), # K投影
        (".qkv_proj", ".v_proj", "v"), # V投影
        (".gate_up_proj", ".gate_proj", 0), # gate投影
        (".gate_up_proj", ".up_proj", 1), # up投影
    ]

    def __init__( # Ernie4.5因果语言模型初始化
        self,
        config: Ernie4_5_MoeConfig, # Ernie4.5 MoE配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.config: Ernie4_5_MoeConfig = config # 保存配置
        self.quant_config = quant_config # 保存量化配置
        self.model = Ernie4Model(config, quant_config, add_prefix("model", prefix)) # 创建模型主体
        if config.tie_word_embeddings: # 如果绑定词嵌入
            self.lm_head = self.model.embed_tokens # 语言模型头共享嵌入层
        else: # 否则
            self.lm_head = ParallelLMHead( # 创建并行语言模型头
                config.vocab_size, # 词表大小
                config.hidden_size, # 隐藏维度
                quant_config=quant_config, # 量化配置
                prefix="lm_head", # 参数前缀
            )
        self.logits_processor = LogitsProcessor(config) # 创建logits处理器

    @torch.no_grad() # 禁用梯度计算
    def forward( # Ernie4.5因果语言模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批信息
    ) -> torch.Tensor: # 返回logits
        hidden_states = self.model(input_ids, positions, forward_batch) # 获取模型隐藏状态
        return self.logits_processor( # 处理logits
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载权重（非MoE版本）
        params_dict = dict(self.named_parameters()) # 获取参数字典
        for name, loaded_weight in weights: # 遍历权重
            if self.config.tie_word_embeddings and "lm_head.weight" in name: # 绑定词嵌入时跳过lm_head
                continue
            for param_name, weight_name, shard_id in self.stacked_params_mapping: # 遍历堆叠参数映射
                if weight_name not in name: # 权重名不匹配则跳过
                    continue
                name = name.replace(weight_name, param_name) # 替换为堆叠参数名
                param = params_dict[name] # 获取参数
                weight_loader = param.weight_loader # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id) # 加载权重分片
                break
            else: # 非堆叠参数
                if name in params_dict.keys(): # 参数名存在于参数字典
                    param = params_dict[name] # 获取参数
                    weight_loader = getattr( # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight) # 加载权重
                else:
                    raise KeyError(f"Parameter '{name}' not found in model.") # 参数未找到则抛出异常

    def get_embed_and_head(self): # 获取嵌入层和语言模型头权重
        return self.model.embed_tokens.weight, self.lm_head.weight


class Ernie4_5_MoeForCausalLM(Ernie4_5_ForCausalLM): # Ernie4.5 MoE因果语言模型，继承自非MoE版本
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载权重（MoE版本）
        expert_params_mapping = FusedMoE.make_expert_params_mapping( # 创建专家参数映射
            ckpt_gate_proj_name="gate_proj", # 检查点gate投影名
            ckpt_down_proj_name="down_proj", # 检查点down投影名
            ckpt_up_proj_name="up_proj", # 检查点up投影名
            num_experts=self.config.moe_num_experts, # 专家数量
        )
        params_dict = dict(self.named_parameters()) # 获取参数字典
        for name, loaded_weight in weights: # 遍历权重
            if self.config.tie_word_embeddings and "lm_head.weight" in name: # 绑定词嵌入时跳过lm_head
                continue
            if name.startswith("model.mtp_"): # 跳过MTP层权重
                continue
            if "moe_statics.e_score_correction_bias" in name: # 替换MoE统计中的校正偏置名称
                name = name.replace("moe_statics", "gate")
            for param_name, weight_name, shard_id in self.stacked_params_mapping: # 遍历堆叠参数映射
                if weight_name not in name: # 权重名不匹配则跳过
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint. # 检查点中有mlp.experts[0].gate_proj
                # Since we handle the experts below in expert_params_mapping, # 因为我们在下面的expert_params_mapping中处理专家
                # we need to skip here BEFORE we update the name, otherwise # 需要在更新名称前跳过，否则
                # name will be updated to mlp.experts[0].gate_up_proj, which # 名称会被更新为mlp.experts[0].gate_up_proj
                # will then be updated below in expert_params_mapping # 然后在下面的expert_params_mapping中被更新
                # for mlp.experts[0].gate_gate_up_proj, which breaks load. # 变为mlp.experts[0].gate_gate_up_proj，导致加载失败
                if ("mlp.experts." in name) and name not in params_dict: # 专家权重且不在参数字典中则跳过
                    continue
                name = name.replace(weight_name, param_name) # 替换为堆叠参数名
                param = params_dict[name] # 获取参数
                weight_loader = param.weight_loader # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id) # 加载权重分片
                break
            else: # 非堆叠参数，尝试专家参数映射
                for mapping in expert_params_mapping: # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping # 解包映射
                    if weight_name not in name: # 权重名不匹配则跳过
                        continue
                    name = name.replace(weight_name, param_name) # 替换为专家参数名
                    if name in params_dict.keys(): # 参数名存在于参数字典
                        param = params_dict[name] # 获取参数
                        weight_loader = param.weight_loader # 获取权重加载器
                        weight_loader( # 加载专家权重
                            param,
                            loaded_weight,
                            name, # 权重名称
                            shard_id=shard_id, # 分片ID
                            expert_id=expert_id, # 专家ID
                        )
                    else:
                        raise KeyError( # 参数未找到则抛出异常
                            f"Parameter '{name}'(replaced) not found in model."
                        )
                    break
                else: # 非专家参数
                    if name in params_dict.keys(): # 参数名存在于参数字典
                        param = params_dict[name] # 获取参数
                        weight_loader = getattr( # 获取权重加载器
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight) # 加载权重
                    else:
                        raise KeyError(f"Parameter '{name}' not found in model.") # 参数未找到则抛出异常


EntryClass = [Ernie4_5_MoeForCausalLM, Ernie4_5_ForCausalLM] # 模型入口类列表，包含MoE和非MoE版本
