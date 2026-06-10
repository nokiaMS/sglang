# MiMo V2 Next-N预测模型（MTP，Multi-Token Prediction）
# 该模块实现了MiMo V2的多token预测层，用于投机解码
# 主要包含MTP层、Next-N模型和权重加载逻辑
# 核心组件：MiMoV2MTPLayer（MTP层）、MiMoV2ModelNextN（Next-N模型）、MiMoV2MTP（MTP因果语言模型）

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

import logging  # 导入日志模块
from typing import Iterable, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.configs.model_config import get_mimo_v2_fused_qkv_expected_tp_size  # 导入MiMo V2融合QKV期望TP大小
from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入全局专家分布记录器
from sglang.srt.layers.communicator import (  # 导入层通信器
    LayerCommunicator,  # 层通信器
    LayerScatterModes,  # 层散射模式
    enable_moe_dense_fully_dp,  # 启用MoE密集全DP
)
from sglang.srt.layers.dp_attention import (  # 导入DP注意力相关
    get_attention_tp_rank,  # 获取注意力TP排名
    is_dp_attention_enabled,  # 判断是否启用DP注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 并行词嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.mimo_v2 import (  # 从mimo_v2模块导入
    MiMoV2Attention,  # MiMo V2注意力层
    MiMoV2ForCausalLM,  # MiMo V2因果语言模型
    MiMoV2MLP,  # MiMo V2 MLP层
    load_mimo_v2_qkv_proj_weight,  # MiMo V2 QKV投影权重加载函数
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

MiMoV2Config = None  # MiMo V2配置占位符，避免循环导入

logger = logging.getLogger(__name__)  # 创建日志记录器


class MiMoV2MTPLayer(nn.Module):
    """MiMo V2多token预测(MTP)层，用于投机解码中的下一个token预测"""
    def __init__(
        self,
        config: MiMoV2Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.hidden_size = config.hidden_size  # 隐藏层大小

        rope_theta = getattr(config, "rope_theta", 10000)  # 获取RoPE theta参数
        rope_scaling = getattr(config, "rope_scaling", None)  # 获取RoPE缩放参数
        if (  # 如果RoPE缩放类型为default
            isinstance(rope_scaling, dict)
            and rope_scaling.get("rope_type") == "default"
        ):
            rope_scaling = None  # 视为无缩放
        max_position_embeddings = getattr(  # 获取最大位置嵌入数
            config,
            "context_len",  # 优先使用context_len
            getattr(config, "max_position_embeddings", 32768),  # 其次使用max_position_embeddings
        )

        self.self_attn = MiMoV2Attention(  # 自注意力层，使用SWA配置
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_heads=config.swa_num_attention_heads,  # SWA注意力头数
            num_kv_heads=config.swa_num_key_value_heads,  # SWA KV头数
            head_dim=config.swa_head_dim,  # SWA头维度
            v_head_dim=getattr(config, "swa_v_head_dim", None),  # SWA V头维度
            v_scale=getattr(config, "attention_value_scale", None),  # 注意力值缩放
            sliding_window_size=config.sliding_window_size,  # 滑动窗口大小
            attention_bias=config.attention_bias,  # 注意力偏置
            attention_sink_bias=getattr(config, "add_swa_attention_sink_bias", False),  # 注意力汇聚偏置
            layer_id=layer_id,  # 层ID
            rope_theta=getattr(config, "swa_rope_theta", rope_theta),  # SWA RoPE theta
            rope_scaling=rope_scaling,  # RoPE缩放
            max_position_embeddings=max_position_embeddings,  # 最大位置嵌入
            quant_config=quant_config,  # 量化配置
            partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),  # 部分旋转因子
            prefix=add_prefix("self_attn", prefix),  # 前缀
        )
        self.is_layer_sparse = False  # MTP层不是稀疏层
        is_previous_layer_sparse = True  # 上一层是稀疏的
        is_next_layer_sparse = False  # 下一层不是稀疏的

        if enable_moe_dense_fully_dp():  # 如果启用MoE密集全DP
            mlp_tp_rank, mlp_tp_size = 0, 1  # 设置MLP TP排名和大小
        else:  # 不启用
            mlp_tp_rank, mlp_tp_size = None, None  # 使用默认值
        self.mlp = MiMoV2MLP(  # MLP层
            hidden_size=self.hidden_size,  # 隐藏层大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            hidden_act=config.hidden_act,  # 隐藏层激活函数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 前缀
            tp_rank=mlp_tp_rank,  # TP排名
            tp_size=mlp_tp_size,  # TP大小
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.layernorm_epsilon
        )
        self.layer_scatter_modes = LayerScatterModes.init_new(  # 层散射模式
            layer_id=layer_id,  # 层ID
            num_layers=1,  # 只有一层
            is_layer_sparse=self.is_layer_sparse,  # 是否稀疏
            is_previous_layer_sparse=is_previous_layer_sparse,  # 上一层是否稀疏
            is_next_layer_sparse=is_next_layer_sparse,  # 下一层是否稀疏
        )
        self.layer_communicator = LayerCommunicator(  # 层通信器
            layer_scatter_modes=self.layer_scatter_modes,  # 散射模式
            input_layernorm=self.input_layernorm,  # 输入层归一化
            post_attention_layernorm=self.post_attention_layernorm,  # 注意力后归一化
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MTP层前向传播：注意力+MLP"""
        hidden_states, residual = self.layer_communicator.prepare_attn(  # 准备注意力
            hidden_states, residual, forward_batch
        )

        if hidden_states.shape[0] != 0:  # 如果有有效token
            hidden_states = self.self_attn(  # 自注意力计算
                positions=positions,  # 位置
                hidden_states=hidden_states,  # 隐藏状态
                forward_batch=forward_batch,  # 前向批次
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(  # 准备MLP
            hidden_states, residual, forward_batch
        )
        with get_global_expert_distribution_recorder().disable_this_region():  # 禁用专家分布记录
            hidden_states = self.mlp(hidden_states)  # MLP计算
        hidden_states, residual = self.layer_communicator.postprocess_layer(  # 后处理
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual  # 返回隐藏状态和残差


class MiMoV2ModelNextN(nn.Module):
    """MiMo V2 Next-N模型，用于多token预测"""
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.vocab_size = config.vocab_size  # 词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            use_attn_tp_group=is_dp_attention_enabled(),  # 是否使用注意力TP组
            prefix=add_prefix("embed_tokens", prefix),  # 前缀
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)  # 嵌入归一化
        self.hnorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)  # 隐藏状态归一化

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)  # 嵌入-隐藏投影层

        self.mtp_block = MiMoV2MTPLayer(  # MTP块
            config,  # 配置
            0,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("decoder", prefix),  # 前缀
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)  # 最终层归一化

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """Next-N模型前向传播"""
        if input_embeds is None:  # 如果没有预计算的嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入获取隐藏状态
        else:  # 有预计算的嵌入
            hidden_states = input_embeds  # 直接使用
        if hidden_states.shape[0] > 0:  # 如果有有效token
            hidden_states = self.eh_proj(  # 通过嵌入-隐藏投影
                torch.cat(
                    (
                        self.enorm(hidden_states),  # 归一化后的嵌入
                        self.hnorm(forward_batch.spec_info.hidden_states),  # 归一化后的隐藏状态
                    ),
                    dim=-1,  # 在最后一维拼接
                )
            )
        hidden_states, residual = self.mtp_block(  # 通过MTP块
            positions=positions,  # 位置
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
            residual=None,  # 残差为空
        )
        hidden_states_before_norm = None  # 归一化前的隐藏状态
        if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
            if forward_batch.return_hidden_states_before_norm:  # 如果需要返回归一化前的隐藏状态
                hidden_states_before_norm = (  # 计算归一化前的隐藏状态
                    hidden_states if residual is None else hidden_states + residual
                )
            if residual is not None:  # 如果有残差
                hidden_states, _ = self.final_layernorm(hidden_states, residual)  # 融合归一化
            else:  # 没有残差
                hidden_states = self.final_layernorm(hidden_states)  # 普通归一化

        return hidden_states, hidden_states_before_norm  # 返回隐藏状态和归一化前隐藏状态


class MiMoV2MTP(MiMoV2ForCausalLM):
    """MiMo V2多token预测模型，继承自MiMoV2ForCausalLM"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        draft_model_idx: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)  # 直接调用nn.Module的初始化
        self.config = config  # 保存配置
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        self.quant_config = quant_config  # 保存量化配置

        self.model = MiMoV2ModelNextN(  # 创建Next-N模型
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("lm_head", prefix),  # 前缀
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用DP LM头
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """MTP模型前向传播"""
        hidden_states, hidden_states_before_norm = self.model(  # 通过Next-N模型
            input_ids, positions, forward_batch
        )
        return self.logits_processor(  # 通过logits处理器
            input_ids,  # 输入ID
            hidden_states,  # 隐藏状态
            self.lm_head,  # 语言模型头
            forward_batch,  # 前向批次
            hidden_states_before_norm=hidden_states_before_norm,  # 归一化前隐藏状态
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        """加载MTP模型权重"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),  # QKV投影中的Q
            ("qkv_proj", "k_proj", "k"),  # QKV投影中的K
            ("qkv_proj", "v_proj", "v"),  # QKV投影中的V
            ("gate_up_proj", "gate_proj", 0),  # 门控上投影中的门控
            ("gate_up_proj", "up_proj", 1),  # 门控上投影中的上投影
        ]

        params_dict = dict(self.named_parameters())  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历权重
            if "rotary_emb.inv_freq" in name or "projector" in name:  # 跳过旋转嵌入和投影器
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过旋转嵌入缓存
                # Models trained using ColossalAI may include these tensors in the
                # checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 跳过绑定的词嵌入
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:  # 跳过视觉塔
                continue
            name = self.map_model_name_to_mtp_param_name(name)  # 映射模型名称到MTP参数名称

            # Support fused qkv_proj checkpoint (Pro format)
            if "qkv_proj" in name:  # 如果是QKV投影
                if name in params_dict:  # 如果参数存在
                    param = params_dict[name]  # 获取参数
                    load_mimo_v2_qkv_proj_weight(  # 加载融合QKV权重
                        name,
                        param,
                        loaded_weight,
                        expected_fused_tp_size=get_mimo_v2_fused_qkv_expected_tp_size(
                            self.config
                        ),
                    )
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射

                if f".{weight_name}." not in name:  # 如果权重名不在名称中
                    continue
                if "mtp_block" not in name:  # 如果不是MTP块中的参数
                    break
                name = name.replace(f".{weight_name}.", f".{param_name}.")  # 替换权重名
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ额外偏置
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break
            else:  # 非堆叠参数
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ额外偏置
                    continue

                if "mtp_block" not in name and (  # 如果不是MTP块且不是特殊参数
                    "embed_tokens" not in name
                    and "lm_head" not in name
                    and "enorm" not in name
                    and "hnorm" not in name
                    and "eh_proj" not in name
                    and "final_layernorm" not in name
                ):
                    continue
                if name in params_dict.keys():  # 如果参数存在
                    param = params_dict[name]  # 获取参数
                    if "attention_sink_bias" in name:  # 如果是注意力汇聚偏置
                        start = get_attention_tp_rank() * param.numel()  # 计算起始索引
                        param.data.copy_(loaded_weight[start : start + param.numel()])  # 复制对应分片
                    else:  # 其他参数
                        weight_loader = getattr(  # 获取权重加载器
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)  # 加载权重
                else:  # 参数不存在
                    logger.warning(f"Parameter {name} not found in params_dict")  # 打印警告

    def map_model_name_to_mtp_param_name(self, name: str) -> str:
        """将模型参数名称映射为MTP参数名称"""
        import re  # 导入正则表达式模块

        if "pre_mlp_layernorm" in name:  # 如果包含pre_mlp_layernorm
            name = name.replace("pre_mlp_layernorm", "post_attention_layernorm")  # 替换为post_attention_layernorm

        name_without_prefix = [  # 不带前缀的名称列表
            "enorm",  # 嵌入归一化
            "hnorm",  # 隐藏状态归一化
            "eh_proj",  # 嵌入-隐藏投影
            "final_layernorm",  # 最终层归一化
        ]
        pattern = r"model.mtp.layers.(\d+)."  # 匹配MTP层的模式
        group = re.match(pattern, name)  # 匹配模式
        if group is not None:  # 如果匹配成功
            for sub_name in name_without_prefix:  # 遍历不带前缀的名称
                if sub_name in name:  # 如果名称包含子名称
                    name = name.replace(group.group(), "model.")  # 替换为model.前缀
                    return name  # 返回映射后的名称
            name = name.replace(group.group(), "model.mtp_block.")  # 替换为mtp_block前缀
        return name  # 返回映射后的名称

    def get_embed_and_head(self):
        """获取嵌入层和语言模型头"""
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入权重和LM头权重

    def set_embed_and_head(self, embed, head):
        """设置嵌入层和语言模型头"""
        del self.model.embed_tokens.weight  # 删除旧的嵌入权重
        del self.lm_head.weight  # 删除旧的LM头权重
        self.model.embed_tokens.weight = embed  # 设置新的嵌入权重
        self.lm_head.weight = head  # 设置新的LM头权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA操作


EntryClass = MiMoV2MTP  # 入口类
