# Qwen3-MoE MTP 推测解码模型实现
# 本文件实现了基于 Qwen3-MoE 的 MTP（Multi-Token Prediction）推测解码模型，
# 用于加速 MoE 模型的推理。MTP 通过预测多个 token 来提高推理吞吐量。
# Copyright 2023-2024 SGLang Team  # SGLang 团队版权
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

"""Inference-only Qwen3-MoE MTP speculative decoding."""  # 仅推理的 Qwen3-MoE MTP 推测解码

import logging  # 导入日志模块
from typing import Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 框架
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 导入分布式函数
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入专家分布记录器
from sglang.srt.layers.layernorm import RMSNorm  # 导入 RMS 归一化
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合 MoE 层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行语言模型头
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel  # 导入 Qwen3-MoE 模型
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

logger = logging.getLogger(__name__)  # 获取日志记录器


class Qwen3MoeForCausalLMMTP(Qwen3MoeForCausalLM):
    """Qwen3-MoE MTP 推测解码模型"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化 Qwen3-MoE MTP 推测解码模型"""
        nn.Module.__init__(self)  # 直接调用 nn.Module 初始化
        self.config = config  # 保存配置
        config.num_hidden_layers = 1  # MTP 仅使用 1 层
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        self.quant_config = quant_config  # 保存量化配置
        self.pp_group = get_pp_group()  # 流水线并行组

        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)  # 特征融合线性层
        self.pre_fc_norm_embedding = RMSNorm(  # 嵌入预融合归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏维度和 epsilon
        )
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 隐藏状态预融合归一化
        self.model = Qwen3MoeModel(  # 创建 Qwen3-MoE 模型主体
            config, quant_config, prefix=add_prefix("model", prefix)  # 传入配置、量化配置和前缀
        )
        self.lm_head = ParallelLMHead(  # 并行语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("lm_head", prefix),  # 参数前缀
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力 TP 组
        )
        self.logits_processor = LogitsProcessor(config)  # logits 处理器

        # Required by Qwen3MoeForCausalLM.load_weights(), which we reuse below.  # Qwen3MoeForCausalLM.load_weights() 所需
        self.stacked_params_mapping = [  # 堆叠参数映射表
            ("qkv_proj", "q_proj", "q"),  # Q 映射
            ("qkv_proj", "k_proj", "k"),  # K 映射
            ("qkv_proj", "v_proj", "v"),  # V 映射
            ("gate_up_proj", "gate_proj", 0),  # gate 映射
            ("gate_up_proj", "up_proj", 1),  # up 映射
        ]
        self.expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 专家参数映射
            ckpt_gate_proj_name="gate_proj",  # gate 投影检查点名
            ckpt_down_proj_name="down_proj",  # down 投影检查点名
            ckpt_up_proj_name="up_proj",  # up 投影检查点名
            num_experts=self.config.num_experts,  # 专家数量
        )
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态

    def set_embed_and_head(self, embed, head):
        """设置嵌入层和语言模型头的权重（共享目标模型的权重）"""
        del self.model.embed_tokens.weight  # 删除原有嵌入权重
        del self.lm_head.weight  # 删除原有语言模型头权重
        self.model.embed_tokens.weight = embed  # 设置新嵌入权重
        self.lm_head.weight = head  # 设置新语言模型头权重
        torch.cuda.empty_cache()  # 清空 GPU 缓存
        torch.cuda.synchronize()  # 同步 CUDA

    @torch.no_grad()  # 禁用梯度计算
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """MTP 前向传播：嵌入 -> 特征融合 -> MoE 模型 -> logits"""
        if input_embeds is None:  # 如果没有输入嵌入
            input_embeds = self.model.embed_tokens(input_ids)  # 通过词嵌入层

        hidden_states = forward_batch.spec_info.hidden_states  # 获取推测信息的隐藏状态

        if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
            input_embeds = self.pre_fc_norm_embedding(input_embeds)  # 归一化嵌入
            hidden_states = self.pre_fc_norm_hidden(hidden_states)  # 归一化隐藏状态
        hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))  # 特征融合

        with get_global_expert_distribution_recorder().disable_this_region():  # 禁用专家分布记录
            hidden_states = self.model(  # 通过 MoE 模型
                input_ids,  # 输入 ID
                positions,  # 位置信息
                forward_batch,  # 前向批次
                hidden_states,  # 隐藏状态
            )

        return self.logits_processor(  # 处理 logits
            input_ids, hidden_states, self.lm_head, forward_batch  # 输入 ID、隐藏状态、语言模型头和批次
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        """加载 MTP 模型权重，标记为 MTP 模式"""
        return super().load_weights(weights, is_mtp=True)  # 以 MTP 模式加载权重


EntryClass = [Qwen3MoeForCausalLMMTP]  # 模型入口类列表
