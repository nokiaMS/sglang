# HunYuan V3 NextN (MTP) 推测解码模型实现
# 本文件实现了 HunYuanV3 模型的 NextN 推测解码（Multi-Token Prediction）模块，
# 用于在推理时预测多个后续 token 以加速解码过程。

# coding=utf-8
# Copyright 2026 The HunYuan team.
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

"""Inference-only HunyuanV3 NextN (MTP) Speculative Decoding."""  # 仅推理用的 HunyuanV3 NextN (MTP) 推测解码模块

import logging  # 导入日志模块
from typing import Iterable, Optional, Tuple  # 导入类型注解工具

import torch  # 导入 PyTorch 深度学习框架
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.layers.layernorm import RMSNorm  # 导入 RMS 归一化层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合 MoE 层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.managers.schedule_batch import ForwardBatch  # 导入前向批处理信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.hunyuan_v3 import HYV3DecoderLayer  # 导入 HunYuanV3 解码器层
from sglang.srt.utils import is_cuda  # 导入 CUDA 检测工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class HYV3ModelNextN(nn.Module):  # HunYuanV3 NextN 模型类，用于推测解码中的多 token 预测

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，用于命名
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存模型配置

        self.embed_tokens = VocabParallelEmbedding(  # 词表并行嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层维度
            prefix=f"{prefix}.embed_tokens",  # 嵌入层参数前缀
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 嵌入归一化层
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 隐藏状态归一化层
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)  # 嵌入-隐藏投影层，将嵌入和隐藏状态拼接后投影

        self.alt_stream = torch.cuda.Stream() if is_cuda() else None  # 创建备用 CUDA 流，用于异步执行

        # Force MoE for the MTP layer: first_k_dense_replace=1 would make
        # layer_id=0 pick a dense MLP instead of MoE, so override it.
        # 强制 MTP 层使用 MoE：first_k_dense_replace=1 会使 layer_id=0 选择密集 MLP 而非 MoE，因此需要覆盖该设置
        orig_first_k = getattr(config, "first_k_dense_replace", 0)  # 保存原始的 first_k_dense_replace 值
        config.first_k_dense_replace = 0  # 临时设置为 0，强制使用 MoE
        self.decoder = HYV3DecoderLayer(  # 解码器层
            config=config,  # 模型配置
            layer_id=0,  # 层 ID 设为 0
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.decoder",  # 解码器参数前缀
            alt_stream=self.alt_stream,  # 传入备用 CUDA 流
        )
        config.first_k_dense_replace = orig_first_k  # 恢复原始的 first_k_dense_replace 值

        self.shared_head = nn.Module()  # 共享头模块
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 共享头的归一化层

    @torch.no_grad()  # 禁用梯度计算，用于推理
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批处理信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选
    ) -> torch.Tensor:  # 返回隐藏状态张量
        if input_embeds is None:  # 如果没有提供输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过词表嵌入层获取隐藏状态
        else:  # 否则
            hidden_states = input_embeds  # 直接使用提供的输入嵌入

        if hidden_states.shape[0] > 0:  # 如果有有效的隐藏状态
            hidden_states = self.eh_proj(  # 通过嵌入-隐藏投影层
                torch.cat(  # 拼接归一化后的嵌入和隐藏状态
                    (
                        self.enorm(hidden_states),  # 对当前嵌入进行归一化
                        self.hnorm(forward_batch.spec_info.hidden_states),  # 对推测信息中的隐藏状态进行归一化
                    ),
                    dim=-1,  # 在最后一维拼接
                )
            )

        residual = None  # 初始化残差为 None
        hidden_states, residual = self.decoder(  # 通过解码器层
            positions, hidden_states, forward_batch, residual  # 传入位置、隐藏状态、批处理信息和残差
        )

        if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
            if residual is not None:  # 如果残差不为 None
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)  # 对隐藏状态和残差进行归一化
            else:  # 否则
                hidden_states = self.shared_head.norm(hidden_states)  # 仅对隐藏状态进行归一化

        return hidden_states  # 返回隐藏状态


class HYV3ForCausalLMNextN(nn.Module):  # HunYuanV3 NextN 因果语言模型类，用于推测解码

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        nn.Module.__init__(self)  # 调用 nn.Module 的初始化
        self.config = config  # 保存模型配置
        self.quant_config = quant_config  # 保存量化配置

        self.model = HYV3ModelNextN(config, quant_config, prefix="model")  # 创建 NextN 模型实例
        self.lm_head = ParallelLMHead(  # 并行语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层维度
            quant_config=quant_config,  # 量化配置
            prefix="lm_head",  # 参数前缀
        )
        self.logits_processor = LogitsProcessor(config)  # 创建 logits 处理器

    @torch.no_grad()  # 禁用梯度计算，用于推理
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批处理信息
    ) -> torch.Tensor:  # 返回 logits 张量
        hidden_states = self.model(input_ids, positions, forward_batch)  # 通过模型获取隐藏状态
        return self.logits_processor(  # 通过 logits 处理器计算并返回 logits
            input_ids, hidden_states, self.lm_head, forward_batch  # 传入输入 ID、隐藏状态、语言模型头和批处理信息
        )

    def get_embed_and_head(self):  # 获取嵌入层和语言模型头的权重
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入权重和语言模型头权重

    def set_embed_and_head(self, embed, head):  # 设置嵌入层和语言模型头的权重
        del self.model.embed_tokens.weight  # 删除旧的嵌入权重
        del self.lm_head.weight  # 删除旧的语言模型头权重
        self.model.embed_tokens.weight = embed  # 设置新的嵌入权重
        self.lm_head.weight = head  # 设置新的语言模型头权重
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        torch.cuda.synchronize()  # 同步 CUDA 操作

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        nextn_layer_id = self.config.num_hidden_layers  # NextN 层的 ID 等于隐藏层数量
        nextn_prefix = f"model.layers.{nextn_layer_id}."  # NextN 层的权重前缀
        spec_weight_names = ("enorm", "hnorm", "eh_proj")  # 推测解码专有的权重名称

        stacked_params_mapping = [  # 堆叠参数映射表，用于合并 gate/up 投影等
            ("qkv_proj", "q_proj", "q"),  # QKV 投影中的 Q 部分
            ("qkv_proj", "k_proj", "k"),  # QKV 投影中的 K 部分
            ("qkv_proj", "v_proj", "v"),  # QKV 投影中的 V 部分
            ("gate_up_proj", "gate_proj", 0),  # gate_up 投影中的 gate 部分
            ("gate_up_proj", "up_proj", 1),  # gate_up 投影中的 up 部分
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 创建专家参数映射
            ckpt_gate_proj_name="gate_proj",  # 检查点中 gate 投影的名称
            ckpt_down_proj_name="down_proj",  # 检查点中 down 投影的名称
            ckpt_up_proj_name="up_proj",  # 检查点中 up 投影的名称
            num_experts=self.config.num_experts,  # 专家数量
        )

        params_dict = dict(self.named_parameters())  # 将模型参数转为字典

        for name, loaded_weight in weights:  # 遍历所有权重
            if name.startswith(nextn_prefix):  # 如果权重属于 NextN 层
                subname = name[len(nextn_prefix) :]  # 去掉前缀获取子名称
                if any(subname.startswith(s) for s in spec_weight_names):  # 如果是推测解码专有权重
                    name = f"model.{subname}"  # 映射到 model 下的对应位置
                else:  # 否则
                    name = f"model.decoder.{subname}"  # 映射到 model.decoder 下的对应位置
            elif name == "model.shared_head.norm.weight":  # 如果是共享头归一化权重
                pass  # 保持名称不变
            elif (  # 如果权重属于嵌入层、共享头或语言模型头
                "embed_tokens" in name  # 嵌入层权重
                or "shared_head.head" in name  # 共享头权重
                or "lm_head" in name  # 语言模型头权重
            ):
                continue  # 跳过，这些权重会在别处加载
            else:  # 其他权重
                continue  # 跳过

            if "rotary_emb.inv_freq" in name:  # 如果是旋转嵌入的逆频率
                continue  # 跳过，不需要加载

            if "router.gate." in name:  # 如果是路由器门控权重
                name = name.replace("router.", "")  # 去掉 "router." 前缀

            is_found = False  # 标记是否找到堆叠参数映射
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不匹配
                    continue  # 跳过
                if "mlp.experts" in name:  # 如果是 MoE 专家权重，不在堆叠映射中处理
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                if name not in params_dict:  # 如果参数不存在
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                is_found = True  # 标记为已找到
                break  # 跳出循环
            if is_found:  # 如果已通过堆叠映射加载
                continue  # 继续下一个权重

            is_expert_weight = False  # 标记是否为专家权重
            for mapping in expert_params_mapping:  # 遍历专家参数映射
                param_name, weight_name, expert_id, shard_id = mapping  # 解包映射信息
                if weight_name not in name:  # 如果权重名不匹配
                    continue  # 跳过
                is_expert_weight = True  # 标记为专家权重
                name_mapped = name.replace(weight_name, param_name)  # 替换为专家参数名
                if name_mapped not in params_dict:  # 如果映射后的参数不存在
                    continue  # 跳过
                param = params_dict[name_mapped]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(  # 加载专家权重
                    param,  # 目标参数
                    loaded_weight,  # 加载的权重
                    name_mapped,  # 映射后的名称
                    shard_id=shard_id,  # 分片 ID
                    expert_id=expert_id,  # 专家 ID
                )
                break  # 跳出循环
            if is_expert_weight:  # 如果已通过专家映射加载
                continue  # 继续下一个权重

            if name not in params_dict:  # 如果参数名不在字典中
                continue  # 跳过
            param = params_dict[name]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器，默认使用 default_weight_loader
            weight_loader(param, loaded_weight)  # 加载权重


EntryClass = [HYV3ForCausalLMNextN]  # 模型入口类列表，用于框架自动发现和注册模型
