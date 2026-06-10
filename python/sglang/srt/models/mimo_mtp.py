# MiMo多Token预测(MTP)模型：基于Qwen2解码器层的推测解码模块
# 本文件实现了MiMo的多Token预测器，用于推测解码以加速推理
# 包含MTP层和完整的MTP模型，支持权重加载和词嵌入绑定

# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识 # Apache 2.0许可证
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # SPDX版权声明 # vLLM项目贡献者版权
# Adapted from https://github.com/vllm-project/vllm/pull/17433/files  and deepseek_nextn.py  # 改编自vLLM和deepseek_nextn # 从vLLM和DeepSeek NextN改编

from typing import Iterable, Optional, Tuple  # 导入类型提示 # 导入类型提示工具

import torch  # 导入PyTorch库 # 导入PyTorch深度学习框架
from torch import nn  # 导入神经网络模块 # 导入PyTorch神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置 # 导入HuggingFace预训练配置类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入TP世界大小 # 导入获取张量并行世界大小的函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化 # 导入RMS归一化层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器 # 导入logits后处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置 # 导入量化基础配置
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入 # 导入词表并行嵌入组件
    ParallelLMHead,  # 并行语言模型头 # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入 # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # 导入前向传播批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器 # 导入默认权重加载工具
from sglang.srt.models.qwen2 import Qwen2DecoderLayer  # 导入Qwen2解码器层 # 导入Qwen2解码器层


class MiMoMultiTokenPredictorLayer(nn.Module):  # MiMo多Token预测层 # MiMo多Token预测器层，用于推测解码

    def __init__(  # 初始化方法 # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置 # 预训练配置对象
        prefix: str,  # 参数前缀 # 参数名称前缀
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 可选的量化配置
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用nn.Module的初始化

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层 # 创建词表并行嵌入层
            config.vocab_size,
            config.hidden_size,
        )
        self.token_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # Token层归一化 # token输入的RMS归一化
        self.hidden_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 隐藏状态层归一化 # 隐藏状态的RMS归一化
        self.input_proj = nn.Linear(  # 输入投影层 # 将token嵌入和隐藏状态拼接后投影
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.mtp_block = Qwen2DecoderLayer(  # MTP解码器块 # 使用Qwen2解码器层作为MTP块
            config=config, quant_config=quant_config, prefix=prefix
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终层归一化 # 最终的RMS归一化

    def forward(  # 前向传播方法 # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入ID # 输入token ID张量
        positions: torch.Tensor,  # 位置ID # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次 # 前向传播批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入 # 可选的输入嵌入
    ) -> torch.Tensor:

        if input_embeds is None:  # 如果没有提供嵌入 # 判断是否使用输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 从token ID获取嵌入 # 通过嵌入层获取隐藏状态
        else:
            hidden_states = input_embeds  # 使用提供的嵌入 # 使用传入的嵌入
        # masking inputs at position 0, as not needed by MTP  # 将位置0的输入掩码，因为MTP不需要 # 将位置0的输入置零，MTP不需要位置0
        hidden_states[positions == 0] = 0  # 掩码位置0 # 将位置0的隐藏状态置零

        hidden_states = self.input_proj(  # 输入投影 # 将归一化后的隐藏状态和token嵌入拼接后投影
            torch.cat(
                (
                    self.hidden_layernorm(forward_batch.spec_info.hidden_states),  # 归一化推测信息隐藏状态 # 归一化来自推测信息的隐藏状态
                    self.token_layernorm(hidden_states),  # 归一化token嵌入 # 归一化token嵌入
                ),
                dim=-1,  # 在最后一维拼接 # 沿最后一维拼接
            )
        )

        hidden_states, residual = self.mtp_block(  # 通过MTP块 # 通过MTP解码器块处理
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=None,  # 无残差 # 初始残差为None
        )
        hidden_states = residual + hidden_states  # 残差连接 # 残差连接
        hidden_states = self.final_layernorm(hidden_states)  # 最终归一化 # 最终层归一化
        return hidden_states  # 返回隐藏状态 # 返回处理后的隐藏状态


class MiMoMTP(nn.Module):  # MiMo多Token预测模型 # MiMo多Token预测模型，包含MTP层和语言模型头
    def __init__(  # 初始化方法 # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置 # 预训练配置对象
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 可选的量化配置
        prefix: str = "",  # 参数前缀 # 参数名称前缀
    ) -> None:
        nn.Module.__init__(self)  # 调用nn.Module初始化 # 直接调用nn.Module的初始化
        self.config = config  # 保存配置 # 存储模型配置
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小 # 获取张量并行世界大小
        self.quant_config = quant_config  # 保存量化配置 # 存储量化配置

        self.model = MiMoMultiTokenPredictorLayer(  # 创建MTP层 # 创建多Token预测器层
            config,
            prefix,
            quant_config,
        )
        self.lm_head = ParallelLMHead(  # 创建语言模型头 # 创建并行的语言模型头
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(config)  # 创建logits处理器 # 创建logits后处理器

    @torch.no_grad()  # 禁用梯度计算 # 装饰器：禁用梯度计算
    def forward(  # 前向传播方法 # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入ID # 输入token ID张量
        positions: torch.Tensor,  # 位置ID # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次 # 前向传播批次信息
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)  # 获取隐藏状态 # 通过MTP模型获取隐藏状态
        return self.logits_processor(  # 返回logits # 通过logits处理器计算并返回logits
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法 # 加载模型权重
        stacked_params_mapping = [  # 堆叠参数映射 # 需要堆叠的参数映射列表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID) # 参数名、分片名和分片ID的映射
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())  # 获取参数字典 # 将模型参数转为字典
        for name, loaded_weight in weights:  # 遍历权重 # 遍历所有权重
            if "rotary_emb.inv_freq" in name or "projector" in name:  # 跳过旋转嵌入和投影器 # 跳过旋转嵌入频率和投影器权重
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的旋转嵌入 # 跳过旋转嵌入的缓存值
                # Models trained using ColossalAI may include these tensors in  # 使用ColossalAI训练的模型可能包含这些张量
                # the checkpoint. Skip them.  # 在检查点中。跳过它们。 # 跳过ColossalAI训练模型中的缓存张量
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 跳过绑定的lm_head权重 # 如果词嵌入绑定则跳过lm_head权重
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:  # 跳过视觉塔权重 # 跳过不在参数字典中的视觉塔权重
                continue
            name = self.map_model_name_to_mtp_param_name(name)  # 映射参数名 # 将模型参数名映射为MTP参数名

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射 # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中 # 检查权重名是否匹配
                    continue
                if "mtp_block" not in name:  # 如果不是MTP块的参数 # 检查是否属于MTP块
                    break
                name = name.replace(weight_name, param_name)  # 替换权重名 # 将分片名替换为堆叠参数名
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载。 # 跳过GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置 # 如果偏置不在参数字典中则跳过
                    continue
                param = params_dict[name]  # 获取参数 # 从字典中获取参数
                weight_loader = param.weight_loader  # 获取权重加载器 # 获取参数的权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重 # 使用权重加载器加载权重
                break
            else:
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载。 # 跳过GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置 # 如果偏置不在参数字典中则跳过
                    continue
                if "mtp_block" not in name and (  # 如果不是MTP块参数且不是特定参数 # 检查是否需要加载该参数
                    "embed_tokens" not in name
                    and "lm_head" not in name
                    and "token_layernorm" not in name
                    and "hidden_layernorm" not in name
                    and "input_proj" not in name
                    and "final_layernorm" not in name
                ):
                    continue  # 跳过非MTP参数 # 跳过不属于MTP的参数
                param = params_dict[name]  # 获取参数 # 从字典中获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器 # 获取权重加载器或使用默认
                weight_loader(param, loaded_weight)  # 加载权重 # 使用权重加载器加载权重

    def map_model_name_to_mtp_param_name(self, name: str) -> str:  # 映射模型参数名为MTP参数名 # 将模型参数名映射为MTP参数名
        import re  # 导入正则表达式 # 导入正则表达式模块

        name_without_prefix = [  # 无前缀的参数名列表 # 不包含mtp_block前缀的参数名
            "token_layernorm",
            "hidden_layernorm",
            "input_proj",
            "final_layernorm",
        ]
        pattern = r"model.mtp_layers.(\d+)."  # MTP层模式 # 匹配MTP层前缀的正则表达式
        group = re.match(pattern, name)  # 匹配模式 # 尝试匹配MTP层前缀
        if group is not None:  # 如果匹配成功 # 如果匹配到MTP层前缀
            for sub_name in name_without_prefix:  # 遍历无前缀参数名 # 遍历不需要mtp_block前缀的参数
                if sub_name in name:  # 如果参数名包含子名 # 检查参数名是否包含该子名
                    name = name.replace(group.group(), "model.")  # 替换前缀 # 替换为model前缀
                    return name
            name = name.replace(group.group(), "model.mtp_block.")  # 替换为MTP块前缀 # 替换为model.mtp_block前缀
        return name  # 返回映射后的名称 # 返回映射后的参数名

    def get_embed_and_head(self):  # 获取嵌入层和语言模型头 # 获取词嵌入和语言模型头的权重
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入和lm_head权重 # 返回嵌入权重和语言模型头权重

    def set_embed_and_head(self, embed, head):  # 设置嵌入层和语言模型头 # 设置词嵌入和语言模型头的权重
        del self.model.embed_tokens.weight  # 删除旧的嵌入权重 # 删除旧的嵌入权重
        del self.lm_head.weight  # 删除旧的lm_head权重 # 删除旧的语言模型头权重
        self.model.embed_tokens.weight = embed  # 设置新的嵌入权重 # 设置新的嵌入权重
        self.lm_head.weight = head  # 设置新的lm_head权重 # 设置新的语言模型头权重
        torch.cuda.empty_cache()  # 清空GPU缓存 # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA # 同步CUDA操作


EntryClass = MiMoMTP  # 入口类 # 模型注册入口类
