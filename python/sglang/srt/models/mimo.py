# MiMo模型：基于Qwen2架构的因果语言模型实现
# 本文件实现了MiMo（小米MoE）模型，继承自Qwen2的解码器层和模型结构
# 支持BitsAndBytes量化、词嵌入绑定以及KV缓存缩放等功能

# 从qwen2.py改编而来 # Adapted from qwen2.py

from typing import Iterable, Optional, Tuple  # 导入类型提示 # 导入类型提示工具

import torch  # 导入PyTorch库 # 导入PyTorch深度学习框架
from torch import nn  # 导入神经网络模块 # 导入PyTorch神经网络模块

from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器 # 导入logits后处理器
from sglang.srt.layers.pooler import Pooler, PoolingType  # 导入池化层 # 导入池化层和池化类型
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置 # 导入量化基础配置
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行词表头 # 导入词表并行嵌入的语言模型头
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # 导入前向传播批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器 # 导入默认权重加载工具
from sglang.srt.models.qwen2 import Qwen2DecoderLayer, Qwen2Model  # 导入Qwen2模型组件 # 导入Qwen2解码器层和模型
from sglang.srt.utils import add_prefix  # 导入前缀工具 # 导入前缀添加工具

MiMoConfig = None  # MiMo配置占位符 # MiMo配置类占位符，将在运行时由外部设置


class MiMoModel(Qwen2Model):  # MiMo模型类，继承自Qwen2Model # MiMo模型，基于Qwen2模型实现
    def __init__(  # 初始化方法 # 初始化函数
        self,
        config: MiMoConfig,  # 模型配置 # MiMo配置对象
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 可选的量化配置
        prefix: str = "",  # 参数前缀 # 参数名称前缀
    ) -> None:
        super().__init__(  # 调用父类初始化 # 调用Qwen2Model的初始化方法
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen2DecoderLayer,  # 使用Qwen2解码器层 # 指定解码器层类型为Qwen2DecoderLayer
        )


class MiMoForCausalLM(nn.Module):  # MiMo因果语言模型类 # MiMo因果语言模型，用于文本生成
    # BitandBytes specific attributes  # BitandBytes特定属性 # BitsAndBytes量化专用属性
    default_bitsandbytes_target_modules = [  # 默认BitsAndBytes目标模块 # 默认需要量化的模块列表
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {  # BitsAndBytes堆叠参数映射 # BitsAndBytes堆叠参数名称映射
        # shard_name, weight_name, index  # 分片名, 权重名, 索引 # 分片名称、权重名称和索引的映射
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(  # 初始化方法 # 初始化函数
        self,
        config: MiMoConfig,  # 模型配置 # MiMo配置对象
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 可选的量化配置
        prefix: str = "",  # 参数前缀 # 参数名称前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用nn.Module的初始化
        self.config = config  # 保存配置 # 存储模型配置
        self.quant_config = quant_config  # 保存量化配置 # 存储量化配置
        self.model = MiMoModel(  # 创建MiMo模型实例 # 创建MiMo模型
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        if config.tie_word_embeddings:  # 如果绑定词嵌入 # 判断是否绑定输入输出词嵌入
            self.lm_head = self.model.embed_tokens  # 使用相同的嵌入层 # 语言模型头与嵌入层共享
        else:
            self.lm_head = ParallelLMHead(  # 创建独立的语言模型头 # 创建并行的语言模型头
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)  # 创建logits处理器 # 创建logits后处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 创建池化层 # 创建池化层，使用最后一 token并归一化

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入 # 获取输入token的嵌入表示
        return self.model.get_input_embeddings(input_ids)  # 返回模型输入嵌入 # 从模型中获取嵌入

    @torch.no_grad()  # 禁用梯度计算 # 装饰器：禁用梯度计算以节省内存
    def forward(  # 前向传播方法 # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入ID # 输入token ID张量
        positions: torch.Tensor,  # 位置ID # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次 # 前向传播批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入 # 可选的输入嵌入
        get_embedding: bool = False,  # 是否获取嵌入 # 是否返回嵌入而非logits
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 获取隐藏状态 # 通过模型获取隐藏状态
        if not get_embedding:  # 如果不需要嵌入 # 判断是否需要嵌入表示
            return self.logits_processor(  # 返回logits # 通过logits处理器计算并返回logits
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return self.pooler(hidden_states, forward_batch)  # 返回池化后的嵌入 # 返回池化后的嵌入表示

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
            if (  # 跳过不需要的权重 # 跳过旋转嵌入频率、投影器和MTP层的权重
                "rotary_emb.inv_freq" in name
                or "projector" in name
                or "mtp_layers" in name
            ):
                continue  # 跳过 # 跳过当前权重
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的旋转嵌入 # 跳过旋转嵌入的缓存值
                # Models trained using ColossalAI may include these tensors in  # 使用ColossalAI训练的模型可能包含这些张量
                # the checkpoint. Skip them.  # 在检查点中。跳过它们。 # 跳过ColossalAI训练模型中的缓存张量
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 跳过绑定的lm_head权重 # 如果词嵌入绑定则跳过lm_head权重
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:  # 跳过视觉塔权重 # 跳过不在参数字典中的视觉塔权重
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射 # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中 # 检查权重名是否匹配
                    continue
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
                param = params_dict[name]  # 获取参数 # 从字典中获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器 # 获取权重加载器或使用默认
                weight_loader(param, loaded_weight)  # 加载权重 # 使用权重加载器加载权重

    def get_embed_and_head(self):  # 获取嵌入层和语言模型头 # 获取词嵌入和语言模型头的权重
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入和lm_head权重 # 返回嵌入权重和语言模型头权重

    def set_embed_and_head(self, embed, head):  # 设置嵌入层和语言模型头 # 设置词嵌入和语言模型头的权重
        del self.model.embed_tokens.weight  # 删除旧的嵌入权重 # 删除旧的嵌入权重
        del self.lm_head.weight  # 删除旧的lm_head权重 # 删除旧的语言模型头权重
        self.model.embed_tokens.weight = embed  # 设置新的嵌入权重 # 设置新的嵌入权重
        self.lm_head.weight = head  # 设置新的lm_head权重 # 设置新的语言模型头权重
        torch.cuda.empty_cache()  # 清空GPU缓存 # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA # 同步CUDA操作

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:  # 加载KV缓存缩放因子 # 加载KV缓存的量化缩放参数
        self.model.load_kv_cache_scales(quantization_param_path)  # 委托给模型 # 委托给模型的KV缓存缩放加载方法


EntryClass = MiMoForCausalLM  # 入口类 # 模型注册入口类
