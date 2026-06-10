# Gemma3n多模态模型实现文件
# 本文件实现了Gemma3n模型的多模态条件生成功能，包括图像和音频的编码与嵌入，
# 以及多模态嵌入器（MultimodalEmbedder）将视觉/音频特征投影到语言模型空间。

import logging  # 导入日志模块
import re  # 导入正则表达式模块
from functools import lru_cache  # 导入LRU缓存装饰器
from typing import Iterable, List, Optional, Set, Tuple, TypedDict, Union  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import (  # 从transformers导入配置类
    Gemma3nAudioConfig,  # Gemma3n音频配置
    Gemma3nConfig,  # Gemma3n主配置
    Gemma3nTextConfig,  # Gemma3n文本配置
    Gemma3nVisionConfig,  # Gemma3n视觉配置
    PreTrainedModel,  # 预训练模型基类
)
from transformers.models.auto.modeling_auto import AutoModel  # 导入自动模型类

from sglang.srt.layers.linear import ReplicatedLinear  # 导入复制线性层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding  # 导入词表并行嵌入层
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
    flatten_nested_list,  # 展平嵌套列表
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    maybe_remap_kv_scale_name,  # 可能重映射KV缩放名称
)
from sglang.srt.models.gemma3n_audio import Gemma3nAudioEncoder  # 导入Gemma3n音频编码器
from sglang.srt.models.gemma3n_causal import Gemma3nRMSNorm, Gemma3nTextModel  # 导入Gemma3n RMSNorm和文本模型
from sglang.srt.utils import add_prefix  # 导入前缀添加工具
from sglang.srt.utils.hf_transformers_utils import get_processor  # 导入处理器获取工具

logger = logging.getLogger(__name__)  # 创建日志记录器

cached_get_processor = lru_cache(get_processor)  # 带LRU缓存的处理器获取函数


class Gemma3nImagePixelInputs(TypedDict):  # Gemma3n图像像素输入类型定义
    pixel_values: torch.Tensor  # 像素值张量
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""  # 形状: `(批次大小 * 图像数量, 通道数, 高度, 宽度)`


class Gemma3nAudioInputs(TypedDict):  # Gemma3n音频输入类型定义
    input_features: torch.Tensor  # 输入特征张量
    """Shape: `(batch_size * num_audio, seq_length, num_features)`"""  # 形状: `(批次大小 * 音频数量, 序列长度, 特征数)`
    input_features_mask: torch.Tensor  # 输入特征掩码张量
    """Shape: `(batch_size * num_audio, seq_length)`"""  # 形状: `(批次大小 * 音频数量, 序列长度)`


class Gemma3nMultimodalEmbedder(nn.Module):  # Gemma3n多模态嵌入器类
    """Embeds token ids or soft tokens for multimodal content into language model space."""  # 将多模态内容的token ID或软token嵌入到语言模型空间中

    def __init__(  # 初始化方法
        self,
        multimodal_config: Union[Gemma3nAudioConfig, Gemma3nVisionConfig],  # 多模态配置（音频或视觉）
        text_config: Gemma3nTextConfig,  # 文本配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化

        self.multimodal_hidden_size = multimodal_config.hidden_size  # 多模态隐藏层大小
        self.eps = multimodal_config.rms_norm_eps  # RMSNorm epsilon值
        self.vocab_offset = multimodal_config.vocab_offset  # 词表偏移量
        self.vocab_size = multimodal_config.vocab_size  # 词表大小
        self.text_hidden_size = text_config.hidden_size  # 文本隐藏层大小

        self.embedding = VocabParallelEmbedding(  # 词表并行嵌入层
            self.vocab_size,  # 词表大小
            self.multimodal_hidden_size,  # 嵌入维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("embedding", prefix),  # 添加前缀
        )

        self.hard_embedding_norm = Gemma3nRMSNorm(  # 硬嵌入归一化层
            self.multimodal_hidden_size,  # 归一化维度
            eps=self.eps,  # epsilon值
        )

        self.soft_embedding_norm = Gemma3nRMSNorm(  # 软嵌入归一化层
            self.multimodal_hidden_size,  # 归一化维度
            eps=self.eps,  # epsilon值
        )

        self.embedding_projection = ReplicatedLinear(  # 嵌入投影线性层
            self.multimodal_hidden_size,  # 输入维度
            self.text_hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("embedding_projection", prefix),  # 添加前缀
        )

        self.embedding_post_projection_norm = Gemma3nRMSNorm(  # 投影后归一化层
            self.text_hidden_size,  # 归一化维度
            eps=self.eps,  # epsilon值
            with_scale=False,  # 不使用缩放
        )

    def forward(  # 前向传播方法
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入token ID（可选）
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:  # 返回张量
        """Embeds token ids or soft tokens for multimodal content into language model space.
        # 将多模态内容的token ID或软token嵌入到语言模型空间中

        Args:
            input_ids: A torch.LongTensor containing the token ids to embed. Values should be in the range
                `[vocab_offset, vocab_offset + vocab_size)`.  # 包含要嵌入的token ID的长整型张量，值应在[vocab_offset, vocab_offset + vocab_size)范围内
            inputs_embeds: A torch.Tensor containing the soft tokens to embed.  # 包含要嵌入的软token的张量

        Returns:
            A torch.Tensor of embeddings with  shape `[batch_size, seq_len, self.config.text_config.hidden_size]`.  # 形状为[批次大小, 序列长度, 隐藏层大小]的嵌入张量
        """
        if (input_ids is None) ^ (inputs_embeds is not None):  # 如果input_ids和inputs_embeds恰好有一个为None
            raise ValueError(  # 抛出值错误
                "You must specify exactly one of input_ids or inputs_embeds"  # 必须指定input_ids或inputs_embeds中的一个
            )

        if inputs_embeds is not None:  # 如果提供了软嵌入
            emb_norm = self.soft_embedding_norm(inputs_embeds)  # 对软嵌入进行归一化
        else:
            # Handle out of vocab ids to prevent CUDA assertion failures  # 处理超出词表范围的ID，防止CUDA断言失败
            out_of_vocab_id = self.vocab_size - 1  # 超出词表的ID设为词表最大索引
            adjusted_ids = input_ids - self.vocab_offset  # 调整ID偏移
            adjusted_ids = torch.where(adjusted_ids < 0, out_of_vocab_id, adjusted_ids)  # 将负值ID替换为超出词表ID
            adjusted_ids = torch.where(  # 将超出词表大小的ID替换
                adjusted_ids >= self.vocab_size, out_of_vocab_id, adjusted_ids  # 超出词表范围的ID替换为超出词表ID
            )
            hard_emb = self.embedding(adjusted_ids)  # 查表获取硬嵌入
            emb_norm = self.hard_embedding_norm(hard_emb)  # 对硬嵌入进行归一化

        emb_norm_proj, _ = self.embedding_projection(emb_norm)  # 投影到文本隐藏空间
        return self.embedding_post_projection_norm(emb_norm_proj)  # 投影后归一化并返回


class Gemma3nForConditionalGeneration(PreTrainedModel):  # Gemma3n条件生成模型类
    config_class = Gemma3nConfig  # 配置类
    """Gemma3n multimodal model for conditional generation."""  # Gemma3n多模态条件生成模型

    # BitandBytes specific attributes  # BitandBytes特定属性
    default_bitsandbytes_target_modules = [  # 默认BitandBytes目标模块列表
        ".gate_proj.",  # 门投影
        ".down_proj.",  # 下投影
        ".up_proj.",  # 上投影
        ".q_proj.",  # Q投影
        ".k_proj.",  # K投影
        ".v_proj.",  # V投影
        ".o_proj.",  # O投影
        ".out_proj.",  # 输出投影
    ]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes堆叠参数映射
        "q_proj": ("qkv_proj", 0),  # Q投影映射到QKV投影的第0个分片
        "k_proj": ("qkv_proj", 1),  # K投影映射到QKV投影的第1个分片
        "v_proj": ("qkv_proj", 2),  # V投影映射到QKV投影的第2个分片
        "gate_proj": ("gate_up_proj", 0),  # 门投影映射到gate_up投影的第0个分片
        "up_proj": ("gate_up_proj", 1),  # 上投影映射到gate_up投影的第1个分片
        "out_proj": ("proj", 0),  # 输出投影映射到proj的第0个分片
    }

    packed_modules_mapping = {  # 打包模块映射
        "qkv_proj": [  # QKV投影打包
            "q_proj",  # Q投影
            "k_proj",  # K投影
            "v_proj",  # V投影
        ],
        "gate_up_proj": [  # gate_up投影打包
            "gate_proj",  # 门投影
            "up_proj",  # 上投影
        ],
    }

    # LoRA specific attributes  # LoRA特定属性
    supported_lora_modules = [  # 支持LoRA的模块列表
        "qkv_proj",  # QKV投影
        "o_proj",  # O投影
        "gate_up_proj",  # gate_up投影
        "down_proj",  # 下投影
    ]
    # Gemma does not apply LoRA to the embedding layer  # Gemma不在嵌入层上应用LoRA
    embedding_modules = {}  # 嵌入模块为空
    embedding_padding_modules = []  # 嵌入填充模块为空
    supports_lora = True  # 支持LoRA

    def __init__(  # 初始化方法
        self,
        config: Gemma3nConfig,  # Gemma3n配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__(config=config)  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        prefix = add_prefix("model", prefix)  # 添加模型前缀

        # Vision components  # 视觉组件
        # TODO: Use sglang's vision model  # TODO: 使用sglang的视觉模型
        self.vision_tower = AutoModel.from_config(config=config.vision_config)  # 从配置创建视觉塔

        self.embed_vision = Gemma3nMultimodalEmbedder(  # 视觉嵌入器
            config.vision_config,  # 视觉配置
            config.text_config,  # 文本配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("embed_vision", prefix),  # 添加前缀
        )

        # Audio components  # 音频组件
        self.embed_audio = Gemma3nMultimodalEmbedder(  # 音频嵌入器
            config.audio_config,  # 音频配置
            config.text_config,  # 文本配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("embed_audio", prefix),  # 添加前缀
        )

        self.audio_tower = Gemma3nAudioEncoder(  # 音频编码塔
            config.audio_config,  # 音频配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("audio_tower", prefix),  # 添加前缀
        )

        self.vocab_size = config.text_config.vocab_size  # 词表大小
        self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input  # 每层输入词表大小

        # Text model  # 文本模型
        self.language_model = Gemma3nTextModel(  # 语言模型
            config.text_config,  # 文本配置
            quant_config,  # 量化配置
            prefix=add_prefix("language_model", prefix),  # 添加前缀
        )

        # Create logits processor for the multimodal model  # 为多模态模型创建logits处理器
        self.logits_processor = LogitsProcessor(config.text_config)  # logits处理器

        self.post_init()  # 后初始化

    def pad_input_ids(  # 填充输入ID方法
        self,
        input_ids: List[int],  # 输入ID列表
        mm_inputs: MultimodalInputs,  # 多模态输入
    ) -> List[int]:  # 返回填充后的ID列表
        """Pad input IDs with image and audio tokens."""  # 用图像和音频token填充输入ID
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建多模态token填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 返回填充后的token

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层
        return self.language_model.get_input_embeddings()  # 返回语言模型的输入嵌入

    def get_attention_sliding_window_size(self):  # 获取注意力滑动窗口大小
        return self.config.text_config.sliding_window - 1  # 返回滑动窗口大小减1

    def get_image_feature(self, items: List[MultimodalDataItem]):  # 获取图像特征
        """
        Projects the last hidden state from the vision model into language model space.
        # 将视觉模型的最后隐藏状态投影到语言模型空间

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).  # 图像特征张量，形状为(图像数量, 图像长度, 嵌入维度)
        """
        # Process images one by one to handle flatten_batch=True constraint in vision_tower  # 逐个处理图像以处理vision_tower中的flatten_batch=True约束
        all_pixel_values = flatten_nested_list([item.feature for item in items])  # 展平所有像素值
        vision_outputs_list = []  # 视觉输出列表

        for pixel_values_batch in all_pixel_values:  # 遍历每个像素值批次
            # Normalize input shape to [batch_size, channels, height, width]  # 将输入形状归一化为[批次大小, 通道数, 高度, 宽度]
            if pixel_values_batch.dim() == 5:  # 如果是5维张量
                pixel_values_batch = pixel_values_batch.squeeze(0)  # 去除第0维
            elif pixel_values_batch.dim() == 3:  # 如果是3维张量
                pixel_values_batch = pixel_values_batch.unsqueeze(0)  # 在第0维添加维度
            elif pixel_values_batch.dim() != 4:  # 如果不是4维张量
                raise ValueError(  # 抛出值错误
                    f"Unexpected pixel_values shape: {pixel_values_batch.shape}"  # 意外的像素值形状
                )

            # Process each image in the batch  # 处理批次中的每张图像
            batch_size = pixel_values_batch.shape[0]  # 获取批次大小
            for i in range(batch_size):  # 遍历每张图像
                pixel_value = pixel_values_batch[i : i + 1]  # Keep batch dimension as 1  # 保持批次维度为1
                pixel_value = pixel_value.to(  # 转移到指定设备和数据类型
                    device=self.vision_tower.device, dtype=self.language_model.dtype()  # 视觉塔设备，语言模型数据类型
                )
                vision_outputs = self.vision_tower(  # 通过视觉塔获取输出
                    pixel_values=pixel_value, do_pooling=False, return_dict=True  # 不做池化，返回字典
                ).last_hidden_state  # 获取最后隐藏状态
                vision_outputs_list.append(vision_outputs)  # 添加到输出列表

        # Concatenate all vision outputs  # 拼接所有视觉输出
        vision_outputs = torch.cat(vision_outputs_list, dim=0)  # 在第0维拼接

        # Convert from (batch, channels, height, width) to (batch, height * width, channels)  # 从(批次, 通道, 高, 宽)转换为(批次, 高*宽, 通道)
        vision_outputs = vision_outputs.reshape(  # 重塑形状
            vision_outputs.shape[0],  # 批次大小
            self.config.vision_config.hidden_size,  # 视觉隐藏层大小
            self.config.vision_soft_tokens_per_image,  # 每张图像的软token数
        ).permute(0, 2, 1)  # 置换维度

        # Normalize and embed the soft tokens into language model space  # 归一化并将软token嵌入到语言模型空间
        vision_outputs *= self.config.vision_config.hidden_size**0.5  # 乘以隐藏层大小的平方根进行缩放
        return self.embed_vision(inputs_embeds=vision_outputs)  # 通过视觉嵌入器返回

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 获取音频特征
        """
        Projects the last hidden state from the audio encoder into language model space.
        # 将音频编码器的最后隐藏状态投影到语言模型空间

        Args:
            items: List of multimodal data items containing audio data.  # 包含音频数据的多模态数据项列表

        Returns:
            audio_features (`torch.Tensor`): Audio feature tensor of shape `(num_audios, audio_length, embed_dim)`).  # 音频特征张量，形状为(音频数量, 音频长度, 嵌入维度)
        """
        # Extract audio features and masks from items  # 从数据项中提取音频特征和掩码
        all_input_features = flatten_nested_list([item.feature for item in items])  # 展平所有输入特征
        all_input_features_mask = flatten_nested_list(  # 展平所有输入特征掩码
            [~item.input_features_mask for item in items]  # 取反掩码
        )  # Note(Xinyuan): reverse the mask according to the HF implementation  # 注意(Xinyuan): 根据HF实现反转掩码

        # Process audio features one by one  # 逐个处理音频特征
        audio_features_list = []  # 音频特征列表

        for input_features, input_features_mask in zip(  # 遍历输入特征和掩码
            all_input_features, all_input_features_mask  # 配对的输入特征和掩码
        ):
            # Ensure proper tensor format  # 确保正确的张量格式
            if input_features.dim() == 2:  # 如果是2维张量
                input_features = input_features.unsqueeze(0)  # 在第0维添加维度
            if input_features_mask.dim() == 1:  # 如果掩码是1维张量
                input_features_mask = input_features_mask.unsqueeze(0)  # 在第0维添加维度

            # Move to device and dtype  # 转移到指定设备和数据类型
            input_features = input_features.to(  # 转移输入特征
                device=next(self.audio_tower.parameters()).device,  # 音频塔设备
                dtype=self.language_model.dtype(),  # 语言模型数据类型
            )
            input_features_mask = input_features_mask.to(device=input_features.device)  # 转移掩码到相同设备

            # Process through audio tower  # 通过音频塔处理
            audio_outputs, audio_mask = self.audio_tower(  # 获取音频输出和掩码
                input_features, input_features_mask  # 输入特征和掩码
            )

            # Embed the audio outputs  # 嵌入音频输出
            audio_embeds = self.embed_audio(inputs_embeds=audio_outputs)  # 通过音频嵌入器嵌入
            audio_features_list.append(audio_embeds)  # 添加到特征列表

        # Concatenate all audio features  # 拼接所有音频特征
        if audio_features_list:  # 如果列表非空
            audio_features = torch.cat(audio_features_list, dim=0)  # 在第0维拼接

            # The Gemma3nProcessor expects all audio will be 30s in length and inserts 188 audio soft tokens into the
            # text to account for this. However, the audio preprocessing and encoder do not gurarantee they will
            # produce 188 soft tokens; they will produce at most that many tokens, but they may produce fewer tokens
            # depending on the length of the longest audio input in the batch. When we encounter this situation, we pad
            # the audio feature out to 188 soft tokens with the emebedding of the last token in the embed_audio vocab.
            # Gemma3nProcessor期望所有音频长度为30秒，并在文本中插入188个音频软token来适应。
            # 然而，音频预处理和编码器不能保证产生188个软token；最多产生那么多，
            # 但根据批次中最长音频输入的长度可能产生更少。遇到这种情况时，
            # 我们用embed_audio词表中最后一个token的嵌入将音频特征填充到188个软token。
            audio_padding_toks = torch.tensor(  # 音频填充token
                [[self.vocab_size - 1]], dtype=torch.long, device=audio_features.device  # 词表最后一个索引
            )
            audio_padding_embs = self.embed_audio(input_ids=audio_padding_toks)  # 获取填充token的嵌入
            audio_features = torch.where(  # 根据掩码选择填充嵌入或原始特征
                audio_mask.unsqueeze(-1), audio_padding_embs, audio_features  # 掩码为True的位置用填充嵌入
            )

            audio_batch_size, audio_seq_len, audio_embed_dim = audio_features.shape  # 获取音频特征形状
            extra_padding_tokens = (  # 计算额外填充token数
                self.config.audio_soft_tokens_per_image - audio_seq_len  # 目标token数减去当前序列长度
            )
            extra_padding_features = audio_padding_embs.expand(  # 扩展填充嵌入
                audio_batch_size, extra_padding_tokens, audio_embed_dim  # 扩展到(批次, 额外token数, 嵌入维度)
            )

            audio_features = torch.cat((audio_features, extra_padding_features), dim=1)  # 拼接额外填充
            return audio_features  # 返回音频特征
        else:  # 如果列表为空
            return torch.empty(  # 返回空张量
                0,  # 第0维大小
                0,  # 第1维大小
                self.language_model.config.hidden_size,  # 隐藏层大小
                device=next(self.parameters()).device,  # 模型设备
                dtype=self.language_model.dtype(),  # 语言模型数据类型
            )

    def get_per_layer_inputs(  # 获取每层输入方法
        self, input_ids: torch.LongTensor  # 输入token ID
    ) -> Optional[torch.Tensor]:  # 返回可选张量
        return self.language_model.get_per_layer_inputs(input_ids)  # 返回语言模型的每层输入

    def project_per_layer_inputs(  # 投影每层输入方法
        self,
        inputs_embeds: torch.Tensor,  # 输入嵌入
        per_layer_inputs: Optional[torch.Tensor] = None,  # 每层输入（可选）
    ) -> torch.Tensor:  # 返回张量
        return self.language_model.project_per_layer_inputs(  # 返回语言模型的每层输入投影
            inputs_embeds, per_layer_inputs  # 输入嵌入和每层输入
        )

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播方法
        self,
        input_ids: torch.LongTensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
        **kwargs: object,  # 其他关键字参数
    ) -> LogitsProcessor:  # 返回logits处理器
        """Forward pass for multimodal Gemma3n."""  # Gemma3n多模态前向传播
        if (input_ids is None) ^ (input_embeds is not None):  # 如果恰好有一个为None
            raise ValueError(  # 抛出值错误
                "You must specify exactly one of input_ids or inputs_embeds"  # 必须指定input_ids或inputs_embeds中的一个
            )

        positions += 1  # 位置编码加1
        if input_ids is not None:  # 如果提供了输入ID
            # Prepare per-layer inputs from inputs_ids  # 从输入ID准备每层输入
            per_layer_inputs_mask = torch.logical_and(  # 创建每层输入掩码
                input_ids >= 0, input_ids < self.vocab_size_per_layer_input  # ID在有效范围内
            )
            per_layer_inputs_tokens = torch.where(  # 获取有效的每层输入token
                per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)  # 无效位置填0
            )
            per_layer_inputs = self.language_model.get_per_layer_inputs(  # 获取每层输入
                per_layer_inputs_tokens  # 每层输入token
            )

        # Use general_mm_embed_routine for handling multimodal data  # 使用通用多模态嵌入例程处理多模态数据
        # This will automatically handle text, image, and audio embeddings  # 这将自动处理文本、图像和音频嵌入
        hidden_states = general_mm_embed_routine(  # 调用通用多模态嵌入例程
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.language_model,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数映射
                Modality.IMAGE: self.get_image_feature,  # 图像模态到图像特征获取函数
                Modality.AUDIO: self.get_audio_feature,  # 音频模态到音频特征获取函数
            },
            positions=positions,  # 位置编码
            per_layer_inputs=per_layer_inputs,  # 每层输入
        )

        # Process hidden states through logits processor  # 通过logits处理器处理隐藏状态
        return self.logits_processor(  # 返回logits处理结果
            input_ids, hidden_states, self.language_model.embed_tokens, forward_batch  # 输入ID，隐藏状态，嵌入层，前向批次
        )

    def tie_weights(self):  # 绑定权重方法
        return self.language_model.tie_weights()  # 返回语言模型的权重绑定

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),  # Q投影映射
            (".qkv_proj", ".k_proj", "k"),  # K投影映射
            (".qkv_proj", ".v_proj", "v"),  # V投影映射
            (".gate_up_proj", ".up_proj", 1),  # 上投影映射
            (".gate_up_proj", ".gate_proj", 0),  # 门投影映射
        ]
        """Load weights for the model."""  # 为模型加载权重
        params_dict = dict(self.named_parameters())  # 获取参数字典
        loaded_params: Set[str] = set()  # 已加载参数集合

        for name, loaded_weight in weights:  # 遍历权重
            name = re.sub(r"^model\.", "", name)  # 移除model.前缀
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果分片名不在名称中
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换分片名为参数名
                # Skip loading extra bias for GPTQ models  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重到分片
                break  # 跳出内层循环
            else:  # 如果没有匹配的堆叠参数映射
                if "vision_model" in name:  # 如果名称包含vision_model
                    # adapt to VisionAttention  # 适配VisionAttention
                    name = name.replace(".self_attn.out_proj", ".self_attn.proj")  # 替换输出投影名称
                # Skip loading extra bias for GPTQ models  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过
                # Remapping the name of FP8 kv-scale  # 重映射FP8 KV缩放的名称
                name = maybe_remap_kv_scale_name(name, params_dict)  # 可能重映射名称
                if name is None:  # 如果名称为None
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重
            loaded_params.add(name)  # 添加到已加载参数集合
        return loaded_params  # 返回已加载参数集合

    lora_pattern = re.compile(  # LoRA匹配模式
        r"^language_model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)"  # 匹配语言模型层中的LoRA目标模块
    )

    def should_apply_lora(self, module_name: str) -> bool:  # 判断是否应应用LoRA
        return bool(self.lora_pattern.match(module_name))  # 返回是否匹配LoRA模式

    def get_hidden_dim(self, module_name, layer_idx):  # 获取隐藏维度
        # return input_dim, output_dim  # 返回输入维度和输出维度
        if module_name == "qkv_proj":  # 如果是QKV投影
            return (  # 返回
                self.config.hidden_size,  # 输入维度
                self.config.head_dim  # 头维度乘以
                * (
                    self.config.num_attention_heads  # 注意力头数加
                    + self.config.num_key_value_heads * 2  # KV头数的两倍
                ),
            )
        elif module_name == "o_proj":  # 如果是O投影
            return (  # 返回
                self.config.head_dim * self.config.num_attention_heads,  # 输入维度
                self.config.hidden_size,  # 输出维度
            )
        elif module_name == "gate_up_proj":  # 如果是gate_up投影
            assert len(set(self.config.intermediate_size)) == 1, (  # 断言所有层中间大小相同
                "Currently SGLang requires uniform intermediate size for all layers. "  # 当前SGLang要求所有层的中间大小一致
                "Please file an issue if you need support for non-uniform intermediate sizes."  # 如果需要非均匀中间大小支持请提交issue
            )
            return self.config.hidden_size, self.config.intermediate_size[0] * 2  # 返回输入和输出维度
        elif module_name == "down_proj":  # 如果是下投影
            assert len(set(self.config.intermediate_size)) == 1, (  # 断言所有层中间大小相同
                "Currently SGLang requires uniform intermediate size for all layers. "  # 当前SGLang要求所有层的中间大小一致
                "Please file an issue if you need support for non-uniform intermediate sizes."  # 如果需要非均匀中间大小支持请提交issue
            )
            return self.config.intermediate_size[0], self.config.hidden_size  # 返回输入和输出维度
        else:  # 其他模块
            raise NotImplementedError()  # 抛出未实现错误


EntryClass = Gemma3nForConditionalGeneration  # 入口类为Gemma3nForConditionalGeneration
