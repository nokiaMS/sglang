# Gemma3多模态模型：实现Gemma3的多模态条件生成，包含视觉编码器投影、图像特征提取和注意力掩码处理
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team
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

# Adapted from:
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma3_mm.py
# 适配自vLLM项目的Gemma3多模态实现

import logging  # 导入日志模块 # import logging module
import re  # 导入正则表达式模块 # import regex module
from functools import lru_cache  # 导入LRU缓存装饰器 # import LRU cache decorator
from typing import Iterable, List, Optional, Set, Tuple, TypedDict  # 导入类型提示工具 # import type hints

import torch  # 导入PyTorch库 # import PyTorch
from torch import nn  # 导入神经网络模块 # import neural network module
from transformers import Gemma3Config, PreTrainedModel  # 导入Gemma3配置和预训练模型 # import Gemma3 config and pretrained model

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend  # 导入Triton注意力后端 # import Triton attention backend
from sglang.srt.layers.layernorm import Gemma3RMSNorm  # 导入Gemma3 RMS归一化层 # import Gemma3 RMS norm layer
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器 # import logits processor
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置 # import quantization config
from sglang.srt.managers.mm_utils import (  # 导入多模态工具 # import multimodal utilities
    MultiModalityDataPaddingPatternTokenPairs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类 # import schedule batch classes
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode  # 导入前向批次信息和模式 # import forward batch info and mode
from sglang.srt.model_executor.forward_context import get_attn_backend  # 导入获取注意力后端函数 # import get attention backend function
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具 # import weight loading utilities
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma3_causal import Gemma3ForCausalLM  # 导入Gemma3因果语言模型 # import Gemma3 causal LM
from sglang.srt.models.siglip import SiglipVisionModel  # 导入Siglip视觉模型 # import Siglip vision model
from sglang.srt.utils import add_prefix  # 导入前缀工具 # import prefix utility
from sglang.srt.utils.hf_transformers_utils import get_processor  # 导入处理器获取函数 # import processor getter

logger = logging.getLogger(__name__)  # 创建日志记录器 # create logger

cached_get_processor = lru_cache(get_processor)  # 带缓存的处理器获取函数 # cached processor getter


class Gemma3ImagePixelInputs(TypedDict):  # Gemma3图像像素输入类型定义 # Gemma3 image pixel inputs type definition
    pixel_values: torch.Tensor  # 像素值张量 # pixel values tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""  # 形状：`(批次大小 * 图像数, 通道数, 高度, 宽度)`


class Gemma3MultiModalProjector(nn.Module):  # Gemma3多模态投影器模块 # Gemma3 multimodal projector module
    """Projector for Gemma3 multimodal."""  # Gemma3多模态投影器 # Projector for Gemma3 multimodal

    def __init__(self, config: Gemma3Config):  # 初始化方法 # initialization method
        super().__init__()  # 调用父类初始化 # call parent class init

        self.mm_input_projection_weight = nn.Parameter(  # 多模态输入投影权重参数 # multimodal input projection weight parameter
            torch.zeros(
                config.vision_config.hidden_size, config.text_config.hidden_size
            )
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(  # 多模态软嵌入归一化层 # multimodal soft embedding norm layer
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(  # 每张图像的patch数 # patches per image
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)  # 每边token数 # tokens per side
        self.kernel_size = self.patches_per_image // self.tokens_per_side  # 池化核大小 # pooling kernel size
        self.avg_pool = nn.AvgPool2d(  # 平均池化层 # average pooling layer
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:  # 前向传播方法 # forward pass method
        batch_size, seq_length, hidden_size = vision_outputs.shape  # 解包视觉输出形状 # unpack vision output shape

        # Reshape for pooling
        # 重塑形状以进行池化
        reshaped_vision_outputs = vision_outputs.transpose(1, 2)  # 转置序列和隐藏维度 # transpose sequence and hidden dims
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(  # 重塑为2D patch网格 # reshape to 2D patch grid
            batch_size, hidden_size, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()  # 确保内存连续 # ensure contiguous memory

        # Apply pooling
        # 应用池化
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)  # 平均池化降采样 # average pooling downsampling
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)  # 展平空间维度 # flatten spatial dimensions
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)  # 转置回原始顺序 # transpose back to original order

        # Apply normalization
        # 应用归一化
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)  # RMS归一化 # RMS normalization

        # Project to text embedding space
        # 投影到文本嵌入空间
        projected_vision_outputs = torch.matmul(  # 矩阵乘法投影 # matrix multiplication projection
            normed_vision_outputs, self.mm_input_projection_weight
        )

        return projected_vision_outputs.type_as(vision_outputs)  # 返回与输入同类型的投影结果 # return projection result in input dtype


class Gemma3ForConditionalGeneration(PreTrainedModel):  # Gemma3条件生成模型类 # Gemma3 conditional generation model class
    config_class = Gemma3Config  # 配置类 # config class
    """Gemma3 multimodal model for conditional generation."""  # Gemma3多模态条件生成模型 # Gemma3 multimodal model for conditional generation

    # BitandBytes specific attributes
    # BitandBytes特定属性
    default_bitsandbytes_target_modules = [  # 默认BitandBytes目标模块 # default BitandBytes target modules
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        ".out_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes堆叠参数映射 # BitandBytes stacked params mapping
        # shard_name, weight_name, index
        # 分片名，权重名，索引
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "out_proj": ("proj", 0),
    }

    packed_modules_mapping = {  # 打包模块映射 # packed modules mapping
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    # LoRA特定属性
    supported_lora_modules = [  # 支持LoRA的模块 # LoRA supported modules
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    # Gemma does not apply LoRA to the embedding layer.
    # Gemma不在嵌入层应用LoRA
    embedding_modules = {}  # 嵌入模块映射 # embedding modules mapping
    embedding_padding_modules = []  # 嵌入填充模块 # embedding padding modules
    supports_lora = True  # 支持LoRA # supports LoRA
    # Pattern to match language model layers only (skip vision_tower and multi_modal_projector)
    # 仅匹配语言模型层的模式（跳过视觉塔和多模态投影器）
    lora_pattern = re.compile(  # LoRA匹配正则 # LoRA match regex
        r"^language_model\.model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)"
    )

    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3Config,  # Gemma3配置 # Gemma3 config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ) -> None:
        super().__init__(config=config)  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config
        self.quant_config = quant_config  # 保存量化配置 # save quantization config

        # For LoRA compatibility: expose text_config attributes at top level
        # This allows LoRA code to work without special multimodal handling
        # 为LoRA兼容性：在顶层暴露text_config属性，使LoRA代码无需特殊多模态处理
        if not hasattr(config, "num_hidden_layers"):  # 无num_hidden_layers属性 # no num_hidden_layers attr
            config.num_hidden_layers = config.text_config.num_hidden_layers  # 从text_config获取 # get from text_config
        if not hasattr(config, "hidden_size"):  # 无hidden_size属性 # no hidden_size attr
            config.hidden_size = config.text_config.hidden_size  # 从text_config获取 # get from text_config

        self.vision_tower = SiglipVisionModel(  # 创建视觉塔模型 # create vision tower model
            config=config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )

        self.multi_modal_projector = Gemma3MultiModalProjector(config)  # 创建多模态投影器 # create multimodal projector
        self.vocab_size = config.text_config.vocab_size  # 词表大小 # vocab size

        # Text model
        # 文本模型
        self.language_model = Gemma3ForCausalLM(  # 创建Gemma3因果语言模型 # create Gemma3 causal LM
            config.text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        if self.language_model.logits_processor.logit_scale:  # 如果有logit缩放 # if has logit scale
            logit_scale = getattr(config, "logit_scale", 1.0)  # 获取logit缩放值 # get logit scale value
            self.language_model.logits_processor.logit_scale *= logit_scale  # 乘以额外缩放 # multiply by extra scale
        self.post_init()  # 调用后初始化 # call post init

    def pad_input_ids(  # 填充输入ID方法 # pad input IDs method
        self, input_ids: List[int], image_inputs: MultimodalInputs
    ) -> List[int]:
        """Pad input IDs with image tokens."""  # 用图像token填充输入ID # Pad input IDs with image tokens
        # Get special token IDs
        # 获取特殊token ID
        im_start_id: int = image_inputs.im_start_id  # 图像起始token ID # image start token ID
        im_end_id: int = image_inputs.im_end_id  # 图像结束token ID # image end token ID

        media_token_pairs = [(im_start_id, im_end_id)]  # 媒体token对 # media token pairs
        pattern = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)  # 创建填充模式 # create padding pattern
        ids = pattern.pad_input_tokens(input_ids, image_inputs)  # 执行填充 # execute padding
        return ids  # 返回填充后的ID # return padded IDs

    def prepare_attn_masks(  # 准备注意力掩码方法 # prepare attention masks method
        self,
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        input_ids: torch.Tensor,  # 输入token ID张量 # input token ID tensor
        mask_dtype: torch.dtype,  # 掩码数据类型 # mask data type
    ):
        """Prepare attention masks for multimodal inputs."""  # 为多模态输入准备注意力掩码 # Prepare attention masks for multimodal inputs
        if isinstance(get_attn_backend(), TritonAttnBackend):  # 使用Triton注意力后端 # using Triton attention backend
            assert forward_batch.forward_mode == ForwardMode.EXTEND  # 仅支持EXTEND模式 # only EXTEND mode supported
            bidirectional_attn_masks_list = []  # 双向注意力掩码列表 # bidirectional attention masks list
            bidirectional_attn_mask_indptr = torch.zeros(  # 掩码索引指针 # mask index pointers
                forward_batch.batch_size + 1, dtype=torch.int32, device=input_ids.device
            )

            for i in range(forward_batch.batch_size):  # 遍历每个批次 # iterate each batch
                bidirectional_attn_mask = torch.empty(  # 创建空掩码 # create empty mask
                    forward_batch.extend_seq_lens[i],
                    forward_batch.extend_seq_lens[i]
                    + forward_batch.extend_prefix_lens[i],
                    dtype=mask_dtype,
                    device=input_ids.device,
                )
                bidirectional_attn_mask.fill_(1)  # 填充1 # fill with 1
                bidirectional_attn_mask = bidirectional_attn_mask.tril(  # 下三角掩码 # lower triangular mask
                    diagonal=forward_batch.extend_prefix_lens[i]
                )

                # Consider bidirectional attention between image tokens
                # 考虑图像token之间的双向注意力
                mm_inputs = forward_batch.mm_inputs[i]  # 获取多模态输入 # get multimodal inputs
                for mm_item in mm_inputs.mm_items:  # 遍历多模态项 # iterate multimodal items
                    if mm_item.is_image():  # 是图像项 # is image item
                        for im_begin, im_end in mm_item.offsets:  # 遍历图像偏移 # iterate image offsets
                            if (
                                im_begin >= forward_batch.extend_prefix_lens[i]
                            ):  # compatible with radix cache
                                # 兼容radix缓存
                                bidirectional_attn_mask[  # 设置图像区域的掩码为1 # set mask to 1 for image region
                                    im_begin
                                    - forward_batch.extend_prefix_lens[i] : im_end
                                    + 1
                                    - forward_batch.extend_prefix_lens[i],
                                    im_begin : im_end + 1,
                                ] = 1
                bidirectional_attn_masks_list.append(bidirectional_attn_mask.flatten())  # 展平并添加 # flatten and append
                bidirectional_attn_mask_indptr[i + 1] = (  # 更新索引指针 # update index pointer
                    bidirectional_attn_mask_indptr[i]
                    + bidirectional_attn_mask.nelement()
                )

            if bidirectional_attn_masks_list:  # 有掩码数据 # has mask data
                bidirectional_attn_masks = torch.cat(  # 拼接所有掩码 # concatenate all masks
                    bidirectional_attn_masks_list, dim=0
                )
                get_attn_backend().forward_metadata.mask_indptr = (  # 设置掩码索引指针 # set mask index pointers
                    bidirectional_attn_mask_indptr
                )
                get_attn_backend().forward_metadata.custom_mask = (  # 设置自定义掩码 # set custom mask
                    bidirectional_attn_masks
                )

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层 # get input embedding layer
        return self.language_model.get_input_embeddings()  # 委托给语言模型 # delegate to language model

    def get_attention_sliding_window_size(self):  # 获取注意力滑动窗口大小 # get attention sliding window size
        """
        This value is used to initialize attention backends in `ForwardBatch`.
        """
        # 此值用于初始化ForwardBatch中的注意力后端
        return self.language_model.get_attention_sliding_window_size()  # 委托给语言模型 # delegate to language model

    def get_image_feature(self, items: List[MultimodalDataItem]):  # 获取图像特征方法 # get image features method
        """
        Projects the last hidden state from the vision model into language model space.
        Supports both raw image pixel values and precomputed embeddings.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        # 将视觉模型的最后隐藏状态投影到语言模型空间。
        # 支持原始图像像素值和预计算的嵌入。
        # 返回：image_features（torch.Tensor）：形状为(num_images, image_length, embed_dim)的图像特征张量
        # Process images one by one to handle flatten_batch=True constraint in vision_tower
        # 逐个处理图像以处理视觉塔中flatten_batch=True的约束
        all_pixel_values = flatten_nested_list([item.feature for item in items])  # 展平所有像素值 # flatten all pixel values

        final_features_list = []  # 最终特征列表 # final features list

        for pixel_values_batch in all_pixel_values:  # 遍历每批像素值 # iterate each pixel values batch
            if (  # 检查是否为预计算嵌入 # check if precomputed embeddings
                pixel_values_batch.dim() == 3
                and pixel_values_batch.shape[-1] == self.config.text_config.hidden_size
            ):
                final_features_list.append(  # 直接添加预计算嵌入 # directly append precomputed embeddings
                    pixel_values_batch.to(self.language_model.device)
                )
                continue  # 跳过后续处理 # skip further processing

            # Normalize input shape to [batch_size, channels, height, width]
            # 规范化输入形状为[batch_size, channels, height, width]
            if pixel_values_batch.dim() == 5:  # 5维输入 # 5D input
                pixel_values_batch = pixel_values_batch.squeeze(0)  # 去除第一维 # remove first dimension
            elif pixel_values_batch.dim() == 3:  # 3维输入 # 3D input
                pixel_values_batch = pixel_values_batch.unsqueeze(0)  # 增加批次维 # add batch dimension
            elif pixel_values_batch.dim() != 4:  # 非标准维度 # non-standard dimensions
                raise ValueError(  # 抛出错误 # raise error
                    f"Unexpected pixel_values shape: {pixel_values_batch.shape}"
                )

            # Process each image in the batch through Vision Tower
            # 通过视觉塔处理批次中的每张图像
            batch_vision_outputs = []  # 批次视觉输出列表 # batch vision outputs list
            batch_size = pixel_values_batch.shape[0]  # 批次大小 # batch size

            for i in range(batch_size):  # 逐张处理图像 # process images one by one
                pixel_value = pixel_values_batch[i : i + 1]  # Keep batch dimension as 1 # 保持批次维度为1
                pixel_value = pixel_value.to(  # 移动到设备和数据类型 # move to device and dtype
                    device=self.vision_tower.device, dtype=self.language_model.dtype()
                )
                vision_output = self.vision_tower(pixel_values=pixel_value)  # 通过视觉塔 # through vision tower
                batch_vision_outputs.append(vision_output)  # 添加到列表 # append to list

            if batch_vision_outputs:  # 有视觉输出 # has vision outputs
                vision_outputs_cat = torch.cat(batch_vision_outputs, dim=0)  # 拼接视觉输出 # concatenate vision outputs

                projected_features = self.multi_modal_projector(vision_outputs_cat)  # 通过多模态投影器 # through multimodal projector
                final_features_list.append(projected_features)  # 添加投影后特征 # append projected features

        # Concatenate all features (all are now in text space)
        # 拼接所有特征（现在都在文本空间中）
        if final_features_list:  # 有特征 # has features
            return torch.cat(final_features_list, dim=0)  # 拼接返回 # concatenate and return
        else:  # 无特征 # no features
            return torch.tensor([], device=self.language_model.device)  # 返回空张量 # return empty tensor

    @torch.no_grad()  # 禁用梯度计算 # disable gradient computation
    def forward(  # 前向传播方法 # forward pass method
        self,
        input_ids: torch.LongTensor,  # 输入token ID张量 # input token ID tensor
        positions: torch.Tensor,  # 位置编码张量 # position encoding tensor
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选 # input embeddings, optional
        **kwargs: object,
    ) -> LogitsProcessor:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/Gemma3-test-224px-hf")
        >>> processor = AutoProcessor.from_pretrained("google/Gemma3-test-224px-hf")

        >>> prompt = "answer en Where is the cow standing?"
        >>> url = "https://huggingface.co/gv-hf/Gemma3-test-224px-hf/resolve/main/cow_beach_1.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "answer en Where is the cow standing?\nbeach"
        ```"""
        # 上述文档字符串描述了标签、logits_to_keep参数及使用示例

        # Important: position_ids in Gemma3 are 1-indexed
        # This really does cost me sometime
        # 重要：Gemma3中的position_ids是从1开始索引的，这确实花了一些时间才发现
        positions += 1  # 位置加1以适配1-indexed # add 1 to positions for 1-indexed

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        # 如果图像token超出词表范围，将其替换为PAD以避免索引错误
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:  # 图像token索引超出词表 # image token index out of vocab
            special_image_mask = input_ids == self.config.image_token_index  # 创建图像token掩码 # create image token mask
            llm_input_ids = input_ids.clone()  # 克隆输入ID # clone input IDs
            llm_input_ids[special_image_mask] = 0  # 将图像token替换为0 # replace image tokens with 0
        else:  # 图像token在词表范围内 # image token within vocab
            llm_input_ids = input_ids  # 直接使用原始输入 # use original input directly

        # NOTE: As described in https://huggingface.co/blog/gemma3#multimodality, in the prefill stage of Gemma-3, image tokens use bidirectional attention. Currently, only the TritonAttnBackend supports bidirectional attention; other backends have not yet implemented this. Bidirectional attention is incompatible with CUDA Graph and chunked prefill.
        # 注意：如HuggingFace博客所述，Gemma-3的预填充阶段中图像token使用双向注意力。目前仅TritonAttnBackend支持双向注意力；其他后端尚未实现。双向注意力与CUDA Graph和分块预填充不兼容。
        if (
            forward_batch.forward_mode
            == ForwardMode.EXTEND  # only Extend mode is supported for now
            # 目前仅支持Extend模式
            and forward_batch.contains_image_inputs()  # Gemma-3 only supports image as mm inputs
            # Gemma-3仅支持图像作为多模态输入
        ):
            self.prepare_attn_masks(  # 准备注意力掩码 # prepare attention masks
                forward_batch,
                llm_input_ids,
                mask_dtype=torch.bool,
            )

        hs = general_mm_embed_routine(  # 通用多模态嵌入处理 # general multimodal embedding routine
            input_ids=llm_input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )

        return hs  # 返回隐藏状态 # return hidden states

    def should_apply_lora(self, module_name: str) -> bool:  # 判断是否应应用LoRA # whether should apply LoRA
        """Skip vision tower and multi_modal_projector for LoRA."""  # 跳过视觉塔和多模态投影器的LoRA # Skip vision tower and multi_modal_projector for LoRA
        return bool(self.lora_pattern.match(module_name))  # 匹配LoRA模式 # match LoRA pattern

    def tie_weights(self, **kwargs):  # 绑定权重方法 # tie weights method
        return self.language_model.tie_weights(**kwargs)  # 委托给语言模型 # delegate to language model

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法 # load weights method
        stacked_params_mapping = [  # 堆叠参数映射 # stacked params mapping
            # (param_name, shard_name, shard_id)
            # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        """Load weights for the model."""  # 加载模型权重 # Load weights for the model
        params_dict = dict(self.named_parameters())  # 参数字典 # parameters dict
        loaded_params: Set[str] = set()  # 已加载参数集合 # loaded params set

        for name, loaded_weight in weights:  # 遍历权重 # iterate weights
            if "language_model" in name:  # 语言模型权重 # language model weights
                # Gemma3ForCausalLM.load_weights(self, [(name.replace("language_model.", ""), loaded_weight)])
                causal_loaded_params = Gemma3ForCausalLM.load_weights(  # 委托给因果LM加载 # delegate to causal LM loading
                    self, [(name, loaded_weight)]
                )
                loaded_params.update(causal_loaded_params)  # 更新已加载参数 # update loaded params
                continue  # 继续下一个权重 # continue to next weight
            else:  # 非语言模型权重（视觉塔等） # non-language model weights (vision tower etc.)
                for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠映射 # iterate stacked mapping
                    if weight_name not in name:  # 权重名不在名称中 # weight name not in name
                        continue  # 跳过 # skip
                    name = name.replace(weight_name, param_name)  # 替换权重名 # replace weight name
                    # Skip loading extra bias for GPTQ models.
                    # 跳过GPTQ模型的额外偏置加载
                    if name.endswith(".bias") and name not in params_dict:  # 额外偏置 # extra bias
                        continue  # 跳过 # skip
                    param = params_dict[name]  # 获取参数 # get parameter
                    weight_loader = param.weight_loader  # 获取权重加载器 # get weight loader
                    weight_loader(param, loaded_weight, shard_id)  # 加载权重 # load weight
                    break  # 跳出内层循环 # break inner loop
                else:  # 非堆叠参数 # non-stacked params
                    if "vision_model" in name:  # 视觉模型权重 # vision model weights
                        # adapt to VisionAttention
                        # 适配VisionAttention
                        name = name.replace(".self_attn.out_proj", ".self_attn.proj")  # 替换输出投影名 # replace output proj name
                    # Skip loading extra bias for GPTQ models
                    # 跳过GPTQ模型的额外偏置加载
                    if name.endswith(".bias") and name not in params_dict:  # 额外偏置 # extra bias
                        continue  # 跳过 # skip
                    # Remapping the name of FP8 kv-scale
                    # 重映射FP8 kv-scale的名称
                    name = maybe_remap_kv_scale_name(name, params_dict)  # 重映射名称 # remap name
                    if name is None:  # 名称无效 # name is None
                        continue  # 跳过 # skip
                    param = params_dict[name]  # 获取参数 # get parameter
                    weight_loader = getattr(  # 获取权重加载器 # get weight loader
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重 # load weight
                loaded_params.add(name)  # 添加到已加载集合 # add to loaded set
        unloaded_params = params_dict.keys() - loaded_params  # 未加载参数 # unloaded params
        if unloaded_params:  # 有未加载参数 # has unloaded params
            pass  # 跳过 # pass
            # raise RuntimeError(
            #     f"Some weights are not initialized from checkpoints: {unloaded_params}")
            # 部分权重未从检查点初始化的错误（已注释）
        return loaded_params  # 返回已加载参数集合 # return loaded params set

    def get_embed_and_head(self):  # 获取嵌入和LM头权重 # get embedding and LM head weights
        # For EAGLE3, we delegate to the language model which should have this method
        # If the language model doesn't have lm_head (like EAGLE3), we return None for head
        # 对于EAGLE3，委托给应有此方法的语言模型；如果语言模型没有lm_head（如EAGLE3），返回None作为head
        embed = self.language_model.get_embed()  # 获取嵌入 # get embedding
        if hasattr(self.language_model, "get_embed_and_head"):  # 语言模型有此方法 # language model has this method
            return self.language_model.get_embed_and_head()  # 委托 # delegate
        elif hasattr(self.language_model, "lm_head"):  # 语言模型有lm_head # language model has lm_head
            return embed, self.language_model.lm_head.weight  # 返回嵌入和头权重 # return embed and head weights
        else:  # 无lm_head # no lm_head
            # For EAGLE3, head might not be needed
            # 对于EAGLE3，head可能不需要
            return embed, None  # 返回嵌入和None # return embed and None

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):  # 设置EAGLE3捕获层 # set EAGLE3 capture layers
        if hasattr(self.language_model, "set_eagle3_layers_to_capture"):  # 语言模型有此方法 # language model has this method
            self.language_model.set_eagle3_layers_to_capture(layer_ids)  # 委托 # delegate


EntryClass = Gemma3ForConditionalGeneration  # 模型入口类 # model entry class
