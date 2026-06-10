# Copyright 2025 SGLang Team  # 版权所有2025 SGLang团队
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证2.0版授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache许可证URL
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and  # 查看许可证了解管理权限和
# limitations under the License.  # 限制的特定语言
# ==============================================================================  # 分隔线
# Gemma4多模态模型实现文件
# 本文件实现了Gemma4的多模态条件生成模型，支持图像、视频和音频输入，
# 包括双向注意力掩码、视觉/音频塔嵌入、权重重映射和流水线并行等功能。


import logging  # 导入日志模块
import re  # 导入正则表达式模块
from functools import lru_cache  # 导入LRU缓存装饰器
from typing import Iterable, List, Optional, Set, Tuple, TypedDict, Union  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import (  # 从transformers导入
    Gemma4AudioConfig,  # Gemma4音频配置
    Gemma4Config,  # Gemma4主配置
    Gemma4TextConfig,  # Gemma4文本配置
    Gemma4VisionConfig,  # Gemma4视觉配置
    PreTrainedModel,  # 预训练模型基类
)

from sglang.srt.distributed import get_pp_group  # 导入PP组获取函数
from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend  # 导入Triton注意力后端
from sglang.srt.layers.layernorm import Gemma4RMSNorm  # 导入Gemma4 RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear  # 导入复制线性层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合MoE层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.utils import PPMissingLayer  # 导入PP缺失层
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行语言模型头
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
from sglang.srt.model_executor.forward_batch_info import (  # 导入前向批次信息
    ForwardBatch,  # 前向批次
    ForwardMode,  # 前向模式
    PPProxyTensors,  # PP代理张量
)
from sglang.srt.model_executor.forward_context import get_attn_backend  # 导入注意力后端获取函数
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    maybe_remap_kv_scale_name,  # 可能重映射KV缩放名称
)
from sglang.srt.models.gemma4_audio import Gemma4AudioEncoder  # 导入Gemma4音频编码器
from sglang.srt.models.gemma4_causal import Gemma4TextModel, pp_filter_load_weight  # 导入Gemma4文本模型和PP过滤函数
from sglang.srt.models.gemma4_vision import Gemma4VisionEncoder  # 导入Gemma4视觉编码器
from sglang.srt.utils import add_prefix  # 导入前缀添加工具
from sglang.srt.utils.hf_transformers_utils import get_processor  # 导入处理器获取工具

logger = logging.getLogger(__name__)  # 创建日志记录器

cached_get_processor = lru_cache(get_processor)  # 带LRU缓存的处理器获取函数


class Gemma4ImagePixelInputs(TypedDict):  # Gemma4图像像素输入类型定义
    pixel_values: torch.Tensor  # 像素值张量
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""  # 形状: `(批次大小 * 图像数量, 通道数, 高度, 宽度)`


class Gemma4AudioInputs(TypedDict):  # Gemma4音频输入类型定义
    input_features_padded: torch.Tensor  # 填充后的输入特征张量
    """Shape: `(batch_size * num_audio, seq_length, num_features)`"""  # 形状: `(批次大小 * 音频数量, 序列长度, 特征数)`
    input_features_mask: torch.Tensor  # 输入特征掩码张量
    """Shape: `(batch_size * num_audio, seq_length)`"""  # 形状: `(批次大小 * 音频数量, 序列长度)`


class Gemma4MultimodalEmbedder(nn.Module):  # Gemma4多模态嵌入器类
    """Projects vision/audio soft tokens into LM embedding space."""  # 将视觉/音频软token投影到语言模型嵌入空间

    def __init__(  # 初始化方法
        self,
        multimodal_config: Union[Gemma4AudioConfig, Gemma4VisionConfig],  # 多模态配置（音频或视觉）
        text_config: Gemma4TextConfig,  # 文本配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化

        self.eps = multimodal_config.rms_norm_eps  # RMSNorm epsilon值
        self.text_hidden_size = text_config.hidden_size  # 文本隐藏层大小

        # Audio tower uses output_proj_dims (1536) rather than hidden_size  # 音频塔使用output_proj_dims(1536)而非hidden_size
        # (1024); vision uses hidden_size (768) directly.  # (1024)；视觉直接使用hidden_size(768)
        embedding_dim = (  # 嵌入维度
            getattr(multimodal_config, "output_proj_dims", None)  # 优先使用output_proj_dims
            or multimodal_config.hidden_size  # 否则使用hidden_size
        )

        self.embedding_projection = ReplicatedLinear(  # 嵌入投影线性层
            embedding_dim,  # 输入维度
            self.text_hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("embedding_projection", prefix),  # 添加前缀
        )

        self.embedding_pre_projection_norm = Gemma4RMSNorm(  # 投影前归一化层
            embedding_dim,  # 归一化维度
            eps=self.eps,  # epsilon值
            with_scale=False,  # 无缩放参数
        )

    def forward(  # 前向传播方法
        self,
        inputs_embeds: torch.Tensor,  # 输入嵌入
    ) -> torch.Tensor:  # 返回张量
        """Project soft tokens from a multimodal tower into LM space."""  # 将多模态塔的软token投影到语言模型空间
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)  # 投影前归一化
        embs_proj, _ = self.embedding_projection(embs_normed)  # 通过投影层
        return embs_proj  # 返回投影结果


class Gemma4ForConditionalGeneration(PreTrainedModel):  # Gemma4条件生成模型类
    config_class = Gemma4Config  # 配置类
    """Gemma4 multimodal model for conditional generation."""  # Gemma4多模态条件生成模型

    # BitandBytes specific attributes  # BitandBytes特定属性
    default_bitsandbytes_target_modules = [  # 默认BitandBytes目标模块列表
        ".gate_proj.",  # 门投影
        ".down_proj.",  # 下投影
        ".up_proj.",  # 上投影
        ".q_proj.",  # Q投影
        ".k_proj.",  # K投影
        ".v_proj.",  # V投影
        ".o_proj.",  # O投影
    ]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes堆叠参数映射
        "q_proj": ("qkv_proj", 0),  # Q投影映射到QKV投影的第0个分片
        "k_proj": ("qkv_proj", 1),  # K投影映射到QKV投影的第1个分片
        "v_proj": ("qkv_proj", 2),  # V投影映射到QKV投影的第2个分片
        "gate_proj": ("gate_up_proj", 0),  # 门投影映射到gate_up投影的第0个分片
        "up_proj": ("gate_up_proj", 1),  # 上投影映射到gate_up投影的第1个分片
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
        config: Gemma4Config,  # Gemma4配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__(config=config)  # 调用父类初始化
        self.pp_group = get_pp_group()  # 获取PP组
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        text_config = config.text_config  # 文本配置

        prefix = add_prefix("model", prefix)  # 添加模型前缀

        # Vision/audio encoders + their projection embedders are only consumed  # 视觉/音频编码器及其投影嵌入器仅被
        # at the input-embedding stage, so they live on the first PP rank only.  # 输入嵌入阶段消费，因此它们仅存在于第一个PP rank上
        if self.pp_group.is_first_rank:  # 如果是第一个PP rank
            self.vision_tower = Gemma4VisionEncoder(  # 视觉编码塔
                config=config.vision_config,  # 视觉配置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("vision_tower", prefix),  # 添加前缀
            )
            self.embed_vision = Gemma4MultimodalEmbedder(  # 视觉嵌入器
                config.vision_config,  # 视觉配置
                config.text_config,  # 文本配置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("embed_vision", prefix),  # 添加前缀
            )
            if getattr(config, "audio_config", None) is not None:  # 如果有音频配置
                self.audio_tower = Gemma4AudioEncoder(  # 音频编码塔
                    config=config.audio_config,  # 音频配置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix("audio_tower", prefix),  # 添加前缀
                )
                self.embed_audio = Gemma4MultimodalEmbedder(  # 音频嵌入器
                    config.audio_config,  # 音频配置
                    config.text_config,  # 文本配置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix("embed_audio", prefix),  # 添加前缀
                )
            else:  # 无音频配置
                self.audio_tower = None  # 音频塔为None
                self.embed_audio = None  # 音频嵌入器为None
        else:  # 非第一个PP rank
            self.vision_tower = PPMissingLayer()  # 视觉塔为PP缺失层
            self.embed_vision = PPMissingLayer()  # 视觉嵌入器为PP缺失层
            self.audio_tower = None  # 音频塔为None
            self.embed_audio = None  # 音频嵌入器为None

        self.vocab_size = config.text_config.vocab_size  # 词表大小
        self.vocab_size_per_layer_input = getattr(  # 每层输入词表大小
            config.text_config,  # 文本配置
            "vocab_size_per_layer_input",  # 属性名
            config.text_config.vocab_size,  # 默认使用主词表大小
        )

        # Text model — internal Gemma4TextModel is already PP-aware.  # 文本模型——内部Gemma4TextModel已支持PP
        self.language_model = Gemma4TextModel(  # 语言模型
            config.text_config,  # 文本配置
            quant_config,  # 量化配置
            prefix=add_prefix("language_model", prefix),  # 添加前缀
        )

        # Tied embeddings: under PP the embed_tokens lives on the first rank  # 绑定嵌入：在PP下embed_tokens在第一个rank上
        # while logits run on the last rank, so we can't reuse the embedding  # 而logits在最后一个rank上运行，因此无法复用嵌入
        # module directly.  For PP=1 keep the original tying; for PP>1  # 模块。PP=1保持原始绑定；PP>1
        # materialize a real ParallelLMHead on the last rank and route the  # 在最后一个rank上实例化真正的ParallelLMHead，并在
        # checkpoint embedding into it during load_weights.  # load_weights期间将检查点嵌入路由到其中
        text_tie = getattr(text_config, "tie_word_embeddings", True)  # 是否绑定词嵌入
        if self.pp_group.world_size == 1 and text_tie:  # PP=1且绑定词嵌入
            self.lm_head = self.language_model.embed_tokens  # lm_head共享嵌入层
        elif self.pp_group.is_last_rank:  # 是最后一个PP rank
            self.lm_head = ParallelLMHead(  # 并行语言模型头
                text_config.vocab_size,  # 词表大小
                text_config.hidden_size,  # 隐藏大小
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("lm_head", prefix),  # 添加前缀
            )
        else:  # 其他rank
            self.lm_head = PPMissingLayer()  # 使用PP缺失层占位

        # Create logits processor for the multimodal model  # 为多模态模型创建logits处理器
        self.logits_processor = LogitsProcessor(config.text_config)  # logits处理器
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态

        self.post_init()  # 后初始化

    @property  # 属性装饰器
    def model(self):  # model属性
        # Alias .model to .language_model so this class satisfies the piecewise  # 将.model别名为.language_model，以便此类满足分段
        # CUDA graph gate (which checks `hasattr(model, "model")`). Implemented  # CUDA图门（检查hasattr(model, "model")）的要求。
        # as a property to avoid registering a duplicate submodule in  # 作为属性实现以避免在_modules中注册重复子模块，
        # `_modules`, which would double state_dict keys and disturb  # 这会使state_dict键翻倍并干扰
        # ShardedStateLoader / CPU-offload / dummy-init paths.  # ShardedStateLoader / CPU卸载 / 虚拟初始化路径
        return self.language_model  # 返回语言模型

    def __setattr__(self, name, value):  # 设置属性方法
        # Block writes to "model" so the runner's  # 阻止对"model"的写入，以便runner的
        # `self.model.model = resolve_language_model(self.model)` (which for  # `self.model.model = resolve_language_model(self.model)`
        # this class returns language_model itself) is a no-op rather than a  # （对此类返回language_model本身）是空操作而非
        # nn.Module submodule registration. Without this, nn.Module.__setattr__  # nn.Module子模块注册。否则nn.Module.__setattr__
        # would bypass the @property's setter for Module values and pollute  # 会绕过属性的setter处理Module值，并污染
        # `_modules` with a duplicate alias, doubling state_dict keys.  # _modules中的重复别名，使state_dict键翻倍
        if name == "model":  # 如果属性名为model
            return  # 直接返回，不设置
        super().__setattr__(name, value)  # 调用父类设置属性

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

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:  # 获取嵌入和头权重
        # Gemma 4 multimodal ties its LM head to the text embed_tokens  # Gemma4多模态将LM头绑定到文本embed_tokens
        embed = self.language_model.embed_tokens.weight  # 获取嵌入权重
        return embed, embed  # 返回相同的嵌入权重

    def get_attention_sliding_window_size(self):  # 获取注意力滑动窗口大小
        return getattr(self.config.text_config, "sliding_window", -1) - 1  # 返回滑动窗口大小减1

    def prepare_attn_masks(  # 准备注意力掩码方法
        self,
        forward_batch: ForwardBatch,  # 前向批次
        input_ids: torch.Tensor,  # 输入ID
        mask_dtype: torch.dtype,  # 掩码数据类型
    ):
        """Prepare bidirectional attention masks for image tokens.  # 为图像token准备双向注意力掩码

        Gemma 4 uses bidirectional attention for image soft tokens  # Gemma4对图像软token使用双向注意力
        during prefill. Following the HF implementation, bidirectional attention  # 在预填充期间。遵循HF实现，双向注意力
        is only enabled within each individual image group (same-item  # 仅在每个单独的图像组内（同项
        tokens), not across items.  # token）启用，不跨项。
        Currently only the TritonAttnBackend supports this.  # 目前仅TritonAttnBackend支持此功能

        TODO(kpham-sgl): Guard appropriately for gemma3_mm.py:prepare_attn_masks()  # TODO(kpham-sgl): 适当保护gemma3_mm.py:prepare_attn_masks()
        """
        if not isinstance(get_attn_backend(), TritonAttnBackend):  # 如果不是Triton注意力后端
            logger.warning_once(  # 记录一次警告
                "Bidirectional attention for image tokens requires TritonAttnBackend. "  # 图像token的双向注意力需要TritonAttnBackend。
                "Falling back to causal attention, which may degrade image quality."  # 回退到因果注意力，可能降低图像质量
            )
            return  # 直接返回
        assert forward_batch.forward_mode == ForwardMode.EXTEND  # 断言是扩展模式

        bidirectional_attn_masks_list = []  # 双向注意力掩码列表
        bidirectional_attn_mask_indptr = torch.zeros(  # 双向注意力掩码索引指针
            forward_batch.batch_size + 1, dtype=torch.int32, device=input_ids.device  # 形状和设备
        )

        split_images = []  # 跨块边界的分割图像列表

        for i in range(forward_batch.batch_size):  # 遍历每个批次
            extend_seq_len = forward_batch.extend_seq_lens[i]  # 扩展序列长度
            prefix_len = forward_batch.extend_prefix_lens[i]  # 前缀长度
            bidirectional_attn_mask = torch.zeros(  # 初始化双向注意力掩码
                extend_seq_len,  # 行数
                extend_seq_len + prefix_len,  # 列数
                dtype=mask_dtype,  # 数据类型
                device=input_ids.device,  # 设备
            )
            # Start with causal mask  # 从因果掩码开始
            bidirectional_attn_mask.fill_(1)  # 填充为1
            bidirectional_attn_mask = bidirectional_attn_mask.tril(diagonal=prefix_len)  # 下三角掩码

            # HF only enables bidirectional attention for image tokens,  # HF仅为图像token启用双向注意力，
            # not video or audio (see create_causal_mask_mapping).  # 不为视频或音频启用（见create_causal_mask_mapping）
            mm_inputs = forward_batch.mm_inputs[i]  # 获取多模态输入
            if mm_inputs is not None:  # 如果有多模态输入
                for mm_item in mm_inputs.mm_items:  # 遍历多模态数据项
                    if mm_item.is_image():  # 如果是图像
                        for im_begin, im_end in mm_item.offsets:  # 遍历图像偏移量
                            # Note(kpham-sgl): We only apply bidirectional attention when the image token span  # 注意(kpham-sgl)：仅当图像token跨度
                            # is fully contained in the extend window. Otherwise, we silently fall back to  # 完全包含在扩展窗口内时才应用双向注意力。否则静默回退到
                            # causal attention.  # 因果注意力
                            # FIXME(kpham-sgl): This is a hack to work around the fact that the image token span  # FIXME(kpham-sgl): 这是一个临时方案，解决图像token跨度
                            # might not be fully contained in the extend window during chunked prefill.  # 在分块预填充期间可能不完全包含在扩展窗口中的问题
                            # We should fix this by properly making chunked prefill mask aware.  # 应通过使分块预填充掩码感知来正确修复此问题
                            if (  # 如果图像跨度完全在扩展窗口内
                                im_begin >= prefix_len  # 图像起始 >= 前缀长度
                                and im_end < prefix_len + extend_seq_len  # 图像结束 < 前缀 + 扩展长度
                            ):
                                bidirectional_attn_mask[  # 设置双向注意力区域
                                    im_begin - prefix_len : im_end + 1 - prefix_len,  # 行范围
                                    im_begin : im_end + 1,  # 列范围
                                ] = 1  # 设为1（双向）
                            elif (  # 如果图像跨度部分在扩展窗口内
                                im_end >= prefix_len  # 图像结束 >= 前缀长度
                                and im_begin < prefix_len + extend_seq_len  # 图像起始 < 前缀 + 扩展长度
                            ):
                                split_images.append((i, im_begin, im_end))  # 记录分割图像

            bidirectional_attn_masks_list.append(bidirectional_attn_mask.flatten())  # 展平并添加到列表
            bidirectional_attn_mask_indptr[i + 1] = (  # 更新索引指针
                bidirectional_attn_mask_indptr[i] + bidirectional_attn_mask.nelement()  # 累加元素数
            )
        if split_images:  # 如果有分割图像
            num_split_images = len(split_images)  # 分割图像数量
            logger.warning_once(  # 记录一次警告
                f"{num_split_images} images are split across chunk boundaries. "  # 多少图像跨块边界分割
                "Below are the first 5 images that are split across chunk boundaries: "  # 以下是前5个跨块边界分割的图像
            )
            for i, im_begin, im_end in split_images[:5]:  # 遍历前5个分割图像
                logger.warning_once(  # 记录一次警告
                    f"Image {i}:{im_begin}-{im_end} is split across chunk boundaries.\n",  # 图像i:起始-结束跨块边界分割
                )
            logger.warning_once(  # 记录一次警告
                "Those images will receive causal attention. Disable chunked prefill (--chunked-prefill-size=-1) for full bidirectional attention.",  # 这些图像将接收因果注意力。禁用分块预填充(--chunked-prefill-size=-1)以获取完整双向注意力
            )
        if bidirectional_attn_masks_list:  # 如果有双向注意力掩码
            bidirectional_attn_masks = torch.cat(bidirectional_attn_masks_list, dim=0)  # 拼接所有掩码
            get_attn_backend().forward_metadata.mask_indptr = (  # 设置掩码索引指针
                bidirectional_attn_mask_indptr  # 双向注意力掩码索引指针
            )
            get_attn_backend().forward_metadata.custom_mask = bidirectional_attn_masks  # 设置自定义掩码

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 获取图像特征方法
        vt = self.vision_tower  # 视觉塔引用

        all_embeds = []  # 所有嵌入列表
        for item in items:  # 遍历每个数据项
            all_pixel_values = flatten_nested_list([item.feature])  # 展平像素值
            all_position_ids = flatten_nested_list(  # 展平位置ID
                [getattr(item, "image_position_ids", None)]  # 获取图像位置ID
            )

            for pv_idx, pv in enumerate(all_pixel_values):  # 遍历每个像素值
                if (  # 如果已经是嵌入特征
                    pv.dim() in (2, 3)  # 是2维或3维
                    and pv.shape[-1] == self.config.text_config.hidden_size  # 最后一维等于文本隐藏大小
                ):
                    all_embeds.append(pv.to(self.language_model.device))  # 直接添加到嵌入列表
                    continue  # 跳过后续处理

                if pv_idx >= len(all_position_ids) or all_position_ids[pv_idx] is None:  # 如果没有匹配的位置ID
                    raise ValueError(  # 抛出值错误
                        f"pixel_values[{pv_idx}] has no matching image_position_ids. "  # 像素值没有匹配的图像位置ID
                        "The HF image processor likely renamed this output — "  # HF图像处理器可能重命名了此输出——
                        "update ATTR_NAME_TO_MODALITY in the Gemma4 processor."  # 更新Gemma4处理器中的ATTR_NAME_TO_MODALITY
                    )
                pp = all_position_ids[pv_idx]  # 获取位置ID

                # Vision tower expects 3-D (batch, num_patches, ...).  # 视觉塔期望3维(批次, 补丁数, ...)
                # A single image may arrive as 2-D; add the batch dim if needed.  # 单个图像可能以2维到达；如果需要添加批次维度
                if pv.dim() == 2:  # 如果是2维
                    pv = pv.unsqueeze(0)  # 添加批次维度
                if pp.dim() == 2:  # 如果位置ID是2维
                    pp = pp.unsqueeze(0)  # 添加批次维度

                pv = pv.to(device=vt.device, dtype=self.language_model.dtype())  # 转移设备和数据类型
                pp = pp.to(device=vt.device)  # 转移设备

                pooled, pooler_mask = vt(pv, pp)  # 通过视觉塔获取池化输出和掩码

                for hs, mask in zip(pooled, pooler_mask):  # 遍历每个池化输出
                    real_tokens = hs[mask]  # 获取真实token（非填充）
                    all_embeds.append(  # 添加到嵌入列表
                        self.embed_vision(  # 通过视觉嵌入器
                            inputs_embeds=real_tokens.unsqueeze(0)  # 添加批次维度
                        ).squeeze(0)  # 去除批次维度
                    )

        if all_embeds:  # 如果有嵌入
            return torch.cat(all_embeds, dim=0)  # 拼接并返回
        else:  # 无嵌入
            return torch.empty(  # 返回空张量
                0,  # 第0维大小
                self.language_model.config.hidden_size,  # 隐藏层大小
                device=next(self.parameters()).device,  # 模型设备
                dtype=self.language_model.dtype(),  # 语言模型数据类型
            )

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 获取视频特征方法
        """Encode video frames through the vision tower with video-specific pooling.  # 通过视觉塔用视频特定的池化编码视频帧

        Each video is (num_frames, num_patches, patch_pixels) with matching  # 每个视频是(帧数, 补丁数, 补丁像素)，带有匹配的
        position_ids (num_frames, num_patches, 2).  Frames are flattened into  # 位置ID(帧数, 补丁数, 2)。帧被展平到
        the batch dimension so each frame is encoded independently, then pooled  # 批次维度，每帧独立编码，然后根据
        dynamically based on the input patch count and pooling_kernel_size.  # 输入补丁数和pooling_kernel_size动态池化
        """
        vt = self.vision_tower  # 视觉塔引用

        all_embeds = []  # 所有嵌入列表
        for item in items:  # 遍历每个数据项
            all_pixel_values = flatten_nested_list([item.feature])  # 展平像素值
            all_position_ids = flatten_nested_list(  # 展平位置ID
                [getattr(item, "video_position_ids", None)]  # 获取视频位置ID
            )

            for pv_idx, pv in enumerate(all_pixel_values):  # 遍历每个像素值
                if (  # 如果已经是嵌入特征
                    pv.dim() in (2, 3)  # 是2维或3维
                    and pv.shape[-1] == self.config.text_config.hidden_size  # 最后一维等于文本隐藏大小
                ):
                    all_embeds.append(pv.to(self.language_model.device))  # 直接添加到嵌入列表
                    continue  # 跳过后续处理

                if pv_idx >= len(all_position_ids) or all_position_ids[pv_idx] is None:  # 如果没有匹配的位置ID
                    raise ValueError(  # 抛出值错误
                        f"pixel_values_videos[{pv_idx}] has no matching video_position_ids."  # 像素值没有匹配的视频位置ID
                    )
                pp = all_position_ids[pv_idx]  # 获取位置ID

                # HF processor returns 4-D tensors  # HF处理器返回4维张量
                # (num_videos, num_frames, num_patches, ...) — collapse to  # (视频数, 帧数, 补丁数, ...)——折叠为
                # 3-D (num_frames, num_patches, ...) so each frame is a  # 3维(帧数, 补丁数, ...)，使每帧成为
                # batch element for the vision tower.  # 视觉塔的批次元素
                if pv.dim() == 4:  # 如果是4维
                    pv = pv.reshape(-1, pv.shape[-2], pv.shape[-1])  # 折叠为3维
                if pp.dim() == 4:  # 如果位置ID是4维
                    pp = pp.reshape(-1, pp.shape[-2], pp.shape[-1])  # 折叠为3维

                pv = pv.to(device=vt.device, dtype=self.language_model.dtype())  # 转移设备和数据类型
                pp = pp.to(device=vt.device)  # 转移设备

                pooled, pooler_mask = vt(pv, pp)  # 通过视觉塔获取池化输出和掩码

                for hs, mask in zip(pooled, pooler_mask):  # 遍历每个池化输出
                    real_tokens = hs[mask]  # 获取真实token（非填充）
                    all_embeds.append(  # 添加到嵌入列表
                        self.embed_vision(  # 通过视觉嵌入器
                            inputs_embeds=real_tokens.unsqueeze(0)  # 添加批次维度
                        ).squeeze(0)  # 去除批次维度
                    )

        if all_embeds:  # 如果有嵌入
            return torch.cat(all_embeds, dim=0)  # 拼接并返回
        else:  # 无嵌入
            return torch.empty(  # 返回空张量
                0,  # 第0维大小
                self.language_model.config.hidden_size,  # 隐藏层大小
                device=next(self.parameters()).device,  # 模型设备
                dtype=self.language_model.dtype(),  # 语言模型数据类型
            )

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 获取音频特征方法
        if self.audio_tower is None:  # 如果没有音频塔
            raise ValueError(  # 抛出值错误
                "Audio inputs provided but the model does not have an audio tower."  # 提供了音频输入但模型没有音频塔
            )

        all_input_features = flatten_nested_list([item.feature for item in items])  # 展平所有输入特征
        all_input_features_mask = flatten_nested_list(  # 展平所有输入特征掩码
            [~item.input_features_mask for item in items]  # 取反掩码
        )

        all_embeds = []  # 所有嵌入列表
        for input_features, input_features_mask in zip(  # 遍历输入特征和掩码
            all_input_features, all_input_features_mask  # 配对的输入特征和掩码
        ):
            if input_features.dim() == 2:  # 如果是2维
                input_features = input_features.unsqueeze(0)  # 添加批次维度
            if input_features_mask.dim() == 1:  # 如果掩码是1维
                input_features_mask = input_features_mask.unsqueeze(0)  # 添加批次维度

            input_features = input_features.to(  # 转移输入特征
                device=self.audio_tower.device,  # 音频塔设备
                dtype=self.language_model.dtype(),  # 语言模型数据类型
            )
            input_features_mask = input_features_mask.to(device=input_features.device)  # 转移掩码到相同设备

            # audio_mel_mask convention: True = padding  # audio_mel_mask约定：True = 填充
            audio_encodings, audio_mask = self.audio_tower(  # 通过音频塔编码
                input_features, input_features_mask  # 输入特征和掩码
            )

            audio_features = self.embed_audio(inputs_embeds=audio_encodings)  # 通过音频嵌入器

            for enc, mask in zip(audio_features, audio_mask):  # 遍历每个编码
                all_embeds.append(enc[~mask])  # 添加非填充位置的编码

        if all_embeds:  # 如果有嵌入
            return torch.cat(all_embeds, dim=0)  # 拼接并返回
        else:  # 无嵌入
            return torch.empty(  # 返回空张量
                0,  # 第0维大小
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
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量（可选）
        **kwargs: object,  # 其他关键字参数
    ) -> Union[LogitsProcessor, PPProxyTensors]:  # 返回类型
        """Forward pass for multimodal Gemma4."""  # Gemma4多模态前向传播
        is_first_rank = self.pp_group.is_first_rank  # 是否是第一个PP rank
        is_last_rank = self.pp_group.is_last_rank  # 是否是最后一个PP rank

        # Only the first PP rank consumes input_ids/input_embeds; later stages  # 仅第一个PP rank消费input_ids/input_embeds；后续阶段
        # receive activations through pp_proxy_tensors.  # 通过pp_proxy_tensors接收激活
        if is_first_rank and (input_ids is None) ^ (input_embeds is not None):  # 如果是第一rank且恰好有一个为None
            raise ValueError(  # 抛出值错误
                "You must specify exactly one of input_ids or inputs_embeds"  # 必须指定input_ids或inputs_embeds中的一个
            )

        if envs.SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION.get():  # 如果启用非原地位置变异
            positions = positions + 1  # 非原地加1
        else:  # 否则
            positions += 1  # 原地加1

        per_layer_inputs = None  # 每层输入初始化为None
        # PLE table and the per-layer projection live on the first rank only,  # PLE表和每层投影仅存在于第一个rank上，
        # so non-first ranks must skip this and pull per_layer_inputs from the  # 因此非第一rank必须跳过此步骤，并从
        # PP proxy (forwarded by Gemma4TextModel).  # PP代理中获取per_layer_inputs（由Gemma4TextModel转发）
        if is_first_rank and input_ids is not None:  # 如果是第一rank且提供了输入ID
            ple_ids = input_ids.clone()  # 克隆输入ID用于PLE
            pad_id = self.config.text_config.pad_token_id  # 填充token ID
            ple_ids[input_ids == self.config.image_token_id] = pad_id  # 将图像token替换为填充ID
            ple_ids[input_ids == self.config.video_token_id] = pad_id  # 将视频token替换为填充ID
            ple_ids[input_ids == self.config.audio_token_id] = pad_id  # 将音频token替换为填充ID
            per_layer_inputs = self.get_per_layer_inputs(ple_ids)  # 获取每层输入

        # Prepare bidirectional attention masks for image tokens during prefill.  # 在预填充期间为图像token准备双向注意力掩码
        # mm_inputs is preserved on every PP rank up to the first-rank embed  # mm_inputs在每个PP rank上保留直到第一rank嵌入
        # routine, so each rank's attn_backend can install the mask locally.  # 例程，因此每个rank的attn_backend可以本地安装掩码
        if (  # 如果
            forward_batch.forward_mode == ForwardMode.EXTEND  # 是扩展模式
            and forward_batch.contains_image_inputs()  # 且包含图像输入
        ):
            self.prepare_attn_masks(  # 准备注意力掩码
                forward_batch,  # 前向批次
                input_ids,  # 输入ID
                mask_dtype=torch.bool,  # 掩码数据类型
            )

        # general_mm_embed_routine already handles PP: it skips the embedding  # general_mm_embed_routine已处理PP：它跳过
        # work on non-first ranks and forwards pp_proxy_tensors via **kwargs.  # 非第一rank上的嵌入工作，并通过**kwargs转发pp_proxy_tensors
        hidden_states = general_mm_embed_routine(  # 调用通用多模态嵌入例程
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.language_model,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数映射
                Modality.IMAGE: self.get_image_feature,  # 图像模态
                Modality.VIDEO: self.get_video_feature,  # 视频模态
                Modality.AUDIO: self.get_audio_feature,  # 音频模态
            },
            positions=positions,  # 位置编码
            per_layer_inputs=per_layer_inputs,  # 每层输入
            pp_proxy_tensors=pp_proxy_tensors,  # PP代理张量
            **kwargs,  # 其他关键字参数
        )

        if not is_last_rank:  # 如果不是最后一个PP rank
            # `hidden_states` is actually a PPProxyTensors flowing to the next  # hidden_states实际上是流向下一阶段的PPProxyTensors
            # stage; logits processing happens on the last rank only.  # ；logits处理仅在最后一个rank上进行
            return hidden_states  # 返回代理张量

        # Unpack aux_hidden_states if Eagle3 capture is active  # 如果Eagle3捕获激活则解包辅助隐藏状态
        aux_hidden_states = None  # 辅助隐藏状态初始化为None
        if self.capture_aux_hidden_states:  # 如果捕获辅助隐藏状态
            hidden_states, aux_hidden_states = hidden_states  # 解包

        # PP=1 keeps the original tied-weight behavior of using embed_tokens  # PP=1保持使用embed_tokens的原始绑定权重行为
        # directly; under PP we route through the dedicated lm_head module.  # ；在PP下我们通过专用的lm_head模块路由
        head = (  # 选择头模块
            self.language_model.embed_tokens  # 使用嵌入层
            if self.pp_group.world_size == 1  # 如果PP=1
            and getattr(self.config.text_config, "tie_word_embeddings", True)  # 且绑定词嵌入
            else self.lm_head  # 否则使用lm_head
        )
        return self.logits_processor(  # 通过logits处理器
            input_ids,  # 输入ID
            hidden_states,  # 隐藏状态
            head,  # 头模块
            forward_batch,  # 前向批次
            aux_hidden_states,  # 辅助隐藏状态
        )

    def tie_weights(self, recompute_mapping=False):  # 绑定权重方法
        # Under PP, embed_tokens (first rank) and lm_head (last rank) live on  # 在PP下，embed_tokens（第一个rank）和lm_head（最后一个rank）位于
        # different processes, so HF's automatic tying would crash on the  # 不同进程上，因此HF的自动绑定会在
        # PPMissingLayer side.  load_weights routes the embedding into lm_head  # PPMissingLayer侧崩溃。load_weights将嵌入路由到lm_head
        # on the last rank explicitly, so the tie is a no-op under PP.  # 在最后一个rank上显式路由，因此绑定在PP下是空操作
        if self.pp_group.world_size > 1:  # 如果PP>1
            return  # 直接返回
        return self.language_model.tie_weights()  # 返回语言模型的权重绑定

    # Standard stacked-params mapping for fused QKV / GateUp linears  # 融合QKV/GateUp线性层的标准堆叠参数映射
    # in the text decoder.  Also consumed by the tower QKV remap (step 2).  # 在文本解码器中。也被塔QKV重映射使用（步骤2）
    stacked_params_mapping = [  # 堆叠参数映射
        # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
        (".qkv_proj", ".q_proj", "q"),  # Q投影映射
        (".qkv_proj", ".k_proj", "k"),  # K投影映射
        (".qkv_proj", ".v_proj", "v"),  # V投影映射
        (".gate_up_proj", ".up_proj", 1),  # 上投影映射
        (".gate_up_proj", ".gate_proj", 0),  # 门投影映射
    ]

    # Regex for fused QKV in vision/audio towers.  # 视觉/音频塔中融合QKV的正则表达式
    # Vision: *.self_attn.{q,k,v}_proj.*  Audio: *.attn.{q,k,v}_proj.*  # 视觉：*.self_attn.{q,k,v}_proj.*  音频：*.attn.{q,k,v}_proj.*
    _RE_TOWER_QKV = re.compile(  # 塔QKV正则
        r"(.+\.(?:self_attn|attn))\.(q_proj|k_proj|v_proj)\.(.*)"  # 匹配Q/K/V投影
    )
    # Regex for fused GateUp in the vision tower MLP.  # 视觉塔MLP中融合GateUp的正则表达式
    _RE_TOWER_GATE_UP = re.compile(r"(.+\.mlp)\.(gate_proj|up_proj)\.(.*)")  # 匹配gate/up投影

    _RE_AUDIO_LAYER = re.compile(r"(audio_tower)\.layers\.(\d+)\.(.*)")  # 音频层正则

    @staticmethod  # 静态方法
    def _remap_audio_tower_name(name: str) -> str:  # 重映射音频塔名称方法
        """Remap audio tower checkpoint names to our module tree.  # 将音频塔检查点名称重映射到我们的模块树

        Checkpoint naming (``layers``, ``self_attn``, ``feed_forward1/2``, etc.)  # 检查点命名(``layers``, ``self_attn``, ``feed_forward1/2``等)
        differs from our module tree (``conformer``, ``attention.attn``,  # 与我们的模块树(``conformer``, ``attention.attn``、
        ``ffw_layer_start/end``, etc.).  Applied before ``_remap_tower_name``.  # ``ffw_layer_start/end``等)不同。在_remap_tower_name之前应用
        """
        if "audio_tower." not in name:  # 如果名称不包含audio_tower
            return name  # 直接返回

        # SSCP conv block: layer0/layer1 → conv_0/conv_1  # SSCP卷积块：layer0/layer1 → conv_0/conv_1
        name = name.replace(  # 替换SSCP层名
            "subsample_conv_projection.layer0.",  # 原始名称
            "subsample_conv_projection.conv_0.",  # 目标名称
        )
        name = name.replace(  # 替换SSCP层名
            "subsample_conv_projection.layer1.",  # 原始名称
            "subsample_conv_projection.conv_1.",  # 目标名称
        )

        # Conformer layers: audio_tower.layers.{i} → audio_tower.conformer.{i}  # Conformer层：audio_tower.layers.{i} → audio_tower.conformer.{i}
        m = Gemma4ForConditionalGeneration._RE_AUDIO_LAYER.match(name)  # 匹配音频层
        if m:  # 如果匹配
            tower, layer_idx, suffix = m.groups()  # 提取塔名、层索引和后缀

            # Order matters: more specific patterns first.  # 顺序重要：更具体的模式优先
            # relative_k_proj → relative_position_embedding.pos_proj  # relative_k_proj → relative_position_embedding.pos_proj
            suffix = suffix.replace(  # 替换相对K投影
                "self_attn.relative_k_proj.",  # 原始名称
                "attention.attn.relative_position_embedding.pos_proj.",  # 目标名称
            )
            # self_attn.post → attention.post (the output projection)  # self_attn.post → attention.post（输出投影）
            suffix = suffix.replace("self_attn.post.", "attention.post.")  # 替换后投影
            # general self_attn → attention.attn  # 通用self_attn → attention.attn
            suffix = suffix.replace("self_attn.", "attention.attn.")  # 替换自注意力
            # norms  # 归一化层
            suffix = suffix.replace("norm_pre_attn.", "attention.pre_attn_norm.")  # 替换注意力前归一化
            suffix = suffix.replace("norm_post_attn.", "attention.post_norm.")  # 替换注意力后归一化
            suffix = suffix.replace("norm_out.", "norm.")  # 替换输出归一化
            # feed-forward blocks  # 前馈块
            suffix = suffix.replace("feed_forward1.", "ffw_layer_start.")  # 替换前馈层1
            suffix = suffix.replace("feed_forward2.", "ffw_layer_end.")  # 替换前馈层2

            name = f"{tower}.conformer.{layer_idx}.{suffix}"  # 构建新名称

        return name  # 返回重映射后的名称

    @staticmethod  # 静态方法
    def _remap_tower_name(name: str, params_dict: dict) -> str:  # 重映射塔名称方法
        """Remap a vision/audio tower checkpoint name to our module tree.  # 将视觉/音频塔检查点名称重映射到我们的模块树

        Three transformations, applied in order:  # 按顺序应用三个变换：

        1. **Fused QKV** — ``{q,k,v}_proj.*`` → ``qkv.*``  # 1. 融合QKV——{q,k,v}_proj.* → qkv.*
           Weight/bias are redirected into the fused ``qkv.{proj}.{attr}``  # 权重/偏置被重定向到融合的qkv.{proj}.{attr}
           namespace (stacked-params then merges them into ``qkv_proj``).  # 命名空间（堆叠参数然后合并为qkv_proj）
           Clip buffers are split: ``input_*`` → shared ``qkv.input_*``,  # 裁剪缓冲区分割：input_* → 共享的qkv.input_*，
           ``output_*`` → per-projection ``qkv.{q,k,v}_output_*``.  # output_* → 每投影的qkv.{q,k,v}_output_*

        2. **Fused GateUp** — ``{gate,up}_proj.*`` → ``gate_up.*``  # 2. 融合GateUp——{gate,up}_proj.* → gate_up.*
           Same pattern as QKV.  # 与QKV相同的模式

        3. **Clippable wrapper** — ``*.weight``/``*.bias`` → ``*.linear.weight``  # 3. 可裁剪包装——*.weight/*.bias → *.linear.weight
           Catches the remaining (non-fused) clippable linears whose inner  # 捕获剩余的（非融合的）可裁剪线性层，其内部
           ``RowParallelLinear``/``ColumnParallelLinear`` lives at ``.linear``.  # RowParallelLinear/ColumnParallelLinear位于.linear
           Falls back to the original name when ``.linear.`` does not exist  # 当.linear.在params_dict中不存在时回退到原始名称
           in ``params_dict`` (plain linears, norms, conv weights, etc.).  # （普通线性层、归一化、卷积权重等）
        """
        # Step 1: fused QKV  # 步骤1：融合QKV
        m = Gemma4ForConditionalGeneration._RE_TOWER_QKV.match(name)  # 匹配QKV
        if m:  # 如果匹配
            pfx, proj, attr = m.groups()  # 提取前缀、投影名和属性
            if attr in ("weight", "bias", "linear.weight", "linear.bias"):  # 如果是权重/偏置
                bare_attr = attr.rsplit(".", 1)[-1]  # 提取裸属性名
                return f"{pfx}.qkv.{proj}.{bare_attr}"  # 返回融合QKV名称
            if attr.startswith("output_"):  # 如果是输出裁剪缓冲区
                return f"{pfx}.qkv.{proj[0]}_{attr}"  # 返回每投影输出名称
            if attr.startswith("input_"):  # 如果是输入裁剪缓冲区
                return f"{pfx}.qkv.{attr}"  # 返回共享输入名称

        # Step 2: fused GateUp  # 步骤2：融合GateUp
        m = Gemma4ForConditionalGeneration._RE_TOWER_GATE_UP.match(name)  # 匹配GateUp
        if m:  # 如果匹配
            pfx, proj, attr = m.groups()  # 提取前缀、投影名和属性
            short = proj.split("_")[0]  # "gate" or "up"  # "gate"或"up"
            if attr in ("weight", "bias", "linear.weight", "linear.bias"):  # 如果是权重/偏置
                bare_attr = attr.rsplit(".", 1)[-1]  # 提取裸属性名
                return f"{pfx}.gate_up.{proj}.{bare_attr}"  # 返回融合GateUp名称
            if attr.startswith("output_"):  # 如果是输出裁剪缓冲区
                return f"{pfx}.gate_up.{short}_{attr}"  # 返回每投影输出名称
            if attr.startswith("input_"):  # 如果是输入裁剪缓冲区
                return f"{pfx}.gate_up.{attr}"  # 返回共享输入名称

        # Step 3: clippable wrapper (.weight → .linear.weight)  # 步骤3：可裁剪包装（.weight → .linear.weight）
        if name.endswith(".weight") or name.endswith(".bias"):  # 如果以.weight或.bias结尾
            base, attr = name.rsplit(".", 1)  # 分割基名和属性
            alt = f"{base}.linear.{attr}"  # 替代名称
            if alt in params_dict:  # 如果替代名称在参数字典中
                return alt  # 返回替代名称

        return name  # 返回原始名称

    def _get_k_eq_v_layers(self) -> set:  # 获取K等于V的层集合
        """Return set of layer indices where attention_k_eq_v applies (full-attention layers)."""  # 返回attention_k_eq_v适用的层索引集合（全注意力层）
        text_config = self.config.text_config  # 文本配置
        if not getattr(text_config, "attention_k_eq_v", False):  # 如果未启用attention_k_eq_v
            return set()  # 返回空集合
        return {  # 返回
            i for i, lt in enumerate(text_config.layer_types) if lt == "full_attention"  # 所有全注意力层的索引
        }

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法
        k_eq_v_layers = self._get_k_eq_v_layers()  # 获取K=V的层集合

        num_experts = getattr(self.config.text_config, "num_experts", 0) or 0  # 专家数量
        expert_params_mapping = [  # 专家参数映射
            # (param_name, ckpt_weight_name, shard_ids)  # (参数名, 检查点权重名, 分片ID)
            # gate_up_proj is fused [E, 2*I, H] — chunk into w1 (gate) + w3 (up)  # gate_up_proj是融合的[E, 2*I, H]——分块为w1(门) + w3(上)
            ("experts.w13_weight", "experts.gate_up_proj", ("w1", "w3")),  # w13权重映射
            ("experts.w2_weight", "experts.down_proj", ("w2",)),  # w2权重映射
        ]

        # Per-expert checkpoint format used by compressed-tensors / FP8  # compressed-tensors / FP8使用的每专家检查点格式
        # (e.g. RedHatAI/*-FP8-Dynamic) and by ModelOpt NVFP4  # （例如RedHatAI/*-FP8-Dynamic）和ModelOpt NVFP4
        # (e.g. nvidia/Gemma-4-*-NVFP4). Each expert is stored as a  # （例如nvidia/Gemma-4-*-NVFP4）。每个专家存储为
        # separate key with shape (out, in):  # 形状为(out, in)的单独键：
        #   experts.<id>.{gate,up,down}_proj.{weight,weight_scale,  #   experts.<id>.{gate,up,down}_proj.{weight,weight_scale,
        #                                     weight_scale_2,input_scale}  #                                     weight_scale_2,input_scale}
        # `make_expert_params_mapping` emits tuples whose `weight_name` ends  # `make_expert_params_mapping`发出weight_name以
        # in a trailing dot, so the standard `name.replace(weight_name,  # 尾部点结尾的元组，因此标准的name.replace(weight_name,
        # param_name)` collapses every suffix uniformly to the fused  # param_name)将每个后缀统一折叠到融合的
        # FusedMoE params (experts.w13_*, experts.w2_*).  # FusedMoE参数（experts.w13_*, experts.w2_*）
        per_expert_params_mapping = (  # 每专家参数映射
            FusedMoE.make_expert_params_mapping(  # 创建每专家参数映射
                ckpt_gate_proj_name="gate_proj",  # 检查点门投影名
                ckpt_down_proj_name="down_proj",  # 检查点下投影名
                ckpt_up_proj_name="up_proj",  # 检查点上投影名
                num_experts=num_experts,  # 专家数量
            )
            if num_experts  # 如果有专家
            else []  # 否则为空列表
        )

        params_dict = dict(self.named_parameters())  # 获取参数字典
        params_dict.update(dict(self.named_buffers()))  # 更新缓冲区字典
        non_persistent_buffers: Set[str] = set()  # 非持久化缓冲区集合
        for mod_name, mod in self.named_modules():  # 遍历所有模块
            for buf_name in getattr(mod, "_non_persistent_buffers_set", set()):  # 遍历非持久化缓冲区
                full = f"{mod_name}.{buf_name}" if mod_name else buf_name  # 完整名称
                non_persistent_buffers.add(full)  # 添加到集合

        text_tie = getattr(self.config.text_config, "tie_word_embeddings", True)  # 是否绑定词嵌入
        start_layer = self.language_model.start_layer  # 起始层
        end_layer = self.language_model.end_layer  # 结束层

        loaded_params: Set[str] = set()  # 已加载参数集合

        for name, loaded_weight in weights:  # 遍历权重
            if "embed_vision.embedding." in name or "embed_audio.embedding." in name:  # 跳过多模态嵌入的embedding层
                continue  # 跳过
            if self.audio_tower is None and (  # 如果没有音频塔且
                "audio_tower." in name or "embed_audio." in name  # 名称包含音频相关
            ):
                continue  # 跳过

            name = re.sub(r"^model\.", "", name)  # 移除model.前缀

            if pp_filter_load_weight(  # PP过滤
                name,  # 参数名
                loaded_weight,  # 权重
                pp_group=self.pp_group,  # PP组
                start_layer=start_layer,  # 起始层
                end_layer=end_layer,  # 结束层
                params_dict=params_dict,  # 参数字典
                loaded_params=loaded_params,  # 已加载参数
                tie_word_embeddings=text_tie,  # 是否绑定词嵌入
                embed_weight_name="language_model.embed_tokens.weight",  # 嵌入权重名称
                first_rank_only_patterns=(  # 仅第一个rank的模式
                    "language_model.embed_tokens",  # 嵌入层
                    "language_model.per_layer_model_projection",  # 每层模型投影
                    "language_model.per_layer_projection_norm",  # 每层投影归一化
                    "vision_tower.",  # 视觉塔
                    "embed_vision.",  # 视觉嵌入器
                    "audio_tower.",  # 音频塔
                    "embed_audio.",  # 音频嵌入器
                ),
                last_rank_only_prefixes=("language_model.norm.", "lm_head."),  # 仅最后一个rank的前缀
            ):
                continue  # 跳过

            # HF has router.per_expert_scale and experts.* on the decoder layer;  # HF在解码器层上有router.per_expert_scale和experts.*
            # remap into our moe.* subtree since Gemma4MoE owns both.  # 重映射到我们的moe.*子树，因为Gemma4MoE拥有两者
            name = name.replace(".router.per_expert_scale", ".moe.per_expert_scale")  # 重映射路由器缩放
            if ".experts." in name and ".moe.experts." not in name:  # 如果有experts但不在moe下
                name = name.replace(".experts.", ".moe.experts.")  # 重映射到moe子树

            # Remap audio tower checkpoint names to our module tree  # 将音频塔检查点名称重映射到我们的模块树
            if "audio_tower." in name:  # 如果名称包含audio_tower
                name = self._remap_audio_tower_name(name)  # 重映射音频塔名称

            # Remap vision / audio tower names (fused QKV/GateUp, clippable wrappers)  # 重映射视觉/音频塔名称（融合QKV/GateUp，可裁剪包装）
            if "vision_tower." in name or "audio_tower." in name:  # 如果名称包含视觉或音频塔
                name = self._remap_tower_name(name, params_dict)  # 重映射塔名称

            # attention_k_eq_v: full-attention layers have no v_proj in the  # attention_k_eq_v：全注意力层在检查点中没有v_proj
            # checkpoint (K and V share weights).  When we see a k_proj weight  # （K和V共享权重）。当我们看到k_proj权重
            # for one of these layers, load it into both the "k" and "v" shards  # 对于这些层之一，将其加载到融合QKV的"k"和"v"分片中
            # of the fused QKV so the forward produces v_raw == k_raw.  # 以便前向传播产生v_raw == k_raw
            should_dup_k_to_v = (  # 是否应将K复制到V
                ".k_proj." in name  # 名称包含k_proj
                and k_eq_v_layers  # 有K=V层
                and "language_model." in name  # 名称包含language_model
                and (m := re.search(r"layers\.(\d+)\.", name)) is not None  # 提取层索引
                and int(m.group(1)) in k_eq_v_layers  # 层索引在K=V集合中
            )

            # MoE expert weights checked first (gate_up_proj contains "up_proj"  # 首先检查MoE专家权重（gate_up_proj包含"up_proj"
            # which would false-match the stacked dense MLP mapping).  # 会误匹配堆叠稠密MLP映射）
            orig_name = name  # 保存原始名称

            # 1) Per-expert checkpoint layout (compressed-tensors FP8 like  # 1) 每专家检查点布局（如compressed-tensors FP8
            #    RedHatAI/*-FP8-Dynamic, ModelOpt NVFP4 like  #    RedHatAI/*-FP8-Dynamic，如ModelOpt NVFP4
            #    nvidia/Gemma-4-*-NVFP4): experts.<id>.{gate,up,down}_proj.*  #    nvidia/Gemma-4-*-NVFP4）：experts.<id>.{gate,up,down}_proj.*
            #    The trailing dot in `weight_name` lets a single mapping fold  # weight_name中的尾部点让单个映射可以折叠
            #    weight, weight_scale, weight_scale_2, and input_scale into  # weight, weight_scale, weight_scale_2和input_scale到
            #    their corresponding fused FusedMoE params (experts.w13_*,  # 对应的融合FusedMoE参数（experts.w13_*，
            #    experts.w2_*).  #    experts.w2_*）
            for (  # 遍历每专家参数映射
                param_name,  # 参数名
                weight_name,  # 权重名
                expert_id,  # 专家ID
                shard_id,  # 分片ID
            ) in per_expert_params_mapping:
                if weight_name not in orig_name:  # 如果权重名不在原始名称中
                    continue  # 跳过
                name = orig_name.replace(weight_name, param_name)  # 替换权重名
                if name not in params_dict:  # 如果名称不在参数字典中
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(  # 加载权重
                    param,  # 参数
                    loaded_weight,  # 权重
                    name,  # 参数名
                    shard_id=shard_id,  # 分片ID
                    expert_id=expert_id,  # 专家ID
                )
                loaded_params.add(name)  # 添加到已加载集合
                break  # 跳出内层循环
            else:  # 如果没有匹配每专家映射
                # 2) BF16 fused checkpoint layout: experts.gate_up_proj is a  # 2) BF16融合检查点布局：experts.gate_up_proj是一个
                #    [E, 2*I, H] tensor that needs per-expert chunking into  #    [E, 2*I, H]张量，需要按专家分块为
                #    w1 (gate) and w3 (up).  #    w1(门)和w3(上)
                for param_name, weight_name, shard_ids in expert_params_mapping:  # 遍历专家参数映射
                    name = orig_name  # 恢复原始名称
                    if weight_name not in name:  # 如果权重名不在名称中
                        continue  # 跳过
                    name = name.replace(weight_name, param_name)  # 替换权重名
                    if name not in params_dict:  # 如果名称不在参数字典中
                        continue  # 跳过
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    for i in range(num_experts):  # 遍历每个专家
                        chunks = loaded_weight[i].chunk(len(shard_ids), dim=0)  # 分块权重
                        for chunk, sid in zip(chunks, shard_ids):  # 遍历分块和分片ID
                            weight_loader(param, chunk, name, sid, i)  # 加载权重
                    loaded_params.add(name)  # 添加到已加载集合
                    break  # 跳出内层循环
                else:  # 如果没有匹配专家映射
                    for (  # 遍历堆叠参数映射
                        param_name,  # 参数名
                        weight_name,  # 权重名
                        shard_id,  # 分片ID
                    ) in self.stacked_params_mapping:
                        name = orig_name  # 恢复原始名称
                        if weight_name not in name:  # 如果权重名不在名称中
                            continue  # 跳过
                        name = name.replace(weight_name, param_name)  # 替换权重名
                        if name not in params_dict:  # 如果名称不在参数字典中
                            continue  # 跳过
                        param = params_dict[name]  # 获取参数
                        weight_loader = param.weight_loader  # 获取权重加载器
                        weight_loader(param, loaded_weight, shard_id)  # 加载权重
                        if should_dup_k_to_v:  # 如果应将K复制到V
                            weight_loader(param, loaded_weight, "v")  # 加载K权重到V分片
                        loaded_params.add(name)  # 添加到已加载集合
                        break  # 跳出内层循环
                    else:  # 如果没有匹配任何映射
                        name = orig_name  # 恢复原始名称
                        if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                            continue  # 跳过
                        name = maybe_remap_kv_scale_name(name, params_dict)  # 可能重映射KV缩放名称
                        if name is None:  # 如果名称为None
                            continue  # 跳过
                        if name not in params_dict:  # 如果名称不在参数字典中
                            continue  # 跳过
                        param = params_dict[name]  # 获取参数
                        weight_loader = getattr(  # 获取权重加载器
                            param, "weight_loader", default_weight_loader  # 默认使用default_weight_loader
                        )
                        weight_loader(param, loaded_weight)  # 加载权重
                        loaded_params.add(name)  # 添加到已加载集合
        unloaded_params = params_dict.keys() - loaded_params  # 未加载参数
        if unloaded_params:  # 如果有未加载参数
            param_names = set(dict(self.named_parameters()).keys())  # 参数名集合
            buckets = {  # 日志级别分桶
                logging.WARNING: (  # 警告级别
                    "Some weights are not initialized from checkpoints",  # 某些权重未从检查点初始化
                    lambda p: p in param_names,  # 过滤条件：是参数
                ),
                logging.INFO: (  # 信息级别
                    "Persistent buffers not in checkpoint (using default init)",  # 持久化缓冲区不在检查点中（使用默认初始化）
                    lambda p: p not in param_names and p not in non_persistent_buffers,  # 过滤条件
                ),
                logging.DEBUG: (  # 调试级别
                    "Non-persistent buffers not in checkpoint (expected)",  # 非持久化缓冲区不在检查点中（预期行为）
                    lambda p: p in non_persistent_buffers,  # 过滤条件
                ),
            }
            for level, (msg, pred) in buckets.items():  # 遍历日志级别
                names = sorted(p for p in unloaded_params if pred(p))  # 过滤并排序
                if names:  # 如果有名称
                    logger.log(level, "%s: %s", msg, names)  # 记录日志
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

    def get_embed(self):  # 获取嵌入权重
        return self.language_model.embed_tokens.weight  # 返回语言模型嵌入权重

    def get_embed_and_head(self):  # 获取嵌入和头权重
        if self.pp_group.world_size > 1:  # 如果PP>1
            # Under PP, embed_tokens lives on the first rank and lm_head on the  # 在PP下，embed_tokens在第一个rank上，lm_head在
            # last; neither rank holds both tensors, so we can't return the  # 最后一个上；两个rank都不持有两个张量，因此无法
            # pair locally without a cross-stage gather.  Callers (RL weight  # 在本地返回该对，除非跨阶段收集。调用者（RL权重
            # sync, remote weight loader) currently assume a single-rank view —  # 同步，远程权重加载器）当前假设单rank视图——
            # fail loudly rather than dereference a PPMissingLayer.  # 大声失败而不是解引用PPMissingLayer
            raise NotImplementedError(  # 抛出未实现错误
                "get_embed_and_head() is not implemented for Gemma4 "  # Gemma4未实现get_embed_and_head()
                "multimodal under pipeline parallelism. embed_tokens lives "  # 多模态在流水线并行下。embed_tokens在
                "on the first PP rank and lm_head on the last; use "  # 第一个PP rank上，lm_head在最后一个上；使用
                "--pp-size 1 if you need this API."  # --pp-size 1如果需要此API
            )
        embed = self.language_model.embed_tokens.weight  # 获取嵌入权重
        # Gemma4 ties word embeddings, so embed_tokens serves as lm_head  # Gemma4绑定词嵌入，因此embed_tokens作为lm_head
        return embed, embed  # 返回嵌入权重对

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):  # 设置Eagle3要捕获的层
        self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
        text_config = self.config.text_config  # 文本配置
        if layer_ids is None:  # 如果未指定层ID
            num_layers = text_config.num_hidden_layers  # 总层数
            self.language_model.layers_to_capture = [  # 默认捕获层
                2,  # 第2层
                num_layers // 2,  # 中间层
                num_layers - 3,  # 倒数第3层
            ]
        else:  # 指定了层ID
            # we plus 1 here because in sglang, for the ith layer, it takes the output  # 这里加1因为在sglang中，第i层取
            # of the (i-1)th layer as aux hidden state  # 第(i-1)层的输出作为辅助隐藏状态
            self.language_model.layers_to_capture = [val + 1 for val in layer_ids]  # 加1偏移


EntryClass = Gemma4ForConditionalGeneration  # 入口类为Gemma4ForConditionalGeneration
