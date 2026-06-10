# GLM-OCR多模态OCR模型推理实现
# 本文件实现了GLM-OCR多模态模型的推理逻辑，包含视觉编码器、
# 视觉块嵌入、视觉模型、条件生成模型等核心组件，
# 支持HuggingFace权重加载和张量并行。

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

# Modeling from:  # 建模参考来源：
# ./llama.py and  # llama.py 和
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/GlmOcr/modular_GlmOcr.py  # HuggingFace的GlmOcr模块化实现
"""Inference-only GLM-OCR model compatible with HuggingFace weights."""  # 仅推理的GLM-OCR模型，兼容HuggingFace权重

import logging  # 导入日志模块
from functools import lru_cache  # 导入LRU缓存装饰器
from typing import Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
from einops import rearrange  # 导入张量重排工具
from transformers.models.glm_ocr.configuration_glm_ocr import (  # 导入GLM-OCR配置
    GlmOcrConfig,  # GLM-OCR总配置
    GlmOcrTextConfig,  # GLM-OCR文本配置
    GlmOcrVisionConfig,  # GLM-OCR视觉配置
)

from sglang.srt.distributed.parallel_state import get_pp_group  # 导入获取流水线并行组的函数
from sglang.srt.layers.attention import vision_utils  # 导入视觉注意力工具
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.pooler import Pooler, PoolingType  # 导入池化层和池化类型
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.layers.utils import PPMissingLayer  # 导入流水线并行缺失层
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行语言模型头
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.glm4 import Glm4Model  # 导入GLM4模型
from sglang.srt.models.glm4v import (  # 导入GLM4V视觉模型组件
    Glm4vForConditionalGeneration,  # GLM4V条件生成模型
    Glm4vPatchMerger,  # GLM4V补丁合并器
    Glm4vRMSNorm,  # GLM4V的RMS归一化
    Glm4vVisionMLP,  # GLM4V视觉MLP
    Glm4vVisionModel,  # GLM4V视觉模型
    Glm4vVisionPatchEmbed,  # GLM4V视觉补丁嵌入
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数
from sglang.srt.utils import add_prefix  # 导入前缀添加工具函数
from sglang.srt.utils.hf_transformers_utils import get_processor  # 导入HuggingFace处理器获取函数

logger = logging.getLogger(__name__)  # 创建日志记录器

cached_get_processor = lru_cache(get_processor)  # 带LRU缓存的处理器获取函数


class GlmOcrRMSNorm(Glm4vRMSNorm):
    """GLM-OCR的RMS归一化层，继承自Glm4vRMSNorm。"""
    pass  # 直接继承，不做额外修改


class GlmOcrVisionMLP(Glm4vVisionMLP):
    """GLM-OCR视觉MLP，继承自Glm4vVisionMLP。"""
    pass  # 直接继承，不做额外修改


class GlmOcrVisionBlock(nn.Module):
    """GLM-OCR视觉编码器块，包含自注意力和MLP。"""

    def __init__(  # 初始化方法
        self,
        dim: int,  # 隐藏维度
        intermediate_dim: int,  # 中间层维度
        num_heads: int,  # 注意力头数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
        attn_qkv_bias: bool = True,  # 注意力QKV是否使用偏置
        num_dummy_heads: int = 0,  # 虚拟头数量
        rms_norm_eps: float = 1e-5,  # RMS归一化epsilon
        use_data_parallel: bool = False,  # 是否使用数据并行
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)  # 第一个RMS归一化层（注意力前）
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)  # 第二个RMS归一化层（MLP前，带残差融合）
        self.attn = VisionAttention(  # 视觉注意力层
            embed_dim=dim,  # 嵌入维度
            num_heads=num_heads,  # 注意力头数
            projection_size=dim,  # 投影大小
            use_qkv_parallel=True,  # 使用QKV并行
            qkv_bias=attn_qkv_bias,  # QKV偏置
            proj_bias=True,  # 输出投影偏置
            qk_normalization_by_head_size=True,  # 按头大小归一化QK
            flatten_batch=True,  # 展平批次
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 添加前缀
            num_dummy_heads=num_dummy_heads,  # 虚拟头数量
            use_data_parallel=use_data_parallel,  # 数据并行
        )
        self.mlp = GlmOcrVisionMLP(  # 视觉MLP
            dim,  # 隐藏维度
            intermediate_dim,  # 中间维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 添加前缀
            use_data_parallel=use_data_parallel,  # 数据并行
        )

    def forward(  # 前向传播方法
        self,
        x: torch.Tensor,  # 输入张量，形状为[S, B, H]
        cu_seqlens: torch.Tensor,  # 累积序列长度
        rotary_pos_emb_cos: torch.Tensor,  # 旋转位置编码cos值
        rotary_pos_emb_sin: torch.Tensor,  # 旋转位置编码sin值
    ) -> torch.Tensor:
        S, B, H = x.shape  # 获取序列长度S、批次大小B、隐藏维度H
        # norm1: flatten to 2D -> [S*B, H], then reshape back  # norm1：展平为2D -> [S*B, H]，然后恢复形状
        x2d = x.reshape(-1, H)  # 展平为2D
        hidden_states = self.norm1(x2d).reshape(S, B, H)  # 归一化后恢复形状

        # Attention expects [B, S, H]  # 注意力期望[B, S, H]的输入形状
        hidden_states = rearrange(hidden_states, "s b h -> b s h")  # 重排为[B, S, H]
        attn = self.attn(  # 通过注意力层
            hidden_states,  # 隐藏状态
            cu_seqlens=cu_seqlens,  # 累积序列长度
            rotary_pos_emb_cos=rotary_pos_emb_cos,  # 旋转位置编码cos
            rotary_pos_emb_sin=rotary_pos_emb_sin,  # 旋转位置编码sin
        )
        attn = rearrange(attn, "b s h -> s b h")  # 重排回[S, B, H]

        # norm2 with fused residual-add: also 2D  # norm2带融合残差加法：也是2D
        attn2d = attn.reshape(-1, H)  # 注意力输出展平为2D
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)  # 带残差的归一化
        x_norm = x_norm_2d.reshape(S, B, H)  # 归一化结果恢复形状
        x_after_add = x_after_add_2d.reshape(S, B, H)  # 残差加法结果恢复形状

        # MLP and final residual  # MLP和最终残差连接
        mlp_out = self.mlp(x_norm)  # 通过MLP
        x = x_after_add + mlp_out  # 残差连接
        return x  # 返回输出


class GlmOcrVisionPatchEmbed(Glm4vVisionPatchEmbed):
    """GLM-OCR视觉补丁嵌入，继承自Glm4vVisionPatchEmbed。"""
    pass  # 直接继承，不做额外修改


class GlmOcrVisionPatchMerger(Glm4vPatchMerger):
    """GLM-OCR视觉补丁合并器，继承自Glm4vPatchMerger。"""
    pass  # 直接继承，不做额外修改


class GlmOcrVisionModel(Glm4vVisionModel):
    """GLM-OCR视觉模型，继承自Glm4vVisionModel，实现视觉编码器的前向传播。"""

    def __init__(  # 初始化方法
        self,
        vision_config: GlmOcrVisionConfig,  # 视觉配置
        text_config: GlmOcrTextConfig,  # 文本配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
        use_data_parallel: bool = False,  # 是否使用数据并行
    ) -> None:
        super().__init__(vision_config, quant_config, prefix, use_data_parallel)  # 调用父类初始化

        patch_size = vision_config.patch_size  # 补丁大小
        temporal_patch_size = vision_config.temporal_patch_size  # 时间补丁大小
        in_channels = vision_config.in_channels  # 输入通道数
        depth = vision_config.depth  # 编码器层数（深度）
        self.hidden_size = vision_config.hidden_size  # 隐藏层大小
        self.num_heads = vision_config.num_heads  # 注意力头数

        self.patch_size = vision_config.patch_size  # 补丁大小
        self.spatial_merge_size = vision_config.spatial_merge_size  # 空间合并大小
        self.out_hidden_size = vision_config.out_hidden_size  # 输出隐藏层大小
        self.intermediate_size = vision_config.intermediate_size  # 中间层大小
        self.use_data_parallel = use_data_parallel  # 是否使用数据并行

        self.patch_embed = GlmOcrVisionPatchEmbed(  # 补丁嵌入层
            patch_size=patch_size,  # 补丁大小
            temporal_patch_size=temporal_patch_size,  # 时间补丁大小
            in_channels=in_channels,  # 输入通道数
            hidden_size=self.hidden_size,  # 隐藏层大小
        )

        head_dim = self.hidden_size // self.num_heads  # 每个头的维度
        self.rotary_pos_emb = get_rope(  # 旋转位置编码
            head_size=head_dim,  # 头维度
            rotary_dim=head_dim // 2,  # 旋转维度为头维度的一半
            max_position=8192,  # 最大位置编码长度
            base=10000.0,  # 旋转位置编码基数
            is_neox_style=True,  # 使用NeoX风格
        )

        self.blocks = nn.ModuleList(  # 视觉编码器块列表
            [
                GlmOcrVisionBlock(  # 视觉编码器块
                    dim=self.hidden_size,  # 隐藏维度
                    intermediate_dim=self.intermediate_size,  # 中间维度
                    num_heads=self.num_heads,  # 注意力头数
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),  # 添加前缀
                    rms_norm_eps=vision_config.rms_norm_eps,  # RMS归一化epsilon
                    attn_qkv_bias=vision_config.attention_bias,  # 注意力偏置
                    use_data_parallel=use_data_parallel,  # 数据并行
                )
                for layer_idx in range(depth)  # 遍历每一层
            ]
        )
        self.merger = GlmOcrVisionPatchMerger(  # 补丁合并器
            d_model=vision_config.out_hidden_size,  # 模型维度
            context_dim=text_config.intermediate_size,  # 上下文维度（文本中间层大小）
            quant_config=quant_config,  # 量化配置
            bias=False,  # 不使用偏置
            prefix=add_prefix("merger", prefix),  # 添加前缀
            use_data_parallel=use_data_parallel,  # 数据并行
        )

        self.downsample = nn.Conv2d(  # 下采样卷积层
            in_channels=vision_config.hidden_size,  # 输入通道数
            out_channels=vision_config.out_hidden_size,  # 输出通道数
            kernel_size=vision_config.spatial_merge_size,  # 卷积核大小（空间合并大小）
            stride=vision_config.spatial_merge_size,  # 步幅（空间合并大小）
        )
        self.post_layernorm = GlmOcrRMSNorm(  # 后层归一化
            vision_config.hidden_size, eps=vision_config.rms_norm_eps  # 隐藏大小和epsilon
        )

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        # patchify  # 补丁化
        x = x.to(device=self.device, dtype=self.dtype)  # 将输入转移到对应设备和数据类型
        x = self.patch_embed(x)  # 通过补丁嵌入层

        # compute position embedding  # 计算位置编码
        rotary_pos_emb_cos, rotary_pos_emb_sin, image_type_ids = self.rot_pos_emb(  # 获取旋转位置编码
            grid_thw  # 网格时间-高度-宽度信息
        )
        # compute cu_seqlens  # 计算累积序列长度
        cu_seqlens = torch.repeat_interleave(  # 重复插值计算每个图像的token数
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]  # 高度*宽度，按时间维度重复
        ).cumsum(dim=0, dtype=torch.int32)  # 累积求和
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])  # 在开头添加0

        rotary_pos_emb_cos = torch.cat([rotary_pos_emb_cos, rotary_pos_emb_cos], dim=-1)  # 拼接cos位置编码
        rotary_pos_emb_sin = torch.cat([rotary_pos_emb_sin, rotary_pos_emb_sin], dim=-1)  # 拼接sin位置编码

        # x.shape: (s, b, d) where b=1 for vision processing  # x形状：(s, b, d)，视觉处理中b=1
        # transformers  # Transformer编码器
        x = x.unsqueeze(1)  # 在第1维添加维度，变为(s, 1, d)
        for blk in self.blocks:  # 遍历每个编码器块
            x = blk(  # 通过编码器块
                x,  # 输入
                cu_seqlens=cu_seqlens,  # 累积序列长度
                rotary_pos_emb_cos=rotary_pos_emb_cos,  # 旋转位置编码cos
                rotary_pos_emb_sin=rotary_pos_emb_sin,  # 旋转位置编码sin
            )

        # adapter  # 适配器（后处理）
        x = self.post_layernorm(x)  # 后层归一化
        x = x.view(-1, self.spatial_merge_size, self.spatial_merge_size, x.shape[-1])  # 重塑为空间维度
        x = x.permute(0, 3, 1, 2)  # 置换维度为[batch, channels, height, width]
        x = self.downsample(x).view(-1, self.out_hidden_size)  # 下采样并展平
        x = self.merger(x)  # 通过补丁合并器

        return x  # 返回处理后的视觉特征


class GlmOcrForConditionalGeneration(Glm4vForConditionalGeneration):
    """GLM-OCR条件生成模型，继承自Glm4vForConditionalGeneration，
    整合视觉编码器和文本语言模型进行多模态推理。"""

    def __init__(  # 初始化方法
        self,
        config: GlmOcrConfig,  # GLM-OCR配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__(config, quant_config, prefix)  # 调用父类初始化

        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.config = config  # 保存配置
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder  # 是否对多模态编码器启用数据并行
        self.visual = GlmOcrVisionModel(  # 视觉编码器模型
            vision_config=config.vision_config,  # 视觉配置
            text_config=config.text_config,  # 文本配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("visual", prefix),  # 添加前缀
            use_data_parallel=self.use_data_parallel,  # 数据并行
        )

        vision_utils.update_vit_attn_dummy_heads_config(self.config)  # 更新ViT注意力虚拟头配置

        self.model = Glm4Model(  # GLM4文本模型
            config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("model", prefix),  # 添加前缀
        )

        if self.pp_group.is_last_rank:  # 如果是流水线并行的最后一个rank
            if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:  # 单卡且词嵌入共享
                self.lm_head = self.model.embed_tokens  # 语言模型头共享词嵌入
            else:  # 否则
                self.lm_head = ParallelLMHead(  # 创建并行语言模型头
                    self.config.vocab_size,  # 词表大小
                    self.config.hidden_size,  # 隐藏层大小
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix("lm_head", prefix),  # 添加前缀
                )
        else:  # 否则
            # ranks other than the last rank will have a placeholder layer  # 非最后一个rank将有一个占位层
            self.lm_head = PPMissingLayer()  # 流水线并行缺失层占位

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling  # 是否启用多维度旋转位置编码

        self.logits_processor = LogitsProcessor(config)  # logits处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 池化层，取最后一个token并归一化

        # For EAGLE3 support  # 用于EAGLE3推测解码支持
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):  # 加载权重方法
        if is_nextn:  # 如果是Next-N推测解码模式
            if hasattr(self.config, "num_nextn_predict_layers"):  # 如果配置中有Next-N预测层数
                num_nextn_layers = self.config.num_nextn_predict_layers  # 获取Next-N预测层数
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"  # 仅支持1个Next-N层
                # compatible with old design  # 兼容旧设计
                nextn_layer_id = (  # Next-N层ID
                    0  # 如果只有1个隐藏层，则为0
                    if self.config.num_hidden_layers == 1  # 单层情况
                    else self.config.num_hidden_layers  # 否则为最后一个层
                )
            else:  # 否则
                raise ValueError("num_nextn_predict_layers is not in the config")  # 配置中没有Next-N预测层数

        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),  # Q投影映射
            (".qkv_proj", ".k_proj", "k"),  # K投影映射
            (".qkv_proj", ".v_proj", "v"),  # V投影映射
            (".gate_up_proj", ".up_proj", 1),  # up投影映射
            (".gate_up_proj", ".gate_proj", 0),  # gate投影映射
        ]

        if is_nextn:  # 如果是Next-N模式
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"  # Next-N层前缀
            nextn_spec_weight_names = [  # Next-N特定权重名称列表
                "shared_head.norm",  # 共享头归一化
                "eh_proj",  # 嵌入隐藏投影
                "enorm",  # 嵌入归一化
                "hnorm",  # 隐藏归一化
            ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))  # 获取所有参数字典

        # For the PP case, we add special handling for lm_head.weight,  # 对于流水线并行情况，对lm_head.weight做特殊处理
        # - On non–last ranks: we continue, because this stage is supposed to  # - 在非最后一个rank上：继续，因为此阶段只是空的PPMissingLayer壳
        #   be just an empty PPMissingLayer shell.  #   
        # - On the last rank: params_dict is expected to contain lm_head.weight,  # - 在最后一个rank上：params_dict应包含lm_head.weight
        #   so it will never hit the branch "if name not in params_dict".  #   所以不会进入"if name not in params_dict"分支
        #  # 
        # For all other parameters, such like  # 对于所有其他参数，如
        # "model.visual.blocks.20.mlp.gate_proj.weight", the unified rule is:  # "model.visual.blocks.20.mlp.gate_proj.weight"，统一规则是：
        # If this name does not exist in the current rank's params_dict,  # 如果此名称不在当前rank的params_dict中
        # it does not belong to this pipeline stage, thus we simply continue.  # 则不属于此流水线阶段，直接跳过

        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转位置编码的逆频率
                continue  # 跳过
            if "language_model" in name:  # 如果包含language_model
                name = name.replace(r"model.language_model.", r"model.")  # 替换为model.
            if "model.visual." in name:  # 如果包含model.visual.
                name = name.replace("model.visual.", "visual.")  # 替换为visual.

            if not is_nextn:  # 如果不是Next-N模式
                if hasattr(self.config, "num_nextn_predict_layers"):  # 如果配置中有Next-N预测层数
                    num_nextn_layers = self.config.num_nextn_predict_layers  # 获取层数
                    if num_nextn_layers > 0 and name.startswith("model.layers"):  # 如果有Next-N层且名称以model.layers开头
                        name_list = name.split(".")  # 按点分割名称
                        if (  # 如果
                            len(name_list) >= 3  # 名称至少有3层
                            and int(name_list[2]) >= self.config.num_hidden_layers  # 层索引大于等于隐藏层数
                        ):
                            continue  # 跳过Next-N层的权重
            else:  # Next-N模式
                if not name.startswith(nextn_layer_prefix):  # 如果不是Next-N层的权重
                    continue  # 跳过

                # Use shared head and embed weights from target model  # 使用目标模型的共享头和嵌入权重
                if "shared_head.head" in name or "embed_tokens" in name:  # 如果是共享头或嵌入
                    continue  # 跳过

                is_decoder = True  # 标记为解码器权重
                # For nextn specific weights  # 对于Next-N特定权重
                for weight_name in nextn_spec_weight_names:  # 遍历Next-N特定权重名
                    if weight_name in name:  # 如果名称中包含Next-N特定权重名
                        name = name.replace(nextn_layer_prefix, "model")  # 替换前缀
                        is_decoder = False  # 不是解码器权重
                        break  # 跳出循环
                # For decoder layer weights  # 对于解码器层权重
                if is_decoder:  # 如果是解码器权重
                    name = name.replace(nextn_layer_prefix, "model.decoder")  # 替换为model.decoder

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在名称中
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换权重名为参数名

                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过

                if name not in params_dict:  # 如果参数名不在参数字典中
                    continue  # 跳过

                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重分片
                break  # 跳出循环
            else:  # 如果不是堆叠参数
                if "visual" in name:  # 如果是视觉模型权重
                    # adapt to VisionAttention  # 适配VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")  # 替换qkv为qkv_proj

                try:  # 尝试
                    # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                    if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                        continue  # 跳过

                    if name not in params_dict:  # 如果参数名不在参数字典中
                        continue  # 跳过

                    param = params_dict[name]  # 获取参数
                except KeyError:  # 如果键错误
                    print(params_dict.keys())  # 打印参数字典的键
                    raise  # 重新抛出异常

                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器，默认使用default_weight_loader
                if "visual" in name:  # 如果是视觉模型权重
                    loaded_weight = vision_utils.pad_vit_attn_dummy_heads(  # 填充ViT注意力虚拟头
                        self.config, name, loaded_weight  # 传入配置、名称和权重
                    )
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = [GlmOcrForConditionalGeneration]  # 入口类列表
