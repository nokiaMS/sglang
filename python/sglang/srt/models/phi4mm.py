# Phi-4多模态模型推理实现文件
# 本文件实现了Phi-4多模态（图像+音频）因果语言模型的推理架构
# 包含图像编码器和多模态条件生成模型等组件

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2024 SGLang Team
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
# Adapted from
# https://github.com/vllm-project/vllm/blob/6071e989df1531b59ef35568f83f7351afb0b51e/vllm/model_executor/models/phi4mm.py
# https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py

import logging  # 导入日志模块
import math  # 导入数学模块
import re  # 导入正则表达式模块
from collections.abc import Iterable  # 导入可迭代类型
from typing import List, Optional, Tuple  # 导入类型提示

import numpy as np  # 导入NumPy
import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.idefics2 import Idefics2VisionTransformer  # 导入Idefics2视觉Transformer
from sglang.srt.models.llama import LlamaForCausalLM  # 导入Llama因果语言模型
from sglang.srt.models.phi4mm_audio import AudioEmbedding  # 导入音频嵌入模块

logger = logging.getLogger(__name__)  # 获取日志记录器

SIGLIP_NAME = "siglip-so400m-patch14-448"  # SigLIP模型名称
VISION_ENCODER_TO_PROCESSING_CONFIG = {  # 视觉编码器到处理配置的映射
    "siglip-so400m-patch14-448": {  # SigLIP配置
        "vit_image_size": 448,  # 视觉Transformer图像大小
        "vit_patch_size": 14,  # 补丁大小
        "token_compression_factor": 2,  # token压缩因子
    },
}


class Phi4MMImageEncoder(nn.Module):  # Phi-4多模态图像编码器模块
    """Image embedding."""  # 图像嵌入

    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig],  # 量化配置
        prefix: str = "",  # 前缀
        model_dir: str = "",  # 模型目录
    ) -> None:
        super().__init__()  # 调用父类初始化

        # n_embed or hidden_size  # n_embed或hidden_size
        hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size  # 获取隐藏层大小
        self.type_feature = "patch"  # 特征类型为补丁
        self.img_processor = Idefics2VisionTransformer(  # 图像处理器，使用Idefics2视觉Transformer
            config=config.vision_config, require_post_norm=False  # 不需要后归一化
        )

        pe_weight = self.img_processor.embeddings.position_embedding.weight  # 位置嵌入权重
        L, D = pe_weight.size()  # 获取位置嵌入大小
        H = int(math.sqrt(L))  # 计算高度
        assert H**2 == L, f"position embedding size {L} is not square"  # 断言位置嵌入为方形
        if H % 2 != 0:  # 如果高度不是偶数
            self.img_processor_padding = nn.ReflectionPad2d((0, 1, 0, 1))  # 添加反射填充
            H += 1  # 高度加1
        image_dim_out = D  # 图像输出维度
        # ((448/14)//2)**2  # 计算图像token数
        self.num_img_tokens = (H // 2) ** 2  # 图像token数量
        self.base_feat_height_target = H  # 基础特征高度目标

        self.image_dim_out = image_dim_out  # 保存图像输出维度
        self.img_sizes = None  # 图像大小
        self.image_attention_mask = None  # 图像注意力掩码

        # global_gn and sub_gn for hd transform, serves as line separator  # HD变换的全局和子分隔符
        self.use_hd_transform = True  # 使用HD变换
        self.with_learnable_separator = True  # 使用可学习分隔符
        self.hd_transform_order = "sub_glb"  # HD变换顺序
        self.freeze_img_processor = False  # 不冻结图像处理器
        self.crop_size = 448  # 裁剪大小

        # image token compression  # 图像token压缩
        self.image_token_compression_cls = "avg_pool_2d"  # 压缩方式为平均池化
        self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化层
        self.base_feat_height_reduction = 1  # 基础特征高度缩减
        self.base_feat_height_target = self.base_feat_height_target // 2  # 目标高度减半

        # with_hd_transform and with_learnable_separator should have same value  # HD变换和可学习分隔符应相同
        assert (  # 断言
            self.use_hd_transform == self.with_learnable_separator
        ), "use_hd_transform and with_learnable_separator should have same value"  # 两者应相同
        assert self.use_hd_transform, "learnable separator is only for hd transform"  # 可学习分隔符仅用于HD变换
        # 1024 * 4, merge spatial to channel dimension  # 1024*4，将空间维度合并到通道维度
        self.glb_GN = nn.Parameter(  # 全局分隔符参数
            torch.zeros([1, 1, self.image_dim_out * self.base_feat_height_reduction**2])  # 全局分隔符
        )
        self.sub_GN = nn.Parameter(  # 子分隔符参数
            torch.zeros(
                [1, 1, 1, self.image_dim_out * self.base_feat_height_reduction**2]  # 子分隔符
            )
        )

        dim_projection = hidden_size  # 投影维度
        depth = 2  # 深度
        layers = [  # 投影层列表
            nn.Linear(
                image_dim_out * self.base_feat_height_reduction**2, dim_projection  # 输入到投影维度
            )
        ]
        for _ in range(1, depth):  # 遍历深度
            layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])  # 添加GELU和线性层
        self.img_projection = nn.Sequential(*layers)  # 图像投影层

        self.vocab_size = config.vocab_size  # 词表大小
        self.img_features = None  # 图像特征

        self.use_out_place_operations = False  # 不使用外部操作

    def get_img_features(  # 获取图像特征函数
        self, img_embeds: torch.FloatTensor, attention_mask=None  # 图像嵌入和注意力掩码
    ) -> torch.FloatTensor:
        img_feature = self.img_processor(  # 通过图像处理器
            img_embeds, patch_attention_mask=attention_mask  # 传入图像嵌入和补丁注意力掩码
        )

        patch_feature = img_feature  # 补丁特征

        use_token_compression = self.image_token_compression is not None  # 是否使用token压缩
        use_padding = getattr(self, "img_processor_padding", None) is not None  # 是否使用填充
        if use_token_compression or use_padding:  # 如果使用压缩或填充
            # reshape to 2D tensor  # 重塑为2D张量
            width = int(math.sqrt(patch_feature.size(1)))  # 计算宽度
            patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))  # 重塑
            # convert to NCHW  # 转换为NCHW格式
            patch_feature = patch_feature.permute(0, 3, 1, 2)  # 排列

            if use_padding:  # 如果使用填充
                patch_feature = self.img_processor_padding(patch_feature)  # 应用填充
            if use_token_compression:  # 如果使用压缩
                patch_feature = self.image_token_compression(patch_feature)  # 应用压缩

            # convert to NHWC  # 转换为NHWC格式
            patch_feature = patch_feature.permute(0, 2, 3, 1)  # 排列
            patch_feature = patch_feature.view(  # 重塑
                -1,  # 批次维度
                patch_feature.size(1) * patch_feature.size(2),  # 空间维度
                patch_feature.size(-1),  # 特征维度
            )

        return patch_feature  # 返回补丁特征

    def forward(  # 前向传播函数，处理图像并返回视觉嵌入
        self,
        pixel_values: torch.FloatTensor,  # 像素值
        image_sizes: torch.Tensor,  # 图像大小
        image_attention_mask: torch.Tensor,  # 图像注意力掩码
    ) -> list[torch.FloatTensor]:
        """
        process image and return vision embeddings.  # 处理图像并返回视觉嵌入

        pixel_values: (num_images, num_crops, c, h, w)  # 像素值
        image_sizes: [[h1, w1], [h2, w2]]  # 图像大小
        image_attention_mask: num_images x num_crops x 32 x 32  # 图像注意力掩码
        output: (num_images, num_img_tokens, hidden_size)  # 输出
        """

        # eg  # 示例
        # pixel_values: torch.Size([1, 7, 3, 448, 448])  # 像素值示例
        # image_sizes: tensor([[ 896, 1344]], device='cuda:0')  # 图像大小示例
        # output: torch.Size([1, 1841, 3072])  # 输出示例

        img_projection_params = next(self.img_projection.parameters())  # 获取投影层参数
        target_device = img_projection_params.device  # 目标设备
        target_dtype = img_projection_params.dtype  # 目标数据类型

        img_sizes = image_sizes  # 图像大小
        num_images, num_crops, c, h, w = pixel_values.shape  # 获取形状
        bs = num_images  # 批次大小
        pixel_values = pixel_values.flatten(0, 1)  # 展平批次和裁剪维度

        img_features = self.get_img_features(  # 获取图像特征
            pixel_values,  # 像素值
            image_attention_mask.type(torch.BoolTensor).flatten(0, 1).to(target_device),  # 注意力掩码
        )

        base_feat_height_target = self.base_feat_height_target  # 基础特征高度目标
        base_resolution = self.crop_size  # 基础分辨率
        base_feat_height_reduction = self.base_feat_height_reduction  # 特征高度缩减

        base_feat_height = base_feat_width = int(np.sqrt(img_features.shape[1]))  # 计算特征高度和宽度
        assert (  # 断言特征尺寸匹配
            base_feat_height == base_feat_height_target
            and base_feat_width == base_feat_height_target
        ), f'base_feat_height: {base_feat_height},"\
                f" base_feat_width: {base_feat_width}, "\
                f"expect {base_feat_height_target} features for hd transform'  # 特征尺寸不匹配

        # bs x max_num_crops x (24x24) x C  # 批次x最大裁剪数x补丁数x通道
        img_features = img_features.view(  # 重塑图像特征
            bs, -1, base_feat_height * base_feat_width, self.image_dim_out  # 形状
        )
        C = self.image_dim_out  # 通道数
        H = base_feat_height  # 特征高度

        output_imgs = []  # 输出图像列表
        output_len = []  # 输出长度列表
        # training is tensor, inference is list  # 训练时是张量，推理时是列表
        if isinstance(img_sizes, torch.Tensor):  # 如果图像大小是张量
            img_sizes = img_sizes.view(-1, 2)  # 重塑
        for _bs in range(bs):  # 遍历批次
            h, w = img_sizes[_bs]  # 获取高度和宽度
            h = h // base_resolution  # 计算裁剪高度
            w = w // base_resolution  # 计算裁剪宽度
            B_ = h * w  # 裁剪数

            # 1 x (24x24) x 1024  # 全局图像特征
            global_img_feature = img_features[_bs, :1]  # 获取全局特征

            # 1 x 12 x 12 x 4096  # 全局图像重塑
            glb_img = (  # 全局图像处理
                global_img_feature.reshape(1, H, H, C)  # 重塑
                .reshape(  # 进一步重塑
                    1,  # 批次
                    H // base_feat_height_reduction,  # 高度缩减
                    base_feat_height_reduction,  # 缩减高度
                    H // base_feat_height_reduction,  # 宽度缩减
                    base_feat_height_reduction,  # 缩减宽度
                    C,  # 通道
                )
                .contiguous()  # 连续化
                .permute(0, 1, 3, 2, 4, 5)  # 排列
                .reshape(  # 重塑
                    1,  # 批次
                    H // base_feat_height_reduction,  # 高度
                    H // base_feat_height_reduction,  # 宽度
                    base_feat_height_reduction * base_feat_height_reduction * C,  # 通道
                )
                .contiguous()  # 连续化
            )
            temp_glb_GN = self.sub_GN.repeat(1, H // base_feat_height_reduction, 1, 1)  # 重复全局分隔符

            # 1 x 156 x 4096  # 拼接全局图像和分隔符
            glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(  # 拼接并重塑
                1, -1, base_feat_height_reduction * base_feat_height_reduction * C  # 形状
            )

            # (max_num_crops-1) x (12x12) x C  # 子图像特征
            sub_img = img_features[_bs, 1:]  # 获取子图像特征
            # 16x574x1024  # 子图像形状
            # get rid of padding sub_img  # 移除填充的子图像
            sub_img = sub_img[:B_]  # 截取有效子图像

            # (num_crops, 12, 2, 12, 2, 1024) ->  # 子图像重塑
            # (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)  # 子图像重塑
            sub_img = (  # 子图像处理
                sub_img.reshape(B_, H, H, C)  # 重塑
                .reshape(  # 进一步重塑
                    B_,  # 裁剪数
                    H // base_feat_height_reduction,  # 高度缩减
                    base_feat_height_reduction,  # 缩减高度
                    H // base_feat_height_reduction,  # 宽度缩减
                    base_feat_height_reduction,  # 缩减宽度
                    C,  # 通道
                )
                .contiguous()  # 连续化
                .permute(0, 1, 3, 2, 4, 5)  # 排列
                .reshape(
                    B_, -1, base_feat_height_reduction * base_feat_height_reduction * C  # 重塑
                )
                .contiguous()  # 连续化
            )
            sub_img = (  # 子图像重塑
                sub_img.reshape(
                    1,  # 批次
                    h,  # 高度
                    w,  # 宽度
                    base_feat_height // base_feat_height_reduction,  # 特征高度缩减
                    base_feat_width // base_feat_height_reduction,  # 特征宽度缩减
                    -1,  # 通道
                )
                .permute(0, 1, 3, 2, 4, 5)  # 排列
                .reshape(
                    1,  # 批次
                    h * base_feat_height // base_feat_height_reduction,  # 高度
                    w * base_feat_width // base_feat_height_reduction,  # 宽度
                    base_feat_height_reduction * base_feat_height_reduction * C,  # 通道
                )
            )

            if image_attention_mask is not None and len(image_attention_mask) > 0:  # 如果有注意力掩码
                reshaped_image_attention_mask = (  # 重塑注意力掩码
                    image_attention_mask[_bs, 1 : B_ + 1, 0::2, 0::2]  # 下采样掩码
                    .reshape(
                        1,  # 批次
                        h,  # 高度
                        w,  # 宽度
                        base_feat_height // base_feat_height_reduction,  # 特征高度缩减
                        base_feat_width // base_feat_height_reduction,  # 特征宽度缩减
                    )
                    .permute(0, 1, 3, 2, 4)  # 排列
                    .reshape(
                        1,  # 批次
                        h * base_feat_height // base_feat_height_reduction,  # 高度
                        w * base_feat_width // base_feat_height_reduction,  # 宽度
                    )
                )
                useful_height = int(reshaped_image_attention_mask[0, :, 0].sum().item())  # 有效高度
                useful_width = int(reshaped_image_attention_mask[0, 0, :].sum().item())  # 有效宽度
                sub_img = sub_img[:, :useful_height, :useful_width]  # 截取有效区域
                temp_sub_GN = self.sub_GN.repeat(1, useful_height, 1, 1)  # 重复子分隔符
                temp_len = (  # 计算临时长度
                    int(image_attention_mask[_bs, : B_ + 1, 0::2, 0::2].sum().item())  # 有效token数
                    + (useful_height + 1)  # 加上分隔符行
                    + base_feat_height // base_feat_height_reduction  # 加上全局行
                )
            else:  # 否则
                temp_sub_GN = self.sub_GN.repeat(  # 重复子分隔符
                    1, h * base_feat_height // base_feat_height_reduction, 1, 1  # 形状
                )
                temp_len = int(  # 计算临时长度
                    (h * w + 1) * self.num_img_tokens  # 图像token数
                    + 1  # 分隔符
                    + (h + 1) * base_feat_height // base_feat_height_reduction  # 行分隔符
                )

            sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(  # 拼接子图像和分隔符并重塑
                1, -1, base_feat_height_reduction * base_feat_height_reduction * C  # 形状
            )
            # (1, num_img_tokens, 1024*4)  # 子图像token形状

            # glb + sub  # 全局+子图像
            if self.hd_transform_order == "glb_sub":  # 如果变换顺序是全局在前
                output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))  # 全局+分隔+子
            elif self.hd_transform_order == "sub_glb":  # 如果变换顺序是子在前
                output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))  # 子+分隔+全局
            else:  # 否则
                raise NotImplementedError(  # 抛出未实现错误
                    f'hd_transform_order = {self.hd_transform_order}, "\
                        "not implemented'  # 未实现的变换顺序
                )

            # temp_len = int((h*w+1)*144 + 1 + (h+1)*12)  # 临时长度计算
            assert (  # 断言长度匹配
                temp_len == output_imgs[-1].shape[1]
            ), f'temp_len: {temp_len}, output_imgs[-1].shape[1]: "\
                    "{output_imgs[-1].shape[1]}'  # 长度不匹配

            output_len.append(temp_len)  # 添加长度

        img_set_tensor = []  # 图像投影结果列表
        for _output_img in output_imgs:  # 遍历输出图像
            img_feature_proj = self.img_projection(  # 通过图像投影层
                _output_img.to(target_device).to(target_dtype)  # 转换设备和数据类型
            )
            img_set_tensor.append(img_feature_proj.squeeze(0))  # 添加到列表

        return img_set_tensor  # 返回图像投影结果


class Phi4MMForCausalLM(nn.Module):  # Phi-4多模态因果语言模型
    packed_modules_mapping = {  # 打包模块映射
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],  # QKV投影
        "gate_up_proj": ["gate_proj", "up_proj"],  # 门控上投影
    }

    lora_pattern = re.compile(  # LoRA正则模式
        r"^language_model\.model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)"  # 匹配语言模型层中的LoRA
    )

    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化

        self.language_model = LlamaForCausalLM(  # 语言模型，基于Llama
            config=config, quant_config=quant_config, prefix=prefix  # 传入配置
        )

        self.vision_encoder = Phi4MMImageEncoder(  # 视觉编码器
            config,  # 配置
            quant_config,  # 量化配置
            prefix="model.vision_embed_tokens",  # 前缀
            model_dir=config._name_or_path,  # 模型目录
        )

        if isinstance(config.embd_layer["audio_embd_layer"], dict):  # 如果音频嵌入层配置是字典
            embedding_config = {  # 嵌入配置
                "embedding_cls": config.embd_layer["audio_embd_layer"]["embedding_cls"],  # 嵌入类
                **config.embd_layer["audio_embd_layer"],  # 其他配置
            }
        else:  # 否则
            embedding_config = {"embedding_cls": config.embd_layer["embedding_cls"]}  # 仅嵌入类

        self.embed_tokens_extend = AudioEmbedding(config, **embedding_config)  # 音频嵌入扩展

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 获取图像特征函数
        dtype = next(self.vision_encoder.parameters()).dtype  # 获取数据类型
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(dtype)  # 拼接像素值
        image_attention_mask = torch.cat(  # 拼接注意力掩码
            [
                item.image_attention_mask  # 图像注意力掩码
                for item in items  # 遍历数据项
                if hasattr(item, "image_attention_mask")  # 如果有注意力掩码
            ],
            dim=0,  # 第0维拼接
        )
        image_sizes = torch.cat([item.image_sizes for item in items], dim=0)  # 拼接图像大小
        image_embeds = self.vision_encoder(  # 通过视觉编码器
            pixel_values, image_sizes, image_attention_mask  # 传入参数
        )
        return torch.cat(image_embeds).type(dtype)  # 拼接并转换数据类型

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 获取音频特征函数
        # (e.g. multiple examples) and the second dim is the multi-audio dim  # 第二维是多音频维度
        # (e.g. multiple audios in the same example)  # 同一样本中的多个音频
        embed_tokens_extend_param = next(self.embed_tokens_extend.parameters())  # 获取参数
        device = embed_tokens_extend_param.device  # 设备
        dtype = embed_tokens_extend_param.dtype  # 数据类型
        audio_embeds = [  # 音频嵌入列表
            self.embed_tokens_extend(  # 通过音频嵌入
                # item.feature: (num_audios_in_a_sequence, T, D)  # 音频特征形状
                # item.audio_attention_mask: (num_audios_in_a_sequence, T, D) BoolTensor or None  # 音频注意力掩码
                audio_features=item.feature.type(dtype),  # 音频特征
                audio_attention_mask=(  # 音频注意力掩码
                    item.audio_attention_mask.to(device)  # 转换设备
                    if hasattr(item, "audio_attention_mask")  # 如果有注意力掩码
                    else None  # 否则为空
                ),
            )
            for item in items  # 遍历数据项
        ]
        return torch.cat(audio_embeds).type(dtype)  # 拼接并转换数据类型

    def forward(  # 前向传播函数，执行多模态因果语言模型计算
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次信息
        **kwargs: object,  # 其他关键字参数
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(  # 通过通用多模态嵌入例程
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.language_model,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数
                Modality.IMAGE: self.get_image_feature,  # 图像特征
                Modality.AUDIO: self.get_audio_feature,  # 音频特征
            },
            positions=positions,  # 位置
        )

        return hidden_states  # 返回隐藏状态

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):  # 填充输入ID函数
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 填充输入token

    def should_apply_lora(self, module_name: str) -> bool:  # 是否应该应用LoRA函数
        return bool(self.lora_pattern.match(module_name))  # 检查模块名是否匹配LoRA模式

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重函数
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),  # Q投影映射
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),  # K投影映射
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),  # V投影映射
        ]
        prefix_mapping = {  # 前缀映射
            "model.embed_tokens_extend.audio_embed.audio_projection.vision.": "embed_tokens_extend.audio_projection_for_vision.",  # 视觉音频投影
            "model.embed_tokens_extend.audio_embed.audio_projection.speech.": "embed_tokens_extend.audio_projection.",  # 语音音频投影
            "model.embed_tokens_extend.audio_embed.": "embed_tokens_extend.",  # 音频嵌入
            "model.embed_tokens_extend.image_embed.": "vision_encoder.",  # 图像嵌入到视觉编码器
            "model.": "language_model.model.",  # 模型到语言模型
        }

        skip_list = [  # 跳过列表
            "img_processor.encoder.layers.26",  # 跳过第26层
            "img_processor.head",  # 跳过头部
            "img_processor.post_layernorm",  # 跳过后层归一化
        ]

        def _should_skip(name: str) -> bool:  # 判断是否跳过权重
            return any(substr in name for substr in skip_list)  # 如果名称包含跳过列表中的子串

        params_dict = dict(self.named_parameters())  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历权重
            # Skip the last layer  # 跳过最后一层
            if _should_skip(name):  # 如果应该跳过
                continue  # 跳过

            for old_name, new_name in prefix_mapping.items():  # 遍历前缀映射
                if name.startswith(old_name):  # 如果名称以旧前缀开头
                    name = name.replace(old_name, new_name)  # 替换前缀
                    break  # 跳出循环

            # Adapt to VisionAttention  # 适配视觉注意力
            name = name.replace(r"self_attn.out_proj", r"self_attn.proj")  # 替换输出投影名称
            name = name.replace(r"base_layer.", r"")  # 移除base_layer前缀

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue  # 继续
                name = name.replace(weight_name, param_name)  # 替换权重名
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出循环
            else:  # 如果没有匹配的堆叠参数
                param = params_dict.get(name)  # 获取参数
                if param is None:  # 如果参数为空
                    if "lora" not in name:  # 如果不是LoRA权重
                        logger.warning(f"Warning: {name} not found in model parameters")  # 记录警告
                    continue  # 跳过
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = [Phi4MMForCausalLM]  # 入口类列表
