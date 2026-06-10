# NVILA-Lite 多模态视觉语言模型实现
# 该文件实现了 NVILA-Lite 模型，结合 Siglip 视觉编码器和 Qwen2 语言模型，
# 支持图像和视频输入，通过多模态投影器将视觉特征映射到语言模型空间。

import math  # 导入数学库 # 导入数学模块
from collections.abc import Iterable  # 导入可迭代类型 # 导入可迭代类型
from typing import Any  # 导入任意类型 # 导入任意类型提示

import einops  # 导入张量操作库 # 导入 einops 张量重排库
import torch  # 导入 PyTorch # 导入 PyTorch 框架
import torch.nn as nn  # 导入神经网络模块 # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口 # 导入函数式接口
from torch import Tensor  # 导入张量类型 # 导入张量类型
from transformers.configuration_utils import PretrainedConfig  # 导入预训练配置基类 # 导入预训练配置基类
from transformers.modeling_outputs import BaseModelOutputWithPooling  # 导入模型输出类 # 导入带池化的模型输出类
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config  # 导入 Qwen2 配置 # 导入 Qwen2 配置类
from transformers.models.siglip import SiglipVisionConfig, SiglipVisionModel  # 导入 Siglip 视觉模型 # 导入 Siglip 视觉配置和模型

import sglang.srt.managers.mm_utils as mm_utils  # 导入多模态工具 # 导入多模态工具模块
import sglang.srt.model_loader.weight_utils as weight_utils  # 导入权重加载工具 # 导入权重加载工具
import sglang.srt.utils as utils  # 导入通用工具 # 导入通用工具模块
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # 导入逻辑斯蒂处理器输出 # 导入 logits 处理器输出类
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置 # 导入量化配置基类
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens  # 导入多模态填充模式 # 导入多模态数据填充模式
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类 # 导入调度批次相关类
    Modality,  # 模态枚举 # 模态类型枚举
    MultimodalDataItem,  # 多模态数据项 # 多模态数据项
    MultimodalInputs,  # 多模态输入 # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # 导入前向批次信息
from sglang.srt.models.qwen2 import Qwen2ForCausalLM  # 导入 Qwen2 因果语言模型 # 导入 Qwen2 因果语言模型

MM_HIDDEN_SIZE = 1152  # 多模态隐藏层大小 # 多模态隐藏层维度大小


class NVILALiteConfig(PretrainedConfig):
    """NVILA-Lite 模型配置类，继承自预训练配置基类"""
    model_type = "nvila_lite"  # 模型类型标识 # 模型类型标识符
    sub_configs = {  # 子配置映射 # 子配置字典
        "text_config": Qwen2Config,  # 文本配置使用 Qwen2 # 文本模型配置类
        "vision_config": SiglipVisionConfig,  # 视觉配置使用 Siglip # 视觉模型配置类
    }
    _auto_class = "AutoConfig"  # 自动类标识 # 自动配置类名称

    def __init__(  # 初始化方法 # 初始化方法
        self,
        *,
        text_config: dict[str, Any] | None = None,  # 文本配置字典 # 文本模型配置字典
        vision_config: dict[str, Any] | None = None,  # 视觉配置字典 # 视觉模型配置字典
        image_token_id: int | None = None,  # 图像标记 ID # 图像标记 ID
        video_token_id: int | None = None,  # 视频标记 ID # 视频标记 ID
        **kwargs,  # 其他关键字参数 # 其他关键字参数
    ):
        self.text_config = (  # 文本配置 # 初始化文本配置
            Qwen2Config(**text_config) if text_config is not None else Qwen2Config()  # 从字典创建或使用默认 # 从字典创建或使用默认
        )
        self.vision_config = (  # 视觉配置 # 初始化视觉配置
            SiglipVisionConfig(**vision_config)
            if vision_config is not None
            else SiglipVisionConfig()  # 从字典创建或使用默认 # 从字典创建或使用默认
        )

        self.image_token_id = image_token_id if image_token_id is not None else -1  # 图像标记 ID # 设置图像标记 ID
        self.video_token_id = video_token_id if video_token_id is not None else -1  # 视频标记 ID # 设置视频标记 ID

        super().__init__(**kwargs)  # 调用父类初始化 # 调用父类初始化


class NVILALiteMultiModalProjectorDownsampleBlock(nn.Module):
    """NVILA-Lite 多模态投影器下采样块，将空间维度缩小 3 倍"""

    def forward(self, x: Tensor) -> Tensor:  # 前向传播 # 前向传播方法
        batch_size, sequence_length, hidden_size = x.shape  # 获取输入形状 # 获取批次大小、序列长度和隐藏维度

        feat_size = math.isqrt(sequence_length)  # 计算特征图边长 # 计算特征图的边长（整数平方根）

        features = x.reshape(batch_size, feat_size, feat_size, hidden_size)  # 重塑为空间特征图 # 重塑为空间特征图

        pad_after = (3 - feat_size % 3) % 3  # 计算填充量使其能被 3 整除 # 计算需要的填充量
        if pad_after > 0:  # 如果需要填充 # 如果需要填充
            features = F.pad(features, (0, 0, 0, pad_after, 0, pad_after))  # 对空间维度填充 # 对空间维度进行填充
            feat_size = feat_size + pad_after  # 更新特征图边长 # 更新特征图边长

        features = features.reshape(  # 重塑特征图以进行下采样 # 重塑特征图以下采样
            batch_size, feat_size // 3, 3, feat_size // 3, 3, hidden_size  # 分成 3x3 块 # 分成 3x3 块
        )
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous()  # 重排维度以合并空间信息 # 重排维度以合并空间信息
        features = features.reshape(batch_size, -1, 9 * hidden_size)  # 展平为序列 # 展平为序列，通道维度扩大 9 倍

        return features  # 返回下采样后的特征 # 返回下采样后的特征


class NVILALiteMultiModalProjector(nn.Module):
    """NVILA-Lite 多模态投影器，将视觉特征投影到语言模型空间"""

    def __init__(self, config: NVILALiteConfig):  # 初始化方法 # 初始化方法
        super().__init__()  # 调用父类初始化 # 调用父类初始化

        self.layers = nn.Sequential(  # 投影层序列 # 顺序投影层
            NVILALiteMultiModalProjectorDownsampleBlock(),  # 下采样块 # 下采样块
            nn.LayerNorm(MM_HIDDEN_SIZE * 9),  # 层归一化 # 层归一化
            nn.Linear(MM_HIDDEN_SIZE * 9, MM_HIDDEN_SIZE * 3),  # 线性投影 # 线性投影层
            nn.GELU(),  # GELU 激活 # GELU 激活函数
            nn.LayerNorm(MM_HIDDEN_SIZE * 3),  # 层归一化 # 层归一化
            nn.Linear(MM_HIDDEN_SIZE * 3, config.text_config.hidden_size),  # 线性投影 # 投影到文本隐藏维度
            nn.GELU(),  # GELU 激活 # GELU 激活函数
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size),  # 最终线性层 # 最终线性投影
        )

    def forward(self, x: Tensor) -> Tensor:  # 前向传播 # 前向传播方法
        return self.layers(x)  # 通过投影层序列 # 通过所有投影层


class NVILALiteForConditionalGeneration(nn.Module):
    """NVILA-Lite 条件生成模型，结合视觉编码器和语言模型"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: NVILALiteConfig,  # 模型配置 # 模型配置
        quant_config: QuantizationConfig | None = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用父类初始化

        self.config = config  # 保存配置 # 保存模型配置

        self.vision_tower = SiglipVisionModel(config.vision_config)  # 视觉编码器 # 创建 Siglip 视觉编码器
        self.mm_projector = NVILALiteMultiModalProjector(config)  # 多模态投影器 # 创建多模态投影器
        self.llm = Qwen2ForCausalLM(  # 语言模型 # 创建 Qwen2 语言模型
            config=config.text_config,  # 文本配置 # 文本模型配置
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=utils.add_prefix("llm", prefix),  # 添加前缀 # 添加参数名前缀
        )

    def forward(  # 前向传播 # 前向传播方法
        self,
        input_ids: Tensor,  # 输入 ID # 输入标记 ID
        positions: Tensor,  # 位置编码 # 位置编码
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
        get_embedding: bool = False,  # 是否获取嵌入 # 是否获取嵌入
    ) -> LogitsProcessorOutput:  # 返回 logits 处理器输出 # 返回 logits 处理器输出
        output = mm_utils.general_mm_embed_routine(  # 通用多模态嵌入流程 # 调用通用多模态嵌入流程
            input_ids=input_ids,  # 输入 ID # 输入标记 ID
            forward_batch=forward_batch,  # 前向批次 # 前向批次信息
            language_model=self.llm,  # 语言模型 # 语言模型
            data_embedding_funcs={  # 数据嵌入函数映射 # 数据嵌入函数映射
                Modality.IMAGE: self.get_image_feature,  # 图像特征提取 # 图像模态使用图像特征提取
                Modality.VIDEO: self.get_image_feature,  # 视频特征提取（复用图像） # 视频模态复用图像特征提取
            },
            get_embedding=get_embedding,  # 是否获取嵌入 # 是否获取嵌入
            positions=positions,  # 位置编码 # 位置编码
        )

        assert isinstance(output, LogitsProcessorOutput)  # 断言输出类型 # 断言输出类型正确

        return output  # 返回输出 # 返回输出

    def get_image_feature(self, mm_input: list[MultimodalDataItem]) -> Tensor:  # 获取图像特征 # 提取图像视觉特征
        pixel_values = torch.cat([torch.tensor(x.feature) for x in mm_input], dim=0)  # 拼接像素值 # 拼接所有图像的像素值

        vision_tower_output: BaseModelOutputWithPooling = self.vision_tower(  # 视觉编码器前向 # 通过视觉编码器
            pixel_values,  # 像素值 # 像素值
            output_hidden_states=True,  # 输出隐藏状态 # 输出所有隐藏层状态
        )
        assert vision_tower_output.hidden_states is not None  # 断言隐藏状态不为空 # 断言隐藏状态存在

        vision_features = vision_tower_output.hidden_states[-2]  # 取倒数第二层特征 # 取倒数第二层的视觉特征

        vision_features = self.mm_projector(vision_features)  # 通过投影器 # 通过多模态投影器

        vision_features = einops.rearrange(vision_features, "n p d -> (n p) d")  # 重排维度 # 将批量和序列维度展平

        return vision_features  # 返回视觉特征 # 返回处理后的视觉特征

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> None:  # 加载权重 # 加载模型权重
        params_dict = dict(self.named_parameters())  # 获取参数字典 # 获取模型参数字典

        for name, loaded_weight in weights:  # 遍历权重 # 遍历所有权重
            if name.startswith("llm."):  # 如果是语言模型权重 # 如果是语言模型的权重
                self.llm.load_weights([(name[len("llm.") :], loaded_weight)])  # 去掉前缀后加载 # 去掉 llm. 前缀后加载
            else:  # 否则 # 其他权重
                param = params_dict[name]  # 获取参数 # 获取对应参数
                weight_loader = getattr(  # 获取权重加载器 # 获取权重加载器
                    param, "weight_loader", weight_utils.default_weight_loader  # 默认使用标准加载器 # 默认使用标准权重加载器
                )
                weight_loader(param, loaded_weight)  # 加载权重 # 加载权重

    def pad_input_ids(  # 填充输入 ID # 填充输入标记 ID
        self, input_ids: list[int], mm_inputs: MultimodalInputs  # 输入 ID 和多模态输入 # 输入标记 ID 和多模态输入
    ) -> list[int]:  # 返回填充后的 ID 列表 # 返回填充后的 ID 列表
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建填充模式 # 创建多模态数据填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 使用填充模式处理 # 使用填充模式处理输入标记


EntryClass = [NVILALiteForConditionalGeneration]  # 入口类 # 模型入口类列表
