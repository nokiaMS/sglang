# Jet VLM 视觉语言模型实现
# 该文件实现了 Jet VLM 条件生成模型，将 Siglip 视觉编码器与
# Jet Nemotron 语言模型结合，支持图像和视频多模态输入，
# 通过2x2下采样块和多层感知投影器将视觉特征映射到语言模型空间。

import math  # 导入数学库
from collections.abc import Iterable  # 导入可迭代类型

import einops  # 导入张量重排库
import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络函数模块
from torch import Tensor  # 导入张量类型
from transformers.modeling_outputs import BaseModelOutputWithPooling  # 导入模型输出类
from transformers.models.siglip import SiglipVisionModel  # 导入Siglip视觉模型

import sglang.srt.managers.mm_utils as mm_utils  # 导入多模态工具
import sglang.srt.model_loader.weight_utils as weight_utils  # 导入权重加载工具
import sglang.srt.utils as utils  # 导入工具函数
from sglang.srt.configs.jet_vlm import JetVLMConfig  # 导入Jet VLM配置
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # 导入logits处理器输出
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens  # 导入多模态填充模式
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.models.jet_nemotron import JetNemotronForCausalLM  # 导入Jet Nemotron语言模型

MM_HIDDEN_SIZE = 1152  # 多模态隐藏层大小（Siglip输出维度）


class JetVLMDownSample2x2BlockFix(nn.Module):
    """Jet VLM 2x2下采样块，将图像patch按2x2块合并以减少序列长度。"""
    def forward(self, x: Tensor) -> Tensor:
        """前向传播：执行2x2下采样操作。"""
        _, seq_len, _ = x.shape  # 获取输入形状

        feat_size = math.isqrt(seq_len)  # 计算特征图的边长（假设为正方形）

        features = einops.rearrange(x, "b (h w) d -> b h w d", h=feat_size, w=feat_size)  # 重排为2D特征图

        if feat_size % 2 == 1:  # 如果边长为奇数
            features = F.pad(features, (0, 0, 0, 1, 0, 1))  # 填充使边长为偶数

        features = einops.rearrange(
            features, "b (h p1) (w p2) d -> b (h w) (p1 p2 d)", p1=2, p2=2
        )  # 将2x2块展平合并，维度乘4

        return features  # 返回下采样后的特征


class JetVLMMultiModalProjector(nn.Module):
    """Jet VLM多模态投影器，将视觉特征投影到语言模型的隐藏空间。"""
    def __init__(self, config: JetVLMConfig) -> None:
        super().__init__()  # 调用父类初始化

        self.layers = nn.Sequential(  # 顺序层组合
            JetVLMDownSample2x2BlockFix(),  # 2x2下采样块
            nn.LayerNorm(MM_HIDDEN_SIZE * 4),  # 层归一化（维度为4倍隐藏大小）
            nn.Linear(MM_HIDDEN_SIZE * 4, config.text_config.hidden_size),  # 线性投影到文本隐藏大小
            nn.GELU(),  # GELU激活
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size),  # 线性层保持文本隐藏大小
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向传播：将视觉特征投影到语言模型空间。"""
        return self.layers(x)  # 通过投影层链


class JetVLMForConditionalGeneration(nn.Module):
    """Jet VLM条件生成模型，结合视觉编码器和语言模型实现多模态推理。"""
    def __init__(
        self,
        config: JetVLMConfig,  # Jet VLM配置
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置

        self.vision_tower = SiglipVisionModel(config.vision_config)  # 创建Siglip视觉编码器
        self.mm_projector = JetVLMMultiModalProjector(config)  # 创建多模态投影器
        self.llm = JetNemotronForCausalLM(  # 创建Jet Nemotron语言模型
            config=config.text_config,  # 文本配置
            quant_config=quant_config,  # 量化配置
            prefix=utils.add_prefix("llm", prefix),  # 参数前缀
        )

    def forward(
        self,
        input_ids: Tensor,  # 输入token ID
        positions: Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        get_embedding: bool = False,  # 是否获取嵌入
    ) -> LogitsProcessorOutput:
        """前向传播：执行多模态条件生成。"""
        output = mm_utils.general_mm_embed_routine(  # 调用通用多模态嵌入例程
            input_ids=input_ids,  # 输入token ID
            forward_batch=forward_batch,  # 前向批次信息
            language_model=self.llm,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数映射
                Modality.IMAGE: self.get_image_feature,  # 图像模态使用图像特征提取
                Modality.VIDEO: self.get_image_feature,  # 视频模态也使用图像特征提取
            },
            get_embedding=get_embedding,  # 是否获取嵌入
            positions=positions,  # 位置编码
        )

        assert isinstance(output, LogitsProcessorOutput)  # 确认输出类型

        return output  # 返回输出

    def get_image_feature(self, mm_input: list[MultimodalDataItem]) -> Tensor:
        """提取图像特征：通过视觉编码器和投影器获取图像嵌入。"""
        pixel_values = torch.cat([torch.tensor(x.feature) for x in mm_input], dim=0)  # 拼接所有图像像素值

        vision_tower_output: BaseModelOutputWithPooling = self.vision_tower(  # 通过视觉编码器
            pixel_values,  # 像素值
            output_hidden_states=True,  # 输出所有隐藏状态
        )
        assert vision_tower_output.hidden_states is not None  # 确认隐藏状态不为空

        vision_features = vision_tower_output.hidden_states[-2]  # 取倒数第二层隐藏状态作为视觉特征

        vision_features = self.mm_projector(vision_features)  # 通过多模态投影器

        vision_features = einops.rearrange(vision_features, "n p d -> (n p) d")  # 展平图像和patch维度

        return vision_features  # 返回视觉特征

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> None:
        """加载模型权重，分离语言模型权重和视觉/投影器权重。"""
        params_dict = dict(self.named_parameters())  # 获取参数字典

        for name, loaded_weight in weights:  # 遍历权重
            if name.startswith("llm."):  # 如果是语言模型权重
                self.llm.load_weights([(name[len("llm.") :], loaded_weight)])  # 传递给语言模型加载
            else:  # 否则是视觉编码器或投影器权重
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(  # 获取权重加载器
                    param, "weight_loader", weight_utils.default_weight_loader  # 默认使用标准加载器
                )
                weight_loader(param, loaded_weight)  # 加载权重

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        """填充输入token ID，替换多模态标记占位符。"""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建多模态填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 执行填充并返回


EntryClass = [JetVLMForConditionalGeneration]  # 入口类列表
