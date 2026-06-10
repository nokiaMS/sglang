# NVILA 多模态视觉语言模型实现
# 该文件实现了 NVILA 模型，结合 Siglip 视觉编码器和 Qwen2 语言模型，
# 支持动态分辨率的图像和视频输入，通过棋盘格分割/合并和多尺度特征融合处理视觉特征。

import itertools  # 导入迭代工具 # 导入迭代工具库
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

MM_HIDDEN_SIZE = 3456  # 多模态隐藏层大小 # 多模态隐藏层维度大小


class NVILAConfig(PretrainedConfig):
    """NVILA 模型配置类，继承自预训练配置基类"""
    model_type = "nvila"  # 模型类型标识 # 模型类型标识符
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


class NVILAMultiModalProjectorDownsampleBlock(nn.Module):
    """NVILA 多模态投影器下采样块，将空间维度缩小 2 倍"""

    def forward(self, x: Tensor) -> Tensor:  # 前向传播 # 前向传播方法
        batch_size, sequence_length, hidden_size = x.shape  # 获取输入形状 # 获取批次大小、序列长度和隐藏维度

        feat_size = math.isqrt(sequence_length)  # 计算特征图边长 # 计算特征图的边长（整数平方根）

        features = x.reshape(batch_size, feat_size, feat_size, hidden_size)  # 重塑为空间特征图 # 重塑为空间特征图

        pad_after = feat_size % 2  # 计算填充量使其能被 2 整除 # 计算需要的填充量
        if pad_after > 0:  # 如果需要填充 # 如果需要填充
            features = F.pad(features, (0, 0, 0, pad_after, 0, pad_after))  # 对空间维度填充 # 对空间维度进行填充
            feat_size = feat_size + pad_after  # 更新特征图边长 # 更新特征图边长

        features = features.reshape(  # 重塑特征图以进行下采样 # 重塑特征图以下采样
            batch_size, feat_size // 2, 2, feat_size // 2, 2, hidden_size  # 分成 2x2 块 # 分成 2x2 块
        )
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous()  # 重排维度以合并空间信息 # 重排维度以合并空间信息
        features = features.reshape(batch_size, -1, 4 * hidden_size)  # 展平为序列 # 展平为序列，通道维度扩大 4 倍

        return features  # 返回下采样后的特征 # 返回下采样后的特征


class NVILAMultiModalProjector(nn.Module):
    """NVILA 多模态投影器，将视觉特征投影到语言模型空间"""

    def __init__(self, config: NVILAConfig):  # 初始化方法 # 初始化方法
        super().__init__()  # 调用父类初始化 # 调用父类初始化

        self.layers = nn.Sequential(  # 投影层序列 # 顺序投影层
            NVILAMultiModalProjectorDownsampleBlock(),  # 下采样块 # 下采样块
            nn.LayerNorm(MM_HIDDEN_SIZE * 4),  # 层归一化 # 层归一化
            nn.Linear(MM_HIDDEN_SIZE * 4, config.text_config.hidden_size),  # 线性投影 # 线性投影层
            nn.GELU(),  # GELU 激活 # GELU 激活函数
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size),  # 最终线性层 # 最终线性投影
        )

    def forward(self, x: Tensor) -> Tensor:  # 前向传播 # 前向传播方法
        return self.layers(x)  # 通过投影层序列 # 通过所有投影层


class NVILAForConditionalGeneration(nn.Module):
    """NVILA 条件生成模型，结合视觉编码器和语言模型，支持动态分辨率"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: NVILAConfig,  # 模型配置 # 模型配置
        quant_config: QuantizationConfig | None = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用父类初始化

        self.config = config  # 保存配置 # 保存模型配置

        self.vision_tower = SiglipVisionModel(config.vision_config)  # 视觉编码器 # 创建 Siglip 视觉编码器
        self.mm_projector = NVILAMultiModalProjector(config)  # 多模态投影器 # 创建多模态投影器
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

    def get_image_feature(self, mm_input: list[MultimodalDataItem]) -> Tensor:  # 获取图像特征 # 提取图像视觉特征（支持动态分辨率）
        block_sizes = (  # 获取块大小 # 获取每个图像的块大小
            list(
                itertools.chain.from_iterable(  # 展平嵌套列表 # 展平嵌套列表
                    x.block_sizes for x in mm_input if hasattr(x, "block_sizes")  # 获取每个项的块大小 # 获取每个数据项的块大小
                )
            )
            or None  # 如果为空则设为 None # 如果为空则设为 None
        )
        pixel_values = torch.cat([torch.tensor(x.feature) for x in mm_input], dim=0)  # 拼接像素值 # 拼接所有图像的像素值

        vision_tower_output: BaseModelOutputWithPooling = self.vision_tower(  # 视觉编码器前向 # 通过视觉编码器
            pixel_values.to(  # 转换设备和数据类型 # 转换到正确的设备和数据类型
                device=self.vision_tower.device, dtype=self.vision_tower.dtype  # 使用视觉编码器的设备和类型 # 使用视觉编码器的设备和类型
            ),
            output_hidden_states=True,  # 输出隐藏状态 # 输出所有隐藏层状态
        )
        assert vision_tower_output.hidden_states is not None  # 断言隐藏状态不为空 # 断言隐藏状态存在

        vision_features: Tensor = vision_tower_output.hidden_states[-2]  # 取倒数第二层特征 # 取倒数第二层的视觉特征

        vision_features_list, block_sizes = merge_features_for_dynamic_s2(  # 合并动态分辨率特征 # 合并动态 S2 特征
            vision_features,  # 视觉特征 # 视觉特征
            block_sizes=(  # 块大小 # 块大小
                block_sizes  # 块大小 # 如果已有块大小
                if block_sizes is not None
                else [None] * vision_features.shape[0]  # 否则设为 None 列表 # 否则为每张图设为 None
            ),
            resize_output_to_scale_idx=-1,  # 输出调整到最大尺度 # 输出调整到最后一个尺度索引
            scales=[448, 896, 1344],  # 支持的尺度列表 # 支持的尺度列表
        )

        vision_features_list = [  # 棋盘格分割 # 对每个特征进行棋盘格分割
            split_chessboard(x, block_size[0], block_size[1])  # 分割棋盘格 # 按块大小分割
            for x, block_size in zip(vision_features_list, block_sizes)  # 遍历特征和块大小 # 遍历特征和块大小
        ]

        vision_features = torch.cat(  # 拼接分割后的特征 # 拼接所有分割后的特征
            [einops.rearrange(x, "b c h w -> b (h w) c") for x in vision_features_list]  # 重排为序列 # 将空间维度展平为序列
        )

        vision_features = self.mm_projector(vision_features)  # 通过投影器 # 通过多模态投影器

        vision_features_list = list(  # 按块大小分割 # 按块大小分割投影后的特征
            vision_features.split(
                [block_size[0] * block_size[1] for block_size in block_sizes], dim=0  # 每个图像的标记数 # 每个图像的标记数量
            )
        )
        vision_features_list = [  # 棋盘格合并 # 对每个特征进行棋盘格合并
            merge_chessboard(x, block_size[0], block_size[1])  # 合并棋盘格 # 按块大小合并
            for x, block_size in zip(vision_features_list, block_sizes)  # 遍历特征和块大小 # 遍历特征和块大小
        ]

        vision_features = torch.stack(  # 堆叠特征 # 堆叠所有图像的特征
            [einops.rearrange(x, "1 c h w -> (h w) c") for x in vision_features_list]  # 重排为序列 # 将空间维度展平为序列
        )

        vision_features = einops.rearrange(vision_features, "n p d -> (n p) d")  # 展平批量和序列维度 # 展平批量和序列维度

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


def merge_chessboard(x, num_split_h, num_split_w):  # 合并棋盘格 # 将分割的子块合并回完整棋盘格
    """
    x: b * n * c or b * h * w * c
    out: b * c * h * w
    Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
    """
    B = x.shape[0]  # 批量大小 # 获取批量大小
    if x.dim() == 3:  # 如果是 3D 张量 # 如果输入是 3 维张量
        N = x.shape[1]  # 序列长度 # 获取序列长度
        x = einops.rearrange(  # 重排为空间格式 # 重排为空间格式
            x, "b (h w) c -> b c h w", h=math.isqrt(N), w=math.isqrt(N)  # 恢复空间维度 # 恢复空间维度
        )

    assert B % (num_split_h * num_split_w) == 0  # 断言批量可被分割数整除 # 断言批量可被分割数整除
    b = B // (num_split_h * num_split_w)  # 每个子图的批量大小 # 计算每个子图的批量大小

    x_merge = torch.cat(  # 合并子块 # 合并子块
        [
            torch.cat(  # 沿宽度合并 # 沿宽度方向合并
                [
                    x[(i * num_split_w + j) * b : (i * num_split_w + j + 1) * b]  # 获取子块 # 获取对应位置的子块
                    for j in range(num_split_w)  # 遍历列 # 遍历列方向
                ],
                dim=-1,  # 沿宽度维度 # 沿宽度维度拼接
            )
            for i in range(num_split_h)  # 遍历行 # 遍历行方向
        ],
        dim=-2,  # 沿高度维度 # 沿高度维度拼接
    )

    return x_merge  # 返回合并结果 # 返回合并结果


def merge_features_for_dynamic_s2(  # 合并动态 S2 特征 # 合并多尺度动态分辨率的特征
    image_features, block_sizes, *, scales, resize_output_to_scale_idx  # 图像特征、块大小、尺度列表、输出调整索引 # 输入参数
):
    image_features_each_image = []  # 每张图像的特征列表 # 存储每张图像的特征
    new_block_sizes = []  # 新的块大小列表 # 存储新的块大小
    block_cnt = 0  # 块计数器 # 块计数器
    for block_size_each_image in block_sizes:  # 遍历每张图像的块大小 # 遍历每张图像的块大小
        if block_size_each_image is None:  # 如果没有块大小信息 # 如果没有块大小信息
            cur_features = image_features[block_cnt : block_cnt + 1]  # 取单个块的特征 # 取单个块的特征
            cur_features = einops.rearrange(  # 重排为空间格式 # 重排为空间格式
                cur_features,
                "1 (h w) c -> 1 c h w",
                h=math.isqrt(cur_features.shape[1]),  # 计算高度 # 计算高度
            )
            cur_features = cur_features.repeat(1, len(scales), 1, 1)  # 在通道维度重复 # 在通道维度重复以匹配多尺度
            image_features_each_image.append(cur_features)  # 添加到列表 # 添加到列表
            new_block_sizes.append((1, 1))  # 默认块大小 # 默认块大小为 1x1
            block_cnt += 1  # 计数器加一 # 计数器加一
        else:  # 否则 # 有块大小信息的情况
            cur_features_each_scale = []  # 每个尺度的特征列表 # 存储每个尺度的特征
            for scale in scales[:-1]:  # 遍历除最大尺度外的所有尺度 # 遍历除最大尺度外的所有尺度
                num_blocks_this_scale = (scale // scales[0]) ** 2  # 该尺度的块数 # 计算该尺度的块数
                cur_features_each_scale.append(
                    merge_chessboard(  # 合并棋盘格 # 合并棋盘格
                        image_features[block_cnt : block_cnt + num_blocks_this_scale],  # 取该尺度的块 # 取该尺度的所有块
                        num_split_h=scale // scales[0],  # 行分割数 # 行方向的分割数
                        num_split_w=scale // scales[0],  # 列分割数 # 列方向的分割数
                    )
                )  # 1 * C * H * W # 1 * C * H * W 格式
                block_cnt += num_blocks_this_scale  # 更新计数器 # 更新计数器
            num_blocks_last_scale = block_size_each_image[0] * block_size_each_image[1]  # 最大尺度的块数 # 最大尺度的块数
            cur_features_each_scale.append(
                merge_chessboard(  # 合并棋盘格 # 合并棋盘格
                    image_features[block_cnt : block_cnt + num_blocks_last_scale],  # 取最大尺度的块 # 取最大尺度的所有块
                    num_split_h=block_size_each_image[0],  # 行分割数 # 行方向的分割数
                    num_split_w=block_size_each_image[1],  # 列分割数 # 列方向的分割数
                )
            )  # 1 * C * H * W # 1 * C * H * W 格式
            block_cnt += num_blocks_last_scale  # 更新计数器 # 更新计数器

            # resize and concat features from different scales # 调整大小并拼接不同尺度的特征 # 调整大小并拼接不同尺度的特征
            output_size = cur_features_each_scale[resize_output_to_scale_idx].shape[-2:]  # 输出尺寸 # 输出尺寸
            cur_features = torch.cat(  # 拼接多尺度特征 # 拼接多尺度特征
                [
                    F.interpolate(  # 插值调整大小 # 插值调整大小
                        cur_features_each_scale[i].to(torch.float32),  # 转为浮点 # 转为浮点数
                        size=output_size,  # 目标尺寸 # 目标尺寸
                        mode="area",  # 面积插值 # 使用面积插值模式
                    ).to(cur_features_each_scale[i].dtype)  # 恢复原始数据类型 # 恢复原始数据类型
                    for i in range(len(cur_features_each_scale))  # 遍历每个尺度 # 遍历每个尺度
                ],
                dim=1,  # 沿通道维度拼接 # 沿通道维度拼接
            )

            image_features_each_image.append(cur_features)  # 添加到列表 # 添加到列表

            if (  # 判断输出调整索引 # 判断是否使用原始块大小
                resize_output_to_scale_idx == len(scales) - 1
                or resize_output_to_scale_idx == -1
            ):
                new_block_sizes.append(block_size_each_image)  # 使用原始块大小 # 使用原始块大小
            else:  # 否则 # 使用调整后的块大小
                new_block_sizes.append(
                    (
                        scales[resize_output_to_scale_idx] // scales[0],  # 行数 # 行方向的分割数
                        scales[resize_output_to_scale_idx] // scales[0],  # 列数 # 列方向的分割数
                    )
                )

    assert block_cnt == len(  # 断言块计数正确 # 断言块计数与特征数一致
        image_features
    ), f"The number of blocks ({block_cnt}) does not match length of image_features ({len(image_features)})!"

    return image_features_each_image, new_block_sizes  # 返回特征和块大小 # 返回每张图像的特征和对应的块大小


def split_chessboard(x, num_split_h, num_split_w):  # 分割棋盘格 # 将特征图分割为棋盘格子块
    """
    x: b * c * h * w
    out: b * c * h * w
    Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape  # 获取形状 # 获取批量、通道、高度、宽度
    assert H % num_split_h == 0 and W % num_split_w == 0  # 断言可被整除 # 断言高度和宽度可被分割数整除
    h, w = H // num_split_h, W // num_split_w  # 子块高度和宽度 # 计算子块的高度和宽度
    x_split = torch.cat(  # 拼接子块 # 拼接所有子块
        [
            x[:, :, i * h : (i + 1) * h, j * w : (j + 1) * w]  # 获取子块 # 获取对应位置的子块
            for i in range(num_split_h)  # 遍历行 # 遍历行方向
            for j in range(num_split_w)  # 遍历列 # 遍历列方向
        ],
        dim=0,  # 沿批量维度拼接 # 沿批量维度拼接
    )
    return x_split  # 返回分割结果 # 返回分割结果


EntryClass = [NVILAForConditionalGeneration]  # 入口类 # 模型入口类列表
