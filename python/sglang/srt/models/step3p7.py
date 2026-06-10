# Step3p7视觉语言模型实现文件
# 本文件实现了Step3p7多模态模型，结合PerceptionEncoder视觉编码器和Step3p5语言模型
# 支持NVFP4量化检查点的权重名称映射和图像特征提取与处理

from typing import Iterable, List, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入PyTorch神经网络模块
from transformers.activations import ACT2FN  # 导入激活函数映射

from sglang.srt.configs.step3p7 import Step3p7Config  # 导入Step3p7配置
from sglang.srt.layers.linear import ColumnParallelLinear  # 导入列并行线性层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.step3_vl_10b import PerceptionEncoder  # 导入感知编码器
from sglang.srt.models.step3p5 import Step3p5ForCausalLM  # 导入Step3p5因果语言模型
from sglang.srt.models.utils import WeightsMapper  # 导入权重映射器
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


class Step3p7ForConditionalGeneration(nn.Module):  # Step3p7条件生成模型类

    # NVFP4 checkpoints (e.g. huangyu-nv/step3p7-nvfp4-moe-only-kvfp8) use
    # "model.language_model." prefix, while sglang parameters are named
    # "language_model.model.". This mapper remaps the quantization ignore
    # patterns so that is_layer_skipped works correctly.
    # NVFP4检查点使用"model.language_model."前缀，而sglang参数命名为"language_model.model."
    # 此映射器重新映射量化忽略模式，以便is_layer_skipped正常工作
    hf_to_sglang_mapper = WeightsMapper(  # HuggingFace到SGLang权重映射器
        orig_to_new_prefix={  # 原始前缀到新前缀的映射
            "model.language_model.": "language_model.model.",  # 语言模型前缀映射
            "model.vision_model": "vision_model",  # 视觉模型前缀映射
            "model.vit_large_projector": "vit_large_projector",  # 投影器前缀映射
        }
    )

    @classmethod  # 类方法
    def get_model_config_for_expert_location(cls, config):  # 获取专家位置模型配置
        return Step3p5ForCausalLM.get_model_config_for_expert_location(  # 委托给Step3p5的方法
            config.text_config  # 使用文本配置
        )

    def __init__(  # 初始化方法
        self,
        config: Step3p7Config,  # Step3p7配置对象
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        self.vision_model = PerceptionEncoder(  # 创建感知编码器（视觉模型）
            config.vision_config,  # 视觉配置
            ACT2FN[config.vision_config.hidden_act],  # 激活函数
            quant_config=None,  # Vision weights are not quantized  # 视觉权重不进行量化
            prefix=add_prefix("vision_model", prefix),  # 添加前缀
        )
        self.vit_large_projector = ColumnParallelLinear(  # 创建ViT大型投影器
            config.vision_config.width * 4,  # 输入维度为视觉宽度乘4
            config.text_config.hidden_size,  # 输出维度为文本隐藏大小
            bias=config.projector_bias,  # 是否使用偏置
            gather_output=True,  # 收集输出
            quant_config=None,  # Projector weights are bf16  # 投影器权重为bf16精度
            prefix=add_prefix("vit_large_projector", prefix),  # 添加前缀
        )
        self.language_model = Step3p5ForCausalLM(  # 创建语言模型
            config=config.text_config,  # 文本配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("language_model", prefix),  # 添加前缀
        )

    def _get_vision_model_output(self, input_tensor: torch.Tensor) -> torch.Tensor:  # 获取视觉模型输出
        return self.vision_model(input_tensor)  # 将输入传递给视觉模型

    @property  # 属性装饰器
    def device(self) -> torch.device:  # 设备属性
        return self.vit_large_projector.weight.device  # 返回投影器权重所在设备

    def _flatten_embeddings(self, embeddings) -> torch.Tensor:  # 展平嵌入
        if isinstance(embeddings, torch.Tensor):  # 如果是张量
            return embeddings.flatten(0, -2)  # 展平除最后一维外的所有维度
        return torch.cat(tuple(self._flatten_embeddings(t) for t in embeddings))  # 递归展平并拼接

    def _process_image_features(self, image_features: torch.Tensor) -> torch.Tensor:  # 处理图像特征
        image_features, _ = self.vit_large_projector(image_features)  # 通过投影器
        return image_features  # 返回投影后的特征

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 获取图像特征
        assert len(items) == 1  # 断言只有一个数据项

        item = items[0]  # 获取第一个数据项
        pixel_values = item.feature.type(self.vision_model.dtype)  # 获取像素值并转换为模型精度
        num_patches = item.model_specific_data.get("num_patches")  # 获取补丁数量
        patch_pixel_values = item.model_specific_data.get("patch_pixel_values", None)  # 获取补丁像素值
        if patch_pixel_values is not None:  # 如果有补丁像素值
            patch_pixel_values = patch_pixel_values.type(self.vision_model.dtype).to(  # 转换精度并移至设备
                self.device
            )

        image_features = self._get_vision_model_output(pixel_values)  # 获取主图像特征
        patch_image_features = (  # 获取补丁图像特征
            self._get_vision_model_output(patch_pixel_values)  # 处理补丁像素值
            if patch_pixel_values is not None  # 如果补丁像素值存在
            else None  # 否则为None
        )
        image_features = self._process_image_features(image_features)  # 处理主图像特征
        patch_image_features = (  # 处理补丁图像特征
            self._process_image_features(patch_image_features)  # 处理补丁特征
            if patch_image_features is not None  # 如果补丁特征存在
            else None  # 否则为None
        )
        merged_image_features = []  # 合并后的图像特征列表
        cur_patch_idx = 0  # 当前补丁索引
        for i, num_patch in enumerate(num_patches):  # 遍历每个补丁数量
            cur_feature = []  # 当前特征列表
            if num_patch > 0:  # 如果有补丁
                patch_slice = patch_image_features[  # 获取补丁切片
                    cur_patch_idx : cur_patch_idx + num_patch  # 切片范围
                ]
                cur_feature.append(patch_slice.view(-1, patch_slice.shape[-1]))  # 添加补丁特征
            cur_feature.append(image_features[i].view(-1, image_features.shape[-1]))  # 添加主特征
            cur_patch_idx += num_patch  # 更新补丁索引
            merged_image_features.append(  # 添加合并特征
                torch.cat(cur_feature) if len(cur_feature) > 1 else cur_feature[0]  # 拼接或直接使用
            )
        return self._flatten_embeddings(merged_image_features)  # 返回展平后的合并特征

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):  # 填充输入ID
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 使用填充模式填充token

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        get_embedding: bool = False,  # 是否获取嵌入
    ):
        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入例程
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.language_model,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数
                Modality.IMAGE: self.get_image_feature,  # 图像模态的特征提取
            },
            positions=positions,  # 位置编码
        )
        return hidden_states  # 返回隐藏状态

    def get_embed_and_head(self):  # 获取嵌入层和语言模型头
        return self.language_model.get_embed_and_head()  # 委托给语言模型

    def set_embed_and_head(self, embed, head):  # 设置嵌入层和语言模型头
        self.language_model.set_embed_and_head(embed, head)  # 委托给语言模型

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法
        weights = list(weights)  # 将权重转换为列表

        vision_weights = []  # 视觉权重列表
        language_weights = []  # 语言权重列表

        for name, loaded_weight in weights:  # 遍历所有权重
            # NVFP4 checkpoints use "model.language_model." prefix for
            # language weights and "model.vision_model." for vision weights,
            # while FP8 checkpoints use "model." and "vision_model." directly.
            # NVFP4检查点使用"model.language_model."前缀表示语言权重，
            # 使用"model.vision_model."表示视觉权重，而FP8检查点直接使用"model."和"vision_model."
            name = name.replace("language_model.", "", 1)  # 移除第一个language_model.前缀

            if "vision_model" in name or "vit_large_projector" in name:  # 如果是视觉权重
                # Strip leading "model." for vision weights (NVFP4 format)
                # 移除视觉权重前导的"model."（NVFP4格式）
                if name.startswith("model."):  # 如果以model.开头
                    name = name[len("model.") :]  # 移除model.前缀
                name = name.replace(r".attn.in_proj_weight", r".attn.qkv_proj.weight")  # 替换注意力权重名
                name = name.replace(r".attn.in_proj_bias", r".attn.qkv_proj.bias")  # 替换注意力偏置名
                name = name.replace(r".attn.out_proj.bias", r".attn.proj.bias")  # 替换输出投影偏置名
                name = name.replace(r".attn.out_proj.weight", r".attn.proj.weight")  # 替换输出投影权重名
                name = name.replace(".mlp.c_fc", ".mlp.fc1")  # 替换MLP第一层名
                name = name.replace(".mlp.c_proj", ".mlp.fc2")  # 替换MLP第二层名
                vision_weights.append((name, loaded_weight))  # 添加到视觉权重列表
            else:  # 否则是语言权重
                language_weights.append((name, loaded_weight))  # 添加到语言权重列表

        # Load vision tower weights
        # 加载视觉塔权重
        params_dict = dict(self.named_parameters(remove_duplicate=False))  # 获取参数字典
        for name, loaded_weight in vision_weights:  # 遍历视觉权重
            if name not in params_dict:  # 如果参数名不存在
                raise ValueError(f"Weight {name} not found in params_dict")  # 抛出异常
            param = params_dict[name]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, loaded_weight)  # 加载权重

        # Load language model weights
        # 加载语言模型权重
        if language_weights:  # 如果有语言权重
            self.language_model.load_weights(language_weights)  # 加载语言模型权重


EntryClass = Step3p7ForConditionalGeneration  # 注册入口类为Step3p7ForConditionalGeneration
