# Dots OCR 光学字符识别模型实现
# 本文件实现了DotsOCR模型，基于Qwen2.5-VL的SGLang实现改编，
# 结合了DotsVisionTransformer视觉编码器和Qwen2语言模型，
# 用于光学字符识别（OCR）任务。

# coding=utf-8
# Adapted from Qwen2.5-VL SGLang implementation
# 改编自Qwen2.5-VL SGLang实现

import logging  # 导入日志模块
from typing import Iterable, List, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入神经网络模块

from sglang.srt.configs import DotsOCRConfig  # 导入DotsOCR配置类
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行语言模型头
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs  # 导入多模态数据项和输入
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.dots_vlm_vit import DotsVisionTransformer  # 导入Dots视觉Transformer
from sglang.srt.models.qwen2 import Qwen2ForCausalLM  # 导入Qwen2因果语言模型
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

logger = logging.getLogger(__name__)  # 创建模块级日志记录器


class DotsOCRForCausalLM(nn.Module):
    """Dots OCR 因果语言模型，用于光学字符识别"""
    def __init__(  # 初始化方法
        self,
        config: DotsOCRConfig,  # DotsOCR配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数名前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        # Initialize vision transformer
        # 初始化视觉Transformer
        self.visual = DotsVisionTransformer(  # 创建视觉Transformer
            config.vision_config,  # 视觉配置
        )

        # Initialize language model
        # 初始化语言模型
        self.model = Qwen2ForCausalLM(config, quant_config)  # 创建Qwen2因果语言模型

        # Initialize LM head
        # 初始化语言模型头
        if config.tie_word_embeddings:  # 如果绑定词嵌入权重
            self.lm_head = self.model.embed_tokens  # LM头与嵌入层共享权重
        else:  # 否则
            self.lm_head = ParallelLMHead(  # 创建独立的并行语言模型头
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层维度
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("lm_head", prefix),  # 添加前缀
            )

        self.logits_processor = LogitsProcessor(config)  # 创建logits处理器

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):  # 填充输入ID，插入多模态token
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建多模态token填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 使用填充模式处理输入token

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 提取图像特征
        # Extract pixel values and grid information (following reference pattern)
        # 提取像素值和网格信息（遵循参考模式）
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(  # 拼接所有图像的像素值
            self.visual.dtype  # 转换为视觉模型的数据类型
        )
        image_grid_thw = torch.concat(  # 拼接所有图像的网格信息（时间、高度、宽度）
            [item.image_grid_thw for item in items], dim=0
        ).to(self.visual.device)  # 移动到视觉模型的设备

        # Add dimension checks like in reference code
        # 添加维度检查，如参考代码中所示
        assert pixel_values.dim() == 2, f"{pixel_values.dim()=}"  # 断言像素值为2维
        assert image_grid_thw.dim() == 2, f"{image_grid_thw.dim()=}"  # 断言网格信息为2维

        # Process through vision tower
        # 通过视觉塔处理
        image_embeds = self.visual(pixel_values, image_grid_thw)  # 调用视觉Transformer获取图像嵌入

        # Ensure consistent dtype for FlashInfer compatibility
        # 确保数据类型一致以兼容FlashInfer
        # Force bfloat16 to match model's expected dtype
        # 强制bfloat16以匹配模型期望的数据类型
        if hasattr(self.model, "embed_tokens"):  # 如果模型有嵌入层
            target_dtype = self.model.embed_tokens.weight.dtype  # 获取目标数据类型
            if image_embeds.dtype != target_dtype:  # 如果数据类型不匹配
                image_embeds = image_embeds.to(target_dtype)  # 转换数据类型

        return image_embeds  # 返回图像嵌入

    def _pad_vit_attn_dummy_heads(self, name: str, loaded_weight: torch.Tensor):  # 为虚拟头填充注意力QKV权重
        """pad attn qkv weights for dummy heads
        为虚拟头填充注意力qkv权重"""
        num_dummy_heads = self.config.vision_config.num_dummy_heads  # 获取虚拟头数量
        if num_dummy_heads == 0:  # 如果没有虚拟头
            return loaded_weight  # 直接返回原始权重
        head_dim = self.config.vision_config.head_dim  # 获取每个头的维度

        if "attn.qkv_proj" in name:  # 如果是QKV投影权重
            wq, wk, wv = loaded_weight.chunk(3, dim=0)  # 将权重拆分为Q、K、V三部分
            if name.endswith(".weight"):  # 如果是权重张量
                dummy_shape = [num_dummy_heads, head_dim, wq.shape[-1]]  # 虚拟头权重的形状
            elif name.endswith(".bias"):  # 如果是偏置张量
                dummy_shape = [num_dummy_heads, head_dim]  # 虚拟头偏置的形状
            else:  # 其他情况
                raise RuntimeError(f"Unsupported weight with name={name}")  # 抛出运行时错误
            pad_func = lambda x: torch.cat(  # 填充函数：在头部维度末尾添加零填充
                [x.unflatten(0, (-1, head_dim)), x.new_zeros(dummy_shape)], dim=0  # 拆分头部维度后拼接零张量
            ).flatten(0, 1)  # 再展平回原格式
            wq, wk, wv = pad_func(wq), pad_func(wk), pad_func(wv)  # 对Q、K、V分别填充
            loaded_weight = torch.cat([wq, wk, wv], dim=0)  # 重新拼接为完整权重
        if "attn.proj.weight" in name:  # 如果是输出投影权重
            padded_weight = loaded_weight.new_zeros(  # 创建零填充权重
                loaded_weight.shape[0], head_dim * num_dummy_heads  # 在最后一维添加虚拟头维度
            )
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=-1)  # 拼接填充权重
        if "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:  # 如果是Q或K归一化权重
            padded_weight = loaded_weight.new_zeros(head_dim * num_dummy_heads)  # 创建零填充归一化权重
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=0)  # 拼接填充权重
        return loaded_weight  # 返回填充后的权重

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        **kwargs: object,  # 其他关键字参数
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(  # 调用通用多模态嵌入例程
            input_ids=input_ids,  # 输入token ID
            positions=positions,  # 位置编码
            forward_batch=forward_batch,  # 前向批次信息
            multimodal_model=self,  # 多模态模型（自身）
            language_model=self.model,  # 语言模型
        )
        return hidden_states  # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        """Load weights for the model, separating vision and language weights
        加载模型权重，分离视觉和语言权重"""
        weights = list(weights)  # 将权重迭代器转为列表

        # Separate vision tower weights and language model weights
        # 分离视觉塔权重和语言模型权重
        vision_weights = []  # 视觉权重列表
        language_weights = []  # 语言权重列表

        for name, loaded_weight in weights:  # 遍历所有权重
            if name.startswith("vision_tower."):  # 如果是视觉塔权重
                vision_name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")  # 替换QKV命名格式

                vision_weights.append((vision_name, loaded_weight))  # 添加到视觉权重列表
            else:  # 否则
                # All other weights go to language model
                # 所有其他权重归属语言模型
                language_weights.append((name, loaded_weight))  # 添加到语言权重列表

        # Load vision tower weights
        # 加载视觉塔权重
        vision_state_dict = dict(vision_weights)  # 将视觉权重转为字典
        params_dict = dict(self.named_parameters(remove_duplicate=False))  # 获取模型参数字典

        for name, loaded_weight in vision_state_dict.items():  # 遍历视觉权重
            name = name.replace("vision_tower", "visual")  # 替换前缀名：vision_tower -> visual
            if name not in params_dict:  # 如果参数名不存在
                raise ValueError(f"Weight {name} not found in params_dict")  # 抛出值错误
            param = params_dict[name]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            loaded_weight = self._pad_vit_attn_dummy_heads(name, loaded_weight)  # 填充虚拟头权重
            weight_loader(param, loaded_weight)  # 加载权重

        if language_weights:  # 如果有语言模型权重
            self.model.load_weights(language_weights)  # 加载语言模型权重

    def get_embed_and_head(self):  # 获取嵌入权重和LM头权重
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入权重和LM头权重


EntryClass = [DotsOCRForCausalLM]  # 模型入口类注册
