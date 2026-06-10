# CLIP模型实现文件 - 实现CLIP视觉-语言对比学习模型的SGLang推理，包含文本编码器和视觉编码器
# Adapted from  # 改编自
# https://github.com/huggingface/transformers/blob/af9b2eaa54c150741f298d6db939af6328e1dc38/src/transformers/models/clip/modeling_clip.py  # HuggingFace CLIP模型参考链接

from functools import partial # 导入偏函数工具
from typing import Iterable, List, Optional, Tuple, Type, Union # 导入类型提示模块

import torch # 导入PyTorch深度学习框架
import torch.nn as nn # 导入神经网络模块
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig # 导入CLIP相关配置类
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask # 导入4D因果注意力掩码创建工具

from sglang.srt.layers.activation import QuickGELU # 导入快速GELU激活函数
from sglang.srt.layers.attention.vision import VisionAttention # 导入视觉注意力层
from sglang.srt.layers.conv import Conv2dLayer # 导入2D卷积层
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear # 导入并行线性层
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType # 导入池化相关组件
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.managers.schedule_batch import MultimodalInputs # 导入多模态输入类
from sglang.srt.model_executor.model_runner import ForwardBatch # 导入前向批次信息类
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.utils import add_prefix, flatten_nested_list # 导入工具函数


class CLIPVisionEmbeddings(nn.Module): # CLIP视觉嵌入层类，将图像转换为patch嵌入

    def __init__(self, config: CLIPVisionConfig): # 初始化CLIP视觉嵌入层
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.embed_dim = config.hidden_size # 嵌入维度
        self.image_size = config.image_size # 输入图像大小
        self.patch_size = config.patch_size # patch大小
        assert self.image_size % self.patch_size == 0 # 确保图像大小可被patch大小整除

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim)) # CLS token嵌入参数

        self.patch_embedding = Conv2dLayer( # patch嵌入卷积层
            in_channels=config.num_channels, # 输入通道数
            out_channels=self.embed_dim, # 输出通道数（嵌入维度）
            kernel_size=self.patch_size, # 卷积核大小等于patch大小
            stride=self.patch_size, # 步幅等于patch大小
            bias=False, # 不使用偏置
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2 # patch数量
        self.num_positions = self.num_patches + 1 # 位置数（patch数+1个CLS token）
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim) # 位置嵌入
        self.register_buffer( # 注册位置ID缓冲区
            "position_ids", # 缓冲区名称
            torch.arange(self.num_positions).expand((1, -1)), # 位置ID从0到num_positions-1
            persistent=False, # 不持久化
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: # 前向传播：将像素值转换为嵌入
        batch_size = pixel_values.shape[0] # 获取批次大小
        target_dtype = self.patch_embedding.weight.dtype # 获取目标数据类型
        patch_embeds = self.patch_embedding( # 计算patch嵌入
            pixel_values.to(dtype=target_dtype) # 转换为目标数据类型
        )  # shape = [*, width, grid, grid]  # 形状 = [*, 宽度, 网格, 网格]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2) # 展平并转置为[批次, patch数, 维度]

        class_embeds = self.class_embedding.expand(batch_size, 1, -1) # 扩展CLS嵌入
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1) # 拼接CLS和patch嵌入
        embeddings = embeddings + self.position_embedding(self.position_ids) # 加上位置嵌入

        return embeddings # 返回嵌入结果


class CLIPTextEmbeddings(nn.Module): # CLIP文本嵌入层类，将文本token转换为嵌入
    def __init__(self, config: CLIPTextConfig): # 初始化CLIP文本嵌入层
        super().__init__() # 调用父类初始化
        embed_dim = config.hidden_size # 嵌入维度

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim) # token嵌入层
        self.position_embedding = nn.Embedding( # 位置嵌入层
            config.max_position_embeddings, embed_dim # 最大位置数和嵌入维度
        )

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized  # position_ids (1, 位置嵌入长度) 在内存中连续，序列化时导出
        self.register_buffer( # 注册位置ID缓冲区
            "position_ids", # 缓冲区名称
            torch.arange(config.max_position_embeddings).expand((1, -1)), # 从0到最大位置数的ID
            persistent=False, # 不持久化
        )

    def forward( # 前向传播：计算文本嵌入
        self,
        input_ids: Optional[torch.LongTensor] = None, # 输入token ID（可选）
        position_ids: Optional[torch.LongTensor] = None, # 位置ID（可选）
        inputs_embeds: Optional[torch.FloatTensor] = None, # 输入嵌入（可选）
    ) -> torch.Tensor:
        seq_length = ( # 获取序列长度
            input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2] # 从input_ids或inputs_embeds获取
        )

        if position_ids is None: # 如果没有提供位置ID
            position_ids = self.position_ids[:, :seq_length] # 使用预定义的位置ID

        if inputs_embeds is None: # 如果没有提供输入嵌入
            inputs_embeds = self.token_embedding(input_ids) # 通过token嵌入层计算

        position_embeddings = self.position_embedding(position_ids) # 计算位置嵌入
        embeddings = inputs_embeds + position_embeddings # token嵌入加上位置嵌入

        return embeddings # 返回嵌入结果


class CLIPMLP(nn.Module): # CLIP MLP类，实现前馈神经网络

    def __init__( # 初始化CLIP MLP层
        self,
        config, # 模型配置
        act_layer: Type[nn.Module] = QuickGELU, # 激活函数类型，默认QuickGELU
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.fc1 = ColumnParallelLinear( # 第一个全连接层（升维）
            config.hidden_size, # 输入维度
            config.intermediate_size, # 中间层维度
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("fc1", prefix), # 参数名前缀
        )
        self.act = act_layer() # 激活函数实例
        self.fc2 = RowParallelLinear( # 第二个全连接层（降维）
            config.intermediate_size, # 输入维度
            config.hidden_size, # 输出维度
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("fc2", prefix), # 参数名前缀
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 前向传播：升维→激活→降维
        x_parallel, _ = self.fc1(x) # 通过第一个全连接层
        x_parallel = self.act(x_parallel) # 通过激活函数
        x, _ = self.fc2(x_parallel) # 通过第二个全连接层
        return x # 返回输出


class CLIPEncoderLayer(nn.Module): # CLIP编码器层类，包含自注意力和MLP

    def __init__( # 初始化CLIP编码器层
        self,
        config: CLIPVisionConfig, # CLIP视觉配置
        act_layer: Type[nn.Module] = QuickGELU, # 激活函数类型
        norm_layer: Type[nn.Module] = None, # 归一化层类型
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        if norm_layer is None: # 如果没有指定归一化层
            norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps) # 使用LayerNorm
        self.layer_norm1 = norm_layer(config.hidden_size) # 第一个层归一化
        self.layer_norm2 = norm_layer(config.hidden_size) # 第二个层归一化
        self.self_attn = VisionAttention( # 视觉自注意力层
            embed_dim=config.hidden_size, # 嵌入维度
            num_heads=config.num_attention_heads, # 注意力头数
            projection_size=config.hidden_size, # 投影维度
            use_qkv_parallel=True, # 使用QKV并行
            flatten_batch=True, # 展平批次
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("self_attn", prefix), # 参数名前缀
        )
        self.mlp = CLIPMLP( # MLP层
            config, # 模型配置
            act_layer=act_layer, # 激活函数
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("mlp", prefix), # 参数名前缀
        )

    def forward( # 前向传播：层归一化→注意力→残差→层归一化→MLP→残差
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        attention_mask: torch.Tensor, # 注意力掩码
        causal_attention_mask: torch.Tensor, # 因果注意力掩码
    ) -> torch.Tensor:

        residual = hidden_states # 保存残差
        hidden_states = self.layer_norm1(hidden_states) # 通过第一个层归一化
        # CLIP text model uses both `causal_attention_mask` and `attention_mask`  # CLIP文本模型同时使用因果注意力掩码和注意力掩码
        if attention_mask is not None and causal_attention_mask is not None: # 如果两种掩码都有
            attn_mask = attention_mask + causal_attention_mask # 合并两种掩码
        elif causal_attention_mask is not None: # 如果只有因果掩码
            attn_mask = causal_attention_mask # 使用因果掩码
        else: # 否则
            attn_mask = attention_mask # 使用注意力掩码
        hidden_states = self.self_attn( # 通过自注意力层
            hidden_states, # 隐藏状态
            attention_mask=attn_mask, # 注意力掩码
            # causal_attention_mask=causal_attention_mask,  # 因果注意力掩码（已注释）
        )

        hidden_states = residual + hidden_states # 残差连接
        residual = hidden_states # 更新残差
        hidden_states = self.layer_norm2(hidden_states) # 通过第二个层归一化
        hidden_states = self.mlp(hidden_states) # 通过MLP层
        hidden_states = residual + hidden_states # 残差连接
        return hidden_states # 返回输出


class CLIPEncoder(nn.Module): # CLIP编码器类，包含多层CLIPEncoderLayer
    """
    Transformer encoder consisting of `config.num_hidden_layers` self  # 由config.num_hidden_layers个自
    attention layers. Each layer is a [`CLIPEncoderLayer`].  # 注意力层组成的Transformer编码器。每层是CLIPEncoderLayer

    Args:  # 参数
        config: CLIPConfig  # 配置：CLIP配置
    """

    def __init__( # 初始化CLIP编码器
        self,
        config: CLIPVisionConfig, # CLIP视觉配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ) -> None:
        super().__init__() # 调用父类初始化

        self.config = config # 保存配置

        num_hidden_layers = config.num_hidden_layers # 隐藏层数
        norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps) # 归一化层
        self.layers = nn.ModuleList( # 创建编码器层列表
            [
                CLIPEncoderLayer(
                    config=config, # 配置
                    norm_layer=norm_layer, # 归一化层
                    quant_config=quant_config, # 量化配置
                    prefix=add_prefix(f"layers.{layer_idx}", prefix), # 参数名前缀
                )
                for layer_idx in range(num_hidden_layers) # 遍历所有层
            ]
        )

    def forward( # 前向传播：依次通过所有编码器层
        self,
        inputs_embeds: torch.Tensor, # 输入嵌入
        attention_mask: torch.Tensor = None, # 注意力掩码（可选）
        causal_attention_mask: torch.Tensor = None, # 因果注意力掩码（可选）
        return_all_hidden_states: bool = False, # 是否返回所有层的隐藏状态
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        hidden_states_pool = [inputs_embeds] # 隐藏状态池，初始包含输入嵌入
        hidden_states = inputs_embeds # 初始化隐藏状态

        for encoder_layer in self.layers: # 遍历所有编码器层
            hidden_states = encoder_layer( # 通过当前编码器层
                hidden_states, attention_mask, causal_attention_mask # 隐藏状态和掩码
            )
            if return_all_hidden_states: # 如果需要返回所有隐藏状态
                hidden_states_pool.append(hidden_states) # 添加到隐藏状态池
        if return_all_hidden_states: # 如果需要返回所有隐藏状态
            return hidden_states_pool # 返回所有隐藏状态
        return hidden_states # 返回最后一层隐藏状态


class CLIPTextTransformer(nn.Module): # CLIP文本Transformer类，包含嵌入层、编码器和最终归一化
    def __init__( # 初始化CLIP文本Transformer
        self,
        config: CLIPTextConfig, # CLIP文本配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        embed_dim = config.hidden_size # 嵌入维度
        self.embeddings = CLIPTextEmbeddings(config) # 文本嵌入层
        self.encoder = CLIPEncoder( # CLIP编码器
            config=config, # 配置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("encoder", prefix), # 参数名前缀
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps) # 最终层归一化

    @property
    def device(self) -> torch.device: # 获取模型所在设备
        return self.encoder.layers[0].layer_norm1.weight.device # 返回第一层归一化权重所在设备

    def forward( # 前向传播：嵌入→编码→归一化
        self,
        input_ids: torch.Tensor, # 输入token ID
        attention_mask: Optional[torch.Tensor] = None, # 注意力掩码（可选）
        position_ids: Optional[torch.Tensor] = None, # 位置ID（可选）
    ):
        input_shape = input_ids.size() # 获取输入形状
        input_ids = input_ids.view(-1, input_shape[-1]) # 重塑输入为2D
        hidden_states = self.embeddings(input_ids, position_ids) # 通过嵌入层
        causal_attention_mask = _create_4d_causal_attention_mask( # 创建4D因果注意力掩码
            input_ids.shape, hidden_states.dtype, device=hidden_states.device # 输入形状、数据类型和设备
        )
        encoder_outputs = self.encoder( # 通过编码器
            hidden_states, attention_mask, causal_attention_mask # 隐藏状态和掩码
        )
        last_hidden_state = self.final_layer_norm(encoder_outputs) # 通过最终层归一化
        return last_hidden_state # 返回最终隐藏状态


class CLIPTextModel(nn.Module): # CLIP文本模型类，包装CLIPTextTransformer
    def __init__( # 初始化CLIP文本模型
        self,
        config: CLIPTextConfig, # CLIP文本配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.text_model = CLIPTextTransformer( # 文本Transformer
            config=config, # 配置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("text_model", prefix), # 参数名前缀
        )

    def forward( # 前向传播：通过文本Transformer
        self,
        input_ids: torch.Tensor, # 输入token ID
        position_ids: torch.Tensor, # 位置ID
    ):
        return self.text_model(input_ids, position_ids) # 通过文本模型并返回


class CLIPVisionTransformer(nn.Module): # CLIP视觉Transformer类，包含视觉嵌入、编码器和归一化

    def __init__( # 初始化CLIP视觉Transformer
        self,
        config: CLIPVisionConfig, # CLIP视觉配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ) -> None:
        super().__init__() # 调用父类初始化

        self.config = config # 保存配置
        embed_dim = config.hidden_size # 嵌入维度

        self.embeddings = CLIPVisionEmbeddings(config) # 视觉嵌入层

        # NOTE: This typo of "layrnorm" is not fixed on purpose to match  # 注意：此"layrnorm"拼写错误是故意保留的，以匹配
        # the original transformers code and name of the model weights.  # 原始transformers代码和模型权重名称
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps) # 预归一化层

        self.encoder = CLIPEncoder( # CLIP编码器
            config=config, # 配置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("encoder", prefix), # 参数名前缀
        )

        num_hidden_layers = config.num_hidden_layers # 隐藏层数
        if len(self.encoder.layers) > config.num_hidden_layers: # 如果编码器层数超过配置
            raise ValueError( # 抛出异常
                f"The original encoder only has {num_hidden_layers} " # 原始编码器只有
                f"layers, but you requested {len(self.encoder.layers)} layers." # 层，但请求了层
            )

        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps) # 后归一化层

    @property
    def device(self) -> torch.device: # 获取模型所在设备
        return self.encoder.layers[0].layer_norm1.weight.device # 返回第一层归一化权重所在设备

    def forward( # 前向传播：嵌入→预归一化→编码→后归一化
        self,
        pixel_values: torch.Tensor, # 像素值张量
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values.to(self.device)) # 通过嵌入层
        hidden_states = self.pre_layrnorm(hidden_states) # 通过预归一化

        return_all_hidden_states = False # 不返回所有隐藏状态

        last_hidden_state = self.encoder( # 通过编码器
            inputs_embeds=hidden_states, # 嵌入后的隐藏状态
            return_all_hidden_states=return_all_hidden_states, # 是否返回所有隐藏状态
        )

        last_hidden_state = self.post_layernorm(last_hidden_state) # 通过后归一化

        return last_hidden_state # 返回最终隐藏状态


class CLIPVisionModel(nn.Module): # CLIP视觉模型类，包装CLIPVisionTransformer
    def __init__( # 初始化CLIP视觉模型
        self,
        config: CLIPVisionConfig, # CLIP视觉配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.vision_model = CLIPVisionTransformer( # 视觉Transformer
            config, quant_config, prefix=add_prefix("vision_model", prefix) # 配置、量化配置和前缀
        )

    @property
    def device(self) -> torch.device: # 获取模型所在设备
        return self.vision_model.device # 返回视觉Transformer所在设备

    def forward(self, pixel_values: torch.Tensor): # 前向传播：通过视觉Transformer
        return self.vision_model(pixel_values) # 通过视觉模型并返回


class CLIPModel(nn.Module): # CLIP模型主类，组合文本模型和视觉模型
    def __init__( # 初始化CLIP模型
        self,
        config: CLIPConfig, # CLIP配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        if not isinstance(config.text_config, CLIPTextConfig): # 检查文本配置类型
            raise TypeError( # 抛出类型错误
                "config.text_config is expected to be of type CLIPTextConfig but is of type" # config.text_config应为CLIPTextConfig类型但实际为
                f" {type(config.text_config)}." # 实际类型
            )

        if not isinstance(config.vision_config, CLIPVisionConfig): # 检查视觉配置类型
            raise TypeError( # 抛出类型错误
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type" # config.vision_config应为CLIPVisionConfig类型但实际为
                f" {type(config.vision_config)}." # 实际类型
            )

        text_config = config.text_config # 获取文本配置
        vision_config = config.vision_config # 获取视觉配置

        self.projection_dim = config.projection_dim # 投影维度
        self.text_embed_dim = text_config.hidden_size # 文本嵌入维度
        self.vision_embed_dim = vision_config.hidden_size # 视觉嵌入维度
        self.visual_projection = nn.Linear( # 视觉投影层
            self.vision_embed_dim, self.projection_dim, bias=False # 从视觉嵌入维度到投影维度，无偏置
        )
        self.text_projection = nn.Linear( # 文本投影层
            self.text_embed_dim, self.projection_dim, bias=False # 从文本嵌入维度到投影维度，无偏置
        )
        self.logit_scale = nn.Parameter( # 对数缩放参数
            torch.tensor(self.config.logit_scale_init_value) # 初始化值
        )

        text_model = CLIPTextModel( # 创建文本模型
            text_config, quant_config, prefix=add_prefix("text_model", prefix) # 配置、量化配置和前缀
        )
        vision_model = CLIPVisionModel( # 创建视觉模型
            vision_config, quant_config, prefix=add_prefix("vision_model", prefix) # 配置、量化配置和前缀
        )
        self.text_model = text_model.text_model # 提取内部文本Transformer
        self.vision_model = vision_model.vision_model # 提取内部视觉Transformer
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True) # 池化层（最后token，归一化）
        monkey_patch_weight_loader() # 猴子补丁权重加载器

    def forward( # 前向传播：根据输入类型处理图像或文本
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置索引
        forward_batch: ForwardBatch, # 前向批次信息
        get_embedding: bool = True, # 是否获取嵌入表示
    ):
        assert get_embedding, "CLIPEmbeddingModel is only used for embedding" # 断言必须获取嵌入，CLIP嵌入模型仅用于嵌入
        mm_inputs = [] # 多模态输入列表
        if forward_batch.mm_inputs is not None: # 如果有多模态输入
            mm_inputs = forward_batch.mm_inputs # 获取多模态输入
        pixel_values_list = [ # 像素值列表
            item.feature # 获取每个多模态项的特征
            for item in flatten_nested_list( # 展平嵌套列表
                [mm_input.mm_items for mm_input in mm_inputs if mm_input is not None] # 获取非空多模态输入的子项
            )
        ]
        if len(pixel_values_list) != 0: # 如果有图像输入
            pixel_values = torch.concat(pixel_values_list) # 拼接所有像素值
            vision_outputs = self.vision_model(pixel_values) # 通过视觉模型
            pooled_output = vision_outputs[:, 0, :] # 取CLS token的输出
            image_embeds = self.visual_projection(pooled_output) # 通过视觉投影层
            image_embeds = nn.functional.normalize(image_embeds, p=2, dim=1) # L2归一化
            return EmbeddingPoolerOutput(embeddings=image_embeds) # 返回图像嵌入

        else: # 否则处理文本输入
            text_outputs = self.text_model(input_ids, position_ids=positions) # 通过文本模型
            pooled_output = self.pooler(text_outputs[0], forward_batch) # 通过池化层
            return EmbeddingPoolerOutput( # 返回文本嵌入
                embeddings=self.text_projection(pooled_output.embeddings) # 通过文本投影层
            )

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs): # 填充输入ID（CLIP不需要填充）
        # Clip embeddings models handle text/image separately, so we don't need to pad input ids  # CLIP嵌入模型分别处理文本/图像，因此不需要填充输入ID
        return input_ids # 直接返回原始输入ID

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载模型权重
        stacked_params_mapping = [ # 堆叠参数映射表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"), # Q投影映射
            ("qkv_proj", "k_proj", "k"), # K投影映射
            ("qkv_proj", "v_proj", "v"), # V投影映射
        ]
        params_dict = dict(self.named_parameters()) # 获取参数字典
        for name, loaded_weight in weights: # 遍历所有权重
            if "position_ids" in name: # 如果是位置ID
                continue # 跳过
            if "out_proj" in name: # 如果是输出投影
                name = name.replace("out_proj", "proj") # 替换为proj
            for param_name, shard_name, shard_id in stacked_params_mapping: # 遍历堆叠参数映射
                if shard_name not in name: # 如果分片名不在参数名中
                    continue # 跳过
                name = name.replace(shard_name, param_name) # 替换分片名为参数名
                param = params_dict[name] # 获取参数
                weight_loader = param.weight_loader # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id) # 加载权重分片
                break # 跳出内层循环
            else: # 如果没有匹配到堆叠参数
                param = params_dict[name] # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取权重加载器
                weight_loader(param, loaded_weight) # 加载权重


# monkey patch weight loader to remove open_clip file  # 猴子补丁权重加载器，移除open_clip文件
def monkey_patch_weight_loader(): # 猴子补丁函数，修改默认权重加载器以过滤open_clip文件
    import glob # 导入glob模块用于文件匹配
    import os # 导入操作系统模块

    from sglang.srt.model_loader.loader import DefaultModelLoader # 导入默认模型加载器
    from sglang.srt.model_loader.weight_utils import ( # 导入权重工具
        download_weights_from_hf, # 从HuggingFace下载权重
        filter_files_not_needed_for_inference, # 过滤推理不需要的文件
    )

    def prepare_weights( # 准备权重文件
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool # 模型路径、版本号、是否回退到pt
    ) -> Tuple[str, List[str], bool]:
        model_name_or_path = ( # 模型名称或路径
            self._maybe_download_from_modelscope(model_name_or_path, revision) # 尝试从ModelScope下载
            or model_name_or_path # 或使用原始路径
        )

        is_local = os.path.isdir(model_name_or_path) # 是否为本地路径
        use_safetensors = False # 不使用safetensors格式
        allow_patterns = ["*.bin"] # 允许的文件模式

        if not is_local: # 如果不是本地路径
            hf_folder = download_weights_from_hf( # 从HuggingFace下载权重
                model_name_or_path, # 模型名称或路径
                self.load_config.download_dir, # 下载目录
                allow_patterns, # 允许的文件模式
                revision, # 版本号
                ignore_patterns=self.load_config.ignore_patterns, # 忽略的模式
            )
        else: # 否则
            hf_folder = model_name_or_path # 使用本地路径

        hf_weights_files: List[str] = [] # HuggingFace权重文件列表
        for pattern in allow_patterns: # 遍历允许的模式
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern)) # 匹配文件

        hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files) # 过滤推理不需要的文件

        # remove open_clip file  # 移除open_clip文件
        hf_weights_files = [ # 过滤权重文件列表
            file for file in hf_weights_files if "open_clip" not in file # 排除包含open_clip的文件
        ]

        if len(hf_weights_files) == 0: # 如果没有找到权重文件
            raise RuntimeError( # 抛出运行时错误
                f"Cannot find any model weights with `{model_name_or_path}`" # 找不到模型权重
            )

        return hf_folder, hf_weights_files, use_safetensors # 返回文件夹、权重文件列表和格式标志

    setattr(DefaultModelLoader, "_prepare_weights", prepare_weights) # 设置_prepare_weights方法


EntryClass = CLIPModel # 入口类，用于模型注册
