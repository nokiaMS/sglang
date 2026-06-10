# InternVL 视觉语言多模态条件生成模型
# 实现了 InternVision 视觉编码器、InternVL 聊天模型及权重加载逻辑
# 支持 InternLM2、Qwen2、Qwen3、Qwen3MoE、GptOss 等多种语言模型后端
# SPDX-License-Identifier: Apache-2.0  # Apache 2.0 许可证声明
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # vLLM 项目版权声明
from typing import Iterable, List, Optional, Tuple, Union  # 导入类型提示工具

import torch  # 导入 PyTorch 深度学习框架

# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/7f62077af5159c625fe3ad1c812e6c1a2b93ba3b/vllm/model_executor/models/internlm2.py  # 适配自 vLLM 的 InternLM2 模型实现
# Adapted from https://raw.githubusercontent.com/hehesangsj/sglang/refs/heads/internvl/python/sglang/srt/models/internvl.py  # 适配自 SGLang 的 InternVL 模型实现
import torch.nn.functional as F  # 导入 PyTorch 函数式神经网络模块
from torch import nn  # 导入 PyTorch 神经网络模块
from transformers import PretrainedConfig, PreTrainedModel  # 导入 transformers 配置类和预训练模型基类
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling  # 导入模型输出类

from sglang.srt.distributed import (  # 导入分布式训练相关工具
    get_tensor_model_parallel_rank,  # 获取张量并行排名
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
)
from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.layers.activation import get_act_fn  # 导入激活函数获取工具
from sglang.srt.layers.attention import vision_utils  # 导入视觉注意力工具
from sglang.srt.layers.attention.vision import SingletonCache, VisionAttention  # 导入单例缓存和视觉注意力类
from sglang.srt.layers.conv import Conv2dLayer  # 导入 2D 卷积层
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear  # 导入列并行和行并行线性层
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合 MoE 层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternTokenPairs,  # 多模态数据填充模式（token 对）
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息类
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.deepseek_janus_pro import DropPath  # 导入 DropPath（随机深度）模块
from sglang.srt.models.gpt_oss import GptOssForCausalLM  # 导入 GptOss 因果语言模型
from sglang.srt.models.internlm2 import InternLM2ForCausalLM  # 导入 InternLM2 因果语言模型
from sglang.srt.models.qwen2 import Qwen2ForCausalLM  # 导入 Qwen2 因果语言模型
from sglang.srt.models.qwen3 import Qwen3ForCausalLM  # 导入 Qwen3 因果语言模型
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM  # 导入 Qwen3 MoE 因果语言模型
from sglang.srt.multimodal.internvl_vit_cuda_graph_runner import (  # 导入 InternViT CUDA Graph 运行器
    InternViTCudaGraphRunner,
)
from sglang.srt.multimodal.mm_utils import run_dp_sharded_vision_model  # 导入数据并行分片视觉模型运行函数
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数
from sglang.srt.utils import is_cuda  # 导入 CUDA 可用性检查函数
from sglang.utils import logger  # 导入日志记录器

_is_cuda = is_cuda()  # 检查当前是否支持 CUDA


class InternAttention(nn.Module):  # Intern 注意力模块
    def __init__(  # 初始化方法
        self,
        config,  # 模型配置
        quant_config: QuantizationConfig = None,  # 量化配置，默认为 None
        use_data_parallel: bool = False,  # 是否使用数据并行，默认为 False
        aux_stream: Optional[torch.cuda.Stream] = None,  # 辅助 CUDA 流，可选
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.embed_dim = config.hidden_size  # 嵌入维度等于隐藏层大小
        self.num_heads = config.num_attention_heads  # 注意力头数
        self.head_dim = self.embed_dim // self.num_heads  # 每个头的维度
        self.scale = self.head_dim**-0.5  # 缩放因子（头维度的 -0.5 次方）

        self.attn = VisionAttention(  # 创建视觉注意力实例
            embed_dim=self.embed_dim,  # 嵌入维度
            num_heads=self.num_heads,  # 注意力头数
            projection_size=self.embed_dim,  # 投影大小
            use_qkv_parallel=True,  # 使用 QKV 并行
            quant_config=quant_config,  # 量化配置
            dropout=getattr(config, "dropout", 0.0),  # Dropout 率，默认 0.0
            qkv_bias=getattr(config, "qkv_bias", False)  # QKV 偏置，默认 False
            or getattr(config, "attention_bias", False),  # 或者从 attention_bias 获取
            num_dummy_heads=getattr(config, "num_dummy_heads", 0),  # 虚拟头数，默认 0
            qk_normalization=getattr(config, "qk_normalization", False)  # QK 归一化，默认 False
            or getattr(config, "use_qk_norm", False),  # 或者从 use_qk_norm 获取
            flatten_batch=False,  # 不展平批次
            use_data_parallel=use_data_parallel,  # 是否使用数据并行
            aux_stream=aux_stream,  # 辅助 CUDA 流
        )

        self.proj_drop = nn.Dropout(config.dropout)  # 创建投影 Dropout 层

    def forward(  # 前向传播方法
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        cu_seqlens: torch.Tensor,  # 累积序列长度张量
        output_ws: Optional[torch.Tensor] = None,  # 输出工作空间张量，可选
    ) -> torch.Tensor:
        out = self.attn(hidden_states, cu_seqlens=cu_seqlens, output_ws=output_ws)  # 计算注意力输出
        outs = self.proj_drop(out)  # 应用投影 Dropout
        return outs  # 返回输出


class InternVisionEmbeddings(nn.Module):  # Intern 视觉嵌入层
    def __init__(self, config: PretrainedConfig):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.embed_dim = config.hidden_size  # 嵌入维度
        self.image_size = (  # 图像大小
            config.image_size  # 如果图像大小是整数直接使用
            if isinstance(config.image_size, int)
            else config.image_size[0]  # 否则取第一个元素
        )
        self.patch_size = (  # patch 大小
            config.patch_size  # 如果 patch 大小是整数直接使用
            if isinstance(config.patch_size, int)
            else config.patch_size[0]  # 否则取第一个元素
        )

        self.class_embedding = nn.Parameter(  # 类别嵌入参数（CLS token）
            torch.randn(1, 1, self.embed_dim),  # 形状为 [1, 1, embed_dim]
        )

        self.patch_embedding = Conv2dLayer(  # patch 嵌入卷积层
            in_channels=3,  # 输入通道数（RGB）
            out_channels=self.embed_dim,  # 输出通道数等于嵌入维度
            kernel_size=self.patch_size,  # 卷积核大小等于 patch 大小
            stride=self.patch_size,  # 步幅等于 patch 大小
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  # patch 数量
        self.num_positions = self.num_patches + 1  # 位置数（patch 数 + 1 个 CLS token）

        self.position_embedding = nn.Parameter(  # 位置嵌入参数
            torch.randn(1, self.num_positions, self.embed_dim)  # 形状为 [1, 位置数, 嵌入维度]
        )

    def _get_pos_embed(self, pos_embed, H, W):  # 获取指定高度和宽度的位置嵌入（插值）
        target_dtype = pos_embed.dtype  # 目标数据类型
        pos_embed = (  # 重塑并转置位置嵌入
            pos_embed.float()  # 转为浮点数
            .reshape(  # 重塑形状
                1,
                self.image_size // self.patch_size,  # 高度方向 patch 数
                self.image_size // self.patch_size,  # 宽度方向 patch 数
                -1,  # 嵌入维度
            )
            .permute(0, 3, 1, 2)  # 转置为 [1, embed_dim, h, w]
        )
        pos_embed = (  # 双三次插值到目标大小
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)  # 双三次插值
            .reshape(1, -1, H * W)  # 重塑为 [1, embed_dim, H*W]
            .permute(0, 2, 1)  # 转置为 [1, H*W, embed_dim]
            .to(target_dtype)  # 转回目标数据类型
        )
        return pos_embed  # 返回位置嵌入

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:  # 前向传播方法
        target_dtype = self.patch_embedding.weight.dtype  # 获取目标数据类型
        patch_embeds = self.patch_embedding(  # 计算 patch 嵌入
            pixel_values
        )  # shape = [*, channel, width, height]  # 输出形状为 [批次, 通道, 宽, 高]
        batch_size, _, height, width = patch_embeds.shape  # 获取批次大小和特征图尺寸
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # 展平并转置为 [批次, patch数, 嵌入维度]
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)  # 扩展 CLS 嵌入到批次大小
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # 拼接 CLS 和 patch 嵌入
        position_embedding = torch.cat(  # 拼接 CLS 和 patch 位置嵌入
            [
                self.position_embedding[:, :1, :],  # CLS 位置嵌入
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),  # patch 位置嵌入（插值后）
            ],
            dim=1,
        )
        embeddings = embeddings + position_embedding.to(target_dtype)  # 加上位置嵌入
        return embeddings  # 返回嵌入结果


class InternRMSNorm(nn.Module):  # Intern RMS 归一化层
    def __init__(self, hidden_size, eps=1e-6):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习的缩放参数，初始化为 1
        self.variance_epsilon = eps  # 方差 epsilon，防止除零

    def forward(self, hidden_states):  # 前向传播方法
        input_dtype = hidden_states.dtype  # 保存输入数据类型
        hidden_states = hidden_states.to(torch.float32)  # 转为 float32 计算方差
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # 计算方差（平方的均值）
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # 乘以方差的逆平方根进行归一化
        return self.weight * hidden_states.to(input_dtype)  # 乘以缩放参数并转回原始数据类型


class InternMLP(nn.Module):  # Intern MLP（多层感知机）模块
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        use_data_parallel: bool = False,  # 是否使用数据并行，默认为 False
    ):
        super().__init__()  # 调用父类初始化
        self.tp_size = (  # 张量并行大小
            1 if use_data_parallel else get_tensor_model_parallel_world_size()  # 数据并行时为 1，否则为张量并行世界大小
        )
        self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()  # 张量并行排名
        self.config = config  # 保存配置
        self.act = get_act_fn(config.hidden_act)  # 获取激活函数
        self.fc1 = ColumnParallelLinear(  # 第一个全连接层（列并行）
            config.hidden_size,  # 输入维度
            config.intermediate_size,  # 输出维度（中间层大小）
            bias=True,  # 使用偏置
            quant_config=None,  # 不使用量化
            tp_size=self.tp_size,  # 张量并行大小
            tp_rank=self.tp_rank,  # 张量并行排名
        )
        self.fc2 = RowParallelLinear(  # 第二个全连接层（行并行）
            config.intermediate_size,  # 输入维度（中间层大小）
            config.hidden_size,  # 输出维度
            bias=True,  # 使用偏置
            quant_config=None,  # 不使用量化
            tp_size=self.tp_size,  # 张量并行大小
            tp_rank=self.tp_rank,  # 张量并行排名
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        hidden_states, _ = self.fc1(hidden_states)  # 通过第一个全连接层
        hidden_states = self.act(hidden_states)  # 应用激活函数
        hidden_states, _ = self.fc2(hidden_states)  # 通过第二个全连接层
        return hidden_states  # 返回输出


NORM2FN = {  # 归一化函数映射字典
    "rms_norm": InternRMSNorm,  # RMS 归一化
    "layer_norm": nn.LayerNorm,  # 层归一化
}


class InternVisionEncoderLayer(nn.Module):  # Intern 视觉编码器层

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        drop_path_rate: float,  # DropPath 丢弃率
        quant_config: QuantizationConfig = None,  # 量化配置，默认为 None
        use_data_parallel: bool = False,  # 是否使用数据并行，默认为 False
        aux_stream: Optional[torch.cuda.Stream] = None,  # 辅助 CUDA 流，可选
    ):
        super().__init__()  # 调用父类初始化
        self.embed_dim = config.hidden_size  # 嵌入维度
        self.intermediate_size = config.intermediate_size  # 中间层大小
        self.norm_type = config.norm_type  # 归一化类型
        self.attn = InternAttention(  # 创建注意力模块
            config=config,
            quant_config=quant_config,
            use_data_parallel=use_data_parallel,
            aux_stream=aux_stream,
        )
        self.mlp = InternMLP(config, use_data_parallel)  # 创建 MLP 模块
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)  # 注意力前的归一化层
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)  # MLP 前的归一化层

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))  # 层缩放参数 1
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))  # 层缩放参数 2
        self.drop_path1 = (  # DropPath 层 1
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()  # 丢弃率大于 0 使用 DropPath，否则使用恒等映射
        )
        self.drop_path2 = (  # DropPath 层 2
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()  # 丢弃率大于 0 使用 DropPath，否则使用恒等映射
        )

    def forward(  # 前向传播方法
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        cu_seqlens: torch.Tensor,  # 累积序列长度张量
        output_ws: Optional[torch.Tensor] = None,  # 输出工作空间张量，可选
    ) -> Tuple[  # 返回类型为元组
        torch.FloatTensor,
        Optional[torch.FloatTensor],
        Optional[Tuple[torch.FloatTensor]],
    ]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`  # 输入到层的隐藏状态，形状为 (批次, 序列长度, 嵌入维度)
        """  # 参数说明

        hidden_states = hidden_states + self.drop_path1(  # 残差连接 + DropPath
            self.attn(  # 计算注意力
                self.norm1(hidden_states).to(hidden_states.dtype),  # 先归一化再计算注意力
                cu_seqlens=cu_seqlens,
                output_ws=output_ws,
            )
            * self.ls1  # 乘以层缩放参数
        )

        hidden_states = hidden_states + self.drop_path2(  # 残差连接 + DropPath
            self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * self.ls2  # 先归一化再通过 MLP，乘以层缩放参数
        )

        return hidden_states  # 返回隐藏状态


class InternVisionEncoder(nn.Module):  # Intern 视觉编码器
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a  # Transformer 编码器，由 config.num_hidden_layers 个自注意力层组成。每层是一个
    [`InternEncoderLayer`].  # InternEncoderLayer

    Args:  # 参数
        config (`InternConfig`):  # 配置（InternConfig 类型）
            The corresponding vision configuration for the `InternEncoder`.  # InternEncoder 对应的视觉配置
    """

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        use_data_parallel: bool = False,  # 是否使用数据并行，默认为 False
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        # stochastic depth decay rule  # 随机深度衰减规则
        dpr = [  # 计算每层的 DropPath 丢弃率
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)  # 从 0 线性增加到配置的丢弃率
        ]

        self.enable_cg = _is_cuda and envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get()  # 是否启用 CUDA Graph
        aux_stream = (  # 辅助 CUDA 流
            None if self.enable_cg else (torch.cuda.Stream() if _is_cuda else None)  # 启用 CUDA Graph 时不使用辅助流
        )
        self.layers = nn.ModuleList(  # 创建编码器层列表
            [
                InternVisionEncoderLayer(
                    config, dpr[idx], quant_config, use_data_parallel, aux_stream  # 传入配置和每层的丢弃率
                )
                for idx in range(config.num_hidden_layers)  # 遍历所有层
            ]
        )

        self.cuda_graph_runner: Optional[InternViTCudaGraphRunner] = None  # CUDA Graph 运行器，初始为 None
        if self.enable_cg:  # 如果启用 CUDA Graph
            self.cuda_graph_runner = InternViTCudaGraphRunner(self)  # 创建 CUDA Graph 运行器

    def forward(  # 前向传播方法
        self,
        inputs_embeds,  # 输入嵌入
        cu_seqlens=None,  # 累积序列长度，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典，可选
    ) -> Union[Tuple, BaseModelOutput]:  # 返回元组或 BaseModelOutput
        r"""
        Args:  # 参数说明
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):  # 输入嵌入，形状为 (批次大小, 序列长度, 隐藏大小)
                Embedded representation of the inputs. Should be float, not int tokens.  # 输入的嵌入表示。应为浮点数，不是整数 token
            output_hidden_states (`bool`, *optional*):  # 是否输出所有层的隐藏状态，可选
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors  # 是否返回所有层的隐藏状态。详见返回张量中的 hidden_states
                for more detail.  # 更多细节
            return_dict (`bool`, *optional*):  # 是否返回字典，可选
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.  # 是否返回 ModelOutput 而不是普通元组
        """
        if self.enable_cg and (not output_hidden_states):  # 如果启用 CUDA Graph 且不需要输出隐藏状态
            # graph path only returns last_hidden_state  # Graph 路径仅返回最后隐藏状态
            hidden_states = inputs_embeds.to(device=inputs_embeds.device).contiguous()  # 确保输入是连续的
            hidden_states = self.cuda_graph_runner.run(hidden_states)  # 通过 CUDA Graph 运行
            if not return_dict:  # 如果不返回字典
                return (hidden_states,)  # 返回元组
            return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=None)  # 返回 BaseModelOutput

        output_hidden_states = (  # 确定是否输出隐藏状态
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states  # 默认使用配置中的值
        )
        return_dict = (  # 确定是否返回字典
            return_dict if return_dict is not None else self.config.use_return_dict  # 默认使用配置中的值
        )

        encoder_states = () if output_hidden_states else None  # 如果输出隐藏状态则初始化为空元组
        hidden_states = inputs_embeds  # 初始化隐藏状态

        if cu_seqlens is None:  # 如果没有提供累积序列长度
            cu_seqlens = SingletonCache()  # 使用单例缓存

        for idx, encoder_layer in enumerate(self.layers):  # 遍历所有编码器层
            if output_hidden_states:  # 如果需要输出隐藏状态
                encoder_states = encoder_states + (hidden_states,)  # 记录当前隐藏状态
            layer_outputs = encoder_layer(hidden_states, cu_seqlens=cu_seqlens)  # 通过当前层
            hidden_states = layer_outputs  # 更新隐藏状态

        if output_hidden_states:  # 如果需要输出隐藏状态
            encoder_states = encoder_states + (hidden_states,)  # 记录最后一层的隐藏状态

        if not return_dict:  # 如果不返回字典
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)  # 返回非空值的元组
        return BaseModelOutput(  # 返回 BaseModelOutput
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )


class InternVisionModel(PreTrainedModel):  # Intern 视觉模型，继承自 PreTrainedModel
    main_input_name = "pixel_values"  # 主输入名称为 pixel_values
    _supports_flash_attn_2 = True  # 支持 Flash Attention 2
    config_class = PretrainedConfig  # 配置类
    _no_split_modules = ["InternVisionEncoderLayer"]  # 不可分割的模块列表

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        use_data_parallel: bool = False,  # 是否使用数据并行，默认为 False
    ):
        super().__init__(config)  # 调用父类初始化

        self.config = config  # 保存配置
        self.use_data_parallel = use_data_parallel  # 保存数据并行标志
        self.embeddings = InternVisionEmbeddings(  # 创建视觉嵌入层
            config,
        )
        self.encoder = InternVisionEncoder(config, quant_config, use_data_parallel)  # 创建视觉编码器

    def resize_pos_embeddings(self, old_size, new_size, patch_size):  # 调整位置嵌入大小
        pos_emb = self.embeddings.position_embedding  # 获取位置嵌入参数
        _, num_positions, embed_dim = pos_emb.shape  # 获取位置嵌入的形状
        cls_emb = pos_emb[:, :1, :]  # 提取 CLS 位置嵌入
        pos_emb = (  # 重塑 patch 位置嵌入
            pos_emb[:, 1:, :]
            .reshape(1, old_size // patch_size, old_size // patch_size, -1)
            .permute(0, 3, 1, 2)  # 转置为 [1, embed_dim, h, w]
        )
        pos_emb = F.interpolate(  # 双三次插值到新大小
            pos_emb.float(),
            size=new_size // patch_size,
            mode="bicubic",
            align_corners=False,
        )
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)  # 调整形状和数据类型
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)  # 拼接 CLS 和 patch 位置嵌入
        self.embeddings.position_embedding = nn.Parameter(pos_emb)  # 更新位置嵌入参数
        self.embeddings.image_size = new_size  # 更新图像大小
        logger.info(  # 记录位置嵌入调整信息
            "Resized position embeddings from {} to {}".format(old_size, new_size)
        )

    def get_input_embeddings(self):  # 获取输入嵌入层
        return self.embeddings  # 返回嵌入层

    def forward(  # 前向传播方法
        self,
        pixel_values: Optional[torch.FloatTensor] = None,  # 像素值，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典，可选
        pixel_embeds: Optional[torch.FloatTensor] = None,  # 像素嵌入，可选
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)  # 将像素值转移到设备和数据类型
        output_hidden_states = (  # 确定是否输出隐藏状态
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (  # 确定是否返回字典
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None and pixel_embeds is None:  # 如果像素值和像素嵌入都为空
            raise ValueError("You have to specify pixel_values or pixel_embeds")  # 抛出异常

        if pixel_embeds is not None:  # 如果提供了像素嵌入
            hidden_states = pixel_embeds  # 直接使用像素嵌入
        else:  # 否则通过嵌入层处理像素值
            if len(pixel_values.shape) == 4:  # 如果像素值是 4D 张量 [N, C, H, W]
                hidden_states = self.embeddings(pixel_values)  # 通过嵌入层计算隐藏状态
            else:
                raise ValueError(f"wrong pixel_values size: {pixel_values.shape}")  # 抛出异常

        if self.use_data_parallel:  # 如果使用数据并行
            encoder_outputs = run_dp_sharded_vision_model(hidden_states, self.encoder)  # 运行数据并行分片视觉模型
            last_hidden_state = encoder_outputs  # 获取最后隐藏状态
        else:  # 不使用数据并行
            encoder_outputs = self.encoder(  # 通过编码器
                inputs_embeds=hidden_states,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            last_hidden_state = encoder_outputs.last_hidden_state  # 获取最后隐藏状态
        pooled_output = last_hidden_state[:, 0, :]  # 池化输出取 CLS token（第 0 个位置）

        if not return_dict:  # 如果不返回字典
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]  # 返回元组

        if self.use_data_parallel:  # 如果使用数据并行
            return BaseModelOutputWithPooling(  # 返回带池化的模型输出
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=None,
                attentions=None,
            )
        else:  # 不使用数据并行
            return BaseModelOutputWithPooling(  # 返回带池化的模型输出（包含隐藏状态）
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )


class InternVLChatModel(nn.Module):  # InternVL 聊天模型
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        use_flash_attn=True,  # 是否使用 Flash Attention，默认为 True
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder  # 获取数据并行编码器标志
        self.quant_config = quant_config  # 保存量化配置
        vision_utils.update_vit_attn_dummy_heads_config(self.config)  # 更新 ViT 注意力虚拟头配置
        image_size = config.force_image_size or config.vision_config.image_size  # 获取强制图像大小或视觉配置中的图像大小
        patch_size = config.vision_config.patch_size  # 获取 patch 大小
        self.patch_size = patch_size  # 保存 patch 大小
        self.select_layer = config.select_layer  # 保存特征选择层
        self.template = config.template  # 保存模板
        self.num_image_token = int(  # 计算图像 token 数量
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio  # 保存下采样比率
        self.ps_version = config.ps_version  # 保存 pixel shuffle 版本

        config.vision_config.use_flash_attn = True if use_flash_attn else False  # 设置视觉配置的 Flash Attention
        config.llm_config._attn_implementation = (  # 设置语言模型的注意力实现
            "flash_attention_2" if use_flash_attn else "eager"  # 使用 Flash Attention 2 或 eager 模式
        )

        logger.info(f"num_image_token: {self.num_image_token}")  # 记录图像 token 数量
        logger.info(f"ps_version: {self.ps_version}")  # 记录 pixel shuffle 版本

        self.vision_model = InternVisionModel(  # 创建视觉模型
            config.vision_config,
            use_data_parallel=self.use_data_parallel,
        )
        if config.llm_config.architectures[0] == "Qwen2ForCausalLM":  # 如果语言模型是 Qwen2
            self.language_model = Qwen2ForCausalLM(  # 创建 Qwen2 因果语言模型
                config=config.llm_config, quant_config=quant_config
            )
        elif config.llm_config.architectures[0] == "InternLM2ForCausalLM":  # 如果语言模型是 InternLM2
            self.language_model = InternLM2ForCausalLM(  # 创建 InternLM2 因果语言模型
                config=config.llm_config, quant_config=quant_config
            )
        elif config.llm_config.architectures[0] == "Qwen3MoeForCausalLM":  # 如果语言模型是 Qwen3MoE
            self.language_model = Qwen3MoeForCausalLM(  # 创建 Qwen3MoE 因果语言模型
                config=config.llm_config, quant_config=quant_config
            )
        elif config.llm_config.architectures[0] == "GptOssForCausalLM":  # 如果语言模型是 GptOss
            self.language_model = GptOssForCausalLM(  # 创建 GptOss 因果语言模型
                config=config.llm_config, quant_config=quant_config
            )
        elif config.llm_config.architectures[0] == "Qwen3ForCausalLM":  # 如果语言模型是 Qwen3
            self.language_model = Qwen3ForCausalLM(  # 创建 Qwen3 因果语言模型
                config=config.llm_config, quant_config=quant_config
            )
        else:  # 其他不支持的语言模型
            raise NotImplementedError(  # 抛出未实现异常
                f"{config.llm_config.architectures[0]} is not implemented."
            )

        vit_hidden_size = config.vision_config.hidden_size  # 获取 ViT 隐藏层大小
        llm_hidden_size = config.llm_config.hidden_size  # 获取语言模型隐藏层大小

        self.mlp1 = nn.Sequential(  # 创建多模态投影 MLP
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),  # 层归一化
            nn.Linear(  # 线性层（ViT 隐藏层到语言模型隐藏层）
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),  # GELU 激活函数
            nn.Linear(llm_hidden_size, llm_hidden_size),  # 线性层（语言模型隐藏层到语言模型隐藏层）
        )

        self.external_mm_data_embedding_funcs = {  # 外部多模态数据嵌入函数映射
            Modality.IMAGE: self.get_image_feature,  # 图像模态对应图像特征提取函数
            Modality.VIDEO: self.get_video_feature,  # 视频模态对应视频特征提取函数
        }

        self.model = self.language_model.model  # 获取语言模型的模型部分

    def pixel_shuffle(self, x, scale_factor=0.5):  # 像素重排（Pixel Shuffle）操作
        n, w, h, c = x.size()  # 获取输入尺寸
        # N, W, H, C --> N, W, H * scale, C // scale  # 第一步：高度扩展，通道缩减
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale  # 第二步：交换宽高维度
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)  # 第三步：宽度扩展，通道进一步缩减
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":  # 如果是 v1 版本
            logger.warn(  # 警告 v1 版本存在转置问题
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:  # v2 及以上版本
            x = x.permute(0, 2, 1, 3).contiguous()  # 交换宽高维度回来
        return x  # 返回像素重排结果

    def extract_feature(self, pixel_values):  # 提取视觉特征
        if self.select_layer == -1:  # 如果选择最后一层
            vit_embeds = self.vision_model(  # 通过视觉模型获取最后隐藏状态
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:  # 如果选择指定层
            vit_embeds = self.vision_model(  # 通过视觉模型获取指定层的隐藏状态
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]  # 去除 CLS token

        h = w = int(vit_embeds.shape[1] ** 0.5)  # 计算特征图的高度和宽度
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)  # 重塑为 2D 特征图
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)  # 像素重排降采样
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])  # 展平空间维度
        vit_embeds = self.mlp1(vit_embeds)  # 通过投影 MLP
        return vit_embeds  # 返回视觉特征

    def get_image_feature(self, items: List[MultimodalDataItem]):  # 获取图像特征
        """
        Projects the last hidden state from the vision model into language model space.  # 将视觉模型的最后隐藏状态投影到语言模型空间

        Returns:  # 返回
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).  # 图像特征张量，形状为 (图像数, 图像长度, 嵌入维度)
        """
        pixel_values = torch.cat([item.feature for item in items])  # 拼接所有图像的像素值
        # If already precomputed embeddings (not raw pixel values), skip vision encoder.  # 如果已经是预计算的嵌入（不是原始像素值），跳过视觉编码器
        # Normal pixel_values are 4D [N, C, H, W]; precomputed embeddings are 2D or 3D.  # 正常的像素值是 4D [N, C, H, W]；预计算嵌入是 2D 或 3D
        if pixel_values.dim() != 4:  # 如果不是 4D 张量
            return pixel_values  # 直接返回预计算嵌入
        image_features = self.extract_feature(pixel_values)  # 提取视觉特征
        return image_features  # 返回图像特征

    def get_video_feature(self, items: List[MultimodalDataItem]):  # 获取视频特征
        # items: each item corresponds to one video (recommended)  # items: 每个项对应一个视频（推荐）
        # item.feature shape: [num_frames, 3, 448, 448]  (or [num_tiles, 3, 448, 448])  # item.feature 形状: [帧数, 3, 448, 448]（或 [瓦片数, 3, 448, 448]）
        pixel_values = torch.cat([item.feature for item in items], dim=0)  # 拼接所有视频的像素值
        # If already precomputed embeddings, skip vision encoder.  # 如果已经是预计算嵌入，跳过视觉编码器
        if pixel_values.dim() != 4:  # 如果不是 4D 张量
            return pixel_values  # 直接返回预计算嵌入
        video_features = self.extract_feature(pixel_values)  # 提取视觉特征
        return video_features  # 返回视频特征

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID 张量
        positions: torch.Tensor,  # 位置张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入张量，可选
    ) -> torch.Tensor:

        hidden_states = general_mm_embed_routine(  # 调用通用多模态嵌入例程
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            data_embedding_funcs=self.external_mm_data_embedding_funcs,
            positions=positions,
        )

        return hidden_states  # 返回隐藏状态

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):  # 填充输入 ID
        # Get all special token IDs  # 获取所有特殊 token ID
        im_start_id: int = mm_inputs.im_start_id  # 图像起始 token ID
        im_end_id: int = mm_inputs.im_end_id  # 图像结束 token ID

        media_token_pairs = [(im_start_id, im_end_id)]  # 媒体 token 对
        helper = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)  # 创建填充模式辅助器

        return helper.pad_input_tokens(input_ids, mm_inputs)  # 返回填充后的输入 token

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        expert_params_mapping = []  # 专家参数映射列表
        if "InternLM2ForCausalLM" in self.config.llm_config.architectures:  # 如果语言模型是 InternLM2
            stacked_params_mapping = [  # 堆叠参数映射
                # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片 ID)
                ("gate_up_proj", "w1", 0),  # gate 和 up 投影堆叠
                ("gate_up_proj", "w3", 1),
            ]
        elif "Qwen2ForCausalLM" in self.config.llm_config.architectures:  # 如果语言模型是 Qwen2
            stacked_params_mapping = [  # 堆叠参数映射
                # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片 ID)
                ("qkv_proj", "q_proj", "q"),  # QKV 投影堆叠
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),  # gate 和 up 投影堆叠
                ("gate_up_proj", "up_proj", 1),
            ]
        elif "Qwen3MoeForCausalLM" in self.config.llm_config.architectures:  # 如果语言模型是 Qwen3MoE
            stacked_params_mapping = [  # 堆叠参数映射
                # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片 ID)
                ("qkv_proj", "q_proj", "q"),  # QKV 投影堆叠
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),  # gate 和 up 投影堆叠
                ("gate_up_proj", "up_proj", 1),
            ]

            expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 创建专家参数映射
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
            )
        elif "Qwen3ForCausalLM" in self.config.llm_config.architectures:  # 如果语言模型是 Qwen3
            stacked_params_mapping = [  # 堆叠参数映射
                # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片 ID)
                ("qkv_proj", "q_proj", "q"),  # QKV 投影堆叠
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),  # gate 和 up 投影堆叠
                ("gate_up_proj", "up_proj", 1),
            ]

        params_dict = dict(self.named_parameters())  # 创建参数字典

        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入的逆频率
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.  # 检查点中有 mlp.experts[0].gate_proj
                # Since we handle the experts below in expert_params_mapping,  # 因为我们下面在 expert_params_mapping 中处理专家
                # we need to skip here BEFORE we update the name, otherwise  # 我们需要在更新名称之前跳过
                # name will be updated to mlp.experts[0].gate_up_proj, which  # 否则名称会被更新为 mlp.experts[0].gate_up_proj
                # will then be updated below in expert_params_mapping  # 然后会在下面的 expert_params_mapping 中被更新
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.  # 变为 mlp.experts[0].gate_gate_up_proj，导致加载失败
                if "mlp.experts" in name:  # 如果是专家层权重
                    continue  # 跳过，由专家参数映射处理
                name = name.replace(weight_name, param_name)  # 替换权重名称为参数名称
                # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break
            else:  # 如果堆叠参数映射中没有匹配
                if "vision_model" in name:  # 如果是视觉模型权重
                    # adapt to VisionAttention  # 适配到 VisionAttention
                    name = name.replace(r"attn.", r"attn.attn.")  # 替换注意力路径
                    name = name.replace(r"qkv.", r"qkv_proj.")  # 替换 QKV 投影路径

                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping  # 解包映射
                    if weight_name not in name:  # 如果权重名不在参数名中
                        continue
                    name = name.replace(weight_name, param_name)  # 替换权重名称
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(  # 加载权重
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:  # 如果专家参数映射中也没有匹配
                    # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置加载
                    if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                        continue
                    param = params_dict[name]  # 获取参数
                    if "wqkv" in name:  # 如果是融合 QKV 权重
                        config = self.config  # 获取配置
                        kv_groups = (  # 计算 KV 组数
                            config.num_attention_heads // config.num_key_value_heads
                        )
                        head_dim = config.hidden_size // config.num_attention_heads  # 计算头维度
                        loaded_weight = loaded_weight.view(  # 重塑权重形状
                            -1, 2 + kv_groups, head_dim, loaded_weight.shape[-1]
                        )
                        wq, wk, wv = torch.split(  # 分割为 Q、K、V 权重
                            loaded_weight, [kv_groups, 1, 1], dim=1
                        )
                        wq = wq.reshape(-1, wq.shape[-1])  # 重塑 Q 权重
                        wk = wk.reshape(-1, wk.shape[-1])  # 重塑 K 权重
                        wv = wv.reshape(-1, wv.shape[-1])  # 重塑 V 权重
                        weight_loader = param.weight_loader  # 获取权重加载器
                        weight_loader(param, wq, "q")  # 加载 Q 权重
                        weight_loader(param, wk, "k")  # 加载 K 权重
                        weight_loader(param, wv, "v")  # 加载 V 权重
                    else:  # 其他权重
                        weight_loader = getattr(  # 获取权重加载器
                            param, "weight_loader", default_weight_loader
                        )
                        if "vision_model" in name:  # 如果是视觉模型权重
                            loaded_weight = vision_utils.pad_vit_attn_dummy_heads(  # 填充 ViT 注意力虚拟头
                                self.config, name, loaded_weight
                            )
                        weight_loader(param, loaded_weight)  # 加载权重


EntryClass = InternVLChatModel  # 入口类，用于模型注册
