# Phi-4多模态音频编码器实现文件
# 本文件实现了Phi-4多模态模型的音频编码器组件
# 包含Conformer编码器层、Transformer编码器基类、Conformer编码器、
# WindowQformer和AudioEmbedding等音频处理组件

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
#!/usr/bin/env python3
import abc  # 导入抽象基类模块
import math  # 导入数学模块
from typing import Literal, Optional  # 导入类型提示

import numpy as np  # 导入NumPy
import torch  # 导入PyTorch
import torch.nn.functional as F  # 导入PyTorch函数式API
from torch import Tensor, nn  # 导入张量和神经网络模块
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # 导入检查点包装器
    CheckpointWrapper,  # 检查点包装器
)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel  # 导入FSDP
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.models.phi4mm_utils import (  # 导入Phi-4多模态工具模块
    AbsolutePositionalEncoding,  # 绝对位置编码
    ConvModule,  # 卷积模块
    FeedForward,  # 前馈网络
    MeanVarianceNormLayer,  # 均值方差归一化层
    MultiHeadedAttention,  # 多头注意力
    MultiSequential,  # 多输入多输出序列
    NemoConvSubsampling,  # NeMo卷积子采样
    T5RelativeAttentionLogitBias,  # T5相对注意力偏置
    adaptive_enc_mask,  # 自适应编码器掩码
    get_offset,  # 获取偏移量
    unfold_tensor,  # 展开张量
)

_AUDIO_PLACEHOLDER_TOKEN_ID = 200011  # <|endoftext11|>  # 音频占位符token ID


class ConformerEncoderLayer(nn.Module):  # Conformer编码器层模块
    """ConformerEncoder Layer module.  # Conformer编码器层模块
    for more details see conformer paper:  # 更多细节见Conformer论文
        https://arxiv.org/abs/2005.08100  # Conformer论文链接
    This module implement the Conformer block layer.  # 本模块实现Conformer块层

    Args:  # 参数
        d_model: int  # 注意力维度
            attention dim.  # 注意力维度
        ext_pw_out_channel: int  # 扩展逐点卷积输出通道
            if > 0, ext_pw_out_channel is a dim channel size  # 如果>0，扩展逐点卷积的输出通道数
             for the last pointwise conv after swish activation.  # swish激活后的最后一个逐点卷积
        depthwise_seperable_out_channel: int  # 深度可分离卷积输出通道
            if set different to 0, the number of  # 如果不等于0
             depthwise_seperable_out_channel will be used as a  # 深度可分离卷积的输出通道数
             channel_out of the second conv1d layer.  # 第二个1D卷积层的输出通道
             otherwise, it equal to 0, the second conv1d layer is skipped.  # 否则跳过第二个1D卷积
        depthwise_multiplier: int  # 深度乘数
            number of input_dim channels duplication. this value  # 输入维度通道复制数
             will be used to compute the hidden channels of the Conv1D.  # 用于计算Conv1D的隐藏通道数
        n_head: int  # 注意力头数
            the number of heads for multihead attention module.  # 多头注意力模块的头数
        d_ffn: int  # 前馈网络维度
            output size of the feed_forward blocks.  # 前馈块输出大小
        ext_pw_kernel_size: int  # 扩展逐点卷积核大小
            kernel size of the conv pointwise of the conformer.  # Conformer逐点卷积核大小
        kernel_size: int  # 卷积核大小
            kernel size.  # 卷积核大小
        dropout_rate: float  # dropout率
            dropout rate.  # dropout率
        causal: bool, optional  # 是否因果
            if set to True, convolution have no access  # 如果为True，卷积不能访问
             to future frames. default False.  # 未来帧，默认False
        batch_norm: bool, optional  # 是否使用批归一化
            if set to True, apply batchnorm before activation  # 如果为True，在激活前应用批归一化
            in ConvModule layer of the conformer.  # 在Conformer的ConvModule层中
            default False  # 默认False
        activation: str, optional  # 激活函数
            activation function name,  # 激活函数名称
            one of ["relu", "swish", "sigmoid"],  # 可选值
            sigmoid activation is only used with "glu_in_fnn=True",  # sigmoid仅用于"glu_in_fnn=True"
            default "relu".  # 默认"relu"
        chunk_se: int, optional  # 块SE模式
            0 for offline SE.  # 0为离线SE
            1 for streaming SE, where mean is computed  # 1为流式SE，均值通过累积历史计算
             by accumulated history until current chunk_se.  # 到当前块为止
            2 for streaming SE, where mean is computed  # 2为流式SE，均值仅通过当前块计算
             by only the current chunk.  # 仅当前块
            default 0.  # 默认0
        chunk_size: int, optional  # 块大小
            chunk_size for cnn. default 18  # CNN的块大小，默认18
        conv_activation: str, optional  # 卷积激活函数
            activation function used in ConvModule part  # ConvModule中使用的激活函数
            of the conformer, default "relu".  # 默认"relu"
        conv_glu_type: str, optional  # 卷积GLU类型
            activation function used for the glu inside  # ConvModule中GLU使用的激活函数
            the ConvModule part of the conformer.  # Conformer的ConvModule部分
            default: "sigmoid".  # 默认"sigmoid"
        bias_in_glu: bool, optional  # GLU中是否使用偏置
            if set to True, use additive bias in the weight module  # 如果为True，在权重模块中使用偏置
             before GLU.  # GLU之前
        linear_glu_in_convm: bool, optional  # ConvModule中是否使用线性GLU
            if set to True, use GLULinear module,  # 如果为True，使用GLULinear模块
             otherwise, used GLUPointWiseConv module.  # 否则使用GLUPointWiseConv模块
              default to False.  # 默认False
        attention_inner_dim: int, optional  # 注意力内部维度
            if equal to -1, attention dim for linears k/q/v is  # 如果等于-1，k/q/v的注意力维度
            equal to d_model. otherwise attention_inner_dim is used.  # 等于d_model，否则使用attention_inner_dim
            default -1.  # 默认-1
        attention_glu_type: str, optional  # 注意力GLU类型
            activation function for glu used in the multihead attention,  # 多头注意力中GLU的激活函数
             default "swish".  # 默认"swish"
        activation_checkpointing: str, optional  # 激活检查点
            a dictionarry of {"module","interval","offload"}, where  # 包含module、interval、offload的字典
                "module": str  # 模块
                    accept ["transformer", "attention"] to select  # 接受["transformer", "attention"]
                    which module should do activation checkpointing.  # 选择哪个模块做激活检查点
                "interval": int, default 1,  # 间隔，默认1
                    interval of applying activation checkpointing,  # 应用激活检查点的间隔
                    interval = 1 means that we apply checkpointing  # interval=1表示每层都应用检查点
                    on every layer (if activation), otherwise,  # 否则
                    we apply it every x interval.  # 每x间隔应用
                "offload": bool, default False,  # 卸载，默认False
                    if set to True, we offload activation to cpu and  # 如果为True，将激活卸载到CPU
                    reload it during backward, otherwise,  # 反向传播时重新加载
                    we recalculate activation in backward.  # 否则反向传播时重新计算
            default "".  # 默认空
        export: bool, optional  # 导出模式
            if set to True, it remove the padding from convolutional layers  # 如果为True，移除卷积层的填充
             and allow the onnx conversion for inference.  # 允许ONNX推理转换
              default False.  # 默认False
        use_pt_scaled_dot_product_attention: bool, optional  # 是否使用PyTorch缩放点积注意力
            if set to True, use pytorch's scaled dot product attention  # 如果为True，使用PyTorch缩放点积注意力
            implementation in training.  # 训练时使用
        attn_group_sizes: int, optional  # 注意力组大小
            the number of groups to use for attention, default 1  # 注意力组数，默认1
            (Multi-Head Attention),  # 多头注意力
            1 = typical Multi-Head Attention,  # 1=典型多头注意力
            1 < attn_group_sizes < attention_heads = Grouped-Query Attention  # 1<组大小<头数=分组查询注意力
            attn_group_sizes = attention_heads = Multi-Query Attention  # 组大小=头数=多查询注意力
    """

    def __init__(  # 初始化函数
        self,
        d_model=512,  # 模型维度
        ext_pw_out_channel=0,  # 扩展逐点卷积输出通道
        depthwise_seperable_out_channel=256,  # 深度可分离卷积输出通道
        depthwise_multiplier=1,  # 深度乘数
        n_head=4,  # 注意力头数
        d_ffn=2048,  # 前馈网络维度
        ext_pw_kernel_size=1,  # 扩展逐点卷积核大小
        kernel_size=3,  # 卷积核大小
        dropout_rate=0.1,  # dropout率
        causal=False,  # 是否因果
        batch_norm=False,  # 是否使用批归一化
        activation="relu",  # 激活函数
        chunk_se=0,  # 块SE模式
        chunk_size=18,  # 块大小
        conv_activation="relu",  # 卷积激活函数
        conv_glu_type="sigmoid",  # 卷积GLU类型
        bias_in_glu=True,  # GLU中是否使用偏置
        linear_glu_in_convm=False,  # ConvModule中是否使用线性GLU
        attention_inner_dim=-1,  # 注意力内部维度
        attention_glu_type="swish",  # 注意力GLU类型
        activation_checkpointing="",  # 激活检查点
        export=False,  # 导出模式
        use_pt_scaled_dot_product_attention=False,  # 是否使用PyTorch缩放点积注意力
        attn_group_sizes: int = 1,  # 注意力组大小
    ):
        super().__init__()  # 调用父类初始化

        self.feed_forward_in = FeedForward(  # 输入前馈网络
            d_model=d_model,  # 模型维度
            d_inner=d_ffn,  # 内部维度
            dropout_rate=dropout_rate,  # dropout率
            activation=activation,  # 激活函数
            bias_in_glu=bias_in_glu,  # GLU偏置
        )

        self.self_attn = MultiHeadedAttention(  # 多头注意力
            n_head,  # 头数
            d_model,  # 模型维度
            dropout_rate,  # dropout率
            attention_inner_dim,  # 内部维度
            attention_glu_type,  # GLU类型
            bias_in_glu,  # GLU偏置
            use_pt_scaled_dot_product_attention=use_pt_scaled_dot_product_attention,  # PyTorch SDP注意力
            group_size=attn_group_sizes,  # 注意力组大小
        )
        self.conv = ConvModule(  # 卷积模块
            d_model,  # 模型维度
            ext_pw_out_channel,  # 扩展逐点输出通道
            depthwise_seperable_out_channel,  # 深度可分离输出通道
            ext_pw_kernel_size,  # 扩展逐点核大小
            kernel_size,  # 卷积核大小
            depthwise_multiplier,  # 深度乘数
            dropout_rate,  # dropout率
            causal,  # 是否因果
            batch_norm,  # 批归一化
            chunk_se,  # 块SE
            chunk_size,  # 块大小
            conv_activation,  # 卷积激活
            conv_glu_type,  # GLU类型
            bias_in_glu,  # GLU偏置
            linear_glu_in_convm,  # 线性GLU
            export=export,  # 导出模式
        )

        self.feed_forward_out = FeedForward(  # 输出前馈网络
            d_model=d_model,  # 模型维度
            d_inner=d_ffn,  # 内部维度
            dropout_rate=dropout_rate,  # dropout率
            activation=activation,  # 激活函数
            bias_in_glu=bias_in_glu,  # GLU偏置
        )

        self.layer_norm_att = nn.LayerNorm(d_model)  # 注意力层归一化
        self.layer_norm = nn.LayerNorm(d_model)  # 最终层归一化

    def forward(  # 前向传播函数，执行Conformer编码器层计算
        self,
        x,  # 输入张量
        pos_k,  # 位置键
        pos_v,  # 位置值
        mask,  # 掩码
        relative_attention_bias: Optional[Tensor] = None,  # 相对注意力偏置
    ):
        """ConformerEncoder forward.  # Conformer编码器前向传播

        Args:  # 参数
            x: torch.Tensor  # 输入张量
                input feature of shape (batch, max_time_in, size)  # 输入特征形状
            pos_k: torch.Tensor  # 位置键
                positional key embedding.  # 位置键嵌入
            mask: torch.Tensor  # 掩码
                mask for x (batch, max_time_in)  # x的掩码
            relative_attention_bias: Optional[torch.Tensor]  # 相对注意力偏置
                bias added to attention logits w.r.t. relative positions  # 添加到注意力logits的相对位置偏置
                (1, n_head, time1, time2)  # 形状
        """
        x = x + 0.5 * self.feed_forward_in(x)  # 半步前馈网络输入
        norm_x = self.layer_norm_att(x)  # 注意力层归一化

        x = x + self.self_attn(  # 自注意力残差连接
            norm_x,  # 归一化后的输入
            norm_x,  # 键
            norm_x,  # 值
            pos_k,  # 位置键
            pos_v,  # 位置值
            mask,  # 掩码
            relative_attention_bias=relative_attention_bias,  # 相对注意力偏置
        )
        x = x + self.conv(x)  # 卷积残差连接
        x = x + 0.5 * self.feed_forward_out(x)  # 半步前馈网络输出

        out = self.layer_norm(x)  # 最终层归一化

        return out, pos_k, pos_v, mask  # 返回输出和中间状态


class TransformerEncoderBase(abc.ABC, nn.Module):  # Transformer编码器基类
    """The Base class for Transformer based encoders  # 基于Transformer的编码器基类

    Please set causal = True in streaming model  # 流式模型请设置causal=True
    Args:  # 参数
        input_size: int  # 输入大小
            input feature dimension.  # 输入特征维度
        chunk_size: int, list(int)  # 块大小
            Number of frames for each chunk  # 每个块的帧数
            This variable can take 2 forms:  # 此变量有两种形式
            int:  Used for inference, or single chunk size training  # 整数：推理或单块大小训练
            list(int) : Used only for variable chunk size training  # 列表：仅用于可变块大小训练
            Some examples for the 2 cases:  # 示例
            chunk_size = 12  # 块大小=12
            chunk_size = [6, 8, 12, 24]  # 块大小列表
        left_chunk: int, list(int)  # 左侧块数
            Number of chunks used for masking in streaming mode.  # 流式模式中用于掩码的块数
            This variable can take 2 forms:  # 此变量有两种形式
            int:  Used for inference, or single chunk size training  # 整数：推理或单块大小训练
            list(int) : Used only for variable chunk size training. When  # 列表：可变块大小训练
            chunk_size is a list, left_chunk must be a list with same length.  # 块大小为列表时，left_chunk须同长度
            Some examples for the 2 cases:  # 示例
            left_chunk = 6  # 左侧块=6
            left_chunk = [12, 9, 6, 3]  # 左侧块列表
        attention_dim: int, optional  # 注意力维度
            attention dimension. default 256.  # 注意力维度，默认256
        attention_heads: int, optional  # 注意力头数
            the number of heads. default 4  # 头数，默认4
        input_layer: str, optional  # 输入层类型
            input layer type before Conformer,  # Conformer前的输入层类型
            one of ["linear", "conv2d", "custom", "vgg2l", "embed"],  # 可选值
            default "conv2d"  # 默认"conv2d"
        cnn_out: int, optional  # CNN输出通道
            the number of CNN channels before Conformer.  # Conformer前的CNN通道数
            default -1.  # 默认-1
        cnn_layer_norm: bool, optional  # CNN层归一化
            layer norm between Conformer and the first CNN.  # Conformer和第一个CNN之间的层归一化
            default False.  # 默认False
        time_reduction: int, optional  # 时间缩减因子
            time reduction factor  # 时间缩减因子
            default 4  # 默认4
        dropout_rate: float, optional  # dropout率
            dropout rate. default 0.1  # dropout率，默认0.1
        padding_idx: int, optional  # 填充索引
            padding index for input_layer=embed  # input_layer=embed时的填充索引
            default -1  # 默认-1
        relative_attention_bias_args: dict, optional  # 相对注意力偏置参数
            use more efficient scalar bias-based relative multihead attention  # 使用更高效的标量偏置相对多头注意力
            (Q*K^T + B) implemented in cmb.basics.embedding.  # 在cmb.basics.embedding中实现
            [T5/ALiBi]RelativeAttentionLogitBias  # T5/ALiBi相对注意力偏置
            usage: relative_attention_bias_args={"type": t5/alibi}  # 用法
            additional method-specific arguments can be provided (see  # 可提供额外方法特定参数
            transformer_base.py)  # 见transformer_base.py
        positional_dropout_rate: float, optional  # 位置dropout率
            dropout rate after positional encoding. default 0.0  # 位置编码后的dropout率，默认0.0
        nemo_conv_settings: dict, optional  # NeMo卷积设置
            A dictionary of settings for NeMo Subsampling.  # NeMo子采样设置字典
            default None  # 默认None
        conv2d_extra_padding: str, optional  # 2D卷积额外填充
            Add extra padding in conv2d subsampling layers. Choices are  # 在2D卷积子采样层添加额外填充
            (feat, feat_time, none, True).  # 可选值
            if True or feat_time, the extra padding is added into non full  # 如果True或feat_time，额外填充添加到非完整
            supraframe utts in batch.  # 批次中的超帧话语
            Default: none  # 默认none
        attention_group_size: int, optional  # 注意力组大小
            the number of groups to use for attention, default 1  # 注意力组数，默认1
            (Multi-Head Attention),  # 多头注意力
            1 = typical Multi-Head Attention,  # 1=典型多头注意力
            1 < attention_group_size < attention_heads = Grouped-Query  # 1<组大小<头数=分组查询注意力
            Attention  # 分组查询注意力
            attention_group_size = attention_heads = Multi-Query Attention  # 组大小=头数=多查询注意力
    """

    def __init__(  # 初始化函数
        self,
        input_size,  # 输入大小
        chunk_size,  # 块大小
        left_chunk,  # 左侧块数
        attention_dim=256,  # 注意力维度
        attention_heads=4,  # 注意力头数
        input_layer="nemo_conv",  # 输入层类型
        cnn_out=-1,  # CNN输出通道
        cnn_layer_norm=False,  # CNN层归一化
        time_reduction=4,  # 时间缩减因子
        dropout_rate=0.0,  # dropout率
        padding_idx=-1,  # 填充索引
        relative_attention_bias_args=None,  # 相对注意力偏置参数
        positional_dropout_rate=0.0,  # 位置dropout率
        nemo_conv_settings=None,  # NeMo卷积设置
        conv2d_extra_padding: Literal["feat", "feat_time", "none", True] = "none",  # 2D卷积额外填充
        attention_group_size=1,  # 注意力组大小
        encoder_embedding_config=None,  # 编码器嵌入配置
    ):
        super().__init__()  # 调用父类初始化
        self.input_size = input_size  # 保存输入大小
        self.input_layer = input_layer  # 保存输入层类型
        self.chunk_size = chunk_size  # 保存块大小
        self.left_chunk = left_chunk  # 保存左侧块数
        self.attention_dim = attention_dim  # 保存注意力维度
        self.num_heads = attention_heads  # 保存注意力头数
        self.attention_group_size = attention_group_size  # 保存注意力组大小
        self.time_reduction = time_reduction  # 保存时间缩减因子
        self.nemo_conv_settings = nemo_conv_settings  # 保存NeMo卷积设置
        self.encoder_embedding_config = encoder_embedding_config  # 保存编码器嵌入配置

        if self.input_layer == "nemo_conv":  # 如果输入层是NeMo卷积
            default_nemo_conv_settings = {  # 默认NeMo卷积设置
                "subsampling": "dw_striding",  # 子采样方式
                "subsampling_factor": self.time_reduction,  # 子采样因子
                "feat_in": input_size,  # 输入特征大小
                "feat_out": attention_dim,  # 输出特征大小
                "conv_channels": 256,  # 卷积通道数
                "subsampling_conv_chunking_factor": 1,  # 子采样卷积分块因子
                "activation": nn.ReLU(),  # 激活函数
                "is_causal": False,  # 非因果
            }
            # Override any of the defaults with the incoming, user settings  # 用用户设置覆盖默认值
            if nemo_conv_settings:  # 如果有NeMo卷积设置
                default_nemo_conv_settings.update(nemo_conv_settings)  # 更新设置
                for i in ["subsampling_factor", "feat_in", "feat_out"]:  # 检查关键参数
                    assert (  # 断言
                        i not in nemo_conv_settings  # 关键参数不应在NeMo字典中
                    ), "{i} should be specified outside of the NeMo dictionary"  # 应在外部指定

            self.embed = NemoConvSubsampling(  # NeMo卷积子采样
                **default_nemo_conv_settings,  # 传入设置
            )
        else:  # 否则
            raise ValueError("unknown input_layer: " + input_layer)  # 抛出值错误

        self.pos_emb = AbsolutePositionalEncoding(  # 绝对位置编码
            attention_dim, positional_dropout_rate  # 注意力维度和dropout率
        )

        self.relative_attention_bias_type = (  # 相对注意力偏置类型
            relative_attention_bias_args.get("type")  # 获取类型
            if relative_attention_bias_args  # 如果有参数
            else None  # 否则为空
        )
        if self.relative_attention_bias_type == "t5":  # 如果是T5类型
            assert (  # 断言
                self.num_heads % self.attention_group_size == 0  # 头数可被组大小整除
            ), "attention_group_size must divide n_head"  # 组大小必须整除头数
            self.relative_attention_bias_layer = T5RelativeAttentionLogitBias(  # T5相对注意力偏置层
                self.num_heads // self.attention_group_size,  # 每组头数
                max_distance=relative_attention_bias_args.get(  # 最大距离
                    "t5_bias_max_distance", 1000  # 默认1000
                ),
                symmetric=relative_attention_bias_args.get("t5_bias_symmetric", False),  # 是否对称
            )
        else:  # 否则
            raise NotImplementedError  # 抛出未实现错误

        self.encoder_embedding = MeanVarianceNormLayer(  # 均值方差归一化层
            self.encoder_embedding_config["input_size"]  # 输入大小
        )

    def compute_lens_change(self, feature_lens):  # 计算长度变化函数
        """feature_lens: int  # 特征长度
        return updated feature lens.  # 返回更新后的特征长度

        This used to return a different lambda function for each case that  # 以前返回不同的lambda函数
        computed the right thing.  That does not work within Torchscript.  # Torchscript中不工作
        If you really need this to be faster, create nn.Module()-s for all  # 如果需要更快，创建nn.Module
        the cases and return one of them.  Torchscript does support that.  # Torchscript支持这种方式
        """
        if self.input_layer == "nemo_conv":  # 如果输入层是NeMo卷积
            # Handle the special causal case  # 处理特殊的因果情况
            subsampling_causal_cond = self.nemo_conv_settings.get(  # 获取子采样因果条件
                "subsampling", "dw_striding"  # 默认值
            ) in [  # 在以下列表中
                "dw_striding",  # 深度步幅
                "striding",  # 步幅
                "striding_conv1d",  # 步幅1D卷积
            ]
            is_causal = self.nemo_conv_settings.get("is_causal", False)  # 获取因果标志
            if is_causal and subsampling_causal_cond:  # 如果因果且满足条件
                lens_change = (  # 计算长度变化
                    torch.ceil(feature_lens / self.time_reduction).long()  # 向上取整
                    if isinstance(feature_lens, Tensor)  # 如果是张量
                    else math.ceil(feature_lens / self.time_reduction)  # 否则用math
                )
                feature_lens_remainder = feature_lens % self.time_reduction  # 计算余数
                if isinstance(feature_lens, Tensor):  # 如果是张量
                    lens_change[feature_lens_remainder != 1] += 1  # 余数不为1时加1
                elif feature_lens_remainder != 1:  # 否则
                    lens_change += 1  # 加1
                return lens_change  # 返回长度变化
            ceil_func = math.ceil if isinstance(feature_lens, int) else torch.ceil  # 选择取整函数
            return ceil_func(feature_lens / self.time_reduction)  # 返回计算结果

    @abc.abstractmethod
    def forward(self):  # 抽象前向传播方法
        """Abstract forward method implementation."""  # 抽象前向传播方法

    def _chunk_size_selection(self, chunk_size=None, left_chunk=None):  # 块大小选择函数
        """If chunk size is a list, we will randomly select a chunk size."""  # 如果块大小是列表，随机选择

        if chunk_size is None:  # 如果块大小为空
            chunk_size = self.chunk_size  # 使用默认值
        if left_chunk is None:  # 如果左侧块为空
            left_chunk = self.left_chunk  # 使用默认值
        if isinstance(chunk_size, list):  # 如果块大小是列表
            # Variable chunk size during training  # 训练时可变块大小
            chunk_size_index = int(  # 随机选择块大小索引
                torch.randint(low=0, high=len(chunk_size), size=(1,))  # 随机整数
            )
            chunk_size_train_eff = chunk_size[chunk_size_index]  # 有效的训练块大小
            if not isinstance(left_chunk, list):  # 如果左侧块不是列表
                raise ValueError(  # 抛出值错误
                    "Since chunk_size is a list, left_chunk must be a list"  # left_chunk必须是列表
                )
            if len(left_chunk) != len(chunk_size):  # 如果长度不匹配
                raise ValueError(  # 抛出值错误
                    "The length of left_chunk must be the same as length of "  # left_chunk长度必须与chunk_size相同
                    "chunk_size."  # 长度不匹配
                )
            left_chunk_train_eff = left_chunk[chunk_size_index]  # 有效的训练左侧块
        else:  # 否则
            chunk_size_train_eff = chunk_size  # 使用给定块大小
            left_chunk_train_eff = left_chunk  # 使用给定左侧块

        return chunk_size_train_eff, left_chunk_train_eff  # 返回有效的块大小和左侧块

    def _get_embed_class(self, embed):  # 获取嵌入类的实际类型
        # pylint: disable=protected-access  # 忽略受保护访问警告
        is_embed_using_act_chkpt = isinstance(embed, CheckpointWrapper)  # 是否使用激活检查点
        is_embed_fsdp_wrapped = isinstance(embed, FullyShardedDataParallel)  # 是否FSDP包装
        embed_class = embed  # 嵌入类
        if is_embed_using_act_chkpt:  # 如果使用激活检查点
            embed_class = embed._checkpoint_wrapped_module  # 获取检查点包装的模块
        if is_embed_fsdp_wrapped:  # 如果FSDP包装
            embed_class = embed.module  # 获取FSDP包装的模块
        return embed_class  # 返回嵌入类

    def _forward_embeddings_core(self, input_tensor, masks):  # 嵌入层核心前向传播
        embed_class = self._get_embed_class(self.embed)  # 获取嵌入类
        assert isinstance(embed_class, NemoConvSubsampling)  # 断言是NeMo卷积子采样
        input_tensor, masks = self.embed(input_tensor, masks)  # 通过嵌入层
        return input_tensor, masks  # 返回输入张量和掩码

    def _position_embedding(self, input_tensor):  # 位置嵌入函数
        pos_k = None  # 位置键初始化为空
        pos_v = None  # 位置值初始化为空
        if self.relative_attention_bias_layer is None:  # 如果没有相对注意力偏置层
            input_tensor = self.pos_emb(  # 添加绝对位置编码
                input_tensor  # 输入张量
            )  # default to add abs sinusoid embedding  # 默认添加绝对正弦嵌入
        return pos_k, pos_v  # 返回位置键和值

    def _streaming_mask(self, seq_len, batch_size, chunk_size, left_chunk):  # 流式掩码生成函数
        chunk_size_train_eff, left_chunk_train_eff = self._chunk_size_selection(  # 选择块大小
            chunk_size, left_chunk  # 传入参数
        )

        # Create mask matrix for streaming  # 创建流式掩码矩阵
        # S stores start index. if chunksize is 18, s is [0,18,36,....]  # S存储起始索引
        chunk_start_idx = np.arange(0, seq_len, chunk_size_train_eff)  # 块起始索引

        enc_streaming_mask = (  # 编码器流式掩码
            adaptive_enc_mask(  # 自适应编码器掩码
                seq_len, chunk_start_idx, left_window=left_chunk_train_eff  # 传入参数
            )
            .unsqueeze(0)  # 添加批次维度
            .expand([batch_size, -1, -1])  # 扩展到批次大小
        )
        return enc_streaming_mask  # 返回流式掩码

    def forward_embeddings(self, xs_pad, masks, chunk_size_nc=None, left_chunk_nc=None):  # 嵌入层前向传播
        """Forwarding the inputs through the top embedding layers  # 通过顶层嵌入层转发输入

        Args:  # 参数
            xs_pad: torch.Tensor  # 输入张量
                input tensor  # 输入张量
            masks: torch.Tensor  # 掩码
                input mask  # 输入掩码
            chunk_size_nc: (optional, default is None) chunk size for  # 非因果层的块大小
                            non-causal layers  # 可选，默认None
            left_chunk_nc: (optional, default is None) # of left chunks for  # 非因果层的左侧块数
                            non-causal layers  # 可选，默认None
        """
        # pylint: disable=R0915  # 忽略方法行数过多警告
        # get new lens.  # 获取新长度
        seq_len = int(self.compute_lens_change(xs_pad.shape[1]))  # 计算新序列长度
        if seq_len <= 0:  # 如果长度无效
            raise ValueError(  # 抛出值错误
                f"""The sequence length after time reduction is invalid:  # 时间缩减后序列长度无效
                {seq_len}. Your input feature is too short. Consider  # 输入特征太短
                filtering out the very short sentence from data  # 考虑过滤掉过短的句子
                loader""",  # 从数据加载器中
            )

        batch_size = xs_pad.shape[0]  # 批次大小

        enc_streaming_mask = self._streaming_mask(  # 生成流式掩码
            seq_len, batch_size, self.chunk_size, self.left_chunk  # 传入参数
        )

        if xs_pad.is_cuda:  # 如果在CUDA上
            enc_streaming_mask = enc_streaming_mask.cuda()  # 转移到CUDA
            xs_pad = xs_pad.cuda()  # 转移到CUDA

        input_tensor = xs_pad  # 输入张量
        input_tensor, masks = self._forward_embeddings_core(input_tensor, masks)  # 通过嵌入层核心

        streaming_mask = enc_streaming_mask  # 流式掩码
        if streaming_mask is not None and masks is not None:  # 如果掩码都存在
            hs_mask = masks & streaming_mask  # 合并掩码
        elif masks is not None:  # 如果只有输入掩码
            hs_mask = masks  # 使用输入掩码
        else:  # 否则
            hs_mask = streaming_mask  # 使用流式掩码

        if chunk_size_nc is not None:  # 如果有非因果块大小
            enc_streaming_mask_nc = self._streaming_mask(  # 生成非因果流式掩码
                seq_len, batch_size, chunk_size_nc, left_chunk_nc  # 传入参数
            )
            if xs_pad.is_cuda:  # 如果在CUDA上
                enc_streaming_mask_nc = enc_streaming_mask_nc.cuda()  # 转移到CUDA
            if masks is not None:  # 如果有输入掩码
                hs_mask_nc = masks & enc_streaming_mask_nc  # 合并掩码
            else:  # 否则
                hs_mask_nc = enc_streaming_mask_nc  # 使用流式掩码
        else:  # 否则
            hs_mask_nc = None  # 非因果掩码为空

        pos_k, pos_v = self._position_embedding(input_tensor)  # 计算位置嵌入

        if chunk_size_nc is None:  # 如果没有非因果块大小
            return input_tensor, pos_k, pos_v, hs_mask, masks  # 返回5个值
        return input_tensor, pos_k, pos_v, hs_mask, masks, hs_mask_nc  # 返回6个值

    def get_offset(self):  # 获取偏移量函数
        """Returns offset used when retaining inputs for decoding.  # 返回解码时保留输入的偏移量

        This is essentially, how many additional frames have to be added to  # 本质上是需要添加多少额外帧
        the front-end CNN input to ensure it can produce a single output.  # 到前端CNN输入以确保能产生单个输出
        So if the "padding" parameter is 0, typically offset will be > 0.  # 如果padding参数为0，通常offset>0
        """
        return get_offset(self.input_layer, self.time_reduction)  # 返回偏移量


class ConformerEncoder(TransformerEncoderBase):  # Conformer编码器，继承自Transformer编码器基类
    """ConformerEncoder module.  # Conformer编码器模块
    see original paper for more details:  # 更多细节见原始论文
        https://arxiv.org/abs/2005.08100  # 论文链接

    Please set causal = True in streaming model  # 流式模型请设置causal=True
    Args:  # 参数（同基类，此处省略详细注释以避免重复）
        input_size, chunk_size, left_chunk, num_lang, attention_dim, attention_heads,
        linear_units, num_blocks, dropout_rate, input_layer, causal, batch_norm,
        cnn_out, cnn_layer_norm, ext_pw_out_channel, ext_pw_kernel_size,
        depthwise_seperable_out_channel, depthwise_multiplier, chunk_se, kernel_size,
        activation, conv_activation, conv_glu_type, bias_in_glu, linear_glu_in_convm,
        attention_glu_type, export, activation_checkpointing, extra_layer_output_idx,
        relative_attention_bias_args, time_reduction,
        use_pt_scaled_dot_product_attention, nemo_conv_settings,
        conv2d_extra_padding, replication_pad_for_subsample_embedding,
        attention_group_size, encoder_embedding_config
    """

    extra_multi_layer_output_idxs: list[int]  # 额外多层输出索引

    def __init__(  # pylint: disable-all  # 初始化函数
        self,
        input_size,  # 输入大小
        chunk_size,  # 块大小
        left_chunk,  # 左侧块数
        num_lang=None,  # 语言数量
        attention_dim=256,  # 注意力维度
        attention_heads=4,  # 注意力头数
        linear_units=2048,  # 线性单元数
        num_blocks=6,  # 块数量
        dropout_rate=0.1,  # dropout率
        input_layer="nemo_conv",  # 输入层类型
        causal=True,  # 是否因果
        batch_norm=False,  # 批归一化
        cnn_out=-1,  # CNN输出通道
        cnn_layer_norm=False,  # CNN层归一化
        ext_pw_out_channel=0,  # 扩展逐点卷积输出通道
        ext_pw_kernel_size=1,  # 扩展逐点卷积核大小
        depthwise_seperable_out_channel=256,  # 深度可分离卷积输出通道
        depthwise_multiplier=1,  # 深度乘数
        chunk_se=0,  # 块SE模式
        kernel_size=3,  # 卷积核大小
        activation="relu",  # 激活函数
        conv_activation="relu",  # 卷积激活函数
        conv_glu_type="sigmoid",  # 卷积GLU类型
        bias_in_glu=True,  # GLU偏置
        linear_glu_in_convm=False,  # 线性GLU
        attention_glu_type="swish",  # 注意力GLU类型
        export=False,  # 导出模式
        extra_layer_output_idx=-1,  # 额外层输出索引
        extra_multi_layer_output_idxs=[],  # 额外多层输出索引  # noqa
        activation_checkpointing="",  # 激活检查点
        relative_attention_bias_args=None,  # 相对注意力偏置参数
        time_reduction=4,  # 时间缩减因子
        use_pt_scaled_dot_product_attention=False,  # PyTorch缩放点积注意力
        nemo_conv_settings=None,  # NeMo卷积设置
        conv2d_extra_padding: Literal["feat", "feat_time", "none", True] = "none",  # 2D卷积额外填充
        replication_pad_for_subsample_embedding=False,  # 子采样嵌入的复制填充
        attention_group_size=1,  # 注意力组大小
        encoder_embedding_config=None,  # 编码器嵌入配置
    ):
        super().__init__(  # 调用父类初始化
            input_size,  # 输入大小
            chunk_size,  # 块大小
            left_chunk,  # 左侧块数
            attention_dim,  # 注意力维度
            attention_heads,  # 注意力头数
            input_layer,  # 输入层类型
            cnn_out,  # CNN输出通道
            cnn_layer_norm,  # CNN层归一化
            time_reduction,  # 时间缩减因子
            dropout_rate=dropout_rate,  # dropout率
            relative_attention_bias_args=relative_attention_bias_args,  # 相对注意力偏置参数
            positional_dropout_rate=0.0,  # 位置dropout率
            nemo_conv_settings=nemo_conv_settings,  # NeMo卷积设置
            conv2d_extra_padding=conv2d_extra_padding,  # 2D卷积额外填充
            attention_group_size=attention_group_size,  # 注意力组大小
            encoder_embedding_config=encoder_embedding_config,  # 编码器嵌入配置
        )
        self.num_blocks = num_blocks  # 保存块数量
        self.num_lang = num_lang  # 保存语言数量
        self.kernel_size = kernel_size  # 保存卷积核大小
        self.replication_pad_for_subsample_embedding: bool = (  # 子采样嵌入复制填充
            replication_pad_for_subsample_embedding  # 传入参数
        )
        assert (  # 断言
            self.num_heads % attention_group_size == 0  # 头数可被组大小整除
        ), "attention_group_size must divide n_head"  # 组大小必须整除头数
        self.num_heads_k = self.num_heads // attention_group_size  # 每组KV头数

        self.encoders = MultiSequential(  # 多输入多输出编码器序列
            *[
                ConformerEncoderLayer(  # Conformer编码器层
                    d_model=attention_dim,  # 模型维度
                    ext_pw_out_channel=ext_pw_out_channel,  # 扩展逐点输出通道
                    depthwise_seperable_out_channel=depthwise_seperable_out_channel,  # 深度可分离输出通道
                    depthwise_multiplier=depthwise_multiplier,  # 深度乘数
                    n_head=attention_heads,  # 头数
                    d_ffn=linear_units,  # 前馈维度
                    ext_pw_kernel_size=ext_pw_kernel_size,  # 扩展逐点核大小
                    kernel_size=kernel_size,  # 卷积核大小
                    dropout_rate=dropout_rate,  # dropout率
                    causal=causal,  # 因果标志
                    batch_norm=batch_norm,  # 批归一化
                    activation=activation,  # 激活函数
                    chunk_se=chunk_se,  # 块SE
                    chunk_size=chunk_size,  # 块大小
                    conv_activation=conv_activation,  # 卷积激活
                    conv_glu_type=conv_glu_type,  # GLU类型
                    bias_in_glu=bias_in_glu,  # GLU偏置
                    linear_glu_in_convm=linear_glu_in_convm,  # 线性GLU
                    attention_glu_type=attention_glu_type,  # 注意力GLU类型
                    activation_checkpointing=activation_checkpointing,  # 激活检查点
                    export=export,  # 导出模式
                    use_pt_scaled_dot_product_attention=use_pt_scaled_dot_product_attention,  # PyTorch SDP注意力
                    attn_group_sizes=attention_group_size,  # 注意力组大小
                )
                for _ in range(num_blocks)  # 重复num_blocks次
            ]
        )
        self.extra_layer_output_idx = extra_layer_output_idx  # 额外层输出索引
        self.extra_multi_layer_output_idxs = extra_multi_layer_output_idxs  # 额外多层输出索引
        # Make a zeros scalar we can use in get_initial_state to determine  # 创建零标量用于确定
        # the device and the needed dtype:  # 设备和所需数据类型
        self.register_buffer("dev_type", torch.zeros(()), persistent=False)  # 注册设备类型缓冲区

    def init_relative_attention_bias(self, input_tensor):  # 初始化相对注意力偏置函数
        if self.relative_attention_bias_layer:  # 如果有相对注意力偏置层
            return self.relative_attention_bias_layer(input_tensor)  # 计算并返回偏置

    def calculate_hs_mask(self, xs_pad, device, mask):  # 计算隐藏状态掩码函数
        max_audio_length = xs_pad.shape[1]  # 最大音频长度
        batch_size = xs_pad.shape[0]  # 批次大小
        enc_streaming_mask = self._streaming_mask(  # 生成流式掩码
            max_audio_length, batch_size, self.chunk_size, self.left_chunk  # 传入参数
        )
        enc_streaming_mask = enc_streaming_mask.to(device)  # 转移到设备
        if mask is None:  # 如果掩码为空
            return enc_streaming_mask  # 返回流式掩码

        feature_lens = mask.sum(1)  # 计算特征长度
        padding_length = feature_lens  # 填充长度
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(  # 创建填充掩码
            padding_length.size(0), -1  # 扩展维度
        ) < padding_length.unsqueeze(1)  # 小于填充长度的位置为True
        pad_mask = pad_mask.unsqueeze(1)  # 添加头维度
        pad_mask = pad_mask & enc_streaming_mask  # 合并掩码
        return pad_mask  # 返回填充掩码

    @torch.jit.ignore  # 忽略JIT编译
    def forward(self, xs_pad, masks):  # 前向传播函数，执行Conformer编码器计算
        """Conformer Forward function  # Conformer前向传播函数

        Args:  # 参数
            xs_pad: torch.Tensor  # 输入张量
                input tensor  # 输入张量
            masks: torch.Tensor  # 掩码
                post-embedding input lengths  # 嵌入后输入长度
        """
        xs_pad = self.encoder_embedding(xs_pad)  # 通过编码器嵌入层
        input_tensor, pos_k, pos_v, hs_mask, masks = self.forward_embeddings(  # 通过嵌入层
            xs_pad, masks  # 传入参数
        )

        unfolded = False  # 是否展开标志
        ori_bz, seq_len, D = input_tensor.shape  # 原始形状
        max_seq_len = 500  # 绝对位置编码的最大位置
        if seq_len > max_seq_len:  # 如果序列长度超过最大值
            # audio sequence is longer than max_seq_len, unfold it into chunks  # 音频序列超过最大长度，展开为块
            # of max_seq_len  # 每块max_seq_len
            unfolded = True  # 标记为已展开
            # the unfold op will drop residual frames, pad it to the multiple  # 展开操作会丢弃残余帧，填充到倍数
            # of max_seq_len  # max_seq_len的倍数
            if seq_len % max_seq_len > 0:  # 如果有余数
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)  # 计算填充大小
            else:  # 否则
                chunk_pad_size = 0  # 不需要填充
            if chunk_pad_size > 0:  # 如果需要填充
                input_tensor_pad = F.pad(  # 填充输入
                    input_tensor, (0, 0, 0, chunk_pad_size), "constant", 0  # 零填充
                )
                input_tensor = input_tensor_pad.to(input_tensor.device)  # 转移设备
            input_tensor = unfold_tensor(input_tensor, max_seq_len)  # 展开张量
            if masks is not None:  # 如果有掩码
                # revise hs_mask here because the previous calculated hs_mask  # 修正hs_mask，因为之前计算的hs_mask
                # did not consider extra pad  # 没有考虑额外填充
                subsampled_pad_mask = masks.squeeze(  # 压缩掩码
                    1  # 压缩第1维
                )  # [bz, subsampled_unmask_seq_len]  # 形状
                extra_padded_subsamlped_pad_mask = F.pad(  # 额外填充掩码
                    subsampled_pad_mask, (0, chunk_pad_size), "constant", False  # 填充False
                )  # extra padding to the pad mask  # 掩码的额外填充
                extra_padded_subsamlped_pad_mask = (  # 转换为浮点
                    extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()  # 添加维度并转浮点
                )
                masks_unfold = unfold_tensor(  # 展开掩码
                    extra_padded_subsamlped_pad_mask, max_seq_len  # 传入参数
                )  # unfold the pad mask like we did to the input tensor  # 像输入张量一样展开掩码
                masks_unfold = masks_unfold.squeeze(  # 压缩掩码
                    -1  # 压缩最后一维
                ).bool()  # unfold op does not support bool tensor  # 展开操作不支持布尔张量
            else:  # 否则
                masks_unfold = None  # 掩码为空
            hs_mask = self.calculate_hs_mask(  # 计算hs_mask
                input_tensor, input_tensor.device, masks_unfold  # 传入参数
            )  # calculate hs_mask based on the unfolded pad mask  # 基于展开后的掩码计算hs_mask

        # layer_emb = None  # 层嵌入（已注释）

        relative_attention_bias = self.init_relative_attention_bias(input_tensor)  # 初始化相对注意力偏置

        _simplified_path = (  # 简化路径条件
            self.extra_layer_output_idx == -1 and relative_attention_bias is None  # 没有额外层输出且无偏置
        )

        if _simplified_path:  # 如果使用简化路径
            input_tensor, *_ = self.encoders(input_tensor, pos_k, pos_v, hs_mask)  # 通过编码器
        else:  # 否则
            for i, layer in enumerate(self.encoders):  # 逐层遍历
                input_tensor, _, _, _ = layer(  # 通过每层
                    input_tensor,  # 输入
                    pos_k,  # 位置键
                    pos_v,  # 位置值
                    hs_mask,  # 掩码
                    relative_attention_bias=relative_attention_bias,  # 相对注意力偏置
                )

                # if i == self.extra_layer_output_idx:  # 如果是额外输出层（已注释）
                #     layer_emb = input_tensor  # 保存层嵌入（已注释）

        if unfolded:  # 如果之前展开过
            embed_dim = input_tensor.shape[-1]  # 嵌入维度
            input_tensor = input_tensor.reshape(ori_bz, -1, embed_dim)  # 重塑回原始批次
            # if we ever padded before unfolding, we need to remove the padding  # 如果展开前填充过，需要移除填充
            if chunk_pad_size > 0:  # 如果有填充
                input_tensor = input_tensor[:, :-chunk_pad_size, :]  # 移除填充

        return input_tensor, masks  # , layer_emb  # 返回输入张量和掩码


class WindowQformer(nn.Module):  # 窗口级Qformer模块
    """Window-level Qformer"""  # 窗口级Qformer

    def __init__(  # 初始化函数
        self,
        window_size: int = 8,  # 窗口大小
        num_queries: int = 1,  # 查询数量
        num_blocks: int = 2,  # 块数量
        attention_dim: int = 512,  # 注意力维度
        attention_heads: int = 8,  # 注意力头数
        linear_units: int = 2048,  # 线性单元数
        dropout_rate: float = 0.0,  # dropout率
        normalize_before: bool = True,  # 是否在前归一化
    ):
        super().__init__()  # 调用父类初始化

        self.decoders = nn.ModuleList(  # 解码器层列表
            [
                nn.TransformerDecoderLayer(  # Transformer解码器层
                    d_model=attention_dim,  # 模型维度
                    nhead=attention_heads,  # 头数
                    dim_feedforward=linear_units,  # 前馈维度
                    dropout=dropout_rate,  # dropout率
                    activation="relu",  # 激活函数
                    batch_first=True,  # 批次在前
                    norm_first=normalize_before,  # 归一化在前  # TODO need to verify  # 需要验证
                )
                for _ in range(num_blocks)  # 重复num_blocks次
            ]
        )

        self.queries = nn.Parameter(torch.zeros(1, num_queries, attention_dim))  # 查询参数
        self.after_norm = (  # 后归一化
            nn.LayerNorm(attention_dim, eps=1e-12) if normalize_before else None  # 如果归一化在前
        )
        self.window_size = window_size  # 保存窗口大小

    def forward(self, audio_embed, mask, embed_len=None):  # 前向传播函数，执行窗口Qformer计算
        """forward decoder"""  # 解码器前向传播
        # audio_embed: N x T x D => N x D x T  # 音频嵌入转置

        audio_embed = audio_embed.transpose(1, 2)  # 转置音频嵌入
        # audio_embed: N x D x 1 x T => N x DK x T'  # 音频嵌入处理
        padding = audio_embed.shape[-1] % self.window_size  # 计算填充
        if padding > 0:  # 如果需要填充
            audio_embed = F.pad(  # 填充音频嵌入
                audio_embed, (0, self.window_size - padding), "constant", 0  # 零填充
            )

        embed_chunk = F.unfold(  # 展开为块
            audio_embed[..., None, :],  # 添加维度
            kernel_size=(1, self.window_size),  # 核大小
            stride=(1, self.window_size),  # 步幅
        )
        bsz, _, slen = embed_chunk.shape  # 获取形状
        # N x D x K x T'  # 重塑为4D
        embed_chunk = embed_chunk.view(bsz, -1, self.window_size, slen)  # 重塑
        # N x T' x K x D  # 转置
        embed_chunk = embed_chunk.transpose(1, 3).contiguous()  # 转置并连续化
        # NT' x K x D  # 重塑为3D
        embed_chunk = embed_chunk.view(bsz * slen, self.window_size, -1)  # 重塑
        # NT' x 1 x D  # 扩展查询
        q = self.queries.expand(bsz * slen, -1, -1)  # 扩展查询
        for layer in self.decoders:  # 遍历解码器层
            q = layer(tgt=q, memory=embed_chunk, tgt_mask=None, memory_mask=mask)  # 通过解码器层

        if self.after_norm is not None:  # 如果有后归一化
            q = self.after_norm(q)  # 应用后归一化

        if embed_len is not None:  # 如果有嵌入长度
            embed_len = embed_len // self.window_size  # 计算窗口化后的长度
        # N x T' x D  # 重塑为3D
        out = q.view(bsz, slen, -1)  # 重塑

        return out, embed_len  # 返回输出和嵌入长度


class AudioEmbedding(nn.Module):  # 音频嵌入模块
    """Image embedding."""  # 图像嵌入（原注释如此，实际为音频嵌入）

    def __init__(self, config: PretrainedConfig, **kwargs) -> None:  # 初始化函数
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        # n_embed or hidden_size for text LM  # 文本LM的n_embed或hidden_size
        hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size  # 获取隐藏层大小

        # self.wte = nn.Embedding(config.vocab_size, hidden_size)  # 词嵌入（已注释）

        audio_dim_out = (  # 音频输出维度
            None  # Set this variable according to the actual audio processor  # 根据实际音频处理器设置
        )
        self.layer_idx = -2  # 层索引

        if (  # 如果
            isinstance(config.audio_processor, dict)  # 音频处理器配置是字典
            and config.audio_processor.get("name", None) == "cascades"  # 且名称为cascades
        ):
            encoder_config = config.audio_processor.get("config", None)  # 获取编码器配置
            assert encoder_config is not None  # 断言编码器配置不为空
            self.encoder = ConformerEncoder(**encoder_config)  # 创建Conformer编码器

            audio_dim_out = encoder_config["attention_dim"]  # 音频输出维度
            n_mels = encoder_config["input_size"]  # 梅尔频率数
        else:  # 否则
            raise NotImplementedError("")  # 抛出未实现错误

        assert audio_dim_out is not None, "Remember to set values for audio_dim_out"  # 断言音频输出维度已设置
        self.audio_dim_out = audio_dim_out  # 保存音频输出维度
        self.audio_dim_in = n_mels  # 保存音频输入维度

        self.freeze_audio_processor = kwargs.get("freeze_audio_processor", False)  # 是否冻结音频处理器

        self.downsample_rate = kwargs.get("downsample_rate", 1)  # 下采样率

        if kwargs.get("use_qformer", False):  # 如果使用Qformer
            qformer_config = kwargs.get("qformer_config", {})  # 获取Qformer配置
            qformer_config["attention_dim"] = audio_dim_out  # 设置注意力维度
            self.qformer = WindowQformer(**qformer_config)  # 创建窗口Qformer
        else:  # 否则
            self.qformer = None  # Qformer为空

        if kwargs.get("use_conv_downsample", False):  # 如果使用卷积下采样
            assert (  # 断言
                self.qformer is None  # 不支持同时使用Qformer和卷积下采样
            ), "don't support use qformer and conv downsample together"  # 不支持同时使用
            nemo_conv_settings = kwargs.get("nemo_conv_settings", {})  # NeMo卷积设置
            default_nemo_conv_settings = {  # 默认NeMo卷积设置
                "subsampling": "dw_striding",  # 子采样方式
                "subsampling_factor": self.downsample_rate,  # 子采样因子
                "feat_in": audio_dim_out,  # 输入特征大小
                "feat_out": audio_dim_out,  # 输出特征大小
                "conv_channels": 256,  # 卷积通道数
                "subsampling_conv_chunking_factor": 1,  # 子采样卷积分块因子
                "activation": nn.ReLU(),  # 激活函数
                "is_causal": False,  # 非因果
            }
            # Override any of the defaults with the incoming, user settings  # 用用户设置覆盖默认值
            if nemo_conv_settings:  # 如果有NeMo卷积设置
                default_nemo_conv_settings.update(nemo_conv_settings)  # 更新设置
                for i in ["subsampling_factor", "feat_in", "feat_out"]:  # 检查关键参数
                    assert (  # 断言
                        i not in nemo_conv_settings  # 关键参数不应在NeMo字典中
                    ), "{i} should be specified outside of the NeMo dictionary"  # 应在外部指定

            self.conv_ds = NemoConvSubsampling(  # 卷积下采样
                **default_nemo_conv_settings,  # 传入设置
            )
        else:  # 否则
            self.conv_ds = None  # 卷积下采样为空

        projection_cls = kwargs.get("projection_cls", "linear")  # 投影类
        if projection_cls == "linear":  # 如果是线性投影
            self.audio_projection = nn.Linear(audio_dim_out, hidden_size)  # 线性投影层
        elif projection_cls == "mlp":  # 如果是MLP投影
            # follow llava-v1.5's implementation  # 遵循llava-v1.5的实现
            # (do not use image_projection and image_proj_norm)  # 不使用image_projection和image_proj_norm
            dim_projection = hidden_size  # 投影维度
            depth = 2  # 深度
            self.linear_downsample_rate = (  # 线性下采样率
                1 if (self.qformer or self.conv_ds) else self.downsample_rate  # 如果有Qformer或卷积下采样则为1
            )
            layers = [  # 投影层列表
                nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)  # 线性层
            ]
            for _ in range(1, depth):  # 遍历深度
                layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])  # 添加GELU和线性层
            self.audio_projection = nn.Sequential(*layers)  # 音频投影层
            # NOTE vision-speech tasks use a separate projection layer  # 注意：视觉-语音任务使用单独的投影层
            layers = [  # 视觉投影层列表
                nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)  # 线性层
            ]
            for _ in range(1, depth):  # 遍历深度
                layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])  # 添加GELU和线性层
            self.audio_projection_for_vision = nn.Sequential(*layers)  # 视觉音频投影层
        else:  # 否则
            raise NotImplementedError(  # 抛出未实现错误
                f"projection_cls = {projection_cls}, not implemented"  # 未实现的投影类
            )

        # TODO: audio sequence compression - Qformer  # TODO：音频序列压缩 - Qformer
        self.vocab_size = config.vocab_size  # 词表大小
        self.input_embeds = None  # 输入嵌入
        self.audio_embed_sizes = None  # 音频嵌入大小

    def set_audio_embeds(self, input_embeds: torch.FloatTensor) -> None:  # 设置音频嵌入函数
        self.input_embeds = input_embeds  # 设置输入嵌入

    def set_audio_embed_sizes(self, audio_embed_sizes: torch.LongTensor) -> None:  # 设置音频嵌入大小函数
        self.audio_embed_sizes = audio_embed_sizes  # 设置音频嵌入大小

    def get_audio_features(  # 获取音频特征函数
        self,
        input_embeds: torch.FloatTensor,  # 输入嵌入
        audio_attention_mask: torch.Tensor = None,  # 音频注意力掩码
        audio_projection_mode: str = "speech",  # 音频投影模式
    ) -> torch.FloatTensor:
        """
        arguments:  # 参数
            input_embeds: audio features (B, T, D)  B: num audios in a sequence  # 音频特征，B为序列中音频数
        """
        if self.freeze_audio_processor:  # 如果冻结音频处理器
            with torch.no_grad():  # 不计算梯度
                audio_features, masks = self.encoder(input_embeds, audio_attention_mask)  # 通过编码器
        else:  # 否则
            audio_features, masks = self.encoder(input_embeds, audio_attention_mask)  # 通过编码器

        if self.qformer is not None:  # 如果有Qformer
            audio_features, _ = self.qformer(audio_features, mask=None)  # 通过Qformer

        if self.conv_ds is not None:  # 如果有卷积下采样
            if masks is not None:  # 如果有掩码
                masks = masks.squeeze(1)  # 压缩掩码

            audio_features, masks = self.conv_ds(audio_features, mask=masks)  # 通过卷积下采样

        if self.linear_downsample_rate != 1:  # 如果线性下采样率不为1
            bs, seq_len, feat_dim = audio_features.size()  # 获取形状
            padding = seq_len % self.linear_downsample_rate  # 计算填充
            if padding > 0:  # 如果需要填充
                audio_features = F.pad(  # 填充音频特征
                    audio_features,
                    (0, 0, 0, self.linear_downsample_rate - padding),  # 填充参数
                    "constant",  # 常数填充
                    0,  # 填充值
                )

            seq_len = audio_features.size(1)  # 更新序列长度
            audio_features = audio_features.view(  # 重塑音频特征
                bs,  # 批次大小
                seq_len // self.linear_downsample_rate,  # 新序列长度
                feat_dim * self.linear_downsample_rate,  # 新特征维度
            )

        if audio_projection_mode == "speech":  # 如果是语音模式
            audio_set_tensor = self.audio_projection(audio_features)  # 通过音频投影
        elif audio_projection_mode == "vision":  # 如果是视觉模式
            audio_set_tensor = self.audio_projection_for_vision(audio_features)  # 通过视觉音频投影
        else:  # 否则
            raise ValueError(  # 抛出值错误
                f"audio_projection_mode = {audio_projection_mode} not " "implemented"  # 未实现的投影模式
            )

        return audio_set_tensor  # 返回音频特征

    def forward(  # 前向传播函数，执行音频嵌入计算
        self,
        audio_features: torch.FloatTensor,  # 音频特征
        audio_attention_mask: torch.Tensor = None,  # 音频注意力掩码
        audio_projection_mode: str = "speech",  # 音频投影模式
    ) -> torch.FloatTensor:
        """
        arguments:  # 参数
            audio_features: audio features (num_audio_tokens, T, D)  # 音频特征形状

        returns:  # 返回
            audio_embeds: audio embeddings (num_audio_tokens, hidden_dim)  # 音频嵌入形状
        """
        audio_embeds = self.get_audio_features(  # 获取音频特征
            audio_features,  # 音频特征
            audio_attention_mask=audio_attention_mask,  # 注意力掩码
            audio_projection_mode=audio_projection_mode,  # 投影模式
        )
        return audio_embeds  # 返回音频嵌入
