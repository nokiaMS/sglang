# Dots视觉Transformer (ViT) 实现
# 本文件实现了Dots-VL和Dots-OCR模型中使用的视觉Transformer组件，
# 包含视觉旋转位置编码、Patch合并器、RMS归一化、SwiGLU前馈网络、
# Patch嵌入、视觉预处理、视觉块和完整的视觉Transformer模型。

import logging  # 导入日志模块
from typing import Optional  # 导入可选类型提示

import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数式接口
import torch.utils.checkpoint  # 导入梯度检查点工具
from torch.nn import LayerNorm  # 导入层归一化
from transformers.modeling_utils import PreTrainedModel  # 导入预训练模型基类

from sglang.srt.configs.dots_vlm import DotsVisionConfig  # 导入Dots视觉配置类
from sglang.srt.distributed import parallel_state  # 导入并行状态模块
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.layers.conv import Conv2dLayer  # 导入2D卷积层
from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置
from sglang.srt.utils import add_prefix, is_npu  # 导入前缀添加工具和NPU判断函数

logger = logging.getLogger(__name__)  # 创建模块级日志记录器


class VisionRotaryEmbedding(nn.Module):
    """视觉旋转位置编码，为视觉token生成位置相关的频率"""
    def __init__(self, dim: int, theta: float = 10000.0) -> None:  # 初始化方法
        super().__init__()  # 调用父类初始化
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))  # 计算逆频率：1 / (theta^(2k/d))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 注册为非持久化缓冲区

    def forward(self, seqlen: int) -> torch.Tensor:  # 前向传播，根据序列长度生成频率矩阵
        seq = torch.arange(  # 生成序列索引
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype  # 设备和数据类型与逆频率一致
        )
        freqs = torch.outer(seq, self.inv_freq)  # 计算外积得到频率矩阵 [seqlen, dim//2]
        return freqs  # 返回频率矩阵


class PatchMerger(nn.Module):
    """Patch合并器，将视觉token的空间特征合并并投影到目标维度"""
    def __init__(  # 初始化方法
        self,
        dim: int,  # 目标输出维度
        context_dim: int,  # 输入上下文维度
        spatial_merge_size: int = 2,  # 空间合并大小，默认2
        pre_norm="layernorm",  # 预归一化类型，默认LayerNorm
        init_merger_std=None,  # 合并器初始化标准差
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = context_dim * (spatial_merge_size**2)  # 隐藏大小 = 上下文维度 × 合并面积的平方
        self.pre_norm = pre_norm  # 保存预归一化类型
        if self.pre_norm == "layernorm":  # 如果使用LayerNorm
            self.ln_q = LayerNorm(context_dim, eps=1e-6)  # 创建LayerNorm
        elif self.pre_norm == "rmsnorm":  # 如果使用RMSNorm
            self.ln_q = RMSNorm(context_dim, eps=1e-6)  # 创建RMSNorm
        else:  # 其他情况
            logger.warning(f"no norm in patch merger: {self.pre_norm}")  # 记录警告：合并器中没有归一化

        self.mlp = nn.Sequential(  # 创建MLP序列
            nn.Linear(self.hidden_size, self.hidden_size),  # 第一层线性变换
            nn.GELU(),  # GELU激活函数
            nn.Linear(self.hidden_size, dim),  # 第二层线性变换，投影到目标维度
        )

        if init_merger_std is not None:  # 如果指定了初始化标准差
            nn.init.normal_(self.mlp[0].weight, mean=0.0, std=init_merger_std)  # 正态初始化第一层权重
            nn.init.zeros_(self.mlp[0].bias)  # 零初始化第一层偏置
            nn.init.normal_(self.mlp[2].weight, mean=0.0, std=init_merger_std)  # 正态初始化第二层权重
            nn.init.zeros_(self.mlp[2].bias)  # 零初始化第二层偏置

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        if self.pre_norm:  # 如果有预归一化
            x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))  # 先归一化再重塑后通过MLP
        else:  # 否则
            x = self.mlp(x.view(-1, self.hidden_size))  # 直接重塑后通过MLP
        return x  # 返回合并后的特征


class RMSNorm(nn.Module):
    """RMS归一化层，Dots视觉模型专用实现"""
    def __init__(self, dim: int, eps: float = 1e-6):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数，初始化为1
        self.eps = eps  # 防止除零的epsilon值

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        output = self._norm(x.float()).type_as(x)  # 在float32下归一化后转回原数据类型
        return output * self.weight  # 乘以可学习缩放参数

    def extra_repr(self) -> str:  # 模型额外表示信息
        return f"{tuple(self.weight.shape)}, eps={self.eps}"  # 返回形状和epsilon信息

    def _norm(self, x: torch.Tensor) -> torch.Tensor:  # 归一化核心计算
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # RMS归一化：x / sqrt(mean(x^2) + eps)


class DotsSwiGLUFFN(nn.Module):
    """Dots SwiGLU前馈网络，使用SwiGLU激活函数"""
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):  # 初始化方法
        super().__init__()  # 调用父类初始化
        hidden_features = config.intermediate_size  # 获取中间层维度
        in_features = config.embed_dim  # 获取输入维度
        bias = config.use_bias  # 获取是否使用偏置

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)  # 第一层线性变换（门控路径）
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)  # 第二层线性变换（输出路径）
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)  # 第三层线性变换（上投影路径）

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        x = F.silu(self.fc1(x)) * self.fc3(x)  # SwiGLU: silu(fc1(x)) * fc3(x)
        x = self.fc2(x)  # 输出线性变换
        return x  # 返回前馈网络输出


class DotsPatchEmbed(nn.Module):
    """Dots Patch嵌入层，将图像分割为patch并嵌入"""
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.num_channels = config.num_channels  # 输入通道数
        self.patch_size = config.patch_size  # Patch大小
        self.temporal_patch_size = config.temporal_patch_size  # 时间Patch大小
        self.embed_dim = config.embed_dim  # 嵌入维度
        self.config = config  # 保存配置
        self.proj = Conv2dLayer(  # 创建2D卷积投影层
            config.num_channels,  # 输入通道数
            config.embed_dim,  # 输出通道数（嵌入维度）
            kernel_size=(config.patch_size, config.patch_size),  # 卷积核大小等于Patch大小
            stride=(config.patch_size, config.patch_size),  # 步幅等于Patch大小，实现非重叠分割
        )
        self.norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)  # Patch嵌入后归一化

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:  # 前向传播方法
        x = x.view(  # 重塑输入形状
            -1,  # 自动推断batch维度
            self.num_channels,  # 通道数
            self.temporal_patch_size,  # 时间Patch大小
            self.patch_size,  # 高度Patch大小
            self.patch_size,  # 宽度Patch大小
        )[:, :, 0]  # 取第一个时间帧
        x = self.proj(x).view(-1, self.embed_dim)  # 通过卷积投影并展平
        x = self.norm(x)  # 归一化
        return x  # 返回Patch嵌入


class DotsViTPreprocessor(nn.Module):
    """Dots ViT预处理器，封装Patch嵌入层"""
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.patch_h = config.patch_size  # Patch高度
        self.patch_w = config.patch_size  # Patch宽度
        self.embed_dim = config.embed_dim  # 嵌入维度
        self.config = config  # 保存配置
        self.patchifier = DotsPatchEmbed(config, quant_config)  # 创建Patch嵌入层

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:  # 前向传播方法
        tokens = self.patchifier(x, grid_thw)  # 通过Patch嵌入层获取token
        return tokens  # 返回token


class DotsVisionBlock(nn.Module):
    """Dots视觉Transformer块，包含自注意力和前馈网络"""
    def __init__(  # 初始化方法
        self,
        config: DotsVisionConfig,  # 视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数名前缀
    ):
        super().__init__()  # 调用父类初始化
        self.attn = VisionAttention(  # 创建视觉注意力层
            embed_dim=config.embed_dim,  # 嵌入维度
            num_heads=config.num_attention_heads,  # 注意力头数
            projection_size=config.embed_dim,  # 投影大小
            use_qkv_parallel=True,  # 使用QKV并行
            flatten_batch=True,  # 展平批次
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 添加前缀
            num_dummy_heads=config.num_dummy_heads,  # 虚拟头数量
            qkv_bias=config.use_bias,  # QKV偏置
            proj_bias=config.use_bias,  # 输出投影偏置
        )
        self.norm1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)  # 注意力前归一化
        self.mlp = DotsSwiGLUFFN(config, quant_config)  # SwiGLU前馈网络
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)  # 前馈网络前归一化

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:  # 前向传播方法
        hidden_states = hidden_states + self.attn(  # 残差连接：隐藏状态 + 注意力输出
            self.norm1(hidden_states),  # 归一化后的隐藏状态作为注意力输入
            cu_seqlens=cu_seqlens,  # 序列累积长度
            position_embeddings=rotary_pos_emb,  # 旋转位置编码
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))  # 残差连接：隐藏状态 + 前馈网络输出
        return hidden_states  # 返回更新后的隐藏状态


class DotsVisionTransformer(PreTrainedModel):
    """Dots视觉Transformer完整模型，处理图像/视频输入并输出视觉嵌入"""
    def __init__(  # 初始化方法
        self,
        config: DotsVisionConfig,  # 视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
    ) -> None:
        super().__init__(config)  # 调用父类初始化
        self.config = config  # 保存配置
        self._update_vision_config()  # 更新视觉配置以支持张量并行
        self.spatial_merge_size = config.spatial_merge_size  # 空间合并大小

        self.patch_embed = DotsViTPreprocessor(config, quant_config)  # 创建Patch嵌入预处理器
        self._init_weights(self.patch_embed.patchifier.proj)  # 初始化卷积投影层权重

        head_dim = config.embed_dim // config.num_attention_heads  # 计算每个头的维度

        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)  # 创建视觉旋转位置编码

        _num_hidden_layers = config.num_hidden_layers  # 获取隐藏层数量
        self.blocks = nn.ModuleList(  # 创建视觉Transformer块列表
            [
                DotsVisionBlock(config, quant_config, f"blocks.{i}")  # 为每层创建视觉块
                for i in range(_num_hidden_layers)  # 遍历所有层
            ]
        )

        if self.config.post_norm:  # 如果启用后归一化
            self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)  # 创建主干后归一化层

        self.merger = PatchMerger(  # 创建Patch合并器
            dim=config.hidden_size,  # 输出维度
            context_dim=config.embed_dim,  # 输入上下文维度
            spatial_merge_size=config.spatial_merge_size,  # 空间合并大小
            init_merger_std=self.config.init_merger_std,  # 合并器初始化标准差
            quant_config=quant_config,  # 量化配置
        )

        self.gradient_checkpointing = False  # 禁用梯度检查点

    def _update_vision_config(self):  # 更新视觉配置以支持张量并行
        """update vision config to support tp
        更新视觉配置以支持张量并行"""
        world_size = parallel_state.get_tensor_model_parallel_world_size()  # 获取张量并行世界大小
        num_heads = self.config.num_attention_heads  # 获取注意力头数
        head_dim = self.config.embed_dim // num_heads  # 计算每个头的维度
        num_dummy_heads = 0  # 初始化虚拟头数为0

        if num_heads % world_size != 0:  # 如果头数不能被世界大小整除
            num_dummy_heads = (  # 计算需要的虚拟头数
                (num_heads + world_size) // world_size  # 向上取整到世界大小的倍数
            ) * world_size - num_heads  # 减去实际头数得到虚拟头数

        setattr(self.config, "head_dim", head_dim)  # 设置头维度属性
        setattr(self.config, "num_dummy_heads", num_dummy_heads)  # 设置虚拟头数属性

    def _init_weights(self, module):  # 权重初始化方法
        std = self.config.initializer_range  # 获取初始化标准差
        if isinstance(module, (nn.Linear, nn.Conv2d)):  # 如果是线性层或卷积层
            module.weight.data.normal_(mean=0.0, std=std)  # 正态分布初始化权重
            if module.bias is not None:  # 如果有偏置
                module.bias.data.zero_()  # 零初始化偏置
        elif isinstance(module, nn.Embedding):  # 如果是嵌入层
            module.weight.data.normal_(mean=0.0, std=std)  # 正态分布初始化权重
            if module.padding_idx is not None:  # 如果有填充索引
                module.weight.data[module.padding_idx].zero_()  # 填充位置的权重置零

    @property  # 属性装饰器
    def dtype(self) -> torch.dtype:  # 获取模型数据类型
        return self.blocks[0].mlp.fc2.weight.dtype  # 从第一个块的MLP第二层获取数据类型

    @property  # 属性装饰器
    def device(self) -> torch.device:  # 获取模型设备
        return self.blocks[0].mlp.fc2.weight.device  # 从第一个块的MLP第二层获取设备

    def get_pos_ids_by_grid(self, grid_thw):  # 根据网格信息获取位置ID
        pos_ids = []  # 位置ID列表
        for t, h, w in grid_thw:  # 遍历每个图像的时间、高度、宽度
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)  # 生成高度方向位置ID并扩展到宽度
            hpos_ids = hpos_ids.reshape(  # 重塑为合并格式
                h // self.spatial_merge_size,  # 高度方向合并后的维度
                self.spatial_merge_size,  # 合并窗口高度
                w // self.spatial_merge_size,  # 宽度方向合并后的维度
                self.spatial_merge_size,  # 合并窗口宽度
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)  # 置换维度：合并块间交替排列
            hpos_ids = hpos_ids.flatten()  # 展平为一维

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)  # 生成宽度方向位置ID并扩展到高度
            wpos_ids = wpos_ids.reshape(  # 重塑为合并格式
                h // self.spatial_merge_size,  # 高度方向合并后的维度
                self.spatial_merge_size,  # 合并窗口高度
                w // self.spatial_merge_size,  # 宽度方向合并后的维度
                self.spatial_merge_size,  # 合并窗口宽度
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)  # 置换维度：合并块间交替排列
            wpos_ids = wpos_ids.flatten()  # 展平为一维
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))  # 堆叠高度和宽度位置ID，按时间帧重复

        return pos_ids  # 返回位置ID列表

    def rot_pos_emb(self, grid_thw):  # 计算旋转位置编码
        pos_ids = self.get_pos_ids_by_grid(grid_thw)  # 获取位置ID
        pos_ids = torch.cat(pos_ids, dim=0)  # 拼接所有图像的位置ID
        max_grid_size = grid_thw[:, 1:].max()  # 获取最大网格尺寸（高度或宽度中的最大值）
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)  # 生成完整的旋转位置编码
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)  # 根据位置ID索引并展平
        return rotary_pos_emb  # 返回旋转位置编码

    def calc_cos_sin(self, rotary_pos_emb):  # 计算旋转位置编码的余弦和正弦值
        cos = rotary_pos_emb.cos()  # 计算余弦值
        sin = rotary_pos_emb.sin()  # 计算正弦值
        cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()  # 扩展维度并转为float32
        sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()  # 扩展维度并转为float32
        rotary_pos_emb = (cos, sin)  # 组合为元组
        return rotary_pos_emb  # 返回(cos, sin)元组

    def forward(  # 前向传播方法
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, bf16=True  # 输入、网格信息、是否使用bf16
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.device)  # 将输入移到模型设备
        if bf16:  # 如果使用bfloat16
            hidden_states = hidden_states.bfloat16()  # 转换为bfloat16
        hidden_states = self.patch_embed(hidden_states, grid_thw)  # 通过Patch嵌入预处理

        rotary_pos_emb = self.rot_pos_emb(grid_thw)  # 计算旋转位置编码
        rotary_pos_emb = self.calc_cos_sin(rotary_pos_emb)  # 计算余弦和正弦值

        cu_seqlens = torch.repeat_interleave(  # 计算序列累积长度
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]  # 每个图像的patch数 = h * w，按时间帧数重复
        ).cumsum(  # 计算累积和
            dim=0,  # 沿第0维
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,  # 追踪模式下使用原始类型，否则int32
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])  # 在开头添加0，使第一个序列从0开始
        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
        # cu_seqlens必须在CPU上，因为npu_flash_attention_unpad算子的限制
        if is_npu():  # 如果是NPU环境
            cu_seqlens = cu_seqlens.to("cpu")  # 将序列长度移到CPU

        for blk in self.blocks:  # 遍历所有视觉Transformer块
            hidden_states = blk(  # 通过每个块
                hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb  # 传入隐藏状态、序列长度和旋转位置编码
            )

        if self.config.post_norm:  # 如果启用后归一化
            hidden_states = self.post_trunk_norm(hidden_states)  # 通过主干后归一化层

        hidden_states = self.merger(hidden_states)  # 通过Patch合并器
        return hidden_states  # 返回最终的视觉嵌入
