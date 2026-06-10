# Kimi K2.5 多模态视觉语言模型实现
# 该文件实现了 Kimi K2.5 条件生成模型，结合 MoonViT3d 视觉编码器（支持3D视频/图像输入）、
# K2VL多模态投影器和 DeepseekV3 语言模型，支持2D旋转位置编码、
# 时间池化patch合并、数据并行视觉编码等功能。

import logging  # 导入日志模块
from copy import deepcopy  # 导入深拷贝
from typing import Iterable, List, Optional, Sequence, Tuple  # 导入类型注解

import numpy as np  # 导入NumPy
import torch  # 导入PyTorch
import torch.nn.functional as F  # 导入神经网络函数模块
from torch import nn  # 导入神经网络模块
from transformers import activations  # 导入激活函数模块

from sglang.srt.configs.kimi_k25 import KimiK25Config, KimiK25VisionConfig  # 导入Kimi K25配置
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation  # 导入专家位置配置
from sglang.srt.layers.conv import Conv2dLayer  # 导入2D卷积层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)

try:  # 尝试导入PytorchGELUTanh
    from transformers.activations import PytorchGELUTanh  # 导入近似GELU激活
except ImportError:  # 如果导入失败
    from transformers.activations import GELUTanh  # 导入标准GELU激活

    activations.PytorchGELUTanh = GELUTanh  # 兼容性别名
    PytorchGELUTanh = GELUTanh  # 兼容性别名

from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.layers.linear import ReplicatedLinear  # 导入复制线性层
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig  # 导入ModelSlim量化配置
from sglang.srt.layers.quantization.quark.quark import QuarkConfig  # 导入Quark量化配置
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM  # 导入DeepseekV3语言模型
from sglang.srt.models.kimi_vl_moonvit import MLP2  # 导入MLP2模块
from sglang.srt.models.utils import WeightsMapper  # 导入权重映射器
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model  # 导入数据并行视觉模型运行
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import add_prefix, is_npu  # 导入工具函数

logger = logging.getLogger(__name__)  # 获取日志记录器

from sglang.srt.layers.dp_attention import is_dp_attention_enabled  # 导入DP注意力使能检查

_is_npu = is_npu()  # 检查是否为NPU设备


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, x_shape=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """对查询和键张量应用2D旋转位置编码（RoPE）。"""
    """
    Args: (The leading dimensions of all inputs should be the same)  # 所有输入的前导维度应相同
        xq: query, tensor of shape (..., num_heads, head_dim)  # 查询张量
        xk: key, tensor of shape (..., num_heads, head_dim)  # 键张量
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.  # 预计算的复数旋转频率
    Returns:  # 返回值
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)  # 应用RoPE后的查询和键
    """

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2 扩展维度以匹配多头
    # ..., num_heads, head_dim/2  # 将查询转换为复数视图
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))  # 将Q转为复数
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))  # 将K转为复数
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim 复数乘法后转回实数
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim 复数乘法后转回实数
    return xq_out.type_as(xq), xk_out.type_as(xk)  # 转回原始数据类型并返回


def tpool_patch_merger(
    x: torch.Tensor,  # 输入张量
    grid_thws: torch.Tensor,  # 网格时间-高度-宽度
    merge_kernel_size: tuple[int, int] = (2, 2),  # 合并核大小
) -> list[torch.Tensor]:
    """时间池化patch合并器，沿空间维度合并patch并进行时间维度池化。"""
    d_model = x.size(-1)  # 获取模型维度

    outputs = []  # 输出列表
    pre_sum = 0  # 前序累计和
    for t, h, w in grid_thws.tolist():  # 遍历每个样本的网格尺寸
        # Get the current sequence  # 获取当前序列
        seq = x[pre_sum : pre_sum + t * h * w]  # 截取当前样本的序列
        # Reshape along self.merge_kernel_size and concat to the last dimension  # 按合并核大小重排
        kernel_height, kernel_width = merge_kernel_size  # 获取合并核的高宽
        new_height, new_width = h // kernel_height, w // kernel_width  # 计算合并后的高宽
        reshaped_seq = seq.view(  # 重排序列形状
            t, new_height, kernel_height, new_width, kernel_width, d_model
        )
        reshaped_seq = (
            reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        )  # temporal pooling  # 时间维度池化（对时间维度求均值）
        padded_seq = reshaped_seq.view(  # 重排为合并后的形状
            new_height * new_width, kernel_height * kernel_width, -1
        )
        outputs.append(padded_seq)  # 添加到输出列表
        pre_sum += t * h * w  # 更新前序累计和

    return outputs  # 返回合并后的patch列表


class MoonViTEncoderLayer(nn.Module):
    """MoonViT编码器层，包含自注意力和MLP。"""

    def __init__(
        self,
        num_heads: int,  # 注意力头数
        hidden_dim: int,  # 隐藏层维度
        mlp_dim: int,  # MLP中间维度
        *,  # 以下为关键字参数
        activation=F.gelu,  # 激活函数
        attn_bias: bool = False,  # 注意力偏置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        use_data_parallel: bool = False,  # 是否使用数据并行
    ):
        super().__init__()  # 调用父类初始化
        self.num_heads = num_heads  # 保存头数
        self.hidden_dim = hidden_dim  # 保存隐藏维度
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads  # 每个注意力头的维度

        self.norm0 = nn.LayerNorm(hidden_dim)  # 注意力前的层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)  # MLP前的层归一化

        self.mlp = MLP2(  # MLP模块
            [hidden_dim, mlp_dim, hidden_dim],  # 维度列表
            activation,  # 激活函数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )

        self.attn = VisionAttention(  # 视觉注意力模块
            embed_dim=hidden_dim,  # 嵌入维度
            num_heads=num_heads,  # 头数
            projection_size=hidden_dim,  # 投影维度
            use_qkv_parallel=True,  # 使用QKV并行
            qkv_bias=attn_bias,  # QKV偏置
            proj_bias=attn_bias,  # 输出投影偏置
            flatten_batch=True,  # 展平批次
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
            use_data_parallel=use_data_parallel,  # 数据并行
            customized_position_embedding_applier=apply_rope,  # 自定义位置编码应用函数
            use_dp_attention_reduce=is_dp_attention_enabled(),  # 是否使用DP注意力归约
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        cu_seqlens: torch.Tensor,  # 累计序列长度
        max_seqlen: int,  # 最大序列长度
        rope_freqs_cis: torch.Tensor | None = None,  # RoPE复数频率
    ):
        """前向传播：执行注意力+MLP的编码器层计算。"""
        residual = hidden_states  # 保存残差
        hidden_states = self.norm0(hidden_states)  # 注意力前归一化

        hidden_states = self.attn(  # 执行视觉注意力
            hidden_states,  # 归一化后的隐藏状态
            cu_seqlens=cu_seqlens,  # 累计序列长度
            position_embeddings=rope_freqs_cis,  # 旋转位置编码
        )

        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 保存残差
        hidden_states = self.norm1(hidden_states)  # MLP前归一化
        hidden_states = self.mlp(hidden_states)  # 通过MLP
        hidden_states = residual + hidden_states  # 残差连接

        return hidden_states  # 返回隐藏状态


def get_rope_shape_decorate(func):
    """装饰器：在首次调用时预热RoPE形状计算。"""
    _get_rope_shape_first_call_flag = set()  # 首次调用标记集合

    def wrapper(org, interpolation_mode, shape):  # 包装函数
        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)  # 生成唯一键
        if key not in _get_rope_shape_first_call_flag:  # 如果首次调用
            _get_rope_shape_first_call_flag.add(key)  # 标记已调用
            _ = func(org, interpolation_mode, shape=(64, 64))  # 使用默认形状预热
        return func(org, interpolation_mode, shape)  # 调用原始函数

    return wrapper  # 返回包装函数


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """从1D网格生成正弦-余弦位置编码。"""
    """
    From:  # 来源
    https://github.com/OpenGVLab/InternVideo/blob/421f6d2361fc8f61a3394244571f2601a4e99e29/InternVideo2/multi_modality/models/backbones/internvideo2/pos_embed.py#L86
    embed_dim: output dimension for each position  # 每个位置的输出维度
    pos: a list of positions to be encoded: size (M,)  # 待编码的位置列表
    out: (M, D)  # 输出形状
    """
    assert embed_dim % 2 == 0  # 确保嵌入维度为偶数
    omega = np.arange(embed_dim // 2, dtype=np.float32)  # 生成频率索引
    omega /= embed_dim / 2.0  # 归一化频率索引
    omega = 1.0 / 10000**omega  # (D/2,) 计算频率

    pos = pos.reshape(-1)  # (M,) 展平位置
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product 外积计算

    emb_sin = np.sin(out)  # (M, D/2) 正弦编码
    emb_cos = np.cos(out)  # (M, D/2) 余弦编码

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D) 拼接正弦和余弦编码
    return emb  # 返回位置编码


@get_rope_shape_decorate  # 应用RoPE形状预热装饰器
@torch.compile(dynamic=True, disable=_is_npu)  # 使用torch编译优化（NPU上禁用）
def get_rope_shape(org, interpolation_mode, shape):
    """获取RoPE形状：通过插值调整位置编码的网格大小。"""
    return (  # 返回插值后的位置编码
        F.interpolate(  # 双线性/双三次插值
            org.permute((2, 0, 1)).unsqueeze(0),  # 调整维度顺序并增加batch维度
            size=shape,  # 目标形状
            mode=interpolation_mode,  # 插值模式
        )
        .squeeze(0)  # 去除batch维度
        .permute((1, 2, 0))  # 恢复维度顺序
        .flatten(end_dim=1)  # 展平前两个维度
    )


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """生成1D正弦-余弦位置编码，用于时间维度。"""
    """
    t_size: int of the temporal size  # 时间维度大小
    return:  # 返回
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)  # 位置编码
    """
    grid_t = np.arange(t_size, dtype=np.float32)  # 生成时间位置索引
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)  # 从网格生成编码
    if cls_token:  # 如果包含CLS token
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)  # 在前面添加零向量
    return pos_embed  # 返回位置编码


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    """可学习的2D插值位置编码（分块固定版），支持多分辨率和时间维度。"""

    def __init__(
        self,
        height: int,  # 初始高度
        width: int,  # 初始宽度
        num_frames: int,  # 帧数
        dim: int,  # 编码维度
        interpolation_mode: str = "bicubic",  # 插值模式
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.height = height  # 保存高度
        self.width = width  # 保存宽度
        self.num_frames = num_frames  # 保存帧数
        self.dim = dim  # 保存维度
        self.interpolation_mode = interpolation_mode  # 保存插值模式
        self.weight = nn.Parameter(torch.empty(height, width, dim))  # 可学习的2D位置编码权重
        self.register_buffer(  # 注册时间权重缓冲区（不参与梯度更新）
            "time_weight",  # 缓冲区名称
            torch.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames))  # 从正弦余弦编码生成
            .float()  # 转为浮点型
            .unsqueeze(1),  # 增加维度
            persistent=False,  # 不持久化保存
        )

        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):
        """重置参数：使用正态分布初始化位置编码权重。"""
        nn.init.normal_(self.weight)  # 正态分布初始化

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """前向传播：根据网格尺寸计算位置编码并加到输入上。"""
        pos_embs = []  # 位置编码列表
        for t, h, w in grid_thws.tolist():  # 遍历每个样本的网格尺寸
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"  # 检查帧数不超限
            if (h, w) == self.weight.shape[:-1]:  # 如果尺寸匹配权重
                pos_emb_2d = self.weight.flatten(end_dim=1)  # 直接展平
            else:  # 否则需要插值
                pos_emb_2d = get_rope_shape(  # 通过插值获取位置编码
                    self.weight,  # 原始权重
                    interpolation_mode=self.interpolation_mode,  # 插值模式
                    shape=(h, w),  # 目标形状
                )

            if t == 1:  # 如果只有一帧
                pos_emb_3d = pos_emb_2d  # 不添加时间编码
            else:  # 多帧情况
                pos_emb_3d = (  # 添加时间编码
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]  # 复制并加上时间权重
                )

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))  # 展平并添加到列表

        out = x + torch.cat(pos_embs)  # 将位置编码加到输入上
        return out  # 返回结果


class Rope2DPosEmbRepeated(nn.Module):
    """2D旋转位置编码，支持多分辨率。
    This class is intended to be used in the following way:  # 使用方式
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.  # 训练前创建实例
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.  # 前向传播前获取频率
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.  # 传递给注意力层
        The rope is shared across all attention layers and all heads.  # RoPE在所有层和头之间共享
    Refs:  # 参考
    - RoFormer: https://arxiv.org/abs/2104.09864  # RoFormer论文
    - VisionLLaMA: https://arxiv.org/abs/2403.00522  # VisionLLaMA论文
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py  # 实现代码
    Args:  # 参数
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)  # 注意力维度，需被4整除
        max_height (int): the maximum height of the 2D grid  # 2D网格最大高度
        max_width (int): the maximum width of the 2D grid  # 2D网格最大宽度
        theta_base (float): the base of the theta  # theta的基数
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):  # 初始化
        super().__init__()  # 调用父类初始化
        self.dim = dim  # 保存维度
        assert self.dim % 4 == 0, "dim must be divisible by 4"  # 确保维度可被4整除
        self.max_height = max_height  # 保存最大高度
        self.max_width = max_width  # 保存最大宽度
        self.theta_base = theta_base  # 保存theta基数

    def extra_repr(self):
        """返回模块的额外表示字符串。"""
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"  # 格式化输出

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """预计算2D网格中每个位置的复数旋转频率。"""
        """Calculate the cis(freqs) for each position in the 2D grid.  # 计算2D网格中每个位置的cis(freqs)
        Return: complex tensor of shape (max_height, max_width, dim//2) and value:  # 返回复数张量
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))  # 高度轴编码
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))  # 宽度轴编码
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,  # cis数学记号
        """
        N = self.max_height * self.max_width  # 总位置数
        flat_pos = torch.arange(0, N).float().to(device)  # 展平的位置索引
        x_pos = flat_pos % self.max_width  # x坐标（宽度方向）
        y_pos = flat_pos // self.max_width  # y坐标（高度方向）
        dim_range = (  # 维度范围
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4 每4个维度取一个频率
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))  # 计算频率
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4 x方向的频率
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4 y方向的频率
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4 x方向的复数旋转
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4 y方向的复数旋转
        # N, C/4, 2  # 拼接x和y的复数旋转
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2  # 重排为2D网格形状
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis  # 返回预计算的复数频率

    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """根据网格尺寸获取对应的复数旋转频率。"""
        """
        Args:  # 参数
            grid_thws (torch.Tensor): grid time, height and width  # 网格时间、高度和宽度
        Returns:  # 返回
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)  # 复数频率张量
        """
        if not hasattr(self, "freqs_cis"):  # 如果尚未缓存
            self.register_buffer(  # 注册缓冲区
                "freqs_cis", self._precompute_freqs_cis(device), persistent=False  # 预计算并缓存
            )

        shapes = grid_thws.tolist()  # 获取网格形状列表
        assert all(  # 验证所有形状在合法范围内
            1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes
        ), (
            shapes,  # 当前形状
            self.max_height,  # 最大高度
            self.max_width,  # 最大宽度
        )
        freqs_cis = torch.cat(  # 拼接所有样本的频率
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)  # 截取并在时间维度复制
                for t, h, w in shapes  # 遍历每个形状
            ],
            dim=0,  # 在序列维度拼接
        )
        return freqs_cis  # 返回复数频率


class MoonVision3dPatchEmbed(nn.Module):
    """MoonVision 3D Patch嵌入模块，将图像/视频分割为patch并添加位置编码。"""

    def __init__(
        self,
        out_dim: int,  # 输出维度
        in_dim: int = 3,  # 输入通道数
        patch_size: int | tuple[int, int] = (14, 14),  # patch大小
        pos_emb_height: int = 14,  # 位置编码高度
        pos_emb_width: int = 14,  # 位置编码宽度
        pos_emb_time: int = 4,  # 位置编码时间维度
        pos_emb_type: str = "divided_fixed",  # 位置编码类型
    ):
        super().__init__()  # 调用父类初始化
        assert isinstance(  # 验证patch_size类型
            patch_size, int | Sequence
        ), f"Invalid patch_size type: {type(patch_size)}"
        if isinstance(patch_size, int):  # 如果是整数
            patch_size = (patch_size, patch_size)  # 转为元组
        assert (  # 验证patch_size长度
            len(patch_size) == 2
        ), f"Expected patch_size to be a tuple of 2, got {patch_size}"
        self.patch_size = patch_size  # 保存patch大小

        self.proj = Conv2dLayer(  # 2D卷积投影层
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size  # 使用patch_size作为卷积核和步长
        )

        if pos_emb_type == "divided_fixed":  # 分块固定位置编码
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(  # 创建位置编码模块
                height=pos_emb_height,  # 高度
                width=pos_emb_width,  # 宽度
                num_frames=pos_emb_time,  # 帧数
                dim=out_dim,  # 维度
            )
        else:  # 其他类型
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")  # 不支持

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """前向传播：将输入投影为patch并添加位置编码。"""
        """
        Args:  # 参数
            x (L, Channels): input tensor  # 输入张量
            grid_hws (N, 3): temporal, height and width  # 时间、高度和宽度
        Returns:  # 返回
            (L, Cout) tensor  # 输出张量
        """
        x = self.proj(x).view(x.size(0), -1)  # 通过卷积投影并展平
        # apply positional embedding  # 应用位置编码
        x = self.pos_emb(x, grid_thws)  # 添加位置编码
        return x  # 返回带位置编码的patch


class MoonViT3dEncoder(nn.Module):
    """MoonViT 3D编码器，使用2D RoPE的Transformer编码器。"""

    def __init__(
        self,
        hidden_dim: int,  # 隐藏维度
        num_layers: int,  # 层数
        block_cfg: dict,  # 块配置
        video_attn_type: str = "spatial_temporal",  # 视频注意力类型
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        assert (  # 验证注意力类型
            video_attn_type == "spatial_temporal"
        ), f'video_attn_type must be "spatial_temporal", got {video_attn_type}'
        self.video_attn_type = video_attn_type  # 保存注意力类型
        self.rope_2d = Rope2DPosEmbRepeated(  # 创建2D RoPE
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512  # 维度和最大分辨率
        )
        self.blocks = nn.ModuleList(  # 创建编码器层列表
            [
                MoonViTEncoderLayer(  # 编码器层
                    **block_cfg,  # 块配置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),  # 参数前缀
                )
                for layer_idx in range(num_layers)  # 遍历层数
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)  # 最终层归一化

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        grid_thws: torch.Tensor,  # 网格时间-高度-宽度
    ) -> torch.Tensor:
        """前向传播：通过所有编码器层处理隐藏状态。"""
        rope_freqs_cis = self.rope_2d.get_freqs_cis(  # 获取2D RoPE频率
            grid_thws=grid_thws, device=hidden_states.device  # 网格尺寸和设备
        )

        lengths = torch.cat(  # 计算每个样本的序列长度
            (
                torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),  # 起始0
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],  # t*h*w
            )
        )

        max_seqlen = lengths.max()  # 最大序列长度
        cu_seqlens = lengths.to(hidden_states.device).cumsum(dim=0, dtype=torch.int32)  # 累计序列长度

        for block in self.blocks:  # 遍历所有编码器层
            hidden_states = block(  # 通过编码器层
                hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis  # 传入参数
            )

        hidden_states = self.final_layernorm(hidden_states)  # 最终层归一化

        return hidden_states  # 返回编码后的隐藏状态


class MoonViT3dPretrainedModel(nn.Module):
    """MoonViT 3D预训练模型，包含patch嵌入和Transformer编码器。"""
    model_type = "moonvit3d"  # 模型类型
    _no_split_modules = ["PackingTransformer"]  # 不可分割模块
    _supports_flash_attn_2 = True  # 支持Flash Attention 2
    _supports_sdpa = True  # 支持SDPA

    def __init__(
        self,
        config,  # 模型配置
        *inputs,  # 输入参数
        use_data_parallel: bool = False,  # 是否使用数据并行
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        **kwargs,  # 其他关键字参数
    ):
        super().__init__()  # 调用父类初始化
        config = deepcopy(config)  # 深拷贝配置
        self.config = config  # 保存配置
        self.merge_kernel_size = config.merge_kernel_size  # 合并核大小
        self.patch_size = config.patch_size  # patch大小
        self.merge_type = config.merge_type  # 合并类型

        self.patch_embed = MoonVision3dPatchEmbed(  # 创建patch嵌入模块
            out_dim=config.hidden_size,  # 输出维度
            patch_size=config.patch_size,  # patch大小
            pos_emb_height=config.init_pos_emb_height,  # 初始位置编码高度
            pos_emb_width=config.init_pos_emb_width,  # 初始位置编码宽度
            pos_emb_time=config.init_pos_emb_time,  # 初始位置编码时间
            pos_emb_type=config.pos_emb_type,  # 位置编码类型
        )

        self.encoder = MoonViT3dEncoder(  # 创建编码器
            hidden_dim=config.hidden_size,  # 隐藏维度
            num_layers=config.num_hidden_layers,  # 层数
            block_cfg={  # 块配置
                "num_heads": config.num_attention_heads,  # 头数
                "hidden_dim": config.hidden_size,  # 隐藏维度
                "mlp_dim": config.intermediate_size,  # MLP中间维度
                "activation": PytorchGELUTanh(),  # 激活函数
                "attn_bias": True,  # 使用注意力偏置
                "use_data_parallel": use_data_parallel,  # 数据并行
            },
            video_attn_type=config.video_attn_type,  # 视频注意力类型
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("encoder", prefix),  # 参数前缀
        )

    @property
    def dtype(self) -> torch.dtype:
        """获取模型的数据类型。"""
        return self.patch_embed.proj.weight.dtype  # 返回patch嵌入的权重数据类型

    @property
    def device(self) -> torch.device:
        """获取模型所在的设备。"""
        return self.patch_embed.proj.weight.device  # 返回patch嵌入的权重所在设备

    def forward(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        """前向传播：处理图像/视频像素值，返回视觉token。"""
        """
        Args:  # 参数
            pixel_values (torch.Tensor): The input pixel values.  # 输入像素值
            grid_thws (torch.Tensor): Temporal, height and width.  # 时间、高度和宽度
        Returns:  # 返回
            torch.Tensor: The output tokens.  # 输出token
        """
        assert grid_thws.ndim == 2, f"grid_thws should be 2D, got {grid_thws.ndim}"  # 验证维度
        assert grid_thws.size(1) == 3, f"No support for _thw: {grid_thws}"  # 验证列数
        hidden_states = self.patch_embed(pixel_values, grid_thws)  # patch嵌入
        hidden_states = self.encoder(hidden_states, grid_thws)  # 编码器
        hidden_states = hidden_states.squeeze(0)  # 去除多余维度
        # spatial downsampling 2x with temporal pooling all  # 2倍空间下采样并进行时间池化
        hidden_states = tpool_patch_merger(  # 时间池化patch合并
            hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size  # 传入参数
        )

        return hidden_states  # 返回合并后的视觉token列表


class K2VLMultiModalProjector(nn.Module):
    """K2-VL多模态投影器，将视觉特征投影到语言模型空间。"""
    """Multi-modal projector with patch merging for K2-VL."""  # 带patch合并的多模态投影器

    def __init__(
        self,
        config: KimiK25VisionConfig,  # K2视觉配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化

        # Hidden size after patch merging  # patch合并后的隐藏大小
        merge_h, merge_w = config.merge_kernel_size  # 获取合并核大小
        self.hidden_size = config.vt_hidden_size * merge_h * merge_w  # 计算投影器隐藏大小

        self.pre_norm = torch.nn.LayerNorm(config.vt_hidden_size, eps=1e-5)  # 预归一化
        self.linear_1 = ReplicatedLinear(  # 第一层线性变换
            self.hidden_size,  # 输入维度
            self.hidden_size,  # 输出维度
            bias=True,  # 使用偏置
            prefix=add_prefix(prefix, "linear_1"),  # 参数前缀
        )
        self.linear_2 = ReplicatedLinear(  # 第二层线性变换
            self.hidden_size,  # 输入维度
            config.text_hidden_size,  # 输出维度（文本隐藏大小）
            bias=True,  # 使用偏置
            prefix=add_prefix(prefix, "linear_2"),  # 参数前缀
        )
        self.act = nn.GELU()  # GELU激活函数

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """前向传播：将视觉特征投影到语言模型空间。"""
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)  # 预归一化并重塑
        hidden_states, _ = self.linear_1(hidden_states)  # 第一层线性变换
        hidden_states = self.act(hidden_states)  # 激活函数
        hidden_states, _ = self.linear_2(hidden_states)  # 第二层线性变换
        return hidden_states  # 返回投影后的特征


@torch.inference_mode()  # 推理模式
def mm_projection_auto(
    mm_projector: torch.nn.Module | None, vt_output: list[torch.Tensor]
):
    """自动应用多模态投影器到视觉塔输出。"""
    """Apply MM projector to vision tower outputs."""  # 将多模态投影器应用于视觉塔输出
    if mm_projector is None:  # 如果没有投影器
        return vt_output  # 直接返回视觉塔输出

    num_embedding_list = [x.shape[0] for x in vt_output]  # 每个样本的token数
    batched = torch.cat(vt_output, dim=0)  # 拼接所有样本
    proj_out = mm_projector(batched) if mm_projector else batched  # 通过投影器
    proj_out = proj_out.reshape(-1, proj_out.shape[-1])  # 重塑形状
    proj_out = torch.split(proj_out, num_embedding_list)  # 按样本拆分
    return proj_out  # 返回投影结果


class KimiK25ForConditionalGeneration(nn.Module):
    """Kimi K2.5条件生成模型，结合视觉编码器和DeepseekV3语言模型。"""
    # Support nvidia/Kimi-K2.5-NVFP4 naming: language_model.layers.*.  # 支持NVIDIA命名
    # Ref: HF config.json for nvidia/Kimi-K2.5-NVFP4  # 参考
    # https://huggingface.co/nvidia/Kimi-K2.5-NVFP4/blob/main/config.json  # 配置链接
    hf_to_sglang_mapper = WeightsMapper(  # HuggingFace到SGLang权重名映射
        orig_to_new_prefix={  # 前缀映射
            "language_model.layers.": "language_model.model.layers.",  # 层命名映射
        }
    )

    def __init__(
        self,
        config: KimiK25Config,  # Kimi K25配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        **kwargs,  # fix init_tts argument error  # 修复init_tts参数错误
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder  # 是否启用数据并行编码器
        # Create vision tower  # 创建视觉塔
        self.vision_tower = MoonViT3dPretrainedModel(  # MoonViT3D视觉编码器
            config.vision_config,  # 视觉配置
            use_data_parallel=self.use_data_parallel,  # 数据并行
            quant_config=(  # 量化配置
                quant_config if isinstance(quant_config, ModelSlimConfig) else None  # 仅ModelSlim量化
            ),
            prefix="vision_tower",  # 参数前缀
        )
        # Create mm projector  # 创建多模态投影器
        self.mm_projector = K2VLMultiModalProjector(config.vision_config)  # K2VL投影器

        self.language_model = None  # 语言模型初始为None
        if not config.encoder_only:  # 如果不是仅编码器模式
            self.language_model = DeepseekV3ForCausalLM(  # 创建DeepseekV3语言模型
                config.text_config,  # 文本配置
                quant_config,  # 量化配置
                prefix=(  # 参数前缀
                    "language_model"  # 如果是ModelSlim或Quark量化
                    if isinstance(quant_config, (ModelSlimConfig, QuarkConfig))
                    else ""  # 否则无前缀
                ),
            )

        # Ensure that the dtype of the vision_tower and mm_projector matches that of the language_model.  # 确保视觉塔和投影器的数据类型与语言模型一致
        # This solves the dtype mismatch issue when using device_map="auto" and torch_dtype.  # 解决数据类型不匹配问题
        if self.language_model is not None and hasattr(self.language_model, "dtype"):  # 如果语言模型存在
            target_dtype = self.language_model.dtype  # 获取目标数据类型
            self.vision_tower = self.vision_tower.to(dtype=target_dtype)  # 转换视觉塔数据类型
            self.mm_projector = self.mm_projector.to(dtype=target_dtype)  # 转换投影器数据类型

    @property
    def model(self):
        """模型属性，返回语言模型（兼容CUDA图检查）。"""
        # Alias .model to .language_model so this class satisfies the piecewise  # 将.model别名为.language_model
        # CUDA graph gate, which checks `hasattr(model, "model")`.  # 以满足CUDA图门控检查
        return self.language_model  # 返回语言模型

    def __setattr__(self, name, value):
        """重写属性设置，跳过冗余的self.model赋值。"""
        # Skip redundant self.model.model assignment in runner to avoid duplicate  # 跳过冗余的model赋值
        # nn.Module registration.  # 避免重复注册
        if name == "model":  # 如果设置model属性
            return  # 跳过
        super().__setattr__(name, value)  # 其他属性正常设置

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """提取图像特征：通过视觉编码器和投影器。"""
        device = self.vision_tower.device  # 获取设备
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype  # 获取目标数据类型
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(  # 拼接像素值
            device=device, dtype=target_dtype  # 转换设备和数据类型
        )
        image_grid_thws = []  # 图像网格THW列表
        for item in items:  # 遍历每个数据项
            grid_thw = item.model_specific_data.get("image_grid_thw")  # 获取图像网格THW
            if grid_thw is None:  # 如果不存在
                grid_thw = item.model_specific_data["grid_thws"]  # 使用备选键
            image_grid_thws.append(grid_thw)  # 添加到列表
        grid_thws = torch.concat(image_grid_thws, dim=0).to(device)  # 拼接并转到设备

        if self.use_data_parallel:  # 如果使用数据并行
            image_embeds = run_dp_sharded_mrope_vision_model(  # 运行数据并行视觉模型
                self.vision_tower,  # 视觉塔
                pixel_values,  # 像素值
                grid_thws.tolist(),  # 网格THW列表
                rope_type="rope_2d",  # RoPE类型
            )
            image_features = self.mm_projector(image_embeds)  # 通过投影器
            return image_features  # 返回特征

        image_embeds = self.vision_tower(pixel_values, grid_thws)  # 通过视觉塔
        proj_out = mm_projection_auto(self.mm_projector, image_embeds)  # 通过投影器
        return torch.cat(proj_out, dim=0)  # 拼接并返回

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """填充输入token ID，替换多模态标记占位符。"""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 执行填充

    @property
    def start_layer(self) -> int:
        """获取起始层ID。"""
        return self.language_model.start_layer if self.language_model is not None else 0  # 返回起始层

    @property
    def end_layer(self) -> int:
        """获取结束层ID。"""
        if self.language_model is not None:  # 如果语言模型存在
            return self.language_model.end_layer  # 返回语言模型的结束层
        text_config = getattr(self.config, "text_config", None)  # 获取文本配置
        return int(getattr(text_config, "num_hidden_layers", 0))  # 返回层数

    @property
    def routed_experts_weights_of_layer(self):
        """获取每层的路由专家权重。"""
        return (  # 返回路由专家权重
            self.language_model._routed_experts_weights_of_layer.value  # 语言模型的路由专家权重
            if self.language_model is not None  # 如果语言模型存在
            else {}  # 否则返回空字典
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        get_embedding: bool = False,  # 是否获取嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ):
        """前向传播：执行多模态条件生成。"""
        hidden_states = general_mm_embed_routine(  # 调用通用多模态嵌入例程
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.language_model,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数
                Modality.IMAGE: self.get_image_feature,  # 图像特征提取
            },
            positions=positions,  # 位置编码
            pp_proxy_tensors=pp_proxy_tensors,  # 流水线代理
        )

        return hidden_states  # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，流式处理视觉权重和语言模型权重。"""
        """Stream weights, loading vision weights inline and yielding language weights.  # 流式加载权重

        The streaming pattern (vs accumulating into lists) is required because RunAI's  # 流式模式是必须的
        iterator reuses backing buffers — collecting tensors before consuming them  # 因为RunAI迭代器复用缓冲区
        would clobber prior tensors.  # 收集后再消费会覆盖之前的张量
        """
        mapper = getattr(self, "hf_to_sglang_mapper", None)  # 获取权重映射器
        if mapper is not None:  # 如果映射器存在
            weights = mapper.apply(weights)  # 应用映射

        vision_params = (  # 视觉参数字典
            None  # 如果仅加载语言模型则为None
            if self.config.language_only  # 仅语言模型模式
            else dict(self.named_parameters(remove_duplicate=False))  # 否则获取所有参数
        )

        def stream_language_weights():  # 流式语言权重生成器
            for name, loaded_weight in weights:  # 遍历权重
                if "vision_tower" in name or "mm_projector" in name:  # 如果是视觉权重
                    if vision_params is None:  # 如果不需要视觉参数
                        continue  # 跳过
                    vname = (  # 映射权重名称
                        name.replace(r"wqkv.", r"attn.qkv_proj.")  # 注意力QKV映射
                        .replace(r"wo.", r"attn.proj.")  # 注意力输出映射
                        .replace("mm_projector.proj.0", "mm_projector.linear_1")  # 投影器第一层映射
                        .replace("mm_projector.proj.2", "mm_projector.linear_2")  # 投影器第二层映射
                    )
                    if vname not in vision_params:  # 如果参数不存在
                        raise ValueError(f"Weight {vname} not found in params_dict")  # 报错
                    param = vision_params[vname]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认加载器
                    )
                    weight_loader(param, loaded_weight)  # 加载权重
                    continue  # 继续下一个权重
                yield name.replace("language_model.", ""), loaded_weight  # 生成语言模型权重

        if self.language_model is not None:  # 如果语言模型存在
            self.language_model.load_weights(stream_language_weights())  # 加载语言模型权重
        else:  # 仅编码器模式
            # encoder-only: drain the generator so inline vision-weight loading fires.  # 排空生成器以触发内联视觉权重加载
            for _ in stream_language_weights():  # 遍历但不使用
                pass  # 跳过

    def post_load_weights(self):
        """权重加载后的后处理。"""
        if self.language_model is not None:  # 如果语言模型存在
            self.language_model.post_load_weights()  # 调用语言模型的后处理

    @property
    def stacked_params_mapping(self):
        """获取堆叠参数映射。"""
        return getattr(self.language_model, "stacked_params_mapping", [])  # 返回语言模型的堆叠参数映射

    @property
    def expert_params_mapping(self):
        """获取专家参数映射。"""
        return getattr(self.language_model, "expert_params_mapping", [])  # 返回语言模型的专家参数映射

    def mutate_weight_preload(self, name):
        """权重预加载时的变换。"""
        return self.language_model.mutate_weight_preload(name)  # 委托给语言模型

    def custom_scale_remap(self, name):
        """自定义缩放重映射。"""
        return self.language_model.custom_scale_remap(name)  # 委托给语言模型

    @classmethod
    def get_model_config_for_expert_location(cls, config: KimiK25Config):
        """获取专家位置模型配置。"""
        text_config = config.text_config  # 获取文本配置
        return ModelConfigForExpertLocation(  # 返回专家位置配置
            num_layers=text_config.num_hidden_layers,  # 层数
            num_logical_experts=text_config.n_routed_experts,  # 逻辑专家数
            num_groups=text_config.n_group,  # 组数
        )

    def set_eagle3_layers_to_capture(
        self, layer_ids: Optional[List[int]] = None
    ) -> None:
        """设置EAGLE3投机解码需要捕获的层。"""
        """Set the layers to capture for EAGLE3 speculative decoding."""  # 设置EAGLE3投机解码捕获层
        if self.language_model is None or not hasattr(  # 如果语言模型不支持
            self.language_model, "set_eagle3_layers_to_capture"
        ):
            raise AttributeError(  # 抛出属性错误
                "language_model does not support EAGLE3 speculative decoding."
            )

        self.language_model.set_eagle3_layers_to_capture(layer_ids)  # 委托给语言模型

    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
        """设置DFLASH草稿模型训练需要捕获的层。"""
        """Set the layers to capture for DFLASH draft model training."""  # 设置DFLASH层捕获
        if not hasattr(self.language_model, "set_dflash_layers_to_capture"):  # 如果不支持
            raise AttributeError(  # 抛出属性错误
                "language_model does not support DFLASH layer capture."
            )

        self.language_model.set_dflash_layers_to_capture(layer_ids)  # 委托给语言模型

    def get_input_embeddings(self):
        """获取输入嵌入层。"""
        if not hasattr(self.language_model, "get_input_embeddings"):  # 如果不支持
            raise AttributeError(  # 抛出属性错误
                "language_model does not support get_input_embeddings()."
            )

        return self.language_model.get_input_embeddings()  # 返回语言模型的输入嵌入

    @property
    def lm_head(self):
        """获取语言模型头。"""
        if not hasattr(self.language_model, "lm_head"):  # 如果不支持
            raise AttributeError("language_model does not expose lm_head.")  # 抛出错误

        return self.language_model.lm_head  # 返回语言模型头

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取嵌入权重和语言模型头权重（用于投机解码）。"""
        """Get embedding and LM head weights for speculative decoding."""  # 获取嵌入和LM头权重
        if self.language_model is None or not hasattr(  # 如果不支持
            self.language_model, "get_embed_and_head"
        ):
            raise AttributeError(  # 抛出属性错误
                "language_model does not support get_embed_and_head()."
            )

        return self.language_model.get_embed_and_head()  # 返回嵌入和头权重

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        """设置嵌入权重和语言模型头权重（用于投机解码）。"""
        """Set embedding and LM head weights for speculative decoding."""  # 设置嵌入和LM头权重
        if self.language_model is None or not hasattr(  # 如果不支持
            self.language_model, "set_embed_and_head"
        ):
            raise AttributeError(  # 抛出属性错误
                "language_model does not support set_embed_and_head()."
            )

        self.language_model.set_embed_and_head(embed, head)  # 委托给语言模型


EntryClass = [KimiK25ForConditionalGeneration]  # 入口类列表
