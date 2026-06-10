# 文件说明：DeepSeek Janus Pro多模态因果语言模型实现
# 本文件实现了Janus Pro多模态模型，包含视觉编码器（SigLIP ViT）、VQ量化模型、
# 视觉-语言对齐投影器、视觉Transformer块等组件，支持图像理解和生成任务。

# Copyright 2023-2024 SGLang Team
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

# Copied and Adapted from:
# 从以下仓库复制和适配：
# https://github.com/deepseek-ai/Janus


import collections  # 导入集合工具模块
import math  # 导入数学工具模块
import os  # 导入操作系统模块
from dataclasses import field  # 导入数据类字段工具
from enum import Enum  # 导入枚举类型
from functools import partial  # 导入偏函数工具
from itertools import repeat  # 导入重复迭代器
from typing import (  # 导入类型注解
    Callable,  # 可调用类型
    Final,  # 最终类型
    Iterable,  # 可迭代类型
    Literal,  # 字面量类型
    Optional,  # 可选类型
    Sequence,  # 序列类型
    Set,  # 集合类型
    Tuple,  # 元组类型
    Type,  # 类型类型
    Union,  # 联合类型
)

import torch  # 导入PyTorch深度学习框架
import torch.nn.functional as F  # 导入PyTorch神经网络功能模块
from einops import rearrange  # 导入张量重排工具
from torch import Tensor, _assert, nn  # 导入张量、断言和神经网络模块
from torch.nn.init import trunc_normal_  # 导入截断正态初始化
from transformers import AutoModel, PreTrainedModel  # 导入Transformers预训练模型

from sglang.srt.configs.janus_pro import *  # 导入Janus Pro配置
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入逻辑处理器
from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternTokenPairs,  # 多模态数据填充模式（令牌对）
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs  # 导入多模态数据项和输入
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.llama import LlamaForCausalLM  # 导入Llama因果语言模型
from sglang.utils import logger  # 导入日志记录器

#################################################################################
#                              VQ Model Configs                                 #
#                            VQ模型配置                                          #
#################################################################################


# Copied from:
# 从以下仓库复制：
# https://github.com/deepseek-ai/Janus/tree/main/janus/models/vq_model.py
@dataclass
class ModelArgs:  # VQ模型参数数据类
    codebook_size: int = 16384  # 码本大小
    codebook_embed_dim: int = 8  # 码本嵌入维度
    codebook_l2_norm: bool = True  # 码本是否使用L2归一化
    codebook_show_usage: bool = True  # 是否显示码本使用情况
    commit_loss_beta: float = 0.25  # 提交损失的beta系数
    entropy_loss_ratio: float = 0.0  # 熵损失比率

    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])  # 编码器通道倍数
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])  # 解码器通道倍数
    z_channels: int = 256  # 潜在空间通道数
    dropout_p: float = 0.0  # Dropout概率


# 递归地对模块及其子模块应用函数
def named_apply(
    fn: Callable,  # 要应用的函数
    module: nn.Module,  # 目标模块
    name="",  # 模块名称
    depth_first: bool = True,  # 是否深度优先遍历
    include_root: bool = False,  # 是否包含根模块
) -> nn.Module:
    if not depth_first and include_root:  # 如果非深度优先且包含根模块
        fn(module=module, name=name)  # 先对根模块应用函数
    for child_name, child_module in module.named_children():  # 遍历子模块
        child_name = ".".join((name, child_name)) if name else child_name  # 构建子模块全名
        named_apply(  # 递归对子模块应用函数
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:  # 如果深度优先且包含根模块
        fn(module=module, name=name)  # 后对根模块应用函数
    return module  # 返回模块


# 创建VQ-16模型工厂函数
def VQ_16(**kwargs):
    return VQModel(  # 返回VQ模型实例
        ModelArgs(
            encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs  # 编码器和解码器通道倍数
        )
    )


VQ_models = {"VQ-16": VQ_16}  # VQ模型注册表

import collections.abc  # 导入集合抽象基类


# From PyTorch internals
# 来自PyTorch内部实现
# 创建一个将输入转换为n元组的函数
def _ntuple(n):
    def parse(x):  # 解析函数
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):  # 如果是可迭代对象但非字符串
            return tuple(x)  # 转为元组
        return tuple(repeat(x, n))  # 否则重复n次组成元组

    return parse  # 返回解析函数


# 截断正态分布初始化的内部实现
def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # 从PyTorch官方master分支复制粘贴，直到它在几个官方版本中发布 - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # 方法基于 https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):  # 标准正态累积分布函数
        # Computes standard normal cumulative distribution function
        # 计算标准正态累积分布函数
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):  # 如果均值距离边界超过2个标准差
        logger.warn(  # 发出警告
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # 值通过截断均匀分布生成，然后
    # then using the inverse CDF for the normal distribution.
    # 使用正态分布的逆CDF转换。
    # Get upper and lower cdf values
    # 获取上下界CDF值
    l = norm_cdf((a - mean) / std)  # 下界CDF值
    u = norm_cdf((b - mean) / std)  # 上界CDF值

    # Uniformly fill tensor with values from [l, u], then translate to
    # 用[l, u]区间的均匀值填充张量，然后转换到
    # [2l-1, 2u-1].
    # [2l-1, 2u-1]区间。
    tensor.uniform_(2 * l - 1, 2 * u - 1)  # 均匀填充

    # Use inverse cdf transform for normal distribution to get truncated
    # 使用逆CDF变换获取截断
    # standard normal
    # 标准正态分布
    if tensor.dtype in [torch.float16, torch.bfloat16]:  # 如果是半精度
        # The `erfinv_` op is not (yet?) defined in float16+cpu, bfloat16+gpu
        # `erfinv_`操作（尚未？）在float16+cpu和bfloat16+gpu上未定义
        og_dtype = tensor.dtype  # 保存原始数据类型
        tensor = tensor.to(torch.float32)  # 转为float32
        tensor.erfinv_()  # 计算逆误差函数
        tensor = tensor.to(og_dtype)  # 转回原始类型
    else:  # 全精度
        tensor.erfinv_()  # 计算逆误差函数

    # Transform to proper mean, std
    # 转换为指定的均值和标准差
    tensor.mul_(std * math.sqrt(2.0))  # 乘以标准差和sqrt(2)
    tensor.add_(mean)  # 加上均值

    # Clamp to ensure it's in the proper range
    # 裁剪以确保在正确范围内
    if tensor.dtype == torch.float16:  # 如果是float16
        # The `clamp_` op is not (yet?) defined in float16+cpu
        # `clamp_`操作（尚未？）在float16+cpu上未定义
        tensor = tensor.to(torch.float32)  # 转为float32
        tensor.clamp_(min=a, max=b)  # 裁剪到[a, b]范围
    else:  # 其他精度
        tensor.clamp_(min=a, max=b)  # 裁剪到[a, b]范围


# 截断正态分布初始化（TensorFlow/JAX风格）
def trunc_normal_tf_(
    tensor: torch.Tensor,  # 待初始化张量
    mean: float = 0.0,  # 均值
    std: float = 1.0,  # 标准差
    a: float = -2.0,  # 下界
    b: float = 2.0,  # 上界
):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsequently scaled and shifted by the mean and std args.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    """用截断正态分布的值填充输入张量。值实际上是从
    正态分布 :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)` 中抽取的，
    超出 :math:`[a, b]` 范围的值会被重抽直到在范围内。
    生成随机值的方法在 :math:`a \\leq \\text{mean} \\leq b` 时效果最好。
    注意：此'tf'变体的行为更接近TensorFlow/JAX实现，其中
    边界[a, b]在采样mean=0, std=1.0的正态分布时应用，
    结果随后由mean和std参数进行缩放和偏移。
    参数：
        tensor: n维`torch.Tensor`
        mean: 正态分布的均值
        std: 正态分布的标准差
        a: 最小截断值
        b: 最大截断值
    """
    with torch.no_grad():  # 禁用梯度计算
        _trunc_normal_(tensor, 0, 1.0, a, b)  # 先用标准截断正态初始化
        tensor.mul_(std).add_(mean)  # 然后缩放和偏移到目标均值和标准差


to_2tuple = _ntuple(2)  # 将输入转换为2元组的函数


# 张量格式枚举
class Format(str, Enum):
    NCHW = "NCHW"  # 批次-通道-高度-宽度格式
    NHWC = "NHWC"  # 批次-高度-宽度-通道格式
    NCL = "NCL"  # 批次-通道-长度格式
    NLC = "NLC"  # 批次-长度-通道格式


# 将NCHW格式张量转换为目标格式
def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:  # 转换为NHWC
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:  # 转换为NLC
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:  # 转换为NCL
        x = x.flatten(2)
    return x  # 返回转换后的张量


# 重采样补丁嵌入权重到目标分辨率
def resample_patch_embed(
    patch_embed,  # 原始补丁嵌入参数
    new_size: List[int],  # 目标尺寸
    interpolation: str = "bicubic",  # 插值方法
    antialias: bool = True,  # 是否使用抗锯齿
    verbose: bool = False,  # 是否输出详细信息
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """
    """将补丁嵌入核的权重重采样到目标分辨率。
    我们通过近似逆转补丁缩放的效果来重采样补丁嵌入核。

    代码基于：
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    通过此缩放，我们可以例如将B/8滤波器加载到B/16模型中，
    在2倍大的输入图像上，结果将匹配。

    参数：
        patch_embed: 待调整大小的原始参数。
        new_size (tuple(int, int)): 目标形状（仅高度和宽度）。
        interpolation (str): 缩放的插值方法
        antialias (bool): 缩放时是否使用抗锯齿滤波器
        verbose (bool): 是否记录操作
    返回：
        调整大小后的补丁嵌入核。
    """
    import numpy as np  # 导入NumPy

    try:
        from torch import vmap  # 导入向量化映射
    except ImportError:
        from torch.func import vmap  # 从torch.func导入vmap

    assert len(patch_embed.shape) == 4, "Four dimensions expected"  # 断言四维
    assert len(new_size) == 2, "New shape should only be hw"  # 断言新形状仅为高宽
    old_size = patch_embed.shape[-2:]  # 获取原始尺寸
    if tuple(old_size) == tuple(new_size):  # 如果尺寸相同则无需重采样
        return patch_embed

    if verbose:  # 如果需要详细信息
        logger.info(
            f"Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation."
        )

    def resize(x_np, _new_size):  # 调整大小辅助函数
        x_tf = torch.Tensor(x_np)[None, None, ...]  # 添加批次和通道维度
        x_upsampled = F.interpolate(  # 执行插值上/下采样
            x_tf, size=_new_size, mode=interpolation, antialias=antialias
        )[0, 0, ...].numpy()  # 移除额外维度并转为NumPy
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):  # 构建重采样矩阵
        mat = []
        for i in range(np.prod(_old_size)):  # 遍历所有基向量
            basis_vec = np.zeros(_old_size)  # 创建基向量
            basis_vec[np.unravel_index(i, _old_size)] = 1.0  # 设置基向量值
            mat.append(resize(basis_vec, _new_size).reshape(-1))  # 重采样基向量并展平
        return np.stack(mat).T  # 堆叠并转置

    resize_mat = get_resize_mat(old_size, new_size)  # 获取重采样矩阵
    resize_mat_pinv = torch.tensor(  # 计算重采样矩阵的伪逆
        np.linalg.pinv(resize_mat.T), device=patch_embed.device
    )

    def resample_kernel(kernel):  # 重采样单个核
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)  # 使用伪逆重采样
        return resampled_kernel.reshape(new_size)  # 重塑为目标尺寸

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)  # 向量化重采样核
    orig_dtype = patch_embed.dtype  # 保存原始数据类型
    patch_embed = patch_embed.float()  # 转为float32进行计算
    patch_embed = v_resample_kernel(patch_embed)  # 执行向量化重采样
    patch_embed = patch_embed.to(orig_dtype)  # 转回原始数据类型
    return patch_embed  # 返回重采样后的补丁嵌入


# Copied from:
# 从以下仓库复制：
# https://github.com/deepseek-ai/Janus/tree/main/janus/models/siglip_vit.py
# 2D图像到补丁嵌入层
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""  # 2D图像到补丁嵌入

    output_fmt: Format  # 输出格式
    dynamic_img_pad: torch.jit.Final[bool]  # 是否动态填充图像

    def __init__(
        self,
        img_size: Optional[int] = 224,  # 输入图像大小
        patch_size: int = 16,  # 补丁大小
        in_chans: int = 3,  # 输入通道数
        embed_dim: int = 768,  # 嵌入维度
        norm_layer: Optional[Callable] = None,  # 归一化层
        flatten: bool = True,  # 是否展平
        output_fmt: Optional[str] = None,  # 输出格式
        bias: bool = True,  # 是否使用偏置
        strict_img_size: bool = True,  # 是否严格检查图像大小
        dynamic_img_pad: bool = False,  # 是否动态填充
    ):
        super().__init__()  # 调用父类初始化
        self.patch_size = tuple(to_2tuple(patch_size))  # 补丁大小转为2元组
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)  # 初始化图像尺寸

        if output_fmt is not None:  # 如果指定了输出格式
            self.flatten = False  # 不展平
            self.output_fmt = Format(output_fmt)  # 设置输出格式
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            # 展平空间维度并转置为通道最后格式，保持向后兼容
            self.flatten = flatten
            self.output_fmt = Format.NCHW  # 默认NCHW格式
        self.strict_img_size = strict_img_size  # 是否严格检查图像大小
        self.dynamic_img_pad = dynamic_img_pad  # 是否动态填充

        self.proj = nn.Conv2d(  # 补丁投影卷积层
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()  # 归一化层或恒等映射

    # 初始化图像尺寸，计算网格大小和补丁数
    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size  # 断言补丁大小已设置
        if img_size is None:  # 如果未指定图像大小
            return None, None, None
        img_size = to_2tuple(img_size)  # 转为2元组
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])  # 计算网格大小
        num_patches = grid_size[0] * grid_size[1]  # 计算补丁数
        return img_size, grid_size, num_patches  # 返回图像大小、网格大小和补丁数

    # 设置新的输入尺寸，支持动态调整补丁大小
    def set_input_size(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,  # 新图像大小
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,  # 新补丁大小
    ):
        new_patch_size = None
        if patch_size is not None:  # 如果指定了新补丁大小
            new_patch_size = to_2tuple(patch_size)  # 转为2元组
        if new_patch_size is not None and new_patch_size != self.patch_size:  # 如果补丁大小变化
            with torch.no_grad():  # 禁用梯度
                new_proj = nn.Conv2d(  # 创建新卷积层
                    self.proj.in_channels,  # 输入通道数
                    self.proj.out_channels,  # 输出通道数
                    kernel_size=new_patch_size,  # 新卷积核大小
                    stride=new_patch_size,  # 新步长
                    bias=self.proj.bias is not None,  # 是否有偏置
                )
                new_proj.weight.copy_(  # 复制重采样后的权重
                    resample_patch_embed(self.proj.weight, new_patch_size, verbose=True)
                )
                if self.proj.bias is not None:  # 如果有偏置
                    new_proj.bias.copy_(self.proj.bias)  # 复制偏置
                self.proj = new_proj  # 替换投影层
            self.patch_size = new_patch_size  # 更新补丁大小
        img_size = img_size or self.img_size  # 使用新图像大小或保持原样
        if img_size != self.img_size or new_patch_size is not None:  # 如果尺寸变化
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(  # 重新计算尺寸信息
                img_size
            )

    # 获取特征比率
    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:  # 如果返回标量
            return max(self.patch_size)
        else:  # 返回元组
            return self.patch_size

    # 获取动态特征大小（考虑动态填充）
    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        """获取给定图像大小的网格（特征）大小，考虑动态填充。
        注意：必须兼容torchscript，因此使用固定元组索引
        """
        if self.dynamic_img_pad:  # 如果使用动态填充
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(  # 向上取整
                img_size[1] / self.patch_size[1]
            )
        else:  # 不使用动态填充
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]  # 向下取整

    # 前向传播：将2D图像转换为补丁嵌入
    def forward(self, x):
        B, C, H, W = x.shape  # 获取批次、通道、高度、宽度
        if self.img_size is not None:  # 如果指定了图像大小
            if self.strict_img_size:  # 严格检查图像大小
                _assert(
                    H == self.img_size[0],  # 检查高度
                    f"Input height ({H}) doesn't match model ({self.img_size[0]}).",
                )
                _assert(
                    W == self.img_size[1],  # 检查宽度
                    f"Input width ({W}) doesn't match model ({self.img_size[1]}).",
                )
            elif not self.dynamic_img_pad:  # 非动态填充时检查可整除性
                _assert(
                    H % self.patch_size[0] == 0,  # 检查高度可被补丁大小整除
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]}).",
                )
                _assert(
                    W % self.patch_size[1] == 0,  # 检查宽度可被补丁大小整除
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]}).",
                )
        if self.dynamic_img_pad:  # 如果使用动态填充
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]  # 计算高度填充量
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]  # 计算宽度填充量
            x = F.pad(x, (0, pad_w, 0, pad_h))  # 执行填充
        x = self.proj(x)  # 执行卷积投影
        if self.flatten:  # 如果需要展平
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:  # 如果需要转换格式
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)  # 归一化
        return x  # 返回补丁嵌入


# MLP（多层感知机）模块
class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks
    MLPMixer和相关网络中使用的MLP

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    注意：当use_conv=True时，期望2D NCHW张量，否则期望N*C。
    """

    def __init__(
        self,
        in_features,  # 输入特征维度
        hidden_features=None,  # 隐藏层特征维度
        out_features=None,  # 输出特征维度
        act_layer=nn.GELU,  # 激活层
        norm_layer=None,  # 归一化层
        bias=True,  # 是否使用偏置
        drop=0.0,  # Dropout概率
        use_conv=False,  # 是否使用卷积替代线性层
    ):
        super().__init__()  # 调用父类初始化
        out_features = out_features or in_features  # 默认输出维度等于输入维度
        hidden_features = hidden_features or in_features  # 默认隐藏维度等于输入维度
        bias = to_2tuple(bias)  # 偏置转为2元组
        drop_probs = to_2tuple(drop)  # Dropout概率转为2元组
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear  # 选择线性层类型

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])  # 第一个全连接层
        self.act = act_layer()  # 激活层
        self.drop1 = nn.Dropout(drop_probs[0])  # 第一个Dropout层
        self.norm = (  # 归一化层
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])  # 第二个全连接层
        self.drop2 = nn.Dropout(drop_probs[1])  # 第二个Dropout层

    # MLP前向传播
    def forward(self, x):
        x = self.fc1(x)  # 第一个全连接
        x = self.act(x)  # 激活
        x = self.drop1(x)  # Dropout
        x = self.norm(x)  # 归一化
        x = self.fc2(x)  # 第二个全连接
        x = self.drop2(x)  # Dropout
        return x  # 返回输出


# 随机深度（Drop Path）函数
def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    每个样本的随机深度（应用于残差块主路径时）。

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    这与我为EfficientNet等网络创建的DropConnect实现相同，但原始名称有误导性，
    因为'Drop Connect'是另一篇论文中不同形式的dropout...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    参见讨论：...我选择将层和参数名改为'drop path'，而不是混合使用DropConnect作为层名和'survival rate'作为参数。

    """
    if drop_prob == 0.0 or not training:  # 如果丢弃概率为0或不在训练模式
        return x  # 直接返回输入
    keep_prob = 1 - drop_prob  # 保留概率
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets  # 适用于不同维度的张量，不只是2D卷积网络
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)  # 生成伯努利随机张量
    if keep_prob > 0.0 and scale_by_keep:  # 如果需要按保留概率缩放
        random_tensor.div_(keep_prob)  # 除以保留概率
    return x * random_tensor  # 返回缩放后的输入


# 随机深度（Drop Path）模块
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    """每个样本的随机深度（应用于残差块主路径时）。"""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()  # 调用父类初始化
        self.drop_prob = drop_prob  # 丢弃概率
        self.scale_by_keep = scale_by_keep  # 是否按保留概率缩放

    # 前向传播：应用随机深度
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    # 额外表示信息
    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


# 视觉Transformer块
class VisionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,  # 嵌入维度
        num_heads: int,  # 注意力头数
        mlp_ratio: float = 4.0,  # MLP隐藏层维度比率
        qkv_bias: bool = False,  # QKV是否使用偏置
        qk_norm: bool = False,  # 是否对QK进行归一化
        proj_drop: float = 0.0,  # 投影Dropout率
        attn_drop: float = 0.0,  # 注意力Dropout率
        init_values: Optional[float] = None,  # 层缩放初始值
        drop_path: float = 0.0,  # 随机深度率
        act_layer: nn.Module = nn.GELU,  # 激活层
        norm_layer: nn.Module = nn.LayerNorm,  # 归一化层
        mlp_layer: nn.Module = Mlp,  # MLP层
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.norm1 = norm_layer(dim)  # 第一个归一化层
        self.attn = VisionAttention(  # 视觉注意力层
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,  # 使用QKV并行
            dropout=attn_drop,
        )

        self.ls1 = (  # 第一个层缩放
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()  # 第一个随机深度

        self.norm2 = norm_layer(dim)  # 第二个归一化层
        self.mlp = mlp_layer(  # MLP层
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (  # 第二个层缩放
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()  # 第二个随机深度

    # 前向传播：Transformer块
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))  # 注意力残差连接
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))  # MLP残差连接
        return x  # 返回输出


LayerType = Union[str, Callable, Type[torch.nn.Module]]  # 层类型联合类型


# 补丁Dropout模块
class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """

    return_indices: torch.jit.Final[bool]  # 是否返回索引

    def __init__(
        self,
        prob: float = 0.5,  # 丢弃概率
        num_prefix_tokens: int = 1,  # 前缀令牌数
        ordered: bool = False,  # 是否保持有序
        return_indices: bool = False,  # 是否返回保留索引
    ):
        super().__init__()  # 调用父类初始化
        assert 0 <= prob < 1.0  # 断言概率在[0, 1)范围内
        self.prob = prob  # 丢弃概率
        self.num_prefix_tokens = (
            num_prefix_tokens  # exclude CLS token (or other prefix tokens)  # 排除CLS令牌（或其他前缀令牌）
        )
        self.ordered = ordered  # 是否有序
        self.return_indices = return_indices  # 是否返回索引

    # 前向传播：随机丢弃补丁
    def forward(
        self, x
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if not self.training or self.prob == 0.0:  # 如果不在训练或概率为0
            if self.return_indices:  # 如果需要返回索引
                return x, None
            return x  # 直接返回输入

        if self.num_prefix_tokens:  # 如果有前缀令牌
            prefix_tokens, x = (  # 分离前缀令牌和补丁
                x[:, : self.num_prefix_tokens],
                x[:, self.num_prefix_tokens :],
            )
        else:  # 无前缀令牌
            prefix_tokens = None

        B = x.shape[0]  # 批次大小
        L = x.shape[1]  # 序列长度
        num_keep = max(1, int(L * (1.0 - self.prob)))  # 计算保留数量
        keep_indices = torch.argsort(torch.randn(B, L, device=x.device), dim=-1)[  # 随机排序获取索引
            :, :num_keep
        ]
        if self.ordered:  # 如果需要有序
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            # 注意：在典型的Transformer使用中不需要维护补丁顺序，
            # 但可能对调试/可视化有用
            keep_indices = keep_indices.sort(dim=-1)[0]  # 对索引排序
        x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))  # 收集保留的补丁

        if prefix_tokens is not None:  # 如果有前缀令牌
            x = torch.cat((prefix_tokens, x), dim=1)  # 拼接前缀令牌和保留的补丁

        if self.return_indices:  # 如果需要返回索引
            return x, keep_indices  # 返回补丁和索引
        return x  # 返回补丁


# 重采样绝对位置嵌入
def resample_abs_pos_embed(
    posemb: torch.Tensor,  # 位置嵌入
    new_size: List[int],  # 新尺寸
    old_size: Optional[List[int]] = None,  # 旧尺寸
    num_prefix_tokens: int = 1,  # 前缀令牌数
    interpolation: str = "bicubic",  # 插值方法
    antialias: bool = True,  # 抗锯齿
    verbose: bool = False,  # 详细信息
):
    # sort out sizes, assume square if old size not provided
    # 整理尺寸，如果未提供旧尺寸则假设为正方形
    num_pos_tokens = posemb.shape[1]  # 位置令牌数
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens  # 新令牌数
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:  # 如果令牌数相同且为正方形
        return posemb  # 无需重采样

    if old_size is None:  # 如果未提供旧尺寸
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))  # 计算正方形边长
        old_size = hw, hw  # 设置旧尺寸

    if num_prefix_tokens:  # 如果有前缀令牌
        posemb_prefix, posemb = (  # 分离前缀和主体
            posemb[:, :num_prefix_tokens],
            posemb[:, num_prefix_tokens:],
        )
    else:  # 无前缀令牌
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    # 执行插值
    embed_dim = posemb.shape[-1]  # 嵌入维度
    orig_dtype = posemb.dtype  # 保存原始数据类型
    posemb = posemb.float()  # interpolate needs float32  # 插值需要float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)  # 重塑为图像格式
    posemb = F.interpolate(  # 执行插值
        posemb, size=new_size, mode=interpolation, antialias=antialias
    )
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)  # 恢复形状
    posemb = posemb.to(orig_dtype)  # 恢复原始数据类型

    # add back extra (class, etc) prefix tokens
    # 添加回额外的（类等）前缀令牌
    if posemb_prefix is not None:  # 如果有前缀令牌
        posemb = torch.cat([posemb_prefix, posemb], dim=1)  # 拼接前缀令牌

    if not torch.jit.is_scripting() and verbose:  # 如果不是脚本模式且需要详细信息
        logger.info(f"Resized position embedding: {old_size} to {new_size}.")

    return posemb  # 返回重采样后的位置嵌入


# 初始化权重（用于注意力池化潜在层）
def init_weights(self):
    if self.pos_embed is not None:  # 如果有位置嵌入
        trunc_normal_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)  # 截断正态初始化
    trunc_normal_(self.latent, std=self.latent_dim**-0.5)  # 潜在变量截断正态初始化


# ViT权重初始化（timm风格）
def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    """ViT权重初始化，原始timm实现（用于可复现性）"""
    if isinstance(module, nn.Linear):  # 如果是线性层
        trunc_normal_(module.weight, std=0.02)  # 截断正态初始化权重
        if module.bias is not None:  # 如果有偏置
            nn.init.zeros_(module.bias)  # 零初始化偏置
    elif hasattr(module, "init_weights"):  # 如果模块有自定义初始化方法
        module.init_weights()  # 调用自定义初始化


# 视觉Transformer模型
class VisionTransformer(nn.Module):
    """Vision Transformer
    视觉Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
    PyTorch实现：`一张图片等价于16x16个单词：大规模图像识别的Transformer`
        - https://arxiv.org/abs/2010.11929
    """

    dynamic_img_size: Final[bool]  # 是否动态图像大小

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,  # 输入图像大小
        patch_size: Union[int, Tuple[int, int]] = 16,  # 补丁大小
        in_chans: int = 3,  # 输入通道数
        num_classes: int = 1000,  # 分类数
        global_pool: Literal["", "avg", "token", "map"] = "token",  # 全局池化类型
        embed_dim: int = 768,  # 嵌入维度
        depth: int = 12,  # Transformer深度
        num_heads: int = 12,  # 注意力头数
        mlp_ratio: float = 4.0,  # MLP比率
        qkv_bias: bool = True,  # QKV偏置
        qk_norm: bool = False,  # QK归一化
        init_values: Optional[float] = None,  # 层缩放初始值
        class_token: bool = True,  # 是否使用类令牌
        no_embed_class: bool = False,  # 是否不嵌入类令牌位置
        reg_tokens: int = 0,  # 寄存器令牌数
        pre_norm: bool = False,  # 是否使用预归一化
        fc_norm: Optional[bool] = None,  # 全连接归一化
        dynamic_img_size: bool = False,  # 动态图像大小
        dynamic_img_pad: bool = False,  # 动态图像填充
        drop_rate: float = 0.0,  # Dropout率
        pos_drop_rate: float = 0.0,  # 位置嵌入Dropout率
        patch_drop_rate: float = 0.0,  # 补丁Dropout率
        proj_drop_rate: float = 0.0,  # 投影Dropout率
        attn_drop_rate: float = 0.0,  # 注意力Dropout率
        drop_path_rate: float = 0.0,  # 随机深度率
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",  # 权重初始化方案
        embed_layer: Callable = PatchEmbed,  # 嵌入层
        _norm_layer: Optional[LayerType] = None,  # 归一化层
        _act_layer: Optional[LayerType] = None,  # 激活层
        block_fn: Type[nn.Module] = VisionTransformerBlock,  # Transformer块
        mlp_layer: Type[nn.Module] = Mlp,  # MLP层
        ignore_head: bool = False,  # 是否忽略分类头
    ) -> None:
        """
        Args:
            img_size: Input image size.  # 输入图像大小。
            patch_size: Patch size.  # 补丁大小。
            in_chans: Number of image input channels.  # 图像输入通道数。
            num_classes: Number of classes for classification head.  # 分类头的类别数。
            global_pool: Type of global pooling for final sequence (default: 'token').  # 最终序列的全局池化类型（默认：'token'）。
            embed_dim: Transformer embedding dimension.  # Transformer嵌入维度。
            depth: Depth of transformer.  # Transformer深度。
            num_heads: Number of attention heads.  # 注意力头数。
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.  # MLP隐藏维度与嵌入维度的比率。
            qkv_bias: Enable bias for qkv projections if True.  # 如果为True，启用qkv投影的偏置。
            init_values: Layer-scale init values (layer-scale enabled if not None).  # 层缩放初始值（如果不为None则启用层缩放）。
            class_token: Use class token.  # 使用类令牌。
            no_embed_class: Don't include position embeddings for class (or reg) tokens.  # 不包含类（或寄存器）令牌的位置嵌入。
            reg_tokens: Number of register tokens.  # 寄存器令牌数。
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.  # 池化后归一化（而非之前），如果为None，当global_pool == 'avg'时启用。
            drop_rate: Head dropout rate.  # 头部Dropout率。
            pos_drop_rate: Position embedding dropout rate.  # 位置嵌入Dropout率。
            attn_drop_rate: Attention dropout rate.  # 注意力Dropout率。
            drop_path_rate: Stochastic depth rate.  # 随机深度率。
            weight_init: Weight initialization scheme.  # 权重初始化方案。
            embed_layer: Patch embedding layer.  # 补丁嵌入层。
            _norm_layer: Normalization layer.  # 归一化层。
            _act_layer: MLP activation layer.  # MLP激活层。
            block_fn: Transformer block layer.  # Transformer块层。
        """
        super().__init__()  # 调用父类初始化
        assert global_pool in ("", "avg", "token", "map")  # 断言全局池化类型有效
        assert class_token or global_pool != "token"  # 断言token池化需要类令牌
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm  # 确定是否使用FC归一化
        # norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        # act_layer = get_act_layer(act_layer) or nn.GELU
        norm_layer = partial(nn.LayerNorm, eps=1e-6)  # 归一化层
        act_layer = nn.GELU  # 激活层

        self.num_classes = num_classes  # 分类数
        self.global_pool = global_pool  # 全局池化类型
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models  # 特征数，与其他模型保持一致
        )
        self.num_prefix_tokens = 1 if class_token else 0  # 前缀令牌数
        self.num_prefix_tokens += reg_tokens  # 加上寄存器令牌数
        self.num_reg_tokens = reg_tokens  # 寄存器令牌数
        self.has_class_token = class_token  # 是否有类令牌
        self.no_embed_class = (
            no_embed_class  # don't embed prefix positions (includes reg)  # 不嵌入前缀位置（包括寄存器）
        )
        self.dynamic_img_size = dynamic_img_size  # 动态图像大小
        self.grad_checkpointing = False  # 梯度检查点
        self.ignore_head = ignore_head  # 是否忽略分类头

        embed_args = {}  # 嵌入参数
        if dynamic_img_size:  # 如果动态图像大小
            # flatten deferred until after pos embed
            # 展平延迟到位置嵌入之后
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        self.patch_embed = embed_layer(  # 补丁嵌入层
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)  # 如果使用预归一化则禁用偏置（如CLIP）
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches  # 补丁数

        self.cls_token = (  # 类令牌
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        self.reg_token = (  # 寄存器令牌
            nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        )
        embed_len = (  # 嵌入长度
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)  # 位置嵌入
        self.pos_drop = nn.Dropout(p=pos_drop_rate)  # 位置嵌入Dropout
        if patch_drop_rate > 0:  # 如果补丁Dropout率大于0
            self.patch_drop = PatchDropout(  # 补丁Dropout
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:  # 否则
            self.patch_drop = nn.Identity()  # 恒等映射
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()  # 预归一化

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule  # 随机深度衰减规则
        self.blocks = nn.Sequential(  # Transformer块序列
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],  # 当前层的随机深度率
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)  # 遍历深度
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()  # 最终归一化

        # Classifier Head
        # 分类头
        if global_pool == "map":  # 如果使用注意力池化
            AttentionPoolLatent.init_weights = init_weights  # 设置初始化方法
            self.attn_pool = AttentionPoolLatent(  # 注意力池化层
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:  # 不使用注意力池化
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()  # FC归一化
        self.head_drop = nn.Dropout(drop_rate)  # 头部Dropout
        self.head = (  # 分类头
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if weight_init != "skip":  # 如果不跳过权重初始化
            self.init_weights(weight_init)  # 初始化权重

    # 权重初始化
    def init_weights(self, mode: Literal["jax", "jax_nlhb", "moco", ""] = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")  # 断言初始化模式有效
        # head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)  # 位置嵌入截断正态初始化
        if self.cls_token is not None:  # 如果有类令牌
            nn.init.normal_(self.cls_token, std=1e-6)  # 正态初始化
        named_apply(init_weights_vit_timm, self)  # 递归应用timm风格初始化

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:  # 不进行权重衰减的参数名
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:  # 参数分组匹配器
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed  # 主干和嵌入
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:  # 获取分类器
        return self.head

    # 重置分类器
    def reset_classifier(self, num_classes: int, global_pool=None) -> None:
        self.num_classes = num_classes  # 更新分类数
        if global_pool is not None:  # 如果指定了新的全局池化类型
            assert global_pool in ("", "avg", "token", "map")  # 断言有效
            if global_pool == "map" and self.attn_pool is None:  # 切换到map但无注意力池化
                assert (
                    False
                ), "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map " and self.attn_pool is not None:  # 切换离开map
                self.attn_pool = None  # remove attention pooling  # 移除注意力池化
            self.global_pool = global_pool  # 更新全局池化类型
        self.head = (  # 更新分类头
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    # 位置嵌入处理
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:  # 如果动态图像大小
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(  # 重采样位置嵌入
                self.pos_embed,
                [H, W],
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)  # 展平空间维度
        else:  # 固定图像大小
            pos_embed = self.pos_embed

        to_cat = []  # 待拼接列表
        if self.cls_token is not None:  # 如果有类令牌
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:  # 如果有寄存器令牌
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:  # 如果不嵌入类令牌位置
            # deit-3, updated JAX (big vision)
            # deit-3，更新的JAX（big vision）
            # position embedding does not overlap with class token, add then concat
            # 位置嵌入不与类令牌重叠，先加后拼接
            x = x + pos_embed
            if to_cat:  # 如果有待拼接的令牌
                x = torch.cat(to_cat + [x], dim=1)
        else:  # 嵌入类令牌位置
            # original timm, JAX, and deit vit impl
            # 原始timm、JAX和deit vit实现
            # pos_embed has entry for class token, concat then add
            # 位置嵌入包含类令牌条目，先拼接后加
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)  # 返回经Dropout的位置嵌入

    # 获取中间层输出
    def _intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # 取最后n层或指定层
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)  # 输出列表和块数
        take_indices = set(  # 需要提取的层索引
            range(num_blocks - n, num_blocks) if isinstance(n, int) else n
        )

        # forward pass
        # 前向传播
        x = self.patch_embed(x)  # 补丁嵌入
        x = self._pos_embed(x)  # 位置嵌入
        x = self.patch_drop(x)  # 补丁Dropout
        x = self.norm_pre(x)  # 预归一化
        for i, blk in enumerate(self.blocks):  # 遍历Transformer块
            x = blk(x)
            if i in take_indices:  # 如果当前层需要提取
                outputs.append(x)

        return outputs  # 返回中间层输出

    # 提取特征
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # 补丁嵌入
        x = self._pos_embed(x)  # 位置嵌入
        x = self.patch_drop(x)  # 补丁Dropout
        x = self.norm_pre(x)  # 预归一化
        x = self.blocks(x)  # Transformer块
        x = self.norm(x)  # 最终归一化
        return x  # 返回特征

    # 前向传播分类头
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:  # 注意力池化
            x = self.attn_pool(x)
        elif self.global_pool == "avg":  # 平均池化
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        elif self.global_pool:  # 类令牌池化
            x = x[:, 0]  # class token
        x = self.fc_norm(x)  # FC归一化
        x = self.head_drop(x)  # 头部Dropout
        return x if pre_logits else self.head(x)  # 返回logits或pre-logits

    # 前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)  # 提取特征
        if not self.ignore_head:  # 如果不忽略分类头
            x = self.forward_head(x)  # 分类头
        return x  # 返回输出


# 根据类名映射到具体类
def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:  # MLP投影器
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:  # CLIP视觉塔
        cls = CLIPVisionTower

    elif "VQ" in cls_name:  # VQ模型

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:  # 视觉头
        cls = vision_head
    else:  # 无效类名
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls  # 返回类


# 视觉头模块，将视觉特征投影到图像令牌空间
class vision_head(torch.nn.Module):
    def __init__(self, params):  # 参数字典
        super().__init__()  # 调用父类初始化
        self.output_mlp_projector = torch.nn.Linear(  # 输出MLP投影器
            params["n_embed"], params["image_token_embed"]
        )
        self.vision_activation = torch.nn.GELU()  # 视觉激活函数
        self.vision_head = torch.nn.Linear(  # 视觉头线性层
            params["image_token_embed"], params["image_token_size"]
        )

    # 前向传播
    def forward(self, x):
        x = self.output_mlp_projector(x)  # MLP投影
        x = self.vision_activation(x)  # 激活
        x = self.vision_head(x)  # 视觉头
        return x  # 返回输出


# SigLIP模型配置
SigLIP_MODEL_CONFIG = {
    "siglip_so400m_patch14_384": {  # SigLIP SO400M 14x14补丁 384分辨率
        "image_size": 336,  # 图像大小
        "patch_size": 14,  # 补丁大小
        "width": 1152,  # 宽度（嵌入维度）
        "layers": 27,  # 层数
        "heads": 16,  # 头数
        "mlp_ratio": 3.7362,  # MLP比率
        "global_pool": "map",  # 全局池化类型
        "use_checkpoint": False,  # 是否使用检查点
    },
    "siglip_so400m_patch14_224": {  # SigLIP SO400M 14x14补丁 224分辨率
        "image_size": 224,  # 图像大小
        "patch_size": 14,  # 补丁大小
        "width": 1152,  # 宽度
        "layers": 27,  # 层数
        "heads": 16,  # 头数
        "mlp_ratio": 3.7362,  # MLP比率
        "global_pool": "map",  # 全局池化类型
        "use_checkpoint": False,  # 是否使用检查点
    },
    "siglip_large_patch16_384": {  # SigLIP Large 16x16补丁 384分辨率
        "image_size": 384,  # 图像大小
        "patch_size": 16,  # 补丁大小
        "width": 1024,  # 宽度
        "layers": 24,  # 层数
        "heads": 16,  # 头数
        "mlp_ratio": 4,  # MLP比率
        "global_pool": "map",  # 全局池化类型
        "use_checkpoint": False,  # 是否使用检查点
    },
}


# 创建SigLIP视觉Transformer模型
def create_siglip_vit(
    model_name: str = "siglip_so400m_patch14_384",  # 模型名称
    image_size: int = 384,  # 图像大小
    select_layer: int = -1,  # 选择层
    ckpt_path: str = "",  # 检查点路径
    **kwargs,
):
    assert (  # 断言模型名称有效
        model_name in SigLIP_MODEL_CONFIG.keys()
    ), f"model name should be in {SigLIP_MODEL_CONFIG.keys()}"

    vision_cfg = SigLIPVisionCfg(**SigLIP_MODEL_CONFIG[model_name])  # 创建视觉配置

    if select_layer <= 0:  # 如果选择层为负（从后往前）
        layers = min(vision_cfg.layers, vision_cfg.layers + select_layer + 1)
    else:  # 正数选择层
        layers = min(vision_cfg.layers, select_layer)

    model = VisionTransformer(  # 创建视觉Transformer模型
        img_size=image_size,
        patch_size=vision_cfg.patch_size,
        embed_dim=vision_cfg.width,
        depth=layers,
        num_heads=vision_cfg.heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        class_token=vision_cfg.class_token,
        global_pool=vision_cfg.global_pool,
        ignore_head=kwargs.get("ignore_head", True),
        weight_init=kwargs.get("weight_init", "skip"),
        num_classes=0,
    )

    if ckpt_path:  # 如果有检查点路径
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # 加载状态字典

        incompatible_keys = model.load_state_dict(state_dict, strict=False)  # 加载权重
        print(
            f"SigLIP-ViT restores from {ckpt_path},\n"
            f"\tincompatible_keys:', {incompatible_keys}."
        )

    return model  # 返回模型


# 归一化模块
class Normalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """
    """使用均值和标准差对张量图像进行归一化。
    此变换不支持PIL图像。
    给定 ``n`` 个通道的均值 ``(mean[1],...,mean[n])`` 和标准差 ``(std[1],..,std[n])``，
    此变换将对输入的 ``torch.*Tensor`` 的每个通道进行归一化，即：
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. 注意::
        此变换是原地外操作，即不会修改输入张量。

    参数：
        mean (序列): 每个通道的均值序列。
        std (序列): 每个通道的标准差序列。
        inplace (布尔值，可选): 是否原地操作的布尔值。
    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()  # 调用父类初始化
        # _log_api_usage_once(self)
        self.mean = mean  # 均值
        self.std = std  # 标准差
        self.inplace = inplace  # 是否原地操作

    # 前向传播：归一化
    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        """
        参数：
            tensor (张量): 待归一化的张量图像。

        返回：
            张量: 归一化后的张量图像。
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)  # 执行归一化

    # 字符串表示
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# CLIP视觉塔模块
class CLIPVisionTower(nn.Module):
    def __init__(
        self,
        model_name: str = "siglip_large_patch16_384",  # 模型名称
        image_size: Union[Tuple[int, int], int] = 336,  # 图像大小
        select_feature: str = "patch",  # 选择特征类型
        select_layer: int = -2,  # 选择层
        select_layers: list = None,  # 选择多层
        ckpt_path: str = "",  # 检查点路径
        pixel_mean: Optional[List[float]] = None,  # 像素均值
        pixel_std: Optional[List[float]] = None,  # 像素标准差
        **kwargs,
    ):
        super().__init__()  # 调用父类初始化

        self.model_name = model_name  # 模型名称
        self.select_feature = select_feature  # 选择特征类型
        self.select_layer = select_layer  # 选择层
        self.select_layers = select_layers  # 选择多层

        vision_tower_params = {  # 视觉塔参数
            "model_name": model_name,
            "image_size": image_size,
            "ckpt_path": ckpt_path,
            "select_layer": select_layer,
        }
        vision_tower_params.update(kwargs)  # 更新额外参数
        self.vision_tower, self.forward_kwargs = self.build_vision_tower(  # 构建视觉塔
            vision_tower_params
        )

        if pixel_mean is not None and pixel_std is not None:  # 如果有像素归一化参数
            image_norm = Normalize(mean=pixel_mean, std=pixel_std)
        else:  # 无归一化参数
            image_norm = None

        self.image_norm = image_norm  # 图像归一化

    @property
    def device(self) -> torch.device:  # 获取设备
        return next(self.vision_tower.parameters()).device

    @property
    def dtype(self):  # 获取数据类型
        return next(self.vision_tower.parameters()).dtype

    # 构建视觉塔
    def build_vision_tower(self, vision_tower_params):
        if self.model_name.startswith("siglip"):  # SigLIP模型
            self.select_feature = "same"  # 使用相同特征
            vision_tower = create_siglip_vit(**vision_tower_params)  # 创建SigLIP ViT
            forward_kwargs = dict()

        elif self.model_name.startswith("sam"):  # SAM模型
            # vision_tower = create_sam_vit(**vision_tower_params)
            forward_kwargs = dict()

        else:  # huggingface  # HuggingFace模型
            from transformers import CLIPVisionModel

            vision_tower = CLIPVisionModel.from_pretrained(**vision_tower_params)  # 从预训练加载
            forward_kwargs = dict(output_hidden_states=True)  # 输出隐藏状态

        return vision_tower, forward_kwargs  # 返回视觉塔和前向参数

    # 特征选择
    def feature_select(self, image_forward_outs):
        if isinstance(image_forward_outs, torch.Tensor):  # 如果输出已是张量
            # the output has been the self.select_layer"s features
            # 输出已经是self.select_layer的特征
            image_features = image_forward_outs
        else:  # 否则从隐藏状态中选择
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == "patch":  # 选择补丁特征
            # if the output has cls_token
            # 如果输出包含cls_token
            image_features = image_features[:, 1:]  # 去掉CLS令牌
        elif self.select_feature == "cls_patch":  # 选择CLS和补丁特征
            image_features = image_features
        elif self.select_feature == "same":  # 选择相同特征
            image_features = image_features

        else:  # 无效选择
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features  # 返回图像特征

    # 前向传播
    def forward(self, images):
        """

        Args:
            images (torch.Tensor): [b, 3, H, W]

        Returns:
            image_features (torch.Tensor): [b, n_patch, d]
        """

        """

        参数：
            images (torch.Tensor): [b, 3, H, W]  # 图像张量

        返回：
            image_features (torch.Tensor): [b, n_patch, d]  # 图像特征
        """

        if self.image_norm is not None:  # 如果有归一化
            images = self.image_norm(images)  # 归一化图像

        image_forward_outs = self.vision_tower(images, **self.forward_kwargs)  # 视觉塔前向传播
        image_features = self.feature_select(image_forward_outs)  # 选择特征
        return image_features  # 返回图像特征


# MLP投影器模块
class MlpProjector(nn.Module):
    def __init__(self, cfg):  # 投影器配置
        super().__init__()  # 调用父类初始化

        self.cfg = cfg  # 保存配置

        if cfg["projector_type"] == "identity":  # 恒等投影
            modules = nn.Identity()

        elif cfg["projector_type"] == "linear":  # 线性投影
            modules = nn.Linear(cfg["input_dim"], cfg["n_embed"])

        elif cfg["projector_type"] == "mlp_gelu":  # MLP+GELU投影
            mlp_depth = cfg.get("depth", 1)  # MLP深度
            modules = [nn.Linear(cfg["input_dim"], cfg["n_embed"])]  # 第一个线性层
            for _ in range(1, mlp_depth):  # 添加隐藏层
                modules.append(nn.GELU())  # GELU激活
                modules.append(nn.Linear(cfg["n_embed"], cfg["n_embed"]))  # 线性层
            modules = nn.Sequential(*modules)  # 组合为序列

        elif cfg["projector_type"] == "low_high_hybrid_split_mlp_gelu":  # 低高分辨率混合分割MLP
            mlp_depth = cfg.get("depth", 1)  # MLP深度
            self.high_up_proj = nn.Linear(cfg["input_dim"], cfg["n_embed"] // 2)  # 高分辨率上投影
            self.low_up_proj = nn.Linear(cfg["input_dim"], cfg["n_embed"] // 2)  # 低分辨率上投影

            modules = []
            for _ in range(1, mlp_depth):  # 添加隐藏层
                modules.append(nn.GELU())  # GELU激活
                modules.append(nn.Linear(cfg["n_embed"], cfg["n_embed"]))  # 线性层
            modules = nn.Sequential(*modules)  # 组合为序列

        else:  # 未知投影类型
            raise ValueError(f"Unknown projector type: {cfg['projector_type']}")

        self.layers = modules  # 投影层

    # 前向传播
    def forward(
        self, x_or_tuple: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        """

        Args:
            x_or_tuple (Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:  if it is a tuple of torch.Tensor,
                then it comes from the hybrid vision encoder, and x = high_res_x, low_res_x);
                otherwise it is the feature from the single vision encoder.

        Returns:
            x (torch.Tensor): [b, s, c]
        """

        """

        参数：
            x_or_tuple (Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: 如果是torch.Tensor的元组，
                则来自混合视觉编码器，x = high_res_x, low_res_x)；
                否则是来自单一视觉编码器的特征。

        返回：
            x (torch.Tensor): [b, s, c]  # 批次，序列，通道
        """

        if isinstance(x_or_tuple, tuple):  # 如果是元组（混合编码器）
            # self.cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            high_x, low_x = x_or_tuple  # 分离高/低分辨率特征
            high_x = self.high_up_proj(high_x)  # 高分辨率上投影
            low_x = self.low_up_proj(low_x)  # 低分辨率上投影
            x = torch.cat([high_x, low_x], dim=-1)  # 拼接
        else:  # 单一编码器
            x = x_or_tuple

        return self.layers(x)  # 返回投影结果


# 层缩放模块
class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,  # 维度
        init_values: float = 1e-5,  # 初始值
        inplace: bool = False,  # 是否原地操作
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.inplace = inplace  # 是否原地操作
        self.gamma = nn.Parameter(init_values * torch.ones(dim))  # 缩放参数

    # 前向传播：层缩放
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma  # 原地或非原地缩放


# use torch.scaled_dot_product_attention where possible
# 尽可能使用torch.scaled_dot_product_attention
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, "scaled_dot_product_attention")  # 是否支持融合注意力
if "TIMM_FUSED_ATTN" in os.environ:  # 如果设置了融合注意力环境变量
    _USE_FUSED_ATTN = int(os.environ["TIMM_FUSED_ATTN"])
else:  # 默认启用
    _USE_FUSED_ATTN = (
        1  # 0 == off, 1 == on (for tested use), 2 == on (for experimental use)  # 0=关闭，1=开启（已测试），2=开启（实验性）
    )

# Set to True if exporting a model with Same padding via ONNX
# 如果通过ONNX导出具有Same填充的模型，则设为True
_EXPORTABLE = False


# 判断是否使用融合注意力
def use_fused_attn(experimental: bool = False) -> bool:
    # NOTE: ONNX export cannot handle F.scaled_dot_product_attention as of pytorch 2.0
    # 注意：截至PyTorch 2.0，ONNX导出无法处理F.scaled_dot_product_attention
    if not _HAS_FUSED_ATTN or _EXPORTABLE:  # 不支持融合注意力或需要导出
        return False
    if experimental:  # 实验性模式
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0  # 正常模式


# 注意力池化潜在层模块
class AttentionPoolLatent(nn.Module):
    """Attention pooling w/ latent query"""
    """带有潜在查询的注意力池化"""

    fused_attn: torch.jit.Final[bool]  # 是否使用融合注意力

    def __init__(
        self,
        in_features: int,  # 输入特征维度
        out_features: int = None,  # 输出特征维度
        embed_dim: int = None,  # 嵌入维度
        num_heads: int = 8,  # 头数
        feat_size: Optional[int] = None,  # 特征大小
        mlp_ratio: float = 4.0,  # MLP比率
        qkv_bias: bool = True,  # QKV偏置
        qk_norm: bool = False,  # QK归一化
        latent_len: int = 1,  # 潜在查询长度
        latent_dim: int = None,  # 潜在维度
        pos_embed: str = "",  # 位置嵌入类型
        pool_type: str = "token",  # 池化类型
        norm_layer: Optional[nn.Module] = None,  # 归一化层
        drop: float = 0.0,  # Dropout率
    ):
        super().__init__()  # 调用父类初始化
        embed_dim = embed_dim or in_features  # 嵌入维度
        out_features = out_features or in_features  # 输出维度
        assert embed_dim % num_heads == 0  # 断言嵌入维度可被头数整除
        self.num_heads = num_heads  # 头数
        self.head_dim = embed_dim // num_heads  # 每头维度
        self.feat_size = feat_size  # 特征大小
        self.scale = self.head_dim**-0.5  # 缩放因子
        self.pool = pool_type  # 池化类型
        self.fused_attn = use_fused_attn()  # 是否使用融合注意力

        if pos_embed == "abs":  # 绝对位置嵌入
            assert feat_size is not None
            self.pos_embed = nn.Parameter(torch.zeros(feat_size, in_features))
        else:  # 无位置嵌入
            self.pos_embed = None

        self.latent_dim = latent_dim or embed_dim  # 潜在维度
        self.latent_len = latent_len  # 潜在长度
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))  # 潜在查询参数

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)  # Q线性层
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)  # KV线性层
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()  # Q归一化
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()  # K归一化
        self.proj = nn.Linear(embed_dim, embed_dim)  # 投影层
        self.proj_drop = nn.Dropout(drop)  # 投影Dropout

        self.norm = (  # 归一化层
            norm_layer(out_features) if norm_layer is not None else nn.Identity()
        )
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio))  # MLP层

        self.init_weights()  # 初始化权重

    # 初始化权重
    def init_weights(self):
        if self.pos_embed is not None:  # 如果有位置嵌入
            trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
        trunc_normal_tf_(self.latent, std=self.latent_dim**-0.5)  # 潜在变量初始化

    # 前向传播：注意力池化
    def forward(self, x):
        B, N, C = x.shape  # 批次、序列长度、通道

        if self.pos_embed is not None:  # 如果有位置嵌入
            # FIXME interpolate
            # 待修复：插值
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = self.latent.expand(B, -1, -1)  # 扩展潜在查询
        q = (  # 计算Q
            self.q(q_latent)
            .reshape(B, self.latent_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv = (  # 计算KV
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # 分离K和V

        q, k = self.q_norm(q), self.k_norm(k)  # QK归一化

        if self.fused_attn:  # 融合注意力
            x = F.scaled_dot_product_attention(q, k, v)
        else:  # 手动注意力
            q = q * self.scale  # 缩放
            attn = q @ k.transpose(-2, -1)  # 注意力矩阵
            attn = attn.softmax(dim=-1)  # Softmax
            x = attn @ v  # 加权求和
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)  # 重塑
        x = self.proj(x)  # 投影
        x = self.proj_drop(x)  # Dropout

        x = x + self.mlp(self.norm(x))  # MLP残差连接

        # optional pool if latent seq_len > 1 and pooled output is desired
        # 如果潜在序列长度>1且需要池化输出，则可选池化
        if self.pool == "token":  # 令牌池化
            x = x[:, 0]
        elif self.pool == "avg":  # 平均池化
            x = x.mean(1)


# 编码器模块
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,  # 输入通道数
        ch=128,  # 基础通道数
        ch_mult=(1, 1, 2, 2, 4),  # 通道倍数
        num_res_blocks=2,  # 残差块数
        norm_type="group",  # 归一化类型
        dropout=0.0,  # Dropout率
        resamp_with_conv=True,  # 是否用卷积重采样
        z_channels=256,  # 潜在通道数
    ):
        super().__init__()  # 调用父类初始化
        self.num_resolutions = len(ch_mult)  # 分辨率级数
        self.num_res_blocks = num_res_blocks  # 残差块数
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)  # 输入卷积

        # downsampling
        # 下采样
        in_ch_mult = (1,) + tuple(ch_mult)  # 输入通道倍数（添加初始1）
        self.conv_blocks = nn.ModuleList()  # 卷积块列表
        for i_level in range(self.num_resolutions):  # 遍历每个分辨率级
            conv_block = nn.Module()  # 卷积块模块
            # res & attn
            # 残差和注意力
            res_block = nn.ModuleList()  # 残差块列表
            attn_block = nn.ModuleList()  # 注意力块列表
            block_in = ch * in_ch_mult[i_level]  # 输入通道数
            block_out = ch * ch_mult[i_level]  # 输出通道数
            for _ in range(self.num_res_blocks):  # 添加残差块
                res_block.append(
                    ResnetBlock(
                        block_in, block_out, dropout=dropout, norm_type=norm_type
                    )
                )
                block_in = block_out  # 更新输入通道数
                if i_level == self.num_resolutions - 1:  # 最后一层添加注意力
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block  # 设置残差块
            conv_block.attn = attn_block  # 设置注意力块
            # downsample
            # 下采样
            if i_level != self.num_resolutions - 1:  # 非最后一层添加下采样
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        # 中间层
        self.mid = nn.ModuleList()
        self.mid.append(  # 残差块
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )
        self.mid.append(AttnBlock(block_in, norm_type))  # 注意力块
        self.mid.append(  # 残差块
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )

        # end
        # 结束层
        self.norm_out = Normalize(block_in, norm_type)  # 输出归一化
        self.conv_out = nn.Conv2d(  # 输出卷积
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )

    # 前向传播：编码器
    def forward(self, x):
        h = self.conv_in(x)  # 输入卷积
        # downsampling
        # 下采样
        for i_level, block in enumerate(self.conv_blocks):  # 遍历每个分辨率级
            for i_block in range(self.num_res_blocks):  # 遍历残差块
                h = block.res[i_block](h)  # 残差块
                if len(block.attn) > 0:  # 如果有注意力块
                    h = block.attn[i_block](h)  # 注意力块
            if i_level != self.num_resolutions - 1:  # 非最后一层下采样
                h = block.downsample(h)

        # middle
        # 中间层
        for mid_block in self.mid:
            h = mid_block(h)  # 中间层块

        # end
        # 结束层
        h = self.norm_out(h)  # 归一化
        h = nonlinearity(h)  # 非线性激活
        h = self.conv_out(h)  # 输出卷积
        return h  # 返回编码结果


# 解码器模块
class Decoder(nn.Module):
    def __init__(
        self,
        z_channels=256,  # 潜在通道数
        ch=128,  # 基础通道数
        ch_mult=(1, 1, 2, 2, 4),  # 通道倍数
        num_res_blocks=2,  # 残差块数
        norm_type="group",  # 归一化类型
        dropout=0.0,  # Dropout率
        resamp_with_conv=True,  # 是否用卷积重采样
        out_channels=3,  # 输出通道数
    ):
        super().__init__()  # 调用父类初始化
        self.num_resolutions = len(ch_mult)  # 分辨率级数
        self.num_res_blocks = num_res_blocks  # 残差块数

        block_in = ch * ch_mult[self.num_resolutions - 1]  # 输入通道数（最大级）
        # z to block_in
        # z到block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        # 中间层
        self.mid = nn.ModuleList()
        self.mid.append(  # 残差块
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )
        self.mid.append(AttnBlock(block_in, norm_type))  # 注意力块
        self.mid.append(  # 残差块
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )

        # upsampling
        # 上采样
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):  # 从高到低遍历
            conv_block = nn.Module()
            # res & attn
            # 残差和注意力
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]  # 输出通道数
            for _ in range(self.num_res_blocks + 1):  # 解码器多一个残差块
                res_block.append(
                    ResnetBlock(
                        block_in, block_out, dropout=dropout, norm_type=norm_type
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions - 1:  # 最高层添加注意力
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            # 上采样
            if i_level != 0:  # 非最底层添加上采样
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        # 结束层
        self.norm_out = Normalize(block_in, norm_type)  # 输出归一化
        self.conv_out = nn.Conv2d(
            block_in, out_channels, kernel_size=3, stride=1, padding=1
        )

    @property
    def last_layer(self):  # 获取最后一层
        return self.conv_out.weight

    # 前向传播：解码器
    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)  # 输入卷积

        # middle
        # 中间层
        for mid_block in self.mid:
            h = mid_block(h)

        # upsampling
        # 上采样
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):  # 遍历残差块
                h = block.res[i_block](h)  # 残差块
                if len(block.attn) > 0:  # 注意力块
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:  # 非最底层上采样
                h = block.upsample(h)

        # end
        # 结束层
        h = self.norm_out(h)  # 归一化
        h = nonlinearity(h)  # 非线性激活
        h = self.conv_out(h)  # 输出卷积
        return h  # 返回解码结果


# 向量量化器模块
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):  # 码本大小、嵌入维度、beta、熵损失率、L2归一化、显示使用情况
        super().__init__()
        self.n_e = n_e  # 码本大小
        self.e_dim = e_dim  # 嵌入维度
        self.beta = beta  # 提交损失系数
        self.entropy_loss_ratio = entropy_loss_ratio  # 熵损失比率
        self.l2_norm = l2_norm  # 是否L2归一化
        self.show_usage = show_usage  # 是否显示码本使用情况

        self.embedding = nn.Embedding(self.n_e, self.e_dim)  # 码本嵌入层
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)  # 均匀初始化
        if self.l2_norm:  # 如果使用L2归一化
            self.embedding.weight.data = F.normalize(
                self.embedding.weight.data, p=2, dim=-1
            )
        if self.show_usage:  # 如果显示使用情况
            # self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))
            self.codebook_used = nn.Parameter(torch.zeros(65536))  # 码本使用计数

    # 前向传播：向量量化
    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        # 重塑z -> (batch, height, width, channel)并展平
        z = torch.einsum("b c h w -> b h w c", z).contiguous()  # 重排维度
        z_flattened = z.view(-1, self.e_dim)  # 展平
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # 从z到嵌入e_j的距离 (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:  # 如果使用L2归一化
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:  # 不归一化
            embedding = self.embedding.weight

        d = (  # 计算距离矩阵
            torch.sum(z_flattened**2, dim=1, keepdim=True)  # z^2
            + torch.sum(embedding**2, dim=1)  # e^2
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, torch.einsum("n d -> d n", embedding)  # -2*e*z
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)  # 最近码本索引
        z_q = embedding[min_encoding_indices].view(z.shape)  # 量化后的向量
        perplexity = None  # 困惑度
        min_encodings = None  # 最小编码
        vq_loss = None  # VQ损失
        commit_loss = None  # 提交损失
        entropy_loss = None  # 熵损失

        # compute loss for embedding
        # 计算嵌入损失
        if self.training:  # 训练时计算损失
            vq_loss = torch.mean((z_q - z.detach()) ** 2)  # VQ损失
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)  # 提交损失
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)  # 熵损失

        # preserve gradients
        # 保留梯度（直通估计器）
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        # 重塑回原始输入形状
        z_q = torch.einsum("b h w c -> b c h w", z_q)

        return (  # 返回量化结果、损失和编码信息
            z_q,
            (vq_loss, commit_loss, entropy_loss),
            (perplexity, min_encodings, min_encoding_indices),
        )

    # 获取码本条目
    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        # 形状 = (batch, channel, height, width) 如果channel_first 否则 (batch, height, width, channel)
        if self.l2_norm:  # 如果使用L2归一化
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:  # 不归一化
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)  # 根据索引获取码本嵌入

        if shape is not None:  # 如果指定了形状
            if channel_first:  # 通道在前
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                # 重塑回原始输入形状
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:  # 通道在后
                z_q = z_q.view(shape)
        return z_q  # 返回码本条目


# 残差块模块
class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,  # 输入通道数
        out_channels=None,  # 输出通道数
        conv_shortcut=False,  # 是否使用卷积快捷连接
        dropout=0.0,  # Dropout率
        norm_type="group",  # 归一化类型
    ):
        super().__init__()
        self.in_channels = in_channels  # 输入通道数
        out_channels = in_channels if out_channels is None else out_channels  # 默认输出等于输入
        self.out_channels = out_channels  # 输出通道数
        self.use_conv_shortcut = conv_shortcut  # 是否使用卷积快捷连接

        self.norm1 = Normalize(in_channels, norm_type)  # 第一个归一化
        self.conv1 = nn.Conv2d(  # 第一个卷积
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels, norm_type)  # 第二个归一化
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.conv2 = nn.Conv2d(  # 第二个卷积
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if self.in_channels != self.out_channels:  # 如果输入输出通道不同
            if self.use_conv_shortcut:  # 使用卷积快捷连接
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:  # 使用1x1卷积快捷连接
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    # 前向传播：残差块
    def forward(self, x):
        h = x
        h = self.norm1(h)  # 第一个归一化
        h = nonlinearity(h)  # 非线性激活
        h = self.conv1(h)  # 第一个卷积
        h = self.norm2(h)  # 第二个归一化
        h = nonlinearity(h)  # 非线性激活
        h = self.dropout(h)  # Dropout
        h = self.conv2(h)  # 第二个卷积

        if self.in_channels != self.out_channels:  # 如果通道不同需要快捷连接
            if self.use_conv_shortcut:  # 卷积快捷连接
                x = self.conv_shortcut(x)
            else:  # 1x1卷积快捷连接
                x = self.nin_shortcut(x)
        return x + h  # 残差连接


# 注意力块模块
class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type="group"):  # 输入通道数和归一化类型
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)  # 归一化
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)  # Q卷积
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)  # K卷积
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)  # V卷积
        self.proj_out = nn.Conv2d(  # 输出投影卷积
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    # 前向传播：注意力块
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)  # 归一化
        q = self.q(h_)  # 计算Q
        k = self.k(h_)  # 计算K
        v = self.v(h_)  # 计算V

        # compute attention
        # 计算注意力
        b, c, h, w = q.shape  # 获取形状
        q = q.reshape(b, c, h * w)  # 重塑Q
        q = q.permute(0, 2, 1)  # b,hw,c  # 转置
        k = k.reshape(b, c, h * w)  # b,c,hw  # 重塑K
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]  # 批量矩阵乘法
        w_ = w_ * (int(c) ** (-0.5))  # 缩放
        w_ = F.softmax(w_, dim=2)  # Softmax

        # attend to values
        # 对值进行注意力
        v = v.reshape(b, c, h * w)  # 重塑V
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)  # 转置
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]  # 加权求和
        h_ = h_.reshape(b, c, h, w)  # 重塑回原始形状

        h_ = self.proj_out(h_)  # 输出投影

        return x + h_  # 残差连接


# 非线性激活函数（Swish）
def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)  # Swish激活


# 归一化层工厂函数
def Normalize(in_channels, norm_type="group"):  # 输入通道数和归一化类型
    assert norm_type in ["group", "batch"]  # 断言归一化类型有效
    if norm_type == "group":  # 组归一化
        return nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
    elif norm_type == "batch":  # 批量归一化
        return nn.SyncBatchNorm(in_channels)


# 上采样模块
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):  # 输入通道数和是否使用卷积
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:  # 如果使用卷积
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    # 前向传播：上采样
    def forward(self, x):
        if x.dtype != torch.float32:  # 非float32时需要类型转换
            x = F.interpolate(x.to(torch.float), scale_factor=2.0, mode="nearest").to(
                torch.bfloat16
            )
        else:  # float32直接插值
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.with_conv:  # 如果使用卷积
            x = self.conv(x)
        return x  # 返回上采样结果


# 下采样模块
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):  # 输入通道数和是否使用卷积
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:  # 如果使用卷积
            # no asymmetric padding in torch conv, must do it ourselves
            # PyTorch卷积没有非对称填充，必须手动实现
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    # 前向传播：下采样
    def forward(self, x):
        if self.with_conv:  # 卷积下采样
            pad = (0, 1, 0, 1)  # 填充
            x = F.pad(x, pad, mode="constant", value=0)  # 填充
            x = self.conv(x)  # 卷积
        else:  # 平均池化下采样
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x  # 返回下采样结果


# 计算熵损失
def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):  # 亲和度矩阵、损失类型、温度
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])  # 展平亲和度
    flat_affinity /= temperature  # 温度缩放
    probs = F.softmax(flat_affinity, dim=-1)  # 概率分布
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)  # 对数概率
    if loss_type == "softmax":  # Softmax损失
        target_probs = probs
    else:  # 不支持的损失类型
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)  # 平均概率
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))  # 平均熵
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))  # 样本熵
    loss = sample_entropy - avg_entropy  # 熵损失
    return loss  # 返回损失


# VQ模型（向量量化变分自编码器）
class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):  # 模型参数
        super().__init__()
        self.config = config  # 保存配置
        self.encoder = Encoder(  # 编码器
            ch_mult=config.encoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
        )
        self.decoder = Decoder(  # 解码器
            ch_mult=config.decoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
        )

        self.quantize = VectorQuantizer(  # 向量量化器
            config.codebook_size,
            config.codebook_embed_dim,
            config.commit_loss_beta,
            config.entropy_loss_ratio,
            config.codebook_l2_norm,
            config.codebook_show_usage,
        )
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)  # 量化前卷积
        self.post_quant_conv = nn.Conv2d(  # 量化后卷积
            config.codebook_embed_dim, config.z_channels, 1
        )

    # 编码
    def encode(self, x):
        h = self.encoder(x)  # 编码
        h = self.quant_conv(h)  # 量化前卷积
        quant, emb_loss, info = self.quantize(h)  # 量化
        return quant, emb_loss, info  # 返回量化结果、嵌入损失和信息

    # 解码
    def decode(self, quant):
        quant = self.post_quant_conv(quant)  # 量化后卷积
        dec = self.decoder(quant)  # 解码
        return dec  # 返回解码结果

    # 从码本解码
    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)  # 获取码本条目
        dec = self.decode(quant_b)  # 解码
        return dec  # 返回解码结果

    # 前向传播：VQ模型
    def forward(self, input):
        quant, diff, _ = self.encode(input)  # 编码
        dec = self.decode(quant)  # 解码
        return dec, diff  # 返回解码结果和损失


# 多模态预训练模型基类
class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig  # 配置类
    base_model_prefix = "multi_modality"  # 基础模型前缀
    _no_split_modules = []  # 不分割模块列表
    _skip_keys_device_placement = "past_key_values"  # 跳过设备放置的键


# Copied and adapted from:
# 从以下仓库复制和适配：
# https://github.com/deepseek-ai/Janus/tree/main/janus/models/modeling_vlm.py
# 多模态因果语言模型
class MultiModalityCausalLM(MultiModalityPreTrainedModel):

    def __init__(
        self,
        config: MultiModalityConfig,  # 多模态配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
    ):
        super().__init__(config)  # 调用父类初始化

        vision_config = config.vision_config  # 视觉配置
        vision_cls = model_name_to_cls(vision_config.cls)  # 获取视觉模型类
        self.vision_model = vision_cls(**vision_config.params)  # 创建视觉模型

        aligner_config = config.aligner_config  # 对齐器配置
        aligner_cls = model_name_to_cls(aligner_config.cls)  # 获取对齐器类
        self.aligner = aligner_cls(aligner_config.params)  # 创建对齐器

        gen_vision_config = config.gen_vision_config  # 生成视觉配置
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)  # 获取生成视觉模型类
        self.gen_vision_model = gen_vision_cls()  # 创建生成视觉模型

        gen_aligner_config = config.gen_aligner_config  # 生成对齐器配置
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)  # 获取生成对齐器类
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)  # 创建生成对齐器

        gen_head_config = config.gen_head_config  # 生成头配置
        gen_head_cls = model_name_to_cls(gen_head_config.cls)  # 获取生成头类
        self.gen_head = gen_head_cls(gen_head_config.params)  # 创建生成头

        self.gen_embed = torch.nn.Embedding(  # 生成嵌入层
            gen_vision_config.params["image_token_size"],
            gen_vision_config.params["n_embed"],
        )

        language_config = config.language_config  # 语言配置
        self.language_model = LlamaForCausalLM(  # 创建语言模型
            language_config, quant_config=quant_config
        )
        self.logits_processor = LogitsProcessor(language_config)  # 逻辑处理器

    # 获取图像特征
    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.concat([item.feature for item in items], dim=0)  # 拼接像素值
        bs, n = pixel_values.shape[0:2]  # 批次大小和视图数
        pixel_values = pixel_values.to(
            device=self.vision_model.device, dtype=self.vision_model.dtype
        )  # 转移到视觉模型设备和类型
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")  # 重排像素值

        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))  # 视觉模型和对齐器

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)  # 重排嵌入

        return images_embeds  # 返回图像嵌入

    # 获取输入嵌入层
    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    @torch.no_grad()
    # 前向传播
    def forward(
        self,
        input_ids: torch.LongTensor,  # 输入令牌ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        get_embedding: bool = False,  # 是否获取嵌入
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入例程
            input_ids=input_ids,
            forward_batch=forward_batch,
            multimodal_model=self,
            language_model=self.language_model,
            positions=positions,
        )

        return hidden_states  # 返回隐藏状态

    # 准备生成图像嵌入
    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))  # 生成嵌入经过对齐器

    # 填充输入ID以适配图像输入
    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        im_start_id = image_inputs.im_start_id  # 图像起始令牌ID
        im_end_id = image_inputs.im_end_id  # 图像结束令牌ID
        media_token_pairs = [(im_start_id, im_end_id)]  # 媒体令牌对

        helper = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)  # 创建填充辅助器

        return helper.pad_input_tokens(input_ids, image_inputs)  # 返回填充后的令牌

    # 加载权重
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())  # 参数字典
        for name, loaded_weight in weights:  # 遍历权重
            if "rotary_emb.inv_freq~" in name or "projector" in name:  # 跳过旋转频率和投影器
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过旋转缓存
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                # 使用ColossalAI训练的模型可能在检查点中包含这些张量。跳过它们。
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:  # 跳过不在参数中的视觉塔权重
                continue

            # skip generation sub model
            # 跳过生成子模型
            if "gen" in name:
                continue

            # adapt to VisionAttention
            # 适配VisionAttention
            name = name.replace(r"self_attn.out_proj", r"self_attn.proj")
            if "vision_model.vision_tower" in name:
                name = name.replace("attn.qkv", "attn.qkv_proj")

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 处理堆叠参数
                # replace the name and load with customized loader
                # 替换名称并使用自定义加载器加载
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                # # Skip loading extra bias for GPTQ models.
                # 跳过GPTQ模型的额外偏置加载。
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", None)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:  # 非堆叠参数
                # Skip loading extra bias for GPTQ models.
                # 跳过GPTQ模型的额外偏置加载。
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


AutoModel.register(config_class=MultiModalityConfig, model_class=MultiModalityCausalLM)  # 注册模型
EntryClass = [MultiModalityCausalLM]  # 入口类
