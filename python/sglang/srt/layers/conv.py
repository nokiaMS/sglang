# Conv2d/Conv3d 卷积层实现，使用 unfold+linear 优化用于 patch embedding。
# 当 kernel_size == stride, padding == 0, dilation == 1, groups == 1 时，
# 卷积等价于 unfold + F.linear，在 CUDA 上显著更快，
# 同时避免了 PyTorch 2.9.1 + CuDNN < 9.15 的 Conv3d bug。

"""
Conv2d/Conv3d layers with unfold+linear optimization for patch embeddings.

When kernel_size == stride, padding == 0, dilation == 1, groups == 1, the conv
is equivalent to unfold + F.linear, which is significantly faster on CUDA and
also avoids the PyTorch 2.9.1 + CuDNN < 9.15 Conv3d bug
(https://github.com/pytorch/pytorch/issues/168167).
"""  # Conv2d/Conv3d 层，使用 unfold+linear 优化用于 patch embedding。当 kernel_size==stride, padding==0, dilation==1, groups==1 时，卷积等价于 unfold+F.linear，在 CUDA 上更快，且避免 CuDNN bug。

import math  # 导入数学库
from typing import Tuple, Union  # 导入类型注解

import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口

from sglang.srt.layers.amx_utils import PackWeightMethod  # 导入 AMX 权重打包方法
from sglang.srt.layers.utils.multi_platform import MultiPlatformOp  # 导入多平台操作基类
from sglang.srt.utils import cpu_has_amx_support, is_cpu, use_intel_amx_backend  # 导入 CPU 检测工具

_is_cpu = is_cpu()  # 检测当前是否为 CPU 平台
_is_cpu_amx_available = cpu_has_amx_support()  # 检测 CPU 是否支持 AMX 指令集
if _is_cpu and _is_cpu_amx_available:  # 如果是 CPU 且支持 AMX
    conv3d_embed = torch.ops.sgl_kernel.conv3d_embed_cpu  # 加载 CPU 端 Conv3d 嵌入算子

_VALID_PADDING_STRINGS = {"same", "valid"}  # 合法的字符串填充模式
_VALID_PADDING_MODES = {"zeros", "reflect", "replicate", "circular"}  # 合法的填充模式


def _tuplify(val, n: int) -> tuple:  # 将标量或序列转为长度为 n 的元组
    if isinstance(val, (list, tuple)):  # 如果已经是列表或元组
        assert len(val) == n  # 断言长度匹配
        return tuple(val)  # 转为元组返回
    return (val,) * n  # 标量重复 n 次构成元组


def _check_enable_linear(  # 检查是否可以将卷积替换为 unfold + F.linear
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> bool:
    """Check if conv can be replaced with unfold + F.linear."""  # 检查卷积是否可替换为 unfold + F.linear
    return (  # 返回是否满足所有条件
        kernel_size == stride  # kernel_size 必须等于 stride
        and all(p == 0 for p in padding)  # 所有 padding 必须为 0
        and all(d == 1 for d in dilation)  # 所有 dilation 必须为 1
        and groups == 1  # groups 必须为 1
    )


def _reverse_repeat_tuple(t: tuple) -> tuple:  # 反转元组并将每个元素重复两次，用于 F.pad 的非零 padding_mode
    """(1, 2, 3) -> (3, 3, 2, 2, 1, 1). Used for F.pad with non-zeros padding_mode."""  # (1,2,3)->(3,3,2,2,1,1)，用于非零 padding_mode 的 F.pad
    return tuple(x for x in reversed(t) for _ in range(2))  # 反转后每个元素重复两次


def _compute_same_padding_for_pad(kernel_size: tuple, dilation: tuple) -> tuple:  # 计算填充为 "same" 时的填充量
    """Compute _reversed_padding_repeated_twice for padding='same'.

    This mirrors PyTorch's nn.Conv*d behavior: pre-compute the exact pad
    amounts so that F.pad can be called before F.conv*d(padding=0).
    """  # 计算 padding='same' 时的 _reversed_padding_repeated_twice。模仿 PyTorch nn.Conv*d 行为：预计算精确填充量，以便 F.pad 在 F.conv*d(padding=0) 之前调用。
    pad = []  # 填充列表
    for k, d in zip(reversed(kernel_size), reversed(dilation)):  # 反向遍历 kernel_size 和 dilation
        total = d * (k - 1)  # 计算总填充量
        pad.append(total // 2)  # 左/上侧填充
        pad.append(total - total // 2)  # 右/下侧填充（处理奇数情况）
    return tuple(pad)  # 返回填充元组


def _validate_conv_args(  # 验证卷积参数的合法性
    in_channels: int,
    out_channels: int,
    groups: int,
    padding,
    padding_mode: str,
    stride: tuple,
) -> None:
    if in_channels % groups != 0:  # 输入通道数必须能被 groups 整除
        raise ValueError(
            f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        )
    if out_channels % groups != 0:  # 输出通道数必须能被 groups 整除
        raise ValueError(
            f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        )
    if padding_mode not in _VALID_PADDING_MODES:  # 填充模式必须合法
        raise ValueError(
            f"padding_mode must be one of {_VALID_PADDING_MODES}, got '{padding_mode}'"
        )
    if isinstance(padding, str):  # 如果填充是字符串
        if padding not in _VALID_PADDING_STRINGS:  # 字符串填充必须合法
            raise ValueError(
                f"padding must be one of {_VALID_PADDING_STRINGS}, got '{padding}'"
            )
        if padding == "same" and any(s != 1 for s in stride):  # "same" 填充不支持步幅卷积
            raise ValueError("padding='same' is not supported for strided convolutions")


class Conv2dLayer(MultiPlatformOp):  # Conv2d 层的替换实现，默认禁用 linear 优化
    """Drop-in replacement for nn.Conv2d. Linear optimization disabled by default."""  # nn.Conv2d 的直接替换。默认禁用 linear 优化。

    def __init__(  # 初始化 Conv2d 层
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        kernel_size: Union[int, Tuple[int, int]],  # 卷积核大小
        stride: Union[int, Tuple[int, int]] = 1,  # 步幅，默认为 1
        padding: Union[int, Tuple[int, int], str] = 0,  # 填充，默认为 0
        dilation: Union[int, Tuple[int, int]] = 1,  # 膨胀率，默认为 1
        groups: int = 1,  # 分组数，默认为 1
        bias: bool = True,  # 是否使用偏置，默认为 True
        padding_mode: str = "zeros",  # 填充模式，默认为 "zeros"
        disable_linear: bool = True,  # 是否禁用 linear 优化，默认为 True
    ):
        super().__init__()  # 调用父类初始化
        self.in_channels = in_channels  # 保存输入通道数
        self.out_channels = out_channels  # 保存输出通道数
        self.kernel_size = _tuplify(kernel_size, 2)  # 将 kernel_size 转为二元组
        self.stride = _tuplify(stride, 2)  # 将 stride 转为二元组
        self.dilation = _tuplify(dilation, 2)  # 将 dilation 转为二元组
        self.groups = groups  # 保存分组数
        self.padding_mode = padding_mode  # 保存填充模式

        _validate_conv_args(  # 验证卷积参数合法性
            in_channels, out_channels, groups, padding, padding_mode, self.stride
        )

        if isinstance(padding, str):  # 如果填充是字符串
            self.padding = (0, 0) if padding == "valid" else padding  # "valid" 设为 (0,0)，否则保留 "same"
        else:
            self.padding = _tuplify(padding, 2)  # 数值填充转为二元组

        # Pre-compute pad tuple for padding_mode != "zeros" (mirrors nn.Conv2d).
        # When padding="same", we need numeric values for F.pad;
        # when padding is already numeric, _reverse_repeat_tuple handles it.
        # 预计算填充元组，用于 padding_mode != "zeros"（模仿 nn.Conv2d）。当 padding="same" 时，需要数值用于 F.pad；当 padding 已是数值时，由 _reverse_repeat_tuple 处理。
        if isinstance(self.padding, str):  # 如果填充仍是字符串（即 "same"）
            self._reversed_padding_repeated_twice = _compute_same_padding_for_pad(  # 计算 "same" 模式的填充
                self.kernel_size, self.dilation
            )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding)  # 数值填充反转重复

        padding_tuple = self.padding if isinstance(self.padding, tuple) else (1, 1)  # 获取数值填充元组
        self.enable_linear = not disable_linear and _check_enable_linear(  # 判断是否启用 linear 优化
            self.kernel_size, self.stride, padding_tuple, self.dilation, groups
        )

        self.weight = nn.Parameter(  # 创建权重参数
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:  # 如果使用偏置
            self.bias = nn.Parameter(torch.empty(out_channels))  # 创建偏置参数
        else:
            self.register_parameter("bias", None)  # 注册 None 偏置

        self._reset_parameters()  # 重置/初始化参数

    def _reset_parameters(self):  # 重置参数，使用 kaiming 均匀初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # 权重使用 kaiming 均匀初始化
        if self.bias is not None:  # 如果有偏置
            fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")  # 计算扇入
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0  # 计算均匀初始化边界
            nn.init.uniform_(self.bias, -bound, bound)  # 偏置使用均匀初始化

    def _forward_mulmat(self, x: torch.Tensor) -> torch.Tensor:  # 使用 unfold + linear 的前向传播
        K1, K2 = self.kernel_size  # 获取卷积核大小
        x = x.unfold(2, K1, K1).unfold(3, K2, K2)  # 在空间维度上展开
        N, _, Hp, Wp = x.shape[:4]  # 获取批次大小和展开后的空间维度
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(N, Hp, Wp, -1)  # 重排并展平通道和核维度
        x = F.linear(x, self.weight.reshape(self.out_channels, -1), self.bias)  # 使用线性层计算
        return x.permute(0, 3, 1, 2)  # 恢复 (N, C, H, W) 格式

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:  # 使用标准 F.conv2d 的前向传播
        if self.padding_mode != "zeros":  # 非零填充模式需要先手动填充
            return F.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),  # 先手动填充
                self.weight,
                self.bias,
                self.stride,
                (0, 0),  # 填充已在 F.pad 中完成
                self.dilation,
                self.groups,
            )
        return F.conv2d(  # 零填充模式直接使用 F.conv2d
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:  # 原生前向传播
        if self.enable_linear:  # 如果启用 linear 优化
            return self._forward_mulmat(x)  # 使用 unfold+linear
        return self._forward_conv(x)  # 否则使用标准卷积

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:  # CUDA 前向传播
        if self.enable_linear:  # 如果启用 linear 优化
            return self._forward_mulmat(x)  # 使用 unfold+linear
        return self._forward_conv(x)  # 否则使用标准卷积


class Conv3dLayer(MultiPlatformOp):  # Conv3d 层的替换实现，自动启用 linear 优化
    """Drop-in replacement for nn.Conv3d with automatic linear optimization."""  # nn.Conv3d 的直接替换，自动启用 linear 优化。

    def __init__(  # 初始化 Conv3d 层
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        kernel_size: Union[int, Tuple[int, int, int]],  # 卷积核大小
        stride: Union[int, Tuple[int, int, int]] = 1,  # 步幅，默认为 1
        padding: Union[int, Tuple[int, int, int], str] = 0,  # 填充，默认为 0
        dilation: Union[int, Tuple[int, int, int]] = 1,  # 膨胀率，默认为 1
        groups: int = 1,  # 分组数，默认为 1
        bias: bool = True,  # 是否使用偏置，默认为 True
        padding_mode: str = "zeros",  # 填充模式，默认为 "zeros"
        disable_linear: bool = False,  # 是否禁用 linear 优化，默认为 False（即默认启用）
    ):
        super().__init__()  # 调用父类初始化
        self.in_channels = in_channels  # 保存输入通道数
        self.out_channels = out_channels  # 保存输出通道数
        self.kernel_size = _tuplify(kernel_size, 3)  # 将 kernel_size 转为三元组
        self.stride = _tuplify(stride, 3)  # 将 stride 转为三元组
        self.dilation = _tuplify(dilation, 3)  # 将 dilation 转为三元组
        self.groups = groups  # 保存分组数
        self.padding_mode = padding_mode  # 保存填充模式

        _validate_conv_args(  # 验证卷积参数合法性
            in_channels, out_channels, groups, padding, padding_mode, self.stride
        )

        if isinstance(padding, str):  # 如果填充是字符串
            self.padding = (0, 0, 0) if padding == "valid" else padding  # "valid" 设为 (0,0,0)，否则保留 "same"
        else:
            self.padding = _tuplify(padding, 3)  # 数值填充转为三元组

        if isinstance(self.padding, str):  # 如果填充仍是字符串（即 "same"）
            self._reversed_padding_repeated_twice = _compute_same_padding_for_pad(  # 计算 "same" 模式的填充
                self.kernel_size, self.dilation
            )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding)  # 数值填充反转重复

        padding_tuple = self.padding if isinstance(self.padding, tuple) else (1, 1, 1)  # 获取数值填充元组
        self.enable_linear = not disable_linear and _check_enable_linear(  # 判断是否启用 linear 优化
            self.kernel_size, self.stride, padding_tuple, self.dilation, groups
        )

        self.weight = nn.Parameter(  # 创建权重参数
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:  # 如果使用偏置
            self.bias = nn.Parameter(torch.empty(out_channels))  # 创建偏置参数
        else:
            self.register_parameter("bias", None)  # 注册 None 偏置

        if _is_cpu and _is_cpu_amx_available and self.bias is not None:  # 如果是 CPU 且支持 AMX 且有偏置
            self.quant_method = PackWeightMethod(weight_names=["weight"])  # 使用 AMX 权重打包方法
        self._reset_parameters()  # 重置/初始化参数

    def _reset_parameters(self):  # 重置参数，使用 kaiming 均匀初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # 权重使用 kaiming 均匀初始化
        if self.bias is not None:  # 如果有偏置
            fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")  # 计算扇入
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0  # 计算均匀初始化边界
            nn.init.uniform_(self.bias, -bound, bound)  # 偏置使用均匀初始化

    def _forward_mulmat(self, x: torch.Tensor) -> torch.Tensor:  # 使用 unfold + linear 的前向传播
        K1, K2, K3 = self.kernel_size  # 获取三维卷积核大小
        x = x.unfold(2, K1, K1).unfold(3, K2, K2).unfold(4, K3, K3)  # 在三个空间维度上展开
        N, Dp, Hp, Wp = x.shape[0], x.shape[2], x.shape[3], x.shape[4]  # 获取批次和展开后的空间维度
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(N, Dp, Hp, Wp, -1)  # 重排并展平通道和核维度
        x = F.linear(x, self.weight.reshape(self.out_channels, -1), self.bias)  # 使用线性层计算
        return x.permute(0, 4, 1, 2, 3)  # 恢复 (N, C, D, H, W) 格式

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:  # 使用标准 F.conv3d 的前向传播
        if self.padding_mode != "zeros":  # 非零填充模式需要先手动填充
            return F.conv3d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),  # 先手动填充
                self.weight,
                self.bias,
                self.stride,
                (0, 0, 0),  # 填充已在 F.pad 中完成
                self.dilation,
                self.groups,
            )
        return F.conv3d(  # 零填充模式直接使用 F.conv3d
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:  # CPU 前向传播
        if use_intel_amx_backend(self):  # 如果使用 Intel AMX 后端
            return conv3d_embed(  # 使用 AMX 优化的 Conv3d 嵌入算子
                x,
                self.weight,
                self.bias,
                is_vnni=True,  # 使用 VNNI 格式
            )
        return self.forward_native(x)  # 否则使用原生前向传播

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:  # 原生前向传播
        if self.enable_linear:  # 如果启用 linear 优化
            return self._forward_mulmat(x)  # 使用 unfold+linear
        return self._forward_conv(x)  # 否则使用标准卷积

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:  # CUDA 前向传播
        if self.enable_linear:  # 如果启用 linear 优化
            return self._forward_mulmat(x)  # 使用 unfold+linear
        return self._forward_conv(x)  # 否则使用标准卷积
