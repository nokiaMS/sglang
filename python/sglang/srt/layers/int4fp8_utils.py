# INT4/FP8量化通用工具函数
# 本文件提供了Quark量化方案所需的通用工具函数，包括：
# - FP8张量级量化
# - INT4列级量化
# - INT4打包为INT32

"""
Common utilities for quark.
Quark通用工具函数。
"""

import logging  # 导入日志模块 # 导入日志模块
from typing import Tuple  # 导入元组类型 # 导入元组类型

import torch  # 导入PyTorch库 # 导入PyTorch库

logger = logging.getLogger(__name__)  # 创建模块级日志记录器 # 创建模块级日志记录器


# FP8张量级量化函数
def quantize_fp8_scale_tensorwise(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # FP8张量级量化：返回量化后张量和缩放因子
    FP8_MAX = 448.0  # FP8 E4M3的最大可表示值 # FP8 E4M3的最大可表示值
    scale = w.abs().amax().float() / FP8_MAX  # 计算缩放因子 = 绝对值最大值 / FP8_MAX # 计算缩放因子
    scaled = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)  # 缩放并裁剪到FP8范围后转换类型 # 缩放并裁剪到FP8范围后转换类型
    return scaled, scale  # 返回量化张量和缩放因子 # 返回量化张量和缩放因子


# INT4列级量化函数
def quantize_int4_scale_columnwise(  # INT4列级量化：按列计算缩放因子
    w: torch.Tensor,  # 输入权重张量 # 输入权重张量
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化后张量和缩放因子 # 返回量化后张量和缩放因子
    S4_MAX = 7  # INT4的最大可表示值（4位有符号：-8到7，取7） # INT4的最大可表示值
    w_flat = w.reshape(-1, w.shape[-1]).float()  # 将权重展平为2D并转为float # 将权重展平为2D并转为float
    scale = w_flat.abs().amax(axis=-1) / S4_MAX  # 按行计算缩放因子 # 按行计算缩放因子
    scaled = torch.round(w_flat / scale[:, None]).to(torch.int8).clamp(-S4_MAX, S4_MAX)  # 量化、四舍五入、裁剪 # 量化、四舍五入、裁剪
    return scaled.reshape(w.shape), scale.reshape(w.shape[:-1])  # 还原形状并返回 # 还原形状并返回


# INT4打包为INT32函数
def pack_int4_to_int32(to_pack: torch.Tensor, reorder: bool = True) -> torch.Tensor:  # 将INT4值打包为INT32：每8个INT4打包为一个INT32
    if to_pack.ndim > 2:  # 如果维度大于2 # 如果维度大于2
        raise ValueError(  # 抛出值错误
            "Pack: Only supports tensors with dimensions not greater than 2."  # 仅支持维度不超过2的张量 # 仅支持维度不超过2的张量
        )

    if reorder:  # 如果需要重排序 # 如果需要重排序
        order_map = [0, 2, 4, 6, 1, 3, 5, 7]  # 交织重排映射 # 交织重排映射
    else:  # 否则 # 否则
        order_map = [0, 1, 2, 3, 4, 5, 6, 7]  # 顺序映射 # 顺序映射
    pack_num = 8  # 每个INT32打包8个INT4值 # 每个INT32打包8个INT4值
    if to_pack.ndim == 2:  # 如果是2维张量 # 如果是2维张量
        packed = torch.zeros(  # 创建打包后的张量
            to_pack.shape[0],  # 行数不变 # 行数不变
            to_pack.shape[1] // pack_num,  # 列数除以8 # 列数除以8
            dtype=torch.int32,  # INT32数据类型 # INT32数据类型
            device=to_pack.device,  # 设备与输入相同 # 设备与输入相同
        )
        new_c = to_pack.shape[1] // pack_num  # 计算新列数 # 计算新列数
        for c in range(new_c):  # 遍历每列 # 遍历每列
            for i in range(pack_num):  # 遍历每个打包位置 # 遍历每个打包位置
                # Use -3 as an example, high_position is 11111111,cause bit_or generate errors, so we can't use int4 directly
                # 以-3为例，高位为11111111，会导致bit_or出错，因此不能直接使用int4
                packed_col = to_pack[:, c * pack_num + order_map[i]].to(torch.int32)  # 获取重排后的列并转为INT32 # 获取重排后的列并转为INT32
                packed_col = packed_col & 0x0F  # 只保留低4位 # 只保留低4位
                packed[:, c] = torch.bitwise_or(  # 按位或打包到结果中
                    packed[:, c], torch.bitwise_left_shift(packed_col, i * 4)  # 左移i*4位后按位或 # 左移i*4位后按位或
                )
    elif to_pack.ndim == 0:  # 如果是0维标量 # 如果是0维标量
        packed = to_pack.to(torch.int32)  # 直接转换为INT32 # 直接转换为INT32
    else:  # 否则是1维张量 # 否则是1维张量
        packed = torch.zeros(  # 创建打包后的张量
            to_pack.shape[0] // pack_num, dtype=torch.int32, device=to_pack.device  # 长度除以8 # 长度除以8
        )
        new_c = to_pack.shape[0] // pack_num  # 计算新长度 # 计算新长度
        for c in range(new_c):  # 遍历每个位置 # 遍历每个位置
            for i in range(pack_num):  # 遍历每个打包位置 # 遍历每个打包位置
                # Use -3 as an example, high_position is 11111111,cause bit_or generate errors, so we can't use int4 directly
                # 以-3为例，高位为11111111，会导致bit_or出错，因此不能直接使用int4
                packed_col = to_pack[c * pack_num + order_map[i]]  # 获取重排后的元素 # 获取重排后的元素
                packed_col = packed_col & 0x0F  # 只保留低4位 # 只保留低4位
                packed[c] = torch.bitwise_or(  # 按位或打包到结果中
                    packed[c], torch.bitwise_left_shift(packed_col, i * 4)  # 左移i*4位后按位或 # 左移i*4位后按位或
                )

    return packed.view(torch.uint32)  # 以UINT32视图返回 # 以UINT32视图返回
