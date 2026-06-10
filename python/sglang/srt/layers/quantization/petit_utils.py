# Petit NVFP4 量化工具函数模块，提供 NVFP4 权重重打包、缩放处理和线性计算功能
from typing import Optional  # 导入可选类型提示

import torch  # 导入 PyTorch

try:  # 尝试导入 petit_kernel
    from petit_kernel import mul_nvfp4_a16, process_nvfp4_scales, repack_nvfp4  # 导入 Petit 内核函数
except ImportError:  # 如果 petit_kernel 未安装

    def _check_petit_nvfp4_supported(  # 检查 Petit NVFP4 是否受支持（未安装版本）
        quant_method: str, group_size: Optional[int]
    ) -> tuple[bool, Optional[str]]:
        return (  # 返回不支持结果
            False,  # 不支持
            "Petit is not installed. Please install it with `pip install petit-kernel`.",  # 提示安装 Petit
        )

    def prepare_nvfp4_layer_for_petit(layer: torch.nn.Module) -> None:  # 准备 NVFP4 层（未安装版本）
        raise ValueError(  # 抛出值错误
            "Petit is not installed. Please install it with `pip install petit-kernel`."
        )  # 提示安装 Petit

    def apply_petit_nvfp4_linear(  # 应用 Petit NVFP4 线性计算（未安装版本）
        input: torch.Tensor,  # 输入张量
        weight: torch.Tensor,  # 权重张量
        weight_scale: torch.Tensor,  # 权重缩放张量
        weight_scale_2: torch.Tensor,  # 二级权重缩放张量
        size_n: int,  # 输出维度大小
        size_k: int,  # 输入维度大小
        bias: Optional[torch.Tensor] = None,  # 偏置项
    ) -> torch.Tensor:
        raise ValueError(  # 抛出值错误
            "Petit is not installed. Please install it with `pip install petit-kernel`."
        )  # 提示安装 Petit


def _check_petit_nvfp4_supported(  # 检查 Petit NVFP4 是否受支持
    quant_method: str, group_size: Optional[int]
) -> tuple[bool, Optional[str]]:
    if quant_method != "NVFP4":  # 如果量化方法不是 NVFP4
        return (  # 返回不支持结果
            False,  # 不支持
            "Petit currently only supports: NVFP4"
            " quantizations in sglang. Please check the "
            "`hf_quant_config.json` file for your model's "
            "quant configuration.",
        )  # Petit 当前仅支持 NVFP4 量化，请检查模型配置文件
    if group_size is not None and group_size != 16:  # 如果分组大小不为 None 且不等于 16
        return (  # 返回不支持结果
            False,  # 不支持
            "Petit currently only supports: group_size=16" " quantizations.",  # Petit 当前仅支持 group_size=16
        )
    return (True, None)  # 支持通过，返回 True 和无错误信息


def verify_petit_nvfp4_supported(quant_method: str, group_size: Optional[int]) -> None:  # 验证 Petit NVFP4 是否受支持
    supported, error_msg = _check_petit_nvfp4_supported(quant_method, group_size)  # 调用检查函数
    if not supported:  # 如果不支持
        raise ValueError(error_msg)  # 抛出值错误


def prepare_nvfp4_layer_for_petit(layer: torch.nn.Module) -> None:  # 准备 NVFP4 层以适配 Petit 内核格式
    # Repack weights to petit format  # 将权重重打包为 Petit 格式
    part_size_n = layer.output_size_per_partition  # 获取分区输出维度大小
    part_size_k = layer.input_size_per_partition  # 获取分区输入维度大小
    qweight = layer.weight.view(torch.int32).contiguous()  # 将权重视图转为 int32 并确保内存连续
    petit_qweight = repack_nvfp4(qweight, size_n=part_size_n, size_k=part_size_k)  # 调用内核重打包权重
    layer.weight = torch.nn.Parameter(petit_qweight, requires_grad=False)  # 更新权重为不可训练参数

    # Permute scales  # 排列缩放因子
    weight_scale = process_nvfp4_scales(  # 处理 NVFP4 缩放因子
        scales=layer.weight_scale, size_k=part_size_k, size_n=part_size_n  # 传入缩放因子和维度信息
    )
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)  # 更新权重缩放为不可训练参数

    return  # 返回


def apply_petit_nvfp4_linear(  # 应用 Petit NVFP4 线性计算
    input: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    weight_scale: torch.Tensor,  # 权重缩放张量
    weight_scale_2: torch.Tensor,  # 二级权重缩放张量（全局缩放）
    size_n: int,  # 输出维度大小
    size_k: int,  # 输入维度大小
    bias: Optional[torch.Tensor] = None,  # 偏置项
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])  # 将输入重塑为二维张量 [batch*seq, hidden]
    out_shape = input.shape[:-1] + (size_n,)  # 计算输出形状

    # TODO: Use auto-tuning to find the performant solution_id  # TODO: 使用自动调优找到高性能的 solution_id
    output = mul_nvfp4_a16(  # 调用 Petit 内核进行 NVFP4 矩阵乘法
        a=reshaped_x,  # 输入矩阵
        b=weight,  # 权重矩阵
        s=weight_scale,  # 权重缩放因子
        global_scale=weight_scale_2,  # 全局缩放因子
        size_m=reshaped_x.size(0),  # 批次大小
        size_n=size_n,  # 输出维度
        size_k=size_k,  # 输入维度
        solution_id=-1,  # 解决方案 ID，-1 表示自动选择
    )
    if bias is not None:  # 如果有偏置项
        output.add_(bias)  # In-place add  # 原地加偏置

    return output.reshape(out_shape)  # 将输出重塑为目标形状并返回
