# FlashInfer TRT-LLM MoE 运行器模块
# 本模块实现了基于 FlashInfer TRT-LLM 的 MoE（混合专家）前向计算。
# 支持 FP8、MXFP8、FP4 和 BF16 多种量化精度的 MoE 计算。
# 包含权重对齐、缩放因子准备、量化信息数据类和融合专家函数。

from __future__ import annotations  # 启用延迟类型注解求值

from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, cast  # 导入类型提示工具

import torch  # 导入PyTorch深度学习框架
from torch.nn import Module  # 导入神经网络模块基类
from torch.nn.parameter import Parameter  # 导入参数类

# Import to register custom ops for torch.compile compatibility
# 导入以注册自定义操作，兼容 torch.compile
from sglang.srt.distributed import get_tp_group  # 导入TP组获取函数
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    is_symmetric_memory_enabled,  # 导入对称内存启用检查
    is_tensor_in_symmetric_mempool,  # 导入对称内存池检查
    use_symmetric_memory,  # 导入对称内存上下文管理器
)
from sglang.srt.environ import envs  # 导入环境变量模块
from sglang.srt.layers.dp_attention import is_allocation_symmetric  # 导入分配对称性检查函数
from sglang.srt.layers.moe.flashinfer_trtllm_moe import (  # 导入FlashInfer TRT-LLM MoE包装器
    trtllm_fp8_block_scale_moe_wrapper,  # FP8块缩放MoE包装器
    trtllm_fp8_block_scale_routed_moe_wrapper,  # FP8块缩放路由MoE包装器
    trtllm_fp8_per_tensor_scale_moe_wrapper,  # FP8逐张量缩放MoE包装器
)
from sglang.srt.layers.moe.moe_runner.base import (  # 从MoE运行器基类导入
    MoeQuantInfo,  # MoE量化信息基类
    MoeRunnerConfig,  # MoE运行器配置类
    _moe_output_buf,  # MoE输出缓冲区
    register_fused_func,  # 融合函数注册装饰器
)
from sglang.srt.layers.quantization.fp8_kernel import (  # 导入FP8量化内核
    per_token_group_quant_fp8,  # 逐令牌分组FP8量化
    scaled_fp8_quant,  # 缩放FP8量化
)
from sglang.srt.layers.utils import copy_or_rebind_param  # 导入参数复制或重绑定工具
from sglang.srt.utils.common import (  # 导入通用工具函数
    is_cuda_alike,  # 检查是否为类CUDA设备
    is_flashinfer_available,  # 检查FlashInfer是否可用
    next_power_of_2,  # 计算下一个2的幂
)

logger = __import__("logging").getLogger(__name__)  # 创建模块级日志记录器


def round_up_to_multiple(x: int, m: int) -> int:
    """Round up *x* to the nearest multiple of *m*.
    将 *x* 向上取整到 *m* 的最近倍数。
    """
    return (x + m - 1) // m * m  # 计算向上取整到m的倍数


if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher import (  # 从令牌分发器导入
        StandardCombineInput,  # 标准合并输入
        StandardDispatchOutput,  # 标准分发输出
    )

if is_flashinfer_available():  # 如果FlashInfer可用
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize  # 导入FlashInfer版FP4量化
elif is_cuda_alike():  # 否则如果是类CUDA设备
    from sglang.jit_kernel.nvfp4 import scaled_fp4_quant as fp4_quantize  # 导入JIT内核版FP4量化
else:
    fp4_quantize = None  # 不可用时设为None

_flashinfer_trtllm_shuffle_row_indices_cache_mxfp8: dict[
    tuple, dict[str, torch.Tensor]
] = {}  # MXFP8行索引洗牌缓存，避免重复计算


def _is_gated(layer: Module) -> bool:
    """Return whether the MoE layer uses a gated activation (default True).
    返回 MoE 层是否使用门控激活（默认为 True）。
    """
    is_gated = (
        getattr(layer, "moe_runner_config", None) and layer.moe_runner_config.is_gated
    )  # 获取门控激活标志
    return True if is_gated is None else is_gated  # None时默认为True


def _align_fp8_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    is_gated: bool,
    min_alignment: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer TRTLLM FP8 kernels' alignment holds.
    填充中间层大小以满足 FlashInfer TRTLLM FP8 内核的对齐要求。

    Returns (w13, w2, padded_intermediate).
    返回 (w13, w2, padded_intermediate)。
    """
    num_experts, hidden_size, intermediate = w2.shape  # 获取专家数、隐藏大小和中间大小

    padded_intermediate = round_up_to_multiple(intermediate, min_alignment)  # 向上取整到对齐倍数
    if padded_intermediate == intermediate:  # 如果无需填充
        return w13, w2, intermediate  # 直接返回

    logger.info(
        "FP8 MoE: padding intermediate size from %d to %d (alignment=%d)",
        intermediate,
        padded_intermediate,
        min_alignment,
    )  # 记录填充信息

    up_mult = 2 if is_gated else 1  # 门控时乘2（gate+up），否则乘1
    padded_gate_up = up_mult * padded_intermediate  # 计算填充后的gate_up维度

    padded_w13 = w13.new_zeros((num_experts, padded_gate_up, w13.shape[2]))  # 创建填充后的W13
    padded_w13[:, : w13.shape[1], :] = w13  # 复制原始W13数据

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate))  # 创建填充后的W2
    padded_w2[:, :, :intermediate] = w2  # 复制原始W2数据

    return padded_w13, padded_w2, padded_intermediate  # 返回填充后的权重和中间大小


def align_fp8_moe_weights_for_flashinfer_trtllm(
    layer: Module, swap_w13_halves: bool = False
) -> None:
    """Prepare FP8 MoE weights/scales for FlashInfer TRT-LLM kernels.
    为 FlashInfer TRT-LLM 内核准备 FP8 MoE 权重/缩放因子。

    Args:
        layer: The MoE layer to process. / 要处理的MoE层。
        swap_w13_halves: If True, swap W13 halves from [Up, Gate] to [Gate, Up].
            This is needed for ModelOpt FP8 checkpoints which store weights in
            [Up, Gate] order, while regular FP8 checkpoints store them in [Gate, Up].
        swap_w13_halves: 如果为 True，将 W13 两半从 [Up, Gate] 交换为 [Gate, Up]。
            这对于 ModelOpt FP8 检查点是必需的，因为它们以 [Up, Gate] 顺序存储权重，
            而常规 FP8 检查点以 [Gate, Up] 顺序存储。
    """
    from flashinfer import shuffle_matrix_a  # 导入矩阵洗牌函数

    is_gated = _is_gated(layer)  # 检查是否为门控激活

    w13_weight = cast(torch.Tensor, layer.w13_weight)  # 获取W13权重
    w2_weight = cast(torch.Tensor, layer.w2_weight)  # 获取W2权重
    num_experts, gate_up_dim, hidden = w13_weight.shape  # 获取权重形状信息

    # Optionally swap W13 halves: [Up, Gate] -> [Gate, Up] (only for gated)
    # 可选地交换 W13 两半：[Up, Gate] -> [Gate, Up]（仅用于门控）
    if swap_w13_halves and is_gated:  # 如果需要交换且为门控
        inter = gate_up_dim // 2  # 计算中间维度
        w13_weight = (
            w13_weight.reshape(num_experts, 2, inter, hidden)
            .flip(dims=[1])
            .reshape(num_experts, gate_up_dim, hidden)
        )  # 翻转W13的两半

    # Pad for kernel alignment (non-gated needs 128, gated needs 16)
    # 为内核对齐进行填充（非门控需要128，门控需要16）
    min_alignment = 16 if is_gated else 128  # 根据是否门控确定对齐值
    w13_weight, w2_weight, _ = _align_fp8_moe_weights(
        w13_weight, w2_weight, is_gated, min_alignment
    )  # 对齐FP8权重
    num_experts, gate_up_dim, hidden = w13_weight.shape  # 更新形状信息

    epilogue_tile_m = 128  # epilogue tile大小

    if is_gated:  # 如果是门控激活
        from flashinfer import reorder_rows_for_gated_act_gemm  # 导入门控激活行重排函数

        w13_interleaved_list = [
            reorder_rows_for_gated_act_gemm(w13_weight[i]) for i in range(num_experts)
        ]  # 对每个专家的W13权重进行行重排
        w13_processed: torch.Tensor = torch.stack(w13_interleaved_list).reshape(
            num_experts, gate_up_dim, hidden
        )  # 堆叠并重塑重排后的权重
    else:
        w13_processed = w13_weight  # 非门控时不需重排

    # Shuffle weights for transposed MMA output (both W13, W2)
    # 为转置MMA输出洗牌权重（W13和W2都需洗牌）
    w13_shuffled = [
        shuffle_matrix_a(w13_processed[i].view(torch.uint8), epilogue_tile_m)
        for i in range(num_experts)
    ]  # 洗牌W13权重
    w2_shuffled = [
        shuffle_matrix_a(w2_weight[i].view(torch.uint8), epilogue_tile_m)
        for i in range(num_experts)
    ]  # 洗牌W2权重

    layer.w13_weight = Parameter(
        torch.stack(w13_shuffled).view(torch.float8_e4m3fn),
        requires_grad=False,
    )  # 将洗牌后的W13权重设为不可训练参数
    layer.w2_weight = Parameter(
        torch.stack(w2_shuffled).view(torch.float8_e4m3fn),
        requires_grad=False,
    )  # 将洗牌后的W2权重设为不可训练参数

    # Precompute and register per-expert output scaling factors for FI MoE.
    # 预计算并注册 FlashInfer MoE 的每专家输出缩放因子。
    # Note: w13_input_scale and w2_input_scale are scalar Parameters post-reduction.
    # 注意：w13_input_scale 和 w2_input_scale 是归约后的标量参数。
    assert hasattr(layer, "w13_input_scale") and layer.w13_input_scale is not None  # 断言w13输入缩放存在
    assert hasattr(layer, "w2_input_scale") and layer.w2_input_scale is not None  # 断言w2输入缩放存在
    assert hasattr(layer, "w13_weight_scale") and layer.w13_weight_scale is not None  # 断言w13权重缩放存在
    assert hasattr(layer, "w2_weight_scale") and layer.w2_weight_scale is not None  # 断言w2权重缩放存在

    input_scale = cast(torch.Tensor, layer.w13_input_scale).to(torch.float32)  # 获取并转换w13输入缩放
    activation_scale = cast(torch.Tensor, layer.w2_input_scale).to(torch.float32)  # 获取并转换w2输入缩放（激活缩放）
    w13_weight_scale = cast(torch.Tensor, layer.w13_weight_scale).to(torch.float32)  # 获取并转换w13权重缩放
    w2_weight_scale = cast(torch.Tensor, layer.w2_weight_scale).to(torch.float32)  # 获取并转换w2权重缩放

    # For gated (SwiGLU): g1_alphas = w1_scale * a1_scale, g1_scale_c = g1_alphas / a2_scale
    # For non-gated (Relu2): g1_scale_c = 1 / a2_scale (no gate dequant contribution)
    # 门控（SwiGLU）：g1_alphas = w1_scale * a1_scale，g1_scale_c = g1_alphas / a2_scale
    # 非门控（Relu2）：g1_scale_c = 1 / a2_scale（无门控反量化贡献）
    if is_gated:  # 如果是门控激活
        output1_scales_scalar = (
            w13_weight_scale * input_scale * (1.0 / activation_scale)
        )  # 计算门控输出1缩放标量
    else:
        output1_scales_scalar = torch.ones_like(w13_weight_scale) * (
            1.0 / activation_scale
        )  # 非门控时输出1缩放仅含激活缩放倒数
    output1_scales_gate_scalar = w13_weight_scale * input_scale  # 计算输出1门控缩放标量
    output2_scales_scalar = activation_scale * w2_weight_scale  # 计算输出2缩放标量

    layer.output1_scales_scalar = Parameter(output1_scales_scalar, requires_grad=False)  # 注册输出1缩放参数
    layer.output1_scales_gate_scalar = Parameter(
        output1_scales_gate_scalar, requires_grad=False
    )  # 注册输出1门控缩放参数
    layer.output2_scales_scalar = Parameter(output2_scales_scalar, requires_grad=False)  # 注册输出2缩放参数


def _align_mxfp8_moe_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    is_gated: bool,
    min_alignment: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer TRTLLM MXFP8 kernels' alignment holds.
    填充中间层大小以满足 FlashInfer TRTLLM MXFP8 内核的对齐要求。

    Returns (w13, w13_scale, w2, w2_scale, padded_intermediate).
    返回 (w13, w13_scale, w2, w2_scale, padded_intermediate)。
    """
    num_experts, hidden_size, intermediate = w2.shape  # 获取专家数、隐藏大小和中间大小

    padded_intermediate = round_up_to_multiple(intermediate, min_alignment)  # 向上取整到对齐倍数
    if padded_intermediate == intermediate:  # 如果无需填充
        return w13, w13_scale, w2, w2_scale, intermediate  # 直接返回

    logger.info(
        "MXFP8 MoE: padding intermediate size from %d to %d (alignment=%d)",
        intermediate,
        padded_intermediate,
        min_alignment,
    )  # 记录MXFP8填充信息

    up_mult = 2 if is_gated else 1  # 门控时乘2（gate+up），否则乘1
    padded_gate_up = up_mult * padded_intermediate  # 计算填充后的gate_up维度

    padded_w13 = w13.new_zeros((num_experts, padded_gate_up, w13.shape[2]))  # 创建填充后的W13
    padded_w13[:, : w13.shape[1], :] = w13  # 复制原始W13数据

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate))  # 创建填充后的W2
    padded_w2[:, :, :intermediate] = w2  # 复制原始W2数据

    padded_w13_scale = w13_scale.new_zeros(
        (num_experts, padded_gate_up, w13_scale.shape[2])
    )  # 创建填充后的W13缩放因子
    padded_w13_scale[:, : w13_scale.shape[1], :] = w13_scale  # 复制原始W13缩放因子

    # Scale's last dim tracks intermediate / block_size (MXFP8 block_size = 32)
    # 缩放因子的最后一维跟踪 intermediate / block_size（MXFP8 block_size = 32）
    scale_block_k = intermediate // w2_scale.shape[2] if w2_scale.shape[2] > 0 else 32  # 计算缩放块大小
    padded_w2_scale = w2_scale.new_zeros(
        (num_experts, hidden_size, padded_intermediate // scale_block_k)
    )  # 创建填充后的W2缩放因子
    padded_w2_scale[:, :, : w2_scale.shape[2]] = w2_scale  # 复制原始W2缩放因子

    return padded_w13, padded_w13_scale, padded_w2, padded_w2_scale, padded_intermediate  # 返回填充后的结果


def align_mxfp8_moe_weights_for_flashinfer_trtllm(layer: Module) -> None:
    """Prepare MXFP8 MoE weights/scales for FlashInfer TRT-LLM kernels.
    为 FlashInfer TRT-LLM 内核准备 MXFP8 MoE 权重/缩放因子。
    """
    from flashinfer import block_scale_interleave  # 导入块缩放交错函数
    from flashinfer.fused_moe.core import (
        get_reorder_rows_for_gated_act_gemm_row_indices,
    )  # 导入门控激活行重排索引获取函数
    from flashinfer.utils import (  # 导入FlashInfer工具函数
        get_shuffle_matrix_a_row_indices,  # 获取矩阵A行洗牌索引
        get_shuffle_matrix_sf_a_row_indices,  # 获取矩阵A缩放因子行洗牌索引
    )

    is_gated = _is_gated(layer)  # 检查是否为门控激活

    w13_weight = cast(torch.Tensor, layer.w13_weight).contiguous()  # 获取连续的W13权重
    w2_weight = cast(torch.Tensor, layer.w2_weight).contiguous()  # 获取连续的W2权重
    w13_scale = cast(torch.Tensor, layer.w13_weight_scale_inv).contiguous()  # 获取连续的W13缩放因子
    w2_scale = cast(torch.Tensor, layer.w2_weight_scale_inv).contiguous()  # 获取连续的W2缩放因子

    assert w13_scale.dtype == torch.uint8  # 断言W13缩放因子类型为uint8
    assert w2_scale.dtype == torch.uint8  # 断言W2缩放因子类型为uint8

    # Pad for kernel alignment (non-gated needs 128, gated needs 16)
    # 为内核对齐进行填充（非门控需要128，门控需要16）
    min_alignment = 16 if is_gated else 128  # 根据是否门控确定对齐值
    w13_weight, w13_scale, w2_weight, w2_scale, _ = _align_mxfp8_moe_weights(
        w13_weight, w13_scale, w2_weight, w2_scale, is_gated, min_alignment
    )  # 对齐MXFP8权重和缩放因子

    num_experts, gate_up_dim, _ = w13_weight.shape  # 获取专家数和gate_up维度
    _, hidden_size, _ = w2_weight.shape  # 获取隐藏大小
    epilogue_tile_m = 128  # epilogue tile大小

    # Reuse precomputed row-index transforms whenever shape/device are unchanged.
    # 当形状/设备不变时，重用预计算的行索引变换。
    w13_weight_u8 = w13_weight.view(torch.uint8)  # 将W13权重视为uint8视图
    w2_weight_u8 = w2_weight.view(torch.uint8)  # 将W2权重视为uint8视图
    cache_key = (
        gate_up_dim,
        hidden_size,
        w2_weight.shape[-1],
        w13_scale.shape[-1],
        w2_scale.shape[-1],
        epilogue_tile_m,
        (w13_weight.device.type, w13_weight.device.index),
        (w2_weight.device.type, w2_weight.device.index),
        (w13_scale.device.type, w13_scale.device.index),
        (w2_scale.device.type, w2_scale.device.index),
    )  # 构建缓存键
    cache = _flashinfer_trtllm_shuffle_row_indices_cache_mxfp8.get(cache_key)  # 查找缓存
    if cache is None:  # 如果缓存不存在
        if is_gated:  # 如果是门控激活
            reorder_row_indices = get_reorder_rows_for_gated_act_gemm_row_indices(
                w13_weight_u8[0]
            ).to(w13_weight.device)  # 计算门控行重排索引
        else:
            reorder_row_indices = torch.arange(
                gate_up_dim, device=w13_weight.device, dtype=torch.long
            )  # 非门控时使用顺序索引
        w13_shuffle_row_indices = get_shuffle_matrix_a_row_indices(
            w13_weight_u8[0], epilogue_tile_m
        ).to(w13_weight.device)  # 计算W13行洗牌索引
        w2_shuffle_row_indices = get_shuffle_matrix_a_row_indices(
            w2_weight_u8[0], epilogue_tile_m
        ).to(w2_weight.device)  # 计算W2行洗牌索引
        w13_scale_shuffle_row_indices = get_shuffle_matrix_sf_a_row_indices(
            w13_scale[0].reshape(gate_up_dim, -1), epilogue_tile_m
        ).to(w13_scale.device)  # 计算W13缩放因子行洗牌索引
        w2_scale_shuffle_row_indices = get_shuffle_matrix_sf_a_row_indices(
            w2_scale[0].reshape(hidden_size, -1), epilogue_tile_m
        ).to(w2_scale.device)  # 计算W2缩放因子行洗牌索引
        cache = {
            "reorder_row_indices": reorder_row_indices,
            "w13_shuffle_row_indices": w13_shuffle_row_indices,
            "w2_shuffle_row_indices": w2_shuffle_row_indices,
            "w13_scale_shuffle_row_indices": w13_scale_shuffle_row_indices,
            "w2_scale_shuffle_row_indices": w2_scale_shuffle_row_indices,
        }  # 构建缓存字典
        _flashinfer_trtllm_shuffle_row_indices_cache_mxfp8[cache_key] = cache  # 存入缓存

    reorder_row_indices = cache["reorder_row_indices"]  # 获取行重排索引
    w13_shuffle_row_indices = cache["w13_shuffle_row_indices"]  # 获取W13行洗牌索引
    w2_shuffle_row_indices = cache["w2_shuffle_row_indices"]  # 获取W2行洗牌索引
    w13_scale_shuffle_row_indices = cache["w13_scale_shuffle_row_indices"]  # 获取W13缩放因子行洗牌索引
    w2_scale_shuffle_row_indices = cache["w2_scale_shuffle_row_indices"]  # 获取W2缩放因子行洗牌索引

    w13_shuffled_u8 = torch.empty_like(w13_weight_u8)  # 创建W13洗牌后输出缓冲区
    w2_shuffled_u8 = torch.empty_like(w2_weight_u8)  # 创建W2洗牌后输出缓冲区
    w13_scale_shuffled = torch.empty_like(w13_scale)  # 创建W13缩放因子洗牌后输出缓冲区
    w2_scale_shuffled = torch.empty_like(w2_scale)  # 创建W2缩放因子洗牌后输出缓冲区

    for i in range(num_experts):  # 遍历每个专家
        w13_interleaved_u8 = w13_weight_u8[i].index_select(0, reorder_row_indices)  # 对W13进行行重排
        w13_scale_interleaved = w13_scale[i].index_select(0, reorder_row_indices)  # 对W13缩放因子进行行重排

        w13_shuffled_u8[i].copy_(
            w13_interleaved_u8.index_select(0, w13_shuffle_row_indices)
        )  # 对W13进行行洗牌
        w2_shuffled_u8[i].copy_(w2_weight_u8[i].index_select(0, w2_shuffle_row_indices))  # 对W2进行行洗牌

        w13_scale_linear = w13_scale_interleaved.reshape(gate_up_dim, -1)  # 将W13缩放因子重塑为2D
        w13_scale_shuffled[i].copy_(
            block_scale_interleave(
                w13_scale_linear.index_select(0, w13_scale_shuffle_row_indices)
            ).reshape_as(w13_scale_shuffled[i])
        )  # 对W13缩放因子进行行洗牌和块交错

        w2_scale_linear = w2_scale[i].reshape(hidden_size, -1)  # 将W2缩放因子重塑为2D
        w2_scale_shuffled[i].copy_(
            block_scale_interleave(
                w2_scale_linear.index_select(0, w2_scale_shuffle_row_indices)
            ).reshape_as(w2_scale_shuffled[i])
        )  # 对W2缩放因子进行行洗牌和块交错

    # Keep parameter identities stable for CUDA graph capture reuse.
    # 保持参数标识稳定，以便CUDA图捕获重用。
    copy_or_rebind_param(layer, "w13_weight", w13_shuffled_u8.view(torch.float8_e4m3fn))  # 更新W13权重参数
    copy_or_rebind_param(layer, "w2_weight", w2_shuffled_u8.view(torch.float8_e4m3fn))  # 更新W2权重参数
    copy_or_rebind_param(
        layer,
        "w13_weight_scale_inv",
        w13_scale_shuffled.contiguous(),
    )  # 更新W13缩放因子参数
    copy_or_rebind_param(
        layer,
        "w2_weight_scale_inv",
        w2_scale_shuffled.contiguous(),
    )  # 更新W2缩放因子参数
    layer.w13_weight_scale_inv.format_ue8m0 = True  # 标记W13缩放因子为UE8M0格式
    layer.w2_weight_scale_inv.format_ue8m0 = True  # 标记W2缩放因子为UE8M0格式


def _align_fp4_moe_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    is_gated: bool,
    min_alignment: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer TRTLLM FP4 kernels' alignment holds.
    填充中间层大小以满足 FlashInfer TRTLLM FP4 内核的对齐要求。

    Returns (w13, w13_scale, w2, w2_scale, padded_intermediate).
    返回 (w13, w13_scale, w2, w2_scale, padded_intermediate)。
    """
    num_experts, hidden_size, intermediate_packed = w2.shape  # 获取专家数、隐藏大小和打包中间大小
    intermediate = intermediate_packed * 2  # FP4 packs 2 values per byte / FP4每个字节打包2个值

    padded_intermediate = round_up_to_multiple(intermediate, min_alignment)  # 向上取整到对齐倍数
    if padded_intermediate == intermediate:  # 如果无需填充
        return w13, w13_scale, w2, w2_scale, intermediate  # 直接返回

    logger.info(
        "FP4 MoE: padding intermediate size from %d to %d (alignment=%d)",
        intermediate,
        padded_intermediate,
        min_alignment,
    )  # 记录FP4填充信息

    up_mult = 2 if is_gated else 1  # 门控时乘2（gate+up），否则乘1
    padded_gate_up = up_mult * padded_intermediate  # 计算填充后的gate_up维度

    padded_w13 = w13.new_zeros((num_experts, padded_gate_up, w13.shape[2]))  # 创建填充后的W13
    padded_w13[:, : w13.shape[1], :] = w13  # 复制原始W13数据

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate // 2))  # 创建填充后的W2（FP4打包）
    padded_w2[:, :, : w2.shape[2]] = w2  # 复制原始W2数据

    padded_w13_scale = w13_scale.new_zeros(
        (num_experts, padded_gate_up, w13_scale.shape[2])
    )  # 创建填充后的W13缩放因子
    padded_w13_scale[:, : w13_scale.shape[1], :] = w13_scale  # 复制原始W13缩放因子

    padded_w2_scale = w2_scale.new_zeros(
        (num_experts, hidden_size, padded_intermediate // 16)
    )  # 创建填充后的W2缩放因子
    padded_w2_scale[:, :, : w2_scale.shape[2]] = w2_scale  # 复制原始W2缩放因子

    return padded_w13, padded_w13_scale, padded_w2, padded_w2_scale, padded_intermediate  # 返回填充后的结果


def align_fp4_moe_weights_for_flashinfer_trtllm(layer: Module) -> None:
    """Prepare FP4 MoE weights/scales for FlashInfer TRT-LLM kernels.
    为 FlashInfer TRT-LLM 内核准备 FP4 MoE 权重/缩放因子。

    This function handles the weight transformation needed for FP4 TRTLLM MoE:
    本函数处理 FP4 TRTLLM MoE 所需的权重变换：
    - Pads intermediate dimension for kernel alignment constraints
    - 填充中间维度以满足内核对齐约束
    - Reorders weights for gated activation GEMM
    - 为门控激活 GEMM 重排权重
    - Shuffles weights and scales for transposed MMA output
    - 为转置 MMA 输出洗牌权重和缩放因子
    - Computes the output scale factors
    - 计算输出缩放因子
    """
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )  # 导入FP4 MoE静态权重准备函数

    w13_weight = cast(torch.Tensor, layer.w13_weight)  # 获取W13权重
    w2_weight = cast(torch.Tensor, layer.w2_weight)  # 获取W2权重
    w13_weight_scale = cast(torch.Tensor, layer.w13_weight_scale)  # 获取W13权重缩放因子
    w2_weight_scale = cast(torch.Tensor, layer.w2_weight_scale)  # 获取W2权重缩放因子

    is_gated = layer.moe_runner_config.is_gated  # 获取门控标志
    min_alignment = 16 if is_gated else 128  # 根据是否门控确定对齐值

    # Pad for kernel alignment before shuffle/reorder
    # 在洗牌/重排前为内核对齐进行填充
    w13_weight, w13_weight_scale, w2_weight, w2_weight_scale, intermediate_size = (
        _align_fp4_moe_weights(
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
            is_gated,
            min_alignment,
        )
    )  # 对齐FP4权重和缩放因子

    (
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
    ) = prepare_static_weights_for_trtllm_fp4_moe(
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w2_weight.size(-2),  # hidden_size / 隐藏大小
        intermediate_size,  # padded intermediate_size / 填充后的中间大小
        w13_weight.size(0),  # num_experts / 专家数
        is_gated=is_gated,
    )  # 准备FP4 MoE静态权重

    # Set flashinfer parameters in-place
    # 就地设置FlashInfer参数
    copy_or_rebind_param(layer, "w13_weight", gemm1_weights_fp4_shuffled.contiguous())  # 更新W13权重
    copy_or_rebind_param(layer, "w2_weight", gemm2_weights_fp4_shuffled.contiguous())  # 更新W2权重
    copy_or_rebind_param(
        layer, "w13_weight_scale", gemm1_scales_fp4_shuffled.contiguous()
    )  # 更新W13权重缩放因子
    copy_or_rebind_param(
        layer, "w2_weight_scale", gemm2_scales_fp4_shuffled.contiguous()
    )  # 更新W2权重缩放因子

    # Compute additional scaling factor needed for TRT-LLM.
    # 计算TRT-LLM所需的额外缩放因子。
    # For gated (SwiGLU): g1_scale_c = g1_alphas * a2_gscale
    # For non-gated (Relu2): g1_scale_c = a2_gscale (no gate dequant contribution)
    # 门控（SwiGLU）：g1_scale_c = g1_alphas * a2_gscale
    # 非门控（Relu2）：g1_scale_c = a2_gscale（无门控反量化贡献）
    w2_input_scale_quant = cast(torch.Tensor, layer.w2_input_scale_quant)  # 获取W2输入缩放量化因子
    g1_alphas = cast(torch.Tensor, layer.g1_alphas)  # 获取GEMM1 alpha
    if layer.moe_runner_config.is_gated:  # 如果是门控激活
        g1_scale_c = (w2_input_scale_quant * g1_alphas).to(torch.float32)  # 计算门控GEMM1缩放因子
    else:
        num_experts = g1_alphas.shape[0]  # 获取专家数
        g1_scale_c = (
            w2_input_scale_quant.to(torch.float32).expand(num_experts).contiguous()
        )  # 非门控时仅使用W2输入缩放量化因子
    copy_or_rebind_param(layer, "g1_scale_c", g1_scale_c)  # 注册GEMM1缩放因子参数

    # Update intermediate_size_per_partition to reflect any padding applied
    # 更新 intermediate_size_per_partition 以反映应用的填充
    layer.intermediate_size_per_partition = intermediate_size  # 更新中间层分区大小


def get_activation_type(activation: str, is_gated: bool = True) -> int:
    """Map SGLang activation string to FlashInfer ActivationType int value.
    将 SGLang 激活字符串映射为 FlashInfer ActivationType 整数值。
    """
    from flashinfer.fused_moe.core import ActivationType  # 导入激活类型枚举

    if is_gated:  # 如果是门控激活
        _ACTIVATION_STR_TO_TYPE = {
            "silu": ActivationType.Swiglu,  # silu映射为Swiglu
            "gelu": ActivationType.Geglu,  # gelu映射为Geglu
        }
    else:  # 如果不是门控激活
        _ACTIVATION_STR_TO_TYPE = {
            "silu": ActivationType.Silu,  # silu映射为Silu
            "gelu": ActivationType.Gelu,  # gelu映射为Gelu
            "relu2": ActivationType.Relu2,  # relu2映射为Relu2
        }
    act = _ACTIVATION_STR_TO_TYPE.get(activation)  # 查找激活类型
    if act is None:  # 如果未找到
        raise ValueError(
            f"Unsupported activation '{activation}' for TRTLLM MoE "
            f"(is_gated={is_gated}). "
            f"Expected one of {list(_ACTIVATION_STR_TO_TYPE.keys())}."
            f"不支持的激活类型 '{activation}' 用于 TRTLLM MoE "
            f"(is_gated={is_gated})。"
            f"期望以下之一: {list(_ACTIVATION_STR_TO_TYPE.keys())}。"
        )
    return act.value  # 返回激活类型的整数值


@dataclass
class FlashInferTrtllmFp8MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer TRT-LLM FP8 MoE kernels.
    FlashInfer TRT-LLM FP8 MoE 内核消费的量化负载。
    """

    # Weights / 权重
    w13_weight: torch.Tensor  # W13权重
    w2_weight: torch.Tensor  # W2权重

    # Expert-parallel metadata / 专家并行元数据
    global_num_experts: int  # 全局专家数
    local_expert_offset: int  # 本地专家偏移
    local_num_experts: int  # 本地专家数
    intermediate_size: int  # 中间层大小

    routing_method_type: int  # 路由方法类型

    # Block-quant path / 块量化路径
    block_quant: bool  # 是否使用块量化
    use_mxfp8: bool = False  # 是否使用MXFP8
    weight_block_k: int | None = None  # 权重块K维度大小
    w13_weight_scale_inv: torch.Tensor | None = None  # W13权重缩放因子倒数
    w2_weight_scale_inv: torch.Tensor | None = None  # W2权重缩放因子倒数

    # Per-tensor path / 逐张量路径
    w13_input_scale: torch.Tensor | None = None  # W13输入缩放因子
    output1_scales_scalar: torch.Tensor | None = None  # 输出1缩放标量
    output1_scales_gate_scalar: torch.Tensor | None = None  # 输出1门控缩放标量
    output2_scales_scalar: torch.Tensor | None = None  # 输出2缩放标量
    use_routing_scales_on_input: bool = False  # 是否在输入上使用路由缩放

    # Activation type (None = kernel default / Swiglu) / 激活类型（None = 内核默认 / Swiglu）
    activation_type: int | None = None  # 激活类型


def _pack_topk_for_flashinfer_routed(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """Pack routed top-k tensors into FlashInfer's int32 format.
    将路由 top-k 张量打包为 FlashInfer 的 int32 格式。
    """
    packed_ids = topk_ids.to(torch.int32)  # 将TopK ID转为int32
    packed_weights = topk_weights.to(torch.bfloat16)  # 将TopK权重转为bfloat16
    packed = (packed_ids << 16) | packed_weights.view(torch.int16).to(torch.int32)  # 将ID和权重打包为一个int32
    return packed  # 返回打包结果


def fused_experts_none_to_flashinfer_trtllm_fp8(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmFp8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    use_routed_topk: bool = False,
) -> StandardCombineInput:
    """FlashInfer TRT-LLM FP8 MoE融合专家前向函数"""
    from flashinfer.fused_moe import Fp8QuantizationType  # 导入FP8量化类型

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput  # 导入标准合并输入
    from sglang.srt.layers.moe.topk import TopKOutputChecker  # 导入TopK输出检查器
    from sglang.srt.layers.moe.utils import RoutingMethodType  # 导入路由方法类型

    _SUPPORTED_FP8_ACTIVATIONS = {"silu", "relu2"}  # FP8支持的激活类型
    assert runner_config.activation in _SUPPORTED_FP8_ACTIVATIONS, (
        f"Only {_SUPPORTED_FP8_ACTIVATIONS} are supported for FP8 MoE, "
        f"got '{runner_config.activation}'."
    )  # 断言激活类型受支持
    assert not runner_config.no_combine, "no_combine is not supported for flashinfer."  # 断言不支持no_combine

    hidden_states = dispatch_output.hidden_states  # 获取隐藏状态
    topk_output = dispatch_output.topk_output  # 获取TopK输出
    if TopKOutputChecker.format_is_bypassed(topk_output):  # 如果TopK输出为绕过格式
        router_logits = topk_output.router_logits  # 获取路由logits
        topk_config = topk_output.topk_config  # 获取TopK配置
        correction_bias = (
            None
            if topk_config.correction_bias is None
            else topk_config.correction_bias.to(hidden_states.dtype)
        )  # 获取校正偏置
    else:
        router_logits = None  # 非绕过格式时路由logits为None
        topk_config = None  # 非绕过格式时TopK配置为None
        correction_bias = None  # 非绕过格式时校正偏置为None

    routing_method_type = quant_info.routing_method_type  # 获取路由方法类型
    fp8_quantization_type = (
        Fp8QuantizationType.MxFp8
        if quant_info.use_mxfp8
        else Fp8QuantizationType.DeepSeekFp8
    )  # 确定FP8量化类型
    use_shuffled_weight = quant_info.use_mxfp8  # 是否使用洗牌权重

    if quant_info.block_quant:  # 如果使用块量化
        assert quant_info.weight_block_k is not None  # 断言权重块K维度存在
        assert quant_info.w13_weight_scale_inv is not None  # 断言W13权重缩放因子存在
        assert quant_info.w2_weight_scale_inv is not None  # 断言W2权重缩放因子存在

        if quant_info.use_mxfp8:  # 如果使用MXFP8
            assert quant_info.weight_block_k == 32  # 断言MXFP8块大小为32
            from flashinfer import mxfp8_quantize  # 导入MXFP8量化函数

            a_q, a_sf = mxfp8_quantize(hidden_states, False)  # 对隐藏状态进行MXFP8量化
            # FlashInfer TRT-LLM MxFP8 expects token-major activation scales:
            # [num_tokens, hidden_size // 32] (no transpose).
            # FlashInfer TRT-LLM MxFP8 期望令牌主序的激活缩放因子：
            # [num_tokens, hidden_size // 32]（不转置）。
            a_sf_t = a_sf.view(torch.uint8).reshape(hidden_states.shape[0], -1)  # 重塑缩放因子为令牌主序
        else:
            a_q, a_sf = per_token_group_quant_fp8(
                hidden_states, quant_info.weight_block_k
            )  # 逐令牌分组FP8量化
            a_sf_t = a_sf.t().contiguous()  # 转置缩放因子

        # Allocate output inside symmetric memory context
        # 在对称内存上下文中分配输出
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            symm_output = torch.empty(
                hidden_states.shape[0],
                hidden_states.shape[1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )  # 分配对称内存输出

        # Move kernel call outside context manager to avoid graph breaks
        # during torch.compile for piecewise cuda graph.
        # Use custom op wrapper for torch.compile compatibility.
        # 将内核调用移出上下文管理器，以避免
        # torch.compile 分段CUDA图时的图中断。
        # 使用自定义操作包装器兼容 torch.compile。
        if use_routed_topk:  # 如果使用路由TopK
            assert (
                runner_config.top_k is not None
            ), "runner_config.top_k is required for flashinfer_trtllm_routed."  # 断言top_k必须存在
            assert TopKOutputChecker.format_is_standard(topk_output)  # 断言TopK输出为标准格式
            packed_topk_ids = _pack_topk_for_flashinfer_routed(
                topk_ids=topk_output.topk_ids,
                topk_weights=topk_output.topk_weights,
            )  # 打包路由TopK

            output = trtllm_fp8_block_scale_routed_moe_wrapper(
                topk_ids=packed_topk_ids,  # 打包的TopK ID
                routing_bias=None,  # 路由偏置
                hidden_states=a_q,  # 量化后的隐藏状态
                hidden_states_scale=a_sf_t,  # 隐藏状态缩放因子
                gemm1_weights=quant_info.w13_weight,  # GEMM1权重
                gemm1_weights_scale=quant_info.w13_weight_scale_inv,  # GEMM1权重缩放因子
                gemm2_weights=quant_info.w2_weight,  # GEMM2权重
                gemm2_weights_scale=quant_info.w2_weight_scale_inv,  # GEMM2权重缩放因子
                num_experts=quant_info.global_num_experts,  # 全局专家数
                top_k=runner_config.top_k,  # Top-K值
                n_group=None,  # 分组数
                topk_group=None,  # Top-K分组数
                intermediate_size=quant_info.intermediate_size,  # 中间层大小
                local_expert_offset=quant_info.local_expert_offset,  # 本地专家偏移
                local_num_experts=quant_info.local_num_experts,  # 本地专家数
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),  # 路由缩放因子
                routing_method_type=(
                    RoutingMethodType.TopK
                    if routing_method_type == RoutingMethodType.DeepSeekV3
                    else routing_method_type
                ),  # 路由方法类型
                use_shuffled_weight=use_shuffled_weight,  # 是否使用洗牌权重
                tune_max_num_tokens=next_power_of_2(a_q.shape[0]),  # 自适应调整最大令牌数
                fp8_quantization_type=int(fp8_quantization_type),  # FP8量化类型
                activation_type=quant_info.activation_type,  # 激活类型
            )  # 调用FP8块缩放路由MoE包装器
        else:
            assert TopKOutputChecker.format_is_bypassed(topk_output)  # 断言TopK输出为绕过格式

            output = trtllm_fp8_block_scale_moe_wrapper(
                routing_logits=router_logits,  # 路由logits
                routing_bias=correction_bias,  # 路由偏置
                hidden_states=a_q,  # 量化后的隐藏状态
                hidden_states_scale=a_sf_t,  # 隐藏状态缩放因子
                gemm1_weights=quant_info.w13_weight,  # GEMM1权重
                gemm1_weights_scale=quant_info.w13_weight_scale_inv,  # GEMM1权重缩放因子
                gemm2_weights=quant_info.w2_weight,  # GEMM2权重
                gemm2_weights_scale=quant_info.w2_weight_scale_inv,  # GEMM2权重缩放因子
                num_experts=quant_info.global_num_experts,  # 全局专家数
                top_k=topk_config.top_k,  # Top-K值
                n_group=topk_config.num_expert_group,  # 专家分组数
                topk_group=topk_config.topk_group,  # Top-K分组数
                intermediate_size=quant_info.intermediate_size,  # 中间层大小
                local_expert_offset=quant_info.local_expert_offset,  # 本地专家偏移
                local_num_experts=quant_info.local_num_experts,  # 本地专家数
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),  # 路由缩放因子
                routing_method_type=routing_method_type,  # 路由方法类型
                use_shuffled_weight=use_shuffled_weight,  # 是否使用洗牌权重
                tune_max_num_tokens=next_power_of_2(a_q.shape[0]),  # 自适应调整最大令牌数
                fp8_quantization_type=int(fp8_quantization_type),  # FP8量化类型
                activation_type=quant_info.activation_type,  # 激活类型
            )  # 调用FP8块缩放MoE包装器
        # TODO: Once https://github.com/flashinfer-ai/flashinfer/issues/2703 is fixed, pass output to moe kernel and remove this copy.
        # TODO: 一旦 https://github.com/flashinfer-ai/flashinfer/issues/2703 修复，将输出传递给moe内核并删除此拷贝。
        symm_output.copy_(output)  # 将输出拷贝到对称内存
        output = symm_output  # 使用对称内存输出
    else:  # 非块量化路径
        assert TopKOutputChecker.format_is_bypassed(topk_output)  # 断言TopK输出为绕过格式
        assert quant_info.w13_input_scale is not None  # 断言W13输入缩放存在
        assert quant_info.output1_scales_scalar is not None  # 断言输出1缩放标量存在
        assert quant_info.output1_scales_gate_scalar is not None  # 断言输出1门控缩放标量存在
        assert quant_info.output2_scales_scalar is not None  # 断言输出2缩放标量存在

        a_q, _ = scaled_fp8_quant(hidden_states, quant_info.w13_input_scale)  # 逐张量FP8量化
        routing_bias_cast = (
            None if correction_bias is None else correction_bias.to(torch.bfloat16)
        )  # 转换路由偏置类型

        # Allocate output inside symmetric memory context
        # 在对称内存上下文中分配输出
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            symm_output = torch.empty(
                hidden_states.shape[0],
                hidden_states.shape[1],
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )  # 分配对称内存输出

        # Move kernel call outside context manager to avoid graph breaks
        # during torch.compile for piecewise cuda graph.
        # Use custom op wrapper for torch.compile compatibility.
        # 将内核调用移出上下文管理器，以避免
        # torch.compile 分段CUDA图时的图中断。
        # 使用自定义操作包装器兼容 torch.compile。

        router_logits = router_logits.to(torch.bfloat16)  # 转换路由logits为bfloat16

        output = trtllm_fp8_per_tensor_scale_moe_wrapper(
            routing_logits=router_logits,  # 路由logits
            routing_bias=routing_bias_cast,  # 路由偏置
            hidden_states=a_q,  # 量化后的隐藏状态
            gemm1_weights=quant_info.w13_weight,  # GEMM1权重
            output1_scales_scalar=quant_info.output1_scales_scalar,  # 输出1缩放标量
            output1_scales_gate_scalar=quant_info.output1_scales_gate_scalar,  # 输出1门控缩放标量
            gemm2_weights=quant_info.w2_weight,  # GEMM2权重
            output2_scales_scalar=quant_info.output2_scales_scalar,  # 输出2缩放标量
            num_experts=quant_info.global_num_experts,  # 全局专家数
            top_k=topk_config.top_k,  # Top-K值
            n_group=topk_config.num_expert_group,  # 专家分组数
            topk_group=topk_config.topk_group,  # Top-K分组数
            intermediate_size=int(quant_info.w2_weight.shape[2]),  # 中间层大小
            local_expert_offset=quant_info.local_expert_offset,  # 本地专家偏移
            local_num_experts=quant_info.local_num_experts,  # 本地专家数
            routed_scaling_factor=(
                runner_config.routed_scaling_factor
                if runner_config.routed_scaling_factor is not None
                else 1.0
            ),  # 路由缩放因子
            use_routing_scales_on_input=False,  # 不在输入上使用路由缩放
            routing_method_type=routing_method_type,  # 路由方法类型
            tune_max_num_tokens=next_power_of_2(a_q.shape[0]),  # 自适应调整最大令牌数
            activation_type=quant_info.activation_type,  # 激活类型
        )  # 调用FP8逐张量缩放MoE包装器
        symm_output.copy_(output)  # 将输出拷贝到对称内存
        output = symm_output  # 使用对称内存输出

    return StandardCombineInput(hidden_states=output)  # 返回标准合并输入


@dataclass
class FlashInferTrtllmFp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer TRT-LLM FP4 MoE kernels.
    FlashInfer TRT-LLM FP4 MoE 内核消费的量化负载。
    """

    w13_weight: torch.Tensor  # W13权重
    w2_weight: torch.Tensor  # W2权重
    w13_weight_scale: torch.Tensor  # W13权重缩放因子
    w2_weight_scale: torch.Tensor  # W2权重缩放因子

    # Scaling factors / 缩放因子
    g1_scale_c: torch.Tensor  # GEMM1缩放因子C
    g1_alphas: torch.Tensor  # GEMM1 alpha
    g2_alphas: torch.Tensor  # GEMM2 alpha
    w13_input_scale_quant: torch.Tensor  # W13输入缩放量化因子

    # Expert-parallel metadata / 专家并行元数据
    global_num_experts: int  # 全局专家数
    local_expert_offset: int  # 本地专家偏移
    local_num_experts: int  # 本地专家数
    intermediate_size_per_partition: int  # 分区中间层大小

    routing_method_type: int  # 路由方法类型


def quantize_hidden_states_fp4(
    hidden_states: torch.Tensor,
    input_scale_quant: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize hidden states to FP4 for TRTLLM MoE.
    将隐藏状态量化为 FP4 用于 TRTLLM MoE。

    Global scale factor is set by ModelOptNvFp4FusedMoEMethod during weight loading.
    全局缩放因子由 ModelOptNvFp4FusedMoEMethod 在权重加载期间设置。
    Only block scales are computed at runtime for efficiency.
    为提高效率，运行时仅计算块缩放因子。

    Returns (packed_fp4_uint8, scale_float8_e4m3fn_runtime)
    返回 (packed_fp4_uint8, scale_float8_e4m3fn_runtime)
    """

    # flashinfer.fp4_quantize returns (packed_uint8, scale_fp8)
    # Only the block scales are computed at runtime
    # flashinfer.fp4_quantize 返回 (packed_uint8, scale_fp8)
    # 运行时仅计算块缩放因子
    hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
        hidden_states,
        input_scale_quant,
        16,  # sf_vec_size / 缩放因子向量大小
        False,  # use_ue8m0 / 不使用UE8M0
        False,  # is_sf_swizzled_layout / 缩放因子非swizzled布局
    )  # 对隐藏状态进行FP4量化

    seq_len, hidden_size = hidden_states.shape  # 获取序列长度和隐藏大小
    hs_fp4 = hs_fp4_bytes.reshape(seq_len, hidden_size // 2)  # 重塑FP4字节为半宽张量
    # TRT-LLM expects hidden state scales shaped as [seq_len, hidden_size // 16]
    # TRT-LLM 期望隐藏状态缩放因子形状为 [seq_len, hidden_size // 16]
    hs_sf = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(seq_len, hidden_size // 16)  # 重塑缩放因子

    return hs_fp4, hs_sf  # 返回FP4张量和缩放因子


def fused_experts_none_to_flashinfer_trtllm_fp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    use_routed_topk: bool = False,
) -> StandardCombineInput:
    """FlashInfer TRTLLM FP4 MoE forward pass.
    FlashInfer TRTLLM FP4 MoE 前向传播。

    This function handles the FP4 TRTLLM MoE path that was previously in
    ModelOptNvFp4FusedMoEMethod.apply.
    本函数处理之前在 ModelOptNvFp4FusedMoEMethod.apply 中的 FP4 TRTLLM MoE 路径。
    """
    from flashinfer.fused_moe import (  # 导入FlashInfer FP4 MoE函数
        trtllm_fp4_block_scale_moe,  # FP4块缩放MoE
        trtllm_fp4_block_scale_routed_moe,  # FP4块缩放路由MoE
    )

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput  # 导入标准合并输入
    from sglang.srt.layers.moe.topk import TopKOutputChecker  # 导入TopK输出检查器
    from sglang.srt.layers.moe.utils import RoutingMethodType  # 导入路由方法类型

    _SUPPORTED_FP4_ACTIVATIONS = {"silu", "relu2", "gelu"}  # FP4支持的激活类型
    assert runner_config.activation in _SUPPORTED_FP4_ACTIVATIONS, (
        f"Only {_SUPPORTED_FP4_ACTIVATIONS} are supported for FP4 MoE, "
        f"got '{runner_config.activation}'."
    )  # 断言激活类型受支持

    hidden_states = dispatch_output.hidden_states  # 获取隐藏状态
    topk_output = dispatch_output.topk_output  # 获取TopK输出

    # Quantize hidden states to FP4
    # 将隐藏状态量化为FP4
    if envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.get():  # 如果启用逐令牌NVFP4激活
        from flashinfer import SfLayout, nvfp4_quantize  # 导入NVFP4量化函数

        hs_fp4_bytes, hs_sf_bytes, per_token_scale = nvfp4_quantize(
            hidden_states,
            1.0 / (448.0 * 6.0),  # NVFP4全局缩放因子
            sfLayout=SfLayout.layout_linear,  # 缩放因子布局为线性
            per_token_activation=True,  # 启用逐令牌激活
        )  # 使用NVFP4量化

        seq_len, hidden_size = hidden_states.shape  # 获取序列长度和隐藏大小
        hs_fp4 = hs_fp4_bytes.reshape(seq_len, hidden_size // 2)  # 重塑FP4字节
        hs_scale_linear = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(
            seq_len, hidden_size // 16
        )  # 重塑缩放因子
    else:
        per_token_scale = None  # 不使用逐令牌缩放
        hs_fp4, hs_scale_linear = quantize_hidden_states_fp4(
            hidden_states, quant_info.w13_input_scale_quant
        )  # 使用标准FP4量化
    hs_scale = hs_scale_linear.view(torch.float8_e4m3fn).reshape(
        *hs_scale_linear.shape[:-1], -1
    )  # 重塑缩放因子
    activation_type = get_activation_type(
        runner_config.activation, is_gated=runner_config.is_gated
    )  # 获取激活类型

    # Build per-expert clamp-limit tensor from the per-layer scalar.
    # 从每层标量构建每专家钳位限制张量。
    _clamp_val = runner_config.gemm1_clamp_limit  # 获取GEMM1钳位限制值
    if _clamp_val is not None:  # 如果钳位限制值存在
        gemm1_clamp_limit = torch.full(
            (quant_info.local_num_experts,),
            _clamp_val,
            dtype=torch.float32,
            device=hs_fp4.device,
        )  # 创建每专家钳位限制张量
    else:
        gemm1_clamp_limit = None  # 不使用钳位限制

    num_tokens = hs_fp4.shape[0]  # 获取令牌数
    hidden_size = (
        hs_fp4.shape[-1] * 2 if hs_fp4.dtype == torch.uint8 else hs_fp4.shape[-1]
    )  # 计算实际隐藏维度（FP4打包时乘2）
    _provided = _moe_output_buf.get()  # 获取预分配输出缓冲区
    _symm_required = is_allocation_symmetric()  # 检查是否需要对称分配
    if (
        _provided is not None
        and _provided.shape == (num_tokens, hidden_size)
        and _provided.dtype == hidden_states.dtype
        and _provided.device == hs_fp4.device
        and (
            not _symm_required
            or not is_symmetric_memory_enabled()
            or is_tensor_in_symmetric_mempool(_provided)
        )
    ):  # 如果预分配缓冲区可用且满足要求
        symm_output = _provided  # 使用预分配缓冲区
    else:
        with use_symmetric_memory(get_tp_group(), disabled=not _symm_required):  # 在对称内存上下文中
            symm_output = torch.empty(
                num_tokens, hidden_size, dtype=hidden_states.dtype, device=hs_fp4.device
            )  # 分配新的输出缓冲区

    # Fall back to routed path when topk was already materialized (e.g. sigmoid routing).
    # 当 topk 已经具体化时（如 sigmoid 路由），回退到路由路径。
    if not use_routed_topk and TopKOutputChecker.format_is_standard(topk_output):  # 如果非路由TopK且TopK为标准格式
        use_routed_topk = True  # 切换到路由TopK路径

    if use_routed_topk:  # 如果使用路由TopK路径
        assert TopKOutputChecker.format_is_standard(topk_output)  # 断言TopK输出为标准格式

        packed_topk_ids = _pack_topk_for_flashinfer_routed(
            topk_output.topk_ids, topk_output.topk_weights
        )  # 打包路由TopK
        result = trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_topk_ids,  # 打包的TopK ID
            routing_bias=None,  # 路由偏置
            hidden_states=hs_fp4,  # FP4隐藏状态
            hidden_states_scale=hs_scale,  # 隐藏状态缩放因子
            gemm1_weights=quant_info.w13_weight,  # GEMM1权重
            gemm1_weights_scale=quant_info.w13_weight_scale.view(torch.float8_e4m3fn),  # GEMM1权重缩放因子
            gemm1_bias=None,  # GEMM1偏置
            gemm1_alpha=None,  # GEMM1 alpha
            gemm1_beta=None,  # GEMM1 beta
            gemm1_clamp_limit=gemm1_clamp_limit,  # GEMM1钳位限制
            gemm2_weights=quant_info.w2_weight,  # GEMM2权重
            gemm2_weights_scale=quant_info.w2_weight_scale.view(torch.float8_e4m3fn),  # GEMM2权重缩放因子
            gemm2_bias=None,  # GEMM2偏置
            output1_scale_scalar=quant_info.g1_scale_c,  # 输出1缩放标量
            output1_scale_gate_scalar=quant_info.g1_alphas,  # 输出1门控缩放标量
            output2_scale_scalar=quant_info.g2_alphas,  # 输出2缩放标量
            per_token_scale=per_token_scale,  # 逐令牌缩放因子
            num_experts=quant_info.global_num_experts,  # 全局专家数
            top_k=topk_output.topk_ids.shape[1],  # Top-K值
            n_group=0,  # 分组数
            topk_group=0,  # Top-K分组数
            intermediate_size=quant_info.intermediate_size_per_partition,  # 分区中间层大小
            local_expert_offset=quant_info.local_expert_offset,  # 本地专家偏移
            local_num_experts=quant_info.local_num_experts,  # 本地专家数
            routed_scaling_factor=None,  # 路由缩放因子
            routing_method_type=1,  # Unused, but must be 1 to pass validation. / 未使用，但必须为1以通过验证。
            do_finalize=True,  # 执行最终化
            activation_type=activation_type,  # 激活类型
            tune_max_num_tokens=next_power_of_2(hs_fp4.shape[0]),  # 自适应调整最大令牌数
            output=symm_output,  # 输出张量
        )[0]  # 调用FP4块缩放路由MoE，取第一个返回值
    else:  # 非路由TopK路径
        assert TopKOutputChecker.format_is_bypassed(topk_output)  # 断言TopK输出为绕过格式

        router_logits = topk_output.router_logits  # 获取路由logits
        topk_config = topk_output.topk_config  # 获取TopK配置
        routing_method_type = quant_info.routing_method_type  # 获取路由方法类型

        correction_bias = (
            None
            if topk_config.correction_bias is None
            else topk_config.correction_bias.to(hidden_states.dtype)
        )  # 获取校正偏置
        result = trtllm_fp4_block_scale_moe(
            routing_logits=router_logits,  # 路由logits
            routing_bias=correction_bias,  # 路由偏置
            hidden_states=hs_fp4,  # FP4隐藏状态
            hidden_states_scale=hs_scale,  # 隐藏状态缩放因子
            gemm1_weights=quant_info.w13_weight,  # GEMM1权重
            gemm1_weights_scale=quant_info.w13_weight_scale.view(torch.float8_e4m3fn),  # GEMM1权重缩放因子
            gemm1_bias=None,  # GEMM1偏置
            gemm1_alpha=None,  # GEMM1 alpha
            gemm1_beta=None,  # GEMM1 beta
            gemm1_clamp_limit=gemm1_clamp_limit,  # GEMM1钳位限制
            gemm2_weights=quant_info.w2_weight,  # GEMM2权重
            gemm2_weights_scale=quant_info.w2_weight_scale.view(torch.float8_e4m3fn),  # GEMM2权重缩放因子
            gemm2_bias=None,  # GEMM2偏置
            output1_scale_scalar=quant_info.g1_scale_c,  # 输出1缩放标量
            output1_scale_gate_scalar=quant_info.g1_alphas,  # 输出1门控缩放标量
            output2_scale_scalar=quant_info.g2_alphas,  # 输出2缩放标量
            per_token_scale=per_token_scale,  # 逐令牌缩放因子
            num_experts=quant_info.global_num_experts,  # 全局专家数
            top_k=topk_config.top_k,  # Top-K值
            n_group=topk_config.num_expert_group,  # 专家分组数
            topk_group=topk_config.topk_group,  # Top-K分组数
            intermediate_size=quant_info.intermediate_size_per_partition,  # 分区中间层大小
            local_expert_offset=quant_info.local_expert_offset,  # 本地专家偏移
            local_num_experts=quant_info.local_num_experts,  # 本地专家数
            routed_scaling_factor=runner_config.routed_scaling_factor,  # 路由缩放因子
            routing_method_type=(
                routing_method_type
                if routing_method_type is not None
                else RoutingMethodType.Default
            ),  # 路由方法类型
            do_finalize=True,  # 执行最终化
            activation_type=activation_type,  # 激活类型
            tune_max_num_tokens=next_power_of_2(hs_fp4.shape[0]),  # 自适应调整最大令牌数
            output=symm_output,  # 输出张量
        )[0]  # 调用FP4块缩放MoE，取第一个返回值

    return StandardCombineInput(hidden_states=result)  # 返回标准合并输入


@dataclass
class FlashInferTrtllmBf16MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer TRT-LLM BF16 MoE kernels.
    FlashInfer TRT-LLM BF16 MoE 内核消费的量化负载。
    """

    gemm1_weights: torch.Tensor  # GEMM1权重
    gemm2_weights: torch.Tensor  # GEMM2权重

    # Expert-parallel metadata / 专家并行元数据
    global_num_experts: int  # 全局专家数
    local_expert_offset: int  # 本地专家偏移


def fused_experts_none_to_flashinfer_trtllm_bf16(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmBf16MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    use_routed_topk: bool = False,
) -> StandardCombineInput:
    """FlashInfer TRT-LLM BF16 MoE融合专家前向函数"""
    # lazy import / 惰性导入
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput  # 导入标准合并输入
    from sglang.srt.layers.moe.topk import TopKOutputChecker  # 导入TopK输出检查器
    from sglang.srt.layers.moe.utils import RoutingMethodType  # 导入路由方法类型

    trtllm_bf16_routed_moe = None  # BF16路由MoE函数初始化为None
    trtllm_bf16_moe = None  # BF16 MoE函数初始化为None
    if use_routed_topk:  # 如果使用路由TopK
        try:
            from flashinfer.fused_moe import trtllm_bf16_routed_moe  # 导入BF16路由MoE
        except ImportError as e:
            raise ImportError(
                "Can't import trtllm_bf16_routed_moe from flashinfer. "
                "Please check flashinfer version to use bf16 with flashinfer_trtllm_routed backend."
                "无法从 flashinfer 导入 trtllm_bf16_routed_moe。"
                "请检查 flashinfer 版本以使用 flashinfer_trtllm_routed 后端的 bf16。"
            ) from e
    else:  # 非路由TopK
        try:
            from flashinfer.fused_moe import trtllm_bf16_moe  # 导入BF16 MoE
        except ImportError as e:
            raise ImportError(
                "Can't import trtllm_bf16_moe from flashinfer. "
                "Please check flashinfer version to use bf16 with flashinfer_trtllm backend."
                "无法从 flashinfer 导入 trtllm_bf16_moe。"
                "请检查 flashinfer 版本以使用 flashinfer_trtllm 后端的 bf16。"
            ) from e

    _SUPPORTED_BF16_ACTIVATIONS = {"silu", "relu2"}  # BF16支持的激活类型
    assert runner_config.activation in _SUPPORTED_BF16_ACTIVATIONS, (
        f"Only {_SUPPORTED_BF16_ACTIVATIONS} are supported for flashinfer trtllm bf16 moe, "
        f"got '{runner_config.activation}'."
    )  # 断言激活类型受支持
    if not use_routed_topk:  # 如果非路由TopK
        assert (
            dispatch_output.topk_output.topk_config.renormalize
        ), "Renormalize is required for flashinfer trtllm moe"  # 断言需要重归一化
    assert (
        runner_config.num_fused_shared_experts == 0
    ), "Fused shared experts are not supported for flashinfer trtllm moe"  # 断言不支持融合共享专家
    activation_type = get_activation_type(
        runner_config.activation, is_gated=runner_config.is_gated
    )  # 获取激活类型

    hidden_states = dispatch_output.hidden_states  # 获取隐藏状态
    topk_output = dispatch_output.topk_output  # 获取TopK输出

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):  # 在对称内存上下文中
        if use_routed_topk:  # 如果使用路由TopK
            assert (
                runner_config.top_k is not None
            ), "runner_config.top_k is required for flashinfer_trtllm_routed."  # 断言top_k必须存在
            assert TopKOutputChecker.format_is_standard(topk_output)  # 断言TopK输出为标准格式
            routing_method_type = runner_config.routing_method_type  # 获取路由方法类型
            if routing_method_type is None:  # 如果路由方法类型为None
                routing_method_type = RoutingMethodType.Default  # 使用默认路由方法
            elif routing_method_type == RoutingMethodType.DeepSeekV3:  # 如果是DeepSeekV3路由
                routing_method_type = RoutingMethodType.TopK  # 转换为TopK路由

            packed_topk_ids = _pack_topk_for_flashinfer_routed(
                topk_ids=topk_output.topk_ids,
                topk_weights=topk_output.topk_weights,
            )  # 打包路由TopK
            final_hidden_states = trtllm_bf16_routed_moe(
                topk_ids=packed_topk_ids,  # 打包的TopK ID
                hidden_states=hidden_states,  # 隐藏状态
                gemm1_weights=quant_info.gemm1_weights,  # GEMM1权重
                gemm2_weights=quant_info.gemm2_weights,  # GEMM2权重
                num_experts=quant_info.global_num_experts,  # 全局专家数
                top_k=runner_config.top_k,  # Top-K值
                n_group=None,  # 分组数
                topk_group=None,  # Top-K分组数
                intermediate_size=runner_config.intermediate_size_per_partition,  # 分区中间层大小
                local_expert_offset=quant_info.local_expert_offset,  # 本地专家偏移
                local_num_experts=runner_config.num_local_experts,  # 本地专家数
                routing_method_type=routing_method_type,  # 路由方法类型
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),  # 路由缩放因子
                tune_max_num_tokens=next_power_of_2(hidden_states.shape[0]),  # 自适应调整最大令牌数
                activation_type=activation_type,  # 激活类型
            )  # 调用BF16路由MoE
        else:  # 非路由TopK路径
            assert TopKOutputChecker.format_is_bypassed(topk_output)  # 断言TopK输出为绕过格式
            topk_config = topk_output.topk_config  # 获取TopK配置

            # Call the fused kernel
            # 调用融合内核
            final_hidden_states = trtllm_bf16_moe(
                routing_logits=topk_output.router_logits,  # 路由logits
                routing_bias=topk_config.correction_bias,  # 路由偏置
                hidden_states=hidden_states,  # 隐藏状态
                gemm1_weights=quant_info.gemm1_weights,  # GEMM1权重
                gemm2_weights=quant_info.gemm2_weights,  # GEMM2权重
                num_experts=quant_info.global_num_experts,  # 全局专家数
                top_k=topk_config.top_k,  # Top-K值
                n_group=topk_config.num_expert_group,  # 专家分组数
                topk_group=topk_config.topk_group,  # Top-K分组数
                intermediate_size=runner_config.intermediate_size_per_partition,  # 分区中间层大小
                local_expert_offset=quant_info.local_expert_offset,  # 本地专家偏移
                local_num_experts=runner_config.num_local_experts,  # 本地专家数
                routing_method_type=runner_config.routing_method_type,  # 路由方法类型
                routed_scaling_factor=runner_config.routed_scaling_factor,  # 路由缩放因子
                tune_max_num_tokens=next_power_of_2(hidden_states.shape[0]),  # 自适应调整最大令牌数
                activation_type=activation_type,  # 激活类型
            )  # 调用BF16 MoE

    return StandardCombineInput(hidden_states=final_hidden_states)  # 返回标准合并输入


@register_fused_func("none", "flashinfer_trtllm")  # 注册融合函数：none a2a后端 + flashinfer_trtllm运行器
def fused_experts_none_to_flashinfer_trtllm(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Dispatch to FP8 or FP4 FlashInfer TRT-LLM MoE based on quant_info type.
    根据 quant_info 类型分发到 FP8 或 FP4 FlashInfer TRT-LLM MoE。
    """
    if isinstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo):  # 如果是FP4量化信息
        return fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch_output, quant_info, runner_config
        )  # 调用FP4路径
    if isinstance(quant_info, FlashInferTrtllmFp8MoeQuantInfo):  # 如果是FP8量化信息
        return fused_experts_none_to_flashinfer_trtllm_fp8(
            dispatch_output, quant_info, runner_config
        )  # 调用FP8路径
    if isinstance(quant_info, FlashInferTrtllmBf16MoeQuantInfo):  # 如果是BF16量化信息
        return fused_experts_none_to_flashinfer_trtllm_bf16(
            dispatch_output, quant_info, runner_config
        )  # 调用BF16路径
    raise TypeError(
        f"Unexpected quant_info type for flashinfer_trtllm: {type(quant_info)}"
        f"flashinfer_trtllm 的意外 quant_info 类型: {type(quant_info)}"
    )  # 抛出类型错误


@register_fused_func("none", "flashinfer_trtllm_routed")  # 注册融合函数：none a2a后端 + flashinfer_trtllm_routed运行器
def fused_experts_none_to_flashinfer_trtllm_routed(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """带路由TopK的FlashInfer TRT-LLM MoE融合专家前向函数"""
    if isinstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo):  # 如果是FP4量化信息
        return fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,  # 使用路由TopK
        )  # 调用FP4路径
    if isinstance(quant_info, FlashInferTrtllmFp8MoeQuantInfo):  # 如果是FP8量化信息
        return fused_experts_none_to_flashinfer_trtllm_fp8(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,  # 使用路由TopK
        )  # 调用FP8路径
    if isinstance(quant_info, FlashInferTrtllmBf16MoeQuantInfo):  # 如果是BF16量化信息
        return fused_experts_none_to_flashinfer_trtllm_bf16(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,  # 使用路由TopK
        )  # 调用BF16路径
    raise TypeError(
        f"Unexpected quant_info type for flashinfer_trtllm_routed: {type(quant_info)}"
        f"flashinfer_trtllm_routed 的意外 quant_info 类型: {type(quant_info)}"
    )  # 抛出类型错误
