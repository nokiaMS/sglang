# GGUF量化配置与推理模块 - 实现GGUF格式的量化线性层、MoE和Embedding推理
# 支持CUDA、MUSA和NPU(Ascend)平台，包含多种量化类型（标准量化、K量化、I-Matrix量化）

# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # SPDX版权声明
# Adapted from: https://github.com/vllm-project/vllm/blob/ab3e80042eac24dd362408e6d63ad98768046359/vllm/model_executor/layers/quantization/gguf.py  # 参考自vLLM项目
from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
import warnings  # 导入警告模块
from typing import TYPE_CHECKING, Any, List, Optional  # 导入类型注解

import gguf  # 导入GGUF库
import torch  # 导入PyTorch
from gguf import GGMLQuantizationType as WeightType  # 导入GGML量化类型别名
from torch.nn.parameter import Parameter, UninitializedParameter  # 导入参数类

from sglang.srt.layers.linear import LinearBase  # 导入线性层基类
from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置
from sglang.srt.layers.quantization.base_config import (  # 导入量化基础配置
    FusedMoEMethodBase,  # 融合MoE方法基类
    LinearMethodBase,  # 线性方法基类
    QuantizationConfig,  # 量化配置基类
    QuantizeMethodBase,  # 量化方法基类
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 导入未量化的线性方法
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu, set_weight_attrs  # 导入平台检测和权重属性设置工具

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入MoE token分发器类型
        CombineInput,  # 合并输入类型
        StandardDispatchOutput,  # 标准分发输出类型
    )

_is_cuda = is_cuda()  # 检测是否为CUDA环境
_is_hip = is_hip()  # 检测是否为HIP(AMD)环境
_is_xpu = is_xpu()  # 检测是否为XPU环境
_is_musa = is_musa()  # 检测是否为MUSA环境
_is_npu = is_npu()  # 检测是否为NPU(Ascend)环境

if _is_cuda:  # CUDA平台导入
    from sgl_kernel import moe_align_block_size, moe_sum  # 导入MoE对齐和求和内核
    from sgl_kernel.quantization import (  # 导入GGML量化内核
        ggml_dequantize,  # GGML反量化
        ggml_moe_a8,  # GGML MoE A8内核
        ggml_moe_a8_vec,  # GGML MoE A8向量内核
        ggml_moe_get_block_size,  # 获取MoE块大小
        ggml_mul_mat_a8,  # GGML矩阵乘法A8内核
        ggml_mul_mat_vec_a8,  # GGML矩阵向量乘法A8内核
    )

    from sglang.jit_kernel.activation import gelu_and_mul, silu_and_mul  # 导入激活函数JIT内核
elif _is_musa:  # MUSA平台导入
    from sgl_kernel import gelu_and_mul, moe_align_block_size, moe_sum, silu_and_mul  # 导入MoE和激活函数内核
    from sgl_kernel.quantization import (  # 导入GGML量化内核
        ggml_dequantize,  # GGML反量化
        ggml_moe_a8,  # GGML MoE A8内核
        ggml_moe_a8_vec,  # GGML MoE A8向量内核
        ggml_moe_get_block_size,  # 获取MoE块大小
        ggml_mul_mat_a8,  # GGML矩阵乘法A8内核
        ggml_mul_mat_vec_a8,  # GGML矩阵向量乘法A8内核
    )
elif _is_npu:  # NPU平台导入
    from gguf import dequantize as gguf_dequantize  # 导入gguf反量化函数
else:  # 其他平台
    if not _is_hip:  # 如果不是HIP平台
        warnings.warn(f"Only CUDA, MUSA and NPU support GGUF quantization currently.")  # 发出警告

logger = logging.getLogger(__name__)  # 创建日志记录器


class GGUFConfig(QuantizationConfig):  # GGUF量化配置类
    """Config class for GGUF."""  # GGUF的配置类

    def __init__(self, modules_to_not_convert: list[str] | None = None) -> None:  # 初始化函数
        super().__init__()  # 调用父类初始化
        if _is_hip:  # 如果是HIP平台
            warnings.warn(f"Only CUDA and MUSA support GGUF quantization currently.")  # 发出兼容性警告
        self.modules_to_not_convert = modules_to_not_convert or []  # 不转换的模块列表

    def __repr__(self) -> str:  # 字符串表示
        return "GGUFConfig()"  # 返回配置类名字符串

    def get_scaled_act_names(self) -> List[str]:  # 获取需要缩放的激活名称
        return []  # 无需缩放的激活

    def get_name(self) -> "str":  # 获取量化方法名称
        return "gguf"  # 返回gguf

    def get_supported_act_dtypes(self) -> list[torch.dtype]:  # 获取支持的激活数据类型
        return [torch.half, torch.bfloat16, torch.float32]  # 支持半精度、bfloat16和float32

    @classmethod  # 类方法装饰器
    def get_min_capability(cls) -> int:  # 获取最低GPU计算能力要求
        return 60 if not _is_musa else 21  # CUDA要求SM60，MUSA要求21

    @classmethod  # 类方法装饰器
    def get_config_filenames(cls) -> list[str]:  # 获取配置文件名列表
        return []  # no extra configs.  # 无额外配置文件

    @classmethod  # 类方法装饰器
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":  # 从配置字典创建配置对象
        modules_to_not_convert = cls.get_from_keys_or(  # 获取不转换的模块列表
            config, ["modules_to_not_convert"], None
        )
        return cls(modules_to_not_convert)  # 创建并返回配置对象

    def get_quant_method(  # 获取量化方法
        self, layer: torch.nn.Module, prefix: str  # 层对象和前缀
    ) -> Optional["QuantizeMethodBase"]:  # 返回量化方法或None
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 延迟导入FusedMoE
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding  # 延迟导入词表并行嵌入

        if isinstance(layer, LinearBase):  # 如果是线性层
            if is_layer_skipped_gguf(prefix, self.modules_to_not_convert):  # 检查是否跳过
                return UnquantizedLinearMethod()  # 返回未量化方法
            if _is_npu:  # 如果是NPU平台
                return GGUFLinearAscendMethod(self)  # 返回NPU专用方法
            return GGUFLinearMethod(self)  # 返回标准GGUF线性方法
        elif isinstance(layer, VocabParallelEmbedding):  # 如果是词表并行嵌入层
            if _is_npu:  # 如果是NPU平台
                return GGUFEmbeddingAscendMethod(self)  # 返回NPU专用嵌入方法
            return GGUFEmbeddingMethod(self)  # 返回标准嵌入方法
        elif isinstance(layer, FusedMoE):  # 如果是融合MoE层
            if _is_npu:  # 如果是NPU平台
                return GGUFMoEAscendMethod(self)  # 返回NPU专用MoE方法
            return GGUFMoEMethod(self)  # 返回标准MoE方法
        return None  # 不支持的层类型返回None


def is_layer_skipped_gguf(prefix: str, modules_to_not_convert: list[str]):  # 检查GGUF层是否跳过量化的函数
    """检查给定前缀的层是否在跳过列表中。"""  # 中文函数说明
    return any(module_name in prefix for module_name in modules_to_not_convert)  # 前缀中包含任一模块名则跳过


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}  # 未量化的数据类型集合
STANDARD_QUANT_TYPES = {  # 标准量化类型集合
    WeightType.Q4_0,  # 4位量化类型0
    WeightType.Q4_1,  # 4位量化类型1
    WeightType.Q5_0,  # 5位量化类型0
    WeightType.Q5_1,  # 5位量化类型1
    WeightType.Q8_0,  # 8位量化类型0
    WeightType.Q8_1,  # 8位量化类型1
}
KQUANT_TYPES = {  # K量化类型集合
    WeightType.Q2_K,  # 2位K量化
    WeightType.Q3_K,  # 3位K量化
    WeightType.Q4_K,  # 4位K量化
    WeightType.Q5_K,  # 5位K量化
    WeightType.Q6_K,  # 6位K量化
}
IMATRIX_QUANT_TYPES = {  # I-Matrix量化类型集合
    WeightType.IQ1_M,  # 1位I-Matrix量化M
    WeightType.IQ1_S,  # 1位I-Matrix量化S
    WeightType.IQ2_XXS,  # 2位I-Matrix量化XXS
    WeightType.IQ2_XS,  # 2位I-Matrix量化XS
    WeightType.IQ2_S,  # 2位I-Matrix量化S
    WeightType.IQ3_XXS,  # 3位I-Matrix量化XXS
    WeightType.IQ3_S,  # 3位I-Matrix量化S
    WeightType.IQ4_XS,  # 4位I-Matrix量化XS
    WeightType.IQ4_NL,  # 4位I-Matrix量化NL
}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.  # 当前没有I-Matrix量化的MMQ内核
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.  # 添加I-Matrix MMQ内核后合并这些类型集合
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES  # 需要反量化的类型集合
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES  # 支持MMVQ的量化类型集合
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES  # 支持MMQ的量化类型集合（不含I-Matrix）


def fused_mul_mat_gguf(  # GGUF融合矩阵乘法函数
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int  # 输入、量化权重、权重类型
) -> torch.Tensor:  # 返回矩阵乘法结果
    """执行GGUF格式的融合矩阵乘法，自动选择最优内核（MMVQ/MMQ/反量化）。"""  # 中文函数说明
    if qweight_type in IMATRIX_QUANT_TYPES:  # 如果是I-Matrix量化类型
        mmvq_safe = 8 if qweight.shape[0] > 5120 else 16  # 大权重时MMVQ安全阈值为8，否则16
    else:  # 非I-Matrix量化
        mmvq_safe = 2 if qweight.shape[0] > 5120 else 6  # 大权重时阈值为2，否则6
    # HACK: when doing chunked prefill we don't generate output tokens
    # so input to logits generator is empty which causes invalid parameter  # 分块预填充时可能产生空输入
    if x.shape[0] == 0:  # 如果输入为空
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)  # 返回空张量
    # there is no need to call any kernel for fp16/bf16  # fp16/bf16无需调用量化内核
    if qweight_type in UNQUANTIZED_TYPES:  # 如果权重未量化
        return x @ qweight.T  # 直接执行矩阵乘法
    # enable MMVQ in contiguous batching with batch_size=1  # 批量大小为1时启用MMVQ
    if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:  # 小批量且支持MMVQ
        y = ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])  # 使用矩阵向量乘法内核
    # Use MMQ Kernel if it's available (standard + k-quants)  # 标准和K量化类型使用MMQ内核
    elif qweight_type in MMQ_QUANT_TYPES:  # 支持MMQ的量化类型
        y = ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])  # 使用矩阵乘法内核
    # If there is no available MMQ kernel, fallback to dequantize  # 无MMQ内核时回退到反量化
    elif qweight_type in DEQUANT_TYPES:  # 支持反量化的类型
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]  # 获取块大小和类型大小
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)  # 计算反量化后的形状
        weight = ggml_dequantize(qweight, qweight_type, *shape, x.dtype)  # 执行反量化
        y = x @ weight.T  # 用反量化后的权重执行矩阵乘法
    else:  # 不支持的量化类型
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.  # 确保量化类型有效
        qweight_type = WeightType(qweight_type)  # 包装为GGML量化类型枚举
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")  # 抛出不支持错误
    return y  # 返回矩阵乘法结果


def fused_moe_gguf(  # GGUF融合MoE函数
    x: torch.Tensor,  # 输入张量
    w1: torch.Tensor,  # 门控/上投影权重
    w2: torch.Tensor,  # 下投影权重
    topk_weights: torch.Tensor,  # Top-K权重
    topk_ids: torch.Tensor,  # Top-K专家ID
    qweight_type: int,  # w1的量化类型
    qweight_type2: int,  # w2的量化类型
    activation: str,  # 激活函数名称
) -> torch.Tensor:  # 返回MoE输出
    """执行GGUF格式的融合MoE推理，支持MMQ和MMVQ两种内核路径。"""  # 中文函数说明
    def act(x: torch.Tensor):  # 激活函数选择器
        if activation == "silu":  # SiLU激活
            return silu_and_mul(x)  # 执行SiLU和乘法
        elif activation == "gelu":  # GELU激活
            return gelu_and_mul(x)  # 执行GELU和乘法
        raise ValueError(f"Unsupported activation: {activation}")  # 不支持的激活函数

    out_hidden_states = torch.empty_like(x)  # 创建输出隐藏状态张量
    # unless we decent expert reuse we are better off running moe_vec kernel  # 除非有充分的专家复用，否则使用moe_vec内核更好
    if (  # 判断是否使用MMQ MoE内核
        qweight_type2 in MMQ_QUANT_TYPES  # w2支持MMQ
        and qweight_type in MMQ_QUANT_TYPES  # w1支持MMQ
        and x.shape[0] > 64  # token数大于64
    ):
        num_tokens, _ = x.shape  # 获取token数
        E, N, _ = w1.shape  # 获取专家数和中间维度
        top_k = topk_ids.shape[1]  # 获取Top-K值
        BLOCK_SIZE = ggml_moe_get_block_size(qweight_type)  # 获取MoE块大小

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(  # 对齐MoE块大小
            topk_ids, BLOCK_SIZE, E
        )
        out = ggml_moe_a8(  # 执行门控/上投影MoE内核
            x,  # 输入
            w1,  # 权重
            sorted_token_ids,  # 排序后的token ID
            expert_ids,  # 专家ID
            num_tokens_post_padded,  # 填充后的token数
            qweight_type,  # 量化类型
            N,  # 中间维度
            top_k,  # Top-K值
            num_tokens,  # token数
        )
        out = act(out)  # 应用激活函数
        out = ggml_moe_a8(  # 执行下投影MoE内核
            out,  # 激活后的输出
            w2,  # 下投影权重
            sorted_token_ids,  # 排序后的token ID
            expert_ids,  # 专家ID
            num_tokens_post_padded,  # 填充后的token数
            qweight_type2,  # w2量化类型
            w2.shape[1],  # 输出维度
            1,  # top_k为1
            num_tokens * top_k,  # 总token数
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(  # 重塑形状并乘以Top-K权重
            topk_weights.view(num_tokens, top_k, 1)
        )
        # TODO(FlamingoPg): maybe we can use moe_sum_reduce here?  # 是否可以使用moe_sum_reduce
        moe_sum(out, out_hidden_states)  # 对专家输出求和
    elif qweight_type2 in MMVQ_QUANT_TYPES and qweight_type in MMVQ_QUANT_TYPES:  # 使用MMVQ内核
        num_tokens, _ = x.shape  # 获取token数
        E, N, _ = w1.shape  # 获取专家数和中间维度
        top_k = topk_ids.shape[1]  # 获取Top-K值

        out = ggml_moe_a8_vec(x, w1, topk_ids, top_k, qweight_type, N, num_tokens)  # 执行MMVQ门控/上投影
        out = act(out)  # 应用激活函数

        out = ggml_moe_a8_vec(  # 执行MMVQ下投影
            out, w2, topk_ids, 1, qweight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(  # 重塑并乘以Top-K权重
            topk_weights.view(num_tokens, top_k, 1)
        )
        moe_sum(out, out_hidden_states)  # 对专家输出求和
    else:  # 无快速MoE内核，使用慢速实现
        logger.warning_once(  # 记录一次警告
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "  # 当前量化方法没有快速MoE内核支持，回退到慢速实现
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):  # 逐token处理
            inp = x[tok].reshape((1,) + x.shape[1:])  # 获取当前token输入
            current_hidden_state = None  # 初始化当前隐藏状态
            for ww, ii in zip(w, idx):  # 遍历每个专家
                expert_up = w1[ii]  # 获取门控/上投影权重

                out = fused_mul_mat_gguf(inp, expert_up, qweight_type)  # 执行矩阵乘法
                out = act(out)  # 应用激活函数

                expert_down = w2[ii]  # 获取下投影权重
                current_state = fused_mul_mat_gguf(  # 执行下投影
                    out, expert_down, qweight_type2
                ).mul_(ww)  # 乘以专家权重
                if current_hidden_state is None:  # 第一个专家
                    current_hidden_state = current_state  # 初始化隐藏状态
                else:  # 后续专家
                    current_hidden_state.add_(current_state)  # 累加专家输出
            out_hidden_states[tok] = current_hidden_state  # 保存当前token的输出
    return out_hidden_states  # 返回MoE输出


def apply_gguf_embedding(  # 应用GGUF嵌入函数
    x: torch.Tensor,  # 输入索引张量
    qweight: torch.Tensor,  # 量化权重
    qweight_type: int,  # 权重量化类型
    hidden_size: int,  # 隐藏层大小
    dtype: torch.dtype | None = None,  # 输出数据类型
) -> torch.Tensor:  # 返回嵌入结果
    """应用GGUF格式的嵌入查找，支持量化和未量化权重。"""  # 中文函数说明
    if qweight_type in UNQUANTIZED_TYPES:  # 如果权重未量化
        return torch.embedding(qweight, x)  # 直接使用标准嵌入
    elif qweight_type in DEQUANT_TYPES:  # 如果需要反量化
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]  # 获取块大小和类型大小
        x_flat = x.flatten()  # 展平索引
        assert hidden_size == qweight.shape[1] // type_size * block_size  # 断言隐藏大小正确
        quant = torch.index_select(qweight, dim=0, index=x_flat)  # 按索引选择量化权重
        dequant = ggml_dequantize(  # 执行反量化
            quant, qweight_type, hidden_size, x_flat.shape[0], dtype
        )
        return dequant.view(*x.shape, hidden_size)  # 重塑并返回
    else:  # 不支持的量化类型
        qweight_type = WeightType(qweight_type)  # 包装为枚举类型
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")  # 抛出不支持错误


class GGUFLinearMethod(LinearMethodBase):  # GGUF线性方法类
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """  # GGUF线性方法，接收量化配置

    def __init__(self, quant_config: GGUFConfig):  # 初始化函数
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建权重参数
        self,
        layer: torch.nn.Module,  # 神经网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: list[int],  # 输出分区大小列表
        input_size: int,  # 总输入大小
        output_size: int,  # 总输出大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        """创建GGUF线性层的量化权重和权重类型参数。"""  # 中文函数说明
        self.params_dtype = params_dtype  # 保存参数数据类型
        output_size_per_partition = sum(output_partition_sizes)  # 计算分区输出总大小

        tensor_shape = (output_size_per_partition, input_size_per_partition)  # 张量形状
        qweight = GGUFUninitializedParameter(requires_grad=False)  # 创建未初始化的量化权重参数
        set_weight_attrs(  # 设置权重属性
            qweight,
            {
                "input_dim": 1,  # 输入维度索引
                "output_dim": 0,  # 输出维度索引
                "tensor_shape": tensor_shape,  # 张量形状
                "is_gguf_weight": True,  # 标记为GGUF权重
                "data_container": [],  # 数据容器
                "shard_id": [],  # 分片ID列表
                "shard_id_map": {},  # 分片ID映射
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)  # 设置额外属性
        layer.register_parameter("qweight", qweight)  # 注册量化权重参数

        qweight_type = Parameter(  # 创建权重类型参数
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),  # 每个分区一个uint8类型值
            requires_grad=False,  # 不需要梯度
        )
        set_weight_attrs(  # 设置权重类型属性
            qweight_type,
            {
                "is_gguf_weight_type": True,  # 标记为GGUF权重类型
                "weight_type": 0,  # 权重类型初始值
                "shard_weight_type": {},  # 分片权重类型映射
                "ignore_warning": True,  # 忽略警告
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)  # 设置额外属性
        layer.register_parameter("qweight_type", qweight_type)  # 注册权重类型参数

    def process_weights_after_loading(self, layer: torch.nn.Module):  # 加载后处理权重
        """权重加载后验证量化类型并创建填充权重参数。"""  # 中文函数说明
        qweight_type = layer.qweight_type.weight_type  # 获取权重量化类型
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):  # 验证类型有效性
            qweight_type = WeightType(qweight_type)  # 包装为枚举
            raise ValueError(  # 抛出值错误
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        # For MergedColumnParallelLinear and QKVParallelLinear, we need to
        # materialize the padded weight parameter for CUDA Graph compatibility.  # 为CUDA Graph兼容性创建填充权重
        self._create_padded_weight_param(layer)  # 创建填充权重参数

    def _create_padded_weight_param(self, layer: torch.nn.Module):  # 创建填充权重参数
        """Create padded weight parameter for GGUF MergedLinear layer."""  # 为GGUF合并线性层创建填充权重参数
        qweight = layer.qweight  # 获取量化权重
        shard_id_map = qweight.shard_id_map  # 获取分片ID映射
        shard_id = qweight.shard_id  # 获取分片ID列表
        if len(data_container := qweight.data_container) > 1:  # 如果有多个数据分片
            dtype = {data.dtype for data in data_container}  # 收集所有数据类型
            assert len(dtype) == 1, ValueError(  # 断言类型一致
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))  # 获取唯一的数据类型
            # concat dim0 and pad dim1  # 连接维度0并填充维度1
            padded_side = max(x.size(1) for x in data_container)  # 计算维度1的最大值
            concat_side = sum(x.size(0) for x in data_container)  # 计算维度0的总和
            # Pad the quantized weights to dense tensor, and create a map
            # with the location of each shard in the padded tensor.  # 将量化权重填充为密集张量，创建分片位置映射
            padded_data = torch.zeros(  # 创建填充数据
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            # (dim0_start, dim0_end, dim1_size)  # 维度0起始、结束和维度1大小的元组
            shard_offset_map = dict[str, tuple[int, int, int]]()  # 分片偏移映射
            for idx in shard_id:  # 遍历每个分片ID
                id_in_container = shard_id_map[idx]  # 获取在容器中的索引
                start = sum(x.size(0) for x in data_container[:id_in_container])  # 计算起始位置
                end = start + data_container[id_in_container].size(0)  # 计算结束位置
                size = data_container[id_in_container].size(1)  # 获取维度1大小
                padded_data[start:end, :size] = data_container[id_in_container]  # 填充数据
                shard_offset_map[idx] = (start, end, size)  # 记录偏移信息
            qweight.data_container.clear()  # 清空数据容器
            padded_param = Parameter(padded_data, requires_grad=False)  # 创建填充参数
            set_weight_attrs(padded_param, vars(qweight))  # 复制属性
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})  # 设置偏移映射
            layer.register_parameter("qweight", padded_param)  # 注册填充参数

    def apply(  # 应用线性运算
        self,
        layer: torch.nn.Module,  # 神经网络层
        x: torch.Tensor,  # 输入张量
        bias: torch.Tensor | None = None,  # 偏置项
    ) -> torch.Tensor:  # 返回输出张量
        """执行GGUF量化线性推理，支持分片和非分片两种模式。"""  # 中文函数说明
        shard_id = layer.qweight.shard_id  # 获取分片ID

        if shard_id:  # 如果有分片（QKV合并等情况）
            # dequantize shard weights respectively  # 分别反量化各分片权重
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id  # QKV映射
            qweight = layer.qweight  # 获取量化权重
            result = []  # 结果列表
            for idx in shard_id:  # 遍历每个分片
                start, end, offset = layer.qweight.shard_offset_map[idx]  # 获取偏移信息
                qweight_type = layer.qweight_type.shard_weight_type[idx]  # 获取量化类型
                result.append(  # 添加到结果列表
                    fused_mul_mat_gguf(
                        x, qweight[start:end, :offset].contiguous(), qweight_type
                    )
                )
            out = torch.cat(result, axis=1)  # 沿维度1拼接结果
        else:  # 无分片
            qweight = layer.qweight  # 获取量化权重
            qweight_type = layer.qweight_type.weight_type  # 获取权重类型
            out = fused_mul_mat_gguf(x, qweight, qweight_type)  # 执行矩阵乘法
        if bias is not None:  # 如果有偏置
            out.add_(bias)  # 添加偏置
        return out  # 返回输出


class GGUFMoEMethod(FusedMoEMethodBase):  # GGUF MoE方法类
    """MoE method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """  # GGUF MoE方法，接收量化配置

    def __init__(self, quant_config: GGUFConfig):  # 初始化函数
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建MoE权重参数
        self,
        layer: torch.nn.Module,  # 神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        """创建GGUF MoE层的门控/上投影和下投影权重参数。"""  # 中文函数说明
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)  # w13张量形状
        # gate up proj  # 门控上投影
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)  # 创建未初始化的w13权重
        set_weight_attrs(  # 设置w13权重属性
            w13_qweight,
            {
                "input_dim": 1,  # 输入维度
                "output_dim": 0,  # 输出维度
                "tensor_shape": tensor_shape,  # 张量形状
                "is_gguf_weight": True,  # GGUF权重标记
                "data_container": [],  # 数据容器
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)  # 设置额外属性
        layer.register_parameter("w13_qweight", w13_qweight)  # 注册w13参数

        w13_qweight_type = Parameter(  # 创建w13权重类型参数
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(  # 设置w13权重类型属性
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)  # 设置额外属性
        layer.register_parameter("w13_qweight_type", w13_qweight_type)  # 注册w13类型参数

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)  # w2张量形状
        # gate down proj  # 下投影
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)  # 创建未初始化的w2权重
        set_weight_attrs(  # 设置w2权重属性
            w2_qweight,
            {
                "input_dim": 1,  # 输入维度
                "output_dim": 0,  # 输出维度
                "tensor_shape": tensor_shape,  # 张量形状
                "is_gguf_weight": True,  # GGUF权重标记
                "data_container": [],  # 数据容器
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)  # 设置额外属性
        layer.register_parameter("w2_qweight", w2_qweight)  # 注册w2参数

        w2_qweight_type = Parameter(  # 创建w2权重类型参数
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(  # 设置w2权重类型属性
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )

        set_weight_attrs(w2_qweight_type, extra_weight_attrs)  # 设置额外属性
        layer.register_parameter("w2_qweight_type", w2_qweight_type)  # 注册w2类型参数

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 层对象和配置
    ):
        """创建并保存MoE运行器配置。"""  # 中文函数说明
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置

    def apply(  # 应用MoE推理
        self,
        layer: torch.nn.Module,  # 神经网络层
        dispatch_output: StandardDispatchOutput,  # 分发输出
    ) -> CombineInput:  # 返回合并输入
        """执行GGUF MoE前向推理。"""  # 中文函数说明
        assert self.fused_experts is None  # 断言不使用融合专家

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入

        assert (  # 断言
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."  # 仅支持SiLU激活

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取Top-K输出

        moe_runner_config = self.moe_runner_config  # 获取MoE运行器配置

        topk_weights, topk_ids, _ = topk_output  # 解包Top-K结果
        output = fused_moe_gguf(  # 执行融合MoE推理
            x=x,  # 隐藏状态
            w1=layer.w13_qweight,  # 门控/上投影权重
            w2=layer.w2_qweight,  # 下投影权重
            topk_weights=topk_weights,  # Top-K权重
            topk_ids=topk_ids,  # Top-K专家ID
            qweight_type=layer.w13_qweight_type.weight_type,  # w1量化类型
            qweight_type2=layer.w2_qweight_type.weight_type,  # w2量化类型
            activation=moe_runner_config.activation,  # 激活函数
        )
        return StandardCombineInput(hidden_states=output)  # 返回标准合并输入


class GGUFEmbeddingMethod(GGUFLinearMethod):  # GGUF嵌入方法类，继承自线性方法
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """  # GGUF嵌入方法，接收量化配置

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:  # 嵌入查找函数
        """执行GGUF格式的嵌入查找操作。"""  # 中文函数说明
        qweight = layer.qweight  # 获取量化权重
        qweight_type = layer.qweight_type.weight_type  # 获取权重类型
        hidden_size = qweight.tensor_shape[1]  # 获取隐藏大小

        return apply_gguf_embedding(  # 应用GGUF嵌入
            x, qweight, qweight_type, hidden_size, dtype=self.params_dtype
        )


class GGUFUninitializedParameter(UninitializedParameter):  # GGUF未初始化参数类
    cls_to_become = Parameter  # 加载后转换为目标类
    data_container: list[torch.Tensor]  # 数据容器类型注解


# =============================================================================
# NPU-specific implementations for Ascend hardware  # NPU(Ascend)专用实现
# =============================================================================
def ggml_dequantize_ascend(  # NPU平台的GGML反量化函数
    qweight: torch.Tensor,  # 量化权重
    qweight_type: int,  # 权重量化类型
    rows: int,  # 行数
    cols: int,  # 列数
    dtype: torch.dtype,  # 输出数据类型
) -> torch.Tensor:  # 返回反量化后的权重
    """Dequantize GGML quantized weights for NPU.

    Uses gguf library's reference implementation which supports all GGML formats
    and is guaranteed to be correct. The dequantization runs on CPU during model
    loading, then the dequantized weights are transferred to NPU for inference.
    """  # NPU平台的GGML反量化，使用gguf库的参考实现，CPU反量化后传输到NPU

    # Move to CPU for dequantization using gguf library  # 移到CPU使用gguf库反量化
    qweight_cpu = qweight.cpu().numpy()  # 转换为CPU上的numpy数组

    # Use gguf library's dequantize (supports all GGML formats)  # 使用gguf库反量化（支持所有GGML格式）
    dequant_np = gguf_dequantize(qweight_cpu, qweight_type)  # 执行反量化

    # Convert to torch and move to target device  # 转换为torch张量并移到目标设备
    result = torch.from_numpy(dequant_np).to(dtype=dtype, device=qweight.device)  # 转换类型和设备
    result = result.reshape(rows, cols)  # 重塑形状

    return result  # 返回反量化结果


class GGUFLinearAscendMethod(LinearMethodBase):  # GGUF线性Ascend方法类
    """Linear method for GGUF on Ascend NPU."""  # Ascend NPU上的GGUF线性方法

    def __init__(self, quant_config: GGUFConfig):  # 初始化函数
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建权重参数
        self,
        layer: torch.nn.Module,  # 神经网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: list[int],  # 输出分区大小列表
        input_size: int,  # 总输入大小
        output_size: int,  # 总输出大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        """创建NPU平台的GGUF线性层权重和权重类型参数。"""  # 中文函数说明
        self.params_dtype = params_dtype  # 保存参数数据类型
        output_size_per_partition = sum(output_partition_sizes)  # 计算分区输出总大小

        tensor_shape = (output_size_per_partition, input_size_per_partition)  # 张量形状
        qweight = GGUFUninitializedParameter(requires_grad=False)  # 创建未初始化的量化权重
        set_weight_attrs(  # 设置权重属性
            qweight,
            {
                "input_dim": 1,  # 输入维度
                "output_dim": 0,  # 输出维度
                "tensor_shape": tensor_shape,  # 张量形状
                "is_gguf_weight": True,  # GGUF权重标记
                "data_container": [],  # 数据容器
                "shard_id": [],  # 分片ID列表
                "shard_id_map": {},  # 分片ID映射
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)  # 设置额外属性
        layer.register_parameter("qweight", qweight)  # 注册量化权重参数

        qweight_type = Parameter(  # 创建权重类型参数
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(  # 设置权重类型属性
            qweight_type,
            {
                "is_gguf_weight_type": True,  # GGUF权重类型标记
                "weight_type": 0,  # 权重类型初始值
                "shard_weight_type": {},  # 分片权重类型映射
                "ignore_warning": True,  # 忽略警告
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)  # 设置额外属性
        layer.register_parameter("qweight_type", qweight_type)  # 注册权重类型参数

    def process_weights_after_loading(self, layer: torch.nn.Module):  # 加载后处理权重
        """权重加载后验证量化类型、创建填充权重并预反量化。"""  # 中文函数说明
        qweight_type = layer.qweight_type.weight_type  # 获取权重量化类型
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):  # 验证类型
            raise ValueError(  # 抛出值错误
                f"Unsupported GGUF quantization type {WeightType(qweight_type)} in layer."
            )
        self._create_padded_weight_param(layer)  # 创建填充权重参数
        # Pre-dequantize weights for faster inference  # 预反量化以加速推理
        self._pre_dequantize_weights(layer)  # 预反量化权重

    def _create_padded_weight_param(self, layer: torch.nn.Module):  # 创建填充权重参数
        """Create padded weight parameter for GGUF MergedLinear layer."""  # 为GGUF合并线性层创建填充权重参数
        qweight = layer.qweight  # 获取量化权重
        shard_id_map = qweight.shard_id_map  # 获取分片ID映射
        shard_id = qweight.shard_id  # 获取分片ID列表
        if len(data_container := qweight.data_container) > 1:  # 如果有多个数据分片
            dtype = {data.dtype for data in data_container}  # 收集数据类型
            assert len(dtype) == 1  # 断言类型一致
            dtype = next(iter(dtype))  # 获取数据类型
            padded_side = max(x.size(1) for x in data_container)  # 计算维度1最大值
            concat_side = sum(x.size(0) for x in data_container)  # 计算维度0总和
            padded_data = torch.zeros(  # 创建填充数据
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()  # 分片偏移映射
            for idx in shard_id:  # 遍历分片ID
                id_in_container = shard_id_map[idx]  # 获取容器索引
                start = sum(x.size(0) for x in data_container[:id_in_container])  # 起始位置
                end = start + data_container[id_in_container].size(0)  # 结束位置
                size = data_container[id_in_container].size(1)  # 维度1大小
                padded_data[start:end, :size] = data_container[id_in_container]  # 填充数据
                shard_offset_map[idx] = (start, end, size)  # 记录偏移
            qweight.data_container.clear()  # 清空数据容器
            padded_param = Parameter(padded_data, requires_grad=False)  # 创建填充参数
            set_weight_attrs(padded_param, vars(qweight))  # 复制属性
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})  # 设置偏移映射
            layer.register_parameter("qweight", padded_param)  # 注册填充参数

    def _pre_dequantize_weights(self, layer: torch.nn.Module):  # 预反量化权重
        """Pre-dequantize GGML weights to FP16 for faster inference.

        This eliminates runtime dequantization overhead at the cost of more memory.
        """  # 预反量化GGML权重为FP16以加速推理，代价是更多内存
        qweight = layer.qweight  # 获取量化权重
        qweight_type = layer.qweight_type.weight_type  # 获取权重类型

        if qweight_type in UNQUANTIZED_TYPES and qweight.dtype in (  # 如果已经是浮点格式
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            layer.dequantized_weight = qweight  # 直接使用原始权重
            return  # 返回

        shard_id = getattr(qweight, "shard_id", None)  # 获取分片ID
        has_shard_offset = hasattr(qweight, "shard_offset_map")  # 是否有偏移映射

        if shard_id and has_shard_offset:  # 如果有分片
            # Handle sharded weights (QKV merged)  # 处理分片权重（QKV合并）
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id  # QKV映射
            dequant_shards = []  # 反量化分片列表
            for idx in shard_id:  # 遍历分片
                start, end, offset = qweight.shard_offset_map[idx]  # 获取偏移
                shard_qtype = layer.qweight_type.shard_weight_type[idx]  # 获取类型
                shard_data = qweight[start:end, :offset].contiguous()  # 获取数据

                block_size, type_size = gguf.GGML_QUANT_SIZES[shard_qtype]  # 获取块和类型大小
                shape = (  # 计算形状
                    shard_data.shape[0],
                    shard_data.shape[1] // type_size * block_size,
                )
                dequant = ggml_dequantize_ascend(  # 在NPU上反量化
                    shard_data, shard_qtype, *shape, self.params_dtype
                )
                dequant_shards.append(dequant)  # 添加到列表

            dequant_weight = torch.cat(dequant_shards, dim=0)  # 沿维度0拼接
        else:  # 单个权重
            # Handle single weight  # 处理单个权重
            block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]  # 获取块和类型大小
            shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)  # 计算形状
            dequant_weight = ggml_dequantize_ascend(  # 执行反量化
                qweight, qweight_type, *shape, self.params_dtype
            )

        layer.dequantized_weight = dequant_weight  # 保存反量化后的权重

        if hasattr(layer, "qweight"):  # 如果有qweight属性
            del layer.qweight  # 删除以节省内存
        if hasattr(layer, "qweight_type"):  # 如果有qweight_type属性
            del layer.qweight_type  # 删除以节省内存

    def apply(  # 应用线性运算
        self,
        layer: torch.nn.Module,  # 神经网络层
        x: torch.Tensor,  # 输入张量
        bias: torch.Tensor | None = None,  # 偏置项
    ) -> torch.Tensor:  # 返回输出张量
        """使用预反量化的权重执行NPU上的线性推理。"""  # 中文函数说明
        # Use pre-dequantized weight (always available after process_weights_after_loading)  # 使用预反量化权重
        weight = layer.dequantized_weight  # 获取反量化权重
        out = x @ weight.T  # 执行矩阵乘法
        if bias is not None:  # 如果有偏置
            out.add_(bias)  # 添加偏置
        return out  # 返回输出


class GGUFMoEAscendMethod(FusedMoEMethodBase):  # GGUF MoE Ascend方法类
    """MoE method for GGUF on Ascend NPU."""  # Ascend NPU上的GGUF MoE方法

    def __init__(self, quant_config: GGUFConfig):  # 初始化函数
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建MoE权重参数
        self,
        layer: torch.nn.Module,  # 神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        """创建NPU平台的GGUF MoE层权重参数。"""  # 中文函数说明
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)  # w13形状
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)  # w13权重
        set_weight_attrs(  # 设置w13属性
            w13_qweight,
            {
                "input_dim": 1,  # 输入维度
                "output_dim": 0,  # 输出维度
                "tensor_shape": tensor_shape,  # 张量形状
                "is_gguf_weight": True,  # GGUF标记
                "data_container": [],  # 数据容器
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)  # 设置额外属性
        layer.register_parameter("w13_qweight", w13_qweight)  # 注册w13参数

        w13_qweight_type = Parameter(  # w13权重类型
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(  # 设置w13类型属性
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)  # 设置额外属性
        layer.register_parameter("w13_qweight_type", w13_qweight_type)  # 注册w13类型

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)  # w2形状
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)  # w2权重
        set_weight_attrs(  # 设置w2属性
            w2_qweight,
            {
                "input_dim": 1,  # 输入维度
                "output_dim": 0,  # 输出维度
                "tensor_shape": tensor_shape,  # 张量形状
                "is_gguf_weight": True,  # GGUF标记
                "data_container": [],  # 数据容器
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)  # 设置额外属性
        layer.register_parameter("w2_qweight", w2_qweight)  # 注册w2参数

        w2_qweight_type = Parameter(  # w2权重类型
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(  # 设置w2类型属性
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w2_qweight_type, extra_weight_attrs)  # 设置额外属性
        layer.register_parameter("w2_qweight_type", w2_qweight_type)  # 注册w2类型

        # Store params_dtype for pre-dequantization  # 保存参数类型用于预反量化
        self.params_dtype = params_dtype  # 保存参数数据类型

    def process_weights_after_loading(self, layer: torch.nn.Module):  # 加载后处理权重
        """Pre-dequantize MoE weights to FP16 for faster inference."""  # 预反量化MoE权重为FP16以加速推理

        if hasattr(layer, "materialize_gguf_weights"):  # 如果有权重物化方法
            layer.materialize_gguf_weights()  # 物化权重

        # Check if weights are actually loaded (not still UninitializedParameter/empty)  # 检查权重是否已加载
        w13_qweight = layer.w13_qweight  # 获取w13权重
        w13_qtype = layer.w13_qweight_type.weight_type  # 获取w13类型

        # Pre-dequantize w13 weights (gate+up projections)  # 预反量化w13权重（门控+上投影）
        if w13_qtype not in UNQUANTIZED_TYPES:  # 如果w13需要反量化
            num_experts = w13_qweight.shape[0]  # 获取专家数量
            w13_dequant_list = []  # w13反量化列表

            block_size, type_size = gguf.GGML_QUANT_SIZES[w13_qtype]  # 获取块和类型大小

            for e in range(num_experts):  # 遍历每个专家
                qweight_cpu = w13_qweight[e].cpu().numpy()  # 转到CPU
                rows = w13_qweight[e].shape[0]  # 行数
                cols = w13_qweight[e].shape[1] // type_size * block_size  # 列数

                dequant_np = gguf_dequantize(qweight_cpu.flatten(), w13_qtype)  # 反量化
                dequant = (  # 转换为torch张量
                    torch.from_numpy(dequant_np)
                    .to(dtype=self.params_dtype, device=w13_qweight.device)  # 转换类型和设备
                    .reshape(rows, cols)  # 重塑形状
                    .transpose(-1, -2)  # 转置
                    .contiguous()  # 确保连续
                )
                w13_dequant_list.append(dequant)  # 添加到列表

            w13_full = torch.stack(w13_dequant_list, dim=0)  # 堆叠所有专家权重

            layer.register_buffer("w13_dequant", w13_full, persistent=False)  # 注册w13缓冲区
        else:  # w13无需反量化
            layer.register_buffer("w13_dequant", w13_qweight.data, persistent=False)  # 直接使用原始数据

        # Pre-dequantize w2 weights (down projection)  # 预反量化w2权重（下投影）
        w2_qweight = layer.w2_qweight  # 获取w2权重
        w2_qtype = layer.w2_qweight_type.weight_type  # 获取w2类型

        if w2_qtype not in UNQUANTIZED_TYPES:  # 如果w2需要反量化
            num_experts = w2_qweight.shape[0]  # 获取专家数量
            w2_dequant_list = []  # w2反量化列表

            block_size, type_size = gguf.GGML_QUANT_SIZES[w2_qtype]  # 获取块和类型大小

            for e in range(num_experts):  # 遍历每个专家
                qweight_cpu = w2_qweight[e].cpu().numpy()  # 转到CPU
                rows = w2_qweight[e].shape[0]  # 行数
                cols = w2_qweight[e].shape[1] // type_size * block_size  # 列数

                dequant_np = gguf_dequantize(qweight_cpu.flatten(), w2_qtype)  # 反量化
                dequant = (  # 转换为torch张量
                    torch.from_numpy(dequant_np)
                    .to(dtype=self.params_dtype, device=w2_qweight.device)  # 转换类型和设备
                    .reshape(rows, cols)  # 重塑形状
                    .transpose(-1, -2)  # 转置
                    .contiguous()  # 确保连续
                )
                w2_dequant_list.append(dequant)  # 添加到列表

            w2_full = torch.stack(w2_dequant_list, dim=0)  # 堆叠所有专家权重

            layer.register_buffer("w2_dequant", w2_full, persistent=False)  # 注册w2缓冲区
        else:  # w2无需反量化
            layer.register_buffer("w2_dequant", w2_qweight.data, persistent=False)  # 直接使用原始数据

        if hasattr(layer, "w2_qweight"):  # 如果有w2_qweight属性
            del layer.w2_qweight  # 删除以节省内存
        if hasattr(layer, "w13_qweight"):  # 如果有w13_qweight属性
            del layer.w13_qweight  # 删除以节省内存

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 层对象和配置
    ):
        """创建并保存MoE运行器配置。"""  # 中文函数说明
        self.moe_runner_config = moe_runner_config  # 保存配置

    def apply(  # 应用MoE推理
        self,
        layer: torch.nn.Module,  # 神经网络层
        dispatch_output: StandardDispatchOutput,  # 分发输出
    ) -> CombineInput:  # 返回合并输入
        """Apply MoE forward pass on NPU using npu_grouped_matmul for maximum performance."""  # 在NPU上使用npu_grouped_matmul执行MoE前向推理
        from sglang.srt.distributed.communication_op import (  # 导入通信操作
            tensor_model_parallel_all_gather,  # 张量并行全收集
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取Top-K输出
        topk_weights, topk_ids, _ = topk_output  # 解包Top-K结果

        # Check if pre-dequantized weights are available  # 检查预反量化权重是否可用
        use_pre_dequant = hasattr(layer, "w13_dequant") and hasattr(layer, "w2_dequant")  # 检查属性

        if not use_pre_dequant:  # 如果没有预反量化权重
            raise RuntimeError(  # 抛出运行时错误
                "GGUF MoE on NPU requires pre-dequantization (FusedMoE fix). Please report if this occurs."
            )

        w13 = layer.w13_dequant  # 获取w13反量化权重
        w2 = layer.w2_dequant  # 获取w2反量化权重

        num_experts = w13.shape[0]  # 获取专家数量

        tp_size = getattr(layer, "moe_tp_size", 1)  # 获取张量并行大小

        original_dtype = x.dtype  # 保存原始数据类型
        num_tokens = x.shape[0]  # 获取token数
        top_k = topk_ids.shape[1]  # 获取Top-K值

        # Ensure correct dtypes for NPU ops  # 确保NPU操作的数据类型正确
        topk_ids = topk_ids.to(torch.int32)  # 转换为int32
        topk_weights = topk_weights.to(x.dtype)  # 转换为输入数据类型

        #  MoE routing initialization - reorder tokens by expert  # MoE路由初始化 - 按专家重排token
        row_idx_len = num_tokens * top_k  # 行索引长度
        row_idx = (  # 创建行索引
            torch.arange(0, row_idx_len, dtype=torch.int32, device=x.device)  # 生成索引序列
            .view(top_k, -1)  # 重塑形状
            .permute(1, 0)  # 转置
            .contiguous()  # 确保连续
        )

        sorted_hidden_states, expanded_row_idx, expanded_expert_idx = (  # NPU MoE初始化路由
            torch.ops.npu.npu_moe_init_routing(
                x, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
            )
        )

        # Compute tokens per expert  # 计算每个专家的token数
        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(  # NPU MoE计算专家token数
            expanded_expert_idx, num_experts
        )
        expert_tokens = expert_tokens.to(torch.int64)  # 转换为int64

        w13_gmm = w13  # No transpose needed  # 不需要转置

        hidden_states = torch.ops.npu.npu_grouped_matmul(  # NPU分组矩阵乘法（w13）
            x=[sorted_hidden_states],  # 输入
            weight=[w13_gmm],  # 权重
            split_item=2,  # 分割项
            group_list_type=0,  # 分组列表类型
            group_type=0,  # 分组类型
            group_list=expert_tokens,  # 每个专家的token数
            output_dtype=original_dtype,  # 输出数据类型
        )[0]  # 取第一个结果

        #  Activation (SwiGLU)  # 激活函数（SwiGLU）
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)  # 执行SwiGLU

        # TP all-gather for intermediate dimension if needed  # 如果需要则对中间维度进行TP全收集
        if tp_size > 1:  # 如果张量并行度大于1
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=-1)  # 全收集

        w2_gmm = w2  # w2权重

        hidden_states = torch.ops.npu.npu_grouped_matmul(  # NPU分组矩阵乘法（w2）
            x=[hidden_states],  # 输入
            weight=[w2_gmm],  # 权重
            split_item=2,  # 分割项
            group_list_type=0,  # 分组列表类型
            group_type=0,  # 分组类型
            group_list=expert_tokens,  # 每个专家的token数
            output_dtype=original_dtype,  # 输出数据类型
        )[0]  # 取第一个结果

        # Finalize routing - reorder back and apply weights  # 完成路由 - 重排回原始顺序并应用权重
        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(  # NPU MoE路由完成
            hidden_states,  # 隐藏状态
            skip1=None,  # 跳跃连接1
            skip2=None,  # 跳跃连接2
            bias=None,  # 偏置
            scales=topk_weights,  # Top-K权重
            expanded_src_to_dst_row=expanded_row_idx,  # 扩展行索引
            export_for_source_row=topk_ids,  # 源行专家ID
        )

        if tp_size > 1:  # 如果张量并行度大于1
            final_hidden_states = tensor_model_parallel_all_gather(  # 全收集
                final_hidden_states, dim=-1
            )

        # Ensure output matches input dtype  # 确保输出与输入类型一致
        final_hidden_states = final_hidden_states.to(dtype=original_dtype)  # 转换数据类型

        return StandardCombineInput(hidden_states=final_hidden_states)  # 返回标准合并输入


class GGUFEmbeddingAscendMethod(GGUFLinearAscendMethod):  # GGUF嵌入Ascend方法类
    """Embedding method for GGUF on Ascend NPU."""  # Ascend NPU上的GGUF嵌入方法

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:  # 嵌入查找函数
        """使用预反量化的权重执行NPU上的嵌入查找。"""  # 中文函数说明
        return torch.embedding(layer.dequantized_weight, x)  # 直接使用反量化权重执行嵌入
