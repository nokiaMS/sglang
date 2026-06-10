# SPDX-License-Identifier: Apache-2.0 # SPDX许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project # SPDX版权声明
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kv_cache.py # 改编自vLLM项目的KV缓存量化模块

# KV缓存量化模块：提供KV缓存的量化方法基类，支持FP8格式的k_scale和v_scale缩放因子加载与处理

import logging  # 导入日志模块 # 导入Python标准日志模块

import torch  # 导入PyTorch库 # 导入PyTorch深度学习框架

from sglang.srt.layers.quantization.base_config import (  # 从基础配置模块导入量化基类 # 从量化基础配置模块导入基类
    QuantizationConfig,  # 量化配置基类 # 量化配置基类
    QuantizeMethodBase,  # 量化方法基类 # 量化方法基类
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz  # 导入FP8 FNUZ格式检测函数 # 导入FP8 FNUZ格式检测函数

logger = logging.getLogger(__name__)  # 创建日志记录器 # 创建当前模块的日志记录器


class BaseKVCacheMethod(QuantizeMethodBase):  # KV缓存量化方法基类，继承自QuantizeMethodBase # KV缓存量化方法基类
    """
    Quant method that adds `k_scale` and `v_scale` attributes to the # 为Attention层添加k_scale和v_scale属性的量化方法
    Attention layer to support loading those scaling factors from checkpoints. # 以支持从检查点加载这些缩放因子
    The k/v_scale will be used to: # k/v_scale将用于：
        - quantize k/v_cache entries before saving them to the cache # 在保存到缓存之前量化k/v_cache条目
        - dequantize k/v_cache entries before fetching them from the cache # 在从缓存获取时反量化k/v_cache条目

    :param quant_config: the appropriate QuantizationConfig # 参数quant_config：对应的量化配置
    """

    def __init__(self, quant_config: QuantizationConfig):  # 初始化方法 # 初始化KV缓存量化方法
        self.quant_config = quant_config  # 保存量化配置 # 保存量化配置对象

    def create_weights(self, layer: torch.nn.Module):  # 创建权重（即k_scale和v_scale） # 为注意力层创建k_scale和v_scale缩放因子
        """
        Create "weight" (aka k_scale and v_scale) for an attention layer. # 为注意力层创建"权重"（即k_scale和v_scale）
        """
        # Initialize the KV cache scales to -1.0, which is an invalid value. # 将KV缓存缩放因子初始化为-1.0，这是一个无效值
        # If the k/v_scale appears in the checkpoint, it will be # 如果k/v_scale出现在检查点中，它将被
        # overwritten when loading weights. # 在加载权重时覆盖
        layer.k_scale = torch.nn.Parameter(  # 创建k_scale参数 # 创建k缩放因子参数
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False  # 初始化为-1.0，不需要梯度 # 初始化为无效值-1.0，不需要梯度更新
        )
        layer.v_scale = torch.nn.Parameter(  # 创建v_scale参数 # 创建v缩放因子参数
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False  # 初始化为-1.0，不需要梯度 # 初始化为无效值-1.0，不需要梯度更新
        )
        layer.k_scale._skip_weight_check = True  # 跳过k_scale权重检查 # 标记k_scale跳过权重检查
        layer.v_scale._skip_weight_check = True  # 跳过v_scale权重检查 # 标记v_scale跳过权重检查

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:  # 应用方法（不应被调用） # 应用方法，此方法不应被调用
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")  # 抛出运行时错误 # 抛出运行时异常，此方法不应被直接调用

    def process_weights_after_loading(self, layer) -> None:  # 加载权重后处理缩放因子 # 加载权重后处理k_scale和v_scale缩放因子
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:  # 两个缩放因子都有效 # 如果k_scale和v_scale都为正值（有效）
            # We prefer to use separate k_scale and v_scale if present # 如果存在独立的k_scale和v_scale，我们优先使用
            k_scale = layer.k_scale.to("cpu").tolist()  # 将k_scale转到CPU并转为Python数值 # 将k_scale移至CPU并转为Python标量
            v_scale = layer.v_scale.to("cpu").tolist()  # 将v_scale转到CPU并转为Python数值 # 将v_scale移至CPU并转为Python标量
            if is_fp8_fnuz():  # 如果是FP8 FNUZ格式 # 检测是否为FP8 FNUZ格式
                k_scale *= 2  # k_scale乘以2 # FNUZ格式需要将缩放因子乘以2
                v_scale *= 2  # v_scale乘以2 # FNUZ格式需要将缩放因子乘以2
        elif layer.k_scale < 0.0 and layer.v_scale < 0.0:  # 两个缩放因子都无效 # 如果k_scale和v_scale都为负值（无效）
            # If no scales were loaded (both scales are invalid negative # 如果没有加载缩放因子（两个缩放因子都是无效的负值
            # values), use the default value of 1.0 # 值），则使用默认值1.0
            k_scale = 1.0  # 默认k缩放因子 # 设置默认k缩放因子为1.0
            v_scale = 1.0  # 默认v缩放因子 # 设置默认v缩放因子为1.0
        else:  # 只有一个缩放因子有效 # 只有一个缩放因子有效的情况
            # If we find a single kv_scale in the checkpoint, we remap # 如果在检查点中找到单一的kv_scale，我们将其重映射
            # kv_scale to k_scale during weight loading, and duplicate # 在权重加载时将kv_scale映射为k_scale，并复制
            # k_scale to v_scale here # k_scale到v_scale
            assert layer.k_scale > 0.0  # 断言k_scale有效 # 确保k_scale为正值
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)  # 取较大的缩放因子 # 取两个缩放因子中较大的值
            k_scale = scale_to_duplicate.to("cpu").tolist()  # 将缩放因子转到CPU并转为Python数值 # 将缩放因子移至CPU并转为Python标量
            v_scale = scale_to_duplicate.to("cpu").tolist()  # 将缩放因子转到CPU并转为Python数值 # 将缩放因子移至CPU并转为Python标量
            if is_fp8_fnuz():  # 如果是FP8 FNUZ格式 # 检测是否为FP8 FNUZ格式
                k_scale *= 2  # k_scale乘以2 # FNUZ格式需要将缩放因子乘以2
                v_scale *= 2  # v_scale乘以2 # FNUZ格式需要将缩放因子乘以2

        if not isinstance(k_scale, float) or not isinstance(v_scale, float):  # 检查是否为标量 # 验证缩放因子是否为浮点标量
            raise ValueError(  # 抛出值错误 # 抛出数值异常
                "Only support per-tensor scaling factor " "for fp8 KV cache"  # 仅支持FP8 KV缓存的逐张量缩放因子 # 仅支持FP8 KV缓存的逐张量缩放因子
            )

        # These are used in the final Attention.forward() # 这些将在最终的Attention.forward()中使用
        layer.k_scale.copy_(k_scale)  # 将处理后的k_scale写回层参数 # 将处理后的k缩放因子写回层参数
        layer.v_scale.copy_(v_scale)  # 将处理后的v_scale写回层参数 # 将处理后的v缩放因子写回层参数
        layer.k_scale_float = k_scale  # 保存k_scale的浮点值 # 保存k_scale的浮点值以供后续使用
        layer.v_scale_float = v_scale  # 保存v_scale的浮点值 # 保存v_scale的浮点值以供后续使用
