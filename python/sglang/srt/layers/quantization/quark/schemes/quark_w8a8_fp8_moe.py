# SPDX-License-Identifier: Apache-2.0
# Quark W8A8 FP8 MoE量化方案实现文件，实现了MoE（混合专家）层的FP8格式量化

from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 导入日志模块
from typing import TYPE_CHECKING, Any  # 导入类型检查工具

import torch  # 导入PyTorch库

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig  # 导入MoE运行器相关类
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo  # 导入Triton MoE量化信息类
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz, scaled_fp8_quant  # 导入FP8内核检测和缩放FP8量化函数
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz  # 导入FP8格式归一化函数
from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme  # 导入Quark MoE量化方案基类
from sglang.srt.layers.quantization.utils import all_close_1d, per_tensor_dequantize  # 导入量化工具函数
from sglang.srt.utils import get_bool_env_var, is_hip, set_weight_attrs  # 导入工具函数

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入token分发器类型
        CombineInput,  # 合并输入类型
        StandardDispatchOutput,  # 标准分发输出类型
    )

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

__all__ = ["QuarkW8A8FP8MoE"]  # 模块公开导出的类列表

_is_fp8_fnuz = is_fp8_fnuz()  # 检测是否使用FP8 FNUZ格式
_is_hip = is_hip()  # 检测当前是否为HIP平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用aiter库（需HIP平台且环境变量启用）
if _use_aiter:  # 如果使用aiter
    from aiter.ops.shuffle import shuffle_weight  # 导入权重打乱操作

    from sglang.srt.layers.moe.rocm_moe_utils import rocm_fused_experts_tkw1  # 导入ROCm融合专家计算函数


class QuarkW8A8FP8MoE(QuarkMoEScheme):  # Quark W8A8 FP8 MoE量化方案类，继承自QuarkMoEScheme

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):  # 初始化方法
        self.is_static_input_scheme: bool = False  # 是否使用静态输入量化方案，默认为False
        self.input_qscheme = None  # 输入量化方案，默认为None

        if input_config is not None:  # 如果输入配置存在
            self.is_static_input_scheme = not input_config.get("is_dynamic")  # 判断是否为静态输入方案
            self.input_qscheme = input_config.get("qscheme")  # 获取输入量化方案

        self.input_per_token = (  # 是否使用逐token输入量化
            not self.is_static_input_scheme and self.input_qscheme == "per_channel"  # 非静态方案且按通道量化时启用
        )
        self.weight_qscheme = weight_config.get("qscheme")  # 获取权重量化方案
        self.is_weight_per_channel = self.weight_qscheme == "per_channel"  # 是否按通道量化权重
        self.out_dtype = torch.get_default_dtype()  # 获取默认输出数据类型

    @classmethod  # 类方法装饰器
    def get_min_capability(cls) -> int:  # 获取最低设备计算能力要求
        # lovelace and up  # Lovelace架构（RTX 40系列）及以上
        return 89  # 最低计算能力为8.9

    def create_weights(  # 创建权重参数的方法
        self,
        layer: torch.nn.Module,  # 目标层模块
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外的权重属性
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入MoE权重缩放支持类型

        params_dtype = torch.float8_e4m3fn  # 参数数据类型设为FP8 E4M3格式

        # WEIGHTS  # 权重
        w13_weight = torch.nn.Parameter(  # 创建w13（门控+上投影）权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                2 * intermediate_size_per_partition,  # 中间维度的两倍（门控和上投影拼接）
                hidden_size,  # 隐藏维度
                dtype=params_dtype,  # 数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight)  # 注册w13权重参数到层
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置w13权重的额外属性

        w2_weight = torch.nn.Parameter(  # 创建w2（下投影）权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏维度
                intermediate_size_per_partition,  # 中间维度
                dtype=params_dtype,  # 数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight", w2_weight)  # 注册w2权重参数到层
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置w2权重的额外属性

        # WEIGHT_SCALES  # 权重缩放因子
        # per-tensor quantization  # 按张量量化
        if self.weight_qscheme == "per_tensor":  # 如果按张量量化
            # Allocate 2 scales for w1 and w3 respectively.  # 为w1和w3分别分配2个缩放因子
            # They will be combined to a single scale after weight loading.  # 权重加载后它们将被合并为一个缩放因子
            w13_weight_scale = torch.nn.Parameter(  # 创建w13权重缩放因子参数
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False  # 每个专家2个缩放值
            )
            w2_weight_scale = torch.nn.Parameter(  # 创建w2权重缩放因子参数
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False  # 每个专家1个缩放值
            )
            weight_quant_method = FusedMoeWeightScaleSupported.TENSOR.value  # 量化方法为按张量
        elif self.weight_qscheme == "per_channel":  # 如果按通道量化
            w13_weight_scale = torch.nn.Parameter(  # 创建w13权重缩放因子参数
                torch.ones(  # 创建全1张量
                    num_experts,  # 专家数量维度
                    2 * intermediate_size_per_partition,  # 每个通道一个缩放值
                    dtype=torch.float32,  # 数据类型
                ),
                requires_grad=False,  # 不需要梯度
            )
            w2_weight_scale = torch.nn.Parameter(  # 创建w2权重缩放因子参数
                torch.ones(num_experts, hidden_size, dtype=torch.float32),  # 每个通道一个缩放值
                requires_grad=False,  # 不需要梯度
            )
            weight_quant_method = FusedMoeWeightScaleSupported.CHANNEL.value  # 量化方法为按通道
        else:
            raise ValueError(  # 不支持的权重量化策略
                f"Unsupported weight quantization strategy: {self.weight_qscheme}."
            )

        layer.register_parameter("w13_weight_scale", w13_weight_scale)  # 注册w13缩放因子到层
        layer.register_parameter("w2_weight_scale", w2_weight_scale)  # 注册w2缩放因子到层
        # Add the quantization method used (per tensor/grouped/channel)  # 添加使用的量化方法（按张量/按组/按通道）
        # to ensure the weight scales are loaded in properly  # 以确保权重缩放因子正确加载
        extra_weight_attrs.update({"quant_method": weight_quant_method})  # 更新量化方法属性
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)  # 设置w13缩放因子的额外属性
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)  # 设置w2缩放因子的额外属性

        # INPUT_SCALES  # 输入缩放因子
        if self.is_static_input_scheme:  # 如果使用静态输入方案
            assert (
                self.input_qscheme == "per_tensor"
            ), "Only per-tensor quantization is supported for static input scales"  # 静态输入缩放仅支持按张量量化
            w13_input_scale = torch.nn.Parameter(  # 创建w13输入缩放因子参数
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False  # 每个专家一个缩放值
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)  # 注册w13输入缩放因子到层
            set_weight_attrs(w13_input_scale, extra_weight_attrs)  # 设置w13输入缩放因子的额外属性

            w2_input_scale = torch.nn.Parameter(  # 创建w2输入缩放因子参数
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False  # 每个专家一个缩放值
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)  # 注册w2输入缩放因子到层
            set_weight_attrs(w2_input_scale, extra_weight_attrs)  # 设置w2输入缩放因子的额外属性
        else:
            layer.w13_input_scale = None  # 动态方案不设w13输入缩放因子
            layer.w2_input_scale = None  # 动态方案不设w2输入缩放因子

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的后处理方法
        # Fp8 moe kernels require a single activation scale.  # FP8 MoE内核需要单一的激活缩放因子
        # We take the max of all the scales in case they differ.  # 如果缩放因子不同，我们取所有缩放因子的最大值
        if self.is_static_input_scheme:  # 如果使用静态输入方案
            if layer.w13_input_scale is None or layer.w2_input_scale is None:  # 检查缩放因子是否存在
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )  # 量化配置为静态量化，但激活缩放因子为None
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(  # 检查各专家的缩放因子是否一致
                layer.w2_input_scale
            ):
                logger.warning(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer."
                )  # 发现FP8 MoE层的输入缩放因子不一致，使用每层的最大值
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )  # 取w13输入缩放因子最大值
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )  # 取w2输入缩放因子最大值

        if _is_fp8_fnuz:  # 如果是FP8 FNUZ格式
            # Normalize the weights and scales  # 归一化权重和缩放因子
            w13_weight, w13_weight_scale, w13_input_scale = (  # 归一化w13权重和缩放因子
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                )
            )
            w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(  # 归一化w2权重和缩放因子
                layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
            )
            # Reset the parameter  # 重置参数
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)  # 重置w13权重
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )  # 重置w13缩放因子
            if w13_input_scale is not None:  # 如果w13输入缩放因子存在
                layer.w13_input_scale = torch.nn.Parameter(
                    w13_input_scale, requires_grad=False
                )  # 重置w13输入缩放因子
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)  # 重置w2权重
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )  # 重置w2缩放因子
            if w2_input_scale is not None:  # 如果w2输入缩放因子存在
                layer.w2_input_scale = torch.nn.Parameter(
                    w2_input_scale, requires_grad=False
                )  # 重置w2输入缩放因子
        if self.weight_qscheme == "per_tensor":  # 如果按张量量化
            # Fp8 moe kernel needs single weight scale for w13 per expert.  # FP8 MoE内核需要每个专家w13的单一权重缩放因子
            # We take the max then dequant and requant each expert.  # 取最大值然后对每个专家进行反量化和重新量化
            assert layer.w13_weight_scale is not None  # 断言w13缩放因子存在
            shard_size = layer.intermediate_size_per_partition  # 获取分片大小
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values  # 取每个专家w13缩放因子的最大值
            for expert_id in range(layer.num_local_experts):  # 遍历每个本地专家
                start = 0  # 起始位置
                for shard_id in range(2):  # 遍历两个分片（w1和w3）
                    dq_weight = per_tensor_dequantize(  # 按张量反量化权重
                        layer.w13_weight[expert_id][start : start + shard_size, :],  # 当前分片的权重
                        layer.w13_weight_scale[expert_id][shard_id],  # 当前分片的缩放因子
                    )
                    (
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        _,
                    ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])  # 使用最大缩放因子重新量化

                    start += shard_size  # 更新起始位置

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )  # 用最大缩放因子替换w13缩放因子
        elif self.weight_qscheme == "per_channel":  # 如果按通道量化
            layer.w13_weight_scale = torch.nn.Parameter(
                layer.w13_weight_scale.unsqueeze(-1), requires_grad=False
            )  # 在w13缩放因子末尾增加一个维度
            layer.w2_weight_scale = torch.nn.Parameter(
                layer.w2_weight_scale.unsqueeze(-1), requires_grad=False
            )  # 在w2缩放因子末尾增加一个维度
        else:
            raise ValueError(
                f"Unsupported weight quantization strategy: {self.weight_qscheme}."
            )  # 不支持的权重量化策略

        if (  # 如果满足以下条件则预打乱权重
            _use_aiter  # 使用aiter
            and self.is_weight_per_channel  # 按通道量化权重
            and self.moe_runner_config.apply_router_weight_on_input  # 在输入上应用路由权重
        ):
            with torch.no_grad():  # 不计算梯度
                # Pre-shuffle weights  # 预打乱权重
                layer.w13_weight = torch.nn.Parameter(
                    shuffle_weight(layer.w13_weight.data, (16, 16)),  # 打乱w13权重
                    requires_grad=False,
                )
                torch.cuda.empty_cache()  # 清空CUDA缓存释放内存
                layer.w2_weight = torch.nn.Parameter(
                    shuffle_weight(layer.w2_weight.data, (16, 16)),  # 打乱w2权重
                    requires_grad=False,
                )
                torch.cuda.empty_cache()  # 清空CUDA缓存释放内存

    def create_moe_runner(  # 创建MoE运行器的方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)  # 创建Triton后端的MoE运行器

    def apply_weights(  # 应用权重执行前向传播的方法
        self,
        layer: torch.nn.Module,  # 目标层模块
        dispatch_output: StandardDispatchOutput,  # 分发输出
    ) -> CombineInput:  # 返回合并输入

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入类型

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取top-k路由输出

        moe_runner_config = self.moe_runner_config  # 获取MoE运行器配置

        if (  # 如果满足aiter优化条件
            _use_aiter  # 使用aiter
            and self.is_weight_per_channel  # 按通道量化权重
            and moe_runner_config.apply_router_weight_on_input  # 在输入上应用路由权重
        ):
            topk_weights, topk_ids, _ = topk_output  # 解包top-k输出
            output = rocm_fused_experts_tkw1(  # 使用ROCm融合专家计算
                hidden_states=x,  # 隐藏状态
                w1=layer.w13_weight,  # w13权重
                w2=layer.w2_weight,  # w2权重
                topk_weights=topk_weights,  # top-k权重
                topk_ids=topk_ids,  # top-k索引
                activation=moe_runner_config.activation,  # 激活函数
                apply_router_weight_on_input=moe_runner_config.apply_router_weight_on_input,  # 是否在输入上应用路由权重
                use_fp8_w8a8=True,  # 使用FP8 W8A8
                per_channel_quant=self.is_weight_per_channel,  # 是否按通道量化
                w1_scale=layer.w13_weight_scale,  # w13缩放因子
                w2_scale=layer.w2_weight_scale,  # w2缩放因子
                a1_scale=layer.w13_input_scale,  # w13输入缩放因子
                a2_scale=layer.w2_input_scale,  # w2输入缩放因子
            )
            return StandardCombineInput(hidden_states=output)  # 返回标准合并输入
        else:
            quant_info = TritonMoeQuantInfo(  # 创建Triton MoE量化信息
                w13_weight=layer.w13_weight,  # w13权重
                w2_weight=layer.w2_weight,  # w2权重
                use_fp8_w8a8=True,  # 使用FP8 W8A8
                per_channel_quant=self.is_weight_per_channel,  # 是否按通道量化
                w13_scale=layer.w13_weight_scale,  # w13缩放因子
                w2_scale=layer.w2_weight_scale,  # w2缩放因子
                a13_scale=layer.w13_input_scale,  # w13输入缩放因子
                a2_scale=layer.w2_input_scale,  # w2输入缩放因子
            )
            return self.runner.run(dispatch_output, quant_info)  # 使用Triton运行器执行MoE计算
