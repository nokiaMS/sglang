# 压缩张量NV FP4 W4A4量化MoE方案模块
# 实现了CompressedTensorsW4A4Nvfp4MoE类，用于Blackwell架构上的
# W4A4 NVFP4量化MoE层推理，支持FlashInfer TRT-LLM和CUTLASS两种后端
from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 导入日志模块
from typing import TYPE_CHECKING  # 导入类型检查常量

import torch  # 导入PyTorch

from sglang.srt.distributed import get_tp_group  # 导入张量并行组获取函数
from sglang.srt.distributed.device_communicators.pynccl_allocator import (  # 导入对称内存使用上下文管理器
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric  # 导入对称分配检查函数
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig  # 导入MoE运行器、后端枚举和配置类
from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType  # 导入CUTLASS MoE参数和类型
from sglang.srt.layers.moe.utils import RoutingMethodType, get_moe_runner_backend  # 导入路由方法类型和MoE运行器后端获取函数
from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # 导入压缩张量MoE方案基类
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.fp8_utils import is_blackwell_supported  # 导入Blackwell架构支持检查函数
from sglang.srt.layers.quantization.utils import (  # 导入量化工具函数
    prepare_static_weights_for_trtllm_fp4_moe,  # 为TRT-LLM FP4 MoE准备静态权重
    reorder_w1w3_to_w3w1,  # 将w1w3顺序重排为w3w1
    replace_parameter,  # 替换参数
    swizzle_blockscale,  # 块缩放交错处理
)
from sglang.srt.utils import next_power_of_2, set_weight_attrs  # 导入下一个2的幂和权重属性设置工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

__all__ = ["CompressedTensorsW4A4Nvfp4MoE"]  # 模块公开导出的类列表

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入MoE token分发器类型
        CombineInput,
        StandardDispatchOutput,
    )


class CompressedTensorsW4A4Nvfp4MoE(CompressedTensorsMoEScheme):  # NV FP4 W4A4量化MoE方案类，继承自CompressedTensorsMoEScheme

    def __init__(self):  # 初始化方法
        if not is_blackwell_supported():  # 检查是否支持Blackwell架构
            raise ValueError(  # 若不支持则抛出异常
                "Current platform does not support NVFP4"
                " quantization. Please use Blackwell and"
                " above."  # 当前平台不支持NVFP4量化，请使用Blackwell及以上架构
            )
        self.group_size = 16  # 量化分组大小为16
        self.use_flashinfer_trtllm = get_moe_runner_backend().is_flashinfer_trtllm()  # 是否使用FlashInfer TRT-LLM后端

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低设备算力要求
        # Requires sm100(blackwell) architecture
        return 100  # 需要sm100(Blackwell)架构

    def create_weights(  # 创建权重参数
        self,
        layer: torch.nn.Module,  # 目标层模块
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入MoE权重缩放支持枚举

        layer.params_dtype = params_dtype  # 保存参数数据类型

        w13_weight = torch.nn.Parameter(  # 创建w13权重参数（gate+up投影FP4打包权重）
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                2 * intermediate_size_per_partition,  # 双倍中间层大小（gate和up拼接）
                # 2 fp4 items are packed in the input dimension
                # 2个FP4元素在输入维度上打包
                hidden_size // 2,  # 输入维度除以2（FP4打包）
                requires_grad=False,  # 不需要梯度
                dtype=torch.uint8,  # 使用uint8类型存储打包的FP4权重
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_packed", w13_weight)  # 注册w13打包权重参数
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置权重属性

        w2_weight = torch.nn.Parameter(  # 创建w2权重参数（down投影FP4打包权重）
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小维度
                # 2 fp4 items are packed in the input dimension
                # 2个FP4元素在输入维度上打包
                intermediate_size_per_partition // 2,  # 中间层维度除以2（FP4打包）
                dtype=torch.uint8,  # 使用uint8类型存储打包的FP4权重
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight_packed", w2_weight)  # 注册w2打包权重参数
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置权重属性

        # Weight Scales  # 权重缩放因子（逐分组）
        w13_weight_scale = torch.nn.Parameter(  # 创建w13权重缩放因子参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                2 * intermediate_size_per_partition,  # 双倍中间层大小
                # 2 fp4 items are packed in the input dimension
                # 2个FP4元素在输入维度上打包
                hidden_size // self.group_size,  # 按分组大小划分的维度
                dtype=torch.float8_e4m3fn,  # 使用FP8 E4M3类型存储缩放因子
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)  # 注册w13权重缩放因子
        extra_weight_attrs.update(  # 更新量化方法为分组量化
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)  # 设置权重属性

        w2_weight_scale = torch.nn.Parameter(  # 创建w2权重缩放因子参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小维度
                # 2 fp4 items are packed in the input dimension
                # 2个FP4元素在输入维度上打包
                intermediate_size_per_partition // self.group_size,  # 按分组大小划分的维度
                dtype=torch.float8_e4m3fn,  # 使用FP8 E4M3类型存储缩放因子
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)  # 注册w2权重缩放因子
        extra_weight_attrs.update(  # 更新量化方法为分组量化
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)  # 设置权重属性

        # Weight Global Scales  # 权重全局缩放因子（逐张量）
        w13_weight_scale_2 = torch.nn.Parameter(  # 创建w13权重全局缩放因子参数
            torch.empty(num_experts, 2, dtype=torch.float32), requires_grad=False  # 每个专家2个值（gate和up）
        )
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)  # 注册w13全局缩放因子
        extra_weight_attrs.update(  # 更新量化方法为逐张量量化
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)  # 设置权重属性

        w2_weight_scale_2 = torch.nn.Parameter(  # 创建w2权重全局缩放因子参数
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False  # 每个专家1个值
        )
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)  # 注册w2全局缩放因子
        extra_weight_attrs.update(  # 更新量化方法为逐张量量化
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)  # 设置权重属性

        # Input Global Scales  # 输入全局缩放因子
        w13_input_scale = torch.nn.Parameter(  # 创建w13输入全局缩放因子参数
            torch.empty(num_experts, 2, dtype=torch.float32), requires_grad=False  # 每个专家2个值（gate和up）
        )
        layer.register_parameter("w13_input_global_scale", w13_input_scale)  # 注册w13输入全局缩放因子
        extra_weight_attrs.update(  # 更新量化方法为逐张量量化
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_input_scale, extra_weight_attrs)  # 设置权重属性

        w2_input_scale = torch.nn.Parameter(  # 创建w2输入全局缩放因子参数
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False  # 每个专家1个值
        )
        layer.register_parameter("w2_input_global_scale", w2_input_scale)  # 注册w2输入全局缩放因子
        extra_weight_attrs.update(  # 更新量化方法为逐张量量化
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_input_scale, extra_weight_attrs)  # 设置权重属性

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的后处理方法
        # From packed to weight
        # 从打包权重转换为实际权重
        layer.w13_weight = torch.nn.Parameter(  # 将w13打包权重转为w13权重参数
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")  # 删除w13_weight_packed属性

        layer.w2_weight = torch.nn.Parameter(  # 将w2打包权重转为w2权重参数
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")  # 删除w2_weight_packed属性

        if self.use_flashinfer_trtllm:  # 若使用FlashInfer TRT-LLM后端
            w, s = reorder_w1w3_to_w3w1(  # 将w1w3顺序重排为w3w1
                layer.w13_weight.data, layer.w13_weight_scale.data, dim=-2  # 在倒数第二维上重排
            )
            layer.w13_weight = torch.nn.Parameter(w, requires_grad=False)  # 替换为重排后的权重
            layer.w13_weight_scale = torch.nn.Parameter(s, requires_grad=False)  # 替换为重排后的缩放因子

        if not torch.allclose(  # 检查gate和up的全局缩放因子是否一致
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            logger.warning_once(  # 不一致则发出警告
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected."  # w1全局缩放必须与w3匹配，否则可能影响精度
            )

        # Take inverse of global scale saved to disk
        # 取保存到磁盘的全局缩放因子的倒数
        layer.w13_weight_scale_2 = torch.nn.Parameter(  # 计算w13全局缩放因子的倒数
            1 / layer.w13_weight_global_scale[:, 0], requires_grad=False
        )

        layer.w2_weight_scale_2 = torch.nn.Parameter(  # 计算w2全局缩放因子的倒数
            1 / layer.w2_weight_global_scale.data, requires_grad=False
        )

        # w13  # 处理w13的输入缩放因子和alpha
        if self.use_flashinfer_trtllm:  # 若使用FlashInfer TRT-LLM后端
            w13_input_global_scale = (  # 取w13输入全局缩放因子的最小值
                layer.w13_input_global_scale.min()
                .to(torch.float32)
                .expand(layer.num_local_experts)  # 扩展到本地专家数量
            )
        else:  # 否则使用CUTLASS后端
            w13_input_global_scale = layer.w13_input_global_scale.min(dim=1).values.to(
                torch.float32  # 逐专家取最小值
            )
        layer.g1_alphas = torch.nn.Parameter(  # 计算g1_alpha = (1/input_global_scale) * weight_scale_2
            ((1 / w13_input_global_scale) * layer.w13_weight_scale_2),
            requires_grad=False,
        )

        layer.w13_input_scale_quant = torch.nn.Parameter(  # 保存w13输入缩放因子用于量化
            (w13_input_global_scale), requires_grad=False
        )

        # w2  # 处理w2的输入缩放因子和alpha
        if self.use_flashinfer_trtllm:  # 若使用FlashInfer TRT-LLM后端
            w2_input_global_scale = (  # 取w2输入全局缩放因子的最小值
                layer.w2_input_global_scale.min()
                .to(torch.float32)
                .expand(layer.num_local_experts)  # 扩展到本地专家数量
            )
        else:  # 否则使用CUTLASS后端
            w2_input_global_scale = layer.w2_input_global_scale  # 直接使用原始缩放因子

        layer.g2_alphas = torch.nn.Parameter(  # 计算g2_alpha = (1/input_global_scale) * weight_scale_2
            ((1 / w2_input_global_scale) * layer.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )

        layer.w2_input_scale_quant = torch.nn.Parameter(  # 保存w2输入缩放因子用于量化
            (w2_input_global_scale), requires_grad=False
        )

        # TensorRT-LLM specific processing  # TensorRT-LLM特定处理
        if self.use_flashinfer_trtllm:  # 若使用FlashInfer TRT-LLM后端
            # Prepare static weights for TRT-LLM kernel
            # 为TRT-LLM内核准备静态权重
            (
                gemm1_weights_fp4_shuffled,  # 重排后的gemm1 FP4权重
                gemm1_scales_fp4_shuffled,  # 重排后的gemm1缩放因子
                gemm2_weights_fp4_shuffled,  # 重排后的gemm2 FP4权重
                gemm2_scales_fp4_shuffled,  # 重排后的gemm2缩放因子
            ) = prepare_static_weights_for_trtllm_fp4_moe(  # 调用TRT-LLM FP4 MoE静态权重准备函数
                layer.w13_weight,  # w13权重
                layer.w2_weight,  # w2权重
                layer.w13_weight_scale,  # w13缩放因子
                layer.w2_weight_scale,  # w2缩放因子
                layer.w2_weight.size(-2),  # hidden_size  # 隐藏层大小
                layer.w13_weight.size(-2) // 2,  # intermediate_size  # 中间层大小
                layer.w13_weight.size(0),  # num_experts  # 专家数量
            )
            logger.debug("Finished shuffling weights for TRT-LLM MOE")  # 日志：完成TRT-LLM MoE权重重排

            replace_parameter(layer, "w13_weight", gemm1_weights_fp4_shuffled)  # 替换w13权重
            replace_parameter(layer, "w2_weight", gemm2_weights_fp4_shuffled)  # 替换w2权重
            replace_parameter(layer, "w13_weight_scale", gemm1_scales_fp4_shuffled)  # 替换w13缩放因子
            replace_parameter(layer, "w2_weight_scale", gemm2_scales_fp4_shuffled)  # 替换w2缩放因子

            # Additional parameter needed for TRT-LLM
            # TRT-LLM所需的额外参数
            layer.g1_scale_c = torch.nn.Parameter(  # 计算g1_scale_c = w2_input_scale * g1_alphas
                (layer.w2_input_scale_quant * layer.g1_alphas).to(torch.float32),
                requires_grad=False,
            )
        else:  # 否则使用CUTLASS后端
            # swizzle weight scales
            # 对权重缩放因子进行交错处理
            layer.w13_weight_scale = torch.nn.Parameter(
                swizzle_blockscale(layer.w13_weight_scale), requires_grad=False  # 交错w13缩放因子
            )

            layer.w2_weight_scale = torch.nn.Parameter(
                swizzle_blockscale(layer.w2_weight_scale), requires_grad=False  # 交错w2缩放因子
            )

            layer.cutlass_moe_params = CutlassMoEParams(  # 创建CUTLASS MoE参数对象
                CutlassMoEType.BlockscaledFP4,  # 使用块缩放FP4类型
                layer.w13_weight.device,  # 设备
                num_experts=layer.num_experts,  # 专家数量
                intermediate_size_per_partition=layer.w2_weight.shape[2] * 2,  # 中间层大小（解包FP4）
                hidden_size=layer.w13_weight.shape[2] * 2,  # 隐藏层大小（解包FP4）
            )

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)  # 创建Triton后端的MoE运行器

    def apply_weights(  # 应用权重，执行MoE前向传播
        self,
        layer: torch.nn.Module,  # 目标层模块
        dispatch_output: StandardDispatchOutput,  # 分发输出
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入类

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取topk输出

        if self.use_flashinfer_trtllm:  # 若使用FlashInfer TRT-LLM后端
            from flashinfer import trtllm_fp4_block_scale_moe  # 导入TRT-LLM FP4块缩放MoE内核

            from sglang.srt.layers.quantization.fp4_utils import fp4_quantize  # 导入FP4量化函数

            router_logits = topk_output.router_logits  # 获取路由器logits
            topk_config = topk_output.topk_config  # 获取topk配置

            # global_scale must be shape [1] (strict in cute-dsl backend).
            # global_scale的形状必须为[1]（cute-dsl后端的严格要求）。
            hs_fp4_bytes, hs_sf_bytes = fp4_quantize(  # 对输入进行FP4量化
                x,  # 输入张量
                layer.w13_input_scale_quant[:1],  # 仅使用第一个输入缩放因子（形状为[1]）
                self.group_size,  # sf_vec_size  # 缩放因子向量大小
                False,  # use_ue8m0  # 不使用UE8M0格式
                False,  # is_sf_swizzled_layout  # 缩放因子非交错布局
            )
            hs_fp4 = hs_fp4_bytes.reshape(x.shape[0], x.shape[1] // 2)  # 重塑FP4量化结果
            hs_scale = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(  # 重塑缩放因子结果
                *hs_sf_bytes.shape[:-1], -1  # 保留前面的维度，最后一维展平
            )

            correction_bias = (  # 获取校正偏置
                None
                if topk_config.correction_bias is None  # 若配置中无校正偏置则为None
                else topk_config.correction_bias.to(x.dtype)  # 否则转为输入数据类型
            )

            assert layer.routing_method_type is not None  # 断言路由方法类型不为空

            # DeepSeekV3 style routing requires float32 router logits
            # DeepSeekV3风格路由需要float32路由器logits
            if layer.routing_method_type == RoutingMethodType.DeepSeekV3:  # 若为DeepSeekV3路由方法
                router_logits = router_logits.to(torch.float32)  # 将路由器logits转为float32

            routed_scaling_factor = self.moe_runner_config.routed_scaling_factor  # 获取路由缩放因子
            routed_scaling_factor = (  # 若缩放因子为None则设为1.0
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            )

            with use_symmetric_memory(  # 使用对称内存上下文
                get_tp_group(), disabled=not is_allocation_symmetric()  # 若非对称分配则禁用
            ):
                num_tokens = hs_fp4.shape[0]  # 获取token数量
                hidden_size = (  # 计算隐藏层大小
                    hs_fp4.shape[-1] * 2  # FP4打包，需要乘2
                    if hs_fp4.dtype == torch.uint8  # 若为uint8则表示打包的FP4
                    else hs_fp4.shape[-1]  # 否则直接使用
                )
                symm_output = torch.empty(  # 预分配对称内存输出张量
                    num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_fp4.device  # 使用bfloat16类型
                )

            output = trtllm_fp4_block_scale_moe(  # 调用TRT-LLM FP4块缩放MoE内核
                routing_logits=router_logits,  # 路由器logits
                routing_bias=correction_bias,  # 路由偏置
                hidden_states=hs_fp4,  # 量化后的隐藏状态
                hidden_states_scale=hs_scale,  # 隐藏状态缩放因子
                gemm1_weights=layer.w13_weight,  # gemm1权重
                gemm1_weights_scale=layer.w13_weight_scale.view(torch.float8_e4m3fn),  # gemm1缩放因子
                gemm1_bias=None,  # gemm1偏置（未使用）
                gemm1_alpha=None,  # gemm1 alpha（未使用）
                gemm1_beta=None,  # gemm1 beta（未使用）
                gemm1_clamp_limit=None,  # gemm1截断限制（未使用）
                gemm2_weights=layer.w2_weight,  # gemm2权重
                gemm2_weights_scale=layer.w2_weight_scale.view(torch.float8_e4m3fn),  # gemm2缩放因子
                gemm2_bias=None,  # gemm2偏置（未使用）
                output1_scale_scalar=layer.g1_scale_c,  # gemm1输出缩放标量
                output1_scale_gate_scalar=layer.g1_alphas,  # gemm1输出门控缩放标量
                output2_scale_scalar=layer.g2_alphas,  # gemm2输出缩放标量
                num_experts=layer.num_experts,  # 专家总数
                top_k=topk_config.top_k,  # top-k值
                n_group=topk_config.num_expert_group,  # 专家分组数
                topk_group=topk_config.topk_group,  # 每组选择的专家数
                intermediate_size=layer.intermediate_size_per_partition,  # 中间层大小
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,  # 本地专家偏移
                local_num_experts=layer.num_local_experts,  # 本地专家数量
                routed_scaling_factor=routed_scaling_factor,  # 路由缩放因子
                routing_method_type=layer.routing_method_type,  # 路由方法类型
                do_finalize=True,  # 执行最终化步骤
                tune_max_num_tokens=next_power_of_2(hs_fp4.shape[0]),  # 调优最大token数（取下一个2的幂）
                output=symm_output,  # 输出张量
            )[0]  # 取内核返回结果的第一个元素
        else:  # 否则使用CUTLASS后端
            from sglang.srt.layers.moe.cutlass_moe import cutlass_moe_fp4  # 导入CUTLASS FP4 MoE函数

            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids  # 获取topk权重和ID

            output = cutlass_moe_fp4(  # 调用CUTLASS FP4 MoE函数
                a=x,  # 输入张量
                a1_gscale=layer.w13_input_scale_quant,  # gemm1输入全局缩放因子
                w1_fp4=layer.w13_weight,  # gemm1 FP4权重
                w1_blockscale=layer.w13_weight_scale,  # gemm1块缩放因子
                w1_alphas=layer.g1_alphas,  # gemm1 alpha参数
                a2_gscale=layer.w2_input_scale_quant,  # gemm2输入全局缩放因子
                w2_fp4=layer.w2_weight,  # gemm2 FP4权重
                w2_blockscale=layer.w2_weight_scale,  # gemm2块缩放因子
                w2_alphas=layer.g2_alphas,  # gemm2 alpha参数
                topk_weights=topk_weights,  # topk权重
                topk_ids=topk_ids,  # topk ID
                params=layer.cutlass_moe_params,  # CUTLASS MoE参数
                apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,  # 是否在输入上应用路由权重
            ).to(x.dtype)  # 转为输入数据类型

        return StandardCombineInput(hidden_states=output)  # 返回标准合并输入，包含输出隐藏状态
