# 压缩张量MX INT4量化MoE方案模块
# 实现了CompressedTensorsMxInt4MoE类，用于Blackwell架构上的
# W4A4 MX INT4块缩放MoE量化推理，依赖FlashInfer TRT-LLM后端
from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 导入日志模块
from typing import TYPE_CHECKING  # 导入类型检查常量

import torch  # 导入PyTorch
from compressed_tensors import CompressionFormat  # 导入压缩格式枚举

from sglang.srt.distributed import get_moe_expert_parallel_rank, get_tp_group  # 导入MoE专家并行秩和TP组获取函数
from sglang.srt.distributed.device_communicators.pynccl_allocator import (  # 导入对称内存使用上下文管理器
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric  # 导入对称分配检查函数
from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置类
from sglang.srt.layers.moe.utils import RoutingMethodType, get_moe_runner_backend  # 导入路由方法类型和MoE运行器后端获取函数
from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # 导入压缩张量MoE方案基类
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.utils import replace_parameter  # 导入参数替换工具函数
from sglang.srt.utils import is_flashinfer_available, next_power_of_2, set_weight_attrs  # 导入FlashInfer可用性检查、下一个2的幂、权重属性设置工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

__all__ = ["CompressedTensorsMxInt4MoE"]  # 模块公开导出的类列表

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入MoE token分发器类型
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (  # 导入压缩张量配置类型
        CompressedTensorsConfig,
    )

if is_flashinfer_available():  # 如果FlashInfer可用，则导入相关函数
    from flashinfer.fp4_quantization import block_scale_interleave  # 导入块缩放交错函数
    from flashinfer.fused_moe import (  # 导入融合MoE相关函数
        convert_to_block_layout,  # 转换为块布局
        trtllm_mxint4_block_scale_moe,  # TRT-LLM MX INT4块缩放MoE内核
    )
    from flashinfer.fused_moe.core import (  # 导入融合MoE核心缓存函数
        _maybe_get_cached_w3_w1_permute_indices,  # 获取w3_w1排列索引（带缓存）
        get_w2_permute_indices_with_cache,  # 获取w2排列索引（带缓存）
    )


class CompressedTensorsMxInt4MoE(CompressedTensorsMoEScheme):  # MX INT4量化MoE方案类，继承自CompressedTensorsMoEScheme
    def __init__(self, quant_config: CompressedTensorsConfig):  # 初始化方法，接收量化配置
        self.quant_config = quant_config  # 保存量化配置
        config = self.quant_config.target_scheme_map["Linear"].get("weights")  # 获取线性层权重的量化方案配置
        self.num_bits = config.num_bits  # 量化位数
        self.packed_factor = 32 // config.num_bits  # 打包因子，32位整数中可打包的元素数
        self.strategy = config.strategy  # 量化策略
        self.group_size = config.group_size  # 量化分组大小
        self.actorder = config.actorder  # 是否使用激活值排序
        assert (  # 断言：仅支持group策略，分组大小32，4位量化
            config.strategy == "group"
            and config.group_size == 32
            and config.num_bits == 4
        ), "MxInt4 only supports group strategy with group size 32"  # MxInt4仅支持分组大小为32的group策略
        assert config.symmetric, "Only symmetric quantization is supported for MoE"  # 断言：MoE仅支持对称量化
        assert (  # 断言：仅支持flashinfer_trtllm后端
            get_moe_runner_backend().is_flashinfer_trtllm()
        ), "MxInt4 only supports flashinfer_trtllm backend"  # MxInt4仅支持flashinfer_trtllm后端
        assert (  # 断言：不支持激活值排序
            not config.actorder
        ), "Actorder is not supported by flashinfer_trtllm backend"  # flashinfer_trtllm后端不支持actorder
        self.moe_ep_rank = get_moe_expert_parallel_rank()  # 获取当前MoE专家并行秩

        if self.quant_config.quant_format != CompressionFormat.pack_quantized.value:  # 检查量化格式是否为打包量化
            raise ValueError(  # 若不是则抛出异常
                f"For Fused MoE layers, only {CompressionFormat.pack_quantized.value} "
                "is supported for the mxint4"  # 融合MoE层仅支持pack_quantized格式
            )
        self._cache_permute_indices = {}  # 初始化排列索引缓存字典

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
        assert (  # 断言参数数据类型必须为bfloat16
            params_dtype == torch.bfloat16
        ), f"Params dtype should be torch.bfloat16, but got: {params_dtype}"  # 参数类型应为torch.bfloat16

        extra_weight_attrs.update({"quant_method": self.strategy})  # 更新额外权重属性，添加量化方法
        w13_weight = torch.nn.Parameter(  # 创建w13权重参数（gate+up投影打包权重）
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                2 * intermediate_size_per_partition,  # 双倍中间层大小（gate和up拼接）
                hidden_size // self.packed_factor,  # 输入维度除以打包因子
                dtype=torch.int32,  # 使用int32类型存储打包的4位权重
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_packed", w13_weight)  # 注册w13打包权重参数
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置权重属性

        w2_weight = torch.nn.Parameter(  # 创建w2权重参数（down投影打包权重）
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小维度
                intermediate_size_per_partition // self.packed_factor,  # 中间层维度除以打包因子
                dtype=torch.int32,  # 使用int32类型存储打包的4位权重
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight_packed", w2_weight)  # 注册w2打包权重参数
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置权重属性

        w2_scales_size = intermediate_size_per_partition  # w2缩放因子尺寸
        num_groups_w2 = w2_scales_size // self.group_size  # w2的分组数量
        num_groups_w13 = hidden_size // self.group_size  # w13的分组数量

        w13_scale = torch.nn.Parameter(  # 创建w13权重缩放因子参数
            torch.ones(  # 初始化为全1
                num_experts,  # 专家数量维度
                2 * intermediate_size_per_partition,  # 双倍中间层大小
                num_groups_w13,  # 分组数量维度
                dtype=params_dtype,  # 使用参数数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_scale", w13_scale)  # 注册w13权重缩放因子
        set_weight_attrs(w13_scale, extra_weight_attrs)  # 设置权重属性

        w2_scale = torch.nn.Parameter(  # 创建w2权重缩放因子参数
            torch.ones(num_experts, hidden_size, num_groups_w2, dtype=params_dtype),  # 初始化为全1
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight_scale", w2_scale)  # 注册w2权重缩放因子
        set_weight_attrs(w2_scale, extra_weight_attrs)  # 设置权重属性

        w13_weight_shape = torch.nn.Parameter(  # 创建w13权重形状参数，用于记录原始形状信息
            torch.empty(num_experts, 2), requires_grad=False  # 每个专家两个维度值
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)  # 注册w13权重形状参数
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)  # 设置权重属性

        w2_weight_shape = torch.nn.Parameter(  # 创建w2权重形状参数，用于记录原始形状信息
            torch.empty(num_experts, 2), requires_grad=False  # 每个专家两个维度值
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)  # 注册w2权重形状参数
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)  # 设置权重属性

        layer.a13_scale = None  # w13激活缩放因子初始化为None
        layer.a2_scale = None  # w2激活缩放因子初始化为None

    # Adapted from https://github.com/flashinfer-ai/flashinfer/blob/main/tests/moe/test_trtllm_gen_fused_moe.py
    def prepare_static_weights_for_kernel(  # 为内核准备静态权重（离线预处理）
        self,
        gemm1_weights,  # gemm1（w13）权重
        gemm2_weights,  # gemm2（w2）权重
        gemm1_scales,  # gemm1缩放因子
        gemm2_scales,  # gemm2缩放因子
        num_experts,  # 专家数量
    ):
        """Prepare quantized weights for kernel (done offline with weights)."""
        """为内核准备量化权重（离线与权重一起完成）。"""

        epilogue_tile_m = 128  # epilogue分块大小
        gemm1_weights_mxint4_shuffled = []  # 存放处理后的gemm1权重列表
        gemm1_scales_shuffled = []  # 存放处理后的gemm1缩放因子列表
        gemm2_weights_mxint4_shuffled = []  # 存放处理后的gemm2权重列表
        gemm2_scales_shuffled = []  # 存放处理后的gemm2缩放因子列表

        def repack(w):  # 重新打包权重：从HuggingFace格式转换为TRT-LLM格式
            assert w.dim() == 2 and w.dtype == torch.int32  # 断言权重为2维int32张量
            shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=w.device)  # 创建位移数组：0,4,8,...,28
            w = (w.unsqueeze(2) >> shifts) & 0x0F  # 右移并掩码，解包每个4位值
            w = (w - 8).to(torch.int8).reshape(w.shape[0], -1, 2)  # 减8转为有符号，重塑为2列
            w = (w[..., 0] & 0x0F) | ((w[..., 1] & 0x0F) << 4)  # 重新打包为TRT-LLM格式
            w = w.to(torch.uint8)  # 转为无符号8位整型
            return w  # 返回重新打包的权重

        for i in range(num_experts):  # 遍历每个专家
            # NOTE(HandH1998):
            # the huggingface weight format follows (w/s + 8) to pack,
            # however, trtllm requires (w/s) to pack
            # we need to convert the weight to trtllm's format first
            # HuggingFace权重格式按(w/s + 8)打包，
            # 但TRT-LLM要求按(w/s)打包
            # 需要先将权重转换为TRT-LLM的格式
            cur_expert_gemm1_weight = repack(gemm1_weights[i])  # 重新打包当前专家的gemm1权重
            cur_expert_gemm2_weight = repack(gemm2_weights[i])  # 重新打包当前专家的gemm2权重

            # Calculate the permute indices for the following:
            # 1. Reorder rows of W1 and scales for fused gated activation
            # 2. Shuffle weights and scaling factors for transposed mma output
            # for both w3_w1 and w2 weights and scale factors
            # 计算以下操作的排列索引：
            # 1. 重排W1和缩放因子的行，用于融合门控激活
            # 2. 对w3_w1和w2的权重与缩放因子进行重排，用于转置mma输出
            permute_indices = _maybe_get_cached_w3_w1_permute_indices(  # 获取w3_w1排列索引（带缓存）
                self._cache_permute_indices,  # 排列索引缓存
                cur_expert_gemm1_weight,  # 当前专家gemm1权重
                epilogue_tile_m,  # epilogue分块大小
            )
            gemm1_weights_shuffled = cur_expert_gemm1_weight[  # 按排列索引重排gemm1权重
                permute_indices.to(gemm1_weights.device)
            ].contiguous()  # 确保内存连续
            permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(  # 获取gemm1缩放因子的排列索引
                self._cache_permute_indices,  # 排列索引缓存
                gemm1_scales[i].to(torch.bfloat16),  # 当前专家gemm1缩放因子转为bfloat16
                epilogue_tile_m,  # epilogue分块大小
                num_elts_per_sf=32,  # 每个缩放因子对应的元素数
            )
            gemm1_scales_shuffled.append(  # 添加处理后的gemm1缩放因子
                block_scale_interleave(  # 对块缩放因子进行交错处理
                    gemm1_scales[i]
                    .to(torch.bfloat16)[permute_sf_indices.to(gemm1_scales.device)]
                    .contiguous()  # 确保内存连续
                )
            )

            permute_indices = get_w2_permute_indices_with_cache(  # 获取w2排列索引（带缓存）
                self._cache_permute_indices,  # 排列索引缓存
                cur_expert_gemm2_weight,  # 当前专家gemm2权重
                epilogue_tile_m,  # epilogue分块大小
            )
            gemm2_weights_shuffled = cur_expert_gemm2_weight[  # 按排列索引重排gemm2权重
                permute_indices.to(gemm2_weights.device)
            ].contiguous()  # 确保内存连续

            permute_sf_indices = get_w2_permute_indices_with_cache(  # 获取gemm2缩放因子的排列索引
                self._cache_permute_indices,  # 排列索引缓存
                gemm2_scales[i].to(torch.bfloat16),  # 当前专家gemm2缩放因子转为bfloat16
                epilogue_tile_m,  # epilogue分块大小
                num_elts_per_sf=16,  # 每个缩放因子对应的元素数
            )
            gemm2_scales_shuffled.append(  # 添加处理后的gemm2缩放因子
                block_scale_interleave(  # 对块缩放因子进行交错处理
                    gemm2_scales[i]
                    .to(torch.bfloat16)[permute_sf_indices.to(gemm2_scales.device)]
                    .contiguous()  # 确保内存连续
                )
            )

            block_k = 128  # 块的k维度大小
            gemm1_weights_shuffled = convert_to_block_layout(  # 将gemm1权重转换为块布局
                gemm1_weights_shuffled.view(torch.uint8), block_k  # 视图转为uint8后转换
            )
            gemm2_weights_shuffled = convert_to_block_layout(  # 将gemm2权重转换为块布局
                gemm2_weights_shuffled.view(torch.uint8), block_k  # 视图转为uint8后转换
            )

            gemm1_weights_mxint4_shuffled.append(gemm1_weights_shuffled)  # 添加处理后的gemm1权重
            gemm2_weights_mxint4_shuffled.append(gemm2_weights_shuffled)  # 添加处理后的gemm2权重

        gemm1_weights_mxint4_shuffled = torch.stack(gemm1_weights_mxint4_shuffled)  # 堆叠所有专家的gemm1权重
        gemm2_weights_mxint4_shuffled = torch.stack(gemm2_weights_mxint4_shuffled)  # 堆叠所有专家的gemm2权重
        gemm1_scales_shuffled = torch.stack(gemm1_scales_shuffled).view(torch.bfloat16)  # 堆叠并视图转为bfloat16
        gemm2_scales_shuffled = torch.stack(gemm2_scales_shuffled).view(torch.bfloat16)  # 堆叠并视图转为bfloat16

        return (  # 返回处理后的权重和缩放因子元组
            gemm1_weights_mxint4_shuffled,  # gemm1权重（已重排和块布局转换）
            gemm1_scales_shuffled,  # gemm1缩放因子（已重排和交错）
            gemm2_weights_mxint4_shuffled,  # gemm2权重（已重排和块布局转换）
            gemm2_scales_shuffled,  # gemm2缩放因子（已重排和交错）
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的后处理方法

        num_experts = layer.w13_weight_packed.shape[0]  # 获取专家数量
        (  # 调用权重预处理函数，获取处理后的权重和缩放因子
            gemm1_weights_mxint4_shuffled,  # 处理后的gemm1权重
            gemm1_scales_shuffled,  # 处理后的gemm1缩放因子
            gemm2_weights_mxint4_shuffled,  # 处理后的gemm2权重
            gemm2_scales_shuffled,  # 处理后的gemm2缩放因子
        ) = self.prepare_static_weights_for_kernel(  # 准备内核所需的静态权重
            layer.w13_weight_packed,  # w13打包权重
            layer.w2_weight_packed,  # w2打包权重
            layer.w13_weight_scale,  # w13权重缩放因子
            layer.w2_weight_scale,  # w2权重缩放因子
            num_experts=num_experts,  # 专家数量
        )
        replace_parameter(layer, "w13_weight_packed", gemm1_weights_mxint4_shuffled)  # 替换w13权重为处理后的版本
        replace_parameter(layer, "w2_weight_packed", gemm2_weights_mxint4_shuffled)  # 替换w2权重为处理后的版本
        replace_parameter(layer, "w13_weight_scale", gemm1_scales_shuffled)  # 替换w13缩放因子为处理后的版本
        replace_parameter(layer, "w2_weight_scale", gemm2_scales_shuffled)  # 替换w2缩放因子为处理后的版本

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置

    def apply_weights(  # 应用权重，执行MoE前向传播
        self,
        layer: torch.nn.Module,  # 目标层模块
        dispatch_output: StandardDispatchOutput,  # 分发输出
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入类

        assert (  # 断言MoE必须是门控类型
            self.moe_runner_config.is_gated
        ), "Only gated MoEs are supported for flashinfer mxint4"  # flashinfer mxint4仅支持门控MoE

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取topk输出

        router_logits = topk_output.router_logits  # 获取路由器logits
        topk_config = topk_output.topk_config  # 获取topk配置
        correction_bias = (  # 获取校正偏置
            None
            if topk_config.correction_bias is None  # 若配置中无校正偏置则为None
            else topk_config.correction_bias.to(x.dtype)  # 否则转为输入数据类型
        )

        local_num_experts = self.moe_runner_config.num_local_experts  # 获取本地专家数量
        routing_method_type = layer.routing_method_type  # 获取路由方法类型
        assert routing_method_type is not None  # 断言路由方法类型不为空
        # DeepSeekV3 style routing requires float32 router logits,
        # see this PR for details: https://github.com/flashinfer-ai/flashinfer/commit/d84e1d560da0a27961c19ca788d96c19cb9dcfb6
        # DeepSeekV3风格路由需要float32路由器logits，
        # 详见此PR：https://github.com/flashinfer-ai/flashinfer/commit/d84e1d560da0a27961c19ca788d96c19cb9dcfb6
        if routing_method_type == RoutingMethodType.DeepSeekV3:  # 若为DeepSeekV3路由方法
            router_logits = router_logits.to(torch.float32)  # 将路由器logits转为float32
        routed_scaling_factor = self.moe_runner_config.routed_scaling_factor  # 获取路由缩放因子
        routed_scaling_factor = (  # 若缩放因子为None则设为1.0
            routed_scaling_factor if routed_scaling_factor is not None else 1.0
        )

        with use_symmetric_memory(  # 使用对称内存上下文
            get_tp_group(), disabled=not is_allocation_symmetric()  # 若非对称分配则禁用
        ):
            num_tokens = x.shape[0]  # 获取token数量
            hidden_size = x.shape[-1]  # 获取隐藏层大小
            symm_output = torch.empty(  # 预分配对称内存输出张量
                num_tokens, hidden_size, dtype=torch.bfloat16, device=x.device  # 使用bfloat16类型
            )

        trtllm_mxint4_block_scale_moe(  # 调用TRT-LLM MX INT4块缩放MoE内核
            routing_logits=router_logits,  # float  # 路由器logits（浮点类型）
            routing_bias=correction_bias,  # 路由偏置
            hidden_states=x,  # 输入隐藏状态
            gemm1_weights=layer.w13_weight_packed,  # gemm1权重
            gemm1_weights_scale=layer.w13_weight_scale,  # gemm1权重缩放因子
            gemm1_alpha=self.moe_runner_config.gemm1_alpha,  # gemm1 alpha参数
            gemm1_beta=None,  # gemm1 beta参数（未使用）
            gemm1_clamp_limit=self.moe_runner_config.gemm1_clamp_limit,  # gemm1截断限制
            gemm2_weights=layer.w2_weight_packed,  # gemm2权重
            gemm2_weights_scale=layer.w2_weight_scale,  # gemm2权重缩放因子
            num_experts=self.moe_runner_config.num_experts,  # 专家总数
            top_k=topk_config.top_k,  # top-k值
            n_group=topk_config.num_expert_group,  # 专家分组数
            topk_group=topk_config.topk_group,  # 每组选择的专家数
            intermediate_size=self.moe_runner_config.intermediate_size_per_partition,  # 中间层大小
            local_expert_offset=self.moe_ep_rank * local_num_experts,  # 本地专家偏移
            local_num_experts=local_num_experts,  # 本地专家数量
            routed_scaling_factor=routed_scaling_factor,  # 路由缩放因子
            routing_method_type=routing_method_type,  # 路由方法类型
            tune_max_num_tokens=next_power_of_2(x.shape[0]),  # 调优最大token数（取下一个2的幂）
            output=symm_output,  # 输出张量
        )

        return StandardCombineInput(hidden_states=symm_output)  # 返回标准合并输入，包含输出隐藏状态
