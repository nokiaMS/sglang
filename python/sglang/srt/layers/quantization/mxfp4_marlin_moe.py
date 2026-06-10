# 文件说明：DeepSeek-V4 MXFP4专家后端，基于Marlin内核的MoE量化方法实现
# 本模块实现了MXFP4量化下MoE层的Marlin后端，支持Hopper(SM90)和SM120架构，
# SM90上使用Marlin内核，SM120上使用Triton反量化回退路径。
from __future__ import annotations # 启用延迟类型注解评估

import logging # 导入日志模块
from typing import TYPE_CHECKING # 导入类型检查常量

import torch # 导入PyTorch
from torch.nn import Module # 导入神经网络模块基类

from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo # 导入Marlin MoE量化信息类
from sglang.srt.layers.moe.utils import MoeRunnerBackend # 导入MoE运行器后端枚举
from sglang.srt.utils import log_info_on_rank0, set_weight_attrs # 导入rank0日志工具和权重属性设置工具
from sglang.srt.utils.common import is_sm90_supported, is_sm120_supported # 导入SM90和SM120支持检查函数

if TYPE_CHECKING: # 仅在类型检查时导入，避免运行时循环依赖
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput # 导入组合输入和分发输出类型

logger = logging.getLogger(__name__) # 获取当前模块的日志记录器


class Mxfp4MarlinMoEMethod: # MXFP4 Marlin MoE方法类
    """MXFP4 (E8M0 scales) MoE quantization method using the Marlin backend.""" # 使用Marlin后端的MXFP4（E8M0缩放）MoE量化方法。

    def __init__(self, fp8_method, prefix: str): # 初始化方法，接收FP8方法实例和前缀名
        self._fp8 = fp8_method # 保存FP8基础方法引用
        self.prefix = prefix # 保存层前缀名

    def create_moe_runner(self, layer, moe_runner_config): # 创建MoE运行器方法
        from sglang.srt.layers.moe.moe_runner import MoeRunner # 导入MoE运行器

        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config) # 创建Marlin MoE运行器

    def create_weights( # 创建权重参数方法
        self,
        layer: Module, # MoE层模块
        num_experts: int, # 专家数量
        hidden_size: int, # 隐藏层维度
        intermediate_size_per_partition: int, # 每个分区的中间层维度
        params_dtype: torch.dtype, # 参数数据类型
        **extra_weight_attrs, # 额外权重属性
    ):
        from sglang.srt.layers.moe.fused_moe_triton import ( # 导入权重缩放支持类型枚举
            FusedMoeWeightScaleSupported,
        )

        layer._dsv4_mxfp4_backend = None  # set in process_weights_after_loading # 在process_weights_after_loading中设置后端类型
        fp4_block_k = 32 # FP4块大小（每32个权重一个缩放因子）

        w13_weight = torch.nn.Parameter( # 创建FC1权重参数（gate+up，4位打包）
            torch.empty( # 创建空张量
                num_experts, # 专家数量
                2 * intermediate_size_per_partition, # gate+up的中间维度
                hidden_size // 2, # FP4打包后K维度减半
                dtype=torch.int8, # int8存储（每个元素包含2个4位权重）
            ),
            requires_grad=False, # 不需要梯度
        )
        w2_weight = torch.nn.Parameter( # 创建FC2权重参数（down投影，4位打包）
            torch.empty( # 创建空张量
                num_experts, # 专家数量
                hidden_size, # 隐藏维度
                intermediate_size_per_partition // 2, # FP4打包后中间维度减半
                dtype=torch.int8, # int8存储
            ),
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight) # 注册w13权重参数
        set_weight_attrs(w13_weight, extra_weight_attrs) # 设置权重属性
        layer.register_parameter("w2_weight", w2_weight) # 注册w2权重参数
        set_weight_attrs(w2_weight, extra_weight_attrs) # 设置权重属性

        w13_weight_scale = torch.nn.Parameter( # 创建FC1权重缩放因子参数
            torch.ones( # 创建全1张量
                num_experts, # 专家数量
                2 * intermediate_size_per_partition, # gate+up的中间维度
                hidden_size // fp4_block_k, # 每个块一个缩放因子
                dtype=torch.float32, # float32存储
            ),
            requires_grad=False, # 不需要梯度
        )
        w2_weight_scale = torch.nn.Parameter( # 创建FC2权重缩放因子参数
            torch.ones( # 创建全1张量
                num_experts, # 专家数量
                hidden_size, # 隐藏维度
                intermediate_size_per_partition // fp4_block_k, # 每个块一个缩放因子
                dtype=torch.float32, # float32存储
            ),
            requires_grad=False, # 不需要梯度
        )
        w13_weight_scale.format_ue8m0 = False # 标记缩放因子不是E8M0格式
        w2_weight_scale.format_ue8m0 = False # 标记缩放因子不是E8M0格式
        scale_attrs = dict(extra_weight_attrs) # 复制权重属性
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value # 标记使用块级缩放
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale) # 注册w13缩放因子参数
        set_weight_attrs(w13_weight_scale, scale_attrs) # 设置缩放因子属性
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale) # 注册w2缩放因子参数
        set_weight_attrs(w2_weight_scale, scale_attrs) # 设置缩放因子属性

    def process_weights_after_loading(self, layer: Module) -> None: # 加载后处理权重方法，进行Marlin格式转换
        from sglang.srt.layers.quantization.marlin_utils import ( # 导入Marlin支持检查函数
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.marlin_utils_fp4 import ( # 导入Marlin FP4层准备函数
            prepare_moe_mxfp4_layer_for_marlin,
        )

        # Let the FP8 base method handle ROCm normalization, etc. # 让FP8基础方法处理ROCm归一化等操作。
        self._fp8.process_weights_after_loading(layer) # 调用FP8基础方法的后加载处理

        if getattr(layer, "_mega_moe_weights_built", False): # 如果已经构建了mega MoE权重
            return # 直接返回，跳过处理

        if not is_sm90_supported() and not is_sm120_supported(): # 如果既不支持SM90也不支持SM120
            raise RuntimeError( # 抛出运行时错误
                "DeepSeekV4 MXFP4 Marlin fallback requires Hopper/SM90 or above." # DeepSeekV4 MXFP4 Marlin回退需要Hopper/SM90或更高架构。
            )

        # SM120: Skip Marlin repacking, keep original weight format # SM120：跳过Marlin重打包，保留原始权重格式
        # for Triton dequant kernel (Marlin kernel produces NaN on SM120) # 用于Triton反量化内核（Marlin内核在SM120上产生NaN）
        if is_sm120_supported(): # 如果支持SM120
            from torch.nn import Parameter # 导入参数类

            log_info_on_rank0( # 在rank0上记录信息
                logger,
                f"SM120 detected: using PyTorch MXFP4 MoE fallback " # 检测到SM120：使用PyTorch MXFP4 MoE回退
                f"(layer: {self.prefix})...", # （层：{前缀}）...
            )
            # Keep weights in original packed int8 format # 保留原始打包int8格式的权重
            # Normalize scales to float32 for direct use in dequant # 将缩放因子归一化为float32以便反量化时直接使用
            w13_s = layer.w13_weight_scale_inv.data # 获取w13缩放因子数据
            w2_s = layer.w2_weight_scale_inv.data # 获取w2缩放因子数据
            if w13_s.dtype == torch.float8_e8m0fnu: # 如果缩放因子已经是E8M0格式
                pass  # already in e8m0 format, will convert at runtime # 已经是e8m0格式，运行时转换
            elif w13_s.dtype in (torch.uint8, torch.int8): # 如果缩放因子是uint8或int8格式
                layer.w13_weight_scale_inv = Parameter( # 转换w13缩放因子为float32
                    w13_s.view(torch.uint8) # 按uint8查看
                    .view(torch.float8_e8m0fnu) # 按E8M0查看
                    .to(torch.float32), # 转为float32
                    requires_grad=False,
                )
                layer.w2_weight_scale_inv = Parameter( # 转换w2缩放因子为float32
                    w2_s.view(torch.uint8).view(torch.float8_e8m0fnu).to(torch.float32), # 同样转换流程
                    requires_grad=False,
                )
            # else: float32 scales are already usable directly # 否则：float32缩放因子已可直接使用
            layer._dsv4_mxfp4_backend = "sm120_triton" # 标记使用SM120 Triton回退后端
            return # 返回，跳过Marlin处理

        if not check_moe_marlin_supports_layer(layer, 32): # 检查当前MoE层是否满足Marlin约束
            raise RuntimeError( # 抛出运行时错误
                "Current DeepSeekV4 MoE layer does not satisfy Marlin constraints." # 当前DeepSeekV4 MoE层不满足Marlin约束。
            )

        # NOTE: the Marlin MoE runner consumes w13 in the checkpoint's # 注意：Marlin MoE运行器以检查点的
        # native ``[w1; w3]`` order -- see ``silu_and_mul`` in # 原生``[w1; w3]``顺序使用w13——参见
        # fused_marlin_moe.py which expects ``gate = intermediate[:, :N]`` # fused_marlin_moe.py中的``silu_and_mul``，其中
        # (first half) and ``up = intermediate[:, N:]`` (second half). # ``gate = intermediate[:, :N]``（前半部分）和``up = intermediate[:, N:]``（后半部分）。
        # Unlike the flashinfer trtllm_fp4 kernel (which wants [w3, w1]), # 与flashinfer trtllm_fp4内核（需要[w3, w1]）不同，
        # we must *not* call ``reorder_w1w3_to_w3w1`` here. # 我们*不能*在此调用``reorder_w1w3_to_w3w1``。

        log_info_on_rank0( # 在rank0上记录信息
            logger,
            f"Preparing DeepSeekV4 MXFP4 experts for Marlin backend " # 正在为Marlin后端准备DeepSeekV4 MXFP4专家
            f"(layer: {self.prefix})...", # （层：{前缀}）...
        )
        prepare_moe_mxfp4_layer_for_marlin(layer) # 调用Marlin FP4层准备函数
        layer._dsv4_mxfp4_backend = "marlin" # 标记使用Marlin后端

    def apply( # 应用方法，执行MoE前向推理
        self,
        layer: Module, # MoE层模块
        dispatch_output: DispatchOutput, # 分发输出
    ) -> CombineInput: # 返回组合输入
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput # 导入标准组合输入类
        from sglang.srt.layers.moe.topk import TopKOutputChecker # 导入TopK输出检查器

        topk_output = dispatch_output.topk_output # 获取TopK路由输出
        if not TopKOutputChecker.format_is_standard(topk_output): # 检查TopK输出格式是否为标准格式
            raise ValueError(f"Unsupported topk output format: {topk_output.format}") # 不支持则抛出错误

        # SM120: use Triton fused dequant+GEMM (Marlin kernel produces NaN on SM120) # SM120：使用Triton融合反量化+GEMM（Marlin内核在SM120上产生NaN）
        if layer._dsv4_mxfp4_backend == "sm120_triton": # 如果使用SM120 Triton回退后端
            from sglang.srt.layers.moe.fused_moe_triton.mxfp4_moe_sm120_triton import ( # 导入SM120 Triton MXFP4 MoE前向函数
                mxfp4_moe_forward_triton,
            )

            hidden_states = dispatch_output.hidden_states # 获取隐藏状态
            w13 = layer.w13_weight.data # 获取w13权重数据
            w2 = layer.w2_weight.data # 获取w2权重数据
            w13_scale = layer.w13_weight_scale_inv.data # 获取w13缩放因子数据
            w2_scale = layer.w2_weight_scale_inv.data # 获取w2缩放因子数据
            intermediate_size = w13.shape[1] // 2 # 计算中间维度（gate+up各一半）
            hidden_size = w13.shape[2] * 2 # 计算隐藏维度（FP4打包需乘2）

            output = mxfp4_moe_forward_triton( # 调用SM120 Triton MXFP4 MoE前向函数
                hidden_states=hidden_states, # 输入隐藏状态
                w13_packed=w13, # 打包的w13权重
                w2_packed=w2, # 打包的w2权重
                w13_scale=w13_scale, # w13缩放因子
                w2_scale=w2_scale, # w2缩放因子
                topk_ids=topk_output.topk_ids, # topk ID
                topk_weights=topk_output.topk_weights, # topk权重
                hidden_size=hidden_size, # 隐藏维度
                intermediate_size=intermediate_size, # 中间维度
                routed_scaling_factor=( # 路由缩放因子
                    self.runner.config.routed_scaling_factor # 从运行器配置获取
                    if hasattr(self.runner, "config") # 如果运行器有配置
                    else None # 否则为None
                ),
                clamp_limit=( # SwiGLU限幅值
                    self.runner.config.swiglu_limit # 从运行器配置获取
                    if hasattr(self.runner, "config") # 如果运行器有配置
                    else None # 否则为None
                ),
            )
            return StandardCombineInput(hidden_states=output) # 返回标准组合输入

        quant_info = MarlinMoeQuantInfo( # 构建Marlin MoE量化信息
            w13_qweight=layer.w13_weight, # FC1量化权重
            w2_qweight=layer.w2_weight, # FC2量化权重
            w13_scales=layer.w13_weight_scale, # FC1缩放因子
            w2_scales=layer.w2_weight_scale, # FC2缩放因子
            w13_g_idx_sort_indices=None, # 无分组索引排序
            w2_g_idx_sort_indices=None, # 无分组索引排序
            weight_bits=4, # 权重位宽为4位
            is_k_full=True, # K维度完整
        )
        runner_output = self.runner.run(dispatch_output, quant_info=quant_info) # 运行Marlin MoE运行器

        return StandardCombineInput(hidden_states=runner_output.hidden_states) # 返回标准组合输入
