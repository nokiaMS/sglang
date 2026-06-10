# 压缩张量W8A8 FP8 MoE量化方案实现
# 本文件实现了CompressedTensors框架下W8A8（权重8比特FP8、激活8比特FP8）量化的MoE（混合专家）层方案
# 支持逐张量、逐通道和分块三种权重量化策略，支持AITER、Triton和FlashInfer-TRTLLM后端

from __future__ import annotations # 启用延迟注解评估

import logging # 导入日志模块
from typing import TYPE_CHECKING # 导入类型检查常量

import torch # 导入PyTorch库
from compressed_tensors.quantization import QuantizationStrategy # 导入量化策略枚举

from sglang.srt.distributed import get_tensor_model_parallel_world_size # 导入获取张量并行世界大小函数
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig # 导入MoE运行器相关类
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import ( # 导入FlashInfer TRTLLM FP8 MoE量化信息类
    FlashInferTrtllmFp8MoeQuantInfo,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo # 导入Triton MoE量化信息类
from sglang.srt.layers.moe.utils import ( # 导入MoE工具函数
    get_moe_a2a_backend, # 获取MoE All-to-All后端
    get_moe_runner_backend, # 获取MoE运行器后端
    get_moe_weight_sizes, # 获取MoE权重尺寸
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import ( # 导入压缩张量MoE方案基类
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz, scaled_fp8_quant # 导入FP8 FNUZ检测和缩放FP8量化函数
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz # 导入E4M3FN到E4M3FNUZ归一化函数
from sglang.srt.layers.quantization.utils import ( # 导入量化工具函数
    all_close_1d, # 一维全近似判断
    per_tensor_dequantize, # 逐张量反量化
    swap_w13_to_w31, # 交换w13到w31（重排gate/up权重顺序）
)
from sglang.srt.utils import get_bool_env_var, is_hip, set_weight_attrs # 导入工具函数

if TYPE_CHECKING: # 仅在类型检查时导入
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE # 导入融合MoE类型
    from sglang.srt.layers.moe.token_dispatcher import ( # 导入token分发器类型
        CombineInput, # 合并输入类型
        StandardDispatchOutput, # 标准分发输出类型
    )

__all__ = ["CompressedTensorsW8A8Fp8MoE"] # 模块公开接口列表

_is_hip = is_hip() # 检测是否为AMD HIP平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip # 是否使用AITER（仅HIP平台支持）

if _use_aiter: # 如果使用AITER
    from aiter.ops.shuffle import shuffle_weight # 导入AITER权重重排函数


logger = logging.getLogger(__name__) # 获取当前模块的日志记录器


class CompressedTensorsW8A8Fp8MoE(CompressedTensorsMoEScheme): # 压缩张量W8A8 FP8 MoE量化方案类
    """W8A8 FP8量化MoE方案：权重和激活均使用FP8格式的混合专家层量化实现"""

    def __init__(self, weight_quant, input_quant): # 初始化方法，接收权重量化参数和输入量化参数
        self.weight_quant = weight_quant # 保存权重量化参数
        self.input_quant = input_quant # 保存输入量化参数
        self.use_flashinfer_trtllm = get_moe_runner_backend().is_flashinfer_trtllm() # 检测是否使用FlashInfer TRTLLM后端

        per_tensor = ( # 判断是否为逐张量量化组合
            self.weight_quant.strategy == QuantizationStrategy.TENSOR # 权重使用逐张量策略
            and self.input_quant.strategy == QuantizationStrategy.TENSOR # 输入使用逐张量策略
        )
        per_channel = ( # 判断是否为逐通道量化组合
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL # 权重使用逐通道策略
            and self.input_quant.strategy == QuantizationStrategy.TOKEN # 输入使用逐token策略
        )
        if not (per_tensor or per_channel): # 如果既不是逐张量也不是逐通道组合
            assert self.weight_quant.strategy == QuantizationStrategy.BLOCK # 断言权重必须为分块策略
            self.weight_block_size = self.weight_quant.block_structure # 保存权重分块结构
            assert self.weight_quant.dynamic is not None # 断言动态量化属性不为None
        else: # 否则（逐张量或逐通道组合）
            self.weight_block_size = None # 分块大小为None
        self.block_quant = self.weight_block_size is not None # 是否为分块量化

        self.static_input_scales = not self.input_quant.dynamic # 是否使用静态输入缩放
        if self.static_input_scales and per_channel: # 如果静态输入缩放且逐通道组合
            raise ValueError( # 抛出值错误
                "For FP8 Fused MoE layer, we require either per tensor or "
                "channelwise, dynamic per token quantization." # FP8融合MoE层要求逐张量或逐通道加动态逐token量化
            )

    @classmethod
    def get_min_capability(cls) -> int: # 获取最低GPU计算能力要求
        """获取运行此量化方案所需的最低GPU计算能力版本"""
        # ampere and up # Ampere架构及以上
        return 80 # 返回8.0（Ampere架构）

    def create_weights( # 创建权重参数方法
        self,
        layer: torch.nn.Module, # 目标神经网络层
        num_experts: int, # 专家数量
        hidden_size: int, # 隐藏层大小
        intermediate_size_per_partition: int, # 每个分区的中间层大小
        params_dtype: torch.dtype, # 参数数据类型
        **extra_weight_attrs, # 额外的权重属性关键字参数
    ):
        """创建并注册W8A8 FP8 MoE量化所需的权重、缩放因子和输入缩放参数"""
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported # 导入融合MoE权重缩放支持枚举

        params_dtype = torch.float8_e4m3fn # 强制设置参数类型为FP8 E4M3

        if self.block_quant: # 如果是分块量化
            assert self.weight_block_size is not None # 断言分块大小不为None
            layer.weight_block_size = self.weight_block_size # 保存分块大小到层
            tp_size = get_tensor_model_parallel_world_size() # 获取张量并行世界大小
            block_n, block_k = ( # 提取N和K方向的分块大小
                self.weight_block_size[0], # N方向分块大小
                self.weight_block_size[1], # K方向分块大小
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # 注意：为确保分块量化缩放因子的正确对齐，gate和up层权重的output_size必须能被block_n整除。
            # Required by column parallel or enabling merged weights
            # 列并行或启用合并权重时需要
            if intermediate_size_per_partition % block_n != 0: # 检查中间层大小是否能被block_n整除
                raise ValueError( # 抛出值错误
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}." # gate和up权重的output_size不能被block_n整除
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0: # 检查TP>1时中间层大小是否能被block_k整除
                # Required by row parallel # 行并行时需要
                raise ValueError( # 抛出值错误
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}." # down权重的input_size不能被block_k整除
                )

        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes( # 获取MoE权重尺寸
            intermediate_size_per_partition, # 每个分区的中间层大小
            is_aiter_moe=_use_aiter, # 是否使用AITER MoE
            is_concat=True, # 是否使用拼接模式
            is_packed=False, # 是否使用打包模式
        )

        extra_weight_attrs.update( # 更新额外权重属性
            {"weight_padded": weight_padded}, # 添加权重填充标志
        )

        # WEIGHTS # 权重
        w13_weight = torch.nn.Parameter( # 创建w13权重参数（gate+up投影）
            torch.empty( # 创建空张量
                num_experts, # 专家数量维度
                w13_up_dim, # w13上升维度
                hidden_size, # 隐藏大小维度
                dtype=params_dtype, # FP8数据类型
            ),
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight) # 在层中注册w13权重参数
        set_weight_attrs(w13_weight, extra_weight_attrs) # 设置w13权重的额外属性

        w2_weight = torch.nn.Parameter( # 创建w2权重参数（down投影）
            torch.empty( # 创建空张量
                num_experts, # 专家数量维度
                hidden_size, # 隐藏大小维度
                w2_down_dim, # w2下降维度
                dtype=params_dtype, # FP8数据类型
            ),
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w2_weight", w2_weight) # 在层中注册w2权重参数
        set_weight_attrs(w2_weight, extra_weight_attrs) # 设置w2权重的额外属性

        # WEIGHT_SCALES # 权重缩放因子
        # per-tensor quantization # 逐张量量化
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR: # 如果是逐张量策略
            # Allocate 2 scales for w1 and w3 respectively.
            # 为w1和w3分别分配2个缩放值。
            # They will be combined to a single scale after weight loading.
            # 它们将在权重加载后合并为一个缩放值。
            w13_weight_scale = torch.nn.Parameter( # 创建w13权重缩放参数（2个值，分别对应w1和w3）
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False # 形状为[专家数, 2]
            )
            w2_weight_scale = torch.nn.Parameter( # 创建w2权重缩放参数
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False # 形状为[专家数]
            )
            weight_quant_method = FusedMoeWeightScaleSupported.TENSOR.value # 量化方法为逐张量
        elif self.weight_quant.strategy == QuantizationStrategy.CHANNEL: # 如果是逐通道策略
            w13_weight_scale = torch.nn.Parameter( # 创建w13权重缩放参数（逐通道）
                torch.ones( # 创建全1张量
                    num_experts, # 专家数量维度
                    w13_up_dim, # w13上升维度
                    1, # 最后一维为1（广播）
                    dtype=torch.float32, # float32类型
                ),
                requires_grad=False, # 不需要梯度
            )
            w2_weight_scale = torch.nn.Parameter( # 创建w2权重缩放参数（逐通道）
                torch.ones(num_experts, hidden_size, 1, dtype=torch.float32), # 形状为[专家数, 隐藏大小, 1]
                requires_grad=False, # 不需要梯度
            )
            weight_quant_method = FusedMoeWeightScaleSupported.CHANNEL.value # 量化方法为逐通道
        elif self.weight_quant.strategy == QuantizationStrategy.BLOCK: # 如果是分块策略
            w13_weight_scale = torch.nn.Parameter( # 创建w13权重缩放参数（分块）
                torch.ones( # 创建全1张量
                    num_experts, # 专家数量维度
                    2 * ((intermediate_size_per_partition + block_n - 1) // block_n), # gate和up的N方向分块数之和
                    (hidden_size + block_k - 1) // block_k, # K方向分块数
                    dtype=torch.float32, # float32类型
                ),
                requires_grad=False, # 不需要梯度
            )
            w2_weight_scale = torch.nn.Parameter( # 创建w2权重缩放参数（分块）
                torch.ones( # 创建全1张量
                    num_experts, # 专家数量维度
                    (hidden_size + block_n - 1) // block_n, # N方向分块数
                    (intermediate_size_per_partition + block_k - 1) // block_k, # K方向分块数
                    dtype=torch.float32, # float32类型
                ),
                requires_grad=False, # 不需要梯度
            )
            weight_quant_method = FusedMoeWeightScaleSupported.BLOCK.value # 量化方法为分块
        else: # 其他不支持的策略
            raise ValueError( # 抛出值错误
                f"Unsupported weight quantization strategy: {self.weight_quant.strategy}" # 不支持的权重量化策略
            )

        layer.register_parameter("w13_weight_scale", w13_weight_scale) # 在层中注册w13权重缩放参数
        layer.register_parameter("w2_weight_scale", w2_weight_scale) # 在层中注册w2权重缩放参数
        # Add the quantization method used (per tensor/grouped/channel)
        # 添加使用的量化方法（逐张量/分组/逐通道）
        # to ensure the weight scales are loaded in properly
        # 以确保权重缩放因子被正确加载
        extra_weight_attrs.update({"quant_method": weight_quant_method}) # 更新量化方法属性
        set_weight_attrs(w13_weight_scale, extra_weight_attrs) # 设置w13缩放因子的额外属性
        set_weight_attrs(w2_weight_scale, extra_weight_attrs) # 设置w2缩放因子的额外属性

        # INPUT_SCALES # 输入缩放因子
        if self.static_input_scales: # 如果使用静态输入缩放
            assert ( # 断言
                self.input_quant.strategy == QuantizationStrategy.TENSOR # 输入量化策略必须为逐张量
            ), "Only per-tensor quantization is supported for static input scales" # 静态输入缩放仅支持逐张量量化
            w13_input_scale = torch.nn.Parameter( # 创建w13输入缩放参数
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False # 形状为[专家数]
            )
            layer.register_parameter("w13_input_scale", w13_input_scale) # 在层中注册w13输入缩放参数
            set_weight_attrs(w13_input_scale, extra_weight_attrs) # 设置w13输入缩放的额外属性

            w2_input_scale = torch.nn.Parameter( # 创建w2输入缩放参数
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False # 形状为[专家数]
            )
            layer.register_parameter("w2_input_scale", w2_input_scale) # 在层中注册w2输入缩放参数
            set_weight_attrs(w2_input_scale, extra_weight_attrs) # 设置w2输入缩放的额外属性
        else: # 否则（动态输入缩放）
            layer.w13_input_scale = None # 设置w13输入缩放为None
            layer.w2_input_scale = None # 设置w2输入缩放为None

    def process_weights_after_loading(self, layer: torch.nn.Module | FusedMoE) -> None: # 权重加载后处理方法
        """权重加载后的后处理：处理输入缩放合并、FNUZ归一化、权重量化策略转换和权重重排"""
        # Fp8 moe kernels require a single activation scale.
        # FP8 MoE内核需要单一的激活缩放值。
        # We take the max of all the scales in case they differ.
        # 如果缩放值不同，我们取所有缩放值的最大值。
        if self.static_input_scales: # 如果使用静态输入缩放
            if layer.w13_input_scale is None or layer.w2_input_scale is None: # 如果任一输入缩放为None
                raise ValueError( # 抛出值错误
                    "QuantConfig has static quantization, but found "
                    "activation scales are None." # 量化配置为静态量化，但发现激活缩放为None
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d( # 检查各专家的输入缩放是否一致
                layer.w2_input_scale
            ):
                logger.warning( # 记录警告
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer." # 发现FP8 MoE层的输入缩放不一致，使用每层各专家的最大值
                )
            layer.w13_input_scale = torch.nn.Parameter( # 将w13输入缩放设为各专家最大值
                layer.w13_input_scale.max(), requires_grad=False # 取最大值，不需要梯度
            )
            layer.w2_input_scale = torch.nn.Parameter( # 将w2输入缩放设为各专家最大值
                layer.w2_input_scale.max(), requires_grad=False # 取最大值，不需要梯度
            )

        if is_fp8_fnuz(): # 如果是FNUZ格式
            # Normalize the weights and scales # 归一化权重和缩放
            w13_weight, w13_weight_scale, w13_input_scale = ( # 归一化w13权重、缩放和输入缩放
                normalize_e4m3fn_to_e4m3fnuz( # 调用E4M3FN到E4M3FNUZ归一化函数
                    layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale # 传入w13的权重、缩放和输入缩放
                )
            )
            w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz( # 归一化w2权重、缩放和输入缩放
                layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale # 传入w2的权重、缩放和输入缩放
            )
            # Reset the parameter # 重置参数
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False) # 重置w13权重参数
            layer.w13_weight_scale = torch.nn.Parameter( # 重置w13权重缩放参数
                w13_weight_scale, requires_grad=False # 不需要梯度
            )
            if w13_input_scale is not None: # 如果w13输入缩放不为None
                layer.w13_input_scale = torch.nn.Parameter( # 重置w13输入缩放参数
                    w13_input_scale, requires_grad=False # 不需要梯度
                )
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False) # 重置w2权重参数
            layer.w2_weight_scale = torch.nn.Parameter( # 重置w2权重缩放参数
                w2_weight_scale, requires_grad=False # 不需要梯度
            )
            if w2_input_scale is not None: # 如果w2输入缩放不为None
                layer.w2_input_scale = torch.nn.Parameter( # 重置w2输入缩放参数
                    w2_input_scale, requires_grad=False # 不需要梯度
                )
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR: # 如果是逐张量策略
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # FP8 MoE内核需要每个专家的w13使用单一权重缩放值。
            # We take the max then dequant and requant each expert.
            # 我们取最大值，然后对每个专家进行反量化和重新量化。
            assert layer.w13_weight_scale is not None # 断言w13权重缩放不为None
            shard_size = layer.intermediate_size_per_partition # 获取分片大小
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values # 计算每个专家w13的最大缩放值
            for expert_id in range(layer.num_local_experts): # 遍历每个本地专家
                start = 0 # 起始索引
                for shard_id in range(2): # 遍历w1和w3两个分片
                    dq_weight = per_tensor_dequantize( # 逐张量反量化
                        layer.w13_weight[expert_id][start : start + shard_size, :], # 获取当前分片权重
                        layer.w13_weight_scale[expert_id][shard_id], # 使用对应的缩放值
                    )
                    ( # 重新量化
                        layer.w13_weight[expert_id][start : start + shard_size, :], # 更新当前分片权重
                        _, # 忽略返回的缩放值
                    ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id]) # 使用最大缩放值进行FP8量化

                    start += shard_size # 更新起始索引

            layer.w13_weight_scale = torch.nn.Parameter( # 更新w13权重缩放为最大值
                max_w13_scales, requires_grad=False # 不需要梯度
            )

        if self.weight_quant.strategy == QuantizationStrategy.CHANNEL and _use_aiter: # 如果是逐通道策略且使用AITER
            with torch.no_grad(): # 在无梯度计算上下文中
                # Pre-shuffle weights # 预重排权重
                layer.w13_weight = torch.nn.Parameter( # 重排w13权重
                    shuffle_weight(layer.w13_weight.data, (16, 16)), # 使用AITER重排，块大小(16,16)
                    requires_grad=False, # 不需要梯度
                )
                torch.cuda.empty_cache() # 清空CUDA缓存
                layer.w2_weight = torch.nn.Parameter( # 重排w2权重
                    shuffle_weight(layer.w2_weight.data, (16, 16)), # 使用AITER重排，块大小(16,16)
                    requires_grad=False, # 不需要梯度
                )
                torch.cuda.empty_cache() # 清空CUDA缓存

        if ( # 如果满足以下条件
            self.weight_quant.strategy == QuantizationStrategy.BLOCK # 权重量化策略为分块
            and self.use_flashinfer_trtllm # 且使用FlashInfer TRTLLM后端
        ):
            layer.w13_weight = torch.nn.Parameter( # 交换w13权重从w13排列到w31排列
                swap_w13_to_w31(layer.w13_weight.data), # 调用w13到w31交换函数
                requires_grad=False, # 不需要梯度
            )
            layer.w13_weight_scale = torch.nn.Parameter( # 交换w13权重缩放从w13排列到w31排列
                swap_w13_to_w31(layer.w13_weight_scale.data), # 调用w13到w31交换函数
                requires_grad=False, # 不需要梯度
            )

    def create_moe_runner( # 创建MoE运行器方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig # 接收层和MoE运行器配置
    ):
        """创建并初始化MoE运行器，根据后端配置选择AITER或Triton"""
        self.moe_runner_config = moe_runner_config # 保存MoE运行器配置
        moe_runner_backend = get_moe_runner_backend() # 获取MoE运行器后端
        if moe_runner_backend.is_auto(): # 如果后端为自动选择
            if ( # 如果满足AITER条件
                _use_aiter # 启用AITER
                and self.weight_quant.strategy == QuantizationStrategy.CHANNEL # 逐通道策略
                and get_moe_a2a_backend().supports_aiter() # All-to-All后端支持AITER
            ):
                moe_runner_backend = MoeRunnerBackend.AITER # 选择AITER后端
            else: # 否则
                moe_runner_backend = MoeRunnerBackend.TRITON # 选择Triton后端

        if ( # 如果后端为AITER、Triton、FlashInfer TRTLLM或FlashInfer TRTLLM路由模式
            moe_runner_backend.is_aiter()
            or moe_runner_backend.is_triton()
            or moe_runner_backend.is_flashinfer_trtllm()
            or moe_runner_backend.is_flashinfer_trtllm_routed()
        ):
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config) # 创建MoE运行器
        else: # 否则
            # TODO(cwan): refactor other backends # 待办(cwan)：重构其他后端
            pass # 暂不处理

    def apply_weights( # 应用权重方法，执行MoE推理
        self,
        layer: torch.nn.Module, # 目标神经网络层
        dispatch_output: StandardDispatchOutput, # 标准分发输出
    ) -> CombineInput: # 返回合并输入

        x = dispatch_output.hidden_states # 获取隐藏状态
        topk_output = dispatch_output.topk_output # 获取Top-K路由输出

        moe_runner_config = self.moe_runner_config # 获取MoE运行器配置

        if self.runner.runner_backend.is_aiter(): # 如果使用AITER后端
            from sglang.srt.layers.moe.moe_runner.aiter import ( # 导入AITER MoE量化信息类
                AiterMoeQuantInfo,
                AiterQuantType,
            )

            assert not moe_runner_config.no_combine, "unsupported" # 断言不支持no_combine模式
            quant_info = AiterMoeQuantInfo( # 创建AITER MoE量化信息
                w13_weight=layer.w13_weight, # w13权重
                w2_weight=layer.w2_weight, # w2权重
                quant_type=AiterQuantType.PER_TOKEN, # 量化类型为逐token
                w13_scale=layer.w13_weight_scale, # w13权重缩放
                w2_scale=layer.w2_weight_scale, # w2权重缩放
                a13_scale=layer.w13_input_scale, # w13输入缩放
                a2_scale=layer.w2_input_scale, # w2输入缩放
            )
            return self.runner.run(dispatch_output, quant_info) # 使用AITER运行器执行计算
        elif self.weight_quant.strategy == QuantizationStrategy.BLOCK: # 如果是分块量化策略
            if self.use_flashinfer_trtllm: # 如果使用FlashInfer TRTLLM后端
                from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import ( # 导入激活类型获取函数
                    get_activation_type,
                )

                activation_type = get_activation_type( # 获取激活类型
                    moe_runner_config.activation, # 运行器配置中的激活类型
                    is_gated=moe_runner_config.is_gated, # 是否为门控激活
                )
                quant_info = FlashInferTrtllmFp8MoeQuantInfo( # 创建FlashInfer TRTLLM FP8 MoE量化信息
                    w13_weight=layer.w13_weight, # w13权重
                    w2_weight=layer.w2_weight, # w2权重
                    global_num_experts=layer.num_experts, # 全局专家数量
                    local_expert_offset=layer.moe_ep_rank * layer.num_local_experts, # 本地专家偏移量
                    local_num_experts=layer.num_local_experts, # 本地专家数量
                    intermediate_size=layer.w2_weight.shape[2], # 中间层大小
                    routing_method_type=layer.routing_method_type, # 路由方法类型
                    block_quant=self.block_quant, # 是否为分块量化
                    weight_block_k=self.weight_block_size[1], # K方向权重分块大小
                    w13_weight_scale_inv=layer.w13_weight_scale, # w13权重缩放逆值
                    w2_weight_scale_inv=layer.w2_weight_scale, # w2权重缩放逆值
                    activation_type=activation_type, # 激活类型
                )
            else: # 否则（使用Triton后端的分块量化）
                quant_info = TritonMoeQuantInfo( # 创建Triton MoE量化信息
                    w13_weight=layer.w13_weight, # w13权重
                    w2_weight=layer.w2_weight, # w2权重
                    use_fp8_w8a8=True, # 使用FP8 W8A8
                    w13_scale=layer.w13_weight_scale, # w13权重缩放
                    w2_scale=layer.w2_weight_scale, # w2权重缩放
                    a13_scale=layer.w13_input_scale, # w13输入缩放
                    a2_scale=layer.w2_input_scale, # w2输入缩放
                    block_shape=self.weight_block_size, # 分块形状
                )
            return self.runner.run(dispatch_output, quant_info) # 使用运行器执行计算
        else: # 否则（逐张量或逐通道策略）
            quant_info = TritonMoeQuantInfo( # 创建Triton MoE量化信息
                w13_weight=layer.w13_weight, # w13权重
                w2_weight=layer.w2_weight, # w2权重
                use_fp8_w8a8=True, # 使用FP8 W8A8
                per_channel_quant=self.weight_quant.strategy # 是否为逐通道量化
                == QuantizationStrategy.CHANNEL, # 与逐通道策略比较
                w13_scale=layer.w13_weight_scale, # w13权重缩放
                w2_scale=layer.w2_weight_scale, # w2权重缩放
                a13_scale=layer.w13_input_scale, # w13输入缩放
                a2_scale=layer.w2_input_scale, # w2输入缩放
            )
            return self.runner.run(dispatch_output, quant_info) # 使用运行器执行计算
