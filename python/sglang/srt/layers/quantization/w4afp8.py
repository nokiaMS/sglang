# W4A8-FP8混合精度量化方法实现文件
# 实现4位权重/8位FP8激活的混合精度量化配置和方法
# 包括W4AFp8Config配置类、interleave_scales缩放因子交错函数、
# W4AFp8MoEMethod混合专家量化方法(支持标准模式和DeepEP模式)
from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 导入日志模块
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # 导入类型提示

import torch  # 导入PyTorch深度学习框架
from torch.nn import Module  # 导入神经网络模块基类
from torch.nn.parameter import Parameter  # 导入参数类

from sglang.srt.layers.quantization.base_config import (  # 导入量化基础配置类
    FusedMoEMethodBase,  # 融合MoE方法基类
    QuantizationConfig,  # 量化配置基类
    QuantizeMethodBase,  # 量化方法基类
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod  # 导入FP8线性方法
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 导入未量化线性方法
from sglang.srt.layers.quantization.utils import is_layer_skipped  # 导入层跳过判断函数
from sglang.srt.utils import set_weight_attrs  # 导入权重属性设置函数

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE  # 导入DeepEP MoE层
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入token分发器类型
        CombineInput,  # 合并输入
        DeepEPLLDispatchOutput,  # DeepEP低延迟分发输出
        DeepEPNormalDispatchOutput,  # DeepEP普通分发输出
        StandardDispatchOutput,  # 标准分发输出
    )

ACTIVATION_SCHEMES = ["static", "dynamic"]  # 支持的激活量化方案列表

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


class W4AFp8Config(QuantizationConfig):  # W4A8-FP8混合精度量化配置类
    """Config class for MIXED_PRECISION W4AFp8."""
    # MIXED_PRECISION W4AFp8配置类

    def __init__(  # 初始化方法
        self,
        is_checkpoint_fp8_serialized: bool = True,  # 检查点是否以FP8序列化
        is_checkpoint_w4afp8_serialized: bool = True,  # 检查点是否以W4AFp8序列化
        linear_activation_scheme: str = "dynamic",  # 线性层激活量化方案(动态)
        moe_activation_scheme: str = "static",  # MoE层激活量化方案(静态)
        ignored_layers: Optional[List[str]] = None,  # 被忽略(不量化)的层列表
        weight_block_size: Optional[List[int]] = None,  # 权重块大小
        group_size: int = 128,  # 量化组大小
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized  # 保存FP8序列化标志
        self.is_checkpoint_w4afp8_serialized = is_checkpoint_w4afp8_serialized  # 保存W4AFp8序列化标志
        if is_checkpoint_w4afp8_serialized:  # 如果是W4AFp8序列化的检查点
            logger.warning("Detected w4afp8 checkpoint. Please note that")  # 记录警告
        if moe_activation_scheme not in ACTIVATION_SCHEMES:  # 如果MoE激活方案不支持
            raise ValueError(f"Unsupported activation scheme {moe_activation_scheme}")  # 抛出错误
        self.linear_activation_scheme = linear_activation_scheme  # 保存线性层激活方案
        self.moe_activation_scheme = moe_activation_scheme  # 保存MoE层激活方案
        self.ignored_layers = ignored_layers or []  # 保存忽略层列表
        self.weight_block_size = [128, 128]  # 设置权重块大小为128x128
        self.group_size = group_size  # 保存量化组大小

    @classmethod
    def get_name(cls) -> str:  # 获取量化方法名称
        return "w4afp8"  # 返回名称

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:  # 获取支持的激活数据类型
        return [torch.bfloat16, torch.float8_e4m3fn]  # 支持bfloat16和FP8 E4M3

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低GPU计算能力要求
        return 90  # 需要计算能力9.0(SM90/H100)

    @classmethod
    def get_config_filenames(cls) -> List[str]:  # 获取配置文件名列表
        return []  # 无配置文件

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W4AFp8Config:  # 从配置字典创建配置实例
        quant_method = cls.get_from_keys(config, ["quant_method"])  # 获取量化方法名
        is_checkpoint_fp8_serialized = "fp8" in quant_method  # 检查是否包含FP8
        is_checkpoint_w4afp8_serialized = "w4afp8" in quant_method  # 检查是否包含W4AFp8
        linear_activation_scheme = "dynamic"  # 线性层使用动态激活量化
        moe_activation_scheme = "static"  # MoE层使用静态激活量化
        weight_block_size = [128, 128]  # 权重块大小
        return cls(  # 返回配置实例
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,  # FP8序列化标志
            is_checkpoint_w4afp8_serialized=is_checkpoint_w4afp8_serialized,  # W4AFp8序列化标志
            linear_activation_scheme=linear_activation_scheme,  # 线性层激活方案
            moe_activation_scheme=moe_activation_scheme,  # MoE层激活方案
            weight_block_size=weight_block_size,  # 权重块大小
        )

    def get_quant_method(  # 获取量化方法
        self, layer: torch.nn.Module, prefix: str  # 网络层和前缀
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase  # 导入线性基类
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE类

        if isinstance(layer, LinearBase):  # 如果是线性层
            if is_layer_skipped(prefix, self.ignored_layers):  # 如果层被跳过
                return UnquantizedLinearMethod()  # 返回未量化方法
            return Fp8LinearMethod(self)  # 返回FP8线性方法
        elif isinstance(layer, FusedMoE):  # 如果是融合MoE层
            return W4AFp8MoEMethod(self)  # 返回W4AFp8 MoE方法
        return None  # 其他情况返回None

    def get_scaled_act_names(self) -> List[str]:  # 获取需要缩放的激活名称
        return []  # 无需缩放的激活


def interleave_scales(scales: torch.Tensor) -> torch.Tensor:  # 交错缩放因子
    """Interleave scales in groups of 4 similar to TRT-LLM implementation."""
    # 以4为一组交错缩放因子，类似于TRT-LLM的实现
    s_shape = scales.shape  # 获取缩放因子形状
    # Reshape to separate groups of 4
    # 重塑以分离4的分组
    alignment = 4 if s_shape[2] % 4 == 0 else 1  # 如果最后一维可被4整除则对齐为4，否则为1
    scales_interleaved = scales.reshape(  # 重塑缩放因子
        s_shape[0], s_shape[1], (s_shape[2] // alignment), alignment  # 分离对齐组
    )
    # Permute dimensions to interleave
    # 交换维度以实现交错
    scales_interleaved = scales_interleaved.permute(0, 2, 1, 3)  # 交换第1和第2维
    # Reshape back to original dimensions but with interleaved values
    # 重塑回原始维度但值已交错
    scales_interleaved = scales_interleaved.reshape(  # 重塑交错后的缩放因子
        s_shape[0], s_shape[2] // alignment, s_shape[1] * alignment  # 新的维度
    )
    return scales_interleaved.contiguous()  # 返回连续的交错缩放因子


class W4AFp8MoEMethod(FusedMoEMethodBase):  # W4A8-FP8 MoE量化方法类
    def __init__(self, quant_config: W4AFp8Config):  # 初始化方法
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建MoE权重
        self,
        layer: Module,  # 目标网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入MoE权重缩放支持枚举

        assert "weight_loader" in extra_weight_attrs  # 断言存在weight_loader

        # Fused gate_up_proj (column parallel)
        # 融合的gate_up_proj(列并行)
        w13_weight = torch.nn.Parameter(  # 创建w13权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量
                intermediate_size_per_partition * 2,  # 中间层大小*2(门控)
                hidden_size // 2,  # 隐藏层大小/2(FP4打包)
                dtype=torch.int8,  # int8类型(每2个4位打包为1个int8)
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight)  # 注册w13权重
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置额外属性

        # down_proj (row parallel)
        # down_proj(行并行)
        w2_weight = torch.nn.Parameter(  # 创建w2权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量
                hidden_size,  # 隐藏层大小
                intermediate_size_per_partition // 2,  # 中间层大小/2(FP4打包)
                dtype=torch.int8,  # int8类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight", w2_weight)  # 注册w2权重
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置额外属性

        extra_weight_attrs.update(  # 更新额外属性
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}  # 设置量化方法为组级
        )
        w13_weight_scale = torch.nn.Parameter(  # 创建w13权重缩放因子参数
            torch.zeros(  # 创建零张量
                num_experts,  # 专家数量
                2 * intermediate_size_per_partition,  # 2倍中间层大小
                hidden_size // self.quant_config.group_size,  # 隐藏层/组大小
                dtype=torch.float32,  # float32类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)  # 注册w13权重缩放因子
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)  # 设置额外属性

        w2_weight_scale = torch.nn.Parameter(  # 创建w2权重缩放因子参数
            torch.zeros(  # 创建零张量
                num_experts,  # 专家数量
                hidden_size,  # 隐藏层大小
                intermediate_size_per_partition // self.quant_config.group_size,  # 中间层/组大小
                dtype=torch.float32,  # float32类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)  # 注册w2权重缩放因子
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)  # 设置额外属性

        # Input scales
        # 输入缩放因子
        w13_input_scale = torch.nn.Parameter(  # 创建w13输入缩放因子参数
            torch.ones((num_experts, 2), dtype=torch.bfloat16),  # 每个专家2个缩放值
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)  # 注册w13输入缩放因子
        set_weight_attrs(w13_input_scale, extra_weight_attrs)  # 设置额外属性

        w2_input_scale = torch.nn.Parameter(  # 创建w2输入缩放因子参数
            torch.ones(num_experts, dtype=torch.bfloat16),  # 每个专家1个缩放值
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)  # 注册w2输入缩放因子
        set_weight_attrs(w2_input_scale, extra_weight_attrs)  # 设置额外属性

        # Pre-populate the strides
        # 预填充步长
        device = layer.w13_weight.device  # 获取设备

        self.a_strides1 = torch.full(  # GEMM1输入矩阵步长
            (num_experts, 3),  # 每个专家3个步长
            hidden_size,  # 步长值为隐藏层大小
            device=device,  # 设备
            dtype=torch.int64,  # int64类型
        )
        self.c_strides1 = torch.full(  # GEMM1输出矩阵步长
            (num_experts, 3),  # 每个专家3个步长
            2 * intermediate_size_per_partition,  # 步长值为2倍中间层大小
            device=device,  # 设备
            dtype=torch.int64,  # int64类型
        )
        self.a_strides2 = torch.full(  # GEMM2输入矩阵步长
            (num_experts, 3),  # 每个专家3个步长
            intermediate_size_per_partition,  # 步长值为中间层大小
            device=device,  # 设备
            dtype=torch.int64,  # int64类型
        )
        self.c_strides2 = torch.full(  # GEMM2输出矩阵步长
            (num_experts, 3),  # 每个专家3个步长
            hidden_size,  # 步长值为隐藏层大小
            device=device,  # 设备
            dtype=torch.int64,  # int64类型
        )
        self.b_strides1 = self.a_strides1  # GEMM1权重矩阵步长等于输入步长
        self.s_strides13 = self.c_strides1  # GEMM1缩放因子步长等于输出步长
        self.b_strides2 = self.a_strides2  # GEMM2权重矩阵步长等于输入步长
        self.s_strides2 = self.c_strides2  # GEMM2缩放因子步长等于输出步长

        self.expert_offsets = torch.empty(  # 专家偏移量
            (num_experts + 1), dtype=torch.int32, device=device  # 专家数+1个int32
        )
        self.problem_sizes1 = torch.empty(  # GEMM1问题大小
            (num_experts, 3), dtype=torch.int32, device=device  # 每个专家3个int32
        )
        self.problem_sizes2 = torch.empty(  # GEMM2问题大小
            (num_experts, 3), dtype=torch.int32, device=device  # 每个专家3个int32
        )

        return  # 返回

    def process_weights_after_loading(self, layer: Module) -> None:  # 加载权重后处理
        dtype = torch.bfloat16  # 使用bfloat16类型
        device = layer.w2_weight.device  # 获取设备

        # Interleave w13_weight_scale (gate_up_proj)
        # 交错w13权重缩放因子(gate_up投影)
        w13_weight_scale = layer.w13_weight_scale_inv.to(dtype)  # 转为bfloat16
        w13_weight_scale = interleave_scales(w13_weight_scale)  # 交错缩放因子
        layer.w13_weight_scale_inv = Parameter(w13_weight_scale, requires_grad=False)  # 更新参数

        # Interleave w2_weight_scale (down_proj)
        # 交错w2权重缩放因子(down投影)
        w2_weight_scale = layer.w2_weight_scale_inv.to(dtype)  # 转为bfloat16
        w2_weight_scale = interleave_scales(w2_weight_scale)  # 交错缩放因子
        layer.w2_weight_scale_inv = Parameter(w2_weight_scale, requires_grad=False)  # 更新参数

        # Process input scales
        # 处理输入缩放因子
        w13_input_scale_max = layer.w13_input_scale.max().to(torch.float32).item()  # 取w13输入缩放最大值
        new_w13_input_scale = torch.tensor(  # 创建新的w13输入缩放因子
            [w13_input_scale_max],  # 使用最大值
            dtype=torch.float32,  # float32类型
            device=device,  # 设备
        )
        layer.w13_input_scale = Parameter(new_w13_input_scale, requires_grad=False)  # 更新参数

        w2_input_scale_max = layer.w2_input_scale.max().to(torch.float32).item()  # 取w2输入缩放最大值
        new_w2_input_scale = torch.tensor(  # 创建新的w2输入缩放因子
            [w2_input_scale_max], dtype=torch.float32, device=device  # 使用最大值
        )
        layer.w2_input_scale = Parameter(new_w2_input_scale, requires_grad=False)  # 更新参数

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 层和配置
    ):
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置

    def apply(  # 应用MoE方法(标准模式)
        self,
        layer: Module,  # 目标网络层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:

        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe  # 导入CUTLASS W4A8 MoE函数
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入类

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取topk输出
        topk_weights, topk_ids, _ = topk_output  # 解包topk输出

        output = cutlass_w4a8_moe(  # 调用CUTLASS W4A8 MoE计算
            x,  # 输入隐藏状态
            layer.w13_weight,  # w13权重
            layer.w2_weight,  # w2权重
            layer.w13_weight_scale_inv,  # w13权重缩放因子
            layer.w2_weight_scale_inv,  # w2权重缩放因子
            topk_weights,  # topk权重
            topk_ids,  # topk专家ID
            self.a_strides1,  # GEMM1输入步长
            self.b_strides1,  # GEMM1权重步长
            self.c_strides1,  # GEMM1输出步长
            self.a_strides2,  # GEMM2输入步长
            self.b_strides2,  # GEMM2权重步长
            self.c_strides2,  # GEMM2输出步长
            self.s_strides13,  # GEMM1缩放步长
            self.s_strides2,  # GEMM2缩放步长
            self.expert_offsets,  # 专家偏移量
            self.problem_sizes1,  # GEMM1问题大小
            self.problem_sizes2,  # GEMM2问题大小
            layer.w13_input_scale,  # w13输入缩放因子
            layer.w2_input_scale,  # w2输入缩放因子
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,  # 路由缩放因子
        )
        return StandardCombineInput(hidden_states=output)  # 返回标准合并输入

    def apply_deepep_ll(  # 应用DeepEP低延迟模式
        self,
        layer: DeepEPMoE,  # DeepEP MoE层
        dispatch_output: DeepEPLLDispatchOutput,  # DeepEP低延迟分发输出
    ) -> torch.Tensor:

        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe_deepep_ll  # 导入DeepEP低延迟函数

        hidden_states, hidden_scales, topk_ids, _, masked_m, _ = dispatch_output  # 解包分发输出

        output = cutlass_w4a8_moe_deepep_ll(  # 调用CUTLASS W4A8 MoE DeepEP低延迟计算
            hidden_states,  # 隐藏状态
            hidden_scales,  # 隐藏状态缩放因子
            layer.w13_weight,  # w13权重
            layer.w2_weight,  # w2权重
            layer.w13_weight_scale_inv,  # w13权重缩放因子
            layer.w2_weight_scale_inv,  # w2权重缩放因子
            topk_ids,  # topk专家ID
            masked_m,  # 掩码矩阵
            layer.quant_method.a_strides1,  # GEMM1输入步长
            layer.quant_method.b_strides1,  # GEMM1权重步长
            layer.quant_method.c_strides1,  # GEMM1输出步长
            layer.quant_method.a_strides2,  # GEMM2输入步长
            layer.quant_method.b_strides2,  # GEMM2权重步长
            layer.quant_method.c_strides2,  # GEMM2输出步长
            layer.quant_method.s_strides13,  # GEMM1缩放步长
            layer.quant_method.s_strides2,  # GEMM2缩放步长
            layer.quant_method.expert_offsets,  # 专家偏移量
            layer.quant_method.problem_sizes1,  # GEMM1问题大小
            layer.quant_method.problem_sizes2,  # GEMM2问题大小
            layer.w13_input_scale,  # w13输入缩放因子
            layer.w2_input_scale,  # w2输入缩放因子
        )

        return output  # 返回输出

    def apply_deepep_normal(  # 应用DeepEP普通模式
        self,
        layer: DeepEPMoE,  # DeepEP MoE层
        dispatch_output: DeepEPNormalDispatchOutput,  # DeepEP普通分发输出
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.cutlass_w4a8_moe import (  # 导入CUTLASS W4A8 MoE DeepEP普通模式函数
            cutlass_w4a8_moe_deepep_normal,
        )

        hidden_states, topk_idx, topk_weights = (  # 解包分发输出
            dispatch_output.hidden_states,  # 隐藏状态
            dispatch_output.topk_ids,  # topk专家ID
            dispatch_output.topk_weights,  # topk权重
        )
        if isinstance(hidden_states, tuple):  # 如果隐藏状态是元组
            hidden_states = hidden_states[0]  # 取第一个元素

        num_tokens = hidden_states.shape[0]  # 获取token数量
        if num_tokens > 0:  # 如果有token
            return cutlass_w4a8_moe_deepep_normal(  # 调用CUTLASS W4A8 MoE DeepEP普通模式计算
                hidden_states,  # 隐藏状态
                layer.w13_weight,  # w13权重
                layer.w2_weight,  # w2权重
                layer.w13_weight_scale_inv,  # w13权重缩放因子
                layer.w2_weight_scale_inv,  # w2权重缩放因子
                topk_weights,  # topk权重
                topk_idx,  # topk专家ID
                self.a_strides1,  # GEMM1输入步长
                self.b_strides1,  # GEMM1权重步长
                self.c_strides1,  # GEMM1输出步长
                self.a_strides2,  # GEMM2输入步长
                self.b_strides2,  # GEMM2权重步长
                self.c_strides2,  # GEMM2输出步长
                self.s_strides13,  # GEMM1缩放步长
                self.s_strides2,  # GEMM2缩放步长
                self.expert_offsets,  # 专家偏移量
                self.problem_sizes1,  # GEMM1问题大小
                self.problem_sizes2,  # GEMM2问题大小
                layer.w13_input_scale,  # w13输入缩放因子
                layer.w2_input_scale,  # w2输入缩放因子
            )
        else:  # 没有token
            return hidden_states  # 直接返回隐藏状态
