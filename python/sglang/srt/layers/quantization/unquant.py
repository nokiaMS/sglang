# 未量化方法的实现文件
# 包含未量化的嵌入方法、线性方法和混合专家(MoE)方法的实现
# 支持CPU(AMX)、GPU(CUDA/ROCm)、NPU等多种硬件平台
from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 导入日志模块
from typing import TYPE_CHECKING, List, Optional  # 导入类型提示相关模块

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

import torch  # 导入PyTorch深度学习框架
import torch.nn.functional as F  # 导入PyTorch函数式神经网络模块
from torch.nn.parameter import Parameter  # 导入参数类

from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.layers.amx_utils import (  # 导入AMX相关工具
    CPUQuantMethod,  # CPU量化方法枚举
    _amx_process_weight_after_loading,  # AMX权重加载后处理函数
)
from sglang.srt.layers.moe import (  # 导入混合专家相关模块
    MoeRunner,  # MoE运行器
    MoeRunnerBackend,  # MoE运行器后端枚举
    MoeRunnerConfig,  # MoE运行器配置
    get_deepep_mode,  # 获取DeepEP模式
    get_moe_a2a_backend,  # 获取MoE全对全通信后端
    get_moe_runner_backend,  # 获取MoE运行器后端
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo  # 导入Triton MoE量化信息类
from sglang.srt.layers.quantization.base_config import (  # 导入量化基础配置类
    FusedMoEMethodBase,  # 融合MoE方法基类
    LinearMethodBase,  # 线性方法基类
    QuantizeMethodBase,  # 量化方法基类
)
from sglang.srt.layers.utils import MultiPlatformOp, copy_or_rebind_param  # 导入多平台操作和参数拷贝工具
from sglang.srt.utils import (  # 导入工具函数
    cpu_has_amx_support,  # 检查CPU是否支持AMX
    get_bool_env_var,  # 获取布尔型环境变量
    is_cpu,  # 判断是否为CPU平台
    is_hip,  # 判断是否为HIP(AMD GPU)平台
    is_npu,  # 判断是否为NPU平台
    next_power_of_2,  # 计算下一个2的幂次
    set_weight_attrs,  # 设置权重属性
    use_intel_amx_backend,  # 判断是否使用Intel AMX后端
    use_intel_xpu_backend,  # 判断是否使用Intel XPU后端
)

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入token分发器相关类型
        CombineInput,  # 合并输入类型
        DispatchOutput,  # 分发输出类型
        StandardDispatchOutput,  # 标准分发输出类型
    )


_is_cpu_amx_available = cpu_has_amx_support()  # 检查CPU AMX是否可用
_is_hip = is_hip()  # 是否为HIP平台
_is_cpu = is_cpu()  # 是否为CPU平台
_is_npu = is_npu()  # 是否为NPU平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITER(仅HIP平台)

if _use_aiter:  # 如果使用AITER
    from aiter.ops.shuffle import shuffle_weight  # 导入权重重排函数
    from aiter.tuned_gemm import tgemm  # 导入调优GEMM模块

if _is_npu:  # 如果是NPU平台
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast  # 导入NPU格式转换函数

try:  # 尝试导入flashinfer的融合MoE实现
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe  # 导入flashinfer CUTLASS融合MoE
    from flashinfer.fused_moe.core import ActivationType  # 导入激活类型枚举
except ImportError:  # 如果导入失败
    flashinfer_cutlass_fused_moe = None  # 设为None表示不可用


def swiglustep_and_mul(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:  # SwiGLU步进激活与乘法函数
    """Out-variant of swiglustep activation.
    # swiglustep激活函数的输出变体

    Writes into `out`:
    # 写入`out`:
      silu(x[:d]).clamp(max=limit) * x[d:].clamp(-limit, limit)
    """
    gate, up = x.chunk(2, dim=-1)  # 将输入沿最后一维分为两半：gate和up
    gate = F.silu(gate)  # 对gate部分应用SiLU激活函数
    gate = gate.clamp(max=limit)  # 将gate值限制在limit以下
    up = up.clamp(min=-limit, max=limit)  # 将up值限制在[-limit, limit]范围内
    out = gate * up  # 将gate和up逐元素相乘
    return out  # 返回输出


class UnquantizedEmbeddingMethod(QuantizeMethodBase):  # 未量化嵌入方法类
    """Unquantized method for embeddings."""
    # 未量化的嵌入层方法

    def create_weights(  # 创建嵌入层权重
        self,
        layer: torch.nn.Module,  # 目标网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: List[int],  # 输出分区大小列表
        input_size: int,  # 输入总大小
        output_size: int,  # 输出总大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        """Create weights for embedding layer."""
        # 为嵌入层创建权重
        weight = Parameter(  # 创建参数
            torch.empty(  # 创建空张量
                sum(output_partition_sizes),  # 输出维度大小之和
                input_size_per_partition,  # 输入分区大小
                dtype=params_dtype,  # 数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})  # 设置权重的输入维度和输出维度属性
        layer.register_parameter("weight", weight)  # 在层中注册权重参数
        set_weight_attrs(weight, extra_weight_attrs)  # 设置额外的权重属性

    def apply(  # 应用嵌入层前向计算
        self,
        layer: torch.nn.Module,  # 目标网络层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置项(可选)
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)  # 执行线性变换

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:  # 嵌入查找操作
        return F.embedding(input_, layer.weight)  # 执行嵌入查找


class UnquantizedLinearMethod(LinearMethodBase):  # 未量化线性方法类
    """Linear method without quantization."""
    # 无量化的线性方法

    def create_weights(  # 创建线性层权重
        self,
        layer: torch.nn.Module,  # 目标网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: List[int],  # 输出分区大小列表
        input_size: int,  # 输入总大小
        output_size: int,  # 输出总大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        weight = Parameter(  # 创建参数
            torch.empty(  # 创建空张量
                sum(output_partition_sizes),  # 输出维度大小之和
                input_size_per_partition,  # 输入分区大小
                dtype=params_dtype,  # 数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})  # 设置权重的输入维度和输出维度属性
        layer.register_parameter("weight", weight)  # 在层中注册权重参数
        set_weight_attrs(weight, extra_weight_attrs)  # 设置额外的权重属性

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理
        if _is_cpu and _is_cpu_amx_available:  # 如果是CPU且AMX可用
            _amx_process_weight_after_loading(layer, ["weight"])  # 对权重进行AMX处理

    def apply(  # 应用线性层前向计算
        self,
        layer: torch.nn.Module,  # 目标网络层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置项(可选)
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):  # 如果使用Intel AMX后端
            x_shapes = x.shape  # 保存原始形状
            if len(x_shapes) == 3:  # 如果输入是3维的
                x = x.view(-1, x.shape[-1])  # 将3维输入展平为2维
            output = torch.ops.sgl_kernel.weight_packed_linear(  # 调用AMX打包权重的线性运算
                x,  # 输入
                layer.weight,  # 权重
                bias,  # 偏置
                True,  # is_vnni # 使用VNNI格式
            )
            if len(x_shapes) == 3:  # 如果原始输入是3维的
                output = output.view(x_shapes[0], x_shapes[1], -1)  # 恢复3维形状
            return output  # 返回输出

        elif _use_aiter and type(layer.weight.data) is torch.Tensor:  # 如果使用AITER且权重是普通张量
            return tgemm.mm(x, layer.weight, bias, otype=x.dtype)  # 使用AITER调优GEMM进行矩阵乘法

        return F.linear(x, layer.weight, bias)  # 默认使用标准线性变换


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):  # 未量化融合MoE方法类
    """MoE method without quantization."""
    # 无量化的混合专家方法

    def __init__(  # 初始化方法
        self,
        use_triton_kernels: bool = False,  # 是否使用Triton内核
        use_flashinfer_trtllm_moe: bool = False,  # 是否使用flashinfer TRT-LLM MoE
        use_deep_gemm: bool = False,  # 是否使用DeepGEMM
    ):
        super().__init__()  # 调用父类初始化
        self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()  # 检查是否使用flashinfer cutlass后端
        self.use_triton_kernels = use_triton_kernels  # 保存Triton内核使用标志
        self.with_bias = False  # 默认不使用偏置
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe  # 保存flashinfer TRT-LLM MoE使用标志
        self.use_deep_gemm = use_deep_gemm  # 保存DeepGEMM使用标志
        self._cache_permute_indices = dict({})  # 缓存排列索引的字典

    def create_weights(  # 创建MoE权重
        self,
        layer: torch.nn.Module,  # 目标网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        with_bias: bool = False,  # 是否使用偏置
        **extra_weight_attrs,  # 额外权重属性
    ):
        self.with_bias = with_bias  # 保存偏置标志

        # Fused gate_up_proj (column parallel)
        # 融合的gate_up_proj(列并行)
        w13_up_dim = (  # 计算w13的上投影维度
            2 * intermediate_size_per_partition  # 如果是门控激活则乘以2
            if layer.moe_runner_config.is_gated  # 检查是否为门控激活
            else intermediate_size_per_partition  # 否则使用原始中间层大小
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)  # 设置w13权重的N和K维度
        if self.use_triton_kernels:  # 如果使用Triton内核
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n  # 交换N和K维度
        w13_weight = torch.nn.Parameter(  # 创建w13权重参数
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),  # 创建空张量
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight)  # 注册w13权重参数
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置额外权重属性

        if self.with_bias:  # 如果使用偏置
            w13_weight_bias = torch.nn.Parameter(  # 创建w13偏置参数
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),  # 创建空偏置张量
                requires_grad=False,  # 不需要梯度
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)  # 注册w13偏置参数
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)  # 设置额外偏置属性

        # down_proj (row parallel)
        # down_proj(行并行)
        w2_weight_n, w2_weight_k = (  # 设置w2权重的N和K维度
            hidden_size,  # N维度为隐藏层大小
            intermediate_size_per_partition,  # K维度为中间层分区大小
        )
        if self.use_triton_kernels:  # 如果使用Triton内核
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n  # 交换N和K维度
        w2_weight = torch.nn.Parameter(  # 创建w2权重参数
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),  # 创建空张量
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight", w2_weight)  # 注册w2权重参数
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置额外权重属性

        if self.with_bias:  # 如果使用偏置
            w2_weight_bias = torch.nn.Parameter(  # 创建w2偏置参数
                torch.empty(num_experts, hidden_size, dtype=torch.float32),  # 创建空偏置张量
                requires_grad=False,  # 不需要梯度
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)  # 注册w2偏置参数
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)  # 设置额外偏置属性

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理
        _should_use_aiter_moe = _use_aiter and (  # 判断是否应使用AITER MoE
            get_moe_runner_backend().is_auto() or get_moe_runner_backend().is_aiter()  # 后端为auto或aiter
        )
        if _should_use_aiter_moe:  # 如果应使用AITER MoE
            copy_or_rebind_param(  # 复制或重新绑定w13权重(经过shuffle)
                layer, "w13_weight", shuffle_weight(layer.w13_weight.data, (16, 16))  # 对w13权重进行(16,16)块重排
            )
            torch.cuda.empty_cache()  # 清空GPU缓存
            copy_or_rebind_param(  # 复制或重新绑定w2权重(经过shuffle)
                layer, "w2_weight", shuffle_weight(layer.w2_weight.data, (16, 16))  # 对w2权重进行(16,16)块重排
            )
            torch.cuda.empty_cache()  # 清空GPU缓存

        # Pack weight for get better performance on CPU
        # 打包权重以在CPU上获得更好的性能
        if _is_cpu and _is_cpu_amx_available:  # 如果是CPU且AMX可用
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])  # 对权重进行AMX打包处理
            if hasattr(layer, "w13_weight_bias"):  # 如果存在w13偏置
                layer.w13_weight_bias = Parameter(  # 将w13偏置转为float32参数
                    layer.w13_weight_bias.float(), requires_grad=False  # 不需要梯度
                )
            if hasattr(layer, "w2_weight_bias"):  # 如果存在w2偏置
                layer.w2_weight_bias = Parameter(  # 将w2偏置转为float32参数
                    layer.w2_weight_bias.float(), requires_grad=False  # 不需要梯度
                )

        if (  # 检查是否需要为DeepGEMM设置量化配置
            self.use_deep_gemm  # 使用DeepGEMM
            and layer.w13_weight.dtype == torch.bfloat16  # 权重类型为bfloat16
            and get_moe_a2a_backend().is_deepep()  # 后端为DeepEP
            and get_deepep_mode().enable_low_latency()  # 启用低延迟模式
            and not _is_npu  # 非NPU平台
            and not _is_hip  # 非HIP平台
            and hasattr(layer, "dispatcher")  # 层有分发器
        ):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})  # 设置分发器输出数据类型为bf16

        # Reorder rows of W1 for fused gated activation
        # 重排W1的行以实现融合门控激活
        if self.use_flashinfer_trtllm_moe:  # 如果使用flashinfer TRT-LLM MoE
            from flashinfer.fused_moe.core import (  # 导入flashinfer融合MoE核心函数
                _maybe_get_cached_w3_w1_permute_indices,  # 获取缓存的w3/w1排列索引
                convert_to_block_layout,  # 转换为块布局
                get_w2_permute_indices_with_cache,  # 获取缓存的w2排列索引
            )

            # w1 and w3 have been swapped, so we don't need do that here
            # w1和w3已经交换过了，所以这里不需要再做
            epilogue_tile_m = 128  # 尾声平铺大小为128
            block_k = 128  # K方向块大小为128
            old_shape_w13 = layer.w13_weight.data[0].shape  # 保存w13权重原始形状
            old_shape_w2 = layer.w2_weight.data[0].shape  # 保存w2权重原始形状
            new_shape_w13 = None  # 新的w13形状(待计算)
            new_shape_w2 = None  # 新的w2形状(待计算)
            for i in range(layer.num_local_experts):  # 遍历每个本地专家
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(  # 获取w3/w1排列索引
                    self._cache_permute_indices,  # 缓存字典
                    layer.w13_weight.data[i].view(torch.uint8),  # w13权重(按uint8视图)
                    epilogue_tile_m,  # 尾声平铺大小
                )
                tmp_weights1 = (  # 根据排列索引重排w13权重
                    layer.w13_weight.data[i]  # 获取第i个专家的w13权重
                    .clone()  # 克隆以避免原地修改
                    .view(torch.uint8)[permute_indices.to(layer.w13_weight.data.device)]  # 按排列索引重排
                    .contiguous()  # 使内存连续
                )

                permute_indices = get_w2_permute_indices_with_cache(  # 获取w2排列索引
                    self._cache_permute_indices,  # 缓存字典
                    layer.w2_weight.data[i].view(torch.uint8),  # w2权重(按uint8视图)
                    epilogue_tile_m,  # 尾声平铺大小
                )
                tmp_weights2 = (  # 根据排列索引重排w2权重
                    layer.w2_weight.data[i]  # 获取第i个专家的w2权重
                    .clone()  # 克隆以避免原地修改
                    .view(torch.uint8)[permute_indices.to(layer.w2_weight.data.device)]  # 按排列索引重排
                    .contiguous()  # 使内存连续
                )

                tmp_weights1 = convert_to_block_layout(  # 将w13权重转换为块布局
                    tmp_weights1.view(torch.uint8), block_k  # 指定K方向块大小
                )
                tmp_weights2 = convert_to_block_layout(  # 将w2权重转换为块布局
                    tmp_weights2.view(torch.uint8), block_k  # 指定K方向块大小
                )

                new_shape_w13 = tmp_weights1.view(torch.bfloat16).shape  # 获取w13的新形状
                new_shape_w2 = tmp_weights2.view(torch.bfloat16).shape  # 获取w2的新形状
                layer.w13_weight.data[i] = (  # 更新w13权重数据
                    tmp_weights1.view(torch.bfloat16)  # 转回bfloat16视图
                    .contiguous()  # 使内存连续
                    .reshape(old_shape_w13)  # 重塑为原始形状
                )
                layer.w2_weight.data[i] = (  # 更新w2权重数据
                    tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)  # 转回bfloat16并重塑
                )

            layer.w13_weight.data = layer.w13_weight.data.reshape(  # 重塑w13权重整体形状
                layer.num_local_experts, *new_shape_w13  # 包含专家维度和新形状
            )
            layer.w2_weight.data = layer.w2_weight.data.reshape(  # 重塑w2权重整体形状
                layer.num_local_experts, *new_shape_w2  # 包含专家维度和新形状
            )

        if _is_npu:  # 如果是NPU平台
            for weight_name in ["w13_weight", "w2_weight"]:  # 遍历w13和w2权重
                weight = getattr(layer, weight_name)  # 获取权重属性
                origin_weight = weight.data.transpose(1, 2)  # 转置权重第1和第2维
                new_weight = origin_weight.contiguous()  # 使转置后的权重内存连续
                origin_weight.untyped_storage().resize_(0)  # 释放原始存储空间
                weight.data = npu_format_cast(new_weight)  # 将权重转换为NPU格式

        return  # 返回

    def maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(  # 可能恢复flashinfer TRT-LLM BF16权重形状用于加载
        self,
        layer: torch.nn.Module,  # 目标网络层
        param: torch.nn.Parameter,  # 参数
        weight_name: str,  # 权重名称
    ) -> None:
        """Restore canonical BF16 MoE load shapes before hot weight copy.
        # 在热权重拷贝前恢复标准的BF16 MoE加载形状

        The flashinfer TRT-LLM BF16 postprocess reshapes expert weights into
        block layout. During weight update, checkpoint tensors are in
        canonical layout and need a temporary shape restore for copy.
        # flashinfer TRT-LLM BF16后处理将专家权重重塑为块布局。
        在权重更新期间，检查点张量是标准布局，需要临时恢复形状以便拷贝。
        """
        if not get_moe_runner_backend().is_flashinfer_trtllm_routed():  # 如果不是flashinfer TRT-LLM路由后端
            return  # 直接返回

        expected_shape = None  # 期望的形状
        if weight_name.endswith(".experts.w13_weight"):  # 如果是w13权重
            w13_rows = (  # 计算w13行数
                2 * layer.intermediate_size_per_partition  # 如果是门控激活则乘以2
                if layer.moe_runner_config.is_gated  # 检查是否为门控激活
                else layer.intermediate_size_per_partition  # 否则使用原始中间层大小
            )
            expected_shape = (layer.num_local_experts, w13_rows, layer.hidden_size)  # 设置期望形状
        elif weight_name.endswith(".experts.w2_weight"):  # 如果是w2权重
            expected_shape = (  # 设置期望形状
                layer.num_local_experts,  # 专家数量
                layer.hidden_size,  # 隐藏层大小
                layer.intermediate_size_per_partition,  # 中间层分区大小
            )

        if expected_shape is None or tuple(param.data.shape) == expected_shape:  # 如果无需恢复
            return  # 直接返回

        expected_numel = expected_shape[0] * expected_shape[1] * expected_shape[2]  # 计算期望的元素总数
        if param.data.numel() != expected_numel:  # 如果元素数量不匹配
            raise RuntimeError(  # 抛出运行时错误
                f"Cannot restore flashinfer TRT-LLM BF16 MoE weight shape for {weight_name}: "  # 错误信息前半部分
                f"current shape={tuple(param.data.shape)}, expected shape={expected_shape}."  # 错误信息后半部分
            )

        param.data = param.data.reshape(expected_shape)  # 将参数数据重塑为期望形状

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 层和配置
    ):
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置
        if self.use_flashinfer_trtllm_moe:  # 如果使用flashinfer TRT-LLM MoE
            backend = (  # 确定后端类型
                MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED  # 路由模式
                if get_moe_runner_backend().is_flashinfer_trtllm_routed()  # 如果是路由模式
                else MoeRunnerBackend.FLASHINFER_TRTLLM  # 非路由模式
            )
        elif self.use_deep_gemm:  # 如果使用DeepGEMM
            backend = MoeRunnerBackend.DEEP_GEMM  # 使用DeepGEMM后端
        elif self.use_triton_kernels:  # 如果使用Triton内核
            backend = MoeRunnerBackend.TRITON_KERNELS  # 使用Triton内核后端
        else:  # 其他情况
            backend = MoeRunnerBackend.TRITON  # 默认使用Triton后端
        self.runner = MoeRunner(backend, moe_runner_config)  # 创建MoE运行器

        # Separate runner so CK-shape errors fall back to self.runner on every call.
        # 单独的运行器，以便CK形状错误时每次调用都能回退到self.runner
        self._aiter_runner: Optional[MoeRunner] = None  # AITER运行器(可选)
        if (  # 如果满足AITER运行器创建条件
            _use_aiter  # 使用AITER
            and (  # 并且
                get_moe_runner_backend().is_auto()  # 后端为auto
                or get_moe_runner_backend().is_aiter()  # 或后端为aiter
            )
            and get_moe_a2a_backend().supports_aiter()  # 且全对全后端支持AITER
        ):
            self._aiter_runner = MoeRunner(MoeRunnerBackend.AITER, moe_runner_config)  # 创建AITER运行器

    @property
    def load_up_proj_weight_first(self) -> bool:  # 是否先加载上投影权重
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        # FlashInfer CUTLASS内核假设[Up, Gate]投影作为W13
        return self.use_flashinfer_cutlass  # 如果使用flashinfer cutlass则返回True

    def apply(  # 应用MoE方法
        self,
        layer: torch.nn.Module,  # 目标网络层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:
        return self.forward(  # 调用forward方法
            layer=layer,  # 传入层
            dispatch_output=dispatch_output,  # 传入分发输出
        )

    def forward_cuda(  # CUDA平台前向计算
        self,
        layer: torch.nn.Module,  # 目标网络层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入类

        x = dispatch_output.hidden_states  # 获取隐藏状态

        moe_runner_config = self.moe_runner_config  # 获取MoE运行器配置

        backend = self.runner.runner_backend  # 获取运行器后端
        if backend.is_triton_kernels():  # 如果是Triton内核后端
            from sglang.srt.layers.moe.moe_runner.triton_kernels import (  # 导入Triton内核相关类
                TritonKernelsQuantInfo,  # Triton内核量化信息类
            )

            quant_info = TritonKernelsQuantInfo(  # 创建Triton内核量化信息
                w13_weight=layer.w13_weight,  # w13权重
                w2_weight=layer.w2_weight,  # w2权重
                w13_bias=getattr(layer, "w13_weight_bias", None),  # w13偏置
                w2_bias=getattr(layer, "w2_weight_bias", None),  # w2偏置
            )
            return self.runner.run(dispatch_output, quant_info)  # 运行MoE
        elif self.runner.runner_backend.is_deep_gemm():  # 如果是DeepGEMM后端
            w13_weight = layer.w13_weight  # 获取w13权重
            w2_weight = layer.w2_weight  # 获取w2权重
            from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo  # 导入DeepGEMM量化信息类

            # Only use_fp8=False when SGLANG_DEEPEP_BF16_DISPATCH is true,
            # otherwise use_fp8=True for FP8 dispatch path
            # 仅当SGLANG_DEEPEP_BF16_DISPATCH为True时use_fp8=False，
            # 否则use_fp8=True用于FP8分发路径
            use_fp8 = not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()  # 确定是否使用FP8
            quant_info = DeepGemmMoeQuantInfo(  # 创建DeepGEMM量化信息
                w13_weight=w13_weight,  # w13权重
                w2_weight=w2_weight,  # w2权重
                use_fp8=use_fp8,  # 是否使用FP8
            )
            return self.runner.run(dispatch_output, quant_info)  # 运行MoE
        elif self.use_flashinfer_cutlass:  # 如果使用flashinfer cutlass
            topk_output = dispatch_output.topk_output  # 获取topk输出
            output = flashinfer_cutlass_fused_moe(  # 调用flashinfer CUTLASS融合MoE
                input=x,  # 输入隐藏状态
                token_selected_experts=topk_output.topk_ids,  # 每个token选择的专家ID
                token_final_scales=topk_output.topk_weights,  # 每个token的专家权重
                fc1_expert_weights=layer.w13_weight,  # 第一层专家权重
                fc2_expert_weights=layer.w2_weight,  # 第二层专家权重
                output_dtype=x.dtype,  # 输出数据类型
                quant_scales=None,  # 量化缩放因子(无)
                ep_size=layer.moe_ep_size,  # 专家并行大小
                ep_rank=layer.moe_ep_rank,  # 专家并行秩
                tp_size=layer.moe_tp_size,  # 张量并行大小
                tp_rank=layer.moe_tp_rank,  # 张量并行秩
                tune_max_num_tokens=next_power_of_2(x.shape[0]),  # 调优最大token数为2的幂
                activation_type=(  # 激活类型
                    ActivationType.Relu2  # ReLU2激活
                    if moe_runner_config.activation == "relu2"  # 如果配置为relu2
                    else ActivationType.Swiglu  # 否则使用SwiGLU激活
                ),
            )[0]  # 取第一个输出
            return StandardCombineInput(hidden_states=output)  # 返回标准合并输入
        elif self.use_flashinfer_trtllm_moe:  # 如果使用flashinfer TRT-LLM MoE
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (  # 导入flashinfer TRT-LLM量化信息类
                FlashInferTrtllmBf16MoeQuantInfo,  # flashinfer TRT-LLM BF16量化信息类
            )

            quant_info = FlashInferTrtllmBf16MoeQuantInfo(  # 创建量化信息
                gemm1_weights=layer.w13_weight,  # GEMM1权重(w13)
                gemm2_weights=layer.w2_weight,  # GEMM2权重(w2)
                global_num_experts=layer.num_experts,  # 全局专家数量
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,  # 本地专家偏移量
            )
            return self.runner.run(dispatch_output, quant_info)  # 运行MoE
        else:  # 其他情况(默认Triton路径)
            if self._aiter_runner is not None:  # 如果AITER运行器可用
                from sglang.srt.layers.moe.moe_runner.aiter import (  # 导入AITER量化信息类
                    AiterMoeQuantInfo,  # AITER量化信息类
                )

                try:  # 尝试使用AITER运行器
                    quant_info = AiterMoeQuantInfo(  # 创建AITER量化信息
                        w13_weight=layer.w13_weight,  # w13权重
                        w2_weight=layer.w2_weight,  # w2权重
                        expert_mask=layer.dispatcher.expert_mask_gpu,  # 专家掩码
                    )
                    return self._aiter_runner.run(dispatch_output, quant_info)  # 使用AITER运行器
                except RuntimeError as e:  # 捕获运行时错误
                    # AITER CK fused_moe may not support all GEMM dimensions
                    # (e.g. Gemma4 MoE with 128 experts x 704 intermediate size)
                    # AITER CK fused_moe可能不支持所有GEMM维度
                    # (例如128专家x 704中间大小的Gemma4 MoE)
                    logger.warning_once(  # 记录警告(仅一次)
                        f"AITER CK fused_moe failed ({e}), "  # AITER CK融合MoE失败信息
                        "falling back to Triton MoE runner."  # 回退到Triton MoE运行器
                    )

            quant_info = TritonMoeQuantInfo(  # 创建Triton量化信息
                w13_weight=layer.w13_weight,  # w13权重
                w2_weight=layer.w2_weight,  # w2权重
                b13=getattr(layer, "w13_weight_bias", None),  # w13偏置
                b2=getattr(layer, "w2_weight_bias", None),  # w2偏置
            )
            return self.runner.run(dispatch_output, quant_info)  # 使用Triton运行器

    def forward_cpu(  # CPU平台前向计算
        self,
        layer: torch.nn.Module,  # 目标网络层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入类

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取topk输出

        moe_runner_config = self.moe_runner_config  # 获取MoE运行器配置

        assert (  # 断言激活函数为silu
            moe_runner_config.activation == "silu"  # 检查激活函数是否为silu
        ), f"activation = {moe_runner_config.activation} is not supported."  # 不支持则报错

        if use_intel_amx_backend(layer):  # 如果使用Intel AMX后端
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu  # 导入CPU topk权重应用函数

            topk_weights, topk_ids, _ = topk_output  # 解包topk输出
            x, topk_weights = apply_topk_weights_cpu(  # 在CPU上应用topk权重
                moe_runner_config.apply_router_weight_on_input, topk_weights, x  # 传入配置和权重
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(  # 调用CPU融合专家计算
                x,  # 输入
                layer.w13_weight,  # w13权重
                layer.w2_weight,  # w2权重
                topk_weights,  # topk权重
                topk_ids,  # topk专家ID
                False,  # inplace # See [Note] inplace should be False in fused_experts. # 原地操作 # 见[注] fused_experts中inplace应为False
                CPUQuantMethod.UNQUANT,  # CPU量化方法: 未量化
                None,  # w1_scale # w1缩放因子
                None,  # w2_scale # w2缩放因子
                None,  # w1_zp # w1零点
                None,  # w2_zp # w2零点
                None,  # block_size # 块大小
                getattr(layer, "w13_weight_bias", None),  # w13偏置
                getattr(layer, "w2_weight_bias", None),  # w2偏置
                layer.moe_runner_config.gemm1_alpha,  # GEMM1 alpha参数
                layer.moe_runner_config.gemm1_clamp_limit,  # GEMM1钳制限制
                True,  # is_vnni # 使用VNNI格式
            )
            return StandardCombineInput(hidden_states=output)  # 返回标准合并输入
        else:  # 非AMX后端
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native  # 导入原生MoE前向函数

            output = moe_forward_native(  # 调用原生MoE前向计算
                layer,  # 网络层
                x,  # 输入
                topk_output,  # topk输出
                moe_runner_config,  # MoE运行器配置
            )
            return StandardCombineInput(hidden_states=output)  # 返回标准合并输入

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:  # 获取Triton量化信息
        return TritonMoeQuantInfo(  # 返回Triton量化信息对象
            w13_weight=layer.w13_weight,  # w13权重
            w2_weight=layer.w2_weight,  # w2权重
            b13=getattr(layer, "w13_weight_bias", None),  # w13偏置
            b2=getattr(layer, "w2_weight_bias", None),  # w2偏置
        )

    def forward_xpu(  # XPU平台前向计算
        self,
        layer: torch.nn.Module,  # 目标网络层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入类

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取topk输出

        moe_runner_config = self.moe_runner_config  # 获取MoE运行器配置
        assert moe_runner_config.activation in [  # 断言激活函数受支持
            "silu",  # SiLU激活
            "gelu",  # GELU激活
        ], f"activation = {moe_runner_config.activation} is not supported."  # 不支持则报错

        backend = self.runner.runner_backend  # 获取运行器后端
        if use_intel_xpu_backend():  # 如果使用Intel XPU后端
            # sgl-kernel-xpu path
            # sgl-kernel-xpu路径
            from sgl_kernel import fused_experts  # 导入融合专家函数

            topk_weights, topk_ids, _ = topk_output  # 解包topk输出
            if moe_runner_config.apply_router_weight_on_input:  # 如果在输入上应用路由权重
                x = x * topk_weights.to(x.dtype)  # 将路由权重乘到输入上
                topk_weights = torch.ones_like(topk_weights)  # 将topk权重设为全1
            output = fused_experts(  # 调用融合专家计算
                x,  # 输入
                layer.w13_weight,  # w13权重
                layer.w2_weight,  # w2权重
                topk_weights,  # topk权重
                topk_ids,  # topk专家ID
                b1=getattr(layer, "w13_weight_bias", None),  # w13偏置
                b2=getattr(layer, "w2_weight_bias", None),  # w2偏置
                activation=moe_runner_config.activation,  # 激活函数
                gemm1_alpha=moe_runner_config.gemm1_alpha,  # GEMM1 alpha参数
                gemm1_limit=moe_runner_config.gemm1_clamp_limit,  # GEMM1钳制限制
            )
            return StandardCombineInput(hidden_states=output)  # 返回标准合并输入
        else:  # 非XPU后端
            assert backend.is_triton()  # 断言后端为Triton
            assert (  # 断言激活函数为silu
                moe_runner_config.activation == "silu"  # 检查激活函数
            ), f"activation = {moe_runner_config.activation} is not supported \
            for Triton PATH, please set ENV SGLANG_USE_SGL_XPU=1."  # 提示设置环境变量

            quant_info = self.get_triton_quant_info(layer)  # 获取Triton量化信息
            return self.runner.run(dispatch_output, quant_info)  # 使用Triton运行器

    def forward_npu(  # NPU平台前向计算
        self,
        layer: torch.nn.Module,  # 目标网络层
        dispatch_output: "DispatchOutput",  # 分发输出
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入类
        from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker  # 导入分发输出检查器

        if DispatchOutputChecker.format_is_deepep(dispatch_output):  # 如果是DeepEP格式
            return self._forward_npu_deepep(layer, dispatch_output)  # 调用DeepEP专用前向计算

        # x.shape = [B*S, H]
        # x形状 = [批次*序列长度, 隐藏维度]
        x = dispatch_output.hidden_states  # 获取隐藏状态
        # topk_weights.shape = [B*S, K]; topk_ids.shape = [B*S, K]
        # topk权重形状 = [批次*序列长度, K]; topk ID形状 = [批次*序列长度, K]
        topk_weights, topk_ids, _ = dispatch_output.topk_output  # 解包topk输出

        original_dtype = x.dtype  # 保存原始数据类型
        num_tokens = x.shape[0]  # 获取token数量
        topk_weights = topk_weights.to(x.dtype)  # 将topk权重转为输入数据类型
        topk_ids = topk_ids.to(torch.int32)  # 将topk ID转为int32
        num_experts = layer.num_experts  # 获取专家数量
        top_k = layer.top_k or topk_ids.shape[1]  # in case layer.top_k is not set # 获取topk值，以防layer.top_k未设置

        hidden_states, expanded_row_idx, expert_tokens, _ = (  # NPU MoE初始化路由
            torch.ops.npu.npu_moe_init_routing_v2(  # 调用NPU MoE初始化路由v2
                x,  # 输入隐藏状态
                topk_ids,  # topk专家ID
                active_num=num_tokens * top_k,  # 活跃token总数
                expert_num=num_experts,  # 专家数量
                expert_tokens_num_type=1,  # 专家token数量类型
                expert_tokens_num_flag=True,  # 专家token数量标志
                active_expert_range=[0, num_experts],  # 活跃专家范围
                quant_mode=-1,  # 量化模式
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)  # 将专家token数量转为int64
        w13_bias = [layer.w13_weight_bias] if self.with_bias else None  # 设置w13偏置列表
        w2_bias = [layer.w2_weight_bias] if self.with_bias else None  # 设置w2偏置列表

        # gmm1: gate_up_proj
        # gmm1: gate_up投影
        hidden_states = torch.ops.npu.npu_grouped_matmul(  # 调用NPU分组矩阵乘法
            x=[hidden_states],  # 输入列表
            weight=[layer.w13_weight],  # 权重列表
            bias=w13_bias,  # 偏置列表
            split_item=2,  # 分割项
            group_list_type=1,  # 分组列表类型
            group_type=0,  # 分组类型
            group_list=expert_tokens,  # 每组token数量
            output_dtype=original_dtype,  # 输出数据类型
        )[0]  # 取第一个输出

        # act_fn:
        # 激活函数:
        if self.moe_runner_config.activation == "npu_swiglu_oai":  # 如果是NPU SwiGLU OAI激活
            from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai  # 导入swiglu_oai函数

            hidden_states = swiglu_oai(layer, hidden_states)  # 应用swiglu_oai激活
        elif self.moe_runner_config.activation == "silu":  # 如果是SiLU激活
            if self.moe_runner_config.gemm1_clamp_limit is not None:  # 如果设置了GEMM1钳制限制
                hidden_states = swiglustep_and_mul(  # 使用swiglustep_and_mul激活
                    hidden_states, self.moe_runner_config.gemm1_clamp_limit  # 传入隐藏状态和钳制限制
                )
            else:  # 未设置钳制限制
                hidden_states = torch.ops.npu.npu_swiglu(hidden_states)  # 使用NPU原生swiglu
        else:  # 其他激活函数
            from sglang.srt.layers.activation import GeluAndMul  # 导入GELU与乘法激活

            hidden_states = GeluAndMul()(hidden_states)  # 应用GELU与乘法激活

        # gmm2: down_proj
        # gmm2: down投影
        hidden_states = torch.ops.npu.npu_grouped_matmul(  # 调用NPU分组矩阵乘法
            x=[hidden_states],  # 输入列表
            weight=[layer.w2_weight],  # 权重列表
            bias=w2_bias,  # 偏置列表
            split_item=2,  # 分割项
            group_list_type=1,  # 分组列表类型
            group_type=0,  # 分组类型
            group_list=expert_tokens,  # 每组token数量
            output_dtype=original_dtype,  # 输出数据类型
        )[0]  # 取第一个输出

        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(  # NPU MoE最终路由
            hidden_states,  # 隐藏状态
            skip1=None,  # 跳跃连接1
            skip2=None,  # 跳跃连接2
            bias=None,  # 偏置
            scales=topk_weights,  # topk权重缩放
            expanded_src_to_dst_row=expanded_row_idx,  # 扩展行索引
            export_for_source_row=topk_ids,  # 源行导出的专家ID
            drop_pad_mode=2,  # 丢弃填充模式
        )

        return StandardCombineInput(hidden_states=final_hidden_states)  # 返回标准合并输入

    def _forward_npu_deepep(  # NPU平台DeepEP前向计算
        self,
        layer: torch.nn.Module,  # 目标网络层
        dispatch_output: "DispatchOutput",  # 分发输出
    ) -> CombineInput:
        from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (  # 导入NPU融合MoE方法
            npu_fused_moe_without_routing_weights_bf16,  # 无路由权重的NPU BF16融合MoE
        )
        from sglang.srt.layers.moe.token_dispatcher import (  # 导入token分发器相关类
            DeepEPLLCombineInput,  # DeepEP低延迟合并输入
            DeepEPNormalCombineInput,  # DeepEP普通合并输入
        )
        from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker  # 导入分发输出检查器

        # NOTE: Ascend's Dispatch & Combine does not support FP16
        # 注意: Ascend的Dispatch & Combine不支持FP16
        output_dtype = torch.bfloat16  # 输出数据类型为bfloat16
        group_list_type = 1  # 分组列表类型

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):  # 如果是DeepEP普通格式
            hidden_states, _, _, _, num_recv_tokens_per_expert = dispatch_output  # 解包分发输出
            group_list = torch.tensor(  # 创建分组列表张量
                num_recv_tokens_per_expert,  # 每个专家接收的token数
                dtype=torch.int64,  # 数据类型
                device=hidden_states.device,  # 设备
            )
            combine_cls = DeepEPNormalCombineInput  # 使用普通合并输入类
        else:  # DeepEP低延迟格式
            hidden_states, _, _, _, group_list, _ = dispatch_output  # 解包分发输出
            group_list = group_list.to(torch.int64)  # 转为int64
            combine_cls = DeepEPLLCombineInput  # 使用低延迟合并输入类

        hidden_states = npu_fused_moe_without_routing_weights_bf16(  # 调用NPU BF16融合MoE
            layer, hidden_states, group_list_type, group_list, output_dtype  # 传入参数
        )
        return combine_cls(  # 返回合并输入对象
            hidden_states=hidden_states,  # 隐藏状态
            topk_ids=dispatch_output.topk_ids,  # topk专家ID
            topk_weights=dispatch_output.topk_weights,  # topk权重
        )

    def forward_tpu(self, *args, **kwargs) -> CombineInput:  # TPU平台前向计算(未实现)
        raise NotImplementedError("The TPU backend currently does not support MoE.")  # TPU后端当前不支持MoE

    def forward_musa(self, *args, **kwargs) -> CombineInput:  # MUSA平台前向计算
        return self.forward_cuda(*args, **kwargs)  # 委托给CUDA前向计算

    forward_native = forward_cpu  # 原生前向计算等同于CPU前向计算
