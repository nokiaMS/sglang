# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识
# 文件说明：KT（KTransformers）专家并行包装器，为MoE层提供CPU-GPU异构专家并行能力。
# 协调GPU专家（使用任意量化方法）和CPU专家（使用AMX/AVX指令）的并行执行。
# 实现了提交-计算-同步模式：先异步提交CPU计算，再并行执行GPU计算，最后同步合并结果。
"""
KT Expert Parallelism Wrapper for MoE layers.  # KT专家并行包装器，用于MoE层

This module provides a generic wrapper that enables CPU-GPU expert parallelism  # 本模块提供通用包装器，启用CPU-GPU专家并行
for any MoE quantization method. It coordinates parallel execution of GPU experts  # 支持任意MoE量化方法，协调GPU专家的并行执行
(using any quantization method) and CPU experts (using AMX/AVX instructions).  # （使用任意量化方法）和CPU专家（使用AMX/AVX指令）
"""

from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Optional  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.distributed import get_tensor_model_parallel_rank  # 导入张量模型并行排名获取函数
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase  # 导入融合MoE方法基类
from sglang.srt.utils import get_compiler_backend  # 导入编译器后端获取函数

if TYPE_CHECKING:  # 仅用于类型检查时导入
    from sglang.srt.layers.moe import MoeRunnerConfig  # MoE运行器配置
    from sglang.srt.layers.moe.token_dispatcher import (  # 令牌调度器相关类型
        CombineInput,  # 合并输入
        StandardDispatchOutput,  # 标准调度输出
    )
    from sglang.srt.server_args import ServerArgs  # 服务器参数

try:  # 尝试导入KT内核
    from kt_kernel import KTMoEWrapper  # 导入KT MoE包装器

    KTRANSFORMERS_AVAILABLE = True  # KT可用
except ImportError:  # 导入失败
    KTRANSFORMERS_AVAILABLE = False  # KT不可用


@dataclass  # 数据类装饰器
class KTConfig:  # KT配置数据类
    """Configuration for KTransformers heterogeneous computing CPU part.  # KTransformers异构计算CPU部分配置

    Args:  # 参数：
        layer_idx: Layer index in the model  # 层索引：模型中的层索引
        num_gpu_experts: Number of experts to run on GPU  # GPU专家数：在GPU上运行的专家数量
        cpuinfer_threads: Number of CPU inference threads  # CPU推理线程数：CPU推理线程数量
        threadpool_count: Number of thread pools for CPU computation  # 线程池数：CPU计算的线程池数量
        weight_path: Path to CPU quantized weights  # 权重路径：CPU量化权重的路径
        chunked_prefill_size: Chunk size for prefill computation  # 分块预填充大小：预填充计算的分块大小
        method: CPU computation method (e.g., "int4")  # 方法：CPU计算方法（如"int4"）
        num_layers: Total number of layers in the model (optional)  # 层数：模型总层数（可选）
    """

    layer_idx: int  # 层索引
    num_gpu_experts: int  # GPU专家数量
    cpuinfer_threads: int  # CPU推理线程数
    threadpool_count: int  # 线程池数量
    weight_path: str  # CPU量化权重路径
    chunked_prefill_size: int  # 分块预填充大小
    max_deferred_experts_per_token: int  # 每个token最大延迟专家数
    method: str  # CPU计算方法
    num_layers: Optional[int] = None  # 模型总层数，可选


def create_kt_config_from_server_args(  # 从服务器参数创建KT配置
    server_args: "ServerArgs", layer_idx: int  # 服务器参数和层索引
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.  # 如果配置了KT，从ServerArgs创建KTConfig

    Args:  # 参数：
        server_args: Global server arguments  # server_args：全局服务器参数
        layer_idx: Layer index in the model  # layer_idx：模型中的层索引

    Returns:  # 返回：
        KTConfig if KT is configured, None otherwise  # 如果配置了KT返回KTConfig，否则返回None
    """
    if server_args.kt_weight_path is None:  # 如果未配置KT权重路径
        return None  # 返回None

    # Try to get num_layers from model config  # 尝试从模型配置获取层数
    num_layers = None  # 初始化层数为None
    try:  # 尝试获取配置
        hf_config = server_args.get_hf_config()  # 获取HuggingFace配置
        num_layers = getattr(hf_config, "num_hidden_layers", None)  # 获取隐藏层数量
    except Exception:  # 获取失败
        # If we can't get the config, num_layers will be None  # 如果无法获取配置，num_layers将为None
        pass  # 跳过

    return KTConfig(  # 返回KT配置实例
        layer_idx=layer_idx,  # 层索引
        num_gpu_experts=server_args.kt_num_gpu_experts,  # GPU专家数
        cpuinfer_threads=server_args.kt_cpuinfer,  # CPU推理线程数
        threadpool_count=server_args.kt_threadpool_count,  # 线程池数
        weight_path=server_args.kt_weight_path,  # 权重路径
        chunked_prefill_size=server_args.chunked_prefill_size,  # 分块预填充大小
        method=server_args.kt_method,  # 计算方法
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,  # 最大延迟专家数
        num_layers=num_layers,  # 模型层数
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())  # 使用torch.compile动态编译
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:  # 掩码CPU专家ID，将其设为-1
    """Mask CPU expert IDs by setting them to -1.  # 通过将CPU专家ID设为-1来掩码

    This function masks expert IDs that should be computed on CPU (IDs >= num_gpu_experts)  # 此函数掩码应在CPU上计算的专家ID（ID >= num_gpu_experts）
    so they won't be computed on GPU. The masked IDs are set to -1, which causes the  # 使其不会在GPU上计算。掩码后的ID设为-1，这会导致
    GPU MoE kernel to skip those experts.  # GPU MoE内核跳过这些专家

    Args:  # 参数：
        topk_ids: Tensor of shape [num_tokens, top_k] containing expert IDs  # topk_ids：形状为[num_tokens, top_k]的专家ID张量
        num_gpu_experts: Number of experts that should run on GPU (experts 0 to num_gpu_experts-1)  # num_gpu_experts：应在GPU上运行的专家数（专家0到num_gpu_experts-1）

    Returns:  # 返回：
        Modified topk_ids tensor with CPU expert IDs masked as -1  # CPU专家ID被掩码为-1的修改后topk_ids张量
    """
    topk_ids[topk_ids >= num_gpu_experts] = -1  # 将大于等于GPU专家数的ID设为-1
    return topk_ids  # 返回修改后的topk_ids


class KTEPWrapperMethod(FusedMoEMethodBase):  # KT专家并行包装方法，继承自融合MoE方法基类
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.  # 任意MoE量化方法的包装器，启用CPU-GPU专家并行

    This wrapper coordinates parallel execution of:  # 此包装器协调以下并行执行：
    - GPU experts (0 to num_gpu_experts-1) using any quantization method  # - GPU专家（0到num_gpu_experts-1），使用任意量化方法
    - CPU experts (num_gpu_experts to total_experts-1) using AMX/AVX instructions  # - CPU专家（num_gpu_experts到total_experts-1），使用AMX/AVX指令

    The wrapper implements the submit-compute-sync pattern:  # 包装器实现提交-计算-同步模式：
    1. Submit CPU expert computation (non-blocking)  # 1. 提交CPU专家计算（非阻塞）
    2. Execute GPU expert computation in parallel  # 2. 并行执行GPU专家计算
    3. Synchronize and merge CPU+GPU results  # 3. 同步并合并CPU+GPU结果

    Example:  # 示例：
        # Wrap any GPU method with AMX/AVX CPU expert support  # 用AMX/AVX CPU专家支持包装任意GPU方法
        gpu_method = CompressedTensorsWNA16MoE(quant_config, prefix)  # GPU量化方法
        kt_config = KTConfig(layer_idx=0, num_gpu_experts=4, ...)  # KT配置
        method = KTEPWrapperMethod(gpu_method, kt_config)  # 包装方法
    """

    def __init__(  # 初始化KT EP包装器
        self,
        gpu_method: FusedMoEMethodBase,  # GPU量化方法
        kt_config: KTConfig,  # KT配置
    ):
        """Initialize the KT EP wrapper.  # 初始化KT EP包装器

        Args:  # 参数：
            gpu_method: The quantization method to use for GPU experts  # gpu_method：GPU专家使用的量化方法
            kt_config: Configuration for KT CPU expert computation  # kt_config：KT CPU专家计算配置
        """
        if not KTRANSFORMERS_AVAILABLE:  # 如果KT不可用
            raise ImportError(  # 抛出导入错误
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."  # kt_kernel未安装，请安装后再使用
            )

        self.gpu_method = gpu_method  # 保存GPU方法
        self.kt_config = kt_config  # 保存KT配置
        self.num_gpu_experts = kt_config.num_gpu_experts  # 保存GPU专家数
        self.override_num_local_experts = True  # 标记需要覆盖本地专家数
        self.gpu_method.num_gpu_experts = self.num_gpu_experts  # 设置GPU方法的专家数
        self.tp_rank = get_tensor_model_parallel_rank()  # 获取张量并行排名

        # KT wrapper will be initialized in create_weights  # KT包装器将在create_weights中初始化
        self.wrapper: Optional[KTMoEWrapper] = None  # KT MoE包装器实例

        # Store parameters needed for KT initialization  # 存储KT初始化所需的参数
        self._layer_params = None  # 层参数，暂存

    def create_weights(  # 创建GPU和CPU专家的权重
        self,
        layer: torch.nn.Module,  # MoE层模块
        num_experts: int,  # 专家总数（GPU+CPU）
        hidden_size: int,  # 隐藏维度大小
        intermediate_size_per_partition: int,  # 每个TP分区的中间维度大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        """Create weights for both GPU and CPU experts.  # 为GPU和CPU专家创建权重

        Args:  # 参数：
            layer: The MoE layer module  # layer：MoE层模块
            num_experts: Total number of experts (GPU + CPU)  # num_experts：专家总数（GPU+CPU）
            hidden_size: Hidden dimension size  # hidden_size：隐藏维度大小
            intermediate_size_per_partition: Intermediate size per TP partition  # intermediate_size_per_partition：每个TP分区的中间大小
            params_dtype: Data type for parameters  # params_dtype：参数数据类型
            **extra_weight_attrs: Additional weight attributes  # **extra_weight_attrs：额外权重属性
        """
        self.global_num_experts = num_experts  # 保存全局专家数
        self.hidden_size = hidden_size  # 保存隐藏维度
        self.intermediate_size_per_partition = intermediate_size_per_partition  # 保存中间维度

        # Get required parameters from layer object  # 从层对象获取所需参数
        # top_k: number of experts selected per token  # top_k：每个token选择的专家数
        num_experts_per_tok = layer.top_k  # 获取每token专家数

        # intermediate_size_full: full intermediate size before TP partitioning  # intermediate_size_full：TP分区前的完整中间维度
        intermediate_size_full = (  # 计算完整中间维度
            layer.intermediate_size_per_partition * layer.moe_tp_size  # 分区大小乘以TP大小
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0  # 获取本层最大延迟专家数
        if (  # 如果满足以下条件
            self.kt_config.max_deferred_experts_per_token is not None  # 配置了最大延迟专家数
            and self.kt_config.num_layers is not None  # 配置了总层数
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1  # 且是最后一层
        ):
            layer_max_deferred = 0  # 最后一层不使用延迟专家

        # 1. Create weights for GPU experts using the wrapped method  # 1. 使用包装方法为GPU专家创建权重
        # GPU experts: 0 to num_gpu_experts-1  # GPU专家：0到num_gpu_experts-1
        self.gpu_method.create_weights(  # 调用GPU方法的权重创建
            layer=layer,  # MoE层
            num_experts=self.num_gpu_experts,  # GPU专家数
            hidden_size=hidden_size,  # 隐藏维度
            intermediate_size_per_partition=intermediate_size_per_partition,  # 中间维度
            params_dtype=params_dtype,  # 参数类型
            **extra_weight_attrs,  # 额外属性
        )

        # 2. Initialize KT wrapper for CPU experts  # 2. 为CPU专家初始化KT包装器
        # CPU experts: num_gpu_experts to num_experts-1  # CPU专家：num_gpu_experts到num_experts-1
        if self.tp_rank == 0:  # 仅在主TP排名上初始化
            self.wrapper = KTMoEWrapper(  # 创建KT MoE包装器实例
                layer_idx=self.kt_config.layer_idx,  # 层索引
                num_experts=num_experts,  # 专家总数
                num_experts_per_tok=num_experts_per_tok,  # 每token专家数
                hidden_size=hidden_size,  # 隐藏维度
                moe_intermediate_size=intermediate_size_full,  # MoE中间维度
                num_gpu_experts=self.num_gpu_experts,  # GPU专家数
                cpuinfer_threads=self.kt_config.cpuinfer_threads,  # CPU推理线程数
                threadpool_count=self.kt_config.threadpool_count,  # 线程池数
                weight_path=self.kt_config.weight_path,  # 权重路径
                chunked_prefill_size=self.kt_config.chunked_prefill_size,  # 分块预填充大小
                method=self.kt_config.method,  # 计算方法
                max_deferred_experts_per_token=layer_max_deferred,  # 最大延迟专家数
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载检查点后处理权重
        """Process weights after loading from checkpoint.  # 从检查点加载后处理权重

        Args:  # 参数：
            layer: The MoE layer module  # layer：MoE层模块
        """
        # 1. Process GPU weights  # 1. 处理GPU权重
        if hasattr(self.gpu_method, "process_weights_after_loading"):  # 如果GPU方法有后处理函数
            self.gpu_method.process_weights_after_loading(layer)  # 调用GPU方法的后处理

        # 2. Load CPU weights using KT wrapper  # 2. 使用KT包装器加载CPU权重
        if self.tp_rank == 0 and self.wrapper is not None:  # 仅在主TP排名且有包装器时
            torch.cuda.synchronize()  # 同步CUDA操作

            # Get expert location metadata for CPU expert mapping  # 获取专家位置元数据用于CPU专家映射
            from sglang.srt.eplb.expert_location_dispatch import (  # 导入专家位置调度模块
                get_global_expert_location_metadata,  # 获取全局专家位置元数据
            )

            physical_to_logical_map_cpu = (  # 获取CPU物理到逻辑映射
                get_global_expert_location_metadata()  # 获取全局元数据
                .physical_to_logical_map_cpu[self.kt_config.layer_idx]  # 获取当前层的映射
                .contiguous()  # 确保内存连续
            )
            self.wrapper.load_weights(physical_to_logical_map_cpu)  # 加载CPU权重

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"  # MoE层和运行器配置
    ):
        """Create MoE runner for computation.  # 创建用于计算的MoE运行器

        Args:  # 参数：
            layer: The MoE layer module  # layer：MoE层模块
            moe_runner_config: Configuration for MoE runner  # moe_runner_config：MoE运行器配置
        """
        self.moe_runner_config = moe_runner_config  # 保存运行器配置
        if self.override_num_local_experts:  # 如果需要覆盖本地专家数
            moe_runner_config.num_local_experts = self.num_gpu_experts  # 设置为GPU专家数
        # Delegate to GPU method to create its runner  # 委托给GPU方法创建其运行器
        self.gpu_method.create_moe_runner(layer, moe_runner_config)  # 调用GPU方法的运行器创建

    def submit(  # 异步提交CPU专家计算（非阻塞）
        self,
        layer: torch.nn.Module,  # MoE层模块
        dispatch_output: "StandardDispatchOutput",  # 调度后的token和路由信息
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).  # 异步提交CPU专家计算（非阻塞）

        This method submits the CPU expert computation to AMX/AVX without waiting  # 此方法将CPU专家计算提交到AMX/AVX，不等待完成
        for completion, allowing GPU computation to proceed in parallel.  # 允许GPU计算并行进行

        Args:  # 参数：
            layer: The MoE layer module  # layer：MoE层模块
            dispatch_output: Dispatched tokens and routing information  # dispatch_output：调度后的token和路由信息
        """
        assert (  # 断言仅支持SiLU激活
            self.moe_runner_config.activation == "silu"  # 激活函数为SiLU
        ), "Only SiLU activation is supported."  # 仅支持SiLU激活

        if self.tp_rank != 0 or self.wrapper is None:  # 非主TP排名或无包装器
            return  # 直接返回

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取TopK输出
        topk_weights, topk_ids, _ = topk_output  # 解包TopK权重、ID和logits

        # Submit forward task to CPU (non-blocking)  # 提交前向任务到CPU（非阻塞）
        self.wrapper.submit_forward(  # 调用包装器的提交前向
            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream  # 传入数据和CUDA流
        )

    def sync(self, x: torch.Tensor) -> torch.Tensor:  # 同步并获取CPU专家计算结果
        """Synchronize and retrieve CPU expert computation results.  # 同步并获取CPU专家计算结果

        This method waits for the CPU computation to complete and returns the results.  # 此方法等待CPU计算完成并返回结果

        Args:  # 参数：
            x: Reference tensor for shape and device information  # x：用于形状和设备信息的参考张量

        Returns:  # 返回：
            CPU expert computation results  # CPU专家计算结果
        """
        if self.tp_rank != 0 or self.wrapper is None:  # 非主TP排名或无包装器
            return torch.zeros_like(x)  # 返回零张量

        # Wait for CPU computation and retrieve results  # 等待CPU计算并获取结果
        return self.wrapper.sync_forward(  # 调用包装器的同步前向
            x, torch.cuda.current_stream(x.device).cuda_stream  # 传入参考张量和CUDA流
        )

    def apply(  # 执行混合CPU+GPU MoE前向传播
        self,
        layer: torch.nn.Module,  # MoE层模块
        dispatch_output: "StandardDispatchOutput",  # 调度后的token和路由信息
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.  # 执行混合CPU+GPU MoE前向传播，带并行性

        This is the main computation method that coordinates:  # 这是协调以下操作的主计算方法：
        1. Submit CPU expert computation (non-blocking)  # 1. 提交CPU专家计算（非阻塞）
        2. Execute GPU expert computation in parallel  # 2. 并行执行GPU专家计算
        3. Synchronize CPU results and merge with GPU results  # 3. 同步CPU结果并与GPU结果合并

        Args:  # 参数：
            layer: The MoE layer module  # layer：MoE层模块
            dispatch_output: Dispatched tokens and routing information  # dispatch_output：调度后的token和路由信息

        Returns:  # 返回：
            Combined computation results from CPU and GPU experts  # CPU和GPU专家的组合计算结果
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取TopK输出

        # Step 1: Submit CPU expert computation (non-blocking)  # 步骤1：提交CPU专家计算（非阻塞）
        if self.tp_rank == 0:  # 仅在主TP排名上
            self.submit(layer, dispatch_output)  # 提交CPU计算

        # Step 2: Prepare GPU computation by masking CPU expert IDs  # 步骤2：通过掩码CPU专家ID准备GPU计算
        # CPU expert IDs (>= num_gpu_experts) are set to -1 so GPU kernel skips them  # CPU专家ID（>= num_gpu_experts）被设为-1，使GPU内核跳过它们
        topk_ids = topk_output.topk_ids  # 获取TopK ID
        masked_topk_ids = mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)  # 掩码CPU专家ID

        # Create modified dispatch output for GPU computation  # 为GPU计算创建修改后的调度输出
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)  # 替换TopK ID
        masked_dispatch_output = dispatch_output._replace(  # 替换TopK输出
            topk_output=masked_topk_output  # 使用掩码后的TopK输出
        )

        # Step 3: Execute GPU expert computation (any quantization method)  # 步骤3：执行GPU专家计算（任意量化方法）
        # This runs in parallel with CPU computation  # 此操作与CPU计算并行运行
        gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)  # 调用GPU方法的前向

        # Step 4: Synchronize CPU results and merge with GPU results  # 步骤4：同步CPU结果并与GPU结果合并
        output = gpu_combine_input.hidden_states  # 获取GPU输出
        if self.tp_rank == 0:  # 仅在主TP排名上
            cpu_output = self.sync(x)  # 同步获取CPU输出
            output = output + cpu_output  # 合并GPU和CPU输出

        return StandardCombineInput(hidden_states=output)  # 返回标准合并输入

    def __getattr__(self, name: str):  # 委托属性访问到包装的GPU方法
        """Delegate attribute access to the wrapped GPU method.  # 委托属性访问到包装的GPU方法

        This allows the wrapper to transparently expose attributes and methods  # 这允许包装器透明地暴露属性和方法
        from the wrapped GPU quantization method.  # 来自包装的GPU量化方法

        Args:  # 参数：
            name: Attribute name  # name：属性名

        Returns:  # 返回：
            Attribute value from gpu_method  # 来自gpu_method的属性值
        """
        # Avoid infinite recursion for internal attributes  # 避免内部属性的无限递归
        if name in ("gpu_method", "wrapper", "kt_config"):  # 如果是内部属性名
            raise AttributeError(  # 抛出属性错误
                f"'{type(self).__name__}' object has no attribute '{name}'"  # 对象没有该属性
            )

        return getattr(self.gpu_method, name)  # 从GPU方法获取属性
