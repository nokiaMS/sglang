# Mori EP令牌分发器模块
# 本模块实现了基于Mori框架（AMD ROCm）的专家并行(EP)令牌分发与合并逻辑，
# 支持普通模式(Normal)和低延迟模式(LowLatency/AsyncLL)，支持FP8/FP4量化分发，
# 支持双流(dual stream)通信与计算重叠，以及SDMA（Scalable DMA）加速。

from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 日志模块
import os  # 操作系统接口
from dataclasses import dataclass  # 数据类装饰器
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple  # 类型注解

from sglang.srt.layers.dp_attention import get_is_extend_in_batch  # 获取当前批次是否为扩展阶段
from sglang.srt.layers.moe.token_dispatcher.base import (  # 分发器基类及相关类型
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPPDispatchHooks  # DeepEP分发钩子
from sglang.srt.layers.moe.topk import TopKOutput  # TopK选择输出类型
from sglang.srt.layers.moe.utils import (  # MoE工具函数
    DeepEPMode,
    is_tbo_enabled,
)
from sglang.srt.utils import (  # 通用工具函数
    get_bool_env_var,
    get_int_env_var,
    is_hip,
)

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.single_batch_overlap import CombineOverlapArgs  # 合并重叠参数
    import mori  # Mori框架

from enum import Enum, auto  # 枚举类型和自动值生成
from functools import lru_cache  # 最近最少使用缓存装饰器

import torch  # PyTorch深度学习框架

from sglang.srt.distributed import (  # 分布式通信相关
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
)
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype  # FP8数据类型

# Blockwise quantization group sizes: number of elements sharing one scale factor
# 块量化组大小：共享一个缩放因子的元素数量 # 块量化组大小：共享一个缩放因子的元素数量
FP8_BLOCK_SIZE = 128  # FP8块大小为128
MXFP4_BLOCK_SIZE = 32  # MXFP4块大小为32

_is_hip = is_hip()  # 是否为AMD HIP平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITer加速库（仅HIP平台）

if _use_aiter:  # 如果使用AITer
    from aiter import QuantType, get_hip_quant  # 导入AITer量化和量化函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class MoriEPPDispatchHooks(DeepEPPDispatchHooks):  # Mori EP分发钩子，继承自DeepEPPDispatchHooks

    def __call__(self, dispatcher: BaseDispatcher):  # 调用所有注册的钩子函数
        for hook_fun in self.hook_dict.values():  # 遍历所有钩子函数
            hook_fun(dispatcher)  # 对分发器执行钩子函数


class MoriEPNormalDispatchOutput(NamedTuple):  # Mori EP普通模式分发输出的具名元组
    """Mori EP normal dispatch output."""  # Mori EP普通模式分发输出。 # Mori EP普通模式分发输出。

    hidden_states: torch.Tensor  # 隐藏状态张量
    hidden_states_scale: Optional[torch.Tensor]  # 隐藏状态缩放因子（可选）
    topk_ids: torch.Tensor  # TopK选择的专家ID
    topk_weights: torch.Tensor  # TopK选择的专家权重
    num_recv_tokens_per_expert: List[int]  # 每个专家接收的令牌数列表
    origin_topk_ids: torch.Tensor  # 原始TopK专家ID
    origin_topk_weights: torch.Tensor  # 原始TopK专家权重
    out_dtype: torch.dtype  # 输出数据类型

    @property
    def format(self) -> DispatchOutputFormat:  # 返回分发输出格式
        return DispatchOutputFormat.DEEPEP_NORMAL  # 返回DeepEP普通模式格式


class MoriEPLLDispatchOutput(NamedTuple):  # Mori EP低延迟模式分发输出的具名元组
    """Mori EP low latency dispatch output."""  # Mori EP低延迟分发输出。 # Mori EP低延迟分发输出。

    hidden_states: torch.Tensor  # 隐藏状态张量
    hidden_states_scale: Optional[torch.Tensor]  # 隐藏状态缩放因子（可选）
    topk_ids: torch.Tensor  # TopK选择的专家ID
    topk_weights: torch.Tensor  # TopK选择的专家权重
    num_recv_tokens_per_expert: List[int]  # 每个专家接收的令牌数列表
    origin_topk_ids: torch.Tensor  # 原始TopK专家ID
    origin_topk_weights: torch.Tensor  # 原始TopK专家权重
    out_dtype: torch.dtype  # 输出数据类型

    @property
    def format(self) -> DispatchOutputFormat:  # 返回分发输出格式
        return DispatchOutputFormat.DEEPEP_LL  # 返回DeepEP低延迟格式


assert isinstance(MoriEPNormalDispatchOutput, DispatchOutput)  # 验证MoriEPNormalDispatchOutput是DispatchOutput的子类
assert isinstance(MoriEPLLDispatchOutput, DispatchOutput)  # 验证MoriEPLLDispatchOutput是DispatchOutput的子类


class MoriEPNormalCombineInput(NamedTuple):  # Mori EP普通模式合并输入的具名元组
    """Mori EP combine input."""  # Mori EP合并输入。 # Mori EP合并输入。

    hidden_states: torch.Tensor  # 隐藏状态张量
    topk_ids: torch.Tensor  # TopK专家ID
    topk_weights: torch.Tensor  # TopK专家权重

    @property
    def format(self) -> CombineInputFormat:  # 返回合并输入格式
        return CombineInputFormat.DEEPEP_NORMAL  # 返回DeepEP普通模式格式


class MoriEPLLCombineInput(NamedTuple):  # Mori EP低延迟模式合并输入的具名元组
    """Mori EP combine input."""  # Mori EP合并输入。 # Mori EP合并输入。

    hidden_states: torch.Tensor  # 隐藏状态张量
    topk_ids: torch.Tensor  # TopK专家ID
    topk_weights: torch.Tensor  # TopK专家权重

    @property
    def format(self) -> CombineInputFormat:  # 返回合并输入格式
        return CombineInputFormat.DEEPEP_LL  # 返回DeepEP低延迟格式


assert isinstance(MoriEPNormalCombineInput, CombineInput)  # 验证MoriEPNormalCombineInput是CombineInput的子类
assert isinstance(MoriEPLLCombineInput, CombineInput)  # 验证MoriEPLLCombineInput是CombineInput的子类


class EpMode(Enum):  # EP通信模式枚举
    INTRA_NODE = "intra_node"  # 节点内通信
    INTER_NODE = "inter_node"  # 节点间通信
    LOW_LATENCY = "low_latency"  # 低延迟模式


class DispatchDtype(Enum):  # 分发数据类型枚举
    bf16 = "bfloat16"  # BF16精度
    fp8 = "float8_blockwise"  # FP8块量化
    fp4 = "mxfp4_blockwise"  # MXFP4块量化


class CombineDtype(Enum):  # 合并数据类型枚举
    bf16 = "bfloat16"  # BF16精度
    fp8 = "float8_blockwise"  # FP8块量化
    fp8_direct_cast = "float8_direct_cast"  # FP8直接转换


@dataclass(frozen=True)  # 不可变数据类
class EpDispatchConfig:  # EP分发配置数据类
    kernel_type: mori.ops.EpDispatchCombineKernelType  # 内核类型
    warp_num_per_block: int  # 每个块的warp数
    block_num: int  # 块数
    rdma_block_num: int  # RDMA块数


def get_ep_dispatch_configs(num_max_dispatch_tokens_per_rank: int = 4096):  # 获取EP分发配置，根据最大分发令牌数选择内核配置
    import mori

    # Selects the inter-node kernel. `InterNodeV1LL` is used if `num_max_dispatch_tokens_per_rank`
    # is less than or equal to the threshold, otherwise `InterNodeV1` is used. The threshold defaults to 256.
    # 选择节点间内核。如果`num_max_dispatch_tokens_per_rank`小于等于阈值则使用`InterNodeV1LL`，否则使用`InterNodeV1`。阈值默认为256。 # 选择节点间内核。如果最大分发令牌数小于等于阈值则使用InterNodeV1LL，否则使用InterNodeV1。阈值默认为256。
    inter_kernel_switch_threshold = get_int_env_var(
        "SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD", 256
    )

    inter_kernel_type = (  # 根据阈值选择节点间内核类型
        mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
        if num_max_dispatch_tokens_per_rank <= inter_kernel_switch_threshold
        else mori.ops.EpDispatchCombineKernelType.InterNodeV1
    )

    return {
        # TODO(billishyahao): need to tune different configs for intra node async
        # Also could be tuned for different AMD platform
        # TODO(billishyahao): 需要为节点内异步调优不同配置，也可以针对不同AMD平台调优 # 需要为节点内异步调优不同配置，也可以针对不同AMD平台调优
        EpMode.INTRA_NODE: EpDispatchConfig(  # 节点内配置
            kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
            warp_num_per_block=16,
            block_num=80,
            rdma_block_num=0,
        ),
        EpMode.INTER_NODE: EpDispatchConfig(  # 节点间配置
            kernel_type=inter_kernel_type,
            warp_num_per_block=8,
            block_num=64,
            rdma_block_num=32,
        ),
        EpMode.LOW_LATENCY: EpDispatchConfig(  # 低延迟配置
            kernel_type=mori.ops.EpDispatchCombineKernelType.AsyncLL,
            warp_num_per_block=8,
            block_num=64,
            rdma_block_num=32,
        ),
    }


# init_mori_op only needs do once in model initial stage
# use lru_cache to reuse the same mori_op instance to avoid the init overhead for mori
# init_mori_op只需在模型初始化阶段执行一次 # 使用lru_cache复用同一mori_op实例以避免mori的初始化开销
@lru_cache(maxsize=4)  # 使用LRU缓存，最多缓存4个实例
def init_mori_op(  # 初始化Mori EP操作实例
    group,  # 进程组
    router_topk,  # 路由器TopK值
    num_experts,  # 专家总数
    num_local_experts,  # 本地专家数
    hidden_size,  # 隐藏层大小
    params_dtype,  # 参数数据类型
    num_max_dispatch_tokens_per_rank,  # 每个rank最大分发令牌数
    deepep_mode,  # DeepEP模式
    instance_id=0,  # 实例ID，默认0
    dispatch_dtype=DispatchDtype.bf16,  # 分发数据类型，默认BF16
    combine_dtype=CombineDtype.bf16,  # 合并数据类型，默认BF16
    enable_sdma=False,  # 是否启用SDMA，默认False
):

    import mori

    world_size = get_moe_expert_parallel_world_size()  # 获取专家并行的世界大小
    rank = get_moe_expert_parallel_rank()  # 获取当前专家并行的rank

    gpu_per_node = 8 if world_size >= 8 else world_size  # 每个节点的GPU数，最多8

    group_name = f"mori"  # 进程组名称
    cpu_group = group.cpu_group  # 获取CPU进程组
    try:
        torch._C._distributed_c10d._register_process_group(group_name, cpu_group)  # 注册进程组
    except Exception as e:
        if "already registered" in str(e):  # 如果进程组已注册
            logger.info(
                f"[MORI init] The same process group is already "
                f"registered. Ignoring [{str(e)}]"
            )  # 同一进程组已注册，忽略
        else:
            raise  # 其他异常则抛出
    else:
        # If new group is newly registered then need to init mori shmem. However
        # if the group is registered already then need to skip init mori shmem
        # and reuse the previous one.
        # 如果新组是新注册的，则需要初始化mori共享内存。然而如果组已注册则需要跳过初始化并复用之前的。 # 如果新组是新注册的则需要初始化mori共享内存，如果已注册则跳过并复用
        mori.shmem.shmem_torch_process_group_init(group_name)

    mode = EpMode.INTRA_NODE if world_size <= 8 else EpMode.INTER_NODE  # 根据世界大小选择通信模式
    async_mode = deepep_mode.enable_low_latency() or enable_sdma  # 判断是否为异步模式
    if async_mode:  # 如果是异步模式
        mode = EpMode.LOW_LATENCY  # 使用低延迟模式

    cfg = get_ep_dispatch_configs(num_max_dispatch_tokens_per_rank)[mode]  # 获取对应模式的分发配置

    kernel_type = cfg.kernel_type  # 内核类型
    warp_num_per_block = cfg.warp_num_per_block  # 每个块的warp数
    block_num = cfg.block_num  # 块数
    rdma_block_num = cfg.rdma_block_num  # RDMA块数

    hidden_dim = hidden_size  # 隐藏维度
    scale_dim = 1  # 缩放维度默认为1
    data_type = fp8_dtype  # 数据类型默认FP8
    scale_type_size = torch.float32.itemsize  # 缩放类型大小为float32

    if dispatch_dtype == DispatchDtype.fp8:  # 如果分发数据类型为FP8
        scale_dim = hidden_size // FP8_BLOCK_SIZE  # 缩放维度为隐藏大小除以FP8块大小
    elif dispatch_dtype == DispatchDtype.fp4:  # 如果分发数据类型为FP4
        # FP4 kernel still takes the original hidden size and do quantization
        # internally, so hidden_dim is not reduced. The reason is that for FP4
        # quantization, we need to keep the original hidden size to calculate
        # the quantization scale correctly. Don't use packed hidden size for FP4 kernel.
        # FP4内核仍使用原始隐藏大小并在内部进行量化，因此hidden_dim不缩减。原因是FP4量化需要保持原始隐藏大小以正确计算量化缩放。不要对FP4内核使用打包的隐藏大小。 # FP4内核仍使用原始隐藏大小并在内部量化，因此hidden_dim不缩减，以正确计算量化缩放
        hidden_dim = hidden_size
        scale_dim = hidden_size // MXFP4_BLOCK_SIZE  # 缩放维度为隐藏大小除以MXFP4块大小
        data_type = torch.float4_e2m1fn_x2  # 数据类型为FP4
        scale_type_size = torch.float8_e8m0fnu.itemsize  # 缩放类型大小为FP8 E8M0

        if mode == EpMode.INTRA_NODE:  # 如果是节点内模式
            if num_max_dispatch_tokens_per_rank < 128:  # 如果最大分发令牌数小于128
                block_num = 225  # 使用较多块数
                warp_num_per_block = 5  # 使用较少warp数
            else:
                block_num = 256  # 使用更多块数
                warp_num_per_block = 16  # 使用更多warp数

    # Fp8 blockwise combine uses its own internal scale_dim driven which can be
    # overridden by env ``MORI_FP8_COMBINE_SCALE_DIM`` (default 56)
    # See https://github.com/ROCm/mori/blob/96ffa169710f214e76e07abe5008d686fe54522b/python/mori/ops/dispatch_combine.py#L81-L84
    # FP8块量化合并使用其内部scale_dim驱动，可通过环境变量`MORI_FP8_COMBINE_SCALE_DIM`覆盖（默认56） # FP8块量化合并使用其内部scale_dim驱动，可通过环境变量MORI_FP8_COMBINE_SCALE_DIM覆盖
    combine_quant_type = "none"  # 合并量化类型默认为none
    if combine_dtype == CombineDtype.fp8:  # 如果合并数据类型为FP8
        combine_quant_type = "fp8_blockwise"  # 使用FP8块量化
    elif combine_dtype == CombineDtype.fp8_direct_cast:  # 如果合并数据类型为FP8直接转换
        combine_quant_type = "fp8_direct_cast"  # 使用FP8直接转换

    logger.info(
        f"[MORI init] {world_size=} {rank=} {hidden_size=} {params_dtype=} "
        f"{num_max_dispatch_tokens_per_rank=} {num_local_experts=} "
        f"{router_topk=} {mode=} {dispatch_dtype=} {combine_dtype=} "
    )  # 记录Mori初始化信息

    def check_mori_compatibility(kwargs: dict) -> None:  # 检查Mori兼容性，移除不支持的参数
        """Remove kwargs not accepted by the installed mori's EpDispatchCombineConfig."""  # 移除已安装mori的EpDispatchCombineConfig不接受的参数。 # 移除已安装mori的EpDispatchCombineConfig不接受的参数
        import dataclasses

        config_cls = mori.ops.EpDispatchCombineConfig  # 获取配置类
        valid_kwargs = {f.name for f in dataclasses.fields(config_cls)}  # 获取有效的参数名集合

        invalid_kwargs = set(kwargs.keys()) - valid_kwargs  # 找出无效的参数
        for arg in invalid_kwargs:  # 遍历无效参数
            logger.warning(f"[MORI compat] Removing incompatible argument {arg} ")  # 记录警告信息
            del kwargs[arg]  # 删除无效参数

    # Definition refer to https://github.com/ROCm/mori/blob/f9be5ee2e5ac87256b9523399ae9d4d0e8a54f53/python/mori/ops/dispatch_combine.py#L66-L121
    # 定义参考 https://github.com/ROCm/mori/blob/... # 定义参考上述URL
    common_kwargs = dict(  # 公共参数字典
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_token_type_size=params_dtype.itemsize,
        max_num_inp_token_per_rank=num_max_dispatch_tokens_per_rank,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=router_topk,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        max_total_recv_tokens=get_int_env_var(
            "SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS", 0
        ),
        kernel_type=kernel_type,
        gpu_per_node=gpu_per_node,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=2,  # Number of queue pairs per processing element # 每个处理元素的队列对数
        quant_type=combine_quant_type,
    )

    check_mori_compatibility(common_kwargs)  # 检查参数兼容性

    mori_config = mori.ops.EpDispatchCombineConfig(**common_kwargs)  # 创建Mori配置
    mori_op = mori.ops.EpDispatchCombineOp(mori_config)  # 创建Mori操作实例
    return mori_op  # 返回Mori操作实例


class CommStreamPool:  # 通信流池，管理CUDA流的复用
    _streams = {}  # key -> torch.cuda.Stream # 键到CUDA流的映射

    @classmethod
    def _make_key(cls, group):  # 生成流的缓存键
        return (torch.cuda.current_device(), id(group))  # 使用设备ID和进程组ID作为键

    @classmethod
    def get_stream_from_pool(cls, group) -> torch.cuda.Stream:  # 从池中获取或创建CUDA流
        key = cls._make_key(group)  # 生成键
        stream = cls._streams.get(key)  # 尝试获取已缓存的流
        if stream is None:  # 如果流不存在
            stream = torch.cuda.Stream(priority=0)  # 创建新的CUDA流
            cls._streams[key] = stream  # 缓存流
        return stream  # 返回流

    @classmethod
    def clear_group(cls, group):  # 清除指定进程组的缓存流
        key = (torch.cuda.current_device(), id(group))  # 生成键
        cls._streams.pop(key, None)  # 移除缓存流


class _MoriEPDispatcherImplBase:  # Mori EP分发器实现基类
    def __init__(  # 初始化Mori EP分发器实现基类
        self,
        group: torch.distributed.ProcessGroup,  # 进程组
        router_topk: int,  # 路由器TopK值
        permute_fusion: bool,  # 是否启用排列融合
        num_experts: int,  # 专家总数
        num_local_experts: int,  # 本地专家数
        hidden_size: int,  # 隐藏层大小
        params_dtype: torch.dtype,  # 参数数据类型
        deepep_mode: DeepEPMode,  # DeepEP模式
        instance_id: int = 0,  # 实例ID，默认0
    ):
        try:
            import mori  # noqa: F401 # 尝试导入Mori
        except ImportError:
            raise ImportError("Mori EP is not installed. Please install.")  # Mori EP未安装
        self.group = group  # 保存进程组引用
        self.router_topk = router_topk  # 保存路由器TopK值
        self.permute_fusion = permute_fusion  # 保存排列融合标志
        self.num_experts = num_experts  # 保存专家总数
        self.num_local_experts = num_local_experts  # 保存本地专家数
        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.params_dtype = params_dtype  # 保存参数数据类型
        self.deepep_mode = deepep_mode  # 保存DeepEP模式
        self.instance_id = instance_id  # 保存实例ID

        self.num_max_dispatch_tokens_per_rank = get_int_env_var(
            "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 4096
        )  # 从环境变量获取每个rank最大分发令牌数，默认4096

        self.enable_sdma = get_bool_env_var("MORI_ENABLE_SDMA", "false")  # 是否启用SDMA，默认False

        self._mori_op = None  # Mori操作实例，延迟初始化
        self.dispatch_dtype = DispatchDtype.bf16  # 分发数据类型默认BF16
        self.combine_dtype = CombineDtype.bf16  # 合并数据类型默认BF16

        self.quant_config: Optional[dict] = None  # 量化配置，可选

        self.overlap_args: Optional[CombineOverlapArgs] = None  # 重叠参数，可选
        self.meta_overlap_args: Optional[dict] = None  # 元数据重叠参数，可选

    @property
    def mori_op(self):  # 获取Mori操作实例（延迟初始化属性）
        if self._mori_op is None:  # 如果Mori操作实例尚未创建
            # If set_quant_config was never called, apply env var override now
            # 如果set_quant_config从未被调用，现在应用环境变量覆盖 # 如果set_quant_config从未被调用，现在应用环境变量覆盖
            if self.quant_config is None:
                self._apply_dispatch_dtype_override()
            self._mori_op = init_mori_op(  # 初始化Mori操作实例
                self.group,
                self.router_topk,
                self.num_experts,
                self.num_local_experts,
                self.hidden_size,
                self.params_dtype,
                self.num_max_dispatch_tokens_per_rank,
                self.deepep_mode,
                self.instance_id,
                self.dispatch_dtype,
                self.combine_dtype,
                self.enable_sdma,
            )
        return self._mori_op  # 返回Mori操作实例

    def _apply_dispatch_dtype_override(self):  # 应用环境变量覆盖分发/合并数据类型
        """Apply env var override to fp8_dispatch/fp4_dispatch/fp8_combine flags."""  # 应用环境变量覆盖fp8_dispatch/fp4_dispatch/fp8_combine标志。 # 应用环境变量覆盖fp8_dispatch/fp4_dispatch/fp8_combine标志
        if "SGLANG_MORI_DISPATCH_DTYPE" in os.environ:  # 如果设置了分发数据类型环境变量
            dispatch_dtype = os.environ["SGLANG_MORI_DISPATCH_DTYPE"].lower()  # 获取并转为小写
            if dispatch_dtype != "auto":  # 如果不是auto
                if dispatch_dtype == "bf16":  # BF16
                    self.dispatch_dtype = DispatchDtype.bf16
                elif dispatch_dtype == "fp8":  # FP8
                    self.dispatch_dtype = DispatchDtype.fp8
                elif dispatch_dtype == "fp4":  # FP4
                    self.dispatch_dtype = DispatchDtype.fp4
        elif (
            "SGLANG_MORI_FP8_DISP" in os.environ or "SGLANG_MORI_FP4_DISP" in os.environ
        ):  # 如果设置了旧的FP8/FP4分发环境变量
            # Deprecated: will be removed in a future release
            # 已弃用：将在未来版本中移除 # 已弃用：将在未来版本中移除
            logger.warning_once(
                "SGLANG_MORI_FP8_DISP and SGLANG_MORI_FP4_DISP are deprecated "
                "and will be removed in a future release. "
                "Use SGLANG_MORI_DISPATCH_DTYPE=auto|bf16|fp8|fp4 instead."
            )  # 旧的环境变量已弃用，请使用新变量
            if get_bool_env_var("SGLANG_MORI_FP8_DISP", "False"):
                self.dispatch_dtype = DispatchDtype.fp8  # 覆盖为FP8
            if get_bool_env_var("SGLANG_MORI_FP4_DISP", "False"):
                self.dispatch_dtype = DispatchDtype.fp4  # 覆盖为FP4

        if "SGLANG_MORI_COMBINE_DTYPE" in os.environ:  # 如果设置了合并数据类型环境变量
            combine_dtype = os.environ["SGLANG_MORI_COMBINE_DTYPE"].lower()  # 获取并转为小写
            if combine_dtype != "auto":  # 如果不是auto
                if combine_dtype == "fp8":  # FP8
                    self.combine_dtype = CombineDtype.fp8
                elif combine_dtype == "bf16":  # BF16
                    self.combine_dtype = CombineDtype.bf16
                elif combine_dtype == "fp8_direct_cast":  # FP8直接转换
                    self.combine_dtype = CombineDtype.fp8_direct_cast
        elif "SGLANG_MORI_FP8_COMB" in os.environ:  # 如果设置了旧的FP8合并环境变量
            # Deprecated: will be removed in a future release
            # 已弃用：将在未来版本中移除 # 已弃用：将在未来版本中移除
            logger.warning_once(
                "SGLANG_MORI_FP8_COMB is deprecated "
                "and will be removed in a future release. "
                "Use SGLANG_MORI_COMBINE_DTYPE=auto|bf16|fp8|fp8_direct_cast instead."
            )  # 旧的环境变量已弃用，请使用新变量
            if get_bool_env_var("SGLANG_MORI_FP8_COMB", "False"):
                self.combine_dtype = CombineDtype.fp8  # 覆盖为FP8

    def dispatch_a(  # 分发阶段A（抽象方法，子类实现）
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_output: TopKOutput,  # TopK选择结果
    ):
        raise NotImplementedError  # 子类必须实现

    def dispatch_b(self, *args, **kwargs):  # 分发阶段B（抽象方法，子类实现）
        raise NotImplementedError  # 子类必须实现

    def combine_a(  # 合并阶段A（抽象方法，子类实现）
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK专家权重
    ):
        raise NotImplementedError  # 子类必须实现

    def combine_b(self, *args, **kwargs):  # 合并阶段B（抽象方法，子类实现）
        raise NotImplementedError  # 子类必须实现

    def set_quant_config(self, quant_config: dict) -> None:  # 设置量化配置
        self.quant_config = quant_config  # 保存量化配置
        # Auto-detect dispatch quantization from weight dtype
        # 根据权重数据类型自动检测分发量化方式 # 根据权重数据类型自动检测分发量化方式
        weight_dtype = quant_config.get("weight_dtype", None)  # 获取权重数据类型
        if weight_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):  # FP8权重
            self.dispatch_dtype = DispatchDtype.fp8  # 分发使用FP8
            self.combine_dtype = CombineDtype.bf16  # 合并使用BF16
        elif weight_dtype == torch.float4_e2m1fn_x2:  # FP4权重
            self.dispatch_dtype = DispatchDtype.fp4  # 分发使用FP4
            self.combine_dtype = CombineDtype.fp8  # 合并使用FP8
        else:  # 其他权重类型
            self.dispatch_dtype = DispatchDtype.bf16  # 分发使用BF16
            self.combine_dtype = CombineDtype.bf16  # 合并使用BF16
        # Apply env var override immediately so dispatch_a sees correct flags
        # 立即应用环境变量覆盖，使dispatch_a能看到正确的标志 # 立即应用环境变量覆盖，使dispatch_a能看到正确的标志
        self._apply_dispatch_dtype_override()

    def set_overlap_args(  # 设置重叠参数
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ) -> None:
        self.overlap_args = combine_overlap_args  # 保存合并重叠参数
        self.meta_overlap_args = meta_overlap_args  # 保存元数据重叠参数

    def clear_overlap_args(self) -> None:  # 清除重叠参数
        self.overlap_args = None  # 清空合并重叠参数
        self.meta_overlap_args = None  # 清空元数据重叠参数


class _MoriEPDispatcherImplNormal(_MoriEPDispatcherImplBase):  # Mori EP普通模式分发器实现
    def __init__(self, async_finish: bool, **kwargs):  # 初始化普通模式分发器
        super().__init__(**kwargs)  # 调用父类初始化

        self.async_finish = async_finish  # 是否异步完成
        self.quant_config = {}  # 量化配置初始化为空字典
        self.fp8_quant_func = get_hip_quant(QuantType.per_1x128)  # FP8量化函数
        self.fp4_quant_func = get_hip_quant(QuantType.per_1x32)  # FP4量化函数
        self.enable_dual_stream = is_tbo_enabled()  # 是否启用双流（基于TBO）
        self._comm_stream = None  # 通信流
        if self.enable_dual_stream:  # 如果启用双流
            self._comm_stream = CommStreamPool.get_stream_from_pool(self.group)  # 从流池获取通信流

    def _capture_event_if_async(self) -> Optional[torch.cuda.Event]:  # 如果启用异步，捕获当前流上的事件
        assert self.enable_dual_stream, "dual stream must be enabled"  # 断言必须启用双流
        if not self.async_finish:  # 如果未启用异步完成
            return None  # 返回None
        ev = torch.cuda.Event(blocking=False, interprocess=False)  # 创建非阻塞事件
        ev.record(torch.cuda.current_stream())  # 在当前流上记录事件
        return ev  # 返回事件

    def dispatch_a(  # 普通模式分发阶段A：执行量化并启动分发
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_output: TopKOutput,  # TopK选择结果
    ):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids  # 解包TopK输出

        num_token = hidden_states.shape[0]  # 令牌数
        output_dtype = hidden_states.dtype  # 输出数据类型
        scale = None  # 缩放因子初始化为None

        if self.dispatch_dtype == DispatchDtype.fp8:  # 如果分发数据类型为FP8
            # FP8 quant # FP8量化
            if num_token > 0:  # 如果有令牌
                # NOTE: aiter is able to handle token=0 case in UT. But for some
                # reason it failed at e2e case. Root cause TBD.
                # 注意：aiter在单元测试中能处理token=0的情况，但在端到端场景中失败。根因待定。 # 注意：aiter在UT中能处理token=0的情况，但在e2e中失败，根因待定
                hidden_states, scale = self.fp8_quant_func(
                    hidden_states, quant_dtype=fp8_dtype
                )  # 执行FP8量化
            else:  # 没有令牌
                hidden_states = torch.empty(
                    hidden_states.shape, dtype=fp8_dtype, device=hidden_states.device
                )  # 创建空的FP8张量
                scale = torch.empty(
                    (0, self.hidden_size // FP8_BLOCK_SIZE),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )  # 创建空的缩放因子

        elif self.dispatch_dtype == DispatchDtype.fp4:  # 如果分发数据类型为FP4
            # FP4 quant # FP4量化
            if num_token > 0:  # 如果有令牌
                hidden_states, scale = self.fp4_quant_func(hidden_states, shuffle=False)  # 执行FP4量化
            else:  # 没有令牌
                hidden_states = torch.empty(
                    (0, self.hidden_size // 2),
                    dtype=torch.float4_e2m1fn_x2,
                    device=hidden_states.device,
                )  # 创建空的FP4张量
                scale = torch.empty(
                    (0, self.hidden_size // MXFP4_BLOCK_SIZE),
                    dtype=torch.float8_e8m0fnu,
                    device=hidden_states.device,
                )  # 创建空的缩放因子

        previous_event = self._capture_event_if_async() if self._comm_stream else None  # 如果有通信流则捕获事件

        return (  # 返回分发中间状态
            hidden_states,
            topk_weights,
            topk_ids,
            scale,
            output_dtype,
            previous_event,
        )

    def dispatch_b(  # 普通模式分发阶段B：执行核心分发并构造输出
        self,
        hidden_states,  # 隐藏状态
        topk_weights,  # TopK权重
        topk_ids,  # TopK ID
        scale,  # 缩放因子
        output_dtype,  # 输出数据类型
        previous_event,  # 前一个事件
    ):

        (
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
            done_event,
        ) = self._dispatch_core(  # 执行分发核心逻辑
            hidden_states,
            topk_weights,
            topk_ids,
            scale=scale,
            previous_event=previous_event,
        )

        if self._comm_stream and self.async_finish and done_event is not None:  # 如果启用双流异步且有完成事件
            torch.cuda.current_stream().wait_event(done_event)  # 当前流等待完成事件

        return MoriEPNormalDispatchOutput(  # 构造并返回普通模式分发输出
            hidden_states=packed_recv_hidden,
            hidden_states_scale=recv_scales,
            topk_ids=recv_topk_ids,
            topk_weights=recv_topk_weights,
            num_recv_tokens_per_expert=packed_recv_count,
            origin_topk_ids=topk_ids,
            origin_topk_weights=topk_weights,
            out_dtype=output_dtype,
        )

    def _dispatch_core(  # 普通模式分发核心逻辑
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_weights: torch.Tensor,  # TopK权重
        topk_ids: torch.Tensor,  # TopK ID
        scale: Optional[torch.Tensor] = None,  # 缩放因子，可选
        previous_event: Optional[torch.cuda.Event] = None,  # 前一个事件，可选
    ):
        done_event: Optional[torch.cuda.Event] = None  # 完成事件初始化为None

        if self._comm_stream:  # 如果有通信流（双流模式）
            compute_stream = torch.cuda.current_stream()  # 获取计算流
            comm_stream = self._comm_stream  # comm stream # 通信流

            for t in (hidden_states, topk_weights, topk_ids):  # 记录张量使用的流
                t.record_stream(comm_stream)
            if scale is not None:  # 如果有缩放因子
                scale.record_stream(comm_stream)  # 记录缩放因子使用的流

            with torch.cuda.stream(comm_stream):  # 在通信流上执行分发
                # if (previous_event) stream_wait(comm_stream, previous_event)
                # else stream_wait(comm_stream, compute_stream)
                # 如果有前一个事件则等待它，否则等待计算流 # 如果有previous_event则等待它，否则等待计算流

                if previous_event is not None:  # 如果有前一个事件
                    comm_stream.wait_event(previous_event)  # 通信流等待前一个事件
                else:
                    comm_stream.wait_stream(compute_stream)  # 通信流等待计算流

                dispatch_fn = (  # 选择分发函数
                    self.mori_op.dispatch_send
                    if self.enable_sdma  # 如果启用SDMA使用dispatch_send
                    else self.mori_op.dispatch  # 否则使用dispatch
                )
                (
                    packed_recv_hidden,
                    recv_topk_weights,
                    recv_scales,
                    recv_topk_ids,
                    packed_recv_count,
                ) = dispatch_fn(hidden_states, topk_weights, scale, topk_ids)  # 执行分发
                if self.enable_sdma:  # 如果启用SDMA
                    self.mori_op.dispatch_recv()  # 执行SDMA接收

                if self.async_finish:  # 如果启用异步完成
                    done_event = torch.cuda.Event(blocking=False, interprocess=False)  # 创建完成事件
                    done_event.record(comm_stream)  # 在通信流上记录完成事件
                else:
                    compute_stream.wait_stream(comm_stream)  # 计算流等待通信流

            for t in (  # 记录接收张量使用的流
                packed_recv_hidden,
                recv_topk_weights,
                recv_scales,
                recv_topk_ids,
            ):
                if t is not None:  # 如果张量不为None
                    t.record_stream(comm_stream)  # 记录使用的流
        else:  # 没有通信流（单流模式）

            (
                packed_recv_hidden,
                recv_topk_weights,
                recv_scales,
                recv_topk_ids,
                packed_recv_count,
            ) = self.mori_op.dispatch(hidden_states, topk_weights, scale, topk_ids)  # 在当前流上执行分发

        # TODO(billishyahao): EPLB
        # get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
        # TODO: EPLB专家并行负载均衡 # TODO: EPLB专家并行负载均衡

        return (  # 返回分发结果
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
            done_event,
        )

    def combine_a(  # 普通模式合并阶段A
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK专家权重
    ):
        previous_event = self._capture_event_if_async() if self._comm_stream else None  # 捕获事件
        return hidden_states, topk_ids, topk_weights, previous_event  # 返回中间状态

    def combine_b(self, hidden_states, topk_ids, topk_weights, previous_event):  # 普通模式合并阶段B

        hidden_states, done_event = self._combine_core(  # 执行合并核心逻辑
            hidden_states, topk_ids, topk_weights, previous_event
        )

        if self._comm_stream and self.async_finish and done_event is not None:  # 如果启用双流异步且有完成事件
            torch.cuda.current_stream().wait_event(done_event)  # 当前流等待完成事件

        return hidden_states  # 返回合并后的隐藏状态

    def _combine_core(  # 普通模式合并核心逻辑
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK专家权重
        previous_event: Optional[torch.cuda.Event],  # 前一个事件
    ):
        done_event: Optional[torch.cuda.Event] = None  # 完成事件初始化为None

        if self._comm_stream:  # 如果有通信流（双流模式）
            compute_stream = torch.cuda.current_stream()  # 获取计算流
            comm_stream = self._comm_stream  # 获取通信流

            for t in (hidden_states, topk_ids, topk_weights):  # 记录张量使用的流
                t.record_stream(comm_stream)

            with torch.cuda.stream(comm_stream):  # 在通信流上执行合并
                if previous_event is not None:  # 如果有前一个事件
                    comm_stream.wait_event(previous_event)  # 通信流等待前一个事件
                else:
                    comm_stream.wait_stream(compute_stream)  # 通信流等待计算流

                combine_fn = (  # 选择合并函数
                    self.mori_op.combine_send
                    if self.enable_sdma  # 如果启用SDMA使用combine_send
                    else self.mori_op.combine  # 否则使用combine
                )
                combined_hidden_states = combine_fn(hidden_states, None, topk_ids)[0]  # 执行合并
                if self.enable_sdma:  # 如果启用SDMA
                    self.mori_op.combine_recv()  # 执行SDMA接收

                if self.async_finish:  # 如果启用异步完成
                    done_event = torch.cuda.Event(blocking=False, interprocess=False)  # 创建完成事件
                    done_event.record(comm_stream)  # 在通信流上记录完成事件
                else:
                    compute_stream.wait_stream(comm_stream)  # 计算流等待通信流

            combined_hidden_states.record_stream(comm_stream)  # 记录合并结果使用的流

        else:  # 没有通信流（单流模式）
            combined_hidden_states = self.mori_op.combine(
                hidden_states, None, topk_ids
            )[0]  # 在当前流上执行合并

        return combined_hidden_states, done_event  # 返回合并结果和完成事件

    def set_quant_config(self, quant_config: dict):  # 设置量化配置
        super().set_quant_config(quant_config)  # 调用父类方法


class _MoriEPDispatcherImplLowLatency(_MoriEPDispatcherImplBase):  # Mori EP低延迟模式分发器实现
    def __init__(self, **kwargs):  # 初始化低延迟模式分发器
        super().__init__(**kwargs)  # 调用父类初始化
        self.quant_config = {}  # 量化配置初始化为空字典
        self.fp8_quant_func = get_hip_quant(QuantType.per_1x128)  # FP8量化函数
        self.fp4_quant_func = get_hip_quant(QuantType.per_1x32)  # FP4量化函数

    def dispatch_a(  # 低延迟模式分发阶段A：量化并启动异步发送
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_output: TopKOutput,  # TopK选择结果
    ):
        import mori

        assert (
            self.mori_op.config.kernel_type
            is mori.ops.EpDispatchCombineKernelType.AsyncLL
        ), "mori asyncll mismatch"  # 断言内核类型为AsyncLL

        num_tokens = hidden_states.shape[0]  # 令牌数
        output_dtype = hidden_states.dtype  # 输出数据类型
        scale = None  # 缩放因子初始化为None

        if self.dispatch_dtype == DispatchDtype.fp8:  # 如果分发数据类型为FP8
            # FP8 quant # FP8量化
            if num_tokens > 0:  # 如果有令牌
                # NOTE: aiter is able to handle token=0 case in UT. But for some
                # reason it failed at e2e case. Root cause TBD.
                # 注意：aiter在单元测试中能处理token=0的情况，但在端到端场景中失败。根因待定。 # 注意：aiter在UT中能处理token=0的情况，但在e2e中失败，根因待定
                hidden_states, scale = self.fp8_quant_func(
                    hidden_states, quant_dtype=fp8_dtype
                )  # 执行FP8量化
            else:  # 没有令牌
                hidden_states = torch.empty(
                    hidden_states.shape, dtype=fp8_dtype, device=hidden_states.device
                )  # 创建空的FP8张量
                scale = torch.empty(
                    (0, self.hidden_size // FP8_BLOCK_SIZE),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )  # 创建空的缩放因子

        elif self.dispatch_dtype == DispatchDtype.fp4:  # 如果分发数据类型为FP4
            # FP4 quant # FP4量化
            if num_tokens > 0:  # 如果有令牌
                hidden_states, scale = self.fp4_quant_func(hidden_states, shuffle=False)  # 执行FP4量化
            else:  # 没有令牌
                hidden_states = torch.empty(
                    (0, self.hidden_size // 2),
                    dtype=torch.float4_e2m1fn_x2,
                    device=hidden_states.device,
                )  # 创建空的FP4张量
                scale = torch.empty(
                    (0, self.hidden_size // MXFP4_BLOCK_SIZE),
                    dtype=torch.float8_e8m0fnu,
                    device=hidden_states.device,
                )  # 创建空的缩放因子

        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids  # 解包TopK输出

        (
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
        ) = self._dispatch_core(hidden_states, topk_weights, topk_ids, scale=scale)  # 执行分发核心逻辑

        return (  # 返回分发中间状态
            packed_recv_hidden,
            recv_topk_weights,
            recv_topk_ids,
            recv_scales,
            packed_recv_count,
            topk_weights,
            topk_ids,
            output_dtype,
        )

    def dispatch_b(  # 低延迟模式分发阶段B：等待异步接收并构造输出
        self,
        hidden_states,  # 隐藏状态
        recv_topk_weights,  # 接收的TopK权重
        recv_topk_ids,  # 接收的TopK ID
        recv_scales,  # 接收的缩放因子
        packed_recv_count,  # 接收的令牌计数
        topk_weights,  # 原始TopK权重
        topk_ids,  # 原始TopK ID
        output_dtype,  # 输出数据类型
    ):

        ##TODO(billishyahao): add assertion here to check async
        ##TODO: 在此处添加断言以检查异步 # TODO: 在此处添加断言以检查异步
        import mori

        assert (
            self.mori_op.config.kernel_type
            is mori.ops.EpDispatchCombineKernelType.AsyncLL
        ), "mori asyncll mismatch"  # 断言内核类型为AsyncLL

        self.mori_op.dispatch_recv()  # 执行异步接收

        return MoriEPLLDispatchOutput(  # 构造并返回低延迟模式分发输出
            hidden_states=hidden_states,
            hidden_states_scale=recv_scales,
            topk_ids=recv_topk_ids,
            topk_weights=recv_topk_weights,
            num_recv_tokens_per_expert=packed_recv_count,
            origin_topk_ids=topk_ids,
            origin_topk_weights=topk_weights,
            out_dtype=output_dtype,
        )

    def _dispatch_core(  # 低延迟模式分发核心逻辑：执行异步发送
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_weights: torch.Tensor,  # TopK权重
        topk_ids: torch.Tensor,  # TopK ID
        scale: Optional[torch.Tensor] = None,  # 缩放因子，可选
    ):
        ##TODO(billishyahao): add assertion here to check async
        ##TODO: 在此处添加断言以检查异步 # TODO: 在此处添加断言以检查异步

        (
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
        ) = self.mori_op.dispatch_send(hidden_states, topk_weights, scale, topk_ids)  # 执行异步发送

        return (  # 返回发送结果
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
        )

    def combine_a(  # 低延迟模式合并阶段A
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK专家权重
        overlap_args: Optional[CombineOverlapArgs] = None,  # 重叠参数，可选
    ):
        hidden_states = self._combine_core(  # 执行合并核心逻辑
            hidden_states,
            topk_ids,
            topk_weights,
            overlap_args=overlap_args,
        )
        return hidden_states, topk_ids, topk_weights, overlap_args  # 返回中间状态

    def combine_b(self, hidden_states, topk_ids, topk_weights, previous_event):  # 低延迟模式合并阶段B

        self.mori_op.combine_recv()  # 执行异步接收

        return hidden_states[0]  # 返回合并后的隐藏状态（取第一个元素）

    def _combine_core(  # 低延迟模式合并核心逻辑：执行异步发送
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK专家权重
        overlap_args: Optional[CombineOverlapArgs] = None,  # 重叠参数，可选
    ):
        combined_hidden_states = self.mori_op.combine_send(
            hidden_states, None, topk_ids
        )  # 执行异步合并发送

        return combined_hidden_states  # 返回合并结果

    def set_quant_config(self, quant_config: dict):  # 设置量化配置
        super().set_quant_config(quant_config)  # 调用父类方法


@dataclass
class _Stage(Enum):  # 分发/合并阶段枚举，用于状态机管理
    INITIAL = auto()  # 初始状态
    AFTER_DISPATCH_A = auto()  # 分发阶段A完成后
    AFTER_DISPATCH_B = auto()  # 分发阶段B完成后
    AFTER_COMBINE_A = auto()  # 合并阶段A完成后


class MoriEPDispatcher(BaseDispatcher):  # Mori EP分发器，继承自BaseDispatcher
    def __init__(  # 初始化Mori EP分发器
        self,
        group: torch.distributed.ProcessGroup,  # 进程组
        router_topk: int,  # 路由器TopK值
        permute_fusion: bool = False,  # 是否启用排列融合，默认False
        num_experts: int = None,  # 专家总数
        num_local_experts: int = None,  # 本地专家数
        hidden_size: int = None,  # 隐藏层大小
        params_dtype: torch.dtype = None,  # 参数数据类型
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,  # DeepEP模式，默认AUTO
        async_finish: bool = False,  # 是否异步完成，默认False
        return_recv_hook: bool = False,  # 是否使用接收钩子，默认False
        instance_id: int = 0,  # 实例ID，默认0
    ):
        super().__init__()  # 调用父类初始化

        self.deepep_mode = deepep_mode  # 保存DeepEP模式

        async_mode = self.deepep_mode.enable_low_latency()  # 判断是否为异步模式
        if get_bool_env_var("SGLANG_ROCM_USE_MULTI_STREAM") and not async_mode:  # 如果设置了多流但未启用异步
            logger.warning_once(
                "SGLANG_ROCM_USE_MULTI_STREAM=1 is set but Mori AsyncLL is "
                "not enabled (--deepep-mode=%s). The alt-stream overlap only "
                "frees up CUs when dispatch/combine runs on the AsyncLL "
                "copy-engine kernel; otherwise it stays on CUs and competes "
                "with the alt-stream work. Pass --deepep-mode low_latency "
                "(or auto) to enable the AsyncLL kernel.",
                self.deepep_mode.value,
            )  # 警告：多流仅在AsyncLL内核下有效

        common_kwargs = dict(  # 公共参数字典
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
            instance_id=instance_id,
        )

        if self.deepep_mode.enable_low_latency():  # 如果启用低延迟模式
            self._low_latency_dispatcher = _MoriEPDispatcherImplLowLatency(
                **common_kwargs,
            )  # 创建低延迟分发器实现

        if self.deepep_mode.enable_normal():  # 如果启用普通模式
            self._normal_dispatcher = _MoriEPDispatcherImplNormal(
                async_finish=async_finish,
                **common_kwargs,
            )  # 创建普通模式分发器实现

        self._stage = _Stage.INITIAL  # 初始化阶段状态为INITIAL
        self._deepep_dispatch_hooks = MoriEPPDispatchHooks()  # 创建分发钩子

        # Mori dispatch produces global topk_ids in [0, num_experts); mask out
        # experts that are not local to this rank.
        # Mori分发产生全局topk_ids在[0, num_experts)范围内；屏蔽不属于本rank的本地专家。 # Mori分发产生全局topk_ids在[0, num_experts)范围内；屏蔽不属于本rank的专家
        self.expert_mask_gpu = None  # GPU上的专家掩码
        if _use_aiter and num_experts is not None and num_local_experts is not None:  # 如果使用AITer且专家参数已设置
            ep_rank = get_moe_expert_parallel_rank()  # 获取EP rank
            expert_mask = torch.zeros(
                num_experts,
                device=torch.cuda.current_device(),
                dtype=torch.int32,
            )  # 创建专家掩码，初始全0
            start = ep_rank * num_local_experts  # 本地专家起始索引
            expert_mask[start : start + num_local_experts] = 1  # 标记本地专家为1
            self.expert_mask_gpu = expert_mask  # 保存专家掩码

    def dispatch(  # 同步分发方法：依次执行dispatch_a、钩子和dispatch_b
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_output: TopKOutput,  # TopK选择结果
    ) -> DispatchOutput:
        self._num_tokens = hidden_states.shape[0]  # 记录令牌数
        self.dispatch_a(hidden_states, topk_output)  # 执行分发阶段A
        if self._deepep_dispatch_hooks is not None:  # 如果有分发钩子
            self._deepep_dispatch_hooks(self)  # 执行分发钩子
        ret = self.dispatch_b()  # 执行分发阶段B
        return ret  # 返回分发输出

    def dispatch_a(  # 分发阶段A：更新状态并调用内部实现的dispatch_a
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_output: TopKOutput,  # TopK选择结果
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)  # 更新阶段状态
        inner_state = self._get_impl().dispatch_a(  # 调用内部实现的dispatch_a
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state  # 保存分发中间状态

    def dispatch_b(self):  # 分发阶段B：更新状态并调用内部实现的dispatch_b
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)  # 更新阶段状态
        inner_state = self._dispatch_intermediate_state  # 获取分发中间状态
        del self._dispatch_intermediate_state  # 删除中间状态引用
        return self._get_impl().dispatch_b(*inner_state)  # 调用内部实现的dispatch_b并返回结果

    def combine(  # 同步合并方法：依次执行combine_a和combine_b
        self,
        combine_input: CombineInput,  # 合并输入
    ) -> Tuple:
        self.combine_a(combine_input)  # 执行合并阶段A
        hidden_states = self.combine_b()  # 执行合并阶段B
        return hidden_states[: self._num_tokens]  # 返回截取到原始令牌数的隐藏状态

    def combine_a(  # 合并阶段A：更新状态并调用内部实现的combine_a
        self,
        combine_input: CombineInput,  # 合并输入
    ):
        hidden_states, topk_ids, topk_weights = combine_input  # 解包合并输入
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)  # 更新阶段状态
        inner_state = self._get_impl().combine_a(  # 调用内部实现的combine_a
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state  # 保存合并中间状态

    def combine_b(self):  # 合并阶段B：更新状态并调用内部实现的combine_b
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)  # 更新阶段状态回初始
        inner_state = self._combine_intermediate_state  # 获取合并中间状态
        del self._combine_intermediate_state  # 删除中间状态引用
        return self._get_impl().combine_b(*inner_state)  # 调用内部实现的combine_b并返回结果

    def _get_impl(self) -> _MoriEPDispatcherImplBase:  # 根据当前状态获取对应的分发器实现
        is_extend_in_batch = get_is_extend_in_batch()  # 获取当前是否为扩展阶段
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)  # 解析实际的DeepEP模式
        if resolved_deepep_mode == DeepEPMode.NORMAL:  # 如果解析为普通模式
            return self._normal_dispatcher  # 返回普通模式分发器实现
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:  # 如果解析为低延迟模式
            return self._low_latency_dispatcher  # 返回低延迟分发器实现
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")  # 无效的DeepEP模式

    def _update_stage(self, old_stage, new_stage):  # 更新阶段状态，确保状态转换合法
        assert self._stage == old_stage  # 断言当前阶段等于期望的旧阶段
        self._stage = new_stage  # 更新为新阶段

    def set_quant_config(self, quant_config: dict):  # 设置量化配置
        super().set_quant_config(quant_config)  # 调用父类方法
        if self.deepep_mode.enable_low_latency():  # 如果启用低延迟模式
            self._low_latency_dispatcher.set_quant_config(quant_config)  # 设置低延迟分发器的量化配置
        if self.deepep_mode.enable_normal():  # 如果启用普通模式
            self._normal_dispatcher.set_quant_config(quant_config)  # 设置普通模式分发器的量化配置

    def set_overlap_args(  # 设置重叠参数
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ):
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)  # 调用父类方法
        if self.deepep_mode.enable_low_latency():  # 如果启用低延迟模式
            self._low_latency_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )  # 设置低延迟分发器的重叠参数
        if self.deepep_mode.enable_normal():  # 如果启用普通模式
            self._normal_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )  # 设置普通模式分发器的重叠参数

    def clear_overlap_args(self):  # 清除重叠参数
        super().clear_overlap_args()  # 调用父类方法
        if self.deepep_mode.enable_low_latency():  # 如果启用低延迟模式
            self._low_latency_dispatcher.clear_overlap_args()  # 清除低延迟分发器的重叠参数
        if self.deepep_mode.enable_normal():  # 如果启用普通模式
            self._normal_dispatcher.clear_overlap_args()  # 清除普通模式分发器的重叠参数

    def register_deepep_dispatch_hook(self, hook):  # 注册DeepEP分发钩子
        return self._deepep_dispatch_hooks.register_hook(hook)  # 注册钩子并返回
