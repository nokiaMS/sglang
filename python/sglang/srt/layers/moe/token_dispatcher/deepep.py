# DeepEP token分发器模块
# 实现基于DeepEP的MoE专家并行token分发和合并，支持普通模式和低延迟模式
from __future__ import annotations  # 启用延迟注解求值

import logging  # 日志模块
from contextlib import nullcontext  # 空上下文管理器
from dataclasses import dataclass  # 数据类装饰器
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Union  # 类型注解工具

from sglang.srt.distributed.parallel_state import get_tp_group  # 获取张量并行组
from sglang.srt.environ import envs  # 环境变量配置
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 专家分布记录器
from sglang.srt.layers import deep_gemm_wrapper  # DeepGEMM包装器
from sglang.srt.layers.dp_attention import get_is_extend_in_batch  # 获取当前是否在批次中扩展
from sglang.srt.layers.moe.token_dispatcher.base import (  # 基础分发器相关类型
    BaseDispatcher,  # 分发器基类
    BaseDispatcherConfig,  # 分发器配置基类
    CombineInput,  # 合并输入协议
    CombineInputFormat,  # 合并输入格式枚举
    DispatcherBaseHooks,  # 分发器钩子基类
    DispatchOutput,  # 分发输出协议
    DispatchOutputFormat,  # 分发输出格式枚举
)
from sglang.srt.layers.moe.topk import TopKOutput  # TopK输出类型
from sglang.srt.layers.moe.utils import (  # MoE工具函数
    DeepEPMode,  # DeepEP模式枚举
    DeepEPOutputDtype,  # DeepEP输出数据类型枚举
    get_deepep_config,  # 获取DeepEP配置
    get_deepep_output_dtype,  # 获取DeepEP输出数据类型
    is_tbo_enabled,  # 检查TBO是否启用
)
from sglang.srt.utils import (  # 通用工具函数
    get_bool_env_var,  # 获取布尔环境变量
    is_blackwell,  # 检查是否为Blackwell架构
    is_hip,  # 检查是否为AMD GPU
    is_npu,  # 检查是否为NPU
    load_json_config,  # 加载JSON配置
)

_is_npu = is_npu()  # 缓存NPU检查结果

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs  # 合并重叠参数

try:  # 尝试导入DeepEP库
    if _is_npu and envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0:  # NPU且启用了zbal本地内存
        from zbal.zbal.deepep_adaptor import Config  # 导入zbal的DeepEP适配器配置
        from zbal.zbal_buffer import Buffer  # 导入zbal缓冲区
    else:  # GPU环境
        from deep_ep import Buffer, Config  # 导入DeepEP的缓冲区和配置

    if not _is_npu:  # 非NPU环境
        from sglang.srt.layers.quantization.fp8_kernel import (  # 导入FP8量化内核
            sglang_per_token_group_quant_fp8,
        )

    use_deepep = True  # 标记DeepEP可用
except ImportError:  # DeepEP不可用
    use_deepep = False  # 标记DeepEP不可用

from enum import Enum, IntEnum, auto  # 枚举类型工具

import torch  # PyTorch深度学习框架
import torch.distributed as dist  # PyTorch分布式通信模块

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()  # 是否使用AITER（仅AMD GPU）

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def _deepep_precompile_tp_barrier() -> None:  # DeepEP预编译阶段的张量并行屏障同步
    # DeepEP's all-to-all operation has a much shorter timeout compared to torch.distributed,
    # so if different ranks compile at different speeds, it may quickly trigger a timeout.
    # To avoid this, we use torch.distributed's barrier during the compile stage.
    # We apply this barrier only in the compile stage to prevent extra all-reduce overhead at runtime.
    # DeepEP的全对全操作超时时间比torch.distributed短得多，
    # 因此如果不同rank编译速度不同，可能会快速触发超时。
    # 为避免此问题，我们在编译阶段使用torch.distributed的屏障同步。
    # 我们仅在编译阶段应用此屏障，以防止运行时产生额外的全归约开销。
    if envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get():  # 如果当前处于DeepGEMM预编译阶段
        get_tp_group().barrier()  # 执行张量并行组屏障同步


class DeepEPPDispatchHooks(DispatcherBaseHooks):  # DeepEP分发钩子，在dispatch_a和dispatch_b之间执行
    def __call__(self, dispatcher: BaseDispatcher):  # 调用所有注册的钩子
        for hook_fun in self.hook_dict.values():  # 遍历所有钩子函数
            hook_fun(dispatcher)  # 执行钩子函数


class DeepEPNormalDispatchOutput(NamedTuple):  # DeepEP普通模式分发输出
    """DeepEP normal dispatch output."""  # DeepEP普通模式分发输出
    # DeepEP普通模式分发输出

    hidden_states: torch.Tensor  # 隐藏状态张量
    hidden_states_scale: Optional[torch.Tensor]  # 隐藏状态缩放因子（FP8量化时使用）
    topk_ids: torch.Tensor  # TopK专家ID
    topk_weights: torch.Tensor  # TopK权重
    num_recv_tokens_per_expert: List[int]  # 每个专家接收到的token数量列表

    @property
    def format(self) -> DispatchOutputFormat:  # 输出格式属性
        return DispatchOutputFormat.DEEPEP_NORMAL  # 返回DeepEP普通模式格式


class DeepEPLLDispatchOutput(NamedTuple):  # DeepEP低延迟模式分发输出
    """DeepEP low latency dispatch output."""  # DeepEP低延迟分发输出
    # DeepEP低延迟分发输出

    hidden_states: torch.Tensor  # 隐藏状态张量
    hidden_states_scale: Optional[torch.Tensor]  # 隐藏状态缩放因子
    topk_ids: torch.Tensor  # TopK专家ID
    topk_weights: torch.Tensor  # TopK权重
    masked_m: torch.Tensor  # 掩码矩阵，标记有效token位置
    expected_m: int  # 预期的token数量

    @property
    def format(self) -> DispatchOutputFormat:  # 输出格式属性
        return DispatchOutputFormat.DEEPEP_LL  # 返回DeepEP低延迟模式格式


assert isinstance(DeepEPNormalDispatchOutput, DispatchOutput)  # 验证普通模式输出符合DispatchOutput协议
assert isinstance(DeepEPLLDispatchOutput, DispatchOutput)  # 验证低延迟模式输出符合DispatchOutput协议


class DeepEPNormalCombineInput(NamedTuple):  # DeepEP普通模式合并输入
    """DeepEP normal combine input."""  # DeepEP普通模式合并输入
    # DeepEP普通模式合并输入

    hidden_states: torch.Tensor  # 隐藏状态张量
    topk_ids: torch.Tensor  # TopK专家ID
    topk_weights: torch.Tensor  # TopK权重

    @property
    def format(self) -> CombineInputFormat:  # 输入格式属性
        return CombineInputFormat.DEEPEP_NORMAL  # 返回DeepEP普通模式格式


class DeepEPLLCombineInput(NamedTuple):  # DeepEP低延迟模式合并输入
    """DeepEP low latency combine input."""  # DeepEP低延迟合并输入
    # DeepEP低延迟合并输入

    hidden_states: torch.Tensor  # 隐藏状态张量
    topk_ids: torch.Tensor  # TopK专家ID
    topk_weights: torch.Tensor  # TopK权重

    @property
    def format(self) -> CombineInputFormat:  # 输入格式属性
        return CombineInputFormat.DEEPEP_LL  # 返回DeepEP低延迟模式格式


assert isinstance(DeepEPNormalCombineInput, CombineInput)  # 验证普通模式输入符合CombineInput协议
assert isinstance(DeepEPLLCombineInput, CombineInput)  # 验证低延迟模式输入符合CombineInput协议


class DeepEPDispatchMode(IntEnum):  # DeepEP分发模式枚举
    NORMAL = auto()  # 普通模式
    LOW_LATENCY = auto()  # 低延迟模式


class DeepEPBuffer:  # DeepEP缓冲区管理类，单例模式管理通信缓冲区
    _buffer = None  # 缓冲区单例
    _dispatch_mode: Optional[DeepEPDispatchMode] = None  # 当前分发模式
    _hidden_size: Optional[int] = None  # 隐藏层大小
    _num_max_dispatch_tokens_per_rank: Optional[int] = None  # 每个rank最大分发token数
    _num_experts: Optional[int] = None  # 专家数量

    @classmethod
    def get_deepep_buffer(  # 获取或创建DeepEP缓冲区（单例模式）
        cls,
        group: dist.ProcessGroup,  # 进程组
        hidden_size: int,  # 隐藏层大小
        param_bytes: int,  # 参数字节数
        deepep_mode: DeepEPMode,  # DeepEP模式
        num_max_dispatch_tokens_per_rank: int = -1,  # 每个rank最大分发token数
        num_experts: int = -1,  # 专家数量
    ):
        if cls._buffer is not None:  # 若缓冲区已存在
            return cls._buffer  # 直接返回已有缓冲区

        cls._hidden_size = hidden_size  # 保存隐藏层大小
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank  # 保存最大分发token数
        cls._num_experts = num_experts  # 保存专家数量

        num_nvl_bytes, num_rdma_bytes = 0, 0  # NVL和RDMA缓冲区大小初始化为0
        if deepep_mode.enable_normal():  # 若启用普通模式
            hidden_bytes = hidden_size * param_bytes  # 计算隐藏状态字节数
            for config in (  # 遍历分发和合并配置
                DeepEPConfig.get_instance().normal_dispatch_config  # 普通分发配置
                or Buffer.get_dispatch_config(group.size()),  # 或使用默认配置
                DeepEPConfig.get_instance().normal_combine_config  # 普通合并配置
                or Buffer.get_combine_config(group.size()),  # 或使用默认配置
            ):
                num_nvl_bytes = max(  # 取NVL缓冲区大小的最大值
                    config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),  # 获取NVL缓冲区大小提示
                    num_nvl_bytes,  # 当前最大值
                )
                num_rdma_bytes = max(  # 取RDMA缓冲区大小的最大值
                    config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),  # 获取RDMA缓冲区大小提示
                    num_rdma_bytes,  # 当前最大值
                )
        if deepep_mode.enable_low_latency():  # 若启用低延迟模式
            assert num_max_dispatch_tokens_per_rank != -1  # 必须指定最大分发token数
            assert num_experts != -1 and num_experts % group.size() == 0  # 专家数必须能被进程组大小整除
            num_rdma_bytes = max(  # 取RDMA缓冲区大小的最大值
                Buffer.get_low_latency_rdma_size_hint(  # 获取低延迟RDMA缓冲区大小提示
                    num_max_dispatch_tokens_per_rank,  # 最大分发token数
                    hidden_size,  # 隐藏层大小
                    group.size(),  # 进程组大小
                    num_experts,  # 专家数量
                ),
                num_rdma_bytes,  # 当前最大值
            )

        # We should calculate num_qps_per_rank consistently with DeepEP's test script logic:
        # 我们应该与DeepEP测试脚本的逻辑一致地计算num_qps_per_rank：
        if deepep_mode == DeepEPMode.NORMAL:  # 普通模式
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            # 参考: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            num_qps_per_rank = DeepEPConfig.get_instance().num_sms  # 使用配置的SM数量
        elif deepep_mode == DeepEPMode.LOW_LATENCY:  # 低延迟模式
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_low_latency.py#L176
            # 参考: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_low_latency.py#L176
            num_qps_per_rank = num_experts // group.size()  # 每个rank的QP数等于专家数除以进程数
        elif deepep_mode == DeepEPMode.AUTO:  # 自动模式
            # low-latency and normal mode all need run
            # 低延迟和普通模式都需要运行
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            # 参考: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            num_qps_per_rank = max(  # 取两种模式的QP数的最大值
                DeepEPConfig.get_instance().num_sms, num_experts // group.size()
            )
        else:  # 不支持的模式
            raise NotImplementedError  # 抛出未实现异常

        if not _is_npu:  # 非NPU环境
            total_num_sms = torch.cuda.get_device_properties(  # 获取GPU总SM数量
                device="cuda"
            ).multi_processor_count
            if (  # 检查SM配置是否过低
                (deepep_mode != DeepEPMode.LOW_LATENCY)  # 非低延迟模式
                and not is_tbo_enabled()  # TBO未启用
                and (DeepEPConfig.get_instance().num_sms < total_num_sms // 2)  # 配置的SM数不足总数一半
            ):
                logger.warning(  # 发出性能警告
                    f"Only use {DeepEPConfig.get_instance().num_sms} SMs for DeepEP communication. "
                    f"This may result in highly suboptimal performance. "
                    f"Consider using --deepep-config to change the behavior."
                )

        cls._buffer = Buffer(  # 创建DeepEP缓冲区
            group,  # 进程组
            num_nvl_bytes,  # NVL缓冲区大小
            num_rdma_bytes,  # RDMA缓冲区大小
            low_latency_mode=deepep_mode.enable_low_latency(),  # 是否启用低延迟模式
            num_qps_per_rank=num_qps_per_rank,  # 每个rank的QP数量
            # TODO can be false when unneeded # TODO: 不需要时可以设为false
            # TODO: 不需要时可以设为false
            allow_mnnvl=True,  # 允许MNNVL
        )
        return cls._buffer  # 返回缓冲区实例

    @classmethod
    def clean_buffer(cls):  # 清理低延迟模式缓冲区
        if not cls._buffer.low_latency_mode:  # 若非低延迟模式则无需清理
            return  # 直接返回
        cls._buffer.clean_low_latency_buffer(  # 清理低延迟缓冲区
            cls._num_max_dispatch_tokens_per_rank,  # 最大分发token数
            cls._hidden_size,  # 隐藏层大小
            cls._num_experts,  # 专家数量
        )

    @classmethod
    def set_dispatch_mode_as_normal(cls):  # 将分发模式设置为普通模式
        cls._dispatch_mode = DeepEPDispatchMode.NORMAL  # 更新分发模式

    @classmethod
    def set_dispatch_mode_as_low_latency(cls):  # 将分发模式设置为低延迟模式
        if cls._dispatch_mode == DeepEPDispatchMode.NORMAL:  # 若当前为普通模式
            cls.clean_buffer()  # 先清理缓冲区
        cls._dispatch_mode = DeepEPDispatchMode.LOW_LATENCY  # 更新分发模式

    @classmethod
    def set_dispatch_mode(cls, mode: DeepEPMode):  # 根据DeepEPMode设置分发模式
        if mode.is_low_latency():  # 低延迟模式
            cls.set_dispatch_mode_as_low_latency()  # 设置为低延迟
        elif mode.is_normal():  # 普通模式
            cls.set_dispatch_mode_as_normal()  # 设置为普通模式
        else:  # 不支持的模式
            raise Exception("unsupported mode")  # 抛出异常


class DeepEPConfig(BaseDispatcherConfig):  # DeepEP配置类，单例模式
    _instance = None  # 单例实例

    def __init__(self):  # 初始化DeepEP配置
        config_str = get_deepep_config()  # 获取DeepEP配置字符串
        if config_str:  # 若配置字符串非空
            config_parsed = load_json_config(config_str)  # 解析JSON配置
            if torch.distributed.get_rank() == 0:  # 仅主进程打印日志
                logger.info(f"Use DeepEP Config: {config_parsed}")  # 打印配置信息
            config_dispatch = config_parsed["normal_dispatch"]  # 获取普通分发配置
            config_combine = config_parsed["normal_combine"]  # 获取普通合并配置

            self.normal_dispatch_config = Config(**config_dispatch)  # 创建普通分发配置对象
            self.normal_combine_config = Config(**config_combine)  # 创建普通合并配置对象

            assert config_dispatch["num_sms"] == config_combine["num_sms"]  # 分发和合并的SM数必须一致
            self.num_sms = config_dispatch["num_sms"]  # 保存SM数量
        else:  # 配置字符串为空
            self.normal_dispatch_config = None  # 使用默认分发配置
            self.normal_combine_config = None  # 使用默认合并配置
            self.num_sms = Buffer.num_sms  # 使用Buffer默认SM数量

    @classmethod
    def get_instance(cls):  # 获取DeepEPConfig单例
        if cls._instance is None:  # 若单例未创建
            cls._instance = DeepEPConfig()  # 创建单例
        return cls._instance  # 返回单例


class _DeepEPDispatcherImplBase:  # DeepEP分发器实现基类，定义分发和合并的接口
    def __init__(  # 初始化DeepEP分发器实现
        self,
        group: torch.distributed.ProcessGroup,  # 进程组
        router_topk: int,  # 路由TopK值
        permute_fusion: bool,  # 是否启用排列融合
        num_experts: int,  # 专家总数
        num_local_experts: int,  # 本地专家数
        hidden_size: int,  # 隐藏层大小
        params_dtype: torch.dtype,  # 参数数据类型
        deepep_mode: DeepEPMode,  # DeepEP模式
    ):
        if not use_deepep:  # 若DeepEP不可用
            raise ImportError(  # 抛出导入错误
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

        self.group = group  # 保存进程组
        self.router_topk = router_topk  # 保存路由TopK值
        self.permute_fusion = permute_fusion  # 保存排列融合标志
        self.num_experts = num_experts  # 保存专家总数
        self.num_local_experts = num_local_experts  # 保存本地专家数
        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.params_dtype = params_dtype  # 保存参数数据类型
        self.deepep_mode = deepep_mode  # 保存DeepEP模式

        self.params_bytes = 2  # 参数字节数（BF16为2字节）
        # A large value will lead to large memory occupation, thus users should change it accordingly
        # 过大的值会导致大量内存占用，因此用户应根据实际情况调整
        self.num_max_dispatch_tokens_per_rank = (  # 每个rank的最大分发token数
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()  # 从环境变量获取
        )
        # DeepEP internode_ll dispatch uses FINISHED_SUM_TAG=1024
        # and the logic requires num-tokens-sent-from-one-rank-to-another-rank less than it
        # DeepEP节点间低延迟分发使用FINISHED_SUM_TAG=1024，
        # 逻辑要求从一个rank发送到另一个rank的token数小于该值
        assert self.num_max_dispatch_tokens_per_rank <= 1024  # 最大分发token数不能超过1024

        self.handle = None  # 通信句柄，用于dispatch和combine之间的同步

        self.quant_config: Optional[dict] = None  # 量化配置

        self.overlap_args: Optional[CombineOverlapArgs] = None  # 重叠参数
        self.meta_overlap_args: Optional[dict] = None  # 元数据重叠参数

        self.set_deepep_dispatcher_dtype()  # 设置DeepEP分发器数据类型

    def dispatch_a(  # 分发阶段A（前置处理）
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        topk_output: TopKOutput,  # TopK输出
    ):
        raise NotImplementedError  # 子类必须实现

    def dispatch_b(self, *args, **kwargs):  # 分发阶段B（通信操作）
        raise NotImplementedError  # 子类必须实现

    def combine_a(  # 合并阶段A（前置处理）
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK权重
    ):
        raise NotImplementedError  # 子类必须实现

    def combine_b(self, *args, **kwargs):  # 合并阶段B（通信操作）
        raise NotImplementedError  # 子类必须实现

    def _get_buffer(self):  # 获取DeepEP缓冲区
        raise NotImplementedError  # 子类必须实现

    def set_quant_config(self, quant_config: dict) -> None:  # 设置量化配置
        self.quant_config = quant_config  # 保存量化配置
        self.set_deepep_dispatcher_dtype()  # 更新数据类型设置

    def set_deepep_dispatcher_dtype(self) -> None:  # 设置DeepEP分发器输出数据类型
        self.deepep_output_dtype = get_deepep_output_dtype(self)  # 获取当前配置的输出数据类型

        # Configuration mapping for each dtype # 每种数据类型的配置映射
        # 每种数据类型的配置映射
        config_map = {
            DeepEPOutputDtype.BF16: {  # BF16格式
                "use_fp8": False,  # 不使用FP8
                "use_nvfp4": False,  # 不使用NVFP4
            },
            DeepEPOutputDtype.FP8: {  # FP8格式
                "use_fp8": True,  # 使用FP8
                "use_nvfp4": False,  # 不使用NVFP4
            },
            # Needed for Ascend A2/A3 NPU case,
            # despite the use_fp8 flag,
            # quantization will be performed in int8
            # 昇腾A2/A3 NPU情况下需要，
            # 尽管设置了use_fp8标志，
            # 量化将以int8执行
            DeepEPOutputDtype.INT8: {  # INT8格式（NPU专用）
                "use_fp8": True,  # 使用FP8标志（实际量化为int8）
                "use_nvfp4": False,  # 不使用NVFP4
            },
            DeepEPOutputDtype.NVFP4: {  # NVFP4格式
                "use_fp8": False,  # 不使用FP8
                "use_nvfp4": True,  # 使用NVFP4
            },
        }

        # Validate and apply hardware-specific adjustments # 验证并应用硬件特定调整
        # 验证并应用硬件特定调整
        self._validate_and_adjust_dtype()

        # Apply configuration # 应用配置
        # 应用配置
        config = config_map[self.deepep_output_dtype]  # 获取对应数据类型的配置
        self.use_fp8 = config["use_fp8"]  # 保存FP8使用标志
        self.use_nvfp4 = config["use_nvfp4"]  # 保存NVFP4使用标志

        # Handle environment variables # 处理环境变量
        # 处理环境变量
        if _is_npu:  # NPU环境
            self._update_int8_quant_env()  # 更新INT8量化环境变量

    def _validate_and_adjust_dtype(self) -> None:  # 验证数据类型与硬件兼容性并调整
        """Validate dtype against hardware and adjust if necessary."""  # 根据硬件验证数据类型并在必要时调整
        # 根据硬件验证数据类型并在必要时调整
        if _is_npu:  # NPU环境
            if self.deepep_output_dtype == DeepEPOutputDtype.FP8:  # NPU不支持FP8
                logger.warning_once(  # 发出警告
                    "Ascend A2/A3 NPU does not support fp8 "
                    "deepep_dispatcher_output_dtype, switching to int8..."
                )
                self.deepep_output_dtype = DeepEPOutputDtype.INT8  # 切换为INT8
            elif self.deepep_output_dtype == DeepEPOutputDtype.NVFP4:  # NPU不支持NVFP4
                raise RuntimeError(  # 抛出运行时错误
                    "Ascend A2/A3 NPU does not support nvfp4 deepep_dispatcher_output_dtype."
                )
        else:  # GPU环境
            if self.deepep_output_dtype == DeepEPOutputDtype.INT8:  # GPU不支持INT8
                logger.warning_once(  # 发出警告
                    "GPU does not support int8 "
                    "deepep_dispatcher_output_dtype, switching to fp8..."
                )
                self.deepep_output_dtype = DeepEPOutputDtype.FP8  # 切换为FP8
            # NVFP4 is supported on GPU, no adjustment needed # GPU支持NVFP4，无需调整
            # GPU支持NVFP4，无需调整

    def _update_int8_quant_env(self) -> None:  # 更新INT8量化环境变量
        """TODO adapt different quantization schemes for base model and draft model on NPU"""  # TODO: 适配NPU上基础模型和草稿模型的不同量化方案
        # TODO: 适配NPU上基础模型和草稿模型的不同量化方案
        pass  # 暂未实现

    def set_overlap_args(  # 设置重叠参数
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict  # 合并重叠参数和元数据重叠参数
    ) -> None:
        self.overlap_args = combine_overlap_args  # 保存合并重叠参数
        self.meta_overlap_args = meta_overlap_args  # 保存元数据重叠参数

    def clear_overlap_args(self) -> None:  # 清除重叠参数
        self.overlap_args = None  # 置空合并重叠参数
        self.meta_overlap_args = None  # 置空元数据重叠参数


class _DeepEPDispatcherImplNormal(_DeepEPDispatcherImplBase):  # DeepEP普通模式分发器实现
    def __init__(self, async_finish: bool, **kwargs):  # 初始化普通模式分发器
        super().__init__(**kwargs)  # 调用父类初始化

        self.async_finish = async_finish  # 是否异步完成通信
        self.src2dst = None  # 源到目标的映射
        self.quant_config = {}  # 量化配置字典

    def dispatch_a(  # 分发阶段A：预处理隐藏状态和TopK输出
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        topk_output: TopKOutput,  # TopK输出
    ):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids  # 解包TopK输出
        topk_ids = topk_ids.to(torch.int64)  # 转换TopK ID为int64类型
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8:  # 启用JIT DeepGEMM且使用FP8
            # TODO hard code 128 block quant,use fp8 communication # TODO: 硬编码128块量化，使用FP8通信
            # TODO: 硬编码128块量化，使用FP8通信
            hidden_states = sglang_per_token_group_quant_fp8(  # 对隐藏状态进行FP8逐token组量化
                hidden_states,
                128,  # 块大小为128
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,  # 列主序缩放
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,  # TMA对齐缩放
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,  # UE8M0缩放
            )
        previous_event = Buffer.capture() if self.async_finish else None  # 异步模式下捕获当前流事件
        return hidden_states, topk_ids, topk_weights, previous_event  # 返回预处理结果

    def dispatch_b(self, hidden_states, topk_ids, topk_weights, previous_event):  # 分发阶段B：执行全对全通信
        (
            hidden_states,  # 接收到的隐藏状态
            topk_ids,  # 接收到的TopK ID
            topk_weights,  # 接收到的TopK权重
            num_recv_tokens_per_expert,  # 每个专家接收的token数
            event,  # 通信完成事件
        ) = self._dispatch_core(hidden_states, topk_ids, topk_weights, previous_event)  # 执行分发核心逻辑
        event.current_stream_wait() if self.async_finish else ()  # 异步模式下等待通信完成

        if isinstance(hidden_states, tuple):  # 若隐藏状态为元组（量化格式）
            hidden_states, hidden_states_scale = hidden_states  # 解包隐藏状态和缩放因子
        else:  # 非量化格式
            hidden_states_scale = None  # 无缩放因子

        return DeepEPNormalDispatchOutput(  # 返回普通模式分发输出
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        )

    def _dispatch_core(  # 分发核心逻辑：布局计算和全对全分发
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # 输入张量或量化张量对
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK权重
        previous_event,  # 前一个通信事件
    ):
        buffer = self._get_buffer()  # 获取DeepEP缓冲区
        (
            num_tokens_per_rank,  # 每个rank发送的token数
            num_tokens_per_rdma_rank,  # 每个RDMA rank发送的token数
            num_tokens_per_expert,  # 每个专家的token数
            is_token_in_rank,  # token是否属于某个rank的标记
            previous_event,  # 更新后的事件
        ) = buffer.get_dispatch_layout(  # 获取分发布局
            topk_ids,
            self.num_experts,  # 专家总数
            previous_event=previous_event,  # 前一个事件
            async_finish=self.async_finish,  # 是否异步完成
            allocate_on_comm_stream=previous_event is not None,  # 是否在通信流上分配
        )
        # FIXME: `handle` should be transmitted with tokens from dispatch to combine.
        # However, doing this would incur an unknown synchronization error, but keeping
        # `handle` as a member variable works.
        # FIXME: `handle`应该随token从分发传到合并。
        # 但这样做会导致未知的同步错误，将`handle`保持为成员变量可以正常工作。

        _deepep_precompile_tp_barrier()  # 预编译阶段屏障同步
        (
            recv_x,  # 接收到的输入
            recv_topk_ids,  # 接收到的TopK ID
            recv_topk_weights,  # 接收到的TopK权重
            num_recv_tokens_per_expert,  # 每个专家接收的token数
            self.handle,  # 通信句柄
            event,  # 通信完成事件
        ) = buffer.dispatch(  # 执行全对全分发
            x,  # 输入数据
            topk_idx=topk_ids,  # TopK索引
            topk_weights=topk_weights,  # TopK权重
            num_tokens_per_rank=num_tokens_per_rank,  # 每个rank的token数
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,  # 每个RDMA rank的token数
            is_token_in_rank=is_token_in_rank,  # token在rank中的标记
            num_tokens_per_expert=num_tokens_per_expert,  # 每个专家的token数
            previous_event=previous_event,  # 前一个事件
            async_finish=self.async_finish,  # 是否异步完成
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,  # 是否在通信流上分配
            expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,  # 专家对齐大小
            config=DeepEPConfig.get_instance().normal_dispatch_config,  # 分发配置
        )
        get_global_expert_distribution_recorder().on_deepep_dispatch_normal(  # 记录专家分布信息
            num_recv_tokens_per_expert,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

        return (  # 返回分发结果
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            event,
        )

    def combine_a(  # 合并阶段A：准备合并输入
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK权重
    ):

        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM or _use_aiter or _is_npu:  # 使用JIT DeepGEMM/AITER/NPU
            output = hidden_states  # 直接使用隐藏状态作为输出
        else:  # 其他情况
            raise NotImplementedError()  # triton runner was supported but it's temporarily disabled # triton runner曾经支持但暂时被禁用
            # triton runner曾经支持但暂时被禁用

        previous_event = Buffer.capture() if self.async_finish else None  # 异步模式下捕获事件
        return output, previous_event  # 返回输出和事件

    def combine_b(self, output, previous_event):  # 合并阶段B：执行全对全合并
        hidden_states, event = self._combine_core(output, previous_event)  # 执行合并核心逻辑
        event.current_stream_wait() if self.async_finish else ()  # 异步模式下等待完成
        self.handle = None  # 清空通信句柄
        self.src2dst = None  # 清空源到目标映射
        return hidden_states  # 返回合并后的隐藏状态

    def _combine_core(self, x: torch.Tensor, previous_event):  # 合并核心逻辑：全对全合并
        buffer = self._get_buffer()  # 获取DeepEP缓冲区
        _deepep_precompile_tp_barrier()  # 预编译阶段屏障同步
        combined_x, _, event = buffer.combine(  # 执行全对全合并
            x,  # 输入张量
            self.handle,  # 通信句柄
            async_finish=self.async_finish,  # 是否异步完成
            previous_event=previous_event,  # 前一个事件
            allocate_on_comm_stream=previous_event is not None,  # 是否在通信流上分配
            config=DeepEPConfig.get_instance().normal_combine_config,  # 合并配置
        )
        return combined_x, event  # 返回合并结果和事件

    def _get_buffer(self):  # 获取普通模式的DeepEP缓冲区
        DeepEPBuffer.set_dispatch_mode_as_normal()  # 设置分发模式为普通模式

        return DeepEPBuffer.get_deepep_buffer(  # 获取缓冲区
            self.group,  # 进程组
            self.hidden_size,  # 隐藏层大小
            self.params_bytes,  # 参数字节数
            self.deepep_mode,  # DeepEP模式
            self.num_max_dispatch_tokens_per_rank,  # 最大分发token数
            self.num_experts,  # 专家数量
        )


class _DeepEPDispatcherImplLowLatency(_DeepEPDispatcherImplBase):  # DeepEP低延迟模式分发器实现
    def __init__(self, return_recv_hook: bool, **kwargs):  # 初始化低延迟模式分发器
        super().__init__(**kwargs)  # 调用父类初始化

        """
        num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
        https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
        """  # num_max_dispatch_tokens_per_rank: 解码引擎中的实际批次大小应小于256
        # num_max_dispatch_tokens_per_rank: 解码引擎中的实际批次大小应小于256
        self.return_recv_hook = return_recv_hook  # 是否返回接收钩子而非等待事件
        self.device_module = torch.get_device_module()  # 获取设备模块
        self.quant_config = {}  # 量化配置字典

    def dispatch_a(  # 分发阶段A：低延迟模式预处理
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        topk_output: TopKOutput,  # TopK输出
    ):
        buffer = self._get_buffer()  # 获取DeepEP缓冲区
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids  # 解包TopK输出
        topk_ids = topk_ids.to(torch.int64)  # 转换TopK ID为int64类型
        expected_m = (  # 计算预期token数
            hidden_states.shape[0] * buffer.group_size * topk_ids.shape[1]  # 总token数 = 输入token * 组大小 * TopK
            + self.num_experts  # 加上专家数（对齐）
        ) // self.num_experts  # 除以专家数
        hidden_states, masked_m, event, hook = self._dispatch_core(  # 执行分发核心逻辑
            hidden_states,
            topk_ids,
        )
        return (  # 返回分发中间结果
            hidden_states,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
            event,
            hook,
        )

    def dispatch_b(  # 分发阶段B：低延迟模式通信和后处理
        self,
        hidden_states,  # 隐藏状态
        topk_ids,  # TopK ID
        topk_weights,  # TopK权重
        masked_m,  # 掩码矩阵
        expected_m,  # 预期token数
        event,  # 通信事件
        hook,  # 接收钩子
    ):
        hook() if self.return_recv_hook else event.current_stream_wait()  # 使用钩子或等待事件完成

        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(  # 记录低延迟模式专家分布
            masked_m
        )

        if isinstance(hidden_states, tuple):  # 若隐藏状态为元组（量化格式）
            hidden_states, hidden_states_scale = hidden_states  # 解包隐藏状态和缩放因子
        else:  # 非量化格式
            hidden_states_scale = None  # 无缩放因子

        deepep_output = DeepEPLLDispatchOutput(  # 创建低延迟模式分发输出
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
        )
        return deepep_output  # 返回分发输出

    def _dispatch_core(  # 低延迟分发核心逻辑
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
    ):
        input_global_scale = self.quant_config.get("input_global_scale", None)  # 获取全局缩放因子

        # round_scale / use_ue8m0 are FP8-DeepGEMM specific; they cause DeepEP
        # to return int32-packed UE8M0 scales that don't feed the flashinfer
        # cutedsl kernel.
        # round_scale / use_ue8m0 是FP8-DeepGEMM特有的选项；它们使DeepEP
        # 返回int32打包的UE8M0缩放因子，这些不会传入flashinfer的cutedsl内核。
        fp8_deepgemm_scale_opts = (  # FP8 DeepGEMM缩放选项
            dict(
                round_scale=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM  # 是否启用缩放舍入
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,  # 仅Blackwell架构
                use_ue8m0=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM  # 是否使用UE8M0缩放
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,  # 仅Blackwell架构
            )
            if self.use_fp8  # 仅在使用FP8时
            else dict()  # 否则为空字典
        )

        buffer = self._get_buffer()  # 获取DeepEP缓冲区
        _deepep_precompile_tp_barrier()  # 预编译阶段屏障同步
        packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (  # 低延迟分发结果
            buffer.low_latency_dispatch(  # 执行低延迟分发
                hidden_states,  # 输入隐藏状态
                topk_ids,  # TopK专家ID
                self.num_max_dispatch_tokens_per_rank,  # 最大分发token数
                self.num_experts,  # 专家数量
                use_fp8=self.use_fp8,  # 是否使用FP8
                **(dict(use_nvfp4=True) if self.use_nvfp4 else dict()),  # NVFP4选项
                **(
                    dict(x_global_scale=input_global_scale)  # 全局缩放因子
                    if input_global_scale is not None  # 若缩放因子存在
                    else dict()  # 否则为空
                ),
                async_finish=not self.return_recv_hook,  # 异步完成标志
                return_recv_hook=self.return_recv_hook,  # 返回接收钩子标志
                **fp8_deepgemm_scale_opts,  # FP8 DeepGEMM缩放选项
            )
        )
        return packed_recv_hidden, self.packed_recv_count, event, hook  # 返回分发结果

    def combine_a(  # 合并阶段A：低延迟模式准备合并
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK权重
    ):
        hidden_states, event, hook = self._combine_core(  # 执行合并核心逻辑
            hidden_states,
            topk_ids,
            topk_weights,
        )
        return hidden_states, event, hook  # 返回合并中间结果

    def combine_b(self, hidden_states, event, hook):  # 合并阶段B：低延迟模式通信和后处理
        overlap_args = self.overlap_args  # 获取重叠参数
        if overlap_args is not None:  # 若存在重叠参数
            overlap_args.stream.wait_stream(self.device_module.current_stream())  # 重叠流等待当前流完成

        hook() if self.return_recv_hook else event.current_stream_wait()  # 使用钩子或等待事件完成

        if overlap_args is not None:  # 若存在重叠参数
            self.device_module.current_stream().wait_stream(overlap_args.stream)  # 当前流等待重叠流完成

        return hidden_states  # 返回合并后的隐藏状态

    def _combine_core(  # 低延迟合并核心逻辑
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK权重
    ):
        buffer = self._get_buffer()  # 获取DeepEP缓冲区
        overlap_args = self.overlap_args  # 获取重叠参数
        meta_overlap_args = self.meta_overlap_args  # 获取元数据重叠参数

        ctx = nullcontext()  # 默认空上下文
        if overlap_args is not None:  # 若存在重叠参数
            overlap_args.stream.wait_event(overlap_args.wait_event)  # 重叠流等待事件
            ctx = torch.cuda.stream(overlap_args.stream)  # 创建CUDA流上下文

            if is_blackwell():  # Blackwell架构
                overlap_args_dict = dict(  # Blackwell重叠参数
                    overlap=overlap_args.overlap,  # 重叠张量
                    src_signals=overlap_args.signal,  # 源信号
                    src_signal_expect_value=overlap_args.threshold,  # 信号期望值
                )
            else:  # 非Blackwell架构
                overlap_args_dict = dict(  # 非Blackwell重叠参数
                    overlap=overlap_args.overlap,  # 重叠张量
                    packed_recv_count=self.packed_recv_count,  # 打包的接收计数
                    comp_signal=overlap_args.signal,  # 计算信号
                    block_m=meta_overlap_args["block_m"],  # 块M大小
                    threshold=meta_overlap_args["threshold"],  # 阈值
                    num_sms=overlap_args.num_sms,  # SM数量
                )
        else:  # 无重叠参数
            overlap_args_dict = {}  # 空字典

        with ctx:  # 在指定上下文中执行
            _deepep_precompile_tp_barrier()  # 预编译阶段屏障同步
            combined_hidden_states, event, hook = buffer.low_latency_combine(  # 执行低延迟合并
                x=hidden_states,  # 输入隐藏状态
                topk_idx=topk_ids,  # TopK索引
                topk_weights=topk_weights,  # TopK权重
                handle=self.handle,  # 通信句柄
                async_finish=not self.return_recv_hook,  # 异步完成标志
                return_recv_hook=self.return_recv_hook,  # 返回接收钩子标志
                **overlap_args_dict,  # 重叠参数
            )

        self.packed_recv_count = self.handle = None  # 清空打包接收计数和通信句柄
        return combined_hidden_states, event, hook  # 返回合并结果

    def _get_buffer(self):  # 获取低延迟模式的DeepEP缓冲区
        DeepEPBuffer.set_dispatch_mode_as_low_latency()  # 设置分发模式为低延迟模式
        return DeepEPBuffer.get_deepep_buffer(  # 获取缓冲区
            self.group,  # 进程组
            self.hidden_size,  # 隐藏层大小
            self.params_bytes,  # 参数字节数
            self.deepep_mode,  # DeepEP模式
            self.num_max_dispatch_tokens_per_rank,  # 最大分发token数
            self.num_experts,  # 专家数量
        )


@dataclass
class _Stage(Enum):  # 分发器执行阶段枚举
    INITIAL = auto()  # 初始状态
    AFTER_DISPATCH_A = auto()  # 分发阶段A完成后
    AFTER_DISPATCH_B = auto()  # 分发阶段B完成后
    AFTER_COMBINE_A = auto()  # 合并阶段A完成后


class DeepEPDispatcher(BaseDispatcher):  # DeepEP分发器，继承基类，支持普通和低延迟两种模式
    def __init__(  # 初始化DeepEP分发器
        self,
        group: torch.distributed.ProcessGroup,  # 进程组
        router_topk: int,  # 路由TopK值
        permute_fusion: bool = False,  # 是否启用排列融合
        num_experts: int = None,  # 专家总数
        num_local_experts: int = None,  # 本地专家数
        hidden_size: int = None,  # 隐藏层大小
        params_dtype: torch.dtype = None,  # 参数数据类型
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,  # DeepEP模式，默认自动
        async_finish: bool = False,  # 是否异步完成（普通模式用）
        return_recv_hook: bool = False,  # 是否返回接收钩子（低延迟模式用）
    ):
        super().__init__()  # 调用父类初始化

        self.deepep_mode = deepep_mode  # 保存DeepEP模式

        common_kwargs = dict(  # 两种模式共用的参数
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        if self.deepep_mode.enable_low_latency():  # 若启用低延迟模式
            self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(  # 创建低延迟分发器实现
                return_recv_hook=return_recv_hook,
                **common_kwargs,
            )
        if self.deepep_mode.enable_normal():  # 若启用普通模式
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(  # 创建普通分发器实现
                async_finish=async_finish,
                **common_kwargs,
            )

        self._stage = _Stage.INITIAL  # 初始化执行阶段
        self._deepep_dispatch_hooks = DeepEPPDispatchHooks()  # DeepEP分发钩子

        # DeepEP/Mooncake/Nixl mark invalid topk slots with -1; the AITER
        # pre_permute reroutes them to a sink slot at index num_local_experts,
        # which is masked off here.
        # DeepEP/Mooncake/Nixl用-1标记无效的topk槽位；AITER的pre_permute
        # 将它们重新路由到索引num_local_experts的汇聚槽位，在此处被屏蔽。
        self.expert_mask_gpu = None  # 专家掩码（GPU上）
        if _use_aiter and num_local_experts is not None:  # 使用AITER且指定了本地专家数
            expert_mask = torch.zeros(  # 创建专家掩码张量
                num_local_experts + 1,  # 大小为本地专家数+1（含汇聚槽位）
                device=torch.cuda.current_device(),  # 在当前GPU设备上
                dtype=torch.int,  # 整型
            )
            expert_mask[:-1] = 1  # 有效专家位置设为1，汇聚槽位保持0
            self.expert_mask_gpu = expert_mask  # 保存专家掩码

    def dispatch(  # 完整分发流程：dispatch_a -> 钩子 -> dispatch_b
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        topk_output: TopKOutput,  # TopK输出
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)  # 执行分发阶段A
        if self._deepep_dispatch_hooks is not None:  # 若存在DeepEP分发钩子
            self._deepep_dispatch_hooks(self)  # 执行钩子
        ret = self.dispatch_b()  # 执行分发阶段B
        return ret  # 返回分发输出

    def dispatch_a(  # 分发阶段A：预处理
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        topk_output: TopKOutput,  # TopK输出
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)  # 更新执行阶段
        inner_state = self._get_impl().dispatch_a(  # 调用内部实现的dispatch_a
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state  # 保存分发中间状态

    def dispatch_b(self):  # 分发阶段B：通信操作
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)  # 更新执行阶段
        inner_state = self._dispatch_intermediate_state  # 获取分发中间状态
        del self._dispatch_intermediate_state  # 删除中间状态引用
        return self._get_impl().dispatch_b(*inner_state)  # 调用内部实现的dispatch_b

    def combine(  # 完整合并流程：combine_a -> combine_b
        self,
        combine_input: CombineInput,  # 合并输入
    ) -> torch.Tensor:
        self.combine_a(combine_input)  # 执行合并阶段A
        ret = self.combine_b()  # 执行合并阶段B
        return ret  # 返回合并结果

    def combine_a(  # 合并阶段A：预处理
        self,
        combine_input: CombineInput,  # 合并输入
    ):
        hidden_states, topk_ids, topk_weights = combine_input  # 解包合并输入
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)  # 更新执行阶段
        inner_state = self._get_impl().combine_a(  # 调用内部实现的combine_a
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state  # 保存合并中间状态

    def combine_b(self):  # 合并阶段B：通信操作
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)  # 更新执行阶段回到初始
        inner_state = self._combine_intermediate_state  # 获取合并中间状态
        del self._combine_intermediate_state  # 删除中间状态引用
        return self._get_impl().combine_b(*inner_state)  # 调用内部实现的combine_b

    def _get_impl(self) -> _DeepEPDispatcherImplBase:  # 根据当前模式获取对应的分发器实现
        is_extend_in_batch = get_is_extend_in_batch()  # 获取是否在批次中扩展
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)  # 解析实际使用的DeepEP模式
        if resolved_deepep_mode == DeepEPMode.NORMAL:  # 普通模式
            return self._normal_dispatcher  # 返回普通模式分发器
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:  # 低延迟模式
            return self._low_latency_dispatcher  # 返回低延迟模式分发器
        else:  # 无效模式
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")  # 抛出值错误

    def _update_stage(self, old_stage, new_stage):  # 更新执行阶段（带断言检查）
        assert self._stage == old_stage  # 确保当前阶段与预期一致
        self._stage = new_stage  # 更新到新阶段

    def set_quant_config(self, quant_config: dict):  # 设置量化配置
        super().set_quant_config(quant_config)  # 调用父类方法
        if self.deepep_mode.enable_low_latency():  # 若启用低延迟模式
            self._low_latency_dispatcher.set_quant_config(quant_config)  # 设置低延迟分发器量化配置
        if self.deepep_mode.enable_normal():  # 若启用普通模式
            self._normal_dispatcher.set_quant_config(quant_config)  # 设置普通分发器量化配置

    def set_overlap_args(  # 设置重叠参数
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict  # 合并重叠参数和元数据重叠参数
    ):
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)  # 调用父类方法
        if self.deepep_mode.enable_low_latency():  # 若启用低延迟模式
            self._low_latency_dispatcher.set_overlap_args(  # 设置低延迟分发器重叠参数
                combine_overlap_args, meta_overlap_args
            )
        if self.deepep_mode.enable_normal():  # 若启用普通模式
            self._normal_dispatcher.set_overlap_args(  # 设置普通分发器重叠参数
                combine_overlap_args, meta_overlap_args
            )

    def clear_overlap_args(self):  # 清除重叠参数
        super().clear_overlap_args()  # 调用父类方法
        if self.deepep_mode.enable_low_latency():  # 若启用低延迟模式
            self._low_latency_dispatcher.clear_overlap_args()  # 清除低延迟分发器重叠参数
        if self.deepep_mode.enable_normal():  # 若启用普通模式
            self._normal_dispatcher.clear_overlap_args()  # 清除普通分发器重叠参数

    def register_deepep_dispatch_hook(self, hook):  # 注册DeepEP分发钩子
        return self._deepep_dispatch_hooks.register_hook(hook)  # 在dispatch_a和dispatch_b之间执行
