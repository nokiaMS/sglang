# 加载快照模块
# 本模块实现了调度器负载指标的发布功能，用于DP（数据并行）负载均衡和/v1/loads API。
# 支持两种传输后端：SHM模式（共享内存，单节点默认）和ZMQ模式
# （多节点DP注意力，通过网络传输）。ZMQ模式下，节点0上的读取器
# 从zmq接收快照并写入本地SHM文件，其他读取器从SHM读取。

"""Load snapshot: publish scheduler load metrics for DP balancing and /v1/loads.
加载快照：发布调度器负载指标，用于DP负载均衡和/v1/loads。

Architecture
架构
------------

Each scheduler periodically publishes a ``LoadSnapshot`` containing its
current load metrics (running reqs, tokens, throughput, ...).  Two
transport backends are supported:
每个调度器周期性发布一个``LoadSnapshot``，包含其当前负载指标
（运行中的请求、Token数、吞吐量等）。支持两种传输后端：

**SHM mode** (single-node, default)::
**SHM模式**（单节点，默认）::

    Scheduler  ──ShmLoadSnapshotWriter──▶  /dev/shm mmap file
                                                ▲
    TokenizerManager  ──ShmLoadSnapshotReader───┘  (for /v1/loads)
    DataParallelController  ──ShmLoadSnapshotReader─┘  (for dispatch)
    调度器  ──ShmLoadSnapshotWriter──▶  /dev/shm mmap 文件
                                             ▲
    TokenizerManager  ──ShmLoadSnapshotReader───┘  （用于/v1/loads）
    DataParallelController  ──ShmLoadSnapshotReader─┘  （用于调度分发）

**ZMQ mode** (multi-node DP attention, or ``SGLANG_LOAD_SNAPSHOT_USE_ZMQ=1``)::
**ZMQ模式**（多节点DP注意力，或``SGLANG_LOAD_SNAPSHOT_USE_ZMQ=1``）::

    Scheduler (any node)  ──ZmqLoadSnapshotWriter (PUSH)──▶  network
                                                                │
    ZmqShmLoadSnapshotReader (PULL, node 0)  ◀─────────────────┘
        │  drains zmq, writes to SHM
        ▼
    /dev/shm mmap file (node 0)
        ▲
    TokenizerManager / DataParallelController  ──ShmLoadSnapshotReader──┘
    调度器（任意节点）  ──ZmqLoadSnapshotWriter (PUSH)──▶  网络
                                                              │
    ZmqShmLoadSnapshotReader (PULL, 节点0)  ◀──────────────────┘
        │  排空zmq，写入SHM
        ▼
    /dev/shm mmap 文件（节点0）
        ▲
    TokenizerManager / DataParallelController  ──ShmLoadSnapshotReader──┘

Shared memory does not work across nodes, so multi-node DP attention
requires the ZMQ transport.  The ``ZmqShmLoadSnapshotReader`` on node 0
receives snapshots from all schedulers via zmq PUSH/PULL and writes them
into the local SHM file.  All readers (tokenizer, dp_controller) on
node 0 then read from SHM.
共享内存不能跨节点工作，因此多节点DP注意力需要ZMQ传输。
节点0上的``ZmqShmLoadSnapshotReader``通过zmq PUSH/PULL从所有调度器
接收快照并写入本地SHM文件。节点0上的所有读取器（tokenizer、dp_controller）
然后从SHM读取。

``zmq_reader_owner()`` decides which process on node 0 binds the zmq
PULL socket (only one can bind); the other reads plain SHM.
``zmq_reader_owner()``决定节点0上哪个进程绑定zmq PULL套接字
（只能有一个绑定）；另一个读取普通SHM。
"""

from __future__ import annotations  # 启用延迟注解评估 # 启用延迟注解评估

import fcntl  # 导入文件锁定模块 # 导入文件锁定模块
import hashlib  # 导入哈希模块 # 导入哈希模块
import logging  # 导入日志模块 # 导入日志模块
import mmap  # 导入内存映射模块 # 导入内存映射模块
import os  # 导入操作系统模块 # 导入操作系统模块
import struct  # 导入结构体模块 # 导入结构体模块
from contextlib import contextmanager  # 导入上下文管理器装饰器 # 导入上下文管理器装饰器
from typing import TYPE_CHECKING, Optional  # 导入类型注解 # 导入类型注解

import msgspec  # 导入msgspec序列化库 # 导入msgspec序列化库
import msgspec.msgpack  # 导入msgpack编解码器 # 导入msgpack编解码器
import msgspec.structs  # 导入msgspec结构体工具 # 导入msgspec结构体工具

from sglang.srt.environ import envs  # 导入环境变量配置 # 导入环境变量配置

if TYPE_CHECKING:  # 类型检查时导入 # 类型检查时导入
    from sglang.srt.managers.io_struct import GetLoadsReqOutput  # 导入获取负载请求输出类型 # 导入获取负载请求输出类型

logger = logging.getLogger(__name__)  # 创建日志记录器 # 创建日志记录器

# ---------------------------------------------------------------------------
# Helpers
# 辅助工具
# ---------------------------------------------------------------------------

DISAGG_MODE_TO_INT = {"null": 0, "prefill": 1, "decode": 2}  # 分离模式到整数的映射 # 分离模式到整数的映射
INT_TO_DISAGG_MODE = {v: k for k, v in DISAGG_MODE_TO_INT.items()}  # 整数到分离模式的反向映射 # 整数到分离模式的反向映射


def _native(v):  # 将numpy标量转换为Python原生类型 # 将numpy标量转换为Python原生类型
    """Coerce numpy scalars to Python int/float for msgpack encoding.
    将numpy标量强制转换为Python int/float，用于msgpack编码。"""
    if hasattr(v, "item"):  # 如果有item方法（numpy标量） # 如果有item方法（numpy标量）
        return v.item()  # 调用item方法转换为Python原生类型 # 调用item方法转换为Python原生类型
    return v  # 否则直接返回 # 否则直接返回


def should_use_zmq(server_args) -> bool:  # 判断是否应使用zmq传输 # 判断是否应使用zmq传输
    """Whether to use zmq PUSH/PULL instead of shared memory for load snapshots.
    是否使用zmq PUSH/PULL代替共享内存进行负载快照传输。

    Shared memory (mmap) only works within a single node.  When schedulers
    run on multiple nodes (multi-node DP attention), they cannot write to
    the SHM file on node 0, so we fall back to zmq transport.  The env var
    ``SGLANG_LOAD_SNAPSHOT_USE_ZMQ`` forces zmq mode for testing.
    共享内存（mmap）仅在同一节点内工作。当调度器运行在多个节点上
    （多节点DP注意力）时，它们无法写入节点0上的SHM文件，
    因此回退到zmq传输。环境变量``SGLANG_LOAD_SNAPSHOT_USE_ZMQ``
    强制启用zmq模式用于测试。
    """
    return (  # 返回判断结果 # 返回判断结果
        server_args.enable_dp_attention and server_args.nnodes > 1
    ) or envs.SGLANG_LOAD_SNAPSHOT_USE_ZMQ.get()  # 多节点DP注意力或环境变量强制启用 # 多节点DP注意力或环境变量强制启用


_LOAD_AWARE_METHODS = frozenset({"total_requests", "total_tokens"})  # 负载感知方法集合 # 负载感知方法集合


def zmq_reader_owner(server_args, caller: str) -> bool:  # 判断哪个进程拥有zmq PULL套接字 # 判断哪个进程拥有zmq PULL套接字
    """Decide which process owns the zmq PULL socket.
    决定哪个进程拥有zmq PULL套接字。

    Exactly one of ``"dp_controller"`` or ``"tokenizer"`` must return True
    when zmq mode is active.  The owner polls zmq -> SHM; the other reads SHM.
    当zmq模式激活时，``"dp_controller"``或``"tokenizer"``中恰好一个必须
    返回True。拥有者从zmq轮询到SHM；另一个读取SHM。

    Rules:
    规则：
      - Non-zero node_rank: no tokenizer, dp_controller only launches
        schedulers and waits -> nobody owns it.
      - 非零node_rank：没有tokenizer，dp_controller仅启动调度器并等待
        -> 没有人拥有。
      - dp_size == 1: no dp_controller exists -> tokenizer owns it.
      - dp_size == 1：没有dp_controller存在 -> tokenizer拥有。
      - dp_size > 1, load-aware method: dp_controller polls on every
        dispatch via refresh_load_budget() -> dp_controller owns it.
      - dp_size > 1，负载感知方法：dp_controller在每次调度时通过
        refresh_load_budget()轮询 -> dp_controller拥有。
      - dp_size > 1, round-robin / other: dp_controller never reads
        load data -> tokenizer owns it (polls on /v1/loads calls).
      - dp_size > 1，轮询/其他：dp_controller从不读取负载数据
        -> tokenizer拥有（在/v1/loads调用时轮询）。
    """
    if not should_use_zmq(server_args):  # 如果不使用zmq # 如果不使用zmq
        return False  # 返回False # 返回False
    if server_args.node_rank != 0:  # 如果不是节点0 # 如果不是节点0
        return False  # 返回False # 返回False
    if server_args.dp_size == 1:  # 如果DP大小为1 # 如果DP大小为1
        return caller == "tokenizer"  # tokenizer拥有 # tokenizer拥有
    if server_args.load_balance_method.lower() in _LOAD_AWARE_METHODS:  # 如果是负载感知方法 # 如果是负载感知方法
        return caller == "dp_controller"  # dp_controller拥有 # dp_controller拥有
    return caller == "tokenizer"  # 默认tokenizer拥有 # 默认tokenizer拥有


# ---------------------------------------------------------------------------
# LoadSnapshot data class
# LoadSnapshot数据类
# ---------------------------------------------------------------------------

CORE_METRIC_FIELDS = (  # 核心指标字段 # 核心指标字段
    "timestamp",
    "dp_rank",
    "num_running_reqs",
    "num_waiting_reqs",
    "num_used_tokens",
    "num_total_tokens",
    "max_total_num_tokens",
    "max_running_requests",
    "token_usage",
    "gen_throughput",
    "cache_hit_rate",
    "utilization",
)
SECTION_FIELDS = (  # 分节字段定义 # 分节字段定义
    (  # 内存节 # 内存节
        "memory",
        "memory",
        "has_memory",
        (
            ("weight_gb", "memory_weight_gb"),
            ("kv_cache_gb", "memory_kv_cache_gb"),
            ("graph_gb", "memory_graph_gb"),
            ("token_capacity", "memory_token_capacity"),
        ),
    ),
    (  # 推测解码节 # 推测解码节
        "spec",
        "speculative",
        "has_speculative",
        (
            ("accept_length", "speculative_accept_length"),
            ("accept_rate", "speculative_accept_rate"),
        ),
    ),
    (  # LoRA节 # LoRA节
        "lora",
        "lora",
        "has_lora",
        (
            ("slots_used", "lora_slots_used"),
            ("slots_total", "lora_slots_total"),
            ("utilization", "lora_utilization"),
        ),
    ),
    (  # 分离调度节 # 分离调度节
        "disagg",
        "disaggregation",
        "has_disaggregation",
        (
            ("mode", "disagg_mode"),
            ("prefill_bootstrap_queue_reqs", "prefill_bootstrap_queue_reqs"),
            ("prefill_inflight_queue_reqs", "prefill_inflight_queue_reqs"),
            ("decode_prealloc_queue_reqs", "decode_prealloc_queue_reqs"),
            ("decode_transfer_queue_reqs", "decode_transfer_queue_reqs"),
            ("decode_retracted_queue_reqs", "decode_retracted_queue_reqs"),
            ("kv_transfer_speed_gb_s", "kv_transfer_speed_gb_s"),
            ("kv_transfer_latency_ms", "kv_transfer_latency_ms"),
        ),
    ),
    (  # 队列节 # 队列节
        "queues",
        "queues",
        "has_queues",
        (
            ("waiting", "queue_waiting"),
            ("grammar", "queue_grammar"),
            ("paused", "queue_paused"),
            ("retracted", "queue_retracted"),
        ),
    ),
)


class LoadSnapshot(msgspec.Struct, omit_defaults=True):  # 负载快照数据类 # 负载快照数据类
    timestamp: float = 0.0  # 时间戳 # 时间戳
    dp_rank: int = 0  # DP排名 # DP排名
    num_running_reqs: int = 0  # 运行中请求数 # 运行中请求数
    num_waiting_reqs: int = 0  # 等待中请求数 # 等待中请求数
    num_used_tokens: int = 0  # 已使用Token数 # 已使用Token数
    num_total_tokens: int = 0  # 总Token数 # 总Token数
    max_total_num_tokens: int = 0  # 最大总Token数 # 最大总Token数
    max_running_requests: int = 0  # 最大运行请求数 # 最大运行请求数
    token_usage: float = 0.0  # Token使用率 # Token使用率
    gen_throughput: float = 0.0  # 生成吞吐量 # 生成吞吐量
    cache_hit_rate: float = 0.0  # 缓存命中率 # 缓存命中率
    utilization: float = 0.0  # 利用率 # 利用率

    has_memory: int = 0  # 是否有内存指标 # 是否有内存指标
    memory_weight_gb: float = 0.0  # 权重内存(GB) # 权重内存(GB)
    memory_kv_cache_gb: float = 0.0  # KV缓存内存(GB) # KV缓存内存(GB)
    memory_graph_gb: float = 0.0  # 计算图内存(GB) # 计算图内存(GB)
    memory_token_capacity: int = 0  # Token容量 # Token容量

    has_speculative: int = 0  # 是否有推测解码指标 # 是否有推测解码指标
    speculative_accept_length: float = 0.0  # 推测解码接受长度 # 推测解码接受长度
    speculative_accept_rate: float = 0.0  # 推测解码接受率 # 推测解码接受率

    has_lora: int = 0  # 是否有LoRA指标 # 是否有LoRA指标
    lora_slots_used: int = 0  # LoRA已使用槽位数 # LoRA已使用槽位数
    lora_slots_total: int = 0  # LoRA总槽位数 # LoRA总槽位数
    lora_utilization: float = 0.0  # LoRA利用率 # LoRA利用率

    has_disaggregation: int = 0  # 是否有分离调度指标 # 是否有分离调度指标
    disagg_mode: int = 0  # 分离模式 # 分离模式
    prefill_bootstrap_queue_reqs: int = 0  # 预填充引导队列请求数 # 预填充引导队列请求数
    prefill_inflight_queue_reqs: int = 0  # 预填充在途队列请求数 # 预填充在途队列请求数
    decode_prealloc_queue_reqs: int = 0  # 解码预分配队列请求数 # 解码预分配队列请求数
    decode_transfer_queue_reqs: int = 0  # 解码传输队列请求数 # 解码传输队列请求数
    decode_retracted_queue_reqs: int = 0  # 解码撤回队列请求数 # 解码撤回队列请求数
    kv_transfer_speed_gb_s: float = 0.0  # KV传输速度(GB/s) # KV传输速度(GB/s)
    kv_transfer_latency_ms: float = 0.0  # KV传输延迟(ms) # KV传输延迟(ms)

    has_queues: int = 0  # 是否有队列指标 # 是否有队列指标
    queue_waiting: int = 0  # 等待队列数 # 等待队列数
    queue_grammar: int = 0  # 语法队列数 # 语法队列数
    queue_paused: int = 0  # 暂停队列数 # 暂停队列数
    queue_retracted: int = 0  # 撤回队列数 # 撤回队列数

    @classmethod  # 类方法装饰器 # 类方法装饰器
    def from_get_loads_output(cls, output: GetLoadsReqOutput) -> LoadSnapshot:  # 从获取负载输出创建快照 # 从获取负载输出创建快照
        snapshot: dict = {}  # 初始化快照字典 # 初始化快照字典
        for name in CORE_METRIC_FIELDS:  # 遍历核心指标字段 # 遍历核心指标字段
            value = getattr(output, name)  # 获取属性值 # 获取属性值
            if name == "dp_rank":  # 如果是dp_rank字段 # 如果是dp_rank字段
                snapshot[name] = int(value) if value is not None else 0  # 转换为整数 # 转换为整数
            else:  # 否则 # 否则
                snapshot[name] = _native(value)  # 转换为原生类型 # 转换为原生类型

        for _, section_name, present_attr, attrs in SECTION_FIELDS:  # 遍历分节字段 # 遍历分节字段
            section = getattr(output, section_name, None)  # 获取分节数据 # 获取分节数据
            snapshot[present_attr] = int(section is not None)  # 设置存在标志 # 设置存在标志
            if section is None:  # 如果分节不存在 # 如果分节不存在
                continue  # 继续下一个 # 继续下一个
            for section_attr, snapshot_attr in attrs:  # 遍历分节属性 # 遍历分节属性
                value = getattr(section, section_attr)  # 获取属性值 # 获取属性值
                if snapshot_attr == "disagg_mode":  # 如果是分离模式属性 # 如果是分离模式属性
                    value = DISAGG_MODE_TO_INT.get(value, 0)  # 转换为整数 # 转换为整数
                else:  # 否则 # 否则
                    value = _native(value)  # 转换为原生类型 # 转换为原生类型
                snapshot[snapshot_attr] = value  # 设置快照属性值 # 设置快照属性值

        return cls(**snapshot)  # 创建并返回LoadSnapshot实例 # 创建并返回LoadSnapshot实例

    VALID_SECTIONS = frozenset(  # 有效的分节集合 # 有效的分节集合
        {"core", "memory", "spec", "lora", "disagg", "queues", "all"}
    )

    def to_dict(self, include: Optional[set[str]] = None) -> dict:  # 转换为字典 # 转换为字典
        load = {  # 构建核心指标字典 # 构建核心指标字典
            "dp_rank": self.dp_rank,
            "num_running_reqs": self.num_running_reqs,
            "num_waiting_reqs": self.num_waiting_reqs,
            "num_used_tokens": self.num_used_tokens,
            "num_total_tokens": self.num_total_tokens,
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_running_requests": self.max_running_requests,
            "token_usage": self.token_usage,
            "gen_throughput": self.gen_throughput,
            "cache_hit_rate": self.cache_hit_rate,
            "utilization": self.utilization,
        }

        if include is None or "all" in include:  # 如果未指定或包含全部 # 如果未指定或包含全部
            include_all = True  # 包含所有分节 # 包含所有分节
        else:  # 否则 # 否则
            if not (include <= self.VALID_SECTIONS):  # 检查包含的分节是否有效 # 检查包含的分节是否有效
                raise ValueError(
                    f"Invalid include sections: {include - self.VALID_SECTIONS}. "
                    f"Valid options: {sorted(self.VALID_SECTIONS)}"
                )  # 抛出无效分节错误 # 抛出无效分节错误
            if include == {"core"}:  # 如果仅包含核心 # 如果仅包含核心
                return load  # 直接返回核心指标 # 直接返回核心指标
            include_all = False  # 不包含所有分节 # 不包含所有分节

        for include_key, section_name, present_attr, attrs in SECTION_FIELDS:  # 遍历分节字段 # 遍历分节字段
            if not getattr(self, present_attr):  # 如果分节不存在 # 如果分节不存在
                continue  # 继续下一个 # 继续下一个
            if not include_all and include_key not in include:  # 如果不包含所有且不在指定集合中 # 如果不包含所有且不在指定集合中
                continue  # 继续下一个 # 继续下一个

            section = {}  # 初始化分节字典 # 初始化分节字典
            for section_attr, snapshot_attr in attrs:  # 遍历分节属性 # 遍历分节属性
                value = getattr(self, snapshot_attr)  # 获取属性值 # 获取属性值
                if snapshot_attr == "disagg_mode":  # 如果是分离模式属性 # 如果是分离模式属性
                    value = INT_TO_DISAGG_MODE.get(value, "null")  # 转换为字符串 # 转换为字符串
                section[section_attr] = value  # 设置分节属性值 # 设置分节属性值
            load[section_name] = section  # 添加分节到负载字典 # 添加分节到负载字典

        return load  # 返回负载字典 # 返回负载字典


snapshot_encoder = msgspec.msgpack.Encoder()  # 创建快照编码器 # 创建快照编码器
snapshot_decoder = msgspec.msgpack.Decoder(LoadSnapshot)  # 创建快照解码器 # 创建快照解码器


# ---------------------------------------------------------------------------
# SHM file layout utilities
# SHM文件布局工具
# ---------------------------------------------------------------------------

MAGIC = b"SLNS"  # 魔数标识 # 魔数标识
VERSION = 2  # 版本号 # 版本号
HEADER_STRUCT = struct.Struct("<4sHHI")  # 头部结构体（小端序：4字节魔数+2字节版本+2字节dp_size+4字节slot_size） # 头部结构体（小端序：4字节魔数+2字节版本+2字节dp_size+4字节slot_size）
SLOT_LEN_STRUCT = struct.Struct("<I")  # 槽位长度结构体（小端序4字节无符号整数） # 槽位长度结构体（小端序4字节无符号整数）
SLOT_SIZE = 16 * 1024  # 槽位大小16KB # 槽位大小16KB


@contextmanager  # 上下文管理器装饰器 # 上下文管理器装饰器
def file_lock(fd: int, lock_type: int):  # 文件锁上下文管理器 # 文件锁上下文管理器
    fcntl.flock(fd, lock_type)  # 获取文件锁 # 获取文件锁
    try:  # 异常处理 # 异常处理
        yield  # 执行上下文代码 # 执行上下文代码
    finally:  # 最终释放锁 # 最终释放锁
        fcntl.flock(fd, fcntl.LOCK_UN)  # 释放文件锁 # 释放文件锁


def shm_path_for(ipc_name: str) -> str:  # 根据IPC名称生成SHM文件路径 # 根据IPC名称生成SHM文件路径
    name = os.path.basename(ipc_name.rstrip("/")) or "default"  # 提取基本名称 # 提取基本名称
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)  # 生成安全名称 # 生成安全名称
    digest = hashlib.blake2s(ipc_name.encode(), digest_size=4).hexdigest()  # 计算哈希摘要 # 计算哈希摘要
    return f"/dev/shm/sglang_loads_{safe_name}_{digest}.shm"  # 返回SHM文件路径 # 返回SHM文件路径


def file_size(dp_size: int, slot_size: int = SLOT_SIZE) -> int:  # 计算SHM文件大小 # 计算SHM文件大小
    return HEADER_STRUCT.size + dp_size * slot_size  # 头部大小+dp_size个槽位 # 头部大小+dp_size个槽位


def slot_offset(dp_rank: int, slot_size: int = SLOT_SIZE) -> int:  # 计算槽位偏移量 # 计算槽位偏移量
    return HEADER_STRUCT.size + dp_rank * slot_size  # 头部大小+排名*槽位大小 # 头部大小+排名*槽位大小


# ---------------------------------------------------------------------------
# Writers
# 写入器
# ---------------------------------------------------------------------------


class ShmLoadSnapshotWriter:  # SHM负载快照写入器类 # SHM负载快照写入器类
    def __init__(  # 初始化方法 # 初始化方法
        self, path: str, dp_size: int, dp_rank: int, publish_interval: int = 1  # SHM路径、DP大小、DP排名、发布间隔 # SHM路径、DP大小、DP排名、发布间隔
    ):
        if dp_rank < 0 or dp_rank >= dp_size:  # 验证dp_rank范围 # 验证dp_rank范围
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_size={dp_size}")  # 抛出值错误 # 抛出值错误
        self.publish_interval = max(1, publish_interval)  # 保存发布间隔 # 保存发布间隔
        self.publish_counter = 0  # 初始化发布计数器 # 初始化发布计数器

        self.path = path  # 保存SHM路径 # 保存SHM路径
        self.dp_size = dp_size  # 保存DP大小 # 保存DP大小
        self.dp_rank = dp_rank  # 保存DP排名 # 保存DP排名
        self.slot_size = SLOT_SIZE  # 保存槽位大小 # 保存槽位大小
        self.fd = -1  # 初始化文件描述符 # 初始化文件描述符
        size = file_size(dp_size, self.slot_size)  # 计算文件大小 # 计算文件大小

        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)  # 打开SHM文件 # 打开SHM文件
        try:  # 异常处理 # 异常处理
            with file_lock(self.fd, fcntl.LOCK_EX):  # 获取排他锁 # 获取排他锁
                os.ftruncate(self.fd, size)  # 截断文件到指定大小 # 截断文件到指定大小
                self.mmap = mmap.mmap(self.fd, size, access=mmap.ACCESS_WRITE)  # 创建可写内存映射 # 创建可写内存映射
                HEADER_STRUCT.pack_into(  # 写入头部信息 # 写入头部信息
                    self.mmap, 0, MAGIC, VERSION, dp_size, self.slot_size
                )
                self._write_payload(LoadSnapshot(dp_rank=dp_rank))  # 写入初始负载 # 写入初始负载
        except Exception:  # 捕获异常 # 捕获异常
            if self.fd >= 0:  # 如果文件描述符有效 # 如果文件描述符有效
                os.close(self.fd)  # 关闭文件 # 关闭文件
            raise  # 重新抛出异常 # 重新抛出异常

    def write(self, snapshot: LoadSnapshot) -> None:  # 写入快照 # 写入快照
        if snapshot.dp_rank != self.dp_rank:  # 验证dp_rank匹配 # 验证dp_rank匹配
            raise ValueError(
                f"snapshot dp_rank={snapshot.dp_rank} does not match writer dp_rank={self.dp_rank}"
            )  # 抛出值错误 # 抛出值错误

        with file_lock(self.fd, fcntl.LOCK_EX):  # 获取排他锁 # 获取排他锁
            self._write_payload(snapshot)  # 写入负载 # 写入负载

    def _write_payload(self, snapshot: LoadSnapshot) -> None:  # 写入负载数据 # 写入负载数据
        payload = snapshot_encoder.encode(snapshot)  # 编码快照 # 编码快照
        max_payload_size = self.slot_size - SLOT_LEN_STRUCT.size  # 计算最大负载大小 # 计算最大负载大小
        if len(payload) > max_payload_size:  # 如果负载过大 # 如果负载过大
            raise ValueError(
                f"load snapshot payload size {len(payload)} exceeds slot payload "
                f"capacity {max_payload_size}"
            )  # 抛出值错误 # 抛出值错误

        offset = slot_offset(self.dp_rank, self.slot_size)  # 计算槽位偏移 # 计算槽位偏移
        payload_start = offset + SLOT_LEN_STRUCT.size  # 计算负载起始位置 # 计算负载起始位置
        payload_end = payload_start + len(payload)  # 计算负载结束位置 # 计算负载结束位置
        slot_end = offset + self.slot_size  # 计算槽位结束位置 # 计算槽位结束位置

        SLOT_LEN_STRUCT.pack_into(self.mmap, offset, 0)  # 先将长度设为0（写入中标记） # 先将长度设为0（写入中标记）
        self.mmap[payload_start:payload_end] = payload  # 写入负载数据 # 写入负载数据
        self.mmap[payload_end:slot_end] = b"\0" * (slot_end - payload_end)  # 清零剩余空间 # 清零剩余空间
        SLOT_LEN_STRUCT.pack_into(self.mmap, offset, len(payload))  # 写入实际负载长度 # 写入实际负载长度

    def close(self) -> None:  # 关闭写入器 # 关闭写入器
        self.mmap.close()  # 关闭内存映射 # 关闭内存映射
        os.close(self.fd)  # 关闭文件 # 关闭文件


class ZmqLoadSnapshotWriter:  # ZMQ负载快照写入器类 # ZMQ负载快照写入器类
    """Sends load snapshots via zmq PUSH to a ZmqShmLoadSnapshotReader.
    通过zmq PUSH向ZmqShmLoadSnapshotReader发送负载快照。

    CONFLATE is set so only the latest message is kept in the send
    buffer when the reader is slower than the writer.
    设置CONFLATE，使读取器比写入器慢时发送缓冲区仅保留最新消息。
    """

    def __init__(  # 初始化方法 # 初始化方法
        self, endpoint: str, dp_size: int, dp_rank: int, publish_interval: int = 1  # ZMQ端点、DP大小、DP排名、发布间隔 # ZMQ端点、DP大小、DP排名、发布间隔
    ):
        import zmq as _zmq  # 导入zmq模块 # 导入zmq模块

        if dp_rank < 0 or dp_rank >= dp_size:  # 验证dp_rank范围 # 验证dp_rank范围
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_size={dp_size}")  # 抛出值错误 # 抛出值错误
        self.publish_interval = max(1, publish_interval)  # 保存发布间隔 # 保存发布间隔
        self.publish_counter = 0  # 初始化发布计数器 # 初始化发布计数器
        self.dp_size = dp_size  # 保存DP大小 # 保存DP大小
        self.dp_rank = dp_rank  # 保存DP排名 # 保存DP排名

        self._zmq = _zmq  # 保存zmq模块引用 # 保存zmq模块引用
        self._ctx = _zmq.Context.instance()  # 获取zmq上下文实例 # 获取zmq上下文实例
        self._socket = self._ctx.socket(_zmq.PUSH)  # 创建PUSH套接字 # 创建PUSH套接字
        self._socket.setsockopt(_zmq.LINGER, 0)  # 设置不等待消息发送完成 # 设置不等待消息发送完成
        self._socket.setsockopt(_zmq.CONFLATE, 1)  # 启用CONFLATE（仅保留最新消息） # 启用CONFLATE（仅保留最新消息）
        self._socket.connect(endpoint)  # 连接到端点 # 连接到端点

    def write(self, snapshot: LoadSnapshot) -> None:  # 写入快照 # 写入快照
        if snapshot.dp_rank != self.dp_rank:  # 验证dp_rank匹配 # 验证dp_rank匹配
            raise ValueError(
                f"snapshot dp_rank={snapshot.dp_rank} does not match "
                f"writer dp_rank={self.dp_rank}"
            )  # 抛出值错误 # 抛出值错误
        try:  # 尝试发送 # 尝试发送
            self._socket.send(snapshot_encoder.encode(snapshot), self._zmq.NOBLOCK)  # 非阻塞发送编码后的快照 # 非阻塞发送编码后的快照
        except self._zmq.Again:  # 如果发送缓冲区满 # 如果发送缓冲区满
            pass  # 忽略，下次发送新数据 # 忽略，下次发送新数据

    def close(self) -> None:  # 关闭写入器 # 关闭写入器
        self._socket.close()  # 关闭套接字 # 关闭套接字


# ---------------------------------------------------------------------------
# Readers
# 读取器
# ---------------------------------------------------------------------------


class ShmLoadSnapshotReader:  # SHM负载快照读取器类 # SHM负载快照读取器类
    def __init__(self, path: str, dp_size: int):  # 初始化方法 # 初始化方法
        self.path = path  # 保存SHM路径 # 保存SHM路径
        self.dp_size = dp_size  # 保存DP大小 # 保存DP大小
        self.mmap: Optional[mmap.mmap] = None  # 内存映射对象 # 内存映射对象
        self.fd: Optional[int] = None  # 文件描述符 # 文件描述符
        self.slot_size = SLOT_SIZE  # 槽位大小 # 槽位大小
        self._header_warning_logged = False  # 头部警告是否已记录 # 头部警告是否已记录
        self._attach()  # 附加到SHM文件 # 附加到SHM文件

    def _attach(self) -> bool:  # 附加到SHM文件 # 附加到SHM文件
        if self.mmap is not None:  # 如果已附加 # 如果已附加
            return True  # 返回成功 # 返回成功

        try:  # 尝试打开文件 # 尝试打开文件
            fd = os.open(self.path, os.O_RDONLY)  # 以只读方式打开 # 以只读方式打开
        except FileNotFoundError:  # 文件不存在 # 文件不存在
            return False  # 返回失败 # 返回失败

        size = os.fstat(fd).st_size  # 获取文件大小 # 获取文件大小
        if size < HEADER_STRUCT.size:  # 如果文件太小 # 如果文件太小
            os.close(fd)  # 关闭文件 # 关闭文件
            return False  # 返回失败 # 返回失败

        try:  # 尝试创建内存映射 # 尝试创建内存映射
            with file_lock(fd, fcntl.LOCK_SH):  # 获取共享锁 # 获取共享锁
                mapped = mmap.mmap(fd, size, access=mmap.ACCESS_READ)  # 创建只读内存映射 # 创建只读内存映射
                magic, version, dp_size, slot_size = HEADER_STRUCT.unpack_from(  # 解析头部信息 # 解析头部信息
                    mapped, 0
                )
        except (OSError, ValueError):  # 捕获操作系统或值错误 # 捕获操作系统或值错误
            os.close(fd)  # 关闭文件 # 关闭文件
            return False  # 返回失败 # 返回失败

        if (  # 验证头部信息 # 验证头部信息
            magic != MAGIC
            or version != VERSION
            or dp_size != self.dp_size
            or slot_size < SLOT_LEN_STRUCT.size
            or size < file_size(self.dp_size, slot_size)
        ):
            mapped.close()  # 关闭内存映射 # 关闭内存映射
            os.close(fd)  # 关闭文件 # 关闭文件
            if not self._header_warning_logged:  # 如果未记录过警告 # 如果未记录过警告
                logger.warning("load shm header mismatch at %s", self.path)  # 记录警告 # 记录警告
                self._header_warning_logged = True  # 设置警告已记录 # 设置警告已记录
            return False  # 返回失败 # 返回失败

        self.mmap = mapped  # 保存内存映射 # 保存内存映射
        self.fd = fd  # 保存文件描述符 # 保存文件描述符
        self.slot_size = slot_size  # 保存实际槽位大小 # 保存实际槽位大小
        return True  # 返回成功 # 返回成功

    def read(self, dp_rank: int) -> Optional[LoadSnapshot]:  # 读取指定DP排名的快照 # 读取指定DP排名的快照
        if dp_rank < 0 or dp_rank >= self.dp_size:  # 验证dp_rank范围 # 验证dp_rank范围
            return None  # 返回None # 返回None
        if not self._attach():  # 尝试附加到SHM文件 # 尝试附加到SHM文件
            return None  # 附加失败返回None # 附加失败返回None

        assert self.fd is not None  # 断言文件描述符有效 # 断言文件描述符有效
        with file_lock(self.fd, fcntl.LOCK_SH):  # 获取共享锁 # 获取共享锁
            return self._read_slot(dp_rank)  # 读取槽位数据 # 读取槽位数据

    def _read_slot(self, dp_rank: int) -> Optional[LoadSnapshot]:  # 读取指定槽位 # 读取指定槽位
        assert self.mmap is not None  # 断言内存映射有效 # 断言内存映射有效
        offset = slot_offset(dp_rank, self.slot_size)  # 计算槽位偏移 # 计算槽位偏移
        (payload_len,) = SLOT_LEN_STRUCT.unpack_from(self.mmap, offset)  # 读取负载长度 # 读取负载长度
        max_payload_size = self.slot_size - SLOT_LEN_STRUCT.size  # 计算最大负载大小 # 计算最大负载大小
        if payload_len == 0 or payload_len > max_payload_size:  # 验证负载长度 # 验证负载长度
            return None  # 无效则返回None # 无效则返回None

        payload_start = offset + SLOT_LEN_STRUCT.size  # 计算负载起始位置 # 计算负载起始位置
        payload_end = payload_start + payload_len  # 计算负载结束位置 # 计算负载结束位置
        try:  # 尝试解码 # 尝试解码
            return snapshot_decoder.decode(self.mmap[payload_start:payload_end])  # 解码快照数据 # 解码快照数据
        except Exception as e:  # 捕获解码异常 # 捕获解码异常
            logger.debug("load snapshot decode failed for rank %s: %s", dp_rank, e)  # 记录调试日志 # 记录调试日志
            return None  # 返回None # 返回None

    def read_all(self) -> list[LoadSnapshot]:  # 读取所有DP排名的快照 # 读取所有DP排名的快照
        if not self._attach():  # 尝试附加到SHM文件 # 尝试附加到SHM文件
            return []  # 附加失败返回空列表 # 附加失败返回空列表

        assert self.fd is not None  # 断言文件描述符有效 # 断言文件描述符有效
        with file_lock(self.fd, fcntl.LOCK_SH):  # 获取共享锁 # 获取共享锁
            loads = []  # 初始化结果列表 # 初始化结果列表
            for r in range(self.dp_size):  # 遍历所有DP排名 # 遍历所有DP排名
                load = self._read_slot(r)  # 读取槽位数据 # 读取槽位数据
                if load is not None:  # 如果读取成功 # 如果读取成功
                    loads.append(load)  # 添加到结果列表 # 添加到结果列表
            return loads  # 返回结果列表 # 返回结果列表

    def close(self) -> None:  # 关闭读取器 # 关闭读取器
        if self.mmap is not None:  # 如果内存映射存在 # 如果内存映射存在
            self.mmap.close()  # 关闭内存映射 # 关闭内存映射
            self.mmap = None  # 置空引用 # 置空引用
        if self.fd is not None:  # 如果文件描述符存在 # 如果文件描述符存在
            os.close(self.fd)  # 关闭文件 # 关闭文件
            self.fd = None  # 置空引用 # 置空引用


class ZmqShmLoadSnapshotReader:  # ZMQ-SHM混合负载快照读取器类 # ZMQ-SHM混合负载快照读取器类
    """Receives snapshots via zmq PULL from writers, writes to SHM, reads from SHM.
    通过zmq PULL从写入器接收快照，写入SHM，从SHM读取。

    Transparently wraps a ShmLoadSnapshotReader.  Every read() / read_all()
    first drains the PULL socket into SHM so callers always see fresh data.
    透明封装ShmLoadSnapshotReader。每次read()/read_all()调用
    先排空PULL套接字到SHM，使调用者始终看到最新数据。
    """

    def __init__(self, endpoint: str, shm_path: str, dp_size: int):  # 初始化方法 # 初始化方法
        import zmq as _zmq  # 导入zmq模块 # 导入zmq模块

        self._zmq = _zmq  # 保存zmq模块引用 # 保存zmq模块引用
        self._ctx = _zmq.Context.instance()  # 获取zmq上下文实例 # 获取zmq上下文实例
        self._socket = self._ctx.socket(_zmq.PULL)  # 创建PULL套接字 # 创建PULL套接字
        self._socket.setsockopt(_zmq.LINGER, 0)  # 设置不等待消息接收完成 # 设置不等待消息接收完成
        self._socket.setsockopt(_zmq.CONFLATE, 1)  # 启用CONFLATE（仅保留最新消息） # 启用CONFLATE（仅保留最新消息）
        self._socket.bind(endpoint)  # 绑定到端点 # 绑定到端点

        self._endpoint = endpoint  # 保存端点地址 # 保存端点地址
        self._shm_path = shm_path  # 保存SHM路径 # 保存SHM路径
        self.dp_size = dp_size  # 保存DP大小 # 保存DP大小
        self._shm_reader = ShmLoadSnapshotReader(shm_path, dp_size)  # 创建SHM读取器 # 创建SHM读取器
        self._shm_writers: dict[int, ShmLoadSnapshotWriter] = {}  # SHM写入器字典 # SHM写入器字典

    def _poll(self) -> None:  # 轮询zmq消息并写入SHM # 轮询zmq消息并写入SHM
        """Drain zmq messages and write latest per dp_rank to SHM.
        排空zmq消息并将每个dp_rank的最新快照写入SHM。"""
        latest: dict[int, LoadSnapshot] = {}  # 存储每个dp_rank的最新快照 # 存储每个dp_rank的最新快照
        while True:  # 循环接收 # 循环接收
            try:  # 尝试接收 # 尝试接收
                data = self._socket.recv(self._zmq.NOBLOCK)  # 非阻塞接收消息 # 非阻塞接收消息
            except self._zmq.Again:  # 没有更多消息 # 没有更多消息
                break  # 跳出循环 # 跳出循环
            try:  # 尝试解码 # 尝试解码
                snapshot = snapshot_decoder.decode(data)  # 解码快照 # 解码快照
                if 0 <= snapshot.dp_rank < self.dp_size:  # 验证dp_rank范围 # 验证dp_rank范围
                    latest[snapshot.dp_rank] = snapshot  # 更新最新快照 # 更新最新快照
            except Exception as e:  # 捕获解码异常 # 捕获解码异常
                logger.warning("load snapshot zmq decode failed: %s", e)  # 记录警告日志 # 记录警告日志

        for dp_rank, snapshot in latest.items():  # 遍历最新快照 # 遍历最新快照
            if dp_rank not in self._shm_writers:  # 如果没有该排名的写入器 # 如果没有该排名的写入器
                self._shm_writers[dp_rank] = ShmLoadSnapshotWriter(  # 创建SHM写入器 # 创建SHM写入器
                    self._shm_path, self.dp_size, dp_rank
                )
            try:  # 尝试写入 # 尝试写入
                self._shm_writers[dp_rank].write(snapshot)  # 写入快照到SHM # 写入快照到SHM
            except Exception as e:  # 捕获写入异常 # 捕获写入异常
                logger.warning(
                    "load snapshot shm write failed for rank %d: %s", dp_rank, e
                )  # 记录警告日志 # 记录警告日志

    def read(self, dp_rank: int) -> Optional[LoadSnapshot]:  # 读取指定DP排名的快照 # 读取指定DP排名的快照
        self._poll()  # 先排空zmq消息 # 先排空zmq消息
        return self._shm_reader.read(dp_rank)  # 从SHM读取 # 从SHM读取

    def read_all(self) -> list[LoadSnapshot]:  # 读取所有DP排名的快照 # 读取所有DP排名的快照
        self._poll()  # 先排空zmq消息 # 先排空zmq消息
        return self._shm_reader.read_all()  # 从SHM读取 # 从SHM读取

    def close(self) -> None:  # 关闭读取器 # 关闭读取器
        for w in self._shm_writers.values():  # 遍历所有SHM写入器 # 遍历所有SHM写入器
            w.close()  # 关闭写入器 # 关闭写入器
        self._shm_writers.clear()  # 清空写入器字典 # 清空写入器字典
        self._shm_reader.close()  # 关闭SHM读取器 # 关闭SHM读取器
        self._socket.close()  # 关闭zmq套接字 # 关闭zmq套接字
        if self._endpoint.startswith("ipc://"):  # 如果是IPC端点 # 如果是IPC端点
            try:  # 尝试删除IPC文件 # 尝试删除IPC文件
                os.unlink(self._endpoint[len("ipc://") :])  # 删除IPC套接字文件 # 删除IPC套接字文件
            except OSError:  # 忽略操作系统错误 # 忽略操作系统错误
                pass  # 忽略 # 忽略


# ---------------------------------------------------------------------------
# Factory functions
# 工厂函数
# ---------------------------------------------------------------------------


def _zmq_addr_for(port_args) -> str:  # 根据端口参数生成zmq地址 # 根据端口参数生成zmq地址
    """Return the zmq PUSH/PULL address from PortArgs.
    从PortArgs返回zmq PUSH/PULL地址。

    For dp_attention (TCP mode), uses the ``load_collector_ipc_name`` field
    stored in PortArgs.  For single-node IPC (env-var override), derives
    a deterministic IPC path from ``instance_id``.
    对于dp_attention（TCP模式），使用PortArgs中存储的``load_collector_ipc_name``字段。
    对于单节点IPC（环境变量覆盖），从``instance_id``推导确定性IPC路径。
    """
    ipc_name = getattr(port_args, "load_collector_ipc_name", "")  # 获取IPC名称 # 获取IPC名称
    if ipc_name:  # 如果IPC名称存在 # 如果IPC名称存在
        return ipc_name  # 直接返回 # 直接返回
    safe = "".join(  # 生成安全名称 # 生成安全名称
        c if c.isalnum() or c in "._-" else "_" for c in port_args.instance_id
    )
    digest = hashlib.blake2s(port_args.instance_id.encode(), digest_size=4).hexdigest()  # 计算哈希摘要 # 计算哈希摘要
    return f"ipc:///tmp/sglang_load_collector_{safe}_{digest}.sock"  # 返回IPC地址 # 返回IPC地址


def create_load_snapshot_writer(  # 创建负载快照写入器的工厂函数 # 创建负载快照写入器的工厂函数
    server_args,
    port_args,
    dp_size: int,  # DP大小 # DP大小
    dp_rank: int,  # DP排名 # DP排名
    publish_interval: int = 1,  # 发布间隔 # 发布间隔
):
    """Return a SHM or ZMQ writer based on server configuration.
    根据服务器配置返回SHM或ZMQ写入器。"""
    if should_use_zmq(server_args):  # 如果应使用zmq # 如果应使用zmq
        return ZmqLoadSnapshotWriter(  # 创建ZMQ写入器 # 创建ZMQ写入器
            _zmq_addr_for(port_args), dp_size, dp_rank, publish_interval
        )
    return ShmLoadSnapshotWriter(  # 创建SHM写入器 # 创建SHM写入器
        shm_path_for(port_args.instance_id), dp_size, dp_rank, publish_interval
    )


def create_load_snapshot_reader(server_args, port_args, caller: str):  # 创建负载快照读取器的工厂函数 # 创建负载快照读取器的工厂函数
    """Create a load snapshot reader.
    创建负载快照读取器。

    Args:
        caller: ``"dp_controller"`` or ``"tokenizer"`` -- determines who
            binds the zmq PULL socket when zmq mode is active.
        caller: ``"dp_controller"``或``"tokenizer"`` -- 决定zmq模式激活时
            谁绑定zmq PULL套接字。
    """
    dp_size = server_args.dp_size  # 获取DP大小 # 获取DP大小
    if zmq_reader_owner(server_args, caller):  # 如果是zmq读取器拥有者 # 如果是zmq读取器拥有者
        return ZmqShmLoadSnapshotReader(  # 创建ZMQ-SHM混合读取器 # 创建ZMQ-SHM混合读取器
            _zmq_addr_for(port_args), shm_path_for(port_args.instance_id), dp_size
        )
    return ShmLoadSnapshotReader(shm_path_for(port_args.instance_id), dp_size)  # 创建SHM读取器 # 创建SHM读取器
