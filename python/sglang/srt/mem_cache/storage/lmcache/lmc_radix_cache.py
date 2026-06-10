# LMCache基数树缓存模块
# 本模块实现了基于LMCache的基数树(RadixCache)KV缓存，支持两种模式：
# MP模式（多进程）和IP模式（进程内），用于在分布式推理中
# 高效地存储、加载和共享KV缓存数据。

from __future__ import annotations  # 启用延迟类型注解求值 # 启用延迟类型注解求值

import enum  # 导入枚举模块 # 导入枚举模块
import logging  # 导入日志模块 # 导入日志模块
import threading  # 导入线程模块 # 导入线程模块
from dataclasses import dataclass  # 导入数据类装饰器 # 导入数据类装饰器
from typing import TYPE_CHECKING, Optional, Tuple  # 导入类型提示工具 # 导入类型提示工具

import torch  # 导入PyTorch深度学习框架 # 导入PyTorch深度学习框架

from sglang.srt.mem_cache.base_prefix_cache import (  # 导入基础前缀缓存相关类 # 导入基础前缀缓存相关类
    EvictParams,  # 驱逐参数 # 驱逐参数
    EvictResult,  # 驱逐结果 # 驱逐结果
    InitLoadBackParams,  # 初始化回加载参数 # 初始化回加载参数
    MatchPrefixParams,  # 前缀匹配参数 # 前缀匹配参数
    MatchResult,  # 匹配结果 # 匹配结果
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode  # 导入基数缓存及相关类 # 导入基数缓存及相关类
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数 # 导入全局服务器参数获取函数

try:  # 尝试导入LMCache相关模块 # 尝试导入LMCache相关模块
    from lmcache.integration.sglang.multi_process_adapter import LMCacheMPConnector  # 导入多进程连接器 # 导入多进程连接器
    from lmcache.integration.sglang.sglang_adapter import (  # 导入SGLang适配器 # 导入SGLang适配器
        LMCacheLayerwiseConnector,  # 按层连接器 # 按层连接器
        LoadMetadata,  # 加载元数据 # 加载元数据
        StoreMetadata,  # 存储元数据 # 存储元数据
    )
    from lmcache.integration.sglang.utils import lmcache_get_config  # 导入LMCache配置获取工具 # 导入LMCache配置获取工具
except ImportError as e:  # 导入失败时抛出运行时错误 # 导入失败时抛出运行时错误
    raise RuntimeError(
        "LMCache is not installed. Please install it by running `pip install lmcache`"  # 提示安装LMCache # 提示安装LMCache
    ) from e


if TYPE_CHECKING:  # 类型检查时导入 # 类型检查时导入
    from sglang.srt.configs.model_config import ModelConfig  # 导入模型配置类 # 导入模型配置类
    from sglang.srt.managers.schedule_batch import Req  # 导入请求类 # 导入请求类
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # 导入缓存初始化参数类 # 导入缓存初始化参数类

logger = logging.getLogger(__name__)  # 获取当前模块日志记录器 # 获取当前模块日志记录器


@dataclass  # 数据类装饰器 # 数据类装饰器
class _LMCacheLoadBackMarker:  # LMCache回加载标记类
    """Carries the data ``init_load_back`` needs from the
    ``match_prefix`` call in MP mode.
    """  # 在MP模式下，携带init_load_back从match_prefix调用中所需的数据 # 在MP模式下，携带init_load_back从match_prefix调用中所需的数据
    # 携带MP模式下init_load_back从match_prefix调用所需的数据

    key: RadixKey  # page-aligned key the scheduler matched on # 页面对齐的键，调度器匹配时使用的键 # 页面对齐的键，调度器匹配时使用的键
    value_numel: int  # number of tokens already in radix at match time # 匹配时基数树中已有的token数量 # 匹配时基数树中已有的token数量


class LMCacheMode(enum.Enum):  # LMCache运行模式枚举类 # LMCache运行模式枚举类
    MP = enum.auto()  # multi-process mode # 多进程模式 # 多进程模式
    IP = enum.auto()  # in-process mode # 进程内模式 # 进程内模式


class LayerTransferCounter:  # 层传输计数器类
    """Minimal adapter that lets the memory pool notify LMCache per-layer.
    最小适配器，允许内存池按层通知LMCache。

    The KV pool calls `wait_until(layer_id)` after finishing a layer, which we
    translate into a `load_kv_layerwise(layer_id)` call on the LMCache connector
    within the provided CUDA stream.
    KV池在完成一层后调用wait_until(layer_id)，我们将其转换为在指定CUDA流上
    对LMCache连接器的load_kv_layerwise(layer_id)调用。
    """

    def __init__(  # 初始化方法 # 初始化方法
        self,
        num_layers: int,  # 层数 # 层数
        load_stream: torch.cuda.Stream,  # 加载CUDA流 # 加载CUDA流
        lmc_connector: LMCacheLayerwiseConnector,  # LMCache按层连接器 # LMCache按层连接器
        printable: bool = False,  # 是否可打印 # 是否可打印
    ):
        self.num_layers = num_layers  # 保存层数 # 保存层数
        self.load_stream = load_stream  # 保存加载流 # 保存加载流
        self.lmc_connector = lmc_connector  # 保存LMCache连接器 # 保存LMCache连接器

    def wait_until(self, layer_id: int):  # 等待直到指定层加载完成 # 等待直到指定层加载完成
        # Ensure ordering of the async loads wrt compute stream(s). # 确保异步加载相对于计算流的顺序 # 确保异步加载相对于计算流的顺序
        self.load_stream.synchronize()  # 同步加载流 # 同步加载流
        with self.load_stream:  # 在加载流上执行 # 在加载流上执行
            self.lmc_connector.load_kv_layerwise(layer_id)  # 调用连接器按层加载KV # 调用连接器按层加载KV


class LMCRadixCache(RadixCache):  # LMCache基数树缓存类，继承自RadixCache # LMCache基数树缓存类
    """RadixCache + LMCache IO.
    基数树缓存 + LMCache IO。

    IP mode keeps the existing layerwise connector and
    its per-layer transfer hook: ``match_prefix`` kicks off the load via
    ``start_load_kv`` and SGLang's per-layer KV-pool hook drives subsequent
    layers during forward.
    IP模式保留现有的按层连接器及其按层传输钩子：match_prefix通过start_load_kv
    启动加载，SGLang的按层KV池钩子在forward期间驱动后续层的加载。

    MP mode uses ``LMCacheMPConnector`` with a two-phase
    load: ``match_prefix`` fires LOOKUP only (``connector.lookup_kv``) and
    returns ``host_hit_length`` on the ``MatchResult``; the SGLang
    scheduler then calls `init_load_back` at dispatch time,
    which fires the actual RETRIEVE (``connector.retrieve_kv``) into
    pre-allocated GPU slots.
    MP模式使用LMCacheMPConnector进行两阶段加载：match_prefix仅执行LOOKUP
    （connector.lookup_kv）并在MatchResult上返回host_hit_length；SGLang调度器
    在分派时调用init_load_back，执行实际的RETRIEVE（connector.retrieve_kv）
    到预分配的GPU槽位。
    """

    def __init__(  # 初始化方法 # 初始化方法
        self,
        params: CacheInitParams,  # 缓存初始化参数 # 缓存初始化参数
        model_config: Optional["ModelConfig"] = None,  # 模型配置，可选 # 模型配置，可选
        tp_size: int = 1,  # 张量并行大小，默认1 # 张量并行大小，默认1
        rank: int = 0,  # 当前进程rank，默认0 # 当前进程rank，默认0
        tp_group: Optional[torch.distributed.ProcessGroup] = None,  # 张量并行进程组，可选 # 张量并行进程组，可选
    ):
        super().__init__(params)  # 调用父类初始化 # 调用父类初始化

        cli_lmc_cfg = get_global_server_args().lmcache_config_file or ""  # 获取LMCache配置文件路径 # 获取LMCache配置文件路径

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()  # 获取KV缓存对象 # 获取KV缓存对象
        connector_kwargs = dict(  # 连接器关键字参数字典 # 连接器关键字参数字典
            sgl_config=model_config,  # SGLang配置 # SGLang配置
            tp_size=tp_size,  # 张量并行大小 # 张量并行大小
            rank=rank,  # 当前rank # 当前rank
            # NOTE: The original implementation accessed private buffers via
            # `_kvcache.k_buffer` / `.v_buffer`. We prefer public accessors when
            # available; fall back to private fields if needed.
            # 注意：原始实现通过_kvcache.k_buffer/.v_buffer访问私有缓冲区。
            # 我们优先使用公共访问器；如果不可用则回退到私有字段。
            k_pool=getattr(  # K缓冲区 # K缓冲区
                kvcache,
                "k_buffer",  # 尝试获取公共属性k_buffer # 尝试获取公共属性k_buffer
                getattr(self.token_to_kv_pool_allocator._kvcache, "k_buffer"),  # 回退到私有属性 # 回退到私有属性
            ),
            v_pool=getattr(  # V缓冲区 # V缓冲区
                kvcache,
                "v_buffer",  # 尝试获取公共属性v_buffer # 尝试获取公共属性v_buffer
                getattr(self.token_to_kv_pool_allocator._kvcache, "v_buffer"),  # 回退到私有属性 # 回退到私有属性
            ),
            tp_group=tp_group.device_group if tp_group is not None else None,  # 张量并行设备组 # 张量并行设备组
        )

        self.load_stream = torch.cuda.Stream()  # 创建加载CUDA流 # 创建加载CUDA流
        self.store_stream = torch.cuda.Stream()  # 创建存储CUDA流 # 创建存储CUDA流

        # MP is the default. To use the in-process layerwise connector,
        # set ``self._mode = LMCacheMode.IP`` here.
        # MP模式为默认模式。要使用进程内按层连接器，在此设置_mode = LMCacheMode.IP。
        self._mode = LMCacheMode.MP  # 默认使用MP模式 # 默认使用MP模式
        if self._mode is LMCacheMode.MP:  # 如果是MP模式 # 如果是MP模式
            if not cli_lmc_cfg:  # 如果没有提供配置文件 # 如果没有提供配置文件
                raise ValueError(
                    "MP mode requires --lmcache-config-file (the YAML "
                    "supplies mp_host / mp_port)."  # MP模式需要配置文件提供主机和端口 # MP模式需要配置文件提供主机和端口
                )
            lm_cfg = lmcache_get_config(cli_lmc_cfg)  # 加载LMCache配置 # 加载LMCache配置
            self.lmcache_connector = LMCacheMPConnector(  # 创建多进程连接器 # 创建多进程连接器
                page_size=params.page_size,  # 页面大小 # 页面大小
                host=lm_cfg.mp_host,  # 多进程主机地址 # 多进程主机地址
                port=lm_cfg.mp_port,  # 多进程端口 # 多进程端口
                **connector_kwargs,  # 其他连接器参数 # 其他连接器参数
            )
        elif self._mode is LMCacheMode.IP:  # 如果是IP模式 # 如果是IP模式
            self.lmcache_connector = LMCacheLayerwiseConnector(  # 创建按层连接器 # 创建按层连接器
                config_file=cli_lmc_cfg, **connector_kwargs  # 配置文件和其他参数 # 配置文件和其他参数
            )
            # Per-layer hook # 按层钩子 # 按层钩子
            self.layer_done_executor = LayerTransferCounter(  # 创建层传输计数器 # 创建层传输计数器
                num_layers=(  # 层数 # 层数
                    model_config.num_hidden_layers if model_config is not None else 0  # 从模型配置获取层数 # 从模型配置获取层数
                ),
                load_stream=self.load_stream,  # 加载流 # 加载流
                lmc_connector=self.lmcache_connector,  # LMCache连接器 # LMCache连接器
            )
            kvcache.register_layer_transfer_counter(self.layer_done_executor)  # 注册层传输计数器到KV缓存 # 注册层传输计数器到KV缓存

        self._in_flight_nodes: list[TreeNode] = []  # 正在传输中的节点列表 # 正在传输中的节点列表
        self._node_lock = threading.Lock()  # 节点操作线程锁 # 节点操作线程锁
        self._mp_load_back_markers: dict[str, _LMCacheLoadBackMarker] = {}  # MP模式回加载标记字典 # MP模式回加载标记字典

    def reset(self):  # 重置缓存状态 # 重置缓存状态
        super().reset()  # 调用父类重置 # 调用父类重置
        if hasattr(self, "_in_flight_nodes"):  # 如果存在传输中节点属性 # 如果存在传输中节点属性
            with self._node_lock:  # 加锁 # 加锁
                self._in_flight_nodes.clear()  # 清空传输中节点列表 # 清空传输中节点列表
        if hasattr(self, "_mp_load_back_markers"):  # 如果存在MP回加载标记属性 # 如果存在MP回加载标记属性
            self._mp_load_back_markers.clear()  # 清空MP回加载标记字典 # 清空MP回加载标记字典

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:  # 前缀匹配方法，根据模式分发调用 # 前缀匹配方法
        """Dispatch to the mode-specific match_prefix.
        分发到模式特定的match_prefix。

        MP mode → ``_mp_match_prefix`` (fires LOOKUP only).
        MP模式 → _mp_match_prefix（仅执行LOOKUP）。

        IP mode → ``_ip_match_prefix`` (single-shot ``start_load_kv``
        plus per-layer hook).
        IP模式 → _ip_match_prefix（一次性start_load_kv加按层钩子）。
        """
        key = params.key  # 获取匹配键 # 获取匹配键
        if self.disable or not key:  # 如果缓存被禁用或键为空 # 如果缓存被禁用或键为空
            return super().match_prefix(params)  # 直接调用父类匹配 # 直接调用父类匹配

        if self.page_size != 1:  # 如果页面大小不为1，需要对齐 # 如果页面大小不为1，需要对齐
            aligned_len = len(key) // self.page_size * self.page_size  # 计算页面对齐后的长度 # 计算页面对齐后的长度
            key = key[:aligned_len]  # 截取对齐后的键 # 截取对齐后的键

        base_res = super().match_prefix(params)  # 调用父类获取基础匹配结果 # 调用父类获取基础匹配结果
        value: torch.Tensor = base_res.device_indices  # 设备索引张量 # 设备索引张量
        last_node: TreeNode = base_res.last_device_node  # 最后一个设备节点 # 最后一个设备节点

        if self._mode is LMCacheMode.MP:  # MP模式 # MP模式
            if params.req is None:  # 如果请求为空 # 如果请求为空
                return base_res  # 返回基础结果 # 返回基础结果
            return self._mp_match_prefix(key, base_res, value, last_node, params.req)  # 调用MP模式匹配 # 调用MP模式匹配
        elif self._mode is LMCacheMode.IP:  # IP模式 # IP模式
            return self._ip_match_prefix(key, base_res, value, last_node)  # 调用IP模式匹配 # 调用IP模式匹配
        return base_res  # 其他情况返回基础结果 # 其他情况返回基础结果

    def _mp_match_prefix(  # MP模式前缀匹配方法 # MP模式前缀匹配方法
        self,
        key: RadixKey,  # 匹配键 # 匹配键
        base_res: MatchResult,  # 基础匹配结果 # 基础匹配结果
        value: torch.Tensor,  # 设备索引 # 设备索引
        last_node: TreeNode,  # 最后节点 # 最后节点
        req: Req,  # 请求对象 # 请求对象
    ) -> MatchResult:
        """MP LOOKUP
        MP模式LOOKUP

        Returns a ``MatchResult`` with ``host_hit_length`` set when
        LMCache has tokens beyond radix. Otherwise releases
        the held read locks and returns the radix-only result.
        当LMCache拥有超出基数树的token时，返回设置了host_hit_length的MatchResult。
        否则释放持有的读锁并返回仅基数树的结果。
        """
        matched = self.lmcache_connector.lookup_kv(key.token_ids, req.rid)  # 执行LOOKUP查询 # 执行LOOKUP查询
        if matched <= value.numel():  # 如果LMCache没有比基数树更多的token # 如果LMCache没有比基数树更多的token
            # Release the read locks; keep the pending session for end_session.
            # 释放读锁；保留待处理会话用于end_session。
            self.lmcache_connector.release_pending(req.rid)  # 释放待处理的读锁 # 释放待处理的读锁
            return base_res  # 返回基础结果 # 返回基础结果

        self._mp_load_back_markers[req.rid] = _LMCacheLoadBackMarker(  # 记录回加载标记 # 记录回加载标记
            key=key,  # 页面对齐的键 # 页面对齐的键
            value_numel=int(value.numel()),  # 已有token数 # 已有token数
        )
        return MatchResult(  # 返回带有host_hit_length的匹配结果 # 返回带有host_hit_length的匹配结果
            device_indices=value,  # 设备索引 # 设备索引
            last_device_node=last_node,  # 最后设备节点 # 最后设备节点
            last_host_node=last_node,  # 最后主机节点 # 最后主机节点
            best_match_node=last_node,  # 最佳匹配节点 # 最佳匹配节点
            host_hit_length=matched - int(value.numel()),  # 主机命中长度=LMCache匹配数-基数树已有数 # 主机命中长度
        )

    def _ip_match_prefix(  # IP模式前缀匹配方法 # IP模式前缀匹配方法
        self,
        key: RadixKey,  # 匹配键 # 匹配键
        base_res: MatchResult,  # 基础匹配结果 # 基础匹配结果
        value: torch.Tensor,  # 设备索引 # 设备索引
        last_node: TreeNode,  # 最后节点 # 最后节点
    ) -> MatchResult:
        """IP mode: ``start_load_kv`` + per-layer hook.
        IP模式：start_load_kv + 按层钩子。

        Allocates slots for the page-aligned uncached tail and kicks off
        the layerwise load. Returns ``base_res`` if there's nothing to
        fetch or alloc/load fails.
        为页面对齐的未缓存尾部分配槽位并启动按层加载。如果没有需要获取的内容
        或分配/加载失败，则返回base_res。
        """
        if value.numel() == len(key):  # 如果所有token都已在缓存中 # 如果所有token都已在缓存中
            return base_res  # 返回基础结果 # 返回基础结果

        uncached_len = len(key) - value.numel()  # 未缓存的token长度 # 未缓存的token长度
        if uncached_len == 0:  # 如果没有未缓存内容 # 如果没有未缓存内容
            return base_res  # 返回基础结果 # 返回基础结果

        result = self._load_back(  # 执行回加载 # 执行回加载
            key=key,  # 匹配键 # 匹配键
            value_numel=int(value.numel()),  # 已有token数 # 已有token数
            uncached_len=uncached_len,  # 未缓存长度 # 未缓存长度
            last_node=last_node,  # 最后节点 # 最后节点
            load_fn=lambda sm, pp: self._ip_load_back(  # IP模式加载函数 # IP模式加载函数
                token_ids=key.token_ids,  # token ID列表 # token ID列表
                value_numel=int(value.numel()),  # 已有token数 # 已有token数
                slot_mapping=sm,  # 槽位映射 # 槽位映射
                prefix_pad=pp,  # 前缀填充 # 前缀填充
            ),
        )
        if result is None:  # 如果加载失败 # 如果加载失败
            return base_res  # 返回基础结果 # 返回基础结果
        new_slots, new_node = result  # 解包新槽位和新节点 # 解包新槽位和新节点
        return MatchResult(  # 返回合并后的匹配结果 # 返回合并后的匹配结果
            device_indices=torch.cat([value, new_slots]),  # 拼接已有索引和新槽位 # 拼接已有索引和新槽位
            last_device_node=new_node,  # 新的最后设备节点 # 新的最后设备节点
            last_host_node=new_node,  # 新的最后主机节点 # 新的最后主机节点
            best_match_node=new_node,  # 新的最佳匹配节点 # 新的最佳匹配节点
        )

    def init_load_back(  # 初始化回加载方法 # 初始化回加载方法
        self, params: InitLoadBackParams  # 回加载参数 # 回加载参数
    ) -> Tuple[torch.Tensor, Optional[TreeNode]]:
        """MP RETRIEVE.
        MP模式RETRIEVE。

        Called by the scheduler when ``match_prefix`` returned
        ``host_hit_length > 0``. Uses the cached LOOKUP result to
        allocate slots and fire RETRIEVE, inserts the resulting
        TreeNode into the radix tree, and returns
        ``(new_indices, new_last_node)``.
        当match_prefix返回host_hit_length > 0时由调度器调用。
        使用缓存的LOOKUP结果分配槽位并执行RETRIEVE，将结果TreeNode插入
        基数树，并返回(new_indices, new_last_node)。
        """
        req = params.req  # 获取请求对象 # 获取请求对象
        marker = self._mp_load_back_markers.pop(req.rid)  # 取出并移除回加载标记 # 取出并移除回加载标记
        last_node: TreeNode = params.best_match_node  # 最佳匹配节点 # 最佳匹配节点

        result = self._load_back(  # 执行回加载 # 执行回加载
            key=marker.key,  # 匹配键 # 匹配键
            value_numel=marker.value_numel,  # 已有token数 # 已有token数
            uncached_len=params.host_hit_length,  # 主机命中长度 # 主机命中长度
            last_node=last_node,  # 最后节点 # 最后节点
            load_fn=lambda sm, pp: self._mp_load_back(  # MP模式加载函数 # MP模式加载函数
                marker=marker,  # 回加载标记 # 回加载标记
                request_id=req.rid,  # 请求ID # 请求ID
                slot_mapping=sm,  # 槽位映射 # 槽位映射
                prefix_pad=pp,  # 前缀填充 # 前缀填充
            ),
        )
        if result is None:  # 如果加载失败 # 如果加载失败
            # Either alloc failed (locks still held by lookup_kv) or
            # retrieve returned nothing (locks already released by
            # retrieve_kv). release_pending is idempotent on locks_held.
            # 分配失败（lookup_kv仍持有锁）或retrieve返回空（retrieve_kv已释放锁）。
            # release_pending对locks_held是幂等的。
            self.lmcache_connector.release_pending(req.rid)  # 释放待处理资源 # 释放待处理资源
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),  # 空索引张量 # 空索引张量
                last_node,  # 最后节点 # 最后节点
            )
        return result  # 返回加载结果 # 返回加载结果

    def _load_back(  # 通用回加载方法 # 通用回加载方法
        self,
        *,
        key: RadixKey,  # 匹配键 # 匹配键
        value_numel: int,  # 已有token数 # 已有token数
        uncached_len: int,  # 未缓存长度 # 未缓存长度
        last_node: TreeNode,  # 最后节点 # 最后节点
        load_fn,  # Callable[[torch.Tensor, int], int] — (slot_mapping, prefix_pad) -> num_retrieved # 加载函数，返回检索到的token数
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        """Alloc slots, run ``load_fn``, attach a TreeNode for what was loaded.
        分配槽位，执行load_fn，为加载的内容附加TreeNode。

        Returns ``(slots, new_node)`` on success, ``None`` if alloc fails
        or the load returned zero (slots are freed in either case).
        成功时返回(slots, new_node)，分配失败或加载返回零时返回None
        （两种情况下槽位都会被释放）。
        """
        chunk_size = self.lmcache_connector.chunk_size()  # 获取LMCache块大小 # 获取LMCache块大小
        prefix_pad = value_numel % chunk_size  # 计算前缀填充量 # 计算前缀填充量

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:  # 如果可用空间不足 # 如果可用空间不足
            self.evict(EvictParams(num_tokens=uncached_len))  # 执行驱逐释放空间 # 执行驱逐释放空间

        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)  # 分配token槽位 # 分配token槽位
        if token_slots is None:  # 如果分配失败 # 如果分配失败
            return None  # 返回None # 返回None

        slot_mapping = torch.empty(  # 创建槽位映射张量 # 创建槽位映射张量
            value_numel + token_slots.numel(),  # 总大小=已有+新分配 # 总大小=已有+新分配
            dtype=torch.int64,  # 64位整型 # 64位整型
            device=self.device,  # 设备 # 设备
        )
        slot_mapping[:value_numel].fill_(-1)  # 已有部分填充-1（不需要加载） # 已有部分填充-1
        slot_mapping[value_numel:].copy_(token_slots)  # 新部分填入分配的槽位 # 新部分填入分配的槽位

        num_retrieved = load_fn(slot_mapping, prefix_pad)  # 执行加载函数 # 执行加载函数
        logger.debug("num_retrieved_tokens: %s", num_retrieved)  # 记录检索到的token数 # 记录检索到的token数

        if num_retrieved > 0:  # 如果成功检索到token # 如果成功检索到token
            self.token_to_kv_pool_allocator.free(  # 释放多余的槽位 # 释放多余的槽位
                token_slots[(num_retrieved - prefix_pad) :]  # 只保留实际检索到的部分 # 只保留实际检索到的部分
            )
        else:  # 如果没有检索到token # 如果没有检索到token
            self.token_to_kv_pool_allocator.free(token_slots)  # 释放所有分配的槽位 # 释放所有分配的槽位

        if num_retrieved > 0:  # 如果成功检索到token # 如果成功检索到token
            fetched = num_retrieved - prefix_pad  # 实际获取的token数（去掉填充） # 实际获取的token数
            new_node = TreeNode(priority=last_node.priority)  # 创建新树节点，继承优先级 # 创建新树节点
            start = value_numel  # 起始位置 # 起始位置
            end = start + fetched  # 结束位置 # 结束位置
            new_node.key = key[start:end]  # 设置新节点的键 # 设置新节点的键
            new_node.value = token_slots[:fetched]  # 设置新节点的值（槽位） # 设置新节点的值
            new_node.parent = last_node  # 设置父节点 # 设置父节点
            last_node.children[new_node.key.child_key(self.page_size)] = new_node  # 将新节点添加到父节点的子节点 # 添加到父节点子节点
            self.evictable_size_ += fetched  # 增加可驱逐大小 # 增加可驱逐大小
            self._update_leaf_status(last_node)  # 更新父节点叶子状态 # 更新父节点叶子状态
            self._update_leaf_status(new_node)  # 更新新节点叶子状态 # 更新新节点叶子状态

            self._record_store_event(new_node.parent)  # 记录父节点存储事件 # 记录父节点存储事件
            self._record_store_event(new_node)  # 记录新节点存储事件 # 记录新节点存储事件

            return token_slots[:fetched], new_node  # 返回槽位和新节点 # 返回槽位和新节点

        return None  # 未检索到任何token，返回None # 未检索到任何token，返回None

    def _mp_load_back(  # MP模式回加载方法 # MP模式回加载方法
        self,
        *,
        marker: _LMCacheLoadBackMarker,  # 回加载标记 # 回加载标记
        request_id: str,  # 请求ID # 请求ID
        slot_mapping: torch.Tensor,  # 槽位映射 # 槽位映射
        prefix_pad: int,  # 前缀填充 # 前缀填充
    ) -> int:
        """MP non-layerwise loader: fire ``retrieve_kv`` and wait for the
        load_stream so the compute stream observes the writes.
        MP非按层加载器：执行retrieve_kv并等待load_stream，使计算流能观察到写入。
        """
        self.load_stream.wait_stream(torch.cuda.current_stream())  # 等待当前流完成 # 等待当前流完成
        with torch.cuda.stream(self.load_stream):  # 在加载流上执行 # 在加载流上执行
            n = self.lmcache_connector.retrieve_kv(  # 执行RETRIEVE操作 # 执行RETRIEVE操作
                LoadMetadata(  # 加载元数据 # 加载元数据
                    token_ids=marker.key.token_ids,  # token ID列表 # token ID列表
                    slot_mapping=slot_mapping,  # 槽位映射 # 槽位映射
                    offset=marker.value_numel - prefix_pad,  # 偏移量 # 偏移量
                    prefix_pad=prefix_pad,  # 前缀填充 # 前缀填充
                    request_id=request_id,  # 请求ID # 请求ID
                )
            )
        torch.cuda.current_stream().wait_stream(self.load_stream)  # 让当前流等待加载流完成 # 让当前流等待加载流完成
        return n  # 返回检索到的token数 # 返回检索到的token数

    def _ip_load_back(  # IP模式回加载方法 # IP模式回加载方法
        self,
        *,
        token_ids: list[int],  # token ID列表 # token ID列表
        value_numel: int,  # 已有token数 # 已有token数
        slot_mapping: torch.Tensor,  # 槽位映射 # 槽位映射
        prefix_pad: int,  # 前缀填充 # 前缀填充
    ) -> int:
        """IP layerwise loader: kick off ``start_load_kv`` on ``self.load_stream``.
        IP按层加载器：在load_stream上启动start_load_kv。

        ``start_load_kv`` enqueues the first layer's transfer; the
        ``LayerTransferCounter`` hook drives the rest during forward.
        start_load_kv将第一层的传输入队；LayerTransferCounter钩子在forward期间
        驱动剩余层的传输。
        """
        with torch.cuda.stream(self.load_stream):  # 在加载流上执行 # 在加载流上执行
            return self.lmcache_connector.start_load_kv(  # 启动按层加载 # 启动按层加载
                LoadMetadata(  # 加载元数据 # 加载元数据
                    token_ids=token_ids,  # token ID列表 # token ID列表
                    slot_mapping=slot_mapping,  # 槽位映射 # 槽位映射
                    offset=value_numel - prefix_pad,  # 偏移量 # 偏移量
                )
            )

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:  # 缓存已完成请求的KV数据 # 缓存已完成请求的KV数据
        """On request completion, insert device KV into radix and store to LMCache.
        请求完成时，将设备KV插入基数树并存储到LMCache。
        """

        super().cache_finished_req(req, is_insert=is_insert)  # 调用父类方法缓存请求 # 调用父类方法缓存请求
        if not is_insert:  # 如果不是插入操作 # 如果不是插入操作
            if self._mode is LMCacheMode.MP:  # MP模式下 # MP模式下
                self._mp_load_back_markers.pop(req.rid, None)  # 移除回加载标记 # 移除回加载标记
                self.lmcache_connector.end_session(req.rid)  # 结束LMCache会话 # 结束LMCache会话
            return  # 返回 # 返回

        global_server_args = get_global_server_args()  # 获取全局服务器参数 # 获取全局服务器参数
        topk = global_server_args.speculative_eagle_topk  # 获取推测解码topk参数 # 获取推测解码topk参数
        enable_kv_committed_len = topk is None or topk == 1  # 是否启用KV提交长度 # 是否启用KV提交长度
        if enable_kv_committed_len:  # 如果启用KV提交长度 # 如果启用KV提交长度
            kv_committed_len = req.kv_committed_len  # 使用请求的KV提交长度 # 使用请求的KV提交长度
        else:  # 否则 # 否则
            kv_committed_len = len(req.origin_input_ids) + max(  # 计算KV提交长度 # 计算KV提交长度
                len(req.output_ids) - 1, 0  # 输入长度+输出长度-1 # 输入长度+输出长度-1
            )

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]  # 截取提交长度内的token ID # 截取提交长度内的token ID
        kv_indices = self.req_to_token_pool.req_to_token[  # 获取KV索引 # 获取KV索引
            req.req_pool_idx, :kv_committed_len  # 请求池索引和提交长度 # 请求池索引和提交长度
        ]

        # Use super() to avoid a redundant LOOKUP — we only need new_last_node from radix.
        # 使用super()避免冗余LOOKUP——我们只需要基数树的new_last_node。
        match_result = super().match_prefix(  # 调用父类前缀匹配 # 调用父类前缀匹配
            MatchPrefixParams(key=RadixKey(token_ids, req.extra_key))  # 匹配参数 # 匹配参数
        )
        new_last_node = match_result.last_device_node  # 获取最后设备节点 # 获取最后设备节点
        assert new_last_node is not None  # 断言节点不为空 # 断言节点不为空

        self.inc_lock_ref(new_last_node)  # 增加节点锁引用计数 # 增加节点锁引用计数
        store_md = StoreMetadata(  # 创建存储元数据 # 创建存储元数据
            last_node=new_last_node,  # 最后节点 # 最后节点
            token_ids=token_ids,  # token ID列表 # token ID列表
            kv_indices=kv_indices,  # KV索引 # KV索引
            offset=0,  # 偏移量为0 # 偏移量为0
            request_id=req.rid,  # 请求ID # 请求ID
        )
        with torch.cuda.stream(self.store_stream):  # 在存储流上执行 # 在存储流上执行
            self.lmcache_connector.store_kv(store_md)  # 存储KV到LMCache # 存储KV到LMCache
        if self._mode is LMCacheMode.MP:  # MP模式下 # MP模式下
            # MP store_kv blocks until the daemon's signal event fires, so the slots are safe to evict immediately.
            # MP store_kv会阻塞直到守护进程的信号事件触发，因此槽位可以立即安全驱逐。
            self._mp_load_back_markers.pop(req.rid, None)  # 移除回加载标记 # 移除回加载标记
            self.dec_lock_ref(new_last_node)  # 减少节点锁引用 # 减少节点锁引用
            self.lmcache_connector.end_session(req.rid)  # 结束LMCache会话 # 结束LMCache会话
        elif self._mode is LMCacheMode.IP:  # IP模式下 # IP模式下
            # Layerwise store is async on store_stream; defer the unlock to evict()'s store_stream.synchronize().
            # 按层存储在store_stream上是异步的；延迟解锁到evict()的store_stream.synchronize()。
            with self._node_lock:  # 加节点锁 # 加节点锁
                self._in_flight_nodes.append(new_last_node)  # 将节点添加到传输中列表 # 将节点添加到传输中列表

    def evict(self, params: EvictParams) -> EvictResult:  # 驱逐缓存方法 # 驱逐缓存方法
        """Before base eviction, wait for any outstanding stores and release locks.
        在基础驱逐之前，等待所有未完成的存储并释放锁。
        """
        if self.disable:  # 如果缓存被禁用 # 如果缓存被禁用
            return EvictResult()  # 返回空驱逐结果 # 返回空驱逐结果

        self.store_stream.synchronize()  # 同步存储流，确保所有存储操作完成 # 同步存储流
        with self._node_lock:  # 加节点锁 # 加节点锁
            for node in self._in_flight_nodes:  # 遍历传输中的节点 # 遍历传输中的节点
                self.dec_lock_ref(node)  # 减少节点锁引用 # 减少节点锁引用
            self._in_flight_nodes.clear()  # 清空传输中节点列表 # 清空传输中节点列表

        return super().evict(params)  # 调用父类驱逐方法 # 调用父类驱逐方法

    def pretty_print(self):  # 格式化打印缓存信息 # 格式化打印缓存信息
        super().pretty_print()  # 调用父类打印方法 # 调用父类打印方法
        try:  # 尝试打印额外信息 # 尝试打印额外信息
            logger.debug(
                "evictable=%d protected=%d", self.evictable_size_, self.protected_size_  # 打印可驱逐和受保护大小 # 打印可驱逐和受保护大小
            )
        except Exception:  # pragma: no cover # 捕获异常，不覆盖 # 捕获异常
            pass  # 忽略异常 # 忽略异常
