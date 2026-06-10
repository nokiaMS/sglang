# 混合池组装器模块 - 负责构建各种混合缓存栈（KV、SWA、Mamba、DeepSeek V4等）
# 本文件提供了构建主机池组（HostPoolGroup）和混合缓存控制器（HybridCacheController）的工厂函数，
# 以及策略模式的选择和注册机制，用于根据不同模型类型自动选择合适的缓存栈构建策略

from __future__ import annotations # 启用延迟类型注解求值

import logging # 导入日志模块
from dataclasses import dataclass, field # 导入数据类和字段工具
from typing import TYPE_CHECKING, Any, Callable, Optional # 导入类型注解

from sglang.srt.mem_cache.hicache_storage import PoolName, SidecarPoolSpec # 导入池名称枚举和侧车池规格
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import ( # 导入混合缓存控制器
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import ( # 导入各种主机内存池
    DeepSeekV4PagedHostPool, # DeepSeek V4分页主机池
    DeepSeekV4StateHostPool, # DeepSeek V4状态主机池
    DSAIndexerPoolHost, # DSA索引器主机池
    HostPoolGroup, # 主机池组
    LogicalHostPool, # 逻辑主机池
    MambaPoolHost, # Mamba主机池
    MHATokenToKVPoolHost, # MHA Token到KV主机池
    MLATokenToKVPoolHost, # MLA Token到KV主机池
    PoolEntry, # 池条目
)
from sglang.srt.mem_cache.unified_cache_components import ComponentType # 导入组件类型枚举

if TYPE_CHECKING: # 类型检查时导入
    import torch

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams # 导入缓存初始化参数
    from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache # 导入HiMamba前缀缓存
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache # 导入Hi前缀缓存
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache # 导入统一前缀缓存
    from sglang.srt.server_args import ServerArgs # 导入服务器参数

logger = logging.getLogger(__name__) # 获取当前模块的日志记录器


def _make_layer_mapper( # 创建层映射函数
    layer_mapping: dict[int, int], # 层映射字典（全局层ID -> 传输层ID）
    transfer_layer_num: int, # 传输层数
) -> Callable[[int], Optional[int]]: # 返回映射函数
    def mapper(layer_id: int) -> Optional[int]: # 映射函数定义
        if not 0 <= layer_id < transfer_layer_num: # 如果层ID不在有效范围内
            return None # 返回None
        return layer_mapping.get(layer_id) # 返回映射后的层ID

    return mapper # 返回映射函数


def build_kv_host_pool( # 构建KV主机池
    *, # 仅限关键字参数
    kv_pool: Any, # KV设备池
    page_size: int, # 页面大小
    server_args: ServerArgs, # 服务器参数
    use_mla: bool, # 是否使用MLA
    override_kv_cache_dim: Optional[int] = None, # 覆盖KV缓存维度（可选）
):
    kv_host_pool_cls = MLATokenToKVPoolHost if use_mla else MHATokenToKVPoolHost # 根据是否使用MLA选择主机池类
    kwargs = {} # 额外参数字典
    if override_kv_cache_dim is not None: # 如果指定了覆盖KV缓存维度
        kwargs["override_kv_cache_dim"] = override_kv_cache_dim # 添加到额外参数
    return kv_host_pool_cls( # 创建并返回KV主机池
        kv_pool, # KV设备池
        server_args.hicache_ratio, # HiCache比例
        server_args.hicache_size, # HiCache大小
        page_size, # 页面大小
        server_args.hicache_mem_layout, # 内存布局
        allocator_type=server_args.hicache_storage_backend, # 分配器类型
        **kwargs, # 额外参数
    )


def build_pool_entry( # 构建池条目
    *, # 仅限关键字参数
    name: PoolName, # 池名称
    host_pool: Any, # 主机池
    device_pool: Any, # 设备池
    layer_mapping: dict[int, int], # 层映射
    transfer_layer_num: int, # 传输层数
    is_anchor: bool = False, # 是否为锚点，默认False
    host_evict_fn: Optional[Callable[[int], Any]] = None, # 主机驱逐函数（可选）
    device_evict_fn: Optional[Callable[[int], Any]] = None, # 设备驱逐函数（可选）
    device_alloc_fn: Optional[Callable[[int], Any]] = None, # 设备分配函数（可选）
    device_free_fn: Optional[Callable[[Any], Any]] = None, # 设备释放函数（可选）
) -> PoolEntry: # 返回池条目
    return PoolEntry( # 创建并返回池条目
        name=name, # 池名称
        host_pool=host_pool, # 主机池
        device_pool=device_pool, # 设备池
        layer_mapper=_make_layer_mapper(layer_mapping, transfer_layer_num), # 层映射函数
        is_primary_index_anchor=is_anchor, # 是否为主索引锚点
        host_evict_fn=host_evict_fn, # 主机驱逐函数
        device_evict_fn=device_evict_fn, # 设备驱逐函数
        device_alloc_fn=device_alloc_fn, # 设备分配函数
        device_free_fn=device_free_fn, # 设备释放函数
    )


def build_kv_only_stack( # 构建仅KV的缓存栈
    *, # 仅限关键字参数
    params: CacheInitParams, # 缓存初始化参数
    server_args: ServerArgs, # 服务器参数
    kv_pool: Any, # KV设备池
    full_layer_mapping: dict[int, int], # 完整层映射
    page_size: int, # 页面大小
    tp_group, # 张量并行进程组
    load_cache_event, # 加载缓存事件
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力上下文并行进程组（可选）
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力张量并行进程组（可选）
    storage_backend: Optional[str], # 存储后端（可选）
    use_mla: bool, # 是否使用MLA
    override_kv_cache_dim: Optional[int] = None, # 覆盖KV缓存维度（可选）
    prefetch_threshold: int = 256, # 预取阈值，默认256
    model_name: Optional[str] = None, # 模型名称（可选）
    storage_backend_extra_config: Optional[dict] = None, # 存储后端额外配置（可选）
    pp_rank: int = 0, # 流水线并行排名
    pp_size: int = 1, # 流水线并行大小
    enable_storage_metrics: bool = False, # 是否启用存储指标
) -> tuple[HostPoolGroup, HybridCacheController]: # 返回主机池组和缓存控制器
    transfer_layer_num = len(full_layer_mapping) # 传输层数=完整层映射的长度
    kv_host_pool = build_kv_host_pool( # 构建KV主机池
        kv_pool=kv_pool, # KV设备池
        page_size=page_size, # 页面大小
        server_args=server_args, # 服务器参数
        use_mla=use_mla, # 是否使用MLA
        override_kv_cache_dim=override_kv_cache_dim, # 覆盖KV缓存维度
    )
    entries = [ # 创建池条目列表
        build_pool_entry( # 构建KV池条目
            name=PoolName.KV, # 池名称为KV
            host_pool=kv_host_pool, # KV主机池
            device_pool=kv_pool, # KV设备池
            layer_mapping=full_layer_mapping, # 完整层映射
            transfer_layer_num=transfer_layer_num, # 传输层数
            is_anchor=True, # 设为锚点
        )
    ]
    host_pool_group = HostPoolGroup(entries) # 创建主机池组
    cache_controller = HybridCacheController( # 创建混合缓存控制器
        params.token_to_kv_pool_allocator, # Token到KV池分配器
        host_pool_group, # 主机池组
        page_size, # 页面大小
        tp_group, # 张量并行进程组
        load_cache_event=load_cache_event, # 加载缓存事件
        attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
        attn_tp_group=attn_tp_group, # 注意力张量并行进程组
        write_policy=server_args.hicache_write_policy, # 写策略
        io_backend=server_args.hicache_io_backend, # IO后端
        storage_backend=storage_backend, # 存储后端
        prefetch_threshold=prefetch_threshold, # 预取阈值
        model_name=model_name, # 模型名称
        storage_backend_extra_config=storage_backend_extra_config, # 存储后端额外配置
        pp_rank=pp_rank, # 流水线并行排名
        pp_size=pp_size, # 流水线并行大小
        transfer_layer_num=transfer_layer_num, # 传输层数
        enable_storage_metrics=enable_storage_metrics, # 存储指标开关
    )
    return host_pool_group, cache_controller # 返回主机池组和缓存控制器


def build_hybrid_swa_stack( # 构建混合SWA缓存栈（KV + 滑动窗口注意力）
    *, # 仅限关键字参数
    params: CacheInitParams, # 缓存初始化参数
    server_args: ServerArgs, # 服务器参数
    full_kv_pool: Any, # 完整KV设备池
    swa_kv_pool: Any, # SWA KV设备池
    full_layer_mapping: dict[int, int], # 完整层映射
    swa_layer_mapping: dict[int, int], # SWA层映射
    page_size: int, # 页面大小
    tp_group, # 张量并行进程组
    load_cache_event, # 加载缓存事件
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力上下文并行进程组（可选）
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力张量并行进程组（可选）
    storage_backend: Optional[str], # 存储后端（可选）
    use_mla: bool, # 是否使用MLA
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None, # 主机SWA驱逐函数（可选）
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None, # 设备SWA驱逐函数（可选）
    prefetch_threshold: int = 256, # 预取阈值
    model_name: Optional[str] = None, # 模型名称（可选）
    storage_backend_extra_config: Optional[dict] = None, # 存储后端额外配置（可选）
    pp_rank: int = 0, # 流水线并行排名
    pp_size: int = 1, # 流水线并行大小
    enable_storage_metrics: bool = False, # 是否启用存储指标
) -> tuple[HostPoolGroup, HybridCacheController]: # 返回主机池组和缓存控制器
    transfer_layer_num = len(full_layer_mapping | swa_layer_mapping) # 传输层数=完整层映射与SWA层映射的并集长度
    kv_host_pool = build_kv_host_pool( # 构建KV主机池
        kv_pool=full_kv_pool, # 完整KV设备池
        page_size=page_size, # 页面大小
        server_args=server_args, # 服务器参数
        use_mla=use_mla, # 是否使用MLA
    )
    swa_host_pool = build_kv_host_pool( # 构建SWA主机池
        kv_pool=swa_kv_pool, # SWA KV设备池
        page_size=page_size, # 页面大小
        server_args=server_args, # 服务器参数
        use_mla=use_mla, # 是否使用MLA
    )

    # For SWA hybrid, the device alloc/free goes through the inner swa_attn_allocator
    # 对于SWA混合模式，设备分配/释放通过内部的swa_attn_allocator进行
    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator # 获取SWA注意力分配器
    entries = [ # 创建池条目列表
        build_pool_entry( # 构建KV池条目
            name=PoolName.KV, # 池名称为KV
            host_pool=kv_host_pool, # KV主机池
            device_pool=full_kv_pool, # 完整KV设备池
            layer_mapping=full_layer_mapping, # 完整层映射
            transfer_layer_num=transfer_layer_num, # 传输层数
            is_anchor=True, # 设为锚点
        ),
        build_pool_entry( # 构建SWA池条目
            name=PoolName.SWA, # 池名称为SWA
            host_pool=swa_host_pool, # SWA主机池
            device_pool=swa_kv_pool, # SWA KV设备池
            layer_mapping=swa_layer_mapping, # SWA层映射
            transfer_layer_num=transfer_layer_num, # 传输层数
            host_evict_fn=host_swa_evict_fn, # 主机SWA驱逐函数
            device_evict_fn=device_swa_evict_fn, # 设备SWA驱逐函数
            device_alloc_fn=swa_attn_allocator.alloc, # 设备分配函数
            device_free_fn=swa_attn_allocator.free, # 设备释放函数
        ),
    ]
    host_pool_group = HostPoolGroup(entries) # 创建主机池组
    cache_controller = HybridCacheController( # 创建混合缓存控制器
        params.token_to_kv_pool_allocator, # Token到KV池分配器
        host_pool_group, # 主机池组
        page_size, # 页面大小
        tp_group, # 张量并行进程组
        load_cache_event=load_cache_event, # 加载缓存事件
        attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
        attn_tp_group=attn_tp_group, # 注意力张量并行进程组
        write_policy=server_args.hicache_write_policy, # 写策略
        io_backend=server_args.hicache_io_backend, # IO后端
        storage_backend=storage_backend, # 存储后端
        prefetch_threshold=prefetch_threshold, # 预取阈值
        model_name=model_name, # 模型名称
        storage_backend_extra_config=storage_backend_extra_config, # 存储后端额外配置
        pp_rank=pp_rank, # 流水线并行排名
        pp_size=pp_size, # 流水线并行大小
        transfer_layer_num=transfer_layer_num, # 传输层数
        enable_storage_metrics=enable_storage_metrics, # 存储指标开关
    )
    return host_pool_group, cache_controller # 返回主机池组和缓存控制器


def _deepseek_v4_num_host_pages( # 计算DeepSeek V4所需的主机页面数
    *, # 仅限关键字参数
    params: CacheInitParams, # 缓存初始化参数
    server_args: ServerArgs, # 服务器参数
    kvcache: Any, # KV缓存
    page_size: int, # 页面大小
    swa_page_size: int, # SWA页面大小
) -> tuple[int, int]: # 返回完整主机页面数和SWA主机页面数
    allocator = params.token_to_kv_pool_allocator # 获取分配器
    device_full_size = getattr(allocator, "size_full", kvcache.size) # 获取设备完整大小
    device_full_pages = (device_full_size + page_size - 1) // page_size # 计算设备完整页面数（向上取整）

    device_swa_pages = (kvcache.swa_size + swa_page_size - 1) // swa_page_size # 计算设备SWA页面数（向上取整）

    if server_args.hicache_size > 0: # 如果指定了hicache_size
        raise ValueError( # 抛出值错误
            "DeepSeek V4 HiCache currently does not support --hicache-size; " # DeepSeek V4 HiCache目前不支持--hicache-size
            "use --hicache-ratio instead." # 请使用--hicache-ratio
        )
    ratio = server_args.hicache_ratio # 获取HiCache比例
    full_host_pages = max(int(device_full_pages * ratio), device_full_pages + 1) # 完整主机页面数=设备页面数*比例和设备页面数+1的较大值
    swa_host_pages = max(int(device_swa_pages * ratio), device_swa_pages + 1) # SWA主机页面数=设备SWA页面数*比例和设备SWA页面数+1的较大值
    return full_host_pages, swa_host_pages # 返回完整主机页面数和SWA主机页面数


def build_deepseek_v4_hicache_stack( # 构建DeepSeek V4 HiCache缓存栈
    *, # 仅限关键字参数
    params: CacheInitParams, # 缓存初始化参数
    server_args: ServerArgs, # 服务器参数
    kvcache: Any, # KV缓存
    page_size: int, # 页面大小
    tp_group, # 张量并行进程组
    load_cache_event, # 加载缓存事件
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力上下文并行进程组（可选）
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力张量并行进程组（可选）
    storage_backend: Optional[str], # 存储后端（可选）
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None, # 主机SWA驱逐函数（可选）
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None, # 设备SWA驱逐函数（可选）
    prefetch_threshold: int = 256, # 预取阈值
    model_name: Optional[str] = None, # 模型名称（可选）
    storage_backend_extra_config: Optional[dict] = None, # 存储后端额外配置（可选）
    pp_rank: int = 0, # 流水线并行排名
    pp_size: int = 1, # 流水线并行大小
    enable_storage_metrics: bool = False, # 是否启用存储指标
) -> tuple[HostPoolGroup, HybridCacheController]: # 返回主机池组和缓存控制器
    # TODO(hzh0425): Support PP for deepseek v4 with hicache
    # TODO(hzh0425): 支持DeepSeek V4 HiCache的流水线并行
    transfer_layer_num = kvcache.end_layer - kvcache.start_layer # 传输层数=结束层-起始层
    full_layer_mapping = {layer_id: layer_id for layer_id in range(transfer_layer_num)} # 完整层映射（1:1映射）
    swa_layer_mapping = { # SWA层映射
        layer_id: layer_id for layer_id in range(len(kvcache.swa_kv_pool.kv_buffer)) # 基于SWA KV缓冲区数量
    }

    c4_layer_mapping = {} # C4压缩层映射
    c128_layer_mapping = {} # C128压缩层映射
    c4_state_global_layers = [] # C4状态全局层ID列表
    c128_state_global_layers = [] # C128状态全局层ID列表
    for layer_id, layer_item in enumerate( # 遍历层映射
        kvcache.layer_mapping[kvcache.start_layer : kvcache.end_layer] # 获取传输范围内的层映射
    ):
        if layer_item.compress_ratio == 4: # 如果压缩比为4
            c4_layer_mapping[layer_id] = layer_item.compress_layer_id # 添加C4层映射
            c4_state_global_layers.append(layer_id) # 添加C4状态全局层ID
        elif layer_item.compress_ratio == 128: # 如果压缩比为128
            c128_layer_mapping[layer_id] = layer_item.compress_layer_id # 添加C128层映射
            c128_state_global_layers.append(layer_id) # 添加C128状态全局层ID

    c4_state_mapping = { # C4状态映射（全局层ID -> 本地ID）
        layer_id: local_id for local_id, layer_id in enumerate(c4_state_global_layers) # 枚举创建映射
    }
    c128_state_mapping = { # C128状态映射（全局层ID -> 本地ID）
        layer_id: local_id for local_id, layer_id in enumerate(c128_state_global_layers) # 枚举创建映射
    }
    num_host_pages, swa_num_host_pages = _deepseek_v4_num_host_pages( # 计算主机页面数
        params=params, # 缓存初始化参数
        server_args=server_args, # 服务器参数
        kvcache=kvcache, # KV缓存
        page_size=page_size, # 页面大小
        swa_page_size=kvcache.swa_page_size, # SWA页面大小
    )

    logical_host_pool = LogicalHostPool(num_host_pages * page_size, page_size) # 创建逻辑主机池
    swa_host_pool = DeepSeekV4PagedHostPool( # 创建SWA主机池
        pool_name=str(PoolName.SWA), # 池名称
        device_buffers=kvcache.swa_kv_pool.kv_buffer, # 设备缓冲区
        item_bytes=kvcache.swa_kv_pool.bytes_per_page_padded, # 每页字节数
        num_host_pages=swa_num_host_pages, # 主机页面数
        slot_page_size=kvcache.swa_page_size, # 槽页面大小
        layout=server_args.hicache_mem_layout, # 内存布局
        allocator_type=server_args.hicache_storage_backend, # 分配器类型
    )
    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator # 获取SWA注意力分配器
    entries = [ # 创建基础池条目列表
        build_pool_entry( # 构建KV池条目
            name=PoolName.KV, # 池名称为KV
            host_pool=logical_host_pool, # 逻辑主机池
            device_pool=kvcache, # KV缓存
            layer_mapping=full_layer_mapping, # 完整层映射
            transfer_layer_num=transfer_layer_num, # 传输层数
            is_anchor=True, # 设为锚点
        ),
        build_pool_entry( # 构建SWA池条目
            name=PoolName.SWA, # 池名称为SWA
            host_pool=swa_host_pool, # SWA主机池
            device_pool=kvcache.swa_kv_pool, # SWA KV设备池
            layer_mapping=swa_layer_mapping, # SWA层映射
            transfer_layer_num=transfer_layer_num, # 传输层数
            host_evict_fn=host_swa_evict_fn, # 主机SWA驱逐函数
            device_evict_fn=device_swa_evict_fn, # 设备SWA驱逐函数
            device_alloc_fn=swa_attn_allocator.alloc, # 设备分配函数
            device_free_fn=swa_attn_allocator.free, # 设备释放函数
        ),
    ]

    if c4_layer_mapping: # 如果有C4压缩层
        c4_host_pool = DeepSeekV4PagedHostPool( # 创建C4主机池
            pool_name=str(PoolName.DEEPSEEK_V4_C4), # 池名称
            device_buffers=kvcache.c4_kv_pool.kv_buffer, # C4设备缓冲区
            item_bytes=kvcache.c4_kv_pool.bytes_per_page_padded, # 每页字节数
            num_host_pages=num_host_pages, # 主机页面数
            slot_page_size=page_size, # 槽页面大小
            layout=server_args.hicache_mem_layout, # 内存布局
            allocator_type=server_args.hicache_storage_backend, # 分配器类型
        )
        c4_indexer_host_pool = DeepSeekV4PagedHostPool( # 创建C4索引器主机池
            pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER), # 池名称
            device_buffers=kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer, # C4索引器设备缓冲区
            item_bytes=( # 每页字节数
                kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].shape[1] # 每页元素数
                * kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].element_size() # 每个元素字节数
            ),
            num_host_pages=num_host_pages, # 主机页面数
            slot_page_size=page_size, # 槽页面大小
            layout=server_args.hicache_mem_layout, # 内存布局
            allocator_type=server_args.hicache_storage_backend, # 分配器类型
        )
        c4_state_host_pool = DeepSeekV4StateHostPool( # 创建C4状态主机池
            pool_name=str(PoolName.DEEPSEEK_V4_C4_STATE), # 池名称
            state_pools=[ # C4状态池列表
                kvcache.compress_state_pools[layer_id] # 获取每层的压缩状态池
                for layer_id in c4_state_global_layers # 遍历C4状态全局层ID
            ],
            num_host_pages=swa_num_host_pages, # 主机页面数（与SWA相同）
            swa_page_size=kvcache.swa_page_size, # SWA页面大小
            layout=server_args.hicache_mem_layout, # 内存布局
            allocator_type=server_args.hicache_storage_backend, # 分配器类型
        )
        c4_indexer_state_host_pool = DeepSeekV4StateHostPool( # 创建C4索引器状态主机池
            pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER_STATE), # 池名称
            state_pools=[ # C4索引器状态池列表
                kvcache.indexer_compress_state_pools[layer_id] # 获取每层的索引器压缩状态池
                for layer_id in c4_state_global_layers # 遍历C4状态全局层ID
            ],
            num_host_pages=swa_num_host_pages, # 主机页面数（与SWA相同）
            swa_page_size=kvcache.swa_page_size, # SWA页面大小
            layout=server_args.hicache_mem_layout, # 内存布局
            allocator_type=server_args.hicache_storage_backend, # 分配器类型
        )
        entries.extend( # 扩展池条目列表
            [
                build_pool_entry( # 构建C4池条目
                    name=PoolName.DEEPSEEK_V4_C4, # 池名称为C4
                    host_pool=c4_host_pool, # C4主机池
                    device_pool=kvcache.c4_kv_pool, # C4设备池
                    layer_mapping=c4_layer_mapping, # C4层映射
                    transfer_layer_num=transfer_layer_num, # 传输层数
                ),
                build_pool_entry( # 构建C4索引器池条目
                    name=PoolName.DEEPSEEK_V4_C4_INDEXER, # 池名称为C4索引器
                    host_pool=c4_indexer_host_pool, # C4索引器主机池
                    device_pool=kvcache.c4_indexer_kv_pool, # C4索引器设备池
                    layer_mapping=c4_layer_mapping, # C4层映射
                    transfer_layer_num=transfer_layer_num, # 传输层数
                ),
                build_pool_entry( # 构建C4状态池条目
                    name=PoolName.DEEPSEEK_V4_C4_STATE, # 池名称为C4状态
                    host_pool=c4_state_host_pool, # C4状态主机池
                    device_pool=None, # 无设备池
                    layer_mapping=c4_state_mapping, # C4状态映射
                    transfer_layer_num=transfer_layer_num, # 传输层数
                ),
                build_pool_entry( # 构建C4索引器状态池条目
                    name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE, # 池名称为C4索引器状态
                    host_pool=c4_indexer_state_host_pool, # C4索引器状态主机池
                    device_pool=None, # 无设备池
                    layer_mapping=c4_state_mapping, # C4状态映射
                    transfer_layer_num=transfer_layer_num, # 传输层数
                ),
            ]
        )

    if c128_layer_mapping: # 如果有C128压缩层
        c128_host_pool = DeepSeekV4PagedHostPool( # 创建C128主机池
            pool_name=str(PoolName.DEEPSEEK_V4_C128), # 池名称
            device_buffers=kvcache.c128_kv_pool.kv_buffer, # C128设备缓冲区
            item_bytes=kvcache.c128_kv_pool.bytes_per_page_padded, # 每页字节数
            num_host_pages=num_host_pages, # 主机页面数
            slot_page_size=page_size, # 槽页面大小
            layout=server_args.hicache_mem_layout, # 内存布局
            allocator_type=server_args.hicache_storage_backend, # 分配器类型
        )
        c128_state_host_pool = DeepSeekV4StateHostPool( # 创建C128状态主机池
            pool_name=str(PoolName.DEEPSEEK_V4_C128_STATE), # 池名称
            state_pools=[ # C128状态池列表
                kvcache.compress_state_pools[layer_id] # 获取每层的压缩状态池
                for layer_id in c128_state_global_layers # 遍历C128状态全局层ID
            ],
            num_host_pages=swa_num_host_pages, # 主机页面数（与SWA相同）
            swa_page_size=kvcache.swa_page_size, # SWA页面大小
            layout=server_args.hicache_mem_layout, # 内存布局
            allocator_type=server_args.hicache_storage_backend, # 分配器类型
        )
        entries.extend( # 扩展池条目列表
            [
                build_pool_entry( # 构建C128池条目
                    name=PoolName.DEEPSEEK_V4_C128, # 池名称为C128
                    host_pool=c128_host_pool, # C128主机池
                    device_pool=kvcache.c128_kv_pool, # C128设备池
                    layer_mapping=c128_layer_mapping, # C128层映射
                    transfer_layer_num=transfer_layer_num, # 传输层数
                ),
                build_pool_entry( # 构建C128状态池条目
                    name=PoolName.DEEPSEEK_V4_C128_STATE, # 池名称为C128状态
                    host_pool=c128_state_host_pool, # C128状态主机池
                    device_pool=None, # 无设备池
                    layer_mapping=c128_state_mapping, # C128状态映射
                    transfer_layer_num=transfer_layer_num, # 传输层数
                ),
            ]
        )

    host_pool_group = HostPoolGroup(entries) # 创建主机池组
    cache_controller = HybridCacheController( # 创建混合缓存控制器
        params.token_to_kv_pool_allocator, # Token到KV池分配器
        host_pool_group, # 主机池组
        page_size, # 页面大小
        tp_group, # 张量并行进程组
        load_cache_event=load_cache_event, # 加载缓存事件
        attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
        attn_tp_group=attn_tp_group, # 注意力张量并行进程组
        write_policy=server_args.hicache_write_policy, # 写策略
        io_backend=server_args.hicache_io_backend, # IO后端
        storage_backend=storage_backend, # 存储后端
        prefetch_threshold=prefetch_threshold, # 预取阈值
        model_name=model_name, # 模型名称
        storage_backend_extra_config=storage_backend_extra_config, # 存储后端额外配置
        pp_rank=pp_rank, # 流水线并行排名
        pp_size=pp_size, # 流水线并行大小
        transfer_layer_num=transfer_layer_num, # 传输层数
        enable_storage_metrics=enable_storage_metrics, # 存储指标开关
    )
    return host_pool_group, cache_controller # 返回主机池组和缓存控制器


def build_hybrid_mamba_stack( # 构建混合Mamba缓存栈（KV + Mamba状态）
    *, # 仅限关键字参数
    params: CacheInitParams, # 缓存初始化参数
    server_args: ServerArgs, # 服务器参数
    kv_pool: Any, # KV设备池
    mamba_pool: Any, # Mamba设备池
    full_layer_mapping: dict[int, int], # 完整层映射
    mamba_layer_mapping: dict[int, int], # Mamba层映射
    page_size: int, # 页面大小
    tp_group, # 张量并行进程组
    load_cache_event, # 加载缓存事件
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力上下文并行进程组（可选）
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力张量并行进程组（可选）
    storage_backend: Optional[str], # 存储后端（可选）
    use_mla: bool, # 是否使用MLA
    host_mamba_evict_fn: Optional[Callable[[int], Any]] = None, # 主机Mamba驱逐函数（可选）
    device_mamba_evict_fn: Optional[Callable[[int], Any]] = None, # 设备Mamba驱逐函数（可选）
    prefetch_threshold: int = 256, # 预取阈值
    model_name: Optional[str] = None, # 模型名称（可选）
    storage_backend_extra_config: Optional[dict] = None, # 存储后端额外配置（可选）
    pp_rank: int = 0, # 流水线并行排名
    pp_size: int = 1, # 流水线并行大小
    enable_storage_metrics: bool = False, # 是否启用存储指标
) -> tuple[HostPoolGroup, HybridCacheController]: # 返回主机池组和缓存控制器
    transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping) # 传输层数=完整层映射与Mamba层映射的并集长度
    kv_host_pool = build_kv_host_pool( # 构建KV主机池
        kv_pool=kv_pool, # KV设备池
        page_size=page_size, # 页面大小
        server_args=server_args, # 服务器参数
        use_mla=use_mla, # 是否使用MLA
    )
    mamba_host_pool = MambaPoolHost( # 创建Mamba主机池
        mamba_pool, # Mamba设备池
        server_args.hicache_ratio, # HiCache比例
        server_args.hicache_size, # HiCache大小
        allocator_type=server_args.hicache_storage_backend, # 分配器类型
        layout=server_args.hicache_mem_layout, # 内存布局
    )
    entries = [ # 创建池条目列表
        build_pool_entry( # 构建KV池条目
            name=PoolName.KV, # 池名称为KV
            host_pool=kv_host_pool, # KV主机池
            device_pool=kv_pool, # KV设备池
            layer_mapping=full_layer_mapping, # 完整层映射
            transfer_layer_num=transfer_layer_num, # 传输层数
            is_anchor=True, # 设为锚点
        ),
        build_pool_entry( # 构建Mamba池条目
            name=PoolName.MAMBA, # 池名称为MAMBA
            host_pool=mamba_host_pool, # Mamba主机池
            device_pool=mamba_pool, # Mamba设备池
            layer_mapping=mamba_layer_mapping, # Mamba层映射
            transfer_layer_num=transfer_layer_num, # 传输层数
            host_evict_fn=host_mamba_evict_fn, # 主机Mamba驱逐函数
            device_evict_fn=device_mamba_evict_fn, # 设备Mamba驱逐函数
        ),
    ]
    host_pool_group = HostPoolGroup(entries) # 创建主机池组
    cache_controller = HybridCacheController( # 创建混合缓存控制器
        params.token_to_kv_pool_allocator, # Token到KV池分配器
        host_pool_group, # 主机池组
        page_size, # 页面大小
        tp_group, # 张量并行进程组
        load_cache_event=load_cache_event, # 加载缓存事件
        attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
        attn_tp_group=attn_tp_group, # 注意力张量并行进程组
        write_policy=server_args.hicache_write_policy, # 写策略
        io_backend=server_args.hicache_io_backend, # IO后端
        storage_backend=storage_backend, # 存储后端
        prefetch_threshold=prefetch_threshold, # 预取阈值
        model_name=model_name, # 模型名称
        storage_backend_extra_config=storage_backend_extra_config, # 存储后端额外配置
        pp_rank=pp_rank, # 流水线并行排名
        pp_size=pp_size, # 流水线并行大小
        transfer_layer_num=transfer_layer_num, # 传输层数
        enable_storage_metrics=enable_storage_metrics, # 存储指标开关
    )
    return host_pool_group, cache_controller # 返回主机池组和缓存控制器


def build_anchor_sidecar_stack( # 构建锚点侧车缓存栈（KV + 侧车池）
    *, # 仅限关键字参数
    params: CacheInitParams, # 缓存初始化参数
    server_args: ServerArgs, # 服务器参数
    kv_pool: Any, # KV设备池
    sidecar_pool_name: PoolName, # 侧车池名称
    full_layer_mapping: dict[int, int], # 完整层映射
    page_size: int, # 页面大小
    tp_group, # 张量并行进程组
    load_cache_event, # 加载缓存事件
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力上下文并行进程组（可选）
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力张量并行进程组（可选）
    storage_backend: Optional[str], # 存储后端（可选）
    use_mla: bool, # 是否使用MLA
    override_kv_cache_dim: Optional[int] = None, # 覆盖KV缓存维度（可选）
    sidecar_host_pool_factory: Callable[[Any], Any], # 侧车主机池工厂函数
    prefetch_threshold: int = 256, # 预取阈值
    model_name: Optional[str] = None, # 模型名称（可选）
    storage_backend_extra_config: Optional[dict] = None, # 存储后端额外配置（可选）
    pp_rank: int = 0, # 流水线并行排名
    pp_size: int = 1, # 流水线并行大小
    enable_storage_metrics: bool = False, # 是否启用存储指标
) -> tuple[HostPoolGroup, HybridCacheController]: # 返回主机池组和缓存控制器
    transfer_layer_num = len(full_layer_mapping) # 传输层数=完整层映射长度
    kv_host_pool = build_kv_host_pool( # 构建KV主机池
        kv_pool=kv_pool, # KV设备池
        page_size=page_size, # 页面大小
        server_args=server_args, # 服务器参数
        use_mla=use_mla, # 是否使用MLA
        override_kv_cache_dim=override_kv_cache_dim, # 覆盖KV缓存维度
    )
    sidecar_host_pool = sidecar_host_pool_factory(kv_host_pool) # 通过工厂函数创建侧车主机池
    entries = [ # 创建池条目列表
        build_pool_entry( # 构建KV池条目
            name=PoolName.KV, # 池名称为KV
            host_pool=kv_host_pool, # KV主机池
            device_pool=kv_pool, # KV设备池
            layer_mapping=full_layer_mapping, # 完整层映射
            transfer_layer_num=transfer_layer_num, # 传输层数
            is_anchor=True, # 设为锚点
        ),
        build_pool_entry( # 构建侧车池条目
            name=sidecar_pool_name, # 侧车池名称
            host_pool=sidecar_host_pool, # 侧车主机池
            device_pool=kv_pool, # 设备池（与KV共享）
            layer_mapping=full_layer_mapping, # 完整层映射
            transfer_layer_num=transfer_layer_num, # 传输层数
        ),
    ]
    host_pool_group = HostPoolGroup(entries) # 创建主机池组
    cache_controller = HybridCacheController( # 创建混合缓存控制器
        params.token_to_kv_pool_allocator, # Token到KV池分配器
        host_pool_group, # 主机池组
        page_size, # 页面大小
        tp_group, # 张量并行进程组
        load_cache_event=load_cache_event, # 加载缓存事件
        attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
        attn_tp_group=attn_tp_group, # 注意力张量并行进程组
        write_policy=server_args.hicache_write_policy, # 写策略
        io_backend=server_args.hicache_io_backend, # IO后端
        storage_backend=storage_backend, # 存储后端
        prefetch_threshold=prefetch_threshold, # 预取阈值
        model_name=model_name, # 模型名称
        storage_backend_extra_config=storage_backend_extra_config, # 存储后端额外配置
        pp_rank=pp_rank, # 流水线并行排名
        pp_size=pp_size, # 流水线并行大小
        transfer_layer_num=transfer_layer_num, # 传输层数
        enable_storage_metrics=enable_storage_metrics, # 存储指标开关
    )
    return host_pool_group, cache_controller # 返回主机池组和缓存控制器


_COMPONENT_HOST_ATTR: dict[ComponentType, tuple[str, str]] = { # 组件类型到主机池属性名的映射
    ComponentType.FULL: ("full_kv_pool_host", "_full_kv_pool_host"), # 完整注意力组件属性名
    ComponentType.SWA: ("swa_kv_pool_host", "_swa_kv_pool_host"), # 滑动窗口注意力组件属性名
    ComponentType.MAMBA: ("mamba_pool_host", "_mamba_pool_host"), # Mamba组件属性名
}


@dataclass
class StackBuildResult: # 栈构建结果数据类
    host_pool_group: HostPoolGroup # 主机池组
    cache_controller: HybridCacheController # 缓存控制器
    component_host_pools: dict[ComponentType, Any] # 组件主机池字典
    sidecars: list[SidecarPoolSpec] = field(default_factory=list) # 侧车池规格列表
    # Mamba state lives in req_to_token_pool, not in kvcache, so its
    # layer_transfer_counter has to be wired separately.
    # Mamba状态存储在req_to_token_pool中而非kvcache中，因此其layer_transfer_counter需要单独连接。
    register_req_to_token_counter: bool = False # 是否需要注册req_to_token的层级传输计数器
    transfer_layer_num: int = 0 # 传输层数
    pools_desc: str = "" # 池描述信息


class StackStrategy: # 栈构建策略基类
    def matches(self, kvcache: Any, components: set[ComponentType]) -> bool: # 判断是否匹配给定的KV缓存和组件
        raise NotImplementedError # 抛出未实现异常

    def build( # 构建缓存栈
        self, # 自身实例
        *, # 仅限关键字参数
        cache: UnifiedRadixCache, # 统一前缀缓存
        kvcache: Any, # KV缓存
        params: CacheInitParams, # 缓存初始化参数
        server_args: ServerArgs, # 服务器参数
        load_cache_event, # 加载缓存事件
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力上下文并行进程组（可选）
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力张量并行进程组（可选）
        storage_backend: Optional[str] = None, # 存储后端（可选）
        storage_backend_extra_config: Optional[dict] = None, # 存储后端额外配置（可选）
        prefetch_threshold: int = 256, # 预取阈值
        model_name: Optional[str] = None, # 模型名称（可选）
        enable_storage_metrics: bool = False, # 是否启用存储指标
    ) -> StackBuildResult: # 返回栈构建结果
        raise NotImplementedError # 抛出未实现异常


class _DeepSeekV4Strategy(StackStrategy): # DeepSeek V4策略
    def matches(self, kvcache, components): # 判断是否匹配DeepSeek V4类型
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import ( # 导入DeepSeek V4内存池
            DeepSeekV4TokenToKVPool,
        )

        return isinstance(kvcache, DeepSeekV4TokenToKVPool) and components == { # 匹配DeepSeek V4和FULL+SWA组件
            ComponentType.FULL,
            ComponentType.SWA,
        }

    def build( # 构建DeepSeek V4缓存栈
        self, # 自身实例
        *, # 仅限关键字参数
        cache, # 统一前缀缓存
        kvcache, # KV缓存
        params, # 缓存初始化参数
        server_args, # 服务器参数
        load_cache_event, # 加载缓存事件
        attn_cp_group=None, # 注意力上下文并行进程组
        attn_tp_group=None, # 注意力张量并行进程组
        storage_backend=None, # 存储后端
        storage_backend_extra_config=None, # 存储后端额外配置
        prefetch_threshold=256, # 预取阈值
        model_name=None, # 模型名称
        enable_storage_metrics=False, # 是否启用存储指标
    ):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams # 导入驱逐参数

        host_pool_group, cache_controller = build_deepseek_v4_hicache_stack( # 构建DeepSeek V4 HiCache栈
            params=params, # 缓存初始化参数
            server_args=server_args, # 服务器参数
            kvcache=kvcache, # KV缓存
            page_size=cache.page_size, # 页面大小
            tp_group=params.tp_cache_group, # 张量并行缓存进程组
            load_cache_event=load_cache_event, # 加载缓存事件
            attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
            attn_tp_group=attn_tp_group, # 注意力张量并行进程组
            storage_backend=storage_backend, # 存储后端
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA), # 主机SWA驱逐函数
            device_swa_evict_fn=lambda n: cache.evict(EvictParams(swa_num_tokens=n)), # 设备SWA驱逐函数
            prefetch_threshold=prefetch_threshold, # 预取阈值
            model_name=model_name, # 模型名称
            storage_backend_extra_config=storage_backend_extra_config, # 存储后端额外配置
            pp_rank=params.pp_rank, # 流水线并行排名
            pp_size=params.pp_size, # 流水线并行大小
            enable_storage_metrics=enable_storage_metrics, # 存储指标开关
        )
        sidecars = [ # 创建侧车池规格列表
            SidecarPoolSpec(pool_name=name, indices_from_pool=src) # 创建侧车池规格
            for name, src in ( # 遍历侧车池名称和源池映射
                (PoolName.DEEPSEEK_V4_C4, PoolName.KV), # C4源自KV
                (PoolName.DEEPSEEK_V4_C4_INDEXER, PoolName.KV), # C4索引器源自KV
                (PoolName.DEEPSEEK_V4_C128, PoolName.KV), # C128源自KV
                (PoolName.DEEPSEEK_V4_C4_STATE, PoolName.SWA), # C4状态源自SWA
                (PoolName.DEEPSEEK_V4_C4_INDEXER_STATE, PoolName.SWA), # C4索引器状态源自SWA
                (PoolName.DEEPSEEK_V4_C128_STATE, PoolName.SWA), # C128状态源自SWA
            )
            if name in host_pool_group.entry_map # 仅包含池组中存在的侧车
        ]
        return StackBuildResult( # 返回栈构建结果
            host_pool_group=host_pool_group, # 主机池组
            cache_controller=cache_controller, # 缓存控制器
            component_host_pools={ # 组件主机池字典
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV), # 完整注意力主机池
                ComponentType.SWA: host_pool_group.get_pool(PoolName.SWA), # SWA主机池
            },
            sidecars=sidecars, # 侧车池规格列表
            transfer_layer_num=kvcache.end_layer - kvcache.start_layer, # 传输层数
            pools_desc="KV + SWA + DeepSeekV4 sidecars", # 池描述
        )


class _MambaStrategy(StackStrategy): # Mamba策略
    def matches(self, kvcache, components): # 判断是否匹配Mamba类型
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool # 导入混合线性KV池

        return isinstance(kvcache, HybridLinearKVPool) and components == { # 匹配混合线性KV池和FULL+MAMBA组件
            ComponentType.FULL,
            ComponentType.MAMBA,
        }

    def build( # 构建Mamba缓存栈
        self, # 自身实例
        *, # 仅限关键字参数
        cache, # 统一前缀缓存
        kvcache, # KV缓存
        params, # 缓存初始化参数
        server_args, # 服务器参数
        load_cache_event, # 加载缓存事件
        attn_cp_group=None, # 注意力上下文并行进程组
        attn_tp_group=None, # 注意力张量并行进程组
        storage_backend=None, # 存储后端
        storage_backend_extra_config=None, # 存储后端额外配置
        prefetch_threshold=256, # 预取阈值
        model_name=None, # 模型名称
        enable_storage_metrics=False, # 是否启用存储指标
    ):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams # 导入驱逐参数

        full_layer_mapping = dict(kvcache.full_attention_layer_id_mapping) # 获取完整注意力层映射
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map) # 获取Mamba层映射
        host_pool_group, cache_controller = build_hybrid_mamba_stack( # 构建混合Mamba栈
            params=params, # 缓存初始化参数
            server_args=server_args, # 服务器参数
            kv_pool=kvcache.full_kv_pool, # 完整KV设备池
            mamba_pool=params.req_to_token_pool.mamba_pool, # Mamba设备池
            full_layer_mapping=full_layer_mapping, # 完整层映射
            mamba_layer_mapping=mamba_layer_mapping, # Mamba层映射
            page_size=cache.page_size, # 页面大小
            tp_group=params.tp_cache_group, # 张量并行缓存进程组
            load_cache_event=load_cache_event, # 加载缓存事件
            attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
            attn_tp_group=attn_tp_group, # 注意力张量并行进程组
            storage_backend=storage_backend, # 存储后端
            use_mla=kvcache.use_mla, # 是否使用MLA
            host_mamba_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA), # 主机Mamba驱逐函数
            device_mamba_evict_fn=lambda n: cache.evict(EvictParams(mamba_num=n)), # 设备Mamba驱逐函数
            prefetch_threshold=prefetch_threshold, # 预取阈值
            model_name=model_name, # 模型名称
            storage_backend_extra_config=storage_backend_extra_config, # 存储后端额外配置
            pp_rank=params.pp_rank, # 流水线并行排名
            pp_size=params.pp_size, # 流水线并行大小
            enable_storage_metrics=enable_storage_metrics, # 存储指标开关
        )
        return StackBuildResult( # 返回栈构建结果
            host_pool_group=host_pool_group, # 主机池组
            cache_controller=cache_controller, # 缓存控制器
            component_host_pools={ # 组件主机池字典
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV), # 完整注意力主机池
                ComponentType.MAMBA: host_pool_group.get_pool(PoolName.MAMBA), # Mamba主机池
            },
            register_req_to_token_counter=True, # 需要注册req_to_token计数器
            transfer_layer_num=len(full_layer_mapping | mamba_layer_mapping), # 传输层数
            pools_desc="KV + MAMBA", # 池描述
        )


def _swa_layer_mappings(kvcache) -> tuple[dict[int, int], dict[int, int]]: # 提取SWA层映射（分为完整层和SWA层）
    full = { # 完整层映射
        gid: lid for gid, (lid, is_swa) in kvcache.layers_mapping.items() if not is_swa # 非SWA的层
    }
    swa = {gid: lid for gid, (lid, is_swa) in kvcache.layers_mapping.items() if is_swa} # SWA的层
    return full, swa # 返回完整层映射和SWA层映射


class _SwaStrategy(StackStrategy): # SWA策略
    def matches(self, kvcache, components): # 判断是否匹配SWA类型
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import ( # 导入DeepSeek V4内存池
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool # 导入SWA KV池

        return ( # 匹配SWA KV池（排除DeepSeek V4）和FULL+SWA组件
            isinstance(kvcache, SWAKVPool)
            and not isinstance(kvcache, DeepSeekV4TokenToKVPool) # 排除DeepSeek V4
            and components == {ComponentType.FULL, ComponentType.SWA} # 匹配FULL+SWA组件
        )

    def build( # 构建SWA缓存栈
        self, # 自身实例
        *, # 仅限关键字参数
        cache, # 统一前缀缓存
        kvcache, # KV缓存
        params, # 缓存初始化参数
        server_args, # 服务器参数
        load_cache_event, # 加载缓存事件
        attn_cp_group=None, # 注意力上下文并行进程组
        attn_tp_group=None, # 注意力张量并行进程组
        storage_backend=None, # 存储后端
        storage_backend_extra_config=None, # 存储后端额外配置
        prefetch_threshold=256, # 预取阈值
        model_name=None, # 模型名称
        enable_storage_metrics=False, # 是否启用存储指标
    ):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams # 导入驱逐参数

        full_layer_mapping, swa_layer_mapping = _swa_layer_mappings(kvcache) # 提取SWA层映射
        host_pool_group, cache_controller = build_hybrid_swa_stack( # 构建混合SWA栈
            params=params, # 缓存初始化参数
            server_args=server_args, # 服务器参数
            full_kv_pool=kvcache.full_kv_pool, # 完整KV设备池
            swa_kv_pool=kvcache.swa_kv_pool, # SWA KV设备池
            full_layer_mapping=full_layer_mapping, # 完整层映射
            swa_layer_mapping=swa_layer_mapping, # SWA层映射
            page_size=cache.page_size, # 页面大小
            tp_group=params.tp_cache_group, # 张量并行缓存进程组
            load_cache_event=load_cache_event, # 加载缓存事件
            attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
            attn_tp_group=attn_tp_group, # 注意力张量并行进程组
            storage_backend=storage_backend, # 存储后端
            use_mla=False, # SWA不使用MLA
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA), # 主机SWA驱逐函数
            device_swa_evict_fn=lambda n: cache.evict(EvictParams(swa_num_tokens=n)), # 设备SWA驱逐函数
            prefetch_threshold=prefetch_threshold, # 预取阈值
            model_name=model_name, # 模型名称
            storage_backend_extra_config=storage_backend_extra_config, # 存储后端额外配置
            pp_rank=params.pp_rank, # 流水线并行排名
            pp_size=params.pp_size, # 流水线并行大小
            enable_storage_metrics=enable_storage_metrics, # 存储指标开关
        )
        return StackBuildResult( # 返回栈构建结果
            host_pool_group=host_pool_group, # 主机池组
            cache_controller=cache_controller, # 缓存控制器
            component_host_pools={ # 组件主机池字典
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV), # 完整注意力主机池
                ComponentType.SWA: host_pool_group.get_pool(PoolName.SWA), # SWA主机池
            },
            transfer_layer_num=len(full_layer_mapping | swa_layer_mapping), # 传输层数
            pools_desc="KV + SWA", # 池描述
        )


class _DsaStrategy(StackStrategy): # DSA策略
    def matches(self, kvcache, components): # 判断是否匹配DSA类型
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool # 导入DSA Token到KV池

        return isinstance(kvcache, DSATokenToKVPool) and components == { # 匹配DSA Token到KV池和FULL组件
            ComponentType.FULL
        }

    def build( # 构建DSA缓存栈
        self, # 自身实例
        *, # 仅限关键字参数
        cache, # 统一前缀缓存
        kvcache, # KV缓存
        params, # 缓存初始化参数
        server_args, # 服务器参数
        load_cache_event, # 加载缓存事件
        attn_cp_group=None, # 注意力上下文并行进程组
        attn_tp_group=None, # 注意力张量并行进程组
        storage_backend=None, # 存储后端
        storage_backend_extra_config=None, # 存储后端额外配置
        prefetch_threshold=256, # 预取阈值
        model_name=None, # 模型名称
        enable_storage_metrics=False, # 是否启用存储指标
    ):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool # 导入MLA Token到KV池

        full_kv_pool = kvcache # 完整KV池即为KV缓存
        use_mla = isinstance(kvcache, MLATokenToKVPool) # 判断是否使用MLA
        full_layer_mapping = {i: i for i in range(full_kv_pool.layer_num)} # 创建1:1层映射
        host_pool_group, cache_controller = build_anchor_sidecar_stack( # 构建锚点侧车栈
            params=params, # 缓存初始化参数
            server_args=server_args, # 服务器参数
            kv_pool=full_kv_pool, # KV设备池
            sidecar_pool_name=PoolName.INDEXER, # 侧车池名称为INDEXER
            full_layer_mapping=full_layer_mapping, # 完整层映射
            page_size=cache.page_size, # 页面大小
            tp_group=params.tp_cache_group, # 张量并行缓存进程组
            load_cache_event=load_cache_event, # 加载缓存事件
            attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
            attn_tp_group=attn_tp_group, # 注意力张量并行进程组
            storage_backend=storage_backend, # 存储后端
            use_mla=use_mla, # 是否使用MLA
            override_kv_cache_dim=full_kv_pool.kv_cache_dim, # 覆盖KV缓存维度
            sidecar_host_pool_factory=lambda kv_host_pool: DSAIndexerPoolHost( # 侧车主机池工厂函数
                full_kv_pool, # 完整KV池
                kv_host_pool, # KV主机池
                server_args.hicache_mem_layout, # 内存布局
                allocator_type=server_args.hicache_storage_backend, # 分配器类型
            ),
            prefetch_threshold=prefetch_threshold, # 预取阈值
            model_name=model_name, # 模型名称
            storage_backend_extra_config=storage_backend_extra_config, # 存储后端额外配置
            pp_rank=params.pp_rank, # 流水线并行排名
            pp_size=params.pp_size, # 流水线并行大小
            enable_storage_metrics=enable_storage_metrics, # 存储指标开关
        )
        return StackBuildResult( # 返回栈构建结果
            host_pool_group=host_pool_group, # 主机池组
            cache_controller=cache_controller, # 缓存控制器
            component_host_pools={ # 组件主机池字典
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV), # 完整注意力主机池
            },
            sidecars=[ # 侧车池规格列表
                SidecarPoolSpec( # 创建侧车池规格
                    pool_name=PoolName.INDEXER, # 侧车池名称为INDEXER
                    indices_from_pool=PoolName.KV, # 索引源自KV池
                ),
            ],
            transfer_layer_num=len(full_layer_mapping), # 传输层数
            pools_desc="KV + INDEXER", # 池描述
        )


class _PlainKvStrategy(StackStrategy): # 纯KV策略（兜底策略）
    def matches(self, kvcache, components): # 判断是否匹配纯KV类型
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import ( # 导入DeepSeek V4内存池
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.memory_pool import ( # 导入内存池
            DSATokenToKVPool,
            HybridLinearKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool # 导入SWA KV池

        if isinstance( # 如果是以下特殊类型之一，则不匹配
            kvcache,
            (SWAKVPool, HybridLinearKVPool, DSATokenToKVPool, DeepSeekV4TokenToKVPool),
        ):
            return False # 返回False
        return components == {ComponentType.FULL} # 匹配仅FULL组件

    def build( # 构建纯KV缓存栈
        self, # 自身实例
        *, # 仅限关键字参数
        cache, # 统一前缀缓存
        kvcache, # KV缓存
        params, # 缓存初始化参数
        server_args, # 服务器参数
        load_cache_event, # 加载缓存事件
        attn_cp_group=None, # 注意力上下文并行进程组
        attn_tp_group=None, # 注意力张量并行进程组
        storage_backend=None, # 存储后端
        storage_backend_extra_config=None, # 存储后端额外配置
        prefetch_threshold=256, # 预取阈值
        model_name=None, # 模型名称
        enable_storage_metrics=False, # 是否启用存储指标
    ):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool # 导入MLA Token到KV池

        full_kv_pool = kvcache # 完整KV池即为KV缓存
        use_mla = isinstance(kvcache, MLATokenToKVPool) # 判断是否使用MLA
        full_layer_mapping = {i: i for i in range(full_kv_pool.layer_num)} # 创建1:1层映射
        host_pool_group, cache_controller = build_kv_only_stack( # 构建仅KV栈
            params=params, # 缓存初始化参数
            server_args=server_args, # 服务器参数
            kv_pool=full_kv_pool, # KV设备池
            full_layer_mapping=full_layer_mapping, # 完整层映射
            page_size=cache.page_size, # 页面大小
            tp_group=params.tp_cache_group, # 张量并行缓存进程组
            load_cache_event=load_cache_event, # 加载缓存事件
            attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
            attn_tp_group=attn_tp_group, # 注意力张量并行进程组
            storage_backend=storage_backend, # 存储后端
            use_mla=use_mla, # 是否使用MLA
            prefetch_threshold=prefetch_threshold, # 预取阈值
            model_name=model_name, # 模型名称
            storage_backend_extra_config=storage_backend_extra_config, # 存储后端额外配置
            pp_rank=params.pp_rank, # 流水线并行排名
            pp_size=params.pp_size, # 流水线并行大小
            enable_storage_metrics=enable_storage_metrics, # 存储指标开关
        )
        return StackBuildResult( # 返回栈构建结果
            host_pool_group=host_pool_group, # 主机池组
            cache_controller=cache_controller, # 缓存控制器
            component_host_pools={ # 组件主机池字典
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV), # 完整注意力主机池
            },
            transfer_layer_num=len(full_layer_mapping), # 传输层数
            pools_desc="KV", # 池描述
        )


# Resolved first-to-last; _PlainKvStrategy is the catch-all fallback.
# 按优先级从高到低解析；_PlainKvStrategy是兜底策略。
_STRATEGIES: list[StackStrategy] = [ # 策略列表
    _DeepSeekV4Strategy(), # DeepSeek V4策略
    _MambaStrategy(), # Mamba策略
    _SwaStrategy(), # SWA策略
    _DsaStrategy(), # DSA策略
    _PlainKvStrategy(), # 纯KV策略（兜底）
]


def register_stack_strategy(strategy: StackStrategy) -> None: # 注册自定义栈构建策略
    """Prepend a strategy so downstream forks can plug in (kvcache, components)
    combinations not in the built-in list."""
    """在列表头部插入策略，使下游分支可以插入内置列表中没有的(kvcache, components)组合。"""
    _STRATEGIES.insert(0, strategy) # 在列表头部插入策略


def _select_strategy(kvcache: Any, components: set[ComponentType]) -> StackStrategy: # 选择匹配的栈构建策略
    for strategy in _STRATEGIES: # 按优先级遍历策略
        if strategy.matches(kvcache, components): # 如果匹配
            return strategy # 返回匹配的策略
    raise AssertionError( # 抛出断言错误
        f"No matching HiCache strategy for kvcache={type(kvcache).__name__}, " # 没有匹配的HiCache策略
        f"components={sorted(c.name for c in components)}"
    )


def _apply_stack_result( # 将栈构建结果应用到统一前缀缓存
    cache: UnifiedRadixCache, # 统一前缀缓存
    kvcache: Any, # KV缓存
    params: CacheInitParams, # 缓存初始化参数
    result: StackBuildResult, # 栈构建结果
) -> None:
    cache.host_pool_group = result.host_pool_group # 设置主机池组
    cache.cache_controller = result.cache_controller # 设置缓存控制器

    for ct, host_pool in result.component_host_pools.items(): # 遍历组件主机池
        cache_attr, component_attr = _COMPONENT_HOST_ATTR[ct] # 获取属性名
        setattr(cache, cache_attr, host_pool) # 设置缓存属性
        setattr(cache.components[ct], component_attr, host_pool) # 设置组件属性

    for sidecar in result.sidecars: # 遍历侧车池规格
        cache.register_sidecar_pool(sidecar) # 注册侧车池

    kvcache.register_layer_transfer_counter(result.cache_controller.layer_done_counter) # 注册层级传输计数器
    if result.register_req_to_token_counter: # 如果需要注册req_to_token计数器
        params.req_to_token_pool.register_layer_transfer_counter( # 注册层级传输计数器
            result.cache_controller.layer_done_counter # 传入层级完成计数器
        )

    logger.info( # 记录信息日志
        "Attached hybrid pool stack to UnifiedRadixCache: pools=%s, transfer_layer_num=%s", # 已将混合池栈附加到UnifiedRadixCache
        result.pools_desc, # 池描述
        result.transfer_layer_num, # 传输层数
    )


def attach_hybrid_pool_to_unified_cache( # 将混合池栈附加到统一前缀缓存
    cache: UnifiedRadixCache, # 统一前缀缓存
    params: CacheInitParams, # 缓存初始化参数
    server_args: ServerArgs, # 服务器参数
    *, # 仅限关键字参数
    load_cache_event, # 加载缓存事件
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力上下文并行进程组（可选）
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力张量并行进程组（可选）
    storage_backend: Optional[str] = None, # 存储后端（可选）
    storage_extra_config: Optional[dict] = None, # 存储额外配置（可选）
    storage_prefetch_threshold: int = 256, # 存储预取阈值
) -> None:
    """Attach HostPoolGroup + HybridCacheController to UnifiedRadixCache."""
    """将HostPoolGroup + HybridCacheController附加到UnifiedRadixCache。"""
    try: # 尝试执行
        kvcache = params.token_to_kv_pool_allocator.get_kvcache() # 获取KV缓存
        components = set(cache.components.keys()) # 获取组件集合
        strategy = _select_strategy(kvcache, components) # 选择匹配的策略
        result = strategy.build( # 构建缓存栈
            cache=cache, # 统一前缀缓存
            kvcache=kvcache, # KV缓存
            params=params, # 缓存初始化参数
            server_args=server_args, # 服务器参数
            load_cache_event=load_cache_event, # 加载缓存事件
            attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
            attn_tp_group=attn_tp_group, # 注意力张量并行进程组
            storage_backend=storage_backend, # 存储后端
            storage_backend_extra_config=storage_extra_config, # 存储后端额外配置
            prefetch_threshold=storage_prefetch_threshold, # 预取阈值
            model_name=server_args.served_model_name, # 模型名称
            enable_storage_metrics=cache._enable_metrics_flag, # 存储指标标志
        )
        _apply_stack_result(cache, kvcache, params, result) # 应用栈构建结果
    except Exception: # 捕获异常
        logger.exception("attach_hybrid_pool_to_unified_cache failed") # 记录异常日志
        raise # 重新抛出异常


def attach_hybrid_dsa_pool_to_hiradix_cache( # 将混合DSA池栈附加到HiRadixCache
    radix_cache: HiRadixCache, # Hi前缀缓存
    params: CacheInitParams, # 缓存初始化参数
    server_args: ServerArgs, # 服务器参数
    *, # 仅限关键字参数
    extra_config: dict, # 额外配置
    prefetch_threshold: int, # 预取阈值
    enable_storage_metrics: bool, # 是否启用存储指标
    load_cache_event, # 加载缓存事件
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力上下文并行进程组（可选）
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力张量并行进程组（可选）
) -> None:
    """Attach HostPoolGroup (KV + indexer) + HybridCacheController for HiRadixCache.
    将HostPoolGroup（KV + 索引器）+ HybridCacheController附加到HiRadixCache。

    This entrypoint is currently intended only for HiRadixCache's DSA path.
    此入口当前仅用于HiRadixCache的DSA路径。
    """
    try: # 尝试执行
        kv = radix_cache.kv_cache # 获取KV缓存
        layer_mapping = {layer_id: layer_id for layer_id in range(kv.layer_num)} # 创建1:1层映射
        host_pool_group, cache_controller = build_anchor_sidecar_stack( # 构建锚点侧车栈
            params=params, # 缓存初始化参数
            server_args=server_args, # 服务器参数
            kv_pool=kv, # KV设备池
            sidecar_pool_name=PoolName.INDEXER, # 侧车池名称为INDEXER
            full_layer_mapping=layer_mapping, # 完整层映射
            page_size=radix_cache.page_size, # 页面大小
            tp_group=radix_cache.tp_group, # 张量并行进程组
            load_cache_event=load_cache_event, # 加载缓存事件
            attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
            attn_tp_group=attn_tp_group, # 注意力张量并行进程组
            storage_backend=server_args.hicache_storage_backend, # 存储后端
            use_mla=True, # 使用MLA
            override_kv_cache_dim=kv.kv_cache_dim, # 覆盖KV缓存维度
            prefetch_threshold=prefetch_threshold, # 预取阈值
            sidecar_host_pool_factory=lambda kv_host_pool: DSAIndexerPoolHost( # 侧车主机池工厂函数
                kv, # KV缓存
                kv_host_pool, # KV主机池
                server_args.hicache_mem_layout, # 内存布局
                allocator_type=server_args.hicache_storage_backend, # 分配器类型
            ),
            model_name=server_args.served_model_name, # 模型名称
            storage_backend_extra_config=extra_config, # 存储后端额外配置
            pp_rank=radix_cache.pp_rank, # 流水线并行排名
            pp_size=radix_cache.pp_size, # 流水线并行大小
            enable_storage_metrics=enable_storage_metrics, # 存储指标开关
        )
        radix_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV) # 设置完整KV主机池
        radix_cache.token_to_kv_pool_host = host_pool_group # 设置Token到KV主机池组
        radix_cache.cache_controller = cache_controller # 设置缓存控制器
        logger.info( # 记录信息日志
            "Attached hybrid DSA pool stack to HiRadixCache: pools=KV + INDEXER, " # 已将混合DSA池栈附加到HiRadixCache
            "transfer_layer_num=%s", # 传输层数
            len(layer_mapping), # 层映射长度
        )
    except Exception: # 捕获异常
        logger.exception("attach_hybrid_dsa_pool_to_hiradix_cache failed") # 记录异常日志
        raise # 重新抛出异常


def attach_hybrid_pool_to_mamba_cache( # 将混合池栈附加到Mamba缓存
    mamba_cache: HiMambaRadixCache, # HiMamba前缀缓存
    params: CacheInitParams, # 缓存初始化参数
    server_args: ServerArgs, # 服务器参数
    *, # 仅限关键字参数
    extra_config: dict, # 额外配置
    prefetch_threshold: int, # 预取阈值
    load_cache_event, # 加载缓存事件
    enable_storage_metrics: bool = False, # 是否启用存储指标
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力上下文并行进程组（可选）
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力张量并行进程组（可选）
) -> None:
    """Attach HostPoolGroup (KV + Mamba) + HybridCacheController for HiMambaRadixCache.
    将HostPoolGroup（KV + Mamba）+ HybridCacheController附加到HiMambaRadixCache。

    This entrypoint is currently intended only for HiMambaRadixCache.
    此入口当前仅用于HiMambaRadixCache。
    """
    try: # 尝试执行
        hybrid_kv = mamba_cache.hybrid_kv_cache # 获取混合KV缓存
        kvcache = mamba_cache.kvcache # 获取KV缓存
        full_layer_mapping = dict(hybrid_kv.full_attention_layer_id_mapping) # 获取完整注意力层映射
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map) # 获取Mamba层映射
        host_pool_group, cache_controller = build_hybrid_mamba_stack( # 构建混合Mamba栈
            params=params, # 缓存初始化参数
            server_args=server_args, # 服务器参数
            kv_pool=kvcache, # KV设备池
            mamba_pool=params.req_to_token_pool.mamba_pool, # Mamba设备池
            full_layer_mapping=full_layer_mapping, # 完整层映射
            mamba_layer_mapping=mamba_layer_mapping, # Mamba层映射
            page_size=params.page_size, # 页面大小
            tp_group=params.tp_cache_group, # 张量并行缓存进程组
            load_cache_event=load_cache_event, # 加载缓存事件
            attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
            attn_tp_group=attn_tp_group, # 注意力张量并行进程组
            storage_backend=server_args.hicache_storage_backend, # 存储后端
            use_mla=hybrid_kv.use_mla, # 是否使用MLA
            host_mamba_evict_fn=mamba_cache.evict_mamba_host, # 主机Mamba驱逐函数
            device_mamba_evict_fn=mamba_cache.evict_mamba, # 设备Mamba驱逐函数
            prefetch_threshold=prefetch_threshold, # 预取阈值
            model_name=server_args.served_model_name, # 模型名称
            storage_backend_extra_config=extra_config, # 存储后端额外配置
            pp_rank=params.pp_rank, # 流水线并行排名
            pp_size=params.pp_size, # 流水线并行大小
            enable_storage_metrics=enable_storage_metrics, # 存储指标开关
        )
        mamba_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV) # 设置完整KV主机池
        mamba_cache.mamba_pool_host = host_pool_group.get_pool(PoolName.MAMBA) # 设置Mamba主机池
        mamba_cache.transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping) # 设置传输层数
        mamba_cache.host_pool_group = host_pool_group # 设置主机池组
        mamba_cache.cache_controller = cache_controller # 设置缓存控制器
        params.req_to_token_pool.register_layer_transfer_counter( # 注册req_to_token层级传输计数器
            cache_controller.layer_done_counter # 传入层级完成计数器
        )
        hybrid_kv.register_layer_transfer_counter(cache_controller.layer_done_counter) # 注册混合KV层级传输计数器
        logger.info( # 记录信息日志
            "Attached hybrid Mamba pool stack to HiMambaRadixCache: pools=KV + MAMBA, " # 已将混合Mamba池栈附加到HiMambaRadixCache
            "transfer_layer_num=%s", # 传输层数
            mamba_cache.transfer_layer_num, # 传输层数值
        )
    except Exception: # 捕获异常
        logger.exception("attach_hybrid_pool_to_mamba_cache failed") # 记录异常日志
        raise # 重新抛出异常
