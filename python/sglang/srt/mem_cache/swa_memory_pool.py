# 文件说明：SWA（Sliding Window Attention）KV缓存内存池实现
# 本文件实现了混合注意力架构下的KV缓存池，将full attention和SWA attention的KV缓存分开管理，
# 并通过full_to_swa_index_mapping维护两者之间的索引映射关系，同时实现了对应的内存分配器。

import logging  # 日志记录库
from typing import Dict, List, Optional, Tuple  # 类型提示

import torch  # PyTorch深度学习框架

from sglang.srt.layers.radix_attention import RadixAttention  # 导入Radix注意力层
from sglang.srt.mem_cache.allocator import (  # 导入内存分配器
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool  # 导入SWA KV池基类
from sglang.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool  # 导入KV缓存和MHA池
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool  # 导入自定义内存池初始化工具
from sglang.srt.utils import is_npu  # 导入NPU检测工具
from sglang.srt.utils.common import get_num_new_pages  # 导入新页数量计算工具

_is_npu = is_npu()  # 检测是否为NPU环境

if _is_npu:  # 如果是NPU环境
    from sglang.srt.hardware_backend.npu.allocator_npu import (  # 导入NPU分页分配器
        NPUPagedTokenToKVPoolAllocator,
    )

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器
GB = 1024 * 1024 * 1024  # GB的字节数常量


class SWAKVPool(BaseSWAKVPool):  # SWA KV缓存池，分别管理full和SWA层的KV缓存
    """KV cache with separate pools for full and SWA attention layers."""  # 分离full和SWA注意力层的KV缓存池

    def __init__(  # 初始化方法
        self,
        size: int,  # full KV池大小
        size_swa: int,  # SWA KV池大小
        page_size: int,  # 页大小
        dtype: torch.dtype,  # 数据类型
        head_num: int,  # 注意力头数量
        head_dim: int,  # 注意力头维度
        swa_attention_layer_ids: List[int],  # SWA注意力层ID列表
        full_attention_layer_ids: List[int],  # full注意力层ID列表
        enable_kvcache_transpose: bool,  # 是否启用KV缓存转置
        device: str,  # 设备类型
        token_to_kv_pool_class: KVCache = MHATokenToKVPool,  # KV池类，默认为MHA
        **kwargs,  # 其他关键字参数
    ):
        self.size = size  # full池大小
        self.size_swa = size_swa  # SWA池大小
        self.dtype = dtype  # 数据类型
        self.head_num = head_num  # 注意力头数量
        self.head_dim = head_dim  # 注意力头维度
        self.device = device  # 设备类型
        self.swa_layer_nums = len(swa_attention_layer_ids)  # SWA层数量
        self.full_layer_nums = len(full_attention_layer_ids)  # full层数量
        self.layer_num = self.full_layer_nums + self.swa_layer_nums  # 总层数
        self.start_layer = 0  # 起始层ID
        self.page_size = page_size  # 页大小
        self.layer_transfer_counter = None  # 层传输计数器

        kwargs["page_size"] = page_size  # 传入页大小
        kwargs["enable_memory_saver"] = False  # 不启用内存节省模式
        kwargs["head_num"] = head_num  # 传入注意力头数量
        kwargs["head_dim"] = head_dim  # 传入注意力头维度
        kwargs["device"] = device  # 传入设备类型
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True  # TODO：如果enable_kvcache_transpose为True，应使用MHATransposedTokenToKVPool
        assert not enable_kvcache_transpose  # 断言暂不支持KV缓存转置

        # for disagg with nvlink  # 用于NVLink分离部署
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (  # 初始化自定义内存池
            maybe_init_custom_mem_pool(device=self.device)
        )

        self.swa_kv_pool = token_to_kv_pool_class(  # 创建SWA KV池
            size=size_swa,  # SWA池大小
            dtype=dtype,  # 数据类型
            layer_num=self.swa_layer_nums,  # SWA层数量
            **kwargs,  # 其他参数
        )
        kwargs.pop("swa_head_num", None)  # 移除SWA头数量参数
        kwargs.pop("swa_head_dim", None)  # 移除SWA头维度参数
        kwargs.pop("swa_v_head_dim", None)  # 移除SWA V头维度参数
        self.full_kv_pool = token_to_kv_pool_class(  # 创建full KV池
            size=size,  # full池大小
            dtype=dtype,  # 数据类型
            layer_num=self.full_layer_nums,  # full层数量
            **kwargs,  # 其他参数
        )
        # {layer_id: (index, is_swa_layer)}  # {层ID: (池内索引, 是否为SWA层)}
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}  # 层ID到池内索引的映射
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):  # 遍历full层
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)  # full层映射为(is_swa=False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):  # 遍历SWA层
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)  # SWA层映射为(is_swa=True)
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None  # full到SWA的索引映射
        self._cached_swa_loc: Optional[torch.Tensor] = None  # 缓存的SWA位置张量
        self._cached_loc_key: Optional[tuple] = None  # 缓存的位置键

        k_size, v_size = self.get_kv_size_bytes()  # 获取KV缓存总字节大小
        self.mem_usage = (k_size + v_size) / GB  # 计算内存使用量（GB）
        logger.info(
            f"SWAKVPool mem usage: {self.mem_usage:.2f} GB, swa size: {self.size_swa}, full size: {self.size}"  # 记录内存使用信息
        )

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):  # 注册full到SWA的索引映射
        self.full_to_swa_index_mapping = full_to_swa_index_mapping  # 保存映射张量
        self.invalidate_loc_cache()  # 使位置缓存失效

    def invalidate_loc_cache(self) -> None:  # 使位置缓存失效
        self._cached_swa_loc = None  # 清空缓存的SWA位置
        self._cached_loc_key = None  # 清空缓存的位置键

    def register_layer_transfer_counter(self, layer_transfer_counter):  # 注册层传输计数器
        # Wait happens at this wrapper. Inner pools must not wait again.  # 等待在此包装器中发生。内部池不得再次等待。
        self.layer_transfer_counter = layer_transfer_counter  # 保存层传输计数器
        self.full_kv_pool.register_layer_transfer_counter(None)  # full池不等待
        self.swa_kv_pool.register_layer_transfer_counter(None)  # SWA池不等待

    def _wait_for_layer(self, layer_id: int):  # 等待指定层的数据传输完成
        if self.layer_transfer_counter is not None:  # 如果有层传输计数器
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)  # 等待到指定层

    def get_kv_size_bytes(self):  # 获取KV缓存的总字节大小
        k_size, v_size = self.full_kv_pool.get_kv_size_bytes()  # 获取full池KV大小
        k_size_swa, v_size_swa = self.swa_kv_pool.get_kv_size_bytes()  # 获取SWA池KV大小
        return k_size + k_size_swa, v_size + v_size_swa  # 返回总K大小和总V大小

    def get_contiguous_buf_infos(self):  # 获取full KV池的连续缓冲区信息
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = (  # 获取full池数据指针、长度和元素长度
            self.full_kv_pool.get_contiguous_buf_infos()
        )
        return (
            full_kv_data_ptrs,  # full池数据指针
            full_kv_data_lens,  # full池数据长度
            full_kv_item_lens,  # full池元素长度
        )

    def get_state_buf_infos(self):  # 获取SWA KV池的状态缓冲区信息
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = (  # 获取SWA池数据指针、长度和元素长度
            self.swa_kv_pool.get_contiguous_buf_infos()
        )

        return swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens  # 返回SWA池缓冲区信息

    def get_key_buffer(self, layer_id: int):  # 获取指定层的K缓存缓冲区
        self._wait_for_layer(layer_id)  # 等待层传输完成
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]  # 获取池内层ID和是否为SWA层
        if is_swa_layer:  # 如果是SWA层
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)  # 从SWA池获取K缓冲区
        else:  # 如果是full层
            return self.full_kv_pool.get_key_buffer(layer_id_pool)  # 从full池获取K缓冲区

    def get_value_buffer(self, layer_id: int):  # 获取指定层的V缓存缓冲区
        self._wait_for_layer(layer_id)  # 等待层传输完成
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]  # 获取池内层ID和是否为SWA层
        if is_swa_layer:  # 如果是SWA层
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)  # 从SWA池获取V缓冲区
        else:  # 如果是full层
            return self.full_kv_pool.get_value_buffer(layer_id_pool)  # 从full池获取V缓冲区

    def get_kv_buffer(self, layer_id: int):  # 获取指定层的KV缓存缓冲区
        self._wait_for_layer(layer_id)  # 等待层传输完成
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]  # 获取池内层ID和是否为SWA层
        if is_swa_layer:  # 如果是SWA层
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)  # 从SWA池获取KV缓冲区
        else:  # 如果是full层
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)  # 从full池获取KV缓冲区

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:  # 将full索引转换为SWA索引
        assert self.full_to_swa_index_mapping is not None  # 断言映射已注册
        # data_ptr() (not untyped_storage().data_ptr()) encodes the offset, so  # data_ptr()（而非untyped_storage().data_ptr()）编码了偏移量，因此
        # views at different positions within the same storage get distinct keys.  # 同一存储中不同位置的视图会获得不同的键。
        # -1 in kv_indices maps to -1 via the sentinel appended to the mapping.  # kv_indices中的-1通过映射末尾的哨兵值映射为-1。
        key = (kv_indices.data_ptr(), kv_indices.numel())  # 构造缓存键（数据指针和元素数）
        if key != self._cached_loc_key:  # 如果缓存键不匹配
            if self._cached_loc_key is not None:  # 如果之前有缓存
                logger.debug(
                    "translate_loc_from_full_to_swa: loc tensor changed mid-forward "
                    "without invalidate_loc_cache() — possible missing call site"  # 转换full到SWA索引：loc张量在前向传播中间变化了，未调用invalidate_loc_cache()
                )
            self._cached_swa_loc = self.full_to_swa_index_mapping[kv_indices].to(  # 通过映射表转换索引
                torch.int32  # 转换为int32类型
            )
            self._cached_loc_key = key  # 更新缓存键
        return self._cached_swa_loc  # 返回缓存的SWA位置

    def set_kv_buffer(  # 设置指定层的KV缓存数据
        self,
        layer: RadixAttention,  # 注意力层对象
        loc: torch.Tensor,  # 位置索引张量
        cache_k: torch.Tensor,  # K缓存数据
        cache_v: torch.Tensor,  # V缓存数据
        k_scale: float = 1.0,  # K缩放因子
        v_scale: float = 1.0,  # V缩放因子
    ):

        layer_id = layer.layer_id  # 获取层ID
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]  # 获取池内层ID和是否为SWA层
        if is_swa_layer:  # 如果是SWA层
            loc = self.translate_loc_from_full_to_swa(loc)  # 将full索引转换为SWA索引
            self.swa_kv_pool.set_kv_buffer(  # 在SWA池中设置KV缓存
                None,  # 层对象为None（使用layer_id_override）
                loc,  # SWA位置索引
                cache_k,  # K缓存
                cache_v,  # V缓存
                k_scale,  # K缩放
                v_scale,  # V缩放
                layer_id_override=layer_id_pool,  # 覆盖层ID
            )
        else:  # 如果是full层
            self.full_kv_pool.set_kv_buffer(  # 在full池中设置KV缓存
                None,  # 层对象为None（使用layer_id_override）
                loc,  # full位置索引
                cache_k,  # K缓存
                cache_v,  # V缓存
                k_scale,  # K缩放
                v_scale,  # V缩放
                layer_id_override=layer_id_pool,  # 覆盖层ID
            )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):  # 移动KV缓存数据
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)  # 在full池中移动
        tgt_loc_swa = self.translate_loc_from_full_to_swa(tgt_loc)  # 转换目标位置到SWA索引
        src_loc_swa = self.translate_loc_from_full_to_swa(src_loc)  # 转换源位置到SWA索引
        self.swa_kv_pool.move_kv_cache(tgt_loc_swa, src_loc_swa)  # 在SWA池中移动

    def _filter_swa_cpu_copy(self, swa_kv_cpu, row_mask: torch.Tensor):  # 根据行掩码过滤SWA的CPU拷贝数据
        if swa_kv_cpu is None:  # 如果SWA CPU数据为None
            return None  # 返回None
        if row_mask is None or bool(torch.all(row_mask).item()):  # 如果掩码为None或全部为True
            return swa_kv_cpu  # 返回原始数据

        chunk_size = getattr(  # 获取分块大小
            self.swa_kv_pool, "cpu_offloading_chunk_size", len(row_mask)  # 默认为行掩码长度
        )
        filtered = []  # 过滤后的结果列表
        for layer_chunks in swa_kv_cpu:  # 遍历每层的分块数据
            if len(layer_chunks) == 0:  # 如果该层无数据
                filtered.append([])  # 添加空列表
                continue  # 继续

            k_cpu = torch.cat([chunk[0] for chunk in layer_chunks], dim=0)  # 拼接所有K分块
            v_cpu = torch.cat([chunk[1] for chunk in layer_chunks], dim=0)  # 拼接所有V分块
            k_cpu = k_cpu[row_mask]  # 根据掩码过滤K数据
            v_cpu = v_cpu[row_mask]  # 根据掩码过滤V数据

            filtered_layer = []  # 过滤后的分块列表
            for i in range(0, len(k_cpu), chunk_size):  # 按分块大小遍历
                filtered_layer.append(
                    [k_cpu[i : i + chunk_size], v_cpu[i : i + chunk_size]]  # 添加过滤后的K-V分块
                )
            filtered.append(filtered_layer)  # 添加该层的过滤结果
        return filtered  # 返回过滤后的数据

    def get_cpu_copy(self, indices, mamba_indices=None):  # 获取KV缓存的CPU拷贝
        # For SWA, we need to copy KV cache from both full and SWA pools  # 对于SWA，需要从full和SWA池都拷贝KV缓存
        # The indices are for the full pool, and we use mapping to get SWA indices  # 索引是针对full池的，使用映射获取SWA索引
        full_kv_cpu = self.full_kv_pool.get_cpu_copy(indices)  # 从full池获取CPU拷贝

        swa_mask = None  # SWA掩码初始化为None
        if self.full_to_swa_index_mapping is not None:  # 如果映射已注册
            swa_indices = self.full_to_swa_index_mapping[indices]  # 获取对应的SWA索引
            # Slot 0 is reserved as a dummy slot. Tail-only SWA allocations leave  # 槽位0保留为虚拟槽位。仅尾部的SWA分配会使
            # the out-of-window full KV indices unmapped, so only copy mapped SWA  # 窗口外的full KV索引未被映射，因此仅拷贝已映射的SWA
            # tokens and keep their positions for load_cpu_copy().  # token并保持其位置供load_cpu_copy()使用。
            swa_mask = swa_indices > 0  # 构造SWA掩码（大于0表示已映射）
            if torch.any(swa_mask):  # 如果有已映射的SWA索引
                swa_kv_cpu = self.swa_kv_pool.get_cpu_copy(swa_indices[swa_mask])  # 获取映射的SWA CPU拷贝
                swa_mask = swa_mask.cpu()  # 将掩码移到CPU
            else:  # 没有已映射的SWA索引
                swa_kv_cpu = None  # SWA CPU数据为None
        else:  # 映射未注册
            swa_kv_cpu = None  # SWA CPU数据为None

        return {"full": full_kv_cpu, "swa": swa_kv_cpu, "swa_mask": swa_mask}  # 返回包含full、swa和掩码的字典

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):  # 从CPU拷贝加载KV缓存
        # Load KV cache back from CPU to both full and SWA pools  # 将KV缓存从CPU加载回full和SWA池
        # Note: indices here are NEW indices (newly allocated), different from get_cpu_copy indices  # 注意：此处的索引是新索引（新分配的），与get_cpu_copy的索引不同
        full_kv_cpu = kv_cache_cpu["full"]  # 获取full KV的CPU数据
        swa_kv_cpu = kv_cache_cpu["swa"]  # 获取SWA KV的CPU数据

        # Load full KV cache to the new indices  # 将full KV缓存加载到新索引
        self.full_kv_pool.load_cpu_copy(full_kv_cpu, indices)  # 加载full KV缓存

        # Load SWA KV cache if it exists  # 如果SWA KV缓存存在则加载
        if swa_kv_cpu is not None and self.full_to_swa_index_mapping is not None:  # 检查SWA数据和映射
            swa_indices = self.full_to_swa_index_mapping[indices]  # 获取新索引对应的SWA索引
            new_swa_mask = swa_indices > 0  # 构造新SWA掩码
            old_swa_mask = kv_cache_cpu.get("swa_mask")  # 获取原始SWA掩码
            if old_swa_mask is not None:  # 如果有原始掩码
                old_swa_mask = old_swa_mask.to(indices.device)  # 将原始掩码移到对应设备
                row_mask = new_swa_mask[old_swa_mask].cpu()  # 计算行掩码
                swa_indices = swa_indices[old_swa_mask][row_mask.to(indices.device)]  # 过滤SWA索引
            else:  # 没有原始掩码
                row_mask = new_swa_mask.cpu()  # 使用新掩码作为行掩码
                swa_indices = swa_indices[new_swa_mask]  # 过滤SWA索引

            if swa_indices.numel() == 0:  # 如果没有需要加载的SWA索引
                return  # 直接返回

            swa_kv_cpu = self._filter_swa_cpu_copy(swa_kv_cpu, row_mask)  # 过滤SWA CPU数据
            self.swa_kv_pool.load_cpu_copy(swa_kv_cpu, swa_indices)  # 加载SWA KV缓存


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):  # SWA混合KV缓存的内存分配器
    """Allocator for SWA hybrid KV cache."""  # SWA混合KV缓存的分配器

    def __init__(  # 初始化方法
        self,
        size: int,  # full池大小
        size_swa: int,  # SWA池大小
        page_size: int,  # 页大小
        dtype: torch.dtype,  # 数据类型
        device: str,  # 设备类型
        kvcache: BaseSWAKVPool,  # KV缓存池对象
        need_sort: bool,  # 是否需要排序
    ):
        assert isinstance(kvcache, BaseSWAKVPool)  # 断言kvcache是BaseSWAKVPool实例
        self._size_full = size  # full池大小
        self._size_swa = size_swa  # SWA池大小
        self.dtype = dtype  # 数据类型
        self.device = device  # 设备类型
        self.page_size = page_size  # 页大小

        full_kv_pool = getattr(kvcache, "full_kv_pool", None)  # 获取full KV池
        swa_kv_pool = getattr(kvcache, "swa_kv_pool", None)  # 获取SWA KV池

        if page_size == 1:  # 如果页大小为1（不分页）
            self.full_attn_allocator = TokenToKVPoolAllocator(  # 创建full注意力分配器
                size,  # 大小
                dtype,  # 数据类型
                device,  # 设备
                full_kv_pool,  # KV池
                need_sort,  # 是否排序
            )
            self.swa_attn_allocator = TokenToKVPoolAllocator(  # 创建SWA注意力分配器
                size_swa,  # 大小
                dtype,  # 数据类型
                device,  # 设备
                swa_kv_pool,  # KV池
                need_sort,  # 是否排序
            )
        else:  # 分页模式
            if _is_npu:  # 如果是NPU环境
                PagedTokenToKVPoolAllocatorClass = NPUPagedTokenToKVPoolAllocator  # 使用NPU分页分配器
            else:  # 非NPU环境
                PagedTokenToKVPoolAllocatorClass = PagedTokenToKVPoolAllocator  # 使用标准分页分配器
            self.full_attn_allocator = PagedTokenToKVPoolAllocatorClass(  # 创建full分页分配器
                size,  # 大小
                page_size,  # 页大小
                dtype,  # 数据类型
                device,  # 设备
                full_kv_pool,  # KV池
                need_sort,  # 是否排序
            )
            self.swa_attn_allocator = PagedTokenToKVPoolAllocatorClass(  # 创建SWA分页分配器
                size_swa,  # 大小
                page_size,  # 页大小
                dtype,  # 数据类型
                device,  # 设备
                swa_kv_pool,  # KV池
                need_sort,  # 是否排序
            )
        # Note: append one more item of value -1 in the end so -1 maps to -1.  # 注意：在末尾追加一个值为-1的元素，使得-1映射为-1。
        # It is needed for the last_loc in alloc_extend, where the first full_last_loc  # 这是alloc_extend中last_loc所必需的，其中第一个full_last_loc
        # is -1, and we need to map it to swa_last_loc -1 as well.  # 为-1，我们需要将其映射为swa_last_loc -1。
        self.full_to_swa_index_mapping = torch.cat(  # 创建full到SWA的索引映射张量
            [
                torch.zeros(  # 初始化为零（0表示未映射）
                    size + self.page_size,  # 大小加page_size
                    dtype=torch.int64,  # int64类型
                    device=device,  # 设备
                ),
                torch.tensor([-1], dtype=torch.int64, device=device),  # 末尾哨兵值-1
            ]
        )

        self.need_sort = need_sort  # 是否需要排序
        self.free_pages = None  # 空闲页
        self.release_pages = None  # 释放页
        self.is_not_in_free_group = True  # 是否不在空闲组中
        self.free_group = []  # 空闲组列表

        self._kvcache = kvcache  # 保存KV缓存池引用
        self.clear()  # 清空分配器
        self._kvcache.register_mapping(self.full_to_swa_index_mapping)  # 在KV池中注册映射

    def available_size(self):  # 获取可用大小（取full和SWA中的较小值）
        return min(
            self.full_attn_allocator.available_size(),  # full分配器可用大小
            self.swa_attn_allocator.available_size(),  # SWA分配器可用大小
        )

    def full_available_size(self):  # 获取full分配器的可用大小
        return self.full_attn_allocator.available_size()  # 返回full分配器可用大小

    def swa_available_size(self):  # 获取SWA分配器的可用大小
        return self.swa_attn_allocator.available_size()  # 返回SWA分配器可用大小

    @property
    def size(self):  # 总可用大小属性
        return min(self._size_full, self._size_swa)  # 取full和SWA中的较小值

    @property
    def size_swa(self):  # SWA池大小属性
        return self._size_swa  # 返回SWA池大小

    @property
    def size_full(self):  # full池大小属性
        return self._size_full  # 返回full池大小

    def debug_print(self) -> str:  # 打印调试信息
        msg = ""  # 初始化消息
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "  # SWA可用大小
        msg += (
            f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "  # full可用大小
        )
        return msg  # 返回调试消息

    def get_kvcache(self):  # 获取KV缓存池对象
        return self._kvcache  # 返回KV缓存池引用

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):  # 将full索引转换为SWA索引
        assert self._kvcache.full_to_swa_index_mapping is not None  # 断言映射已注册
        return self._kvcache.translate_loc_from_full_to_swa(kv_indices)  # 委托给KV池转换

    def alloc(self, need_size: int):  # 分配指定数量的token空间（不分页模式）
        self._kvcache.invalidate_loc_cache()  # 使位置缓存失效
        assert self.page_size == 1  # 断言页大小为1
        if need_size > self.full_attn_allocator.available_size():  # 如果full分配器空间不足
            return None  # 返回None
        if need_size > self.swa_attn_allocator.available_size():  # 如果SWA分配器空间不足
            return None  # 返回None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size)  # 从full分配器分配
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)  # 从SWA分配器分配
        assert alloc_full_indices is not None  # 断言full分配成功
        assert alloc_swa_indices is not None  # 断言SWA分配成功

        if _is_npu:  # 如果是NPU环境
            self.full_to_swa_index_mapping[alloc_full_indices.to(torch.int64)] = (  # 更新映射（NPU需要转int64）
                alloc_swa_indices.to(torch.int64)
            )
        else:  # 非NPU环境
            self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices  # 更新映射
        return alloc_full_indices  # 返回full分配的索引

    def alloc_extend(  # 分配扩展token空间（分页模式）
        self,
        prefix_lens: torch.Tensor,  # 前缀长度张量
        prefix_lens_cpu: torch.Tensor,  # 前缀长度CPU张量
        seq_lens: torch.Tensor,  # 序列长度张量
        seq_lens_cpu: torch.Tensor,  # 序列长度CPU张量
        last_loc: torch.Tensor,  # last_loc for full layers  # full层的上一个位置
        extend_num_tokens: int,  # 扩展token数量
    ):
        self._kvcache.invalidate_loc_cache()  # 使位置缓存失效
        assert self.page_size > 1  # 断言分页模式

        num_new_pages = get_num_new_pages(  # 计算需要的新页数
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu  # 传入序列长度、页大小和前缀长度
        )
        if num_new_pages > self.full_attn_allocator.available_size() // self.page_size:  # full空间不足
            return None  # 返回None
        if num_new_pages > self.swa_attn_allocator.available_size() // self.page_size:  # SWA空间不足
            return None  # 返回None

        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)  # 转换last_loc到SWA索引

        alloc_full_indices = self.full_attn_allocator.alloc_extend(  # 从full分配器分配扩展页
            prefix_lens,  # 前缀长度
            prefix_lens_cpu,  # 前缀长度CPU
            seq_lens,  # 序列长度
            seq_lens_cpu,  # 序列长度CPU
            last_loc,  # 上一个位置
            extend_num_tokens,  # 扩展token数
            num_new_pages=num_new_pages,  # 新页数
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(  # 从SWA分配器分配扩展页
            prefix_lens,  # 前缀长度
            prefix_lens_cpu,  # 前缀长度CPU
            seq_lens,  # 序列长度
            seq_lens_cpu,  # 序列长度CPU
            swa_last_loc,  # SWA上一个位置
            extend_num_tokens,  # 扩展token数
            num_new_pages=num_new_pages,  # 新页数
        )
        assert alloc_full_indices is not None  # 断言full分配成功
        assert alloc_swa_indices is not None  # 断言SWA分配成功

        if _is_npu:  # 如果是NPU环境
            self.full_to_swa_index_mapping[alloc_full_indices.to(torch.int64)] = (  # 更新映射（NPU）
                alloc_swa_indices.to(torch.int64)
            )
        else:  # 非NPU环境
            self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices  # 更新映射

        return alloc_full_indices  # 返回full分配的索引

    def alloc_extend_swa_tail(  # 分配扩展token，SWA仅分配尾部部分
        self,
        prefix_lens: torch.Tensor,  # 前缀长度张量
        prefix_lens_cpu: torch.Tensor,  # 前缀长度CPU张量
        seq_lens: torch.Tensor,  # 序列长度张量
        seq_lens_cpu: torch.Tensor,  # 序列长度CPU张量
        last_loc: torch.Tensor,  # last_loc for full layers  # full层的上一个位置
        extend_num_tokens: int,  # 扩展token数量
        swa_tail_len: int,  # SWA尾部长度
    ):
        self._kvcache.invalidate_loc_cache()  # 使位置缓存失效
        """Allocate full KV for the whole extend and SWA KV only for the tail.  # 为整个扩展分配full KV，但SWA KV仅分配尾部

        This is used by disaggregated decode preallocation: decode receives full  # 用于分离式解码预分配：解码接收完整的
        prompt KV for full-attention layers, but only the sliding-window state is  # full注意力层的提示KV，但仅传输滑动窗口状态用于
        transferred for SWA layers.  # SWA层。
        """
        assert self.page_size > 1  # 断言分页模式
        assert len(seq_lens_cpu) == 1, "SWA tail allocation currently supports bs=1"  # 断言仅支持batch_size=1
        assert len(prefix_lens_cpu) == 1  # 断言前缀长度为1
        assert 0 <= swa_tail_len <= extend_num_tokens  # 断言SWA尾部长度合法

        num_full_pages = get_num_new_pages(  # 计算full需要的新页数
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu  # 传入序列长度、页大小和前缀长度
        )
        num_swa_pages = (swa_tail_len + self.page_size - 1) // self.page_size  # 计算SWA需要的新页数（向上取整）
        if num_full_pages > self.full_attn_allocator.available_size() // self.page_size:  # full空间不足
            return None  # 返回None
        if num_swa_pages > self.swa_attn_allocator.available_size() // self.page_size:  # SWA空间不足
            return None  # 返回None

        alloc_full_indices = self.full_attn_allocator.alloc_extend(  # 从full分配器分配全部扩展
            prefix_lens,  # 前缀长度
            prefix_lens_cpu,  # 前缀长度CPU
            seq_lens,  # 序列长度
            seq_lens_cpu,  # 序列长度CPU
            last_loc,  # 上一个位置
            extend_num_tokens,  # 扩展token数
        )
        assert alloc_full_indices is not None  # 断言full分配成功

        if swa_tail_len == 0:  # 如果SWA尾部长度为0
            return alloc_full_indices  # 直接返回full索引

        device = self.device  # 获取设备
        swa_prefix_lens = torch.zeros((1,), dtype=torch.int64, device=device)  # SWA前缀长度为0
        swa_prefix_lens_cpu = torch.zeros((1,), dtype=torch.int64)  # SWA前缀长度CPU为0
        swa_seq_lens = torch.tensor([swa_tail_len], dtype=torch.int64, device=device)  # SWA序列长度等于尾部长度
        swa_seq_lens_cpu = torch.tensor([swa_tail_len], dtype=torch.int64)  # SWA序列长度CPU
        swa_last_loc = torch.tensor([-1], dtype=torch.int64, device=device)  # SWA上一个位置为-1

        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(  # 从SWA分配器分配尾部
            swa_prefix_lens,  # SWA前缀长度
            swa_prefix_lens_cpu,  # SWA前缀长度CPU
            swa_seq_lens,  # SWA序列长度
            swa_seq_lens_cpu,  # SWA序列长度CPU
            swa_last_loc,  # SWA上一个位置
            swa_tail_len,  # SWA尾部长度
        )
        assert alloc_swa_indices is not None  # 断言SWA分配成功

        self.full_to_swa_index_mapping[alloc_full_indices[-swa_tail_len:]] = (  # 映射尾部full索引到SWA索引
            alloc_swa_indices
        )
        if swa_tail_len < extend_num_tokens:  # 如果尾部小于总扩展长度
            self.full_to_swa_index_mapping[alloc_full_indices[:-swa_tail_len]] = 0  # 非尾部映射为0（未映射）
        return alloc_full_indices  # 返回full分配的索引

    def alloc_decode(  # 分配解码阶段的token空间
        self,
        seq_lens: torch.Tensor,  # 序列长度张量
        seq_lens_cpu: torch.Tensor,  # 序列长度CPU张量
        last_loc: torch.Tensor,  # last_loc for full layers  # full层的上一个位置
    ):
        self._kvcache.invalidate_loc_cache()  # 使位置缓存失效
        assert self.page_size > 1  # 断言分页模式
        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)  # 转换last_loc到SWA索引

        alloc_full_indices = self.full_attn_allocator.alloc_decode(  # 从full分配器分配解码页
            seq_lens, seq_lens_cpu, last_loc  # 传入序列长度和上一个位置
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_decode(  # 从SWA分配器分配解码页
            seq_lens, seq_lens_cpu, swa_last_loc  # 传入序列长度和SWA上一个位置
        )

        if alloc_full_indices is None or alloc_swa_indices is None:  # 如果任一分配失败
            return None  # 返回None

        if _is_npu:  # 如果是NPU环境
            self.full_to_swa_index_mapping[alloc_full_indices.to(torch.int64)] = (  # 更新映射（NPU）
                alloc_swa_indices.to(torch.int64)
            )
        else:  # 非NPU环境
            self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices  # 更新映射

        return alloc_full_indices  # 返回full分配的索引

    def free(self, free_index: torch.Tensor):  # 释放指定索引的KV缓存空间
        if free_index.numel() == 0:  # 如果没有要释放的索引
            return  # 直接返回

        # NOTE: the API is not idempotent.  # 注意：此API不是幂等的。
        if self.is_not_in_free_group:  # 如果不在空闲组中
            self.full_attn_allocator.free(free_index)  # 从full分配器释放
            self.free_swa(free_index)  # 从SWA分配器释放
        else:  # 在空闲组中
            self.free_group.append(free_index)  # 添加到空闲组延迟释放
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size  # 断言full可用大小不超过总大小
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size  # 断言SWA可用大小不超过总大小

    def set_full_to_swa_mapping(  # 设置full到SWA的索引映射
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor  # full索引和SWA索引
    ) -> None:
        """Write full_to_swa_index_mapping[full_indices[i]] = swa_indices[i].  # 写入 full_to_swa_index_mapping[full_indices[i]] = swa_indices[i]。

        Used by HiCache load-back path to rebuild the mapping after FULL and SWA device alloc.  # 由HiCache加载回路径使用，在FULL和SWA设备分配后重建映射。
        """
        if full_indices.numel() == 0:  # 如果没有需要映射的索引
            return  # 直接返回
        assert full_indices.numel() == swa_indices.numel()  # 断言两者长度一致
        self._kvcache.invalidate_loc_cache()  # 使位置缓存失效
        if _is_npu:  # 如果是NPU环境
            self.full_to_swa_index_mapping[full_indices.to(torch.int64)] = (  # 更新映射（NPU）
                swa_indices.to(torch.int64)
            )
        else:  # 非NPU环境
            self.full_to_swa_index_mapping[full_indices] = swa_indices  # 更新映射

    def free_swa(self, free_index: torch.Tensor):  # 仅释放SWA侧的KV缓存空间
        self._kvcache.invalidate_loc_cache()  # 使位置缓存失效
        swa_indices = self.full_to_swa_index_mapping[free_index]  # 获取对应的SWA索引
        swa_indices = swa_indices[swa_indices > 0]  # 过滤掉未映射的索引（<=0）
        self.swa_attn_allocator.free(swa_indices)  # 从SWA分配器释放
        self.full_to_swa_index_mapping[free_index] = 0  # 重置映射为0（未映射）

    def backup_state(self):  # 备份分配器状态
        return [  # 返回状态列表
            self.full_attn_allocator.backup_state(),  # full分配器状态
            self.swa_attn_allocator.backup_state(),  # SWA分配器状态
        ]

    def restore_state(self, state):  # 恢复分配器状态
        assert len(state) == 2  # 断言状态列表长度为2
        self.full_attn_allocator.restore_state(state[0])  # 恢复full分配器状态
        self.swa_attn_allocator.restore_state(state[1])  # 恢复SWA分配器状态

    def clear(self):  # 清空分配器
        self._kvcache.invalidate_loc_cache()  # 使位置缓存失效
        self.swa_attn_allocator.clear()  # 清空SWA分配器
        self.full_attn_allocator.clear()  # 清空full分配器
        # Note: the last item is -1, we don't clear it, see the comment in __init__  # 注意：最后一项是-1，不清除它，参见__init__中的注释
        self.full_to_swa_index_mapping[:-1].fill_(0)  # 将除哨兵值外的映射重置为0
        self.is_not_in_free_group = True  # 重置空闲组标志
        self.free_group = []  # 清空空闲组

    def get_cpu_copy(self, indices, mamba_indices=None):  # 获取KV缓存的CPU拷贝
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)  # 委托给KV池获取

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):  # 从CPU拷贝加载KV缓存
        return self._kvcache.load_cpu_copy(  # 委托给KV池加载
            kv_cache_cpu, indices, mamba_indices=mamba_indices  # 传入CPU数据、索引和mamba索引
        )
