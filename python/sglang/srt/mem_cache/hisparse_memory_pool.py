# HiSparse内存池模块 - 实现设备内存、主机内存和内存分配器的映射
# 本文件包含HiSparse DSA Token到KV池、HiSparse Token到KV池分配器、
# DeepSeekV4主机KV池以及DeepSeekV4 HiSparse Token到KV池分配器的实现

# mapping on device memory, host memory and memory allocator # 设备内存、主机内存和内存分配器的映射

import logging # 导入日志模块
import weakref # 导入弱引用模块
from typing import Optional # 导入可选类型

import psutil # 导入系统进程和内存工具
import torch # 导入PyTorch

from sglang.srt.layers.radix_attention import RadixAttention # 导入Radix注意力层
from sglang.srt.mem_cache.allocator import ( # 导入内存分配器基类
    BaseTokenToKVPoolAllocator, # Token到KV池的基础分配器
    PagedTokenToKVPoolAllocator, # 分页Token到KV池分配器
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import ( # 导入DeepSeek V4内存池
    DeepSeekV4TokenToKVPool, # DeepSeek V4 Token到KV池
    HiSparseC4DevicePool, # HiSparse C4设备池
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool # 导入DSA Token到KV池
from sglang.srt.mem_cache.memory_pool_host import HiSparseHostPoolMixin # 导入HiSparse主机池混入类
from sglang.srt.utils import is_cuda, is_hip # 导入CUDA和HIP检测工具
from sglang.srt.utils.common import get_num_new_pages # 导入获取新页面数工具

logger = logging.getLogger(__name__) # 获取当前模块的日志记录器

# sgl_kernel.kvcacheio is only available in CUDA/ROCm sgl-kernel builds (not XPU/MPS/NPU/CPU).
# sgl_kernel.kvcacheio仅在CUDA/ROCm的sgl-kernel构建中可用（不支持XPU/MPS/NPU/CPU）
_is_cuda = is_cuda() # 检测是否为CUDA环境
_is_hip = is_hip() # 检测是否为HIP环境
if _is_cuda or _is_hip: # 如果是CUDA或HIP环境
    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla # 导入全层MLA KV传输函数
else: # 否则（非CUDA/HIP环境）

    def transfer_kv_all_layer_mla(*args, **kwargs): # 定义占位函数，在非CUDA/HIP环境下抛出异常
        raise RuntimeError( # 抛出运行时错误
            "HiSparse device KV transfer requires sgl_kernel.kvcacheio (CUDA/ROCm). " # HiSparse设备KV传输需要sgl_kernel.kvcacheio（CUDA/ROCm）
            "It is not available on this backend." # 当前后端不可用
        )


class HiSparseDSATokenToKVPool(DSATokenToKVPool): # HiSparse DSA Token到KV池，继承自DSA Token到KV池
    def __init__( # 初始化方法
        self, # 自身实例
        size: int, # 池大小
        page_size: int, # 页面大小
        kv_lora_rank: int, # KV LoRA秩
        dtype: torch.dtype, # 数据类型
        qk_rope_head_dim: int, # QK旋转位置编码头维度
        layer_num: int, # 层数
        device: str, # 设备类型
        index_head_dim: int, # 索引头维度
        enable_memory_saver: bool, # 是否启用内存节省模式
        kv_cache_dim: int, # KV缓存维度
        start_layer: Optional[int] = None, # 起始层（可选）
        end_layer: Optional[int] = None, # 结束层（可选）
        host_to_device_ratio: int = 2, # 主机到设备比例，默认为2
    ):
        super().__init__( # 调用父类初始化
            size=size, # 传入池大小
            page_size=page_size, # 传入页面大小
            kv_lora_rank=kv_lora_rank, # 传入KV LoRA秩
            dtype=dtype, # 传入数据类型
            qk_rope_head_dim=qk_rope_head_dim, # 传入QK旋转头维度
            layer_num=layer_num, # 传入层数
            device=device, # 传入设备类型
            index_head_dim=index_head_dim, # 传入索引头维度
            enable_memory_saver=enable_memory_saver, # 传入内存节省开关
            kv_cache_dim=kv_cache_dim, # 传入KV缓存维度
            start_layer=start_layer, # 传入起始层
            end_layer=end_layer, # 传入结束层
            index_buf_size=size * host_to_device_ratio, # 索引缓冲区大小=池大小*主机到设备比例
        )
        self.bytes_per_token = self.kv_cache_dim * self.dtype.itemsize # 计算每个token的字节数

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor): # 注册全索引到HiSparse设备索引的映射
        self.full_to_hisparse_device_index_mapping = ( # 保存映射关系
            full_to_hisparse_device_index_mapping # 传入的映射张量
        )

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor): # 将压缩索引转换为HiSparse设备索引（转换为int32）
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to( # 通过映射表查找并转换类型
            torch.int32 # 转换为int32类型
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor): # 将压缩索引转换为HiSparse设备索引（保持原类型）
        return self.full_to_hisparse_device_index_mapping[compressed_indices] # 通过映射表查找，不改变类型

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor): # 将完整索引转换为HiSparse设备索引
        return self._translate_loc_to_hisparse_device(full_indices) # 委托给内部方法

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor): # 将完整索引转换为压缩索引（HiSparse DSA下不压缩，直接返回）
        return full_indices # 直接返回完整索引

    def set_kv_buffer( # 设置KV缓冲区
        self, # 自身实例
        layer: RadixAttention, # 注意力层
        loc: torch.Tensor, # 位置索引
        cache_k: torch.Tensor, # 缓存K张量
        cache_v: torch.Tensor, # 缓存V张量
    ):
        loc = self.translate_loc_to_hisparse_device(loc) # 将位置索引转换为HiSparse设备索引
        super().set_kv_buffer(layer, loc, cache_k, cache_v) # 调用父类方法设置KV缓冲区

    def set_mla_kv_buffer( # 设置MLA KV缓冲区
        self, # 自身实例
        layer: RadixAttention, # 注意力层
        loc: torch.Tensor, # 位置索引
        cache_k_nope: torch.Tensor, # 无位置编码的K缓存
        cache_k_rope: torch.Tensor, # 旋转位置编码的K缓存
    ):
        loc = self.translate_loc_to_hisparse_device(loc) # 将位置索引转换为HiSparse设备索引
        super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope) # 调用父类方法设置MLA KV缓冲区

    def get_mla_kv_buffer( # 获取MLA KV缓冲区
        self, # 自身实例
        layer: RadixAttention, # 注意力层
        loc: torch.Tensor, # 位置索引
        dst_dtype: Optional[torch.dtype] = None, # 目标数据类型（可选）
    ):
        loc = self.translate_loc_to_hisparse_device(loc) # 将位置索引转换为HiSparse设备索引
        return super().get_mla_kv_buffer(layer, loc, dst_dtype) # 调用父类方法获取MLA KV缓冲区

    def transfer_values_on_device(self, dst_indices, src_indices): # 在设备上传输KV值（从源索引到目标索引）
        transfer_kv_all_layer_mla( # 调用全层MLA KV传输函数
            src_layers=self.data_ptrs, # 源层的数据指针
            dst_layers=self.data_ptrs, # 目标层的数据指针
            src_indices=src_indices, # 源索引
            dst_indices=dst_indices, # 目标索引
            item_size=self.bytes_per_token, # 每个token的字节数
            num_layers=self.layer_num, # 层数
        )

    def get_cpu_copy(self, indices, mamba_indices=None): # 获取CPU副本（HiSparse设备池不支持）
        raise NotImplementedError("HiSparseDevicePool does not support get_cpu_copy") # 抛出未实现异常

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None): # 加载CPU副本（HiSparse设备池不支持）
        raise NotImplementedError("HiSparseDevicePool does not support load_cpu_copy") # 抛出未实现异常


class HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator): # HiSparse Token到KV池分配器，继承自基础分配器
    def __init__( # 初始化方法
        self, # 自身实例
        size: int, # HiSparse池大小
        page_size: int, # 页面大小
        dtype: torch.dtype, # 数据类型
        device: torch.device, # 设备
        kvcache: HiSparseDSATokenToKVPool, # KV缓存
        need_sort: bool, # 是否需要排序
        host_to_device_ratio: int = 2, # 主机到设备比例，默认为2
    ):
        self._kvcache = kvcache # 保存KV缓存引用
        self._size_full = size * host_to_device_ratio # 完整大小=HiSparse大小*主机到设备比例
        self._size_hisparse = size # HiSparse大小
        self.compress_ratio = 1 # 压缩比例（HiSparse DSA下为1，不压缩）
        self.dtype = dtype # 数据类型
        self.device = device # 设备
        self.page_size = page_size # 页面大小
        self.need_sort = need_sort # 是否需要排序

        self.logical_attn_allocator = PagedTokenToKVPoolAllocator( # 创建逻辑注意力分配器
            self._size_full, # 传入完整大小
            self.page_size, # 传入页面大小
            self.dtype, # 传入数据类型
            self.device, # 传入设备
            kvcache, # 传入KV缓存
            need_sort, # 传入排序标志
        )
        self.hisparse_attn_allocator = PagedTokenToKVPoolAllocator( # 创建HiSparse注意力分配器
            self._size_hisparse, # 传入HiSparse大小
            self.page_size, # 传入页面大小
            self.dtype, # 传入数据类型
            self.device, # 传入设备
            kvcache, # 传入KV缓存
            need_sort, # 传入排序标志
        )
        self.full_to_hisparse_device_index_mapping = torch.cat( # 创建完整索引到HiSparse设备索引的映射表
            [ # 拼接以下张量
                torch.zeros( # 零填充的映射区域
                    self._size_full + self.page_size, # 大小为完整大小+页面大小
                    dtype=torch.int64, # int64类型
                    device=self.device, # 在设备上
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device), # 末尾添加-1哨兵值
            ]
        )

        self.free_pages = None # 空闲页面列表
        self.release_pages = None # 释放页面列表
        self.is_not_in_free_group = True # 是否不在空闲组中
        self.free_group = [] # 空闲组列表
        self.clear() # 清空初始化
        self._kvcache.register_mapping( # 注册映射关系到KV缓存
            weakref.proxy(self.full_to_hisparse_device_index_mapping) # 使用弱引用代理映射表
        )

    @property
    def size_full(self) -> int: # 获取完整池大小的属性
        return self._size_full # 返回完整大小

    @property
    def size(self) -> int: # 获取池大小的属性
        return self._size_full # 返回完整大小（与size_full相同）

    def available_size(self) -> int: # 获取可用大小
        return min( # 返回逻辑分配器和HiSparse分配器可用大小的最小值
            self.logical_attn_allocator.available_size(), # 逻辑分配器可用大小
            self.hisparse_attn_allocator.available_size(), # HiSparse分配器可用大小
        )

    def get_kvcache(self): # 获取KV缓存
        return self._kvcache # 返回KV缓存引用

    def alloc(self, need_size: int): # 直接分配token（不支持，需使用alloc_extend或alloc_decode）
        raise NotImplementedError( # 抛出未实现异常
            "HiSparse allocator does not support direct token allocation; " # HiSparse分配器不支持直接token分配
            "use alloc_extend or alloc_decode instead." # 请使用alloc_extend或alloc_decode
        )

    def alloc_logical_only( # 仅分配逻辑索引，不分配HiSparse设备索引
        self, # 自身实例
        prefix_lens: torch.Tensor, # 前缀长度
        prefix_lens_cpu: torch.Tensor, # CPU上的前缀长度
        seq_lens: torch.Tensor, # 序列长度
        seq_lens_cpu: torch.Tensor, # CPU上的序列长度
        last_loc: torch.Tensor, # 上一个位置索引
        extend_num_tokens: int, # 扩展token数量
    ):
        """Allocate only logical indices without hisparse device indices.
        仅分配逻辑索引，不分配HiSparse设备索引。

        Used in the direct-to-host transfer path where KV data is written
        directly to host memory by the prefill node, skipping GPU staging.
        用于直接到主机的传输路径，KV数据由prefill节点直接写入主机内存，跳过GPU暂存。
        """
        return self.logical_attn_allocator.alloc_extend( # 委托给逻辑分配器进行扩展分配
            prefix_lens, # 传入前缀长度
            prefix_lens_cpu, # 传入CPU上的前缀长度
            seq_lens, # 传入序列长度
            seq_lens_cpu, # 传入CPU上的序列长度
            last_loc, # 传入上一个位置索引
            extend_num_tokens, # 传入扩展token数量
        )

    def alloc_device_buffer(self, allocated_indices, need_size: int): # 分配设备缓冲区，从已分配的索引中回收HiSparse设备索引
        assert need_size % self.page_size == 0 # 断言需要的大小是页面大小的整数倍
        # clear original reference and isolate the buffer from outside addressing, allocate new buffer if needed
        # 清除原始引用，将缓冲区与外部寻址隔离，如果需要则分配新缓冲区
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices] # 获取已分配索引对应的HiSparse索引
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0 # 将映射表中对应位置清零
        # Filter valid (non-zero) hisparse indices.
        # 过滤有效的（非零）HiSparse索引。
        # In the direct-to-host path, mapping is all zeros since no hisparse
        # device indices were pre-allocated.
        # 在直接到主机的路径中，映射全为零，因为没有预分配HiSparse设备索引。
        hisparse_indices = hisparse_indices[hisparse_indices > 0] # 过滤掉零值索引
        if len(hisparse_indices) >= need_size: # 如果有效索引数量足够
            buffer_indices = hisparse_indices[:need_size] # 取所需数量的索引
            self.free_hisparse_indices(hisparse_indices[need_size:]) # 释放多余的索引
        else: # 否则索引不够，需要额外分配
            # page alignment, claiming the residual space for an incomplete page
            # 页面对齐，为不完整页面声明剩余空间
            page_residual_length = len(hisparse_indices) % self.page_size # 计算页面残余长度
            if page_residual_length != 0: # 如果有残余
                hisparse_indices = torch.cat( # 补齐页面对齐
                    [ # 拼接
                        hisparse_indices, # 原有索引
                        torch.arange( # 补充连续索引
                            hisparse_indices[-1] + 1, # 从最后一个索引+1开始
                            hisparse_indices[-1] # 到最后一个索引
                            + self.page_size # 加上页面大小
                            - page_residual_length # 减去残余长度
                            + 1, # 加1（含端点）
                            device=self.device, # 在设备上
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc( # 从HiSparse分配器额外分配索引
                need_size - len(hisparse_indices) # 需要额外分配的数量
            )
            assert ( # 断言分配成功
                extra_indices is not None
            ), "Hisparse allocation failed in alloc_device_buffer" # HiSparse分配在alloc_device_buffer中失败
            buffer_indices = torch.cat([hisparse_indices, extra_indices]) # 合并现有索引和额外分配的索引
        return buffer_indices # 返回缓冲区索引

    def free_hisparse_indices(self, buffer_indices: torch.Tensor): # 释放HiSparse索引
        # disable free group mechanism for device buffer free
        # 禁用设备缓冲区释放的空闲组机制
        self.hisparse_attn_allocator.is_not_in_free_group = True # 确保不在空闲组中
        self.hisparse_attn_allocator.free(buffer_indices[buffer_indices > 0]) # 释放大于0的索引

    def get_last_loc_compressed(self, last_locs: torch.Tensor): # 获取压缩后的最后位置（HiSparse DSA下不压缩）
        return last_locs # 直接返回

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor): # 获取HiSparse设备上的最后位置
        return self._kvcache._translate_loc_to_hisparse_device(last_locs) # 委托给KV缓存的转换方法

    def alloc_extend( # 扩展分配（同时分配逻辑索引和HiSparse设备索引）
        self, # 自身实例
        prefix_lens: torch.Tensor, # 前缀长度
        prefix_lens_cpu: torch.Tensor, # CPU上的前缀长度
        seq_lens: torch.Tensor, # 序列长度
        seq_lens_cpu: torch.Tensor, # CPU上的序列长度
        last_loc: torch.Tensor,  # last_loc for full layers # 完整层的最后位置索引
        extend_num_tokens: int, # 扩展token数量
    ):
        assert self.page_size > 1 # 断言页面大小大于1

        num_new_pages = get_num_new_pages( # 计算需要的新页面数
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu # 传入序列长度、页面大小和前缀长度
        )
        if ( # 如果逻辑分配器没有足够的页面
            num_new_pages
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            return None # 返回None表示分配失败
        if ( # 如果HiSparse分配器没有足够的页面
            num_new_pages
            > self.hisparse_attn_allocator.available_size() // self.page_size
        ):
            return None # 返回None表示分配失败

        logical_indices = self.logical_attn_allocator.alloc_extend( # 从逻辑分配器分配扩展索引
            prefix_lens, # 传入前缀长度
            prefix_lens_cpu, # 传入CPU上的前缀长度
            seq_lens, # 传入序列长度
            seq_lens_cpu, # 传入CPU上的序列长度
            last_loc, # 传入上一个位置索引
            extend_num_tokens, # 传入扩展token数量
        )
        assert logical_indices is not None, "Logical allocation failed in alloc_extend" # 断言逻辑分配成功

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc) # 获取HiSparse设备上的最后位置
        hisparse_indices = self.hisparse_attn_allocator.alloc_extend( # 从HiSparse分配器分配扩展索引
            prefix_lens, # 传入前缀长度
            prefix_lens_cpu, # 传入CPU上的前缀长度
            seq_lens, # 传入序列长度
            seq_lens_cpu, # 传入CPU上的序列长度
            hisparse_last_loc, # 传入HiSparse最后位置
            len(logical_indices), # 传入逻辑索引数量
            num_new_pages=num_new_pages, # 传入新页面数
        )
        assert ( # 断言HiSparse分配成功
            hisparse_indices is not None
        ), "Hisparse allocation failed in alloc_extend" # HiSparse分配在alloc_extend中失败
        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices # 建立逻辑索引到HiSparse索引的映射
        return logical_indices # 返回逻辑索引

    def alloc_decode( # 解码分配（只需逻辑索引）
        self, # 自身实例
        seq_lens: torch.Tensor, # 序列长度
        seq_lens_cpu: torch.Tensor, # CPU上的序列长度
        last_loc: torch.Tensor,  # last_loc for full layers # 完整层的最后位置索引
    ):
        return self.logical_attn_allocator.alloc_decode( # 委托给逻辑分配器进行解码分配
            seq_lens, seq_lens_cpu, last_loc # 传入序列长度和最后位置
        )

    def free_hisparse(self, free_indices: torch.Tensor): # 释放HiSparse设备索引
        hisparse_indices = self._kvcache._translate_loc_to_hisparse_device(free_indices) # 将要释放的索引转换为HiSparse设备索引
        hisparse_indices = hisparse_indices[hisparse_indices > 0] # 过滤掉零值索引
        self.free_hisparse_indices(hisparse_indices) # 释放HiSparse索引
        self.full_to_hisparse_device_index_mapping[free_indices] = 0 # 将映射表中对应位置清零

    def clear(self): # 清空分配器状态
        self.logical_attn_allocator.clear() # 清空逻辑分配器
        self.hisparse_attn_allocator.clear() # 清空HiSparse分配器
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        # 注意：最后一项是-1哨兵值，不清除它，参见__init__中的注释
        self.full_to_hisparse_device_index_mapping[:-1].fill_(0) # 将映射表除最后一项外全部清零
        self.is_not_in_free_group = True # 重置空闲组标志
        self.free_group = [] # 清空空闲组

    def free_group_begin(self): # 开始空闲组（HiSparse DSA下不操作）
        return # 直接返回

    def free_group_end(self): # 结束空闲组（HiSparse DSA下不操作）
        return # 直接返回

    def free(self, free_index: torch.Tensor): # 释放索引
        if free_index.numel() == 0: # 如果没有要释放的索引
            return # 直接返回
        if self.is_not_in_free_group: # 如果不在空闲组中
            self.logical_attn_allocator.free(free_index) # 释放逻辑索引
            self.free_hisparse(free_index) # 释放HiSparse索引
        else: # 如果在空闲组中
            self.free_group.append(free_index) # 添加到空闲组延迟释放
        assert ( # 断言逻辑分配器可用大小不超过总大小
            self.logical_attn_allocator.available_size()
            <= self.logical_attn_allocator.size
        )
        assert ( # 断言HiSparse分配器可用大小不超过总大小
            self.hisparse_attn_allocator.available_size()
            <= self.hisparse_attn_allocator.size
        )


class DeepSeekV4SingleKVPoolHost(HiSparseHostPoolMixin): # DeepSeek V4单一KV池主机端实现，继承自HiSparse主机池混入类

    def __init__( # 初始化方法
        self, # 自身实例
        device_pool: HiSparseC4DevicePool, # 设备池
        host_size: int, # 主机池大小
        page_size: int, # 页面大小
        pin_memory: bool = True, # 是否锁定内存，默认为True
        device: str = "cpu", # 设备类型，默认为cpu
    ):

        assert host_size > 0, "Host size must be specified and greater than 0" # 断言主机大小必须大于0

        self.device_pool = device_pool # 保存设备池引用
        self.size = host_size # 保存主机池大小
        self.page_size = page_size # 保存页面大小
        self.num_pages = (self.size + self.page_size - 1) // self.page_size # 计算总页数（向上取整）
        self.size = self.num_pages * self.page_size # 对齐大小到页面整数倍
        self.pin_memory = pin_memory # 保存内存锁定标志
        self.device = device # 保存设备类型

        self.dtype = device_pool.store_dtype # 从设备池获取存储数据类型
        self.layer_num = device_pool.layer_num # 从设备池获取层数
        self.kv_cache_total_dim = device_pool.kv_cache_total_dim # 从设备池获取KV缓存总维度

        self.kv_buffer = self.init_kv_buffer() # 初始化KV缓冲区
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)] # 获取每层的数据引用
        self.data_ptrs = torch.tensor( # 创建数据指针张量
            [x.data_ptr() for x in self.data_refs], # 获取每层的数据指针
            dtype=torch.uint64, # uint64类型
            device=self.device_pool.device, # 在设备池的设备上
        )
        self.clear() # 清空初始化

    def clear(self): # 清空主机池
        self.free_slots = torch.arange( # 初始化空闲槽位
            1, self.size + 1, dtype=torch.int64, device="cpu" # 从1到size+1的连续整数
        )

    def init_kv_buffer(self): # 初始化KV缓冲区
        dims = (self.layer_num, self.size + self.page_size, self.kv_cache_total_dim) # 计算缓冲区维度
        requested_bytes = ( # 计算请求的字节数
            self.layer_num # 层数
            * (self.size + self.page_size) # 每层的大小+页面大小
            * self.kv_cache_total_dim # KV缓存总维度
            * self.dtype.itemsize # 每个元素的字节数
        )
        host_mem = psutil.virtual_memory() # 获取主机内存信息
        # preserve at least 10GB for other usage
        # 为其他用途保留至少10GB
        ten_gb = 10 * (1024**3) # 10GB的字节数
        available_bytes = host_mem.available - ten_gb # 可用字节数=实际可用-10GB保留
        if requested_bytes > available_bytes: # 如果请求超过可用内存
            raise ValueError( # 抛出值错误
                f"Not enough host memory available. Requesting " # 主机内存不足。请求
                f"{requested_bytes / 1e9:.2f} GB but only have " # GB但只有
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the " # GB可用。请减少
                f"size of the hierarchical cache." # 层次缓存的大小
            )
        else: # 否则内存充足
            logger.info( # 记录信息日志
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache." # 为层次KV缓存分配XX GB主机内存
            )

        host_pool = torch.empty(dims, dtype=self.dtype, device=self.device) # 创建空的主机池张量
        assert self.pin_memory, "DeepSeekV4SingleKVPoolHost requires pin_memory=True" # 断言必须启用内存锁定
        if self.pin_memory: # 如果启用内存锁定
            torch.cuda.cudart().cudaHostRegister( # 注册主机内存为页锁定内存
                host_pool.data_ptr(), host_pool.numel() * host_pool.element_size(), 0 # 传入数据指针、总字节数和标志
            )
        return host_pool # 返回主机池

    def backup_from_device_all_layer( # 从设备池备份所有层到主机池
        self, device_pool, host_indices, device_indices, io_backend="kernel" # 设备池、主机索引、设备索引、IO后端
    ):
        if io_backend != "kernel": # 如果IO后端不是kernel
            raise ValueError(f"Unsupported IO backend: {io_backend}") # 抛出不支持的IO后端异常

        from sglang.jit_kernel.dsv4 import hisparse_offload_to_host # 导入HiSparse卸载到主机的JIT内核

        if host_indices.device != device_indices.device: # 如果主机索引和设备索引不在同一设备上
            host_indices = host_indices.to(device=device_indices.device) # 将主机索引移到设备索引所在设备
        host_indices_i64 = ( # 将主机索引转换为int64
            host_indices.to(torch.int64) # 转换
            if host_indices.dtype != torch.int64 # 如果不是int64
            else host_indices # 否则保持不变
        )
        device_indices_i64 = ( # 将设备索引转换为int64
            device_indices.to(torch.int64) # 转换
            if device_indices.dtype != torch.int64 # 如果不是int64
            else device_indices # 否则保持不变
        )
        hisparse_offload_to_host( # 调用HiSparse卸载到主机内核
            gpu_ptrs=device_pool.data_ptrs, # GPU数据指针
            cpu_ptrs=self.data_ptrs, # CPU数据指针
            gpu_indices=device_indices_i64, # GPU索引
            cpu_indices=host_indices_i64, # CPU索引
        )

    def available_size(self): # 获取可用大小
        return len(self.free_slots) # 返回空闲槽位数

    def alloc(self, need_size: int) -> Optional[torch.Tensor]: # 分配指定大小的主机槽位
        if need_size > self.available_size(): # 如果需要的大小超过可用大小
            return None # 返回None表示分配失败

        select_index = self.free_slots[:need_size] # 从空闲槽位中选取前need_size个
        self.free_slots = self.free_slots[need_size:] # 更新空闲槽位列表

        return select_index # 返回选中的索引

    def free(self, indices: torch.Tensor) -> int: # 释放主机槽位
        self.free_slots = torch.cat([self.free_slots, indices.cpu()]) # 将释放的索引加回空闲槽位列表
        return len(indices) # 返回释放的数量


class DeepSeekV4HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator): # DeepSeek V4 HiSparse Token到KV池分配器，继承自基础分配器

    def __init__( # 初始化方法
        self, # 自身实例
        logical_attn_allocator: BaseTokenToKVPoolAllocator, # 逻辑注意力分配器
    ):
        assert isinstance(logical_attn_allocator._kvcache, DeepSeekV4TokenToKVPool) # 断言KV缓存为DeepSeek V4类型
        assert isinstance( # 断言C4 KV池为HiSparse C4设备池类型
            logical_attn_allocator._kvcache.c4_kv_pool, HiSparseC4DevicePool
        )
        self.compress_ratio = 4 # 压缩比例为4

        self.hisparse_kvcache = logical_attn_allocator._kvcache.c4_kv_pool # 保存HiSparse KV缓存引用
        self._size_full = logical_attn_allocator.size_full # 保存完整大小
        self._size_hisparse = self.hisparse_kvcache.size # 保存HiSparse大小

        self.dtype = self.hisparse_kvcache.dtype # 数据类型
        self.device = self.hisparse_kvcache.device # 设备
        self.page_size = self.hisparse_kvcache.page_size # 页面大小

        self.logical_attn_allocator = logical_attn_allocator # 保存逻辑分配器引用
        self._kvcache = logical_attn_allocator._kvcache # 保存KV缓存引用
        self.hisparse_attn_allocator = PagedTokenToKVPoolAllocator( # 创建HiSparse注意力分配器
            self._size_hisparse, # 传入HiSparse大小
            self.page_size, # 传入页面大小
            self.dtype, # 传入数据类型
            self.device, # 传入设备
            self.hisparse_kvcache, # 传入HiSparse KV缓存
            logical_attn_allocator.need_sort, # 传入排序标志
        )

        self.full_to_hisparse_device_index_mapping = torch.cat( # 创建完整索引到HiSparse设备索引的映射表
            [ # 拼接以下张量
                torch.zeros( # 零填充的映射区域
                    self._kvcache.c4_logical_size + self.page_size, # 大小为C4逻辑大小+页面大小
                    dtype=torch.int64, # int64类型
                    device=self.device, # 在设备上
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device), # 末尾添加-1哨兵值
            ]
        )

        self.need_sort = logical_attn_allocator.need_sort # 保存排序标志
        self.free_pages = None # 空闲页面列表
        self.release_pages = None # 释放页面列表
        self.is_not_in_free_group = True # 是否不在空闲组中
        self.free_group = [] # 空闲组列表
        self.clear() # 清空初始化

        self.hisparse_kvcache.register_mapping( # 注册映射关系到HiSparse KV缓存
            weakref.proxy(self.full_to_hisparse_device_index_mapping) # 使用弱引用代理映射表
        )

    @property
    def size_full(self) -> int: # 获取完整池大小的属性
        return self._size_full # 返回完整大小

    @property
    def size(self) -> int: # 获取池大小的属性
        return self.logical_attn_allocator.size # 返回逻辑分配器的大小

    @property
    def size_swa(self) -> int: # 获取滑动窗口注意力池大小的属性
        return self.logical_attn_allocator.size_swa # 返回逻辑分配器的SWA大小

    @property
    def full_to_swa_index_mapping(self): # 获取完整索引到SWA索引的映射
        return self.logical_attn_allocator.full_to_swa_index_mapping # 返回逻辑分配器的映射

    def debug_print(self) -> str: # 调试打印方法
        msg = self.logical_attn_allocator.debug_print() # 获取逻辑分配器的调试信息
        msg += ( # 追加HiSparse可用大小信息
            f"#hisparse-available-size: " # HiSparse可用大小
            f"{self.hisparse_attn_allocator.available_size()}, " # 具体数值
        )
        return msg # 返回调试信息

    def get_kvcache(self): # 获取KV缓存
        return self._kvcache # 返回KV缓存引用

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor): # 将完整索引转换为SWA索引
        return self.logical_attn_allocator.translate_loc_from_full_to_swa(kv_indices) # 委托给逻辑分配器

    def full_available_size(self): # 获取完整可用大小（考虑HiSparse压缩比）
        return min( # 返回逻辑分配器和HiSparse分配器（乘以压缩比）可用大小的最小值
            self.logical_attn_allocator.full_available_size(), # 逻辑分配器完整可用大小
            self.hisparse_attn_allocator.available_size() * self.compress_ratio, # HiSparse分配器可用大小*压缩比
        )

    def swa_available_size(self): # 获取SWA可用大小
        return self.logical_attn_allocator.swa_available_size() # 返回逻辑分配器的SWA可用大小

    def free_swa(self, free_indices: torch.Tensor): # 释放SWA索引
        self.logical_attn_allocator.free_swa(free_indices) # 委托给逻辑分配器释放

    def available_size(self) -> int: # 获取可用大小（考虑HiSparse压缩比）
        return min( # 返回逻辑分配器和HiSparse分配器（乘以压缩比）可用大小的最小值
            self.logical_attn_allocator.available_size(), # 逻辑分配器可用大小
            self.hisparse_attn_allocator.available_size() * self.compress_ratio, # HiSparse分配器可用大小*压缩比
        )

    def alloc(self, need_size: int): # 直接分配token（不支持）
        raise NotImplementedError( # 抛出未实现异常
            "DeepSeek V4 HiSparse allocator does not support direct token allocation; " # DeepSeek V4 HiSparse分配器不支持直接token分配
            "use alloc_extend or alloc_decode instead." # 请使用alloc_extend或alloc_decode
        )

    def alloc_device_buffer(self, allocated_indices, need_size: int): # 分配设备缓冲区
        assert need_size % self.page_size == 0 # 断言需要的大小是页面大小的整数倍
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices] # 获取已分配索引对应的HiSparse索引
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0 # 将映射表中对应位置清零

        device_buffer_size = need_size - self.page_size # 设备缓冲区大小=需要大小-一个页面
        P = len(hisparse_indices) # HiSparse索引数量
        if P > device_buffer_size + 1: # 如果索引数量超过缓冲区大小+1
            newest_src = hisparse_indices[P - 1].clone() # 克隆最新的源索引
            old_at_dbs = hisparse_indices[device_buffer_size].clone() # 克隆缓冲区位置的旧索引
            hisparse_indices[device_buffer_size] = newest_src # 将最新索引放到缓冲区位置
            hisparse_indices[P - 1] = old_at_dbs # 将旧索引放到最后位置

        if len(hisparse_indices) >= need_size: # 如果HiSparse索引足够
            buffer_indices = hisparse_indices[:need_size] # 取所需数量的索引
            surplus = hisparse_indices[need_size:] # 剩余索引
            if surplus.numel() > 0: # 如果有剩余索引
                buffer_pages = torch.unique(buffer_indices // self.page_size) # 缓冲区占用的页面
                surplus_pages = torch.unique(surplus // self.page_size) # 剩余索引占用的页面
                pure_surplus = surplus_pages[~torch.isin(surplus_pages, buffer_pages)] # 不与缓冲区重叠的纯剩余页面
                if pure_surplus.numel() > 0: # 如果有纯剩余页面
                    self.hisparse_attn_allocator.is_not_in_free_group = True # 确保不在空闲组中
                    self.hisparse_attn_allocator.free(pure_surplus * self.page_size) # 释放纯剩余页面
        else: # 否则索引不够，需要额外分配
            page_residual_length = len(hisparse_indices) % self.page_size # 计算页面残余长度
            if page_residual_length != 0: # 如果有残余
                hisparse_indices = torch.cat( # 补齐页面对齐
                    [ # 拼接
                        hisparse_indices, # 原有索引
                        torch.arange( # 补充连续索引
                            hisparse_indices[-1] + 1, # 从最后一个索引+1开始
                            hisparse_indices[-1] # 到最后一个索引
                            + self.page_size # 加上页面大小
                            - page_residual_length # 减去残余长度
                            + 1, # 加1（含端点）
                            device=self.device, # 在设备上
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc( # 从HiSparse分配器额外分配索引
                need_size - len(hisparse_indices) # 需要额外分配的数量
            )
            assert ( # 断言分配成功
                extra_indices is not None
            ), "Hisparse allocation failed in alloc_device_buffer" # HiSparse分配在alloc_device_buffer中失败
            buffer_indices = torch.cat([hisparse_indices, extra_indices]) # 合并现有索引和额外分配的索引
        return buffer_indices # 返回缓冲区索引

    def free_hisparse_indices(self, buffer_indices: torch.Tensor): # 释放HiSparse索引
        self.hisparse_attn_allocator.is_not_in_free_group = True # 确保不在空闲组中
        self.hisparse_attn_allocator.free(buffer_indices[buffer_indices > 0]) # 释放大于0的索引

    def get_last_loc_compressed(self, last_locs: torch.Tensor): # 获取压缩后的最后位置
        return (last_locs - 3) // self.compress_ratio # 根据压缩比计算压缩位置

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor): # 获取HiSparse设备上的最后位置
        return self.hisparse_kvcache._translate_loc_to_hisparse_device( # 委托给HiSparse KV缓存的转换方法
            self.get_last_loc_compressed(last_locs) # 先压缩再转换
        )

    def alloc_extend( # 扩展分配（同时分配逻辑索引和HiSparse设备索引）
        self, # 自身实例
        prefix_lens: torch.Tensor, # 前缀长度
        prefix_lens_cpu: torch.Tensor, # CPU上的前缀长度
        seq_lens: torch.Tensor, # 序列长度
        seq_lens_cpu: torch.Tensor, # CPU上的序列长度
        last_loc: torch.Tensor, # 最后位置索引
        extend_num_tokens: int, # 扩展token数量
    ):
        assert self.page_size > 1 # 断言页面大小大于1

        num_new_pages_logical = get_num_new_pages( # 计算逻辑层需要的新页面数
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu # 传入序列长度、页面大小和前缀长度
        )
        num_new_pages_hisparse = get_num_new_pages( # 计算HiSparse层需要的新页面数（考虑压缩比）
            seq_lens=seq_lens_cpu // self.compress_ratio, # 序列长度除以压缩比
            page_size=self.page_size, # 页面大小
            prefix_lens=prefix_lens_cpu // self.compress_ratio, # 前缀长度除以压缩比
        )
        if ( # 如果逻辑分配器没有足够的页面
            num_new_pages_logical
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            return None # 返回None表示分配失败
        if ( # 如果HiSparse分配器没有足够的页面
            num_new_pages_hisparse
            > self.hisparse_attn_allocator.available_size() // self.page_size
        ):
            return None # 返回None表示分配失败

        logical_indices = self.logical_attn_allocator.alloc_extend( # 从逻辑分配器分配扩展索引
            prefix_lens, # 传入前缀长度
            prefix_lens_cpu, # 传入CPU上的前缀长度
            seq_lens, # 传入序列长度
            seq_lens_cpu, # 传入CPU上的序列长度
            last_loc, # 传入上一个位置索引
            extend_num_tokens, # 传入扩展token数量
        )
        assert logical_indices is not None, "Logical allocation failed in alloc_extend" # 断言逻辑分配成功

        compressed_logical_indices = ( # 将逻辑索引转换为压缩索引
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(logical_indices) # 委托给HiSparse KV缓存转换
        )
        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc) # 获取HiSparse设备上的最后位置
        hisparse_indices = self.hisparse_attn_allocator.alloc_extend( # 从HiSparse分配器分配扩展索引
            prefix_lens // self.compress_ratio, # 传入前缀长度（除以压缩比）
            prefix_lens_cpu // self.compress_ratio, # 传入CPU上的前缀长度（除以压缩比）
            seq_lens // self.compress_ratio, # 传入序列长度（除以压缩比）
            seq_lens_cpu // self.compress_ratio, # 传入CPU上的序列长度（除以压缩比）
            hisparse_last_loc, # 传入HiSparse最后位置
            len(compressed_logical_indices), # 传入压缩索引数量
        )
        assert ( # 断言HiSparse分配成功
            hisparse_indices is not None
        ), "Hisparse allocation failed in alloc_extend" # HiSparse分配在alloc_extend中失败

        self.full_to_hisparse_device_index_mapping[compressed_logical_indices] = ( # 建立压缩索引到HiSparse索引的映射
            hisparse_indices.to(torch.int64) # 转换为int64类型
        )
        return logical_indices # 返回逻辑索引

    def alloc_decode( # 解码分配（只需逻辑索引）
        self, # 自身实例
        seq_lens: torch.Tensor, # 序列长度
        seq_lens_cpu: torch.Tensor, # CPU上的序列长度
        last_loc: torch.Tensor, # 最后位置索引
    ):
        return self.logical_attn_allocator.alloc_decode( # 委托给逻辑分配器进行解码分配
            seq_lens, seq_lens_cpu, last_loc # 传入序列长度和最后位置
        )

    def free_compressed(self, compressed_indices: torch.Tensor): # 释放压缩索引对应的HiSparse设备索引
        hisparse_indices = self.hisparse_kvcache.translate_loc_to_hisparse_device( # 将压缩索引转换为HiSparse设备索引
            compressed_indices # 传入压缩索引
        )
        hisparse_indices = hisparse_indices[hisparse_indices > 0] # 过滤掉零值索引
        self.free_hisparse_indices(hisparse_indices) # 释放HiSparse索引
        self.full_to_hisparse_device_index_mapping[compressed_indices] = 0 # 将映射表中对应位置清零

    def free_hisparse(self, free_indices: torch.Tensor): # 释放HiSparse索引（先将完整索引转换为压缩索引）
        compressed_indices = ( # 将完整索引转换为压缩索引
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(free_indices) # 委托给HiSparse KV缓存转换
        )
        self.free_compressed(compressed_indices) # 释放压缩索引

    def clear(self): # 清空分配器状态
        self.logical_attn_allocator.clear() # 清空逻辑分配器
        self.hisparse_attn_allocator.clear() # 清空HiSparse分配器

        self.full_to_hisparse_device_index_mapping[:-1].fill_(0) # 将映射表除最后一项外全部清零
        self.is_not_in_free_group = True # 重置空闲组标志
        self.free_group = [] # 清空空闲组

    def free(self, free_index: torch.Tensor): # 释放索引
        if free_index.numel() == 0: # 如果没有要释放的索引
            return # 直接返回

        if self.is_not_in_free_group: # 如果不在空闲组中
            self.logical_attn_allocator.free(free_index) # 释放逻辑索引
        else: # 如果在空闲组中
            self.free_group.append(free_index) # 添加到空闲组延迟释放
        assert ( # 断言逻辑分配器可用大小不超过总大小
            self.logical_attn_allocator.available_size()
            <= self.logical_attn_allocator.size
        )
        assert ( # 断言HiSparse分配器可用大小不超过总大小
            self.hisparse_attn_allocator.available_size()
            <= self.hisparse_attn_allocator.size
        )
