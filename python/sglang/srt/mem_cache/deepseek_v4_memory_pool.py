# DeepSeek V4内存池模块
# 实现DeepSeek V4模型的多级KV缓存内存池，包括SWA（滑动窗口注意力）、C4（4倍压缩）、C128（128倍压缩）和索引器池
# 支持FP8量化存储、FlashMLA布局、在线压缩以及分层稀疏（HiSparse）设备池

from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
from contextlib import nullcontext  # 导入空上下文管理器
from typing import List, Literal, NamedTuple, Optional, Tuple  # 导入类型提示工具

import torch  # 导入PyTorch张量库

from sglang.jit_kernel.dsv4 import fused_k_norm_rope_flashmla, fused_store_cache  # 导入DSV4融合内核
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE  # 导入GPU内存类型常量
from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.layers.attention.dsa import index_buf_accessor  # 导入DSA索引缓冲区访问器
from sglang.srt.layers.attention.dsv4 import (  # 导入DSV4索引缓冲区访问器
    index_buf_accessor as dsv4_index_buf_accessor,
)
from sglang.srt.layers.attention.dsv4.index_buf_accessor import NopeFp8RopeBf16Pack  # 导入FP8-Rope-BF16打包结构
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool  # 导入SWA KV池基类
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool  # 导入压缩状态池
from sglang.srt.mem_cache.memory_pool import KVCache  # 导入KV缓存基类
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数
from sglang.srt.utils import ceil_div, is_hip  # 导入向上取整除法和HIP平台检测

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

_is_hip = is_hip()  # 检测当前是否为AMD HIP平台

ONLINE_C128 = not _is_hip and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()  # 是否启用在线C128压缩


def get_compress_state_ring_size(  # 获取压缩状态的环形缓冲区大小
    compress_ratio: int, is_speculative: bool = False  # 压缩比率和是否为推测解码模式
) -> int:
    assert compress_ratio in [4, 128], f"Unsupported {compress_ratio = }"  # 断言压缩比率合法
    # Online c128 keeps a single (max, sum, kv) state per index instead of a
    # 128-slot ring buffer of raw tokens, so ring_size collapses to 1. Online
    # is incompatible with speculative decode for now.
    # 在线C128为每个索引保留单个(max, sum, kv)状态，而非128槽的原始token环形缓冲区，
    # 因此ring_size折叠为1。在线模式目前与推测解码不兼容。
    if compress_ratio == 128 and ONLINE_C128:  # C128比率且启用了在线压缩
        assert not is_speculative, "online c128 does not support MTP"  # 在线C128不支持MTP
        return 1  # 在线模式下环形大小为1
    if is_speculative:  # 推测解码模式
        return 16 if compress_ratio == 4 else 256  # C4返回16，C128返回256
    else:  # 普通模式
        return 8 if compress_ratio == 4 else 128  # C4返回8，C128返回128


class DeepSeekV4SingleKVPool(KVCache):  # DeepSeek V4单层KV缓存池
    def __init__(  # 初始化单层KV缓存池
        self,
        size: int,  # 缓存池大小
        page_size: int,  # 页面大小
        dtype: torch.dtype,  # 数据类型
        qk_nope_head_dim: int,  # QK无旋转位置编码头维度
        qk_rope_head_dim: int,  # QK旋转位置编码头维度
        layer_num: int,  # 层数
        device: str,  # 设备类型
        enable_memory_saver: bool,  # 是否启用内存节省
        start_layer: Optional[int] = None,  # 起始层ID
        end_layer: Optional[int] = None,  # 结束层ID
    ):
        super().__init__(  # 调用父类KVCache的初始化
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.qk_nope_head_dim = qk_nope_head_dim  # 保存无RoPE头维度
        self.qk_rope_head_dim = qk_rope_head_dim  # 保存RoPE头维度

        self.scale_pad = 1  # 量化缩放填充值
        self.quantize_block_size = 64  # 量化块大小
        self.rope_storage_dtype = torch.bfloat16  # RoPE存储数据类型
        self.k_with_scale_buffer_dtype = torch.int8  # 带缩放的K缓冲区数据类型
        self._create_buffers()  # 创建缓冲区

    def _create_buffers(self):  # 创建KV缓冲区
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):  # 进入KV缓存内存区域
            with (  # 使用自定义内存池分配
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool  # 如果有自定义内存池
                else nullcontext()  # 否则使用空上下文
            ):
                self.kv_buffer = [  # 为每一层创建KV缓冲区
                    self.create_buffer(  # 创建单层缓冲区
                        num_pages=(self.size + self.page_size + 1) // self.page_size,  # 计算页数
                    )
                    for _ in range(self.layer_num)  # 遍历每一层
                ]

    def get_bytes_per_token(self) -> int:  # 获取每个token的字节数
        dim_per_token = (  # 计算每个token的维度
            self.qk_nope_head_dim  # 无RoPE维度（FP8存储）
            + self.qk_rope_head_dim * self.rope_storage_dtype.itemsize  # RoPE维度（BF16存储）
            + self.qk_nope_head_dim // self.quantize_block_size  # 量化缩放因子数
            + self.scale_pad  # 缩放填充
        )
        return dim_per_token  # 返回每token字节数

    def create_buffer(self, *, num_pages: int):  # 创建单个层的KV缓冲区
        bytes_per_token = self.get_bytes_per_token()  # 获取每token字节数
        self.kv_cache_total_dim = bytes_per_token  # 保存KV缓存总维度
        bytes_per_page_non_padded = self.page_size * bytes_per_token  # 计算未填充的每页字节数
        self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576  # 对齐到576字节的倍数

        assert bytes_per_token == 448 + 64 * 2 + 8, (  # 断言字节数符合DSV4布局
            "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 "
            "(64*2) + nope FP8 scales + scale_pad = 584 bytes/token"
        )
        assert self.store_dtype == torch.uint8  # 断言存储类型为uint8

        return torch.zeros(  # 返回零初始化的缓冲区张量
            num_pages,  # 页数
            self.bytes_per_page_padded,  # 每页填充后字节数
            dtype=self.store_dtype,  # 存储类型
            device=self.device,  # 设备
        )

    def set_key_buffer(  # 设置键缓冲区
        self,
        layer_id: int,  # 层ID
        loc: torch.Tensor,  # 位置索引张量
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,  # FP8-Rope-BF16打包数据
    ):
        dsv4_index_buf_accessor.SetKAndS.execute(  # 调用DSV4索引缓冲区设置方法
            pool=self,  # 内存池自身
            buf=self.kv_buffer[layer_id],  # 目标层的KV缓冲区
            loc=loc,  # 位置索引
            nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,  # 打包数据
        )

    def set_key_buffer_fused(  # 融合设置键缓冲区（使用JIT内核）
        self,
        layer_id: int,  # 层ID
        loc: torch.Tensor,  # 位置索引张量
        cache_k: torch.Tensor,  # 缓存K张量
    ) -> None:
        return fused_store_cache(  # 调用融合存储缓存内核
            input=cache_k,  # 输入K张量
            cache=self.kv_buffer[layer_id],  # 目标层的KV缓冲区
            indices=loc,  # 位置索引
            page_size=self.page_size,  # 页面大小
            type="flashmla",  # 类型为FlashMLA
        )

    def get_key_buffer(self, layer_id: int):  # 获取键缓冲区
        if self.store_dtype != self.dtype:  # 如果存储类型与数据类型不同
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)  # 以数据类型视图返回

        return self.kv_buffer[layer_id]  # 直接返回原始缓冲区

    def set_kv_buffer(self, *args, **kwargs) -> None:  # 设置KV缓冲区（未实现）
        raise NotImplementedError()  # 抛出未实现异常

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:  # 获取值缓冲区（未实现）
        raise NotImplementedError("Use get_key_buffer instead.")  # 提示使用get_key_buffer

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:  # 获取KV缓冲区（未实现）
        raise NotImplementedError("Use get_key_buffer instead.")  # 提示使用get_key_buffer


class HiSparseC4DevicePool(DeepSeekV4SingleKVPool):  # 分层稀疏C4设备池

    def __init__(  # 初始化分层稀疏C4设备池
        self,
        size: int,  # 缓存池大小
        page_size: int,  # 页面大小
        dtype: torch.dtype,  # 数据类型
        qk_nope_head_dim: int,  # QK无RoPE头维度
        qk_rope_head_dim: int,  # QK RoPE头维度
        layer_num: int,  # 层数
        device: str,  # 设备类型
        enable_memory_saver: bool,  # 是否启用内存节省
        start_layer: int | None = None,  # 起始层ID
        end_layer: int | None = None,  # 结束层ID
    ):
        super().__init__(  # 调用父类初始化
            size,
            page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.data_ptrs = torch.tensor(  # 创建每层缓冲区的数据指针张量
            [x.data_ptr() for x in self.kv_buffer],  # 获取每层缓冲区的数据指针
            dtype=torch.uint64,  # 指针类型为uint64
            device=self.device,  # 设备
        )
        self.compress_ratio = 4  # 压缩比率为4

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):  # 注册全量到稀疏设备的索引映射
        self.full_to_hisparse_device_index_mapping = (  # 保存映射表
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):  # 将全量索引转换为压缩索引
        mask = (full_indices + 1) % self.compress_ratio == 0  # 筛选压缩边界位置的索引
        compressed_indices = full_indices[mask] // self.compress_ratio  # 计算压缩后的索引
        return compressed_indices  # 返回压缩索引

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):  # 将压缩索引转换为稀疏设备索引
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32  # 转换为int32类型
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):  # 内部方法：转换索引到稀疏设备（不改变数据类型）
        return self.full_to_hisparse_device_index_mapping[compressed_indices]  # 直接使用映射表查找

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):  # 将全量索引直接转换为稀疏设备索引
        return self._translate_loc_to_hisparse_device(
            self.translate_loc_from_full_to_compressed(full_indices)  # 先压缩再映射
        )

    def set_key_buffer(  # 设置键缓冲区（重写，添加索引转换）
        self,
        layer_id: int,  # 层ID
        loc: torch.Tensor,  # 位置索引张量
        cache_nope_fp8_rope_bf16_pack,  # FP8-Rope-BF16打包数据
    ):
        loc = self.translate_loc_to_hisparse_device(loc)  # 将位置索引转换为稀疏设备索引
        super().set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)  # 调用父类方法

    def set_key_buffer_fused(  # 融合设置键缓冲区（重写，添加索引转换）
        self,
        layer_id: int,  # 层ID
        loc: torch.Tensor,  # 位置索引张量
        cache_k: torch.Tensor,  # 缓存K张量
    ) -> None:
        loc = self.translate_loc_to_hisparse_device(loc)  # 将位置索引转换为稀疏设备索引
        return super().set_key_buffer_fused(layer_id, loc, cache_k)  # 调用父类方法

    def get_cpu_copy(self, indices, mamba_indices=None):  # 获取CPU副本（不支持）
        raise NotImplementedError("HiSparseC4DevicePool does not support get_cpu_copy")  # 抛出不支持异常

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):  # 加载CPU副本（不支持）
        raise NotImplementedError("HiSparseC4DevicePool does not support load_cpu_copy")  # 抛出不支持异常


class DeepSeekV4IndexerPool(KVCache):  # DeepSeek V4索引器KV缓存池
    quant_block_size = 128  # 量化块大小
    index_k_with_scale_buffer_dtype = torch.uint8  # 索引K带缩放缓冲区数据类型

    def __init__(  # 初始化索引器KV缓存池
        self,
        size: int,  # 缓存池大小
        page_size: int,  # 页面大小
        dtype: torch.dtype,  # 数据类型
        index_head_dim: int,  # 索引头维度
        layer_num: int,  # 层数
        device: str,  # 设备类型
        enable_memory_saver: bool,  # 是否启用内存节省
        start_layer: Optional[int] = None,  # 起始层ID
        end_layer: Optional[int] = None,  # 结束层ID
    ):
        super().__init__(  # 调用父类KVCache的初始化
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.index_head_dim = index_head_dim  # 保存索引头维度

        self._create_buffer()  # 创建索引器缓冲区

    def _create_buffer(self):  # 创建索引器缓冲区
        num_scales_per_token = self.index_head_dim // self.quant_block_size  # 计算每token的缩放因子数
        page_bytes = self.page_size * self.index_head_dim  # 计算每页索引K字节数
        page_bytes += self.page_size * num_scales_per_token * 4  # 加上每页缩放因子字节数
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):  # 进入KV缓存内存区域
            with (  # 使用自定义内存池分配
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool  # 如果有自定义内存池
                else nullcontext()  # 否则使用空上下文
            ):
                self.index_k_with_scale_buffer = [  # 为每一层创建索引K带缩放缓冲区
                    torch.zeros(  # 零初始化张量
                        (self.size + self.page_size + 1) // self.page_size,  # 页数
                        page_bytes,  # 每页字节数
                        dtype=self.index_k_with_scale_buffer_dtype,  # 数据类型
                        device=self.device,  # 设备
                    )
                    for _ in range(self.layer_num)  # 遍历每一层
                ]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:  # 获取KV缓冲区（未实现）
        raise NotImplementedError()  # 抛出未实现异常

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:  # 获取键缓冲区（未实现）
        raise NotImplementedError()  # 抛出未实现异常

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:  # 获取值缓冲区（未实现）
        raise NotImplementedError()  # 抛出未实现异常

    def set_kv_buffer(self, *args, **kwargs) -> None:  # 设置KV缓冲区（未实现）
        raise NotImplementedError()  # 抛出未实现异常

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:  # 获取索引K带缩放缓冲区
        return self.index_k_with_scale_buffer[layer_id]  # 返回指定层的缓冲区

    def get_index_k_scale_buffer(  # 获取索引K和缩放缓冲区
        self,
        layer_id: int,  # 层ID
        seq_len: int,  # 序列长度
        page_indices: torch.Tensor,  # 页面索引张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.index_k_with_scale_buffer[layer_id]  # 获取指定层的缓冲区
        return index_buf_accessor.GetKAndS.execute(  # 调用索引缓冲区访问器获取K和缩放
            self, buf, seq_len=seq_len, page_indices=page_indices  # 传递参数
        )

    def set_index_k_scale_buffer(  # 设置索引K和缩放缓冲区
        self,
        layer_id: int,  # 层ID
        loc: torch.Tensor,  # 位置索引张量
        index_k: torch.Tensor,  # 索引K张量
        index_k_scale: torch.Tensor,  # 索引K缩放张量
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]  # 获取指定层的缓冲区
        index_buf_accessor.SetKAndS.execute(  # 调用索引缓冲区访问器设置K和缩放
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale  # 传递参数
        )

    def set_index_fused(  # 融合设置索引（使用JIT内核）
        self,
        layer_id: int,  # 层ID
        loc: torch.Tensor,  # 位置索引张量
        cache_k: torch.Tensor,  # 缓存K张量
    ) -> None:
        return fused_store_cache(  # 调用融合存储缓存内核
            input=cache_k,  # 输入K张量
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],  # 目标层的缓冲区
            indices=loc,  # 位置索引
            page_size=self.page_size,  # 页面大小
            type="indexer",  # 类型为索引器
        )


class DeepSeekV4LayerItem(NamedTuple):  # DeepSeek V4层项命名元组
    compress_ratio: Literal[0, 4, 128]  # 压缩比率：0（无压缩）、4或128
    compress_layer_id: int  # 压缩层ID
    compress_kv_pool: Optional[DeepSeekV4SingleKVPool] = None  # 压缩KV池（可选）


class DeepSeekV4TokenToKVPool(BaseSWAKVPool):  # DeepSeek V4 Token到KV池的映射管理器

    def __init__(  # 初始化Token到KV池映射管理器
        self,
        max_num_reqs: int,  # 最大请求数
        swa_size: int,  # SWA大小
        c4_size: int,  # C4压缩大小
        c128_size: int,  # C128压缩大小
        c4_state_pool_size: int,  # C4状态池大小
        c128_state_pool_size: int,  # C128状态池大小
        page_size: int,  # 页面大小
        swa_page_size: int,  # SWA页面大小
        dtype: torch.dtype,  # 数据类型
        state_dtype: torch.dtype,  # 状态数据类型
        qk_nope_head_dim: int,  # QK无RoPE头维度
        qk_rope_head_dim: int,  # QK RoPE头维度
        indexer_head_dim: int,  # 索引器头维度
        layer_num: int,  # 层数
        device: str,  # 设备类型
        enable_memory_saver: bool,  # 是否启用内存节省
        compression_ratios: List[int],  # 每层压缩比率列表
        start_layer: Optional[int] = None,  # 起始层ID
        end_layer: Optional[int] = None,  # 结束层ID
        enable_hisparse: bool = False,  # 是否启用分层稀疏
    ):
        super().__init__(  # 调用父类BaseSWAKVPool的初始化
            swa_size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        c4_logical_size = c128_size * 32  # 计算C4逻辑大小

        logger.info(  # 记录初始化信息
            "Initialize DeepSeekV4TokenToKVPool with "
            f"{max_num_reqs=} {swa_size=} {c4_size=} "
            f"{c4_logical_size=} {c128_size=} "
            f"{c4_state_pool_size=} {c128_state_pool_size=}"
        )

        self.max_num_reqs = max_num_reqs  # 保存最大请求数
        self.c4_size = c4_size  # 保存C4大小
        self.c4_logical_size = c4_logical_size  # 保存C4逻辑大小
        self.c128_size = c128_size  # 保存C128大小
        self.c4_state_pool_size = c4_state_pool_size  # 保存C4状态池大小
        self.c128_state_pool_size = c128_state_pool_size  # 保存C128状态池大小
        self.state_dtype = state_dtype  # 保存状态数据类型
        self.compression_ratios = compression_ratios  # 保存压缩比率列表

        # Determine this PP stage's absolute layer range
        # 确定此流水线并行阶段的绝对层范围
        if (
            start_layer is not None
            and end_layer is not None
            and len(compression_ratios) >= end_layer
        ):
            self._stage_start = start_layer  # 设置阶段起始层
            self._stage_end = end_layer  # 设置阶段结束层
        else:
            self._stage_start = 0  # 默认从第0层开始
            self._stage_end = len(compression_ratios)  # 默认到最后一层
        stage_ratios = compression_ratios[self._stage_start : self._stage_end]  # 获取当前阶段的压缩比率

        assert page_size % swa_page_size == 0  # 断言page_size是swa_page_size的倍数

        self.swa_size = swa_size  # 保存SWA大小
        self.swa_window_size = swa_page_size  # 保存SWA窗口大小
        self.swa_page_size = swa_page_size  # 保存SWA页面大小
        self.scale_pad = 1  # 保存缩放填充值

        self.qk_nope_head_dim = qk_nope_head_dim  # 保存无RoPE头维度
        self.qk_rope_head_dim = qk_rope_head_dim  # 保存RoPE头维度
        self.indexer_head_dim = indexer_head_dim  # 保存索引器头维度

        c4_layer_num = sum(1 for r in stage_ratios if r == 4)  # 计算C4层数
        c128_layer_num = sum(1 for r in stage_ratios if r == 128)  # 计算C128层数
        c4_page_size = page_size // 4  # 计算C4页面大小
        c128_page_size = page_size // 128  # 计算C128页面大小
        self.swa_kv_pool = DeepSeekV4SingleKVPool(  # 创建SWA KV池
            swa_size,
            swa_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        )

        c4_kv_pool_type = DeepSeekV4SingleKVPool  # 默认C4 KV池类型
        if enable_hisparse:  # 如果启用分层稀疏
            c4_kv_pool_type = HiSparseC4DevicePool  # 使用分层稀疏C4设备池
        self.c4_kv_pool = c4_kv_pool_type(  # 创建C4 KV池
            c4_size,
            c4_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
        )

        self.c128_kv_pool = DeepSeekV4SingleKVPool(  # 创建C128 KV池
            c128_size,
            c128_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            c128_layer_num,
            device,
            enable_memory_saver,
        )

        indexer_size = (  # 计算索引器大小
            self.c4_logical_size
            if (not _is_hip or envs.SGLANG_OPT_USE_COMPRESSOR_V2.get())  # HIP平台V2压缩器或非HIP
            else c4_size  # HIP平台V1压缩器
        )
        self.c4_indexer_kv_pool = DeepSeekV4IndexerPool(  # 创建C4索引器KV池
            indexer_size,
            c4_page_size,
            dtype,
            indexer_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
        )

        self._init_compressed_layer_mapping()  # 初始化压缩层映射

        if _is_hip:  # AMD HIP平台
            self._init_paged_compress_states(False)  # 不启用内存节省
        else:  # NVIDIA CUDA平台
            self._init_paged_compress_states(enable_memory_saver)  # 根据配置决定是否启用

        self._should_cache_swa = envs.SGLANG_OPT_CACHE_SWA_TRANSLATION.get()  # 是否缓存SWA位置转换
        self.cached_loc = None  # 缓存的位置索引

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):  # 注册全量到SWA的索引映射
        self.full_to_swa_index_mapping = full_to_swa_index_mapping  # 保存映射表
        self.cached_loc = None  # mapping replaced; discard any cached translation  # 映射已替换；丢弃任何缓存的转换

    def invalidate_loc_cache(self) -> None:  # 使位置缓存失效
        self.cached_loc = None  # 清空缓存的位置索引

    def get_ring_size(self, compress_ratio: int) -> int:  # 获取指定压缩比率的环形缓冲区大小
        server_args = get_global_server_args()  # 获取全局服务器参数
        is_speculative = server_args.speculative_algorithm is not None  # 判断是否为推测解码模式
        return get_compress_state_ring_size(compress_ratio, is_speculative)  # 返回对应的环形大小

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):  # 将全量索引转换为SWA索引
        assert self.full_to_swa_index_mapping is not None  # 断言映射表已初始化

        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)  # 使用映射表查找并转为int32

    def get_cached_swa_loc(self, raw_loc: torch.Tensor, layer_id: int) -> torch.Tensor:  # 获取缓存的SWA位置（带缓存优化）
        if self._should_cache_swa:  # 如果启用SWA位置缓存
            if layer_id == self.start_layer or self.cached_loc is None:  # 首层或缓存不存在时重新计算
                self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)  # 计算并缓存
            return self.cached_loc  # 返回缓存结果
        return self.translate_loc_from_full_to_swa(raw_loc)  # 不缓存则每次重新计算

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:  # 获取连续缓冲区信息（指针、长度、项长度）
        data_ptrs: List[int] = []  # 数据指针列表
        data_lens: List[int] = []  # 数据长度列表
        item_lens: List[int] = []  # 项长度列表

        for bufs in [  # 遍历三组缓冲区
            self.c4_kv_pool.kv_buffer,  # C4 KV缓冲区
            self.c4_indexer_kv_pool.index_k_with_scale_buffer,  # C4索引器缓冲区
            self.c128_kv_pool.kv_buffer,  # C128 KV缓冲区
        ]:
            for buf in bufs:  # 遍历每层的缓冲区
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"  # 断言为2D缓冲区
                data_ptrs.append(buf.data_ptr())  # 添加数据指针
                data_lens.append(buf.nbytes)  # 添加数据长度
                item_lens.append(buf[0].nbytes)  # 添加项长度

        return data_ptrs, data_lens, item_lens  # 返回三组信息

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:  # 获取状态缓冲区信息（指针、长度、项长度）
        data_ptrs: List[int] = []  # 数据指针列表
        data_lens: List[int] = []  # 数据长度列表
        item_lens: List[int] = []  # 项长度列表

        for buf in self.swa_kv_pool.kv_buffer:  # 遍历SWA KV缓冲区
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"  # 断言为2D缓冲区
            data_ptrs.append(buf.data_ptr())  # 添加数据指针
            data_lens.append(buf.nbytes)  # 添加数据长度
            item_lens.append(buf[0].nbytes)  # 添加项长度

        for pools in [  # 遍历两组状态池
            self.compress_state_pools,  # 压缩状态池
            self.indexer_compress_state_pools,  # 索引器压缩状态池
        ]:
            for pool in pools:  # 遍历每层的状态池
                if pool is None:  # 跳过空池
                    continue
                t = pool.kv_score_buffer.kv_score  # 获取kv_score张量
                assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"  # 断言为2D缓冲区
                data_ptrs.append(t.data_ptr())  # 添加数据指针
                data_lens.append(t.nbytes)  # 添加数据长度
                item_lens.append(t[0].nbytes * pool.ring_size)  # 添加项长度（乘以环形大小）

        return data_ptrs, data_lens, item_lens  # 返回三组信息

    def _init_paged_compress_states(self, enable_memory_saver: bool):  # 初始化分页压缩状态池
        c4_state_pool_size = self.c4_state_pool_size  # 获取C4状态池大小
        c128_state_pool_size = self.c128_state_pool_size  # 获取C128状态池大小
        total_L = len(self.compression_ratios)  # 获取总层数
        self.compress_state_pools: List[Optional[CompressStatePool]] = [None] * total_L  # 初始化压缩状态池列表
        self.indexer_compress_state_pools: List[Optional[CompressStatePool]] = [
            None
        ] * total_L  # 初始化索引器压缩状态池列表

        for idx in range(self._stage_start, self._stage_end):  # 遍历当前阶段的层
            ratio = self.compression_ratios[idx]  # 获取该层的压缩比率
            if ratio == 0:  # 无压缩层跳过
                continue
            overlap = ratio == 4  # C4层启用重叠
            size = c4_state_pool_size if ratio == 4 else c128_state_pool_size  # 根据比率选择大小
            ring_size = self.get_ring_size(ratio)  # 获取环形缓冲区大小

            self.compress_state_pools[idx] = CompressStatePool(  # 创建压缩状态池
                size=size,
                ring_size=ring_size,
                overlap=overlap,
                head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,  # 注意力头总维度
                dtype=self.state_dtype,  # 状态数据类型
                device=self.device,  # 设备
                enable_memory_saver=enable_memory_saver,  # 内存节省标志
                ratio=ratio,  # 压缩比率
                online=(ratio == 128 and ONLINE_C128),  # C128且启用在线压缩
                swa_page_size=self.swa_page_size,  # SWA页面大小
            )

            if ratio == 4:  # C4层还需要索引器状态池
                self.indexer_compress_state_pools[idx] = CompressStatePool(  # 创建索引器压缩状态池
                    size=size,
                    ring_size=ring_size,
                    overlap=overlap,
                    head_dim=self.indexer_head_dim,  # 索引器头维度
                    device=self.device,  # 设备
                    dtype=self.state_dtype,  # 状态数据类型
                    enable_memory_saver=enable_memory_saver,  # 内存节省标志
                    ratio=ratio,  # 压缩比率
                    swa_page_size=self.swa_page_size,  # SWA页面大小
                )

    def _init_compressed_layer_mapping(self):  # 初始化压缩层映射
        c1_cnt = c4_cnt = c128_cnt = 0  # 初始化三种层的计数器
        total_L = len(self.compression_ratios)  # 获取总层数
        self.layer_mapping: List[Optional[DeepSeekV4LayerItem]] = [None] * total_L  # 初始化层映射列表

        for idx in range(self._stage_start, self._stage_end):  # 遍历当前阶段的层
            ratio = self.compression_ratios[idx]  # 获取该层的压缩比率
            if ratio == 0:  # 无压缩层
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=0,  # 压缩比率为0
                    compress_layer_id=c1_cnt,  # 无压缩层ID
                )
                c1_cnt += 1  # 递增无压缩层计数
            elif ratio == 4:  # C4压缩层
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=4,  # 压缩比率为4
                    compress_layer_id=c4_cnt,  # C4层ID
                    compress_kv_pool=self.c4_kv_pool,  # C4 KV池
                )
                c4_cnt += 1  # 递增C4层计数
            elif ratio == 128:  # C128压缩层
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=128,  # 压缩比率为128
                    compress_layer_id=c128_cnt,  # C128层ID
                    compress_kv_pool=self.c128_kv_pool,  # C128 KV池
                )
                c128_cnt += 1  # 递增C128层计数
            else:
                raise ValueError(f"Unsupported compression ratio: {ratio}")  # 不支持的压缩比率

    def wait_layer_transfer(self, layer_id: int) -> None:  # 等待层传输完成
        if self.layer_transfer_counter is not None:  # 如果有层传输计数器
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)  # 等待指定层传输完成

    def get_attention_compress_states(self, layer_id: int) -> CompressStatePool:  # 获取注意力压缩状态池
        self.wait_layer_transfer(layer_id)  # 等待层传输完成
        compress_state_pool = self.compress_state_pools[layer_id]  # 获取指定层的压缩状态池
        assert (
            compress_state_pool is not None
        ), "Only c4/c128 layers have attention states."  # 断言状态池存在
        return compress_state_pool  # 返回压缩状态池

    def get_indexer_compress_states(self, layer_id: int) -> CompressStatePool:  # 获取索引器压缩状态池
        self.wait_layer_transfer(layer_id)  # 等待层传输完成
        indexer_compress_state_pool = self.indexer_compress_state_pools[layer_id]  # 获取指定层的索引器状态池
        assert (
            indexer_compress_state_pool is not None
        ), "Only c4 layers have indexer states."  # 断言状态池存在
        return indexer_compress_state_pool  # 返回索引器状态池

    def _swa_local_layer_id(self, layer_id: int) -> int:  # 将绝对层ID转换为SWA池本地索引
        """Convert absolute model layer_id to SWA-pool-local (PP-stage-local) index.
        将绝对模型层ID转换为SWA池本地（流水线阶段本地）索引。"""
        return layer_id - self._stage_start  # 计算本地层ID

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:  # 获取SWA键缓冲区
        self.wait_layer_transfer(layer_id)  # 等待层传输完成
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))  # 返回SWA键缓冲区

    def set_swa_key_buffer(  # 设置SWA键缓冲区
        self,
        layer_id: int,  # 层ID
        loc: torch.Tensor,  # 位置索引张量
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,  # FP8-Rope-BF16打包数据
    ) -> None:
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id), loc, cache_nope_fp8_rope_bf16_pack  # 调用SWA KV池设置键
        )

    def get_extra_key_page_size(self, layer_id: int) -> int:  # 获取额外键的页面大小
        _, _, compress_kv_pool = self.layer_mapping[layer_id]  # 获取层映射项
        assert compress_kv_pool is not None  # 断言压缩KV池存在
        return compress_kv_pool.page_size  # 返回压缩KV池的页面大小

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:  # 获取额外键缓冲区
        self.wait_layer_transfer(layer_id)  # 等待层传输完成
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]  # 获取层映射项
        assert compress_kv_pool is not None  # 断言压缩KV池存在
        return compress_kv_pool.get_key_buffer(compress_layer_id)  # 返回压缩层的键缓冲区

    def set_extra_key_buffer(  # 设置额外键缓冲区
        self,
        layer_id: int,  # 层ID
        loc: torch.Tensor,  # 位置索引张量
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,  # FP8-Rope-BF16打包数据
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]  # 获取层映射项
        assert compress_kv_pool is not None  # 断言压缩KV池存在
        compress_kv_pool.set_key_buffer(
            compress_layer_id, loc, cache_nope_fp8_rope_bf16_pack  # 调用压缩KV池设置键
        )

    def get_index_k_page_size(self) -> int:  # 获取索引K页面大小
        return self.c4_indexer_kv_pool.page_size  # 返回C4索引器KV池的页面大小

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:  # 获取索引K带缩放缓冲区
        self.wait_layer_transfer(layer_id)  # 等待层传输完成
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]  # 获取层映射项
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"  # 断言为C4层
        return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(compress_layer_id)  # 返回索引K缓冲区

    def get_index_k_scale_buffer(  # 获取索引K和缩放缓冲区
        self,
        layer_id: int,  # 层ID
        seq_len: int,  # 序列长度
        page_indices: torch.Tensor,  # 页面索引张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.wait_layer_transfer(layer_id)  # 等待层传输完成
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]  # 获取层映射项
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"  # 断言为C4层
        return self.c4_indexer_kv_pool.get_index_k_scale_buffer(
            compress_layer_id, seq_len, page_indices  # 调用索引器KV池获取K和缩放
        )

    def set_index_k_scale_buffer(  # 设置索引K和缩放缓冲区
        self,
        layer_id: int,  # 层ID
        loc: torch.Tensor,  # 位置索引张量
        index_k: torch.Tensor,  # 索引K张量
        index_k_scale: torch.Tensor,  # 索引K缩放张量
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]  # 获取层映射项
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"  # 断言为C4层
        self.c4_indexer_kv_pool.set_index_k_scale_buffer(
            compress_layer_id, loc, index_k, index_k_scale  # 调用索引器KV池设置K和缩放
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:  # 获取键缓冲区（未实现）
        raise NotImplementedError()  # 抛出未实现异常

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:  # 获取值缓冲区（未实现）
        raise NotImplementedError()  # 抛出未实现异常

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:  # 获取KV缓冲区（未实现）
        raise NotImplementedError()  # 抛出未实现异常

    def set_kv_buffer(self, *args, **kwargs) -> None:  # 设置KV缓冲区（未实现）
        raise NotImplementedError()  # 抛出未实现异常

    def set_swa_key_buffer_radix(  # 使用基数树索引设置SWA键缓冲区
        self,
        layer_id: int,  # 层ID
        raw_loc: torch.Tensor,  # 原始位置索引张量
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,  # FP8-Rope-BF16打包数据
    ) -> None:
        swa_loc = self.translate_loc_from_full_to_swa(raw_loc)  # 将全量索引转换为SWA索引
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id), swa_loc, cache_nope_fp8_rope_bf16_pack  # 设置SWA键
        )

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:  # 使用基数树索引获取SWA键缓冲区
        self.wait_layer_transfer(layer_id)  # 等待层传输完成
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))  # 返回SWA键缓冲区

    def set_swa_key_buffer_radix_fused(  # 融合设置SWA键缓冲区（使用基数树索引）
        self,
        layer_id: int,  # 层ID
        raw_loc: torch.Tensor,  # 原始位置索引张量
        cache_k: torch.Tensor,  # 缓存K张量
    ) -> None:
        swa_loc = self.get_cached_swa_loc(raw_loc, layer_id)  # 获取缓存的SWA位置
        return self.swa_kv_pool.set_key_buffer_fused(
            self._swa_local_layer_id(layer_id), swa_loc, cache_k  # 调用融合设置方法
        )

    def set_swa_key_buffer_radix_fused_norm_rope(  # 融合设置SWA键缓冲区（含归一化和RoPE）
        self,
        layer_id: int,  # 层ID
        raw_loc: torch.Tensor,  # 原始位置索引张量
        kv: torch.Tensor,  # KV张量
        kv_weight: torch.Tensor,  # KV权重张量
        eps: float,  # 归一化epsilon值
        freqs_cis: torch.Tensor,  # 旋转位置编码频率
        positions: torch.Tensor,  # 位置张量
    ) -> None:
        swa_loc = self.get_cached_swa_loc(raw_loc, layer_id)  # 获取缓存的SWA位置
        fused_k_norm_rope_flashmla(  # 调用融合K-归一化-RoPE-FlashMLA内核
            kv=kv,  # KV张量
            kv_weight=kv_weight,  # KV权重
            eps=eps,  # epsilon值
            freqs_cis=freqs_cis,  # 旋转位置编码频率
            positions=positions,  # 位置张量
            out_loc=swa_loc,  # 输出位置索引
            kvcache=self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)],  # 目标KV缓存
            page_size=self.swa_kv_pool.page_size,  # 页面大小
        )

    def set_extra_key_buffer_fused(  # 融合设置额外键缓冲区
        self,
        layer_id: int,  # 层ID
        loc: torch.Tensor,  # 位置索引张量
        cache_k: torch.Tensor,  # 缓存K张量
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]  # 获取层映射项
        assert compress_kv_pool is not None  # 断言压缩KV池存在
        return compress_kv_pool.set_key_buffer_fused(compress_layer_id, loc, cache_k)  # 调用融合设置方法

    def set_index_k_fused(  # 融合设置索引K
        self,
        layer_id: int,  # 层ID
        loc: torch.Tensor,  # 位置索引张量
        cache_k: torch.Tensor,  # 缓存K张量
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]  # 获取层映射项
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"  # 断言为C4层
        return self.c4_indexer_kv_pool.set_index_fused(compress_layer_id, loc, cache_k)  # 调用融合设置方法
