# FP4 KV缓存量化方法
# 本文件实现了FP4格式的KV缓存量化策略，采用三层设计：
# quant_method（纯计算）→ Pool（缓冲区+批量反量化）→ Backend（视图适配）。
# 包含NVFP4（两级缩放：全局FP32+逐块FP8）和BlockFP4（单级块缩放）两种实现。
# Copyright 2025 SGLang Team  # SGLang团队版权声明
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证2.0版授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache许可证网址
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的担保
# See the License for the specific language governing permissions and  # 请参阅许可证以了解管理权限和
# limitations under the License.  # 限制的具体条款
# ==============================================================================  # 分隔线
"""
KV cache quantization strategy pattern.  # KV缓存量化策略模式

Three-player design:  # 三层设计：
  quant_method (pure compute)  ►  Pool (buffer + batch dequant)  ►  Backend (view adaptation)  # 量化方法（纯计算）► 池（缓冲区+批量反量化）► 后端（视图适配）
"""

from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import Optional  # 导入可选类型提示

import torch  # 导入PyTorch深度学习框架
from torch import Tensor  # 导入张量类型

from sglang.srt.layers.quantization.kvfp4_tensor import E2M1_MAX  # 导入E2M1格式最大值常量


class FP4KVCacheQuantMethod(ABC):  # FP4 KV缓存量化方法抽象基类
    """Abstract base for FP4 KV cache quantization strategies.  # FP4 KV缓存量化策略的抽象基类

    Owns the quantize/dequantize computation.  The Pool owns the buffers and  # 拥有量化/反量化计算。池拥有缓冲区并
    orchestrates the batch dequant loop.  Backends only do view/reshape.  # 编排批量反量化循环。后端仅做视图/重塑操作

    All operations (quantize_and_store, dequantize_prev_kv) use FlashInfer  # 所有操作（quantize_and_store、dequantize_prev_kv）使用FlashInfer
    kernels or pure tensor ops, so they are CUDA-graph compatible.  # 内核或纯张量操作，因此兼容CUDA图
    """

    name: str  # 量化方法名称
    SCALE_BLOCK_SIZE: int = 1  # 缩放块大小，默认为1

    def needs_dequant_workspace(self) -> bool:  # 是否需要反量化工作空间
        """Whether the pool should allocate dq_k_buffer / dq_v_buffer for prefill."""  # 池是否应为prefill分配dq_k_buffer / dq_v_buffer
        return False  # 默认不需要

    def needs_global_scale(self) -> bool:  # 是否需要全局缩放因子
        """Whether this method uses a per-layer global FP32 scale."""  # 此方法是否使用逐层的全局FP32缩放因子
        return False  # 默认不需要

    @abstractmethod
    def create_buffers(  # 创建缓冲区（抽象方法）
        self, size: int, head_num: int, head_dim: int, layer_num: int, device: str  # 大小、头数、头维度、层数、设备
    ) -> dict:  # 返回缓冲区字典
        """Allocate and return a buffer dict:  # 分配并返回缓冲区字典：
        {  # 包含：
            "k_buffer": list[Tensor],       # per-layer, shape (size, head_num, head_dim//2)  # 逐层，形状(size, head_num, head_dim//2)
            "v_buffer": list[Tensor],  # V缓冲区列表
            "k_scale_buffer": list[Tensor] | None,  # K缩放缓冲区列表或None
            "v_scale_buffer": list[Tensor] | None,  # V缩放缓冲区列表或None
            "dq_k_buffer": Tensor | None,   # shared across layers (FP8 E4M3)  # 跨层共享(FP8 E4M3)
            "dq_v_buffer": Tensor | None,  # V反量化缓冲区或None
            "store_dtype": torch.dtype,  # 存储数据类型
        }
        """

    @abstractmethod
    def quantize_and_store(  # 量化并存储（抽象方法）
        self,
        k_buffer: Tensor,  # K缓冲区
        v_buffer: Tensor,  # V缓冲区
        k_scale_buffer: Optional[Tensor],  # K缩放缓冲区
        v_scale_buffer: Optional[Tensor],  # V缩放缓冲区
        loc: Tensor,  # 存储位置索引
        cache_k: Tensor,  # 缓存的K张量
        cache_v: Tensor,  # 缓存的V张量
        k_scale=None,  # K全局缩放因子
        v_scale=None,  # V全局缩放因子
    ) -> None:
        """Quantize cache_k / cache_v and write into buffers at loc."""  # 量化cache_k / cache_v并写入loc位置的缓冲区

    @abstractmethod
    def dequantize_prev_kv(  # 反量化之前的KV（抽象方法）
        self,
        k_fp4: Tensor,  # FP4格式的K
        k_scales: Tensor,  # K缩放因子
        v_fp4: Tensor,  # FP4格式的V
        v_scales: Tensor,  # V缩放因子
        layer_id: int,  # 层ID
    ) -> tuple[Tensor, Tensor]:  # 返回K和V的元组
        """Dequantize stored FP4 KV (selected token indices already applied).  # 反量化已存储的FP4 KV（已应用选定的token索引）

        Returns:  # 返回值：
            (k_fp8, v_fp8): Both in torch.float8_e4m3fn dtype with shape  # 均为torch.float8_e4m3fn类型，形状
            matching the input (after unpacking). These are written into the  # 与输入匹配（解包后）。这些被写入
            shared dequant workspace buffer for the FlashInfer FP8 prefill kernel.  # 共享的反量化工作空间缓冲区，用于FlashInfer FP8预填充内核
        """

    @abstractmethod
    def compute_cell_size(  # 计算单元大小（抽象方法）
        self, head_num: int, head_dim: int, num_layers: int, kv_size: int  # 头数、头维度、层数、KV大小
    ) -> int:  # 返回字节数
        """Per-token memory footprint in bytes (for capacity estimation)."""  # 每token内存占用字节数（用于容量估算）

    def load_scales_from_model(self, model_runner, sm_version: int = None) -> None:  # 从模型加载缩放因子
        """Load per-layer global scales from model weights (no-op by default)."""  # 从模型权重加载逐层全局缩放因子（默认无操作）
        pass  # 默认无操作


class NVFP4KVMethod(FP4KVCacheQuantMethod):  # NVFP4 KV缓存量化方法
    """NVFP4 two-level scaling: global FP32 + per-block FP8 E4M3.  # NVFP4两级缩放：全局FP32 + 逐块FP8 E4M3

    Supported on SM100 and SM120.  # 支持SM100和SM120架构
    """

    name = "nvfp4"  # 方法名称
    SCALE_BLOCK_SIZE = 16  # 缩放块大小为16

    def __init__(self, num_layers: int, device: str, sm_version: int = 120):  # 初始化方法
        """初始化NVFP4量化方法，创建逐层全局缩放因子。"""
        self.num_layers = num_layers  # 保存层数
        self.device = device  # 保存设备
        self.sm_version = sm_version  # 保存SM版本
        # Per-layer global FP32 scales; filled by load_scales_from_model()  # 逐层全局FP32缩放因子；由load_scales_from_model()填充
        self.k_scales_gpu = torch.ones(num_layers, dtype=torch.float32, device=device)  # K全局缩放因子，初始为1
        self.v_scales_gpu = torch.ones(num_layers, dtype=torch.float32, device=device)  # V全局缩放因子，初始为1

    def needs_dequant_workspace(self) -> bool:  # 是否需要反量化工作空间
        """NVFP4方法需要反量化工作空间用于prefill阶段。"""
        return (
            True  # prefill uses FP8 dequant workspace; future native FP4 kernel → False  # prefill使用FP8反量化工作空间；未来原生FP4内核→False
        )

    def needs_global_scale(self) -> bool:  # 是否需要全局缩放因子
        """NVFP4方法使用逐层全局FP32缩放因子。"""
        return True  # 需要全局缩放因子

    def load_scales_from_model(self, model_runner, sm_version: int = None) -> None:  # 从模型加载缩放因子
        """从模型权重中加载逐层全局缩放因子，处理SM100和SM120架构差异。"""
        if sm_version is not None:  # 如果提供了SM版本
            self.sm_version = sm_version  # 更新SM版本

        from sglang.srt.model_executor.model_runner import resolve_language_model  # 导入语言模型解析函数

        language_model = resolve_language_model(model_runner.model)  # 从模型运行器中解析语言模型

        attention_layers = []  # 注意力层列表
        for layer in language_model.layers:  # 遍历语言模型的所有层
            if hasattr(layer, "self_attn"):  # 如果有self_attn属性
                if hasattr(layer.self_attn, "attn"):  # 如果self_attn有attn属性
                    attention_layers.append(layer.self_attn.attn)  # 添加attn模块
                elif hasattr(layer.self_attn, "attn_mqa"):  # 如果self_attn有attn_mqa属性
                    attention_layers.append(layer.self_attn.attn_mqa)  # 添加attn_mqa模块
            elif hasattr(layer, "attn"):  # 如果有attn属性
                attention_layers.append(layer.attn)  # 添加attn模块
            elif hasattr(layer, "attention"):  # 如果有attention属性
                if hasattr(layer.attention, "attn"):  # 如果attention有attn属性
                    attention_layers.append(layer.attention.attn)  # 添加attn模块

        if not attention_layers:  # 如果没有找到注意力层
            return  # 直接返回

        # k_scales_gpu is indexed by global (absolute) layer_id.  Resize if the model  # k_scales_gpu通过全局（绝对）layer_id索引。如果模型
        # has layers with global IDs larger than what was pre-allocated.  # 有全局ID大于预分配大小的层，则调整大小
        # This happens in hybrid models (e.g., GDN) where only a subset of layers  # 这发生在混合模型（如GDN）中，其中只有部分层
        # are full-attention, but their layer_ids are non-contiguous.  # 是全注意力层，但它们的layer_id是非连续的
        max_global_id = max(layer.layer_id for layer in attention_layers)  # 计算最大全局层ID
        required_size = max_global_id + 1  # 计算所需大小
        if required_size > len(self.k_scales_gpu):  # 如果所需大小大于当前分配
            self.k_scales_gpu = torch.ones(  # 重新创建K缩放因子张量
                required_size, dtype=torch.float32, device=self.device  # 新大小，float32，原设备
            )
            self.v_scales_gpu = torch.ones(  # 重新创建V缩放因子张量
                required_size, dtype=torch.float32, device=self.device  # 新大小，float32，原设备
            )

        k_scales_cpu = self.k_scales_gpu.cpu().clone()  # 将K缩放因子复制到CPU
        v_scales_cpu = self.v_scales_gpu.cpu().clone()  # 将V缩放因子复制到CPU

        for layer in attention_layers:  # 遍历所有注意力层
            layer_id = layer.layer_id  # global id  # 获取全局层ID
            k_scale = (  # 获取K缩放值
                float(layer.k_scale)  # 转换为浮点数
                if hasattr(layer, "k_scale") and layer.k_scale is not None  # 如果有k_scale属性且不为None
                else 1.0  # 否则默认为1.0
            )
            v_scale = (  # 获取V缩放值
                float(layer.v_scale)  # 转换为浮点数
                if hasattr(layer, "v_scale") and layer.v_scale is not None  # 如果有v_scale属性且不为None
                else 1.0  # 否则默认为1.0
            )
            # SM100 uses TRT-LLM XQA kernels that expect KV scales as  # SM100使用TRT-LLM XQA内核，期望KV缩放因子为
            # amax / 448, but the calibrated checkpoint stores amax / (6 * 448).  # amax / 448，但校准检查点存储的是amax / (6 * 448)
            # We multiply by E2M1_MAX (6.0) to bridge the gap.  SM120 uses a  # 我们乘以E2M1_MAX(6.0)来弥合差距。SM120使用
            # different kernel path where scales already include this factor.  # 不同的内核路径，缩放因子已包含此因子
            # The FP4 data type itself is identical on both architectures.  # 两种架构的FP4数据类型本身是相同的
            # Reference: TRT-LLM FP8QDQLinearMethod.process_weights_after_loading_fused_qkv_linear  # 参考：TRT-LLM FP8QDQLinearMethod.process_weights_after_loading_fused_qkv_linear
            # https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/linear.py  # 参考链接
            if self.sm_version == 100:  # 如果是SM100架构
                k_scale *= E2M1_MAX  # K缩放因子乘以E2M1最大值
                v_scale *= E2M1_MAX  # V缩放因子乘以E2M1最大值
            k_scales_cpu[layer_id] = k_scale  # 设置CPU上的K缩放值
            v_scales_cpu[layer_id] = v_scale  # 设置CPU上的V缩放值

        self.k_scales_gpu.copy_(k_scales_cpu, non_blocking=True)  # 异步复制K缩放因子到GPU
        self.v_scales_gpu.copy_(v_scales_cpu, non_blocking=True)  # 异步复制V缩放因子到GPU

    def create_buffers(  # 创建缓冲区方法
        self, size: int, head_num: int, head_dim: int, layer_num: int, device: str  # 大小、头数、头维度、层数、设备
    ) -> dict:  # 返回缓冲区字典
        """为NVFP4量化方法创建K/V缓冲区和缩放因子缓冲区。"""
        m = size  # 序列长度维度
        n = head_num  # 头数维度
        k = head_dim  # 头维度
        store_dtype = torch.uint8  # 存储数据类型为uint8（FP4打包）
        dq_dtype = torch.float8_e4m3fn  # 反量化数据类型为FP8 E4M3

        k_buffer = [  # 创建K缓冲区列表（逐层）
            torch.zeros((m, n, k // 2), dtype=store_dtype, device=device)  # 每层K缓冲区，FP4占原尺寸一半
            for _ in range(layer_num)  # 遍历每层
        ]
        v_buffer = [  # 创建V缓冲区列表（逐层）
            torch.zeros((m, n, k // 2), dtype=store_dtype, device=device)  # 每层V缓冲区，FP4占原尺寸一半
            for _ in range(layer_num)  # 遍历每层
        ]
        k_scale_buffer = [  # 创建K缩放因子缓冲区列表（逐层）
            torch.zeros(  # 创建全零张量
                (m, n, k // self.SCALE_BLOCK_SIZE), dtype=store_dtype, device=device  # 形状考虑缩放块大小
            )
            for _ in range(layer_num)  # 遍历每层
        ]
        v_scale_buffer = [  # 创建V缩放因子缓冲区列表（逐层）
            torch.zeros(  # 创建全零张量
                (m, n, k // self.SCALE_BLOCK_SIZE), dtype=store_dtype, device=device  # 形状考虑缩放块大小
            )
            for _ in range(layer_num)  # 遍历每层
        ]
        # Shared dequant workspace — one copy, reused per layer during prefill  # 共享反量化工作空间——一个副本，prefill期间每层复用
        dq_k_buffer = torch.zeros((m, n, k), dtype=dq_dtype, device=device)  # K反量化工作空间
        dq_v_buffer = torch.zeros((m, n, k), dtype=dq_dtype, device=device)  # V反量化工作空间

        return {  # 返回缓冲区字典
            "k_buffer": k_buffer,  # K缓冲区
            "v_buffer": v_buffer,  # V缓冲区
            "k_scale_buffer": k_scale_buffer,  # K缩放因子缓冲区
            "v_scale_buffer": v_scale_buffer,  # V缩放因子缓冲区
            "dq_k_buffer": dq_k_buffer,  # K反量化工作空间
            "dq_v_buffer": dq_v_buffer,  # V反量化工作空间
            "store_dtype": store_dtype,  # 存储数据类型
        }

    def quantize_and_store(  # 量化并存储方法
        self,
        k_buffer: Tensor,  # K缓冲区
        v_buffer: Tensor,  # V缓冲区
        k_scale_buffer: Optional[Tensor],  # K缩放因子缓冲区
        v_scale_buffer: Optional[Tensor],  # V缩放因子缓冲区
        loc: Tensor,  # 存储位置索引
        cache_k: Tensor,  # 缓存的K张量
        cache_v: Tensor,  # 缓存的V张量
        k_scale=None,  # K全局缩放因子
        v_scale=None,  # V全局缩放因子
    ) -> None:
        """将K/V缓存量化为NVFP4格式并存储到缓冲区中。"""
        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil  # 导入NVFP4量化工具类

        cache_k, cache_k_fp4_sf, _ = NVFP4KVQuantizeUtil.quantize(  # 量化K缓存
            cache_k.contiguous(), k_scale  # 确保连续并传入全局缩放因子
        )
        cache_v, cache_v_fp4_sf, _ = NVFP4KVQuantizeUtil.quantize(  # 量化V缓存
            cache_v.contiguous(), v_scale  # 确保连续并传入全局缩放因子
        )

        k_buffer[loc] = cache_k.view(torch.uint8)  # 将量化后的K以uint8视图存储
        v_buffer[loc] = cache_v.view(torch.uint8)  # 将量化后的V以uint8视图存储
        k_scale_buffer[loc] = cache_k_fp4_sf.view(torch.uint8)  # 将K缩放因子以uint8视图存储
        v_scale_buffer[loc] = cache_v_fp4_sf.view(torch.uint8)  # 将V缩放因子以uint8视图存储

    def dequantize_prev_kv(  # 反量化之前的KV方法
        self,
        k_fp4: Tensor,  # FP4格式的K
        k_scales: Tensor,  # K缩放因子
        v_fp4: Tensor,  # FP4格式的V
        v_scales: Tensor,  # V缩放因子
        layer_id: int,  # 层ID
    ) -> tuple[Tensor, Tensor]:  # 返回K和V的元组
        """Dequantize FP4 KV (indexed tokens) → FP8 E4M3."""  # 反量化FP4 KV（已索引的token）→ FP8 E4M3
        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil  # 导入NVFP4量化工具类

        cur_k_scale = self.k_scales_gpu[layer_id : layer_id + 1]  # 获取当前层的K全局缩放因子
        cur_v_scale = self.v_scales_gpu[layer_id : layer_id + 1]  # 获取当前层的V全局缩放因子
        k_bf16 = NVFP4KVQuantizeUtil.dequantize(  # 反量化K
            k_fp4.view(torch.uint8), k_scales, cur_k_scale  # FP4数据、块缩放因子、全局缩放因子
        )
        v_bf16 = NVFP4KVQuantizeUtil.dequantize(  # 反量化V
            v_fp4.view(torch.uint8), v_scales, cur_v_scale  # FP4数据、块缩放因子、全局缩放因子
        )
        return k_bf16.to(torch.float8_e4m3fn), v_bf16.to(torch.float8_e4m3fn)  # 转为FP8 E4M3格式并返回

    def compute_cell_size(  # 计算单元大小方法
        self, head_num: int, head_dim: int, num_layers: int, kv_size: int  # 头数、头维度、层数、KV大小
    ) -> int:  # 返回字节数
        """计算NVFP4量化下每个token的KV缓存内存占用（字节）。"""
        # FP4 data: per-layer, K+V  # FP4数据：逐层，K+V
        fp4_size = head_num * (head_dim // 2) * num_layers * 2 * kv_size  # FP4占用=头数*(头维度/2)*层数*2*KV大小
        # Block scales: per-layer, K+V (uint8)  # 块缩放因子：逐层，K+V（uint8）
        scale_size = (  # 缩放因子占用
            head_num * (head_dim // self.SCALE_BLOCK_SIZE) * num_layers * 2 * kv_size  # 头数*(头维度/块大小)*层数*2*KV大小
        )
        # Dequant workspace: shared across layers (not multiplied by num_layers), FP8  # 反量化工作空间：跨层共享（不乘以层数），FP8
        dq_size = head_num * head_dim * 2 * kv_size  # 反量化空间=头数*头维度*2*KV大小
        return fp4_size + scale_size + dq_size  # 返回总大小


class BlockFP4KVMethod(FP4KVCacheQuantMethod):  # BlockFP4 KV缓存量化方法
    """Block-wise FP4 single-level scaling (similar to MXFP4 but block_size=16)."""  # 块级FP4单级缩放（类似MXFP4，块大小为16）

    name = "blockfp4"  # 方法名称
    SCALE_BLOCK_SIZE = 16  # 缩放块大小为16

    def needs_dequant_workspace(self) -> bool:  # 是否需要反量化工作空间
        """BlockFP4方法需要反量化工作空间。"""
        return True  # 需要反量化工作空间

    def create_buffers(  # 创建缓冲区方法
        self, size: int, head_num: int, head_dim: int, layer_num: int, device: str  # 大小、头数、头维度、层数、设备
    ) -> dict:  # 返回缓冲区字典
        """为BlockFP4量化方法创建K/V缓冲区和缩放因子缓冲区。"""
        m = size  # 序列长度维度
        store_dtype = torch.uint8  # 存储数据类型为uint8
        dq_dtype = torch.float8_e4m3fn  # 反量化数据类型为FP8 E4M3

        k_buffer = [  # 创建K缓冲区列表（逐层）
            torch.zeros((m, head_num, head_dim // 2), dtype=store_dtype, device=device)  # 每层K缓冲区，FP4占原尺寸一半
            for _ in range(layer_num)  # 遍历每层
        ]
        v_buffer = [  # 创建V缓冲区列表（逐层）
            torch.zeros((m, head_num, head_dim // 2), dtype=store_dtype, device=device)  # 每层V缓冲区，FP4占原尺寸一半
            for _ in range(layer_num)  # 遍历每层
        ]
        # MXFP4 flattens head dimensions for scale storage  # MXFP4将头维度展平用于缩放因子存储
        k_scale_buffer = [  # 创建K缩放因子缓冲区列表（逐层）
            torch.zeros(  # 创建全零张量
                (m, (head_num * head_dim) // self.SCALE_BLOCK_SIZE),  # 头数*头维度除以缩放块大小
                dtype=store_dtype,  # uint8类型
                device=device,  # 指定设备
            )
            for _ in range(layer_num)  # 遍历每层
        ]
        v_scale_buffer = [  # 创建V缩放因子缓冲区列表（逐层）
            torch.zeros(  # 创建全零张量
                (m, (head_num * head_dim) // self.SCALE_BLOCK_SIZE),  # 头数*头维度除以缩放块大小
                dtype=store_dtype,  # uint8类型
                device=device,  # 指定设备
            )
            for _ in range(layer_num)  # 遍历每层
        ]
        dq_k_buffer = torch.zeros(  # 创建K反量化工作空间
            (m, head_num, head_dim), dtype=dq_dtype, device=device  # 完整尺寸，FP8类型
        )
        dq_v_buffer = torch.zeros(  # 创建V反量化工作空间
            (m, head_num, head_dim), dtype=dq_dtype, device=device  # 完整尺寸，FP8类型
        )

        return {  # 返回缓冲区字典
            "k_buffer": k_buffer,  # K缓冲区
            "v_buffer": v_buffer,  # V缓冲区
            "k_scale_buffer": k_scale_buffer,  # K缩放因子缓冲区
            "v_scale_buffer": v_scale_buffer,  # V缩放因子缓冲区
            "dq_k_buffer": dq_k_buffer,  # K反量化工作空间
            "dq_v_buffer": dq_v_buffer,  # V反量化工作空间
            "store_dtype": store_dtype,  # 存储数据类型
        }

    def quantize_and_store(  # 量化并存储方法
        self,
        k_buffer,  # K缓冲区
        v_buffer,  # V缓冲区
        k_scale_buffer,  # K缩放因子缓冲区
        v_scale_buffer,  # V缩放因子缓冲区
        loc,  # 存储位置索引
        cache_k,  # 缓存的K张量
        cache_v,  # 缓存的V张量
        k_scale=None,  # K全局缩放因子（未使用）
        v_scale=None,  # V全局缩放因子（未使用）
    ) -> None:
        """将K/V缓存量化为BlockFP4格式并存储到缓冲区中。"""
        from sglang.srt.layers.quantization.kvfp4_tensor import BlockFP4KVQuantizeUtil  # 导入BlockFP4量化工具类

        cache_k_fp4, cache_k_sf = BlockFP4KVQuantizeUtil.batched_quantize(cache_k)  # 批量量化K
        cache_v_fp4, cache_v_sf = BlockFP4KVQuantizeUtil.batched_quantize(cache_v)  # 批量量化V
        k_buffer[loc] = cache_k_fp4  # 存储量化后的K
        v_buffer[loc] = cache_v_fp4  # 存储量化后的V
        k_scale_buffer[loc] = cache_k_sf  # 存储K缩放因子
        v_scale_buffer[loc] = cache_v_sf  # 存储V缩放因子

    def dequantize_prev_kv(  # 反量化之前的KV方法
        self,
        k_fp4: Tensor,  # FP4格式的K
        k_scales: Tensor,  # K缩放因子
        v_fp4: Tensor,  # FP4格式的V
        v_scales: Tensor,  # V缩放因子
        layer_id: int,  # 层ID（BlockFP4未使用）
    ) -> tuple[Tensor, Tensor]:  # 返回K和V的元组
        """将BlockFP4格式的KV反量化为FP8 E4M3格式。"""
        from sglang.srt.layers.quantization.kvfp4_tensor import BlockFP4KVQuantizeUtil  # 导入BlockFP4量化工具类

        k_bf16 = BlockFP4KVQuantizeUtil.batched_dequantize(k_fp4, k_scales)  # 批量反量化K
        v_bf16 = BlockFP4KVQuantizeUtil.batched_dequantize(v_fp4, v_scales)  # 批量反量化V
        return k_bf16.to(torch.float8_e4m3fn), v_bf16.to(torch.float8_e4m3fn)  # 转为FP8 E4M3格式并返回

    def compute_cell_size(  # 计算单元大小方法
        self, head_num: int, head_dim: int, num_layers: int, kv_size: int  # 头数、头维度、层数、KV大小
    ) -> int:  # 返回字节数
        """计算BlockFP4量化下每个token的KV缓存内存占用（字节）。"""
        fp4_size = head_num * (head_dim // 2) * num_layers * 2 * kv_size  # FP4数据占用
        scale_size = (  # 缩放因子占用
            (head_num * head_dim // self.SCALE_BLOCK_SIZE) * num_layers * 2 * kv_size  # 头数*头维度/块大小*层数*2*KV大小
        )
        dq_size = head_num * head_dim * 2 * kv_size  # 反量化工作空间占用
        return fp4_size + scale_size + dq_size  # 返回总大小


# Registry: name → class.  Only classes for fp4_e2m1 dtype need to be listed.  # 注册表：名称→类。仅fp4_e2m1数据类型的类需要列出
FP4_KV_CACHE_QUANT_REGISTRY: dict[str, type[FP4KVCacheQuantMethod]] = {  # FP4 KV缓存量化方法注册表
    "nvfp4": NVFP4KVMethod,  # NVFP4方法
    "blockfp4": BlockFP4KVMethod,  # BlockFP4方法
}


def get_fp4_kv_cache_quant_method(name: str, **kwargs) -> FP4KVCacheQuantMethod:  # 根据名称获取FP4 KV缓存量化方法
    """Instantiate a FP4KVCacheQuantMethod by recipe name."""  # 根据配方名称实例化FP4KVCacheQuantMethod
    if name not in FP4_KV_CACHE_QUANT_REGISTRY:  # 如果名称不在注册表中
        raise ValueError(  # 抛出值错误
            f"Unknown fp4_kv_cache_recipe: '{name}'. "  # 未知的FP4 KV缓存配方
            f"Available: {list(FP4_KV_CACHE_QUANT_REGISTRY)}"  # 可用的配方列表
        )
    return FP4_KV_CACHE_QUANT_REGISTRY[name](**kwargs)  # 从注册表中实例化并返回
