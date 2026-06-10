# 本文件实现 InternVL 视觉编码器的 CUDA Graph 运行器，通过捕获视觉 Transformer 的
# 前向计算图并重放，以减少 GPU 内核启动开销，加速推理过程
# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ViT CUDA Graph Runner class."""

from __future__ import annotations  # 启用延迟类型注解解析

from typing import Dict, Hashable, Tuple  # 导入类型提示工具

import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入神经网络模块

from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数


class InternViTCudaGraphRunner:
    """CUDA Graph runner for InternVL vision encoder.
    InternVL 视觉编码器的 CUDA Graph 运行器。

    Captures:
      y = layer_N(...layer_2(layer_1(x)))

    Keyed by (B, S). This is REQUIRED because InternVL uses [B,S,H].
    """

    def __init__(self, encoder: nn.Module) -> None:  # 初始化，接收视觉编码器模块
        self.encoder = encoder  # 保存编码器引用

        # key -> graph & stable buffers
        self.graphs: Dict[Hashable, torch.cuda.CUDAGraph] = {}  # 存储每个图键对应的 CUDA Graph
        self.inp: Dict[Hashable, torch.Tensor] = {}  # 存储每个图键对应的稳定输入缓冲区
        self.ws: Dict[Hashable, torch.Tensor] = {}  # 存储每个图键对应的工作空间缓冲区
        self.out: Dict[Hashable, torch.Tensor] = {}  # 存储每个图键对应的稳定输出缓冲区

        # key -> stable cu_seqlens buffers (addresses must be stable)
        self.cu: Dict[Hashable, torch.Tensor] = {}  # 存储累积序列长度缓冲区
        self.cu_kk: Dict[Hashable, torch.Tensor] = {}  # 存储累积序列长度的差分缓冲区

        # cache attention metadata
        first_layer = encoder.layers[0]  # 获取第一层
        # InternAttention wraps VisionAttention as first_layer.attn.attn
        self._attn: VisionAttention = first_layer.attn.attn  # type: ignore  # 获取视觉注意力层引用

    @property
    def device(self) -> torch.device:  # 获取设备属性
        return next(self.encoder.parameters()).device  # 返回编码器参数所在的设备

    @property
    def dtype(self) -> torch.dtype:  # 获取数据类型属性
        return next(self.encoder.parameters()).dtype  # 返回编码器参数的数据类型

    def _graph_key(self, x: torch.Tensor) -> Tuple[int, int]:  # 根据输入张量生成图键
        # x: [B,S,H]
        return (x.shape[0], x.shape[1])  # 返回 (批次大小, 序列长度) 作为图键

    def _build_cu(self, B: int, S: int, device: torch.device) -> torch.Tensor:  # 构建累积序列长度张量
        # [0, S, 2S, ..., B*S]
        return torch.arange(0, (B + 1) * S, step=S, device=device, dtype=torch.int32)  # 生成等间距的累积序列长度

    def _alloc_ws(
        self, B: int, S: int, H: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:  # 分配注意力工作空间张量
        # InternVL shape: [tokens, nheads, head_dim]
        tokens = B * S  # 计算总 token 数

        num_heads = getattr(self._attn, "num_attention_heads_per_partition", None)  # 尝试获取分区注意力头数
        if num_heads is None:  # 如果分区头数不存在
            num_heads = getattr(self._attn, "num_heads", None)  # 尝试获取总注意力头数
        if num_heads is None:  # 如果都不存在
            raise RuntimeError("Cannot infer num_heads from VisionAttention")  # 抛出运行时异常

        head_dim = getattr(self._attn, "head_size", None)  # 尝试获取注意力头维度
        if head_dim is None:  # 如果不存在
            # fallback (should rarely happen)
            head_dim = H // int(num_heads)  # 根据隐藏维度和头数推算

        return torch.empty(  # 返回空的工作空间张量
            tokens,
            int(num_heads),
            int(head_dim),
            device=device,
            dtype=dtype,
        )

    def _warmup_once(self, key: Hashable) -> None:  # 对预分配缓冲区执行一次热身运行以触发延迟初始化
        """Run a tiny eager warmup on the preallocated buffers to trigger lazy init.
        在预分配缓冲区上执行一次 eager 热身运行，触发延迟初始化。"""
        override_backend = get_global_server_args().mm_attention_backend  # 获取多模态注意力后端配置
        cu = self.cu[key]  # 获取累积序列长度
        cu_kk = self.cu_kk[key]  # 获取累积序列长度差分
        max_len = int(cu_kk.max().item()) if cu_kk.numel() else 0  # 计算最大序列长度

        if override_backend == "triton_attn":  # 如果使用 Triton 注意力后端
            cu_ws = [cu, cu_kk, max_len]  # 构建 Triton 注意力所需的参数列表
        elif override_backend == "fa3":  # 如果使用 FlashAttention3 后端
            cu_ws = [cu, max_len]  # 构建 FA3 所需的参数列表
        else:  # 其他后端
            raise RuntimeError("Not supported ViT attention backend for InternVL CG")  # 抛出不支持异常

        x = self.inp[key]  # 获取输入缓冲区
        y = x  # 初始化输出为输入
        with torch.no_grad():  # 禁用梯度计算
            for blk in self.encoder.layers:  # 遍历编码器的每一层
                y = blk(y, cu_seqlens=cu_ws, output_ws=self.ws[key])  # 执行前向传播

    def _capture_graph(self, key: Hashable) -> None:  # 捕获 CUDA Graph
        g = torch.cuda.CUDAGraph()  # 创建新的 CUDA Graph 对象
        override_backend = get_global_server_args().mm_attention_backend  # 获取注意力后端配置

        cu = self.cu[key]  # 获取累积序列长度
        cu_kk = self.cu_kk[key]  # 获取累积序列长度差分
        max_len = int(cu_kk.max().item()) if cu_kk.numel() else 0  # 计算最大序列长度

        if override_backend == "triton_attn":  # 如果使用 Triton 注意力后端
            cu_ws = [cu, cu_kk, max_len]  # 构建 Triton 注意力参数
        elif override_backend == "fa3":  # 如果使用 FA3 后端
            cu_ws = [cu, max_len]  # 构建 FA3 参数
        else:  # 其他后端
            raise RuntimeError("Not supported ViT attention backend for InternVL CG")  # 抛出异常

        torch.cuda.synchronize()  # 同步 CUDA 流

        with torch.cuda.graph(g):  # 开始捕获 CUDA Graph
            y = self.inp[key]  # 获取输入
            for blk in self.encoder.layers:  # 遍历编码器的每一层
                y = blk(y, cu_seqlens=cu_ws, output_ws=self.ws[key])  # 执行前向传播
            # y is a stable output tensor produced during capture; keep reference
            self.out[key] = y  # 保存输出引用

        self.graphs[key] = g  # 保存 CUDA Graph

    def create_graph(self, x: torch.Tensor) -> Hashable:  # 创建 CUDA Graph（如已存在则复用）
        # x: [B, S, H]
        x = x.contiguous()  # 确保输入张量在内存中连续
        key = self._graph_key(x)  # 计算图键
        if key in self.graphs:  # 如果图已存在
            return key  # 直接返回图键

        B, S, H = x.shape  # 解包输入形状
        device = x.device  # 获取设备
        dtype = x.dtype  # 获取数据类型

        # stable input buffer
        self.inp[key] = torch.empty_like(x, device=device).contiguous()  # 分配稳定的输入缓冲区

        # stable cu buffers
        cu = self._build_cu(B, S, device=device)  # 构建累积序列长度张量
        self.cu[key] = cu  # 保存累积序列长度
        self.cu_kk[key] = cu[1:] - cu[:-1]  # 计算并保存差分

        # stable attention workspace
        self.ws[key] = self._alloc_ws(B, S, H, device=device, dtype=dtype)  # 分配注意力工作空间

        self.inp[key].copy_(x)  # 将输入数据拷贝到稳定缓冲区
        self._warmup_once(key)  # 执行热身运行

        # capture
        self._capture_graph(key)  # 捕获 CUDA Graph
        return key  # 返回图键

    def run(self, x: torch.Tensor) -> torch.Tensor:  # 使用 CUDA Graph 执行前向推理
        # x: [B, S, H]
        x = x.contiguous()  # 确保输入张量连续
        key = self._graph_key(x)  # 计算图键
        if key not in self.graphs:  # 如果图不存在
            self.create_graph(x)  # 创建图

        # update input content (address stable)
        self.inp[key].copy_(x)  # 更新输入数据（地址保持稳定）

        # replay
        self.graphs[key].replay()  # 重放 CUDA Graph

        return self.out[key]  # 返回输出张量
