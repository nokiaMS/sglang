# 本文件实现通用的 ViT（Vision Transformer）CUDA Graph 运行器，支持 Qwen2.5-VL 窗口注意力
# 和 Qwen3-VL 深度堆栈结构，通过捕获和重放 CUDA Graph 来加速视觉 Transformer 推理
# Copyright 2023-2025 SGLang Team
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

import inspect  # 导入检查模块，用于检测函数签名
from contextlib import nullcontext  # 导入空上下文管理器
from typing import Dict, Hashable, List, Optional, Tuple  # 导入类型提示工具

import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入神经网络模块

from sglang.srt.distributed.parallel_state import get_tp_group  # 导入张量并行组获取函数
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数


class ViTCudaGraphRunner:
    """Generic ViT CUDA Graph Runner.
    通用 ViT CUDA Graph 运行器。

    This runner captures the "blocks + merger + deepstack merger (optional)" part
    of a vision transformer into a CUDA graph and replays it for identical shapes.

    Optional for Qwen2.5 windowed attention:
      - vit.fullatt_block_indexes: Sequence[int]
      - run() provides both cu_seqlens and cu_window_seqlens

    Optional for Qwen3 deepstack:
      - vit.deepstack_vision_indexes: Sequence[int]
      - vit.deepstack_merger_list: nn.ModuleList (same length as deepstack_vision_indexes)
    """

    def __init__(
        self,
        vit: nn.Module,  # 视觉 Transformer 模型
    ) -> None:
        self.vit = vit  # 保存 ViT 模型引用

        # graph_key -> buffers / graphs
        self.block_input: Dict[Hashable, torch.Tensor] = {}  # 存储每个图键对应的输入缓冲区
        self.block_ws: Dict[Hashable, torch.Tensor] = {}  # 存储每个图键对应的工作空间
        self.block_graphs: Dict[Hashable, torch.cuda.CUDAGraph] = {}  # 存储每个图键对应的 CUDA Graph
        self.block_output: Dict[Hashable, torch.Tensor] = {}  # 存储每个图键对应的输出缓冲区

        # captured seqlens buffers (addresses must be stable for cuda-graph replay)
        self.cu_full_len: Dict[Hashable, torch.Tensor] = {}  # 全序列累积长度缓冲区
        self.cu_window_len: Dict[Hashable, torch.Tensor] = {}  # 窗口序列累积长度缓冲区
        self.cu_full_len_kk: Dict[Hashable, torch.Tensor] = {}  # 全序列累积长度差分缓冲区
        self.cu_window_len_kk: Dict[Hashable, torch.Tensor] = {}  # 窗口序列累积长度差分缓冲区

        # rotary position buffers shared across graphs
        self.sin_cos_ws: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # 旋转位置编码的工作空间 (cos, sin)
        self.max_context_len = getattr(vit, "max_context_len", None)  # 获取最大上下文长度

        # Qwen2.5-VL specific viarable.
        self._fullatt_block_indexes = set(getattr(vit, "fullatt_block_indexes", ()))  # 全注意力块索引集合

        # Qwen3-VL specific variables.
        self._deepstack_visual_indexes = list(  # 深度堆栈视觉索引列表
            getattr(vit, "deepstack_visual_indexes", []) or []
        )
        self._deepstack_merger_list = getattr(vit, "deepstack_merger_list", None)  # 深度堆栈合并器列表

        first_blk = vit.blocks[0]  # 获取第一个 Transformer 块
        self._blk_accepts_output_ws = (  # 检查块是否接受 output_ws 参数
            "output_ws" in inspect.signature(first_blk.forward).parameters
        )

        self._attn: Optional[VisionAttention] = getattr(first_blk, "attn", None)  # 获取注意力层
        self._attn_backend = getattr(self._attn, "qkv_backend", None)  # 获取 QKV 后端

    @property
    def device(self) -> torch.device:  # 获取设备属性
        return self.vit.device  # 返回 ViT 模型所在设备

    @property
    def dtype(self) -> torch.dtype:  # 获取数据类型属性
        return self.vit.dtype  # 返回 ViT 模型的数据类型

    def _ensure_sin_cos_ws(self, seq_len: int, head_dim: int):  # 确保旋转位置编码工作空间足够大
        if self.sin_cos_ws is None:  # 如果工作空间未初始化
            max_shape = self.max_context_len or seq_len  # 确定最大长度
            max_shape = max(max_shape, seq_len)  # 取最大值
            cos_ws = torch.empty(  # 创建 cos 工作空间
                max_shape, head_dim, dtype=self.dtype, device=self.device
            )
            sin_ws = torch.empty(  # 创建 sin 工作空间
                max_shape, head_dim, dtype=self.dtype, device=self.device
            )
            self.sin_cos_ws = (cos_ws, sin_ws)  # 保存工作空间元组
        else:  # 如果工作空间已存在
            if self.sin_cos_ws[0].size(0) < seq_len:  # 如果当前工作空间不够大
                max_shape = max(self.sin_cos_ws[0].size(0) * 2, seq_len)  # 扩展为两倍或足够大小
                cos_ws = torch.empty(  # 创建新的 cos 工作空间
                    max_shape, head_dim, dtype=self.dtype, device=self.device
                )
                sin_ws = torch.empty(  # 创建新的 sin 工作空间
                    max_shape, head_dim, dtype=self.dtype, device=self.device
                )
                self.sin_cos_ws = (cos_ws, sin_ws)  # 更新工作空间

    def _get_graph_key(self, x_3d: torch.Tensor) -> int:  # 根据输入张量获取图键
        # x_3d: [S, B, H], B=1, S as graph_key
        return x_3d.shape[0]  # 返回序列长度作为图键

    def _create_graph(
        self,
        graph_key: int,  # 图键
        position_embeddings: Optional[  # 位置嵌入 (cos, sin)，形状 [S, D]
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # (cos, sin), [S, D]
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,  # 旋转位置编码 cos 分量
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,  # 旋转位置编码 sin 分量
    ):  # 捕获 CUDA Graph
        graph = torch.cuda.CUDAGraph()  # 创建新的 CUDA Graph
        vit = self.vit  # 获取 ViT 引用

        # Qwen2.5-VL
        if self._fullatt_block_indexes:  # 如果有全注意力块索引（Qwen2.5-VL）
            cu_window = self.cu_window_len[graph_key]  # 获取窗口序列累积长度
            cu_window_kk = self.cu_window_len_kk[graph_key]  # 获取窗口序列差分
            max_window_len = int(cu_window_kk.max().item())  # 计算窗口最大序列长度

        cu_full = self.cu_full_len[graph_key]  # 获取全序列累积长度
        cu_full_kk = self.cu_full_len_kk[graph_key]  # 获取全序列差分
        max_full_len = int(cu_full_kk.max().item())  # 计算全序列最大长度

        override_backend = get_global_server_args().mm_attention_backend  # 获取注意力后端

        tp_group = get_tp_group()  # 获取张量并行组
        ca_comm = tp_group.ca_comm  # 获取通信对象
        capture_ctx = ca_comm.capture() if ca_comm is not None else nullcontext()  # 获取捕获上下文

        with capture_ctx, torch.cuda.graph(graph):  # 在捕获上下文中开始录制 CUDA Graph
            y = None  # 初始化输出
            deepstack_outs: List[torch.Tensor] = []  # 存储深度堆栈输出
            deepstack_capture_idx = 0  # 深度堆栈索引

            for layer_num, blk in enumerate(vit.blocks):  # 遍历所有 Transformer 块
                if self._fullatt_block_indexes:  # 如果使用窗口注意力（Qwen2.5-VL）
                    if layer_num in vit.fullatt_block_indexes:  # 如果当前层是全注意力层
                        cu_seqlens_now = cu_full  # 使用全序列累积长度
                        cu_seqlens_kk_now = cu_full_kk  # 使用全序列差分
                        max_len = max_full_len  # 使用全序列最大长度
                    else:  # 否则使用窗口注意力
                        cu_seqlens_now = cu_window  # 使用窗口累积长度
                        cu_seqlens_kk_now = cu_window_kk  # 使用窗口差分
                        max_len = max_window_len  # 使用窗口最大长度
                else:  # 不使用窗口注意力
                    cu_seqlens_now = cu_full  # 使用全序列累积长度
                    cu_seqlens_kk_now = cu_full_kk  # 使用全序列差分
                    max_len = max_full_len  # 使用全序列最大长度

                if override_backend == "triton_attn":  # Triton 注意力后端
                    cu_seq_len_ws = [cu_seqlens_now, cu_seqlens_kk_now, max_len]  # 构建参数列表
                elif override_backend == "fa3":  # FlashAttention3 后端
                    cu_seq_len_ws = [cu_seqlens_now, max_len]  # 构建参数列表
                else:  # 不支持的后端
                    raise RuntimeError("Not supported ViT attention backend")  # 抛出异常

                if position_embeddings is not None:  # 如果使用位置嵌入
                    if layer_num == 0:  # 第一层
                        y = blk(
                            self.block_input[graph_key],
                            cu_seqlens=cu_seq_len_ws,
                            position_embeddings=position_embeddings,
                            output_ws=self.block_ws[graph_key],
                        )
                    else:  # 后续层
                        y = blk(
                            y,
                            cu_seqlens=cu_seq_len_ws,
                            position_embeddings=position_embeddings,
                            output_ws=self.block_ws[graph_key],
                        )
                elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:  # 如果使用旋转位置编码
                    if layer_num == 0:  # 第一层
                        y = blk(
                            self.block_input[graph_key],
                            cu_seqlens=cu_seq_len_ws,
                            rotary_pos_emb_cos=rotary_pos_emb_cos,
                            rotary_pos_emb_sin=rotary_pos_emb_sin,
                            output_ws=self.block_ws[graph_key],
                        )
                    else:  # 后续层
                        y = blk(
                            y,
                            cu_seqlens=cu_seq_len_ws,
                            rotary_pos_emb_cos=rotary_pos_emb_cos,
                            rotary_pos_emb_sin=rotary_pos_emb_sin,
                            output_ws=self.block_ws[graph_key],
                        )

                # Optional deepstack support (Qwen3-VL)
                if (  # 如果使用深度堆栈（Qwen3-VL）
                    self._deepstack_visual_indexes
                    and layer_num in self._deepstack_visual_indexes
                ):
                    if self._deepstack_merger_list is None:  # 如果合并器列表缺失
                        raise RuntimeError(
                            "deepstack_visual_indexes exists but deepstack_merger_list is missing."
                        )
                    deepstack_out = self._deepstack_merger_list[deepstack_capture_idx](  # 执行深度堆栈合并
                        y
                    )
                    deepstack_outs.append(deepstack_out)  # 保存深度堆栈输出
                    deepstack_capture_idx += 1  # 递增索引

            main_out = vit.merger(y)  # 对最终输出执行合并操作

            if deepstack_outs:  # 如果有深度堆栈输出
                self.block_output[graph_key] = torch.cat(
                    [main_out] + deepstack_outs, dim=1
                )  # 将主输出和深度堆栈输出在序列维度拼接
            else:  # 无深度堆栈输出
                self.block_output[graph_key] = main_out  # 直接使用主输出

        self.block_graphs[graph_key] = graph  # 保存 CUDA Graph

    def create_graph(
        self,
        x_3d: torch.Tensor,  # [S, 1, H]
        cu_seqlens: torch.Tensor,  # 累积序列长度
        cu_window_seqlens: torch.Tensor,  # 窗口累积序列长度
        position_embeddings: Optional[  # 位置嵌入 (cos, sin)
            Tuple[torch.Tensor, torch.Tensor]
        ],  # (cos, sin), [S, D]
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,  # 旋转位置编码 cos
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,  # 旋转位置编码 sin
    ) -> int:  # 返回图键
        """创建 CUDA Graph，预分配所有缓冲区并捕获计算图"""
        vit = self.vit  # 获取 ViT 引用
        graph_key = self._get_graph_key(x_3d)  # 计算图键

        if graph_key in self.block_graphs:  # 如果图已存在
            return graph_key  # 直接返回

        # pre-allocate workspace
        attn_module: VisionAttention = vit.blocks[0].attn  # 获取注意力模块
        num_heads = attn_module.num_attention_heads_per_partition  # 获取分区注意力头数
        attn_head_dim = attn_module.head_size  # 获取注意力头维度

        if graph_key not in self.block_output:  # 如果输出缓冲区尚未分配
            self.block_output[graph_key] = torch.empty_like(  # 分配输出缓冲区
                x_3d, device=self.device
            ).contiguous()
            self.block_input[graph_key] = torch.empty_like(  # 分配输入缓冲区
                x_3d, device=self.device
            ).contiguous()
            self.block_ws[graph_key] = torch.empty(  # 分配工作空间
                graph_key,
                num_heads,
                attn_head_dim,
                device=self.device,
                dtype=self.dtype,
            )

        # Qwen2.5-VL
        if self._fullatt_block_indexes:  # 如果使用窗口注意力（Qwen2.5-VL）
            if graph_key not in self.cu_window_len:  # 如果窗口长度缓冲区尚未分配
                self.cu_window_len[graph_key] = cu_window_seqlens  # 保存窗口累积长度
                self.cu_full_len[graph_key] = cu_seqlens  # 保存全序列累积长度
                self.cu_window_len_kk[graph_key] = (
                    cu_window_seqlens[1:] - cu_window_seqlens[:-1]
                )  # 计算窗口差分
                self.cu_full_len_kk[graph_key] = cu_seqlens[1:] - cu_seqlens[:-1]  # 计算全序列差分
        else:  # 不使用窗口注意力
            if graph_key not in self.cu_full_len:  # 如果全序列长度缓冲区尚未分配
                self.cu_full_len[graph_key] = cu_seqlens  # 保存全序列累积长度
                self.cu_full_len_kk[graph_key] = cu_seqlens[1:] - cu_seqlens[:-1]  # 计算差分

        if position_embeddings is not None:  # 如果使用位置嵌入
            # make sure rotary workspace
            head_dim = position_embeddings[0].shape[1]  # 获取注意力头维度
            self._ensure_sin_cos_ws(graph_key, head_dim)  # 确保旋转位置编码工作空间

            used_cos_ws = self.sin_cos_ws[0][:graph_key, :]  # 截取需要长度的 cos
            used_sin_ws = self.sin_cos_ws[1][:graph_key, :]  # 截取需要长度的 sin
            used_cos_ws.copy_(position_embeddings[0])  # 复制 cos 数据到工作空间
            used_sin_ws.copy_(position_embeddings[1])  # 复制 sin 数据到工作空间
            persist_position_embeddings = (used_cos_ws, used_sin_ws)  # 包装为持久化位置嵌入
            self._create_graph(
                graph_key=graph_key, position_embeddings=persist_position_embeddings
            )  # 使用位置嵌入创建图
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:  # 如果使用旋转位置编码
            # make sure rotary workspace
            head_dim = rotary_pos_emb_cos.shape[1]  # 获取注意力头维度
            self._ensure_sin_cos_ws(graph_key, head_dim)  # 确保工作空间

            used_cos_ws = self.sin_cos_ws[0][:graph_key, :]  # 截取 cos
            used_sin_ws = self.sin_cos_ws[1][:graph_key, :]  # 截取 sin
            used_cos_ws.copy_(rotary_pos_emb_cos)  # 复制 cos
            used_sin_ws.copy_(rotary_pos_emb_sin)  # 复制 sin
            self._create_graph(
                graph_key=graph_key,
                position_embeddings=None,
                rotary_pos_emb_cos=used_cos_ws,
                rotary_pos_emb_sin=used_sin_ws,
            )  # 使用旋转位置编码创建图

        return graph_key  # 返回图键

    def replay(
        self,
        graph_key: int,  # 图键
        x_3d: torch.Tensor,  # 输入张量 [S, 1, H]
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 位置嵌入
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,  # 旋转位置编码 cos
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,  # 旋转位置编码 sin
        output_indices: Optional[torch.Tensor] = None,  # 输出重排序索引（Qwen2.5-VL）
    ) -> torch.Tensor:  # 返回输出张量
        """重放已捕获的 CUDA Graph，更新输入和位置编码后执行推理"""

        if position_embeddings is not None:  # 如果使用位置嵌入
            # update rotary workspace content
            head_dim = position_embeddings[0].shape[1]  # 获取注意力头维度
            self._ensure_sin_cos_ws(graph_key, head_dim)  # 确保工作空间足够
            used_cos_ws = self.sin_cos_ws[0][:graph_key, :]  # 截取 cos
            used_sin_ws = self.sin_cos_ws[1][:graph_key, :]  # 截取 sin
            used_cos_ws.copy_(position_embeddings[0])  # 更新 cos 数据
            used_sin_ws.copy_(position_embeddings[1])  # 更新 sin 数据
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:  # 如果使用旋转位置编码
            # update rotary workspace content
            head_dim = rotary_pos_emb_cos.shape[1]  # 获取注意力头维度
            self._ensure_sin_cos_ws(graph_key, head_dim)  # 确保工作空间足够
            used_cos_ws = self.sin_cos_ws[0][:graph_key, :]  # 截取 cos
            used_sin_ws = self.sin_cos_ws[1][:graph_key, :]  # 截取 sin
            used_cos_ws.copy_(rotary_pos_emb_cos)  # 更新 cos
            used_sin_ws.copy_(rotary_pos_emb_sin)  # 更新 sin

        # copy input
        self.block_input[graph_key].copy_(x_3d)  # 将输入数据拷贝到稳定缓冲区

        # replay
        self.block_graphs[graph_key].replay()  # 重放 CUDA Graph

        out = self.block_output[graph_key]  # 获取输出

        # Optional output reordering (Qwen2.5-VL window permutation inverse)
        if output_indices is not None:  # 如果需要输出重排序（Qwen2.5-VL 窗口排列逆变换）
            out = out.index_select(0, output_indices)  # 按索引重新排列输出

        return out  # 返回输出

    def run(
        self,
        x: torch.Tensor,  # 输入张量 [seq_len, hidden]
        cu_seqlens: torch.Tensor,  # 累积序列长度
        cu_window_seqlens: torch.Tensor,  # 窗口累积序列长度
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],  # 位置嵌入
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,  # 旋转位置编码 cos
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,  # 旋转位置编码 sin
        output_indices: Optional[torch.Tensor] = None,  # 输出重排序索引
    ) -> torch.Tensor:  # 返回输出张量
        """运行 ViT 推理，自动创建或重放 CUDA Graph"""
        # x: [seq_len, hidden] -> [S, B=1, H]
        x_3d = x.unsqueeze(1)  # 增加批次维度
        graph_key = self._get_graph_key(x_3d)  # 计算图键

        if graph_key not in self.block_graphs:  # 如果图不存在
            self.create_graph(  # 创建图
                x_3d=x_3d,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                cu_window_seqlens=cu_window_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )

        return self.replay(  # 重放图并返回结果
            graph_key=graph_key,
            x_3d=x_3d,
            position_embeddings=position_embeddings,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            output_indices=output_indices,
        )
