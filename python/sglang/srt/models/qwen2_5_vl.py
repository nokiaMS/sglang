# Qwen2.5-VL 视觉语言模型实现
# 本文件实现了 Qwen2.5-VL 多模态视觉语言模型，支持图像和视频输入。
# 包含视觉 Transformer 编码器、视觉补丁合并器、语言模型等组件，
# 支持多模态旋转位置编码（mrope）、视觉 CUDA Graph 加速和 EAGLE3 推测解码。
# coding=utf-8  # 编码声明
# Adapted from  # 适配自
# https://github.com/huggingface/transformers/blob/19e6e80e10118f855137b90740936c0b11ac397f/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py  # HuggingFace Transformers 中的 Qwen2-VL 实现
# Copyright 2024 The Qwen team.  # Qwen 团队版权
# Copyright 2023 The vLLM team.  # vLLM 团队版权
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.  # EleutherAI 和 HuggingFace 版权
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX  # 基于 GPT-NeoX 库
# and OPT implementations in this library. It has been modified from its  # 和 OPT 实现，已修改
# original forms to accommodate minor architectural differences compared  # 以适应架构差异
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.  # 与 Meta AI 团队使用的模型
#
# Licensed under the Apache License, Version 2.0 (the "License");  # Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 不得违反许可证使用
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 除非法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 按原样分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何担保
# See the License for the specific language governing permissions and  # 查看许可证获取权限
# limitations under the License.  # 许可证限制
"""Inference-only Qwen2-VL model compatible with HuggingFace weights."""  # 仅推理的 Qwen2-VL 模型，兼容 HuggingFace 权重

import logging  # 导入日志模块
import re  # 导入正则表达式模块
from functools import partial  # 导入偏函数工具
from typing import Iterable, List, Optional, Tuple, Type  # 导入类型提示

import torch  # 导入 PyTorch 框架
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式模块
from einops import rearrange  # 导入张量重排工具
from transformers.activations import ACT2FN  # 导入激活函数映射
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (  # 导入 Qwen2.5-VL 配置
    Qwen2_5_VLConfig,  # Qwen2.5-VL 主配置
    Qwen2_5_VLVisionConfig,  # Qwen2.5-VL 视觉配置
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # 导入 Qwen2.5-VL 视觉组件
    Qwen2_5_VisionPatchEmbed,  # 视觉补丁嵌入
    Qwen2_5_VisionRotaryEmbedding,  # 视觉旋转嵌入
)

from sglang.srt.distributed import (  # 导入分布式模块
    get_tensor_model_parallel_rank,  # 获取张量并行排名
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
)
from sglang.srt.distributed.parallel_state import get_pp_group  # 导入流水线并行组
from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.layers.activation import SiluAndMul  # 导入 SiLU 与乘法激活
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力
from sglang.srt.layers.layernorm import RMSNorm  # 导入 RMS 归一化
from sglang.srt.layers.linear import (  # 导入线性层
    ColumnParallelLinear,  # 列并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器
from sglang.srt.layers.pooler import Pooler, PoolingType  # 导入池化器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id  # 导入流水线缺失层和层 ID 获取工具
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行语言模型头
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态类型
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.qwen2 import Qwen2Model  # 导入 Qwen2 模型
from sglang.srt.models.utils import RotaryPosMixin, WeightsMapper, permute_inv  # 导入旋转位置混合、权重映射和逆排列工具
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model  # 导入 DP 分片 mrope 视觉模型运行工具
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner  # 导入 ViT CUDA Graph 运行器
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import add_prefix, is_cuda, is_npu  # 导入前缀添加和设备检测工具

_is_cuda = is_cuda()  # 检测是否为 CUDA 设备

logger = logging.getLogger(__name__)  # 获取日志记录器


class Qwen2_5_VLMLP(nn.Module):
    """Qwen2.5-VL 视觉编码器中的 MLP 模块"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        bias: bool = True,
        hidden_act="silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        """初始化视觉 MLP 模块"""
        super().__init__()  # 调用父类初始化
        self.tp_size = (  # 张量并行大小
            1 if use_data_parallel else get_tensor_model_parallel_world_size()  # 数据并行时为 1
        )
        self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()  # 张量并行排名
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控上投影合并层
            input_size=in_features,  # 输入维度
            output_sizes=[hidden_features] * 2,  # [gate_proj, up_proj]  # 输出为两倍隐藏特征
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
            tp_size=self.tp_size,  # 张量并行大小
            tp_rank=self.tp_rank,  # 张量并行排名
        )
        self.down_proj = RowParallelLinear(  # 下投影层
            hidden_features,  # 输入维度
            in_features,  # 输出维度
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 参数前缀
            tp_size=self.tp_size,  # 张量并行大小
            tp_rank=self.tp_rank,  # 张量并行排名
        )
        self.hidden_act = hidden_act  # 保存激活函数名
        if self.hidden_act == "silu":  # 如果是 SiLU
            self.act = SiluAndMul()  # 使用融合 SiLU 与乘法
        else:  # 其他激活函数
            base_act = ACT2FN[self.hidden_act]  # 获取基础激活函数

            def _act_fn(x: torch.Tensor) -> torch.Tensor:  # 定义通用激活函数
                gate, up = x.chunk(2, dim=-1)  # 分割为门控和上投影
                return base_act(gate) * up  # 返回激活后相乘

            self.act = _act_fn  # 保存激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP 前向传播：门控上投影 -> 激活 -> 下投影"""
        gate_up, _ = self.gate_up_proj(x)  # 通过门控上投影
        x = self.act(gate_up)  # 应用激活函数
        x_down, _ = self.down_proj(x)  # 通过下投影
        return x_down  # 返回输出


class Qwen2_5_VisionBlock(nn.Module):
    """Qwen2.5 视觉 Transformer 块，包含注意力和 MLP"""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        hidden_act="silu",
        norm_layer: Type[nn.Module] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        num_dummy_heads: int = 0,
        rms_norm_eps: float = 1e-6,
        use_data_parallel: bool = False,
    ) -> None:
        """初始化视觉 Transformer 块"""
        super().__init__()  # 调用父类初始化
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)  # 第一个 RMS 归一化
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)  # 第二个 RMS 归一化

        self.attn = VisionAttention(  # 视觉注意力模块
            embed_dim=dim,  # 嵌入维度
            num_heads=num_heads,  # 注意力头数
            projection_size=dim,  # 投影大小
            use_qkv_parallel=True,  # 使用 QKV 并行
            proj_bias=True,  # 投影使用偏置
            flatten_batch=True,  # 展平批次
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
            num_dummy_heads=num_dummy_heads,  # 虚拟头数
            use_data_parallel=use_data_parallel,  # 是否使用数据并行
        )
        self.mlp = Qwen2_5_VLMLP(  # 视觉 MLP 模块
            dim,  # 维度
            intermediate_dim,  # 中间维度
            hidden_act=hidden_act,  # 激活函数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
            use_data_parallel=use_data_parallel,  # 是否使用数据并行
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_ws=None,
    ) -> torch.Tensor:
        """视觉 Transformer 块前向传播：归一化 -> 注意力 -> 残差 -> 归一化 -> MLP -> 残差"""
        S, B, H = x.shape  # 解包序列长度、批次大小和隐藏维度
        # norm1: flatten to 2D -> [S*B, H], then reshape back  # 归一化1：展平为2D再恢复
        x2d = x.reshape(-1, H)  # 展平为 2D
        hidden_states = self.norm1(x2d).reshape(S, B, H)  # 归一化并恢复形状

        # Attention expects [B, S, H]  # 注意力期望 [B, S, H] 格式
        hidden_states = rearrange(hidden_states, "s b h -> b s h")  # 重排为 [B, S, H]
        attn = self.attn(  # 注意力计算
            hidden_states,  # 隐藏状态
            cu_seqlens=cu_seqlens,  # 变长序列边界
            position_embeddings=position_embeddings,  # 位置嵌入
            output_ws=output_ws,  # 输出权重
        )
        attn = rearrange(attn, "b s h -> s b h")  # 重排回 [S, B, H]

        # norm2 with fused residual-add: also 2D  # 归一化2与融合残差加法：也是2D
        attn2d = attn.reshape(-1, H)  # 展平注意力输出
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)  # 归一化并加残差
        x_norm = x_norm_2d.reshape(S, B, H)  # 恢复归一化后形状
        x_after_add = x_after_add_2d.reshape(S, B, H)  # 恢复残差后形状

        # MLP and final residual  # MLP 和最终残差
        mlp_out = self.mlp(x_norm)  # MLP 计算
        x = x_after_add + mlp_out  # 加残差
        return x  # 返回输出


class Qwen2_5_VisionPatchMerger(nn.Module):
    """Qwen2.5 视觉补丁合并器，将视觉特征合并并投影到语言模型维度"""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        """初始化视觉补丁合并器"""
        super().__init__()  # 调用父类初始化
        self.hidden_size = context_dim * (spatial_merge_size**2)  # 合并后的隐藏维度
        self.ln_q = RMSNorm(context_dim, eps=1e-6)  # RMS 归一化
        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()  # 张量并行大小
        tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()  # 张量并行排名
        self.mlp = nn.ModuleList(  # MLP 模块列表
            [
                ColumnParallelLinear(  # 列并行全连接层
                    self.hidden_size,  # 输入维度
                    self.hidden_size,  # 输出维度
                    bias=True,  # 使用偏置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix("mlp.0", prefix),  # 参数前缀
                    tp_size=tp_size,  # 张量并行大小
                    tp_rank=tp_rank,  # 张量并行排名
                ),
                nn.GELU(),  # GELU 激活函数
                RowParallelLinear(  # 行并行全连接层
                    self.hidden_size,  # 输入维度
                    dim,  # 输出维度（语言模型维度）
                    bias=True,  # 使用偏置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix("mlp.2", prefix),  # 参数前缀
                    tp_size=tp_size,  # 张量并行大小
                    tp_rank=tp_rank,  # 张量并行排名
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """补丁合并器前向传播：归一化 -> 空间合并 -> MLP"""
        # x expected shape: [S, B, context_dim]  # 期望输入形状
        S, B, D = x.shape  # 解包形状
        x2d = x.reshape(-1, D)  # 展平为 2D
        x2d = self.ln_q(x2d)  # RMSNorm expects 2D  # RMS 归一化期望 2D 输入
        x2d = x2d.view(-1, self.hidden_size)  # group into spatial_merge_unit  # 按空间合并单元分组
        mlp_fc1, mlp_act, mlp_fc2 = self.mlp  # 解包 MLP 层
        x_parallel, _ = mlp_fc1(x2d)  # 通过第一个全连接层
        x_parallel = mlp_act(x_parallel)  # 通过激活函数
        out, _ = mlp_fc2(x_parallel)  # 通过第二个全连接层
        return out  # 返回输出


class Qwen2_5_VisionTransformer(nn.Module, RotaryPosMixin):
    """Qwen2.5 视觉 Transformer 编码器，处理图像/视频输入"""

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        max_context_len: Optional[int] = None,
    ) -> None:
        """初始化视觉 Transformer 编码器"""
        super().__init__()  # 调用父类初始化

        patch_size: int = vision_config.patch_size  # 补丁大小
        temporal_patch_size: int = vision_config.temporal_patch_size  # 时间补丁大小
        spatial_merge_size: int = vision_config.spatial_merge_size  # 空间合并大小
        self.spatial_merge_size = spatial_merge_size  # 保存空间合并大小
        self.spatial_merge_unit: int = spatial_merge_size * spatial_merge_size  # 空间合并单元
        in_channels: int = vision_config.in_channels  # 输入通道数
        hidden_size: int = vision_config.hidden_size  # 隐藏维度
        depth: int = vision_config.depth  # Transformer 深度
        num_heads: int = vision_config.num_heads  # 注意力头数
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes  # 全注意力块索引
        self.window_size = vision_config.window_size  # 窗口大小
        self.patch_size = vision_config.patch_size  # 补丁大小
        mlp_hidden_size: int = ((vision_config.intermediate_size + 7) // 8) * 8  # MLP 隐藏维度（8对齐）
        self.use_data_parallel = use_data_parallel  # 是否使用数据并行
        self.out_hidden_size = vision_config.out_hidden_size  # 输出隐藏维度
        self.patch_embed = Qwen2_5_VisionPatchEmbed(  # 补丁嵌入层
            patch_size=patch_size,  # 补丁大小
            temporal_patch_size=temporal_patch_size,  # 时间补丁大小
            in_channels=in_channels,  # 输入通道数
            embed_dim=hidden_size,  # 嵌入维度
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)  # 创建归一化层工厂
        head_dim = hidden_size // num_heads  # 每个头的维度
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)  # 旋转位置嵌入
        self.blocks = nn.ModuleList(  # Transformer 块列表
            [
                Qwen2_5_VisionBlock(  # 视觉块
                    dim=hidden_size,  # 维度
                    intermediate_dim=mlp_hidden_size,  # 中间维度
                    num_heads=num_heads,  # 头数
                    hidden_act=vision_config.hidden_act,  # 激活函数
                    norm_layer=norm_layer,  # 归一化层
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"blocks.{i}", prefix),  # 参数前缀
                    use_data_parallel=use_data_parallel,  # 数据并行
                )
                for i in range(depth)  # 遍历所有层
            ]
        )
        self.merger = Qwen2_5_VisionPatchMerger(  # 补丁合并器
            dim=vision_config.out_hidden_size,  # 输出维度
            context_dim=hidden_size,  # 上下文维度
            spatial_merge_size=spatial_merge_size,  # 空间合并大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("merger", prefix),  # 参数前缀
            use_data_parallel=use_data_parallel,  # 数据并行
        )

        # Resource prepared for vit cuda graph  # 为 ViT CUDA Graph 准备资源
        self.tp_size = (  # 张量并行大小
            1 if use_data_parallel else get_tensor_model_parallel_world_size()  # 数据并行时为 1
        )
        self.max_context_len = max_context_len  # 最大上下文长度
        self.enable_cg = _is_cuda and envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get()  # 是否启用 CUDA Graph

        self.cuda_graph_runner: Optional[ViTCudaGraphRunner] = None  # CUDA Graph 运行器
        if self.enable_cg:  # 如果启用 CUDA Graph
            self.cuda_graph_runner = ViTCudaGraphRunner(self)  # 创建运行器

    def get_window_index(self, grid_thw):
        """根据网格维度计算窗口索引和累积窗口序列长度"""
        cu_window_seqlens: list = [0]  # 累积窗口序列长度列表
        window_index_id = 0  # 窗口索引 ID
        vit_merger_window_size = (  # ViT 合并器窗口大小
            self.window_size // self.spatial_merge_size // self.patch_size  # 根据窗口大小和合并比计算
        )
        window_index: list = []  # 窗口索引列表
        for grid_t, grid_h, grid_w in grid_thw:  # 遍历每个图像的网格维度
            llm_grid_h, llm_grid_w = (  # LLM 网格高度和宽度
                grid_h // self.spatial_merge_size,  # 除以空间合并大小
                grid_w // self.spatial_merge_size,  # 除以空间合并大小
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(  # 创建索引
                grid_t, llm_grid_h, llm_grid_w  # 重塑为 3D
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size  # 高度填充
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size  # 宽度填充
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size  # 高度方向窗口数
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size  # 宽度方向窗口数
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)  # 填充索引
            index_padded = index_padded.reshape(  # 重塑为 5D
                grid_t,  # 时间维度
                num_windows_h,  # 高度窗口数
                vit_merger_window_size,  # 窗口高度
                num_windows_w,  # 宽度窗口数
                vit_merger_window_size,  # 窗口宽度
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(  # 排列并重塑为 3D
                grid_t,  # 时间维度
                num_windows_h * num_windows_w,  # 总窗口数
                vit_merger_window_size,  # 窗口高度
                vit_merger_window_size,  # 窗口宽度
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)  # 计算每个窗口的有效长度
            index_padded = index_padded.reshape(-1)  # 展平索引
            index_new = index_padded[index_padded != -100]  # 过滤填充值
            window_index.append(index_new + window_index_id)  # 添加偏移后的索引
            cu_seqlens_tmp = (  # 计算临时累积序列长度
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]  # 乘以合并单元并累加
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())  # 添加到列表
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()  # 更新索引偏移
        window_index = torch.cat(window_index, dim=0)  # 拼接所有窗口索引
        return window_index, cu_window_seqlens  # 返回窗口索引和累积窗口序列长度

    @property
    def dtype(self) -> torch.dtype:
        """获取模型数据类型"""
        return self.patch_embed.proj.weight.dtype  # 返回补丁嵌入权重的数据类型

    @property
    def device(self) -> torch.device:
        """获取模型设备"""
        return self.patch_embed.proj.weight.device  # 返回补丁嵌入权重的设备

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """计算旋转位置编码"""
        pos_ids = []  # 位置 ID 列表
        for t, h, w in grid_thw:  # 遍历每个图像的时间、高度、宽度维度
            base = self.rot_pos_ids(h, w, self.spatial_merge_size)  # 计算基础位置 ID
            pos_ids.append(base if t == 1 else base.repeat(t, 1))  # 如果多帧则重复

        pos_ids = torch.cat(pos_ids, dim=0)  # 拼接所有位置 ID
        max_grid_size = grid_thw[:, 1:].max()  # 获取最大网格尺寸
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)  # 计算完整旋转位置编码
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)  # 索引并展平
        return rotary_pos_emb  # 返回旋转位置编码

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """视觉 Transformer 前向传播：补丁嵌入 -> 位置编码 -> Transformer 块 -> 合并器"""
        if self.enable_cg:  # 如果启用 CUDA Graph
            return self.forward_with_cuda_graph(x, grid_thw)  # 使用 CUDA Graph 前向传播

        # patchify  # 补丁化
        x = x.to(device=self.device, dtype=self.dtype)  # 转换设备和数据类型
        x = self.patch_embed(x)  # 补丁嵌入

        # compute position embedding  # 计算位置嵌入
        rotary_pos_emb = self.rot_pos_emb(grid_thw)  # 旋转位置编码

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)  # 获取窗口索引
        cu_window_seqlens = torch.tensor(  # 转换为张量
            cu_window_seqlens,  # 累积窗口序列长度
            device=x.device,  # 设备
            dtype=torch.int32,  # 数据类型
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)  # 去除连续重复值

        # Move window_index to the same device as x before using it to index x  # 将窗口索引移到与 x 相同的设备
        window_index = window_index.to(device=x.device)  # 移动设备
        reverse_indices = permute_inv(window_index)  # 计算逆排列索引

        # Ensure rotary_pos_emb is on the same device/dtype as x  # 确保旋转位置编码与 x 同设备同类型
        rotary_pos_emb = rotary_pos_emb.to(device=x.device, dtype=x.dtype)  # 转换设备和类型

        seq_len, _ = x.size()  # 获取序列长度

        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)  # 按合并单元重塑
        x = x[window_index, :, :]  # 按窗口索引重排
        x = x.reshape(seq_len, -1)  # 恢复形状
        rotary_pos_emb = rotary_pos_emb.reshape(  # 重塑旋转位置编码
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1  # 按合并单元重塑
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]  # 按窗口索引重排
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)  # 恢复形状
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # 拼接得到完整嵌入
        position_embeddings = (emb.cos(), emb.sin())  # 分解为余弦和正弦
        # After building position_embeddings, make sure both cos and sin are on the same device/dtype as the attention input  # 确保位置嵌入与注意力输入同设备同类型
        position_embeddings = (  # 转换设备和类型
            position_embeddings[0].to(x.device, x.dtype),  # 余弦分量
            position_embeddings[1].to(x.device, x.dtype),  # 正弦分量
        )

        # compute cu_seqlens - move cu_seqlens to GPU and make it int32  # 计算累积序列长度
        cu_seqlens = torch.cat(  # 拼接累积序列长度
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),  # 起始位置
                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2])  # 每个图像的 patch 总数
                .cumsum(dim=0)  # 累积求和
                .to(device=x.device, dtype=torch.int32),  # 转换设备和类型
            ]
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])  # 在前面补零
        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction  # NPU 上 cu_seqlens 必须在 CPU
        if is_npu():  # 如果是 NPU 设备
            cu_seqlens = cu_seqlens.to("cpu")  # 移到 CPU
            cu_window_seqlens = cu_window_seqlens.to("cpu")  # 移到 CPU
        # transformers  # Transformer 块
        x = x.unsqueeze(1)  # 增加维度
        for layer_num, blk in enumerate(self.blocks):  # 遍历所有块
            fullatt_indexes = self.fullatt_block_indexes  # 全注意力块索引
            if isinstance(fullatt_indexes, torch.Tensor):  # 如果是张量
                fullatt_indexes = fullatt_indexes.tolist()  # 转换为列表
            if layer_num in fullatt_indexes:  # 如果是全注意力块
                cu_seqlens_now = cu_seqlens  # 使用全序列长度
            else:  # 窗口注意力块
                cu_seqlens_now = cu_window_seqlens  # 使用窗口序列长度
            x = blk(  # 通过当前块
                x, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings  # 传入参数
            )

        # adapter  # 适配器
        x = self.merger(x)  # 通过合并器
        x = x[reverse_indices, :]  # 恢复原始顺序

        return x  # 返回输出

    def forward_with_cuda_graph(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """使用 CUDA Graph 加速的视觉 Transformer 前向传播"""
        # patchify  # 补丁化
        x = x.to(device=self.device, dtype=self.dtype)  # 转换设备和类型
        x = self.patch_embed(x)  # 补丁嵌入

        # compute position embedding  # 计算位置嵌入
        rotary_pos_emb = self.rot_pos_emb(grid_thw)  # 旋转位置编码

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)  # 获取窗口索引
        cu_window_seqlens = torch.tensor(  # 转换为张量
            cu_window_seqlens,  # 累积窗口序列长度
            device=x.device,  # 设备
            dtype=torch.int32,  # 数据类型
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)  # 去除连续重复值

        window_index = window_index.to(device=x.device)  # 移动设备
        reverse_indices = permute_inv(window_index)  # 计算逆排列索引
        rotary_pos_emb = rotary_pos_emb.to(device=x.device, dtype=x.dtype)  # 转换设备和类型

        # patch token num  # 补丁 token 数量
        seq_len, _ = x.size()  # 获取序列长度

        # [G, M, hidden]  # 形状说明
        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)  # 按合并单元重塑
        x = x[window_index, :, :]  # [G, M, hidden]  # 按窗口索引重排
        x = x.reshape(seq_len, -1)  # [seq_len, hidden]  # 恢复形状

        rotary_pos_emb = rotary_pos_emb.reshape(  # 重塑旋转位置编码
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1  # 按合并单元重塑
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]  # 按窗口索引重排
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)  # 恢复形状

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # 拼接得到完整嵌入
        position_embeddings = (emb.cos(), emb.sin())  # 分解为余弦和正弦
        # After building position_embeddings, make sure both cos and sin are on  # 构建位置嵌入后确保同设备同类型
        # the same device/dtype as the attention input  # 与注意力输入相同
        position_embeddings = (  # 转换设备和类型
            position_embeddings[0].to(x.device, x.dtype),  # 余弦分量
            position_embeddings[1].to(x.device, x.dtype),  # 正弦分量
        )

        # compute cu_seqlens - move cu_seqlens to GPU and make it int32  # 计算累积序列长度
        cu_seqlens = torch.cat(  # 拼接累积序列长度
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),  # 起始位置
                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2])  # 每个图像的 patch 总数
                .cumsum(dim=0)  # 累积求和
                .to(device=x.device, dtype=torch.int32),  # 转换设备和类型
            ]
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])  # 在前面补零

        return self.cuda_graph_runner.run(  # 通过 CUDA Graph 运行器执行
            x=x,  # 输入
            position_embeddings=position_embeddings,  # 位置嵌入
            cu_seqlens=cu_seqlens,  # 累积序列长度
            cu_window_seqlens=cu_window_seqlens,  # 窗口序列长度
            output_indices=reverse_indices,  # 输出索引
        )


class Qwen2_5_VLForConditionalGeneration(nn.Module):
    """Qwen2.5-VL 条件生成模型，整合视觉编码器和语言模型"""

    # BitandBytes specific attributes  # BitandBytes 特定属性
    default_bitsandbytes_target_modules = [  # 默认 BitandBytes 目标模块
        ".gate_up_proj.",  # 门控上投影
        ".down_proj.",  # 下投影
        ".q_proj.",  # Q 投影
        ".k_proj.",  # K 投影
        ".v_proj.",  # V 投影
        ".o_proj.",  # O 投影
    ]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes 堆叠参数映射
        # shard_name, weight_name, index  # 分片名, 权重名, 索引
        "q_proj": ("qkv_proj", 0),  # Q 映射
        "k_proj": ("qkv_proj", 1),  # K 映射
        "v_proj": ("qkv_proj", 2),  # V 映射
        "gate_proj": ("gate_up_proj", 0),  # gate 映射
        "up_proj": ("gate_up_proj", 1),  # up 映射
    }

    packed_modules_mapping = {  # 打包模块映射
        "gate_up_proj": ["gate_proj", "up_proj"],  # 门控上投影映射
    }
    # To ensure correct weight loading and mapping.  # 确保正确的权重加载和映射
    hf_to_sglang_mapper = WeightsMapper(  # HuggingFace 到 SGLang 的权重映射器
        orig_to_new_substr={  # 子字符串映射
            "attn.qkv": "attn.qkv_proj",  # 注意力 QKV 映射
        },
        orig_to_new_prefix={  # 前缀映射
            # mapping for new names in checkpoint saved after transformers v4.52  # transformers v4.52 后的检查点映射
            "model.language_model.": "language_model.model.",  # 语言模型前缀映射
            "model.visual.": "visual.",  # 视觉模型前缀映射
            # mapping for original checkpoint  # 原始检查点映射
            "lm_head.": "language_model.lm_head.",  # 语言模型头映射
            "model.": "language_model.model.",  # 模型前缀映射
        },
    )

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化 Qwen2.5-VL 条件生成模型"""
        super().__init__()  # 调用父类初始化

        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.config = config  # 保存配置
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder  # 是否启用数据并行编码器

        if not self.config.encoder_only:  # 如果不是仅编码器模式
            self.model = Qwen2Model(  # 创建 Qwen2 语言模型
                config,  # 配置
                quant_config,  # 量化配置
                prefix=add_prefix("model", prefix),  # 参数前缀
            )

            if self.pp_group.is_last_rank:  # 如果是最后一个流水线并行排名
                if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:  # 单卡且共享词嵌入
                    self.lm_head = self.model.embed_tokens  # 共享嵌入
                else:  # 非共享词嵌入
                    self.lm_head = ParallelLMHead(  # 创建并行语言模型头
                        self.config.vocab_size,  # 词表大小
                        self.config.hidden_size,  # 隐藏维度
                        quant_config=quant_config,  # 量化配置
                        prefix=add_prefix("lm_head", prefix),  # 参数前缀
                    )
            else:  # 非最后一个排名
                # ranks other than the last rank will have a placeholder layer  # 非最后排名使用占位层
                self.lm_head = PPMissingLayer()  # 流水线缺失层
        else:  # 仅编码器模式
            # encoder_only mode: no language model, so no lm_head needed  # 仅编码器模式不需要语言模型头
            self.lm_head = None  # 语言模型头为空

        self.visual = Qwen2_5_VisionTransformer(  # 创建视觉编码器
            config.vision_config,  # 视觉配置
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),  # 归一化 epsilon
            # NOTE: Qwen2_5-VL vision encoder currently supports BitsAndBytes 4-bit quantization.  # 注意：当前仅支持 BitsAndBytes 4-bit 量化
            # Other quantization methods (e.g., GPTQ, AWQ) are untested and may not be supported.  # 其他量化方法未经测试
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("visual", prefix),  # 参数前缀
            use_data_parallel=self.use_data_parallel,  # 数据并行
            max_context_len=self.config.max_position_embeddings,  # 最大上下文长度
        )

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling  # 是否启用多模态旋转位置编码

        self.logits_processor = LogitsProcessor(config)  # 创建 logits 处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 创建池化器

        # For EAGLE3 support  # EAGLE3 支持
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """使用多模态 token 填充模式对输入 ID 进行填充"""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 返回填充后的 token

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """从多模态数据项中提取图像特征"""
        # in qwen-vl, last dim is the same  # 在 qwen-vl 中，最后维度相同
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(  # 拼接像素值并转换类型
            self.visual.dtype  # 视觉编码器数据类型
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)  # 拼接网格维度

        expected_dim = getattr(self.visual, "embed_dim", -1)  # 期望的嵌入维度

        if expected_dim == -1:  # 如果未获取到
            vision_conf = self.config.vision_config  # 获取视觉配置
            expected_dim = getattr(  # 尝试获取嵌入维度
                vision_conf, "embed_dim", getattr(vision_conf, "hidden_size", -1)  # 依次尝试 embed_dim 和 hidden_size
            )

        raw_patch_dim = 1176  # 原始补丁维度

        if pixel_values.dim() == 2:  # 如果像素值是 2 维
            current_dim = pixel_values.shape[-1]  # 当前维度
            if current_dim == expected_dim:  # 如果维度匹配
                return pixel_values  # 直接返回
            if current_dim != raw_patch_dim:  # 如果不是原始补丁维度

                return pixel_values  # 直接返回

        assert pixel_values.dim() == 2, pixel_values.dim()  # 断言像素值为 2 维
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()  # 断言网格维度为 2 维
        if self.use_data_parallel:  # 如果使用数据并行
            return run_dp_sharded_mrope_vision_model(  # 运行 DP 分片 mrope 视觉模型
                self.visual, pixel_values, image_grid_thw.tolist(), rope_type="rope_3d"  # 传入参数
            )
        else:  # 非数据并行
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)  # 通过视觉编码器
        return image_embeds  # 返回图像嵌入

    _lora_pattern = re.compile(  # LoRA 匹配正则表达式
        r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)$"  # 匹配 LoRA 目标层
    )

    def should_apply_lora(self, module_name: str) -> bool:
        """判断是否应对指定模块应用 LoRA"""
        return bool(self._lora_pattern.match(module_name))  # 返回是否匹配 LoRA 模式

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """从多模态数据项中提取视频特征"""
        # in qwen-vl, last dim is the same  # 在 qwen-vl 中，最后维度相同
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(  # 拼接像素值并转换类型
            self.visual.dtype  # 视觉编码器数据类型
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)  # 拼接视频网格维度
        assert pixel_values.dim() == 2, pixel_values.dim()  # 断言像素值为 2 维
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()  # 断言网格维度为 2 维
        if self.use_data_parallel:  # 如果使用数据并行
            return run_dp_sharded_mrope_vision_model(  # 运行 DP 分片 mrope 视觉模型
                self.visual, pixel_values, video_grid_thw.tolist(), rope_type="rope_3d"  # 传入参数
            )
        else:  # 非数据并行
            video_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)  # 通过视觉编码器
        return video_embeds  # 返回视频嵌入

    def post_process(
        self,
        inputs_embeds,
        modalities: List[Modality],
        embeddings: List[torch.Tensor],
        indices: List[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """多模态嵌入后处理，过滤空嵌入"""
        # Placeholder for post_process  # 后处理占位
        new_embeddings = []  # 新嵌入列表
        for i, (modality, embedding, index) in enumerate(  # 遍历模态、嵌入和索引
            zip(modalities, embeddings, indices)  # 打包遍历
        ):
            if embedding is None or index is None:  # 如果嵌入或索引为空
                continue  # 跳过

            new_embeddings.append(embedding)  # 添加非空嵌入
        return new_embeddings, forward_batch  # 返回新嵌入列表和前向批次

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.model.embed_tokens  # 返回词嵌入层

    @torch.no_grad()  # 禁用梯度计算
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """Qwen2.5-VL 前向传播

        Args:
            input_ids: 扁平化的输入 ID
            positions: 扁平化的位置 ID，mrope 启用时形状为 (3, seq_len)
            forward_batch: 前向批次信息
        """
        """Run forward pass for Qwen2_5-VL.  # 运行 Qwen2.5-VL 前向传播

        Args:  # 参数
            input_ids: Flattened (concatenated) input_ids corresponding to a  # 扁平化的输入 ID
                batch.  # 批次
            positions: Flattened (concatenated) position ids corresponding to a  # 扁平化的位置 ID
                batch.  # 批次
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL  # 注意：如果启用 mrope
                opensource models), the shape will be `(3, seq_len)`,  # 开源模型，形状为 (3, seq_len)
                otherwise it will be `(seq_len,).  # 否则为 (seq_len,)
                (Use input_metadata.mrope_positions to replace it)  # 使用 mrope_positions 替代
        """
        if self.is_mrope_enabled:  # 如果启用 mrope
            positions = forward_batch.mrope_positions  # 使用多模态旋转位置

        if not (  # 如果不是
            forward_batch.forward_mode.is_decode()  # 解码模式
            or not forward_batch.contains_image_inputs()  # 或不包含图像输入
        ):
            if self.is_mrope_enabled:  # 如果启用 mrope
                assert positions.ndim == 2 and positions.size(0) == 3, (  # 断言位置维度正确
                    "multimodal section rotary embedding requires "  # 多模态旋转嵌入需要
                    f"(3, seq_len) positions, but got {positions.size()}"  # (3, seq_len) 格式的位置
                )

        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入例程
            input_ids=input_ids,  # 输入 ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.model,  # 语言模型
            multimodal_model=self,  # 多模态模型（自身）
            positions=positions,  # 位置信息
            pp_proxy_tensors=pp_proxy_tensors,  # 流水线代理张量
        )

        aux_hidden_states = None  # 辅助隐藏状态
        if self.capture_aux_hidden_states:  # 如果捕获辅助隐藏状态
            hidden_states, aux_hidden_states = hidden_states  # 解包隐藏状态

        if self.pp_group.is_last_rank:  # 如果是最后一个流水线排名
            if not get_embedding:  # 如果不需要嵌入
                return self.logits_processor(  # 返回 logits 处理结果
                    input_ids,  # 输入 ID
                    hidden_states,  # 隐藏状态
                    self.lm_head,  # 语言模型头
                    forward_batch,  # 前向批次
                    aux_hidden_states,  # 辅助隐藏状态
                )
            else:  # 需要嵌入
                return self.pooler(hidden_states, forward_batch)  # 返回池化结果
        else:  # 非最后排名
            return hidden_states  # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，处理堆叠参数映射和词嵌入共享"""
        stacked_params_mapping = [  # 堆叠参数映射表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),  # Q 映射
            (".qkv_proj", ".k_proj", "k"),  # K 映射
            (".qkv_proj", ".v_proj", "v"),  # V 映射
            ("gate_up_proj", "up_proj", 1),  # up 映射
            ("gate_up_proj", "gate_proj", 0),  # gate 映射
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入逆频率
                continue

            if (  # 如果共享词嵌入
                self.config.tie_word_embeddings  # 共享词嵌入标志
                and self.pp_group.is_last_rank  # 最后一个排名
                and "model.embed_tokens.weight" in name  # 嵌入权重
            ):
                if "lm_head.weight" in params_dict:  # 如果语言模型头权重存在
                    lm_head_param = params_dict["lm_head.weight"]  # 获取语言模型头参数
                    weight_loader = getattr(  # 获取权重加载器
                        lm_head_param, "weight_loader", default_weight_loader  # 默认权重加载器
                    )
                    weight_loader(lm_head_param, loaded_weight)  # 加载权重

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue
                if (  # 如果是视觉模块且非 MLP 堆叠
                    "visual" in name  # 视觉模块
                    and "up_proj" not in name  # 不是 up 投影
                    and "gate_proj" not in name  # 不是 gate 投影
                ):
                    continue
                name = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                layer_id = get_layer_id(name)  # 获取层 ID
                if (  # 如果层在当前流水线阶段之外
                    layer_id is not None  # 层 ID 存在
                    and hasattr(self, "model")  # 有 model 属性
                    and hasattr(self.model, "start_layer")  # 有起始层属性
                    and (  # 且层 ID 在范围之外
                        layer_id < self.model.start_layer  # 低于起始层
                        or layer_id >= self.model.end_layer  # 或高于等于结束层
                    )
                ):
                    continue

                # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果偏置不在参数字典中
                    continue
                # Skip loading visual/language model weights  # 跳过视觉/语言模型权重
                if (  # 如果是仅编码器或仅语言模式
                    self.config.encoder_only or self.config.language_only  # 仅编码器或仅语言
                ) and name not in params_dict:  # 且参数不在字典中
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break
            else:  # 非堆叠参数处理
                if "visual" in name:  # 视觉模块参数
                    # adapt to VisionAttention  # 适配 VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")  # 替换注意力参数名

                try:  # 尝试加载参数
                    # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置
                    if name.endswith(".bias") and name not in params_dict:  # 如果偏置不在参数字典中
                        continue
                    if name in params_dict.keys():  # 如果参数在字典中
                        param = params_dict[name]  # 获取参数
                    else:  # 参数不在字典中
                        continue

                except KeyError:  # 参数未找到
                    print(params_dict.keys())  # 打印可用参数名
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重

    def get_embed_and_head(self):
        """获取嵌入层和语言模型头的权重"""
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入权重和语言模型头权重

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        """设置 EAGLE3 推测解码需要捕获辅助隐藏状态的层"""
        self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
        self.model.capture_aux_hidden_states = True  # 语言模型也启用
        if layer_ids is None:  # 如果未指定层 ID
            num_layers = self.config.num_hidden_layers  # 获取总层数
            self.model.layers_to_capture = [  # 设置默认捕获层
                2,  # 第 2 层
                num_layers // 2,  # 中间层
                num_layers - 3,  # 倒数第 3 层
            ]  # Specific layers for EAGLE3 support  # EAGLE3 支持的特定层
        else:  # 指定了层 ID
            self.model.layers_to_capture = [val + 1 for val in layer_ids]  # 偏移 1 后设置


EntryClass = [Qwen2_5_VLForConditionalGeneration]  # 模型入口类列表
