# MiMo视觉模型：注意力机制和视觉Transformer实现
# 该模块实现了MiMo多模态模型中的视觉编码器部分
# 包含视觉配置、补丁嵌入、视觉块、视觉Transformer等组件
# 支持行列双方向位置编码和窗口注意力机制

"""Inference-only MiMo vision model: attention + ViT."""

from __future__ import annotations  # 启用延迟类型注解评估

from functools import partial  # 导入偏函数工具
from typing import Optional, Tuple, Type  # 导入类型注解

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数式接口
from einops import rearrange  # 导入einops重排工具
from transformers.configuration_utils import PretrainedConfig  # 导入预训练配置基类
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # 导入Qwen2.5 VL旋转嵌入
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionPatchMerger, Qwen2_5_VLMLP  # 导入Qwen2.5 VL视觉组件
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


class MiMoVLVisionConfig(PretrainedConfig):
    """MiMo视觉模型配置类"""
    model_type = "mimovl"  # 模型类型标识
    base_config_key = "vision_config"  # 基础配置键名

    def __init__(
        self,
        depth=28,
        hidden_size=1280,
        hidden_act="silu",
        intermediate_size=4608,
        num_heads=32,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=2,
        window_size=128,
        out_hidden_size=2048,
        fullatt_block_indexes=[7, 15, 23, 31],
        initializer_range=0.02,
        kv_channels=64,
        qk_channels=64,
        num_query_groups=4,
        num_key_value_heads=8,
        vit_window_attn_types=None,
        visual_token_window_size=64,
        **kwargs,
    ):
        """初始化MiMo视觉配置"""
        super().__init__(**kwargs)  # 调用父类初始化

        self.depth = depth  # Transformer深度（层数）
        self.hidden_size = hidden_size  # 隐藏层大小
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.intermediate_size = intermediate_size  # 中间层大小
        self.num_heads = num_heads  # 注意力头数
        if num_key_value_heads is None:  # 如果KV头数未指定
            num_key_value_heads = num_heads  # 默认等于注意力头数
        self.num_key_value_heads = num_key_value_heads  # KV头数
        self.in_channels = in_channels  # 输入通道数
        self.patch_size = patch_size  # 补丁大小
        self.spatial_merge_size = spatial_merge_size  # 空间合并大小
        self.temporal_patch_size = temporal_patch_size  # 时间补丁大小
        self.tokens_per_second = tokens_per_second  # 每秒token数
        self.window_size = window_size  # 窗口大小
        self.fullatt_block_indexes = fullatt_block_indexes  # 全注意力块索引
        self.out_hidden_size = out_hidden_size  # 输出隐藏层大小
        self.initializer_range = initializer_range  # 初始化范围
        self.kv_channels = kv_channels  # KV通道数
        self.qk_channels = qk_channels  # QK通道数
        self.num_query_groups = num_query_groups  # 查询组数
        self.vit_window_attn_types = vit_window_attn_types or [-1] * depth  # ViT窗口注意力类型
        self.visual_token_window_size = visual_token_window_size  # 视觉token窗口大小


class MiMoVisionPatchEmbed(nn.Module):
    """MiMo视觉补丁嵌入层，使用3D卷积将图像/视频转换为补丁嵌入"""
    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1536,
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.patch_size = patch_size  # 空间补丁大小
        self.temporal_patch_size = temporal_patch_size  # 时间补丁大小
        self.in_channels = in_channels  # 输入通道数
        self.embed_dim = embed_dim  # 嵌入维度

        kernel_size = [temporal_patch_size, patch_size, patch_size]  # 3D卷积核大小
        self.proj = nn.Conv3d(  # 3D卷积投影
            in_channels,  # 输入通道
            embed_dim,  # 输出维度
            kernel_size=kernel_size,  # 卷积核大小
            stride=kernel_size,  # 步幅等于卷积核大小
            bias=False,  # 不使用偏置
        )
        self.proj_weight_linear_format = None  # 线性格式投影权重

    @torch.no_grad()
    def sync_proj_weight_linear_format(self):
        """将卷积权重同步为线性格式，用于高效推理"""
        self.proj_weight_linear_format = self.proj.weight.view(self.embed_dim, -1)  # 重塑权重

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播：使用线性投影计算补丁嵌入"""
        target_dtype = self.proj.weight.dtype  # 目标数据类型
        hidden_states = F.linear(  # 使用线性运算替代卷积
            hidden_states.to(dtype=target_dtype), self.proj_weight_linear_format
        )
        return hidden_states  # 返回嵌入结果


class MiMoVisionBlock(nn.Module):
    """MiMo视觉Transformer块，包含注意力和MLP"""
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        hidden_act="silu",
        norm_layer: Type[nn.Module] = None,
        attn_implementation: Optional[str] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        num_dummy_heads: int = 0,
        rms_norm_eps: float = 1e-6,
        use_sink: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()  # 调用父类初始化
        if norm_layer is None:  # 如果没有指定归一化层
            norm_layer = partial(nn.LayerNorm, eps=1e-6)  # 默认使用LayerNorm
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)  # 第一个RMS归一化
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)  # 第二个RMS归一化
        self.use_data_parallel = use_data_parallel  # 是否使用数据并行

        if attn_implementation is None:  # 根据注意力实现设置参数
            softmax_in_single_precision = False  # 不使用单精度softmax
            qkv_backend = None  # QKV后端为空
            flatten_batch = True  # 展平批次
        elif attn_implementation == "sdpa":  # SDPA实现
            softmax_in_single_precision = False
            qkv_backend = "sdpa"
            flatten_batch = True
        elif attn_implementation == "flash_attention_2":  # Flash Attention 2
            softmax_in_single_precision = False
            qkv_backend = "triton_attn"
            flatten_batch = True
        elif attn_implementation == "eager":  # Eager实现
            softmax_in_single_precision = True
            qkv_backend = "sdpa"
            flatten_batch = True
        elif attn_implementation == "flash_attention_3":  # Flash Attention 3
            softmax_in_single_precision = False
            qkv_backend = "fa3"
            flatten_batch = True

        self.attn = VisionAttention(  # 视觉注意力层
            embed_dim=dim,  # 嵌入维度
            num_heads=num_heads,  # 头数
            num_kv_heads=num_kv_heads,  # KV头数
            head_dim=head_dim,  # 头维度
            projection_size=dim,  # 投影尺寸
            use_qkv_parallel=True,  # 使用QKV并行
            proj_bias=True,  # 投影偏置
            qkv_bias=True,  # QKV偏置
            qkv_backend=qkv_backend,  # QKV后端
            softmax_in_single_precision=softmax_in_single_precision,  # 单精度softmax
            flatten_batch=flatten_batch,  # 展平批次
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 前缀
            num_dummy_heads=num_dummy_heads,  # 虚拟头数
            use_sink=use_sink,  # 是否使用汇聚
            window_size=window_size,  # 窗口大小
            use_data_parallel=use_data_parallel,  # 数据并行
        )
        self.mlp = Qwen2_5_VLMLP(  # MLP层，复用Qwen2.5 VL的实现
            dim,  # 输入维度
            intermediate_dim,  # 中间维度
            hidden_act=hidden_act,  # 激活函数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 前缀
            use_data_parallel=use_data_parallel,  # 数据并行
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: torch.Tensor,
        full_attn: bool = True,
    ) -> torch.Tensor:
        """视觉块前向传播：注意力+MLP残差连接"""
        S, B, H = x.shape  # 获取序列长度、批次大小和隐藏维度
        # norm1: flatten to 2D -> [S*B, H], then reshape back
        x2d = x.reshape(-1, H)  # 展平为2D
        hidden_states = self.norm1(x2d).reshape(S, B, H)  # 归一化后重塑

        # Attention expects [B, S, H]
        hidden_states = rearrange(hidden_states, "s b h -> b s h")  # 重排维度
        attn = self.attn(  # 计算注意力
            hidden_states,  # 隐藏状态
            cu_seqlens=cu_seqlens,  # 累计序列长度
            max_seqlen=max_seqlen,  # 最大序列长度
            position_embeddings=position_embeddings,  # 位置嵌入
            full_attn=full_attn,  # 是否全注意力
        )
        attn = rearrange(attn, "b s h -> s b h")  # 重排回原维度

        # norm2 with fused residual-add: also 2D
        attn2d = attn.reshape(-1, H)  # 展平注意力输出
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)  # 融合归一化和残差加法
        x_norm = x_norm_2d.reshape(S, B, H)  # 重塑归一化结果
        x_after_add = x_after_add_2d.reshape(S, B, H)  # 重塑残差加法结果

        # MLP and final residual
        mlp_out = self.mlp(x_norm)  # MLP计算
        x = x_after_add + mlp_out  # 加上MLP输出
        return x  # 返回输出


class MiMoVisionTransformer(nn.Module):
    """MiMo视觉Transformer，整合补丁嵌入、Transformer块和补丁合并器"""
    def __init__(
        self,
        vision_config: MiMoVLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.server_args = get_global_server_args()  # 获取全局服务器参数
        self.vit_window_attn_types = vision_config.vit_window_attn_types  # ViT窗口注意力类型
        patch_size: int = vision_config.patch_size  # 补丁大小
        temporal_patch_size: int = vision_config.temporal_patch_size  # 时间补丁大小
        spatial_merge_size: int = vision_config.spatial_merge_size  # 空间合并大小
        self.spatial_merge_size = spatial_merge_size  # 保存空间合并大小
        self.spatial_merge_unit: int = spatial_merge_size * spatial_merge_size  # 空间合并单元大小
        in_channels: int = vision_config.in_channels  # 输入通道数
        hidden_size: int = vision_config.hidden_size  # 隐藏层大小
        depth: int = vision_config.depth  # Transformer深度
        num_heads: int = vision_config.num_heads  # 注意力头数
        num_kv_heads = getattr(vision_config, "num_key_value_heads", None)  # KV头数
        if num_kv_heads is None:  # 如果未指定KV头数
            num_kv_heads = num_heads  # 默认等于注意力头数
        self.num_kv_heads = num_kv_heads  # 保存KV头数
        self.qk_channels = getattr(vision_config, "qk_channels", None)  # QK通道数
        self.kv_channels = getattr(vision_config, "kv_channels", None)  # KV通道数
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes  # 全注意力块索引
        self.window_size = vision_config.window_size  # 窗口大小
        self.patch_size = vision_config.patch_size  # 补丁大小
        self.use_data_parallel = self.server_args.mm_enable_dp_encoder  # 是否启用DP编码器
        mlp_hidden_size: int = vision_config.intermediate_size  # MLP隐藏层大小
        self.patch_embed = MiMoVisionPatchEmbed(  # 补丁嵌入层
            patch_size=patch_size,  # 补丁大小
            temporal_patch_size=temporal_patch_size,  # 时间补丁大小
            in_channels=in_channels,  # 输入通道
            embed_dim=hidden_size,  # 嵌入维度
        )
        self.use_sink = getattr(vision_config, "use_sink", False)  # 是否使用汇聚
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)  # 归一化层
        head_dim = (  # 计算头维度
            self.qk_channels
            if self.qk_channels is not None
            else hidden_size // num_heads
        )
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)  # 旋转位置嵌入
        self.visual_token_window_size = getattr(  # 视觉token窗口大小
            vision_config, "visual_token_window_size", -1
        )
        self.blocks = nn.ModuleList(  # Transformer块列表
            [
                MiMoVisionBlock(  # 视觉块
                    dim=hidden_size,  # 维度
                    intermediate_dim=mlp_hidden_size,  # 中间维度
                    num_heads=num_heads,  # 头数
                    hidden_act=vision_config.hidden_act,  # 激活函数
                    norm_layer=norm_layer,  # 归一化层
                    attn_implementation="flash_attention_3",  # 注意力实现
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"blocks.{i}", prefix),  # 前缀
                    use_sink=(  # 是否使用汇聚
                        self.use_sink if i not in self.fullatt_block_indexes else False
                    ),
                    window_size=(  # 窗口大小
                        self.visual_token_window_size,
                        self.visual_token_window_size,
                    ),
                    num_kv_heads=num_kv_heads,  # KV头数
                    head_dim=self.qk_channels,  # 头维度
                    use_data_parallel=self.use_data_parallel,  # 数据并行
                )
                for i in range(depth)  # 遍历每一层
            ]
        )

        self.vision_config = vision_config  # 保存视觉配置
        self.merger = Qwen2_5_VisionPatchMerger(  # 补丁合并器
            dim=vision_config.out_hidden_size,  # 输出维度
            context_dim=hidden_size,  # 上下文维度
            spatial_merge_size=spatial_merge_size,  # 空间合并大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("merger", prefix),  # 前缀
            use_data_parallel=self.use_data_parallel,  # 数据并行
        )
        self._post_init()  # 后初始化

    def apply_index(self, tensor: torch.Tensor, index: torch.Tensor):
        """根据索引重排张量的空间维度"""
        tensor = tensor.unflatten(0, (-1, self.spatial_merge_unit))  # 反展平空间合并单元
        tensor = tensor[index]  # 按索引重排
        tensor = tensor.flatten(0, 1)  # 展平回来
        return tensor  # 返回重排后的张量

    def _post_init(self):
        """后初始化：将所有偏置参数归零"""
        for name, param in self.named_parameters():  # 遍历所有参数
            if "bias" in name:  # 如果是偏置参数
                param.data.zero_()  # 归零

    def get_window_index_1d(self, grid_thw, col=True):
        """获取1D窗口索引，用于列方向窗口注意力的重排"""
        window_index: list = []  # 窗口索引列表
        window_index_id = 0  # 窗口索引偏移
        for grid_t, grid_h, grid_w in grid_thw:  # 遍历每个网格
            llm_grid_h, llm_grid_w = (  # 计算LLM网格大小
                grid_h // self.spatial_merge_size,  # 高度除以空间合并大小
                grid_w // self.spatial_merge_size,  # 宽度除以空间合并大小
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(  # 生成索引
                grid_t, llm_grid_h, llm_grid_w
            )
            if col:  # 如果是列方向
                index_new = index.transpose(1, 2).reshape(-1)  # 转置并展平
            else:  # 行方向
                index_new = index.reshape(-1)  # 直接展平
            window_index.append(index_new + window_index_id)  # 添加偏移后的索引
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()  # 更新偏移
        window_index = torch.cat(  # 拼接所有窗口索引
            window_index,
            dim=0,
        )
        return window_index  # 返回窗口索引

    @property
    def dtype(self) -> torch.dtype:
        """获取模型数据类型"""
        return self.patch_embed.proj.weight.dtype  # 返回补丁嵌入权重的数据类型

    @property
    def device(self) -> torch.device:
        """获取模型设备"""
        return self.blocks[0].mlp.gate_up_proj.weight.device  # 返回第一个块MLP的设备

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """计算旋转位置嵌入"""
        pos_ids = []  # 位置ID列表
        for i in range(grid_thw.size(0)):  # 遍历每个网格
            t, h, w = grid_thw[i].tolist()  # 获取时间、高度、宽度
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)  # 高度位置ID

            hpos_ids = hpos_ids.reshape(  # 重塑高度位置ID
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)  # 重排维度
            hpos_ids = hpos_ids.flatten()  # 展平

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)  # 宽度位置ID
            wpos_ids = wpos_ids.reshape(  # 重塑宽度位置ID
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)  # 重排维度
            wpos_ids = wpos_ids.flatten()  # 展平

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))  # 拼接并重复时间维度
        pos_ids = torch.cat(pos_ids, dim=0)  # 拼接所有位置ID
        max_grid_size = grid_thw[:, 1:].max()  # 最大网格尺寸
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)  # 计算完整旋转位置嵌入
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)  # 按位置ID索引并展平
        return rotary_pos_emb  # 返回旋转位置嵌入

    def _prepare_forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ):
        """准备前向传播：计算补丁嵌入、位置编码和注意力参数"""
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)  # 转换设备和数据类型
        x = self.patch_embed(x)  # 补丁嵌入
        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)  # 计算旋转位置嵌入

        window_index_1d_col = self.get_window_index_1d(grid_thw, col=True).to(  # 获取列方向窗口索引
            device=x.device
        )
        reverse_window_index_1d_col = torch.argsort(window_index_1d_col).to(  # 获取逆索引
            device=x.device
        )

        rotary_pos_emb = rotary_pos_emb.to(device=x.device)  # 转换位置嵌入设备
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # 拼接位置嵌入

        def get_position_embeddings(emb, x):
            """获取位置嵌入的余弦和正弦"""
            position_embeddings = (emb.cos(), emb.sin())  # 计算余弦和正弦
            position_embeddings = (  # 转换设备
                position_embeddings[0].to(x.device),
                position_embeddings[1].to(x.device),
            )
            return position_embeddings  # 返回位置嵌入

        seqlens = torch.repeat_interleave(  # 计算每个序列的长度
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu_seqlens = torch.cat(  # 计算累计序列长度
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),  # 起始为0
                seqlens.cumsum(dim=0).to(device=x.device, dtype=torch.int32),  # 累计求和
            ]
        )
        max_seqlen = seqlens.max().item()  # 最大序列长度

        row_based_embeddings = get_position_embeddings(emb, x)  # 行方向位置嵌入
        col_based_embeddings = get_position_embeddings(  # 列方向位置嵌入
            self.apply_index(emb, window_index_1d_col), x
        )

        # transformers
        x = x.unsqueeze(1)  # [S, 1, H] 增加批次维度

        return (  # 返回准备好的所有参数
            x,
            row_based_embeddings,  # 行方向位置嵌入
            col_based_embeddings,  # 列方向位置嵌入
            window_index_1d_col,  # 列方向窗口索引
            reverse_window_index_1d_col,  # 逆窗口索引
            cu_seqlens,  # 累计序列长度
            max_seqlen,  # 最大序列长度
        )

    def run_blocks(
        self,
        x: torch.Tensor,
        row_based_embeddings: Tuple[torch.Tensor, torch.Tensor],
        col_based_embeddings: Tuple[torch.Tensor, torch.Tensor],
        window_index_1d_col: torch.Tensor,
        reverse_window_index_1d_col: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """运行Transformer块，支持行列双方向窗口注意力"""
        for layer_num, blk in enumerate(self.blocks):  # 遍历每个块
            window_attn_type = self.vit_window_attn_types[layer_num]  # 获取窗口注意力类型

            # window_attn_type = 1: col-based SWA
            if window_attn_type == 1 and (  # 列方向窗口注意力的进入
                layer_num == 0 or self.vit_window_attn_types[layer_num - 1] != 1
            ):
                x = self.apply_index(x, window_index_1d_col)  # 重排为列方向

            if (  # 列方向窗口注意力的退出
                layer_num > 0
                and window_attn_type != 1
                and self.vit_window_attn_types[layer_num - 1] == 1
            ):
                x = self.apply_index(x, reverse_window_index_1d_col)  # 恢复为行方向

            position_embeddings = (  # 选择位置嵌入
                col_based_embeddings if window_attn_type == 1 else row_based_embeddings
            )
            full_attn = layer_num in self.fullatt_block_indexes  # 是否全注意力

            x = blk(  # 通过视觉块
                x,
                cu_seqlens=cu_seqlens,  # 累计序列长度
                max_seqlen=max_seqlen,  # 最大序列长度
                position_embeddings=position_embeddings,  # 位置嵌入
                full_attn=full_attn,  # 是否全注意力
            )
        x = self.merger(x)  # 通过补丁合并器
        return x  # 返回输出

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """视觉Transformer前向传播"""
        (  # 准备前向传播参数
            x,
            row_based_embeddings,  # 行方向位置嵌入
            col_based_embeddings,  # 列方向位置嵌入
            window_index_1d_col,  # 列方向窗口索引
            reverse_window_index_1d_col,  # 逆窗口索引
            cu_seqlens,  # 累计序列长度
            max_seqlen,  # 最大序列长度
        ) = self._prepare_forward(x, grid_thw)  # 调用准备函数

        return self.run_blocks(  # 运行Transformer块
            x,
            row_based_embeddings,  # 行方向位置嵌入
            col_based_embeddings,  # 列方向位置嵌入
            window_index_1d_col,  # 列方向窗口索引
            reverse_window_index_1d_col,  # 逆窗口索引
            cu_seqlens,  # 累计序列长度
            max_seqlen,  # 最大序列长度
        )
