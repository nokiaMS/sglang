# 视觉注意力层工具函数模块
# 提供视觉注意力层的虚拟头(dummy heads)配置更新和权重填充功能，
# 用于确保注意力头数可被张量并行大小整除

"""Utility functions for vision attention layers.
视觉注意力层的工具函数。"""

import torch  # PyTorch核心库

from sglang.srt.layers.dp_attention import get_attention_tp_size  # 获取张量并行大小


def update_vit_attn_dummy_heads_config(config):  # 更新视觉注意力虚拟头配置，确保注意力头数可被TP大小整除
    """Update HF config to ensure vision attention num_attention_heads is divisible by tp_size
    更新HuggingFace配置，确保视觉注意力头数可被张量并行大小整除"""
    tp_size = get_attention_tp_size()  # 获取张量并行大小
    num_heads = getattr(  # 获取注意力头数
        config.vision_config,  # 视觉配置
        "num_heads",  # 优先查找num_heads
        getattr(config.vision_config, "num_attention_heads", None),  # 回退查找num_attention_heads
    )
    head_dim = config.vision_config.hidden_size // num_heads  # 计算头维度
    num_dummy_heads = 0  # 初始化虚拟头数为0

    if num_heads % tp_size != 0:  # 如果头数不能被TP大小整除
        num_dummy_heads = ((num_heads + tp_size - 1) // tp_size) * tp_size - num_heads  # 计算需要添加的虚拟头数

    setattr(config.vision_config, "head_dim", head_dim)  # 设置头维度
    setattr(config.vision_config, "num_dummy_heads", num_dummy_heads)  # 设置虚拟头数


def pad_vit_attn_dummy_heads(config, name: str, loaded_weight: torch.Tensor):  # 为虚拟头填充注意力QKV权重
    """Pad attention qkv weights for dummy heads
    为虚拟头填充注意力QKV权重"""
    num_dummy_heads = config.vision_config.num_dummy_heads  # 获取虚拟头数
    if num_dummy_heads == 0:  # 如果没有虚拟头
        return loaded_weight  # 直接返回原始权重
    head_dim = config.vision_config.head_dim  # 获取头维度

    if "attn.qkv_proj" in name:  # 如果是QKV合并投影权重
        wq, wk, wv = loaded_weight.chunk(3, dim=0)  # 沿第0维拆分为Q、K、V权重
        if name.endswith(".weight"):  # 如果是权重
            dummy_shape = [num_dummy_heads, head_dim, wq.shape[-1]]  # 虚拟头权重的形状
        elif name.endswith(".bias"):  # 如果是偏置
            dummy_shape = [num_dummy_heads, head_dim]  # 虚拟头偏置的形状
        else:  # 其他情况
            raise RuntimeError(f"Unsupported weight with name={name}")  # 不支持的权重名
        pad_func = lambda x: torch.cat(  # 填充函数
            [x.unflatten(0, (-1, head_dim)), x.new_zeros(dummy_shape)], dim=0  # 原始权重+零填充
        ).flatten(0, 1)  # 重新展平
        wq, wk, wv = pad_func(wq), pad_func(wk), pad_func(wv)  # 分别填充Q、K、V
        loaded_weight = torch.cat([wq, wk, wv], dim=0)  # 合并填充后的QKV
    elif any([_ in name for _ in ["attn.q_proj", "attn.k_proj", "attn.v_proj"]]):  # 如果是分离的Q/K/V投影
        if name.endswith(".weight"):  # 如果是权重
            dummy_shape = [num_dummy_heads, head_dim, loaded_weight.shape[-1]]  # 虚拟头权重形状
        elif name.endswith(".bias"):  # 如果是偏置
            dummy_shape = [num_dummy_heads, head_dim]  # 虚拟头偏置形状
        else:  # 其他情况
            raise RuntimeError(f"Unsupported weight with name={name}")  # 不支持的权重名
        padded_weight = loaded_weight.new_zeros(dummy_shape)  # 创建零填充权重
        loaded_weight = torch.cat(  # 拼接原始权重和填充
            [loaded_weight.unflatten(0, (-1, head_dim)), padded_weight], dim=0  # 原始+零填充
        ).flatten(0, 1)  # 重新展平
    elif "attn.proj.weight" in name:  # 如果是输出投影权重
        padded_weight = loaded_weight.new_zeros(  # 创建零填充权重
            loaded_weight.shape[0], head_dim * num_dummy_heads  # 输出行数 x 虚拟头维度
        )
        loaded_weight = torch.cat([loaded_weight, padded_weight], dim=-1)  # 在最后一维拼接
    elif "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:  # 如果是Q/K归一化权重
        padded_weight = loaded_weight.new_zeros(head_dim * num_dummy_heads)  # 创建零填充
        loaded_weight = torch.cat([loaded_weight, padded_weight], dim=0)  # 在第0维拼接
    return loaded_weight  # 返回填充后的权重
