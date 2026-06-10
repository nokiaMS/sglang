# 注意力前向方法枚举定义
# 本文件定义了 DeepSeek 模型中所有支持的注意力前向计算方法，
# 包括 MHA（多头注意力）、MLA（多潜在注意力）及其各种变体。

from enum import IntEnum, auto()


# 注意力前向方法枚举类
# 使用 IntEnum 以支持高效比较和序列化
class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    # 使用标准多头注意力
    MHA = auto()

    # Use absorbed multi-latent attention
    # 使用吸收式多潜在注意力（将 kv_b_proj 权重吸收到 Q 中）
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    # 使用多头注意力但 KV 缓存分块处理，避免长前缀时的 OOM
    MHA_CHUNKED_KV = auto()

    # Use multi-head attention, execute the MHA for prefix and extended kv in a single kernel
    # when the sequence lengths are below the threshold.
    # 使用单次核函数执行 MHA（前缀和扩展 KV 一起计算），仅在序列长度低于阈值时使用
    MHA_ONE_SHOT = auto()

    # Use MLA but with fused RoPE
    # 使用 MLA 并融合 RoPE 操作（ROCm 平台专用）
    MLA_FUSED_ROPE_ROCM = auto()

    # Use MLA with fused RoPE kernel for CPU
    # 使用 MLA 并融合 CPU 上的 RoPE 操作（Intel AMX 平台专用）
    MLA_FUSED_ROPE_CPU = auto()

    # Use multi-head attention for NPU
    # NPU（华为昇腾）平台上的多头注意力
    MHA_NPU = auto()

    # Use absorbed multi-latent attention for NPU
    # NPU（华为昇腾）平台上的吸收式多潜在注意力
    MLA_NPU = auto()

    # Use Deepseek V3.2 sparse multi-latent attention for NPU
    # NPU（华为昇腾）平台上的 DeepSeek V3.2 稀疏多潜在注意力（DSA）
    DSA_NPU = auto()
