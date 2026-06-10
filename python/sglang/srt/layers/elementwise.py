# 逐元素操作的 Triton 融合内核实现，包括 Softcap、RMSNorm、
# 双残差 RMSNorm、专家组合、GeLU 和 SiLU 激活函数。

from typing import Optional, Tuple  # 导入类型注解

import torch  # 导入 PyTorch
import triton  # 导入 Triton
import triton.language as tl  # 导入 Triton 语言

from sglang.srt.utils import is_hip  # 导入平台检测
from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册

_is_hip = is_hip()  # 检测是否为 AMD ROCm 平台


fused_softcap_autotune = triton.autotune(  # Softcap 内核的自动调优配置
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=4),  # 块大小 128，4 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=8),  # 块大小 128，8 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=16),  # 块大小 128，16 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=4),  # 块大小 256，4 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=8),  # 块大小 256，8 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=4),  # 块大小 512，4 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=8),  # 块大小 512，8 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=16),  # 块大小 512，16 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4),  # 块大小 1024，4 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8),  # 块大小 1024，8 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16),  # 块大小 1024，16 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=32),  # 块大小 1024，32 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=32),  # 块大小 2048，32 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=32),  # 块大小 4096，32 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32),  # 块大小 8192，32 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32),  # 块大小 16384，32 个 warp
        triton.Config(kwargs={"BLOCK_SIZE": 32768}, num_warps=32),  # 块大小 32768，32 个 warp
    ],
    key=["n_ele"],  # 按元素数调优
)


@triton.jit  # Triton JIT 编译的 Softcap 内核
def fused_softcap_kernel(  # 融合 Softcap 内核
    output_ptr,  # 输出指针
    input_ptr,  # 输入指针
    n_ele,  # 元素总数
    softcap_const: tl.constexpr,  # Softcap 常数
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    pid = tl.program_id(axis=0)  # 获取程序 ID
    block_start = pid * BLOCK_SIZE  # 计算块起始位置
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 计算偏移
    mask = offsets < n_ele  # 生成掩码
    x = tl.load(input_ptr + offsets, mask=mask)  # 加载输入
    fx = x.to(tl.float32)  # 转为 float32
    fxs = fx / softcap_const  # 除以 softcap 常数
    exped = tl.exp(2 * fxs)  # 计算 exp(2 * x / softcap)
    top = exped - 1  # 分子：exp - 1
    bottom = exped + 1  # 分母：exp + 1
    output = top / bottom * softcap_const  # tanh(x/softcap) * softcap
    tl.store(output_ptr + offsets, output, mask=mask)  # 存储输出


fused_softcap_kernel_autotuned = fused_softcap_autotune(fused_softcap_kernel)  # 应用自动调优的内核


def fused_softcap(x, softcap_const, autotune=False):  # 融合 Softcap 操作
    output = torch.empty_like(x, dtype=torch.float32)  # 分配输出缓冲区
    n_elements = output.numel()  # 获取元素总数
    if autotune:  # 如果启用自动调优
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # 动态网格
        fused_softcap_kernel_autotuned[grid](output, x, n_elements, softcap_const)  # 调用自动调优内核
    else:  # 使用固定配置
        fused_softcap_kernel[(triton.cdiv(n_elements, 128),)](  # 固定块大小 128
            output, x, n_elements, softcap_const, BLOCK_SIZE=128, num_warps=8
        )
    return output  # 返回输出


# cast to float + softcap
# 转为浮点 + softcap
class Softcap:  # Softcap 操作类
    def __init__(self, softcap_const: float):  # 初始化，设置 softcap 常数
        self.softcap_const = softcap_const

    def __call__(self, *args, **kwargs):  # 可调用接口
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播
        if x.is_cuda:  # 如果在 CUDA 上
            return self.forward_cuda(x)  # 使用 CUDA 实现
        else:
            return self.forward_native(x)  # 使用原生实现

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:  # 原生 Softcap 实现
        return torch.tanh(x.float() / self.softcap_const) * self.softcap_const  # tanh(x/cap) * cap

    def forward_cuda(self, x: torch.Tensor, autotune=False) -> torch.Tensor:  # CUDA Softcap 实现
        return fused_softcap(x, self.softcap_const, autotune=autotune)


rmsnorm_autotune = triton.autotune(  # RMSNorm 内核的自动调优配置
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),  # 1024 块，4 warp，1 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),  # 1024 块，8 warp，1 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16, num_stages=1),  # 1024 块，16 warp，1 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4),  # 1024 块，4 warp
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8),  # 1024 块，8 warp
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16),  # 1024 块，16 warp
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),  # 1024 块，4 warp，4 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),  # 1024 块，8 warp，4 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16, num_stages=4),  # 1024 块，16 warp，4 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=8),  # 1024 块，8 warp，8 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16, num_stages=8),  # 1024 块，16 warp，8 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=8),  # 2048 块，8 warp
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=16),  # 2048 块，16 warp
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),  # 2048 块，8 warp，4 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=16, num_stages=4),  # 2048 块，16 warp，4 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=8),  # 4096 块，8 warp
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=16),  # 4096 块，16 warp
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=8),  # 8192 块，8 warp
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=16),  # 8192 块，16 warp
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32),  # 8192 块，32 warp
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=8, num_stages=1),  # 8192 块，8 warp，1 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=16, num_stages=1),  # 8192 块，16 warp，1 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32, num_stages=1),  # 8192 块，32 warp，1 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),  # 8192 块，8 warp，4 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=16, num_stages=4),  # 8192 块，16 warp，4 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32, num_stages=4),  # 8192 块，32 warp，4 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=8),  # 16384 块，8 warp
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=16),  # 16384 块，16 warp
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32),  # 16384 块，32 warp
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=8, num_stages=1),  # 16384 块，8 warp，1 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=16, num_stages=1),  # 16384 块，16 warp，1 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32, num_stages=1),  # 16384 块，32 warp，1 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=8, num_stages=4),  # 16384 块，8 warp，4 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=16, num_stages=4),  # 16384 块，16 warp，4 阶段
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32, num_stages=4),  # 16384 块，32 warp，4 阶段
    ],
    key=["hidden_dim"],  # 按隐藏维度调优
)


@triton.jit  # Triton JIT 编译的双残差 RMSNorm 内核
def fused_dual_residual_rmsnorm_kernel(  # 融合双残差 RMSNorm 内核
    output_ptr,  # 输出指针
    mid_ptr,  # 中间结果指针（第一次 RMSNorm + 残差）
    activ_ptr,  # 激活输入指针
    residual_ptr,  # 残差输入指针
    weight1_ptr,  # 第一个 RMSNorm 权重指针
    weight2_ptr,  # 第二个 RMSNorm 权重指针
    eps: tl.constexpr,  # RMSNorm epsilon
    hidden_dim: tl.constexpr,  # 隐藏维度
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    pid = tl.program_id(axis=0)  # 获取程序 ID
    input_start = pid * hidden_dim  # 计算行起始偏移

    offsets = tl.arange(0, BLOCK_SIZE)  # 生成偏移
    mask = offsets < hidden_dim  # 生成掩码

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)  # 加载激活输入
    a = a_.to(tl.float32)  # 转为 float32
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)  # 计算 RMS

    r = tl.load(residual_ptr + input_start + offsets, mask=mask, other=0.0)  # 加载残差
    w1_ = tl.load(weight1_ptr + offsets, mask=mask, other=0.0)  # 加载第一个权重
    w1 = w1_.to(tl.float32)  # 转为 float32

    a2r = r + (a / rms * w1).to(r.dtype)  # 第一次 RMSNorm + 残差连接
    tl.store(  # 存储中间结果
        mid_ptr + input_start + offsets,
        a2r,
        mask=mask,
    )

    a2r = a2r.to(tl.float32)  # 转为 float32
    rms2 = tl.sqrt(tl.sum(a2r * a2r, axis=0) / hidden_dim + eps)  # 计算第二次 RMS

    w2_ = tl.load(weight2_ptr + offsets, mask=mask, other=0.0)  # 加载第二个权重
    w2 = w2_.to(tl.float32)  # 转为 float32

    tl.store(  # 存储最终输出
        output_ptr + input_start + offsets,
        a2r / rms2 * w2,  # implicitly casts to output dtype here # 隐式转换为输出类型
        mask=mask,
    )


fused_dual_residual_rmsnorm_kernel_autotune = rmsnorm_autotune(  # 应用自动调优
    fused_dual_residual_rmsnorm_kernel
)


def fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps, autotune=False):  # 融合双残差 RMSNorm
    assert len(x.shape) == 2  # 断言输入为 2D
    assert (
        x.shape == residual.shape and x.dtype == residual.dtype
    ), f"{x.shape=} {residual.shape=} {x.dtype=} {residual.dtype=}"  # 断言形状和类型一致
    output, mid = torch.empty_like(x), torch.empty_like(x)  # 分配输出和中间缓冲区
    bs, hidden_dim = x.shape  # 获取批次大小和隐藏维度
    if autotune:  # 如果启用自动调优
        fused_dual_residual_rmsnorm_kernel_autotune[(bs,)](  # 调用自动调优内核
            output, mid, x, residual, weight1, weight2, eps=eps, hidden_dim=hidden_dim
        )
    else:  # 使用固定配置
        max_warps = 16 if _is_hip else 32  # HIP 平台最多 16 warp，否则 32
        config = {
            "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),  # 块大小为隐藏维度的 2 的幂
            "num_warps": max(
                min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
            ),  # 计算 warp 数
        }

        fused_dual_residual_rmsnorm_kernel[(bs,)](  # 调用内核
            output,
            mid,
            x,
            residual,
            weight1,
            weight2,
            eps=eps,
            hidden_dim=hidden_dim,
            **config,
        )

    return output, mid  # 返回输出和中间结果


@triton.jit  # Triton JIT 编译的 RMSNorm 内核
def fused_rmsnorm_kernel(  # 融合 RMSNorm 内核
    output_ptr,  # 输出指针
    activ_ptr,  # 激活输入指针
    weight_ptr,  # 权重指针
    eps: tl.constexpr,  # RMSNorm epsilon
    hidden_dim: tl.constexpr,  # 隐藏维度
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    pid = tl.program_id(axis=0).to(tl.int64)  # 获取程序 ID
    input_start = pid * hidden_dim  # 计算行起始偏移

    offsets = tl.arange(0, BLOCK_SIZE)  # 生成偏移
    mask = offsets < hidden_dim  # 生成掩码

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)  # 加载激活输入
    a = a_.to(tl.float32)  # 转为 float32
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)  # 计算 RMS

    w1_ = tl.load(weight_ptr + offsets, mask=mask, other=0.0)  # 加载权重
    w1 = w1_.to(tl.float32)  # 转为 float32

    a_rms = a / rms * w1  # 计算 RMSNorm 结果

    tl.store(  # 存储输出
        output_ptr + input_start + offsets,
        a_rms,  # implicitly casts to output dtype here # 隐式转换为输出类型
        mask=mask,
    )


def fused_rmsnorm(x, weight, eps, autotune=False, inplace=False):  # 融合 RMSNorm
    assert len(x.shape) == 2  # 断言输入为 2D
    if inplace:  # 如果原地操作
        output = x  # 直接使用输入作为输出
    else:
        output = torch.empty_like(x)  # 分配新输出缓冲区
    bs, hidden_dim = x.shape  # 获取批次大小和隐藏维度
    max_warps = 16 if _is_hip else 32  # HIP 平台最多 16 warp，否则 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),  # 块大小为隐藏维度的 2 的幂
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),  # 计算 warp 数
    }

    fused_rmsnorm_kernel[(bs,)](  # 调用内核
        output, x, weight, eps=eps, hidden_dim=hidden_dim, **config
    )
    return output  # 返回输出


class FusedDualResidualRMSNorm:  # 融合双残差 RMSNorm 类
    """
    Fused implementation of
    y = RMSNorm2(RMSNorm1(x) + residual))
    """  # 融合实现：y = RMSNorm2(RMSNorm1(x) + residual)

    def __init__(self, rmsnorm1, rmsnorm2) -> None:  # the one after rmsnorm1 # 初始化，rmsnorm2 是 rmsnorm1 之后的
        self.rmsnorm1 = rmsnorm1  # 第一个 RMSNorm
        self.rmsnorm2 = rmsnorm2  # 第二个 RMSNorm
        self.variance_epsilon = self.rmsnorm1.variance_epsilon  # 方差 epsilon
        assert self.rmsnorm1.variance_epsilon == self.rmsnorm2.variance_epsilon  # 断言 epsilon 一致
        assert self.rmsnorm1.weight.shape == self.rmsnorm2.weight.shape  # 断言权重形状一致

    def __call__(self, *args, **kwargs):  # 可调用接口
        return self.forward(*args, **kwargs)

    def forward(  # 前向传播
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.is_cuda:  # 如果在 CUDA 上
            return self.forward_cuda(x, residual)  # 使用 CUDA 实现
        else:
            return self.forward_flashinfer(x, residual)  # 使用 flashinfer 实现

    def forward_cuda(  # CUDA 前向传播
        self, x: torch.Tensor, residual: torch.Tensor, autotune=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return fused_dual_residual_rmsnorm(
            x,
            residual,
            self.rmsnorm1.weight,
            self.rmsnorm2.weight,
            self.variance_epsilon,
            autotune=autotune,
        )

    def forward_flashinfer(  # flashinfer 前向传播（非 CUDA 回退）
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed1 = self.rmsnorm1(x)  # 第一次 RMSNorm
        residual = normed1 + residual  # 残差连接
        return self.rmsnorm2(residual), residual  # 第二次 RMSNorm 并返回

    def forward_native(  # 原生前向传播
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed1 = self.rmsnorm1.forward_native(x)  # 第一次 RMSNorm（原生）
        residual = normed1 + residual  # 残差连接
        return self.rmsnorm2.forward_native(residual), residual  # 第二次 RMSNorm（原生）并返回


@triton.jit  # Triton JIT 编译的专家组合内核
def experts_combine_kernel(  # 专家组合内核
    out_hidden_states,  # 输出隐藏状态指针
    moe_hidden_states,  # MoE 隐藏状态指针
    mlp_hidden_states,  # MLP 隐藏状态指针
    combine_k: tl.constexpr,  # 组合的 K 值
    hidden_dim: tl.constexpr,  # 隐藏维度
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    pid = tl.program_id(0)  # 获取程序 ID
    start_index_mlp = pid * hidden_dim  # MLP 起始索引
    start_index_rmoe = pid * hidden_dim * combine_k  # MoE 起始索引
    offsets = tl.arange(0, BLOCK_SIZE)  # 生成偏移
    mask = offsets < hidden_dim  # 生成掩码
    combine_k_offsets = tl.arange(0, combine_k)  # K 维度偏移

    moe_x = tl.load(  # 加载 MoE 隐藏状态
        moe_hidden_states
        + start_index_rmoe
        + combine_k_offsets[:, None] * hidden_dim
        + offsets[None, :],
        mask=mask[None, :],
        other=0.0,
    )
    moe_x = tl.sum(moe_x, axis=0)  # 对 K 维度求和
    mlp_x = tl.load(mlp_hidden_states + start_index_mlp + offsets, mask=mask, other=0.0)  # 加载 MLP 隐藏状态
    combined_x = (moe_x + mlp_x) / 1.4142135623730951  # 合并并除以 sqrt(2)

    tl.store(out_hidden_states + start_index_mlp + offsets, combined_x, mask=mask)  # 存储合并结果


@register_custom_op(out_shape="mlp_hidden_states")  # 注册为自定义算子，输出形状与 mlp_hidden_states 相同
def experts_combine_triton(  # 专家组合 Triton 实现
    moe_hidden_states: torch.Tensor,  # MoE 隐藏状态
    mlp_hidden_states: torch.Tensor,  # MLP 隐藏状态
    output_buffer: Optional[torch.Tensor] = None,  # 可选的输出缓冲区
) -> torch.Tensor:
    assert moe_hidden_states.is_contiguous()  # 断言 MoE 隐藏状态连续
    assert mlp_hidden_states.is_contiguous()  # 断言 MLP 隐藏状态连续

    if len(moe_hidden_states.shape) == 2:  # 如果 MoE 隐藏状态为 2D
        combine_k = 1  # pre-combined # 已预组合，K=1
    else:
        combine_k = moe_hidden_states.shape[1]  # 从第二维获取 K 值

    if output_buffer is None:  # 如果没有提供输出缓冲区
        out_hidden_states = torch.empty_like(mlp_hidden_states)  # 分配新输出
    else:
        flat_output_buffer = output_buffer.view(mlp_hidden_states.dtype).reshape(-1)  # 展平缓冲区
        assert flat_output_buffer.numel() >= mlp_hidden_states.numel()  # 断言缓冲区足够大
        out_hidden_states = flat_output_buffer[: mlp_hidden_states.numel()].reshape(  # 截取并重塑
            mlp_hidden_states.shape
        )

    bs, hidden_dim = mlp_hidden_states.shape  # 获取批次大小和隐藏维度

    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),  # 块大小为隐藏维度的 2 的幂
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 1024)), 8), 4
        ),  # 计算 warp 数
    }

    experts_combine_kernel[(bs,)](  # 调用内核
        out_hidden_states,
        moe_hidden_states,
        mlp_hidden_states,
        combine_k,
        hidden_dim,
        **config,
    )

    return out_hidden_states  # 返回输出


# gelu on first half of vector
# 对向量的前半部分应用 gelu
@triton.jit  # Triton JIT 编译的 GeLU 和乘法内核
def gelu_and_mul_kernel(  # GeLU 和乘法内核
    out_hidden_states_ptr,  # (bs, hidden_dim) # 输出隐藏状态指针
    out_scales_ptr,  # (bs,) # 输出缩放指针
    hidden_states_ptr,  # (bs, hidden_dim * 2) # 输入隐藏状态指针
    quant_max: tl.constexpr,  # 量化最大值
    static_scale: tl.constexpr,  # 静态缩放标志
    hidden_dim: tl.constexpr,  # the output hidden_dim # 输出隐藏维度
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    pid = tl.program_id(axis=0)  # 获取程序 ID

    input_start = pid * hidden_dim * 2  # 输入起始偏移（2 倍隐藏维度）
    output_start = pid * hidden_dim  # 输出起始偏移

    input1_offs = tl.arange(0, BLOCK_SIZE)  # 第一半偏移
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim  # shared for input1, input3, output # 共享掩码
    input3_offs = hidden_dim + tl.arange(0, BLOCK_SIZE)  # 第二半偏移
    output_offs = tl.arange(0, BLOCK_SIZE)  # 输出偏移

    x1 = tl.load(  # 加载第一半（用于 GeLU）
        hidden_states_ptr + input_start + input1_offs, mask=mask, other=0.0
    ).to(tl.float32)
    x3 = tl.load(  # 加载第二半（用于乘法）
        hidden_states_ptr + input_start + input3_offs, mask=mask, other=0.0
    ).to(tl.float32)

    # gelu
    # gelu 激活
    # cast down before mul to better match training?
    # 乘法前降低精度以更好地匹配训练？
    gelu_x1 = 0.5 * (1.0 + tl.erf(x1 * 0.7071067811865475)) * x1  # GeLU(x1) = 0.5*(1+erf(x/sqrt(2)))*x
    out = x3 * gelu_x1.to(hidden_states_ptr.dtype.element_ty)  # x3 * GeLU(x1)

    if quant_max is not None:  # 如果需要量化
        raise NotImplementedError()  # 量化功能未实现

    tl.store(out_hidden_states_ptr + output_start + output_offs, out, mask=mask)  # 存储输出


def gelu_and_mul_triton(  # GeLU 和乘法的 Triton 实现
    hidden_states,  # 输入隐藏状态
    scales=None,  # 可选缩放
    quantize=None,  # dtype to quantize to # 量化目标类型
    out=None,  # 可选输出缓冲区
):
    bs, in_hidden_dim = hidden_states.shape  # 获取批次大小和输入隐藏维度
    hidden_dim = in_hidden_dim // 2  # 输出隐藏维度为输入的一半

    if out is None:  # 如果没有提供输出
        out_hidden_states = torch.empty(  # 分配新输出
            (bs, hidden_dim),
            dtype=quantize or hidden_states.dtype,  # 使用量化类型或输入类型
            device=hidden_states.device,
        )
    else:
        assert out.shape == (bs, hidden_dim)  # 断言输出形状正确
        assert out.dtype == (quantize or hidden_states.dtype)  # 断言输出类型正确
        out_hidden_states = out
    out_scales = None  # 输出缩放初始化为 None
    static_scale = False  # 静态缩放标志初始化为 False
    if quantize is not None:  # 如果需要量化
        if scales is None:  # 如果没有提供缩放
            out_scales = torch.empty(  # 分配缩放缓冲区
                (bs,), dtype=torch.float32, device=hidden_states.device
            )
        else:
            out_scales = scales  # 使用提供的缩放
            static_scale = True  # 标记为静态缩放

    max_warps = 16 if _is_hip else 32  # HIP 平台最多 16 warp，否则 32
    config = {
        # 8 ele per thread (not tuned)
        # 每线程 8 个元素（未调优）
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
        ),  # 计算 warp 数
    }

    gelu_and_mul_kernel[(bs,)](  # 调用内核
        out_hidden_states,
        out_scales,
        hidden_states,
        quant_max=torch.finfo(quantize).max if quantize is not None else None,  # 量化最大值
        static_scale=static_scale,  # 静态缩放标志
        hidden_dim=hidden_dim,  # 输出隐藏维度
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),  # 块大小
        **config,
    )

    if quantize is not None:  # 如果量化
        return out_hidden_states, out_scales  # 返回输出和缩放
    else:
        return out_hidden_states, None  # 返回输出和 None 缩放


# silu on first half of vector
# 对向量的前半部分应用 silu
@triton.jit  # Triton JIT 编译的 SiLU 和乘法内核
def silu_and_mul_kernel(  # SiLU 和乘法内核
    out_hidden_states_ptr,  # (bs, hidden_dim) # 输出隐藏状态指针
    out_scales_ptr,  # (bs,) # 输出缩放指针
    hidden_states_ptr,  # (bs, hidden_dim * 2) # 输入隐藏状态指针
    quant_max: tl.constexpr,  # 量化最大值
    static_scale: tl.constexpr,  # 静态缩放标志
    hidden_dim: tl.constexpr,  # the output hidden_dim # 输出隐藏维度
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    pid = tl.program_id(axis=0)  # 获取程序 ID

    input_start = pid * hidden_dim * 2  # 输入起始偏移（2 倍隐藏维度）
    output_start = pid * hidden_dim  # 输出起始偏移

    input1_offs = tl.arange(0, BLOCK_SIZE)  # 第一半偏移
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim  # shared for input1, input3, output # 共享掩码
    input3_offs = hidden_dim + tl.arange(0, BLOCK_SIZE)  # 第二半偏移
    output_offs = tl.arange(0, BLOCK_SIZE)  # 输出偏移

    x1 = tl.load(  # 加载第一半（用于 SiLU）
        hidden_states_ptr + input_start + input1_offs, mask=mask, other=0.0
    ).to(tl.float32)
    x3 = tl.load(  # 加载第二半（用于乘法）
        hidden_states_ptr + input_start + input3_offs, mask=mask, other=0.0
    ).to(tl.float32)

    # silu
    # silu 激活
    # cast down before mul to better match training?
    # 乘法前降低精度以更好地匹配训练？
    silu_x1 = x1 * tl.sigmoid(x1)  # SiLU(x1) = x1 * sigmoid(x1)
    out = x3 * silu_x1.to(hidden_states_ptr.dtype.element_ty)  # x3 * SiLU(x1)

    if quant_max is not None:  # 如果需要量化
        raise NotImplementedError()  # 量化功能未实现

    tl.store(out_hidden_states_ptr + output_start + output_offs, out, mask=mask)  # 存储输出


def silu_and_mul_triton(  # SiLU 和乘法的 Triton 实现
    hidden_states,  # 输入隐藏状态
    scales=None,  # 可选缩放
    quantize=None,  # dtype to quantize to # 量化目标类型
    out=None,  # 可选输出缓冲区
):
    bs, in_hidden_dim = hidden_states.shape  # 获取批次大小和输入隐藏维度
    hidden_dim = in_hidden_dim // 2  # 输出隐藏维度为输入的一半

    if out is None:  # 如果没有提供输出
        out_hidden_states = torch.empty(  # 分配新输出
            (bs, hidden_dim),
            dtype=quantize or hidden_states.dtype,  # 使用量化类型或输入类型
            device=hidden_states.device,
        )
    else:
        assert out.shape == (bs, hidden_dim)  # 断言输出形状正确
        assert out.dtype == (quantize or hidden_states.dtype)  # 断言输出类型正确
        out_hidden_states = out
    out_scales = None  # 输出缩放初始化为 None
    static_scale = False  # 静态缩放标志初始化为 False
    if quantize is not None:  # 如果需要量化
        if scales is None:  # 如果没有提供缩放
            out_scales = torch.empty(  # 分配缩放缓冲区
                (bs,), dtype=torch.float32, device=hidden_states.device
            )
        else:
            out_scales = scales  # 使用提供的缩放
            static_scale = True  # 标记为静态缩放

    max_warps = 16 if _is_hip else 32  # HIP 平台最多 16 warp，否则 32
    config = {
        # 8 ele per thread (not tuned)
        # 每线程 8 个元素（未调优）
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
        ),  # 计算 warp 数
    }

    silu_and_mul_kernel[(bs,)](  # 调用内核
        out_hidden_states,
        out_scales,
        hidden_states,
        quant_max=torch.finfo(quantize).max if quantize is not None else None,  # 量化最大值
        static_scale=static_scale,  # 静态缩放标志
        hidden_dim=hidden_dim,  # 输出隐藏维度
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),  # 块大小
        **config,
    )

    if quantize is not None:  # 如果量化
        return out_hidden_states, out_scales  # 返回输出和缩放
    else:
        return out_hidden_states, None  # 返回输出和 None 缩放
