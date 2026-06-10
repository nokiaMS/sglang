# DeepSeek V4 融合多头部选择(MHC)算子模块
# 提供基于Triton的融合mHC后处理与前处理内核，用于AMD ROCm平台上的DeepSeek模型推理加速
# 主要功能包括：延迟加载aiter的mhc_post_pre算子、管理融合缓冲区缓存、以及尝试执行融合mHC内核

import logging  # 导入日志模块
from typing import Optional, Tuple  # 导入类型提示工具

import torch  # 导入PyTorch深度学习框架
import triton  # 导入Triton GPU编程框架

from sglang.srt.environ import envs  # 从sglang运行时环境模块导入环境变量配置

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

_FUSED_HC_POST_PRE_M_THRESHOLD = 64  # 融合HC后处理/前处理的最大token数阈值，超过此值则不使用融合路径
_FUSED_HC_POST_PRE_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}  # 融合HC后处理/前处理的缓冲区缓存字典，键为配置元组，值为张量字典
_TRITON_MHC_POST_PRE_OPS = None  # 延迟加载的Triton MHC后处理/前处理算子元组，初始为None
_TRITON_MHC_POST_PRE_RUNTIME_DISABLED = False  # 运行时禁用标志，当融合内核执行失败时设为True


def _get_triton_mhc_post_pre_ops():  # 延迟加载并返回Triton MHC后处理/前处理算子
    global _TRITON_MHC_POST_PRE_OPS  # 声明使用全局变量

    if _TRITON_MHC_POST_PRE_OPS is not None:  # 如果算子已加载
        return _TRITON_MHC_POST_PRE_OPS  # 直接返回缓存的算子

    try:  # 尝试从aiter库导入MHC算子
        from aiter.ops.triton.fusions.mhc import mhc_post_pre  # 导入mhc_post_pre融合算子
        from aiter.ops.triton.utils.mhc_config_utils import get_mhc_config  # 导入MHC配置获取工具
    except Exception as err:  # 导入失败时捕获异常
        logger.warning(  # 记录警告日志
            "Triton fused mHC (mhc_post_pre) is unavailable, falling back: %s", err  # Triton融合mHC不可用，回退到普通路径
        )
        return None  # 返回None表示不可用

    _TRITON_MHC_POST_PRE_OPS = (mhc_post_pre, get_mhc_config)  # 缓存算子元组
    return _TRITON_MHC_POST_PRE_OPS  # 返回算子元组


def _get_fused_hc_post_pre_buffers(  # 获取或创建融合HC后处理/前处理所需的缓冲区
    num_tokens: int,  # token数量
    hidden_size: int,  # 隐藏层维度大小
    hc_mult: int,  # HC乘数（多头选择的倍数）
    dtype: torch.dtype,  # 数据类型
    device: torch.device,  # 设备类型
) -> Optional[dict[str, torch.Tensor]]:  # 返回缓冲区字典，失败时返回None
    ops = _get_triton_mhc_post_pre_ops()  # 获取Triton MHC算子
    if ops is None:  # 如果算子不可用
        return None  # 返回None
    _, get_mhc_config = ops  # 解构获取配置函数

    key = (num_tokens, hidden_size, hc_mult, dtype, device.type, device.index)  # 构建缓存键
    bufs = _FUSED_HC_POST_PRE_CACHE.get(key)  # 从缓存中查找缓冲区
    if bufs is not None:  # 如果缓存命中
        return bufs  # 返回缓存的缓冲区

    try:  # 尝试获取MHC配置
        cfg, _ = get_mhc_config("MHC_FUSED", num_tokens, hidden_size, mode="sinkhorn")  # 获取融合MHC的配置参数
    except Exception as err:  # 配置获取失败时捕获异常
        logger.warning("Failed to initialize fused mHC config, falling back: %s", err)  # 记录警告：融合mHC配置初始化失败
        return None  # 返回None

    n_total = 2 * hc_mult + hc_mult * hc_mult  # 计算总维度：h_post维度(2*hc_mult) + h_res维度(hc_mult*hc_mult)
    k_dim = hc_mult * hidden_size  # 计算K维度大小
    block_k = cfg.get("BLOCK_K", min(512, triton.next_power_of_2(k_dim)))  # 从配置获取BLOCK_K参数，默认取512和k_dim最近的2的幂中的较小值
    block_k = min(block_k, triton.next_power_of_2(k_dim))  # 确保block_k不超过k_dim的最近2的幂
    block_c_split = max(block_k // hc_mult, 1)  # 计算每个C分块的块大小
    num_ksplit = triton.cdiv(hidden_size, block_c_split)  # 计算K维度的分块数量

    bufs = {  # 创建缓冲区字典
        "residual_out": torch.empty(  # 残差输出缓冲区
            num_tokens, hc_mult, hidden_size, dtype=dtype, device=device  # 形状为[num_tokens, hc_mult, hidden_size]
        ),
        "layer_input_out": torch.empty(  # 层输入输出缓冲区
            num_tokens, hidden_size, dtype=dtype, device=device  # 形状为[num_tokens, hidden_size]
        ),
        "h_post": torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device),  # 后处理h值缓冲区，float32精度
        "h_res": torch.empty(  # 残差h值缓冲区
            num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=device  # 形状为[num_tokens, hc_mult, hc_mult]，float32精度
        ),
        "acc_partial": torch.empty(  # 部分累加缓冲区
            num_ksplit, num_tokens, n_total, dtype=torch.float32, device=device  # 形状为[num_ksplit, num_tokens, n_total]，float32精度
        ),
        "acc_sq_partial": torch.empty(  # 部分平方累加缓冲区
            num_ksplit, num_tokens, dtype=torch.float32, device=device  # 形状为[num_ksplit, num_tokens]，float32精度
        ),
    }
    _FUSED_HC_POST_PRE_CACHE[key] = bufs  # 将缓冲区存入缓存
    return bufs  # 返回缓冲区字典


def try_fused_hc_post_pre(  # 尝试执行融合HC后处理与前处理操作
    x: torch.Tensor,  # 输入张量x
    residual: torch.Tensor,  # 残差张量
    post: torch.Tensor,  # 后处理权重张量
    comb: torch.Tensor,  # 组合权重张量
    hc_fn_t: torch.Tensor,  # HC函数转置权重
    hc_scale: torch.Tensor,  # HC缩放因子
    hc_base: torch.Tensor,  # HC基准值
    hc_mult: int,  # HC乘数
    norm_eps: float,  # 归一化epsilon值
    hc_eps: float,  # HC的epsilon值
    hc_post_mult: float,  # HC后处理乘数
    sinkhorn_iters: int,  # Sinkhorn迭代次数
    is_gfx95_supported: bool,  # 是否支持gfx95架构
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]:  # 返回结果元组或None
    global _TRITON_MHC_POST_PRE_RUNTIME_DISABLED  # 声明使用全局禁用标志

    if (  # 检查是否满足融合路径的前提条件
        _TRITON_MHC_POST_PRE_RUNTIME_DISABLED  # 运行时已被禁用
        or not envs.SGLANG_OPT_USE_TRITON_FUSED_MHC.get()  # 环境变量未启用融合MHC
        or not is_gfx95_supported  # 不支持gfx95架构
        or x.shape[0] == 0  # 输入为空
        or x.shape[0] > _FUSED_HC_POST_PRE_M_THRESHOLD  # token数超过阈值
        or x.dim() != 2  # x不是2维张量
        or residual.dim() != 3  # residual不是3维张量
    ):
        return None  # 不满足条件，返回None回退到普通路径

    ops = _get_triton_mhc_post_pre_ops()  # 获取Triton MHC算子
    if ops is None:  # 如果算子不可用
        return None  # 返回None
    mhc_post_pre, _ = ops  # 解构获取mhc_post_pre算子

    bufs = _get_fused_hc_post_pre_buffers(  # 获取融合操作所需的缓冲区
        x.shape[0], x.shape[1], hc_mult, residual.dtype, x.device  # 传入token数、隐藏维度、HC乘数、数据类型和设备
    )
    if bufs is None:  # 如果缓冲区获取失败
        return None  # 返回None

    try:  # 尝试执行融合MHC内核
        _, _, layer_input_out, new_residual = mhc_post_pre(  # 调用融合mhc_post_pre算子
            x,  # 输入张量
            residual,  # 残差张量
            post,  # 后处理权重
            comb,  # 组合权重
            hc_fn_t,  # HC函数转置权重
            hc_scale,  # HC缩放因子
            hc_base,  # HC基准值
            hc_mult,  # HC乘数
            norm_eps,  # 归一化epsilon
            hc_eps,  # HC的epsilon
            hc_post_mult,  # HC后处理乘数
            sinkhorn_iters,  # Sinkhorn迭代次数
            # Match sglang's exp-domain asymmetric Sinkhorn used in hc_pre.  # 匹配sglang在hc_pre中使用的指数域非对称Sinkhorn
            asymmetric_exp_domain=True,  # 启用非对称指数域Sinkhorn
            hc_sinkhorn_eps=hc_eps,  # HC Sinkhorn的epsilon值
            residual_out=bufs["residual_out"],  # 残差输出缓冲区
            h_post=bufs["h_post"],  # 后处理h值缓冲区
            h_res=bufs["h_res"],  # 残差h值缓冲区
            layer_input_out=bufs["layer_input_out"],  # 层输入输出缓冲区
            acc_partial=bufs["acc_partial"],  # 部分累加缓冲区
            acc_sq_partial=bufs["acc_sq_partial"],  # 部分平方累加缓冲区
        )
    except Exception as err:  # 融合内核执行失败时捕获异常
        logger.warning(  # 记录警告日志
            "Triton fused mHC kernel failed, disabling fallback path: %s", err  # Triton融合mHC内核失败，禁用回退路径
        )
        _TRITON_MHC_POST_PRE_RUNTIME_DISABLED = True  # 设置运行时禁用标志，后续不再尝试融合路径
        return None  # 返回None

    return new_residual, layer_input_out, bufs["h_post"], bufs["h_res"], False  # 返回结果元组：新残差、层输入输出、h_post、h_res、以及是否需要重新计算的标志(False表示不需要)
