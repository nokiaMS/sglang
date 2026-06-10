# DFlash投机解码工具函数模块
# 提供DFlash草稿模型的配置解析、验证策略、KV缓存大小计算、
# 贪心和采样验证等核心工具函数。
from __future__ import annotations  # 启用延迟注解求值

from dataclasses import dataclass  # 导入数据类装饰器
from numbers import Integral  # 导入整数类型基类
from typing import Any, List, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch
import torch.nn.functional as F  # 导入PyTorch神经网络函数模块

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 导入未量化线性方法
from sglang.srt.managers.schedule_batch import Req  # 导入请求类
from sglang.srt.utils import is_cuda, is_musa  # 导入设备检测工具

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"  # DFlash默认掩码token

_DFLASH_SAMPLING_VERIFY_AVAILABLE = False  # 采样验证是否可用的标志
_DFLASH_CHAIN_VERIFY_BUFFERS: dict[tuple[Optional[int], int], dict[str, Any]] = {}  # 链式验证缓冲区缓存
_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS = frozenset(  # 跳过自定义掩码的注意力后端集合
    {
        "FlashInferAttnBackend",
        "FlashInferMLAAttnBackend",
        "FlashAttentionBackend",
        "TRTLLMHAAttnBackend",
        "TRTLLMMLABackend",
    }
)


if is_cuda() or is_musa():  # CUDA或MUSA设备
    try:
        from sgl_kernel import (  # 导入sgl_kernel中的采样验证内核
            top_k_renorm_prob,  # top-k重归一化概率
            top_p_renorm_prob,  # top-p重归一化概率
            tree_speculative_sampling_target_only,  # 树投机采样（仅目标）
        )

        _DFLASH_SAMPLING_VERIFY_AVAILABLE = True  # 标记采样验证可用
    except Exception:  # 导入失败
        top_k_renorm_prob = None  # 设为None
        top_p_renorm_prob = None  # 设为None
        tree_speculative_sampling_target_only = None  # 设为None
else:  # 非CUDA/MUSA设备
    top_k_renorm_prob = None  # 设为None
    top_p_renorm_prob = None  # 设为None
    tree_speculative_sampling_target_only = None  # 设为None


def is_dflash_sampling_verify_available() -> bool:
    # 检查DFlash采样验证是否可用
    return _DFLASH_SAMPLING_VERIFY_AVAILABLE  # 返回可用标志


def scale_kv_cell_size_per_token_for_dflash(
    *,
    target_cell_size_per_token: int,  # 目标模型每个token的KV单元格大小
    target_num_layers: int,  # 目标模型层数
    draft_num_layers: int,  # 草稿模型层数
    draft_cell_size_per_token: Optional[int] = None,  # 草稿模型每个token的KV单元格大小
) -> int:
    """Compute bytes/token budget for combined target+draft KV pools (DFLASH).
    # 计算DFLASH中目标+草稿KV池组合的每token字节预算。

    DFLASH runs a separate draft runner with its own KV pool. The target runner's
    token capacity must fit both pools in aggregate.
    # DFLASH运行一个有独立KV池的草稿运行器。目标运行器的token容量必须能容纳两个池的总和。

    Returns:
    # 返回值：
        Approximate per-token bytes for (target KV + draft KV), expressed as a
        scaled version of `target_cell_size_per_token`, unless an explicit
        `draft_cell_size_per_token` is provided (in which case we sum them).
        # 目标KV + 草稿KV的近似每token字节数，表示为target_cell_size_per_token的缩放版本，
        # 除非提供了显式的draft_cell_size_per_token（此时直接求和）。
    """
    if target_cell_size_per_token <= 0:  # 目标单元格大小必须为正
        raise ValueError(
            "target_cell_size_per_token must be positive, "
            f"got {target_cell_size_per_token}."
        )

    if draft_cell_size_per_token is not None:  # 如果提供了草稿单元格大小
        draft_cell_size_per_token = int(draft_cell_size_per_token)  # 转为整数
        if draft_cell_size_per_token <= 0:  # 草稿单元格大小必须为正
            raise ValueError(
                "draft_cell_size_per_token must be positive when provided, "
                f"got {draft_cell_size_per_token}."
            )
        return int(target_cell_size_per_token) + int(draft_cell_size_per_token)  # 直接求和

    if target_num_layers <= 0 or draft_num_layers <= 0:  # 层数无效时
        return int(target_cell_size_per_token)  # 返回目标大小

    total_layers = int(target_num_layers) + int(draft_num_layers)  # 总层数
    return (  # 按比例缩放
        int(target_cell_size_per_token) * int(total_layers) + int(target_num_layers) - 1
    ) // int(target_num_layers)


def resolve_dflash_verify_mask_policy(attn_backend: Any) -> tuple[str, bool]:
    # 解析DFlash验证的掩码策略，返回后端名称和是否需要构建自定义掩码
    backend = attn_backend  # 获取后端
    for _ in range(4):  # 最多解包4层包装
        full_backend = getattr(backend, "full_attn_backend", None)  # 获取完整注意力后端
        if full_backend is None:  # 如果没有包装
            break  # 停止解包
        backend = full_backend  # 继续解包
    backend_name = type(backend).__name__  # 获取后端类名
    return backend_name, (backend_name not in _DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS)  # 返回名称和是否需要掩码


def _get_or_create_chain_verify_buffers(
    *,
    bs: int,  # 批次大小
    draft_token_num: int,  # 草稿token数
    device: torch.device,  # 设备
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    # 获取或创建链式验证缓冲区
    key = (device.index, int(draft_token_num))  # 缓存键
    cached = _DFLASH_CHAIN_VERIFY_BUFFERS.get(key)  # 查找缓存
    cap_bs = 0 if cached is None else int(cached["cap_bs"])  # 当前容量
    if cap_bs < bs:  # 如果容量不足
        new_cap = max(int(bs), cap_bs * 2 if cap_bs > 0 else int(bs))  # 新容量
        retrieve_index = torch.arange(  # 检索索引
            new_cap * draft_token_num, dtype=torch.int64, device=device
        ).view(new_cap, draft_token_num)
        row_next = torch.arange(  # 行内下一个token索引
            1, draft_token_num + 1, dtype=torch.int64, device=device
        )
        row_next[-1] = -1  # 最后一个设为-1（无下一个）
        retrieve_next_token = row_next.unsqueeze(0).expand(new_cap, -1).clone()  # 扩展到批次维度
        retrieve_next_sibling = torch.full(  # 同级下一个token索引（链式结构中无同级）
            (new_cap, draft_token_num), -1, dtype=torch.int64, device=device
        )
        predicts = torch.empty(  # 预测token缓冲区
            (new_cap * draft_token_num,), dtype=torch.int32, device=device
        )
        accept_index = torch.empty(  # 接受索引缓冲区
            (new_cap, draft_token_num), dtype=torch.int32, device=device
        )
        accept_token_num = torch.empty((new_cap,), dtype=torch.int32, device=device)  # 接受token数缓冲区
        cached = {  # 缓存字典
            "cap_bs": int(new_cap),
            "retrieve_index": retrieve_index,
            "retrieve_next_token": retrieve_next_token,
            "retrieve_next_sibling": retrieve_next_sibling,
            "predicts": predicts,
            "accept_index": accept_index,
            "accept_token_num": accept_token_num,
        }
        _DFLASH_CHAIN_VERIFY_BUFFERS[key] = cached  # 存入缓存

    assert cached is not None  # 确保缓存存在
    retrieve_index = cached["retrieve_index"][:bs]  # 切片到当前批次大小
    retrieve_next_token = cached["retrieve_next_token"][:bs]  # 切片
    retrieve_next_sibling = cached["retrieve_next_sibling"][:bs]  # 切片
    predicts = cached["predicts"][: bs * draft_token_num]  # 切片
    accept_index = cached["accept_index"][:bs]  # 切片
    accept_token_num = cached["accept_token_num"][:bs]  # 切片
    return (  # 返回所有缓冲区
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    )


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Select target layer indices used to build DFlash context features.
    # 选择用于构建DFlash上下文特征的目标层索引。

    Args:
    # 参数：
        num_target_layers: Number of transformer layers in the runtime target model.
        # 运行时目标模型中的Transformer层数。
        num_draft_layers: Number of layers in the DFlash draft model.
        # DFlash草稿模型中的层数。

    Returns:
    # 返回值：
        A list of 0-based target layer indices of length `num_draft_layers`.
        # 长度为num_draft_layers的从0开始的目标层索引列表。

    Notes:
    # 注意：
        - DFlash uses hidden states after each selected target layer (HF-style).
        # DFlash使用每个选定目标层之后的隐藏状态（HF风格）。
        - SGLang captures "before layer i", so the model hook will typically add +1
          when mapping to capture points.
        # SGLang捕获"层i之前"，因此模型钩子映射到捕获点时通常加1。
    """
    if num_target_layers <= 0:  # 目标层数必须为正
        raise ValueError(
            f"num_target_layers must be positive, got {num_target_layers}."
        )
    if num_draft_layers <= 0:  # 草稿层数必须为正
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}.")

    if num_draft_layers == 1:  # 单层草稿模型
        return [num_target_layers // 2]  # 返回目标模型的中间层

    start = 1  # 起始层索引
    end = num_target_layers - 3  # 结束层索引
    if end < start:  # 层数不足
        raise ValueError(
            "DFlash layer selection requires num_target_layers >= 4. "
            f"Got num_target_layers={num_target_layers}."
        )

    span = end - start  # 层索引跨度
    return [  # 均匀分布的层索引列表
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    # 从配置对象或字典中获取值
    if isinstance(config, dict):  # 字典类型
        return config.get(key, default)  # 使用get方法
    return getattr(config, key, default)  # 对象类型使用getattr


def _get_text_config(config: Any) -> Any:
    # 从模型配置中获取文本配置
    if config is None:  # 配置为空
        return None  # 返回None
    if isinstance(config, dict):  # 字典类型
        return config.get("text_config", config)  # 获取text_config或返回自身
    text_config = getattr(config, "text_config", None)  # 对象类型获取text_config属性
    if text_config is not None:  # 如果存在
        return text_config  # 返回
    get_text_config = getattr(config, "get_text_config", None)  # 获取get_text_config方法
    if callable(get_text_config):  # 如果可调用
        try:
            resolved = get_text_config()  # 调用获取
            if resolved is not None:  # 如果结果不为空
                return resolved  # 返回
        except TypeError:  # 类型错误
            pass  # 忽略
    return config  # 返回原始配置


def _get_dflash_config(config: Any) -> dict:
    # 从模型配置中获取DFlash配置
    if isinstance(config, dict):  # 字典类型
        cfg = config.get("dflash_config", None)  # 获取dflash_config
    else:
        cfg = getattr(config, "dflash_config", None)  # 对象类型获取属性
    if cfg is None:  # 不存在
        return {}  # 返回空字典
    if isinstance(cfg, dict):  # 已是字典
        return cfg  # 直接返回

    try:
        return dict(cfg)  # 尝试转为字典
    except Exception:  # 转换失败
        return {}  # 返回空字典


def _parse_optional_int(
    value: Any,  # 输入值
    *,
    field_name: str,  # 字段名（用于错误信息）
    min_value: Optional[int] = None,  # 最小值约束
) -> Optional[int]:
    # 解析可选整数字段
    if value is None:  # 值为None
        return None  # 返回None
    try:
        parsed = int(value)  # 尝试转为整数
    except Exception as e:  # 转换失败
        raise ValueError(f"Invalid {field_name}={value!r}.") from e  # 抛出异常
    if min_value is not None and parsed < int(min_value):  # 小于最小值
        comparator = "positive" if int(min_value) == 1 else f">= {int(min_value)}"  # 比较描述
        raise ValueError(f"{field_name} must be {comparator}, got {parsed}.")  # 抛出异常
    return parsed  # 返回解析后的整数


@dataclass(frozen=True)  # 不可变数据类
class DFlashDraftConfig:
    num_hidden_layers: Optional[int]  # 草稿模型隐藏层数
    num_target_layers: Optional[int]  # 目标层数
    block_size: Optional[int]  # 块大小
    target_layer_ids: Optional[List[int]]  # 目标层ID列表
    mask_token: str  # 掩码token字符串
    mask_token_id: Optional[int]  # 掩码token ID

    def require_num_layers(self) -> int:
        # 要求并返回草稿模型隐藏层数
        if self.num_hidden_layers is None:  # 未设置
            raise ValueError(
                "DFLASH requires draft num_hidden_layers in config. "
                "Got config without num_hidden_layers."
            )
        return int(self.num_hidden_layers)  # 返回层数

    def resolve_block_size(self, *, default: Optional[int] = None) -> Optional[int]:
        # 解析块大小，未设置时使用默认值
        return self.block_size if self.block_size is not None else default  # 返回块大小或默认值

    def resolve_target_layer_ids(
        self,
        *,
        target_num_layers: int,  # 目标模型层数
        draft_num_layers: Optional[int] = None,  # 草稿模型层数
    ) -> List[int]:
        # 解析目标层ID列表
        target_num_layers = int(target_num_layers)  # 转为整数
        if target_num_layers <= 0:  # 层数必须为正
            raise ValueError(
                f"target_num_layers must be positive, got {target_num_layers}."
            )

        if self.target_layer_ids is None:  # 未显式指定层ID
            if draft_num_layers is None:  # 也未指定草稿层数
                draft_num_layers = self.require_num_layers()  # 从配置获取
            return build_target_layer_ids(target_num_layers, int(draft_num_layers))  # 自动构建

        resolved = list(self.target_layer_ids)  # 使用显式指定的层ID
        if len(resolved) <= 0:  # 列表不能为空
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(resolved)}."
            )
        for idx, val in enumerate(resolved):  # 检查范围
            if val < 0 or val >= target_num_layers:  # 超出范围
                raise ValueError(
                    "DFLASH target_layer_ids contains an out-of-range layer id. "
                    f"target_layer_ids[{idx}]={val}, target_num_layers={target_num_layers}."
                )
        return resolved  # 返回层ID列表


def parse_dflash_draft_config(*, draft_hf_config: Any) -> DFlashDraftConfig:
    """Parse and validate DFLASH draft config fields from HF config/dict."""
    # 从HuggingFace配置/字典解析并验证DFLASH草稿配置字段。
    dflash_cfg = _get_dflash_config(draft_hf_config)  # 获取DFlash配置
    draft_text_config = _get_text_config(draft_hf_config)  # 获取文本配置

    num_hidden_layers = _parse_optional_int(  # 解析隐藏层数
        _cfg_get(draft_text_config, "num_hidden_layers", None),
        field_name="DFLASH draft num_hidden_layers",
        min_value=1,
    )
    raw_num_target_layers = dflash_cfg.get(  # 获取原始目标层数
        "num_target_layers",
        _cfg_get(draft_hf_config, "num_target_layers", None),
    )
    num_target_layers = _parse_optional_int(  # 解析目标层数
        raw_num_target_layers,
        field_name="DFLASH draft num_target_layers",
        min_value=1,
    )

    # Keep support for current checkpoints where block_size is top-level.
    # 保持对当前检查点的支持，其中block_size是顶级字段。
    raw_block_size = dflash_cfg.get(  # 获取原始块大小
        "block_size",
        _cfg_get(draft_hf_config, "block_size", None),
    )
    block_size = _parse_optional_int(  # 解析块大小
        raw_block_size,
        field_name="DFLASH block_size",
        min_value=1,
    )

    layer_ids = dflash_cfg.get(  # 获取目标层ID
        "target_layer_ids",
        _cfg_get(draft_hf_config, "target_layer_ids", None),
    )
    parsed_target_layer_ids: Optional[List[int]]  # 解析后的层ID
    if layer_ids is None:  # 未指定
        parsed_target_layer_ids = None  # 设为None
    else:  # 已指定
        if not isinstance(layer_ids, (list, tuple)):  # 必须是列表或元组
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be a list of ints, "
                f"got type={type(layer_ids).__name__}."
            )
        parsed_target_layer_ids = [int(x) for x in layer_ids]  # 转为整数列表
        if len(parsed_target_layer_ids) <= 0:  # 不能为空
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(parsed_target_layer_ids)}."
            )

    mask_token = dflash_cfg.get("mask_token", None)  # 获取掩码token
    if mask_token is None:  # 未指定
        mask_token = DEFAULT_DFLASH_MASK_TOKEN  # 使用默认值
    if not isinstance(mask_token, str) or not mask_token:  # 必须是非空字符串
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )

    mask_token_id = dflash_cfg.get("mask_token_id", None)  # 获取掩码token ID
    if mask_token_id is not None:  # 已指定
        if not isinstance(mask_token_id, Integral) or isinstance(mask_token_id, bool):  # 必须是整数
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be an integer, "
                f"got {mask_token_id!r} (type={type(mask_token_id).__name__})."
            )
        mask_token_id = int(mask_token_id)  # 转为整数
        if mask_token_id < 0:  # 不能为负
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be non-negative, "
                f"got {mask_token_id}."
            )

    return DFlashDraftConfig(  # 返回配置对象
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
        block_size=block_size,
        target_layer_ids=parsed_target_layer_ids,
        mask_token=mask_token,
        mask_token_id=mask_token_id,
    )


def can_dflash_slice_qkv_weight(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether DFlash can slice KV weights from a fused QKV linear layer."""
    # 验证DFlash是否可以从融合QKV线性层中切片KV权重。
    quant_method = getattr(qkv_proj, "quant_method", None)  # 获取量化方法
    if not isinstance(quant_method, UnquantizedLinearMethod):  # 必须是未量化的
        return (
            False,
            "quantized qkv_proj is not supported for this path "
            f"(quant_method={type(quant_method).__name__})",
        )
    if not hasattr(qkv_proj, "weight"):  # 必须有权重
        return False, "qkv weight tensor is missing"
    return True, ""  # 可以切片


def can_dflash_use_fused_qkv_proj(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether a QKV layer is eligible for DFlash fused KV materialization."""
    # 验证QKV层是否适用于DFlash融合KV物化。
    eligible, reason = can_dflash_slice_qkv_weight(qkv_proj)  # 检查切片条件
    if not eligible:  # 不满足
        return False, reason  # 返回原因
    if getattr(qkv_proj, "bias", None) is not None:  # 不支持偏置
        return False, "qkv bias is not supported for fused KV path"
    return True, ""  # 可以使用


def compute_dflash_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,  # 草稿候选token
    target_predict: torch.Tensor,  # 目标模型预测token
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).
    # 计算DFlash接受长度和奖励token（贪心验证规则）。

    Args:
    # 参数：
        candidates: Token ids proposed by the DFlash draft, including the current token.
        # DFlash草稿提出的token ID，包括当前token。
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
            # 形状：[bs, block_size]。candidates[:, 0]是当前token。
        target_predict: Token ids predicted by the target model for each position in the block.
        # 目标模型对块中每个位置预测的token ID。
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.
            # 形状：[bs, block_size]。target_predict[:, t]对应位置t的argmax。

    Returns:
    # 返回值：
        correct_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        # correct_len：int32张量[bs]，接受的草稿token数（不包括当前token和奖励token）。
        bonus: int64 tensor [bs], the target-predicted token at index correct_len (the "bonus" token to append).
        # bonus：int64张量[bs]，在correct_len索引处目标预测的token（要追加的"奖励"token）。

    Notes:
    # 注意：
        Matches the reference implementation rule:
        # 匹配参考实现规则：
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
        # 当candidates[:, 1:] == target_predict[:, :-1]连续相等时接受。
    """
    if candidates.ndim != 2:  # 候选必须是2D
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:  # 形状必须一致
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape  # 获取批次大小和块大小
    if bs <= 0:  # 批次大小必须为正
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:  # 块大小必须为正
        raise ValueError(f"block_size must be positive, got {block_size}.")

    matches = candidates[:, 1:] == target_predict[:, :-1]  # 比较候选与预测是否匹配
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)  # 累积乘积求和得到连续正确长度
    bonus = target_predict[torch.arange(bs, device=target_predict.device), correct_len]  # 获取奖励token
    return correct_len, bonus.to(torch.int64)  # 返回正确长度和奖励token


def compute_dflash_sampling_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,  # 草稿候选token
    next_token_logits: torch.Tensor,  # 目标模型logits
    sampling_info: Any,  # 采样信息
    threshold_single: Optional[float] = None,  # 单步接受阈值
    threshold_acc: Optional[float] = None,  # 累积接受阈值
    uniform_samples: Optional[torch.Tensor] = None,  # 均匀随机样本
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,  # 最终采样的均匀随机样本
    use_sparse_topk: bool = True,  # 是否使用稀疏topk
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens for non-greedy sampling.
    # 计算DFlash非贪心采样的接受长度和奖励token。

    This is a chain-specialized variant of speculative target-only verification:
    # 这是投机仅目标验证的链式特化变体：
      - DFlash proposals are linear (topk == 1), so each verify level has at most one candidate.
      # DFlash提议是线性的（topk==1），因此每个验证级别最多有一个候选。
      - When a candidate is rejected at a level, the final token is sampled from
        `relu(q - p)` where `p` has only the rejected candidate mass.
      # 当候选在某级别被拒绝时，最终token从relu(q-p)采样，其中p只包含被拒绝候选的质量。
    """
    if not _DFLASH_SAMPLING_VERIFY_AVAILABLE:  # 采样验证不可用
        raise RuntimeError(
            "DFLASH non-greedy verification is unavailable on this build/device."
        )
    if candidates.ndim != 2:  # 候选必须是2D
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if next_token_logits.ndim != 2:  # logits必须是2D
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )

    bs, draft_token_num = candidates.shape  # 批次大小和草稿token数
    if bs <= 0:  # 批次大小必须为正
        raise ValueError(f"batch size must be positive, got {bs}.")
    if draft_token_num <= 0:  # 草稿token数必须为正
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")
    if next_token_logits.shape[0] != bs * draft_token_num:  # logits行数不匹配
        raise ValueError(
            "next_token_logits row count mismatch. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )
    if candidates.device != next_token_logits.device:  # 设备必须一致
        raise ValueError(
            "candidates and next_token_logits must be on the same device, "
            f"got {candidates.device} and {next_token_logits.device}."
        )

    if threshold_single is None:  # 未指定单步阈值
        from sglang.srt.server_args import get_global_server_args  # 延迟导入

        threshold_single = get_global_server_args().speculative_accept_threshold_single  # 获取全局配置
    if threshold_acc is None:  # 未指定累积阈值
        from sglang.srt.server_args import get_global_server_args  # 延迟导入

        threshold_acc = get_global_server_args().speculative_accept_threshold_acc  # 获取全局配置
    threshold_single = float(threshold_single)  # 转为浮点
    threshold_acc = max(float(threshold_acc), 1e-9)  # 确保不为零

    device = next_token_logits.device  # 设备

    if uniform_samples is None:  # 未提供随机样本
        uniform_samples = torch.rand(  # 生成随机样本
            (bs, draft_token_num), dtype=torch.float32, device=device
        )
    else:  # 已提供
        if uniform_samples.shape != (bs, draft_token_num):  # 形状不匹配
            raise ValueError(
                "uniform_samples shape mismatch. "
                f"Expected {(bs, draft_token_num)}, got {tuple(uniform_samples.shape)}."
            )
        uniform_samples = uniform_samples.to(device=device, dtype=torch.float32)  # 转换设备和类型

    if uniform_samples_for_final_sampling is None:  # 未提供最终采样随机样本
        uniform_samples_for_final_sampling = torch.rand(  # 生成
            (bs,), dtype=torch.float32, device=device
        )
    else:  # 已提供
        if uniform_samples_for_final_sampling.shape != (bs,):  # 形状不匹配
            raise ValueError(
                "uniform_samples_for_final_sampling shape mismatch. "
                f"Expected {(bs,)}, got {tuple(uniform_samples_for_final_sampling.shape)}."
            )
        uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.to(  # 转换
            device=device,
            dtype=torch.float32,
        )

    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))  # 是否需要top-k采样
    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))  # 是否需要top-p采样
    # Build target distribution once over all verify rows.
    # 在所有验证行上一次性构建目标分布。
    expanded_temperature = torch.repeat_interleave(  # 扩展温度到所有验证行
        sampling_info.temperatures, draft_token_num, dim=0
    )
    scaled_logits = next_token_logits / expanded_temperature  # 缩放logits
    sparse_topk_applied = False  # 稀疏topk是否已应用

    if use_sparse_topk and need_top_k:  # 使用稀疏topk且需要top-k
        repeated_top_ks = torch.repeat_interleave(  # 扩展top-k值
            sampling_info.top_ks, draft_token_num, dim=0
        ).to(dtype=torch.int64)
        vocab_size = int(scaled_logits.shape[-1])  # 词表大小
        repeated_top_ks.clamp_(min=1, max=vocab_size)  # 限制top-k范围
        max_top_k = int(repeated_top_ks.max().item())  # 最大top-k值

        # Sparse exact path for top-k/top-p (top-k-first semantics), then scatter to dense.
        # top-k/top-p的稀疏精确路径（top-k优先语义），然后散射到稠密。
        if 0 < max_top_k < vocab_size:  # top-k小于词表大小
            topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)  # 取top-k
            if not torch.all(repeated_top_ks == max_top_k):  # top-k不全部相同
                ranks = torch.arange(max_top_k, device=device, dtype=torch.int64)[
                    None, :
                ]
                valid = ranks < repeated_top_ks.unsqueeze(1)  # 有效位置掩码
                topk_logits = topk_logits.masked_fill(~valid, float("-inf"))  # 无效位置设为负无穷

            topk_probs = F.softmax(topk_logits, dim=-1)  # 计算概率
            if need_top_p:  # 需要top-p
                repeated_top_ps = torch.repeat_interleave(  # 扩展top-p值
                    sampling_info.top_ps, draft_token_num, dim=0
                )
                topk_probs = top_p_renorm_prob(topk_probs, repeated_top_ps)  # top-p重归一化

            target_probs = torch.zeros_like(scaled_logits, dtype=topk_probs.dtype)  # 创建全零概率张量
            target_probs.scatter_(1, topk_indices, topk_probs)  # 散射到稠密
            sparse_topk_applied = True  # 标记已应用

    if not sparse_topk_applied:  # 未使用稀疏topk
        target_probs = F.softmax(scaled_logits, dim=-1)  # 直接softmax
        if need_top_k:  # 需要top-k
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
            )
        if need_top_p:  # 需要top-p
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
            )
    target_probs = target_probs.view(bs, draft_token_num, -1).contiguous()  # 重塑并确保连续
    draft_probs = torch.zeros_like(target_probs)  # 草稿概率（全零，由内核填充）

    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(  # 获取链式验证缓冲区
        bs=bs,
        draft_token_num=draft_token_num,
        device=device,
    )
    candidates_i64 = (  # 转换候选为int64
        candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
    )
    tree_speculative_sampling_target_only(  # 调用CUDA内核进行树投机采样
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates_i64,
        # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
        # 关键字参数保留为retrive_*以匹配sgl_kernel算子模式。
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    correct_len = accept_token_num  # 接受的token数
    row_ids = torch.arange(bs, dtype=torch.long, device=device)  # 行ID
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)  # 接受位置
    bonus = predicts[accept_pos].to(torch.int64)  # 奖励token
    return correct_len, bonus  # 返回正确长度和奖励token


def validate_dflash_request(req: Req) -> Optional[str]:
    # 验证请求是否支持DFlash投机解码，不支持则返回原因字符串
    if req.return_logprob:  # 不支持返回logprob
        return "DFLASH speculative decoding does not support return_logprob yet."

    if (  # 不支持语法约束解码
        req.sampling_params.json_schema is not None
        or req.sampling_params.regex is not None
        or req.sampling_params.ebnf is not None
        or req.sampling_params.structural_tag is not None
    ):
        return (
            "DFLASH speculative decoding does not support "
            "grammar-constrained decoding yet."
        )

    return None  # 请求合法，返回None
