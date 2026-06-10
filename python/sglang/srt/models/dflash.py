# DFlash推测解码草稿模型实现
# 本文件实现了DFlash推测解码的草稿模型（draft model），
# 基于DFlash参考实现但使用SGLang原语（RadixAttention + SGLang KV缓存）。
# 该模型故意不包含token嵌入层或LM头；DFlash使用目标模型的嵌入/LM头。
# 包含注意力层、MLP层、解码器层和完整草稿模型。

# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or an LM head; DFlash uses the target model's
# embedding/lm_head.
# 改编自DFlash参考实现（HF），但使用SGLang原语（RadixAttention + SGLang KV缓存）实现。
# 该模型故意不包含token嵌入或LM头；DFlash使用目标模型的嵌入/lm_head。

from __future__ import annotations  # 启用延迟注解评估

import logging  # 导入日志模块
from typing import Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch核心库
import torch.nn.functional as F  # 导入PyTorch函数式接口
from torch import nn  # 导入神经网络模块

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小函数
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU与乘法激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 导入线性层模块
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # 导入logits处理器输出
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention  # 导入注意力类型和Radix注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.utils import apply_qk_norm  # 导入QK归一化应用函数
from sglang.srt.speculative.dflash_utils import (  # 导入DFlash推测解码工具
    can_dflash_slice_qkv_weight,  # 判断是否可以切片QKV权重
    parse_dflash_draft_config,  # 解析DFlash草稿配置
)
from sglang.srt.utils import is_npu  # 导入NPU判断函数

_is_npu = is_npu()  # 判断当前是否为NPU环境
if _is_npu:  # 如果是NPU环境
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope  # 导入NPU融合核函数
logger = logging.getLogger(__name__)  # 创建模块级日志记录器


class DFlashAttention(nn.Module):
    """DFlash注意力层，实现非因果注意力用于草稿块"""
    def __init__(self, config, layer_id: int) -> None:  # 初始化方法
        super().__init__()  # 调用父类初始化
        hidden_size = int(config.hidden_size)  # 获取隐藏层维度
        tp_size = int(get_tensor_model_parallel_world_size())  # 获取张量并行大小
        total_num_heads = int(config.num_attention_heads)  # 获取总注意力头数
        total_num_kv_heads = int(  # 获取总KV头数
            getattr(config, "num_key_value_heads", total_num_heads)  # 如果未指定则与注意力头数相同
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))  # 获取每个头的维度

        self.hidden_size = hidden_size  # 保存隐藏层维度
        self.total_num_heads = total_num_heads  # 保存总注意力头数
        self.total_num_kv_heads = total_num_kv_heads  # 保存总KV头数
        assert self.total_num_heads % tp_size == 0, (  # 断言总头数可被TP大小整除
            f"DFlashAttention requires total_num_heads divisible by tp_size. "
            f"total_num_heads={self.total_num_heads}, tp_size={tp_size}."
        )
        self.num_heads = self.total_num_heads // tp_size  # 计算当前TP秩的头数
        if self.total_num_kv_heads >= tp_size:  # 如果KV头数 >= TP大小
            assert self.total_num_kv_heads % tp_size == 0, (  # 断言KV头数可被TP大小整除
                f"DFlashAttention requires total_num_kv_heads divisible by tp_size when >= tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        else:  # 否则KV头数 < TP大小
            assert tp_size % self.total_num_kv_heads == 0, (  # 断言TP大小可被KV头数整除
                f"DFlashAttention requires tp_size divisible by total_num_kv_heads when total_num_kv_heads < tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 计算当前TP秩的KV头数，至少为1
        self.head_dim = head_dim  # 保存每个头的维度
        self.q_size = self.num_heads * head_dim  # 计算Q的总维度
        self.kv_size = self.num_kv_heads * head_dim  # 计算KV的总维度

        attention_bias = bool(getattr(config, "attention_bias", False))  # 获取是否使用注意力偏置
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))  # 获取RMS归一化epsilon

        self.qkv_proj = QKVParallelLinear(  # 创建QKV并行投影层
            hidden_size=hidden_size,  # 输入隐藏维度
            head_size=head_dim,  # 每个头的维度
            total_num_heads=self.total_num_heads,  # 总Q头数
            total_num_kv_heads=self.total_num_kv_heads,  # 总KV头数
            bias=attention_bias,  # 是否使用偏置
            prefix="qkv_proj",  # 前缀名
        )
        self.o_proj = RowParallelLinear(  # 创建输出投影层
            self.total_num_heads * head_dim,  # 输入维度
            hidden_size,  # 输出维度
            bias=attention_bias,  # 是否使用偏置
            prefix="o_proj",  # 前缀名
        )

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        # 每头Q/K的RMS归一化，与HF Qwen3匹配
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)  # Q归一化层
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)  # K归一化层

        rope_theta = float(getattr(config, "rope_theta", 1000000))  # 获取RoPE基础频率
        rope_scaling = getattr(config, "rope_scaling", None)  # 获取RoPE缩放配置
        rope_is_neox_style = bool(  # 获取RoPE是否为Neox风格
            getattr(
                config, "rope_is_neox_style", getattr(config, "is_neox_style", True)  # 默认为True
            )
        )
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))  # 获取最大位置嵌入数
        self.rotary_emb = get_rope(  # 创建旋转位置编码
            head_dim,  # 头维度
            rotary_dim=head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置数
            base=rope_theta,  # 基础频率
            rope_scaling=rope_scaling,  # 缩放配置
            is_neox_style=rope_is_neox_style,  # 是否Neox风格
        )

        self.scaling = head_dim**-0.5  # 计算注意力缩放因子
        # DFlash uses non-causal attention over the draft block.
        # DFlash在草稿块上使用非因果注意力
        self.attn = RadixAttention(  # 创建Radix注意力
            num_heads=self.num_heads,  # 注意力头数
            head_dim=head_dim,  # 每个头的维度
            scaling=self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            attn_type=AttentionType.ENCODER_ONLY,  # 使用编码器-only（非因果）注意力
        )

    def forward_prepare_npu(self, positions, hidden_states):  # NPU专用前向准备方法
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影层获取QKV

        if self.attn.layer_id == 0:  # 如果是第一层
            self.rotary_emb.get_cos_sin_with_position(positions)  # 预计算位置相关的cos/sin值
        q, k, v = split_qkv_rmsnorm_rope(  # 使用NPU融合核函数一次性完成QKV拆分、归一化和RoPE
            qkv,  # QKV张量
            self.rotary_emb.position_sin,  # 位置sin值
            self.rotary_emb.position_cos,  # 位置cos值
            self.q_size,  # Q维度
            self.kv_size,  # KV维度
            self.head_dim,  # 每个头的维度
            eps=self.q_norm.variance_epsilon,  # 归一化epsilon
            q_weight=self.q_norm.weight,  # Q归一化权重
            k_weight=self.k_norm.weight,  # K归一化权重
            q_bias=getattr(self.q_norm, "bias", None),  # Q归一化偏置
            k_bias=getattr(self.k_norm, "bias", None),  # K归一化偏置
        )
        return q, k, v  # 返回Q、K、V

    def forward(  # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影层获取QKV
        if _is_npu:  # 如果是NPU环境
            q, k, v = self.forward_prepare_npu(positions, hidden_states)  # 使用NPU专用前向准备
        else:  # 否则
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分QKV
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)  # 对Q和K应用归一化
            q, k = self.rotary_emb(positions, q, k)  # 对Q和K应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 通过注意力层计算输出
        output, _ = self.o_proj(attn_output)  # 通过输出投影层
        return output  # 返回注意力输出

    def kv_proj_only(  # 仅KV投影方法，用于将上下文token物化到草稿KV缓存
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).
        仅将隐藏状态投影到K/V（跳过Q）。

        This is used by DFlash to materialize ctx tokens into the draft KV cache:
        we only need K/V for the cached tokens; Q is never consumed.
        这被DFlash用于将上下文token物化到草稿KV缓存中：
        我们只需要缓存token的K/V；Q永远不会被消费。
        """
        # Fast path for unquantized weights: slice the fused QKV weight and run one GEMM.
        # 非量化权重的快速路径：切片融合的QKV权重并运行一次GEMM
        can_slice_qkv_weight, _ = can_dflash_slice_qkv_weight(self.qkv_proj)  # 判断是否可以切片QKV权重
        if can_slice_qkv_weight:  # 如果可以切片
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)  # 计算KV切片范围
            weight = self.qkv_proj.weight[kv_slice]  # 获取KV权重切片
            bias = (  # 获取KV偏置切片
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None  # 如果偏置存在则切片
            )
            kv = F.linear(hidden_states, weight, bias)  # 一次GEMM计算KV
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)  # 拆分为K和V
            return k, v  # 返回K和V

        # Fallback: compute full QKV and discard Q (keeps compatibility with quantized weights).
        # 回退方案：计算完整QKV并丢弃Q（保持与量化权重的兼容性）
        qkv, _ = self.qkv_proj(hidden_states)  # 计算完整QKV
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分并丢弃Q
        return k, v  # 返回K和V

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:  # 对K应用归一化
        k_by_head = k.reshape(-1, self.head_dim)  # 按头重塑K的形状
        k_by_head = self.k_norm(k_by_head)  # 对每个头应用K归一化
        return k_by_head.view_as(k)  # 恢复原始形状并返回

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:  # 对K应用旋转位置编码
        # Match K shape so RoPE kernel head-count check passes on all backends.
        # 匹配K形状，以便RoPE核函数头数检查在所有后端上通过
        dummy_q = k.new_empty(k.shape)  # 创建与K同形状的空Q张量（占位用）
        _, k = self.rotary_emb(positions, dummy_q, k)  # 应用旋转位置编码到K
        return k  # 返回编码后的K


class DFlashMLP(nn.Module):
    """DFlash MLP层，使用SiLU激活函数的门控线性单元"""
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:  # 初始化方法
        super().__init__()  # 调用父类初始化
        hidden_size = int(config.hidden_size)  # 获取隐藏层维度
        intermediate_size = int(getattr(config, "intermediate_size", 0))  # 获取中间层维度
        if intermediate_size <= 0:  # 如果中间层维度无效
            raise ValueError(  # 抛出值错误
                f"Invalid intermediate_size={intermediate_size} for DFlash MLP."
            )

        self.gate_up_proj = MergedColumnParallelLinear(  # 创建门控-上投影合并线性层
            hidden_size,  # 输入维度
            [intermediate_size] * 2,  # 输出维度列表（门控和上投影各一份）
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",  # 前缀名
        )
        self.down_proj = RowParallelLinear(  # 创建下投影线性层
            intermediate_size,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",  # 前缀名
        )
        hidden_act = getattr(config, "hidden_act", "silu")  # 获取隐藏层激活函数名
        if hidden_act != "silu":  # 如果不是SiLU
            raise ValueError(  # 抛出值错误
                f"Unsupported DFlash activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()  # 创建SiLU与乘法激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        gate_up, _ = self.gate_up_proj(x)  # 通过门控-上投影层
        x = self.act_fn(gate_up)  # 应用SiLU激活并做门控乘法
        x, _ = self.down_proj(x)  # 通过下投影层
        return x  # 返回MLP输出


class DFlashDecoderLayer(nn.Module):
    """DFlash解码器层，包含自注意力和MLP"""
    def __init__(self, config, layer_id: int) -> None:  # 初始化方法
        super().__init__()  # 调用父类初始化
        hidden_size = int(config.hidden_size)  # 获取隐藏层维度
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))  # 获取RMS归一化epsilon

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)  # 输入层归一化
        self.self_attn = DFlashAttention(config=config, layer_id=layer_id)  # 自注意力层
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)  # 注意力后归一化
        self.mlp = DFlashMLP(config=config)  # MLP层

    def forward(  # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差连接，可选
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:  # 如果隐藏状态为空
            # Keep return types consistent for upstream callers.
            # 保持返回类型一致，以便上游调用者使用
            if residual is None:  # 如果没有残差
                residual = hidden_states  # 残差等于隐藏状态
            return hidden_states, residual  # 返回隐藏状态和残差

        # Pre-norm attention with fused residual+norm when possible (Qwen3-style).
        # 预归一化注意力，可能时使用融合残差+归一化（Qwen3风格）
        if residual is None:  # 如果没有残差（第一层）
            residual = hidden_states  # 残差等于隐藏状态
            hidden_states = self.input_layernorm(hidden_states)  # 对隐藏状态做输入层归一化
        else:  # 否则有残差
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 融合归一化

        attn_out = self.self_attn(  # 通过自注意力层
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次信息
        )
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)  # 注意力后归一化
        hidden_states = self.mlp(hidden_states)  # 通过MLP层
        return hidden_states, residual  # 返回隐藏状态和残差


class DFlashDraftModel(nn.Module):
    """SGLang DFlash draft model (no embedding / lm_head weights).
    SGLang DFlash草稿模型（不含嵌入/LM头权重）。

    The checkpoint provides:
    检查点提供：
      - transformer weights for `layers.*`
      `layers.*`的transformer权重
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      `fc.weight`、`hidden_norm.weight`用于投影目标上下文特征
      - `norm.weight` for final normalization
      `norm.weight`用于最终归一化
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        hidden_size = int(config.hidden_size)  # 获取隐藏层维度
        num_layers = int(config.num_hidden_layers)  # 获取隐藏层数
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))  # 获取RMS归一化epsilon

        self.layers = nn.ModuleList(  # 创建解码器层列表
            [DFlashDecoderLayer(config=config, layer_id=i) for i in range(num_layers)]  # 为每层创建解码器层
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)  # 最终归一化层

        # Project per-token target context features:
        # concat(K * hidden_size) -> hidden_size, where K is the number of target-layer
        # feature tensors concatenated per token (not necessarily equal to num_layers).
        # 投影每token的目标上下文特征：
        # concat(K * hidden_size) -> hidden_size，其中K是每个token连接的目标层特征张量数
        # （不一定等于num_layers）
        draft_config = parse_dflash_draft_config(draft_hf_config=config)  # 解析草稿配置
        target_num_layers = (  # 获取目标层数
            int(draft_config.num_target_layers)  # 转换为整数
            if draft_config.num_target_layers is not None  # 如果配置中指定了
            else num_layers  # 否则使用草稿层数
        )
        target_layer_ids = draft_config.resolve_target_layer_ids(  # 解析目标层ID列表
            target_num_layers=target_num_layers, draft_num_layers=num_layers  # 传入目标层数和草稿层数
        )
        num_context_features = len(target_layer_ids)  # 上下文特征数量等于目标层数

        self.num_context_features = int(num_context_features)  # 保存上下文特征数量
        self.fc = nn.Linear(  # 创建上下文特征投影层
            self.num_context_features * hidden_size, hidden_size, bias=False  # 输入维度为特征数×隐藏维度，输出为隐藏维度
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)  # 隐藏归一化层

        self.block_size = draft_config.resolve_block_size(default=16)  # 解析草稿块大小，默认16

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:  # 投影目标层隐藏状态
        """Project concatenated target-layer hidden states into draft hidden_size.
        将连接的目标层隐藏状态投影到草稿的隐藏维度。
        """
        expected = int(self.fc.in_features)  # 获取期望的特征维度
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:  # 如果维度不匹配
            raise ValueError(  # 抛出值错误
                "DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}. "
                "This usually means the target model is capturing a different number of layer features than "
                "the draft checkpoint/config expects."
            )
        return self.hidden_norm(self.fc(target_hidden))  # 通过线性投影和归一化返回

    @torch.no_grad()  # 禁用梯度计算装饰器
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入，可选
        get_embedding: bool = False,  # 是否获取嵌入，默认否
        pp_proxy_tensors=None,  # 流水线并行代理张量
    ) -> LogitsProcessorOutput:
        if input_embeds is None:  # 如果没有提供输入嵌入
            raise ValueError(  # 抛出值错误
                "DFlashDraftModel requires `input_embeds` (use the target embedding)."
            )
        hidden_states = input_embeds  # 使用目标模型的嵌入作为初始隐藏状态
        residual: Optional[torch.Tensor] = None  # 初始化残差为None

        for layer in self.layers:  # 遍历所有解码器层
            hidden_states, residual = layer(  # 通过解码器层
                positions, hidden_states, forward_batch, residual  # 传入位置、隐藏状态、批次和残差
            )

        if hidden_states.numel() != 0:  # 如果隐藏状态非空
            if residual is None:  # 如果没有残差
                hidden_states = self.norm(hidden_states)  # 直接归一化
            else:  # 否则有残差
                hidden_states, _ = self.norm(hidden_states, residual)  # 融合归一化

        return LogitsProcessorOutput(  # 返回logits处理器输出
            next_token_logits=None,  # 不计算logits（草稿模型不需要）
            hidden_states=hidden_states,  # 隐藏状态
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法
        stacked_params_mapping = [  # 堆叠参数映射表
            # (param_name, weight_name, shard_id)
            # （参数名，权重名，分片ID）
            ("qkv_proj", "q_proj", "q"),  # Q投影映射
            ("qkv_proj", "k_proj", "k"),  # K投影映射
            ("qkv_proj", "v_proj", "v"),  # V投影映射
            ("gate_up_proj", "gate_proj", 0),  # 门控投影映射
            ("gate_up_proj", "up_proj", 1),  # 上投影映射
        ]

        params_dict = dict(self.named_parameters())  # 获取模型参数字典

        def resolve_param_name(name: str) -> Optional[str]:  # 解析参数名，处理有无model.前缀的情况
            if name in params_dict:  # 如果名称直接存在于参数字典
                return name  # 返回原始名称
            if name.startswith("model."):  # 如果名称以"model."开头
                stripped_name = name[len("model.") :]  # 去掉"model."前缀
                if stripped_name in params_dict:  # 如果去掉前缀后存在
                    return stripped_name  # 返回去掉前缀的名称
            else:  # 否则名称不以"model."开头
                prefixed_name = f"model.{name}"  # 添加"model."前缀
                if prefixed_name in params_dict:  # 如果添加前缀后存在
                    return prefixed_name  # 返回添加前缀的名称
            return None  # 都不匹配返回None

        for name, loaded_weight in weights:  # 遍历所有权重
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if f".{weight_name}." not in name:  # 如果权重名不在参数名中
                    continue  # 跳过
                mapped_name = name.replace(weight_name, param_name)  # 替换权重名为参数名
                resolved_name = resolve_param_name(mapped_name)  # 解析参数名
                if resolved_name is None:  # 如果解析失败
                    continue  # 跳过
                param = params_dict[resolved_name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break  # 跳出内层循环
            else:  # 如果没有匹配到堆叠参数映射
                resolved_name = resolve_param_name(name)  # 解析参数名
                if resolved_name is None:  # 如果解析失败
                    # Ignore unexpected weights (e.g., HF rotary caches).
                    # 忽略意外的权重（例如HF旋转缓存）
                    continue  # 跳过
                param = params_dict[resolved_name]  # 获取参数
                if resolved_name.endswith("fc.weight") and tuple(  # 如果是fc.weight且形状不匹配
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(  # 抛出值错误
                        "DFLASH fc.weight shape mismatch. This usually means the draft checkpoint's "
                        "number of context features (K) does not match this config. "
                        f"Expected fc.weight.shape={tuple(param.shape)} "
                        f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                        f"but got {tuple(loaded_weight.shape)} for weight '{name}'."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = DFlashDraftModel  # 模型入口类注册
