# EAGLE3/EAGLE3.1 草稿模型实现（基于DeepSeek-V2 MLA注意力）
# 该文件实现了与Kimi-K2.x目标模型配对的EAGLE3投机解码草稿模型，
# 使用DeepSeek-V2多潜在注意力(MLA)以保持与目标模型一致的KV缓存形状。
# EAGLE3.1变体增加了fc_norm和norm_output配置标志。

"""EAGLE3 / EAGLE3.1 draft model with MLA attention for Kimi-K2.x.  # EAGLE3/EAGLE3.1草稿模型

The ``kimi-k2.5-eagle3-mla`` checkpoint pairs an EAGLE3 layout  # kimi-k2.5-eagle3-mla检查点
(concatenated [embed_norm, hidden_norm] pre-attention input, fc projection  # 拼接[嵌入归一化, 隐藏归一化]预注意力输入
over the concatenated multi-layer aux hidden states, single decoder layer,  # 对拼接的多层辅助隐藏状态进行fc投影
dense MLP) with DeepSeek-V2 multi-latent attention. Sharing the MLA layout  # 单解码器层，密集MLP
with the Kimi-K2.x target keeps the draft KV cache small.  # 与Kimi-K2.x目标共享MLA布局以保持KV缓存较小

The eagle3.1 variant (e.g. ``kimi-k2.6-eagle3.1-mla``) adds two optional  # eagle3.1变体增加了两个可选
config flags on top of the same layout:  # 配置标志

* ``fc_norm``: per-chunk RMSNorm applied to each auxiliary hidden state  # fc_norm: 对每个辅助隐藏状态应用RMSNorm
  before the fc projection.  # 在fc投影之前
* ``norm_output``: emit post-norm (rather than pre-norm) hidden states as  # norm_output: 输出后归一化隐藏状态
  the auxiliary output consumed by the next draft step.  # 作为下一个草稿步骤的辅助输出
"""

import copy  # 导入拷贝模块
import logging  # 导入日志模块
import re  # 导入正则表达式模块
from typing import Iterable, List, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.distributed import get_pp_group  # 导入流水线并行组
from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context  # 导入注意力通信器
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import ReplicatedLinear  # 导入复制线性层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, DeepseekV2MLP  # 导入DeepseekV2组件
from sglang.srt.utils import BumpAllocator, add_prefix  # 导入工具函数

logger = logging.getLogger(__name__)  # 获取日志记录器


def _get_eagle_aux_layer_count(config: PretrainedConfig) -> int:
    """获取EAGLE3辅助隐藏状态的层数（用于fc投影的拼接）。"""
    """Number of target layers whose hidden states get concatenated into fc."""  # 拼接到fc中的目标层数
    eagle_config = getattr(config, "eagle_config", None)  # 获取eagle配置
    if isinstance(eagle_config, dict):  # 如果是字典
        layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")  # 获取辅助层ID列表
    else:  # 否则
        layer_ids = getattr(eagle_config, "eagle_aux_hidden_state_layer_ids", None)  # 从对象获取
    if layer_ids is None:  # 如果列表不存在
        layer_ids = getattr(config, "eagle_aux_hidden_state_layer_ids", None)  # 从顶层配置获取
    if layer_ids is None:  # 如果仍然不存在
        return 3  # 默认3层
    return len(layer_ids)  # 返回层数


class Eagle3MLADecoderLayer(nn.Module):
    """EAGLE3草稿层，使用DeepSeek-V2多潜在注意力。"""
    """One EAGLE3 draft layer that uses DeepSeek-V2 multi-latent attention.  # 一个使用MLA的EAGLE3草稿层

    Pre-attention concatenates the input embedding and the target hidden  # 预注意力拼接输入嵌入和目标隐藏状态
    state along the channel dim, doubling the input width to MLA's fused  # 沿通道维度拼接，将MLA的输入宽度加倍
    QKV-down projection.  # 用于融合QKV下投影
    """

    def __init__(
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 保存隐藏大小

        if hasattr(config, "rope_parameters") and config.rope_parameters is not None:  # 如果有RoPE参数
            rope_params = config.rope_parameters  # 获取RoPE参数
            rope_theta = rope_params.get("rope_theta", 10000)  # 获取theta
            rope_scaling = (  # 获取缩放
                rope_params if rope_params.get("rope_type") != "default" else None  # 非默认时使用
            )
        else:  # 否则
            rope_theta = config.rope_theta  # 使用配置中的theta
            rope_scaling = config.rope_scaling  # 使用配置中的缩放
        max_position_embeddings = config.max_position_embeddings  # 最大位置数

        self.self_attn = DeepseekV2AttentionMLA(  # 创建MLA注意力模块
            config=config,  # 配置
            hidden_size=config.hidden_size,  # 隐藏大小
            num_heads=config.num_attention_heads,  # 头数
            qk_nope_head_dim=config.qk_nope_head_dim,  # 非旋转头维度
            qk_rope_head_dim=config.qk_rope_head_dim,  # 旋转头维度
            v_head_dim=config.v_head_dim,  # 值头维度
            q_lora_rank=config.q_lora_rank,  # Q LoRA秩
            kv_lora_rank=config.kv_lora_rank,  # KV LoRA秩
            rope_theta=rope_theta,  # RoPE theta
            rope_scaling=rope_scaling,  # RoPE缩放
            max_position_embeddings=max_position_embeddings,  # 最大位置
            quant_config=quant_config,  # 量化配置
            layer_id=layer_id,  # 层ID
            reduce_results=True,  # 归约结果
            prefix=add_prefix("self_attn", prefix),  # 参数前缀
        )

        # EAGLE3 doubles MLA's QKV-down input by concatenating  # EAGLE3将MLA的QKV下投影输入加倍
        # input_layernorm(embed) and hidden_norm(target_hidden) along the  # 通过拼接input_layernorm(嵌入)和hidden_norm(目标隐藏)
        # feature dim. Replace the projection that DeepseekV2AttentionMLA  # 替换DeepseekV2AttentionMLA
        # built for a single-hidden input.  # 为单隐藏输入构建的投影
        attn = self.self_attn  # 获取注意力模块引用
        if attn.q_lora_rank is None:  # 如果q_lora_rank不存在
            raise ValueError(  # 抛出错误
                "Eagle3 MLA layer requires q_lora_rank in the draft config"
            )
        attn.fused_qkv_a_proj_with_mqa = ReplicatedLinear(  # 替换为2倍输入维度的融合投影
            2 * config.hidden_size,  # 输入维度为2倍隐藏大小
            attn.q_lora_rank + attn.kv_lora_rank + attn.qk_rope_head_dim,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn.fused_qkv_a_proj_with_mqa", prefix),  # 参数前缀
        )
        # Recompute fused-proj-dependent flags so they reflect the new input dim.  # 重新计算依赖融合投影的标志
        attn.has_fused_proj = True  # 标记有融合投影
        attn.use_min_latency_fused_a_gemm = False  # 禁用最小延迟融合GEMM
        quant_method = getattr(attn.fused_qkv_a_proj_with_mqa, "quant_method", None)  # 获取量化方法
        attn.is_packed_weight = (  # 判断是否为打包权重
            quant_method is not None  # 量化方法存在
            and hasattr(quant_method, "quant_config")  # 有量化配置属性
            and quant_method.quant_config is not None  # 量化配置不为None
            and quant_method.quant_config.get_name()  # 量化名称
            in {"awq", "awq_marlin", "moe_wna16"}  # 支持的量化方法集合
        )

        self.mlp = DeepseekV2MLP(  # MLP模块
            hidden_size=config.hidden_size,  # 隐藏大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            hidden_act=config.hidden_act,  # 隐藏层激活
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 隐藏状态归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏大小和epsilon
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        embeds: torch.Tensor,  # 输入嵌入
        hidden_states: torch.Tensor,  # 目标隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
        zero_allocator: BumpAllocator,  # 零分配器
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播：EAGLE3草稿层计算。"""
        residual = hidden_states  # 保存残差
        embeds = self.input_layernorm(embeds)  # 对嵌入进行归一化
        hidden_states = self.hidden_norm(hidden_states)  # 对隐藏状态进行归一化
        attn_input = torch.cat([embeds, hidden_states], dim=-1)  # 拼接嵌入和隐藏状态

        # MLA's forward_absorb_prepare reads the qkv-down projection result  # MLA的forward_absorb_prepare读取QKV下投影结果
        # from attn_tp_context. We bypass LayerCommunicator here (the eagle3  # 从attn_tp_context。我们绕过LayerCommunicator
        # draft layer is one isolated layer with custom pre-attention norms),  # （eagle3草稿层是隔离的单层，有自定义预注意力归一化）
        # so publish the attention input ourselves.  # 因此我们自己发布注意力输入
        get_attn_tp_context().set_attn_inputs(  # 设置注意力输入
            AttentionInputs(  # 注意力输入对象
                attn_input,  # 注意力输入张量
                forward_batch,  # 前向批次
                self.self_attn.prepare_qkv_latent,  # QKV潜在准备函数
            )
        )

        attn_out = self.self_attn(  # 执行MLA注意力
            positions=positions,  # 位置编码
            hidden_states=attn_input,  # 注意力输入
            forward_batch=forward_batch,  # 前向批次
            zero_allocator=zero_allocator,  # 零分配器
        )
        if isinstance(attn_out, tuple):  # 如果输出是元组
            attn_out = attn_out[0]  # 取第一个元素

        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)  # 注意力后归一化
        hidden_states = self.mlp(hidden_states)  # 通过MLP
        return hidden_states, residual  # 返回隐藏状态和残差


class Eagle3MLAModel(nn.Module):
    """EAGLE3 MLA模型主体，包含嵌入层、fc投影、单层解码器和归一化。"""
    def __init__(
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.vocab_size = config.vocab_size  # 保存词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 词表并行嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏大小
            prefix=add_prefix("embed_tokens", prefix),  # 参数前缀
        )

        target_hidden_size = (  # 目标隐藏大小
            getattr(config, "target_hidden_size", None) or config.hidden_size  # 从配置获取或使用默认值
        )
        self.num_aux_hidden_states = _get_eagle_aux_layer_count(config)  # 辅助隐藏状态层数
        self.fc = nn.Linear(  # fc投影层
            target_hidden_size * self.num_aux_hidden_states,  # 输入维度（辅助状态拼接）
            config.hidden_size,  # 输出维度
            bias=getattr(config, "bias", False),  # 偏置
        )

        # Per-aux RMSNorm before fc; enabled via `fc_norm` or legacy  # fc前的每个辅助RMSNorm；通过fc_norm或遗留
        # `use_aux_norm` flag. Matches the eagle3.1 layout.  # use_aux_norm标志启用。匹配eagle3.1布局
        use_fc_norm = getattr(config, "fc_norm", None) or getattr(  # 检查是否启用fc_norm
            config, "use_aux_norm", False  # 或use_aux_norm
        )
        if use_fc_norm:  # 如果启用fc归一化
            self.fc_norm = nn.ModuleList(  # fc归一化列表
                [
                    RMSNorm(target_hidden_size, eps=config.rms_norm_eps)  # 每个辅助层的RMSNorm
                    for _ in range(self.num_aux_hidden_states)  # 遍历辅助层数
                ]
            )
        else:  # 否则
            self.fc_norm = None  # 不使用fc归一化

        if config.num_hidden_layers != 1:  # 如果层数不为1
            raise ValueError("EAGLE3 currently only supports 1 layer")  # 报错
        self.midlayer = Eagle3MLADecoderLayer(  # 创建单层EAGLE3解码器
            config,  # 配置
            layer_id=0,  # 层ID为0
            quant_config=quant_config,  # 量化配置
            prefix=prefix,  # 参数前缀
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化层
        # Draft decode captures pre-norm hidden by default; eagle3.1 opts for  # 草稿解码默认捕获预归一化隐藏；eagle3.1选择
        # post-norm via `norm_output: true`.  # 通过norm_output:true使用后归一化
        self.norm_output = getattr(config, "norm_output", False)  # 是否输出后归一化

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """前向传播：EAGLE3模型主体推理。"""
        if input_embeds is None:  # 如果没有输入嵌入
            # MM positions in input_ids hold MM_PAD_SHIFT_VALUE+hash sentinels (far above  # 多模态位置持有远超词表大小的标记
            # vocab_size). Use target-produced mm_input_embeds for these positions and  # 使用目标生成的mm_input_embeds
            # only call embed_tokens on the appended next-token to avoid embed OOB.  # 仅对追加的下一个token调用embed_tokens
            embeds = forward_batch.mm_input_embeds  # 获取多模态输入嵌入
            if (  # 如果是扩展模式且包含多模态输入
                forward_batch.forward_mode.is_extend()
                and forward_batch.contains_mm_inputs()
                and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                assert embeds is not None  # 确保嵌入不为空
                embeds = torch.cat(  # 拼接嵌入
                    [embeds[:-1], self.embed_tokens(input_ids[-1].unsqueeze(0))]  # 保留前面的嵌入，最后一个token重新嵌入
                )
            if embeds is None:  # 如果仍然为空
                embeds = self.embed_tokens(input_ids)  # 通过嵌入层获取
        else:  # 如果有输入嵌入
            embeds = input_embeds  # 直接使用

        hidden_states = forward_batch.spec_info.hidden_states  # 获取投机解码的隐藏状态
        if hidden_states.shape[-1] != embeds.shape[-1]:  # 如果维度不匹配
            if self.fc_norm is not None:  # 如果有fc归一化
                chunks = hidden_states.chunk(self.num_aux_hidden_states, dim=-1)  # 分块
                hidden_states = torch.cat(  # 对每块归一化后拼接
                    [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                    dim=-1,  # 在最后一维拼接
                )
            hidden_states = self.fc(hidden_states)  # 通过fc投影

        if hidden_states.shape[0] == 0:  # 如果没有token
            return hidden_states, [hidden_states]  # 直接返回

        zero_allocator = BumpAllocator(  # 创建零分配器
            buffer_size=2,  # 缓冲区大小
            dtype=torch.float32,  # 数据类型
            device=embeds.device,  # 设备
        )
        hidden_states, residual = self.midlayer(  # 通过EAGLE3解码器层
            positions, embeds, hidden_states, forward_batch, zero_allocator  # 传入参数
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(  # 最终归一化
            hidden_states, residual  # 传入隐藏状态和残差
        )
        aux = hidden_states_to_logits if self.norm_output else hidden_states_to_aux  # 选择辅助输出
        return hidden_states_to_logits, [aux]  # 返回logits隐藏状态和辅助隐藏状态


class Eagle3DeepseekV2ForCausalLM(nn.Module):
    """EAGLE3草稿模型，使用DeepSeek-V2 MLA注意力架构。"""
    """EAGLE3 draft model architecture with DeepSeek-V2 MLA attention.  # EAGLE3草稿模型架构

    Used by checkpoints like ``kimi-k2.5-eagle3-mla`` that pair  # 用于如kimi-k2.5-eagle3-mla等检查点
    an EAGLE3 layout with multi-latent attention so the draft KV cache shape  # 将EAGLE3布局与MLA配对
    matches the target's MLA cache.  # 使草稿KV缓存形状与目标MLA缓存匹配
    """

    def __init__(
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        # Match deepseek_nextn behavior: modelopt_fp4 is target-only and the  # 匹配deepseek_nextn行为：modelopt_fp4仅用于目标
        # bf16 draft must not inherit the FP4 quant method.  # bf16草稿不能继承FP4量化方法
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":  # 如果是modelopt_fp4量化
            logger.warning(  # 记录警告
                "Overriding Eagle3DeepseekV2ForCausalLM quant config for "  # 覆盖Eagle3量化配置
                "modelopt_fp4 target; draft weights are bf16."  # modelopt_fp4目标；草稿权重为bf16
            )
            quant_config = None  # 重置量化配置
        self.quant_config = quant_config  # 保存量化配置
        self.pp_group = get_pp_group()  # 获取流水线并行组

        self.model = Eagle3MLAModel(  # 创建EAGLE3 MLA模型
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)  # 配置和前缀
        )

        # llama_eagle3 sets a load-from-target flag when draft_vocab_size is  # llama_eagle3在draft_vocab_size缺失时设置从目标加载标志
        # missing. This checkpoint declares its own draft head, so keep ours.  # 此检查点声明了自己的草稿头，保留我们的
        self.load_lm_head_from_target = False  # 不从目标加载语言模型头
        draft_vocab_size = getattr(config, "draft_vocab_size", None)  # 获取草稿词表大小
        if config.tie_word_embeddings:  # 如果共享词嵌入
            self.lm_head = self.model.embed_tokens  # 语言模型头共享嵌入层
        else:  # 否则
            if draft_vocab_size is None:  # 如果草稿词表大小未指定
                self.load_lm_head_from_target = True  # 标记从目标加载
                draft_vocab_size = config.vocab_size  # 使用目标词表大小
                config.draft_vocab_size = draft_vocab_size  # 更新配置
            self.lm_head = ParallelLMHead(  # 创建并行语言模型头
                draft_vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏大小
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("lm_head", prefix),  # 参数前缀
            )

        # Logits processor sees the draft vocab.  # logits处理器使用草稿词表
        config_for_logits = copy.deepcopy(config)  # 深拷贝配置
        config_for_logits.vocab_size = draft_vocab_size or config.vocab_size  # 设置词表大小
        self.logits_processor = LogitsProcessor(config_for_logits)  # 创建logits处理器

        self.capture_aux_hidden_states = True  # 捕获辅助隐藏状态
        self.hot_token_id = None  # 热门token ID

    @torch.no_grad()  # 禁用梯度计算
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ) -> torch.Tensor:
        """前向传播：EAGLE3草稿模型推理。"""
        with get_attn_tp_context().maybe_input_scattered(forward_batch):  # 可能在注意力TP上下文中分散输入
            hidden_states = self.model(  # 通过模型主体
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors  # 传入参数
            )
        aux_hidden_states = None  # 辅助隐藏状态
        if isinstance(hidden_states, tuple):  # 如果输出是元组
            hidden_states, aux_hidden_states = hidden_states  # 拆分
        return self.logits_processor(  # 返回logits处理结果
            input_ids,  # 输入ID
            hidden_states,  # 隐藏状态
            self.lm_head,  # 语言模型头
            forward_batch,  # 前向批次
            aux_hidden_states,  # 辅助隐藏状态
        )

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层。"""
        return self.model.embed_tokens  # 返回嵌入层

    def get_embed_and_head(self):
        """获取嵌入权重和语言模型头权重。"""
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入和头权重

    def set_embed(self, embed: torch.Tensor) -> None:
        """设置嵌入权重（如果草稿和目标隐藏大小不同则跳过）。"""
        # If draft hidden size != target hidden size, embeddings can't be shared.  # 如果草稿和目标隐藏大小不同，嵌入不能共享
        if (  # 检查是否可以共享
            hasattr(self.config, "target_hidden_size")  # 有目标隐藏大小配置
            and self.config.target_hidden_size is not None  # 配置不为空
            and self.config.target_hidden_size != self.config.hidden_size  # 大小不匹配
        ):
            return  # 跳过
        del self.model.embed_tokens.weight  # 删除旧权重
        self.model.embed_tokens.weight = embed  # 设置新权重
        torch.cuda.empty_cache()  # 清空GPU缓存
        torch.cuda.synchronize()  # 同步CUDA

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        """同时设置嵌入权重和语言模型头权重。"""
        del self.model.embed_tokens.weight  # 删除旧嵌入权重
        del self.lm_head.weight  # 删除旧头权重
        self.model.embed_tokens.weight = embed  # 设置新嵌入权重
        self.lm_head.weight = head  # 设置新头权重
        torch.cuda.empty_cache()  # 清空GPU缓存
        torch.cuda.synchronize()  # 同步CUDA

    def get_hot_token_id(self):
        """获取热门token ID（用于草稿-目标词表映射）。"""
        return self.hot_token_id  # 返回热门token ID

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """加载模型权重，支持融合QKV投影和堆叠参数。"""
        params_dict = dict(self.named_parameters())  # 获取参数字典
        stacked_params_mapping = [  # 堆叠参数映射
            (".gate_up_proj", ".gate_proj", 0),  # gate_up投影中的gate
            (".gate_up_proj", ".up_proj", 1),  # gate_up投影中的up
        ]
        cached_a_proj: dict[str, torch.Tensor] = {}  # 缓存的a投影权重

        for name, loaded_weight in weights:  # 遍历权重
            if name == "d2t" or name.endswith(".d2t"):  # 如果是d2t权重
                # d2t stores diffs between draft id and target id; absent in  # d2t存储草稿ID和目标ID的差值
                # checkpoints whose draft_vocab_size equals vocab_size.  # 在draft_vocab_size等于vocab_size的检查点中不存在
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])  # 计算热门token ID
                continue  # 继续
            if name == "t2d" or name.endswith(".t2d"):  # 如果是t2d权重
                continue  # 跳过

            # Map checkpoint layout (layers.0.X, embed_tokens.X, fc, norm) to the  # 映射检查点布局到内部布局
            # internal layout (model.midlayer.X, model.embed_tokens.X, model.fc,  # 内部布局使用model.midlayer等前缀
            # model.norm). lm_head stays at the top level.  # lm_head保持顶层
            mapped_name = re.sub(r"^layers\.0\.", "midlayer.", name)  # 替换层命名
            if not mapped_name.startswith("lm_head."):  # 如果不是lm_head
                mapped_name = f"model.{mapped_name}"  # 添加model前缀

            handled = False  # 是否已处理
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in mapped_name:  # 如果权重名不匹配
                    continue  # 跳过
                target_name = mapped_name.replace(weight_name, param_name)  # 替换权重名
                if target_name not in params_dict:  # 如果目标参数不存在
                    continue  # 跳过
                param = params_dict[target_name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                handled = True  # 标记已处理
                break  # 跳出循环
            if handled:  # 如果已处理
                continue  # 继续下一个权重

            if "q_a_proj" in mapped_name or "kv_a_proj_with_mqa" in mapped_name:  # 如果是Q或KV的a投影
                cached_a_proj[mapped_name] = loaded_weight  # 缓存权重
                q_name = (  # Q投影名
                    mapped_name
                    if "q_a_proj" in mapped_name  # 如果是Q投影
                    else mapped_name.replace("kv_a_proj_with_mqa", "q_a_proj")  # 替换为Q投影名
                )
                kv_name = (  # KV投影名
                    mapped_name
                    if "kv_a_proj_with_mqa" in mapped_name  # 如果是KV投影
                    else mapped_name.replace("q_a_proj", "kv_a_proj_with_mqa")  # 替换为KV投影名
                )
                if q_name in cached_a_proj and kv_name in cached_a_proj:  # 如果两者都缓存了
                    fused_weight = torch.cat(  # 拼接为融合权重
                        [cached_a_proj[q_name], cached_a_proj[kv_name]], dim=0  # 在输出维度拼接
                    )
                    fused_name = q_name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")  # 替换为融合名
                    if fused_name in params_dict:  # 如果融合参数存在
                        param = params_dict[fused_name]  # 获取参数
                        weight_loader = getattr(  # 获取权重加载器
                            param, "weight_loader", default_weight_loader  # 默认加载器
                        )
                        weight_loader(param, fused_weight)  # 加载融合权重
                    cached_a_proj.pop(q_name)  # 移除Q缓存
                    cached_a_proj.pop(kv_name)  # 移除KV缓存
                continue  # 继续

            if mapped_name not in params_dict:  # 如果映射名不在参数字典中
                logger.warning("Eagle3 MLA: skipping unexpected weight %s", name)  # 记录警告
                continue  # 跳过
            param = params_dict[mapped_name]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, loaded_weight)  # 加载权重

        self.post_load_weights()  # 后处理权重

    def post_load_weights(self) -> None:
        """权重加载后处理：将kv_b_proj拆分为w_kc和w_vc张量。"""
        """Split kv_b_proj into w_kc / w_vc tensors used by MLA absorb_core.  # 将kv_b_proj拆分为MLA absorb_core使用的w_kc/w_vc

        DeepseekV2 normally does this in DeepseekV2WeightLoaderMixin.post_load_weights;  # DeepseekV2通常在post_load_weights中执行
        we re-implement the bf16 fast-path directly here to keep the eagle3 draft  # 我们在此直接重新实现bf16快速路径
        path independent of the full DeepseekV2 weight loader.  # 以保持eagle3草稿路径独立于完整DeepseekV2权重加载器
        """
        self_attn = self.model.midlayer.self_attn  # 获取MLA注意力模块
        w = self_attn.kv_b_proj.weight  # 获取kv_b_proj权重
        if w.dtype not in (torch.bfloat16, torch.float16, torch.float32):  # 检查数据类型
            raise NotImplementedError(  # 不支持的类型
                f"Eagle3 MLA draft post_load_weights only supports float dtypes, got {w.dtype}"
            )
        w_kc, w_vc = w.unflatten(  # 反展平并拆分
            0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
        ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)  # 拆分为kc和vc
        self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)  # 设置w_kc
        self_attn.w_vc = w_vc.contiguous().transpose(1, 2)  # 设置w_vc


EntryClass = [Eagle3DeepseekV2ForCausalLM]  # 入口类列表
