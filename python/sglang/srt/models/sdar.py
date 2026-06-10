# SDAR 模型推理实现文件
# 本文件实现了SDAR（块扩散/dLLM风格）因果语言模型
# 使用ENCODER_ONLY注意力类型支持非因果/块掩码
# 包含MLP、注意力、解码块和模型权重加载等核心组件

# coding=utf-8
"""
SGLang SDARModelLM (block diffusion / dLLM-style forward).
"""

import logging  # 导入日志模块
from typing import Iterable, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 导入分布式
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU激活
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes  # 导入层通信器
from sglang.srt.layers.dp_attention import (  # 导入DP注意力
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化
from sglang.srt.layers.linear import (  # 导入线性层
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入逻辑处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id  # 导入工具
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次
from sglang.srt.model_loader.weight_utils import (  # 导入权重工具
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.utils import (  # 导入模型工具
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.server_args import get_global_server_args  # 导入服务器参数
from sglang.srt.utils import add_prefix, is_cuda, make_layers  # 导入工具函数

logger = logging.getLogger(__name__)  # 创建日志记录器
_is_cuda = is_cuda()  # 是否CUDA


class SDARMLP(nn.Module):
    """SDAR MLP模块，使用SiLU门控激活"""
    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config=None,  # 量化配置
        reduce_results: bool = True,  # 是否归约结果
        prefix: str = "",  # 前缀
    ):
        """初始化MLP，配置门控上投影和下投影"""
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor, use_reduce_scatter: bool = False):
        """MLP前向传播：门控上投影、SiLU激活、下投影"""
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(
            hidden_states, skip_all_reduce=use_reduce_scatter
        )
        return hidden_states


class SDARAttention(nn.Module):
    """SDAR注意力模块，使用ENCODER_ONLY类型支持非因果注意力"""
    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config=None,  # 量化配置
        reduce_results: bool = True,  # 是否归约结果
        prefix: str = "",  # 前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 替代CUDA流
    ):
        """初始化SDAR注意力，配置QKV投影、QK归一化、RoPE和注意力层"""
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)

        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            reduce_results=reduce_results,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # Q归一化
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # K归一化

        rope_theta = getattr(config, "rope_theta", 10000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_pos = getattr(config, "max_position_embeddings", 32768)
        self.rotary_dim = self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_pos,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # RadixAttention: ENCODER_ONLY lets ForwardBatch provide non-causal / block masks (dLLM)
        # NOTE: this is the key change vs AR Llama-style DECODER self-attn.
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,  # 编码器注意力类型（非因果）
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream

    def forward_prepare_native(self, positions, hidden_states, forward_batch):
        """原生注意力准备：QKV投影、QK归一化、RoPE"""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=self.attn,
                    forward_batch=forward_batch,
                )
                if enable_fused_set_kv_buffer(forward_batch)
                else None
            ),
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,  # 位置
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ):
        """SDAR注意力前向传播"""
        if get_global_server_args().rl_on_policy_target is not None:  # RL训练模式
            hidden_states = hidden_states.bfloat16()

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=self.attn,
                    forward_batch=forward_batch,
                )
                if enable_fused_set_kv_buffer(forward_batch)
                else None
            ),
        )

        if get_global_server_args().rl_on_policy_target is not None:  # RL模式转换精度
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        context_layer = self.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=not enable_fused_set_kv_buffer(forward_batch),
        )
        out, _ = self.o_proj(context_layer)
        return out


class SDARBlock(nn.Module):
    """SDAR解码块，包含注意力和MLP"""
    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 替代CUDA流
    ):
        """初始化SDAR块，配置归一化、注意力和MLP"""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.input_layernorm = RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )

        self.self_attn = SDARAttention(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )

        self.mlp = SDARMLP(
            config=config,
            quant_config=quant_config,
            reduce_results=True,
            prefix=add_prefix("mlp", prefix),
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SDAR块前向传播：注意力+MLP"""
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
        )

        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.mlp(hidden_states, use_reduce_scatter=use_reduce_scatter)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class SDARModel(nn.Module):
    """SDAR模型，包含嵌入层、解码块和最终归一化"""
    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 替代CUDA流
    ):
        """初始化SDAR模型"""
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SDARBlock(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            norm_kwargs = (
                dict(
                    weight_dtype=torch.float32,
                    cast_x_before_out_mul=True,
                    override_orig_dtype=torch.float32,
                    fp32_residual=True,
                )
                if get_global_server_args().rl_on_policy_target is not None
                else {}
            )
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps, **norm_kwargs)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量
    ) -> Union[torch.Tensor, PPProxyTensors]:
        """SDAR模型前向传播"""
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors.get("residual", None)

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                hidden_states, residual = self.norm(hidden_states, residual)
            return hidden_states


class SDARForCausalLM(nn.Module):
    """SDAR因果语言模型，使用块扩散/dLLM风格前向"""
    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        """初始化SDAR因果语言模型"""
        super().__init__()
        self.pp_group = get_pp_group()
        assert self.pp_group.world_size == 1, (
            f"SDARMoeForCausalLM does not support pipeline parallel (pp_size={self.pp_group.world_size}). "
            "Please set pp_size=1."
        )

        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream() if _is_cuda else None

        self.model = SDARModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", ""),
            alt_stream=alt_stream,
        )

        if self.pp_group.is_last_rank:
            tp_size = get_tensor_model_parallel_world_size()
            if (
                self.pp_group.world_size == 1
                and config.tie_word_embeddings
                and tp_size == 1
            ):
                self.lm_head = self.model.embed_tokens  # 权重共享
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config, return_full_logits=True)

    @property
    def start_layer(self):
        """起始层"""
        return self.model.start_layer

    @property
    def end_layer(self):
        """结束层"""
        return self.model.end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量
    ) -> torch.Tensor:
        """SDAR因果语言模型前向传播"""
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重"""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not name.startswith("model.") and (
                name.startswith("layers.")
                or name.startswith("embed_tokens.")
                or name.startswith("norm.")
            ):
                name = add_prefix(name, "model")

            if name == "model.embed_tokens.weight":
                if self.pp_group.is_last_rank and self.config.tie_word_embeddings:
                    param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")


EntryClass = SDARForCausalLM  # 模型注册入口类
