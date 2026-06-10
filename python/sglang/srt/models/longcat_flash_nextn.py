# Longcat-Flash NextN模型推理实现文件
# 本文件实现了Longcat-Flash模型的NextN（多token预测）变体
# 继承自LongcatFlashForCausalLM，添加了MTP（Multi-Token Prediction）解码器
# 包含密集解码器层、NextN模型主体和NextN因果语言模型
# 支持权重名重映射、QKV融合、fp8/int8量化和DeepGEMM加速

# Apache License, Version 2.0:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import concurrent.futures  # 并发执行
import logging  # 日志模块
from typing import Iterable, Optional, Tuple  # 类型提示

import torch  # PyTorch核心库
from torch import nn  # 神经网络模块

from sglang.srt.configs import LongcatFlashConfig  # Longcat-Flash配置类
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 专家分布记录器
from sglang.srt.layers import deep_gemm_wrapper  # DeepGEMM包装器
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes  # 层通信器
from sglang.srt.layers.dp_attention import (  # 数据并行注意力相关
    get_attention_tp_rank,  # 获取注意力TP秩
    get_attention_tp_size,  # 获取注意力TP大小
    is_dp_attention_enabled,  # 是否启用DP注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # 均方根归一化层
from sglang.srt.layers.linear import ReplicatedLinear  # 复制线性层
from sglang.srt.layers.logits_processor import LogitsProcessor  # logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置基类
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz  # 是否为FP8 FNUZ格式
from sglang.srt.layers.quantization.fp8_utils import (  # FP8量化工具
    block_quant_dequant,  # 块量化反量化
    block_quant_to_tensor_quant,  # 块量化到张量量化
    channel_quant_to_tensor_quant,  # 通道量化到张量量化
    normalize_e4m3fn_to_e4m3fnuz,  # E4M3FN到E4M3FNUZ归一化
    requant_weight_ue8m0_inplace,  # 就地重量化权重为UE8M0
)
from sglang.srt.layers.quantization.int8_utils import (  # INT8量化工具
    block_dequant as int8_block_dequant,  # INT8块反量化
)
from sglang.srt.layers.vocab_parallel_embedding import (  # 词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 前向批次信息
from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0  # DeepGEMM重量化判断
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 默认权重加载器
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA  # DeepSeek V2 MLA注意力
from sglang.srt.models.longcat_flash import LongcatFlashForCausalLM, LongcatFlashMLP  # Longcat-Flash模型和MLP
from sglang.srt.utils import (  # 工具函数
    BumpAllocator,  # 凸包分配器
    add_prefix,
    bind_or_assign,  # 绑定或赋值
    cpu_has_amx_support,  # CPU是否支持AMX
    get_bool_env_var,  # 获取布尔环境变量
    get_device_sm,  # 获取设备计算能力
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
)

_is_hip = is_hip()  # 是否为HIP设备
_is_cuda = is_cuda()  # 是否为CUDA设备
_is_npu = is_npu()  # 是否为NPU设备
_is_fp8_fnuz = is_fp8_fnuz()  # 是否为FP8 FNUZ格式
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITer
_is_cpu_amx_available = cpu_has_amx_support()  # CPU AMX是否可用
_is_cpu = is_cpu()  # 是否为CPU设备
_device_sm = get_device_sm()  # 设备计算能力版本

if _is_cuda:  # CUDA平台
    from sgl_kernel import awq_dequantize  # AWQ反量化
elif _is_cpu and _is_cpu_amx_available:  # CPU AMX平台
    pass
elif _is_hip:  # HIP平台
    from sglang.srt.layers.quantization.awq.awq_triton import (
        awq_dequantize_triton as awq_dequantize,  # AWQ Triton反量化
    )
else:  # 其他平台
    pass


logger = logging.getLogger(__name__)  # 获取当前模块日志器


class LongcatFlashDenseDecoderLayer(nn.Module):
    """Longcat-Flash密集解码器层（用于MTP），包含单个MLA注意力和MLP"""

    def __init__(
        self,
        config: LongcatFlashConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.layer_id = layer_id  # 保存层ID
        self.alt_stream = alt_stream  # 保存备用流

        self.self_attn = DeepseekV2AttentionMLA(  # MLA注意力层
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=config.rope_parameters["rope_theta"],
            rope_scaling=None,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            reduce_results=False,  # 不自动归约
            prefix=add_prefix(f"self_attn", prefix),
            alt_stream=self.alt_stream,
        )

        self.mlp = LongcatFlashMLP(  # 密集MLP
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix(f"mlps", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.attn_tp_size = get_attention_tp_size()  # 注意力TP大小
        self.attn_tp_rank = get_attention_tp_rank()  # 注意力TP秩
        self.layer_scatter_modes = LayerScatterModes.init_new(  # 层散射模式
            layer_id=self.layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,  # 非稀疏
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(  # 层通信器
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引
        hidden_states: torch.Tensor,  # 输入隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差连接
        zero_allocator: BumpAllocator,  # 零分配器
    ) -> torch.Tensor:
        """密集解码器层前向传播：注意力准备 -> 注意力 -> MLP准备 -> MLP -> 后处理"""
        hidden_states, residual = self.layer_communicator.prepare_attn(  # 准备注意力输入
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:  # 非空输入时计算注意力
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(  # 准备MLP输入
            hidden_states, residual, forward_batch
        )
        hidden_states = self.mlp(hidden_states)  # MLP计算
        hidden_states, residual = self.layer_communicator.postprocess_layer(  # 后处理
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual  # 返回隐藏状态和残差


class LongcatFlashModelNextN(nn.Module):
    """Longcat-Flash NextN模型主体，用于多token预测"""

    def __init__(
        self,
        config: LongcatFlashConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.vocab_size = config.vocab_size  # 词表大小
        self.alt_stream = torch.cuda.Stream()  # 备用CUDA流

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,
            config.hidden_size,
            use_attn_tp_group=is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 嵌入归一化
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 隐藏状态归一化

        self.eh_proj = ReplicatedLinear(  # 嵌入-隐藏投影
            2 * config.hidden_size,  # 输入为嵌入和隐藏的拼接
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("eh_proj", ""),
        )
        self.decoder = LongcatFlashDenseDecoderLayer(  # 密集解码器层
            config, 0, quant_config=quant_config, alt_stream=self.alt_stream
        )

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化

    def get_input_embeddings(self) -> torch.Tensor:
        """获取输入嵌入层"""
        return self.embed_tokens  # 返回嵌入层

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        """NextN模型前向传播：嵌入 -> 嵌入隐藏投影 -> 解码器 -> 归一化"""
        total_num_layers = 1  # 单层解码器
        device = input_embeds.device if input_embeds is not None else input_ids.device  # 设备
        zero_allocator = BumpAllocator(  # 零分配器
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),  # 缓冲区大小
            dtype=torch.float32,
            device=device,
        )
        if input_embeds is None:  # 无预计算嵌入
            hidden_states = self.embed_tokens(input_ids)
        else:  # 使用预计算嵌入
            hidden_states = input_embeds

        if hidden_states.shape[0] > 0:  # 非空输入
            hidden_states, _ = self.eh_proj(  # 嵌入-隐藏投影
                torch.cat(
                    (
                        self.enorm(hidden_states),  # 归一化嵌入
                        self.hnorm(forward_batch.spec_info.hidden_states),  # 归一化隐藏状态
                    ),
                    dim=-1,  # 沿最后一维拼接
                )
            )

        residual = None  # 初始无残差
        with get_global_expert_distribution_recorder().disable_this_region():  # 禁用专家分布记录
            hidden_states, residual = self.decoder(  # 解码器前向
                positions, hidden_states, forward_batch, residual, zero_allocator
            )

        if not forward_batch.forward_mode.is_idle():  # 非空闲模式
            if residual is not None:  # 有残差时融合归一化
                hidden_states, _ = self.final_layernorm(hidden_states, residual)
            else:  # 无残差时直接归一化
                hidden_states = self.final_layernorm(hidden_states)
        return hidden_states  # 返回隐藏状态


class LongcatFlashForCausalLMNextN(LongcatFlashForCausalLM):
    """Longcat-Flash NextN因果语言模型，继承自LongcatFlashForCausalLM，添加MTP支持"""

    def __init__(
        self,
        config: LongcatFlashConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
    ) -> None:
        nn.Module.__init__(self)  # 直接调用nn.Module初始化
        self.config = config  # 保存配置
        self.quant_config = (  # 量化配置
            None
            if "mtp" in getattr(config, "disable_quant_module", [])  # MTP禁用量化时
            else quant_config
        )
        self.model = LongcatFlashModelNextN(config, self.quant_config)  # NextN模型主体
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """NextN因果语言模型前向传播：模型主体 -> logits处理"""
        hidden_states = self.model(input_ids, positions, forward_batch)  # 模型前向
        return self.logits_processor(  # 返回logits
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def post_load_weights(self):
        """权重加载后处理：分解kv_b_proj权重、处理量化缩放"""
        self_attn = self.model.decoder.self_attn  # 获取注意力层
        if hasattr(self_attn.kv_b_proj, "qweight"):  # AWQ量化
            # AWQ compatible
            if _is_cuda or _is_hip:  # CUDA或HIP平台
                w = awq_dequantize(
                    self_attn.kv_b_proj.qweight,
                    self_attn.kv_b_proj.scales,
                    self_attn.kv_b_proj.qzeros,
                ).T
            else:  # 其他平台
                w = awq_dequantize(
                    self_attn.kv_b_proj.qweight,
                    self_attn.kv_b_proj.scales,
                    self_attn.kv_b_proj.qzeros,
                    0,
                    0,
                    0,
                ).T
        else:  # 非量化权重
            w = self_attn.kv_b_proj.weight
        use_deep_gemm_bmm = False  # 是否使用DeepGEMM BMM
        if w.dtype in (  # FP8权重处理
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            if (  # 块量化
                hasattr(self.quant_config, "weight_block_size")
                and self.quant_config.weight_block_size is not None
            ):
                weight_block_size = self.quant_config.weight_block_size
                assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                if _is_fp8_fnuz:  # FNUZ格式归一化
                    weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                        weight=w,
                        weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                        input_scale=None,
                    )
                else:  # 标准FP8格式
                    weight = w
                    weight_scale = self_attn.kv_b_proj.weight_scale_inv
                if (  # CUDA + 128x128块大小
                    _is_cuda
                    and weight_block_size[0] == 128
                    and weight_block_size[1] == 128
                ):
                    if (  # DeepGEMM BMM
                        deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                        and not deep_gemm_wrapper.DEEPGEMM_BLACKWELL
                        and get_bool_env_var("SGL_USE_DEEPGEMM_BMM", "false")
                    ):
                        block_scale = weight_scale  # 块缩放
                        use_deep_gemm_bmm = True
                    else:  # 块量化反量化
                        w = block_quant_dequant(
                            weight,
                            weight_scale,
                            weight_block_size,
                            torch.bfloat16,
                        )
                else:  # 其他块大小
                    w, scale = block_quant_to_tensor_quant(
                        weight, weight_scale, weight_block_size
                    )
                    self_attn.w_scale = scale
            else:  # 通道量化
                if _is_fp8_fnuz:  # FNUZ格式归一化
                    weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                        weight=w,
                        weight_scale=self_attn.kv_b_proj.weight_scale,
                        input_scale=None,
                    )
                else:  # 标准FP8格式
                    weight = w
                    weight_scale = self_attn.kv_b_proj.weight_scale
                w, scale = channel_quant_to_tensor_quant(weight, weight_scale)  # 通道量化到张量
                self_attn.w_scale = scale
        if w.dtype == torch.int8:  # INT8权重处理
            if hasattr(self.quant_config, "weight_block_size"):  # 块级INT8
                # block-wise int8 need it
                weight_block_size = self.quant_config.weight_block_size
                if weight_block_size is not None:
                    assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                    weight = w
                    weight_scale = self_attn.kv_b_proj.weight_scale_inv
                    w = int8_block_dequant(weight, weight_scale, weight_block_size).to(  # INT8块反量化
                        torch.bfloat16
                    )
            else:  # 通道级INT8
                # channel-wise int8 need it
                w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(
                    torch.bfloat16
                )
        w_kc, w_vc = w.unflatten(  # 分离k和v分量
            0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
        ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
        if not use_deep_gemm_bmm:  # 非DeepGEMM BMM模式
            self_attn.w_kc = bind_or_assign(  # 绑定k权重
                self_attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            )
            self_attn.w_vc = bind_or_assign(  # 绑定v权重
                self_attn.w_vc, w_vc.contiguous().transpose(1, 2)
            )
            if (  # 绑定权重缩放
                hasattr(self_attn.kv_b_proj, "weight_scale")
                and self_attn.w_scale is None
            ):
                self_attn.w_scale = bind_or_assign(
                    self_attn.w_scale, self_attn.kv_b_proj.weight_scale
                )
                if _is_hip:  # HIP平台缩放2倍
                    self_attn.w_scale *= 2.0
        else:  # DeepGEMM BMM模式
            num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[1]  # K分块数
            num_tiles_n = self_attn.v_head_dim // weight_block_size[0]  # N分块数
            ws_kc, ws_vc = block_scale.unflatten(  # 分离k和v缩放
                0, (-1, (num_tiles_k + num_tiles_n))
            ).split([num_tiles_k, num_tiles_n], dim=1)
            self_attn.w_scale_k = bind_or_assign(  # 绑定k缩放
                self_attn.w_scale_k, ws_kc.transpose(1, 2).contiguous()
            )
            self_attn.w_scale_v = bind_or_assign(  # 绑定v缩放
                self_attn.w_scale_v, ws_vc.contiguous()
            )
            self_attn.w_kc = bind_or_assign(  # 绑定k权重
                self_attn.w_kc, w_kc.transpose(1, 2).contiguous()
            )
            self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc.contiguous())  # 绑定v权重
            self_attn.use_deep_gemm_bmm = True  # 启用DeepGEMM BMM

        if self.config.mla_scale_q_lora:  # 缩放Q LoRA归一化权重
            self_attn.q_a_layernorm.weight.data *= (
                self.config.hidden_size / self.config.q_lora_rank
            ) ** 0.5
        if self.config.mla_scale_kv_lora:  # 缩放KV LoRA归一化权重
            self_attn.kv_a_layernorm.weight.data *= (
                self.config.hidden_size / self.config.kv_lora_rank
            ) ** 0.5

        if should_deepgemm_weight_requant_ue8m0(  # 需要DeepGEMM权重重量化
            weight_block_size=getattr(self.quant_config, "weight_block_size", None)
        ):
            self._weight_requant_ue8m0()  # 执行重量化

    def _weight_requant_ue8m0(self):
        """将权重重量化为UE8M0格式以配合DeepGEMM"""
        weight_block_size = self.quant_config.weight_block_size  # 权重块大小
        layer = self.model.decoder  # 解码器层
        self_attn = layer.self_attn  # 注意力层
        module_list = [  # 需要重量化的模块列表
            self_attn.kv_b_proj,
            self_attn.o_proj,
        ]

        if self.config.q_lora_rank is not None:  # 有LoRA时
            module_list.append(self_attn.fused_qkv_a_proj_with_mqa)
            module_list.append(self_attn.q_b_proj)
        else:  # 无LoRA时
            module_list.append(self_attn.kv_a_proj_with_mqa)
            module_list.append(self_attn.q_proj)

        for module in module_list:  # 重量化注意力模块
            if hasattr(module, "weight_scale_inv"):
                requant_weight_ue8m0_inplace(
                    module.weight, module.weight_scale_inv, weight_block_size
                )

        mlp = layer.mlps  # MLP模块
        assert isinstance(mlp, LongcatFlashMLP)
        for module in [  # 重量化MLP模块
            mlp.gate_up_proj,
            mlp.down_proj,
        ]:
            if hasattr(module, "weight_scale_inv"):
                requant_weight_ue8m0_inplace(
                    module.weight, module.weight_scale_inv, weight_block_size
                )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载MTP权重，支持权重名重映射和QKV融合"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),  # 门控投影合并
            ("gate_up_proj", "up_proj", 1),  # 上投影合并
        ]

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (  # 是否融合QKV a投影
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None  # 缓存的a投影权重

        nextn_layer_prefix = "model.layers.0"  # NextN层前缀
        nextn_spec_weight_names = [  # NextN特定权重名
            "shared_head.norm",
            "eh_proj",
            "enorm",
            "hnorm",
            "final_layernorm",
        ]

        weight_names_mapping = {  # 权重名映射表
            "model.mtp.embed_tokens.weight": "embed_tokens.weight",
            "model.mtp.layers.0.eh_proj.weight": "eh_proj.weight",
            "model.mtp.layers.0.eh_proj.weight_scale_inv": "eh_proj.weight_scale_inv",
            "model.mtp.layers.0.enorm.m.weight": "enorm.weight",
            "model.mtp.layers.0.hnorm.m.weight": "hnorm.weight",
            "model.mtp.layers.0.input_layernorm.weight": "layers.0.input_layernorm.weight",
            "model.mtp.layers.0.post_attention_layernorm.weight": "layers.0.post_attention_layernorm.weight",
            "model.mtp.layers.0.self_attn.kv_a_layernorm.weight": "layers.0.self_attn.kv_a_layernorm.weight",
            "model.mtp.layers.0.self_attn.kv_a_proj_with_mqa.weight": "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.mtp.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv": "layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
            "model.mtp.layers.0.self_attn.kv_b_proj.weight": "layers.0.self_attn.kv_b_proj.weight",
            "model.mtp.layers.0.self_attn.kv_b_proj.weight_scale_inv": "layers.0.self_attn.kv_b_proj.weight_scale_inv",
            "model.mtp.layers.0.self_attn.o_proj.weight": "layers.0.self_attn.o_proj.weight",
            "model.mtp.layers.0.self_attn.o_proj.weight_scale_inv": "layers.0.self_attn.o_proj.weight_scale_inv",
            "model.mtp.layers.0.self_attn.q_a_layernorm.weight": "layers.0.self_attn.q_a_layernorm.weight",
            "model.mtp.layers.0.self_attn.q_a_proj.weight": "layers.0.self_attn.q_a_proj.weight",
            "model.mtp.layers.0.self_attn.q_a_proj.weight_scale_inv": "layers.0.self_attn.q_a_proj.weight_scale_inv",
            "model.mtp.layers.0.self_attn.q_b_proj.weight": "layers.0.self_attn.q_b_proj.weight",
            "model.mtp.layers.0.self_attn.q_b_proj.weight_scale_inv": "layers.0.self_attn.q_b_proj.weight_scale_inv",
            "model.mtp.layers.0.transformer_layer.mlp.down_proj.weight": "layers.0.mlp.down_proj.weight",
            "model.mtp.layers.0.transformer_layer.mlp.down_proj.weight_scale_inv": "layers.0.mlp.down_proj.weight_scale_inv",
            "model.mtp.layers.0.transformer_layer.mlp.gate_proj.weight": "layers.0.mlp.gate_proj.weight",
            "model.mtp.layers.0.transformer_layer.mlp.gate_proj.weight_scale_inv": "layers.0.mlp.gate_proj.weight_scale_inv",
            "model.mtp.layers.0.transformer_layer.mlp.up_proj.weight": "layers.0.mlp.up_proj.weight",
            "model.mtp.layers.0.transformer_layer.mlp.up_proj.weight_scale_inv": "layers.0.mlp.up_proj.weight_scale_inv",
            "model.mtp.norm.weight": "layers.0.final_layernorm.weight",
        }
        with concurrent.futures.ThreadPoolExecutor() as executor:  # 线程池
            futures = []  # 异步任务列表
            params_dict = dict(self.named_parameters())  # 参数字典
            weight_names = []  # 权重名列表
            for name, loaded_weight in weights:  # 遍历所有权重
                if ".mtp." not in name:  # 仅处理MTP权重
                    continue
                if name in weight_names_mapping:  # 重映射权重名
                    name = weight_names_mapping[name]
                if name.startswith("layers.0"):  # 添加model前缀
                    name = "model." + name
                if (  # NextN特定权重名处理
                    name.startswith("enorm")
                    or name.startswith("hnorm")
                    or name.startswith("eh_proj")
                ):
                    name = nextn_layer_prefix + "." + name  # 添加层前缀
                if not name.startswith(nextn_layer_prefix):  # 跳过非NextN层权重
                    continue

                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:  # 跳过共享头和嵌入
                    continue

                is_decoder = True  # 是否为解码器权重
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:  # 检查NextN特定权重
                    if weight_name in name:
                        name = name.replace(nextn_layer_prefix, "model")  # 替换前缀
                        is_decoder = False  # 标记为非解码器权重
                        break
                # For decoder layer weights
                if is_decoder:  # 解码器权重
                    name = name.replace(nextn_layer_prefix, "model.decoder")  # 替换为解码器前缀

                weight_names.append(name)  # 记录权重名
                if "rotary_emb.inv_freq" in name:  # 跳过旋转频率
                    continue
                for param_name, weight_name, shard_id in stacked_params_mapping:  # 处理堆叠参数
                    # Skip non-stacked layers and experts (experts handled below).
                    if weight_name not in name:  # 名称不包含权重名则跳过
                        continue
                    # We have mlp.experts[0].gate_proj in the checkpoint.
                    # Since we handle the experts below in expert_params_mapping,
                    # we need to skip here BEFORE we update the name, otherwise
                    # name will be updated to mlp.experts[0].gate_up_proj, which
                    # will then be updated below in expert_params_mapping
                    # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                    if ("mlp.experts." in name) and name not in params_dict:  # 跳过专家参数
                        continue
                    name = name.replace(weight_name, param_name)  # 替换为参数名
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ额外偏置
                        continue
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    futures.append(  # 提交加载任务
                        executor.submit(weight_loader, param, loaded_weight, shard_id)
                    )
                    break
                else:  # 处理常规权重
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ额外偏置
                        continue
                    if fuse_qkv_a_proj and (  # QKV a投影融合
                        "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                    ):
                        cached_a_proj[name] = loaded_weight  # 缓存a投影权重
                        q_a_proj_name = (  # q_a投影名
                            name
                            if "q_a_proj" in name
                            else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                        )
                        kv_a_proj_name = (  # kv_a投影名
                            name
                            if "kv_a_proj_with_mqa" in name
                            else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                        )

                        # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                        if (  # 两个a投影都已缓存
                            q_a_proj_name in cached_a_proj
                            and kv_a_proj_name in cached_a_proj
                        ):
                            q_a_proj_weight = cached_a_proj[q_a_proj_name]  # 获取q_a权重
                            kv_a_proj_weight = cached_a_proj[kv_a_proj_name]  # 获取kv_a权重
                            cat_dim = 0  # 拼接维度
                            if self.quant_config is not None and (  # AWQ量化时沿维度1拼接
                                self.quant_config.get_name() == "awq"
                                or self.quant_config.get_name() == "awq_marlin"
                                or self.quant_config.get_name() == "moe_wna16"
                            ):
                                cat_dim = 1
                            fused_weight = torch.cat(  # 拼接融合权重
                                [q_a_proj_weight, kv_a_proj_weight], dim=cat_dim
                            )
                            param_name = (  # 融合参数名
                                name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                                if "q_a_proj" in name
                                else name.replace(
                                    "kv_a_proj_with_mqa",
                                    "fused_qkv_a_proj_with_mqa",
                                )
                            )
                            param = params_dict[param_name]  # 获取参数

                            weight_loader = getattr(  # 获取权重加载器
                                param, "weight_loader", default_weight_loader
                            )
                            futures.append(  # 提交融合权重加载任务
                                executor.submit(weight_loader, param, fused_weight)
                            )
                            cached_a_proj.pop(q_a_proj_name)  # 移除已加载的缓存
                            cached_a_proj.pop(kv_a_proj_name)
                    else:  # 常规权重加载
                        if (  # modelopt的KV缩放重命名
                            "k_scale" in name or "v_scale" in name
                        ) and name not in params_dict:
                            # modelopt attn kv scale is named differently
                            for scale in ["k_scale", "v_scale"]:
                                if scale in name:
                                    name = name.replace(f"{scale[0]}_proj", "attn_mqa")  # 替换名称
                                    break
                        if name not in params_dict:  # 参数不存在
                            # modelopt ckpt contains not needed weights for MTP module:
                            # model.decoder.self_attn.attn_mqa.v_scale and
                            # model.decoder.self_attn.attn_mqa.k_scale
                            logger.warning(f"{name} not found in params_dict.")  # 发出警告
                            continue
                        param = params_dict[name]  # 获取参数
                        weight_loader = getattr(  # 获取权重加载器
                            param, "weight_loader", default_weight_loader
                        )
                        futures.append(  # 提交加载任务
                            executor.submit(weight_loader, param, loaded_weight)
                        )
        self.post_load_weights()  # 权重加载后处理


EntryClass = [LongcatFlashForCausalLMNextN]  # 入口类列表
