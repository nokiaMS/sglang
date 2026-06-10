# 文件说明：DeepSeek NextN推测解码模型实现
# 本文件实现了DeepSeek V3的NextN（多令牌预测）推测解码模型，
# 包含NextN模型主体和因果语言模型包装类，支持DSA/MLA上下文并行。

# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

"""Inference-only DeepSeek NextN Speculative Decoding."""  # 仅推理的DeepSeek NextN推测解码

import logging  # 导入日志模块
import os  # 导入操作系统模块
from contextlib import ExitStack  # 导入上下文管理器
from typing import Iterable, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch
from safetensors.torch import load_file  # 导入安全张量加载器
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.configs.model_config import is_deepseek_dsa  # 导入DeepSeek DSA判断函数
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 导入分布式工具
from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入专家分布记录器
from sglang.srt.layers.attention.dsa.utils import (  # 导入DSA注意力工具
    can_dsa_cp_split,  # 判断DSA是否可以上下文并行分割
    dsa_use_prefill_cp,  # 判断DSA是否使用预填充上下文并行
    is_dsa_enable_prefill_cp,  # 判断DSA是否启用预填充上下文并行
)
from sglang.srt.layers.dp_attention import (  # 导入数据并行注意力工具
    get_attention_cp_rank,  # 获取上下文并行秩
    get_attention_cp_size,  # 获取上下文并行大小
    is_dp_attention_enabled,  # 判断是否启用数据并行注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import ReplicatedLinear  # 导入复制线性层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入逻辑处理器
from sglang.srt.layers.quantization import Fp8Config  # 导入FP8量化配置
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.utils.cp_utils import (  # 导入上下文并行工具
    can_cp_split,  # 判断是否可以上下文并行分割
    cp_all_gather_rerange_output,  # 上下文并行全收集并重排输出
    cp_split_and_rebuild_data,  # 上下文并行分割并重建数据
    cp_split_and_rebuild_position,  # 上下文并行分割并重建位置
    is_mla_prefill_cp_enabled,  # 判断MLA预填充上下文并行是否启用
    mla_use_prefill_cp,  # 判断MLA是否使用预填充上下文并行
    prepare_context_parallel_metadata,  # 准备上下文并行元数据
)
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.models.deepseek_common.utils import enable_nextn_moe_bf16_cast_to_fp8  # 导入NextN MoE BF16转FP8判断
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV3ForCausalLM  # 导入DeepSeek V2解码器层和V3模型
from sglang.srt.models.utils import WeightsMapper  # 导入权重映射器
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import BumpAllocator, add_prefix, is_cuda, is_npu  # 导入工具函数

logger = logging.getLogger(__name__)  # 获取日志记录器


_is_cuda = is_cuda()  # 是否在CUDA上运行
_is_npu = is_npu()  # 是否在NPU上运行


# DeepSeek NextN模型，实现多令牌预测的推测解码
class DeepseekModelNextN(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        if enable_nextn_moe_bf16_cast_to_fp8(quant_config):  # 如果需要将MoE BF16转为FP8
            # refer to real DeepSeek V3 quant config
            # 参考真实的DeepSeek V3量化配置
            moe_quant_config_override = Fp8Config(  # 创建FP8量化配置覆盖
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[128, 128],
            )
        else:  # 不需要覆盖
            moe_quant_config_override = None

        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":  # 如果是modelopt_fp4量化
            logger.warning(
                "Overriding DeepseekV3ForCausalLMNextN quant config for modelopt_fp4 Deepseek model."
            )
            quant_config = None  # 重置量化配置

        self.vocab_size = config.vocab_size  # 词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,
            config.hidden_size,
            use_attn_tp_group=is_dp_attention_enabled(),  # 是否使用注意力TP组
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 嵌入归一化层
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 隐藏状态归一化层

        if quant_config is not None and quant_config.get_name() == "quark":  # 如果是quark量化
            self.eh_proj = ReplicatedLinear(  # 使用复制线性层
                2 * config.hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("eh_proj", prefix),
            )
        else:  # 非quark量化
            self.eh_proj = nn.Linear(  # 使用普通线性层
                2 * config.hidden_size, config.hidden_size, bias=False
            )

        self.rot_weight = None  # 旋转权重初始化
        if _is_npu:  # 如果在NPU上运行
            rot_weight_path = get_global_server_args().model_path + "/rot.safetensors"  # 旋转权重路径
            if os.path.isfile(rot_weight_path):  # 如果权重文件存在
                self.rot_weight = load_file(rot_weight_path)  # 加载旋转权重
                self.rot_weight = self.rot_weight["rot.weight"].npu()  # 转移到NPU

        self.alt_stream = (  # 备用CUDA流
            torch.cuda.Stream()
            if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()  # CUDA或NPU多流
            else None
        )

        layer_name = "decoder"  # 解码器层名
        if _is_npu and (  # NPU上且draft模型与主模型同路径
            get_global_server_args().speculative_draft_model_path
            == get_global_server_args().model_path
        ):
            layer_name = "layers." + str(config.num_hidden_layers)  # 使用层号命名

        self.quant_config = quant_config  # 量化配置
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()  # DSA是否启用预填充上下文并行
        self.mla_enable_prefill_cp = (  # MLA是否启用预填充上下文并行
            is_mla_prefill_cp_enabled() and not is_deepseek_dsa(config)
        )
        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:  # 如果启用上下文并行
            self.cp_size = get_attention_cp_size()
        else:  # 未启用
            self.cp_size = None
        self.decoder = DeepseekV2DecoderLayer(  # 解码器层
            config,
            0,
            quant_config=quant_config,
            moe_quant_config_override=moe_quant_config_override,
            is_nextn=True,  # 标记为NextN层
            prefix=add_prefix(layer_name, prefix),
            alt_stream=self.alt_stream,
            dsa_enable_prefill_cp=self.dsa_enable_prefill_cp,
            mla_enable_prefill_cp=self.mla_enable_prefill_cp,
        )

        self.shared_head = nn.Module()  # 共享头模块
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 共享头归一化

    # NextN模型前向传播
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入令牌ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        exit_stack = ExitStack()  # 创建退出栈
        if (  # NPU上非量化模型但服务器指定了量化
            _is_npu
            and self.quant_config is None
            and get_global_server_args().quantization is not None
        ):
            # ascend mtp unquant
            # Ascend MTP非量化模式
            exit_stack.enter_context(envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True))  # 覆盖BF16分发
            exit_stack.enter_context(
                envs.DEEP_NORMAL_MODE_USE_INT8_QUANT.override(False)  # 禁用INT8量化
            )

        try:
            zero_allocator = BumpAllocator(  # 创建零值内存分配器
                buffer_size=2,
                dtype=torch.float32,
                device=(
                    input_embeds.device  # 使用输入嵌入设备
                    if input_embeds is not None
                    else input_ids.device  # 或输入ID设备
                ),
            )

            if input_embeds is None:  # 如果没有提供输入嵌入
                hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层获取隐藏状态
            else:  # 使用提供的输入嵌入
                hidden_states = input_embeds

            if hidden_states.shape[0] > 0:  # 如果有令牌需要处理
                eh_input = torch.cat(  # 拼接嵌入和隐藏状态
                    (
                        self.enorm(hidden_states),  # 归一化后的当前隐藏状态
                        self.hnorm(  # 归一化后的推测隐藏状态
                            forward_batch.spec_info.hidden_states  # 推测信息中的隐藏状态
                            if self.rot_weight is None
                            else torch.matmul(  # 如果有旋转权重则先旋转
                                forward_batch.spec_info.hidden_states, self.rot_weight
                            )
                        ),
                    ),
                    dim=-1,  # 在最后一个维度拼接
                )
                if isinstance(self.eh_proj, ReplicatedLinear):  # 如果是复制线性层
                    hidden_states, _ = self.eh_proj(eh_input)  # 通过复制线性层
                else:  # 普通线性层
                    hidden_states = self.eh_proj(eh_input)  # 通过普通线性层

            if dsa_use_prefill_cp(  # 如果DSA使用预填充上下文并行
                forward_batch, self.dsa_enable_prefill_cp
            ) or mla_use_prefill_cp(forward_batch, self.mla_enable_prefill_cp):  # 或MLA使用
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)  # 分割并重建数据
                positions = cp_split_and_rebuild_position(forward_batch, positions)  # 分割并重建位置
            residual = None  # 残差初始化
            with get_global_expert_distribution_recorder().disable_this_region():  # 禁用专家分布记录
                hidden_states, residual, topk_indices = self.decoder(  # 通过解码器层
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    zero_allocator,
                )

            if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
                if residual is not None:  # 如果有残差
                    hidden_states, _ = self.shared_head.norm(hidden_states, residual)  # 共享头归一化（含残差）
                else:  # 无残差
                    hidden_states = self.shared_head.norm(hidden_states)  # 共享头归一化

                if dsa_use_prefill_cp(  # 如果DSA使用预填充上下文并行
                    forward_batch, self.dsa_enable_prefill_cp
                ) or mla_use_prefill_cp(forward_batch, self.mla_enable_prefill_cp):  # 或MLA使用
                    # allgather + rerrange
                    # 全收集 + 重排
                    hidden_states = cp_all_gather_rerange_output(  # 上下文并行全收集并重排
                        hidden_states,
                        self.cp_size,
                        forward_batch,
                        torch.cuda.current_stream(),
                    )
        finally:
            exit_stack.close()  # 关闭退出栈

        return hidden_states  # 返回隐藏状态


# DeepSeek V3 NextN因果语言模型，继承自DeepSeekV3ForCausalLM
class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):

    # Support amd/DeepSeek-R1-0528-MXFP4 renaming: model.layers.61*.
    # 支持amd/DeepSeek-R1-0528-MXFP4重命名：model.layers.61*。
    # Ref: HF config.json for amd/DeepSeek-R1-0528-MXFP4
    # 参考：amd/DeepSeek-R1-0528-MXFP4的HF config.json
    # https://huggingface.co/amd/DeepSeek-R1-0528-MXFP4/blob/main/config.json
    hf_to_sglang_mapper = WeightsMapper(  # HuggingFace到SGLang权重名称映射
        orig_to_new_substr={
            "model.layers.61": "model.decoder",  # 将layers.61映射为decoder
        },
    )

    def __init__(
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        nn.Module.__init__(self)  # 调用nn.Module初始化（跳过父类）
        self.config = config  # 保存配置
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        self.quant_config = quant_config  # 量化配置
        # if not set, model load will be broken in DeepseekV3ForCausalLM load_weights()
        # 如果不设置，模型加载将在DeepseekV3ForCausalLM的load_weights()中出错
        self.pp_group = get_pp_group()  # 流水线并行组
        self.determine_num_fused_shared_experts("DeepseekV3ForCausalLMNextN")  # 确定融合共享专家数
        self.use_dsa = is_deepseek_dsa(config)  # 是否使用DSA
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()  # DSA是否启用预填充上下文并行
        self.mla_enable_prefill_cp = is_mla_prefill_cp_enabled() and not self.use_dsa  # MLA是否启用预填充上下文并行
        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:  # 如果启用上下文并行
            self.cp_rank = get_attention_cp_rank()  # 上下文并行秩
            self.cp_size = get_attention_cp_size()  # 上下文并行大小
        else:  # 未启用
            self.cp_rank = None
            self.cp_size = None

        nextn_quant_config = quant_config  # NextN量化配置
        # For quark, if the MTP layer is listed in exclude_layers, set quant_config to None.
        # 对于quark，如果MTP层列在exclude_layers中，设置quant_config为None。
        if nextn_quant_config is not None and nextn_quant_config.get_name() == "quark":  # quark量化检查
            from sglang.srt.layers.quantization.quark.utils import (
                should_ignore_layer,
            )

            ckpt_prefix = f"model.layers.{config.num_hidden_layers}"  # 检查点前缀
            mapped_prefix = self.hf_to_sglang_mapper._map_name(ckpt_prefix)  # 映射后的前缀
            if should_ignore_layer(mapped_prefix, nextn_quant_config.exclude_layers):  # 是否应忽略该层
                nextn_quant_config = None  # 设为None

        self.model = DeepseekModelNextN(  # NextN模型
            config, nextn_quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力TP组
        )
        self.logits_processor = LogitsProcessor(config)  # 逻辑处理器

    @torch.no_grad()
    # NextN因果语言模型前向传播
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入令牌ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        # 待办 当前仅支持预填充batch=1且len(input_ids) > self.cp_size * 2
        if self.dsa_enable_prefill_cp:  # 如果DSA启用预填充上下文并行
            if can_dsa_cp_split(  # 如果DSA可以上下文并行分割
                len(input_ids), self.cp_size, self.use_dsa, forward_batch
            ):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(  # 准备上下文并行元数据
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
        elif self.mla_enable_prefill_cp:  # 如果MLA启用预填充上下文并行
            if can_cp_split(len(input_ids), self.cp_size, forward_batch):  # 如果可以上下文并行分割
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(  # 准备上下文并行元数据
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
        hidden_states = self.model(input_ids, positions, forward_batch)  # 通过NextN模型
        return self.logits_processor(  # 通过逻辑处理器
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    # 加载权重
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=True)  # 调用父类加载权重（NextN模式）

    # 权重加载后处理
    def post_load_weights(self, is_nextn=True, weight_names=None):
        # `is_nextn` is pinned to True for the NextN subclass; the parameter is kept
        # only because the mixin's `do_load_weights` calls `self.post_load_weights`
        # with `is_nextn=...` as a kwarg.
        # `is_nextn`对NextN子类固定为True；保留该参数仅因为
        # mixin的`do_load_weights`调用`self.post_load_weights`时
        # 传入了`is_nextn=...`作为关键字参数。
        super().post_load_weights(is_nextn=True, weight_names=weight_names)  # 调用父类后处理


EntryClass = [DeepseekV3ForCausalLMNextN]  # 入口类列表
