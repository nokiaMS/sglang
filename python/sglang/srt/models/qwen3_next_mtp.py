# Qwen3-Next MTP 推测解码模型实现
# 本文件实现了基于 Qwen3-Next 的 MTP（Multi-Token Prediction）推测解码模型，
# 用于加速混合线性注意力 + MoE 模型的推理。支持 NPU 量化回退和 EAGLE3 推测解码。
# Copyright 2023-2024 SGLang Team  # SGLang 团队版权
# Licensed under the Apache License, Version 2.0 (the "License");  # Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 不得违反许可证使用
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 除非法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 按原样分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何担保
# See the License for the specific language governing permissions and  # 查看许可证获取权限
# limitations under the License.  # 许可证限制
# ==============================================================================

"""Inference-only Qwen3Next MTP Speculative Decoding."""  # 仅推理的 Qwen3Next MTP 推测解码

import copy  # 导入深拷贝模块
import logging  # 导入日志模块
from contextlib import ExitStack  # 导入上下文管理栈
from typing import Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 框架
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 导入分布式函数
from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入专家分布记录器
from sglang.srt.layers.layernorm import GemmaRMSNorm  # 导入 Gemma RMS 归一化
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行语言模型头
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM, Qwen3NextModel  # 导入 Qwen3-Next 模型
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import add_prefix, is_npu  # 导入前缀添加和 NPU 检测工具

logger = logging.getLogger(__name__)  # 获取日志记录器


class Qwen3NextForCausalLMMTP(Qwen3NextForCausalLM):
    """Qwen3-Next MTP 推测解码模型"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化 Qwen3-Next MTP 推测解码模型"""
        nn.Module.__init__(self)  # 直接调用 nn.Module 初始化
        # Deep-copy so MTP mutations below don't leak into the target's config.  # 深拷贝以防止 MTP 修改影响目标模型配置
        config = copy.deepcopy(config)  # 深拷贝配置
        self.config = config  # 保存配置
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        if (  # NPU 量化回退
            is_npu()  # 如果是 NPU
            and get_global_server_args().speculative_draft_model_quantization is None  # 且未指定草稿模型量化
        ):
            quant_config = None  # 清空量化配置
        self.quant_config = quant_config  # 保存量化配置
        # if not set, model load will be broken in Qwen3NextForCausalLM load_weights()  # 如果不设置，权重加载会出错
        self.pp_group = get_pp_group()  # 流水线并行组
        # self.determine_num_fused_shared_experts("Qwen3NextForCausalLMMTP")  # 确定融合共享专家数（注释掉）

        # currently based on the provided ckpt, we:  # 目前根据提供的检查点，我们：
        # (1) do not use_dedicated_mtp_embeddings provided in ckpt since not provided and directly use the target model embeddings  # (1) 不使用检查点中的专用 MTP 嵌入，直接使用目标模型嵌入
        # (2) hardcode bias=False since not provided  # (2) 硬编码 bias=False 因为检查点中未提供
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)  # 特征融合线性层
        RMSNorm_cls = GemmaRMSNorm  # 归一化类
        self.pre_fc_norm_embedding = RMSNorm_cls(  # 嵌入预融合归一化
            config.hidden_size, config.rms_norm_eps  # 隐藏维度和 epsilon
        )
        self.pre_fc_norm_hidden = RMSNorm_cls(config.hidden_size, config.rms_norm_eps)  # 隐藏状态预融合归一化
        config.num_hidden_layers = 1  # MTP 仅使用 1 层
        config.full_attention_interval = 1  # 全注意力间隔设为 1
        self.model = Qwen3NextModel(  # 创建 Qwen3-Next 模型主体
            config,  # 配置
            quant_config,  # 量化配置
            prefix=add_prefix("model", prefix),  # 参数前缀
            is_nextn=True,  # 标记为 nextn 模式
        )
        self.lm_head = ParallelLMHead(  # 并行语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("model.shared_head.head", prefix),  # 参数前缀（共享头）
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力 TP 组
        )
        self.logits_processor = LogitsProcessor(config)  # logits 处理器

    @torch.no_grad()  # 禁用梯度计算
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """MTP 前向传播：嵌入 -> 特征融合 -> 模型主体 -> logits"""
        exit_stack = ExitStack()  # 创建上下文管理栈
        if (  # NPU 非量化回退
            is_npu()  # 如果是 NPU
            and self.quant_config is None  # 且没有量化配置
            and get_global_server_args().quantization is not None  # 但服务器有量化配置
        ):
            # ascend mtp unquant  # Ascend MTP 非量化模式
            exit_stack.enter_context(envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True))  # 覆盖 BF16 调度
            exit_stack.enter_context(  # 覆盖深度归一化模式
                envs.DEEP_NORMAL_MODE_USE_INT8_QUANT.override(False)  # 禁用 INT8 量化
            )

        try:  # 尝试执行
            if input_embeds is None:  # 如果没有输入嵌入
                input_embeds = self.model.embed_tokens(input_ids)  # 通过词嵌入层

            hidden_states = forward_batch.spec_info.hidden_states  # 获取推测信息的隐藏状态
            # Some idle batch has 0 batch size. GemmaRMSNorm.forward would fail due to bs=0.  # 空闲批次大小为 0 时 GemmaRMSNorm 会失败
            if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
                input_embeds = self.pre_fc_norm_embedding(input_embeds)  # 归一化嵌入
                hidden_states = self.pre_fc_norm_hidden(hidden_states)  # 归一化隐藏状态
            hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))  # 特征融合

            with get_global_expert_distribution_recorder().disable_this_region():  # 禁用专家分布记录
                hidden_states = self.model(  # 通过模型主体
                    input_ids,  # 输入 ID
                    positions,  # 位置信息
                    forward_batch,  # 前向批次
                    hidden_states,  # 隐藏状态
                )
        finally:  # 确保清理
            exit_stack.close()  # 关闭上下文管理栈

        return self.logits_processor(  # 处理 logits
            input_ids, hidden_states, self.lm_head, forward_batch  # 输入 ID、隐藏状态、语言模型头和批次
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        """加载 MTP 模型权重，标记为 MTP 模式"""
        super().load_weights(weights, is_mtp=True)  # 以 MTP 模式加载权重


EntryClass = [Qwen3NextForCausalLMMTP]  # 模型入口类列表
