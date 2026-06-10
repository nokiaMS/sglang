# Nemotron-H 多令牌预测器（MTP）模型实现
# 该文件实现了 Nemotron-H 的多令牌预测器，用于投机解码，
# 包含注意力解码层和 MoE 解码层的 MTP 变体，以及完整的 MTP 模型。

# Copyright 2023-2025 SGLang Team
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

from collections.abc import Iterable  # 导入可迭代类型 # 导入可迭代类型

import torch  # 导入 PyTorch # 导入 PyTorch 框架
from torch import nn  # 导入神经网络模块 # 导入神经网络模块

from sglang.srt.configs import NemotronHConfig  # 导入 Nemotron-H 配置 # 导入 Nemotron-H 配置
from sglang.srt.distributed import get_pp_group  # 导入流水线并行组 # 导入流水线并行组
from sglang.srt.layers.layernorm import RMSNorm  # 导入 RMS 归一化 # 导入 RMS 归一化层
from sglang.srt.layers.linear import ColumnParallelLinear  # 导入列并行线性层 # 导入列并行线性层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器 # 导入 logits 处理器
from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置 # 导入量化配置
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入 # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头 # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入 # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # 导入前向批次信息
from sglang.srt.models.nemotron_h import (  # 导入 Nemotron-H 模型组件 # 导入 Nemotron-H 模型组件
    NemotronHAttentionDecoderLayer,  # 注意力解码层 # 注意力解码层
    NemotronHForCausalLM,  # Nemotron-H 因果语言模型 # Nemotron-H 因果语言模型
    NemotronHMoEDecoderLayer,  # MoE 解码层 # MoE 解码层
)
from sglang.srt.server_args import get_global_server_args  # 导入服务器参数 # 导入服务器参数
from sglang.srt.utils import add_prefix  # 导入前缀工具 # 导入前缀添加工具


class NemotronHMTPAttentionDecoderLayer(NemotronHAttentionDecoderLayer):  # MTP 注意力解码层 # 继承自 Nemotron-H 注意力解码层
    """Nemotron-H MTP 注意力解码层，增加了起始投影和终端归一化"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: NemotronHConfig,  # 模型配置 # 模型配置
        layer_idx: int,  # 层索引 # 层索引
        quant_config: QuantizationConfig | None = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
        has_start_projections: bool = False,  # 是否有起始投影 # 是否有起始投影
        has_end_norm: bool = False,  # 是否有终端归一化 # 是否有终端归一化
    ) -> None:
        super().__init__(  # 调用父类初始化 # 调用父类初始化
            config=config,  # 配置 # 模型配置
            layer_idx=layer_idx,  # 层索引 # 层索引
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=prefix,  # 前缀 # 参数名前缀
        )
        self.has_start_projections = has_start_projections  # 保存起始投影标志 # 保存起始投影标志
        self.has_end_norm = has_end_norm  # 保存终端归一化标志 # 保存终端归一化标志

        if has_start_projections:  # 如果有起始投影 # 如果有起始投影
            self.enorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)  # 嵌入归一化 # 嵌入归一化层
            self.hnorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)  # 隐藏状态归一化 # 隐藏状态归一化层

            # Fusion layer to combine embeddings with target hidden states # 融合层，将嵌入与目标隐藏状态合并
            self.eh_proj = ColumnParallelLinear(  # 嵌入-隐藏投影 # 嵌入-隐藏状态融合投影
                input_size=config.hidden_size * 2,  # 输入大小（拼接后） # 输入维度（拼接后为两倍）
                output_size=config.hidden_size,  # 输出大小 # 输出维度
                bias=False,  # 无偏置 # 无偏置
                gather_output=True,  # 收集输出 # 收集所有并行结果
                params_dtype=(  # 参数数据类型 # 参数数据类型
                    config.dtype if hasattr(config, "dtype") else torch.bfloat16  # 从配置获取或默认 bfloat16 # 从配置获取或默认 bfloat16
                ),
                quant_config=quant_config,  # 量化配置 # 量化配置
                prefix=f"{prefix}.eh_proj",  # 前缀 # 参数名前缀
            )

        if has_end_norm:  # 如果有终端归一化 # 如果有终端归一化
            self.final_layernorm = RMSNorm(  # 最终层归一化 # 最终层归一化
                config.hidden_size,  # 隐藏层大小 # 隐藏层维度
                eps=getattr(config, "layer_norm_epsilon", 1e-5),  # epsilon # epsilon 值
            )

    def forward(  # 前向传播 # 前向传播方法
        self,
        *,
        inputs_embeds: torch.Tensor,  # 输入嵌入 # 输入嵌入
        hidden_states: torch.Tensor,  # 隐藏状态 # 隐藏状态
        residual: torch.Tensor | None = None,  # 残差 # 残差张量
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回隐藏状态和残差 # 返回隐藏状态和残差
        if self.has_start_projections:  # 如果有起始投影 # 如果有起始投影
            inputs_embeds_normed = self.enorm(inputs_embeds)  # 归一化嵌入 # 归一化嵌入
            previous_hidden_states_normed = self.hnorm(hidden_states)  # 归一化隐藏状态 # 归一化隐藏状态

            fused = torch.cat(  # 拼接归一化后的嵌入和隐藏状态 # 拼接归一化后的嵌入和隐藏状态
                [inputs_embeds_normed, previous_hidden_states_normed], dim=-1  # 沿最后一维拼接 # 沿最后一维拼接
            )
            hidden_states, _ = self.eh_proj(fused)  # 融合投影 # 通过融合投影层

        hidden_states, residual = super().forward(  # 调用父类前向 # 调用父类前向传播
            hidden_states=hidden_states,  # 隐藏状态 # 隐藏状态
            residual=residual,  # 残差 # 残差
            forward_batch=forward_batch,  # 前向批次 # 前向批次信息
        )

        if self.has_end_norm:  # 如果有终端归一化 # 如果有终端归一化
            if residual is not None:  # 如果有残差 # 如果有残差
                hidden_states = hidden_states + residual  # 合并残差 # 合并残差
                residual = None  # 清空残差 # 清空残差

            hidden_states = self.final_layernorm(hidden_states)  # 终端归一化 # 应用终端归一化

        return hidden_states, residual  # 返回隐藏状态和残差 # 返回隐藏状态和残差


class NemotronHMTPMoEDecoderLayer(NemotronHMoEDecoderLayer):  # MTP MoE 解码层 # 继承自 Nemotron-H MoE 解码层
    """Nemotron-H MTP MoE 解码层，增加了起始投影和终端归一化"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: NemotronHConfig,  # 模型配置 # 模型配置
        layer_idx: int,  # 层索引 # 层索引
        quant_config: QuantizationConfig | None = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
        has_start_projections: bool = False,  # 是否有起始投影 # 是否有起始投影
        has_end_norm: bool = False,  # 是否有终端归一化 # 是否有终端归一化
    ) -> None:
        super().__init__(  # 调用父类初始化 # 调用父类初始化
            config=config,  # 配置 # 模型配置
            layer_idx=layer_idx,  # 层索引 # 层索引
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=prefix,  # 前缀 # 参数名前缀
        )
        self.has_start_projections = has_start_projections  # 保存起始投影标志 # 保存起始投影标志
        self.has_end_norm = has_end_norm  # 保存终端归一化标志 # 保存终端归一化标志

        if has_start_projections:  # 如果有起始投影 # 如果有起始投影
            self.enorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)  # 嵌入归一化 # 嵌入归一化层
            self.hnorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)  # 隐藏状态归一化 # 隐藏状态归一化层

            self.eh_proj = ColumnParallelLinear(  # 嵌入-隐藏投影 # 嵌入-隐藏状态融合投影
                input_size=config.hidden_size * 2,  # 输入大小（拼接后） # 输入维度（拼接后为两倍）
                output_size=config.hidden_size,  # 输出大小 # 输出维度
                bias=False,  # 无偏置 # 无偏置
                gather_output=True,  # 收集输出 # 收集所有并行结果
                params_dtype=(  # 参数数据类型 # 参数数据类型
                    config.dtype if hasattr(config, "dtype") else torch.bfloat16  # 从配置获取或默认 bfloat16 # 从配置获取或默认 bfloat16
                ),
                quant_config=quant_config,  # 量化配置 # 量化配置
                prefix=f"{prefix}.eh_proj",  # 前缀 # 参数名前缀
            )

        if has_end_norm:  # 如果有终端归一化 # 如果有终端归一化
            self.final_layernorm = RMSNorm(  # 最终层归一化 # 最终层归一化
                config.hidden_size,  # 隐藏层大小 # 隐藏层维度
                eps=getattr(config, "layer_norm_epsilon", 1e-5),  # epsilon # epsilon 值
            )

    def forward(  # 前向传播 # 前向传播方法
        self,
        *,
        inputs_embeds: torch.Tensor,  # 输入嵌入 # 输入嵌入
        hidden_states: torch.Tensor,  # 隐藏状态 # 隐藏状态
        residual: torch.Tensor | None = None,  # 残差 # 残差张量
        forward_batch: ForwardBatch,  # 前向批次 # 前达批次信息
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回隐藏状态和残差 # 返回隐藏状态和残差
        if self.has_start_projections:  # 如果有起始投影 # 如果有起始投影
            inputs_embeds_normed = self.enorm(inputs_embeds)  # 归一化嵌入 # 归一化嵌入
            previous_hidden_states_normed = self.hnorm(hidden_states)  # 归一化隐藏状态 # 归一化隐藏状态

            fused = torch.cat(  # 拼接归一化后的嵌入和隐藏状态 # 拼接归一化后的嵌入和隐藏状态
                [inputs_embeds_normed, previous_hidden_states_normed], dim=-1  # 沿最后一维拼接 # 沿最后一维拼接
            )
            hidden_states, _ = self.eh_proj(fused)  # 融合投影 # 通过融合投影层

        hidden_states, residual = super().forward(  # 调用父类前向 # 调用父类前向传播
            hidden_states=hidden_states,  # 隐藏状态 # 隐藏状态
            residual=residual,  # 残差 # 残差
            forward_batch=forward_batch,  # 前向批次 # 前向批次信息
        )

        if self.has_end_norm:  # 如果有终端归一化 # 如果有终端归一化
            if residual is not None:  # 如果有残差 # 如果有残差
                hidden_states = hidden_states + residual  # 合并残差 # 合并残差
                residual = None  # 清空残差 # 清空残差

            hidden_states = self.final_layernorm(hidden_states)  # 终端归一化 # 应用终端归一化

        return hidden_states, residual  # 返回隐藏状态和残差 # 返回隐藏状态和残差


class NemotronHMultiTokenPredictor(nn.Module):  # 多令牌预测器 # Nemotron-H 多令牌预测器
    """Nemotron-H 多令牌预测器，用于投机解码"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: NemotronHConfig,  # 模型配置 # 模型配置
        quant_config: QuantizationConfig | None = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用父类初始化

        self.config = config  # 保存配置 # 保存模型配置
        self.vocab_size = config.vocab_size  # 词表大小 # 词表大小
        self.org_vocab_size = config.vocab_size  # 原始词表大小 # 原始词表大小

        self.mtp_start_layer_idx = config.num_hidden_layers  # MTP 起始层索引 # MTP 起始层索引
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)  # MTP 层数 # MTP 层数
        assert (  # 断言 # 断言只支持一层 MTP
            self.num_mtp_layers == 1
        ), "Only one MTP layer is supported for NemotronH-MTP"  # 只支持一层 # 只支持一层 MTP

        self.pattern_str = config.mtp_hybrid_override_pattern  # MTP 混合模式字符串 # MTP 混合模式字符串
        self.pattern_len = len(self.pattern_str)  # 模式长度 # 模式长度
        assert self.pattern_len > 0  # 断言模式非空 # 断言模式非空

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入 # 词表并行嵌入层
            self.vocab_size,  # 词表大小 # 词表大小
            config.hidden_size,  # 隐藏层大小 # 隐藏层维度
        )

        # Build flat list of layers # 构建扁平的层列表
        self.layers = nn.ModuleDict()  # 层字典 # 使用 ModuleDict 存储层

        # Total number of physical layers = num_steps * pattern_len # 物理层总数 = 步数 * 模式长度
        total_layers = self.num_mtp_layers * self.pattern_len  # 总层数 # 总层数
        for i in range(total_layers):  # 遍历所有层 # 遍历所有层
            step_rel_idx = i % self.pattern_len  # 步内相对索引 # 步内相对索引

            char = self.pattern_str[step_rel_idx]  # 当前字符 # 当前模式字符

            is_start_of_step = step_rel_idx == 0  # 是否为步的起始 # 是否为步的起始
            is_end_of_step = step_rel_idx == self.pattern_len - 1  # 是否为步的结束 # 是否为步的结束

            layer_prefix = f"{prefix}.layers.{i}"  # 层前缀 # 层前缀

            common_kwargs = dict(  # 公共参数 # 公共参数
                config=config,  # 配置 # 模型配置
                layer_idx=i,  # 层索引 # 层索引
                quant_config=quant_config,  # 量化配置 # 量化配置
                prefix=layer_prefix,  # 前缀 # 参数名前缀
                has_start_projections=is_start_of_step,  # 起始投影 # 是否有起始投影
                has_end_norm=is_end_of_step,  # 终端归一化 # 是否有终端归一化
            )

            if char == "*":  # 注意力层 # 注意力层
                self.layers[str(i)] = NemotronHMTPAttentionDecoderLayer(**common_kwargs)  # 创建注意力层 # 创建注意力层
            elif char == "E":  # MoE 层 # MoE 层
                self.layers[str(i)] = NemotronHMTPMoEDecoderLayer(**common_kwargs)  # 创建 MoE 层 # 创建 MoE 层
            else:  # 其他 # 不支持的类型
                raise NotImplementedError(  # 抛出异常 # 抛出未实现异常
                    f"Pattern char '{char}' in {self.pattern_str} not implemented"
                )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入 # 获取输入嵌入
        """通过嵌入层获取输入 ID 的嵌入表示"""
        assert (  # 断言 # 断言嵌入层已初始化
            self.embed_tokens is not None
        ), "embed_tokens not initialized - must be shared from target model"  # 嵌入层必须从目标模型共享 # 嵌入层必须从目标模型共享
        return self.embed_tokens(input_ids)  # 返回嵌入 # 返回嵌入

    def forward(  # 前向传播 # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 ID # 输入标记 ID
        hidden_states: torch.Tensor,  # 隐藏状态 # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
        inputs_embeds: torch.Tensor | None = None,  # 输入嵌入 # 输入嵌入（可选）
    ) -> torch.Tensor:  # 返回隐藏状态 # 返回隐藏状态
        if inputs_embeds is None:  # 如果没有提供嵌入 # 如果没有提供嵌入
            inputs_embeds = self.get_input_embeddings(input_ids)  # 获取嵌入 # 通过嵌入层获取嵌入

        residual = None  # 初始化残差 # 初始化残差

        for i in range(self.pattern_len):  # 遍历模式长度 # 遍历模式中的每一层
            hidden_states, residual = self.layers[str(i)](  # 层前向 # 层前向传播
                inputs_embeds=inputs_embeds,  # 输入嵌入 # 输入嵌入
                hidden_states=hidden_states,  # 隐藏状态 # 隐藏状态
                residual=residual,  # 残差 # 残差
                forward_batch=forward_batch,  # 前向批次 # 前向批次信息
            )
        return hidden_states  # 返回隐藏状态 # 返回隐藏状态


class NemotronHForCausalLMMTP(NemotronHForCausalLM):  # MTP 因果语言模型 # 继承自 Nemotron-H 因果语言模型
    """Nemotron-H 多令牌预测因果语言模型"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: NemotronHConfig,  # 模型配置 # 模型配置
        quant_config: QuantizationConfig | None = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ):
        nn.Module.__init__(self)  # 直接调用 Module 初始化 # 直接调用 Module 初始化（跳过父类）
        config = config.get_mtp_config()  # 获取 MTP 配置 # 获取 MTP 配置
        self.config = config  # 保存配置 # 保存模型配置
        self.quant_config = quant_config  # 保存量化配置 # 保存量化配置
        # Required for parent's load_weights # 父类 load_weights 所需
        self.pp_group = get_pp_group()  # 流水线并行组 # 获取流水线并行组

        # Override config for MTP pattern (which has no Mamba layers) # 覆盖配置以适配 MTP 模式（无 Mamba 层）
        config.num_hidden_layers = len(config.mtp_hybrid_override_pattern)  # 覆盖隐藏层数 # 覆盖隐藏层数
        # Set hybrid_override_pattern to MTP pattern so attention backend
        # doesn't use Mamba2AttnBackend (MTP has no Mamba layers) # 设置混合模式为 MTP 模式
        config.hybrid_override_pattern = config.mtp_hybrid_override_pattern  # 覆盖混合模式 # 覆盖混合模式

        self.model = NemotronHMultiTokenPredictor(  # MTP 模型 # 创建多令牌预测器
            config=config,  # 配置 # 模型配置
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("mtp", prefix),  # 前缀 # 参数名前缀
        )

        self.lm_head = ParallelLMHead(  # 语言模型头 # 创建并行语言模型头
            self.config.vocab_size,  # 词表大小 # 词表大小
            self.config.hidden_size,  # 隐藏层大小 # 隐藏层维度
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("lm_head", prefix),  # 前缀 # 参数名前缀
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力 TP 组 # 是否使用注意力 TP 组
        )

        self.logits_processor = LogitsProcessor(config)  # logits 处理器 # 创建 logits 处理器

    @torch.no_grad()  # 禁用梯度 # 禁用梯度计算
    def forward(  # 前向传播 # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 ID # 输入标记 ID
        positions: torch.Tensor,  # 位置编码 # 位置编码
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
        input_embeds: torch.Tensor | None = None,  # 输入嵌入 # 输入嵌入（可选）
        **kwargs,  # 其他参数 # 其他参数
    ) -> torch.Tensor:  # 返回 logits # 返回 logits
        hidden_states = forward_batch.spec_info.hidden_states  # 获取投机解码的隐藏状态 # 获取投机解码的隐藏状态

        hidden_states = self.model(  # MTP 模型前向 # MTP 模型前向传播
            input_ids,  # 输入 ID # 输入标记 ID
            hidden_states,  # 隐藏状态 # 隐藏状态
            forward_batch,  # 前向批次 # 前向批次信息
            input_embeds,  # 输入嵌入 # 输入嵌入
        )
        return self.logits_processor(  # logits 处理 # 通过 logits 处理器计算 logits
            input_ids, hidden_states, self.lm_head, forward_batch  # 输入参数 # 输入参数
        )

    def load_weights(  # 加载权重 # 加载模型权重
        self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False  # 权重和 MTP 标志 # 权重迭代器和 MTP 标志
    ):
        """加载 MTP 权重，委托给父类并标记为 MTP"""
        super().load_weights(weights, is_mtp=True)  # 委托给父类加载 MTP 权重 # 委托给父类加载 MTP 权重


EntryClass = [NemotronHForCausalLMMTP]  # 入口类 # 模型入口类列表
