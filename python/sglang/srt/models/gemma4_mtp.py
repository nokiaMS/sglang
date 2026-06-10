# Copyright 2026 SGLang Team  # 版权所有2026 SGLang团队
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证2.0版授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache许可证URL
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and  # 查看许可证了解管理权限和
# limitations under the License.  # 限制的特定语言
# ==============================================================================  # 分隔线
# Gemma4 MTP（Multi-Token Prediction）助手模型实现文件
# 本文件实现了Gemma4的MTP助手模型，用于推测性解码，支持冻结KV上下文、
# 质心掩码（centroid masking）有序嵌入和预/后投影等功能。

from __future__ import annotations  # 启用延迟注解评估

import copy  # 导入深拷贝模块
import logging  # 导入日志模块
from typing import Dict, Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig, PreTrainedModel  # 导入预训练配置和模型基类

from sglang.srt.distributed import get_pp_group  # 导入PP组获取函数
from sglang.srt.layers.linear import ReplicatedLinear  # 导入复制线性层
from sglang.srt.layers.logits_processor import (  # 导入logits处理器相关类
    LogitsMetadata,  # logits元数据
    LogitsProcessor,  # logits处理器
    LogitsProcessorOutput,  # logits处理器输出
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.mem_cache.memory_pool import KVCache  # 导入KV缓存
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM, Gemma4TextModel  # 导入Gemma4因果模型和文本模型
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPContext  # 导入冻结KV MTP上下文
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

logger = logging.getLogger(__name__)  # 创建日志记录器


def _get_text_config(model_or_config) -> PretrainedConfig:  # 获取文本配置函数
    """Normalize either a model or a (possibly wrapped) config to ``Gemma4TextConfig``."""  # 将模型或（可能包装的）配置归一化为Gemma4TextConfig
    cfg = getattr(model_or_config, "config", model_or_config)  # 获取config属性或使用自身
    return getattr(cfg, "text_config", cfg)  # 获取text_config属性或使用自身


def _resolve_target_text_model(target_model):  # 解析目标文本模型函数
    for attr in ("language_model", "model"):  # 遍历可能的属性名
        candidate = getattr(target_model, attr, None)  # 获取候选属性
        if candidate is not None and hasattr(candidate, "layers"):  # 如果候选存在且有layers属性
            return candidate  # 返回候选
    raise AttributeError(  # 抛出属性错误
        f"Frozen-KV MTP cannot locate the target trunk on "  # 冻结KV MTP无法定位目标主干
        f"{type(target_model).__name__}; expected ``.language_model`` "  # 在{类型名}上；预期.language_model
        "(multimodal) or ``.model`` (text-only) with a ``.layers`` attribute."  # （多模态）或.model（纯文本）带有.layers属性
    )


class Gemma4AssistantForCausalLM(Gemma4ForCausalLM):  # Gemma4 MTP助手因果模型类
    """Gemma 4 MTP assistant: target embed + recurrent hidden through pre/post projection; own ``lm_head``."""  # Gemma4 MTP助手：目标嵌入 + 通过预/后投影的循环隐藏状态；自己的lm_head

    base_model_prefix = "model"  # 基础模型前缀

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ) -> None:
        text_config = copy.deepcopy(_get_text_config(config))  # 深拷贝文本配置
        text_config.num_kv_shared_layers = 0  # 助手模型不使用KV共享层
        PreTrainedModel.__init__(self, config=text_config)  # 调用PreTrainedModel初始化
        self.assistant_config = config  # 保存助手配置
        self.config = text_config  # 保存文本配置
        self.quant_config = quant_config  # 保存量化配置
        self.pp_group = get_pp_group()  # 获取PP组

        self.vocab_size = text_config.vocab_size  # 词表大小
        self.hidden_size = text_config.hidden_size  # 隐藏层大小
        self.backbone_hidden_size = config.backbone_hidden_size  # 骨干隐藏层大小
        self.target_embed_scale = self.backbone_hidden_size**0.5  # 目标嵌入缩放因子
        self.use_ordered_embeddings = bool(  # 是否使用有序嵌入
            getattr(config, "use_ordered_embeddings", False)  # 从配置获取
        )
        self.centroid_intermediate_top_k = int(  # 质心中间top-k值
            getattr(config, "centroid_intermediate_top_k", 32)  # 默认32
        )

        self.target_embed_weight: Optional[torch.Tensor] = None  # 目标嵌入权重（后续绑定）
        self.pre_projection = ReplicatedLinear(  # 预投影层
            2 * self.backbone_hidden_size,  # 输入维度（token嵌入 + 上一隐藏状态）
            self.hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=None,  # 无量化
            prefix=add_prefix("pre_projection", prefix),  # 添加前缀
        )
        self.model = Gemma4TextModel(  # 文本模型
            config=text_config,  # 文本配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("model", prefix),  # 添加前缀
        )
        self.post_projection = ReplicatedLinear(  # 后投影层
            self.hidden_size,  # 输入维度
            self.backbone_hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=None,  # 无量化
            prefix=add_prefix("post_projection", prefix),  # 添加前缀
        )

        if text_config.tie_word_embeddings:  # 如果绑定词嵌入
            self.lm_head = self.model.embed_tokens  # lm_head共享嵌入层
        else:  # 不绑定
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)  # 创建独立的lm_head
        self.logits_processor = LogitsProcessor(text_config, skip_all_gather=True)  # logits处理器（跳过全收集）

        if self.use_ordered_embeddings:  # 如果使用有序嵌入
            self.num_centroids = int(config.num_centroids)  # 质心数量
            self.vocab_size_per_centroid, rem = divmod(  # 每个质心的词表大小
                self.vocab_size, self.num_centroids  # 词表大小除以质心数
            )
            if rem:  # 如果有余数
                raise ValueError(  # 抛出值错误
                    "Frozen-KV MTP centroid head requires vocab_size to be a "  # 冻结KV MTP质心头要求词表大小是
                    f"multiple of num_centroids (vocab={self.vocab_size}, "  # 质心数的整数倍
                    f"num_centroids={self.num_centroids})."  # 词表大小和质心数
                )
            self.centroids = nn.Linear(self.hidden_size, self.num_centroids, bias=False)  # 质心分类线性层
            self.register_buffer(  # 注册缓冲区
                "token_ordering",  # token排序
                torch.zeros(self.vocab_size, dtype=torch.long),  # 初始化为零
                persistent=True,  # 持久化
            )
        else:  # 不使用有序嵌入
            self.num_centroids = self.vocab_size_per_centroid = self.centroids = None  # 质心相关属性设为None
            self.register_buffer("token_ordering", None, persistent=False)  # token排序为None

        self.kv_context: Optional[FrozenKVMTPContext] = None  # 冻结KV上下文
        self.post_init()  # 后初始化

    def bind_frozen_kv_context(self, ctx: FrozenKVMTPContext) -> None:  # 绑定冻结KV上下文方法
        """Bind assistant attention to target-owned KV and suppress assistant KV writes."""  # 将助手注意力绑定到目标拥有的KV，并抑制助手KV写入
        for assistant_logical, layer in enumerate(self.model.layers):  # 遍历助手模型的每一层
            target_phys = ctx.get_physical_layer_id(assistant_logical)  # 获取目标物理层ID
            layer.self_attn.is_kv_shared_layer = True  # 标记为KV共享层
            layer.self_attn.kv_shared_layer_index = target_phys  # 设置KV共享层索引
            layer.self_attn.attn.layer_id = target_phys  # 设置注意力层ID
            layer.self_attn.layer_id = assistant_logical  # 设置层ID
        self.kv_context = ctx  # 保存KV上下文

    def build_frozen_kv_mtp_context(  # 构建冻结KV MTP上下文方法
        self,
        target_model,  # 目标模型
        target_token_to_kv_pool: KVCache,  # 目标token到KV池映射
    ) -> FrozenKVMTPContext:  # 返回冻结KV MTP上下文
        """Map each assistant layer to the target physical layer that owns its K/V.  # 将每个助手层映射到拥有其K/V的目标物理层

        HF Gemma 4 ties each typed (sliding/full) assistant layer to the target's  # HF Gemma4将每个类型化（滑动/全）助手层绑定到目标的
        last layer of the same type; that layer is itself KV-shared with an  # 同类型最后一层；该层本身通过kv_shared_layer_index
        earlier non-shared layer (via ``kv_shared_layer_index``). We collapse  # 与更早的非共享层KV共享。我们将
        those two hops once so attention can hand a direct ``layer_id`` to  # 这两跳折叠一次，以便注意力可以直接将layer_id
        ``RadixAttention`` at bind time.  # 传递给RadixAttention
        """
        target_text = _get_text_config(target_model)  # 获取目标文本配置
        assistant_text = _get_text_config(self)  # 获取助手文本配置
        layers = _resolve_target_text_model(target_model).layers  # 获取目标模型层列表

        def kv_owner(idx: int) -> int:  # KV拥有者函数
            attn = layers[idx].self_attn  # 获取注意力层
            owner = (  # 获取拥有者
                getattr(attn, "kv_shared_layer_index", None)  # 如果是KV共享层则获取共享索引
                if getattr(attn, "is_kv_shared_layer", False)  # 检查是否是KV共享层
                else idx  # 否则拥有者就是自身
            )
            if owner is None or getattr(  # 如果拥有者为None或是KV共享层
                layers[owner].self_attn, "is_kv_shared_layer", False
            ):
                raise RuntimeError(  # 抛出运行时错误
                    f"Frozen-KV MTP: target layer {idx} resolved to physical "  # 冻结KV MTP：目标层解析到物理
                    f"{owner!r}, which is missing or itself KV-shared "  # 层{owner!r}，该层缺失或本身是KV共享层
                    "(HF invariant changed?)."  # （HF不变量已变更？）
                )
            return owner  # 返回拥有者

        L = target_text.num_hidden_layers  # 目标层数
        by_type = {target_text.layer_types[i]: kv_owner(i) for i in (L - 2, L - 1)}  # 按类型映射最后两层的KV拥有者

        physical: Dict[int, int] = {}  # 物理层映射
        for i, t in enumerate(assistant_text.layer_types):  # 遍历助手层类型
            if t not in by_type:  # 如果类型不在映射中
                raise ValueError(  # 抛出值错误
                    f"Frozen-KV MTP assistant layer {i} has type {t!r}, "  # 冻结KV MTP助手层i的类型
                    f"expected one of {sorted(by_type)}."  # 预期类型
                )
            physical[i] = by_type[t]  # 映射到物理层

        return FrozenKVMTPContext(  # 返回冻结KV MTP上下文
            target_token_to_kv_pool=target_token_to_kv_pool,  # 目标KV池
            physical_layer_ids=physical,  # 物理层ID映射
        )

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:  # 获取嵌入和头权重
        if self.target_embed_weight is None:  # 如果目标嵌入未绑定
            raise RuntimeError(  # 抛出运行时错误
                "Gemma4AssistantForCausalLM target embedding is not bound yet."  # Gemma4助手模型的目标嵌入尚未绑定
            )
        return self.target_embed_weight, self.lm_head.weight  # 返回目标嵌入权重和lm_head权重

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:  # 设置嵌入和头权重
        """Rebind target embedding; ``head`` ignored (assistant keeps ``lm_head``)."""  # 重新绑定目标嵌入；head被忽略（助手保留自己的lm_head）
        del head  # 删除head参数
        self.target_embed_weight = embed  # 保存目标嵌入权重
        if torch.cuda.is_available():  # 如果CUDA可用
            torch.cuda.empty_cache()  # 清空CUDA缓存

    def get_attention_sliding_window_size(self) -> int:  # 获取注意力滑动窗口大小
        # Gemma 4 config treats the bound as inclusive; SGLang attention metadata  # Gemma4配置将边界视为包含的；SGLang注意力元数据
        # uses an exclusive window size, matching the target Gemma 4 models.  # 使用排他窗口大小，匹配目标Gemma4模型
        return self.config.sliding_window - 1  # 返回滑动窗口减1

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入（可选）
        **kwargs,  # 其他关键字参数
    ) -> LogitsProcessorOutput:  # 返回logits处理器输出
        if input_embeds is None:  # 如果没有提供输入嵌入
            if self.target_embed_weight is None:  # 如果目标嵌入未绑定
                raise RuntimeError(  # 抛出运行时错误
                    "Gemma4AssistantForCausalLM requires set_embed_and_head() "  # Gemma4助手模型需要先调用set_embed_and_head()
                    "before token-id forward."  # 才能进行token-ID前向传播
                )
            token_embed = (  # 计算token嵌入
                torch.nn.functional.embedding(input_ids, self.target_embed_weight)  # 查表获取嵌入
                * self.target_embed_scale  # 乘以目标嵌入缩放
            )
        else:  # 提供了输入嵌入
            token_embed = input_embeds  # 使用提供的嵌入

        if forward_batch.spec_info is None or not hasattr(  # 如果没有spec_info
            forward_batch.spec_info, "hidden_states"  # 或spec_info没有hidden_states属性
        ):
            raise RuntimeError(  # 抛出运行时错误
                "Frozen-KV MTP forward requires forward_batch.spec_info."  # 冻结KV MTP前向传播需要forward_batch.spec_info.
                "hidden_states to carry the recurrent state. The worker's "  # hidden_states来携带循环状态。worker的
                "_frozen_kv_target_view context manager must be exited "  # _frozen_kv_target_view上下文管理器必须在
                "before model forward, leaving spec_info populated."  # 模型前向传播之前退出，保留spec_info
            )
        prev_hidden = forward_batch.spec_info.hidden_states  # 获取上一隐藏状态
        if token_embed.shape != prev_hidden.shape:  # 如果形状不匹配
            raise ValueError(  # 抛出值错误
                "Frozen-KV MTP forward: token_embed and prev_hidden must have "  # 冻结KV MTP前向传播：token_embed和prev_hidden必须
                f"the same shape (got {token_embed.shape} vs {prev_hidden.shape})."  # 具有相同形状
            )

        z, _ = self.pre_projection(torch.cat([token_embed, prev_hidden], dim=-1))  # 预投影：拼接token嵌入和上一隐藏状态
        hidden_states = self.model(  # 通过文本模型
            input_ids=None,  # 不使用输入ID
            positions=positions,  # 位置编码
            forward_batch=forward_batch,  # 前向批次
            input_embeds=z,  # 使用预投影输出作为嵌入
            per_layer_inputs=None,  # 无每层输入
            **kwargs,  # 其他关键字参数
        )
        projected_states, _ = self.post_projection(hidden_states)  # 后投影回骨干维度

        if self.use_ordered_embeddings:  # 如果使用有序嵌入
            return self._centroid_logits_processor(  # 使用质心logits处理器
                input_ids, hidden_states, projected_states, forward_batch  # 输入ID，隐藏状态，投影状态和批次
            )

        return self.logits_processor(  # 使用标准logits处理器
            input_ids,  # 输入ID
            hidden_states,  # 隐藏状态
            self.lm_head,  # lm_head
            forward_batch,  # 前向批次
            hidden_states_before_norm=projected_states,  # 归一化前的隐藏状态
        )

    def _apply_centroid_masking(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 应用质心掩码方法
        """Centroid-masked logits for E2B/E4B assistant heads."""  # E2B/E4B助手头的质心掩码logits
        if self.centroids is None or self.token_ordering is None:  # 如果质心未初始化
            raise RuntimeError(  # 抛出运行时错误
                "Frozen-KV MTP centroid head invoked but centroid weights "  # 冻结KV MTP质心头被调用但质心权重
                "are not initialized."  # 未初始化
            )
        prefix_shape = hidden_states.shape[:-1]  # 保存前缀形状
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])  # 展平隐藏状态
        num_tokens = flat_hidden.shape[0]  # token数量

        _, top_k_indices = torch.topk(  # 获取top-k质心索引
            self.centroids(flat_hidden),  # 质心logits
            k=self.centroid_intermediate_top_k,  # top-k值
            dim=-1,  # 在最后一维
        )

        # Contiguous gather: [C, vpc, H] indexed by centroid IDs.  # 连续收集：通过质心ID索引[C, vpc, H]
        num_selected = self.centroid_intermediate_top_k * self.vocab_size_per_centroid  # 选中token数
        selected_embeddings = self.lm_head.weight.view(  # 重塑lm_head权重
            self.num_centroids,  # 质心数
            self.vocab_size_per_centroid,  # 每质心词表大小
            self.hidden_size,  # 隐藏大小
        )[top_k_indices].reshape(num_tokens, num_selected, self.hidden_size)  # 按top-k索引收集并重塑

        selected_logits = torch.bmm(  # 批量矩阵乘法计算选中logits
            flat_hidden.unsqueeze(1),  # 隐藏状态增加维度
            selected_embeddings.transpose(1, 2),  # 选中嵌入转置
        ).squeeze(1)  # 去除中间维度

        # Scatter to real vocab positions via token_ordering.  # 通过token_ordering散射到真实词表位置
        centroid_vocab_indices = (  # 质心词表索引
            self.token_ordering.long()  # 转为长整型
            .view(self.num_centroids, self.vocab_size_per_centroid)[top_k_indices]  # 按质心索引
            .view(num_tokens, -1)  # 重塑
        )
        mask_value = torch.finfo(selected_logits.dtype).min / 2  # 掩码值（数据类型最小值的一半）
        output = torch.full(  # 创建全掩码输出
            (num_tokens, self.vocab_size),  # 形状
            mask_value,  # 掩码值
            dtype=selected_logits.dtype,  # 数据类型
            device=selected_logits.device,  # 设备
        )
        output.scatter_(dim=-1, index=centroid_vocab_indices, src=selected_logits)  # 散射选中logits到对应位置
        return output.view(*prefix_shape, self.vocab_size)  # 重塑并返回

    def _centroid_logits_processor(  # 质心logits处理器方法
        self,
        input_ids: torch.Tensor,  # 输入ID
        hidden_states: torch.Tensor,  # 隐藏状态
        projected_states: torch.Tensor,  # 投影状态
        forward_batch: ForwardBatch,  # 前向批次
    ) -> LogitsProcessorOutput:  # 返回logits处理器输出
        logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)  # 从前向批次创建logits元数据
        if logits_metadata.extend_return_logprob:  # 如果扩展模式需要返回log概率
            raise NotImplementedError(  # 抛出未实现错误
                "Frozen-KV MTP centroid head does not support input logprobs yet."  # 冻结KV MTP质心头尚不支持输入log概率
            )

        (  # 解包pruned states
            pruned_states,  # 裁剪后的状态
            pruned_states_before_norm,  # 归一化前的裁剪状态
            aux_pruned_states,  # 辅助裁剪状态
            sample_indices,  # 采样索引
            *_,  # 其他值
        ) = self.logits_processor._get_pruned_states(  # 获取裁剪状态
            hidden_states, projected_states, None, logits_metadata  # 隐藏状态，投影状态，None，元数据
        )
        hidden_states_to_store = self.logits_processor._get_hidden_states_to_store(  # 获取要存储的隐藏状态
            hidden_states,  # 隐藏状态
            projected_states,  # 投影状态
            None,  # 无辅助状态
            pruned_states,  # 裁剪状态
            pruned_states_before_norm,  # 归一化前裁剪状态
            aux_pruned_states,  # 辅助裁剪状态
            sample_indices,  # 采样索引
            logits_metadata,  # logits元数据
        )
        del input_ids, hidden_states, projected_states  # 删除不再需要的变量

        logits = self._apply_centroid_masking(pruned_states)  # 应用质心掩码
        sampled_logits = (  # 采样logits
            logits[sample_indices] if sample_indices is not None else logits  # 如果有采样索引则索引
        )
        return LogitsProcessorOutput(  # 返回logits处理器输出
            next_token_logits=sampled_logits,  # 下一个token的logits
            hidden_states=hidden_states_to_store,  # 要存储的隐藏状态
            mm_input_embeds=logits_metadata.mm_input_embeds,  # 多模态输入嵌入
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法
        def remap_assistant_weights():  # 重映射助手权重生成器函数
            for name, weight in weights:  # 遍历权重
                if name.startswith("masked_embedding."):  # 如果名称以masked_embedding.开头
                    name = name.removeprefix("masked_embedding.")  # 移除前缀
                yield name, weight  # 生成名称和权重

        result = super().load_weights(remap_assistant_weights())  # 调用父类加载权重
        if self.use_ordered_embeddings:  # 如果使用有序嵌入
            self._reorder_embedding_to_centroid_order()  # 重排序嵌入到质心顺序
        return result  # 返回加载结果

    @torch.no_grad()  # 禁用梯度计算
    def _reorder_embedding_to_centroid_order(self) -> None:  # 重排序嵌入到质心顺序方法
        """Reorder lm_head.weight from natural vocab order to centroid order."""  # 将lm_head.weight从自然词表顺序重排序到质心顺序
        if self.token_ordering is None:  # 如果没有token排序
            return  # 直接返回
        ordering = self.token_ordering.long()  # 获取排序索引
        lm_head_w = self.lm_head.weight  # 获取lm_head权重
        reordered = lm_head_w.data[ordering]  # 按排序索引重排序
        lm_head_w.data.copy_(reordered)  # 复制回原权重
        logger.info(  # 记录信息日志
            "Reordered lm_head/embed_tokens (%s) to centroid order "  # 重排序lm_head/embed_tokens到质心顺序
            "for contiguous centroid masking.",  # 以实现连续质心掩码
            list(lm_head_w.shape),  # 权重形状
        )


EntryClass = Gemma4AssistantForCausalLM  # 入口类为Gemma4AssistantForCausalLM
