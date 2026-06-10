# Step3.5 多Token预测(MTP)模型实现
# 本文件实现了Step3.5的推测解码模型，包含共享头(SharedHead)和多Token预测器。
# 支持链式多层MTP结构，每个MTP层消费前一层的隐藏状态。

import logging  # 导入日志模块
from collections.abc import Iterable  # 导入可迭代类型
from typing import Optional  # 导入可选类型

import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入TP世界大小获取函数
from sglang.srt.layers.layernorm import GemmaRMSNorm  # 导入Gemma RMS归一化
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.step3p5 import Step3p5DecoderLayer, Step3p5ForCausalLM  # 导入Step3.5模型组件
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

logger = logging.getLogger(__name__)  # 获取日志记录器


def get_spec_layer_idx_from_weight_name(
    config: PretrainedConfig, weight_name: str
) -> Optional[int]:
    """Return MTP/nextn layer index if this weight belongs to spec layers.
    
    Step3p5 MTP/nextn checkpoints append extra layers after the main decoder:
      model.layers.[num_hidden_layers ... num_hidden_layers + num_nextn_predict_layers)
    """  # 返回MTP/nextn层索引，如果权重属于推测层
    if hasattr(config, "num_nextn_predict_layers") and (  # 如果配置有nextn层数
        getattr(config, "num_nextn_predict_layers", 0) > 0
    ):
        base = config.num_hidden_layers  # 基准层ID
        for i in range(config.num_nextn_predict_layers):  # 遍历nextn层
            if weight_name.startswith(f"model.layers.{base + i}."):  # 匹配权重名前缀
                return base + i  # 返回层索引
    return None  # 不属于推测层返回None


class SharedHead(nn.Module):
    """共享头模块，包含归一化和语言模型头"""

    def __init__(
        self,
        config,  # 模型配置
        quant_config=None,  # 量化配置
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)  # 归一化层
        self.head = ParallelLMHead(  # 语言模型头
            config.vocab_size, config.hidden_size, quant_config=quant_config
        )
        self.lm_head = self.head  # LM头引用

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """共享头前向传播，执行归一化"""
        return self.norm(hidden_states)  # 返回归一化后的隐藏状态


class Step3p5AMultiTokenPredictor(nn.Module):
    """Step3.5多Token预测器，包含嵌入、投影和MTP解码块"""

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏大小
        )
        self.mtp_start_layer_idx = config.num_hidden_layers  # MTP起始层索引
        self.num_mtp_layers = config.num_nextn_predict_layers  # MTP层数

        layer_id = 45  # FIXME # 固定层ID

        self.enorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)  # 嵌入归一化
        self.hnorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)  # 隐藏状态归一化
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)  # 嵌入-隐藏投影
        self.shared_head = SharedHead(config=config, quant_config=quant_config)  # 共享头
        self.mtp_block = Step3p5DecoderLayer(  # MTP解码块
            config=config, layer_id=layer_id, prefix=f"{prefix}.mtp_block"
        )
        self.lm_head = self.shared_head.head  # LM头引用

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
    ) -> torch.Tensor:
        """多Token预测器前向传播：嵌入 -> 投影 -> MTP块 -> 归一化"""
        if input_embeds is None:  # 如果没有输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过嵌入层
        else:
            hidden_states = input_embeds  # 使用输入嵌入

        if hidden_states.shape[0] > 0:  # 如果有token
            hidden_states = self.eh_proj(  # 通过嵌入-隐藏投影
                torch.cat(  # 拼接归一化后的嵌入和隐藏状态
                    (
                        self.enorm(hidden_states),  # 归一化后的嵌入
                        self.hnorm(forward_batch.spec_info.hidden_states),  # 归一化后的推测隐藏状态
                    ),
                    dim=-1,  # 在最后一维拼接
                )
            )
        hidden_states, residual = self.mtp_block(  # 通过MTP解码块
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
            residual=None,  # 初始残差为None
        )
        hidden_states_before_norm = None  # 归一化前隐藏状态
        if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
            # if forward_batch.return_hidden_states_before_norm:
            hidden_states_before_norm = (  # 计算归一化前隐藏状态
                hidden_states if residual is None else hidden_states + residual
            )
            if residual is not None:  # 如果有残差
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)  # 带残差归一化
            else:
                hidden_states = self.shared_head.norm(hidden_states)  # 无残差归一化

        return hidden_states, hidden_states_before_norm  # 返回归一化后和前的隐藏状态

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将输入ID转换为嵌入"""
        return self.embed_tokens(input_ids)  # 通过嵌入层


# Chain-style multi-layer MTP (standard Step-3.5 Flash design):
# each MTP layer consumes the hidden states produced by the preceding MTP layer,
# while layer-0 consumes the hidden states from the target model.
# The chain propagation is driven by MultiLayerEagleDraftWorker via the
# ``chain_mtp_hidden_states`` flag: between speculative steps it overwrites
# ``forward_batch.spec_info.hidden_states`` (and the CUDA-graph hidden_states
# buffer in the draft-extend graph) with the previous layer's
# ``hidden_states_before_norm`` returned by ``Step3p5AMultiTokenPredictor``.
class Step3p5MTP(Step3p5ForCausalLM):
    """Step3.5多Token预测模型，继承自Step3p5ForCausalLM"""

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        draft_model_idx: Optional[int] = None,  # 推测模型索引
        prefix: str = "",  # 参数前缀
    ) -> None:
        nn.Module.__init__(self)  # 直接调用nn.Module初始化
        self.config = config  # 保存配置
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        self.quant_config = quant_config  # 保存量化配置
        self.draft_model_idx = draft_model_idx  # 保存推测模型索引

        self.model = Step3p5AMultiTokenPredictor(  # 创建多Token预测器
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.logits_processor = LogitsProcessor(config)  # 创建logits处理器
        self.lm_head = self.model.lm_head  # LM头引用

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将输入ID转换为嵌入"""
        return self.model.embed_input_ids(input_ids)  # 通过模型的嵌入层

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """MTP模型前向传播：多Token预测器 -> logits处理"""
        hidden_states, hidden_states_before_norm = self.model(  # 通过多Token预测器
            input_ids, positions, forward_batch
        )
        return self.logits_processor(  # 处理logits
            input_ids,  # 输入ID
            hidden_states,  # 隐藏状态
            self.model.shared_head.head,  # 共享头
            forward_batch,  # 前向批次
            hidden_states_before_norm=hidden_states_before_norm,  # 归一化前隐藏状态
        )

    def get_embed_and_head(self):
        """获取嵌入权重和LM头权重"""
        return self.model.embed_tokens.weight, self.model.shared_head.head.weight  # 返回嵌入和头权重

    def set_embed_and_head(self, embed, head):
        """设置嵌入权重和LM头权重（MTP模型不支持）"""
        return  # MTP模型不支持设置嵌入和头

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """加载MTP模型权重"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),  # Q投影
            ("qkv_proj", "k_proj", "k"),  # K投影
            ("qkv_proj", "v_proj", "v"),  # V投影
            ("gate_up_proj", "gate_proj", 0),  # 门控投影
            ("gate_up_proj", "up_proj", 1),  # 上投影
        ]

        expert_params_mapping = [  # 专家参数映射
            (".moe.experts.w13_weight", ".moe.gate_proj.weight", "w1"),  # 门控权重
            (".moe.experts.w13_weight", ".moe.up_proj.weight", "w3"),  # 上投影权重
            (".moe.experts.w2_weight", ".moe.down_proj.weight", "w2"),  # 下投影权重
        ]

        params_dict = dict(self.named_parameters())  # 参数字典
        loaded_params: set[str] = set()  # 已加载参数集合
        for name, loaded_weight in weights:  # 遍历权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转位置编码频率
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)  # 获取推测层索引
            if spec_layer is not None and spec_layer != (  # 如果不是当前推测层
                self.config.num_hidden_layers + self.draft_model_idx
            ):
                continue  # 跳过
            if "embed_tokens" not in name and spec_layer is None:  # 如果不是嵌入且非推测层
                continue  # 跳过
            name = self._rewrite_spec_layer_name(spec_layer, name)  # 重写推测层名称
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:  # 如果权重名不包含
                    continue  # 跳过
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:  # 专家权重但参数不存在
                    continue  # 跳过
                if "experts" in name or "moe" in name:  # MoE专家权重
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换权重名
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ额外偏置
                    continue

                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出内循环
            else:  # 没有匹配到堆叠参数
                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, shard_id = mapping  # 解包映射
                    if weight_name not in name:  # 如果权重名不包含
                        continue  # 跳过
                    name = name.replace(weight_name, param_name)  # 替换权重名
                    # Skip loading extra bias for GPTQ models.
                    if (  # 跳过GPTQ额外偏置
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    for expert_id in range(loaded_weight.shape[0]):  # 遍历专家
                        loaded_weight_expert = loaded_weight[expert_id]  # 获取专家权重
                        weight_loader(  # 加载权重
                            param,
                            loaded_weight_expert,  # 专家权重
                            name,  # 权重名
                            shard_id=shard_id,  # 分片ID
                            expert_id=expert_id,  # 专家ID
                        )
                    loaded_params.add(name)  # 添加到已加载集合
                    break  # 跳出内循环
                else:  # 没有匹配到专家参数映射
                    # Skip loading extra bias for GPTQ models.
                    if (  # 跳过GPTQ额外偏置
                        name.endswith(".bias")
                        and name not in params_dict
                        or "tok_embeddings" in name  # 跳过tok_embeddings
                    ):
                        continue

                    if "shared_head" in name:  # 如果是共享头权重
                        name = name.replace("shared_head.output", "shared_head.head")  # 重命名
                    if "embed_tokens" in name:  # 如果是嵌入权重
                        assert (  # 断言有nextn配置
                            hasattr(self.config, "num_nextn_predict_layers")
                            and self.config.num_nextn_predict_layers > 0
                        )
                        name = "model.embed_tokens.weight"  # 统一嵌入权重名
                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重
            loaded_params.add(name)  # 添加到已加载集合
        params_need_to_load = set(params_dict.keys())  # 需要加载的参数
        if params_need_to_load != loaded_params:  # 如果有未加载的参数
            missing_params = list(params_need_to_load - loaded_params)  # 缺失参数列表
            param_name_example = missing_params[0]  # 示例参数名
            raise RuntimeError(  # 抛出运行时错误
                f"Some parameters like {param_name_example} are not in the checkpoint and will falsely use random initialization"
            )
        return loaded_params  # 返回已加载参数集合

    def _rewrite_spec_layer_name(self, spec_layer: Optional[int], name: str) -> str:
        """重写推测层权重名称以匹配模型格式"""
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        """
        if spec_layer is None:  # 如果不是推测层
            return name  # 直接返回

        # Some checkpoints place MTP weights under "model.layers.<id>.transformer.*".
        # Our modules use "model.layers.<id>.*", so drop the ".transformer." segment.
        transformer_prefix = f"model.layers.{spec_layer}.transformer."  # transformer前缀
        if name.startswith(transformer_prefix):  # 如果有transformer前缀
            name = name.replace(".transformer.", ".", 1)  # 移除transformer段

        spec_layer_weight_names = [  # 推测层权重名称列表
            "embed_tokens",  # 嵌入层
            "enorm",  # 嵌入归一化
            "hnorm",  # 隐藏归一化
            "eh_proj",  # 嵌入-隐藏投影
            "shared_head",  # 共享头
        ]
        spec_layer_weight = False  # 是否为推测层特殊权重
        for weight_name in spec_layer_weight_names:  # 遍历权重名称
            if weight_name in name:  # 如果名称包含
                spec_layer_weight = True  # 标记为特殊权重
                break  # 跳出
        if not spec_layer_weight:  # 如果不是特殊权重
            # treat rest weights as weights for transformer layer block
            name = name.replace(  # 添加mtp_block前缀
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )

        # NEW: drop "layers.<idx>." from the rewritten name (minimal change).
        layers_prefix = f"model.layers.{spec_layer}."  # 层前缀
        if name.startswith(layers_prefix):  # 如果有层前缀
            name = name.replace(layers_prefix, "model.", 1)  # 替换为model前缀

        return name  # 返回重写后的名称


EntryClass = [Step3p5MTP]  # 入口类列表
