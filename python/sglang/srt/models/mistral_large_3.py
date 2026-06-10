# Mistral Large 3 模型实现
# 本文件实现了基于DeepseekV3架构的Mistral Large 3模型，
# 支持从Mistral原生格式到DeepseekV2/HuggingFace格式的权重名称映射。
# 主要用于MLA（Multi-head Latent Attention）架构的Mistral Large 3推理。

# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mistral_large_3.py  # 适配自vLLM项目的Mistral Large 3实现
# SPDX-License-Identifier: Apache-2.0  # Apache 2.0许可证
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # vLLM项目贡献者版权声明
from collections.abc import Iterable  # 导入可迭代类型

import regex as re  # 导入正则表达式库（用于权重名称重映射）
import torch  # 导入PyTorch

from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM  # 导入DeepseekV3因果语言模型基类


class MistralLarge3ForCausalLM(DeepseekV3ForCausalLM):  # Mistral Large 3因果语言模型，继承自DeepseekV3
    # fmt: off  # 关闭格式化
    remapping = {  # Mistral原生格式到DeepseekV2/HuggingFace格式的权重名称映射表
        r"layers\.(\d+)\.attention_norm\.weight": r"model.layers.\1.input_layernorm.weight",  # noqa: E501  # 注意力层归一化权重映射
        r"layers\.(\d+)\.attention\.wq\.(\w+)": r"model.layers.\1.self_attn.q_proj.\2",  # noqa: E501  # 查询投影权重映射
        r"layers\.(\d+)\.attention\.wq_a\.(\w+)": r"model.layers.\1.self_attn.q_a_proj.\2",  # noqa: E501  # 查询低秩投影a权重映射
        r"layers\.(\d+)\.attention\.q_a_norm\.weight": r"model.layers.\1.self_attn.q_a_layernorm.weight",  # noqa: E501  # 查询低秩投影a归一化权重映射
        r"layers\.(\d+)\.attention\.wq_b\.(\w+)": r"model.layers.\1.self_attn.q_b_proj.\2",  # noqa: E501  # 查询低秩投影b权重映射
        r"layers\.(\d+)\.attention\.wkv_a_with_mqa\.(\w+)": r"model.layers.\1.self_attn.kv_a_proj_with_mqa.\2",  # noqa: E501  # KV低秩投影a（多查询注意力）权重映射
        r"layers\.(\d+)\.attention\.kv_a_norm\.weight": r"model.layers.\1.self_attn.kv_a_layernorm.weight",  # noqa: E501  # KV低秩投影a归一化权重映射
        r"layers\.(\d+)\.attention\.wkv_b\.(\w+)": r"model.layers.\1.self_attn.kv_b_proj.\2",  # noqa: E501  # KV低秩投影b权重映射
        r"layers\.(\d+)\.attention\.wo\.(\w+)": r"model.layers.\1.self_attn.o_proj.\2",  # noqa: E501  # 输出投影权重映射
        # FP8 scales  # FP8量化缩放因子
        r"layers\.(\d+)\.attention\.k_fake_quantizer\.qscale_act": r"model.layers.\1.self_attn.mla_attn.mla_attn.k_scale",  # noqa: E501  # K的FP8激活缩放因子映射
        r"layers\.(\d+)\.attention\.q_fake_quantizer\.qscale_act": r"model.layers.\1.self_attn.mla_attn.mla_attn.q_scale",  # noqa: E501  # Q的FP8激活缩放因子映射
        r"layers\.(\d+)\.attention\.v_fake_quantizer\.qscale_act": r"model.layers.\1.self_attn.mla_attn.mla_attn.v_scale",  # noqa: E501  # V的FP8激活缩放因子映射
        r"layers\.(\d+)\.ffn_norm\.weight": r"model.layers.\1.post_attention_layernorm.weight",  # noqa: E501  # FFN层归一化权重映射
        r"layers\.(\d+)\.feed_forward\.w1\.(\w+)": r"model.layers.\1.mlp.gate_proj.\2",  # noqa: E501  # 前馈网络门控投影权重映射
        r"layers\.(\d+)\.feed_forward\.w2\.(\w+)": r"model.layers.\1.mlp.down_proj.\2",  # noqa: E501  # 前馈网络下投影权重映射
        r"layers\.(\d+)\.feed_forward\.w3\.(\w+)": r"model.layers.\1.mlp.up_proj.\2",  # noqa: E501  # 前馈网络上投影权重映射
        r"layers\.(\d+)\.gate\.weight": r"model.layers.\1.mlp.gate.weight",  # noqa: E501  # MoE门控权重映射
        r"layers\.(\d+)\.shared_experts\.w1\.(\w+)": r"model.layers.\1.mlp.shared_experts.gate_proj.\2",  # noqa: E501  # 共享专家门控投影权重映射
        r"layers\.(\d+)\.shared_experts\.w2\.(\w+)": r"model.layers.\1.mlp.shared_experts.down_proj.\2",  # noqa: E501  # 共享专家下投影权重映射
        r"layers\.(\d+)\.shared_experts\.w3\.(\w+)": r"model.layers.\1.mlp.shared_experts.up_proj.\2",  # noqa: E501  # 共享专家上投影权重映射
        r"layers\.(\d+)\.experts\.(\d+)\.w1\.(\w+)": r"model.layers.\1.mlp.experts.\2.gate_proj.\3",  # noqa: E501  # 专家门控投影权重映射
        r"layers\.(\d+)\.experts\.(\d+)\.w2\.(\w+)": r"model.layers.\1.mlp.experts.\2.down_proj.\3",  # noqa: E501  # 专家下投影权重映射
        r"layers\.(\d+)\.experts\.(\d+)\.w3\.(\w+)": r"model.layers.\1.mlp.experts.\2.up_proj.\3",  # noqa: E501  # 专家上投影权重映射
        r"layers\.(\d+)\.router_biases": r"model.layers.\1.mlp.gate.e_score_correction_bias",  # noqa: E501  # 路由偏置权重映射
        r"norm\.weight": "model.norm.weight",  # noqa: E501  # 最终归一化权重映射
        r"tok_embeddings\.weight": "model.embed_tokens.weight",  # noqa: E501  # 词嵌入权重映射
        r"output\.weight": "lm_head.weight",  # noqa: E501  # 输出头权重映射
    }
    # fmt: on  # 恢复格式化

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:  # 加载模型权重，将Mistral格式映射为DeepseekV2格式
        return super().load_weights(self._iterable_remap_mistral_to_ds(weights))  # 先重映射权重名称再调用父类加载

    def _iterable_remap_mistral_to_ds(  # 将Mistral参数名迭代重映射为DeepseekV2参数名
        self, weights: Iterable[tuple[str, torch.Tensor]]  # 权重迭代器
    ) -> Iterable[tuple[str, torch.Tensor]]:  # 返回重映射后的权重迭代器
        """Remap Mistral parameters to DeepseekV2 parameters."""  # 将Mistral参数重映射为DeepseekV2参数
        for name, loaded_weight in weights:  # 遍历所有权重
            for k, v in self.remapping.items():  # 遍历映射规则
                match = re.fullmatch(k, name)  # 尝试完整匹配权重名称
                if match:  # 如果匹配成功
                    name = re.sub(k, v, name)  # 替换为映射后的名称
                    break  # 跳出映射规则循环
            else:  # 如果没有任何映射规则匹配
                import logging  # 导入日志模块

                logging.warning(f"Unrecognized weight: {name}. Skipping.")  # 记录无法识别的权重并跳过
                continue  # 跳过此权重

            # Note(Andy): Unlike Llama, this implementation uses  # 注意：与Llama不同，此实现使用
            # is_neox_style=False for RoPE, which matches Mistral's implementation.  # is_neox_style=False的RoPE，与Mistral的实现一致
            # Thus we don't need to permute the q/k weights (unlike Llama)  # 因此不需要对q/k权重进行排列（与Llama不同）

            # Remapping scale names. We could do this in the regex above but it  # 重映射缩放因子名称。可以在上面的正则中处理，但
            # would triple the number of lines for most layers.  # 那样会使大多数层的行数增加三倍。
            if name.endswith(".qscale_act"):  # 如果名称以.qscale_act结尾（激活缩放因子）
                name = re.sub(r"\.qscale_act$", ".input_scale", name)  # 替换为.input_scale
            elif name.endswith(".qscale_weight"):  # 如果名称以.qscale_weight结尾（权重缩放因子）
                name = re.sub(r"\.qscale_weight$", ".weight_scale", name)  # 替换为.weight_scale

            yield name, loaded_weight  # 产出重映射后的权重名称和张量


EntryClass = MistralLarge3ForCausalLM  # 模型注册入口类
