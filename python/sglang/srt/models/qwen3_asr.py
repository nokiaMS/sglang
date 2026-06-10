# Qwen3-ASR 语音识别模型实现
# 本文件实现了 Qwen3-ASR 自动语音识别模型，结合了音频编码器和 Qwen3 语言模型，
# 支持音频输入的语音识别和转写任务。
"""Qwen3-ASR model compatible with HuggingFace weights"""  # 兼容 HuggingFace 权重的 Qwen3-ASR 模型

import logging  # 导入日志模块
from typing import Any, Iterable, List, Optional, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 框架
import torch.nn as nn  # 导入神经网络模块

from sglang.srt.configs.qwen3_asr import Qwen3ASRConfig  # 导入 Qwen3-ASR 配置
from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig  # 导入 Qwen3 全能音频编码器配置
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态类型
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.qwen3 import Qwen3ForCausalLM  # 导入 Qwen3 因果语言模型
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder  # 导入 Qwen3 全能音频编码器
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

logger = logging.getLogger(__name__)  # 获取日志记录器


class Qwen3ASRForConditionalGeneration(nn.Module):
    """Qwen3-ASR 条件生成模型，整合音频编码器和语言模型"""

    default_bitsandbytes_target_modules = [  # 默认 BitandBytes 目标模块
        ".gate_proj.",  # gate 投影
        ".down_proj.",  # 下投影
        ".up_proj.",  # 上投影
        ".q_proj.",  # Q 投影
        ".k_proj.",  # K 投影
        ".v_proj.",  # V 投影
        ".o_proj.",  # O 投影
    ]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes 堆叠参数映射
        "q_proj": ("qkv_proj", 0),  # Q 映射
        "k_proj": ("qkv_proj", 1),  # K 映射
        "v_proj": ("qkv_proj", 2),  # V 映射
        "gate_proj": ("gate_up_proj", 0),  # gate 映射
        "up_proj": ("gate_up_proj", 1),  # up 映射
    }

    def __init__(
        self,
        config: Qwen3ASRConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化 Qwen3-ASR 条件生成模型"""
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        thinker_config = config.thinker_config  # 获取思考器配置

        if getattr(thinker_config, "audio_config", None) is None:  # 如果没有音频配置
            thinker_config.audio_config = Qwen3OmniMoeAudioEncoderConfig()  # 创建默认音频编码器配置

        self.audio_tower = Qwen3OmniMoeAudioEncoder(thinker_config.audio_config)  # 创建音频编码器
        self.language_model = Qwen3ForCausalLM(  # 创建 Qwen3 语言模型
            thinker_config.text_config,  # 文本配置
            quant_config,  # 量化配置
            prefix=add_prefix("language_model", prefix),  # 参数前缀
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建多模态填充模式

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """使用多模态 token 填充模式对输入 ID 进行填充"""
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)  # 返回填充后的 token

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """从多模态数据项中提取音频特征"""
        device = next(self.audio_tower.parameters()).device  # 获取音频编码器设备

        input_features = (  # 拼接音频特征
            torch.cat([item.feature for item in items])  # 拼接所有特征
            .type(self.audio_tower.dtype)  # 转换数据类型
            .to(device)  # 移到编码器设备
        )

        has_mask = all(  # 检查是否所有项都有注意力掩码
            getattr(item, "feature_attention_mask", None) is not None for item in items  # 检查每项
        )

        if has_mask:  # 如果有掩码
            feature_attention_mask = (  # 拼接注意力掩码
                torch.cat([item.feature_attention_mask for item in items], dim=0)  # 拼接
                .type(torch.long)  # 转换为 long 类型
                .to(device)  # 移到设备
            )
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)  # 计算有效特征长度
            input_features = input_features.permute(0, 2, 1)[  # 转置并用掩码过滤
                feature_attention_mask.bool()  # 使用布尔掩码
            ].permute(1, 0)  # 转回原维度
        else:  # 没有掩码
            audio_feature_lengths = torch.tensor(  # 使用全部特征长度
                [input_features.shape[-1]] * input_features.shape[0],  # 每个样本的特征长度
                dtype=torch.long,  # long 类型
                device=device,  # 设备
            )
            input_features = input_features.permute(0, 2, 1).reshape(  # 转置并重塑
                -1, input_features.shape[1]  # 合并批次和特征维度
            )

        audio_outputs = self.audio_tower(  # 通过音频编码器
            input_features,  # 输入特征
            feature_lens=audio_feature_lengths,  # 特征长度
        )
        return audio_outputs.last_hidden_state  # 返回最后一层隐藏状态

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Qwen3-ASR 前向传播：处理音频输入并生成隐藏状态"""
        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入例程
            input_ids=input_ids,  # 输入 ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.language_model,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数映射
                Modality.AUDIO: self.get_audio_feature,  # 音频模态对应的特征提取函数
            },
            positions=positions,  # 位置信息
        )
        return hidden_states  # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，处理思考器前缀映射和音频塔特殊参数"""
        llm_stacked_params = [  # 语言模型堆叠参数映射
            ("qkv_proj", "q_proj", "q"),  # Q 映射
            ("qkv_proj", "k_proj", "k"),  # K 映射
            ("qkv_proj", "v_proj", "v"),  # V 映射
            ("gate_up_proj", "gate_proj", 0),  # gate 映射
            ("gate_up_proj", "up_proj", 1),  # up 映射
        ]
        # Audio tower has separate q/k/v in checkpoint → stack into qkv_proj  # 音频塔检查点中 Q/K/V 分开，需堆叠
        audio_stacked_params = [  # 音频堆叠参数映射
            ("qkv_proj", "q_proj", "q"),  # Q 映射
            ("qkv_proj", "k_proj", "k"),  # K 映射
            ("qkv_proj", "v_proj", "v"),  # V 映射
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))  # 获取参数字典

        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入逆频率
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的余弦/正弦
                continue

            if (  # 如果共享词嵌入
                getattr(
                    self.config.thinker_config.text_config, "tie_word_embeddings", False  # 获取共享词嵌入标志
                )
                and "lm_head.weight" in name  # 且是语言模型头权重
            ):
                continue

            if "talker" in name or "code2wav" in name:  # 跳过 talker 和 code2wav 权重
                continue

            if name.startswith("thinker.audio_tower."):  # 映射音频塔前缀
                name = name.replace("thinker.audio_tower.", "audio_tower.", 1)  # 替换前缀
            elif name.startswith("thinker.lm_head."):  # 映射语言模型头前缀
                name = name.replace("thinker.lm_head.", "language_model.lm_head.", 1)  # 替换前缀
            elif name.startswith("thinker.model."):  # 映射模型前缀
                name = name.replace("thinker.model.", "language_model.model.", 1)  # 替换前缀

            is_audio = "audio_tower" in name  # 判断是否为音频塔权重

            # Audio tower: remap out_proj → proj for VisionAttention  # 音频塔：将 out_proj 重映射为 proj
            if is_audio and "out_proj" in name:  # 如果是音频塔且包含 out_proj
                name = name.replace("out_proj", "proj")  # 替换为 proj

            stacked_params = audio_stacked_params if is_audio else llm_stacked_params  # 选择对应的堆叠参数映射

            for param_name, weight_name, shard_id in stacked_params:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue
                name_tmp = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                if name_tmp.endswith(".bias") and name_tmp not in params_dict:  # 跳过不存在的偏置
                    continue
                if name_tmp not in params_dict:  # 如果参数不在字典中
                    continue
                param = params_dict[name_tmp]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break
            else:  # 非堆叠参数处理
                if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置
                    continue
                if name not in params_dict:  # 如果参数不在字典中
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = Qwen3ASRForConditionalGeneration  # 模型入口类
