# Qwen2-Audio多模态处理器模块
# 本模块为Qwen2-Audio模型提供音频数据处理功能
# 支持音频特征长度计算和嵌入切片

import re  # 导入正则表达式模块

from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalProcessorOutput,  # 多模态处理器输出
)
from sglang.srt.models.qwen2_audio import Qwen2AudioForConditionalGeneration  # 导入Qwen2-Audio模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器和特殊标记类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class Qwen2AudioMultimodalProcessor(BaseMultimodalProcessor):  # Qwen2-Audio多模态处理器，继承基础多模态处理器
    models = [Qwen2AudioForConditionalGeneration]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化Qwen2-Audio多模态处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化
        self.AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"  # 设置音频标记
        self.AUDIO_TOKEN_REGEX = re.compile(  # 编译音频标记的正则表达式
            r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>"  # 匹配音频开始、内容和结束标记
        )
        # Collect special token ids
        tokenizer = self._processor.tokenizer  # 获取分词器
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")  # 获取音频开始标记ID
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")  # 获取音频内容标记ID
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")  # 获取音频结束标记ID

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            audio_token=self.AUDIO_TOKEN,  # 音频标记
            audio_token_regex=self.AUDIO_TOKEN_REGEX,  # 音频标记正则表达式
            audio_token_id=self.audio_token_id,  # 音频标记ID
        ).build(_processor)  # 构建标记对象

        self.ATTR_NAME_TO_MODALITY.update({"feature_attention_mask": Modality.AUDIO})  # 将特征注意力掩码映射为音频模态

    def get_mm_data(self, prompt, embeddings, **kwargs):  # 获取多模态数据（用于转换器后端）
        audio_feature_lens = kwargs.get("audio_feature_lens", None)  # 获取音频特征长度

        # Convert audio_feature_lens to token counts for build_input_ids
        output_lengths = None  # 输出长度初始化
        input_lengths = None  # 输入长度初始化
        if audio_feature_lens is not None:  # 如果音频特征长度不为空
            if audio_feature_lens.dim() > 1:  # 如果维度大于1
                audio_feature_lens = audio_feature_lens.flatten()  # 展平为一维
            input_lengths = (audio_feature_lens - 1) // 2 + 1  # 计算输入长度
            output_lengths = (input_lengths - 2) // 2 + 1  # 计算输出长度

        input_ids, offsets, modality_list = self.build_input_ids(  # 构建输入ID
            prompt,  # 提示文本
            audio_seq_lens=output_lengths,  # 音频序列长度
        )

        mm_items = []  # 多模态数据项列表
        consumed_per_modality = {}  # 每种模态已消耗的嵌入数量

        for modality, offset in zip(modality_list, offsets):  # 遍历模态和偏移量
            num_tokens = offset[1] - offset[0] + 1  # 计算标记数量
            embedding_start = consumed_per_modality.get(modality, 0)  # 获取当前模态的嵌入起始位置
            embedding_slice = embeddings[modality][  # 获取嵌入切片
                embedding_start : embedding_start + num_tokens  # 从起始位置到起始位置+标记数
            ]
            consumed_per_modality[modality] = embedding_start + num_tokens  # 更新已消耗数量
            mm_items.append(  # 添加多模态数据项
                MultimodalDataItem(  # 创建数据项
                    modality=modality,  # 模态类型
                    offsets=[offset],  # 偏移量
                    precomputed_embeddings=embedding_slice,  # 预计算嵌入
                )
            )

        if mm_items:  # 如果存在多模态数据项
            mm_items[0].audio_feature_lens = output_lengths  # 设置第一个数据项的音频特征长度

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=mm_items,  # 多模态数据项
            input_ids=input_ids,  # 输入ID
            audio_start_id=self.audio_start_id,  # 音频开始标记ID
            audio_token_id=self.audio_token_id,  # 音频标记ID
            audio_end_id=self.audio_end_id,  # 音频结束标记ID
        )

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        audio_data,  # 音频数据
        input_text,  # 输入文本
        **kwargs,  # 其他关键字参数
    ):
        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 输入提示文本
            audio_data=audio_data,  # 音频数据
            multimodal_tokens=self.mm_tokens,  # 多模态标记
        )
        if base_output is None:  # 如果基础输出为None
            return None  # 返回None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 基础输出和多模态标记
        )

        assert (  # 断言特征注意力掩码存在
            "feature_attention_mask" in ret
        ), "feature_attention_mask not found in processor output"  # 特征注意力掩码未在处理器输出中找到
        input_lengths = ret["feature_attention_mask"].sum(dim=-1)  # 计算输入长度
        input_lengths = (input_lengths - 1) // 2 + 1  # 转换输入长度
        output_lengths = (input_lengths - 2) // 2 + 1  # 计算输出长度

        mm_items[0].audio_feature_lens = output_lengths  # 设置第一个数据项的音频特征长度

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=mm_items,  # 多模态数据项
            input_ids=input_ids.tolist(),  # 输入ID列表
            audio_start_id=self.audio_start_id,  # 音频开始标记ID
            audio_token_id=self.audio_token_id,  # 音频标记ID
            audio_end_id=self.audio_end_id,  # 音频结束标记ID
        )
