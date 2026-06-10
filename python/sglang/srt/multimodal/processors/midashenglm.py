# MiDashengLM多模态处理器模块
# 实现MiDashengLM音频语言模型的多模态数据处理
# 支持音频输入的处理和Mel频谱特征提取
import logging  # 导入日志模块
import re  # 导入正则表达式模块

import torch  # 导入PyTorch

from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput  # 导入模态枚举和处理器输出类
from sglang.srt.models.midashenglm import MiDashengLMModel  # 导入MiDashengLM模型
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器和相关类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)  # 获取模块日志器


class MiDashengLMMultimodalProcessor(BaseMultimodalProcessor):  # MiDashengLM多模态处理器类
    """Multimodal processor for MiDashengLM audio-language model."""  # MiDashengLM音频语言模型的多模态处理器

    models = [MiDashengLMModel]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化MiDashengLM多模态处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化

        self.AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"  # 音频令牌字符串
        self.AUDIO_TOKEN_REGEX = re.compile(  # 编译音频令牌正则表达式
            r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>"  # 匹配音频开始/内容/结束
        )

        tokenizer = self._processor.tokenizer  # 获取分词器
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")  # 音频开始令牌ID
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")  # 音频内容令牌ID
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")  # 音频结束令牌ID

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊令牌
            audio_token=self.AUDIO_TOKEN,  # 音频令牌字符串
            audio_token_regex=self.AUDIO_TOKEN_REGEX,  # 音频令牌正则
            audio_token_id=self.audio_token_id,  # 音频令牌ID
        ).build(_processor)  # 构建令牌映射

        self.ATTR_NAME_TO_MODALITY.update(  # 更新属性名到模态的映射
            {
                "input_values": Modality.AUDIO,  # 输入值映射到音频模态
                "audio_length": Modality.AUDIO,  # 音频长度映射到音频模态
            }
        )

        if "input_values" not in self.FEATURE_NAMES:  # 如果特征名中没有input_values
            self.FEATURE_NAMES.append("input_values")  # 添加input_values特征名

    def process_mm_data(  # 处理多模态数据（同步方法）
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        """Override to use correct audio parameter name for MiDashengLM processor."""  # 重写以使用MiDashengLM处理器正确的音频参数名
        if images:  # 如果有图像数据
            kwargs["images"] = images  # 传递图像
        if videos:  # 如果有视频数据
            kwargs["videos"] = videos  # 传递视频
        if audios:  # 如果有音频数据
            kwargs["audio"] = audios  # 注意：参数名为audio而非audios
            kwargs.setdefault("audio_kwargs", {})  # 设置默认音频参数字典
            kwargs["audio_kwargs"].setdefault("truncation", False)  # 默认不截断
            if self.audio_config:  # 如果有音频配置
                kwargs["audio_kwargs"].update(self.audio_config)  # 更新音频参数

        processor = self._processor  # 获取HF处理器
        result = processor.__call__(  # 调用处理器
            text=[input_text],  # 文本输入
            padding=True,  # 启用填充
            return_tensors="pt",  # 返回PyTorch张量
            **kwargs,  # 其他参数
        )

        if not getattr(self.server_args, "keep_mm_feature_on_device", False):  # 如果不保留特征在设备上
            for feature_name in ["input_values"]:  # 遍历需要移动的特征
                if feature_name in result:  # 如果结果中有该特征
                    result[feature_name] = result[feature_name].cpu()  # 移到CPU

        return result  # 返回处理结果

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        audio_data,  # 音频数据
        input_text,  # 输入文本
        **kwargs,  # 其他关键字参数
    ):
        """Process audio data for MiDashengLM model.
        # 处理MiDashengLM模型的音频数据

        Args:
            audio_data: Audio input data  # 音频输入数据
            input_text: Text prompt  # 文本提示
            **kwargs: Additional arguments  # 额外参数

        Returns:
            Dictionary containing processed multimodal data  # 包含处理后多模态数据的字典
        """
        logger.info("=" * 80)  # 打印分隔线
        logger.info("process_mm_data_async called")  # 记录方法被调用
        logger.info(f"audio_data is not None: {audio_data is not None}")  # 记录音频数据是否非空
        logger.info(f"input_text: {input_text}")  # 记录输入文本
        logger.info("=" * 80)  # 打印分隔线

        if audio_data and not self.AUDIO_TOKEN_REGEX.search(input_text):  # 如果有音频但文本中没有音频令牌
            input_text = f"{self.AUDIO_TOKEN}{input_text}"  # 在文本前自动添加音频令牌
            logger.info("Auto-prepended audio token")  # 记录自动添加了音频令牌

        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 输入提示文本
            audio_data=audio_data,  # 音频数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊令牌
        )
        if base_output is None:  # 如果基础输出为空
            logger.info("base_output is None")  # 记录输出为空
            return None  # 返回None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 基础输出和特殊令牌
        )
        logger.info(f"mm_items count: {len(mm_items)}")  # 记录多模态项数量
        logger.info(f"ret keys: {list(ret.keys())}")  # 记录返回结果的键
        logger.info(f"input_ids shape: {input_ids.shape}")  # 记录输入ID形状
        logger.info(  # 记录音频令牌ID信息
            f"audio_token_id={self.audio_token_id}, audio_start_id={self.audio_start_id}, audio_end_id={self.audio_end_id}"
        )
        logger.info(  # 记录音频令牌在输入ID中的数量
            f"Count of audio_token_id in input_ids: {(input_ids == self.audio_token_id).sum().item()}"
        )
        for i, item in enumerate(mm_items):  # 遍历每个多模态项
            logger.info(f"mm_item[{i}] modality: {item.modality}")  # 记录模态类型
            logger.info(  # 记录填充值
                f"mm_item[{i}] pad_value: {getattr(item, 'pad_value', 'NOT SET')}"
            )
            logger.info(f"mm_item[{i}] offsets: {getattr(item, 'offsets', 'NOT SET')}")  # 记录偏移
            logger.info(f"mm_item[{i}] has feature: {hasattr(item, 'feature')}")  # 记录是否有特征
            if hasattr(item, "feature") and item.feature is not None:  # 如果有特征
                logger.info(f"mm_item[{i}] feature shape: {item.feature.shape}")  # 记录特征形状

        if "audio_length" in ret and len(mm_items) > 0:  # 如果结果中有音频长度且有多模态项
            audio_length = ret["audio_length"]  # 获取音频长度
            if isinstance(audio_length, torch.Tensor):  # 如果是张量
                audio_length = (  # 转换为标量
                    audio_length.item()  # 单元素张量取值
                    if audio_length.numel() == 1  # 如果只有一个元素
                    else audio_length[0].item()  # 否则取第一个元素
                )
            mm_items[0].audio_length = audio_length  # 设置第一个多模态项的音频长度
            logger.info(  # 记录音频长度来源
                f"Set audio_length={audio_length} (from processor, mel frame count)"
            )
        elif "input_values" in ret and len(mm_items) > 0:  # 如果没有audio_length但有input_values
            input_values = ret["input_values"]  # 获取输入值
            audio_length = (  # 计算音频长度
                input_values.shape[-1]  # 多维取最后一个维度
                if input_values.ndim >= 2  # 如果至少2维
                else input_values.shape[0]  # 1维取第一个维度
            )
            mm_items[0].audio_length = audio_length  # 设置音频长度
            logger.info(f"Set audio_length={audio_length} (fallback, waveform length)")  # 记录回退方式

        result = MultimodalProcessorOutput(  # 创建多模态处理器输出
            mm_items=mm_items,  # 多模态数据项
            input_ids=input_ids.tolist(),  # 输入ID列表
            audio_start_id=self.audio_start_id,  # 音频开始令牌ID
            audio_token_id=self.audio_token_id,  # 音频令牌ID
            audio_end_id=self.audio_end_id,  # 音频结束令牌ID
        )
        logger.info(f"Returning {len(result.mm_items)} mm_items")  # 记录返回的多模态项数量
        return result  # 返回处理结果
