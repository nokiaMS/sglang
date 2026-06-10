# NVILA多模态处理器模块
# 实现NVILA、NVILA-Lite和JetVLM视觉语言模型的多模态数据处理
# 支持图像和视频输入，使用基类的load_mm_data + process_and_combine_mm_data流程
from typing import Any  # 导入类型提示

import torch.nn as nn  # 导入PyTorch神经网络模块
from transformers.configuration_utils import PretrainedConfig  # 导入预训练配置基类
from transformers.processing_utils import ProcessorMixin  # 导入处理器混合类
from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # 导入分词器基类

from sglang.srt.managers.io_struct import GenerateReqInput  # 导入生成请求输入结构
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.jet_vlm import JetVLMForConditionalGeneration  # 导入JetVLM模型
from sglang.srt.models.nvila import NVILAForConditionalGeneration  # 导入NVILA模型
from sglang.srt.models.nvila_lite import NVILALiteForConditionalGeneration  # 导入NVILA-Lite模型
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器和特殊令牌类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.server_args import ServerArgs  # 导入服务器参数类

NUM_VIDEO_FRAMES = 8  # 视频默认采样帧数


class NVILAMultimodalProcessor(BaseMultimodalProcessor):  # NVILA多模态处理器类
    models: list[type[nn.Module]] = [  # 关联的模型列表
        NVILAForConditionalGeneration,  # NVILA模型
        NVILALiteForConditionalGeneration,  # NVILA-Lite模型
        JetVLMForConditionalGeneration,  # JetVLM模型
    ]

    def __init__(  # 初始化NVILA多模态处理器
        self,
        hf_config: PretrainedConfig,  # HF配置
        server_args: ServerArgs,  # 服务器参数
        _processor: ProcessorMixin,  # HF处理器
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ) -> None:
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化

        self._processor: ProcessorMixin  # 类型标注处理器

        tokenizer: PreTrainedTokenizerBase = getattr(self._processor, "tokenizer")  # 获取分词器

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊令牌
            image_token=tokenizer.image_token,  # 图像令牌
            image_token_id=hf_config.image_token_id,  # 图像令牌ID
            video_token=tokenizer.video_token,  # 视频令牌
            video_token_id=hf_config.video_token_id,  # 视频令牌ID
        ).build(_processor)  # 构建令牌映射

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data,  # 图像数据
        audio_data,  # 音频数据
        input_text,  # 输入文本
        request_obj: GenerateReqInput,  # 请求对象
        **kwargs,  # 关键字参数
    ) -> dict[str, Any] | None:
        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 输入提示文本
            multimodal_tokens=self.mm_tokens,  # 多模态特殊令牌
            image_data=request_obj.image_data,  # type: ignore  # 图像数据
            video_data=request_obj.video_data,  # type: ignore  # 视频数据
        )

        for i, video in enumerate(base_output.videos):  # type: ignore  # 遍历视频数据
            base_output.videos[i] = [x.asnumpy() for x in video]  # type: ignore  # 将视频帧转为numpy数组

        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output,  # 基础输出
            self.mm_tokens,  # 多模态特殊令牌
            do_sample_frames=True,  # 启用帧采样
            num_frames=NUM_VIDEO_FRAMES,  # 采样帧数
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入ID列表
            mm_items=mm_items,  # 多模态数据项
            im_token_id=self.mm_tokens.image_token_id,  # 图像令牌ID
            video_token_id=self.mm_tokens.video_token_id,  # 视频令牌ID
        )
