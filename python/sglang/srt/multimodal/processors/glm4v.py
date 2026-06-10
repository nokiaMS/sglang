# GLM-4V 多模态图像处理器模块
# 本模块实现了 GLM-4V 模型的图像和视频数据处理逻辑，
# 包括特殊标记的定义、MRotaryEmbedding 位置编码的计算，
# 以及图像和视频数据的异步加载与处理。
from typing import List, Union  # 导入类型提示模块

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding  # 导入多维旋转位置编码模块
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration  # 导入 GLM-4V 条件生成模型类
from sglang.srt.models.glm4v_moe import Glm4vMoeForConditionalGeneration  # 导入 GLM-4V MoE 条件生成模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器相关类
    BaseMultimodalProcessor as SGLangBaseProcessor,  # 将基础处理器重命名为 SGLangBaseProcessor
)
from sglang.srt.multimodal.processors.base_processor import (  # 导入多模态特殊标记类
    MultimodalSpecialTokens,  # 多模态特殊标记类
)

try:  # 尝试导入 GLM OCR 模型
    from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration  # 导入 GLM OCR 条件生成模型类
except ImportError:  # 如果导入失败
    GlmOcrForConditionalGeneration = None  # 设置为 None


class Glm4vImageProcessor(SGLangBaseProcessor):  # GLM-4V 图像处理器类
    models = [  # 支持的模型列表，过滤掉无法导入的模型
        m  # 模型类
        for m in [  # 遍历可能的模型类
            Glm4vForConditionalGeneration,  # GLM-4V 模型
            Glm4vMoeForConditionalGeneration,  # GLM-4V MoE 模型
            GlmOcrForConditionalGeneration,  # GLM OCR 模型
        ]
        if m is not None  # 过滤掉为 None 的模型
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化 GLM-4V 图像处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化方法

        # GLM-V specific tokens
        self.IMAGE_TOKEN = "ख़"  # GLM-V 专用图像标记
        self.VIDEO_TOKEN = "ग़"  # GLM-V 专用视频标记
        self.IMAGE_START_TOKEN = "🌦"  # 图像起始标记
        self.IMAGE_END_TOKEN = "cleta"  # 图像结束标记
        self.VIDEO_START_TOKEN = " clima"  # 视频起始标记
        self.VIDEO_END_TOKEN = "tó"  # 视频结束标记

        # Token IDs
        self.IM_TOKEN_ID = hf_config.image_token_id  # 图像标记 token ID
        self.VIDEO_TOKEN_ID = hf_config.video_token_id  # 视频标记 token ID
        self.IMAGE_START_TOKEN_ID = hf_config.image_start_token_id  # 图像起始标记 token ID
        self.IMAGE_END_TOKEN_ID = hf_config.image_end_token_id  # 图像结束标记 token ID
        self.VIDEO_START_TOKEN_ID = hf_config.video_start_token_id  # 视频起始标记 token ID
        self.VIDEO_END_TOKEN_ID = hf_config.video_end_token_id  # 视频结束标记 token ID

        # Vision config
        self.IMAGE_FACTOR = 28  # 图像因子，用于尺寸对齐
        self.MIN_PIXELS = 112 * 112  # 最小像素数
        self.MAX_PIXELS = 30000 * 28 * 28 * 2  # 最大像素数

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token=self.IMAGE_TOKEN,  # 图像标记
            image_token_id=self.IM_TOKEN_ID,  # 图像标记 ID
            video_token=self.VIDEO_TOKEN,  # 视频标记
            # Note: For GLM4v videos, it uses the video token before tokenization but uses image token after tokenization
            video_token_id=self.IM_TOKEN_ID,  # 视频标记在分词后使用图像标记 ID
        ).build(_processor)  # 使用处理器构建标记

    def compute_mrope_positions(self, input_ids, mm_items):  # 计算 GLM-4V 专用的多维旋转位置编码
        image_grid_thw = None  # 初始化图像网格信息
        video_grid_thw = None  # 初始化视频网格信息
        for item in mm_items:  # 遍历多模态项
            if "image_grid_thw" in item.model_specific_data:  # 如果项中包含图像网格信息
                image_grid_thw = item.model_specific_data["image_grid_thw"]  # 获取图像网格信息
            if "video_grid_thw" in item.model_specific_data:  # 如果项中包含视频网格信息
                video_grid_thw = item.model_specific_data["video_grid_thw"]  # 获取视频网格信息

        import torch  # 导入 PyTorch 模块

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # 将输入 ID 转为张量并增加批次维度
        attention_mask = torch.ones_like(input_ids_tensor)  # 创建全 1 的注意力掩码
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(  # 调用 GLM-4V 专用的旋转位置编码计算
            input_ids=input_ids_tensor,  # 输入 ID 张量
            hf_config=self.hf_config,  # HuggingFace 配置
            image_grid_thw=image_grid_thw,  # 图像网格信息
            video_grid_thw=video_grid_thw,  # 视频网格信息
            attention_mask=attention_mask,  # 注意力掩码
        )
        return mrope_positions.squeeze(1), mrope_position_delta  # 返回位置编码和增量，去除多余维度

    async def process_mm_data_async(  # 异步处理多模态数据，支持图像和视频
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        base_output = await self.load_mm_data(  # 异步加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            video_data=request_obj.video_data,  # 视频数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊标记
        )

        if base_output.videos:  # 如果有视频数据
            base_output.videos = request_obj.video_data  # 使用原始视频数据
        mm_items, input_ids, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 传入基础输出和标记
        )

        input_ids = input_ids.flatten()  # 将输入 ID 展平为一维
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(  # 计算 GLM-4V 的旋转位置编码
            input_ids=input_ids.unsqueeze(0),  # 增加批次维度
            hf_config=self.hf_config,  # HuggingFace 配置
            image_grid_thw=getattr(ret, "image_grid_thw", None),  # 获取图像网格信息
            video_grid_thw=getattr(ret, "video_grid_thw", None),  # 获取视频网格信息
            attention_mask=getattr(ret, "attention_mask", None),  # 获取注意力掩码
        )
        mrope_positions = mrope_positions.squeeze(1)  # 去除多余的中间维度

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入 ID 列表
            mm_items=mm_items,  # 多模态项
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记 token ID
            video_token_id=self.mm_tokens.video_token_id,  # 视频标记 token ID
            mrope_positions=mrope_positions,  # 多维旋转位置编码
            mrope_position_delta=mrope_position_delta,  # 位置编码增量
        )
