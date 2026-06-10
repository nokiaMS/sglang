# InternS1 Pro 多模态图像处理器模块
# 本模块实现了 InternS1 Pro 模型的多模态数据处理逻辑，
# 继承自 QwenVL 图像处理器，支持图像、视频和音频输入，
# 并包含性能计时日志记录功能。
import time  # 导入时间模块
from typing import List, Union  # 导入类型提示模块

from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举类
    MultimodalDataItem,  # 多模态数据项类
    MultimodalProcessorOutput,  # 多模态处理器输出类
)
from sglang.srt.models.interns1pro import InternS1ProForConditionalGeneration  # 导入 InternS1 Pro 条件生成模型类
from sglang.srt.multimodal.processors.qwen_vl import (  # 导入 Qwen VL 处理器相关模块
    QwenVLImageProcessor,  # Qwen VL 图像处理器类
    preprocess_video,  # 视频预处理函数
)
from sglang.utils import logger  # 导入日志记录器


class InternS1_1ImageProcessor(QwenVLImageProcessor):  # InternS1_1 图像处理器类，继承自 Qwen VL 图像处理器
    models = [  # 支持的模型列表
        InternS1ProForConditionalGeneration,  # InternS1 Pro 模型
    ]

    def get_mm_data(self, prompt, embeddings, img_grid_thw):  # 获取多模态数据，构建输入 ID 和多模态项
        input_ids, offsets = self.build_input_ids(prompt, img_grid_thw)  # 构建输入 ID 和偏移量

        mm_items = [  # 创建多模态数据项列表
            MultimodalDataItem(  # 创建多模态数据项
                modality=Modality.IMAGE,  # 模态类型为图像
                offsets=offsets,  # 偏移量
                precomputed_embeddings=embeddings,  # 预计算的嵌入
            )
        ]

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids,  # 输入 ID
            mm_items=mm_items,  # 多模态项
            im_start_id=self.IM_START_TOKEN_ID,  # 图像起始标记 ID
            im_end_id=self.IM_END_TOKEN_ID,  # 图像结束标记 ID
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记 token ID
            video_token_id=self.mm_tokens.video_token_id,  # 视频标记 token ID
            audio_token_id=self.mm_tokens.audio_token_id,  # 音频标记 token ID
        )

    async def process_mm_data_async(  # 异步处理多模态数据，支持图像、视频和音频
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        entry_time = time.perf_counter()  # 记录入口时间
        base_output = await self.load_mm_data(  # 异步加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            video_data=request_obj.video_data,  # 视频数据
            audio_data=request_obj.audio_data,  # 音频数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊标记
        )
        load_time = time.perf_counter()  # 记录加载完成时间
        rid = getattr(request_obj, "rid", "anonymous_rid")  # 获取请求 ID

        video_metadata = None  # 初始化视频元数据
        if base_output.videos:  # 如果有视频数据
            videos_processed = [  # 预处理所有视频
                await preprocess_video(video, video_config=self.video_config)  # 异步预处理视频
                for video in base_output.videos  # 遍历所有视频
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))  # 分离视频数据和元数据

        preprocess_time = time.perf_counter()  # 记录预处理完成时间

        mm_items, input_ids, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output,  # 基础输出
            self.mm_tokens,  # 多模态标记
            video_metadata=video_metadata,  # 视频元数据
            do_sample_frames=False,  # 不进行帧采样
        )

        second_per_grid_ts = getattr(ret, "second_per_grid_ts", None)  # 获取每网格时间戳
        if second_per_grid_ts is None:  # 如果没有获取到
            second_per_grid_ts = getattr(ret, "video_second_per_grid", None)  # 尝试获取视频每秒网格数

        process_time = time.perf_counter()  # 记录处理完成时间

        input_ids = input_ids.flatten()  # 将输入 ID 展平为一维

        image_grid_thw = None  # 初始化图像网格信息
        if hasattr(ret, "image_grid_thw"):  # 如果结果中有图像网格信息
            image_grid_thw = ret.image_grid_thw  # 获取图像网格信息

        if image_grid_thw is None and image_data and isinstance(image_data[0], dict):  # 如果图像网格信息为空且图像数据是字典
            image_grid_thw = image_data[0].get("image_grid_thw")  # 从图像数据中获取网格信息

        video_grid_thw = None  # 初始化视频网格信息
        if hasattr(ret, "video_grid_thw"):  # 如果结果中有视频网格信息
            video_grid_thw = ret.video_grid_thw  # 获取视频网格信息

        if video_grid_thw is None and request_obj.video_data:  # 如果视频网格信息为空且有视频数据
            first_video = request_obj.video_data[0]  # 获取第一个视频
            if isinstance(first_video, dict):  # 如果视频是字典类型
                video_grid_thw = first_video.get("video_grid_thw")  # 从视频数据中获取网格信息

        get_rope_index_time = time.perf_counter()  # 记录获取位置编码索引完成时间

        logger.debug(  # 输出性能调试日志
            f"[QwenVLProcessor Perf] {rid=}, "  # 请求 ID
            f"load_time: {(load_time - entry_time) * 1000:.2f} ms, "  # 加载耗时
            f"preprocess_time: {(preprocess_time - load_time) * 1000:.2f} ms, "  # 预处理耗时
            f"process_time: {(process_time - preprocess_time) * 1000:.2f} ms, "  # 处理耗时
            f"get_rope_index_time: {(get_rope_index_time - process_time) * 1000:.2f} ms, "  # 获取位置编码耗时
            f"total_time: {(get_rope_index_time - entry_time) * 1000:.2f} ms"  # 总耗时
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入 ID 列表
            mm_items=mm_items,  # 多模态项
            im_start_id=self.vision_start_token_id,  # 视觉起始标记 ID
            im_end_id=self.vision_end_token_id,  # 视觉结束标记 ID
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记 token ID
            video_token_id=self.mm_tokens.video_token_id,  # 视频标记 token ID
            audio_token_id=self.mm_tokens.audio_token_id,  # 音频标记 token ID
        )
