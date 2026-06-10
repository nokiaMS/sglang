# Kimi 通用网格多模态数据辅助模块
# 本模块提供了 Kimi 系列模型（KimiVL 和 KimiK2.5）共享的
# 基于网格的多模态数据处理辅助功能，包括图像 token 数量计算、
# 网格元数据解析和多模态数据构建等。
"""Kimi-specific grid-based multimodal data helpers.
Kimi 专用的基于网格的多模态数据辅助工具。

Shared by KimiVLImageProcessor and KimiK2_5VLImageProcessor.
由 KimiVLImageProcessor 和 KimiK2_5VLImageProcessor 共享。
"""

from typing import Union  # 导入联合类型提示

import numpy as np  # 导入 NumPy 模块
import torch  # 导入 PyTorch 模块

from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举类
    MultimodalDataItem,  # 多模态数据项类
    MultimodalProcessorOutput,  # 多模态处理器输出类
)


class KimiGridMMDataMixin:  # Kimi 网格多模态数据混入类
    """Mixin providing Kimi-specific grid-based multimodal data helpers.
    提供 Kimi 专用的基于网格的多模态数据辅助功能的混入类。

    Expects the concrete class to supply:
      - self.hf_config  (with vision_config.merge_kernel_size)
      - self._tokenizer (with .encode())
    期望具体类提供：
      - self.hf_config（包含 vision_config.merge_kernel_size）
      - self._tokenizer（包含 .encode()）
    """

    def resolve_image_token_counts(self, images):  # 解析图像 token 数量，使用 Kimi 的 media_tokens_calculator
        """Kimi's processor is remote-code and does not implement the
        transformers ``_get_num_multimodal_tokens`` convention; use its
        ``media_tokens_calculator`` instead.
        Kimi 的处理器是远程代码，不实现 transformers 的 _get_num_multimodal_tokens 约定；
        使用其 media_tokens_calculator 代替。

        """
        assert images is not None  # 断言图像列表不为空
        media_tokens_calculator = (  # 获取媒体 token 计算器
            self._processor.media_processor.media_tokens_calculator  # 从处理器获取
        )
        return [  # 返回每张图像的 token 数量列表
            int(media_tokens_calculator({"type": "image", "image": image}))  # 计算单张图像的 token 数
            for image in images  # 遍历所有图像
        ]

    def _num_image_tokens_from_grid(  # 从网格元数据计算 Kimi 风格的图像 token 数量
        self, grid_thw: Union[torch.Tensor, np.ndarray, list, tuple]  # 网格时间-高度-宽度元数据
    ) -> int:
        """Compute Kimi-style image token count from 2D/3D grid metadata."""
        """从 2D/3D 网格元数据计算 Kimi 风格的图像 token 数量。"""
        merge_h, merge_w = self.hf_config.vision_config.merge_kernel_size  # 获取合并核大小

        if isinstance(grid_thw, torch.Tensor):  # 如果是 PyTorch 张量
            vals = grid_thw.flatten().tolist()  # 展平并转为列表
        elif isinstance(grid_thw, np.ndarray):  # 如果是 NumPy 数组
            vals = grid_thw.reshape(-1).tolist()  # 重塑为一维并转为列表
        elif isinstance(grid_thw, (list, tuple)):  # 如果是列表或元组
            vals = list(np.array(grid_thw).reshape(-1).tolist())  # 转换为 NumPy 后重塑并转为列表
        else:  # 其他类型
            raise TypeError(  # 抛出类型错误
                f"Unsupported grid type for kimi image tokens: {type(grid_thw)}"  # 不支持的网格类型
            )

        if len(vals) >= 3:  # 如果值数量 >= 3，包含时间维度
            _t, h, w = vals[-3], vals[-2], vals[-1]  # 解析时间、高度、宽度
        elif len(vals) == 2:  # 如果值数量 == 2，不包含时间维度
            _t, h, w = 1, vals[0], vals[1]  # 时间设为 1，解析高度和宽度
        else:  # 其他情况
            raise ValueError(  # 抛出值错误
                f"Invalid grid metadata for kimi image tokens: {vals} "  # 无效的网格元数据
                "(expected [t,h,w] or [h,w])"  # 期望格式
            )

        h, w = int(h), int(w)  # 转换为整数
        return (h * w) // (merge_h * merge_w)  # 计算合并后的 token 数量

    def _build_kimi_mm_data_from_grids(  # 从网格元数据构建 Kimi 多模态数据输出
        self, prompt, embeddings, **kwargs  # 提示文本、嵌入、其他参数
    ) -> MultimodalProcessorOutput:
        image_token_id = kwargs.get("image_token_id", 0)  # 获取图像标记 ID
        img_grid_thw = kwargs.get("img_grid_thw", None)  # 获取图像网格信息

        if not isinstance(prompt, list):  # 如果提示文本不是列表（即不是 token ID 列表）
            prompt = self._tokenizer.encode(prompt)  # 使用分词器编码提示文本

        image_token_counts = [  # 计算每张图像的 token 数量
            self._num_image_tokens_from_grid(grid) for grid in img_grid_thw  # 遍历网格元数据
        ]

        input_ids = []  # 初始化输入 ID 列表
        offsets = []  # 初始化偏移量列表
        img_idx = 0  # 初始化图像索引

        for token in prompt:  # 遍历提示中的每个 token
            if token != image_token_id:  # 如果不是图像标记
                input_ids.append(token)  # 直接添加到输入 ID
                continue  # 跳过后续处理

            if img_idx >= len(image_token_counts):  # 如果图像占位符数量超过网格条目
                raise ValueError(  # 抛出值错误
                    "The number of image placeholders exceeds img_grid_thw entries."  # 图像占位符数量超过网格条目数
                )

            num_tokens = image_token_counts[img_idx]  # 获取当前图像的 token 数量
            start = len(input_ids)  # 记录起始位置
            input_ids.extend([image_token_id] * num_tokens)  # 扩展图像标记
            offsets.append((start, len(input_ids) - 1))  # 记录偏移量（起止位置）
            img_idx += 1  # 递增图像索引

        if img_idx != len(image_token_counts):  # 如果图像占位符数量与网格条目数不匹配
            raise ValueError(  # 抛出值错误
                "The number of image placeholders does not match img_grid_thw entries."  # 图像占位符数量与网格条目数不匹配
            )

        image_embeddings = embeddings[Modality.IMAGE]  # 获取图像嵌入
        mm_items = []  # 初始化多模态数据项列表
        consumed = 0  # 已消耗的嵌入数量
        for start, end in offsets:  # 遍历偏移量
            num_tokens = end - start + 1  # 计算当前图像的 token 数量
            embedding_slice = image_embeddings[consumed : consumed + num_tokens]  # 获取对应的嵌入切片
            consumed += num_tokens  # 更新已消耗数量
            mm_items.append(  # 添加多模态数据项
                MultimodalDataItem(  # 创建数据项
                    modality=Modality.IMAGE,  # 图像模态
                    offsets=[(start, end)],  # 偏移量
                    precomputed_embeddings=embedding_slice,  # 预计算的嵌入
                )
            )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids,  # 输入 ID 列表
            mm_items=mm_items,  # 多模态数据项
            im_token_id=image_token_id,  # 图像标记 ID
        )
