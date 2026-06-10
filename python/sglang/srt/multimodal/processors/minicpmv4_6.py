# MiniCPM-V 4.6多模态处理器模块
# 实现MiniCPM-V 4.6版本模型的多模态数据处理
# 由于HF尚未提供可用的MiniCPMV4_6Processor，此模块在sglang端实现了完整的预处理和聊天模板扩展
# 一旦HF处理器可用，此模块可简化为薄包装器
# Copyright 2026 The SGLang team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""sglang multimodal processor for MiniCPM-V 4.6.
# MiniCPM-V 4.6的sglang多模态处理器

Ports per-image preprocessing + chat-template expansion sglang-side because
no working HF ``MiniCPMV4_6Processor`` is reachable yet: transformers main
does not ship one until 5.7+, and the released 4.6 checkpoints ship only a
tokenizer (no remote-code processor), so ``AutoProcessor.from_pretrained``
falls through to a bare tokenizer. Once a real processor is loadable, this
module collapses to a thin wrapper that delegates to it.
# 在sglang端移植逐图像预处理和聊天模板扩展，因为目前没有可用的HF MiniCPMV4_6Processor：
# transformers主线直到5.7+才提供，而4.6检查点只附带分词器（无远程代码处理器），
# 因此AutoProcessor.from_pretrained会回退到裸分词器。一旦真正的处理器可加载，
# 此模块将简化为委托给它的薄包装器。
"""

from __future__ import annotations  # 启用延迟类型注解求值

import math  # 导入数学模块
from itertools import chain  # 导入迭代器链工具
from typing import Any, List, Optional, Sequence, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch
import torchvision.transforms.functional as F  # 导入torchvision变换函数
from PIL import Image  # 导入PIL图像模块

from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.minicpmv import MiniCPMV4_6ForConditionalGeneration  # 导入MiniCPM-V 4.6模型
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模odal处理器和特殊令牌类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

IMAGENET_STANDARD_MEAN = (0.5, 0.5, 0.5)  # ImageNet标准化均值
IMAGENET_STANDARD_STD = (0.5, 0.5, 0.5)  # ImageNet标准化标准差

# Inner per-feature pad sentinel: prevents the next per-image
# ``replace(image_token, ...)`` from clobbering a previous expansion's inner
# pads. Swapped back to the real pad token once per modality after splicing.
# 内部逐特征填充占位符：防止下一个逐图像的replace(image_token, ...)覆盖前一次扩展的内部填充。
# 在拼接后，每种模态一次性将占位符替换回真实填充令牌。
_PAD_PLACEHOLDER = "<|placeholder|>"


def _ensure_divide(length: int, divisor: int) -> int:  # 确保长度能被除数整除
    return max(round(length / divisor) * divisor, divisor)  # 返回最接近的可整除值，最小为divisor


def _to_chw_tensor(image) -> torch.Tensor:  # 将图像转换为(C, H, W)格式的float32张量
    """PIL / torch / numpy -> ``(C, H, W)`` float32 in ``[0, 255]``.
    # PIL/torch/numpy图像转为(C, H, W)格式的float32张量，值范围[0, 255]

    Image inputs from ``load_mm_data`` are PIL; video frames from sglang's
    video decoder come back as numpy arrays.
    # 来自load_mm_data的图像输入是PIL格式；来自sglang视频解码器的视频帧返回numpy数组
    """
    if isinstance(image, torch.Tensor):  # 如果是PyTorch张量
        if image.dim() == 4:  # 如果是4维（批次维度）
            image = image.squeeze(0)  # 移除批次维度
        if image.dim() != 3:  # 如果不是3维
            raise ValueError(f"expected 3-D image tensor, got {image.shape}")  # 抛出异常
        if image.shape[0] not in (1, 3, 4):  # 如果通道不在第0维
            image = image.permute(2, 0, 1).contiguous()  # 转置为CHW格式
        if image.shape[0] == 4:  # 如果有4个通道（RGBA）
            image = image[:3]  # 取前3个通道（RGB）
        if image.shape[0] == 1:  # 如果是单通道
            image = image.repeat(3, 1, 1)  # 复制为3通道
        return image.float()  # 返回float32张量

    if isinstance(image, Image.Image):  # 如果是PIL图像
        if image.mode != "RGB":  # 如果不是RGB模式
            image = image.convert("RGB")  # 转换为RGB
        return F.pil_to_tensor(image).float()  # 转为张量并转float32

    import numpy as np  # 延迟导入numpy

    if isinstance(image, np.ndarray):  # 如果是numpy数组
        t = torch.from_numpy(image)  # 转为PyTorch张量
        if t.dim() == 3 and t.shape[-1] in (1, 3, 4):  # 如果是3维且通道在最后
            t = t.permute(2, 0, 1).contiguous()  # 转为CHW格式
        if t.shape[0] == 4:  # 如果有4通道
            t = t[:3]  # 取RGB
        if t.shape[0] == 1:  # 如果是单通道
            t = t.repeat(3, 1, 1)  # 复制为3通道
        return t.float()  # 返回float32张量

    raise TypeError(f"Unsupported image type: {type(image)!r}")  # 不支持的类型抛出异常


def _resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:  # 调整图像大小
    return F.resize(  # 使用双三次插值调整大小
        image,
        size=[height, width],  # 目标高度和宽度
        interpolation=F.InterpolationMode.BICUBIC,  # 双三次插值
        antialias=True,  # 启用抗锯齿
    )


def _divide_to_patches(  # 将图像分割为补丁块
    image: torch.Tensor, patch_h: int, patch_w: int
) -> List[torch.Tensor]:
    _, H, W = image.shape  # 获取图像尺寸
    if H % patch_h != 0 or W % patch_w != 0:  # 检查是否可整除
        raise ValueError(f"image ({H}, {W}) not divisible by ({patch_h}, {patch_w})")  # 不可整除则报错
    rows = H // patch_h  # 计算行数
    cols = W // patch_w  # 计算列数
    patches: List[torch.Tensor] = []  # 补丁列表
    for r in range(rows):  # 遍历每行
        for c in range(cols):  # 遍历每列
            patches.append(  # 添加补丁
                image[
                    :, r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w
                ]
            )
    return patches  # 返回补丁列表


def _reshape_by_patch(image: torch.Tensor, patch_size: int) -> torch.Tensor:  # 按补丁重塑图像（NaViT打包）
    """``(C, H, W) -> (C, P, H*W/P)`` NaViT packing."""  # NaViT打包：将图像从(C,H,W)转为(C,P,H*W/P)
    C = image.shape[0]  # 通道数
    patches = torch.nn.functional.unfold(  # 使用unfold展开补丁
        image.unsqueeze(0), (patch_size, patch_size), stride=(patch_size, patch_size)
    )
    patches = patches.reshape(C, patch_size, patch_size, -1)  # 重塑为(C, P, P, num_patches)
    patches = patches.permute(0, 1, 3, 2).reshape(C, patch_size, -1)  # 排列为(C, P, H*W/P)
    return patches  # 返回重塑后的补丁


def _flatten_patches(  # 展平补丁
    per_item_pv: List[List[torch.Tensor]],  # 每项的像素值
    per_item_ts: List[List[List[int]]],  # 每项的目标尺寸
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Per-item per-patch -> flat per-patch (source first, slices row-major)."""  # 逐项逐补丁 -> 扁平逐补丁（源优先，切片行主序）
    flat_pv = list(chain.from_iterable(per_item_pv))  # 展平像素值
    flat_ts = [  # 展平目标尺寸
        torch.tensor(ts, dtype=torch.int32) for ts in chain.from_iterable(per_item_ts)
    ]
    return flat_pv, flat_ts  # 返回展平后的像素值和目标尺寸


class MiniCPMV4_6ImageProcessor:  # MiniCPM-V 4.6图像预处理器类
    """Per-image preprocessing.
    # 逐图像预处理

    Pipeline: pick a slice grid (rows x cols, up to ``max_slice_nums``); resize
    source and (optionally) tiles to multiples of ``patch_size * 4`` (factor 4
    = the two successive 2x2 spatial merges: mid-ViT merger + DownsampleMLP);
    rescale, normalize, and NaViT-pack each tile into ``(C, P, H*W/P)``.
    # 流水线：选择切片网格（行x列，最多max_slice_nums个）；将源图和（可选的）瓦片调整大小为patch_size*4的倍数
    # （因子4 = 两次连续的2x2空间合并：中间ViT合并器 + DownsampleMLP）；
    # 重新缩放、标准化，并将每个瓦片NaViT打包为(C, P, H*W/P)。
    """

    def __init__(  # 初始化图像预处理器
        self,
        max_slice_nums: int = 9,  # 最大切片数
        scale_resolution: int = 448,  # 缩放分辨率
        patch_size: int = 14,  # 补丁大小
        slice_mode: bool = True,  # 是否启用切片模式
        downsample_mode: str = "16x",  # 下采样模式
        use_image_id: bool = True,  # 是否使用图像ID
        image_mean: Sequence[float] = IMAGENET_STANDARD_MEAN,  # 图像均值
        image_std: Sequence[float] = IMAGENET_STANDARD_STD,  # 图像标准差
        rescale_factor: float = 1.0 / 255.0,  # 重新缩放因子
    ) -> None:
        self.max_slice_nums = max_slice_nums  # 保存最大切片数
        self.scale_resolution = scale_resolution  # 保存缩放分辨率
        self.patch_size = patch_size  # 保存补丁大小
        self.slice_mode = slice_mode  # 保存切片模式
        self.downsample_mode = downsample_mode  # 保存下采样模式
        self.use_image_id = use_image_id  # 保存图像ID设置
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)  # 图像均值张量
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)  # 图像标准差张量
        self.rescale_factor = rescale_factor  # 保存重新缩放因子

    def _find_best_resize(  # 查找最佳缩放尺寸
        self,
        image_size: Tuple[int, int],  # 原始图像尺寸
        allow_upscale: bool = False,  # 是否允许放大
    ) -> Tuple[int, int]:
        height, width = image_size  # 解包高度和宽度
        scale = self.scale_resolution  # 获取缩放分辨率
        # factor 4 = two successive 2x2 spatial merges (mid-ViT + DownsampleMLP)
        # 因子4 = 两次连续的2x2空间合并（中间ViT + DownsampleMLP）
        divisor = self.patch_size * 4  # 计算除数
        if (height * width > scale * scale) or allow_upscale:  # 如果图像较大或允许放大
            aspect_ratio = width / height  # 计算宽高比
            height = int(scale / math.sqrt(aspect_ratio))  # 按比例计算新高度
            width = int(height * aspect_ratio)  # 按比例计算新宽度
        best_w = _ensure_divide(width, divisor)  # 确保宽度可整除
        best_h = _ensure_divide(height, divisor)  # 确保高度可整除
        return best_h, best_w  # 返回最佳高度和宽度

    def _get_refine_size(  # 获取精炼尺寸
        self,
        image_size: Tuple[int, int],  # 原始图像尺寸
        grid: Tuple[int, int],  # 切片网格
        allow_upscale: bool = False,  # 是否允许放大
    ) -> Tuple[int, int]:
        height, width = image_size  # 解包高度和宽度
        grid_y, grid_x = grid  # 解包网格行数和列数
        refine_w = _ensure_divide(width, grid_x)  # 宽度按列数对齐
        refine_h = _ensure_divide(height, grid_y)  # 高度按行数对齐
        bh, bw = self._find_best_resize(  # 查找每个瓦片的最佳尺寸
            (refine_h // grid_y, refine_w // grid_x),
            allow_upscale=allow_upscale,  # 是否允许放大
        )
        return bh * grid_y, bw * grid_x  # 返回精炼后的总高度和宽度

    def _get_sliced_grid(  # 计算切片网格布局
        self, image_size: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        original_h, original_w = image_size  # 原始图像尺寸
        scale = self.scale_resolution  # 缩放分辨率
        log_ratio = math.log(original_w / original_h)  # 对数宽高比
        ratio = original_w * original_h / (scale * scale)  # 像素比率
        multiple = min(math.ceil(ratio), self.max_slice_nums)  # 计算切片倍数
        if multiple <= 1:  # 如果不需要切片
            return None  # 返回None

        best_grid = (1, 1)  # 初始最佳网格
        min_error = float("inf")  # 最小误差初始化
        for num_slices in (multiple - 1, multiple, multiple + 1):  # 尝试附近3个切片数
            if num_slices == 1 or num_slices > self.max_slice_nums:  # 跳过无效值
                continue
            for num_rows in range(1, num_slices + 1):  # 遍历可能的行数
                if num_slices % num_rows != 0:  # 必须整除
                    continue
                num_cols = num_slices // num_rows  # 计算列数
                error = abs(log_ratio - math.log(num_rows / num_cols))  # 计算误差
                if error < min_error:  # 如果误差更小
                    # Ref returns ``[cols, rows]``; preserve the convention so
                    # downstream code matches HF.
                    # 参考实现返回[cols, rows]；保持这个约定以使下游代码与HF匹配
                    best_grid = (num_cols, num_rows)  # 更新最佳网格
                    min_error = error  # 更新最小误差
        return best_grid  # 返回最佳网格

    def _normalize(self, t: torch.Tensor) -> torch.Tensor:  # 标准化张量
        t = t * self.rescale_factor  # 重新缩放
        return (t - self.image_mean.to(t.dtype)) / self.image_std.to(t.dtype)  # 减均值除标准差

    def __call__(self, images: List) -> dict:  # 可调用接口
        return self.preprocess(images)  # 调用预处理方法

    def preprocess(self, images: List) -> dict:  # 预处理图像列表
        """Returns ``{pixel_values, tgt_sizes, grids, num_patches_per_image}``.
        # 返回{pixel_values, tgt_sizes, grids, num_patches_per_image}

        Per image, ``pixel_values[i]`` is a list whose first entry is the
        source patch and remaining entries are slice tiles in row-major grid
        order. ``grids[i]`` is ``[cols, rows]`` (zeros if no slicing).
        # 每张图像的pixel_values[i]是一个列表，第一个条目是源补丁，其余条目是按行主序的切片瓦片。
        # grids[i]是[cols, rows]（无切片时为零）。
        """
        per_image_pv: List[List[torch.Tensor]] = []  # 每张图像的像素值
        per_image_ts: List[List[List[int]]] = []  # 每张图像的目标尺寸
        all_grids: List[List[int]] = []  # 所有网格
        num_patches_per_image: List[int] = []  # 每张图像的补丁数

        for image in images:  # 遍历每张图像
            chw = _to_chw_tensor(image)  # 转为CHW张量
            H0, W0 = chw.shape[-2], chw.shape[-1]  # 原始高度和宽度
            best_grid = self._get_sliced_grid((H0, W0)) if self.slice_mode else None  # 计算切片网格

            allow_upscale_src = best_grid is None  # 无切片时允许放大源图
            src_h, src_w = self._find_best_resize(  # 查找源图最佳尺寸
                (H0, W0), allow_upscale=allow_upscale_src
            )
            source = _resize(chw, src_h, src_w)  # 调整源图大小

            patches: List[torch.Tensor] = [source]  # 补丁列表，第一个是源图
            patch_h = patch_w = 0  # 初始化瓦片尺寸
            if best_grid is not None:  # 如果有切片网格
                refine_h, refine_w = self._get_refine_size(  # 获取精炼尺寸
                    (H0, W0), best_grid, allow_upscale=True
                )
                refined = _resize(chw, refine_h, refine_w)  # 调整精炼后的图像大小
                grid_y, grid_x = best_grid  # 获取网格行列数
                patch_h = refine_h // grid_y  # 每个瓦片的高度
                patch_w = refine_w // grid_x  # 每个瓦片的宽度
                patches.extend(_divide_to_patches(refined, patch_h, patch_w))  # 分割并添加瓦片

            patches = [self._normalize(p) for p in patches]  # 标准化所有补丁

            pv = [_reshape_by_patch(patches[0], self.patch_size)]  # 源图的NaViT打包
            ts = [[src_h // self.patch_size, src_w // self.patch_size]]  # 源图的目标尺寸
            for p in patches[1:]:  # 遍历切片瓦片
                pv.append(_reshape_by_patch(p, self.patch_size))  # NaViT打包
                ts.append([patch_h // self.patch_size, patch_w // self.patch_size])  # 目标尺寸

            per_image_pv.append(pv)  # 添加像素值
            per_image_ts.append(ts)  # 添加目标尺寸
            all_grids.append(list(best_grid) if best_grid is not None else [0, 0])  # 添加网格
            num_patches_per_image.append(len(pv))  # 添加补丁数

        return {  # 返回预处理结果
            "pixel_values": per_image_pv,  # 像素值
            "tgt_sizes": per_image_ts,  # 目标尺寸
            "grids": all_grids,  # 网格布局
            "num_patches_per_image": num_patches_per_image,  # 每张图像的补丁数
        }


class MiniCPMV4_6MultimodalProcessor(BaseMultimodalProcessor):  # MiniCPM-V 4.6多模态处理器类
    """4.6-only mm processor.
    # 仅用于4.6版本的多模态处理器

    The legacy ``MiniCPMMultimodalProcessor`` stays for 2.6/4.0/4.5 because its
    ``_processor.tokenizer`` shape and ``(<image>./</image>)`` placeholder
    format don't fit 4.6.
    # 旧版MiniCPMMultimodalProcessor保留给2.6/4.0/4.5，因为其_processor.tokenizer结构
    # 和(<image>./</image>)占位符格式不适用于4.6。
    """

    models = [MiniCPMV4_6ForConditionalGeneration]  # 关联的模型列表
    support_dynamic_frame_expansion = False  # 不支持动态帧扩展
    gpu_image_decode = False  # 禁用GPU图像解码

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化4.6多模态处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化

        # ``_processor`` is either the bare tokenizer (current state — no
        # ``MiniCPMV4_6Processor`` shipped) or a real processor whose
        # ``.tokenizer`` exposes the same.
        # _processor可能是裸分词器（当前状态——没有MiniCPMV4_6Processor）或真正的处理器，
        # 其.tokenizer暴露相同接口。
        self.tokenizer = getattr(_processor, "tokenizer", _processor)  # 获取分词器

        vision_cfg = getattr(hf_config, "vision_config", None)  # 获取视觉配置
        patch_size = (  # 获取补丁大小
            getattr(vision_cfg, "patch_size", 14) if vision_cfg is not None else 14
        )
        downsample_mode = getattr(hf_config, "downsample_mode", "16x")  # 获取下采样模式
        # Per-image preprocessor; reused for video frames (HF ref's
        # video slicing geometry matches image slicing exactly).
        # 逐图像预处理器；复用于视频帧（HF参考实现的视频切片几何与图像切片完全一致）。
        self.image_processor = MiniCPMV4_6ImageProcessor(  # 创建图像预处理器
            max_slice_nums=9,  # 最大切片数
            scale_resolution=448,  # 缩放分辨率
            patch_size=patch_size,  # 补丁大小
            slice_mode=True,  # 启用切片模式
            downsample_mode=downsample_mode,  # 下采样模式
            use_image_id=True,  # 使用图像ID
        )

        self.image_token = "<|image_pad|>"  # 图像填充令牌
        self.video_token = "<|video_pad|>"  # 视频填充令牌
        self.image_token_id = getattr(hf_config, "image_token_id", None)  # 获取图像令牌ID
        if self.image_token_id is None:  # 如果配置中没有
            self.image_token_id = self._token_id(self.image_token)  # 从分词器获取
        self.video_token_id = getattr(hf_config, "video_token_id", None)  # 获取视频令牌ID
        if self.video_token_id is None:  # 如果配置中没有
            self.video_token_id = self._token_id(self.video_token)  # 从分词器获取

        # ``<image>``/``<slice>`` wrap the expanded regions for both images and
        # video frames; only the inner per-feature pad token differs.
        # <image>/<slice>包装扩展区域（图像和视频帧都使用）；仅内部逐特征填充令牌不同。
        self.image_start_token = "<image>"  # 图像开始令牌
        self.image_end_token = "</image>"  # 图像结束令牌
        self.slice_start_token = "<slice>"  # 切片开始令牌
        self.slice_end_token = "</slice>"  # 切片结束令牌
        self.image_id_start_token = "<image_id>"  # 图像ID开始令牌
        self.image_id_end_token = "</image_id>"  # 图像ID结束令牌

        self.image_start_id = self._token_id(self.image_start_token)  # 图像开始令牌ID
        self.image_end_id = self._token_id(self.image_end_token)  # 图像结束令牌ID
        self.slice_start_id = self._token_id(self.slice_start_token)  # 切片开始令牌ID
        self.slice_end_id = self._token_id(self.slice_end_token)  # 切片结束令牌ID

        self.pad_divisor = 16 if downsample_mode != "4x" else 4  # 填充除数，4x模式为4否则为16

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊令牌
            image_token=self.image_token,  # 图像填充令牌
            image_token_id=self.image_token_id,  # 图像令牌ID
            video_token=self.video_token,  # 视频填充令牌
            video_token_id=self.video_token_id,  # 视频令牌ID
        ).build(_processor)  # 构建令牌映射

    def _token_id(self, token: str):  # 将令牌字符串转换为ID
        try:
            ids = self.tokenizer.convert_tokens_to_ids([token])  # 转换令牌为ID
            if ids and ids[0] is not None:  # 如果转换成功
                return int(ids[0])  # 返回整数ID
        except Exception:  # 捕获异常
            pass
        return None  # 转换失败返回None

    def _expand_frame(  # 扩展单帧的令牌模板
        self,
        tgt_sizes: List[List[int]],  # 目标尺寸列表
        grid: List[int],  # 切片网格
    ) -> str:
        """``<image>...</image>`` (+ optional ``<slice>...</slice>`` rows) for
        one image or video frame; inner pads are ``_PAD_PLACEHOLDER`` (caller
        swaps back after splicing).
        # <image>...</image>（+ 可选的<slice>...</slice>行）用于一张图像或视频帧；
        # 内部填充为_PAD_PLACEHOLDER（调用者在拼接后替换回真实令牌）。
        """
        h0, w0 = tgt_sizes[0]  # 源图目标尺寸
        n_src = (h0 * w0) // self.pad_divisor  # 计算源图填充令牌数
        out = self.image_start_token + _PAD_PLACEHOLDER * n_src + self.image_end_token  # 源图令牌序列

        if len(tgt_sizes) > 1 and grid and grid[0] > 0 and grid[1] > 0:  # 如果有切片
            grid_y, grid_x = int(grid[0]), int(grid[1])  # 网格行列数
            h_s, w_s = tgt_sizes[1]  # 切片瓦片尺寸
            n_slice = (h_s * w_s) // self.pad_divisor  # 每个瓦片的填充令牌数
            slice_chunk = (  # 一个切片瓦片的令牌序列
                self.slice_start_token
                + _PAD_PLACEHOLDER * n_slice
                + self.slice_end_token
            )
            row_chunks = [slice_chunk * grid_x for _ in range(grid_y)]  # 每行的瓦片序列
            out += "\n".join(row_chunks)  # 行之间用换行符连接
        return out  # 返回扩展后的令牌字符串

    def _expand_media(  # 扩展单个媒体项（图像或视频）
        self,
        index: int,  # 媒体索引
        frames: Sequence[Tuple[List[List[int]], List[int]]],  # 帧列表
    ) -> str:
        """One image or one video. Image is a single-frame video."""  # 一张图像或一个视频。图像是单帧视频。
        body = "".join(self._expand_frame(ts, grid) for ts, grid in frames)  # 拼接所有帧的扩展
        return f"{self.image_id_start_token}{index}{self.image_id_end_token}" + body  # 添加图像ID包装

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: Sequence[Union[str, bytes]],  # 图像数据
        audio_data: Sequence[Union[str, bytes]],  # 音频数据
        input_text,  # 输入文本
        request_obj,  # 请求对象
        **kwargs: Any,  # 关键字参数
    ):
        # ``TokenizerManager`` does not pass ``video_data`` through the
        # processor signature; read it off the request the way qwen_vl does.
        # TokenizerManager不通过处理器签名传递video_data；像qwen_vl那样从请求中读取。
        video_data = getattr(request_obj, "video_data", None) or kwargs.get(  # 获取视频数据
            "video_data"
        )
        base = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 提示文本
            audio_data=audio_data,  # 音频数据
            image_data=image_data,  # 图像数据
            video_data=video_data,  # 视频数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊令牌
        )
        if base is None:  # 如果基础输出为空
            return None  # 返回None

        prompt: str = base.input_text or ""  # 获取提示文本
        images = base.images or []  # 获取图像列表
        videos = base.videos or []  # 获取视频列表

        # Image: one "frame" per image. Video: per-frame nesting kept so each
        # frame becomes its own ``<image>...</image>`` block in the expansion.
        # 图像：每张图像一个"帧"。视频：保持逐帧嵌套，使每帧在扩展中成为独立的<image>...</image>块。
        img_per_pv, img_per_ts, img_grids = self._preprocess_images(images)  # 预处理图像
        vid_per_pv, vid_per_ts, vid_grids = self._preprocess_videos(videos)  # 预处理视频

        prompt = self._splice_expansions(  # 将扩展插入提示文本
            prompt,
            (  # 图像扩展迭代器
                self._expand_media(i, [(ts, gd)])
                for i, (ts, gd) in enumerate(zip(img_per_ts, img_grids))
            ),
            (  # 视频扩展迭代器
                self._expand_media(i, list(zip(fts, fgd)))
                for i, (fts, fgd) in enumerate(zip(vid_per_ts, vid_grids))
            ),
        )

        input_ids: List[int] = self.tokenizer.encode(prompt, add_special_tokens=False)  # 编码提示文本
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)  # 转为张量

        # Each patch's pad tokens are guaranteed contiguous (the expansion
        # functions wrap them in ``<image>...</image>`` / ``<slice>...</slice>``
        # with nothing else in between), so a per-token-id contiguous-run scan
        # — base's ``get_mm_items_offset`` — gives one (start, end) per patch.
        # 每个补丁的填充令牌保证是连续的（扩展函数将它们包装在<image>...</image>/
        # <slice>...</slice>中，中间没有其他内容），因此按令牌ID的连续运行扫描
        # ——基类的get_mm_items_offset——为每个补丁提供一个(start, end)。
        mm_items: List[MultimodalDataItem] = []  # 多模态数据项列表
        mm_items.extend(  # 添加图像数据项
            self._build_items(
                input_ids_tensor,  # 输入ID张量
                self.image_token_id,  # 图像令牌ID
                _flatten_patches(img_per_pv, img_per_ts),  # 展平的图像补丁
                Modality.IMAGE,  # 图像模态
            )
        )
        # Video: extra ``per-frame -> per-patch`` nesting; pre-flatten one
        # level so ``_flatten_patches`` sees the same shape as image.
        # 视频：额外的逐帧到逐补丁嵌套；预先展平一层，使_flatten_patches看到与图像相同的形状。
        vid_pv_flat = [list(chain.from_iterable(v)) for v in vid_per_pv]  # 展平视频像素值
        vid_ts_flat = [list(chain.from_iterable(v)) for v in vid_per_ts]  # 展平视频目标尺寸
        mm_items.extend(  # 添加视频数据项
            self._build_items(
                input_ids_tensor,  # 输入ID张量
                self.video_token_id,  # 视频令牌ID
                _flatten_patches(vid_pv_flat, vid_ts_flat),  # 展平的视频补丁
                Modality.VIDEO,  # 视频模态
            )
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=mm_items,  # 多模态数据项
            input_ids=input_ids,  # 输入ID列表
            im_token_id=self.image_token_id,  # 图像令牌ID
            im_start_id=self.image_start_id,  # 图像开始令牌ID
            im_end_id=self.image_end_id,  # 图像结束令牌ID
            slice_start_id=self.slice_start_id,  # 切片开始令牌ID
            slice_end_id=self.slice_end_id,  # 切片结束令牌ID
        )

    def _preprocess_images(self, images):  # 预处理图像列表
        if not images:  # 如果没有图像
            return [], [], []  # 返回空列表
        out = self.image_processor.preprocess(images)  # 使用图像预处理器处理
        return out["pixel_values"], out["tgt_sizes"], out["grids"]  # 返回像素值、目标尺寸和网格

    def _preprocess_videos(self, videos):  # 预处理视频列表
        per_video_pv: List[List[List[torch.Tensor]]] = []  # 每个视频的像素值
        per_video_ts: List[List[List[List[int]]]] = []  # 每个视频的目标尺寸
        per_video_grids: List[List[List[int]]] = []  # 每个视频的网格
        for frames in videos:  # 遍历每个视频
            out = self.image_processor.preprocess(list(frames))  # 预处理所有帧
            per_video_pv.append(out["pixel_values"])  # 添加像素值
            per_video_ts.append(out["tgt_sizes"])  # 添加目标尺寸
            per_video_grids.append(out["grids"])  # 添加网格
        return per_video_pv, per_video_ts, per_video_grids  # 返回结果

    def _splice_expansions(self, prompt, image_expansions, video_expansions):  # 将媒体扩展拼接到提示文本中
        # The chat template emits exactly one marker per media item; a
        # sequential ``replace(..., n=1)`` walk lines them up by left-to-right
        # order. Expansions carry ``_PAD_PLACEHOLDER`` for inner pads so the
        # next replace doesn't trip on a previous expansion's pads — we swap
        # placeholders back to the real pad token in one pass per modality.
        # 聊天模板为每个媒体项发出一个标记；顺序的replace(..., n=1)按从左到右的顺序对齐。
        # 扩展使用_PAD_PLACEHOLDER作为内部填充，这样下一次replace不会被前一次扩展的填充干扰——
        # 我们在每种模态的一次遍历中将占位符替换回真实填充令牌。
        for token, expansions in (  # 遍历图像和视频令牌
            (self.image_token, image_expansions),  # 图像令牌和扩展
            (self.video_token, video_expansions),  # 视频令牌和扩展
        ):
            for expansion in expansions:  # 遍历每个扩展
                if token not in prompt:  # 如果提示中没有该令牌
                    break  # 跳出循环
                prompt = prompt.replace(token, expansion, 1)  # 替换第一个匹配的令牌
            prompt = prompt.replace(_PAD_PLACEHOLDER, token)  # 将占位符替换回真实令牌
        return prompt  # 返回拼接后的提示

    def _build_items(  # 构建多模态数据项列表
        self,
        input_ids: torch.Tensor,  # 输入ID张量
        pad_token_id: int,  # 填充令牌ID
        flat: Tuple[List[torch.Tensor], List[torch.Tensor]],  # 展平的像素值和目标尺寸
        modality: Modality,  # 模态类型
    ) -> List[MultimodalDataItem]:
        flat_pv, flat_ts = flat  # 解包像素值和目标尺寸
        runs = self.get_mm_items_offset(input_ids, pad_token_id)  # 获取填充令牌的连续运行偏移
        if len(runs) != len(flat_pv):  # 检查运行数和补丁数是否匹配
            raise RuntimeError(
                f"[minicpmv4_6] {modality} pad run / feature count mismatch: "
                f"{len(runs)} runs vs {len(flat_pv)} patches"
            )
        return [  # 返回数据项列表
            MultimodalDataItem(
                feature=[pv],  # 像素值
                offsets=[run],  # 偏移
                model_specific_data={"tgt_size": [ts]},  # 目标尺寸
                modality=modality,  # 模态
            )
            for run, pv, ts in zip(runs, flat_pv, flat_ts)  # 逐补丁配对
        ]
