# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Source: https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/llava/mm_utils.py
"""
Utilities for multi-modal models.

This python file mainly contains utilities that were used in the
image processing logic of llava-next including operations such as
anyres and anyres_max

Currently supports the anyres and anyres_max operation for CLIP and
SigLip. For more information, you may refer to the paper or the blog

LLaVA-NeXT : https://llava-vl.github.io/blog/2024-01-30-llava-next/
LLaVA-Onevision : https://arxiv.org/pdf/2408.03326

"""

# 多模态模型工具函数：提供图像预处理（任意分辨率裁剪、填充、分块）、
# 图像编解码（base64加载）、去填充还原、数据并行视觉模型推理等功能。
# 主要用于 LLaVA-NeXT 等多模态模型的图像处理流水线。

import ast
import itertools
import math
import re
from io import BytesIO
from typing import Literal

import numpy as np
import pybase64
import torch
from PIL import Image

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.utils import flatten_nested_list


def ensure_numpy(x):
    """Convert torch.Tensor to numpy array if needed (v5 compat)."""
    # 将 torch.Tensor 转换为 numpy 数组（如果输入已经是 numpy 则直接返回），用于兼容 transformers v5
    return x.numpy() if isinstance(x, torch.Tensor) else x


def has_valid_data(data) -> bool:
    # 检查多模态数据是否包含有效内容，递归检查嵌套列表
    if data is None:
        return False
    if isinstance(data, list):
        return any(has_valid_data(item) for item in flatten_nested_list(data))
    return True


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    # 根据原始图像尺寸，从候选分辨率列表中选择最匹配的分辨率。
    # 选择标准：最大化有效分辨率，最小化浪费分辨率。
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0  # 最大有效分辨率
    min_wasted_resolution = float("inf")  # 最小浪费分辨率

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        # 计算保持宽高比的缩放比例
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )

        # Calculate effective and wasted resolutions
        # 计算有效分辨率和浪费分辨率
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        # 优先选择有效分辨率更大的；若相同则选择浪费分辨率更小的
        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    # 将图像缩放并填充到目标分辨率，同时保持原始宽高比，空白区域用黑色填充
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    # 选择较小的缩放因子以保持宽高比，使图像完整显示在目标区域内
    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # 创建黑色背景的目标尺寸画布，将缩放后的图像居中粘贴
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    # 将图像按指定块大小切割为多个 patch（图像块）
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    # 计算任意分辨率图像预处理后的 patch 网格形状（宽、高方向各多少个 patch）
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [
            224,
            336,
            384,
            448,
            512,
        ], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        # 使用正则表达式从字符串中提取分辨率范围，如 "(1x1)x(2x2)"
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        # 生成从起始到结束的网格点组合
        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]
        # Multiply all elements by patch_size
        # 将每个网格点乘以 patch_size 得到实际像素分辨率
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        # 将字符串形式的 grid_pinpoints 解析为列表
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    # 选择最佳分辨率并计算网格形状
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        np.array: An np array containing the processed image patches.
    """
    # 处理任意分辨率的图像：选择最佳分辨率 → 缩放填充 → 分块 → 预处理
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [
            224,
            336,
            384,
            448,
            512,
        ], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        # 使用正则表达式提取分辨率范围
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        # 生成网格点组合
        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]
        # Multiply all elements by patch_size
        # 乘以 patch_size 得到实际像素分辨率
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    # 选择最佳分辨率并进行缩放填充
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    # For Siglip processor, only have size but no crop size.
    # In transformers v5, crop_size may exist but be None.
    # 获取裁剪尺寸：优先使用 crop_size，否则回退到 size（兼容 Siglip 处理器和 transformers v5）
    crop_size = (
        processor.crop_size["height"]
        if getattr(processor, "crop_size", None) is not None
        else processor.size["height"]
    )
    shortest_edge = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else processor.size["height"]
    )
    # 将填充后的图像分割为 patch 块
    patches = divide_to_patches(image_padded, crop_size)

    # 缩放原始图像到最短边尺寸，作为基础 patch
    image_original_resize = image.resize((shortest_edge, shortest_edge))

    # 将原始缩放图像与分块 patch 拼接
    image_patches = [image_original_resize] + patches
    # 对每个 patch 进行预处理（归一化等）
    image_patches = [
        processor.preprocess(image_patch.convert("RGB"))["pixel_values"][0]
        for image_patch in image_patches
    ]
    # In transformers v5, image processors may return torch.Tensor instead of numpy arrays
    # 兼容 transformers v5：确保所有 patch 都是 numpy 数组
    image_patches = [ensure_numpy(p) for p in image_patches]
    return np.stack(image_patches, axis=0)


def load_image_from_base64(image):
    # 从 base64 编码字符串加载图像
    return Image.open(BytesIO(pybase64.b64decode(image, validate=True)))


def expand2square(pil_img, background_color):
    # 将非正方形图像扩展为正方形，空白区域用指定背景色填充
    width, height = pil_img.size
    if width == height:
        return pil_img
    # 灰度图转换为 RGB
    if pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")
    if width > height:
        # 宽大于高：创建正方形画布，将图像垂直居中粘贴
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        # 高大于宽：创建正方形画布，将图像水平居中粘贴
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    # 去除填充后的图像张量中的填充部分，恢复原始宽高比
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        # 原图更宽：去除上下填充
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # 原图更高：去除左右填充
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


def unpad_image_shape(current_height, current_width, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image
    and returns the new shape.
    """
    # 计算去填充后的图像形状（不实际操作张量，仅返回形状）
    original_width, original_height = original_size

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        # 原图更宽：计算去除上下填充后的高度
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        new_shape = (current_height - 2 * padding, current_width)
    else:
        # 原图更高：计算去除左右填充后的宽度
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        new_shape = (current_height, current_width - 2 * padding)

    return new_shape


def process_images(images, image_processor, model_cfg):
    # 根据模型配置的图像宽高比策略处理图像列表（填充/任意分辨率/默认）
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        # 填充策略：将每张图像扩展为正方形后预处理
        for image in images:
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
            image = image_processor.preprocess(image)["pixel_values"][0]
            new_images.append(image)
    elif "anyres" in image_aspect_ratio:
        # 任意分辨率策略：使用 anyres 方法处理每张图像
        for image in images:
            image = process_anyres_image(
                image, image_processor, model_cfg.image_grid_pinpoints
            )
            new_images.append(image)
    else:
        # 默认策略：直接使用处理器批量处理
        return image_processor(images)["pixel_values"]
    # 若所有图像形状一致，则堆叠为 numpy 数组
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = np.stack(new_images, axis=0)
    return new_images


# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/vision.py
def get_dp_encoder_lb_assignment(
    sizes: list[int],
    num_gpus: int = 2,
) -> tuple[list[int], list[int], list[int]]:
    """
    Generate load balancing assignment and metadata
    for distributing data across GPUs.
    The load is determined by the total image sizes,
    not the number of images.

    Args:
        sizes: The size of each image
        num_gpus: Number of GPUs to balance across

    Returns:
        shuffle_indices:
            Indices to reorder data for balanced loading
        gpu_sample_counts:
            Number of samples assigned to each GPU
        grouped_sizes_per_gpu:
            Total size assigned to each GPU

    Example:
        ```
        sizes = [1000, 100, 200, 50]
        num_gpus = 2
        ```

    """

    # 数据并行编码器负载均衡分配：按图像总尺寸（而非图像数量）在多个 GPU 间均衡分配任务
    n_samples = len(sizes)

    # Handle edge cases
    # 边界情况：无数据时返回空结果
    if n_samples == 0:
        return [], [0] * num_gpus, [0] * num_gpus

    # Use greedy algorithm - balance by total size, not sample count
    # 贪心算法：按总尺寸均衡分配，而非按样本数量
    gpu_assignments = [list[int]() for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus  # This tracks total SIZE, not sample count
    # 跟踪每个 GPU 的总尺寸负载（非样本数量）

    # Sort indices by size (largest first for better load balancing)
    # sizes = [1000, 100, 200, 50]
    # large_to_small_indices = [0, 2, 1, 3]
    # 按尺寸降序排列索引，大任务优先分配以获得更好的负载均衡
    large_to_small_indices = sorted(
        range(n_samples), key=lambda i: sizes[i], reverse=True
    )

    for idx in large_to_small_indices:
        # Find GPU with minimum current load (by total size)
        # 找到当前负载最小的 GPU，将任务分配给它
        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]

    # Create shuffle indices and counts
    # 创建重排索引和每个 GPU 的样本计数
    shuffle_indices = list[int]()
    gpu_sample_counts = list[int]()
    for gpu_id in range(num_gpus):
        # GPU_0 = [1000] = [0]
        # GPU_1 = [200, 100, 50] = [2, 1, 3]
        # shuffle_indices = [0, 2, 1, 3]
        shuffle_indices.extend(gpu_assignments[gpu_id])
        # GPU_0 = [1]
        # GPU_1 = [3]
        # gpu_sample_counts = [1, 3]
        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

    return (shuffle_indices, gpu_sample_counts, gpu_loads)


# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/vision.py
def run_dp_sharded_vision_model(
    image_input: torch.Tensor, vision_model: torch.nn.Module
) -> torch.Tensor:
    """Run a vision model with data parallelism (DP) sharding. The function
    will shard the input image tensor on the first dimension and run the vision
    model

    Args:
        image_input (torch.Tensor): Image input tensor.
        vision_model (torch.nn.Module): Vision model.
    Returns:
        torch.Tensor: Output image embeddings
    """

    # 数据并行分片运行视觉模型：将图像输入在第一维上分片，各 rank 独立运行后 all_gather 汇总
    num_chunks = image_input.shape[0]
    mp_world_size = get_tensor_model_parallel_world_size()
    # 计算每个 rank 处理的 chunk 数量（向上取整）
    num_chunks_per_rank = (num_chunks + mp_world_size - 1) // mp_world_size
    # 计算需要填充的 chunk 数量，使总数能被 world_size 整除
    num_padded_chunks = num_chunks_per_rank * mp_world_size - num_chunks
    pad = (0,) * (2 * (image_input.dim() - 1)) + (0, num_padded_chunks)
    image_input_padded = torch.nn.functional.pad(image_input, pad)
    # 获取当前 rank，并提取对应的输入分片
    rank = get_tensor_model_parallel_rank()
    image_input_per_rank = image_input_padded[
        rank * num_chunks_per_rank : (rank + 1) * num_chunks_per_rank, ...
    ]

    # 在当前 rank 上运行视觉模型
    vision_embeddings = vision_model(image_input_per_rank)
    # Ensure tensor is contiguous before all_gather
    # 确保 all_gather 前张量是连续的
    vision_embeddings = vision_embeddings.last_hidden_state.contiguous()
    # 通过 all_gather 收集所有 rank 的嵌入结果
    vision_embeddings = tensor_model_parallel_all_gather(vision_embeddings, dim=0)
    # 去除填充部分，恢复原始 chunk 数量
    vision_embeddings = vision_embeddings[:num_chunks, ...]
    return vision_embeddings


# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/vision.py
def run_dp_sharded_mrope_vision_model(
    vision_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw_list: list,
    *,
    rope_type: Literal["rope_3d", "rope_2d"],
):
    """Run a vision model with data parallelism (DP) sharding.
    The function will shard the input image tensor on the
    first dimension and run the vision model.
    This function is used to run the vision model with mrope.

    Args:
        vision_model (torch.nn.Module): Vision model.
        pixel_values (torch.Tensor): Image/Video input tensor.
        grid_thw_list: List of grid dimensions for each image
        rope_type: Type of rope used in the vision model.
                   Different rope types have different dimension to do ViT.
                   "rope_3d" for 3D rope (e.g., Qwen2.5-VL)
                   "rope_2d" for 2D rope (e.g., Kimi-VL)
    Returns:
        torch.Tensor: Output image embeddings

    Example:
        ```
        vision_model.out_hidden_size = 64
        vision_model.spatial_merge_size = 2
        pixel_values.shape = (1350, channel)
        grid_thw_list = [[1, 10, 100], [1, 10, 10], [1, 10, 20], [1, 50]]
        tp_size = 2
        ```

    """

    # 数据并行分片运行带 mRoPE 的视觉模型：按图像 patch 数量做负载均衡分配，
    # 各 rank 独立运行视觉模型后通过 all_gather 汇总并恢复原始顺序。
    from sglang.srt.layers.dp_attention import (
        get_attention_tp_group,
        get_attention_tp_rank,
        get_attention_tp_size,
    )

    tp_size = get_attention_tp_size()
    # 若 TP 大小为 1，直接运行无需分片
    if tp_size == 1:
        return vision_model(pixel_values, grid_thw=torch.tensor(grid_thw_list))

    # GPU_0 tp_rank_local = 0
    # GPU_1 tp_rank_local = 1
    tp_rank_local = get_attention_tp_rank()

    # patches_per_image = [1000, 100, 200, 50]
    # 计算每张图像的 patch 数量（grid_thw 各维度之积）
    patches_per_image = [math.prod(grid_thw) for grid_thw in grid_thw_list]
    # print(f"{patches_per_image = }")
    # patches_per_image = [0, 1000, 1100, 1300, 1350]
    # 计算累计 patch 数量，用于按索引切片
    cum_patches_per_image = [0, *itertools.accumulate(patches_per_image)]

    # Get load balancing assignment with all metadata
    # image_to_tp_rank = [0, 2, 1, 3]
    # gpu_sample_counts = [1, 3]
    # grouped_pixel_values_len = [1000, 350]
    # 获取负载均衡分配结果
    image_to_tp_rank, gpu_sample_counts, grouped_pixel_values_len = (
        get_dp_encoder_lb_assignment(patches_per_image, tp_size)
    )

    # cu_gpu_sample_counts = [0, 1, 4]
    # 累计每个 GPU 的样本计数
    cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]

    # GPU_0 image_idxs_local = [0]
    # GPU_1 image_idxs_local = [2, 1, 3]
    # 获取当前 rank 分配到的图像索引
    image_idxs_local = image_to_tp_rank[
        cum_gpu_sample_counts[tp_rank_local] : cum_gpu_sample_counts[tp_rank_local + 1]
    ]

    # Get the pixel values for the local images based on the image_idxs_local
    # 根据 image_idxs_local 提取当前 rank 需要处理的像素值
    if len(image_idxs_local) > 0:
        pixel_values_local = torch.cat(
            [
                pixel_values[cum_patches_per_image[i] : cum_patches_per_image[i + 1]]
                for i in image_idxs_local
            ]
        )
    else:
        # Handle case where this rank has no images
        # 处理当前 rank 无图像的情况
        pixel_values_local = torch.empty(
            (0, pixel_values.shape[1]),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
    # embed_dim_reduction_factor = 2 * 2
    # 计算嵌入维度缩减因子：spatial merge 或 merge kernel 的面积
    if rope_type == "rope_2d":
        embed_dim_reduction_factor = (
            vision_model.merge_kernel_size[0] * vision_model.merge_kernel_size[1]
        )
    else:
        embed_dim_reduction_factor = (
            vision_model.spatial_merge_size * vision_model.spatial_merge_size
        )

    # Find the max length across all ranks
    # The output embedding of every DP rank has to be
    # padded to this length for tensor_model_parallel_all_gather
    # to work
    # 找到所有 rank 中最大的输出长度，用于 all_gather 时对齐填充
    max_len_per_rank = max(grouped_pixel_values_len) // embed_dim_reduction_factor
    # 获取当前 rank 分配到的图像的 grid_thw 列表
    local_grid_thw_list = [grid_thw_list[i] for i in image_idxs_local]

    # Run the vision model on the local pixel_values_local
    # 在当前 rank 上运行视觉模型
    if rope_type == "rope_2d":
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(
                pixel_values_local, torch.tensor(local_grid_thw_list)
            )
            # 2D rope 可能返回列表，拼接为单个张量
            if isinstance(image_embeds_local, list):
                image_embeds_local = torch.cat(image_embeds_local, dim=0)
        else:
            out_dim = getattr(vision_model.config, "hidden_size", None)
            image_embeds_local = torch.empty(
                (0, embed_dim_reduction_factor, out_dim),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
    else:
        if pixel_values_local.shape[0] > 0:
            # print(f"{local_grid_thw_list = }", flush=True)
            image_embeds_local = vision_model(
                pixel_values_local, torch.tensor(local_grid_thw_list)
            )
        else:
            # Handle empty case
            # 处理空输入情况
            image_embeds_local = torch.empty(
                (0, vision_model.out_hidden_size),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )

    # Pad the output based on max_len_per_rank
    # for tensor_model_parallel_all_gather to work
    # 将输出填充到所有 rank 的最大长度，以便 all_gather 操作
    current_len = image_embeds_local.shape[0]
    if current_len < max_len_per_rank:
        padding_size = max_len_per_rank - current_len
        if rope_type == "rope_2d":
            # 2D rope 输出为 3D 张量 (seq_len, merge_size, hidden_size)
            padding = torch.empty(
                (
                    padding_size,
                    image_embeds_local.shape[1],
                    image_embeds_local.shape[2],
                ),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        else:
            # 3D rope 输出为 2D 张量 (seq_len, hidden_size)
            padding = torch.empty(
                (padding_size, image_embeds_local.shape[1]),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        image_embeds_local_padded = torch.cat([image_embeds_local, padding], dim=0)
    else:
        image_embeds_local_padded = image_embeds_local

    # Do all_gather to collect embeddings from all ranks
    # 执行 all_gather 收集所有 rank 的嵌入结果
    gathered_embeds = get_attention_tp_group().all_gather(
        image_embeds_local_padded, dim=0
    )

    # Remove padding and reconstruct per-rank embeddings
    # 去除填充，重建每个 rank 的嵌入
    rank_embeddings = list[torch.Tensor]()
    for rank in range(tp_size):
        start_idx = rank * max_len_per_rank
        end_idx = start_idx + (
            grouped_pixel_values_len[rank] // embed_dim_reduction_factor
        )
        rank_embeddings.append(gathered_embeds[start_idx:end_idx])

    # 计算每张输出图像的 patch 数量（经过空间合并后的）
    patches_per_output_image = [
        (patch_size // embed_dim_reduction_factor) for patch_size in patches_per_image
    ]

    # Reconstruct embeddings in the original order
    # 按原始图像顺序重建嵌入
    original_order_embeddings = [None] * len(grid_thw_list)
    current_idx = 0
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        if count > 0:
            # Get images assigned to this rank in shuffled order
            # GPU_0 = image_idxs_local  [0]
            # GPU_1 = image_idxs_local  [2, 1, 3]
            # 获取该 rank 被分配到的图像索引（重排后的顺序）
            rank_images = image_to_tp_rank[current_idx : current_idx + count]

            rank_embed = rank_embeddings[rank]
            # Split rank embeddings back to individual images
            # 将该 rank 的嵌入按图像拆分回各自的片段
            embed_start = 0
            for img_idx in rank_images:
                img_patches = patches_per_output_image[img_idx]
                original_order_embeddings[img_idx] = rank_embed[
                    embed_start : embed_start + img_patches
                ]
                embed_start += img_patches
            current_idx += count
    # 按原始顺序拼接所有嵌入
    out_embeddings = torch.cat(original_order_embeddings, dim=0)
    return out_embeddings
