# Kimi K2.5 多模态图像处理器模块
# 本模块实现了 Kimi K2.5 模型的图像数据处理逻辑，
# 包括 GPU 加速的图像预处理（缩放、填充、归一化、分块）、
# NaViT 缩放配置计算以及 KimiGPUProcessorWrapper 包装器。
import math  # 导入数学模块
import re  # 导入正则表达式模块
from collections import defaultdict  # 导入默认字典
from typing import Dict, List, Union  # 导入类型提示模块

import numpy as np  # 导入 NumPy 模块
import torch  # 导入 PyTorch 模块
import torch.nn.functional as F  # 导入 PyTorch 函数式模块
from PIL import Image  # 导入 PIL 图像模块

from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    MultimodalProcessorOutput,  # 多模态处理器输出类
)
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration  # 导入 Kimi K2.5 条件生成模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器相关类
    BaseMultimodalProcessor as SGLangBaseProcessor,  # 将基础处理器重命名为 SGLangBaseProcessor
)
from sglang.srt.multimodal.processors.base_processor import (  # 导入多模态特殊标记类
    MultimodalSpecialTokens,  # 多模态特殊标记类
)
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin  # 导入 Kimi 网格多模态数据混入类

# ---------------------------------------------------------------------------
# GPU image preprocessing utilities (resize, pad, normalize, patchify on CUDA)
# GPU 图像预处理工具（在 CUDA 上执行缩放、填充、归一化、分块）
# ---------------------------------------------------------------------------


def navit_resize_config(  # 计算 NaViT 缩放目标尺寸和 token 数量
    width: int,  # 原始宽度
    height: int,  # 原始高度
    patch_size: int,  # patch 大小
    merge_kernel_size: int,  # 合并核大小
    in_patch_limit: int,  # 输入 patch 限制
    patch_limit_on_one_side: int,  # 单侧 patch 限制
    fixed_output_tokens: int | None = None,  # 固定输出 token 数，可选
) -> dict:
    """Compute NaViT resize target dimensions and token count.
    计算 NaViT 缩放目标尺寸和 token 数量。

    Pure math -- no image data needed, only (width, height).
    纯数学计算——不需要图像数据，只需要 (width, height)。
    """
    s1 = math.sqrt(  # 计算基于 patch 限制的缩放因子
        in_patch_limit  # 输入 patch 限制
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))  # 当前 patch 数量
    )
    s2 = patch_limit_on_one_side * patch_size / width  # 计算基于宽度限制的缩放因子
    s3 = patch_limit_on_one_side * patch_size / height  # 计算基于高度限制的缩放因子
    scale = min(1.0, s1, s2, s3)  # 取最小缩放因子，确保不放大
    new_w = min(max(1, int(width * scale)), patch_limit_on_one_side * patch_size)  # 计算新宽度
    new_h = min(max(1, int(height * scale)), patch_limit_on_one_side * patch_size)  # 计算新高度

    factor = merge_kernel_size * patch_size  # 计算对齐因子
    pad_height = (factor - new_h % factor) % factor  # 计算高度填充量
    pad_width = (factor - new_w % factor) % factor  # 计算宽度填充量

    if fixed_output_tokens is not None:  # 如果指定了固定输出 token 数
        num_tokens = fixed_output_tokens  # 使用固定值
    else:  # 否则
        token_height = (new_h + pad_height) // factor  # 计算 token 高度
        token_width = (new_w + pad_width) // factor  # 计算 token 宽度
        num_tokens = token_height * token_width  # 计算 token 总数

    return {  # 返回缩放配置字典
        "num_tokens": num_tokens,  # token 数量
        "new_width": new_w,  # 新宽度
        "new_height": new_h,  # 新高度
        "pad_width": pad_width,  # 宽度填充量
        "pad_height": pad_height,  # 高度填充量
    }


def _get_image_dimensions(image: Union[torch.Tensor, Image.Image]) -> tuple[int, int]:  # 获取图像的 (宽度, 高度)
    """Get (width, height) from a CUDA tensor or PIL Image."""
    """从 CUDA 张量或 PIL Image 获取 (宽度, 高度)。"""
    if isinstance(image, torch.Tensor):  # 如果是 PyTorch 张量
        # nvJPEG returns (C, H, W) uint8
        return image.shape[2], image.shape[1]  # 返回 (宽度, 高度)
    return image.size  # PIL returns (width, height)  # PIL 返回 (宽度, 高度)


def _pil_to_cuda_chw(image: Image.Image) -> torch.Tensor:  # 将 PIL Image 转换为 (C, H, W) uint8 CUDA 张量
    """Convert PIL Image to (C, H, W) uint8 CUDA tensor."""
    """将 PIL Image 转换为 (C, H, W) uint8 CUDA 张量。"""
    arr = np.asarray(image.convert("RGB"))  # 转换为 RGB 格式的 numpy 数组
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()  # 转换为 (C, H, W) 格式的 CUDA 张量


def _process_single_image(  # 在 GPU 上处理单张图像：缩放 -> 填充 -> 归一化 -> 分块
    image: Union[torch.Tensor, Image.Image],  # 输入图像
    config: dict,  # 缩放配置
    image_mean: torch.Tensor,  # 图像均值张量
    image_std_inv: torch.Tensor,  # 图像标准差倒数张量
    patch_size: int,  # patch 大小
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process a single image on GPU: resize -> pad -> normalize -> patchify."""
    """在 GPU 上处理单张图像：缩放 -> 填充 -> 归一化 -> 分块。"""
    if isinstance(image, Image.Image):  # 如果是 PIL Image
        image = _pil_to_cuda_chw(image)  # 转换为 CUDA 张量

    new_h, new_w = config["new_height"], config["new_width"]  # 获取目标高度和宽度
    pad_h, pad_w = config["pad_height"], config["pad_width"]  # 获取填充量

    x = image.unsqueeze(0).float()  # 增加批次维度并转为浮点型
    x = F.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)  # 双三次插值缩放

    if pad_h > 0 or pad_w > 0:  # 如果需要填充
        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)  # 用零填充

    x = x / 255.0  # 归一化到 [0, 1]
    x = (x - image_mean) * image_std_inv  # 应用均值和标准差归一化

    _, C, H, W = x.shape  # 获取通道数、高度和宽度
    T = 1  # 时间维度设为 1
    gh, gw = H // patch_size, W // patch_size  # 计算网格高度和宽度
    x = x.view(T, C, gh, patch_size, gw, patch_size)  # 重塑为分块视图
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, patch_size, patch_size)  # 重排并展平分块

    grid_thw = torch.tensor([T, gh, gw], dtype=torch.int64, device=x.device)  # 创建网格元数据
    return x, grid_thw  # 返回分块后的张量和网格信息


def _gpu_preprocess_images(  # GPU 批量图像预处理流水线
    images: list[Union[torch.Tensor, Image.Image]],  # 图像列表
    resize_configs: list[dict],  # 缩放配置列表
    image_mean: torch.Tensor,  # 图像均值张量
    image_std_inv: torch.Tensor,  # 图像标准差倒数张量
    patch_size: int,  # patch 大小
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU preprocessing pipeline for a batch of images.
    GPU 批量图像预处理流水线。

    Groups images with the same target padded size for batch processing.
    将具有相同目标填充尺寸的图像分组以进行批量处理。
    """
    n = len(images)  # 图像数量
    if n == 0:  # 如果没有图像
        device = image_mean.device  # 获取设备
        return (  # 返回空张量
            torch.empty(0, 3, patch_size, patch_size, device=device),  # 空像素值张量
            torch.empty(0, 3, dtype=torch.int64, device=device),  # 空网格张量
        )

    groups = defaultdict(list)  # 按目标尺寸分组的字典
    for idx, (image, config) in enumerate(zip(images, resize_configs)):  # 遍历图像和配置
        padded_h = config["new_height"] + config["pad_height"]  # 计算填充后高度
        padded_w = config["new_width"] + config["pad_width"]  # 计算填充后宽度
        target_h = config["new_height"]  # 目标高度
        target_w = config["new_width"]  # 目标宽度
        groups[(target_h, target_w, padded_h, padded_w)].append((idx, image, config))  # 添加到对应分组

    all_patches = [None] * n  # 初始化所有分块列表
    all_grids = [None] * n  # 初始化所有网格列表

    for (target_h, target_w, padded_h, padded_w), group in groups.items():  # 遍历每个分组
        if len(group) == 1:  # 如果组内只有一张图像
            idx, image, config = group[0]  # 获取图像信息
            patches, grid = _process_single_image(  # 处理单张图像
                image, config, image_mean, image_std_inv, patch_size
            )
            all_patches[idx] = patches  # 保存分块结果
            all_grids[idx] = grid  # 保存网格信息
        else:  # 如果组内有多张图像，进行批量处理
            tensors = []  # 初始化张量列表
            for _, image, _ in group:  # 遍历组内图像
                if isinstance(image, Image.Image):  # 如果是 PIL Image
                    image = _pil_to_cuda_chw(image)  # 转换为 CUDA 张量
                tensors.append(image.unsqueeze(0).float())  # 增加批次维度并转为浮点型

            resized = []  # 初始化缩放后的张量列表
            for t in tensors:  # 遍历张量
                r = F.interpolate(  # 双三次插值缩放
                    t, size=(target_h, target_w), mode="bicubic", align_corners=False
                )
                resized.append(r)  # 添加到列表
            batch = torch.cat(resized, dim=0)  # 拼接为批次

            pad_h = padded_h - target_h  # 计算高度填充量
            pad_w = padded_w - target_w  # 计算宽度填充量
            if pad_h > 0 or pad_w > 0:  # 如果需要填充
                batch = F.pad(batch, (0, pad_w, 0, pad_h), value=0.0)  # 用零填充

            batch = batch / 255.0  # 归一化到 [0, 1]
            batch = (batch - image_mean) * image_std_inv  # 应用均值和标准差归一化

            B, C, H, W = batch.shape  # 获取批次大小、通道数、高度和宽度
            T = 1  # 时间维度设为 1
            gh, gw = H // patch_size, W // patch_size  # 计算网格高度和宽度
            batch = batch.view(B, C, gh, patch_size, gw, patch_size)  # 重塑为分块视图
            batch = batch.permute(0, 2, 4, 1, 3, 5).reshape(  # 重排并展平分块
                B, -1, C, patch_size, patch_size
            )

            grid = torch.tensor([T, gh, gw], dtype=torch.int64, device=batch.device)  # 创建网格元数据
            for i, (idx, _, _) in enumerate(group):  # 遍历组内图像
                all_patches[idx] = batch[i]  # 保存分块结果
                all_grids[idx] = grid  # 保存网格信息

    pixel_values = torch.cat(all_patches, dim=0)  # 拼接所有分块
    grid_thws = torch.stack(all_grids, dim=0)  # 堆叠所有网格信息
    return pixel_values, grid_thws  # 返回像素值和网格信息


# ---------------------------------------------------------------------------
# Kimi K2.5 GPU processor wrapper
# Kimi K2.5 GPU 处理器包装器
# ---------------------------------------------------------------------------


class KimiGPUProcessorWrapper:  # Kimi GPU 处理器包装器类
    """Wraps Kimi's HF processor to do GPU image preprocessing.
    包装 Kimi 的 HuggingFace 处理器以进行 GPU 图像预处理。

    GPU path: nvJPEG CUDA tensor / PIL -> _gpu_preprocess_images()
    GPU 路径：nvJPEG CUDA 张量 / PIL -> _gpu_preprocess_images()
    CPU fallback: PIL -> medias kwarg -> original HF KimiK25Processor.__call__
    CPU 回退路径：PIL -> medias 关键字参数 -> 原始 HuggingFace KimiK25Processor.__call__

    Exposes attributes that base class's process_mm_data needs so it behaves
    like a normal HF processor from the outside.
    暴露基类 process_mm_data 所需的属性，使其行为类似普通的 HuggingFace 处理器。
    """

    def __init__(  # 初始化 GPU 处理器包装器
        self,
        hf_processor,  # HuggingFace 处理器
        image_token,  # 图像标记
        patch_size,  # patch 大小
        merge_kernel_size,  # 合并核大小
        in_patch_limit,  # 输入 patch 限制
        patch_limit_on_one_side,  # 单侧 patch 限制
        fixed_output_tokens,  # 固定输出 token 数
        image_mean,  # 图像均值
        image_std,  # 图像标准差
    ):
        self._hf_processor = hf_processor  # 保存 HuggingFace 处理器
        self._image_token = image_token  # 保存图像标记
        self._patch_size = patch_size  # 保存 patch 大小
        self._merge_kernel_size = merge_kernel_size  # 保存合并核大小
        self._in_patch_limit = in_patch_limit  # 保存输入 patch 限制
        self._patch_limit_on_one_side = patch_limit_on_one_side  # 保存单侧 patch 限制
        self._fixed_output_tokens = fixed_output_tokens  # 保存固定输出 token 数
        self._image_mean = image_mean  # 保存图像均值
        self._image_std = image_std  # 保存图像标准差
        self._gpu_norm_tensors = None  # GPU 归一化张量缓存

        # Explicitly expose attributes that base class process_mm_data needs:
        # - image_processor: checked via isinstance(..., BaseImageProcessor)
        # - tokenizer: used for tokenization
        # - media_processor: used by CPU fallback path
        # 显式暴露基类 process_mm_data 所需的属性：
        # - image_processor：通过 isinstance(..., BaseImageProcessor) 检查
        # - tokenizer：用于分词
        # - media_processor：用于 CPU 回退路径
        self.image_processor = hf_processor.image_processor  # 暴露图像处理器
        self.tokenizer = hf_processor.tokenizer  # 暴露分词器
        self.media_processor = hf_processor.media_processor  # 暴露媒体处理器

    def __call__(self, text=None, images=None, **kwargs):  # 调用处理器，自动选择 GPU 或 CPU 路径
        # process_mm_data passes images via kwargs["images"]
        images = images or kwargs.pop("images", None)  # 获取图像数据

        if images and torch.cuda.is_available():  # 如果有图像且 CUDA 可用
            return self._gpu_call(text, images)  # 使用 GPU 路径
        return self._cpu_call(text, images, **kwargs)  # 否则使用 CPU 回退路径

    def _gpu_call(self, text, images):  # GPU 调用路径，绕过 HuggingFace 预处理，直接使用 GPU 操作
        """Bypass HF KimiK25VisionProcessor.preprocess entirely -- use GPU ops."""
        """完全绕过 HuggingFace KimiK25VisionProcessor.preprocess —— 使用 GPU 操作。"""
        input_text = text[0] if isinstance(text, list) else text  # 获取输入文本

        # 1. Compute resize configs (CPU math)
        resize_configs = []  # 初始化缩放配置列表
        for image in images:  # 遍历所有图像
            w, h = _get_image_dimensions(image)  # 获取图像尺寸
            resize_configs.append(  # 添加缩放配置
                navit_resize_config(  # 计算 NaViT 缩放配置
                    w,  # 宽度
                    h,  # 高度
                    self._patch_size,  # patch 大小
                    self._merge_kernel_size,  # 合并核大小
                    self._in_patch_limit,  # 输入 patch 限制
                    self._patch_limit_on_one_side,  # 单侧 patch 限制
                    self._fixed_output_tokens,  # 固定输出 token 数
                )
            )

        # 2. Expand image tokens
        parts = input_text.split(self._image_token)  # 按图像标记分割文本
        result = [parts[0]]  # 初始化结果列表，添加第一部分
        for config, part in zip(resize_configs, parts[1:]):  # 遍历配置和文本部分
            result.append(self._image_token * config["num_tokens"] + part)  # 扩展图像标记
        input_text = "".join(result)  # 重新拼接文本

        # 3. Tokenize
        text_inputs = self._hf_processor.tokenizer(input_text, return_tensors="pt")  # 分词

        # 4. GPU image preprocessing
        image_mean, image_std_inv = self._get_gpu_norm_tensors()  # 获取 GPU 归一化张量
        pixel_values, grid_thws = _gpu_preprocess_images(  # GPU 预处理图像
            images, resize_configs, image_mean, image_std_inv, self._patch_size
        )

        grid_thws = grid_thws.cpu()  # 将网格信息移至 CPU

        return {  # 返回处理结果字典
            "input_ids": text_inputs["input_ids"],  # 输入 ID
            "pixel_values": pixel_values,  # 像素值
            # Use SGL-standard key so get_new_expanded_mm_items() can split
            # per-image for cache granularity (it looks up 'image_grid_thw').
            # 使用 SGL 标准键，以便 get_new_expanded_mm_items() 可以按图像拆分以获得缓存粒度
            "image_grid_thw": grid_thws,  # 图像网格信息
        }

    def _cpu_call(self, text, images, **kwargs):  # CPU 回退路径，使用 HuggingFace 原始处理器
        """Fallback: token expansion + medias kwarg -> original HF processor."""
        """回退路径：token 扩展 + medias 关键字参数 -> 原始 HuggingFace 处理器。"""
        input_text = text[0] if isinstance(text, list) else text  # 获取输入文本

        if images:  # 如果有图像
            # Token expansion via media_tokens_calculator
            parts = input_text.split(self._image_token)  # 按图像标记分割文本
            result = [parts[0]]  # 初始化结果列表
            for image, part in zip(images, parts[1:]):  # 遍历图像和文本部分
                num_tokens = self._hf_processor.media_processor.media_tokens_calculator(  # 计算 token 数量
                    {"type": "image", "image": image}  # 媒体描述
                )
                result.append(self._image_token * num_tokens + part)  # 扩展图像标记
            input_text = "".join(result)  # 重新拼接文本

            # Convert to medias format for Kimi's HF processor
            kwargs["medias"] = [{"type": "image", "image": img} for img in images]  # 转换为 Kimi 的 medias 格式

        out = self._hf_processor(text=[input_text], **kwargs)  # 调用原始 HuggingFace 处理器
        grid_thws = out.pop("grid_thws", None)  # 弹出网格信息
        if grid_thws is not None:  # 如果有网格信息
            out["image_grid_thw"] = grid_thws  # 重命名为标准键
        return out  # 返回处理结果

    def _get_gpu_norm_tensors(self, device="cuda"):  # 获取或初始化 GPU 归一化张量（带缓存）
        if self._gpu_norm_tensors is None:  # 如果缓存为空
            image_mean = torch.tensor(  # 创建均值张量
                self._image_mean, device=device, dtype=torch.float32  # 使用配置的均值
            ).view(1, 3, 1, 1)  # 调整形状为 (1, 3, 1, 1)
            image_std_inv = (  # 创建标准差倒数张量
                1.0 / torch.tensor(self._image_std, device=device, dtype=torch.float32)  # 使用配置的标准差
            ).view(1, 3, 1, 1)  # 调整形状为 (1, 3, 1, 1)
            self._gpu_norm_tensors = (image_mean, image_std_inv)  # 缓存归一化张量
        return self._gpu_norm_tensors  # 返回缓存的归一化张量


# ---------------------------------------------------------------------------
# Kimi K2.5 SGLang multimodal processor
# Kimi K2.5 SGLang 多模态处理器
# ---------------------------------------------------------------------------


# Compatible with KimiVLForConditionalGeneration
class KimiK2_5VLImageProcessor(KimiGridMMDataMixin, SGLangBaseProcessor):  # Kimi K2.5 VL 图像处理器类
    models = [KimiK25ForConditionalGeneration]  # 支持的模型列表
    gpu_image_decode = True  # nvJPEG for JPEG, PIL fallback for others  # nvJPEG 处理 JPEG，PIL 回退处理其他格式

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化 Kimi K2.5 VL 图像处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化方法
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token="<|media_pad|>",  # 媒体填充标记
            # TODO: could we convert in MultimodalSpecialTokens?
            image_token_id=hf_config.media_placeholder_token_id,  # 媒体占位符标记 ID
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),  # 匹配媒体填充标记的正则表达式
        ).build(_processor)  # 使用处理器构建标记

        # Extract media processing config from HF processor
        media_proc_cfg = _processor.media_processor.media_proc_cfg  # 从 HuggingFace 处理器获取媒体处理配置

        # Replace with GPU-capable wrapper
        self._processor = KimiGPUProcessorWrapper(  # 用支持 GPU 的包装器替换原始处理器
            _processor,  # 原始 HuggingFace 处理器
            image_token=self.mm_tokens.image_token,  # 图像标记
            patch_size=media_proc_cfg["patch_size"],  # patch 大小
            merge_kernel_size=media_proc_cfg["merge_kernel_size"],  # 合并核大小
            in_patch_limit=media_proc_cfg["in_patch_limit"],  # 输入 patch 限制
            patch_limit_on_one_side=media_proc_cfg["patch_limit_on_one_side"],  # 单侧 patch 限制
            fixed_output_tokens=media_proc_cfg.get("fixed_output_tokens"),  # 固定输出 token 数
            image_mean=media_proc_cfg["image_mean"],  # 图像均值
            image_std=media_proc_cfg["image_std"],  # 图像标准差
        )

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes, Dict]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        base_output = await self.load_mm_data(  # 异步加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊标记
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 传入基础输出和标记
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入 ID 列表
            mm_items=mm_items,  # 多模态项
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记 token ID
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):  # 获取多模态数据，使用网格信息构建
        img_grid_thw = kwargs.get("img_grid_thw", None)  # 获取图像网格信息
        return self._build_kimi_mm_data_from_grids(  # 调用混入类的方法构建多模态数据
            prompt=prompt,  # 提示文本
            embeddings=embeddings,  # 嵌入
            image_token_id=self.mm_tokens.image_token_id,  # 图像标记 token ID
            img_grid_thw=img_grid_thw,  # 图像网格信息
        )
