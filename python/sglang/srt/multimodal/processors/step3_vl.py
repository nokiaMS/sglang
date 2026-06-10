# Step3-VL多模态处理器模块
# 本模块为Step3视觉语言模型提供图像数据处理功能
# 包含图像分块、滑动窗口裁剪和视觉预处理

import math  # 导入数学模块
import re  # 导入正则表达式模块
from itertools import product  # 导入笛卡尔积函数
from typing import List, Optional, Union  # 导入类型提示

import numpy as np  # 导入NumPy
import torch  # 导入PyTorch
from PIL import Image  # 导入PIL图像处理库
from torchvision import transforms  # 导入torchvision变换
from torchvision.transforms import InterpolationMode  # 导入插值模式枚举
from torchvision.transforms import functional as F  # 导入torchvision功能函数
from transformers import BatchFeature, ProcessorMixin, TensorType  # 导入transformers相关类

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.step3_vl import Step3VLForConditionalGeneration  # 导入Step3-VL模型
from sglang.srt.models.step3_vl_10b import StepVLForConditionalGeneration  # 导入Step3-VL-10B模型
from sglang.srt.models.step3p7 import Step3p7ForConditionalGeneration  # 导入Step3p7模型
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (  # 导入多模态特殊标记类
    MultimodalSpecialTokens,
)

Step3Image = Union[Image.Image, torch.Tensor]  # Step3图像类型，可以是PIL图像或张量
ImageWithPatches = tuple[Step3Image, list[Step3Image], list[int] | None]  # 带补丁的图像类型


class GPUToTensor(torch.nn.Module):  # GPU张量转换模块，将各种格式图像转为张量

    def forward(  # 前向传播，将原始图像转换为张量
        self, raw_image: Union[np.ndarray, Image.Image, torch.Tensor]  # 原始图像
    ) -> torch.Tensor:
        if isinstance(raw_image, torch.Tensor):  # 如果已经是张量
            image_tensor = raw_image  # 直接使用
            if image_tensor.ndim != 3:  # 检查维度是否为3（CHW）
                raise TypeError(
                    f"Expected CHW image tensor, got shape {tuple(image_tensor.shape)}"  # 期望CHW格式
                )
            if image_tensor.shape[0] == 1:  # 如果是单通道
                image_tensor = image_tensor.repeat(3, 1, 1)  # 重复为3通道
            elif image_tensor.shape[0] != 3:  # 如果不是3通道
                raise TypeError(
                    f"Expected CHW image tensor with 1 or 3 channels, got shape {tuple(image_tensor.shape)}"  # 期望1或3通道
                )
            if image_tensor.dtype == torch.uint8:  # 如果是uint8类型
                image_tensor = image_tensor.to(torch.float32).div(255)  # 转为float32并归一化
            elif not image_tensor.is_floating_point():  # 如果不是浮点类型
                image_tensor = image_tensor.to(torch.float32)  # 转为float32
            return image_tensor.contiguous()  # 返回连续张量
        if isinstance(raw_image, Image.Image):  # 如果是PIL图像
            image_tensor = transforms.ToTensor()(raw_image)  # 转换为张量
            if torch.cuda.is_available():  # 如果CUDA可用
                image_tensor = image_tensor.to(torch.device("cuda"))  # 移到GPU
            return image_tensor  # 返回张量
        if raw_image.ndim == 2:  # 如果是灰度图像（2维数组）
            raw_image = raw_image[:, :, None].repeat(3, -1)  # 扩展为3通道
        if torch.cuda.is_available():  # 如果CUDA可用
            device = torch.device("cuda")  # 使用GPU
        else:  # 否则
            device = torch.device("cpu")  # 使用CPU
        image_tensor = torch.from_numpy(raw_image).to(device)  # 从NumPy数组转为张量
        image_tensor = torch.permute(image_tensor, (2, 0, 1)).contiguous()  # HWC转CHW
        if image_tensor.dtype == torch.uint8:  # 如果是uint8类型
            image_tensor = image_tensor.to(torch.float32).div(255)  # 转为float32并归一化
        return image_tensor  # 返回张量


class Step3VisionProcessor:  # Step3视觉预处理器
    def __init__(self, size, interpolation_mode="bicubic", patch_size=None):  # 初始化视觉预处理器
        mean = [0.48145466, 0.4578275, 0.40821073]  # 归一化均值
        std = [0.26862954, 0.26130258, 0.27577711]  # 归一化标准差
        patch_size = patch_size if patch_size is not None else size  # 补丁尺寸默认与图像尺寸相同

        self.transform = transforms.Compose(  # 主图像变换管道
            [
                GPUToTensor(),  # 转换为张量
                transforms.Normalize(mean, std),  # 归一化
                transforms.Resize(  # 缩放
                    (size, size),  # 目标尺寸
                    interpolation=(  # 插值方法
                        InterpolationMode.BICUBIC  # 双三次插值
                        if interpolation_mode == "bicubic"  # 如果指定双三次
                        else InterpolationMode.BILINEAR  # 否则双线性
                    ),
                    antialias=True,  # 抗锯齿
                ),
            ]
        )

        self.patch_transform = (  # 补丁变换管道
            transforms.Compose(
                [
                    GPUToTensor(),  # 转换为张量
                    transforms.Normalize(mean, std),  # 归一化
                    transforms.Resize(  # 缩放
                        (patch_size, patch_size),  # 目标补丁尺寸
                        interpolation=(  # 插值方法
                            InterpolationMode.BICUBIC  # 双三次插值
                            if interpolation_mode == "bicubic"  # 如果指定双三次
                            else InterpolationMode.BILINEAR  # 否则双线性
                        ),
                        antialias=True,  # 抗锯齿
                    ),
                ]
            )
            if patch_size is not None  # 仅在指定补丁尺寸时创建
            else None  # 否则为None
        )

    def __call__(self, image, is_patch=False):  # 调用预处理器
        if is_patch:  # 如果是补丁
            return {"pixel_values": self.patch_transform(image).unsqueeze(0)}  # 使用补丁变换
        else:  # 否则
            return {"pixel_values": self.transform(image).unsqueeze(0)}  # 使用主变换


class ImagePatcher:  # 图像分块器，将大图像分割为多个补丁
    def get_image_size(self, img: Step3Image) -> tuple[int, int]:  # 获取图像尺寸
        if isinstance(img, Image.Image):  # 如果是PIL图像
            return img.size  # 返回(宽, 高)
        if isinstance(img, torch.Tensor):  # 如果是张量
            if img.ndim != 3:  # 检查维度
                raise TypeError(
                    f"Expected CHW image tensor, got shape {tuple(img.shape)}"  # 期望CHW格式
                )
            return int(img.shape[-1]), int(img.shape[-2])  # 返回(宽, 高)
        raise TypeError(f"Unsupported image type: {type(img)}")  # 不支持的类型

    def determine_window_size(self, long: int, short: int) -> int:  # 确定滑动窗口大小
        if long <= 728:  # 如果长边不超过728
            return short if long / short > 1.5 else 0  # 宽高比>1.5时返回短边，否则不分割
        return min(short, 504) if long / short > 4 else 504  # 宽高比>4时返回短边和504的较小值

    def slide_window(  # 滑动窗口裁剪
        self,
        width: int,  # 图像宽度
        height: int,  # 图像高度
        sizes: list[tuple[int, int]],  # 窗口尺寸列表
        steps: list[tuple[int, int]],  # 步长列表
        img_rate_thr: float = 0.6,  # 图像覆盖率阈值
    ) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:  # 返回裁剪框和网格数
        assert 1 >= img_rate_thr >= 0, "The `img_rate_thr` should lie in 0~1"  # 验证阈值范围
        windows = []  # 窗口列表
        # Sliding windows.
        for size, step in zip(sizes, steps):  # 遍历尺寸和步长
            size_w, size_h = size  # 解包窗口宽高
            step_w, step_h = step  # 解包步长

            x_num = 1 if width <= size_w else math.ceil((width - size_w) / step_w + 1)  # 计算x方向窗口数
            x_start = [step_w * i for i in range(x_num)]  # 生成x起始位置
            if len(x_start) > 1 and x_start[-1] + size_w > width:  # 调整最后一个起始位置
                x_start[-1] = width - size_w  # 确保不超出边界

            y_num = 1 if height <= size_h else math.ceil((height - size_h) / step_h + 1)  # 计算y方向窗口数
            y_start = [step_h * i for i in range(y_num)]  # 生成y起始位置
            if len(y_start) > 1 and y_start[-1] + size_h > height:  # 调整最后一个起始位置
                y_start[-1] = height - size_h  # 确保不超出边界

            start = np.array(list(product(y_start, x_start)), dtype=int)  # 生成所有起始点组合
            start[:, [0, 1]] = start[:, [1, 0]]  # 交换列顺序
            windows.append(np.concatenate([start, start + size], axis=1))  # 拼接起始和结束坐标
        windows = np.concatenate(windows, axis=0)  # 拼接所有窗口

        return [
            (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))  # 转换为(x, y, w, h)格式
            for box in windows
        ], (x_num, y_num)  # 返回裁剪框和网格数

    def square_pad(self, img: Step3Image) -> Step3Image:  # 正方形填充图像
        w, h = self.get_image_size(img)  # 获取图像尺寸
        if w == h:  # 如果已经是正方形
            return img  # 直接返回
        size = max(w, h)  # 取最大边长
        if isinstance(img, Image.Image):  # 如果是PIL图像
            padded = Image.new(img.mode, (size, size), 0)  # 创建正方形画布
            padded.paste(img, (0, 0))  # 粘贴原始图像
            return padded  # 返回填充后的图像
        return torch.nn.functional.pad(img, (0, size - w, 0, size - h), value=0)  # 张量填充

    def get_image_size_for_padding(  # 获取填充所需的图像尺寸
        self, img_width: int, img_height: int
    ) -> tuple[int, int]:
        ratio = img_width / img_height  # 计算宽高比
        if min(img_height, img_width) < 32 and (ratio > 4 or ratio < 1 / 4):  # 如果太小且宽高比过大
            new_size = max(img_height, img_width)  # 使用较大边
            return new_size, new_size  # 返回正方形尺寸
        return img_width, img_height  # 返回原始尺寸

    def get_image_size_for_preprocess(  # 获取预处理所需的图像尺寸
        self, img_width: int, img_height: int
    ) -> tuple[int, int]:

        if max(img_height, img_width) > 3024:  # 如果最大边超过3024
            scale_factor = 3024 / max(img_height, img_width)  # 计算缩放因子
            img_width = int(img_width * scale_factor)  # 缩放宽度
            img_height = int(img_height * scale_factor)  # 缩放高度
            return img_width, img_height  # 返回缩放后尺寸
        else:  # 否则
            return img_width, img_height  # 返回原始尺寸

    def get_image_size_for_crop(  # 获取裁剪所需的图像尺寸
        self, img_width: int, img_height: int, window_size: int
    ):
        w_ratio = img_width / window_size  # 计算宽度比
        h_ratio = img_height / window_size  # 计算高度比

        if w_ratio < 1:  # 如果宽度比小于1
            width_new = img_width  # 保持原始宽度
        else:  # 否则
            decimal_w = w_ratio - img_width // window_size  # 计算小数部分
            w_ratio = int(w_ratio) + 1 if decimal_w > 0.2 else int(w_ratio)  # 根据小数部分调整
            width_new = window_size * w_ratio  # 计算新宽度
        if h_ratio < 1:  # 如果高度比小于1
            height_new = img_height  # 保持原始高度
        else:  # 否则
            decimal_h = h_ratio - img_height // window_size  # 计算小数部分
            h_ratio = int(h_ratio) + 1 if decimal_h > 0.2 else int(h_ratio)  # 根据小数部分调整
            height_new = window_size * h_ratio  # 计算新高度
        return int(width_new), int(height_new)  # 返回新尺寸

    def resize(self, img: Step3Image, size: tuple[int, int]) -> Step3Image:  # 缩放图像
        if isinstance(img, Image.Image):  # 如果是PIL图像
            return img.resize(size, Image.Resampling.BILINEAR)  # 使用双线性插值缩放
        return F.resize(  # 使用torchvision功能缩放
            img,
            [size[1], size[0]],  # 目标尺寸（高度, 宽度）
            interpolation=InterpolationMode.BILINEAR,  # 双线性插值
            antialias=True,  # 抗锯齿
        ).contiguous()  # 返回连续张量

    def patch_crop(  # 裁剪图像补丁
        self, img: Step3Image, i: int, j: int, th: int, tw: int
    ) -> Step3Image:
        if isinstance(img, Image.Image):  # 如果是PIL图像
            return img.crop((j, i, j + tw, i + th))  # 裁剪图像
        return img[:, i : i + th, j : j + tw].contiguous()  # 裁剪张量

    def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]:  # 获取补丁数量
        img_width, img_height = self.get_image_size_for_padding(img_width, img_height)  # 获取填充尺寸
        img_width, img_height = self.get_image_size_for_preprocess(  # 获取预处理尺寸
            img_width, img_height
        )
        window_size = self.determine_window_size(  # 确定窗口大小
            max(img_height, img_width), min(img_height, img_width)
        )
        if window_size == 0:  # 如果不需要分割
            return 0, 0  # 返回0个补丁
        else:  # 否则
            img_width, img_height = self.get_image_size_for_crop(  # 获取裁剪尺寸
                img_width, img_height, window_size
            )
            center_list, (x_num, y_num) = self.slide_window(  # 获取滑动窗口裁剪列表
                img_width,
                img_height,
                [(window_size, window_size)],  # 窗口尺寸
                [(window_size, window_size)],  # 步长
            )
            full_rows = (len(center_list) - 1) // x_num + 1  # 计算完整行数
            if len(center_list) > 0 and len(center_list) % x_num == 0:  # 如果最后一行完整
                full_rows -= 1  # 减去1（因为最后一行不需要换行符）
            return len(center_list), full_rows  # 返回补丁数和完整行数

    def __call__(  # 调用图像分块器
        self, img: Step3Image
    ) -> tuple[Step3Image, list[Step3Image], list[bool] | None]:  # 返回原图、补丁列表和换行掩码
        img_width, img_height = self.get_image_size(img)  # 获取图像尺寸
        new_img_width, new_img_height = self.get_image_size_for_padding(  # 获取填充尺寸
            img_width, img_height
        )
        if new_img_width != img_width or new_img_height != img_height:  # 如果需要填充
            img = self.square_pad(img)  # 正方形填充
            img_width, img_height = self.get_image_size(img)  # 更新尺寸

        new_img_width, new_img_height = self.get_image_size_for_preprocess(  # 获取预处理尺寸
            img_width, img_height
        )
        img = self.resize(img, (new_img_width, new_img_height))  # 缩放图像
        window_size = self.determine_window_size(  # 确定窗口大小
            max(new_img_height, new_img_width), min(new_img_height, new_img_width)
        )
        if window_size == 0:  # 如果不需要分割
            return img, [], None  # 返回原图，无补丁
        else:  # 否则
            new_img_width, new_img_height = self.get_image_size_for_crop(  # 获取裁剪尺寸
                new_img_width, new_img_height, window_size
            )
            if (new_img_width, new_img_height) != (img_width, img_height):  # 如果需要裁剪缩放
                img_for_crop = self.resize(img, (new_img_width, new_img_height))  # 缩放图像用于裁剪
            else:  # 否则
                img_for_crop = img  # 直接使用原图

            patches = []  # 补丁列表
            newlines = []  # 换行位置列表
            center_list, (x_num, y_num) = self.slide_window(  # 获取滑动窗口
                new_img_width,
                new_img_height,
                [(window_size, window_size)],  # 窗口尺寸
                [(window_size, window_size)],  # 步长
            )
            for patch_id, center_lf_point in enumerate(center_list):  # 遍历裁剪框
                x, y, patch_w, patch_h = center_lf_point  # 解包裁剪位置和尺寸
                big_patch = self.patch_crop(img_for_crop, y, x, patch_h, patch_w)  # 裁剪补丁
                patches.append(big_patch)  # 添加到列表
                if (patch_id + 1) % x_num == 0:  # 如果到达行末
                    newlines.append(patch_id)  # 添加换行位置

            if newlines and newlines[-1] == len(patches) - 1:  # 如果最后一个换行是最后一个补丁
                newlines.pop()  # 移除（不需要换行）

            return (
                img,  # 原图
                patches,  # 补丁列表
                (
                    [i in newlines for i in range(len(patches))]  # 生成换行掩码
                    if len(patches) > 0  # 如果有补丁
                    else None  # 否则为None
                ),
            )


class Step3VLProcessor:  # Step3-VL处理器
    def __init__(  # 初始化Step3-VL处理器
        self,
        config,  # 模型配置
        tokenizer,  # 分词器
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置
        if isinstance(tokenizer, ProcessorMixin):  # 如果分词器是处理器混入类
            tokenizer = tokenizer.tokenizer  # 获取内部分词器
        self.tokenizer = tokenizer  # 保存分词器

        self.image_size = 728  # 图像尺寸
        self.patch_size = 504  # 补丁尺寸
        self.image_preprocessor = Step3VisionProcessor(  # 创建视觉预处理器
            self.image_size, "bilinear", self.patch_size  # 图像尺寸、插值方法、补丁尺寸
        )

        self.num_image_feature_size = 169  # 图像特征大小
        self.num_patch_feature_size = 81  # 补丁特征大小
        self.image_token = "<im_patch>"  # 图像标记
        self.image_feature_placeholder = self.image_token * self.num_image_feature_size  # 图像特征占位符
        self.patch_feature_placeholder = self.image_token * self.num_patch_feature_size  # 补丁特征占位符

        self.patcher = ImagePatcher()  # 创建图像分块器

    @property
    def image_token_id(self) -> int:  # 获取图像标记ID
        return self.tokenizer.get_vocab()[self.image_token]  # 从词表中获取

    def get_num_image_tokens(self, img_width: int, img_height: int) -> int:  # 计算图像标记数
        num_patches, num_newlines = self.patcher.get_num_patches(img_width, img_height)  # 获取补丁和换行数

        return (  # 计算总标记数
            num_patches * (self.num_patch_feature_size + 2)  # 补丁标记数
            + self.num_image_feature_size  # 图像特征标记数
            + 2  # 起始和结束标记
            + num_newlines  # 换行标记数
        )

    def _split_images(self, images: list[Image.Image]) -> list[ImageWithPatches]:  # 分割图像为原图和补丁
        result = []  # 结果列表
        for img in images:  # 遍历图像
            result.append(self.patcher(img))  # 分块并添加
        return result  # 返回结果

    def _convert_images_to_pixel_values(  # 将图像转换为像素值
        self,
        images: list[Step3Image],  # 图像列表
        is_patch: bool = False,  # 是否为补丁
    ) -> list[torch.Tensor]:
        return [
            self.image_preprocessor(img, is_patch=is_patch)["pixel_values"]  # 预处理每张图像
            for img in images  # 遍历图像
        ]

    def _get_patch_repl(  # 获取补丁替换文本和标记ID
        self,
        num_patches: int,  # 补丁数量
        patch_newline_mask: list[bool] | None,  # 补丁换行掩码
    ) -> tuple[str, list[int]]:
        text = ""  # 文本
        token_ids = []  # 标记ID列表
        for i in range(num_patches):  # 遍历每个补丁
            assert len(patch_newline_mask) == num_patches  # 断言掩码长度匹配
            text += f"<patch_start>{self.patch_feature_placeholder}<patch_end>"  # 添加补丁占位文本
            token_ids.extend(  # 添加补丁标记ID
                [self.tokenizer.convert_tokens_to_ids("<patch_start>")]  # 补丁起始标记ID
                + [self.image_token_id] * self.num_patch_feature_size  # 补丁特征标记ID
                + [self.tokenizer.convert_tokens_to_ids("<patch_end>")]  # 补丁结束标记ID
            )
            if patch_newline_mask and patch_newline_mask[i]:  # 如果需要换行
                text += "<patch_newline>"  # 添加换行文本
                token_ids.append(  # 添加换行标记ID
                    self.tokenizer.convert_tokens_to_ids("<patch_newline>")
                )
        return text, token_ids  # 返回文本和标记ID

    def _get_image_repl(  # 获取图像替换文本和标记ID
        self,
        num_images: int,  # 图像数量
    ) -> tuple[str, list[int]]:
        text = f"<im_start>{self.image_feature_placeholder}<im_end>"  # 图像特征占位文本
        token_ids = (  # 图像标记ID
            [self.tokenizer.convert_tokens_to_ids("<im_start>")]  # 图像起始标记ID
            + [self.image_token_id] * self.num_image_feature_size  # 图像特征标记ID
            + [self.tokenizer.convert_tokens_to_ids("<im_end>")]  # 图像结束标记ID
        )
        return text * num_images, token_ids * num_images  # 乘以图像数量

    def _get_image_repl_features(  # 获取图像替换特征（包含补丁和图像）
        self,
        num_images: int,  # 图像数量
        num_patches: int,  # 补丁数量
        patch_new_line_idx: Optional[list[bool]],  # 补丁换行索引
    ) -> tuple[str, list[int]]:
        if num_patches > 0:  # 如果有补丁
            patch_repl, patch_repl_ids = self._get_patch_repl(  # 获取补丁替换
                num_patches, patch_new_line_idx
            )
        else:  # 否则
            patch_repl = ""  # 空文本
            patch_repl_ids = []  # 空ID列表
        image_repl, image_repl_ids = self._get_image_repl(num_images)  # 获取图像替换
        return patch_repl + image_repl, patch_repl_ids + image_repl_ids  # 拼接并返回

    def replace_placeholder(self, text: str, placeholder: str, repls: list[str]) -> str:  # 替换占位符
        parts = text.split(placeholder)  # 按占位符分割文本

        if len(parts) - 1 != len(repls):  # 检查占位符和替换数量是否匹配
            raise ValueError(
                "The number of placeholders does not match the number of replacements."  # noqa: E501
            )

        result = [parts[0]]  # 结果以第一部分开始
        for i, repl in enumerate(repls):  # 遍历替换文本
            result.append(repl)  # 添加替换文本
            result.append(parts[i + 1])  # 添加下一部分

        return "".join(result)  # 拼接结果

    def __call__(  # 调用处理器
        self,
        text: Optional[Union[str, list[str]]] = None,  # 输入文本
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,  # 输入图像
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量类型
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ) -> BatchFeature:
        if text is None:  # 如果文本为空
            text = []  # 使用空列表
        if not isinstance(text, list):  # 如果文本不是列表
            text = [text]  # 转为列表
        if images is None:  # 如果图像为空
            images = []  # 使用空列表
        if not isinstance(images, list):  # 如果图像不是列表
            images = [images]  # 转为列表

        if len(images) == 0:  # 如果没有图像
            image_inputs = {}  # 空图像输入
            text_inputs = self.tokenizer(text)  # 仅分词
        else:  # 有图像
            splitted_images_data = self._split_images(images)  # 分割图像
            pixel_values_lst = []  # 像素值列表
            patch_pixel_values_lst = []  # 补丁像素值列表
            patch_newline_mask_lst = []  # 补丁换行掩码列表
            image_repl_str_lst = []  # 图像替换字符串列表
            image_repl_ids_lst = []  # 图像替换ID列表
            num_patches = []  # 补丁数量列表
            for (  # 遍历分割后的图像数据
                raw_img,
                img_patches,
                patch_newline_mask,
            ) in splitted_images_data:  # noqa: E501
                pixel_values_lst.extend(self._convert_images_to_pixel_values([raw_img]))  # 转换原图像素值

                if len(img_patches) > 0:  # 如果有补丁
                    patch_pixel_values_lst.extend(  # 转换补丁像素值
                        self._convert_images_to_pixel_values(img_patches, is_patch=True)
                    )
                num_patches.append(len(img_patches))  # 记录补丁数量

                image_repl_str, image_repl_ids = self._get_image_repl_features(  # 获取图像替换特征
                    1, len(img_patches), patch_newline_mask
                )
                image_repl_str_lst.append(image_repl_str)  # 添加替换字符串
                image_repl_ids_lst.extend(image_repl_ids)  # 添加替换ID

                if patch_newline_mask is not None:  # 如果有换行掩码
                    patch_newline_mask_lst.extend(patch_newline_mask)  # 添加到列表

            image_inputs = {  # 构建图像输入字典
                "pixel_values": torch.cat(pixel_values_lst),  # 像素值
                "num_patches": num_patches,  # 补丁数量
            }
            if patch_pixel_values_lst:  # 如果有补丁像素值
                image_inputs["patch_pixel_values"] = torch.cat(patch_pixel_values_lst)  # 添加补丁像素值
            if patch_newline_mask_lst:  # 如果有换行掩码
                image_inputs["patch_newline_mask"] = torch.tensor(  # 转为张量
                    patch_newline_mask_lst, dtype=torch.bool
                )

            text = [  # 替换文本中的占位符
                self.replace_placeholder(t, self.image_token, image_repl_str_lst)
                for t in text  # 遍历每条文本
            ]
            text_inputs = self.tokenizer(text)  # 分词

        return BatchFeature(  # 返回批次特征
            {
                **text_inputs,  # 文本输入
                **image_inputs,  # 图像输入
            },
            tensor_type=return_tensors,  # 张量类型
        )


################################################


class Step3VLImageProcessor(SGLangBaseProcessor):  # Step3-VL图像处理器，继承SGLang基础处理器
    models = [  # 关联的模型列表
        Step3VLForConditionalGeneration,  # Step3-VL模型
        StepVLForConditionalGeneration,  # Step-VL模型
        Step3p7ForConditionalGeneration,  # Step3p7模型
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化Step3-VL图像处理器
        # TODO, check _processor is tokenizer or processor.
        processor = Step3VLProcessor(hf_config, _processor)  # 创建Step3-VL处理器
        super().__init__(hf_config, server_args, processor, *args, **kwargs)  # 调用父类初始化
        self.IM_TOKEN = "<im_patch>"  # 图像标记
        self.IM_TOKEN_ID = self._processor.tokenizer.get_vocab()[self.IM_TOKEN]  # 获取图像标记ID
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token=self.IM_TOKEN,  # 图像标记
            image_token_id=self.IM_TOKEN_ID,  # 图像标记ID
            image_token_regex=re.compile(r"(?:<im_patch>)"),  # 图像标记正则表达式
        ).build(_processor)  # 构建标记对象

        mean = [0.48145466, 0.4578275, 0.40821073]  # 归一化均值
        std = [0.26862954, 0.26130258, 0.27577711]  # 归一化标准差

    def preprocess(self, image):  # 预处理图像
        return {"pixel_values": self.transform(image).unsqueeze(0)}  # 变换并增加批次维度

    def __call__(self, image):  # 调用预处理器
        return self.preprocess(image)  # 调用预处理方法

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        input_text: str | List[int],  # 输入文本或标记ID列表
        request_obj,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            video_data=request_obj.video_data,  # 视频数据
            multimodal_tokens=self.mm_tokens,  # 多模态标记
        )

        mm_items, input_ids, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 基础输出和多模态标记
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入ID列表
            mm_items=mm_items,  # 多模态数据项
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记ID
        )
