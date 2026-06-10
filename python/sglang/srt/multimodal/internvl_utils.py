# 本文件提供 InternVL 系列模型的多模态图像/视频预处理工具函数，包括图像变换构建、
# 动态分辨率裁剪、像素值转换、以及视频帧尺寸计算等功能
# copy from https://huggingface.co/OpenGVLab/InternVL3-1B
import math  # 导入数学模块，用于平方根等计算

import torch  # 导入 PyTorch 深度学习框架
import torchvision.transforms as T  # 导入 torchvision 的变换模块
from PIL import Image  # 导入 PIL 图像处理库
from torchvision.transforms.functional import InterpolationMode  # 导入插值模式枚举

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # ImageNet 数据集的均值（RGB 三通道）
IMAGENET_STD = (0.229, 0.224, 0.225)  # ImageNet 数据集的标准差（RGB 三通道）


def build_transform(
    input_size,  # 输入图像的目标尺寸（正方形边长）
    *,  # 以下参数必须以关键字参数形式传入
    mean: tuple[float, float, float],  # 归一化均值
    std: tuple[float, float, float],  # 归一化标准差
):
    """构建图像预处理变换流水线，包括颜色转换、缩放、转张量和归一化"""
    transform = T.Compose(  # 组合多个变换操作
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # 确保图像为 RGB 模式
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),  # 双三次插值缩放到目标尺寸
            T.ToTensor(),  # 将 PIL 图像转为 [0,1] 范围的张量
            T.Normalize(mean=mean, std=std),  # 使用给定均值和标准差进行归一化
        ]
    )
    return transform  # 返回组合变换


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """在目标宽高比列表中找到与原始图像宽高比最接近的比例"""
    best_ratio_diff = float("inf")  # 初始化最小宽高比差值为正无穷
    best_ratio = (1, 1)  # 初始化最佳比例为 (1, 1)
    area = width * height  # 计算图像面积
    for ratio in target_ratios:  # 遍历所有目标比例
        target_aspect_ratio = ratio[0] / ratio[1]  # 计算目标宽高比
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)  # 计算当前比例与目标的绝对差
        if ratio_diff < best_ratio_diff:  # 如果差值更小
            best_ratio_diff = ratio_diff  # 更新最小差值
            best_ratio = ratio  # 更新最佳比例
        elif ratio_diff == best_ratio_diff:  # 如果差值相等
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:  # 如果图像面积大于该比例对应面积的一半
                best_ratio = ratio  # 选择更大的比例
    return best_ratio  # 返回最接近的宽高比


def dynamic_preprocess(
    image: Image.Image,  # 输入的 PIL 图像
    *,  # 以下参数必须以关键字参数形式传入
    min_num: int,  # 最小分块数
    max_num: int,  # 最大分块数
    image_size: int,  # 每个分块的尺寸
    use_thumbnail: bool,  # 是否添加缩略图
) -> list[Image.Image]:  # 返回裁剪后的图像分块列表
    """动态预处理图像，根据宽高比将图像裁剪为多个分块，可选添加缩略图"""
    orig_width, orig_height = image.size  # 获取原始图像宽高
    aspect_ratio = orig_width / orig_height  # 计算原始宽高比

    # calculate the existing image aspect ratio
    target_ratios = set(  # 生成所有合法的目标宽高比组合
        (i, j)
        for n in range(min_num, max_num + 1)  # 遍历可能的分块总数
        for i in range(1, n + 1)  # 遍历行数
        for j in range(1, n + 1)  # 遍历列数
        if i * j <= max_num and i * j >= min_num  # 约束分块数在 [min_num, max_num] 范围内
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])  # 按分块总数升序排列

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(  # 找到最接近的宽高比
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]  # 计算目标宽度
    target_height = image_size * target_aspect_ratio[1]  # 计算目标高度
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]  # 计算总分块数

    # resize the image
    resized_img = image.resize((target_width, target_height))  # 将图像缩放到目标尺寸
    processed_images = []  # 存储裁剪后的图像分块
    for i in range(blocks):  # 遍历每个分块
        box = (  # 计算当前分块的裁剪区域 (left, upper, right, lower)
            (i % (target_width // image_size)) * image_size,  # 左边界
            (i // (target_width // image_size)) * image_size,  # 上边界
            ((i % (target_width // image_size)) + 1) * image_size,  # 右边界
            ((i // (target_width // image_size)) + 1) * image_size,  # 下边界
        )
        # split the image
        split_img = resized_img.crop(box)  # 裁剪当前分块
        processed_images.append(split_img)  # 添加到列表
    assert len(processed_images) == blocks  # 确保分块数量正确
    if use_thumbnail and len(processed_images) != 1:  # 如果需要缩略图且分块数不为1
        thumbnail_img = image.resize((image_size, image_size))  # 生成缩略图
        processed_images.append(thumbnail_img)  # 将缩略图添加到列表末尾
    return processed_images  # 返回处理后的图像列表


def image_to_pixel_values(
    image: Image.Image,  # 输入的 PIL 图像
    *,  # 以下参数必须以关键字参数形式传入
    input_size: int,  # 输入尺寸
    min_num_tiles: int = 1,  # 最小分块数，默认为 1
    max_num_tiles: int,  # 最大分块数
    use_thumbnail: bool,  # 是否使用缩略图
    mean: tuple[float, float, float] = IMAGENET_MEAN,  # 归一化均值
    std: tuple[float, float, float] = IMAGENET_STD,  # 归一化标准差
) -> torch.Tensor:  # 返回像素值张量
    """将图像转换为像素值张量，包括动态预处理和归一化变换"""
    images = dynamic_preprocess(  # 对图像进行动态预处理
        image,
        min_num=min_num_tiles,
        max_num=max_num_tiles,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
    )
    transform = build_transform(input_size, mean=mean, std=std)  # 构建图像变换流水线
    pixel_values = [transform(image) for image in images]  # 对每个分块应用变换
    pixel_values = torch.stack(pixel_values)  # 将所有分块堆叠为张量
    return pixel_values  # 返回像素值张量


def compute_dynamic_image_size(
    orig_w: int,  # 原始宽度
    orig_h: int,  # 原始高度
    patch_size: int,  # 每个补丁的像素尺寸
    downsample_ratio: float,  # 下采样比率
    min_num_patches: int,  # 最小补丁数
    max_num_patches: int,  # 最大补丁数
) -> tuple[int, int, int]:  # 返回 (目标宽度, 目标高度, token数)
    """Compute optimal resize dimensions for dynamic resolution.
    计算动态分辨率的最佳缩放尺寸。

    The image is resized (not tiled) to a variable size that respects the
    aspect ratio while staying within the patch budget. Dimensions are
    snapped to multiples of ``patch_size * ds`` so that pixel-shuffle
    downsampling produces integer grid sizes.

    Returns:
        (target_w, target_h, num_tokens) where num_tokens is the
        post-pixel-shuffle token count.
    """
    ds = int(1 / downsample_ratio)  # 计算下采样因子
    snap = patch_size * ds  # 计算对齐步长（宽高必须是其倍数）

    pw = max(1, round(orig_w / patch_size))  # 计算原始宽度对应的补丁数
    ph = max(1, round(orig_h / patch_size))  # 计算原始高度对应的补丁数
    native_patches = pw * ph  # 计算原始补丁总数

    budget = min(native_patches, max_num_patches)  # 限制补丁预算不超过最大值
    budget = max(budget, min_num_patches)  # 确保补丁预算不小于最小值
    factor = math.sqrt(budget / max(native_patches, 1))  # 计算缩放因子
    factor = min(factor, 1.0)  # 缩放因子不超过 1（不放大）

    target_pw = max(ds, int(round(pw * factor / ds)) * ds)  # 计算目标宽度补丁数并对齐
    target_ph = max(ds, int(round(ph * factor / ds)) * ds)  # 计算目标高度补丁数并对齐

    if target_pw * target_ph < min_num_patches:  # 如果补丁数低于最小值
        up = math.sqrt(min_num_patches / (target_pw * target_ph))  # 计算上调因子
        target_pw = max(ds, int(math.ceil(target_pw * up / ds)) * ds)  # 向上调整宽度补丁数
        target_ph = max(ds, int(math.ceil(target_ph * up / ds)) * ds)  # 向上调整高度补丁数

    if target_pw * target_ph > max_num_patches:  # 如果补丁数超过最大值
        down = math.sqrt(max_num_patches / (target_pw * target_ph))  # 计算下调因子
        target_pw = max(ds, int(math.floor(target_pw * down / ds)) * ds)  # 向下调整宽度补丁数
        target_ph = max(ds, int(math.floor(target_ph * down / ds)) * ds)  # 向下调整高度补丁数

    target_w = target_pw * patch_size  # 将补丁数转换为像素宽度
    target_h = target_ph * patch_size  # 将补丁数转换为像素高度
    num_tokens = (target_pw * target_ph) // (ds * ds)  # 计算像素混洗后的 token 数

    return target_w, target_h, num_tokens  # 返回目标尺寸和 token 数


def dynamic_resize_image(
    image: Image.Image,  # 输入的 PIL 图像
    patch_size: int,  # 补丁尺寸
    downsample_ratio: float,  # 下采样比率
    min_num_patches: int,  # 最小补丁数
    max_num_patches: int,  # 最大补丁数
    mean: tuple[float, float, float] = IMAGENET_MEAN,  # 归一化均值
    std: tuple[float, float, float] = IMAGENET_STD,  # 归一化标准差
) -> tuple[torch.Tensor, int]:  # 返回 (像素值张量, token数)
    """Resize image for dynamic resolution and return pixel tensor + token count.
    动态调整图像大小并返回像素张量和 token 数。

    Returns:
        (pixel_values [1, 3, H, W], num_tokens)
    """
    orig_w, orig_h = image.size  # 获取原始图像尺寸
    target_w, target_h, num_tokens = compute_dynamic_image_size(  # 计算目标尺寸和 token 数
        orig_w,
        orig_h,
        patch_size,
        downsample_ratio,
        min_num_patches,
        max_num_patches,
    )
    image = image.convert("RGB")  # 确保图像为 RGB 模式
    image = image.resize((target_w, target_h), Image.BICUBIC)  # 双三次插值缩放到目标尺寸
    transform = T.Compose(  # 构建变换流水线
        [
            T.ToTensor(),  # 转为张量
            T.Normalize(mean=mean, std=std),  # 归一化
        ]
    )
    pixel_values = transform(image).unsqueeze(0)  # 应用变换并增加批次维度
    return pixel_values, num_tokens  # 返回像素值张量和 token 数


def resize_image_to_pixels(
    image: Image.Image,  # 输入的 PIL 图像
    target_w: int,  # 目标宽度
    target_h: int,  # 目标高度
    mean: tuple[float, float, float] = IMAGENET_MEAN,  # 归一化均值
    std: tuple[float, float, float] = IMAGENET_STD,  # 归一化标准差
) -> torch.Tensor:  # 返回像素值张量
    """Resize image to exact target dimensions and return normalized tensor.
    将图像缩放到精确的目标尺寸并返回归一化张量。

    Returns:
        pixel_values tensor of shape [1, 3, target_h, target_w].
    """
    image = image.convert("RGB")  # 确保图像为 RGB 模式
    image = image.resize((target_w, target_h), Image.BICUBIC)  # 双三次插值缩放到目标尺寸
    transform = T.Compose(  # 构建变换流水线
        [
            T.ToTensor(),  # 转为张量
            T.Normalize(mean=mean, std=std),  # 归一化
        ]
    )
    return transform(image).unsqueeze(0)  # 应用变换并增加批次维度后返回


def compute_budgeted_image_sizes(
    image_sizes: list[tuple[int, int]],  # 图像尺寸列表 [(宽, 高), ...]
    total_token_budget: int,  # 总 token 预算
    patch_size: int,  # 补丁尺寸
    downsample_ratio: float,  # 下采样比率
    min_num_patches: int,  # 每张图最小补丁数
    max_num_patches: int,  # 每张图最大补丁数
    max_iterations: int = 10,  # 最大迭代次数
) -> list[tuple[int, int, int]]:  # 返回每张图的 (目标宽, 目标高, token数)
    """Compute per-image sizes that fit within a total token budget.
    在总 token 预算内计算每张图的尺寸，迭代调整各图补丁上限直到总 token 数不超预算。

    When multiple images share a prompt, their combined post-pixel-shuffle
    tokens must not exceed ``total_token_budget``.  This function iteratively
    reduces per-image patch limits until the total fits.

    Returns:
        List of (target_w, target_h, num_tokens) per image.
    """
    n = len(image_sizes)  # 获取图像数量
    if n == 0:  # 如果没有图像
        return []  # 返回空列表

    ds = int(round(1 / downsample_ratio))  # 计算下采样因子
    per_image_max = [max_num_patches] * n  # 初始化每张图的补丁上限为全局最大值
    results: list[tuple[int, int, int]] = []  # 存储计算结果

    for _ in range(max_iterations):  # 迭代调整
        results = [  # 计算每张图的动态尺寸
            compute_dynamic_image_size(
                orig_w,
                orig_h,
                patch_size,
                downsample_ratio,
                min_num_patches,
                per_image_max[i],
            )
            for i, (orig_w, orig_h) in enumerate(image_sizes)  # 遍历每张图的原始尺寸
        ]
        total_tokens = sum(num_tokens for _, _, num_tokens in results)  # 计算 token 总数

        if total_tokens <= total_token_budget:  # 如果不超预算
            return results  # 返回结果

        scale = total_token_budget / total_tokens  # 计算缩放比例
        for i in range(n):  # 遍历每张图
            current_patches = results[i][2] * ds * ds  # 计算当前补丁数
            per_image_max[i] = max(min_num_patches, int(current_patches * scale))  # 按比例降低上限

    return results  # 返回最终结果（可能仍未完全满足预算）


def get_video_target_size_and_feature_size(
    orig_w: int,  # 原始宽度
    orig_h: int,  # 原始高度
    target_num_patches: int,  # 目标补丁数
    maintain_aspect_ratio: bool,  # 是否保持宽高比
    patch_size: int,  # 补丁尺寸
    downsample_ratio: float,  # 下采样比率
) -> tuple[int, int, int]:  # 返回 (目标宽, 目标高, 特征大小)
    """Compute target resize dimensions and post-downsample token count for video.
    计算视频帧的目标缩放尺寸和下采样后的 token 数。

    Single source of truth for video spatial dimensions — used by both
    video_to_pixel_values (resize) and the processor (token counting).

    Returns:
        (target_w, target_h, feature_size) where feature_size is the
        post-pixel-shuffle token count.
    """
    ds = int(1 / downsample_ratio)  # 计算下采样因子

    if target_num_patches > 0 and maintain_aspect_ratio:  # 如果需要保持宽高比
        aspect = orig_w / max(orig_h, 1)  # 计算宽高比
        ph = math.sqrt(target_num_patches / max(aspect, 1e-6))  # 根据补丁数和宽高比计算高度补丁数
        pw = ph * aspect  # 计算宽度补丁数
        target_pw = max(ds, int(round(pw / ds)) * ds)  # 对齐宽度补丁数
        target_ph = max(ds, int(round(ph / ds)) * ds)  # 对齐高度补丁数
    elif target_num_patches > 0:  # 如果不需要保持宽高比
        side = int(math.sqrt(target_num_patches))  # 按正方形计算边长补丁数
        target_pw = max(ds, int(round(side / ds)) * ds)  # 对齐宽度补丁数
        target_ph = target_pw  # 高度等于宽度（正方形）
    else:  # 如果没有指定目标补丁数
        target_pw = max(ds, round(orig_w / patch_size / ds) * ds)  # 按原始尺寸计算并对齐宽度
        target_ph = max(ds, round(orig_h / patch_size / ds) * ds)  # 按原始尺寸计算并对齐高度

    target_w = target_pw * patch_size  # 将补丁数转换为像素宽度
    target_h = target_ph * patch_size  # 将补丁数转换为像素高度
    feature_size = (target_pw // ds) * (target_ph // ds)  # 计算下采样后的特征大小

    return target_w, target_h, feature_size  # 返回目标尺寸和特征大小


def video_to_pixel_values(
    frame: Image.Image,  # 输入的视频帧（PIL 图像）
    patch_size: int,  # 补丁尺寸
    downsample_ratio: float,  # 下采样比率
    target_num_patches: int,  # 目标补丁数
    maintain_aspect_ratio: bool,  # 是否保持宽高比
    mean: tuple[float, float, float] = IMAGENET_MEAN,  # 归一化均值
    std: tuple[float, float, float] = IMAGENET_STD,  # 归一化标准差
) -> tuple[torch.Tensor, int]:  # 返回 (像素值张量, 特征大小)
    """Resize a single video frame for temporal compression pipeline.
    缩放单个视频帧用于时序压缩流水线。

    Returns:
        (pixel_values [1, 3, H, W], feature_size) where feature_size is
        the post-pixel-shuffle token count.
    """
    orig_w, orig_h = frame.size  # 获取帧的原始尺寸
    target_w, target_h, feature_size = get_video_target_size_and_feature_size(  # 计算目标尺寸和特征大小
        orig_w,
        orig_h,
        target_num_patches,
        maintain_aspect_ratio,
        patch_size,
        downsample_ratio,
    )

    frame = frame.convert("RGB")  # 确保帧为 RGB 模式
    frame = frame.resize((target_w, target_h), Image.BICUBIC)  # 双三次插值缩放到目标尺寸
    transform = T.Compose(  # 构建变换流水线
        [
            T.ToTensor(),  # 转为张量
            T.Normalize(mean=mean, std=std),  # 归一化
        ]
    )
    pixel_values = transform(frame).unsqueeze(0)  # 应用变换并增加批次维度
    return pixel_values, feature_size  # 返回像素值张量和特征大小
