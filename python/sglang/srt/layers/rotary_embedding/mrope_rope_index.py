# 多模态旋转位置编码(MRoPE)的rope_index计算模块
# 为Qwen2-VL/Qwen3-VL、Qwen3-Omni、GLM4V、Ernie4.5等模型提供get_rope_index实现
"""get_rope_index implementations for Qwen2-VL/Qwen3-VL, Qwen3-Omni, GLM4V, Ernie4.5."""  # get_rope_index的实现，用于Qwen2-VL/Qwen3-VL、Qwen3-Omni、GLM4V、Ernie4.5

from __future__ import annotations  # 启用延迟注解评估

import itertools  # 迭代工具模块
from typing import Any, List, Optional, Tuple, Union  # 类型提示工具

import torch  # PyTorch深度学习框架


def _get_feat_extract_output_lengths(input_lengths):  # 计算音频特征提取后的输出长度
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    计算卷积层的输出长度和音频编码器的输出长度
    """
    input_lengths_leave = input_lengths % 100  # 计算输入长度除以100的余数
    feat_lengths = (input_lengths_leave - 1) // 2 + 1  # 计算特征长度
    output_lengths = (  # 计算最终输出长度
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13  # 综合计算输出长度
    )
    return output_lengths  # 返回输出长度


def _get_llm_pos_ids_for_vision(  # 为视觉token生成LLM位置ID（时间、高度、宽度三维）
    st_idx, vision_idx, spatial_merge_size, t_index, grid_hs, grid_ws, device  # 起始索引、视觉索引、空间合并大小、时间索引、网格高列表、网格宽列表、设备
):
    grid_h = grid_hs[vision_idx] // spatial_merge_size  # 计算当前视觉块的网格高度（除以空间合并大小）
    grid_w = grid_ws[vision_idx] // spatial_merge_size  # 计算当前视觉块的网格宽度（除以空间合并大小）

    h_index = (  # 生成高度维度的位置索引
        torch.arange(grid_h, device=device)  # 生成0到grid_h-1的序列
        .view(1, -1, 1)  # 重塑为(1, grid_h, 1)的形状
        .expand(len(t_index), -1, grid_w)  # 扩展到(time_len, grid_h, grid_w)
        .flatten()  # 展平为一维
    )
    w_index = (  # 生成宽度维度的位置索引
        torch.arange(grid_w, device=device)  # 生成0到grid_w-1的序列
        .view(1, 1, -1)  # 重塑为(1, 1, grid_w)的形状
        .expand(len(t_index), grid_h, -1)  # 扩展到(time_len, grid_h, grid_w)
        .flatten()  # 展平为一维
    )
    t_index = t_index.view(-1, 1).expand(-1, grid_h * grid_w).flatten()  # 将时间索引扩展并与空间维度对齐后展平

    llm_pos_ids = torch.stack([t_index, h_index, w_index], dim=0) + st_idx  # 堆叠三个维度的索引并加上起始偏移
    return llm_pos_ids  # 返回LLM位置ID，形状为(3, total_tokens)


def get_rope_index(  # 获取多模态旋转位置编码的位置索引（主入口函数）
    spatial_merge_size: int,  # 空间合并大小
    image_token_id: int,  # 图像token的ID
    video_token_id: int,  # 视频token的ID
    vision_start_token_id: int,  # 视觉起始token的ID
    model_type: str,  # 模型类型字符串
    tokens_per_second: Optional[int] = None,  # 每秒token数
    input_ids: Optional[torch.LongTensor] = None,  # 输入token ID张量
    image_grid_thw: Optional[torch.LongTensor] = None,  # 图像网格的时间-高度-宽度信息
    video_grid_thw: Optional[torch.LongTensor] = None,  # 视频网格的时间-高度-宽度信息
    second_per_grid_ts: Optional[torch.Tensor] = None,  # 每个网格的时间间隔（秒）
    **kwargs,  # 其他关键字参数
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回位置ID和位置增量的元组
    if model_type == "qwen3_omni_moe":  # 如果是Qwen3-Omni-MoE模型
        return get_rope_index_qwen3_omni(  # 调用Qwen3-Omni专用的rope_index函数
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            tokens_per_second,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            **kwargs,
        )
    if (  # 如果是Qwen3-VL系列模型且视频网格信息不为空
        model_type.startswith("qwen3_vl")  # 检查是否以qwen3_vl开头
        or model_type.startswith("qwen3_vl_moe")  # 检查是否以qwen3_vl_moe开头
        or model_type.startswith("qwen3_5")  # 检查是否以qwen3_5开头
    ) and video_grid_thw is not None:  # 且视频网格信息不为空
        video_grid_thw = torch.repeat_interleave(  # 按时间维度重复视频网格行
            video_grid_thw, video_grid_thw[:, 0], dim=0  # 根据每行的t值进行重复
        )
        video_grid_thw[:, 0] = 1  # 将时间维度设为1

    mrope_position_deltas = []  # 存储每个样本的位置增量
    if input_ids is not None and (  # 如果输入ID存在且存在视觉信息
        image_grid_thw is not None or video_grid_thw is not None  # 图像或视频网格信息不为空
    ):
        total_input_ids = input_ids  # 保存原始输入ID用于后续遍历
        position_ids = torch.ones(  # 初始化位置ID张量，全1
            3,  # 三个维度（时间、高度、宽度）
            input_ids.shape[0],  # 批次大小
            input_ids.shape[1],  # 序列长度
            dtype=input_ids.dtype,  # 数据类型与输入一致
            device=input_ids.device,  # 设备与输入一致
        )
        image_index, video_index = 0, 0  # 初始化图像和视频的索引计数器
        for i, input_ids in enumerate(total_input_ids):  # 遍历批次中的每个样本
            image_nums, video_nums = 0, 0  # 初始化当前样本的图像和视频数量
            vision_start_indices = torch.argwhere(  # 找到视觉起始token的位置
                input_ids == vision_start_token_id
            ).squeeze(1)  # 去除多余维度
            vision_tokens = input_ids[vision_start_indices + 1]  # 获取视觉起始token后面的token（图像或视频token）
            image_nums = (vision_tokens == image_token_id).sum()  # 统计图像token数量
            video_nums = (vision_tokens == video_token_id).sum()  # 统计视频token数量
            input_tokens = input_ids.tolist()  # 将输入token转为Python列表
            llm_pos_ids_list: list = []  # 存储LLM位置ID的列表
            st = 0  # 当前处理位置的起始点
            remain_images, remain_videos = image_nums, video_nums  # 剩余待处理的图像和视频数量
            for _ in range(image_nums + video_nums):  # 遍历所有图像和视频
                if image_token_id in input_tokens and remain_images > 0:  # 如果还有剩余图像token
                    ed_image = input_tokens.index(image_token_id, st)  # 找到下一个图像token的位置
                else:
                    ed_image = len(input_tokens) + 1  # 设为超出范围，表示没有更多图像
                if video_token_id in input_tokens and remain_videos > 0:  # 如果还有剩余视频token
                    ed_video = input_tokens.index(video_token_id, st)  # 找到下一个视频token的位置
                else:
                    ed_video = len(input_tokens) + 1  # 设为超出范围，表示没有更多视频
                if ed_image < ed_video:  # 图像token先出现
                    t, h, w = (  # 获取当前图像的网格维度
                        image_grid_thw[image_index][0],  # 时间维度
                        image_grid_thw[image_index][1],  # 高度维度
                        image_grid_thw[image_index][2],  # 宽度维度
                    )
                    second_per_grid_t = 0  # 图像的时间间隔为0
                    image_index += 1  # 图像索引加1
                    remain_images -= 1  # 剩余图像数减1
                    ed = ed_image  # 当前结束位置为图像token位置
                else:  # 视频token先出现
                    t, h, w = (  # 获取当前视频的网格维度
                        video_grid_thw[video_index][0],  # 时间维度
                        video_grid_thw[video_index][1],  # 高度维度
                        video_grid_thw[video_index][2],  # 宽度维度
                    )
                    if second_per_grid_ts is not None:  # 如果提供了时间间隔信息
                        second_per_grid_t = second_per_grid_ts[video_index]  # 获取当前视频的时间间隔
                    else:
                        second_per_grid_t = 1.0  # 默认时间间隔为1.0秒
                    video_index += 1  # 视频索引加1
                    remain_videos -= 1  # 剩余视频数减1
                    ed = ed_video  # 当前结束位置为视频token位置
                t_int, h_int, w_int = int(t), int(h), int(w)  # 将维度转为整数
                llm_grid_t = t_int  # LLM网格的时间维度
                llm_grid_h = h_int // spatial_merge_size  # LLM网格的高度维度（除以空间合并大小）
                llm_grid_w = w_int // spatial_merge_size  # LLM网格的宽度维度（除以空间合并大小）
                text_len = ed - st  # 前面文本段的长度
                st_idx = (  # 计算当前位置的起始索引
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0  # 如果列表非空则取最大值+1，否则为0
                )
                llm_pos_ids_list.append(  # 添加文本段的位置ID
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx  # 生成文本位置并加上起始偏移
                )
                if model_type in ("qwen2_5_vl", "paddleocr_vl"):  # Qwen2.5-VL或PaddleOCR-VL模型
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)  # 生成时间范围张量
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)  # 扩展到所有空间位置
                    time_tensor = expanded_range * second_per_grid_t * tokens_per_second  # 计算时间张量
                    t_index = time_tensor.long().flatten()  # 转为长整型并展平
                elif model_type in (  # 支持的模型类型列表
                    "qwen2_vl",
                    "qwen3_vl",
                    "qwen3_vl_moe",
                    "qwen3_5",
                    "qwen3_5_moe",
                    "intern_s2_preview",
                ):
                    t_index = (  # 生成时间维度的位置索引
                        torch.arange(llm_grid_t, device=position_ids.device)  # 生成0到llm_grid_t-1的序列
                        .view(-1, 1)  # 重塑为列向量
                        .expand(llm_grid_t, llm_grid_h * llm_grid_w)  # 扩展到所有空间位置
                        .reshape(-1)  # 展平为一维
                    )
                else:
                    raise RuntimeError(f"Unimplemented model type: {model_type}")  # 不支持的模型类型抛出异常
                h_index = (  # 生成高度维度的位置索引
                    torch.arange(llm_grid_h, device=position_ids.device)  # 生成0到llm_grid_h-1的序列
                    .view(1, -1, 1)  # 重塑为(1, grid_h, 1)的形状
                    .expand(llm_grid_t, llm_grid_h, llm_grid_w)  # 扩展到三维网格
                    .reshape(-1)  # 展平为一维
                )
                w_index = (  # 生成宽度维度的位置索引
                    torch.arange(llm_grid_w, device=position_ids.device)  # 生成0到llm_grid_w-1的序列
                    .view(1, 1, -1)  # 重塑为(1, 1, grid_w)的形状
                    .expand(llm_grid_t, llm_grid_h, llm_grid_w)  # 扩展到三维网格
                    .reshape(-1)  # 展平为一维
                )
                llm_pos_ids_list.append(  # 添加视觉token的位置ID
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx  # 堆叠三维索引并加上偏移
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w  # 更新起始位置，跳过视觉token
            if st < len(input_tokens):  # 如果还有剩余的文本token
                st_idx = (  # 计算尾部文本的起始索引
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0  # 如果列表非空则取最大值+1，否则为0
                )
                text_len = len(input_tokens) - st  # 计算尾部文本长度
                llm_pos_ids_list.append(  # 添加尾部文本的位置ID
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx  # 生成文本位置并加上起始偏移
                )
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)  # 拼接所有位置ID并重塑形状
            position_ids[..., i, :] = llm_positions.to(position_ids.device)  # 将位置ID赋值到对应批次位置
            mrope_position_deltas.append(  # 记录位置增量
                llm_positions.max() + 1 - len(total_input_ids[i])  # 最大位置+1减去序列长度
            )
        mrope_position_deltas = torch.tensor(  # 将位置增量转为张量
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)  # 增加一个维度
        return position_ids, mrope_position_deltas  # 返回位置ID和位置增量
    else:  # 没有视觉信息的情况
        s = input_ids.shape[1]  # 获取序列长度
        position_ids = torch.arange(s)  # 生成0到s-1的位置序列
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)  # 扩展为三维并转移到对应设备
        max_position_ids = position_ids.amax(dim=0, keepdim=False)  # 在维度0上取最大值
        mrope_position_deltas = max_position_ids.amax(-1, keepdim=True) + 1 - s  # 计算位置增量
        return position_ids, mrope_position_deltas  # 返回位置ID和位置增量


def get_rope_index_qwen3_omni(  # 获取Qwen3-Omni模型的MRoPE位置索引（支持音频、图像、视频混合）
    spatial_merge_size: int,  # 空间合并大小
    image_token_id: int,  # 图像token的ID
    video_token_id: int,  # 视频token的ID
    vision_start_token_id: int,  # 视觉起始token的ID
    tokens_per_second: Optional[int] = None,  # 每秒token数
    input_ids: Optional[torch.LongTensor] = None,  # 输入token ID张量
    image_grid_thw: Optional[torch.LongTensor] = None,  # 图像网格的时间-高度-宽度信息
    video_grid_thw: Optional[torch.LongTensor] = None,  # 视频网格的时间-高度-宽度信息
    second_per_grid_ts: Optional[torch.Tensor] = None,  # 每个网格的时间间隔（秒）
    **kwargs,  # 其他关键字参数
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回位置ID和位置增量的元组
    audio_token_id = kwargs["audio_token_id"]  # 获取音频token的ID
    audio_start_token_id = kwargs["audio_start_token_id"]  # 获取音频起始token的ID
    position_id_per_seconds = kwargs["position_id_per_seconds"]  # 获取每秒的位置ID增量
    use_audio_in_video = kwargs.get("use_audio_in_video", False)  # 是否在视频中使用音频
    audio_seqlens = kwargs.get("audio_seqlens", None)  # 音频序列长度列表
    second_per_grids = second_per_grid_ts  # 每个网格的时间间隔

    mrope_position_deltas = []  # 存储每个样本的位置增量
    if input_ids is not None and (  # 如果输入ID存在且存在视觉信息
        image_grid_thw is not None or video_grid_thw is not None  # 图像或视频网格信息不为空
    ):
        total_input_ids = input_ids  # 保存原始输入ID用于后续遍历
        position_ids = torch.zeros(  # 初始化位置ID张量，全0
            3,  # 三个维度（时间、高度、宽度）
            input_ids.shape[0],  # 批次大小
            input_ids.shape[1],  # 序列长度
            dtype=torch.float,  # 使用浮点类型
            device=input_ids.device,  # 设备与输入一致
        )
        image_idx, video_idx, audio_idx = 0, 0, 0  # 初始化图像、视频、音频的索引计数器
        for i, current_input_ids in enumerate(total_input_ids):  # 遍历批次中的每个样本
            image_nums, video_nums, audio_nums = 0, 0, 0  # 初始化当前样本的图像、视频、音频数量
            vision_start_indices = torch.argwhere(  # 找到视觉起始token的位置
                current_input_ids == vision_start_token_id
            ).squeeze(1)  # 去除多余维度
            if vision_start_indices.numel() > 0:  # 如果存在视觉起始token
                vision_tokens = current_input_ids[vision_start_indices + 1]  # 获取视觉起始token后面的token
                image_nums = (vision_tokens == image_token_id).sum()  # 统计图像token数量
                video_nums = (  # 统计视频token数量
                    (vision_tokens == audio_start_token_id).sum()  # 如果音频在视频中，统计音频起始token
                    if use_audio_in_video  # 当use_audio_in_video为True时
                    else (vision_tokens == video_token_id).sum()  # 否则统计视频token
                )
            audio_nums = torch.sum(current_input_ids == audio_start_token_id)  # 统计音频起始token数量
            input_tokens = current_input_ids.tolist()  # 将输入token转为Python列表
            llm_pos_ids_list: list = []  # 存储LLM位置ID的列表
            st = 0  # 当前处理位置的起始点
            remain_images, remain_videos, remain_audios = (  # 剩余待处理的图像、视频、音频数量
                image_nums,
                video_nums,
                audio_nums,
            )
            multimodal_nums = (  # 计算多模态元素总数
                image_nums + audio_nums  # 音频在视频中时，不单独计算视频
                if use_audio_in_video
                else image_nums + video_nums + audio_nums  # 否则计算所有模态
            )
            for _ in range(multimodal_nums):  # 遍历所有多模态元素
                st_idx = (  # 计算当前位置的起始索引
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0  # 如果列表非空则取最大值+1，否则为0
                )
                ed_vision_start = (  # 找到下一个视觉起始token的位置
                    input_tokens.index(vision_start_token_id, st)  # 从st位置开始搜索
                    if (  # 如果满足条件
                        (
                            image_token_id in input_tokens  # 输入中包含图像token
                            or video_token_id in input_tokens  # 或包含视频token
                        )
                        and (remain_videos > 0 or remain_images > 0)  # 且还有剩余视觉内容
                    )
                    else len(input_tokens) + 1  # 否则设为超出范围
                )
                ed_audio_start = (  # 找到下一个音频起始token的位置
                    input_tokens.index(audio_start_token_id, st)  # 从st位置开始搜索
                    if (audio_token_id in input_tokens and remain_audios > 0)  # 如果输入中有音频且还有剩余
                    else len(input_tokens) + 1  # 否则设为超出范围
                )
                min_ed = min(ed_vision_start, ed_audio_start)  # 取最早出现的模态起始位置
                text_len = min_ed - st  # 前面文本段的长度
                if text_len != 0:  # 如果文本长度不为0
                    llm_pos_ids_list.append(  # 添加文本段的位置ID
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx  # 生成文本位置并加上起始偏移
                    )
                    st_idx += text_len  # 更新起始索引
                if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:  # 视觉和音频紧邻（音频在视频中）
                    bos_len, eos_len = 2, 2  # 起始和结束token长度为2
                else:
                    bos_len, eos_len = 1, 1  # 否则长度为1
                llm_pos_ids_list.append(  # 添加起始token的位置ID
                    torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx  # 生成起始token位置
                )
                st_idx += bos_len  # 更新起始索引
                # Audio Only  # 仅音频情况
                if min_ed == ed_audio_start:  # 如果最早出现的是音频起始token
                    audio_len = _get_feat_extract_output_lengths(  # 计算音频特征提取后的输出长度
                        audio_seqlens[audio_idx]
                    )
                    llm_pos_ids = (  # 生成音频的位置ID
                        torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx  # 三维相同位置
                    )
                    llm_pos_ids_list.append(llm_pos_ids)  # 添加音频位置ID到列表
                    st += int(text_len + bos_len + audio_len + eos_len)  # 更新处理位置
                    audio_idx += 1  # 音频索引加1
                    remain_audios -= 1  # 剩余音频数减1
                # Image Only  # 仅图像情况
                elif (
                    min_ed == ed_vision_start  # 最早出现的是视觉起始token
                    and current_input_ids[ed_vision_start + 1] == image_token_id  # 下一个token是图像token
                ):
                    grid_t = image_grid_thw[image_idx][0]  # 获取图像的时间维度
                    grid_hs = image_grid_thw[:, 1]  # 获取所有图像的高度维度列表
                    grid_ws = image_grid_thw[:, 2]  # 获取所有图像的宽度维度列表
                    t_index = (  # 计算时间维度的位置索引
                        torch.arange(grid_t) * 1 * position_id_per_seconds  # 乘以每秒位置增量
                    ).float()
                    llm_pos_ids = _get_llm_pos_ids_for_vision(  # 调用视觉位置ID生成函数
                        st_idx,
                        image_idx,
                        spatial_merge_size,
                        t_index,
                        grid_hs,
                        grid_ws,
                        input_ids.device,
                    )
                    image_len = image_grid_thw[image_idx].prod() // (  # 计算图像token总数
                        spatial_merge_size**2  # 除以空间合并大小的平方
                    )
                    llm_pos_ids_list.append(llm_pos_ids)  # 添加图像位置ID到列表
                    st += int(text_len + bos_len + image_len + eos_len)  # 更新处理位置
                    image_idx += 1  # 图像索引加1
                    remain_images -= 1  # 剩余图像数减1
                # Video Only  # 仅视频情况
                elif (
                    min_ed == ed_vision_start  # 最早出现的是视觉起始token
                    and current_input_ids[ed_vision_start + 1] == video_token_id  # 下一个token是视频token
                ):
                    grid_t = video_grid_thw[video_idx][0]  # 获取视频的时间维度
                    grid_hs = video_grid_thw[:, 1]  # 获取所有视频的高度维度列表
                    grid_ws = video_grid_thw[:, 2]  # 获取所有视频的宽度维度列表
                    t_index = (  # 计算时间维度的位置索引
                        torch.arange(grid_t)
                        * second_per_grids[video_idx].cpu().float()  # 乘以每个网格的时间间隔
                        * position_id_per_seconds  # 乘以每秒位置增量
                    ).float()
                    llm_pos_ids = _get_llm_pos_ids_for_vision(  # 调用视觉位置ID生成函数
                        st_idx,
                        video_idx,
                        spatial_merge_size,
                        t_index,
                        grid_hs,
                        grid_ws,
                        input_ids.device,
                    )
                    video_len = video_grid_thw[video_idx].prod() // (  # 计算视频token总数
                        spatial_merge_size**2  # 除以空间合并大小的平方
                    )
                    llm_pos_ids_list.append(llm_pos_ids)  # 添加视频位置ID到列表
                    st += int(text_len + bos_len + video_len + eos_len)  # 更新处理位置
                    video_idx += 1  # 视频索引加1
                    remain_videos -= 1  # 剩余视频数减1
                # Audio in Video  # 视频中的音频情况
                elif (
                    min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start  # 视觉起始和音频起始紧邻
                ):
                    audio_len = _get_feat_extract_output_lengths(  # 计算音频特征提取后的输出长度
                        audio_seqlens[audio_idx]
                    )
                    audio_llm_pos_ids = (  # 生成音频的位置ID
                        torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx  # 三维相同位置
                    )
                    grid_t = video_grid_thw[video_idx][0]  # 获取视频的时间维度
                    grid_hs = video_grid_thw[:, 1]  # 获取所有视频的高度维度列表
                    grid_ws = video_grid_thw[:, 2]  # 获取所有视频的宽度维度列表
                    t_index = (  # 计算时间维度的位置索引
                        torch.arange(grid_t)
                        * second_per_grids[video_idx].cpu().float()  # 乘以每个网格的时间间隔
                        * position_id_per_seconds  # 乘以每秒位置增量
                    ).float()
                    video_llm_pos_ids = _get_llm_pos_ids_for_vision(  # 调用视觉位置ID生成函数
                        st_idx,
                        video_idx,
                        spatial_merge_size,
                        t_index,
                        grid_hs,
                        grid_ws,
                        input_ids.device,
                    )
                    video_data_index, audio_data_index = 0, 0  # 初始化视频和音频的数据索引
                    while (  # 按时间顺序合并视频和音频的位置ID
                        video_data_index < video_llm_pos_ids.shape[-1]  # 视频数据未处理完
                        and audio_data_index < audio_llm_pos_ids.shape[-1]  # 音频数据未处理完
                    ):
                        if (  # 视频位置的时间维度更小或相等
                            video_llm_pos_ids[0][video_data_index]
                            <= audio_llm_pos_ids[0][audio_data_index]
                        ):
                            llm_pos_ids_list.append(  # 添加视频位置ID
                                video_llm_pos_ids[
                                    :, video_data_index : video_data_index + 1
                                ]
                            )
                            video_data_index += 1  # 视频数据索引加1
                        else:  # 音频位置的时间维度更小
                            llm_pos_ids_list.append(  # 添加音频位置ID
                                audio_llm_pos_ids[
                                    :, audio_data_index : audio_data_index + 1
                                ]
                            )
                            audio_data_index += 1  # 音频数据索引加1
                    if video_data_index < video_llm_pos_ids.shape[-1]:  # 如果视频数据还有剩余
                        llm_pos_ids_list.append(  # 添加剩余视频位置ID
                            video_llm_pos_ids[
                                :, video_data_index : video_llm_pos_ids.shape[-1]
                            ]
                        )
                    if audio_data_index < audio_llm_pos_ids.shape[-1]:  # 如果音频数据还有剩余
                        llm_pos_ids_list.append(  # 添加剩余音频位置ID
                            audio_llm_pos_ids[
                                :, audio_data_index : audio_llm_pos_ids.shape[-1]
                            ]
                        )
                    video_len = video_grid_thw[video_idx].prod() // (  # 计算视频token总数
                        spatial_merge_size**2  # 除以空间合并大小的平方
                    )
                    st += int(text_len + bos_len + audio_len + video_len + eos_len)  # 更新处理位置
                    audio_idx += 1  # 音频索引加1
                    video_idx += 1  # 视频索引加1
                    remain_videos -= 1  # 剩余视频数减1
                    remain_audios -= 1  # 剩余音频数减1
                st_idx = (  # 更新起始索引
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0  # 如果列表非空则取最大值+1，否则为0
                )
                llm_pos_ids_list.append(  # 添加结束token的位置ID
                    torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx  # 生成结束token位置
                )
            if st < len(input_tokens):  # 如果还有剩余的文本token
                st_idx = (  # 计算尾部文本的起始索引
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0  # 如果列表非空则取最大值+1，否则为0
                )
                text_len = len(input_tokens) - st  # 计算尾部文本长度
                llm_pos_ids_list.append(  # 添加尾部文本的位置ID
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx  # 生成文本位置并加上起始偏移
                )
            llm_positions = torch.cat(  # 拼接所有位置ID
                [item.float() for item in llm_pos_ids_list], dim=1  # 转为浮点型后拼接
            ).reshape(3, -1)  # 重塑形状
            position_ids[..., i, :] = llm_positions.to(position_ids.device)  # 将位置ID赋值到对应批次位置
            mrope_position_deltas.append(  # 记录位置增量
                llm_positions.max() + 1 - len(current_input_ids)  # 最大位置+1减去序列长度
            )
        mrope_position_deltas = torch.tensor(  # 将位置增量转为张量
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)  # 增加一个维度
        return position_ids, mrope_position_deltas  # 返回位置ID和位置增量
    else:  # 没有视觉信息的情况
        s = input_ids.shape[1]  # 获取序列长度
        position_ids = torch.arange(s)  # 生成0到s-1的位置序列
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)  # 扩展为三维并转移到对应设备
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[  # 计算最大位置ID
            0
        ]
        mrope_position_deltas = max_position_ids + 1 - s  # 计算位置增量
        return position_ids, mrope_position_deltas  # 返回位置ID和位置增量


def get_rope_index_glm4v(  # 获取GLM4V模型的MRoPE位置索引
    input_ids: torch.Tensor,  # 输入token ID张量
    hf_config: Any,  # HuggingFace模型配置
    image_grid_thw: Union[List[List[int]], torch.Tensor],  # 图像网格的时间-高度-宽度信息
    video_grid_thw: Union[List[List[int]], torch.Tensor],  # 视频网格的时间-高度-宽度信息
    attention_mask: torch.Tensor,  # 注意力掩码
    **kwargs,  # 其他关键字参数
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回位置ID和位置增量的元组
    """Get mrope input positions and delta value for GLM4V."""  # 获取GLM4V的mrope输入位置和增量值
    image_token_id = hf_config.image_token_id  # 获取图像token的ID
    video_start_token_id = hf_config.video_start_token_id  # 获取视频起始token的ID
    video_end_token_id = hf_config.video_end_token_id  # 获取视频结束token的ID
    spatial_merge_size = hf_config.vision_config.spatial_merge_size  # 获取空间合并大小

    mrope_position_deltas = []  # 存储每个样本的位置增量

    if input_ids is not None and (  # 如果输入ID存在且存在视觉信息
        image_grid_thw is not None or video_grid_thw is not None  # 图像或视频网格信息不为空
    ):
        total_input_ids = input_ids  # 保存原始输入ID用于后续遍历

        if attention_mask is None:  # 如果没有提供注意力掩码
            attention_mask = torch.ones_like(total_input_ids)  # 创建全1的注意力掩码

        position_ids = torch.ones(  # 初始化位置ID张量，全1
            3,  # 三个维度（时间、高度、宽度）
            input_ids.shape[0],  # 批次大小
            input_ids.shape[1],  # 序列长度
            dtype=input_ids.dtype,  # 数据类型与输入一致
            device=input_ids.device,  # 设备与输入一致
        )

        image_index, video_index = 0, 0  # 初始化图像和视频的索引计数器
        video_group_index = 0  # 视频帧组索引
        attention_mask = attention_mask.to(total_input_ids.device)  # 将注意力掩码转移到对应设备

        for i, ids in enumerate(total_input_ids):  # 遍历批次中的每个样本
            curr_mask = attention_mask[i]  # 获取当前样本的注意力掩码
            ids_masked = ids[curr_mask == 1]  # 根据掩码过滤有效的token

            input_tokens = ids_masked.tolist()  # 将过滤后的token转为Python列表
            input_token_type = [""] * len(input_tokens)  # 初始化token类型列表

            video_check_flg = False  # 视频检查标志，用于判断当前token是否在视频范围内
            for j, token in enumerate(input_tokens):  # 遍历所有token
                if token == video_start_token_id:  # 遇到视频起始token
                    video_check_flg = True  # 设置视频检查标志为True
                elif token == video_end_token_id:  # 遇到视频结束token
                    video_check_flg = False  # 设置视频检查标志为False

                if token == image_token_id and not video_check_flg:  # 图像token且不在视频范围内
                    input_token_type[j] = "image"  # 标记为图像类型
                elif token == image_token_id and video_check_flg:  # 图像token但在视频范围内（视频帧）
                    input_token_type[j] = "video"  # 标记为视频类型
                else:  # 其他token
                    input_token_type[j] = "text"  # 标记为文本类型

            input_type_group = []  # 存储连续相同类型的token分组
            for key, group in itertools.groupby(  # 按连续类型分组
                enumerate(input_token_type), lambda x: x[1]  # 按类型值分组
            ):
                group = list(group)  # 将迭代器转为列表
                start_index = group[0][0]  # 分组起始索引
                end_index = group[-1][0] + 1  # 分组结束索引（不包含）
                input_type_group.append((key, start_index, end_index))  # 添加分组信息

            llm_pos_ids_list = []  # 存储LLM位置ID的列表
            video_frame_num = 1  # 视频帧编号计数器

            for modality_type, start_idx, end_idx in input_type_group:  # 遍历每个模态分组
                if llm_pos_ids_list:  # 如果位置ID列表非空
                    st_idx = llm_pos_ids_list[-1].max().item() + 1  # 计算起始索引
                else:
                    st_idx = 0  # 第一个分组的起始索引为0

                if modality_type == "image":  # 图像类型
                    t, h, w = (  # 获取当前图像的网格维度
                        image_grid_thw[image_index][0],  # 时间维度
                        image_grid_thw[image_index][1],  # 高度维度
                        image_grid_thw[image_index][2],  # 宽度维度
                    )
                    t_int, h_int, w_int = int(t), int(h), int(w)  # 将维度转为整数
                    llm_grid_t = t_int  # LLM网格的时间维度
                    llm_grid_h = h_int // spatial_merge_size  # LLM网格的高度维度
                    llm_grid_w = w_int // spatial_merge_size  # LLM网格的宽度维度

                    t_index = (  # 生成时间维度的位置索引
                        torch.arange(llm_grid_t, device=position_ids.device)  # 生成0到llm_grid_t-1的序列
                        .view(-1, 1)  # 重塑为列向量
                        .expand(llm_grid_t, llm_grid_h * llm_grid_w)  # 扩展到所有空间位置
                        .reshape(-1)  # 展平为一维
                    )
                    h_index = (  # 生成高度维度的位置索引
                        torch.arange(llm_grid_h, device=position_ids.device)  # 生成0到llm_grid_h-1的序列
                        .view(1, -1, 1)  # 重塑为(1, grid_h, 1)的形状
                        .expand(llm_grid_t, llm_grid_h, llm_grid_w)  # 扩展到三维网格
                        .reshape(-1)  # 展平为一维
                    )
                    w_index = (  # 生成宽度维度的位置索引
                        torch.arange(llm_grid_w, device=position_ids.device)  # 生成0到llm_grid_w-1的序列
                        .view(1, 1, -1)  # 重塑为(1, 1, grid_w)的形状
                        .expand(llm_grid_t, llm_grid_h, llm_grid_w)  # 扩展到三维网格
                        .reshape(-1)  # 展平为一维
                    )
                    llm_pos_ids_list.append(  # 添加图像的位置ID
                        torch.stack([t_index, h_index, w_index]) + st_idx  # 堆叠三维索引并加上起始偏移
                    )
                    image_index += 1  # 图像索引加1
                    video_frame_num = 1  # 重置视频帧编号

                elif modality_type == "video":  # 视频类型
                    t = video_frame_num  # 视频的帧数
                    h = video_grid_thw[video_index][1]  # 获取视频的高度维度
                    w = video_grid_thw[video_index][2]  # 获取视频的宽度维度
                    h_int, w_int = int(h), int(w)  # 将维度转为整数
                    llm_grid_h = h_int // spatial_merge_size  # LLM网格的高度维度
                    llm_grid_w = w_int // spatial_merge_size  # LLM网格的宽度维度

                    for t_idx in range(t):  # 遍历视频的每一帧
                        t_index = (  # 生成当前帧的时间位置索引
                            torch.tensor(t_idx, device=position_ids.device)  # 帧索引作为时间位置
                            .view(-1, 1)  # 重塑为列向量
                            .expand(1, llm_grid_h * llm_grid_w)  # 扩展到所有空间位置
                            .reshape(-1)  # 展平为一维
                        )
                        h_index = (  # 生成高度维度的位置索引
                            torch.arange(llm_grid_h, device=position_ids.device)  # 生成0到llm_grid_h-1的序列
                            .view(1, -1, 1)  # 重塑为(1, grid_h, 1)的形状
                            .expand(1, llm_grid_h, llm_grid_w)  # 扩展到单帧三维网格
                            .reshape(-1)  # 展平为一维
                        )
                        w_index = (  # 生成宽度维度的位置索引
                            torch.arange(llm_grid_w, device=position_ids.device)  # 生成0到llm_grid_w-1的序列
                            .view(1, 1, -1)  # 重塑为(1, 1, grid_w)的形状
                            .expand(1, llm_grid_h, llm_grid_w)  # 扩展到单帧三维网格
                            .reshape(-1)  # 展平为一维
                        )
                        llm_pos_ids_list.append(  # 添加当前帧的位置ID
                            torch.stack([t_index, h_index, w_index]) + st_idx  # 堆叠三维索引并加上起始偏移
                        )

                    video_group_index += 1  # 视频帧组索引加1
                    if video_group_index >= video_grid_thw[video_index][0]:  # 如果已处理完当前视频的所有帧组
                        video_index += 1  # 视频索引加1
                        video_group_index = 0  # 重置帧组索引
                    video_frame_num += 1  # 视频帧编号加1

                else:  # text  # 文本类型
                    text_len = end_idx - start_idx  # 计算文本长度
                    text_range = torch.arange(text_len, device=position_ids.device)  # 生成文本位置序列
                    text_pos = text_range.view(1, -1).expand(3, text_len) + st_idx  # 扩展为三维并加上起始偏移
                    llm_pos_ids_list.append(text_pos)  # 添加文本位置ID到列表
                    video_frame_num = 1  # 重置视频帧编号

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)  # 拼接所有位置ID并重塑形状
            idx_mask = curr_mask == 1  # 获取有效位置的掩码
            position_ids[..., i, idx_mask] = llm_positions.to(position_ids.device)  # 将位置ID赋值到有效位置
            mrope_position_deltas.append(  # 记录位置增量
                llm_positions.max() + 1 - len(total_input_ids[i])  # 最大位置+1减去序列长度
            )
        mrope_position_deltas = torch.tensor(  # 将位置增量转为张量
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)  # 增加一个维度
        return position_ids, mrope_position_deltas  # 返回位置ID和位置增量
    else:  # 没有视觉信息的情况
        if attention_mask is not None:  # 如果提供了注意力掩码
            position_ids = attention_mask.long().cumsum(-1) - 1  # 基于掩码的累积和计算位置ID
            position_ids.masked_fill_(attention_mask == 0, 1)  # 将掩码为0的位置填充为1
            position_ids = (  # 扩展为三维位置ID
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)  # 扩展并转移到对应设备
            )
            max_position_ids = position_ids.amax(dim=0, keepdim=False)  # 在维度0上取最大值
            mrope_position_deltas = (  # 计算位置增量
                max_position_ids.amax(-1, keepdim=True) + 1 - attention_mask.shape[-1]  # 最大位置+1减去序列长度
            )
        else:  # 没有注意力掩码
            length = input_ids.shape[1]  # 获取序列长度
            batch_size = input_ids.shape[0]  # 获取批次大小
            arange_ids = torch.arange(length, device=input_ids.device).view(1, 1, -1)  # 生成位置序列
            position_ids = arange_ids.expand(3, batch_size, length)  # 扩展为三维
            mrope_position_deltas = torch.zeros(  # 位置增量为全0
                [batch_size, 1],  # 形状为(batch_size, 1)
                device=input_ids.device,  # 设备与输入一致
                dtype=input_ids.dtype,  # 数据类型与输入一致
            )
        return position_ids, mrope_position_deltas  # 返回位置ID和位置增量


def get_rope_index_ernie45(  # 获取Ernie4.5模型的MRoPE位置索引
    input_ids: torch.Tensor,  # 输入token ID张量
    hf_config: Any,  # HuggingFace模型配置
    image_grid_thw: Union[List[List[int]], torch.Tensor],  # 图像网格的时间-高度-宽度信息
    video_grid_thw: Union[List[List[int]], torch.Tensor],  # 视频网格的时间-高度-宽度信息
    **kwargs,  # 其他关键字参数
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回位置ID和位置增量的元组
    """Get mrope input positions and delta value for Ernie VL."""  # 获取Ernie VL的mrope输入位置和增量值
    image_token_id = hf_config.im_patch_id  # 获取图像补丁token的ID
    video_start_token_id = hf_config.video_start_token_id  # 获取视频起始token的ID
    video_end_token_id = hf_config.video_end_token_id  # 获取视频结束token的ID
    spatial_conv_size = hf_config.spatial_conv_size  # 获取空间卷积大小
    temporal_conv_size = hf_config.temporal_conv_size  # 获取时间卷积大小

    mrope_position_deltas = []  # 存储每个样本的位置增量
    if input_ids is not None and (  # 如果输入ID存在且存在视觉信息
        image_grid_thw is not None or video_grid_thw is not None  # 图像或视频网格信息不为空
    ):
        total_input_ids = input_ids  # 保存原始输入ID用于后续遍历
        position_ids = torch.ones(  # 初始化位置ID张量，全1
            3,  # 三个维度（时间、高度、宽度）
            input_ids.shape[0],  # 批次大小
            input_ids.shape[1],  # 序列长度
            dtype=input_ids.dtype,  # 数据类型与输入一致
            device=input_ids.device,  # 设备与输入一致
        )
        image_index, video_index = 0, 0  # 初始化图像和视频的索引计数器
        for i, input_ids in enumerate(total_input_ids):  # 遍历批次中的每个样本
            input_tokens = input_ids.tolist()  # 将输入token转为Python列表

            input_token_type = []  # 存储token类型列表
            video_check_flg = False  # 视频检查标志，用于判断当前token是否在视频范围内
            for token in input_tokens:  # 遍历所有token
                if token == video_start_token_id:  # 遇到视频起始token
                    video_check_flg = True  # 设置视频检查标志为True
                elif token == video_end_token_id:  # 遇到视频结束token
                    video_check_flg = False  # 设置视频检查标志为False

                if token == image_token_id and not video_check_flg:  # 图像token且不在视频范围内
                    input_token_type.append("image")  # 标记为图像类型
                elif token == image_token_id and video_check_flg:  # 图像token但在视频范围内（视频帧）
                    input_token_type.append("video")  # 标记为视频类型
                else:  # 其他token
                    input_token_type.append("text")  # 标记为文本类型

            input_type_group = []  # 存储连续相同类型的token分组
            for key, group in itertools.groupby(  # 按连续类型分组
                enumerate(input_token_type), lambda x: x[1]  # 按类型值分组
            ):
                group = list(group)  # 将迭代器转为列表
                start_index = group[0][0]  # 分组起始索引
                end_index = group[-1][0] + 1  # 分组结束索引（不包含）
                input_type_group.append((key, start_index, end_index))  # 添加分组信息

            llm_pos_ids_list = []  # 存储LLM位置ID的列表
            video_frame_num = 1  # 视频帧编号计数器
            for modality_type, start_idx, end_idx in input_type_group:  # 遍历每个模态分组
                st_idx = (  # 计算当前位置的起始索引
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0  # 如果列表非空则取最大值+1，否则为0
                )

                if modality_type == "image":  # 图像类型
                    t, h, w = (  # 获取当前图像的网格维度
                        image_grid_thw[image_index][0],  # 时间维度
                        image_grid_thw[image_index][1],  # 高度维度
                        image_grid_thw[image_index][2],  # 宽度维度
                    )
                    llm_grid_t, llm_grid_h, llm_grid_w = (  # 计算LLM网格的各维度
                        t.item(),  # 时间维度直接使用
                        h.item() // spatial_conv_size,  # 高度除以空间卷积大小
                        w.item() // spatial_conv_size,  # 宽度除以空间卷积大小
                    )

                    t_index = (  # 生成时间维度的位置索引
                        torch.arange(llm_grid_t)  # 生成0到llm_grid_t-1的序列
                        .view(-1, 1)  # 重塑为列向量
                        .expand(-1, llm_grid_h * llm_grid_w)  # 扩展到所有空间位置
                        .flatten()  # 展平为一维
                    )
                    h_index = (  # 生成高度维度的位置索引
                        torch.arange(llm_grid_h)  # 生成0到llm_grid_h-1的序列
                        .view(1, -1, 1)  # 重塑为(1, grid_h, 1)的形状
                        .expand(llm_grid_t, -1, llm_grid_w)  # 扩展到三维网格
                        .flatten()  # 展平为一维
                    )
                    w_index = (  # 生成宽度维度的位置索引
                        torch.arange(llm_grid_w)  # 生成0到llm_grid_w-1的序列
                        .view(1, 1, -1)  # 重塑为(1, 1, grid_w)的形状
                        .expand(llm_grid_t, llm_grid_h, -1)  # 扩展到三维网格
                        .flatten()  # 展平为一维
                    )
                    llm_pos_ids_list.append(  # 添加图像的位置ID
                        torch.stack([t_index, h_index, w_index]) + st_idx  # 堆叠三维索引并加上起始偏移
                    )
                    image_index += 1  # 图像索引加1
                    video_frame_num = 1  # 重置视频帧编号

                elif modality_type == "video":  # 视频类型
                    t, h, w = (  # 获取当前视频的网格维度
                        video_grid_thw[video_index][0],  # 时间维度
                        video_grid_thw[video_index][1],  # 高度维度
                        video_grid_thw[video_index][2],  # 宽度维度
                    )
                    llm_grid_t, llm_grid_h, llm_grid_w = (  # 计算LLM网格的各维度
                        t.item() // temporal_conv_size,  # 时间维度除以时间卷积大小
                        h.item() // spatial_conv_size,  # 高度除以空间卷积大小
                        w.item() // spatial_conv_size,  # 宽度除以空间卷积大小
                    )

                    for t_idx in range(llm_grid_t):  # 遍历视频的每一帧
                        t_index = (  # 生成当前帧的时间位置索引
                            torch.tensor(t_idx)  # 帧索引作为时间位置
                            .view(-1, 1)  # 重塑为列向量
                            .expand(-1, llm_grid_h * llm_grid_w)  # 扩展到所有空间位置
                            .flatten()  # 展平为一维
                        )
                        h_index = (  # 生成高度维度的位置索引
                            torch.arange(llm_grid_h)  # 生成0到llm_grid_h-1的序列
                            .view(1, -1, 1)  # 重塑为(1, grid_h, 1)的形状
                            .expand(1, -1, llm_grid_w)  # 扩展到单帧三维网格
                            .flatten()  # 展平为一维
                        )
                        w_index = (  # 生成宽度维度的位置索引
                            torch.arange(llm_grid_w)  # 生成0到llm_grid_w-1的序列
                            .view(1, 1, -1)  # 重塑为(1, 1, grid_w)的形状
                            .expand(1, llm_grid_h, -1)  # 扩展到单帧三维网格
                            .flatten()  # 展平为一维
                        )
                        llm_pos_ids_list.append(  # 添加当前帧的位置ID
                            torch.stack([t_index, h_index, w_index]) + st_idx  # 堆叠三维索引并加上起始偏移
                        )
                    video_index += 1  # 视频索引加1
                    video_frame_num += 1  # 视频帧编号加1

                else:  # 文本类型
                    text_len = end_idx - start_idx  # 计算文本长度
                    llm_pos_ids_list.append(  # 添加文本的位置ID
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx  # 生成文本位置并加上起始偏移
                    )
                    video_frame_num = 1  # 重置视频帧编号

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)  # 拼接所有位置ID并重塑形状
            position_ids[..., i, :] = llm_positions.to(position_ids.device)  # 将位置ID赋值到对应批次位置
            mrope_position_deltas.append(  # 记录位置增量
                llm_positions.max() + 1 - len(total_input_ids[i])  # 最大位置+1减去序列长度
            )
        mrope_position_deltas = torch.tensor(  # 将位置增量转为张量
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)  # 增加一个维度
        return position_ids, mrope_position_deltas  # 返回位置ID和位置增量
    else:  # 没有视觉信息的情况
        s = input_ids.shape[1]  # 获取序列长度
        position_ids = torch.arange(s)  # 生成0到s-1的位置序列
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)  # 扩展为三维并转移到对应设备
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[  # 计算最大位置ID
            0
        ]
        mrope_position_deltas = max_position_ids + 1 - s  # 计算位置增量
        return position_ids, mrope_position_deltas  # 返回位置ID和位置增量
