# 模型权重下载与初始化工具模块
# 提供模型权重的下载、格式转换、迭代加载、分片处理、量化配置获取等功能，支持safetensors/pt/gguf等多种权重格式

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/weight_utils.py
# 改编自 https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/weight_utils.py

"""Utilities for downloading and initializing model weights."""  # 下载和初始化模型权重的工具函数

import collections  # 导入集合模块
import concurrent.futures  # 导入并发未来模块
import fnmatch  # 导入文件名匹配模块
import glob  # 导入文件通配符匹配模块
import hashlib  # 导入哈希模块
import itertools  # 导入迭代工具模块
import json  # 导入JSON模块
import logging  # 导入日志模块
import os  # 导入操作系统模块
import re  # 导入正则表达式模块
import struct  # 导入结构体模块
import tempfile  # 导入临时文件模块
from collections import defaultdict  # 从collections导入默认字典
from pathlib import Path  # 从pathlib导入路径类
from typing import (  # 导入类型提示
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import filelock  # 导入文件锁模块
import huggingface_hub.constants  # 导入HuggingFace Hub常量
import numpy as np  # 导入NumPy
import safetensors.torch  # 导入safetensors的PyTorch支持
import torch  # 导入PyTorch
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download  # 从HuggingFace Hub导入下载函数
from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator  # 从pydantic导入数据验证工具
from tqdm.auto import tqdm  # 导入进度条

from sglang.srt.configs.load_config import LoadConfig  # 导入加载配置类
from sglang.srt.configs.model_config import ModelConfig  # 导入模型配置类
from sglang.srt.distributed import (  # 导入分布式通信函数
    get_tensor_model_parallel_rank,  # 获取张量模型并行秩
    get_tensor_model_parallel_world_size,  # 获取张量模型并行世界大小
    get_world_group,  # 获取世界通信组
)
from sglang.srt.layers.dp_attention import get_attention_tp_rank  # 导入注意力TP秩获取函数
from sglang.srt.layers.quantization import QuantizationConfig, get_quantization_config  # 导入量化配置
from sglang.srt.layers.quantization.fp8 import Fp8Config  # 导入FP8配置
from sglang.srt.layers.quantization.modelopt_quant import (  # 导入ModelOpt量化配置
    ModelOptFp4Config,  # ModelOpt FP4配置
    ModelOptFp8Config,  # ModelOpt FP8配置
)
from sglang.srt.model_loader.ci_weight_validation import (  # 导入CI权重验证函数
    ci_download_with_validation_and_retry,  # CI下载并验证重试
    ci_validate_and_cleanup_local_snapshot,  # CI验证并清理本地快照
)
from sglang.srt.utils import (  # 导入SGLang工具函数
    BAR_FORMAT,  # 进度条格式
    find_local_repo_dir,  # 查找本地仓库目录
    is_cpu,  # 判断是否为CPU
    log_info_on_rank0,  # 在秩0上记录信息
    print_warning_once,  # 打印一次警告
)
from sglang.srt.utils.common import is_cuda_alike  # 导入CUDA类判断函数
from sglang.utils import is_in_ci  # 导入CI环境判断函数

try:  # 尝试导入fastsafetensors
    from fastsafetensors import SafeTensorsFileLoader, SingleGroup  # 导入快速safetensors加载器
except ImportError as e:  # 导入失败
    SafeTensorsFileLoader = SingleGroup = None  # 设为None

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

RUNAI_STREAMER_TENSOR_ATTR = "_sglang_runai_streamer_tensor"  # RunAI流式传输器张量属性名

# Matches routed-expert weight keys in both HF-style layouts
# 匹配HF风格布局中的路由专家权重键
# (``...mlp.experts.<N>.{gate,up,down}_proj.weight``) and DeepSeek V4
# （``...mlp.experts.<N>.{gate,up,down}_proj.weight``）和DeepSeek V4
# layouts (``...ffn.experts.<N>.w{1,2,3}.weight``). ``shared_experts`` is
# 布局（``...ffn.experts.<N>.w{1,2,3}.weight``）。``shared_experts``被
# excluded because the index segment requires a digit after ``.experts.``.
# 排除，因为索引段要求``.experts.``后为数字。
_ROUTED_EXPERT_KEY_RE = re.compile(  # 路由专家权重键的正则表达式
    r"\.experts\.\d+\.(?:w[123]|down_proj|up_proj|gate_proj)\.weight$"  # 匹配专家权重键模式
)


def probe_routed_expert_weight_dtype(model_path: str) -> Optional[str]:  # 探测路由专家权重的数据类型
    """Return the safetensors dtype string (e.g. ``F8_E4M3``, ``U8``) of one
    """返回一个路由专家权重张量的safetensors dtype字符串（例如``F8_E4M3``、``U8``），
    routed-expert weight tensor, or ``None`` if the checkpoint is remote or has
    如果检查点为远程或没有匹配键则返回``None``。
    no matching key. Reads only the safetensors header of the relevant shard.
    仅读取相关分片的safetensors头部。
    """
    if not os.path.isdir(model_path):  # 如果模型路径不是目录
        return None  # 返回None（远程检查点）

    index_file = os.path.join(model_path, "model.safetensors.index.json")  # 索引文件路径
    target_key = None  # 目标权重键
    target_shard_path = None  # 目标分片路径

    if os.path.exists(index_file):  # 如果索引文件存在
        with open(index_file) as f:  # 打开索引文件
            index = json.load(f)  # 加载索引JSON
        weight_map = index.get("weight_map", {}) or {}  # 获取权重映射
        for k, shard in weight_map.items():  # 遍历权重映射
            if _ROUTED_EXPERT_KEY_RE.search(k):  # 如果键匹配路由专家模式
                target_key = k  # 设置目标键
                target_shard_path = os.path.join(model_path, shard)  # 设置目标分片路径
                break  # 找到第一个即退出
        if target_key is None:  # 如果没有找到匹配键
            return None  # 返回None
    else:  # 索引文件不存在
        shards = sorted(Path(model_path).glob("*.safetensors"))  # 查找所有safetensors文件
        if not shards:  # 如果没有分片文件
            return None  # 返回None
        target_shard_path = str(shards[0])  # 使用第一个分片文件

    with open(target_shard_path, "rb") as f:  # 以二进制模式打开目标分片
        (header_len,) = struct.unpack("<Q", f.read(8))  # 读取头部长度（8字节小端无符号长整型）
        header = json.loads(f.read(header_len))  # 读取并解析头部JSON

    if target_key is not None:  # 如果有目标键
        meta = header.get(target_key)  # 获取目标键的元数据
        return meta.get("dtype") if meta else None  # 返回dtype或None

    for k, meta in header.items():  # 遍历头部所有条目
        if k == "__metadata__" or not isinstance(meta, dict):  # 跳过元数据和非字典条目
            continue  # 跳过
        if _ROUTED_EXPERT_KEY_RE.search(k):  # 如果键匹配路由专家模式
            return meta.get("dtype")  # 返回dtype
    return None  # 没有找到匹配键，返回None


# Block size for sequential checkpoint prefetch reads (page cache warming).
# 顺序检查点预读的块大小（页面缓存预热）。
_PREFETCH_BLOCK_SIZE = None  # 预读块大小全局变量


def _get_prefetch_block_size() -> int:  # 获取预读块大小（字节）
    global _PREFETCH_BLOCK_SIZE  # 声明全局变量
    if _PREFETCH_BLOCK_SIZE is None:  # 如果未初始化
        from sglang.srt.environ import envs  # 导入环境变量

        _PREFETCH_BLOCK_SIZE = envs.SGLANG_PREFETCH_BLOCK_SIZE_MB.get() * 1024 * 1024  # 将MB转换为字节
    return _PREFETCH_BLOCK_SIZE  # 返回预读块大小


# use system-level temp directory for file locks, so that multiple users
# 使用系统级临时目录存放文件锁，以便多个用户
# can share the same lock without error.
# 可以无错误地共享同一锁。
# lock files in the temp directory will be automatically deleted when the
# 临时目录中的锁文件将在系统重启时自动删除，
# system reboots, so users will not complain about annoying lock files
# 因此用户不会抱怨烦人的锁文件
temp_dir = tempfile.gettempdir()  # 获取系统临时目录


def get_lock(  # 获取文件锁
    model_name_or_path: str, cache_dir: Optional[str] = None, suffix: str = ""  # 模型名称或路径，缓存目录，锁文件后缀
):
    lock_dir = cache_dir or temp_dir  # 锁文件目录为缓存目录或临时目录
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)  # 确保锁目录存在
    model_name = model_name_or_path.replace("/", "-")  # 替换路径分隔符
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()  # 计算模型名的SHA256哈希
    # add hash to avoid conflict with old users' lock files
    # 添加哈希以避免与旧用户的锁文件冲突
    lock_file_name = hash_name + model_name + suffix + ".lock"  # 构造锁文件名
    # mode 0o666 is required for the filelock to be shared across users
    # mode 0o666 是filelock跨用户共享所必需的
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)  # 创建文件锁
    return lock  # 返回文件锁


def _shared_pointers(tensors):  # 查找共享指针的张量（即多个名称指向同一数据）
    ptrs = defaultdict(list)  # 指针到名称列表的映射
    for k, v in tensors.items():  # 遍历张量字典
        ptrs[v.data_ptr()].append(k)  # 将名称添加到对应指针的列表
    failing = []  # 共享指针的名称组列表
    for _, names in ptrs.items():  # 遍历所有指针
        if len(names) > 1:  # 如果有多个名称共享同一指针
            failing.append(names)  # 添加到共享列表
    return failing  # 返回共享指针名称组


def convert_bin_to_safetensor_file(  # 将PyTorch bin文件转换为safetensor文件
    pt_filename: str,  # PyTorch文件名
    sf_filename: str,  # safetensor文件名
) -> None:
    loaded = torch.load(pt_filename, map_location="cpu", weights_only=True)  # 加载PyTorch权重文件
    if "state_dict" in loaded:  # 如果包含state_dict键
        loaded = loaded["state_dict"]  # 提取state_dict
    shared = _shared_pointers(loaded)  # 查找共享指针
    for shared_weights in shared:  # 遍历共享指针组
        for name in shared_weights[1:]:  # 跳过第一个，移除重复的
            loaded.pop(name)  # 移除重复名称的权重

    # For tensors to be contiguous
    # 使张量连续
    loaded = {k: v.contiguous() for k, v in loaded.items()}  # 将所有张量转为连续内存布局

    dirname = os.path.dirname(sf_filename)  # 获取输出文件目录
    os.makedirs(dirname, exist_ok=True)  # 确保目录存在

    from safetensors.torch import save_file  # 导入safetensors保存函数

    save_file(loaded, sf_filename, metadata={"format": "pt"})  # 保存为safetensor文件

    # check file size
    # 检查文件大小
    sf_size = os.stat(sf_filename).st_size  # 获取safetensor文件大小
    pt_size = os.stat(pt_filename).st_size  # 获取PyTorch文件大小
    if (sf_size - pt_size) / pt_size > 0.01:  # 如果大小差异超过1%
        raise RuntimeError(f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """)  # 文件大小差异超过1%时抛出运行时错误

    # check if the tensors are the same
    # 检查张量是否相同
    reloaded = safetensors.torch.load_file(sf_filename)  # 重新加载safetensor文件
    for k in loaded:  # 遍历原始权重
        pt_tensor = loaded[k]  # 原始张量
        sf_tensor = reloaded[k]  # 重新加载的张量
        if not torch.equal(pt_tensor, sf_tensor):  # 如果张量不相等
            raise RuntimeError(f"The output tensors do not match for key {k}")  # 抛出运行时错误


def replace_prefix(key: str, prefix_mapping: dict[str, str]) -> str:  # 替换键的前缀
    for prefix, new_prefix in prefix_mapping.items():  # 遍历前缀映射
        if key.startswith(prefix):  # 如果键以旧前缀开头
            key = key.replace(prefix, new_prefix, 1)  # 替换第一个匹配的前缀
    return key  # 返回替换后的键


def replace_substrings(key: str, substring_mapping: dict[str, str]) -> str:  # 替换键中的子字符串
    for substr, new_substr in substring_mapping.items():  # 遍历子字符串映射
        if substr in key:  # 如果键中包含旧子字符串
            key = key.replace(substr, new_substr)  # 替换所有匹配的子字符串
    return key  # 返回替换后的键


class DisabledTqdm(tqdm):  # 禁用进度条的tqdm子类
    def __init__(self, *args, **kwargs):  # 初始化方法
        kwargs["disable"] = True  # 强制禁用进度条
        super().__init__(*args, **kwargs)  # 调用父类初始化


# TODO(woosuk): Move this to other place.
# TODO(woosuk): 将此移到其他位置。
def get_quant_config(  # 获取量化配置
    model_config: ModelConfig,  # 模型配置
    load_config: LoadConfig,  # 加载配置
    packed_modules_mapping: Dict[str, List[str]],  # 打包模块映射
    remap_prefix: Dict[str, str] | None = None,  # 前缀重映射
) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)  # 获取量化配置类

    # GGUF doesn't have config file
    # GGUF没有配置文件
    if model_config.quantization == "gguf":  # 如果是GGUF量化
        return quant_cls.from_config({})  # 使用空配置创建量化配置

    # Read the quantization config from the HF model config, if available.
    # 如果可用，从HF模型配置读取量化配置。
    hf_quant_config = getattr(model_config.hf_config, "quantization_config", None)  # 获取HF量化配置
    # some vision model may keep quantization_config in their text_config
    # 一些视觉模型可能将quantization_config保存在text_config中
    hf_text_config = getattr(model_config.hf_config, "text_config", None)  # 获取HF文本配置
    if hf_quant_config is None and hf_text_config is not None:  # 如果没有量化配置但有文本配置
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)  # 从文本配置获取量化配置
    if hf_quant_config is None:  # 如果仍然没有量化配置
        # compressed-tensors uses a compressions_config
        # compressed-tensors使用compressions_config
        hf_quant_config = getattr(model_config.hf_config, "compression_config", None)  # 获取压缩配置
    if hf_quant_config is not None:  # 如果找到量化配置
        if not isinstance(hf_quant_config, dict):  # 如果不是字典类型
            hf_quant_config = hf_quant_config.to_dict()  # 转换为字典
        hf_quant_config["packed_modules_mapping"] = packed_modules_mapping  # 添加打包模块映射
        return quant_cls.from_config(hf_quant_config)  # 从配置创建量化配置

    # In case of bitsandbytes/QLoRA, get quant config from the adapter model.
    # 在bitsandbytes/QLoRA情况下，从适配器模型获取量化配置。
    if model_config.quantization == "bitsandbytes":  # 如果是bitsandbytes量化
        if (  # 如果没有额外配置或没有QLoRA适配器路径
            not load_config.model_loader_extra_config
            or "qlora_adapter_name_or_path" not in load_config.model_loader_extra_config
        ):
            return quant_cls.from_config({"adapter_name_or_path": ""})  # 使用空适配器路径创建配置
        model_name_or_path = load_config.model_loader_extra_config[  # 获取QLoRA适配器路径
            "qlora_adapter_name_or_path"
        ]
    else:  # 其他量化方式
        model_name_or_path = model_config.model_path  # 使用模型路径

    is_local = os.path.isdir(model_name_or_path)  # 检查是否为本地路径
    if not is_local:  # 如果不是本地路径
        # Download the config files.
        # 下载配置文件。
        with get_lock(model_name_or_path, load_config.download_dir):  # 获取文件锁
            hf_folder = snapshot_download(  # 下载模型快照
                model_name_or_path,  # 模型名称或路径
                revision=model_config.revision,  # 模型版本
                allow_patterns="*.json",  # 仅下载JSON文件
                cache_dir=load_config.download_dir,  # 缓存目录
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,  # 是否仅使用本地文件
                tqdm_class=DisabledTqdm,  # 禁用进度条
            )
    else:  # 本地路径
        hf_folder = model_name_or_path  # 直接使用本地路径

    possible_config_filenames = quant_cls.get_config_filenames()  # 获取可能的配置文件名

    # If the quantization config is not found, use the default config.
    # 如果未找到量化配置，使用默认配置。
    if not possible_config_filenames:  # 如果没有配置文件名
        if model_config.quantization == "mxfp8":  # 如果是MX-FP8量化
            return Fp8Config(use_mxfp8=True, is_checkpoint_fp8_serialized=False)  # 返回MX-FP8配置
        return quant_cls()  # 返回默认量化配置

    config_files = glob.glob(os.path.join(hf_folder, "*.json"))  # 查找所有JSON配置文件

    quant_config_files = [  # 筛选量化配置文件
        f for f in config_files if any(f.endswith(x) for x in possible_config_filenames)  # 匹配配置文件名
    ]
    if len(quant_config_files) == 0:  # 如果没有找到配置文件
        raise ValueError(f"Cannot find the config file for {model_config.quantization}")  # 抛出值错误
    if len(quant_config_files) > 1:  # 如果找到多个配置文件
        raise ValueError(  # 抛出值错误
            f"Found multiple config files for {model_config.quantization}: "
            f"{quant_config_files}"  # 多个配置文件的错误信息
        )

    quant_config_file = quant_config_files[0]  # 获取唯一的量化配置文件
    with open(quant_config_file) as f:  # 打开配置文件
        config = json.load(f)  # 加载配置JSON
        if remap_prefix is not None:  # 如果需要重映射前缀
            exclude_modules = [  # 重映射排除模块的前缀
                replace_prefix(key, remap_prefix)  # 替换前缀
                for key in config["quantization"]["exclude_modules"]  # 遍历排除模块
            ]
            config["quantization"]["exclude_modules"] = exclude_modules  # 更新排除模块列表
        config["packed_modules_mapping"] = packed_modules_mapping  # 添加打包模块映射

        if model_config.quantization == "bitsandbytes":  # 如果是bitsandbytes量化
            config["adapter_name_or_path"] = model_name_or_path  # 添加适配器路径
        elif model_config.quantization.startswith("modelopt") and (  # 如果是modelopt量化且
            config.get("producer", {}).get("name", "").startswith("modelopt")  # 生产者为modelopt
        ):
            quant_algo = config["quantization"]["quant_algo"]  # 获取量化算法
            if quant_algo is None:  # 如果量化算法为None
                # (yizhang2077) workaround for nvidia/Llama-4-Maverick-17B-128E-Eagle3
                # (yizhang2077) 针对nvidia/Llama-4-Maverick-17B-128E-Eagle3的临时解决方案
                if model_config.hf_config.architectures[0] != "LlamaForCausalLMEagle3":  # 如果不是Eagle3架构
                    raise ValueError(  # 抛出值错误
                        f"Invalid quant_config, quantization method: {model_config.quantization},"
                        f"hf architectures: {model_config.hf_config.architectures[0]}. "
                    )
                return None  # Eagle3架构返回None
            elif quant_algo == "FP8" or model_config.quantization == "modelopt_fp8":  # FP8量化
                return ModelOptFp8Config.from_config(config)  # 返回ModelOpt FP8配置
            elif "FP4" in quant_algo:  # FP4量化
                return ModelOptFp4Config.from_config(config)  # 返回ModelOpt FP4配置
        return quant_cls.from_config(config)  # 从配置创建量化配置


def _check_index_files_exist(snapshot_dir: str) -> Tuple[bool, Optional[str]]:  # 检查索引文件中列出的所有文件是否存在
    """
    Check if all files listed in safetensors index files actually exist on disk.
    检查safetensors索引文件中列出的所有文件是否实际存在于磁盘上。

    This catches cases where the snapshot directory exists but files are missing
    这捕获快照目录存在但文件缺失的情况
    (e.g., due to incomplete downloads or corrupted cache).
    （例如，由于不完整的下载或损坏的缓存）。

    Args:
    参数：
        snapshot_dir: Path to the model snapshot directory
        snapshot_dir: 模型快照目录的路径

    Returns:
    返回：
        Tuple of (all_exist, error_message)
        元组：(全部存在, 错误消息)
    """
    index_files = [  # 查找所有safetensors索引文件
        f for f in os.listdir(snapshot_dir) if f.endswith(".safetensors.index.json")  # 匹配索引文件名
    ]

    if not index_files:  # 如果没有索引文件
        return True, None  # Not a sharded model  # 不是分片模型

    for index_file in index_files:  # 遍历所有索引文件
        index_path = os.path.join(snapshot_dir, index_file)  # 索引文件路径
        if not os.path.exists(index_path):  # 如果索引文件不存在
            continue  # 跳过
        try:  # 尝试读取索引文件
            with open(index_path) as f:  # 打开索引文件
                weight_map = json.load(f).get("weight_map", {})  # 加载权重映射
            if not weight_map:  # 如果权重映射为空
                continue  # 跳过
            required_files = set(weight_map.values())  # 获取所需的文件集合
            missing_files = [  # 查找缺失的文件
                fn
                for fn in required_files
                if not os.path.exists(os.path.join(snapshot_dir, fn))  # 文件不存在
            ]
            if missing_files:  # 如果有缺失文件
                return (  # 返回缺失信息
                    False,  # 不是全部存在
                    f"Missing {len(missing_files)} file(s) from index {index_file}: "
                    f"{missing_files[:3]}{'...' if len(missing_files) > 3 else ''}",  # 缺失文件信息
                )
        except Exception as e:  # 捕获异常
            logger.warning("Failed to read index file %s: %s", index_file, e)  # 记录警告日志
            continue  # 跳过

    return True, None  # 全部存在，无错误


def _find_local_hf_snapshot_dir_unlocked(  # 在不加锁的情况下查找本地HF快照目录
    model_name_or_path: str,  # 模型名称或路径
    cache_dir: Optional[str],  # 缓存目录
    allow_patterns: List[str],  # 允许的文件模式列表
    revision: Optional[str] = None,  # 模型版本
) -> Optional[str]:
    """Find local HF snapshot directory without locking.
    """不加锁地查找本地HF快照目录。

    IMPORTANT: Caller MUST hold the model lock before calling this function
    重要：调用此函数前，调用者必须持有模型锁
    to prevent race conditions during validation and cleanup.
    以防止在验证和清理期间出现竞态条件。

    If the weights are already local, skip downloading and returns the path.
    如果权重已在本地，跳过下载并返回路径。
    """
    if os.path.isdir(model_name_or_path):  # 如果模型路径是本地目录
        return None  # 返回None（表示无需查找）

    found_local_snapshot_dir = None  # 找到的本地快照目录

    # Check custom cache_dir (if provided)
    # 检查自定义缓存目录（如果提供）
    if cache_dir:  # 如果有缓存目录
        try:  # 尝试查找
            repo_folder = os.path.join(  # 构造仓库文件夹路径
                cache_dir,  # 缓存目录
                huggingface_hub.constants.REPO_ID_SEPARATOR.join(  # 用分隔符连接
                    ["models", *model_name_or_path.split("/")]  # models/org/repo格式
                ),
            )
            rev_to_use = revision  # 要使用的版本
            if not rev_to_use:  # 如果没有指定版本
                ref_main = os.path.join(repo_folder, "refs", "main")  # main引用文件路径
                if os.path.isfile(ref_main):  # 如果main引用文件存在
                    with open(ref_main) as f:  # 打开引用文件
                        rev_to_use = f.read().strip()  # 读取版本号
            if rev_to_use:  # 如果有版本号
                rev_dir = os.path.join(repo_folder, "snapshots", rev_to_use)  # 快照目录路径
                if os.path.isdir(rev_dir):  # 如果快照目录存在
                    found_local_snapshot_dir = rev_dir  # 设置找到的快照目录
        except Exception as e:  # 捕获异常
            logger.warning(  # 记录警告日志
                "Failed to find local snapshot in custom cache_dir %s: %s",  # 在自定义缓存目录中查找本地快照失败
                cache_dir,
                e,
            )

    # Check default HF cache as well
    # 同时检查默认HF缓存
    if not found_local_snapshot_dir:  # 如果未在自定义缓存中找到
        try:  # 尝试在默认缓存中查找
            rev_dir = find_local_repo_dir(model_name_or_path, revision)  # 在默认缓存中查找
            if rev_dir and os.path.isdir(rev_dir):  # 如果找到且目录存在
                found_local_snapshot_dir = rev_dir  # 设置找到的快照目录
        except Exception as e:  # 捕获异常
            logger.warning("Failed to find local snapshot in default HF cache: %s", e)  # 记录警告日志

    # if local snapshot exists, validate it contains at least one weight file
    # 如果本地快照存在，验证其至少包含一个权重文件
    # matching allow_patterns before skipping download.
    # 匹配allow_patterns后再跳过下载。
    if found_local_snapshot_dir is None:  # 如果未找到本地快照
        return None  # 返回None

    # Check if snapshot dir exists (might have been cleaned by another process
    # 检查快照目录是否存在（可能已被其他进程清理
    # before we acquired the lock)
    # 在我们获取锁之前）
    if not os.path.isdir(found_local_snapshot_dir):  # 如果快照目录不存在
        return None  # 返回None

    local_weight_files: List[str] = []  # 本地权重文件列表
    try:  # 尝试扫描权重文件
        for pattern in allow_patterns:  # 遍历允许的文件模式
            matched_files = glob.glob(os.path.join(found_local_snapshot_dir, pattern))  # 匹配文件
            for f in matched_files:  # 遍历匹配的文件
                # os.path.exists returns False for broken symlinks.
                # os.path.exists对损坏的符号链接返回False。
                if not os.path.exists(f):  # 如果文件不存在（可能是损坏的符号链接）
                    continue  # 跳过
                local_weight_files.append(f)  # 添加到本地文件列表
    except Exception as e:  # 捕获异常
        logger.warning(  # 记录警告日志
            "Failed to scan local snapshot %s with patterns %s: %s",  # 扫描本地快照失败
            found_local_snapshot_dir,
            allow_patterns,
            e,
        )
        local_weight_files = []  # 重置本地文件列表

    # Check for missing files from index (lightweight, for all users)
    # 检查索引中的缺失文件（轻量级，适用于所有用户）
    # This catches incomplete downloads before they cause cryptic load errors
    # 这在不完整下载导致晦涩的加载错误之前捕获它们
    if local_weight_files:  # 如果有本地权重文件
        is_complete, error_msg = _check_index_files_exist(found_local_snapshot_dir)  # 检查文件完整性
        if not is_complete:  # 如果不完整
            log_info_on_rank0(  # 在秩0上记录信息
                logger,
                f"Local snapshot incomplete for {model_name_or_path}: {error_msg}. "
                f"Will download missing files.",  # 本地快照不完整，将下载缺失文件
            )
            return None  # Triggers snapshot_download() which handles partial downloads  # 触发snapshot_download()处理部分下载

    # Only perform cache validation and cleanup in CI to avoid
    # 仅在CI中执行缓存验证和清理，以避免
    # unnecessary overhead for regular users
    # 对普通用户造成不必要的开销
    if is_in_ci() and local_weight_files:  # 如果在CI环境且有本地文件
        is_valid = ci_validate_and_cleanup_local_snapshot(  # 验证并清理本地快照
            model_name_or_path, found_local_snapshot_dir, local_weight_files  # 传入参数
        )
        if not is_valid:  # 如果验证失败
            return None  # 返回None

    if len(local_weight_files) > 0:  # 如果有本地权重文件
        log_info_on_rank0(  # 在秩0上记录信息
            logger,
            f"Found local HF snapshot for {model_name_or_path} at "
            f"{found_local_snapshot_dir}; skipping download.",  # 找到本地HF快照，跳过下载
        )
        return found_local_snapshot_dir  # 返回快照目录
    else:  # 没有匹配的权重文件
        log_info_on_rank0(  # 在秩0上记录信息
            logger,
            f"Local HF snapshot at {found_local_snapshot_dir} has no files matching "
            f"{allow_patterns}; will attempt download.",  # 本地快照无匹配文件，尝试下载
        )
        return None  # 返回None


def download_weights_from_hf(  # 从HuggingFace Hub下载模型权重
    model_name_or_path: str,  # 模型名称或路径
    cache_dir: Optional[str],  # 缓存目录
    allow_patterns: List[str],  # 允许的文件模式列表
    revision: Optional[str] = None,  # 模型版本
    ignore_patterns: Optional[Union[str, List[str]]] = None,  # 忽略的文件模式
    max_retries: int = 3,  # 最大重试次数
) -> str:
    """Download model weights from Hugging Face Hub.
    """从HuggingFace Hub下载模型权重。

    Args:
    参数：
        model_name_or_path (str): The model name or path.  # 模型名称或路径
        cache_dir (Optional[str]): The cache directory to store the model
        cache_dir (Optional[str]): 存储模型权重的缓存目录
            weights. If None, will use HF defaults.
            如果为None，将使用HF默认值。
        allow_patterns (List[str]): The allowed patterns for the
        allow_patterns (List[str]): 权重文件的允许模式
            weight files. Files matched by any of the patterns will be
            匹配任一模式的文件将被
            downloaded.
            下载。
        revision (Optional[str]): The revision of the model.  # 模型版本
        ignore_patterns (Optional[Union[str, List[str]]]): The patterns to
        ignore_patterns (Optional[Union[str, List[str]]]): 过滤权重文件的模式
            filter out the weight files. Files matched by any of the patterns
            匹配任一模式的文件将被
            will be ignored.
            忽略。
        max_retries (int): Maximum number of download retries if corruption
        max_retries (int): 如果检测到损坏时的最大下载重试次数
            is detected. Defaults to 3.
            默认为3。

    Returns:
    返回：
        str: The path to the downloaded model weights.  # 下载的模型权重路径
    """
    # For local paths, no HF operations needed
    # 对于本地路径，不需要HF操作
    if os.path.isdir(model_name_or_path):  # 如果是本地目录
        return model_name_or_path  # 直接返回路径

    # Use a SINGLE lock for the entire operation (validation + cleanup + download)
    # 对整个操作（验证+清理+下载）使用单一锁
    # to prevent race conditions where:
    # 以防止以下竞态条件：
    # 1. Process A validates, finds corruption, deletes corrupted file
    # 1. 进程A验证，发现损坏，删除损坏文件
    # 2. Process B validates, sees missing file, deletes ENTIRE cache
    # 2. 进程B验证，看到缺失文件，删除整个缓存
    # 3. Process A tries to download but cache is gone
    # 3. 进程A尝试下载但缓存已消失
    # By using one lock, validation/cleanup and download are atomic.
    # 通过使用一个锁，验证/清理和下载是原子的。
    with get_lock(model_name_or_path, cache_dir):  # 获取文件锁
        # Check for valid local cache first (validates and cleans up if needed)
        # 首先检查有效的本地缓存（如需要则验证和清理）
        path = _find_local_hf_snapshot_dir_unlocked(  # 查找本地快照目录
            model_name_or_path, cache_dir, allow_patterns, revision  # 传入参数
        )
        if path is not None:  # 如果找到有效本地缓存
            # Valid local cache found, skip download
            # 找到有效的本地缓存，跳过下载
            return path  # 返回本地路径

        # In CI, skip HF API calls if we're in offline mode or want to avoid rate limits
        # 在CI中，如果处于离线模式或想避免速率限制，跳过HF API调用
        # But we already checked for local cache above, so if we're here we need to download
        # 但上面已检查了本地缓存，如果到这里则需要下载
        if not huggingface_hub.constants.HF_HUB_OFFLINE:  # 如果不是离线模式
            # Before we download we look at what is available:
            # 下载前先查看可用内容：
            fs = HfFileSystem()  # 创建HF文件系统
            file_list = fs.ls(model_name_or_path, detail=False, revision=revision)  # 列出远程文件

            # depending on what is available we download different things
            # 根据可用内容下载不同文件
            for pattern in allow_patterns:  # 遍历允许的文件模式
                matching = fnmatch.filter(file_list, pattern)  # 过滤匹配的文件
                if len(matching) > 0:  # 如果有匹配的文件
                    allow_patterns = [pattern]  # 使用匹配的模式
                    break  # 退出循环

        log_info_on_rank0(logger, f"Using model weights format {allow_patterns}")  # 记录使用的权重格式

        if not is_in_ci():  # 如果不在CI环境中
            # Simple download without validation for non-CI environments
            # 非CI环境的简单下载（不验证）
            hf_folder = snapshot_download(  # 下载模型快照
                model_name_or_path,  # 模型名称或路径
                allow_patterns=allow_patterns,  # 允许的文件模式
                ignore_patterns=ignore_patterns,  # 忽略的文件模式
                cache_dir=cache_dir,  # 缓存目录
                tqdm_class=DisabledTqdm,  # 禁用进度条
                revision=revision,  # 模型版本
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,  # 是否仅使用本地文件
            )
            return hf_folder  # 返回下载路径
        else:  # CI环境
            # Only perform validation and retry in CI to avoid overhead for regular users
            # 仅在CI中执行验证和重试，以避免普通用户的额外开销
            return ci_download_with_validation_and_retry(  # CI下载并验证重试
                model_name_or_path=model_name_or_path,  # 模型名称或路径
                allow_patterns=allow_patterns,  # 允许的文件模式
                ignore_patterns=ignore_patterns,  # 忽略的文件模式
                cache_dir=cache_dir,  # 缓存目录
                revision=revision,  # 模型版本
                max_retries=max_retries,  # 最大重试次数
            )


def download_safetensors_index_file_from_hf(  # 从HuggingFace Hub下载safetensors索引文件
    model_name_or_path: str,  # 模型名称或路径
    index_file: str,  # 索引文件名
    cache_dir: Optional[str],  # 缓存目录
    revision: Optional[str] = None,  # 模型版本
) -> None:
    """Download hf safetensors index file from Hugging Face Hub.
    """从HuggingFace Hub下载HF safetensors索引文件。

    Args:
    参数：
        model_name_or_path (str): The model name or path.  # 模型名称或路径
        cache_dir (Optional[str]): The cache directory to store the model
        cache_dir (Optional[str]): 存储模型权重的缓存目录
            weights. If None, will use HF defaults.
            如果为None，将使用HF默认值。
        revision (Optional[str]): The revision of the model.  # 模型版本
    """
    # Use file lock to prevent multiple processes from
    # 使用文件锁防止多个进程
    # downloading the same model weights at the same time.
    # 同时下载相同的模型权重。
    with get_lock(model_name_or_path, cache_dir):  # 获取文件锁
        try:  # 尝试下载
            # Download the safetensors index file.
            # 下载safetensors索引文件。
            hf_hub_download(  # 从HF Hub下载文件
                repo_id=model_name_or_path,  # 仓库ID
                filename=index_file,  # 文件名
                cache_dir=cache_dir,  # 缓存目录
                revision=revision,  # 模型版本
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,  # 是否仅使用本地文件
            )
        # If file not found on remote or locally, we should not fail since
        # 如果在远程或本地未找到文件，不应失败，因为
        # only some models will have index_file.
        # 只有部分模型会有索引文件。
        except huggingface_hub.utils.EntryNotFoundError:  # 远程条目未找到
            logger.debug("No %s found in remote.", index_file)  # 记录调试日志
        except huggingface_hub.utils.LocalEntryNotFoundError:  # 本地条目未找到
            logger.debug("No %s found in local cache.", index_file)  # 记录调试日志


# For models like Mistral-7B-v0.3, there are both sharded
# 对于像Mistral-7B-v0.3这样的模型，同时存在分片的
# safetensors files and a consolidated safetensors file.
# safetensors文件和合并的safetensors文件。
# Passing both of these to the weight loader functionality breaks.
# 将两者都传给权重加载功能会导致问题。
# So, we use the index_file to
# 因此，我们使用索引文件来
# look up which safetensors files should be used.
# 查找应使用哪些safetensors文件。
def filter_duplicate_safetensors_files(  # 过滤重复的safetensors文件
    hf_weights_files: List[str], hf_folder: str, index_file: str  # 权重文件列表，HF文件夹，索引文件名
) -> List[str]:
    # model.safetensors.index.json is a mapping from keys in the
    # model.safetensors.index.json是从
    # torch state_dict to safetensors file holding that weight.
    # torch state_dict中的键到持有该权重的safetensors文件的映射。
    index_file_name = os.path.join(hf_folder, index_file)  # 索引文件的完整路径
    if not os.path.isfile(index_file_name):  # 如果索引文件不存在
        # NOTE: this is a trick of handling mistral model
        # 注意：这是处理mistral模型的技巧
        # skip the unsupported consolidated.safetensors file
        # 跳过不支持的consolidated.safetensors文件
        if len(hf_weights_files) == 2:  # 如果只有2个权重文件
            hf_weights_files.sort()  # 排序
            if hf_weights_files[0].endswith(  # 如果第一个文件是
                "consolidated.safetensors"  # 合并的safetensors文件
            ) and hf_weights_files[1].endswith("model.safetensors"):  # 第二个是model.safetensors
                return [hf_weights_files[1]]  # 只返回model.safetensors
        return hf_weights_files  # 返回原始文件列表

    # Iterate through the weight_map (weight_name: safetensors files)
    # 遍历weight_map（权重名称: safetensors文件）
    # to identify weights that we should use.
    # 以识别应使用的权重。
    with open(index_file_name) as f:  # 打开索引文件
        weight_map = json.load(f)["weight_map"]  # 加载权重映射
    weight_files_in_index = set()  # 索引中引用的文件集合
    for weight_name in weight_map:  # 遍历权重映射
        weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))  # 添加文件路径到集合
    # Filter out any fields that are not found in the index file.
    # 过滤掉索引文件中未找到的任何字段。
    hf_weights_files = [f for f in hf_weights_files if f in weight_files_in_index]  # 仅保留索引中引用的文件
    return hf_weights_files  # 返回过滤后的文件列表


def maybe_add_mtp_safetensors(  # 可选地为GLM4Moe MTP/NextN模型添加mtp.safetensors
    hf_weights_files: List[str], hf_folder: str, index_file: str, hf_config  # 权重文件列表，HF文件夹，索引文件名，HF配置
) -> List[str]:
    """
    Auto-detect and add mtp.safetensors for GLM4Moe MTP/NextN models if:
    自动检测并为GLM4Moe MTP/NextN模型添加mtp.safetensors，如果：
    1. mtp.safetensors exists in the model directory
    1. mtp.safetensors存在于模型目录中
    2. mtp.safetensors is NOT in the index (checkpoint packaging bug)
    2. mtp.safetensors不在索引中（检查点打包bug）
    3. Model architecture is Glm4MoeForCausalLM with num_nextn_predict_layers > 0
    3. 模型架构为Glm4MoeForCausalLM且num_nextn_predict_layers > 0

    This works around incorrectly packaged FP4 checkpoints like
    这规避了错误打包的FP4检查点，如
    baseten-admin/glm-4.7-fp4 where mtp.safetensors exists but
    baseten-admin/glm-4.7-fp4中mtp.safetensors存在但
    isn't referenced in model.safetensors.index.json.
    未在model.safetensors.index.json中引用。
    """
    # Only apply for GLM4Moe architecture with nextn layers
    # 仅适用于具有nextn层的GLM4Moe架构
    arch = getattr(hf_config, "architectures", [None])[0]  # 获取架构名称
    num_nextn_layers = getattr(  # 获取nextn预测层数
        getattr(hf_config, "text_config", hf_config),  # 从text_config或hf_config获取
        "num_nextn_predict_layers",  # nextn预测层数属性
        getattr(hf_config, "num_nextn_predict_layers", 0),  # 默认值为0
    )
    if not (  # 如果不满足条件
        arch
        in [
            "Glm4MoeForCausalLM",  # GLM4MoE因果语言模型
            "Glm4MoeForCausalLMNextN",  # GLM4MoE因果语言模型NextN
            "Glm4MoeLiteForCausalLM",  # GLM4MoE精简版因果语言模型
            "Glm4MoeLiteForCausalLMNextN",  # GLM4MoE精简版因果语言模型NextN
        ]
        and num_nextn_layers > 0  # 且nextn层数大于0
    ):
        return hf_weights_files  # 返回原始文件列表

    # Check if mtp.safetensors exists and is not already in the file list
    # 检查mtp.safetensors是否存在且不在文件列表中
    mtp_path = os.path.join(hf_folder, "mtp.safetensors")  # mtp.safetensors路径
    if not os.path.isfile(mtp_path) or mtp_path in hf_weights_files:  # 如果不存在或已在列表中
        return hf_weights_files  # 返回原始文件列表

    # mtp.safetensors exists but not in index - this is a bug
    # mtp.safetensors存在但不在索引中 - 这是一个bug
    logger.warning(  # 记录警告日志
        f"Found mtp.safetensors but it's not referenced in {index_file}. "
        f"This is a checkpoint packaging bug. Auto-adding it for loading. "
        f"Please report this to the checkpoint provider."  # 发现mtp.safetensors但未在索引中引用，自动添加
    )

    # Add it to the files list
    # 将其添加到文件列表
    return hf_weights_files + [mtp_path]  # 返回添加了mtp.safetensors的文件列表


def filter_files_not_needed_for_inference(hf_weights_files: List[str]) -> List[str]:  # 过滤掉推理不需要的文件
    """
    Exclude files that are not needed for inference.
    排除推理不需要的文件。

    See https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
    参见 https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
    """
    blacklist = [  # 黑名单文件列表
        "training_args.bin",  # 训练参数
        "optimizer.bin",  # 优化器状态
        "optimizer.pt",  # 优化器状态（PT格式）
        "scheduler.pt",  # 学习率调度器
        "scaler.pt",  # 梯度缩放器
    ]
    hf_weights_files = [  # 过滤黑名单文件
        f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)  # 排除黑名单文件
    ]
    return hf_weights_files  # 返回过滤后的文件列表


def np_cache_weights_iterator(  # NumPy缓存权重迭代器
    model_name_or_path: str,  # 模型名称或路径
    cache_dir: Optional[str],  # 缓存目录
    hf_folder: str,  # HF文件夹路径
    hf_weights_files: List[str],  # HF权重文件列表
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model np files.
    """迭代模型numpy文件中的权重。

    Will dump the model weights to numpy files if they are not already dumped.
    如果模型权重尚未转储为numpy文件，则进行转储。
    """
    enable_tqdm = (  # 是否启用进度条
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0  # 仅在秩0启用
    )
    # Convert the model weights from torch tensors to numpy arrays for
    # 将模型权重从torch张量转换为numpy数组以
    # faster loading.
    # 实现更快的加载。
    np_folder = os.path.join(hf_folder, "np")  # NumPy缓存文件夹路径
    os.makedirs(np_folder, exist_ok=True)  # 确保目录存在
    weight_names_file = os.path.join(np_folder, "weight_names.json")  # 权重名称文件路径
    # Use file lock to prevent multiple processes from
    # 使用文件锁防止多个进程
    # dumping the same model weights to numpy at the same time.
    # 同时将相同的模型权重转储为numpy格式。
    with get_lock(model_name_or_path, cache_dir):  # 获取文件锁
        if not os.path.exists(weight_names_file):  # 如果权重名称文件不存在
            weight_names: List[str] = []  # 权重名称列表
            for bin_file in tqdm(  # 遍历权重文件（带进度条）
                hf_weights_files,  # 权重文件列表
                desc="Loading np_cache checkpoint shards",  # 进度条描述
                disable=not enable_tqdm,  # 是否禁用进度条
                bar_format=BAR_FORMAT,  # 进度条格式
                position=tqdm._get_free_pos(),  # 进度条位置
            ):
                state = torch.load(bin_file, map_location="cpu", weights_only=True)  # 加载权重到CPU
                for name, param in state.items():  # 遍历权重
                    param_path = os.path.join(np_folder, name)  # 参数文件路径
                    with open(param_path, "wb") as f:  # 打开文件写入
                        np.save(f, param.cpu().detach().numpy())  # 保存为numpy文件
                    weight_names.append(name)  # 添加权重名称
            with open(weight_names_file, "w") as f:  # 写入权重名称文件
                json.dump(weight_names, f)  # 保存为JSON

    with open(weight_names_file) as f:  # 读取权重名称文件
        weight_names = json.load(f)  # 加载权重名称列表

    for name in weight_names:  # 遍历权重名称
        param_path = os.path.join(np_folder, name)  # 参数文件路径
        with open(param_path, "rb") as f:  # 打开文件读取
            param = np.load(f)  # 加载numpy数组
        yield name, torch.from_numpy(param)  # 生成权重名称和PyTorch张量


def _prefetch_checkpoint_file(file_path: str) -> None:  # 预取检查点文件到OS页面缓存
    """Prefetch a checkpoint file into the OS page cache.
    """将检查点文件预取到OS页面缓存。

    Reads the file sequentially in 16 MB blocks so the kernel caches its pages
    以16 MB块顺序读取文件，以便内核缓存其页面
    before workers load the same file via mmap.
    在工作者通过mmap加载同一文件之前。
    """
    with open(file_path, "rb") as f:  # 以二进制模式打开文件
        while f.read(_get_prefetch_block_size()):  # 按块大小顺序读取
            pass  # 丢弃读取的数据（仅用于预热页面缓存）


def _prefetch_all_checkpoints(  # 在后台线程中开始预取检查点文件到页面缓存
    sorted_files: List[str],  # 排序后的检查点文件列表
    num_threads: int = 4,  # 预取线程数
) -> None:
    """Start prefetching checkpoint files into page cache in a background thread.
    """在后台线程中开始将检查点文件预取到页面缓存。

    When multiple ranks on the same node load the same checkpoint (e.g.
    当同一节点上的多个秩加载同一检查点时（例如
    DP-attention), each rank independently mmaps the same files, causing
    DP-attention），每个秩独立地mmap相同的文件，导致
    redundant NFS/Lustre reads. By distributing the prefetch across ranks
    冗余的NFS/Lustre读取。通过在秩间分配预取
    (each rank reads 1/Nth of the shards), the total network I/O is reduced
    （每个秩读取1/N的分片），总网络I/O从
    from N * checkpoint_size to 1 * checkpoint_size, with subsequent
    N * checkpoint_size减少到1 * checkpoint_size，随后的
    mmap accesses hitting the shared OS page cache.
    mmap访问命中共享的OS页面缓存。

    The prefetch runs in a background thread so that loading can start
    预取在后台线程中运行，以便加载可以立即开始
    immediately and benefit from pages that have already been cached,
    并从已缓存的页面中受益，
    rather than blocking until all files are prefetched. This pipelining
    而不是阻塞直到所有文件预取完成。这种流水线
    naturally adapts to any RAM size — even if the full checkpoint does
    自然适应任何RAM大小——即使完整的检查点
    not fit in page cache, the prefetch thread stays ahead of the loader.
    不适合页面缓存，预取线程也领先于加载器。
    """
    import asyncio  # 导入异步IO模块
    import threading  # 导入线程模块
    import time  # 导入时间模块

    # Use node-local rank so that each node independently prefetches the
    # 使用节点本地秩，以便每个节点独立预取
    # full checkpoint into its own page cache. Global rank would split files
    # 完整检查点到自己的页面缓存。全局秩会分割文件
    # across nodes, but page cache is not shared across nodes.
    # 跨节点，但页面缓存不在节点间共享。
    if torch.distributed.is_initialized():  # 如果分布式已初始化
        world_group = get_world_group()  # 获取世界通信组
        local_rank = world_group.local_rank  # 获取本地秩
        local_world_size = world_group.local_size or world_group.world_size  # 获取本地世界大小
    else:  # 分布式未初始化
        local_rank = 0  # 本地秩为0
        local_world_size = 1  # 本地世界大小为1

    my_files = sorted_files[local_rank::local_world_size]  # 当前秩负责预取的文件列表
    total_for_rank = len(my_files)  # 当前秩需预取的文件总数

    logger.info(  # 记录信息日志
        "Rank %d: prefetching %d/%d checkpoint shards into page cache "
        "(background, %d local ranks sharing the work, %d threads per rank)...",  # 秩%d：正在预取%d/%d检查点分片到页面缓存
        local_rank,  # 本地秩
        total_for_rank,  # 当前秩的文件数
        len(sorted_files),  # 总文件数
        local_world_size,  # 本地世界大小
        num_threads,  # 线程数
    )

    async def _prefetch_all() -> None:  # 异步预取所有文件
        semaphore = asyncio.Semaphore(num_threads)  # 信号量限制并发数
        completed = 0  # 已完成文件数
        next_log_pct = 10  # 下次日志输出的百分比

        async def prefetch_one(path: str) -> None:  # 异步预取单个文件
            nonlocal completed, next_log_pct  # 声明非局部变量
            try:  # 尝试预取
                async with semaphore:  # 获取信号量
                    await asyncio.to_thread(_prefetch_checkpoint_file, path)  # 在线程中预取文件
                completed += 1  # 增加完成计数
                if total_for_rank > 0 and next_log_pct <= 100:  # 如果需要输出进度
                    pct = 100 * completed / total_for_rank  # 计算完成百分比
                    if pct >= next_log_pct:  # 如果达到日志输出阈值
                        logger.info(  # 记录信息日志
                            "Rank %d: prefetching checkpoint files: %d%% (%d/%d)",  # 秩%d：预取检查点文件：%d%%
                            local_rank,  # 本地秩
                            next_log_pct,  # 进度百分比
                            completed,  # 已完成数
                            total_for_rank,  # 总数
                        )
                        next_log_pct += 10  # 下次日志阈值增加10%
            except Exception:  # 捕获异常
                logger.warning(  # 记录警告日志
                    "Failed to prefetch checkpoint file %r.",  # 预取检查点文件失败
                    path,  # 文件路径
                    exc_info=True,  # 包含异常信息
                )

        await asyncio.gather(*(prefetch_one(p) for p in my_files))  # 并发预取所有文件

    def _run_prefetch() -> None:  # 在线程中运行预取
        start = time.perf_counter()  # 记录开始时间
        asyncio.run(_prefetch_all())  # 运行异步预取
        elapsed = time.perf_counter() - start  # 计算耗时
        logger.info(  # 记录信息日志
            "Rank %d: prefetching checkpoint files into page cache "
            "finished in %.2fs",  # 秩%d：预取检查点文件到页面缓存完成，耗时%.2f秒
            local_rank,  # 本地秩
            elapsed,  # 耗时
        )

    threading.Thread(target=_run_prefetch, daemon=True).start()  # 启动后台守护线程运行预取


def _drop_file_cache_after_load(path: str) -> None:  # 加载权重后释放检查点页面缓存，避免RL中的CPU OOM
    """Release of checkpoint pages after weights have been copied out. Used to avoid CPU OOM in RL."""  # 权重拷出后释放检查点页面。用于避免RL中的CPU OOM。
    posix_fadvise = getattr(os, "posix_fadvise", None)  # 获取posix_fadvise函数
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)  # 获取POSIX_FADV_DONTNEED常量
    if posix_fadvise is None or dontneed is None:  # 如果不支持
        return  # 直接返回

    fd = None  # 文件描述符
    try:  # 尝试释放缓存
        fd = os.open(path, os.O_RDONLY)  # 以只读模式打开文件
        posix_fadvise(fd, 0, 0, dontneed)  # 告知内核不再需要该文件缓存
    except OSError as e:  # 捕获OS错误
        logger.debug("Failed to drop file cache for %s: %s", path, e)  # 记录调试日志
    finally:  # 最终
        if fd is not None:  # 如果文件描述符有效
            os.close(fd)  # 关闭文件


def safetensors_weights_iterator(  # safetensors权重文件迭代器
    hf_weights_files: List[str],  # 权重文件列表
    disable_mmap: bool = False,  # 是否禁用内存映射
    prefetch: bool = False,  # 是否预取
    prefetch_num_threads: int = 4,  # 预取线程数
    drop_cache_after_load: bool = False,  # 加载后是否释放缓存
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""  # 迭代模型safetensor文件中的权重
    enable_tqdm = (  # 是否启用进度条
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0  # 仅在秩0启用
    )

    sorted_files = sorted(hf_weights_files)  # 排序权重文件

    if prefetch and not disable_mmap:  # 如果启用预取且未禁用mmap
        _prefetch_all_checkpoints(sorted_files, num_threads=prefetch_num_threads)  # 预取所有检查点文件

    for st_file in tqdm(  # 遍历权重文件（带进度条）
        sorted_files,  # 排序后的文件列表
        desc="Loading safetensors checkpoint shards",  # 进度条描述
        disable=not enable_tqdm,  # 是否禁用进度条
        bar_format=BAR_FORMAT,  # 进度条格式
        position=tqdm._get_free_pos(),  # 进度条位置
    ):
        if disable_mmap:  # 如果禁用内存映射
            with open(st_file, "rb") as f:  # 以二进制模式打开文件
                result = safetensors.torch.load(f.read())  # 读取并加载全部内容
                for name in sorted(result.keys()):  # 按名称排序遍历
                    yield name, result[name]  # 生成权重名称和张量
        else:  # 使用内存映射
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:  # 以PT框架和CPU设备打开
                for name in f.keys():  # 遍历所有键
                    yield name, f.get_tensor(name)  # 生成权重名称和张量
        if drop_cache_after_load:  # 如果加载后释放缓存
            _drop_file_cache_after_load(st_file)  # 释放文件缓存


def fastsafetensors_weights_iterator(  # 使用fastsafetensor库的权重迭代器（支持GPU Direct Storage加速）
    hf_weights_files: List[str],  # 权重文件列表
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """
    Iterate over the weights in the model safetensor files
    迭代模型safetensor文件中的权重
    using fastsafetensor library to accelerate loading via GPU Direct Storage (if available).
    使用fastsafetensor库通过GPU Direct Storage加速加载（如果可用）。
    """
    if SafeTensorsFileLoader is None:  # 如果fastsafetensors未安装
        raise ImportError(  # 抛出导入错误
            "Please install fastsafetensors via `pip install fastsafetensors`"  # 提示安装
        )

    if torch.distributed.is_initialized():  # 如果分布式已初始化
        pg = torch.distributed.group.WORLD  # 使用世界进程组
    else:  # 分布式未初始化
        pg = SingleGroup()  # 使用单例进程组

    try:  # 尝试获取秩
        rank = pg.rank()  # 获取进程组中的秩
    except Exception:  # 捕获异常
        rank = 0  # 默认秩为0

    device = torch.device(f"cuda:{rank}")  # 根据秩设置CUDA设备

    weight_files_sub_lists = [  # 将文件列表按进程组大小分组
        hf_weights_files[i : i + pg.size()]  # 每组pg.size()个文件
        for i in range(0, len(hf_weights_files), pg.size())  # 按步长分组
    ]

    _BAR_FORMAT = (  # 自定义进度条格式
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"  # 进度条格式字符串
    )

    for f_list in tqdm(  # 遍历文件子列表（带进度条）
        weight_files_sub_lists,  # 文件子列表
        desc="Loading safetensors using Fastsafetensor loader",  # 进度条描述
        disable=False,  # 不禁用进度条
        bar_format=_BAR_FORMAT,  # 进度条格式
    ):
        loader = SafeTensorsFileLoader(pg, device)  # 创建文件加载器
        rank_file_map = {i: [f] for i, f in enumerate(f_list)}  # 秩到文件的映射
        loader.add_filenames(rank_file_map)  # 添加文件名映射
        try:  # 尝试加载
            fb = loader.copy_files_to_device()  # 将文件拷贝到设备
            try:  # 尝试遍历张量
                keys = list(fb.key_to_rank_lidx.keys())  # 获取所有键
                for k in keys:  # 遍历键
                    t = fb.get_tensor(k)  # 获取张量
                    yield k, t  # 生成键和张量
            finally:  # 最终
                pass  # 无操作
        finally:  # 最终
            loader.close()  # 关闭加载器


def multi_thread_safetensors_weights_iterator(  # 多线程safetensors权重迭代器
    hf_weights_files: List[str],  # 权重文件列表
    max_workers: int,  # 最大工作线程数
    disable_mmap: bool = False,  # 是否禁用内存映射
    drop_cache_after_load: bool = False,  # 加载后是否释放缓存
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Multi-Thread iterate over the weights in the model safetensor files."""  # 多线程迭代模型safetensor文件中的权重
    enable_tqdm = (  # 是否启用进度条
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0  # 仅在秩0启用
    )

    def _load_file(st_file: str):  # 加载单个safetensor文件
        if disable_mmap:  # 如果禁用内存映射
            with open(st_file, "rb") as f:  # 以二进制模式打开
                result = safetensors.torch.load(f.read())  # 读取并加载全部内容
        else:  # 使用内存映射
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:  # 以PT框架和CPU设备打开
                result = {k: f.get_tensor(k) for k in f.keys()}  # 加载所有张量到字典

        return st_file, result  # 返回文件名和结果字典

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  # 创建线程池
        futures = [executor.submit(_load_file, st_file) for st_file in hf_weights_files]  # 提交所有加载任务

        if enable_tqdm:  # 如果启用进度条
            futures_iter = tqdm(  # 带进度条的迭代器
                concurrent.futures.as_completed(futures),  # 按完成顺序迭代
                total=len(hf_weights_files),  # 总任务数
                desc="Multi-thread loading shards",  # 进度条描述
                disable=not enable_tqdm,  # 是否禁用进度条
                bar_format=BAR_FORMAT,  # 进度条格式
            )
        else:  # 不启用进度条
            futures_iter = concurrent.futures.as_completed(futures)  # 按完成顺序迭代

        for future in futures_iter:  # 遍历已完成的任务
            st_file, state_dict = future.result()  # 获取加载结果
            for name, param in state_dict.items():  # 遍历权重字典
                yield name, param  # 生成权重名称和参数
            del state_dict  # 删除权重字典释放内存
            if drop_cache_after_load:  # 如果加载后释放缓存
                _drop_file_cache_after_load(st_file)  # 释放文件缓存


def buffered_multi_thread_safetensors_weights_iterator(  # 带缓冲的多线程safetensors权重迭代器
    hf_weights_files: List[str],  # 权重文件列表
    max_workers: int,  # 最大工作线程数
    disable_mmap: bool = False,  # 是否禁用内存映射
    prefetch: bool = False,  # 是否预取
    prefetch_num_threads: int = 4,  # 预取线程数
    drop_cache_after_load: bool = False,  # 加载后是否释放缓存
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Multi-threaded safetensor loader with bounded memory via a sliding window.
    """通过滑动窗口实现有界内存的多线程safetensor加载器。

    At most (max_workers + 1) shard files are in-flight at any time:
    任意时刻最多有(max_workers + 1)个分片文件在传输中：
    max_workers loading concurrently + 1 prefetched and ready to yield.
    max_workers并发加载 + 1个已预取并准备产出。
    Peak CPU RAM ≈ (max_workers + 2) × shard_file_size.
    峰值CPU RAM ≈ (max_workers + 2) × 分片文件大小。
    """
    sorted_files = sorted(hf_weights_files)  # 排序权重文件
    if prefetch and not disable_mmap:  # 如果启用预取且未禁用mmap
        _prefetch_all_checkpoints(sorted_files, num_threads=prefetch_num_threads)  # 预取所有检查点文件
    enable_tqdm = (  # 是否启用进度条
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0  # 仅在秩0启用
    )

    def _load_file(st_file: str):  # 加载单个safetensor文件
        if disable_mmap:  # 如果禁用内存映射
            with open(st_file, "rb") as f:  # 以二进制模式打开
                result = safetensors.torch.load(f.read())  # 读取并加载全部内容
        else:  # 使用内存映射
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:  # 以PT框架和CPU设备打开
                result = {k: f.get_tensor(k) for k in f.keys()}  # 加载所有张量到字典
        return result  # 返回结果字典

    # Sliding window: max_workers loading + 1 prefetched.
    # 滑动窗口：max_workers加载中 + 1个已预取。
    buffer_size = max_workers + 1  # 缓冲区大小

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  # 创建线程池
        file_iter = iter(sorted_files)  # 创建文件迭代器
        pending: collections.deque = collections.deque()  # 待处理的Future队列

        # Seed the buffer.
        # 初始化缓冲区。
        for st_file in itertools.islice(file_iter, buffer_size):  # 取前buffer_size个文件
            pending.append(executor.submit(_load_file, st_file))  # 提交加载任务到队列

        with tqdm(  # 创建进度条
            total=len(hf_weights_files),  # 总文件数
            desc="Multi-thread loading shards",  # 进度条描述
            disable=not enable_tqdm,  # 是否禁用进度条
            bar_format=BAR_FORMAT,  # 进度条格式
            position=tqdm._get_free_pos(),  # 进度条位置
        ) as pbar:
            while pending:  # 当有待处理的任务
                future = pending.popleft()  # 从队列左侧取出任务
                state_dict = future.result()  # 获取加载结果
                del future  # let GC reclaim the Future's internal result  # 让GC回收Future的内部结果

                # Replenish: submit the next file to keep the buffer full.
                # 补充：提交下一个文件以保持缓冲区满。
                next_file = next(file_iter, None)  # 获取下一个文件
                if next_file is not None:  # 如果有下一个文件
                    pending.append(executor.submit(_load_file, next_file))  # 提交加载任务到队列

                for name in sorted(state_dict.keys()):  # 按名称排序遍历权重
                    yield name, state_dict[name]  # 生成权重名称和参数
                del state_dict  # 删除权重字典释放内存
                pbar.update(1)  # 更新进度条


def _load_pt_file(bin_file: str) -> dict:  # 加载PyTorch检查点文件
    """Load a PyTorch checkpoint file, handling legacy tar format.
    """加载PyTorch检查点文件，处理旧版tar格式。

    PyTorch 2.6 changed the default of weights_only from False to True.
    PyTorch 2.6将weights_only的默认值从False改为True。
    Legacy tar format files cannot be loaded with weights_only=True.
    旧版tar格式文件无法使用weights_only=True加载。
    This function tries weights_only=True first, then falls back to False
    此函数先尝试weights_only=True，然后回退到False
    for legacy tar format files from trusted sources (HuggingFace Hub).
    用于来自可信来源（HuggingFace Hub）的旧版tar格式文件。
    """
    try:  # 尝试以weights_only=True加载
        return torch.load(bin_file, map_location="cpu", weights_only=True)  # 安全加载
    except RuntimeError as e:  # 捕获运行时错误
        if "legacy .tar format" in str(e):  # 如果是旧版tar格式
            logger.warning(  # 记录警告日志
                "Loading %s with weights_only=False (legacy tar format)",  # 以weights_only=False加载（旧版tar格式）
                os.path.basename(bin_file),  # 文件名
            )
            return torch.load(bin_file, map_location="cpu", weights_only=False)  # 以不安全模式加载
        raise  # 重新抛出其他运行时错误


def pt_weights_iterator(  # PyTorch权重文件迭代器
    hf_weights_files: List[str],  # 权重文件列表
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model bin/pt files."""  # 迭代模型bin/pt文件中的权重
    enable_tqdm = (  # 是否启用进度条
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0  # 仅在秩0启用
    )
    for bin_file in tqdm(  # 遍历权重文件（带进度条）
        hf_weights_files,  # 权重文件列表
        desc="Loading pt checkpoint shards",  # 进度条描述
        disable=not enable_tqdm,  # 是否禁用进度条
        bar_format=BAR_FORMAT,  # 进度条格式
        position=tqdm._get_free_pos(),  # 进度条位置
    ):
        state = _load_pt_file(bin_file)  # 加载PT文件
        yield from state.items()  # 生成所有权重项
        del state  # 删除状态字典释放内存


def multi_thread_pt_weights_iterator(  # 多线程PyTorch权重文件迭代器
    hf_weights_files: List[str],  # 权重文件列表
    max_workers: int,  # 最大工作线程数
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Multi-Thread iterate over the weights in the model bin/pt files."""  # 多线程迭代模型bin/pt文件中的权重
    enable_tqdm = (  # 是否启用进度条
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0  # 仅在秩0启用
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  # 创建线程池
        futures = [  # 提交所有加载任务
            executor.submit(_load_pt_file, bin_file) for bin_file in hf_weights_files  # 加载每个PT文件
        ]

        if enable_tqdm:  # 如果启用进度条
            futures_iter = tqdm(  # 带进度条的迭代器
                concurrent.futures.as_completed(futures),  # 按完成顺序迭代
                total=len(hf_weights_files),  # 总任务数
                desc="Multi-thread loading pt checkpoint shards",  # 进度条描述
                disable=not enable_tqdm,  # 是否禁用进度条
                bar_format=BAR_FORMAT,  # 进度条格式
            )
        else:  # 不启用进度条
            futures_iter = concurrent.futures.as_completed(futures)  # 按完成顺序迭代

        for future in futures_iter:  # 遍历已完成的任务
            state = future.result()  # 获取加载结果
            yield from state.items()  # 生成所有权重项


def get_gguf_extra_tensor_names(  # 获取GGUF文件中额外张量的名称
    gguf_file: str, gguf_to_hf_name_map: Dict[str, str]  # GGUF文件路径，GGUF到HF名称映射
) -> List[str]:
    import gguf  # 导入gguf模块

    reader = gguf.GGUFReader(gguf_file)  # 创建GGUF读取器
    expected_gguf_keys = set(gguf_to_hf_name_map.keys())  # 预期的GGUF键集合
    exact_gguf_keys = set([tensor.name for tensor in reader.tensors])  # 实际的GGUF键集合
    extra_keys = expected_gguf_keys - exact_gguf_keys  # 多余的键（预期有但实际没有）
    return [gguf_to_hf_name_map[key] for key in extra_keys]  # 返回额外键对应的HF名称


def gguf_quant_weights_iterator(  # GGUF量化权重迭代器
    gguf_file: str, gguf_to_hf_name_map: Dict[str, str]  # GGUF文件路径，GGUF到HF名称映射
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """
    Iterate over the quant weights in the model gguf files and convert
    迭代模型gguf文件中的量化权重并转换
    them to torch tensors
    为PyTorch张量
    """

    import gguf  # 导入gguf模块

    reader = gguf.GGUFReader(gguf_file)  # 创建GGUF读取器

    # MoE expert weight name patterns
    # MoE专家权重名称模式
    MOE_WEIGHT_PATTERNS = {  # MoE权重模式映射
        "ffn_gate_exps": "gate_proj",  # gate projection  # gate投影
        "ffn_up_exps": "up_proj",  # up projection  # up投影
        "ffn_down_exps": "down_proj",  # down projection  # down投影
    }

    # First pass: yield weight types
    # 第一遍：产出权重类型
    for tensor in reader.tensors:  # 遍历所有张量
        weight_type = tensor.tensor_type  # 获取张量类型
        tensor_name = tensor.name  # 获取张量名称

        # Check if this is a MoE expert weight (packed format)
        # 检查是否为MoE专家权重（打包格式）
        is_moe_weight = any(  # 检查名称中是否包含MoE模式
            pattern in tensor_name for pattern in MOE_WEIGHT_PATTERNS.keys()  # 匹配模式
        )

        if is_moe_weight:  # 如果是MoE权重
            # MoE weights need special handling - extract layer_id and weight type
            # MoE权重需要特殊处理 - 提取层ID和权重类型
            # Format: blk.{layer_id}.ffn_gate_exps.weight
            # 格式：blk.{layer_id}.ffn_gate_exps.weight
            import re  # 导入正则表达式模块

            match = re.match(r"blk\.(\d+)\.(ffn_\w+_exps)\.weight", tensor_name)  # 匹配MoE权重名称
            if match:  # 如果匹配成功
                layer_id = int(match.group(1))  # 提取层ID
                weight_pattern = match.group(2)  # 提取权重模式
                hf_weight_name = MOE_WEIGHT_PATTERNS.get(weight_pattern)  # 获取HF权重名称

                if hf_weight_name and weight_type.name != "F32":  # 如果有HF名称且非F32类型
                    # Yield weight type for each expert
                    # 为每个专家产出权重类型
                    weight = tensor.data  # 获取权重数据
                    num_experts = weight.shape[0]  # 获取专家数量
                    for expert_id in range(num_experts):  # 遍历每个专家
                        hf_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{hf_weight_name}.qweight_type"  # 构造HF权重类型名称
                        yield hf_name, torch.tensor(weight_type)  # 产出权重类型
        elif tensor_name in gguf_to_hf_name_map:  # 普通权重
            # Normal weight handling
            # 普通权重处理
            name = gguf_to_hf_name_map[tensor_name]  # 获取HF名称

            if weight_type.name != "F32":  # 如果非F32类型
                weight_type_name = name.replace("weight", "qweight_type")  # 构造权重类型名称
                yield weight_type_name, torch.tensor(weight_type)  # 产出权重类型

    # Second pass: yield actual weights
    # 第二遍：产出实际权重
    for tensor in reader.tensors:  # 遍历所有张量
        weight = tensor.data  # 获取权重数据
        weight_type = tensor.tensor_type  # 获取张量类型
        tensor_name = tensor.name  # 获取张量名称

        # Check if this is a MoE expert weight (packed format)
        # 检查是否为MoE专家权重（打包格式）
        is_moe_weight = any(  # 检查名称中是否包含MoE模式
            pattern in tensor_name for pattern in MOE_WEIGHT_PATTERNS.keys()  # 匹配模式
        )

        if is_moe_weight:  # 如果是MoE权重
            # MoE weights: split packed format into individual expert weights
            # MoE权重：将打包格式拆分为单个专家权重
            import re  # 导入正则表达式模块

            match = re.match(r"blk\.(\d+)\.(ffn_\w+_exps)\.weight", tensor_name)  # 匹配MoE权重名称
            if match:  # 如果匹配成功
                layer_id = int(match.group(1))  # 提取层ID
                weight_pattern = match.group(2)  # 提取权重模式
                hf_weight_name = MOE_WEIGHT_PATTERNS.get(weight_pattern)  # 获取HF权重名称

                if hf_weight_name:  # 如果有HF名称
                    # Packed format: [num_experts, ...]
                    # 打包格式：[num_experts, ...]
                    num_experts = weight.shape[0]  # 获取专家数量
                    for expert_id in range(num_experts):  # 遍历每个专家
                        expert_weight = weight[expert_id]  # 获取单个专家权重

                        if weight_type.name != "F32":  # 如果非F32类型
                            hf_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{hf_weight_name}.qweight"  # 量化权重名称
                        else:  # F32类型
                            hf_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{hf_weight_name}.weight"  # 全精度权重名称

                        yield hf_name, torch.tensor(expert_weight)  # 产出权重
        elif tensor_name in gguf_to_hf_name_map:  # 普通权重
            # Normal weight handling
            # 普通权重处理
            name = gguf_to_hf_name_map[tensor_name]  # 获取HF名称

            if weight_type.name != "F32":  # 如果非F32类型
                name = name.replace("weight", "qweight")  # 替换为量化权重名称
            param = torch.tensor(weight)  # 转换为PyTorch张量
            yield name, param  # 产出权重


def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:  # 将PySafeSlice对象转换为PyTorch张量
    """convert PySafeSlice object from safetensors to torch.Tensor
    """将safetensors的PySafeSlice对象转换为torch.Tensor

    PySafeSlice object supports indexing, which is done before loading the
    PySafeSlice对象支持索引，这是在加载
    actual tensor and can reduce the amount of memory being read into the
    实际张量之前完成的，可以减少读入
    memory. However, it does not support more advanced functionalities
    内存的量。但是，它不支持更高级的功能
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    如`.view()`或`.t()`。因此，如果我们需要修改加载的
    tensor with these more complicated operators, we need to convert to
    张量使用这些更复杂的运算符，我们需要先转换为
    tensor first.
    张量。
    """
    if not isinstance(x, torch.Tensor):  # 如果不是PyTorch张量
        x = x[:]  # 切片加载全部数据
    return x  # 返回PyTorch张量


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:  # 默认权重加载器
    """Default weight loader."""  # 默认权重加载器
    try:  # 尝试加载
        if param.numel() == 1 and loaded_weight.numel() == 1:  # 如果参数和加载权重都是标量
            # Sometimes scalar values aren't considered tensors with shapes
            # 有时标量值不被视为具有形状的张量
            # so if both param and loaded_weight are a scalar,
            # 因此如果参数和加载权重都是标量，
            # "broadcast" instead of copy
            # 使用"广播"而不是拷贝
            param.data.fill_(loaded_weight.item())  # 用标量值填充参数
        else:  # 非标量情况
            assert param.size() == loaded_weight.size(), (  # 断言尺寸匹配
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"  # 尺寸不匹配的错误信息
            )

            param.data.copy_(loaded_weight)  # 拷贝加载权重到参数
    except Exception:  # 捕获异常
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # 注意：此异常是为了设置断点以
        # debug weight loading issues.
        # 调试权重加载问题而添加的。
        raise  # 重新抛出异常


def row_parallel_weight_loader(  # 行并行权重加载器
    param: torch.Tensor, loaded_weight: torch.Tensor  # 目标参数，加载的权重
) -> None:
    """Load weights that are row-parallelized."""  # 加载行并行化的权重
    tp_rank = get_tensor_model_parallel_rank()  # 获取张量模型并行秩
    shard_dim = 0 if param.dim() != 1 else None  # 分片维度：非1维时为0，1维时为None

    if shard_dim is not None:  # 如果有分片维度
        shard_size = param.data.shape[shard_dim]  # 获取分片大小
        start_idx = tp_rank * shard_size  # 计算起始索引
        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)  # 截取对应分片

    return default_weight_loader(param, loaded_weight)  # 使用默认加载器加载


LoaderFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # 加载器函数类型别名


def sharded_weight_loader(shard_axis: int) -> LoaderFunction:  # 创建沿指定轴分片的权重加载器
    """Create a weight loader that shards the weights along the given axis"""  # 创建沿给定轴分片权重的加载器

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:  # 分片加载器函数
        tp_rank = get_attention_tp_rank()  # 获取注意力TP秩

        shard_size = param.data.shape[shard_axis]  # 获取分片大小
        start_idx = tp_rank * shard_size  # 计算起始索引

        if (  # 如果满足CPU不等分条件
            is_cpu()  # 是CPU
            and (
                loaded_weight.size(0) % get_tensor_model_parallel_world_size() != 0  # 加载权重第0维不能被TP大小整除
                or loaded_weight.size(0)
                < get_tensor_model_parallel_world_size() * shard_size  # 加载权重第0维小于TP大小*分片大小
            )
            and loaded_weight.dim() == 1  # 且为1维张量
        ):
            param_data = param.data  # view copy on param for uneven padding  # 对参数进行视图拷贝用于不等分填充
            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(  # 截取填充参数和加载权重
                param_data,  # 参数数据
                loaded_weight,  # 加载权重
                0,  # param_data_start  # 参数数据起始位置
                start_idx,  # 权重起始索引
                shard_axis,  # 分片轴
                shard_size,  # 分片大小
            )
            return default_weight_loader(param_data, loaded_weight)  # 使用默认加载器加载
        else:  # 正常分片情况
            loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)  # 截取对应分片
            return default_weight_loader(param, loaded_weight)  # 使用默认加载器加载

    return loader  # 返回分片加载器函数


def composed_weight_loader(  # 创建组合权重加载器（加载后对权重进行后处理）
    loader: LoaderFunction, fn: Callable[[torch.Tensor], torch.Tensor]  # 原始加载器，后处理函数
) -> LoaderFunction:
    """Create a weight loader that post-processes the weights after loading"""  # 创建加载后对权重进行后处理的加载器

    def composed_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:  # 组合加载器函数
        loader(param, loaded_weight)  # 先使用原始加载器加载
        param.data.copy_(fn(param))  # 再对参数应用后处理函数
        return  # 返回

    return composed_loader  # 返回组合加载器函数


def runai_safetensors_weights_iterator(  # RunAI流式safetensors权重迭代器
    hf_weights_files: List[str], is_distributed: bool = False, device: str = "cpu"  # 权重文件列表，是否分布式，设备
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""  # 迭代模型safetensor文件中的权重
    from runai_model_streamer import SafetensorsStreamer  # 导入RunAI流式加载器

    enable_tqdm = (  # 是否启用进度条
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0  # 仅在秩0启用
    )
    device = device if is_distributed and is_cuda_alike() else "cpu"  # 分布式且CUDA类设备时使用指定设备，否则使用CPU

    with SafetensorsStreamer() as streamer:  # 创建流式加载器

        streamer.stream_files(  # 开始流式传输文件
            hf_weights_files,  # 权重文件列表
            device=device,  # 设备
            is_distributed=is_distributed,  # 是否分布式
        )
        total_tensors = sum(  # 计算总张量数
            len(tensors_meta)  # 每个文件中的张量数
            for tensors_meta in streamer.files_to_tensors_metadata.values()  # 遍历所有文件
        )

        tensor_iter = tqdm(  # 创建进度条
            streamer.get_tensors(),  # 获取张量迭代器
            total=total_tensors,  # 总张量数
            desc="Loading safetensors using Runai Model Streamer",  # 进度条描述
            bar_format=BAR_FORMAT,  # 进度条格式
            disable=not enable_tqdm,  # 是否禁用进度条
            mininterval=2,  # 最小更新间隔
        )

        for name, tensor in tensor_iter:  # 遍历所有张量
            setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)  # 标记为RunAI流式传输器张量
            yield name, tensor  # 生成张量名称和张量


def set_runai_streamer_env(load_config: LoadConfig):  # 设置RunAI流式传输器环境变量
    if load_config.model_loader_extra_config:  # 如果有额外配置
        extra_config = load_config.model_loader_extra_config  # 获取额外配置

        if "concurrency" in extra_config and isinstance(  # 如果配置中有concurrency且为整数
            extra_config.get("concurrency"), int
        ):
            os.environ["RUNAI_STREAMER_CONCURRENCY"] = str(  # 设置并发数环境变量
                extra_config.get("concurrency")  # 获取并发数
            )

        if "memory_limit" in extra_config and isinstance(  # 如果配置中有memory_limit且为整数
            extra_config.get("memory_limit"), int
        ):
            os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] = str(  # 设置内存限制环境变量
                extra_config.get("memory_limit")  # 获取内存限制
            )

    runai_streamer_s3_endpoint = os.getenv("RUNAI_STREAMER_S3_ENDPOINT")  # 获取RunAI S3端点
    aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")  # 获取AWS端点URL
    if runai_streamer_s3_endpoint is None and aws_endpoint_url is not None:  # 如果未设置RunAI S3端点但设置了AWS端点
        os.environ["RUNAI_STREAMER_S3_ENDPOINT"] = aws_endpoint_url  # 使用AWS端点作为RunAI S3端点


def initialize_dummy_weights(  # 用随机值初始化模型权重
    model: torch.nn.Module,  # PyTorch模型
    low: float = -1e-3,  # 随机值下界
    high: float = 1e-3,  # 随机值上界
    seed: int = 1234,  # 随机种子
) -> None:
    """Initialize model weights with random values.
    """用随机值初始化模型权重。

    The model weights must be randomly initialized for accurate performance
    模型权重必须随机初始化以获得准确的性能
    measurements. Additionally, the model weights should not cause NaNs in the
    测量。此外，模型权重不应在前向传播中导致NaN。
    forward pass. We empirically found that initializing the weights with
    我们经验性地发现，用
    values between -1e-3 and 1e-3 works well for most models.
    -1e-3到1e-3之间的值初始化权重对大多数模型效果良好。

    We use per-parameter random seed, so that dummy weights are consistent,
    我们使用每参数随机种子，以便虚拟权重一致，
    even if the model is partitioned across multiple devices. When the seed
    即使模型分布在多个设备上。当种子
    is fixed, the random values generated by this function only depends on
    固定时，此函数生成的随机值仅取决于
    the parameter's number of elements and its data type.
    参数的元素数量和数据类型。
    """
    for param in model.state_dict().values():  # 遍历模型所有参数
        if torch.is_floating_point(param):  # 如果参数为浮点类型
            generator = torch.Generator(device=param.data.device)  # 创建随机数生成器
            generator.manual_seed(seed)  # 设置随机种子
            if torch.finfo(param.data.dtype).bits < 16:  # 如果数据类型小于16位（FP8等）
                # uniform_ doesn't support < 16-bit datatypes (FP8)
                # uniform_不支持小于16位的数据类型（FP8）
                dtype = param.data.dtype  # 保存原始数据类型
                tmp_param = param.data.to(torch.float16)  # 转换为float16
                tmp_param = tmp_param.uniform_(low, high, generator=generator).to(dtype)  # 均匀分布初始化并转换回原始类型
                param.data.copy_(tmp_param)  # 拷贝回参数
            else:  # 16位及以上浮点类型
                param.uniform_(low, high, generator=generator)  # 直接均匀分布初始化


def maybe_remap_kv_scale_name(name: str, params_dict: dict) -> Optional[str]:  # 可能重映射KV缩放参数名称
    """Remap the name of FP8 k/v_scale parameters.
    """重映射FP8 k/v_scale参数的名称。

    This function handles the remapping of FP8 k/v_scale parameter names.
    此函数处理FP8 k/v_scale参数名称的重映射。
    It detects if the given name ends with a suffix and attempts to remap
    它检测给定名称是否以特定后缀结尾，并尝试重映射
    it to the expected name format in the model. If the remapped name is not
    为模型中预期的名称格式。如果重映射后的名称未
    found in the params_dict, a warning is printed and None is returned.
    在params_dict中找到，则打印警告并返回None。

    Args:
    参数：
        name (str): The original loaded checkpoint parameter name.  # 原始加载的检查点参数名称
        params_dict (dict): Dictionary containing the model's named parameters.  # 包含模型命名参数的字典

    Returns:
    返回：
        str: The remapped parameter name if successful, or the original name
        str: 成功时为重映射的参数名称，或无需重映射时为原始名称
             if no remapping is needed.
        None: If the remapped name is not found in params_dict.  # 如果重映射名称未在params_dict中找到
    """
    if name.endswith(".kv_scale"):  # 如果名称以.kv_scale结尾
        print_warning_once(  # 打印一次警告
            "DEPRECATED. Found kv_scale in the checkpoint. "
            "This format is deprecated in favor of separate k_scale and "
            "v_scale tensors and will be removed in a future release. "
            "Functionally, we will remap kv_scale to k_scale and duplicate "
            "k_scale to v_scale"  # 已废弃：发现kv_scale，将重映射为k_scale
        )
        # NOTE: we remap the deprecated kv_scale to k_scale
        # 注意：我们将废弃的kv_scale重映射为k_scale
        remapped_name = name.replace(".kv_scale", ".attn.k_scale")  # 重映射名称
        if remapped_name not in params_dict:  # 如果重映射名称不在参数字典中
            print_warning_once(  # 打印一次警告
                f"Found kv_scale in the checkpoint (e.g. {name}), "
                "but not found the expected name in the model "
                f"(e.g. {remapped_name}). kv_scale is "
                "not loaded."  # 未找到对应的k_scale，跳过加载
            )
            return None  # 返回None
        return remapped_name  # 返回重映射后的名称

    possible_scale_names = [".k_scale", ".v_scale"]  # 可能的缩放名称后缀
    # Patterns where modelopt stores scales under k_proj/v_proj
    # modelopt将缩放存储在k_proj/v_proj下的模式
    # but the model expects them under attn (RadixAttention)
    # 但模型期望它们在attn下（RadixAttention）
    modelopt_attn_prefixes = [".self_attn.", ".mixer."]  # modelopt注意力前缀
    for scale_name in possible_scale_names:  # 遍历可能的缩放名称
        if name.endswith(scale_name):  # 如果名称以缩放名称结尾
            # Check if this is a modelopt-style scale under k_proj/v_proj
            # 检查是否为k_proj/v_proj下的modelopt风格缩放
            matched_prefix = None  # 匹配的前缀
            for attn_prefix in modelopt_attn_prefixes:  # 遍历注意力前缀
                if f"{attn_prefix}{scale_name[1]}_proj{scale_name}" in name:  # 如果名称包含modelopt风格路径
                    matched_prefix = attn_prefix  # 设置匹配的前缀
                    break  # 退出循环

            if matched_prefix is not None:  # 如果匹配了modelopt前缀
                remapped_name = name.replace(  # 重映射名称
                    f"{matched_prefix}{scale_name[1]}_proj{scale_name}",  # 旧路径
                    f"{matched_prefix}attn{scale_name}",  # 新路径
                )
            else:  # 没有匹配modelopt前缀
                remapped_name = name.replace(scale_name, f".attn{scale_name}")  # 重映射到attn路径
            if remapped_name not in params_dict:  # 如果重映射名称不在参数字典中
                print_warning_once(  # 打印一次警告
                    f"Found {scale_name} in the checkpoint (e.g. {name}), "
                    "but not found the expected name in the model "
                    f"(e.g. {remapped_name}). {scale_name} is "
                    "not loaded."  # 未找到对应的缩放参数，跳过加载
                )
                return None  # 返回None
            return remapped_name  # 返回重映射后的名称

    quark_scale_names = {  # Quark缩放名称映射
        ".q_proj.output_scale": ".attn.q_scale",  # Q投影输出缩放 -> Q缩放
        ".k_proj.output_scale": ".attn.k_scale",  # K投影输出缩放 -> K缩放
        ".v_proj.output_scale": ".attn.v_scale",  # V投影输出缩放 -> V缩放
        "self_attn.prob_output_scale": ".attn.prob_scale",  # 注意力概率输出缩放 -> 概率缩放
    }
    for quark_scale_name, sglang_scale_name in quark_scale_names.items():  # 遍历Quark缩放映射
        if name.endswith(quark_scale_name):  # 如果名称以Quark缩放名结尾
            return name.replace(quark_scale_name, sglang_scale_name)  # 替换为SGLang缩放名

    # If there were no matches, return the untouched param name
    # 如果没有匹配项，返回未修改的参数名称
    return name  # 返回原始名称


# Adapted from https://github.com/vllm-project/vllm/blob/68ad4e3a8d8a66fb2a43be57471ee13a8bec4ec0/vllm/model_executor/layers/quantization/schema.py
# 改编自 https://github.com/vllm-project/vllm/blob/68ad4e3a8d8a66fb2a43be57471ee13a8bec4ec0/vllm/model_executor/layers/quantization/schema.py
class KVCacheQuantSchema(BaseModel):  # KV缓存量化模式类
    dtype: str  # 数据类型
    # Each key is a TP rank. Each value is a dictionary mapping a TP rank's
    # 每个键是一个TP秩。每个值是一个字典，映射TP秩的
    # layer indices to their per-tensor KV cache scaling factor.
    # 层索引到其逐张量KV缓存缩放因子。
    # TODO: Consider pulling this and its validation methods out into its
    # TODO: 考虑将其及其验证方法提取到其
    # own schema class (tricky as its members are variable)
    # 自己的schema类中（因为其成员是可变的，比较棘手）
    scaling_factor: Dict[int, Dict[int, float]]  # 缩放因子字典

    @model_validator(mode="after")  # 模型验证器（在初始化后执行）
    def check_is_fp8(self) -> "KVCacheQuantSchema":  # 检查是否为FP8数据类型
        assert self.dtype == "float8_e4m3fn", (  # 断言数据类型为float8_e4m3fn
            "Loaded scaling factors intended for KV cache dtype = "
            f"{self.dtype} rather than float8_e4m3fn!"  # 加载的缩放因子对应于非FP8的KV缓存数据类型
        )
        return self  # 返回自身

    @model_validator(mode="after")  # 模型验证器（在初始化后执行）
    def check_tp_ranks(self, info: ValidationInfo) -> "KVCacheQuantSchema":  # 检查TP秩是否匹配
        context = info.context  # 获取验证上下文
        if context:  # 如果有上下文
            tp_size = context["tp_size"]  # 获取TP大小
            num_hidden_layers = context["num_hidden_layers"]  # 获取隐藏层数量
            assert len(self.scaling_factor) == tp_size, (  # 断言缩放因子字典大小等于TP大小
                f"Loaded dictionary has TP size {len(self.scaling_factor)} "
                f"but LLM engine is currently running with TP size {tp_size}."  # 加载的TP大小与当前运行的TP大小不匹配
            )
            for tp_rank, layer_maps in self.scaling_factor.items():  # 遍历每个TP秩
                assert len(layer_maps) == num_hidden_layers, (  # 断言层映射数量等于隐藏层数量
                    f"KV cache scales map for TP rank {tp_rank} is malformed. "
                    f"Expected {num_hidden_layers} layers, got "
                    f"{len(layer_maps)}."  # KV缓存缩放映射格式不正确
                )
            for i in range(tp_size):  # 遍历所有TP秩
                assert (
                    i in self.scaling_factor
                ), f"KV cache scales map for TP rank {i} not found."  # 未找到TP秩的KV缓存缩放映射
        return self  # 返回自身

    @model_validator(mode="after")  # 模型验证器（在初始化后执行）
    def check_current_rank(self, info: ValidationInfo) -> "KVCacheQuantSchema":  # 检查当前秩的缩放因子是否完整
        context = info.context  # 获取验证上下文
        if context:  # 如果有上下文
            tp_rank = context["tp_rank"]  # 获取当前TP秩
            num_hidden_layers = context["num_hidden_layers"]  # 获取隐藏层数量
            layer_scales_map = self.scaling_factor[tp_rank]  # 获取当前秩的层缩放映射
            for i in range(num_hidden_layers):  # 遍历所有隐藏层
                assert i in layer_scales_map, (  # 断言每层都有缩放因子
                    f"Could not find KV cache scales for layer {i} in "
                    f"TP rank {tp_rank}."  # 在TP秩中未找到某层的KV缓存缩放因子
                )
        return self  # 返回自身


class QuantParamSchema(BaseModel):  # 量化参数模式类
    # TODO: Generalize and extend with more fields
    # TODO: 泛化并扩展更多字段
    # (e.g. weights/activations params) once functionality is enabled
    # （例如权重/激活参数）一旦功能启用
    model_config = ConfigDict(protected_namespaces=())  # 模型配置，允许受保护命名空间
    model_type: Optional[str]  # 模型类型
    kv_cache: KVCacheQuantSchema  # KV缓存量化模式

    @model_validator(mode="after")  # 模型验证器（在初始化后执行）
    def check_model_type(self, info: ValidationInfo) -> "QuantParamSchema":  # 检查模型类型是否匹配
        context = info.context  # 获取验证上下文
        if context:  # 如果有上下文
            model_type = context.get("model_type", None)  # 获取预期的模型类型
            if model_type is not None:  # 如果有预期模型类型
                assert model_type == self.model_type, (  # 断言模型类型匹配
                    f"Model type is {model_type} but loaded "
                    f"scaling factors belonging to different "
                    f"model type {self.model_type}!"  # 模型类型与加载的缩放因子所属的模型类型不同
                )
        return self  # 返回自身


def kv_cache_scales_loader(  # KV缓存缩放因子加载器
    filename: str,  # 缩放因子文件路径
    tp_rank: int,  # 张量并行秩
    tp_size: int,  # 张量并行大小
    num_hidden_layers: int,  # 隐藏层数量
    model_type: Optional[str],  # 模型类型
) -> Iterable[Tuple[int, float]]:
    """
    A simple utility to read in KV cache scaling factors that have been
    一个简单的工具，用于读取已
    previously serialized to disk. Used by the model to populate the appropriate
    序列化到磁盘的KV缓存缩放因子。被模型用来填充相应的
    KV cache scaling factors. The serialization should represent a dictionary
    KV缓存缩放因子。序列化应表示一个字典，
    whose keys are the TP ranks and values are another dictionary mapping layers
    其键为TP秩，值为另一个映射层
    to their KV cache scaling factors.
    到其KV缓存缩放因子的字典。
    """
    try:  # 尝试加载缩放因子
        with open(filename) as f:  # 打开文件
            context = {  # 验证上下文
                "model_type": model_type,  # 模型类型
                "num_hidden_layers": num_hidden_layers,  # 隐藏层数量
                "tp_rank": tp_rank,  # TP秩
                "tp_size": tp_size,  # TP大小
            }
            schema_dct = json.load(f)  # 加载JSON数据
            schema = QuantParamSchema.model_validate(schema_dct, context=context)  # 验证并创建模式
            layer_scales_map = schema.kv_cache.scaling_factor[tp_rank]  # 获取当前秩的层缩放映射
            return layer_scales_map.items()  # 返回层缩放映射项
    except FileNotFoundError:  # 文件未找到
        logger.error("File or directory '%s' not found.", filename)  # 记录错误日志
    except json.JSONDecodeError:  # JSON解码错误
        logger.error("Error decoding JSON in file '%s'.", filename)  # 记录错误日志
    except Exception:  # 其他异常
        logger.error("An error occurred while reading '%s'.", filename)  # 记录错误日志
    # This section is reached if and only if any of the excepts are hit
    # 此部分仅在任一except被触发时到达
    # Return an empty iterable (list) => no KV cache scales are loaded
    # 返回空可迭代对象（列表）=> 不加载KV缓存缩放因子
    # which ultimately defaults to 1.0 scales
    # 最终默认使用1.0缩放
    logger.warning(  # 记录警告日志
        "Defaulting to KV cache scaling factors = 1.0 for all "
        "layers in TP rank %d as an error occurred during loading.",  # 加载出错，默认KV缓存缩放因子为1.0
        tp_rank,  # TP秩
    )
    return []  # 返回空列表


def get_actual_shard_size(shard_size, weight_start, weight_end):  # 获取实际的分片大小
    if weight_end < weight_start:  # 如果权重结束位置小于起始位置
        return 0  # 返回0

    return min(shard_size, weight_end - weight_start)  # 返回分片大小和权重范围的最小值


def reset_param_data_if_needed(param_data, dim, start, length):  # 如果需要则重置参数数据为零
    if length == 0:  # 如果长度为0
        return  # 无需重置

    assert length > 0, f"Length should be positive, but got {length}"  # 断言长度为正数

    param_data.narrow(dim, start, length).zero_()  # 将指定范围的参数数据置零
    return  # 返回


def narrow_padded_param_and_loaded_weight(  # 截取填充参数和加载权重
    param_data,  # 参数数据
    loaded_weight,  # 加载的权重
    param_data_start,  # 参数数据起始位置
    weight_start,  # 权重起始位置
    dim,  # 维度
    shard_size,  # 分片大小
    narrow_weight=True,  # 是否截取权重
):
    actual_shard_size = get_actual_shard_size(  # 获取实际分片大小
        shard_size, weight_start, loaded_weight.size(dim)  # 分片大小，权重起始位置，权重在该维的大小
    )

    if narrow_weight:  # 如果需要截取权重
        if actual_shard_size > 0:  # 如果实际分片大小大于0
            loaded_weight = loaded_weight.narrow(dim, weight_start, actual_shard_size)  # 截取权重
        else:  # 实际分片大小为0
            # No real data to load; create a dummy tensor filled with zeros
            # 无真实数据可加载；创建一个用零填充的虚拟张量
            loaded_weight = torch.zeros_like(  # 创建零张量
                param_data.narrow(dim, param_data_start, actual_shard_size)  # 按参数数据范围创建
            )

    # [Note] Reset padded weights to zero.
    # [注意] 将填充的权重重置为零。
    # If the actual shard size is less than the shard size, we need to reset
    # 如果实际分片大小小于分片大小，我们需要重置
    # the padded param_data to zero and then copy the loaded_weight into it.
    # 填充的param_data为零，然后将loaded_weight拷贝进去。
    reset_param_data_if_needed(  # 重置填充部分
        param_data,  # 参数数据
        dim,  # 维度
        param_data_start + actual_shard_size,  # 填充起始位置
        shard_size - actual_shard_size,  # 填充长度
    )

    param_data = param_data.narrow(dim, param_data_start, actual_shard_size)  # 截取参数数据

    return param_data, loaded_weight  # 返回截取后的参数数据和加载权重


def pad_loaded_weight(loaded_weight, output_dim, output_sizes):  # 对加载权重进行零填充
    # This function is for padding zeros when loaded_weight is less than output_sizes.
    # 此函数用于在加载权重小于输出大小时填充零。
    # Most cases, sum(output_sizes) = loaded_weight.size(output_dim),
    # 大多数情况下，sum(output_sizes) = loaded_weight.size(output_dim)，
    # while in some TP cases like TP6, output_sizes will be padded, thus loaded_weight needs padding.
    # 但在某些TP情况下如TP6，output_sizes会被填充，因此loaded_weight需要填充。
    total_output_size = sum(output_sizes)  # 计算总输出大小
    raw_output_size = loaded_weight.size(output_dim)  # 获取加载权重在输出维度的大小
    if total_output_size > raw_output_size:  # 如果总输出大小大于原始大小
        loaded_weight_pad = []  # 填充后的权重列表
        weight_split_size = [  # 计算权重分割大小
            int(output_size / total_output_size * raw_output_size)  # 按比例计算每个输出的大小
            for output_size in output_sizes  # 遍历每个输出大小
        ]
        assert (  # 断言分割大小之和等于原始大小
            sum(weight_split_size) == raw_output_size
        ), f"Padding the loaded weight failed due to sizes are not divisible cleanly from {output_sizes} to {raw_output_size}"  # 填充权重失败：大小不能干净整除

        split_weight = loaded_weight.split_with_sizes(weight_split_size, dim=output_dim)  # 按分割大小拆分权重
        for i, output_size in enumerate(output_sizes):  # 遍历每个输出大小
            pad_size = output_size - weight_split_size[i]  # 计算填充大小
            target_pad_shape = list(loaded_weight.size())  # 目标填充形状
            target_pad_shape[output_dim] = pad_size  # 设置填充维度的大小
            pad_tensor = torch.zeros(target_pad_shape).to(loaded_weight.dtype)  # 创建零填充张量
            loaded_weight_pad.append(  # 添加填充后的权重
                torch.cat([split_weight[i], pad_tensor], dim=output_dim)  # 拼接原始权重和填充张量
            )
        return torch.cat(loaded_weight_pad, dim=output_dim)  # 返回拼接后的填充权重
    else:  # 不需要填充
        return loaded_weight  # 返回原始加载权重
