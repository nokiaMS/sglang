# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/model_executor/model_loader/loader.py

# 本文件实现了 SGLang 的模型权重加载器框架。
# 提供了多种模型加载器（如默认加载器、分片状态加载器、BitsAndBytes 加载器、GGUF 加载器、
# 远程加载器等），用于从不同来源和格式加载模型权重，并支持量化配置的处理。
# 核心类包括 BaseModelLoader（基类）、DefaultModelLoader（默认磁盘加载器）、
# ShardedStateLoader（分片状态加载器）、BitsAndBytesModelLoader（BNB 量化加载器）、
# GGUFModelLoader（GGUF 格式加载器）、RemoteInstanceModelLoader（远程实例加载器）、
# RemoteModelLoader（远程存储加载器）、ModelOptModelLoader（ModelOpt 量化加载器）、
# RunaiModelStreamerLoader（Runai 流式加载器）等。

from __future__ import annotations

# ruff: noqa: SIM117
import collections
import dataclasses
import fnmatch
import gc
import glob
import json
import logging
import math
import os
import re
import socket
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, suppress
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import huggingface_hub
import numpy as np
import torch

from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    get_remote_instance_transfer_engine_info_per_rank,
    register_memory_region,
)
from sglang.srt.server_args import get_global_server_args

# Try to import accelerate (optional dependency)
try:
    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate.utils import get_max_memory

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    infer_auto_device_map = None
    init_empty_weights = None
    get_max_memory = None

from huggingface_hub import HfApi, hf_hub_download
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.connector import (
    ConnectorType,
    create_remote_connector,
    get_connector_type,
)
from sglang.srt.connector.utils import parse_model_name
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
)
from sglang.srt.layers.modelopt_utils import QUANT_CFG_CHOICES
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    trigger_transferring_weights_request,
)
from sglang.srt.model_loader.utils import (
    get_model_architecture,
    set_default_torch_dtype,
)

# Constants for memory management
DEFAULT_GPU_MEMORY_FRACTION_FOR_CALIBRATION = (
    0.8  # Reserve 20% GPU memory headroom for ModelOpt calibration
)
from sglang.srt.environ import envs
from sglang.srt.model_loader.weight_utils import (
    buffered_multi_thread_safetensors_weights_iterator,
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    get_gguf_extra_tensor_names,
    get_quant_config,
    gguf_quant_weights_iterator,
    initialize_dummy_weights,
    maybe_add_mtp_safetensors,
    multi_thread_pt_weights_iterator,
    np_cache_weights_iterator,
    pt_weights_iterator,
    safetensors_weights_iterator,
    set_runai_streamer_env,
)
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_capability,
    is_npu,
    is_pin_memory_available,
    rank0_log,
    set_weight_attrs,
)

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

_is_npu = is_npu()
# ModelOpt: QUANT_CFG_CHOICES is imported from modelopt_utils.py
# which contains the complete mapping of quantization config choices

logger = logging.getLogger(__name__)


@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
    # 设备加载上下文管理器：将模块参数临时移至目标设备，操作完成后恢复原设备
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        # 如果目标设备是 CPU，无需移动任何参数
        yield module
        return

    original_infos: Dict[str, Dict] = {}
    # 保存原始设备状态，并将 CPU 上的参数移至 GPU

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_data = p.data
            device_data = p.data.to(target_device)  # 将参数数据移至目标设备
            original_infos[name] = dict(
                device=p.device,
                original_data=original_data,
                device_data=device_data,
            )
            p.data = device_data
        # Parameters already on target device are not touched
        # 已在目标设备上的参数不做处理

    try:
        yield module

    finally:
        # Restore parameters to their original devices, ignoring new parameters
        # 将参数恢复到原始设备，忽略新增的参数
        pin_memory = is_pin_memory_available()
        for name, p in module.named_parameters():
            if name in original_infos:
                original_info = original_infos[name]
                device_data = original_info["device_data"]
                original_data = original_info["original_data"]
                original_device: torch.device = original_info["device"]

                if (
                    (device_data.device == p.data.device)
                    and (device_data.data_ptr() == p.data.data_ptr())
                    and (device_data.shape == p.data.shape)
                    and (device_data.dtype == p.data.dtype)
                ):
                    # 数据未变化，直接复制回原始数据并恢复
                    original_data.copy_(p.data.to(original_data.device))
                    p.data = original_data
                elif original_device.type == "cpu":
                    # `torch.empty_like` does not support `pin_memory` argument
                    # 原始设备为 CPU，需要创建新的 CPU 张量并复制数据
                    cpu_data = torch.empty_strided(
                        size=p.data.size(),
                        stride=p.data.stride(),
                        dtype=p.data.dtype,
                        layout=p.data.layout,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    cpu_data.copy_(p.data)
                    p.data = cpu_data
                else:
                    p.data = p.data.to(original_device)
        # New parameters or parameters already on target device are untouched
        # 新增参数或已在目标设备上的参数不做处理


logger = logging.getLogger(__name__)


def _get_quantization_config(
    model_config: ModelConfig,
    load_config: LoadConfig,
) -> Optional[QuantizationConfig]:
    """Get the quantization config."""
    # 获取量化配置，根据模型配置和加载配置返回对应的 QuantizationConfig
    model_class, _ = get_model_architecture(model_config)
    packed_modules_mapping = getattr(model_class, "packed_modules_mapping", {})
    remap_prefix = getattr(model_class, "remap_prefix", None)
    # TODO: we should remove this code and switch to the packed_modules_mapping declared inside the modeling files
    # TODO: 我们应该移除此代码，改用 modeling 文件中声明的 packed_modules_mapping
    if model_config.quantization == "quark":
        packed_modules_mapping.update(
            {
                "gate_up_proj": ["gate_proj", "up_proj"],
                "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
            }
        )  # 针对 quark 量化添加打包模块映射

    if _is_npu:
        packed_modules_mapping.update(
            {
                "visual": {
                    "qkv_proj": ["qkv"],
                    "gate_up_proj": ["gate_proj", "up_proj"],
                },
                "vision_model": {
                    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                    "proj": ["out_proj"],
                },
                "model": {
                    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                    "gate_up_proj": ["gate_proj", "up_proj"],
                    "fused_qkv_a_proj_with_mqa": [
                        "q_a_proj",
                        "kv_a_proj_with_mqa",
                    ],
                },
            }
        )  # NPU 平台需要额外的打包模块映射

    if model_config.quantization is not None:
        quant_config = get_quant_config(
            model_config, load_config, packed_modules_mapping, remap_prefix
        )
        # (yizhang2077) workaround for nvidia/Llama-4-Maverick-17B-128E-Eagle3
        # 针对 nvidia/Llama-4-Maverick-17B-128E-Eagle3 的临时解决方案
        if quant_config is None:
            return None
        # Carry DSV4 expert layout into Fp8Config so downstream readers don't read env.
        # 将 DSV4 专家布局信息传递给 Fp8Config，以便下游读取
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        if isinstance(quant_config, Fp8Config):
            quant_config.is_fp4_experts = model_config.is_fp4_experts  # 设置是否为 FP4 专家
        if not _is_npu:
            major, minor = get_device_capability()

            if major is not None and minor is not None:
                assert 0 <= minor < 10
                capability = major * 10 + minor  # 计算设备能力值
                if capability < quant_config.get_min_capability():
                    raise ValueError(
                        f"The quantization method {model_config.quantization} "
                        "is not supported for the current GPU. "
                        f"Minimum capability: {quant_config.get_min_capability()}. "
                        f"Current capability: {capability}."
                    )  # 检查 GPU 能力是否满足量化方法的最低要求
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}"
            )  # 检查模型数据类型是否在量化方法支持的范围内
        hf_to_sglang_mapper = getattr(model_class, "hf_to_sglang_mapper", None)
        # pass mappings by reference to quant_config
        # 将映射按引用传递给 quant_config
        if hf_to_sglang_mapper is not None and quant_config is not None:
            quant_config.apply_weight_name_mapper(hf_to_sglang_mapper)  # 应用权重名称映射
        return quant_config
    return None


def _initialize_model(
    model_config: ModelConfig,
    load_config: LoadConfig,
    quant_config: Optional[QuantizationConfig] = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    # 根据给定的配置初始化模型实例
    model_class, _ = get_model_architecture(model_config)
    kwargs = {
        "config": model_config.hf_config,
        "quant_config": quant_config,
    }

    # Only add sparse head kwargs if envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()
    # 仅在设置了稀疏嵌入头环境变量时添加相关参数
    if envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set():
        kwargs["sparse_head"] = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.get()
        kwargs["model_path"] = model_config.model_path

    if load_config.draft_model_idx is not None:
        kwargs["draft_model_idx"] = load_config.draft_model_idx  # 添加草稿模型索引参数

    return model_class(**kwargs)


def _post_load_weights(model: nn.Module) -> None:
    # 权重加载后的后处理：调用模型的 post_load_weights 方法进行修正
    # Loaders that bypass `model.load_weights()` (dummy / sharded state / remote instance /
    # remote fs) must trigger the model's post-load fixup explicitly; `model.load_weights()`
    # would normally do it internally. NextN subclasses override the method to fill in
    # `is_nextn=True`, so the loader doesn't need to know.
    if hasattr(model, "post_load_weights"):
        model.post_load_weights()


class BaseModelLoader(ABC):
    """Base class for model loaders."""
    # 模型加载器基类，定义了下载模型和加载模型的抽象接口

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        # 下载模型，使其可以被立即加载
        raise NotImplementedError

    @abstractmethod
    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        """Load a model with the given configurations."""
        # 根据给定配置加载模型
        raise NotImplementedError


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""
    # 默认模型加载器，可以从磁盘加载不同文件类型的模型权重

    # default number of thread when enable multithread weight loading
    # 启用多线程权重加载时的默认线程数
    DEFAULT_NUM_THREADS = 8

    _MTP_PATTERN = re.compile(r"model\.mtp\.layers\.(\d+)\.")  # MTP 层名称匹配模式

    @dataclasses.dataclass
    class Source:
        """A source for weights."""
        # 权重来源数据类，描述从哪里加载权重

        model_or_path: str
        """The model ID or path."""
        # 模型 ID 或路径

        revision: Optional[str]
        """The optional model revision."""
        # 可选的模型版本

        prefix: str = ""
        """A prefix to prepend to all weights."""
        # 所有权重名称前添加的前缀

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""
        # 是否可以回退使用 .pt 权重文件

        model_config: Optional["ModelConfig"] = None
        """The model configuration (for checking architecture, etc)."""
        # 模型配置（用于检查架构等）

        @classmethod
        def init_new(cls, model_config: ModelConfig, model):
            # 根据模型配置和模型实例创建新的 Source 对象
            return cls(
                model_config.model_path,
                model_config.revision,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                model_config=model_config,
            )

    counter_before_loading_weights: float = 0.0  # 加载权重前的计时器
    counter_after_loading_weights: float = 0.0  # 加载权重后的计时器

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys  # 检查不允许的配置键

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )  # 发现未预期的配置键，抛出错误

    def _maybe_download_from_modelscope(
        self, model: str, revision: Optional[str]
    ) -> str:
        """Download model from ModelScope hub if SGLANG_USE_MODELSCOPE is True.

        Returns the path to the downloaded model, or the original model path if
        not downloaded from ModelScope."""
        # 如果设置了 SGLANG_USE_MODELSCOPE，则从 ModelScope 下载模型；否则返回原始路径
        if get_bool_env_var("SGLANG_USE_MODELSCOPE"):
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            # 延迟导入 modelscope，以便正常使用时不需要安装
            # pylint: disable=C.
            from modelscope.hub.snapshot_download import snapshot_download

            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=self.load_config.download_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    revision=revision,
                    ignore_file_pattern=self.load_config.ignore_patterns,
                )  # 从 ModelScope 下载模型快照
            else:
                model_path = model
            return model_path
        return model

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        # 准备模型权重文件：如果是远程模型则下载，返回文件夹路径、权重文件列表和是否使用 safetensors
        model_name_or_path = self._maybe_download_from_modelscope(
            model_name_or_path, revision
        )

        is_local = os.path.isdir(model_name_or_path)  # 检查是否为本地路径
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        # Some quantized models use .pt files for storing the weights.
        # 某些量化模型使用 .pt 文件存储权重
        if load_format == LoadFormat.AUTO:
            allow_patterns = ["*.safetensors", "*.bin"]  # 自动模式允许 safetensors 和 bin
        elif (
            load_format == LoadFormat.SAFETENSORS
            or load_format == LoadFormat.FASTSAFETENSORS
        ):
            use_safetensors = True
            allow_patterns = ["*.safetensors"]  # 仅允许 safetensors 格式
        elif load_format == LoadFormat.MISTRAL:
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]  # Mistral 模型格式
            index_file = "consolidated.safetensors.index.json"
        elif load_format == LoadFormat.PT:
            allow_patterns = ["*.pt"]  # 仅允许 PyTorch 格式
        elif load_format == LoadFormat.NPCACHE:
            allow_patterns = ["*.bin"]  # 仅允许 bin 格式（用于 NP 缓存）
        elif load_format == LoadFormat.DUMMY:
            raise ValueError(
                f"DUMMY load_format should use DummyModelLoader and not call _prepare_weights"
            )
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]  # 允许回退到 .pt 文件

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )  # 从 HuggingFace 下载权重文件
        else:
            hf_folder = model_name_or_path

        server_args = get_global_server_args()
        if server_args and server_args.model_checksum is not None:
            # 如果设置了模型校验和，验证下载的文件
            from sglang.srt.utils.model_file_verifier import verify

            checksums_source = server_args.model_checksum or model_name_or_path
            verify(model_path=hf_folder, checksums_source=checksums_source)  # 验证模型文件校验和

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True  # 如果找到 safetensors 文件，标记使用
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # 对于某些模型（如 Mistral-7B-Instruct-v0.3），同时存在分片和合并的 safetensors 文件，
            # 同时使用两者会出问题。
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            # 下载索引文件并过滤掉不在索引中的文件
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )  # 过滤重复的 safetensors 文件
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )  # 未找到任何权重文件，抛出错误

        if envs.SGLANG_SORT_WEIGHT_FILES.get():
            hf_weights_files.sort()  # 按环境变量决定是否排序权重文件

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        # 根据加载格式获取模型权重的迭代器
        extra_config = self.load_config.model_loader_extra_config
        use_multithread = extra_config.get("enable_multithread_load", True)  # 是否启用多线程加载
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path, source.revision, source.fall_back_to_pt
        )

        if use_safetensors and source.model_config is not None:
            hf_weights_files = maybe_add_mtp_safetensors(
                hf_weights_files,
                hf_folder,
                "model.safetensors.index.json",
                source.model_config.hf_config,
            )  # 可能需要添加 MTP（多token预测）的 safetensors 文件

        if self.load_config.load_format == LoadFormat.NPCACHE:
            # Currently np_cache only support *.bin checkpoints
            # 目前 np_cache 仅支持 *.bin 格式的检查点
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
            )
        elif use_safetensors:
            server_args = get_global_server_args()
            weight_loader_disable_mmap = server_args.weight_loader_disable_mmap  # 是否禁用内存映射
            weight_loader_prefetch = server_args.weight_loader_prefetch_checkpoints  # 是否预取检查点
            prefetch_num_threads = server_args.weight_loader_prefetch_num_threads  # 预取线程数
            weight_loader_drop_cache_after_load = (
                server_args.weight_loader_drop_cache_after_load
            )  # 加载后是否丢弃缓存

            if self.load_config.load_format == LoadFormat.FASTSAFETENSORS:
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                )  # 使用快速 safetensors 迭代器
            elif use_multithread:
                weights_iterator = buffered_multi_thread_safetensors_weights_iterator(
                    hf_weights_files,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                    disable_mmap=weight_loader_disable_mmap,
                    prefetch=weight_loader_prefetch,
                    prefetch_num_threads=prefetch_num_threads,
                    drop_cache_after_load=weight_loader_drop_cache_after_load,
                )  # 使用多线程缓冲 safetensors 迭代器
            else:
                weights_iterator = safetensors_weights_iterator(
                    hf_weights_files,
                    disable_mmap=weight_loader_disable_mmap,
                    prefetch=weight_loader_prefetch,
                    prefetch_num_threads=prefetch_num_threads,
                    drop_cache_after_load=weight_loader_drop_cache_after_load,
                )  # 使用单线程 safetensors 迭代器

        else:
            if use_multithread:
                weights_iterator = multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                )  # 使用多线程 PyTorch 权重迭代器
            else:
                weights_iterator = pt_weights_iterator(hf_weights_files)  # 使用单线程 PyTorch 权重迭代器

        if self.load_config.draft_model_idx is not None:
            # 如果有草稿模型索引，过滤 MTP 权重
            return self._filter_mtp_weights(
                weights_iterator, source.prefix, self.load_config.draft_model_idx
            )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()  # 记录开始加载权重的时间
        # Apply the prefix.
        # 应用前缀到权重名称
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    @classmethod
    def _filter_mtp_weights(
        cls, weights_iterator, prefix: str, draft_model_idx: int
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Filter MTP weights to keep only the specified draft model layer
        and remap it to layer 0. Yields lazily so the upstream buffered
        iterator's sliding window actually bounds CPU memory — eager
        materialization caused page-reclaim hangs on large MoE checkpoints
        with multi-layer EAGLE."""
        # 过滤 MTP（多token预测）权重，仅保留指定草稿模型层并重映射到第 0 层。
        # 使用惰性生成以避免大模型导致内存问题。
        for name, tensor in weights_iterator:
            match = cls._MTP_PATTERN.match(name)
            if match is not None:
                idx = int(match.group(1))  # 提取 MTP 层索引
                if idx != draft_model_idx:
                    continue  # 跳过非目标草稿模型层
                new_name = name.replace(match.group(), "model.mtp.layers.0.")  # 重映射到第 0 层
            else:
                new_name = name
            yield (prefix + new_name, tensor)

    def _get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        # 获取模型的所有权重，包括主权重和辅助权重
        primary_weights = DefaultModelLoader.Source.init_new(model_config, model)
        yield from self._get_weights_iterator(primary_weights)  # 迭代主权重

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source], getattr(model, "secondary_weights", ())
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)  # 迭代辅助权重

    def download_model(self, model_config: ModelConfig) -> None:
        # 下载模型，准备权重文件
        self._prepare_weights(
            model_config.model_path, model_config.revision, fall_back_to_pt=True
        )

    def _load_modelopt_base_model(self, model_config: ModelConfig) -> nn.Module:
        """Load and prepare the base model for ModelOpt quantization.

        This method handles the common model loading logic shared between
        DefaultModelLoader (conditional) and ModelOptModelLoader (dedicated).
        """
        # 加载并准备用于 ModelOpt 量化的基础模型
        if not HAS_ACCELERATE:
            raise ImportError(
                "accelerate is required for ModelOpt quantization. "
                "Please install it with: pip install accelerate"
            )  # 需要安装 accelerate 库

        try:
            hf_config = AutoConfig.from_pretrained(
                model_config.model_path,
                trust_remote_code=True,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )  # 加载 HuggingFace 配置
        except (KeyError, ValueError):
            from sglang.srt.utils.hf_transformers_utils import get_config

            hf_config = get_config(
                model_config.model_path,
                trust_remote_code=True,
            )  # 回退到自定义配置加载
        with init_empty_weights():
            torch_dtype = getattr(hf_config, "torch_dtype", torch.float16)
            model = AutoModelForCausalLM.from_config(
                hf_config, torch_dtype=torch_dtype, trust_remote_code=True
            )  # 在空权重上下文中创建模型
        max_memory = get_max_memory()  # 获取最大可用内存
        inferred_device_map = infer_auto_device_map(model, max_memory=max_memory)  # 推断设备映射

        on_cpu = "cpu" in inferred_device_map.values()  # 检查是否需要在 CPU 上放置部分层
        model_kwargs = {"torch_dtype": "auto"}
        device_map = "auto"

        if on_cpu:
            for device in max_memory.keys():
                if isinstance(device, int):
                    max_memory[device] *= DEFAULT_GPU_MEMORY_FRACTION_FOR_CALIBRATION  # 为校准预留 GPU 内存

            logger.warning(
                "Model does not fit to the GPU mem. "
                f"We apply the following memory limit for calibration: \n{max_memory}\n"
                f"If you hit GPU OOM issue, please adjust the memory fraction "
                f"(currently {DEFAULT_GPU_MEMORY_FRACTION_FOR_CALIBRATION}) or "
                "reduce the calibration `batch_size` manually."
            )  # 模型不适合 GPU 内存，应用内存限制
            model_kwargs["max_memory"] = max_memory

        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_path,
            config=hf_config,
            device_map=device_map,
            **model_kwargs,
            trust_remote_code=True,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )  # 加载预训练模型
        # Handle both legacy modelopt_quant and unified quantization flags
        # 处理旧版 modelopt_quant 和统一量化标志
        if hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant:
            # Legacy approach
            # 旧版方式
            quant_choice_str = model_config.modelopt_quant
            rank0_log(f"ModelOpt quantization requested (legacy): {quant_choice_str}")
        else:
            # Unified approach - extract quantization type
            # 统一方式 - 提取量化类型
            quant_choice_str = model_config._get_modelopt_quant_type()
            rank0_log(
                f"ModelOpt quantization requested (unified): {model_config.quantization} -> {quant_choice_str}"
            )

        if not isinstance(quant_choice_str, str):
            raise TypeError(
                f"Quantization type must be a string (e.g., 'fp8'), "
                f"got {type(quant_choice_str)}"
            )  # 量化类型必须是字符串

        return model

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        # 加载模型：初始化模型、加载权重、后处理
        if hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant:
            # Load base model using shared method
            # 使用共享方法加载基础模型
            model = self._load_modelopt_base_model(model_config)
            # Note: DefaultModelLoader doesn't do additional quantization processing
            # For full ModelOpt quantization, use ModelOptModelLoader
            # 注意：DefaultModelLoader 不做额外的量化处理，完整量化请使用 ModelOptModelLoader
            return model.eval()

        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )  # 在目标设备上初始化模型

            self.load_weights_and_postprocess(
                model, self._get_all_weights(model_config, model), target_device
            )  # 加载权重并进行后处理

        self.counter_after_loading_weights = time.perf_counter()  # 记录权重加载完成时间
        return model.eval()

    @staticmethod
    def load_weights_and_postprocess(model, weights, target_device):
        # 加载权重并对模块进行量化后处理
        model.load_weights(weights)

        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                # When quant methods need to process weights after loading
                # (for repacking, quantizing, etc), they expect parameters
                # to be on the global target device. This scope is for the
                # case where cpu offloading is used, where we will move the
                # parameters onto device for processing and back off after.
                # 量化方法在加载后需要处理权重（如重打包、量化等）时，
                # 期望参数在全局目标设备上。此上下文用于 CPU 卸载场景，
                # 将参数移至设备处理后再移回。
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)


class LayeredModelLoader(DefaultModelLoader):
    """Model loader that loads weights layer by layer so that one can quantize a
    layer before loading another to make the peak memory envelope smaller."""
    # 分层模型加载器：逐层加载权重，以便在加载下一层之前量化当前层，减少峰值内存占用

    def __init__(self, load_config: LoadConfig):
        # Back to the default load format
        # 恢复为默认加载格式
        load_config.load_format = LoadFormat.AUTO
        super().__init__(load_config)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        # 逐层加载模型权重，支持逐层量化以降低峰值内存
        from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
        from sglang.srt.server_args import get_global_server_args

        torchao_config = get_global_server_args().torchao_config
        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            # Create model on meta device
            # 在 meta 设备上创建模型（不分配实际内存）
            with torch.device("meta"):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

            # Check model's layered load support
            # 检查模型是否支持分层加载
            if not hasattr(model, "load_weights_to_module"):
                raise ValueError(
                    "LayeredModelLoader requires the model to have a "
                    "`load_weights_to_module` method. "
                    f"{model_config.model_path} does not support it."
                )

            # Get all weights from disk
            # 从磁盘获取所有权重
            weights = self._get_all_weights(model_config, model)

            # Helper function to recursively fill the weights of a module
            # 递归填充模块权重的辅助函数
            def fill_module(module, fqn: List[str], weights):
                """
                fqn: list of strings representing the fully qualified name of `module`.
                """
                # Layer by layer
                # 逐层递归处理
                for name, submod in module.named_children():
                    fill_module(submod, fqn + [name], weights)

                # First materialize on target device
                # 先在目标设备上物化模块
                module.to_empty(device=target_device, recurse=False)
                fqn_path = ".".join(fqn)
                # Fill weights
                # 填充权重
                model.load_weights_to_module(
                    fqn_path,
                    weights,
                )
                # Quantize weights if applicable
                # 如果适用，量化权重
                if torchao_config and "proj" in fqn_path:
                    # Note: `None` here is needed to indicate no filter, see
                    # `apply_torchao_config_to_model` for details.
                    apply_torchao_config_to_model(module, torchao_config, None)

            # Start calling on root module
            # 从根模块开始递归调用
            fill_module(model, [], weights)

        if torchao_config:
            model.torchao_applied = True

        return model.eval()


class QuantizedRLModelLoader(DefaultModelLoader):
    """
    Model loader for RL training with FP8 quantization (profile-free, native SGLang).
    # 用于 RL 训练的 FP8 量化模型加载器（免分析，原生 SGLang 支持）

    Workflow:
    # 工作流程：
      1. Initial load: Load base model → Record state → Apply FP8 quantization
      # 1. 初始加载：加载基础模型 → 记录状态 → 应用 FP8 量化
      2. Training Actor in full precision
      # 2. 以全精度训练 Actor
      3. Reload: Trainer sends full precision weights → Quantize to FP8 → Copy to original memory
      # 3. 重新加载：训练器发送全精度权重 → 量化为 FP8 → 复制到原始内存
      4. Use torch.as_strided to preserve memory locations across reloads
      # 4. 使用 torch.as_strided 在重新加载时保持内存位置不变

    Usage:
      --model-path Qwen/Qwen2.5-7B --quantization fp8 --load-format flash_rl
    """

    # Parameter attributes to record for weight reloading
    # 权重重新加载时需要记录的参数属性
    RECORDED_LOADER_KEYS = [
        "weight_loader",
        "load_qkv_weight",
        "load_column_parallel_weight",
        "load_row_parallel_weight",
        "load_merged_column_weight",
        "output_dim",
        "input_dim",
        "_assert_and_load",
    ]

    # Parameters to skip during FP8 quantization (matches FlashRL's exclude_list)
    # FP8 量化时跳过的参数列表（与 FlashRL 的排除列表一致）
    SKIP_QUANTIZATION_PARAMS = [
        "weight_scale",
        "input_scale",
        "output_scale",
        ".bias",
        "lm_head.weight",
        "model.norm.weight",
        "embed_tokens",  # BF16 params / BF16 参数
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
        "projector",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",  # LayerNorms / 层归一化
    ]

    # Stacked parameters (Qwen2): shards loaded separately, then combined
    # 堆叠参数（Qwen2）：分片单独加载后合并
    STACKED_PARAMS_MAPPING = [
        ("qkv_proj", ["q_proj", "k_proj", "v_proj"]),
        ("gate_up_proj", ["gate_proj", "up_proj"]),
    ]
    _QKV_SHARD_ALIASES = {  # QKV 分片别名映射
        "q_proj": "q",
        "k_proj": "k",
        "v_proj": "v",
    }

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        logger.info("[QuantizedRL] Profile-free FP8 quantization enabled")
        self._initial_load_complete = False  # 标记是否完成初始加载

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ):
        """Standard weight preparation using base model path."""
        # 使用基础模型路径的标准权重准备方法
        logger.info(f"[QuantizedRL] Loading from base model: {model_name_or_path}")
        temp_config = LoadConfig(load_format=LoadFormat.AUTO)
        temp_loader = DefaultModelLoader(temp_config)
        return temp_loader._prepare_weights(
            model_name_or_path, revision, fall_back_to_pt
        )  # 委托给 DefaultModelLoader 的权重准备方法

    @staticmethod
    def _bind_method_to_cls(func, obj):
        """Bind function to object instance (for weight_loader methods)."""
        # 将函数绑定到对象实例（用于 weight_loader 方法）
        import types

        if hasattr(func, "__self__") or not callable(func):
            return func  # 已经是绑定方法或不可调用，直接返回
        return types.MethodType(func, obj)  # 创建绑定方法

    def load_weights_and_postprocess(self, model, weights, target_device):
        """
        Initial load: Load BF16 → Record state → Apply FP8 quantization.
        Called ONCE during model initialization.
        """
        # 初始加载：加载 BF16 权重 → 记录状态 → 应用 FP8 量化。仅在模型初始化时调用一次。
        logger.info("[QuantizedRL] Initial load with FP8 quantization")

        original_load_weights = model.load_weights

        def load_weights_proxy(weights):
            # 权重加载代理：根据是否为重新加载场景选择不同路径
            if QuantizedRLModelLoader.is_reload_scenario(model):
                logger.info("[QuantizedRL] Using fast path reload in load_weights")
                QuantizedRLModelLoader.rebinding_and_load_weights(
                    model, original_load_weights, weights
                )  # 重新加载场景：快速路径
            else:
                original_load_weights(weights)  # 首次加载：使用原始加载方法

        model.load_weights = load_weights_proxy  # 替换模型的 load_weights 方法

        model.load_weights(weights)
        original_weights = dict(model.named_parameters())

        # Record pre-quantization state (shape/stride) for torch.as_strided reset
        # 记录量化前的状态（形状/步幅），用于 torch.as_strided 重置

        model.original_weights_rebuild_keys = {}
        for name, p in original_weights.items():
            model.original_weights_rebuild_keys[name] = {
                "shape": p.shape,
                "stride": p.stride(),
                "dtype": p.dtype,
                "nbytes": p.untyped_storage().nbytes(),
            }  # 保存每个参数的形状、步幅、数据类型和字节数

        # Record parameter attributes (weight_loader, etc.) before quantization
        # 量化前记录参数属性（weight_loader 等）
        recorded_loader = {
            k: dict() for k in QuantizedRLModelLoader.RECORDED_LOADER_KEYS
        }
        for name, p in original_weights.items():
            for key in QuantizedRLModelLoader.RECORDED_LOADER_KEYS:
                if hasattr(p, key):
                    attr = getattr(p, key)
                    if not callable(attr):
                        recorded_loader[key][name] = attr
                    elif hasattr(attr, "__self__") and p is attr.__self__:
                        recorded_loader[key][name] = attr.__func__  # Store unbound / 存储未绑定方法
                    else:
                        recorded_loader[key][name] = attr
        model.recorded_loader = recorded_loader  # 保存记录的加载器属性

        # Apply FP8 quantization (creates new Parameters, loses attributes)
        # 应用 FP8 量化（创建新参数，会丢失属性）
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)  # 量化处理后加载

        model.flash_rl_initial_load_complete = True
        self._initial_load_complete = True  # 标记初始加载完成
        logger.info("[QuantizedRL] Initial load complete")

    @staticmethod
    def is_reload_scenario(model):
        """Check if model is ready for reloading (initial load completed)."""
        # 检查模型是否已准备好重新加载（初始加载已完成）
        return (
            hasattr(model, "original_weights_rebuild_keys")
            and hasattr(model, "recorded_loader")
            and getattr(model, "flash_rl_initial_load_complete", False)
        )

    @staticmethod
    def _is_stacked_param(name):
        """Check if parameter is stacked (qkv_proj, gate_up_proj)."""
        # 检查参数是否为堆叠参数（如 qkv_proj、gate_up_proj）
        for stacked_name, _ in QuantizedRLModelLoader.STACKED_PARAMS_MAPPING:
            if stacked_name in name:
                return True
        return False

    @staticmethod
    def _resolve_stacked_info(name: str) -> Tuple[str, Optional[str], Optional[Any]]:
        # 解析堆叠参数信息，返回目标名称、堆叠键和分片 ID
        for target, shard_names in QuantizedRLModelLoader.STACKED_PARAMS_MAPPING:
            for idx, shard in enumerate(shard_names):
                if shard in name:
                    shard_id = (
                        QuantizedRLModelLoader._QKV_SHARD_ALIASES.get(shard, shard)
                        if target == "qkv_proj"
                        else idx
                    )  # QKV 使用别名，gate_up 使用索引
                    return name.replace(shard, target), target, shard_id
        return name, None, None

    @staticmethod
    def _store_quantized_scale(
        scale_store: Dict[str, Union[torch.Tensor, Dict[Any, torch.Tensor]]],
        name: str,
        scale: torch.Tensor,
    ) -> None:
        # 存储量化缩放因子，处理普通参数和堆叠参数两种情况
        param_name, stacked_key, shard_id = (
            QuantizedRLModelLoader._resolve_stacked_info(name)
        )
        if stacked_key is None:
            scale_store[param_name] = scale  # 非堆叠参数，直接存储
        else:
            shard_dict = scale_store.setdefault(param_name, {})
            assert isinstance(shard_dict, dict)
            shard_dict[shard_id] = scale  # 堆叠参数，按分片 ID 存储

    @staticmethod
    def _apply_scale_update(
        all_params: Dict[str, torch.nn.Parameter],
        param_name: str,
        scale_info: Union[torch.Tensor, Dict[Any, torch.Tensor], None],
    ) -> None:
        # 应用缩放因子更新到模型参数
        if scale_info is None:
            return
        # Get tp rank and size
        # 获取张量并行排名和大小
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        def _get_tp_sharded_scale(full_scale_tensor):
            """Get tp sharded scale from full scale tensor"""
            # 从完整缩放张量中获取张量并行分片的缩放
            if tp_size == 1:
                return full_scale_tensor

            full_dim = full_scale_tensor.shape[0]
            shard_dim = full_dim // tp_size
            start_idx = tp_rank * shard_dim
            end_idx = start_idx + shard_dim
            return full_scale_tensor[start_idx:end_idx]  # 返回当前 rank 对应的分片

        if param_name.endswith(".weight"):
            scale_param_name = f"{param_name[:-7]}.weight_scale"
        else:
            scale_param_name = f"{param_name}.weight_scale"  # 构造缩放参数名称

        scale_param = all_params.get(scale_param_name)
        if scale_param is None:
            logger.warning(
                "[QuantizedRL] Scale parameter not found: %s", scale_param_name
            )  # 未找到缩放参数
            return
        if isinstance(scale_info, torch.Tensor):
            new_scale = scale_info.t().contiguous()  # 转置并确保连续
            if scale_param.data.shape == new_scale.shape:
                scale_param.data.copy_(new_scale)  # 形状匹配，直接复制
            else:
                logger.warning(
                    "[QuantizedRL] Scale shape mismatch for %s: expected %s, got %s",
                    scale_param_name,
                    scale_param.data.shape,
                    new_scale.shape,
                )  # 形状不匹配
        else:
            # 处理堆叠参数的缩放更新
            stacked_key = next(
                (
                    target
                    for target, _ in QuantizedRLModelLoader.STACKED_PARAMS_MAPPING
                    if target in param_name
                ),
                None,
            )
            shard_names = next(
                (
                    names
                    for target, names in QuantizedRLModelLoader.STACKED_PARAMS_MAPPING
                    if target == stacked_key
                ),
                [],
            )
            rows_per_shard = scale_param.data.shape[-1] // max(len(shard_names), 1)  # 每个分片的行数
            if rows_per_shard * len(shard_names) != scale_param.data.shape[-1]:
                logger.warning(
                    f"Scale param shape {scale_param.data.shape[-1]} not divisible by {len(shard_names)}"
                )  # 缩放参数形状不可整除
            offset = 0
            for idx, shard in enumerate(shard_names):
                shard_id = (
                    QuantizedRLModelLoader._QKV_SHARD_ALIASES.get(shard, shard)
                    if stacked_key == "qkv_proj"
                    else idx
                )
                shard_scale = scale_info.get(shard_id)
                shard_scale = _get_tp_sharded_scale(shard_scale)  # 获取 TP 分片的缩放
                if shard_scale is None:
                    offset += rows_per_shard
                    continue
                shard_rows = shard_scale.shape[0]
                start = offset
                end = start + shard_rows
                scale_param.data[..., start:end] = shard_scale.t().contiguous()  # 写入对应位置
                offset = end

    @staticmethod
    def rebinding_and_load_weights(model, first_time_load_weights, weights):
        """
        Reload: VERL sends BF16 → Quantize to FP8 → Copy to original memory.
        # 重新加载：VERL 发送 BF16 权重 → 量化为 FP8 → 复制到原始内存

        Flow: Reset params → Restore attributes → Quantize in iterator → Load → Copy back
        # 流程：重置参数 → 恢复属性 → 在迭代器中量化 → 加载 → 复制回原始内存
        """
        logger.info("[QuantizedRL] Reload: Updating weights with FP8 quantization")

        weights_list = list(weights)
        updated_param_names, is_last_update = (
            QuantizedRLModelLoader._get_updated_params(weights_list, model)
        )  # 获取需要更新的参数名称和是否为最后一次更新

        # Save current FP8 parameter data pointers
        # 保存当前 FP8 参数的数据指针
        existing_params = dict(model.named_parameters())
        current_param_data = {}
        for name in updated_param_names:
            if name in existing_params:
                current_param_data[name] = existing_params[name].data

        # Reset to pre-quantization shape using torch.as_strided
        # Keeps same storage, just changes view - critical for memory preservation
        # 使用 torch.as_strided 重置为量化前的形状
        # 保持相同的存储，仅改变视图——对于内存保持至关重要
        for name, rebuild_info in model.original_weights_rebuild_keys.items():
            if name in updated_param_names and name in existing_params:
                existing_params[name].data = torch.as_strided(
                    # Note: avoid clone here
                    # 注意：此处避免使用 clone
                    existing_params[name].data.clone(),
                    rebuild_info["shape"],
                    rebuild_info["stride"],
                )

        # Restore weight loader attributes (only if missing)
        # 恢复权重加载器属性（仅在缺失时恢复）
        for k, loader_dict in model.recorded_loader.items():
            for param_name, loader in loader_dict.items():
                if param_name in updated_param_names and param_name in existing_params:
                    param = existing_params[param_name]
                    if not hasattr(param, k):
                        if callable(loader):
                            if hasattr(loader, "__self__"):
                                setattr(param, k, loader)  # 已绑定方法，直接设置
                            else:
                                setattr(
                                    param,
                                    k,
                                    QuantizedRLModelLoader._bind_method_to_cls(
                                        loader, param
                                    ),
                                )  # 未绑定方法，绑定到参数实例
                        else:
                            setattr(param, k, loader)  # 非可调用属性，直接设置

        del existing_params

        # Quantize BF16 weights to FP8 in iterator (before weight_loader)
        # Store scales for later update
        # 在迭代器中将 BF16 权重量化为 FP8（在 weight_loader 之前）
        # 存储缩放因子以供后续更新
        quantized_scales: Dict[str, Union[torch.Tensor, Dict[Any, torch.Tensor]]] = {}

        def quantize_weights_iterator(weights_iter):
            """Quantize individual shards before weight_loader stacks them."""
            # 在 weight_loader 堆叠分片之前，对每个分片进行量化
            from sglang.srt.layers.quantization.fp8_kernel import (
                per_token_group_quant_fp8,
            )

            for name, weight in weights_iter:
                if any(
                    skip in name
                    for skip in QuantizedRLModelLoader.SKIP_QUANTIZATION_PARAMS
                ):
                    logger.info(f"[QuantizedRL] Skip: {name} ({weight.dtype})")
                    yield (name, weight)  # 跳过不需要量化的参数
                elif weight.dtype in [torch.bfloat16, torch.float32, torch.float16]:
                    qweight, scale = per_token_group_quant_fp8(weight, weight.shape[-1])  # 执行 FP8 量化
                    logger.info(f"[QuantizedRL] Quantize: {name} {weight.dtype}→FP8")
                    QuantizedRLModelLoader._store_quantized_scale(
                        quantized_scales, name, scale
                    )  # 存储量化缩放因子
                    yield (name, qweight)
                else:
                    logger.info(f"[QuantizedRL] Keep: {name} ({weight.dtype})")
                    yield (name, weight)  # 保持原样

        # Load quantized weights (weight_loader stacks FP8 shards)
        # 加载量化后的权重（weight_loader 会堆叠 FP8 分片）
        first_time_load_weights(quantize_weights_iterator(iter(weights_list)))

        # Copy back to original FP8 memory locations and update scales
        # 复制回原始 FP8 内存位置并更新缩放因子
        all_params = dict(model.named_parameters())

        for name in updated_param_names:
            if name not in all_params or name not in current_param_data:
                continue
            if any(
                skip in name for skip in QuantizedRLModelLoader.SKIP_QUANTIZATION_PARAMS
            ):
                continue  # 跳过不需要量化的参数

            new_param = all_params[name]
            old_fp8_data = current_param_data[name]

            # Handle embeddings/lm_head (BF16) and quantized weights (FP8)
            # 处理嵌入/lm_head（BF16）和量化权重（FP8）
            if "embed_tokens" in name or "lm_head" in name:
                old_fp8_data.copy_(new_param.data)
                new_param.data = old_fp8_data  # BF16 参数直接复制
            elif (
                new_param.dtype == torch.float8_e4m3fn
                and old_fp8_data.dtype == torch.float8_e4m3fn
            ):
                # FP8: Use strided view for transposed storage
                # FP8：使用步幅视图处理转置存储
                strided_data = torch.as_strided(
                    new_param.data, old_fp8_data.shape, old_fp8_data.stride()
                )
                old_fp8_data.copy_(strided_data)
                new_param.data = old_fp8_data  # 复制回原始内存位置
                QuantizedRLModelLoader._apply_scale_update(
                    all_params,
                    name,
                    quantized_scales.get(name),
                )  # 更新缩放因子
            elif new_param.dtype == old_fp8_data.dtype:
                # Same dtype (LayerNorm, etc.): Direct copy
                # 相同数据类型（LayerNorm 等）：直接复制
                old_fp8_data.copy_(new_param.data)
                new_param.data = old_fp8_data
            else:
                raise RuntimeError(
                    f"Unexpected dtype mismatch for {name}: "
                    f"new={new_param.dtype}, old={old_fp8_data.dtype}"
                )  # 数据类型不匹配

        # Cleanup
        # 清理临时数据
        del current_param_data
        if is_last_update:
            gc.collect()
            current_platform.empty_cache()  # 如果是最后一次更新，清理内存缓存

        logger.info("[QuantizedRL] Reload complete")
        return updated_param_names, is_last_update

    @staticmethod
    def _get_updated_params(weights_list, model):
        """Identify which parameters need updating from incoming weights."""
        # 从传入的权重中识别需要更新的参数
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(model.named_parameters())
        updated_params = set()
        is_last_update = False

        for name, _ in weights_list:
            if name == "lm_head.weight":
                is_last_update = True  # lm_head.weight 是最后一个更新的参数

            if any(
                skip in name for skip in QuantizedRLModelLoader.SKIP_QUANTIZATION_PARAMS
            ):
                continue  # 跳过不需要量化的参数

            from sglang.srt.layers.utils import get_layer_id

            # Skip params outside layer range (for pipeline parallelism)
            # 跳过层范围之外的参数（用于流水线并行）
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(model, "start_layer")
                and (layer_id < model.start_layer or layer_id >= model.end_layer)
            ):
                continue

            # Skip tied embeddings and vision tower params
            # 跳过共享嵌入和视觉塔参数
            if (
                hasattr(model, "config")
                and model.config.tie_word_embeddings
                and "lm_head.weight" in name
            ):
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            # Map stacked param shards (q/k/v_proj → qkv_proj)
            # 映射堆叠参数分片（如 q/k/v_proj → qkv_proj）
            mapped = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    name = name.replace(weight_name, param_name)
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    updated_params.add(name)
                    mapped = True
                    break

            if not mapped:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict:
                    updated_params.add(name)  # 添加需要更新的参数名

        return list(updated_params), is_last_update


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""
    # 虚拟模型加载器：将模型权重设置为随机值（用于性能测试等场景）

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )  # 虚拟加载器不支持额外配置

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download / 无需下载

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        # 加载虚拟模型（使用随机权重）
        if get_bool_env_var("SGL_CPU_QUANTIZATION"):
            return load_model_with_cpu_quantization(
                self, model_config=model_config, device_config=device_config
            )  # 使用 CPU 量化加载

        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # Skip FusedMoE layers already quantized during init (FP8 or FP4)
                    # 跳过在初始化时已量化的 FusedMoE 层（FP8 或 FP4）
                    if (
                        hasattr(module, "is_weights_quantized")
                        and module.is_weights_quantized()
                    ):
                        continue
                    quant_method.process_weights_after_loading(module)

            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            # 注意：为了准确的性能评估，我们将权重设置为随机值
            initialize_dummy_weights(model)

            _post_load_weights(model)  # 执行权重加载后处理

        return model.eval()


class ShardedStateLoader(BaseModelLoader):
    """
    Model loader that directly loads each worker's model state dict, which
    enables a fast load path for large tensor-parallel models where each worker
    only needs to read its own shard rather than the entire checkpoint. See
    `examples/runtime/engine/save_sharded_state.py` for creating a sharded checkpoint.
    """
    # 分片状态加载器：直接加载每个 worker 的模型状态字典，
    # 为大型张量并行模型提供快速加载路径，每个 worker 只需读取自己的分片。
    # 参见 `examples/runtime/engine/save_sharded_state.py` 创建分片检查点。

    DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"  # 分片文件名模式

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else load_config.model_loader_extra_config.copy()
        )
        self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{load_config.model_loader_extra_config.keys()}"
            )

    @staticmethod
    def _filter_subtensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        # 过滤掉共享相同内存或是另一个张量内存子集的张量
        same_storage_groups: Dict[Any, List[Tuple[str, torch.Tensor]]] = (
            collections.defaultdict(list)
        )
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))  # 按设备和存储指针分组

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result: Dict[str, torch.Tensor] = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 < k:
                        # Same tensors, keep the one with the smaller key.
                        break
                else:
                    result[k] = t
        return result

    def _prepare_weights(self, model_name_or_path: str, revision: Optional[str]):
        # 准备分片权重文件：本地路径直接返回，否则从 HuggingFace 下载
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            allow_patterns = ["*.safetensors"]
            return download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model_path, model_config.revision)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        # 加载分片状态模型：读取每个 worker 对应的分片检查点
        from safetensors.torch import safe_open

        from sglang.srt.distributed import get_tensor_model_parallel_rank

        local_model_path = self._prepare_weights(
            model_config.model_path, model_config.revision
        )

        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, quant_config)
                for _, module in model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
            rank = get_tensor_model_parallel_rank()
            pattern = os.path.join(
                local_model_path,
                self.pattern.format(rank=rank, part="*"),
            )  # 构造当前 rank 的分片文件匹配模式
            filepaths = glob.glob(pattern)
            if not filepaths:
                # TODO: support un-sharded checkpoints too
                # TODO: 也支持未分片的检查点
                raise ValueError(
                    f"Could not find checkpoint files '{pattern}', only "
                    f"pre-sharded checkpoints are currently supported!"
                )
            state_dict = self._filter_subtensors(model.state_dict())
            for path in filepaths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        # If loading with LoRA enabled, additional padding may
                        # be added to certain parameters. We only load into a
                        # narrowed view of the parameter data.
                        # 如果启用了 LoRA 加载，某些参数可能有额外的填充。
                        # 我们只加载到参数数据的窄视图中。
                        param_data = state_dict[key].data
                        param_shape = state_dict[key].shape
                        for dim, size in enumerate(tensor.shape):
                            if size < param_shape[dim]:
                                param_data = param_data.narrow(dim, 0, size)  # 缩窄视图以适应张量大小
                        if tensor.shape != param_shape:
                            logger.warning(
                                "loading tensor of shape %s into "
                                "parameter '%s' of shape %s",
                                tensor.shape,
                                key,
                                param_shape,
                            )
                        param_data.copy_(tensor)
                        state_dict.pop(key)
            if state_dict:
                raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")  # 加载状态中缺少键

            _post_load_weights(model)  # 执行权重加载后处理

        return model.eval()

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        # 保存模型为分片状态格式，支持按大小分片
        from safetensors.torch import save_file

        from sglang.srt.distributed import get_tensor_model_parallel_rank

        if pattern is None:
            pattern = ShardedStateLoader.DEFAULT_PATTERN
        rank = get_tensor_model_parallel_rank()
        part_idx = 0
        total_size = 0
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        state_dict_part: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            param_size = tensor.nelement() * tensor.element_size()
            if max_size is not None and total_size + param_size > max_size:
                filename = pattern.format(rank=rank, part=part_idx)
                save_file(
                    state_dict_part,
                    os.path.join(path, filename),
                )  # 保存当前分片
                part_idx += 1
                total_size = 0
                state_dict_part = {}
            state_dict_part[key] = tensor
            total_size += param_size
        if len(state_dict_part) > 0:
            filename = pattern.format(rank=rank, part=part_idx)
            save_file(
                state_dict_part,
                os.path.join(path, filename),
            )


class BitsAndBytesModelLoader(BaseModelLoader):
    """Model loader to load model weights with BitAndBytes quantization."""
    # BitsAndBytes 量化模型加载器：支持加载使用 BitsAndBytes 量化的模型权重

    possible_config_file_names = ["adapter_config.json"]  # 可能的配置文件名

    default_target_modules = [  # 默认需要量化的目标模块列表
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        ".fc1.",
        ".fc2.",
        ".dense.",
        ".query_key_value.",
        ".qkv_proj.",
        ".dense_h_to_4h.",
        ".dense_4h_to_h.",
        ".out_proj.",
    ]

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        # we don't need to quantize the whole model, only the target modules
        # that are specified in the adapter config file. If the adapter config
        # file is not provided, we will quantize the default modules.
        # 我们不需要量化整个模型，只需量化适配器配置文件中指定的目标模块。
        # 如果未提供适配器配置文件，将量化默认模块。
        if (
            not load_config.model_loader_extra_config
            or "qlora_adapter_name_or_path" not in load_config.model_loader_extra_config
        ):
            self.target_modules = []
            return

        qlora_adapter = load_config.model_loader_extra_config[
            "qlora_adapter_name_or_path"
        ]

        config_file_path = self._get_config_file(qlora_adapter)  # 获取 QLoRA 适配器配置文件路径

        with open(config_file_path, "r") as f:
            config = json.load(f)
            self.target_modules = config["target_modules"]  # 从配置中读取目标模块

    def _get_config_file(self, qlora_adapter: str) -> str:
        # 获取 QLoRA 适配器的配置文件路径，支持本地和 HuggingFace 远程两种方式
        is_local = os.path.isdir(qlora_adapter)
        config_file_path = None
        if is_local:
            for file in self.possible_config_file_names:
                config_file_path = os.path.join(qlora_adapter, file)
                if os.path.exists(config_file_path):
                    break  # 在本地目录中查找配置文件
        else:
            hf_api = HfApi()
            repo_files = hf_api.list_repo_files(repo_id=qlora_adapter)
            for file in self.possible_config_file_names:
                if file in repo_files:
                    config_file_path = hf_hub_download(
                        repo_id=qlora_adapter, filename=file
                    )  # 从 HuggingFace 下载配置文件
                    break

        if not config_file_path:
            raise ValueError(f"Cannot find adapter config file in {qlora_adapter}")

        return config_file_path

    def _get_weight_files(
        self,
        model_name_or_path: str,
        allowed_patterns: List[str],
        revision: Optional[str] = None,
    ) -> Tuple[List[str], str]:
        """Retrieve weight files. Download the files if necessary.

        Return the weight files and the file pattern."""
        # 获取权重文件，必要时下载。返回权重文件列表和匹配的模式
        is_local = os.path.isdir(model_name_or_path)

        if is_local:
            for pattern in allowed_patterns:
                weight_files = glob.glob(os.path.join(model_name_or_path, pattern))
                if weight_files:
                    return weight_files, pattern  # 本地查找权重文件
        else:
            hf_api = HfApi()
            repo_files = hf_api.list_repo_files(repo_id=model_name_or_path)
            for pattern in allowed_patterns:
                matching_files = fnmatch.filter(repo_files, pattern)
                if matching_files:
                    hf_folder = download_weights_from_hf(
                        model_name_or_path,
                        self.load_config.download_dir,
                        [pattern],
                        revision,
                        ignore_patterns=self.load_config.ignore_patterns,
                    )  # 从 HuggingFace 下载权重
                    return glob.glob(os.path.join(hf_folder, pattern)), pattern

        raise RuntimeError(f"No model weights found in: `{model_name_or_path}`")

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str]
    ) -> Tuple[List[str], bool]:
        """Prepare weight files for the model."""
        # 准备模型的权重文件，返回文件列表和是否为 safetensors 格式

        allowed_patterns = ["*.safetensors", "*.bin", "*.pt"]

        hf_weights_files, matched_pattern = self._get_weight_files(
            model_name_or_path, allowed_patterns, revision
        )

        if matched_pattern != "*.safetensors":
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)  # 过滤非推理所需文件

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_weights_files, matched_pattern == "*.safetensors"

    def _hf_weight_iter(self, hf_weights_files, use_safetensors: bool):
        # 根据文件格式返回 safetensors 或 PyTorch 权重迭代器
        if use_safetensors:
            return safetensors_weights_iterator(hf_weights_files)
        else:
            return pt_weights_iterator(hf_weights_files)

    def _get_quantized_weights_iterator(
        self,
        model_name_or_path: str,
        revision: Optional[str],
        pre_quant: bool,
        load_8bit: bool,
    ) -> Tuple[Generator[Tuple[str, torch.Tensor], None, None], Dict[str, Any]]:
        """Get an iterator to the model weights with bitsandbytes quantization,
        as well as the quantization state dictionary."""
        # 获取带有 bitsandbytes 量化的模型权重迭代器及量化状态字典

        # only load the bitsandbytes module when needed
        # 仅在需要时加载 bitsandbytes 模块
        try:
            import bitsandbytes

            if bitsandbytes.__version__ < "0.44.0":
                raise ImportError(
                    "bitsandbytes version is wrong. Please "
                    "install bitsandbytes>=0.44.0."
                )
        except ImportError as err:
            raise ImportError(
                "Please install bitsandbytes>=0.44.0 via "
                "`pip install bitsandbytes>=0.44.0` to use "
                "bitsandbytes quantizer."
            ) from err

        hf_weights_files, use_safetensors = self._prepare_weights(
            model_name_or_path, revision
        )

        quant_state_dict: Dict[str, Any] = {}

        if pre_quant:
            if load_8bit:
                return (
                    self._quantized_8bit_generator(
                        hf_weights_files, use_safetensors, quant_state_dict
                    ),  # 8-bit 预量化权重生成器
                    quant_state_dict,
                )
            else:
                return (
                    self._quantized_4bit_generator(
                        hf_weights_files, use_safetensors, quant_state_dict
                    ),  # 4-bit 预量化权重生成器
                    quant_state_dict,
                )

        return (
            self._unquantized_generator(
                hf_weights_files, use_safetensors, quant_state_dict
            ),  # 未量化权重生成器（运行时量化）
            quant_state_dict,
        )

    def _is_8bit_weight_name(self, weight_name: str):
        # 判断是否为 8-bit 量化权重的名称
        quantized_suffix = {".scb", ".weight_format"}
        return any(weight_name.lower().endswith(suffix) for suffix in quantized_suffix)

    def _is_4bit_weight_name(self, weight_name: str):
        # 判断是否为 4-bit 量化权重的名称
        quantized_suffix = {
            "absmax",
            "quant_map",
            "nested_absmax",
            "nested_quant_map",
            "bitsandbytes",
        }
        suffix = weight_name.split(".")[-1]
        return any(q_suffix in suffix for q_suffix in quantized_suffix)

    def _quantized_8bit_generator(
        self, hf_weights_files, use_safetensors, quant_state_dict
    ) -> Generator:
        # 8-bit 预量化权重生成器：先提取量化状态，再生成权重
        for weight_name, weight_tensor in self._hf_weight_iter(
            hf_weights_files, use_safetensors
        ):
            if not weight_name.lower().endswith(".scb"):
                continue  # 只处理 .scb 量化状态文件

            weight_key = weight_name.lower().replace(".scb", ".weight")
            quant_state_dict[weight_key] = weight_tensor  # 记录 8-bit 量化状态

        for weight_name, weight_tensor in self._hf_weight_iter(
            hf_weights_files, use_safetensors
        ):
            if self._is_8bit_weight_name(weight_name):
                continue  # 跳过量化状态文件

            if weight_name in quant_state_dict:
                set_weight_attrs(weight_tensor, {"load_in_8bit": True})  # 标记为 8-bit 加载
                yield weight_name, weight_tensor
            else:
                yield weight_name, weight_tensor

    def _quantized_4bit_generator(
        self, hf_weights_files, use_safetensors, quant_state_dict
    ) -> Generator:
        # 4-bit 预量化权重生成器：先收集量化状态，再生成权重
        from bitsandbytes.functional import QuantState

        # First iterate over all quant state weights
        # 首先遍历所有量化状态权重
        weight_iterator = self._hf_weight_iter(hf_weights_files, use_safetensors)
        temp_state_dict = {}
        for weight_name, weight_tensor in weight_iterator:
            if not self._is_4bit_weight_name(weight_name):
                continue
            # bitsandbytes library requires
            # weight.quant_state.bitsandbytes__* in CPU
            # bitsandbytes 库要求 quant_state 中的 bitsandbytes__* 在 CPU 上
            if "quant_state.bitsandbytes" in weight_name:
                temp_state_dict[weight_name] = weight_tensor.cpu().data
            else:
                temp_state_dict[weight_name] = weight_tensor

        # Closure to parse quant_state for each prequant weight
        # 解析每个预量化权重的量化状态的闭包
        def _parse_quant_state(param_name: str, temp_state_dict: Dict) -> QuantState:
            quant_state = {}
            for k in temp_state_dict:
                if param_name + "." in k:
                    quant_state[k] = temp_state_dict[k]

            return QuantState.from_dict(quant_state, device="cuda")

        # Second iterate over all prequant and normal weights
        # pre quantized weights would have a quant_state
        # 第二次遍历所有预量化和普通权重
        for weight_name, weight_tensor in self._hf_weight_iter(
            hf_weights_files, use_safetensors
        ):

            if self._is_4bit_weight_name(weight_name):
                continue  # 跳过量化状态权重

            if (f"{weight_name}.quant_state.bitsandbytes__nf4" in temp_state_dict) or (
                f"{weight_name}.quant_state.bitsandbytes__fp4" in temp_state_dict
            ):
                quant_state = _parse_quant_state(weight_name, temp_state_dict)
                quant_state_dict[weight_name] = quant_state  # 记录 4-bit 量化状态
                yield weight_name, weight_tensor
            else:
                yield weight_name, weight_tensor

    def _unquantized_generator(
        self, hf_weights_files, use_safetensors, quant_state_dict
    ) -> Generator:
        # 未量化权重生成器：对目标模块进行运行时 4-bit 量化
        from bitsandbytes.functional import quantize_4bit

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        for weight_name, weight_tensor in self._hf_weight_iter(
            hf_weights_files, use_safetensors
        ):

            if any(
                target_module in weight_name for target_module in self.target_modules
            ) and weight_name.endswith(".weight"):
                weight_name = weight_name.replace(".weight", ".qweight")  # 重命名为 qweight

                if any(
                    module in weight_name
                    for module in self.column_parallel_weights_modules
                ):
                    # 列并行权重：在最后一个维度上分片
                    total_size = weight_tensor.size(-1)
                    start_index = total_size // tp_size * tp_rank
                    end_index = total_size // tp_size * (tp_rank + 1)
                    weight_sub_tensor = weight_tensor[..., start_index:end_index]

                else:
                    # 行并行权重：在第一个维度上分片
                    total_size = weight_tensor.size(0)
                    start_index = total_size // tp_size * tp_rank
                    end_index = total_size // tp_size * (tp_rank + 1)
                    weight_sub_tensor = weight_tensor[start_index:end_index, ...]

                # bitsandbytes requires data in GPU
                # bitsandbytes 要求数据在 GPU 上
                if weight_sub_tensor.is_cuda:
                    loaded_weight = weight_sub_tensor
                else:
                    loaded_weight = weight_sub_tensor.cuda()

                # remove the following after the issue is fixed:
                # https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1342
                # 此问题修复后移除以下代码
                if loaded_weight.is_contiguous() is False:
                    loaded_weight = loaded_weight.contiguous()

                with set_default_torch_dtype(torch.float32):
                    processed_weight, quant_state = quantize_4bit(
                        loaded_weight, compress_statistics=True, quant_type="nf4"
                    )  # 执行 4-bit NF4 量化

                quant_state_dict[weight_name] = quant_state
            else:
                processed_weight = weight_tensor  # 非目标模块直接使用原始权重

            yield weight_name, processed_weight

    def _load_weights(self, model_config: ModelConfig, model: nn.Module) -> None:
        # 加载 BitsAndBytes 量化权重并设置量化状态属性
        if not hasattr(model, "load_weights"):
            raise AttributeError(
                "The required method 'load_weights' is not defined in class"
                f" {type(model).__name__}."
            )

        if not hasattr(model, "bitsandbytes_stacked_params_mapping"):
            raise AttributeError(
                f"Model {type(model).__name__} does not support BitsAndBytes "
                "quantization yet."
            )  # 模型不支持 BitsAndBytes 量化

        if len(self.target_modules) == 0:
            if hasattr(model, "default_bitsandbytes_target_modules"):
                self.target_modules = model.default_bitsandbytes_target_modules
            else:
                self.target_modules = self.default_target_modules  # 使用默认目标模块

        if hasattr(model, "column_parallel_weights_modules"):
            self.column_parallel_weights_modules = model.column_parallel_weights_modules
        else:
            self.column_parallel_weights_modules = []

        self.model_type = type(model).__name__

        logger.info(
            "Loading weights with BitsAndBytes quantization. " " May take a while ..."
        )

        quant_config = getattr(model_config.hf_config, "quantization_config", None)

        pre_quant = False
        if quant_config is not None:
            quant_method = quant_config.get("quant_method")
            if quant_method == "bitsandbytes":
                pre_quant = True
            else:
                raise ValueError(
                    f"BitsAndBytes loader does not support {quant_method} "
                    "quantization"
                )

        # The quant_states in pre_quantized models cannot work with a split
        # weight tensor. So TP does not work with pre_quantized bnb models.
        # 预量化模型的 quant_states 不能与分割的权重张量配合使用，因此 TP 不支持预量化 BNB 模型
        if pre_quant and get_tensor_model_parallel_world_size() > 1:
            raise ValueError(
                "Prequant BitsAndBytes models with TP is not supported."
                "Please try with PP."
            )

        load_8bit = False
        if pre_quant:
            load_8bit = quant_config.get("load_in_8bit", False)

        qweight_iterator, quant_state_dict = self._get_quantized_weights_iterator(
            model_config.model_path, model_config.revision, pre_quant, load_8bit
        )

        model.load_weights(qweight_iterator)

        current_platform.empty_cache()  # 清理 GPU 缓存

        param_dict = dict(model.named_parameters())
        stacked_quant_state_dict: Dict[str, Dict[int, Any]] = {}
        model_type = model_config.hf_config.model_type
        for quant_param_name in quant_state_dict:
            non_stacked_param_name = quant_param_name
            if model_type == "mllama" and "vision_model" in quant_param_name:
                # adapt to VisionAttention
                # 适配 VisionAttention
                quant_param_name = quant_param_name.replace(
                    "self_attn.o_proj", "self_attn.proj"
                )
            shard_index = 0
            for shard_name, (
                weight_name,
                index,
            ) in model.bitsandbytes_stacked_params_mapping.items():
                if (
                    model_type in ["qwen2_vl", "qwen2_5_vl"]
                    and "visual" in quant_param_name
                ):
                    break  # Qwen2 VL 视觉模型特殊处理
                if shard_name in quant_param_name:
                    shard_index = index
                    quant_param_name = quant_param_name.replace(shard_name, weight_name)
                    break  # 替换分片名称为合并后的名称

            if (
                model_type in ["qwen2_vl", "qwen2_5_vl"]
                and "visual" in quant_param_name
            ):
                quant_param_name = quant_param_name.replace(
                    r"attn.qkv.", r"attn.qkv_proj."
                )

            if quant_param_name not in param_dict:
                raise ValueError(
                    f"Parameter {quant_param_name} not found in the model."
                )  # 参数未找到

            if quant_param_name not in stacked_quant_state_dict:
                stacked_quant_state_dict[quant_param_name] = {}

            stacked_quant_state_dict[quant_param_name][shard_index] = quant_state_dict[
                non_stacked_param_name
            ]  # 按分片索引存储量化状态

        # save quant_states and offsets as the attributes of the parameters
        # 将量化状态和偏移量保存为参数的属性
        for param_name, param in param_dict.items():
            if param_name in stacked_quant_state_dict:
                quant_states = stacked_quant_state_dict[param_name]
                set_weight_attrs(param, {"bnb_quant_state": quant_states})  # 设置量化状态属性

                pack_ratio = getattr(param, "pack_factor", -1)
                if pack_ratio == -1:
                    raise ValueError(f"pack_factor not set for parameter {param_name}.")

                num_elements = [0] * len(quant_states)
                for seq, quant_state in quant_states.items():
                    num_elements[seq] = math.prod(quant_state.shape) // pack_ratio  # 计算每个分片的元素数

                offsets = np.concatenate(([0], np.cumsum(num_elements)))  # 计算分片偏移量
                # Make torch infer_schema happy(Compatible with vLLM)
                # 使 torch 的 infer_schema 兼容（与 vLLM 兼容）
                offsets = torch.tensor(offsets).cpu()
                set_weight_attrs(param, {"bnb_shard_offsets": offsets})  # 设置分片偏移量属性

                if load_8bit:
                    set_weight_attrs(
                        param, {"matmul_state": [None] * len(quant_states)}
                    )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model_path, model_config.revision)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

                self._load_weights(model_config, model)

        return model.eval()


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """
    # GGUF 模型加载器：可以加载 GGUF 格式的文件，支持加载完整模型和分片模型

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def _prepare_weights(self, model_name_or_path: str):
        # 准备 GGUF 权重文件：必须是本地文件
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        else:
            raise ValueError(f"{model_name_or_path} is not a file.")

    def _get_gguf_weights_map(self, model_config: ModelConfig):
        """
        GGUF uses this naming convention for their tensors from HF checkpoint:
        `blk.N.BB.weight` and `blk.N.BB.bias`
        where N signifies the block number of a layer, and BB signifies the
        attention/mlp layer components.
        See "Standardized tensor names" in
        https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.
        """
        # 获取 GGUF 权重名称到 HuggingFace 权重名称的映射
        # GGUF 使用 `blk.N.BB.weight` 命名约定

        # only load the gguf module when needed
        # 仅在需要时加载 gguf 模块
        try:
            import gguf

            # FIXME: add version check for gguf
            # FIXME: 添加 gguf 版本检查
        except ImportError as err:
            raise ImportError(
                "Please install gguf via `pip install gguf` to use gguf quantizer."
            ) from err

        config = model_config.hf_config
        model_type = config.model_type
        # hack: ggufs have a different name than transformers
        # 临时处理：gguf 使用与 transformers 不同的模型类型名称
        if model_type == "cohere":
            model_type = "command-r"
        elif model_type == "qwen3_moe":
            model_type = "qwen3moe"
        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break  # 查找 GGUF 架构名称
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")
        num_layers = config.num_hidden_layers
        name_map = gguf.get_tensor_name_map(arch, num_layers)  # 获取 GGUF 张量名称映射器
        with torch.device("meta"):
            dummy_model = AutoModelForCausalLM.from_config(config)
        state_dict = dummy_model.state_dict()

        gguf_to_hf_name_map = {}
        for hf_name in state_dict:
            name, suffix = hf_name.rsplit(".", 1)
            gguf_name = name_map.get_name(name)
            gguf_to_hf_name_map[f"{gguf_name}.{suffix}"] = hf_name  # 构建 GGUF 到 HF 的名称映射
        return gguf_to_hf_name_map

    def _get_weights_iterator(
        self, model_name_or_path: str, gguf_to_hf_name_map: Dict[str, str]
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        # 获取 GGUF 权重迭代器
        return gguf_quant_weights_iterator(model_name_or_path, gguf_to_hf_name_map)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model_path)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        # 加载 GGUF 格式模型
        local_model_path = self._prepare_weights(model_config.model_path)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        # we can only know if tie word embeddings after mapping weights
        # 映射权重后才能确定是否共享词嵌入
        if "lm_head.weight" in get_gguf_extra_tensor_names(
            local_model_path, gguf_weights_map
        ):
            model_config.hf_config.update({"tie_word_embeddings": True})  # 设置共享词嵌入

        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(model_config, self.load_config, quant_config)
            model.load_weights(
                self._get_weights_iterator(local_model_path, gguf_weights_map)
            )

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
        return model


class RemoteInstanceModelLoader(BaseModelLoader):
    """Model loader that can load Tensors from remote sglang instance."""
    # 远程实例模型加载器：从远程 SGLang 实例加载张量

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )
        self.remote_instance_transfer_engine_weight_info = None  # 远程传输引擎权重信息

    def download_model(self, model_config: ModelConfig) -> None:
        raise NotImplementedError  # 远程实例加载器不支持下载模型

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        # 从远程实例加载模型，支持 NCCL、TransferEngine 和 ModelExpress 三种后端
        logger.info("Loading weights from remote instance ...")
        load_config = self.load_config

        assert load_config.load_format == LoadFormat.REMOTE_INSTANCE, (
            f"Model loader {self.load_config.load_format} is not supported for "
            f"load format {load_config.load_format}"
        )

        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, quant_config)

        if (
            load_config.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            model_weights = f"instance://{load_config.remote_instance_weight_loader_seed_instance_ip}:{load_config.remote_instance_weight_loader_send_weights_group_ports[load_config.tp_rank]}"
            with create_remote_connector(model_weights, device_config.device) as client:
                connector_type = get_connector_type(client)
                if connector_type == ConnectorType.INSTANCE:
                    self.load_model_from_remote_instance_by_nccl(
                        model, client, model_config, device_config
                    )
                else:
                    raise ValueError(
                        f"Unsupported connector type {connector_type} for "
                        f"remote tensor model loading."
                    )
        elif (
            load_config.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE
        ):
            if load_config.remote_instance_weight_loader_transfer_engine is None:
                raise RuntimeError(
                    "Transfer engine is not initialized for remote instance "
                    "model loader with `transfer_engine` backend. "
                )
            logger.info(
                "TransferEngine registering memory regions (this may take a few seconds)..."
            )
            # register memory region
            self.remote_instance_transfer_engine_weight_info = register_memory_region(
                model, load_config.remote_instance_weight_loader_transfer_engine
            )
            logger.info(
                "TransferEngine memory regions have been successfully registered."
            )

            # transfer weights
            success = self.load_model_from_remote_instance_by_transfer_engine(
                model,
                load_config.remote_instance_weight_loader_transfer_engine,
                f"http://{load_config.remote_instance_weight_loader_seed_instance_ip}:{load_config.remote_instance_weight_loader_seed_instance_service_port}",
                load_config.tp_rank,
            )
            if not success:
                raise RuntimeError(
                    "Failed to load weights from remote instance via transfer engine."
                )
        elif (
            load_config.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.MODELEXPRESS
        ):
            try:
                from modelexpress.engines.sglang.loader import MxModelLoader
            except ImportError as exc:
                raise ImportError(
                    "ModelExpress support requires the 'modelexpress' "
                    "package. Install it in the SGLang image."
                ) from exc

            model = MxModelLoader(load_config).load_model(
                model=model,
                model_config=model_config,
                device_config=device_config,
            )
        else:
            raise ValueError("Invalid remote instance weight loader backend.")

        return model.eval()

    def load_model_from_remote_instance_by_nccl(
        self, model, client, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        # 通过 NCCL 从远程实例加载模型权重
        load_config = self.load_config
        instance_ip = socket.gethostbyname(socket.gethostname())  # 获取本机 IP
        start_build_group_tic = time.time()
        client.build_group(
            gpu_id=device_config.gpu_id,
            tp_rank=load_config.tp_rank,
            instance_ip=instance_ip,
        )
        current_platform.synchronize()
        end_build_group_tic = time.time()
        logger.debug(
            f"finish building group for remote instance, time used: {(end_build_group_tic - start_build_group_tic):.4f}s"
        )

        if load_config.tp_rank == 0:
            t = threading.Thread(
                target=trigger_transferring_weights_request,
                args=(
                    load_config.remote_instance_weight_loader_seed_instance_ip,
                    load_config.remote_instance_weight_loader_seed_instance_service_port,
                    load_config.remote_instance_weight_loader_send_weights_group_ports,
                    instance_ip,
                ),
            )
            t.start()  # rank 0 触发远程权重传输请求

        start_get_weights_tic = time.time()
        with set_default_torch_dtype(model_config.dtype):
            for _, tensor in model.named_parameters():
                torch.distributed.broadcast(
                    tensor.data,
                    src=0,
                    group=client._model_update_group,
                )  # 通过 NCCL 广播权重
            current_platform.synchronize()

            _post_load_weights(model)
        end_get_weights_tic = time.time()
        logger.debug(
            f"finish getting all weights from remote instance, time used: {(end_get_weights_tic - start_get_weights_tic):.4f}s"
        )
        # destroy the process group after loading weights
        # 加载权重后销毁进程组
        torch.distributed.distributed_c10d.destroy_process_group(
            client._model_update_group
        )
        current_platform.empty_cache()  # 清理缓存

    def load_model_from_remote_instance_by_transfer_engine(
        self, model, transfer_engine, seed_url, tp_rank
    ) -> bool:
        # 通过 TransferEngine 从远程实例加载模型权重（RDMA 方式）
        # get remote weights metadata from source instance
        # 从源实例获取远程权重元数据
        seed_transfer_engine_session_id, seed_transfer_engine_weight_info = (
            get_remote_instance_transfer_engine_info_per_rank(seed_url, tp_rank)
        )
        if (
            seed_transfer_engine_session_id is None
            or seed_transfer_engine_weight_info is None
        ):
            logger.error("Cannot get transfer engine session or weight info.")
            return False

        # prepare local/remote RDMA keys
        # 准备本地/远程 RDMA 密钥
        seed_ptr_list = []
        client_ptr_list = []
        client_len_list = []
        for name, tensor in model.named_parameters():
            weight_info = seed_transfer_engine_weight_info.get(name, None)
            if weight_info is None:
                logger.error(f"Cannot find weight info for {name}.")
                return False

            seed_ptr, seed_numel, seed_element_size = weight_info
            if (
                seed_numel != tensor.numel()
                or seed_element_size != tensor.element_size()
            ):
                logger.error(
                    f"Weight info does not match for {name}, "
                    f"expected ({seed_numel}, {seed_element_size}), "
                    f"got ({tensor.numel()}, {tensor.element_size()})"
                )  # 权重信息不匹配
                return False
            client_ptr = tensor.data_ptr()
            client_len = tensor.numel() * tensor.element_size()
            seed_ptr_list.append(seed_ptr)
            client_ptr_list.append(client_ptr)
            client_len_list.append(client_len)

        # load weights from source instance through TransferEngine
        # 通过 TransferEngine 从源实例加载权重
        ret = transfer_engine.batch_transfer_sync_read(
            seed_transfer_engine_session_id,
            client_ptr_list,
            seed_ptr_list,
            client_len_list,
        )
        if ret < 0:
            logger.error(f"batch transfer failed, error: {ret}")
            return False

        _post_load_weights(model)

        return True


class RemoteModelLoader(BaseModelLoader):
    """Model loader that can load Tensors from remote database."""
    # 远程存储模型加载器：从远程数据库（KV 存储）或文件系统加载模型权重

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        # TODO @DellCurry: move to s3 connector only
        set_runai_streamer_env(load_config)

    def _get_weights_iterator_kv(
        self,
        client,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights from remote storage."""
        # 从远程 KV 存储获取模型权重迭代器
        assert get_connector_type(client) == ConnectorType.KV
        rank = get_tensor_model_parallel_rank()
        return client.weight_iterator(rank)

    def _get_weights_iterator_fs(
        self,
        client,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights from remote storage."""
        # 从远程文件系统获取模型权重迭代器
        assert get_connector_type(client) == ConnectorType.FS
        return client.weight_iterator()

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        model_path: str,
        url: str,
    ) -> None:
        # 将模型保存到远程 KV 存储
        with create_remote_connector(url) as client:
            assert get_connector_type(client) == ConnectorType.KV
            model_name = parse_model_name(url)
            rank = get_tensor_model_parallel_rank()
            state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
            for key, tensor in state_dict.items():
                r_key = f"{model_name}/keys/rank_{rank}/{key}"
                client.set(r_key, tensor)  # 将张量写入远程 KV 存储

            for root, _, files in os.walk(model_path):
                for file_name in files:
                    # ignore hidden files
                    if file_name.startswith("."):
                        continue
                    if os.path.splitext(file_name)[1] in (".json", ".py"):
                        file_path = os.path.join(root, file_name)
                        with open(file_path, encoding="utf-8") as file:
                            file_content = file.read()
                            f_key = f"{model_name}/files/{file_name}"
                            client.setstr(f_key, file_content)

    def _load_model_from_remote_kv(
        self, model: nn.Module, model_config: ModelConfig, client
    ):
        # 从远程 KV 存储加载模型权重
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module)
        weights_iterator = self._get_weights_iterator_kv(client)
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        for key, tensor in weights_iterator:
            # If loading with LoRA enabled, additional padding may
            # be added to certain parameters. We only load into a
            # narrowed view of the parameter data.
            param_data = state_dict[key].data
            param_shape = state_dict[key].shape
            for dim, size in enumerate(tensor.shape):
                if size < param_shape[dim]:
                    param_data = param_data.narrow(dim, 0, size)
            if tensor.shape != param_shape:
                logger.warning(
                    "loading tensor of shape %s into " "parameter '%s' of shape %s",
                    tensor.shape,
                    key,
                    param_shape,
                )
            param_data.copy_(tensor)
            state_dict.pop(key)
        if state_dict:
            raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")

        _post_load_weights(model)

    def _load_model_from_remote_fs(
        self, model, client, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        # 从远程文件系统加载模型权重
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            model.load_weights(self._get_weights_iterator_fs(client))

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    # 量化方法在加载后需要处理权重时，期望参数在全局目标设备上。
                    # 此上下文用于 CPU 卸载场景，将参数移至设备处理后再移回。
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        logger.info("Loading weights from remote storage ...")
        start = time.perf_counter()
        load_config = self.load_config

        assert load_config.load_format == LoadFormat.REMOTE, (
            f"Model loader {self.load_config.load_format} is not supported for "
            f"load format {load_config.load_format}"
        )

        model_weights = model_config.model_path
        if hasattr(model_config, "model_weights"):
            model_weights = model_config.model_weights

        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, quant_config)

            with create_remote_connector(
                model_weights, device=device_config.device
            ) as client:
                connector_type = get_connector_type(client)
                if connector_type == ConnectorType.KV:
                    self._load_model_from_remote_kv(model, model_config, client)
                elif connector_type == ConnectorType.FS:
                    self._load_model_from_remote_fs(
                        model, client, model_config, device_config
                    )

        end = time.perf_counter()
        logger.info("Loaded weights from remote storage in %.2f seconds.", end - start)
        return model.eval()


def load_model_with_cpu_quantization(
    self,
    *,
    model_config: ModelConfig,
    device_config: DeviceConfig,
) -> nn.Module:
    # 使用 CPU 量化加载模型：在 CPU 上加载权重并量化，然后移至目标设备
    target_device = torch.device(device_config.device)
    quant_config = _get_quantization_config(model_config, self.load_config)
    with set_default_torch_dtype(model_config.dtype):
        model = _initialize_model(
            model_config,
            self.load_config,
            quant_config,
        )

        if not isinstance(self, DummyModelLoader):
            model.load_weights(self._get_all_weights(model_config, model))  # 加载权重

        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                # When quant methods need to process weights after loading
                # (for repacking, quantizing, etc), they expect parameters
                # to be on the global target device. This scope is for the
                # case where cpu offloading is used, where we will move the
                # parameters onto device for processing and back off after.
                # 量化方法在加载后需要处理权重时，期望参数在全局目标设备上。
                # 此上下文用于 CPU 卸载场景，将参数移至设备处理后再移回。
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        model.to(target_device)  # 将模型移至目标设备

    return model.eval()


class ModelOptModelLoader(DefaultModelLoader):
    """
    Model loader that applies NVIDIA Model Optimizer quantization
    """
    # NVIDIA ModelOpt 量化模型加载器：应用 NVIDIA Model Optimizer 量化

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        # Any ModelOpt specific initialization if needed
        # 如果需要，进行 ModelOpt 特定的初始化

    def _setup_modelopt_quantization(
        self,
        model,
        tokenizer,
        quant_cfg,
        quantized_ckpt_restore_path: str | None = None,
        quantized_ckpt_save_path: str | None = None,
        export_path: str | None = None,
    ) -> None:
        """
        Set up ModelOpt quantization for the given model.
        # 为给定模型设置 ModelOpt 量化

        Args:
            model: The model to quantize / 要量化的模型
            tokenizer: The tokenizer associated with the model / 模型关联的分词器
            quant_cfg: The quantization configuration / 量化配置
            quantized_ckpt_restore_path: Path to restore quantized checkpoint from / 恢复量化检查点的路径
            quantized_ckpt_save_path: Path to save quantized checkpoint to / 保存量化检查点的路径
            export_path: Path to export the quantized model in HuggingFace format / 导出量化模型的 HuggingFace 格式路径

        Raises:
            ImportError: If ModelOpt is not available / 如果 ModelOpt 不可用
            Exception: If quantization setup fails / 如果量化设置失败
        """
        try:
            import modelopt.torch.opt as mto
            import modelopt.torch.quantization as mtq
            from modelopt.torch.quantization.utils import is_quantized
        except ImportError as e:
            raise ImportError(
                "ModelOpt is not available. Please install modelopt."
            ) from e

        if is_quantized(model):
            rank0_log("Model is already quantized, skipping quantization setup.")
            return  # 模型已量化，跳过量化设置
        # Restore from checkpoint if provided
        # 如果提供了检查点路径，从检查点恢复
        if quantized_ckpt_restore_path:
            try:
                mto.restore(model, quantized_ckpt_restore_path)  # 从检查点恢复量化模型
                rank0_log(
                    f"Restored quantized model from {quantized_ckpt_restore_path}"
                )

                # Export model if path provided (even when restoring from checkpoint)
                # 即使从检查点恢复，如果提供了导出路径也导出模型
                self._maybe_export_modelopt(model, export_path)
                return
            except Exception as e:
                logger.warning(
                    f"Failed to restore from {quantized_ckpt_restore_path}: {e}"
                )
                rank0_log("Proceeding with calibration-based quantization...")

        # Set up calibration-based quantization
        # 设置基于校准的量化
        try:
            # Left padding tends to work better for batched generation with decoder-only LMs
            # 左填充通常更适合 decoder-only 语言模型的批量生成
            with suppress(Exception):
                tokenizer.padding_side = "left"

            from modelopt.torch.utils.dataset_utils import (
                create_forward_loop,
                get_dataset_dataloader,
            )

            # Create calibration dataloader
            # 创建校准数据加载器
            calib_dataloader = get_dataset_dataloader(
                dataset_name="cnn_dailymail",  # TODO: Consider making this configurable / 考虑使其可配置
                tokenizer=tokenizer,
                batch_size=36,  # TODO: Consider making this configurable
                num_samples=512,  # TODO: Consider making this configurable
                device=model.device,
                include_labels=False,
            )

            calibrate_loop = create_forward_loop(dataloader=calib_dataloader)

            # Apply quantization
            # 应用量化
            mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

            if (
                not model_parallel_is_initialized()
                or get_tensor_model_parallel_rank() == 0
            ):
                mtq.print_quant_summary(model)

            # Save checkpoint if path provided
            if quantized_ckpt_save_path:
                try:
                    mto.save(model, quantized_ckpt_save_path)
                    rank0_log(f"Quantized model saved to {quantized_ckpt_save_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to save quantized checkpoint to {quantized_ckpt_save_path}: {e}"
                    )

            # Export model if path provided
            self._maybe_export_modelopt(model, export_path)

        except Exception as e:
            raise Exception(f"Failed to set up ModelOpt quantization: {e}") from e

    def _maybe_export_modelopt(self, model, export_path: str | None) -> None:
        """Export model to HuggingFace format if export_path is provided."""
        # 如果提供了导出路径，将模型导出为 HuggingFace 格式
        if export_path:
            try:
                # Get the original model path from the model config
                original_model_path = getattr(self, "_original_model_path", None)
                self._export_modelopt_checkpoint(
                    model, export_path, original_model_path
                )
                rank0_log(
                    f"Quantized model exported to HuggingFace format at {export_path}"
                )
            except Exception as e:
                rank0_log(
                    f"Warning: Failed to export quantized model to {export_path}: {e}"
                )

    def _export_modelopt_checkpoint(
        self,
        model,
        export_path: str,
        model_path: str = None,
        trust_remote_code: bool = True,
    ) -> None:
        """
        Export the quantized model to HuggingFace format using ModelOpt export API.
        # 使用 ModelOpt 导出 API 将量化模型导出为 HuggingFace 格式

        Args:
            model: The quantized model to export / 要导出的量化模型
            export_path: Directory path to export the model to / 导出目录路径
            model_path: Path to the original model (for tokenizer export) / 原始模型路径（用于分词器导出）
            trust_remote_code: Whether to trust remote code for tokenizer loading / 是否信任远程代码

        Raises:
            ImportError: If ModelOpt export functionality is not available / 如果 ModelOpt 导出功能不可用
            Exception: If export fails / 如果导出失败
        """
        try:
            from modelopt.torch.export import export_hf_checkpoint
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "ModelOpt export functionality is not available. "
                "Please ensure you have the latest version of modelopt installed."
            ) from e

        # Create export directory if it doesn't exist
        # 创建导出目录（如果不存在）
        os.makedirs(export_path, exist_ok=True)

        # Export the quantized model
        # 导出量化模型
        export_hf_checkpoint(model, export_dir=export_path)

        # Export the tokenizer if model_path is provided
        # 如果提供了模型路径，导出分词器
        if model_path:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code
                )
                tokenizer.save_pretrained(export_path)
                rank0_log(f"Tokenizer exported to {export_path}")
            except Exception as e:
                rank0_log(f"Warning: Failed to export tokenizer: {e}")

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        # ModelOpt 模型加载：根据模型是否已量化选择加载路径
        logger.info("ModelOptModelLoader: Loading base model...")

        # Store the original model path for tokenizer export
        # 保存原始模型路径，用于分词器导出
        self._original_model_path = model_config.model_path

        # Check if model is already quantized
        # 检查模型是否已量化
        if model_config._is_already_quantized():
            logger.info("Model is already quantized, loading directly...")
            # Use default loading for pre-quantized models
            # 对预量化模型使用默认加载
            return super().load_model(
                model_config=model_config, device_config=device_config
            )

        # TODO: Quantize-and-serve mode has been disabled at the ModelConfig level
        # All quantization now uses the standard workflow (quantize + export/save)
        # TODO: 量化即服务模式已在 ModelConfig 层面禁用
        # 所有量化现在使用标准工作流（量化 + 导出/保存）
        logger.info("Standard quantization mode: Will quantize and export/save")
        return self._standard_quantization_workflow(model_config, device_config)

    def _standard_quantization_workflow(
        self, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        """Standard quantization workflow: quantize, save checkpoint, export, then return model."""
        # 标准量化工作流：量化、保存检查点、导出、然后返回模型
        # Use shared method from parent class to load base model for quantization
        # 使用父类的共享方法加载用于量化的基础模型
        model = self._load_modelopt_base_model(model_config)

        # Import ModelOpt modules
        # 导入 ModelOpt 模块
        try:
            import modelopt.torch.quantization as mtq
        except ImportError:
            logger.error(
                "NVIDIA Model Optimizer (modelopt) library not found. "
                "Please install it to use ModelOpt quantization."
            )
            raise

        # Handle both old modelopt_quant and new unified quantization flags
        # 处理旧版 modelopt_quant 和新版统一量化标志
        if hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant:
            # Legacy modelopt_quant flag
            # 旧版 modelopt_quant 标志
            quant_choice_str = model_config.modelopt_quant
        else:
            # Unified quantization flag - extract the type (fp8/fp4)
            # 统一量化标志 - 提取量化类型（fp8/fp4）
            quant_choice_str = model_config._get_modelopt_quant_type()

        quant_cfg_name = QUANT_CFG_CHOICES.get(quant_choice_str)
        if not quant_cfg_name:
            raise ValueError(
                f"Invalid quantization choice: '{quant_choice_str}'. "
                f"Available choices: {list(QUANT_CFG_CHOICES.keys())}"
            )

        try:
            # getattr will fetch the config object, e.g., mtq.FP8_DEFAULT_CFG
            quant_cfg = getattr(mtq, quant_cfg_name)
        except AttributeError:
            raise AttributeError(
                f"ModelOpt quantization config '{quant_cfg_name}' not found. "
                "Please verify the ModelOpt library installation."
            )

        logger.info(
            f"Quantizing model with ModelOpt using config: mtq.{quant_cfg_name}"
        )

        # Get ModelOpt configuration from LoadConfig
        modelopt_config = self.load_config.modelopt_config
        quantized_ckpt_restore_path = (
            modelopt_config.checkpoint_restore_path if modelopt_config else None
        )
        quantized_ckpt_save_path = (
            modelopt_config.checkpoint_save_path if modelopt_config else None
        )
        export_path = modelopt_config.export_path if modelopt_config else None
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_path, use_fast=True
        )

        try:
            self._setup_modelopt_quantization(
                model,
                tokenizer,
                quant_cfg,
                quantized_ckpt_restore_path=quantized_ckpt_restore_path,
                quantized_ckpt_save_path=quantized_ckpt_save_path,
                export_path=export_path,
            )
        except Exception as e:
            logger.warning(f"ModelOpt quantization failed: {e}")
            rank0_log("Proceeding without quantization...")

        return model.eval()


class RunaiModelStreamerLoader(BaseModelLoader):
    """
    Model loader that uses Runai Model Streamer to load a model.

    Supports fast model loading from SSDs, shared filesystems and object storage (S3, GCS, Azure blob) with weight streaming.
    # 使用 Runai Model Streamer 加载模型的加载器。
    # 支持从 SSD、共享文件系统和对象存储（S3、GCS、Azure blob）快速加载模型，支持权重流式传输。

    Configuration (via load_config.model_loader_extra_config):
        - distributed (bool): Enable distributed streaming - True by default for url paths (object storage)
        - concurrency (int): Number of concurrent downloads
        - memory_limit (int): Memory limit for streaming buffer

    Note: Metadata files must be pre-downloaded via
    ObjectStorageModel.download_and_get_path() before instantiation.
    # 注意：元数据文件必须在实例化前通过 ObjectStorageModel.download_and_get_path() 预下载。
    """

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: Optional[str]
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        model_config: Optional["ModelConfig"] = None
        """The model configuration (for checking architecture, etc)."""

        @classmethod
        def init_new(cls, model_config: ModelConfig, model):
            model_weights = model_config.model_path
            if hasattr(model_config, "model_weights"):
                model_weights = model_config.model_weights
            return cls(
                model_weights,
                model_config.revision,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                model_config=model_config,
            )

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"distributed", "concurrency", "memory_limit"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

        set_runai_streamer_env(load_config)

        self._is_distributed = None
        if load_config.model_loader_extra_config:
            extra_config = load_config.model_loader_extra_config

            if "distributed" in extra_config and isinstance(
                extra_config.get("distributed"), bool
            ):
                self._is_distributed = extra_config.get("distributed")

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str]
    ) -> Tuple[str, List[str]]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        from sglang.srt.utils.runai_utils import is_runai_obj_uri, list_safetensors

        is_object_storage_path = is_runai_obj_uri(model_name_or_path)
        if self._is_distributed is None:
            self._is_distributed = is_object_storage_path
        is_local = os.path.isdir(model_name_or_path)
        safetensors_pattern = "*.safetensors"
        index_file = SAFE_WEIGHTS_INDEX_NAME

        hf_folder = (
            model_name_or_path
            if (is_local or is_object_storage_path)
            else download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                [safetensors_pattern],
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        )

        server_args = get_global_server_args()
        if server_args and server_args.model_checksum is not None:
            from sglang.srt.utils.model_file_verifier import verify

            checksums_source = server_args.model_checksum or model_name_or_path
            verify(model_path=hf_folder, checksums_source=checksums_source)

        hf_weights_files = list_safetensors(path=hf_folder)

        # For models like Mistral-7B-Instruct-v0.3
        # there are both sharded safetensors files and a consolidated
        # safetensors file. Using both breaks.
        # Here, we download the `model.safetensors.index.json` and filter
        # any files not found in the index.
        if not is_local and not is_object_storage_path:
            download_safetensors_index_file_from_hf(
                model_name_or_path,
                index_file,
                self.load_config.download_dir,
                revision,
            )
        hf_weights_files = filter_duplicate_safetensors_files(
            hf_weights_files, hf_folder, index_file
        )

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        from sglang.srt.model_loader.weight_utils import (
            runai_safetensors_weights_iterator,
        )

        hf_folder, hf_weights_files = self._prepare_weights(
            source.model_or_path, source.revision
        )

        if source.model_config is not None:
            hf_weights_files = maybe_add_mtp_safetensors(
                hf_weights_files,
                hf_folder,
                "model.safetensors.index.json",
                source.model_config.hf_config,
            )

        weights_iterator = runai_safetensors_weights_iterator(
            hf_weights_files, self._is_distributed, self.target_device_str
        )

        if self.load_config.draft_model_idx is not None:
            import re

            def filter_weights(original_weights_iterator):
                pattern = r"model.mtp.layers.(\d+)."
                for name, tensor in original_weights_iterator:
                    group = re.match(pattern, name)
                    if group is not None:
                        idx = int(group.group(1))
                        if idx != self.load_config.draft_model_idx:
                            continue
                        new_name = name.replace(group.group(), "model.mtp.layers.0.")
                    else:
                        new_name = name
                    yield (new_name, tensor)

            weights_iterator = filter_weights(weights_iterator)

        def apply_prefix(original_weights_iterator):
            yield from (
                (source.prefix + name, tensor)
                for (name, tensor) in original_weights_iterator
            )

        return apply_prefix(weights_iterator)

    def _get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:

        primary_weights = RunaiModelStreamerLoader.Source.init_new(model_config, model)
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[RunaiModelStreamerLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model_path, model_config.revision)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        # 使用 Runai Model Streamer 加载模型
        if hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant:
            # Load base model using shared method
            # Runai 加载器不支持 ModelOpt 量化
            raise NotImplementedError(
                "Runai Model Streamer Loader does not support ModelOpt quantization yet"
            )

        assert device_config.device_type in ("cuda", "cpu"), (
            f"Runai Model Streamer only supports CUDA and CPU, "
            f"got {device_config.device_type}"
        )

        if device_config.device_type == "cuda":
            self.target_device_str = (
                device_config.device_type + ":" + str(device_config.gpu_id)
            )
        else:
            self.target_device_str = "cpu"

        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

            DefaultModelLoader.load_weights_and_postprocess(
                model, self._get_all_weights(model_config, model), target_device
            )

        return model.eval()


def get_model_loader(
    load_config: LoadConfig, model_config: Optional[ModelConfig] = None
) -> BaseModelLoader:
    """Get a model loader based on the load format."""
    # 根据加载格式获取对应的模型加载器实例

    if load_config.load_format == LoadFormat.DUMMY:
        return DummyModelLoader(load_config)

    # ModelOptModelLoader's local-copy quantize-and-export workflow doesn't apply
    # to non-local loaders. These loaders own their weight transport path and still
    # initialize the model with ModelOpt quantization config where applicable.
    # ModelOptModelLoader 的本地量化导出工作流不适用于非本地加载器。
    # 这些加载器拥有自己的权重传输路径，但仍然会使用 ModelOpt 量化配置初始化模型。
    model_optloader_allowed = model_config and load_config.load_format not in (
        LoadFormat.RUNAI_STREAMER,
        LoadFormat.REMOTE_INSTANCE,
    )

    if model_optloader_allowed and (
        (hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant)
        or model_config.quantization
        in ["modelopt_fp8", "modelopt_fp4", "modelopt_mixed", "modelopt"]
    ):
        logger.info("Using ModelOptModelLoader due to ModelOpt quantization config.")
        return ModelOptModelLoader(load_config)

    # Use ModelOptModelLoader for unified quantization flags
    # 使用 ModelOptModelLoader 处理统一量化标志
    if (
        model_optloader_allowed
        and hasattr(model_config, "quantization")
        and model_config.quantization
        in ["modelopt_fp8", "modelopt_fp4", "modelopt_mixed"]
    ):
        if model_config._is_already_quantized():
            logger.info(
                f"Using ModelOptModelLoader for pre-quantized model: {model_config.quantization}"
            )
        else:
            logger.info(
                f"Using ModelOptModelLoader for quantization: {model_config.quantization}"
            )
        return ModelOptModelLoader(load_config)

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)  # 自定义加载格式类型

    if load_config.load_format == LoadFormat.SHARDED_STATE:
        return ShardedStateLoader(load_config)

    if load_config.load_format == LoadFormat.BITSANDBYTES:
        return BitsAndBytesModelLoader(load_config)

    if load_config.load_format == LoadFormat.GGUF:
        return GGUFModelLoader(load_config)

    if load_config.load_format == LoadFormat.LAYERED:
        return LayeredModelLoader(load_config)

    # Check for FLASH_RL format early
    # FP8 approach: BF16/FP16 model with native FP8 quantization
    # 尽早检查 FLASH_RL 格式
    # FP8 方案：BF16/FP16 模型配合原生 FP8 量化
    if load_config.load_format == LoadFormat.FLASH_RL:
        logger.info(
            "Using QuantizedRLModelLoader for RL training with native FP8 quantization."
        )
        logger.info(
            "FP8 approach: Model loads with native SGLang FP8 quantization. "
            "Same model path for both training and inference."
        )

        # Set quantization to FP8 for native SGLang support
        # 为原生 SGLang 支持设置量化为 FP8
        if model_config and not model_config.quantization:
            logger.info(
                "QuantizedRL: Setting quantization to fp8 (native SGLang support). "
                "Model will be loaded with FP8 infrastructure"
            )
            model_config.quantization = "fp8"

        return QuantizedRLModelLoader(load_config)

    if load_config.load_format == LoadFormat.REMOTE:
        return RemoteModelLoader(load_config)

    if load_config.load_format == LoadFormat.REMOTE_INSTANCE:
        return RemoteInstanceModelLoader(load_config)

    if load_config.load_format == LoadFormat.PRIVATE:
        import importlib

        try:
            module = importlib.import_module("sglang.private.private_model_loader")
            return module.PrivateModelLoader(load_config)  # 加载私有模型加载器
        except ImportError:
            raise ValueError("Failed to import sglang.private.private_model_loader")

    if load_config.load_format == LoadFormat.RUNAI_STREAMER:
        return RunaiModelStreamerLoader(load_config)

    return DefaultModelLoader(load_config)  # 默认使用 DefaultModelLoader
