# 本文件定义了推理引擎的抽象基类，提供生成、权重更新和内存控制等统一接口
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch


# 引擎抽象基类，为HTTP引擎和直接引擎提供统一API
class EngineBase(ABC):
    """
    Abstract base class for engine interfaces that support generation, weight updating, and memory control.
    This base class provides a unified API for both HTTP-based engines and engines.
    """

    # 根据输入生成文本输出
    @abstractmethod
    def generate(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        image_data: Optional[Union[List[str], str]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: Optional[bool] = None,
        stream: Optional[bool] = None,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        routed_dp_rank: Optional[int] = None,
        disagg_prefill_dp_rank: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
        rid: Optional[Union[List[str], str]] = None,
        priority: Optional[int] = None,
    ) -> Union[Dict, Iterator[Dict]]:
        """Generate outputs based on given inputs."""
        pass

    # 清空引擎缓存
    @abstractmethod
    def flush_cache(self):
        """Flush the cache of the engine."""
        pass

    # 通过内存中的张量数据更新模型权重
    @abstractmethod
    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update model weights with in-memory tensor data."""
        pass

    # 加载新的LoRA适配器，无需重启引擎
    def load_lora_adapter(self, lora_name: str, lora_path: str):
        """Load a new LoRA adapter without re-launching the engine."""
        pass

    # 卸载LoRA适配器，无需重启引擎
    def unload_lora_adapter(self, lora_name: str):
        """Unload a LoRA adapter without re-launching the engine."""
        pass

    # 临时释放GPU显存占用
    @abstractmethod
    def release_memory_occupation(self):
        """Release GPU memory occupation temporarily."""
        pass

    # 恢复之前释放的GPU显存占用
    @abstractmethod
    def resume_memory_occupation(self):
        """Resume GPU memory occupation which is previously released."""
        pass

    # 关闭引擎并清理资源
    @abstractmethod
    def shutdown(self):
        """Shutdown the engine and clean up resources."""
        pass
