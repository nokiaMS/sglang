# 稀疏注意力工厂模块，提供稀疏算法和后端适配器的创建工厂函数，
# 以及稀疏配置的解析和全局稀疏协调器的注册与获取
import json  # 导入JSON解析模块
import logging  # 导入日志模块
from typing import Optional  # 导入可选类型

import torch  # 导入PyTorch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm  # 导入稀疏算法基类
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_dsa import DeepSeekDSAAlgorithm  # 导入DeepSeek DSA算法
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm  # 导入Quest算法
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (  # 导入后端适配器
    DSABackendAdaptor,  # DSA后端适配器
    FlashAttentionAdaptor,  # FlashAttention后端适配器
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (  # 导入稀疏协调器核心类
    SparseConfig,  # 稀疏配置类
    SparseCoordinator,  # 稀疏协调器类
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

_global_sparse_coordinator: Optional[SparseCoordinator] = None  # 全局稀疏协调器实例，初始为None

_ALGORITHM_REGISTRY = {  # 稀疏算法注册表，映射算法名称到创建函数
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),  # Quest算法的创建lambda
    "deepseek_dsa": lambda config, device, **kw: DeepSeekDSAAlgorithm(  # DeepSeek DSA算法的创建lambda
        config, device, **kw
    ),
}


def _create_sparse_algorithm(  # 创建稀疏算法实例的私有工厂函数
    config: SparseConfig,  # 稀疏配置
    device: torch.device,  # 设备信息
    **kwargs,  # 其他关键字参数
) -> BaseSparseAlgorithm:  # 返回稀疏算法实例
    algorithm_name = config.algorithm.lower()  # 将算法名称转为小写
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)  # 从注册表中查找对应的工厂函数

    if factory is None:  # 如果未找到对应的算法
        raise ValueError(f"Unknown sparse algorithm: {algorithm_name}")  # 抛出值错误异常

    return factory(config, device, **kwargs)  # 调用工厂函数创建并返回算法实例


def _create_backend_adaptor(  # 创建后端适配器的私有工厂函数
    backend: str,  # 后端名称
    device: torch.device,  # 设备信息
    sparse_algorithm: BaseSparseAlgorithm,  # 稀疏算法实例
    req_to_token_pool,  # 请求到token映射池
):
    """Create backend adaptor."""  # 创建后端适配器 / 创建后端适配器
    if isinstance(sparse_algorithm, DeepSeekDSAAlgorithm):  # 如果是DeepSeek DSA算法
        return DSABackendAdaptor(device, req_to_token_pool)  # 创建并返回DSA后端适配器

    if backend in ["fa3", "flashattention"]:  # 如果后端是FlashAttention系列
        return FlashAttentionAdaptor(device)  # 创建并返回FlashAttention后端适配器

    raise ValueError(f"Unknown attention backend: {backend}")  # 未知后端，抛出值错误异常


def _parse_sparse_config(server_args) -> SparseConfig:  # 从服务参数解析稀疏配置的私有函数
    """Parse hierarchical sparse config from JSON string.  # 从JSON字符串解析分层稀疏配置

    Required fields with defaults: top_k (2048), device_buffer_size (2*top_k),
    host_to_device_ratio (2).  # 必填字段及默认值：top_k (2048)、device_buffer_size (2*top_k)、host_to_device_ratio (2)
    Optional fields (default None): algorithm, backend, min_sparse_prompt_len,
    page_size. All remaining fields go to sparse_extra_config.  # 可选字段（默认None）：algorithm、backend、min_sparse_prompt_len、page_size。其余字段存入sparse_extra_config
    """
    extra_config_str = server_args.hisparse_config  # 获取hisparse_config字符串
    if extra_config_str is not None:  # 如果配置字符串不为空
        try:
            extra_config = json.loads(extra_config_str)  # 解析JSON字符串为字典
        except json.JSONDecodeError as e:  # 如果JSON解析失败
            raise ValueError(f"Failed to parse hisparse_config: {e}") from e  # 抛出值错误异常
    else:  # 如果配置字符串为空
        extra_config = {}  # 使用空字典作为默认配置

    top_k = extra_config.pop("top_k", 2048)  # 提取top_k参数，默认2048
    device_buffer_size = extra_config.pop("device_buffer_size", 2 * top_k)  # 提取设备缓冲区大小，默认2倍top_k
    host_to_device_ratio = extra_config.pop("host_to_device_ratio", 2)  # 提取主机到设备比率，默认2

    if device_buffer_size < top_k:  # 校验设备缓冲区大小不小于top_k
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be no smaller than top_k ({top_k})"
        )  # 抛出值错误异常

    algorithm = extra_config.pop("algorithm", None)  # 提取算法名称，默认None
    backend = extra_config.pop("backend", None)  # 提取后端名称，默认None
    min_sparse_prompt_len = extra_config.pop("min_sparse_prompt_len", None)  # 提取最小稀疏提示长度，默认None
    page_size = extra_config.pop("page_size", None)  # 提取页面大小，默认None

    return SparseConfig(  # 构建并返回SparseConfig实例
        top_k=top_k,  # top_k值
        device_buffer_size=device_buffer_size,  # 设备缓冲区大小
        host_to_device_ratio=host_to_device_ratio,  # 主机到设备比率
        algorithm=algorithm,  # 算法名称
        backend=backend,  # 后端名称
        page_size=page_size,  # 页面大小
        min_sparse_prompt_len=min_sparse_prompt_len,  # 最小稀疏提示长度
        sparse_extra_config=extra_config,  # 剩余的额外配置
    )


def parse_hisparse_config(server_args) -> SparseConfig:  # 从服务参数解析hisparse配置的公开函数
    """Parse hisparse config from server_args, returning defaults if no config provided."""  # 从server_args解析hisparse配置，如无配置则返回默认值 / 从server_args解析hisparse配置，如无配置则返回默认值
    return _parse_sparse_config(server_args)  # 委托给私有解析函数


def create_sparse_coordinator(  # 创建稀疏协调器的公开工厂函数
    device: torch.device,  # 设备信息
    req_to_token_pool,  # 请求到token映射池
    token_to_kv_pool,  # token到KV缓存池
    start_layer: int,  # 起始层ID
    end_layer: int,  # 结束层ID
    server_args,  # 服务参数
    **kwargs,  # 其他关键字参数
) -> SparseCoordinator:  # 返回稀疏协调器实例
    config = _parse_sparse_config(server_args)  # 解析稀疏配置
    algorithm = _create_sparse_algorithm(config, device, **kwargs)  # 创建稀疏算法实例
    backend_adaptor = _create_backend_adaptor(  # 创建后端适配器
        config.backend, device, algorithm, req_to_token_pool
    )

    coordinator = SparseCoordinator(  # 创建稀疏协调器实例
        config=config,  # 稀疏配置
        algorithm=algorithm,  # 稀疏算法
        backend_adaptor=backend_adaptor,  # 后端适配器
        req_to_token_pool=req_to_token_pool,  # 请求到token映射池
        token_to_kv_pool=token_to_kv_pool,  # token到KV缓存池
        start_layer=start_layer,  # 起始层ID
        end_layer=end_layer,  # 结束层ID
        device=device,  # 设备信息
    )
    register_sparse_coordinator(coordinator)  # 注册为全局稀疏协调器
    return coordinator  # 返回协调器实例


def register_sparse_coordinator(coordinator: SparseCoordinator) -> None:  # 注册全局稀疏协调器
    global _global_sparse_coordinator  # 声明使用全局变量
    _global_sparse_coordinator = coordinator  # 设置全局稀疏协调器实例


def get_sparse_coordinator() -> Optional[SparseCoordinator]:  # 获取全局稀疏协调器实例
    return _global_sparse_coordinator  # 返回全局稀疏协调器实例
