# 索引器Topk捕获模块
# 实现IndexerTopkCapturer，用于捕获索引器层的topk索引
# 支持全局单例模式以及从元信息中提取索引器topk数据

import logging  # 日志记录模块
from typing import Optional  # 可选类型注解

import numpy as np  # NumPy数值计算库
import pybase64  # Base64编解码库
import torch  # PyTorch深度学习框架

from sglang.srt.layers.dp_attention import get_attention_tp_size  # 获取注意力张量并行大小
from sglang.srt.state_capturer.base import BaseTopkCapturer  # Topk捕获器基类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class IndexerTopkCapturer(BaseTopkCapturer):  # 索引器Topk捕获器，继承自基类
    def __init__(  # 初始化索引器Topk捕获器
        self,
        num_tokens: int,  # token总数
        num_indexer_layers: int,  # 索引器层数
        index_topk: int,  # 索引topk大小
        max_running_requests: int,  # 最大运行请求数
        device: str,  # 设备类型
    ):
        from sglang.srt.server_args import get_global_server_args  # 延迟导入全局服务器参数

        self.num_indexer_layers = num_indexer_layers  # 保存索引器层数
        self.index_topk = index_topk  # 保存索引topk大小

        attn_tp_size = get_attention_tp_size()  # 获取注意力张量并行大小
        assert attn_tp_size == 1, "IndexerTopkCapturer now only supports DP attention"  # 仅支持DP注意力

        # DP-attention capture is per-rank-local: each rank writes [:local_batch, ...]
        # to its own device_cache, so the buffer only needs to fit one rank's batch.
        # DP注意力捕获是每个rank本地的：每个rank将[:local_batch, ...]写入自己的device_cache，
        # 因此缓冲区只需容纳一个rank的批大小。
        server_args = get_global_server_args()  # 获取全局服务器参数
        max_batch_size = max(server_args.chunked_prefill_size, max_running_requests)  # 取分块预填充大小和最大运行请求数的较大值

        super().__init__(  # 调用父类初始化
            num_tokens=num_tokens,  # token总数
            max_batch_size=max_batch_size,  # 最大批大小
            num_layers=self.num_indexer_layers,  # 索引器层数
            topk_size=self.index_topk,  # topk大小
            device=device,  # 设备类型
            name="indexer_topk",  # 捕获器名称
        )


_global_indexer_capturer: Optional[IndexerTopkCapturer] = None  # 全局索引器捕获器单例


def get_global_indexer_capturer() -> Optional[IndexerTopkCapturer]:  # 获取全局索引器捕获器
    return _global_indexer_capturer  # 返回全局单例


def set_global_indexer_capturer(capturer: Optional[IndexerTopkCapturer]):  # 设置全局索引器捕获器
    global _global_indexer_capturer  # 声明全局变量
    _global_indexer_capturer = capturer  # 更新全局单例


def maybe_capture_indexer_topk(  # 如果已设置捕获器则捕获索引器topk，否则原样传递
    layer_id: int, topk_indices: Optional[torch.Tensor]  # 层ID和topk索引张量
) -> Optional[torch.Tensor]:
    """Capture topk for layer_id if a capturer is set; pass through unchanged.

    Works in both expression context (`return maybe_capture_indexer_topk(...)`)
    and statement context (call for side-effect, ignore return).
    """
    # 如果已设置捕获器，则捕获指定层的topk；否则原样传递。
    # 同时适用于表达式上下文（return maybe_capture_indexer_topk(...)）
    # 和语句上下文（调用产生副作用，忽略返回值）。
    if topk_indices is None:  # 如果topk索引为空
        return None  # 返回None
    if (cap := get_global_indexer_capturer()) is not None:  # 如果全局捕获器存在
        cap.capture(layer_id=layer_id, topk_indices=topk_indices)  # 执行捕获
    return topk_indices  # 返回原始topk索引


def extract_indexer_topk_from_meta_info(data):  # 从元信息中提取索引器topk数据
    # Mirrors extract_routed_experts_from_meta_info: indices are returned as
    # base64-encoded int32 bytes. Caller reshapes to (seqlen-1, num_indexer_layers,
    # index_topk).
    # 与extract_routed_experts_from_meta_info镜像：索引以base64编码的int32字节返回。
    # 调用者需要重塑为(seqlen-1, num_indexer_layers, index_topk)形状。
    indexer_topk_base64 = data["meta_info"].get("indexer_topk", None)  # 获取base64编码的索引器topk
    indexer_topk = np.frombuffer(  # 从缓冲区创建NumPy数组
        pybase64.b64decode(indexer_topk_base64.encode("utf-8")), dtype=np.int32  # base64解码为int32
    )
    return indexer_topk  # 返回索引器topk数组


def create_indexer_capturer(  # 创建索引器捕获器工厂函数
    enable: bool,  # 是否启用
    num_indexer_layers: int,  # 索引器层数
    index_topk: int,  # 索引topk大小
    num_tokens: int,  # token总数
    max_running_requests: int,  # 最大运行请求数
    device: str,  # 设备类型
) -> Optional[IndexerTopkCapturer]:
    if not enable:  # 如果未启用
        return None  # 返回None
    if num_indexer_layers == 0:  # 如果没有索引器层
        logger.warning("No indexer layers found, IndexerTopkCapturer disabled")  # 输出警告日志
        return None  # 返回None
    return IndexerTopkCapturer(  # 创建并返回索引器捕获器
        num_tokens=num_tokens,  # token总数
        num_indexer_layers=num_indexer_layers,  # 索引器层数
        index_topk=index_topk,  # topk大小
        max_running_requests=max_running_requests,  # 最大运行请求数
        device=device,  # 设备类型
    )
