"""Start bootstrap/kv-store-related server"""

# 本文件用于启动 disaggregated（分离式）推理架构中的引导服务（bootstrap server）和 KV 缓存存储服务。
# 在分离式推理中，prefill（预填充）和 decode（解码）阶段运行在不同的节点上，
# 需要通过引导服务来协调 KV 缓存的传输。

import os

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.server_args import ServerArgs


def start_disagg_service(
    server_args: ServerArgs,
):
    """启动分离式推理的引导服务。

    根据 disaggregation 模式和传输后端，在 prefill 节点上启动
    KV 引导服务器（bootstrap server），用于协调 prefill 和 decode
    节点之间的 KV 缓存传输。
    """
    # Start kv bootstrap server on prefill
    # 获取分离式推理模式（prefill 或 decode）
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    # 获取 KV 缓存传输后端类型（如 Mooncake、Nixl、Ascend 等）
    transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)

    if disagg_mode == DisaggregationMode.PREFILL:
        # only start bootstrap server on prefill tm
        # 仅在 prefill 节点上启动引导服务器
        kv_bootstrap_server_class = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        # 使用对应传输后端的引导服务器类创建实例
        bootstrap_server = kv_bootstrap_server_class(
            host=server_args.host,
            port=server_args.disaggregation_bootstrap_port,
        )
        # 当使用 Ascend 传输后端且当前节点为 rank 0 时，需要创建配置存储
        is_create_store = (
            server_args.node_rank == 0 and transfer_backend == TransferBackend.ASCEND
        )
        if is_create_store:
            try:
                from memfabric_hybrid import create_config_store

                # 从环境变量获取 Ascend memfabric 存储的 URL
                ascend_url = os.getenv("ASCEND_MF_STORE_URL")
                # 创建 Ascend 配置存储，用于 KV 缓存传输的协调
                create_config_store(ascend_url)
            except Exception as e:
                error_message = f"Failed create mf store, invalid ascend_url."
                error_message += f" With exception {e}"
                raise error_message

        return bootstrap_server
