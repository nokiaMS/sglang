# 启动推理服务器的入口模块，根据参数选择不同的服务器模式
"""Launch the inference server."""

import asyncio
import os
import sys
import warnings

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

# 抑制嘈杂的警告信息
suppress_noisy_warnings()


# 根据服务器参数运行服务器，支持编码器分离、gRPC、Ray 和 HTTP 模式
def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.encoder_only:
        # 编码器分离模式
        # For encoder disaggregation
        if server_args.grpc_mode:
            # 编码器 gRPC 模式
            from sglang.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            # 编码器 HTTP 模式
            from sglang.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
    elif server_args.grpc_mode:
        # gRPC 模式（旧版 SMG 路径，后续将迁移至默认路径）
        # TODO: Once the native Rust gRPC server starts alongside HTTP in the
        # default path below (controlled by SGLANG_ENABLE_GRPC / SGLANG_GRPC_PORT),
        # remove this legacy SMG path and the grpc_mode flag.
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.use_ray:
        # Ray 模式
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # 默认模式：HTTP 模式
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


# 命令行入口点
if __name__ == "__main__":
    # 提示用户使用新的推荐入口点
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    # 加载插件
    from sglang.srt.plugins import load_plugins

    load_plugins()

    # 解析服务器命令行参数
    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        # 确保服务器退出时清理整个进程树
        kill_process_tree(os.getpid(), include_parent=False)
