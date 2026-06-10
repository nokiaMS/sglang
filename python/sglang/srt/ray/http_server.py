# Ray感知的HTTP服务器启动模块
# 本模块提供基于Ray的HTTP服务器启动函数，使用RayEngine替代标准Engine
# 来启动调度器Actor。
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ray-aware HTTP server launcher."""  # Ray感知的HTTP服务器启动器

from typing import Callable, Optional  # 导入类型注解

from sglang.srt.entrypoints.engine import (  # 导入引擎相关函数
    init_tokenizer_manager,
    run_detokenizer_process,
    run_scheduler_process,
)
from sglang.srt.server_args import ServerArgs  # 导入服务器参数类


def launch_server(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable = init_tokenizer_manager,
    run_scheduler_process_func: Callable = run_scheduler_process,
    run_detokenizer_process_func: Callable = run_detokenizer_process,
    execute_warmup_func: Optional[Callable] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):  # 启动HTTP服务器
    """Launch HTTP server with Ray-based scheduler actors.  # 使用基于Ray的调度器Actor启动HTTP服务器

    Mirrors http_server.launch_server() but uses RayEngine for scheduler launching.  # 镜像http_server.launch_server()但使用RayEngine启动调度器
    """
    from sglang.srt.entrypoints.http_server import (  # 导入HTTP服务器相关函数
        _execute_server_warmup,
        _setup_and_run_http_server,
    )
    from sglang.srt.ray.engine import RayEngine  # 导入Ray引擎类

    if execute_warmup_func is None:  # 如果未指定预热函数
        execute_warmup_func = _execute_server_warmup  # 使用默认预热函数

    server_args.placement_group = None  # 初始化放置组为None

    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
    ) = RayEngine._launch_subprocesses(  # 使用RayEngine启动子进程
        server_args,
        init_tokenizer_manager_func=init_tokenizer_manager_func,
        run_scheduler_process_func=run_scheduler_process_func,
        run_detokenizer_process_func=run_detokenizer_process_func,
    )

    _setup_and_run_http_server(  # 设置并运行HTTP服务器
        server_args,
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result.scheduler_infos,
        subprocess_watchdog,
        execute_warmup_func=execute_warmup_func,
        launch_callback=launch_callback,
    )
