# 可中断CUDA图（BCG）运行时上下文管理模块
# 提供BCG运行状态的查询与上下文管理功能，
# 用于标记当前是否处于可中断CUDA图捕获/重放阶段。

# Copyright 2023-2026 SGLang Team
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
"""Runtime state for the breakable CUDA graph (BCG) runner.  # 可中断CUDA图（BCG）运行器的运行时状态

Kept intentionally separate from ``compilation/piecewise_context_manager.py``:  # 故意与 compilation/piecewise_context_manager.py 分开：
BCG no longer inherits from the torch.compile-based PCG path, so its  # BCG不再继承基于torch.compile的PCG路径，因此
capture/replay lifecycle is managed on its own.  # 其捕获/重放生命周期独立管理。
"""

from __future__ import annotations  # 启用延迟类型注解评估

from contextlib import contextmanager  # 导入上下文管理器装饰器

_in_breakable_cuda_graph = False  # 全局标志，标记当前是否处于BCG上下文中


def is_in_breakable_cuda_graph() -> bool:  # 查询当前是否处于可中断CUDA图上下文中
    """Return whether the current thread is inside a BCG context."""  # 返回当前线程是否处于BCG上下文中
    return _in_breakable_cuda_graph  # 返回BCG上下文标志


@contextmanager  # 上下文管理器装饰器
def enable_breakable_cuda_graph():  # 启用可中断CUDA图上下文的管理器
    """Context manager that sets the BCG active flag for the enclosed block."""  # 为所包含的代码块设置BCG活跃标志的上下文管理器
    global _in_breakable_cuda_graph  # 声明使用全局变量
    _in_breakable_cuda_graph = True  # 进入上下文时设置标志为True
    try:  # 尝试执行
        yield  # 让出控制权给被管理的代码块
    finally:  # 无论如何都执行清理
        _in_breakable_cuda_graph = False  # 退出上下文时恢复标志为False
