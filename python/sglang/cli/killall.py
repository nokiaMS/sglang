# SGLang 进程清理模块（仅限 CI 模式）
# 在 CUDA_VISIBLE_DEVICES 指定的 GPU 上终止所有 SGLang 相关进程
# 用于每个 CI 作业开始时清理上一次（可能被取消的）运行留下的孤儿进程
# 需要 SGLANG_IS_IN_CI=true 环境变量
# 对于本地/非 CI 用途，请使用 scripts/killall_sglang.sh
#!/usr/bin/env python3  # Python3 解释器声明
"""Kill SGLang processes on CUDA_VISIBLE_DEVICES GPUs (CI mode only).
# 终止 CUDA_VISIBLE_DEVICES 指定 GPU 上的 SGLang 进程（仅限 CI 模式）

Called at the start of every CI job to clean up orphaned processes from
# 在每个 CI 作业开始时调用，清理上一次运行留下的孤儿进程
previous (possibly cancelled) runs. Requires SGLANG_IS_IN_CI=true.
# （可能被取消的）。需要 SGLANG_IS_IN_CI=true。

For local/non-CI usage, use scripts/killall_sglang.sh instead.
# 对于本地/非 CI 用途，请使用 scripts/killall_sglang.sh。

Usage:
# 使用方法：
    python killall.py
# python killall.py

Exit codes:
# 退出码：
    0 - Clean: all target GPUs have <10% memory usage after cleanup
# 0 - 干净：清理后所有目标 GPU 的内存使用率低于 10%
    1 - Dirty: GPU memory still >10% after cleanup, indicating stuck processes
# 1 - 脏：清理后 GPU 内存仍超过 10%，表示有卡住的进程
        or orphaned CUDA contexts that need a container restart
# 或孤儿 CUDA 上下文，需要重启容器
"""

import os  # 导入操作系统接口模块
import re  # 导入正则表达式模块
import signal  # 导入信号处理模块
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关模块
import time  # 导入时间模块
from pathlib import Path  # 导入路径操作模块

# Constants  # 常量定义
MEMORY_THRESHOLD_PCT = 10  # 内存使用率阈值百分比，超过此值视为脏 GPU

# Patterns matching SGLang process command lines (equivalent to pgrep -f in killall_sglang.sh)  # 匹配 SGLang 进程命令行的模式（等同于 killall_sglang.sh 中的 pgrep -f）
_SGLANG_PROCESS_PATTERNS = re.compile(  # 编译正则表达式，用于匹配 SGLang 相关进程
    r"sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::|sglang serve"
)

# Boxed output helpers  # 框格式输出辅助工具
_LOG_LINES = []  # 日志行缓冲区


def _log(msg=""):  # 将一行消息缓冲到日志列表中
    """Buffer a line for boxed output."""  # 缓冲一行用于框格式输出
    _LOG_LINES.append(msg)  # 将消息添加到日志行列表


def _flush_box(title, status=""):  # 将缓冲的日志行以框格式打印，然后清空缓冲区
    """Print all buffered lines inside a box, then clear buffer."""  # 将所有缓冲行在框内打印，然后清空缓冲区
    lines = _LOG_LINES.copy()  # 复制日志行列表
    _LOG_LINES.clear()  # 清空日志行列表

    all_text = [title] + ([status] if status else []) + lines  # 合并标题、状态和日志行
    width = max((len(line) for line in all_text), default=40) + 4  # 计算框的宽度，取最长行加 4
    width = max(width, 60)  # 最小宽度为 60

    h_bar = "─" * (width - 2)  # 生成水平分隔线
    print(f"\n┌{h_bar}┐")  # 打印框顶部
    print(f"│ {title:<{width - 3}}│")  # 打印标题行
    print(f"├{h_bar}┤")  # 打印分隔线
    for line in lines:  # 遍历每一行日志
        print(f"│ {line:<{width - 3}}│")  # 打印日志行
    if status:  # 如果有状态信息
        print(f"├{h_bar}┤")  # 打印分隔线
        print(f"│ {status:<{width - 3}}│")  # 打印状态行
    print(f"└{h_bar}┘")  # 打印框底部


# nvidia-smi helpers  # nvidia-smi 辅助函数
def _run_smi(query, query_type="gpu"):  # 运行 nvidia-smi 查询并返回原始 CSV 行
    """Run nvidia-smi query and return raw CSV lines."""  # 运行 nvidia-smi 查询并返回原始 CSV 行
    flag = "--query-gpu" if query_type == "gpu" else "--query-compute-apps"  # 根据查询类型选择标志
    try:  # 尝试执行 nvidia-smi
        out = subprocess.check_output(  # 执行 nvidia-smi 命令并获取输出
            ["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],  # nvidia-smi 命令及参数
            text=True,  # 以文本模式返回输出
            timeout=10,  # 超时 10 秒
        )
        return [line.strip() for line in out.strip().splitlines() if line.strip()]  # 返回非空的 CSV 行列表
    except (subprocess.SubprocessError, FileNotFoundError):  # 捕获子进程错误或命令未找到
        return []  # 出错时返回空列表


def _get_smi_version():  # 返回 nvidia-smi 驱动版本和 GPU 名称，失败返回 None
    """Return nvidia-smi driver version and GPU name, or None on failure."""  # 返回 nvidia-smi 驱动版本和 GPU 名称，失败返回 None
    # Inline nvidia-smi query — killall.py runs before pip install, so sglang  # 内联 nvidia-smi 查询——killall.py 在 pip install 之前运行，因此 sglang
    # internals may not be importable.  # 内部模块可能无法导入。
    try:  # 尝试获取驱动版本
        result = subprocess.run(  # 运行 nvidia-smi 查询驱动版本
            [
                "nvidia-smi",  # nvidia-smi 命令
                "--query-gpu=driver_version",  # 查询驱动版本
                "--format=csv,noheader,nounits",  # CSV 格式，无表头，无单位
            ],
            capture_output=True,  # 捕获输出
            text=True,  # 文本模式
            check=True,  # 检查返回码
            timeout=10,  # 超时 10 秒
        )
        driver = result.stdout.strip().split("\n")[0].strip() or None  # 获取第一行作为驱动版本
    except (subprocess.SubprocessError, FileNotFoundError):  # 捕获错误
        driver = None  # 失败时设为 None
    if driver is None:  # 如果驱动版本获取失败
        return None  # 返回 None
    try:  # 尝试获取 GPU 名称
        out = subprocess.check_output(  # 运行 nvidia-smi 查询 GPU 名称
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # 查询 GPU 名称
            text=True,  # 文本模式
            timeout=10,  # 超时 10 秒
        )
        gpu_name = out.strip().splitlines()[0].strip() if out.strip() else "unknown"  # 获取 GPU 名称
    except (subprocess.SubprocessError, FileNotFoundError, IndexError):  # 捕获错误
        gpu_name = "unknown"  # 失败时设为 "unknown"
    return f"driver {driver}, {gpu_name}"  # 返回驱动版本和 GPU 名称的字符串


def _get_target_gpus():  # 返回 CUDA_VISIBLE_DEVICES 中的 GPU 索引，或所有可见 GPU
    """Return GPU indices from CUDA_VISIBLE_DEVICES, or all visible GPUs.
# 返回 CUDA_VISIBLE_DEVICES 中的 GPU 索引，或所有可见 GPU

    Note: only numeric indices are supported (e.g. "0,1,2").
# 注意：仅支持数字索引（例如 "0,1,2"）。
    UUID-style CUDA_VISIBLE_DEVICES values (e.g. "GPU-d4f1...") are not handled.
# UUID 风格的 CUDA_VISIBLE_DEVICES 值（例如 "GPU-d4f1..."）不被处理。
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")  # 获取 CUDA_VISIBLE_DEVICES 环境变量
    if cvd is not None and cvd.strip():  # 如果环境变量存在且非空
        return {int(g.strip()) for g in cvd.split(",") if g.strip().isdigit()}  # 解析逗号分隔的数字索引为集合
    return {int(line) for line in _run_smi("index") if line.isdigit()}  # 否则查询 nvidia-smi 获取所有 GPU 索引


def _get_gpu_pids(gpu_indices):  # 返回使用指定 GPU（按索引）的进程 PID 集合
    """Return PIDs using the specified GPUs (by index)."""  # 返回使用指定 GPU（按索引）的 PID
    target_uuids = set()  # 目标 GPU 的 UUID 集合
    for line in _run_smi("index,uuid"):  # 查询 GPU 索引和 UUID
        parts = line.split(",", 1)  # 按逗号分割为两部分
        if len(parts) == 2 and parts[0].strip().isdigit():  # 如果分割正确且索引为数字
            if int(parts[0].strip()) in gpu_indices:  # 如果索引在目标集合中
                target_uuids.add(parts[1].strip())  # 添加对应的 UUID
    pids = set()  # 进程 PID 集合
    for line in _run_smi("gpu_uuid,pid", query_type="apps"):  # 查询 GPU 上的进程
        parts = line.split(",", 1)  # 按逗号分割
        if len(parts) == 2 and parts[0].strip() in target_uuids:  # 如果 UUID 在目标集合中
            pid = parts[1].strip()  # 获取 PID 字符串
            if pid.isdigit():  # 如果 PID 为数字
                pids.add(int(pid))  # 添加到 PID 集合
    return pids  # 返回 PID 集合


def _get_gpu_memory(gpu_indices):  # 查询目标 GPU 的内存使用情况
    """Query memory usage for target GPUs.
# 查询目标 GPU 的内存使用情况

    Returns list of (idx, used_mib, total_mib, pct) tuples.
# 返回 (索引, 已用MiB, 总MiB, 百分比) 元组列表
    """
    result = []  # 结果列表
    for line in _run_smi("index,memory.used,memory.total"):  # 查询 GPU 内存使用情况
        parts = line.split(",")  # 按逗号分割
        if len(parts) != 3 or not parts[0].strip().isdigit():  # 如果格式不对
            continue  # 跳过
        idx = int(parts[0].strip())  # 获取 GPU 索引
        if idx not in gpu_indices:  # 如果不是目标 GPU
            continue  # 跳过
        try:  # 尝试解析内存数据
            used, total = int(float(parts[1].strip())), int(float(parts[2].strip()))  # 解析已用和总内存
        except ValueError:  # 捕获值错误
            continue  # 跳过
        pct = used / total * 100 if total > 0 else 0  # 计算使用百分比
        result.append((idx, used, total, pct))  # 添加到结果列表
    return result  # 返回结果列表


def _get_dirty_gpus(gpu_indices):  # 返回脏 GPU（内存使用率超过阈值）的描述字符串列表
    """Return list of dirty GPU description strings (memory >= threshold)."""  # 返回脏 GPU 描述字符串列表（内存使用率 >= 阈值）
    return [  # 返回列表推导式
        f"GPU {idx} ({pct:.0f}%)"  # 格式化 GPU 描述
        for idx, _, _, pct in _get_gpu_memory(gpu_indices)  # 遍历 GPU 内存信息
        if pct >= MEMORY_THRESHOLD_PCT  # 仅包含内存使用率超过阈值的 GPU
    ]


def _log_gpu_memory(gpu_indices):  # 记录所有目标 GPU 的内存使用情况，并返回脏 GPU 描述
    """Log memory usage for all target GPUs and return dirty GPU descriptions."""  # 记录所有目标 GPU 的内存使用情况并返回脏 GPU 描述
    dirty = []  # 脏 GPU 列表
    for idx, used, total, pct in _get_gpu_memory(gpu_indices):  # 遍历 GPU 内存信息
        _log(f"  GPU {idx}: {used} MiB / {total} MiB ({pct:.0f}%)")  # 记录内存使用信息
        if pct >= MEMORY_THRESHOLD_PCT:  # 如果内存使用率超过阈值
            dirty.append(f"GPU {idx} ({pct:.0f}%)")  # 添加到脏 GPU 列表
    return dirty  # 返回脏 GPU 列表


# /proc helpers  # /proc 文件系统辅助函数
def _read_proc_cmdline(pid):  # 读取 /proc/{pid}/cmdline 并返回解码后的字符串，失败返回 None
    """Read /proc/{pid}/cmdline and return as decoded string, or None on failure."""  # 读取 /proc/{pid}/cmdline 并返回解码后的字符串，失败返回 None
    try:  # 尝试读取进程命令行
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()  # 读取原始字节
        return raw.decode("utf-8", errors="replace").replace("\x00", " ")  # 解码为 UTF-8，替换空字节为空格
    except (FileNotFoundError, PermissionError):  # 捕获文件不存在或权限错误
        return None  # 返回 None


def _get_pid_cmdline(pid):  # 获取进程的截断命令行字符串
    """Get truncated command line for a PID."""  # 获取 PID 的截断命令行
    cmdline = _read_proc_cmdline(pid)  # 读取进程命令行
    if cmdline is None:  # 如果读取失败
        return "<unknown>"  # 返回 "<unknown>"
    cmdline = cmdline.strip()  # 去除首尾空白
    return cmdline[:120] + ("..." if len(cmdline) > 120 else "")  # 截断到 120 字符，超过则添加 "..."


def _find_sglang_pids_by_name():  # 通过命令行模式匹配查找 SGLang 进程 PID
    """Find SGLang process PIDs by command-line pattern matching.
# 通过命令行模式匹配查找 SGLang 进程 PID

    Scans /proc/*/cmdline for patterns matching known SGLang entry points.
# 扫描 /proc/*/cmdline 查找匹配已知 SGLang 入口点的模式
    Equivalent to: pgrep -f 'sglang::|sglang.launch_server|...'
# 等同于：pgrep -f 'sglang::|sglang.launch_server|...'

    Safe in shared-GPU containers: without --pid=host, /proc only exposes
# 在共享 GPU 容器中安全：没有 --pid=host 时，/proc 仅暴露
    processes in our own PID namespace, so this cannot kill other containers.
# 我们自己 PID 命名空间中的进程，因此不会杀死其他容器
    """
    my_pid = os.getpid()  # 获取当前进程 PID
    pids = set()  # SGLang 进程 PID 集合
    for entry in Path("/proc").iterdir():  # 遍历 /proc 目录
        if not entry.name.isdigit():  # 如果不是数字目录（不是进程目录）
            continue  # 跳过
        pid = int(entry.name)  # 获取 PID
        if pid <= 1 or pid == my_pid:  # 跳过 init 进程和自身
            continue  # 跳过
        cmdline = _read_proc_cmdline(pid)  # 读取进程命令行
        if cmdline and _SGLANG_PROCESS_PATTERNS.search(cmdline):  # 如果匹配 SGLang 模式
            pids.add(pid)  # 添加到 PID 集合
    return pids  # 返回 PID 集合


def _check_pid_namespace(pid):  # 检查指定 PID 是否与当前进程在同一 PID 命名空间中
    """Check if a PID is in our PID namespace. Linux-only via /proc."""  # 检查 PID 是否在我们的 PID 命名空间中。仅限 Linux，通过 /proc
    try:  # 尝试读取当前进程的 PID 命名空间
        my_ns = os.readlink("/proc/self/ns/pid")  # 读取当前进程的 PID 命名空间
    except OSError:  # 捕获操作系统错误
        return "unknown (can't read self ns)"  # 返回未知信息
    try:  # 尝试读取目标进程的 PID 命名空间
        target_ns = os.readlink(f"/proc/{pid}/ns/pid")  # 读取目标进程的 PID 命名空间
    except FileNotFoundError:  # 如果目标进程不存在
        return f"NOT in our namespace (pid not in /proc, self={my_ns})"  # 返回不在同一命名空间
    except PermissionError:  # 如果没有权限
        return "unknown (no permission to read ns)"  # 返回未知信息
    if my_ns == target_ns:  # 如果命名空间相同
        return f"same namespace ({my_ns})"  # 返回同一命名空间
    return f"DIFFERENT namespace (self={my_ns}, target={target_ns})"  # 返回不同命名空间


def _get_orchestrator_ancestors(pids):  # 向上遍历进程树，返回是测试编排器的祖先进程
    """Walk process tree upward from PIDs, return ancestors that are test orchestrators.
# 从 PID 向上遍历进程树，返回是测试编排器的祖先进程

    Linux-only: reads /proc filesystem. Returns empty set on other platforms.
# 仅限 Linux：读取 /proc 文件系统。在其他平台返回空集合
    """
    orchestrator_patterns = ["run_suite.py", "run_tests.py"]  # 编排器命令行匹配模式
    ancestors, visited = set(), set()  # 祖先集合和已访问集合
    for pid in pids:  # 遍历每个 PID
        current = pid  # 当前进程 PID
        while current > 1 and current not in visited:  # 向上遍历，直到 init 或已访问
            visited.add(current)  # 标记为已访问
            cmdline = _read_proc_cmdline(current)  # 读取命令行
            if cmdline is None:  # 如果读取失败
                break  # 跳出循环
            if any(p in cmdline for p in orchestrator_patterns):  # 如果匹配编排器模式
                ancestors.add(current)  # 添加到祖先集合
            try:  # 尝试获取父进程 PID
                current = int(Path(f"/proc/{current}/stat").read_text().split()[3])  # 从 /proc/stat 读取父 PID
            except (FileNotFoundError, PermissionError, IndexError, ValueError):  # 捕获各种错误
                break  # 跳出循环
    return ancestors  # 返回祖先集合


# Kill & diagnostic helpers  # 终止和诊断辅助函数
def _kill_pids(pids, label="", quiet=False):  # 向指定 PID 发送 SIGKILL 信号，跳过自身和 init 进程
    """Send SIGKILL to PIDs, skipping self and init.
# 向 PID 发送 SIGKILL，跳过自身和 init

    Returns dict of {pid: exception_name} for PIDs that could not be killed.
# 返回无法终止的 PID 及其异常名称的字典
    When quiet=True, does not log individual kill results.
# 当 quiet=True 时，不记录单独的终止结果
    """
    my_pid = os.getpid()  # 获取当前进程 PID
    pids = {p for p in pids if p != my_pid and p > 1}  # 过滤掉自身和 init 进程
    if not pids:  # 如果没有需要终止的进程
        return {}  # 返回空字典
    if label and not quiet:  # 如果有标签且不是安静模式
        _log(f"  Killing {label}:")  # 记录正在终止的进程组
    failed = {}  # 终止失败的进程字典
    for pid in sorted(pids):  # 按排序顺序遍历 PID
        try:  # 尝试发送 SIGKILL
            os.kill(pid, signal.SIGKILL)  # 发送 SIGKILL 信号
            if not quiet:  # 如果不是安静模式
                _log(f"    PID {pid}: killed ({_get_pid_cmdline(pid)})")  # 记录成功终止
        except (ProcessLookupError, PermissionError) as e:  # 捕获进程不存在或权限错误
            failed[pid] = type(e).__name__  # 记录失败原因
            if not quiet:  # 如果不是安静模式
                _log(f"    PID {pid}: failed ({type(e).__name__})")  # 记录终止失败
    return failed  # 返回失败的进程字典


def _get_ps_diagnostic():  # 返回 ps auxf 输出中与 GPU/SGLang 相关的进程信息
    """Return ps auxf output filtered for GPU/sglang-related processes."""  # 返回过滤后的 ps auxf 输出（GPU/sglang 相关进程）
    try:  # 尝试运行 ps 命令
        out = subprocess.run(["ps", "auxf"], capture_output=True, text=True, timeout=5)  # 执行 ps auxf 命令
        return [  # 返回过滤后的行列表
            line.strip()[:140]  # 截断到 140 字符
            for line in out.stdout.splitlines()  # 遍历输出行
            if any(k in line.lower() for k in ["sglang", "python", "cuda", "gpu"])  # 过滤包含关键词的行
        ][:20]  # 最多返回 20 行
    except (subprocess.SubprocessError, FileNotFoundError):  # 捕获错误
        return []  # 返回空列表


def _print_diagnostics(unkillable_pids):  # 在失败框之后打印详细的诊断信息
    """Print detailed diagnostics after the FAIL box (to stdout, outside box)."""  # 在失败框之后打印详细诊断信息（到标准输出，在框外）
    if unkillable_pids:  # 如果有无法终止的 PID
        print("\n[killall] Diagnostic — unkillable PIDs:")  # 打印标题
        for pid in sorted(unkillable_pids):  # 遍历排序后的 PID
            ns_info = _check_pid_namespace(pid)  # 检查命名空间信息
            print(f"  PID {pid}: ns: {ns_info}")  # 打印命名空间信息
    ps_lines = _get_ps_diagnostic()  # 获取进程诊断信息
    if ps_lines:  # 如果有进程信息
        print("\n[killall] Diagnostic — processes in this container (ps auxf):")  # 打印标题
        for line in ps_lines:  # 遍历每行
            print(f"  {line}")  # 打印进程信息
    else:  # 如果没有进程信息
        print(  # 打印无进程提示
            "\n[killall] Diagnostic — no sglang/python/gpu processes "
            "in this container"
        )


# CI mode  # CI 模式
def _kill_all_targets(gpu_indices, gpu_pids):  # 终止所有目标进程：名称匹配的、编排器祖先、GPU 进程
    """Kill all target processes: name-matched, orchestrator ancestors, GPU processes."""  # 终止所有目标进程：名称匹配、编排器祖先、GPU 进程
    # Kill name-matched SGLang processes (catches processes not visible to nvidia-smi)  # 终止名称匹配的 SGLang 进程（捕获 nvidia-smi 不可见的进程）
    name_only = _find_sglang_pids_by_name() - gpu_pids  # 仅通过名称匹配的进程（不在 GPU 进程集合中的）
    if name_only:  # 如果有仅名称匹配的进程
        _kill_pids(name_only, "name-matched SGLang processes")  # 终止这些进程
        time.sleep(1)  # 等待 1 秒
        _log()  # 记录空行

    # Kill orchestrator ancestors first, then GPU processes (retry once)  # 先终止编排器祖先，再终止 GPU 进程（重试一次）
    if gpu_pids:  # 如果有 GPU 进程
        _kill_pids(_get_orchestrator_ancestors(gpu_pids), "orchestrator ancestors")  # 终止编排器祖先进程
        time.sleep(1)  # 等待 1 秒
        for attempt in range(2):  # 最多重试 2 次
            current_pids = _get_gpu_pids(gpu_indices)  # 获取当前 GPU 上的进程
            if not current_pids:  # 如果没有进程
                break  # 跳出循环
            label = "GPU processes" if attempt == 0 else "stubborn GPU processes"  # 第一次用 "GPU processes"，第二次用 "stubborn GPU processes"
            _kill_pids(current_pids, label)  # 终止 GPU 进程
            time.sleep(3)  # 等待 3 秒
    _log()  # 记录空行


def _verify_gpu_clean(gpu_indices):  # 重试循环：等待 GPU 变干净
    """Retry loop: wait for GPUs to become clean.
# 重试循环：等待 GPU 变干净

    Returns (dirty_list, unkillable_pids, elapsed_seconds).
# 返回 (脏GPU列表, 无法终止的PID字典, 已用秒数)
    """
    max_wait_secs = 100  # 最大等待时间（秒）
    retry_interval = 10  # 重试间隔（秒）
    elapsed = 0  # 已用时间
    dirty = None  # 脏 GPU 列表
    unkillable_pids = {}  # 无法终止的 PID 字典

    while True:  # 循环检查
        dirty = _get_dirty_gpus(gpu_indices)  # 获取脏 GPU 列表
        remaining_pids = _get_gpu_pids(gpu_indices)  # 获取剩余 GPU 进程

        if not dirty:  # 如果没有脏 GPU
            _log(f"Check at {elapsed}s: GPUs clean")  # 记录 GPU 已干净
            break  # 跳出循环

        dirty_summary = ", ".join(dirty)  # 汇总脏 GPU 描述

        if elapsed >= max_wait_secs:  # 如果超过最大等待时间
            remaining_info = (  # 构造剩余进程信息
                f", {len(remaining_pids)} processes remaining" if remaining_pids else ""
            )
            _log(f"Check at {elapsed}s: still dirty [{dirty_summary}]{remaining_info}")  # 记录仍然脏
            break  # 跳出循环

        # Kill remaining processes before waiting (silently for retries)  # 在等待前终止剩余进程（重试时静默）
        if remaining_pids:  # 如果有剩余进程
            failed = _kill_pids(remaining_pids, quiet=True)  # 静默终止剩余进程
            unkillable_pids.update(failed)  # 更新无法终止的 PID

        print(  # 打印重试信息
            f"[killall] GPUs still dirty at {elapsed}s [{dirty_summary}], "
            f"retrying in {retry_interval}s "
            f"({elapsed + retry_interval}/{max_wait_secs}s)..."
        )
        time.sleep(retry_interval)  # 等待重试间隔
        elapsed += retry_interval  # 更新已用时间

    if unkillable_pids:  # 如果有无法终止的 PID
        parts = [f"{p} ({unkillable_pids[p]})" for p in sorted(unkillable_pids)]  # 格式化信息
        _log(f"  Unkillable PIDs: {', '.join(parts)}")  # 记录无法终止的 PID

    return dirty, unkillable_pids, elapsed  # 返回脏 GPU 列表、无法终止的 PID 和已用时间


def _ci_mode():  # CI 模式主逻辑：GPU 范围内终止进程，如果 GPU 仍脏则中止
    """GPU-scoped kill, abort if GPUs remain dirty."""  # GPU 范围内终止，如果 GPU 仍脏则中止
    gpu_indices = _get_target_gpus()  # 获取目标 GPU 索引
    if not gpu_indices:  # 如果没有检测到 GPU
        _log("No GPUs detected, skipping cleanup")  # 记录跳过清理
        _flush_box("killall_sglang", status="SKIP")  # 打印跳过框
        return 0  # 返回成功

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")  # 获取 CUDA_VISIBLE_DEVICES 环境变量
    gpu_list = ", ".join(str(g) for g in sorted(gpu_indices))  # 格式化 GPU 列表

    smi_info = _get_smi_version()  # 获取 nvidia-smi 版本信息
    if smi_info:  # 如果有版本信息
        _log(f"nvidia-smi: {smi_info}")  # 记录版本信息
    if cvd is None or not cvd.strip():  # 如果未设置 CUDA_VISIBLE_DEVICES
        _log(  # 记录警告
            "WARNING: CUDA_VISIBLE_DEVICES is not set. "
            "Falling back to all visible GPUs."
        )
        _log("This may kill processes from other CI jobs on shared hosts.")  # 记录风险提示
    else:  # 如果设置了 CUDA_VISIBLE_DEVICES
        _log(f"CUDA_VISIBLE_DEVICES={cvd}")  # 记录环境变量值
    _log()  # 记录空行

    # Log pre-cleanup state  # 记录清理前的状态
    _log("Before cleanup:")  # 记录清理前标题
    _log_gpu_memory(gpu_indices)  # 记录 GPU 内存使用情况
    gpu_pids = _get_gpu_pids(gpu_indices)  # 获取 GPU 上的进程 PID
    if not gpu_pids:  # 如果没有 GPU 进程
        _log("  No processes on target GPUs")  # 记录无进程
    else:  # 如果有 GPU 进程
        _log(f"  Processes ({len(gpu_pids)}):")  # 记录进程数量
        for pid in sorted(gpu_pids):  # 遍历排序后的 PID
            _log(f"    PID {pid}: {_get_pid_cmdline(pid)}")  # 记录进程信息
    _log()  # 记录空行

    # Kill phase  # 终止阶段
    _kill_all_targets(gpu_indices, gpu_pids)  # 终止所有目标进程

    # Verify phase  # 验证阶段
    dirty, unkillable_pids, elapsed = _verify_gpu_clean(gpu_indices)  # 验证 GPU 是否干净

    if dirty:  # 如果仍有脏 GPU
        _log()  # 记录空行
        _log("Final GPU memory:")  # 记录最终内存标题
        _log_gpu_memory(gpu_indices)  # 记录 GPU 内存使用情况
        _log(f"ERROR: memory >={MEMORY_THRESHOLD_PCT}%: {', '.join(dirty)}")  # 记录错误信息
        _log(f"Orphaned CUDA contexts after {elapsed}s — container needs restart.")  # 记录需要重启容器
        _flush_box(f"killall_sglang: GPUs [{gpu_list}]", status="FAIL — Aborting CI")  # 打印失败框
        _print_diagnostics(unkillable_pids)  # 打印诊断信息
        return 1  # 返回失败

    _flush_box(f"killall_sglang: GPUs [{gpu_list}]", status="PASS — GPUs clean")  # 打印通过框
    return 0  # 返回成功


# Entry point  # 入口点
def main():  # 主入口函数，启动 CI 模式
    return _ci_mode()  # 调用 CI 模式并返回结果


if __name__ == "__main__":  # 如果直接运行此脚本
    sys.exit(main())  # 以 main() 的返回值作为退出码
