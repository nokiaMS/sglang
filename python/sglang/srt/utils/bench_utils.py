# 基准测试工具模块
# 提供CUDA内核性能基准测试功能，包括输出抑制和kineto性能分析器集成
# 支持L2缓存刷新、Nsight Systems兼容以及回退到墙上时钟计时

import os  # 操作系统接口
import re  # 正则表达式
import sys  # 系统相关功能
from contextlib import nullcontext  # 空上下文管理器

import torch  # PyTorch深度学习框架


# NOTE copied and modified from DeepGEMM
# 注意：从DeepGEMM复制并修改
class suppress_stdout_stderr:  # 抑制标准输出和标准错误的上下文管理器
    def __enter__(self):  # 进入上下文时重定向输出到/dev/null
        self.outnull_file = open(os.devnull, "w")  # 打开/dev/null用于抑制标准输出
        self.errnull_file = open(os.devnull, "w")  # 打开/dev/null用于抑制标准错误

        self.old_stdout_fileno_undup = sys.stdout.fileno()  # 保存原始标准输出文件描述符
        self.old_stderr_fileno_undup = sys.stderr.fileno()  # 保存原始标准错误文件描述符

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())  # 复制标准输出文件描述符
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())  # 复制标准错误文件描述符

        self.old_stdout = sys.stdout  # 保存原始标准输出对象
        self.old_stderr = sys.stderr  # 保存原始标准错误对象

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)  # 将标准输出重定向到/dev/null
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)  # 将标准错误重定向到/dev/null

        sys.stdout = self.outnull_file  # 替换Python标准输出对象
        sys.stderr = self.errnull_file  # 替换Python标准错误对象
        return self  # 返回自身

    def __exit__(self, *_):  # 退出上下文时恢复原始输出
        sys.stdout = self.old_stdout  # 恢复Python标准输出对象
        sys.stderr = self.old_stderr  # 恢复Python标准错误对象

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)  # 恢复标准输出文件描述符
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)  # 恢复标准错误文件描述符

        os.close(self.old_stdout_fileno)  # 关闭复制的标准输出文件描述符
        os.close(self.old_stderr_fileno)  # 关闭复制的标准错误文件描述符

        self.outnull_file.close()  # 关闭/dev/null文件（标准输出）
        self.errnull_file.close()  # 关闭/dev/null文件（标准错误）


# NOTE copied and modified from DeepGEMM
# 注意：从DeepGEMM复制并修改
def bench_kineto(  # 使用kineto性能分析器进行CUDA内核基准测试
    fn,  # 待测试的函数
    kernel_names,  # 内核名称（字符串或元组）
    num_tests: int = 30,  # 每轮测试次数
    suppress_kineto_output: bool = False,  # 是否抑制kineto输出
    trace_path: str = None,  # Chrome追踪文件保存路径
    flush_l2: bool = True,  # 是否在每次调用前刷新L2缓存
    with_multiple_kernels: bool = False,  # 是否允许多个内核匹配
):
    # Conflict with Nsight Systems
    # 与Nsight Systems冲突
    using_nsys = int(os.environ.get("SGLANG_NSYS_PROFILING", 0))  # 检查是否使用Nsight Systems

    # By default, flush L2 with an excessive 8GB memset to give the GPU some (literal) chill time without full idle
    # 默认使用8GB memset刷新L2缓存，给GPU一些冷却时间而不是完全空闲
    flush_l2_size = int(8e9 // 4)  # 8GB对应的int32元素数量

    # For some auto-tuning kernels with prints
    # 针对带有打印输出的自动调优内核进行预热
    fn()  # 运行一次预热

    # Profile
    # 性能分析
    suppress = (  # 选择是否抑制输出
        suppress_stdout_stderr  # 使用自定义抑制器
        if suppress_kineto_output and not using_nsys  # 抑制输出且不使用nsys时
        else nullcontext  # 否则使用空上下文
    )
    with suppress():  # 进入抑制上下文
        schedule = (  # 创建分析器调度
            torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)  # 等待0步，预热1步，活跃1步
            if not using_nsys  # 不使用nsys时创建调度
            else None  # 使用nsys时不需要调度
        )
        profiler = (  # 创建性能分析器
            torch.profiler.profile(  # PyTorch性能分析器
                activities=[torch.profiler.ProfilerActivity.CUDA],  # 仅分析CUDA活动
                schedule=schedule,  # 分析调度
                acc_events=True,  # 累积事件
            )
            if not using_nsys  # 不使用nsys时创建分析器
            else nullcontext()  # 使用nsys时使用空上下文
        )
        with profiler:  # 进入分析器上下文
            for i in range(2):  # 运行两轮（warmup + active）
                for _ in range(num_tests):  # 每轮运行num_tests次
                    if flush_l2:  # 如果需要刷新L2缓存
                        torch.empty(  # 分配8GB显存
                            flush_l2_size, dtype=torch.int, device="cuda"  # int32类型在CUDA上
                        ).zero_()  # 零初始化以刷新L2缓存
                    fn()  # 执行待测试函数
                if not using_nsys:  # 如果不使用nsys
                    torch.cuda.synchronize()  # 同步CUDA流
                    profiler.step()  # 推进分析器到下一步

    # Return 1 if using Nsight Systems
    # 使用Nsight Systems时返回1
    if using_nsys:  # 如果使用nsys
        return 1  # 返回1

    # Parse the profiling table
    # 解析性能分析表
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)  # 验证kernel_names类型
    is_tuple = isinstance(kernel_names, tuple)  # 记录是否为元组类型
    prof_lines = (  # 获取分析结果表格并按行分割
        profiler.key_averages()  # 获取按键平均的事件
        .table(sort_by="cuda_time_total", max_name_column_width=100)  # 按CUDA总时间排序，列宽100
        .split("\n")  # 按换行符分割
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names  # 统一为元组
    assert all([isinstance(name, str) for name in kernel_names])  # 验证所有名称为字符串
    # Check if profiler captured any events (can be empty with some CUDA versions)
    # 检查分析器是否捕获了任何事件（某些CUDA版本可能为空）
    non_empty_lines = [l for l in prof_lines if l.strip() and not l.startswith("-")]  # 过滤非空行和非分隔线
    if len(non_empty_lines) <= 1:  # 如果没有有意义的数据行
        print(  # 输出警告
            "WARNING: Profiler returned empty table — falling back to wall-clock timing"  # 分析器返回空表，回退到墙上时钟计时
        )
        import time  # 导入时间模块

        torch.cuda.synchronize()  # 同步CUDA
        start = time.perf_counter()  # 记录起始时间
        for _ in range(num_tests):  # 运行num_tests次
            fn()  # 执行待测试函数
        torch.cuda.synchronize()  # 同步CUDA
        elapsed = (time.perf_counter() - start) / num_tests  # 计算平均耗时
        return tuple([elapsed] * len(kernel_names)) if is_tuple else elapsed  # 返回元组或单个值

    if not with_multiple_kernels:  # 如果不允许多内核匹配
        for name in kernel_names:  # 遍历每个内核名称
            assert (  # 断言每个名称只匹配一行
                sum([int(re.search(name, line) is not None) for line in prof_lines])  # 计算匹配行数
                == 1  # 必须恰好匹配1行
            ), f"Errors of the kernel {name} in the profiling table (table: {prof_lines})"  # 错误信息

    # Save chrome traces
    # 保存Chrome追踪文件
    if trace_path is not None:  # 如果指定了追踪文件路径
        profiler.export_chrome_trace(trace_path)  # 导出Chrome追踪格式

    # Return average kernel times
    # 返回平均内核时间
    units = {"ms": 1e3, "us": 1e6}  # 时间单位映射：毫秒和微秒
    kernel_times = []  # 内核时间列表
    for name in kernel_names:  # 遍历每个内核名称
        total_time = 0  # 总时间累计
        total_num = 0  # 总调用次数累计
        for line in prof_lines:  # 遍历分析表的每一行
            if re.search(name, line) is not None:  # 如果行中包含内核名称
                time_str = line.split()[-2]  # 获取时间字符串（倒数第二列）
                num_str = line.split()[-1]  # 获取调用次数（最后一列）
                for unit, scale in units.items():  # 遍历时间单位
                    if unit in time_str:  # 如果时间字符串包含该单位
                        total_time += (  # 累加总时间（转换为秒）
                            float(time_str.replace(unit, "")) / scale * int(num_str)  # 时间值除以单位缩放乘以调用次数
                        )
                        total_num += int(num_str)  # 累加调用次数
                        break  # 匹配到一个单位后跳出循环
        kernel_times.append(total_time / total_num)  # 计算平均时间

    return tuple(kernel_times) if is_tuple else kernel_times[0]  # 返回元组或单个值
