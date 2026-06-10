# CPU监控模块
# 本模块提供CPU使用率监控线程，定期采集进程CPU时间并上报Prometheus指标

import threading  # 导入线程模块
import time  # 导入时间模块

import psutil  # 导入系统进程工具


def start_cpu_monitor_thread(component: str, interval: float = 5.0) -> threading.Thread:  # 启动CPU监控线程
    from prometheus_client import Counter  # 导入Prometheus计数器

    cpu_seconds_total = Counter(  # 创建CPU时间计数器
        name="sglang:process_cpu_seconds_total",  # 指标名称
        documentation="Total CPU time consumed by this process (user + system)",  # 文档描述
        labelnames=["component"],  # 标签名称
    )

    def monitor():  # 监控函数
        process = psutil.Process()  # 获取当前进程对象
        last_times = process.cpu_times()  # 获取初始CPU时间

        while True:  # 无限循环
            time.sleep(interval)  # 按间隔休眠
            curr_times = process.cpu_times()  # 获取当前CPU时间
            delta = (curr_times.user - last_times.user) + (  # 计算用户态时间增量
                curr_times.system - last_times.system  # 加上系统态时间增量
            )
            cpu_seconds_total.labels(component=component).inc(delta)  # 上报增量
            last_times = curr_times  # 更新上次时间

    t = threading.Thread(target=monitor, daemon=True)  # 创建守护线程
    t.start()  # 启动线程
    return t  # 返回线程对象
