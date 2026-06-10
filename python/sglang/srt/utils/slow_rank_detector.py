# 慢速秩检测器，用于检测分布式推理中性能不均衡的GPU
# 通过运行GEMM和逐元素基准测试比较各秩的相对速度，发现性能瓶颈
import logging  # 导入日志记录模块
from typing import Any, Dict, List  # 导入类型注解

import torch  # 导入PyTorch张量库
import torch.distributed as dist  # 导入PyTorch分布式通信模块
import triton  # 导入Triton GPU编程框架

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def execute():  # 执行慢速秩检测基准测试
    if dist.get_rank() == 0:  # 仅在主进程（秩0）打印开始信息
        logger.info(f"[slow_rank_detector] Start benchmarking...")  # 记录基准测试开始

    local_metrics = {  # 在本地运行所有基准测试并收集指标
        bench_name: _compute_local_metric(bench_name) for bench_name in _BENCH_NAMES  # 遍历基准名称计算指标
    }

    all_metrics = [None for _ in range(dist.get_world_size())]  # 初始化收集所有秩指标列表
    dist.gather_object(local_metrics, all_metrics if dist.get_rank() == 0 else None)  # 将所有秩的指标收集到主进程

    if dist.get_rank() == 0:  # 仅在主进程分析结果
        _analyze_metrics(all_metrics)  # 分析所有秩的性能指标


class _GemmExecutor:  # GEMM（矩阵乘法）基准测试执行器
    def __init__(self):  # 初始化GEMM执行器
        self.lhs = torch.randn((8192, 8192), dtype=torch.bfloat16, device="cuda")  # 左侧随机矩阵
        self.rhs = torch.randn((8192, 8192), dtype=torch.bfloat16, device="cuda")  # 右侧随机矩阵

    def __call__(self):  # 执行一次矩阵乘法
        self.lhs @ self.rhs  # 执行bfloat16矩阵乘法


class _ElementwiseExecutor:  # 逐元素运算基准测试执行器
    def __init__(self):  # 初始化逐元素执行器
        self.value = torch.randint(  # 创建随机整数张量
            0, 10000, (128 * 1024**2,), dtype=torch.int32, device="cuda"  # 128M个int32元素
        )

    def __call__(self):  # 执行一次逐元素加法
        self.value += 1  # 张量逐元素加1


_EXECUTOR_CLS_OF_BENCH = {  # 基准名称到执行器类的映射
    "gemm": _GemmExecutor,  # GEMM基准
    "elementwise": _ElementwiseExecutor,  # 逐元素基准
}

_BENCH_NAMES = list(_EXECUTOR_CLS_OF_BENCH.keys())  # 基准名称列表


def _compute_local_metric(bench_name):  # 在本地GPU上运行基准测试并返回执行时间
    executor = _EXECUTOR_CLS_OF_BENCH[bench_name]()  # 创建基准测试执行器实例
    ms = triton.testing.do_bench_cudagraph(executor, return_mode="mean", rep=20)  # 使用CUDA图基准测试，返回平均时间
    return ms  # 返回执行时间（毫秒）


def _analyze_metrics(all_metrics: List[Dict[str, Any]]):  # 分析所有秩的性能指标，检测慢速秩
    for bench_name in _BENCH_NAMES:  # 遍历所有基准名称
        time_of_rank = torch.tensor([m[bench_name] for m in all_metrics])  # 收集各秩的执行时间
        speed_of_rank = 1 / time_of_rank  # 计算各秩的速度（时间的倒数）
        rel_speed_of_rank = speed_of_rank / speed_of_rank.max()  # 计算相对于最快秩的相对速度
        slowest_rel_speed = rel_speed_of_rank.min().item()  # 获取最慢秩的相对速度
        logger.info(  # 记录基准测试分析结果
            f"[slow_rank_detector] {bench_name=} {slowest_rel_speed=} {rel_speed_of_rank=} {time_of_rank=}"  # 显示基准名称、最慢相对速度、各秩相对速度和时间
        )
        if slowest_rel_speed < 0.9:  # 如果最慢秩的速度不到最快秩的90%
            logger.warning(  # 发出慢速秩警告
                "[slow_rank_detector] Some ranks are too slow compared with others"  # 某些秩相比其他秩太慢
            )
