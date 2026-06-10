# 适用于Grafana热力图可视化的带gt/le桶标签的计量器（Gauge）
# 与使用累积桶的Prometheus直方图不同，它使用非累积桶（gt < value <= le），适合热力图显示
# 注意：需与Rust实现 sgl-model-gateway/src/observability/gauge_histogram.rs 保持同步
"""Gauge with gt/le bucket labels for Grafana heatmap visualization.

Unlike Prometheus Histogram which uses cumulative buckets, this uses
non-cumulative buckets (gt < value <= le) suitable for heatmap display.

Note: Keep in sync with Rust implementation in
sgl-model-gateway/src/observability/gauge_histogram.rs
"""

import bisect  # 导入二分查找模块
from typing import Dict, Iterator, List, Tuple, Union  # 导入类型提示


class BucketLabels:  # 桶标签类，管理计量器直方图的桶标签对
    """Bucket label pairs and count computation for a GaugeHistogram."""

    def __init__(self, upper_bounds: List[Union[int, float]]):  # 初始化桶标签，接收上界列表
        self._upper_bounds = upper_bounds  # 保存桶的上界列表
        self._labels: List[Tuple[str, str]] = []  # 初始化标签列表
        for i, upper in enumerate(upper_bounds):  # 遍历每个上界
            lower = upper_bounds[i - 1] if i > 0 else 0  # 下界为前一个上界，第一个桶的下界为0
            self._labels.append((str(lower), str(upper)))  # 添加(gt, le)标签对
        self._labels.append((str(upper_bounds[-1]), "+Inf"))  # 添加最后一个桶(最大上界, +Inf)

    def __len__(self) -> int:  # 返回标签数量
        return len(self._labels)

    def __iter__(self) -> Iterator[Tuple[str, str]]:  # 返回标签迭代器
        return iter(self._labels)

    def compute_bucket_counts(self, observations: List[Union[int, float]]) -> List[int]:  # 计算每个桶中的观测值数量，O(n)复杂度
        """Compute how many observations fall into each bucket. O(n) complexity."""
        counts = [0] * len(self)  # 初始化每个桶的计数为零
        for v in observations:  # 遍历每个观测值
            # bisect_left finds insertion point; values at boundary go to current bucket
            idx = bisect.bisect_left(self._upper_bounds, v)  # 使用二分查找确定观测值所属桶的索引
            counts[idx] += 1  # 对应桶计数加一
        return counts  # 返回每个桶的计数列表


class GaugeHistogram:  # 计量器直方图类，用于Grafana热力图可视化
    """Gauge with gt/le bucket labels for Grafana heatmap visualization."""

    def __init__(  # 初始化计量器直方图
        self,
        name: str,  # 指标名称
        documentation: str,  # 指标文档描述
        labelnames: List[str],  # 标签名称列表
        bucket_bounds: List[Union[int, float]],  # 桶的上界列表
        multiprocess_mode: str = "mostrecent",  # 多进程模式，默认为"mostrecent"
    ):
        from prometheus_client import Gauge  # 延迟导入Prometheus Gauge类

        self._buckets = BucketLabels(bucket_bounds)  # 创建桶标签对象

        self._gauge = Gauge(  # 创建Prometheus Gauge指标
            name=name,  # 指标名称
            documentation=documentation,  # 文档描述
            labelnames=list(labelnames) + ["gt", "le"],  # 标签名列表加上gt和le
            multiprocess_mode=multiprocess_mode,  # 多进程模式
        )

    def set_raw(self, labels: Dict[str, str], values: List[int]):  # 直接设置桶计数
        """Set bucket counts directly."""
        for (gt, le), count in zip(self._buckets, values):  # 遍历桶标签和对应计数值
            self._gauge.labels(**labels, gt=gt, le=le).set(count)  # 设置每个桶的计数值

    def set_by_current_observations(  # 根据当前观测值计算桶计数并设置
        self, labels: Dict[str, str], observations: List[Union[int, float]]
    ):
        """Compute bucket counts from observations and set them."""
        counts = self._buckets.compute_bucket_counts(observations)  # 从观测值计算桶计数
        self.set_raw(labels, counts)  # 设置桶计数

    def buckets(self) -> BucketLabels:  # 返回桶标签对象
        return self._buckets
