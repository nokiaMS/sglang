# 标签转换模块
# 本模块提供优先级到字符串的转换功能
# 用于指标上报时限制标签基数，防止高基数问题

from typing import Optional  # 导入可选类型

_PRIORITY_MIN = 0  # 优先级最小值
_PRIORITY_MAX = 31  # 优先级最大值
_LOW_PRIORITY_VALUE = "LOW"  # 低优先级标签值
_HIGH_PRIORITY_VALUE = "HIGH"  # 高优先级标签值

UNKNOWN_PRIORITY_VALUE = "UNKNOWN"  # 未知优先级标签值


def transform_priority(priority: Optional[int]) -> str:  # 将优先级转换为字符串用于指标上报
    """Transform the priority to a string for metrics reporting.
    Limit the range to prevent high cardinality issues.

    Args:
        priority: The priority to transform.
    Returns:
        The transformed priority.
    """
    if priority is None:  # 如果优先级为None
        return UNKNOWN_PRIORITY_VALUE  # 返回未知值
    elif priority < _PRIORITY_MIN:  # 如果低于最小值
        return _LOW_PRIORITY_VALUE  # 返回低优先级
    elif priority >= _PRIORITY_MAX:  # 如果高于最大值
        return _HIGH_PRIORITY_VALUE  # 返回高优先级
    else:  # 在有效范围内
        return str(priority)  # 转换为字符串
