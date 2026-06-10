# 该文件提供了Prometheus指标工具函数
# 包含生成直方图桶（buckets）的辅助函数
# 支持双侧指数桶、自定义桶和默认桶的生成

# Copyright 2023-2025 SGLang Team
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
"""Utilities for Prometheus Metrics."""  # Prometheus指标工具

import math  # 数学模块
from typing import List  # 列表类型注解


def two_sides_exponential_buckets(
    middle: float, base: float, count: int  # 中心值  # 指数基数  # 桶数量
) -> List[float]:  # 生成双侧指数分布的桶边界
    """生成以middle为中心，向两侧以base为基数指数增长的桶边界列表"""
    buckets = []  # 桶边界列表
    half_count = math.ceil(count / 2)  # 半侧桶数量
    distance = 1  # 初始距离
    buckets.append(middle)  # 添加中心值
    for i in range(half_count):  # 遍历半侧数量
        distance *= base  # 距离乘以基数
        buckets.append(middle + distance)  # 添加右侧桶边界
        buckets.append(max(0, middle - distance))  # 添加左侧桶边界（最小为0）
    return sorted(set(buckets))  # 去重并排序后返回


def generate_buckets(
    buckets_rule: List[str], default_buckets: List[float]  # 桶规则  # 默认桶列表
) -> List[float]:  # 根据规则生成桶边界列表
    """根据规则字符串生成直方图桶边界，支持tse（双侧指数）、default和custom三种规则"""
    if not buckets_rule:  # 如果规则为空
        buckets_rule = ["default"]  # 使用默认规则

    assert len(buckets_rule) > 0  # 断言规则列表非空
    rule = buckets_rule[0]  # 获取规则类型
    if rule == "tse":  # 双侧指数规则
        middle, base, count = buckets_rule[1:]  # 解析参数
        assert float(base) > 1.0, "Base must be greater than 1.0"  # 断言基数大于1
        return two_sides_exponential_buckets(float(middle), float(base), int(count))  # 生成双侧指数桶
    if rule == "default":  # 默认规则
        return sorted(set(default_buckets))  # 去重并排序默认桶
    assert rule == "custom"  # 断言为自定义规则
    return sorted(set([float(x) for x in buckets_rule[1:]]))  # 解析自定义桶值并去重排序


def exponential_buckets(start: float, width: float, length: int) -> List[float]:  # 起始值  # 宽度/基数  # 长度
    """生成指数增长的桶边界列表"""
    buckets = []  # 桶边界列表
    for i in range(length):  # 遍历长度
        buckets.append(start * (width**i))  # 计算第i个桶边界
    return buckets  # 返回桶边界列表
