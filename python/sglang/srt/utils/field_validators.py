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
# API字段验证器模块
# 提供轻量级、可复用的热路径API字段验证函数
# 用于配合pydantic.PlainValidator使用，替代pydantic默认的逐元素遍历以降低请求延迟

"""Lightweight, reusable validators for hot-path API fields.

These are intended to be paired with ``pydantic.PlainValidator`` on
dataclass fields whose JSON shape is large or homogeneously typed, where
pydantic's default per-element walk has been measured to dominate
request latency.

Usage::

    from typing import Annotated, List, Optional, Union
    from pydantic import PlainValidator
    from sglang.srt.utils.field_validators import validate_optional_list_i64_1d_2d

    @dataclass
    class MyReq:
        input_ids: Annotated[
            Optional[Union[List[List[int]], List[int]]],
            PlainValidator(validate_optional_list_i64_1d_2d),
        ] = None
"""

from __future__ import annotations  # 延迟注解评估

from array import array  # 数组模块，用于int64范围检查
from typing import Any  # 任意类型注解


def validate_list_i64_1d(v: Any) -> list[int]:  # 验证类型为list[int]的一维int64列表
    """Validates type: list[int]"""
    # 验证类型：list[int]
    if v is None:  # 如果值为None
        raise ValueError("must not be None")  # 抛出异常：不能为None
    if not isinstance(v, list):  # 如果不是列表类型
        raise ValueError(f"must be list; got {type(v).__name__}")  # 抛出异常：必须是列表
    if not v:  # 如果是空列表
        return v  # 空列表直接返回
    if not isinstance(v[0], int):  # 如果第一个元素不是整数
        raise ValueError(f"elements must be int; got {type(v[0]).__name__}")  # 抛出异常：元素必须是int
    try:  # 尝试创建int64数组以验证范围
        array("q", v)  # "q"表示有符号long long（int64）
    except (TypeError, OverflowError) as e:  # 类型错误或溢出错误
        raise ValueError(f"contains non-int64 element: {e}") from None  # 抛出异常：包含非int64元素
    return v  # 验证通过，返回原值


def validate_optional_list_i64_1d_2d(  # 验证可选的一维或二维int64列表
    v: Any,
) -> list[int] | list[list[int]] | None:
    """Validates type: list[int] | list[list[int]] | None"""
    # 验证类型：list[int] | list[list[int]] | None
    if v is None:  # 如果值为None
        # Accept None
        # 接受None
        return v  # 直接返回None
    if not isinstance(v, list):  # 如果不是列表类型
        raise ValueError(f"must be list or null; got {type(v).__name__}")  # 抛出异常
    if not v:  # 如果是空列表
        # Accept empty list
        # 接受空列表
        return v  # 直接返回空列表
    if isinstance(v[0], int):  # 如果第一个元素是整数（一维列表）
        # Accept list[int]
        # 接受list[int]
        return validate_list_i64_1d(v)  # 使用一维验证器验证
    if isinstance(v[0], list):  # 如果第一个元素是列表（二维列表）
        # Accept list[list[int]]
        # 接受list[list[int]]
        for i, row in enumerate(v):  # 遍历每一行
            try:  # 尝试验证每一行
                validate_list_i64_1d(row)  # 使用一维验证器验证行
            except ValueError as e:  # 验证失败
                raise ValueError(f"row {i}: {e}") from None  # 抛出包含行号的异常
        return v  # 验证通过，返回原值
    raise ValueError(f"elements must be int or list; got {type(v[0]).__name__}")  # 抛出异常：元素类型无效
