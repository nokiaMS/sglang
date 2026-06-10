# HTTP响应中JSON序列化的工具模块
# 提供统一的JSON序列化选项和响应类，确保各端点的序列化行为一致
"""Utilities for JSON serialization in HTTP responses."""

from typing import Any  # 导入类型提示

import orjson  # 导入高性能JSON库
from fastapi.responses import Response  # 导入FastAPI响应基类

# Keep response serialization behavior consistent across endpoints:
# - Support non-string dictionary keys used in some metadata payloads.
# - Support numpy scalars/arrays without pre-conversion.
ORJSON_RESPONSE_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY  # ORJSON序列化选项：支持非字符串键和NumPy序列化


def dumps_json(content: Any) -> bytes:  # 使用SGLang的ORJSON选项将内容序列化为JSON字节
    """Serialize content to JSON bytes using SGLang's ORJSON options."""
    return orjson.dumps(content, option=ORJSON_RESPONSE_OPTIONS)  # 使用指定选项序列化


class SGLangORJSONResponse(Response):  # 使用SGLang特定序列化选项的ORJSON响应类
    """ORJSON response with SGLang-specific serialization options."""

    media_type = "application/json"  # 媒体类型为JSON

    def render(self, content: Any) -> bytes:  # 将内容渲染为JSON字节
        return dumps_json(content)


def orjson_response(content: Any, status_code: int = 200) -> Response:  # 创建具有稳定ORJSON序列化选项的JSON响应
    """Create a JSON response with stable ORJSON serialization options."""
    return SGLangORJSONResponse(content=content, status_code=status_code)  # 返回ORJSON响应对象
