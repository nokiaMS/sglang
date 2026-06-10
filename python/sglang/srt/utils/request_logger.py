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
# 请求日志记录器，用于记录推理请求的接收和完成信息
# 支持不同日志级别（0-3）控制输出详细程度，支持JSON和文本格式输出
from __future__ import annotations  # 启用延迟注解求值

import dataclasses  # 导入数据类工具模块
import logging  # 导入日志记录模块
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union  # 导入类型注解

from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.utils.log_utils import create_log_targets, log_json  # 导入日志目标创建和JSON日志工具

if TYPE_CHECKING:  # 仅在类型检查时导入
    import fastapi  # 导入FastAPI框架

    from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput  # 导入请求输入结构

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

_DEFAULT_WHITELISTED_HEADERS = ["x-smg-routing-key"]  # 默认白名单请求头
WHITELISTED_HEADERS = _DEFAULT_WHITELISTED_HEADERS + [  # 合并默认和环境变量中的白名单请求头
    h.lower() for h in envs.SGLANG_LOG_REQUEST_HEADERS.get()  # 从环境变量获取额外请求头并转为小写
]


def _extract_whitelisted_headers(  # 从请求中提取白名单内的请求头
    request: Optional["fastapi.Request"],  # FastAPI请求对象，可为None
) -> Optional[Dict[str, str]]:  # 返回白名单请求头字典或None
    if request is None:  # 如果请求为None
        return None  # 返回None
    return {h: v for h in WHITELISTED_HEADERS if (v := request.headers.get(h))}  # 提取白名单中存在的请求头


class RequestLogger:  # 请求日志记录器类
    def __init__(  # 初始化请求日志记录器
        self,
        log_requests: bool,  # 是否启用请求日志
        log_requests_level: int,  # 日志详细级别（0-3）
        log_requests_format: str,  # 日志格式（"json"或文本）
        log_requests_target: Optional[List[str]],  # 日志输出目标列表
    ):
        self.log_requests = log_requests  # 保存是否启用请求日志
        self.log_requests_level = log_requests_level  # 保存日志级别
        self.log_requests_format = log_requests_format  # 保存日志格式
        self.log_requests_target = log_requests_target  # 保存日志目标

        self.metadata: Tuple[Optional[int], Optional[Set[str]], Optional[Set[str]]] = (  # 元数据（最大长度、跳过字段名、输出跳过字段名）
            self._compute_metadata()  # 计算元数据
        )
        self.targets = self._setup_targets()  # 设置日志输出目标

        self.log_exceeded_ms = envs.SGLANG_LOG_REQUEST_EXCEEDED_MS.get()  # 超时日志阈值（毫秒）

    def _setup_targets(self) -> List[logging.Logger]:  # 设置日志输出目标
        return create_log_targets(  # 创建日志目标
            targets=self.log_requests_target, name_prefix=__name__  # 使用当前模块名作为前缀
        )

    def configure(  # 动态配置请求日志记录器
        self,
        log_requests: Optional[bool] = None,  # 是否启用请求日志
        log_requests_level: Optional[int] = None,  # 日志级别
        log_requests_format: Optional[str] = None,  # 日志格式
        log_requests_target: Optional[List[str]] = None,  # 日志目标
    ) -> None:
        if log_requests is not None:  # 如果提供了log_requests参数
            self.log_requests = log_requests  # 更新启用标志
        if log_requests_level is not None:  # 如果提供了log_requests_level参数
            self.log_requests_level = log_requests_level  # 更新日志级别
        if log_requests_format is not None:  # 如果提供了log_requests_format参数
            self.log_requests_format = log_requests_format  # 更新日志格式
        if log_requests_target is not None:  # 如果提供了log_requests_target参数
            self.log_requests_target = log_requests_target  # 更新日志目标

        self.metadata = self._compute_metadata()  # 重新计算元数据
        self.targets = self._setup_targets()  # 重新设置日志目标

    def log_received_request(  # 记录接收到的推理请求
        self,
        obj: Union["GenerateReqInput", "EmbeddingReqInput"],  # 请求输入对象
        tokenizer: Any = None,  # 分词器（可选，用于解码input_ids）
        request: Optional["fastapi.Request"] = None,  # FastAPI请求对象（可选）
    ) -> None:
        if not self.log_requests:  # 如果未启用请求日志
            return  # 直接返回

        max_length, skip_names, _ = self.metadata  # 获取最大长度和跳过字段名
        headers = _extract_whitelisted_headers(request)  # 提取白名单请求头
        if self.log_requests_format == "json":  # 如果使用JSON格式
            log_data = {  # 构造JSON日志数据
                "rid": obj.rid,  # 请求ID
                "obj": _transform_data_for_logging(obj, max_length, skip_names),  # 转换后的请求数据
            }
            if headers:  # 如果有白名单请求头
                log_data["headers"] = headers  # 添加请求头
            log_json(self.targets, "request.received", log_data)  # 以JSON格式记录日志
        else:  # 使用文本格式
            headers_str = f", headers={headers}" if headers else ""  # 格式化请求头字符串
            self._log(  # 记录文本日志
                f"Receive: obj={_dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}{headers_str}"  # 截断后的请求字符串
            )

        # FIXME: This is a temporary fix to get the text from the input ids.  # 临时修复：从input_ids获取文本
        # We should remove this once we have a proper way.  # 找到合适方法后应移除此代码
        if (  # 如果满足以下条件
            self.log_requests_level >= 2  # 日志级别至少为2
            and obj.text is None  # 请求中没有文本
            and obj.input_ids is not None  # 但有input_ids
            and tokenizer is not None  # 且有分词器
        ):
            if obj.input_ids and isinstance(obj.input_ids[0], list):  # 如果input_ids是嵌套列表
                # Prefill node warmup while PD disaggregated.  # PD分离模式下的预填充节点预热
                decoded = [  # 解码每个子列表
                    tokenizer.decode(_input_ids, skip_special_tokens=False)  # 保留特殊token解码
                    for _input_ids in obj.input_ids  # 遍历每个input_ids子列表
                ]
            else:  # 单个input_ids列表
                decoded = tokenizer.decode(obj.input_ids, skip_special_tokens=False)  # 保留特殊token解码
            obj.text = decoded  # 将解码结果赋值给text字段

    def log_openai_received_request(  # 记录接收到的OpenAI格式请求
        self,
        obj: Any,  # OpenAI请求对象
        request: Optional["fastapi.Request"] = None,  # FastAPI请求对象（可选）
    ) -> None:
        """Log the raw OpenAI request payload before request adaptation/tokenization."""  # 在请求适配/分词前记录原始OpenAI请求载荷
        max_length, _, _ = self.metadata  # 获取最大长度
        max_length = max_length if max_length is not None else 2048  # 默认最大长度2048
        headers = _extract_whitelisted_headers(request)  # 提取白名单请求头

        if hasattr(obj, "model_dump"):  # 如果对象有model_dump方法（Pydantic模型）
            obj_to_log = obj.model_dump(exclude_none=True)  # 导出非None字段
        else:  # 否则直接使用对象
            obj_to_log = obj  # 直接使用原始对象

        if self.log_requests_format == "json":  # 如果使用JSON格式
            log_data = {  # 构造JSON日志数据
                "obj": _transform_data_for_logging(obj_to_log, max_length=max_length),  # 转换后的数据
            }
            if headers:  # 如果有白名单请求头
                log_data["headers"] = headers  # 添加请求头
            log_json(self.targets, "request.received.openai", log_data)  # 以JSON格式记录日志
        else:  # 使用文本格式
            headers_str = f", headers={headers}" if headers else ""  # 格式化请求头字符串
            self._log(  # 记录文本日志
                f"Receive OpenAI: obj={_dataclass_to_string_truncated(obj_to_log, max_length)}{headers_str}"  # 截断后的请求字符串
            )

    def log_finished_request(  # 记录已完成的推理请求
        self,
        obj: Union["GenerateReqInput", "EmbeddingReqInput"],  # 请求输入对象
        out: Any,  # 推理输出结果
        request: Optional["fastapi.Request"] = None,  # FastAPI请求对象（可选）
    ) -> None:
        if not self.log_requests:  # 如果未启用请求日志
            return  # 直接返回

        e2e_latency_ms = out["meta_info"].get("e2e_latency", 0) * 1000  # 计算端到端延迟（毫秒）
        if self.log_exceeded_ms > 0 and e2e_latency_ms < self.log_exceeded_ms:  # 如果延迟未超过阈值
            return  # 跳过记录

        max_length, skip_names, out_skip_names = self.metadata  # 获取最大长度和跳过字段名
        headers = _extract_whitelisted_headers(request)  # 提取白名单请求头
        if self.log_requests_format == "json":  # 如果使用JSON格式
            log_data = {  # 构造JSON日志数据
                "rid": obj.rid,  # 请求ID
                "obj": _transform_data_for_logging(obj, max_length, skip_names),  # 转换后的请求数据
            }
            if headers:  # 如果有白名单请求头
                log_data["headers"] = headers  # 添加请求头
            log_data["out"] = _transform_data_for_logging(  # 添加转换后的输出数据
                out, max_length, out_skip_names  # 使用输出跳过字段名
            )
            log_json(self.targets, "request.finished", log_data)  # 以JSON格式记录日志
        else:  # 使用文本格式
            obj_str = _dataclass_to_string_truncated(  # 截断请求数据字符串
                obj, max_length, skip_names=skip_names  # 使用输入跳过字段名
            )
            out_str = f", out={_dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"  # 截断输出数据字符串
            headers_str = f", headers={headers}" if headers else ""  # 格式化请求头字符串
            self._log(f"Finish: obj={obj_str}{headers_str}{out_str}")  # 记录完成日志

    def _compute_metadata(  # 根据日志级别计算元数据（最大长度和跳过字段名）
        self,
    ) -> Tuple[Optional[int], Optional[Set[str]], Optional[Set[str]]]:
        max_length: Optional[int] = None  # 最大长度，None表示不限制
        skip_names: Optional[Set[str]] = None  # 输入跳过字段名集合
        out_skip_names: Optional[Set[str]] = None  # 输出跳过字段名集合
        if self.log_requests:  # 如果启用了请求日志
            if self.log_requests_level == 0:  # 级别0：仅记录元信息
                max_length = 1 << 30  # 设置很大的最大长度
                skip_names = {  # 跳过大体积字段
                    "text",
                    "input_ids",
                    "input_embeds",
                    "image_data",
                    "audio_data",
                    "video_data",
                    "mm_data_mooncake",
                    "lora_path",
                    "sampling_params",
                }
                out_skip_names = {"text", "output_ids", "embedding"}  # 输出跳过字段
            elif self.log_requests_level == 1:  # 级别1：记录sampling_params但不记录数据
                max_length = 1 << 30  # 设置很大的最大长度
                skip_names = {  # 跳过大体积字段（保留sampling_params）
                    "text",
                    "input_ids",
                    "input_embeds",
                    "image_data",
                    "audio_data",
                    "video_data",
                    "mm_data_mooncake",
                    "lora_path",
                }
                out_skip_names = {"text", "output_ids", "embedding"}  # 输出跳过字段
            elif self.log_requests_level == 2:  # 级别2：记录所有字段但截断长度
                max_length = 2048  # 最大长度2048
            elif self.log_requests_level == 3:  # 级别3：记录所有字段不截断
                max_length = 1 << 30  # 设置很大的最大长度
            else:  # 无效的日志级别
                raise ValueError(  # 抛出异常
                    f"Invalid --log-requests-level: {self.log_requests_level=}"  # 显示无效级别
                )
        return max_length, skip_names, out_skip_names  # 返回元数据

    def _log(self, msg: str) -> None:  # 向所有日志目标发送消息
        for target in self.targets:  # 遍历所有日志目标
            target.info(msg)  # 以INFO级别记录消息


# TODO unify this w/ `_transform_data_for_logging` if we find performance enough  # 如果性能足够，将此函数与_transform_data_for_logging统一
def _dataclass_to_string_truncated(  # 将数据类/对象转为截断后的字符串表示
    data: Any, max_length: int = 2048, skip_names: Optional[Set[str]] = None  # 数据、最大长度、跳过字段名
) -> str:
    if skip_names is None:  # 如果没有提供跳过字段名
        skip_names = set()  # 使用空集合
    if isinstance(data, str):  # 如果数据是字符串
        if len(data) > max_length:  # 如果字符串超过最大长度
            half_length = max_length // 2  # 计算半长度
            return f"{repr(data[:half_length])} ... {repr(data[-half_length:])}"  # 截断前后各半
        else:  # 字符串未超过最大长度
            return f"{repr(data)}"  # 返回完整字符串的repr
    elif isinstance(data, (list, tuple)):  # 如果数据是列表或元组
        if len(data) > max_length:  # 如果超过最大长度
            half_length = max_length // 2  # 计算半长度
            return str(data[:half_length]) + " ... " + str(data[-half_length:])  # 截断前后各半
        else:  # 未超过最大长度
            return str(data)  # 返回完整字符串
    elif isinstance(data, dict):  # 如果数据是字典
        return (  # 递归处理字典值
            "{"
            + ", ".join(
                f"'{k}': {_dataclass_to_string_truncated(v, max_length)}"  # 递归截断值
                for k, v in data.items()
                if k not in skip_names  # 跳过指定字段
            )
            + "}"
        )
    elif dataclasses.is_dataclass(data):  # 如果数据是dataclass
        fields = dataclasses.fields(data)  # 获取所有字段
        return (  # 递归处理字段值
            f"{data.__class__.__name__}("
            + ", ".join(
                f"{f.name}={_dataclass_to_string_truncated(getattr(data, f.name), max_length)}"  # 递归截断字段值
                for f in fields
                if f.name not in skip_names  # 跳过指定字段
            )
            + ")"
        )
    else:  # 其他类型
        return str(data)  # 直接转为字符串


def _transform_data_for_logging(  # 将数据转换为适合JSON日志的格式（截断长数据）
    data: Any, max_length: int = 2048, skip_names: Optional[Set[str]] = None  # 数据、最大长度、跳过字段名
) -> Any:
    if skip_names is None:  # 如果没有提供跳过字段名
        skip_names = set()  # 使用空集合
    if isinstance(data, str):  # 如果数据是字符串
        if len(data) > max_length:  # 如果字符串超过最大长度
            half_length = max_length // 2  # 计算半长度
            return data[:half_length] + "..." + data[-half_length:]  # 截断前后各半
        return data  # 返回原字符串
    elif isinstance(data, (list, tuple)):  # 如果数据是列表或元组
        if len(data) > max_length:  # 如果超过最大长度
            half_length = max_length // 2  # 计算半长度
            return list(data[:half_length]) + ["..."] + list(data[-half_length:])  # 截断前后各半
        return [_transform_data_for_logging(v, max_length) for v in data]  # 递归转换每个元素
    elif isinstance(data, dict):  # 如果数据是字典
        return {  # 递归转换字典值
            k: _transform_data_for_logging(v, max_length)
            for k, v in data.items()
            if k not in skip_names  # 跳过指定字段
        }
    elif dataclasses.is_dataclass(data):  # 如果数据是dataclass
        fields = dataclasses.fields(data)  # 获取所有字段
        return {  # 递归转换字段值为字典
            f.name: _transform_data_for_logging(getattr(data, f.name), max_length)
            for f in fields
            if f.name not in skip_names  # 跳过指定字段
        }
    elif isinstance(data, (int, float, bool, type(None))):  # 如果是基本标量类型
        return data  # 直接返回
    else:  # 其他类型
        return str(data)  # 转为字符串
