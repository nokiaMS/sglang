# 请求指标导出器模块
# 本模块提供请求级别性能指标的导出功能
# 支持将指标写入文件，并管理导出器实例的生命周期

import asyncio  # 导入异步IO模块
import dataclasses  # 导入数据类工具
import json  # 导入JSON模块
import logging  # 导入日志模块
import os  # 导入操作系统模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法
from datetime import datetime  # 导入日期时间
from typing import List, Optional, Union  # 导入类型提示

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX  # 导入健康检查请求ID前缀
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput  # 导入请求输入结构
from sglang.srt.server_args import ServerArgs  # 导入服务器参数

logger = logging.getLogger(__name__)  # 创建日志记录器

# Fields that should always be excluded from request parameters
# because they contain non-JSON-serializable objects (e.g., ImageData, tensors)
ALWAYS_EXCLUDE_FIELDS = {"image_data", "video_data", "audio_data", "input_embeds"}  # 始终排除的字段（不可JSON序列化）


class RequestMetricsExporter(ABC):  # 请求指标导出器抽象基类
    """Abstract base class for exporting request-level performance metrics to a data destination."""

    def __init__(  # 初始化请求指标导出器
        self,
        server_args: ServerArgs,  # 服务器参数
        obj_skip_names: Optional[set[str]],  # 对象中跳过的字段名
        out_skip_names: Optional[set[str]],  # 输出中跳过的字段名
    ):
        self.server_args = server_args  # 保存服务器参数
        self.obj_skip_names = obj_skip_names or set()  # 保存对象跳过字段
        self.out_skip_names = out_skip_names or set()  # 保存输出跳过字段

    def _format_output_data(  # 格式化请求级输出数据
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], out_dict: dict  # 请求对象和输出字典
    ) -> dict:
        """Format request-level output data containing performance metrics. This method
        should be called prior to writing the data record with `self.write_record()`."""

        request_params = {}  # 请求参数字典
        for field in dataclasses.fields(obj):  # 遍历请求对象的所有字段
            field_name = field.name  # 获取字段名
            # Skip fields in obj_skip_names or fields that are always excluded (not JSON serializable)
            if (  # 过滤跳过字段
                field_name not in self.obj_skip_names  # 不在对象跳过列表中
                and field_name not in ALWAYS_EXCLUDE_FIELDS  # 不在始终排除列表中
            ):
                value = getattr(obj, field_name)  # 获取字段值
                # Convert to serializable format
                if value is not None:  # 如果值不为None
                    request_params[field_name] = value  # 添加到参数字典

        meta_info = out_dict.get("meta_info", {})  # 获取元信息
        filtered_out_meta_info = {  # 过滤元信息
            k: v for k, v in meta_info.items() if k not in self.out_skip_names  # 排除跳过字段
        }

        request_output_data = {  # 构建请求输出数据
            "request_parameters": json.dumps(request_params),  # 请求参数JSON
            **filtered_out_meta_info,  # 过滤后的元信息
        }
        return request_output_data  # 返回请求数据

    @abstractmethod
    async def write_record(  # 写入数据记录（抽象方法）
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], out_dict: dict  # 请求对象和输出字典
    ):
        """Write a data record corresponding to a single request, containing performance metric data."""
        pass  # 子类实现


class FileRequestMetricsExporter(RequestMetricsExporter):  # 文件请求指标导出器
    """Lightweight `RequestMetricsExporter` implementation that writes records to files on disk.

    Records are written to files in the directory specified by `--export-metrics-to-file-dir`
    server launch flag. File names are of the form `"sglang-request-metrics-{hour_suffix}.log"`.
    """

    def __init__(  # 初始化文件请求指标导出器
        self,
        server_args: ServerArgs,  # 服务器参数
        obj_skip_names: Optional[set[str]],  # 对象中跳过的字段名
        out_skip_names: Optional[set[str]],  # 输出中跳过的字段名
    ):
        super().__init__(server_args, obj_skip_names, out_skip_names)  # 调用父类初始化
        self.export_dir = getattr(server_args, "export_metrics_to_file_dir")  # 获取导出目录
        os.makedirs(self.export_dir, exist_ok=True)  # 创建导出目录

        # File handler state management
        self._current_file_handler = None  # 当前文件处理器
        self._current_file_lock = asyncio.Lock()  # 文件操作异步锁
        self._current_hour_suffix = None  # 当前小时后缀

    def _ensure_file_handler(self, hour_suffix: str):  # 确保文件处理器已打开
        """Ensure the file handler is open for the current hour suffix."""
        if self._current_hour_suffix != hour_suffix:  # 如果小时后缀变化
            # Close previous file handler if it exists
            if self._current_file_handler is not None:  # 如果有旧的文件处理器
                try:  # 尝试关闭
                    self._current_file_handler.close()  # 关闭文件
                except Exception as e:  # 捕获异常
                    logger.warning(f"Failed to close previous file handler: {e}")  # 记录警告

            # Open new file handler
            log_filename = f"sglang-request-metrics-{hour_suffix}.log"  # 日志文件名
            log_filepath = os.path.join(self.export_dir, log_filename)  # 日志文件路径

            try:  # 尝试打开文件
                self._current_file_handler = open(log_filepath, "a", encoding="utf-8")  # 追加模式打开
                self._current_hour_suffix = hour_suffix  # 更新当前小时后缀
            except Exception as e:  # 捕获异常
                logger.error(f"Failed to open log file {log_filepath}: {e}")  # 记录错误
                self._current_file_handler = None  # 重置文件处理器
                self._current_hour_suffix = None  # 重置小时后缀
                raise  # 重新抛出异常

    def close(self):  # 关闭文件处理器
        """Close the current file handler."""
        if self._current_file_handler is not None:  # 如果文件处理器存在
            try:  # 尝试关闭
                self._current_file_handler.close()  # 关闭文件
            except Exception as e:  # 捕获异常
                logger.warning(f"Failed to close file handler: {e}")  # 记录警告
            finally:  # 最终
                self._current_file_handler = None  # 重置文件处理器
                self._current_hour_suffix = None  # 重置小时后缀

    async def write_record(  # 异步写入数据记录
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], out_dict: dict  # 请求对象和输出字典
    ):
        # Do not log health check requests, since they don't represent real user requests.
        if isinstance(obj.rid, str) and HEALTH_CHECK_RID_PREFIX in obj.rid:  # 如果是健康检查请求
            return  # 跳过

        try:  # 尝试写入
            # Get the log file path for the current time.
            current_time = datetime.now()  # 获取当前时间
            hour_suffix = current_time.strftime("%Y%m%d_%H")  # 生成小时后缀

            async with self._current_file_lock:  # 获取文件锁
                # Ensure correct file handler is open for current hour
                self._ensure_file_handler(hour_suffix)  # 确保文件处理器正确

                if self._current_file_handler is None:  # 如果文件处理器为空
                    return  # 返回

                metrics_data = self._format_output_data(obj, out_dict)  # 格式化输出数据

                def write_file():  # 文件写入函数
                    json.dump(metrics_data, self._current_file_handler)  # 写入JSON
                    self._current_file_handler.write("\n")  # 写入换行
                    self._current_file_handler.flush()  # 刷新缓冲区

                await asyncio.to_thread(write_file)  # 在线程中执行写入
        except Exception as e:  # 捕获异常
            logger.exception(f"Failed to write perf metrics to file: {e}")  # 记录异常


class RequestMetricsExporterManager:  # 请求指标导出器管理器
    """Manager class for creating and managing RequestMetricsExporter instances."""

    def __init__(  # 初始化请求指标导出器管理器
        self,
        server_args: ServerArgs,  # 服务器参数
        obj_skip_names: Optional[set[str]] = None,  # 对象中跳过的字段名
        out_skip_names: Optional[set[str]] = None,  # 输出中跳过的字段名
    ):
        self.server_args = server_args  # 保存服务器参数
        self.obj_skip_names = obj_skip_names or set()  # 保存对象跳过字段
        self.out_skip_names = out_skip_names or set()  # 保存输出跳过字段
        self._exporters: List[RequestMetricsExporter] = []  # 导出器列表
        self._create_exporters()  # 创建导出器

    def _create_exporters(self) -> None:  # 创建导出器实例
        """Create and configure RequestMetricsExporter instances based on server args."""
        # Create standard exporters
        self._exporters.extend(  # 添加标准导出器
            create_request_metrics_exporters(
                self.server_args, self.obj_skip_names, self.out_skip_names
            )
        )

        # Import additional RequestMetricsExporter from private fork if available; skip otherwise.
        try:  # 尝试导入私有导出器
            from sglang.private.managers.request_metrics_exporter_factory import (  # 导入私有工厂
                create_private_request_metrics_exporters,
            )

            self._exporters.extend(  # 添加私有导出器
                create_private_request_metrics_exporters(
                    self.server_args, self.obj_skip_names, self.out_skip_names
                )
            )
        except ImportError:  # 如果导入失败
            pass  # 跳过

    def exporter_enabled(self) -> bool:  # 检查是否有导出器启用
        """Return true if at least one RequestMetricsExporter is enabled."""
        return len(self._exporters) > 0  # 返回导出器列表是否非空

    async def write_record(self, obj, out_dict: dict) -> None:  # 异步写入记录到所有导出器
        """Write a record using all configured exporters."""
        for exporter in self._exporters:  # 遍历所有导出器
            await exporter.write_record(obj, out_dict)  # 写入记录


def create_request_metrics_exporters(  # 创建请求指标导出器
    server_args: ServerArgs,  # 服务器参数
    obj_skip_names: Optional[set[str]] = None,  # 对象中跳过的字段名
    out_skip_names: Optional[set[str]] = None,  # 输出中跳过的字段名
) -> List[RequestMetricsExporter]:
    """Create and configure `RequestMetricsExporter`s based on server args."""
    metrics_exporters = []  # 导出器列表

    if server_args.export_metrics_to_file:  # 如果启用文件导出
        metrics_exporters.append(  # 添加文件导出器
            FileRequestMetricsExporter(server_args, obj_skip_names, out_skip_names)
        )

    return metrics_exporters  # 返回导出器列表
