# Chrome性能追踪文件合并工具，将多个分布式秩（TP/DP/PP/EP）的追踪文件合并为单个追踪文件
# 支持按秩信息排序、更新排序索引和添加秩标签，便于在Chrome追踪查看器中可视化分析
"""Merge Chrome trace files from multiple ranks (TP, DP, PP, EP) into a single trace."""  # 合并多个秩的Chrome追踪文件为单个追踪文件

import glob  # 导入文件路径模式匹配模块
import gzip  # 导入gzip压缩模块
import json  # 导入JSON解析模块
import logging  # 导入日志记录模块
import os  # 导入操作系统接口模块
import re  # 导入正则表达式模块
from typing import Any, Dict, List, Optional, Tuple  # 导入类型注解

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class ProfileMerger:  # 性能追踪文件合并器
    """Merge profile traces from all parallelism types: TP, DP, PP, EP."""  # 合并所有并行类型的性能追踪

    def __init__(self, output_dir: str, profile_id: str):  # 初始化合并器
        self.output_dir = output_dir  # 输出目录
        self.profile_id = profile_id  # 性能追踪ID
        self.merged_trace_path = os.path.join(  # 合并后的追踪文件路径
            output_dir, f"merged-{profile_id}.trace.json.gz"  # 使用gzip压缩的JSON格式
        )

        # Rank types in priority order (used for sorting and labeling)  # 按优先级排序的秩类型（用于排序和标签）
        self.rank_types = ["tp", "dp", "pp", "ep"]  # 张量并行、数据并行、流水线并行、专家并行

        # Sort index multipliers: DP (highest) > EP > PP > TP (lowest)  # 排序索引乘数：DP（最高）> EP > PP > TP（最低）
        # These ensure proper visual ordering in trace viewer  # 确保在追踪查看器中的正确视觉排序
        self.sort_index_multipliers = {  # 各秩类型的排序索引乘数
            "dp_rank": 100_000_000,  # 数据并行秩乘数
            "ep_rank": 1_000_000,  # 专家并行秩乘数
            "pp_rank": 10_000,  # 流水线并行秩乘数
            "tp_rank": 100,  # 张量并行秩乘数
        }

        # PID threshold for sort_index updates (only update for system PIDs < 1000)  # 排序索引更新的PID阈值（仅更新小于1000的系统PID）
        self.pid_sort_index_threshold = 1000  # PID排序索引阈值

    def merge_chrome_traces(self) -> str:  # 合并所有秩的Chrome追踪文件为单个文件
        """Merge Chrome traces from all ranks into a single trace.  # 将所有秩的Chrome追踪合并为单个追踪

        Returns:  # 返回值
            Path to merged trace file.  # 合并后的追踪文件路径

        Raises:  # 异常
            ValueError: If no trace files found.  # 如果未找到追踪文件
        """
        trace_files = self._discover_trace_files()  # 发现所有待合并的追踪文件
        if not trace_files:  # 如果没有找到追踪文件
            raise ValueError(f"No trace files found for profile_id: {self.profile_id}")  # 抛出异常

        logger.info(f"Found {len(trace_files)} trace files to merge")  # 记录找到的追踪文件数量

        merged_trace = {"traceEvents": []}  # 初始化合并后的追踪数据结构
        all_device_properties = []  # 收集所有设备属性

        for trace_file in sorted(trace_files, key=self._get_rank_sort_key):  # 按秩排序遍历追踪文件
            rank_info = self._extract_rank_info(trace_file)  # 从文件名中提取秩信息
            logger.info(f"Processing {trace_file} with rank info: {rank_info}")  # 记录正在处理的文件

            output = self._handle_file(trace_file, rank_info)  # 处理单个追踪文件

            merged_trace["traceEvents"].extend(output["traceEvents"])  # 将事件扩展到合并结果中

            if "deviceProperties" in output:  # 如果输出包含设备属性
                all_device_properties.extend(output["deviceProperties"])  # 收集设备属性
                del output["deviceProperties"]  # 从输出中删除已收集的设备属性

            for key, value in output.items():  # 遍历输出中的其他字段
                if key != "traceEvents" and key not in merged_trace:  # 跳过traceEvents和已存在的字段
                    merged_trace[key] = value  # 添加到合并结果

        if all_device_properties:  # 如果收集到了设备属性
            merged_trace["deviceProperties"] = all_device_properties  # 将设备属性添加到合并结果

        with gzip.open(self.merged_trace_path, "wb") as f:  # 以gzip压缩写入模式打开输出文件
            f.write(json.dumps(merged_trace).encode("utf-8"))  # 将合并结果序列化并写入

        logger.info(f"Merged profile saved to: {self.merged_trace_path}")  # 记录保存路径
        logger.info(f"Total events merged: {len(merged_trace['traceEvents'])}")  # 记录合并的事件总数

        return self.merged_trace_path  # 返回合并后的文件路径

    def _discover_trace_files(self) -> List[str]:  # 发现匹配profile_id的追踪文件
        """Discover trace files matching profile_id (supports TP/DP/PP/EP formats)."""  # 发现匹配profile_id的追踪文件（支持TP/DP/PP/EP格式）
        patterns = [f"{self.profile_id}*.trace.json.gz"]  # 追踪文件的匹配模式

        trace_files = []  # 初始化追踪文件列表
        for pattern in patterns:  # 遍历所有匹配模式
            search_pattern = os.path.join(self.output_dir, pattern)  # 构造完整搜索路径
            trace_files.extend(glob.glob(search_pattern))  # 使用glob搜索匹配文件

        trace_files = [  # 过滤追踪文件
            f
            for f in trace_files
            if not f.endswith(f"merged-{self.profile_id}.trace.json.gz")  # 排除已合并的文件
            and not f.endswith("-memory.pickle")  # 排除内存pickle文件
            and "TP-" in f  # 仅保留包含TP-的文件
        ]
        trace_files = list(set(trace_files))  # 去重
        return trace_files  # 返回过滤后的文件列表

    def _extract_rank_info(self, filename: str) -> Dict[str, int]:  # 从文件名中提取秩信息
        """Extract rank info (TP/DP/PP/EP) from filename."""  # 从文件名中提取秩信息（TP/DP/PP/EP）
        basename = os.path.basename(filename)  # 获取文件基本名称
        rank_info = {}  # 初始化秩信息字典

        for rank_type in self.rank_types:  # 遍历所有秩类型
            match = re.search(rf"{rank_type.upper()}-(\d+)", basename)  # 在文件名中搜索秩编号
            if match:  # 如果找到匹配
                rank_info[f"{rank_type}_rank"] = int(match.group(1))  # 将秩编号存入字典

        return rank_info  # 返回秩信息字典

    def _create_rank_label(self, rank_info: Dict[str, int]) -> str:  # 创建秩标签字符串
        parts = []  # 初始化标签部分列表
        for rank_type in self.rank_types:  # 遍历所有秩类型
            rank_key = f"{rank_type}_rank"  # 构造秩键名
            if rank_key in rank_info:  # 如果秩信息中包含该键
                parts.append(f"{rank_type.upper()}{rank_info[rank_key]:02d}")  # 添加格式化的秩标签

        return f"[{'-'.join(parts)}]" if parts else "[Unknown]"  # 返回组合标签或Unknown

    def _handle_file(self, path: str, rank_info: Dict[str, int]) -> Dict[str, Any]:  # 处理单个追踪文件
        logger.info(f"Processing file: {path}")  # 记录正在处理的文件

        try:  # 尝试读取和解压追踪文件
            with gzip.open(path, "rt", encoding="utf-8") as f:  # 以文本模式打开gzip文件
                trace = json.load(f)  # 解析JSON内容

            output = {  # 构造输出字典，排除traceEvents
                key: value for key, value in trace.items() if key != "traceEvents"
            }
            output["traceEvents"] = self._process_events(  # 处理事件列表
                trace.get("traceEvents", []), rank_info  # 传入事件和秩信息
            )
            return output  # 返回处理后的输出

        except Exception as e:  # 捕获处理异常
            logger.error(f"Failed to process trace file {path}: {e}")  # 记录错误
            return {"traceEvents": []}  # 返回空事件列表

    def _process_events(  # 处理事件：更新排序索引并添加秩标签
        self, events: List[Dict], rank_info: Dict[str, int]  # 事件列表和秩信息
    ) -> List[Dict]:
        """Process events: update sort_index and add rank labels to PIDs."""  # 处理事件：更新sort_index并为PID添加秩标签
        rank_label = self._create_rank_label(rank_info)  # 创建秩标签字符串

        for event in events:  # 遍历所有事件
            if event.get("name") == "process_sort_index":  # 如果是排序索引事件
                pid = self._maybe_cast_int(event.get("pid"))  # 尝试将PID转为整数
                if pid is not None and pid < self.pid_sort_index_threshold:  # 如果PID有效且小于阈值
                    event["args"]["sort_index"] = self._calculate_sort_index(  # 计算并更新排序索引
                        rank_info, pid  # 传入秩信息和PID
                    )

            event["pid"] = f"{rank_label} {event['pid']}"  # 为PID添加秩标签前缀

        return events  # 返回处理后的事件列表

    def _calculate_sort_index(self, rank_info: Dict[str, int], pid: int) -> int:  # 根据秩信息和PID计算排序索引
        sort_index = pid  # 初始排序索引为PID值
        for rank_type, multiplier in self.sort_index_multipliers.items():  # 遍历秩类型和乘数
            sort_index += rank_info.get(rank_type, 0) * multiplier  # 累加秩编号乘以乘数
        return sort_index  # 返回计算后的排序索引

    def _get_rank_sort_key(self, path: str) -> Tuple[int, int, int, int]:  # 获取文件排序键（按DP/EP/PP/TP排序）
        rank_info = self._extract_rank_info(path)  # 提取秩信息
        return tuple(  # 返回排序键元组
            rank_info.get(f"{rank_type}_rank", 0)  # 获取各秩类型的编号，默认为0
            for rank_type in ["dp", "ep", "pp", "tp"]  # 按DP/EP/PP/TP顺序
        )

    def _maybe_cast_int(self, x) -> Optional[int]:  # 尝试将值转换为整数
        try:  # 尝试转换
            return int(x)  # 返回整数值
        except (ValueError, TypeError):  # 如果转换失败
            return None  # 返回None

    def get_merge_summary(self) -> Dict[str, Any]:  # 获取合并摘要信息
        if not os.path.exists(self.merged_trace_path):  # 如果合并文件不存在
            return {"error": "Merged trace file not found"}  # 返回错误信息

        try:  # 尝试读取合并后的追踪文件
            with gzip.open(self.merged_trace_path, "rt") as f:  # 以文本模式打开gzip文件
                merged_data = json.load(f)  # 解析JSON内容

            trace_files = self._discover_trace_files()  # 发现源追踪文件

            return {  # 返回合并摘要信息
                "merged_file": self.merged_trace_path,  # 合并文件路径
                "total_events": len(merged_data.get("traceEvents", [])),  # 事件总数
                "total_files": len(trace_files),  # 源文件总数
                "source_files": [os.path.basename(f) for f in trace_files],  # 源文件名列表
                "profile_id": self.profile_id,  # 追踪ID
                "device_properties_count": len(merged_data.get("deviceProperties", [])),  # 设备属性数量
            }
        except Exception as e:  # 捕获读取异常
            return {"error": f"Failed to read merged trace: {str(e)}"}  # 返回错误信息
