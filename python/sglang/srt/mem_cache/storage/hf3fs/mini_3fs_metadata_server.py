# HF3FS元数据服务器与客户端模块
# 本文件实现了HF3FS存储系统的元数据管理服务，包括：
# 1. RankMetadata - 单个rank的元数据管理（键值到页索引的映射、页面分配与释放）
# 2. GlobalMetadataState - 全局元数据状态管理（持久化、定时保存）
# 3. Hf3fsMetadataServer - 基于FastAPI的元数据HTTP服务器
# 4. Hf3fsGlobalMetadataClient - 远程HTTP元数据客户端
# 5. Hf3fsLocalMetadataClient - 本地内存元数据客户端（无需独立服务器）

import argparse  # 导入命令行参数解析模块 # 命令行参数解析库
import atexit  # 导入退出处理模块 # 退出处理库
import json  # 导入JSON解析模块 # JSON库
import logging  # 导入日志模块 # 日志库
import threading  # 导入线程模块 # 线程库
from collections import OrderedDict  # 导入有序字典 # 有序字典
from pathlib import Path  # 导入路径工具 # 路径工具
from typing import Dict, List, Optional, Tuple  # 导入类型注解 # 类型注解

import orjson  # 导入orjson高性能JSON库 # 高性能JSON库
import requests  # 导入HTTP请求库 # HTTP请求库
from fastapi import FastAPI, HTTPException, Request, Response  # 导入FastAPI相关组件 # FastAPI组件
from fastapi.responses import ORJSONResponse  # 导入ORJSON响应类 # ORJSON响应
from requests.adapters import HTTPAdapter  # 导入HTTP适配器 # HTTP适配器
from urllib3.util.retry import Retry  # 导入重试策略 # 重试策略

from sglang.srt.mem_cache.hicache_storage import PoolName  # 导入池名称枚举 # 池名称枚举
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import Hf3fsMetadataInterface  # 导入元数据接口 # 元数据接口

# --- Configuration ---  # --- 配置 --- # 配置部分
logging.basicConfig(  # 配置日志格式 # 配置日志
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"  # 日志级别INFO，格式包含时间、级别、消息 # 日志级别和格式
)


# --- Data Models ---  # --- 数据模型 --- # 数据模型部分
class RankMetadata:  # 单个rank的元数据管理类 # Rank元数据管理类
    """Holds all metadata for a single rank."""  # 保存单个rank的所有元数据 # 保存单个rank的所有元数据

    def __init__(self, num_pages: int):  # 初始化Rank元数据
        self.lock = threading.Lock()  # 线程锁，保护并发访问 # 线程锁
        self.num_pages = num_pages  # 总页面数 # 总页面数
        self.free_pages: List[int] = list(range(num_pages))  # 空闲页面列表 # 空闲页面列表
        self.key_to_index: OrderedDict[str, int] = OrderedDict()  # 键到页索引的有序映射 # 键到页索引映射
        # Todo: Support multi files for HF3FS  # 待办：支持HF3FS多文件 # 待办：支持多文件

    def exists_keys(self, keys: List[str]) -> List[bool]:  # 检查键是否存在于元数据中
        """Check if keys exist in metadata."""  # 检查键是否存在于元数据中 # 检查键是否存在
        with self.lock:  # 获取锁 # 获取锁
            return [key in self.key_to_index for key in keys]  # 返回每个键的存在状态 # 返回存在状态列表

    def reserve_and_allocate_page_indices(  # 预留并分配页面索引
        self, keys: List[Tuple[str, str]]  # 键列表，每个元素为(键, 前缀键)元组 # 键列表
    ) -> List[Tuple[bool, int]]:  # 返回(是否已存在, 页索引)元组列表 # 返回分配结果
        """Reserve and allocate page indices for keys."""  # 为键预留并分配页面索引 # 预留并分配页索引
        with self.lock:  # 获取锁 # 获取锁
            results = [None] * len(keys)  # 初始化结果列表 # 初始化结果
            new_keys_to_process = []  # 需要处理的新键列表 # 新键列表

            for i, (key, prefix_key) in enumerate(keys):  # 遍历所有键 # 遍历键
                if key in self.key_to_index:  # 键已存在 # 键已存在
                    results[i] = (True, self.key_to_index[key])  # 记录已存在标志和页索引 # 记录已存在
                    self.key_to_index.move_to_end(key)  # 将键移到有序字典末尾（LRU更新） # 更新LRU顺序
                else:  # 键不存在 # 键不存在
                    new_keys_to_process.append((i, key, prefix_key))  # 加入新键处理列表 # 加入新键列表

            # Todo: Implementing data eviction logic after HiCache supports prefix information pass-through  # 待办：HiCache支持前缀信息透传后实现数据驱逐逻辑 # 待办：实现驱逐逻辑
            for i, key, prefix_key in new_keys_to_process:  # 处理新键 # 处理新键
                if len(self.free_pages) > 0:  # 有空闲页面 # 有空闲页面
                    page_index = self.free_pages.pop()  # 从空闲列表弹出一个页面 # 弹出空闲页面
                else:  # 无空闲页面，需要驱逐 # 无空闲页面
                    page_index = self.key_to_index.popitem(last=False)[1]  # 驱逐最早（FIFO）的键值对，获取其页索引 # 驱逐最早的项

                results[i] = (False, page_index)  # 记录新分配标志和页索引 # 记录新分配

            return results  # 返回分配结果 # 返回结果

    def confirm_write(  # 确认写入操作并释放页面
        self,
        written_keys_to_confirm: List[Tuple[str, int]],  # 已写入的键和页索引列表 # 已写入键列表
        pages_to_release: List[int],  # 需要释放的页索引列表 # 需释放页面列表
    ) -> None:
        """Confirm write operations and release pages."""  # 确认写入操作并释放页面 # 确认写入并释放页面
        with self.lock:  # 获取锁 # 获取锁
            for key, page_index in written_keys_to_confirm:  # 遍历已写入的键 # 遍历已写入键
                self.key_to_index[key] = page_index  # 更新键到页索引的映射 # 更新映射
                self.key_to_index.move_to_end(key)  # 将键移到有序字典末尾（LRU更新） # 更新LRU顺序

            for page_index in pages_to_release:  # 遍历需要释放的页面 # 遍历释放页面
                if page_index not in self.free_pages:  # 页面不在空闲列表中 # 页面不在空闲列表
                    self.free_pages.append(page_index)  # 将页面加入空闲列表 # 加入空闲列表

    def delete_keys(self, keys: List[str]) -> int:  # 删除键并返回已删除数量
        """Delete keys and return count of deleted keys."""  # 删除键并返回已删除键的数量 # 删除键并返回数量
        with self.lock:  # 获取锁 # 获取锁
            count = 0  # 已删除计数 # 删除计数
            for key in keys:  # 遍历要删除的键 # 遍历键
                if key in self.key_to_index:  # 键存在 # 键存在
                    page_index = self.key_to_index.pop(key)  # 移除键并获取页索引 # 移除键
                    if page_index not in self.free_pages:  # 页面不在空闲列表中 # 页面不在空闲列表
                        self.free_pages.append(page_index)  # 将页面加入空闲列表 # 加入空闲列表
                    count += 1  # 递增删除计数 # 递增计数
            return count  # 返回删除数量 # 返回数量

    def clear_all(self) -> None:  # 清除所有元数据
        """Clear all metadata."""  # 清除所有元数据 # 清除所有元数据
        with self.lock:  # 获取锁 # 获取锁
            self.free_pages = list(range(self.num_pages))  # 重置空闲页面列表 # 重置空闲页面
            self.key_to_index.clear()  # 清空键到页索引的映射 # 清空映射

    def get_page_indices(self, keys: List[str]) -> List[Optional[int]]:  # 获取键对应的页面索引
        """Get page indices for keys."""  # 获取键的页面索引 # 获取键的页索引
        with self.lock:  # 获取锁 # 获取锁
            results = []  # 结果列表 # 结果列表
            for key in keys:  # 遍历键 # 遍历键
                if key in self.key_to_index:  # 键存在 # 键存在
                    results.append(self.key_to_index[key])  # 添加页索引 # 添加页索引
                    self.key_to_index.move_to_end(key)  # 将键移到有序字典末尾（LRU更新） # 更新LRU顺序
                else:  # 键不存在 # 键不存在
                    results.append(None)  # 添加None表示未找到 # 添加None
            return results  # 返回结果列表 # 返回结果


class GlobalMetadataState:  # 全局元数据状态管理类 # 全局元数据状态
    """Manages the state for all ranks and persistence."""  # 管理所有rank的状态和持久化 # 管理状态和持久化

    def __init__(self, persistence_path: Optional[str], save_interval: int):  # 初始化全局状态
        self.global_lock = threading.RLock()  # 全局可重入锁 # 全局锁
        self.ranks: Dict[str, RankMetadata] = {}  # rank键到RankMetadata的映射 # rank字典
        self.persistence_path = Path(persistence_path) if persistence_path else None  # 持久化文件路径 # 持久化路径
        self.save_interval = save_interval  # 保存间隔（秒） # 保存间隔
        self.save_timer: Optional[threading.Timer] = None  # 定时保存计时器 # 保存计时器
        self.is_shutting_down = False  # 是否正在关闭 # 关闭标志

    def load_from_disk(self):  # 从磁盘加载持久化状态
        if not self.persistence_path or not self.persistence_path.exists():  # 检查持久化路径是否存在 # 检查路径存在
            logging.info("Persistence file not found. Starting with a clean state.")  # 未找到文件，使用干净状态 # 未找到文件
            return  # 直接返回 # 返回

        logging.info(f"Loading state from {self.persistence_path}")  # 记录加载信息 # 记录加载日志
        try:  # 尝试加载 # 尝试
            with open(self.persistence_path, "r") as f:  # 打开持久化文件 # 打开文件
                persisted_data = json.load(f)  # 加载JSON数据 # 加载JSON

            with self.global_lock:  # 获取全局锁 # 获取全局锁
                for key_str, data in persisted_data.items():  # 遍历持久化数据 # 遍历数据
                    if ":" not in key_str:  # 兼容旧格式 # 兼容旧格式
                        key_str = f"{key_str}:kv"  # For backward compatibility  # 向后兼容，添加默认命名空间 # 添加默认命名空间
                    num_pages = data["num_pages"]  # 获取页面数 # 获取页面数
                    rank_meta = RankMetadata(num_pages)  # 创建RankMetadata实例 # 创建元数据实例
                    rank_meta.free_pages = data["free_pages"]  # 恢复空闲页面列表 # 恢复空闲页面
                    rank_meta.key_to_index = OrderedDict(data["key_to_index"])  # 恢复键到页索引映射 # 恢复映射
                    self.ranks[key_str] = rank_meta  # 存入ranks字典 # 存入字典
                logging.info(  # 记录加载成功信息 # 记录成功日志
                    f"Successfully loaded metadata for {len(self.ranks)} ranks."  # 加载的rank数量 # rank数量
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:  # 捕获JSON解析等异常 # 捕获异常
            logging.error(  # 记录错误日志 # 记录错误
                f"Failed to load or parse persistence file: {e}. Starting fresh.",  # 加载失败，重新开始 # 加载失败
                exc_info=True,  # 包含异常堆栈信息 # 包含堆栈
            )
            self.ranks.clear()  # 清空ranks字典 # 清空字典

    def save_to_disk(self):  # 将状态保存到磁盘
        if not self.persistence_path:  # 检查持久化路径是否配置 # 检查路径
            return  # 未配置则不保存 # 返回

        logging.info("Persisting metadata to disk...")  # 记录保存开始 # 记录保存日志
        with self.global_lock:  # 获取全局锁 # 获取全局锁
            serializable_state = {}  # 可序列化的状态字典 # 可序列化状态
            for key_str, rank_meta in self.ranks.items():  # 遍历所有rank # 遍历rank
                with rank_meta.lock:  # 获取rank锁 # 获取rank锁
                    serializable_state[key_str] = {  # 构建可序列化的状态 # 构建状态
                        "num_pages": rank_meta.num_pages,  # 页面数 # 页面数
                        "free_pages": rank_meta.free_pages,  # 空闲页面列表 # 空闲页面
                        "key_to_index": list(rank_meta.key_to_index.items()),  # 键到页索引映射转为列表 # 映射转列表
                    }

        try:  # 尝试写入文件 # 尝试
            temp_path = self.persistence_path.with_suffix(".tmp")  # 临时文件路径 # 临时文件
            with open(temp_path, "w") as f:  # 打开临时文件 # 打开文件
                json.dump(serializable_state, f, indent=4)  # 写入JSON数据 # 写入JSON
            temp_path.rename(self.persistence_path)  # 原子性重命名为目标文件 # 原子重命名
            logging.info(f"Metadata successfully persisted to {self.persistence_path}")  # 记录保存成功 # 记录成功
        except Exception as e:  # 捕获异常 # 捕获异常
            logging.error(f"Failed to save metadata to disk: {e}", exc_info=True)  # 记录保存失败 # 记录失败

    def schedule_save(self):  # 调度定时保存任务
        if self.is_shutting_down or not self.persistence_path:  # 正在关闭或无持久化路径 # 检查条件
            return  # 不调度 # 返回
        self.save_to_disk()  # 立即保存一次 # 保存到磁盘
        self.save_timer = threading.Timer(self.save_interval, self.schedule_save)  # 创建定时器 # 创建定时器
        self.save_timer.start()  # 启动定时器 # 启动定时器

    def shutdown(self):  # 关闭状态管理器
        logging.info("Shutting down metadata server...")  # 记录关闭信息 # 记录关闭日志
        self.is_shutting_down = True  # 设置关闭标志 # 设置关闭标志
        if self.save_timer:  # 检查定时器是否存在 # 检查定时器
            self.save_timer.cancel()  # 取消定时器 # 取消定时器
        self.save_to_disk()  # 最后保存一次 # 最终保存
        logging.info("Shutdown complete.")  # 记录关闭完成 # 记录完成


# --- Global MetadataServer implementation ---  # --- 全局元数据服务器实现 --- # 元数据服务器实现
class Hf3fsMetadataServer:  # HF3FS元数据服务器类 # HF3FS元数据服务器
    """HF3FS Metadata Server that manages metadata for multiple ranks."""  # 管理多个rank元数据的HF3FS元数据服务器 # 管理多rank元数据的服务器

    def __init__(self, persistence_path: Optional[str] = None, save_interval: int = 60):  # 初始化元数据服务器
        self.state = GlobalMetadataState(persistence_path, save_interval)  # 创建全局状态管理器 # 创建全局状态
        self.app = FastAPI(default_response_class=ORJSONResponse)  # 创建FastAPI应用 # 创建FastAPI应用

        self._setup_routes()  # 设置路由 # 设置路由

    def _setup_routes(self):  # 设置FastAPI路由
        """Setup FastAPI routes."""  # 设置FastAPI路由 # 设置路由
        self.app.post("/{rank}/initialize")(self.initialize)  # 初始化rank路由 # 初始化路由
        self.app.post("/{rank}/exists")(self.exists)  # 检查键存在路由 # exists路由
        self.app.post("/{rank}/reserve_and_allocate_page_indices")(  # 预留并分配页索引路由 # 预留分配路由
            self.reserve_and_allocate_page_indices
        )
        self.app.post("/{rank}/confirm_write")(self.confirm_write)  # 确认写入路由 # 确认写入路由
        self.app.post("/{rank}/delete_keys")(self.delete_keys)  # 删除键路由 # 删除键路由
        self.app.post("/{rank}/clear")(self.clear)  # 清除路由 # 清除路由
        self.app.post("/{rank}/get_page_indices")(self.get_page_indices)  # 获取页索引路由 # 获取页索引路由

    def _rank_key(self, rank: int, namespace: str) -> str:  # 生成rank和命名空间的复合键
        """Generate the composite key for rank+namespace."""  # 生成rank+命名空间的复合键 # 生成复合键
        return f"{rank}:{namespace}"  # 返回复合键字符串 # 返回复合键

    def get_rank_metadata(self, rank: int, namespace: str = "kv") -> RankMetadata:  # 获取rank元数据，含错误处理
        """Get rank metadata with proper error handling."""  # 获取rank元数据（含错误处理） # 获取rank元数据
        key = self._rank_key(rank, namespace)  # 生成复合键 # 生成复合键
        if key not in self.state.ranks:  # 检查rank是否已初始化 # 检查是否初始化
            raise HTTPException(  # 抛出HTTP异常 # 抛出异常
                status_code=404,  # 404状态码 # 404
                detail=f"Rank {rank} namespace '{namespace}' not initialized. Please call /{rank}/initialize first.",  # 错误详情 # 错误详情
            )
        return self.state.ranks[key]  # 返回对应的RankMetadata # 返回元数据

    async def _read_json(self, request: Request) -> dict:  # 解析请求JSON
        """Parse request JSON using orjson if available."""  # 使用orjson解析请求JSON # 解析JSON
        body = await request.body()  # 读取请求体 # 读取请求体
        return orjson.loads(body)  # 用orjson解析 # 解析JSON

    def _json_response(self, content: dict):  # 返回ORJSON响应
        """Return ORJSONResponse when available to bypass jsonable_encoder."""  # 返回ORJSONResponse以绕过jsonable_encoder # 返回ORJSON响应
        return ORJSONResponse(content)  # 创建ORJSON响应 # 创建响应

    async def initialize(self, rank: int, request: Request):  # 初始化rank
        """Initialize a rank with specified number of pages."""  # 使用指定页面数初始化rank # 初始化rank
        data = await self._read_json(request)  # 解析请求数据 # 解析请求
        num_pages = data["num_pages"]  # 获取页面数 # 获取页面数
        namespace = data.get("namespace", "kv")  # 获取命名空间，默认为kv # 获取命名空间
        key = self._rank_key(rank, namespace)  # 生成复合键 # 生成复合键
        with self.state.global_lock:  # 获取全局锁 # 获取全局锁
            if key in self.state.ranks:  # rank已存在 # rank已存在
                logging.info(  # 记录信息 # 记录日志
                    f"Rank {rank} namespace '{namespace}' already exists. Initialization request ignored."  # rank已存在，忽略初始化请求 # 忽略重复初始化
                )
                if self.state.ranks[key].num_pages != num_pages:  # 页面数不一致 # 页面数不匹配
                    logging.warning(  # 记录警告 # 记录警告
                        f"Rank {rank} namespace '{namespace}' initialized with different num_pages. Existing: {self.state.ranks[key].num_pages}, New: {num_pages}"  # 页面数不一致警告 # 页面数不匹配
                    )
            else:  # rank不存在，需要初始化 # 新rank
                logging.info(  # 记录信息 # 记录日志
                    f"Initializing new Rank {rank} namespace '{namespace}' with {num_pages} pages."  # 初始化新rank # 初始化新rank
                )
                self.state.ranks[key] = RankMetadata(num_pages)  # 创建新的RankMetadata # 创建元数据
        return Response(status_code=204)  # 返回204无内容响应 # 返回204

    async def exists(self, rank: int, request: Request):  # 检查键是否存在
        """Check if keys exist in metadata."""  # 检查键是否存在于元数据中 # 检查键是否存在
        data = await self._read_json(request)  # 解析请求数据 # 解析请求
        keys = data["keys"]  # 获取键列表 # 获取键
        namespace = data.get("namespace", "kv")  # 获取命名空间 # 获取命名空间
        metadata = self.get_rank_metadata(rank, namespace)  # 获取rank元数据 # 获取元数据
        results = metadata.exists_keys(keys)  # 检查键是否存在 # 检查键
        return self._json_response({"exists": results})  # 返回JSON响应 # 返回响应

    async def reserve_and_allocate_page_indices(self, rank: int, request: Request):  # 预留并分配页索引
        """Reserve and allocate page indices for keys."""  # 为键预留并分配页面索引 # 预留分配页索引
        data = await self._read_json(request)  # 解析请求数据 # 解析请求
        namespace = data.get("namespace", "kv")  # 获取命名空间 # 获取命名空间
        metadata = self.get_rank_metadata(rank, namespace)  # 获取rank元数据 # 获取元数据
        keys = data["keys"]  # 获取键列表 # 获取键
        results = metadata.reserve_and_allocate_page_indices(keys)  # 预留并分配页索引 # 预留分配
        return self._json_response({"indices": results})  # 返回JSON响应 # 返回响应

    async def confirm_write(self, rank: int, request: Request):  # 确认写入操作
        """Confirm write operations and release pages."""  # 确认写入操作并释放页面 # 确认写入
        data = await self._read_json(request)  # 解析请求数据 # 解析请求
        namespace = data.get("namespace", "kv")  # 获取命名空间 # 获取命名空间
        metadata = self.get_rank_metadata(rank, namespace)  # 获取rank元数据 # 获取元数据
        success_written_keys = data.get("written_keys_to_confirm", [])  # 获取已写入的键列表 # 获取已写入键
        released_pages = data.get("pages_to_release", [])  # 获取需要释放的页面列表 # 获取释放页面

        metadata.confirm_write(success_written_keys, released_pages)  # 确认写入 # 确认写入

        return Response(status_code=204)  # 返回204无内容响应 # 返回204

    async def delete_keys(self, rank: int, request: Request):  # 删除键
        """Delete keys from metadata."""  # 从元数据中删除键 # 删除键
        data = await self._read_json(request)  # 解析请求数据 # 解析请求
        namespace = data.get("namespace", "kv")  # 获取命名空间 # 获取命名空间
        metadata = self.get_rank_metadata(rank, namespace)  # 获取rank元数据 # 获取元数据
        count = metadata.delete_keys(data["keys"])  # 删除键 # 删除键
        return Response(status_code=204)  # 返回204无内容响应 # 返回204

    async def clear(self, rank: int, request: Request):  # 清除rank的所有元数据
        """Clear all metadata for a rank."""  # 清除rank的所有元数据 # 清除元数据
        data = await self._read_json(request)  # 解析请求数据 # 解析请求
        namespace = data.get("namespace", "kv")  # 获取命名空间 # 获取命名空间
        metadata = self.get_rank_metadata(rank, namespace)  # 获取rank元数据 # 获取元数据
        metadata.clear_all()  # 清除所有元数据 # 清除元数据
        return Response(status_code=204)  # 返回204无内容响应 # 返回204

    async def get_page_indices(self, rank: int, request: Request):  # 获取键的页索引
        """Get page indices for keys."""  # 获取键的页面索引 # 获取页索引
        data = await self._read_json(request)  # 解析请求数据 # 解析请求
        namespace = data.get("namespace", "kv")  # 获取命名空间 # 获取命名空间
        metadata = self.get_rank_metadata(rank, namespace)  # 获取rank元数据 # 获取元数据
        keys = data["keys"]  # 获取键列表 # 获取键
        results = metadata.get_page_indices(keys)  # 获取页索引 # 获取页索引
        return self._json_response({"indices": results})  # 返回JSON响应 # 返回响应

    def run(self, host: str = "0.0.0.0", port: int = 18000):  # 运行元数据服务器
        """Run the metadata server."""  # 运行元数据服务器 # 运行服务器
        self.state.load_from_disk()  # 从磁盘加载持久化状态 # 加载状态
        if self.state.persistence_path:  # 配置了持久化路径 # 检查持久化
            self.state.schedule_save()  # 启动定时保存 # 启动定时保存
            atexit.register(self.state.shutdown)  # 注册退出回调 # 注册退出回调

        import uvicorn  # 导入uvicorn ASGI服务器 # 导入uvicorn

        logging.info(f"Starting metadata server on http://{host}:{port}")  # 记录启动信息 # 记录启动日志
        if self.state.persistence_path:  # 配置了持久化路径 # 检查持久化
            logging.info(  # 记录持久化信息 # 记录日志
                f"Persistence is ENABLED. Saving to '{self.state.persistence_path}' every {self.state.save_interval} seconds."  # 持久化已启用 # 持久化启用
            )
        else:  # 未配置持久化 # 无持久化
            logging.info("Persistence is DISABLED.")  # 持久化已禁用 # 持久化禁用

        uvicorn.run(self.app, host=host, port=port)  # 启动uvicorn服务器 # 启动服务器


# --- Client implementation ---  # --- 客户端实现 --- # 客户端实现部分
class Hf3fsGlobalMetadataClient(Hf3fsMetadataInterface):  # 远程HTTP元数据客户端 # 全局HTTP元数据客户端
    """Global http metadata client for HF3FS."""  # HF3FS的全局HTTP元数据客户端 # HF3FS全局HTTP客户端

    def __init__(self, base_url: str, max_retries: int = 3):  # 初始化全局元数据客户端
        self.base_url = base_url.rstrip("/")  # 去掉末尾斜杠的基础URL # 基础URL
        self._session = requests.Session()  # 创建HTTP会话 # 创建会话

        retry_strategy = Retry(  # 创建重试策略 # 重试策略
            total=max_retries,  # 最大重试次数 # 最大重试次数
            backoff_factor=0.3,  # 退避因子 # 退避因子
            status_forcelist=[500, 502, 503, 504],  # 触发重试的状态码 # 触发重试的状态码
            allowed_methods=["GET", "POST"],  # 允许重试的HTTP方法 # 允许的方法
        )
        adapter = HTTPAdapter(  # 创建HTTP适配器 # HTTP适配器
            max_retries=retry_strategy, pool_connections=256, pool_maxsize=256  # 最大重试、连接池大小 # 重试和连接池
        )
        self._session.mount("http://", adapter)  # 挂载HTTP适配器 # 挂载适配器

    def _post(self, endpoint: str, json_data: dict) -> dict:  # 发送POST请求
        try:  # 尝试发送请求 # 尝试
            url = f"{self.base_url}/{endpoint}"  # 构建完整URL # 构建URL
            headers = {"Content-Type": "application/json"}  # 请求头 # 请求头
            payload = orjson.dumps(json_data)  # type: ignore[union-attr]  # 序列化JSON数据 # 序列化数据
            response = self._session.post(url, data=payload, headers=headers)  # 发送POST请求 # 发送请求
            response.raise_for_status()  # 检查HTTP状态码 # 检查状态码

            if response.status_code == 204 or not response.content:  # 204或空内容 # 空响应
                return {}  # 返回空字典 # 返回空
            return orjson.loads(response.content)  # type: ignore[union-attr]  # 解析响应JSON # 解析响应
        except requests.exceptions.RequestException as e:  # 捕获请求异常 # 捕获异常
            logging.error(f"Failed to POST to {endpoint} after retries: {e}")  # 记录失败日志 # 记录失败
            raise RuntimeError(f"Failed to connect to metadata server: {e}") from e  # 抛出运行时异常 # 抛出异常

    def initialize(  # 初始化rank
        self, rank: int, num_pages: int, namespace: PoolName = PoolName.KV  # rank、页面数、命名空间 # 参数
    ) -> None:
        self._post(  # 发送POST请求 # 发送请求
            f"{rank}/initialize", {"num_pages": num_pages, "namespace": str(namespace)}  # 端点和参数 # 端点和数据
        )

    def reserve_and_allocate_page_indices(  # 预留并分配页索引
        self, rank: int, keys: List[Tuple[str, str]], namespace: PoolName = PoolName.KV  # rank、键列表、命名空间 # 参数
    ) -> List[Tuple[bool, int]]:
        response = self._post(  # 发送POST请求 # 发送请求
            f"{rank}/reserve_and_allocate_page_indices",  # 端点 # 端点
            {"keys": keys, "namespace": str(namespace)},  # 参数 # 数据
        )
        return [tuple(item) for item in response.get("indices")]  # 将响应转为元组列表 # 转为元组

    def confirm_write(  # 确认写入操作
        self,
        rank: int,  # rank编号 # rank
        written_keys_to_confirm: List[Tuple[str, int]],  # 已写入的键和页索引 # 已写入键
        pages_to_release: List[int],  # 需要释放的页面 # 释放页面
        namespace: PoolName = PoolName.KV,  # 命名空间 # 命名空间
    ) -> None:
        self._post(  # 发送POST请求 # 发送请求
            f"{rank}/confirm_write",  # 端点 # 端点
            {  # 请求体 # 请求体
                "written_keys_to_confirm": written_keys_to_confirm,  # 已写入键 # 已写入键
                "pages_to_release": pages_to_release,  # 释放页面 # 释放页面
                "namespace": str(namespace),  # 命名空间 # 命名空间
            },
        )

    def delete_keys(  # 删除键
        self, rank: int, keys: List[str], namespace: PoolName = PoolName.KV  # rank、键列表、命名空间 # 参数
    ) -> None:
        self._post(f"{rank}/delete_keys", {"keys": keys, "namespace": str(namespace)})  # 发送删除请求 # 发送请求

    def exists(  # 检查键是否存在
        self, rank: int, keys: List[str], namespace: PoolName = PoolName.KV  # rank、键列表、命名空间 # 参数
    ) -> List[bool]:
        response = self._post(  # 发送POST请求 # 发送请求
            f"{rank}/exists", {"keys": keys, "namespace": str(namespace)}  # 端点和参数 # 端点和数据
        )
        return response.get("exists", [])  # 返回存在状态列表 # 返回结果

    def clear(self, rank: int, namespace: PoolName = PoolName.KV) -> None:  # 清除rank元数据
        self._post(f"{rank}/clear", {"namespace": str(namespace)})  # 发送清除请求 # 发送请求

    def get_page_indices(  # 获取键的页索引
        self, rank: int, keys: List[str], namespace: PoolName = PoolName.KV  # rank、键列表、命名空间 # 参数
    ) -> List[Optional[int]]:
        response = self._post(  # 发送POST请求 # 发送请求
            f"{rank}/get_page_indices", {"keys": keys, "namespace": str(namespace)}  # 端点和参数 # 端点和数据
        )
        return response.get("indices")  # 返回页索引列表 # 返回索引


class Hf3fsLocalMetadataClient(Hf3fsMetadataInterface):  # 本地内存元数据客户端 # 本地元数据客户端
    """Local metadata client that directly operates on RankMetadata in memory without metadata server."""  # 直接在内存中操作RankMetadata的本地元数据客户端，无需元数据服务器 # 无需服务器的本地客户端

    def __init__(self):  # 初始化本地元数据客户端
        self._metadata: Dict[str, RankMetadata] = {}  # key: "rank:namespace"  # 键到RankMetadata的映射 # 元数据字典

    def _ns_key(self, rank: int, namespace: PoolName) -> str:  # 生成命名空间键
        return f"{rank}:{namespace}"  # 返回复合键 # 返回复合键

    def _get_metadata(self, rank: int, namespace) -> RankMetadata:  # 获取指定rank和命名空间的元数据
        key = self._ns_key(rank, namespace)  # 生成复合键 # 生成复合键
        if key not in self._metadata:  # 检查是否已初始化 # 检查初始化
            raise RuntimeError(  # 抛出运行时异常 # 抛出异常
                f"Namespace '{namespace}' for rank {rank} not initialized"  # 命名空间未初始化 # 未初始化
            )
        return self._metadata[key]  # 返回对应的RankMetadata # 返回元数据

    def initialize(  # 初始化rank
        self, rank: int, num_pages: int, namespace: PoolName = PoolName.KV  # rank、页面数、命名空间 # 参数
    ) -> None:
        key = self._ns_key(rank, namespace)  # 生成复合键 # 生成复合键
        if key not in self._metadata:  # 检查是否已初始化 # 检查初始化
            self._metadata[key] = RankMetadata(num_pages)  # 创建新的RankMetadata # 创建元数据

    def reserve_and_allocate_page_indices(  # 预留并分配页索引
        self, rank: int, keys: List[Tuple[str, str]], namespace: PoolName = PoolName.KV  # rank、键列表、命名空间 # 参数
    ) -> List[Tuple[bool, int]]:
        """Reserve and allocate page indices for keys."""  # 为键预留并分配页面索引 # 预留分配页索引
        return self._get_metadata(rank, namespace).reserve_and_allocate_page_indices(  # 调用RankMetadata的方法 # 调用方法
            keys  # 键列表 # 键列表
        )

    def confirm_write(  # 确认写入操作
        self,
        rank: int,  # rank编号 # rank
        written_keys_to_confirm: List[Tuple[str, int]],  # 已写入的键和页索引 # 已写入键
        pages_to_release: List[int],  # 需要释放的页面 # 释放页面
        namespace: PoolName = PoolName.KV,  # 命名空间 # 命名空间
    ) -> None:
        """Confirm write operations."""  # 确认写入操作 # 确认写入
        self._get_metadata(rank, namespace).confirm_write(  # 调用RankMetadata的方法 # 调用方法
            written_keys_to_confirm, pages_to_release  # 已写入键和释放页面 # 参数
        )

    def delete_keys(  # 删除键
        self, rank: int, keys: List[str], namespace: PoolName = PoolName.KV  # rank、键列表、命名空间 # 参数
    ) -> None:
        """Delete keys."""  # 删除键 # 删除键
        self._get_metadata(rank, namespace).delete_keys(keys)  # 调用RankMetadata的方法 # 调用方法

    def exists(  # 检查键是否存在
        self, rank: int, keys: List[str], namespace: PoolName = PoolName.KV  # rank、键列表、命名空间 # 参数
    ) -> List[bool]:
        """Check if keys exist."""  # 检查键是否存在 # 检查键是否存在
        return self._get_metadata(rank, namespace).exists_keys(keys)  # 调用RankMetadata的方法 # 调用方法

    def clear(self, rank: int, namespace: PoolName = PoolName.KV) -> None:  # 清除rank元数据
        """Clear all metadata for rank."""  # 清除rank的所有元数据 # 清除元数据
        self._get_metadata(rank, namespace).clear_all()  # 调用RankMetadata的方法 # 调用方法

    def get_page_indices(  # 获取键的页索引
        self, rank: int, keys: List[str], namespace: PoolName = PoolName.KV  # rank、键列表、命名空间 # 参数
    ) -> List[Optional[int]]:
        """Get page indices for keys."""  # 获取键的页面索引 # 获取页索引
        return self._get_metadata(rank, namespace).get_page_indices(keys)  # 调用RankMetadata的方法 # 调用方法


def run_metadata_server(  # 运行HF3FS元数据服务器
    host: str = "0.0.0.0",  # 绑定主机地址 # 主机地址
    port: int = 18000,  # 绑定端口 # 端口
    persistence_path: Optional[str] = None,  # 持久化文件路径 # 持久化路径
    save_interval: int = 60,  # 保存间隔（秒） # 保存间隔
):
    """Run the HF3FS metadata server."""  # 运行HF3FS元数据服务器 # 运行元数据服务器
    global server  # 全局server变量 # 全局变量
    server = Hf3fsMetadataServer(  # 创建元数据服务器实例 # 创建服务器
        persistence_path=persistence_path, save_interval=save_interval  # 持久化路径和保存间隔 # 持久化参数
    )

    server.run(host=host, port=port)  # 运行服务器 # 运行服务器


# --- Main Execution ---  # --- 主执行入口 --- # 主执行入口
if __name__ == "__main__":  # 作为主脚本运行 # 主脚本入口
    parser = argparse.ArgumentParser(description="HF3FS Metadata Server")  # 创建参数解析器 # 创建解析器
    parser.add_argument(  # 添加主机参数 # 添加host参数
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to."  # 绑定主机地址 # 绑定主机
    )
    parser.add_argument(  # 添加端口参数 # 添加port参数
        "--port", type=int, default=18000, help="Port to run the server on."  # 服务器端口 # 服务器端口
    )
    parser.add_argument(  # 添加持久化路径参数 # 添加持久化路径参数
        "--persistence-path",
        type=str,
        default=None,
        help="Path to the file for persisting metadata. If not provided, persistence is disabled.",  # 元数据持久化文件路径，不提供则禁用持久化 # 持久化路径
    )
    parser.add_argument(  # 添加保存间隔参数 # 添加保存间隔参数
        "--save-interval",
        type=int,
        default=60,
        help="Interval in seconds for periodically saving metadata to disk.",  # 定期保存元数据到磁盘的间隔秒数 # 保存间隔
    )
    args = parser.parse_args()  # 解析命令行参数 # 解析参数

    run_metadata_server(args.host, args.port, args.persistence_path, args.save_interval)  # 运行元数据服务器 # 运行服务器
