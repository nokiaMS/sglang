# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/runai_utils.py
# RunAI模型流式加载器工具，支持从对象存储（S3/GCS/Azure）流式加载模型权重
# 实现元数据预下载和权重懒加载，避免多进程协调问题和重复下载

import hashlib  # 导入哈希计算模块
import logging  # 导入日志记录模块
import os  # 导入操作系统接口模块
from pathlib import Path  # 导入路径处理模块

from sglang.srt.environ import envs  # 导入环境变量配置

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

SUPPORTED_SCHEMES = ["s3://", "gs://", "az://"]  # 支持的对象存储协议方案

# Design Pattern: Single Metadata Download Before Process Launch  # 设计模式：子进程启动前的单次元数据下载

#   1. Engine entrypoint (engine.py) or server arguments post init  (server_args.py):  # 1. 引擎入口或服务器参数初始化后：
#     - Downloads config/tokenizer metadata ONCE before launching subprocesses  # 在启动子进程前一次性下载配置/分词器元数据
#     - This happens in the main process, avoiding multi-process coordination  # 在主进程中执行，避免多进程协调
#
#   2. ModelConfig/HF Utils (model_config.py, hf_transformers_utils.py):  # 2. ModelConfig/HF工具：
#     - Use ObjectStorageModel.get_path() to retrieve the cached local path  # 使用ObjectStorageModel.get_path()获取缓存的本地路径
#     - NO re-download - just path resolution  # 不重复下载 - 仅路径解析
#
#   3. RunaiModelStreamerLoader (loader.py):  # 3. RunAI模型流式加载器：
#     - Calls list_safetensors() which operates directly on the object storage URI  # 调用list_safetensors()直接操作对象存储URI
#     - Streams weights lazily during model loading  # 在模型加载期间懒加载权重

#   This avoids file locks, race conditions, and duplicate downloads  # 避免文件锁、竞态条件和重复下载


def list_safetensors(path: str = "") -> list[str]:  # 列出对象存储路径中的safetensors文件
    """
    List full file names from object path and filter by allow pattern.  # 从对象路径列出完整文件名并按允许模式过滤

    Args:  # 参数
        path: The object storage path to list from.  # 要列出的对象存储路径

    Returns:  # 返回值
        list[str]: List of full object storage paths allowed by the pattern  # 符合模式的对象存储路径列表
    """
    from runai_model_streamer import list_safetensors as runai_list_safetensors  # 导入RunAI的safetensors列表函数

    return runai_list_safetensors(path)  # 调用RunAI函数列出文件


def is_runai_obj_uri(model_or_path: str | Path) -> bool:  # 判断路径是否为RunAI对象存储URI
    # Cast to str to handle pathlib.Path inputs which lack string methods (like .lower)  # 转为字符串以处理缺少字符串方法的Path输入
    return str(model_or_path).lower().startswith(tuple(SUPPORTED_SCHEMES))  # 检查路径是否以支持的协议开头


class ObjectStorageModel:  # 对象存储模型加载器
    """
    Model loader that uses Runai Model Streamer to load a model.  # 使用RunAI模型流式加载器加载模型

      Supports object storage (S3, GCS) with lazy weight streaming.  # 支持对象存储（S3、GCS）的懒加载权重流式传输

      Configuration (via load_config.model_loader_extra_config):  # 配置（通过load_config.model_loader_extra_config）：
          - distributed (bool): Enable distributed streaming  # distributed (bool): 启用分布式流式传输
          - concurrency (int): Number of concurrent downloads  # concurrency (int): 并发下载数
          - memory_limit (int): Memory limit for streaming buffer  # memory_limit (int): 流式缓冲区内存限制

      Note: Metadata files must be pre-downloaded via  # 注意：元数据文件必须预先下载
      ObjectStorageModel.download_and_get_path() before instantiation.  # 通过ObjectStorageModel.download_and_get_path()在实例化前预下载

    Attributes:  # 属性
        dir: The temporary created directory.  # dir: 创建的临时目录
    """

    def __init__(self, url: str) -> None:  # 初始化对象存储模型加载器
        self.dir = ObjectStorageModel.get_path(url)  # 获取本地缓存目录路径

        from runai_model_streamer import ObjectStorageModel as RunaiObjectStorageModel  # 导入RunAI对象存储模型

        self._runai_obj = RunaiObjectStorageModel(model_path=url, dst=self.dir)  # 创建RunAI对象存储模型实例

    def __enter__(self):  # 上下文管理器入口
        return self  # 返回自身

    def __exit__(self, exc_type, exc_val, exc_tb):  # 上下文管理器退出
        return self._runai_obj.__exit__(exc_type, exc_val, exc_tb)  # 委托给RunAI对象的退出方法

    def pull_files(  # 从对象存储拉取文件到本地缓存目录
        self,
        allow_pattern: list[str] | None = None,  # 允许的文件模式（如["*.json"]）
        ignore_pattern: list[str] | None = None,  # 忽略的文件模式
    ) -> None:
        """Pull files from object storage into the local cache directory.  # 从对象存储拉取文件到本地缓存目录

        Args:  # 参数
            allow_pattern: File patterns to include (e.g. ["*.json"]).  # 允许的文件模式（如["*.json"]）
            ignore_pattern: File patterns to exclude.  # 忽略的文件模式
        """
        self._runai_obj.pull_files(allow_pattern, ignore_pattern)  # 调用RunAI对象拉取文件

    @classmethod
    def download_and_get_path(cls, model_path: str) -> str:  # 下载模型元数据并返回本地目录路径
        """
        Downloads the model metadata (excluding heavy weights) and returns  # 下载模型元数据（排除大权重文件）并返回
        the local directory path. Safe for concurrent usage by multiple processes  # 本地目录路径。多进程并发使用安全
        """
        with cls(url=model_path) as downloader:  # 使用上下文管理器创建下载器
            downloader.pull_files(  # 拉取文件，排除权重文件
                ignore_pattern=[  # 忽略的文件模式列表
                    "*.pt",  # PyTorch权重文件
                    "*.safetensors",  # safetensors权重文件
                    "*.bin",  # 二进制权重文件
                    "*.tensors",  # 张量文件
                    "*.pth",  # PyTorch权重文件
                ],
            )
            cache_dir = downloader.dir  # 获取缓存目录路径
            logger.info(f"Runai Model : {cache_dir}, metadata ready.")  # 记录元数据已就绪
        return cache_dir  # 返回缓存目录路径

    @classmethod
    def get_path(cls, model_path: str) -> str:  # 获取模型本地缓存目录路径
        """
        Returns the local directory path.  # 返回本地目录路径
        """
        model_hash = hashlib.sha256(str(model_path).encode()).hexdigest()[:16]  # 计算模型路径的SHA256哈希前16位
        base_dir = envs.SGLANG_CACHE_DIR.get()  # 获取基础缓存目录

        # Ensure base cache dir exists  # 确保基础缓存目录存在
        os.makedirs(os.path.join(base_dir, "model_streamer"), exist_ok=True)  # 创建model_streamer子目录

        return os.path.join(  # 返回完整的缓存路径
            base_dir,  # 基础缓存目录
            "model_streamer",  # 模型流式传输子目录
            model_hash,  # 模型哈希子目录
        )
