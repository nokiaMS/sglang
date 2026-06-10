# 日志工具模块
# 提供日志目标创建和结构化JSON日志记录功能
# 支持标准输出和文件（按小时轮转）两种日志输出方式
from __future__ import annotations  # 启用延迟注解评估

import json  # 导入JSON模块
import logging  # 导入日志模块
import os  # 导入操作系统模块
import socket  # 导入套接字模块
import sys  # 导入系统模块
from datetime import datetime  # 导入日期时间类
from logging.handlers import TimedRotatingFileHandler  # 导入定时轮转文件处理器
from typing import List, Optional, Union  # 导入类型提示

import torch.distributed as dist  # 导入PyTorch分布式模块


def create_log_targets(  # 创建日志目标列表，根据指定的目标类型创建对应的日志记录器
    *, targets: Optional[List[str]], name_prefix: str
) -> List[logging.Logger]:
    if not targets:  # 如果没有指定目标
        return [_create_log_target_stdout(name_prefix)]  # 默认使用标准输出
    return [_create_log_target(t, name_prefix) for t in targets]  # 为每个目标创建日志记录器


def _create_log_target(target: str, name_prefix: str) -> logging.Logger:  # 根据目标字符串创建日志记录器
    if target.lower() == "stdout":  # 如果目标是标准输出
        return _create_log_target_stdout(name_prefix)  # 创建标准输出日志记录器
    return _create_log_target_file(target, name_prefix)  # 否则创建文件日志记录器


def _create_log_target_stdout(name_prefix: str) -> logging.Logger:  # 创建标准输出日志记录器
    return _create_logger_with_handler(  # 使用流处理器创建日志记录器
        f"{name_prefix}.stdout", logging.StreamHandler(sys.stdout)
    )


def _create_log_target_file(directory: str, name_prefix: str) -> logging.Logger:  # 创建文件日志记录器（按小时轮转）
    os.makedirs(directory, exist_ok=True)  # 确保日志目录存在
    hostname = socket.gethostname()  # 获取主机名
    rank = dist.get_rank() if dist.is_initialized() else 0  # 获取分布式排名，未初始化则为0
    filename = os.path.join(directory, f"{hostname}_{rank}.log")  # 生成日志文件名
    handler = TimedRotatingFileHandler(  # 创建定时轮转文件处理器
        filename, when="H", backupCount=0, encoding="utf-8"  # 每小时轮转，不保留备份，UTF-8编码
    )
    return _create_logger_with_handler(  # 使用文件处理器创建日志记录器
        f"{name_prefix}.file.{directory}.{hostname}_{rank}", handler
    )


def _create_logger_with_handler(name: str, handler: logging.Handler) -> logging.Logger:  # 使用指定处理器创建日志记录器
    logger = logging.getLogger(name)  # 获取或创建日志记录器
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO
    logger.propagate = False  # 禁止日志传播到父记录器
    if not logger.handlers:  # 如果记录器没有处理器
        handler.setFormatter(  # 设置日志格式
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)  # 添加处理器
    return logger


def log_json(  # 以JSON格式记录结构化日志信息
    loggers: Union[logging.Logger, List[logging.Logger]], event: str, data: dict
) -> None:
    log_data = {  # 构建日志数据字典
        "timestamp": datetime.now().isoformat(),  # 当前时间戳
        "event": event,  # 事件名称
        **data,  # 附加数据
    }
    msg = json.dumps(log_data, ensure_ascii=False)  # 序列化为JSON字符串，不转义非ASCII字符

    if not isinstance(loggers, list):  # 如果loggers不是列表
        loggers = [loggers]  # 将其包装为列表

    for logger in loggers:  # 遍历每个日志记录器
        logger.info(msg)  # 记录信息级别日志
