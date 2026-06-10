# MindSpore分布式运行器模块，负责启动MindSpore分布式模块并复用HCCL通信句柄
# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识：Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project  # SPDX版权声明：SGLang项目的贡献者
"""ms_runner launch MindSpore distributed modules."""  # ms_runner启动MindSpore分布式模块

import logging  # 导入日志模块
import multiprocessing as mp  # 导入多进程模块
import os  # 导入操作系统模块
import sys  # 导入系统模块
from pathlib import Path  # 导入路径模块

import mindspore as ms  # 导入MindSpore框架
import torch  # 导入PyTorch框架
from mindspore._c_expression import GroupOptions  # 导入MindSpore组选项类
from mindspore.communication import create_group  # 导入MindSpore通信组创建函数

from sglang.srt.distributed.parallel_state import _groups  # 导入分布式并行状态中的组信息

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


class _Tmp:  # 临时类，用于管理调度器子进程的生命周期
    def __init__(self):  # 初始化方法
        self.sched_p = None  # 调度器子进程对象

    def set_sched_process(self, p):  # 设置调度器子进程
        self.sched_p = p  # 保存子进程引用

    def __del__(self):  # 析构方法，对象销毁时终止子进程
        if self.sched_p:  # 如果子进程存在
            self.sched_p.kill()  # 终止子进程


_tmp = _Tmp()  # 创建临时类全局实例


def _get_host_and_ip(distributed_init_method):  # 从分布式初始化方法字符串中解析主机IP和端口
    try:  # 尝试解析
        _, ip_str, port_str = distributed_init_method.split(":")  # 按冒号分割字符串
        ip = ip_str.split("/")[-1]  # 提取IP地址（去掉可能的路径前缀）
        port = int(port_str)  # 将端口转为整数
    except Exception as e:  # 解析失败时抛出异常
        raise RuntimeError(
            "Cannot get host and port information from %s, error: %s!"
            % (distributed_init_method, str(e))
        )

    return ip, port  # 返回IP地址和端口号


def run_scheduler_init(rank, local_rank, world_size, master_addr, master_port):  # 运行MindSpore调度器初始化（在子进程中执行）
    with open(str(Path() / "schedule.log"), "w") as scheduler_f:  # 打开调度器日志文件
        # For Python outputs.  # 用于Python输出重定向
        sys.stdout = scheduler_f  # 将标准输出重定向到日志文件
        sys.stderr = scheduler_f  # 将标准错误重定向到日志文件
        # For C++ outputs.  # 用于C++输出重定向
        os.dup2(scheduler_f.fileno(), 1)  # 将文件描述符1（stdout）重定向到日志文件
        os.dup2(scheduler_f.fileno(), 2)  # 将文件描述符2（stderr）重定向到日志文件
        os.environ["DEVICE_ID"] = str(local_rank)  # 设置设备ID为本地排名
        os.environ["MS_WORKER_NUM"] = str(world_size)  # 设置MindSpore工作进程数量
        os.environ["MS_ROLE"] = "MS_SCHED"  # 设置MindSpore角色为调度器
        os.environ["MS_NODE_ID"] = str(rank)  # 设置节点ID为全局排名
        os.environ["MS_SCHED_HOST"] = str(master_addr)  # 设置调度器主机地址
        os.environ["MS_SCHED_PORT"] = str(master_port)  # 设置调度器端口
        # This function is blocked until the whole cluster exits.  # 此函数会阻塞直到整个集群退出
        ms.communication.init()  # 初始化MindSpore通信


def set_ms_parallel_env(rank, local_rank, world_size, init_method):  # 设置MindSpore并行环境变量和调度器子进程
    master_addr, master_port = _get_host_and_ip(init_method)  # 解析主节点地址和端口
    # change port avoiding port conflicts with torch  # 更改端口以避免与PyTorch端口冲突
    master_port = master_port + 35 if master_port < 65500 else master_port - 35  # 端口偏移35避免冲突
    if not os.getenv("MS_ROLE"):  # 如果MS_ROLE环境变量未设置
        if rank == 0:  # 仅在rank 0上创建调度器子进程
            # Create a subprocess for scheduler of MindSpore, just for internal collaboration, not for collective communication  # 为MindSpore调度器创建子进程，仅用于内部协作，不用于集合通信
            sched_p = mp.Process(  # 创建调度器子进程
                target=run_scheduler_init,  # 目标函数为调度器初始化
                args=(rank, local_rank, world_size, master_addr, master_port),  # 传入参数
            )
            sched_p.start()  # 启动调度器子进程
            global _tmp  # 声明全局变量
            _tmp.set_sched_process(sched_p)  # 保存调度器子进程引用以便后续清理

        os.environ["DEVICE_ID"] = str(local_rank)  # 设置设备ID
        os.environ["MS_WORKER_NUM"] = str(world_size)  # 设置工作进程数量
        os.environ["MS_ROLE"] = "MS_WORKER"  # 设置MindSpore角色为工作进程
        os.environ["MS_NODE_ID"] = str(rank)  # 设置节点ID
        os.environ["MS_SCHED_HOST"] = str(master_addr)  # 设置调度器主机地址
        os.environ["MS_SCHED_PORT"] = str(master_port)  # 设置调度器端口


def reuse_hccl_comm():  # 复用PyTorch HCCL通信句柄创建MindSpore通信组
    for group_name, group in _groups.items():  # 遍历所有并行组
        # Torch ProcessGroupHccl  # PyTorch的HCCL进程组
        device_group = group().device_group  # 获取设备组
        hccl_comm_handle = device_group._get_backend(torch.device("npu")).get_hccl_comm(  # 获取HCCL通信句柄
            group().local_rank  # 使用本地排名获取对应句柄
        )
        logger.info(
            f"MindSpore reuse torch group: {device_group}, group_name: {group_name}, local rank: {group().local_rank},"
            f"hccl communicator handle: {hex(hccl_comm_handle)}",
        )
        # Create MS communication group by hccl comm handle to reuse Torch group.  # 通过HCCL通信句柄创建MindSpore通信组以复用PyTorch组
        group_options = GroupOptions()  # 创建组选项对象
        group_options.hccl_config = {"hccl_comm": hccl_comm_handle}  # 设置HCCL配置为复用的通信句柄
        create_group(group_name, group().ranks, group_options)  # 使用复用句柄创建MindSpore通信组


def init_ms_distributed(world_size, rank, local_rank, server_args, port):  # 初始化MindSpore分布式环境
    if server_args.dist_init_addr:  # 如果指定了分布式初始化地址
        dist_init_method = f"tcp://{server_args.dist_init_addr}"  # 使用指定地址构造初始化方法
    else:
        dist_init_method = f"tcp://{server_args.host}:{port}"  # 使用主机和端口构造初始化方法
    set_ms_parallel_env(rank, local_rank, world_size, dist_init_method)  # 设置MindSpore并行环境

    ms.set_context(infer_boost="on", jit_level="O0")  # 设置推理加速和JIT级别
    ms.set_context(mode=ms.context.PYNATIVE_MODE)  # 设置为动态图模式
    ms.set_device("Ascend", local_rank)  # 设置Ascend设备
    ms.communication.init("hccl")  # 使用HCCL后端初始化通信
    # After distributed job is initialized, reuse hccl comms for MindSpore.  # 分布式作业初始化后，复用HCCL通信句柄给MindSpore使用
    reuse_hccl_comm()  # 复用HCCL通信句柄
