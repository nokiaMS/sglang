# 远程实例权重加载器工具模块
# 提供远程实例间模型权重传输的辅助函数，包括通信组初始化、权重传输触发和内存区域注册等功能

# SPDX-License-Identifier: Apache-2.0

import enum  # 导入枚举模块
import importlib  # 导入模块导入工具
import importlib.util  # 导入模块导入工具的实用功能
import logging  # 导入日志模块
import time  # 导入时间模块
from typing import List  # 从typing模块导入List类型

import requests  # 导入HTTP请求库

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


class RemoteInstanceWeightLoaderBackend(str, enum.Enum):  # 远程实例权重加载器后端枚举类
    NCCL = "nccl"  # NCCL后端
    TRANSFER_ENGINE = "transfer_engine"  # 传输引擎后端
    MODELEXPRESS = "modelexpress"  # ModelExpress后端


def trigger_init_weights_send_group_for_remote_instance_request(  # 触发远程实例初始化权重发送通信组的请求
    remote_instance_weight_loader_seed_instance_ip: str,  # 种子实例IP地址
    remote_instance_weight_loader_seed_instance_service_port: int,  # 种子实例服务端口
    remote_instance_weight_loader_send_weights_group_ports: List[int],  # 权重发送通信组端口列表
    remote_instance_weight_loader_client_id: str,  # 客户端标识符
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"  # 构造种子实例服务URL
    # Only support loading weights from instance with same parallelism strategy.
    # 仅支持从具有相同并行策略的实例加载权重。
    # Per TP rank pair between seed and dst instances will build a communication group for sending weights.
    # 种子实例和目标实例之间每对TP秩将建立一个用于发送权重的通信组。
    # i.e. seed TP 0 <-> dst TP 0, seed TP 1 <-> dst TP 1, etc.
    # 即种子TP 0 <-> 目标TP 0，种子TP 1 <-> 目标TP 1，等等。
    # Each communication group will have a world size 2.
    # 每个通信组的全局大小为2。
    try:  # 尝试发送请求
        requests.post(  # 发送POST请求
            f"{seed_instance_service_url}/init_weights_send_group_for_remote_instance",  # 请求初始化权重发送组的端点
            json={  # 请求体JSON数据
                "master_address": remote_instance_weight_loader_seed_instance_ip,  # 主节点地址
                "ports": (  # 端口列表
                    ",".join(  # 用逗号连接端口字符串
                        str(p)  # 将端口转换为字符串
                        for p in remote_instance_weight_loader_send_weights_group_ports  # 遍历所有端口
                    )
                ),
                "group_rank": 0,  # 通信组中的秩
                "world_size": 2,  # 通信组全局大小
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",  # 通信组名称
                "backend": "nccl",  # 通信后端为NCCL
            },
        )
    except Exception as e:  # 捕获异常
        logger.error(  # 记录错误日志
            f"Failed to trigger init_weights_send_group_for_remote_instance_request to seed instance {seed_instance_service_url}: {e}."  # 错误信息
        )
        raise  # 重新抛出异常


def trigger_transferring_weights_request(  # 触发远程实例权重传输请求
    remote_instance_weight_loader_seed_instance_ip: str,  # 种子实例IP地址
    remote_instance_weight_loader_seed_instance_service_port: int,  # 种子实例服务端口
    remote_instance_weight_loader_send_weights_group_ports: List[int],  # 权重发送通信组端口列表
    remote_instance_weight_loader_client_id: str,  # 客户端标识符
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"  # 构造种子实例服务URL
    try:  # 尝试发送请求
        requests.post(  # 发送POST请求
            f"{seed_instance_service_url}/send_weights_to_remote_instance",  # 请求发送权重到远程实例的端点
            json={  # 请求体JSON数据
                "master_address": remote_instance_weight_loader_seed_instance_ip,  # 主节点地址
                "ports": (  # 端口列表
                    ",".join(  # 用逗号连接端口字符串
                        str(p)  # 将端口转换为字符串
                        for p in remote_instance_weight_loader_send_weights_group_ports  # 遍历所有端口
                    )
                ),
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",  # 通信组名称
            },
        )
    except Exception as e:  # 捕获异常
        logger.error(f"Failed to trigger send weights to remote instance request: {e}")  # 记录错误日志
        raise  # 重新抛出异常


def get_remote_instance_transfer_engine_info_per_rank(seed_url: str, rank: int):  # 获取每个秩的远程实例传输引擎信息
    try:  # 尝试发送请求
        response = requests.get(  # 发送GET请求
            f"{seed_url}/get_remote_instance_transfer_engine_info",  # 请求获取传输引擎信息的端点
            params={  # 请求参数
                "rank": rank,  # 张量并行秩
            },
        )

        if response.status_code == 200:  # 如果响应状态码为200（成功）
            data = response.json()  # 解析响应JSON数据

            if "remote_instance_transfer_engine_info" in data:  # 如果响应中包含传输引擎信息
                return data["remote_instance_transfer_engine_info"]  # 返回传输引擎信息
            else:  # 否则
                logger.error(  # 记录错误日志
                    "Failed to get `remote_instance_transfer_engine_info` in response."  # 响应中未找到传输引擎信息
                )
                return None, None  # 返回None元组
        else:  # 响应状态码非200
            logger.error(f"request.get failed: {response.status_code}")  # 记录错误日志
            return None, None  # 返回None元组
    except Exception as e:  # 捕获异常
        logger.error(f"Exception: {e}")  # 记录异常日志
        return None, None  # 返回None元组


def register_memory_region(model, transfer_engine):  # 注册模型参数的内存区域到传输引擎
    if importlib.util.find_spec("torch") is None:  # 如果未安装torch
        return register_memory_region_v1(model, transfer_engine)  # 使用v1版本注册内存区域
    else:  # 否则
        return register_memory_region_v2(model, transfer_engine)  # 使用v2版本注册内存区域


def register_memory_region_v1(model, transfer_engine):  # v1版本内存区域注册（逐参数注册）
    start_tic = time.time()  # 记录开始时间

    weight_mr_dict = {}  # 权重内存区域字典
    for name, weight in model.named_parameters():  # 遍历模型所有命名参数
        ret = transfer_engine.register_memory(  # 将权重内存注册到传输引擎
            weight.data_ptr(), weight.numel() * weight.element_size()  # 传入数据指针和字节大小
        )
        if ret != 0:  # 如果注册失败
            raise RuntimeError(  # 抛出运行时错误
                f"register memory failed for weight {name}, error: {ret}"  # 注册内存失败的错误信息
            )
        weight_mr_dict[name] = (  # 记录权重内存信息
            weight.data_ptr(),  # 数据指针
            weight.numel(),  # 元素数量
            weight.element_size(),  # 每个元素的字节大小
        )

    end_tic = time.time()  # 记录结束时间
    logger.debug(f"Register memory region time: {(end_tic - start_tic):.4f}s")  # 记录注册耗时
    return weight_mr_dict  # 返回权重内存区域字典


def register_memory_region_v2(model, transfer_engine):  # v2版本内存区域注册（合并连续内存块注册）
    start_tic = time.time()  # 记录开始时间

    weight_mr_dict = {}  # 权重内存区域字典
    weight_addr_set = set()  # 权重地址集合
    for name, weight in model.named_parameters():  # 遍历模型所有命名参数
        weight_mr_dict[name] = (  # 记录权重内存信息
            weight.data_ptr(),  # 数据指针
            weight.numel(),  # 元素数量
            weight.element_size(),  # 每个元素的字节大小
        )
        weight_addr_set.add(weight.data_ptr())  # 将数据指针添加到地址集合

    import torch  # 导入torch模块

    memory_snapshot = torch.cuda.memory.memory_snapshot()  # 获取CUDA内存快照
    weight_blocks_for_reg_mr = []  # 需要注册的权重内存块列表
    # Blocks in each segment have continuous physical addresses,
    # 每个段中的内存块具有连续的物理地址，
    # so they can be merged for memory registration.
    # 因此可以合并进行内存注册。
    for segment in memory_snapshot:  # 遍历内存快照中的每个段
        current_weight_block = None  # 当前权重块
        blocks = segment.get("blocks", [])  # 获取段中的内存块列表
        for block in blocks:  # 遍历每个内存块
            address = block.get("address", -1)  # 获取内存块地址
            size = block.get("size", -1)  # 获取内存块大小
            state = block.get("state", "")  # 获取内存块状态
            if address < 0 or size < 0 or state == "":  # 如果地址、大小或状态无效
                continue  # 跳过此块
            # Only register active allocated memory blocks that hold weights.
            # 仅注册持有权重的活跃已分配内存块。
            if state == "active_allocated":  # 如果内存块为活跃已分配状态
                if address in weight_addr_set:  # 如果该地址是权重地址
                    if current_weight_block is None:  # 如果当前没有合并块
                        current_weight_block = (address, size)  # 创建新的合并块
                    elif current_weight_block[0] + current_weight_block[1] == address:  # 如果地址与当前块连续
                        current_weight_block = (  # 扩展当前合并块
                            current_weight_block[0],  # 保留起始地址
                            current_weight_block[1] + size,  # 增加大小
                        )
                    else:  # 地址不连续
                        weight_blocks_for_reg_mr.append(current_weight_block)  # 保存当前合并块
                        current_weight_block = (address, size)  # 创建新的合并块
        if current_weight_block is not None:  # 如果最后一个合并块不为空
            weight_blocks_for_reg_mr.append(current_weight_block)  # 保存最后一个合并块

    # Register merged memory blocks that hold weights.
    # 注册持有权重的合并内存块。
    for weight_block in weight_blocks_for_reg_mr:  # 遍历所有需要注册的权重块
        address, size = weight_block  # 解包地址和大小
        ret = transfer_engine.register_memory(address, size)  # 将合并块注册到传输引擎
        if ret != 0:  # 如果注册失败
            raise RuntimeError(  # 抛出运行时错误
                f"register memory failed for weight block at address {address} with size {size}, error: {ret}"  # 注册内存失败的错误信息
            )

    end_tic = time.time()  # 记录结束时间
    logger.debug(f"Register memory region v2 time: {(end_tic - start_tic):.4f}s")  # 记录注册耗时
    return weight_mr_dict  # 返回权重内存区域字典
