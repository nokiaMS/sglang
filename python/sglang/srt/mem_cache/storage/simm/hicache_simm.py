# 文件说明：HiCacheSiMM - 基于SiMM（Shared InfiniBand Memory Manager）的HiCache存储后端实现
# 本文件实现了通过RDMA网络进行KV缓存分布式存储的SiMM连接器，支持零拷贝的批量读写操作，
# 并根据NUMA拓扑自动选择RDMA网卡设备，支持MHA和MLA两种注意力后端模式。

import json  # JSON解析库
import logging  # 日志记录库
import os  # 操作系统接口
import re  # 正则表达式库
import time  # 时间相关功能
import uuid  # 唯一标识符生成
from collections import defaultdict  # 默认字典
from dataclasses import dataclass  # 数据类装饰器
from datetime import datetime  # 日期时间
from typing import Any, Dict, List, Optional  # 类型提示

import torch  # PyTorch深度学习框架

from sglang.srt.mem_cache.hicache_storage import (  # 导入HiCache存储基类及配置
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache  # 导入主机端KV缓存

# Third Party  # 第三方库
try:
    from simm.kv import BlockView, Store, register_mr, set_flag  # 导入SiMM客户端API
except ImportError as e:
    raise ImportError(
        "Please install simm by following the instructions at https://github.com/scitix/SiMM "
        "to run SGLang with SimmConnector."
    ) from e  # 若SiMM未安装则抛出导入错误

SGLANG_HICACHE_SIMM_JSON_ENV_VAR = "SGLANG_HICACHE_SIMM_CONFIG_PATH"  # SiMM配置文件路径的环境变量名

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@dataclass
class SiMMConfig:  # SiMM配置数据类
    manager_address: str  # 管理器地址（IP:端口）
    clnt_threadpool_size: int  # 客户端线程池大小
    enable_profile: bool  # 是否启用性能分析

    @staticmethod
    def from_file() -> "SiMMConfig":  # 从JSON文件加载配置
        """Load the config from a JSON file."""  # 从JSON文件加载配置
        if os.environ.get(SGLANG_HICACHE_SIMM_JSON_ENV_VAR) is None:  # 检查环境变量是否设置
            raise RuntimeError(
                f"Config file path not set. Please set {SGLANG_HICACHE_SIMM_JSON_ENV_VAR}"
            )  # 未设置则抛出运行时错误
        file_path = os.environ.get(SGLANG_HICACHE_SIMM_JSON_ENV_VAR)  # 获取配置文件路径
        try:
            with open(file_path) as fin:  # 打开配置文件
                config = json.load(fin)  # 解析JSON内容
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {file_path}: {str(e)}")  # 加载失败抛出错误

        if "manager_address" not in config:  # 检查是否包含必需的manager_address字段
            raise ValueError("Manager_address is required in config file")  # 缺少必需字段则抛出值错误

        return SiMMConfig(  # 返回SiMMConfig实例
            manager_address=config.get("manager_address"),  # 获取管理器地址
            clnt_threadpool_size=config.get("clnt_threadpool_size", 10),  # 获取线程池大小，默认10
            enable_profile=config.get("enable_profile", False),  # 获取是否启用性能分析，默认False
        )

    @staticmethod
    def load_from_extra_config(extra_config: dict) -> "SiMMConfig":  # 从extra_config字典加载配置
        """Load config from extra_config dictionary."""  # 从extra_config字典加载配置
        if "manager_address" not in extra_config:  # 检查是否包含必需的manager_address
            raise ValueError("manager_address is required in extra_config")  # 缺少必需字段则抛出值错误

        return SiMMConfig(  # 返回SiMMConfig实例
            manager_address=extra_config.get("manager_address"),  # 获取管理器地址
            clnt_threadpool_size=extra_config.get("clnt_threadpool_size", 10),  # 获取线程池大小，默认10
            enable_profile=extra_config.get("enable_profile", False),  # 获取是否启用性能分析，默认False
        )


def get_current_process_numa() -> int:  # 获取当前进程所在的NUMA节点
    """
    Return value: numa_node of current process, failed return -1  # 返回值：当前进程的NUMA节点号，失败返回-1
    """
    try:
        # get current cpu  # 获取当前CPU
        with open("/proc/self/stat", "r") as f:  # 读取进程状态信息
            stat_data = f.read()  # 读取全部内容

        # the 39th field is processor  # 第39个字段是处理器编号
        fields = stat_data.split()  # 按空白分割字段
        if len(fields) < 39:  # 字段数不足则返回-1
            return -1
        current_cpu = int(fields[38])  # 获取当前CPU编号（第39个字段，索引38）
        numa_path = f"/sys/devices/system/cpu/cpu{current_cpu}/node0"  # 构造NUMA节点路径
        if os.path.exists(numa_path) and os.path.islink(numa_path):  # 检查路径是否存在且为符号链接
            link_target = os.readlink(numa_path)  # 读取符号链接目标
            # parse numa node from path  # 从路径中解析NUMA节点号
            match = re.search(r"node(\d+)$", link_target)  # 用正则匹配节点号
            if match:
                return int(match.group(1))  # 返回NUMA节点号

        return -1  # 无法确定NUMA节点则返回-1
    except Exception:
        return -1  # 发生异常返回-1


def get_numa_nic_mapping() -> Dict[int, List[str]]:  # 获取NUMA节点到RDMA网卡设备的映射
    """
    Return value: Dict[numa_node, List(rdma_device_name)]  # 返回值：字典{NUMA节点号: [RDMA设备名列表]}
    """
    ib_root = "/sys/class/infiniband"  # InfiniBand设备根路径
    device_map = defaultdict(list)  # NUMA节点到设备名的映射字典

    if not os.path.exists(ib_root):  # 检查InfiniBand设备路径是否存在
        logger.error(f"SiMM ERROR: {ib_root} not found. Are RDMA drivers loaded?")  # 记录错误日志
        return []  # 返回空列表

    for device_name in os.listdir(ib_root):  # 遍历所有InfiniBand设备
        numa_path = os.path.join(ib_root, device_name, "device", "numa_node")  # 构造NUMA节点路径
        numa_node = -1  # default value, if system is UMA.  # 默认值-1，如果系统是UMA架构

        try:
            if os.path.exists(numa_path):  # 检查NUMA节点文件是否存在
                with open(numa_path, "r") as f:  # 打开NUMA节点文件
                    content = f.read().strip()  # 读取并去除空白
                    numa_node = int(content)  # 转换为整数
        except (IOError, ValueError):  # 捕获IO错误和值错误
            pass  # 忽略异常
        device_map[numa_node].append(device_name)  # 将设备名添加到对应NUMA节点的列表中

    return device_map  # 返回NUMA节点到设备名的映射


class HiCacheSiMM(HiCacheStorage):  # 基于SiMM的HiCache存储实现类

    def __init__(  # 初始化方法
        self, storage_config: HiCacheStorageConfig = None, mem_pool: HostKVCache = None  # 存储配置和主机内存池
    ):
        try:
            extra_config = (  # 获取额外配置
                getattr(storage_config, "extra_config", None)  # 尝试获取extra_config属性
                if storage_config  # 如果storage_config不为None
                else None  # 否则为None
            )
            # Load configuration with manager_address prioritized from extra_config if available  # 优先从extra_config加载配置中的manager_address
            if (
                extra_config is not None  # 如果extra_config不为None
                and extra_config.get("manager_address") is not None  # 且包含manager_address
            ):
                # Load from extra_config  # 从extra_config加载配置
                self.config = SiMMConfig.load_from_extra_config(extra_config)  # 从额外配置加载
                logger.info("SiMM Configuration loaded from extra_config successfully.")  # 记录成功日志
            else:
                # Load from config file  # 从配置文件加载配置
                self.config = SiMMConfig.from_file()  # 从文件加载
                logger.info("SiMM Configuration loaded from file successfully.")  # 记录成功日志

            # Check if extra_backend_tag should be passed to SiMM data server  # 检查是否需要传递extra_backend_tag给SiMM数据服务器
            self.extra_backend_tag = None  # 初始化extra_backend_tag为None
            if extra_config and "extra_backend_tag" in extra_config:  # 如果extra_config中包含extra_backend_tag
                self.extra_backend_tag = extra_config["extra_backend_tag"]  # 获取extra_backend_tag
                logger.info(f"Using extra_backend_tag: {self.extra_backend_tag}")  # 记录使用的tag

            # Set nic device according to current process numa node  # 根据当前进程的NUMA节点设置网卡设备
            nic_mapping = get_numa_nic_mapping()  # 获取NUMA到网卡设备的映射
            logger.info(f"SiMM NUMA-awared allocation: {nic_mapping}")  # 记录NUMA感知分配信息
            current_numa = get_current_process_numa()  # 获取当前进程的NUMA节点
            if current_numa >= 0:  # 如果成功获取NUMA节点
                rdma_devices = nic_mapping.get(current_numa)  # 获取对应NUMA节点的RDMA设备列表
                if rdma_devices is not None and len(rdma_devices) > 0:  # 如果有可用的RDMA设备
                    rdma_device_str = ",".join(rdma_devices)  # 将设备名用逗号连接
                    os.environ["SICL_NET_DEVICES"] = rdma_device_str  # 设置RDMA设备环境变量
                    logger.info(f"SiMM using rdma {rdma_device_str}")  # 记录使用的RDMA设备

            # Set simm log path: /var/log/simm/{filename_ts}-{pid}/simm_clnt.log  # 设置SiMM日志路径
            filename_ts = datetime.now().strftime("%Y%m%d-%H%M%S")  # 生成时间戳文件名
            log_file_path: str = (  # 构造日志文件路径
                f"/var/log/simm/{filename_ts}-{os.getpid()}/simm_clnt.log"  # 包含时间戳和进程ID
            )

            cm_ip = self.config.manager_address.split(":")[0]  # 提取管理器IP地址
            cm_port = self.config.manager_address.split(":")[1]  # 提取管理器端口号
            set_flag("cm_primary_node_ip", cm_ip)  # 设置管理器主节点IP标志
            set_flag("cm_primary_node_port", cm_port)  # 设置管理器主节点端口标志
            set_flag("clnt_log_file", log_file_path)  # 设置客户端日志文件路径标志
            set_flag("clnt_thread_pool_size", str(self.config.clnt_threadpool_size))  # 设置线程池大小标志

            self.store = Store()  # 创建SiMM Store实例
            logger.info("SiMM store setup successfully.")  # 记录Store创建成功
            self.mr_ext = None  # 初始化内存注册扩展为None

            self.warmup()  # 执行预热操作
            logger.info("SiMM store warmup successfully.")  # 记录预热成功

            if storage_config is not None:  # 如果存储配置不为None
                self.model_name = storage_config.model_name  # 获取模型名称
                self.is_mla_backend = storage_config.is_mla_model  # 是否为MLA后端
                self.local_rank = storage_config.tp_rank  # 张量并行排名
                self.pp_rank = storage_config.pp_rank  # 流水线并行排名
                self.pp_size = storage_config.pp_size  # 流水线并行大小
            else:  # 否则使用默认值
                self.model_name = ""  # 默认模型名为空
                self.is_mla_backend = False  # 默认非MLA后端
                self.local_rank = 0  # 默认本地排名为0
                self.pp_rank = 0  # 默认流水线排名为0
                self.pp_size = 1  # 默认流水线大小为1

            self.enable_pp = self.pp_size > 1  # 是否启用流水线并行
            if self.enable_pp:  # 如果启用流水线并行
                self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"  # MHA后缀包含TP和PP排名
                self.mla_suffix = f"{self.pp_rank}"  # MLA后缀仅包含PP排名
            else:  # 否则
                self.mha_suffix = f"{self.local_rank}"  # MHA后缀仅包含TP排名
                self.mla_suffix = ""  # MLA后缀为空

        except ValueError as e:  # 捕获值错误
            logger.error("Configuration loading failed: %s", e)  # 记录配置加载失败
            raise  # 重新抛出异常
        except Exception as exc:  # 捕获其他异常
            logger.error("An error occurred while loading the configuration: %s", exc)  # 记录加载配置错误
            raise  # 重新抛出异常

    def warmup(self):  # 预热SiMM客户端
        """Dryrun a key to warmup SiMM client"""  # 执行一次干运行来预热SiMM客户端
        logger.info("begin warm up SiMM client")  # 记录开始预热
        start_time = time.perf_counter_ns()  # 记录开始时间（纳秒）
        warmup_key = "sglang_simm_warmup_key" + uuid.uuid4().hex  # 生成唯一的预热键
        warmup_tensor = torch.frombuffer(  # 将键字符串转换为张量
            bytearray(warmup_key.encode()), dtype=torch.uint8  # 编码为字节数组再转为uint8张量
        )
        warmup_size = 4 * 1024  # 4 KB  # 预热数据大小为4KB
        block = self.store.allocate(warmup_size)  # 分配预热数据块
        block_ = block.as_ref()  # 获取数据块引用
        block_[: len(warmup_key)] = warmup_tensor  # 将预热张量写入数据块
        if self.store.put(warmup_key, block.view()) != 0:  # 执行put操作并检查返回值
            logger.warning(f"SiMM client warmup put key {warmup_key} failed")  # 记录put失败警告
        if not self.store.exists(warmup_key):  # 检查键是否存在
            logger.warning(f"SiMM client warmup key {warmup_key} not exists")  # 记录键不存在警告
        got_block = self.store.allocate(warmup_size)  # 分配获取数据块
        if self.store.get(warmup_key, got_block.view()) < 0:  # 执行get操作并检查返回值
            logger.warning(f"SiMM client warmup get key {warmup_key} failed")  # 记录get失败警告
        if not all(got_block.as_ref()[: len(warmup_key)] == warmup_tensor):  # 验证获取的数据是否一致
            logger.warning(f"SiMM client warmup key {warmup_key} data wrong")  # 记录数据不一致警告
        logger.info(
            f"finish SiMM client warm up, cost {(time.perf_counter_ns() - start_time)/1000:.2f} us"  # 记录预热完成及耗时
        )

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):  # 注册主机端内存池
        super().register_mem_pool_host(mem_pool_host)  # 调用父类注册方法
        assert self.mem_pool_host.layout in [  # 检查内存布局是否支持
            "page_first",  # 页优先布局
            "page_first_direct",  # 页优先直接布局
        ], "simm storage backend only support page first or page first direct layout"  # SiMM仅支持这两种布局
        buffer = self.mem_pool_host.kv_buffer  # 获取KV缓存缓冲区
        try:
            self.mr_ext = register_mr(buffer)  # 注册内存区域到SiMM
            if self.mr_ext is None:  # 如果注册失败
                logger.error(
                    f"Failed to register buffer, {buffer=}, please check buffer and RDMA network"  # 记录注册失败及缓冲区信息
                )
                raise RuntimeError(f"Failed to register buffer to SiMM")  # 抛出运行时错误
        except TypeError as err:  # 捕获类型错误
            logger.error("Failed to register buffer to SiMM: %s", err)  # 记录注册失败
            raise TypeError("SiMM Register Buffer Error.") from err  # 抛出类型错误

    def _get_mha_buffer_meta(self, keys, indices):  # 获取MHA模式的缓冲区元数据
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)  # 获取页缓冲区指针和大小
        key_list = []  # 初始化键列表
        for key_ in keys:  # 遍历所有键
            key_list.append(f"{key_}_{self.mha_suffix}_k")  # 添加K键（带TP和PP后缀）
            key_list.append(f"{key_}_{self.mha_suffix}_v")  # 添加V键（带TP和PP后缀）
        if len(key_list) != len(ptr_list):  # 检查键数量与指针数量是否一致
            logger.error(
                f"key size {len(key_list)} not equal with incides ptr size {len(ptr_list)}"  # 记录不一致错误
            )
        assert len(key_list) == len(ptr_list)  # 断言键数量等于指针数量
        return key_list, ptr_list, element_size_list  # 返回键列表、指针列表和大小列表

    def _get_mla_buffer_meta(self, keys, indices):  # 获取MLA模式的缓冲区元数据
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)  # 获取页缓冲区指针和大小
        key_list = []  # 初始化键列表
        for key_ in keys:  # 遍历所有键
            key_list.append(f"{key_}_{self.mla_suffix}_k")  # 添加K键（带PP后缀）
        if len(key_list) != len(ptr_list):  # 检查键数量与指针数量是否一致
            logger.error(
                f"key size {len(key_list)} not equal with incides ptr size {len(ptr_list)}"  # 记录不一致错误
            )
        assert len(key_list) == len(ptr_list)  # 断言键数量等于指针数量
        return key_list, ptr_list, element_size_list  # 返回键列表、指针列表和大小列表

    def _batch_preprocess(self, keys, host_indices):  # 批量预处理，根据后端类型选择元数据获取方式
        assert len(keys) > 0  # 断言键列表非空
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size  # 断言键数量与索引数量匹配
        if self.is_mla_backend:  # 如果是MLA后端
            return self._get_mla_buffer_meta(keys, host_indices)  # 返回MLA缓冲区元数据
        else:  # 否则是MHA后端
            return self._get_mha_buffer_meta(keys, host_indices)  # 返回MHA缓冲区元数据

    def _batch_postprocess(self, results: List[int], is_set_operate=False):  # 批量后处理，将操作结果转换为布尔列表
        """
        for batch_get_into, results is Vector of integers,  # 对于batch_get_into，results是整数向量
            where each element is the number of bytes read on success, or a negative value on error  # 每个元素为成功读取的字节数，或错误时的负值
        for batch_put_from, results is Vector of integers,  # 对于batch_put_from，results是整数向量
            where each element is 0 on success, or a negative value on error  # 每个元素为成功时0，或错误时的负值
        """
        if self.is_mla_backend:  # 如果是MLA后端（每个键对应一个结果）
            return [k_res == 0 if is_set_operate else k_res > 0 for k_res in results]  # set操作检查等于0，get操作检查大于0
        else:  # MHA后端（每个键对应K和V两个结果）
            kv_pairs = zip(results[::2], results[1::2])  # 将结果配对为(K结果, V结果)
            return [
                (
                    (k_res == 0 and v_res == 0)  # set操作：K和V都成功
                    if is_set_operate
                    else (k_res > 0 and v_res > 0)  # get操作：K和V都成功
                )
                for k_res, v_res in kv_pairs  # 遍历每个K-V对
            ]

    def batch_get_v1(  # 批量获取KV缓存（v1版本）
        self,
        keys: List[str],  # 键列表
        host_indices: torch.Tensor,  # 主机端索引张量
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> List[bool]:  # 返回每个键是否获取成功
        # Apply extra_backend_tag prefix if available  # 如果有extra_backend_tag前缀则应用
        if self.extra_backend_tag is not None:  # 如果设置了额外后端标签
            prefix = self.extra_backend_tag  # 获取前缀
            keys = [f"{prefix}_{key}" for key in keys]  # 为每个键添加前缀

        t1 = time.perf_counter_ns()  # 记录开始时间（纳秒）
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)  # 批量预处理
        get_results = self._get_batch_zero_copy_impl(  # 执行零拷贝批量获取
            key_strs, buffer_ptrs, buffer_sizes
        )
        t2 = time.perf_counter_ns()  # 记录结束时间（纳秒）
        total_size = sum([k_res if k_res > 0 else 0 for k_res in get_results])  # 计算总读取字节数
        if self.config.enable_profile:  # 如果启用性能分析
            logger.info(
                f"SiMM batch_get_v1 {len(keys)} keys, total size: {total_size / 1024**2} MiB, \
                    using {(t2 - t1)/1000} us, Throughput: {total_size / 1024**3 / ((t2 - t1) / 1000**3):.2f} GiB/s"  # 记录性能分析信息
            )
        return self._batch_postprocess(get_results, is_set_operate=False)  # 后处理并返回结果

    def batch_set_v1(  # 批量设置KV缓存（v1版本）
        self,
        keys: List[str],  # 键列表
        host_indices: torch.Tensor,  # 主机端索引张量
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> List[bool]:  # 返回每个键是否设置成功
        # Apply extra_backend_tag prefix if available  # 如果有extra_backend_tag前缀则应用
        if self.extra_backend_tag is not None:  # 如果设置了额外后端标签
            prefix = self.extra_backend_tag  # 获取前缀
            keys = [f"{prefix}_{key}" for key in keys]  # 为每个键添加前缀

        t1 = time.perf_counter_ns()  # 记录开始时间（纳秒）
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)  # 批量预处理
        exist_result = self._batch_exist_impl(key_strs)  # 检查哪些键已存在
        t2 = time.perf_counter_ns()  # 记录检查完成时间
        if self.config.enable_profile:  # 如果启用性能分析
            logger.info(
                f"SiMM batch exists {len(keys)} keys, using {(t2 - t1)/1000} us"  # 记录exists耗时
            )

        set_keys = []  # 待设置的键列表
        set_buffer_ptrs = []  # 待设置的缓冲区指针列表
        set_buffer_sizes = []  # 待设置的缓冲区大小列表
        set_indices = []  # 待设置的结果索引列表
        set_results = [-1] * len(key_strs)  # 初始化所有结果为-1（失败）
        total_size = 0  # 总数据大小
        for i in range(len(key_strs)):  # 遍历所有键
            if not exist_result[i]:  # 如果键不存在
                set_keys.append(key_strs[i])  # 添加到待设置列表
                set_buffer_ptrs.append(buffer_ptrs[i])  # 添加对应缓冲区指针
                set_buffer_sizes.append(buffer_sizes[i])  # 添加对应缓冲区大小
                set_indices.append(i)  # 记录原始索引
                total_size += buffer_sizes[i]  # 累加数据大小
            else:  # 如果键已存在
                set_results[i] = 0  # 标记为成功（0表示set成功）

        # Only set non-existing keys to storage  # 仅将不存在的键设置到存储中
        if len(set_keys) > 0:  # 如果有待设置的键
            put_results = self._put_batch_zero_copy_impl(  # 执行零拷贝批量写入
                set_keys, set_buffer_ptrs, set_buffer_sizes
            )
            for i in range(len(set_indices)):  # 遍历设置结果
                set_results[set_indices[i]] = put_results[i]  # 将结果写回对应位置
        t3 = time.perf_counter_ns()  # 记录put完成时间
        if self.config.enable_profile:  # 如果启用性能分析
            logger.info(
                f"SiMM batch_put_v1 {len(keys)} keys, total size: {total_size / 1024**2} MiB, \
                    using {(t3 - t2)/1000} us, Throughput: {total_size / 1024**3 / ((t3 - t2) / 1000**3):.2f} GiB/s"  # 记录put性能分析信息
            )

        return self._batch_postprocess(set_results, is_set_operate=True)  # 后处理并返回结果

    def set(  # 设置单个键值对
        self,
        key,  # 键
        value: Optional[Any] = None,  # 值（未使用）
        target_location: Optional[List[int]] = None,  # 目标位置（缓冲区指针）
        target_sizes: Optional[List[int]] = None,  # 目标大小
    ) -> bool:  # 返回是否设置成功
        # Only support zero copy set for now  # 目前仅支持零拷贝设置
        assert target_location is not None and target_sizes is not None  # 断言目标位置和大小不为None
        exist_result = self._batch_exist_impl([key])  # 检查键是否已存在
        if exist_result[0]:  # 如果键已存在
            return True  # 直接返回成功
        put_result = self._put_batch_zero_copy_impl(  # 执行零拷贝写入
            [key], [target_location], [target_sizes]
        )
        return put_result[0] == 0  # 返回是否写入成功

    def batch_set(  # 批量设置键值对
        self,
        keys: List[str],  # 键列表
        values: Optional[List[torch.Tensor]] = None,  # 值列表（未使用）
        target_locations: Optional[List[int]] = None,  # 目标位置列表
        target_sizes: Optional[List[int]] = None,  # 目标大小列表
    ) -> bool:  # 返回是否全部设置成功
        # Only support zero copy set for now  # 目前仅支持零拷贝设置
        assert target_locations is not None and target_sizes is not None  # 断言目标位置和大小不为None
        assert len(keys) == len(target_locations) == len(target_sizes)  # 断言三者长度一致

        if len(keys) == 0:  # 如果键列表为空
            return False  # 返回失败

        for i in range(len(keys)):  # 遍历所有键
            if (
                keys[i] is None  # 键为None
                or target_locations[i] is None  # 目标位置为None
                or target_sizes[i] is None  # 目标大小为None
            ):
                return False  # 返回失败

        exist_result = self._batch_exist_impl(keys)  # 批量检查键是否存在
        set_keys = []  # 待设置的键列表
        set_target_locations = []  # 待设置的目标位置列表
        set_target_sizes = []  # 待设置的目标大小列表
        set_indices = []  # 待设置的结果索引列表
        for i in range(len(keys)):  # 遍历所有键
            if not exist_result[i]:  # 如果键不存在
                set_keys.append(keys[i])  # 添加到待设置列表
                set_target_locations.append(target_locations[i])  # 添加对应目标位置
                set_target_sizes.append(target_sizes[i])  # 添加对应目标大小
                set_indices.append(i)  # 记录原始索引
        # Only set non-existing keys to storage  # 仅将不存在的键设置到存储中
        put_result = self._put_batch_zero_copy_impl(  # 执行零拷贝批量写入
            set_keys, set_target_locations, set_target_sizes
        )
        for i in range(len(set_indices)):  # 遍历写入结果
            if put_result[i] == 0:  # 如果写入成功
                exist_result[set_indices[i]] = 1  # 标记为存在

        # return the number of consecutive successful operations from the start.  # 返回从开始连续成功的操作数
        success_count = 0  # 成功计数
        for i in range(len(keys)):  # 遍历所有键
            if exist_result[i] == 0:  # 如果存在失败
                break  # 跳出循环
            success_count += 1  # 成功计数加1
        return success_count == len(keys)  # 返回是否全部成功

    def get(  # 获取单个键的值
        self,
        key,  # 键
        target_location: Optional[Any] = None,  # 目标位置（缓冲区指针）
        target_sizes: Optional[Any] = None,  # 目标大小
    ) -> bool:  # 返回是否获取成功
        assert target_location is not None and target_sizes is not None  # 断言目标位置和大小不为None
        get_result = self._get_batch_zero_copy_impl(  # 执行零拷贝获取
            [key], [target_location], [target_sizes]
        )
        return get_result[0] >= 0  # 返回是否获取成功

    def batch_get(  # 批量获取键的值
        self,
        keys: List[str],  # 键列表
        target_locations: Optional[Any] = None,  # 目标位置列表
        target_sizes: Optional[Any] = None,  # 目标大小列表
    ) -> int:  # 返回成功获取的键数量
        assert len(keys) == len(target_locations) == len(target_sizes)  # 断言三者长度一致
        if len(keys) == 0:  # 如果键列表为空
            return 0  # 返回0
        get_result = self._get_batch_zero_copy_impl(  # 执行零拷贝批量获取
            keys, target_locations, target_sizes
        )
        if self.is_mla_backend:  # 如果是MLA后端
            key_multiplier = 1  # MLA每个键对应1个结果
        else:  # MHA后端
            key_multiplier = 2  # MHA每个键对应2个结果（K和V）
        for i in range(len(keys)):  # 遍历所有键的结果
            if get_result[i] < 0:  # 如果获取失败
                return i // key_multiplier  # 返回失败前的成功数量
        return len(keys) // key_multiplier  # 全部成功则返回总数

    def exists(self, key) -> bool:  # 检查单个键是否存在
        exist_result = self._batch_exist_impl([key])  # 批量检查存在性
        return exist_result[0]  # 返回第一个结果

    def batch_exists(  # 批量检查键是否存在
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None  # 键列表和额外信息
    ) -> int:  # 返回连续存在的键数量
        if self.is_mla_backend:  # 如果是MLA后端
            query_keys = [f"{key}_{self.mla_suffix}_k" for key in keys]  # 构造MLA查询键（仅K）
            key_multiplier = 1  # MLA每个键对应1个查询
        else:  # MHA后端
            query_keys = []  # 初始化查询键列表
            for key in keys:  # 遍历所有键
                query_keys.append(f"{key}_{self.mha_suffix}_k")  # 添加K键
                query_keys.append(f"{key}_{self.mha_suffix}_v")  # 添加V键
            key_multiplier = 2  # MHA每个键对应2个查询

        t1 = time.perf_counter_ns()  # 记录开始时间（纳秒）
        exist_result = self._batch_exist_impl(query_keys)  # 批量检查存在性
        t2 = time.perf_counter_ns()  # 记录结束时间（纳秒）
        if self.config.enable_profile:  # 如果启用性能分析
            logger.info(
                f"SiMM batch exists {len(keys)} keys, using {(t2 - t1)/1000} us"  # 记录exists耗时
            )
        for i in range(len(query_keys)):  # 遍历所有查询键
            if not exist_result[i]:  # 如果某个键不存在
                return i // key_multiplier  # 返回不存在之前的连续存在数量
        return len(query_keys) // key_multiplier  # 全部存在则返回总数

    def _put_batch_zero_copy_impl(  # 零拷贝批量写入的内部实现
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]  # 键列表、缓冲区指针和大小列表
    ) -> List[int]:  # 返回每个操作的结果码
        block_views = []  # 块视图列表
        for i in range(len(buffer_ptrs)):  # 遍历所有缓冲区
            block_view = BlockView.from_buffer(  # 从缓冲区创建块视图
                buffer_ptrs[i], buffer_sizes[i], self.mr_ext  # 传入指针、大小和内存注册扩展
            )
            block_views.append(block_view)  # 添加到列表
        return self.store.mput(key_strs, block_views)  # 批量写入并返回结果

    def _get_batch_zero_copy_impl(  # 零拷贝批量读取的内部实现
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]  # 键列表、缓冲区指针和大小列表
    ) -> List[int]:  # 返回每个操作的结果码
        block_views = []  # 块视图列表
        for i in range(len(buffer_ptrs)):  # 遍历所有缓冲区
            block_view = BlockView.from_buffer(  # 从缓冲区创建块视图
                buffer_ptrs[i], buffer_sizes[i], self.mr_ext  # 传入指针、大小和内存注册扩展
            )
            block_views.append(block_view)  # 添加到列表
        return self.store.mget(key_strs, block_views)  # 批量读取并返回结果

    def _batch_exist_impl(self, key_strs: List[str]) -> List[bool]:  # 批量检查键是否存在的内部实现
        return self.store.mexists(key_strs)  # 调用SiMM的mexists方法
