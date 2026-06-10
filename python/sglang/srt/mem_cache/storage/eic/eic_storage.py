# EIC 存储后端实现
# 该模块实现了基于 EIC（弹性智能缓存）的 HiCache 存储后端，支持零拷贝和通用模式的 KV 缓存远程存储与检索，
# 包含 GPU 直连 RDMA、GPU-NIC 亲和性配置等高级功能

import json  # 导入JSON解析模块
import logging  # 导入日志模块
import os  # 导入操作系统模块
import time  # 导入时间模块
from typing import Any, List, Optional, Tuple  # 导入类型提示

import eic  # 导入EIC客户端库
import torch  # 导入PyTorch深度学习框架
import yaml  # 导入YAML配置文件解析库

from sglang.srt.mem_cache.hicache_storage import (  # 从SGLang的hicache_storage模块导入基础类
    HiCacheStorage,  # HiCache存储基类
    HiCacheStorageConfig,  # HiCache存储配置类
    HiCacheStorageExtraInfo,  # HiCache存储额外信息类
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache  # 导入主机端KV缓存内存池

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


TensorPoolSize = 2048  # 张量池大小，用于内存池管理

REMOTE_EIC_YAML_ENV_VAR = "REMOTE_EIC_YAML"  # 远程EIC配置文件路径的环境变量名

# gpu direct rdma for kv set  # KV设置时是否启用GPU直连RDMA
G_EnableKVSetGPUDirect = False  # 全局标志：KV设置GPU直连RDMA，默认关闭

# gpu direct rdma for kv get  # KV获取时是否启用GPU直连RDMA
G_EnableKVGetGPUDirect = False  # 全局标志：KV获取GPU直连RDMA，默认关闭

# gpu nic affinity  # GPU与网卡亲和性
G_EnableGPUNicAffinity = False  # 全局标志：是否启用GPU-NIC亲和性，默认关闭

# default H20 gpu nic affinity  # 默认H20 GPU网卡亲和性配置
GPUNicAffinity = {  # GPU设备到网卡的映射
    "cuda:0": "eth1",  # GPU 0 映射到 eth1
    "cuda:1": "eth1",  # GPU 1 映射到 eth1
    "cuda:2": "eth2",  # GPU 2 映射到 eth2
    "cuda:3": "eth2",  # GPU 3 映射到 eth2
    "cuda:4": "eth3",  # GPU 4 映射到 eth3
    "cuda:5": "eth3",  # GPU 5 映射到 eth3
    "cuda:6": "eth4",  # GPU 6 映射到 eth4
    "cuda:7": "eth4",  # GPU 7 映射到 eth4
}

# default H20 cpu nic affinity  # 默认H20 CPU网卡亲和性配置
CPUNicAffinity = {  # CPU侧GPU设备到网卡的映射
    "cuda:0": "cpu",  # GPU 0 使用CPU网卡
    "cuda:1": "cpu",  # GPU 1 使用CPU网卡
    "cuda:2": "cpu",  # GPU 2 使用CPU网卡
    "cuda:3": "cpu",  # GPU 3 使用CPU网卡
    "cuda:4": "cpu",  # GPU 4 使用CPU网卡
    "cuda:5": "cpu",  # GPU 5 使用CPU网卡
    "cuda:6": "cpu",  # GPU 6 使用CPU网卡
    "cuda:7": "cpu",  # GPU 7 使用CPU网卡
}


def get_eic_config_file_path():  # 获取EIC配置文件路径
    if os.environ.get(REMOTE_EIC_YAML_ENV_VAR) is not None:  # 如果设置了环境变量
        logger.info(f"eic init with env var {REMOTE_EIC_YAML_ENV_VAR}")  # 记录使用环境变量初始化
        config_file = os.environ.get(REMOTE_EIC_YAML_ENV_VAR)  # 从环境变量获取配置文件路径
    else:  # 如果未设置环境变量
        config_file = "/sgl-workspace/config/remote-eic.yaml"  # 使用默认配置文件路径
        logger.info(f"eic init with default config, config_file {config_file}")  # 记录使用默认配置初始化
    return config_file  # 返回配置文件路径


class FlexibleKVCacheMemoryPool:  # 灵活KV缓存内存池类，用于管理RDMA注册的内存区域
    def __init__(self, conn, kvcache_shape, kvcache_dtype, device):  # 初始化内存池
        self.connection = conn  # 保存EIC连接对象

        if device.startswith("cpu") and G_EnableGPUNicAffinity:  # 如果设备是CPU且启用了GPU-NIC亲和性
            gpu_id = torch.cuda.current_device()  # 获取当前GPU设备ID
            self.device = CPUNicAffinity["cuda:" + str(gpu_id)]  # 根据亲和性配置选择设备
            # current memory pool size is 5 times of CPU TensorPoolSize  # 当前内存池大小是CPU TensorPoolSize的5倍
            mempool_size = TensorPoolSize * 5  # CPU模式下内存池大小为5倍
        else:  # 其他设备
            self.device = device  # 直接使用指定设备
            mempool_size = TensorPoolSize  # 内存池大小为默认值

        self.kvcache_shape = kvcache_shape  # 保存KV缓存的形状
        self.kvcache_dtype = kvcache_dtype  # 保存KV缓存的数据类型

        self.kv_cache_numel = 1  # 计算KV缓存的总元素数
        for i in self.kvcache_shape:  # 遍历形状的每个维度
            self.kv_cache_numel *= i  # 累乘计算总元素数

        self.free_data_addr = set()  # 空闲数据地址索引集合
        self.data_ptr_to_index = dict()  # 数据指针到索引的映射字典

        if self.device.startswith("cpu"):  # 如果设备是CPU
            self.kvcache_mempool = torch.zeros(  # 创建CPU内存池，使用固定内存
                (mempool_size,) + kvcache_shape,  # 内存池形状：池大小+KV缓存形状
                dtype=kvcache_dtype,  # 数据类型
                device=self.device,  # 设备
                pin_memory=True,  # 使用固定内存以加速数据传输
            )
        else:  # 如果是GPU设备
            self.kvcache_mempool = torch.zeros(  # 创建GPU内存池
                (mempool_size,) + kvcache_shape, dtype=kvcache_dtype, device=self.device  # 内存池形状和数据类型
            )

        for i in range(mempool_size):  # 初始化空闲地址集合和指针映射
            self.free_data_addr.add(i)  # 将索引添加到空闲集合
            self.data_ptr_to_index[self.kvcache_mempool[i].data_ptr()] = i  # 记录数据指针到索引的映射

        meminfo = eic.MemoryInfo()  # 创建EIC内存信息对象
        meminfo.type = eic.MemoryType.MEMORY_CUDA  # 设置内存类型为CUDA
        meminfo.cuda_id = 0  # 设置CUDA设备ID
        vals = eic.IOBuffers()  # 创建IO缓冲区列表
        vals.append(  # 将内存池添加到IO缓冲区
            self.kvcache_mempool.data_ptr(),  # 内存池数据指针
            self.kvcache_mempool.numel() * self.kvcache_mempool.element_size(),  # 内存池总字节数
            True,  # 表示这是RDMA可访问的内存
        )
        self.connection.register_memory(vals, meminfo)  # 向EIC连接注册内存区域
        logger.info(  # 记录内存池分配信息
            f"allocate memory pool, size {self.kvcache_mempool.numel() * self.kvcache_mempool.element_size()}, device {self.device}"  # 输出内存池大小和设备
        )

    def try_allocate_kv_cache(self, shape, dtype, count=1):  # 尝试从内存池分配指定数量的KV缓存
        if len(self.free_data_addr) < count:  # 如果空闲地址不足
            return None  # 返回None表示分配失败

        numel = 1  # 计算请求的总元素数
        for i in shape:  # 遍历形状的每个维度
            numel *= i  # 累乘计算总元素数
        if numel != self.kv_cache_numel or dtype != self.kvcache_dtype:  # 如果元素数或类型不匹配
            logger.error(  # 记录错误日志
                f"allocate from mempool failed, self.kvcache_shape {self.kvcache_shape}, dtype {self.kvcache_dtype}, require shape {shape}, dtype {dtype}"  # 输出内存池和请求的形状及类型
            )
            return None  # 返回None表示分配失败

        ret = []  # 分配结果列表
        for _ in range(count):  # 分配指定数量的缓存块
            free_index = self.free_data_addr.pop()  # 从空闲集合中弹出一个索引
            ret.append(self.kvcache_mempool[free_index])  # 将对应的内存块添加到结果列表
        return ret  # 返回分配的缓存块列表

    def free_to_mempool(self, data_ptr):  # 将数据指针对应的内存块释放回内存池
        if data_ptr not in self.data_ptr_to_index:  # 如果数据指针不在映射中
            logger.error(  # 记录错误日志
                f"free_to_mempool failed, data_ptr {data_ptr} not in allocated_data_addr"  # 输出指针不在已分配地址中
            )
            return  # 直接返回
        self.free_data_addr.add(self.data_ptr_to_index[data_ptr])  # 将索引添加回空闲集合

    def check_data_ptr_allocated(self, data_ptr):  # 检查数据指针是否已分配
        return data_ptr in self.data_ptr_to_index  # 返回指针是否在映射中

    def left_count(self):  # 获取内存池中剩余可用块数
        return len(self.free_data_addr)  # 返回空闲地址集合的大小


class EICStorage(HiCacheStorage):  # EIC存储类，继承自HiCacheStorage基类
    def __init__(  # 初始化方法
        self, hicache_config: HiCacheStorageConfig, memory_pool_host: HostKVCache  # 接收存储配置和主机端内存池
    ):
        global G_EnableKVSetGPUDirect, G_EnableKVGetGPUDirect  # 声明全局变量
        global GPUNicAffinity, CPUNicAffinity, G_EnableGPUNicAffinity  # 声明全局变量

        config_file = get_eic_config_file_path()  # 获取EIC配置文件路径
        if os.path.exists(config_file) is False:  # 如果配置文件不存在
            logger.error(f"config file {config_file} not exists")  # 记录错误日志
            raise RuntimeError(f"eic config file {config_file} not exists")  # 抛出运行时异常

        with open(config_file, "r") as fin:  # 打开配置文件
            config = yaml.safe_load(fin)  # 安全加载YAML配置

        remote_url = config.get("remote_url", None)  # 获取远程URL
        if remote_url is None:  # 如果远程URL为空
            AssertionError("remote_url is None")  # 断言失败

        endpoint = remote_url[len("eic://") :]  # 从URL中提取端点地址，去掉"eic://"前缀

        logger.info(f"eic remote_url:" + remote_url + " endpoint: " + endpoint)  # 记录远程URL和端点信息

        eic_instance_id = config.get("eic_instance_id", None)  # 获取EIC实例ID
        logger.info(f"eic instance_id: {eic_instance_id}")  # 记录实例ID

        eic_thread_num = config.get("eic_thread_num", 1)  # 获取EIC线程数，默认为1
        logger.info(f"eic thread_num: {eic_thread_num}")  # 记录线程数

        eic_log_dir = config.get("eic_log_dir", None)  # 获取EIC日志目录
        logger.info(f"eic log_dir: {eic_log_dir}")  # 记录日志目录

        eic_log_level = config.get("eic_log_level", 2)  # 获取EIC日志级别，默认为2
        logger.info(f"eic log_level: {eic_log_level}")  # 记录日志级别

        eic_trans_type = config.get("eic_trans_type", 3)  # 获取EIC传输类型，默认为3
        logger.info(f"eic trans_type: {eic_trans_type}")  # 记录传输类型

        eic_flag_file = config.get("eic_flag_file", None)  # 获取EIC标志文件路径
        logger.info(f"eic flag_file: {eic_flag_file}")  # 记录标志文件路径

        # GDR now is not used  # GPU直连RDMA当前未使用
        G_EnableKVSetGPUDirect = (  # 设置KV设置的GPU直连RDMA标志
            config.get("enable_kvset_gpu_direct", False) and torch.cuda.is_available()  # 配置启用且CUDA可用时开启
        )
        logger.debug(f"eic enable_kvset_gpu_direct: {G_EnableKVSetGPUDirect}")  # 记录KV设置GPU直连状态

        G_EnableKVGetGPUDirect = (  # 设置KV获取的GPU直连RDMA标志
            config.get("enable_kvget_gpu_direct", False) and torch.cuda.is_available()  # 配置启用且CUDA可用时开启
        )
        logger.debug(f"eic enable_kvget_gpu_direct: {G_EnableKVGetGPUDirect}")  # 记录KV获取GPU直连状态

        self.model_name = hicache_config.model_name  # 保存模型名称

        # rdma  # RDMA相关配置
        enable_kv_set_direct = config.get("enable_kvset_direct", True)  # 获取KV设置直连配置，默认启用
        logger.info(f"eic enable_kv_set_direct: {enable_kv_set_direct}")  # 记录KV设置直连状态
        self.enable_kv_set_direct = enable_kv_set_direct  # 保存KV设置直连标志

        enable_kv_get_direct = config.get("enable_kvget_direct", True)  # 获取KV获取直连配置，默认启用
        logger.info(f"eic enable_kv_get_direct: {enable_kv_get_direct}")  # 记录KV获取直连状态
        self.enable_kv_get_direct = enable_kv_get_direct  # 保存KV获取直连标志

        # gpu nic affinity  # GPU网卡亲和性配置
        G_EnableGPUNicAffinity = config.get("enable_gpu_nic_affinity", False)  # 获取GPU-NIC亲和性开关，默认关闭
        logger.info(f"eic enable_gpu_nic_affinity: {G_EnableGPUNicAffinity}")  # 记录GPU-NIC亲和性状态
        self.enable_gpu_nic_affinity = G_EnableGPUNicAffinity  # 保存GPU-NIC亲和性标志

        if G_EnableGPUNicAffinity:  # 如果启用了GPU-NIC亲和性
            if "gpu_nic_affinity_config" in config:  # 如果配置中有GPU亲和性配置
                GPUNicAffinity = json.loads(config["gpu_nic_affinity_config"])  # 解析GPU亲和性配置
            if "cpu_nic_affinity_config" in config:  # 如果配置中有CPU亲和性配置
                CPUNicAffinity = json.loads(config["cpu_nic_affinity_config"])  # 解析CPU亲和性配置
            logger.info(f"eic gpu nic affinity {GPUNicAffinity}")  # 记录GPU亲和性配置
            logger.info(f"eic cpu nic affinity {CPUNicAffinity}")  # 记录CPU亲和性配置

        eic_namespace = config.get("eic_namespace", "")  # 获取EIC命名空间，默认为空
        logger.info(f"eic namespace: {eic_namespace}")  # 记录命名空间
        self.eic_namespace = eic_namespace  # 保存EIC命名空间

        if not os.path.exists(eic_log_dir) and not os.path.isdir(eic_log_dir):  # 如果日志目录不存在
            os.makedirs(eic_log_dir, exist_ok=True)  # 创建日志目录

        self.connection = eic.Client()  # 创建EIC客户端
        init_option = eic.InitOption()  # 创建初始化选项
        init_option.log_dir = eic_log_dir  # 设置日志目录
        init_option.log_level = eic.LogLevel(eic_log_level)  # 设置日志级别
        init_option.transport_type = eic.TransportType(eic_trans_type)  # 设置传输类型
        init_option.flag_file = eic_flag_file  # 设置标志文件

        if G_EnableGPUNicAffinity:  # 如果启用了GPU-NIC亲和性
            gpu_id = torch.cuda.current_device()  # 获取当前GPU设备ID
            init_option.multi_net_local_interface_names = GPUNicAffinity[  # 设置多网卡本地接口名称
                "cuda:" + str(gpu_id)  # 根据GPU ID选择网卡
            ]
            logger.info(  # 记录GPU亲和性配置
                f"gpu {gpu_id} set gpu nic affinity to {init_option.multi_net_local_interface_names}"  # 输出GPU到网卡的映射
            )

        ret = self.connection.init(eic_instance_id, endpoint, init_option)  # 初始化EIC客户端连接
        if ret != 0:  # 如果初始化失败
            logger.error(f"fail to init eic client, ret: {ret}")  # 记录错误日志
            raise RuntimeError("EIC Client Init Failed.")  # 抛出运行时异常
        self.warmup()  # 执行预热操作

        self.memory_pool_host = memory_pool_host  # 保存主机端内存池
        self.host_kvcache_layout = self.memory_pool_host.layout  # 保存主机端KV缓存布局
        self.trans_type = eic.TransportType(eic_trans_type)  # 保存传输类型
        self.kv_cache_dtype = self.memory_pool_host.dtype  # 保存KV缓存数据类型
        self.is_mla_model = hicache_config.is_mla_model  # 保存是否为MLA模型标志
        self.rank = hicache_config.tp_rank  # 保存当前张量并行秩
        self.world_size = hicache_config.tp_size  # 保存张量并行世界大小
        self.page_size = self.memory_pool_host.page_size  # 保存页面大小
        self.use_zero_copy = self.memory_pool_host.layout == "page_first"  # 判断是否使用零拷贝模式
        if not self.use_zero_copy:  # 如果不使用零拷贝模式
            self.kv_cache_shape = self.memory_pool_host.get_data_page(  # 获取数据页形状
                0, flat=True  # 获取第0页的展平形状
            ).shape  # 取shape属性
            if self.enable_kv_set_direct:  # 如果启用KV设置直连
                self.kv_cache_write_mem_pool = FlexibleKVCacheMemoryPool(  # 创建写内存池
                    self.connection, self.kv_cache_shape, self.kv_cache_dtype, "cpu"  # 使用CPU设备
                )
            if self.enable_kv_get_direct:  # 如果启用KV获取直连
                self.kv_cache_get_mem_pool = FlexibleKVCacheMemoryPool(  # 创建读内存池
                    self.connection, self.kv_cache_shape, self.kv_cache_dtype, "cpu"  # 使用CPU设备
                )
        self._init_eic_prefix()  # 初始化EIC键前缀

    def warmup(self):  # EIC客户端预热方法
        logger.info("begin warm up eic client")  # 记录开始预热
        start_time = time.perf_counter()  # 记录开始时间
        num_warmup = 1024  # 预热键数量
        preheat_keys = ["warmup_key_" + str(i) for i in range(num_warmup)]  # 生成预热键列表
        batch_size = 32  # 每批处理的键数量
        for i in range(0, num_warmup, batch_size):  # 分批执行预热
            keys_vec = eic.StringVector()  # 创建字符串向量
            for key in preheat_keys[i : i + batch_size]:  # 遍历当前批次的键
                keys_vec.append(key)  # 将键添加到向量中
            exist_option = eic.ExistOption()  # 创建存在性检查选项
            _, _ = self.connection.mexist(keys_vec, exist_option)  # 批量检查键是否存在（预热连接）
        logger.info(  # 记录预热完成信息
            f"finish eic client warm up, warm up cost {time.perf_counter() - start_time:.2f} seconds"  # 输出预热耗时
        )

    def register_mem_pool_host(self, memory_pool_host: HostKVCache) -> None:  # 注册主机端内存池到EIC连接
        # no need judge meminfo type, cuda_id, etc.  # 无需判断内存信息类型、CUDA ID等
        meminfo = eic.MemoryInfo()  # 创建EIC内存信息对象
        meminfo.type = eic.MemoryType.MEMORY_CUDA  # 设置内存类型为CUDA
        meminfo.cuda_id = 0  # 设置CUDA设备ID
        vals = eic.IOBuffers()  # 创建IO缓冲区列表
        buffer = memory_pool_host.kv_buffer  # 获取主机端KV缓存缓冲区
        vals.append(  # 将缓冲区添加到IO缓冲区
            buffer.data_ptr(),  # 缓冲区数据指针
            buffer.numel() * buffer.element_size(),  # 缓冲区总字节数
            True,  # 表示这是RDMA可访问的内存
        )
        self.connection.register_memory(vals, meminfo)  # 向EIC连接注册内存区域

    def _init_eic_prefix(self):  # 初始化EIC键前缀
        if self.is_mla_model:  # 如果是MLA模型
            self.eic_prefix = (  # 设置MLA模型的键前缀
                f"{self.model_name}_mla_att_{self.host_kvcache_layout}@sglang"  # 格式：模型名_mla_att_布局@sglang
            )
        else:  # 如果是MHA模型
            self.eic_prefix = f"{self.model_name}_mha_attn_{self.host_kvcache_layout}_{self.rank}_{self.world_size}_@sglang"  # 格式：模型名_mha_attn_布局_秩_世界大小@sglang

    def _get_eic_key(self, keys: List[str]) -> str:  # 将原始键转换为EIC键，添加前缀
        return [f"{self.eic_prefix}_{key}" for key in keys]  # 为每个键添加EIC前缀

    def set(  # 设置单个KV缓存数据
        self,
        key: str,  # 键
        value: Optional[Any] = None,  # 值（可选）
        target_location: Optional[Any] = None,  # 目标位置（可选）
        target_size: Optional[Any] = None,  # 目标大小（可选）
    ) -> bool:  # 返回是否成功
        # now is not used  # 当前未使用
        if self.use_zero_copy:  # 如果使用零拷贝模式
            return self.zero_copy_batch_set([key], [target_location])  # 调用零拷贝批量设置
        else:  # 如果使用通用模式
            return self.generic_batch_set([key], [value])  # 调用通用批量设置

    # target_locations and target_sizes are not used for now  # target_locations和target_sizes当前未使用
    def batch_set(  # 批量设置KV缓存数据
        self,
        keys: List[str],  # 键列表
        values: Optional[Any] = None,  # 值列表（可选）
        target_locations: Optional[Any] = None,  # 目标位置列表（可选）
        target_sizes: Optional[Any] = None,  # 目标大小列表（可选）
    ) -> bool:  # 返回是否成功
        if len(keys) == 0:  # 如果键列表为空
            return True  # 直接返回成功
        if self.use_zero_copy:  # 如果使用零拷贝模式
            return self.zero_copy_batch_set(keys, values)  # 调用零拷贝批量设置
        else:  # 如果使用通用模式
            return self.generic_batch_set(keys, values)  # 调用通用批量设置

    def get(  # 获取单个KV缓存数据
        self,
        key,  # 键
        target_location: Optional[Any] = None,  # 目标位置（可选）
        target_size: Optional[Any] = None,  # 目标大小（可选）
    ) -> torch.Tensor | None:  # 返回张量或None
        # now is not used  # 当前未使用
        if self.use_zero_copy:  # 如果使用零拷贝模式
            return self.zero_copy_batch_get([key], [target_location])  # 调用零拷贝批量获取
        else:  # 如果使用通用模式
            return self.generic_batch_get([key], [target_location])  # 调用通用批量获取

    # use for v1 interface, and shound not be called directly  # 用于v1接口，不应直接调用
    def batch_get(  # 批量获取KV缓存数据
        self,
        keys: List[str],  # 键列表
        target_locations: Optional[Any] = None,  # 目标位置列表（可选）
        target_sizes: Optional[Any] = None,  # 目标大小列表（可选）
    ) -> List[torch.Tensor | None]:  # 返回张量列表或None列表
        assert len(keys) == len(target_locations)  # 断言键和目标位置数量一致
        if len(keys) == 0:  # 如果键列表为空
            return None  # 返回None
        if self.use_zero_copy:  # 如果使用零拷贝模式
            return self.zero_copy_batch_get(keys, target_locations)  # 调用零拷贝批量获取
        else:  # 如果使用通用模式
            return self.generic_batch_get(keys, target_locations)  # 调用通用批量获取

    def _batch_exists_impl(self, keys) -> List[bool]:  # 批量检查键是否存在的内部实现
        if len(keys) == 0:  # 如果键列表为空
            return 0  # 返回0
        eic_keys = self._get_eic_key(keys)  # 将原始键转换为EIC键
        logger.debug(f"eic exists {len(keys)}")  # 记录检查的键数量
        result = []  # 存储检查结果
        exist_bs = 1024  # 每批检查的键数量
        for i in range(0, len(eic_keys), exist_bs):  # 分批检查键是否存在
            batch_keys = eic_keys[i : i + exist_bs]  # 获取当前批次的键
            keys_vec = eic.StringVector()  # 创建字符串向量
            for key in batch_keys:  # 遍历当前批次的键
                keys_vec.append(key)  # 将键添加到向量中
            exist_option = eic.ExistOption()  # 创建存在性检查选项
            exist_option.ns = self.eic_namespace  # 设置命名空间
            status_code, exist_outcome = self.connection.mexist(keys_vec, exist_option)  # 批量检查键是否存在
            if status_code != eic.StatusCode.SUCCESS:  # 如果检查失败
                logger.error(  # 记录错误日志
                    f"eic exists {len(keys)} failed, status_code {status_code}"  # 输出检查失败的状态码
                )
                result.extend([False] * len(batch_keys))  # 所有键标记为不存在
            for err_code in exist_outcome.status_codes:  # 遍历每个键的状态码
                result.append(err_code == eic.StatusCode.SUCCESS)  # 将成功状态转换为布尔值添加到结果
        return result  # 返回检查结果列表

    def exists(self, key) -> bool:  # 检查单个键是否存在
        exist_num = self.batch_exists([key])  # 调用批量存在性检查
        return exist_num == 1  # 返回是否存在

    def batch_exists(  # 批量检查键是否存在，返回前缀匹配成功的键数量
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None  # 键列表和额外信息（可选）
    ) -> int:  # 返回存在的键数量
        if len(keys) == 0:  # 如果键列表为空
            return 0  # 返回0
        if self.use_zero_copy and not self.is_mla_model:  # 零拷贝且非MLA模型
            keys = self._get_mha_zero_copy_keys(keys)  # 将MHA键拆分为K和V键
        exist_mask = self._batch_exists_impl(keys)  # 执行批量存在性检查
        prefix_success = 0  # 前缀匹配成功计数
        for exist in exist_mask:  # 遍历检查结果
            if exist:  # 如果存在
                prefix_success += 1  # 计数加1
            else:  # 如果不存在
                break  # 前缀匹配中断
        if not self.is_mla_model and self.use_zero_copy:  # 非MLA模型且零拷贝
            prefix_success = prefix_success // 2  # MHA模式下K和V成对，需要除以2
        return prefix_success  # 返回前缀匹配成功的键数量

    def delete(self, key) -> None:  # 删除指定键的KV缓存
        eic_keys = self._get_eic_key([key])  # 将原始键转换为EIC键
        keys_vec = eic.StringVector()  # 创建字符串向量
        for eic_key in eic_keys:  # 遍历EIC键
            keys_vec.append(eic_key)  # 将键添加到向量中
        del_option = eic.DelOption()  # 创建删除选项
        self.connection.mdel(keys_vec, del_option)  # 批量删除键

    def clear(self) -> None:  # 清空所有KV缓存（当前未实现）
        return  # 直接返回

    # Not used for now  # 当前未使用
    def _filter_kv_cache(self, total_len) -> Tuple[int, int]:  # 根据张量并行配置过滤KV缓存的键范围
        mean_len = total_len // self.world_size  # 每个秩的平均键长度
        remainder = total_len % self.world_size  # 余数
        tp_keys_len = mean_len + (1 if self.rank < remainder else 0)  # 当前秩的键长度
        start = self.rank * mean_len + min(self.rank, remainder)  # 起始位置
        end = start + tp_keys_len  # 结束位置
        logger.debug(f"start: {start}, end: {end}, tp_keys_len: {tp_keys_len}")  # 记录键范围
        return start, end  # 返回起始和结束位置

    def zero_copy_batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:  # 零拷贝模式批量设置KV缓存
        logger.debug(f"eic zero copy set {len(keys)} keys")  # 记录设置的键数量
        if len(keys) == 0:  # 如果键列表为空
            return True  # 直接返回成功
        eic_keys = self._get_eic_key(keys)  # 将原始键转换为EIC键
        keys_vec = eic.StringVector()  # 创建字符串向量
        vals_vec = eic.IOBuffers()  # 创建IO缓冲区列表
        # set data key & value  # 设置数据键和值
        for i, key in enumerate(eic_keys):  # 遍历EIC键
            # set data key & value  # 设置数据键和值
            keys_vec.append(key)  # 将键添加到字符串向量
            vals_vec.append(  # 将值添加到IO缓冲区
                values[i].data_ptr(),  # 值的数据指针
                values[i].element_size() * values[i].numel(),  # 值的总字节数
                True,  # 表示这是RDMA可访问的内存
            )
        # set options  # 设置选项
        set_option = eic.SetOption()  # 创建设置选项
        set_option.ns = self.eic_namespace  # 设置命名空间
        set_option.ttl_second = -1  # 设置永不过期
        status_code, set_outcome = self.connection.mset(keys_vec, vals_vec, set_option)  # 批量设置数据
        if status_code != eic.StatusCode.SUCCESS:  # 如果设置失败
            logger.error(f"eic mset {len(keys)} failed, status_code {status_code}")  # 记录错误日志
            return [False] * len(keys)  # 返回全部失败
        else:  # 如果设置成功
            logger.debug(f"eic zero copy mset {len(keys)} success")  # 记录成功日志
        return [True] * len(keys)  # 返回全部成功

    def zero_copy_batch_get(  # 零拷贝模式批量获取KV缓存
        self, keys: List[str], values: List[torch.Tensor]  # 键列表和值张量列表
    ) -> List[bool]:  # 返回每个键的成功状态列表
        logger.debug(f"eic zero copy get {len(keys)} keys")  # 记录获取的键数量
        # Get Data: generate data keys and vals  # 获取数据：生成数据键和值
        get_data_start_time = time.perf_counter()  # 记录获取开始时间
        eic_keys = self._get_eic_key(keys)  # 将原始键转换为EIC键
        data_keys = eic.StringVector()  # 创建数据键字符串向量
        data_vals = eic.IOBuffers()  # 创建数据值IO缓冲区列表
        success_mask = [True] * len(keys)  # 初始化成功掩码，默认全部成功
        count = len(keys)  # 键的数量
        for i, key in enumerate(eic_keys):  # 遍历EIC键
            data_keys.append(key)  # 将键添加到字符串向量
            data_vals.append(  # 将值缓冲区添加到IO缓冲区列表
                values[i].data_ptr(),  # 值的数据指针
                values[i].element_size() * values[i].numel(),  # 值的总字节数
                True,  # 表示这是RDMA可访问的内存
            )

        # Get data: recv data buffer tensor  # 获取数据：接收数据缓冲区张量
        get_option = eic.GetOption()  # 创建获取选项
        get_option.ns = self.eic_namespace  # 设置命名空间
        status_code, data_vals, get_outcome = self.connection.mget(  # 批量获取数据
            data_keys, get_option, data_vals  # 传入键、选项和值缓冲区
        )

        if status_code != eic.StatusCode.SUCCESS:  # 如果获取失败
            if status_code == eic.StatusCode.PARTIAL_FAILED:  # 如果部分失败
                for i, err_code in enumerate(get_outcome.status_codes):  # 遍历每个键的状态码
                    success = err_code == eic.StatusCode.SUCCESS  # 判断是否成功
                    if success:  # 如果成功
                        logger.debug(f"eic get data {eic_keys[i]} success")  # 记录成功日志
                    else:  # 如果失败
                        logger.error(  # 记录错误日志
                            f"eic get data {eic_keys[i]} failed, err_code {err_code}"  # 输出获取数据失败和错误码
                        )
                        success_mask[i] = False  # 标记该键获取失败
            else:  # 如果完全失败
                logger.error(  # 记录错误日志
                    f"eic mget {len(eic_keys)} keys failed, status_code {status_code}"  # 输出批量获取失败和状态码
                )
                success_mask = [False] * len(keys)  # 标记全部失败
                return success_mask  # 返回失败掩码

        get_data_end_time = time.perf_counter()  # 记录获取结束时间
        get_data_execution_time = (get_data_end_time - get_data_start_time) * 1e6  # 计算获取耗时（微秒）
        logger.debug(f"eic get {count} keys data cost %.2f us", get_data_execution_time)  # 记录获取耗时
        return success_mask  # 返回成功掩码

    def generic_batch_set(  # 通用模式批量设置KV缓存
        self,
        keys: List[str],  # 键列表
        values: List[torch.Tensor],  # 值张量列表
    ) -> List[bool]:  # 返回每个键的成功状态列表
        assert len(keys) == len(values)  # 断言键和值数量一致
        logger.debug(f"eic generic set {len(keys)} keys")  # 记录设置的键数量
        if len(keys) == 0:  # 如果键列表为空
            return True  # 直接返回成功
        eic_keys = self._get_eic_key(keys)  # 将原始键转换为EIC键
        keys_vec = eic.StringVector()  # 创建字符串向量
        vals_vec = eic.IOBuffers()  # 创建IO缓冲区列表
        count = len(keys)  # 键的数量
        registered = False  # 内存是否已注册标志
        items = []  # 从内存池分配的项列表
        if self.enable_kv_set_direct:  # 如果启用KV设置直连
            values_data_ptrs = []  # 值数据指针列表
            items = self.kv_cache_write_mem_pool.try_allocate_kv_cache(  # 尝试从写内存池分配KV缓存
                self.kv_cache_shape, self.kv_cache_dtype, count  # 传入形状、类型和数量
            )
            if items is None:  # 如果分配失败
                logger.warning("can not allocate tensor from pool")  # 记录分配失败警告
                for i, value in enumerate(values):  # 遍历值张量
                    values_data_ptrs.append(  # 直接使用原始张量的数据指针
                        (value.data_ptr(), value.element_size() * value.numel(), False)  # 指针、字节数、未注册标志
                    )
            else:  # 如果分配成功
                objs = items  # 保存分配的项
                registered = True  # 标记为已注册
                for i, key in enumerate(eic_keys):  # 遍历EIC键
                    temp = objs[i].reshape(values[i].shape).contiguous()  # 将分配的项重塑为值张量的形状并确保连续
                    temp.copy_(values[i])  # 将值张量复制到分配的项中
                    if temp.data_ptr() != objs[i].data_ptr():  # 如果重塑后数据指针发生变化
                        registered = False  # 标记为未注册
                        temp = temp.cpu()  # 将数据移到CPU
                    values_data_ptrs.append(  # 添加数据指针信息
                        (
                            temp.data_ptr(),  # 数据指针
                            temp.element_size() * temp.numel(),  # 总字节数
                            registered,  # 是否已注册
                        )
                    )

            for i, key in enumerate(eic_keys):  # 遍历EIC键
                keys_vec.append(key)  # 将键添加到字符串向量
                data_ptr, data_size, registered = values_data_ptrs[i]  # 解包数据指针信息
                vals_vec.append(data_ptr, data_size, registered)  # 将数据添加到IO缓冲区
        else:  # 如果未启用KV设置直连
            # use tensor direct  # 直接使用张量
            for i, key in enumerate(eic_keys):  # 遍历EIC键
                keys_vec.append(key)  # 将键添加到字符串向量
                vals_vec.append(  # 将值张量添加到IO缓冲区
                    values[i].data_ptr(),  # 值的数据指针
                    values[i].element_size() * values[i].numel(),  # 值的总字节数
                    False,  # 未注册标志
                )

        # set options  # 设置选项
        set_option = eic.SetOption()  # 创建设置选项
        set_option.ns = self.eic_namespace  # 设置命名空间
        set_option.ttl_second = -1  # 设置永不过期
        status_code, set_outcome = self.connection.mset(keys_vec, vals_vec, set_option)  # 批量设置数据
        if status_code != eic.StatusCode.SUCCESS:  # 如果设置失败
            logger.error(f"eic mset {len(eic_keys)} failed, status_code {status_code}")  # 记录错误日志
        else:  # 如果设置成功
            logger.debug(f"eic mset {len(eic_keys)} success")  # 记录成功日志

        if self.enable_kv_set_direct and items is not None:  # 如果启用了KV设置直连且有分配的项
            for item in items:  # 遍历分配的项
                self.kv_cache_write_mem_pool.free_to_mempool(item.data_ptr())  # 将项释放回写内存池

        err_code = set_outcome.status_codes[0]  # 获取第一个键的状态码
        if err_code != eic.StatusCode.SUCCESS:  # 如果设置失败
            logger.error(f"set data key {len(eic_keys)} failed, err_code {err_code}")  # 记录错误日志
            return [False] * len(keys)  # 返回全部失败

        logger.debug(f"set data key {len(eic_keys)} success")  # 记录成功日志
        return [True] * len(keys)  # 返回全部成功

    def generic_batch_get(  # 通用模式批量获取KV缓存
        self, keys: List[str], buffers: List[torch.Tensor]  # 键列表和缓冲区张量列表
    ) -> List[bool]:  # 返回每个键的成功状态列表
        # all success or all fail  # 全部成功或全部失败
        logger.debug(f"eic generic get {len(keys)} keys")  # 记录获取的键数量
        eic_keys = self._get_eic_key(keys)  # 将原始键转换为EIC键
        get_data_start_time = time.perf_counter()  # 记录获取开始时间
        data_keys = eic.StringVector()  # 创建数据键字符串向量
        data_vals = eic.IOBuffers()  # 创建数据值IO缓冲区列表
        count = len(eic_keys)  # 键的数量
        registered = False  # 内存是否已注册标志
        items = []  # 从内存池分配的项列表
        success_mask = [True] * len(keys)  # 初始化成功掩码，默认全部成功
        if self.enable_kv_get_direct:  # 如果启用KV获取直连
            items = self.kv_cache_get_mem_pool.try_allocate_kv_cache(  # 尝试从读内存池分配KV缓存
                self.kv_cache_shape, self.kv_cache_dtype, count  # 传入形状、类型和数量
            )
            if items is None:  # 如果分配失败
                logger.warning("can not allocate tensor from pool")  # 记录分配失败警告
                for i, key in enumerate(eic_keys):  # 遍历EIC键
                    data_keys.append(key)  # 将键添加到字符串向量
                    data_vals.append(  # 直接使用缓冲区的数据指针
                        buffers[i].data_ptr(),  # 缓冲区的数据指针
                        buffers[i].element_size() * buffers[i].numel(),  # 缓冲区的总字节数
                        False,  # 未注册标志
                    )
            else:  # 如果分配成功
                registered = True  # 标记为已注册
                for i, key in enumerate(eic_keys):  # 遍历EIC键
                    data_keys.append(key)  # 将键添加到字符串向量
                    data_vals.append(  # 使用内存池分配的项作为接收缓冲区
                        items[i].data_ptr(),  # 分配项的数据指针
                        items[i].element_size() * items[i].numel(),  # 分配项的总字节数
                        registered,  # 是否已注册标志
                    )

        else:  # 如果未启用KV获取直连
            for i, key in enumerate(eic_keys):  # 遍历EIC键
                data_keys.append(key)  # 将键添加到字符串向量
                data_vals.append(  # 直接使用缓冲区的数据指针
                    buffers[i].data_ptr(),  # 缓冲区的数据指针
                    buffers[i].element_size() * buffers[i].numel(),  # 缓冲区的总字节数
                    False,  # 未注册标志
                )

        # Get data: recv data buffer tensor  # 获取数据：接收数据缓冲区张量
        get_option = eic.GetOption()  # 创建获取选项
        get_option.ns = self.eic_namespace  # 设置命名空间
        status_code, data_vals, get_outcome = self.connection.mget(  # 批量获取数据
            data_keys, get_option, data_vals  # 传入键、选项和值缓冲区
        )

        if status_code != eic.StatusCode.SUCCESS:  # 如果获取失败
            if status_code == eic.StatusCode.PARTIAL_FAILED:  # 如果部分失败
                for i, err_code in enumerate(get_outcome.status_codes):  # 遍历每个键的状态码
                    success = err_code == eic.StatusCode.SUCCESS  # 判断是否成功
                    if success:  # 如果成功
                        logger.debug(f"eic get data {eic_keys[i]} success")  # 记录成功日志
                    else:  # 如果失败
                        logger.error(  # 记录错误日志
                            f"eic get data {eic_keys[i]} failed, err_code {err_code}"  # 输出获取数据失败和错误码
                        )
                        success_mask[i] = False  # 标记该键获取失败
            else:  # 如果完全失败
                logger.error(  # 记录错误日志
                    f"eic mget {len(eic_keys)} keys failed, status_code {status_code}"  # 输出批量获取失败和状态码
                )
                success_mask = [False] * len(keys)  # 标记全部失败

        if registered:  # 如果使用了内存池注册
            for i, item in enumerate(items):  # 遍历分配的项
                if success_mask[i]:  # 如果该键获取成功
                    buffers[i].copy_(item)  # 将数据从内存池项复制到缓冲区
                self.kv_cache_get_mem_pool.free_to_mempool(item.data_ptr())  # 将项释放回读内存池

        get_data_end_time = time.perf_counter()  # 记录获取结束时间
        get_data_execution_time = (get_data_end_time - get_data_start_time) * 1e6  # 计算获取耗时（微秒）
        logger.debug(f"eic get {count} keys data cost %.2f us", get_data_execution_time)  # 记录获取耗时
        return success_mask  # 返回成功掩码

    def _get_mha_zero_copy_keys(self, keys: List[str]) -> List[str]:  # 将MHA模式的键拆分为K和V键
        new_keys = []  # 新键列表
        for k in keys:  # 遍历原始键
            new_keys.append(f"{k}_k")  # 添加K键
            new_keys.append(f"{k}_v")  # 添加V键
        return new_keys  # 返回拆分后的键列表

    def _get_mha_zero_copy_values(  # 将MHA模式的值张量拆分为K和V值
        self, values: List[torch.Tensor]  # 值张量列表
    ) -> List[torch.Tensor]:  # 返回拆分后的值张量列表
        new_values = []  # 新值列表
        for value in values:  # 遍历值张量
            new_values.append(value[0])  # 添加K值（第一维）
            new_values.append(value[1])  # 添加V值（第二维）
        return new_values  # 返回拆分后的值列表

    def _batch_get_preprocess(self, keys, host_indices):  # 批量获取的预处理，准备键和值
        page_num = len(host_indices) // self.page_size  # 计算页面数量
        # use memory pool directly or dummy page  # 直接使用内存池或虚拟页
        values = (  # 准备值张量列表
            [  # 零拷贝模式：直接使用内存池数据页
                self.memory_pool_host.get_data_page(  # 获取数据页
                    host_indices[i * self.page_size], flat=False  # 不展平
                )
                for i in range(page_num)  # 遍历每个页面
            ]
            if self.use_zero_copy  # 零拷贝模式
            else [  # 通用模式：使用虚拟展平数据页
                self.memory_pool_host.get_dummy_flat_data_page()  # 获取虚拟展平数据页
                for _ in range(page_num)  # 遍历每个页面
            ]
        )

        if self.use_zero_copy and not self.is_mla_model:  # 零拷贝且非MLA模型
            keys = self._get_mha_zero_copy_keys(keys)  # 将MHA键拆分为K和V键
            values = self._get_mha_zero_copy_values(values)  # 将MHA值拆分为K和V值

        return keys, values  # 返回处理后的键和值

    def _batch_get_postprocess(self, host_indices, values, results):  # 批量获取的后处理，处理获取结果
        page_num = len(host_indices) // self.page_size  # 计算页面数量

        if self.use_zero_copy:  # 如果使用零拷贝模式
            if not self.is_mla_model:  # 如果不是MLA模型
                results = [  # 合并K和V的结果
                    (results[2 * i] and results[2 * i + 1]) for i in range(page_num)  # K和V都成功才算成功
                ]
                results = results[:page_num]  # 截取到页面数量
            return results  # 返回处理后的结果

        # dummy page copy to host memory pool  # 将虚拟页数据复制到主机端内存池
        for i in range(page_num):  # 遍历每个页面
            if not results[i]:  # 如果获取失败
                break  # 中断循环
            self.memory_pool_host.set_from_flat_data_page(  # 将展平数据页写入主机端内存池
                host_indices[i * self.memory_pool_host.page_size], values[i]  # 指定索引和值
            )

        return results  # 返回处理后的结果

    def batch_get_v1(  # V1接口的批量获取方法
        self,
        keys: List[str],  # 键列表
        host_indices: torch.Tensor,  # 主机端索引张量
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息（可选）
    ) -> List[bool]:  # 返回每个键的成功状态列表
        keys, values = self._batch_get_preprocess(keys, host_indices)  # 预处理键和值
        results = self.batch_get(keys, values)  # 执行批量获取
        return self._batch_get_postprocess(host_indices, values, results)  # 后处理并返回结果

    def _batch_set_preprocess(self, keys, host_indices):  # 批量设置的预处理，准备键和值
        page_num = len(host_indices) // self.page_size  # 计算页面数量
        flat = not self.use_zero_copy  # 通用模式需要展平，零拷贝模式不需要
        values = [  # 准备值张量列表
            self.memory_pool_host.get_data_page(  # 获取数据页
                host_indices[i * self.page_size], flat=flat  # 根据模式决定是否展平
            )
            for i in range(page_num)  # 遍历每个页面
        ]

        if self.use_zero_copy and not self.is_mla_model:  # 零拷贝且非MLA模型
            keys = self._get_mha_zero_copy_keys(keys)  # 将MHA键拆分为K和V键
            values = self._get_mha_zero_copy_values(values)  # 将MHA值拆分为K和V值

        return keys, values  # 返回处理后的键和值

    def batch_set_v1(  # V1接口的批量设置方法
        self,
        keys: List[str],  # 键列表
        host_indices: torch.Tensor,  # 主机端索引张量
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息（可选）
    ) -> List[bool]:  # 返回每个键的成功状态列表
        keys, values = self._batch_set_preprocess(keys, host_indices)  # 预处理键和值
        results = self.batch_set(keys, values)  # 执行批量设置
        return results  # 返回设置结果
