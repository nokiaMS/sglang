# 模型参数卸载器模块
# 提供多种卸载策略（V1 CPU卸载、V2分组卸载），将模型参数从GPU卸载到CPU或共享内存
# 支持预取和并行加载，优化GPU显存使用，适用于大模型推理场景
import logging  # 导入日志模块
import os  # 导入操作系统模块
from abc import ABC  # 导入抽象基类
from typing import Callable, Generator, List, Optional  # 导入类型提示

import torch  # 导入PyTorch
from torch.func import functional_call  # 导入函数式调用工具

from sglang.srt.distributed.naive_distributed import (  # 导入分布式通信工具
    NaiveDistributed,
    get_naive_distributed,
    set_naive_distributed,
)
from sglang.srt.layers.parameter import ModelWeightParameter  # 导入模型权重参数类
from sglang.srt.server_args import ServerArgs  # 导入服务器参数类
from sglang.srt.utils import MultiprocessingSerializer, is_pin_memory_available  # 导入工具函数
from sglang.srt.utils.host_shared_memory import (  # 导入主机共享内存管理工具
    HostSharedMemoryManager,
    get_host_shared_memory_manager,
    set_host_shared_memory_manager,
)

logger = logging.getLogger(__name__)  # 创建日志记录器

_SubmoduleAccessor = Callable[[torch.nn.Module], torch.nn.Module]  # 子模块访问器类型
_WhitelistParamNamesCreator = Callable[[torch.nn.Module], List[str]]  # 白名单参数名创建器类型


class BaseOffloader(ABC):  # 卸载器抽象基类
    def wrap_modules(  # 包装模块，默认直接收集到列表
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        return list(all_modules_generator)  # 将生成器转换为列表

    def post_init(self):  # 初始化后处理，默认不做任何事
        pass

    @property
    def forbid_copy_engine_usage(self):  # 是否禁止使用拷贝引擎，默认为否
        return False


class NoopOffloader(BaseOffloader):  # 空操作卸载器，不做任何卸载
    pass


# For simplicity use singleton, but can surely support multi instance  # 为简单起见使用单例，但也可以支持多实例
_instance: Optional[BaseOffloader] = NoopOffloader()  # 全局单例实例


def get_offloader():  # 获取全局卸载器实例
    assert _instance is not None  # 断言实例已初始化
    return _instance


def set_offloader(instance: BaseOffloader):  # 设置全局卸载器实例
    global _instance  # 声明使用全局变量
    _instance = instance


def create_offloader_from_server_args(server_args: ServerArgs, dp_rank: int):  # 根据服务器参数创建卸载器
    if server_args.cpu_offload_gb > 0:  # 如果指定了CPU卸载大小
        return OffloaderV1(  # 创建V1卸载器
            cpu_offload_max_bytes=int(server_args.cpu_offload_gb * 1024**3)
        )
    if server_args.offload_group_size > 0:  # 如果指定了卸载分组大小
        assert (
            server_args.cpu_offload_gb == 0
        ), "V2 offload does not support cpu_offload_gb yet"  # V2不支持cpu_offload_gb
        return OffloaderV2(  # 创建V2卸载器
            group_size=server_args.offload_group_size,
            num_in_group=server_args.offload_num_in_group,
            prefetch_step=server_args.offload_prefetch_step,
            mode=server_args.offload_mode,
            dp_rank=dp_rank,
            dp_size=server_args.dp_size,
        )
    return NoopOffloader()  # 默认返回空操作卸载器


class OffloaderV1(BaseOffloader):  # V1版本卸载器，简单地将参数卸载到CPU
    def __init__(self, cpu_offload_max_bytes: int):  # 初始化V1卸载器
        self._cpu_offload_bytes = 0  # 已卸载到CPU的字节数
        self._cpu_offload_max_bytes = cpu_offload_max_bytes  # CPU卸载最大字节数

    def wrap_modules(  # 包装模块，将参数卸载到CPU
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        return [self.maybe_offload_to_cpu(module) for module in all_modules_generator]  # 对每个模块尝试卸载

    def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:  # 将模块参数卸载到CPU（如果可能）
        if (params := next(module.parameters(), None)) is None:  # 如果模块没有参数
            return module  # 直接返回

        device = params.device  # 获取参数所在设备

        if device == torch.device("cpu"):  # 如果已在CPU上
            return module  # 无需卸载

        if self._cpu_offload_bytes >= self._cpu_offload_max_bytes:  # 如果已达到CPU卸载上限
            return module  # 不再卸载

        pin_memory = is_pin_memory_available()  # 检查是否可以使用pin_memory
        # offload parameters to CPU  # 将参数卸载到CPU
        # use pin_memory if possible, which helps cudagraph capture speed  # 尽可能使用pin_memory，有助于CUDA图捕获速度
        offloaded_parameters = False  # 标记是否有参数被卸载
        for p in module.parameters():  # 遍历模块所有参数
            if self._cpu_offload_bytes >= self._cpu_offload_max_bytes:  # 如果达到上限
                # we use per-parameter offloading  # 使用逐参数卸载
                # one module might have some parameters offloaded and some not  # 一个模块可能部分参数卸载，部分不卸载
                break

            # `torch.empty_like` does not support `pin_memory` argument  # torch.empty_like不支持pin_memory参数
            cpu_data = torch.empty_strided(  # 在CPU上创建空张量
                size=p.data.size(),
                stride=p.data.stride(),
                dtype=p.data.dtype,
                layout=p.data.layout,
                device="cpu",
                pin_memory=pin_memory,
            )
            cpu_data.copy_(p.data)  # 将数据复制到CPU
            p.data = cpu_data  # 替换参数数据为CPU版本
            self._cpu_offload_bytes += p.data.numel() * p.data.element_size()  # 更新已卸载字节数
            offloaded_parameters = True  # 标记有参数被卸载

        if offloaded_parameters:  # 如果有参数被卸载
            original_forward = module.forward  # 保存原始前向方法

            def forward(*args, **kwargs):  # 替换的前向方法，在前向时将参数移回GPU
                module.forward = original_forward  # 恢复原始前向方法
                device_state = {  # 将所有参数移到GPU
                    # here we blindly call `to(device)`  # 直接调用to(device)
                    # if the parameter is already on the device, it will be a no-op  # 如果参数已在设备上则为空操作
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }
                output = functional_call(module, device_state, args=args, kwargs=kwargs)  # 函数式调用
                module.forward = forward  # 恢复替换的前向方法
                return output

            module.forward = forward  # 替换前向方法

        return module


class OffloaderV2(BaseOffloader):  # V2版本卸载器，支持分组卸载和预取
    def __init__(  # 初始化V2卸载器
        self,
        group_size: int,  # 分组大小
        num_in_group: int,  # 每组中卸载的模块数
        prefetch_step: int,  # 预取步数
        mode: str,  # 卸载模式
        dp_rank: int,  # 数据并行排名
        dp_size: int,  # 数据并行大小
    ):
        self.group_size = group_size  # 保存分组大小
        self.num_in_group = num_in_group  # 保存每组卸载数
        self.prefetch_step = prefetch_step  # 保存预取步数
        self.mode = mode  # 保存卸载模式

        run_id = os.environ["SGLANG_RUN_ID"]  # 获取运行ID

        # Temporarily init inside Offloader, can move if other modules also need this  # 临时在Offloader内初始化，如果其他模块也需要可以移出
        if self.mode in {"sharded_gpu", "shm_cpu"}:  # 如果是分片GPU或共享内存CPU模式
            from sglang.srt.distributed import get_tensor_model_parallel_world_size

            assert (
                get_tensor_model_parallel_world_size() == 1
            ), "not yet support tp_size!=1"  # 尚不支持张量并行
            set_naive_distributed(  # 设置朴素分布式通信
                NaiveDistributed(
                    rank=dp_rank,
                    world_size=dp_size,
                    rendezvous=f"/tmp/{run_id}",
                )
            )
        if self.mode in {"shm_cpu"}:  # 如果是共享内存CPU模式
            set_host_shared_memory_manager(  # 设置主机共享内存管理器
                HostSharedMemoryManager(
                    base_name=run_id,
                )
            )

        self.offloaders = []  # 模块卸载器列表

    def wrap_modules(  # 包装模块，为需要卸载的子模块创建卸载器并注册钩子
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        assert len(self.offloaders) == 0, "should only call wrap_modules once"  # 只能调用一次

        alt_stream = torch.cuda.Stream()  # 创建辅助CUDA流

        all_modules = []  # 所有模块列表
        offload_submodules = []  # 需要卸载的子模块列表
        for module_index, module in enumerate(all_modules_generator):  # 遍历所有模块
            all_modules.append(module)  # 添加到全部模块列表
            if module_index % self.group_size >= self.group_size - self.num_in_group:  # 如果模块在卸载范围内
                submodule = submodule_accessor(module)  # 获取子模块
                whitelist_param_names = whitelist_param_names_creator(submodule)  # 获取白名单参数名
                logger.info(
                    f"[offloader] offload {module_index=} submodule={type(submodule)} params={whitelist_param_names} memory_allocated={torch.cuda.memory_allocated()}"
                )
                offload_submodules.append(submodule)  # 添加到卸载子模块列表
                self.offloaders.append(  # 创建模块卸载器
                    _ModuleOffloader(
                        mode=self.mode,
                        module=submodule,
                        alt_stream=alt_stream,
                        whitelist_param_names=whitelist_param_names,
                    )
                )

        for index, module in enumerate(offload_submodules):  # 为每个卸载子模块注册前向钩子
            _hook_module_forward_for_offloader(
                index=index,
                module=module,
                offloaders=self.offloaders,
                prefetch_step=self.prefetch_step,
            )

        return all_modules  # 返回所有模块列表

    def post_init(self):  # 初始化后处理，执行所有卸载器的post_init并预取初始模块
        for offloader in self.offloaders:  # 遍历所有模块卸载器
            offloader.post_init()  # 调用各自的后初始化

        for i in range(self.prefetch_step):  # 预取前N个模块
            self.offloaders[i].start_onload()  # 开始加载到GPU

    @property
    def forbid_copy_engine_usage(self):  # CPU模式下禁止使用拷贝引擎
        return self.mode == "cpu"


def _hook_module_forward_for_offloader(index, module, offloaders, prefetch_step):  # 为模块的前向方法注册卸载钩子
    def _on_forward_end():  # 前向完成后的回调
        offloaders[(index + prefetch_step) % len(offloaders)].start_onload()  # 预取后续模块
        offloaders[index].offload()  # 卸载当前模块

    _hook_module_forward_raw(  # 注册原始前向钩子
        module,
        on_forward_end=_on_forward_end,  # 前向结束回调
        get_parameter_and_buffer_dicts=lambda: offloaders[  # 获取参数和缓冲区字典
            index
        ].wait_and_get_device_tensors(),
    )


def _hook_module_forward_raw(module, on_forward_end, get_parameter_and_buffer_dicts):  # 替换模块的前向方法，在调用时使用卸载的参数
    original_forward = module.forward  # 保存原始前向方法

    def forward(*args, **kwargs):  # 替换的前向方法
        module.forward = original_forward  # 临时恢复原始前向方法
        output = functional_call(  # 使用函数式调用，传入GPU上的参数
            module, get_parameter_and_buffer_dicts(), args=args, kwargs=kwargs
        )
        on_forward_end()  # 调用前向结束回调
        module.forward = forward  # 恢复替换的前向方法
        return output

    module.forward = forward  # 替换前向方法


class _ModuleOffloader(ABC):  # 模块级卸载器，管理单个模块的参数卸载和加载
    def __init__(
        self,
        mode: str,  # 卸载模式
        module: torch.nn.Module,  # 要卸载的模块
        alt_stream: torch.cuda.Stream,  # 辅助CUDA流
        whitelist_param_names: List[str],  # 白名单参数名列表
    ):
        self.mode = mode  # 保存模式
        self.module = module  # 保存模块
        self.device = next(module.parameters()).device  # 获取参数所在设备
        self.alt_stream = alt_stream  # 保存辅助流

        assert self.device != torch.device(
            "cpu"
        ), "not handled device=cpu case yet (should skip this tensor)"  # 尚未处理CPU设备的情况

        self._device_tensors = None  # GPU上的参数字典（延迟初始化）
        self._load_event = None  # 加载完成事件

        param_dict = dict(self.module.named_parameters())  # 获取模块参数字典
        assert all(
            name in param_dict for name in whitelist_param_names
        ), f"{whitelist_param_names=} {list(param_dict.keys())=}"  # 确保白名单参数都存在

        self._param_offloaders = {  # 为每个白名单参数创建参数卸载器
            name: _BaseParamOffloader.create(mode, module=module, param_name=name)
            for name in whitelist_param_names
        }

    def post_init(self):  # 初始化后处理，调用各参数卸载器的post_init
        for name, param_offloader in self._param_offloaders.items():
            param_offloader.post_init()

    def start_onload(self):  # 开始将参数加载到GPU
        self.alt_stream.wait_stream(torch.cuda.current_stream())  # 等待主流完成
        with torch.cuda.stream(self.alt_stream):  # 在辅助流上执行
            self._device_tensors = self._create_device_tensors()  # 创建GPU上的参数字典
            self._load_event = torch.cuda.Event()  # 创建加载完成事件
            self._load_event.record()  # 记录事件

    def offload(self):  # 卸载参数（释放GPU上的引用）
        self._device_tensors = None  # 清空GPU参数字典
        self._load_event = None  # 清空加载事件

    def wait_and_get_device_tensors(self):  # 等待加载完成并返回GPU上的参数字典
        assert self._device_tensors is not None  # 断言参数已加载
        self._load_event.wait()  # 等待加载事件完成
        return self._device_tensors

    def _create_device_tensors(self):  # 创建GPU上的参数字典
        return {k: v.create_device_tensor() for k, v in self._param_offloaders.items()}


class _BaseParamOffloader(ABC):  # 参数卸载器抽象基类
    @staticmethod
    def create(mode: str, **kwargs) -> "_BaseParamOffloader":  # 根据模式创建对应的参数卸载器
        return {
            "meta": _MetaParamOffloader,  # 元设备卸载器（调试用）
            "cpu": _CpuParamOffloader,  # CPU卸载器
            "shm_cpu": _ShmCpuParamOffloader,  # 共享内存CPU卸载器
            "sharded_gpu": _ShardedGpuParamOffloader,  # 分片GPU卸载器
        }[mode](**kwargs)

    def __init__(self, module, param_name):  # 初始化，保存模块和参数名
        self._module = module
        self._param_name = param_name

    @property
    def _param(self):  # 获取参数对象
        return getattr(self._module, self._param_name)

    def post_init(self):  # 初始化后处理，默认不做任何事
        pass

    def create_device_tensor(self):  # 创建GPU上的参数张量，子类必须实现
        raise NotImplementedError


class _MetaParamOffloader(_BaseParamOffloader):  # 元设备参数卸载器，通常用于调试
    """Usually used for debugging."""

    def __init__(self, module, param_name):  # 初始化，将参数移动到元设备
        super().__init__(module, param_name)
        _move_param_to_meta(module, param_name)

    def create_device_tensor(self):  # 在GPU上创建空张量
        return torch.empty_like(self._param.data, device="cuda")


class _CpuParamOffloader(_BaseParamOffloader):  # CPU参数卸载器，将参数移动到CPU
    def __init__(self, module, param_name):  # 初始化，将参数移动到CPU（使用pin_memory）
        super().__init__(module, param_name)
        _move_param_to_cpu(self._param, pin_memory=True)

    def create_device_tensor(self):  # 将参数从CPU复制到GPU
        return self._param.to("cuda", non_blocking=True)


class _ShmCpuParamOffloader(_BaseParamOffloader):  # 共享内存CPU参数卸载器，使用跨进程共享内存
    def __init__(self, module, param_name):  # 初始化，分配共享内存并复制数据
        super().__init__(module, param_name)
        self._rank = get_naive_distributed().get_rank()  # 获取当前进程排名
        self._world_size = get_naive_distributed().get_world_size()  # 获取进程总数

        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        assert get_tensor_model_parallel_world_size() == 1, "not yet support tp_size!=1"  # 不支持张量并行
        assert (
            self._param.data.is_contiguous()
        ), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"  # 不支持非连续张量

        self.shm_cpu_data = get_host_shared_memory_manager().malloc(  # 在共享内存中分配空间
            shape=self._param.shape, dtype=self._param.dtype
        )

        if self._rank == 0:  # 如果是0号进程
            self.shm_cpu_data.copy_(self._param.data.to("cpu"))  # 将参数数据复制到共享内存
            self._param.data = self.shm_cpu_data  # 指向共享内存
        else:  # 如果不是0号进程
            _move_param_to_meta(self._module, self._param_name)  # 将参数移动到元设备
        get_naive_distributed().barrier()  # 进程同步

    def post_init(self):  # 初始化后处理，验证共享内存指针并移动参数到元设备
        if self._rank == 0:  # 如果是0号进程
            assert (
                self.shm_cpu_data.data_ptr() == self._param.data.data_ptr()
            ), f"{self.shm_cpu_data.data_ptr()=} {self._param.data.data_ptr()=} {self.shm_cpu_data=} {self._param.data=}"  # 验证数据指针一致

        _move_param_to_meta(self._module, self._param_name)  # 将参数移动到元设备

    def create_device_tensor(self):  # 将共享内存数据复制到GPU
        return self.shm_cpu_data.to("cuda", non_blocking=True)


def update_param(param, new_tensor):  # 更新参数，同时保留卸载器需要的属性（如固定主机内存）
    """Update parameter while keeping properties needed by Offloader (e.g. pinned host memory)."""

    if param.device == new_tensor.device:  # 如果设备相同
        param.data = new_tensor  # 直接替换
    else:
        assert param.device == torch.device(
            "cpu"
        ), f"{param.device=} {new_tensor.device=}"  # 参数必须在CPU上
        param.data = _create_cpu_data(new_tensor, pin_memory=True)  # 在CPU上创建固定内存数据


def _move_param_to_cpu(param, pin_memory: bool):  # 将参数移动到CPU
    param.data = _create_cpu_data(param.data, pin_memory=pin_memory)


def _create_cpu_data(data, pin_memory: bool):  # 在CPU上创建数据副本（可选固定内存）
    cpu_data = _empty_strided_like(  # 创建CPU空张量
        data,
        device="cpu",
        pin_memory=pin_memory,
    )
    cpu_data.copy_(data)  # 复制数据
    return cpu_data


def _move_param_to_meta(module, param_name):  # 将模块参数移动到元设备
    old_param = getattr(module, param_name)  # 获取旧参数
    old_param_type = type(old_param)  # 获取参数类型

    new_data = old_param.data.to("meta")  # 将数据移动到元设备

    if old_param_type == ModelWeightParameter:  # 如果是模型权重参数
        # manually checked how `w13_weight` and `w2_weight` are constructed  # 手动检查了w13_weight和w2_weight的构造方式
        new_param = ModelWeightParameter(
            data=new_data,
            **{
                k: getattr(old_param, k)
                for k in ["input_dim", "output_dim", "weight_loader"]
            },
        )
    elif old_param_type == torch.nn.Parameter:  # 如果是普通参数
        new_param = torch.nn.Parameter(
            data=new_data,
            requires_grad=False,
        )
        if hasattr(old_param, "weight_loader"):  # 如果旧参数有weight_loader
            new_param.weight_loader = old_param.weight_loader  # 复制weight_loader
        else:
            new_param.weight_loader = lambda *args, **kwargs: None  # 设置空weight_loader
    else:
        raise ValueError(f"Unknown {old_param_type=} {old_param=}")  # 未知参数类型

    setattr(module, param_name, new_param)  # 替换模块参数


def _empty_strided_like(x: torch.Tensor, device, pin_memory=False):  # 创建与输入张量相同大小和步幅的空张量
    return torch.empty_strided(
        size=x.size(),
        stride=x.stride(),
        dtype=x.dtype,
        layout=x.layout,
        device=device,
        pin_memory=pin_memory,
    )


# ----------------------------------------- ShardedGpu ------------------------------------------------------


# TODO unify with ShmCpu mode  # TODO: 与ShmCpu模式统一
class _ShardedGpuParamOffloader(_BaseParamOffloader):  # 分片GPU参数卸载器，将参数分片分布在各GPU上
    def __init__(self, module, param_name):  # 初始化，将参数移动到CPU或元设备
        super().__init__(module, param_name)
        self._rank = get_naive_distributed().get_rank()  # 获取当前进程排名
        self._world_size = get_naive_distributed().get_world_size()  # 获取进程总数

        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        assert get_tensor_model_parallel_world_size() == 1, "not yet support tp_size!=1"  # 不支持张量并行
        assert (
            self._param.data.is_contiguous()
        ), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"  # 不支持非连续张量

        if self._rank == 0:  # 如果是0号进程
            _move_param_to_cpu(self._param, pin_memory=True)  # 将参数移动到CPU
        else:  # 如果不是0号进程
            _move_param_to_meta(self._module, self._param_name)  # 将参数移动到元设备

        self.sharded_param_handles = None  # 分片参数句柄列表

    def post_init(self):  # 初始化后处理，分发参数到各GPU
        # check again since it may be changed  # 再次检查，因为参数可能已被修改
        assert (
            self._param.data.is_contiguous()
        ), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"  # 不支持非连续张量

        scatter_src = self._param.data  # 获取要分发的数据源

        logger.info(
            f"[offloader] post_init {scatter_src.nbytes=} {scatter_src.dtype=} {scatter_src.shape=} {torch.cuda.memory_allocated()=}"
        )

        if self._rank == 0:  # 如果是0号进程
            scatter_src = scatter_src.to("cuda")  # 将数据移到GPU
        scatter_list = _even_chunk(scatter_src, self._world_size)  # 将数据均匀分块

        sharded_param = torch.empty(  # 创建本地分片张量
            scatter_list[0].shape, dtype=scatter_list[0].dtype, device="cuda"
        )
        self.sharded_param_handles = _create_shared_buffer_tensors(  # 创建共享缓冲区张量
            local_tensor=sharded_param
        )

        get_naive_distributed().scatter(  # 执行分发操作
            sharded_param, scatter_list if self._rank == 0 else None
        )

        _move_param_to_meta(self._module, self._param_name)  # 将参数移动到元设备

    def create_device_tensor(self):  # 从各GPU收集分片并组装完整参数
        output = _empty_strided_like(self._param, device="cuda")  # 创建输出张量
        output_chunks = output.chunk(self._world_size)  # 分块

        for index in range(self._world_size):  # 遍历每个进程
            src_rank = (self._rank + index) % self._world_size  # 计算源排名
            src_buf = self.sharded_param_handles[src_rank]  # 获取源缓冲区
            output_chunks[src_rank].copy_(src_buf)  # 复制数据

        return output


def _even_chunk(x: torch.Tensor, chunks: int):  # 将张量均匀分块
    assert x.shape[0] % chunks == 0, f"{x.shape=} {chunks=}"  # 断言可以整除
    return list(x.chunk(chunks))


def _create_shared_buffer_tensors(local_tensor: torch.Tensor) -> List[torch.Tensor]:  # 创建跨进程共享的缓冲区张量列表
    self_rank = get_naive_distributed().get_rank()  # 获取当前排名
    world_size = get_naive_distributed().get_world_size()  # 获取进程总数

    object_list = get_naive_distributed().all_gather_object(  # 全局收集序列化张量对象
        dict(
            dup_serialized_local_tensor=[
                (
                    None
                    if interesting_rank == self_rank
                    else MultiprocessingSerializer.serialize(local_tensor)
                )
                for interesting_rank in range(world_size)
            ]
        )
    )

    output_tensors = []  # 输出张量列表
    for output_rank in range(world_size):  # 遍历每个进程
        remote_serialized_tensor = object_list[output_rank][  # 获取远程序列化张量
            "dup_serialized_local_tensor"
        ][self_rank]
        if output_rank == self_rank:  # 如果是自身
            assert remote_serialized_tensor is None  # 自身不需要序列化
            output_tensors.append(local_tensor)  # 使用本地张量
        else:
            output_tensors.append(  # 反序列化远程张量
                MultiprocessingSerializer.deserialize(remote_serialized_tensor)
            )

    return output_tensors  # 返回共享缓冲区张量列表
