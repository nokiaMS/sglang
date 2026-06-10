# PyTorch内存节省器适配器，提供GPU显存暂停/恢复功能的统一接口
# 支持真实模式（使用torch-memory-saver库）和空操作模式（不进行任何内存节省操作）
import logging  # 导入日志记录模块
from abc import ABC  # 导入抽象基类
from contextlib import contextmanager  # 导入上下文管理器工具

try:  # 尝试导入torch_memory_saver
    import torch_memory_saver  # 导入PyTorch内存节省器

    _memory_saver = torch_memory_saver.torch_memory_saver  # 获取内存节省器实例
    import_error = None  # 导入成功，无错误
except ImportError as e:  # 如果导入失败
    import_error = e  # 保存导入错误
    pass  # 继续

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class TorchMemorySaverAdapter(ABC):  # PyTorch内存节省器适配器抽象基类
    @staticmethod
    def create(enable: bool):  # 工厂方法：根据启用标志创建适配器实例
        if enable and import_error is not None:  # 如果启用但导入失败
            logger.warning(  # 记录警告
                "enable_memory_saver is enabled, but "
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )
            raise import_error  # 抛出导入错误
        return (  # 根据启用标志返回对应实例
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()  # 启用则返回真实适配器，否则返回空操作适配器
        )

    def check_validity(self, caller_name):  # 检查内存节省器是否有效启用
        if not self.enabled:  # 如果未启用
            logger.warning(  # 记录警告
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )

    def configure_subprocess(self):  # 配置子进程（抽象方法）
        raise NotImplementedError  # 子类必须实现

    def region(self, tag: str, enable_cpu_backup: bool = False):  # 创建内存节省区域（抽象方法）
        raise NotImplementedError  # 子类必须实现

    def cuda_graph(self, **kwargs):  # CUDA图内存节省（抽象方法）
        raise NotImplementedError  # 子类必须实现

    def disable(self):  # 禁用内存节省（抽象方法）
        raise NotImplementedError  # 子类必须实现

    def pause(self, tag: str):  # 暂停内存区域（抽象方法）
        raise NotImplementedError  # 子类必须实现

    def resume(self, tag: str):  # 恢复内存区域（抽象方法）
        raise NotImplementedError  # 子类必须实现

    @property
    def enabled(self):  # 是否启用内存节省（抽象属性）
        raise NotImplementedError  # 子类必须实现


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):  # 真实内存节省器适配器实现
    """Adapter for TorchMemorySaver with tag-based control"""  # 基于标签控制的TorchMemorySaver适配器

    def configure_subprocess(self):  # 配置子进程的内存节省
        return torch_memory_saver.configure_subprocess()  # 委托给torch_memory_saver

    def region(self, tag: str, enable_cpu_backup: bool = False):  # 创建标签化的内存节省区域
        return _memory_saver.region(tag=tag, enable_cpu_backup=enable_cpu_backup)  # 委托给内存节省器

    def cuda_graph(self, **kwargs):  # 创建CUDA图内存节省
        return _memory_saver.cuda_graph(**kwargs)  # 委托给内存节省器

    def disable(self):  # 禁用内存节省
        return _memory_saver.disable()  # 委托给内存节省器

    def pause(self, tag: str):  # 暂停指定标签的内存区域
        return _memory_saver.pause(tag=tag)  # 委托给内存节省器

    def resume(self, tag: str):  # 恢复指定标签的内存区域
        return _memory_saver.resume(tag=tag)  # 委托给内存节省器

    @property
    def enabled(self):  # 检查内存节省器是否启用
        return _memory_saver is not None and _memory_saver.enabled  # 检查实例存在且已启用


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):  # 空操作内存节省器适配器（不执行任何操作）
    @contextmanager
    def configure_subprocess(self):  # 空操作的子进程配置
        yield  # 不执行任何操作

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):  # 空操作的内存区域
        yield  # 不执行任何操作

    @contextmanager
    def cuda_graph(self, **kwargs):  # 空操作的CUDA图
        yield  # 不执行任何操作

    @contextmanager
    def disable(self):  # 空操作的禁用
        yield  # 不执行任何操作

    def pause(self, tag: str):  # 空操作的暂停
        pass  # 不执行任何操作

    def resume(self, tag: str):  # 空操作的恢复
        pass  # 不执行任何操作

    @property
    def enabled(self):  # 空操作始终返回未启用
        return False  # 返回False
