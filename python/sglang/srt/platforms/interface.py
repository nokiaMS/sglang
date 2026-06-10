# SGLang SRT硬件平台抽象接口模块
# 本模块定义SRTPlatform基类，继承DeviceMixin获取共享设备操作，
# 并添加SRT特定的子系统工厂方法、能力标志和配置生命周期钩子。
# 外部平台通过setuptools entry_points在"sglang.srt.platforms"组下注册。
"""
SGLang SRT Hardware Platform Abstraction.

Defines SRTPlatform — the base class for SRT (LLM inference) platform
backends.  SRTPlatform inherits DeviceMixin for shared device operations
and adds SRT-specific subsystem factory methods, capability flags, and
configuration lifecycle hooks.

Out-of-tree platforms register via setuptools entry_points under the
"sglang.srt.platforms" group and should subclass SRTPlatform.
"""

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum  # 导入设备混入类和平台枚举

# Re-export for convenience  # 为方便使用而重新导出
__all__ = ["SRTPlatform", "PlatformEnum"]  # 公开导出列表


class SRTPlatform(DeviceMixin):
    """
    Base class for SRT hardware platform backends.  # SRT硬件平台后端的基类

    Inherits device identity queries and operations from DeviceMixin.  # 继承DeviceMixin的设备身份查询和操作
    Adds SRT-specific factory methods, capability flags, and lifecycle hooks.  # 添加SRT特定的工厂方法、能力标志和生命周期钩子

    OOT platforms should subclass SRTPlatform and override the methods
    relevant to their hardware.  # OOT平台应子类化SRTPlatform并覆盖与其硬件相关的方法
    """

    # SRT-specific class-level attribute  # SRT特定的类级属性
    supported_quantization: list[str] = []  # 支持的量化方法列表

    # ------------------------------------------------------------------
    # Configuration lifecycle  # 配置生命周期
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:  # 应用服务器参数默认值
        """Apply platform-specific default values to server arguments.  # 应用平台特定的服务器参数默认值

        Called after ServerArgs is parsed.  # 在ServerArgs解析后调用
        """
        pass  # 默认空操作

    # ------------------------------------------------------------------
    # Subsystem factory methods  # 子系统工厂方法
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:  # 获取默认注意力后端
        """Return the default attention backend name for this platform."""  # 返回此平台的默认注意力后端名称
        raise NotImplementedError  # 子类必须实现

    def get_graph_runner_cls(self) -> type:  # 获取图运行器类
        """Return the graph runner class for this platform."""  # 返回此平台的图运行器类
        raise NotImplementedError  # 子类必须实现

    def get_mha_kv_pool_cls(self) -> type:  # 获取MHA KV池类
        """Return the MHA KV pool class for this platform."""  # 返回此平台的MHA KV池类
        raise NotImplementedError  # 子类必须实现

    def get_mla_kv_pool_cls(self) -> type:  # 获取MLA KV池类
        """Return the MLA KV pool class for this platform."""  # 返回此平台的MLA KV池类
        raise NotImplementedError  # 子类必须实现

    def get_dsa_kv_pool_cls(self) -> type:  # 获取DSA KV池类
        """Return the DSA KV pool class for this platform (DeepSeek V3.2)."""  # 返回此平台的DSA KV池类（DeepSeek V3.2）
        raise NotImplementedError  # 子类必须实现

    def get_paged_allocator_cls(self) -> type:  # 获取分页分配器类
        """Return the paged allocator class for this platform."""  # 返回此平台的分页分配器类
        raise NotImplementedError  # 子类必须实现

    def get_compile_backend(self, mode: str | None = None) -> str:  # 获取编译后端标识
        """Return the compilation backend identifier.  # 返回编译后端标识符

        ``mode`` is an optional hint for the platform (e.g. "npugraph_ex").  # mode是平台的可选提示
        """
        return "inductor"  # 默认使用inductor后端

    def get_piecewise_backend_cls(self) -> type:  # 获取分段编译后端类
        """Return the piecewise compilation backend class for this platform."""  # 返回此平台的分段编译后端类
        raise NotImplementedError  # 子类必须实现

    # ------------------------------------------------------------------
    # Capability flags (safe conservative defaults)  # 能力标志（安全保守的默认值）
    # ------------------------------------------------------------------

    def supports_fp8(self) -> bool:  # 是否支持FP8量化
        """Whether this platform supports FP8 quantization."""  # 此平台是否支持FP8量化
        return False  # 默认不支持

    def is_pin_memory_available(self) -> bool:  # 是否支持锁页内存
        """Whether pinned memory is available on this platform."""  # 此平台是否支持锁页内存
        return True  # 默认支持

    def support_cuda_graph(self) -> bool:  # 是否支持CUDA图
        """Whether this platform supports device graph capture and replay.  # 此平台是否支持设备图捕获和重放
        Controls CUDA graph (CudaGraphRunner) for the decode path.  # 控制解码路径的CUDA图
        OOT platforms that support graph-style capture should return True.  # 支持图式捕获的OOT平台应返回True
        """
        return False  # 默认不支持

    def support_piecewise_cuda_graph(self) -> bool:  # 是否支持分段CUDA图
        """Whether this platform supports piecewise CUDA graph.  # 此平台是否支持分段CUDA图

        Controls PiecewiseCudaGraphRunner for the prefill/extend path  # 控制预填充/扩展路径的分段CUDA图运行器
        (torch.compile backend).  # torch.compile后端
        """
        return False  # 默认不支持

    # ------------------------------------------------------------------
    # Initialization  # 初始化
    # ------------------------------------------------------------------

    def init_backend(self) -> None:  # 初始化后端
        """One-time backend initialization.  Called in each worker."""  # 一次性后端初始化，在每个worker中调用
        pass  # 默认空操作

    # ------------------------------------------------------------------
    # MultiPlatformOp integration  # MultiPlatformOp集成
    # ------------------------------------------------------------------

    def get_dispatch_key_name(self) -> str:  # 获取分派键名
        """Return the dispatch key name for MultiPlatformOp.  # 返回MultiPlatformOp的分派键名

        Determines which ``forward_<key>()`` method is selected.  # 决定选择哪个forward_<key>()方法
        E.g. "cuda", "npu", "hip", "xpu", "cpu".  # 例如"cuda"、"npu"、"hip"、"xpu"、"cpu"
        """
        return "native"  # 默认使用native
