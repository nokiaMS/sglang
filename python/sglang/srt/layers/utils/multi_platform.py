# 多平台算子分发模块
# 提供跨平台（CUDA、HIP、CPU、NPU、XPU、MUSA、HPU）的算子前向分发机制，
# 支持OOT(Out-of-Tree)平台注册和torch.compile模式切换。

from typing import Callable, ClassVar  # 导入类型提示

from torch import nn  # 导入PyTorch神经网络模块

from sglang.kernel_api_logging import debug_kernel_api  # 导入内核API调试装饰器
from sglang.srt.platforms import current_platform  # 导入当前平台实例
from sglang.srt.utils import (  # 导入平台检测工具函数
    cpu_has_amx_support,  # CPU AMX支持检测
    is_cpu,  # CPU平台检测
    is_cuda,  # CUDA平台检测
    is_hip,  # HIP平台检测
    is_musa,  # MUSA平台检测
    is_npu,  # NPU平台检测
    is_xpu,  # XPU平台检测
)

_is_cuda = is_cuda()  # 是否为CUDA平台
_is_hip = is_hip()  # 是否为HIP平台
_is_cpu = is_cpu()  # 是否为CPU平台
_is_cpu_amx_available = cpu_has_amx_support()  # CPU AMX指令是否可用
_is_npu = is_npu()  # 是否为NPU平台
_is_xpu = is_xpu()  # 是否为XPU平台
_is_musa = is_musa()  # 是否为MUSA平台


class MultiPlatformOp(nn.Module):  # 多平台算子基类，根据运行平台自动分发前向方法

    # OOT forward registry: maps dispatch_key -> {op_cls -> forward_fn}
    # OOT前向注册表：映射 dispatch_key -> {op_cls -> forward_fn}
    _oot_forward_registry: ClassVar[dict[str, dict[type, Callable]]] = {}  # 类变量，存储OOT平台的前向函数注册

    @classmethod
    def register_oot_forward(cls, op_cls: type, fn: Callable, platform_key: str):  # 注册OOT平台的前向实现
        """Register an OOT forward implementation for a specific op class and platform."""
        # 为特定算子类和平台注册OOT前向实现。
        cls._oot_forward_registry.setdefault(platform_key, {})[op_cls] = fn  # 按平台键和算子类注册前向函数

    def __init__(self):  # 初始化多平台算子
        super().__init__()  # 调用父类初始化
        self._forward_method: Callable = self.dispatch_forward()  # 根据平台分发并缓存前向方法

        # States for torch.compile
        # torch.compile状态
        self._original_forward_method = None  # 保存编译前的原始前向方法
        self.is_torch_compile = False  # 是否处于torch.compile模式

    def enter_torch_compile(self, num_tokens: int):  # 进入torch.compile模式
        # Skip if Op is already entered compile mode.
        # NOTE(alcanderian): Some Ops(for example RotaryEmbedding) will be reused
        # among layers and `enter_torch_compile` will be called many times.
        # We should prevent `self._original_forward_method` from being overridden when
        # it is not the first time `enter_torch_compile` called.
        # 如果算子已进入编译模式则跳过。
        # NOTE(alcanderian): 某些算子（例如RotaryEmbedding）会在层之间复用，
        # `enter_torch_compile` 会被多次调用。
        # 我们应防止 `self._original_forward_method` 在非首次调用时被覆盖。
        if self.is_torch_compile:  # 如果已处于编译模式
            return  # 直接返回

        self._original_forward_method = self._forward_method  # 保存当前前向方法
        # NOTE: Temporarily workaround MoE
        # The performance of torch.compile on this layer is not always good when bs > 1,
        # so we decide to only use torch.compile when bs=1
        # NOTE: MoE的临时解决方案
        # torch.compile在该层上当bs > 1时性能不一定好，
        # 因此我们决定仅在bs=1时使用torch.compile
        if "FusedMoE" in self.__class__.__name__:  # 如果是FusedMoE算子
            if num_tokens == 1:  # 仅当token数为1时使用原生实现
                from sglang.srt.layers.moe.fused_moe_native import (  # 导入原生MoE前向函数
                    fused_moe_forward_native,
                )

                self._forward_method = fused_moe_forward_native  # 替换为原生前向方法
        elif "TopK" in self.__class__.__name__:  # 如果是TopK算子
            if num_tokens == 1:  # 仅当token数为1时使用原生实现
                self._forward_method = self.forward_native  # 替换为原生前向方法
        else:  # 其他算子
            self._forward_method = self.forward_native  # 使用原生前向方法
        self.is_torch_compile = True  # 标记为编译模式

    def leave_torch_compile(self):  # 退出torch.compile模式
        # Skip if Op is already exited compile mode.
        # 如果算子已退出编译模式则跳过。
        if not self.is_torch_compile:  # 如果未处于编译模式
            return  # 直接返回

        self._forward_method = self._original_forward_method  # 恢复原始前向方法
        self._original_forward_method = None  # 清空保存的方法
        self.is_torch_compile = False  # 标记为非编译模式

    # Please do not override this method, because `self._forward_method` can change when in torch compile mode
    # 请勿覆盖此方法，因为 `self._forward_method` 在torch.compile模式下会改变
    @debug_kernel_api  # 内核API调试装饰器
    def forward(self, *args, **kwargs):  # 前向调用入口，委托给分发的前向方法
        return self._forward_method(*args, **kwargs)  # 调用当前分发的前向方法

    def forward_native(self, *args, **kwargs):  # 原生前向方法（基类默认未实现）
        raise NotImplementedError  # 子类需实现

    def forward_cuda(self, *args, **kwargs):  # CUDA前向方法（基类默认未实现）
        raise NotImplementedError  # 子类需实现

    def forward_npu(self, *args, **kwargs):  # NPU前向方法，默认回退到原生实现
        return self.forward_native(*args, **kwargs)  # 调用原生前向方法

    def forward_hip(self, *args, **kwargs):  # HIP前向方法，默认回退到CUDA实现
        return self.forward_cuda(*args, **kwargs)  # 调用CUDA前向方法

    def forward_xpu(self, *args, **kwargs):  # XPU前向方法，默认回退到原生实现
        return self.forward_native(*args, **kwargs)  # 调用原生前向方法

    def forward_musa(self, *args, **kwargs):  # MUSA前向方法，默认回退到CUDA实现
        return self.forward_cuda(*args, **kwargs)  # 调用CUDA前向方法

    def forward_hpu(self, *args, **kwargs):  # HPU前向方法，默认回退到原生实现
        return self.forward_native(*args, **kwargs)  # 调用原生前向方法

    def forward_cpu(self, *args, **kwargs):  # CPU前向方法，默认回退到原生实现
        return self.forward_native(*args, **kwargs)  # 调用原生前向方法

    def dispatch_forward(self):  # 根据当前平台分发前向方法
        # OOT platform dispatch: check registry then method lookup
        # OOT平台分发：先检查注册表，再查找方法
        if current_platform.is_out_of_tree():  # 如果是OOT平台
            key = current_platform.get_dispatch_key_name()  # 获取分发键名
            oot = self._oot_forward_registry.get(key, {})  # 查找OOT注册表
            if type(self) in oot:  # 如果当前算子类在OOT注册表中
                return oot[type(self)].__get__(self)  # 返回绑定的OOT前向方法
            method = getattr(self, f"forward_{key}", None)  # 查找平台特定前向方法
            if method is not None:  # 如果找到
                return method  # 返回该方法
            return self.forward_native  # 否则回退到原生实现

        if _is_cuda:  # CUDA平台
            return self.forward_cuda  # 返回CUDA前向方法
        elif _is_hip:  # HIP平台
            return self.forward_hip  # 返回HIP前向方法
        elif _is_cpu and _is_cpu_amx_available:  # CPU平台且AMX可用
            return self.forward_cpu  # 返回CPU前向方法
        elif _is_npu:  # NPU平台
            return self.forward_npu  # 返回NPU前向方法
        elif _is_xpu:  # XPU平台
            return self.forward_xpu  # 返回XPU前向方法
        elif _is_musa:  # MUSA平台
            return self.forward_musa  # 返回MUSA前向方法
        else:  # 其他平台
            return self.forward_native  # 返回原生前向方法
