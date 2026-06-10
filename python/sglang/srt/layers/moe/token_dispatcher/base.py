# MoE token分发器基础模块
# 定义了分发器的基类、钩子机制、分发输出和合并输入的协议与格式
from __future__ import annotations  # 启用延迟注解求值，支持前向引用

import weakref  # 弱引用模块，用于避免循环引用
from abc import ABC, abstractmethod  # 抽象基类和抽象方法装饰器
from enum import Enum  # 枚举类型基类
from typing import (  # 类型注解工具
    TYPE_CHECKING,  # 类型检查标志
    Any,  # 任意类型
    Callable,  # 可调用类型
    Optional,  # 可选类型
    OrderedDict,  # 有序字典类型
    Protocol,  # 协议类型
    Tuple,  # 元组类型
    TypeGuard,  # 类型守卫
    Union,  # 联合类型
    runtime_checkable,  # 运行时可检查装饰器
)

import torch  # PyTorch深度学习框架

if TYPE_CHECKING:  # 仅在类型检查时导入，运行时不导入
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs  # 合并重叠参数
    from sglang.srt.layers.moe.token_dispatcher import (  # 分发器相关类型
        DeepEPLLCombineInput,  # DeepEP低延迟合并输入
        DeepEPLLDispatchOutput,  # DeepEP低延迟分发输出
        DeepEPNormalCombineInput,  # DeepEP普通模式合并输入
        DeepEPNormalDispatchOutput,  # DeepEP普通模式分发输出
        FlashinferCombineInput,  # FlashInfer合并输入
        FlashinferDispatchOutput,  # FlashInfer分发输出
        StandardCombineInput,  # 标准合并输入
        StandardDispatchOutput,  # 标准分发输出
    )
    from sglang.srt.layers.moe.topk import TopKOutput  # TopK输出类型


# ------------------------------ Dispatcher Hook -------------------------------------
# 分发器钩子机制，允许在分发和合并操作前后插入自定义逻辑


class _RemovableDispatcherHandle:
    # 可移除的分发器钩子句柄，用于注册后可安全移除钩子

    next_id = 0  # Global counter for unique IDs # 全局计数器，用于生成唯一ID

    def __init__(self, hooks_dict: OrderedDict):  # 初始化钩子句柄
        self.id = _RemovableDispatcherHandle.next_id  # 分配唯一ID
        _RemovableDispatcherHandle.next_id += 1  # 递增全局计数器
        self.weak_hooks_dict = weakref.ref(hooks_dict)  # 对钩子字典的弱引用

    def remove(self):  # 从钩子字典中移除自身
        hooks_dict = self.weak_hooks_dict()  # 获取弱引用对象
        if hooks_dict is not None and self.id in hooks_dict:  # 若字典仍存在且自身ID在字典中
            del hooks_dict[self.id]  # 从字典中删除该钩子


class DispatcherBaseHooks:
    # 分发器钩子基类，管理钩子函数的注册和调用

    def __init__(self):  # 初始化钩子字典
        self.hook_dict = OrderedDict[int, Callable]()  # 有序字典存储钩子函数

    def register_hook(self, hook_fun: Callable) -> _RemovableDispatcherHandle:  # 注册一个钩子函数
        handle = _RemovableDispatcherHandle(self.hook_dict)  # 创建可移除句柄
        self.hook_dict[handle.id] = hook_fun  # 将钩子函数存入字典
        return handle  # 返回句柄，便于后续移除

    def __call__(self, *args, **kwargs) -> Optional[Any]:  # 调用钩子（子类需重写）
        raise NotImplementedError("This method should be overridden by subclasses")  # 子类必须重写此方法


class _PreDispatchHooks(DispatcherBaseHooks):
    # 分发前钩子，在dispatch操作执行前修改输入

    def __call__(  # 调用所有预分发钩子
        self,
        dispatcher: BaseDispatcher,  # 分发器实例
        hidden_states: torch.Tensor,  # 隐藏状态张量
        topk_output: TopKOutput,  # TopK输出
    ) -> Optional[Tuple[torch.Tensor, TopKOutput]]:  # 返回修改后的输入
        for hook_fun in self.hook_dict.values():  # 遍历所有钩子函数
            hook_output = hook_fun(dispatcher, hidden_states, topk_output)  # 执行钩子
            if hook_output is not None:  # 若钩子返回非None值
                hidden_states, topk_output = hook_output  # 用钩子输出更新输入
        return hidden_states, topk_output  # 返回最终的输入


class _PostDispatchHooks(DispatcherBaseHooks):
    # 分发后钩子，在dispatch操作执行后修改输出

    def __call__(  # 调用所有后分发钩子
        self, dispatcher: BaseDispatcher, dispatch_output: DispatchOutput  # 分发器实例和分发输出
    ) -> Optional[DispatchOutput]:  # 返回修改后的分发输出
        for hook_fun in self.hook_dict.values():  # 遍历所有钩子函数
            hook_output = hook_fun(dispatcher, dispatch_output)  # 执行钩子
            if hook_output is not None:  # 若钩子返回非None值
                dispatch_output = hook_output  # 用钩子输出更新分发输出
        return dispatch_output  # 返回最终的分发输出


class _PreCombineHooks(DispatcherBaseHooks):
    # 合并前钩子，在combine操作执行前修改输入

    def __call__(  # 调用所有预合并钩子
        self, dispatcher: BaseDispatcher, combine_input: CombineInput  # 分发器实例和合并输入
    ) -> Optional[CombineInput]:  # 返回修改后的合并输入
        for hook_fun in self.hook_dict.values():  # 遍历所有钩子函数
            hook_output = hook_fun(dispatcher, combine_input)  # 执行钩子
            if hook_output is not None:  # 若钩子返回非None值
                combine_input = hook_output  # 用钩子输出更新合并输入
        return combine_input  # 返回最终的合并输入


class _PostCombineHooks(DispatcherBaseHooks):
    # 合并后钩子，在combine操作执行后修改输出

    def __call__(  # 调用所有后合并钩子
        self, dispatcher: BaseDispatcher, hidden_states: torch.Tensor  # 分发器实例和隐藏状态
    ) -> Optional[torch.Tensor]:  # 返回修改后的隐藏状态
        for hook_fun in self.hook_dict.values():  # 遍历所有钩子函数
            hook_output = hook_fun(dispatcher, hidden_states)  # 执行钩子
            if hook_output is not None:  # 若钩子返回非None值
                hidden_states = hook_output  # 用钩子输出更新隐藏状态
        return hidden_states  # 返回最终的隐藏状态


# ------------------------------ Dispatch Output -------------------------------------
# 分发输出定义，包含格式检查器、格式枚举和输出协议


class DispatchOutputChecker:
    # 分发输出格式检查器，提供静态方法判断输出格式类型

    @staticmethod
    def format_is_standard(  # 判断是否为标准格式
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[StandardDispatchOutput]:
        return dispatch_output.format.is_standard()  # 检查格式是否为标准格式

    @staticmethod
    def format_is_triton_kernels(  # 判断是否为triton_kernels格式
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[StandardDispatchOutput]:
        return dispatch_output.format.is_standard()  # triton_kernels格式也使用标准格式检查

    @staticmethod
    def format_is_deepep_normal(  # 判断是否为DeepEP普通模式格式
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPNormalDispatchOutput]:
        return dispatch_output.format.is_deepep_normal()  # 检查格式是否为DeepEP普通模式

    @staticmethod
    def format_is_deepep_ll(  # 判断是否为DeepEP低延迟格式
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPLLDispatchOutput]:
        return dispatch_output.format.is_deepep_ll()  # 检查格式是否为DeepEP低延迟模式

    @staticmethod
    def format_is_deepep(  # 判断是否为DeepEP格式（普通或低延迟）
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput]]:
        return dispatch_output.format.is_deepep()  # 检查格式是否为DeepEP（任一模式）

    @staticmethod
    def format_is_flashinfer(  # 判断是否为FlashInfer格式
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[FlashinferDispatchOutput]:
        return dispatch_output.format.is_flashinfer()  # 检查格式是否为FlashInfer


class DispatchOutputFormat(Enum):
    # 分发输出格式枚举，定义所有支持的输出格式

    STANDARD = "standard"  # 标准格式
    DEEPEP_NORMAL = "deepep_normal"  # DeepEP普通模式格式
    DEEPEP_LL = "deepep_ll"  # DeepEP低延迟模式格式
    FLASHINFER = "flashinfer"  # FlashInfer格式

    def is_standard(self) -> bool:  # 判断是否为标准格式
        return self == DispatchOutputFormat.STANDARD  # 比较当前值与STANDARD

    def is_deepep_normal(self) -> bool:  # 判断是否为DeepEP普通模式格式
        return self == DispatchOutputFormat.DEEPEP_NORMAL  # 比较当前值与DEEPEP_NORMAL

    def is_deepep_ll(self) -> bool:  # 判断是否为DeepEP低延迟模式格式
        return self == DispatchOutputFormat.DEEPEP_LL  # 比较当前值与DEEPEP_LL

    def is_deepep(self) -> bool:  # 判断是否为DeepEP格式（普通或低延迟）
        return self in [  # 检查当前值是否在DeepEP格式列表中
            DispatchOutputFormat.DEEPEP_NORMAL,  # DeepEP普通模式
            DispatchOutputFormat.DEEPEP_LL,  # DeepEP低延迟模式
        ]

    def is_flashinfer(self) -> bool:  # 判断是否为FlashInfer格式
        return self == DispatchOutputFormat.FLASHINFER  # 比较当前值与FLASHINFER


@runtime_checkable
class DispatchOutput(Protocol):
    """Protocol for dispatch outputs in different formats."""  # 不同格式分发输出的协议定义
    # 不同格式分发输出的协议

    hidden_states: torch.Tensor  # 隐藏状态张量

    @property
    def format(self) -> DispatchOutputFormat: ...  # 输出格式属性


# ------------------------------ Combine Input -------------------------------------
# 合并输入定义，包含格式检查器、格式枚举和输入协议


class CombineInputChecker:
    # 合并输入格式检查器，提供静态方法判断输入格式类型
    @staticmethod
    def format_is_standard(  # 判断是否为标准格式
        combine_input: CombineInput,
    ) -> TypeGuard[StandardCombineInput]:
        return combine_input.format == CombineInputFormat.STANDARD  # 检查格式是否为标准格式

    @staticmethod
    def format_is_deepep_normal(  # 判断是否为DeepEP普通模式格式
        combine_input: CombineInput,
    ) -> TypeGuard[DeepEPNormalCombineInput]:
        return combine_input.format == CombineInputFormat.DEEPEP_NORMAL  # 检查格式是否为DeepEP普通模式

    @staticmethod
    def format_is_deepep_ll(  # 判断是否为DeepEP低延迟格式
        combine_input: CombineInput,
    ) -> TypeGuard[DeepEPLLCombineInput]:
        return combine_input.format == CombineInputFormat.DEEPEP_LL  # 检查格式是否为DeepEP低延迟模式

    @staticmethod
    def format_is_deepep(  # 判断是否为DeepEP格式（普通或低延迟）
        combine_input: CombineInput,
    ) -> TypeGuard[Union[DeepEPNormalCombineInput, DeepEPLLCombineInput]]:
        return combine_input.format in [  # 检查格式是否在DeepEP格式列表中
            CombineInputFormat.DEEPEP_NORMAL,  # DeepEP普通模式
            CombineInputFormat.DEEPEP_LL,  # DeepEP低延迟模式
        ]

    @staticmethod
    def format_is_flashinfer(  # 判断是否为FlashInfer格式
        combine_input: CombineInput,
    ) -> TypeGuard[FlashinferCombineInput]:
        return combine_input.format == CombineInputFormat.FLASHINFER  # 检查格式是否为FlashInfer


class CombineInputFormat(Enum):
    # 合并输入格式枚举，定义所有支持的输入格式
    STANDARD = "standard"  # 标准格式
    DEEPEP_NORMAL = "deepep_normal"  # DeepEP普通模式格式
    DEEPEP_LL = "deepep_ll"  # DeepEP低延迟模式格式
    FLASHINFER = "flashinfer"  # FlashInfer格式


@runtime_checkable
class CombineInput(Protocol):
    """Protocol for combine inputs in different formats."""  # 不同格式合并输入的协议定义
    # 不同格式合并输入的协议

    # TODO: add hidden_states to the protocol # TODO: 向协议中添加hidden_states
    # TODO: 向协议中添加hidden_states

    @property
    def format(self) -> CombineInputFormat: ...  # 输入格式属性


# ------------------------------ Base Dispatcher -------------------------------------
# 分发器基类，定义分发和合并的抽象接口及钩子管理


class BaseDispatcherConfig(ABC):
    """Base class for dispatcher configs."""  # 分发器配置基类
    # 分发器配置基类

    pass  # 占位，子类需扩展


class BaseDispatcher(ABC):
    """Base class for dispatchers."""  # 分发器基类
    # 分发器基类

    def __init__(self):  # 初始化分发器
        self.quant_config: dict = {}  # 量化配置字典

        # Overlap args # 重叠参数
        self.overlap_args: Optional[CombineOverlapArgs] = None  # 合并重叠参数
        self.meta_overlap_args: Optional[dict] = None  # 元数据重叠参数

        # Hooks # 钩子
        self._pre_dispatch_hooks: Optional[_PreDispatchHooks] = None  # 预分发钩子
        self._post_dispatch_hooks: Optional[_PostDispatchHooks] = None  # 后分发钩子
        self._pre_combine_hooks: Optional[_PreCombineHooks] = None  # 预合并钩子
        self._post_combine_hooks: Optional[_PostCombineHooks] = None  # 后合并钩子
        self._original_dispatch_func: Optional[Callable] = None  # 原始分发函数
        self._original_combine_func: Optional[Callable] = None  # 原始合并函数

    @abstractmethod
    def dispatch(  # 分发方法（抽象，子类必须实现）
        self, hidden_states: torch.Tensor, topk_output: TopKOutput  # 隐藏状态和TopK输出
    ) -> DispatchOutput:
        pass  # 子类实现具体分发逻辑

    def _dispatch_with_hook(  # 带钩子的分发方法
        self, hidden_states: torch.Tensor, topk_output: TopKOutput  # 隐藏状态和TopK输出
    ) -> DispatchOutput:
        if self._pre_dispatch_hooks is not None:  # 若存在预分发钩子
            hidden_states, topk_output = self._pre_dispatch_hooks(  # 执行预分发钩子
                self, hidden_states, topk_output
            )
        dispatch_output = self._original_dispatch_func(  # 调用原始分发函数
            hidden_states=hidden_states, topk_output=topk_output
        )
        if self._post_dispatch_hooks is not None:  # 若存在后分发钩子
            dispatch_output = self._post_dispatch_hooks(self, dispatch_output)  # 执行后分发钩子
        return dispatch_output  # 返回分发输出

    def _override_dispatch_func(self) -> None:  # 用带钩子版本替换原始分发函数
        if self._original_dispatch_func is None:  # 若尚未替换
            self._original_dispatch_func = self.dispatch  # 保存原始分发函数
            self.dispatch = self._dispatch_with_hook  # 替换为带钩子版本

    @abstractmethod
    def combine(self, combine_input: CombineInput) -> torch.Tensor:  # 合并方法（抽象，子类必须实现）
        pass  # 子类实现具体合并逻辑

    def _combine_with_hook(self, combine_input: CombineInput) -> torch.Tensor:  # 带钩子的合并方法
        if self._pre_combine_hooks is not None:  # 若存在预合并钩子
            combine_input = self._pre_combine_hooks(self, combine_input)  # 执行预合并钩子
        hidden_states = self._original_combine_func(combine_input=combine_input)  # 调用原始合并函数
        if self._post_combine_hooks is not None:  # 若存在后合并钩子
            hidden_states = self._post_combine_hooks(self, hidden_states)  # 执行后合并钩子
        return hidden_states  # 返回合并后的隐藏状态

    def _override_combine_func(self) -> None:  # 用带钩子版本替换原始合并函数
        if self._original_combine_func is None:  # 若尚未替换
            self._original_combine_func = self.combine  # 保存原始合并函数
            self.combine = self._combine_with_hook  # 替换为带钩子版本

    def register_pre_dispatch_hook(  # 注册预分发钩子
        self,
        hook: Callable[  # 钩子函数类型
            [BaseDispatcher, torch.Tensor, TopKOutput],
            Optional[Tuple[torch.Tensor, TopKOutput]],
        ],
    ) -> _RemovableDispatcherHandle:
        if self._pre_dispatch_hooks is None:  # 若预分发钩子尚未创建
            self._pre_dispatch_hooks = _PreDispatchHooks()  # 创建预分发钩子实例
            self._override_dispatch_func()  # 替换分发函数
        handle = self._pre_dispatch_hooks.register_hook(hook)  # 注册钩子
        return handle  # 返回可移除句柄

    def register_post_dispatch_hook(  # 注册后分发钩子
        self, hook: Callable[[BaseDispatcher, DispatchOutput], Optional[DispatchOutput]]
    ) -> _RemovableDispatcherHandle:
        if self._post_dispatch_hooks is None:  # 若后分发钩子尚未创建
            self._post_dispatch_hooks = _PostDispatchHooks()  # 创建后分发钩子实例
            self._override_dispatch_func()  # 替换分发函数
        handle = self._post_dispatch_hooks.register_hook(hook)  # 注册钩子
        return handle  # 返回可移除句柄

    def register_pre_combine_hook(  # 注册预合并钩子
        self, hook: Callable[[BaseDispatcher, CombineInput], Optional[CombineInput]]
    ) -> _RemovableDispatcherHandle:
        if self._pre_combine_hooks is None:  # 若预合并钩子尚未创建
            self._pre_combine_hooks = _PreCombineHooks()  # 创建预合并钩子实例
            self._override_combine_func()  # 替换合并函数
        handle = self._pre_combine_hooks.register_hook(hook)  # 注册钩子
        return handle  # 返回可移除句柄

    def register_post_combine_hook(  # 注册后合并钩子
        self, hook: Callable[[BaseDispatcher, torch.Tensor], Optional[torch.Tensor]]
    ) -> _RemovableDispatcherHandle:
        if self._post_combine_hooks is None:  # 若后合并钩子尚未创建
            self._post_combine_hooks = _PostCombineHooks()  # 创建后合并钩子实例
            self._override_combine_func()  # 替换合并函数
        handle = self._post_combine_hooks.register_hook(hook)  # 注册钩子
        return handle  # 返回可移除句柄

    def set_quant_config(self, quant_config: dict) -> None:  # 设置量化配置
        self.quant_config = quant_config  # 更新量化配置字典

    def set_overlap_args(  # 设置重叠参数
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict  # 合并重叠参数和元数据重叠参数
    ) -> None:
        self.overlap_args = combine_overlap_args  # 保存合并重叠参数
        self.meta_overlap_args = meta_overlap_args  # 保存元数据重叠参数

    def clear_overlap_args(self) -> None:  # 清除重叠参数
        self.overlap_args = None  # 置空合并重叠参数
        self.meta_overlap_args = None  # 置空元数据重叠参数
