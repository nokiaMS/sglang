# GPU端阶段状态检查器，用于验证和推进整数键控的阶段序列状态机
# 提供GPU内核级别的阶段断言检查，确保推理流程中各阶段按预期顺序执行
from __future__ import annotations  # 启用延迟注解求值

from enum import IntEnum  # 导入整数枚举基类

import torch  # 导入PyTorch张量库
import triton  # 导入Triton GPU编程框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.environ import envs  # 导入环境变量配置


def _phase_repr(phase: int | IntEnum) -> str:  # 将阶段值转换为可读字符串表示
    if isinstance(phase, IntEnum):  # 如果是枚举类型
        return f"{phase.name}({int(phase)})"  # 返回枚举名称和整数值
    return str(int(phase))  # 否则直接返回整数值的字符串


def _host_debug(msg: str) -> None:  # 主机端调试打印函数
    if envs.SGLANG_PHASE_CHECKER_DEBUG.get():  # 如果启用了阶段检查器调试
        print(msg, flush=True)  # 立即刷新打印调试信息


# debug=True so tl.device_assert below actually raises. Without it the assert
# is stripped at compile time and only tl.device_print fires (the assert is
# gated on the TRITON_DEBUG env var by default — see tl.device_assert docstring).
# debug=True 使 tl.device_assert 实际触发断言。否则断言在编译时被剥离，只有 tl.device_print 生效
@triton.jit(debug=True)  # Triton JIT编译内核，启用调试模式使断言生效
def _phase_check_kernel(  # 阶段检查GPU内核函数
    phase_ptr,  # 指向当前阶段值的设备内存指针
    enable_assert_ptr,  # 指向断言启用标志的设备内存指针
    EXPECT_PHASE: tl.constexpr,  # 期望的阶段值（编译时常量）
    NEXT_PHASE: tl.constexpr,  # 下一阶段值（编译时常量）
    CALLER_TAG: tl.constexpr,  # 调用者标签（编译时常量）
):
    cur = tl.load(phase_ptr)  # 从设备内存加载当前阶段值
    enable_assert = tl.load(enable_assert_ptr)  # 加载断言启用标志
    if enable_assert != 0:  # 如果断言已启用
        if cur != EXPECT_PHASE:  # 如果当前阶段与期望阶段不匹配
            # constexpr values get baked into the prefix string at compile time;
            # only `cur` is runtime.
            # constexpr值在编译时嵌入前缀字符串；只有cur是运行时值
            tl.device_print(  # 在设备端打印阶段不匹配信息
                f"[SimplePhaseChecker FAIL] caller_tag={CALLER_TAG} "
                f"expect={EXPECT_PHASE} next={NEXT_PHASE} actual=",
                cur,  # 打印实际阶段值
            )
        tl.device_assert(cur == EXPECT_PHASE, "SimplePhaseChecker: phase mismatch")  # 设备端断言检查阶段是否匹配
    tl.store(phase_ptr, NEXT_PHASE)  # 将阶段值更新为下一阶段


class SimplePhaseChecker:  # 简单阶段检查器类
    """GPU-side state machine for any int-keyed phase sequence."""  # GPU端状态机，用于任意整数键控的阶段序列

    def __init__(self, *, initial_phase: int | IntEnum, device: torch.device) -> None:  # 初始化阶段检查器
        self._initial_phase = int(initial_phase)  # 保存初始阶段值（转为整数）
        self._phase = torch.tensor(  # 创建存储当前阶段值的设备张量
            self._initial_phase, dtype=torch.int32, device=device  # 使用int32类型，存储在指定设备上
        )
        self._enable_assert_device = torch.zeros(1, dtype=torch.int32, device=device)  # 断言启用标志，初始为0（关闭）
        self._caller_tag_registry: dict[str, int] = {}  # 调用者标签注册表，将名称映射到数字标签
        _host_debug(  # 打印初始化调试信息
            f"[SimplePhaseChecker.__init__] device={device} "
            f"initial_phase={_phase_repr(initial_phase)} "
            f"enable_assert=OFF (call enable_assert() after init is done)"  # 断言默认关闭，初始化完成后调用enable_assert()开启
        )

    def enable_assert(self) -> None:  # 启用设备端断言检查
        """Reset phase to initial_phase, then enable the device-side assert."""  # 重置阶段到初始值，然后启用设备端断言
        self._reset_to_idle()  # 先重置阶段到初始值
        self._enable_assert_device.fill_(1)  # 将断言启用标志设为1
        _host_debug(f"[SimplePhaseChecker.enable_assert] assert ENABLED")  # 打印断言已启用

    def update(  # 更新阶段检查器（验证当前阶段并推进到下一阶段）
        self,
        *,
        expect_phase: int | IntEnum,  # 期望的当前阶段值
        next_phase: int | IntEnum,  # 要推进到的下一阶段值
        caller_name: str = "",  # 调用者名称（用于调试）
    ) -> None:
        caller_tag = self._resolve_caller_tag(caller_name)  # 解析调用者名称为数字标签
        _host_debug(  # 打印更新调试信息
            f"[SimplePhaseChecker.update] caller={caller_name!r} "
            f"caller_tag={caller_tag} "
            f"expect={_phase_repr(expect_phase)} "
            f"next={_phase_repr(next_phase)} "
            f"capturing={torch.cuda.is_current_stream_capturing()}"  # 显示当前CUDA流是否正在捕获
        )
        _phase_check_kernel[(1,)](  # 启动阶段检查内核（1个线程块）
            self._phase,  # 当前阶段值张量
            self._enable_assert_device,  # 断言启用标志张量
            EXPECT_PHASE=int(expect_phase),  # 期望阶段值
            NEXT_PHASE=int(next_phase),  # 下一阶段值
            CALLER_TAG=caller_tag,  # 调用者标签
        )

    def _reset_to_idle(self) -> None:  # 重置阶段到初始值
        self._phase.fill_(self._initial_phase)  # 将阶段值填充回初始值
        _host_debug(  # 打印重置调试信息
            f"[SimplePhaseChecker._reset_to_idle] phase reset to "
            f"{self._initial_phase}"  # 显示重置后的阶段值
        )

    def _resolve_caller_tag(self, caller_name: str) -> int:  # 将调用者名称解析为数字标签
        registry = self._caller_tag_registry  # 获取调用者标签注册表
        if caller_name not in registry:  # 如果调用者名称尚未注册
            registry[caller_name] = len(registry) + 1  # 分配新的数字标签（从1开始递增）
            _host_debug(  # 打印新标签注册信息
                f"[SimplePhaseChecker] registered caller_tag "
                f"{registry[caller_name]} <- {caller_name!r}"  # 显示标签编号与名称的映射
            )
        return registry[caller_name]  # 返回调用者对应的数字标签
