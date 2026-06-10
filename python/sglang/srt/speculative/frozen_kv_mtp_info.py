# Frozen-KV MTP 信息类模块
# 定义Frozen-KV MTP（多token预测）的上下文、草稿输入、扩展输入和验证输入数据类，
# 复用EAGLE调度/注意力契约但使用专用类型以便算法特定行为迁移。
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations  # 启用延迟注解求值

from dataclasses import dataclass, fields  # 导入数据类和字段工具
from typing import Dict  # 导入字典类型

from sglang.srt.mem_cache.memory_pool import KVCache  # 导入KV缓存类
from sglang.srt.speculative.eagle_info import (  # 导入EAGLE信息类
    EagleDraftExtendInput,  # EAGLE草稿扩展输入
    EagleDraftInput,  # EAGLE草稿输入
    EagleVerifyInput,  # EAGLE验证输入
    EagleVerifyOutput,  # EAGLE验证输出
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType  # 导入投机输入基类和枚举


@dataclass(frozen=True)  # 不可变数据类
class FrozenKVMTPContext:
    # Frozen-KV MTP上下文：目标KV池和助手逻辑层到目标物理层的映射
    """Target KV pool + assistant-logical -> target-physical layer map."""

    target_token_to_kv_pool: KVCache  # 目标token到KV缓存池
    physical_layer_ids: Dict[int, int]  # 逻辑层ID到物理层ID的映射

    def get_physical_layer_id(self, idx: int) -> int:
        # 根据助手逻辑层索引获取目标物理层ID
        if idx not in self.physical_layer_ids:  # 索引不存在
            raise KeyError(
                f"FrozenKVMTPContext has no physical layer id for assistant "
                f"logical index {idx}; available: {sorted(self.physical_layer_ids)}"
            )
        return self.physical_layer_ids[idx]  # 返回物理层ID


@dataclass  # 数据类
class FrozenKVMTPDraftInput(EagleDraftInput):
    # Frozen-KV MTP草稿输入，复用EAGLE调度契约但使用专用类型
    """Draft input for Frozen-KV MTP.
    # Frozen-KV MTP的草稿输入。

    Frozen-KV MTP currently reuses the EAGLE scheduler/attention contract, but
    has a dedicated type so algorithm-specific behavior can move here over time.
    # Frozen-KV MTP目前复用EAGLE调度/注意力契约，但有专用类型，
    # 以便算法特定行为随时间迁移到这里。
    """

    def __post_init__(self):  # 后初始化方法
        SpecInput.__init__(self, SpecInputType.FROZEN_KV_MTP_DRAFT)  # 设置输入类型


@dataclass  # 数据类
class FrozenKVMTPDraftExtendInput(EagleDraftExtendInput):
    # Frozen-KV MTP草稿扩展输入，仅标签子类
    """Draft-extend input for Frozen-KV MTP. Tag-only subclass."""

    def __post_init__(self):  # 后初始化方法
        SpecInput.__init__(self, SpecInputType.FROZEN_KV_MTP_DRAFT_EXTEND)  # 设置输入类型


@dataclass  # 数据类
class FrozenKVMTPVerifyInput(EagleVerifyInput):
    # Frozen-KV MTP验证输入
    """Verify input for Frozen-KV MTP."""

    def __post_init__(self):  # 后初始化方法
        SpecInput.__init__(self, SpecInputType.FROZEN_KV_MTP_VERIFY)  # 设置输入类型

    def verify(self, *args, **kwargs) -> EagleVerifyOutput:
        # 执行验证，将draft_extend_input转换为Frozen-KV MTP类型
        output = super().verify(*args, **kwargs)  # 调用父类验证
        output.draft_extend_input = _to_frozen_kv_mtp_draft_extend_input(
            output.draft_extend_input
        )  # 转换类型
        return output  # 返回验证输出


FrozenKVMTPVerifyOutput = EagleVerifyOutput  # Frozen-KV MTP验证输出类型别名


def _to_frozen_kv_mtp_draft_extend_input(
    draft_extend_input: EagleDraftExtendInput,  # EAGLE草稿扩展输入
) -> FrozenKVMTPDraftExtendInput:
    # 将EAGLE草稿扩展输入转换为Frozen-KV MTP草稿扩展输入
    if isinstance(draft_extend_input, FrozenKVMTPDraftExtendInput):  # 已经是目标类型
        return draft_extend_input  # 直接返回
    return FrozenKVMTPDraftExtendInput(  # 创建新实例
        **{
            field.name: getattr(draft_extend_input, field.name)
            for field in fields(EagleDraftExtendInput)  # 复制所有字段
        }
    )
