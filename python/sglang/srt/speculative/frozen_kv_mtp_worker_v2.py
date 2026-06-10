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
# 冻结KV MTP（多token预测）工作器的V2版本，用于重叠调度场景。
# 目前尚未实现，仅作为占位符存在，调用时会抛出NotImplementedError。
"""Overlap-scheduling placeholder for frozen-KV MTP (raises until implemented)."""

from __future__ import annotations  # 启用延迟类型注解评估

from typing import Optional  # 导入可选类型

from sglang.srt.managers.tp_worker import TpModelWorker  # 导入张量并行模型工作器
from sglang.srt.server_args import ServerArgs  # 导入服务器参数类
from sglang.srt.speculative.frozen_kv_mtp_worker import FrozenKVMTPWorker  # 导入冻结KV MTP工作器基类


class FrozenKVMTPWorkerV2(FrozenKVMTPWorker):
    """冻结KV MTP工作器V2版本，支持重叠调度，目前尚未实现。"""

    def __init__(
        self,
        server_args: ServerArgs,  # 服务器参数
        gpu_id: int,  # GPU设备ID
        tp_rank: int,  # 张量并行排名
        dp_rank: Optional[int],  # 数据并行排名，可选
        moe_ep_rank: int,  # 混合专家专家并行排名
        attn_cp_rank: int,  # 注意力上下文并行排名
        moe_dp_rank: int,  # 混合专家数据并行排名
        nccl_port: int,  # NCCL通信端口
        target_worker: TpModelWorker,  # 目标模型工作器
    ):
        """初始化冻结KV MTP工作器V2，目前未实现，抛出异常。"""
        raise NotImplementedError(  # 抛出未实现异常
            "FrozenKVMTPWorkerV2 (overlap scheduling for Frozen-KV MTP) is "  # 提示V2版本尚未实现
            "not yet implemented. Pass --disable-overlap-schedule to use "  # 建议使用V1版本
            "FrozenKVMTPWorker."  # 建议使用FrozenKVMTPWorker
        )
