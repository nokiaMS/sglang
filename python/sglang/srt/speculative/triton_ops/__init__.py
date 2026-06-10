# Copyright 2023-2024 SGLang Team
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
# 推测解码Triton算子包的初始化模块。
# 导出FusedKVMaterializeHelper用于融合KV缓存物化操作。
"""Triton kernels for speculative decoding."""

from sglang.srt.speculative.triton_ops.fused_kv_materialize import (  # 导入融合KV物化辅助类
    FusedKVMaterializeHelper,
)

__all__ = ["FusedKVMaterializeHelper"]  # 模块公开接口列表
