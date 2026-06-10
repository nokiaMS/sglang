# Copyright 2025 SGLang Team  # 版权所有 2025 SGLang团队
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证2.0版（"许可证"）授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的担保
# See the License for the specific language governing permissions and  # 参见许可证了解管理权限和
# limitations under the License.  # 限制的特定语言
# ==============================================================================  # ==============================================================================
"""TP-sharded linear wrappers with per-tensor activation clamping.  # 带有逐张量激活裁剪的TP分片线性层包装器

Used by the Gemma 4 vision and audio encoders.  Each wrapper owns a parallel  # 由Gemma 4视觉和音频编码器使用。每个包装器拥有一个并行
linear and four scalar clip buffers (``input_min/max``, ``output_min/max``)  # 线性层和四个标量裁剪缓冲区（input_min/max、output_min/max）
that default to ±inf (no-op) and are populated from the checkpoint.  # 默认值为±inf（无操作），从检查点加载填充

For fused projections (QKV, GateUp), input bounds are shared (the checkpoint  # 对于融合投影（QKV、GateUp），输入边界是共享的（检查点
stores identical copies per projection — last write wins during loading) and  # 每个投影存储相同副本——加载时最后写入生效），
output bounds are per-projection.  # 输出边界是逐投影的
"""

from typing import Optional, Tuple  # 导入类型提示工具

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块

from sglang.srt.layers.dp_attention import get_attention_tp_size  # 导入获取注意力TP大小函数
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,  # 列并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.utils import add_prefix  # 导入前缀添加工具函数

_INF = float("inf")  # 无穷大常量


class ClippableRowParallelLinear(nn.Module):  # 带输入/输出激活裁剪的行并行线性层
    """``RowParallelLinear`` with input/output activation clamping.  # 带输入/输出激活裁剪的RowParallelLinear

    Checkpoint weight at ``<name>.weight`` is remapped to ``<name>.linear.weight``  # 检查点中<name>.weight的权重被重映射到<name>.linear.weight
    by the model's ``load_weights``.  # 由模型的load_weights方法完成
    """

    def __init__(  # 初始化方法
        self,
        input_size: int,  # 输入维度大小
        output_size: int,  # 输出维度大小
        *,
        bias: bool = True,  # 是否使用偏置（默认True）
        quant_config: Optional[QuantizationConfig] = None,  # 可选的量化配置
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.linear = RowParallelLinear(  # 创建行并行线性层
            input_size=input_size,  # 输入维度
            output_size=output_size,  # 输出维度
            bias=bias,  # 偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("linear", prefix),  # 添加前缀
        )
        self.input_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)  # 输入最小裁剪值，默认-∞
        self.input_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # 输入最大裁剪值，默认+∞
        self.output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)  # 输出最小裁剪值，默认-∞
        self.output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # 输出最大裁剪值，默认+∞

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播，执行裁剪→线性→裁剪
        x = torch.clamp(x, self.input_min, self.input_max)  # 裁剪输入到[input_min, input_max]
        x, _ = self.linear(x)  # 通过行并行线性层
        x = torch.clamp(x, self.output_min, self.output_max)  # 裁剪输出到[output_min, output_max]
        return x  # 返回裁剪后的输出


class ClippableColumnParallelLinear(nn.Module):  # 带输入/输出激活裁剪的列并行线性层
    """``ColumnParallelLinear`` with input/output activation clamping.  # 带输入/输出激活裁剪的ColumnParallelLinear"""

    def __init__(  # 初始化方法
        self,
        input_size: int,  # 输入维度大小
        output_size: int,  # 输出维度大小
        *,
        bias: bool = False,  # 是否使用偏置（默认False）
        quant_config: Optional[QuantizationConfig] = None,  # 可选的量化配置
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.linear = ColumnParallelLinear(  # 创建列并行线性层
            input_size=input_size,  # 输入维度
            output_size=output_size,  # 输出维度
            bias=bias,  # 偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("linear", prefix),  # 添加前缀
        )
        self.input_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)  # 输入最小裁剪值，默认-∞
        self.input_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # 输入最大裁剪值，默认+∞
        self.output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)  # 输出最小裁剪值，默认-∞
        self.output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # 输出最大裁剪值，默认+∞

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播，执行裁剪→线性→裁剪
        x = torch.clamp(x, self.input_min, self.input_max)  # 裁剪输入到[input_min, input_max]
        x, _ = self.linear(x)  # 通过列并行线性层
        x = torch.clamp(x, self.output_min, self.output_max)  # 裁剪输出到[output_min, output_max]
        return x  # 返回裁剪后的输出


class ClippableQKVParallelLinear(nn.Module):  # 带逐投影激活裁剪的融合QKV投影层
    """Fused QKV projection with per-projection activation clamping.  # 带逐投影激活裁剪的融合QKV投影

    Owns a single ``QKVParallelLinear`` for the fused matmul.  Clip bounds  # 拥有单个QKVParallelLinear用于融合矩阵乘法。裁剪边界
    are stored as flat buffers: shared ``input_min/max`` (applied before the  # 存储为扁平缓冲区：共享的input_min/max（在矩阵乘法前应用）
    matmul) and per-projection ``q/k/v_output_min/max`` (applied after split).  # 和逐投影的q/k/v_output_min/max（在拆分后应用）
    """

    def __init__(  # 初始化方法
        self,
        hidden_size: int,  # 隐藏维度大小
        head_size: int,  # 每个注意力头的维度大小
        total_num_heads: int,  # 总查询头数
        total_num_kv_heads: int,  # 总KV头数
        *,
        bias: bool = False,  # 是否使用偏置（默认False）
        quant_config: Optional[QuantizationConfig] = None,  # 可选的量化配置
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        tp_size = get_attention_tp_size()  # 获取注意力TP大小
        self.q_size = (total_num_heads // tp_size) * head_size  # 当前分片的查询维度大小
        self.kv_size = (total_num_kv_heads // tp_size) * head_size  # 当前分片的KV维度大小

        self.qkv_proj = QKVParallelLinear(  # 创建QKV并行线性层
            hidden_size=hidden_size,  # 隐藏维度
            head_size=head_size,  # 头维度
            total_num_heads=total_num_heads,  # 总查询头数
            total_num_kv_heads=total_num_kv_heads,  # 总KV头数
            bias=bias,  # 偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 添加前缀
        )
        self.input_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)  # 共享输入最小裁剪值
        self.input_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # 共享输入最大裁剪值
        self.q_output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)  # 查询输出最小裁剪值
        self.q_output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # 查询输出最大裁剪值
        self.k_output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)  # 键输出最小裁剪值
        self.k_output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # 键输出最大裁剪值
        self.v_output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)  # 值输出最小裁剪值
        self.v_output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # 值输出最大裁剪值

    def forward(  # 前向传播，执行裁剪→QKV投影→拆分→逐投影裁剪
        self, hidden_states: torch.Tensor  # 输入隐藏状态
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回裁剪后的(q, k, v)
        x = torch.clamp(hidden_states, self.input_min, self.input_max)  # 裁剪输入
        qkv, _ = self.qkv_proj(x)  # 通过QKV并行线性层
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分为q、k、v
        q = torch.clamp(q, self.q_output_min, self.q_output_max)  # 裁剪查询输出
        k = torch.clamp(k, self.k_output_min, self.k_output_max)  # 裁剪键输出
        v = torch.clamp(v, self.v_output_min, self.v_output_max)  # 裁剪值输出
        return q, k, v  # 返回裁剪后的q、k、v


class ClippableGLUParallelLinear(nn.Module):  # 带正确TP分片的融合线性+GLU门控层
    """Fused linear + GLU gating with correct TP sharding.  # 带正确TP分片的融合线性+GLU门控

    Used by the audio encoder's ``LightConv1d``, where a single linear  # 由音频编码器的LightConv1d使用，其中单个线性层
    projects to ``[hidden * 2]`` and GLU splits into value/gate halves.  # 投影到[hidden * 2]，GLU拆分为value/gate两半
    A plain ``ColumnParallelLinear`` is *incorrect* here under TP because it  # 普通的ColumnParallelLinear在TP下是*不正确*的，因为它
    shards the output contiguously, mixing value and gate across ranks.  # 连续分片输出，跨rank混合value和gate
    This wrapper uses ``MergedColumnParallelLinear`` to shard each half  # 此包装器使用MergedColumnParallelLinear分别分片两半
    independently, then applies GLU (``value * sigmoid(gate)``) on each  # 然后在每个rank的正确配对分片上应用GLU（value * sigmoid(gate)）
    rank's correctly-paired shard.  # 
    rank的正确配对分片。

    Output clamping is applied once *after* the GLU gate, using a single  # 输出裁剪在GLU门控后应用一次，使用单个
    ``output_min/max`` pair (matching the checkpoint layout).  # output_min/max对（与检查点布局匹配）

    The checkpoint stores a single fused ``[hidden * 2, input]`` weight.  # 检查点存储单个融合的[hidden * 2, input]权重
    A custom ``weight_loader`` on the inner param automatically splits it  # 内部参数上的自定义weight_loader自动将其拆分
    into value (first half) and gate (second half) shards, so no special  # 为value（前半）和gate（后半）分片，因此不需要
    handling is needed in the model's ``load_weights``.  # 在模型的load_weights中特殊处理
    """

    def __init__(  # 初始化方法
        self,
        input_size: int,  # 输入维度大小
        hidden_size: int,  # 隐藏维度大小
        *,
        bias: bool = False,  # 是否使用偏置（默认False）
        quant_config: Optional[QuantizationConfig] = None,  # 可选的量化配置
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        tp_size = get_attention_tp_size()  # 获取注意力TP大小
        self.proj_size = hidden_size // tp_size  # 当前分片的投影维度大小

        self.linear = MergedColumnParallelLinear(  # 创建合并列并行线性层
            input_size=input_size,  # 输入维度
            output_sizes=[hidden_size, hidden_size],  # 两个输出维度（value和gate）
            bias=bias,  # 偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("linear", prefix),  # 添加前缀
        )

        # The checkpoint has a single fused weight; MergedColumnParallelLinear  # 检查点有单个融合权重；MergedColumnParallelLinear
        # expects per-shard loading.  Wrap the original weight_loader so that  # 期望逐分片加载。包装原始weight_loader以便
        # a call *without* shard_id (the generic load_weights path) splits  # 不带shard_id的调用（通用load_weights路径）自动拆分
        # automatically.  # 
        orig_loader = self.linear.weight.weight_loader  # 保存原始权重加载器

        def _fused_weight_loader(param, loaded_weight, loaded_shard_id=None):  # 自定义融合权重加载器
            if loaded_shard_id is not None:  # 如果提供了分片ID
                return orig_loader(param, loaded_weight, loaded_shard_id)  # 使用原始加载器加载
            half = loaded_weight.shape[0] // 2  # 计算半数行数
            orig_loader(param, loaded_weight[:half], 0)  # 前半部分作为value分片（shard_id=0）
            orig_loader(param, loaded_weight[half:], 1)  # 后半部分作为gate分片（shard_id=1）

        self.linear.weight.weight_loader = _fused_weight_loader  # 替换为自定义权重加载器

        self.input_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)  # 输入最小裁剪值
        self.input_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # 输入最大裁剪值
        self.output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)  # 输出最小裁剪值
        self.output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # 输出最大裁剪值

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播，执行裁剪→融合投影→GLU→裁剪
        x = torch.clamp(x, self.input_min, self.input_max)  # 裁剪输入
        merged, _ = self.linear(x)  # 通过合并列并行线性层
        value, gate = merged.split([self.proj_size, self.proj_size], dim=-1)  # 拆分为value和gate
        x = value * torch.sigmoid(gate)  # 应用GLU：value * sigmoid(gate)
        x = torch.clamp(x, self.output_min, self.output_max)  # 裁剪输出
        return x  # 返回裁剪后的输出


class ClippableGateUpParallelLinear(nn.Module):  # 带逐投影激活裁剪的融合gate/up投影层
    """Fused gate/up projection with per-projection activation clamping.  # 带逐投影激活裁剪的融合gate/up投影

    Used by the MLP layers in the vision/audio encoders.  Owns a single  # 由视觉/音频编码器中的MLP层使用。拥有单个
    ``MergedColumnParallelLinear`` for the fused matmul and returns the  # MergedColumnParallelLinear用于融合矩阵乘法，并返回
    two projections separately so the caller can apply its own activation  # 两个独立的投影，以便调用者可以应用自己的激活
    (e.g. ``SiLU(gate) * up``).  # （例如SiLU(gate) * up）

    Output clamping is applied *per-projection before* the caller's  # 输出裁剪在调用者的激活之前逐投影应用
    activation, using separate ``gate_output_min/max`` and  # 使用独立的gate_output_min/max和
    ``up_output_min/max`` bounds.  # up_output_min/max边界
    """

    def __init__(  # 初始化方法
        self,
        input_size: int,  # 输入维度大小
        intermediate_size: int,  # 中间层维度大小
        *,
        bias: bool = False,  # 是否使用偏置（默认False）
        quant_config: Optional[QuantizationConfig] = None,  # 可选的量化配置
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        tp_size = get_attention_tp_size()  # 获取注意力TP大小
        self.proj_size = intermediate_size // tp_size  # 当前分片的投影维度大小

        self.gate_up_proj = MergedColumnParallelLinear(  # 创建gate/up合并列并行线性层
            input_size=input_size,  # 输入维度
            output_sizes=[intermediate_size, intermediate_size],  # 两个输出维度（gate和up）
            bias=bias,  # 偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 添加前缀
        )
        self.input_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)  # 共享输入最小裁剪值
        self.input_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # 共享输入最大裁剪值
        self.gate_output_min = nn.parameter.Buffer(  # gate输出最小裁剪值
            torch.tensor(-_INF), persistent=False  # 默认-∞
        )
        self.gate_output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # gate输出最大裁剪值
        self.up_output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)  # up输出最小裁剪值
        self.up_output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)  # up输出最大裁剪值

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # 前向传播，执行裁剪→融合投影→拆分→逐投影裁剪
        x = torch.clamp(x, self.input_min, self.input_max)  # 裁剪输入
        gate_up, _ = self.gate_up_proj(x)  # 通过gate/up合并列并行线性层
        gate, up = gate_up.split([self.proj_size, self.proj_size], dim=-1)  # 拆分为gate和up
        gate = torch.clamp(gate, self.gate_output_min, self.gate_output_max)  # 裁剪gate输出
        up = torch.clamp(up, self.up_output_min, self.up_output_max)  # 裁剪up输出
        return gate, up  # 返回裁剪后的gate和up
