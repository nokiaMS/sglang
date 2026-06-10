# Copyright 2023-2024 SGLang Team  # 版权所有 2023-2024 SGLang团队
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
# DSA（Data-Shared Attention）上下文并行(CP)通信器模块
# 实现DSA/MLA预填充CP模式下的层间通信逻辑，包括CP全聚集和归约散射操作


from functools import partial  # 导入偏函数工具
from typing import Callable, Optional  # 导入类型提示

import torch  # 导入PyTorch库

from sglang.srt.layers.attention.dsa.utils import (  # 导入DSA工具函数
    dsa_use_prefill_cp,  # DSA是否使用预填充CP
    is_dsa_enable_prefill_cp,  # DSA是否启用预填充CP
)
from sglang.srt.layers.communicator import (  # 导入基础通信器模块
    CommunicateContext,  # 通信上下文
    CommunicateSimpleFn,  # 简单通信函数类
    CommunicateSummableTensorPairFn,  # 可求和张量对通信函数类
    CommunicateWithAllReduceAndLayerNormFn,  # 带全归约和层归一化的通信函数类
    LayerCommunicator,  # 层通信器
    LayerScatterModes,  # 层散射模式
    ScatterMode,  # 散射模式枚举
)
from sglang.srt.layers.dp_attention import (  # 导入DP注意力函数
    attn_cp_all_gather_into_tensor,  # 注意力CP全聚集到张量
    attn_cp_reduce_scatter_tensor,  # 注意力CP归约散射到张量
    get_attention_cp_group,  # 获取注意力CP通信组
    get_attention_cp_rank,  # 获取注意力CP rank
    get_attention_cp_size,  # 获取注意力CP大小
    get_attention_dp_size,  # 获取注意力DP大小
    get_attention_tp_size,  # 获取注意力TP大小
    get_local_dp_buffer,  # 获取本地DP缓冲区
)
from sglang.srt.layers.utils.cp_utils import mla_use_prefill_cp  # 导入MLA预填充CP判断函数
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息


def dsa_enable_prefill_cp():  # 判断DSA是否启用预填充上下文并行
    # After using cp, the communication mode of this part changes.  # 使用CP后，此部分的通信模式发生变化
    # The three parts of prepare_attn, prepare_mlp, and postprocess_layer  # prepare_attn、prepare_mlp和postprocess_layer三部分
    # no longer require additional communication for reduce, scatter, etc.  # 不再需要额外的归约、散射等通信
    return is_dsa_enable_prefill_cp()  # 返回DSA预填充CP是否启用


def dsa_cp_gather_hidden_states(hidden_states: torch.Tensor):  # DSA CP全聚集隐藏状态
    attn_dp_size = get_attention_dp_size()  # 获取注意力DP大小
    attn_tp_size = get_attention_tp_size()  # 获取注意力TP大小
    assert attn_dp_size == 1 and attn_tp_size == 1  # 断言DP和TP大小都为1
    hidden_states, local_hidden_states = (  # 准备全聚集
        get_local_dp_buffer(get_attention_cp_group()),  # 从DP缓冲区获取输出
        hidden_states,  # 本地隐藏状态
    )
    attn_cp_all_gather_into_tensor(hidden_states, local_hidden_states)  # 注意力CP全聚集
    return hidden_states  # 返回聚集后的隐藏状态


def dsa_cp_reduce_scatter_hidden_states(hidden_states: torch.Tensor):  # DSA CP归约散射隐藏状态
    attn_dp_size = get_attention_dp_size()  # 获取注意力DP大小
    attn_tp_size = get_attention_tp_size()  # 获取注意力TP大小
    assert attn_dp_size == 1 and attn_tp_size == 1  # 断言DP和TP大小都为1
    cp_size = get_attention_cp_size()  # 获取CP大小
    cp_rank = get_attention_cp_rank()  # 获取CP rank
    input_hidden_states = hidden_states  # 保存原始隐藏状态引用
    hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]  # 按CP大小拆分取当前rank
    attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)  # 注意力CP归约散射
    return hidden_states  # 返回散射后的隐藏状态


class DSACPLayerCommunicator(LayerCommunicator):  # DSA上下文并行的层通信器，继承自LayerCommunicator
    def __init__(  # 初始化方法
        self,
        layer_scatter_modes: LayerScatterModes,  # 层散射模式
        input_layernorm: torch.nn.Module,  # 输入层归一化模块
        post_attention_layernorm: torch.nn.Module,  # 注意力后层归一化模块
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.  # 归约散射需要跳过MoE/MLP后模型代码中的全归约，因此仅对已实现的模型启用。对所有使用LayerCommunicator的模型完成实现后移除此标志
        allow_reduce_scatter: bool = False,  # 是否允许归约散射
        is_last_layer: bool = False,  # 是否是最后一层
        qkv_latent_func: Optional[Callable] = None,  # 可选的QKV潜变量函数
    ):
        super().__init__(  # 调用父类初始化
            layer_scatter_modes,  # 层散射模式
            input_layernorm,  # 输入层归一化
            post_attention_layernorm,  # 注意力后层归一化
            allow_reduce_scatter,  # 归约散射标志
            is_last_layer,  # 最后层标志
            qkv_latent_func,  # QKV潜变量函数
        )

    def _post_init_communicate(self):  # 后初始化通信函数，使用DSA CP专用的通信函数
        # SCATTERED in attn tp is different from SCATTERED in global tp when dp_size > 1  # 当dp_size > 1时，注意力TP中的SCATTERED与全局TP中的SCATTERED不同
        if self.layer_scatter_modes.mlp_mode != ScatterMode.SCATTERED:  # 如果MLP模式不是散射
            assert (  # 断言DP大小为1
                self._context.attn_dp_size == 1
            ), f"dp_size should be 1 when moe_runner_backend is none"  # moe_runner_backend为None时dp_size应为1
        self._communicate_simple_fn = DSACPCommunicateSimpleFn.get_fn(  # 获取DSA CP简单通信函数
            input_mode=ScatterMode.SCATTERED,  # 输入模式为散射
            output_mode=ScatterMode.SCATTERED,  # 输出模式为散射
            context=self._context,  # 通信上下文
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = DSACPCommunicateWithAllReduceAndLayerNormFn.get_fn(  # 获取DSA CP带全归约和层归一化的通信函数
            hidden_states_input_mode=ScatterMode.SCATTERED,  # 隐藏状态输入模式为散射
            residual_input_mode=ScatterMode.SCATTERED,  # 残差输入模式为散射
            hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL  # 隐藏状态输出模式
            residual_output_mode=ScatterMode.SCATTERED,  # 残差输出模式为散射
            context=self._context,  # 通信上下文
        )
        self._communicate_summable_tensor_pair_fn = DSACPCommunicateSummableTensorPairFn.get_fn(  # 获取DSA CP可求和张量对通信函数
            hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL  # 隐藏状态输入模式
            residual_input_mode=ScatterMode.SCATTERED,  # 残差输入模式为散射
            output_mode=ScatterMode.SCATTERED,  # 输出模式为散射
            context=self._context,  # 通信上下文
        )


class DSACPCommunicateSimpleFn(CommunicateSimpleFn):  # DSA CP简单通信函数类，继承自CommunicateSimpleFn
    @staticmethod
    def get_fn(  # 根据输入输出模式获取对应的通信函数
        input_mode: ScatterMode,  # 输入散射模式
        output_mode: ScatterMode,  # 输出散射模式
        context: CommunicateContext,  # 通信上下文
    ):
        if context.is_same_group_size(input_mode, output_mode):  # 如果输入输出组大小相同
            return DSACPCommunicateSimpleFn._trivial  # 返回平凡函数

        raise NotImplementedError(f"{input_mode=} {output_mode=}")  # 其他模式未实现


class DSACPCommunicateWithAllReduceAndLayerNormFn(  # DSA CP带全归约和层归一化的通信函数类
    CommunicateWithAllReduceAndLayerNormFn
):
    """Besides communication, needs to  # 除通信外，还需要
    1. All reduce in tp_attn_group on hidden_states  # 1. 在tp_attn_group中对隐藏状态执行全归约
    2. Apply layer norm  # 2. 应用层归一化
    """

    @staticmethod
    def get_fn(  # 根据输入输出模式获取对应的通信函数
        hidden_states_input_mode: ScatterMode,  # 隐藏状态输入模式
        residual_input_mode: ScatterMode,  # 残差输入模式
        hidden_states_output_mode: ScatterMode,  # 隐藏状态输出模式
        residual_output_mode: ScatterMode,  # 残差输出模式
        context: CommunicateContext,  # 通信上下文
    ):
        assert hidden_states_input_mode == ScatterMode.SCATTERED  # 断言隐藏状态输入为散射模式
        assert residual_input_mode == ScatterMode.SCATTERED  # 断言残差输入为散射模式
        assert residual_output_mode == ScatterMode.SCATTERED  # 断言残差输出为散射模式
        if hidden_states_output_mode == ScatterMode.SCATTERED:  # 如果隐藏状态输出为散射模式
            return DSACPCommunicateWithAllReduceAndLayerNormFn._simple  # 返回简单函数

        if hidden_states_output_mode == ScatterMode.FULL:  # 如果隐藏状态输出为完整模式
            return partial(  # 返回部分应用函数
                DSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,  # 传入残差输入模式
            )

        raise NotImplementedError(  # 其他模式未实现
            f"{hidden_states_input_mode=} {residual_input_mode=} {hidden_states_output_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _gather_hidden_states_and_residual(  # 聚集隐藏状态和残差
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        layernorm: torch.nn.Module,  # 层归一化模块
        context: CommunicateContext,  # 通信上下文
        *,  # 以下为关键字参数
        residual_input_mode,  # 残差输入模式
    ):
        if hidden_states.shape[0] != 0:  # 如果有token
            hidden_states, residual = layernorm(hidden_states, residual)  # 应用层归一化
        # for prefill: attn tp scattered -> full  # 预填充：注意力TP散射 -> 完整
        # for decode: attn tp full -> full  # 解码：注意力TP完整 -> 完整
        if dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch):  # 如果DSA或MLA使用预填充CP
            hidden_states = dsa_cp_gather_hidden_states(hidden_states)  # DSA CP全聚集隐藏状态
        return hidden_states, residual  # 返回隐藏状态和残差


class DSACPCommunicateSummableTensorPairFn(CommunicateSummableTensorPairFn):  # DSA CP可求和张量对通信函数类
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""  # 如果需要，允许将(hidden_states, residual)设为(hidden_states + residual, None)

    @staticmethod
    def get_fn(  # 根据输入输出模式获取对应的通信函数
        hidden_states_input_mode: ScatterMode,  # 隐藏状态输入模式
        residual_input_mode: ScatterMode,  # 残差输入模式
        output_mode: ScatterMode,  # 输出模式
        context: CommunicateContext,  # 通信上下文
    ):
        # Check exact enum match first: even if group sizes happen to be equal  # 首先检查精确的枚举匹配：即使组大小恰好相等
        # (e.g. tp_size == attn_cp_size makes FULL and SCATTERED both size 1),  # （例如tp_size == attn_cp_size使FULL和SCATTERED大小都为1），
        # FULL and SCATTERED have different data layouts under CP and require  # FULL和SCATTERED在CP下有不同的数据布局，需要
        # an explicit scatter operation.  # 显式散射操作
        if (  # FULL + SCATTERED -> SCATTERED
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            return DSACPCommunicateSummableTensorPairFn._scatter_hidden_states  # 返回散射隐藏状态函数

        if context.is_same_group_size(  # 如果隐藏状态和输出组大小相同
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):  # 残差和输出组大小也相同
            return DSACPCommunicateSummableTensorPairFn._trivial  # 返回平凡函数

        raise NotImplementedError(  # 其他模式未实现
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    @staticmethod
    def _scatter_hidden_states(  # 散射隐藏状态（FULL -> SCATTERED）
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        context: CommunicateContext,  # 通信上下文
        allow_reduce_scatter: bool = False,  # 是否允许归约散射
    ):
        # for prefill: full -> attn tp scattered  # 预填充：完整 -> 注意力TP散射
        # for decode: full -> attn tp full  # 解码：完整 -> 注意力TP完整
        if dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch):  # 如果DSA或MLA使用预填充CP
            hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)  # DSA CP归约散射隐藏状态
        return hidden_states, residual  # 返回隐藏状态和残差
