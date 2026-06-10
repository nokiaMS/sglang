# 百灵MoE（Mixture of Experts）模型推理实现
# 该文件实现了BailingMoE混合专家模型的推理专用版本，主要特点包括：
# - 稀疏MoE层和密集MLP层的混合架构
# - 共享专家和路由专家的并行计算
# - 支持DeepEP后端和张量并行
# - 支持QK归一化和部分旋转位置编码
# - 支持EPLB（专家并行负载均衡）
# coding=utf-8
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved. # 版权归属Antgroup和HuggingFace
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX # 本代码基于EleutherAI的GPT-NeoX库
# and OPT implementations in this library. It has been modified from its # 和OPT实现，已从原始形式修改
# original forms to accommodate minor architectural differences compared # 以适应与GPT-NeoX和OPT的轻微架构差异
# to GPT-NeoX and OPT used by the Meta AI team that trained the model. # 这些差异由训练模型的Meta AI团队引入
#
# Licensed under the Apache License, Version 2.0 (the "License"); # 许可证：Apache 2.0
# you may not use this file except in compliance with the License. # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at # 您可以在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS, # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # 不附带任何明示或暗示的保证
# See the License for the specific language governing permissions and # 请参阅许可证以了解管理权限和
# limitations under the License. # 限制的具体条款
"""SGLang BailingMoE model.""" # SGLang百灵MoE模型

import logging # 导入日志模块
from typing import Iterable, List, Optional, Tuple, Union # 导入类型提示

import torch # 导入PyTorch
import torch.nn.functional as F # 导入PyTorch函数式模块
from torch import nn # 导入神经网络模块
from transformers import PretrainedConfig # 导入预训练配置类

from sglang.srt.distributed import ( # 导入分布式相关模块
    get_pp_group, # 获取流水线并行组
    get_tensor_model_parallel_world_size, # 获取张量并行世界大小
    parallel_state, # 并行状态
    tensor_model_parallel_all_reduce, # 张量并行全归约
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder # 导入全局专家分布记录器
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation # 导入专家位置模型配置
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo # 导入专家位置调度信息
from sglang.srt.layers.activation import SiluAndMul # 导入SiLU与乘法激活函数
from sglang.srt.layers.communicator import ( # 导入通信器模块
    LayerCommunicator, # 层通信器
    LayerScatterModes, # 层散射模式
    enable_moe_dense_fully_dp, # 启用MoE密集全DP
)
from sglang.srt.layers.dp_attention import ( # 导入数据并行注意力模块
    get_attention_dp_size, # 获取注意力DP大小
    get_attention_tp_rank, # 获取注意力TP排名
    get_attention_tp_size, # 获取注意力TP大小
    is_dp_attention_enabled, # 是否启用DP注意力
)
from sglang.srt.layers.layernorm import RMSNorm # 导入RMS层归一化
from sglang.srt.layers.linear import ( # 导入线性层
    MergedColumnParallelLinear, # 合并列并行线性层
    QKVParallelLinear, # QKV并行线性层
    RowParallelLinear, # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor # 导入logits处理器
from sglang.srt.layers.moe import ( # 导入MoE相关模块
    get_deepep_mode, # 获取DeepEP模式
    get_moe_a2a_backend, # 获取MoE全互连后端
    should_skip_post_experts_all_reduce, # 是否跳过专家后全归约
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class # 导入MoE实现类获取工具
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE # 导入融合MoE Triton层
from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher # 导入DeepEP分发器
from sglang.srt.layers.moe.topk import TopK # 导入TopK选择器
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert # 导入MoE权重参数过滤工具
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope # 导入旋转位置编码获取工具
from sglang.srt.layers.utils import PPMissingLayer # 导入流水线并行缺失层
from sglang.srt.layers.vocab_parallel_embedding import ( # 导入词表并行嵌入
    ParallelLMHead, # 并行语言模型头
    VocabParallelEmbedding, # 词表并行嵌入层
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode # 导入CUDA图捕获模式检测
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.models.utils import ( # 导入模型工具
    apply_qk_norm, # 应用QK归一化
    create_fused_set_kv_buffer_arg, # 创建融合设置KV缓冲区参数
    enable_fused_set_kv_buffer, # 启用融合设置KV缓冲区
)
from sglang.srt.server_args import get_global_server_args # 导入全局服务器参数
from sglang.srt.utils import add_prefix, is_cuda, is_non_idle_and_non_empty, make_layers # 导入工具函数

LoraConfig = None # LoRA配置初始化为空
logger = logging.getLogger(__name__) # 获取当前模块的日志记录器
_is_cuda = is_cuda() # 检测是否为CUDA环境


class BailingMoEMLP(nn.Module): # 百灵MoE的MLP模块
    def __init__( # MLP初始化方法
        self,
        intermediate_size: int, # 中间层大小
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        reduce_results: Optional[bool] = True, # 是否归约结果
        prefix: str = "", # 参数前缀
        tp_rank: Optional[int] = None, # 张量并行排名
        tp_size: Optional[int] = None, # 张量并行大小
    ) -> None:
        super().__init__() # 调用父类初始化
        self.tp_size = tp_size # 保存TP大小

        self.gate_up_proj = MergedColumnParallelLinear( # 合并的gate和up投影
            config.hidden_size, # 输入维度
            [intermediate_size] * 2, # 输出维度（gate和up各一份）
            bias=config.use_bias, # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("gate_up_proj", prefix), # 参数前缀
            tp_rank=tp_rank, # TP排名
            tp_size=tp_size, # TP大小
        )
        self.down_proj = RowParallelLinear( # 下投影线性层
            intermediate_size, # 输入维度
            config.hidden_size, # 输出维度
            bias=config.use_bias, # 是否使用偏置
            reduce_results=reduce_results, # 是否归约结果
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("down_proj", prefix), # 参数前缀
            tp_rank=tp_rank, # TP排名
            tp_size=tp_size, # TP大小
        )

        if config.hidden_act != "silu": # 如果激活函数不是silu
            raise ValueError("Unsupported activation. Only silu is supported for now.") # 仅支持silu
        self.act_fn = SiluAndMul() # SiLU与乘法激活函数

    def forward( # MLP前向传播
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: Optional[ForwardBatch] = None, # 前向批次
        should_allreduce_fusion: bool = False, # 是否融合全归约
        use_reduce_scatter: bool = False, # 是否使用reduce-scatter
    ) -> torch.Tensor:
        if (self.tp_size == 1) and hidden_states.shape[0] == 0: # 如果TP大小为1且无token
            return hidden_states # 直接返回空张量

        gate_up, _ = self.gate_up_proj(hidden_states) # gate和up投影
        hidden_states = self.act_fn(gate_up) # 应用激活函数
        hidden_states, _ = self.down_proj( # 下投影
            hidden_states, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter # 是否跳过全归约
        )
        return hidden_states # 返回输出


class BailingMoEGate(nn.Module): # 百灵MoE门控模块
    def __init__( # 门控初始化方法
        self,
        config, # 模型配置
        params_dtype: Optional[torch.dtype] = None, # 参数数据类型
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        if params_dtype is None: # 如果未指定数据类型
            params_dtype = torch.get_default_dtype() # 使用默认数据类型
        self.params_dtype = params_dtype # 保存参数数据类型
        self.weight = nn.Parameter( # 门控权重参数
            torch.empty(
                (config.num_experts, config.hidden_size), # 形状为（专家数，隐藏大小）
                dtype=self.params_dtype,
            ),
        )
        if getattr(config, "moe_router_enable_expert_bias", False): # 如果启用专家偏置
            self.expert_bias = nn.Parameter( # 专家偏置参数
                torch.empty((config.num_experts,), dtype=torch.float32),
            )
        else: # 否则
            self.expert_bias = None # 无专家偏置

    def forward(self, hidden_states): # 门控前向传播
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None).to( # 线性变换计算路由logits
            hidden_states.dtype
        )
        return logits # 返回路由logits


class BailingMoESparseMoeBlock(nn.Module): # 百灵MoE稀疏专家块
    def __init__( # 稀疏MoE块初始化方法
        self,
        layer_id: int, # 层ID
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        alt_stream: Optional[torch.cuda.Stream] = None, # 备用CUDA流
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.layer_id = layer_id # 保存层ID
        self.alt_stream = alt_stream # 保存备用CUDA流
        self.tp_size = get_tensor_model_parallel_world_size() # 获取TP大小
        self.top_k = config.num_experts_per_tok # 每个token选择的专家数
        self.norm_topk_prob = config.norm_topk_prob # 是否归一化top-k概率
        self.hidden_size = config.hidden_size # 隐藏层大小
        self.num_shared_experts = config.num_shared_experts # 共享专家数
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0) # 路由缩放因子
        self.score_function = getattr(config, "score_function", None) # 分数函数类型

        if config.hidden_act != "silu": # 如果激活函数不是silu
            raise ValueError( # 抛出值错误
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now." # 仅支持silu
            )

        # Gate always runs at half / full precision for now. # 门控目前始终以半精度/全精度运行
        router_dtype = getattr(config, "router_dtype", None) # 获取路由器数据类型
        if router_dtype is None: # 如果未指定
            self.router_dtype = None # 不指定
        elif router_dtype == "fp32": # 如果指定为fp32
            self.router_dtype = torch.float32 # 使用fp32
        else: # 否则
            self.router_dtype = torch.bfloat16 # 使用bfloat16

        # TODO global_server_args.ep_num_redundant_experts is used for eplb, not supported now # TODO 全局服务器参数的冗余专家数用于EPLB，目前不支持
        assert get_global_server_args().ep_num_redundant_experts == 0 # 断言冗余专家数为0
        # check group topk # 检查分组top-k
        self.num_expert_group = getattr(config, "n_group", 0) # 专家分组数
        self.topk_group = getattr(config, "topk_group", 0) # 每组选择的专家数
        if self.num_expert_group > 0 or self.topk_group > 0: # 如果使用分组top-k
            assert ( # 断言分组参数有效
                self.num_expert_group > 0
                and 0 < self.topk_group <= self.num_expert_group
            )
            self.use_grouped_topk = True # 使用分组top-k
        else: # 否则
            self.num_expert_group = self.topk_group = None # 分组参数为空
            self.use_grouped_topk = False # 不使用分组top-k

        self.num_experts = ( # 总专家数
            config.num_experts + get_global_server_args().ep_num_redundant_experts
        )

        self.gate = BailingMoEGate( # 门控模块
            config=config,
            params_dtype=self.router_dtype,
            prefix=add_prefix("gate", prefix),
        )
        self.correction_bias = ( # 修正偏置
            self.gate.expert_bias.data if self.gate.expert_bias is not None else None
        )

        if self.score_function is not None: # 如果指定了分数函数
            assert ( # 断言分数函数和修正偏置的组合有效
                self.score_function == "softmax" and self.correction_bias is None
            ) or (
                self.score_function == "sigmoid" and self.correction_bias is not None
            ), "score_function and correction_bias should be in 2 combination (softmax, None) or (sigmoid, not None)"

        self.topk = TopK( # TopK选择器
            top_k=self.top_k, # 每个token选择的专家数
            renormalize=self.norm_topk_prob, # 是否重归一化
            use_grouped_topk=self.use_grouped_topk, # 是否使用分组top-k
            num_expert_group=self.num_expert_group, # 专家分组数
            # num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=self.topk_group, # 每组选择的专家数
            correction_bias=self.correction_bias, # 修正偏置
            routed_scaling_factor=self.routed_scaling_factor, # 路由缩放因子
        )

        self.experts = get_moe_impl_class(quant_config)( # 获取MoE实现类并实例化专家
            num_experts=self.num_experts, # 专家数
            top_k=self.top_k, # top-k值
            layer_id=self.layer_id, # 层ID
            hidden_size=config.hidden_size, # 隐藏层大小
            intermediate_size=config.moe_intermediate_size, # MoE中间层大小
            quant_config=quant_config, # 量化配置
            routed_scaling_factor=self.routed_scaling_factor, # 路由缩放因子
            prefix=add_prefix("experts", prefix), # 参数前缀
        )
        # shared expert # 共享专家
        if config.num_shared_experts is not None: # 如果有共享专家
            if hasattr(config, "moe_shared_expert_intermediate_size"): # 如果指定了共享专家中间大小
                intermediate_size = config.moe_shared_expert_intermediate_size # 使用指定值
            else: # 否则
                intermediate_size = config.moe_intermediate_size # 使用MoE中间大小
            intermediate_size *= config.num_shared_experts # 乘以共享专家数
            # disable tp for shared experts when enable deepep moe # 启用DeepEP MoE时禁用共享专家的TP
            self.shared_experts = BailingMoEMLP( # 共享专家MLP
                intermediate_size=intermediate_size, # 中间层大小
                config=config, # 模型配置
                quant_config=quant_config, # 量化配置
                reduce_results=False, # 不归约结果
                prefix=add_prefix("shared_experts", prefix), # 参数前缀
                **( # 条件参数
                    dict(tp_rank=0, tp_size=1) # DeepEP模式下禁用TP
                    if get_moe_a2a_backend().is_deepep()
                    else {}
                ),
            )
        # dispatcher # 分发器
        if get_moe_a2a_backend().is_deepep(): # 如果使用DeepEP后端
            # TODO: we will support tp < ep in the future # TODO：将来会支持TP < EP
            self.ep_size = get_tensor_model_parallel_world_size() # EP大小等于TP大小

            self.deepep_dispatcher = DeepEPDispatcher( # DeepEP分发器
                group=parallel_state.get_tp_group().device_group, # TP设备组
                router_topk=self.top_k, # 路由top-k值
                permute_fusion=True, # 启用排列融合
                num_experts=self.num_experts, # 专家数
                num_local_experts=config.num_experts // self.tp_size, # 本地专家数
                hidden_size=config.hidden_size, # 隐藏层大小
                params_dtype=config.torch_dtype, # 参数数据类型
                deepep_mode=get_deepep_mode(), # DeepEP模式
                async_finish=True,  # TODO # 异步完成
                return_recv_hook=True, # 返回接收钩子
            )

    def forward( # 稀疏MoE块前向传播
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: Optional[ForwardBatch] = None, # 前向批次
        should_allreduce_fusion: bool = False, # 是否融合全归约
        use_reduce_scatter: bool = False, # 是否使用reduce-scatter
    ) -> torch.Tensor:
        if not get_moe_a2a_backend().is_deepep(): # 如果不使用DeepEP
            return self.forward_normal( # 使用普通前向传播
                hidden_states,
                should_allreduce_fusion,
                use_reduce_scatter,
            )
        else: # 否则
            return self.forward_deepep(hidden_states, forward_batch) # 使用DeepEP前向传播

    def get_moe_weights(self): # 获取MoE权重
        return [ # 返回权重列表
            x.data
            for name, x in self.experts.named_parameters() # 遍历专家参数
            if name not in ["correction_bias"] # 排除修正偏置
            and filter_moe_weight_param_global_expert( # 过滤全局专家权重参数
                name, x, self.experts.num_local_experts
            )
        ]

    def _forward_shared_experts(self, hidden_states: torch.Tensor): # 共享专家前向传播
        shared_output = None # 共享输出初始化为空
        if self.num_shared_experts > 0: # 如果有共享专家
            shared_output = self.shared_experts(hidden_states) # 计算共享专家输出
        return shared_output # 返回共享输出

    def _forward_router_experts(self, hidden_states: torch.Tensor): # 路由专家前向传播
        # router_logits: (num_tokens, n_experts) # 路由logits形状：(token数, 专家数)
        router_logits = self.gate(hidden_states) # 计算路由logits
        topk_output = self.topk(hidden_states, router_logits) # 计算top-k选择
        return self.experts(hidden_states, topk_output) # 返回专家输出

    def forward_normal_dual_stream( # 双流普通前向传播
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream() # 获取当前CUDA流
        self.alt_stream.wait_stream(current_stream) # 等待当前流完成
        shared_output = self._forward_shared_experts(hidden_states.clone()) # 在主流上计算共享专家

        with torch.cuda.stream(self.alt_stream): # 在备用流上计算
            router_output = self._forward_router_experts(hidden_states) # 路由专家计算
        current_stream.wait_stream(self.alt_stream) # 等待备用流完成

        return router_output, shared_output # 返回路由输出和共享输出

    def forward_normal( # 普通前向传播
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        should_allreduce_fusion: bool = False, # 是否融合全归约
        use_reduce_scatter: bool = False, # 是否使用reduce-scatter
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape # 获取token数和隐藏大小
        hidden_states = hidden_states.view(-1, hidden_size) # 重塑形状

        if ( # 如果可以使用双流
            self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream( # 双流计算
                hidden_states
            )
        else: # 否则顺序计算
            shared_output = self._forward_shared_experts(hidden_states) # 计算共享专家
            final_hidden_states = self._forward_router_experts(hidden_states) # 计算路由专家

        if self.num_shared_experts > 0: # 如果有共享专家
            final_hidden_states = final_hidden_states + shared_output # 合并共享专家输出

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce( # 如果TP大小>1且需要全归约
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states) # 张量并行全归约
        return final_hidden_states.view(num_tokens, hidden_size) # 返回重塑后的输出

    def forward_deepep( # DeepEP前向传播
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        shared_output = None # 共享输出初始化为空
        forward_mode = forward_batch.forward_mode # 获取前向模式
        if is_non_idle_and_non_empty(forward_mode, hidden_states): # 如果非空闲且非空
            router_logits = self.gate(hidden_states) # 计算路由logits
            if self.num_shared_experts > 0: # 如果有共享专家
                shared_output = self.shared_experts(hidden_states) # 计算共享专家输出

            topk_output = self.topk( # 计算top-k选择
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded, # 非填充token数
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new( # 专家位置调度信息
                    layer_id=self.layer_id,
                ),
            )
        else: # 否则
            topk_output = self.topk.empty_topk_output(hidden_states.device) # 空的top-k输出

        final_hidden_states = self.experts( # 专家计算
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if shared_output is not None: # 如果有共享专家输出
            final_hidden_states += shared_output # 合并共享专家输出
        return final_hidden_states # 返回最终输出


class BailingMoEAttention(nn.Module): # 百灵MoE注意力模块
    def __init__( # 注意力初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        reduce_results: bool = True, # 是否归约结果
        prefix: str = "", # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None, # 备用CUDA流
    ):
        super().__init__() # 调用父类初始化
        self.hidden_size = config.hidden_size # 隐藏层大小
        self.total_num_heads = config.num_attention_heads # 总注意力头数
        self.total_kv_heads = config.num_key_value_heads # 总KV头数
        self.dp_size = get_attention_dp_size() # 注意力DP大小
        attn_tp_rank = get_attention_tp_rank() # 注意力TP排名
        attn_tp_size = get_attention_tp_size() # 注意力TP大小

        assert self.total_num_heads % attn_tp_size == 0 # 断言总头数可被TP大小整除
        if self.total_kv_heads >= attn_tp_size: # 如果KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition # KV头数大于TP大小，因此分区
            # the KV heads across multiple tensor parallel GPUs. # 将KV头分配到多个TP GPU上
            assert self.total_kv_heads % attn_tp_size == 0 # 断言KV头数可被TP大小整除
        else: # 否则
            # Number of KV heads is less than TP size, so we replicate # KV头数小于TP大小，因此复制
            # the KV heads across multiple tensor parallel GPUs. # 将KV头复制到多个TP GPU上
            assert attn_tp_size % self.total_kv_heads == 0 # 断言TP大小可被KV头数整除
        assert self.total_num_heads >= self.total_kv_heads # 断言总头数大于等于KV头数

        self.num_heads = self.total_num_heads // attn_tp_size # 每个TP rank的头数
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads) # 头维度
        self.q_size = self.head_dim * self.num_heads # Q的大小

        self.num_kv_heads = max(1, self.total_kv_heads // attn_tp_size) # 每个TP rank的KV头数
        self.kv_size = max(1, self.num_kv_heads * self.head_dim) # KV的大小

        self.scale = self.head_dim**-0.5 # 缩放因子

        self.use_qk_norm = getattr(config, "use_qk_norm", False) # 是否使用QK归一化

        self.query_key_value = QKVParallelLinear( # QKV并行线性投影
            self.hidden_size, # 输入维度
            self.head_dim, # 头维度
            self.total_num_heads, # 总Q头数
            self.total_kv_heads, # 总KV头数
            bias=(config.use_bias or config.use_qkv_bias), # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("query_key_value", prefix), # 参数前缀
            tp_rank=attn_tp_rank, # TP排名
            tp_size=attn_tp_size, # TP大小
        )

        if self.use_qk_norm: # 如果使用QK归一化
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) # Q层归一化
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) # K层归一化

        self.dense = RowParallelLinear( # 输出投影（dense层）
            self.total_num_heads * self.head_dim, # 输入维度
            self.hidden_size, # 输出维度
            bias=config.use_bias, # 是否使用偏置
            quant_config=quant_config, # 量化配置
            reduce_results=reduce_results, # 是否归约结果
            prefix=add_prefix("dense", prefix), # 参数前缀
            tp_rank=attn_tp_rank, # TP排名
            tp_size=attn_tp_size, # TP大小
        )

        if hasattr(config, "partial_rotary_factor"): # 如果有部分旋转因子
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor) # 计算旋转维度
        elif hasattr(config, "rotary_dim"): # 如果有旋转维度
            self.rotary_dim = config.rotary_dim # 使用配置值
        else: # 否则
            self.rotary_dim = self.head_dim # 旋转维度等于头维度
        self.rotary_emb = get_rope( # 获取旋转位置编码
            self.head_dim, # 头维度
            rotary_dim=self.rotary_dim, # 旋转维度
            max_position=config.max_position_embeddings, # 最大位置
            base=config.rope_parameters["rope_theta"], # 基础频率
            rope_scaling=config.rope_parameters, # 旋转缩放
        )

        self.attn = RadixAttention( # 基数注意力
            self.num_heads, # 注意力头数
            self.head_dim, # 头维度
            self.scale, # 缩放因子
            num_kv_heads=self.num_kv_heads, # KV头数
            layer_id=layer_id, # 层ID
            prefix=add_prefix("attn", prefix), # 参数前缀
        )

        self.alt_stream = alt_stream # 备用CUDA流

    def forward( # 注意力前向传播
        self,
        positions: torch.Tensor, # 位置张量
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0: # 如果无token
            return hidden_states # 直接返回
        qkv, _ = self.query_key_value(hidden_states) # QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1) # 拆分QKV
        if self.use_qk_norm: # 如果使用QK归一化
            q, k = apply_qk_norm( # 应用QK归一化
                q=q,
                k=k,
                q_norm=self.query_layernorm,
                k_norm=self.key_layernorm,
                head_dim=self.head_dim,
                alt_stream=self.alt_stream,
            )
        can_fuse_set_kv = ( # 是否可以融合设置KV缓冲区
            self.head_dim == self.rotary_emb.rotary_dim
            and enable_fused_set_kv_buffer(forward_batch)
        )
        q, k = self.rotary_emb( # 应用旋转位置编码
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=( # 融合设置KV缓冲区参数
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=self.attn,
                    forward_batch=forward_batch,
                )
                if can_fuse_set_kv
                else None
            ),
        )
        context_layer = self.attn( # 计算注意力
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=not can_fuse_set_kv, # 是否保存KV缓存
        )
        attn_output, _ = self.dense(context_layer) # 输出投影
        return attn_output # 返回注意力输出


class BailingMoEBlock(nn.Module): # 百灵MoE块（解码器层）
    def __init__( # MoE块初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None, # 备用CUDA流
    ):
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        hidden_size = config.hidden_size # 隐藏层大小

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps) # 输入层归一化
        self.dp_size = get_attention_dp_size() # DP大小
        self.attention = BailingMoEAttention( # 注意力层
            config,
            layer_id,
            quant_config,
            reduce_results=False, # 不归约结果（由通信器处理）
            prefix=add_prefix("attention", prefix),
            alt_stream=alt_stream,
        )
        self.layer_id = layer_id # 保存层ID
        self.attn_tp_size = get_attention_tp_size() # 注意力TP大小
        self.attn_tp_rank = get_attention_tp_rank() # 注意力TP排名

        self.is_layer_sparse = self._is_layer_sparse( # 当前层是否为稀疏层
            config, layer_id=layer_id, is_nextn=False
        )
        is_previous_layer_sparse = self._is_layer_sparse( # 前一层是否为稀疏层
            config, layer_id=layer_id - 1, is_nextn=False
        )
        is_next_layer_sparse = self._is_layer_sparse( # 下一层是否为稀疏层
            config, layer_id=layer_id + 1, is_nextn=False
        )

        self.layer_scatter_modes = LayerScatterModes.init_new( # 初始化层散射模式
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1 # 是否为最后一层

        if self.is_layer_sparse: # 如果是稀疏层
            self.mlp = BailingMoESparseMoeBlock( # 使用稀疏MoE
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else: # 否则使用密集MLP
            if enable_moe_dense_fully_dp(): # 如果启用MoE密集全DP
                mlp_tp_rank, mlp_tp_size = 0, 1 # 设置TP参数为单卡
            else: # 否则
                mlp_tp_rank, mlp_tp_size = None, None # 使用默认TP参数
            self.mlp = BailingMoEMLP( # 密集MLP
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps) # 注意力后层归一化

        self.layer_communicator = LayerCommunicator( # 层通信器
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True, # 允许reduce-scatter
            is_last_layer=(self.layer_id == self.config.num_hidden_layers - 1), # 是否为最后一层
        )

    def _is_layer_sparse( # 判断层是否为稀疏层
        self, config: PretrainedConfig, layer_id: int, is_nextn: bool
    ) -> bool:
        return is_nextn or ( # 如果是nextn层或者
            config.num_experts is not None and layer_id >= config.first_k_dense_replace # 层ID超过第一个密集替换层
        )

    def forward( # MoE块前向传播
        self,
        positions: torch.Tensor, # 位置张量
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次
        residual: Optional[torch.Tensor], # 残差连接
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None, # 捕获的最后层输出
    ) -> torch.Tensor:
        hidden_states, residual = ( # 准备注意力输入
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
            )
        )

        if hidden_states.shape[0] != 0: # 如果有token
            hidden_states = self.attention( # 注意力计算
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp( # 准备MLP输入
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

        should_allreduce_fusion = ( # 是否融合全归约
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce. # 对于带填充的DP，可以使用reduce-scatter替代all-reduce
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter( # 是否使用reduce-scatter
            forward_batch
        )

        hidden_states = self.mlp( # MLP前向传播
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )

        if should_allreduce_fusion: # 如果需要全归约融合
            hidden_states._sglang_needs_allreduce_fusion = True # 标记需要全归约融合
        else: # 否则
            hidden_states, residual = self.layer_communicator.postprocess_layer( # 层后处理
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual # 返回隐藏状态和残差


class BailingMoEModel(nn.Module): # 百灵MoE模型主体

    def __init__( # 模型初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        alt_stream: Optional[torch.cuda.Stream] = None, # 备用CUDA流
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.pp_group = get_pp_group() # 获取流水线并行组
        self.config = config # 保存配置
        self.vocab_size = config.vocab_size # 词表大小
        self.embed_dim = config.hidden_size # 嵌入维度
        if self.pp_group.is_first_rank: # 如果是第一个rank
            self.word_embeddings = VocabParallelEmbedding( # 词嵌入层
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                prefix=add_prefix("word_embeddings", prefix),
                use_attn_tp_group=is_dp_attention_enabled(), # 是否使用注意力TP组
            )
        else: # 否则
            self.word_embeddings = PPMissingLayer() # 使用缺失层占位

        self.embedding_dropout = torch.nn.Dropout(config.embedding_dropout) # 嵌入dropout层

        self.layers, self.start_layer, self.end_layer = make_layers( # 创建解码器层
            config.num_hidden_layers, # 隐藏层数量
            lambda idx, prefix: BailingMoEBlock( # 解码器层工厂函数
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group, # 流水线并行rank
            pp_size=self.pp_group.world_size, # 流水线并行大小
            prefix=add_prefix("layers", prefix), # 参数前缀
        )
        if self.pp_group.is_last_rank: # 如果是最后一个rank
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps) # 最终层归一化
        else: # 否则
            self.norm = PPMissingLayer(return_tuple=True) # 使用缺失层占位

        self.layers_to_capture = [] # 需要捕获隐藏状态的层列表

    def forward( # 模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置张量
        forward_batch: ForwardBatch, # 前向批次
        input_embeds: torch.Tensor = None, # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None, # 流水线代理张量
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank: # 如果是第一个rank
            if input_embeds is None: # 如果没有输入嵌入
                hidden_states = self.word_embeddings(input_ids) # 通过词嵌入获取隐藏状态
            else: # 否则
                hidden_states = input_embeds # 直接使用输入嵌入
            residual = None # 初始化残差为空
        else: # 否则
            assert pp_proxy_tensors is not None # 断言代理张量不为空
            hidden_states = pp_proxy_tensors["hidden_states"] # 从代理获取隐藏状态
            residual = pp_proxy_tensors["residual"] # 从代理获取残差

        aux_hidden_states = [] # 辅助隐藏状态列表
        for i in range(self.start_layer, self.end_layer): # 遍历每一层
            with get_global_expert_distribution_recorder().with_current_layer(i): # 记录专家分布
                if i in self.layers_to_capture: # 如果需要捕获该层
                    aux_hidden_states.append( # 添加辅助隐藏状态
                        hidden_states if residual is None else hidden_states + residual
                    )
                layer = self.layers[i] # 获取当前层
                hidden_states, residual = layer( # 前向传播当前层
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    captured_last_layer_outputs=( # 捕获最后层输出
                        aux_hidden_states
                        if getattr(layer, "_is_layer_to_capture", False)
                        else None
                    ),
                )
        if not self.pp_group.is_last_rank: # 如果不是最后一个rank
            return PPProxyTensors( # 返回代理张量
                {
                    "hidden_states": hidden_states, # 隐藏状态
                    "residual": residual, # 残差
                }
            )
        else: # 否则
            if not forward_batch.forward_mode.is_idle(): # 如果不是空闲模式
                if residual is None: # 如果没有残差
                    hidden_states = self.norm(hidden_states) # 层归一化
                else: # 否则
                    hidden_states, _ = self.norm(hidden_states, residual) # 带残差的层归一化

        if len(aux_hidden_states) == 0: # 如果没有辅助隐藏状态
            return hidden_states # 只返回隐藏状态
        return hidden_states, aux_hidden_states # 返回隐藏状态和辅助隐藏状态


class BailingMoEForCausalLM(nn.Module): # 百灵MoE因果语言模型
    def __init__( # 因果语言模型初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.pp_group = get_pp_group() # 获取流水线并行组
        self.config = config # 保存配置
        self.quant_config = quant_config # 保存量化配置
        alt_stream = torch.cuda.Stream() if _is_cuda else None # 创建备用CUDA流

        self.model = BailingMoEModel( # 百灵MoE模型
            config,
            quant_config,
            alt_stream=alt_stream,
            prefix=add_prefix("model", ""), # 模型前缀
        )

        # tie_word_embeddings为true，复用tie_word_embeddings，反之是独立的 # 如果绑定词嵌入则复用，否则使用独立头
        if config.tie_word_embeddings: # 如果绑定词嵌入
            self.lm_head = self.model.word_embeddings # 语言模型头复用词嵌入
        else: # 否则
            # TODO something wrong with ParallelLMHead with DP attention enabled # TODO 启用DP注意力时ParallelLMHead有问题
            self.lm_head = ParallelLMHead( # 独立的语言模型头
                config.vocab_size, # 词表大小
                config.hidden_size, # 隐藏层大小
                quant_config=quant_config, # 量化配置
                prefix=add_prefix("lm_head", prefix), # 参数前缀
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head, # 是否使用注意力TP组
            )
        self.logits_processor = LogitsProcessor(config) # logits处理器

        self.capture_aux_hidden_states = False # 是否捕获辅助隐藏状态

    @property
    def start_layer(self): # 起始层属性
        return self.model.start_layer # 返回模型的起始层

    @property
    def end_layer(self): # 结束层属性
        return self.model.end_layer # 返回模型的结束层

    def get_embed_and_head(self): # 获取嵌入和语言模型头权重
        """Used by the eagle_worker.""" # 由eagle_worker使用
        return self.model.word_embeddings.weight, self.lm_head.weight # 返回嵌入权重和头权重

    def set_embed_and_head(self, embed, head): # 设置嵌入和语言模型头权重
        """Used by the eagle_worker.""" # 由eagle_worker使用
        del self.model.word_embeddings.weight # 删除旧嵌入权重
        del self.lm_head.weight # 删除旧头权重
        self.model.word_embeddings.weight = embed # 设置新嵌入权重
        self.lm_head.weight = head # 设置新头权重
        torch.cuda.empty_cache() # 清空CUDA缓存
        torch.cuda.synchronize() # 同步CUDA

    @torch.no_grad() # 禁用梯度计算
    def forward( # 因果语言模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置张量
        forward_batch: ForwardBatch, # 前向批次
        input_embeds: torch.Tensor = None, # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None, # 流水线代理张量
    ) -> torch.Tensor:
        hidden_states = self.model( # 通过模型获取隐藏状态
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None # 辅助隐藏状态初始化为空
        if self.capture_aux_hidden_states: # 如果需要捕获辅助隐藏状态
            hidden_states, aux_hidden_states = hidden_states # 拆分隐藏状态

        if self.pp_group.is_last_rank: # 如果是最后一个rank
            return self.logits_processor( # 通过logits处理器返回
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else: # 否则
            return hidden_states # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False): # 加载权重
        if is_nextn: # 如果是nextn模式
            if hasattr(self.config, "num_nextn_predict_layers"): # 如果配置中有nextn预测层数
                num_nextn_layers = self.config.num_nextn_predict_layers # 获取nextn层数
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported" # 断言仅支持1个nextn层
                # compatible with old design # 兼容旧设计
                nextn_layer_id = ( # nextn层ID
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else: # 否则
                raise ValueError("num_nextn_predict_layers is not in the config") # 抛出配置错误

        stacked_params_mapping = [ # 堆叠参数映射
            # (param_name, shard_name, shard_id) # （参数名，分片名，分片ID）
            ("gate_up_proj", "gate_proj", 0), # gate投影映射
            ("gate_up_proj", "up_proj", 1), # up投影映射
        ]

        if is_nextn: # 如果是nextn模式
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}" # nextn层前缀
            nextn_spec_weight_names = [ # nextn特有权重名称
                "final_layernorm", # 最终层归一化
                "eh_proj", # eh投影
                "enorm", # enorm
                "hnorm", # hnorm
            ]
        # Params for weights, fp8 weight scales, fp8 activation scales # 权重、FP8权重缩放、FP8激活缩放的参数
        # (param_name, weight_name, expert_id, shard_id) # （参数名，权重名，专家ID，分片ID）
        expert_params_mapping = FusedMoE.make_expert_params_mapping( # 创建专家参数映射
            ckpt_gate_proj_name="gate_proj", # 检查点gate投影名
            ckpt_down_proj_name="down_proj", # 检查点down投影名
            ckpt_up_proj_name="up_proj", # 检查点up投影名
            num_experts=self.config.num_experts, # 专家数
        )

        params_dict = dict(self.named_parameters()) # 获取参数字典
        for name, loaded_weight in weights: # 遍历所有权重
            if ( # 跳过不需要的权重
                ("v_head" in name) # v_head
                or ("inv_freq" in name) # 逆频率
                or (self.config.tie_word_embeddings and "lm_head" in name) # 绑定嵌入时的lm_head
            ):
                continue # 跳过

            if ( # 如果需要归一化头权重
                hasattr(self.config, "norm_head")
                and self.config.norm_head
                and "lm_head.weight" in name
            ):
                import torch.nn.functional as F # 导入函数式模块

                loaded_weight = F.normalize(loaded_weight, dim=0, p=2, eps=1e-7) # L2归一化

            if is_nextn: # 如果是nextn模式
                if not name.startswith(nextn_layer_prefix): # 如果名称不以nextn层前缀开头
                    continue # 跳过

                # Use shared head and embed weights from target model # 使用目标模型的共享头和嵌入权重
                if "shared_head.head" in name or "embed_tokens" in name: # 共享头或嵌入token
                    continue # 跳过

                is_decoder = True # 标记为解码器权重
                # For nextn specific weights # nextn特有权重
                for weight_name in nextn_spec_weight_names: # 遍历nextn特有权重名
                    if weight_name in name: # 如果名称中包含特有权重名
                        name = name.replace(nextn_layer_prefix, "model") # 替换前缀
                        is_decoder = False # 标记为非解码器权重
                        break # 跳出循环
                # For decoder layer weights # 解码器层权重
                if is_decoder: # 如果是解码器权重
                    name = name.replace(nextn_layer_prefix, "model.decoder") # 替换为解码器前缀

            for param_name, weight_name, shard_id in stacked_params_mapping: # 遍历堆叠参数映射
                if weight_name not in name: # 如果权重名不在参数名中
                    continue # 跳过
                # We have mlp.experts[0].gate_proj in the checkpoint. # 检查点中有mlp.experts[0].gate_proj
                # Since we handle the experts below in expert_params_mapping, # 因为在下面的expert_params_mapping中处理专家
                # we need to skip here BEFORE we update the name, otherwise # 需要在更新名称之前跳过，否则
                # name will be updated to mlp.experts[0].gate_up_proj, which # 名称会被更新为mlp.experts[0].gate_up_proj
                # will then be updated below in expert_params_mapping # 然后在下面的expert_params_mapping中更新
                # for mlp.experts[0].gate_gate_up_proj, which breaks load. # 为mlp.experts[0].gate_gate_up_proj，导致加载失败
                if "mlp.experts" in name: # 如果名称包含专家
                    continue # 跳过
                name = name.replace(weight_name, param_name) # 替换权重名为参数名
                # Skip loading extra bias for GPTQ models. # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict: # 如果是偏置且不在参数字典中
                    continue # 跳过
                if name not in params_dict: # 如果参数名不在参数字典中
                    continue # 跳过

                param = params_dict[name] # 获取参数
                weight_loader = param.weight_loader # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id) # 加载权重
                break # 跳出循环
            else: # 如果没有匹配堆叠映射
                for mapping in expert_params_mapping: # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping # 解包映射
                    if weight_name not in name: # 如果权重名不在参数名中
                        continue # 跳过
                    name = name.replace(weight_name, param_name) # 替换权重名
                    if name not in params_dict: # 如果参数名不在参数字典中
                        continue # 跳过
                    param = params_dict[name] # 获取参数
                    weight_loader = param.weight_loader # 获取权重加载器
                    weight_loader( # 加载权重
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break # 跳出循环
                else: # 如果没有匹配专家映射
                    # Skip loading extra bias for GPTQ models. # 跳过GPTQ模型的额外偏置加载
                    if name.endswith(".bias") and name not in params_dict: # 如果是偏置且不在参数字典中
                        continue # 跳过
                    if name not in params_dict: # 如果参数名不在参数字典中
                        continue # 跳过

                    param = params_dict[name] # 获取参数
                    weight_loader = getattr( # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight) # 加载权重

        if not is_nextn: # 如果不是nextn模式
            self.routed_experts_weights_of_layer = { # 保存每层的路由专家权重
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers) # 遍历所有层
                if not isinstance(layer, PPMissingLayer) # 排除缺失层
                and isinstance(layer.mlp, BailingMoESparseMoeBlock) # 仅包含稀疏MoE层
            }

    @classmethod
    def get_model_config_for_expert_location(cls, config): # 获取专家位置的模型配置
        num_groups = getattr(config, "n_group", 0) # 获取分组数
        return ModelConfigForExpertLocation( # 返回专家位置模型配置
            num_layers=config.num_hidden_layers, # 隐藏层数
            num_logical_experts=config.num_experts, # 逻辑专家数
            num_groups=None if num_groups == 0 else num_groups, # 分组数
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None): # 设置Eagle3捕获层
        if not self.pp_group.is_last_rank: # 如果不是最后一个rank
            return # 直接返回

        self.capture_aux_hidden_states = True # 启用辅助隐藏状态捕获
        if layer_ids is None: # 如果未指定层ID
            num_layers = self.config.num_hidden_layers # 获取总层数
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3] # 默认捕获第2、中间、倒数第3层
        else: # 否则
            # Add +1 because in SGLang, for the i-th layer, the auxiliary hidden state # 加1是因为在SGLang中，第i层的辅助隐藏状态
            # corresponds to the output of layer (i - 1). # 对应于第(i-1)层的输出
            self.model.layers_to_capture = [val + 1 for val in layer_ids] # 层ID加1


class BailingMoeForCausalLM(BailingMoEForCausalLM): # 百灵MoE因果语言模型（别名）
    pass # 无额外实现


class BailingMoeV2ForCausalLM(BailingMoEForCausalLM): # 百灵MoE V2因果语言模型（别名）
    pass # 无额外实现


EntryClass = [BailingMoEForCausalLM, BailingMoeForCausalLM, BailingMoeV2ForCausalLM] # 入口类列表
