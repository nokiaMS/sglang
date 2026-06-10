# Step3.5 模型实现
# 本文件实现了 Step3.5 因果语言模型，包含MLP、混合专家(MoE)MLP、注意力层、解码器层和完整模型。
# 支持张量并行(TP)、专家并行(EP)、流水线并行(PP)以及DeepEP后端。

from typing import Any, Dict, Iterable, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch核心库
import torch.nn.functional as F  # 导入神经网络函数式接口
from torch import nn  # 导入神经网络模块

from sglang.srt.distributed import (  # 导入分布式通信相关函数
    get_moe_expert_parallel_world_size,  # 获取MoE专家并行世界大小
    get_pp_group,  # 获取流水线并行组
    get_tensor_model_parallel_rank,  # 获取张量并行秩
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
    tensor_model_parallel_all_reduce,  # 张量并行全归约
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入专家分布记录器
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation  # 导入专家位置模型配置
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo  # 导入专家位置调度信息
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU和乘法激活函数
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes  # 导入层通信器和散射模式
from sglang.srt.layers.dp_attention import (  # 导入数据并行注意力相关函数
    get_attention_tp_rank,  # 获取注意力TP秩
    get_attention_tp_size,  # 获取注意力TP大小
    is_dp_attention_enabled,  # 是否启用DP注意力
)
from sglang.srt.layers.layernorm import GemmaRMSNorm  # 导入Gemma RMS归一化层
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,  # 列并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.moe import (  # 导入MoE相关函数
    get_moe_a2a_backend,  # 获取MoE全互连后端
    should_skip_post_experts_all_reduce,  # 是否跳过专家后全归约
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class  # 导入MoE实现类获取函数
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合MoE层
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK  # 导入Top-K选择模块
from sglang.srt.layers.moe.utils import (  # 导入MoE工具函数
    RoutingMethodType,  # 路由方法类型
    filter_moe_weight_param_global_expert,  # 过滤MoE权重参数
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.layers.utils import PPMissingLayer  # 导入流水线并行缺失层
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息和PP代理张量
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import add_prefix, is_cuda, is_non_idle_and_non_empty, make_layers  # 导入工具函数

Step3p5Config = None  # Step3.5配置，后续会被设置

_is_cuda = is_cuda()  # 是否为CUDA设备


class Step3p5MLP(nn.Module):
    """Step3.5模型的MLP模块，支持SwiGLU限幅"""

    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        swiglu_limit: Optional[float] = None,  # SwiGLU限幅值
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        tp_size: Optional[int] = None,  # 张量并行大小
        tp_rank: Optional[int] = None,  # 张量并行秩
        reduce_results: bool = True,  # 是否归约结果
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.intermediate_size = intermediate_size  # 保存中间层大小
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控和上投影合并层
            hidden_size,  # 输入大小
            [intermediate_size] * 2,  # 输出大小列表（gate和up各一个）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
            tp_size=tp_size,  # 张量并行大小
            tp_rank=tp_rank,  # 张量并行秩
        )
        self.down_proj = RowParallelLinear(  # 下投影层
            intermediate_size,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 参数前缀
            tp_size=tp_size,  # 张量并行大小
            tp_rank=tp_rank,  # 张量并行秩
            reduce_results=reduce_results,  # 是否归约结果
        )
        self.act_fn = SiluAndMul()  # SiLU和乘法激活函数
        self.limit = swiglu_limit  # SwiGLU限幅值

    def forward(self, x):
        """MLP前向传播，支持SwiGLU限幅模式"""
        if self.limit is not None:  # 如果有限幅值
            gate_up, _ = self.gate_up_proj(x)  # 通过门控上投影
            gate, up = gate_up.chunk(2, dim=-1)  # 分离门控和上投影
            gate = F.silu(gate)  # 对门控应用SiLU
            gate = gate.clamp(min=None, max=self.limit)  # 门控限幅
            up = up.clamp(min=-self.limit, max=self.limit)  # 上投影限幅
            output, _ = self.down_proj(gate * up)  # 下投影
        else:  # 无限幅
            gate_up, _ = self.gate_up_proj(x)  # 通过门控上投影
            x = self.act_fn(gate_up)  # 应用SiLU和乘法
            output, _ = self.down_proj(x)  # 下投影
        return output  # 返回输出


class Step3p5MoEMLP(nn.Module):
    """Step3.5模型的混合专家(MLP)模块，支持DeepEP和普通模式"""

    def __init__(
        self,
        config,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        self.layer_id = layer_id  # 保存层ID

        self.need_fp32_gate = config.need_fp32_gate  # 是否需要FP32门控
        self.routed_scaling_factor = config.moe_router_scaling_factor  # 路由缩放因子
        self.use_moe_router_bias = config.use_moe_router_bias  # 是否使用MoE路由偏置
        if self.use_moe_router_bias:  # 如果使用路由偏置
            self.router_bias = nn.Parameter(  # 路由偏置参数
                torch.zeros(config.moe_num_experts, dtype=torch.float32),  # 全零初始化
                requires_grad=False,  # 不需要梯度
            )

        if self.tp_size > config.moe_num_experts:  # 如果TP大小大于专家数
            raise ValueError(  # 抛出异常
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        self.limit = config.swiglu_limits[layer_id]  # 获取当前层的SwiGLU限幅
        self.limit = self.limit if self.limit > 0 else None  # 非正值设为None

        self.topk = TopK(  # Top-K选择模块
            top_k=config.moe_top_k,  # Top-K值
            renormalize=True,  # 重归一化
            use_grouped_topk=False,  # 不使用分组Top-K
            scoring_func="sigmoid",  # 评分函数使用sigmoid
            correction_bias=self.router_bias,  # 校正偏置
            apply_routed_scaling_factor_on_output=False,  # 不在输出上应用路由缩放因子
            layer_id=layer_id,  # 层ID
        )

        self.experts = get_moe_impl_class(quant_config)(  # 专家实现
            num_experts=config.moe_num_experts  # 专家数量
            + get_global_server_args().ep_num_redundant_experts,  # 加上冗余专家数
            top_k=config.moe_top_k,  # Top-K值
            layer_id=layer_id,  # 层ID
            hidden_size=config.hidden_size,  # 隐藏大小
            intermediate_size=config.moe_intermediate_size,  # 中间层大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("experts", prefix),  # 参数前缀
            routing_method_type=RoutingMethodType.Renormalize,  # 路由方法类型
            gemm1_clamp_limit=self.limit,  # GEMM1限幅值
        )

        self.gate = ReplicatedLinear(  # 路由门控
            config.hidden_size,  # 输入大小
            config.moe_num_experts,  # 输出大小（专家数）
            bias=False,  # 不使用偏置
            quant_config=None,  # 不量化
            prefix=add_prefix("gate", prefix),  # 参数前缀
        )

        if get_moe_a2a_backend().is_deepep():  # 如果使用DeepEP后端
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()  # 获取EP大小
            self.moe_num_experts = (  # MoE专家数
                config.moe_num_experts  # 配置中的专家数
                + get_global_server_args().ep_num_redundant_experts  # 加上冗余专家数
            )
            self.top_k = config.moe_top_k  # Top-K值

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: Optional[ForwardBatch] = None,  # 前向批次
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        """MoE MLP前向传播，根据后端选择普通或DeepEP模式"""
        if (  # 如果不是DeepEP和Ascend FuseEP后端
            not get_moe_a2a_backend().is_deepep()
            and not get_moe_a2a_backend().is_ascend_fuseep()
        ):
            return self.forward_normal(  # 使用普通前向传播
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )
        else:
            return self.forward_deepep(hidden_states, forward_batch)  # 使用DeepEP前向传播

    def get_moe_weights(self):
        """获取MoE权重列表"""
        return [
            x.data  # 获取参数数据
            for name, x in self.experts.named_parameters()  # 遍历专家参数
            if name not in ["correction_bias"]  # 排除校正偏置
            and filter_moe_weight_param_global_expert(  # 过滤全局专家权重
                name, x, self.experts.num_local_experts
            )
        ]

    def forward_normal(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        """普通MoE前向传播（非DeepEP模式）"""
        num_tokens, hidden_dim = hidden_states.shape  # 获取token数和隐藏维度
        hidden_states = hidden_states.view(-1, hidden_dim)  # 重塑为2D
        # router_logits: (num_tokens, n_experts)
        if self.need_fp32_gate:  # 如果需要FP32门控
            router_logits = torch.matmul(  # FP32矩阵乘法
                hidden_states.to(torch.float32), self.gate.weight.t().to(torch.float32)
            )
        else:
            # router_logits: (batch * sequence_length, n_experts)
            router_logits, _ = self.gate(hidden_states)  # 通过门控
        topk_output = self.topk(hidden_states, router_logits)  # Top-K选择
        if hasattr(topk_output, "to_standard"):  # 如果有转换为标准输出的方法
            topk_output = topk_output.to_standard(layer_id=self.layer_id)  # 转换
        if self.routed_scaling_factor != 1.0:  # 如果路由缩放因子不为1
            topk_output = StandardTopKOutput(  # 创建标准Top-K输出
                topk_weights=topk_output.topk_weights * self.routed_scaling_factor,  # 缩放权重
                topk_ids=topk_output.topk_ids,  # Top-K ID
                router_logits=topk_output.router_logits,  # 路由logits
            )
        final_hidden_states = self.experts(hidden_states, topk_output)  # 通过专家
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(  # 如果需要全归约
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 全归约

        return final_hidden_states.view(num_tokens, hidden_dim)  # 返回重塑后的输出

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """DeepEP模式的MoE前向传播"""
        if hidden_states.shape[0] > 0:  # 如果有token
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)  # 通过门控
            topk_output = self.topk(  # Top-K选择
                hidden_states,  # 隐藏状态
                router_logits,  # 路由logits
                num_token_non_padded=forward_batch.num_token_non_padded,  # 非填充token数
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(  # 专家位置调度信息
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)  # 空Top-K输出
        final_hidden_states = self.experts(  # 通过专家
            hidden_states=hidden_states,  # 隐藏状态
            topk_output=topk_output,  # Top-K输出
        )
        return final_hidden_states  # 返回输出

    def op_gate(self, state):
        """执行门控操作，计算路由logits"""
        if is_non_idle_and_non_empty(  # 如果非空闲且非空
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits, _ = self.gate(state.hidden_states_mlp_input)  # 通过门控
        else:
            state.router_logits = None  # 空闲时设为None

    def op_select_experts(self, state):
        """执行专家选择操作"""
        router_logits = state.pop("router_logits")  # 弹出路由logits
        hidden_states = state.hidden_states_mlp_input  # 获取MLP输入
        if router_logits is not None:  # 如果有路由logits
            with get_global_expert_distribution_recorder().with_current_layer(  # 记录专家分布
                self.layer_id
            ):
                state.topk_output = self.topk(  # Top-K选择
                    hidden_states=hidden_states,  # 隐藏状态
                    router_logits=router_logits,  # 路由logits
                    num_token_non_padded=state.forward_batch.num_token_non_padded,  # 非填充token数
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(  # 调度信息
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_output = self.topk.empty_topk_output(hidden_states.device)  # 空Top-K输出

    def op_dispatch_a(self, state):
        """执行专家调度的第一阶段（发送）"""
        if self.ep_size > 1:  # 如果有专家并行
            self.experts.dispatcher.dispatch_a(  # 发送隐藏状态和Top-K输出
                hidden_states=state.pop("hidden_states_mlp_input"),  # 弹出MLP输入
                topk_output=state.pop("topk_output"),  # 弹出Top-K输出
                tbo_subbatch_index=state.get("tbo_subbatch_index"),  # TBO子批次索引
            )

    def op_dispatch_b(self, state):
        """执行专家调度的第二阶段（接收）"""
        if self.ep_size > 1:  # 如果有专家并行
            with get_global_expert_distribution_recorder().with_current_layer(  # 记录专家分布
                self.layer_id
            ):
                state.dispatch_output = self.experts.dispatcher.dispatch_b(  # 接收调度输出
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),  # TBO子批次索引
                )

    def op_experts(self, state):
        """执行专家计算"""
        state.combine_input = self.experts.run_moe_core(  # 运行MoE核心计算
            dispatch_output=state.dispatch_output,  # 调度输出
        )

    def op_combine_a(self, state):
        """执行专家结果合并的第一阶段（发送）"""
        if self.ep_size > 1:  # 如果有专家并行
            self.experts.dispatcher.combine_a(  # 发送合并输入
                combine_input=state.pop("combine_input"),  # 弹出合并输入
                tbo_subbatch_index=state.get("tbo_subbatch_index"),  # TBO子批次索引
            )
            state.pop("dispatch_output")  # 弹出调度输出

    def op_combine_b(self, state):
        """执行专家结果合并的第二阶段（接收）"""
        if self.ep_size > 1:  # 如果有专家并行
            state.hidden_states_after_combine = self.experts.dispatcher.combine_b(  # 接收合并结果
                tbo_subbatch_index=state.get("tbo_subbatch_index"),  # TBO子批次索引
            )

    def op_output(self, state):
        """设置MLP输出为合并后的隐藏状态"""
        state.hidden_states_mlp_output = state.pop("hidden_states_after_combine")  # 弹出合并结果设为输出


class Step3p5Attention(nn.Module):
    """Step3.5模型的注意力模块，支持QK归一化和头部级注意力门控"""

    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        layer_id: int = 0,  # 层ID
        rope_theta: float = 1000000,  # RoPE theta值
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放配置
        head_dim: Optional[int] = None,  # 头维度
        max_position_embeddings: int = 32768,  # 最大位置编码长度
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        rms_norm_eps: float = None,  # RMS归一化eps
        partial_rotary_factor: float = 1.0,  # 部分旋转因子
        use_head_wise_attn_gate: bool = False,  # 是否使用头部级注意力门控
        sliding_window_size: int = -1,  # if is -1 ,normal attention,else ,window attention # 滑动窗口大小
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        self.total_num_heads = num_heads  # 总注意力头数
        attn_tp_rank = get_attention_tp_rank()  # 获取注意力TP秩
        attn_tp_size = get_attention_tp_size()  # 获取注意力TP大小

        assert self.total_num_heads % attn_tp_size == 0  # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // attn_tp_size  # 每个TP的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= attn_tp_size:  # KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0  # 断言KV头数可被TP大小整除
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)  # 每个TP的KV头数
        self.head_dim = head_dim or hidden_size // self.total_num_heads  # 头维度
        self.q_size = self.num_heads * self.head_dim  # Q大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV大小
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.rope_theta = rope_theta  # 保存RoPE theta
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置编码长度
        self.tp_rank = get_tensor_model_parallel_rank()  # 获取TP秩
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)  # Q归一化
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)  # K归一化

        self.qkv_proj = QKVParallelLinear(  # QKV并行投影
            hidden_size,  # 输入大小
            self.head_dim,  # 头维度
            self.total_num_heads,  # 总头数
            self.total_num_kv_heads,  # 总KV头数
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            tp_rank=attn_tp_rank,  # 注意力TP秩
            tp_size=attn_tp_size,  # 注意力TP大小
            prefix=add_prefix("qkv_proj", prefix),  # 参数前缀
        )
        self.o_proj = RowParallelLinear(  # 输出投影
            self.total_num_heads * self.head_dim,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            tp_rank=attn_tp_rank,  # 注意力TP秩
            tp_size=attn_tp_size,  # 注意力TP大小
            reduce_results=False,  # 不自动归约
            prefix=add_prefix("o_proj", prefix),  # 参数前缀
        )

        self.use_head_wise_attn_gate = use_head_wise_attn_gate  # 是否使用头部级门控
        if self.use_head_wise_attn_gate:  # 如果使用头部级门控
            self.g_proj = ColumnParallelLinear(  # 门控投影
                hidden_size,  # 输入大小
                self.total_num_heads,  # 输出大小（头数）
                bias=False,  # 不使用偏置
                tp_rank=attn_tp_rank,  # 注意力TP秩
                tp_size=attn_tp_size,  # 注意力TP大小
                prefix=add_prefix("g_proj", prefix),  # 参数前缀
            )

        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置
            base=rope_theta,  # 基数
            rope_scaling=rope_scaling,  # 缩放配置
            partial_rotary_factor=partial_rotary_factor,  # 部分旋转因子
            is_neox_style=True,  # 使用Neox风格
        )
        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            sliding_window_size=sliding_window_size,  # if is -1 ,normal attention,else ,window attention # 滑动窗口大小
            layer_id=layer_id,  # 层ID
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )
        self.alt_stream = alt_stream  # 备用CUDA流

    def forward_prepare_native(self, positions, hidden_states):
        """准备原生注意力计算的Q、K、V"""
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分离Q、K、V
        q_shape, k_shape = q.shape, k.shape  # 保存原始形状
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q_shape)  # Q归一化
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k_shape)  # K归一化
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        return q, k, v  # 返回Q、K、V

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """注意力前向传播：QKV投影 -> QK归一化 -> RoPE -> 注意力 -> 可选门控 -> 输出投影"""
        q, k, v = self.forward_prepare_native(  # 准备Q、K、V
            positions=positions,  # 位置
            hidden_states=hidden_states,  # 隐藏状态
        )
        if self.use_head_wise_attn_gate:  # 如果使用头部级门控
            gate_states, _ = self.g_proj(hidden_states)  # 计算门控状态
        attn_output = self.attn(q, k, v, forward_batch)  # 通过注意力层
        if self.use_head_wise_attn_gate:  # 如果使用头部级门控
            output = (  # 应用门控
                attn_output.view(  # 重塑注意力输出
                    attn_output.shape[0],
                    self.num_heads,  # TODO: check if this is correct # 头数
                    self.head_dim,  # 头维度
                )
                * gate_states.unsqueeze(-1).sigmoid()  # 门控sigmoid
            )
            attn_output = output.view(*attn_output.shape)  # 恢复原始形状
        output, _ = self.o_proj(attn_output)  # 通过输出投影
        return output  # 返回输出


class Step3p5DecoderLayer(nn.Module):
    """Step3.5解码器层，包含自注意力和MLP/MoE"""

    def __init__(
        self,
        config: Step3p5Config,  # 模型配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 保存隐藏层大小
        layer_types = config.layer_types  # 层类型列表
        yarn_only_types = config.yarn_only_types  # 仅YaRN类型
        if layer_types[layer_id] not in yarn_only_types:  # 如果不是YaRN类型
            rope_scaling = None  # 不使用RoPE缩放
        else:
            rope_scaling = config.rope_scaling  # 使用RoPE缩放
        rope_theta = config.rope_theta  # RoPE theta列表
        max_position_embeddings = config.max_position_embeddings  # 最大位置编码长度
        head_dim = config.head_dim  # 头维度
        moe_layers_set = {int(x) for x in config.moe_layers_enum.split(",")}  # MoE层集合
        self.num_attention_heads = config.num_attention_heads  # 注意力头数
        self.num_key_value_heads = config.num_attention_groups  # KV头数（注意力组数）
        self.is_moe_layer = layer_id in moe_layers_set  # 当前层是否为MoE层
        self.is_previous_layer_sparse = (layer_id - 1) in moe_layers_set  # 前一层是否为MoE层
        self.is_next_layer_sparse = (layer_id + 1) in moe_layers_set  # 下一层是否为MoE层
        num_hidden_layers = config.num_hidden_layers  # 总隐藏层数

        if (  # 检查共享SwiGLU限幅
            config.swiglu_limits_shared
            and config.swiglu_limits_shared[layer_id] is not None
            and config.swiglu_limits_shared[layer_id] != 0
        ):
            swiglu_limit_shared = config.swiglu_limits_shared[layer_id]  # 使用共享限幅
        else:
            swiglu_limit_shared = None  # 不使用限幅

        self.sliding_window = -1  # 滑动窗口大小，-1表示不使用

        enable_sliding_window = layer_types[layer_id] == "sliding_attention"  # 是否启用滑动窗口

        if enable_sliding_window:  # 如果启用滑动窗口
            self.sliding_window = config.sliding_window  # 设置滑动窗口大小
            self.num_attention_heads = config.attention_other_setting[  # 更新注意力头数
                "num_attention_heads"
            ]
            self.num_key_value_heads = config.attention_other_setting[  # 更新KV头数
                "num_attention_groups"
            ]

        self.self_attn = Step3p5Attention(  # 自注意力层
            hidden_size=self.hidden_size,  # 隐藏大小
            num_heads=self.num_attention_heads,  # 头数
            num_kv_heads=self.num_key_value_heads,  # KV头数
            layer_id=(  # 层ID
                layer_id  # 使用原始ID
                if layer_id < num_hidden_layers  # 如果小于总层数
                else layer_id - num_hidden_layers  # 否则减去总层数
            ),
            rope_theta=rope_theta[layer_id],  # RoPE theta
            rope_scaling=rope_scaling,  # RoPE缩放
            head_dim=head_dim,  # 头维度
            max_position_embeddings=max_position_embeddings,  # 最大位置编码
            sliding_window_size=self.sliding_window,  # 滑动窗口大小
            partial_rotary_factor=config.partial_rotary_factors[layer_id],  # 部分旋转因子
            quant_config=quant_config,  # 量化配置
            rms_norm_eps=config.rms_norm_eps,  # RMS归一化eps
            use_head_wise_attn_gate=config.use_head_wise_attn_gate,  # 头部级门控
            prefix=add_prefix("self_attn", prefix),  # 参数前缀
            alt_stream=alt_stream,  # 备用CUDA流
        )
        self.use_moe = False  # 是否使用MoE
        if self.is_moe_layer:  # 如果是MoE层
            self.moe = Step3p5MoEMLP(  # MoE MLP
                config,  # 配置
                layer_id=layer_id,  # 层ID
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("mlp", prefix),  # 参数前缀
            )
            # reduce_results=False: share_expert output stays unreduced and is
            # combined with the (also unreduced) MoE output, then a single
            # all-reduce covers both — saving one full-TP all-reduce per layer.
            self.share_expert = Step3p5MLP(  # 共享专家
                hidden_size=self.hidden_size,  # 隐藏大小
                intermediate_size=config.share_expert_dim,  # 共享专家维度
                swiglu_limit=swiglu_limit_shared,  # SwiGLU限幅
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("share_expert", prefix),  # 参数前缀
                reduce_results=False,  # 不归约结果
            )
            self.use_moe = True  # 标记使用MoE
        else:
            self.mlp = Step3p5MLP(  # 普通MLP
                hidden_size=self.hidden_size,  # 隐藏大小
                intermediate_size=config.intermediate_size,  # 中间层大小
                swiglu_limit=swiglu_limit_shared,  # SwiGLU限幅
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("mlp", prefix),  # 参数前缀
            )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = GemmaRMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(  # 层散射模式
            layer_id=layer_id,  # 层ID
            num_layers=(  # 层数
                config.num_hidden_layers if layer_id < config.num_hidden_layers else 1  # 1 is for mtp # 1用于MTP
            ),
            is_layer_sparse=self.is_moe_layer,  # 当前层是否稀疏
            is_previous_layer_sparse=self.is_previous_layer_sparse,  # 前一层是否稀疏
            is_next_layer_sparse=self.is_next_layer_sparse,  # 下一层是否稀疏
        )
        self.layer_communicator = LayerCommunicator(  # 层通信器
            layer_scatter_modes=self.layer_scatter_modes,  # 散射模式
            input_layernorm=self.input_layernorm,  # 输入归一化
            post_attention_layernorm=self.post_attention_layernorm,  # 注意力后归一化
            allow_reduce_scatter=True,  # 允许reduce-scatter
            is_last_layer=(layer_id == config.num_hidden_layers - 1),  # 是否为最后一层
        )

        self.layer_id = layer_id  # 保存层ID

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差
        post_residual_addition: Optional[torch.Tensor] = None,  # 后残差加法
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码器层前向传播：自注意力 + MLP/MoE"""
        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(  # 准备注意力
            hidden_states,  # 隐藏状态
            residual,  # 残差
            forward_batch,  # 前向批次
            post_residual_addition=post_residual_addition,  # 后残差加法
        )
        if hidden_states.shape[0] != 0:  # 如果有token
            hidden_states = self.self_attn(  # 通过自注意力
                positions=positions,  # 位置
                hidden_states=hidden_states,  # 隐藏状态
                forward_batch=forward_batch,  # 前向批次
            )
        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(  # 准备MLP
            hidden_states,  # 隐藏状态
            residual,  # 残差
            forward_batch,  # 前向批次
        )

        should_allreduce_fusion = (  # 是否融合全归约
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(  # 是否使用reduce-scatter
            forward_batch
        )

        if self.use_moe:  # 如果使用MoE
            # Both share_expert and MoE return unreduced (TP-partial) outputs.
            # Combine them first, then do a single all-reduce — saving one
            # full-TP all-reduce per layer.
            share_output = self.share_expert(hidden_states)  # 共享专家输出
            moe_output = self.moe(  # MoE输出
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次
                should_allreduce_fusion=True,  # 融合全归约
                use_reduce_scatter=use_reduce_scatter,  # reduce-scatter
            )
            hidden_states = moe_output + share_output  # 合并MoE和共享专家输出
            if not should_allreduce_fusion and not use_reduce_scatter:  # 如果不需要融合
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)  # 全归约
        else:  # 不使用MoE
            hidden_states = self.mlp(hidden_states)  # 通过普通MLP
            # Dense MLP uses reduce_results=True, so the output is already
            # all-reduced.  Do NOT set the fusion flag — otherwise the next
            # layer would all-reduce again, multiplying values by world_size.
            should_allreduce_fusion = False  # 不融合全归约

        if should_allreduce_fusion:  # 如果融合全归约
            hidden_states._sglang_needs_allreduce_fusion = True  # 标记需要融合
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(  # 后处理层
                hidden_states, residual, forward_batch
            )
        return hidden_states, residual  # 返回隐藏状态和残差


class Step3p5Model(nn.Module):
    """Step3.5模型主体，包含嵌入层、解码器层和归一化层"""

    def __init__(
        self,
        config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.vocab_size = config.vocab_size  # 保存词表大小
        self.pp_group = get_pp_group()  # 获取流水线并行组

        alt_stream = torch.cuda.Stream() if _is_cuda else None  # 创建备用CUDA流

        if self.pp_group.is_first_rank:  # 如果是第一个PP秩
            self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏大小
                quant_config=quant_config,  # 量化配置
                enable_tp=not is_dp_attention_enabled(),  # 是否启用TP
                prefix=add_prefix("embed_tokens", prefix),  # 参数前缀
                params_dtype=(  # 参数数据类型
                    torch.float32  # FP32
                    if get_global_server_args().rl_on_policy_target is not None  # 如果启用RL
                    else None  # 默认
                ),
            )
        else:
            self.embed_tokens = PPMissingLayer()  # 非首秩使用缺失层

        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,  # 层数
            # 1,
            lambda idx, prefix: Step3p5DecoderLayer(  # 每层创建函数
                layer_id=idx,  # 层ID
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                prefix=prefix,  # 参数前缀
                alt_stream=alt_stream,  # 备用CUDA流
            ),
            pp_rank=self.pp_group.rank_in_group,  # PP秩
            pp_size=self.pp_group.world_size,  # PP大小
            prefix=add_prefix("layers", prefix),  # 参数前缀
        )
        if self.pp_group.is_last_rank:  # 如果是最后一个PP秩
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 归一化层
        else:
            self.norm = PPMissingLayer(return_tuple=True)  # 非末秩使用缺失层

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取输入嵌入，支持嵌入缩放"""
        if hasattr(self.config, "scale_emb"):  # 如果配置中有嵌入缩放
            return self.get_input_embeddings()(input_ids) * self.config.scale_emb  # 缩放嵌入
        else:
            return self.get_input_embeddings()(input_ids)  # 不缩放

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.embed_tokens  # 返回嵌入层

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量
    ) -> Union[torch.Tensor, PPProxyTensors]:
        """模型主体前向传播：嵌入 -> 解码器层 -> 归一化"""
        if self.pp_group.is_first_rank:  # 如果是第一个PP秩
            if input_embeds is None:  # 如果没有输入嵌入
                hidden_states = self.embed_tokens(input_ids)  # 通过嵌入层
            else:
                hidden_states = input_embeds  # 使用输入嵌入
            residual = None  # 初始化残差为None
        else:
            assert pp_proxy_tensors is not None  # 断言有代理张量
            hidden_states = pp_proxy_tensors["hidden_states"]  # 从代理张量获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 从代理张量获取残差

        for i in range(self.start_layer, self.end_layer):  # 遍历解码器层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 通过当前层
                positions,  # 位置
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次
                residual,  # 残差
            )
            # break
        if not self.pp_group.is_last_rank:  # 如果不是最后一个PP秩
            return PPProxyTensors(  # 返回代理张量
                {
                    "hidden_states": hidden_states,  # 隐藏状态
                    "residual": residual,  # 残差
                }
            )
        else:
            hidden_states_before_norm = None  # 归一化前的隐藏状态
            if not self.pp_group.is_last_rank:  # 如果不是最后一个PP秩
                return PPProxyTensors(  # 返回代理张量
                    {
                        "hidden_states": hidden_states,  # 隐藏状态
                        "residual": residual,  # 残差
                    }
                )
            else:
                if hidden_states.shape[0] > 0:  # 如果有token
                    # if forward_batch.return_hidden_states_before_norm:
                    hidden_states_before_norm = (  # 计算归一化前的隐藏状态
                        hidden_states if residual is None else hidden_states + residual
                    )
                    if residual is None:  # 如果没有残差
                        hidden_states = self.norm(hidden_states)  # 归一化
                    else:
                        hidden_states, _ = self.norm(hidden_states, residual)  # 带残差归一化
            return hidden_states, hidden_states_before_norm  # 返回隐藏状态和归一化前状态


class Step3p5ForCausalLM(nn.Module):
    """Step3.5因果语言模型，支持BitandBytes量化"""

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [  # BitandBytes目标模块
        ".gate_proj.",  # 门控投影
        ".down_proj.",  # 下投影
        ".up_proj.",  # 上投影
        ".q_proj.",  # Q投影
        ".k_proj.",  # K投影
        ".v_proj.",  # V投影
        ".o_proj.",  # 输出投影
    ]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes堆叠参数映射
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),  # Q投影映射
        "k_proj": ("qkv_proj", 1),  # K投影映射
        "v_proj": ("qkv_proj", 2),  # V投影映射
        "gate_proj": ("gate_up_proj", 0),  # 门控投影映射
        "up_proj": ("gate_up_proj", 1),  # 上投影映射
    }

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        """获取专家位置的模型配置"""
        return ModelConfigForExpertLocation(  # 返回专家位置配置
            num_layers=config.num_hidden_layers,  # 层数
            num_logical_experts=config.moe_num_experts,  # 逻辑专家数
        )

    def __init__(
        self,
        config: Step3p5Config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = Step3p5Model(  # 创建模型主体
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        self.tie_word_embeddings = False  # 是否绑定词嵌入
        self.num_fused_shared_experts = 0  # 融合共享专家数

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:  # 如果是最后一个PP秩
            if self.pp_group.world_size == 1 and self.tie_word_embeddings:  # 单GPU且绑定词嵌入
                self.lm_head = self.model.embed_tokens  # 使用嵌入层作为LM头
            else:
                self.lm_head = ParallelLMHead(  # 并行语言模型头
                    config.vocab_size,  # 词表大小
                    config.hidden_size,  # 隐藏大小
                    quant_config=quant_config,  # 量化配置
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用DP LM头
                    prefix=add_prefix("lm_head", prefix),  # 参数前缀
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()  # 非末秩使用缺失层

        # perform weight tying for PP
        if self.pp_group.world_size > 1 and self.tie_word_embeddings:  # 多PP且绑定词嵌入
            if self.pp_group.is_first_rank:  # 第一个秩发送嵌入权重
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.world_size - 1
                )
            elif self.pp_group.is_last_rank:  # 最后一个秩接收嵌入权重
                emb_token_weight = self.pp_group.recv(
                    size=self.lm_head.weight.shape,
                    dtype=next(self.model.parameters()).dtype,
                    src=0,
                )
                self.lm_head.weight.copy_(emb_token_weight)  # 复制权重

        self.logits_processor = LogitsProcessor(config)  # 创建logits处理器

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.model.get_input_embeddings()  # 返回模型的输入嵌入

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量
    ) -> torch.Tensor:
        """因果语言模型前向传播：模型主体 -> logits处理"""
        hidden_states, hidden_states_before_norm = self.model(  # 通过模型主体
            input_ids,  # 输入ID
            positions,  # 位置
            forward_batch,  # 前向批次
            input_embeds,  # 输入嵌入
            pp_proxy_tensors=pp_proxy_tensors,  # PP代理张量
        )

        if self.pp_group.is_last_rank:  # 如果是最后一个PP秩
            return self.logits_processor(  # 处理logits
                input_ids,  # 输入ID
                hidden_states,  # 隐藏状态
                self.lm_head,  # LM头
                forward_batch,  # 前向批次
                hidden_states_before_norm=hidden_states_before_norm,  # 归一化前隐藏状态
            )
        else:
            return hidden_states  # 非末秩直接返回隐藏状态

    @property
    def start_layer(self):
        """获取起始层索引"""
        return self.model.start_layer  # 返回模型起始层

    @property
    def end_layer(self):
        """获取结束层索引"""
        return self.model.end_layer  # 返回模型结束层

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        """加载模型权重，支持nextn/MTP层权重过滤"""  # NOTE:
        # Step3p5 HF checkpoints (e.g. MTP/nextn variants) may include an extra
        # "nextn predict layer" appended after the main decoder layers, such as:
        #   model.layers.<num_hidden_layers>.(eh_proj|enorm|hnorm|transformer.shared_head.*)
        # This implementation currently does NOT instantiate those nextn modules,
        # so we must safely skip them (or load them only when a corresponding
        # nextn model is implemented).

        def _get_layer_id_from_weight_name(weight_name: str) -> Optional[int]:
            """从权重名称中提取层ID"""
            # Expected format: "model.layers.<id>...."
            parts = weight_name.split(".")  # 分割权重名称
            if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":  # 格式匹配
                try:
                    return int(parts[2])  # 返回层ID
                except ValueError:
                    return None  # 转换失败返回None
            return None  # 不匹配返回None

        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),  # Q投影
            (".qkv_proj", ".k_proj", "k"),  # K投影
            (".qkv_proj", ".v_proj", "v"),  # V投影
            (".gate_up_proj", ".gate_proj", 0),  # 门控投影
            (".gate_up_proj", ".up_proj", 1),  # 上投影
        ]

        if self.num_fused_shared_experts > 0:  # 如果有融合共享专家
            assert self.num_fused_shared_experts == 1  # 断言只有1个

        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 专家参数映射
            ckpt_gate_proj_name="gate_proj",  # 检查点门控投影名称
            ckpt_down_proj_name="down_proj",  # 检查点下投影名称
            ckpt_up_proj_name="up_proj",  # 检查点上投影名称
            num_experts=self.config.moe_num_experts + self.num_fused_shared_experts,  # 专家数
        )

        params_dict = dict(self.named_parameters())  # 参数字典
        loaded_params = set()  # 已加载参数集合

        def match_expert_and_shard_ids(name_path: str, weight_path: str) -> bool:
            """匹配专家和分片ID"""
            name_parts = name_path.split(".")  # 分割名称路径
            weight_parts = weight_path.split(".")  # 分割权重路径
            # Be defensive: some unexpected weight names may not match the shape.
            if len(name_parts) <= 4 or len(weight_parts) <= 2:  # 长度检查
                return False  # 不匹配
            shard_id_matches = name_parts[4] == weight_parts[2]  # 检查分片ID匹配
            return shard_id_matches  # 返回匹配结果

        for name, loaded_weight in weights:  # 遍历所有权重
            # Filter nextn layer weights.
            if hasattr(self.config, "num_nextn_predict_layers"):  # 如果有nextn层配置
                num_nextn_layers = getattr(self.config, "num_nextn_predict_layers", 0)  # 获取nextn层数
                if num_nextn_layers and name.startswith("model.layers."):  # 如果是nextn权重
                    layer_id = _get_layer_id_from_weight_name(name)  # 获取层ID
                    if layer_id is not None:  # 如果有层ID
                        if not is_nextn:  # 非nextn加载模式
                            # Normal load: skip layers appended after the main decoder.
                            if layer_id >= self.config.num_hidden_layers:  # 跳过主解码器之后的层
                                continue
                        else:
                            # nextn load: only keep the appended nextn layer.
                            # (Only 1 nextn layer is supported by current checkpoints.)
                            if num_nextn_layers != 1:  # 仅支持1个nextn层
                                raise ValueError(
                                    "Only 1 nextn layer is supported for Step3p5 checkpoints."
                                )
                            nextn_layer_id = (  # nextn层ID
                                0  # 如果总层数为1
                                if self.config.num_hidden_layers == 1
                                else self.config.num_hidden_layers  # 否则等于总层数
                            )
                            if layer_id != nextn_layer_id:  # 不是nextn层
                                # # nextn/MTP load: only keep the appended nextn layers.
                                # # Expected layer ids: [num_hidden_layers, num_hidden_layers + num_nextn_layers).
                                # start = self.config.num_hidden_layers
                                # end = self.config.num_hidden_layers + num_nextn_layers
                                # if not (start <= layer_id < end):
                                continue  # 跳过

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不包含
                    continue  # 跳过
                if "gate." not in name and "moe" in name:  # MoE权重但不包含gate
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换权重名
                if name not in params_dict:  # 如果参数不存在
                    # Extra / unsupported weights (e.g. nextn) should not crash loading.
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                loaded_params.add(name)  # 添加到已加载集合
                break  # 跳出内循环
            else:  # 没有匹配到堆叠参数
                if "moe" not in name or "router_bias" in name:  # 非MoE权重或路由偏置
                    if name not in params_dict:  # 如果参数不存在
                        continue  # 跳过
                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重
                    loaded_params.add(name)  # 添加到已加载集合
                else:  # MoE权重
                    if "gate." in name:  # 如果是门控权重
                        if name not in params_dict:  # 如果参数不存在
                            continue  # 跳过
                        param = params_dict[name]  # 获取参数
                        weight_loader = param.weight_loader  # 获取权重加载器
                        weight_loader(param, loaded_weight)  # 加载权重
                        loaded_params.add(name)  # 添加到已加载集合
                        continue  # 继续下一个权重

                    for mapping in expert_params_mapping:  # 遍历专家参数映射
                        param_name, weight_name, expert_id, shard_id = mapping  # 解包映射
                        if expert_id == self.config.moe_num_experts:  # 跳过融合共享专家
                            continue
                        if not match_expert_and_shard_ids(name, weight_name):  # 检查匹配
                            continue
                        part_name = weight_name.split(".")[-2]  # 获取部分名称
                        fake_weight_name = name.replace(part_name, weight_name[:-1])  # 构造假权重名
                        actual_param_name = name.replace(part_name + ".", param_name)  # 实际参数名
                        if actual_param_name not in params_dict:  # 如果参数不存在
                            continue  # 跳过
                        param = params_dict[actual_param_name]  # 获取参数
                        weight_loader = param.weight_loader  # 获取权重加载器
                        weight_loader(  # 加载权重
                            param,
                            loaded_weight[expert_id],  # 对应专家的权重
                            name,  # 权重名
                            shard_id=shard_id,  # 分片ID
                            expert_id=expert_id,  # 专家ID
                        )
                        loaded_params.add(actual_param_name)  # 添加到已加载集合

        # Derived parameters (e.g. blockscale_swizzled from NVFP4 quantization)
        # are computed in process_weights_after_loading, not loaded from checkpoint.
        print_params = {  # 未加载参数
            p
            for p in set(params_dict.keys()) - loaded_params  # 参数字典键减去已加载
            if "blockscale_swizzled" not in p  # 排除blockscale_swizzled
        }
        assert len(print_params) == 0, f"Some parameters are not loaded: {print_params}"  # 断言所有参数已加载

    def get_embed_and_head(self):
        """获取嵌入权重和LM头权重"""
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入和LM头权重

    def set_embed_and_head(self, embed, head):
        """设置嵌入权重和LM头权重"""
        del self.model.embed_tokens.weight  # 删除旧嵌入权重
        del self.lm_head.weight  # 删除旧LM头权重
        self.model.embed_tokens.weight = embed  # 设置新嵌入权重
        self.lm_head.weight = head  # 设置新LM头权重
        torch.cuda.empty_cache()  # 清空GPU缓存
        torch.cuda.synchronize()  # 同步CUDA


EntryClass = Step3p5ForCausalLM  # 入口类
