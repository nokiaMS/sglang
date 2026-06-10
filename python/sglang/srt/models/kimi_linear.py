# Kimi Linear 模型实现（混合线性注意力和MLA注意力）
# 该文件实现了Kimi Linear因果语言模型，结合了KimiDeltaAttention（线性注意力/门控delta规则）
# 和DeepseekV2 MLA注意力，支持MoE（混合专家）和Dense MLP，
# 并提供了融合QKV+beta+f_a+g_a投影的优化路径。
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/0384aa7150c4c9778efca041ffd1beb3ad2bd694/vllm/model_executor/models/kimi_linear.py

from collections.abc import Iterable  # 导入可迭代类型
from typing import Optional  # 导入可选类型

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块

from sglang.srt.configs.kimi_linear import KimiLinearConfig  # 导入KimiLinear配置
from sglang.srt.distributed import (  # 导入分布式通信工具
    divide,  # 除法工具
    get_pp_group,  # 获取流水线并行组
    get_tensor_model_parallel_world_size,  # 获取TP世界大小
    tensor_model_parallel_all_reduce,  # 张量模型并行全归约
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入专家分布记录器
from sglang.srt.layers.attention.fla.fused_norm_gate import FusedRMSNormGated  # 导入融合RMS归一化门控
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size  # 导入DP注意力工具
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 导入线性层
    ColumnParallelBatchedLinear,  # 列并行批量线性层
    ColumnParallelLinear,  # 列并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    MergedColumnParallelRepeatedLinear,  # 合并列并行重复线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class  # 导入MoE实现类获取器
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合MoE层
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat  # 导入TopK选择器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention  # 导入基数线性注意力
from sglang.srt.layers.utils import PPMissingLayer  # 导入流水线缺失层
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode  # 导入CUDA图模式检查
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    maybe_remap_kv_scale_name,  # KV缩放名称重映射
    sharded_weight_loader,  # 分片权重加载器
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA as KimiMLAAttention  # 导入MLA注意力（别名）
from sglang.srt.models.llama import LlamaMLP as KimiMLP  # 导入LlamaMLP（别名）
from sglang.srt.models.transformers import maybe_prefix  # 导入前缀工具
from sglang.srt.utils import make_layers  # 导入层创建工具
from sglang.srt.utils.common import BumpAllocator, add_prefix, set_weight_attrs  # 导入通用工具


class KimiMoE(nn.Module):
    """Kimi MoE（混合专家）模块，支持共享专家和路由专家。"""
    def __init__(
        self,
        config: KimiLinearConfig,  # KimiLinear配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        layer_idx: int = 0,  # 层索引
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ):
        super().__init__()  # 调用父类初始化
        hidden_size = config.hidden_size  # 隐藏层大小
        intermediate_size = config.intermediate_size  # 中间层大小
        moe_intermediate_size = config.moe_intermediate_size  # MoE中间层大小
        num_experts = config.num_experts  # 专家数量
        moe_renormalize = config.moe_renormalize  # 是否重归一化
        self.tp_size = get_tensor_model_parallel_world_size()  # TP世界大小
        self.routed_scaling_factor = config.routed_scaling_factor  # 路由缩放因子
        self.num_shared_experts = config.num_shared_experts  # 共享专家数量
        self.layer_idx = layer_idx  # 层索引
        self.alt_stream = alt_stream  # 备用CUDA流

        if config.hidden_act != "silu":  # 检查激活函数
            raise ValueError(  # 抛出错误
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."  # 仅支持SiLU
            )

        # Gate always runs at half / full precision for now.  # 门控始终以半/全精度运行
        self.gate = ReplicatedLinear(  # 门控网络
            hidden_size,  # 输入维度
            num_experts,  # 输出维度（专家数）
            bias=False,  # 不使用偏置
            quant_config=None,  # 门控不量化
            prefix=f"{prefix}.gate",  # 参数前缀
        )

        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(num_experts))  # 专家分数修正偏置

        self.experts = get_moe_impl_class(quant_config)(  # 创建MoE专家实现
            num_experts=config.n_routed_experts,  # 路由专家数
            top_k=config.num_experts_per_token,  # 每token选择的专家数
            hidden_size=config.hidden_size,  # 隐藏大小
            intermediate_size=config.moe_intermediate_size,  # MoE中间层大小
            layer_id=self.layer_idx,  # 层ID
            quant_config=quant_config,  # 量化配置
            routed_scaling_factor=self.routed_scaling_factor,  # 路由缩放因子
            prefix=add_prefix("experts", prefix),  # 参数前缀
        )

        self.topk = TopK(  # TopK选择器
            top_k=config.num_experts_per_token,  # Top-K值
            renormalize=moe_renormalize,  # 是否重归一化
            use_grouped_topk=True,  # 使用分组TopK
            num_expert_group=config.num_expert_group,  # 专家组数
            topk_group=config.topk_group,  # 每组Top-K
            correction_bias=self.gate.e_score_correction_bias,  # 修正偏置
            quant_config=quant_config,  # 量化配置
            routed_scaling_factor=self.routed_scaling_factor,  # 路由缩放因子
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,  # 是否在TopK中融合缩放
            # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized  # 某些FP4 MoE后端需要绕过输出格式
            # and requires the output format to be standard. We use quant_config to determine the output format.  # MTP层未量化需要标准输出格式
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,  # 输出格式
        )

        if self.num_shared_experts is not None:  # 如果有共享专家
            intermediate_size = moe_intermediate_size * self.num_shared_experts  # 共享专家中间层大小
            self.shared_experts = KimiMLP(  # 创建共享专家MLP
                hidden_size=config.hidden_size,  # 隐藏大小
                intermediate_size=intermediate_size,  # 中间层大小
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                reduce_results=False,  # 不归约结果（后续手动归约）
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播：执行MoE路由和专家计算。"""
        num_tokens, hidden_size = hidden_states.shape  # 获取token数和隐藏大小
        hidden_states = hidden_states.view(-1, hidden_size)  # 展平

        shared_output = None  # 共享专家输出

        if (  # 如果可以使用备用流并行
            self.alt_stream is not None  # 有备用流
            and self.num_shared_experts is not None  # 有共享专家
            and hidden_states.shape[0] > 0  # 有token
            and get_is_capture_mode()  # 在CUDA图捕获模式
        ):
            current_stream = torch.cuda.current_stream()  # 获取当前流
            self.alt_stream.wait_stream(current_stream)  # 等待当前流完成

            shared_output = self.shared_experts(hidden_states.clone())  # 在当前流执行共享专家

            with torch.cuda.stream(self.alt_stream):  # 在备用流执行路由专家
                router_logits, _ = self.gate(hidden_states)  # 门控计算
                topk_output = self.topk(hidden_states, router_logits)  # TopK选择
                final_hidden_states = self.experts(hidden_states, topk_output)  # 专家计算

            current_stream.wait_stream(self.alt_stream)  # 等待备用流完成
        else:  # 不使用并行
            if self.num_shared_experts is not None and hidden_states.shape[0] > 0:  # 如果有共享专家
                shared_output = self.shared_experts(hidden_states)  # 执行共享专家
            router_logits, _ = self.gate(hidden_states)  # 门控计算
            topk_output = self.topk(hidden_states, router_logits)  # TopK选择
            final_hidden_states = self.experts(hidden_states, topk_output)  # 专家计算

        if shared_output is not None:  # 如果有共享专家输出
            final_hidden_states = final_hidden_states + shared_output  # 加上共享专家输出

        if self.tp_size > 1:  # 如果TP大于1
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 全归约
        return final_hidden_states.view(num_tokens, hidden_size)  # 返回重塑后的结果


class KimiDeltaAttention(nn.Module):
    """Kimi Delta注意力模块，实现门控delta规则线性注意力。"""
    def __init__(
        self,
        layer_idx: int,  # 层索引
        hidden_size: int,  # 隐藏层大小
        config: KimiLinearConfig,  # KimiLinear配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        rms_norm_eps: float = 1e-5,  # RMS归一化epsilon
        prefix: str = "",  # 参数前缀
        **kwargs,  # 其他关键字参数
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.tp_size = get_tensor_model_parallel_world_size()  # TP世界大小
        self.attn_tp_size = get_attention_tp_size()  # 注意力TP大小
        self.hidden_size = hidden_size  # 保存隐藏大小
        self.config = config  # 保存配置
        self.head_dim = config.linear_attn_config["head_dim"]  # 每个头的维度
        self.num_heads = config.linear_attn_config["num_heads"]  # 注意力头数
        self.num_k_heads = config.linear_attn_config["num_heads"]  # 键头数
        self.num_v_heads = config.linear_attn_config["num_heads"]  # 值头数
        self.head_k_dim = config.linear_attn_config["head_dim"]  # 键头维度
        self.head_v_dim = config.v_head_dim  # 值头维度
        self.layer_idx = layer_idx  # 层索引
        self.prefix = prefix  # 参数前缀
        assert self.num_heads % self.tp_size == 0  # 头数必须能被TP大小整除
        self.local_num_heads = divide(self.num_heads, self.tp_size)  # 本地头数

        projection_size = self.head_dim * self.num_heads  # 投影大小
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]  # 短卷积核大小

        # TODO: support fusion with quant  # TODO: 支持与量化融合
        self.do_fuse_qkvbfg = quant_config is None  # 无量化时启用融合

        if self.do_fuse_qkvbfg:  # 如果启用融合
            # Fuse: q, k, v, beta (column parallel) + f_a, g_a (replicated)  # 融合：q,k,v,beta(列并行) + f_a,g_a(复制)
            self.qkvb_sizes = [  # QKV+beta分片大小
                projection_size,  # q
                projection_size,  # k
                projection_size,  # v
                self.num_heads,  # beta
            ]
            self.fg_sizes = [self.head_dim, self.head_dim]  # f_a, g_a分片大小

            self.fused_qkvbfg_a_proj = MergedColumnParallelRepeatedLinear(  # 融合QKV+beta+fg投影
                self.hidden_size,  # 输入维度
                self.qkvb_sizes,  # 列并行分片大小
                self.fg_sizes,  # 复制分片大小
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.fused_qkvbfg_a_proj",  # 参数前缀
            )
            self.split_sizes = [  # 拆分大小
                3 * projection_size // self.tp_size,  # qkv
                self.num_heads // self.tp_size,  # beta
                2 * self.head_dim,  # f_a, g_a
            ]
            self.fused_fg_b_proj = ColumnParallelBatchedLinear(  # 融合fg的b投影
                2, self.head_dim, projection_size, dtype=config.dtype
            )
        else:  # 不使用融合路径
            # Unfused path: separate QKVParallelLinear  # 非融合路径：独立的QKV并行线性层
            attn_tp_rank = get_attention_tp_rank()  # 获取注意力TP秩
            self.qkv_proj = QKVParallelLinear(  # QKV并行投影
                self.hidden_size,  # 输入维度
                self.head_dim,  # 每头维度
                self.num_heads,  # Q头数
                self.num_k_heads,  # K头数
                bias=False,  # 不使用偏置
                quant_config=quant_config,  # 量化配置
                tp_rank=attn_tp_rank,  # TP秩
                tp_size=self.attn_tp_size,  # TP大小
                v_head_size=self.head_v_dim,  # V头维度
                prefix=f"{prefix}.qkv_proj",  # 参数前缀
            )

            self.f_a_proj = ReplicatedLinear(  # f_a投影
                self.hidden_size,  # 输入维度
                self.head_dim,  # 输出维度
                bias=False,  # 不使用偏置
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.f_a_proj",  # 参数前缀
            )

            self.f_b_proj = ColumnParallelLinear(  # f_b投影
                self.head_dim,  # 输入维度
                projection_size,  # 输出维度
                bias=False,  # 不使用偏置
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.f_b_proj",  # 参数前缀
            )

            self.b_proj = ColumnParallelLinear(  # beta投影
                self.hidden_size,  # 输入维度
                self.num_heads,  # 输出维度
                bias=False,  # 不使用偏置
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.b_proj",  # 参数前缀
            )

            self.g_a_proj = ReplicatedLinear(  # g_a投影
                self.hidden_size,  # 输入维度
                self.head_dim,  # 输出维度
                bias=False,  # 不使用偏置
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.g_a_proj",  # 参数前缀
            )
            self.g_b_proj = ColumnParallelLinear(  # g_b投影
                self.head_dim,  # 输入维度
                projection_size,  # 输出维度
                bias=False,  # 不使用偏置
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.g_b_proj",  # 参数前缀
            )

        self.dt_bias = nn.Parameter(  # dt偏置参数
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)  # 每个本地头的偏置
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})  # 设置权重加载属性

        self.qkv_conv1d = MergedColumnParallelLinear(  # QKV 1D卷积层
            input_size=self.conv_size,  # 输入大小（卷积核大小）
            output_sizes=[projection_size, projection_size, projection_size],  # 输出大小
            bias=False,  # 不使用偏置
            params_dtype=torch.float32,  # 参数数据类型
            prefix=f"{prefix}.qkv_conv1d",  # 参数前缀
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.  # 扩展维度以将conv1d权重形状适配线性权重形状
        # Can't do this in `weight_loader` since it already exists in  # 不能在weight_loader中做
        # `ColumnParallelLinear` and `set_weight_attrs`  # 因为ColumnParallelLinear和set_weight_attrs已有
        # doesn't allow to override it  # 不允许覆盖
        self.qkv_conv1d.weight.data = self.qkv_conv1d.weight.data.unsqueeze(1)  # 在维度1增加一维

        self.A_log = nn.Parameter(  # A的对数参数（衰减率）
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)  # 形状
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})  # 设置权重加载属性

        self.o_norm = FusedRMSNormGated(  # 融合RMS归一化门控
            self.head_dim, eps=rms_norm_eps, activation="sigmoid"  # 维度和激活
        )
        self.o_proj = RowParallelLinear(  # 输出投影
            projection_size,  # 输入维度
            self.hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.o_proj",  # 参数前缀
        )

        conv_weights = self.qkv_conv1d.weight.squeeze(1)  # 压缩卷积权重
        bias = self.qkv_conv1d.bias  # 卷积偏置

        self.attn = RadixLinearAttention(  # 基数线性注意力
            layer_id=self.layer_idx,  # 层ID
            num_q_heads=self.num_k_heads // self.attn_tp_size,  # Q头数
            num_k_heads=self.num_k_heads // self.attn_tp_size,  # K头数
            num_v_heads=self.num_v_heads // self.attn_tp_size,  # V头数
            head_q_dim=self.head_k_dim,  # Q头维度
            head_k_dim=self.head_k_dim,  # K头维度
            head_v_dim=self.head_v_dim,  # V头维度
            conv_weights=conv_weights,  # 卷积权重
            bias=bias,  # 偏置
            A_log=self.A_log,  # A对数参数
            dt_bias=self.dt_bias,  # dt偏置
        )

    def forward_qkvbfg(self, hidden_states: torch.Tensor):
        """非融合路径：分别计算QKV、beta、遗忘门和g投影。"""
        qkv, _ = self.qkv_proj(hidden_states)  # QKV投影

        # Compute beta, forget_gate, and g_proj_states  # 计算beta、遗忘门和g投影
        beta = self.b_proj(hidden_states)[0]  # beta投影
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]  # 遗忘门（两步投影）
        g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]  # g投影（两步投影）

        return (  # 返回四元组
            qkv,  # QKV
            beta,  # beta
            forget_gate,  # 遗忘门
            g_proj_states,  # g投影状态
        )

    def forward_qkvbfg_fused(self, hidden_states: torch.Tensor):
        """融合路径：一次投影计算QKV+beta+f_a+g_a，然后批量计算遗忘门和g投影。"""
        # Single fused projection for all: qkv + beta + f_a + g_a  # 单次融合投影
        fused_states = self.fused_qkvbfg_a_proj(hidden_states)  # 融合投影

        qkv, beta, fg_a_states = torch.split(  # 拆分为QKV、beta和fg_a
            fused_states,  # 融合状态
            self.split_sizes,  # 拆分大小
            dim=-1,  # 在最后一维拆分
        )

        # use batch matmul to calculate forget_gate and g_proj_states  # 使用批量矩阵乘法计算遗忘门和g投影
        forget_gate, g_proj_states = self.fused_fg_b_proj(  # 批量b投影
            fg_a_states.view(-1, 2, self.head_dim).transpose(0, 1)  # 重排为批量格式
        )

        return (  # 返回四元组
            qkv,  # QKV
            beta,  # beta
            forget_gate,  # 遗忘门
            g_proj_states,  # g投影状态
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        zero_allocator: BumpAllocator,  # 零分配器
    ) -> None:
        """前向传播：执行Kimi Delta线性注意力计算。"""
        if self.do_fuse_qkvbfg:  # 如果使用融合路径
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg_fused(  # 融合投影
                hidden_states
            )
        else:  # 否则使用非融合路径
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(  # 非融合投影
                hidden_states
            )

        # For prefill: raw gate is passed to chunk_kda_fwd, which fuses gate  # 预填充：原始门控传递给chunk_kda_fwd
        # activation with chunk_local_cumsum (kda_gate_chunk_cumsum kernel).  # 融合门控激活与分块局部累加
        # For decode: gate activation is handled inside fused_recurrent kernel.  # 解码：门控激活在fused_recurrent内核中处理
        if not forward_batch.forward_mode.is_decode():  # 如果不是解码模式
            forget_gate = forget_gate.unflatten(
                -1, (-1, self.head_dim)
            )  # [T, H*K] -> [T, H, K]  # 重排遗忘门
            beta = beta.float().sigmoid()  # 对beta应用sigmoid
            forget_gate = forget_gate.unsqueeze(0)  # 增加batch维度
        beta = beta.unsqueeze(0)  # 增加batch维度

        core_attn_out = self.attn(  # 执行线性注意力
            forward_batch,  # 前向批次
            mixed_qkv=mixed_qkv,  # 混合QKV
            a=forget_gate,  # 遗忘门作为a
            b=beta,  # beta作为b
        )

        norm_gate = g_proj_states.unflatten(
            -1, (-1, self.head_dim)
        )  # ... (h d) -> ... h d  # 重排归一化门控
        core_attn_out = self.o_norm(core_attn_out, norm_gate)  # 应用融合RMS归一化门控
        core_attn_out = core_attn_out.squeeze(0).flatten(-2)  # 1 n h d -> n (h d)  # 展平

        return self.o_proj(core_attn_out)[0]  # 输出投影


class KimiDecoderLayer(nn.Module):
    """Kimi解码器层，根据配置选择线性注意力或MLA注意力。"""
    def __init__(
        self,
        config: KimiLinearConfig,  # KimiLinear配置
        layer_idx: int,  # 层索引
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 保存隐藏大小
        self.alt_stream = alt_stream  # 保存备用流

        self.is_moe = config.is_moe  # 是否为MoE层

        if config.is_kda_layer(layer_idx):  # 如果是KDA层
            self.self_attn = KimiDeltaAttention(  # 使用Kimi Delta注意力
                layer_idx=layer_idx,  # 层索引
                hidden_size=config.hidden_size,  # 隐藏大小
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.self_attn",  # 参数前缀
            )
        else:  # 标准MLA注意力层
            self.self_attn = KimiMLAAttention(  # 使用MLA注意力
                layer_id=layer_idx,  # 层ID
                hidden_size=self.hidden_size,  # 隐藏大小
                num_heads=config.num_attention_heads,  # 头数
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.self_attn",  # 参数前缀
                config=config,  # 配置
                qk_nope_head_dim=config.qk_nope_head_dim,  # 非旋转头维度
                qk_rope_head_dim=config.qk_rope_head_dim,  # 旋转头维度
                v_head_dim=config.v_head_dim,  # 值头维度
                q_lora_rank=config.q_lora_rank,  # Q LoRA秩
                kv_lora_rank=config.kv_lora_rank,  # KV LoRA秩
                skip_rope=True,  # 跳过RoPE（KDA层自行处理）
            )

        if (  # 如果是MoE层
            self.is_moe
            and config.num_experts is not None  # 有专家数配置
            and layer_idx >= config.first_k_dense_replace  # 超过前k个Dense层
            and layer_idx % config.moe_layer_freq == 0  # 符合MoE层频率
        ):
            self.block_sparse_moe = KimiMoE(  # 创建MoE模块
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                layer_idx=layer_idx,  # 层索引
                prefix=f"{prefix}.mlp",  # 参数前缀
                alt_stream=self.alt_stream,  # 备用流
            )
            self.mlp = self.block_sparse_moe  # MLP使用MoE
        else:  # Dense MLP
            self.mlp = KimiMLP(  # 创建Dense MLP
                hidden_size=self.hidden_size,  # 隐藏大小
                intermediate_size=config.intermediate_size,  # 中间层大小
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.mlp",  # 参数前缀
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏大小和epsilon
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差
        zero_allocator: BumpAllocator,  # 零分配器
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播：执行注意力 + MLP的解码器层计算。"""
        # Self Attention  # 自注意力
        if residual is None:  # 如果没有残差
            residual = hidden_states  # 保存隐藏状态作为残差
            hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化
        else:  # 有残差
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 带残差的归一化

        hidden_states = self.self_attn(  # 执行自注意力
            hidden_states=hidden_states,  # 隐藏状态
            positions=positions,  # 位置编码
            forward_batch=forward_batch,  # 前向批次
            zero_allocator=zero_allocator,  # 零分配器
        )

        # Fully Connected  # 全连接层
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # 注意力后归一化
        hidden_states = self.mlp(hidden_states)  # 通过MLP/MoE
        return hidden_states, residual  # 返回隐藏状态和残差


class KimiLinearModel(nn.Module):
    """Kimi Linear模型主体，包含嵌入层、解码器层和归一化。"""
    def __init__(
        self,
        config: KimiLinearConfig,  # KimiLinear配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置

        self.padding_idx = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小
        self.pp_group = get_pp_group()  # 获取流水线并行组

        if self.pp_group.is_first_rank:  # 如果是第一个流水线秩
            self.embed_tokens = VocabParallelEmbedding(  # 创建词表并行嵌入层
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏大小
                prefix=f"{prefix}.embed_tokens",  # 参数前缀
            )
        else:  # 否则
            self.embed_tokens = PPMissingLayer()  # 使用缺失层占位

        self.alt_stream = torch.cuda.Stream()  # 创建备用CUDA流

        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,  # 层数
            lambda idx, prefix: KimiDecoderLayer(  # 层创建函数
                layer_idx=idx,  # 层索引
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                prefix=prefix,  # 参数前缀
                alt_stream=self.alt_stream,  # 备用流
            ),
            pp_rank=self.pp_group.rank_in_group,  # 流水线秩
            pp_size=self.pp_group.world_size,  # 流水线世界大小
            prefix=f"{prefix}.layers",  # 参数前缀
        )

        if self.pp_group.is_last_rank:  # 如果是最后一个流水线秩
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 创建最终归一化
        else:  # 否则
            self.norm = PPMissingLayer()  # 使用缺失层占位

        world_size = get_tensor_model_parallel_world_size()  # 获取TP世界大小
        assert (  # 验证头数可被TP大小整除
            config.num_attention_heads % world_size == 0
        ), "num_attention_heads must be divisible by world_size"

    def forward(
        self,
        input_ids: torch.Tensor | None,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        inputs_embeds: torch.Tensor | None = None,  # 输入嵌入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ) -> torch.Tensor:
        """前向传播：Kimi Linear模型主体推理。"""
        if get_pp_group().is_first_rank:  # 如果是第一个流水线秩
            if inputs_embeds is not None:  # 如果有输入嵌入
                hidden_states = inputs_embeds  # 直接使用
            else:  # 否则
                hidden_states = self.embed_tokens(input_ids)  # 通过嵌入层
            residual = None  # 初始无残差
        else:  # 非第一个秩
            assert pp_proxy_tensors is not None  # 确保有代理张量
            hidden_states = pp_proxy_tensors["hidden_states"]  # 获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 获取残差

        total_num_layers = self.end_layer - self.start_layer  # 本秩的层数
        device = hidden_states.device  # 设备
        zero_allocator = BumpAllocator(  # 创建零分配器
            buffer_size=total_num_layers * 2,  # 缓冲区大小
            dtype=torch.float32,  # 数据类型
            device=device,  # 设备
        )
        # TODO: capture aux hidden states  # TODO: 捕获辅助隐藏状态
        aux_hidden_states = []  # 辅助隐藏状态列表
        for i in range(self.start_layer, self.end_layer):  # 遍历本秩的层
            ctx = get_global_expert_distribution_recorder().with_current_layer(i)  # 设置当前层
            with ctx:  # 在上下文中
                layer = self.layers[i]  # 获取层
                hidden_states, residual = layer(  # 通过层
                    positions=positions,  # 位置编码
                    hidden_states=hidden_states,  # 隐藏状态
                    forward_batch=forward_batch,  # 前向批次
                    residual=residual,  # 残差
                    zero_allocator=zero_allocator,  # 零分配器
                )

        if not self.pp_group.is_last_rank:  # 如果不是最后一个秩
            return PPProxyTensors(  # 返回代理张量
                {
                    "hidden_states": hidden_states,  # 隐藏状态
                    "residual": residual,  # 残差
                }
            )
        else:  # 最后一个秩
            if hidden_states.shape[0] != 0:  # 如果有token
                if residual is None:  # 无残差
                    hidden_states = self.norm(hidden_states)  # 归一化
                else:  # 有残差
                    hidden_states, _ = self.norm(hidden_states, residual)  # 带残差归一化

        if len(aux_hidden_states) == 0:  # 如果没有辅助隐藏状态
            return hidden_states  # 返回隐藏状态

        return hidden_states, aux_hidden_states  # 返回隐藏状态和辅助状态


class KimiLinearForCausalLM(nn.Module):
    """Kimi Linear因果语言模型，结合线性注意力和MLA注意力。"""
    def __init__(
        self,
        config: KimiLinearConfig,  # KimiLinear配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = KimiLinearModel(  # 创建模型主体
            config, quant_config, prefix=maybe_prefix(prefix, "model")  # 配置和前缀
        )
        self.pp_group = get_pp_group()  # 获取流水线并行组
        if self.pp_group.is_last_rank:  # 如果是最后一个秩
            self.lm_head = ParallelLMHead(  # 创建语言模型头
                self.config.vocab_size,  # 词表大小
                self.config.hidden_size,  # 隐藏大小
                quant_config=quant_config,  # 量化配置
                prefix=maybe_prefix(prefix, "lm_head"),  # 参数前缀
            )
        else:  # 非最后秩
            self.lm_head = PPMissingLayer()  # 使用缺失层占位
        logit_scale = getattr(self.config, "logit_scale", 1.0)  # 获取logit缩放因子
        self.logits_processor = LogitsProcessor(config=config, logit_scale=logit_scale)  # 创建logits处理器

    @torch.no_grad()  # 禁用梯度计算
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ) -> torch.Tensor:
        """前向传播：Kimi Linear因果语言模型推理。"""
        hidden_states = self.model(  # 通过模型主体
            input_ids,  # 输入ID
            positions,  # 位置编码
            forward_batch,  # 前向批次
            inputs_embeds,  # 输入嵌入
            pp_proxy_tensors,  # 流水线代理
        )
        if self.pp_group.is_last_rank:  # 如果是最后一个秩
            return self.logits_processor(  # 返回logits处理结果
                input_ids, hidden_states, self.lm_head, forward_batch  # 参数
            )
        else:  # 非最后秩
            return hidden_states  # 返回隐藏状态

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """加载模型权重，支持堆叠参数、融合投影和专家参数映射。"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            (".gate_up_proj", ".gate_proj", 0),  # gate_up中的gate
            (".gate_up_proj", ".up_proj", 1),  # gate_up中的up
            # Fused path  # 融合路径
            (".fused_qkvbfg_a_proj", ".q_proj", 0),  # 融合投影中的Q
            (".fused_qkvbfg_a_proj", ".k_proj", 1),  # 融合投影中的K
            (".fused_qkvbfg_a_proj", ".v_proj", 2),  # 融合投影中的V
            (".fused_qkvbfg_a_proj", ".b_proj", 3),  # 融合投影中的B
            (".fused_qkvbfg_a_proj", ".f_a_proj", 4),  # 融合投影中的f_a
            (".fused_qkvbfg_a_proj", ".g_a_proj", 5),  # 融合投影中的g_a
            (".fused_fg_b_proj", ".f_b_proj", 0),  # 融合fg_b中的f_b
            (".fused_fg_b_proj", ".g_b_proj", 1),  # 融合fg_b中的g_b
            # Unfused path: separate qkv_proj (when do_fuse_qkvbfg=False)  # 非融合路径
            (".qkv_proj", ".q_proj", "q"),  # QKV中的Q
            (".qkv_proj", ".k_proj", "k"),  # QKV中的K
            (".qkv_proj", ".v_proj", "v"),  # QKV中的V
            # qkv conv fuse  # QKV卷积融合
            (".qkv_conv1d", ".q_conv1d", 0),  # Q卷积
            (".qkv_conv1d", ".k_conv1d", 1),  # K卷积
            (".qkv_conv1d", ".v_conv1d", 2),  # V卷积
        ]
        if self.config.is_moe:  # 如果是MoE模型
            # Params for weights, fp8 weight scales, fp8 activation scales  # 权重、FP8权重缩放、FP8激活缩放参数
            # (param_name, weight_name, expert_id, shard_id)  # (参数名, 权重名, 专家ID, 分片ID)
            expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 创建专家参数映射
                ckpt_gate_proj_name="w1",  # 检查点gate名
                ckpt_down_proj_name="w2",  # 检查点down名
                ckpt_up_proj_name="w3",  # 检查点up名
                num_experts=self.config.num_experts,  # 专家数
            )
        else:  # 非MoE
            expert_params_mapping = []  # 空映射
        params_dict = dict(self.named_parameters())  # 参数字典
        loaded_params: set[str] = set()  # 已加载参数集合
        for args in weights:  # 遍历权重
            name, loaded_weight = args[:2]  # 获取名称和权重
            kwargs = args[2] if len(args) > 2 else {}  # 获取额外参数
            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入频率
                continue

            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的cos/sin
                # Models trained using ColossalAI may include these tensors in  # ColossalAI训练的模型可能包含这些张量
                # the checkpoint. Skip them.  # 跳过
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不匹配
                    continue  # 跳过
                # We have mlp.experts[0].gate_proj in the checkpoint.  # 检查点中有mlp.experts[0].gate_proj
                # Since we handle the experts below in expert_params_mapping,  # 因为专家在下面的expert_params_mapping中处理
                # we need to skip here BEFORE we update the name, otherwise  # 需要在更新名称前跳过
                # name will be updated to mlp.experts[0].gate_up_proj, which  # 否则名称会更新为gate_up_proj
                # will then be updated below in expert_params_mapping  # 然后在expert_params_mapping中再次更新
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.  # 导致加载失败
                if ("mlp.experts." in name) and name not in params_dict:  # 专家权重且不在参数字典中
                    continue  # 跳过
                # Check if this mapping targets a fused projection (only apply fusion check to fused params)  # 检查是否针对融合投影
                if param_name in {".fused_qkvbfg_a_proj", ".fused_fg_b_proj"}:  # 如果是融合投影
                    layer_id = int(name.split(".")[2])  # 获取层ID
                    if not self.config.is_kda_layer(layer_id):  # 如果不是KDA层
                        continue  # 跳过
                    layer = self.model.layers[layer_id].self_attn  # 获取注意力层
                    # Only load to fused projection if fusion is enabled  # 仅在融合启用时加载到融合投影
                    if not getattr(layer, "do_fuse_qkvbfg", False):  # 如果未启用融合
                        continue  # 跳过
                if weight_name in {".q_proj", ".k_proj", ".v_proj"}:  # QKV投影
                    layer_id = int(name.split(".")[2])  # 获取层ID
                    if not self.config.is_kda_layer(layer_id):  # 如果不是KDA层
                        continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换权重名
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果偏置不存在
                    continue  # 跳过
                # if is_pp_missing_parameter(name, self):  # 如果是PP缺失参数
                #     continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出内层循环
            else:  # 如果没有匹配到堆叠参数
                for idx, (param_name, weight_name, expert_id, shard_id) in enumerate(  # 遍历专家参数映射
                    expert_params_mapping
                ):
                    if weight_name not in name:  # 如果权重名不匹配
                        continue  # 跳过
                    name = name.replace(weight_name, param_name)  # 替换权重名
                    # if is_pp_missing_parameter(name, self):  # PP缺失参数检查
                    #     continue  # 跳过
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(  # 加载权重
                        param,  # 参数
                        loaded_weight,  # 权重数据
                        name,  # 参数名
                        expert_id=expert_id,  # 专家ID
                        shard_id=shard_id,  # 分片ID
                    )
                    break  # 跳出内层循环
                else:  # 如果也没有匹配到专家参数
                    # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置
                    if (  # 如果是偏置且不在参数字典中
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn  # 非线性注意力允许加载偏置
                    ):  # noqa: E501
                        continue  # 跳过
                    # Remapping the name of FP8 kv-scale.  # 重映射FP8 KV缩放名称
                    name = maybe_remap_kv_scale_name(name, params_dict)  # 重映射
                    if name is None:  # 如果重映射后为None
                        continue  # 跳过
                    # if is_pp_missing_parameter(name, self):  # PP缺失参数检查
                    #     continue  # 跳过

                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认加载器
                    )
                    weight_loader(param, loaded_weight, **kwargs)  # 加载权重
            loaded_params.add(name)  # 记录已加载的参数

        for layer_id in self.config.full_attention_layer_ids:  # 遍历全注意力层
            self_attn = self.model.layers[layer_id].self_attn  # 获取注意力层
            w_kc, w_vc = self_attn.kv_b_proj.weight.unflatten(  # 反展平kv_b_proj权重
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)  # 拆分为kc和vc
            self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)  # 设置w_kc
            self_attn.w_vc = w_vc.contiguous().transpose(1, 2)  # 设置w_vc
            if hasattr(self_attn.kv_b_proj, "weight_scale"):  # 如果有权重缩放
                self_attn.w_scale = self_attn.kv_b_proj.weight_scale  # 设置权重缩放


EntryClass = KimiLinearForCausalLM  # 入口类：Kimi Linear因果语言模型
