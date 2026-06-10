# 百灵MoE线性注意力混合模型推理实现
# 该文件实现了BailingMoE线性注意力混合架构的推理版本，主要特点包括：
# - 线性注意力和标准softmax注意力的混合架构
# - 支持MLA（Multi-head Latent Attention）和GQA
# - 分组RMS归一化和门控机制
# - 支持FP8/INT8量化和DeepEP后端
# - 支持FP8权重量化的后处理和重量化
# - 兼容DeepSeek V2/V3的MLA注意力
# coding=utf-8
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved. # 版权归属Antgroup和HuggingFace
import copy # 导入深拷贝模块
import logging # 导入日志模块
from typing import Callable, Iterable, Optional, Set, Tuple, Union # 导入类型提示

import torch # 导入PyTorch
import torch.nn.functional as F # 导入PyTorch函数式模块
from torch import nn # 导入神经网络模块
from transformers import PretrainedConfig # 导入预训练配置类

from sglang.srt.distributed import ( # 导入分布式相关模块
    get_pp_group, # 获取流水线并行组
    get_tensor_model_parallel_rank, # 获取张量并行排名
    get_tensor_model_parallel_world_size, # 获取张量并行世界大小
    tensor_model_parallel_all_reduce, # 张量并行全归约
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder # 导入全局专家分布记录器
from sglang.srt.layers import deep_gemm_wrapper # 导入DeepGEMM包装器
from sglang.srt.layers.activation import SiluAndMul # 导入SiLU与乘法激活函数
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated # 导入门控RMS归一化
from sglang.srt.layers.attention.fla.layernorm_gated import layernorm_fn # 导入层归一化函数
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes # 导入层通信器和散射模式
from sglang.srt.layers.dp_attention import ( # 导入数据并行注意力模块
    get_attention_tp_rank, # 获取注意力TP排名
    get_attention_tp_size, # 获取注意力TP大小
    is_dp_attention_enabled, # 是否启用DP注意力
)
from sglang.srt.layers.layernorm import RMSNorm # 导入RMS层归一化
from sglang.srt.layers.linear import ( # 导入线性层
    ColumnParallelLinear, # 列并行线性层
    MergedColumnParallelLinear, # 合并列并行线性层
    QKVParallelLinear, # QKV并行线性层
    RowParallelLinear, # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor # 导入logits处理器
from sglang.srt.layers.moe import should_skip_post_experts_all_reduce # 导入是否跳过专家后全归约
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class # 导入DeepEP MoE和MoE实现类
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE # 导入融合MoE Triton层
from sglang.srt.layers.moe.topk import TopK # 导入TopK选择器
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz # 导入FP8 FNUZ检测
from sglang.srt.layers.quantization.fp8_utils import ( # 导入FP8量化工具
    block_quant_dequant, # 块量化反量化
    block_quant_to_tensor_quant, # 块量化到张量量化
    channel_quant_to_tensor_quant, # 通道量化到张量量化
    normalize_e4m3fn_to_e4m3fnuz, # E4M3FN到E4M3FNUZ归一化
    requant_weight_ue8m0_inplace, # UE8M0原位重量化
)
from sglang.srt.layers.quantization.int8_utils import ( # 导入INT8量化工具
    block_dequant as int8_block_dequant, # INT8块反量化
)
from sglang.srt.layers.radix_attention import RadixAttention # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope_wrapper # 导入旋转位置编码包装器
from sglang.srt.layers.utils import PPMissingLayer # 导入流水线并行缺失层
from sglang.srt.layers.vocab_parallel_embedding import ( # 导入词表并行嵌入
    ParallelLMHead, # 并行语言模型头
    VocabParallelEmbedding, # 词表并行嵌入层
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode # 导入CUDA图捕获模式检测
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, DeepseekV2MLP, _is_hip # 导入DeepSeek V2模型组件
from sglang.srt.models.utils import WeightsMapper # 导入权重映射器
from sglang.srt.server_args import get_global_server_args # 导入全局服务器参数
from sglang.srt.utils import ( # 导入工具函数
    BumpAllocator, # 凸起分配器
    add_prefix, # 添加前缀
    bind_or_assign, # 绑定或赋值
    cpu_has_amx_support, # CPU AMX支持检测
    get_bool_env_var, # 获取布尔环境变量
    get_device_sm, # 获取设备SM版本
    is_cpu, # CPU检测
    is_cuda, # CUDA检测
    is_flashinfer_available, # FlashInfer可用性检测
    is_gfx95_supported, # GFX95支持检测
    is_hip, # HIP检测
    is_npu, # NPU检测
    is_sm100_supported, # SM100支持检测
    make_layers, # 创建层
)
from sglang.srt.utils.common import rank0_log # 导入rank0日志

_is_hip = is_hip() # 是否为HIP环境
_is_cuda = is_cuda() # 是否为CUDA环境
_is_npu = is_npu() # 是否为NPU环境
_is_fp8_fnuz = is_fp8_fnuz() # 是否为FP8 FNUZ格式
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip # 是否使用AITER
_is_cpu_amx_available = cpu_has_amx_support() # CPU AMX是否可用
_is_cpu = is_cpu() # 是否为CPU环境
_device_sm = get_device_sm() # 设备SM版本
_is_gfx95_supported = is_gfx95_supported() # GFX95是否支持

_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported # 是否使用AITER GFX95

if _use_aiter_gfx95: # 如果使用AITER GFX95
    pass # 目前无额外操作

if _is_cuda: # 如果是CUDA环境
    from sgl_kernel import awq_dequantize # 从sgl_kernel导入AWQ反量化
elif _is_cpu and _is_cpu_amx_available: # 如果是CPU且支持AMX
    pass # 目前无额外操作
elif _is_hip: # 如果是HIP环境
    from sglang.srt.layers.quantization.awq.awq_triton import ( # 从AWQ Triton导入
        awq_dequantize_triton as awq_dequantize, # AWQ反量化Triton版本
    )
else: # 否则
    from vllm._custom_ops import awq_dequantize # 从vLLM导入AWQ反量化

if _is_hip: # 如果是HIP环境
    pass # 目前无额外操作

_is_flashinfer_available = is_flashinfer_available() # FlashInfer是否可用
_is_sm100_supported = is_cuda() and is_sm100_supported() # SM100是否支持


class DsV3MLA(DeepseekV2AttentionMLA): # DeepSeek V3 MLA注意力适配器
    def __init__(self, **kwargs): # MLA初始化方法
        super().__init__(**kwargs) # 调用父类初始化
        if kwargs["rope_scaling"]: # 如果有ROPE缩放
            self.rotary_emb.forward = self.rotary_emb.forward_cuda # 使用CUDA前向传播


LoraConfig = None # LoRA配置初始化为空
logger = logging.getLogger(__name__) # 获取当前模块的日志记录器
_is_cpu = is_cpu() # 是否为CPU环境


def is_linear_layer(layer_idx, layer_group_size): # 判断是否为线性注意力层
    if layer_idx is None: # 如果层索引为空
        return False # 不是线性层
    if layer_group_size > 0: # 如果层分组大小大于0
        return (layer_idx + 1) % layer_group_size != 0 # 每组最后一层为全注意力，其余为线性
    else: # 否则
        return False # 不是线性层


def is_pp_missing_parameter( # 判断是否为PP缺失参数
    name: str, # 参数名
    model: torch.nn.Module, # 模型
) -> bool:
    if isinstance(model, PPMissingLayer): # 如果模型是PP缺失层
        return True # 是缺失参数
    return False # 不是缺失参数


def weight_loader_with_alias(alias: str): # 带别名的权重加载器装饰器
    def wrapper(func: Callable): # 包装函数
        def inner_func( # 内部函数
            param: torch.Tensor, # 参数张量
            loaded_weight: torch.Tensor, # 加载的权重
            *args,
            prefix: str = None, # 前缀
            **kwargs,
        ):
            # pf = "[vLLM][load]" + " " if prefix is None else f"[{prefix}] " # 注释掉的日志前缀
            value = func(param, loaded_weight, *args, **kwargs) # 调用原始加载函数
            return value # 返回值

        return inner_func # 返回内部函数

    return wrapper # 返回包装函数


class BailingMLP(nn.Module): # 百灵MLP模块

    def __init__( # MLP初始化方法
        self,
        hidden_size: int, # 隐藏层大小
        intermediate_size: int, # 中间层大小
        reduce_results=True, # 是否归约结果
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear( # 合并的gate和up投影
            hidden_size, # 输入维度
            [intermediate_size] * 2, # 输出维度（gate和up各一份）
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.gate_up_proj", # 参数前缀
        )
        self.down_proj = RowParallelLinear( # 下投影线性层
            intermediate_size, # 输入维度
            hidden_size, # 输出维度
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            reduce_results=reduce_results, # 是否归约结果
            prefix=f"{prefix}.down_proj", # 参数前缀
        )
        self.act_fn = SiluAndMul() # SiLU与乘法激活函数

    def forward( # MLP前向传播
        self,
        x, # 输入张量
        should_allreduce_fusion: bool = False, # 是否融合全归约
        use_reduce_scatter: bool = False, # 是否使用reduce-scatter
    ):
        x, _ = self.gate_up_proj(x) # gate和up投影
        x = self.act_fn(x) # 应用激活函数
        x, _ = self.down_proj( # 下投影
            x,
            skip_all_reduce=use_reduce_scatter or should_allreduce_fusion, # 是否跳过全归约
        )
        return x # 返回输出


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


class BailingMoE(nn.Module): # 百灵MoE模块（线性注意力版本）

    def __init__( # MoE初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        layer_id: int = 0, # 层ID
        prefix: str = "moe", # 参数前缀
        alt_stream=None, # 备用CUDA流
    ):
        super().__init__() # 调用父类初始化

        self.alt_stream = alt_stream # 保存备用CUDA流
        self.layer_id = layer_id # 保存层ID

        self.tp_size = get_tensor_model_parallel_world_size() # 获取TP大小
        self.tp_rank = get_tensor_model_parallel_rank() # 获取TP排名

        self.top_k = config.num_experts_per_tok # 每个token选择的专家数
        self.norm_expert_prob = getattr(config, "norm_topk_prob", False) # 是否归一化专家概率
        self.hidden_size = config.hidden_size # 隐藏层大小
        self.intermediate_size = config.moe_intermediate_size # MoE中间层大小
        self.num_shared_experts = getattr(config, "num_shared_experts", 0) # 共享专家数
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0) # 路由缩放因子
        self.score_function = getattr(config, "score_function", None) # 分数函数类型

        # Gate always runs at half / full precision for now. # 门控目前始终以半精度/全精度运行
        router_dtype = getattr(config, "router_dtype", None) # 获取路由器数据类型
        if router_dtype is None: # 如果未指定
            self.router_dtype = torch.float32 # 默认使用fp32
        elif router_dtype == "fp32": # 如果指定为fp32
            self.router_dtype = torch.float32 # 使用fp32
        else: # 否则
            self.router_dtype = torch.bfloat16 # 使用bfloat16

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

        self.num_experts = config.num_experts # 专家数

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
            use_grouped_topk=self.use_grouped_topk, # 是否使用分组top-k
            renormalize=self.norm_expert_prob, # 是否重归一化
            num_expert_group=self.num_expert_group, # 专家分组数
            topk_group=self.topk_group, # 每组选择的专家数
            correction_bias=self.correction_bias, # 修正偏置
            routed_scaling_factor=self.routed_scaling_factor, # 路由缩放因子
        )
        moe_cls = get_moe_impl_class(quant_config) # 获取MoE实现类
        self.experts = moe_cls( # 实例化专家模块
            num_experts=self.num_experts, # 专家数
            top_k=self.top_k, # top-k值
            layer_id=self.layer_id, # 层ID
            hidden_size=self.hidden_size, # 隐藏层大小
            intermediate_size=self.intermediate_size, # 中间层大小
            quant_config=quant_config, # 量化配置
            routed_scaling_factor=self.routed_scaling_factor, # 路由缩放因子
            prefix=f"{prefix}.experts", # 参数前缀
        )

        if self.num_shared_experts > 0: # 如果有共享专家
            intermediate_size = self.intermediate_size * self.num_shared_experts # 计算共享专家中间大小
            self.shared_experts = BailingMLP( # 共享专家MLP
                hidden_size=self.hidden_size,
                intermediate_size=intermediate_size,
                reduce_results=False, # 不归约结果
                prefix=f"{prefix}.shared_experts",
                quant_config=quant_config,
            )

    def forward( # MoE前向传播
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        should_allreduce_fusion: bool = False, # 是否融合全归约
        use_reduce_scatter: bool = False, # 是否使用reduce-scatter
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape # 获取token数和隐藏大小
        hidden_states = hidden_states.view(-1, hidden_size) # 重塑形状

        if ( # 如果可以使用双流
            self.alt_stream is not None
            and self.num_shared_experts > 0
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            with torch.no_grad(): # 禁用梯度
                current_stream = torch.cuda.current_stream() # 获取当前CUDA流
                self.alt_stream.wait_stream(current_stream) # 等待当前流完成
                # Main stream: shared experts (smaller computation) # 主流：共享专家（较小计算量）
                shared_output = self.shared_experts(hidden_states) # 计算共享专家
                # Alt stream: gate + topk + routed experts # 备用流：门控 + top-k + 路由专家
                with torch.cuda.stream(self.alt_stream): # 在备用流上执行
                    router_logits = self.gate(hidden_states) # 计算路由logits
                    topk_output = self.topk(hidden_states, router_logits) # 计算top-k
                    final_hidden_states = self.experts(hidden_states, topk_output) # 计算路由专家
                current_stream.wait_stream(self.alt_stream) # 等待备用流完成
                final_hidden_states = final_hidden_states + shared_output # 合并输出
        else: # 否则顺序计算
            if self.num_shared_experts > 0: # 如果有共享专家
                shared_output = self.shared_experts(hidden_states) # 计算共享专家

            router_logits = self.gate(hidden_states) # 计算路由logits
            topk_output = self.topk(hidden_states, router_logits) # 计算top-k
            final_hidden_states = self.experts(hidden_states, topk_output) # 计算路由专家

            if self.num_shared_experts > 0: # 如果有共享专家
                final_hidden_states = final_hidden_states + shared_output # 合并输出

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce( # 如果需要全归约
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states) # 张量并行全归约
        return final_hidden_states # 返回最终输出


class BailingGroupRMSNormGate(RMSNormGated): # 百灵分组RMS归一化门控
    def __init__( # 分组RMS归一化门控初始化方法
        self,
        hidden_size, # 隐藏层大小
        eps=1e-5, # epsilon值
        group_size=None, # 分组大小
        norm_before_gate=True, # 是否在门控前归一化
        device=None, # 设备
        dtype=None, # 数据类型
    ):
        super().__init__( # 调用父类初始化
            hidden_size,
            eps=eps,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            device=device,
            dtype=dtype,
            activation="sigmoid", # 使用sigmoid激活
        )
        self.weight.weight_loader = self.weight_loader # 设置权重加载器

    @staticmethod
    def weight_loader( # 静态权重加载器
        param: torch.nn.Parameter, # 参数
        loaded_weight: torch.Tensor, # 加载的权重
    ) -> None:
        tp_size = get_attention_tp_size() # 获取注意力TP大小
        tp_rank = get_attention_tp_rank() # 获取注意力TP排名
        shard_size = loaded_weight.shape[0] // tp_size # 计算分片大小
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size) # 计算分片切片
        param.data.copy_(loaded_weight[shard].contiguous()) # 复制分片数据
        return # 返回


class BailingMoELinearAttention(nn.Module): # 百灵MoE线性注意力模块
    def __init__( # 线性注意力初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        layer_id: int = 0, # 层ID
        prefix: str = "linear_attn", # 参数前缀
        alt_stream=None, # 备用CUDA流
    ):
        super().__init__() # 调用父类初始化

        self.alt_stream = alt_stream # 保存备用CUDA流
        self.layer_id = layer_id # 保存层ID
        self.hidden_size = config.hidden_size # 隐藏层大小
        self.total_num_heads = config.num_attention_heads # 总注意力头数
        self.total_kv_heads = config.num_attention_heads  # MHA # KV头数等于注意力头数（MHA）

        self.head_dim = getattr(config, "head_dim", None) # 获取头维度
        if self.head_dim is None: # 如果头维度未指定
            self.head_dim = config.hidden_size // self.total_num_heads # 由隐藏大小和头数计算

        self.hidden_inner_size = self.head_dim * self.total_num_heads # 隐藏内部大小
        self.scaling = self.head_dim**-0.5 # 缩放因子
        self.tp_size = get_attention_tp_size() # 注意力TP大小
        self.tp_rank = get_attention_tp_rank() # 注意力TP排名

        assert self.total_num_heads % self.tp_size == 0 # 断言总头数可被TP大小整除
        self.tp_heads = self.total_num_heads // self.tp_size # 每个TP rank的头数

        self.max_position_embeddings = config.max_position_embeddings # 最大位置嵌入数
        self.rope_theta = getattr(config, "rope_theta", 600000) # 旋转位置编码theta

        self.tp_kv_heads = self.total_kv_heads // self.tp_size # 每个TP rank的KV头数
        self.q_size_per_rank = self.head_dim * self.tp_heads # 每个rank的Q大小
        self.kv_size_per_rank = self.head_dim * self.tp_kv_heads # 每个rank的KV大小

        self.use_qk_norm = getattr(config, "use_qk_norm", False) # 是否使用QK归一化
        # minimax / seg_la / fla # 线性注意力后端选项
        # TODO support fla # TODO 支持FLA
        self.linear_backend = getattr(config, "linear_backend", "seg_la") # 线性注意力后端
        logger.debug(f"linear_backend in bailing_moe_linear: {self.linear_backend}") # 记录日志
        self.linear_scale = True if self.linear_backend == "minimax" else False # 是否使用线性缩放
        self.linear_rope = getattr(config, "linear_rope", True) # 是否在线性注意力中使用ROPE
        if hasattr(config, "use_linear_silu"): # 如果有use_linear_silu配置
            self.linear_silu = config.use_linear_silu # 使用配置值
        elif hasattr(config, "linear_silu"): # 如果有linear_silu配置
            self.linear_silu = config.linear_silu # 使用配置值
        else: # 否则
            self.linear_silu = False # 不使用线性SiLU

        self.query_key_value = QKVParallelLinear( # QKV并行线性投影
            self.hidden_size, # 输入维度
            self.head_dim, # 头维度
            self.total_num_heads, # 总Q头数
            self.total_kv_heads, # 总KV头数
            bias=(config.use_bias or config.use_qkv_bias), # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.qkv_proj", # 参数前缀
            tp_rank=self.tp_rank, # TP排名
            tp_size=self.tp_size, # TP大小
        )

        if self.use_qk_norm: # 如果使用QK归一化
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) # Q层归一化
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) # K层归一化

        self.g_proj = ColumnParallelLinear( # 门控投影（g_proj）
            self.hidden_size, # 输入维度
            self.hidden_inner_size, # 输出维度
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.output_gate", # 参数前缀
            tp_rank=self.tp_rank, # TP排名
            tp_size=self.tp_size, # TP大小
        )
        self.dense = RowParallelLinear( # 输出投影
            self.hidden_inner_size, # 输入维度
            self.hidden_size, # 输出维度
            bias=config.use_bias, # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.out_proj", # 参数前缀
            tp_rank=self.tp_rank, # TP排名
            tp_size=self.tp_size, # TP大小
            reduce_results=False, # 不归约结果
        )
        self.attn = RadixAttention( # 基数注意力
            self.tp_heads, # 注意力头数
            self.head_dim, # 头维度
            self.scaling, # 缩放因子
            num_kv_heads=self.tp_kv_heads, # KV头数
            layer_id=layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.attn", # 参数前缀
        )
        # Marker for HybridLinearAttnBackend._is_full_attn: Bailing wraps # HybridLinearAttnBackend._is_full_attn标记：百灵将
        # linear-attention layers in a plain RadixAttention, so the # 线性注意力层包装在普通RadixAttention中，因此
        # dispatcher can't tell from the type alone that this is a linear # 调度器无法仅从类型判断这是线性
        # layer (would otherwise default to the full-attn backend, e.g. the # 层（否则会默认为全注意力后端，例如
        # same way MTP/NEXTN draft layers are routed). # 与MTP/NEXTN草稿层相同的方式路由）
        self.attn._is_linear_attention = True # 标记为线性注意力层

        self.group_norm_size = getattr(config, "group_norm_size", 1) # 分组归一化大小
        self.rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-5)) # RMS归一化epsilon
        assert ( # 断言TP大小不超过分组归一化大小
            self.tp_size <= self.group_norm_size
        ), "tp_size must be less than or equal to group_norm_size that can use local rms norm"
        assert ( # 断言分组归一化大小可被TP大小整除
            self.group_norm_size % self.tp_size == 0
        ), "group_norm_size must be divisible by tp_size"
        self.g_norm = BailingGroupRMSNormGate( # 分组RMS归一化门控
            hidden_size=self.hidden_inner_size // self.tp_size,
            eps=self.rms_norm_eps,
            group_size=self.hidden_inner_size // self.group_norm_size,
        )
        # use fp32 rotary embedding # 使用FP32旋转位置编码
        if hasattr(config, "rotary_dim"): # 如果有旋转维度配置
            rotary_dim = config.rotary_dim # 使用配置值
        elif hasattr(config, "partial_rotary_factor"): # 如果有部分旋转因子
            rotary_dim = int(self.head_dim * config.partial_rotary_factor) # 计算旋转维度
        else: # 否则
            rotary_dim = self.head_dim # 旋转维度等于头维度

        self.rotary_emb = get_rope_wrapper( # 获取旋转位置编码包装器
            self.head_dim, # 头维度
            rotary_dim=rotary_dim, # 旋转维度
            max_position=self.max_position_embeddings, # 最大位置
            base=self.rope_theta, # 基础频率
            rope_scaling=config.rope_scaling, # 旋转缩放
            is_neox_style=True, # Neox风格
            device=get_global_server_args().device, # 设备
            dtype=torch.float32, # 使用FP32精度
        )

    @staticmethod
    def weight_direct_load(param: torch.Tensor, loaded_weight: torch.Tensor) -> None: # 直接加载权重
        assert param.size() == loaded_weight.size() # 断言大小一致
        param.data.copy_(loaded_weight) # 复制数据
        return # 返回

    def forward( # 线性注意力前向传播
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        positions: torch.Tensor, # 位置张量
        forward_batch: ForwardBatch, # 前向批次
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states) # QKV投影
        qkv = qkv.to(torch.float32) # 转换为FP32
        if self.linear_silu: # 如果使用线性SiLU
            qkv = F.silu(qkv) # 应用SiLU

        q, k, v = torch.split( # 拆分QKV
            qkv,
            [self.q_size_per_rank, self.kv_size_per_rank, self.kv_size_per_rank],
            dim=-1,
        )
        if self.use_qk_norm: # 如果使用QK归一化
            q = q.reshape(-1, self.tp_heads, self.head_dim) # 重塑Q形状
            k = k.reshape(-1, self.tp_kv_heads, self.head_dim) # 重塑K形状
            if self.alt_stream is not None and get_is_capture_mode(): # 如果可以使用双流
                current_stream = torch.cuda.current_stream() # 获取当前流
                self.alt_stream.wait_stream(current_stream) # 等待当前流
                q = layernorm_fn( # Q归一化
                    q,
                    self.query_layernorm.weight.data,
                    bias=None,
                    eps=self.rms_norm_eps,
                    is_rms_norm=True,
                )
                with torch.cuda.stream(self.alt_stream): # 在备用流上K归一化
                    k = layernorm_fn(
                        k,
                        self.key_layernorm.weight.data,
                        bias=None,
                        eps=self.rms_norm_eps,
                        is_rms_norm=True,
                    )
                current_stream.wait_stream(self.alt_stream) # 等待备用流
            else: # 否则顺序执行
                q = layernorm_fn( # Q归一化
                    q,
                    self.query_layernorm.weight.data,
                    bias=None,
                    eps=self.rms_norm_eps,
                    is_rms_norm=True,
                )
                k = layernorm_fn( # K归一化
                    k,
                    self.key_layernorm.weight.data,
                    bias=None,
                    eps=self.rms_norm_eps,
                    is_rms_norm=True,
                )
            q = q.reshape(-1, self.q_size_per_rank) # 重塑Q形状
            k = k.reshape(-1, self.kv_size_per_rank) # 重塑K形状

        if self.linear_rope: # 如果线性注意力使用ROPE
            q, k = self.rotary_emb(positions, q, k) # 应用旋转位置编码

        q = q.view((qkv.shape[0], self.tp_heads, self.head_dim)) # 重塑Q为多头格式
        k = k.view((qkv.shape[0], self.tp_kv_heads, self.head_dim)) # 重塑K为多头格式
        v = v.view((qkv.shape[0], self.tp_kv_heads, self.head_dim)) # 重塑V为多头格式
        # logger.warning(f"===={self.layer_id=}, 1-2 {q.shape=}, {k.shape=}, {v.shape=}") # 注释掉的调试日志

        if self.linear_scale: # 如果使用线性缩放
            q = q * self.scaling # 缩放Q
        hidden = self.attn(q, k, v, forward_batch).to(hidden_states.dtype) # 计算注意力
        gate, _ = self.g_proj(hidden_states) # 计算门控

        if self.group_norm_size > 1: # 如果分组归一化大小大于1
            hidden = self.g_norm(hidden, gate) # 分组归一化（含门控）
        else: # 否则
            hidden = self.g_norm(hidden) # 简单分组归一化
            hidden = F.sigmoid(gate) * hidden # 应用sigmoid门控

        hidden = hidden.data.to(hidden_states.dtype) # 转换回原始数据类型
        hidden, _ = self.dense(hidden) # 输出投影

        return hidden # 返回输出


class BailingMoEAttention(nn.Module): # 百灵MoE注意力模块（softmax版本）

    def __init__( # 注意力初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        layer_id: int = None, # 层ID
        prefix: str = "mha", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.layer_id = layer_id # 保存层ID

        self.hidden_size = config.hidden_size # 隐藏层大小
        tp_size = get_attention_tp_size() # 获取注意力TP大小
        self.total_num_heads = config.num_attention_heads # 总注意力头数
        assert self.total_num_heads % tp_size == 0 # 断言总头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size # 每个TP rank的头数
        self.total_num_kv_heads = config.num_key_value_heads # 总KV头数
        if self.total_num_kv_heads >= tp_size: # 如果KV头数大于等于TP大小
            assert self.total_num_kv_heads % tp_size == 0 # 断言KV头数可被TP大小整除
        else: # 否则
            assert tp_size % self.total_num_kv_heads == 0 # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size) # 每个TP rank的KV头数
        self.head_dim = getattr(config, "head_dim", None) # 获取头维度
        if self.head_dim is None: # 如果头维度未指定
            self.head_dim = self.hidden_size // self.total_num_heads # 由隐藏大小和头数计算

        self.q_size = self.num_heads * self.head_dim # Q的大小
        self.kv_size = self.num_kv_heads * self.head_dim # KV的大小
        self.scaling = self.head_dim**-0.5 # 缩放因子

        self.split_qkv = getattr(config, "using_split_qkv_in_self_attention", False) # 是否使用拆分QKV
        assert not self.split_qkv, "split_qkv is not supported for now" # 断言不支持拆分QKV
        self.use_qk_norm = getattr(config, "use_qk_norm", False) # 是否使用QK归一化

        self.query_key_value = QKVParallelLinear( # QKV并行线性投影
            self.hidden_size, # 输入维度
            self.head_dim, # 头维度
            self.total_num_heads, # 总Q头数
            self.total_num_kv_heads, # 总KV头数
            bias=(config.use_bias or config.use_qkv_bias), # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.qkv_proj", # 参数前缀
        )
        if self.use_qk_norm: # 如果使用QK归一化
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) # Q层归一化
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) # K层归一化

        self.dense = RowParallelLinear( # 输出投影（dense层）
            self.total_num_heads * self.head_dim, # 输入维度
            self.hidden_size, # 输出维度
            bias=config.use_bias, # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.o_proj", # 参数前缀
        )
        if hasattr(config, "rotary_dim"): # 如果有旋转维度配置
            self.rotary_dim = config.rotary_dim # 使用配置值
        elif hasattr(config, "partial_rotary_factor"): # 如果有部分旋转因子
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor) # 计算旋转维度
        else: # 否则
            self.rotary_dim = self.head_dim # 旋转维度等于头维度
        self.max_position_embeddings = config.max_position_embeddings # 最大位置嵌入数
        self.rope_theta = getattr(config, "rope_theta", 600000) # 旋转theta
        self.rotary_emb = get_rope_wrapper( # 获取旋转位置编码包装器
            self.head_dim, # 头维度
            rotary_dim=self.rotary_dim, # 旋转维度
            max_position=self.max_position_embeddings, # 最大位置
            base=self.rope_theta, # 基础频率
            rope_scaling=config.rope_scaling, # 旋转缩放
            device=get_global_server_args().device, # 设备
        )
        self.attn = RadixAttention( # 基数注意力
            self.num_heads, # 注意力头数
            self.head_dim, # 头维度
            self.scaling, # 缩放因子
            num_kv_heads=self.num_kv_heads, # KV头数
            layer_id=layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.attn", # 参数前缀
        )

    def _apply_qk_norm( # 应用QK归一化
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.head_dim) # 重塑Q为每头格式
        q_by_head = self.query_layernorm(q_by_head) # Q层归一化
        q = q_by_head.view(q.shape) # 恢复原始形状
        k_by_head = k.reshape(-1, self.head_dim) # 重塑K为每头格式
        k_by_head = self.key_layernorm(k_by_head) # K层归一化
        k = k_by_head.view(k.shape) # 恢复原始形状
        return q, k # 返回归一化后的Q和K

    def forward( # 注意力前向传播
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        positions: torch.Tensor, # 位置张量
        forward_batch: ForwardBatch, # 前向批次
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states) # QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1) # 拆分QKV
        if self.use_qk_norm: # 如果使用QK归一化
            q, k = self._apply_qk_norm(q, k) # 应用QK归一化
        q, k = self.rotary_emb(positions, q, k) # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch) # 计算注意力
        output, _ = self.dense(attn_output) # 输出投影
        return output # 返回输出


class BailingMoELinearDecoderLayer(nn.Module): # 百灵MoE线性解码器层

    def __init__( # 解码器层初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        layer_id: int = 0, # 层ID
        prefix: str = "layer", # 参数前缀
        is_nextn: bool = False, # 是否为nextn层
        alt_stream=None, # 备用CUDA流
    ) -> None:
        super().__init__() # 调用父类初始化
        self.layer_id = layer_id # 保存层ID
        self.use_mla = getattr(config, "full_attention_type", "mla") == "mla" # 是否使用MLA

        if config.attention_type == 0:  # Linear layer # 线性注意力层
            self.attention = BailingMoELinearAttention( # 线性注意力
                config,
                quant_config=quant_config,
                layer_id=self.layer_id,
                prefix=prefix + ".attention",
                alt_stream=alt_stream,
            )
        elif config.attention_type == 1:  # softmax layer # softmax注意力层
            if self.use_mla: # 如果使用MLA
                self.attention = DsV3MLA( # DeepSeek V3 MLA注意力
                    config=config,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim, # QK非旋转头维度
                    qk_rope_head_dim=config.qk_rope_head_dim, # QK旋转头维度
                    v_head_dim=config.v_head_dim, # V头维度
                    q_lora_rank=( # Q LoRA秩
                        config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                    ),
                    kv_lora_rank=config.kv_lora_rank, # KV LoRA秩
                    rope_theta=getattr(config, "rope_theta", 600000), # 旋转theta
                    rope_scaling=config.rope_scaling, # 旋转缩放
                    max_position_embeddings=262144, # 最大位置嵌入
                    quant_config=quant_config,
                    layer_id=layer_id,
                    reduce_results=False, # 不归约结果
                    prefix=add_prefix("attention", prefix),
                    alt_stream=alt_stream,
                )
            else: # 否则使用GQA
                logger.debug(f"layer {layer_id} use gqa") # 记录使用GQA
                self.attention = BailingMoEAttention( # GQA注意力
                    config,
                    quant_config=quant_config,
                    layer_id=self.layer_id,
                    prefix=prefix + ".attention",
                )
        else: # 否则
            raise ValueError(f"Unsupported attention type: {config.attention_type}") # 不支持的注意力类型

        self.expert_num = config.num_experts # 专家数
        self.hidden_size = config.hidden_size # 隐藏层大小
        is_moe_layer = self._is_layer_sparse(config, self.layer_id) # 当前层是否为MoE层
        is_previous_moe_layer = self._is_layer_sparse(config, self.layer_id - 1) # 前一层是否为MoE层
        is_next_layer_moe_layer = self._is_layer_sparse(config, self.layer_id + 1) # 下一层是否为MoE层
        if self.expert_num == 1: # 如果只有1个专家
            self.mlp = BailingMLP( # 使用密集MLP
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else: # 否则
            if is_nextn or self.layer_id >= config.first_k_dense_replace: # 如果是nextn或超过密集替换层
                # MoE layer # MoE层
                self.mlp = BailingMoE( # 使用MoE
                    config,
                    quant_config=quant_config,
                    layer_id=self.layer_id,
                    prefix=add_prefix("mlp", prefix),
                    alt_stream=alt_stream,
                )
            else: # 否则
                # dense layer # 密集层
                self.mlp = BailingMLP( # 使用密集MLP
                    hidden_size=self.hidden_size,
                    intermediate_size=config.intermediate_size,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp", prefix),
                )
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-5)) # RMS归一化epsilon
        self.input_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps) # 输入层归一化
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps) # 注意力后层归一化

        self.layer_scatter_modes = LayerScatterModes.init_new( # 初始化层散射模式
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_moe_layer,
            is_previous_layer_sparse=is_previous_moe_layer,
            is_next_layer_sparse=is_next_layer_moe_layer,
        )

        qkv_latent_func = ( # QKV潜在函数
            self.attention.prepare_qkv_latent
            if config.attention_type == 1 and self.use_mla # 仅softmax注意力+MLA时使用
            else None
        )
        self.layer_communicator = LayerCommunicator( # 层通信器
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=False, # 不允许reduce-scatter
            qkv_latent_func=qkv_latent_func, # QKV潜在函数
        )

    def _is_layer_sparse( # 判断层是否为稀疏层
        self, config: PretrainedConfig, layer_id: int, is_nextn: bool = False
    ) -> bool:
        return is_nextn or ( # 如果是nextn层或者
            config.num_experts is not None and layer_id >= config.first_k_dense_replace # 层ID超过第一个密集替换层
        )

    def forward( # 解码器层前向传播
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        positions: torch.Tensor, # 位置张量
        forward_batch: ForwardBatch, # 前向批次
        residual: Optional[torch.Tensor], # 残差连接
        zero_allocator: BumpAllocator, # 零分配器
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.layer_communicator.prepare_attn( # 准备注意力输入
            hidden_states, residual, forward_batch
        )
        # logger.warning( # 注释掉的调试日志
        #     f"===={self.layer_id=}, 1 shape= {hidden_states.shape}, {residual.shape}"
        # )
        if not forward_batch.forward_mode.is_idle(): # 如果不是空闲模式
            if self.use_mla: # 如果使用MLA
                hidden_states = self.attention( # MLA注意力计算
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    zero_allocator=zero_allocator,
                )
            else: # 否则使用GQA
                hidden_states = self.attention( # GQA注意力计算
                    hidden_states=hidden_states,
                    positions=positions,
                    forward_batch=forward_batch,
                )
        # logger.warning( # 注释掉的调试日志
        #     f"===={self.layer_id=}, 2 shape= {hidden_states.shape}, {residual.shape}"
        # )
        hidden_states, residual = self.layer_communicator.prepare_mlp( # 准备MLP输入
            hidden_states, residual, forward_batch
        )
        # logger.warning( # 注释掉的调试日志
        #     f"===={self.layer_id=}, 3 shape= {hidden_states.shape}, {residual.shape}"
        # )
        should_allreduce_fusion = ( # 是否融合全归约
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter( # 是否使用reduce-scatter
            forward_batch
        )
        hidden_states = self.mlp( # MLP前向传播
            hidden_states, should_allreduce_fusion, use_reduce_scatter
        )
        hidden_states, residual = self.layer_communicator.postprocess_layer( # 层后处理
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual # 返回隐藏状态和残差

    @staticmethod
    def shared_moe_coefficient_loader( # 共享MoE系数加载器
        param: torch.Tensor, loaded_weight: torch.Tensor
    ) -> None:
        assert param.size() == loaded_weight.size() # 断言大小一致

        param.data.copy_(loaded_weight.to(torch.float32)) # 复制并转换为FP32
        return # 返回


class BailingMoELinearModel(nn.Module): # 百灵MoE线性模型主体

    def __init__( # 模型初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.pp_group = get_pp_group() # 获取流水线并行组
        self.config = config # 保存配置
        self.vocab_size = config.vocab_size # 词表大小
        self.embed_dim = config.hidden_size # 嵌入维度
        self.num_layers = config.num_hidden_layers # 隐藏层数

        self.layer_group_size = getattr(config, "layer_group_size", 1) # 层分组大小
        self.decoder_attention_types = [ # 每层的注意力类型
            0 if is_linear_layer(i, self.layer_group_size) else 1 # 0=线性注意力，1=softmax注意力
            for i in range(self.num_layers)
        ]
        num_linear = sum(1 for t in self.decoder_attention_types if t == 0) # 线性注意力层数
        num_full = sum(1 for t in self.decoder_attention_types if t == 1) # 全注意力层数
        rank0_log( # 记录层配置信息
            f"Layer config: {num_linear} linear attention layers, {num_full} full attention layers"
        )

        assert ( # 断言层数可被分组大小整除
            self.num_layers % self.layer_group_size == 0
        ), f"num_layers={self.num_layers} must be divided by layer_group_size={self.layer_group_size}"

        if self.pp_group.is_first_rank: # 如果是第一个rank
            self.word_embeddings = VocabParallelEmbedding( # 词嵌入层
                self.vocab_size,
                self.embed_dim,
                enable_tp=not is_dp_attention_enabled(), # 是否启用TP
                org_num_embeddings=self.vocab_size, # 原始嵌入数量
            )
        else: # 否则
            self.word_embeddings = PPMissingLayer() # 使用缺失层占位

        self.alt_stream = torch.cuda.Stream() if _is_cuda else None # 创建备用CUDA流

        def layer_fn(idx, prefix): # 层工厂函数
            layer_idx = idx # 层索引
            layer_config = copy.deepcopy(config) # 深拷贝配置
            layer_config.attention_type = self.decoder_attention_types[layer_idx] # 设置注意力类型

            decoder_kwargs = {"quant_config": quant_config, "layer_id": layer_idx} # 解码器参数
            return BailingMoELinearDecoderLayer( # 创建解码器层
                layer_config,
                **decoder_kwargs,
                prefix=prefix,
                alt_stream=self.alt_stream,
            )

        self.layers, self.start_layer, self.end_layer = make_layers( # 创建解码器层
            self.num_layers, # 层数
            layer_fn, # 层工厂函数
            pp_rank=self.pp_group.rank_in_group, # 流水线并行rank
            pp_size=self.pp_group.world_size, # 流水线并行大小
            prefix=f"{prefix}.layers", # 参数前缀
        )

        norm_kwargs = {} # 归一化参数
        if hasattr(config, "rms_norm_eps"): # 如果有RMS归一化epsilon
            norm_kwargs["eps"] = config.rms_norm_eps # 设置epsilon
        if self.pp_group.is_last_rank: # 如果是最后一个rank
            self.norm = RMSNorm(config.hidden_size, **norm_kwargs) # 最终层归一化
        else: # 否则
            self.norm = PPMissingLayer() # 使用缺失层占位
        self.embed_scale = 1.0 # 嵌入缩放因子
        return # 返回

    def forward( # 模型前向传播
        self,
        input_ids: Optional[torch.Tensor], # 输入token ID
        positions: torch.Tensor, # 位置张量
        forward_batch: Optional[ForwardBatch] = None, # 前向批次
        inputs_embeds: Optional[torch.Tensor] = None, # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None, # 流水线代理张量
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank: # 如果是第一个rank
            if inputs_embeds is None: # 如果没有输入嵌入
                hidden_states = self.word_embeddings(input_ids) # 通过词嵌入获取隐藏状态
            else: # 否则
                hidden_states = inputs_embeds # 直接使用输入嵌入
            residual = None # 初始化残差为空
        else: # 否则
            assert pp_proxy_tensors is not None # 断言代理张量不为空
            hidden_states = pp_proxy_tensors["hidden_states"] # 从代理获取隐藏状态
            residual = pp_proxy_tensors["residual"] # 从代理获取残差

        total_num_layers = self.end_layer - self.start_layer # 当前PP范围内的层数
        device = inputs_embeds.device if inputs_embeds is not None else input_ids.device # 设备
        zero_allocator = BumpAllocator( # 零分配器（用于MLA）
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1), # 缓冲区大小
            dtype=torch.float32, # FP32精度
            device=device, # 设备
        )

        for i in range(self.start_layer, self.end_layer): # 遍历每一层
            with get_global_expert_distribution_recorder().with_current_layer(i): # 记录专家分布
                layer = self.layers[i] # 获取当前层
                hidden_states, residual = layer( # 前向传播当前层
                    hidden_states=hidden_states,
                    positions=positions,
                    forward_batch=forward_batch,
                    residual=residual,
                    zero_allocator=zero_allocator,
                )
        if not self.pp_group.is_last_rank: # 如果不是最后一个rank
            return PPProxyTensors( # 返回代理张量
                {"hidden_states": hidden_states, "residual": residual}
            )
        else: # 否则
            if not forward_batch.forward_mode.is_idle(): # 如果不是空闲模式
                if residual is None: # 如果没有残差
                    hidden_states = self.norm(hidden_states) # 层归一化
                else: # 否则
                    hidden_states, _ = self.norm(hidden_states, residual) # 带残差的层归一化
            return hidden_states # 返回隐藏状态


class BailingMoELinearForCausalLM(nn.Module): # 百灵MoE线性因果语言模型

    packed_modules_mapping = { # 打包模块映射
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"], # 融合QKV投影映射
        "gate_up_proj": ["gate_proj", "up_proj"], # gate和up投影映射
    }
    # To ensure correct weight loading and mapping. # 确保正确的权重加载和映射
    hf_to_sglang_mapper = WeightsMapper( # HuggingFace到SGLang权重映射器
        orig_to_new_substr={
            "attention.dense": "attention.out_proj", # dense到out_proj
            "layers.7.attention.out_proj": "layers.7.attention.o_proj", # 第7层输出投影
            "layers.15.attention.out_proj": "layers.15.attention.o_proj", # 第15层输出投影
            "layers.23.attention.out_proj": "layers.23.attention.o_proj", # 第23层输出投影
            "layers.31.attention.out_proj": "layers.31.attention.o_proj", # 第31层输出投影
            "layers.39.attention.out_proj": "layers.39.attention.o_proj", # 第39层输出投影
            "layers.47.attention.out_proj": "layers.47.attention.o_proj", # 第47层输出投影
            "layers.55.attention.out_proj": "layers.55.attention.o_proj", # 第55层输出投影
            "layers.63.attention.out_proj": "layers.63.attention.o_proj", # 第63层输出投影
            "layers.71.attention.out_proj": "layers.71.attention.o_proj", # 第71层输出投影
            "layers.79.attention.out_proj": "layers.79.attention.o_proj", # 第79层输出投影
            "attention.query_key_value": "attention.qkv_proj", # QKV投影映射
            "attention.g_proj": "attention.output_gate", # g_proj到output_gate
        },
    )

    def __init__( # 因果语言模型初始化方法
        self,
        *,
        config, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.pp_group = get_pp_group() # 获取流水线并行组
        self.config = config # 保存配置
        self.quant_config = quant_config # 保存量化配置
        self.model = BailingMoELinearModel( # 百灵MoE线性模型
            self.config, quant_config, prefix=add_prefix("model", prefix)
        )

        if self.pp_group.is_last_rank: # 如果是最后一个rank
            self.lm_head = ( # 语言模型头
                self.word_embeddings # 复用词嵌入
                if config.tie_word_embeddings # 如果绑定词嵌入
                else ParallelLMHead( # 独立的语言模型头
                    config.vocab_size, # 词表大小
                    config.hidden_size, # 隐藏层大小
                    params_dtype=torch.float32, # FP32精度
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head, # 是否使用注意力TP组
                )
            )
            self.logits_processor = LogitsProcessor(config) # logits处理器
        else: # 否则
            self.lm_head = PPMissingLayer() # 使用缺失层占位

    @property
    def start_layer(self): # 起始层属性
        return self.model.start_layer # 返回模型的起始层

    @property
    def end_layer(self): # 结束层属性
        return self.model.end_layer # 返回模型的结束层

    def get_embed_and_head(self): # 获取嵌入和语言模型头权重
        """Used by the eagle_worker.""" # 由eagle_worker使用
        return self.model.word_embeddings.weight, self.lm_head.weight # 返回嵌入权重和头权重

    def post_load_weights(self, is_nextn=False, weight_names=None): # 权重加载后处理

        # Perform post-processing after loading weights # 加载权重后执行后处理
        if is_nextn: # 如果是nextn模式
            layer_ids = [self.config.num_hidden_layers] # nextn层ID
        else: # 否则
            if weight_names is None: # 如果未指定权重名称
                layer_ids = range(self.model.start_layer, self.model.end_layer) # 处理所有层
            else: # 否则
                layer_ids = set() # 初始化层ID集合
                for name in weight_names: # 遍历权重名称
                    if "kv_b_proj" in name: # 如果包含kv_b_proj
                        layer_id = int(name.split(".")[2]) # 提取层ID
                        if ( # 如果层在当前PP范围内
                            layer_id < self.model.end_layer
                            and layer_id >= self.model.start_layer
                        ):
                            layer_ids.add(layer_id) # 添加层ID
        logger.debug(f"weight loading layer_ids: {layer_ids}") # 记录日志

        for layer_id in layer_ids: # 遍历层ID
            self_attn = ( # 获取自注意力层
                self.model.layers[layer_id].attention
                if not is_nextn # 非nextn模式
                else self.model.decoder.attention # nextn模式
            )
            if not hasattr(self_attn, "kv_b_proj"): # 如果没有kv_b_proj
                continue # 跳过
            if hasattr(self_attn.kv_b_proj, "qweight"): # 如果是AWQ量化
                # AWQ compatible # AWQ兼容
                if _is_cuda or _is_hip: # CUDA或HIP环境
                    w = awq_dequantize( # AWQ反量化
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                    ).T # 转置
                else: # 否则
                    w = awq_dequantize( # AWQ反量化（其他平台）
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                        0,
                        0,
                        0,
                    ).T # 转置
            else: # 否则
                w = self_attn.kv_b_proj.weight # 直接使用权重
            # NOTE(HandH1998): Since `bmm_fp8` only supports per-tensor scale, we have to requantize `self_attn.kv_b_proj`. # 注意：由于bmm_fp8仅支持逐张量缩放，我们必须对kv_b_proj重量化
            # This may affect the accuracy of fp8 model. # 这可能影响FP8模型的精度
            # Fix deepseek v3 blockwise bmm by using deep_gemm # 使用deep_gemm修复DeepSeek V3块级bmm
            use_deep_gemm_bmm = False # 是否使用DeepGEMM bmm

            if w.dtype in ( # 如果是FP8权重
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ):
                if ( # 如果有块级量化配置
                    hasattr(self.quant_config, "weight_block_size")
                    and self.quant_config.weight_block_size is not None
                ):
                    weight_block_size = self.quant_config.weight_block_size # 获取权重块大小
                    assert hasattr(self_attn.kv_b_proj, "weight_scale_inv") # 断言有缩放因子
                    if _is_fp8_fnuz: # 如果是FNUZ格式
                        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz( # 归一化FP8格式
                            weight=w,
                            weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                            input_scale=None,
                        )
                    else: # 否则
                        weight = w # 直接使用权重
                        weight_scale = self_attn.kv_b_proj.weight_scale_inv # 使用缩放因子

                    if ( # 如果是CUDA且块大小为128x128
                        _is_cuda
                        and weight_block_size[0] == 128
                        and weight_block_size[1] == 128
                    ):
                        if ( # 如果启用JIT DeepGEMM且非Blackwell
                            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                            and not deep_gemm_wrapper.DEEPGEMM_BLACKWELL
                            and get_bool_env_var("SGL_USE_DEEPGEMM_BMM", "false")
                        ):
                            block_scale = weight_scale # 使用块缩放
                            use_deep_gemm_bmm = True # 启用DeepGEMM bmm
                        else: # 否则
                            w = block_quant_dequant( # 块量化反量化
                                weight,
                                weight_scale,
                                weight_block_size,
                                torch.bfloat16,
                            )
                    else: # 否则
                        w, scale = block_quant_to_tensor_quant( # 块量化到张量量化
                            weight, weight_scale, weight_block_size
                        )
                        self_attn.w_scale = scale # 设置权重缩放
                else: # 否则（无块级量化）
                    if _is_fp8_fnuz: # 如果是FNUZ格式
                        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz( # 归一化FP8格式
                            weight=w,
                            weight_scale=self_attn.kv_b_proj.weight_scale,
                            input_scale=None,
                        )
                    else: # 否则
                        weight = w # 直接使用权重
                        weight_scale = self_attn.kv_b_proj.weight_scale # 使用缩放因子

                    w, scale = channel_quant_to_tensor_quant(weight, weight_scale) # 通道量化到张量量化
                    self_attn.w_scale = scale # 设置权重缩放

            if w.dtype == torch.int8: # 如果是INT8权重
                if hasattr(self.quant_config, "weight_block_size"): # 如果有块级量化配置
                    # block-wise int8 need it # 块级INT8需要此处理
                    weight_block_size = self.quant_config.weight_block_size # 获取权重块大小
                    if weight_block_size is not None: # 如果块大小不为空
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv") # 断言有缩放因子
                        weight = w # 保存权重
                        weight_scale = self_attn.kv_b_proj.weight_scale_inv # 获取缩放因子
                        w = int8_block_dequant( # INT8块反量化
                            weight, weight_scale, weight_block_size
                        ).to(torch.bfloat16) # 转换为bfloat16
                else: # 否则
                    # channel-wise int8 need it # 通道级INT8需要此处理
                    w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to( # 乘以缩放因子
                        torch.bfloat16
                    )

            w_kc, w_vc = w.unflatten( # 拆分为K和V的权重
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            if not use_deep_gemm_bmm: # 如果不使用DeepGEMM bmm
                self_attn.w_kc = bind_or_assign( # 绑定或赋值K权重
                    self_attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                )
                self_attn.w_vc = bind_or_assign( # 绑定或赋值V权重
                    self_attn.w_vc, w_vc.contiguous().transpose(1, 2)
                )
                if ( # 如果有权重缩放但w_scale未设置
                    hasattr(self_attn.kv_b_proj, "weight_scale")
                    and self_attn.w_scale is None
                ):
                    self_attn.w_scale = bind_or_assign( # 绑定或赋值权重缩放
                        self_attn.w_scale, self_attn.kv_b_proj.weight_scale
                    )
                    if _is_hip: # 如果是HIP环境
                        self_attn.w_scale *= 2.0 # 缩放因子乘2
            else: # 否则（使用DeepGEMM bmm）
                num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[1] # K的tile数
                num_tiles_n = self_attn.v_head_dim // weight_block_size[0] # V的tile数
                ws_kc, ws_vc = block_scale.unflatten( # 拆分块缩放为K和V
                    0, (-1, (num_tiles_k + num_tiles_n))
                ).split([num_tiles_k, num_tiles_n], dim=1)
                self_attn.w_scale_k = bind_or_assign( # 绑定或赋值K缩放
                    self_attn.w_scale_k, ws_kc.transpose(1, 2).contiguous()
                )
                self_attn.w_scale_v = bind_or_assign( # 绑定或赋值V缩放
                    self_attn.w_scale_v, ws_vc.contiguous()
                )
                self_attn.w_kc = bind_or_assign( # 绑定或赋值K权重
                    self_attn.w_kc, w_kc.transpose(1, 2).contiguous()
                )
                self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc.contiguous()) # 绑定或赋值V权重
                self_attn.use_deep_gemm_bmm = True # 标记使用DeepGEMM bmm

        if ( # 如果需要UE8M0重量化
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and hasattr(self.quant_config, "weight_block_size")
            and self.quant_config.weight_block_size is not None
        ):
            self._weight_requant_ue8m0(is_nextn) # 执行UE8M0重量化

    @classmethod
    def get_model_config_for_expert_location(cls, config): # 获取专家位置的模型配置
        num_groups = getattr(config, "n_group", 0) # 获取分组数
        from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation # 导入专家位置模型配置

        return ModelConfigForExpertLocation( # 返回专家位置模型配置
            num_layers=config.num_hidden_layers, # 隐藏层数
            num_logical_experts=config.num_experts, # 逻辑专家数
            num_groups=None if num_groups == 0 else num_groups, # 分组数
        )

    def _weight_requant_ue8m0(self, is_nextn=False): # UE8M0权重重量化
        weight_block_size = self.quant_config.weight_block_size # 获取权重块大小

        moe_layers = list( # MoE层列表
            range(
                self.config.first_k_dense_replace, # 第一个MoE层
                self.config.num_hidden_layers, # 到最后一层
                self.config.moe_layer_freq, # MoE层频率
            )
        )

        num_hidden_layers = 1 if is_nextn else self.config.num_hidden_layers # 处理的层数

        for layer_id in range(num_hidden_layers): # 遍历每一层
            if is_nextn: # 如果是nextn模式
                layer = self.model.decoder # 获取解码器层
            else: # 否则
                layer = self.model.layers[layer_id] # 获取对应层

            module_list = [ # 需要重量化的模块列表
                layer.self_attn.kv_b_proj, # KV投影
                layer.self_attn.o_proj, # 输出投影
            ]

            if self.config.q_lora_rank is not None: # 如果使用Q LoRA
                module_list.append(layer.self_attn.fused_qkv_a_proj_with_mqa) # 融合QKV投影
                module_list.append(layer.self_attn.q_b_proj) # Q B投影
            else: # 否则
                module_list.append(layer.self_attn.kv_a_proj_with_mqa) # KV A投影
                module_list.append(layer.self_attn.q_proj) # Q投影

            for module in module_list: # 遍历模块
                requant_weight_ue8m0_inplace( # UE8M0原位重量化
                    module.weight, module.weight_scale_inv, weight_block_size
                )

            if layer_id in moe_layers or is_nextn: # 如果是MoE层或nextn
                shared_experts = getattr(layer.mlp, "shared_experts", None) # 获取共享专家
                if shared_experts is not None: # 如果有共享专家
                    for module in [ # 遍历共享专家模块
                        shared_experts.gate_up_proj, # gate和up投影
                        shared_experts.down_proj, # 下投影
                    ]:
                        requant_weight_ue8m0_inplace( # UE8M0原位重量化
                            module.weight, module.weight_scale_inv, weight_block_size
                        )

                experts = layer.mlp.experts # 获取专家模块
                if isinstance(experts, DeepEPMoE): # 如果是DeepEP MoE
                    for w in [ # 遍历FP8权重
                        experts.w13_weight_fp8, # w13权重
                        experts.w2_weight_fp8, # w2权重
                    ]:
                        requant_weight_ue8m0_inplace(w[0], w[1], weight_block_size) # UE8M0原位重量化
            else: # 否则（密集MLP层）
                mlp = layer.mlp # 获取MLP
                assert isinstance(mlp, DeepseekV2MLP) # 断言是DeepSeek V2 MLP
                for module in [ # 遍历MLP模块
                    mlp.gate_up_proj, # gate和up投影
                    mlp.down_proj, # 下投影
                ]:
                    requant_weight_ue8m0_inplace( # UE8M0原位重量化
                        module.weight, module.weight_scale_inv, weight_block_size
                    )

    def get_decoder_attention_types(self): # 获取解码器注意力类型列表
        return self.model.decoder_attention_types # 返回注意力类型列表

    def forward( # 因果语言模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置张量
        forward_batch: ForwardBatch, # 前向批次
        inputs_embeds: Optional[torch.Tensor] = None, # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None, # 流水线代理张量
    ) -> Union[torch.Tensor, PPProxyTensors]:
        hidden_states = self.model( # 通过模型获取隐藏状态
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank: # 如果是最后一个rank
            return self.logits_processor( # 通过logits处理器返回
                input_ids, hidden_states.float(), self.lm_head, forward_batch
            )
        else: # 否则
            return hidden_states # 返回隐藏状态

    def load_weights( # 加载权重
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False
    ) -> Set[str]:
        def load_linear_attn_weight( # 加载线性注意力权重
            name: str, loaded_weight: torch.Tensor, self
        ) -> None:
            if is_pp_missing_parameter(name, self): # 如果是PP缺失参数
                return # 跳过
            param = params_dict[name] # 获取参数
            weight_loader = getattr( # 获取权重加载器
                param, "weight_loader", BailingMoELinearAttention.weight_direct_load
            )
            weight_loader = weight_loader_with_alias(name)(weight_loader) # 包装加载器
            weight_loader(param, loaded_weight) # 加载权重
            return # 返回

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
                raise ValueError("num nextn_predict_layers is not in the config") # 抛出配置错误

        stacked_params_mapping = [ # 堆叠参数映射
            # (param_name, shard_name, shard_id) # （参数名，分片名，分片ID）
            ("gate_up_proj", "gate_proj", 0), # gate投影映射
            ("gate_up_proj", "up_proj", 1), # up投影映射
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping( # 创建专家参数映射
            ckpt_gate_proj_name="gate_proj", # 检查点gate投影名
            ckpt_down_proj_name="down_proj", # 检查点down投影名
            ckpt_up_proj_name="up_proj", # 检查点up投影名
            num_experts=self.config.num_experts, # 专家数
        )

        if is_nextn: # 如果是nextn模式
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}" # nextn层前缀
            nextn_spec_weight_names = [ # nextn特有权重名称
                "final_layernorm", # 最终层归一化
                "eh_proj", # eh投影
                "enorm", # enorm
                "hnorm", # hnorm
            ]

        params_dict = dict(self.named_parameters()) # 获取参数字典
        loaded_params: Set[str] = set() # 已加载参数集合
        weight_names = [] # 权重名称列表
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and ( # 是否融合QKV A投影
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None # 缓存的A投影权重

        for name, loaded_weight in weights: # 遍历所有权重
            if name.startswith("model.mtp"): # 跳过MTP权重
                continue
            layer_idx = None # 层索引初始化
            if "model.layers." in name: # 如果名称包含层前缀
                layer_idx = int(name.split(".")[2]) # 提取层索引
            if ( # 跳过不需要的权重
                ("v_head" in name) # v_head
                or ("inv_freq" in name) # 逆频率
                or (self.config.tie_word_embeddings and "lm_head" in name) # 绑定嵌入时的lm_head
            ):
                continue # 跳过

            weight_names.append(name) # 记录权重名称

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
                if "mlp.experts" in name: # 如果名称包含专家
                    continue # 跳过

                name = name.replace(weight_name, param_name) # 替换权重名为参数名
                if name.endswith(".bias") and name not in params_dict: # 如果是偏置且不在参数字典中
                    continue # 跳过
                if name not in params_dict: # 如果参数名不在参数字典中
                    continue # 跳过
                if is_pp_missing_parameter(name, self): # 如果是PP缺失参数
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
                    if is_pp_missing_parameter(name, self): # 如果是PP缺失参数
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

                    if name.endswith(".bias") and name not in params_dict: # 如果是偏置且不在参数字典中
                        continue # 跳过
                    if "slope" in name: # 如果名称包含slope
                        continue # 跳过

                    if fuse_qkv_a_proj and ( # 如果融合QKV A投影且
                        "q_a_proj" in name or "kv_a_proj_with_mqa" in name # 名称包含q_a_proj或kv_a_proj_with_mqa
                    ):
                        cached_a_proj[name] = loaded_weight # 缓存A投影权重
                        q_a_proj_name = ( # Q A投影名称
                            name
                            if "q_a_proj" in name
                            else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                        )
                        kv_a_proj_name = ( # KV A投影名称
                            name
                            if "kv_a_proj_with_mqa" in name
                            else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                        )

                        # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter # 当q_a_proj和kv_a_proj_with_mqa都被缓存后，加载融合权重到参数
                        if ( # 如果两个投影都已缓存
                            q_a_proj_name in cached_a_proj
                            and kv_a_proj_name in cached_a_proj
                        ):
                            q_a_proj_weight = cached_a_proj[q_a_proj_name] # 获取Q A投影权重
                            kv_a_proj_weight = cached_a_proj[kv_a_proj_name] # 获取KV A投影权重
                            cat_dim = 0 # 拼接维度
                            if self.quant_config is not None and ( # 如果是AWQ量化
                                self.quant_config.get_name() == "awq"
                                or self.quant_config.get_name() == "awq_marlin"
                                or self.quant_config.get_name() == "moe_wna16"
                            ):
                                cat_dim = 1 # AWQ使用维度1拼接
                            fused_weight = torch.cat( # 拼接融合权重
                                [q_a_proj_weight, kv_a_proj_weight], dim=cat_dim
                            )
                            param_name = ( # 融合参数名
                                name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                                if "q_a_proj" in name
                                else name.replace(
                                    "kv_a_proj_with_mqa",
                                    "fused_qkv_a_proj_with_mqa",
                                )
                            )
                            if param_name not in params_dict: # 如果参数名不在参数字典中
                                continue # 跳过
                            param = params_dict[param_name] # 获取参数
                            weight_loader = getattr( # 获取权重加载器
                                param, "weight_loader", default_weight_loader
                            )

                            weight_loader(param, fused_weight) # 加载融合权重
                            cached_a_proj.pop(q_a_proj_name) # 移除缓存
                            cached_a_proj.pop(kv_a_proj_name) # 移除缓存
                    else: # 否则（非融合QKV A投影）

                        if name not in params_dict: # 如果参数名不在参数字典中
                            name = name.replace(".dense.", ".o_proj.") # 尝试替换dense为o_proj
                            if name not in params_dict: # 如果仍然不在
                                continue # 跳过
                        if is_pp_missing_parameter(name, self): # 如果是PP缺失参数
                            continue # 跳过
                        if ( # 如果是线性注意力层的权重
                            "attention" in name
                            and "slope" not in name
                            and is_linear_layer(layer_idx, self.model.layer_group_size)
                        ):
                            load_linear_attn_weight(name, loaded_weight, self) # 加载线性注意力权重
                            loaded_params.add(name) # 添加到已加载集合
                            continue # 跳过

                        param = params_dict[name] # 获取参数
                        weight_loader = getattr( # 获取权重加载器
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight) # 加载权重
            loaded_params.add(name) # 添加到已加载集合
        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names) # 权重加载后处理

        return loaded_params # 返回已加载参数集合


class BailingMoeV2_5ForCausalLM(BailingMoELinearForCausalLM): # 百灵MoE V2.5因果语言模型（别名）
    pass # 无额外实现


EntryClass = [ # 入口类列表
    BailingMoeV2_5ForCausalLM,
]
