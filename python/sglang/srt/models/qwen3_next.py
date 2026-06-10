# Qwen3-Next 混合线性注意力模型实现
# 本文件实现了 Qwen3-Next 模型，该模型采用混合架构，结合了全注意力（full attention）
# 和门控 DeltaNet 线性注意力（linear attention），并使用 MoE（混合专家）MLP。
# 支持推测解码（EAGLE3）和 DFLASH 辅助隐藏状态捕获。
import enum  # 导入枚举模块
import logging  # 导入日志模块
from typing import Any, Iterable, Optional, Set, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 框架
import triton  # 导入 Triton 编译器
from torch import nn  # 导入神经网络模块

from sglang.srt.configs.qwen3_next import Qwen3NextConfig  # 导入 Qwen3-Next 配置
from sglang.srt.distributed import get_pp_group  # 导入流水线并行组
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入专家分布记录器
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation  # 导入专家位置模型配置
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated  # 导入门控 RMS 归一化
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader  # 导入 Mamba v2 分片权重加载器
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes  # 导入层通信器和散射模式
from sglang.srt.layers.dp_attention import (  # 导入数据并行注意力相关函数
    get_attention_tp_rank,  # 获取注意力 TP 排名
    get_attention_tp_size,  # 获取注意力 TP 大小
    is_dp_attention_enabled,  # 是否启用 DP 注意力
)
from sglang.srt.layers.layernorm import GemmaRMSNorm  # 导入 Gemma RMS 归一化
from sglang.srt.layers.linear import (  # 导入线性层
    ColumnParallelLinear,  # 列并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV 并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合 MoE 层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入 Radix 注意力
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention  # 导入 Radix 线性注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 并行词表嵌入
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode  # 导入 CUDA Graph 捕获模式检查
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    sharded_weight_loader,  # 分片权重加载器
)
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock  # 导入 Qwen2 MoE MLP 和稀疏 MoE 块
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import (  # 导入工具函数
    LazyValue,  # 懒加载值
    add_prefix,  # 添加前缀
    cpu_has_amx_support,  # CPU AMX 支持检查
    is_cpu,  # CPU 检测
    is_cuda,  # CUDA 检测
    is_npu,  # NPU 检测
    make_layers,  # 创建层
    set_weight_attrs,  # 设置权重属性
)

logger = logging.getLogger(__name__)  # 获取日志记录器

from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkvzba_split_reshape_cat  # 导入 GDN 融合投影内核
from sglang.srt.layers.attention.fla.fused_norm_gate import FusedRMSNormGated  # 导入融合归一化门控

_is_cuda = is_cuda()  # 是否为 CUDA 设备
_is_npu = is_npu()  # 是否为 NPU 设备
_is_cpu = is_cpu()  # 是否为 CPU 设备
_is_amx_available = cpu_has_amx_support()  # CPU 是否支持 AMX


if _is_npu:  # 如果是 NPU 设备
    from sgl_kernel_npu.fla.utils import (  # 导入 NPU FLA 工具
        fused_qkvzba_split_reshape_cat as fused_qkvzba_split_reshape_cat_npu,  # NPU 版本
    )
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import (  # 导入 NPU 分割归一化 RoPE 内核
        split_qkvgate_gemma_rmsnorm_rope,  # 分割 QKV 门控 Gemma RMS 归一化 RoPE
    )

    fused_qkvzba_split_reshape_cat = fused_qkvzba_split_reshape_cat_npu  # 替换为 NPU 版本


class Qwen3GatedDeltaNet(nn.Module):
    """Qwen3 门控 DeltaNet 线性注意力模块"""

    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ) -> None:
        """初始化门控 DeltaNet 模块"""
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.attn_tp_rank = get_attention_tp_rank()  # 注意力 TP 排名
        self.attn_tp_size = get_attention_tp_size()  # 注意力 TP 大小
        self.hidden_size = config.hidden_size  # 隐藏维度
        self.num_v_heads = (  # 值头数
            config.linear_num_value_heads  # GPU 上使用配置值
            if not _is_cpu  # 非CPU
            else config.linear_num_value_heads_cpu  # CPU 上使用 CPU 配置值
        )
        self.num_k_heads = (  # 键头数
            config.linear_num_key_heads  # GPU 上使用配置值
            if not _is_cpu  # 非CPU
            else config.linear_num_key_heads_cpu  # CPU 上使用 CPU 配置值
        )
        self.head_k_dim = config.linear_key_head_dim  # 键头维度
        self.head_v_dim = config.linear_value_head_dim  # 值头维度
        self.key_dim = self.head_k_dim * self.num_k_heads  # 键总维度
        self.value_dim = self.head_v_dim * self.num_v_heads  # 值总维度
        self.alt_stream = alt_stream  # 替代 CUDA 流

        self.conv_kernel_size = config.linear_conv_kernel_dim  # 卷积核大小
        self.layer_id = layer_id  # 层 ID
        self.activation = config.hidden_act  # 激活函数
        self.output_gate_type = config.output_gate_type  # 输出门控类型
        self.layer_norm_epsilon = config.rms_norm_eps  # 归一化 epsilon

        self.conv_dim = self.key_dim * 2 + self.value_dim  # 卷积维度
        self.conv1d = ColumnParallelLinear(  # 一维卷积（实现为列并行线性层）
            input_size=self.conv_kernel_size,  # 输入大小（卷积核大小）
            output_size=self.conv_dim,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=None,  # 卷积不量化
            tp_rank=self.attn_tp_rank,  # TP 排名
            tp_size=self.attn_tp_size,  # TP 大小
            prefix=add_prefix("conv1d", prefix),  # 参数前缀
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)  # 增加 kernel 维度

        # projection of the input hidden states  # 输入隐藏状态投影
        self.in_proj_qkvz = self.create_qkvz_proj(  # QKV+Z 投影
            hidden_size=self.hidden_size,  # 隐藏维度
            key_dim=self.key_dim,  # 键维度
            value_dim=self.value_dim,  # 值维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("in_proj_qkvz", prefix),  # 参数前缀
            tp_rank=self.attn_tp_rank,  # TP 排名
            tp_size=self.attn_tp_size,  # TP 大小
        )

        self.in_proj_ba = MergedColumnParallelLinear(  # B+A 投影
            input_size=self.hidden_size,  # 隐藏维度
            output_sizes=[self.num_v_heads] * 2,  # B 和 A 各一个输出
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("in_proj_ba", prefix),  # 参数前缀
            tp_rank=self.attn_tp_rank,  # TP 排名
            tp_size=self.attn_tp_size,  # TP 大小
        )

        # Override weight_loader for packed checkpoint format.  # 覆盖权重加载器以支持打包检查点格式
        # Must capture original_loader BEFORE overwriting.  # 必须在覆盖前捕获原始加载器
        self._override_weight_loader(  # 覆盖 QKV+Z 投影的权重加载器
            self.in_proj_qkvz, self._make_packed_weight_loader(self.in_proj_qkvz)  # 使用打包权重加载器
        )
        self._override_weight_loader(  # 覆盖 B+A 投影的权重加载器
            self.in_proj_ba, self._make_packed_weight_loader(self.in_proj_ba)  # 使用打包权重加载器
        )

        # Conv1d weight loader setup  # 一维卷积权重加载器设置
        query_key_settings = (self.key_dim, 0, False)  # 查询键设置
        value_settings = (self.value_dim, 0, False)  # 值设置

        delattr(self.conv1d.weight, "weight_loader")  # 删除原始权重加载器
        set_weight_attrs(  # 设置权重属性
            self.conv1d.weight,  # 卷积权重
            {
                "weight_loader": mamba_v2_sharded_weight_loader(  # Mamba v2 分片权重加载器
                    [  # 分片配置列表
                        query_key_settings,  # 查询键设置
                        query_key_settings,  # 键设置
                        value_settings,  # 值设置
                    ],
                    self.attn_tp_size,  # TP 大小
                    self.attn_tp_rank,  # TP 排名
                )
            },
        )

        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads // self.attn_tp_size))  # dt 偏置参数

        self.A_log = nn.Parameter(  # A 对数参数
            torch.zeros(self.num_v_heads // self.attn_tp_size, dtype=torch.float32)  # 初始化为零
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})  # 设置 A_log 分片加载器
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})  # 设置 dt_bias 分片加载器
        self.norm = (  # 归一化层
            RMSNormGated(  # 门控 RMS 归一化
                self.head_v_dim,  # 头维度
                eps=self.layer_norm_epsilon,  # epsilon
                group_size=None,  # 分组大小
                norm_before_gate=True,  # 在门控前归一化
                device=torch.get_device_module().current_device(),  # 当前设备
                dtype=config.torch_dtype,  # 数据类型
                **(  # 可选激活函数
                    {"activation": self.output_gate_type}  # 输出门控激活
                    if self.output_gate_type is not None  # 如果指定了门控类型
                    else {}  # 否则无额外参数
                ),
            )
            if not get_global_server_args().disable_piecewise_cuda_graph  # 如果未禁用分段 CUDA Graph
            else FusedRMSNormGated(  # 融合 RMS 归一化门控
                self.head_v_dim,  # 头维度
                eps=self.layer_norm_epsilon,  # epsilon
                activation=(  # 激活函数
                    self.output_gate_type  # 输出门控类型
                    if self.output_gate_type is not None  # 如果指定了
                    else self.activation  # 否则使用默认激活
                ),
                device=torch.get_device_module().current_device(),  # 当前设备
                dtype=config.torch_dtype,  # 数据类型
            )
        )

        self.out_proj = RowParallelLinear(  # 输出投影
            self.value_dim,  # 输入维度
            self.hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            input_is_parallel=True,  # 输入已并行分区
            reduce_results=False,  # 不在投影中规约
            tp_rank=self.attn_tp_rank,  # TP 排名
            tp_size=self.attn_tp_size,  # TP 大小
            prefix=add_prefix("out_proj", prefix),  # 参数前缀
        )

        self.attn = RadixLinearAttention(  # 线性注意力
            layer_id=layer_id,  # 层 ID
            num_q_heads=self.num_k_heads // self.attn_tp_size,  # 查询头数
            num_k_heads=self.num_k_heads // self.attn_tp_size,  # 键头数
            num_v_heads=self.num_v_heads // self.attn_tp_size,  # 值头数
            head_q_dim=self.head_k_dim,  # 查询头维度
            head_k_dim=self.head_k_dim,  # 键头维度
            head_v_dim=self.head_v_dim,  # 值头维度
            conv_weights=self.conv1d.weight.squeeze(1),  # 卷积权重
            bias=self.conv1d.bias,  # 卷积偏置
            activation=self.activation,  # 激活函数
            A_log=self.A_log,  # A 对数参数
            dt_bias=self.dt_bias,  # dt 偏置参数
        )

    @staticmethod
    def _override_weight_loader(module, new_loader):
        """覆盖模块权重参数的权重加载器

        ModelWeightParameter exposes weight_loader as a read-only property  # ModelWeightParameter 将 weight_loader 暴露为只读属性
        backed by _weight_loader, while plain parameters store it as a  # 由 _weight_loader 支撑，而普通参数存储为
        regular attribute.  This helper handles both cases."""  # 常规属性。此辅助函数处理两种情况
        for attr_name in (  # 遍历可能的属性名
            "weight",  # 权重
            "weight_scale_inv",  # 权重缩放倒数
            "weight_scale",  # 权重缩放
            "input_scale",  # 输入缩放
            "weight_offset",  # 权重偏移
        ):
            param = getattr(module, attr_name, None)  # 获取属性
            if param is None:  # 如果不存在
                continue  # 跳过
            if hasattr(param, "_weight_loader"):  # 如果有 _weight_loader 属性
                param._weight_loader = new_loader  # 覆盖内部加载器
            else:  # 普通参数
                param.weight_loader = new_loader  # 覆盖加载器

    @staticmethod
    def _make_packed_weight_loader(module):
        """创建支持打包格式检查点权重的连续 TP 切片加载器

        对于融合（打包格式）的检查点权重（shard_id=None）执行连续 TP 切片，
        对于分离的检查点权重（shard_id=int/tuple）委托给标准加载器"""
        original_loader = module.weight.weight_loader  # 保存原始加载器

        def weight_loader(param, loaded_weight, loaded_shard_id=None):  # 定义权重加载函数
            if loaded_shard_id is None:  # 融合检查点格式
                # Fused checkpoint: weight is in packed (per-head-group)  # 融合检查点：权重为打包（按头组）格式
                # format. Do contiguous TP slice like ColumnParallelLinear.  # 执行连续 TP 切片
                output_dim = getattr(param, "output_dim", None)  # 获取输出维度
                if output_dim is not None and module.tp_size > 1:  # 如果需要 TP 切片
                    shard_size = param.data.shape[output_dim]  # 获取分片大小
                    start_idx = module.tp_rank * shard_size  # 计算起始索引
                    if (  # CPU AMX 边界处理
                        _is_cpu and _is_amx_available  # CPU 且支持 AMX
                    ) and start_idx + shard_size > loaded_weight.shape[output_dim]:  # 超出范围
                        shard_size = loaded_weight.shape[output_dim] - start_idx  # 调整分片大小
                    loaded_weight = loaded_weight.narrow(  # 窄化到分片范围
                        output_dim, start_idx, shard_size  # 维度、起始和大小
                    )
                if _is_cpu and _is_amx_available:  # CPU AMX 模式
                    slices = tuple(slice(0, s) for s in loaded_weight.shape)  # 创建切片
                    param.data.zero_()  # 清零
                    param.data[slices].copy_(loaded_weight)  # 复制权重
                else:  # GPU 模式
                    assert param.data.shape == loaded_weight.shape, (  # 断言形状匹配
                        f"Shape mismatch: param {param.data.shape} vs "  # 形状不匹配提示
                        f"loaded {loaded_weight.shape}"  # 加载的权重形状
                    )
                    param.data.copy_(loaded_weight)  # 直接复制
            else:  # 分离检查点格式
                # Split checkpoint (int or tuple shard_id) → standard path  # 分离检查点 → 标准路径
                original_loader(param, loaded_weight, loaded_shard_id)  # 使用原始加载器

        return weight_loader  # 返回权重加载函数

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> MergedColumnParallelLinear:
        """创建 QKV+Z 投影层"""
        return MergedColumnParallelLinear(  # 返回合并列并行线性层
            input_size=hidden_size,  # 输入维度
            output_sizes=[key_dim, key_dim, value_dim, value_dim],  # Q, K, V, Z 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=prefix,  # 参数前缀
            tp_rank=tp_rank,  # TP 排名
            tp_size=tp_size,  # TP 大小
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """从混合 QKVZ 和 BA 张量中分离出 query、key、value、z、b、a"""
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.  # 从混合 QKVZBA 中提取 Q、K、V
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (  # QKVZ 的新形状
            self.num_k_heads // self.attn_tp_size,  # 头组数
            (  # 每组维度
                self.head_k_dim  # 键头维度
                + self.head_k_dim  # 键头维度
                + (self.head_v_dim + self.head_v_dim)  # 值头维度 * 2
                * self.num_v_heads  # 值头数
                // self.num_k_heads  # 按组分配
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (  # BA 的新形状
            self.num_k_heads // self.attn_tp_size,  # 头组数
            2 * self.num_v_heads // self.num_k_heads,  # 每组 B+A 维度
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)  # 重塑 QKVZ
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)  # 重塑 BA

        split_arg_list_qkvz = [  # QKVZ 分割参数
            self.head_k_dim,  # Q 维度
            self.head_k_dim,  # K 维度
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),  # V 维度
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),  # Z 维度
        ]
        split_arg_list_ba = [  # BA 分割参数
            self.num_v_heads // self.num_k_heads,  # B 维度
            self.num_v_heads // self.num_k_heads,  # A 维度
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]  # 形状说明
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]  # 分割后的形状
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)  # 分割 QKVZ
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=2)  # 分割 BA

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]  # 重塑值头维度
        value = value.reshape(value.size(0), -1, self.head_v_dim)  # 重塑值
        z = z.reshape(z.size(0), -1, self.head_v_dim)  # 重塑 z
        b = b.reshape(b.size(0), self.num_v_heads // self.attn_tp_size)  # 重塑 b
        a = a.reshape(a.size(0), self.num_v_heads // self.attn_tp_size)  # 重塑 a

        return query, key, value, z, b, a  # 返回分离后的张量

    def _forward_input_proj(self, hidden_states: torch.Tensor):
        """执行输入投影，支持双流并行以优化延迟"""
        if (  # 判断是否使用单流模式
            _is_cpu  # CPU
            or _is_npu  # NPU
            or not get_global_server_args().disable_piecewise_cuda_graph  # 未禁用分段 CUDA Graph
        ):
            DUAL_STREAM_TOKEN_THRESHOLD = 0  # 单流阈值设为 0
        else:  # GPU 模式
            DUAL_STREAM_TOKEN_THRESHOLD = 1024  # 双流阈值

        seq_len, _ = hidden_states.shape  # 获取序列长度
        if (  # 判断是否使用双流
            self.alt_stream is not None  # 有替代流
            and get_is_capture_mode()  # 处于捕获模式
            and seq_len < DUAL_STREAM_TOKEN_THRESHOLD  # 序列长度小于阈值
        ):
            current_stream = torch.cuda.current_stream()  # 获取当前流
            self.alt_stream.wait_stream(current_stream)  # 等待当前流完成
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)  # 投影 QKVZ
            with torch.cuda.stream(self.alt_stream):  # 在替代流上执行
                projected_states_ba, _ = self.in_proj_ba(hidden_states)  # 投影 BA
            current_stream.wait_stream(self.alt_stream)  # 等待替代流完成
        else:  # 单流模式
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)  # 投影 QKVZ
            projected_states_ba, _ = self.in_proj_ba(hidden_states)  # 投影 BA
        return projected_states_qkvz, projected_states_ba  # 返回投影结果

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """门控 DeltaNet 前向传播：输入投影 -> 注意力 -> 归一化门控 -> 输出投影"""
        projected_states_qkvz, projected_states_ba = self._forward_input_proj(  # 输入投影
            hidden_states  # 隐藏状态
        )

        if self.num_v_heads // self.num_k_heads in [1, 2, 4] and not _is_cpu:  # GPU 融合路径
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(  # 融合 QKVZBA 分割
                projected_states_qkvz,  # QKVZ 投影
                projected_states_ba,  # BA 投影
                triton.cdiv(self.num_k_heads, self.attn_tp_size),  # 键头数（向上取整）
                triton.cdiv(self.num_v_heads, self.attn_tp_size),  # 值头数（向上取整）
                self.head_k_dim,  # 键头维度
                self.head_v_dim,  # 值头维度
            )
        elif _is_cpu and _is_amx_available:  # CPU AMX 融合路径
            mixed_qkv, z, b, a = (  # CPU 融合 QKVZBA 分割
                torch.ops.sgl_kernel.fused_qkvzba_split_reshape_cat_cpu(  # CPU 内核
                    projected_states_qkvz,  # QKVZ 投影
                    projected_states_ba,  # BA 投影
                    self.num_k_heads // self.attn_tp_size,  # 键头数
                    self.num_v_heads // self.attn_tp_size,  # 值头数
                    self.head_k_dim,  # 键头维度
                    self.head_v_dim,  # 值头维度
                )
            )
        else:  # 回退路径
            query, key, value, z, b, a = self.fix_query_key_value_ordering(  # 手动分割
                projected_states_qkvz, projected_states_ba  # 输入投影
            )
            query, key, value = map(  # 展平头维度
                lambda x: x.reshape(x.shape[0], -1), (query, key, value)  # 展平为 2D
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)  # 拼接为混合 QKV
        core_attn_out = self.attn(  # 线性注意力计算
            forward_batch,  # 前向批次
            mixed_qkv=mixed_qkv,  # 混合 QKV
            a=a,  # A 参数
            b=b,  # B 参数
        )

        z_shape_og = z.shape  # 保存原始 z 形状
        # reshape input data into 2D tensor  # 将输入数据重塑为 2D 张量
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])  # 展平注意力输出
        z = z.reshape(-1, z.shape[-1])  # 展平 z

        # Add padding for DP-Attn  # 为数据并行注意力添加填充
        if core_attn_out.shape != z.shape:  # 如果形状不匹配
            core_attn_out_pad = torch.zeros_like(z)  # 创建零填充
            core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out  # 复制有效数据
            core_attn_out = core_attn_out_pad  # 使用填充后的张量

        core_attn_out = self.norm(core_attn_out, z)  # 归一化门控
        core_attn_out = core_attn_out.reshape(z_shape_og)  # 恢复原始形状
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)  # 合并最后两个维度

        output, _ = self.out_proj(core_attn_out)  # 输出投影
        return output  # 返回输出


def _apply_qwen3_next_mlp(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """应用 Qwen3-Next MLP（稀疏 MoE 或密集 MLP），处理通信和融合"""
    hidden_states, residual = layer.layer_communicator.prepare_mlp(  # 准备 MLP 输入
        hidden_states, residual, forward_batch  # 隐藏状态、残差和批次
    )
    use_reduce_scatter = layer.layer_communicator.should_use_reduce_scatter(  # 是否使用 reduce-scatter
        forward_batch  # 前向批次
    )
    should_allreduce_fusion = (  # 是否融合 allreduce
        layer.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(  # 是否与下一层融合
            forward_batch  # 前向批次
        )
    )

    if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):  # 如果是稀疏 MoE
        hidden_states = layer.mlp(  # 通过稀疏 MoE
            hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
            use_reduce_scatter=use_reduce_scatter,  # 是否使用 reduce-scatter
            should_allreduce_fusion=should_allreduce_fusion,  # 是否融合 allreduce
        )
    else:  # 密集 MLP
        hidden_states = layer.mlp(  # 通过密集 MLP
            hidden_states,  # 隐藏状态
            should_allreduce_fusion=should_allreduce_fusion,  # 是否融合 allreduce
            use_reduce_scatter=use_reduce_scatter,  # 是否使用 reduce-scatter
        )

    if should_allreduce_fusion:  # 如果使用 allreduce 融合
        hidden_states._sglang_needs_allreduce_fusion = True  # 标记需要 allreduce 融合
    else:  # 不融合
        hidden_states, residual = layer.layer_communicator.postprocess_layer(  # 后处理层
            hidden_states, residual, forward_batch  # 隐藏状态、残差和批次
        )

    return hidden_states, residual  # 返回隐藏状态和残差


class Qwen3HybridLinearDecoderLayer(nn.Module):
    """Qwen3 混合线性注意力解码器层，包含门控 DeltaNet 和 MoE MLP"""

    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        """初始化混合线性注意力解码器层"""
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.linear_attn = Qwen3GatedDeltaNet(  # 门控 DeltaNet 线性注意力
            config, layer_id, quant_config, alt_stream, prefix  # 配置、层 ID、量化、流和前缀
        )

        # Qwen3Next all layers are sparse and have no nextn now  # Qwen3Next 所有层都是稀疏的，目前没有 nextn
        self.is_layer_sparse = True  # 所有层均为稀疏
        is_previous_layer_sparse = True  # 前一层为稀疏
        is_next_layer_sparse = True  # 下一层为稀疏
        self.layer_id = layer_id  # 层 ID

        self.layer_scatter_modes = LayerScatterModes.init_new(  # 初始化层散射模式
            layer_id=layer_id,  # 层 ID
            num_layers=config.num_hidden_layers,  # 总层数
            is_layer_sparse=self.is_layer_sparse,  # 当前层是否稀疏
            is_previous_layer_sparse=is_previous_layer_sparse,  # 前一层是否稀疏
            is_next_layer_sparse=is_next_layer_sparse,  # 下一层是否稀疏
        )

        if self.is_layer_sparse:  # 如果是稀疏层
            self.mlp = Qwen2MoeSparseMoeBlock(  # 稀疏 MoE MLP
                layer_id=layer_id,  # 层 ID
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                alt_stream=alt_stream,  # 替代流
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),  # 参数前缀（去除 linear_attn）
                is_nextn=is_nextn,  # 是否为 nextn 层
            )
        else:  # 密集层
            self.mlp = Qwen2MoeMLP(  # 密集 MLP
                hidden_size=config.hidden_size,  # 隐藏维度
                intermediate_size=config.intermediate_size,  # 中间维度
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),  # 参数前缀
            )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = GemmaRMSNorm(  # 注意力后归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏维度和 epsilon
        )
        self.layer_communicator = LayerCommunicator(  # 层通信器
            layer_scatter_modes=self.layer_scatter_modes,  # 散射模式
            input_layernorm=self.input_layernorm,  # 输入归一化
            post_attention_layernorm=self.post_attention_layernorm,  # 注意力后归一化
            allow_reduce_scatter=True,  # 允许 reduce-scatter
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ):
        """混合线性注意力解码器层前向传播"""
        forward_batch = kwargs.get("forward_batch", None)  # 获取前向批次

        hidden_states, residual = (  # 准备注意力输入
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(  # 通信器准备
                hidden_states,  # 隐藏状态
                residual,  # 残差
                forward_batch,  # 前向批次
                captured_last_layer_outputs=captured_last_layer_outputs,  # 捕获的最后一层输出
            )
        )

        if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
            hidden_states = self.linear_attn(  # 通过线性注意力
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次
            )
        hidden_states, residual = _apply_qwen3_next_mlp(  # 应用 MLP
            self, hidden_states, residual, forward_batch  # 层、隐藏状态、残差和批次
        )

        return hidden_states, residual  # 返回隐藏状态和残差


class Qwen3HybridAttentionDecoderLayer(nn.Module):
    """Qwen3 混合全注意力解码器层，包含标准自注意力和 MoE MLP"""

    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        """初始化混合全注意力解码器层"""
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.hidden_size = config.hidden_size  # 隐藏维度
        self.attn_tp_rank = get_attention_tp_rank()  # 注意力 TP 排名
        self.attn_tp_size = get_attention_tp_size()  # 注意力 TP 大小
        self.total_num_heads = config.num_attention_heads  # 总注意力头数
        assert self.total_num_heads % self.attn_tp_size == 0  # 断言头数可被 TP 大小整除
        self.num_heads = self.total_num_heads // self.attn_tp_size  # 每个并行单元的头数
        self.total_num_kv_heads = config.num_key_value_heads  # 总 KV 头数
        if self.total_num_kv_heads >= self.attn_tp_size:  # KV 头数大于等于 TP 大小
            # Number of KV heads is greater than TP size, so we partition  # KV 头数大于 TP 大小，按 TP 分区
            # the KV heads across multiple tensor parallel GPUs.  # 在多个张量并行 GPU 上分区 KV 头
            assert self.total_num_kv_heads % self.attn_tp_size == 0  # 断言可整除
        else:  # KV 头数小于 TP 大小
            # Number of KV heads is less than TP size, so we replicate  # KV 头数小于 TP 大小，复制 KV 头
            # the KV heads across multiple tensor parallel GPUs.  # 在多个张量并行 GPU 上复制 KV 头
            assert self.attn_tp_size % self.total_num_kv_heads == 0  # 断言 TP 大小可被 KV 头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)  # 每个并行单元的 KV 头数
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)  # 头维度
        self.q_size = self.num_heads * self.head_dim  # Q 大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV 大小
        self.scaling = self.head_dim**-0.5  # 注意力缩放因子
        self.rope_theta = getattr(config, "rope_theta", 10000)  # RoPE 基底
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置数
        if "rope_parameters" in config:  # 如果有 rope_parameters
            self.rope_scaling = getattr(config, "rope_parameters", None)  # 使用 rope_parameters
        else:  # 否则
            self.rope_scaling = getattr(config, "rope_scaling", None)  # 使用 rope_scaling
        self.partial_rotary_factor = config.partial_rotary_factor  # 部分旋转因子
        self.layer_id = layer_id  # 层 ID

        self.attn_output_gate = getattr(config, "attn_output_gate", True)  # 注意力输出门控
        if self.attn_output_gate:  # 如果启用输出门控
            logger.warning_once("using attn output gate!")  # 警告使用注意力输出门控

        self.rotary_emb = get_rope(  # 创建旋转位置编码
            head_size=self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=self.max_position_embeddings,  # 最大位置
            rope_scaling=self.rope_scaling,  # 旋转缩放
            base=self.rope_theta,  # 基底
            partial_rotary_factor=self.partial_rotary_factor,  # 部分旋转因子
            is_neox_style=True,  # NeoX 风格
            dtype=torch.get_default_dtype(),  # see impl of get_rope  # 数据类型
        )

        # qkv_proj is not quantized for fp4  # FP4 模式下 qkv_proj 不量化
        self.qkv_proj = QKVParallelLinear(  # QKV 投影层
            config.hidden_size,  # 输入维度
            self.head_dim,  # 头维度
            self.total_num_heads * (1 + self.attn_output_gate),  # 总头数（含门控头）
            self.total_num_kv_heads,  # KV 头数
            bias=False,  # 不使用偏置
            quant_config=(  # 量化配置
                quant_config  # 如果有量化
                if quant_config is not None  # 且
                and quant_config.get_name() != "modelopt_fp4"  # 不是 FP4
                else None  # 否则不量化
            ),
            tp_rank=self.attn_tp_rank,  # TP 排名
            tp_size=self.attn_tp_size,  # TP 大小
            prefix=add_prefix("qkv_proj", prefix),  # 参数前缀
        )

        self.o_proj = RowParallelLinear(  # 输出投影层
            self.total_num_heads * self.head_dim,  # 输入维度
            config.hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            reduce_results=False,  # 不在投影中规约
            tp_rank=self.attn_tp_rank,  # TP 排名
            tp_size=self.attn_tp_size,  # TP 大小
            prefix=add_prefix("o_proj", prefix),  # 参数前缀
        )

        self.attn = RadixAttention(  # Radix 注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV 头数
            layer_id=layer_id,  # 层 ID
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.attn",  # 参数前缀
        )

        # Qwen3Next all layers are sparse and have no nextn now  # Qwen3Next 所有层都是稀疏的，目前没有 nextn
        self.is_layer_sparse = True  # 所有层均为稀疏
        is_previous_layer_sparse = True  # 前一层为稀疏
        is_next_layer_sparse = True  # 下一层为稀疏

        self.layer_scatter_modes = LayerScatterModes.init_new(  # 初始化层散射模式
            layer_id=layer_id,  # 层 ID
            num_layers=config.num_hidden_layers,  # 总层数
            is_layer_sparse=self.is_layer_sparse,  # 当前层是否稀疏
            is_previous_layer_sparse=is_previous_layer_sparse,  # 前一层是否稀疏
            is_next_layer_sparse=is_next_layer_sparse,  # 下一层是否稀疏
        )

        if self.is_layer_sparse:  # 如果是稀疏层
            self.mlp = Qwen2MoeSparseMoeBlock(  # 稀疏 MoE MLP
                layer_id=layer_id,  # 层 ID
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                alt_stream=alt_stream,  # 替代流
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),  # 参数前缀（去除 self_attn）
                is_nextn=is_nextn,  # 是否为 nextn 层
            )
        else:  # 密集层
            self.mlp = Qwen2MoeMLP(  # 密集 MLP
                hidden_size=config.hidden_size,  # 隐藏维度
                intermediate_size=config.intermediate_size,  # 中间维度
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),  # 参数前缀
            )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = GemmaRMSNorm(  # 注意力后归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏维度和 epsilon
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # Q 归一化
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # K 归一化

        self.layer_communicator = LayerCommunicator(  # 层通信器
            layer_scatter_modes=self.layer_scatter_modes,  # 散射模式
            input_layernorm=self.input_layernorm,  # 输入归一化
            post_attention_layernorm=self.post_attention_layernorm,  # 注意力后归一化
            allow_reduce_scatter=True,  # 允许 reduce-scatter
        )

        self.alt_stream = alt_stream  # 替代 CUDA 流

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 QK 归一化，支持双流并行"""
        # overlap qk norm  # 重叠 QK 归一化
        if self.alt_stream is not None and get_is_capture_mode():  # 双流模式
            current_stream = torch.cuda.current_stream()  # 获取当前流
            self.alt_stream.wait_stream(current_stream)  # 等待当前流
            q_by_head = q.reshape(-1, self.head_dim)  # 重塑 Q
            q_by_head = self.q_norm(q_by_head)  # 归一化 Q
            with torch.cuda.stream(self.alt_stream):  # 在替代流上
                k_by_head = k.reshape(-1, self.head_dim)  # 重塑 K
                k_by_head = self.k_norm(k_by_head)  # 归一化 K
            current_stream.wait_stream(self.alt_stream)  # 等待替代流
        else:  # 单流模式
            q_by_head = q.reshape(-1, self.head_dim)  # 重塑 Q
            q_by_head = self.q_norm(q_by_head)  # 归一化 Q
            k_by_head = k.reshape(-1, self.head_dim)  # 重塑 K
            k_by_head = self.k_norm(k_by_head)  # 归一化 K
        q = q_by_head.view(q.shape)  # 恢复 Q 形状
        k = k_by_head.view(k.shape)  # 恢复 K 形状
        return q, k  # 返回归一化后的 Q 和 K

    def forward_prepare_native(self, positions, hidden_states):
        """原生 QKV 准备：QKV 投影 -> 门控分割 -> QK 归一化 -> RoPE"""
        qkv, _ = self.qkv_proj(hidden_states)  # QKV 投影
        if self.attn_output_gate:  # 如果有输出门控
            q_gate, k, v = qkv.split(  # 分割 Q+gate、K、V
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1  # Q+gate 占两倍 Q 大小
            )
            orig_shape = q_gate.shape[:-1]  # 保存原始形状
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)  # 重塑为多头格式
            q, gate = torch.chunk(q_gate, 2, dim=-1)  # 分割 Q 和 gate
            q = q.reshape(*orig_shape, -1)  # 恢复 Q 形状
            gate = gate.reshape(*orig_shape, -1)  # 恢复 gate 形状
        else:  # 无输出门控
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割 Q、K、V
            gate = None  # 无门控

        q, k = self._apply_qk_norm(q, k)  # QK 归一化
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        return q, k, v, gate  # 返回 Q、K、V 和门控

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        """NPU 专用 QKV 准备：使用融合归一化+RoPE 内核"""
        qkv, _ = self.qkv_proj(hidden_states)  # QKV 投影
        # Calculate first full attention layer ID based on config  # 根据配置计算第一个全注意力层 ID
        if self.attn.layer_id == (self.config.full_attention_interval - 1):  # 如果是第一个全注意力层
            self.rotary_emb.get_cos_sin_with_position(positions)  # 预计算余弦和正弦

        q, k, v, gate = split_qkvgate_gemma_rmsnorm_rope(  # 融合分割+归一化+RoPE
            qkv,  # QKV 张量
            self.rotary_emb.position_sin,  # 正弦
            self.rotary_emb.position_cos,  # 余弦
            self.q_size,  # Q 大小
            self.kv_size,  # KV 大小
            self.head_dim,  # 头维度
            int(self.head_dim * self.partial_rotary_factor),  # 部分旋转维度
            eps=self.q_norm.variance_epsilon,  # 归一化 epsilon
            q_weight=self.q_norm.weight,  # Q 归一化权重
            k_weight=self.k_norm.weight,  # K 归一化权重
        )
        return q, k, v, gate  # 返回 Q、K、V 和门控

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """全注意力前向传播"""
        """Full attention forward pass."""  # 全注意力前向传播
        if (  # 判断使用原生还是 NPU 路径
            not _is_npu  # 非 NPU
            or forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()  # 扩展模式
            or not self.attn_output_gate  # 无输出门控
        ):
            q, k, v, gate = self.forward_prepare_native(  # 原生 QKV 准备
                positions=positions,  # 位置
                hidden_states=hidden_states,  # 隐藏状态
            )
        else:  # NPU 路径
            q, k, v, gate = self.forward_prepare_npu(  # NPU QKV 准备
                positions=positions,  # 位置
                hidden_states=hidden_states,  # 隐藏状态
                forward_batch=forward_batch,  # 前向批次
            )

        attn_output = self.attn(q, k, v, forward_batch)  # 注意力计算

        if self.attn_output_gate:  # 如果有输出门控
            gate = torch.sigmoid(gate)  # 对门控应用 sigmoid
            attn_output = attn_output * gate  # 乘以门控

        output, _ = self.o_proj(attn_output)  # 输出投影
        return output  # 返回输出

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        captured_last_layer_outputs: Optional[list[torch.Tensor]] = None,
        **kwargs: Any,
    ):
        """混合全注意力解码器层前向传播"""
        hidden_states, residual = (  # 准备注意力输入
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(  # 通信器准备
                hidden_states,  # 隐藏状态
                residual,  # 残差
                forward_batch,  # 前向批次
                captured_last_layer_outputs=captured_last_layer_outputs,  # 捕获的最后一层输出
            )
        )

        if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
            hidden_states = self.self_attention(  # 自注意力计算
                positions=positions,  # 位置
                hidden_states=hidden_states,  # 隐藏状态
                forward_batch=forward_batch,  # 前向批次
            )

        hidden_states, residual = _apply_qwen3_next_mlp(  # 应用 MLP
            self, hidden_states, residual, forward_batch  # 层、隐藏状态、残差和批次
        )

        return hidden_states, residual  # 返回隐藏状态和残差


ALL_DECODER_LAYER_TYPES = {  # 所有解码器层类型映射
    "attention": Qwen3HybridAttentionDecoderLayer,  # 全注意力层
    "linear_attention": Qwen3HybridLinearDecoderLayer,  # 线性注意力层
}


class Qwen3NextModel(nn.Module):
    """Qwen3-Next 模型主体，包含嵌入层和混合解码器层"""

    def __init__(
        self,
        config: Qwen3NextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_nextn: bool = False,
    ) -> None:
        """初始化 Qwen3-Next 模型主体"""
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        alt_stream = torch.cuda.Stream() if _is_cuda else None  # 创建替代 CUDA 流

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 嵌入维度
            org_num_embeddings=config.vocab_size,  # 原始词表大小
            use_attn_tp_group=is_dp_attention_enabled(),  # 是否使用注意力 TP 组
        )

        def get_layer(idx: int, prefix: str):  # 定义获取层的函数
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[idx]]  # 根据配置选择层类型
            if config.layers_block_type[idx] == "attention":  # 全注意力层
                prefix = add_prefix("self_attn", prefix)  # 使用 self_attn 前缀
            else:  # 线性注意力层
                prefix = add_prefix("linear_attn", prefix)  # 使用 linear_attn 前缀
            return layer_class(  # 返回层实例
                config,  # 配置
                idx,  # 层索引
                quant_config=quant_config,  # 量化配置
                prefix=prefix,  # 参数前缀
                alt_stream=alt_stream,  # 替代流
                is_nextn=is_nextn,  # 是否为 nextn 层
            )

        self.layers = make_layers(  # 创建层列表
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"  # 层数、获取层函数和前缀
        )

        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化
        self.infer_count = 0  # 推理计数器

        # For EAGLE3 support  # EAGLE3 支持
        self.layers_to_capture = []  # 需要捕获辅助隐藏状态的层列表

    def set_eagle3_layers_to_capture(self, layers_to_capture: list[int]):
        """设置 EAGLE3 需要捕获辅助隐藏状态的层"""
        self.layers_to_capture = layers_to_capture  # 保存层列表
        for layer_id in self.layers_to_capture:  # 遍历每个层 ID
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)  # 标记为捕获层

    def set_dflash_layers_to_capture(self, layers_to_capture: list[int]):
        """设置 DFLASH 需要捕获辅助隐藏状态的层"""
        self.layers_to_capture = layers_to_capture  # 保存层列表
        for layer_id in self.layers_to_capture:  # 遍历每个层 ID
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)  # 标记为捕获层

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        # mamba_cache_params: MambaCacheParams,  # Mamba 缓存参数（注释掉）
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """模型主体前向传播：嵌入 -> 多层解码器 -> 归一化"""

        # pass a sequence index tensor, that is required for  # 传递序列索引张量
        # proper continuous batching computation including  # 用于正确的连续批处理计算
        # chunked prefill  # 包括分块预填充
        if inputs_embeds is not None:  # 如果有输入嵌入
            hidden_states = inputs_embeds  # 直接使用
        else:  # 没有输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层

        residual = None  # 残差初始化为空
        aux_hidden_states = []  # 辅助隐藏状态列表
        for i in range(len(self.layers)):  # 遍历所有层
            layer = self.layers[i]  # 获取当前层
            with get_global_expert_distribution_recorder().with_current_layer(i):  # 记录专家分布
                hidden_states, residual = layer(  # 通过当前层
                    layer_id=i,  # 层 ID
                    positions=positions,  # 位置
                    hidden_states=hidden_states,  # 隐藏状态
                    residual=residual,  # 残差
                    forward_batch=forward_batch,  # 前向批次
                    captured_last_layer_outputs=(  # 捕获的最后一层输出
                        aux_hidden_states  # 辅助隐藏状态
                        if getattr(layer, "_is_layer_to_capture", False)  # 如果标记为捕获层
                        else None  # 否则为空
                    ),
                )

        if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
            if residual is None:  # 如果没有残差
                hidden_states = self.norm(hidden_states)  # 归一化
            else:  # 有残差
                hidden_states, _ = self.norm(hidden_states, residual)  # 融合归一化和残差

        if len(aux_hidden_states) == 0:  # 如果没有辅助隐藏状态
            return hidden_states  # 返回隐藏状态

        return hidden_states, aux_hidden_states  # 返回隐藏状态和辅助隐藏状态


class HybridLayerType(enum.Enum):
    """混合层类型枚举"""
    full_attention = "attention"  # 全注意力
    swa_attention = "swa_attention"  # 滑动窗口注意力
    linear_attention = "linear_attention"  # 线性注意力
    mamba2 = "mamba"  # Mamba2


class Qwen3NextForCausalLM(nn.Module):
    """Qwen3-Next 因果语言模型，整合混合解码器和 MoE MLP"""

    fall_back_to_pt_during_load = False  # 加载权重时不回退到 PyTorch

    # Map fused module names to their checkpoint (unfused) counterparts.  # 将融合模块名映射到检查点（未融合）名称
    # This is needed so the quantization exclusion logic can match  # 这是量化排除逻辑需要的
    # checkpoint-style names (e.g. "q_proj") against the fused sglang  # 以匹配检查点名和融合的 sglang 名
    # module names (e.g. "qkv_proj").  # 例如 q_proj 和 qkv_proj
    packed_modules_mapping = {  # 打包模块映射
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],  # QKV 映射
        "gate_up_proj": ["gate_proj", "up_proj"],  # gate_up 映射
    }

    def __init__(
        self,
        config: Qwen3NextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化 Qwen3-Next 因果语言模型"""
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.pp_group = get_pp_group()  # 流水线并行组
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank  # 断言同时是第一和最后排名

        # The quant config's packed_modules_mapping may be None if it wasn't  # 量化配置的打包模块映射可能为空
        # in the checkpoint config. The base class (QuantizationConfig) intends  # 如果不在检查点配置中
        # for models to set this. We need it so is_layer_skipped can unfuse  # 基类要求模型设置此项
        # "qkv_proj" into ["q_proj","k_proj","v_proj"] when checking exclusions.  # 用于 is_layer_skipped 检查时解包
        if quant_config is not None and hasattr(quant_config, "packed_modules_mapping"):  # 如果有量化配置和映射属性
            quant_config.packed_modules_mapping = self.packed_modules_mapping  # 设置映射

        self.quant_config = quant_config  # 保存量化配置
        self.model = Qwen3NextModel(  # 创建模型主体
            config, quant_config, prefix=add_prefix("model", prefix)  # 配置、量化配置和前缀
        )
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏维度
            quant_config=quant_config,  # 量化配置
            org_num_embeddings=config.vocab_size,  # 原始词表大小
            prefix=add_prefix("lm_head", prefix),  # 参数前缀
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力 TP 组
        )
        self.logits_processor = LogitsProcessor(config)  # logits 处理器
        # For EAGLE3 support  # EAGLE3 支持
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态

        self._routed_experts_weights_of_layer = LazyValue(  # 懒加载路由专家权重
            lambda: {  # 匿名函数
                layer_id: layer.mlp.get_moe_weights()  # 获取每层 MoE 权重
                for layer_id, layer in enumerate(self.model.layers)  # 遍历所有层
                if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)  # 仅稀疏 MoE 层
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        """获取每层路由专家的权重"""
        return self._routed_experts_weights_of_layer.value  # 返回懒加载值

    @torch.no_grad()  # 禁用梯度计算
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Qwen3-Next 因果语言模型前向传播"""
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)  # 通过模型主体

        aux_hidden_states = None  # 辅助隐藏状态初始化为空
        if self.capture_aux_hidden_states:  # 如果捕获辅助隐藏状态
            hidden_states, aux_hidden_states = hidden_states  # 解包

        return self.logits_processor(  # 处理 logits
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states  # 输入 ID、隐藏状态、语言模型头、批次和辅助隐藏状态
        )

    def get_embed_and_head(self):
        """获取嵌入层和语言模型头的权重"""
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入权重和语言模型头权重

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.model.embed_tokens  # 返回词嵌入层

    def set_embed_and_head(self, embed, head):
        """设置嵌入层和语言模型头的权重（共享目标模型的权重）"""
        del self.model.embed_tokens.weight  # 删除原有嵌入权重
        del self.lm_head.weight  # 删除原有语言模型头权重
        self.model.embed_tokens.weight = embed  # 设置新嵌入权重
        self.lm_head.weight = head  # 设置新语言模型头权重
        torch.cuda.empty_cache()  # 清空 GPU 缓存
        torch.cuda.synchronize()  # 同步 CUDA

    def get_embed(self):
        """获取嵌入层权重"""
        return self.model.embed_tokens.weight  # 返回嵌入权重

    def set_embed(self, embed):
        """设置嵌入层权重"""
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared for EAGLE3  # 注意：如果草稿隐藏大小不等于目标隐藏大小，EAGLE3 不能共享嵌入权重
        if (  # 检查是否可以共享
            hasattr(self.config, "target_hidden_size")  # 有目标隐藏大小配置
            and self.config.target_hidden_size != self.config.hidden_size  # 且不等于当前隐藏大小
        ):
            return  # 不能共享，直接返回
        del self.model.embed_tokens.weight  # 删除原有嵌入权重
        self.model.embed_tokens.weight = embed  # 设置新嵌入权重
        torch.cuda.empty_cache()  # 清空 GPU 缓存
        torch.cuda.synchronize()  # 同步 CUDA

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> Set[str]:
        """加载模型权重，处理堆叠参数、专家参数和 MTP 前缀映射"""
        stacked_params_mapping = [  # 堆叠参数映射表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            # self attention  # 自注意力
            ("qkv_proj", "q_proj", "q"),  # Q 映射
            ("qkv_proj", "k_proj", "k"),  # K 映射
            ("qkv_proj", "v_proj", "v"),  # V 映射
            # mlp  # MLP
            ("gate_up_proj", "gate_proj", 0),  # gate 映射
            ("gate_up_proj", "up_proj", 1),  # up 映射
            # GDN  # 门控 DeltaNet
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),  # QKV 映射
            ("in_proj_qkvz.", "in_proj_z.", 3),  # Z 映射
            ("in_proj_ba.", "in_proj_b.", 0),  # B 映射
            ("in_proj_ba.", "in_proj_a.", 1),  # A 映射
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales  # 权重、FP8 权重缩放和激活缩放参数
        # (param_name, weight_name, expert_id, shard_id)  # (参数名, 权重名, 专家ID, 分片ID)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 专家参数映射
            ckpt_gate_proj_name="gate_proj",  # gate 投影检查点名
            ckpt_down_proj_name="down_proj",  # down 投影检查点名
            ckpt_up_proj_name="up_proj",  # up 投影检查点名
            num_experts=self.config.num_experts,  # 专家数量
        )

        params_dict = dict(self.named_parameters())  # 获取参数字典
        loaded_params: Set[str] = set()  # 已加载参数集合
        for name, loaded_weight in weights:  # 遍历所有权重

            if is_mtp:  # 如果是 MTP 模式

                if "mtp" not in name:  # 跳过非 MTP 权重
                    continue

                if name in [  # MTP 特定权重
                    "mtp.fc.weight",  # 融合层权重
                    "mtp.pre_fc_norm_embedding.weight",  # 嵌入预融合归一化权重
                    "mtp.pre_fc_norm_hidden.weight",  # 隐藏预融合归一化权重
                ]:
                    name = name.replace("mtp.", "")  # 去掉 mtp 前缀
                else:  # 其他 MTP 权重
                    name = name.replace("mtp", "model")  # 替换 mtp 为 model

            if not is_mtp and "mtp" in name:  # 非 MTP 模式跳过 MTP 权重
                continue

            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入逆频率
                continue

            if ".self_attn." in name:  # 去掉 self_attn 中间前缀
                name = name.replace(".self_attn", "")  # 替换

            # Remap modelopt FP8 KV cache scale names:  # 重映射 modelopt FP8 KV 缓存缩放名称
            # checkpoint: k_proj.k_scale / v_proj.v_scale  # 检查点格式
            # model:      attn.k_scale   / attn.v_scale  # 模型格式
            if name.endswith(".k_proj.k_scale"):  # K 缩放
                name = name.replace(".k_proj.k_scale", ".attn.k_scale")  # 替换
            elif name.endswith(".v_proj.v_scale"):  # V 缩放
                name = name.replace(".v_proj.v_scale", ".attn.v_scale")  # 替换

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue

                # TODO(fix mtp loading)  # TODO（修复 MTP 加载）
                if "mlp.experts" in name:  # 跳过 MLP 专家（在专家映射中处理）
                    continue

                replaced_name = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置
                if replaced_name.endswith(".bias") and replaced_name not in params_dict:  # 如果偏置不在字典中
                    continue
                # Skip layers on other devices.  # 跳过其他设备上的层
                # if is_pp_missing_parameter(name, self):  # 如果是流水线缺失参数
                #     continue  # 跳过
                if replaced_name not in params_dict:  # 如果参数不在字典中
                    continue
                name = replaced_name  # 使用替换后的名称
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader")  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break
            else:  # 非堆叠参数处理
                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping  # 解包映射
                    if weight_name not in name:  # 如果权重名不在参数名中
                        continue
                    replaced_name = name.replace(weight_name, param_name)  # 替换为专家参数名
                    # Skip layers on other devices.  # 跳过其他设备上的层
                    # if is_pp_missing_parameter(name, self):  # 如果是流水线缺失参数
                    #     continue  # 跳过
                    # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置
                    if (  # 检查是否为额外偏置
                        replaced_name.endswith(".bias")  # 以 .bias 结尾
                        or replaced_name.endswith("_bias")  # 或以 _bias 结尾
                    ) and replaced_name not in params_dict:  # 且不在参数字典中
                        continue
                    name = replaced_name  # 使用替换后的名称
                    param = params_dict[name]  # 获取参数

                    weight_loader = getattr(param, "weight_loader")  # 获取权重加载器
                    weight_loader(  # 加载专家权重
                        param,  # 参数
                        loaded_weight,  # 加载的权重
                        name,  # 参数名
                        shard_id=shard_id,  # 分片 ID
                        expert_id=expert_id,  # 专家 ID
                    )
                    break
                else:  # 非专家参数处理
                    # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置
                    if name.endswith(".bias") and name not in params_dict:  # 如果偏置不在字典中
                        continue
                    # if is_pp_missing_parameter(name, self):  # 如果是流水线缺失参数
                    #     continue  # 跳过

                    if name.endswith("_scale") and name not in params_dict:  # 跳过不在字典中的缩放参数
                        assert (  # 断言缩放值为 1.0
                            abs(loaded_weight.item() - 1.0) < 1e-6  # 差值小于 1e-6
                        ), f"Expected 1.0, got {loaded_weight.item()} in skipped {name}"  # 否则报错
                        continue
                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认权重加载器
                    )
                    weight_loader(param, loaded_weight)  # 加载权重
            loaded_params.add(name)  # 记录已加载参数
        return loaded_params  # 返回已加载参数集合

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        """获取专家位置模型配置"""
        return ModelConfigForExpertLocation(  # 返回模型配置
            num_layers=config.num_hidden_layers,  # 层数
            num_logical_experts=config.num_experts,  # 逻辑专家数
            num_groups=None,  # 分组数为空
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
        """设置 EAGLE3 推测解码需要捕获辅助隐藏状态的层"""
        if not self.pp_group.is_last_rank:  # 非最后排名不执行
            return

        self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
        if layer_ids is None:  # 如果未指定层 ID
            num_layers = self.config.num_hidden_layers  # 获取总层数
            self.model.set_eagle3_layers_to_capture(  # 设置默认捕获层
                [  # 默认捕获层列表
                    2,  # 第 2 层
                    num_layers // 2,  # 中间层
                    num_layers - 3,  # 倒数第 3 层
                ]
            )  # Specific layers for EAGLE3 support  # EAGLE3 支持的特定层
        else:  # 指定了层 ID
            self.model.set_eagle3_layers_to_capture([val + 1 for val in layer_ids])  # 偏移 1 后设置

    def set_dflash_layers_to_capture(self, layer_ids: list[int]):
        """设置 DFLASH 需要捕获辅助隐藏状态的层"""
        if not self.pp_group.is_last_rank:  # 非最后排名不执行
            return

        if layer_ids is None:  # 如果未指定层 ID
            raise ValueError(  # 抛出异常
                "DFLASH requires explicit layer_ids for aux hidden capture."  # DFLASH 需要显式指定层 ID
            )

        self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
        self.model.set_dflash_layers_to_capture([val + 1 for val in layer_ids])  # 偏移 1 后设置


EntryClass = Qwen3NextForCausalLM  # 模型入口类
