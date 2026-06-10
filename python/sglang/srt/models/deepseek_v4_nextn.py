# DeepSeek V4 Next-N 推测解码模型实现
# 本文件实现了 DeepSeek V4 的 Next-N（推测解码）变体，
# 包含 DeepseekV4ModelNextN 模型和 DeepseekV4ForCausalLMNextN 因果语言模型类。
# Next-N 模型用于在推测解码中生成多个候选 token，加速推理过程。

import logging  # 导入日志模块
from typing import Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch核心库
import torch.nn.functional as F  # 导入PyTorch函数式接口
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置基类

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 导入分布式并行工具函数
from sglang.srt.layers.attention.dsa.utils import (  # 导入DSA注意力上下文并行相关工具
    can_dsa_cp_split,  # 判断是否可以DSA上下文并行拆分
    dsa_use_prefill_cp,  # 判断DSA是否使用预填充上下文并行
    is_dsa_enable_prefill_cp,  # 判断是否启用DSA预填充上下文并行
    is_dsa_prefill_cp_round_robin_split,  # 判断DSA预填充上下文并行是否使用轮询拆分
)
from sglang.srt.layers.dp_attention import (  # 导入数据并行注意力相关工具
    _DpGatheredBufferWrapper,  # 数据并行聚集缓冲区包装器
    dp_gather_partial,  # 数据并行部分聚集
    get_attention_cp_rank,  # 获取注意力上下文并行秩
    get_attention_cp_size,  # 获取注意力上下文并行大小
    get_attention_dp_size,  # 获取注意力数据并行大小
    is_dp_attention_enabled,  # 判断是否启用数据并行注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import ReplicatedLinear  # 导入复制线性层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.moe.utils import get_moe_a2a_backend  # 导入MoE全互联后端工具
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.utils.cp_utils import (  # 导入上下文并行工具函数
    cp_all_gather_rerange_output,  # 上下文并行全聚集重排输出
    cp_round_robin_input_ids,  # 上下文并行轮询分配输入ID
    cp_split_and_rebuild_data,  # 上下文并行拆分和重建数据
    cp_split_and_rebuild_position,  # 上下文并行拆分和重建位置
    prepare_context_parallel_metadata,  # 准备上下文并行元数据
)
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_executor.forward_context import get_attn_backend  # 导入获取注意力后端函数
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer, DeepseekV4ForCausalLM  # 导入DeepSeek V4解码器层和因果语言模型
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

logger = logging.getLogger(__name__)  # 创建模块级日志记录器

COMPRESS_RATIO_NEXTN_LAYER = 0  # NextN层的压缩比率，0表示不压缩


class DeepseekV4ModelNextN(nn.Module):
    """DeepSeek V4 Next-N 模型主体，用于推测解码的候选token生成"""
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数名前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.vocab_size = config.vocab_size  # 获取词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 创建词表并行嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层维度
            enable_tp=not is_dp_attention_enabled(),  # 当未启用DP注意力时启用张量并行
            prefix=add_prefix("embed_tokens", prefix),  # 添加前缀
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 嵌入归一化层
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 隐藏状态归一化层
        self.rms_norm_eps = config.rms_norm_eps  # RMS归一化epsilon值

        self.hc_eps = config.hc_eps  # HC头epsilon值
        self.hc_mult = hc_mult = config.hc_mult  # HC头倍数，即每个token生成的候选数
        hc_dim = hc_mult * config.hidden_size  # HC头维度 = 倍数 × 隐藏层维度
        self.hc_head_fn = nn.Parameter(  # HC头函数参数
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)  # 形状为[倍数, 倍数×隐藏维度]
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))  # HC头偏置参数
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))  # HC头缩放参数

        self.e_proj = ReplicatedLinear(  # 嵌入投影层，将嵌入映射到隐藏空间
            config.hidden_size,  # 输入维度
            config.hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("e_proj", prefix),  # 添加前缀
        )
        self.h_proj = ReplicatedLinear(  # 隐藏状态投影层，将隐藏状态映射到候选空间
            config.hidden_size,  # 输入维度
            config.hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("h_proj", prefix),  # 添加前缀
        )

        layer_name = "decoder"  # 解码器层名称

        self.decoder = DeepseekV4DecoderLayer(  # 创建DeepSeek V4解码器层
            config,  # 模型配置
            layer_id=0,  # 层ID为0（NextN只有一层）
            quant_config=quant_config,  # 量化配置
            is_nextn=True,  # 标记为NextN层
            prefix=add_prefix(layer_name, prefix),  # 添加前缀
            alt_streams=None,  # 无备用流
            compress_ratio_override=COMPRESS_RATIO_NEXTN_LAYER,  # 压缩比率覆盖
        )

        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()  # 判断是否启用DSA预填充上下文并行
        if self.dsa_enable_prefill_cp:  # 如果启用DSA预填充上下文并行
            self.cp_size = get_attention_cp_size()  # 获取上下文并行大小
        else:  # 否则
            self.cp_size = None  # 不设置上下文并行大小

        self.shared_head = nn.Module()  # 创建共享头模块
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 共享头归一化层

    def hc_head(  # HC头计算方法，用于从多个候选中选择最优token
        self,
        x: torch.Tensor,  # 输入张量，形状为[batch, hc_mult, hidden_dim]
        hc_fn: torch.Tensor,  # HC头函数参数
        hc_scale: torch.Tensor,  # HC头缩放参数
        hc_base: torch.Tensor,  # HC头偏置参数
    ):
        shape, dtype = x.size(), x.dtype  # 保存原始形状和数据类型
        x = x.flatten(1).float()  # 展平为[batch, hc_mult*hidden_dim]并转为float32
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.rms_norm_eps)  # 计算RMS归一化的逆平方根
        mixes = F.linear(x, hc_fn) * rsqrt  # 线性变换后乘以归一化因子，得到混合系数
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps  # 计算sigmoid激活加偏移，确保非零
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)  # 加权求和，将多个候选聚合为一个表示
        return y.to(dtype)  # 转回原始数据类型并返回

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选
    ) -> torch.Tensor:
        if input_embeds is None:  # 如果没有提供输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过词表嵌入层获取隐藏状态
        else:  # 否则
            hidden_states = input_embeds  # 直接使用提供的嵌入

        if hidden_states.shape[0] > 0:  # 如果有token需要处理
            n_tokens = hidden_states.shape[0]  # 获取token数量
            d = self.config.hidden_size  # 获取隐藏层维度
            hc_flat = forward_batch.spec_info.hidden_states.view(  # 将推测信息中的隐藏状态展平
                n_tokens * self.hc_mult, d  # 形状为[token数×候选倍数, 隐藏维度]
            )
            h_proj_out, _ = self.h_proj(self.hnorm(hc_flat))  # 对隐藏状态归一化后投影
            h_proj_hidden_states = h_proj_out.view(n_tokens, self.hc_mult, d)  # 重塑为[token数, 候选倍数, 隐藏维度]

            e_proj_hidden_states, _ = self.e_proj(self.enorm(hidden_states))  # 对嵌入归一化后投影
            hidden_states = e_proj_hidden_states[:, None, :] + h_proj_hidden_states  # 将嵌入投影与隐藏投影相加，广播合并
        else:  # 如果没有token
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)  # 扩展维度并重复以匹配候选倍数

        if get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none():  # 如果DP注意力>1且非全互联后端
            input_ids_global = torch.empty(  # 创建全局输入ID缓冲区
                (_DpGatheredBufferWrapper._global_dp_buffer_len, 1),  # 形状为[全局DP缓冲长度, 1]
                dtype=input_ids.dtype,  # 数据类型与输入一致
                device=input_ids.device,  # 设备与输入一致
            )
            dp_gather_partial(input_ids_global, input_ids[:, None], forward_batch)  # 部分聚集输入ID
            input_ids_global = input_ids_global.squeeze(-1)  # 移除最后一维
        else:  # 否则
            input_ids_global = input_ids  # 直接使用原始输入ID

        if dsa_use_prefill_cp(forward_batch):  # 如果DSA使用预填充上下文并行
            hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)  # 上下文并行拆分重建数据
            positions = cp_split_and_rebuild_position(forward_batch, positions)  # 上下文并行拆分重建位置
            input_ids = cp_round_robin_input_ids(input_ids)  # 轮询分配输入ID
            input_ids_global = input_ids  # 使用轮询后的输入ID作为全局输入ID

        hidden_states, residual, post, comb = self.decoder(  # 通过解码器层
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次信息
            input_ids=input_ids,  # 输入ID
            input_ids_global=input_ids_global,  # 全局输入ID
        )
        if residual is not None:  # 如果存在残差
            # NextN has a single decoder layer, so no later layer can consume a
            # deferred fused hc_post state.
            # NextN只有一个解码器层，因此没有后续层可以消费延迟融合的hc_post状态
            hidden_states = self.decoder.hc_post(hidden_states, residual, post, comb)  # 执行hc_post处理

        if dsa_use_prefill_cp(forward_batch):  # 如果DSA使用预填充上下文并行
            hidden_states = cp_all_gather_rerange_output(  # 全聚集并重排输出
                hidden_states,  # 隐藏状态
                self.cp_size,  # 上下文并行大小
                forward_batch,  # 前向批次信息
                torch.cuda.current_stream(),  # 当前CUDA流
            )

        pre_hc_head = hidden_states.flatten(1)  # 保存HC头处理前的展平隐藏状态

        hidden_states = self.hc_head(  # 通过HC头处理
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base  # 传入输入和HC头参数
        )
        hidden_states = self.shared_head.norm(hidden_states)  # 通过共享头归一化层

        return hidden_states, pre_hc_head  # 返回归一化后的隐藏状态和HC头前的状态

class DeepseekV4ForCausalLMNextN(DeepseekV4ForCausalLM):  # DeepSeek V4 Next-N 因果语言模型，继承自DeepseekV4ForCausalLM
    """DeepSeek V4 Next-N 因果语言模型，用于推测解码"""

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数名前缀
    ) -> None:
        nn.Module.__init__(self)  # 直接调用nn.Module的初始化，而非父类
        self.config = config  # 保存配置
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.quant_config = quant_config  # 保存量化配置
        self.determine_num_fused_shared_experts()  # 确定融合共享专家数量
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()  # 判断是否启用DSA预填充上下文并行
        if self.dsa_enable_prefill_cp:  # 如果启用DSA预填充上下文并行
            self.cp_rank = get_attention_cp_rank()  # 获取上下文并行秩
            self.cp_size = get_attention_cp_size()  # 获取上下文并行大小
        else:  # 否则
            self.cp_rank = None  # 不设置上下文并行秩
            self.cp_size = None  # 不设置上下文并行大小

        self.model = DeepseekV4ModelNextN(  # 创建Next-N模型主体
            config, quant_config, prefix=add_prefix("model", prefix)  # 传入配置、量化配置和前缀
        )
        self.lm_head = ParallelLMHead(  # 创建并行语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("model.shared_head.head", prefix),  # 添加前缀
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力TP组
        )
        self.logits_processor = LogitsProcessor(config)  # 创建logits处理器

    @torch.no_grad()  # 禁用梯度计算装饰器
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        if self.dsa_enable_prefill_cp:  # 如果启用DSA预填充上下文并行
            if can_dsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):  # 判断是否可以上下文并行拆分
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(  # 准备上下文并行元数据
                    len(input_ids),  # 输入token数
                    self.cp_rank,  # 上下文并行秩
                    self.cp_size,  # 上下文并行大小
                    forward_batch.seq_lens_cpu.tolist(),  # 序列长度列表
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,  # 扩展序列长度
                )
                if is_dsa_prefill_cp_round_robin_split():  # 如果使用轮询拆分
                    attn_backend = get_attn_backend()  # 获取注意力后端
                    metadata = attn_backend.forward_metadata  # 获取前向元数据
                    core_meta = metadata.core_attn_metadata  # 获取核心注意力元数据
                    core_meta.apply_cp_reindex()  # 应用上下文并行重索引
                    core_meta.init_flashmla_related()  # 初始化FlashMLA相关数据
                    if metadata.indexer_metadata is not None:  # 如果索引器元数据存在
                        metadata.indexer_metadata = (  # 重新初始化索引器元数据
                            attn_backend.init_forward_metadata_indexer(core_meta)  # 使用核心注意力元数据初始化
                        )

        hidden_states, pre_hc_head = self.model(input_ids, positions, forward_batch)  # 通过模型主体获取隐藏状态
        return self.logits_processor(  # 通过logits处理器计算最终输出
            input_ids,  # 输入token ID
            hidden_states,  # 隐藏状态
            self.lm_head,  # 语言模型头
            forward_batch,  # 前向批次信息
            hidden_states_before_norm=pre_hc_head,  # 归一化前的隐藏状态
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重方法
        super().load_weights(weights, is_nextn=True)  # 调用父类加载权重，标记为NextN

    def post_load_weights(self, is_nextn=False, weight_names=None):  # 权重加载后处理方法
        super().post_load_weights(is_nextn=True, weight_names=weight_names)  # 调用父类后处理，强制标记为NextN


EntryClass = [DeepseekV4ForCausalLMNextN]  # 模型入口类注册
