# DBRX模型实现文件 - 实现Databricks DBRX混合专家(MoE)因果语言模型的SGLang推理
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

# Adapted from:  # 改编自：
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/dbrx.py#L1  # vLLM DBRX参考链接

from typing import Iterable, Optional, Tuple # 导入类型提示模块

import torch # 导入PyTorch深度学习框架
import torch.nn as nn # 导入神经网络模块

from sglang.srt.configs import DbrxConfig # 导入DBRX配置
from sglang.srt.distributed import ( # 导入分布式相关函数
    get_tensor_model_parallel_rank, # 获取张量并行排名
    get_tensor_model_parallel_world_size, # 获取张量并行世界大小
    tensor_model_parallel_all_reduce, # 张量并行全归约
)
from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import ( # 导入NPU MoE融合方法
    fused_moe_npu, # NPU融合MoE方法
)
from sglang.srt.layers.linear import ( # 导入线性层
    QKVParallelLinear, # QKV并行线性层
    ReplicatedLinear, # 复制线性层
    RowParallelLinear, # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor # 导入逻辑处理器
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig # 导入MoE运行器配置
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe # 导入融合MoE方法
from sglang.srt.layers.moe.topk import TopK # 导入TopK选择器
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import ( # 导入词表并行嵌入层
    DEFAULT_VOCAB_PADDING_SIZE, # 默认词表填充大小
    ParallelLMHead, # 并行语言模型头
    VocabParallelEmbedding, # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批次信息类
from sglang.srt.model_loader.weight_utils import ( # 导入权重加载工具
    default_weight_loader, # 默认权重加载器
    maybe_remap_kv_scale_name, # 可能重映射KV缩放名称
)
from sglang.srt.utils import add_prefix, is_npu, set_weight_attrs # 导入工具函数

_is_npu = is_npu() # 是否为NPU设备


class DbrxRouter(nn.Module): # DBRX路由器类，为每个token返回各专家的logits
    """A Router implementation for DBRX that returns logits for each expert  # DBRX路由器实现，返回每个专家的logits
    per token.  # 每个token
    """

    def __init__( # 初始化DBRX路由器
        self,
        config: DbrxConfig, # DBRX配置
        params_dtype: Optional[torch.dtype] = None, # 参数数据类型
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.tp_size = get_tensor_model_parallel_world_size() # 张量并行大小
        self.num_total_experts = config.ffn_config.moe_num_experts # 总专家数
        self.d_model = config.d_model # 模型维度
        self.layer = ReplicatedLinear( # 复制线性层（路由计算）
            self.d_model, # 输入维度
            self.num_total_experts, # 输出维度（专家数）
            bias=False, # 不使用偏置
            params_dtype=params_dtype, # 参数数据类型
            quant_config=None, # 不使用量化
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: # 前向传播：计算路由logits
        router_logits, _ = self.layer(hidden_states) # 通过线性层计算路由logits
        return router_logits # 返回路由logits


class DbrxExperts(nn.Module): # DBRX专家类，实现张量并行的混合专家(MoE)前馈网络
    """A tensor-parallel MoE implementation for DBRX.  # DBRX的张量并行MoE实现

    Each expert's weights are sharded across all ranks and a fused MoE  # 每个专家的权重在所有排名间分片，使用融合MoE
    kernel is used for the forward pass, and finally we reduce the outputs  # 内核进行前向传播，最后在排名间
    across ranks.  # 归约输出
    """

    def __init__( # 初始化DBRX专家层
        self,
        config: DbrxConfig, # DBRX配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        params_dtype: Optional[torch.dtype] = None, # 参数数据类型
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.tp_size = get_tensor_model_parallel_world_size() # 张量并行大小
        self.num_total_experts = config.ffn_config.moe_num_experts # 总专家数
        self.top_k = config.ffn_config.moe_top_k # Top-K选择数
        self.d_model = config.d_model # 模型维度
        self.intermediate_size = config.ffn_config.ffn_hidden_size // self.tp_size # 中间层维度（分片后）

        if params_dtype is None: # 如果没有指定参数类型
            params_dtype = torch.get_default_dtype() # 使用默认数据类型
        self.params_dtype = params_dtype # 保存参数数据类型

        self.router = DbrxRouter(config, self.params_dtype) # 路由器
        self.topk = TopK( # TopK选择器
            self.top_k, # Top-K值
            renormalize=True, # 重新归一化
        )
        self.moe_runner_config = MoeRunnerConfig(inplace=True) # MoE运行器配置（原地操作）
        self.ws = nn.Parameter( # w1和v1合并的权重参数（升维投影）
            torch.empty(
                self.num_total_experts, # 专家数
                2 * self.intermediate_size, # 2倍中间维度（gate和up）
                self.d_model, # 模型维度
                device="cuda", # CUDA设备
                dtype=self.params_dtype, # 数据类型
            )
        )
        self.w2s = nn.Parameter( # w2权重参数（降维投影）
            torch.empty(
                self.num_total_experts, # 专家数
                self.d_model, # 模型维度
                self.intermediate_size, # 中间维度
                device="cuda", # CUDA设备
                dtype=self.params_dtype, # 数据类型
            )
        )

        set_weight_attrs( # 设置ws的权重属性
            self.ws,
            {
                "weight_loader": self.weight_loader, # 权重加载器
            },
        )
        set_weight_attrs( # 设置w2s的权重属性
            self.w2s,
            {
                "weight_loader": self.weight_loader, # 权重加载器
            },
        )
        self.fused_moe_method = fused_moe if not _is_npu else fused_moe_npu # 根据设备选择融合MoE方法

    def weight_loader( # 权重加载器：处理GLU的w1、v1和w2权重加载与分片
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str # 参数、加载的权重、权重名称
    ):
        tp_rank = get_tensor_model_parallel_rank() # 获取当前张量并行排名
        param_data = param.data # 获取参数数据
        shard_size = self.intermediate_size # 分片大小
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size) # 当前排名的分片范围
        # DBRX uses GLU for each experts.  # DBRX每个专家使用GLU
        # GLU has 3 linear layers: w1, v1 and w2.  # GLU有3个线性层：w1、v1和w2
        if weight_name.endswith("w1"): # 如果是w1权重（gate投影）
            loaded_weight = torch.reshape( # 重塑权重形状
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.d_model], # [专家数, 中间维度*TP, 模型维度]
            )
            param_data[:, 0:shard_size, :] = loaded_weight[:, shard, :] # 加载w1分片到前半部分
        if weight_name.endswith("v1"): # 如果是v1权重（up投影）
            loaded_weight = torch.reshape( # 重塑权重形状
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.d_model], # [专家数, 中间维度*TP, 模型维度]
            )
            param_data[:, shard_size : 2 * shard_size, :] = loaded_weight[:, shard, :] # 加载v1分片到后半部分
        if weight_name.endswith("w2"): # 如果是w2权重（降维投影）
            loaded_weight = torch.reshape( # 重塑权重形状
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.d_model], # [专家数, 中间维度*TP, 模型维度]
            ).transpose(1, 2) # 转置
            param_data[:] = loaded_weight[:, :, shard] # 加载w2分片

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: # 前向传播：路由→TopK→融合MoE
        num_tokens, hidden_size = hidden_states.shape # 获取token数和隐藏维度
        hidden_states = hidden_states.view(-1, self.d_model) # 重塑为(d_model,)的形状
        # router_logits: (num_tokens, n_experts)  # 路由logits: (token数, 专家数)
        router_logits = self.router(hidden_states) # 通过路由器计算logits
        topk_output = self.topk(hidden_states, router_logits) # TopK选择
        final_hidden_states = self.fused_moe_method( # 融合MoE计算
            hidden_states, # 输入隐藏状态
            self.ws, # 升维权重
            self.w2s, # 降维权重
            topk_output, # TopK输出
            self.moe_runner_config, # MoE运行器配置
        )

        if self.tp_size > 1: # 如果使用张量并行
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states) # 全归约

        return final_hidden_states.view(num_tokens, hidden_size) # 恢复原始形状并返回


class DbrxAttention(nn.Module): # DBRX注意力层类，实现多头注意力机制
    def __init__( # 初始化DBRX注意力层
        self,
        config: DbrxConfig, # DBRX配置
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.d_model = config.d_model # 模型维度
        self.total_num_heads = config.n_heads # 总注意力头数
        self.head_dim = self.d_model // self.total_num_heads # 每个头的维度
        self.total_num_kv_heads = config.attn_config.kv_n_heads # 总KV头数
        self.clip_qkv = config.attn_config.clip_qkv # QKV裁剪值
        self.rope_theta = config.attn_config.rope_theta # RoPE的theta参数
        self.max_position = config.max_seq_len # 最大序列长度

        # pylint: disable=invalid-name  # 禁用无效名称的pylint检查
        self.Wqkv = QKVParallelLinear( # QKV联合投影层
            self.d_model, # 输入维度
            self.head_dim, # 每个头的维度
            self.total_num_heads, # 总Q头数
            self.total_num_kv_heads, # 总KV头数
            bias=False, # 不使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("Wqkv", prefix), # 参数名前缀
        )
        self.out_proj = RowParallelLinear( # 输出投影层
            self.d_model, # 输入维度
            self.d_model, # 输出维度
            bias=False, # 不使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("out_proj", prefix), # 参数名前缀
        )
        self.rotary_emb = get_rope( # 旋转位置编码
            self.head_dim, # 每个头的维度
            rotary_dim=self.head_dim, # 旋转维度
            max_position=self.max_position, # 最大位置数
            base=int(self.rope_theta), # 基数
            is_neox_style=True, # NeoX风格
        )

        tp_world_size = get_tensor_model_parallel_world_size() # 获取张量并行世界大小
        self.tp_size = tp_world_size # 保存TP大小
        assert self.total_num_heads % tp_world_size == 0 # 确保头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_world_size # 每个并行分片的头数
        if self.total_num_kv_heads >= tp_world_size: # 如果KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，因此在多个张量并行GPU间分配
            # the KV heads across multiple tensor parallel GPUs.  # KV头
            assert self.total_num_kv_heads % tp_world_size == 0 # 确保可整除
        else: # 否则
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，因此在多个张量并行GPU间复制
            # the KV heads across multiple tensor parallel GPUs.  # KV头
            assert tp_world_size % self.total_num_kv_heads == 0 # 确保可整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size) # 每个并行分片的KV头数
        self.q_size = self.num_heads * self.head_dim # Q的总维度
        self.kv_size = self.num_kv_heads * self.head_dim # KV的总维度
        self.scaling = self.head_dim**-0.5 # 缩放因子
        self.attn = RadixAttention( # 基数注意力层
            self.num_heads, # 注意力头数
            self.head_dim, # 每个头的维度
            self.scaling, # 缩放因子
            num_kv_heads=self.num_kv_heads, # KV头数
            layer_id=layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("attn", prefix), # 参数名前缀
        )

    def forward( # 前向传播：QKV投影→裁剪→RoPE→注意力→输出投影
        self,
        position_ids: torch.Tensor, # 位置ID
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.Wqkv(hidden_states) # 通过QKV投影层
        if self.clip_qkv is not None: # 如果设置了QKV裁剪
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv) # 裁剪QKV值
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1) # 拆分QKV
        q, k = self.rotary_emb(position_ids, q, k) # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch) # 执行注意力计算
        hidden_states, _ = self.out_proj(attn_output) # 通过输出投影层
        return hidden_states # 返回隐藏状态


class DbrxFusedNormAttention(nn.Module): # DBRX融合归一化注意力类，将归一化和注意力融合在一起
    def __init__( # 初始化融合归一化注意力层
        self,
        config: DbrxConfig, # DBRX配置
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.d_model = config.d_model # 模型维度
        self.attn = DbrxAttention( # 注意力层
            config, # DBRX配置
            layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("attn", prefix), # 参数名前缀
        )
        self.norm_1 = nn.LayerNorm(self.d_model) # 第一个层归一化（注意力前）
        self.norm_2 = nn.LayerNorm(self.d_model) # 第二个层归一化（FFN前）

    def forward( # 前向传播：归一化→注意力→残差→归一化
        self,
        position_ids: torch.Tensor, # 位置ID
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states # 保存残差
        hidden_states = self.norm_1(hidden_states) # 通过第一个层归一化
        x = self.attn( # 通过注意力层
            position_ids=position_ids, # 位置ID
            hidden_states=hidden_states, # 归一化后的隐藏状态
            forward_batch=forward_batch, # 前向批次信息
        )
        hidden_states = residual + x # 残差连接
        residual = hidden_states # 更新残差
        hidden_states = self.norm_2(hidden_states) # 通过第二个层归一化
        return hidden_states, residual # 返回归一化后的隐藏状态和残差


class DbrxBlock(nn.Module): # DBRX块类，包含融合归一化注意力和MoE FFN
    def __init__( # 初始化DBRX块
        self,
        config: DbrxConfig, # DBRX配置
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.norm_attn_norm = DbrxFusedNormAttention( # 融合归一化注意力层
            config, # DBRX配置
            layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("norm_attn_norm", prefix), # 参数名前缀
        )
        self.ffn = DbrxExperts(config, quant_config=quant_config) # MoE FFN层

    def forward( # 前向传播：融合注意力→MoE→残差
        self,
        position_ids: torch.Tensor, # 位置ID
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        hidden_states, residual = self.norm_attn_norm( # 通过融合归一化注意力层
            position_ids=position_ids, # 位置ID
            hidden_states=hidden_states, # 隐藏状态
            forward_batch=forward_batch, # 前向批次信息
        )
        hidden_states = self.ffn(hidden_states) # 通过MoE FFN层
        hidden_states = hidden_states + residual # 残差连接
        return hidden_states # 返回隐藏状态


class DbrxModel(nn.Module): # DBRX模型类，包含词嵌入、多个DBRX块和最终归一化
    def __init__( # 初始化DBRX模型
        self,
        config: DbrxConfig, # DBRX配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.wte = VocabParallelEmbedding( # 词嵌入层
            config.vocab_size, # 词表大小
            config.d_model, # 模型维度
        )
        self.blocks = nn.ModuleList( # 创建DBRX块列表
            [
                DbrxBlock(
                    config, # DBRX配置
                    i, # 块索引
                    quant_config=quant_config, # 量化配置
                    prefix=add_prefix(f"blocks.{i}", prefix), # 参数名前缀
                )
                for i in range(config.n_layers) # 遍历所有层
            ]
        )
        self.norm_f = nn.LayerNorm(config.d_model, eps=1e-5) # 最终层归一化
        for module in self.modules(): # 遍历所有子模块
            if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter): # 如果模块有偏置参数
                # Remove the bias term in Linear and LayerNorm.  # 移除Linear和LayerNorm中的偏置项
                module.register_parameter("bias", None) # 将偏置设为None

    def forward( # 前向传播：词嵌入→多层DBRX块→归一化
        self,
        input_ids: torch.Tensor, # 输入token ID
        position_ids: torch.Tensor, # 位置ID
        forward_batch: ForwardBatch, # 前向批次信息
        input_embeds: torch.Tensor = None, # 输入嵌入（可选）
    ) -> torch.Tensor:
        if input_embeds is None: # 如果没有提供输入嵌入
            hidden_states = self.wte(input_ids) # 通过词嵌入层
        else: # 否则
            hidden_states = input_embeds # 使用提供的嵌入
        for i in range(len(self.blocks)): # 遍历所有块
            block = self.blocks[i] # 获取当前块
            hidden_states = block(position_ids, hidden_states, forward_batch) # 通过当前块
        hidden_states = self.norm_f(hidden_states) # 通过最终层归一化
        return hidden_states # 返回隐藏状态


class DbrxForCausalLM(nn.Module): # DBRX因果语言模型类
    def __init__( # 初始化DBRX因果语言模型
        self,
        config: DbrxConfig, # DBRX配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.quant_config = quant_config # 保存量化配置
        self.unpadded_vocab_size = config.vocab_size # 未填充的词表大小
        self.transformer = DbrxModel( # DBRX Transformer模型
            config, quant_config=quant_config, prefix=add_prefix("transformer", prefix) # 配置、量化配置和前缀
        )
        self.lm_head = ParallelLMHead( # 语言模型输出头
            config.vocab_size, # 词表大小
            config.d_model, # 模型维度
            org_num_embeddings=config.vocab_size, # 原始嵌入数
            padding_size=DEFAULT_VOCAB_PADDING_SIZE, # 填充大小
            prefix=add_prefix("lm_head", prefix), # 参数名前缀
        )
        self.logits_processor = LogitsProcessor(config) # 逻辑处理器

    @torch.no_grad() # 禁用梯度计算
    def forward( # 前向传播：通过Transformer和逻辑处理器
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置索引
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, forward_batch) # 通过Transformer模型
        return self.logits_processor( # 通过逻辑处理器
            input_ids, hidden_states, self.lm_head, forward_batch # 输入ID、隐藏状态、LM头和批次信息
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载模型权重
        expert_params_mapping = [ # 专家参数映射表
            (
                "ws" if weight_name in ["w1", "v1"] else "w2s", # w1/v1映射到ws，w2映射到w2s
                f"experts.mlp.{weight_name}", # 权重名称模式
            )
            for weight_name in ["w1", "v1", "w2"] # 遍历三种权重
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False)) # 获取参数字典（不去重）
        for name, loaded_weight in weights: # 遍历所有权重
            for param_name, weight_name in expert_params_mapping: # 遍历专家参数映射
                if weight_name not in name: # 如果权重名不在参数名中
                    continue # 跳过
                name = name.replace(weight_name, param_name) # 替换权重名为参数名
                param = params_dict[name] # 获取参数
                weight_loader = param.weight_loader # 获取权重加载器
                weight_loader(param, loaded_weight, weight_name) # 加载权重（带权重名）
                break # 跳出内层循环
            else: # 如果没有匹配到专家参数
                # Remapping the name of FP8 kv-scale.  # 重映射FP8 KV缩放的名称
                name = maybe_remap_kv_scale_name(name, params_dict) # 可能重映射KV缩放名称
                if name is None: # 如果名称被过滤
                    continue # 跳过

                param = params_dict[name] # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取权重加载器
                weight_loader(param, loaded_weight) # 加载权重


EntryClass = DbrxForCausalLM # 入口类，用于模型注册
