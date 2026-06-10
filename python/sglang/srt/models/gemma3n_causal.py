# Gemma3n因果语言模型：实现Gemma3n文本因果语言建模，包含RMS归一化、MLP、Laurel块、AltUp模块、注意力、解码层和完整语言模型
from typing import Iterable, Optional, Set, Tuple  # 导入类型提示工具 # import type hints

import torch  # 导入PyTorch库 # import PyTorch
import torch.nn.functional as F  # 导入神经网络函数模块 # import neural network functional module
from torch import nn  # 导入神经网络模块 # import neural network module
from transformers import AutoModel, Gemma3nTextConfig, PretrainedConfig, PreTrainedModel  # 导入transformers相关类 # import transformers classes

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入张量并行世界大小函数 # import TP world size function
from sglang.srt.layers.activation import GeluAndMul  # 导入GELU激活与乘法组合层 # import GELU activation and multiply layer
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层 # import RMS norm layer
from sglang.srt.layers.linear import (  # 导入并行线性层 # import parallel linear layers
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器 # import logits processor
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置 # import quantization config
from sglang.srt.layers.radix_attention import RadixAttention  # 导入Radix注意力层 # import Radix attention
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数 # import get RoPE function
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入词表并行嵌入头 # import vocab parallel LM head
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # import forward batch info
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具 # import weight loading utilities
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma3_causal import Gemma3TextScaledWordEmbedding  # 导入Gemma3缩放词嵌入 # import Gemma3 scaled word embedding
from sglang.srt.utils import add_prefix, make_layers  # 导入工具函数 # import utility functions


# Aligned with HF's implementation, using sliding window inclusive with the last token
# SGLang assumes exclusive
# 与HuggingFace实现对齐，滑动窗口包含最后一个token；SGLang假设不包含（排他）
def get_attention_sliding_window_size(config):  # 获取注意力滑动窗口大小 # get attention sliding window size
    return config.sliding_window - 1  # 返回滑动窗口大小减1 # return sliding window minus 1


class Gemma3nRMSNorm(RMSNorm):  # Gemma3n RMS归一化层，继承自RMSNorm # Gemma3n RMS norm layer, inherits from RMSNorm
    def __init__(  # 初始化方法 # initialization method
        self,
        dim: int,  # 归一化维度 # normalization dimension
        eps: float = 1e-6,  # 防止除零的小常数 # epsilon to prevent division by zero
        with_scale: bool = True,  # 是否使用可学习缩放参数 # whether to use learnable scale parameter
    ) -> None:
        super().__init__(dim, eps=eps)  # 调用父类初始化 # call parent class init
        if not with_scale:  # 不使用缩放 # without scale
            del self.weight  # 删除权重 # delete weight
            self.register_buffer(  # 注册不可训练的全1缓冲区 # register non-trainable all-ones buffer
                "weight",
                torch.ones(dim, dtype=torch.get_default_dtype()),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法 # forward pass method
        original_shape = x.shape  # 保存原始形状 # save original shape
        x_2d = x.contiguous().reshape(-1, original_shape[-1])  # 重塑为2D # reshape to 2D
        x_2d = super().forward(x_2d)  # 调用父类前向传播 # call parent forward
        x = x_2d.reshape(original_shape)  # 恢复原始形状 # restore original shape
        return x  # 返回归一化结果 # return normalized result


class Gemma3nTextScaledWordEmbedding(Gemma3TextScaledWordEmbedding):  # Gemma3n缩放词嵌入，继承自Gemma3版本 # Gemma3n scaled word embedding, inherits from Gemma3 version
    pass  # 直接继承，无额外实现 # direct inheritance, no extra implementation


class Gemma3nTextMLP(nn.Module):  # Gemma3n文本MLP模块 # Gemma3n text MLP module
    def __init__(  # 初始化方法 # initialization method
        self,
        hidden_size: int,  # 隐藏层大小 # hidden layer size
        intermediate_size: int,  # 中间层大小 # intermediate layer size
        hidden_activation: str,  # 隐藏层激活函数名 # hidden activation function name
        activation_sparsity: float = 0.0,  # 激活稀疏度 # activation sparsity
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ) -> None:
        super().__init__()  # 调用父类初始化 # call parent class init
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控和上投影合并的并行线性层 # merged gate and up projection parallel linear layer
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(  # 下投影行并行线性层 # down projection row parallel linear layer
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_activation != "gelu_pytorch_tanh":  # 检查激活函数 # check activation function
            raise ValueError(  # 抛出值错误 # raise value error
                "Gemma3n uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )
        # Use proper GELU with tanh approximation as specified
        # 使用指定的tanh近似GELU
        self.act_fn = GeluAndMul()  # GELU激活与乘法组合函数 # GELU activation and multiply function
        self.activation_sparsity = activation_sparsity  # 保存激活稀疏度 # save activation sparsity
        self.register_buffer(  # 注册目标稀疏度张量缓冲区 # register target sparsity tensor buffer
            "target_sparsity_tensor",
            torch.tensor(self.activation_sparsity, dtype=torch.float32),
            persistent=False,
        )  # moved from _gaussian_topk for cuda graph
        # 从_gaussian_topk中移出以支持CUDA graph

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法 # forward pass method
        gate_up, _ = self.gate_up_proj(x)  # 通过门控上投影层 # through gate up projection layer

        # Split gate and up projections
        # 分割门控和上投影
        gate_proj, up_proj = gate_up.chunk(2, dim=-1)  # 分割为门控和上投影 # split into gate and up projections

        # Apply activation sparsity if needed
        # 如果需要则应用激活稀疏化
        if self.activation_sparsity > 0.0:  # 需要稀疏化 # need sparsity
            gate_proj = self._gaussian_topk(gate_proj)  # 高斯TopK稀疏化 # Gaussian topk sparsification

        gate_up = torch.cat([gate_proj, up_proj], dim=-1)  # 重新拼接 # re-concatenate

        # Apply GELU activation to gate projection and multiply with up projection
        # 对门控投影应用GELU激活并与上投影相乘
        x = self.act_fn(gate_up)  # GELU激活与乘法 # GELU activation and multiply
        x, _ = self.down_proj(x)  # 通过下投影层 # through down projection layer
        return x  # 返回输出 # return output

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:  # 高斯TopK稀疏化方法 # Gaussian topk sparsification method
        normal_dist = torch.distributions.normal.Normal(0, 1)  # 标准正态分布 # standard normal distribution
        std_multiplier = normal_dist.icdf(self.target_sparsity_tensor)  # 逆CDF计算标准差乘数 # compute std multiplier via inverse CDF
        std_multiplier = std_multiplier.type(inputs.dtype)  # 转换类型 # convert dtype
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)  # 输入均值 # input mean
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)  # 输入标准差 # input std
        cutoff_x = inputs_mean + inputs_std * std_multiplier  # 截断阈值 # cutoff threshold
        return F.relu(inputs - cutoff_x)  # ReLU截断低于阈值的部分 # ReLU to cut below threshold


class Gemma3nLaurelBlock(nn.Module):  # Gemma3n LAUREL块（学习增强残差层） # Gemma3n LAUREL block (Learned Augmented Residual Layer)
    """Learned Augmented Residual Layer"""  # 学习增强残差层 # Learned Augmented Residual Layer

    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nTextConfig,  # Gemma3n文本配置 # Gemma3n text config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ):
        super().__init__()  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config

        self.linear_left = ColumnParallelLinear(  # 左侧线性层（降维） # left linear layer (dimension reduction)
            config.hidden_size,
            config.laurel_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("linear_left", prefix),
        )
        self.linear_right = RowParallelLinear(  # 右侧线性层（升维） # right linear layer (dimension increase)
            config.laurel_rank,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("linear_right", prefix),
        )
        self.post_laurel_norm = Gemma3nRMSNorm(  # LAUREL后归一化 # post-LAUREL norm
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法 # forward pass method
        # [num_tokens, hidden_size]
        # [token数, 隐藏维度]
        laurel_x, _ = self.linear_left(x)  # 左侧降维 # left dimension reduction
        laurel_x, _ = self.linear_right(laurel_x)  # 右侧升维 # right dimension increase
        normed_laurel_x = self.post_laurel_norm(laurel_x)  # 归一化 # normalize
        return x + normed_laurel_x  # 残差连接 # residual connection


class Gemma3nAltUp(nn.Module):  # Gemma3n AltUp模块（交替更新） # Gemma3n AltUp module (Alternating Updates)
    """Alternating Updates (AltUp)"""  # 交替更新 # Alternating Updates

    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nTextConfig,  # Gemma3n文本配置 # Gemma3n text config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ):
        super().__init__()  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config

        self.correct_output_scale = nn.Parameter(  # 修正输出缩放参数 # correction output scale parameter
            torch.zeros(config.hidden_size, dtype=torch.float32)
        )
        self.correction_coefs = ReplicatedLinear(  # 修正系数线性层 # correction coefficients linear layer
            config.altup_num_inputs,
            config.altup_num_inputs,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("correction_coefs", prefix),
        )
        self.prediction_coefs = ReplicatedLinear(  # 预测系数线性层 # prediction coefficients linear layer
            config.altup_num_inputs,
            config.altup_num_inputs**2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("prediction_coefs", prefix),
        )
        self.modality_router = ReplicatedLinear(  # 模态路由线性层 # modality router linear layer
            config.hidden_size,
            config.altup_num_inputs,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("modality_router", prefix),
        )

        self.router_norm = Gemma3nRMSNorm(  # 路由归一化层 # router norm layer
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.register_buffer(  # 注册路由输入缩放缓冲区 # register router input scale buffer
            "router_input_scale",
            torch.tensor(config.hidden_size**-1.0),
            persistent=False,
        )

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:  # 计算路由模态 # compute router modalities
        # x  : [num_tokens, hidden_size]
        # x : [token数, 隐藏维度]
        router_inputs = self.router_norm(x) * self.router_input_scale.to(  # 归一化并缩放 # normalize and scale
            self.router_norm.weight.dtype
        )
        # router_inputs : [num_tokens, hidden_size]
        # router_inputs : [token数, 隐藏维度]
        routed, _ = self.modality_router(router_inputs)  # 通过模态路由 # through modality router

        # routed : [num_tokens, altup_num_inputs]
        # routed : [token数, altup输入数]
        return torch.tanh(routed.float()).type_as(routed)  # tanh截断并转回原类型 # tanh clipping and convert back

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 预测方法 # predict method
        """Predicts the output of a layer using a trainable map.
        hidden_states: [num_altup_inputs, num_tokens, hidden_size]
        """
        # 使用可训练映射预测层的输出
        # hidden_states: [altup输入数, token数, 隐藏维度]
        modalities = self.compute_router_modalities(
            hidden_states[self.config.altup_active_idx]
        )  # (n_tokens, altup_num_inputs)
        # 计算活跃输入的模态
        # TODO: CHECK DO WE NEED THIS: self.prediction_coefs.float()  # Force computation in float32, in-place operation
        # 待检查：是否需要强制float32计算

        if self.config.altup_coef_clip is not None:  # 有系数裁剪 # has coefficient clipping
            self.prediction_coefs.weight.data.clamp_(  # 裁剪预测系数 # clip prediction coefficients
                -self.config.altup_coef_clip, self.config.altup_coef_clip
            )

        all_coefs, _ = self.prediction_coefs(
            modalities
        )  # (n_tokens, altup_num_inputs) -> (n_tokens, altup_num_inputs**2)
        # 计算预测系数

        all_coefs = all_coefs.reshape(  # 重塑系数矩阵 # reshape coefficient matrix
            *modalities.shape[:-1],
            self.config.altup_num_inputs,
            self.config.altup_num_inputs,
        ).permute(0, 2, 1)

        # permute hidden_states from [num_altup_inputs, num_tokens, hidden_size] to [num_tokens, hidden_size, altup_num_inputs]
        # 将hidden_states从[num_altup_inputs, num_tokens, hidden_size]置换为[num_tokens, hidden_size, altup_num_inputs]
        predictions = torch.matmul(hidden_states.permute(1, 2, 0), all_coefs)  # 矩阵乘法预测 # matrix multiply prediction
        predictions = predictions.permute(2, 0, 1)  # undo the permute # 撤销置换
        predictions += hidden_states  # add the original input # 加上原始输入
        return predictions.contiguous().type_as(
            hidden_states
        )  # [num_altup_inputs, num_tokens, hidden_size]
        # 返回连续且类型一致的预测结果

    def correct(  # 修正方法 # correction method
        self, predictions: torch.Tensor, activated: torch.Tensor
    ) -> torch.Tensor:
        """Corrects the predictions relative to the activated inputs."""
        # 相对于激活输入修正预测
        # prediction : [num_altup_inputs, num_tokens, hidden_size]
        # activated  : [num_tokens, hidden_size]
        modalities = self.compute_router_modalities(
            activated
        )  # [num_tokens, altup_num_inputs]
        # 计算激活输入的模态
        innovation = (
            activated - predictions[self.config.altup_active_idx]
        )  # [num_tokens, hidden_size]
        # 计算创新值（激活与预测的差）
        innovation = innovation.repeat(
            self.config.altup_num_inputs, 1, 1
        )  # (self.config.altup_num_inputs, num_tokens, hidden_size)
        # 重复创新值以匹配所有altup输入

        if self.config.altup_coef_clip is not None:  # 有系数裁剪 # has coefficient clipping
            self.correction_coefs.weight.data.clamp_(  # 裁剪修正系数 # clip correction coefficients
                -self.config.altup_coef_clip, self.config.altup_coef_clip
            )

        all_coefs, _ = self.correction_coefs(
            modalities
        )  # [num_tokens, altup_num_inputs]
        # 计算修正系数
        all_coefs = (all_coefs + 1.0).permute(1, 0).unsqueeze(-1)  # 加1并调整形状 # add 1 and adjust shape
        # # [num_tokens, altup_num_inputs, 1]

        corrected = torch.mul(innovation, all_coefs)  # 逐元素乘法 # element-wise multiplication
        corrected += predictions  # 加上预测值 # add predictions
        return corrected.contiguous().type_as(activated)  # 返回修正后的结果 # return corrected result

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:  # 缩放修正输出 # scale corrected output
        """Scales the provided 3D tensor."""  # 缩放提供的3D张量 # Scales the provided 3D tensor
        return corrected * self.correct_output_scale.to(corrected.dtype)  # 乘以缩放参数 # multiply by scale parameter

    def forward(  # 前向传播方法 # forward pass method
        self, hidden_states: torch.Tensor, activated: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts, correct, and optionally scales the output of a layer using trainable maps.

        hidden_states: [num_altup_inputs, num_tokens, hidden_size]
        """
        # 预测、修正并可选缩放层的输出
        # hidden_states: [altup输入数, token数, 隐藏维度]

        predictions = self.predict(hidden_states)  # 预测 # predict
        corrected = self.correct(predictions=predictions, activated=activated)  # 修正 # correct
        output = corrected[self.config.altup_active_idx]  # 取活跃索引的输出 # take active index output
        if self.config.altup_correct_scale:  # 需要缩放 # need scaling
            output = self.scale_corrected_output(output)  # 缩放输出 # scale output
        return corrected, output  # 返回修正后的状态和活跃输出 # return corrected states and active output


class Gemma3nAttention(nn.Module):  # Gemma3n注意力模块 # Gemma3n attention module
    """Multi-headed attention from 'Attention Is All You Need' paper"""  # 来自"Attention Is All You Need"论文的多头注意力 # Multi-headed attention from 'Attention Is All You Need' paper

    def __init__(  # 初始化方法 # initialization method
        self,
        layer_id: int,  # 层ID # layer ID
        config: Gemma3nTextConfig,  # Gemma3n文本配置 # Gemma3n text config
        max_position_embeddings: int,  # 最大位置编码数 # max position embeddings
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ) -> None:
        super().__init__()  # 调用父类初始化 # call parent class init
        self.layer_id = layer_id  # 保存层ID # save layer ID
        self.config = config  # 保存配置 # save config
        tp_size = get_tensor_model_parallel_world_size()  # 张量并行世界大小 # tensor parallel world size

        self.total_num_heads = config.num_attention_heads  # 总注意力头数 # total number of attention heads
        assert self.total_num_heads % tp_size == 0  # 断言头数能被并行度整除 # assert heads divisible by parallelism
        self.num_heads = self.total_num_heads // tp_size  # 每个并行单元的头数 # heads per parallel unit
        self.total_num_kv_heads = config.num_key_value_heads  # 总KV头数 # total KV heads

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个并行单元的KV头数 # KV heads per parallel unit

        if self.total_num_kv_heads >= tp_size:  # KV头数大于等于并行度 # KV heads >= parallelism
            assert self.total_num_kv_heads % tp_size == 0  # 断言可整除 # assert divisible
        else:  # KV头数小于并行度 # KV heads < parallelism
            assert tp_size % self.total_num_kv_heads == 0  # 断言并行度可被KV头数整除 # assert parallelism divisible by KV heads

        hidden_size = config.hidden_size  # 隐藏层大小 # hidden size
        head_dim = getattr(  # 获取头维度 # get head dimension
            config, "head_dim", hidden_size // config.num_attention_heads
        )
        self.head_dim = head_dim  # 保存头维度 # save head dimension

        self.q_size = self.num_heads * self.head_dim  # 查询向量大小 # query vector size
        self.kv_size = self.num_kv_heads * self.head_dim  # KV向量大小 # KV vector size
        # self.scaling = config.query_rescale_scalar / config.query_pre_attn_scalar
        # 缩放因子（已注释）
        self.scaling = 1.0  # 缩放因子设为1.0 # scaling factor set to 1.0

        self.qkv_proj = QKVParallelLinear(  # QKV并行投影层 # QKV parallel projection layer
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(  # 输出投影层 # output projection layer
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # Determine if layer uses sliding window based on pattern
        # 根据模式判断层是否使用滑动窗口
        self.is_sliding = config.layer_types[layer_id] == "sliding_attention"  # 是否为滑动注意力 # whether sliding attention

        # Check if this is a KV shared layer
        # 检查是否为KV共享层
        first_kv_shared_layer_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
        )  # 第一个KV共享层的索引 # first KV shared layer index
        self.is_kv_shared_layer = layer_id >= first_kv_shared_layer_idx  # 是否为KV共享层 # whether KV shared layer

        # Compute the layer index from which shared KV cache values will be retrieved
        # 计算共享KV缓存值的来源层索引
        if not self.is_kv_shared_layer:  # 非KV共享层 # not KV shared layer
            self.kv_shared_layer_index = None  # 无共享索引 # no shared index
        elif self.is_sliding:  # 滑动注意力的KV共享层 # sliding attention KV shared layer
            self.kv_shared_layer_index = first_kv_shared_layer_idx - 2  # 前两个位置 # two positions before
        else:  # 全局注意力的KV共享层 # global attention KV shared layer
            self.kv_shared_layer_index = first_kv_shared_layer_idx - 1  # 前一个位置 # one position before

        if self.is_sliding:  # 滑动注意力 # sliding attention
            self.rotary_emb = get_rope(  # 创建局部旋转位置编码 # create local rotary position embedding
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_local_base_freq,
                rope_scaling={"rope_type": "default"},
            )
        else:  # 全局注意力 # global attention
            self.rotary_emb = get_rope(  # 创建全局旋转位置编码 # create global rotary position embedding
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_parameters["rope_theta"],
                rope_scaling=config.rope_parameters,
            )

        self.sliding_window = config.sliding_window if self.is_sliding else None  # 滑动窗口大小 # sliding window size

        self.attn = RadixAttention(  # 创建Radix注意力层 # create Radix attention layer
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=(
                layer_id if not self.is_kv_shared_layer else self.kv_shared_layer_index
            ),  # KV共享层使用共享的层ID # KV shared layer uses shared layer ID
            logit_cap=0.0,
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        # Gemma3n adds normalization for q, k, v
        # Gemma3n对q, k, v添加归一化
        self.q_norm = Gemma3nRMSNorm(  # 查询归一化层 # query norm layer
            dim=config.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Gemma3nRMSNorm(  # 键归一化层 # key norm layer
            dim=config.head_dim,
            eps=config.rms_norm_eps,
        )
        self.v_norm = Gemma3nRMSNorm(  # 值归一化层（无可学习缩放） # value norm layer (no learnable scale)
            dim=config.head_dim,
            eps=config.rms_norm_eps,
            with_scale=False,
        )

    def forward(  # 前向传播方法 # forward pass method
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量 # hidden states tensor
        positions: Tuple[torch.Tensor, torch.Tensor],  # 位置元组 # positions tuple
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        **kwargs,
    ) -> torch.Tensor:

        qkv, _ = self.qkv_proj(hidden_states)  # 计算QKV投影 # compute QKV projection
        # TODO: for first 20 layers, we use QKVParallelLinear
        #       for others, we only calc Q.
        # 待办：前20层使用QKVParallelLinear，其余仅计算Q
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割QKV # split QKV

        # Apply normalization to q, k, v
        # 对q, k, v应用归一化
        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # 重塑Q # reshape Q
        q = self.q_norm(q)  # Q归一化 # Q normalization

        # Check if we should use shared KV cache
        # 检查是否应使用共享KV缓存
        if self.is_kv_shared_layer and self.kv_shared_layer_index is not None:  # KV共享层 # KV shared layer
            # For KV shared layers, we skip K/V computation and normalization
            # The RadixAttention will handle retrieving shared KV from cache
            # 对于KV共享层，跳过K/V计算和归一化，RadixAttention将从缓存中获取共享KV
            k = None  # K设为None # set K to None
            v = None  # V设为None # set V to None
        else:  # 非KV共享层 # non-KV shared layer
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))  # 重塑K # reshape K
            k = self.k_norm(k)  # K归一化 # K normalization

            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))  # 重塑V # reshape V
            v = self.v_norm(v)  # V归一化 # V normalization

        # Flatten back for rotary embedding
        # 展平以应用旋转位置编码
        q = q.flatten(-2, -1)  # 展平Q # flatten Q

        # Apply rotary embedding
        # 应用旋转位置编码
        if k is not None:  # K存在 # K exists
            k = k.flatten(-2, -1)  # 展平K # flatten K
            q, k = self.rotary_emb(positions, q, k)  # 应用旋转编码 # apply rotary embedding
            # Reshape k back to head format for attention
            # 将K重塑回头格式以进行注意力计算
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))  # 重塑K # reshape K
        else:  # K不存在（KV共享层） # K is None (KV shared layer)
            # For shared KV layers, create a dummy key for rotary embedding and discard it
            # 对于共享KV层，创建一个虚拟键用于旋转编码然后丢弃
            dummy_k = torch.zeros_like(
                q[:, : self.kv_size]
            )  # Create dummy key with same shape as needed
            # 创建所需形状的虚拟键
            q, _ = self.rotary_emb(positions, q, dummy_k)  # 应用旋转编码（忽略虚拟K） # apply rotary embedding (ignore dummy K)

        # Reshape q back to head format for attention
        # 将Q重塑回头格式以进行注意力计算
        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # 重塑Q # reshape Q

        attn_output = self.attn(  # 计算注意力输出 # compute attention output
            q,
            k,
            v,
            forward_batch=forward_batch,
            save_kv_cache=not self.is_kv_shared_layer,  # 非共享层才保存KV缓存 # save KV cache only for non-shared layers
        )

        output, _ = self.o_proj(attn_output)  # 通过输出投影层 # through output projection layer
        return output  # 返回输出 # return output


class Gemma3nDecoderLayer(nn.Module):  # Gemma3n解码器层 # Gemma3n decoder layer
    def __init__(  # 初始化方法 # initialization method
        self,
        layer_id: int,  # 层ID # layer ID
        config: PretrainedConfig,  # 预训练配置 # pretrained config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ) -> None:
        super().__init__()  # 调用父类初始化 # call parent class init
        self.hidden_size = config.hidden_size  # 隐藏层大小 # hidden size
        self.layer_id = layer_id  # 保存层ID # save layer ID
        self.attention_type = config.layer_types[layer_id]  # 注意力类型 # attention type
        self.config = config  # 保存配置 # save config

        self.self_attn = Gemma3nAttention(  # 自注意力层 # self attention layer
            layer_id=layer_id,
            config=config,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        intermediate_size = config.intermediate_size[layer_id]  # 该层的中间层大小 # intermediate size for this layer
        activation_sparsity = config.activation_sparsity_pattern[layer_id]  # 该层的激活稀疏度 # activation sparsity for this layer
        self.mlp = Gemma3nTextMLP(  # MLP模块 # MLP module
            hidden_size=self.hidden_size,
            intermediate_size=intermediate_size,
            hidden_activation=config.hidden_activation,
            activation_sparsity=activation_sparsity,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化 # input layer norm
        self.post_attention_layernorm = Gemma3nRMSNorm(  # 注意力后归一化 # post attention norm
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3nRMSNorm(  # 前馈前归一化 # pre feedforward norm
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3nRMSNorm(  # 前馈后归一化 # post feedforward norm
            self.hidden_size, eps=config.rms_norm_eps
        )

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input  # 每层输入的隐藏维度 # per-layer input hidden size

        self.altup = Gemma3nAltUp(  # AltUp模块 # AltUp module
            config, quant_config, prefix=add_prefix("altup", prefix)
        )
        self.laurel = Gemma3nLaurelBlock(  # LAUREL块 # LAUREL block
            config, quant_config, prefix=add_prefix("laurel", prefix)
        )

        self.per_layer_input_gate = ReplicatedLinear(  # 每层输入门控线性层 # per-layer input gate linear layer
            self.hidden_size,
            self.hidden_size_per_layer_input,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("per_layer_input_gate", prefix),
        )
        self.per_layer_projection = ReplicatedLinear(  # 每层投影线性层 # per-layer projection linear layer
            self.hidden_size_per_layer_input,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("per_layer_projection", prefix),
        )
        self.post_per_layer_input_norm = Gemma3nRMSNorm(  # 每层输入后归一化 # post per-layer input norm
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.is_sliding = self.self_attn.is_sliding  # 是否为滑动注意力 # whether sliding attention

    def forward(  # 前向传播方法 # forward pass method
        self,
        positions: torch.Tensor,  # 位置编码张量 # position encoding tensor
        hidden_states: torch.Tensor,  # 隐藏状态张量 # hidden states tensor
        per_layer_input: torch.Tensor,  # 每层输入张量 # per-layer input tensor
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        **kwargs,
    ) -> torch.Tensor:
        predictions = self.altup.predict(
            hidden_states
        )  # [num_altup_inputs, num_tokens, hidden_size]
        # AltUp预测
        active_prediction = predictions[self.config.altup_active_idx]  # 活跃预测 # active prediction

        active_prediction_normed = self.input_layernorm(active_prediction)  # 输入层归一化 # input layer norm
        laurel_output = self.laurel(
            active_prediction_normed
        )  # laurel_output: [num_tokens, hidden_size]
        # LAUREL输出
        # active_prediction: [num_tokens, hidden_size]

        attn = self.self_attn(  # 自注意力计算 # self attention computation
            positions=positions,
            hidden_states=active_prediction_normed,
            forward_batch=forward_batch,
            **kwargs,
        )
        attn = self.post_attention_layernorm(attn)  # [num_tokens, hidden_size] # 注意力后归一化

        attn_gated = active_prediction + attn  # [num_tokens, hidden_size] # 门控注意力（残差） # gated attention (residual)
        attn_laurel = (attn_gated + laurel_output) / torch.sqrt(torch.tensor(2.0))  # 合并注意力和LAUREL # combine attention and LAUREL

        attn_norm = self.pre_feedforward_layernorm(
            attn_laurel
        )  # [num_tokens, hidden_size]
        # 前馈前归一化
        attn_ffw = self.mlp(attn_norm)  # [num_tokens, hidden_size] # MLP前馈 # MLP feed-forward
        attn_ffw_norm = self.post_feedforward_layernorm(
            attn_ffw
        )  # [num_tokens, hidden_size]
        # 前馈后归一化
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm  # [num_tokens, hidden_size] # 前馈和LAUREL门控 # feed-forward and LAUREL gated
        corrected_predictions = self.altup.correct(
            predictions, attn_ffw_laurel_gated
        )  # prediction : [num_altup_inputs, num_tokens, hidden_size]
        # AltUp修正
        # attn_ffw_laurel_gated: [num_tokens, hidden_size]
        first_prediction = corrected_predictions[self.config.altup_active_idx]  # 活跃预测 # active prediction

        if self.config.altup_correct_scale:  # 需要缩放 # need scaling
            first_prediction = self.altup.scale_corrected_output(first_prediction)  # 缩放 # scale

        # per_layer_input_gate
        # 每层输入门控
        first_prediction = first_prediction.to(self.per_layer_input_gate.weight.dtype)  # 类型转换 # dtype conversion
        first_prediction, _ = self.per_layer_input_gate(first_prediction)  # 门控线性层 # gate linear layer
        first_prediction = F.gelu(first_prediction, approximate="tanh")  # GELU激活 # GELU activation
        first_prediction = torch.multiply(first_prediction, per_layer_input)  # 逐元素乘法 # element-wise multiply

        # per_layer_projection
        # 每层投影
        first_prediction, _ = self.per_layer_projection(first_prediction)  # 投影线性层 # projection linear layer
        first_prediction = self.post_per_layer_input_norm(first_prediction)  # 归一化 # normalize
        corrected_predictions[1:] += first_prediction  # 加到非活跃预测上 # add to non-active predictions

        return corrected_predictions  # 返回修正后的预测 # return corrected predictions


class Gemma3nTextModel(PreTrainedModel):  # Gemma3n文本模型类 # Gemma3n text model class
    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nTextConfig,  # Gemma3n文本配置 # Gemma3n text config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ) -> None:
        super().__init__(config=config)  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config
        self.quant_config = quant_config  # 保存量化配置 # save quantization config
        self.vocab_size = config.vocab_size  # 词表大小 # vocab size
        self.padding_idx = config.pad_token_id  # 填充token ID # padding token ID

        # Gemma3n downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # Gemma3n将以下值向下转换为float16，导致sqrt(3072)=55.4256变为55.5
        self.embed_tokens = Gemma3nTextScaledWordEmbedding(  # 缩放词嵌入层 # scaled word embedding layer
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )

        self.norm = Gemma3nRMSNorm(  # RMS归一化层 # RMS norm layer
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.layers = make_layers(  # 创建解码器层列表 # create decoder layer list
            config.num_hidden_layers,
            lambda idx, prefix: Gemma3nDecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )

        # Per-layer input embeddings
        # 每层输入嵌入
        self.hidden_size = config.hidden_size  # 隐藏层大小 # hidden size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input  # 每层输入隐藏维度 # per-layer input hidden size

        self.embed_tokens_per_layer = Gemma3nTextScaledWordEmbedding(  # 每层词嵌入 # per-layer word embedding
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            self.padding_idx,
            embed_scale=self.config.hidden_size_per_layer_input**0.5,
        )

        self.per_layer_model_projection = ColumnParallelLinear(  # 每层模型投影 # per-layer model projection
            self.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("per_layer_model_projection", prefix),
        )

        self.per_layer_projection_norm = Gemma3nRMSNorm(  # 每层投影归一化 # per-layer projection norm
            dim=config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
        )

        self.altup_projections = make_layers(  # AltUp投影层 # AltUp projection layers
            self.config.altup_num_inputs - 1,
            lambda idx, prefix: ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                gather_output=True,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("altup_projections", prefix),
        )

        self.altup_unembed_projections = make_layers(  # AltUp反嵌入投影层 # AltUp unembed projection layers
            self.config.altup_num_inputs - 1,
            lambda idx, prefix: ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                gather_output=True,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("altup_unembed_projections", prefix),
        )

        self.register_buffer(  # 注册每层投影缩放缓冲区 # register per-layer projection scale buffer
            "per_layer_projection_scale",
            torch.tensor(self.hidden_size**-0.5),
            persistent=False,
        )
        self.register_buffer(  # 注册每层输入缩放缓冲区 # register per-layer input scale buffer
            "per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False
        )

        self.post_init()  # 调用后初始化 # call post init

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层 # get input embedding layer
        return self.embed_tokens  # 返回词嵌入层 # return word embedding layer

    def dtype(self) -> torch.dtype:  # 获取模型数据类型 # get model data type
        return next(self.parameters()).dtype  # 返回第一个参数的数据类型 # return dtype of first parameter

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:  # 获取每层输入 # get per-layer inputs
        embeddings = self.embed_tokens_per_layer(input_ids)  # 通过每层词嵌入获取 # get through per-layer word embedding
        return embeddings.reshape(  # 重塑为层维度 # reshape to layer dimension
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(  # 投影每层输入 # project per-layer inputs
        self,
        inputs_embeds: torch.Tensor,  # 输入嵌入 # input embeddings
        per_layer_inputs: Optional[torch.Tensor] = None,  # 每层输入，可选 # per-layer inputs, optional
    ) -> torch.Tensor:
        per_layer_projection, _ = self.per_layer_model_projection(inputs_embeds)  # 模型投影 # model projection
        per_layer_projection *= self.per_layer_projection_scale.type(  # 应用缩放 # apply scale
            inputs_embeds.dtype
        )
        per_layer_projection = per_layer_projection.reshape(  # 重塑为层维度 # reshape to layer dimension
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)  # 归一化 # normalize

        if per_layer_inputs is None:  # 无每层输入 # no per-layer inputs
            return per_layer_projection  # 仅返回投影 # return projection only

        if per_layer_projection.shape != per_layer_inputs.shape:  # 形状不匹配 # shape mismatch
            # per-layer inputs are sometimes padded with zeros, slice the relevant embeddings
            # 每层输入有时用零填充，切片相关嵌入
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]  # 切片 # slice

        return (
            per_layer_projection + per_layer_inputs
        ) * self.per_layer_input_scale.type(inputs_embeds.dtype)  # 求和并缩放 # sum and scale

    def forward(  # 前向传播方法 # forward pass method
        self,
        input_ids: torch.Tensor,  # 输入token ID张量 # input token ID tensor
        positions: torch.Tensor,  # 位置编码张量 # position encoding tensor
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选 # input embeddings, optional
        per_layer_inputs: Optional[torch.Tensor] = None,  # 每层输入，可选 # per-layer inputs, optional
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (input_embeds is not None):  # 恰好一个为None # exactly one is None
            raise ValueError(  # 抛出值错误 # raise value error
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if input_ids is not None:  # 有输入ID # has input IDs
            input_embeds = self.embed_tokens(input_ids)  # 通过词嵌入层获取 # get through word embedding layer
            per_layer_inputs = self.get_per_layer_inputs(input_ids)  # 获取每层输入 # get per-layer inputs

        per_layer_inputs = self.project_per_layer_inputs(input_embeds, per_layer_inputs)  # 投影每层输入 # project per-layer inputs

        # Expand hidden_states to support per-layer inputs
        # 扩展隐藏状态以支持每层输入
        target_magnitude = torch.mean(input_embeds**2, dim=-1, keepdim=True) ** 0.5  # 目标幅度 # target magnitude
        epsilon_tensor = torch.tensor(torch.finfo(input_embeds.dtype).min)  # 极小值 # epsilon value

        # embed positions
        # 嵌入位置
        hidden_states_0 = input_embeds  # 初始隐藏状态 # initial hidden states
        temp_hidden_states = [hidden_states_0]  # 临时隐藏状态列表 # temp hidden states list

        for i in range(1, self.config.altup_num_inputs):  # 遍历AltUp额外输入 # iterate AltUp extra inputs
            altup_proj, _ = self.altup_projections[i - 1](hidden_states_0)  # AltUp投影 # AltUp projection
            current_hidden_state = altup_proj.type(hidden_states_0.dtype)  # 类型转换 # dtype conversion
            new_magnitude = (  # 新幅度 # new magnitude
                torch.mean(current_hidden_state**2, dim=-1, keepdim=True) ** 0.5
            )
            current_hidden_state = current_hidden_state * (  # 幅度归一化 # magnitude normalization
                target_magnitude / torch.maximum(new_magnitude, epsilon_tensor)
            )
            temp_hidden_states.append(current_hidden_state)  # 添加到列表 # append to list

        hidden_states = torch.stack(
            temp_hidden_states, dim=0
        )  # [num_altup_inputs, n_tokens, hidden_size]
        # 堆叠所有AltUp输入

        for layer_idx, layer in enumerate(self.layers):  # 遍历每层 # iterate each layer
            per_layer_input = per_layer_inputs[:, layer_idx, :]  # 该层的输入 # input for this layer
            hidden_states = layer(  # 通过解码器层 # through decoder layer
                positions=positions,
                per_layer_input=per_layer_input,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                **kwargs,
            )

        # Per-layer inputs to single output
        # 将每层输入合并为单一输出
        target_magnitude = (  # 目标幅度 # target magnitude
            torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        )

        temp_hidden_states = [hidden_states[0]]  # 初始隐藏状态 # initial hidden state

        for i in range(1, self.config.altup_num_inputs):  # 遍历AltUp额外输入 # iterate AltUp extra inputs
            # altup_unembed_projections adapted from jax.numpy.einsum("btp,pd->btd", ...)
            # altup_unembed_projections适配自jax.numpy.einsum("btp,pd->btd", ...)
            altup_unemb_proj, _ = self.altup_unembed_projections[i - 1](
                hidden_states[i]
            )
            current_hidden_state = altup_unemb_proj.type(hidden_states_0.dtype)  # 类型转换 # dtype conversion
            new_magnitude = (  # 新幅度 # new magnitude
                torch.mean(current_hidden_state**2, dim=-1, keepdim=True) ** 0.5
            )
            current_hidden_state = current_hidden_state * (  # 幅度归一化 # magnitude normalization
                target_magnitude / torch.maximum(new_magnitude, epsilon_tensor)
            )
            temp_hidden_states.append(current_hidden_state)  # 添加到列表 # append to list

        hidden_states = torch.stack(temp_hidden_states)  # 堆叠 # stack
        hidden_states = torch.mean(hidden_states, dim=0)  # 取均值 # take mean
        hidden_states = self.norm(hidden_states)  # 最终归一化 # final normalization

        return hidden_states  # 返回隐藏状态 # return hidden states


class Gemma3nForCausalLM(PreTrainedModel):  # Gemma3n因果语言模型类 # Gemma3n causal language model class
    config_class = Gemma3nTextConfig  # 配置类 # config class

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 绑定权重的键 # tied weight keys
    _tp_plan = {"lm_head": "colwise_rep"}  # 张量并行计划 # tensor parallel plan
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}  # 流水线并行计划 # pipeline parallel plan
    config_class = Gemma3nTextConfig  # 配置类 # config class
    base_model_prefix = "language_model"  # 基础模型前缀 # base model prefix

    # BitandBytes specific attributes
    # BitandBytes特定属性
    default_bitsandbytes_target_modules = [  # 默认BitandBytes目标模块 # default BitandBytes target modules
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes堆叠参数映射 # BitandBytes stacked params mapping
        ".q_proj": (".qkv_proj", 0),
        ".k_proj": (".qkv_proj", 1),
        ".v_proj": (".qkv_proj", 2),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    packed_modules_mapping = {  # 打包模块映射 # packed modules mapping
        ".qkv_proj": [
            ".q_proj",
            ".k_proj",
            ".v_proj",
        ],
        ".gate_up_proj": [
            ".gate_proj",
            ".up_proj",
        ],
    }

    # LoRA specific attributes
    # LoRA特定属性
    supported_lora_modules = [  # 支持LoRA的模块 # LoRA supported modules
        ".qkv_proj",
        ".o_proj",
        ".gate_up_proj",
        ".down_proj",
    ]
    # Gemma does not apply LoRA to the embedding layer
    # Gemma不在嵌入层应用LoRA
    embedding_modules = {}  # 嵌入模块映射 # embedding modules mapping
    embedding_padding_modules = []  # 嵌入填充模块 # embedding padding modules
    supports_lora = True  # 支持LoRA # supports LoRA

    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3nTextConfig,  # Gemma3n文本配置 # Gemma3n text config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ) -> None:
        super().__init__(config=config)  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config
        self.quant_config = quant_config  # 保存量化配置 # save quantization config
        self.model = Gemma3nTextModel(  # 创建Gemma3n文本模型 # create Gemma3n text model
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.logits_processor = LogitsProcessor(config)  # 创建logits处理器 # create logits processor

        if self.config.tie_word_embeddings:  # 绑定词嵌入权重 # tie word embedding weights
            self.lm_head = self.model.embed_tokens  # lm_head与词嵌入共享 # lm_head shares with embed_tokens
        else:  # 不绑定 # not tied
            self.lm_head = ParallelLMHead(  # 创建并行LM头 # create parallel LM head
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.post_init()  # 调用后初始化 # call post init

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层 # get input embedding layer
        return self.model.embed_tokens  # 返回词嵌入层 # return word embedding layer

    def get_attention_sliding_window_size(self):  # 获取注意力滑动窗口大小 # get attention sliding window size
        return get_attention_sliding_window_size(self.config)  # 委托给模块级函数 # delegate to module-level function

    def dtype(self) -> torch.dtype:  # 获取模型数据类型 # get model data type
        return next(self.parameters()).dtype  # 返回第一个参数的数据类型 # return dtype of first parameter

    @torch.no_grad()  # 禁用梯度计算 # disable gradient computation
    def forward(  # 前向传播方法 # forward pass method
        self,
        input_ids: torch.Tensor,  # 输入token ID张量 # input token ID tensor
        positions: torch.Tensor,  # 位置编码张量 # position encoding tensor
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选 # input embeddings, optional
        per_layer_inputs: Optional[torch.Tensor] = None,  # 每层输入，可选 # per-layer inputs, optional
        **kwargs,
    ) -> LogitsProcessor:
        hidden_states = self.model(  # 通过文本模型获取隐藏状态 # get hidden states through text model
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            per_layer_inputs,
            **kwargs,
        )

        return self.logits_processor(  # 通过logits处理器返回结果 # return result through logits processor
            input_ids, hidden_states, self.model.embed_tokens, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法 # load weights method
        stacked_params_mapping = [  # 堆叠参数映射 # stacked params mapping
            # (param_name, shard_name, shard_id)
            # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())  # 参数字典 # parameters dict
        loaded_params: Set[str] = set()  # 已加载参数集合 # loaded params set

        for name, loaded_weight in weights:  # 遍历权重 # iterate weights
            name = name.replace("model.language_model.", "model.")  # 替换名称前缀 # replace name prefix
            for param_name, shard_name, shard_id in stacked_params_mapping:  # 遍历堆叠映射 # iterate stacked mapping
                if shard_name not in name:  # 分片名不在权重名中 # shard name not in weight name
                    continue  # 跳过 # skip
                name = name.replace(shard_name, param_name)  # 替换分片名 # replace shard name
                # Skip loading extra bias for GPTQ models
                # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 额外偏置 # extra bias
                    continue  # 跳过 # skip
                if name not in params_dict:  # 参数不存在 # param not found
                    # Skip loading weights that are not in the model
                    # 跳过模型中不存在的权重
                    continue  # 跳过 # skip
                param = params_dict[name]  # 获取参数 # get parameter
                weight_loader = param.weight_loader  # 获取权重加载器 # get weight loader
                weight_loader(param, loaded_weight, shard_id)  # 加载权重 # load weight
                break  # 跳出内层循环 # break inner loop
            else:  # 非堆叠参数 # non-stacked params
                # lm_head is not used in vllm as it is tied with embed_token
                # lm_head在vllm中未使用，因为它与embed_token绑定
                if "lm_head.weight" in name:  # lm_head权重 # lm_head weight
                    continue  # 跳过 # skip
                # Skip loading extra bias for GPTQ models
                # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 额外偏置 # extra bias
                    continue  # 跳过 # skip
                # Remapping the name of FP8 kv-scale
                # 重映射FP8 kv-scale的名称
                name = maybe_remap_kv_scale_name(name, params_dict)  # 重映射名称 # remap name
                if name is None:  # 名称无效 # name is None
                    continue  # 跳过 # skip
                if name not in params_dict:  # 参数不存在 # param not found
                    # Skip loading weights that are not in the model
                    # 跳过模型中不存在的权重
                    continue  # 跳过 # skip

                param = params_dict[name]  # 获取参数 # get parameter
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器 # get weight loader
                weight_loader(param, loaded_weight)  # 加载权重 # load weight
            loaded_params.add(name)  # 添加到已加载集合 # add to loaded set
        return loaded_params  # 返回已加载参数集合 # return loaded params set


EntryClass = Gemma3nForCausalLM  # 模型入口类 # model entry class
AutoModel.register(Gemma3nTextConfig, Gemma3nForCausalLM, exist_ok=True)  # 注册模型到AutoModel # register model to AutoModel
