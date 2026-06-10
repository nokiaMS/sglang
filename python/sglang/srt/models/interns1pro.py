# InternS1Pro 多模态条件生成模型
# 基于 Qwen3VLMoe 架构，支持 FOPE（Fractional Order Position Encoding）位置编码
# 实现了分组路由（Group Router）的混合专家模型（MoE）机制
import functools  # 导入 functools 模块，用于 LRU 缓存装饰器
import logging  # 导入 logging 模块，用于日志记录
from typing import Any, Dict, Iterable, Optional, Tuple  # 导入类型提示工具

import torch  # 导入 PyTorch 深度学习框架
from transformers import PretrainedConfig  # 从 transformers 导入预训练配置类

from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size  # 导入注意力张量并行相关工具
from sglang.srt.layers.moe.topk import TopK  # 导入 TopK 路由选择模块
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息类
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.qwen3_moe import Qwen3MoeAttention, Qwen3MoeDecoderLayer  # 导入 Qwen3 MoE 注意力层和解码层
from sglang.srt.models.qwen3_vl_moe import (  # 导入 Qwen3 VL MoE 模型类
    Qwen3MoeLLMModel,
    Qwen3VLMoeForConditionalGeneration,
)
from sglang.srt.utils import add_prefix  # 导入前缀添加工具函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class InternS1ProTextAttention(Qwen3MoeAttention):  # InternS1Pro 文本注意力层，继承自 Qwen3MoE 注意力层
    def __init__(  # 初始化方法
        self,
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV 头数（用于 GQA）
        layer_id: int = 0,  # 层 ID，默认为 0
        rope_theta: float = 1000000,  # RoPE 基础频率参数，默认 1000000
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE 缩放配置，可选
        max_position_embeddings: int = 32768,  # 最大位置嵌入数，默认 32768
        **kwargs,  # 其他关键字参数
    ) -> None:
        super().__init__(  # 调用父类初始化
            hidden_size,
            num_heads,
            num_kv_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            **kwargs,
        )
        # for fope  # 用于 FOPE（分数阶位置编码）
        fope_keys = {"fope_init_factor", "fope_sep_head", "num_inv_freq"}  # FOPE 相关配置键集合
        use_fope = any(rope_scaling.get(key) is not None for key in fope_keys)  # 检查是否启用了 FOPE
        if use_fope:  # 如果启用 FOPE
            rope_scaling["use_fope"] = True  # 在配置中标记启用 FOPE
            rope_scaling["num_kv_heads"] = self.num_kv_heads  # 设置 KV 头数到配置中

        self.rotary_emb = get_rope(  # 创建旋转位置编码实例
            self.head_dim,  # 每个头的维度
            rotary_dim=self.head_dim,  # 旋转维度等于头维度
            max_position=max_position_embeddings,  # 最大位置数
            base=rope_theta,  # 基础频率
            rope_scaling=rope_scaling,  # 缩放配置
        )
        self.compatible_with_fused_kv_buffer = False  # 不兼容融合 KV 缓冲区
        self.use_fused_qk_norm_rope = False  # 不使用融合 QK 归一化 RoPE
        self._used_fused_qk_norm_rope_last_call = False  # 上次调用是否使用了融合 QK 归一化 RoPE 标记

    def forward_prepare_npu(  # NPU 前向准备方法（尚未实现）
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ):
        raise NotImplementedError()  # 抛出未实现异常


class InternS1ProTextDecoderLayer(Qwen3MoeDecoderLayer):  # InternS1Pro 文本解码层，继承自 Qwen3MoE 解码层
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int,  # 层 ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用 CUDA 流，可选
    ) -> None:
        super().__init__(  # 调用父类初始化
            config,
            layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
        )

        rope_theta = getattr(config, "rope_theta", 1000000)  # 获取 RoPE theta 参数，默认 1000000
        rope_scaling = getattr(config, "rope_scaling", None)  # 获取 RoPE 缩放配置，默认 None
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)  # 获取最大位置嵌入数，默认 32768
        head_dim = getattr(  # 获取头维度
            config, "head_dim", config.hidden_size // config.num_attention_heads  # 默认为隐藏大小除以注意力头数
        )
        rms_norm_eps = config.rms_norm_eps  # 获取 RMS 归一化 epsilon
        attention_bias = config.attention_bias  # 获取注意力偏置配置

        self.self_attn = InternS1ProTextAttention(  # 创建 InternS1Pro 文本注意力实例
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        # update with group router  # 更新分组路由配置
        self.router_n_groups = getattr(config, "router_n_groups", -1)  # 获取路由分组数，默认 -1
        if self.router_n_groups > 0:  # 如果启用了分组路由
            assert (  # 断言每个 token 的专家数能被分组数整除
                config.num_experts_per_tok % self.router_n_groups == 0
            ), f"{config.num_experts_per_tok} cannot be divided by {self.router_n_groups}"
            self.mlp.topk = TopK(  # 创建新的 TopK 路由实例
                top_k=config.num_experts_per_tok,  # 每个 token 选择的专家数
                renormalize=config.norm_topk_prob,  # 是否重新归一化概率
                use_grouped_topk=False,  # 不使用分组 TopK（由自定义路由函数处理）
                layer_id=layer_id,
                custom_routing_function=self._custom_routing_function,  # 使用自定义路由函数
            )

    @staticmethod  # 静态方法
    @functools.lru_cache  # 使用 LRU 缓存装饰器
    def get_group_offsets(router_n_groups: int, group_size: int, device: str):  # 获取分组偏移量
        group_offsets = (  # 计算每个分组的偏移量
            torch.arange(router_n_groups, device=device) * group_size  # 每组偏移 = 组索引 * 组大小
        ).view(
            1, -1, 1
        )  # [1, n_groups, 1]  # 调整形状为 [1, 分组数, 1]
        return group_offsets  # 返回分组偏移量

    def _custom_routing_function(  # 自定义路由函数，实现分组路由逻辑
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        gating_output: torch.Tensor,  # 门控输出张量
        topk: int,  # TopK 值
        renormalize: bool,  # 是否重新归一化
    ) -> torch.Tensor:
        """Group router"""  # 分组路由器
        routing_weights = torch.softmax(gating_output, dim=-1, dtype=torch.float32)  # 对门控输出进行 softmax 得到路由权重
        if self.router_n_groups > 0:  # 如果启用了分组路由
            assert (  # 断言权重最后一维能被分组数整除
                routing_weights.shape[-1] % self.router_n_groups == 0
            ), f"{routing_weights.shape[-1]} cannot be divided by {self.router_n_groups}"
            per_group_top_k = topk // self.router_n_groups  # 每组选择的专家数
            group_size = routing_weights.shape[-1] // self.router_n_groups  # 每组的专家数
            group_offsets = self.get_group_offsets(  # 获取分组偏移量
                self.router_n_groups, group_size, routing_weights.device
            )
            routing_weights = routing_weights.unflatten(  # 将权重重塑为 (分组数, 组大小) 形状
                -1, (self.router_n_groups, group_size)
            )
            topk_weights, topk_ids = torch.topk(  # 在每个分组内进行 TopK 选择
                routing_weights, per_group_top_k, dim=-1
            )
            topk_ids = (topk_ids + group_offsets).flatten(-2, -1)  # 加上分组偏移量并展平，得到全局专家 ID
            topk_weights = topk_weights.flatten(-2, -1)  # 展平权重
        else:  # 如果未启用分组路由
            topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)  # 直接进行 TopK 选择

        if renormalize:  # 如果需要重新归一化
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # 归一化权重使其和为 1

        return topk_weights, topk_ids  # 返回 TopK 权重和 ID


class InternS1ProTextModel(Qwen3MoeLLMModel):  # InternS1Pro 文本模型，继承自 Qwen3MoE LLM 模型
    def __init__(  # 初始化方法
        self,
        *,
        config: PretrainedConfig,  # 预训练配置（仅限关键字参数）
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        decoder_layer_type=InternS1ProTextDecoderLayer,  # 解码层类型，默认为 InternS1Pro 解码层
        prefix: str = "",  # 参数前缀，默认为空
    ):
        super().__init__(  # 调用父类初始化
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=decoder_layer_type,
        )


class InternS1ProForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):  # InternS1Pro 条件生成模型，继承自 Qwen3VL MoE 条件生成模型

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空
        language_model_cls=InternS1ProTextModel,  # 语言模型类，默认为 InternS1Pro 文本模型
    ) -> None:
        # deal with no deepstack  # 处理没有 deepstack 的情况
        if not hasattr(config.vision_config, "deepstack_visual_indexes"):  # 如果视觉配置没有 deepstack 视觉索引属性
            config.vision_config.deepstack_visual_indexes = []  # 设置为空列表

        super().__init__(  # 调用父类初始化
            config,
            quant_config=quant_config,
            prefix=prefix,
            language_model_cls=language_model_cls,
        )

        # disable deepstack  # 禁用 deepstack
        if len(config.vision_config.deepstack_visual_indexes) == 0:  # 如果 deepstack 视觉索引为空
            self.use_deepstack = {}  # 将 deepstack 使用标记设为空字典（禁用）

    def _load_fope_weights(self, name: str, loaded_weight: torch.Tensor, params_dict):  # 加载 FOPE 权重
        """load fope weights"""  # 加载 FOPE（分数阶位置编码）权重
        attn_tp_size = get_attention_tp_size()  # 获取注意力张量并行大小
        attn_tp_rank = get_attention_tp_rank()  # 获取注意力张量并行排名

        num_key_value_heads = loaded_weight.size(0)  # 获取权重中 KV 头的数量
        # replicate head if necessary  # 如果需要则复制头
        if num_key_value_heads < attn_tp_size:  # 如果 KV 头数小于张量并行大小
            n_replicate = attn_tp_size // num_key_value_heads  # 计算需要复制的次数
            attn_tp_size = num_key_value_heads  # 更新并行大小为 KV 头数
            attn_tp_rank = attn_tp_rank // n_replicate  # 更新并行排名
        loaded_weight = loaded_weight.chunk(attn_tp_size, dim=0)[attn_tp_rank]  # 按张量并行分割权重并取当前排名对应的分片

        # rotary_emb is shared cross layers  # 旋转位置编码跨层共享
        param_name = name.replace(".rotary_emb.", ".layers.0.self_attn.rotary_emb.")  # 将权重名称映射到第 0 层的旋转嵌入参数
        assert param_name in params_dict  # 断言参数名称存在于参数字典中
        param = params_dict[param_name]  # 获取参数
        weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
        weight_loader(param, loaded_weight)  # 加载权重

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        """load weights"""  # 加载权重
        # Cache params_dict to avoid repeated expensive traversal of model parameters  # 缓存参数字典以避免重复的昂贵模型参数遍历
        if not hasattr(self, "_cached_params_dict"):  # 如果还没有缓存参数字典
            self._cached_params_dict = dict(self.named_parameters())  # 创建并缓存参数字典
        params_dict = self._cached_params_dict  # 使用缓存的参数字典
        other_weights = dict()  # 存储非 FOPE 权重的字典
        for name, loaded_weight in weights:  # 遍历所有权重
            if "sin_coef" in name or "cos_coef" in name:  # 如果是 FOPE 的正弦/余弦系数权重
                name = name.replace(r"model.language_model.", r"model.")  # 替换权重名称前缀
                self._load_fope_weights(name, loaded_weight, params_dict)  # 调用 FOPE 权重加载方法
            else:  # 其他权重
                other_weights[name] = loaded_weight  # 存入其他权重字典

        super().load_weights(other_weights.items())  # 调用父类加载剩余权重


EntryClass = InternS1ProForConditionalGeneration  # 入口类，用于模型注册
