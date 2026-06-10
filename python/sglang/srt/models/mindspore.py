# MindSpore后端推理适配模块
# 该模块为SGLang提供基于MindSpore框架的推理支持，主要用于NPU设备
# 实现了MindSpore模型与SGLang推理框架的桥接，包括张量格式转换、注意力掩码生成、KV缓存管理等功能
# 核心类：MindSporeForCausalLM，将MindSpore模型包装为SGLang兼容的因果语言模型

from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
from typing import Any, Iterable, List, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch库

from sglang.srt.distributed import (  # 导入分布式通信函数
    get_tensor_model_parallel_rank,  # 获取张量并行排名
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # 导入logits处理器输出
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_executor.forward_context import (  # 导入前向上下文获取函数
    get_req_to_token_pool,  # 获取请求到token的映射池
    get_token_to_kv_pool,  # 获取token到KV的映射池
)
from sglang.srt.models.registry import import_model_classes  # 导入模型注册表导入函数
from sglang.srt.utils import is_npu  # 导入NPU检测函数

_is_npu = is_npu()  # 检测当前是否为NPU设备

if _is_npu:  # 如果是NPU设备
    import mindspore as ms  # 导入MindSpore框架
    import numpy as np  # 导入NumPy库
    import torch_npu  # 导入PyTorch NPU扩展
    from mindspore import Tensor, mint, mutable  # 导入MindSpore核心组件

logger = logging.getLogger(__name__)  # 创建日志记录器


def _get_arch_from_config(config):
    """根据配置获取MindSpore模型架构类"""
    mindspore_models = import_model_classes("sgl_mindspore.models")  # 导入MindSpore模型类
    architectures = getattr(config, "architectures", [])  # 获取架构列表
    if isinstance(architectures, str):  # 如果架构是字符串
        architectures = [architectures]  # 转换为列表
    if not architectures:  # 如果架构列表为空
        raise ValueError("No model architectures are specified")  # 抛出异常
    for arch in architectures:  # 遍历架构列表
        if arch in mindspore_models:  # 如果架构在MindSpore模型中
            return mindspore_models[arch]  # 返回对应的模型类
    raise ValueError(f"Unsupported arch {architectures}")  # 抛出不支持的架构异常


def tensor_torch2ms(x: torch.Tensor):
    """将PyTorch张量转换为MindSpore张量，通过DLPack中间格式"""
    if x is None or not isinstance(x, torch.Tensor):  # 如果输入为空或不是PyTorch张量
        return x  # 直接返回

    # torch tensor -> dlpack -> mindspore tensor
    pt_dlpack = torch.utils.dlpack.to_dlpack(x)  # 将PyTorch张量转换为DLPack格式
    ms_tensor = ms.utils.dlpack.from_dlpack(pt_dlpack)  # 从DLPack格式转换为MindSpore张量
    return ms_tensor  # 返回MindSpore张量


def tensor_ms2torch(x: "ms.Tensor"):
    """将MindSpore张量转换为PyTorch张量，通过DLPack中间格式"""
    if x is None or not isinstance(x, ms.Tensor):  # 如果输入为空或不是MindSpore张量
        return x  # 直接返回

    # ms tensor -> dlpack -> torch tensor
    ms_dlpack = ms.utils.dlpack.to_dlpack(x)  # 将MindSpore张量转换为DLPack格式
    torch_tensor = torch.utils.dlpack.from_dlpack(ms_dlpack)  # 从DLPack格式转换为PyTorch张量
    torch_npu.npu.synchronize()  # 同步NPU操作
    return torch_tensor  # 返回PyTorch张量


# Adapt from: https://gitee.com/mindspore/vllm-mindspore/blob/master/vllm_mindspore/model_executor/models/attention_mask.py
class LowerTriangularMask:
    """下三角注意力掩码生成器，用于推理时的注意力掩码"""
    r"""
    Provide Infer model attention mask.
    Args:
        dtype (ms dtype): The compute type of Infer model.
        max_model_len (int): The max model length of Infer model.
    """

    def __init__(self, dtype, max_model_len, decode_mask_coeff=-10000.0):
        """初始化下三角掩码生成器"""
        self.dtype = dtype  # 计算数据类型
        self.max_model_len = max_model_len  # 最大模型长度
        self.cached_mask_len = 8 * 1024  # 缓存掩码长度
        self.decode_mask_coeff = decode_mask_coeff  # 解码掩码系数

        prefill_mask_coeff = 1.0 if self.dtype == ms.bfloat16 else -10000.0  # 预填充掩码系数
        self.prefill_mask = Tensor(  # 预填充掩码
            np.triu(np.ones(shape=(128, 128), dtype=np.float16), k=1)  # 生成上三角矩阵
            * prefill_mask_coeff,  # 乘以系数
            dtype=self.dtype,  # 设置数据类型
        )

        self.hard_mask = mint.zeros((1, 1), dtype=dtype)  # 硬掩码（全零）
        self.decode_mask = (  # 解码掩码
            Tensor(
                np.triu(  # 生成上三角矩阵
                    np.ones(
                        shape=(self.cached_mask_len, self.cached_mask_len),  # 掩码形状
                        dtype=np.int8,  # 数据类型
                    ),
                    k=1,  # 主对角线以上
                ),
                dtype=self.dtype,  # MindSpore数据类型
            )
            * self.decode_mask_coeff  # 乘以解码系数
        )

    def create_mask(self, query_lens_np, seq_lens_np):
        """根据查询长度和序列长度创建注意力掩码"""
        """
        when query_lens_np = [3], seq_lens_np = [6], decode_mask_coeff = 1
        init attention mask
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 0 0 0 0 0
        """
        max_seq_len = seq_lens_np.max().item()  # 最大序列长度
        total_q_len = query_lens_np.sum().item()  # 总查询长度
        attention_mask = mint.zeros((total_q_len, max_seq_len), dtype=self.dtype)  # 初始化注意力掩码

        req_num = query_lens_np.shape[0]  # 请求数量
        current_row = 0  # 当前行索引
        for i in range(req_num):  # 遍历每个请求
            q_len = query_lens_np[i].item()  # 查询长度
            current_row += q_len  # 更新当前行
            # skip row when q_len <= 1, to decrease execute time
            if q_len <= 1:  # 如果查询长度小于等于1，跳过
                continue
            seq_len = seq_lens_np[i].item()  # 序列长度
            context_len = seq_len - q_len  # 上下文长度
            """
            set the right half to 1
            0 0 0 1 1 1
            0 0 0 1 1 1
            0 0 0 1 1 1
            """
            attention_mask[current_row - q_len : current_row, context_len:] = (  # 设置右半部分
                self.decode_mask_coeff  # 使用解码掩码系数
            )
            """
            set the lower triangle of the right half to 0
            0 0 0 0 1 1
            0 0 0 0 0 1
            0 0 0 0 0 0
            """
            right_tensor = attention_mask[  # 获取右半部分张量
                current_row - q_len : current_row, context_len:seq_len
            ]

            # use masked_fill_ to inplace modify attention_mask
            right_tensor.masked_fill_(right_tensor.tril() == self.decode_mask_coeff, 0)  # 填充下三角部分为0

        return attention_mask  # 返回注意力掩码

    def gen_attention_mask(
        self,
        is_prefill: bool,
        position_ids: "ms.Tensor",
        query_lens_np: np.ndarray,
        seq_lens_np: np.ndarray,
    ):
        """生成注意力掩码，根据是否为预填充阶段选择不同的掩码"""
        max_query_len = query_lens_np.max()  # 最大查询长度
        max_seq_len = seq_lens_np.max()  # 最大序列长度
        if is_prefill:  # 预填充阶段
            attention_mask = self.prefill_mask  # 使用预填充掩码
        elif max_query_len > 1:  # 解码阶段且查询长度大于1
            if max_seq_len <= self.cached_mask_len:  # 如果序列长度不超过缓存长度
                attention_mask = mint.index_select(self.decode_mask, 0, position_ids)  # 从缓存中索引掩码
            else:  # 序列长度超过缓存长度
                attention_mask = self.create_mask(query_lens_np, seq_lens_np)  # 动态创建掩码
        else:  # 单token解码
            attention_mask = self.hard_mask  # 使用硬掩码
        return attention_mask  # 返回注意力掩码


class MindSporeForCausalLM(torch.nn.Module):
    """基于MindSpore框架的因果语言模型，适配SGLang推理框架"""
    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        ms.set_context(graph_kernel_flags="--disable_pass=gather_pre_rms_norm_fusion")  # 禁用gather_pre_rms_norm_fusion优化
        ms.set_kernel_launch_capture(False)  # 禁用内核启动捕获

        logger.info(  # 打印张量并行信息
            "MindSporeForCausalLM tp size %d tp rank %d",
            get_tensor_model_parallel_world_size(),  # 张量并行世界大小
            get_tensor_model_parallel_rank(),  # 张量并行排名
        )
        if get_tensor_model_parallel_world_size() not in (1, 2, 4, 8):  # 检查TP大小是否支持
            # MatMulAllReduce only support tp size in (1, 2, 4, 8)
            ms.set_context(graph_kernel_flags="--disable_pass=MatMulAllReduce")  # 禁用MatMulAllReduce优化

        arch = self.get_arch(self.config)  # 获取模型架构
        self.model = arch(config=config, quant_config=quant_config)  # 创建模型实例

        self.causal_mask = LowerTriangularMask(  # 创建因果掩码生成器
            self.config.param_dtype, self.config.max_position_embeddings  # 使用参数数据类型和最大位置嵌入
        )
        self.key_cache = []  # 键缓存列表
        self.value_cache = []  # 值缓存列表

    @property
    def hot_token_id(self):
        """获取热门token ID属性"""
        if hasattr(self.model, "hot_token_id"):  # 如果模型有hot_token_id属性
            return tensor_ms2torch(self.model.hot_token_id)  # 转换为PyTorch张量返回
        return None  # 否则返回None

    def get_arch(self, config):
        """获取模型架构类"""
        return _get_arch_from_config(config)  # 调用架构获取函数

    @property
    def use_mla(self):
        """判断是否使用多头潜在注意力(MLA)"""
        return self.config.architectures[0] in ("DeepseekV3ForCausalLM")  # 检查架构是否为DeepseekV3

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重"""
        self.model.load_weights(weights)  # 加载权重到MindSpore模型
        for _, cell in self.model.cells_and_names():  # 遍历模型的所有单元
            quant_method = getattr(cell, "quant_method", None)  # 获取量化方法
            if quant_method is not None:  # 如果有量化方法
                quant_method.process_weights_after_loading(cell)  # 加载后处理权重

    def get_kvcache(self, forward_batch: ForwardBatch):
        """获取KV缓存，支持MLA和非MLA两种模式"""
        def prepare_cache(cache_list, is_key_cache):
            """准备缓存，将PyTorch张量转换为MindSpore张量"""
            for i in range(self.config.num_hidden_layers):  # 遍历每一层
                if is_key_cache:  # 如果是键缓存
                    cache = get_token_to_kv_pool().get_key_buffer(i)  # 获取键缓存
                else:  # 如果是值缓存
                    cache = get_token_to_kv_pool().get_value_buffer(i)  # 获取值缓存
                cache_ms = tensor_torch2ms(cache)  # 转换为MindSpore张量
                if self.use_mla and cache_ms.ndim == 3:  # MLA模式下需要增加维度
                    cache_ms = mint.unsqueeze(cache_ms, 2)  # 在第2维增加维度
                cache_list.append(cache_ms)  # 添加到缓存列表

        if self.use_mla:  # 如果使用MLA
            if not self.key_cache:  # 如果键缓存为空
                prepare_cache(self.key_cache, is_key_cache=True)  # 准备键缓存
            return mutable(self.key_cache)  # 返回可变的键缓存

        if self.key_cache and self.value_cache:  # 如果键缓存和值缓存都已准备
            return mutable(self.key_cache), mutable(self.value_cache)  # 返回两者

        prepare_cache(self.key_cache, is_key_cache=True)  # 准备键缓存
        prepare_cache(self.value_cache, is_key_cache=False)  # 准备值缓存

        return mutable(self.key_cache), mutable(self.value_cache)  # 返回键缓存和值缓存

    def _is_prefill(self, forward_batch: ForwardBatch):
        """判断当前是否为预填充阶段"""
        # Different processing for the mindspore attention operator
        # Without any prefix cache => Use FlashAttentionScore
        # With cache => Use PagedAttention, no matter the query length is 1 or not
        is_prefill = (  # 判断是否为预填充
            forward_batch.forward_mode.is_extend()  # 是否为扩展模式
            and not forward_batch.forward_mode.is_draft_extend_v2()  # 不是草稿扩展v2
            and not forward_batch.forward_mode.is_draft_extend()  # 不是草稿扩展
            and not forward_batch.forward_mode.is_target_verify()  # 不是目标验证
        )
        if forward_batch.extend_prefix_lens is not None:  # 如果有扩展前缀长度
            is_prefill = (  # 更新预填充状态
                is_prefill and forward_batch.extend_prefix_lens.sum().item() == 0  # 前缀长度之和为0
            )
        return is_prefill  # 返回预填充状态

    def prepare_inputs(self, input_ids, positions, forward_batch):
        """准备模型输入，包括KV缓存、注意力掩码、块表等"""
        if self.use_mla:  # 如果使用MLA
            key_cache = self.get_kvcache(forward_batch)  # 获取键缓存
        else:  # 不使用MLA
            key_cache, value_cache = self.get_kvcache(forward_batch)  # 获取键缓存和值缓存

        is_prefill = self._is_prefill(forward_batch)  # 判断是否为预填充
        batch_valid_length = forward_batch.seq_lens.cpu().numpy()  # 获取有效序列长度
        if forward_batch.forward_mode.is_target_verify():  # 如果是目标验证模式
            batch_valid_length += forward_batch.spec_info.num_tokens_per_req  # 加上每个请求的token数
        if forward_batch.extend_seq_lens is not None:  # 如果有扩展序列长度
            q_seq_lens = forward_batch.extend_seq_lens.cpu().numpy()  # 获取查询序列长度
        else:  # 没有扩展序列长度
            q_seq_lens = np.ones([forward_batch.batch_size], dtype=np.int32)  # 默认全1
            if forward_batch.forward_mode.is_target_verify():  # 如果是目标验证模式
                q_seq_lens = q_seq_lens * forward_batch.spec_info.num_tokens_per_req  # 乘以每个请求的token数

        page_size = get_token_to_kv_pool().page_size  # 获取页大小
        block_tables = tensor_torch2ms(  # 转换块表为MindSpore张量
            (
                get_req_to_token_pool().req_to_token[  # 获取请求到token映射
                    forward_batch.req_pool_indices, : batch_valid_length.max()  # 取有效长度
                ][:, ::page_size]  # 按页大小采样
                // page_size  # 除以页大小得到块编号
            )
        ).to(ms.int32)  # 转换为int32类型

        model_inputs = {}  # 模型输入字典
        model_inputs["input_ids"] = tensor_torch2ms(input_ids).to(ms.int32)  # 输入ID
        model_inputs["batch_valid_length"] = ms.Tensor(  # 批次有效长度
            batch_valid_length, dtype=ms.int32
        )
        model_inputs["position_ids"] = tensor_torch2ms(positions)  # 位置ID
        model_inputs["q_seq_lens"] = ms.Tensor(q_seq_lens, dtype=ms.int32)  # 查询序列长度
        model_inputs["attention_mask"] = self.causal_mask.gen_attention_mask(  # 生成注意力掩码
            is_prefill, model_inputs["position_ids"], q_seq_lens, batch_valid_length
        ).contiguous()  # 确保内存连续
        model_inputs["out_cache_loc"] = tensor_torch2ms(forward_batch.out_cache_loc).to(  # 输出缓存位置
            ms.int32
        )
        model_inputs["is_prefill"] = is_prefill  # 是否预填充
        model_inputs["key_cache"] = key_cache  # 键缓存
        if not self.use_mla:  # 如果不使用MLA
            model_inputs["value_cache"] = value_cache  # 添加值缓存
        model_inputs["block_tables"] = block_tables  # 块表
        # for speculative decode
        model_inputs["forward_mode"] = forward_batch.forward_mode  # 前向模式（用于投机解码）
        return model_inputs  # 返回模型输入字典

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> "ms.Tensor":
        """前向传播：准备输入、执行模型推理、返回logits结果"""
        # prepare base inputs
        model_inputs = self.prepare_inputs(input_ids, positions, forward_batch)  # 准备基础输入
        # prepare model inputs
        model_inputs = self.model.prepare_inputs(forward_batch, model_inputs)  # 模型特定输入准备

        # Used by speculative decoding (EAGLE)
        if self.model.capture_aux_hidden_states:  # 如果需要捕获辅助隐藏状态
            logits, hidden_states = self.model(**model_inputs)  # 获取logits和隐藏状态
        else:  # 不需要捕获
            logits = self.model(**model_inputs)  # 只获取logits
            hidden_states = None  # 隐藏状态为空

        logits_result = LogitsProcessorOutput(  # 创建logits处理器输出
            next_token_logits=tensor_ms2torch(logits),  # 将logits转换为PyTorch张量
            hidden_states=tensor_ms2torch(hidden_states),  # 将隐藏状态转换为PyTorch张量
        )
        return logits_result  # 返回logits结果

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        """获取专家位置模型配置"""
        try:  # 尝试获取配置
            arch_cls = _get_arch_from_config(config)  # 获取架构类
            method = getattr(arch_cls, "get_model_config_for_expert_location", None)  # 获取配置方法
            if method is None:  # 如果没有配置方法
                return None  # 返回None
            return method(config)  # 调用配置方法
        except Exception:  # 捕获异常
            return None  # 返回None

    # The following methods are used for speculative decoding
    def get_embed_and_head(self):
        """获取嵌入层和语言模型头（用于投机解码）"""
        embed, head = self.model.get_embed_and_head()  # 从MindSpore模型获取
        return tensor_ms2torch(embed), tensor_ms2torch(head)  # 转换为PyTorch张量返回

    def set_embed_and_head(self, embed, head):
        """设置嵌入层和语言模型头（用于投机解码）"""
        self.model.set_embed_and_head(tensor_torch2ms(embed), tensor_torch2ms(head))  # 转换为MindSpore张量设置

    def get_embed(self):
        """获取嵌入层权重（用于投机解码）"""
        return tensor_ms2torch(self.model.get_embed())  # 转换为PyTorch张量返回

    def set_embed(self, embed):
        """设置嵌入层权重（用于投机解码）"""
        self.model.set_embed(tensor_torch2ms(embed))  # 转换为MindSpore张量设置

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        """设置EAGLE3需要捕获的层（用于投机解码）"""
        self.model.set_eagle3_layers_to_capture(layer_ids)  # 设置捕获层


EntryClass = [MindSporeForCausalLM]  # 入口类列表
