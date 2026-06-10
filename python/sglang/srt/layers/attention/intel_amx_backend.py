# Intel AMX注意力机制后端实现
# 本文件实现了基于Intel AMX（Advanced Matrix Extensions）的注意力计算后端，
# 用于在Intel CPU上进行注意力推理，支持prefill和decode两种模式。

from __future__ import annotations  # 启用延迟类型注解评估

from typing import TYPE_CHECKING  # 导入类型检查常量

import torch  # 导入PyTorch库

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend  # 导入注意力后端基类
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息类

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力类
    from sglang.srt.model_executor.model_runner import ModelRunner  # 导入模型运行器类


class IntelAMXAttnBackend(AttentionBackend):  # Intel AMX注意力后端类，继承自AttentionBackend
    def __init__(self, model_runner: ModelRunner):  # 初始化方法
        import sgl_kernel  # noqa: F401  # 导入sgl_kernel以确保AMX内核可用

        super().__init__()  # 调用父类初始化
        self.forward_metadata = None  # 前向传播元数据，初始化为None
        self.device = model_runner.device  # 设备信息
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        # 池引用 — 在构造时捕获，确保在ForwardBatch字段被删除后仍然存活
        self.req_to_token_pool = model_runner.req_to_token_pool  # 请求到token的映射池
        self.token_to_kv_pool = model_runner.token_to_kv_pool  # token到KV缓存的映射池

        self.num_head = (  # 注意力头数
            model_runner.model_config.num_attention_heads // model_runner.tp_size  # 根据张量并行度计算每GPU头数
        )

        # [NB]: `layer_id` set to 0 for qwen3-next models, as not all attn layers require kv pool
        # using "full_attention_layer_id_mapping" to map which layer needs kv pool
        # [注]：对于qwen3-next模型，`layer_id`设为0，因为并非所有注意力层都需要KV池
        # 使用"full_attention_layer_id_mapping"来映射哪些层需要KV池
        layer_id = 0  # 默认层ID为0
        if hasattr(model_runner.token_to_kv_pool, "full_attention_layer_id_mapping"):  # 如果KV池有完整注意力层ID映射
            layer_id = [*model_runner.token_to_kv_pool.full_attention_layer_id_mapping][  # 获取第一个完整注意力层ID
                0
            ]
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(  # V头维度
            layer_id  # 使用确定的层ID
        ).shape[-1]  # 取最后一个维度
        self.decode_attention_fwd = torch.ops.sgl_kernel.decode_attention_cpu  # 解码注意力前向函数
        self.extend_attention_fwd = torch.ops.sgl_kernel.extend_attention_cpu  # 扩展注意力前向函数

    def init_forward_metadata(self, forward_batch: ForwardBatch):  # 初始化前向传播元数据
        """Init the metadata for a forward pass."""  # 初始化前向传播的元数据

        bs = forward_batch.batch_size  # 批次大小
        attn_logits = torch.zeros(  # 注意力logits张量
            (
                bs,  # 批次维度
                self.num_head,  # 注意力头维度
                8,  # self.num_kv_splits,  # KV分割数
                self.v_head_dim + 1,  # V头维度加1
            ),
            dtype=torch.float32,  # 数据类型为float32
            device=self.device,  # 设备
        )
        if forward_batch.forward_mode.is_decode_or_idle():  # 如果是解码或空闲模式
            max_extend_len = None  # 最大扩展长度为None
        else:  # 否则
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()  # 获取最大扩展序列长度
        self.forward_metadata = (attn_logits, max_extend_len)  # 保存元数据

    def get_cpu_graph_seq_len_fill_value(self):  # 获取CPU图序列长度填充值
        return 1  # 返回1

    def init_forward_metadata_capture_cpu_graph(  # 初始化CPU图捕获时的前向元数据
        self,
        bs: int,  # 批次大小
        num_tokens: int,  # token数量
        req_pool_indices: torch.Tensor,  # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度
        encoder_lens,  # 编码器长度
        forward_mode,  # 前向模式
        spec_info,  # 推测信息
    ):
        attn_logits = torch.zeros(  # 注意力logits张量
            (
                bs,  # 批次维度
                self.num_head,  # 注意力头维度
                8,  # self.num_kv_splits,  # KV分割数
                self.v_head_dim + 1,  # V头维度加1
            ),
            dtype=torch.float32,  # 数据类型为float32
            device=self.device,  # 设备
        )
        max_extend_len = None  # 最大扩展长度初始化为None
        self.forward_metadata = (attn_logits, max_extend_len)  # 保存元数据

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):  # 初始化CPU图状态
        pass  # 空操作，无需额外初始化

    def forward_extend(  # 前向扩展（prefill）计算
        self,
        q,  # 查询张量
        k,  # 键张量
        v,  # 值张量
        layer: RadixAttention,  # 注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存
        sinks=None,  # 注意力汇聚点
    ):
        if layer.qk_head_dim != layer.v_head_dim:  # 如果QK头维度与V头维度不同
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))  # 创建不同维度的输出张量
        else:  # 否则
            o = torch.empty_like(q)  # 创建与q相同形状的输出张量
        cache_loc = (  # 缓存位置
            forward_batch.out_cache_loc  # 非交叉注意力的缓存位置
            if not layer.is_cross_attention  # 如果不是交叉注意力
            else forward_batch.encoder_out_cache_loc  # 否则使用编码器缓存位置
        )
        if save_kv_cache and k is not None and v is not None:  # 如果需要保存KV缓存且k、v不为None
            self.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)  # 设置KV缓存

        _, max_extend_len = self.forward_metadata  # 获取最大扩展长度
        self.extend_attention_fwd(  # 调用扩展注意力前向函数
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),  # 重塑查询张量
            k,  # 键张量
            v,  # 值张量
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),  # 重塑输出张量
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),  # 获取键缓存
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),  # 获取值缓存
            self.req_to_token_pool.req_to_token,  # 请求到token映射
            forward_batch.req_pool_indices,  # 请求池索引
            forward_batch.seq_lens,  # 序列长度
            forward_batch.extend_seq_lens,  # 扩展序列长度
            forward_batch.extend_start_loc,  # 扩展起始位置
            max_extend_len,  # 最大扩展长度
            layer.scaling,  # 缩放因子
            layer.logit_cap,  # logits上限
            layer.is_cross_attention,  # 是否交叉注意力
            layer.sliding_window_size + 1,  # 滑动窗口大小加1
            forward_batch.encoder_lens,  # 编码器长度
            sinks,  # 注意力汇聚点
        )
        return o  # 返回输出

    def forward_decode(  # 前向解码计算
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存
        sinks=None,  # 注意力汇聚点
    ):
        attn_logits, _ = self.forward_metadata  # 获取注意力logits

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)  # 重塑查询张量

        if layer.qk_head_dim != layer.v_head_dim:  # 如果QK头维度与V头维度不同
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))  # 创建不同维度的输出张量
        else:  # 否则
            o = torch.empty_like(q)  # 创建与q相同形状的输出张量
        cache_loc = (  # 缓存位置
            forward_batch.out_cache_loc  # 非交叉注意力的缓存位置
            if not layer.is_cross_attention  # 如果不是交叉注意力
            else forward_batch.encoder_out_cache_loc  # 否则使用编码器缓存位置
        )
        self.decode_attention_fwd(  # 调用解码注意力前向函数
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),  # 重塑查询张量
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),  # 获取键缓存
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),  # 获取值缓存
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),  # 重塑输出张量
            k,  # 键张量
            v,  # 值张量
            cache_loc,  # 缓存位置
            attn_logits,  # 注意力logits
            self.req_to_token_pool.req_to_token,  # 请求到token映射
            forward_batch.req_pool_indices,  # 请求池索引
            forward_batch.seq_lens,  # 序列长度
            layer.scaling,  # 缩放因子
            layer.logit_cap,  # logits上限
            layer.is_cross_attention,  # 是否交叉注意力
            layer.sliding_window_size + 1,  # 滑动窗口大小加1
            forward_batch.encoder_lens,  # 编码器长度
            sinks,  # 注意力汇聚点
        )
        return o  # 返回输出

    def support_triton(self):  # 是否支持Triton
        return False  # 不支持Triton
