# 文件说明：PyTorch FlexAttention注意力后端实现
# 该模块使用PyTorch的flex_attention算子实现注意力机制，
# 支持扩展（prefill）和解码（decode）阶段，利用块掩码优化注意力计算。

from __future__ import annotations  # 启用延迟注解评估 # 启用延迟类型注解

from typing import TYPE_CHECKING  # 导入类型检查 # 导入类型检查

import torch  # 导入PyTorch # 导入PyTorch框架
from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # 导入FlexAttention相关函数 # 导入FlexAttention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend  # 导入注意力后端基类 # 导入注意力后端基类
from sglang.srt.layers.radix_attention import AttentionType  # 导入注意力类型枚举 # 导入注意力类型枚举
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # 导入前向批次信息

if TYPE_CHECKING:  # 如果是类型检查阶段 # 类型检查时才导入
    from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力类 # 导入基数注意力类
    from sglang.srt.model_executor.model_runner import ModelRunner  # 导入模型运行器 # 导入模型运行器类


class TorchFlexAttnBackend(AttentionBackend):  # PyTorch FlexAttention后端类 # FlexAttention注意力后端
    def __init__(self, model_runner: ModelRunner):  # 初始化方法 # 初始化方法
        super().__init__()  # 调用父类初始化 # 调用基类初始化
        self.forward_metadata = None  # 前向元数据初始化为空 # 前向元数据
        self.device = model_runner.device  # 保存设备信息 # 保存设备
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        # 池引用——在构造时捕获，以便在删除对应的ForwardBatch字段后仍能存活。
        self.req_to_token_pool = model_runner.req_to_token_pool  # 保存请求到token池 # 保存请求-token映射池
        self.token_to_kv_pool = model_runner.token_to_kv_pool  # 保存token到KV池 # 保存KV缓存池
        self.flex_attention = torch.compile(flex_attention, dynamic=True)  # 编译FlexAttention，启用动态形状 # 编译FlexAttention
        torch._dynamo.config.cache_size_limit = 1024  # 设置dynamo缓存大小上限 # 设置dynamo缓存上限
        torch._dynamo.config.accumulated_cache_size_limit = 1024  # 设置dynamo累积缓存大小上限 # 设置dynamo累积缓存上限

    def init_forward_metadata(self, forward_batch: ForwardBatch):  # 初始化前向元数据 # 初始化前向元数据
        """Init the metadata for a forward pass."""
        # 初始化前向传播的元数据。
        # TODO: find a more elegant way to save memory
        # 待办：寻找更优雅的节省内存的方式
        # Currently maintain the same memory as torch_native_backend
        # 目前保持与torch_native_backend相同的内存策略
        torch.cuda.empty_cache()  # 清空CUDA缓存以释放内存 # 清空CUDA缓存

        # Provide two block_mask Lists per seq_idx for lower latency, later will support per layer level mask generation
        # 为每个序列索引提供两个block_mask列表以降低延迟，后续将支持逐层级别掩码生成
        self.extend_block_masks = []  # 扩展阶段的块掩码列表 # 扩展块掩码列表
        self.decode_block_masks = []  # 解码阶段的块掩码列表 # 解码块掩码列表

        if forward_batch.forward_mode.is_extend():  # 如果是扩展模式 # 检查扩展模式
            for seq_idx in range(forward_batch.seq_lens.shape[0]):  # 遍历每个序列 # 遍历序列
                seq_len_kv = forward_batch.seq_lens[seq_idx]  # 获取KV序列长度 # KV序列长度
                seq_len_q = seq_len_kv  # 扩展阶段查询长度等于KV长度 # 查询长度等于KV长度
                self.extend_block_masks.append(  # 添加扩展块掩码 # 添加扩展块掩码
                    create_block_mask(  # 创建块掩码 # 创建块掩码
                        self._causal_mask,  # 因果掩码函数 # 因果掩码
                        None,  # 批次维度（None表示不限制） # 批次维度
                        None,  # 头维度（None表示不限制） # 头维度
                        seq_len_q,  # 查询长度 # 查询长度
                        seq_len_kv,  # KV长度 # KV长度
                        device=self.device,  # 设备 # 设备
                        _compile=False,  # 不预编译 # 不预编译
                    )
                )

        elif forward_batch.forward_mode.is_decode():  # 如果是解码模式 # 检查解码模式
            for seq_idx in range(forward_batch.seq_lens.shape[0]):  # 遍历每个序列 # 遍历序列
                seq_len_q = 1  # 解码阶段查询长度为1 # 查询长度为1
                seq_len_kv = forward_batch.seq_lens[seq_idx]  # 获取KV序列长度 # KV序列长度

                self.decode_block_masks.append(  # 添加解码块掩码 # 添加解码块掩码
                    create_block_mask(  # 创建块掩码 # 创建块掩码
                        self._decode_mask,  # 解码掩码函数 # 解码掩码
                        None,  # 批次维度 # 批次维度
                        None,  # 头维度 # 头维度
                        seq_len_q,  # 查询长度 # 查询长度
                        seq_len_kv,  # KV长度 # KV长度
                        device=self.device,  # 设备 # 设备
                        _compile=False,  # 不预编译 # 不预编译
                    )
                )

    @staticmethod  # 静态方法装饰器 # 静态方法
    def _causal_mask(b, h, q_idx, kv_idx):  # 因果注意力掩码函数 # 因果掩码
        return q_idx >= kv_idx  # 查询位置大于等于键位置时为True # 因果条件

    @staticmethod  # 静态方法装饰器 # 静态方法
    def _decode_mask(b, h, q_idx, kv_idx):  # 解码注意力掩码函数 # 解码掩码
        return q_idx <= kv_idx  # 查询位置小于等于键位置时为True（解码时关注所有之前的token） # 解码条件

    def _run_flex_forward_extend(  # 使用FlexAttention运行扩展阶段前向传播 # 使用FlexAttention运行扩展前向
        self,
        query: torch.Tensor,  # 查询张量 # 查询张量
        output: torch.Tensor,  # 输出张量 # 输出张量
        k_cache: torch.Tensor,  # 键缓存 # 键缓存
        v_cache: torch.Tensor,  # 值缓存 # 值缓存
        req_to_token: torch.Tensor,  # 请求到token映射 # 请求-token映射
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度
        extend_prefix_lens: torch.Tensor,  # 扩展前缀长度 # 扩展前缀长度
        extend_seq_lens: torch.Tensor,  # 扩展序列长度 # 扩展序列长度
        scaling=None,  # 缩放因子 # 缩放因子
        enable_gqa=False,  # 是否启用GQA # 是否启用GQA
        causal=False,  # 是否因果注意力 # 是否因果
    ):
        """Run the extend forward by using torch flex attention op.
        # 使用PyTorch FlexAttention算子运行扩展阶段前向传播。

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        # 参数：
        #   query: [token数, 头数, 头维度]
        #   output: [token数, 头数, 头维度]
        #   k_cache: [最大token数, 头数, 头维度]
        #   v_cache: [最大token数, 头数, 头维度]
        #   req_to_token: [最大请求数, 最大上下文长度]
        #   req_pool_indices: [序列数]
        #   seq_lens: [序列数]
        #   extend_prefix_lens: [序列数]
        #   extend_seq_lens: [序列数]
        #   scaling: float 或 None
        #   enable_gqa: bool
        #   causal: bool
        # 返回：
        #   output: [token数, 头数, 头维度]

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]  # 断言序列长度和前缀长度维度一致 # 断言维度一致
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]  # 断言序列长度和扩展长度维度一致 # 断言维度一致

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        # [token数, 头数, 头维度] -> [头数, token数, 头维度]
        query = query.movedim(0, query.dim() - 2)  # 将token维度移到中间 # 转置维度

        start_q, start_kv = 0, 0  # 初始化查询和KV的起始位置 # 初始化起始位置

        for seq_idx in range(seq_lens.shape[0]):  # 遍历每个序列 # 遍历序列
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.
            # 待办：此循环每次迭代处理一个序列，效率低下。需要后续优化性能。
            extend_seq_len_q = extend_seq_lens[seq_idx]  # 当前序列的扩展长度 # 扩展长度
            prefill_seq_len_q = extend_prefix_lens[seq_idx]  # 当前序列的前缀长度 # 前缀长度

            seq_len_kv = seq_lens[seq_idx]  # 当前序列的KV长度 # KV序列长度
            end_q = start_q + extend_seq_len_q  # 查询结束位置 # 查询结束位置
            end_kv = start_kv + seq_len_kv  # KV结束位置 # KV结束位置

            per_req_query = query[:, start_q:end_q, :]  # 获取当前请求的查询 # 当前请求查询
            per_req_query_redundant = torch.empty(  # 创建冗余查询张量（包含前缀和扩展部分） # 创建冗余查询
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),  # 形状 # 张量形状
                dtype=per_req_query.dtype,  # 数据类型 # 数据类型
                device=per_req_query.device,  # 设备 # 设备
            )

            per_req_query_redundant[:, prefill_seq_len_q:, :] = per_req_query  # 将扩展部分填入冗余查询 # 填入扩展部分

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            # 从缓存中获取键和值。per_req_tokens包含序列中每个token的KV缓存索引。
            req_pool_idx = req_pool_indices[seq_idx]  # 当前请求的池索引 # 请求池索引
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]  # 当前请求的token索引 # token索引
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)  # 获取键缓存并转置 # 键缓存
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)  # 获取值缓存并转置 # 值缓存

            if not causal:  # 如果不是因果注意力 # 检查因果标志
                raise NotImplementedError("Non-causal mode is not yet implemented.")  # 抛出未实现错误 # 非因果模式未实现

            per_req_out_redundant = (  # 执行FlexAttention计算 # 执行FlexAttention
                self.flex_attention(  # 调用编译后的FlexAttention # 调用FlexAttention
                    per_req_query_redundant.unsqueeze(0),  # 增加批次维度 # 增加批次维度
                    per_req_key.unsqueeze(0),  # 增加批次维度 # 增加批次维度
                    per_req_value.unsqueeze(0),  # 增加批次维度 # 增加批次维度
                    block_mask=self.extend_block_masks[seq_idx],  # 扩展块掩码 # 扩展块掩码
                    scale=scaling,  # 缩放因子 # 缩放因子
                    enable_gqa=enable_gqa,  # 是否启用GQA # GQA标志
                )
                .squeeze(0)  # 移除批次维度 # 移除批次维度
                .movedim(query.dim() - 2, 0)  # 恢复原始维度顺序 # 恢复维度顺序
            )
            output[start_q:end_q, :, :] = per_req_out_redundant[  # 提取扩展部分的输出 # 提取扩展输出
                prefill_seq_len_q:, :, :  # 只取扩展部分 # 扩展部分切片
            ]
            start_q, start_kv = end_q, end_kv  # 更新起始位置 # 更新位置
        return output  # 返回输出 # 返回输出

    def _run_flex_forward_decode(  # 使用FlexAttention运行解码阶段前向传播 # 使用FlexAttention运行解码前向
        self,
        query: torch.Tensor,  # 查询张量 # 查询张量
        output: torch.Tensor,  # 输出张量 # 输出张量
        k_cache: torch.Tensor,  # 键缓存 # 键缓存
        v_cache: torch.Tensor,  # 值缓存 # 值缓存
        req_to_token: torch.Tensor,  # 请求到token映射 # 请求-token映射
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求池索引
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度
        scaling=None,  # 缩放因子 # 缩放因子
        enable_gqa=False,  # 是否启用GQA # 是否启用GQA
        causal=False,  # 是否因果注意力 # 是否因果
    ):
        """Run the decode forward by using torch flex attention op.
        # 使用PyTorch FlexAttention算子运行解码阶段前向传播。

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        # 参数：
        #   query: [token数, 头数, 头维度]
        #   output: [token数, 头数, 头维度]
        #   k_cache: [最大token数, 头数, 头维度]
        #   v_cache: [最大token数, 头数, 头维度]
        #   req_to_token: [最大请求数, 最大上下文长度]
        #   req_pool_indices: [序列数]
        #   seq_lens: [序列数]
        #   scaling: float 或 None
        #   enable_gqa: bool
        #   causal: bool
        # 返回：
        #   output: [token数, 头数, 头维度]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        # [token数, 头数, 头维度] -> [头数, token数, 头维度]
        query = query.movedim(0, query.dim() - 2)  # 将token维度移到中间 # 转置维度

        start_q, start_kv = 0, 0  # 初始化查询和KV的起始位置 # 初始化起始位置
        for seq_idx in range(seq_lens.shape[0]):  # 遍历每个序列 # 遍历序列
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.
            # 待办：此循环每次迭代处理一个序列，效率低下。需要后续优化性能。

            seq_len_q = 1  # 解码阶段每个序列的查询长度为1 # 解码查询长度
            seq_len_kv = seq_lens[seq_idx]  # 当前序列的KV长度 # KV序列长度
            end_q = start_q + seq_len_q  # 查询结束位置 # 查询结束位置
            end_kv = start_kv + seq_len_kv  # KV结束位置 # KV结束位置

            per_req_query = query[:, start_q:end_q, :]  # 获取当前请求的查询 # 当前请求查询

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            # 从缓存中获取键和值。per_req_tokens包含序列中每个token的KV缓存索引。
            req_pool_idx = req_pool_indices[seq_idx]  # 当前请求的池索引 # 请求池索引
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]  # 当前请求的token索引 # token索引
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)  # 获取键缓存并转置 # 键缓存
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)  # 获取值缓存并转置 # 值缓存

            per_req_out = (  # 执行FlexAttention计算 # 执行FlexAttention
                self.flex_attention(  # 调用编译后的FlexAttention # 调用FlexAttention
                    per_req_query.unsqueeze(0),  # 增加批次维度 # 增加批次维度
                    per_req_key.unsqueeze(0),  # 增加批次维度 # 增加批次维度
                    per_req_value.unsqueeze(0),  # 增加批次维度 # 增加批次维度
                    block_mask=self.decode_block_masks[seq_idx],  # 解码块掩码 # 解码块掩码
                    scale=scaling,  # 缩放因子 # 缩放因子
                    enable_gqa=enable_gqa,  # 是否启用GQA # GQA标志
                )
                .squeeze(0)  # 移除批次维度 # 移除批次维度
                .movedim(query.dim() - 2, 0)  # 恢复原始维度顺序 # 恢复维度顺序
            )

            output[start_q:end_q, :, :] = per_req_out  # 将输出写入结果张量 # 写入输出
            start_q, start_kv = end_q, end_kv  # 更新起始位置 # 更新位置

        return output  # 返回输出 # 返回输出

    def forward_extend(  # 扩展阶段前向传播 # 扩展阶段前向传播
        self,
        q,  # 查询张量 # 查询
        k,  # 键张量 # 键
        v,  # 值张量 # 值
        layer: RadixAttention,  # 注意力层 # 注意力层
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存 # 是否保存KV缓存
    ):
        if layer.qk_head_dim != layer.v_head_dim:  # 如果QK头维度不等于V头维度 # 检查头维度差异
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))  # 创建不同维度的输出 # 创建适配维度输出
        else:  # 否则 # 维度相同
            o = torch.empty_like(q)  # 创建与查询相同形状的输出 # 创建同形状输出

        if save_kv_cache:  # 如果需要保存KV缓存 # 检查是否保存KV
            self.token_to_kv_pool.set_kv_buffer(  # 保存KV缓存 # 保存KV到缓存
                layer, forward_batch.out_cache_loc, k, v  # 层、缓存位置、键、值 # 传入参数
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num  # 判断是否使用GQA # 判断GQA

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)  # 重塑查询形状 # 重塑查询
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)  # 重塑输出形状 # 重塑输出

        causal = True  # 默认使用因果注意力 # 默认因果
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:  # 如果是交叉注意力或仅编码器 # 检查非因果情况
            raise NotImplementedError(  # 抛出未实现错误 # 抛出异常
                "TorchFlexAttnBackend does not support non-causal attention for now."  # 错误提示 # 错误提示
            )

        self._run_flex_forward_extend(  # 调用FlexAttention扩展前向 # 调用Flex扩展前向
            q_,  # 查询 # 查询
            o_,  # 输出 # 输出
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),  # 键缓存 # 键缓存
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),  # 值缓存 # 值缓存
            self.req_to_token_pool.req_to_token,  # 请求到token映射 # 请求-token映射
            forward_batch.req_pool_indices,  # 请求池索引 # 请求池索引
            forward_batch.seq_lens,  # 序列长度 # 序列长度
            forward_batch.extend_prefix_lens,  # 扩展前缀长度 # 扩展前缀长度
            forward_batch.extend_seq_lens,  # 扩展序列长度 # 扩展序列长度
            scaling=layer.scaling,  # 缩放因子 # 缩放因子
            enable_gqa=use_gqa,  # 是否启用GQA # GQA标志
            causal=causal,  # 是否因果 # 因果标志
        )
        return o  # 返回输出 # 返回输出

    def forward_decode(  # 解码阶段前向传播 # 解码阶段前向传播
        self,
        q,  # 查询张量 # 查询
        k,  # 键张量 # 键
        v,  # 值张量 # 值
        layer: RadixAttention,  # 注意力层 # 注意力层
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存 # 是否保存KV缓存
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        # 在torch.compile期间，rotary_emb有一个bug导致输出值为3D张量形状。此处正确重塑输出。
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)  # 重塑查询为一维token # 重塑查询

        if layer.qk_head_dim != layer.v_head_dim:  # 如果QK头维度不等于V头维度 # 检查头维度差异
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))  # 创建不同维度的输出 # 创建适配维度输出
        else:  # 否则 # 维度相同
            o = torch.empty_like(q)  # 创建与查询相同形状的输出 # 创建同形状输出

        if save_kv_cache:  # 如果需要保存KV缓存 # 检查是否保存KV
            self.token_to_kv_pool.set_kv_buffer(  # 保存KV缓存 # 保存KV到缓存
                layer, forward_batch.out_cache_loc, k, v  # 层、缓存位置、键、值 # 传入参数
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num  # 判断是否使用GQA # 判断GQA
        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)  # 重塑查询形状 # 重塑查询
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)  # 重塑输出形状 # 重塑输出

        self._run_flex_forward_decode(  # 调用FlexAttention解码前向 # 调用Flex解码前向
            q_,  # 查询 # 查询
            o_,  # 输出 # 输出
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),  # 键缓存 # 键缓存
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),  # 值缓存 # 值缓存
            self.req_to_token_pool.req_to_token,  # 请求到token映射 # 请求-token映射
            forward_batch.req_pool_indices,  # 请求池索引 # 请求池索引
            forward_batch.seq_lens,  # 序列长度 # 序列长度
            scaling=layer.scaling,  # 缩放因子 # 缩放因子
            enable_gqa=use_gqa,  # 是否启用GQA # GQA标志
            causal=False,  # 解码阶段不使用因果 # 解码不使用因果
        )

        return o  # 返回输出 # 返回输出

    def support_triton(self):  # 是否支持Triton后端 # 是否支持Triton
        return False  # 不支持Triton # 不支持Triton
