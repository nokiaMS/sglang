# DeepSeek DSA稀疏注意力算法实现文件
# 该算法使用DeepSeek DSA的原生索引器进行TopK检索，重写了父类的所有方法
from typing import Any, Optional  # 导入类型提示中的Any和Optional类型

import torch  # 导入PyTorch张量库

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (  # 从基类模块导入稀疏注意力算法实现基类
    BaseSparseAlgorithmImpl,  # 稀疏注意力算法实现基类
)


class DeepSeekDSAAlgorithm(BaseSparseAlgorithmImpl):  # DeepSeek DSA稀疏注意力算法，继承自BaseSparseAlgorithmImpl
    """
    Sparse attention algorithm for DeepSeek DSA.
    DeepSeek DSA的稀疏注意力算法。

    This algorithm uses DSA's native indexer for TopK retrieval.
    该算法使用DSA的原生索引器进行TopK检索。
    Overrides all parent methods as DSA has its own specialized flow.
    重写了所有父类方法，因为DSA有自己的专用流程。
    """

    def __init__(self, config, device: torch.device, **kwargs):  # 初始化DeepSeek DSA算法
        super().__init__(config, device, **kwargs)  # 调用父类初始化

    def retrieve_topk(  # 使用DSA原生索引器检索Top-K KV索引
        self,
        queries: torch.Tensor,  # 查询向量张量
        layer_id: int,  # 当前层索引
        req_pool_indices: torch.Tensor,  # 请求池索引张量
        sparse_mask: torch.Tensor,  # 稀疏注意力掩码
        attn_metadata: Optional[Any],  # 注意力元数据（可选）
        **kwargs,  # 额外参数
    ) -> tuple:  # 返回索引和None的元组
        indexer, forward_batch, x, q_lora, positions = (  # 从kwargs中提取DSA所需的参数
            kwargs.get("indexer"),  # DSA索引器
            kwargs.get("forward_batch"),  # 前向批次信息
            kwargs.get("x"),  # 输入张量x
            kwargs.get("q_lora"),  # 查询LoRA张量
            kwargs.get("positions"),  # 位置编码
        )

        if any(v is None for v in [indexer, x, q_lora, positions, forward_batch]):  # 检查必要参数是否都存在
            raise ValueError("Required: indexer, forward_batch, x, q_lora, positions")  # 缺少必要参数时抛出异常

        return (  # 返回DSA索引器的结果
            indexer(  # 调用DSA原生索引器
                x=x,  # 输入张量x
                q_lora=q_lora,  # 查询LoRA张量
                positions=positions,  # 位置编码
                forward_batch=forward_batch,  # 前向批次信息
                layer_id=layer_id,  # 当前层索引
            ),
            None,  # DSA不需要返回有效长度，返回None
        )

    def initialize_representation_pool(  # 初始化表示池（DSA不需要，空实现）
        self,
        start_layer: int,  # 起始层索引
        end_layer: int,  # 结束层索引
        token_to_kv_pool,  # token到KV缓存池的映射
        req_to_token_pool,  # 请求到token的映射池
        states,  # 算法状态对象
    ):
        pass  # DSA使用原生索引器，不需要自建表示池

    def construct_representations(  # 构建表示（DSA不需要，空实现）
        self,
        layer_id,  # 当前层索引
        req_pool_indices,  # 请求池索引
        seq_lens,  # 序列长度
        k_buffer,  # 键缓存缓冲区
        forward_batch,  # 前向批次信息
    ):
        pass  # DSA使用原生索引器，不需要构建表示

    def update_representations(  # 更新表示（DSA不需要，空实现）
        self,
        layer_id,  # 当前层索引
        req_pool_indices,  # 请求池索引
        seq_lens,  # 序列长度
        k_buffer,  # 键缓存缓冲区
        forward_batch,  # 前向批次信息
    ):
        pass  # DSA使用原生索引器，不需要更新表示
