# Quest稀疏注意力算法实现文件
# 基于Quest论文的边界框估计方法进行查询感知的页选择
# 为每个KV页维护键的逐维度最小/最大值，使用它们来上界估计注意力分数
"""
Quest sparse attention algorithm.
Quest稀疏注意力算法。

This implementation follows the Quest paper's bounding-box estimation for
query-aware page selection. For each KV page, it maintains per-dimension
min/max of keys and uses them to upper-bound attention scores without
materializing full dot products.
本实现遵循Quest论文的边界框估计方法进行查询感知的页选择。对于每个KV页，它维护
键的逐维度最小/最大值，使用它们来上界估计注意力分数，而无需计算完整的点积。
"""

import logging  # 导入日志模块

import torch  # 导入PyTorch张量库

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (  # 从基类模块导入稀疏注意力算法实现基类
    BaseSparseAlgorithmImpl,  # 稀疏注意力算法实现基类
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class QuestAlgorithm(BaseSparseAlgorithmImpl):  # Quest逐页稀疏注意力算法，使用边界框关键性估计
    """Quest page-wise sparse attention using bounding-box criticality. Quest逐页稀疏注意力，使用边界框关键性估计。"""

    def __init__(self, config, device: torch.device, **kwargs):  # 初始化Quest算法
        super().__init__(config, device, **kwargs)  # 调用父类初始化
        self.page_k_min = {}  # 存储每页键的最小值，按层索引
        self.page_k_max = {}  # 存储每页键的最大值，按层索引
        self.page_valid = {}  # 存储每页是否有效（已初始化），按层索引

    def _initialize_representation_pools(  # 初始化Quest的页表示池（每页键的min/max边界框）
        self, start_layer: int, end_layer: int, total_num_pages: int  # 起始层、结束层、总页数
    ):
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)  # 获取起始层的键缓存缓冲区
        head_num, head_dim = key_buf.shape[1], key_buf.shape[2]  # 获取注意力头数和头维度

        for layer_id in range(start_layer, end_layer):  # 遍历每一层
            self.page_k_min[layer_id] = torch.zeros(  # 初始化该层页键最小值张量
                (total_num_pages, head_num, head_dim),  # 形状：[总页数, 头数, 头维度]
                dtype=torch.float32,  # 使用float32精度
                device=self.device,  # 放在指定设备上
            )
            self.page_k_max[layer_id] = torch.zeros_like(self.page_k_min[layer_id])  # 初始化该层页键最大值张量，与最小值同形状
            self.page_valid[layer_id] = torch.zeros(  # 初始化该层页有效标记张量
                total_num_pages, dtype=torch.bool, device=self.device  # 形状：[总页数]，布尔类型
            )

        logger.info(  # 记录初始化信息
            "Initialized Quest page reps: %d pages, %d layers, head_num=%d, head_dim=%d",  # 格式化字符串
            total_num_pages,  # 总页数
            end_layer - start_layer,  # 层数
            head_num,  # 头数
            head_dim,  # 头维度
        )

    def _compute_page_representations(  # 计算并存储给定页范围的页表示（键的min/max边界框）
        self,
        layer_id: int,  # 当前层索引
        reqs: torch.Tensor,  # 请求索引张量
        seq_lens: torch.Tensor,  # 序列长度张量
        start_page,  # 起始页索引
        end_page: torch.Tensor,  # 结束页索引张量
        k_buffer: torch.Tensor,  # 键缓存缓冲区
    ):
        if isinstance(start_page, int):  # 如果起始页是整数（预填充阶段）
            start_page = torch.full_like(end_page, start_page)  # 将其扩展为与end_page同形状的张量

        device = k_buffer.device  # 获取键缓存所在设备
        req_to_token = self.req_to_token_pool.req_to_token  # 获取请求到token的映射矩阵
        n = reqs.shape[0]  # 请求数量
        max_pages = int((end_page - start_page).max().item())  # 计算最大需要处理的页数
        if max_pages <= 0:  # 如果没有页需要处理
            return  # 直接返回

        pg_off = torch.arange(max_pages, device=device).unsqueeze(0)  # 页偏移量[0, 1, ..., max_pages-1]，增加batch维度
        pg_id = start_page.unsqueeze(1) + pg_off  # 计算每个请求的页ID
        pg_mask = pg_id < end_page.unsqueeze(1)  # 构建页有效掩码（页ID小于结束页）

        tok_start = pg_id * self.page_size  # 计算每页起始token的逻辑位置
        tok_off = torch.arange(self.page_size, device=device).view(1, 1, -1)  # 页内token偏移量，形状[1, 1, page_size]
        tok_pos = tok_start.unsqueeze(2) + tok_off  # 计算每个token的逻辑位置
        tok_mask = (  # 构建token有效掩码
            tok_pos  # token位置
            < (tok_start + self.page_size).clamp(max=seq_lens.unsqueeze(1)).unsqueeze(2)  # 不超过序列长度
        ) & pg_mask.unsqueeze(2)  # 且所在页有效

        phys_tok = req_to_token[  # 获取token的物理索引
            reqs.view(n, 1, 1).expand(n, max_pages, self.page_size),  # 扩展请求索引以匹配token位置
            tok_pos.clamp(0, req_to_token.shape[1] - 1),  # 限制token位置在有效范围内
        ].clamp(0, k_buffer.shape[0] - 1)  # 限制物理索引在键缓存范围内

        keys = k_buffer[phys_tok].to(torch.float32)  # 从键缓存中获取键向量，转为float32
        mask = tok_mask.unsqueeze(-1).unsqueeze(-1)  # 扩展掩码维度以匹配键向量形状

        page_min = torch.where(mask, keys, torch.full_like(keys, float("inf"))).amin(  # 计算每页键的最小值（无效位置用inf替代）
            dim=2  # 沿页内token维度取最小
        )
        page_max = torch.where(mask, keys, torch.full_like(keys, float("-inf"))).amax(  # 计算每页键的最大值（无效位置用-inf替代）
            dim=2  # 沿页内token维度取最大
        )

        phys_pg = (  # 计算物理页索引
            req_to_token[  # 获取页起始token的物理索引
                reqs.unsqueeze(1).expand(n, max_pages),  # 扩展请求索引
                tok_start.clamp(0, req_to_token.shape[1] - 1),  # 限制在有效范围内
            ]
            // self.page_size  # 物理页索引 = 物理token索引 // 页大小
        )

        idx = pg_mask.nonzero(as_tuple=False)  # 获取有效页的索引
        if idx.numel() == 0:  # 如果没有有效页
            return  # 直接返回

        target_pages = phys_pg[idx[:, 0], idx[:, 1]].clamp(  # 获取目标物理页索引，限制在存储范围内
            0, self.page_k_min[layer_id].shape[0] - 1  # 不超过表示池大小
        )
        self.page_k_min[layer_id][target_pages] = page_min[idx[:, 0], idx[:, 1]]  # 更新目标页的键最小值
        self.page_k_max[layer_id][target_pages] = page_max[idx[:, 0], idx[:, 1]]  # 更新目标页的键最大值
        self.page_valid[layer_id][target_pages] = True  # 标记目标页为有效

    def _retrieve_page_scores(  # 使用边界框关键性估计检索页分数
        self,
        layer_id: int,  # 当前层索引
        phys_pages: torch.Tensor,  # 物理页索引张量
        req_pool_indices: torch.Tensor,  # 请求池索引张量
        queries: torch.Tensor,  # 查询向量张量
    ) -> torch.Tensor:  # 返回页关键性分数张量
        # Clamp pages to valid storage range 将页索引限制在有效存储范围内
        phys_pages_clamped = phys_pages.clamp(0, self.page_k_min[layer_id].shape[0] - 1)  # 限制页索引不超过表示池大小

        k_min = self.page_k_min[layer_id][phys_pages_clamped]  # 获取对应页的键最小值
        k_max = self.page_k_max[layer_id][phys_pages_clamped]  # 获取对应页的键最大值
        valid_mask = self.page_valid[layer_id][phys_pages_clamped]  # 获取对应页的有效标记
        # Align query shape to KV heads. 将查询形状对齐到KV头数。
        head_dim = k_min.shape[-1]  # 获取头维度
        if queries.dim() == 2:  # 如果查询是2维的[bs, hidden]
            bs, hidden = queries.shape  # 获取批次大小和隐藏维度
            if hidden % head_dim != 0:  # 如果隐藏维度不能被头维度整除
                raise ValueError(  # 抛出值错误异常
                    f"Quest query hidden size {hidden} not divisible by head_dim {head_dim}"  # 查询隐藏维度不能被头维度整除
                )
            q_heads = hidden // head_dim  # 计算查询头数
            q = queries.view(bs, q_heads, head_dim)  # 重塑查询为[bs, q_heads, head_dim]
        elif queries.dim() == 3:  # 如果查询已经是3维的[bs, num_heads, head_dim]
            q = queries  # 直接使用
        else:  # 其他维度不支持
            raise ValueError(f"Unsupported query shape for Quest: {queries.shape}")  # 不支持的查询形状

        kv_heads = k_min.shape[-2]  # 获取KV头数
        q_heads = q.shape[1]  # 获取查询头数
        if q_heads != kv_heads:  # 如果查询头数与KV头数不匹配（GQA/MQA情况）
            if q_heads % kv_heads != 0:  # 如果查询头数不能被KV头数整除
                raise ValueError(  # 抛出值错误异常
                    f"Query heads {q_heads} not divisible by KV heads {kv_heads}"  # 查询头数不能被KV头数整除
                )
            group = q_heads // kv_heads  # 计算每组查询头数
            # Average grouped query heads to align with KV heads (approximation for MQA/GQA).
            # 对分组查询头取平均以对齐KV头数（MQA/GQA的近似处理）。
            q = q.view(q.shape[0], kv_heads, group, head_dim).mean(dim=2)  # 重塑并沿组维度取平均

        q = q.to(k_min.dtype).unsqueeze(1)  # 转换查询类型并增加维度 -> [bs, 1, kv_heads, head_dim]

        criticality = torch.where(q >= 0, q * k_max, q * k_min).sum(dim=(2, 3))  # 计算边界框关键性分数：查询为正时乘k_max，为负时乘k_min，然后求和
        criticality = torch.where(  # 处理无效页
            valid_mask, criticality, torch.full_like(criticality, float("-inf"))  # 有效页使用计算分数，无效页设为负无穷
        )

        return criticality  # 返回关键性分数
