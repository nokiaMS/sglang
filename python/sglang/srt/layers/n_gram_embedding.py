# N-gram 嵌入层模块
# 本文件实现了 N-gram 嵌入层（NgramEmbedding），用于基于 N-gram 的词嵌入计算，
# 支持重叠嵌入（overlap embedding）和张量并行，主要用于语言模型中 N-gram 特征的提取。
import torch  # 导入 PyTorch 库
from torch import nn  # 导入 PyTorch 神经网络模块
from torch.nn import Parameter  # 导入参数类

from sglang.jit_kernel.ngram_embedding import compute_n_gram_ids  # 导入 N-gram ID 计算 JIT 内核
from sglang.srt.layers.dp_attention import is_dp_attention_enabled  # 导入数据并行注意力启用检查函数
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding  # 导入词表并行嵌入层
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息类


class NgramEmbedding(torch.nn.Module):  # N-gram 嵌入层类，继承自 torch.nn.Module

    def __init__(  # 初始化函数，定义 N-gram 嵌入层的参数
        self,
        num_embeddings: int,  # 嵌入数量（词表大小）
        embedding_dim: int,  # 嵌入维度
        over_embedding_m: int,  # 重叠嵌入参数 m
        over_embedding_k: int,  # 重叠嵌入参数 k
        over_embedding_n: int,  # 重叠嵌入参数 n
    ):
        super().__init__()  # 调用父类初始化
        assert (  # 断言 over_embedding_n 必须大于 1
            over_embedding_n > 1
        ), f"over_embedding_n must be > 1, got {over_embedding_n}"  # over_embedding_n 必须大于 1，否则报错
        self.num_embeddings = num_embeddings  # 保存嵌入数量
        self.embedding_dim = embedding_dim  # 保存嵌入维度
        self.over_embedding_m = over_embedding_m  # 保存重叠嵌入参数 m
        self.over_embedding_k = over_embedding_k  # 保存重叠嵌入参数 k
        self.over_embedding_n = over_embedding_n  # 保存重叠嵌入参数 n

        self.word_embeder = VocabParallelEmbedding(  # 创建词嵌入层
            num_embeddings,  # 词表大小
            embedding_dim,  # 嵌入维度
            enable_tp=is_dp_attention_enabled(),  # 根据数据并行注意力是否启用来决定是否开启张量并行
        )
        self.n_grams = (over_embedding_n - 1) * over_embedding_k  # 计算 N-gram 的总数
        oe_hidden_dim = embedding_dim // (over_embedding_k * (over_embedding_n - 1))  # 计算重叠嵌入的隐藏维度
        self.exclusive_oe_embedder_size_sums = torch.zeros(  # 创建排他性重叠嵌入器大小累加和数组
            [over_embedding_k * (over_embedding_n - 1) + 1],  # 数组长度
            dtype=torch.int32,  # 数据类型为 int32
            device="cuda",  # 设备为 CUDA
        )
        for i in range(over_embedding_k * (over_embedding_n - 1)):  # 遍历每个重叠嵌入器
            self.exclusive_oe_embedder_size_sums[i + 1] = (  # 计算排他性累加和
                self.exclusive_oe_embedder_size_sums[i]  # 前一个累加值
                + int(over_embedding_m + i * 2 + 1)  # 加上当前嵌入器的大小
            )
        self.oe_embeder = VocabParallelEmbedding(  # 创建重叠嵌入层
            num_embeddings=self.exclusive_oe_embedder_size_sums[-1],  # 嵌入数量为最后一个累加和值
            embedding_dim=oe_hidden_dim,  # 嵌入维度为计算出的隐藏维度
            enable_tp=is_dp_attention_enabled(),  # 根据数据并行注意力是否启用来决定是否开启张量并行
        )

        self.oe_projection = nn.Parameter(  # 创建重叠嵌入投影参数
            torch.empty(  # 创建空张量
                (over_embedding_n - 1) * over_embedding_k, oe_hidden_dim, embedding_dim  # 形状为 (n_grams, oe_hidden_dim, embedding_dim)
            ),
            requires_grad=False,  # 不需要梯度
        )

        self.oe_mods = torch.zeros(  # 创建取模值数组
            [self.over_embedding_n - 1, self.over_embedding_k], dtype=torch.int32  # 形状为 (n-1, k)
        )
        self.oe_weights = torch.zeros(  # 创建权重数组
            [self.over_embedding_n - 1, self.over_embedding_k, self.over_embedding_n],  # 形状为 (n-1, k, n)
            dtype=torch.int32,  # 数据类型为 int32
        )
        for n in range(2, self.over_embedding_n + 1):  # 遍历 n 从 2 到 over_embedding_n
            for k in range(self.over_embedding_k):  # 遍历 k 从 0 到 over_embedding_k-1
                mod = (  # 计算取模值
                    self.over_embedding_m  # 基础参数 m
                    + 2 * ((n - 2) * self.over_embedding_k + k)  # 加上偏移量
                    + 1  # 加 1
                )
                self.oe_mods[n - 2][k] = mod  # 保存取模值
                for delta in range(self.over_embedding_n):  # 遍历 delta 从 0 到 over_embedding_n-1
                    self.oe_weights[n - 2][k][delta] = pow(num_embeddings, delta, mod)  # 计算 num_embeddings^delta mod mod

    def init_buffers(  # 初始化缓冲区函数，为前向传播预分配 GPU 缓冲区
        self, max_running_requests: int, chunked_prefill_size: int, device: str  # 最大运行请求数、分块预填充大小、设备
    ):
        max_tokens = max(chunked_prefill_size, max_running_requests)  # 计算最大 token 数
        self.oe_n_gram_ids = torch.zeros(  # 创建 N-gram ID 缓冲区
            [max_tokens, self.n_grams],  # 形状为 (max_tokens, n_grams)
            dtype=torch.int32,  # 数据类型为 int32
            device=device,  # 指定设备
        )
        self.exclusive_req_len_sums = torch.zeros(  # 创建排他性请求长度累加和缓冲区
            max_running_requests + 1, dtype=torch.int32, device=device  # 形状为 (max_running_requests+1,)
        )

    def load_weight(  # 加载权重函数，将预训练权重加载到对应的嵌入层参数中
        self, param: Parameter, weight_name: str, loaded_weight: torch.Tensor  # 参数对象、权重名称、加载的权重张量
    ):
        if ".embed_tokens." in weight_name:  # 如果是词嵌入权重
            param.weight_loader(param, loaded_weight)  # 使用权重加载器加载
        elif "model.ngram_embeddings.embedders." in weight_name:  # 如果是 N-gram 嵌入器权重
            index = int(  # 解析嵌入器索引
                weight_name.replace("model.ngram_embeddings.embedders.", "").replace(  # 移除前缀
                    ".weight", ""  # 移除后缀
                )
            )
            oe_weight_start = self.exclusive_oe_embedder_size_sums[index]  # 获取当前嵌入器在重叠嵌入层中的起始位置
            oe_weight_end = self.exclusive_oe_embedder_size_sums[index + 1]  # 获取当前嵌入器在重叠嵌入层中的结束位置
            assert (  # 断言权重大小匹配
                oe_weight_end - oe_weight_start == loaded_weight.shape[0]  # 检查权重行数是否一致
            ), f"{oe_weight_end - oe_weight_start=} {loaded_weight.shape[0]=}"  # 权重大小不匹配时报错
            tp_start = self.oe_embeder.shard_indices.org_vocab_start_index  # 张量并行分片的起始索引
            tp_end = self.oe_embeder.shard_indices.org_vocab_end_index  # 张量并行分片的结束索引
            to_load_start = max(oe_weight_start, tp_start)  # 计算实际需要加载的起始位置
            to_load_end = min(oe_weight_end, tp_end)  # 计算实际需要加载的结束位置
            if to_load_start < to_load_end:  # 如果有重叠区域需要加载
                src_start = to_load_start - oe_weight_start  # 计算源张量的起始索引
                src_end = to_load_end - oe_weight_start  # 计算源张量的结束索引
                dest_start = to_load_start - tp_start  # 计算目标张量的起始索引
                dest_end = to_load_end - tp_start  # 计算目标张量的结束索引
                self.oe_embeder.weight.data[dest_start:dest_end] = loaded_weight[  # 将权重复制到目标位置
                    src_start:src_end  # 源张量的切片范围
                ]
            else:  # 如果当前分片不需要加载该权重
                return  # 直接返回
        elif "model.ngram_embeddings.post_projs." in weight_name:  # 如果是后投影权重
            index = int(  # 解析投影索引
                weight_name.replace("model.ngram_embeddings.post_projs.", "").replace(  # 移除前缀
                    ".weight", ""  # 移除后缀
                )
            )
            self.oe_projection[index].copy_(loaded_weight.data.t())  # 将权重转置后复制到投影参数
        else:  # 其他未知权重名称
            assert False, f"Unknown ngram embedding weight name: {weight_name}"  # 报错：未知的 N-gram 嵌入权重名称

    def forward(self, input_ids: torch.Tensor, forward_batch: ForwardBatch):  # 前向传播函数，计算 N-gram 嵌入
        if (  # 如果是扩展模式或解码模式
            forward_batch.forward_mode.is_extend()  # 检查是否为扩展模式
            or forward_batch.forward_mode.is_decode()  # 检查是否为解码模式
        ):
            ngram_embedding_info = forward_batch.ngram_embedding_info  # 获取 N-gram 嵌入信息
            torch.cumsum(  # 计算请求长度的累加和
                ngram_embedding_info.req_lens,  # 请求长度张量
                dim=0,  # 沿第 0 维累加
                dtype=torch.int32,  # 输出数据类型为 int32
                out=self.exclusive_req_len_sums[1 : 1 + forward_batch.batch_size],  # 输出到排他性请求长度累加和缓冲区
            )
            compute_n_gram_ids(  # 调用 JIT 内核计算 N-gram ID
                ne_n=self.over_embedding_n,  # N-gram 的 n 参数
                ne_k=self.over_embedding_k,  # N-gram 的 k 参数
                ne_weights=self.oe_weights,  # N-gram 权重
                ne_mods=self.oe_mods,  # N-gram 取模值
                tokens=input_ids.to(torch.int32),  # 输入 token ID，转换为 int32
                exclusive_ne_embedder_size_sums=self.exclusive_oe_embedder_size_sums,  # 排他性嵌入器大小累加和
                exclusive_req_len_sums=self.exclusive_req_len_sums[  # 排他性请求长度累加和
                    : forward_batch.batch_size + 1  # 截取到批次大小+1
                ],
                ne_token_table=ngram_embedding_info.token_table,  # N-gram token 表
                row_indices=forward_batch.req_pool_indices,  # 请求池索引
                column_starts=ngram_embedding_info.column_starts,  # 列起始索引
                n_gram_ids=self.oe_n_gram_ids[: len(input_ids)],  # 输出的 N-gram ID 缓冲区
            )

        # [13, seq_len, hidden_dim]  # 形状为 [13, seq_len, hidden_dim] 的全隐藏状态张量
        all_hidden_states = torch.empty(  # 创建全隐藏状态张量
            [self.n_grams + 1, len(input_ids), self.embedding_dim],  # 形状为 (n_grams+1, seq_len, embedding_dim)
            dtype=self.oe_projection.dtype,  # 数据类型与投影参数一致
            device=input_ids.device,  # 设备与输入一致
        )
        all_hidden_states[0] = self.word_embeder(input_ids)  # 第一个隐藏状态为词嵌入结果
        # oe_hidden_states: [12, seq_len, hidden_dim / 12]  # 重叠嵌入隐藏状态：形状为 [12, seq_len, hidden_dim/12]
        oe_hidden_states = self.oe_embeder(  # 计算重叠嵌入
            self.oe_n_gram_ids[: len(input_ids)].permute(1, 0).contiguous()  # 将 N-gram ID 转置并确保连续内存
        )
        torch.bmm(oe_hidden_states, self.oe_projection, out=all_hidden_states[1:])  # 批量矩阵乘法，投影重叠嵌入到全维度
        return all_hidden_states.mean(dim=0)  # 返回所有隐藏状态的均值作为最终嵌入
