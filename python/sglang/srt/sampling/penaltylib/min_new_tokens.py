# 该文件实现了批处理最小新token数惩罚器
# 当输出token数未达到最小新token数要求时，对停止token施加负无穷惩罚
# 防止模型过早生成停止token而结束输出

import torch  # PyTorch张量库

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer  # 批处理惩罚器基类


class BatchedMinNewTokensPenalizer(_BatchedPenalizer):  # 批处理最小新token数惩罚器
    """
    Min new tokens penalizer penalizes tokens based on the length of the output.
    """  # 最小新token数惩罚器根据输出长度对token进行惩罚

    def _is_required(self) -> bool:  # 检查是否有请求需要最小新token数惩罚
        """检查是否有任何请求的最小新token数参数大于零"""
        return any(
            req.sampling_params.min_new_tokens > 0 for req in self.orchestrator.reqs()
        )

    def _prepare(self):  # 准备最小新token数惩罚所需的张量
        """初始化最小新token数、停止token惩罚和输出token计数张量"""
        self.min_new_tokens = torch.tensor(  # 创建最小新token数张量
            data=[
                req.sampling_params.min_new_tokens for req in self.orchestrator.reqs()  # 每个请求的最小新token数
            ],
            dtype=torch.int32,  # 32位整数
            device=self.orchestrator.device,  # 设备
        ).unsqueeze_(1)  # 增加维度以进行广播

        padded_stop_token_ids = torch.nn.utils.rnn.pad_sequence(  # 对停止token ID进行填充
            sequences=[
                torch.tensor(
                    data=(
                        list(
                            (req.sampling_params.stop_token_ids or set())  # 用户指定的停止token
                            | (req.tokenizer.additional_stop_token_ids or set())  # 分词器额外的停止token
                            | {req.tokenizer.eos_token_id}  # EOS token
                        )
                    ),
                    dtype=torch.int64,  # 64位整数
                    device=self.orchestrator.device,  # 设备
                )
                for req in self.orchestrator.reqs()  # 遍历所有请求
            ],
            batch_first=True,  # 批次维度在前
            padding_value=self.orchestrator.vocab_size,  # 使用词表大小作为填充值（超出有效范围）
        )
        self.stop_token_penalties = torch.zeros(  # 创建停止token惩罚张量
            size=(len(self.orchestrator.reqs()), self.orchestrator.vocab_size + 1),  # 额外+1用于填充位置
            dtype=torch.float32,  # 32位浮点
            device=self.orchestrator.device,  # 设备
        ).scatter_add_(  # 使用scatter_add_将负无穷写入停止token位置
            dim=1,  # 在词表维度上散列
            index=padded_stop_token_ids,  # 停止token ID索引
            src=torch.full_like(
                input=padded_stop_token_ids,
                dtype=torch.float32,  # 32位浮点
                fill_value=float("-inf"),  # 填充值为负无穷
                device=self.orchestrator.device,  # 设备
            ),
        )[
            :, : self.orchestrator.vocab_size  # 截断到词表大小
        ]

        self.len_output_tokens = torch.zeros(  # 创建输出token计数张量
            size=(len(self.orchestrator.reqs()), 1),  # 形状为(请求数, 1)
            dtype=torch.int32,  # 32位整数
            device=self.orchestrator.device,  # 设备
        )

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):  # 累计输出token数量
        """每生成一个token，增加输出计数"""
        self.len_output_tokens += 1  # 输出token计数加1

    def _apply(self, logits: torch.Tensor):  # 应用最小新token数惩罚
        """当输出token数未达到最小新token数时，对停止token施加负无穷惩罚"""
        mask = (self.len_output_tokens < self.min_new_tokens).expand_as(logits)  # 生成掩码：输出token数小于最小要求
        logits[mask] += self.stop_token_penalties[mask]  # 对停止token位置加上负无穷惩罚

    def _filter(self, keep_indices: torch.Tensor):  # 根据保留索引过滤张量
        """根据保留的请求索引过滤相关张量"""
        self.min_new_tokens = self.min_new_tokens[keep_indices]  # 过滤最小新token数
        self.stop_token_penalties = self.stop_token_penalties[keep_indices]  # 过滤停止token惩罚
        self.len_output_tokens = self.len_output_tokens[keep_indices]  # 过滤输出token计数

    def _merge(self, their: "BatchedMinNewTokensPenalizer"):  # 合并另一个最小新token数惩罚器
        """将另一个惩罚器的张量合并到当前惩罚器"""
        self.min_new_tokens = torch.cat(
            [self.min_new_tokens, their.min_new_tokens], dim=0  # 沿批次维度拼接最小新token数
        )
        self.stop_token_penalties = torch.cat(
            [self.stop_token_penalties, their.stop_token_penalties], dim=0  # 沿批次维度拼接停止token惩罚
        )
        self.len_output_tokens = torch.cat(
            [self.len_output_tokens, their.len_output_tokens], dim=0  # 沿批次维度拼接输出token计数
        )

    # Explicit resource cleanup to aid GC and free CUDA memory promptly  # 显式资源清理以帮助GC并及时释放CUDA内存
    def _teardown(self) -> None:  # 清理资源
        """清理最小新token数惩罚器持有的张量资源"""
        for name in ("min_new_tokens", "stop_token_penalties", "len_output_tokens"):  # 遍历需要清理的属性名
            if hasattr(self, name):  # 如果属性存在
                delattr(self, name)  # 删除属性以释放内存
