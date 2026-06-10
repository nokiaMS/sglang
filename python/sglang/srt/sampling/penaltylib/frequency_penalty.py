# 该文件实现了批处理频率惩罚器
# 频率惩罚根据token在输出中出现的频率（出现次数）进行惩罚
# 出现次数越多的token受到的惩罚越大，从而减少重复生成

import torch  # PyTorch张量库

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer  # 批处理惩罚器基类


class BatchedFrequencyPenalizer(_BatchedPenalizer):  # 批处理频率惩罚器
    """
    Frequency penalizer penalizes tokens based on their frequency in the output.
    """  # 频率惩罚器根据token在输出中的出现频率进行惩罚

    def _is_required(self) -> bool:  # 检查是否有请求需要频率惩罚
        """检查是否有任何请求的频率惩罚参数不为零"""
        return any(
            req.sampling_params.frequency_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):  # 准备频率惩罚所需的张量
        """初始化累计频率惩罚张量和频率惩罚系数张量"""
        self.cumulated_frequency_penalties = torch.zeros(  # 创建累计频率惩罚张量
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),  # 形状为(请求数, 词表大小)
            dtype=torch.float32,  # 32位浮点
            device=self.orchestrator.device,  # 设备
        )

        self.frequency_penalties = (  # 创建频率惩罚系数张量
            torch.tensor(
                data=[
                    req.sampling_params.frequency_penalty
                    for req in self.orchestrator.reqs()  # 每个请求的频率惩罚系数
                ],
                dtype=torch.float32,  # 32位浮点
                device=self.orchestrator.device,  # 设备
            )
        ).unsqueeze_(1)  # 增加维度以进行广播

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):  # 累计输出token的频率惩罚
        """根据输出token更新累计频率惩罚"""
        self.cumulated_frequency_penalties.scatter_add_(  # 使用scatter_add_累加频率惩罚
            dim=1,  # 在词表维度上累加
            index=output_ids.unsqueeze(1),  # 输出token ID索引
            src=self.frequency_penalties,  # 频率惩罚系数作为源
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:  # 应用频率惩罚到logits
        """从logits中减去累计频率惩罚"""
        logits.sub_(self.cumulated_frequency_penalties)  # 原地减去累计频率惩罚

    def _filter(self, keep_indices: torch.Tensor):  # 根据保留索引过滤张量
        """根据保留的请求索引过滤频率惩罚张量"""
        self.frequency_penalties = self.frequency_penalties[keep_indices]  # 过滤频率惩罚系数
        self.cumulated_frequency_penalties = self.cumulated_frequency_penalties[
            keep_indices  # 过滤累计频率惩罚
        ]

    def _merge(self, their: "BatchedFrequencyPenalizer"):  # 合并另一个频率惩罚器
        """将另一个频率惩罚器的张量合并到当前惩罚器"""
        self.frequency_penalties = torch.cat(
            [self.frequency_penalties, their.frequency_penalties], dim=0  # 沿批次维度拼接频率惩罚系数
        )
        self.cumulated_frequency_penalties = torch.cat(
            [self.cumulated_frequency_penalties, their.cumulated_frequency_penalties],
            dim=0,  # 沿批次维度拼接累计频率惩罚
        )

    def _teardown(self) -> None:  # 清理资源
        """清理频率惩罚器持有的张量资源"""
        for name in ("frequency_penalties", "cumulated_frequency_penalties"):  # 遍历需要清理的属性名
            if hasattr(self, name):  # 如果属性存在
                delattr(self, name)  # 删除属性以释放内存
