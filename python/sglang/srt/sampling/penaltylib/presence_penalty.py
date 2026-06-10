# 该文件实现了批处理存在惩罚器
# 存在惩罚根据token是否在输出中出现过进行惩罚
# 只要token出现过（无论次数），就施加固定值的惩罚，从而减少重复生成的可能性

import torch  # PyTorch张量库

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer  # 批处理惩罚器基类


class BatchedPresencePenalizer(_BatchedPenalizer):  # 批处理存在惩罚器
    """
    Presence penalizer penalizes tokens based on their presence in the output.
    """  # 存在惩罚器根据token在输出中是否出现进行惩罚

    def _is_required(self) -> bool:  # 检查是否有请求需要存在惩罚
        """检查是否有任何请求的存在惩罚参数不为零"""
        return any(
            req.sampling_params.presence_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):  # 准备存在惩罚所需的张量
        """初始化累计存在惩罚张量和存在惩罚系数张量"""
        self.cumulated_presence_penalties = torch.zeros(  # 创建累计存在惩罚张量
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),  # 形状为(请求数, 词表大小)
            dtype=torch.float32,  # 32位浮点
            device=self.orchestrator.device,  # 设备
        )

        self.presence_penalties = (  # 创建存在惩罚系数张量
            torch.tensor(
                data=[
                    req.sampling_params.presence_penalty
                    for req in self.orchestrator.reqs()  # 每个请求的存在惩罚系数
                ],
                dtype=torch.float32,  # 32位浮点
                device=self.orchestrator.device,  # 设备
            )
        ).unsqueeze_(1)  # 增加维度以进行广播

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):  # 累计输出token的存在惩罚
        """根据输出token更新累计存在惩罚（使用scatter_而非scatter_add_，因此只标记是否出现）"""
        self.cumulated_presence_penalties.scatter_(  # 使用scatter_标记token的出现（覆盖而非累加）
            dim=1,  # 在词表维度上散列
            index=output_ids.unsqueeze(1),  # 输出token ID索引
            src=self.presence_penalties,  # 存在惩罚系数作为源
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:  # 应用存在惩罚到logits
        """从logits中减去累计存在惩罚"""
        logits.sub_(self.cumulated_presence_penalties)  # 原地减去累计存在惩罚

    def _filter(self, keep_indices: torch.Tensor):  # 根据保留索引过滤张量
        """根据保留的请求索引过滤存在惩罚张量"""
        self.presence_penalties = self.presence_penalties[keep_indices]  # 过滤存在惩罚系数
        self.cumulated_presence_penalties = self.cumulated_presence_penalties[
            keep_indices  # 过滤累计存在惩罚
        ]

    def _merge(self, their: "BatchedPresencePenalizer"):  # 合并另一个存在惩罚器
        """将另一个存在惩罚器的张量合并到当前惩罚器"""
        self.presence_penalties = torch.cat(
            [self.presence_penalties, their.presence_penalties], dim=0  # 沿批次维度拼接存在惩罚系数
        )
        self.cumulated_presence_penalties = torch.cat(
            [self.cumulated_presence_penalties, their.cumulated_presence_penalties],
            dim=0,  # 沿批次维度拼接累计存在惩罚
        )

    def _teardown(self) -> None:  # 清理资源
        """清理存在惩罚器持有的张量资源"""
        for name in ("presence_penalties", "cumulated_presence_penalties"):  # 遍历需要清理的属性名
            if hasattr(self, name):  # 如果属性存在
                delattr(self, name)  # 删除属性以释放内存
