# 该文件实现了批处理重复惩罚器
# 重复惩罚根据token是否在输出中出现过进行乘性惩罚
# 对正logit除以惩罚系数，对负logit乘以惩罚系数，从而降低已出现token被再次选中的概率

import torch  # PyTorch张量库

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer  # 批处理惩罚器基类
from sglang.srt.utils import get_compiler_backend, is_npu  # 获取编译器后端和NPU检测

_is_npu = is_npu()  # 检测是否为NPU设备


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)  # 使用torch.compile优化
def apply_scaling_penalties(logits, scaling_penalties):  # 应用缩放惩罚到logits
    """对logits应用缩放惩罚：正logit除以系数，负logit乘以系数"""
    logits[:] = torch.where(
        logits < 0,  # 如果logit为负
        logits * scaling_penalties,  # 负logit乘以缩放系数（使其更负）
        logits / scaling_penalties,  # 正logit除以缩放系数（使其更小）
    )


class BatchedRepetitionPenalizer(_BatchedPenalizer):  # 批处理重复惩罚器
    """
    Repetition penalizer penalizes tokens based on their presence in the generated output.
    """  # 重复惩罚器根据token在生成输出中是否出现进行惩罚

    is_multiplicative: bool = True  # 标记为乘性惩罚

    def _is_required(self) -> bool:  # 检查是否有请求需要重复惩罚
        """检查是否有任何请求的重复惩罚参数不为1.0（1.0表示无惩罚）"""
        return any(
            req.sampling_params.repetition_penalty != 1.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):  # 准备重复惩罚所需的张量
        """初始化累计重复惩罚张量和重复惩罚系数张量"""
        self.cumulated_repetition_penalties = torch.ones(  # 创建累计重复惩罚张量（初始为1.0，表示无惩罚）
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),  # 形状为(请求数, 词表大小)
            dtype=torch.float32,  # 32位浮点
            device=self.orchestrator.device,  # 设备
        )
        self.repetition_penalties = (  # 创建重复惩罚系数张量
            torch.tensor(
                data=[
                    req.sampling_params.repetition_penalty
                    for req in self.orchestrator.reqs()  # 每个请求的重复惩罚系数
                ],
                dtype=torch.float32,  # 32位浮点
                device=self.orchestrator.device,  # 设备
            )
        ).unsqueeze_(1)  # 增加维度以进行广播

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):  # 累计输出token的重复惩罚
        """根据输出token更新累计重复惩罚（在出现过的token位置写入惩罚系数）"""
        self.cumulated_repetition_penalties.scatter_(  # 使用scatter_在已出现token位置写入惩罚系数
            dim=1,  # 在词表维度上散列
            index=output_ids.unsqueeze(1),  # 输出token ID索引
            src=self.repetition_penalties,  # 重复惩罚系数作为源
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:  # 应用重复惩罚到logits
        """使用缩放惩罚函数将重复惩罚应用到logits"""
        apply_scaling_penalties(logits, self.cumulated_repetition_penalties)  # 应用缩放惩罚
        return logits  # 返回修改后的logits

    def get_scaling_penalties(self) -> torch.Tensor:  # 获取缩放惩罚张量
        """返回累计重复惩罚张量，用于编排器累积乘性惩罚"""
        return self.cumulated_repetition_penalties  # 返回累计重复惩罚张量

    def _filter(self, keep_indices: torch.Tensor):  # 根据保留索引过滤张量
        """根据保留的请求索引过滤重复惩罚张量"""
        self.repetition_penalties = self.repetition_penalties[keep_indices]  # 过滤重复惩罚系数
        self.cumulated_repetition_penalties = self.cumulated_repetition_penalties[
            keep_indices  # 过滤累计重复惩罚
        ]

    def _merge(self, their: "BatchedRepetitionPenalizer"):  # 合并另一个重复惩罚器
        """将另一个重复惩罚器的张量合并到当前惩罚器"""
        self.repetition_penalties = torch.cat(
            [self.repetition_penalties, their.repetition_penalties], dim=0  # 沿批次维度拼接重复惩罚系数
        )
        self.cumulated_repetition_penalties = torch.cat(
            [self.cumulated_repetition_penalties, their.cumulated_repetition_penalties],
            dim=0,  # 沿批次维度拼接累计重复惩罚
        )

    def _teardown(self) -> None:  # 清理资源
        """清理重复惩罚器持有的张量资源"""
        for name in ("repetition_penalties", "cumulated_repetition_penalties"):  # 遍历需要清理的属性名
            if hasattr(self, name):  # 如果属性存在
                delattr(self, name)  # 删除属性以释放内存
