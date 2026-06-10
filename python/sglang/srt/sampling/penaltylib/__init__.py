# 该文件是penaltylib包的初始化模块
# 导出所有批处理惩罚器类供外部使用
# 包括频率惩罚、最小新token数、存在惩罚、重复惩罚和惩罚编排器

from sglang.srt.sampling.penaltylib.frequency_penalty import BatchedFrequencyPenalizer  # 导入批处理频率惩罚器
from sglang.srt.sampling.penaltylib.min_new_tokens import BatchedMinNewTokensPenalizer  # 导入批处理最小新token数惩罚器
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator  # 导入批处理惩罚编排器
from sglang.srt.sampling.penaltylib.presence_penalty import BatchedPresencePenalizer  # 导入批处理存在惩罚器
from sglang.srt.sampling.penaltylib.repetition_penalty import BatchedRepetitionPenalizer  # 导入批处理重复惩罚器

__all__ = [  # 公开导出列表
    "BatchedFrequencyPenalizer",  # 批处理频率惩罚器
    "BatchedMinNewTokensPenalizer",  # 批处理最小新token数惩罚器
    "BatchedPresencePenalizer",  # 批处理存在惩罚器
    "BatchedPenalizerOrchestrator",  # 批处理惩罚编排器
    "BatchedRepetitionPenalizer",  # 批处理重复惩罚器
]
