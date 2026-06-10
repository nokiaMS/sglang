# InternS2Preview 视觉语言多模态条件生成模型
# 基于 Qwen3.5 MoE 架构的简单封装
# Models  # 模型导入
from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration  # 从 Qwen3.5 导入 MoE 条件生成模型


class InternS2PreviewForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):  # InternS2Preview 条件生成模型，继承自 Qwen3.5 MoE 条件生成模型
    """InternS2Preview Vision-Language Model."""  # InternS2Preview 视觉语言模型


EntryClass = [InternS2PreviewForConditionalGeneration]  # 入口类列表，用于模型注册
