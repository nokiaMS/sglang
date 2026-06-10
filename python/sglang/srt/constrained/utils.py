# 本文件提供约束解码相关的工具函数，如判断结构化标签是否为旧版格式。
from typing import Dict


# 判断给定的对象是否为旧版结构化标签格式
def is_legacy_structural_tag(obj: Dict) -> bool:
    # test whether an object is a legacy structural tag
    # 测试对象是否为旧版结构化标签
    # see `StructuralTagResponseFormat` at `sglang.srt.entrypoints.openai.protocol`
    if obj.get("structures", None) is not None:
        assert obj.get("triggers", None) is not None
        return True
    else:
        assert obj.get("format", None) is not None
        return False
