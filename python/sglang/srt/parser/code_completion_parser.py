# 代码补全解析器模块
# 本模块提供代码补全（FIM，Fill-in-the-Middle）功能的提示模板管理，
# 包括模板注册、查询和补全提示生成等功能。
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Completion templates."""  # 补全模板的文档字符串

import dataclasses  # 导入数据类装饰器模块
import logging  # 导入日志模块
from enum import Enum, auto  # 导入枚举类型和自动赋值功能
from typing import Optional  # 导入可选类型注解

from sglang.srt.entrypoints.openai.protocol import CompletionRequest  # 导入补全请求协议类

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器
completion_template_name: Optional[str] = None  # 当前使用的补全模板名称，初始为None


class FimPosition(Enum):
    """Position of fim middle token."""  # FIM中间标记的位置枚举

    MIDDLE = auto()  # 中间位置，fim_middle_token在suffix之前
    END = auto()  # 末尾位置，fim_middle_token在suffix之后


@dataclasses.dataclass
class CompletionTemplate:
    """A class that manages completion prompt templates. only for code completion currently."""  # 管理补全提示模板的数据类，目前仅用于代码补全

    # The name of this template  # 模板名称
    name: str

    # the fim begin token  # FIM开始标记
    fim_begin_token: str

    # The fim middle token  # FIM中间标记
    fim_middle_token: str

    # The fim end token  # FIM结束标记
    fim_end_token: str

    # The position of the fim middle token  # FIM中间标记的位置
    fim_position: FimPosition


# A global registry for all completion templates  # 所有补全模板的全局注册表
completion_templates: dict[str, CompletionTemplate] = {}


def register_completion_template(template: CompletionTemplate, override: bool = False):
    """Register a new completion template."""  # 注册新的补全模板
    if not override:  # 如果不允许覆盖
        assert (
            template.name not in completion_templates
        ), f"{template.name} has been registered."  # 断言模板名称未注册

    completion_templates[template.name] = template  # 将模板存入注册表


def completion_template_exists(template_name: str) -> bool:
    """检查指定名称的补全模板是否已注册"""  # Check if a completion template exists by name
    return template_name in completion_templates  # 返回模板名称是否在注册表中


def set_completion_template(template_name: str) -> None:
    """设置全局使用的补全模板名称"""  # Set the global completion template name
    global completion_template_name  # 声明使用全局变量
    if completion_template_name is None:  # 如果尚未设置模板名称
        completion_template_name = template_name  # 设置为指定名称


def is_completion_template_defined() -> bool:
    """检查是否已定义全局补全模板"""  # Check if a global completion template is defined
    global completion_template_name  # 声明使用全局变量
    return completion_template_name is not None  # 返回模板名称是否已设置


def generate_completion_prompt_from_request(request: CompletionRequest) -> str:
    """根据补全请求生成补全提示文本"""  # Generate completion prompt from a CompletionRequest
    global completion_template_name  # 声明使用全局变量
    if request.suffix == "":  # 如果请求中没有后缀文本
        return request.prompt  # 直接返回原始提示

    return generate_completion_prompt(  # 否则使用模板生成补全提示
        request.prompt, request.suffix, completion_template_name
    )


def generate_completion_prompt(prompt: str, suffix: str, template_name: str) -> str:
    """根据模板生成代码补全提示文本"""  # Generate a completion prompt using the specified template

    completion_template = completion_templates[template_name]  # 获取指定模板
    fim_begin_token = completion_template.fim_begin_token  # 获取FIM开始标记
    fim_middle_token = completion_template.fim_middle_token  # 获取FIM中间标记
    fim_end_token = completion_template.fim_end_token  # 获取FIM结束标记
    fim_position = completion_template.fim_position  # 获取FIM中间标记位置

    if fim_position == FimPosition.MIDDLE:  # 如果中间标记在中间位置
        prompt = f"{fim_begin_token}{prompt}{fim_middle_token}{suffix}{fim_end_token}"  # 按MIDDLE模式拼接
    elif fim_position == FimPosition.END:  # 如果中间标记在末尾位置
        prompt = f"{fim_begin_token}{prompt}{fim_end_token}{suffix}{fim_middle_token}"  # 按END模式拼接

    return prompt  # 返回生成的提示文本


register_completion_template(  # 注册DeepSeek Coder模板
    CompletionTemplate(
        name="deepseek_coder",  # 模板名称
        fim_begin_token="<｜fim▁begin｜>",  # DeepSeek FIM开始标记
        fim_middle_token="<｜fim▁hole｜>",  # DeepSeek FIM中间标记
        fim_end_token="<｜fim▁end｜>",  # DeepSeek FIM结束标记
        fim_position=FimPosition.MIDDLE,  # 使用MIDDLE位置模式
    )
)


register_completion_template(  # 注册StarCoder模板
    CompletionTemplate(
        name="star_coder",  # 模板名称
        fim_begin_token="<fim_prefix>",  # StarCoder FIM开始标记
        fim_middle_token="<fim_middle>",  # StarCoder FIM中间标记
        fim_end_token="<fim_suffix>",  # StarCoder FIM结束标记
        fim_position=FimPosition.END,  # 使用END位置模式
    )
)

register_completion_template(  # 注册Qwen Coder模板
    CompletionTemplate(
        name="qwen_coder",  # 模板名称
        fim_begin_token="<|fim_prefix|>",  # Qwen FIM开始标记
        fim_middle_token="<|fim_middle|>",  # Qwen FIM中间标记
        fim_end_token="<|fim_suffix|>",  # Qwen FIM结束标记
        fim_position=FimPosition.END,  # 使用END位置模式
    )
)
