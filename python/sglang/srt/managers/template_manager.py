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
# 模板管理器：集中管理聊天模板（chat template）和代码补全模板（completion template）。
# 提供统一的模板加载、检测和初始化接口，消除全局状态，提升模块化程度。
"""
Centralized template management for chat templates and completion templates.

This module provides a unified interface for managing both chat conversation templates
and code completion templates, eliminating global state and improving modularity.
"""

import json
import logging
import os
from typing import Dict, Optional

from sglang.srt.managers.template_detection import (
    REASONING_PARSER_RULES,
    TOOL_CALL_PARSER_RULES,
    ReasoningToggleConfig,
    build_detection_context,
    detect_reasoning_pattern,
    match_rules,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.parser.code_completion_parser import (
    CompletionTemplate,
    FimPosition,
    completion_template_exists,
    register_completion_template,
    set_completion_template,
)
from sglang.srt.parser.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    get_conv_template_by_model_path,
    register_conv_template,
)
from sglang.srt.parser.jinja_template_utils import detect_jinja_template_content_format

logger = logging.getLogger(__name__)


class TemplateManager:
    # 模板管理器类：集中管理聊天模板和代码补全模板，封装所有模板相关的状态和操作，
    # 消除全局变量，提供清晰的模板管理接口。
    """
    Centralized manager for chat and completion templates.

    This class encapsulates all template-related state and operations,
    eliminating the need for global variables and providing a clean
    interface for template management.
    """

    def __init__(self):
        # 当前聊天模板名称
        self._chat_template_name: Optional[str] = None
        # 当前代码补全模板名称
        self._completion_template_name: Optional[str] = None
        # Jinja 模板内容格式（'string' 或 'openai'）
        self._jinja_template_content_format: Optional[str] = "openai"
        # 是否强制启用推理/思考模式
        self._force_reasoning: bool = False
        # 推理开关配置，从聊天模板中推断得到
        self._reasoning_config: Optional[ReasoningToggleConfig] = None
        # 自动检测到的推理解析器名称
        self._suggested_reasoning_parser: Optional[str] = None
        # 自动检测到的工具调用解析器名称
        self._suggested_tool_call_parser: Optional[str] = None

    @property
    def chat_template_name(self) -> Optional[str]:
        """Get the current chat template name."""
        # 获取当前聊天模板名称
        return self._chat_template_name

    @property
    def completion_template_name(self) -> Optional[str]:
        """Get the current completion template name."""
        # 获取当前代码补全模板名称
        return self._completion_template_name

    @property
    def jinja_template_content_format(self) -> Optional[str]:
        """Get the detected template content format ('string' or 'openai' or None)."""
        # 获取检测到的模板内容格式
        return self._jinja_template_content_format

    @property
    def force_reasoning(self) -> bool:
        """
        Check if the current chat template enforces reasoning/thinking.

        Returns:
            True if the template contains reasoning patterns like HLT tags
        """
        # 检查当前聊天模板是否强制要求推理/思考模式
        return self._force_reasoning

    @property
    def reasoning_config(self) -> Optional[ReasoningToggleConfig]:
        """Get the reasoning toggle config inferred from chat template."""
        # 获取从聊天模板推断出的推理开关配置
        return self._reasoning_config

    @property
    def suggested_reasoning_parser(self) -> Optional[str]:
        """Get the auto-detected reasoning parser name, or None."""
        # 获取自动检测到的推理解析器名称
        return self._suggested_reasoning_parser

    @property
    def suggested_tool_call_parser(self) -> Optional[str]:
        """Get the auto-detected tool-call parser name, or None."""
        # 获取自动检测到的工具调用解析器名称
        return self._suggested_tool_call_parser

    def _run_template_detection(self, template, tokenizer) -> None:
        # 对模板运行推理模式检测和解析器匹配，设置强制推理标志、推理配置、建议的推理解析器和工具调用解析器。
        """Run reasoning pattern and parser detection on a template."""
        # 检测模板中的推理模式，返回是否强制推理和推理配置
        self._force_reasoning, self._reasoning_config = detect_reasoning_pattern(
            template
        )
        # Build context once, reuse for both parser detections (avoids
        # duplicate tokenizer.get_vocab() calls).
        # 构建检测上下文，复用于推理解析器和工具调用解析器的匹配，避免重复调用 get_vocab()
        ctx = build_detection_context(
            template, tokenizer, self._reasoning_config, self._force_reasoning
        )
        if ctx is None:
            return
        # 匹配推理解析器规则
        self._suggested_reasoning_parser = match_rules(
            ctx, REASONING_PARSER_RULES, "reasoning parser"
        )
        # 匹配工具调用解析器规则
        self._suggested_tool_call_parser = match_rules(
            ctx, TOOL_CALL_PARSER_RULES, "tool-call parser"
        )

    def load_chat_template(
        self,
        tokenizer_manager: TokenizerManager,
        chat_template_arg: Optional[str],
        model_path: str,
    ) -> None:
        # 加载聊天模板：根据参数从内置模板、文件路径或自动检测中加载聊天模板。
        """
        Load a chat template from various sources.

        Args:
            tokenizer_manager: The tokenizer manager instance
            chat_template_arg: Template name, file path, or None to auto-detect
            model_path: Path to the model
        """
        if chat_template_arg:
            # 如果显式指定了聊天模板参数，则加载指定的模板
            self._load_explicit_chat_template(tokenizer_manager, chat_template_arg)
        else:
            # Guess chat template from model path
            # 从模型路径推断聊天模板名称
            self.guess_chat_template_from_model_path(model_path)

            # If no pre-defined template was found, fallback to HuggingFace template
            # 如果没有找到预定义模板，则回退到 HuggingFace 模板
            if self._chat_template_name is None:
                # Try HuggingFace template first
                # 首先尝试获取 HuggingFace 聊天模板
                hf_template = self._resolve_hf_chat_template(tokenizer_manager)
                if hf_template:
                    # override the chat template
                    # 覆盖 tokenizer 的聊天模板
                    if tokenizer_manager.tokenizer:
                        tokenizer_manager.tokenizer.chat_template = hf_template
                    # 检测 Jinja 模板的内容格式
                    self._jinja_template_content_format = (
                        detect_jinja_template_content_format(hf_template)
                    )
                    logger.info(
                        f"Using default HuggingFace chat template with detected content format: {self._jinja_template_content_format}"
                    )
                else:
                    # Default to string content format if no template was found
                    # 如果没有找到任何模板，默认使用 'string' 内容格式
                    self._jinja_template_content_format = "string"
                    logger.info(
                        "No chat template found, defaulting to 'string' content format"
                    )

        # Detect reasoning pattern and suggest parser from chat template
        # 从聊天模板中检测推理模式并建议对应的解析器
        if tokenizer_manager.tokenizer:
            template = tokenizer_manager.tokenizer.chat_template
            self._run_template_detection(template, tokenizer_manager.tokenizer)
            parts = []
            if self._reasoning_config:
                parts.append(f"reasoning_config={self._reasoning_config}")
            if self._suggested_reasoning_parser:
                parts.append(f"reasoning_parser={self._suggested_reasoning_parser}")
            if self._suggested_tool_call_parser:
                parts.append(f"tool_call_parser={self._suggested_tool_call_parser}")
            if parts:
                logger.info(f"Auto-detected template features: {', '.join(parts)}")

    def _load_explicit_chat_template(
        self, tokenizer_manager: TokenizerManager, chat_template_arg: str
    ) -> None:
        # 加载显式指定的聊天模板：支持内置模板名、Jinja 文件和 JSON 文件。
        """Load explicitly specified chat template."""
        logger.info(f"Loading chat template from argument: {chat_template_arg}")

        # 如果是内置模板名称，直接使用
        if chat_template_exists(chat_template_arg):
            self._chat_template_name = chat_template_arg
            return

        # 如果不是内置名称也不是有效文件路径，则报错
        if not os.path.exists(chat_template_arg):
            raise RuntimeError(
                f"Chat template {chat_template_arg} is not a built-in template name "
                "or a valid chat template file path."
            )

        # 根据文件扩展名选择加载方式
        if chat_template_arg.endswith(".jinja"):
            self._load_jinja_template(tokenizer_manager, chat_template_arg)
        else:
            self._load_json_chat_template(chat_template_arg)

    def guess_chat_template_from_model_path(self, model_path: str) -> None:
        # 从模型路径推断聊天模板名称，如果找到匹配的内置模板则设置。
        """
        Infer chat template name from model path.

        Args:
            model_path: Path to the model
        """
        template_name = get_conv_template_by_model_path(model_path)
        if template_name is not None:
            logger.info(f"Inferred chat template from model path: {template_name}")
            self._chat_template_name = template_name

    def load_completion_template(self, completion_template_arg: str) -> None:
        # 加载代码补全模板：支持内置模板名称和 JSON 文件路径。
        """
        Load completion template for code completion.

        Args:
            completion_template_arg: Template name or file path
        """
        logger.info(f"Loading completion template: {completion_template_arg}")

        if not completion_template_exists(completion_template_arg):
            # 不是内置模板，尝试从文件路径加载
            if not os.path.exists(completion_template_arg):
                raise RuntimeError(
                    f"Completion template {completion_template_arg} is not a built-in template name "
                    "or a valid completion template file path."
                )

            # 从 JSON 文件加载补全模板
            self._load_json_completion_template(completion_template_arg)
        else:
            # 使用内置补全模板
            self._completion_template_name = completion_template_arg

        # 将补全模板设置为全局当前使用的模板
        set_completion_template(self._completion_template_name)

    def initialize_templates(
        self,
        tokenizer_manager: TokenizerManager,
        model_path: str,
        chat_template: Optional[str] = None,
        completion_template: Optional[str] = None,
    ) -> None:
        # 初始化所有模板：依次加载聊天模板和代码补全模板。
        """
        Initialize all templates based on provided configuration.

        Args:
            tokenizer_manager: The tokenizer manager instance
            model_path: Path to the model
            chat_template: Optional chat template name/path
            completion_template: Optional completion template name/path
        """
        # Load chat template
        # 加载聊天模板
        self.load_chat_template(tokenizer_manager, chat_template, model_path)

        # Load completion template
        # 加载代码补全模板（如果指定了的话）
        if completion_template:
            self.load_completion_template(completion_template)

    def _load_jinja_template(
        self, tokenizer_manager: TokenizerManager, template_path: str
    ) -> None:
        # 从 Jinja 文件加载聊天模板，读取文件内容并设置到 tokenizer 上。
        """Load a Jinja template file."""
        with open(template_path, "r") as f:
            chat_template = "".join(f.readlines()).strip("\n")
        # 将转义的换行符替换为实际换行符，并设置到 tokenizer
        tokenizer_manager.tokenizer.chat_template = chat_template.replace("\\n", "\n")
        # Jinja 模板不使用内置模板名称
        self._chat_template_name = None
        # Detect content format from the loaded template
        # 检测加载的模板的内容格式
        self._jinja_template_content_format = detect_jinja_template_content_format(
            chat_template
        )
        logger.info(
            f"Detected user specified Jinja chat template with content format: {self._jinja_template_content_format}"
        )

    def _load_json_chat_template(self, template_path: str) -> None:
        # 从 JSON 文件加载聊天模板，解析 JSON 内容并注册为新的会话模板。
        """Load a JSON chat template file."""
        assert template_path.endswith(
            ".json"
        ), "unrecognized format of chat template file"

        with open(template_path, "r") as filep:
            template = json.load(filep)
            try:
                # 从字符串获取分隔符风格枚举值
                sep_style = SeparatorStyle[template["sep_style"]]
            except KeyError:
                raise ValueError(
                    f"Unknown separator style: {template['sep_style']}"
                ) from None

            # 注册新的会话模板到全局模板注册表
            register_conv_template(
                Conversation(
                    name=template["name"],
                    system_template=template["system"] + "\n{system_message}",
                    system_message=template.get("system_message", ""),
                    roles=(template["user"], template["assistant"]),
                    sep_style=sep_style,
                    sep=template.get("sep", "\n"),
                    stop_str=template["stop_str"],
                ),
                override=True,
            )
        # 设置当前使用的聊天模板名称
        self._chat_template_name = template["name"]

    def _load_json_completion_template(self, template_path: str) -> None:
        # 从 JSON 文件加载代码补全模板，解析 JSON 内容并注册为新的补全模板。
        """Load a JSON completion template file."""
        assert template_path.endswith(
            ".json"
        ), "unrecognized format of completion template file"

        with open(template_path, "r") as filep:
            template = json.load(filep)
            try:
                # 从字符串获取 FIM 位置枚举值
                fim_position = FimPosition[template["fim_position"]]
            except KeyError:
                raise ValueError(
                    f"Unknown fim position: {template['fim_position']}"
                ) from None

            # 注册新的代码补全模板到全局模板注册表
            register_completion_template(
                CompletionTemplate(
                    name=template["name"],
                    fim_begin_token=template["fim_begin_token"],
                    fim_middle_token=template["fim_middle_token"],
                    fim_end_token=template["fim_end_token"],
                    fim_position=fim_position,
                ),
                override=True,
            )
        # 设置当前使用的补全模板名称
        self._completion_template_name = template["name"]

    def _resolve_hf_chat_template(
        self, tokenizer_manager: TokenizerManager
    ) -> Optional[str]:
        # 从 HuggingFace tokenizer/processor 中解析聊天模板，支持单模板和字典形式的多命名模板。
        try:
            # Try (mm-)processor first, then tokenizer
            # 优先从多模态处理器获取聊天模板，其次从 tokenizer 获取
            template = (
                getattr(tokenizer_manager.processor, "chat_template", None)
                if tokenizer_manager.processor
                else None
            ) or (
                getattr(tokenizer_manager.tokenizer, "chat_template", None)
                if tokenizer_manager.tokenizer
                else None
            )

            if template is None:
                logger.warning("No HuggingFace chat template found")
                return None

            # Handle dict templates (multiple named templates)
            # 处理字典形式的多命名模板
            if isinstance(template, dict):
                return self._select_named_template(template, tokenizer_manager)

            # Single string template
            # 单个字符串模板，直接返回
            return template

        except Exception as e:
            logger.warning(f"Error getting chat template: {e}")
            return None

    def _select_named_template(
        self, templates: Dict[str, str], tokenizer_manager: TokenizerManager
    ) -> str:
        # 从多个命名模板中选择一个：优先使用用户指定的模板名称，否则回退到第一个可用模板。
        if not templates:
            raise ValueError("Empty templates dict provided")

        available_names = list(templates.keys())
        logger.info(f"Multiple HuggingFace chat templates available: {available_names}")

        # Use specified template if provided
        # 如果用户指定了模板名称，则使用指定的模板
        if preferred_name := tokenizer_manager.server_args.hf_chat_template_name:
            if preferred_name not in templates:
                raise ValueError(
                    f"Specified template '{preferred_name}' not found. "
                    f"Available templates: {available_names}"
                )
            logger.info(f"Using specified chat template: '{preferred_name}'")
            return templates[preferred_name]

        # Fallback: Use first available template
        # 回退方案：使用第一个可用的模板
        first_name = available_names[0]
        logger.info(f"Using first available template: '{first_name}'")
        return templates[first_name]
