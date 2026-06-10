# 模板检测工具 - 自动检测推理模式和工具调用解析器，基于聊天模板和分词器词表进行规则匹配

# Copyright 2026 SGLang Team
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
"""
Template detection utilities for auto-detecting reasoning and tool-call parsers.
模板检测工具，用于自动检测推理模式和工具调用解析器。

Provides rule-based detection of reasoning mode, reasoning parser, and tool-call
parser from chat templates and tokenizer vocabularies.
提供基于规则的检测，从聊天模板和分词器词表中检测推理模式、推理解析器和工具调用解析器。
"""

import logging  # 导入日志模块
import re  # 导入正则表达式模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Callable, Optional, Tuple  # 导入类型提示

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@dataclass(frozen=True)  # 不可变数据类
class TemplateDetectionContext:  # 模板检测上下文，封装检测所需的模板、配置和词表信息
    template: str  # 聊天模板字符串
    reasoning_config: Optional["ReasoningToggleConfig"]  # 推理开关配置
    force_reasoning: bool  # 是否强制开启推理
    vocab: set[str]  # 分词器词表集合

    def has_text(self, needle: str) -> bool:  # 检查模板中是否包含指定文本
        return needle in self.template  # 直接字符串包含检查

    def has_vocab(self, token: str) -> bool:  # 检查词表中是否存在指定token
        return token in self.vocab  # 在词表集合中查找

    def has_pattern(self, pattern: str, flags: int = 0) -> bool:  # 检查模板是否匹配指定正则模式
        return re.search(pattern, self.template, flags) is not None  # 正则搜索并判断是否匹配


@dataclass(frozen=True)  # 不可变数据类
class DetectionRule:  # 检测规则，将名称、值和判断谓词绑定在一起
    name: str  # 规则名称
    value: object  # 匹配成功时返回的值
    predicate: Callable[[TemplateDetectionContext], bool]  # 判断谓词函数


@dataclass(frozen=True)  # 不可变数据类
class ReasoningToggleConfig:  # 推理开关配置，描述推理模式的切换参数和默认状态
    toggle_param: Optional[str] = None  # 推理开关参数名（如"enable_thinking"）
    default_enabled: Optional[bool] = None  # 默认是否启用推理
    special_case: Optional[str] = None  # 特殊情况标识（如"always"、"mistral"）

    @property
    def always_on(self) -> bool:  # 判断推理模式是否始终开启
        return self.special_case == "always"  # 特殊情况为"always"时始终开启


# ---------------------------------------------------------------------------
# Reasoning mode rules (detect toggle config from template)  推理模式规则（从模板检测开关配置）
# ---------------------------------------------------------------------------

REASONING_MODE_RULES = (  # 推理模式检测规则元组
    DetectionRule(
        name="gpt_oss_channel_markers",  # GPT开源频道标记
        value=ReasoningToggleConfig(special_case="always"),  # 值为始终开启
        predicate=lambda ctx: ctx.has_text("<|channel|>"),  # 谓词：模板包含频道标记
    ),
    DetectionRule(
        name="force_reasoning_pattern",  # 强制推理模式
        value=ReasoningToggleConfig(special_case="always"),  # 值为始终开启
        predicate=lambda ctx: ctx.has_pattern(r"<\|im_start\|>assistant\\n account\\n")  # 谓词：匹配特定模式
        and not ctx.has_text("enable_thinking")  # 且不包含enable_thinking
        and not ctx.has_text("thinking"),  # 且不包含thinking
    ),
    DetectionRule(
        name="mistral_reasoning_effort",  # Mistral推理努力参数
        value=ReasoningToggleConfig(special_case="mistral"),  # 值为mistral特殊模式
        predicate=lambda ctx: ctx.has_text("reasoning_effort")  # 谓词：包含reasoning_effort
        and ctx.has_text("[THINK]"),  # 且包含[THINK]标记
    ),
    DetectionRule(
        name="explicit_enable_thinking_default_false",  # 显式enable_thinking默认关闭
        value=ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=False  # 开关参数为enable_thinking，默认关闭
        ),
        predicate=lambda ctx: ctx.has_pattern(  # 谓词：匹配Jinja模板中默认设为false的模式
            r"{%\s*if\s+not\s+enable_thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+enable_thinking\s*=\s*(?:false|False)\s*%}",
            re.DOTALL,  # DOTALL使.匹配换行符
        ),
    ),
    DetectionRule(
        name="enable_thinking_default_true",  # enable_thinking默认开启
        value=ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=True  # 开关参数为enable_thinking，默认开启
        ),
        predicate=lambda ctx: ctx.has_pattern(  # 谓词：匹配Jinja模板中默认设为true的模式
            r"{%\s*if\s+not\s+enable_thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+enable_thinking\s*=\s*(?:true|True)\s*%}",
            re.DOTALL,  # DOTALL使.匹配换行符
        )
        or ctx.has_pattern(  # 或匹配三元表达式形式
            r"set\s+enable_thinking\s*=\s*enable_thinking\s+if\s+enable_thinking\s+is\s+defined\s+else\s+(?:true|True)"
        )
        or ctx.has_pattern(  # 或匹配is defined and is false形式
            r"enable_thinking\s+is\s+defined\s+and\s+(?:enable_thinking\s+is\s+false|not\s+enable_thinking)"
        )
        or ctx.has_pattern(  # 或匹配is not defined or形式
            r"enable_thinking\s+is\s+not\s+defined\s+or\s+enable_thinking"
        )
        or ctx.has_pattern(r"namespace\([^)]*enable_thinking\s*=\s*true"),  # 或匹配namespace形式
    ),
    DetectionRule(
        name="explicit_thinking_default_false",  # 显式thinking参数默认关闭
        value=ReasoningToggleConfig(toggle_param="thinking", default_enabled=False),  # 开关参数为thinking，默认关闭
        predicate=lambda ctx: ctx.has_pattern(  # 谓词：匹配Jinja模板中thinking默认设为false
            r"{%\s*if\s+not\s+thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+thinking\s*=\s*(?:false|False)\s*%}",
            re.DOTALL,  # DOTALL使.匹配换行符
        ),
    ),
    DetectionRule(
        name="thinking_default_true",  # thinking参数默认开启
        value=ReasoningToggleConfig(toggle_param="thinking", default_enabled=True),  # 开关参数为thinking，默认开启
        predicate=lambda ctx: ctx.has_pattern(  # 谓词：匹配Jinja模板中thinking默认设为true
            r"{%\s*if\s+not\s+thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+thinking\s*=\s*(?:true|True)\s*%}",
            re.DOTALL,  # DOTALL使.匹配换行符
        )
        or ctx.has_pattern(  # 或匹配三元表达式形式
            r"set\s+thinking\s*=\s*thinking\s+if\s+thinking\s+is\s+defined\s+else\s+(?:true|True)"
        )
        or ctx.has_pattern(  # 或匹配is defined and is false形式
            r"thinking\s+is\s+defined\s+and\s+(?:thinking\s+is\s+false|not\s+thinking)"
        )
        or ctx.has_pattern(r"thinking\s+is\s+not\s+defined\s+or\s+thinking")  # 或匹配is not defined or形式
        or ctx.has_pattern(r"namespace\([^)]*thinking\s*=\s*true"),  # 或匹配namespace形式
    ),
)


# ---------------------------------------------------------------------------
# Shared predicates for model-family detection  模型族检测的共享谓词
# ---------------------------------------------------------------------------


def _is_gemma4(ctx):  # 判断是否为Gemma4模型
    return ctx.has_text("<|channel>")  # 检查是否包含频道标记


def _is_kimi(ctx):  # 判断是否为Kimi模型
    return ctx.has_text("◁think▷")  # 检查是否包含Kimi思维标记


def _is_interns1(ctx):  # 判断是否为InternS1模型
    return ctx.has_text("default_thinking_sys") and ctx.reasoning_config == (  # 包含默认思维系统提示
        ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True)  # 且推理配置匹配
    )


def _is_mistral(ctx):  # 判断是否为Mistral模型
    return (  # 返回是否为Mistral模型
        ctx.reasoning_config is not None  # 推理配置存在
        and ctx.reasoning_config.special_case == "mistral"  # 且特殊情况标记为mistral
    )


def _is_gpt_oss(ctx):  # 判断是否为GPT开源模型
    return ctx.has_text("<|channel|>")  # 检查是否包含频道标记


def _is_kimi_k2(ctx):  # 判断是否为Kimi K2模型
    return ctx.has_vocab("<|tool_calls_section_begin|>")  # 检查词表中是否有工具调用段开始标记


def _is_nemotron_3(ctx):  # 判断是否为Nemotron-3模型
    return ctx.has_text("truncate_history_thinking") and ctx.reasoning_config == (  # 包含截断历史思维文本
        ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True)  # 且推理配置匹配
    )


def _is_glm45(ctx):  # 判断是否为GLM-4.5模型
    return (  # 返回是否为GLM-4.5模型
        (
            ctx.has_text("")  # 包含思维开始标记
            or ctx.has_pattern(r"(?<!<) account")  # 或匹配account标记
            or ctx.has_pattern(r"(?<!<)/think")  # 或匹配/think标记
        )
        and ctx.has_vocab("ürü")  # 且词表包含特定token
        and ctx.reasoning_config  # 且推理配置存在
        == ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True)  # 配置为enable_thinking默认开启
        and (ctx.has_vocab("ürü") or ctx.has_vocab("ürü"))  # 且词表包含额外token
    )


def _is_xml_kv_tool_call(ctx):  # 判断是否使用XML KV风格工具调用
    # Structural signature for the GLM-4.5 / GLM-4.6 style tool-call format  GLM-4.5/GLM-4.6风格工具调用格式的结构特征
    # (`ürünameürükv\nüvürü...ür`).  （`ürünameürükv\nüvürü...ürü`格式）
    # Matches any model whose tokenizer carries `ürü` and `ürü` as  匹配分词器中包含 `ürü` 和 `ürü` 的任何模型
    # added tokens — e.g., inclusionAI/Ring-2.6, which borrows GLM's tool-call  作为添加token——例如 inclusionAI/Ring-2.6
    # format but doesn't share the `ürü` / `enable_thinking` family  借用了GLM的工具调用格式但不共享 `ürü` / `enable_thinking` 家族
    # signature checked by `_is_glm45`.  特征签名（由 `_is_glm45` 检查）。
    return ctx.has_vocab("ürü") and ctx.has_vocab("ürü")  # 检查词表中是否包含两个关键token


def _is_mimo(ctx):  # 判断是否为MiMo模型
    return ctx.reasoning_config == ReasoningToggleConfig(  # 推理配置为
        toggle_param="enable_thinking", default_enabled=False  # enable_thinking默认关闭
    )


def _is_minimax(ctx):  # 判断是否为MiniMax模型
    return ctx.has_text("<minimax:tool_call>")  # 检查是否包含MiniMax工具调用标记


def _is_minicpm5(ctx):  # 判断是否为MiniCPM-5模型
    if ctx.has_vocab("<function") and ctx.has_vocab("<param"):  # 词表中包含function和param标记
        return True  # 是MiniCPM-5
    return ctx.has_pattern(r"<function\s+name=") and ctx.has_pattern(r"<param\s+name=")  # 或模板匹配函数和参数标签


def _is_qwen3(ctx):  # 判断是否为Qwen3模型
    return ctx.reasoning_config == ReasoningToggleConfig(  # 推理配置为
        toggle_param="enable_thinking", default_enabled=True  # enable_thinking默认开启
    )


def _is_deepseek_v3(ctx):  # 判断是否为DeepSeek-V3模型
    return ctx.reasoning_config == ReasoningToggleConfig(  # 推理配置为
        toggle_param="thinking", default_enabled=False  # thinking默认关闭
    )


def _is_deepseek_r1(ctx):  # 判断是否为DeepSeek-R1模型
    return ctx.force_reasoning  # 强制推理模式


def _is_deepseek_r1_think_tags(ctx):  # 判断是否为DeepSeek-R1思维标签
    return ctx.has_text("ürü") or ctx.has_text("ürü")  # 检查是否包含思维标签


# ---------------------------------------------------------------------------
# Reasoning parser rules  推理解析器规则
# ---------------------------------------------------------------------------

REASONING_PARSER_RULES = (  # 推理解析器检测规则元组
    DetectionRule(name="gemma4", value="gemma4", predicate=_is_gemma4),  # Gemma4推理解析器
    DetectionRule(name="kimi", value="kimi", predicate=_is_kimi),  # Kimi推理解析器
    DetectionRule(name="interns1", value="interns1", predicate=_is_interns1),  # InternS1推理解析器
    DetectionRule(name="mistral", value="mistral", predicate=_is_mistral),  # Mistral推理解析器
    DetectionRule(name="gpt_oss", value="gpt-oss", predicate=_is_gpt_oss),  # GPT开源推理解析器
    DetectionRule(name="kimi_k2", value="kimi_k2", predicate=_is_kimi_k2),  # Kimi K2推理解析器
    DetectionRule(name="nemotron_3", value="nemotron_3", predicate=_is_nemotron_3),  # Nemotron-3推理解析器
    DetectionRule(name="glm45", value="glm45", predicate=_is_glm45),  # GLM-4.5推理解析器
    DetectionRule(name="mimo", value="mimo", predicate=_is_mimo),  # MiMo推理解析器
    DetectionRule(name="minimax", value="minimax", predicate=_is_minimax),  # MiniMax推理解析器
    DetectionRule(name="qwen3", value="qwen3", predicate=_is_qwen3),  # Qwen3推理解析器
    DetectionRule(name="deepseek_v3", value="deepseek-v3", predicate=_is_deepseek_v3),  # DeepSeek-V3推理解析器
    DetectionRule(
        name="deepseek_r1_force", value="deepseek-r1", predicate=_is_deepseek_r1  # DeepSeek-R1强制推理解析器
    ),
    DetectionRule(
        name="deepseek_r1_think_tags",  # DeepSeek-R1思维标签
        value="deepseek-r1",  # 值为deepseek-r1
        predicate=_is_deepseek_r1_think_tags,  # 使用思维标签检测谓词
    ),
)

# ---------------------------------------------------------------------------
# Tool-call parser rules (reuse shared predicates, different values)  工具调用解析器规则（复用共享谓词，不同值）
# ---------------------------------------------------------------------------

TOOL_CALL_PARSER_RULES = (  # 工具调用解析器检测规则元组
    DetectionRule(name="gemma4", value="gemma4", predicate=_is_gemma4),  # Gemma4工具调用解析器
    DetectionRule(name="gpt_oss", value="gpt-oss", predicate=_is_gpt_oss),  # GPT开源工具调用解析器
    DetectionRule(name="kimi_k2", value="kimi_k2", predicate=_is_kimi_k2),  # Kimi K2工具调用解析器
    DetectionRule(name="minimax", value="minimax-m2", predicate=_is_minimax),  # MiniMax M2工具调用解析器
    DetectionRule(name="interns1", value="interns1", predicate=_is_interns1),  # InternS1工具调用解析器
    DetectionRule(name="mistral", value="mistral", predicate=_is_mistral),  # Mistral工具调用解析器
    DetectionRule(name="glm45", value="glm45", predicate=_is_glm45),  # GLM-4.5工具调用解析器
    DetectionRule(name="minicpm5", value="minicpm5", predicate=_is_minicpm5),  # MiniCPM-5工具调用解析器
    DetectionRule(
        name="xml_kv_tool_call", value="glm45", predicate=_is_xml_kv_tool_call  # XML KV工具调用解析器
    ),
    DetectionRule(name="mimo", value="mimo", predicate=_is_mimo),  # MiMo工具调用解析器
    DetectionRule(name="qwen", value="qwen", predicate=_is_qwen3),  # Qwen工具调用解析器
    DetectionRule(name="deepseek_v3", value="deepseekv3", predicate=_is_deepseek_v3),  # DeepSeek-V3工具调用解析器
    DetectionRule(name="deepseek_r1", value="deepseekv3", predicate=_is_deepseek_r1),  # DeepSeek-R1工具调用解析器
)


# ---------------------------------------------------------------------------
# Detection functions  检测函数
# ---------------------------------------------------------------------------


def build_detection_context(  # 构建模板检测上下文
    template: Optional[str],  # 聊天模板字符串
    tokenizer,  # 分词器对象
    reasoning_config: Optional[ReasoningToggleConfig] = None,  # 推理配置
    force_reasoning: bool = False,  # 是否强制推理
) -> Optional[TemplateDetectionContext]:  # 返回检测上下文或None
    if template is None:  # 如果模板为None
        return None  # 返回None
    vocab = set()  # 初始化空词表集合
    if tokenizer is not None:  # 如果分词器存在
        try:
            vocab = set(tokenizer.get_vocab().keys())  # 获取分词器词表
        except Exception as e:  # 捕获异常
            logger.warning(  # 记录警告
                "Failed to load tokenizer vocab for template detection: %s. "  # 加载分词器词表失败
                "Vocab-dependent detection rules will be skipped.",  # 依赖词表的检测规则将被跳过
                e,  # 异常信息
            )
    return TemplateDetectionContext(  # 返回检测上下文对象
        template=template,  # 模板字符串
        reasoning_config=reasoning_config,  # 推理配置
        force_reasoning=force_reasoning,  # 强制推理标志
        vocab=vocab,  # 词表集合
    )


def match_rules(  # 匹配检测规则
    ctx: TemplateDetectionContext,  # 检测上下文
    rules: Tuple[DetectionRule, ...],  # 检测规则元组
    label: str,  # 规则类别标签（用于日志）
) -> Optional[str]:  # 返回匹配到的值或None
    for rule in rules:  # 遍历所有规则
        try:
            if rule.predicate(ctx):  # 如果谓词匹配成功
                return rule.value  # 返回规则对应的值
        except Exception as e:  # 捕获异常
            logger.warning(  # 记录警告
                "Detection rule '%s' for %s raised an exception: %s. Skipping.",  # 检测规则异常，跳过
                rule.name,  # 规则名称
                label,  # 类别标签
                e,  # 异常信息
                exc_info=True,  # 包含异常堆栈信息
            )
    return None  # 无匹配则返回None


def detect_reasoning_pattern(  # 检测推理模式
    template: Optional[str],  # 聊天模板字符串
) -> Tuple[bool, Optional[ReasoningToggleConfig]]:  # 返回（是否始终开启，推理配置）
    """Detect if the chat template contains reasoning/thinking patterns."""  # 检测聊天模板是否包含推理/思维模式
    if template is None:  # 如果模板为None
        return False, None  # 返回不包含推理模式

    ctx = TemplateDetectionContext(  # 创建检测上下文
        template=template,  # 模板字符串
        reasoning_config=None,  # 推理配置为None（仅检测模式）
        force_reasoning=False,  # 不强制推理
        vocab=set(),  # 空词表
    )
    for rule in REASONING_MODE_RULES:  # 遍历推理模式规则
        if rule.predicate(ctx):  # 如果谓词匹配
            return rule.value.always_on, rule.value  # 返回是否始终开启和配置

    return False, None  # 无匹配则返回不包含


def detect_reasoning_parser(  # 检测推理解析器类型
    template: Optional[str],  # 聊天模板字符串
    tokenizer,  # 分词器对象
    reasoning_config: Optional[ReasoningToggleConfig] = None,  # 推理配置
    force_reasoning: bool = False,  # 是否强制推理
) -> Optional[str]:  # 返回解析器名称或None
    """Auto-detect which reasoning parser to use from the chat template."""  # 从聊天模板自动检测应使用的推理解析器
    ctx = build_detection_context(  # 构建检测上下文
        template, tokenizer, reasoning_config, force_reasoning  # 传入模板、分词器、推理配置和强制推理标志
    )
    if ctx is None:  # 如果上下文为None
        return None  # 返回None
    return match_rules(ctx, REASONING_PARSER_RULES, "reasoning parser")  # 匹配推理解析器规则


def detect_tool_call_parser(  # 检测工具调用解析器类型
    template: Optional[str],  # 聊天模板字符串
    tokenizer,  # 分词器对象
    reasoning_config: Optional[ReasoningToggleConfig] = None,  # 推理配置
    force_reasoning: bool = False,  # 是否强制推理
) -> Optional[str]:  # 返回解析器名称或None
    """Auto-detect which tool-call parser to use from the chat template."""  # 从聊天模板自动检测应使用的工具调用解析器
    ctx = build_detection_context(  # 构建检测上下文
        template, tokenizer, reasoning_config, force_reasoning  # 传入模板、分词器、推理配置和强制推理标志
    )
    if ctx is None:  # 如果上下文为None
        return None  # 返回None
    return match_rules(ctx, TOOL_CALL_PARSER_RULES, "tool-call parser")  # 匹配工具调用解析器规则


def _resolve_auto_parser(  # 解析auto模式的解析器
    server_args,  # 服务器参数
    attr: str,  # 要设置的属性名
    ctx: TemplateDetectionContext,  # 检测上下文
    rules: Tuple[DetectionRule, ...],  # 检测规则
    label: str,  # 规则类别标签
) -> None:  # 无返回值，原地修改server_args
    """Resolve a single auto parser, updating server_args in place."""  # 解析单个auto解析器，原地更新server_args
    detected = match_rules(ctx, rules, label)  # 尝试匹配规则
    if detected:  # 如果检测到
        setattr(server_args, attr, detected)  # 设置server_args的属性为检测到的值
        logger.info(  # 记录信息
            f"Auto-detected --{attr.replace('_', '-')} as '{detected}' from chat template"  # 自动检测到解析器
        )
    else:  # 如果未检测到
        logger.warning(  # 记录警告
            f"--{attr.replace('_', '-')}=auto specified but could not detect "  # 指定了auto但无法检测
            f"{label} from chat template. Disabling {label}."  # 从聊天模板，禁用该解析器
        )
        setattr(server_args, attr, None)  # 设置为None以禁用


def resolve_auto_parsers(server_args) -> None:  # 解析所有auto模式的解析器
    """Resolve --reasoning-parser=auto and --tool-call-parser=auto before scheduler.
    在调度器启动前解析 --reasoning-parser=auto 和 --tool-call-parser=auto

    This performs a lightweight tokenizer load to detect parsers from the chat
    template. Called early in engine init before scheduler subprocesses are spawned.
    执行轻量级分词器加载以从聊天模板检测解析器。在引擎初始化早期、调度器子进程派生之前调用。
    """
    needs_reasoning = server_args.reasoning_parser == "auto"  # 是否需要检测推理解析器
    needs_tool_call = server_args.tool_call_parser == "auto"  # 是否需要检测工具调用解析器

    if not needs_reasoning and not needs_tool_call:  # 都不需要则直接返回
        return

    from sglang.srt.utils.hf_transformers_utils import get_tokenizer  # 延迟导入分词器工具

    try:
        tokenizer = get_tokenizer(  # 加载分词器
            server_args.model_path,  # 模型路径
            trust_remote_code=server_args.trust_remote_code,  # 是否信任远程代码
        )
        template = getattr(tokenizer, "chat_template", None)  # 获取聊天模板
    except Exception as e:  # 捕获异常
        logger.warning(f"Failed to load tokenizer for auto-detection: {e}")  # 加载分词器失败
        if needs_reasoning:  # 如果需要推理解析器
            logger.warning(  # 记录警告
                "--reasoning-parser=auto specified but could not detect "  # 指定了auto但无法检测
                "reasoning parser from chat template. Disabling reasoning parser."  # 推理解析器，禁用
            )
            server_args.reasoning_parser = None  # 设置为None
        if needs_tool_call:  # 如果需要工具调用解析器
            logger.warning(  # 记录警告
                "--tool-call-parser=auto specified but could not detect "  # 指定了auto但无法检测
                "tool-call parser from chat template. Disabling tool-call parser."  # 工具调用解析器，禁用
            )
            server_args.tool_call_parser = None  # 设置为None
        return

    force_reasoning, reasoning_config = detect_reasoning_pattern(template)  # 检测推理模式
    ctx = build_detection_context(  # 构建检测上下文
        template, tokenizer, reasoning_config, force_reasoning  # 传入模板、分词器、推理配置和强制推理标志
    )
    if ctx is None:  # 如果上下文为None
        return

    if needs_reasoning:  # 如果需要推理解析器
        _resolve_auto_parser(  # 解析推理解析器
            server_args,  # 服务器参数
            "reasoning_parser",  # 属性名
            ctx,  # 检测上下文
            REASONING_PARSER_RULES,  # 推理解析器规则
            "reasoning parser",  # 标签
        )

    if needs_tool_call:  # 如果需要工具调用解析器
        _resolve_auto_parser(  # 解析工具调用解析器
            server_args,  # 服务器参数
            "tool_call_parser",  # 属性名
            ctx,  # 检测上下文
            TOOL_CALL_PARSER_RULES,  # 工具调用解析器规则
            "tool-call parser",  # 标签
        )
