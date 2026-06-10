# 本文件定义了语法引导约束解码后端的基类，包括语法对象基类、语法后端基类以及语法后端的注册与创建机制。
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
"""The baseclass of a backend for grammar-guided constrained decoding."""

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# 语法统计信息数据类，记录语法编译时间、缓存命中、超时次数等信息
@dataclass
class GrammarStats:
    compilation_time: Optional[float] = None
    schema_count: Optional[int] = None
    ebnf_size: Optional[int] = None
    is_cache_hit: bool = False
    is_grammar_aborted: bool = False
    tree_traversal_time: List[float] = field(default_factory=list)
    dispatch_type: Optional[str] = None
    num_timeout: int = 0


# 语法对象基类，定义了约束解码中语法匹配器的通用接口
class BaseGrammarObject:

    def __init__(self):
        self._finished = False
        self.grammar_stats = None
        self.current_token = None

    # 初始化推理模式（可选）
    def maybe_init_reasoning(self, reasoning: bool):
        pass

    # 接受一个token到语法匹配器中
    def accept_token(self, token: int) -> None:
        """
        Accept a token in the grammar.
        """
        raise NotImplementedError()

    # 回滚k个已接受的token
    def rollback(self, k: int):
        raise NotImplementedError()

    # 判断语法匹配是否已终止
    def is_terminated(self):
        return False

    # 分配词表掩码张量
    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        raise NotImplementedError()

    # 填充词表掩码中指定批次的合法token位
    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        raise NotImplementedError()

    # 将词表掩码移动到指定设备
    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        raise NotImplementedError()

    # 将词表掩码应用到logits上，屏蔽不合法token
    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        raise NotImplementedError()

    # 复制当前语法对象
    def copy(self) -> "BaseGrammarObject":
        return self

    @property
    def finished(self):
        return self._finished

    @finished.setter
    def finished(self, finished):
        self._finished = finished

    # 尝试进行语法跳跃前移优化
    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        """
        Try to jump forward in the grammar.

        Returns:
            A jump forward helper which may be used in `jump_forward_str_state`.
            None if the jump forward is not possible.
        """
        raise NotImplementedError()

    # 根据跳跃前移辅助信息获取跳跃字符串和下一状态
    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        """
        Jump forward for the grammar.

        Returns:
            A tuple of the jump forward string and the next state of the grammar
            (which can be used in `jump_and_retokenize` if needed).
        """
        raise NotImplementedError()

    # 执行跳跃前移后重新分词并更新语法状态
    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ) -> None:
        """
        Jump forward occurs, and update the grammar state if needed.
        """
        raise NotImplementedError()


# 无效语法对象，表示语法编译失败，携带原始错误信息
class InvalidGrammarObject(BaseGrammarObject):
    """Represents a grammar that failed to compile, carrying the original error message."""

    def __init__(self, error_message: str = "Unknown grammar error"):
        super().__init__()
        self.error_message = error_message

    def __repr__(self):
        return f"InvalidGrammarObject(error_message={self.error_message!r})"


# 语法后端基类，提供语法的缓存、分发和线程池编译机制
class BaseGrammarBackend:
    _enable_strict_thinking: bool = False

    def __init__(self):
        # 线程池用于异步编译语法
        self.executor = ThreadPoolExecutor()
        # 语法缓存字典，键为(类型, 字符串)元组
        self.cache: Dict[Tuple[str, str], BaseGrammarObject] = {}

    # 处理不支持的语法类型，返回无效语法对象
    def _not_supported(self, key_type: str, key_string: str) -> BaseGrammarObject:
        logger.warning(f"Skip unsupported {key_type=}, {key_string=}")
        return InvalidGrammarObject()

    @property
    def enable_strict_thinking(self):
        return self._enable_strict_thinking

    # 是否支持token过滤功能
    @property
    def is_support_token_filter(self):
        return False

    # 在词表掩码中设置或清除特定token的过滤（默认无操作）
    def set_token_filter(
        self, vocab_mask, token_ids, batch_idx, is_allowed=True, reset_vocab_mask=True
    ):
        """Set or clear specific tokens in the vocab mask. No-op by default."""
        pass

    # 创建仅用于严格token过滤的语法对象（默认返回None）
    def init_strict_reasoning_grammar(self, reasoning: bool):
        """Create a grammar object for strict token filtering only. Returns None by default."""
        return None

    # 不应被调用的回退分发函数
    def dispatch_fallback(self, key_type: str, key_string: str) -> BaseGrammarObject:
        """
        This function should not be reached in any case.
        """
        raise ValueError(f"Invalid key_type: {key_type}={key_string}")

    # 分发JSON语法编译
    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("json", key_string)

    # 分发正则表达式语法编译
    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("regex", key_string)

    # 分发EBNF语法编译
    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("ebnf", key_string)

    # 分发结构化标签语法编译
    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("structural_tag", key_string)

    # 根据键类型和字符串初始化并分发语法编译
    def _init_value_dispatch(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> BaseGrammarObject:
        s = time.perf_counter()
        key_type, key_string = key
        if key_type == "json":
            grammar = self.dispatch_json(key_string)
        elif key_type == "regex":
            grammar = self.dispatch_regex(key_string)
        elif key_type == "ebnf":
            grammar = self.dispatch_ebnf(key_string)
        elif key_type == "structural_tag":
            grammar = self.dispatch_structural_tag(key_string)
        else:
            grammar = self.dispatch_fallback(key_type, key_string)

        if grammar is not None and grammar.grammar_stats is not None:
            grammar.grammar_stats.compilation_time = time.perf_counter() - s
        return grammar

    # 获取缓存的语法对象或提交异步编译任务
    def get_cached_or_future_value(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> Tuple[BaseGrammarObject | Future[BaseGrammarObject], bool]:
        value = self.cache.get(key)
        if value:
            copied_value = value.copy()
            copied_value.maybe_init_reasoning(require_reasoning)
            return copied_value, True
        # 缓存未命中，提交异步编译任务
        value = self.executor.submit(self._init_value_dispatch, key, require_reasoning)
        return value, False

    # 设置语法缓存
    def set_cache(self, key: Tuple[str, str], value: BaseGrammarObject):
        self.cache[key] = value

    # 重置语法缓存
    def reset(self):
        self.cache.clear()


# 语法后端注册表
GRAMMAR_BACKEND_REGISTRY = {}


# 注册自定义语法后端
def register_grammar_backend(name, init_func):
    GRAMMAR_BACKEND_REGISTRY[name] = init_func


# 根据服务器参数创建语法后端实例
def create_grammar_backend(
    server_args: ServerArgs,
    tokenizer,
    vocab_size: int,
    eos_token_ids: Optional[set] = None,
    think_end_id: Optional[int] = None,
) -> Optional[BaseGrammarBackend]:
    name = server_args.grammar_backend

    # Custom grammar backend has the highest priority
    # 自定义语法后端具有最高优先级
    if name in GRAMMAR_BACKEND_REGISTRY:
        return GRAMMAR_BACKEND_REGISTRY[name](
            server_args, tokenizer, vocab_size, eos_token_ids
        )

    # Default grammar backends
    # 默认语法后端
    if name == "outlines":
        from sglang.srt.constrained.outlines_backend import OutlinesGrammarBackend

        grammar_backend = OutlinesGrammarBackend(
            tokenizer,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
        )
    elif name == "xgrammar":
        from sglang.srt.constrained.xgrammar_backend import (
            TokenizerNotSupportedError,
            XGrammarGrammarBackend,
        )

        # Convert Set[int] to List[int] if needed
        # 将Set[int]转换为List[int]
        eos_list = list(eos_token_ids) if eos_token_ids else None

        try:
            grammar_backend = XGrammarGrammarBackend(
                tokenizer,
                vocab_size=vocab_size,
                model_eos_token_ids=eos_list,
                any_whitespace=not server_args.constrained_json_disable_any_whitespace,
            )
        except TokenizerNotSupportedError as e:
            if server_args.enable_strict_thinking:
                raise ValueError(
                    f"--enable-strict-thinking requires a grammar backend with "
                    f"token filtering support, but XGrammar failed to initialize: "
                    f"{e}. Cannot fall back to grammar_backend='none' with strict "
                    f"thinking enabled."
                ) from e
            logger.warning(
                f"Grammar backend disabled because tokenizer is not supported by XGrammar: {e}. "
                "Falling back to grammar_backend='none'. "
                "Structured outputs (JSON schema, regex, EBNF) will not be available."
            )
            server_args.grammar_backend = "none"
            return None
    elif name == "llguidance":
        from sglang.srt.constrained.llguidance_backend import GuidanceBackend

        grammar_backend = GuidanceBackend(
            tokenizer=tokenizer,
            any_whitespace=not server_args.constrained_json_disable_any_whitespace,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
        )
    elif name == "none":
        if server_args.enable_strict_thinking:
            raise ValueError(
                "--enable-strict-thinking requires a grammar backend that supports "
                "token filtering, but grammar_backend='none' was specified. Use "
                "--grammar-backend xgrammar or another backend that supports token "
                "filtering."
            )
        return None
    else:
        raise ValueError(f"Invalid grammar backend: {name}")

    # 如果启用了推理解析器，则用推理语法后端包装原始后端
    if server_args.reasoning_parser and think_end_id is not None:
        from sglang.srt.constrained.reasoner_grammar_backend import (
            ReasonerGrammarBackend,
        )

        reasoning_parser = ReasoningParser(
            model_type=server_args.reasoning_parser, stream_reasoning=False
        )

        grammar_backend = ReasonerGrammarBackend(
            grammar_backend,
            reasoning_parser,
            tokenizer,
            enable_strict_thinking=server_args.enable_strict_thinking,
        )

    return grammar_backend
