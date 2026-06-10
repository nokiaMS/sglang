# 本文件实现了基于xgrammar的约束解码后端，包括xgrammar语法对象、语法编译器后端以及JSON/EBNF/正则/结构化标签的分发编译。
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
"""Constrained decoding with xgrammar backend."""

import dataclasses
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from xgrammar import (
    CompiledGrammar,
    GrammarCompiler,
    GrammarMatcher,
    StructuralTag,
    StructuralTagItem,
    TokenizerInfo,
    allocate_token_bitmask,
)

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    GrammarStats,
    InvalidGrammarObject,
)
from sglang.srt.constrained.utils import is_legacy_structural_tag
from sglang.srt.utils import is_hip

_is_hip = is_hip()

if _is_hip:
    from sgl_kernel import apply_token_bitmask_inplace_cuda
else:
    from sglang.srt.constrained.triton_ops.bitmask_ops import (
        apply_token_bitmask_inplace_triton,
    )

from sglang.srt.constrained.torch_ops.token_filter_torch_ops import (
    set_token_filter_torch,
)
from sglang.srt.constrained.triton_ops.token_filter_ops import set_token_filter_triton

logger = logging.getLogger(__name__)
# 最大回滚token数量
MAX_ROLLBACK_TOKENS = 200


# xgrammar语法对象，封装GrammarMatcher并提供词表掩码操作接口
class XGrammarGrammar(BaseGrammarObject):

    def __init__(
        self,
        matcher: GrammarMatcher,
        vocab_size: int,
        ctx: CompiledGrammar,
        override_stop_tokens: Optional[Union[List[int], int]],
        key_string: Optional[str] = None,
        grammar_stats: Optional[GrammarStats] = GrammarStats(),
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.override_stop_tokens = override_stop_tokens
        # 已接受的token列表
        self.accepted_tokens = []
        self.key_string = key_string
        self.grammar_stats = grammar_stats

    # 接受一个token，若语法匹配器拒绝则抛出异常
    def accept_token(self, token: int):
        if not self.is_terminated():
            self.current_token = token
            accepted = self.matcher.accept_token(token)
            if not accepted:
                # log for debugging
                raise ValueError(
                    f"Tokens not accepted: {token}\n"
                    f"Accepted tokens: {self.accepted_tokens}\n"
                    f"Key string: {self.key_string}"
                )
            else:
                self.accepted_tokens.append(token)

    # 回滚k个已接受的token
    def rollback(self, k: int):
        self.matcher.rollback(k)
        self.accepted_tokens = self.accepted_tokens[:-k]

    # 判断语法匹配是否已终止
    def is_terminated(self):
        return self.matcher.is_terminated()

    # 分配词表位掩码张量
    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, vocab_size)

    # 填充词表位掩码中指定批次的下一个合法token位
    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(vocab_mask, idx)

    # 将词表位掩码异步移动到指定设备
    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    # 将词表位掩码应用到logits上，根据设备类型选择不同的实现
    def apply_vocab_mask(self, logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if logits.device.type in {"cuda", "xpu", "musa"}:
            if _is_hip:
                apply_token_bitmask_inplace_cuda(logits, vocab_mask)
            else:
                apply_token_bitmask_inplace_triton(logits, vocab_mask)
        elif logits.device.type == "npu":
            import sgl_kernel_npu  # noqa: F401

            torch.ops.npu.apply_token_bitmask(logits, vocab_mask)
        else:
            raise RuntimeError(f"Unsupported device: {logits.device.type}")

    # 复制当前语法对象，创建新的GrammarMatcher实例
    def copy(self):
        matcher = GrammarMatcher(
            self.ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        if grammar_stats := self.grammar_stats:
            grammar_stats = dataclasses.replace(
                grammar_stats, is_cache_hit=True, tree_traversal_time=[]
            )
        return XGrammarGrammar(
            matcher,
            self.vocab_size,
            self.ctx,
            self.override_stop_tokens,
            self.key_string,
            grammar_stats,
        )

    # 尝试查找可跳跃前移的字符串
    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        s = self.matcher.find_jump_forward_string()
        if s:
            return [], s
        return None

    # 根据跳跃前移辅助信息返回跳跃字符串和状态
    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        _, data = helper
        return data, -1

    # 跳跃前移后重新分词并更新语法匹配器状态
    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        k = 0
        for i, old_id in enumerate(old_output_ids):
            if old_id == new_output_ids[i]:
                k = i + 1
            else:
                break

        # rollback to the last token that is the same
        # 回滚到最后一个相同的token
        if k < len(old_output_ids):
            self.matcher.rollback(len(old_output_ids) - k)

        # 接受重新分词后的新token
        for i in range(k, len(new_output_ids)):
            if not self.matcher.accept_token(new_output_ids[i]):
                raise ValueError(
                    f"Token not accepted during retokenization: {new_output_ids[i]} "
                    f"at position {i}\n"
                    f"Old output IDs: {old_output_ids}\n"
                    f"New output IDs: {new_output_ids}\n"
                    f"Key string: {self.key_string}"
                )

    def __repr__(self):
        return f"XGrammarGrammar({self.key_string=}, {self.accepted_tokens=}, {self.current_token=})"


# 分词器不支持异常，当xgrammar不支持某分词器时抛出
class TokenizerNotSupportedError(Exception):
    """Raised when tokenizer is not supported by XGrammar backend."""

    pass


# xgrammar语法后端，提供JSON/EBNF/正则/结构化标签的编译和词表掩码操作
class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        model_eos_token_ids: Optional[List[int]] = None,
        any_whitespace: bool = True,
    ):
        super().__init__()

        if hasattr(tokenizer, "init_xgrammar"):
            # For special tokenizer
            # 特殊分词器的初始化路径
            tokenizer_info, override_stop_tokens = tokenizer.init_xgrammar()

            if tokenizer_info is None:
                # Not supported tokenizer
                # 不支持的分词器类型
                raise TokenizerNotSupportedError(
                    f"Tokenizer type {type(tokenizer).__name__} is not supported by XGrammar"
                )
        else:
            # Create TokenizerInfo with model's EOS tokens as the authoritative stop tokens
            # This ensures consistency between what the model considers EOS and what XGrammar uses
            # 使用模型的EOS token作为权威停止token创建TokenizerInfo
            # 确保模型认为的EOS和XGrammar使用的一致
            try:
                tokenizer_info = TokenizerInfo.from_huggingface(
                    tokenizer, vocab_size=vocab_size, stop_token_ids=model_eos_token_ids
                )
                override_stop_tokens = None
            except Exception as e:
                raise TokenizerNotSupportedError(
                    f"Failed to create XGrammar TokenizerInfo from tokenizer: {e}"
                )

        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        self.vocab_size = vocab_size
        self.override_stop_tokens = override_stop_tokens
        self.any_whitespace = any_whitespace

    # xgrammar后端支持token过滤功能
    @property
    def is_support_token_filter(self):
        return True

    # 分配词表位掩码（静态方法）
    @staticmethod
    def allocate_vocab_mask(vocab_size: int, batch_size: int, device) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, vocab_size)

    # 将词表位掩码移动到指定设备（静态方法）
    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    # 将词表位掩码应用到logits上（静态方法），根据设备类型选择实现
    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if logits.device.type in {"cuda", "npu", "xpu", "musa"}:
            if _is_hip:
                apply_token_bitmask_inplace_cuda(logits, vocab_mask)
            else:
                apply_token_bitmask_inplace_triton(logits, vocab_mask)
        else:
            raise RuntimeError(f"Unsupported device: {logits.device.type}")

    # 在词表掩码中设置或清除特定token的过滤
    @staticmethod
    def set_token_filter(
        vocab_mask: torch.Tensor,
        token_ids: List[int],
        batch_idx: int,
        is_allowed: bool = True,
        reset_vocab_mask: bool = True,
    ):
        if _is_hip or (vocab_mask.device.type != "cuda"):
            set_token_filter_torch(
                vocab_mask,
                token_ids,
                batch_idx,
                is_allowed=is_allowed,
                reset_vocab_mask=reset_vocab_mask,
            )
        else:
            set_token_filter_triton(
                vocab_mask,
                token_ids,
                batch_idx,
                is_allowed=is_allowed,
                reset_vocab_mask=reset_vocab_mask,
            )

    # 递归地补全结构化格式中缺失的json_schema字段为空schema
    @staticmethod
    def _sanitize_structural_format(structural_format):
        """Recursively replace missing json_schema fields with an empty schema."""
        if not isinstance(structural_format, dict):
            return

        fmt_type = structural_format.get("type")
        if fmt_type in {"json_schema", "qwen_xml_parameter"}:
            if structural_format.get("json_schema") is None:
                structural_format["json_schema"] = {}

        if fmt_type == "tag":
            XGrammarGrammarBackend._sanitize_structural_format(
                structural_format.get("content")
            )
        elif fmt_type in {"sequence", "or"}:
            for element in structural_format.get("elements", []):
                XGrammarGrammarBackend._sanitize_structural_format(element)
        elif fmt_type in {"triggered_tags", "tags_with_separator"}:
            for tag in structural_format.get("tags", []):
                XGrammarGrammarBackend._sanitize_structural_format(tag)

    # 清理旧版结构化标签中缺失的schema字段
    @staticmethod
    def _sanitize_structural_tag_structures(structural_tag: Dict) -> None:
        for structure in structural_tag.get("structures", []):
            if structure.get("schema") is None:
                structure["schema"] = {}

    # 从已编译的语法上下文创建XGrammarGrammar实例
    def _from_context(
        self, ctx: CompiledGrammar, key_string: str, grammar_stats: GrammarStats
    ) -> XGrammarGrammar:
        matcher = GrammarMatcher(
            ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        return XGrammarGrammar(
            matcher,
            self.vocab_size,
            ctx,
            self.override_stop_tokens,
            key_string,
            grammar_stats,
        )

    # 编译JSON schema语法
    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        try:
            if key_string == "$$ANY$$":
                # Note: This builtin JSON grammar includes *all* valid JSON (including, for example, arrays at the root)
                # 内置JSON语法包含所有合法JSON（例如根级别数组）
                ctx = self.grammar_compiler.compile_builtin_json_grammar()
            else:
                ctx = self.grammar_compiler.compile_json_schema(
                    schema=key_string, any_whitespace=self.any_whitespace
                )

        except (RuntimeError, json.decoder.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Hit invalid json_schema: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="json"))

    # 编译EBNF语法
    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        try:
            ctx = self.grammar_compiler.compile_grammar(key_string)
        except RuntimeError as e:
            logger.error(f"Hit invalid ebnf: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="ebnf"))

    # 编译正则表达式语法
    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        try:
            ctx = self.grammar_compiler.compile_regex(key_string)
        except RuntimeError as e:
            logger.error(f"Hit invalid regex: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="regex"))

    # 编译结构化标签语法，兼容旧版和新版格式
    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
        try:
            # TODO(dark): it's REALLY stupid to construct object from string and decode it again
            # 从字符串解析结构化标签对象
            structural_tag = json.loads(key_string)
            if is_legacy_structural_tag(structural_tag):
                # 处理旧版结构化标签格式
                self._sanitize_structural_tag_structures(structural_tag)
                tags = [
                    StructuralTagItem(
                        begin=structure["begin"],
                        schema=json.dumps(structure["schema"]),
                        end=structure["end"],
                    )
                    for structure in structural_tag["structures"]
                ]
                new_tag = StructuralTag.from_legacy_structural_tag(
                    tags, structural_tag["triggers"]
                )
                new_tag.format.at_least_one = structural_tag.get("at_least_one", False)
                ctx = self.grammar_compiler.compile_structural_tag(new_tag)
            else:
                # 处理新版结构化标签格式
                format_dict = structural_tag.get("format")
                if isinstance(format_dict, dict):
                    self._sanitize_structural_format(format_dict)
                    structural_tag["format"] = format_dict
                    key_string = json.dumps(structural_tag)
                ctx = self.grammar_compiler.compile_structural_tag(key_string)
        except (RuntimeError, json.decoder.JSONDecodeError) as e:
            logger.error(f"Hit invalid structural_tag: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))
        return self._from_context(
            ctx, key_string, GrammarStats(dispatch_type="structural_tag")
        )

    # 重置语法后端缓存和编译器缓存
    def reset(self):
        super().reset()
        self.grammar_compiler.clear_cache()


# 演示测试函数
def demo_test():
    from transformers import AutoConfig, AutoTokenizer

    from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME_FOR_TEST)
    hf_config = AutoConfig.from_pretrained(DEFAULT_MODEL_NAME_FOR_TEST)

    # Should use vocab size from model config
    # 应使用模型配置中的词表大小
    vocab_size = hf_config.vocab_size
    eos_token_id = tokenizer.eos_token_id

    backend = XGrammarGrammarBackend(
        tokenizer, vocab_size=vocab_size, model_eos_token_ids=[eos_token_id]
    )
    regex = r"hello (world|there)"
    grammar = backend.dispatch_regex(regex)
    tokens = [
        tokenizer.encode(t, add_special_tokens=False)[0] for t in ["hello", " world"]
    ]

    # Test termination
    # 测试终止状态
    grammar.accept_token(tokens[0])  # accept "hello"
    grammar.accept_token(tokens[1])  # accept " world"
    grammar.accept_token(eos_token_id)  # accept EOS
    assert grammar.is_terminated()

    # Test rollback the terminated state
    # 测试回滚终止状态
    grammar.rollback(1)
    assert not grammar.is_terminated()


if __name__ == "__main__":
    demo_test()
