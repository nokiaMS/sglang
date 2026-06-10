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
"""Sampling parameters for text generation."""
# 文本生成的采样参数模块，定义了控制文本生成行为的各种参数

import logging  # 导入日志模块
from typing import Any, Dict, List, Optional, Union  # 导入类型提示

# sre_parse is deprecated in Python 3.11+, use re._parser instead
# sre_parse 在 Python 3.11+ 中已弃用，改用 re._parser
try:
    import re._parser as sre_parse  # 尝试导入 re._parser（Python 3.11+）
except ImportError:
    import sre_parse  # Python < 3.11  # Python 3.11 以下版本回退导入 sre_parse

_SAMPLING_EPS = 1e-6  # 采样的最小浮点数阈值，用于判断温度是否接近零
TOP_K_ALL = 1 << 30  # 表示不限制 top_k 的值（即使用整个词表）

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


class SamplingParams:  # 采样参数类，用于控制文本生成的各种行为
    """
    The sampling parameters.

    See docs/backend/sampling_params.md or
    https://docs.sglang.io/backend/sampling_params.html
    for the documentation.
    """
    # 采样参数类，用于配置文本生成的各种策略，
    # 详细文档参见 docs/backend/sampling_params.md 或
    # https://docs.sglang.io/backend/sampling_params.html

    def __init__(  # 初始化方法，设置所有采样参数
        self,
        max_new_tokens: int = 128,  # 生成的最大新 token 数量，默认 128
        stop: Optional[Union[str, List[str]]] = None,  # 停止生成的字符串或字符串列表
        stop_token_ids: Optional[List[int]] = None,  # 停止生成的 token ID 列表
        stop_regex: Optional[Union[str, List[str]]] = None,  # 用于停止生成的正则表达式
        temperature: float = 1.0,  # 采样温度，控制随机性，越高越随机
        top_p: float = 1.0,  # nucleus 采样的概率阈值
        top_k: int = -1,  # top-k 采样的 k 值，-1 表示不限制
        min_p: float = 0.0,  # 最小概率阈值，低于此值的 token 被过滤
        frequency_penalty: float = 0.0,  # 频率惩罚，降低已出现 token 的概率
        presence_penalty: float = 0.0,  # 存在惩罚，降低已出现 token 的概率（不依赖频率）
        repetition_penalty: float = 1.0,  # 重复惩罚，1.0 表示不惩罚
        min_new_tokens: int = 0,  # 最少生成的新 token 数量
        n: int = 1,  # 并行生成的序列数量
        json_schema: Optional[str] = None,  # JSON schema 约束输出格式
        regex: Optional[str] = None,  # 正则表达式约束输出格式
        ebnf: Optional[str] = None,  # EBNF 文法约束输出格式
        structural_tag: Optional[str] = None,  # 结构化标签约束输出格式
        ignore_eos: bool = False,  # 是否忽略结束符（EOS）
        skip_special_tokens: bool = True,  # 是否在输出中跳过特殊 token
        spaces_between_special_tokens: bool = True,  # 特殊 token 之间是否添加空格
        no_stop_trim: bool = False,  # 是否不裁剪停止字符串
        custom_params: Optional[Dict[str, Any]] = None,  # 自定义参数字典
        stream_interval: Optional[int] = None,  # 流式输出的间隔时间
        logit_bias: Optional[Dict[str, float]] = None,  # 对特定 token 的 logits 偏置
        sampling_seed: Optional[int] = None,  # 采样的随机种子
    ) -> None:  # 构造函数无返回值
        # For non-optional params, treat None as "use default" so that callers
        # (e.g. /generate) can pass null without crashing verify().
        # 对于非可选参数，将 None 视为"使用默认值"，这样调用者（如 /generate）
        # 可以传 null 而不会导致 verify() 崩溃
        self.max_new_tokens = max_new_tokens  # 设置最大新 token 数量
        self.stop_strs = stop  # 设置停止字符串
        if stop_token_ids:  # 如果提供了停止 token ID 列表
            filtered = {int(t) for t in stop_token_ids if t is not None}  # 过滤掉 None 值并转为整数集合
            self.stop_token_ids = filtered or None  # 如果集合非空则使用，否则设为 None
        else:  # 如果没有提供停止 token ID
            self.stop_token_ids = None  # 设为 None
        self.stop_regex_strs = stop_regex  # 设置停止正则表达式
        self.temperature = temperature if temperature is not None else 1.0  # 设置温度，None 时使用默认值 1.0
        self.top_p = top_p if top_p is not None else 1.0  # 设置 top_p，None 时使用默认值 1.0
        self.top_k = top_k if top_k is not None else -1  # 设置 top_k，None 时使用默认值 -1
        self.min_p = min_p if min_p is not None else 0.0  # 设置 min_p，None 时使用默认值 0.0
        self.frequency_penalty = (  # 设置频率惩罚
            frequency_penalty if frequency_penalty is not None else 0.0  # None 时使用默认值 0.0
        )
        self.presence_penalty = (  # 设置存在惩罚
            presence_penalty if presence_penalty is not None else 0.0  # None 时使用默认值 0.0
        )
        self.repetition_penalty = (  # 设置重复惩罚
            repetition_penalty if repetition_penalty is not None else 1.0  # None 时使用默认值 1.0
        )
        self.min_new_tokens = min_new_tokens if min_new_tokens is not None else 0  # 设置最少新 token 数，None 时默认 0
        self.regex = regex  # 设置正则约束
        self.n = n if n is not None else 1  # 设置并行序列数，None 时默认 1
        self.json_schema = json_schema  # 设置 JSON schema 约束
        self.ebnf = ebnf  # 设置 EBNF 文法约束
        self.structural_tag = structural_tag  # 设置结构化标签约束
        self.ignore_eos = ignore_eos if ignore_eos is not None else False  # 设置是否忽略 EOS，None 时默认 False
        self.skip_special_tokens = (  # 设置是否跳过特殊 token
            skip_special_tokens if skip_special_tokens is not None else True  # None 时默认 True
        )
        self.spaces_between_special_tokens = (  # 设置特殊 token 间是否加空格
            spaces_between_special_tokens  # 条件表达式开始
            if spaces_between_special_tokens is not None  # 如果不为 None
            else True  # 否则默认 True
        )
        self.no_stop_trim = no_stop_trim if no_stop_trim is not None else False  # 设置是否不裁剪停止字符串，None 时默认 False
        self.custom_params = custom_params  # 设置自定义参数
        self.stream_interval = stream_interval  # 设置流式输出间隔
        self.logit_bias = logit_bias  # 设置 logit 偏置
        self.sampling_seed = sampling_seed  # 设置采样随机种子

        # Process some special cases
        # 处理一些特殊情况
        if 0 <= self.temperature < _SAMPLING_EPS:  # 如果温度接近零（在 0 和采样阈值之间）
            # top_k = 1 means greedy sampling
            # top_k = 1 表示贪心采样
            self.temperature = 1.0  # 将温度重置为 1.0（因为 top_k=1 已保证贪心）
            self.top_k = 1  # 设置 top_k 为 1，实现贪心解码
        if self.top_k == -1:  # 如果 top_k 仍为 -1（即未限制）
            self.top_k = TOP_K_ALL  # whole vocabulary  # 设置为极大值，等效于使用整个词表

    def verify(self, vocab_size):  # 验证采样参数的合法性
        """Verify the sampling parameters are valid."""  # 验证采样参数是否合法
        # 验证采样参数的合法性，确保所有参数在有效范围内
        if self.temperature < 0.0:  # 检查温度是否为负
            raise ValueError(  # 抛出值错误
                f"temperature must be non-negative, got {self.temperature}."  # 温度必须非负
            )
        if not 0.0 < self.top_p <= 1.0:  # 检查 top_p 是否在 (0, 1] 范围内
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")  # top_p 必须在 (0, 1] 范围内
        if not 0.0 <= self.min_p <= 1.0:  # 检查 min_p 是否在 [0, 1] 范围内
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")  # min_p 必须在 [0, 1] 范围内
        if self.top_k < 1 or self.top_k == -1:  # 检查 top_k 是否合法（此处 -1 已被转换，不应出现）
            raise ValueError(  # 抛出值错误
                f"top_k must be -1 (disable) or at least 1, got {self.top_k}."  # top_k 必须为 -1 或至少为 1
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:  # 检查频率惩罚是否在 [-2, 2] 范围内
            raise ValueError(  # 抛出值错误
                "frequency_penalty must be in [-2, 2], got "  # 频率惩罚必须在 [-2, 2] 范围内
                f"{self.frequency_penalty}."  # 显示实际值
            )
        if not -2.0 <= self.presence_penalty <= 2.0:  # 检查存在惩罚是否在 [-2, 2] 范围内
            raise ValueError(  # 抛出值错误
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."  # 存在惩罚必须在 [-2, 2] 范围内
            )
        if not 0.0 < self.repetition_penalty <= 2.0:  # 检查重复惩罚是否在 (0, 2] 范围内
            raise ValueError(  # 抛出值错误
                "repetition_penalty must be in (0, 2] (1.0 = no penalty), "  # 重复惩罚必须在 (0, 2] 范围内，1.0 表示无惩罚
                f"got {self.repetition_penalty}."  # 显示实际值
            )
        if not 0 <= self.min_new_tokens:  # 检查最少新 token 数是否非负
            raise ValueError(  # 抛出值错误
                f"min_new_tokens must be in [0, max_new_tokens], got "  # 最少新 token 数必须在 [0, max_new_tokens] 范围内
                f"{self.min_new_tokens}."  # 显示实际值
            )
        if self.max_new_tokens is not None:  # 如果设置了最大新 token 数
            if self.max_new_tokens < 0:  # 检查是否为负数
                raise ValueError(  # 抛出值错误
                    f"max_new_tokens must be at least 0, got {self.max_new_tokens}."  # 最大新 token 数必须至少为 0
                )
            if not self.min_new_tokens <= self.max_new_tokens:  # 检查最少是否不超过最多
                raise ValueError(  # 抛出值错误
                    f"min_new_tokens must be in [0, max_new_tokens({self.max_new_tokens})], got "  # 最少新 token 数不能超过最大值
                    f"{self.min_new_tokens}."  # 显示实际值
                )
        if self.logit_bias is not None:  # 如果设置了 logit 偏置
            for token_id in self.logit_bias:  # 遍历所有偏置的 token ID
                if not 0 <= int(token_id) < vocab_size:  # 检查 token ID 是否在有效范围内
                    raise ValueError(  # 抛出值错误
                        f"logit_bias must has keys in [0, {vocab_size - 1}], got "  # logit 偏置的键必须在 [0, vocab_size-1] 范围内
                        f"{token_id}."  # 显示无效的 token ID
                    )

        grammars = [  # 收集所有文法约束
            self.json_schema,  # JSON schema 约束
            self.regex,  # 正则约束
            self.ebnf,  # EBNF 文法约束
        ]  # since mutually exclusive, only one can be set  # 因为互斥，只能设置一个
        if sum(x is not None for x in grammars) > 1:  # 如果设置了多个文法约束
            raise ValueError("Only one of regex, json_schema, or ebnf can be set.")  # 只能设置其中一种文法约束

    def normalize(self, tokenizer):  # 规范化采样参数，处理停止字符串和正则表达式
        """Normalize the sampling parameters."""  # 规范化采样参数
        # 规范化采样参数，处理停止字符串和停止正则表达式的长度计算
        # Process stop strings
        # 处理停止字符串
        if self.stop_strs is None:  # 如果没有设置停止字符串
            self.stop_strs = []  # 初始化为空列表
            self.stop_str_max_len = 0  # 停止字符串最大长度设为 0
        else:  # 如果设置了停止字符串
            if isinstance(self.stop_strs, str):  # 如果是单个字符串
                self.stop_strs = [self.stop_strs]  # 将其转为列表

            stop_str_max_len = 0  # 初始化停止字符串最大长度
            for stop_str in self.stop_strs:  # 遍历每个停止字符串
                if tokenizer is not None:  # 如果提供了分词器
                    stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)  # 对停止字符串进行编码（不添加特殊 token）
                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))  # 取最大编码长度
                else:  # 如果没有提供分词器
                    stop_str_max_len = max(stop_str_max_len, len(stop_str))  # 使用字符串字符长度作为估计
            self.stop_str_max_len = stop_str_max_len  # 保存停止字符串的最大长度

        # Process stop regex strings
        # 处理停止正则表达式
        if self.stop_regex_strs is None:  # 如果没有设置停止正则表达式
            self.stop_regex_strs = []  # 初始化为空列表
            self.stop_regex_max_len = 0  # 停止正则最大长度设为 0
        else:  # 如果设置了停止正则表达式
            if isinstance(self.stop_regex_strs, str):  # 如果是单个字符串
                self.stop_regex_strs = [self.stop_regex_strs]  # 将其转为列表

            stop_regex_max_len = 0  # 初始化停止正则最大长度
            for stop_regex in self.stop_regex_strs:  # 遍历每个停止正则
                stop_regex_max_len = max(  # 取最大长度
                    stop_regex_max_len, get_max_seq_length(stop_regex)  # 调用 get_max_seq_length 计算正则的最大序列长度
                )

            self.stop_regex_max_len = stop_regex_max_len  # 保存停止正则的最大长度


# This function gets a strict upperbound on the maximum number of tokens that would need
# to be buffered to match the input regex string
# NOTE: in the worst case, one character that needs to be buffered corresponds to one
# token
# 此函数获取匹配输入正则表达式所需缓冲的最大 token 数的严格上界
# 注意：在最坏情况下，每个需要缓冲的字符对应一个 token
def get_max_seq_length(regex_str: str):  # 获取正则表达式可能匹配的最大序列长度
    return _max_length_from_subpattern(sre_parse.parse(regex_str))  # 解析正则并递归计算最大长度


MAX_LEN = 2**30  # 最大长度常量，用于表示无上限的情况


def _max_length_from_subpattern(subpattern: sre_parse.SubPattern):  # 从正则子模式递归计算最大匹配长度
    """Recursively compute the maximum length a regex subpattern can match."""  # 递归计算正则子模式能匹配的最大长度
    # 递归计算正则子模式能匹配的最大长度
    total = 0  # 初始化总长度
    for token, value in subpattern:  # 遍历子模式中的每个 token 及其值
        if token in {  # 如果 token 是以下类型之一
            sre_parse.LITERAL,  # `value` is any one character  # 字面量，匹配单个字符
            sre_parse.IN,  # Any character within `value`  # 字符集，匹配 value 中的任意字符
            sre_parse.ANY,  # "."  # 任意字符，匹配"."
        }:
            total += 1  # 这些类型各匹配一个字符，长度加 1
        elif token == sre_parse.SUBPATTERN:  # 如果是子模式（即括号分组）
            # EG: (a\d+) ->
            # [(SUBPATTERN,
            #   (1, 0, 0, [(LITERAL, 97),
            #              (MAX_REPEAT, (1, MAXREPEAT, [(IN, [(CATEGORY, CATEGORY_DIGIT)])]))]))]
            # 例如：(a\d+) 会被解析为 SUBPATTERN 包含 LITERAL 和 MAX_REPEAT
            _, _, _, inner_subpattern = value  # 解构 value，取内部子模式
            total += _max_length_from_subpattern(inner_subpattern)  # 递归计算内部子模式的最大长度
        elif token == sre_parse.BRANCH:  # 如果是分支（即 | 操作符）
            _, branches = value  # 解构 value，取所有分支
            total += max(_max_length_from_subpattern(branch) for branch in branches)  # 取所有分支中最大长度
        elif token in {sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT}:  # 如果是重复（即 * + {m,n} 操作符）
            _, max_num_repeat, inner_subpattern = value  # 解构 value，取最大重复次数和内部子模式
            if max_num_repeat == sre_parse.MAXREPEAT:  # 如果最大重复次数无上限
                total += MAX_LEN  # 加上最大长度常量表示无上限
            else:  # 如果最大重复次数有上限
                total += max_num_repeat * _max_length_from_subpattern(inner_subpattern)  # 最大重复次数 × 内部模式长度
        elif token == sre_parse.AT:  # 如果是锚点（零宽断言）
            # These are zero-width assertions like ^, $, and \b that don't add to the max
            # length
            # 这些是零宽断言，如 ^、$ 和 \b，不增加最大长度
            total += 0  # 零宽断言不匹配字符，长度加 0
        else:  # 其他未处理的 token 类型
            logger.warning(f"Got unhandled regex token: {token}")  # 记录警告日志

            total += MAX_LEN  # 对未处理的 token 使用最大长度常量，保证安全性

    return total  # 返回计算的最大长度
