# 该文件实现了自定义logit处理器框架
# 提供了自定义logit处理器的抽象基类和多种具体实现
# 包括禁止token处理器、思考预算处理器、n-gram重复抑制处理器等
# 用于在采样阶段对logits进行自定义修改

import json  # JSON序列化/反序列化
from abc import ABC, abstractmethod  # 抽象基类和抽象方法装饰器
from functools import lru_cache  # LRU缓存装饰器
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set  # 类型注解

import dill  # 序列化库，支持序列化函数和类
import orjson  # 高性能JSON库
import torch  # PyTorch张量库

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.managers.schedule_batch import Req  # 请求类


@lru_cache(maxsize=None)  # 无限制缓存
def _cache_from_str(json_str: str):  # 从JSON字符串反序列化为可调用对象
    """Deserialize a json string to a Callable object.
    This function is cached to avoid redundant deserialization.
    """  # 将JSON字符串反序列化为可调用对象，使用缓存避免重复反序列化
    data = orjson.loads(json_str)  # 解析JSON字符串
    return dill.loads(bytes.fromhex(data["callable"]))  # 从十六进制字符串还原可调用对象


class CustomLogitProcessor(ABC):  # 自定义logit处理器抽象基类
    """Abstract base class for callable functions."""  # 可调用函数的抽象基类

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,  # logits张量
        custom_param_list: Optional[List[Dict[str, Any]]] = None,  # 自定义参数列表，可选
    ) -> torch.Tensor:  # 调用处理器
        """Define the callable behavior."""  # 定义可调用行为
        raise NotImplementedError  # 未实现异常

    @classmethod
    def to_str(cls) -> str:  # 将处理器类序列化为JSON字符串
        """Serialize the callable function to a JSON-compatible string."""  # 将可调用函数序列化为JSON兼容的字符串
        return json.dumps({"callable": dill.dumps(cls).hex()})  # 使用dill序列化后转为十六进制

    @classmethod
    def from_str(cls, json_str: str):  # 从JSON字符串反序列化处理器实例
        """Deserialize a callable function from a JSON string."""  # 从JSON字符串反序列化可调用函数
        return _cache_from_str(json_str)()  # 从缓存获取类并实例化


class DisallowedTokensLogitsProcessor(CustomLogitProcessor):  # 禁止token的logit处理器
    """将指定的token ID的logit设为负无穷，从而禁止生成这些token"""
    def __call__(
        self,
        logits: torch.Tensor,  # logits张量
        custom_param_list: Optional[List[Dict[str, Any]]] = None,  # 自定义参数列表，可选
    ) -> torch.Tensor:  # 处理logits
        """将禁止的token ID对应的logit设为负无穷"""
        disallowed_token_ids = custom_param_list[0]["token_ids"]  # 获取禁止的token ID列表
        assert all(
            disallowed_token_ids == c["token_ids"] for c in custom_param_list
        ), f"{custom_param_list=}"  # 断言所有请求的禁止token列表一致
        logits[..., disallowed_token_ids] = -float("inf")  # 将禁止token的logit设为负无穷
        return logits  # 返回修改后的logits


class ThinkingBudgetLogitProcessor(CustomLogitProcessor):  # 思考预算logit处理器
    """A logit processor that controls the length of thinking."""  # 控制思考长度的logit处理器

    THINKING_START_TOKEN_ID: int  # 思考开始token ID
    THINKING_END_TOKEN_ID: int  # 思考结束token ID
    NEW_LINE_TOKEN_ID: int  # 换行token ID

    def __call__(self, logits, custom_param_list: list[dict[str, Any]]):  # 处理logits
        """根据思考预算控制思考阶段的长度，超出预算时强制结束思考"""
        if custom_param_list is None or not custom_param_list:  # 如果参数列表为空
            return logits  # 直接返回
        for i, param_dict in enumerate(custom_param_list):  # 遍历每个请求的参数
            if param_dict is None:  # 如果参数为空
                continue  # 跳过

            thinking_budget: int | None = param_dict.get("thinking_budget")  # 获取思考预算

            # Skip if thinking_budget is unset, or not an integer, or negative  # 如果思考预算未设置、不是整数或为负数则跳过
            if (
                thinking_budget is None
                or not isinstance(thinking_budget, int)
                or thinking_budget < 0
            ):
                continue  # 跳过
            req: Req = param_dict.get("__req__")  # 获取请求对象
            cur_ids: list[int] = [*req.origin_input_ids, *req.output_ids]  # 当前所有token ID

            # Check if out of thinking stage  # 检查是否已不在思考阶段
            if (
                self.THINKING_START_TOKEN_ID not in cur_ids
                or self.THINKING_END_TOKEN_ID in cur_ids
            ):
                continue  # 不在思考阶段则跳过

            # Find the index of the thinking start token  # 查找思考开始token的索引
            start_index = cur_ids.index(self.THINKING_START_TOKEN_ID)  # 获取开始token的索引

            # Count the number of tokens after the thinking start token  # 计算思考开始后的token数量
            num_tokens_after_start = len(cur_ids) - start_index - 1  # 思考开始后的token数量

            if num_tokens_after_start < thinking_budget:  # 如果未超出预算
                continue  # 跳过

            # Ensure new line token before thinking end token  # 确保思考结束token前有换行token
            if not req.output_ids or req.output_ids[-1] != self.NEW_LINE_TOKEN_ID:  # 如果最后不是换行
                logits[i, :] = -float("inf")  # 将所有logit设为负无穷
                logits[i, self.NEW_LINE_TOKEN_ID] = 0.0  # 只保留换行token的logit
                continue  # 继续下一个请求

            # Assign highest probability to the thinking end token  # 将最高概率分配给思考结束token
            logits[i, :] = -float("inf")  # 将所有logit设为负无穷
            logits[i, self.THINKING_END_TOKEN_ID] = 0.0  # 只保留思考结束token的logit

        return logits  # 返回修改后的logits


class Glm4MoeThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):  # GLM-4系列的思考预算处理器
    """A logit processor that controls the length of thinking for GLM-4.5 / GLM-4.6 / GLM-4.5V / GLM-4.6V models."""  # 控制GLM-4系列模型思考长度的logit处理器

    THINKING_START_TOKEN_ID: int = 151350  # GLM-4思考开始token ID
    THINKING_END_TOKEN_ID: int = 151351  # GLM-4思考结束token ID
    NEW_LINE_TOKEN_ID: int = 198  # GLM-4换行token ID


class Qwen3ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):  # Qwen3的思考预算处理器
    """A logit processor that controls the length of thinking for Qwen3 models."""  # 控制Qwen3模型思考长度的logit处理器

    THINKING_START_TOKEN_ID: int = 151667  # Qwen3思考开始token ID
    THINKING_END_TOKEN_ID: int = 151668  # Qwen3思考结束token ID
    NEW_LINE_TOKEN_ID: int = 198  # Qwen3换行token ID


class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):  # DeepSeek-R1的思考预算处理器
    """A logit processor that controls the length of thinking for DeepSeek-R1 models."""  # 控制DeepSeek-R1模型思考长度的logit处理器

    THINKING_START_TOKEN_ID: int = 128798  # DeepSeek-R1思考开始token ID
    THINKING_END_TOKEN_ID: int = 128799  # DeepSeek-R1思考结束token ID
    NEW_LINE_TOKEN_ID: int = 201  # DeepSeek-R1换行token ID


# Adapted from DeepSeek's implementation: https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py  # 适配自DeepSeek的实现
class DeepseekOCRNoRepeatNGramLogitProcessor(CustomLogitProcessor):  # DeepSeek OCR n-gram重复抑制处理器
    """Block n-gram repetitions within a sliding window for DeepSeek-OCR outputs."""  # 在滑动窗口内阻止n-gram重复，用于DeepSeek-OCR输出

    def __call__(
        self,
        logits: torch.Tensor,  # logits张量
        custom_param_list: Optional[List[Dict[str, Any]]] = None,  # 自定义参数列表，可选
    ) -> torch.Tensor:  # 处理logits
        """在滑动窗口内检测并抑制n-gram重复"""
        if not custom_param_list:  # 如果参数列表为空
            return logits  # 直接返回

        for batch_idx, params in enumerate(custom_param_list):  # 遍历每个请求的参数
            if not params:  # 如果参数为空
                continue  # 跳过

            req = params.get("__req__")  # 获取请求对象
            if req is None:  # 如果请求为空
                continue  # 跳过

            try:
                ngram_size = int(params.get("ngram_size") or 0)  # 获取n-gram大小
                window_size = int(params.get("window_size") or 0)  # 获取窗口大小
            except (TypeError, ValueError):  # 类型或值错误
                continue  # 跳过

            if ngram_size <= 0 or window_size <= 0:  # 如果参数无效
                continue  # 跳过

            sequence: List[int] = req.origin_input_ids + req.output_ids  # 完整的token序列
            if len(sequence) < ngram_size:  # 如果序列短于n-gram大小
                continue  # 跳过

            search_start = max(0, len(sequence) - window_size)  # 搜索起始位置
            search_end = len(sequence) - ngram_size + 1  # 搜索结束位置
            if search_end <= search_start:  # 如果搜索范围为空
                continue  # 跳过

            if ngram_size > 1:  # 如果n-gram大小大于1
                current_prefix = tuple(sequence[-(ngram_size - 1) :])  # 获取当前前缀
            else:
                current_prefix = tuple()  # 空前缀

            banned_tokens: Set[int] = set()  # 被禁止的token集合
            for idx in range(search_start, search_end):  # 遍历搜索范围
                ngram = sequence[idx : idx + ngram_size]  # 获取n-gram
                if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:  # 如果前缀匹配
                    banned_tokens.add(ngram[-1])  # 添加到禁止集合

            whitelist_ids = params.get("whitelist_token_ids") or []  # 获取白名单token ID
            try:
                whitelist = {int(token_id) for token_id in whitelist_ids}  # 转换为整数集合
            except (TypeError, ValueError):  # 类型或值错误
                whitelist = set()  # 使用空集合

            banned_tokens.difference_update(whitelist)  # 从禁止集合中移除白名单token

            if not banned_tokens:  # 如果没有需要禁止的token
                continue  # 跳过

            indices = list(banned_tokens)  # 转换为列表
            logits[batch_idx, indices] = -float("inf")  # 将禁止token的logit设为负无穷

        return logits  # 返回修改后的logits
