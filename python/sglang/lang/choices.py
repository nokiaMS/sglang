# 文件说明：本文件定义了SGLang中选择（choices）功能的采样方法，包括基于token长度归一化、贪婪token选择和无条件似然归一化三种选择策略，用于在多个候选选项中决策最佳选项。

from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Any, Dict, List, Optional  # 导入类型注解

import numpy as np  # 导入NumPy库用于数值计算


@dataclass  # 使用数据类装饰器定义ChoicesDecision
class ChoicesDecision:
    """选择决策的数据类，包含决策结果和元信息。"""
    decision: str  # 决策结果字符串
    meta_info: Optional[Dict[str, Any]] = None  # 元信息字典，可选


class ChoicesSamplingMethod(ABC):
    """选择采样方法的抽象基类，定义了选择方法的接口。"""

    @property
    def requires_unconditional_logprobs(self) -> bool:
        """是否需要无条件logprobs，默认不需要。"""
        return False  # 默认返回False

    @abstractmethod  # 标记为抽象方法
    def __call__(
        self,
        *,
        choices: List[str],  # 候选选项列表
        normalized_prompt_logprobs: List[float],  # 归一化的提示logprobs列表
        input_token_logprobs: List[List[Any]],  # 输入token的logprobs列表
        output_token_logprobs: List[List[Any]],  # 输出token的logprobs列表
        unconditional_token_logprobs: Optional[List[List[Any]]] = None,  # 无条件token logprobs列表，可选
    ) -> ChoicesDecision: ...  # 返回选择决策对象


class TokenLengthNormalized(ChoicesSamplingMethod):
    """基于token长度归一化的选择方法，选择归一化提示logprob最高的选项。"""

    def __call__(
        self,
        *,
        choices: List[str],  # 候选选项列表
        normalized_prompt_logprobs: List[float],  # 归一化的提示logprobs列表
        input_token_logprobs: List[List[Any]],  # 输入token的logprobs列表
        output_token_logprobs: List[List[Any]],  # 输出token的logprobs列表
        unconditional_token_logprobs: Optional[List[List[Any]]] = None,  # 无条件token logprobs列表，可选
    ) -> ChoicesDecision:
        """选择具有最高token长度归一化提示logprob的选项。"""
        # Select the option with the highest token length normalized prompt logprob.
        best_choice = choices[np.argmax(normalized_prompt_logprobs)]  # 找到归一化logprob最大值对应的选项
        meta_info = {  # 构建元信息字典
            "normalized_prompt_logprobs": normalized_prompt_logprobs,  # 归一化提示logprobs
            "input_token_logprobs": input_token_logprobs,  # 输入token logprobs
            "output_token_logprobs": output_token_logprobs,  # 输出token logprobs
        }
        return ChoicesDecision(decision=best_choice, meta_info=meta_info)  # 返回选择决策


token_length_normalized = TokenLengthNormalized()  # 创建TokenLengthNormalized的默认实例


class GreedyTokenSelection(ChoicesSamplingMethod):
    """贪婪token选择方法，基于贪婪logprob选择策略来决策最佳选项。"""

    def __call__(
        self,
        *,
        choices: List[str],  # 候选选项列表
        normalized_prompt_logprobs: List[float],  # 归一化的提示logprobs列表
        input_token_logprobs: List[List[Any]],  # 输入token的logprobs列表
        output_token_logprobs: List[List[Any]],  # 输出token的logprobs列表
        unconditional_token_logprobs: Optional[List[List[Any]]] = None,  # 无条件token logprobs列表，可选
    ) -> ChoicesDecision:
        """基于贪婪logprob选择来选择选项。对于重叠选项，其中一个选项是较长选项的子集，
        使用较短选项的平均logprob进行扩展，以便与较长选项进行比较。"""
        # Select the option based on greedy logprob selection. For overlapping options
        # where one option is a subset of a longer option, extend the shorter option using
        # its average logprob for comparison against the longer option.

        num_options = len(choices)  # 获取选项数量
        max_tokens = max(len(option) for option in input_token_logprobs)  # 获取最大token数
        logprob_matrix = self._build_logprob_matrix(  # 构建logprob矩阵
            input_token_logprobs, max_tokens, num_options
        )
        remaining = self._greedy_selection(logprob_matrix, num_options, max_tokens)  # 执行贪婪选择

        best_choice = choices[remaining[0]]  # 获取最佳选项
        meta_info = {  # 构建元信息字典
            "normalized_prompt_logprobs": normalized_prompt_logprobs,  # 归一化提示logprobs
            "input_token_logprobs": input_token_logprobs,  # 输入token logprobs
            "output_token_logprobs": output_token_logprobs,  # 输出token logprobs
            "greedy_logprob_matrix": logprob_matrix.tolist(),  # 贪婪logprob矩阵转为列表
        }
        return ChoicesDecision(decision=best_choice, meta_info=meta_info)  # 返回选择决策

    def _build_logprob_matrix(self, input_token_logprobs, max_tokens, num_options):
        """构建logprob矩阵，将各选项的token logprobs填充到矩阵中，
        较短的选项用平均logprob填充剩余位置。"""
        logprob_matrix = np.zeros((num_options, max_tokens))  # 初始化全零矩阵
        for i, option in enumerate(input_token_logprobs):  # 遍历每个选项
            actual_logprobs = [token[0] for token in option]  # 提取实际logprob值
            avg_logprob = np.mean(actual_logprobs)  # 计算平均logprob
            logprob_matrix[i, : len(option)] = actual_logprobs  # 填充实际logprob
            if len(option) < max_tokens:  # 如果选项比最大长度短
                logprob_matrix[i, len(option) :] = avg_logprob  # 用平均logprob填充剩余位置
        return logprob_matrix  # 返回构建的矩阵

    def _greedy_selection(self, logprob_matrix, num_options, max_tokens):
        """执行贪婪选择，逐token比较logprob，保留每步logprob最大的选项。"""
        remaining = np.arange(num_options)  # 初始化剩余选项索引
        for j in range(max_tokens):  # 逐token位置遍历
            max_logprob = np.max(logprob_matrix[remaining, j])  # 找到当前步最大logprob
            remaining = remaining[logprob_matrix[remaining, j] == max_logprob]  # 保留logprob等于最大值的选项
            if len(remaining) == 1:  # 如果只剩一个选项
                break  # 跳出循环
        return remaining  # 返回剩余选项索引


greedy_token_selection = GreedyTokenSelection()  # 创建GreedyTokenSelection的默认实例


class UnconditionalLikelihoodNormalized(ChoicesSamplingMethod):
    """无条件似然归一化的选择方法，基于无条件token logprobs归一化后选择最佳选项。"""

    @property
    def requires_unconditional_logprobs(self) -> bool:
        """此方法需要无条件logprobs。"""
        return True  # 返回True表示需要

    def __call__(
        self,
        *,
        choices: List[str],  # 候选选项列表
        normalized_prompt_logprobs: List[float],  # 归一化的提示logprobs列表
        input_token_logprobs: List[List[Any]],  # 输入token的logprobs列表
        output_token_logprobs: List[List[Any]],  # 输出token的logprobs列表
        unconditional_token_logprobs: Optional[List[List[Any]]] = None,  # 无条件token logprobs列表，可选
    ) -> ChoicesDecision:
        """选择在无条件token logprobs归一化后平均token logprob最高的选项。

        第一个无条件token logprob假定为None。如果是这样，将其替换为0用于归一化。"""
        # Select the option with the highest average token logprob once normalized by
        # the unconditional token logprobs.

        # The first unconditional token logprob is assumed to be None. If so, it is
        # replaced with 0 for the purposes of normalization.

        if unconditional_token_logprobs is None:  # 检查无条件logprobs是否为None
            raise ValueError(  # 抛出错误
                "Unconditional token logprobs are required for this method."
            )

        normalized_unconditional_prompt_logprobs = self._normalize_logprobs(  # 计算归一化的无条件logprobs
            input_token_logprobs, unconditional_token_logprobs
        )

        best_choice = choices[np.argmax(normalized_unconditional_prompt_logprobs)]  # 找到最大归一化值对应的选项
        meta_info = {  # 构建元信息字典
            "normalized_prompt_logprobs": normalized_prompt_logprobs,  # 归一化提示logprobs
            "input_token_logprobs": input_token_logprobs,  # 输入token logprobs
            "output_token_logprobs": output_token_logprobs,  # 输出token logprobs
            "unconditional_token_logprobs": unconditional_token_logprobs,  # 无条件token logprobs
            "normalized_unconditional_prompt_logprobs": normalized_unconditional_prompt_logprobs,  # 归一化无条件提示logprobs
        }
        return ChoicesDecision(decision=best_choice, meta_info=meta_info)  # 返回选择决策

    def _normalize_logprobs(self, input_token_logprobs, unconditional_token_logprobs):
        """将输入logprobs减去无条件logprobs进行归一化，返回归一化后的平均logprob列表。"""
        normalized_unconditional_prompt_logprobs = []  # 初始化归一化结果列表
        for inputs, unconditionals in zip(  # 遍历输入和无条件logprobs对
            input_token_logprobs, unconditional_token_logprobs
        ):
            inputs_logprobs = np.array([token[0] for token in inputs])  # 提取输入logprob值
            unconditionals_logprobs = np.array([token[0] for token in unconditionals])  # 提取无条件logprob值
            unconditionals_logprobs[0] = unconditionals_logprobs[0] or 0  # 将首个None值替换为0
            normalized_unconditional_prompt_logprobs.append(  # 添加归一化结果
                float(np.mean(inputs_logprobs - unconditionals_logprobs))  # 计算差值的均值
            )
        return normalized_unconditional_prompt_logprobs  # 返回归一化结果列表


unconditional_likelihood_normalized = UnconditionalLikelihoodNormalized()  # 创建UnconditionalLikelihoodNormalized的默认实例
