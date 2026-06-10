# 本文件实现了SGLang的OpenAI后端，支持聊天模型和补全模型，
# 提供推测执行、流式生成以及通过logit bias操纵实现选项选择等功能。
import dataclasses  # 导入数据类装饰器模块
import logging  # 导入日志记录模块
import time  # 导入时间模块，用于重试等待
import warnings  # 导入警告模块，用于发出警告信息
from typing import List, Optional, Union  # 导入类型提示相关工具

import numpy as np  # 导入numpy库，用于数值计算

from sglang.lang.backend.base_backend import BaseBackend  # 导入基础后端类
from sglang.lang.chat_template import ChatTemplate, get_chat_template_by_model_path  # 导入聊天模板及根据模型路径获取模板的函数
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod  # 导入选项决策类和选项采样方法枚举
from sglang.lang.interpreter import StreamExecutor  # 导入流执行器类
from sglang.lang.ir import SglSamplingParams  # 导入SGLang采样参数类

try:  # 尝试导入openai和tiktoken库
    import openai  # 导入OpenAI Python SDK
    import tiktoken  # 导入tiktoken分词器库
except ImportError as e:  # 如果导入失败，捕获异常
    openai = tiktoken = e  # 将openai和tiktoken设置为异常对象，后续使用时会抛出


logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


def create_logit_bias_int(tokenizer):
    """Get logit bias for integer numbers."""  # 获取整数数字的logit偏置
    """创建用于限制生成整数数字的logit偏置字典"""
    int_token_ids = []  # 初始化整数token ID列表

    tokens = tokenizer._mergeable_ranks  # 获取分词器中可合并的token及其排名
    for token, token_id in tokens.items():  # 遍历每个token及其ID
        s = tokenizer.decode([token_id])  # 将token ID解码为字符串
        if all([c.isdigit() for c in s]) or s in [" "]:  # 如果字符串全部是数字或为空格
            int_token_ids.append(token_id)  # 将该token ID加入整数列表
            if len(int_token_ids) >= 300:  # OpenAI API limit  # 达到OpenAI API的300个限制时跳出
                break  # 跳出循环
    special_tokens = tokenizer._special_tokens  # 获取分词器的特殊token字典
    mask = {t: 100 for t in int_token_ids[:299]}  # 为前299个整数token ID创建偏置字典，偏置值为100
    mask[special_tokens["<|endoftext|>"]] = 100  # 将结束文本特殊token也加入偏置字典
    return mask  # 返回logit偏置字典


INSTRUCT_MODEL_NAMES = [  # 指令模型的名称列表
    "gpt-3.5-turbo-instruct",  # GPT-3.5 Turbo指令模型
]


@dataclasses.dataclass  # 使用dataclass装饰器定义数据类
class TokenUsage:  # Token使用量数据类
    """Token使用量记录类，用于跟踪提示和补全的token数量"""
    prompt_tokens: int  # 提示token数量
    completion_tokens: int  # 补全token数量

    def reset(self):  # 重置token使用量
        """重置提示和补全token计数为零"""
        self.prompt_tokens = self.completion_tokens = 0  # 将两个计数器都置为零


class OpenAI(BaseBackend):  # OpenAI后端类，继承自BaseBackend
    """OpenAI后端实现类，支持聊天和补全两种模式，提供推测执行、流式生成和选项选择功能"""
    def __init__(  # 初始化方法
        self,
        model_name: str,  # 模型名称
        is_chat_model: Optional[bool] = None,  # 是否为聊天模型，可选
        chat_template: Optional[ChatTemplate] = None,  # 聊天模板，可选
        is_azure: bool = False,  # 是否使用Azure OpenAI服务
        *args,  # 可变位置参数
        **kwargs,  # 可变关键字参数
    ):
        super().__init__()  # 调用父类初始化方法

        if isinstance(openai, Exception):  # 如果openai是异常对象（导入失败）
            raise openai  # 抛出导入异常

        if is_azure:  # 如果使用Azure OpenAI
            self.client = openai.AzureOpenAI(*args, **kwargs)  # 创建Azure OpenAI客户端
        else:  # 否则使用标准OpenAI
            self.client = openai.OpenAI(*args, **kwargs)  # 创建OpenAI客户端

        self.model_name = model_name  # 保存模型名称
        try:  # 尝试获取模型专用的分词器
            self.tokenizer = tiktoken.encoding_for_model(model_name)  # 根据模型名称获取对应的编码器
        except KeyError:  # 如果模型名称没有对应的编码器
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # 使用cl100k_base作为默认编码器
        self.logit_bias_int = create_logit_bias_int(self.tokenizer)  # 创建整数logit偏置字典

        self.chat_template = chat_template or get_chat_template_by_model_path(  # 设置聊天模板
            model_name  # 根据模型路径获取模板
        )

        if is_chat_model is not None:  # 如果显式指定了是否为聊天模型
            self.is_chat_model = is_chat_model  # 使用指定值
        else:  # 否则根据模型名称自动判断
            if model_name in INSTRUCT_MODEL_NAMES:  # 如果模型名称在指令模型列表中
                self.is_chat_model = False  # 标记为非聊天模型
            else:  # 否则
                self.is_chat_model = True  # 标记为聊天模型

        self.chat_prefix = self.chat_template.role_prefix_and_suffix["assistant"][0]  # 获取助手角色的前缀字符串

        # Usage  # 使用量统计
        self.token_usage = TokenUsage(0, 0)  # 初始化token使用量计数器

        # API speculative execution  # API推测执行相关参数
        # TODO(ying): This does not support multi-threading (run_batch)  # TODO: 不支持多线程(run_batch)
        self.spec_kwargs = {}  # 推测执行的API调用参数字典
        self.spec_format = []  # 推测执行的格式列表，用于模式匹配
        self.spec_max_num_tries = 3  # 推测执行的最大重试次数

    def get_chat_template(self):  # 获取聊天模板方法
        """返回当前使用的聊天模板"""
        return self.chat_template  # 返回聊天模板对象

    def _prepare_spec_execution(  # 准备推测执行
        self,
        sampling_params: SglSamplingParams,  # 采样参数
        num_api_spec_tokens: int,  # API推测的token数量
        spec_var_name: str,  # 推测变量名称
    ):
        """准备API推测执行，收集采样参数并注册格式信息"""
        if "max_tokens" not in self.spec_kwargs:  # 如果尚未设置max_tokens
            self.spec_kwargs["max_tokens"] = num_api_spec_tokens  # 设置为推测token数量
        else:  # 如果已经设置
            assert self.spec_kwargs["max_tokens"] == num_api_spec_tokens  # 确保与当前推测token数量一致

        params = sampling_params.to_openai_kwargs()  # 将采样参数转换为OpenAI API参数
        for key, value in params.items():  # 遍历每个参数键值对
            if key in ["stop"]:  # 跳过stop参数
                continue  # 继续下一个参数
            if key in ["max_tokens"]:  # 如果是max_tokens参数
                warnings.warn(  # 发出警告
                    "The parameter max_tokens will be overwritten by speculated number of tokens."  # max_tokens将被推测token数量覆盖
                )
                continue  # 继续下一个参数
            if key not in self.spec_kwargs:  # 如果该参数尚未在推测参数中
                self.spec_kwargs[key] = value  # 添加到推测参数字典
            else:  # 如果参数已存在
                assert (  # 断言参数值一致
                    value == self.spec_kwargs[key]
                ), "sampling parameters should be consistent if turn on api speculative execution."  # 开启API推测执行时采样参数应保持一致
        self.spec_format.append(  # 将格式信息追加到推测格式列表
            {"text": "", "stop": params["stop"], "name": spec_var_name}  # 包含空文本、停止条件和变量名
        )
        return "", {}  # 返回空字符串和空字典

    def generate(  # 生成方法
        self,
        s: StreamExecutor,  # 流执行器
        sampling_params: SglSamplingParams,  # 采样参数
        spec_var_name: str = None,  # 推测变量名，默认为None
    ):
        """根据采样参数生成文本，支持无约束生成、字符串约束和整数约束"""
        if sampling_params.dtype is None:  # 如果没有数据类型约束
            if self.is_chat_model:  # 如果是聊天模型
                if s.num_api_spec_tokens is None:  # 如果未启用API推测执行
                    if not s.text_.endswith(self.chat_prefix):  # 如果文本不以聊天前缀结尾
                        raise RuntimeError(  # 抛出运行时错误
                            "This use case is not supported if api speculative execution is off. "  # 未开启API推测执行时不支持此用例
                            "For OpenAI chat models, sgl.gen must be right after sgl.assistant. "  # 对于OpenAI聊天模型，sgl.gen必须紧跟在sgl.assistant之后
                            "Example of adding api speculative execution: @function(num_api_spec_tokens=128)."  # 添加API推测执行的示例
                        )
                    prompt = s.messages_  # 使用消息列表作为提示
                else:  # 如果启用了API推测执行
                    return self._prepare_spec_execution(  # 调用推测执行准备方法
                        sampling_params, s.num_api_spec_tokens, spec_var_name  # 传入采样参数、推测token数和变量名
                    )
            else:  # 如果不是聊天模型
                prompt = s.text_  # 使用文本作为提示

            kwargs = sampling_params.to_openai_kwargs()  # 将采样参数转换为OpenAI API参数
            if (  # 如果模型名称以o1或o3开头，或包含o1
                self.model_name.startswith("o1")  # 模型名称以o1开头
                or self.model_name.startswith("o3")  # 或以o3开头
                or "o1" in self.model_name  # 或包含o1
            ):
                kwargs.pop("max_tokens", None)  # 移除max_tokens参数（o系列模型不支持）
            else:  # 其他模型
                kwargs.pop("max_completion_tokens", None)  # 移除max_completion_tokens参数

            comp = openai_completion(  # 调用OpenAI补全函数
                client=self.client,  # 传入客户端
                token_usage=self.token_usage,  # 传入token使用量追踪器
                is_chat=self.is_chat_model,  # 传入是否为聊天模型
                model=self.model_name,  # 传入模型名称
                prompt=prompt,  # 传入提示
                **kwargs,  # 传入其他API参数
            )
            # Keep the returned list (or string) as is.  # 保持返回的列表（或字符串）不变
        elif sampling_params.dtype in [str, "str", "string"]:  # 如果约束类型为字符串
            assert (  # 断言
                not self.is_chat_model  # 不是聊天模型
            ), "constrained type not supported on chat model"  # 聊天模型不支持约束类型
            kwargs = sampling_params.to_openai_kwargs()  # 将采样参数转换为OpenAI API参数
            kwargs.pop("stop")  # 移除stop参数
            comp = openai_completion(  # 调用OpenAI补全函数
                client=self.client,  # 传入客户端
                token_usage=self.token_usage,  # 传入token使用量追踪器
                is_chat=self.is_chat_model,  # 传入是否为聊天模型
                model=self.model_name,  # 传入模型名称
                prompt=s.text_ + '"',  # 在文本末尾添加引号，引导生成字符串
                stop='"',  # 设置停止条件为引号
                **kwargs,  # 传入其他API参数
            )
            # Wrap each element in quotes if we have a list.  # 如果返回列表，则为每个元素包裹引号
            if isinstance(comp, list):  # 如果结果是列表
                comp = ['"' + x + '"' for x in comp]  # 为每个元素添加引号
            else:  # 如果结果是字符串
                comp = '"' + comp + '"'  # 为字符串添加引号
        elif sampling_params.dtype in [int, "int"]:  # 如果约束类型为整数
            assert (  # 断言
                not self.is_chat_model  # 不是聊天模型
            ), "constrained type not supported on chat model"  # 聊天模型不支持约束类型
            kwargs = sampling_params.to_openai_kwargs()  # 将采样参数转换为OpenAI API参数
            kwargs.pop("stop")  # 移除stop参数
            comp = openai_completion(  # 调用OpenAI补全函数
                client=self.client,  # 传入客户端
                token_usage=self.token_usage,  # 传入token使用量追踪器
                is_chat=self.is_chat_model,  # 传入是否为聊天模型
                model=self.model_name,  # 传入模型名称
                prompt=s.text_,  # 传入文本提示
                logit_bias=self.logit_bias_int,  # 传入整数logit偏置，限制只生成数字
                stop=[" "],  # 设置空格为停止条件
                **kwargs,  # 传入其他API参数
            )
            # Leave as a list if that's what is returned.  # 如果返回列表则保持为列表
        else:  # 其他未知的数据类型
            raise ValueError(f"Unknown dtype: {sampling_params.dtype}")  # 抛出值错误

        return comp, {}  # 返回补全结果和空元信息字典

    def spec_fill(self, value: str):  # 推测执行填充方法
        """在推测执行中填充固定文本，用于模式匹配"""
        assert self.is_chat_model  # 断言当前是聊天模型
        self.spec_format.append({"text": value, "stop": None, "name": None})  # 追加固定文本到格式列表

    def spec_pattern_match(self, comp):  # 推测执行模式匹配方法
        """将生成的补全文本与推测格式进行模式匹配，提取变量值"""
        for i, term in enumerate(self.spec_format):  # 遍历格式列表中的每一项
            text = term["text"]  # 获取该项的文本
            if text != "":  # 如果文本不为空（是固定文本）
                if comp.startswith(text):  # 如果补全文本以固定文本开头
                    comp = comp[len(text) :]  # 从补全文本中移除固定文本前缀
                else:  # 如果不匹配
                    return False  # 匹配失败
            else:  # 如果文本为空（是变量占位符）
                pos = comp.find(term["stop"])  # 在补全文本中查找停止条件的位置
                if pos != -1:  # 如果找到了停止条件
                    term["text"] = comp[:pos]  # 提取停止条件之前的文本作为变量值
                    comp = comp[pos:]  # 保留停止条件及之后的文本
                else:  # 如果没找到停止条件
                    if i == len(self.spec_format) - 1:  # 如果是最后一个格式项
                        term["text"] = comp  # 将剩余全部文本作为变量值
                    else:  # 如果不是最后一个格式项
                        return False  # 匹配失败
        return True  # 所有格式项都匹配成功

    def role_end_generate(  # 角色结束时的生成方法
        self,
        s: StreamExecutor,  # 流执行器
    ):
        """在角色标签结束时执行推测生成，将生成的文本和变量填充到执行器中"""
        if s.num_api_spec_tokens is None or not s.text_.endswith(self.chat_prefix):  # 如果未启用推测执行或文本不以聊天前缀结尾
            return  # 直接返回

        comp = ""  # 初始化补全结果
        if not all(x["name"] is None for x in self.spec_format):  # 如果格式列表中存在需要提取的变量
            # TODO(ying): throw errors or warnings  # TODO: 抛出错误或警告
            for i in range(self.spec_max_num_tries):  # 最多重试指定次数
                comp = openai_completion(  # 调用OpenAI补全函数
                    client=self.client,  # 传入客户端
                    token_usage=self.token_usage,  # 传入token使用量追踪器
                    is_chat=self.is_chat_model,  # 传入是否为聊天模型
                    model=self.model_name,  # 传入模型名称
                    prompt=s.messages_,  # 传入消息列表
                    **self.spec_kwargs,  # 传入推测执行参数
                )
                # Use a string for pattern matching.  # 使用字符串进行模式匹配
                comp_for_match = comp[0] if isinstance(comp, list) else comp  # 如果结果是列表则取第一个元素，否则直接使用
                if self.spec_pattern_match(comp_for_match):  # 如果模式匹配成功
                    break  # 跳出重试循环

        for term in self.spec_format:  # 遍历格式列表中的每一项
            s.text_ += term["text"]  # 将生成的文本追加到执行器的文本中
            name = term["name"]  # 获取变量名
            if name is not None:  # 如果变量名不为空
                s.variables[name] = term["text"]  # 将变量值存储到执行器的变量字典中
                s.meta_info[name] = {}  # 初始化变量的元信息为空字典
                s.variable_event[name].set()  # 设置变量事件，通知等待的线程

        self.spec_kwargs = {}  # 清空推测执行参数
        self.spec_format = []  # 清空推测格式列表

    def generate_stream(  # 流式生成方法
        self,
        s: StreamExecutor,  # 流执行器
        sampling_params: SglSamplingParams,  # 采样参数
    ):
        """以流式方式生成文本，逐步返回生成结果"""
        if sampling_params.dtype is None:  # 如果没有数据类型约束
            if self.is_chat_model:  # 如果是聊天模型
                if not s.text_.endswith(self.chat_prefix):  # 如果文本不以聊天前缀结尾
                    raise RuntimeError(  # 抛出运行时错误
                        "This use case is not supported. "  # 不支持此用例
                        "For OpenAI chat models, sgl.gen must be right after sgl.assistant"  # 对于OpenAI聊天模型，sgl.gen必须紧跟在sgl.assistant之后
                    )
                prompt = s.messages_  # 使用消息列表作为提示
            else:  # 如果不是聊天模型
                prompt = s.text_  # 使用文本作为提示

            kwargs = sampling_params.to_openai_kwargs()  # 将采样参数转换为OpenAI API参数
            generator = openai_completion_stream(  # 调用OpenAI流式补全函数
                client=self.client,  # 传入客户端
                token_usage=self.token_usage,  # 传入token使用量追踪器
                is_chat=self.is_chat_model,  # 传入是否为聊天模型
                model=self.model_name,  # 传入模型名称
                prompt=prompt,  # 传入提示
                **kwargs,  # 传入其他API参数
            )
            return generator  # 返回流式生成器
        else:  # 如果有数据类型约束
            raise ValueError(f"Unknown dtype: {sampling_params.dtype}")  # 抛出值错误

    def select(  # 选项选择方法
        self,
        s: StreamExecutor,  # 流执行器
        choices: List[str],  # 选项列表
        temperature: float,  # 采样温度
        choices_method: ChoicesSamplingMethod,  # 选项采样方法
    ) -> ChoicesDecision:  # 返回选项决策结果
        """Note: `choices_method` is not used by the OpenAI backend."""  # 注意：choices_method不被OpenAI后端使用
        """通过逐步解码和logit偏置操纵，从给定选项中选择最匹配的一个"""
        if self.is_chat_model:  # 如果是聊天模型
            raise NotImplementedError(  # 抛出未实现错误
                "select/choices is not supported for chat models. "  # 聊天模型不支持select/choices
                "Please try to use a non-chat model such as gpt-3.5-turbo-instruct"  # 请尝试使用非聊天模型如gpt-3.5-turbo-instruct
            )

        n_choices = len(choices)  # 获取选项数量
        token_ids = [self.tokenizer.encode(x) for x in choices]  # 将每个选项编码为token ID列表
        scores = [0] * n_choices  # 初始化每个选项的匹配分数为零
        valid = [len(x) > 0 for x in token_ids]  # 标记每个选项是否有有效的token编码
        prompt_tokens = self.tokenizer.encode(s.text_)  # 将提示文本编码为token ID列表

        max_len = max([len(x) for x in token_ids])  # 找到选项中最长的token序列长度
        for step in range(max_len):  # 逐步解码，每步处理一个token
            # Build logit bias  # 构建logit偏置
            logit_bias = {}  # 初始化logit偏置字典
            for i in range(n_choices):  # 遍历每个选项
                if valid[i]:  # 如果该选项仍然有效
                    logit_bias[token_ids[i][step]] = 100  # 将当前步对应的token ID加入偏置，值为100

            # Call API  # 调用API
            ret = self.client.completions.create(  # 调用补全API
                model=self.model_name,  # 指定模型名称
                prompt=prompt_tokens,  # 指定提示token
                logit_bias=logit_bias,  # 指定logit偏置
                max_tokens=1,  # 每次只生成1个token
                temperature=temperature,  # 指定采样温度
            )
            ret_str = ret.choices[0].text  # 获取返回的第一个选择文本
            ret_token = self.tokenizer.encode(ret_str)[0]  # 将返回文本编码为token并取第一个
            self.token_usage.prompt_tokens += ret.usage.prompt_tokens  # 累加提示token使用量
            self.token_usage.completion_tokens = ret.usage.completion_tokens  # 更新补全token使用量

            # TODO:  # 待办事项
            # 1. return logits as the scores  # 1. 返回logits作为分数
            # 2. compute logits of the full choice  # 2. 计算完整选项的logits
            # 3. consider chunk-based decoding  # 3. 考虑基于块的解码

            # Update valid  # 更新有效标记
            hit = False  # 初始化命中标记为False
            for i in range(n_choices):  # 遍历每个选项
                if valid[i]:  # 如果该选项仍然有效
                    if step == len(token_ids[i]) - 1:  # 如果已到达该选项token序列的末尾
                        valid[i] = False  # 将该选项标记为无效

                    if ret_token == token_ids[i][step]:  # 如果返回的token与该选项当前步的token匹配
                        scores[i] += 1  # 该选项分数加1
                        hit = True  # 标记为命中
                    else:  # 如果不匹配
                        valid[i] = False  # 将该选项标记为无效
            assert hit  # 断言至少有一个选项命中

            if np.sum(valid) <= 1:  # 如果有效选项数量不超过1
                break  # 跳出循环

            prompt_tokens.append(ret_token)  # 将返回的token追加到提示token列表中

        return ChoicesDecision(  # 返回选项决策结果
            decision=choices[np.argmax(scores)],  # 选择分数最高的选项作为决策
            meta_info={"scores": scores},  # 将所有分数作为元信息返回
        )


def openai_completion(  # OpenAI补全函数
    client, token_usage, is_chat=None, retries=3, prompt=None, **kwargs  # 客户端、token使用量、是否聊天、重试次数、提示、其他参数
) -> Union[str, List[str]]:  # 返回字符串或字符串列表
    """调用OpenAI API进行文本补全，支持聊天和补全两种模式，包含重试逻辑"""
    # if "ebnf" is in kwargs, warn and remove  # 如果参数中包含ebnf，发出警告并移除
    if "ebnf" in kwargs:  # 如果参数中包含ebnf
        warnings.warn("EBNF is not officially supported by OpenAI endpoints. Ignoring.")  # 发出警告：EBNF不被OpenAI端点官方支持
        del kwargs["ebnf"]  # 从参数中移除ebnf

    for attempt in range(retries):  # 重试循环
        try:  # 尝试调用API
            if is_chat:  # 如果是聊天模式
                if "stop" in kwargs and kwargs["stop"] is None:  # 如果stop参数为None
                    kwargs.pop("stop")  # 移除stop参数
                ret = client.chat.completions.create(messages=prompt, **kwargs)  # 调用聊天补全API
                if len(ret.choices) == 1:  # 如果只有一个选择
                    comp = ret.choices[0].message.content  # 获取消息内容
                else:  # 如果有多个选择
                    comp = [c.message.content for c in ret.choices]  # 获取所有选择的消息内容列表
            else:  # 如果是补全模式
                ret = client.completions.create(prompt=prompt, **kwargs)  # 调用补全API
                if isinstance(prompt, (list, tuple)):  # 如果提示是列表或元组（批量请求）
                    comp = [c.text for c in ret.choices]  # 获取所有选择的文本列表
                else:  # 单个请求
                    comp = ret.choices[0].text  # 获取第一个选择的文本
                    if len(ret.choices) > 1:  # 如果有多个选择
                        comp = [c.text for c in ret.choices]  # 获取所有选择的文本列表

            token_usage.prompt_tokens += ret.usage.prompt_tokens  # 累加提示token使用量
            token_usage.completion_tokens += ret.usage.completion_tokens  # 累加补全token使用量
            break  # 成功则跳出重试循环
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:  # 捕获OpenAI API相关异常
            logger.error(f"OpenAI Error: {e}. Waiting 5 seconds...")  # 记录错误日志
            time.sleep(5)  # 等待5秒后重试
            if attempt == retries - 1:  # 如果是最后一次重试
                raise e  # 抛出异常
        except Exception as e:  # 捕获其他异常
            logger.error(f"RuntimeError {e}.")  # 记录错误日志
            raise e  # 直接抛出异常

    return comp  # 返回补全结果


def openai_completion_stream(  # OpenAI流式补全函数
    client, token_usage, is_chat=None, retries=3, prompt=None, **kwargs  # 客户端、token使用量、是否聊天、重试次数、提示、其他参数
):
    """以流式方式调用OpenAI API进行文本补全，逐步生成并返回结果"""
    # if "ebnf" is in kwargs, warn and remove  # 如果参数中包含ebnf，发出警告并移除
    if "ebnf" in kwargs:  # 如果参数中包含ebnf
        warnings.warn("EBNF is not officially supported by OpenAI endpoints. Ignoring.")  # 发出警告：EBNF不被OpenAI端点官方支持
        del kwargs["ebnf"]  # 从参数中移除ebnf

    for attempt in range(retries):  # 重试循环
        try:  # 尝试调用API
            if is_chat:  # 如果是聊天模式
                if "stop" in kwargs and kwargs["stop"] is None:  # 如果stop参数为None
                    kwargs.pop("stop")  # 移除stop参数
                generator = client.chat.completions.create(  # 调用聊天补全API（流式）
                    messages=prompt,  # 传入消息列表
                    stream=True,  # 启用流式返回
                    stream_options={"include_usage": True},  # 在流中包含使用量信息
                    **kwargs,  # 传入其他API参数
                )
                for ret in generator:  # 遍历流式返回的每个结果
                    if len(ret.choices) == 0:  # 如果没有选择项
                        continue  # 跳过本次迭代
                    try:  # 尝试获取增量内容
                        content = ret.choices[0].delta.content  # 获取增量内容
                    except IndexError:  # 如果索引越界
                        content = None  # 内容设为None
                    yield content or "", {}  # 生成内容（空则返回空字符串）和空元信息字典
            else:  # 如果是补全模式
                generator = client.completions.create(  # 调用补全API（流式）
                    prompt=prompt,  # 传入提示
                    stream=True,  # 启用流式返回
                    stream_options={"include_usage": True},  # 在流中包含使用量信息
                    **kwargs,  # 传入其他API参数
                )
                for ret in generator:  # 遍历流式返回的每个结果
                    if len(ret.choices) == 0:  # 如果没有选择项
                        continue  # 跳过本次迭代
                    content = ret.choices[0].text  # 获取选择文本
                    yield content or "", {}  # 生成内容（空则返回空字符串）和空元信息字典

            token_usage.prompt_tokens += ret.usage.prompt_tokens  # 累加提示token使用量
            token_usage.completion_tokens += ret.usage.completion_tokens  # 累加补全token使用量
            break  # 成功则跳出重试循环
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:  # 捕获OpenAI API相关异常
            logger.error(f"OpenAI Error: {e}. Waiting 5 seconds...")  # 记录错误日志
            time.sleep(5)  # 等待5秒后重试
            if attempt == retries - 1:  # 如果是最后一次重试
                raise e  # 抛出异常
        except Exception as e:  # 捕获其他异常
            logger.error(f"RuntimeError {e}.")  # 记录错误日志
            raise e  # 直接抛出异常
