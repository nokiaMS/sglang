# 本文件实现了 RuntimeEndpoint 后端（SGLang 服务器的 HTTP 客户端）和 Runtime 类（在子进程中启动 SGLang 服务器的封装器）。
# 支持文本生成、流式生成、选项选择、KV 缓存操作以及异步生成等功能。

import atexit  # 导入 atexit 模块，用于注册程序退出时的回调函数
import json  # 导入 json 模块，用于 JSON 序列化和反序列化
import multiprocessing  # 导入 multiprocessing 模块，用于创建子进程启动服务器
import time  # 导入 time 模块，用于时间相关操作（如超时等待）
import warnings  # 导入 warnings 模块，用于发出警告信息
from typing import Dict, List, Optional, Union  # 从 typing 模块导入类型注解工具

import aiohttp  # 导入 aiohttp 模块，用于异步 HTTP 请求
import requests  # 导入 requests 模块，用于同步 HTTP 请求

from sglang.global_config import global_config  # 导入全局配置对象
from sglang.lang.backend.base_backend import BaseBackend  # 导入基类 BaseBackend
from sglang.lang.chat_template import get_chat_template, get_chat_template_by_model_path  # 导入聊天模板获取函数
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod  # 导入选项决策和采样方法类
from sglang.lang.interpreter import StreamExecutor  # 导入流执行器类
from sglang.lang.ir import (  # 从中间表示模块导入正则表达式常量和采样参数类
    REGEX_BOOL,  # 布尔类型的正则表达式
    REGEX_FLOAT,  # 浮点数类型的正则表达式
    REGEX_INT,  # 整数类型的正则表达式
    REGEX_STR,  # 字符串类型的正则表达式
    SglSamplingParams,  # SGLang 采样参数类
)
from sglang.utils import http_request  # 导入 HTTP 请求工具函数


class RuntimeEndpoint(BaseBackend):  # RuntimeEndpoint 类，继承自 BaseBackend，作为 SGLang 服务器的 HTTP 客户端
    def __init__(  # 初始化方法
        self,  # 实例自身
        base_url: str,  # 服务器的基础 URL 地址
        api_key: Optional[str] = None,  # 可选的 API 密钥
        verify: Optional[str] = None,  # 可选的 SSL 证书验证路径
        chat_template_name: Optional[str] = None,  # 可选的聊天模板名称
    ):  # 初始化方法参数结束
        super().__init__()  # 调用父类的初始化方法
        self.support_concate_and_append = True  # 标记支持连接和追加操作

        self.base_url = base_url  # 保存基础 URL
        self.api_key = api_key  # 保存 API 密钥
        self.verify = verify  # 保存 SSL 验证路径

        res = http_request(  # 发送 HTTP 请求获取模型信息
            self.base_url + "/get_model_info",  # 请求的完整 URL 路径
            api_key=self.api_key,  # 附带 API 密钥
            verify=self.verify,  # 附带 SSL 验证路径
        )  # HTTP 请求结束
        self._assert_success(res)  # 断言请求成功
        self.model_info = res.json()  # 将响应解析为 JSON 并保存模型信息

        if chat_template_name:  # 如果指定了聊天模板名称
            self.chat_template = get_chat_template(chat_template_name)  # 根据名称获取聊天模板
        else:  # 否则
            self.chat_template = get_chat_template_by_model_path(  # 根据模型路径获取聊天模板
                self.model_info["model_path"]  # 使用模型信息中的模型路径
            )  # 获取聊天模板结束

    def get_model_name(self):  # 获取模型名称的方法
        return self.model_info["model_path"]  # 返回模型信息中的模型路径

    def flush_cache(self):  # 刷新（清空）KV 缓存的方法
        res = http_request(  # 发送 HTTP 请求刷新缓存
            self.base_url + "/flush_cache",  # 请求的完整 URL 路径
            api_key=self.api_key,  # 附带 API 密钥
            verify=self.verify,  # 附带 SSL 验证路径
            method="POST",  # 使用 POST 方法
        )  # HTTP 请求结束
        self._assert_success(res)  # 断言请求成功

    def get_server_info(self):  # 获取服务器信息的方法
        res = http_request(  # 发送 HTTP 请求获取服务器信息
            self.base_url + "/server_info",  # 请求的完整 URL 路径
            api_key=self.api_key,  # 附带 API 密钥
            verify=self.verify,  # 附带 SSL 验证路径
        )  # HTTP 请求结束
        self._assert_success(res)  # 断言请求成功
        return res.json()  # 返回解析后的 JSON 响应

    def get_chat_template(self):  # 获取聊天模板的方法
        return self.chat_template  # 返回当前使用的聊天模板

    def cache_prefix(self, prefix_str: str):  # 缓存前缀字符串的方法
        res = http_request(  # 发送 HTTP 请求缓存前缀
            self.base_url + "/generate",  # 请求的完整 URL 路径
            json={"text": prefix_str, "sampling_params": {"max_new_tokens": 0}},  # 请求体：文本为前缀字符串，最大生成令牌数为0（仅缓存不生成）
            api_key=self.api_key,  # 附带 API 密钥
            verify=self.verify,  # 附带 SSL 验证路径
        )  # HTTP 请求结束
        self._assert_success(res)  # 断言请求成功

    def start_profile(self):  # 启动性能分析的方法
        res = http_request(  # 发送 HTTP 请求启动性能分析
            self.base_url + "/start_profile",  # 请求的完整 URL 路径
            api_key=self.api_key,  # 附带 API 密钥
            verify=self.verify,  # 附带 SSL 验证路径
        )  # HTTP 请求结束
        self._assert_success(res)  # 断言请求成功

    def stop_profile(self):  # 停止性能分析的方法
        res = http_request(  # 发送 HTTP 请求停止性能分析
            self.base_url + "/stop_profile",  # 请求的完整 URL 路径
            api_key=self.api_key,  # 附带 API 密钥
            verify=self.verify,  # 附带 SSL 验证路径
        )  # HTTP 请求结束
        self._assert_success(res)  # 断言请求成功

    def commit_lazy_operations(self, s: StreamExecutor):  # 提交惰性操作的方法
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}  # 构造请求数据：文本和采样参数（最大生成令牌数为0）
        self._add_images(s, data)  # 添加图像数据到请求中
        res = http_request(  # 发送 HTTP 请求提交惰性操作
            self.base_url + "/generate",  # 请求的完整 URL 路径
            json=data,  # 请求体
            api_key=self.api_key,  # 附带 API 密钥
            verify=self.verify,  # 附带 SSL 验证路径
        )  # HTTP 请求结束
        self._assert_success(res)  # 断言请求成功

    def fill_image(self, s: StreamExecutor):  # 填充图像数据的方法
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}  # 构造请求数据：文本和采样参数（最大生成令牌数为0）
        self._add_images(s, data)  # 添加图像数据到请求中
        res = http_request(  # 发送 HTTP 请求填充图像
            self.base_url + "/generate",  # 请求的完整 URL 路径
            json=data,  # 请求体
            api_key=self.api_key,  # 附带 API 密钥
            verify=self.verify,  # 附带 SSL 验证路径
        )  # HTTP 请求结束
        self._assert_success(res)  # 断言请求成功

    def _handle_dtype_to_regex(self, sampling_params: SglSamplingParams):  # 处理数据类型到正则表达式转换的内部方法
        if sampling_params.dtype is None:  # 如果未指定数据类型
            return  # 直接返回，不做处理

        if sampling_params.stop == ():  # 如果停止条件为空元组
            sampling_params.stop = []  # 将停止条件改为空列表

        dtype_regex = None  # 初始化数据类型对应的正则表达式为 None
        if sampling_params.dtype in ["int", int]:  # 如果数据类型为整数

            dtype_regex = REGEX_INT  # 使用整数正则表达式
            sampling_params.stop.extend([" ", "\n"])  # 添加空格和换行作为停止条件
        elif sampling_params.dtype in ["float", float]:  # 如果数据类型为浮点数

            dtype_regex = REGEX_FLOAT  # 使用浮点数正则表达式
            sampling_params.stop.extend([" ", "\n"])  # 添加空格和换行作为停止条件
        elif sampling_params.dtype in ["str", str]:  # 如果数据类型为字符串

            dtype_regex = REGEX_STR  # 使用字符串正则表达式
        elif sampling_params.dtype in ["bool", bool]:  # 如果数据类型为布尔值

            dtype_regex = REGEX_BOOL  # 使用布尔值正则表达式
        else:  # 其他不支持的数据类型
            raise RuntimeError(f"Invalid dtype: {sampling_params.dtype}")  # 抛出运行时异常

        if dtype_regex is not None and sampling_params.regex is not None:  # 如果同时指定了数据类型正则和自定义正则
            warnings.warn(  # 发出警告
                f"Both dtype and regex are set. Only dtype will be used. dtype: {sampling_params.dtype}, regex: {sampling_params.regex}"  # 警告内容：两者同时设置时仅使用 dtype
            )  # 警告结束

        sampling_params.regex = dtype_regex  # 将数据类型对应的正则表达式赋值给采样参数的 regex 字段

    def generate(  # 文本生成方法
        self,  # 实例自身
        s: StreamExecutor,  # 流执行器对象
        sampling_params: SglSamplingParams,  # 采样参数对象
    ):  # 生成方法参数结束
        self._handle_dtype_to_regex(sampling_params)  # 处理数据类型到正则表达式的转换
        data = {  # 构造请求数据字典
            "text": s.text_,  # 文本内容
            "sampling_params": {  # 采样参数子字典
                "skip_special_tokens": global_config.skip_special_tokens_in_output,  # 是否跳过输出中的特殊令牌
                "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,  # 特殊令牌之间是否添加空格
                **sampling_params.to_srt_kwargs(),  # 将采样参数转换为 SRT 格式的关键字参数并合并
            },  # 采样参数子字典结束
        }  # 请求数据字典结束

        for item in [  # 遍历需要额外处理的字段列表
            "return_logprob",  # 是否返回对数概率
            "logprob_start_len",  # 对数概率起始长度
            "top_logprobs_num",  # 返回的顶部对数概率数量
            "return_text_in_logprobs",  # 是否在对数概率中返回文本
        ]:  # 字段列表结束
            value = getattr(sampling_params, item, None)  # 从采样参数中获取该字段的值，默认为 None
            if value is not None:  # 如果值不为 None
                data[item] = value  # 将该字段添加到请求数据中

        self._add_images(s, data)  # 添加图像数据到请求中

        res = http_request(  # 发送 HTTP 请求进行文本生成
            self.base_url + "/generate",  # 请求的完整 URL 路径
            json=data,  # 请求体
            api_key=self.api_key,  # 附带 API 密钥
            verify=self.verify,  # 附带 SSL 验证路径
        )  # HTTP 请求结束
        self._assert_success(res)  # 断言请求成功

        obj = res.json()  # 将响应解析为 JSON 对象
        comp = obj["text"]  # 获取生成的文本内容
        return comp, obj["meta_info"]  # 返回生成的文本和元信息

    def generate_stream(  # 流式文本生成方法
        self,  # 实例自身
        s: StreamExecutor,  # 流执行器对象
        sampling_params: SglSamplingParams,  # 采样参数对象
    ):  # 流式生成方法参数结束
        self._handle_dtype_to_regex(sampling_params)  # 处理数据类型到正则表达式的转换

        data = {  # 构造请求数据字典
            "text": s.text_,  # 文本内容
            "sampling_params": {  # 采样参数子字典
                "skip_special_tokens": global_config.skip_special_tokens_in_output,  # 是否跳过输出中的特殊令牌
                "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,  # 特殊令牌之间是否添加空格
                **sampling_params.to_srt_kwargs(),  # 将采样参数转换为 SRT 格式的关键字参数并合并
            },  # 采样参数子字典结束
        }  # 请求数据字典结束

        for item in [  # 遍历需要额外处理的字段列表
            "return_logprob",  # 是否返回对数概率
            "logprob_start_len",  # 对数概率起始长度
            "top_logprobs_num",  # 返回的顶部对数概率数量
            "return_text_in_logprobs",  # 是否在对数概率中返回文本
        ]:  # 字段列表结束
            value = getattr(sampling_params, item, None)  # 从采样参数中获取该字段的值，默认为 None
            if value is not None:  # 如果值不为 None
                data[item] = value  # 将该字段添加到请求数据中

        data["stream"] = True  # 设置流式传输标志为 True
        self._add_images(s, data)  # 添加图像数据到请求中

        res = http_request(  # 发送 HTTP 请求进行流式文本生成
            self.base_url + "/generate",  # 请求的完整 URL 路径
            json=data,  # 请求体
            stream=True,  # 启用流式传输
            api_key=self.api_key,  # 附带 API 密钥
            verify=self.verify,  # 附带 SSL 验证路径
        )  # HTTP 请求结束
        self._assert_success(res)  # 断言请求成功
        pos = 0  # 初始化当前位置指针，用于跟踪已处理的文本位置

        for chunk in res.iter_lines(decode_unicode=False):  # 逐行迭代响应中的流式数据
            chunk = chunk.decode("utf-8")  # 将字节块解码为 UTF-8 字符串
            if chunk and chunk.startswith("data:"):  # 如果数据块非空且以"data:"开头
                if chunk == "data: [DONE]":  # 如果收到结束标记
                    break  # 跳出循环，结束流式处理
                data = json.loads(chunk[5:].strip("\n"))  # 去掉"data:"前缀和换行符后解析 JSON 数据
                chunk_text = data["text"][pos:]  # 获取当前位置之后的增量文本
                meta_info = data["meta_info"]  # 获取元信息
                pos += len(chunk_text)  # 更新当前位置指针
                yield chunk_text, meta_info  # 生成器产出增量文本和元信息

    def select(  # 选项选择方法，根据对数概率从多个选项中选择最佳选项
        self,  # 实例自身
        s: StreamExecutor,  # 流执行器对象
        choices: List[str],  # 候选选项列表
        temperature: float,  # 采样温度
        choices_method: ChoicesSamplingMethod,  # 选项采样方法
    ) -> ChoicesDecision:  # 返回选项决策结果
        assert temperature <= 1e-5  # 断言温度必须非常低（接近贪心选择）

        # Cache common prefix  # 缓存公共前缀
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}  # 构造请求数据：文本和采样参数（最大生成令牌数为0，仅缓存前缀）
        obj = self._generate_http_request(s, data)  # 发送 HTTP 请求并获取响应
        prompt_len = obj["meta_info"]["prompt_tokens"]  # 获取提示词的令牌数量
        logprob_start_len = max(prompt_len - 2, 0)  # For token healing  # 计算对数概率的起始长度，最多回退2个令牌用于令牌修复

        # Compute logprob  # 计算对数概率
        data = {  # 构造请求数据字典
            "text": [s.text_ + c for c in choices],  # 将每个选项拼接到提示文本后，形成多个输入
            "sampling_params": {  # 采样参数子字典
                "max_new_tokens": 0,  # 最大生成令牌数为0（不生成新令牌）
                "temperature": 0,  # 温度为0（贪心选择）
            },  # 采样参数子字典结束
            "return_logprob": True,  # 请求返回对数概率
            "return_text_in_logprobs": True,  # 请求在对数概率中返回文本
            "logprob_start_len": logprob_start_len,  # 对数概率的起始长度
        }  # 请求数据字典结束
        obj = self._generate_http_request(s, data)  # 发送 HTTP 请求并获取响应

        input_token_logprobs = [r["meta_info"]["input_token_logprobs"] for r in obj]  # 提取每个选项的输入令牌对数概率
        output_token_logprobs = [r["meta_info"]["output_token_logprobs"] for r in obj]  # 提取每个选项的输出令牌对数概率
        normalized_prompt_logprobs = [  # 计算每个选项的归一化提示对数概率
            compute_normalized_prompt_logprobs(r["meta_info"]["input_token_logprobs"])  # 调用辅助函数计算归一化提示对数概率
            for r in obj  # 遍历每个选项的响应
        ]  # 归一化提示对数概率列表结束

        # Remove extra token if no token healing occurred  # 如果没有发生令牌修复，则移除多余的令牌
        for i in range(len(input_token_logprobs)):  # 遍历每个选项的输入令牌对数概率
            healed_token_str = input_token_logprobs[i][0][-1]  # 获取修复令牌的字符串表示
            if s.text_.endswith(healed_token_str):  # 如果提示文本以修复令牌结尾（说明没有发生令牌修复）
                healed_token_logprob = input_token_logprobs[i][0][0]  # 获取修复令牌的对数概率
                normalized_prompt_logprobs[i] = (  # 重新计算归一化提示对数概率
                    normalized_prompt_logprobs[i] * len(input_token_logprobs[i])  # 先乘以令牌数量恢复总和
                    - healed_token_logprob  # 减去修复令牌的对数概率
                ) / (len(input_token_logprobs[i]) - 1)  # 再除以令牌数量减1得到新的平均值
                input_token_logprobs[i] = input_token_logprobs[i][1:]  # 移除第一个令牌（修复令牌）

        # Compute unconditional logprobs if required  # 如果需要，计算无条件对数概率
        if choices_method.requires_unconditional_logprobs:  # 如果选项方法需要无条件对数概率
            input_ids = [[el[1] for el in subl] for subl in input_token_logprobs]  # 从输入令牌对数概率中提取令牌 ID
            data = {  # 构造请求数据字典
                "input_ids": input_ids,  # 输入令牌 ID 列表
                "sampling_params": {"max_new_tokens": 0},  # 最大生成令牌数为0
                "return_logprob": True,  # 请求返回对数概率
            }  # 请求数据字典结束
            obj = self._generate_http_request(s, data)  # 发送 HTTP 请求并获取响应
            unconditional_token_logprobs = [  # 提取无条件对数概率
                r["meta_info"]["input_token_logprobs"] for r in obj  # 从每个响应中获取输入令牌对数概率
            ]  # 无条件对数概率列表结束
        else:  # 否则
            unconditional_token_logprobs = None  # 无条件对数概率设为 None

        return choices_method(  # 调用选项采样方法进行决策
            choices=choices,  # 候选选项列表
            normalized_prompt_logprobs=normalized_prompt_logprobs,  # 归一化提示对数概率
            input_token_logprobs=input_token_logprobs,  # 输入令牌对数概率
            output_token_logprobs=output_token_logprobs,  # 输出令牌对数概率
            unconditional_token_logprobs=unconditional_token_logprobs,  # 无条件对数概率
        )  # 返回选项决策结果

    def concatenate_and_append(self, src_rids: List[str], dst_rid: str):  # 连接并追加请求的方法，将多个源请求的 KV 缓存合并到目标请求
        res = http_request(  # 发送 HTTP 请求连接并追加
            self.base_url + "/concate_and_append_request",  # 请求的完整 URL 路径
            json={"src_rids": src_rids, "dst_rid": dst_rid},  # 请求体：源请求 ID 列表和目标请求 ID
            api_key=self.api_key,  # 附带 API 密钥
            verify=self.verify,  # 附带 SSL 验证路径
        )  # HTTP 请求结束
        self._assert_success(res)  # 断言请求成功

    def _generate_http_request(self, s: StreamExecutor, data):  # 发送生成 HTTP 请求的内部方法
        self._add_images(s, data)  # 添加图像数据到请求中
        res = http_request(  # 发送 HTTP 请求
            self.base_url + "/generate",  # 请求的完整 URL 路径
            json=data,  # 请求体
            api_key=self.api_key,  # 附带 API 密钥
            verify=self.verify,  # 附带 SSL 验证路径
        )  # HTTP 请求结束
        self._assert_success(res)  # 断言请求成功
        return res.json()  # 返回解析后的 JSON 响应

    def _add_images(self, s: StreamExecutor, data):  # 添加图像数据到请求的内部方法
        if s.images_:  # 如果流执行器中存在图像数据
            assert len(s.images_) == 1, "Only support one image."  # 断言仅支持一张图像
            data["image_data"] = s.images_[0][1]  # 将图像数据添加到请求中（取第一张图像的数据部分）

    def _assert_success(self, res):  # 断言 HTTP 请求成功的内部方法
        if res.status_code != 200:  # 如果响应状态码不等于 200（请求失败）
            try:  # 尝试解析错误响应
                content = res.json()  # 将响应解析为 JSON
            except json.JSONDecodeError:  # 如果 JSON 解析失败
                content = res.text  # 使用响应的原始文本作为错误内容
            raise RuntimeError(content)  # 抛出运行时异常，包含错误内容


def compute_normalized_prompt_logprobs(input_logprobs):  # 计算归一化提示对数概率的辅助函数
    values = [x[0] for x in input_logprobs if x[0]]  # 提取所有非空的对数概率值
    return sum(values) / len(values)  # 返回对数概率的平均值


class Runtime:  # Runtime 类，封装 SGLang HTTP 服务器的启动和管理
    """
    A wrapper for the HTTP server.
    This is used for launching the server in a python program without
    using the command line interface.

    It is mainly used for the frontend language.
    You should use the Engine class if you want to do normal offline processing without the frontend language.
    """  # Runtime 类的文档字符串：HTTP 服务器的封装器，用于在 Python 程序中启动服务器，主要用于前端语言；如需离线处理请使用 Engine 类

    def __init__(  # 初始化方法
        self,  # 实例自身
        log_level: str = "error",  # 服务器的日志级别，默认为 "error"
        launch_timeout: float = 300.0,  # 服务器启动超时时间（秒），默认为 300 秒
        *args,  # 额外的位置参数，传递给 ServerArgs
        **kwargs,  # 额外的关键字参数，传递给 ServerArgs
    ):  # 初始化方法参数结束
        """See the arguments in server_args.py::ServerArgs

        Args:
            log_level: Log level for the server.
            timeout: Timeout in seconds for waiting for the server to start.
            *args: Additional arguments passed to ServerArgs.
            **kwargs: Additional keyword arguments passed to ServerArgs.
        """  # 文档字符串：详细参数请参考 server_args.py 中的 ServerArgs 类
        # We delay the import of any `sglang.srt` components in `sglang.lang`, so users can run
        # client code without installing SRT server and its dependency if they want.
        # 我们延迟导入任何 `sglang.srt` 组件到 `sglang.lang` 中，这样用户可以在不安装 SRT 服务器及其依赖的情况下运行客户端代码
        from sglang.srt.entrypoints.http_server import launch_server  # 导入服务器启动函数
        from sglang.srt.server_args import ServerArgs  # 导入服务器参数类
        from sglang.srt.utils.network import is_port_available  # 导入端口可用性检查函数

        self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)  # 创建服务器参数对象

        # Pre-allocate ports  # 预分配端口
        for port in range(self.server_args.port, 40000):  # 从指定端口开始遍历到 40000
            if is_port_available(port):  # 如果端口可用
                break  # 跳出循环
        self.server_args.port = port  # 将找到的可用端口设置到服务器参数中

        self.url = self.server_args.url()  # 获取服务器的 URL
        self.generate_url = self.url + "/generate"  # 构造生成接口的完整 URL

        # NOTE: We store pid instead of proc to fix some issues during __delete__
        # 注意：我们存储进程 ID 而非进程对象，以修复 __delete__ 中的一些问题
        self.pid = None  # 初始化进程 ID 为 None

        ctx = multiprocessing.get_context("spawn")  # 获取 "spawn" 方式的多进程上下文（避免 fork 方式的问题）
        proc = ctx.Process(  # 创建子进程
            target=launch_server,  # 子进程的目标函数为服务器启动函数
            args=(self.server_args,),  # 传递服务器参数作为参数
        )  # 子进程创建结束
        proc.start()  # 启动子进程
        self.pid = proc.pid  # 保存子进程的进程 ID

        # Before python program terminates, call shutdown implicitly. Therefore, users don't have to explicitly call .shutdown()
        # 在 Python 程序终止之前，隐式调用 shutdown。因此用户不必显式调用 .shutdown()
        atexit.register(self.shutdown)  # 注册 shutdown 方法为退出时的回调函数

        # Wait for server to be ready by polling /health_generate
        # 通过轮询 /health_generate 端点等待服务器就绪
        start_time = time.time()  # 记录开始等待的时间
        with requests.Session() as session:  # 创建 HTTP 会话
            while time.time() - start_time < launch_timeout:  # 在超时时间内循环等待
                try:  # 尝试发送健康检查请求
                    response = session.get(f"{self.url}/health_generate")  # 发送 GET 请求到健康检查端点
                    if response.status_code == 200:  # 如果响应状态码为 200，表示服务器已就绪
                        break  # 跳出循环
                except requests.RequestException:  # 捕获请求异常（服务器尚未启动）
                    pass  # 忽略异常，继续等待

                if not proc.is_alive():  # 如果子进程已经不在运行
                    self.shutdown()  # 调用 shutdown 清理资源
                    raise RuntimeError(  # 抛出运行时异常
                        "Initialization failed. Please see the error messages above."  # 初始化失败的错误消息
                    )  # 异常抛出结束

                time.sleep(2)  # 等待 2 秒后再次尝试
            else:  # 如果超时仍未就绪
                self.shutdown()  # 调用 shutdown 清理资源
                raise TimeoutError("Server failed to start within the timeout period.")  # 抛出超时异常

        self.endpoint = RuntimeEndpoint(self.url)  # 创建 RuntimeEndpoint 实例，连接到已启动的服务器

    def shutdown(self):  # 关闭服务器的方法
        from sglang.srt.utils import kill_process_tree  # 导入终止进程树的工具函数

        if self.pid is not None:  # 如果进程 ID 不为 None（服务器正在运行）
            kill_process_tree(self.pid)  # 终止整个进程树（包括子进程）
            self.pid = None  # 将进程 ID 重置为 None

    def start_profile(self):  # 启动性能分析的方法
        self.endpoint.start_profile()  # 委托给 RuntimeEndpoint 的 start_profile 方法

    def stop_profile(self):  # 停止性能分析的方法
        self.endpoint.stop_profile()  # 委托给 RuntimeEndpoint 的 stop_profile 方法

    def cache_prefix(self, prefix: str):  # 缓存前缀字符串的方法
        self.endpoint.cache_prefix(prefix)  # 委托给 RuntimeEndpoint 的 cache_prefix 方法

    def get_tokenizer(self):  # 获取分词器的方法
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer  # 导入 HuggingFace 分词器获取函数

        return get_tokenizer(  # 返回获取的分词器
            self.server_args.tokenizer_path,  # 分词器路径
            tokenizer_mode=self.server_args.tokenizer_mode,  # 分词器模式
            trust_remote_code=self.server_args.trust_remote_code,  # 是否信任远程代码
            revision=self.server_args.revision,  # 模型版本号
        )  # 返回分词器结束

    async def async_generate(  # 异步生成方法
        self,  # 实例自身
        prompt: str,  # 输入提示文本
        sampling_params: Optional[Dict] = None,  # 可选的采样参数字典
    ):  # 异步生成方法参数结束
        if self.server_args.skip_tokenizer_init:  # 如果服务器跳过了分词器初始化
            json_data = {  # 构造使用 input_ids 的请求数据
                "input_ids": prompt,  # 输入令牌 ID
                "sampling_params": sampling_params,  # 采样参数
                "stream": True,  # 启用流式传输
            }  # 请求数据字典结束
        else:  # 否则
            json_data = {  # 构造使用文本的请求数据
                "text": prompt,  # 输入文本
                "sampling_params": sampling_params,  # 采样参数
                "stream": True,  # 启用流式传输
            }  # 请求数据字典结束
        pos = 0  # 初始化当前位置指针，用于跟踪已处理的文本位置

        timeout = aiohttp.ClientTimeout(total=3 * 3600)  # 设置超时时间为 3 小时
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:  # 创建异步 HTTP 会话
            async with session.post(self.generate_url, json=json_data) as response:  # 发送异步 POST 请求
                async for chunk, _ in response.content.iter_chunks():  # 异步迭代响应内容的块
                    chunk = chunk.decode("utf-8")  # 将字节块解码为 UTF-8 字符串
                    if chunk and chunk.startswith("data:"):  # 如果数据块非空且以"data:"开头
                        if chunk == "data: [DONE]\n\n":  # 如果收到结束标记
                            break  # 跳出循环，结束异步流式处理
                        data = json.loads(chunk[5:].strip("\n"))  # 去掉"data:"前缀和换行符后解析 JSON 数据
                        if "text" in data:  # 如果数据中包含 "text" 字段
                            cur = data["text"][pos:]  # 获取当前位置之后的增量文本
                            if cur:  # 如果增量文本非空
                                yield cur  # 生成器产出增量文本
                            pos += len(cur)  # 更新当前位置指针
                        else:  # 否则（数据中不包含 "text" 字段）
                            yield data  # 直接生成器产出整个数据对象

    add_request = async_generate  # add_request 是 async_generate 的别名

    def generate(  # 同步生成方法
        self,  # 实例自身
        prompt: Union[str, List[str]],  # 输入提示文本，可以是单个字符串或字符串列表
        sampling_params: Optional[Dict] = None,  # 可选的采样参数字典
        return_logprob: Optional[Union[List[bool], bool]] = False,  # 是否返回对数概率，默认为 False
        logprob_start_len: Optional[Union[List[int], int]] = None,  # 对数概率起始长度
        top_logprobs_num: Optional[Union[List[int], int]] = None,  # 返回的顶部对数概率数量
        lora_path: Optional[List[Optional[str]]] = None,  # LoRA 适配器路径列表
    ):  # 同步生成方法参数结束
        json_data = {  # 构造请求数据字典
            "text": prompt,  # 输入文本
            "sampling_params": sampling_params,  # 采样参数
            "return_logprob": return_logprob,  # 是否返回对数概率
            "logprob_start_len": logprob_start_len,  # 对数概率起始长度
            "top_logprobs_num": top_logprobs_num,  # 顶部对数概率数量
            "lora_path": lora_path,  # LoRA 适配器路径
        }  # 请求数据字典结束
        assert not isinstance(lora_path, list) or len(lora_path) == len(prompt)  # 断言 LoRA 路径数量与提示数量一致
        response = requests.post(  # 发送同步 POST 请求
            self.url + "/generate",  # 请求的完整 URL 路径
            json=json_data,  # 请求体
        )  # POST 请求结束
        return json.dumps(response.json())  # 将响应 JSON 序列化后返回

    def encode(  # 编码方法，将文本编码为令牌 ID
        self,  # 实例自身
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],  # 输入提示，支持多种格式
    ):  # 编码方法参数结束
        json_data = {"text": prompt}  # 构造请求数据，包含文本
        response = requests.post(self.url + "/encode", json=json_data)  # 发送 POST 请求到编码端点
        return json.dumps(response.json())  # 将响应 JSON 序列化后返回

    async def get_server_info(self):  # 异步获取服务器信息的方法
        async with aiohttp.ClientSession() as session:  # 创建异步 HTTP 会话
            async with session.get(f"{self.url}/server_info") as response:  # 发送异步 GET 请求
                if response.status == 200:  # 如果响应状态码为 200（成功）
                    return await response.json()  # 返回解析后的 JSON 响应
                else:  # 否则（请求失败）
                    error_data = await response.json()  # 解析错误响应的 JSON 数据
                    raise RuntimeError(  # 抛出运行时异常
                        f"Failed to get server info. {error_data['error']['message']}"  # 错误消息
                    )  # 异常抛出结束

    def __del__(self):  # 析构方法，在对象被垃圾回收时调用
        self.shutdown()  # 调用 shutdown 方法关闭服务器
