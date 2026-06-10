# 文件说明：本文件定义了LiteLLM的SGLang后端，通过LiteLLM库统一调用多种LLM API，
# 支持文本生成和流式生成功能。

from typing import Mapping, Optional  # 导入类型注解

from sglang.lang.backend.base_backend import BaseBackend  # 导入后端基类
from sglang.lang.chat_template import get_chat_template_by_model_path  # 导入按模型路径获取聊天模板函数
from sglang.lang.interpreter import StreamExecutor  # 导入流执行器
from sglang.lang.ir import SglSamplingParams  # 导入采样参数类

try:  # 尝试导入litellm库
    import litellm  # 导入LiteLLM库
except ImportError as e:  # 如果导入失败
    litellm = e  # 将异常保存
    litellm.num_retries = 1  # 设置默认重试次数为1


class LiteLLM(BaseBackend):
    """LiteLLM的SGLang后端，通过LiteLLM库统一调用多种LLM API。"""
    def __init__(
        self,
        model_name,  # 模型名称
        chat_template=None,  # 聊天模板，可选
        api_key=None,  # API密钥，可选
        organization: Optional[str] = None,  # 组织，可选
        base_url: Optional[str] = None,  # 基础URL，可选
        timeout: Optional[float] = 600,  # 超时时间，默认600秒
        max_retries: Optional[int] = litellm.num_retries,  # 最大重试次数，默认使用litellm配置
        default_headers: Optional[Mapping[str, str]] = None,  # 默认请求头，可选
    ):
        super().__init__()  # 调用父类初始化

        if isinstance(litellm, Exception):  # 检查litellm是否导入失败
            raise litellm  # 如果失败则抛出异常

        self.model_name = model_name  # 保存模型名称

        self.chat_template = chat_template or get_chat_template_by_model_path(  # 设置聊天模板
            model_name  # 根据模型路径获取
        )

        self.client_params = {  # 客户端参数字典
            "api_key": api_key,  # API密钥
            "organization": organization,  # 组织
            "base_url": base_url,  # 基础URL
            "timeout": timeout,  # 超时时间
            "max_retries": max_retries,  # 最大重试次数
            "default_headers": default_headers,  # 默认请求头
        }

    def get_chat_template(self):  # 获取聊天模板
        """获取当前后端使用的聊天模板。"""
        return self.chat_template  # 返回聊天模板

    def generate(  # 生成文本
        self,
        s: StreamExecutor,  # 流执行器
        sampling_params: SglSamplingParams,  # 采样参数
    ):
        """调用LiteLLM生成文本，返回生成结果和空元信息。"""
        if s.messages_:  # 如果存在消息列表
            messages = s.messages_  # 使用消息列表
        else:  # 否则
            messages = [{"role": "user", "content": s.text_}]  # 使用纯文本构造用户消息

        ret = litellm.completion(  # 调用LiteLLM完成API
            model=self.model_name,  # 模型名称
            messages=messages,  # 消息列表
            **self.client_params,  # 客户端参数
            **sampling_params.to_litellm_kwargs(),  # 转换为LiteLLM参数格式
        )
        comp = ret.choices[0].message.content  # 提取生成文本

        return comp, {}  # 返回生成文本和空元信息

    def generate_stream(  # 流式生成文本
        self,
        s: StreamExecutor,  # 流执行器
        sampling_params: SglSamplingParams,  # 采样参数
    ):
        """流式调用LiteLLM生成文本，逐步返回生成结果。"""
        if s.messages_:  # 如果存在消息列表
            messages = s.messages_  # 使用消息列表
        else:  # 否则
            messages = [{"role": "user", "content": s.text_}]  # 使用纯文本构造用户消息

        ret = litellm.completion(  # 调用LiteLLM完成API（流式）
            model=self.model_name,  # 模型名称
            messages=messages,  # 消息列表
            stream=True,  # 启用流式模式
            **self.client_params,  # 客户端参数
            **sampling_params.to_litellm_kwargs(),  # 转换为LiteLLM参数格式
        )
        for chunk in ret:  # 遍历流中的每个块
            text = chunk.choices[0].delta.content  # 提取增量文本内容
            if text is not None:  # 如果文本不为None
                yield text, {}  # 返回文本片段和空元信息
