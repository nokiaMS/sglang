# 文件说明：本文件定义了Crusoe云平台推理服务的SGLang后端，继承自OpenAI后端，
# Crusoe暴露了与OpenAI兼容的API，因此这是一个薄封装，处理Crusoe特定的默认配置。

import os  # 导入操作系统模块
from typing import Optional  # 导入可选类型注解

from sglang.lang.backend.openai import OpenAI  # 导入OpenAI后端
from sglang.lang.chat_template import ChatTemplate  # 导入聊天模板类

CRUSOE_BASE_URL = "https://managed-inference-api-proxy.crusoecloud.com/v1/"  # Crusoe API的基础URL


class Crusoe(OpenAI):
    """SGLang backend for Crusoe managed inference.

    Crusoe exposes an OpenAI-compatible API, so this is a thin wrapper
    around the OpenAI backend that handles Crusoe-specific defaults.

    Args:
        model_name: The model to use, e.g. "meta-llama/Llama-3.1-8B-Instruct".
        api_key: Crusoe API key. Defaults to CRUSOE_API_KEY env var.
        base_url: Override the Crusoe endpoint. Defaults to the Crusoe API.
        chat_template: Optional custom chat template.
    """
    # Crusoe托管推理的SGLang后端。
    # Crusoe暴露了与OpenAI兼容的API，因此这是OpenAI后端的薄封装，处理Crusoe特定的默认值。
    # 参数:
    #   model_name: 要使用的模型，例如"meta-llama/Llama-3.1-8B-Instruct"。
    #   api_key: Crusoe API密钥。默认使用CRUSOE_API_KEY环境变量。
    #   base_url: 覆盖Crusoe端点。默认为Crusoe API。
    #   chat_template: 可选的自定义聊天模板。

    def __init__(
        self,
        model_name: str,  # 模型名称
        api_key: Optional[str] = None,  # API密钥，可选
        base_url: Optional[str] = None,  # 基础URL，可选
        chat_template: Optional[ChatTemplate] = None,  # 聊天模板，可选
        **kwargs,  # 其他关键字参数
    ):
        resolved_api_key = api_key or os.environ.get("CRUSOE_API_KEY")  # 解析API密钥，优先使用传入值，否则从环境变量获取
        if not resolved_api_key:  # 如果没有可用的API密钥
            raise ValueError(  # 抛出值错误
                "Crusoe API key required. Pass api_key= or set CRUSOE_API_KEY."
            )

        super().__init__(  # 调用父类OpenAI的初始化方法
            model_name=model_name,  # 传递模型名称
            chat_template=chat_template,  # 传递聊天模板
            api_key=resolved_api_key,  # 传递解析后的API密钥
            base_url=base_url or CRUSOE_BASE_URL,  # 传递基础URL，默认为Crusoe API地址
            **kwargs,  # 传递其他关键字参数
        )
