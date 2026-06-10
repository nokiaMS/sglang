# 文件说明：本文件定义了Anthropic（Claude）模型的SGLang后端，实现了文本生成和流式生成功能，
# 通过Anthropic Python SDK与Claude API交互。

from sglang.lang.backend.base_backend import BaseBackend  # 导入后端基类
from sglang.lang.chat_template import get_chat_template  # 导入获取聊天模板函数
from sglang.lang.interpreter import StreamExecutor  # 导入流执行器
from sglang.lang.ir import SglSamplingParams  # 导入采样参数类

try:  # 尝试导入anthropic库
    import anthropic  # 导入Anthropic Python SDK
except ImportError as e:  # 如果导入失败
    anthropic = e  # 将异常保存，后续在初始化时检查


class Anthropic(BaseBackend):
    """Anthropic（Claude）模型的SGLang后端，支持文本生成和流式生成。"""
    def __init__(self, model_name, *args, **kwargs):  # 初始化方法
        super().__init__()  # 调用父类初始化

        if isinstance(anthropic, Exception):  # 检查anthropic是否导入失败
            raise anthropic  # 如果失败则抛出异常

        self.model_name = model_name  # 保存模型名称
        self.chat_template = get_chat_template("claude")  # 获取Claude聊天模板
        self.client = anthropic.Anthropic(*args, **kwargs)  # 创建Anthropic客户端

    def get_chat_template(self):  # 获取聊天模板
        """获取Claude专用的聊天模板。"""
        return self.chat_template  # 返回Claude聊天模板

    def generate(  # 生成文本
        self,
        s: StreamExecutor,  # 流执行器
        sampling_params: SglSamplingParams,  # 采样参数
    ):
        """调用Anthropic API生成文本，返回生成结果和空元信息。"""
        if s.messages_:  # 如果存在消息列表
            messages = s.messages_  # 使用消息列表
        else:  # 否则
            messages = [{"role": "user", "content": s.text_}]  # 使用纯文本构造用户消息

        if messages and messages[0]["role"] == "system":  # 如果第一条消息是系统消息
            system = messages.pop(0)["content"]  # 提取系统消息内容
        else:  # 否则
            system = ""  # 使用空字符串

        ret = self.client.messages.create(  # 调用Anthropic API创建消息
            model=self.model_name,  # 模型名称
            system=system,  # 系统提示
            messages=messages,  # 消息列表
            **sampling_params.to_anthropic_kwargs(),  # 转换为Anthropic参数格式
        )
        comp = ret.content[0].text  # 提取生成文本

        return comp, {}  # 返回生成文本和空元信息

    def generate_stream(  # 流式生成文本
        self,
        s: StreamExecutor,  # 流执行器
        sampling_params: SglSamplingParams,  # 采样参数
    ):
        """流式调用Anthropic API生成文本，逐步返回生成结果。"""
        if s.messages_:  # 如果存在消息列表
            messages = s.messages_  # 使用消息列表
        else:  # 否则
            messages = [{"role": "user", "content": s.text_}]  # 使用纯文本构造用户消息

        if messages and messages[0]["role"] == "system":  # 如果第一条消息是系统消息
            system = messages.pop(0)["content"]  # 提取系统消息内容
        else:  # 否则
            system = ""  # 使用空字符串

        with self.client.messages.stream(  # 使用流式API
            model=self.model_name,  # 模型名称
            system=system,  # 系统提示
            messages=messages,  # 消息列表
            **sampling_params.to_anthropic_kwargs(),  # 转换为Anthropic参数格式
        ) as stream:  # 获取流对象
            for text in stream.text_stream:  # 遍历流中的文本
                yield text, {}  # 返回文本片段和空元信息
