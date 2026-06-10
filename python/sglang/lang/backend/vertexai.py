# VertexAI 后端实现模块
# 本模块实现了 SGLang 的 VertexAI 后端，支持文本生成、流式生成，
# 以及 OpenAI 消息格式与 VertexAI 消息格式之间的转换。

import os  # 导入操作系统模块，用于读取环境变量
import warnings  # 导入警告模块，用于发出警告信息

from sglang.lang.backend.base_backend import BaseBackend  # 导入基础后端类
from sglang.lang.chat_template import get_chat_template  # 导入获取聊天模板的函数
from sglang.lang.interpreter import StreamExecutor  # 导入流执行器类
from sglang.lang.ir import SglSamplingParams  # 导入采样参数类

try:  # 尝试导入 vertexai 相关模块
    import vertexai  # 导入 vertexai 主模块
    from vertexai.preview.generative_models import (  # 从 vertexai 预览版生成模型中导入
        GenerationConfig,  # 生成配置类
        GenerativeModel,  # 生成模型类
        Image,  # 图像类
    )
except ImportError as e:  # 如果导入失败，捕获异常
    GenerativeModel = e  # 将异常赋值给 GenerativeModel，以便后续检查


class VertexAI(BaseBackend):  # VertexAI 后端类，继承自 BaseBackend
    def __init__(self, model_name, safety_settings=None):  # 初始化方法，接收模型名称和安全设置
        super().__init__()  # 调用父类的初始化方法

        if isinstance(GenerativeModel, Exception):  # 如果 GenerativeModel 是异常对象，说明导入失败
            raise GenerativeModel  # 抛出导入异常

        project_id = os.environ["GCP_PROJECT_ID"]  # 从环境变量获取 GCP 项目 ID
        location = os.environ.get("GCP_LOCATION")  # 从环境变量获取 GCP 区域位置（可选）
        vertexai.init(project=project_id, location=location)  # 初始化 vertexai 客户端

        self.model_name = model_name  # 保存模型名称
        self.chat_template = get_chat_template("default")  # 获取默认聊天模板
        self.safety_settings = safety_settings  # 保存安全设置

    def get_chat_template(self):  # 获取聊天模板方法
        return self.chat_template  # 返回当前聊天模板

    def generate(  # 文本生成方法
        self,
        s: StreamExecutor,  # 流执行器对象
        sampling_params: SglSamplingParams,  # 采样参数
    ):
        if s.messages_:  # 如果存在多轮对话消息
            prompt = self.messages_to_vertexai_input(s.messages_)  # 将消息转换为 VertexAI 输入格式
        else:  # 否则为单轮对话
            # single-turn
            prompt = (  # 构造提示内容
                self.text_to_vertexai_input(s.text_, s.cur_images)  # 如果有图片，将文本和图片转换为 VertexAI 输入格式
                if s.cur_images  # 如果当前存在图片
                else s.text_  # 否则直接使用文本
            )
        ret = GenerativeModel(self.model_name).generate_content(  # 调用 VertexAI 模型生成内容
            prompt,  # 提示内容
            generation_config=GenerationConfig(**sampling_params.to_vertexai_kwargs()),  # 生成配置，将采样参数转换为 VertexAI 关键字参数
            safety_settings=self.safety_settings,  # 安全设置
        )

        comp = ret.text  # 获取生成的文本结果

        return comp, {}  # 返回生成的文本和空元信息字典

    def generate_stream(  # 流式文本生成方法
        self,
        s: StreamExecutor,  # 流执行器对象
        sampling_params: SglSamplingParams,  # 采样参数
    ):
        if s.messages_:  # 如果存在多轮对话消息
            prompt = self.messages_to_vertexai_input(s.messages_)  # 将消息转换为 VertexAI 输入格式
        else:  # 否则为单轮对话
            # single-turn
            prompt = (  # 构造提示内容
                self.text_to_vertexai_input(s.text_, s.cur_images)  # 如果有图片，将文本和图片转换为 VertexAI 输入格式
                if s.cur_images  # 如果当前存在图片
                else s.text_  # 否则直接使用文本
            )
        generator = GenerativeModel(self.model_name).generate_content(  # 调用 VertexAI 模型生成内容（流式）
            prompt,  # 提示内容
            stream=True,  # 启用流式输出
            generation_config=GenerationConfig(**sampling_params.to_vertexai_kwargs()),  # 生成配置，将采样参数转换为 VertexAI 关键字参数
            safety_settings=self.safety_settings,  # 安全设置
        )
        for ret in generator:  # 遍历流式生成器
            yield ret.text, {}  # 逐个返回生成的文本片段和空元信息字典

    def text_to_vertexai_input(self, text, images):  # 将文本和图片转换为 VertexAI 输入格式的方法
        input = []  # 初始化输入列表
        # split with image token
        text_segs = text.split(self.chat_template.image_token)  # 按图片标记分割文本
        for image_path, image_base64_data in images:  # 遍历每张图片的路径和 base64 数据
            text_seg = text_segs.pop(0)  # 弹出第一个文本片段
            if text_seg != "":  # 如果文本片段不为空
                input.append(text_seg)  # 将文本片段添加到输入列表
            input.append(Image.from_bytes(image_base64_data))  # 将 base64 图片数据转换为 Image 对象并添加到输入列表
        text_seg = text_segs.pop(0)  # 弹出最后一个文本片段
        if text_seg != "":  # 如果文本片段不为空
            input.append(text_seg)  # 将文本片段添加到输入列表
        return input  # 返回转换后的输入列表

    def messages_to_vertexai_input(self, messages):  # 将 OpenAI 消息格式转换为 VertexAI 消息格式的方法
        vertexai_message = []  # 初始化 VertexAI 消息列表
        # from openai message format to vertexai message format
        for msg in messages:  # 遍历每条消息
            if isinstance(msg["content"], str):  # 如果消息内容是字符串
                text = msg["content"]  # 直接获取文本内容
            else:  # 否则消息内容是列表格式
                text = msg["content"][0]["text"]  # 从列表的第一个元素获取文本内容

            if msg["role"] == "system":  # 如果消息角色是系统
                warnings.warn("Warning: system prompt is not supported in VertexAI.")  # 发出警告：VertexAI 不支持系统提示
                vertexai_message.append(  # 将系统提示作为用户消息添加
                    {
                        "role": "user",  # 角色设为用户
                        "parts": [{"text": "System prompt: " + text}],  # 文本内容加上系统提示前缀
                    }
                )
                vertexai_message.append(  # 添加模型确认回复
                    {
                        "role": "model",  # 角色设为模型
                        "parts": [{"text": "Understood."}],  # 模型回复"理解了"
                    }
                )
                continue  # 跳过后续处理，继续下一条消息
            if msg["role"] == "user":  # 如果消息角色是用户
                vertexai_msg = {  # 构造 VertexAI 用户消息
                    "role": "user",  # 角色设为用户
                    "parts": [{"text": text}],  # 文本内容
                }
            elif msg["role"] == "assistant":  # 如果消息角色是助手
                vertexai_msg = {  # 构造 VertexAI 模型消息
                    "role": "model",  # 角色设为模型（VertexAI 中助手称为 model）
                    "parts": [{"text": text}],  # 文本内容
                }

            # images
            if isinstance(msg["content"], list) and len(msg["content"]) > 1:  # 如果消息内容是列表且包含多个元素（说明有图片）
                for image in msg["content"][1:]:  # 遍历第一个元素之后的图片内容
                    assert image["type"] == "image_url"  # 断言内容类型为图片 URL
                    vertexai_msg["parts"].append(  # 将图片数据添加到消息的 parts 中
                        {
                            "inline_data": {  # 内联数据格式
                                "data": image["image_url"]["url"].split(",")[1],  # 从 base64 数据 URL 中提取实际的 base64 编码数据
                                "mime_type": "image/jpeg",  # MIME 类型设为 JPEG
                            }
                        }
                    )

            vertexai_message.append(vertexai_msg)  # 将构造好的 VertexAI 消息添加到列表中
        return vertexai_message  # 返回转换后的 VertexAI 消息列表
