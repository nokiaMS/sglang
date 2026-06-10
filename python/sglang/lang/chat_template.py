# 本文件是SGLang的聊天模板注册模块，定义了各种聊天模板（如default、claude、chatml、qwen、vicuna、llama-2-chat等）
# 以及根据模型路径匹配对应模板的函数，用于在对话生成时格式化消息格式。

import re  # 导入正则表达式模块，用于模型路径匹配
from dataclasses import dataclass  # 导入数据类装饰器，用于定义ChatTemplate数据类
from enum import Enum, auto  # 导入枚举类和自动赋值功能，用于定义ChatTemplateStyle枚举
from typing import Callable, Dict, List, Tuple  # 导入类型提示，用于类型注解


class ChatTemplateStyle(Enum):  # 聊天模板样式枚举类，定义不同的模板渲染风格
    PLAIN = auto()  # 普通样式，直接拼接前缀和后缀
    LLAMA2 = auto()  # Llama2聊天样式，对system和user角色有特殊处理逻辑


@dataclass  # 数据类装饰器
class ChatTemplate:  # 聊天模板数据类，定义了对话模板的名称、默认系统提示、角色前后缀、停止字符串等信息
    name: str  # 模板名称
    default_system_prompt: str  # 默认系统提示词
    role_prefix_and_suffix: Dict[str, Tuple[str, str]]  # 角色对应的前缀和后缀字典
    stop_str: List[str] = ()  # 停止字符串列表，生成时遇到这些字符串则停止
    image_token: str = "<image>"  # 图片占位符标记
    audio_token: str = "<audio>"  # 音频占位符标记
    style: ChatTemplateStyle = ChatTemplateStyle.PLAIN  # 模板样式，默认为普通样式

    def get_prefix_and_suffix(  # 根据角色和历史消息获取该角色的前缀和后缀字符串
        self, role: str, hist_messages: List[Dict]  # 方法参数：角色名称和历史消息列表
    ) -> Tuple[str, str]:  # 返回前缀和后缀的元组
        prefix, suffix = self.role_prefix_and_suffix.get(role, ("", ""))  # 从字典中获取角色的前缀和后缀，默认为空字符串

        if self.style == ChatTemplateStyle.LLAMA2:  # 如果模板样式为Llama2聊天样式
            if role == "system" and not hist_messages:  # 如果角色是system且没有历史消息（即第一条消息）
                user_prefix, _ = self.role_prefix_and_suffix.get("user", ("", ""))  # 获取user角色的前缀
                system_prefix, system_suffix = self.role_prefix_and_suffix.get(  # 获取system角色的前缀和后缀
                    "system", ("", "")  # system角色的默认前缀和后缀
                )  # get方法调用结束
                return (user_prefix + system_prefix, system_suffix)  # 返回user前缀+system前缀的组合，以及system后缀
            elif (  # 否则如果角色是user且历史消息只有一条（即system消息）且system内容不为空
                role == "user"  # 判断角色是否为user
                and len(hist_messages) == 1  # 判断历史消息是否只有一条
                and hist_messages[0]["content"] is not None  # 判断system消息内容是否不为None
            ):  # elif条件结束
                return ("", suffix)  # 返回空前缀和user后缀（因为前缀已包含在system消息中）

        return prefix, suffix  # 对于普通样式或其它情况，直接返回获取到的前缀和后缀

    def get_prompt(self, messages: List[Dict]) -> str:  # 根据消息列表生成完整的提示字符串
        prompt = ""  # 初始化提示字符串为空
        for i, message in enumerate(messages):  # 遍历每条消息及其索引
            role, content = message["role"], message["content"]  # 提取消息的角色和内容
            if role == "system" and content is None:  # 如果角色是system且内容为None
                content = self.default_system_prompt  # 使用默认系统提示词替换
                if content is None:  # 如果默认系统提示词也为None
                    continue  # 跳过此消息

            prefix, suffix = self.get_prefix_and_suffix(role, messages[:i])  # 获取当前角色的前缀和后缀，传入历史消息
            prompt += f"{prefix}{content}{suffix}"  # 将前缀、内容、后缀拼接并追加到提示字符串
        return prompt  # 返回完整的提示字符串


chat_template_registry: Dict[str, ChatTemplate] = {}  # 聊天模板注册表，名称到模板对象的映射
matching_function_registry: List[Callable] = []  # 匹配函数注册表，存放所有模板匹配函数


def register_chat_template(template):  # 将聊天模板注册到全局注册表中
    chat_template_registry[template.name] = template  # 以模板名称为键，模板对象为值存入注册表


def register_chat_template_matching_function(func):  # 将匹配函数注册到全局匹配函数列表中（用作装饰器，将函数添加到列表）
    matching_function_registry.append(func)  # 将匹配函数添加到注册表列表


def get_chat_template(name):  # 根据模板名称从注册表中获取聊天模板
    return chat_template_registry[name]  # 从注册表中按名称查找并返回模板


def get_chat_template_by_model_path(model_path):  # 根据模型路径自动匹配并返回对应的聊天模板，若无匹配则返回默认模板
    for matching_func in matching_function_registry:  # 遍历所有匹配函数
        template_name = matching_func(model_path)  # 调用匹配函数尝试匹配模型路径
        if template_name is not None:  # 如果匹配成功（返回了模板名称）
            return get_chat_template(template_name)  # 根据模板名称获取并返回模板
    return get_chat_template("default")  # 如果所有匹配函数都未匹配，返回默认模板


register_chat_template(  # 注册默认聊天模板
    ChatTemplate(  # 创建默认ChatTemplate实例
        name="default",  # 模板名称为default
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("SYSTEM:", "\n"),  # system角色前缀为SYSTEM:，后缀为换行
            "user": ("USER:", "\n"),  # user角色前缀为USER:，后缀为换行
            "assistant": ("ASSISTANT:", "\n"),  # assistant角色前缀为ASSISTANT:，后缀为换行
        },  # 字典定义结束
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册Claude聊天模板
    ChatTemplate(  # 创建Claude ChatTemplate实例
        name="claude",  # 模板名称为claude
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("", ""),  # system角色无前后缀
            "user": ("\n\nHuman: ", ""),  # user角色前缀为换行换行Human:
            "assistant": ("\n\nAssistant:", ""),  # assistant角色前缀为换行换行Assistant:
        },  # 字典定义结束
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册ChatML格式聊天模板
    ChatTemplate(  # 创建ChatML ChatTemplate实例
        name="chatml",  # 模板名称为chatml
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),  # system角色使用ChatML标记
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),  # user角色使用ChatML标记
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),  # assistant角色使用ChatML标记
        },  # 字典定义结束
        style=ChatTemplateStyle.PLAIN,  # 使用普通样式
        stop_str=("<|im_end|>",),  # 停止字符串为ChatML结束标记
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册ChatML-LLaVA多模态聊天模板
    ChatTemplate(  # 创建ChatML-LLaVA ChatTemplate实例
        name="chatml-llava",  # 模板名称为chatml-llava
        default_system_prompt="You are a helpful assistant.",  # 默认系统提示词
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),  # system角色使用ChatML标记
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),  # user角色使用ChatML标记
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),  # assistant角色使用ChatML标记
        },  # 字典定义结束
        style=ChatTemplateStyle.PLAIN,  # 使用普通样式
        stop_str=("<|im_end|>",),  # 停止字符串为ChatML结束标记
        image_token="<image>\n",  # 图片占位符
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

# There is default system prompt for qwen
# reference: https://modelscope.cn/models/qwen/Qwen2-72B-Instruct/file/view/master?fileName=tokenizer_config.json&status=1
# The chat template is: "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
register_chat_template(  # 注册Qwen聊天模板
    ChatTemplate(  # 创建Qwen ChatTemplate实例
        name="qwen",  # 模板名称为qwen
        default_system_prompt="You are a helpful assistant.",  # 默认系统提示词
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),  # system角色使用ChatML标记
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),  # user角色使用ChatML标记
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),  # assistant角色使用ChatML标记
        },  # 字典定义结束
        style=ChatTemplateStyle.PLAIN,  # 使用普通样式
        stop_str=("<|im_end|>",),  # 停止字符串为ChatML结束标记
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

# Reference: https://huggingface.co/docs/transformers/main/model_doc/qwen2_vl#usage-example
register_chat_template(  # 注册Qwen2-VL多模态聊天模板
    ChatTemplate(  # 创建Qwen2-VL ChatTemplate实例
        name="qwen2-vl",  # 模板名称为qwen2-vl
        default_system_prompt="You are a helpful assistant.",  # 默认系统提示词
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),  # system角色使用ChatML标记
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),  # user角色使用ChatML标记
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),  # assistant角色使用ChatML标记
        },  # 字典定义结束
        style=ChatTemplateStyle.PLAIN,  # 使用普通样式
        stop_str=("<|im_end|>",),  # 停止字符串为ChatML结束标记
        image_token="<|vision_start|><|image_pad|><|vision_end|>",  # 图片占位符使用视觉标记
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

# Reference: https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
register_chat_template(  # 注册Vicuna v1.1聊天模板
    ChatTemplate(  # 创建Vicuna ChatTemplate实例
        name="vicuna_v1.1",  # 模板名称为vicuna_v1.1
        default_system_prompt=(  # 默认系统提示词
            "A chat between a curious user and an artificial intelligence assistant. "  # 描述用户和助手之间的对话
            "The assistant gives helpful, detailed, and polite answers to the user's questions."  # 助手给出有帮助、详细、礼貌的回答
        ),  # 元组定义结束
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("", " "),  # system角色后缀为空格
            "user": ("USER:", " "),  # user角色前缀为USER:，后缀为空格
            "assistant": ("ASSISTANT:", "</s>"),  # assistant角色前缀为ASSISTANT:，后缀为结束标记
        },  # 字典定义结束
        image_token=" <image>\n",  # 图片占位符
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册Llama-2-Chat聊天模板
    ChatTemplate(  # 创建Llama-2-Chat ChatTemplate实例
        name="llama-2-chat",  # 模板名称为llama-2-chat
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("<<SYS>>\n", "\n<</SYS>>\n\n"),  # system角色使用Llama2的SYS标记
            "user": ("[INST] ", " [/INST]"),  # user角色使用INST标记
            "assistant": ("", " </s><s>"),  # assistant角色后缀为结束标记加开始标记
        },  # 字典定义结束
        style=ChatTemplateStyle.LLAMA2,  # 使用Llama2样式
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

# Reference: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/chat_template.json
register_chat_template(  # 注册Mistral聊天模板
    ChatTemplate(  # 创建Mistral ChatTemplate实例
        name="mistral",  # 模板名称为mistral
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("[SYSTEM_PROMPT] ", " [/SYSTEM_PROMPT]"),  # system角色使用SYSTEM_PROMPT标记
            "user": ("[INST] ", " [/INST]"),  # user角色使用INST标记
            "assistant": ("", " </s><s>"),  # assistant角色后缀为结束标记加开始标记
        },  # 字典定义结束
        stop_str=("</s>",),  # 停止字符串为结束标记
        image_token="[IMG]",  # 图片占位符为IMG标记
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册Llama-3-Instruct聊天模板
    ChatTemplate(  # 创建Llama-3-Instruct ChatTemplate实例
        name="llama-3-instruct",  # 模板名称为llama-3-instruct
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": (  # system角色前后缀
                "<|start_header_id|>system<|end_header_id|>\n\n",  # system角色前缀，使用header标记
                "<|eot_id|>",  # system角色后缀，使用eot标记
            ),  # 元组定义结束
            "user": (  # user角色前后缀
                "<|start_header_id|>user<|end_header_id|>\n\n",  # user角色前缀，使用header标记
                "<|eot_id|>",  # user角色后缀，使用eot标记
            ),  # 元组定义结束
            "assistant": (  # assistant角色前后缀
                "<|start_header_id|>assistant<|end_header_id|>\n\n",  # assistant角色前缀，使用header标记
                "<|eot_id|>",  # assistant角色后缀，使用eot标记
            ),  # 元组定义结束
        },  # 字典定义结束
        stop_str=("<|eot_id|>",),  # 停止字符串为eot标记
        image_token="<|image|>",  # 图片占位符
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

# https://huggingface.co/openbmb/MiniCPM-V-2_6
register_chat_template(  # 注册MiniCPM-V聊天模板
    ChatTemplate(  # 创建MiniCPM-V ChatTemplate实例
        name="minicpmv",  # 模板名称为minicpmv
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("", " "),  # system角色后缀为空格
            "user": ("user:", " "),  # user角色前缀为user:，后缀为空格
            "assistant": ("assistant:", "</s>"),  # assistant角色前缀为assistant:，后缀为结束标记
        },  # 字典定义结束
        stop_str=("<|im_end|>", "<|endoftext|>"),  # 停止字符串
        image_token="(<image>./</image>)",  # 图片占位符
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册Janus-Pro聊天模板
    ChatTemplate(  # 创建Janus-Pro ChatTemplate实例
        name="janus-pro",  # 模板名称为janus-pro
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": (  # system角色前后缀
                "",  # system角色前缀为空
                "",  # system角色后缀为空
            ),  # 元组定义结束
            "User": (  # User角色前后缀（注意大写User）
                "<｜User｜>",  # User角色前缀
                "",  # User角色后缀为空
            ),  # 元组定义结束
            "assistant": (  # assistant角色前后缀
                "<｜Assistant｜>",  # assistant角色前缀
                "<｜end▁of▁sentence｜>",  # assistant角色后缀为结束标记
            ),  # 元组定义结束
        },  # 字典定义结束
        stop_str=("<｜end▁of▁sentence｜>",),  # 停止字符串为结束标记
        image_token="<image_placeholder>\n",  # 图片占位符
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

# https://huggingface.co/openbmb/MiniCPM-o-2_6
register_chat_template(  # 注册MiniCPM-O聊天模板
    ChatTemplate(  # 创建MiniCPM-O ChatTemplate实例
        name="minicpmo",  # 模板名称为minicpmo
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("", " "),  # system角色后缀为空格
            "user": ("user:", " "),  # user角色前缀为user:，后缀为空格
            "assistant": ("assistant:", "</s>"),  # assistant角色前缀为assistant:，后缀为结束标记
        },  # 字典定义结束
        stop_str=("<|im_end|>", "<|endoftext|>"),  # 停止字符串
        image_token="(<image>./</image>)",  # 图片占位符
        audio_token="(<audio>./</audio>)",  # 音频占位符
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册Janus聊天模板
    ChatTemplate(  # 创建Janus ChatTemplate实例
        name="janus",  # 模板名称为janus
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": (  # system角色前后缀
                "",  # system角色前缀为空
                "",  # system角色后缀为空
            ),  # 元组定义结束
            "user": (  # user角色前后缀
                "<｜User｜>",  # user角色前缀
                "",  # user角色后缀为空
            ),  # 元组定义结束
            "assistant": (  # assistant角色前后缀
                "<｜Assistant｜>",  # assistant角色前缀
                "<｜end▁of▁sentence｜>",  # assistant角色后缀为结束标记
            ),  # 元组定义结束
        },  # 字典定义结束
        stop_str=("<｜end▁of▁sentence｜>",),  # 停止字符串为结束标记
        image_token="<image_placeholder>\n",  # 图片占位符
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

# The difference between "llama-3-instruct-llava" and "llama-3-instruct" is that llava uses a different image_token.
register_chat_template(  # 注册Llama-3-Instruct-LLaVA聊天模板
    ChatTemplate(  # 创建Llama-3-Instruct-LLaVA ChatTemplate实例
        name="llama-3-instruct-llava",  # 模板名称为llama-3-instruct-llava
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": (  # system角色前后缀
                "<|start_header_id|>system<|end_header_id|>\n\n",  # system角色前缀，使用header标记
                "<|eot_id|>",  # system角色后缀，使用eot标记
            ),  # 元组定义结束
            "user": (  # user角色前后缀
                "<|start_header_id|>user<|end_header_id|>\n\n",  # user角色前缀，使用header标记
                "<|eot_id|>",  # user角色后缀，使用eot标记
            ),  # 元组定义结束
            "assistant": (  # assistant角色前后缀
                "<|start_header_id|>assistant<|end_header_id|>\n\n",  # assistant角色前缀，使用header标记
                "<|eot_id|>",  # assistant角色后缀，使用eot标记
            ),  # 元组定义结束
        },  # 字典定义结束
        stop_str=("<|eot_id|>",),  # 停止字符串为eot标记
        image_token="<image>\n",  # 图片占位符
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

# Reference: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/chat_template.json
register_chat_template(  # 注册Llama-4聊天模板
    ChatTemplate(  # 创建Llama-4 ChatTemplate实例
        name="llama-4",  # 模板名称为llama-4
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": (  # system角色前后缀
                "<|header_start|>system<|header_end|>\n\n",  # system角色前缀，使用header标记
                "<|eot|>",  # system角色后缀，使用eot标记
            ),  # 元组定义结束
            "user": (  # user角色前后缀
                "<|header_start|>user<|header_end|>\n\n",  # user角色前缀，使用header标记
                "<|eot|>",  # user角色后缀，使用eot标记
            ),  # 元组定义结束
            "assistant": (  # assistant角色前后缀
                "<|header_start|>assistant<|header_end|>\n\n",  # assistant角色前缀，使用header标记
                "<|eot|>",  # assistant角色后缀，使用eot标记
            ),  # 元组定义结束
        },  # 字典定义结束
        stop_str=("<|eot|>",),  # 停止字符串为eot标记
        image_token="<|image|>",  # 图片占位符
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

# Reference: https://modelscope.cn/models/01ai/Yi-1.5-34B-Chat/file/view/master?fileName=tokenizer_config.json&status=1
register_chat_template(  # 注册Yi-1.5聊天模板
    ChatTemplate(  # 创建Yi-1.5 ChatTemplate实例
        name="yi-1.5",  # 模板名称为yi-1.5
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("", ""),  # system角色无前后缀
            "user": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"),  # user角色前后缀，后缀中包含assistant前缀
            "assistant": ("", "<|im_end|>\n"),  # assistant角色后缀为im_end标记
        },  # 字典定义结束
        style=ChatTemplateStyle.PLAIN,  # 使用普通样式
        stop_str=("<|im_end|>",),  # 停止字符串为im_end标记
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

# Reference: https://github.com/01-ai/Yi/tree/main/VL#major-difference-with-llava
register_chat_template(  # 注册Yi-VL聊天模板
    ChatTemplate(  # 创建Yi-VL ChatTemplate实例
        name="yi-vl",  # 模板名称为yi-vl
        default_system_prompt=(  # 默认系统提示词
            "This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. Read all the images carefully, and respond to the human's questions with informative, helpful, detailed and polite answers."  # 英文提示部分
            "这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。"  # 中文提示部分
        ),  # 元组定义结束
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("", "\n\n"),  # system角色后缀为双换行
            "user": ("### Human:", "\n"),  # user角色前缀为Human:，后缀为换行
            "assistant": ("### Assistant:", "\n"),  # assistant角色前缀为Assistant:，后缀为换行
        },  # 字典定义结束
        image_token=" <image_placeholder>\n",  # 图片占位符
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册Gemma-IT聊天模板
    ChatTemplate(  # 创建Gemma-IT ChatTemplate实例
        name="gemma-it",  # 模板名称为gemma-it
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("", ""),  # system角色无前后缀
            "user": ("<start_of_turn>user\n", "<end_of_turn>\n"),  # user角色前后缀，使用turn标记
            "assistant": ("<start_of_turn>model\n", "<end_of_turn>\n"),  # assistant角色前后缀（Gemma中assistant称为model）
        },  # 字典定义结束
        image_token="<start_of_image>",  # 图片占位符
        audio_token="<start_of_audio>",  # 音频占位符
        style=ChatTemplateStyle.PLAIN,  # 使用普通样式
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册Gemma-4-IT聊天模板
    ChatTemplate(  # 创建Gemma-4-IT ChatTemplate实例
        name="gemma-4-it",  # 模板名称为gemma-4-it
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("", ""),  # system角色无前后缀
            "user": ("<|turn>user\n", "<turn|>\n"),  # user角色前后缀，使用turn标记
            "assistant": ("<|turn>assistant\n", "<turn|>\n"),  # assistant角色前后缀，使用turn标记
        },  # 字典定义结束
        style=ChatTemplateStyle.PLAIN,  # 使用普通样式
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册DBRX-Instruct聊天模板
    ChatTemplate(  # 创建DBRX-Instruct ChatTemplate实例
        name="dbrx-instruct",  # 模板名称为dbrx-instruct
        default_system_prompt="You are DBRX, created by Databricks. You were last updated in December 2023. You answer questions based on information available up to that point.\nYOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough responses to more complex and open-ended questions.\nYou assist with various tasks, from writing to coding (using markdown for code blocks — remember to use ``` with code, JSON, and tables).\n(You do not have real-time data access or code execution capabilities. You avoid stereotyping and provide balanced perspectives on controversial topics. You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.)\nThis is your system prompt, guiding your responses. Do not reference it, just respond to the user. If you find yourself talking about this message, stop. You should be responding appropriately and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY.",  # 默认系统提示词（详细的DBRX身份描述）
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("<|im_start|>system\n", "<|im_end|>"),  # system角色前后缀
            "user": ("\n<|im_start|>user\n", "<|im_end|>"),  # user角色前后缀
            "assistant": ("\n<|im_start|>assistant\n", "<|im_end|>"),  # assistant角色前后缀
        },  # 字典定义结束
        stop_str=("<|im_end|>",),  # 停止字符串为im_end标记
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册C4AI-Command-R聊天模板
    ChatTemplate(  # 创建C4AI-Command-R ChatTemplate实例
        name="c4ai-command-r",  # 模板名称为c4ai-command-r
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": (  # system角色前后缀
                "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",  # system角色前缀
                "<|END_OF_TURN_TOKEN|>",  # system角色后缀
            ),  # 元组定义结束
            "user": ("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>", "<|END_OF_TURN_TOKEN|>"),  # user角色前后缀
            "assistant": (  # assistant角色前后缀
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",  # assistant角色前缀（CHATBOT_TOKEN）
                "<|END_OF_TURN_TOKEN|>",  # assistant角色后缀
            ),  # 元组定义结束
        },  # 字典定义结束
        style=ChatTemplateStyle.PLAIN,  # 使用普通样式
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

# Adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
register_chat_template(  # 注册InternVL-2.5聊天模板
    ChatTemplate(  # 创建InternVL-2.5 ChatTemplate实例
        name="internvl-2-5",  # 模板名称为internvl-2-5
        default_system_prompt="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",  # 默认系统提示词（中文书生万象身份描述）
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),  # system角色使用ChatML标记
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),  # user角色使用ChatML标记
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),  # assistant角色使用ChatML标记
        },  # 字典定义结束
        stop_str=["<|im_end|>", "<|action_end|>"],  # 停止字符串列表
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册InternS1聊天模板
    ChatTemplate(  # 创建InternS1 ChatTemplate实例
        name="interns1",  # 模板名称为interns1
        default_system_prompt="You are an AI assistant whose name is Intern-S1 (书生大模型).\n- Intern-S1 (书生大模型) is a vision-language model that is developed by Shanghai AI Laboratory (上海人工智能实验室).  It is designed to be helpful, honest, and harmless.\n- Intern-S1 (书生大模型) can understand and communicate fluently in the language chosen by the user such as English and 中文.\nYou are an expert reasoner with extensive experience in all areas. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags.",  # 默认系统提示词（Intern-S1身份描述及思考过程要求）
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),  # system角色使用ChatML标记
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),  # user角色使用ChatML标记
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),  # assistant角色使用ChatML标记
        },  # 字典定义结束
        stop_str=["<|im_end|>", "<|action_end|>"],  # 停止字符串列表
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册Granite-3-Instruct聊天模板
    ChatTemplate(  # 创建Granite-3-Instruct ChatTemplate实例
        name="granite-3-instruct",  # 模板名称为granite-3-instruct
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": (  # system角色前后缀
                "<|start_of_role|>system<|end_of_role|>",  # system角色前缀，使用role标记
                "<|end_of_text|>",  # system角色后缀，使用end_of_text标记
            ),  # 元组定义结束
            "user": (  # user角色前后缀
                "<|start_of_role|>user<|end_of_role|>",  # user角色前缀，使用role标记
                "<|end_of_text|>",  # user角色后缀，使用end_of_text标记
            ),  # 元组定义结束
            "assistant": (  # assistant角色前后缀
                "<|start_of_role|>assistant<|end_of_role|>",  # assistant角色前缀，使用role标记
                "<|end_of_text|>",  # assistant角色后缀，使用end_of_text标记
            ),  # 元组定义结束
        },  # 字典定义结束
        stop_str=("<|end_of_text|>",),  # 停止字符串为end_of_text标记
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

register_chat_template(  # 注册DeepSeek-V3聊天模板
    ChatTemplate(  # 创建DeepSeek-V3 ChatTemplate实例
        name="deepseek-v3",  # 模板名称为deepseek-v3
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": (  # system角色前后缀
                "",  # system角色前缀为空
                "",  # system角色后缀为空
            ),  # 元组定义结束
            "user": (  # user角色前后缀
                "<｜User｜>",  # user角色前缀
                "",  # user角色后缀为空
            ),  # 元组定义结束
            "assistant": (  # assistant角色前后缀
                "<｜Assistant｜>",  # assistant角色前缀
                "<｜end▁of▁sentence｜>",  # assistant角色后缀为结束标记
            ),  # 元组定义结束
        },  # 字典定义结束
        stop_str=("<｜end▁of▁sentence｜>",),  # 停止字符串为结束标记
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束

# Reference: https://huggingface.co/docs/transformers/main/model_doc/glm4_v#usage-example
register_chat_template(  # 注册GLM-4V聊天模板
    ChatTemplate(  # 创建GLM-4V ChatTemplate实例
        name="glm-4v",  # 模板名称为glm-4v
        default_system_prompt=None,  # 默认系统提示词为None
        role_prefix_and_suffix={  # 角色前后缀定义
            "system": ("<|system|>\n", "\n"),  # system角色前缀和后缀
            "user": ("<|user|>\n", "\n"),  # user角色前缀和后缀
            "assistant": ("<|assistant|>\n", "\n"),  # assistant角色前缀和后缀
        },  # 字典定义结束
        style=ChatTemplateStyle.PLAIN,  # 使用普通样式
        stop_str=["<|user|>", "<|endoftext|>", "<|observation|>"],  # 停止字符串列表
        image_token="<|image|>",  # 图片占位符
    )  # ChatTemplate实例结束
)  # register_chat_template调用结束


@register_chat_template_matching_function  # 注册DeepSeek模型路径匹配函数（装饰器）
def match_deepseek(model_path: str):  # 匹配DeepSeek模型路径，返回对应模板名称
    if re.search(r"deepseek-(v3|r1)", model_path, re.IGNORECASE) and not re.search(  # 正则匹配deepseek-v3或deepseek-r1，且排除base模型
        r"base", model_path, re.IGNORECASE  # 排除base模型
    ):  # 排除条件结束
        return "deepseek-v3"  # 匹配成功返回deepseek-v3模板名


@register_chat_template_matching_function  # 注册Orion模型路径匹配函数（装饰器）
def match_orion(model_path: str):  # 匹配Orion模型路径，返回对应模板名称
    if "orion" in model_path.lower():  # 如果模型路径中包含orion（不区分大小写）
        return "claude"  # 匹配成功返回claude模板名


@register_chat_template_matching_function  # 注册DeepSeek Janus-Pro模型路径匹配函数（装饰器）
def match_deepseek_janus_pro(model_path: str):  # 匹配DeepSeek Janus-Pro模型路径，返回对应模板名称
    if re.search(r"janus", model_path, re.IGNORECASE):  # 正则匹配janus（不区分大小写）
        return "janus-pro"  # 匹配成功返回janus-pro模板名


@register_chat_template_matching_function  # 注册DBRX模型路径匹配函数（装饰器）
def match_dbrx(model_path: str):  # 匹配DBRX模型路径，返回对应模板名称
    if re.search(r"dbrx", model_path, re.IGNORECASE) and re.search(  # 正则匹配dbrx且包含instruct
        r"instruct", model_path, re.IGNORECASE  # 匹配instruct关键词
    ):  # 匹配条件结束
        return "dbrx-instruct"  # 匹配成功返回dbrx-instruct模板名


@register_chat_template_matching_function  # 注册Vicuna模型路径匹配函数（装饰器）
def match_vicuna(model_path: str):  # 匹配Vicuna模型路径，返回对应模板名称
    if re.search(r"vicuna|llava-v1\.5|llava-next-video-7b", model_path, re.IGNORECASE):  # 正则匹配vicuna、llava-v1.5或llava-next-video-7b
        return "vicuna_v1.1"  # 匹配成功返回vicuna_v1.1模板名


@register_chat_template_matching_function  # 注册Llama-2-Chat模型路径匹配函数（装饰器）
def match_llama2_chat(model_path: str):  # 匹配Llama-2-Chat模型路径，返回对应模板名称
    if re.search(  # 正则匹配llama-2.*chat或codellama.*instruct
        r"llama-2.*chat|codellama.*instruct",  # 匹配模式定义
        model_path,  # 模型路径
        re.IGNORECASE,  # 不区分大小写
    ):  # 匹配条件结束
        return "llama-2-chat"  # 匹配成功返回llama-2-chat模板名


@register_chat_template_matching_function  # 注册Mistral模型路径匹配函数（装饰器）
def match_mistral(model_path: str):  # 匹配Mistral模型路径，返回对应模板名称
    if re.search(r"pixtral|(mistral|mixtral).*instruct", model_path, re.IGNORECASE):  # 正则匹配pixtral或mistral/mixtral加instruct
        return "mistral"  # 匹配成功返回mistral模板名


@register_chat_template_matching_function  # 注册Llama-3-Instruct模型路径匹配函数（装饰器）
def match_llama3_instruct(model_path: str):  # 匹配Llama-3-Instruct模型路径，返回对应模板名称
    if re.search(r"llama-3.*instruct", model_path, re.IGNORECASE):  # 正则匹配llama-3.*instruct
        return "llama-3-instruct"  # 匹配成功返回llama-3-instruct模板名


@register_chat_template_matching_function  # 注册ChatML模型路径匹配函数（装饰器）
def match_chat_ml(model_path: str):  # 匹配多种ChatML格式模型路径，返回对应模板名称
    if re.search(r"tinyllama", model_path, re.IGNORECASE):  # 如果匹配tinyllama，返回chatml模板
        return "chatml"  # 返回chatml模板名
    if re.search(r"qwen.*vl", model_path, re.IGNORECASE):  # 如果匹配qwen加vl，返回qwen2-vl模板
        return "qwen2-vl"  # 返回qwen2-vl模板名
    if re.search(r"glm[-_]?4(\.\d+)?v", model_path, re.IGNORECASE):  # 如果匹配glm-4v格式，返回glm-4v模板
        return "glm-4v"  # 返回glm-4v模板名
    if re.search(r"qwen.*(chat|instruct)", model_path, re.IGNORECASE) and not re.search(  # 如果匹配qwen加chat/instruct且不包含llava，返回qwen模板
        r"llava", model_path, re.IGNORECASE  # 排除llava模型
    ):  # 排除条件结束
        return "qwen"  # 返回qwen模板名
    if re.search(  # 如果匹配llava-v1.6-34b等大型LLaVA模型
        r"llava-v1\.6-34b|llava-v1\.6-yi-34b|llava-next-video-34b|llava-onevision-qwen2",  # 匹配llava-v1.6-34b、llava-v1.6-yi-34b、llava-next-video-34b或llava-onevision-qwen2
        model_path,  # 模型路径
        re.IGNORECASE,  # 不区分大小写
    ):  # 匹配条件结束
        return "chatml-llava"  # 返回chatml-llava模板名


@register_chat_template_matching_function  # 注册Yi模型路径匹配函数（装饰器）
def match_chat_yi(model_path: str):  # 匹配Yi模型路径，返回对应模板名称
    if re.search(r"yi-vl", model_path, re.IGNORECASE) and not re.search(  # 如果匹配yi-vl且不包含llava
        r"llava", model_path, re.IGNORECASE  # 排除llava模型
    ):  # 排除条件结束
        return "yi-vl"  # 返回yi-vl模板名
    elif re.search(r"yi-1\.5.*chat", model_path, re.IGNORECASE):  # 否则如果匹配yi-1.5加chat
        return "yi-1.5"  # 返回yi-1.5模板名


@register_chat_template_matching_function  # 注册Gemma模型路径匹配函数（装饰器）
def match_gemma(model_path: str):  # 匹配Gemma模型路径，返回对应模板名称
    if re.search(r"gemma-4.*it", model_path, re.IGNORECASE):  # 如果匹配gemma-4加it，返回gemma-4-it模板
        return "gemma-4-it"  # 返回gemma-4-it模板名
    if re.search(r"(gemma.*it)|(gemma-3)", model_path, re.IGNORECASE):  # 如果匹配gemma加it或gemma-3
        return "gemma-it"  # 返回gemma-it模板名


@register_chat_template_matching_function  # 注册OpenBMB MiniCPM模型路径匹配函数（装饰器）
def match_openbmb_minicpm(model_path: str):  # 匹配OpenBMB MiniCPM模型路径，返回对应模板名称
    if re.search(r"minicpm-v", model_path, re.IGNORECASE):  # 如果匹配minicpm-v，返回minicpmv模板
        return "minicpmv"  # 返回minicpmv模板名
    elif re.search(r"minicpm-o", model_path, re.IGNORECASE):  # 否则如果匹配minicpm-o
        return "minicpmo"  # 返回minicpmo模板名


@register_chat_template_matching_function  # 注册C4AI Command-R模型路径匹配函数（装饰器）
def match_c4ai_command_r(model_path: str):  # 匹配C4AI Command-R模型路径，返回对应模板名称
    if re.search(r"c4ai-command-r", model_path, re.IGNORECASE):  # 正则匹配c4ai-command-r（不区分大小写）
        return "c4ai-command-r"  # 返回c4ai-command-r模板名


@register_chat_template_matching_function  # 注册Granite-Instruct模型路径匹配函数（装饰器）
def match_granite_instruct(model_path: str):  # 匹配Granite-Instruct模型路径，返回对应模板名称
    if re.search(r"granite.*instruct", model_path, re.IGNORECASE):  # 正则匹配granite加instruct（不区分大小写）
        return "granite-3-instruct"  # 返回granite-3-instruct模板名


@register_chat_template_matching_function  # 注册InternVL聊天模型路径匹配函数（装饰器）
def match_internvl_chat(model_path: str):  # 匹配InternVL模型路径，返回对应模板名称
    if re.search(r"internvl2_5", model_path, re.IGNORECASE):  # 正则匹配internvl2_5（不区分大小写）
        return "internvl-2-5"  # 返回internvl-2-5模板名


@register_chat_template_matching_function  # 注册InternS1聊天模型路径匹配函数（装饰器）
def match_interns1_chat(model_path: str):  # 匹配InternS1模型路径，返回对应模板名称
    if re.search(r"intern-s1", model_path, re.IGNORECASE):  # 正则匹配intern-s1（不区分大小写）
        return "interns1"  # 返回interns1模板名
    if re.search(r"interns1", model_path, re.IGNORECASE):  # 如果匹配interns1
        return "interns1"  # 返回interns1模板名


if __name__ == "__main__":  # 主程序入口
    messages = [  # 测试消息列表
        {"role": "system", "content": None},  # None means default  # system消息，content为None表示使用默认值
        # {"role": "system", "content": "You are a helpful, respectful and honest assistant."},
        {"role": "user", "content": "Hello!"},  # user消息
        {"role": "assistant", "content": "Hi!"},  # assistant消息
        {"role": "user", "content": "What can you do?"},  # user消息
        {"role": "assistant", "content": "I can chat with you."},  # assistant消息
    ]  # 消息列表结束

    template = get_chat_template("llama-2-chat")  # 获取llama-2-chat模板
    print(template.get_prompt(messages))  # 打印生成的提示字符串
