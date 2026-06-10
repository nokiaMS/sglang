# Jinja模板工具模块
# 本模块提供Jinja聊天模板的分析和处理工具，包括内容格式检测
# （判断模板期望'string'还是'openai'格式）和消息内容处理功能。
"""Template utilities for Jinja template processing.

This module provides utilities for analyzing and processing Jinja chat templates,
including content format detection and message processing.
"""  # 模板处理工具的文档字符串

import logging  # 导入日志模块

import jinja2  # 导入Jinja2模板引擎
import transformers.utils.chat_template_utils as hf_chat_utils  # 导入HuggingFace聊天模板工具

from sglang.srt.utils import ImageData  # 导入图像数据类

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

# ============================================================================
# JINJA TEMPLATE CONTENT FORMAT DETECTION  # Jinja模板内容格式检测
# ============================================================================
#
# This adapts vLLM's approach for detecting chat template content format:  # 此部分适配vLLM的聊天模板内容格式检测方法
# https://github.com/vllm-project/vllm/blob/02f0c7b220422792f5e53de2a7d51d2d3ff2df28/vllm/entrypoints/chat_utils.py#L296-L313
# - Analyzes Jinja template AST to detect content iteration patterns  # - 分析Jinja模板AST以检测内容迭代模式
# - 'openai' format: templates with {%- for content in message['content'] -%} loops  # - 'openai'格式：带有内容循环的模板
# - 'string' format: templates that expect simple string content  # - 'string'格式：期望简单字符串内容的模板
# - Processes content accordingly to match template expectations  # - 相应地处理内容以匹配模板期望


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    """Check if node is a variable access like {{ varname }}"""  # 检查节点是否是变量访问，如{{ varname }}
    if isinstance(node, jinja2.nodes.Name):  # 如果是Name类型节点
        return node.ctx == "load" and node.name == varname  # 检查是否为加载上下文且名称匹配
    return False  # 不是变量访问


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    """Check if node is an attribute access like {{ varname['key'] }} or {{ varname.key }}"""  # 检查节点是否是属性访问，如{{ varname['key'] }}或{{ varname.key }}
    if isinstance(node, jinja2.nodes.Getitem):  # 如果是Getitem类型节点（下标访问）
        return (  # 返回是否匹配
            _is_var_access(node.node, varname)  # 检查对象是否为目标变量
            and isinstance(node.arg, jinja2.nodes.Const)  # 检查键是否为常量
            and node.arg.value == key  # 检查键值是否匹配
        )

    if isinstance(node, jinja2.nodes.Getattr):  # 如果是Getattr类型节点（属性访问）
        return _is_var_access(node.node, varname) and node.attr == key  # 检查对象和属性名是否匹配

    return False  # 不是属性访问


def _is_var_or_elems_access(
    node: jinja2.nodes.Node,
    varname: str,
    key: str = None,
) -> bool:
    """Check if node accesses varname or varname[key] with filters/tests"""  # 检查节点是否通过过滤器/测试访问varname或varname[key]
    if isinstance(node, jinja2.nodes.Filter):  # 如果是Filter类型节点
        return node.node is not None and _is_var_or_elems_access(  # 递归检查被过滤的节点
            node.node, varname, key
        )
    if isinstance(node, jinja2.nodes.Test):  # 如果是Test类型节点
        return _is_var_or_elems_access(node.node, varname, key)  # 递归检查被测试的节点

    if isinstance(node, jinja2.nodes.Getitem) and isinstance(  # 如果是切片访问
        node.arg, jinja2.nodes.Slice
    ):
        return _is_var_or_elems_access(node.node, varname, key)  # 递归检查切片的对象

    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)  # 根据是否有key选择检查方式


def _try_extract_ast(chat_template: str):  # 尝试从聊天模板提取AST
    """Try to parse the Jinja template into an AST"""  # 尝试将Jinja模板解析为AST
    try:  # 尝试解析
        jinja_compiled = hf_chat_utils._compile_jinja_template(chat_template)  # 编译Jinja模板
        return jinja_compiled.environment.parse(chat_template)  # 返回解析后的AST
    except Exception as e:  # 解析失败
        logger.debug(f"Error when compiling Jinja template: {e}")  # 记录调试日志
        return None  # 返回None


def detect_jinja_template_content_format(chat_template: str) -> str:
    """
    Detect whether a chat template expects 'string' or 'openai' content format.

    - 'string': content is a simple string (like DeepSeek templates)  # content是简单字符串
    - 'openai': content is a list of structured dicts (like Llama4 templates)  # content是结构化字典列表

    Detection logic:  # 检测逻辑
    - If template has loops like {%- for content in message['content'] -%} → 'openai'  # 如果模板有内容循环则为openai格式
    - Otherwise → 'string'  # 否则为string格式
    """  # 检测聊天模板期望的内容格式是'string'还是'openai'
    # Shortcut for multimodal templates  # 多模态模板的快捷判断
    if any(
        keyword in chat_template for keyword in ["image", "audio", "video", "vision"]  # 检查模板是否包含多模态关键词
    ):
        return "openai"  # 包含多模态关键词则使用openai格式

    jinja_ast = _try_extract_ast(chat_template)  # 尝试提取模板AST
    if jinja_ast is None:  # 如果提取失败
        return "string"  # 默认使用string格式

    try:  # 尝试分析AST
        # Look for patterns like: {%- for content in message['content'] -%}  # 查找内容迭代模式
        for loop_ast in jinja_ast.find_all(jinja2.nodes.For):  # 遍历所有For循环节点
            loop_iter = loop_ast.iter  # 获取循环迭代目标

            # Check if iterating over message['content'] or similar  # 检查是否迭代message['content']或类似结构
            if _is_var_or_elems_access(loop_iter, "message", "content"):  # 检查是否迭代message['content']
                return "openai"  # Found content iteration → openai format  # 找到内容迭代则为openai格式

            # Also check for patterns like: {%- for item in msg.content -%} or {%- for item in m.content -%}  # 也检查msg.content或m.content模式
            if _is_var_or_elems_access(
                loop_iter, "msg", "content"  # 检查是否迭代msg['content']
            ) or _is_var_or_elems_access(loop_iter, "m", "content"):  # 检查是否迭代m['content']
                return "openai"  # Found content iteration → openai format (glm4v)  # 找到内容迭代则为openai格式（glm4v）

        return "string"  # No content loops found → string format  # 未找到内容循环则为string格式
    except Exception as e:  # 分析失败
        logger.debug(f"Error when parsing AST of Jinja template: {e}")  # 记录调试日志
        return "string"  # 默认使用string格式


def process_content_for_template_format(
    msg_dict: dict,
    content_format: str,
    image_data: list,
    video_data: list,
    audio_data: list,
    modalities: list,
    use_dpsk_v32_encoding: bool = False,
) -> dict:
    """
    Process message content based on detected template format.

    Args:
        msg_dict: Message dictionary with content  # 包含content的消息字典
        content_format: 'string' or 'openai' (detected via AST analysis)  # 内容格式，通过AST分析检测
        image_data: List to append extracted image URLs  # 用于追加提取的图像URL的列表
        video_data: List to append extracted video URLs  # 用于追加提取的视频URL的列表
        audio_data: List to append extracted audio URLs  # 用于追加提取的音频URL的列表
        modalities: List to append modalities  # 用于追加模态信息的列表
        use_dpsk_v32_encoding: If True, extract multimodal data and convert content to string (for DeepSeek-V3.2 encoding)  # 如果为True，提取多模态数据并将内容转换为字符串

    Returns:
        Processed message dictionary  # 处理后的消息字典
    """  # 根据检测到的模板格式处理消息内容
    if not isinstance(msg_dict.get("content"), list):  # 如果content不是列表
        # Already a string or None, no processing needed  # 已经是字符串或None，无需处理
        return {k: v for k, v in msg_dict.items() if v is not None}  # 过滤None值后返回

    if content_format == "openai" or use_dpsk_v32_encoding:  # openai格式或DeepSeek-V3.2编码
        # OpenAI format: preserve structured content list, normalize types  # OpenAI格式：保留结构化内容列表，规范化类型
        # V32 encoding: extract multimodal data but convert content to string  # V32编码：提取多模态数据但将内容转为字符串
        processed_content_parts = []  # 处理后的内容部分列表
        text_parts = []  # 文本部分列表
        for chunk in msg_dict["content"]:  # 遍历content列表中的每个元素
            if isinstance(chunk, dict):  # 如果元素是字典
                chunk_type = chunk.get("type")  # 获取元素类型

                if chunk_type == "image_url":  # 如果是图像URL类型
                    image_obj = chunk.get("image_url") or {}  # 获取图像对象
                    mdp = image_obj.get("max_dynamic_patch", None)  # 获取最大动态补丁数
                    # Also allow flat style: chunk["max_dynamic_patch"]  # 也允许扁平风格
                    image_data.append(  # 追加图像数据
                        ImageData(
                            url=image_obj["url"],  # 图像URL
                            detail=image_obj.get("detail", "auto"),  # 图像细节级别
                            max_dynamic_patch=mdp,  # 最大动态补丁数
                        )
                    )

                    if chunk.get("modalities"):  # 如果有模态信息
                        modalities.append(chunk.get("modalities"))  # 追加模态信息
                    # Normalize to simple 'image' type for template compatibility  # 规范化为简单的'image'类型以兼容模板
                    processed_content_parts.append({"type": "image"})  # 追加规范化的图像类型
                elif chunk_type == "video_url":  # 如果是视频URL类型
                    video_obj = chunk.get("video_url") or {}  # 获取视频对象
                    mdp = video_obj.get("max_dynamic_patch", None)  # 获取最大动态补丁数
                    if mdp is None:  # 如果没有动态补丁数
                        video_data.append(chunk["video_url"]["url"])  # 仅追加视频URL
                    else:  # 有动态补丁数
                        # Keep structured info for backend, but template only sees {"type":"video"}  # 保留后端的结构化信息，但模板只看到{"type":"video"}
                        video_data.append(  # 追加结构化视频数据
                            {
                                "url": video_obj["url"],  # 视频URL
                                "max_dynamic_patch": mdp,  # 最大动态补丁数
                            }
                        )
                    if chunk.get("modalities"):  # 如果有模态信息
                        modalities.append(chunk.get("modalities"))  # 追加模态信息
                    # Normalize to simple 'video' type for template compatibility  # 规范化为简单的'video'类型以兼容模板
                    processed_content_parts.append({"type": "video"})  # 追加规范化的视频类型
                elif chunk_type == "audio_url":  # 如果是音频URL类型
                    audio_data.append(chunk["audio_url"]["url"])  # 追加音频URL
                    # Normalize to simple 'audio' type  # 规范化为简单的'audio'类型
                    processed_content_parts.append({"type": "audio"})  # 追加规范化的音频类型
                elif chunk_type == "text":  # 如果是文本类型
                    # For v32 encoding, collect text parts separately  # 对于V32编码，单独收集文本部分
                    if use_dpsk_v32_encoding:  # 如果使用V32编码
                        text_parts.append(chunk["text"])  # 追加到文本部分列表
                    else:  # 不使用V32编码
                        # Keep text content as-is for openai format  # 对于openai格式保留原始文本内容
                        processed_content_parts.append(chunk)  # 追加原始文本块
                elif chunk_type == "tool_reference":  # 如果是工具引用类型
                    # GLM-specific extension: pass through so the chat template  # GLM特定扩展：直接传递以便聊天模板
                    # can match tool_reference.name against tools[*].function.name  # 可以将tool_reference.name与tools[*].function.name匹配
                    # and render the referenced tool schemas inline.  # 并内联渲染引用的工具模式
                    processed_content_parts.append(chunk)  # 直接追加工具引用块

        new_msg = {  # 构建新消息字典
            k: v for k, v in msg_dict.items() if v is not None and k != "content"  # 过滤None值并排除content键
        }
        if use_dpsk_v32_encoding:  # 如果使用V32编码
            new_msg["content"] = " ".join(text_parts) if text_parts else ""  # 将文本部分合并为字符串
        else:  # 不使用V32编码
            new_msg["content"] = processed_content_parts  # 使用处理后的结构化内容列表
        return new_msg  # 返回处理后的消息

    elif content_format == "string":  # string格式
        # String format: flatten to text only (for templates like DeepSeek)  # 字符串格式：仅展平为文本（用于DeepSeek等模板）
        text_parts = []  # 文本部分列表
        for chunk in msg_dict["content"]:  # 遍历content列表
            if isinstance(chunk, dict) and chunk.get("type") == "text":  # 如果是文本类型字典
                text_parts.append(chunk["text"])  # 追加文本内容
            # Note: For string format, we ignore images/audio since the template  # 注意：对于string格式，忽略图像/音频，因为模板
            # doesn't expect structured content - multimodal placeholders would  # 不期望结构化内容 - 多模态占位符需要
            # need to be inserted differently  # 以不同方式插入

        new_msg = msg_dict.copy()  # 复制消息字典
        new_msg["content"] = " ".join(text_parts) if text_parts else ""  # 将文本部分合并为字符串
        new_msg = {k: v for k, v in new_msg.items() if v is not None}  # 过滤None值
        return new_msg  # 返回处理后的消息

    else:  # 其他格式
        raise ValueError(f"Invalid content format: {content_format}")  # 抛出值错误异常
