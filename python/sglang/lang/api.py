"""Public APIs of the language."""
# 本文件提供SGLang语言的公共API，包括函数装饰器、运行时/引擎创建、生成函数、图像/视频处理、选择函数、角色函数和推理分离功能。

import re  # 导入正则表达式模块，用于验证regex参数
from typing import Callable, List, Optional, Union  # 导入类型注解工具

from sglang.global_config import global_config  # 导入全局配置对象
from sglang.lang.backend.base_backend import BaseBackend  # 导入后端基类
from sglang.lang.choices import ChoicesSamplingMethod, token_length_normalized  # 导入选择采样方法和默认的token长度归一化方法
from sglang.lang.ir import (  # 导入SGLang中间表示（IR）中的各种表达式类
    SglExpr,  # SGLang表达式基类
    SglExprList,  # SGLang表达式列表类
    SglFunction,  # SGLang函数类
    SglGen,  # SGLang生成表达式类
    SglImage,  # SGLang图像表达式类
    SglRoleBegin,  # SGLang角色开始标记类
    SglRoleEnd,  # SGLang角色结束标记类
    SglSelect,  # SGLang选择表达式类
    SglSeparateReasoning,  # SGLang推理分离表达式类
    SglVideo,  # SGLang视频表达式类
)


def function(
    func: Optional[Callable] = None, num_api_spec_tokens: Optional[int] = None
):
    """函数装饰器，用于将Python函数包装为SGLang函数，支持直接调用或带参数装饰。"""
    if func:  # 如果直接传入了函数（无参数装饰器用法）
        return SglFunction(func, num_api_spec_tokens=num_api_spec_tokens)  # 直接包装函数并返回SglFunction实例

    def decorator(func):  # 定义内部装饰器函数，用于带参数的装饰器用法
        return SglFunction(func, num_api_spec_tokens=num_api_spec_tokens)  # 包装函数并返回SglFunction实例

    return decorator  # 返回装饰器函数


def Runtime(*args, **kwargs):
    """创建并返回一个Runtime实例，用于连接到运行中的SGLang服务端点。"""
    # Avoid importing unnecessary dependency  # 避免导入不必要的依赖
    from sglang.lang.backend.runtime_endpoint import Runtime  # 延迟导入Runtime类

    return Runtime(*args, **kwargs)  # 创建并返回Runtime实例


def Engine(*args, **kwargs):
    """创建并返回一个Engine实例，用于启动本地SGLang推理引擎。"""
    # Avoid importing unnecessary dependency  # 避免导入不必要的依赖
    from sglang.srt.entrypoints.engine import Engine  # 延迟导入Engine类

    return Engine(*args, **kwargs)  # 创建并返回Engine实例


def set_default_backend(backend: BaseBackend):
    """设置全局默认后端，所有SGLang函数调用将使用此后端执行。"""
    global_config.default_backend = backend  # 将指定的后端设置为全局默认后端


def flush_cache(backend: Optional[BaseBackend] = None):
    """刷新后端的KV缓存，释放显存。可指定后端或使用默认后端。"""
    backend = backend or global_config.default_backend  # 使用指定的后端，若未指定则使用全局默认后端
    if backend is None:  # 如果后端为None（未设置默认后端）
        return False  # 返回False表示刷新失败

    # If backend is Runtime  # 如果后端是Runtime类型
    if hasattr(backend, "endpoint"):  # 检查后端是否具有endpoint属性
        backend = backend.endpoint  # 获取Runtime内部的endpoint对象
    return backend.flush_cache()  # 调用后端的flush_cache方法刷新缓存并返回结果


def get_server_info(backend: Optional[BaseBackend] = None):
    """获取服务端信息，如模型名称、最大上下文长度等。可指定后端或使用默认后端。"""
    backend = backend or global_config.default_backend  # 使用指定的后端，若未指定则使用全局默认后端
    if backend is None:  # 如果后端为None（未设置默认后端）
        return None  # 返回None

    # If backend is Runtime  # 如果后端是Runtime类型
    if hasattr(backend, "endpoint"):  # 检查后端是否具有endpoint属性
        backend = backend.endpoint  # 获取Runtime内部的endpoint对象
    return backend.get_server_info()  # 调用后端的get_server_info方法获取服务信息


def gen(
    name: Optional[str] = None,  # 生成结果的变量名
    max_tokens: Optional[int] = None,  # 最大生成token数
    min_tokens: Optional[int] = None,  # 最小生成token数
    n: Optional[int] = None,  # 生成候选数量
    stop: Optional[Union[str, List[str]]] = None,  # 停止生成的字符串或字符串列表
    stop_token_ids: Optional[List[int]] = None,  # 停止生成的token ID列表
    stop_regex: Optional[Union[str, List[str]]] = None,  # 停止生成的正则表达式
    temperature: Optional[float] = None,  # 采样温度
    top_p: Optional[float] = None,  # top-p采样参数
    top_k: Optional[int] = None,  # top-k采样参数
    min_p: Optional[float] = None,  # min-p采样参数
    frequency_penalty: Optional[float] = None,  # 频率惩罚系数
    presence_penalty: Optional[float] = None,  # 存在惩罚系数
    ignore_eos: Optional[bool] = None,  # 是否忽略结束符
    return_logprob: Optional[bool] = None,  # 是否返回对数概率
    logprob_start_len: Optional[int] = None,  # 对数概率计算的起始位置
    top_logprobs_num: Optional[int] = None,  # 返回的top对数概率数量
    return_text_in_logprobs: Optional[bool] = None,  # 是否在对数概率中返回文本
    dtype: Optional[Union[type, str]] = None,  # 生成结果的数据类型
    choices: Optional[List[str]] = None,  # 受限选择的候选列表
    choices_method: Optional[ChoicesSamplingMethod] = None,  # 选择采样方法
    regex: Optional[str] = None,  # 生成结果必须匹配的正则表达式
    json_schema: Optional[str] = None,  # 生成结果必须符合的JSON Schema
):
    """调用模型进行文本生成。参数含义详见docs/backend/sampling_params.md。"""
    """Call the model to generate. See the meaning of the arguments in docs/backend/sampling_params.md"""

    if choices:  # 如果提供了choices参数，则转换为选择模式
        return SglSelect(  # 返回SglSelect选择表达式
            name,  # 变量名
            choices,  # 候选列表
            0.0 if temperature is None else temperature,  # 默认温度为0.0（确定性选择）
            token_length_normalized if choices_method is None else choices_method,  # 默认使用token长度归一化方法
        )

    # check regex is valid  # 检查正则表达式是否有效
    if regex is not None:  # 如果提供了regex参数
        try:  # 尝试编译正则表达式
            re.compile(regex)  # 编译正则表达式以验证其有效性
        except re.error as e:  # 如果正则表达式无效，捕获异常
            raise e  # 抛出正则表达式错误异常

    return SglGen(  # 返回SglGen生成表达式
        name,  # 变量名
        max_tokens,  # 最大生成token数
        min_tokens,  # 最小生成token数
        n,  # 生成候选数量
        stop,  # 停止字符串
        stop_token_ids,  # 停止token ID列表
        stop_regex,  # 停止正则表达式
        temperature,  # 采样温度
        top_p,  # top-p参数
        top_k,  # top-k参数
        min_p,  # min-p参数
        frequency_penalty,  # 频率惩罚
        presence_penalty,  # 存在惩罚
        ignore_eos,  # 是否忽略结束符
        return_logprob,  # 是否返回对数概率
        logprob_start_len,  # 对数概率起始位置
        top_logprobs_num,  # top对数概率数量
        return_text_in_logprobs,  # 是否在对数概率中返回文本
        dtype,  # 数据类型
        regex,  # 正则约束
        json_schema,  # JSON Schema约束
    )


def gen_int(
    name: Optional[str] = None,  # 生成结果的变量名
    max_tokens: Optional[int] = None,  # 最大生成token数
    n: Optional[int] = None,  # 生成候选数量
    stop: Optional[Union[str, List[str]]] = None,  # 停止生成的字符串或字符串列表
    stop_token_ids: Optional[List[int]] = None,  # 停止生成的token ID列表
    stop_regex: Optional[Union[str, List[str]]] = None,  # 停止生成的正则表达式
    temperature: Optional[float] = None,  # 采样温度
    top_p: Optional[float] = None,  # top-p采样参数
    top_k: Optional[int] = None,  # top-k采样参数
    min_p: Optional[float] = None,  # min-p采样参数
    frequency_penalty: Optional[float] = None,  # 频率惩罚系数
    presence_penalty: Optional[float] = None,  # 存在惩罚系数
    ignore_eos: Optional[bool] = None,  # 是否忽略结束符
    return_logprob: Optional[bool] = None,  # 是否返回对数概率
    logprob_start_len: Optional[int] = None,  # 对数概率计算的起始位置
    top_logprobs_num: Optional[int] = None,  # 返回的top对数概率数量
    return_text_in_logprobs: Optional[bool] = None,  # 是否在对数概率中返回文本
):
    """生成整数类型的结果，是gen函数的整数特化版本，dtype自动设为int。"""
    return SglGen(  # 返回SglGen生成表达式，dtype固定为int
        name,  # 变量名
        max_tokens,  # 最大生成token数
        None,  # min_tokens不设置
        n,  # 生成候选数量
        stop,  # 停止字符串
        stop_token_ids,  # 停止token ID列表
        stop_regex,  # 停止正则表达式
        temperature,  # 采样温度
        top_p,  # top-p参数
        top_k,  # top-k参数
        min_p,  # min-p参数
        frequency_penalty,  # 频率惩罚
        presence_penalty,  # 存在惩罚
        ignore_eos,  # 是否忽略结束符
        return_logprob,  # 是否返回对数概率
        logprob_start_len,  # 对数概率起始位置
        top_logprobs_num,  # top对数概率数量
        return_text_in_logprobs,  # 是否在对数概率中返回文本
        int,  # 数据类型固定为int
        None,  # regex不设置
    )


def gen_string(
    name: Optional[str] = None,  # 生成结果的变量名
    max_tokens: Optional[int] = None,  # 最大生成token数
    n: Optional[int] = None,  # 生成候选数量
    stop: Optional[Union[str, List[str]]] = None,  # 停止生成的字符串或字符串列表
    stop_token_ids: Optional[List[int]] = None,  # 停止生成的token ID列表
    stop_regex: Optional[Union[str, List[str]]] = None,  # 停止生成的正则表达式
    temperature: Optional[float] = None,  # 采样温度
    top_p: Optional[float] = None,  # top-p采样参数
    top_k: Optional[int] = None,  # top-k采样参数
    min_p: Optional[float] = None,  # min-p采样参数
    frequency_penalty: Optional[float] = None,  # 频率惩罚系数
    presence_penalty: Optional[float] = None,  # 存在惩罚系数
    ignore_eos: Optional[bool] = None,  # 是否忽略结束符
    return_logprob: Optional[bool] = None,  # 是否返回对数概率
    logprob_start_len: Optional[int] = None,  # 对数概率计算的起始位置
    top_logprobs_num: Optional[int] = None,  # 返回的top对数概率数量
    return_text_in_logprobs: Optional[bool] = None,  # 是否在对数概率中返回文本
):
    """生成字符串类型的结果，是gen函数的字符串特化版本，dtype自动设为str。"""
    return SglGen(  # 返回SglGen生成表达式，dtype固定为str
        name,  # 变量名
        max_tokens,  # 最大生成token数
        None,  # min_tokens不设置
        n,  # 生成候选数量
        stop,  # 停止字符串
        stop_token_ids,  # 停止token ID列表
        stop_regex,  # 停止正则表达式
        temperature,  # 采样温度
        top_p,  # top-p参数
        top_k,  # top-k参数
        min_p,  # min-p参数
        frequency_penalty,  # 频率惩罚
        presence_penalty,  # 存在惩罚
        ignore_eos,  # 是否忽略结束符
        return_logprob,  # 是否返回对数概率
        logprob_start_len,  # 对数概率起始位置
        top_logprobs_num,  # top对数概率数量
        return_text_in_logprobs,  # 是否在对数概率中返回文本
        str,  # 数据类型固定为str
        None,  # regex不设置
    )


def image(expr: SglExpr):
    """创建图像表达式，用于在提示中嵌入图像输入。"""
    return SglImage(expr)  # 返回SglImage图像表达式实例


def video(path: str, num_frames: int):
    """创建视频表达式，用于在提示中嵌入视频输入，需指定视频路径和采样帧数。"""
    return SglVideo(path, num_frames)  # 返回SglVideo视频表达式实例，传入路径和帧数


def select(
    name: Optional[str] = None,  # 选择结果的变量名
    choices: Optional[List[str]] = None,  # 候选选项列表
    temperature: float = 0.0,  # 采样温度，默认0.0表示确定性选择
    choices_method: ChoicesSamplingMethod = token_length_normalized,  # 选择采样方法，默认使用token长度归一化
):
    """从候选列表中选择一个选项，通过约束生成实现受控选择。"""
    assert choices is not None  # 断言choices必须提供，否则报错
    return SglSelect(name, choices, temperature, choices_method)  # 返回SglSelect选择表达式实例


def _role_common(name: str, expr: Optional[SglExpr] = None):
    """角色通用函数，构建角色开始-内容-角色结束的表达式列表。"""
    if expr is None:  # 如果没有提供表达式内容
        return SglExprList([SglRoleBegin(name), SglRoleEnd(name)])  # 返回仅包含角色开始和结束标记的表达式列表
    else:  # 如果提供了表达式内容
        return SglExprList([SglRoleBegin(name), expr, SglRoleEnd(name)])  # 返回包含角色开始、内容和结束标记的表达式列表


def system(expr: Optional[SglExpr] = None):
    """创建system角色的表达式，用于设置系统提示词。"""
    return _role_common("system", expr)  # 调用角色通用函数，角色名为"system"


def user(expr: Optional[SglExpr] = None):
    """创建user角色的表达式，用于设置用户消息。"""
    return _role_common("user", expr)  # 调用角色通用函数，角色名为"user"


def assistant(expr: Optional[SglExpr] = None):
    """创建assistant角色的表达式，用于设置助手回复。"""
    return _role_common("assistant", expr)  # 调用角色通用函数，角色名为"assistant"


def system_begin():
    """返回system角色的开始标记表达式。"""
    return SglRoleBegin("system")  # 返回角色名为"system"的开始标记


def system_end():
    """返回system角色的结束标记表达式。"""
    return SglRoleEnd("system")  # 返回角色名为"system"的结束标记


def user_begin():
    """返回user角色的开始标记表达式。"""
    return SglRoleBegin("user")  # 返回角色名为"user"的开始标记


def user_end():
    """返回user角色的结束标记表达式。"""
    return SglRoleEnd("user")  # 返回角色名为"user"的结束标记


def assistant_begin():
    """返回assistant角色的开始标记表达式。"""
    return SglRoleBegin("assistant")  # 返回角色名为"assistant"的开始标记


def assistant_end():
    """返回assistant角色的结束标记表达式。"""
    return SglRoleEnd("assistant")  # 返回角色名为"assistant"的结束标记


def separate_reasoning(
    expr: Optional[SglExpr] = None, model_type: Optional[str] = None
):
    """分离推理内容，用于将思考过程与最终回答分开处理，支持指定模型类型。"""
    return SglExprList([expr, SglSeparateReasoning(model_type, expr=expr)])  # 返回包含原表达式和推理分离表达式的列表
