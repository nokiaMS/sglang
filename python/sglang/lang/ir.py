"""The intermediate representation."""
# 本文件定义了SGLang的中间表示（IR），包括采样参数、函数包装器以及各种表达式类型，
# 用于在SGLang编程语言中描述和控制语言模型的生成行为。

import dataclasses  # 导入dataclasses模块，用于定义数据类
import inspect  # 导入inspect模块，用于获取函数参数信息
import warnings  # 导入warnings模块，用于发出警告信息
from typing import List, Optional, Union  # 导入类型提示相关的工具

from sglang.global_config import global_config  # 导入全局配置
from sglang.lang.choices import ChoicesSamplingMethod  # 导入选择采样方法的枚举

REGEX_INT = r"[-+]?[0-9]+[ \n]*"  # 匹配整数的正则表达式
REGEX_FLOAT = r"[-+]?[0-9]*\.?[0-9]+[ \n]*"  # 匹配浮点数的正则表达式
REGEX_BOOL = r"(True|False)"  # 匹配布尔值的正则表达式
REGEX_STR = r"\"[\w\d\s]*\""  # bugs with regex r"\".*\"" in interegular pkg  # 匹配字符串的正则表达式（简化版本，避免interegular包的bug）


@dataclasses.dataclass  # 使用dataclass装饰器，自动生成__init__等方法
class SglSamplingParams:  # 采样参数数据类，定义模型生成时的各种采样参数
    max_new_tokens: int = 128  # 最大生成token数，默认128
    min_new_tokens: int = 0  # 最小生成token数，默认0
    n: int = 1  # 生成候选数量，默认1
    stop: Union[str, List[str]] = ()  # 停止生成的字符串或字符串列表
    stop_token_ids: Optional[List[int]] = ()  # 停止生成的token ID列表
    stop_regex: Optional[Union[str, List[str]]] = ()  # 停止生成的正则表达式
    temperature: float = 1.0  # 采样温度，默认1.0
    top_p: float = 1.0  # top-p采样参数，默认1.0
    top_k: int = -1  # top-k采样参数，-1表示禁用  # -1 means disable
    min_p: float = 0.0  # min-p采样参数，默认0.0
    frequency_penalty: float = 0.0  # 频率惩罚参数，默认0.0
    presence_penalty: float = 0.0  # 存在惩罚参数，默认0.0
    ignore_eos: bool = False  # 是否忽略结束符，默认False
    return_logprob: Optional[bool] = None  # 是否返回对数概率
    logprob_start_len: Optional[int] = None  # 对数概率起始长度
    top_logprobs_num: Optional[int] = None  # 返回的top对数概率数量
    return_text_in_logprobs: Optional[bool] = None  # 对数概率中是否返回文本
    json_schema: Optional[str] = None  # JSON schema约束

    # for constrained generation, not included in to_xxx_kwargs  # 用于受约束生成，不包含在to_xxx_kwargs中
    dtype: Optional[str] = None  # 数据类型约束
    regex: Optional[str] = None  # 正则表达式约束

    def clone(self):  # 克隆当前采样参数对象，返回一个新的SglSamplingParams实例
        return SglSamplingParams(  # 返回一个新的SglSamplingParams实例
            self.max_new_tokens,  # 最大生成token数
            self.min_new_tokens,  # 最小生成token数
            self.n,  # 生成候选数量
            self.stop,  # 停止字符串
            self.stop_token_ids,  # 停止token ID列表
            self.stop_regex,  # 停止正则表达式
            self.temperature,  # 采样温度
            self.top_p,  # top-p参数
            self.top_k,  # top-k参数
            self.min_p,  # min-p参数
            self.frequency_penalty,  # 频率惩罚
            self.presence_penalty,  # 存在惩罚
            self.ignore_eos,  # 是否忽略结束符
            self.return_logprob,  # 是否返回对数概率
            self.logprob_start_len,  # 对数概率起始长度
            self.top_logprobs_num,  # top对数概率数量
            self.return_text_in_logprobs,  # 对数概率中是否返回文本
            self.json_schema,  # JSON schema约束
        )

    def to_openai_kwargs(self):  # 将采样参数转换为OpenAI API的关键字参数格式
        # OpenAI does not support top_k, so we drop it here  # OpenAI不支持top_k，因此这里忽略它
        if self.regex is not None:  # 如果设置了正则表达式约束
            warnings.warn("Regular expression is not supported in the OpenAI backend.")  # 发出警告：OpenAI后端不支持正则表达式
        return {  # 返回OpenAI API格式的参数字典
            "max_tokens": self.max_new_tokens,  # 最大token数
            "max_completion_tokens": self.max_new_tokens,  # 最大完成token数
            "n": self.n,  # 候选数量
            "stop": self.stop or None,  # 停止字符串，空元组转为None
            "temperature": self.temperature,  # 采样温度
            "top_p": self.top_p,  # top-p参数
            "frequency_penalty": self.frequency_penalty,  # 频率惩罚
            "presence_penalty": self.presence_penalty,  # 存在惩罚
        }

    def to_vertexai_kwargs(self):  # 将采样参数转换为VertexAI API的关键字参数格式
        if self.regex is not None:  # 如果设置了正则表达式约束
            warnings.warn(  # 发出警告
                "Regular expression is not supported in the VertexAI backend."  # VertexAI后端不支持正则表达式
            )
        return {  # 返回VertexAI API格式的参数字典
            "candidate_count": 1,  # 候选数量，VertexAI固定为1
            "max_output_tokens": self.max_new_tokens,  # 最大输出token数
            "stop_sequences": self.stop,  # 停止序列
            "temperature": self.temperature,  # 采样温度
            "top_p": self.top_p,  # top-p参数
            "top_k": self.top_k if self.top_k > 0 else None,  # top-k参数，-1转为None
        }

    def to_anthropic_kwargs(self):  # 将采样参数转换为Anthropic API的关键字参数格式
        # Anthropic does not support frequency_penalty or presence_penalty, so we drop it here  # Anthropic不支持frequency_penalty和presence_penalty，因此这里忽略它们
        if self.regex is not None:  # 如果设置了正则表达式约束
            warnings.warn(  # 发出警告
                "Regular expression is not supported in the Anthropic backend."  # Anthropic后端不支持正则表达式
            )
        return {  # 返回Anthropic API格式的参数字典
            "max_tokens": self.max_new_tokens,  # 最大token数
            "stop_sequences": (  # 停止序列
                self.stop if isinstance(self.stop, (list, tuple)) else [self.stop]  # 如果stop是列表或元组则直接使用，否则包装为列表
            ),
            "temperature": self.temperature,  # 采样温度
            "top_p": self.top_p,  # top-p参数
            "top_k": self.top_k,  # top-k参数
        }

    def to_litellm_kwargs(self):  # 将采样参数转换为LiteLLM API的关键字参数格式
        if self.regex is not None:  # 如果设置了正则表达式约束
            warnings.warn("Regular expression is not supported in the LiteLLM backend.")  # 发出警告：LiteLLM后端不支持正则表达式
        return {  # 返回LiteLLM API格式的参数字典
            "max_tokens": self.max_new_tokens,  # 最大token数
            "stop": self.stop or None,  # 停止字符串，空元组转为None
            "temperature": self.temperature,  # 采样温度
            "top_p": self.top_p,  # top-p参数
            "frequency_penalty": self.frequency_penalty,  # 频率惩罚
            "presence_penalty": self.presence_penalty,  # 存在惩罚
        }

    def to_srt_kwargs(self):  # 将采样参数转换为SRT（SGLang Runtime）后端的关键字参数格式
        return {  # 返回SRT后端格式的参数字典
            "max_new_tokens": self.max_new_tokens,  # 最大生成token数
            "min_new_tokens": self.min_new_tokens,  # 最小生成token数
            "n": self.n,  # 候选数量
            "stop": self.stop,  # 停止字符串
            "stop_token_ids": self.stop_token_ids,  # 停止token ID列表
            "stop_regex": self.stop_regex,  # 停止正则表达式
            "temperature": self.temperature,  # 采样温度
            "top_p": self.top_p,  # top-p参数
            "top_k": self.top_k,  # top-k参数
            "min_p": self.min_p,  # min-p参数
            "frequency_penalty": self.frequency_penalty,  # 频率惩罚
            "presence_penalty": self.presence_penalty,  # 存在惩罚
            "ignore_eos": self.ignore_eos,  # 是否忽略结束符
            "regex": self.regex,  # 正则表达式约束
            "json_schema": self.json_schema,  # JSON schema约束
        }


class SglFunction:  # SGLang函数包装器，封装用户定义的SGLang函数
    def __init__(self, func, num_api_spec_tokens=None, bind_arguments=None):  # 初始化SglFunction，接受函数、API规范token数和绑定参数
        self.func = func  # 保存被包装的原始函数
        self.num_api_spec_tokens = num_api_spec_tokens  # API规范token数量
        self.bind_arguments = bind_arguments or {}  # 绑定的参数字典，默认为空字典
        self.pin_prefix_rid = None  # 固定前缀的请求ID，默认为None

        # Parse arguments  # 解析函数参数
        argspec = inspect.getfullargspec(func)  # 获取函数的完整参数规格
        assert argspec.args[0] == "s", 'The first argument must be "s"'  # 断言第一个参数必须名为"s"
        self.arg_names = argspec.args[1:]  # 保存除第一个参数"s"之外的参数名列表
        self.arg_defaults = argspec.defaults if argspec.defaults is not None else []  # 保存参数默认值列表

    def bind(self, **kwargs):  # 绑定参数到函数，返回一个新的SglFunction实例
        assert all(key in self.arg_names for key in kwargs)  # 断言所有绑定的参数名都在函数参数名列表中

        new_bind_dict = {**self.bind_arguments, **kwargs}  # 合并已有的绑定参数和新传入的绑定参数
        return SglFunction(self.func, bind_arguments=new_bind_dict)  # 返回绑定了新参数的SglFunction实例

    def run(  # 运行SGLang函数，执行单次推理
        self,
        *args,  # 位置参数
        max_new_tokens: int = 128,  # 最大生成token数
        n: int = 1,  # 候选数量
        stop: Optional[Union[str, List[str]]] = None,  # 停止字符串
        stop_token_ids: Optional[List[int]] = None,  # 停止token ID列表
        stop_regex: Optional[Union[str, List[str]]] = None,  # 停止正则表达式
        temperature: float = 1.0,  # 采样温度
        top_p: float = 1.0,  # top-p参数
        top_k: int = -1,  # top-k参数
        min_p: float = 0.0,  # min-p参数
        frequency_penalty: float = 0.0,  # 频率惩罚
        presence_penalty: float = 0.0,  # 存在惩罚
        ignore_eos: bool = False,  # 是否忽略结束符
        return_logprob: Optional[bool] = None,  # 是否返回对数概率
        logprob_start_len: Optional[int] = None,  # 对数概率起始长度
        top_logprobs_num: Optional[int] = None,  # top对数概率数量
        return_text_in_logprobs: Optional[bool] = None,  # 对数概率中是否返回文本
        stream: bool = False,  # 是否使用流式输出
        backend=None,  # 后端引擎
        use_thread: bool = True,  # 是否使用线程
        **kwargs,  # 其他关键字参数
    ):  # 执行函数的run方法，运行单次推理
        from sglang.lang.interpreter import run_program  # 延迟导入解释器的run_program函数

        # avoid using [] as the default arg: https://nikos7am.com/posts/mutable-default-arguments/  # 避免使用[]作为默认参数，防止可变默认参数陷阱
        if stop is None:  # 如果stop为None
            stop = []  # 将stop设为空列表
        if stop_token_ids is None:  # 如果stop_token_ids为None
            stop_token_ids = []  # 将stop_token_ids设为空列表
        if stop_regex is None:  # 如果stop_regex为None
            stop_regex = []  # 将stop_regex设为空列表

        default_sampling_para = SglSamplingParams(  # 创建默认采样参数对象
            max_new_tokens=max_new_tokens,  # 最大生成token数
            n=n,  # 候选数量
            stop=stop,  # 停止字符串
            stop_token_ids=stop_token_ids,  # 停止token ID列表
            stop_regex=stop_regex,  # 停止正则表达式
            temperature=temperature,  # 采样温度
            top_p=top_p,  # top-p参数
            top_k=top_k,  # top-k参数
            min_p=min_p,  # min-p参数
            frequency_penalty=frequency_penalty,  # 频率惩罚
            presence_penalty=presence_penalty,  # 存在惩罚
            ignore_eos=ignore_eos,  # 是否忽略结束符
            return_logprob=return_logprob,  # 是否返回对数概率
            logprob_start_len=logprob_start_len,  # 对数概率起始长度
            top_logprobs_num=top_logprobs_num,  # top对数概率数量
            return_text_in_logprobs=return_text_in_logprobs,  # 对数概率中是否返回文本
        )
        backend = backend or global_config.default_backend  # 如果未指定后端，使用全局默认后端
        return run_program(  # 调用解释器的run_program函数执行推理
            self,  # SglFunction自身
            backend,  # 后端引擎
            args,  # 位置参数
            kwargs,  # 关键字参数
            default_sampling_para,  # 默认采样参数
            stream,  # 是否流式输出
            use_thread=use_thread,  # 是否使用线程
        )

    def run_batch(  # 批量运行SGLang函数，对多组输入执行推理
        self,
        batch_kwargs,  # 批量输入参数列表
        *,
        max_new_tokens: int = 128,  # 最大生成token数
        n: int = 1,  # 候选数量
        stop: Optional[Union[str, List[str]]] = None,  # 停止字符串
        stop_token_ids: Optional[List[int]] = None,  # 停止token ID列表
        stop_regex: Optional[Union[str, List[str]]] = None,  # 停止正则表达式
        temperature: float = 1.0,  # 采样温度
        top_p: float = 1.0,  # top-p参数
        top_k: int = -1,  # top-k参数
        min_p: float = 0.0,  # min-p参数
        frequency_penalty: float = 0.0,  # 频率惩罚
        presence_penalty: float = 0.0,  # 存在惩罚
        ignore_eos: bool = False,  # 是否忽略结束符
        return_logprob: Optional[bool] = None,  # 是否返回对数概率
        logprob_start_len: Optional[int] = None,  # 对数概率起始长度
        top_logprobs_num: Optional[int] = None,  # top对数概率数量
        return_text_in_logprobs: Optional[bool] = None,  # 对数概率中是否返回文本
        backend=None,  # 后端引擎
        num_threads: Union[str, int] = "auto",  # 线程数量，默认自动
        progress_bar: bool = False,  # 是否显示进度条
        generator_style: bool = False,  # 是否使用生成器风格
    ):  # 执行函数的run_batch方法，批量运行推理
        from sglang.lang.interpreter import run_program_batch  # 延迟导入解释器的run_program_batch函数

        if stop is None:  # 如果stop为None
            stop = []  # 将stop设为空列表
        if stop_token_ids is None:  # 如果stop_token_ids为None
            stop_token_ids = []  # 将stop_token_ids设为空列表
        if stop_regex is None:  # 如果stop_regex为None
            stop_regex = []  # 将stop_regex设为空列表

        assert isinstance(batch_kwargs, (list, tuple))  # 断言batch_kwargs必须是列表或元组
        if len(batch_kwargs) == 0:  # 如果批量参数列表为空
            return []  # 返回空列表
        if not isinstance(batch_kwargs[0], dict):  # 如果第一个元素不是字典（即位置参数形式）
            num_programs = len(batch_kwargs)  # 记录原始程序数量
            # change the list of argument values to dict of arg_name -> arg_value  # 将参数值列表转换为参数名到参数值的字典
            batch_kwargs = [  # 重新构建batch_kwargs
                {self.arg_names[i]: v for i, v in enumerate(arg_values)}  # 将位置参数转为关键字参数字典
                for arg_values in batch_kwargs  # 遍历每组参数值
                if isinstance(arg_values, (list, tuple))  # 只处理列表或元组类型的参数
                and len(self.arg_names) - len(self.arg_defaults)  # 确保参数数量不小于必选参数数量
                <= len(arg_values)
                <= len(self.arg_names)  # 确保参数数量不超过总参数数量
            ]
            # Ensure to raise an exception if the number of arguments mismatch  # 确保在参数数量不匹配时抛出异常
            if len(batch_kwargs) != num_programs:  # 如果转换后的数量与原始数量不一致
                raise Exception("Given arguments mismatch the SGL function signature")  # 抛出参数不匹配异常

        default_sampling_para = SglSamplingParams(  # 创建默认采样参数对象
            max_new_tokens=max_new_tokens,  # 最大生成token数
            n=n,  # 候选数量
            stop=stop,  # 停止字符串
            stop_token_ids=stop_token_ids,  # 停止token ID列表
            stop_regex=stop_regex,  # 停止正则表达式
            temperature=temperature,  # 采样温度
            top_p=top_p,  # top-p参数
            top_k=top_k,  # top-k参数
            min_p=min_p,  # min-p参数
            frequency_penalty=frequency_penalty,  # 频率惩罚
            presence_penalty=presence_penalty,  # 存在惩罚
            ignore_eos=ignore_eos,  # 是否忽略结束符
            return_logprob=return_logprob,  # 是否返回对数概率
            logprob_start_len=logprob_start_len,  # 对数概率起始长度
            top_logprobs_num=top_logprobs_num,  # top对数概率数量
            return_text_in_logprobs=return_text_in_logprobs,  # 对数概率中是否返回文本
        )
        backend = backend or global_config.default_backend  # 如果未指定后端，使用全局默认后端
        return run_program_batch(  # 调用解释器的run_program_batch函数执行批量推理
            self,  # SglFunction自身
            backend,  # 后端引擎
            batch_kwargs,  # 批量输入参数
            default_sampling_para,  # 默认采样参数
            num_threads,  # 线程数量
            progress_bar,  # 是否显示进度条
            generator_style=generator_style,  # 是否使用生成器风格
        )

    def trace(self, *, backend=None, **kwargs):  # 跟踪函数执行，用于调试和可视化
        from sglang.lang.tracer import trace_program  # 延迟导入跟踪器的trace_program函数

        backend = backend or global_config.default_backend  # 如果未指定后端，使用全局默认后端
        return trace_program(self, kwargs, backend)  # 调用跟踪器函数执行跟踪

    def cache(self, backend=None):  # 缓存函数的执行结果
        from sglang.lang.interpreter import cache_program  # 延迟导入解释器的cache_program函数

        backend = backend or global_config.default_backend  # 如果未指定后端，使用全局默认后端
        return cache_program(self, backend)  # 调用缓存函数

    def __call__(self, *args, **kwargs):  # 使SglFunction实例可调用，根据上下文选择运行或跟踪
        from sglang.lang.tracer import TracingScope  # 延迟导入跟踪作用域

        tracing_scope = TracingScope.get_current_scope()  # 获取当前的跟踪作用域
        if tracing_scope is None:  # 如果不在跟踪作用域中
            return self.run(*args, **kwargs)  # 直接运行函数
        else:  # 如果在跟踪作用域中
            kwargs["backend"] = tracing_scope.tracer_state.backend  # 将跟踪作用域的后端传入
            return self.trace(*args, **kwargs)  # 执行跟踪


class SglExpr:  # SGLang表达式基类，所有IR节点的父类
    node_ct = 0  # 全局节点计数器，用于生成唯一的节点ID

    def __init__(self):  # 初始化表达式节点
        self.node_id = SglExpr.node_ct  # 分配当前节点ID
        self.prev_node = None  # 前驱节点，用于构建链式IR图
        self.pid = None  # 进程ID，用于并行执行
        SglExpr.node_ct += 1  # 递增全局节点计数器

    def __add__(self, other):  # 重载加法运算符，实现表达式拼接（self + other）
        if isinstance(other, str):  # 如果右侧操作数是字符串
            other = SglConstantText(other)  # 将字符串转换为常量文本表达式
        assert isinstance(other, SglExpr)  # 断言右侧操作数必须是SglExpr类型

        return self.concatenate_ir(self, other)  # 拼接两个表达式并返回

    def __radd__(self, other):  # 重载右加法运算符，实现表达式拼接（other + self）
        if isinstance(other, str):  # 如果左侧操作数是字符串
            other = SglConstantText(other)  # 将字符串转换为常量文本表达式
        assert isinstance(other, SglExpr), f"{other}"  # 断言左侧操作数必须是SglExpr类型

        return self.concatenate_ir(other, self)  # 拼接两个表达式并返回

    def concatenate_ir(self, a, b):  # 拼接两个IR表达式，返回一个SglExprList
        if isinstance(a, SglExprList):  # 如果a是表达式列表
            if isinstance(b, SglExprList):  # 如果b也是表达式列表
                return SglExprList(a.expr_list + b.expr_list)  # 合并两个列表
            else:  # b不是表达式列表
                return SglExprList(a.expr_list + [b])  # 将b追加到a的列表中
        elif isinstance(b, SglExprList):  # 如果b是表达式列表但a不是
            return SglExprList([a] + b.expr_list)  # 将a追加到b的列表前面

        return SglExprList([a, b])  # 两者都不是列表，创建新的列表包含a和b

    def print_graph_dfs(self):  # 以深度优先搜索方式打印IR图
        ret = [""]  # 用列表包装结果字符串，以便在嵌套函数中修改
        visited = set()  # 已访问节点集合，防止重复访问

        def dfs_print(x):  # 深度优先搜索打印函数
            if x is None or x in visited:  # 如果节点为空或已访问过
                return  # 直接返回
            visited.add(x)  # 将节点标记为已访问

            # Print dependency  # 打印依赖节点
            if x.prev_node is not None:  # 如果存在前驱节点
                dfs_print(x.prev_node)  # 递归打印前驱节点

            if isinstance(x, SglExprList):  # 如果是表达式列表
                for y in x.expr_list:  # 遍历列表中的每个表达式
                    dfs_print(y)  # 递归打印每个表达式
            # elif isinstance(x, SglRole):  # 如果是角色表达式（已注释）
            #    dfs_print(x.expr)  # 递归打印角色内的表达式（已注释）
            elif isinstance(x, SglVariable):  # 如果是变量表达式
                dfs_print(x.source)  # 递归打印变量的来源节点

            # Print the node itself  # 打印节点自身
            if isinstance(x, (SglFork, SglGetForkItem)):  # 如果是Fork或GetForkItem节点
                ret[0] += f"%{x.node_id} = {x}\n"  # 格式化输出
            else:  # 其他类型的节点
                if x.prev_node is not None:  # 如果存在前驱节点
                    ret[0] += (  # 格式化输出，包含前驱节点ID
                        f"%{x.node_id} = %{x.prev_node.node_id} + " + str(x) + "\n"
                    )
                else:  # 没有前驱节点
                    ret[0] += f"%{x.node_id} = " + str(x) + "\n"  # 格式化输出

        dfs_print(self)  # 从当前节点开始深度优先搜索打印
        return ret[0]  # 返回打印结果字符串


class SglExprList(SglExpr):  # 表达式列表，包含多个子表达式的IR节点
    def __init__(self, expr_list: List[SglExpr]):  # 初始化表达式列表
        super().__init__()  # 调用父类初始化
        self.expr_list = expr_list  # 保存子表达式列表

    def __repr__(self):  # 返回表达式列表的字符串表示
        return f"ExprList({self.expr_list})"  # 格式化输出


class SglArgument(SglExpr):  # 参数表达式，表示传入SGLang函数的参数
    def __init__(self, name: str, value: str):  # 初始化参数表达式
        super().__init__()  # 调用父类初始化
        self.name = name  # 参数名
        self.value = value  # 参数值

    def __repr__(self):  # 返回参数表达式的字符串表示
        return f"Argument(name={self.name}, value={repr(self.value)})"  # 格式化输出

    def __len__(self):  # 返回参数值的长度
        return len(self.value)  # 返回值的长度

    def __getitem__(self, i):  # 通过索引获取参数值中的元素
        return self.value[i]  # 返回值中指定索引的元素

    def __int__(self):  # 将参数值转换为整数
        return self.value  # 返回值本身（在运行时会被替换为实际值）

    def __bool__(self):  # 将参数值转换为布尔值
        return self.value  # 返回值本身（在运行时会被替换为实际值）

    def __format__(self, *args):  # 禁止在f-string中使用参数
        raise TypeError(  # 抛出类型错误
            "Cannot put argument inside a f-string. "  # 不能将参数放入f-string中
            "This is not compatible with the tracer. "  # 这与跟踪器不兼容
        )


class SglImage(SglExpr):  # 图像表达式，表示输入的图像
    def __init__(self, path: str):  # 初始化图像表达式
        self.path = path  # 图像路径

    def __repr__(self) -> str:  # 返回图像表达式的字符串表示
        return f"SglImage({self.path})"  # 格式化输出


class SglVideo(SglExpr):  # 视频表达式，表示输入的视频
    def __init__(self, path: str, num_frames: int):  # 初始化视频表达式
        self.path = path  # 视频路径
        self.num_frames = num_frames  # 采样的帧数

    def __repr__(self) -> str:  # 返回视频表达式的字符串表示
        return f"SglVideo({self.path}, {self.num_frames})"  # 格式化输出


class SglGen(SglExpr):  # 生成表达式，表示调用模型生成文本
    def __init__(  # 初始化生成表达式
        self,
        name: Optional[str] = None,  # 生成的变量名
        max_new_tokens: Optional[int] = None,  # 最大生成token数
        min_new_tokens: Optional[int] = None,  # 最小生成token数
        n: Optional[int] = None,  # 候选数量
        stop: Optional[Union[str, List[str]]] = None,  # 停止字符串
        stop_token_ids: Optional[List[int]] = None,  # 停止token ID列表
        stop_regex: Optional[Union[str, List[str]]] = None,  # 停止正则表达式
        temperature: Optional[float] = None,  # 采样温度
        top_p: Optional[float] = None,  # top-p参数
        top_k: Optional[int] = None,  # top-k参数
        min_p: Optional[float] = None,  # min-p参数
        frequency_penalty: Optional[float] = None,  # 频率惩罚
        presence_penalty: Optional[float] = None,  # 存在惩罚
        ignore_eos: Optional[bool] = None,  # 是否忽略结束符
        return_logprob: Optional[bool] = None,  # 是否返回对数概率
        logprob_start_len: Optional[int] = None,  # 对数概率起始长度
        top_logprobs_num: Optional[int] = None,  # top对数概率数量
        return_text_in_logprobs: Optional[bool] = None,  # 对数概率中是否返回文本
        dtype: Optional[type] = None,  # 数据类型约束
        regex: Optional[str] = None,  # 正则表达式约束
        json_schema: Optional[str] = None,  # JSON schema约束
    ):  # 初始化生成表达式的参数列表
        """Call the model to generate. See the meaning of the arguments in docs/backend/sampling_params.md"""  # 调用模型生成。参数含义见docs/backend/sampling_params.md
        super().__init__()  # 调用父类初始化
        self.name = name  # 保存生成变量名
        self.sampling_params = SglSamplingParams(  # 创建采样参数对象
            max_new_tokens=max_new_tokens,  # 最大生成token数
            min_new_tokens=min_new_tokens,  # 最小生成token数
            n=n,  # 候选数量
            stop=stop,  # 停止字符串
            stop_regex=stop_regex,  # 停止正则表达式
            stop_token_ids=stop_token_ids,  # 停止token ID列表
            temperature=temperature,  # 采样温度
            top_p=top_p,  # top-p参数
            top_k=top_k,  # top-k参数
            min_p=min_p,  # min-p参数
            frequency_penalty=frequency_penalty,  # 频率惩罚
            presence_penalty=presence_penalty,  # 存在惩罚
            ignore_eos=ignore_eos,  # 是否忽略结束符
            return_logprob=return_logprob,  # 是否返回对数概率
            logprob_start_len=logprob_start_len,  # 对数概率起始长度
            top_logprobs_num=top_logprobs_num,  # top对数概率数量
            return_text_in_logprobs=return_text_in_logprobs,  # 对数概率中是否返回文本
            dtype=dtype,  # 数据类型约束
            regex=regex,  # 正则表达式约束
            json_schema=json_schema,  # JSON schema约束
        )

    def __repr__(self):  # 返回生成表达式的字符串表示
        return f"Gen('{self.name}')"  # 格式化输出


class SglConstantText(SglExpr):  # 常量文本表达式，表示固定的文本字符串
    def __init__(self, value: str):  # 初始化常量文本表达式
        super().__init__()  # 调用父类初始化
        self.value = value  # 保存文本值

    def __repr__(self):  # 返回常量文本表达式的字符串表示
        return f"Constant({repr(self.value)})"  # 格式化输出


class SglRoleBegin(SglExpr):  # 角色开始表达式，标记对话角色的开始
    def __init__(self, role: str):  # 初始化角色开始表达式
        super().__init__()  # 调用父类初始化
        self.role = role  # 保存角色名称

    def __repr__(self):  # 返回角色开始表达式的字符串表示
        return f"RoleBegin({self.role})"  # 格式化输出


class SglRoleEnd(SglExpr):  # 角色结束表达式，标记对话角色的结束
    def __init__(self, role: str):  # 初始化角色结束表达式
        super().__init__()  # 调用父类初始化
        self.role = role  # 保存角色名称

    def __repr__(self):  # 返回角色结束表达式的字符串表示
        return f"RoleEnd({self.role})"  # 格式化输出


class SglSelect(SglExpr):  # 选择表达式，从给定选项中选择一个

    def __init__(  # 初始化选择表达式
        self,
        name: str,  # 变量名
        choices: List[str],  # 选项列表
        temperature: float,  # 采样温度
        choices_method: ChoicesSamplingMethod,  # 选择采样方法
    ):  # 初始化选择表达式的参数列表
        super().__init__()  # 调用父类初始化
        self.name = name  # 保存变量名
        self.choices = choices  # 保存选项列表
        self.temperature = temperature  # 保存采样温度
        self.choices_method = choices_method  # 保存选择采样方法

    def __repr__(self):  # 返回选择表达式的字符串表示
        return f"Select({self.name}, choices={self.choices}, choices_method={self.choices_method})"  # 格式化输出


class SglFork(SglExpr):  # 分支表达式，创建多个并行执行分支
    def __init__(self, number: int, position_ids_offset=None):  # 初始化分支表达式
        super().__init__()  # 调用父类初始化
        self.number = number  # 分支数量
        self.position_ids_offset = position_ids_offset  # 位置ID偏移量

    def __repr__(self):  # 返回分支表达式的字符串表示
        return (  # 格式化输出
            f"Fork(%{self.prev_node.node_id}, number={self.number}, "  # 显示前驱节点ID和分支数量
            f"position_ids_offset={self.position_ids_offset})"  # 显示位置ID偏移量
        )


class SglGetForkItem(SglExpr):  # 获取分支项表达式，从分支结果中获取指定索引的结果
    def __init__(self, index: int):  # 初始化获取分支项表达式
        super().__init__()  # 调用父类初始化
        self.index = index  # 要获取的分支索引

    def __repr__(self):  # 返回获取分支项表达式的字符串表示
        return f"GetForkItem(%{self.prev_node.node_id}, index={self.index})"  # 格式化输出


class SglVariable(SglExpr):  # 变量表达式，表示引用之前生成的结果
    def __init__(self, name: str, source):  # 初始化变量表达式
        super().__init__()  # 调用父类初始化
        self.name = name  # 变量名
        self.source = source  # 变量的来源表达式

    def __repr__(self):  # 返回变量表达式的字符串表示
        return f"Variable('{self.name}', source=%{self.source.node_id})"  # 格式化输出，显示变量名和来源节点ID


class SglVarScopeBegin(SglExpr):  # 变量作用域开始表达式，标记变量作用域的起点
    def __init__(self, name: str):  # 初始化变量作用域开始表达式
        super().__init__()  # 调用父类初始化
        self.name = name  # 作用域名

    def __repr__(self):  # 返回变量作用域开始表达式的字符串表示
        return f"VarScopeBegin('{self.name}')"  # 格式化输出


class SglVarScopeEnd(SglExpr):  # 变量作用域结束表达式，标记变量作用域的终点
    def __init__(self, name: str):  # 初始化变量作用域结束表达式
        super().__init__()  # 调用父类初始化
        self.name = name  # 作用域名

    def __repr__(self):  # 返回变量作用域结束表达式的字符串表示
        return f"VarScopeEnd('{self.name}')"  # 格式化输出


class SglConcateAndAppend(SglExpr):  # 拼接并追加表达式，将多个状态拼接并追加到KV缓存
    def __init__(self, states):  # 初始化拼接并追加表达式
        super().__init__()  # 调用父类初始化
        self.states = states  # 要拼接的状态列表

    def __repr__(self):  # 返回拼接并追加表达式的字符串表示
        return f"ConcatenateAndAppend('{self.states}')"  # 格式化输出


class SglCommitLazy(SglExpr):  # 提交延迟计算表达式，将之前延迟的计算提交执行
    def __init__(self):  # 初始化提交延迟计算表达式
        super().__init__()  # 调用父类初始化

    def __repr__(self):  # 返回提交延迟计算表达式的字符串表示
        return "CommitLazy()"  # 格式化输出


class SglSeparateReasoning(SglExpr):  # 分离推理表达式，将推理内容从生成结果中分离出来
    def __init__(self, model_type: str, expr: SglExpr):  # 初始化分离推理表达式
        super().__init__()  # 调用父类初始化
        self.model_type = model_type  # 模型类型

        self.expr = expr  # 要分离推理内容的表达式
        self.name = None  # 推理内容变量名，后续通过_process_expr设置
        self._process_expr(expr)  # 处理表达式，提取推理内容变量名

    def process_name_for_reasoning(self, name):  # 为推理内容生成变量名，在原变量名后添加后缀
        if not name:  # 如果变量名为空
            raise ValueError("name must be provided")  # 抛出值错误
        return f"{name}_reasoning_content"  # 返回添加了_reasoning_content后缀的变量名

    def _process_expr(self, expr):  # 内部方法，递归处理表达式以提取推理内容变量名
        if isinstance(expr, SglGen):  # 如果是生成表达式
            self.name = self.process_name_for_reasoning(expr.name)  # 为其生成推理内容变量名
        elif isinstance(expr, SglSelect):  # 如果是选择表达式
            self.name = self.process_name_for_reasoning(expr.name)  # 为其生成推理内容变量名
        elif isinstance(expr, SglExprList):  # 如果是表达式列表
            for x in expr.expr_list:  # 遍历列表中的每个表达式
                self._process_expr(x)  # 递归处理每个表达式

    def __repr__(self):  # 返回分离推理表达式的字符串表示
        return f"SeparateReasoning(model_type={self.model_type}, name={self.name})"  # 格式化输出
