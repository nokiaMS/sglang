# 文件说明：本文件定义了SGLang后端的抽象基类BaseBackend，提供了所有后端实现的公共接口，
# 包括生成文本、流式生成、选择、缓存前缀等方法，具体后端需继承并实现这些方法。

from typing import List, Optional, Union  # 导入类型注解

from sglang.lang.chat_template import get_chat_template  # 导入获取聊天模板函数
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod  # 导入选择决策和采样方法
from sglang.lang.interpreter import StreamExecutor  # 导入流执行器
from sglang.lang.ir import SglSamplingParams  # 导入采样参数类


class BaseBackend:
    """SGLang后端的抽象基类，定义了所有后端需要实现的接口方法。"""
    def __init__(self) -> None:  # 初始化方法
        self.support_concate_and_append = False  # 是否支持拼接和追加操作，默认不支持
        self.chat_template = get_chat_template("default")  # 获取默认聊天模板

    def get_model_name(self):  # 获取模型名称
        """获取后端使用的模型名称，子类必须实现。"""
        raise NotImplementedError()  # 抛出未实现异常

    def get_chat_template(self):  # 获取聊天模板
        """获取当前后端使用的聊天模板。"""
        return self.chat_template  # 返回聊天模板

    def cache_prefix(self, prefix_str: str):  # 缓存前缀字符串
        """缓存指定的前缀字符串，用于加速后续生成。"""
        pass  # 默认空实现

    def uncache_prefix(self, rid: str):  # 取消缓存前缀
        """取消指定请求ID的前缀缓存。"""
        pass  # 默认空实现

    def end_request(self, rid: Union[str, List[str]]):  # 结束请求
        """结束指定的请求。"""
        pass  # 默认空实现

    def begin_program(self, s: StreamExecutor):  # 开始程序执行
        """在程序开始执行时调用。"""
        pass  # 默认空实现

    def end_program(self, s: Union[StreamExecutor, List[StreamExecutor]]):  # 结束程序执行
        """在程序执行结束时调用。"""
        pass  # 默认空实现

    def commit_lazy_operations(self, s: StreamExecutor):  # 提交延迟操作
        """提交延迟执行的操作。"""
        pass  # 默认空实现

    def fork_program(  # 分叉程序
        self,
        src: StreamExecutor,  # 源流执行器
        dst: List[StreamExecutor],  # 目标流执行器列表
        position_ids_offset: Optional[List[int]] = None,  # 位置ID偏移量，可选
    ):
        """将程序从源执行器分叉到目标执行器。"""
        pass  # 默认空实现

    def fill_image(self, s: StreamExecutor):  # 填充图像
        """将图像数据填充到流执行器中。"""
        pass  # 默认空实现

    def generate(  # 生成文本
        self,
        s: StreamExecutor,  # 流执行器
        sampling_params: SglSamplingParams,  # 采样参数
    ):
        """根据给定参数生成文本，子类必须实现。"""
        raise NotImplementedError()  # 抛出未实现异常

    def generate_stream(  # 流式生成文本
        self,
        s: StreamExecutor,  # 流执行器
        sampling_params: SglSamplingParams,  # 采样参数
    ):
        """流式生成文本，逐步返回生成结果，子类必须实现。"""
        raise NotImplementedError()  # 抛出未实现异常

    def select(  # 选择方法
        self,
        s: StreamExecutor,  # 流执行器
        choices: List[str],  # 候选选项列表
        temperature: float,  # 温度参数
        choices_method: Optional[ChoicesSamplingMethod] = None,  # 选择方法，可选
    ) -> ChoicesDecision:
        """在给定候选选项中进行选择，子类必须实现。"""
        raise NotImplementedError()  # 抛出未实现异常

    def concatenate_and_append(self, src_rids: List[str], dst_rid: str):  # 拼接并追加
        """将多个源请求的KV缓存拼接并追加到目标请求，子类必须实现。"""
        raise NotImplementedError()  # 抛出未实现异常

    def shutdown(self):  # 关闭后端
        """关闭后端，释放资源。"""
        pass  # 默认空实现

    def flush_cache(self):  # 刷新缓存
        """刷新后端缓存。"""
        pass  # 默认空实现

    def get_server_info(self):  # 获取服务器信息
        """获取后端服务器的信息。"""
        pass  # 默认空实现
