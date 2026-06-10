"""Tracing a program."""
# 本文件实现了SGLang程序的追踪功能，包括TracerProgramState（追踪SGL程序执行以构建IR图）、
# TracingScope（管理追踪上下文）以及提取前缀和追踪程序的工具函数。

import uuid  # 导入uuid模块，用于生成唯一标识符
from typing import Any, Dict, List, Optional  # 导入类型注解工具

from sglang.lang.backend.base_backend import BaseBackend  # 导入基础后端类
from sglang.lang.interpreter import ProgramState, ProgramStateGroup  # 导入程序状态类和状态组类
from sglang.lang.ir import (  # 导入IR（中间表示）节点类型
    SglArgument,  # 参数节点
    SglConstantText,  # 常量文本节点
    SglExpr,  # 表达式基类
    SglExprList,  # 表达式列表节点
    SglFork,  # 分支节点
    SglGen,  # 生成节点
    SglGetForkItem,  # 获取分支项节点
    SglRoleBegin,  # 角色开始节点
    SglRoleEnd,  # 角色结束节点
    SglSelect,  # 选择节点
    SglVariable,  # 变量节点
    SglVarScopeBegin,  # 变量作用域开始节点
    SglVarScopeEnd,  # 变量作用域结束节点
)


class StopTracing(Exception):
    """停止追踪异常，用于在仅追踪前缀时提前终止追踪过程。"""
    pass  # 空实现，仅作为信号异常使用


def extract_prefix_by_tracing(program, backend):
    """通过追踪程序执行来提取常量文本前缀。"""
    # Create dummy arguments
    dummy_arguments = {name: SglArgument(name, None) for name in program.arg_names}  # 为程序的每个参数名创建虚拟参数节点
    arguments = dummy_arguments  # 将虚拟参数赋值给arguments
    arguments.update(program.bind_arguments)  # 用绑定参数更新arguments字典

    # Trace
    tracer = TracerProgramState(backend, arguments, only_trace_prefix=True)  # 创建追踪器，设置仅追踪前缀模式
    try:  # 尝试执行追踪
        with TracingScope(tracer):  # 进入追踪上下文
            tracer.ret_value = program.func(tracer, **arguments)  # 执行程序函数，将追踪器作为状态传入
    except (StopTracing, TypeError, AttributeError):  # 捕获停止追踪异常及可能的类型/属性错误
        # Some exceptions may not be caught
        pass  # 忽略异常，继续执行

    # Run and cache prefix
    prefix = ""  # 初始化前缀字符串为空
    for expr in tracer.flatten_nodes():  # 遍历追踪器中扁平化的所有节点
        if isinstance(expr, SglConstantText):  # 如果节点是常量文本类型
            prefix += expr.value  # 将常量文本值追加到前缀中
        else:  # 如果遇到非常量文本节点
            break  # 停止提取前缀
    return prefix  # 返回提取到的前缀字符串


def trace_program(program, arguments, backend):
    """追踪程序执行，构建完整的IR图并返回追踪器状态。"""
    # Create dummy backend
    if backend is None:  # 如果未提供后端
        backend = BaseBackend()  # 创建一个基础后端实例

    # Create dummy arguments
    dummy_arguments = {  # 为未在arguments中提供的参数名创建虚拟参数
        name: SglArgument(name, None)  # 创建参数名为name、值为None的虚拟参数
        for name in program.arg_names  # 遍历程序的所有参数名
        if name not in arguments  # 仅对arguments中不存在的参数创建虚拟参数
    }
    arguments.update(dummy_arguments)  # 用虚拟参数更新arguments字典
    arguments.update(program.bind_arguments)  # 用绑定参数更新arguments字典

    # Trace
    tracer = TracerProgramState(backend, arguments, only_trace_prefix=False)  # 创建追踪器，设置完整追踪模式
    with TracingScope(tracer):  # 进入追踪上下文
        tracer.ret_value = program.func(tracer, **arguments)  # 执行程序函数，将追踪器作为状态传入
    return tracer  # 返回追踪器状态对象


class TracerProgramState(ProgramState):
    """追踪器程序状态类，追踪SGL程序执行以构建IR图。"""

    def __init__(self, backend, arguments, only_trace_prefix):
        """初始化追踪器程序状态。"""
        self.pid = uuid.uuid4().hex  # 生成唯一进程ID
        self.backend = backend  # 保存后端引用
        self.arguments: Dict[str, Any] = arguments  # 保存参数字典
        self.only_trace_prefix = only_trace_prefix  # 设置是否仅追踪前缀模式

        if hasattr(backend, "endpoint"):  # 如果后端有endpoint属性
            self.backend = backend.endpoint  # 使用后端的endpoint替代后端本身

        self.nodes = []  # 初始化IR节点列表
        self.last_node = None  # 初始化最后一个节点为None
        self.variables = {}  # 初始化变量字典
        self.ret_value = None  # 初始化返回值为None

        # For completion

        # For chat
        self.messages_ = []  # 初始化聊天消息列表
        self.cur_role = None  # 初始化当前角色为None
        self.chat_template = self.backend.get_chat_template()  # 从后端获取聊天模板

        # For multi states
        self.child_states = []  # 初始化子状态列表

        cur_scope = TracingScope.get_current_scope()  # 获取当前追踪作用域
        if cur_scope is not None:  # 如果存在当前作用域
            cur_scope.add_child_state(self)  # 将当前状态添加为作用域的子状态

    ##################################
    ########### Public API ###########
    ##################################

    def fork(self, size: int = 1, position_ids_offset: Optional[List[int]] = None):
        """创建指定数量的分支状态，用于并行生成多个结果。"""
        assert size >= 1  # 断言分支数量至少为1

        if self.only_trace_prefix:  # 如果仅追踪前缀模式
            raise StopTracing()  # 抛出停止追踪异常

        fork_node = SglFork(size)  # 创建分支节点
        fork_node.prev_node = self.last_node  # 设置分支节点的前驱为最后一个节点

        states = [  # 创建size个追踪器状态
            TracerProgramState(self.backend, self.arguments, self.only_trace_prefix)  # 每个状态使用相同后端和参数
            for _ in range(size)  # 循环size次
        ]

        for i in range(size):  # 遍历每个分支索引
            node = SglGetForkItem(i)  # 创建获取分支项节点
            node.prev_node = fork_node  # 设置其前驱为分支节点
            states[i].last_node = node  # 设置第i个状态的最后一个节点
            states[i].variables = dict(self.variables)  # 复制当前变量字典到第i个状态
            states[i].messages_ = list(self.messages_)  # 复制当前消息列表到第i个状态
            states[i].cur_role = self.cur_role  # 复制当前角色到第i个状态
            states[i].chat_template = self.chat_template  # 复制聊天模板到第i个状态

        state_group = ProgramStateGroup(states, self)  # 创建程序状态组，包含所有分支状态

        return state_group  # 返回状态组

    ##################################
    ########## Internal API ##########
    ##################################

    def _append_node(self, other: SglExpr):
        """将IR节点追加到节点列表中。"""
        self.nodes.append(other)  # 将节点添加到节点列表末尾
        other.prev_node = self.last_node  # 设置新节点的前驱为最后一个节点
        self.last_node = other  # 更新最后一个节点为新节点

    def _execute(self, other: SglExpr):
        """执行一个IR表达式，根据类型分派到对应的处理方法。"""
        if isinstance(other, str):  # 如果表达式是字符串
            other = SglConstantText(other)  # 将字符串转换为常量文本节点

        other.pid = self.pid  # 设置表达式的进程ID

        if isinstance(other, SglConstantText):  # 如果是常量文本节点
            self._execute_fill(other)  # 执行填充操作
        elif isinstance(other, SglGen):  # 如果是生成节点
            self._execute_gen(other)  # 执行生成操作
        elif isinstance(other, SglSelect):  # 如果是选择节点
            self._execute_select(other)  # 执行选择操作
        elif isinstance(other, SglExprList):  # 如果是表达式列表节点
            for x in other.expr_list:  # 遍历表达式列表中的每个表达式
                self._execute(x)  # 递归执行每个表达式
        elif isinstance(other, SglRoleBegin):  # 如果是角色开始节点
            self._execute_role_begin(other)  # 执行角色开始操作
        elif isinstance(other, SglRoleEnd):  # 如果是角色结束节点
            self._execute_role_end(other)  # 执行角色结束操作
        elif isinstance(other, SglVarScopeBegin):  # 如果是变量作用域开始节点
            self._execute_var_scope_begin(other)  # 执行变量作用域开始操作
        elif isinstance(other, SglVarScopeEnd):  # 如果是变量作用域结束节点
            self._execute_var_scope_end(other)  # 执行变量作用域结束操作
        else:  # 其他未知类型的表达式
            if self.only_trace_prefix:  # 如果仅追踪前缀模式
                raise StopTracing()  # 抛出停止追踪异常
            else:  # 否则完整追踪模式
                self._append_node(other)  # 直接将节点追加到节点列表

        return self  # 返回自身以支持链式调用

    def __iadd__(self, other):
        """重载+=运算符，执行表达式并返回自身。"""
        self._execute(other)  # 执行表达式
        return self  # 返回自身

    def _execute_fill(self, expr: SglConstantText):
        """执行常量文本填充操作，将文本节点追加到IR图。"""
        if isinstance(expr, str):  # 如果表达式是字符串
            expr = SglConstantText(expr)  # 将字符串转换为常量文本节点
        self._append_node(expr)  # 将常量文本节点追加到节点列表

    def _execute_gen(self, expr: SglGen):
        """执行生成操作，创建变量节点并记录到变量字典。"""
        name = expr.name if expr.name is not None else "gen_" + str(len(self.variables))  # 使用表达式名称或自动生成名称
        new_node = SglVariable(name, source=expr)  # 创建变量节点，源为生成表达式
        self.variables[name] = new_node  # 将变量节点记录到变量字典
        self._append_node(expr)  # 将生成表达式追加到节点列表

    def _execute_select(self, expr: SglSelect):
        """执行选择操作，创建变量节点并记录到变量字典。"""
        name = (  # 确定变量名称
            expr.name if expr.name is not None else "select_" + str(len(self.variables))  # 使用表达式名称或自动生成名称
        )
        new_node = SglVariable(name, source=expr)  # 创建变量节点，源为选择表达式
        self.variables[name] = new_node  # 将变量节点记录到变量字典
        self._append_node(expr)  # 将选择表达式追加到节点列表

    def _execute_role_begin(self, expr: SglRoleBegin):
        """执行角色开始操作，处理聊天模板中的角色前缀和默认系统消息。"""
        assert self.cur_role is None, "Nested roles are not allowed."  # 断言不允许嵌套角色

        if len(self.messages_) == 0 and expr.role != "system":  # 如果没有消息且角色不是system
            # Insert default system message
            default_system = self.chat_template.default_system_prompt  # 获取默认系统提示
            if default_system:  # 如果存在默认系统提示
                self._execute_role_begin(SglRoleBegin("system"))  # 插入system角色开始节点
                self._execute_fill(default_system)  # 填充默认系统提示文本
                self._execute_role_end(SglRoleEnd("system"))  # 插入system角色结束节点

        self.cur_role = expr.role  # 设置当前角色

        prefix, suffix = self.chat_template.get_prefix_and_suffix(  # 获取角色的前缀和后缀
            expr.role, self.messages_  # 传入角色和当前消息列表
        )

        self._execute_fill(prefix)  # 执行填充角色前缀文本

    def _execute_role_end(self, expr: SglRoleEnd):
        """执行角色结束操作，填充角色后缀并记录消息。"""
        prefix, suffix = self.chat_template.get_prefix_and_suffix(  # 获取角色的前缀和后缀
            expr.role, self.messages_  # 传入角色和当前消息列表
        )

        self._execute_fill(suffix)  # 执行填充角色后缀文本

        self.messages_.append({"role": expr.role, "content": ""})  # 将角色消息添加到消息列表

        self.cur_role = None  # 重置当前角色为None

    def _execute_var_scope_end(self, expr: SglVarScopeEnd):
        """执行变量作用域结束操作，创建变量节点绑定到最后一个节点。"""
        new_node = SglVariable(expr.name, source=self.last_node)  # 创建变量节点，源为最后一个节点
        self.variables[expr.name] = new_node  # 将变量节点记录到变量字典

    def get_var(self, name):
        """根据名称获取变量，优先从参数中查找，再从变量字典中查找。"""
        ret = self.arguments.get(name, None)  # 先从参数字典中查找变量
        if ret is not None:  # 如果在参数中找到
            return ret  # 返回参数中的值

        v = self.variables[name]  # 从变量字典中获取变量
        return SglVariable(v.name, v.source)  # 返回新的变量节点副本

    def flatten_nodes(self):
        """将嵌套的表达式列表扁平化为线性的节点列表。"""
        def traverse(cur):  # 定义递归遍历函数
            if isinstance(cur, SglExprList):  # 如果当前节点是表达式列表
                for child in cur.expr_list:  # 遍历子表达式
                    traverse(child)  # 递归遍历子表达式
            else:  # 如果是普通表达式节点
                ret.append(cur)  # 将节点添加到结果列表

        ret = []  # 初始化结果列表
        for x in self.nodes:  # 遍历所有顶级节点
            traverse(x)  # 递归遍历每个节点
        return ret  # 返回扁平化后的节点列表

    def __del__(self):
        """析构函数，空实现。"""
        pass  # 空实现


class TracingScope:
    """追踪作用域类，管理追踪上下文的进入和退出，维护作用域链。"""

    cur_scope = None  # 类变量，当前活跃的追踪作用域

    def __init__(self, tracer_state: TracerProgramState):
        """初始化追踪作用域，保存追踪器状态和上一个作用域。"""
        self.tracer_state = tracer_state  # 保存追踪器状态
        self.last_scope = TracingScope.cur_scope  # 保存上一个作用域引用

    def __enter__(self):
        """进入追踪作用域，将当前作用域设置为活跃作用域。"""
        TracingScope.cur_scope = self  # 设置当前作用域为自身
        return self  # 返回自身

    def __exit__(self, exc_type, exc_value, traceback):
        """退出追踪作用域，恢复上一个作用域为活跃作用域。"""
        TracingScope.cur_scope = self.last_scope  # 恢复上一个作用域为当前作用域

    @staticmethod
    def get_current_scope():
        """获取当前活跃的追踪作用域。"""
        return TracingScope.cur_scope  # 返回当前活跃的作用域

    def add_child_state(self, state: TracerProgramState):
        """将子状态添加到当前作用域及所有祖先作用域的追踪器中。"""
        cur_scope = self  # 从当前作用域开始
        while cur_scope is not None:  # 沿作用域链向上遍历
            cur_scope.tracer_state.child_states.append(state)  # 将状态添加到当前作用域追踪器的子状态列表
            cur_scope = cur_scope.last_scope  # 移动到上一个作用域
