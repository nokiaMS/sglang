# 修复@app.middleware("http")的问题模块
# BaseHTTPMiddleware的call_next替换了ASGI的receive，导致request.is_disconnected()失效
# 并阻止了非流式请求在客户端断开时中止
# patch_app_http_middleware(app)将@app.middleware("http")替换为
# 一个call_next原样传递receive的版本
"""
Fix @app.middleware("http") whose BaseHTTPMiddleware call_next replaces
ASGI ``receive``, breaking request.is_disconnected() and preventing
non-streaming request abort on client disconnect.

patch_app_http_middleware(app) replaces @app.middleware("http") with a
version whose call_next passes ``receive`` through untouched.
"""

from __future__ import annotations  # 启用延迟注解评估

from starlette.requests import Request  # 导入Starlette请求类


class _SentResponse:  # 已发送响应的代理类，在真实响应已发送后返回
    """Response proxy returned after the real response was already sent."""

    def __init__(self, status_code: int):  # 初始化，接收状态码
        self.status_code = status_code  # 保存状态码


class _PureASGIDispatch:  # 纯ASGI中间件，提供修正的call_next，原样传递receive
    """Pure ASGI middleware providing a fixed call_next that passes
    ``receive`` through untouched (unlike BaseHTTPMiddleware)."""

    def __init__(self, app, dispatch):  # 初始化，接收ASGI应用和调度函数
        self.app = app  # 保存ASGI应用
        self.dispatch = dispatch  # 保存调度函数

    async def __call__(self, scope, receive, send):  # ASGI调用接口
        if scope["type"] != "http":  # 如果不是HTTP请求类型
            await self.app(scope, receive, send)  # 直接传递给应用
            return  # 返回

        request = Request(scope, receive)  # 创建Starlette请求对象
        status_code = 500  # 默认状态码为500

        async def call_next(_request):  # call_next函数，传递receive而非替换它
            nonlocal status_code  # 声明使用外部作用域的status_code

            async def send_and_capture(message):  # 捕获响应状态的send包装器
                nonlocal status_code  # 声明使用外部作用域的status_code
                if message["type"] == "http.response.start":  # 如果是响应开始消息
                    status_code = message["status"]  # 捕获状态码
                await send(message)  # 将消息发送给客户端

            await self.app(scope, receive, send_and_capture)  # 调用应用，传递原始receive
            return _SentResponse(status_code)  # 返回已发送响应代理

        await self.dispatch(request, call_next)  # 执行调度函数


def patch_app_http_middleware(app):  # 将@app.middleware("http")替换为修正call_next的版本
    """Replace @app.middleware("http") with a fixed-call_next version."""
    _orig = app.middleware  # 保存原始的middleware方法

    def _fixed(middleware_type):  # 修正版的middleware方法
        if middleware_type == "http":  # 如果是HTTP类型中间件

            def decorator(fn):  # 装饰器函数
                app.add_middleware(_PureASGIDispatch, dispatch=fn)  # 使用纯ASGI调度器替代BaseHTTPMiddleware
                return fn  # 返回原始函数

            return decorator  # 返回装饰器
        return _orig(middleware_type)  # 对于非HTTP类型，调用原始方法

    app.middleware = _fixed  # 替换app的middleware方法
