# HTTP服务器认证工具模块
# 提供三级端点认证（普通/管理可选/管理强制）、请求认证决策和API密钥中间件
# 故意保持轻量级（不导入torch），以便在单元测试中使用

"""Auth utilities for HTTP servers.

This module is intentionally lightweight (no torch import) so it can be used in unit tests.
"""

from __future__ import annotations  # 延迟注解评估

import secrets  # 安全随机数和常量时间比较
from dataclasses import dataclass  # 数据类装饰器
from enum import Enum  # 枚举基类
from typing import Any, Optional  # 类型注解


@dataclass(frozen=True)  # 不可变数据类
class AuthDecision:  # 认证决策结果，表示请求是否被允许
    allowed: bool  # 是否允许
    error_status_code: int = 401  # Only meaningful when allowed=False  # 仅在allowed=False时有意义，默认401未授权


class AuthLevel(str, Enum):  # 端点认证级别枚举
    """Per-endpoint auth level (attached to endpoint function via `@auth_level`)."""
    # 每个端点的认证级别（通过@auth_level装饰器附加到端点函数）。

    NORMAL = "normal"  # 普通级别：使用api_key保护所有端点
    ADMIN_OPTIONAL = "admin_optional"  # 管理可选级别：无密钥配置时允许访问
    ADMIN_FORCE = "admin_force"  # 管理强制级别：必须使用admin_api_key


def auth_level(level: AuthLevel):  # 端点认证级别装饰器，将认证级别附加到端点函数
    """Mark endpoint with auth level (stored in endpoint metadata)."""
    # 用认证级别标记端点（存储在端点元数据中）。

    def decorator(func):  # 内部装饰器函数
        func._auth_level = level  # 在函数对象上设置_auth_level属性
        return func  # 返回原函数

    return decorator  # 返回装饰器


def _get_auth_level_from_app_and_scope(app: Any, scope: dict) -> AuthLevel:  # 根据请求路由解析端点的认证级别
    """Best-effort resolve auth level by matching the request to a route."""
    # 尽力通过将请求匹配到路由来解析认证级别。
    # Import lazily to keep this module unit-test friendly (FastAPI/Starlette are not
    # required unless you actually use the middleware / route matching).
    # 延迟导入以保持此模块的单元测试友好性（除非实际使用中间件/路由匹配，否则不需要FastAPI/Starlette）。
    from starlette.routing import Match  # 导入Starlette路由匹配枚举

    # Prefer app.router.routes when available; fall back to app.routes.
    # 优先使用app.router.routes（如果可用）；否则回退到app.routes。
    routes = getattr(getattr(app, "router", None), "routes", None) or getattr(  # 尝试获取路由列表
        app, "routes", []
    )

    for route in routes:  # 遍历所有路由
        try:  # 捕获路由匹配异常
            match, child_scope = route.matches(scope)  # 尝试匹配请求
        except Exception:  # 匹配失败
            continue  # 跳过此路由
        if match == Match.FULL:  # 如果完全匹配
            endpoint = child_scope.get("endpoint") or getattr(route, "endpoint", None)  # 获取端点函数
            level = getattr(endpoint, "_auth_level", None)  # 获取端点的认证级别
            return level if isinstance(level, AuthLevel) else AuthLevel.NORMAL  # 返回级别或默认普通级别

    return AuthLevel.NORMAL  # 未匹配时返回普通级别


def app_has_admin_force_endpoints(app: Any) -> bool:  # 检查应用是否有管理强制级别的端点
    """Return True if any route endpoint is marked as ADMIN_FORCE."""
    # 如果任何路由端点被标记为ADMIN_FORCE则返回True。
    routes = getattr(getattr(app, "router", None), "routes", None) or getattr(  # 获取路由列表
        app, "routes", []
    )
    for route in routes:  # 遍历所有路由
        endpoint = getattr(route, "endpoint", None)  # 获取端点函数
        if getattr(endpoint, "_auth_level", None) == AuthLevel.ADMIN_FORCE:  # 如果是管理强制级别
            return True  # 返回True
    return False  # 没有管理强制端点


def decide_request_auth(  # 纯函数认证决策（易于单元测试）
    *,
    method: str,  # HTTP方法
    path: str,  # 请求路径
    authorization_header: Optional[str],  # Authorization请求头
    api_key: Optional[str],  # API密钥
    admin_api_key: Optional[str],  # 管理API密钥
    auth_level: AuthLevel,  # 端点认证级别
) -> AuthDecision:
    """Pure auth decision function (easy to unit test).

    Auth levels:
    - NORMAL: legacy behavior (api_key protects all endpoints when configured)
    - ADMIN_OPTIONAL: can be accessed without any key (if no keys configured),
      or with api_key/admin_api_key depending on server config.
    - ADMIN_FORCE: requires admin_api_key; if admin_api_key is NOT configured,
      it must be rejected (403) even if api_key is provided.

    NOTE :
    - Health/metrics endpoints are always allowed (even when api_key/admin_api_key is set),
      to support k8s/liveness/readiness and Prometheus scraping without embedding secrets.
    - We match them by prefix to cover common variants like /health_generate.
    """
    # 纯函数认证决策（易于单元测试）。
    # 认证级别：
    # - NORMAL：传统行为（配置api_key时保护所有端点）
    # - ADMIN_OPTIONAL：无密钥配置时允许访问，或根据服务器配置使用api_key/admin_api_key
    # - ADMIN_FORCE：需要admin_api_key；如果未配置admin_api_key，
    #   即使提供了api_key也必须拒绝（403）
    # 注意：
    # - Health/metrics端点始终允许访问（即使设置了api_key/admin_api_key），
    #   以支持k8s存活性/就绪性和Prometheus抓取而无需嵌入密钥
    # - 通过前缀匹配以覆盖/health_generate等常见变体
    if method == "OPTIONS":  # OPTIONS预检请求始终允许
        return AuthDecision(allowed=True)  # 返回允许决策

    if path.startswith("/health") or path.startswith("/metrics"):  # 健康检查和指标端点始终允许
        return AuthDecision(allowed=True)  # 返回允许决策

    def _check_bearer_token(  # 使用常量时间比较检查Bearer token
        authorization_header: Optional[str], expected_token: str  # Authorization头和期望的token
    ) -> bool:
        """Check bearer token with constant-time comparison."""
        # 使用常量时间比较检查Bearer token。
        if not authorization_header:  # 如果没有Authorization头
            return False  # 认证失败
        parts = authorization_header.split(" ", 1)  # 按空格分割为两部分
        if len(parts) != 2 or parts[0].lower() != "bearer":  # 格式不正确或不是Bearer类型
            return False  # 认证失败
        return secrets.compare_digest(parts[1], expected_token)  # 常量时间比较token

    # Force-auth endpoints: only admin_api_key can unlock them; if admin_api_key is unset,
    # reject them unconditionally (explicitly "not allowed").
    # 强制认证端点：只有admin_api_key可以解锁；如果未设置admin_api_key，
    # 无条件拒绝（明确"不允许"）。
    if auth_level == AuthLevel.ADMIN_FORCE:  # 管理强制级别
        if not admin_api_key:  # 如果未配置admin_api_key
            return AuthDecision(allowed=False, error_status_code=403)  # 返回403禁止
        if not _check_bearer_token(authorization_header, admin_api_key):  # 检查admin_api_key
            return AuthDecision(allowed=False)  # 认证失败
        return AuthDecision(allowed=True)  # 认证成功

    # Optional-auth endpoints:
    # - no keys configured: allow
    # - only api_key: require api_key
    # - only admin_api_key: require admin_api_key
    # - both: require admin_api_key (api_key is NOT accepted)
    # 可选认证端点：
    # - 无密钥配置：允许
    # - 仅api_key：需要api_key
    # - 仅admin_api_key：需要admin_api_key
    # - 两者都有：需要admin_api_key（不接受api_key）
    if auth_level == AuthLevel.ADMIN_OPTIONAL:  # 管理可选级别
        if admin_api_key:  # 如果配置了admin_api_key
            return AuthDecision(  # 需要admin_api_key认证
                allowed=_check_bearer_token(authorization_header, admin_api_key)  # 检查admin_api_key
            )
        elif api_key:  # 如果仅配置了api_key
            return AuthDecision(  # 需要api_key认证
                allowed=_check_bearer_token(authorization_header, api_key)  # 检查api_key
            )
        else:  # 无密钥配置
            return AuthDecision(allowed=True)  # 允许访问

    # Normal endpoints:
    # - if api_key is configured, require api_key (even if admin_api_key is also configured)
    # - otherwise allow (including the "admin_api_key only" case)
    # 普通端点：
    # - 如果配置了api_key，需要api_key（即使同时配置了admin_api_key）
    # - 否则允许（包括仅配置admin_api_key的情况）
    if api_key:  # 如果配置了api_key
        return AuthDecision(allowed=_check_bearer_token(authorization_header, api_key))  # 需要api_key认证

    return AuthDecision(allowed=True)  # 无密钥配置，允许访问


def add_api_key_middleware(  # 添加API密钥认证中间件到FastAPI应用
    app,  # FastAPI应用实例
    *,
    api_key: Optional[str],  # API密钥
    admin_api_key: Optional[str],  # 管理API密钥
):
    """Add middleware for three endpoint auth levels: normal/admin_optional/admin_force."""
    # 为三种端点认证级别添加中间件：普通/管理可选/管理强制。
    # Import lazily so `decide_request_auth()` can be unit-tested without FastAPI installed.
    # 延迟导入以便`decide_request_auth()`可以在不安装FastAPI的情况下进行单元测试。
    from fastapi.responses import ORJSONResponse  # 导入ORJSON响应类
    from starlette.requests import Request  # 导入Starlette请求类

    class _ApiKeyASGIMiddleware:  # ASGI原生中间件，保留客户端断开事件
        """ASGI-native middleware to preserve client disconnect events."""
        # ASGI原生中间件，保留客户端断开连接事件。

        def __init__(self, app, *, api_key, admin_api_key, fastapi_app):  # 初始化中间件
            self.app = app  # ASGI应用
            self.api_key = api_key  # API密钥
            self.admin_api_key = admin_api_key  # 管理API密钥
            self.fastapi_app = fastapi_app  # FastAPI应用引用（用于路由匹配）

        async def __call__(self, scope, receive, send):  # ASGI调用接口
            if scope["type"] != "http":  # 如果不是HTTP请求
                await self.app(scope, receive, send)  # 直接传递给下一个ASGI应用
                return  # 返回

            request = Request(scope, receive=receive)  # 创建Starlette请求对象
            path = request.url.path  # 获取请求路径
            authz = request.headers.get("Authorization")  # 获取Authorization头
            level = _get_auth_level_from_app_and_scope(self.fastapi_app, scope)  # 解析端点认证级别
            decision = decide_request_auth(  # 执行认证决策
                method=request.method,  # HTTP方法
                path=path,  # 请求路径
                authorization_header=authz,  # Authorization头
                api_key=self.api_key,  # API密钥
                admin_api_key=self.admin_api_key,  # 管理API密钥
                auth_level=level,  # 认证级别
            )

            if not decision.allowed:  # 如果认证未通过
                response = ORJSONResponse(  # 创建错误响应
                    content={  # 响应内容
                        "error": (  # 错误消息
                            "Unauthorized"  # 401未授权
                            if decision.error_status_code == 401  # 根据状态码选择消息
                            else "Forbidden"  # 403禁止
                        )
                    },
                    status_code=decision.error_status_code,  # HTTP状态码
                )
                await response(scope, receive, send)  # 发送错误响应
                return  # 返回

            await self.app(scope, receive, send)  # 认证通过，传递给下一个ASGI应用

    app.add_middleware(  # 将中间件添加到应用
        _ApiKeyASGIMiddleware,  # 中间件类
        api_key=api_key,  # API密钥
        admin_api_key=admin_api_key,  # 管理API密钥
        fastapi_app=app,  # FastAPI应用引用
    )
