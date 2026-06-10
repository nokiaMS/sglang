# 网络工具模块
# 提供端口管理、套接字绑定、IP地址检测、ZeroMQ套接字配置、
# 网络地址解析等网络相关工具函数
from __future__ import annotations  # 启用延迟注解评估

import ipaddress  # 导入IP地址处理模块
import logging  # 导入日志模块
import os  # 导入操作系统模块
import socket  # 导入套接字模块
import time  # 导入时间模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Optional, Tuple, Union  # 导入类型提示

import psutil  # 导入系统工具库
import zmq  # 导入ZeroMQ库

logger = logging.getLogger(__name__)  # 创建日志记录器


def get_open_port() -> int:  # 获取一个可用的端口号
    port = os.getenv("SGLANG_PORT")  # 从环境变量获取端口
    if port is not None:  # 如果设置了环境变量
        port = int(port)  # 转换为整数
        while True:  # 循环查找可用端口
            if is_port_available(port):  # 如果端口可用
                return port  # 返回该端口
            logger.info("Port %d is already in use, trying port %d", port, port + 1)  # 记录日志
            port += 1  # 尝试下一个端口
    sock = try_bind_socket()  # 绑定一个套接字获取可用端口
    port = sock.getsockname()[1]  # 获取绑定的端口号
    sock.close()  # 关闭套接字
    return port  # 返回端口号


def is_valid_ipv6_address(address: str) -> bool:  # 验证是否为有效的IPv6地址
    try:
        ipaddress.IPv6Address(address)  # 尝试解析为IPv6地址
        return True  # 解析成功，返回True
    except ValueError:  # 如果解析失败
        return False


def find_process_using_port(port: int) -> Optional[psutil.Process]:  # 查找占用指定端口的进程
    for conn in psutil.net_connections(kind="inet"):  # 遍历所有网络连接
        if conn.laddr.port == port:  # 如果连接的本地端口匹配
            try:
                return psutil.Process(conn.pid)  # 返回对应的进程对象
            except psutil.NoSuchProcess:
                # It could happen by race condition (the proc dies when psutil.Process is called).
                pass  # 竞态条件：进程可能在查询时已终止

    return None  # 未找到占用端口的进程


def wait_port_available(  # 等待指定端口变为可用
    port: int, port_name: str, timeout_s: int = 30, raise_exception: bool = True
) -> bool:
    for i in range(timeout_s):  # 循环等待
        if is_port_available(port):  # 如果端口可用
            return True  # 返回True

        if i > 10 and i % 5 == 0:  # 等待超过1秒后，每0.5秒检查一次占用进程
            process = find_process_using_port(port)  # 查找占用进程
            if process is None:  # 如果找不到进程
                logger.warning(
                    f"The port {port} is in use, but we could not find the process that uses it."
                )

            pid = process.pid  # 获取进程PID
            error_message = f"{port_name} is used by a process already. {process.name()=}' {process.cmdline()=} {process.status()=} {pid=}"
            logger.info(
                f"port {port} is in use. Waiting for {i} seconds for {port_name} to be available. {error_message}"
            )
        time.sleep(0.1)  # 等待100毫秒

    if raise_exception:  # 如果需要抛出异常
        raise ValueError(
            f"{port_name} at {port} is not available in {timeout_s} seconds. {error_message}"
        )
    return False


def _get_addrinfos_for_bind(host=None, port=0):  # 获取用于绑定的去重地址信息元组
    """Return deduplicated addrinfo tuples for binding (one per address family).

    Args:
        host: Bind address. None (with AI_PASSIVE) resolves to wildcard
              addresses (0.0.0.0 / ::) suitable for accepting on all interfaces.
        port: Port number. 0 lets the OS assign an available ephemeral port.

    Flags:
        AI_ADDRCONFIG — only return families actually configured on this host.
        AI_PASSIVE    — return wildcard addresses suitable for bind().

    Falls back to AF_INET if getaddrinfo fails (e.g. DNS misconfiguration).
    """
    try:
        infos = socket.getaddrinfo(  # 获取地址信息
            host,
            port,
            socket.AF_UNSPEC,  # 不指定地址族
            socket.SOCK_STREAM,  # TCP流套接字
            0,
            socket.AI_ADDRCONFIG | socket.AI_PASSIVE,  # 仅返回已配置的地址族和通配地址
        )
        deduped = []  # 去重后的地址信息列表
        seen_families = set()  # 已见的地址族集合
        for info in infos:  # 遍历所有地址信息
            if info[0] not in seen_families:  # 如果地址族未见过
                seen_families.add(info[0])  # 添加到已见集合
                deduped.append(info)  # 添加到去重列表
        # Prefer IPv4 so that callers without an explicit host get consistent
        # behaviour across platforms (some OSes list IPv6 first).
        deduped.sort(key=lambda x: (x[0] != socket.AF_INET,))  # IPv4优先排序
        return deduped
    except socket.gaierror:  # 如果地址解析失败
        fallback_host = "0.0.0.0" if host is None else host  # 回退到0.0.0.0或指定主机
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (fallback_host, port))]  # 返回IPv4回退地址


def try_bind_socket(host=None, port=0, *, reuse_addr=True, listen=False):  # 在第一个可用的地址族上绑定TCP套接字
    """Bind a TCP socket on the first available address family (IPv4/IPv6).

    Iterates over address families returned by _get_addrinfos_for_bind and
    returns the first socket that successfully binds.

    Args:
        host: Bind address. None binds to all interfaces (0.0.0.0 / ::).
        port: Port number. 0 lets the OS assign an available ephemeral port;
              use sock.getsockname()[1] to retrieve the assigned port.
        reuse_addr: Set SO_REUSEADDR to allow quick port reuse after close.
        listen: Call listen(1) after bind, making the socket ready to accept.

    Returns:
        The bound socket. Caller is responsible for closing it.

    Raises:
        OSError: If bind fails on all configured address families.
    """
    for family, socktype, proto, _, sockaddr in _get_addrinfos_for_bind(host, port):  # 遍历地址族
        sock = socket.socket(family, socktype, proto)  # 创建套接字
        try:
            if family == socket.AF_INET6:  # 如果是IPv6
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)  # 设置仅IPv6模式
            if reuse_addr:  # 如果启用地址重用
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 设置SO_REUSEADDR
            sock.bind(sockaddr)  # 绑定地址
            if listen:  # 如果需要监听
                sock.listen(1)  # 开始监听
            return sock  # 返回已绑定的套接字
        except (OSError, OverflowError):  # 如果绑定失败
            sock.close()  # 关闭套接字
    raise OSError(f"Could not bind port {port} on any configured address family")  # 所有地址族都绑定失败


def is_port_available(port):  # 检查端口在所有配置的地址族上是否可用
    """Return whether a port is available on all configured address families."""
    try:
        for family, socktype, proto, _, sockaddr in _get_addrinfos_for_bind(port=port):  # 遍历地址族
            sock = socket.socket(family, socktype, proto)  # 创建套接字
            try:
                if family == socket.AF_INET6:  # 如果是IPv6
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)  # 设置仅IPv6模式
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 设置SO_REUSEADDR
                sock.bind(sockaddr)  # 尝试绑定
            finally:
                sock.close()  # 关闭套接字
        return True  # 所有地址族都可绑定，端口可用
    except (OSError, OverflowError):  # 如果任何地址族绑定失败
        return False


def get_free_port():  # 获取一个空闲端口号
    sock = try_bind_socket()  # 绑定一个套接字
    port = sock.getsockname()[1]  # 获取绑定的端口号
    sock.close()  # 关闭套接字
    return port  # 返回端口号


def bind_port(port):  # 绑定到指定端口（假设端口可用）
    """Bind to a specific port, assuming it's available."""
    return try_bind_socket(port=port, listen=True)


def get_zmq_socket_on_host(  # 在指定主机上创建并配置ZeroMQ套接字
    context: zmq.Context,
    socket_type: zmq.SocketType,
    host: Optional[str] = None,
) -> Tuple[int, zmq.Socket]:
    """Create and configure a ZeroMQ socket.

    Args:
        context: ZeroMQ context to create the socket from.
        socket_type: Type of ZeroMQ socket to create.
        host: Host to bind to, without "tcp://" prefix. Defaults to
            "127.0.0.1" (localhost-only) to avoid exposing unauthenticated
            sockets to the network (CVE-2026-3060). Callers that need
            cross-machine reachability must pass an explicit host.

    Returns:
        Tuple of (port, socket) where port is the randomly assigned TCP port.
    """
    socket = context.socket(socket_type)  # 创建ZeroMQ套接字
    config_socket(socket, socket_type)  # 配置套接字
    if host is None:  # 如果未指定主机
        host = "127.0.0.1"  # 默认绑定到本地主机
    if is_valid_ipv6_address(host):  # 如果是IPv6地址
        socket.setsockopt(zmq.IPV6, 1)  # 启用IPv6
        bind_host = f"tcp://[{host}]"  # IPv6地址用方括号包裹
    else:
        bind_host = f"tcp://{host}"  # IPv4或主机名直接拼接
    port = socket.bind_to_random_port(bind_host)  # 绑定到随机端口
    return port, socket  # 返回端口和套接字


def config_socket(socket, socket_type: zmq.SocketType):  # 根据套接字类型配置缓冲区大小和水位线
    mem = psutil.virtual_memory()  # 获取虚拟内存信息
    total_mem = mem.total / 1024**3  # 总内存（GB）
    available_mem = mem.available / 1024**3  # 可用内存（GB）
    if total_mem > 32 and available_mem > 16:  # 如果内存充足
        buf_size = int(0.5 * 1024**3)  # 设置0.5GB缓冲区
    else:
        buf_size = -1  # 使用系统默认缓冲区大小

    def set_send_opt():  # 设置发送端选项
        socket.setsockopt(zmq.SNDHWM, 0)  # 发送高水位线设为0（无限制）
        socket.setsockopt(zmq.SNDBUF, buf_size)  # 设置发送缓冲区大小

    def set_recv_opt():  # 设置接收端选项
        socket.setsockopt(zmq.RCVHWM, 0)  # 接收高水位线设为0（无限制）
        socket.setsockopt(zmq.RCVBUF, buf_size)  # 设置接收缓冲区大小

    if socket_type == zmq.PUSH:  # 如果是PUSH套接字
        set_send_opt()  # 仅设置发送选项
    elif socket_type == zmq.PULL:  # 如果是PULL套接字
        set_recv_opt()  # 仅设置接收选项
    elif socket_type in [zmq.DEALER, zmq.REQ, zmq.REP]:  # 如果是双向通信套接字
        set_send_opt()  # 设置发送选项
        set_recv_opt()  # 设置接收选项
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")  # 不支持的套接字类型


def get_local_ip_by_nic(interface: str = None) -> Optional[str]:  # 通过网络接口名称获取本地IP地址
    if not (interface := interface or os.environ.get("SGLANG_LOCAL_IP_NIC", None)):  # 获取接口名称
        return None  # 未指定接口
    try:
        import netifaces  # 导入网络接口库
    except ImportError as e:
        raise ImportError(
            "Environment variable SGLANG_LOCAL_IP_NIC requires package netifaces, please install it through 'pip install netifaces'"
        ) from e

    try:
        addresses = netifaces.ifaddresses(interface)  # 获取接口地址
        if netifaces.AF_INET in addresses:  # 如果有IPv4地址
            for addr_info in addresses[netifaces.AF_INET]:  # 遍历IPv4地址
                ip = addr_info.get("addr")  # 获取IP地址
                if ip and ip != "127.0.0.1" and ip != "0.0.0.0":  # 排除回环和全零地址
                    return ip
        if netifaces.AF_INET6 in addresses:  # 如果有IPv6地址
            for addr_info in addresses[netifaces.AF_INET6]:  # 遍历IPv6地址
                ip = addr_info.get("addr")  # 获取IP地址
                if ip and not ip.startswith("fe80::") and ip != "::1":  # 排除链路本地和回环地址
                    return ip.split("%")[0]  # 去除区域标识符
    except (ValueError, OSError) as e:
        logger.warning(
            f"{e} Can not get local ip from NIC. Please verify whether SGLANG_LOCAL_IP_NIC is set correctly."
        )
    return None  # 无法获取IP地址


def get_local_ip_by_remote() -> Optional[str]:  # 通过连接远程DNS服务器发现本地IP地址
    # Google's public DNS servers, used to discover the local IP.
    # UDP connect doesn't send packets; it just selects the right source address.
    # https://developers.google.com/speed/public-dns/docs/using#addresses
    # Try IPv4 first, then IPv6. getaddrinfo on a literal IP returns exactly
    # one result, so we unpack directly instead of looping.
    for dns_host, dns_port in [("8.8.8.8", 80), ("2001:4860:4860::8888", 80)]:  # Google公共DNS
        try:
            family, socktype, proto, _, sockaddr = socket.getaddrinfo(  # 解析DNS服务器地址
                dns_host,
                dns_port,
                socket.AF_UNSPEC,
                socket.SOCK_DGRAM,
                0,
                socket.AI_ADDRCONFIG,
            )[0]
            with socket.socket(family, socktype, proto) as s:  # 创建UDP套接字
                s.connect(sockaddr)  # 连接到DNS服务器（UDP不发送数据包）
                return s.getsockname()[0]  # 返回本地IP地址
        except (socket.gaierror, OSError):  # 如果连接失败
            continue  # 尝试下一个DNS服务器

    # Fallback: resolve the local hostname to an IP address via /etc/hosts or DNS.
    # Unreliable — many machines resolve hostname to 127.0.0.1, so we skip loopback.
    try:
        hostname = socket.gethostname()  # 获取本机主机名
        ip = socket.getaddrinfo(  # 解析主机名
            hostname, None, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG
        )[0][4][0]
        if ip and ip not in ("127.0.0.1", "0.0.0.0", "::1"):  # 排除回环地址
            return ip
    except Exception:
        pass  # 忽略异常

    logger.warning("Can not get local ip by remote")  # 记录警告
    return None


def get_local_ip_auto(fallback: str = None) -> str:  # 自动检测本地IP地址，使用多种回退策略
    """
    Automatically detect the local IP address using multiple fallback strategies.

    This function attempts to obtain the local IP address through several methods.
    If all methods fail, it returns the specified fallback value or raises an exception.

    Args:
        fallback (str, optional): Fallback IP address to return if all detection
            methods fail. For server applications, explicitly set this to
            "0.0.0.0" (IPv4) or "::" (IPv6) to bind to all available interfaces.
            Defaults to None.

    Returns:
        str: The detected local IP address, or the fallback value if detection fails.

    Raises:
        ValueError: If IP detection fails and no fallback value is provided.

    Note:
        The function tries detection methods in the following order:
        1. Direct IP detection via get_ip()
        2. Network interface enumeration via get_local_ip_by_nic()
        3. Remote connection method via get_local_ip_by_remote()
    """
    # Try environment variable  # 尝试环境变量
    host_ip = os.getenv("SGLANG_HOST_IP", "") or os.getenv("HOST_IP", "")  # 从环境变量获取IP
    if host_ip:  # 如果有值
        return host_ip  # 返回环境变量中的IP
    logger.debug("get_ip failed")  # 记录调试日志
    # Fallback  # 回退策略1
    if ip := get_local_ip_by_nic():  # 通过网络接口获取IP
        return ip
    logger.debug("get_local_ip_by_nic failed")  # 记录调试日志
    # Fallback  # 回退策略2
    if ip := get_local_ip_by_remote():  # 通过远程连接获取IP
        return ip
    logger.debug("get_local_ip_by_remote failed")  # 记录调试日志
    if fallback:  # 如果有回退值
        return fallback  # 返回回退值
    raise ValueError("Can not get local ip")  # 无法获取IP，抛出异常


def get_zmq_socket(  # 创建并配置ZeroMQ套接字（支持绑定和连接模式）
    context: zmq.Context,
    socket_type: zmq.SocketType,
    endpoint: Optional[str] = None,
    bind: bool = True,
) -> Union[zmq.Socket, Tuple[int, zmq.Socket]]:
    """Create and configure a ZeroMQ socket.

    Args:
        context: ZeroMQ context to create the socket from.
        socket_type: Type of ZeroMQ socket to create.
        endpoint: Optional endpoint to bind/connect to. If None, binds to a random TCP port.
        bind: Whether to bind (True) or connect (False) to the endpoint. Ignored if endpoint is None.

    Returns:
        If endpoint is None: Tuple of (port, socket) where port is the randomly assigned TCP port.
        If endpoint is provided: The configured ZeroMQ socket.
    """
    socket = context.socket(socket_type)  # 创建ZeroMQ套接字

    if endpoint is None:  # 如果未指定端点
        # Bind to random TCP port  # 绑定到随机TCP端口
        config_socket(socket, socket_type)  # 配置套接字
        port = socket.bind_to_random_port("tcp://*")  # 绑定到随机端口
        return port, socket  # 返回端口和套接字
    else:
        # Handle IPv6 if endpoint contains brackets  # 处理IPv6端点
        if endpoint.find("[") != -1:  # 如果端点包含方括号（IPv6）
            socket.setsockopt(zmq.IPV6, 1)  # 启用IPv6

        config_socket(socket, socket_type)  # 配置套接字

        if bind:  # 如果是绑定模式
            socket.bind(endpoint)  # 绑定到端点
        else:  # 如果是连接模式
            socket.connect(endpoint)  # 连接到端点

        return socket  # 返回配置好的套接字


def _is_ipv6(host: str) -> bool:  # 检查主机名是否为有效的IPv6地址（不含方括号）
    """Check whether *host* is a valid IPv6 address (without brackets)."""
    try:
        ipaddress.IPv6Address(host)  # 尝试解析为IPv6地址
        return True
    except ValueError:
        return False


def _wrap(host: str) -> str:  # 将IPv6地址用方括号包裹；IPv4/主机名原样传递
    """Wrap an IPv6 address in brackets; pass IPv4/hostname through."""
    return f"[{host}]" if _is_ipv6(host) else host


def _parse_port(s: str) -> int:  # 解析端口号字符串
    try:
        port = int(s)  # 转换为整数
    except ValueError:
        raise ValueError(f"Invalid port number: {s!r}")  # 无效端口号
    if not (0 <= port <= 65535):  # 端口范围检查
        raise ValueError(f"Port out of range (0-65535): {port}")
    return port


@dataclass(frozen=True)
class NetworkAddress:  # 网络地址数据类，封装主机和端口信息
    host: str  # 主机地址
    port: int  # 端口号

    def __post_init__(self):  # 初始化后处理，自动去除IPv6地址的方括号
        # Auto-strip IPv6 brackets so callers can pass "[::1]" or "::1"
        if self.host.startswith("[") and self.host.endswith("]"):  # 如果地址被方括号包裹
            object.__setattr__(self, "host", self.host[1:-1])  # 去除方括号

    @property
    def is_ipv6(self) -> bool:  # 判断是否为IPv6地址
        return _is_ipv6(self.host)

    @property
    def family(self) -> socket.AddressFamily:  # 获取地址族
        return socket.AF_INET6 if self.is_ipv6 else socket.AF_INET  # IPv6返回AF_INET6，否则AF_INET

    def to_url(self, scheme: str = "http") -> str:  # 转换为URL字符串
        """``http://127.0.0.1:30000`` or ``http://[::1]:30000``."""
        return f"{scheme}://{_wrap(self.host)}:{self.port}"

    def to_tcp(self) -> str:  # 转换为TCP端点字符串（用于ZMQ/torch分布式）
        """``tcp://`` endpoint for ZMQ / torch distributed."""
        return self.to_url("tcp")

    def to_host_port_str(self) -> str:  # 转换为host:port字符串（用于gRPC、会话ID、日志）
        """``host:port`` string for gRPC listen address, session IDs, logs."""
        return f"{_wrap(self.host)}:{self.port}"

    @staticmethod
    def resolve_host(host: str) -> str:  # 解析主机名，如果是IP则原样返回，否则DNS解析
        """Return *host* as-is if it's an IP, otherwise DNS-resolve to one."""
        try:
            ipaddress.ip_address(host)  # 尝试解析为IP地址
            return host  # 是IP地址，原样返回
        except ValueError:
            pass  # 不是IP地址，继续DNS解析
        try:
            return socket.getaddrinfo(  # DNS解析主机名
                host, None, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG
            )[0][4][0]
        except socket.gaierror as e:
            raise ValueError(f"Cannot resolve host {host!r}: {e}") from e  # 解析失败

    def resolved(self) -> NetworkAddress:  # DNS解析主机名为IP，如果已是IP则返回自身
        """DNS-resolve hostname to IP; return self if already an IP."""
        ip = self.resolve_host(self.host)  # 解析主机名
        return self if ip == self.host else NetworkAddress(ip, self.port)  # 如果IP未变则返回自身

    def to_bind_tuple(self) -> Tuple[str, int]:  # 转换为用于socket.bind()/connect()的(host, port)元组
        """Raw ``(host, port)`` tuple for ``socket.bind()`` / ``socket.connect()``.

        Returns the *unwrapped* host — sockets need the raw address, not
        the bracketed form.
        """
        return (self.host, self.port)

    @staticmethod
    def parse(addr: str) -> NetworkAddress:  # 解析host:port字符串为NetworkAddress对象
        """Parse a ``host:port`` string into a ``NetworkAddress``.

        Accepted formats::

            [::1]:8000          → NetworkAddress("::1", 8000)
            127.0.0.1:8000      → NetworkAddress("127.0.0.1", 8000)
            my-hostname:8000    → NetworkAddress("my-hostname", 8000)

        IPv6 addresses **must** be bracketed.  Bare ``::1:8000`` is
        ambiguous and will raise ``ValueError``.

        Raises:
            ValueError: If the string cannot be unambiguously parsed.
        """
        if not addr:  # 如果地址为空
            raise ValueError("Empty address string")

        # --- Bracketed IPv6: [addr]:port ---  # 带方括号的IPv6格式
        if addr.startswith("["):  # 如果以方括号开头
            close = addr.find("]")  # 查找右方括号位置
            if close == -1:  # 如果没有右方括号
                raise ValueError(f"Missing closing bracket in IPv6 address: {addr!r}")
            host = addr[1:close]  # 提取方括号内的地址
            if not _is_ipv6(host):  # 如果不是有效的IPv6地址
                raise ValueError(f"Invalid IPv6 address inside brackets: {host!r}")
            rest = addr[close + 1 :]  # 提取右方括号后的部分
            if not rest.startswith(":") or len(rest) < 2:  # 如果缺少端口号
                raise ValueError(
                    f"Expected ':port' after closing bracket, got: {rest!r}"
                )
            return NetworkAddress(host, _parse_port(rest[1:]))  # 返回解析结果

        # --- Plain host:port (IPv4 / hostname) ---  # 普通host:port格式（IPv4/主机名）
        if ":" not in addr:  # 如果没有冒号
            raise ValueError(f"Missing port in address (expected host:port): {addr!r}")
        host, port_str = addr.rsplit(":", 1)  # 从右侧分割，提取主机和端口
        if not host:  # 如果主机为空
            raise ValueError(f"Empty host in address: {addr!r}")
        # Guard against bare IPv6 slipping through  # 防止裸IPv6地址混入
        if ":" in host and _is_ipv6(host):  # 如果主机包含冒号且是IPv6
            raise ValueError(
                f"Bare IPv6 address without brackets is ambiguous: {addr!r}. "
                f"Use [{host}]:{port_str} instead."
            )
        return NetworkAddress(host, _parse_port(port_str))  # 返回解析结果

    def __str__(self) -> str:  # 字符串表示
        return self.to_host_port_str()

    def __repr__(self) -> str:  # 详细字符串表示
        return f"NetworkAddress({self.host!r}, {self.port})"
