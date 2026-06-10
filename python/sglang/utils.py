# SGLang 通用工具模块，提供 HTTP 请求、图像/视频编码、端口管理、进程管理等实用功能
"""Common utilities"""

import importlib
import json
import logging
import os
import random
import ssl
import subprocess
import sys
import time
import traceback
import urllib.request
import warnings
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, wraps
from io import BytesIO
from json import dumps
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import numpy as np
import pybase64
import requests
from IPython.display import HTML, display
from pydantic import BaseModel
from tqdm import tqdm

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# 已知的非 Diffusers 格式扩散模型名称模式映射
KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS: dict[str, str] = {
    "hunyuan3d": "Hunyuan3D2Pipeline",
    "flux.2-dev-nvfp4": "Flux2NvfpPipeline",
}


# 从环境变量加载扩散模型覆盖注册表
def load_diffusion_overlay_registry_from_env() -> dict[str, dict[str, Any]]:
    raw_value = os.getenv("SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY", "").strip()
    if not raw_value:
        return {}

    # 根据是否以 { 开头判断是内联 JSON 还是文件路径
    if raw_value.startswith("{"):
        payload = json.loads(raw_value)
    else:
        with open(os.path.expanduser(raw_value), encoding="utf-8") as f:
            payload = json.load(f)

    if not isinstance(payload, dict):
        return {}

    # 规范化注册表项，统一转换为字典格式
    normalized: dict[str, dict[str, Any]] = {}
    for source_model_id, spec in payload.items():
        if isinstance(spec, str):
            normalized[source_model_id] = {"overlay_repo_id": spec}
        elif isinstance(spec, dict) and spec.get("overlay_repo_id"):
            normalized[source_model_id] = dict(spec)
    return normalized


# 检查模型路径是否在扩散模型覆盖注册表中有匹配
def has_diffusion_overlay_registry_match(
    model_path: str, registry: dict[str, dict[str, Any]] | None = None
) -> bool:
    registry = (
        load_diffusion_overlay_registry_from_env() if registry is None else registry
    )
    if model_path in registry:
        return True
    if not os.path.exists(model_path):
        return False
    # 使用路径基本名称进行模糊匹配
    base_name = os.path.basename(os.path.normpath(model_path))
    return any(base_name == key.rsplit("/", 1)[-1] for key in registry)


# 检查模型路径是否属于已知的非 Diffusers 格式扩散模型
def is_known_non_diffusers_diffusion_model(model_path: str) -> bool:
    model_path_lower = model_path.lower()
    return any(
        pattern in model_path_lower
        for pattern in KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS
    )


# 装饰器：确保函数只执行一次
def execute_once(func):
    has_run = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal has_run
        if not has_run:
            func(*args, **kwargs)
            has_run = True

    return wrapper


# 只记录一次 info 级别日志的辅助函数
@execute_once
def info_once(message: str):
    logger.info(message)


# 将 JSON 模式转换为字符串
def convert_json_schema_to_str(json_schema: Union[dict, str, Type[BaseModel]]) -> str:
    """Convert a JSON schema to a string.
    Parameters
    ----------
    json_schema
        The JSON schema.
    Returns
    -------
    str
        The JSON schema converted to a string.
    Raises
    ------
    ValueError
        If the schema is not a dictionary, a string or a Pydantic class.
    """
    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema)
    elif isinstance(json_schema, str):
        schema_str = json_schema
    elif issubclass(json_schema, BaseModel):
        # 从 Pydantic 模型生成 JSON 模式字符串
        schema_str = json.dumps(json_schema.model_json_schema())
    else:
        raise ValueError(
            f"Cannot parse schema {json_schema}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    return schema_str


# 获取当前异常的完整回溯字符串
def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


# 检查列表中的元素是否为同一类型
def is_same_type(values: list):
    """Return whether the elements in values are of the same type."""
    if len(values) <= 1:
        return True
    else:
        t = type(values[0])
        return all(isinstance(v, t) for v in values[1:])


# 读取 JSONL 格式文件，跳过注释行
def read_jsonl(filename: str):
    """Read a JSONL file."""
    with open(filename) as fin:
        for line in fin:
            # 跳过以 # 开头的注释行
            if line.startswith("#"):
                continue
            yield json.loads(line)


# 将程序状态转储为文本文件
def dump_state_text(filename: str, states: list, mode: str = "w"):
    """Dump program state in a text file."""
    from sglang.lang.interpreter import ProgramState

    with open(filename, mode) as fout:
        for i, s in enumerate(states):
            if isinstance(s, str):
                pass
            elif isinstance(s, ProgramState):
                # 将 ProgramState 转换为文本
                s = s.text()
            else:
                s = str(s)

            fout.write(
                "=" * 40 + f" {i} " + "=" * 40 + "\n" + s + "\n" + "=" * 80 + "\n\n"
            )


# 规范化基础 URL，将主机和端口合并为完整 URL
def normalize_base_url(host: str, port: int) -> str:
    from sglang.srt.utils.network import NetworkAddress

    # 如果 host 已包含协议前缀，发出弃用警告
    if host.startswith("http://") or host.startswith("https://"):
        warnings.warn(
            f"Including the scheme in --host ('{host}') is deprecated. "
            f"Pass just the hostname (e.g. '127.0.0.1') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return f"{host}:{port}"
    return NetworkAddress(host, port).to_url()


# HTTP 响应封装类，提供 json、text 和 status_code 属性
class HttpResponse:
    def __init__(self, resp):
        self.resp = resp

    # 延迟读取响应体
    @cached_property
    def _body(self):
        return self.resp.read()

    # 将响应体解析为 JSON
    def json(self):
        return json.loads(self._body)

    # 将响应体解码为 UTF-8 文本
    @property
    def text(self):
        return self._body.decode("utf-8", errors="replace")

    # 获取 HTTP 状态码
    @property
    def status_code(self):
        return self.resp.status


# 发送 HTTP 请求，支持流式和非流式模式，使用底层 urllib API 以提升性能
def http_request(
    url,
    json=None,
    stream=False,
    api_key=None,
    verify=None,
    method: Optional[str] = None,
):
    """A faster version of requests.post with low-level urllib API."""
    headers = {"Content-Type": "application/json; charset=utf-8"}

    # add the Authorization header if an api key is provided
    # 如果提供了 API 密钥，添加 Authorization 请求头
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    if stream:
        # 流式请求使用 requests 库
        return requests.post(url, json=json, stream=True, headers=headers)
    else:
        # 非流式请求使用底层 urllib API
        req = urllib.request.Request(url, headers=headers, method=method)
        if json is None:
            data = None
        else:
            data = bytes(dumps(json), encoding="utf-8")

        try:
            if sys.version_info >= (3, 13):
                # Python 3.13+ 版本：使用 SSL 上下文（cafile 参数已移除）
                # Python 3.13+: Use SSL context (cafile removed)
                if verify and isinstance(verify, str):
                    context = ssl.create_default_context(cafile=verify)
                else:
                    context = ssl.create_default_context()
                resp = urllib.request.urlopen(req, data=data, context=context)
            else:
                resp = urllib.request.urlopen(req, data=data, cafile=verify)
            return HttpResponse(resp)
        except urllib.error.HTTPError as e:
            # HTTP 错误也返回 HttpResponse 对象
            return HttpResponse(e)


# 将图像编码为 Base64 字符串，支持文件路径、字节数据、张量和 PIL 图像
def encode_image_base64(image_path: Union[str, bytes]):
    """Encode an image in base64."""
    if isinstance(image_path, str):
        # 从文件路径读取并编码
        with open(image_path, "rb") as image_file:
            data = image_file.read()
            return pybase64.b64encode(data).decode("utf-8")
    elif isinstance(image_path, bytes):
        # 直接编码字节数据
        return pybase64.b64encode(image_path).decode("utf-8")
    else:
        import torch

        if isinstance(image_path, torch.Tensor):
            # 将 GPU 解码的图像张量 (C, H, W) uint8 转换为 PIL 图像
            # Convert GPU-decoded image tensor (C, H, W) uint8 to PIL Image
            from PIL import Image

            tensor = image_path.cpu() if image_path.device.type != "cpu" else image_path
            image_path = Image.fromarray(tensor.permute(1, 2, 0).numpy())

        # image_path is a PIL Image
        # 将 PIL 图像编码为 Base64
        image = image_path
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return pybase64.b64encode(buffered.getvalue()).decode("utf-8")


# 将单帧图像编码为 PNG 字节
def encode_frame(frame):
    import cv2  # pip install opencv-python-headless
    from PIL import Image

    # Convert the frame to RGB (OpenCV uses BGR by default)
    # 将帧从 BGR 转换为 RGB（OpenCV 默认使用 BGR）
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image to easily convert to bytes
    # 将帧转换为 PIL 图像以便转换为字节
    im_pil = Image.fromarray(frame)

    # Convert to bytes
    # 转换为字节
    buffered = BytesIO()

    # frame_format = str(os.getenv('FRAME_FORMAT', "JPEG"))

    im_pil.save(buffered, format="PNG")

    frame_bytes = buffered.getvalue()

    # Return the bytes of the frame
    # 返回帧的字节数据
    return frame_bytes


# 将视频编码为 Base64 字符串，均匀采样指定数量的帧
def encode_video_base64(video_path: str, num_frames: int = 16):
    import cv2  # pip install opencv-python-headless

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file:{video_path}")

    # 获取视频总帧数并均匀采样帧索引
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"target_frames: {num_frames}")

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            # Handle the case where the frame could not be read
            # 处理无法读取帧的情况
            # print(f"Warning: Could not read frame at index {i}.")
            pass

    cap.release()

    # Safely select frames based on frame_indices, avoiding IndexError
    # 根据帧索引安全选择帧，避免索引越界
    frames = [frames[i] for i in frame_indices if i < len(frames)]

    # If there are not enough frames, duplicate the last frame until we reach the target
    # 如果帧数不足，复制最后一帧直到达到目标数量
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Use ThreadPoolExecutor to process and encode frames in parallel
    # 使用线程池并行处理和编码帧
    with ThreadPoolExecutor() as executor:
        encoded_frames = list(executor.map(encode_frame, frames))

    # encoded_frames = list(map(encode_frame, frames))

    # Concatenate all frames bytes
    # 拼接所有帧的字节数据
    video_bytes = b"".join(encoded_frames)

    # Encode the concatenated bytes to base64
    # 将拼接后的字节编码为 Base64
    video_base64 = "video:" + pybase64.b64encode(video_bytes).decode("utf-8")

    return video_base64


# 检查给定的 Unicode 码点是否为中文字符
def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    # 以下定义了"中文字符"为 CJK Unicode 区块中的任何字符
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


# 查找最长可打印文本子串，确保只包含完整单词
def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    # 遇到换行符时，直接返回全部文本
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    # 如果最后一个字符是中文字符，直接返回全部文本
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    # 如果倒数第二个字符是中文字符，返回除最后一个字符外的文本
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    # 否则，返回到最后一个空格处的文本（避免打印不完整单词的简单启发式方法）
    else:
        return text[: text.rfind(" ") + 1]


# 延迟导入类，使 import sglang 更快
class LazyImport:
    """Lazy import to make `import sglang` run faster."""

    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None

    # 实际执行延迟导入
    def _load(self):
        if self._module is None:
            module = importlib.import_module(self.module_name)
            self._module = getattr(module, self.class_name)
        return self._module

    # 代理属性访问到实际模块
    def __getattr__(self, name: str):
        module = self._load()
        return getattr(module, name)

    # 代理调用到实际模块
    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)


# 从 URL 下载文件并缓存到本地
def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    # 检查缓存文件是否已存在
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    # 以流式方式下载响应以显示进度条
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    # 获取文件总大小（字节）
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB
    # 以 1KB 分块下载

    # Use tqdm to display the progress bar
    # 使用 tqdm 显示下载进度条
    with (
        open(filename, "wb") as f,
        tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


# 检查是否在 CI 环境中运行
def is_in_ci() -> bool:
    return envs.SGLANG_IS_IN_CI.get()


# 在 CI 环境中以高亮蓝色显示内容，否则普通打印
def print_highlight(html_content: str):
    if is_in_ci():
        html_content = str(html_content).replace("\n", "<br>")
        display(HTML(f"<strong style='color: #00008B;'>{html_content}</strong>"))
    else:
        print(html_content)


# 进程与端口锁的映射表（弱引用字典）
process_socket_map = weakref.WeakKeyDictionary()


# 保留一个可用端口，通过尝试绑定套接字来获取
def reserve_port(host, start=30000, end=40000):
    """
    Reserve an available port by trying to bind a socket.
    Returns a tuple (port, lock_socket) where `lock_socket` is kept open to hold the lock.
    """
    from sglang.srt.utils.network import try_bind_socket

    # 随机打乱候选端口以避免冲突
    candidates = list(range(start, end))
    random.shuffle(candidates)
    for port in candidates:
        try:
            sock = try_bind_socket(host, port)
            return port, sock
        except OSError:
            continue
    raise RuntimeError("No free port available.")


# 释放保留的端口，关闭锁定的套接字
def release_port(lock_socket):
    """
    Release the reserved port by closing the lock socket.
    """
    try:
        lock_socket.close()
    except Exception as e:
        print(f"Error closing socket: {e}")


# 执行 shell 命令并返回进程句柄，支持前导 KEY=VALUE 环境变量
def execute_shell_command(command: str) -> subprocess.Popen:
    """
    Execute a shell command and return its process handle.
    Supports leading KEY=VALUE env vars (e.g. "VAR=1 python script.py") so that
    notebook/CI commands work without requiring shell=True.
    """
    # 清理命令中的换行续行符
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    env = os.environ.copy()
    # 提取前导的环境变量赋值
    i = 0
    while i < len(parts):
        part = parts[i]
        if "=" in part and not part.startswith("-") and not part.startswith("/"):
            key, _, value = part.partition("=")
            if key and value is not None and key.replace("_", "").isalnum():
                env[key] = value
                i += 1
                continue
        break
    parts = parts[i:]
    if not parts:
        raise ValueError(
            "Command contains only environment variable assignments, no executable"
        )
    return subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT, env=env)


# 启动服务器命令，自动分配或使用指定端口
def launch_server_cmd(command: str, host: str = "0.0.0.0", port: int = None):
    """
    Launch the server using the given command.
    If no port is specified, a free port is reserved.
    """
    if port is None:
        # 未指定端口时自动保留一个可用端口
        port, lock_socket = reserve_port(host)
    else:
        lock_socket = None

    full_command = f"{command} --port {port}"
    process = execute_shell_command(full_command)

    # 记录进程与端口锁的映射关系
    if lock_socket is not None:
        process_socket_map[process] = lock_socket

    return process, port


# 终止进程并自动释放保留的端口
def terminate_process(process):
    """
    Terminate the process and automatically release the reserved port.
    """
    from sglang.srt.utils import kill_process_tree

    # 终止整个进程树
    kill_process_tree(process.pid)

    # 释放端口锁
    lock_socket = process_socket_map.pop(process, None)
    if lock_socket is not None:
        release_port(lock_socket)


# 检查进程是否已退出，若已退出则抛出 RuntimeError
def _raise_if_process_exited(process: Optional[Any]) -> None:
    if process is None:
        return

    if hasattr(process, "poll"):
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"Server process exited with code {return_code}")
        return

    if hasattr(process, "is_alive") and not process.is_alive():
        return_code = getattr(process, "exitcode", None)
        if return_code is None:
            raise RuntimeError("Server process exited")
        raise RuntimeError(f"Server process exited with code {return_code}")


# 检查是否已超过等待超时时间
def _is_wait_timeout(start_time: float, timeout: Optional[int]) -> bool:
    if timeout is None:
        return False
    return time.perf_counter() - start_time > timeout


# 等待 HTTP 端点返回 200 状态码
def wait_for_http_ready(
    url: str,
    timeout: Optional[int] = None,
    process: Optional[Any] = None,
    headers: Optional[dict] = None,
    request_timeout: int = 5,
) -> None:
    """Wait for an HTTP endpoint to return status 200."""
    start_time = time.perf_counter()
    while True:
        _raise_if_process_exited(process)
        try:
            response = requests.get(url, headers=headers, timeout=request_timeout)
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException:
            _raise_if_process_exited(process)

        # 检查是否超时
        if _is_wait_timeout(start_time, timeout):
            raise TimeoutError(
                f"Endpoint {url} did not become ready within timeout period"
            )
        time.sleep(1)


# 通过轮询 /v1/models 端点等待服务器就绪
def wait_for_server(
    base_url: str,
    timeout: int = None,
    process: Optional[subprocess.Popen] = None,
) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server.
        timeout: Maximum time to wait in seconds. None means wait forever.
        process: Optional server process used for early-exit checks.
    """
    wait_for_http_ready(
        url=f"{base_url}/v1/models",
        timeout=timeout,
        process=process,
        headers={"Authorization": "Bearer None"},
    )
    # 等待额外 5 秒确保服务器完全就绪
    time.sleep(5)
    print_highlight("""\n
        NOTE: Typically, the server runs in a separate terminal.
        In this notebook, we run the server and notebook code together, so their outputs are combined.
        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.
        To reduce the log length, we set the log level to warning for the server, the default log level is info.
        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.
        """)


# 基于类型的分发器，支持精确匹配、MRO 缓存和继承匹配
class TypeBasedDispatcher:
    def __init__(self, mapping: List[Tuple[Type, Callable]]):
        # Use dictionary for fast exact type matching, using OrderedDict(mapping)
        # to maintains registration order
        # 使用有序字典实现快速精确类型匹配，保持注册顺序
        self._mapping = OrderedDict(mapping)
        # MRO cache for inheritance-based matching
        # MRO 缓存用于基于继承的匹配
        self._mro_cache = {}
        self._fallback_fn = None

    # 添加回退函数，当没有匹配的类型时调用
    def add_fallback_fn(self, fallback_fn: Callable):
        self._fallback_fn = fallback_fn

    # 合并另一个分发器的映射
    def __iadd__(self, other: "TypeBasedDispatcher"):
        for ty, fn in other._mapping.items():
            if ty not in self._mapping:
                self._mapping[ty] = fn

        self._mro_cache.clear()
        return self

    # 根据对象类型分发到对应的处理函数
    def __call__(self, obj: Any):
        obj_type = type(obj)
        # 1. First try exact match(o(1))
        # 1. 首先尝试精确匹配（O(1)）
        fn = self._mapping.get(obj_type)
        if fn is not None:
            return fn(obj)

        # 2. If exact match fails, check MRO cache
        # 2. 精确匹配失败，检查 MRO 缓存
        cached_fn = self._mro_cache.get(obj_type)
        if cached_fn is not None:
            return cached_fn(obj)

        # 3.search in registration order for compatible type(maintains origin behavior)
        # 3. 按注册顺序搜索兼容类型（保持原始行为）
        for ty, fn in self._mapping.items():
            if isinstance(obj, ty):
                self._mro_cache[obj_type] = fn
                return fn(obj)

        # 4. if no matching type found, cache this result
        # 4. 未找到匹配类型，缓存此结果
        self._mro_cache[obj_type] = None

        # 调用回退函数或抛出异常
        if self._fallback_fn is not None:
            return self._fallback_fn(obj)
        raise ValueError(f"Invalid object: {obj}")


# 去除已有文本和新数据块之间的重叠部分
def trim_overlap(existing_text, new_chunk):
    """
    Finds the largest suffix of 'existing_text' that is a prefix of 'new_chunk'
    and removes that overlap from the start of 'new_chunk'.
    """
    max_overlap = 0
    max_possible = min(len(existing_text), len(new_chunk))
    # 从最大可能重叠长度向下搜索
    for i in range(max_possible, 0, -1):
        if existing_text.endswith(new_chunk[:i]):
            max_overlap = i
            break
    return new_chunk[max_overlap:]


# 流式生成文本并合并去重
def stream_and_merge(llm, prompt, sampling_params):
    """
    1) Streams the text,
    2) Removes chunk overlaps,
    3) Returns the merged text.
    """
    final_text = ""
    for chunk in llm.generate(prompt, sampling_params, stream=True):
        chunk_text = chunk["text"]
        cleaned_chunk = trim_overlap(final_text, chunk_text)
        final_text += cleaned_chunk
    return final_text


# 异步流式生成文本并合并去重，实时产出清理后的数据块
async def async_stream_and_merge(llm, prompt, sampling_params):
    """
    Streams tokens asynchronously, removes chunk overlaps,
    and yields the cleaned chunk in real time for printing.
    """
    final_text = ""
    generator = await llm.async_generate(prompt, sampling_params, stream=True)
    async for chunk in generator:
        chunk_text = chunk["text"]
        cleaned_chunk = trim_overlap(final_text, chunk_text)
        final_text += cleaned_chunk
        yield cleaned_chunk  # yield the non-overlapping portion
        # 产出非重叠部分


# 通过完全限定名解析对象
def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully qualified name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
