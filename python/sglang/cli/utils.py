# SGLang CLI 工具函数模块
# 提供模型类型检测（扩散模型/语言模型）、模型路径提取和 Git 提交哈希获取等工具函数
import json  # 导入 JSON 解析模块
import logging  # 导入日志记录模块
import os  # 导入操作系统接口模块
import subprocess  # 导入子进程管理模块
from functools import lru_cache  # 导入 LRU 缓存装饰器

from huggingface_hub import HfApi  # 导入 HuggingFace Hub API

from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.utils import (  # 导入扩散模型检测工具函数
    has_diffusion_overlay_registry_match,  # 检查模型路径是否匹配扩散覆盖注册表
    is_known_non_diffusers_diffusion_model,  # 检查是否为已知的非 diffusers 扩散模型
    load_diffusion_overlay_registry_from_env,  # 从环境变量加载扩散覆盖注册表
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@lru_cache(maxsize=1)  # 使用 LRU 缓存，最多缓存 1 个结果
def _load_overlay_registry() -> dict:  # 加载扩散覆盖注册表（带缓存）
    return load_diffusion_overlay_registry_from_env()  # 从环境变量加载覆盖注册表


def _is_overlay_diffusion_model(model_path: str) -> bool:  # 检查模型路径是否匹配扩散覆盖注册表
    return has_diffusion_overlay_registry_match(model_path, _load_overlay_registry())  # 检查是否匹配覆盖注册表


def _is_registered_diffusion_model(model_path: str) -> bool:  # 检查模型路径是否在扩散模型注册表中
    try:  # 尝试导入扩散模型注册表
        from sglang.multimodal_gen.registry import has_registered_diffusion_model_path  # 导入注册表检查函数
    except ImportError:  # 如果扩散模型依赖未安装
        # if diffusion dependencies are not installed  # 如果扩散模型依赖未安装
        return False  # 返回 False

    return has_registered_diffusion_model_path(model_path)  # 检查模型路径是否已注册


def _is_diffusers_model_dir(model_dir: str) -> bool:  # 检查本地目录是否包含有效的 diffusers 模型
    """Check if a local directory contains a valid diffusers model_index.json."""  # 检查本地目录是否包含有效的 diffusers model_index.json
    config_path = os.path.join(model_dir, "model_index.json")  # 构造配置文件路径
    if not os.path.exists(config_path):  # 如果配置文件不存在
        return False  # 返回 False

    with open(config_path) as f:  # 打开配置文件
        config = json.load(f)  # 加载 JSON 配置

    return "_diffusers_version" in config  # 检查是否包含 _diffusers_version 键


def _is_gated_diffusion_repo(repo_id: str) -> bool:  # 查询 HF 模型卡元数据检查受限仓库是否为 diffusers 模型
    """Query HF model card metadata to check if a gated repo is a diffusers model."""  # 查询 HF 模型卡元数据检查受限仓库是否为 diffusers 模型
    try:  # 尝试获取模型信息
        info = HfApi().model_info(repo_id)  # 获取 HuggingFace 模型信息
        return getattr(info, "library_name", None) == "diffusers"  # 检查 library_name 是否为 "diffusers"
    except Exception:  # 捕获所有异常
        return False  # 出错时返回 False


def get_is_diffusion_model(model_path: str) -> bool:  # 检测模型路径是否指向扩散模型
    """Detect whether model_path points to a diffusion model.
# 检测 model_path 是否指向扩散模型

    For local directories, checks the filesystem directly.
# 对于本地目录，直接检查文件系统
    For HF/ModelScope model IDs, attempts to fetch only model_index.json.
# 对于 HF/ModelScope 模型 ID，尝试仅获取 model_index.json
    For gated repos where file download fails, falls back to HF model card
# 对于文件下载失败受限仓库，回退到 HF 模型卡
    metadata (library_name == "diffusers").
# 元数据（library_name == "diffusers"）
    Returns False on any failure (network error, 404, offline mode, etc.)
# 任何失败时返回 False（网络错误、404、离线模式等）
    so that the caller falls through to the standard LLM server path.
# 以便调用者回退到标准 LLM 服务器路径
    """
    if _is_overlay_diffusion_model(model_path):  # 如果匹配覆盖注册表
        # short-circuit, if applicable for the overlay mechanism (diffusion-only)  # 短路，如果适用于覆盖机制（仅扩散模型）
        return True  # 返回 True

    if os.path.isdir(model_path):  # 如果是本地目录
        if _is_diffusers_model_dir(model_path):  # 检查是否为 diffusers 模型目录
            return True  # 返回 True
        return is_known_non_diffusers_diffusion_model(model_path)  # 检查是否为已知的非 diffusers 扩散模型

    if is_known_non_diffusers_diffusion_model(model_path):  # 如果是已知的非 diffusers 扩散模型
        return True  # 返回 True

    if _is_registered_diffusion_model(model_path):  # 如果在扩散模型注册表中
        return True  # 返回 True

    try:  # 尝试从远程下载 model_index.json
        if envs.SGLANG_USE_MODELSCOPE.get():  # 如果使用 ModelScope
            from modelscope import model_file_download  # 导入 ModelScope 文件下载函数

            file_path = model_file_download(  # 从 ModelScope 下载文件
                model_id=model_path, file_path="model_index.json"
            )
        else:  # 否则使用 HuggingFace
            from huggingface_hub import hf_hub_download  # 导入 HuggingFace 文件下载函数

            file_path = hf_hub_download(repo_id=model_path, filename="model_index.json")  # 从 HuggingFace 下载文件

        return _is_diffusers_model_dir(os.path.dirname(file_path))  # 检查下载目录是否为 diffusers 模型
    except Exception as e:  # 捕获下载异常
        logger.debug("Failed to auto-detect diffusion model for %s: %s", model_path, e)  # 记录调试日志
        return False  # 返回 False


def get_model_path(extra_argv):  # 从命令行参数中提取模型路径
    # Find the model_path argument  # 查找 model_path 参数
    model_path = None  # 模型路径，初始为 None
    for i, arg in enumerate(extra_argv):  # 遍历所有参数
        if arg in ("--model-path", "--model"):  # 如果是 --model-path 或 --model 形式
            if i + 1 < len(extra_argv):  # 如果有后续值
                model_path = extra_argv[i + 1]  # 获取模型路径
                break  # 找到后跳出循环
        elif arg.startswith("--model-path=") or arg.startswith("--model="):  # 如果是等号形式
            model_path = arg.split("=", 1)[1]  # 提取等号后的值
            break  # 找到后跳出循环

    if model_path is None:  # 如果未找到模型路径
        # Fallback for --help or other cases where model-path is not provided  # 回退处理 --help 或其他未提供 model-path 的情况
        if any(h in extra_argv for h in ["-h", "--help"]):  # 如果请求帮助
            raise Exception(  # 抛出使用说明异常
                "Usage: sglang serve --model-path <model-name-or-path> [additional-arguments]\n\n"
                "This command can launch either a standard language model server or a diffusion model server.\n"
                "The server type is determined by the --model-path.\n"
            )
        else:  # 其他情况
            raise Exception(  # 抛出必须提供模型路径的异常
                "Error: --model-path is required. "
                "Please provide the path to the model."
            )
    return model_path  # 返回模型路径


@lru_cache(maxsize=1)  # 使用 LRU 缓存，最多缓存 1 个结果
def get_git_commit_hash() -> str:  # 获取当前 Git 提交哈希值（带缓存）
    try:  # 尝试获取 Git 提交哈希
        commit_hash = os.environ.get("SGLANG_GIT_COMMIT")  # 优先从环境变量获取
        if not commit_hash:  # 如果环境变量未设置
            commit_hash = (  # 通过 git 命令获取
                subprocess.check_output(  # 执行 git 命令
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL  # 获取 HEAD 的提交哈希
                )
                .strip()  # 去除首尾空白
                .decode("utf-8")  # 解码为 UTF-8 字符串
            )
        _CACHED_COMMIT_HASH = commit_hash  # 缓存提交哈希
        return commit_hash  # 返回提交哈希
    except (subprocess.CalledProcessError, FileNotFoundError):  # 捕获 git 命令失败
        _CACHED_COMMIT_HASH = "N/A"  # 缓存为 "N/A"
        return "N/A"  # 返回 "N/A"
