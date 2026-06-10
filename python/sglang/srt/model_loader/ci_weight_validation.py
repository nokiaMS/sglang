# CI权重验证和缓存清理工具模块，仅在CI环境中使用，包含safetensors文件验证、缺失分片检查、损坏文件清理和自动重试逻辑
"""
CI-specific weight validation and cache cleanup utilities.

This module contains validation and cleanup logic that is ONLY used in CI environments.
These functions handle:
- Validating safetensors files for corruption
- Checking for missing shards in sharded models
- Cleaning up corrupted files (selective or full cache deletion)
- Automatic retry logic for corrupted downloads
- Validating config/tokenizer files completeness to enable offline mode

For regular users, weight_utils.py provides simple download functionality without
the overhead of validation and automatic cleanup. The CI-specific behavior is
gated by is_in_ci() checks in weight_utils.py.
"""  # CI专用的权重验证和缓存清理工具。此模块包含仅在CI环境中使用的验证和清理逻辑，包括：验证safetensors文件是否损坏、检查分片模型中缺失的分片、清理损坏的文件（选择性或完整缓存删除）、损坏下载的自动重试逻辑、验证config/tokenizer文件完整性以启用离线模式。对于普通用户，weight_utils.py提供简单的下载功能，无需验证和自动清理的开销。CI专用行为由weight_utils.py中的is_in_ci()检查控制。

import glob as glob_module  # 导入glob模块（重命名避免与变量名冲突）
import hashlib  # 导入哈希模块
import json  # 导入JSON模块
import logging  # 导入日志模块
import os  # 导入操作系统模块
import re  # 导入正则表达式模块
import shutil  # 导入文件操作模块
import tempfile  # 导入临时文件模块
import time  # 导入时间模块
from typing import List, Optional, Tuple  # 导入类型提示

import safetensors  # 导入safetensors库

from sglang.srt.utils import log_info_on_rank0  # 导入在rank 0上记录信息的工具函数

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

# Validation marker version - increment when validation logic changes  # 验证标记版本 - 当验证逻辑变更时递增
# v2: Added trust_remote_code module validation (modeling_*.py must exist in snapshot)  # v2：添加了trust_remote_code模块验证（快照中必须存在modeling_*.py）
# v3: Added remote file existence checks for hf_quant_config.json  # v3：添加了hf_quant_config.json的远程文件存在性检查
# v5: Invalidate all previous markers to force fresh validation  # v5：使所有之前的标记失效以强制重新验证
VALIDATION_MARKER_VERSION = "5"  # 当前验证标记版本号为5


def _remote_file_exists(  # 检查Hugging Face Hub上特定版本的文件是否存在
    repo_id: str, filename: str, revision: Optional[str], allow_remote_check: bool
) -> Optional[bool]:
    """
    Check if a file exists on Hugging Face Hub for a specific revision.

    Args:
        repo_id: Repository ID (e.g., "meta-llama/Llama-2-7b-hf")
        filename: File name to check (e.g., "hf_quant_config.json")
        revision: Git revision (commit hash, branch, or tag). None means default branch.
        allow_remote_check: Whether remote checks are allowed (e.g., CI validation phase)

    Returns:
        True if file exists on hub, False if it doesn't exist, None if we cannot determine
        (network error or remote check not allowed - be conservative and assume incomplete)
    """  # 检查Hugging Face Hub上特定版本的文件是否存在。返回True表示存在，False表示不存在，None表示无法确定（网络错误或不允许远程检查）
    if not allow_remote_check:  # 如果不允许远程检查
        logger.debug(
            "Remote check disabled for %s/%s, returning None (unknown)",
            repo_id,
            filename,
        )
        return None  # 返回None表示未知

    try:  # 尝试进行远程检查
        from huggingface_hub import HfApi  # 导入HuggingFace Hub API

        api = HfApi()  # 创建API实例
        exists = api.file_exists(repo_id=repo_id, filename=filename, revision=revision)  # 检查文件是否存在
        logger.debug(
            "Remote file check: %s/%s (revision=%s) exists=%s",
            repo_id,
            filename,
            revision or "default",
            exists,
        )
        return exists  # 返回检查结果
    except Exception as e:  # 处理异常
        # Network errors, auth issues, repo not found, etc.  # 网络错误、认证问题、仓库未找到等
        # Return None (unknown) - caller will treat as optional  # 返回None（未知）- 调用者将其视为可选
        logger.debug(
            "Failed to check remote file existence for %s/%s (revision=%s): %s. "
            "Will treat as optional.",
            repo_id,
            filename,
            revision or "default",
            e,
        )
        return None  # 返回None表示无法确定


def _get_validation_marker_path(snapshot_dir: str) -> Optional[str]:  # 获取快照目录的验证标记文件路径
    """
    Get the path to validation marker file for a snapshot.

    Marker is stored in /tmp to avoid permission issues with HF cache directory.
    Marker key is sha256(snapshot_dir) to avoid any collisions regardless of
    model_name_or_path format.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        Path to marker file or None if snapshot_dir is invalid
    """  # 获取快照的验证标记文件路径。标记存储在/tmp中以避免HF缓存目录的权限问题。标记键为sha256(snapshot_dir)以避免任何冲突
    if not snapshot_dir or not os.path.isdir(snapshot_dir):  # 如果快照目录无效
        return None

    # Normalize path to avoid marker misses due to trailing slashes or symlinks  # 规范化路径以避免因尾部斜杠或符号链接导致标记未命中
    # realpath resolves symlinks, rstrip removes trailing slashes  # realpath解析符号链接，rstrip移除尾部斜杠
    normalized_dir = os.path.realpath(snapshot_dir).rstrip("/")  # 规范化目录路径

    # Use sha256 of normalized snapshot_dir path as unique key  # 使用规范化快照目录路径的sha256作为唯一键
    # This avoids any collision issues with repo naming or snapshot hash reuse  # 这避免了仓库命名或快照哈希复用的冲突问题
    dir_hash = hashlib.sha256(normalized_dir.encode("utf-8")).hexdigest()[:12]  # 取sha256哈希的前12个字符

    # Store in /tmp with directory hash  # 存储在/tmp中，使用目录哈希作为标识
    return f"/tmp/sglang_hf_validation_{dir_hash}.json"  # 返回标记文件路径


def _get_per_run_marker_dir() -> str:  # 获取每次运行的验证标记目录
    """
    Get the directory for per-run validation markers.

    These markers are specific to the current CI run and are not shared across
    runners. They are stored in a temporary directory that is cleaned up after
    the run completes.

    Returns:
        Path to per-run marker directory
    """  # 获取每次运行的验证标记目录。这些标记特定于当前CI运行，不跨运行器共享
    # Prefer RUNNER_TEMP (GitHub Actions) or TMPDIR, fallback to /tmp  # 优先使用RUNNER_TEMP（GitHub Actions）或TMPDIR，回退到/tmp
    base_dir = os.environ.get("RUNNER_TEMP", os.environ.get("TMPDIR", "/tmp"))  # 获取基础目录
    marker_dir = os.path.join(base_dir, "sglang_ci_offline_markers")  # 标记目录路径
    os.makedirs(marker_dir, exist_ok=True)  # 创建目录（如不存在）
    return marker_dir  # 返回标记目录路径


def _get_per_run_marker_path(snapshot_dir: str) -> Optional[str]:  # 获取快照的每次运行验证标记文件路径
    """
    Get the path to per-run validation marker file for a snapshot.

    Per-run markers are specific to the current CI run and are not shared
    across runners. This prevents cross-runner cache state pollution.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        Path to per-run marker file or None if snapshot_dir is invalid
    """  # 获取快照的每次运行验证标记文件路径。每次运行的标记特定于当前CI运行，不跨运行器共享，防止跨运行器缓存状态污染
    if not snapshot_dir or not os.path.isdir(snapshot_dir):  # 如果快照目录无效
        return None

    normalized_dir = os.path.realpath(snapshot_dir).rstrip("/")  # 规范化目录路径
    dir_hash = hashlib.sha256(normalized_dir.encode("utf-8")).hexdigest()[:12]  # 计算目录哈希

    marker_dir = _get_per_run_marker_dir()  # 获取标记目录
    return os.path.join(marker_dir, f"{dir_hash}.json")  # 返回标记文件路径


def _read_per_run_marker(snapshot_dir: str) -> Optional[dict]:  # 读取快照的每次运行验证标记
    """
    Read per-run validation marker for a snapshot.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        Marker dict if exists and valid, None otherwise
    """  # 读取快照的每次运行验证标记。如果存在且有效返回标记字典，否则返回None
    marker_path = _get_per_run_marker_path(snapshot_dir)  # 获取标记文件路径
    if not marker_path or not os.path.exists(marker_path):  # 如果路径无效或文件不存在
        return None

    try:  # 尝试读取标记
        with open(marker_path, "r", encoding="utf-8") as f:  # 打开标记文件
            marker = json.load(f)  # 加载JSON内容

        # Validate marker structure  # 验证标记结构
        if not isinstance(marker, dict):  # 如果不是字典
            return None

        required_keys = ["timestamp", "model_id", "snapshot_hash", "validation_passed"]  # 必需的键
        if not all(k in marker for k in required_keys):  # 如果缺少必需的键
            return None

        if marker.get("validation_passed") is not True:  # 如果验证未通过
            return None

        return marker  # 返回有效标记

    except Exception as e:  # 处理异常
        logger.debug("Failed to read per-run marker from %s: %s", marker_path, e)
        return None


def _write_per_run_marker(  # 写入快照的每次运行验证标记
    snapshot_dir: str, model_id: str, required_files: Optional[list] = None
) -> None:
    """
    Write per-run validation marker for a snapshot.

    Args:
        snapshot_dir: Path to snapshot directory
        model_id: Model identifier
        required_files: List of required files that were validated
    """  # 写入快照的每次运行验证标记
    marker_path = _get_per_run_marker_path(snapshot_dir)  # 获取标记文件路径
    if not marker_path:  # 如果路径无效
        logger.debug("Cannot write per-run marker: invalid snapshot_dir")  # 记录调试信息
        return

    from datetime import datetime  # 导入日期时间模块

    snapshot_hash = os.path.basename(snapshot_dir)  # 获取快照哈希值

    marker = {  # 构建标记字典
        "timestamp": datetime.utcnow().isoformat() + "Z",  # 时间戳
        "model_id": model_id,  # 模型ID
        "snapshot_hash": snapshot_hash,  # 快照哈希
        "validation_passed": True,  # 验证通过
        "required_files": required_files or [],  # 必需文件列表
    }

    try:  # 尝试写入标记
        marker_dir = os.path.dirname(marker_path)  # 获取标记目录
        os.makedirs(marker_dir, exist_ok=True)  # 创建目录

        with tempfile.NamedTemporaryFile(  # 创建临时文件（原子写入）
            mode="w",
            encoding="utf-8",
            dir=marker_dir,
            delete=False,
            suffix=".tmp",
        ) as f:
            temp_path = f.name  # 临时文件路径
            json.dump(marker, f, indent=2)  # 写入JSON内容

        os.replace(temp_path, marker_path)  # 原子替换为目标文件
        logger.debug("Wrote per-run marker to %s", marker_path)  # 记录调试信息
    except Exception as e:  # 处理异常
        logger.warning("Failed to write per-run marker to %s: %s", marker_path, e)
        try:  # 尝试清理临时文件
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _remove_per_run_marker(snapshot_dir: str) -> None:  # 删除快照的每次运行验证标记
    """
    Remove per-run validation marker for a snapshot.

    Args:
        snapshot_dir: Path to snapshot directory
    """  # 删除快照的每次运行验证标记
    marker_path = _get_per_run_marker_path(snapshot_dir)  # 获取标记文件路径
    if marker_path and os.path.exists(marker_path):  # 如果文件存在
        try:  # 尝试删除
            os.remove(marker_path)  # 删除标记文件
            logger.debug("Removed per-run marker: %s", marker_path)  # 记录调试信息
        except Exception as e:  # 处理异常
            logger.warning("Failed to remove per-run marker %s: %s", marker_path, e)


def _read_validation_marker(snapshot_dir: str) -> Optional[dict]:  # 读取快照的验证标记
    """
    Read validation marker for a snapshot.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        Marker dict with keys: version, validated_at, validation_passed
        None if marker doesn't exist or is invalid or validation_passed is not True
    """  # 读取快照的验证标记。返回包含version、validated_at、validation_passed键的标记字典，如果标记不存在、无效或validation_passed不为True则返回None
    marker_path = _get_validation_marker_path(snapshot_dir)  # 获取标记文件路径
    if not marker_path:  # 如果路径无效
        return None

    if not os.path.exists(marker_path):  # 如果文件不存在
        return None

    try:  # 尝试读取标记
        with open(marker_path, "r", encoding="utf-8") as f:  # 打开标记文件
            marker = json.load(f)  # 加载JSON内容

        # Validate marker structure  # 验证标记结构
        if not isinstance(marker, dict):  # 如果不是字典
            return None

        required_keys = ["version", "validated_at", "validation_passed"]  # 必需的键
        if not all(key in marker for key in required_keys):  # 如果缺少必需的键
            return None

        # Check version match  # 检查版本匹配
        if marker["version"] != VALIDATION_MARKER_VERSION:  # 如果版本不匹配
            logger.debug(
                "Validation marker version mismatch: %s != %s, will re-validate",
                marker["version"],
                VALIDATION_MARKER_VERSION,
            )
            return None

        # Explicitly check validation_passed is True (defensive check)  # 显式检查validation_passed为True（防御性检查）
        # Even though we only write markers on success, this guards against  # 即使我们只在成功时写入标记，这也可以防御
        # manual edits or future code changes  # 手动编辑或未来的代码变更
        if marker.get("validation_passed") is not True:  # 如果验证未通过
            logger.debug(
                "Validation marker has validation_passed=%s, treating as invalid",
                marker.get("validation_passed"),
            )
            return None

        return marker  # 返回有效标记
    except (json.JSONDecodeError, OSError) as e:  # 处理JSON解析或IO错误
        logger.debug("Failed to read validation marker at %s: %s", marker_path, e)
        return None


def _write_validation_marker(snapshot_dir: str, passed: bool) -> None:  # 写入快照的验证标记（原子写入）
    """
    Write validation marker for a snapshot (atomic write).

    IMPORTANT: We only cache successful validations. Failed validations are NOT
    cached to allow retry after files are downloaded.

    Args:
        snapshot_dir: Path to snapshot directory
        passed: Whether validation passed
    """  # 写入快照的验证标记（原子写入）。重要：只缓存成功的验证，失败的验证不缓存以允许文件下载后重试
    if not passed:  # 如果验证未通过
        # Don't cache failures - allow retry on next launch  # 不缓存失败 - 允许下次启动时重试
        return

    marker_path = _get_validation_marker_path(snapshot_dir)  # 获取标记文件路径
    if not marker_path:  # 如果路径无效
        logger.debug("Cannot write marker: invalid snapshot_dir")  # 记录调试信息
        return

    from datetime import datetime  # 导入日期时间模块

    marker = {  # 构建标记字典
        "version": VALIDATION_MARKER_VERSION,  # 版本号
        "validated_at": datetime.utcnow().isoformat() + "Z",  # 验证时间
        "validation_passed": passed,  # 验证是否通过
    }

    try:  # 尝试写入标记
        # Atomic write: write to temp file then os.replace  # 原子写入：先写入临时文件然后os.replace
        marker_dir = os.path.dirname(marker_path)  # 获取标记目录
        os.makedirs(marker_dir, exist_ok=True)  # 创建目录

        with tempfile.NamedTemporaryFile(  # 创建临时文件
            mode="w",
            encoding="utf-8",
            dir=marker_dir,
            delete=False,
            suffix=".tmp",
        ) as f:
            temp_path = f.name  # 临时文件路径
            json.dump(marker, f, indent=2)  # 写入JSON内容

        # Atomic replace (overwrites existing file if any)  # 原子替换（如果存在则覆盖）
        os.replace(temp_path, marker_path)  # 原子替换文件
        logger.debug("Wrote validation marker to %s (passed=%s)", marker_path, passed)
    except Exception as e:  # 处理异常
        logger.warning("Failed to write validation marker to %s: %s", marker_path, e)
        # Clean up temp file if it exists  # 清理临时文件（如果存在）
        try:
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _validate_json_file(file_path: str, file_name: str) -> bool:  # 验证JSON文件是否存在、非空且可解析
    """
    Validate that a JSON file exists, is non-empty, and can be parsed.

    Args:
        file_path: Path to the JSON file
        file_name: Name of the file (for logging)

    Returns:
        True if the file is valid, False otherwise
    """  # 验证JSON文件是否存在、非空且可解析
    if not os.path.exists(file_path):  # 如果文件不存在
        logger.debug("CI cache validation: %s not found at %s", file_name, file_path)
        return False

    if not os.path.isfile(file_path):  # 如果不是文件
        logger.warning(
            "CI cache validation: %s is not a file: %s", file_name, file_path
        )
        return False

    # Check if file is non-empty  # 检查文件是否非空
    try:
        file_size = os.path.getsize(file_path)  # 获取文件大小
        if file_size == 0:  # 如果文件为空
            logger.warning("CI cache validation: %s is empty: %s", file_name, file_path)
            return False
    except OSError as e:  # 处理IO错误
        logger.warning("CI cache validation: Cannot get size of %s: %s", file_name, e)
        return False

    # Try to parse JSON  # 尝试解析JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:  # 打开文件
            json.load(f)  # 尝试解析JSON
        return True  # 解析成功
    except json.JSONDecodeError as e:  # 处理JSON解析错误
        logger.warning(
            "CI cache validation: %s is not valid JSON: %s - %s",
            file_name,
            file_path,
            e,
        )
        return False
    except Exception as e:  # 处理其他异常
        logger.warning(
            "CI cache validation: Failed to read %s: %s - %s",
            file_name,
            file_path,
            e,
        )
        return False


def _validate_config_and_tokenizer_files(  # 验证关键配置和分词器文件是否存在且有效
    snapshot_dir: str,
    model_id: Optional[str] = None,
    revision: Optional[str] = None,
    allow_remote_check: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate that critical config and tokenizer files exist and are valid.

    This checks for:
    - config.json (required)
    - tokenizer_config.json (required)
    - generation_config.json (optional but validated if present)
    - hf_quant_config.json (conditionally required based on Hub) - for FP4/FP8/ModelOpt
    - quantize_config.json / quant_config.json (optional but validated if present) - for AWQ/GPTQ
    - params.json (optional but validated if present) - for Mistral native format
    - preprocessor_config.json (optional but validated if present) - for vision models
    - trust_remote_code dynamic modules (required if auto_map present in config.json)
    - At least one tokenizer file: tokenizer.json, tokenizer.model, or tiktoken.model

    Args:
        snapshot_dir: Path to the model snapshot directory
        model_id: Model repository ID (e.g., "meta-llama/Llama-2-7b-hf"), used for remote checks
        revision: Git revision (commit hash), used for remote checks
        allow_remote_check: Whether to check Hub for file existence to determine requirements

    Returns:
        Tuple of (is_valid, missing_files)
        - is_valid: True if all required files are present and valid
        - missing_files: List of missing or invalid file names
    """  # 验证关键配置和分词器文件是否存在且有效。检查config.json、tokenizer_config.json、generation_config.json、hf_quant_config.json、量化配置、params.json、preprocessor_config.json、trust_remote_code动态模块以及至少一个分词器文件
    missing_files = []  # 缺失文件列表

    # Check required config files  # 检查必需的配置文件
    required_files = [  # 必需文件列表
        "config.json",  # 模型配置文件
        "tokenizer_config.json",  # 分词器配置文件
    ]

    for file_name in required_files:  # 遍历必需文件
        file_path = os.path.join(snapshot_dir, file_name)  # 构建文件路径
        if not _validate_json_file(file_path, file_name):  # 验证文件
            missing_files.append(file_name)  # 添加到缺失列表

    # Check optional generation_config.json (validate if exists)  # 检查可选的generation_config.json（如果存在则验证）
    generation_config_path = os.path.join(snapshot_dir, "generation_config.json")  # 生成配置路径
    if os.path.exists(generation_config_path):  # 如果文件存在
        if not _validate_json_file(generation_config_path, "generation_config.json"):  # 验证文件
            missing_files.append("generation_config.json (exists but invalid)")  # 添加到缺失列表

    # Check hf_quant_config.json with remote existence check  # 检查hf_quant_config.json并结合远程存在性检查
    # This file is needed for quantized models (FP4/FP8/ModelOpt)  # 此文件用于量化模型（FP4/FP8/ModelOpt）
    # Example: nvidia/Llama-3.1-8B-Instruct-FP8, nvidia/DeepSeek-V3-0324-FP4  # 示例：nvidia/Llama-3.1-8B-Instruct-FP8等
    hf_quant_config_path = os.path.join(snapshot_dir, "hf_quant_config.json")  # HF量化配置路径
    local_hf_quant_exists = os.path.exists(hf_quant_config_path)  # 检查本地是否存在

    # Check if file exists on Hub for this revision  # 检查Hub上此版本的文件是否存在
    # Only do remote check if model_id looks like a HF repo_id (org/model format)  # 仅当model_id看起来像HF仓库ID（org/model格式）时才进行远程检查
    # Skip if it's a local path (absolute path or doesn't contain '/')  # 如果是本地路径（绝对路径或不包含'/'）则跳过
    remote_hf_quant_exists = None  # 远程存在性结果，初始为None
    is_hf_repo = (  # 判断是否为HF仓库
        model_id is not None
        and "/" in model_id
        and not os.path.isabs(model_id)
        and not model_id.startswith("/")
    )
    if is_hf_repo and allow_remote_check:  # 如果是HF仓库且允许远程检查
        remote_hf_quant_exists = _remote_file_exists(  # 检查远程文件是否存在
            repo_id=model_id,
            filename="hf_quant_config.json",
            revision=revision,
            allow_remote_check=allow_remote_check,
        )

    # Apply conditional requirement logic  # 应用条件性需求逻辑
    if remote_hf_quant_exists is True:  # 如果Hub上存在此文件
        # Hub has this file for this revision - it's REQUIRED  # Hub上此版本有此文件 - 这是必需的
        if not local_hf_quant_exists:  # 如果本地不存在
            missing_files.append(
                f"hf_quant_config.json (required: exists on Hub for revision {revision or 'default'} but missing locally)"
            )  # 添加到缺失列表
            log_info_on_rank0(
                logger,
                f"Hub has hf_quant_config.json for {model_id} revision {revision or 'default'} "
                f"but local snapshot missing it. Cache incomplete, will not write marker.",
            )
        elif not _validate_json_file(hf_quant_config_path, "hf_quant_config.json"):  # 如果本地存在但无效
            missing_files.append("hf_quant_config.json (exists but invalid)")  # 添加到缺失列表
    elif remote_hf_quant_exists is False:  # 如果Hub上不存在此文件
        # Hub doesn't have this file - it's OPTIONAL  # Hub上没有此文件 - 这是可选的
        # Only validate if it happens to exist locally  # 仅在本地恰好存在时验证
        if local_hf_quant_exists:  # 如果本地存在
            if not _validate_json_file(hf_quant_config_path, "hf_quant_config.json"):
                missing_files.append("hf_quant_config.json (exists but invalid)")  # 添加到缺失列表
    else:  # remote_hf_quant_exists为None - 未知（网络错误或远程检查被禁用）
        # remote_hf_quant_exists is None - unknown (network error or remote check disabled)  # 远程存在性未知 - 网络错误或远程检查被禁用
        # Treat as OPTIONAL - only enforce when we can positively confirm Hub has it  # 视为可选 - 仅在能确认Hub有时才强制要求
        if local_hf_quant_exists:  # 如果本地存在
            # Local file exists - validate it  # 本地文件存在 - 验证它
            if not _validate_json_file(hf_quant_config_path, "hf_quant_config.json"):
                missing_files.append("hf_quant_config.json (exists but invalid)")  # 添加到缺失列表
        # If local file missing and remote unknown, just log it - don't block marker  # 如果本地文件缺失且远程状态未知，仅记录日志 - 不阻止标记
        logger.debug(
            "Cannot verify hf_quant_config.json on Hub for %s (revision=%s), "
            "treating as optional since remote status unknown",
            model_id or "unknown",
            revision or "default",
        )

    # Check optional quantize_config.json / quant_config.json (validate if exists)  # 检查可选的量化配置文件（如果存在则验证）
    # These files are needed for AWQ/GPTQ/AutoRound quantized models  # 这些文件用于AWQ/GPTQ/AutoRound量化模型
    # Example: TheBloke/Llama-2-7B-AWQ, casperhansen/vicuna-7b-v1.5-awq  # 示例模型
    for quant_config_name in ["quantize_config.json", "quant_config.json"]:  # 遍历两种量化配置文件名
        quant_config_path = os.path.join(snapshot_dir, quant_config_name)  # 构建路径
        if os.path.exists(quant_config_path):  # 如果存在
            if not _validate_json_file(quant_config_path, quant_config_name):  # 验证文件
                missing_files.append(f"{quant_config_name} (exists but invalid)")  # 添加到缺失列表
            break  # Only need to check one of these  # 只需检查其中一个

    # Check optional params.json (validate if exists)  # 检查可选的params.json（如果存在则验证）
    # This file is needed for Mistral native format models  # 此文件用于Mistral原生格式模型
    # Example: mistralai/Mistral-7B-v0.1  # 示例模型
    params_json_path = os.path.join(snapshot_dir, "params.json")  # 构建路径
    if os.path.exists(params_json_path):  # 如果存在
        if not _validate_json_file(params_json_path, "params.json"):  # 验证文件
            missing_files.append("params.json (exists but invalid)")  # 添加到缺失列表

    # Check optional preprocessor_config.json (validate if exists)  # 检查可选的preprocessor_config.json（如果存在则验证）
    # This file is needed for vision/multimodal models  # 此文件用于视觉/多模态模型
    # Example: llava-hf/llava-1.5-7b-hf, Qwen/Qwen2-VL-7B-Instruct  # 示例模型
    preprocessor_config_path = os.path.join(snapshot_dir, "preprocessor_config.json")  # 构建路径
    if os.path.exists(preprocessor_config_path):  # 如果存在
        if not _validate_json_file(
            preprocessor_config_path, "preprocessor_config.json"
        ):  # 验证文件
            missing_files.append("preprocessor_config.json (exists but invalid)")  # 添加到缺失列表

    # Check for trust_remote_code dynamic module files if needed  # 如果需要，检查trust_remote_code动态模块文件
    # When auto_map exists in config.json, the model requires custom Python files  # 当config.json中存在auto_map时，模型需要自定义Python文件
    # These files must be present for offline mode to work  # 这些文件必须存在才能在离线模式下工作
    config_path = os.path.join(snapshot_dir, "config.json")  # 配置文件路径
    if os.path.exists(config_path):  # 如果配置文件存在
        try:  # 尝试读取和解析
            with open(config_path, "r", encoding="utf-8") as f:  # 打开配置文件
                config = json.load(f)  # 加载JSON

            auto_map = config.get("auto_map", {})  # 获取auto_map
            if auto_map and isinstance(auto_map, dict):  # 如果存在且为字典
                # Extract Python module files from auto_map  # 从auto_map中提取Python模块文件
                # auto_map format: {"AutoConfig": "configuration_xxx.ConfigClass", ...}  # auto_map格式
                # We need to check if the .py files exist  # 需要检查.py文件是否存在
                custom_files = set()  # 自定义文件集合
                for key, value in auto_map.items():  # 遍历auto_map
                    if isinstance(value, str) and "." in value:  # 如果值是字符串且包含点
                        # Extract module name (e.g., "configuration_xxx" from "configuration_xxx.ConfigClass")  # 提取模块名（如从"configuration_xxx.ConfigClass"中提取"configuration_xxx"）
                        module_name = value.split(".")[0]  # 获取模块名
                        custom_files.add(f"{module_name}.py")  # 添加.py文件名

                # Check if all custom files exist in snapshot directory  # 检查快照目录中是否存在所有自定义文件
                # NOTE: Some models (like nvidia/DeepSeek-V3-0324-FP4) have auto_map  # 注意：某些模型（如nvidia/DeepSeek-V3-0324-FP4）有auto_map
                # but don't include modeling_*.py in their repo, relying on transformers  # 但仓库中不包含modeling_*.py，依赖transformers
                # to fetch it from the base model. We MUST mark these as missing to  # 从基础模型获取。我们必须将这些标记为缺失
                # prevent offline mode, which would fail to load the dynamic modules.  # 以防止离线模式加载动态模块失败
                for custom_file in custom_files:  # 遍历自定义文件
                    custom_file_path = os.path.join(snapshot_dir, custom_file)  # 构建路径
                    if not os.path.exists(custom_file_path):  # 如果文件不存在
                        missing_files.append(
                            f"{custom_file} (required for trust_remote_code)"
                        )  # 添加到缺失列表
                        logger.debug(
                            f"Custom module file not in snapshot: {custom_file} for {snapshot_dir}"
                        )
                    elif not os.path.isfile(custom_file_path):  # 如果路径不是文件
                        missing_files.append(f"{custom_file} (exists but not a file)")  # 添加到缺失列表
        except (json.JSONDecodeError, OSError, KeyError) as e:  # 处理异常
            # If we can't read config.json, it will be caught by earlier validation  # 如果无法读取config.json，将由先前的验证捕获
            logger.debug("Failed to check auto_map in config.json: %s", e)

    # Check for at least one tokenizer file  # 检查至少存在一个分词器文件
    tokenizer_files = [  # 分词器文件列表
        "tokenizer.json",  # JSON格式分词器
        "tokenizer.model",  # SentencePiece格式分词器
        "tiktoken.model",  # TikToken格式分词器
    ]

    tokenizer_found = False  # 是否找到分词器文件
    for tokenizer_file in tokenizer_files:  # 遍历分词器文件
        tokenizer_path = os.path.join(snapshot_dir, tokenizer_file)  # 构建路径
        if os.path.exists(tokenizer_path) and os.path.isfile(tokenizer_path):  # 如果文件存在且是文件
            # For tokenizer.json, validate it's proper JSON  # 对于tokenizer.json，验证它是有效的JSON
            if tokenizer_file == "tokenizer.json":  # 如果是JSON格式
                if _validate_json_file(tokenizer_path, tokenizer_file):  # 验证文件
                    tokenizer_found = True  # 标记为已找到
                    break
            else:  # 对于.model文件
                # For .model files, just check they're non-empty  # 对于.model文件，只检查是否非空
                try:
                    if os.path.getsize(tokenizer_path) > 0:  # 如果文件非空
                        tokenizer_found = True  # 标记为已找到
                        break
                except OSError:
                    pass

    if not tokenizer_found:  # 如果未找到分词器文件
        missing_files.append("tokenizer file")  # 添加到缺失列表

    is_valid = len(missing_files) == 0  # 判断是否所有必需文件都存在
    return is_valid, missing_files  # 返回验证结果和缺失文件列表


def ci_validate_cache_and_enable_offline_if_complete(  # CI专用：验证本地缓存完整性并判断是否可以安全启用离线模式
    snapshot_dir: str,
    weight_files: List[str],
    model_name_or_path: str,
) -> bool:
    """
    Validate local cache completeness (config/tokenizer/weights) and determine
    if offline mode can be safely enabled.

    This function uses a snapshot-level marker to cache validation results,
    so the heavy validation is done at most once per snapshot per runner.

    This function checks:
    1. Validation marker (if exists and version matches, skip re-validation)
    2. Config and tokenizer files (config.json, tokenizer_config.json, etc.)
    3. Weight files (safetensors shards, index files, corruption check)

    If all are present and valid, it returns True to signal that offline
    mode can be safely enabled.

    IMPORTANT: This should be called BEFORE any HF operations, and if it
    returns True, the caller should set HF_HUB_OFFLINE=1 for the server
    subprocess env ONLY (not global environment).

    Args:
        snapshot_dir: Path to the model snapshot directory
        weight_files: List of weight file paths to validate (must be non-empty)
        model_name_or_path: Model identifier for logging

    Returns:
        True if cache is complete and offline mode can be enabled, False otherwise
    """  # 验证本地缓存完整性（配置/分词器/权重）并判断是否可以安全启用离线模式。使用快照级标记缓存验证结果，每个快照每个运行器最多执行一次重量级验证
    # Guard: weight_files is required  # 守卫条件：weight_files是必需的
    if not weight_files:  # 如果没有权重文件
        log_info_on_rank0(
            logger,
            f"CI_OFFLINE: No weight files provided, skip offline, keep online allowed - {model_name_or_path}",
        )
        return False

    # Fast-path: Check if validation marker exists and is valid  # 快速路径：检查验证标记是否存在且有效
    # We only cache successful validations, so if marker exists, it means cache is complete  # 我们只缓存成功的验证，所以如果标记存在，意味着缓存完整
    marker = _read_validation_marker(snapshot_dir)  # 读取验证标记
    if marker is not None:  # 如果标记存在
        marker_path = _get_validation_marker_path(snapshot_dir)  # 获取标记路径
        marker_name = os.path.basename(marker_path) if marker_path else "unknown"  # 获取标记文件名
        log_info_on_rank0(
            logger,
            f"CI_OFFLINE: Marker hit (marker={marker_name}), skip re-validation, offline mode will be enabled - {model_name_or_path}",
        )
        return True  # 缓存完整，可以启用离线模式

    # No marker - perform full validation  # 没有标记 - 执行完整验证
    # (Failures are not cached, so we'll retry validation each time until success)  # （失败不缓存，所以每次都会重试验证直到成功）

    # Extract revision (snapshot hash) from snapshot_dir path  # 从snapshot_dir路径中提取版本（快照哈希）
    # snapshot_dir format: /path/to/cache/models--org--model/snapshots/<commit_hash>  # snapshot_dir格式
    revision = os.path.basename(snapshot_dir)  # 获取版本号

    # Only allow remote checks if we're not in offline mode  # 仅在非离线模式下允许远程检查
    # This avoids unnecessary API calls and warnings in offline CI environments  # 避免在离线CI环境中进行不必要的API调用和警告
    import huggingface_hub.constants  # 导入HuggingFace Hub常量

    allow_remote_check = not huggingface_hub.constants.HF_HUB_OFFLINE  # 判断是否允许远程检查

    log_info_on_rank0(
        logger,
        f"CI_OFFLINE: No marker found, performing full validation "
        f"(snapshot={revision}, allow_remote_check={allow_remote_check}) - {model_name_or_path}",
    )

    # Validate config and tokenizer files with remote existence checks  # 验证配置和分词器文件，结合远程存在性检查
    config_valid, missing_config_files = _validate_config_and_tokenizer_files(
        snapshot_dir=snapshot_dir,
        model_id=model_name_or_path,
        revision=revision,
        allow_remote_check=allow_remote_check,
    )

    if not config_valid:  # 如果配置验证失败
        log_info_on_rank0(
            logger,
            f"CI_OFFLINE: Missing config/tokenizer files {missing_config_files}, skip offline, keep online allowed - {model_name_or_path}",
        )
        # Don't write marker for failures - allow retry after download  # 不为失败写入标记 - 允许下载后重试
        return False

    # Validate weight files using existing validation from PR #15216  # 使用PR #15216中的现有验证来验证权重文件
    # This checks for missing shards, corrupted safetensors, etc.  # 检查缺失的分片、损坏的safetensors等
    weights_valid, error_msg, _ = _validate_sharded_model(snapshot_dir, weight_files)  # 验证分片模型
    if not weights_valid:  # 如果权重验证失败
        log_info_on_rank0(
            logger,
            f"CI_OFFLINE: Weight validation failed ({error_msg}), skip offline, keep online allowed - {model_name_or_path}",
        )
        # Don't write marker for failures - allow retry after download  # 不为失败写入标记 - 允许下载后重试
        return False

    log_info_on_rank0(
        logger,
        f"CI_OFFLINE: Cache validation PASSED, offline mode will be enabled - {model_name_or_path}",
    )

    # Write marker with passed=True for future reuse  # 写入passed=True的标记以供未来复用
    # (Failures are not cached, so this only happens on success)  # （失败不缓存，所以仅在成功时发生）
    _write_validation_marker(snapshot_dir, passed=True)  # 写入验证标记
    return True  # 返回验证通过


def _infer_component_type(component_name: str, component_info: list) -> str:  # 从组件名称和信息推断组件类型
    """
    Infer component type from component name and info.

    Args:
        component_name: Name of the component (e.g., "scheduler", "tokenizer")
        component_info: Component info from model_index.json (e.g., ["diffusers", "SchedulerClass"])

    Returns:
        Component type string for validation rules
    """  # 从组件名称和信息推断组件类型，返回用于验证规则的组件类型字符串
    # Normalize component name for type detection  # 规范化组件名称以进行类型检测
    name_lower = component_name.lower()  # 转为小写

    # Infer type based on name  # 根据名称推断类型
    if "scheduler" in name_lower:  # 如果名称包含scheduler
        return "scheduler"
    elif "tokenizer" in name_lower:  # 如果名称包含tokenizer
        return "tokenizer"
    elif "image_processor" in name_lower:  # 如果名称包含image_processor
        return "image_processor"
    elif "feature_extractor" in name_lower:  # 如果名称包含feature_extractor
        return "feature_extractor"
    elif "processor" in name_lower:  # 如果名称包含processor
        return "processor"
    else:  # 默认
        # Default to model component (needs config.json + weights)  # 默认为模型组件（需要config.json + 权重）
        return "model"


def _check_component_config(  # 检查组件是否具有基于类型的必需配置文件
    component_dir: str, component_type: str
) -> Tuple[bool, List[str]]:
    """
    Check if component has required config files based on type.

    Args:
        component_dir: Path to component directory
        component_type: Type of component (scheduler, tokenizer, processor, model, etc.)

    Returns:
        Tuple of (has_valid_config, list_of_candidates_tried)
    """  # 根据类型检查组件是否具有必需的配置文件，返回(是否有有效配置, 尝试的候选文件列表)
    if component_type == "scheduler":  # 如果是调度器
        # Scheduler: scheduler_config.json or config.json  # 调度器：scheduler_config.json或config.json
        candidates = ["scheduler_config.json", "config.json"]  # 候选文件列表
        for candidate in candidates:  # 遍历候选文件
            candidate_path = os.path.join(component_dir, candidate)  # 构建路径
            if _validate_json_file(candidate_path, candidate):  # 验证文件
                return True, candidates  # 返回有效
        return False, candidates  # 返回无效

    elif component_type == "tokenizer":  # 如果是分词器
        # Tokenizer must have actual tokenizer files (not just tokenizer_config.json)  # 分词器必须有实际的分词器文件（不仅是tokenizer_config.json）
        # Valid combinations:  # 有效的组合：
        # - tokenizer.json  # - tokenizer.json
        # - tokenizer.model  # - tokenizer.model
        # - vocab.json + merges.txt  # - vocab.json + merges.txt
        candidates = [  # 候选文件列表
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json+merges.txt",
        ]

        # Check tokenizer.json (validate as JSON)  # 检查tokenizer.json（作为JSON验证）
        tokenizer_json_path = os.path.join(component_dir, "tokenizer.json")  # 构建路径
        if _validate_json_file(tokenizer_json_path, "tokenizer.json"):  # 验证文件
            return True, candidates  # 返回有效

        # Check tokenizer.model (non-empty file)  # 检查tokenizer.model（非空文件）
        tokenizer_model_path = os.path.join(component_dir, "tokenizer.model")  # 构建路径
        if os.path.exists(tokenizer_model_path) and os.path.isfile(
            tokenizer_model_path
        ):  # 如果文件存在
            try:
                if os.path.getsize(tokenizer_model_path) > 0:  # 如果文件非空
                    return True, candidates  # 返回有效
            except OSError:
                pass

        # Check vocab.json + merges.txt pair  # 检查vocab.json + merges.txt对
        vocab_path = os.path.join(component_dir, "vocab.json")  # vocab.json路径
        merges_path = os.path.join(component_dir, "merges.txt")  # merges.txt路径
        if _validate_json_file(vocab_path, "vocab.json") and os.path.exists(
            merges_path
        ):  # 如果两者都存在
            return True, candidates  # 返回有效

        return False, candidates  # 返回无效

    elif component_type in ["processor", "feature_extractor", "image_processor"]:  # 如果是处理器/特征提取器/图像处理器
        # Processor/feature_extractor/image_processor: preprocessor_config.json or config.json  # 处理器/特征提取器/图像处理器：preprocessor_config.json或config.json
        candidates = ["preprocessor_config.json", "config.json"]  # 候选文件列表
        for candidate in candidates:  # 遍历候选文件
            candidate_path = os.path.join(component_dir, candidate)  # 构建路径
            if _validate_json_file(candidate_path, candidate):  # 验证文件
                return True, candidates  # 返回有效
        return False, candidates  # 返回无效

    else:  # 默认模型组件
        # Default model components: config.json  # 默认模型组件：config.json
        candidates = ["config.json"]  # 候选文件列表
        config_path = os.path.join(component_dir, "config.json")  # 构建路径
        if _validate_json_file(config_path, "config.json"):  # 验证文件
            return True, candidates  # 返回有效
        return False, candidates  # 返回无效


def _check_component_weights(component_dir: str) -> bool:  # 检查组件目录是否有权重文件
    """
    Check if component directory has weight files.

    Args:
        component_dir: Path to component directory

    Returns:
        True if weight files found, False otherwise
    """  # 检查组件目录是否包含权重文件
    weight_patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]  # 权重文件匹配模式

    for pattern in weight_patterns:  # 遍历匹配模式
        weight_files = glob_module.glob(os.path.join(component_dir, pattern))  # 匹配文件
        if weight_files:  # 如果找到文件
            return True

    return False  # 未找到权重文件


def _format_component_list(components: List[str], max_show: int = 5) -> str:  # 格式化组件列表，支持截断显示
    """
    Format component list with truncation.

    Args:
        components: List of component names
        max_show: Maximum number to show before truncating

    Returns:
        Formatted string like "comp1, comp2, comp3" or "comp1, comp2, +3 more"
    """  # 格式化组件列表并支持截断，返回如"comp1, comp2, comp3"或"comp1, comp2, +3 more"的字符串
    if len(components) <= max_show:  # 如果组件数不超过最大显示数
        return ", ".join(components)  # 直接拼接
    else:  # 超过则截断
        shown = components[:max_show]  # 显示前max_show个
        remaining = len(components) - max_show  # 剩余数量
        return f"{', '.join(shown)}, +{remaining} more"  # 返回截断格式


def _validate_diffusion_model(  # 验证扩散模型（diffusers管道）缓存完整性
    snapshot_dir: str,
) -> Tuple[bool, Optional[str]]:
    """
    Validate diffusion model (diffusers pipeline) cache completeness.

    This validation is based on model_index.json as the single source of truth.
    Error reporting uses coarse-grained error codes unless verbose mode is enabled.

    Error codes:
    - DIFFUSERS_INVALID_INDEX: model_index.json missing or corrupted
    - DIFFUSERS_INVALID_COMPONENTS: model_index.json has no valid components
    - DIFFUSERS_MISSING_COMPONENT: component directory or config missing
    - DIFFUSERS_MISSING_WEIGHTS: component weights missing

    Args:
        snapshot_dir: Path to the model snapshot directory

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if validation passed
        - (False, error_code_with_components) if validation failed
    """  # 验证扩散模型（diffusers管道）缓存完整性，基于model_index.json作为唯一真相来源
    # Check verbose mode from environment  # 从环境变量检查详细模式
    verbose = os.environ.get("SGLANG_CI_VALIDATE_VERBOSE") == "1"  # 是否启用详细模式

    # 1. Check for model_index.json (required for diffusers models)  # 1. 检查model_index.json（diffusers模型必需）
    model_index_path = os.path.join(snapshot_dir, "model_index.json")  # 构建路径
    if not os.path.exists(model_index_path):  # 如果不存在
        return False, "DIFFUSERS_INVALID_INDEX: model_index.json not found"  # 返回无效

    # Parse model_index.json  # 解析model_index.json
    try:
        with open(model_index_path, "r", encoding="utf-8") as f:  # 打开文件
            model_index = json.load(f)  # 加载JSON
    except (json.JSONDecodeError, OSError) as e:  # 处理解析错误
        if verbose:  # 如果详细模式
            return False, f"DIFFUSERS_INVALID_INDEX: model_index.json parse error - {e}"
        return False, "DIFFUSERS_INVALID_INDEX: model_index.json corrupted"  # 返回无效

    # 2. Extract components (non-underscore keys with list values)  # 2. 提取组件（非下划线开头且值为列表的键）
    components = {
        k: v
        for k, v in model_index.items()
        if not k.startswith("_") and isinstance(v, list)
    }

    if not components:  # 如果没有有效组件
        return False, "DIFFUSERS_INVALID_COMPONENTS: no valid components defined"  # 返回无效

    # Categorize errors by type  # 按类型分类错误
    missing_dirs = []  # 缺失目录列表
    missing_configs = []  # 缺失配置列表
    missing_configs_verbose = []  # 缺失配置详细列表
    missing_weights = []  # 缺失权重列表

    # 3. Validate each component  # 3. 验证每个组件
    for component_name, component_info in components.items():  # 遍历组件
        component_dir = os.path.join(snapshot_dir, component_name)  # 构建组件目录路径

        # Component directory must exist  # 组件目录必须存在
        if not os.path.isdir(component_dir):  # 如果目录不存在
            missing_dirs.append(component_name)  # 添加到缺失目录列表
            continue

        # Infer component type for validation rules  # 推断组件类型以确定验证规则
        component_type = _infer_component_type(component_name, component_info)  # 推断类型

        # Check for required config files based on component type  # 根据组件类型检查必需的配置文件
        has_valid_config, config_candidates = _check_component_config(
            component_dir, component_type
        )

        if not has_valid_config:  # 如果没有有效配置
            missing_configs.append(component_name)  # 添加到缺失配置列表
            if verbose:  # 如果详细模式
                candidates_str = ", ".join(config_candidates)  # 拼接候选文件名
                missing_configs_verbose.append(
                    f"{component_name} (tried: {candidates_str})"
                )
            continue

        # 4. Check for weights if component needs them  # 4. 如果组件需要权重，检查权重文件
        # These components don't require weight files (config-only)  # 以下组件不需要权重文件（仅配置）
        needs_weights = component_type not in [  # 判断是否需要权重
            "scheduler",
            "tokenizer",
            "processor",
            "feature_extractor",
            "image_processor",
        ]

        if needs_weights:  # 如果需要权重
            has_weights = _check_component_weights(component_dir)  # 检查权重文件
            if not has_weights:  # 如果没有权重
                missing_weights.append(component_name)  # 添加到缺失权重列表

    # 5. Build error message based on categorized errors  # 5. 根据分类的错误构建错误消息
    if missing_dirs or missing_configs or missing_weights:  # 如果有任何缺失
        errors = []  # 错误列表

        if missing_dirs:  # 如果有缺失目录
            dir_str = _format_component_list(missing_dirs)  # 格式化目录列表
            if verbose:  # 如果详细模式
                errors.append(f"DIFFUSERS_MISSING_COMPONENT (dirs): {dir_str}")
            else:
                errors.append(f"DIFFUSERS_MISSING_COMPONENT(dir): {dir_str}")

        if missing_configs:  # 如果有缺失配置
            if verbose:  # 如果详细模式
                config_str = "; ".join(missing_configs_verbose)  # 拼接详细配置信息
                errors.append(f"DIFFUSERS_MISSING_COMPONENT (configs): {config_str}")
            else:
                config_str = _format_component_list(missing_configs)  # 格式化配置列表
                errors.append(f"DIFFUSERS_MISSING_COMPONENT(cfg): {config_str}")

        if missing_weights:  # 如果有缺失权重
            weight_str = _format_component_list(missing_weights)  # 格式化权重列表
            errors.append(f"DIFFUSERS_MISSING_WEIGHTS: {weight_str}")

        return False, " | ".join(errors)  # 返回合并的错误消息

    return True, None  # 验证通过


def validate_cache_with_detailed_reason(  # 验证缓存并返回详细的失败原因（不依赖共享验证标记）
    snapshot_dir: str, weight_files: List[str], model_name_or_path: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate cache and return detailed reason for failure.

    This function performs validation without relying on shared validation markers.
    Used by prevalidate_cached_models.py to provide detailed feedback.

    Args:
        snapshot_dir: Path to the model snapshot directory
        weight_files: List of weight file paths to validate
        model_name_or_path: Model identifier for logging

    Returns:
        Tuple of (success, reason):
        - (True, None) if validation passed
        - (False, reason_str) if validation failed with specific reason
    """  # 验证缓存并返回详细的失败原因，不依赖共享验证标记，由prevalidate_cached_models.py使用以提供详细反馈
    # Guard: weight_files is required  # 守卫条件：weight_files是必需的
    if not weight_files:  # 如果没有权重文件
        return False, "No weight files provided"  # 返回失败

    # Perform full validation and capture failure reasons  # 执行完整验证并捕获失败原因
    revision = os.path.basename(snapshot_dir)  # 获取版本号

    # Read from environment variable instead of huggingface_hub.constants  # 从环境变量读取而非huggingface_hub.constants
    allow_remote_check = os.environ.get("HF_HUB_OFFLINE") != "1"  # 判断是否允许远程检查

    # Validate config and tokenizer files  # 验证配置和分词器文件
    config_valid, missing_config_files = _validate_config_and_tokenizer_files(
        snapshot_dir=snapshot_dir,
        model_id=model_name_or_path,
        revision=revision,
        allow_remote_check=allow_remote_check,
    )

    if not config_valid:  # 如果配置验证失败
        missing_files_str = ", ".join(missing_config_files)  # 拼接缺失文件列表
        return False, f"Missing config/tokenizer files: {missing_files_str}"  # 返回失败

    # Validate weight files  # 验证权重文件
    weights_valid, error_msg, _ = _validate_sharded_model(snapshot_dir, weight_files)  # 验证分片模型
    if not weights_valid:  # 如果权重验证失败
        return False, f"Weight validation failed: {error_msg}"  # 返回失败

    # All validations passed  # 所有验证通过
    return True, None


def validate_cache_lightweight(  # 轻量级运行时缓存完整性验证，仅检查文件存在性不检查损坏
    snapshot_dir: str, requires_hf_quant_config: bool = False
) -> bool:
    """
    Lightweight runtime validation for cache completeness.

    This is used during test runs to ensure the current runner's cache
    is complete before enabling offline mode. Much faster than full validation
    as it only checks file existence, not corruption.

    Args:
        snapshot_dir: Path to the model snapshot directory
        requires_hf_quant_config: If True, hf_quant_config.json must exist
                                  (required for modelopt quantization)

    Returns:
        True if cache is complete, False otherwise
    """  # 轻量级运行时缓存完整性验证，用于测试运行期间确保当前运行器的缓存完整，仅检查文件存在性不检查损坏
    # Check required config files  # 检查必需的配置文件
    required_files = [  # 必需文件列表
        "config.json",  # 模型配置
        "tokenizer_config.json",  # 分词器配置
    ]

    for fname in required_files:  # 遍历必需文件
        if not os.path.exists(os.path.join(snapshot_dir, fname)):  # 如果文件不存在
            return False

    # Check tokenizer files (at least one must exist)  # 检查分词器文件（至少一个必须存在）
    tokenizer_files = [  # 分词器文件列表
        "tokenizer.json",
        "tokenizer.model",
        "tiktoken.model",
    ]

    has_tokenizer = any(
        os.path.exists(os.path.join(snapshot_dir, fname)) for fname in tokenizer_files
    )  # 检查是否至少存在一个分词器文件
    if not has_tokenizer:  # 如果没有分词器文件
        return False

    # Check for trust_remote_code dynamic module files if needed  # 如果需要，检查trust_remote_code动态模块文件
    # When auto_map exists in config.json, the model requires custom Python files  # 当config.json中存在auto_map时，模型需要自定义Python文件
    # These files must be present for offline mode to work  # 这些文件必须存在才能在离线模式下工作
    config_path = os.path.join(snapshot_dir, "config.json")  # 配置文件路径
    if os.path.exists(config_path):  # 如果配置文件存在
        try:  # 尝试读取和解析
            with open(config_path, "r", encoding="utf-8") as f:  # 打开配置文件
                config = json.load(f)  # 加载JSON

            auto_map = config.get("auto_map", {})  # 获取auto_map
            if auto_map and isinstance(auto_map, dict):  # 如果存在且为字典
                # Extract Python module files from auto_map  # 从auto_map中提取Python模块文件
                # auto_map format: {"AutoConfig": "configuration_xxx.ConfigClass", ...}  # auto_map格式
                # We need to check if the .py files exist  # 需要检查.py文件是否存在
                custom_files = set()  # 自定义文件集合
                for key, value in auto_map.items():  # 遍历auto_map
                    if isinstance(value, str) and "." in value:  # 如果值是字符串且包含点
                        # Extract module name (e.g., "configuration_xxx" from "configuration_xxx.ConfigClass")  # 提取模块名
                        module_name = value.split(".")[0]  # 获取模块名
                        custom_files.add(f"{module_name}.py")  # 添加.py文件名

                # Check if all custom files exist in snapshot directory  # 检查快照目录中是否存在所有自定义文件
                for custom_file in custom_files:  # 遍历自定义文件
                    custom_file_path = os.path.join(snapshot_dir, custom_file)  # 构建路径
                    if not os.path.exists(custom_file_path):  # 如果文件不存在
                        logger.debug(
                            "Custom module file not in snapshot: %s for %s",
                            custom_file,
                            snapshot_dir,
                        )
                        return False
                    elif not os.path.isfile(custom_file_path):  # 如果不是文件
                        logger.debug(
                            "Custom module path exists but not a file: %s",
                            custom_file_path,
                        )
                        return False
        except (json.JSONDecodeError, OSError, KeyError) as e:  # 处理异常
            # If we can't read config.json, it will be caught by earlier validation  # 如果无法读取config.json，将由先前的验证捕获
            logger.debug("Failed to check auto_map in config.json: %s", e)

    # Check for weight files with index self-consistency  # 检查权重文件及索引自一致性
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")  # 索引文件路径
    has_index = os.path.exists(index_path)  # 是否存在索引文件

    if has_index:  # 如果存在索引文件
        # If index exists, validate that all shards listed in it exist  # 如果索引存在，验证其中列出的所有分片都存在
        try:
            with open(index_path, "r", encoding="utf-8") as f:  # 打开索引文件
                index_data = json.load(f)  # 加载JSON
            weight_map = index_data.get("weight_map", {})  # 获取权重映射
            if weight_map:  # 如果有权重映射
                # Check that all shard files referenced in index exist  # 检查索引中引用的所有分片文件是否存在
                required_shards = set(weight_map.values())  # 获取所有必需的分片名
                for shard_name in required_shards:  # 遍历分片
                    shard_path = os.path.join(snapshot_dir, shard_name)  # 构建路径
                    if not os.path.exists(shard_path):  # 如果分片不存在
                        logger.debug(
                            "Index validation failed: missing shard %s in %s",
                            shard_name,
                            snapshot_dir,
                        )
                        return False
        except (json.JSONDecodeError, OSError, KeyError) as e:  # 处理异常
            logger.debug("Failed to validate index file %s: %s", index_path, e)
            return False
    else:  # 没有索引文件
        # No index file - check for weight files and validate shard completeness  # 没有索引文件 - 检查权重文件并验证分片完整性
        safetensors_files = glob_module.glob(
            os.path.join(snapshot_dir, "*.safetensors")
        )  # 查找所有safetensors文件
        if not safetensors_files:  # 如果没有权重文件
            return False

        # Check shard completeness for sharded models (e.g., model-00001-of-00047.safetensors)  # 检查分片模型的分片完整性
        # Pattern: prefix-NNNNN-of-NNNNN.safetensors  # 模式：前缀-序号-of-总数.safetensors
        shard_pattern = re.compile(r"(.*?)-(\d+)-of-(\d+)\.safetensors$")  # 分片文件名正则模式
        shard_groups = {}  # 分片组字典

        for f in safetensors_files:  # 遍历safetensors文件
            base_name = os.path.basename(f)  # 获取文件名
            match = shard_pattern.match(base_name)  # 匹配分片模式
            if match:  # 如果匹配
                prefix = match.group(1)  # 前缀
                shard_id = int(match.group(2))  # 分片序号
                total_shards = int(match.group(3))  # 总分片数
                group_key = f"{prefix}-of-{total_shards}"  # 分组键

                if group_key not in shard_groups:  # 如果分组不存在
                    shard_groups[group_key] = {
                        "total": total_shards,
                        "found_shards": set(),
                    }
                shard_groups[group_key]["found_shards"].add(shard_id)  # 添加找到的分片序号

        # Validate each shard group has all expected shards  # 验证每个分片组是否包含所有预期的分片
        for group_key, group_info in shard_groups.items():  # 遍历分片组
            total_shards = group_info["total"]  # 总分片数
            found_shards = group_info["found_shards"]  # 已找到的分片
            expected_shards = set(range(1, total_shards + 1))  # 预期的分片序号集合
            missing_shards = expected_shards - found_shards  # 缺失的分片

            if missing_shards:  # 如果有缺失分片
                logger.debug(
                    "Shard validation failed: missing shards %s in %s for %s",
                    sorted(missing_shards),
                    group_key,
                    snapshot_dir,
                )
                return False

    # Check hf_quant_config.json if required (for modelopt quantization)  # 如果需要，检查hf_quant_config.json（用于modelopt量化）
    if requires_hf_quant_config:  # 如果需要HF量化配置
        hf_quant_path = os.path.join(snapshot_dir, "hf_quant_config.json")  # 构建路径
        if not os.path.exists(hf_quant_path):  # 如果文件不存在
            return False

    return True  # 缓存完整


def _validate_safetensors_file(file_path: str) -> bool:  # 验证safetensors文件是否可读且未损坏
    """
    Validate that a safetensors file is readable and not corrupted.

    Args:
        file_path: Path to the safetensors file

    Returns:
        True if the file is valid, False if corrupted
    """  # 验证safetensors文件是否可读且未损坏
    try:  # 尝试打开并读取
        # Attempt to open and read the header  # 尝试打开并读取头部
        # This will fail if the file is corrupted or incomplete  # 如果文件损坏或不完整将失败
        with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:  # 安全打开文件
            # Just accessing the keys validates the header is readable  # 仅访问键即可验证头部可读
            _ = list(f.keys())  # 读取所有键
        return True  # 文件有效
    except Exception as e:  # 处理异常
        logger.warning(
            "Corrupted safetensors file detected: %s - %s: %s",
            file_path,
            type(e).__name__,
            str(e),
        )
        return False  # 文件损坏


def _validate_pytorch_bin_file(file_path: str) -> bool:  # 验证PyTorch .bin文件是否可读且未损坏
    """
    Validate that a PyTorch .bin file is readable and not corrupted.

    This catches corruption issues like truncated downloads or invalid archives
    that would cause errors like:
    "RuntimeError: PytorchStreamReader failed reading file data/X: invalid header
    or archive is corrupted"

    Args:
        file_path: Path to the .bin file

    Returns:
        True if the file is valid, False if corrupted
    """  # 验证PyTorch .bin文件是否可读且未损坏，捕获如截断下载或无效归档等损坏问题
    try:  # 尝试加载文件
        import torch  # 导入PyTorch

        # Use weights_only=True for security and to avoid executing arbitrary code  # 使用weights_only=True以确保安全并避免执行任意代码
        # mmap=False to fully read the file and catch all corruption  # mmap=False以完全读取文件并捕获所有损坏
        torch.load(file_path, map_location="cpu", weights_only=True, mmap=False)  # 加载文件
        return True  # 文件有效
    except Exception as e:  # 处理异常
        logger.warning(
            "Corrupted PyTorch bin file detected: %s - %s: %s",
            file_path,
            type(e).__name__,
            str(e),
        )
        return False  # 文件损坏


def _check_index_files_exist(snapshot_dir: str) -> Tuple[bool, Optional[str]]:  # 检查safetensors索引文件中列出的所有文件是否实际存在于磁盘
    """
    Check if all files listed in safetensors index files actually exist on disk.

    This catches cases where the snapshot directory exists but files are missing
    (e.g., due to incomplete downloads or corrupted cache).

    Args:
        snapshot_dir: Path to the model snapshot directory

    Returns:
        Tuple of (all_exist, error_message)
    """  # 检查safetensors索引文件中列出的所有文件是否实际存在于磁盘，捕获快照目录存在但文件缺失的情况
    # Find all safetensors index files  # 查找所有safetensors索引文件
    index_files = [
        f for f in os.listdir(snapshot_dir) if f.endswith(".safetensors.index.json")
    ]

    if not index_files:  # 如果没有索引文件
        # No index files means it's not a sharded model, skip this check  # 没有索引文件意味着不是分片模型，跳过此检查
        return True, None

    for index_file in index_files:  # 遍历索引文件
        index_path = os.path.join(snapshot_dir, index_file)  # 构建路径

        # Check if index file is a broken symlink (exists in listing but blob missing)  # 检查索引文件是否为损坏的符号链接（在列表中存在但blob缺失）
        if os.path.islink(index_path) and not os.path.exists(index_path):  # 如果是损坏的符号链接
            # Broken symlink - clean it up so download can proceed  # 损坏的符号链接 - 清理它以便下载可以继续
            try:
                blob_path = os.path.realpath(index_path)  # 获取真实路径
                os.remove(index_path)  # 删除符号链接
                logger.warning(
                    "Removed broken index symlink: %s (blob missing)", index_file
                )
                # Also try to remove dangling blob reference if it somehow exists  # 同时尝试删除可能存在的悬空blob引用
                if os.path.exists(blob_path):  # 如果blob存在
                    os.remove(blob_path)  # 删除blob
            except Exception as e:  # 处理异常
                logger.error("Failed to remove broken symlink %s: %s", index_file, e)
            return (
                False,
                f"Broken index file symlink: {index_file} (cleaned up, will re-download)",
            )  # 返回失败

        try:  # 尝试读取索引文件
            with open(index_path) as f:  # 打开文件
                index_data = json.load(f)  # 加载JSON

            weight_map = index_data.get("weight_map", {})  # 获取权重映射
            if not weight_map:  # 如果没有权重映射
                continue  # 跳过

            # Check that all files in weight_map exist  # 检查weight_map中的所有文件是否存在
            required_files = set(weight_map.values())  # 获取所有必需文件
            missing_files = []  # 缺失文件列表

            for file_name in required_files:  # 遍历必需文件
                file_path = os.path.join(snapshot_dir, file_name)  # 构建路径
                # Check both existence and that it's not a broken symlink  # 同时检查存在性和是否为损坏的符号链接
                if not os.path.exists(file_path):  # 如果文件不存在
                    missing_files.append(file_name)  # 添加到缺失列表

            if missing_files:  # 如果有缺失文件
                return (
                    False,
                    f"Missing {len(missing_files)} file(s) from index {index_file}: {missing_files[:3]}{'...' if len(missing_files) > 3 else ''}",
                )  # 返回失败

        except FileNotFoundError as e:  # 处理文件未找到错误
            # Index file was listed but can't be read - could be race condition or broken state  # 索引文件已列出但无法读取 - 可能是竞态条件或损坏状态
            logger.warning("Failed to read index file %s: %s", index_file, e)
            return (
                False,
                f"Index file {index_file} unreadable (will re-download)",
            )
        except Exception as e:  # 处理其他异常
            logger.warning("Failed to read index file %s: %s", index_file, e)
            continue

    return True, None  # 所有文件都存在


def _validate_sharded_model(  # 验证所有模型分片是否完整且未损坏
    snapshot_dir: str, weight_files: List[str]
) -> Tuple[bool, Optional[str], List[str]]:
    """
    Validate that all model shards are present and not corrupted.

    Args:
        snapshot_dir: Path to the model snapshot directory
        weight_files: List of weight file paths

    Returns:
        Tuple of (is_valid, error_message, corrupted_files)
        - corrupted_files: List of file paths that are corrupted (for selective cleanup)
    """  # 验证所有模型分片是否完整且未损坏，返回(是否有效, 错误消息, 损坏文件列表)
    # First, check if all files from the index actually exist  # 首先检查索引中的所有文件是否实际存在
    # This catches missing files that wouldn't be found by glob  # 这捕获glob无法发现的缺失文件
    index_check_valid, index_error = _check_index_files_exist(snapshot_dir)  # 检查索引文件
    if not index_check_valid:  # 如果索引检查失败
        return False, index_error, []  # 返回失败

    # Pattern for sharded files: model-00001-of-00009.safetensors  # 分片文件模式：model-00001-of-00009.safetensors
    shard_pattern = re.compile(r"(.*?)-(\d+)-of-(\d+)\.(safetensors|bin)")  # 分片文件名正则模式

    # Group files by shard pattern (prefix-*-of-N)  # 按分片模式分组文件
    shard_groups = {}  # 分片组字典
    for f in weight_files:  # 遍历权重文件
        base_name = os.path.basename(f)  # 获取文件名
        match = shard_pattern.match(base_name)  # 匹配分片模式
        if match:  # 如果匹配
            prefix = match.group(1)  # 前缀
            total_shards_str = match.group(3)  # 总分片数字符串
            suffix = match.group(4)  # 后缀（safetensors或bin）

            group_key = f"{prefix}-of-{total_shards_str}.{suffix}"  # 分组键
            if group_key not in shard_groups:  # 如果分组不存在
                shard_groups[group_key] = {
                    "prefix": prefix,  # 前缀
                    "total": int(total_shards_str),  # 总分片数
                    "suffix": suffix,  # 后缀
                    "found_shards": [],  # 已找到的分片列表
                    "files": [],  # 文件路径列表
                }

            shard_id = int(match.group(2))  # 分片序号
            shard_groups[group_key]["found_shards"].append(shard_id)  # 添加分片序号
            shard_groups[group_key]["files"].append(f)  # 添加文件路径

    # Track corrupted files for selective cleanup  # 跟踪损坏文件以便选择性清理
    corrupted_files = []  # 损坏文件列表

    # Validate each shard group  # 验证每个分片组
    for group_key, group_info in shard_groups.items():  # 遍历分片组
        total_shards = group_info["total"]  # 总分片数
        found_shards = set(group_info["found_shards"])  # 已找到的分片集合
        # Shards may be 0-indexed (e.g. inclusionAI/Ring-2.5-1T) or 1-indexed  # 分片可能从0开始编号或从1开始编号
        # (e.g. deepseek-ai/DeepSeek-V3); both are valid HF conventions.  # （如deepseek-ai/DeepSeek-V3）；两者都是有效的HF约定
        min_idx = min(found_shards) if found_shards else 1  # 最小分片索引
        expected_shards = set(range(min_idx, min_idx + total_shards))  # 预期的分片集合

        # Check for missing shards  # 检查缺失的分片
        missing_shards = expected_shards - found_shards  # 缺失的分片
        if missing_shards:  # 如果有缺失
            return (
                False,
                f"Missing shards in {group_key}: {sorted(missing_shards)}",
                [],
            )  # 返回失败

        # Validate weight files for corruption  # 验证权重文件是否损坏
        if group_info["suffix"] == "safetensors":  # 如果是safetensors格式
            for f in group_info["files"]:  # 遍历文件
                if not _validate_safetensors_file(f):  # 验证文件
                    corrupted_files.append(f)  # 添加到损坏列表
        elif group_info["suffix"] == "bin":  # 如果是bin格式
            for f in group_info["files"]:  # 遍历文件
                if not _validate_pytorch_bin_file(f):  # 验证文件
                    corrupted_files.append(f)  # 添加到损坏列表

        # Check for required index file for safetensors shards  # 检查safetensors分片的必需索引文件
        if group_info["suffix"] == "safetensors":  # 如果是safetensors格式
            index_file = os.path.join(
                snapshot_dir, f"{group_info['prefix']}.safetensors.index.json"
            )  # 索引文件路径
            if not os.path.exists(index_file):  # 如果索引文件不存在
                return (
                    False,
                    f"Missing index file: {os.path.basename(index_file)}",
                    [],
                )  # 返回失败

    if corrupted_files:  # 如果有损坏文件
        return (
            False,
            f"Corrupted shard files: {[os.path.basename(f) for f in corrupted_files]}",
            corrupted_files,
        )  # 返回失败和损坏文件列表

    return True, None, []  # 验证通过


def _cleanup_corrupted_files_selective(  # 选择性地删除损坏的文件及其blob以强制重新下载
    model_name_or_path: str, corrupted_files: List[str]
) -> int:
    """
    Selectively remove corrupted files and their blobs to force re-download.

    This is more efficient than removing the entire model cache as it only
    re-downloads corrupted files rather than the entire model.

    Args:
        model_name_or_path: Model identifier
        corrupted_files: List of corrupted file paths (symlinks in snapshot)

    Returns:
        Number of files successfully cleaned up
    """  # 选择性地删除损坏文件及其blob以强制重新下载，比删除整个模型缓存更高效
    cleaned_count = 0  # 清理计数

    for file_path in corrupted_files:  # 遍历损坏文件
        try:  # 尝试清理
            # Resolve symlink to get blob path before deleting symlink  # 在删除符号链接之前解析符号链接以获取blob路径
            if os.path.islink(file_path):  # 如果是符号链接
                blob_path = os.path.realpath(file_path)  # 获取真实路径

                # Delete the symlink  # 删除符号链接
                os.remove(file_path)  # 删除符号链接
                logger.info(
                    "Removed corrupted symlink: %s", os.path.basename(file_path)
                )

                # Delete the blob (the actual corrupted data)  # 删除blob（实际损坏的数据）
                if os.path.exists(blob_path):  # 如果blob存在
                    os.remove(blob_path)  # 删除blob
                    logger.info(
                        "Removed corrupted blob: %s", os.path.basename(blob_path)
                    )

                cleaned_count += 1  # 增加清理计数
            elif os.path.exists(file_path):  # 如果是普通文件
                # Not a symlink, just delete the file  # 不是符号链接，直接删除文件
                os.remove(file_path)  # 删除文件
                logger.info("Removed corrupted file: %s", os.path.basename(file_path))
                cleaned_count += 1  # 增加清理计数

        except Exception as e:  # 处理异常
            logger.error(
                "Failed to remove corrupted file %s: %s",
                os.path.basename(file_path),
                e,
            )

    if cleaned_count > 0:  # 如果有文件被清理
        logger.warning(
            "Removed %d corrupted file(s) for %s. "
            "These will be re-downloaded on next load.",
            cleaned_count,
            model_name_or_path,
        )

    return cleaned_count  # 返回清理数量


def _cleanup_corrupted_model_cache(  # 删除整个损坏的模型缓存目录以强制完全重新下载
    model_name_or_path: str, snapshot_dir: str, reason: str
) -> None:
    """
    Remove entire corrupted model cache directory to force a clean re-download.

    This is used when we cannot selectively clean (e.g., missing shards, incomplete
    downloads with unknown affected files).

    Args:
        model_name_or_path: Model identifier
        snapshot_dir: Path to the snapshot directory
        reason: Reason for cleanup
    """  # 删除整个损坏的模型缓存目录以强制完全重新下载，用于无法选择性清理的情况
    # Navigate up to the model root directory: snapshots/hash -> snapshots -> model_root  # 向上导航到模型根目录：snapshots/hash -> snapshots -> model_root
    repo_folder = os.path.abspath(os.path.join(snapshot_dir, "..", ".."))  # 获取仓库文件夹路径

    try:  # 尝试删除
        logger.warning(
            "Removing entire cache for %s at %s. Reason: %s",
            model_name_or_path,
            repo_folder,
            reason,
        )
        shutil.rmtree(repo_folder)  # 删除整个目录树
        logger.info("Successfully removed corrupted cache directory")  # 记录成功信息
    except Exception as e:  # 处理异常
        logger.error(
            "Failed to remove corrupted cache directory %s: %s. "
            "Manual cleanup may be required.",
            repo_folder,
            e,
        )


def ci_validate_and_cleanup_local_snapshot(  # CI专用：验证本地快照并自动清理损坏文件
    model_name_or_path: str,
    found_local_snapshot_dir: str,
    local_weight_files: List[str],
) -> bool:
    """
    CI-specific validation and cleanup for local model snapshots.

    This function validates the local snapshot and performs automatic cleanup
    if corruption or missing files are detected. This behavior is only appropriate
    for CI environments where we want automatic recovery.

    Args:
        model_name_or_path: Model identifier for logging
        found_local_snapshot_dir: Path to the local snapshot directory
        local_weight_files: List of weight file paths found in the snapshot

    Returns:
        True if the snapshot is valid and can be used, False if it was invalid
        and cleanup was performed (caller should re-download)
    """  # CI专用：验证本地快照并在检测到损坏或缺失文件时执行自动清理，此行为仅适用于CI环境
    # Check for incomplete files and clean up if found  # 检查不完整文件并在发现时清理
    repo_folder = os.path.abspath(os.path.join(found_local_snapshot_dir, "..", ".."))  # 仓库文件夹路径
    blobs_dir = os.path.join(repo_folder, "blobs")  # blobs目录路径

    # Check for incomplete download markers  # 检查不完整下载标记
    incomplete_files = []  # 不完整文件列表
    if os.path.isdir(blobs_dir):  # 如果blobs目录存在
        incomplete_files = glob_module.glob(os.path.join(blobs_dir, "*.incomplete"))  # 查找.incomplete文件

    if incomplete_files:  # 如果有不完整文件
        log_info_on_rank0(
            logger,
            f"Found {len(incomplete_files)} .incomplete files in {blobs_dir} for "
            f"{model_name_or_path}. Will clean up and re-download.",
        )
        _cleanup_corrupted_model_cache(  # 清理整个模型缓存
            model_name_or_path,
            found_local_snapshot_dir,
            f"Incomplete download detected ({len(incomplete_files)} incomplete files)",
        )
        return False  # 返回需要重新下载

    # Validate sharded models and check for corruption  # 验证分片模型并检查损坏
    if local_weight_files:  # 如果有本地权重文件
        is_valid, error_msg, corrupted_files = _validate_sharded_model(
            found_local_snapshot_dir, local_weight_files
        )  # 验证分片模型
        if not is_valid:  # 如果验证失败
            if corrupted_files:  # 如果有损坏文件
                # Selective cleanup: only remove corrupted files  # 选择性清理：仅删除损坏文件
                log_info_on_rank0(
                    logger,
                    f"Found {len(corrupted_files)} corrupted file(s) for "
                    f"{model_name_or_path}: {error_msg}. "
                    "Will selectively clean and re-download only these files.",
                )
                _cleanup_corrupted_files_selective(model_name_or_path, corrupted_files)  # 选择性清理
                return False  # 返回需要重新下载
            else:  # 没有损坏文件但验证失败（缺失分片）
                # Missing shards (not corruption) - let snapshot_download handle it.  # 缺失分片（非损坏）- 让snapshot_download处理
                # IMPORTANT: Do NOT delete the entire cache here, as other processes  # 重要：不要在此删除整个缓存，因为其他进程
                # (TP/EP ranks) may already be loading weights from these files.  # （TP/EP排名）可能正在从这些文件加载权重
                log_info_on_rank0(
                    logger,
                    f"Validation failed for {model_name_or_path}: {error_msg}. "
                    "Will attempt to download missing files.",
                )
                return False  # 返回需要重新下载

        # Also validate single (non-sharded) weight files  # 同时验证单个（非分片）权重文件
        for f in local_weight_files:  # 遍历本地权重文件
            base_name = os.path.basename(f)  # 获取文件名
            # Check if this is a single model file (not sharded)  # 检查是否为单个模型文件（非分片）
            # Include adapter_model.safetensors for LoRA adapters  # 包括LoRA适配器的adapter_model.safetensors
            if base_name in [  # 如果是单个safetensors文件
                "model.safetensors",
                "pytorch_model.safetensors",
                "adapter_model.safetensors",
            ]:
                if not _validate_safetensors_file(f):  # 验证文件
                    log_info_on_rank0(
                        logger,
                        f"Corrupted model file {base_name} for {model_name_or_path}. "
                        "Will selectively clean and re-download this file.",
                    )
                    # Selective cleanup for single file  # 对单个文件进行选择性清理
                    _cleanup_corrupted_files_selective(model_name_or_path, [f])
                    return False  # 返回需要重新下载
            # Also validate single PyTorch .bin files  # 同时验证单个PyTorch .bin文件
            elif base_name in [  # 如果是单个bin文件
                "pytorch_model.bin",
                "model.bin",
                "adapter_model.bin",
            ]:
                if not _validate_pytorch_bin_file(f):  # 验证文件
                    log_info_on_rank0(
                        logger,
                        f"Corrupted model file {base_name} for {model_name_or_path}. "
                        "Will selectively clean and re-download this file.",
                    )
                    # Selective cleanup for single file  # 对单个文件进行选择性清理
                    _cleanup_corrupted_files_selective(model_name_or_path, [f])
                    return False  # 返回需要重新下载

    return True  # 快照有效


def _validate_weights_after_download(  # 下载后验证权重文件以尽早发现损坏
    hf_folder: str,
    allow_patterns: List[str],
    model_name_or_path: str,
) -> bool:
    """
    Validate downloaded weight files to catch corruption early.

    This function validates safetensors files after download to catch
    corruption issues (truncated downloads, network errors, etc.) before
    model loading fails with cryptic errors. If corruption is found,
    the corrupted files are automatically cleaned up.

    Args:
        hf_folder: Path to the downloaded model folder
        allow_patterns: Patterns used to match weight files
        model_name_or_path: Model identifier for error messages

    Returns:
        True if all files are valid, False if corrupted files were found and cleaned up
    """  # 下载后验证权重文件以尽早发现损坏，在模型加载因晦涩错误而失败之前捕获损坏问题
    # Find all weight files that were downloaded  # 查找所有已下载的权重文件
    weight_files: List[str] = []  # 权重文件列表
    for pattern in allow_patterns:  # 遍历匹配模式
        weight_files.extend(glob_module.glob(os.path.join(hf_folder, pattern)))  # 匹配文件

    if not weight_files:  # 如果没有权重文件
        return True  # No weight files to validate  # 没有需要验证的权重文件

    # Validate weight files (safetensors and .bin)  # 验证权重文件（safetensors和.bin）
    corrupted_files = []  # 损坏文件列表
    for f in weight_files:  # 遍历权重文件
        if f.endswith(".safetensors") and os.path.exists(f):  # 如果是safetensors文件
            if not _validate_safetensors_file(f):  # 验证文件
                corrupted_files.append(os.path.basename(f))  # 添加到损坏列表
        elif f.endswith(".bin") and os.path.exists(f):  # 如果是bin文件
            if not _validate_pytorch_bin_file(f):  # 验证文件
                corrupted_files.append(os.path.basename(f))  # 添加到损坏列表

    if corrupted_files:  # 如果有损坏文件
        # Clean up corrupted files so next attempt re-downloads them  # 清理损坏文件以便下次尝试重新下载
        _cleanup_corrupted_files_selective(
            model_name_or_path,
            [os.path.join(hf_folder, f) for f in corrupted_files],
        )  # 选择性清理
        log_info_on_rank0(
            logger,
            f"Downloaded model files are corrupted for {model_name_or_path}: "
            f"{corrupted_files}. The corrupted files have been removed. "
            "Will retry download.",
        )
        return False  # 返回有损坏

    return True  # 所有文件有效


def _get_lock_file_path(  # 生成用于下载协调的唯一锁文件路径
    model_name_or_path: str, cache_dir: Optional[str] = None
) -> str:
    """
    Generate a unique lock file path for download coordination.

    In CI environments where multiple containers share an NFS-mounted HF cache,
    the lock file is placed on the shared cache directory so ALL containers
    coordinate on the same lock. This prevents cross-container .incomplete
    file race conditions.

    Falls back to /dev/shm (container-local) for non-CI or when the cache
    dir is not accessible.

    Args:
        model_name_or_path: Model identifier
        cache_dir: HF cache directory (None to use default)

    Returns:
        Path to the lock file
    """  # 生成用于下载协调的唯一锁文件路径。在CI环境中，锁文件放在共享HF缓存目录上以便所有容器协调
    key_hash = hashlib.sha256(model_name_or_path.encode()).hexdigest()[:16]  # 计算模型标识符的哈希

    # In CI, place lock on the shared HF cache directory so that ALL containers  # 在CI中，将锁放在共享HF缓存目录上，以便所有容器
    # sharing the same NFS-mounted cache coordinate downloads.  # 共享同一个NFS挂载缓存协调下载
    # /dev/shm is container-local and doesn't prevent cross-container races.  # /dev/shm是容器本地的，不能防止跨容器竞态
    try:  # 尝试在HF缓存目录创建锁
        import huggingface_hub.constants

        effective_cache_dir = cache_dir or huggingface_hub.constants.HF_HUB_CACHE  # 获取有效缓存目录
        if os.path.isdir(effective_cache_dir):  # 如果目录存在
            lock_dir = os.path.join(effective_cache_dir, ".sglang_locks")  # 锁目录路径
            os.makedirs(lock_dir, exist_ok=True)  # 创建锁目录
            return os.path.join(lock_dir, f"download_{key_hash}.lock")  # 返回锁文件路径
    except Exception:
        pass

    # Fallback to container-local lock  # 回退到容器本地锁
    if os.path.isdir("/dev/shm"):  # 如果/dev/shm存在
        return f"/dev/shm/sglang_download_lock_{key_hash}"  # 返回/dev/shm中的锁路径
    return f"/tmp/sglang_download_lock_{key_hash}"  # 返回/tmp中的锁路径


def _cleanup_incomplete_blobs(model_name_or_path: str, cache_dir: Optional[str]) -> int:  # 删除模型blobs目录中的过期.incomplete文件
    """
    Remove stale .incomplete files from the model's blobs directory.

    This is lighter than _cleanup_corrupted_model_cache (which deletes the
    entire cache). We only remove .incomplete files so snapshot_download
    starts fresh on retry, preserving any successfully downloaded blobs.

    Args:
        model_name_or_path: Model identifier (e.g., "meta-llama/Llama-2-7b-hf")
        cache_dir: HF cache directory (None to use default)

    Returns:
        Number of .incomplete files removed
    """  # 删除模型blobs目录中的过期.incomplete文件，比_cleanup_corrupted_model_cache更轻量
    try:  # 尝试清理
        import huggingface_hub.constants

        effective_cache_dir = cache_dir or huggingface_hub.constants.HF_HUB_CACHE  # 获取有效缓存目录
        repo_folder_name = huggingface_hub.constants.REPO_ID_SEPARATOR.join(
            ["models", *model_name_or_path.split("/")]
        )  # 构建仓库文件夹名
        blobs_dir = os.path.join(effective_cache_dir, repo_folder_name, "blobs")  # blobs目录路径

        if not os.path.isdir(blobs_dir):  # 如果blobs目录不存在
            return 0

        incomplete_files = glob_module.glob(os.path.join(blobs_dir, "*.incomplete"))  # 查找.incomplete文件
        removed = 0  # 删除计数
        for f in incomplete_files:  # 遍历.incomplete文件
            try:  # 尝试删除
                os.remove(f)  # 删除文件
                removed += 1  # 增加计数
                logger.debug("Removed incomplete blob: %s", os.path.basename(f))
            except OSError as e:  # 处理IO错误
                logger.debug(
                    "Failed to remove incomplete blob %s: %s", os.path.basename(f), e
                )

        if removed > 0:  # 如果有文件被删除
            logger.warning(
                "Cleaned up %d .incomplete blob(s) for %s in %s",
                removed,
                model_name_or_path,
                blobs_dir,
            )
        return removed  # 返回删除数量

    except Exception as e:  # 处理异常
        logger.debug("Failed to clean up incomplete blobs: %s", e)
        return 0


def ci_download_with_validation_and_retry(  # CI专用：带验证和自动重试的下载函数
    model_name_or_path: str,
    allow_patterns: List[str],
    ignore_patterns,
    cache_dir: Optional[str],
    revision: Optional[str],
    max_retries: int = 3,
) -> str:
    """
    CI-specific download with validation and automatic retry on corruption.

    This function handles the download of model weights in CI environments,
    with automatic validation and retry logic for handling corrupted downloads.

    Uses filelock.FileLock on the shared HF cache directory to coordinate
    downloads across all processes AND all containers sharing the same
    NFS-mounted cache. Only one process downloads at a time; others wait
    for the lock then use the cached result.

    Args:
        model_name_or_path: The model name or path
        allow_patterns: The allowed patterns for weight files
        ignore_patterns: The patterns to filter out weight files
        cache_dir: The cache directory to store model weights
        revision: The revision of the model
        max_retries: Maximum number of download retries if corruption is detected

    Returns:
        str: The path to the downloaded model weights

    Raises:
        RuntimeError: If download fails after max_retries attempts
    """  # CI专用：带验证和自动重试的下载函数，使用文件锁协调跨进程和跨容器的下载
    import filelock  # 导入文件锁模块
    import huggingface_hub.constants
    from huggingface_hub import snapshot_download  # 导入快照下载函数
    from tqdm.auto import tqdm  # 导入进度条

    class DisabledTqdm(tqdm):  # 禁用进度条的tqdm子类
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True  # 强制禁用进度条
            super().__init__(*args, **kwargs)

    # Use filelock on the shared HF cache directory to coordinate downloads  # 在共享HF缓存目录上使用文件锁协调下载
    # across all processes AND all containers sharing the same NFS mount.  # 跨所有进程和共享同一NFS挂载的所有容器
    # This prevents cross-container .incomplete file race conditions.  # 防止跨容器.incomplete文件竞态条件
    lock_file_path = _get_lock_file_path(model_name_or_path, cache_dir)  # 获取锁文件路径

    logger.info(
        "[CI Download] Process %d using lock file: %s",
        os.getpid(),
        lock_file_path,
    )

    # filelock.FileLock handles creation, acquisition, and release cleanly.  # filelock.FileLock干净地处理创建、获取和释放
    # timeout=-1 means wait indefinitely (another container may be downloading  # timeout=-1表示无限等待（另一个容器可能正在下载
    # a large model for 30+ minutes).  # 一个大模型，可能需要30多分钟）
    lock = filelock.FileLock(lock_file_path, timeout=-1, mode=0o666)  # 创建文件锁

    logger.info(
        "[CI Download] Process %d waiting to acquire lock for %s",
        os.getpid(),
        model_name_or_path,
    )

    with lock:  # 获取锁
        logger.info(
            "[CI Download] Process %d ACQUIRED lock for %s",
            os.getpid(),
            model_name_or_path,
        )

        # Re-check if another container already downloaded the model while  # 重新检查在我们等待锁期间是否有其他容器已经下载了模型
        # we were waiting for the lock. This avoids redundant downloads.  # 这避免了冗余下载
        try:  # 尝试检查
            from sglang.srt.model_loader.weight_utils import (
                _find_local_hf_snapshot_dir_unlocked,
            )

            cached_path = _find_local_hf_snapshot_dir_unlocked(
                model_name_or_path, cache_dir, allow_patterns, revision
            )  # 查找本地缓存
            if cached_path is not None:  # 如果找到缓存
                logger.info(
                    "[CI Download] Process %d found cached model after "
                    "acquiring lock (downloaded by another container): %s",
                    os.getpid(),
                    cached_path,
                )
                return cached_path  # 返回缓存路径
        except Exception as e:  # 处理异常
            logger.debug(
                "[CI Download] Re-check for cached model failed (non-fatal): %s", e
            )

        # Clean up stale .incomplete files from previous failed downloads  # 清理之前失败下载的过期.incomplete文件
        # before starting. Only do this once before the first attempt.  # 在开始之前。仅在第一次尝试之前执行一次
        cleaned = _cleanup_incomplete_blobs(model_name_or_path, cache_dir)  # 清理.incomplete文件
        if cleaned > 0:  # 如果有文件被清理
            logger.info(
                "[CI Download] Pre-download cleanup: removed %d stale "
                ".incomplete file(s) for %s",
                cleaned,
                model_name_or_path,
            )

        hf_folder = None  # 下载目录
        for attempt in range(max_retries):  # 遍历重试次数
            try:  # 尝试下载
                hf_folder = snapshot_download(  # 执行快照下载
                    model_name_or_path,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    cache_dir=cache_dir,
                    tqdm_class=DisabledTqdm,  # 使用禁用进度条的tqdm
                    revision=revision,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,  # 离线模式标志
                    # Force single-threaded downloads to prevent race conditions  # 强制单线程下载以防止竞态条件
                    # on NFS. HF hub defaults to max_workers=8, which can cause  # 在NFS上。HF hub默认max_workers=8，可能导致
                    # .incomplete file conflicts when multiple threads operate  # 当多线程操作同一文件时产生.incomplete文件冲突
                    # on the same files  #
                    max_workers=1,  # 单线程下载
                )
            except (FileNotFoundError, OSError) as e:  # 处理文件未找到或IO错误
                # Race condition: .incomplete file was moved/deleted by another  # 竞态条件：.incomplete文件被另一个进程移动/删除
                # process. With NFS-level locking this should be rare, but can  # 使用NFS级别锁定这应该很少见，但
                # still happen if lock acquisition fails on some NFS setups.  # 仍可能在某些NFS设置上发生锁定获取失败
                logger.warning(
                    "[CI Download] Process %d hit download error "
                    "(attempt %d/%d) for %s: %s: %s",
                    os.getpid(),
                    attempt + 1,
                    max_retries,
                    model_name_or_path,
                    type(e).__name__,
                    e,
                )
                if attempt < max_retries - 1:  # 如果还有重试机会
                    # Backoff: 10s, 20s, 40s. Clean only the stale  # 退避：10秒、20秒、40秒。仅清理过期的
                    # .incomplete files (not active ones from other processes).  # .incomplete文件（不清理其他进程的活跃文件）
                    backoff = 10 * (2**attempt)  # 计算退避时间
                    logger.info(
                        "[CI Download] Cleaning up .incomplete files and "
                        "retrying in %ds...",
                        backoff,
                    )
                    _cleanup_incomplete_blobs(model_name_or_path, cache_dir)  # 清理.incomplete文件
                    time.sleep(backoff)  # 等待退避时间
                    continue  # 继续重试
                raise RuntimeError(
                    f"Download failed for {model_name_or_path} after "
                    f"{max_retries} attempts due to download errors. "
                    f"Last error: {type(e).__name__}: {e}"
                ) from e  # 抛出运行时错误

            # Validate downloaded files to catch corruption early  # 验证下载的文件以尽早发现损坏
            is_valid = _validate_weights_after_download(
                hf_folder, allow_patterns, model_name_or_path
            )  # 验证下载的权重文件

            if is_valid:  # 如果验证通过
                return hf_folder  # 返回下载目录

            # Validation failed, corrupted files were cleaned up  # 验证失败，损坏文件已清理
            if attempt < max_retries - 1:  # 如果还有重试机会
                log_info_on_rank0(
                    logger,
                    f"Retrying download for {model_name_or_path} "
                    f"(attempt {attempt + 2}/{max_retries})...",
                )
            else:  # 已达到最大重试次数
                raise RuntimeError(
                    f"Downloaded model files are still corrupted for "
                    f"{model_name_or_path} after {max_retries} attempts. "
                    "This may indicate a persistent issue with the model files "
                    "on Hugging Face Hub or network problems."
                )  # 抛出运行时错误

        # Should never reach here, but return hf_folder just in case  # 不应到达此处，但以防万一返回hf_folder
        return hf_folder


def ci_validate_and_clean_hf_cache(model_path: str) -> None:  # CI专用：验证并清理HF缓存中的损坏safetensors文件
    """
    Validate and clean corrupted safetensors files in HF cache before loading.

    This function is needed because HFRunner (used in tests) calls transformers'
    from_pretrained() directly, which bypasses SGLang's weight validation.
    Corrupted cached files can cause cryptic errors like "EOF while parsing"
    from safetensors.

    Only runs in CI to avoid overhead for regular users.

    Args:
        model_path: Model identifier (e.g., "meta-llama/Llama-2-7b")
    """  # 在加载前验证并清理HF缓存中的损坏safetensors文件，因为HFRunner直接调用transformers的from_pretrained()绕过了SGLang的权重验证
    from sglang.utils import is_in_ci  # 导入CI环境检测函数

    if not is_in_ci():  # 如果不在CI环境中
        return  # 直接返回

    # Skip for local paths  # 跳过本地路径
    if os.path.isdir(model_path):  # 如果是本地目录
        return  # 直接返回

    try:  # 尝试验证和清理
        import huggingface_hub.constants

        # Find the HF cache directory for this model  # 查找此模型的HF缓存目录
        cache_dir = huggingface_hub.constants.HF_HUB_CACHE  # 获取缓存目录
        repo_folder = os.path.join(
            cache_dir,
            huggingface_hub.constants.REPO_ID_SEPARATOR.join(
                ["models", *model_path.split("/")]
            ),
        )  # 构建仓库文件夹路径

        if not os.path.isdir(repo_folder):  # 如果仓库文件夹不存在
            return  # 直接返回

        # Find snapshot directories  # 查找快照目录
        snapshots_dir = os.path.join(repo_folder, "snapshots")  # 快照目录路径
        if not os.path.isdir(snapshots_dir):  # 如果快照目录不存在
            return  # 直接返回

        # Check each snapshot for corrupted files  # 检查每个快照是否有损坏文件
        corrupted_files = []  # 损坏文件列表
        for snapshot_hash in os.listdir(snapshots_dir):  # 遍历快照哈希
            snapshot_dir = os.path.join(snapshots_dir, snapshot_hash)  # 快照目录路径
            if not os.path.isdir(snapshot_dir):  # 如果不是目录
                continue  # 跳过

            # Find all safetensors files  # 查找所有safetensors文件
            safetensors_files = glob_module.glob(
                os.path.join(snapshot_dir, "*.safetensors")
            )

            for sf_file in safetensors_files:  # 遍历safetensors文件
                # Skip broken symlinks (os.path.exists returns False for them)  # 跳过损坏的符号链接（os.path.exists对它们返回False）
                if not os.path.exists(sf_file):  # 如果文件不存在（损坏的符号链接）
                    continue  # 跳过

                if not _validate_safetensors_file(sf_file):  # 验证文件
                    corrupted_files.append(sf_file)  # 添加到损坏列表

            # Also find and validate PyTorch .bin files  # 同时查找并验证PyTorch .bin文件
            bin_files = glob_module.glob(os.path.join(snapshot_dir, "*.bin"))  # 查找bin文件

            for bin_file in bin_files:  # 遍历bin文件
                # Skip broken symlinks (os.path.exists returns False for them)  # 跳过损坏的符号链接
                if not os.path.exists(bin_file):  # 如果文件不存在
                    continue  # 跳过

                if not _validate_pytorch_bin_file(bin_file):  # 验证文件
                    corrupted_files.append(bin_file)  # 添加到损坏列表

        if corrupted_files:  # 如果有损坏文件
            logger.warning(
                "HFRunner: Found %d corrupted weight file(s) for %s. "
                "Removing to force re-download.",
                len(corrupted_files),
                model_path,
            )
            _cleanup_corrupted_files_selective(model_path, corrupted_files)  # 选择性清理损坏文件

    except Exception as e:  # 处理异常
        # Don't fail if validation itself fails - let HF handle it  # 如果验证本身失败，不抛出异常 - 让HF处理
        logger.debug("HF cache validation failed (non-fatal): %s", e)
