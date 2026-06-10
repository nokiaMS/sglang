# 模型文件校验模块 - 使用SHA256校验和验证模型文件完整性
# 支持从HuggingFace在线元数据或本地生成的校验和文件进行验证
# 示例命令见下方文档字符串
"""
Model File Verifier - Verify model file integrity using SHA256 checksums.

Example commands:
    # Verify using HuggingFace model online metadata
    python -m sglang.srt.utils.model_file_verifier verify --model-path /path/to/model --model-checksum Qwen/Qwen3-0.6B

    # Verify using locally generated checksum
    python -m sglang.srt.utils.model_file_verifier generate --model-path <hf-id-or-model-path> --model-checksum checksums.json
    python -m sglang.srt.utils.model_file_verifier verify --model-path /path/to/model --model-checksum checksums.json
"""

import argparse  # 导入命令行参数解析模块
import fnmatch  # 导入文件名模式匹配模块
import hashlib  # 导入哈希算法模块
import json  # 导入JSON模块
import warnings  # 导入警告模块
from concurrent.futures import ThreadPoolExecutor  # 导入线程池执行器
from dataclasses import asdict, dataclass  # 导入数据类工具
from pathlib import Path  # 导入路径处理模块
from typing import Dict, List, Optional, Tuple  # 导入类型提示

# ======== Data Format ========  # 数据格式定义


@dataclass
class FileInfo:  # 文件信息数据类，包含SHA256哈希和文件大小
    sha256: str  # 文件的SHA256哈希值
    size: int  # 文件大小（字节）


@dataclass
class Manifest:  # 清单数据类，包含所有文件的校验信息
    files: Dict[str, FileInfo]  # 文件名到文件信息的映射

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":  # 从字典创建Manifest对象
        if "checksums" in data:  # 如果使用旧版checksums格式
            warnings.warn(  # 发出弃用警告
                "The 'checksums' format is deprecated. "
                "Please regenerate with the latest version to use the new 'files' format.",
                DeprecationWarning,
                stacklevel=3,
            )
            return cls(  # 从旧格式转换为新格式
                files={
                    k: FileInfo(sha256=v, size=-1) for k, v in data["checksums"].items()
                }
            )
        return cls(files={k: FileInfo(**v) for k, v in data["files"].items()})  # 从新格式创建

    def to_dict(self) -> dict:  # 将Manifest转换为字典
        return asdict(self)


# ======== Constants ========  # 常量定义


IGNORE_PATTERNS = [  # 校验时忽略的文件模式列表
    ".DS_Store",  # macOS系统文件
    "*.lock",  # 锁文件
    ".gitattributes",  # Git属性文件
    "LICENSE",  # 许可证文件
    "LICENSE.*",  # 许可证文件（带后缀）
    "README.md",  # 说明文件
    "README.*",  # 说明文件（带后缀）
    "NOTICE",  # 通知文件
]


# ======== Verify ========  # 验证功能


def verify(*, model_path: str, checksums_source: str, max_workers: int = 4) -> None:  # 验证模型文件完整性
    model_path = Path(model_path).resolve()  # 解析为绝对路径
    expected = _load_checksums(checksums_source)  # 加载期望的校验和
    actual = _compute_manifest_from_folder(  # 计算实际文件的校验和
        model_path=model_path,
        filenames=list(expected.files.keys()),
        max_workers=max_workers,
    )
    _compare_manifests(expected=expected, actual=actual)  # 比较期望和实际的清单
    print(f"[ModelFileVerifier] All {len(expected.files)} files verified successfully.")  # 打印成功信息


def _compare_manifests(*, expected: Manifest, actual: Manifest) -> None:  # 比较期望和实际的清单
    errors = []  # 错误列表
    for filename, exp in expected.files.items():  # 遍历每个期望的文件
        if filename not in actual.files:  # 如果文件缺失
            errors.append(f"{filename}: missing (expected size={exp.size})")
        elif actual.files[filename].sha256 != exp.sha256:  # 如果哈希不匹配
            act = actual.files[filename]  # 获取实际文件信息
            errors.append(
                f"{filename}: mismatch (expected={exp.sha256[:16]}... size={exp.size}, actual={act.sha256[:16]}... size={act.size})"
            )

    if errors:  # 如果有错误
        raise IntegrityError("Integrity check failed: " + "; ".join(errors))  # 抛出完整性错误


# ======== Generate ========  # 生成功能


def generate_checksums(  # 生成模型文件的校验和清单
    *, source: str, output_path: str, max_workers: int = 4
) -> Manifest:
    if Path(source).is_dir():  # 如果源路径是目录
        model_path = Path(source).resolve()  # 解析为绝对路径
        files = _discover_files(model_path)  # 发现模型文件
        if not files:  # 如果没有找到文件
            raise IntegrityError(f"No model files found in {model_path}")  # 抛出错误
        manifest = _compute_manifest_from_folder(  # 从目录计算清单
            model_path=model_path, filenames=files, max_workers=max_workers
        )
    else:  # 如果源是HuggingFace仓库ID
        manifest = Manifest(files=_load_file_infos_from_hf(repo_id=source))  # 从HuggingFace加载

    Path(output_path).write_text(  # 将清单写入文件
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    )

    print(
        f"[ModelFileVerifier] Generated checksums for {len(manifest.files)} files -> {output_path}"
    )
    return manifest  # 返回清单


def _discover_files(model_path: Path) -> List[str]:  # 发现目录中的模型文件（忽略特定模式）
    return sorted(
        e.name
        for e in model_path.iterdir()  # 遍历目录中的条目
        if e.is_file()  # 仅选择文件
        and not e.name.startswith(".")  # 排除隐藏文件
        and not any(fnmatch.fnmatch(e.name, p) for p in IGNORE_PATTERNS)  # 排除忽略模式匹配的文件
    )


# ======== Load Checksums ========  # 加载校验和


def _load_checksums(source: str) -> Manifest:  # 从文件或HuggingFace加载校验和清单
    if Path(source).is_file():  # 如果源是文件
        data = json.loads(Path(source).read_text())  # 读取并解析JSON
        return Manifest.from_dict(data)  # 从字典创建Manifest
    return Manifest(files=_load_file_infos_from_hf(repo_id=source))  # 从HuggingFace加载


def _load_file_infos_from_hf(*, repo_id: str) -> Dict[str, FileInfo]:  # 从HuggingFace仓库加载文件信息
    from huggingface_hub import HfFileSystem  # 导入HuggingFace文件系统

    fs = HfFileSystem()  # 创建文件系统实例
    files = fs.ls(repo_id, detail=True)  # 列出仓库中的文件

    file_infos = dict(
        r for r in map(lambda f: _get_filename_and_info_from_hf_file(fs, f), files) if r
    )
    if not file_infos:  # 如果没有找到文件
        raise IntegrityError(f"No files found in HF repo {repo_id}.")  # 抛出错误

    return file_infos  # 返回文件信息字典


def _get_filename_and_info_from_hf_file(  # 从HuggingFace文件条目获取文件名和文件信息
    fs, file_info
) -> Optional[Tuple[str, FileInfo]]:
    if file_info.get("type") != "file":  # 如果不是文件类型
        return None  # 返回None

    filename = Path(file_info.get("name", "")).name  # 提取文件名
    if any(fnmatch.fnmatch(filename, pat) for pat in IGNORE_PATTERNS):  # 如果匹配忽略模式
        return None  # 返回None

    size = file_info.get("size", -1)  # 获取文件大小
    lfs_info = file_info.get("lfs")  # 获取LFS信息
    if lfs_info and "sha256" in lfs_info:  # 如果有LFS信息且包含SHA256
        return filename, FileInfo(sha256=lfs_info["sha256"], size=size)  # 返回LFS中的SHA256

    if "sha256" in file_info:  # 如果文件信息中直接包含SHA256
        return filename, FileInfo(sha256=file_info["sha256"], size=size)  # 返回文件信息中的SHA256

    content = fs.read_bytes(file_info.get("name", ""))  # 读取文件内容
    return filename, FileInfo(  # 计算并返回SHA256
        sha256=hashlib.sha256(content).hexdigest(), size=len(content)
    )


# ======== Compute Checksums ========  # 计算校验和


def _compute_manifest_from_folder(  # 从文件夹计算模型文件的校验和清单
    *, model_path: Path, filenames: List[str], max_workers: int
) -> Manifest:
    from tqdm import tqdm  # 导入进度条模块

    def compute_one(filename: str) -> Tuple[str, Optional[FileInfo]]:  # 计算单个文件的校验和
        full_path = model_path / filename  # 构建完整文件路径
        if not full_path.exists():  # 如果文件不存在
            return filename, None  # 返回None
        sha256 = compute_sha256(file_path=full_path)  # 计算SHA256
        size = full_path.stat().st_size  # 获取文件大小
        return filename, FileInfo(sha256=sha256, size=size)  # 返回文件信息

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # 使用线程池并行计算
        results = list(
            tqdm(
                executor.map(compute_one, filenames),  # 映射计算函数到文件名列表
                total=len(filenames),  # 总文件数
                desc="Computing checksums",  # 进度条描述
            )
        )

    return Manifest(files={k: v for k, v in results if v is not None})  # 过滤掉None结果并创建清单


def compute_sha256(*, file_path) -> str:  # 计算文件的SHA256哈希值
    sha256 = hashlib.sha256()  # 创建SHA256哈希对象
    with open(file_path, "rb") as f:  # 以二进制模式打开文件
        while chunk := f.read(64 * 1024):  # 按64KB块读取文件
            sha256.update(chunk)  # 更新哈希
    return sha256.hexdigest()  # 返回十六进制哈希字符串


# ======== Exceptions ========  # 异常定义


class IntegrityError(Exception):  # 完整性校验错误异常
    pass


# ======== CLI ========  # 命令行接口


def _add_common_args(parser):  # 向解析器添加公共参数
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local model directory or HuggingFace repo ID",
    )
    parser.add_argument(
        "--model-checksum",
        required=True,
        help="Checksums JSON file path",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )


def main():  # 命令行主函数
    parser = argparse.ArgumentParser(
        description="Model File Verifier - Verify model file integrity using checksums"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)  # 创建子命令解析器

    gen_parser = subparsers.add_parser(
        "generate", help="Generate checksums.json for a model"
    )
    _add_common_args(gen_parser)  # 添加公共参数
    gen_parser.set_defaults(
        func=lambda args: generate_checksums(  # 设置生成校验和的回调函数
            source=args.model_path,
            output_path=args.model_checksum,
            max_workers=args.workers,
        )
    )

    verify_parser = subparsers.add_parser(
        "verify", help="Verify model files against checksums"
    )
    _add_common_args(verify_parser)  # 添加公共参数
    verify_parser.set_defaults(
        func=lambda args: verify(  # 设置验证的回调函数
            model_path=args.model_path,
            checksums_source=args.model_checksum,
            max_workers=args.workers,
        )
    )

    args = parser.parse_args()  # 解析命令行参数
    args.func(args)  # 执行对应的命令函数


if __name__ == "__main__":
    main()
