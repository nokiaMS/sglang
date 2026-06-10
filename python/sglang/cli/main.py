# SGLang CLI 主入口模块
# 定义 serve、generate、version 等子命令，并分发到对应的处理函数
import argparse  # 导入命令行参数解析模块

from sglang.cli.utils import get_git_commit_hash  # 导入 Git 提交哈希获取函数
from sglang.version import __version__  # 导入 SGLang 版本号


def version(args, extra_argv):  # 版本信息显示函数
    print(f"sglang version: {__version__}")  # 打印 SGLang 版本号
    print(f"git revision: {get_git_commit_hash()[:7]}")  # 打印 Git 提交哈希前 7 位


def main():  # CLI 主入口函数，解析子命令并分发
    parser = argparse.ArgumentParser()  # 创建参数解析器

    # complex sub commands  # 复杂子命令（需要额外参数处理）
    subparsers = parser.add_subparsers(dest="subcommand", required=True)  # 添加子命令解析器，必须指定子命令
    subparsers.add_parser(  # 添加 serve 子命令
        "serve",
        help="Launch an SGLang server.",  # 启动 SGLang 服务器
        add_help=False,  # 不自动添加帮助选项（延迟到 serve 模块处理）
    )
    subparsers.add_parser(  # 添加 generate 子命令
        "generate",
        help="Run inference on a multimodal model.",  # 对多模态模型运行推理
        add_help=False,  # 不自动添加帮助选项（延迟到 generate 模块处理）
    )

    # simple commands  # 简单命令
    version_parser = subparsers.add_parser(  # 添加 version 子命令
        "version",
        help="Show the version information.",  # 显示版本信息
    )
    version_parser.set_defaults(func=version)  # 设置默认处理函数为 version

    args, extra_argv = parser.parse_known_args()  # 解析已知参数，保留未知参数

    if args.subcommand == "serve":  # 如果子命令为 serve
        from sglang.cli.serve import serve  # 导入 serve 处理函数

        serve(args, extra_argv)  # 调用 serve 函数
    elif args.subcommand == "generate":  # 如果子命令为 generate
        from sglang.cli.generate import generate  # 导入 generate 处理函数

        generate(args, extra_argv)  # 调用 generate 函数
    elif args.subcommand == "version":  # 如果子命令为 version
        version(args, extra_argv)  # 调用 version 函数
