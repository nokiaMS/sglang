# SGLang CLI serve 子命令模块
# 根据模型类型（扩散模型或语言模型）分发到对应的服务器启动逻辑
# 支持 --model-type 参数覆盖自动检测
# SPDX-License-Identifier: Apache-2.0  # Apache 2.0 许可证声明

import argparse  # 导入命令行参数解析模块
import logging  # 导入日志记录模块
import os  # 导入操作系统接口模块

from sglang.cli.utils import get_is_diffusion_model, get_model_path  # 导入模型类型检测和路径获取工具
from sglang.srt.utils import kill_process_tree  # 导入进程树终止函数
from sglang.srt.utils.common import suppress_noisy_warnings  # 导入噪声警告抑制函数

suppress_noisy_warnings()  # 抑制噪声警告

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def _extract_model_type_override(extra_argv):  # 从命令行参数中提取并移除 --model-type 覆盖值
    """Extract and remove --model-type override from argv."""  # 从 argv 中提取并移除 --model-type 覆盖值
    model_type = "auto"  # 默认模型类型为自动检测
    filtered_argv = []  # 过滤后的参数列表
    i = 0  # 参数索引
    while i < len(extra_argv):  # 遍历所有参数
        arg = extra_argv[i]  # 获取当前参数
        if arg == "--model-type":  # 如果是 --model-type 形式
            if i + 1 >= len(extra_argv):  # 如果没有后续值
                raise Exception(  # 抛出异常
                    "Error: --model-type requires a value. "
                    "Valid values are: auto, llm, diffusion."
                )
            model_type = extra_argv[i + 1]  # 获取模型类型值
            i += 2  # 跳过两个参数
            continue  # 继续下一个参数

        if arg.startswith("--model-type="):  # 如果是 --model-type= 形式
            model_type = arg.split("=", 1)[1]  # 提取等号后的值
            i += 1  # 跳过一个参数
            continue  # 继续下一个参数

        filtered_argv.append(arg)  # 将非 model-type 参数添加到过滤列表
        i += 1  # 移动到下一个参数

    if model_type not in ("auto", "llm", "diffusion"):  # 检查模型类型是否合法
        raise Exception(  # 抛出异常
            f"Error: invalid --model-type '{model_type}'. "
            "Valid values are: auto, llm, diffusion."
        )
    return model_type, filtered_argv  # 返回模型类型和过滤后的参数列表


def serve(args, extra_argv):  # serve 子命令入口函数，根据模型类型启动对应的服务器
    if any(h in extra_argv for h in ("-h", "--help")):  # 如果请求帮助
        # Since the server type is determined by the model, and we don't have a model path,  # 由于服务器类型由模型决定，而我们没有模型路径，
        # we can't show the exact help. Instead, we show a general help message and then  # 无法显示精确的帮助。改为显示通用帮助信息，然后
        # the help for both possible server types.  # 显示两种可能的服务器类型的帮助
        print(  # 打印通用使用说明
            "Usage: sglang serve --model-path <model-name-or-path> [additional-arguments]\n\n"
            "This command can launch either a standard language model server or a diffusion model server.\n"
            "The server type is determined by the --model-path.\n"
            "Optional override: --model-type {auto,llm,diffusion} "
            "(default: auto, fallback to LLM on detection failure)."
        )

        print("\n--- Help for Standard Language Model Server ---")  # 打印语言模型服务器帮助标题
        from sglang.srt.server_args import prepare_server_args  # 导入服务器参数准备函数

        try:  # 尝试显示语言模型服务器帮助
            prepare_server_args(["--help"])  # 触发帮助信息显示
        except SystemExit:  # 捕获 argparse --help 导致的退出
            pass  # 忽略退出异常

        print("\n--- Help for Diffusion Model Server ---")  # 打印扩散模型服务器帮助标题
        try:  # 尝试显示扩散模型服务器帮助
            from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (  # 导入扩散模型参数添加函数
                add_multimodal_gen_serve_args,
            )

            parser = argparse.ArgumentParser(  # 创建参数解析器
                prog="sglang serve",
                description="SGLang Diffusion Model Serving",
            )
            add_multimodal_gen_serve_args(parser)  # 添加扩散模型服务参数
            parser.print_help()  # 打印帮助信息
        except ImportError:  # 如果扩散模型依赖未安装
            print(  # 打印安装提示
                "Diffusion model support is not available. "
                'Install with: pip install "sglang[diffusion]"'
            )
        return  # 返回

    from sglang.srt.plugins import load_plugins  # 导入插件加载函数

    load_plugins()  # 加载插件

    model_type, dispatch_argv = _extract_model_type_override(extra_argv)  # 提取模型类型覆盖值和过滤后的参数
    model_path = get_model_path(dispatch_argv)  # 获取模型路径
    try:  # 尝试启动服务器
        if model_type == "auto":  # 如果是自动检测模式
            is_diffusion_model = get_is_diffusion_model(model_path)  # 自动检测是否为扩散模型
            if is_diffusion_model:  # 如果检测到扩散模型
                logger.info("Diffusion model detected")  # 记录检测到扩散模型
        else:  # 如果手动指定了模型类型
            is_diffusion_model = model_type == "diffusion"  # 根据指定类型判断
            logger.info(  # 记录覆盖模式
                "Dispatch override enabled: --model-type=%s " "(skip auto detection)",
                model_type,
            )

        if is_diffusion_model:  # 如果是扩散模型
            # Logic for Diffusion Models  # 扩散模型的逻辑
            from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (  # 导入扩散模型服务函数
                add_multimodal_gen_serve_args,
                execute_serve_cmd,
            )

            parser = argparse.ArgumentParser(  # 创建参数解析器
                description="SGLang Diffusion Model Serving"
            )
            add_multimodal_gen_serve_args(parser)  # 添加扩散模型服务参数
            parsed_args, remaining_argv = parser.parse_known_args(dispatch_argv)  # 解析参数

            execute_serve_cmd(parsed_args, remaining_argv)  # 执行扩散模型服务命令
        else:  # 如果是语言模型
            # Logic for Standard Language Models  # 标准语言模型的逻辑
            from sglang.launch_server import run_server  # 导入服务器运行函数
            from sglang.srt.server_args import prepare_server_args  # 导入服务器参数准备函数

            server_args = prepare_server_args(dispatch_argv)  # 准备服务器参数

            run_server(server_args)  # 运行语言模型服务器
    finally:  # 无论成功与否
        kill_process_tree(os.getpid(), include_parent=False)  # 终止当前进程的子进程树
