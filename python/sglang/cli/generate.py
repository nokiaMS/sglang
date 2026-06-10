# SGLang CLI 生成子命令模块
# 根据模型类型（扩散模型或语言模型）分发到对应的多模态生成命令
# 目前仅支持扩散模型的生成，语言模型的生成尚未实现
import argparse  # 导入命令行参数解析模块

from sglang.cli.utils import get_is_diffusion_model, get_model_path  # 导入模型类型检测和路径获取工具


def generate(args, extra_argv):  # 生成子命令的入口函数，根据模型类型分发处理
    # If help is requested, show generate subcommand help without requiring --model-path  # 如果请求帮助，显示生成子命令帮助信息，无需 --model-path
    if any(h in extra_argv for h in ("-h", "--help")):  # 检查是否包含帮助标志
        from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (  # 导入多模态生成参数添加函数
            add_multimodal_gen_generate_args,
        )

        parser = argparse.ArgumentParser(description="SGLang Multimodal Generation")  # 创建参数解析器
        add_multimodal_gen_generate_args(parser)  # 添加多模态生成相关参数
        parser.parse_args(extra_argv)  # 解析参数（触发帮助信息显示后退出）
        return  # 返回

    model_path = get_model_path(extra_argv)  # 从额外参数中获取模型路径
    is_diffusion_model = get_is_diffusion_model(model_path)  # 检测模型是否为扩散模型
    if is_diffusion_model:  # 如果是扩散模型
        from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (  # 导入扩散模型生成相关函数
            add_multimodal_gen_generate_args,
            generate_cmd,
        )

        parser = argparse.ArgumentParser(description="SGLang Multimodal Generation")  # 创建参数解析器
        add_multimodal_gen_generate_args(parser)  # 添加多模态生成相关参数
        parsed_args, unknown_args = parser.parse_known_args(extra_argv)  # 解析已知参数，保留未知参数
        generate_cmd(parsed_args, unknown_args)  # 执行扩散模型生成命令
    else:  # 如果不是扩散模型
        raise Exception(  # 抛出异常，当前不支持语言模型的生成子命令
            f"Generate subcommand is not yet supported for model: {model_path}"
        )
