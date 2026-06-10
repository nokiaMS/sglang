# 配置日志设置模块
# 本模块提供命令行工具来远程配置SGLang服务器的日志设置，
# 包括日志级别、请求日志、请求转储等功能。
# 通过向服务器发送HTTP POST请求来动态调整日志配置。

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Configure the logging settings of a server.
配置服务器的日志设置。

Usage:
python3 -m sglang.srt.managers.configure_logging --url http://localhost:30000
"""

import argparse  # 导入命令行参数解析模块 # 导入命令行参数解析模块

import requests  # 导入HTTP请求模块 # 导入HTTP请求模块

if __name__ == "__main__":  # 主程序入口 # 主程序入口
    parser = argparse.ArgumentParser()  # 创建参数解析器 # 创建参数解析器
    parser.add_argument("--url", type=str, default="http://localhost:30000")  # 服务器URL参数 # 服务器URL参数
    parser.add_argument(  # 添加日志级别参数 # 添加日志级别参数
        "--log-level",
        type=str,
        default=None,
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set runtime log level",  # 设置运行时日志级别 # 设置运行时日志级别
    )
    parser.add_argument("--log-requests", action="store_true")  # 是否记录请求日志参数 # 是否记录请求日志参数
    parser.add_argument("--log-requests-level", type=int, default=3)  # 请求日志级别参数 # 请求日志级别参数
    parser.add_argument(  # 添加转储请求文件夹参数 # 添加转储请求文件夹参数
        "--dump-requests-folder", type=str, default="/tmp/sglang_request_dump"
    )
    parser.add_argument("--dump-requests-threshold", type=int, default=1000)  # 转储请求阈值参数 # 转储请求阈值参数
    parser.add_argument(  # 添加转储请求排除元数据键参数 # 添加转储请求排除元数据键参数
        "--dump-requests-exclude-meta-keys",
        type=str,
        default=None,
        help=(
            "Comma-separated meta_info keys to strip from each dumped request "
            "(e.g. 'routed_experts,hidden_states'). Pass an empty string to "
            "keep all keys. If not set, the server default is used."
            # 逗号分隔的meta_info键，用于从每个转储请求中移除
            # （例如'routed_experts,hidden_states'）。传入空字符串以
            # 保留所有键。如果未设置，则使用服务器默认值。
        ),
    )
    args = parser.parse_args()  # 解析命令行参数 # 解析命令行参数

    payload = {  # 构建请求负载 # 构建请求负载
        "log_requests": args.log_requests,  # 是否记录请求 # 是否记录请求
        "log_requests_level": args.log_requests_level,  # Log full requests  # 记录完整请求 # 记录完整请求
        "dump_requests_folder": args.dump_requests_folder,  # 转储请求文件夹 # 转储请求文件夹
        "dump_requests_threshold": args.dump_requests_threshold,  # 转储请求阈值 # 转储请求阈值
        "log_level": args.log_level,  # 日志级别 # 日志级别
    }
    if args.dump_requests_exclude_meta_keys is not None:  # 如果指定了排除元数据键 # 如果指定了排除元数据键
        payload["dump_requests_exclude_meta_keys"] = [  # 添加排除键到负载 # 添加排除键到负载
            k.strip()
            for k in args.dump_requests_exclude_meta_keys.split(",")
            if k.strip()
        ]

    response = requests.post(args.url + "/configure_logging", json=payload)  # 发送POST请求配置日志 # 发送POST请求配置日志
    assert response.status_code == 200  # 断言响应状态码为200 # 断言响应状态码为200
