# CUDA运行时绑定工具模块
# 提供对CUDA运行时API的错误检查与封装功能，
# 依赖cuda-python包（cuda.bindings）进行底层CUDA调用。

# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CUDA runtime binding utilities."""  # CUDA运行时绑定工具

try:  # 尝试导入CUDA绑定模块
    from cuda.bindings import runtime as rt  # 导入CUDA运行时绑定
except ImportError:  # 如果导入失败
    rt = None  # 将rt设为None，表示CUDA绑定不可用


def _cudaGetErrorString(error):  # 获取CUDA错误码对应的错误描述字符串
    """Get human-readable error string from a CUDA error code."""  # 从CUDA错误码获取可读的错误字符串
    if rt is None:  # 如果CUDA绑定不可用
        return "<cuda.bindings not available>"  # 返回不可用提示
    err, msg = rt.cudaGetErrorString(error)  # 调用CUDA API获取错误字符串
    if err != rt.cudaError_t.cudaSuccess:  # 如果获取错误字符串本身失败
        return "<unknown>"  # 返回未知错误
    if isinstance(msg, bytes):  # 如果消息是字节类型
        return msg.decode("utf-8", "replace")  # 解码为UTF-8字符串
    return str(msg)  # 否则转为字符串返回


def checkCudaErrors(result):  # 检查CUDA调用结果，若出错则抛出异常
    """Check CUDA API return values and raise on error."""  # 检查CUDA API返回值，出错时抛出异常
    if rt is None:  # 如果CUDA绑定不可用
        raise RuntimeError(  # 抛出运行时错误
            "cuda.bindings is not available. "  # 提示CUDA绑定不可用
            "Install it with: pip install cuda-python"  # 提示安装命令
        )
    if rt is None:  # 冗余检查（保留原始代码逻辑） # redundant check kept as-is
        raise RuntimeError(  # 抛出运行时错误
            "cuda.bindings is not available. "  # 提示CUDA绑定不可用
            "Install it with: pip install cuda-python"  # 提示安装命令
        )
    if result[0] != rt.cudaError_t.cudaSuccess:  # 如果第一个返回值不是成功状态
        raise RuntimeError(  # 抛出运行时错误
            f"CUDA error {int(result[0])}({_cudaGetErrorString(result[0])})"  # 格式化错误码和描述
        )
    if len(result) == 1:  # 如果结果只有一个元素（即错误码）
        return None  # 返回None，表示无额外返回值
    elif len(result) == 2:  # 如果结果有两个元素
        return result[1]  # 返回第二个元素（实际返回值）
    else:  # 如果结果有更多元素
        return result[1:]  # 返回除错误码外的所有元素
