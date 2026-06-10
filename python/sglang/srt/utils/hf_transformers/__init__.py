# Copyright 2023-2024 SGLang Team
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
# HF Transformers工具包的入口模块，重新导出所有公共符号
# 将common、config、tokenizer、processor、mistral_utils子模块的公共接口集中导出
"""Hugging Face Transformers utilities.  # HuggingFace Transformers工具

This package provides HF Transformers helpers, split into submodules  # 本包提供HF Transformers辅助工具，分为子模块
(common, config, tokenizer, processor, mistral_utils).  Compatibility  # (common、config、tokenizer、processor、mistral_utils)
monkey-patches live in the sibling ``sglang.srt.utils.hf_transformers_patches``  # 兼容性猴子补丁位于同级模块hf_transformers_patches
module and are applied at sglang import time.  # 在sglang导入时应用
All public symbols are re-exported here for convenience.  The old import  # 所有公共符号在此重新导出以方便使用
path ``sglang.srt.utils.hf_transformers_utils`` is preserved by a  # 旧的导入路径hf_transformers_utils由
separate shim module.  # 单独的垫片模块保留
"""

from ..hf_transformers_patches import normalize_rope_scaling_compat  # 导入RoPE缩放兼容性归一化
from .common import (  # 从通用模块导入
    CONTEXT_LENGTH_KEYS,  # 上下文长度键列表
    AutoConfig,  # 自动配置类
    attach_additional_stop_token_ids,  # 附加停止token ID
    check_gguf_file,  # GGUF文件检查
    download_from_hf,  # 从HF下载
    get_context_length,  # 获取上下文长度
    get_generation_config,  # 获取生成配置
    get_hf_text_config,  # 获取HF文本配置
    get_rope_config,  # 获取RoPE配置
    get_sparse_attention_config,  # 获取稀疏注意力配置
    get_tokenizer_from_processor,  # 从处理器获取分词器
)
from .config import get_config  # 导入配置获取函数
from .processor import get_processor  # 导入处理器获取函数
from .tokenizer import (  # 从分词器模块导入
    _fix_added_tokens_encoding,  # 修复添加的token编码
    _fix_v5_add_bos_eos_token,  # 修复v5的BOS/EOS token
    get_tokenizer,  # 获取分词器
)

__all__ = [  # 模块公开导出符号列表
    "AutoConfig",  # 自动配置类
    "CONTEXT_LENGTH_KEYS",  # 上下文长度键列表
    "_fix_added_tokens_encoding",  # 修复添加的token编码
    "_fix_v5_add_bos_eos_token",  # 修复v5的BOS/EOS token
    "attach_additional_stop_token_ids",  # 附加停止token ID
    "check_gguf_file",  # GGUF文件检查
    "download_from_hf",  # 从HF下载
    "get_config",  # 获取配置
    "get_context_length",  # 获取上下文长度
    "get_generation_config",  # 获取生成配置
    "get_hf_text_config",  # 获取HF文本配置
    "get_processor",  # 获取处理器
    "get_rope_config",  # 获取RoPE配置
    "get_sparse_attention_config",  # 获取稀疏注意力配置
    "get_tokenizer",  # 获取分词器
    "get_tokenizer_from_processor",  # 从处理器获取分词器
    "normalize_rope_scaling_compat",  # RoPE缩放兼容性归一化
]
