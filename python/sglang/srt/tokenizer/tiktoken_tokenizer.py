# Tiktoken分词器模块
# 实现基于tiktoken的分词器，支持从xtok_dict格式加载词表、
# 特殊token处理、编码/解码、聊天模板以及xgrammar初始化

import functools  # 函数工具模块，用于偏函数
import json  # JSON解析模块
from typing import AbstractSet, Collection, List, Literal, Union  # 类型注解


class TiktokenProcessor:  # Tiktoken处理器，封装分词器和图像处理器
    def __init__(self, name: str):  # 初始化处理器
        self.tokenizer = TiktokenTokenizer(name)  # 创建Tiktoken分词器实例

    def image_processor(self, image):  # 图像处理器，将图像包装为像素值列表
        return {"pixel_values": [image]}  # 返回包含像素值的字典


RESERVED_TOKEN_TEXTS = [f"<|reserved_{i}|>" for i in range(3, 128)]  # 保留token文本列表（3-127）
CONTROL_TOKEN_TEXTS = [f"<|control{i}|>" for i in range(1, 705)]  # 控制token文本列表（1-704）


PAD = "<|pad|>"  # 填充token
EOS = "<|eos|>"  # 结束token
SEP = "<|separator|>"  # 分隔token

DEFAULT_SPECIAL_TOKENS = [PAD, SEP, EOS]  # 默认特殊token列表
DEFAULT_CONTROL_TOKENS = {"pad": PAD, "sep": EOS, "eos": SEP}  # 默认控制token映射

# default + separate each single digit
# 默认 + 分离每个单数字符的正则表达式
PAT_STR_B = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class TiktokenTokenizer:  # Tiktoken分词器类，提供编码、解码和聊天模板功能
    def __init__(self, tokenizer_path):  # 从路径初始化分词器
        import tiktoken  # 导入tiktoken库
        from jinja2 import Template  # 导入Jinja2模板引擎

        # Read the JSON
        # 读取JSON配置文件
        with open(tokenizer_path, "rb") as fin:  # 以二进制模式打开分词器文件
            xtok_dict = json.load(fin)  # 加载JSON字典

        # Copy from train/xlm/tokenizers/tiktoken_wrapper.py::Encoding::from_xtok_dict
        # 从train/xlm/tokenizers/tiktoken_wrapper.py::Encoding::from_xtok_dict复制
        mergeable_ranks = {  # 可合并token的排名表
            bytes(item["bytes"]): item["token"] for item in xtok_dict["regular_tokens"]  # 字节到token ID的映射
        }
        special_tokens = {  # 特殊token表
            bytes(item["bytes"]).decode(): item["token"]  # 将字节解码为字符串，映射到token ID
            for item in xtok_dict["special_tokens"]  # 遍历特殊token列表
        }
        if xtok_dict["word_split"] == "V1":  # 如果分词模式为V1
            pad_str = PAT_STR_B  # 使用默认正则表达式
        else:  # 其他分词模式
            assert False, f"Unknown word_split: {xtok_dict['word_split']}"  # 报错：未知分词模式
        pad_str = xtok_dict.get("pat_str", pad_str)  # 优先使用配置中的正则表达式

        kwargs = {  # tiktoken编码器的构造参数
            "name": tokenizer_path,  # 分词器名称/路径
            "pat_str": pad_str,  # 分词正则表达式
            "mergeable_ranks": mergeable_ranks,  # 可合并token排名
            "special_tokens": special_tokens,  # 特殊token
        }
        if "default_allowed_special" in xtok_dict:  # 如果配置中有默认允许的特殊token
            default_allowed_special = set(  # 构建默认允许的特殊token集合
                [
                    bytes(bytes_list).decode()  # 将字节列表解码为字符串
                    for bytes_list in xtok_dict["default_allowed_special"]  # 遍历配置中的列表
                ]
            )
        if "vocab_size" in xtok_dict:  # 如果配置中有词表大小
            kwargs["explicit_n_vocab"] = xtok_dict["vocab_size"]  # 设置显式词表大小

        # Copy from train/xlm/tokenizers/tiktoken_wrapper.py::Encoding::__init__
        # 从train/xlm/tokenizers/tiktoken_wrapper.py::Encoding::__init__复制
        default_allowed_special = None  # 重置默认允许的特殊token为None
        control_tokens = DEFAULT_CONTROL_TOKENS  # 使用默认控制token映射
        tokenizer = tiktoken.Encoding(**kwargs)  # 创建tiktoken编码器实例
        tokenizer._default_allowed_special = default_allowed_special or set()  # 设置默认允许的特殊token
        tokenizer._control_tokens = control_tokens  # 设置控制token

        def encode_patched(  # 修补后的编码方法，自动包含默认允许的特殊token
            self,
            text: str,  # 待编码文本
            *,
            allowed_special: Union[  # 允许的特殊token集合
                Literal["all"], AbstractSet[str]
            ] = set(),  # noqa: B006
            disallowed_special: Union[Literal["all"], Collection[str]] = "all",  # 禁止的特殊token
        ) -> List[int]:  # 返回token ID列表
            if isinstance(allowed_special, set):  # 如果允许的特殊token是集合类型
                allowed_special |= self._default_allowed_special  # 合并默认允许的特殊token
            return tiktoken.Encoding.encode(  # 调用原始编码方法
                self,  # 编码器实例
                text,  # 待编码文本
                allowed_special=allowed_special,  # 允许的特殊token
                disallowed_special=(),  # 不禁止任何特殊token
            )

        tokenizer.encode = functools.partial(encode_patched, tokenizer)  # 用修补后的方法替换编码方法

        # Allow more tokens to prevent crash
        # 允许更多token以防止崩溃
        tokenizer._default_allowed_special |= set(DEFAULT_CONTROL_TOKENS.values())  # 添加默认控制token
        tokenizer._default_allowed_special |= set(  # 添加保留和控制token文本
            CONTROL_TOKEN_TEXTS + RESERVED_TOKEN_TEXTS
        )

        # Convert to HF interface
        # 转换为HuggingFace接口
        self.tokenizer = tokenizer  # 保存tiktoken编码器实例
        self.bos_token_id = None  # 句首token ID（未设置）
        self.eos_token_id = tokenizer._special_tokens[EOS]  # 句尾token ID
        self.vocab_size = tokenizer.n_vocab  # 词表大小
        self.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'Human: ' + message['content'].strip() + '<|separator|>\n\n' }}{% elif message['role'] == 'system' %}{{ 'System: ' + message['content'].strip() + '<|separator|>\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: '  + message['content'] + '<|separator|>\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"  # 默认聊天模板字符串
        self.chat_template_jinja = Template(self.chat_template)  # 编译Jinja2聊天模板
        self.additional_stop_token_ids = None  # 额外停止token ID（未设置）

    def encode(self, x, add_special_tokens=False):  # 编码文本为token ID列表
        return self.tokenizer.encode(x)  # 调用修补后的编码方法

    def decode(self, x, *args, **kwargs):  # 解码token ID列表为文本
        return self.tokenizer.decode(x)  # 调用tiktoken的解码方法

    def batch_decode(  # 批量解码token ID列表为文本列表
        self, batch, skip_special_tokens=True, spaces_between_special_tokens=False  # 批量和解码选项
    ):
        if len(batch) > 0 and isinstance(batch[0], int):  # 如果批量为单个整数列表
            batch = [[x] for x in batch]  # 将每个整数包装为列表
        return self.tokenizer.decode_batch(batch)  # 调用批量解码方法

    def apply_chat_template(  # 应用聊天模板生成文本或token ID
        self,
        messages,  # 消息列表
        tokenize,  # 是否返回token ID
        add_generation_prompt,  # 是否添加生成提示
        tools=None,  # 工具列表（未使用）
        reasoning_effort=None,  # 推理努力级别（未使用）
        **kwargs,  # Accept additional parameters (e.g., return_dict) for compatibility  # 接受额外参数以保持兼容性
    ):
        ret = self.chat_template_jinja.render(  # 渲染Jinja2模板
            messages=messages, add_generation_prompt=add_generation_prompt  # 传入消息和生成提示参数
        )
        return self.encode(ret) if tokenize else ret  # 根据tokenize参数返回token ID或文本

    def __call__(self, text: List[str], **kwargs):  # 可调用接口，返回输入ID字典
        return {  # 返回字典
            "input_ids": [self.encode(x) for x in text],  # 编码每个文本字符串
        }

    def init_xgrammar(self):  # 初始化xgrammar语法约束所需的信息
        from xgrammar import TokenizerInfo  # 导入xgrammar的TokenizerInfo

        XGRAMMAR_SPECIAL_TOKEN_TEMPLATE = "<|xg_special_token_{}|>"  # xgrammar特殊token模板

        enc = self.tokenizer  # 获取tiktoken编码器
        encoded_vocab = {**enc._mergeable_ranks, **enc._special_tokens}  # 合并可合并token和特殊token
        encoded_vocab = [  # 按token ID排序构建词表列表
            token for token, _ in sorted(encoded_vocab.items(), key=lambda x: x[1])  # 按值（ID）排序
        ]
        override_stop_tokens = [2]  # eos  # 覆盖的停止token ID列表（eos）
        # These are treated as special tokens in xgrammar; we want to avoid them
        # For now, xgrammar treats anything starting with b'\x00' as a special token
        # 这些在xgrammar中被视为特殊token；我们需要避免它们
        # 目前，xgrammar将任何以b'\x00'开头的内容视为特殊token
        xgrammar_special_token_ids = []  # xgrammar特殊token ID列表
        for i, token in enumerate(encoded_vocab):  # 遍历词表
            if isinstance(token, bytes) and token.startswith(b"\x00"):  # 如果是字节类型且以空字节开头
                xgrammar_special_token_ids.append(i)  # 记录为特殊token ID

        for i, id in enumerate(xgrammar_special_token_ids):  # 遍历特殊token ID
            encoded_vocab[id] = XGRAMMAR_SPECIAL_TOKEN_TEMPLATE.format(i)  # 用模板替换原始token文本
        tokenizer_info = TokenizerInfo(  # 创建xgrammar的TokenizerInfo
            encoded_vocab, stop_token_ids=override_stop_tokens  # 传入词表和停止token ID
        )
        assert len(tokenizer_info.special_token_ids) == 0  # 确保没有剩余的特殊token

        return tokenizer_info, override_stop_tokens  # 返回tokenizer信息和停止token列表
