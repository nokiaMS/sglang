# 外部语料库加载模块，用于从JSONL文件中加载和分块token化文档，供N-gram推测解码使用。
# 支持流式分块读取，控制最大token数量，并在文档之间插入分隔符。

import json  # 导入JSON解析库
from collections.abc import Iterator  # 导入迭代器抽象基类
from pathlib import Path  # 导入路径处理库

# Must match SuffixAutomaton::kSeparatorToken in suffix_automaton.h.
SEPARATOR_TOKEN = -(2**31)  # 文档分隔符token，与C++后缀自动机中的常量匹配

# Default chunk size for streaming tokenized documents into the SAM.
DEFAULT_CHUNK_SIZE = 4096  # 流式token化文档的默认分块大小


def iter_external_corpus_chunks(
    path: str, tokenizer, max_tokens: int, chunk_size: int = DEFAULT_CHUNK_SIZE  # 外部语料库路径, # 分词器实例, # 最大token数量, # 分块大小，默认4096
) -> Iterator[list[int]]:
    """将JSONL语料库文件中的文档分块并按固定大小yield token块。"""
    corpus_path = Path(path)  # 将路径字符串转换为Path对象
    if not corpus_path.is_file():  # 检查文件是否存在
        raise ValueError(f"External ngram corpus path does not exist: {path}")  # 文件不存在则抛出异常
    if tokenizer is None:  # 检查分词器是否提供
        raise ValueError("A tokenizer is required to load an external ngram corpus.")  # 未提供分词器则抛出异常
    if max_tokens <= 0:  # 检查最大token数是否为正
        raise ValueError("External ngram corpus max tokens must be positive.")  # 非正数则抛出异常

    total_tokens = 0  # 已加载的token总数
    has_previous_doc = False  # 是否已有前一个文档（用于决定是否插入分隔符）
    with corpus_path.open("r", encoding="utf-8") as f:  # 以UTF-8编码打开文件
        for line_no, line in enumerate(f, start=1):  # 逐行读取，记录行号
            if not line.strip():  # 跳过空行
                continue

            try:
                record = json.loads(line)  # 解析JSON行
            except json.JSONDecodeError as e:  # 捕获JSON解析错误
                raise ValueError(
                    f"Invalid JSON in external ngram corpus at line {line_no}: {e.msg}"
                ) from e  # 重新抛出带行号的异常

            if not isinstance(record, str):  # 检查记录是否为字符串类型
                raise ValueError(
                    "Invalid external ngram corpus record at line "
                    f"{line_no}: expected a JSON string."
                )  # 非字符串类型则抛出异常

            token_ids = list(tokenizer.encode(record, add_special_tokens=False))  # 使用分词器编码，不添加特殊token
            if not token_ids:  # 跳过空token列表
                continue

            separator_cost = 1 if has_previous_doc else 0  # 计算分隔符开销，有前文档则需1个token
            next_total_tokens = total_tokens + separator_cost + len(token_ids)  # 计算加载后的总token数
            if next_total_tokens > max_tokens:  # 检查是否超过最大token限制
                raise ValueError(
                    "External ngram corpus exceeds the configured token limit "
                    f"({max_tokens}) at line {line_no} after loading "
                    f"{total_tokens} tokens."
                )  # 超过限制则抛出异常
            total_tokens = next_total_tokens  # 更新已加载的token总数

            if has_previous_doc:  # 如果有前一个文档
                token_ids = [SEPARATOR_TOKEN] + token_ids  # 在token列表前插入分隔符
            for i in range(0, len(token_ids), chunk_size):  # 按分块大小切分token列表
                yield token_ids[i : i + chunk_size]  # yield每个分块
            has_previous_doc = True  # 标记已有前文档
