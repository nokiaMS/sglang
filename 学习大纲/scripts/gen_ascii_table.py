# -*- coding: utf-8 -*-
"""
生成中英文混排对齐的 ASCII 框线表格。

核心问题：中文字符在终端中占 2 列宽度，而英文字符占 1 列，
直接用 len() 计算填充空格会导致竖线不对齐。
本脚本使用 unicodedata.east_asian_width() 精确计算视觉宽度。

用法:
  python gen_ascii_table.py > output.txt

输出编码为 UTF-8，可直接嵌入 Markdown 代码块中。
"""

import unicodedata


def visual_width(s):
    """计算字符串的视觉宽度（中文字符算 2 列，英文算 1 列）。"""
    w = 0
    for ch in s:
        eaw = unicodedata.east_asian_width(ch)
        if eaw in ("W", "F"):
            w += 2
        else:
            w += 1
    return w


def pad_right(s, width):
    """右填充空格到指定视觉宽度。"""
    return s + " " * max(0, width - visual_width(s))


def pad_left(s, width):
    """左填充空格到指定视觉宽度。"""
    return " " * max(0, width - visual_width(s)) + s


def center(s, width):
    """居中到指定视觉宽度。"""
    vw = visual_width(s)
    left = (width - vw) // 2
    right = width - vw - left
    return " " * left + s + " " * right


def gen_table(columns, rows):
    """
    生成 ASCII 框线表格。

    参数:
        columns: 列定义列表，每列为 (标题, 视觉宽度)
        rows: 行数据列表，每行为与 columns 等长的字符串列表

    返回:
        字符串，包含完整的 ASCII 框线表格
    """
    widths = [w for _, w in columns]
    headers = [t for t, _ in columns]
    n = len(columns)

    h_line = "┌" + "┬".join("─" * w for w in widths) + "┐"
    m_line = "├" + "┼".join("─" * w for w in widths) + "┤"
    b_line = "└" + "┴".join("─" * w for w in widths) + "┘"

    lines = [h_line]
    # header
    line = "│" + "│".join(pad_right(headers[i], widths[i]) for i in range(n)) + "│"
    lines.append(line)
    lines.append(m_line)

    for row in rows:
        line = "│" + "│".join(pad_right(row[i], widths[i]) for i in range(n)) + "│"
        lines.append(line)

    lines.append(b_line)
    return "\n".join(lines)


def gen_span_table(title, columns, rows):
    """
    生成带顶部横跨标题行的 ASCII 框线表格。

    参数:
        title: 横跨所有列的标题
        columns: 列定义列表，每列为 (标题, 视觉宽度)
        rows: 行数据列表

    返回:
        字符串
    """
    widths = [w for _, w in columns]
    headers = [t for t, _ in columns]
    n = len(columns)
    total_inner = sum(widths) + n - 1  # 内部总宽度（含竖线分隔）

    h_line = "┌" + "─" * total_inner + "┐"
    title_line = "│" + center(title, total_inner) + "│"
    sep_line = "├" + "┬".join("─" * w for w in widths) + "┤"
    m_line = "├" + "┼".join("─" * w for w in widths) + "┤"
    b_line = "└" + "┴".join("─" * w for w in widths) + "┘"

    lines = [h_line, title_line, sep_line]
    # header
    line = "│" + "│".join(pad_right(headers[i], widths[i]) for i in range(n)) + "│"
    lines.append(line)
    lines.append(m_line)

    for row in rows:
        line = "│" + "│".join(pad_right(row[i], widths[i]) for i in range(n)) + "│"
        lines.append(line)

    lines.append(b_line)
    return "\n".join(lines)


# ============================================================
# 示例：14.5 GPU 显存分配全景图
# ============================================================
if __name__ == "__main__":
    print(gen_span_table(
        title="GPU 显存总量",
        columns=[
            ("mem_fraction_static", 30),
            ("预留显存", 40),
        ],
        rows=[
            ["  - 模型权重", "  - 激活内存"],
            ["  - KV Cache 池", "  - CUDA Graph 缓冲区"],
            ["", "  - 常量元数据 (512 MB)"],
            ["", "  - 大并行度调整"],
            ["", "  - DP Attention 额外预留"],
            ["", "  - 分段 CUDA Graph 额外预留"],
            ["", "  - 大显存 GPU 保底 (≥10 GB)"],
            ["", "  - 推测解码额外预留"],
        ],
    ))
