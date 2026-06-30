from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


SRC = Path("docs")
DST = Path("docs_CN")

TEXT_EXTS = {".md", ".rst", ".txt", ".html"}

PHRASES = [
    ("SGLang Documentation", "SGLang 中文文档"),
    ("Getting Started", "快速开始"),
    ("Get Started", "快速开始"),
    ("Basic Usage", "基础用法"),
    ("Advanced Features", "高级功能"),
    ("Supported Models", "支持的模型"),
    ("Developer Guide", "开发者指南"),
    ("References", "参考资料"),
    ("Installation", "安装"),
    ("Install", "安装"),
    ("Usage", "用法"),
    ("Overview", "概览"),
    ("Quick Start", "快速开始"),
    ("OpenAI-Compatible API", "兼容 OpenAI 的 API"),
    ("OpenAI API", "OpenAI API"),
    ("Native API", "原生 API"),
    ("Sampling Parameters", "采样参数"),
    ("Server Arguments", "服务器参数"),
    ("Environment Variables", "环境变量"),
    ("Performance Dashboard", "性能看板"),
    ("Performance", "性能"),
    ("Benchmark and Profiling", "基准测试与性能分析"),
    ("Benchmark", "基准测试"),
    ("Profiling", "性能分析"),
    ("Production Metrics", "生产环境指标"),
    ("Production Request Trace", "生产环境请求跟踪"),
    ("Multi-Node Deployment", "多节点部署"),
    ("Custom Chat Template", "自定义聊天模板"),
    ("Attention Backend", "注意力后端"),
    ("Speculative Decoding", "投机解码"),
    ("Quantization", "量化"),
    ("Quantized KV Cache", "量化 KV 缓存"),
    ("Expert Parallelism", "专家并行"),
    ("Pipeline Parallelism", "流水线并行"),
    ("Data Parallelism", "数据并行"),
    ("Prefill-Decode Disaggregation", "Prefill-Decode 分离"),
    ("PD Disaggregation", "PD 分离"),
    ("Object Storage", "对象存储"),
    ("Observability", "可观测性"),
    ("Deterministic Inference", "确定性推理"),
    ("Contribution Guide", "贡献指南"),
    ("Development Guide", "开发指南"),
    ("Release Process", "发布流程"),
    ("Support New Models", "支持新模型"),
    ("Extending Models", "扩展模型"),
    ("Text Generation Models", "文本生成模型"),
    ("Multimodal Language Models", "多模态语言模型"),
    ("Embedding Models", "嵌入模型"),
    ("Reward Models", "奖励模型"),
    ("Retrieval and Ranking", "检索与排序"),
    ("Diffusion", "扩散模型"),
    ("Compatibility Matrix", "兼容性矩阵"),
    ("Command Line Interface", "命令行接口"),
    ("Post-Processing", "后处理"),
    ("Reference", "参考"),
    ("Table of Contents", "目录"),
    ("Prerequisites", "前置条件"),
    ("Requirements", "依赖要求"),
    ("Examples", "示例"),
    ("Example", "示例"),
    ("Configuration", "配置"),
    ("Troubleshooting", "故障排查"),
    ("FAQ", "常见问题"),
    ("Note", "注意"),
    ("Warning", "警告"),
    ("Tip", "提示"),
    ("Important", "重要"),
    ("Description", "说明"),
    ("Default", "默认值"),
    ("Options", "选项"),
    ("Parameters", "参数"),
    ("Argument", "参数"),
    ("Models", "模型"),
    ("Model", "模型"),
    ("Request", "请求"),
    ("Response", "响应"),
    ("Server", "服务器"),
    ("Client", "客户端"),
    ("Backend", "后端"),
    ("Frontend", "前端"),
    ("Scheduler", "调度器"),
    ("Router", "路由器"),
    ("Worker", "工作进程"),
    ("Deployment", "部署"),
    ("Deploy", "部署"),
    ("Serving", "服务化"),
    ("Inference", "推理"),
    ("Training", "训练"),
    ("Memory", "内存"),
    ("KV Cache", "KV 缓存"),
    ("Prefix Cache", "前缀缓存"),
    ("Cache", "缓存"),
    ("Features", "功能"),
    ("Feature", "功能"),
    ("Supported", "已支持"),
    ("Unsupported", "不支持"),
    ("Support", "支持"),
    ("Release Lookup", "版本查询"),
    ("Security", "安全"),
    ("Read More", "了解更多"),
    ("Learn More", "了解更多"),
]

SENTENCES = [
    ("This document", "本文档"),
    ("This guide", "本指南"),
    ("This page", "本页面"),
    ("Please refer to", "请参考"),
    ("For more details", "更多详细信息"),
    ("For details", "详细信息"),
    ("For example", "例如"),
    ("For production usage", "用于生产环境时"),
    ("In this section", "本节中"),
    ("You can", "你可以"),
    ("We support", "我们支持"),
    ("SGLang supports", "SGLang 支持"),
    ("The following", "以下"),
    ("Currently", "目前"),
    ("By default", "默认情况下"),
    ("Make sure", "请确保"),
    ("If you want to", "如果你想要"),
    ("To use", "要使用"),
    ("To install", "要安装"),
    ("To deploy", "要部署"),
    ("To enable", "要启用"),
    ("To disable", "要禁用"),
    ("It is recommended", "建议"),
]

REPL = sorted(PHRASES + SENTENCES, key=lambda x: len(x[0]), reverse=True)

LINK_MD = re.compile(r"(!?\[[^\]]*\]\()([^)]*)(\))")
HTML_ATTR = re.compile(r'((?:href|src)=")([^"]+)(")')


def docs_url_to_cn(url: str) -> str:
    if url.startswith(("https://docs.sglang.io/", "http://docs.sglang.io/")):
        rest = re.sub(r"^https?://docs\.sglang\.io/?", "", url)
        rest = rest.replace(".html", ".md")
        return "/docs_CN/" + rest

    url = url.replace("docs\\", "docs_CN\\")
    return re.sub(r"(?<![A-Za-z0-9_./-])docs/", "docs_CN/", url)


def rewrite_links(text: str) -> str:
    text = LINK_MD.sub(lambda m: m.group(1) + docs_url_to_cn(m.group(2)) + m.group(3), text)
    text = HTML_ATTR.sub(lambda m: m.group(1) + docs_url_to_cn(m.group(2)) + m.group(3), text)
    text = re.sub(
        r"https?://docs\.sglang\.io/([A-Za-z0-9_./%#?=&+-]+)",
        lambda m: "/docs_CN/" + m.group(1).replace(".html", ".md"),
        text,
    )
    text = re.sub(
        r"https://github\.com/sgl-project/sglang/blob/main/docs/([A-Za-z0-9_./%#?=&+-]+)",
        lambda m: "/docs_CN/" + m.group(1),
        text,
    )
    text = text.replace('"https://docs.sglang.io"', '"/docs_CN"')
    text = text.replace('conf_py_path": "/docs/"', 'conf_py_path": "/docs_CN/"')
    text = text.replace('conf_py_path = "/docs/"', 'conf_py_path = "/docs_CN/"')
    return text


def protect_inline_code(line: str) -> list[str]:
    return re.split(r"(`[^`]*`)", line)


def zh_line(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith(("```", ":::", ".. ", "   ", "\t", "$ ", "#!", "<script", "</script")):
        return line

    parts = protect_inline_code(line)
    for i, part in enumerate(parts):
        if part.startswith("`") and part.endswith("`"):
            continue
        s = part
        for src, dst in REPL:
            s = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(src) + r"(?![A-Za-z0-9_])", dst, s)
        parts[i] = s
    return "".join(parts)


def process_markdown_like(text: str, source_name: str) -> str:
    text = rewrite_links(text)
    out: list[str] = []
    in_fence = False
    for line in text.splitlines(True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            out.append(zh_line(line))

    if source_name.endswith((".md", ".rst")):
        banner = (
            "<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；"
            "本地 docs 链接已改写到 docs_CN。 -->\n\n"
        )
        content = "".join(out)
        return content if content.startswith("<!-- 本文件由 docs/") else banner + content
    return "".join(out)


def process_ipynb(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        cell["source"] = process_markdown_like(text, src.name).splitlines(True)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)

    for src in SRC.rglob("*"):
        rel = src.relative_to(SRC)
        dst = DST / rel
        if rel.as_posix() in {
            "index.rst",
            "get_started/install.md",
            "basic_usage/openai_api.rst",
            "basic_usage/popular_model_usage.rst",
            "basic_usage/ollama_api.md",
            "basic_usage/qwen3.md",
            "basic_usage/qwen3_5.md",
            "basic_usage/minimax_m2.md",
            "basic_usage/glm45.md",
            "basic_usage/deepseek_ocr.md",
            "basic_usage/llama4.md",
            "basic_usage/glmv.md",
            "basic_usage/qwen3_vl.md",
            "basic_usage/gpt_oss.md",
            "basic_usage/hy3_preview.md",
            "basic_usage/sampling_params.md",
            "advanced_features/server_arguments.md",
            "TRANSLATION_STATUS.md",
        } and dst.exists():
            continue
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        ext = src.suffix.lower()
        if ext in {".png", ".ico"}:
            shutil.copy2(src, dst)
        elif ext == ".ipynb":
            process_ipynb(src, dst)
        elif ext in TEXT_EXTS:
            text = src.read_text(encoding="utf-8", errors="replace")
            dst.write_text(process_markdown_like(text, src.name), encoding="utf-8")
        else:
            try:
                text = src.read_text(encoding="utf-8")
            except Exception:
                shutil.copy2(src, dst)
            else:
                dst.write_text(rewrite_links(text), encoding="utf-8")

    status_path = DST / "TRANSLATION_STATUS.md"
    if status_path.exists():
        return

    status_path.write_text(
        "# docs_CN 生成说明\n\n"
        "本目录从 `docs/` 生成，保留原目录结构。\n\n"
        "- Markdown/RST/HTML/notebook Markdown 单元已进行中文化处理。\n"
        "- 代码块、命令、配置键、模型名、文件路径和外部 URL 保持原样，避免示例失真。\n"
        "- 指向 `docs/` 或 `https://docs.sglang.io/...` 的本地文档链接已改写为 `docs_CN` 对应位置。\n"
        "- 静态资源、脚本、配置和数据文件按原样复制或仅做链接改写。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
