#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地文档问答（Local Document QA）

只用标准库解析 pptx / docx / txt / md，切块后用 BM25 检索，
再把检索到的片段交给本地大模型（Ollama）或任意 OpenAI 兼容接口生成回答。
没有模型时退化为「检索摘要」模式，依然可用。

用法：
    python qa.py                              # 交互式问答（索引当前目录文档）
    python qa.py "项目的周期是多长"             # 单次提问
    python qa.py --docs 答辩展示.pptx "导师是谁" # 只查指定文档（可重复/逗号分隔）
    python qa.py --list                       # 看索引了哪些块
    python qa.py --no-llm "数据仓库分了哪几层"   # 只检索不生成
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 基础配置

SUPPORTED_SUFFIXES = {".pptx", ".docx", ".txt", ".md"}
INDEX_FILE = ".qa_index.json"
INDEX_VERSION = 2  # 切块逻辑变化时递增，让旧缓存失效

NS_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
NS_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# 幻灯片里混进来的版式残留文本
NOISE_PATTERNS = (
    re.compile(r"style\.visibility"),
    re.compile(r"ppt_[xywh]\b"),
    re.compile(r"\{[0-9A-Fa-f-]{20,}\}"),
)

MAX_CHUNK_CHARS = 420
CHUNK_OVERLAP = 80
MIN_SCORE = 0.4  # 低于该分数视为「文档里没有」

# jieba 是可选依赖，装了就分词更准，没装就用字二元组
try:  # pragma: no cover - 取决于环境
    import jieba

    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:  # noqa: BLE001
    _HAS_JIEBA = False


# ---------------------------------------------------------------- 文本抽取


def _clean_line(line: str) -> str:
    line = line.replace("\u00a0", " ").replace("\u3000", " ")
    for pattern in NOISE_PATTERNS:
        line = pattern.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def _is_meaningful(text: str) -> bool:
    """过滤页码占位符（如简历里的 "00"）和纯符号行。"""
    return bool(re.search(r"[0-9A-Za-z\u4e00-\u9fff]{2,}", text))


def extract_docx(path: Path) -> list[tuple[str, str]]:
    """返回 [(位置描述, 段落文本)]。"""
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    body = root.find(f"{NS_W}body")
    if body is None:
        return []

    paragraphs: list[tuple[str, str]] = []
    for index, para in enumerate(body.iter(f"{NS_W}p"), start=1):
        parts: list[str] = []
        for node in para.iter():
            if node.tag == f"{NS_W}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{NS_W}tab":
                parts.append(" ")
            elif node.tag in (f"{NS_W}br", f"{NS_W}cr"):
                parts.append("\n")
        text = _clean_line("".join(parts))
        if text and _is_meaningful(text):
            paragraphs.append((f"第{index}段", text))
    return paragraphs


def extract_pptx(path: Path) -> list[tuple[str, str]]:
    """返回 [(“第N页”, 该页文本)]。"""
    with zipfile.ZipFile(path) as zf:
        slides = [n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        slides.sort(key=lambda n: int(re.search(r"slide(\d+)\.xml", n).group(1)))

        pages: list[tuple[str, str]] = []
        for name in slides:
            number = re.search(r"slide(\d+)\.xml", name).group(1)
            root = ET.fromstring(zf.read(name))
            lines: list[str] = []
            for para in root.iter(f"{NS_A}p"):
                text = _clean_line("".join(node.text or "" for node in para.iter(f"{NS_A}t")))
                if text and _is_meaningful(text):
                    lines.append(text)
            if lines:
                pages.append((f"第{int(number)}页", "\n".join(lines)))
    return pages


def extract_text(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pptx":
        return extract_pptx(path)
    if suffix == ".docx":
        return extract_docx(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    return [(f"第{i}段", b) for i, b in enumerate(blocks, start=1)]


# ---------------------------------------------------------------- 切块


@dataclass
class Chunk:
    doc: str
    loc: str
    text: str
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            digest = hashlib.sha1(f"{self.doc}|{self.loc}|{self.text}".encode()).hexdigest()
            self.chunk_id = digest[:12]


def split_long_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按句子边界切分长文本，相邻块保留一点重叠，避免答非所问。"""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[。！？!?；;\n])", text)
    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(buffer) + len(sentence) <= max_chars:
            buffer += sentence
            continue
        if buffer:
            pieces.append(buffer.strip())
            buffer = buffer[-overlap:] if overlap else ""
        while len(sentence) > max_chars:
            pieces.append(sentence[:max_chars].strip())
            sentence = sentence[max_chars - overlap:]
        buffer += sentence
    if buffer.strip():
        pieces.append(buffer.strip())
    return [p for p in pieces if p]


def merge_blocks(blocks: list[tuple[str, str]], max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[str, str]]:
    """把相邻短块拼成一段，让「项目名称 + 技术栈 + 要点」这类内容落在同一块里。"""
    merged: list[tuple[str, str]] = []
    buffer_loc: list[str] = []
    buffer_text: list[str] = []

    def flush() -> None:
        if not buffer_text:
            return
        head, tail = buffer_loc[0], buffer_loc[-1]
        loc = head if head == tail else f"{head}-{tail}"
        merged.append((loc, "\n".join(buffer_text)))
        buffer_loc.clear()
        buffer_text.clear()

    for loc, text in blocks:
        if len(text) > max_chars:
            flush()
            merged.append((loc, text))
            continue
        pending = sum(len(t) + 1 for t in buffer_text)
        if buffer_text and pending + len(text) > max_chars:
            flush()
        buffer_loc.append(loc)
        buffer_text.append(text)
    flush()
    return merged


def build_chunks(docs: dict[Path, list[tuple[str, str]]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, blocks in docs.items():
        # PPT 每页本身就是一个语义单元，跨页拼会串味；Word/TXT 的碎段落才需要合并
        grouped = blocks if path.suffix.lower() == ".pptx" else merge_blocks(blocks)
        for loc, text in grouped:
            for piece in split_long_text(text):
                chunks.append(Chunk(doc=path.name, loc=loc, text=piece))
    return chunks


# ---------------------------------------------------------------- 分词与检索

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_TOKEN = re.compile(r"[a-z0-9][a-z0-9_+#.\-]*")


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = _ASCII_TOKEN.findall(lowered)
    for run in _CJK_RUN.findall(lowered):
        if _HAS_JIEBA:
            # 搜索模式会额外切出子词（如「哪些项目」→ 哪些/项目），召回更好
            tokens.extend(w for w in jieba.lcut_for_search(run) if w.strip())
        else:
            # 没装 jieba 时用字二元组兜底
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
            if len(run) == 1:
                tokens.append(run)
    return tokens


# 提问里的虚词会干扰 BM25，检索前先去掉
QUERY_STOPWORDS = {
    "的", "了", "是", "在", "有", "和", "与", "及", "或", "把", "被", "对", "中", "上",
    "什么", "哪些", "哪个", "多少", "多长", "多久", "怎么", "怎样", "如何", "为什么",
    "是否", "有没有", "提到", "提到过", "请问", "请", "介绍", "一下", "告诉", "说说",
    "我", "你", "他", "她", "它", "这", "那", "这个", "那个", "吗", "呢", "啊", "呀", "吧",
    "用", "做", "过", "说", "看", "找", "哪", "些", "谁", "时候",
    "没有", "有没", "讲到", "涉及", "内容包括", "分别", "几", "几个", "大概", "主要", "具体",
}

# 行业缩写 → 文档里更常见的全称，命中后额外加入检索词
SYNONYMS = {
    "数仓": "数据仓库",
    "离线数仓": "离线数据仓库",
    "简历": "个人简历",
}


def query_terms(query: str) -> list[str]:
    terms = [token for token in tokenize(query) if token not in QUERY_STOPWORDS]
    for short, full in SYNONYMS.items():
        if short in query:
            terms.extend(token for token in tokenize(full) if token not in QUERY_STOPWORDS)
    # 单字中文区分度太低，只在没别的词时兜底使用
    focused = [token for token in terms if not (len(token) == 1 and _CJK_RUN.fullmatch(token))]
    return focused or terms


class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        self.doc_len = [len(tokens) for tokens in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 1.0
        self.tf: list[Counter[str]] = [Counter(tokens) for tokens in corpus_tokens]
        df: Counter[str] = Counter()
        for counter in self.tf:
            df.update(counter.keys())
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        if not query_tokens or not self.n_docs:
            return scores
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counter in enumerate(self.tf):
                freq = counter.get(term)
                if not freq:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * freq * (self.k1 + 1) / (freq + norm)
        return scores


_PAGE_RE = re.compile(r"^第(\d+)页$")


def _spread_to_neighbor_pages(chunks: list[Chunk], scores: list[float], factor: float = 0.55) -> None:
    """PPT 的上下文常常在相邻页（如「监控」标题页 + 下一页 Zabbix 安装），给邻居页一点分数。"""
    pages: dict[str, dict[int, int]] = {}
    for index, chunk in enumerate(chunks):
        matched = _PAGE_RE.match(chunk.loc)
        if matched:
            pages.setdefault(chunk.doc, {})[int(matched.group(1))] = index

    base = list(scores)
    for doc_pages in pages.values():
        for page, index in doc_pages.items():
            if base[index] <= 0:
                continue
            for delta in (-1, 1):
                neighbor = doc_pages.get(page + delta)
                if neighbor is not None:
                    scores[neighbor] = max(scores[neighbor], base[index] * factor)


def search(chunks: list[Chunk], query: str, top_k: int = 5) -> list[tuple[float, Chunk]]:
    if not chunks:
        return []
    query_tokens = query_terms(query)
    if not query_tokens:
        query_tokens = tokenize(query)
    bm25 = BM25([tokenize(c.text) for c in chunks])
    scores = bm25.scores(query_tokens)

    # 整串命中额外加权：短问题里专有名词很关键
    compact_query = re.sub(r"\s+", "", query.lower())
    if len(compact_query) >= 2:
        for i, chunk in enumerate(chunks):
            if compact_query in re.sub(r"\s+", "", chunk.text.lower()):
                scores[i] += 6.0

    # 长度先验：只有标题的页面信息量低，轻微降权
    for i, chunk in enumerate(chunks):
        scores[i] *= 0.85 + 0.15 * min(1.0, len(chunk.text) / 200)

    _spread_to_neighbor_pages(chunks, scores)

    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

    # 近似重复的页（PPT 里常见：整节共用同一标题）只保留一条，让位给有内容的页
    results: list[tuple[float, Chunk]] = []
    kept_tokens: list[set[str]] = []
    for index in ranked:
        score = scores[index]
        if score < MIN_SCORE:
            break
        chunk = chunks[index]
        tokens = set(tokenize(chunk.text))
        if any(_jaccard(tokens, kept) > 0.85 for kept in kept_tokens):
            continue
        results.append((score, chunk))
        kept_tokens.append(tokens)
        if len(results) >= top_k:
            break
    return results


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------- 索引缓存


def load_documents(paths: list[Path]) -> dict[Path, list[tuple[str, str]]]:
    docs: dict[Path, list[tuple[str, str]]] = {}
    for path in paths:
        try:
            blocks = extract_text(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 解析失败 {path.name}: {exc}", file=sys.stderr)
            continue
        if blocks:
            docs[path] = blocks
    return docs


def index_signature(paths: list[Path]) -> str:
    parts = [f"v{INDEX_VERSION}"]
    parts += [f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in sorted(paths)]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def collect_paths(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            for candidate in sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES):
                # 目录扫描时跳过项目自身的说明文件，避免问答被 README 里的示例污染
                if re.match(r"readme\b", candidate.name, re.IGNORECASE):
                    continue
                if candidate.name.lower() in {"requirements.txt"}:
                    continue
                paths.append(candidate)
        elif path.is_file():
            paths.append(path)
    # 去重并排除缓存文件
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def build_index(paths: list[Path], use_cache: bool = True) -> list[Chunk]:
    cache_path = Path(INDEX_FILE)
    signature = index_signature(paths)
    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                return [Chunk(**item) for item in cached["chunks"]]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    docs = load_documents(paths)
    chunks = build_chunks(docs)
    try:
        payload = {"signature": signature, "chunks": [c.__dict__ for c in chunks]}
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return chunks


# ---------------------------------------------------------------- 大模型


class LLMUnavailable(RuntimeError):
    pass


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def ollama_models(base_url: str) -> list[str]:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return []


def ask_ollama(messages: list[dict], model: str | None = None) -> str:
    base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    models = ollama_models(base_url)
    if not models:
        raise LLMUnavailable("没找到本地 Ollama 服务")
    chosen = model or os.getenv("OLLAMA_MODEL") or models[0]
    try:
        data = _post_json(
            f"{base_url}/api/chat",
            {"model": chosen, "messages": messages, "stream": False},
            {},
            timeout=180,
        )
    except (urllib.error.URLError, OSError) as exc:
        raise LLMUnavailable(f"调用 Ollama 失败：{exc}") from exc
    return data.get("message", {}).get("content", "").strip()


def ask_openai_compatible(messages: list[dict], model: str | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise LLMUnavailable("没有配置 OPENAI_API_KEY / DEEPSEEK_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    chosen = model or os.getenv("OPENAI_MODEL", "deepseek-chat")
    try:
        data = _post_json(
            f"{base_url}/chat/completions",
            {"model": chosen, "messages": messages, "temperature": 0.2, "stream": False},
            {"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
    except (urllib.error.URLError, OSError) as exc:
        raise LLMUnavailable(f"调用在线接口失败：{exc}") from exc
    choices = data.get("choices") or [{}]
    return (choices[0].get("message", {}) or {}).get("content", "").strip()


def generate_answer(question: str, hits: list[tuple[float, Chunk]], model: str | None = None) -> tuple[str, str]:
    """返回 (回答, 模型名)。模型都不可用时抛 LLMUnavailable。"""
    context = "\n\n".join(
        f"[{i}] 来源：{chunk.doc} · {chunk.loc}\n{chunk.text}" for i, (_, chunk) in enumerate(hits, start=1)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个严谨的本地文档问答助手。只能依据用户提供的资料回答，"
                "不要编造资料以外的内容；资料不足时明确说“资料中没有提到”。"
                "回答用中文，简洁分点，并在结论后用 [编号] 标注依据来源。"
            ),
        },
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
    ]

    errors: list[str] = []
    for name, caller in (
        ("Ollama", ask_ollama),
        ("OpenAI-compatible", ask_openai_compatible),
    ):
        try:
            return caller(messages, model), name
        except LLMUnavailable as exc:
            errors.append(f"{name}: {exc}")
    raise LLMUnavailable("；".join(errors))


# ---------------------------------------------------------------- 兜底回答


def extractive_answer(question: str, hits: list[tuple[float, Chunk]], max_snippets: int = 4) -> str:
    """没有大模型时，给出检索到的原文，保证「有答案可看」。"""
    if not hits:
        return "没有检索到相关内容。可以换个说法，或检查文档是否在索引范围内。"

    query_tokens = set(query_terms(question)) or set(tokenize(question))
    lines: list[str] = []

    # 最相关的一块完整展示
    top_chunk = hits[0][1]
    text = top_chunk.text.strip()
    if len(text) > 260:
        text = text[:260].rstrip() + "…"
    lines.append(f"- {text}\n  （{top_chunk.doc} · {top_chunk.loc}）")

    # 其余块各挑一句最贴题的
    for _, chunk in hits[1:]:
        sentences = [s.strip() for s in re.split(r"(?<=[。！？!?；;\n])", chunk.text) if len(s.strip()) >= 8]
        if not sentences:
            continue
        sentence = max(sentences, key=lambda s: len(query_tokens & set(tokenize(s))) / math.sqrt(len(s)))
        if not (query_tokens & set(tokenize(sentence))):
            continue
        lines.append(f"- {sentence}\n  （{chunk.doc} · {chunk.loc}）")
        if len(lines) >= max_snippets:
            break

    return "未连接大模型，下面是检索到的原文摘录：\n" + "\n".join(lines)


# ---------------------------------------------------------------- 问答流程


def answer(question: str, chunks: list[Chunk], top_k: int, use_llm: bool, model: str | None) -> str:
    hits = search(chunks, question, top_k=top_k)
    if not hits:
        return "文档里没有找到和这个问题相关的内容。"

    if use_llm:
        try:
            text, backend = generate_answer(question, hits, model)
            return f"{text}\n\n—— 由 {backend} 生成，依据 {len(hits)} 段原文"
        except LLMUnavailable as exc:
            print(f"[提示] 大模型不可用（{exc}），改为返回检索摘要。", file=sys.stderr)
    return extractive_answer(question, hits)


def preview_sources(hits: list[tuple[float, Chunk]]) -> str:
    lines = []
    for i, (score, chunk) in enumerate(hits, start=1):
        snippet = chunk.text.replace("\n", " ")[:120]
        lines.append(f"[{i}] {chunk.doc} · {chunk.loc}（score={score:.2f}）\n    {snippet}...")
    return "\n".join(lines)


def interactive(chunks: list[Chunk], top_k: int, use_llm: bool, model: str | None) -> None:
    print(f"已索引 {len(chunks)} 个文本块。输入问题回车提问，/sources 看来源，/exit 退出。\n")
    while True:
        try:
            question = input("问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question in {"/exit", "/quit", "exit", "quit"}:
            return
        if question == "/sources":
            print(f"共 {len(chunks)} 块，来自 {len({c.doc for c in chunks})} 个文档。\n")
            continue
        hits = search(chunks, question, top_k=top_k)
        if not hits:
            print("文档里没有找到相关内容。\n")
            continue
        print()
        text = answer(question, chunks, top_k, use_llm, model)
        print(text)
        # 检索摘要里已经带了原文，就不重复贴来源
        if use_llm and "未连接大模型" not in text:
            print()
            print(preview_sources(hits))
        print()


# ---------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="本地文档问答：解析 pptx/docx/txt/md，BM25 检索 + 本地大模型作答。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", nargs="*", help="要问的问题；留空则进入交互模式")
    parser.add_argument(
        "--docs",
        action="append",
        default=None,
        metavar="PATH",
        help="文档或目录，可重复指定或用逗号分隔；默认当前目录",
    )
    parser.add_argument("--top-k", type=int, default=5, help="检索片段数，默认 5")
    parser.add_argument("--model", default=None, help="模型名，例如 qwen2.5:7b / deepseek-chat")
    parser.add_argument("--no-llm", action="store_true", help="只用检索，不调用大模型")
    parser.add_argument("--no-cache", action="store_true", help="忽略索引缓存，重新解析")
    parser.add_argument("--list", action="store_true", help="列出索引内容后退出")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    raw_targets: list[str] = []
    for item in args.docs or ["."]:
        raw_targets.extend(part.strip() for part in re.split(r"[,;]", item) if part.strip())
    paths = collect_paths(raw_targets)
    if not paths:
        print("没有找到可解析的文档（支持 .pptx .docx .txt .md）。", file=sys.stderr)
        return 1

    chunks = build_index(paths, use_cache=not args.no_cache)
    if not chunks:
        print("文档解析后没有文本内容。", file=sys.stderr)
        return 1

    print(f"文档：{', '.join(p.name for p in paths)} | 文本块：{len(chunks)} | 分词：{'jieba' if _HAS_JIEBA else '字二元组'}\n")

    if args.list:
        for chunk in chunks:
            print(f"{chunk.chunk_id}  {chunk.doc} · {chunk.loc}  {len(chunk.text)} 字")
            print(f"    {chunk.text[:80].replace(chr(10), ' ')}")
        return 0

    question = " ".join(args.question).strip()
    use_llm = not args.no_llm
    if not question:
        interactive(chunks, args.top_k, use_llm, args.model)
        return 0

    print(answer(question, chunks, args.top_k, use_llm, args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
