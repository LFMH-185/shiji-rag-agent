#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《史记》语料预处理：解析卷篇结构 → 段落聚合 → 递归切块 → 输出带元数据的 chunks.jsonl

用法:
    python preprocess.py                    # 读 data/shiji.txt（或当前目录 shiji.txt）
    python preprocess.py --show 3           # 额外打印 3 个样例块
    python preprocess.py --chunk-size 400 --overlap 80
    python preprocess.py --stats-only       # 只做解析统计，不写文件

产物:
    data/chunks.jsonl       每行一个块，含 text / embed_text / 元数据
    data/index_meta.json    本次切块的配置与统计

设计要点:
    1. 卷 → 段 → 句 三级结构，RecursiveCharacterTextSplitter 只在段内做二次切分，
       避免跨篇、跨事件拼块。
    2. embed_text 前置「《史记》卷X·篇名：」，让 embedding 知道上下文；text 保持纯净。
    3. 每块保留 char_start/char_end 全局偏移，可回原文核对。
    4. 「太史公曰」强制独立成块；「（表略）」类空篇保留标题块。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - 依赖缺失时给出明确提示
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]


# ---------------------------------------------------------------- 常量

DEFAULT_INPUT_CANDIDATES = ("data/shiji.txt", "shiji.txt")
DEFAULT_OUTPUT = "data/chunks.jsonl"
META_OUTPUT = "data/index_meta.json"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
# 句末标点优先于换行：段落常超过 80 字，若先按 \n 切，splitter 会因整段放不进重叠窗口
# 而丢弃重叠；先按句切则能把「上一句」带进下一块，保证跨块语义连续。
SEPARATORS = ["。", "！", "？", "；", "\n\n", "\n", "，", ""]

CHAPTER_TYPES = ("本纪", "表", "书", "世家", "列传")

# 行内定位「卷X 篇名第N」，允许前缀（(表略)卷二十三 礼书第一）
# 篇名以「第X」结尾，可避免把正文里的「卷X」引用误判为标题
CHAPTER_RE = re.compile(r"卷\s*([一二三四五六七八九十百零〇\d]+)[\s\u3000]+(\S{2,20}?第[一二三四五六七八九十百零〇]+)")
TABLE_PREFIX_RE = re.compile(r"^[（(]\s*表略\s*[)）]\s*")

CN_NUM = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
CN_UNIT = {"十": 10, "百": 100}


# ---------------------------------------------------------------- 数据结构


@dataclass
class Paragraph:
    index: int      # 章内段号，从 1 开始
    text: str
    start: int      # 在整份语料中的字符偏移
    is_comment: bool = False


@dataclass
class Chapter:
    volume_no: int
    volume: str
    title: str
    chapter_type: str
    start: int = 0
    header_text: str = ""
    paragraphs: list[Paragraph] = field(default_factory=list)
    is_table_placeholder: bool = False

    @property
    def chars(self) -> int:
        return sum(len(p.text) for p in self.paragraphs)


@dataclass
class Chunk:
    id: str
    book: str
    volume_no: int
    volume: str
    chapter: str
    chapter_type: str
    chunk_index: int
    para_start: int
    para_end: int
    char_start: int
    char_end: int
    is_taishigong: bool
    text: str
    embed_text: str


# ---------------------------------------------------------------- 工具函数


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding != "utf-8-sig" and text.startswith("\ufeff"):
            text = text.lstrip("\ufeff")
        # 统一换行，保证 char_start/char_end 与切块文本严格对齐
        return text.replace("\r\n", "\n").replace("\r", "\n")
    raise SystemExit(f"无法解码 {path}：不是 UTF-8 / GB18030 文本")


def cn_to_int(token: str) -> int:
    """把「六」「二十三」「一百二十九」转成整数。"""
    if token.isdigit():
        return int(token)
    total = 0
    current = 0
    for char in token:
        if char in CN_NUM:
            current = CN_NUM[char]
        elif char in CN_UNIT:
            unit = CN_UNIT[char]
            total += (current or 1) * unit
            current = 0
        else:
            raise ValueError(f"无法识别的数字：{token}")
    return total + current


def chapter_type_of(title: str) -> str:
    for kind in CHAPTER_TYPES:
        if kind in title:
            return kind
    if "自序" in title:      # 太史公自序传统上归入列传
        return "列传"
    return "其他"


@dataclass
class HeaderHit:
    line_no: int
    start: int          # 标题在行内的起始列
    end: int            # 标题在行内的结束列
    volume_no: int
    title: str
    prefix: str         # 同一行标题之前的正文（粘接情形）
    is_standalone: bool  # 整行只有标题（可带缩进或（表略））


def find_headers(lines: list[str]) -> list[HeaderHit]:
    """扫描全文找卷标题，兼容三种形态：独立行、缩进行、粘在上一段末尾。"""
    hits: list[HeaderHit] = []
    for line_no, line in enumerate(lines):
        for matched in CHAPTER_RE.finditer(line):
            try:
                volume_no = cn_to_int(matched.group(1))
            except ValueError:
                continue
            title = matched.group(2).strip()
            prefix = line[: matched.start()]
            suffix = line[matched.end():]
            prefix_clean = TABLE_PREFIX_RE.sub("", prefix).strip()
            at_start = prefix_clean == ""
            at_end = suffix.strip() == ""
            if not (at_start or at_end):
                continue
            hits.append(
                HeaderHit(
                    line_no=line_no,
                    start=matched.start(),
                    end=matched.end(),
                    volume_no=volume_no,
                    title=title,
                    prefix=prefix if not at_start else "",
                    is_standalone=at_start and at_end,
                )
            )
    return hits


def detect_table_of_contents(hits: list[HeaderHit]) -> set[int]:
    """目录是连续成段的一串独立标题行，取最长连续区间。"""
    lines = sorted({hit.line_no for hit in hits if hit.is_standalone})
    if not lines:
        return set()
    best_start = best_len = 0
    start = prev = lines[0]
    length = 1
    for line_no in lines[1:]:
        if line_no == prev + 1:
            length += 1
        else:
            if length > best_len:
                best_start, best_len = start, length
            start, length = line_no, 1
        prev = line_no
    if length > best_len:
        best_start, best_len = start, length
    return set(range(best_start, best_start + best_len))


# ---------------------------------------------------------------- 解析


def parse_chapters(text: str) -> tuple[list[Chapter], dict]:
    lines = text.splitlines(keepends=True)
    line_offsets: list[int] = []
    cursor = 0
    for line in lines:
        line_offsets.append(cursor)
        cursor += len(line)

    hits = find_headers(lines)
    toc_lines = detect_table_of_contents(hits)
    body_hits = [hit for hit in hits if hit.line_no not in toc_lines]
    if not body_hits:
        raise SystemExit("没有识别到正文卷标题，请检查输入文件格式")

    hits_by_line: dict[int, HeaderHit] = {hit.line_no: hit for hit in body_hits}

    chapters: list[Chapter] = []
    current: Chapter | None = None
    para_index = 0

    def add_paragraph(chapter: Chapter, text_value: str, start: int) -> None:
        nonlocal para_index
        para_index += 1
        chapter.paragraphs.append(
            Paragraph(
                index=para_index,
                text=text_value,
                start=start,
                is_comment=text_value.startswith("太史公曰"),
            )
        )

    for line_no, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        hit = hits_by_line.get(line_no)
        if hit is None:
            if current is not None:
                add_paragraph(current, raw_line.strip(), line_offsets[line_no])
            continue

        # 粘在上一段末尾的标题：先把前缀正文归还给上一卷
        glued_prefix = TABLE_PREFIX_RE.sub("", hit.prefix).strip()
        if glued_prefix and current is not None:
            add_paragraph(current, glued_prefix, line_offsets[line_no])

        header_text = raw_line[hit.start:hit.end].strip()
        current = Chapter(
            volume_no=hit.volume_no,
            volume=f"卷{cn_numeral(hit.volume_no)}",
            title=hit.title,
            chapter_type=chapter_type_of(hit.title),
            start=line_offsets[line_no] + hit.start,
            header_text=header_text,
        )
        chapters.append(current)
        para_index = 0

        suffix = raw_line[hit.end:].strip()
        if suffix:
            add_paragraph(current, suffix, line_offsets[line_no] + hit.end)

    for chapter in chapters:
        if not chapter.paragraphs:
            chapter.is_table_placeholder = True

    stats = {
        "toc_entries": len(toc_lines),
        "chapter_headers": len(body_hits),
        "table_placeholder_chapters": sum(1 for c in chapters if c.is_table_placeholder),
        "comment_paragraphs": sum(1 for c in chapters for p in c.paragraphs if p.is_comment),
    }
    return chapters, stats


def cn_numeral(number: int) -> str:
    """整数转中文数字（1-999），用于生成「卷六」这类标签。"""
    digits = "零一二三四五六七八九"
    if number < 10:
        return digits[number]
    if number < 20:
        return "十" + (digits[number % 10] if number % 10 else "")
    if number < 100:
        tens, ones = divmod(number, 10)
        return digits[tens] + "十" + (digits[ones] if ones else "")
    hundreds, rest = divmod(number, 100)
    prefix = digits[hundreds] + "百"
    if rest == 0:
        return prefix
    if rest < 10:
        return prefix + "零" + digits[rest]
    return prefix + cn_numeral(rest)


# ---------------------------------------------------------------- 切块


def make_splitter(chunk_size: int, chunk_overlap: int):
    if RecursiveCharacterTextSplitter is None:
        raise SystemExit(
            "缺少依赖 langchain-text-splitters，请先执行：\n"
            "  pip install langchain-text-splitters"
        )
    kwargs = dict(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
    )
    try:
        # 新版本支持把标点留在前一块句尾
        return RecursiveCharacterTextSplitter(keep_separator="end", **kwargs)
    except (TypeError, ValueError):
        return RecursiveCharacterTextSplitter(keep_separator=True, **kwargs)


def chapter_segments(chapter: Chapter) -> list[list[Paragraph]]:
    """按「太史公曰」把一章切成若干段组：评论段单独成组，其余连续正文成组。"""
    segments: list[list[Paragraph]] = []
    buffer: list[Paragraph] = []
    for paragraph in chapter.paragraphs:
        if paragraph.is_comment:
            if buffer:
                segments.append(buffer)
                buffer = []
            segments.append([paragraph])
        else:
            buffer.append(paragraph)
    if buffer:
        segments.append(buffer)
    return segments


def paragraph_range(segment: list[Paragraph], start: int, end: int) -> tuple[int, int]:
    """返回与该字符区间有交集的段号范围。"""
    covered = [p.index for p in segment if p.start + len(p.text) > start and p.start < end]
    if not covered:
        return segment[0].index, segment[0].index
    return min(covered), max(covered)


def build_chapter_chunks(chapter: Chapter, splitter, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0

    if chapter.is_table_placeholder:
        # 表内容在底本中省略，仅保留标题块，保证「三代世表讲了什么」能命中
        text = chapter.header_text or f"{chapter.volume} {chapter.title}"
        return [
            Chunk(
                id=f"shiji-v{chapter.volume_no:03d}-c000",
                book="史记",
                volume_no=chapter.volume_no,
                volume=chapter.volume,
                chapter=chapter.title,
                chapter_type=chapter.chapter_type,
                chunk_index=0,
                para_start=0,
                para_end=0,
                char_start=chapter.start,
                char_end=chapter.start + len(text),
                is_taishigong=False,
                text=text,
                embed_text=f"《史记》{chapter.volume}·{chapter.title}：（表略，底本未收录表格内容）",
            )
        ]

    for segment in chapter_segments(chapter):
        joined = "\n".join(p.text for p in segment)
        base = segment[0].start
        pieces = splitter.split_text(joined) or [joined]
        cursor = 0
        for raw_piece in pieces:
            piece = raw_piece.strip()
            if not piece:
                continue
            index += 1
            relative = joined.find(raw_piece, cursor)
            if relative < 0:                      # 兜底：从全文找第一次出现
                relative = joined.find(piece)
            if relative < 0:
                relative = cursor
            # 允许下一块回到重叠区（cursor 只后退一个 overlap）
            cursor = max(cursor, relative + max(1, len(raw_piece) - overlap))
            char_start = base + relative + (len(raw_piece) - len(raw_piece.lstrip()))
            char_end = char_start + len(piece)
            para_start, para_end = paragraph_range(segment, char_start, char_end)
            chunks.append(
                Chunk(
                    id=f"shiji-v{chapter.volume_no:03d}-c{index:03d}",
                    book="史记",
                    volume_no=chapter.volume_no,
                    volume=chapter.volume,
                    chapter=chapter.title,
                    chapter_type=chapter.chapter_type,
                    chunk_index=index,
                    para_start=para_start,
                    para_end=para_end,
                    char_start=char_start,
                    char_end=char_end,
                    is_taishigong=segment[0].is_comment,
                    text=piece,
                    embed_text=f"《史记》{chapter.volume}·{chapter.title}：{piece}",
                )
            )
    return chunks


# ---------------------------------------------------------------- 输出


def write_outputs(
    chunks: list[Chunk], chapters: list[Chapter], parse_stats: dict, args, output: Path, input_path: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    lengths = [len(c.text) for c in chunks]
    meta = {
        "source": str(input_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.overlap,
        "chapters": len(chapters),
        "chunks": len(chunks),
        "chars_indexed": sum(lengths),  # 含块间重叠，因此大于语料总字数
        "chunk_len_min": min(lengths) if lengths else 0,
        "chunk_len_median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "chunk_len_max": max(lengths) if lengths else 0,
        "type_distribution": {
            kind: sum(1 for c in chapters if c.chapter_type == kind)
            for kind in CHAPTER_TYPES
        },
        **parse_stats,
    }
    (output.parent / META_OUTPUT.split("/")[-1]).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.meta = meta


def validate(chapters: list[Chapter], chunks: list[Chunk], parse_stats: dict, text: str) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    numbers = [c.volume_no for c in chapters]
    if numbers and numbers[0] != 1:
        problems.append(f"首卷编号为 {numbers[0]}，预期 1")
    if len(set(numbers)) != len(numbers):
        problems.append("存在重复卷号")
    gaps = sorted(set(range(1, 131)) - set(numbers))
    if gaps:
        warnings.append(f"语料缺少卷号 {gaps}（共 {len(gaps)} 篇），按实际内容处理")
    empty = [c.id for c in chunks if not c.text.strip()]
    if empty:
        problems.append(f"{len(empty)} 个空块")

    # 强校验：按 char_start/char_end 回切原文，必须与 text 完全一致
    for chunk in chunks:
        if chunk.char_end <= chunk.char_start:
            problems.append(f"偏移异常：{chunk.id}")
            break
        if text[chunk.char_start:chunk.char_end] != chunk.text:
            problems.append(f"原文偏移对不上：{chunk.id}")
            break
    return problems, warnings


def show_samples(chunks: list[Chunk], count: int) -> None:
    if not count:
        return
    step = max(1, len(chunks) // count)
    print("\n===== 样例块 =====")
    for chunk in chunks[::step][:count]:
        print(f"\n[{chunk.id}] {chunk.volume}·{chunk.chapter} ({chunk.chapter_type}) "
              f"段{chunk.para_start}-{chunk.para_end} 字符{chunk.char_start}-{chunk.char_end} "
              f"{'·太史公曰' if chunk.is_taishigong else ''}")
        print(f"  text      : {chunk.text[:120]}")
        print(f"  embed_text: {chunk.embed_text[:120]}")


# ---------------------------------------------------------------- 入口


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="《史记》语料预处理与切块")
    parser.add_argument("--input", default=None, help="语料路径，默认 data/shiji.txt 或 shiji.txt")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"输出 jsonl，默认 {DEFAULT_OUTPUT}")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="块大小（字符）")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="块间重叠（字符）")
    parser.add_argument("--stats-only", action="store_true", help="只输出统计，不写文件")
    parser.add_argument("--show", type=int, default=0, help="打印 N 个样例块")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = next((Path(p) for p in DEFAULT_INPUT_CANDIDATES if Path(p).exists()), Path("shiji.txt"))
    if not input_path.exists():
        print(f"找不到语料文件：{input_path}", file=sys.stderr)
        return 1

    text = read_text(input_path)
    chapters, parse_stats = parse_chapters(text)
    splitter = make_splitter(args.chunk_size, args.overlap)

    chunks: list[Chunk] = []
    for chapter in chapters:
        chunks.extend(build_chapter_chunks(chapter, splitter, args.chunk_size, args.overlap))

    problems, warnings = validate(chapters, chunks, parse_stats, text)
    lengths = [len(c.text) for c in chunks]
    missing = sorted(set(range(1, 131)) - {c.volume_no for c in chapters})
    parse_stats["missing_chapters"] = missing

    print(f"语料      : {input_path}（{len(text):,} 字）")
    print(f"正文卷数  : {len(chapters)}  目录条目: {parse_stats['toc_entries']}  "
          f"（表略）篇: {parse_stats['table_placeholder_chapters']}  太史公曰段: {parse_stats['comment_paragraphs']}")
    print(f"切块      : {len(chunks)} 块  size={args.chunk_size} overlap={args.overlap}  "
          f"块长 min/median/max = {min(lengths)}/{sorted(lengths)[len(lengths)//2]}/{max(lengths)}")
    print("篇型分布  : " + "  ".join(
        f"{kind}={sum(1 for c in chapters if c.chapter_type == kind)}" for kind in CHAPTER_TYPES
    ))

    if warnings:
        for item in warnings:
            print(f"[提示] {item}")
    if problems:
        print("\n[校验错误]")
        for item in problems:
            print(f"  - {item}")
    else:
        print("校验      : 通过（无空块、原文偏移逐块可回切）")

    show_samples(chunks, args.show)

    if args.stats_only:
        return 0

    output = Path(args.output)
    write_outputs(chunks, chapters, parse_stats, args, output, input_path)
    print(f"\n已写入    : {output}（{len(chunks)} 行）")
    print(f"元数据    : {output.parent / META_OUTPUT.split('/')[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
