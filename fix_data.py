#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把缺失的《史记》篇目补进 shiji.txt，生成完整语料。

输入:
    shiji.txt   现有语料（127 篇，缺卷四十三、四十四、四十八）
    bu.txt      补丁文本（含「三十世家·赵世家第十三 朗读」这类小标题）

输出:
    shiji_full.txt    默认产物，不覆盖原文件
    --in-place        直接写回 shiji.txt，并留 shiji.txt.bak 备份

用法:
    python fix_data.py                      # 生成 shiji_full.txt
    python fix_data.py --dry-run            # 只打印插入计划，不写文件
    python fix_data.py --in-place           # 覆盖 shiji.txt（自动备份）
    python fix_data.py --patch bu.txt --source shiji.txt --output shiji_full.txt
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from preprocess import (
    CHAPTER_RE,
    cn_numeral,
    cn_to_int,
    detect_table_of_contents,
    find_headers,
)

DEFAULT_SOURCE = "shiji.txt"
DEFAULT_PATCH = "bu.txt"
DEFAULT_OUTPUT = "shiji_full.txt"

# 补丁用字 → 主库用字。依据主库实际频率选定方向：
#   主库 於 4100 / 于 843（且 適嗣 9、適戍 5），故补丁的 于/适/谪 统一向「於/適」靠
#   主库 后 2273 / 後 0，彊 493 / 强 9，沈 50 / 沉 0，谿 28 / 溪 0，翦 60 / 剪 0
#   闇 8 / 暗 0，夥 1 / 伙 0，叩 14 / 扣 0，邺 16 / 鄴 0
VARIANT_MAP = {
    "後": "后",
    "适": "適",
    "谪": "適",
    "强": "彊",
    "鄴": "邺",
    "沉": "沈",
    "溪": "谿",
    "剪": "翦",
    "暗": "闇",
    "伙": "夥",
    "扣": "叩",
}

# 「三十世家·赵世家第十三 朗读」→ 赵世家第十三
SECTION_RE = re.compile(r"^[^·]{0,8}·\s*(\S.*?)\s*朗读\s*$")
JUNK_PATTERNS = (
    re.compile(r"^古籍(赏析|注释|译文)"),
    re.compile(r"^朗读$"),
    re.compile(r"^[\s·\-—_]+$"),
)


# ---------------------------------------------------------------- 数据结构


@dataclass
class PatchSection:
    title: str
    paragraphs: list[str]
    source_line: int

    @property
    def chars(self) -> int:
        return sum(len(p) for p in self.paragraphs)


@dataclass
class InsertPlan:
    section: PatchSection
    volume_no: int
    anchor_line: int      # 插入到该行之前（-1 表示追加到末尾）


# ---------------------------------------------------------------- 读取与清洗


def read_text(path: Path) -> tuple[str, str, bool]:
    """返回 (文本, 原换行符, 是否带 BOM)。"""
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    newline = "\r\n" if b"\r\n" in raw else "\n"
    return text.replace("\r\n", "\n").replace("\r", "\n"), newline, has_bom


def clean_paragraph(line: str) -> str:
    text = line.replace("\u3000", " ").replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", text)


def align_variants(text: str) -> tuple[str, dict[str, int]]:
    """把补丁里的异体/简体写法对齐到主库用法。"""
    counts: dict[str, int] = {}
    text, hits = re.subn(r"(?<!淳)于", "於", text)   # 「淳于」是复姓，不能动
    if hits:
        counts["于→於"] = hits
    for source, target in VARIANT_MAP.items():
        hits = text.count(source)
        if hits:
            text = text.replace(source, target)
            counts[f"{source}→{target}"] = hits
    return text, counts


def parse_patch(text: str, normalize: bool = True) -> list[PatchSection]:
    """按「…·篇名 朗读」小标题切分补丁文本。"""
    lines = text.split("\n")
    sections: list[PatchSection] = []
    current: PatchSection | None = None
    skipped: list[str] = []
    align_counts: dict[str, int] = {}

    for index, line in enumerate(lines):
        marker = SECTION_RE.match(line.strip())
        if marker:
            current = PatchSection(title=marker.group(1).strip(), paragraphs=[], source_line=index + 1)
            sections.append(current)
            continue

        cleaned = clean_paragraph(line)
        if not cleaned:
            continue
        if any(pattern.match(cleaned) for pattern in JUNK_PATTERNS):
            skipped.append(cleaned)
            continue
        if current is None:
            raise SystemExit(f"bu.txt 第 {index + 1} 行出现在任何篇标题之前：{cleaned[:30]}")
        if normalize:
            cleaned, hits = align_variants(cleaned)
            for key, value in hits.items():
                align_counts[key] = align_counts.get(key, 0) + value
        current.paragraphs.append(cleaned)

    for section in sections:
        if not section.paragraphs:
            raise SystemExit(f"补丁篇目《{section.title}》没有正文")
    if not sections:
        raise SystemExit("补丁文本里没有识别到任何「…·篇名 朗读」小标题")
    if skipped:
        print(f"[清洗] 丢弃 {len(skipped)} 行广告/噪声：" + "、".join(sorted(set(skipped))[:3]))
    return sections, align_counts


# ---------------------------------------------------------------- 定位


def build_index(lines: list[str]) -> tuple[dict[str, int], list[tuple[int, int, str]], set[int]]:
    """返回 (篇名→卷号, 正文标题列表[(行号, 卷号, 篇名)], 目录行号集合)。"""
    headers = find_headers(lines)
    toc_lines = detect_table_of_contents(headers)

    title_to_volume: dict[str, int] = {}
    for hit in headers:
        if hit.line_no in toc_lines:
            title_to_volume.setdefault(hit.title, hit.volume_no)
    if not title_to_volume:
        raise SystemExit("未能从目录中解析出篇名，请检查源文件格式")

    body = [(hit.line_no, hit.volume_no, hit.title) for hit in headers if hit.line_no not in toc_lines]
    body.sort()
    return title_to_volume, body, toc_lines


def plan_inserts(
    sections: list[PatchSection],
    title_to_volume: dict[str, int],
    body: list[tuple[int, int, str]],
) -> list[InsertPlan]:
    plans: list[InsertPlan] = []
    for section in sections:
        if section.title not in title_to_volume:
            raise SystemExit(f"目录里找不到篇名《{section.title}》，无法确定卷号")
        volume_no = title_to_volume[section.title]
        later = [line for line, number, _ in body if number > volume_no]
        if not later:
            raise SystemExit(f"卷{cn_numeral(volume_no)} 之后没有可用来定位的篇目")
        plans.append(InsertPlan(section=section, volume_no=volume_no, anchor_line=min(later)))
    # 同一锚点内按卷号升序，实际插入时从后往前
    plans.sort(key=lambda plan: (plan.anchor_line, plan.volume_no))
    return plans


def render_block(plan: InsertPlan, header_style: str) -> list[str]:
    header = plan.section.title
    volume = f"卷{cn_numeral(plan.volume_no)}"
    line = f"{header_style}{volume} {header}"
    return [line] + plan.section.paragraphs


# ---------------------------------------------------------------- 校验


def verify(lines: list[str], inserted: list[int], expected_total: int = 130) -> list[str]:
    problems: list[str] = []
    headers = find_headers(lines)
    body = [hit for hit in headers if hit.line_no not in detect_table_of_contents(headers)]
    numbers = [hit.volume_no for hit in body]
    if len(numbers) != expected_total:
        problems.append(f"正文篇数为 {len(numbers)}，预期 {expected_total}")
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    if duplicates:
        problems.append(f"重复卷号：{duplicates}")
    missing = sorted(set(range(1, expected_total + 1)) - set(numbers))
    if missing:
        problems.append(f"仍然缺少卷号：{missing}")
    for volume_no in inserted:
        if volume_no not in numbers:
            problems.append(f"插入的卷{cn_numeral(volume_no)} 未出现在正文标题中")
    if any(not line.strip() for line in lines):
        pass  # 允许空行，仅作占位检查
    return problems


# ---------------------------------------------------------------- 主流程


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把缺失的《史记》篇目补入语料")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="现有语料")
    parser.add_argument("--patch", default=DEFAULT_PATCH, help="补丁文本")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出文件")
    parser.add_argument("--in-place", action="store_true", help="覆盖 --source（自动备份 .bak）")
    parser.add_argument("--dry-run", action="store_true", help="只打印插入计划")
    parser.add_argument("--no-normalize", action="store_true", help="不做异体字对齐，保留补丁原用字")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    source_path = Path(args.source)
    patch_path = Path(args.patch)
    for path in (source_path, patch_path):
        if not path.exists():
            print(f"找不到文件：{path}", file=sys.stderr)
            return 1

    source_text, newline, has_bom = read_text(source_path)
    patch_text, _, _ = read_text(patch_path)
    lines = source_text.split("\n")

    title_to_volume, body, toc_lines = build_index(lines)
    sections, align_counts = parse_patch(patch_text, normalize=not args.no_normalize)
    plans = plan_inserts(sections, title_to_volume, body)

    print(f"源文件    : {source_path}（{len(lines)} 行，正文 {len(body)} 篇）")
    print(f"补丁文件  : {patch_path}（{len(sections)} 篇）")
    if align_counts:
        summary = "、".join(f"{key}×{value}" for key, value in sorted(align_counts.items(), key=lambda kv: -kv[1]))
        print(f"用字对齐  : {sum(align_counts.values())} 处 → {summary}")
    print("插入计划  :")
    for plan in plans:
        anchor = "文件末尾" if plan.anchor_line < 0 else f"第 {plan.anchor_line + 1} 行之前"
        print(
            f"  {plan.section.title:<12} → 卷{cn_numeral(plan.volume_no):<4} "
            f"{len(plan.section.paragraphs):>3} 段 / {plan.section.chars:>5} 字，插到 {anchor}"
        )

    if args.dry_run:
        return 0

    # 从后往前插入，避免行号漂移；同一锚点内先插高卷号，
    # 这样低卷号最终落在前面（保证 43 → 44 → 45 的顺序）
    for plan in sorted(plans, key=lambda p: (p.anchor_line, p.volume_no), reverse=True):
        style = " " * max(0, len(lines[plan.anchor_line]) - len(lines[plan.anchor_line].lstrip()))
        block = render_block(plan, style)
        lines[plan.anchor_line:plan.anchor_line] = block

    problems = verify(lines, [plan.volume_no for plan in plans])
    if problems:
        print("\n[校验失败]", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        return 1

    output_text = newline.join(lines)
    target = source_path if args.in_place else Path(args.output)
    if args.in_place:
        backup = source_path.with_suffix(source_path.suffix + ".bak")
        shutil.copy2(source_path, backup)
        print(f"备份      : {backup}")
    payload = output_text.encode("utf-8-sig") if has_bom else output_text.encode("utf-8")
    target.write_bytes(payload)

    total_chars = len(output_text.replace("\n", ""))
    print(f"\n校验      : 通过（130 篇齐全、无重复卷号）")
    print(f"已写入    : {target}（{len(lines)} 行，{total_chars:,} 字）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
