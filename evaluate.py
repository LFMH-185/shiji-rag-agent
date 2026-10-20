#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检索评测：Recall@5 / MRR + 负样本拒答检查。

评测集: eval/questions.jsonl（20 正 + 10 负）
正样本: gold 用「卷号 + 原文片段」定位，运行时解析成 chunk id
负样本:
    negative_absent    —— 实体在《史记》中不存在（三国/唐/明人物），检索应空
    negative_mismatch  —— 实体都在，但事实张冠李戴，检索层无法判伪，交给 LLM 拒答

用法:
    python evaluate.py
    python evaluate.py --verbose
    python evaluate.py --top-k 5 --json eval/report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from retrieve import (
    DEFAULT_COLLECTION,
    DEFAULT_DB,
    DEFAULT_MODEL,
    DEFAULT_CHUNKS,
    Retriever,
)

DATASET = "eval/questions.jsonl"


@dataclass
class EvalItem:
    id: str
    type: str
    question: str
    gold: list[dict]
    note: str = ""


def load_dataset(path: Path) -> list[EvalItem]:
    if not path.exists():
        raise SystemExit(f"找不到评测集 {path}")
    items: list[EvalItem] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        items.append(EvalItem(
            id=raw["id"],
            type=raw["type"],
            question=raw["question"],
            gold=raw.get("gold", []),
            note=raw.get("note", ""),
        ))
    return items


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    """Retriever 需要的完整参数。评测固定用混合检索，不启用 rerank。"""
    return argparse.Namespace(
        chunks=args.chunks,
        db=args.db,
        collection=args.collection,
        model=args.model,
        device=args.device,
        dense_only=False,
        sparse_only=False,
        dense_k=args.dense_k,
        sparse_k=args.sparse_k,
        expand_dense=True,
        neighbors=0,
        enable_rerank=False,
        reranker_model="BAAI/bge-reranker-base",
    )


def resolve_gold(item: EvalItem, chunks) -> set[str]:
    resolved: set[str] = set()
    for target in item.gold:
        volume_no = target["volume_no"]
        needle = target["contains"]
        for chunk in chunks:
            if chunk.volume_no == volume_no and needle in chunk.text:
                resolved.add(chunk.id)
    return resolved


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="《史记》检索评测")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dense-k", type=int, default=30)
    parser.add_argument("--sparse-k", type=int, default=30)
    parser.add_argument("--gate", type=float, default=0.65, help="负样本拒答的 dense 相似度阈值")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", default=None, help="把明细写到该文件")
    args = parser.parse_args(argv)

    dataset = load_dataset(Path(args.dataset))
    positives = [item for item in dataset if item.type == "positive"]
    absent = [item for item in dataset if item.type == "negative_absent"]
    mismatch = [item for item in dataset if item.type == "negative_mismatch"]

    started = time.time()
    retriever = Retriever(build_args(args))
    ready = time.time() - started

    # 解析 gold，顺便校验评测集
    gold_map: dict[str, set[str]] = {}
    unresolved: list[str] = []
    for item in positives:
        ids = resolve_gold(item, retriever.chunks)
        gold_map[item.id] = ids
        if not ids:
            unresolved.append(item.id)
    if unresolved:
        raise SystemExit(
            "以下正样本的 gold 片段在语料中找不到，请修正评测集：" + ", ".join(unresolved)
        )

    hits_at_k = 0
    hits_at_3 = 0
    reciprocal_ranks: list[float] = []
    failures: list[dict] = []
    per_item: list[dict] = []
    positive_sims: list[float] = []
    positive_bm25: list[float] = []

    for item in positives:
        results, _ = retriever.search(item.question, args.top_k)
        ids = [chunk.id for _, chunk, _ in results]
        gold = gold_map[item.id]
        rank = next((i for i, doc_id in enumerate(ids, start=1) if doc_id in gold), None)
        if rank:
            hits_at_k += 1
            if rank <= 3:
                hits_at_3 += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            failures.append({
                "id": item.id,
                "question": item.question,
                "gold": sorted(gold),
                "top1": ids[0] if ids else None,
            })

        top1_info = results[0][2] if results else {}
        top1_sim = top1_info.get("dense_sim", 0.0) or 0.0
        top1_bm25 = top1_info.get("bm25_score", 0.0) or 0.0
        positive_sims.append(top1_sim)
        positive_bm25.append(top1_bm25)
        per_item.append({
            "id": item.id, "type": "positive", "question": item.question,
            "rank": rank, "top1": ids[0] if ids else None,
            "top1_dense_sim": top1_sim, "top1_bm25": top1_bm25,
        })
        if args.verbose:
            mark = "✓" if rank else "✗"
            print(f"  {mark} {item.id} rank={rank}  {item.question}")

    # 负样本：缺失实体应被拒答；错配实体只能靠 LLM，这里只查「是否有块同时包含两个实体」
    absent_rejected = 0
    absent_sims: list[float] = []
    absent_details: list[dict] = []
    for item in absent:
        results, _ = retriever.search(item.question, args.top_k)
        top1 = results[0][2] if results else {}
        sim = top1.get("dense_sim", 0.0) or 0.0
        bm25 = top1.get("bm25_score", 0.0) or 0.0
        rejected = bm25 == 0 and sim < args.gate
        absent_rejected += int(rejected)
        absent_sims.append(sim)
        absent_details.append({
            "id": item.id, "question": item.question,
            "rejected": rejected, "top1_dense_sim": sim, "top1_bm25": bm25,
        })
        if args.verbose:
            print(f"  {'○' if rejected else '×'} {item.id} 拒答={rejected} sim={sim:.3f} bm25={bm25:.1f}  {item.question}")

    mismatch_supported = 0
    mismatch_details: list[dict] = []
    for item in mismatch:
        raw = next(raw for raw in
                   (json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip())
                   if raw["id"] == item.id)
        entities = raw.get("entities", [])
        results, _ = retriever.search(item.question, args.top_k)
        supported = any(all(entity in chunk.text for entity in entities) for _, chunk, _ in results)
        mismatch_supported += int(supported)
        mismatch_details.append({
            "id": item.id, "question": item.question,
            "supported_by_single_chunk": supported,
        })
        if args.verbose:
            print(f"  {'×' if supported else '○'} {item.id} 单块支持={supported}  {item.question}")

    total = len(positives)
    recall5 = hits_at_k / total if total else 0.0
    recall3 = hits_at_3 / total if total else 0.0
    mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    print("\n" + "=" * 56)
    print(f"检索评测    : {len(dataset)} 题（正 {total} / 负 {len(absent) + len(mismatch)}）")
    print(f"语料        : {len(retriever.chunks)} 块   RRF top-{args.top_k}")
    if positives:
        print(f"Recall@{args.top_k}    : {recall5:.3f}  ({hits_at_k}/{total})")
        print(f"Recall@3     : {recall3:.3f}  ({hits_at_3}/{total})")
        print(f"MRR          : {mrr:.3f}")
    print(f"正样本 top1 dense 相似度: 均值 {statistics.mean(positive_sims):.3f}  "
          f"中位 {statistics.median(positive_sims):.3f}")
    if absent:
        print(f"空实体负样本 : 拒答 {absent_rejected}/{len(absent)}"
              f"（bm25=0 且 dense<{args.gate}）  平均相似度 {statistics.mean(absent_sims):.3f}")
    if mismatch:
        print(f"错配负样本   : 单块同时包含实体的 {mismatch_supported}/{len(mismatch)}"
              f"（应为 0，其余必须由 LLM 拒答）")
    print(f"检索耗时     : 30 题共 {time.time() - started - ready:.2f}s（索引就绪 {ready:.1f}s）")

    if failures:
        print("\n未命中样本:")
        for item in failures:
            print(f"  {item['id']} {item['question']}\n     gold={item['gold']}  top1={item['top1']}")

    # 阈值敏感性：给 qa.py 的门控选参数
    print("\n阈值敏感性（正样本应通过 / 空实体负样本应拒答）:")
    print(f"  {'阈值':<6}{'正样本通过':>10}{'负样本拒答':>12}")
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        pos_pass = sum(1 for sim, bm25 in zip(positive_sims, positive_bm25) if sim >= threshold or bm25 > 0)
        neg_reject = sum(1 for detail in absent_details
                         if detail["top1_bm25"] == 0 and detail["top1_dense_sim"] < threshold)
        print(f"  {threshold:<6.2f}{pos_pass:>8}/{total}{neg_reject:>10}/{len(absent)}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "recall_at_k": round(recall5, 4),
            "recall_at_3": round(recall3, 4),
            "mrr": round(mrr, 4),
            "top_k": args.top_k,
            "positives": per_item,
            "negative_absent": absent_details,
            "negative_mismatch": mismatch_details,
            "gate": args.gate,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细已写入: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
