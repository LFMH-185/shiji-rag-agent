#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《史记》问答：混合检索 + 邻块扩展 + DeepSeek 流式作答。

特点:
    - 只用 DeepSeek（OpenAI 兼容接口），不依赖本地大模型
    - 流式输出（SSE），逐字打印
    - 上下文 = top-5 + 前后各 1 块，避免断章取义
    - 强约束 Prompt：查不到必须回「《史记》未见记载」，禁止白话脑补
    - 置信度门控：低于阈值直接拒答，不调用 API（省钱）

用法:
    setx DEEPSEEK_API_KEY "sk-..."        # 只需配置一次（新开终端生效）
    python qa.py "秦始皇焚书坑儒"
    python qa.py                          # 交互模式
    python qa.py "司马迁为什么写史记" --show-context
    python qa.py "李斯的下场" --dry-run     # 不调 API，只看检索与 Prompt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from retrieve import (
    DEFAULT_CHUNKS,
    DEFAULT_COLLECTION,
    DEFAULT_DB,
    DEFAULT_MODEL,
    Retriever,
)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-chat"
DEFAULT_GATE = 0.65
REFUSAL = "《史记》未见记载"

SYSTEM_PROMPT = """你是《史记》科普问答助手，服务对象是没读过文言的普通读者。
只能依据用户提供的【资料】回答，不得使用资料以外的知识。

【回答要求】
1. 先用通俗易懂的现代白话文直接回答问题：讲清谁、什么时候、发生了什么、结果如何。
   不要照抄文言，不要写成翻译腔，避免生僻词。
2. 关键情节必须引用 1—2 句最核心的文言原文作佐证，用「」括起，紧跟 [编号]。
   引用要短（一句即可），不要整段搬运；每处引用前后都要有白话说明它在讲什么。
3. 绝对禁止只堆文言原文而不解释。如果读者看完不知道发生了什么，就是不合格的回答。
4. 每条结论后标注资料编号 [n]；回答末尾必须列出「出处」，格式：
   [1] 《史记·卷六·秦始皇本纪》
5. 资料中没有明确记载的，只输出一行：《史记》未见记载。
   禁止用常识、后世传说、影视剧情补充，禁止编造卷号、篇名或原文。
6. 若资料互相矛盾或不足以判断，说明「资料不足」，不要强行下结论。
7. 全文以白话为主，控制在 300 字以内，可用小标题或分点，条理清晰。

【风格示例】
问：赵高指鹿为马是怎么回事？
答：秦二世时，权臣赵高想试探大臣们是否听命于自己。他故意牵一头鹿上朝，硬说这是马。
二世不信，问左右，有人沉默，有人附和说是马，也有人说是鹿。事后赵高暗中陷害了那些说真话的人，
群臣从此更加畏惧他。原文记载：「赵高欲为乱，恐群臣不听，乃先设验，持鹿献於二世，曰：『马也。』」[1]
出处：[1] 《史记·卷六·秦始皇本纪》"""

USER_TEMPLATE = """【资料】
{context}

【问题】{question}"""


# ---------------------------------------------------------------- 检索


def build_retriever(args: argparse.Namespace) -> Retriever:
    namespace = argparse.Namespace(
        chunks=args.chunks,
        db=args.db,
        collection=args.collection,
        model=args.model,
        device=args.device,
        dense_only=False,
        sparse_only=False,
        dense_k=args.dense_k,
        sparse_k=args.sparse_k,
        expand_dense=False,
        neighbors=args.neighbors,
        enable_rerank=False,
        reranker_model="BAAI/bge-reranker-base",
    )
    return Retriever(namespace)


def build_context(results) -> tuple[str, list[dict]]:
    """把检索结果（含邻块）编号成 Prompt 资料，返回 (文本, 引用表)。"""
    lines: list[str] = []
    citations: list[dict] = []
    for index, (_, chunk, info) in enumerate(results, start=1):
        tag = "（上下文补充）" if info.get("neighbor_of") else ""
        lines.append(f"[{index}] 《史记·{chunk.volume}·{chunk.chapter}》段{chunk.para_start}-{chunk.para_end}{tag}")
        lines.append(chunk.text.strip())
        lines.append("")
        citations.append({
            "index": index,
            "volume": chunk.volume,
            "chapter": chunk.chapter,
            "para_start": chunk.para_start,
            "para_end": chunk.para_end,
            "is_neighbor": bool(info.get("neighbor_of")),
            "text": chunk.text,
        })
    return "\n".join(lines).strip(), citations


def confidence_of(results) -> tuple[float, float]:
    """取首个非邻块的 dense 相似度与 BM25 分数。"""
    for _, _, info in results:
        if not info.get("neighbor_of"):
            return float(info.get("dense_sim") or 0.0), float(info.get("bm25_score") or 0.0)
    return 0.0, 0.0


# ---------------------------------------------------------------- DeepSeek


def resolve_api_key(explicit: str | None) -> str:
    key = explicit or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "缺少 API Key。请先配置环境变量（新终端生效）：\n"
            "  setx DEEPSEEK_API_KEY \"sk-...\"\n"
            "或临时指定： python qa.py \"问题\" --api-key sk-..."
        )
    return key


def endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def iter_chat_stream(
    messages: list[dict], api_key: str, base_url: str, model: str, temperature: float = 0.3
):
    """逐块产出 DeepSeek 的流式回答（给 CLI 和 Streamlit 共用）。"""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    request = urllib.request.Request(
        endpoint(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise SystemExit(f"\nDeepSeek 接口错误 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"\n网络错误，无法连接 {endpoint(base_url)}: {exc.reason}") from exc


def stream_chat(messages: list[dict], api_key: str, base_url: str, model: str,
                temperature: float = 0.3) -> str:
    """流式打印到终端，返回完整文本。"""
    pieces: list[str] = []
    for piece in iter_chat_stream(messages, api_key, base_url, model, temperature):
        pieces.append(piece)
        print(piece, end="", flush=True)
    print()
    return "".join(pieces)


def chat_once(messages: list[dict], api_key: str, base_url: str, model: str,
              temperature: float = 0.3) -> str:
    payload = {"model": model, "messages": messages, "temperature": temperature, "stream": False}
    request = urllib.request.Request(
        endpoint(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise SystemExit(f"DeepSeek 接口错误 HTTP {exc.code}: {detail}") from exc
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    print(content)
    return content


# ---------------------------------------------------------------- 主流程


def answer(args: argparse.Namespace, retriever: Retriever, question: str) -> int:
    results, meta = retriever.search(question, args.top_k, args.volume)
    if not results:
        print(REFUSAL)
        return 0

    similarity, bm25_score = confidence_of(results)
    if similarity < args.gate and bm25_score <= 0:
        print(f"{REFUSAL}")
        print(f"（检索置信度不足：dense {similarity:.3f} < {args.gate}，BM25 无命中）", file=sys.stderr)
        return 0

    context, citations = build_context(results)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]

    if args.show_context or args.dry_run:
        print("=" * 60)
        print(f"检索: {len(results)} 块（top-{args.top_k} + 前后各 {args.neighbors} 邻块）"
              f"  dense {similarity:.3f}")
        print(f"扩展: {'、'.join(meta.get('expansions') or []) or '无'}")
        print("-" * 60)
        print(context[: args.context_chars])
        print("=" * 60)

    if args.dry_run:
        print("\n[Prompt 预览]")
        print(messages[1]["content"][: args.context_chars])
        return 0

    api_key = resolve_api_key(args.api_key)
    if args.no_stream:
        answer_text = chat_once(messages, api_key, args.base_url, args.llm_model, args.temperature)
    else:
        answer_text = stream_chat(messages, api_key, args.base_url, args.llm_model, args.temperature)

    if REFUSAL in answer_text:
        return 0

    if args.json:
        print(json.dumps({
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "confidence": {"dense": round(similarity, 4), "bm25": round(bm25_score, 3)},
        }, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="《史记》问答（DeepSeek 流式）")
    parser.add_argument("question", nargs="*", help="问题；留空进入交互模式")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding 模型")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-k", type=int, default=30)
    parser.add_argument("--sparse-k", type=int, default=30)
    parser.add_argument("--neighbors", type=int, default=1, help="命中块前后各取几块，默认 1")
    parser.add_argument("--volume", type=int, default=None, help="限定卷号")
    parser.add_argument("--gate", type=float, default=DEFAULT_GATE, help="低于该 dense 相似度且 BM25 无命中则拒答")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--no-stream", action="store_true", help="关闭流式（调试用）")
    parser.add_argument("--show-context", action="store_true", help="打印检索到的资料")
    parser.add_argument("--context-chars", type=int, default=4000, help="打印上下文时截断长度")
    parser.add_argument("--dry-run", action="store_true", help="只检索并预览 Prompt，不调 API")
    parser.add_argument("--json", action="store_true", help="回答后输出 JSON 结构")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    retriever = build_retriever(args)

    question = " ".join(args.question).strip()
    if question:
        return answer(args, retriever, question)

    print(f"已就绪（{len(retriever.chunks)} 块）。输入问题回车提问，/exit 退出。")
    while True:
        try:
            line = input("\n问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"/exit", "/quit", "exit", "quit"}:
            return 0
        try:
            answer(args, retriever, line)
        except SystemExit as exc:
            print(exc, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
