#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《史记》问答 Web 界面（面向普通用户）。

设计原则：
    - 打开即用：没有任何配置项，API Key 只从环境变量读取
    - 不引导用户填 Key：缺少配置时只显示「系统维护中」
    - 回答 = 白话解释 + 原文引用 + 卷篇出处；查不到就明确说未记载
    - 提问后 2 秒冷却，防止连点刷请求

启动:
    setx DEEPSEEK_API_KEY "sk-..."      # 只需一次
    python -m streamlit run app.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import streamlit as st

from qa import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    build_context,
    build_retriever,
    confidence_of,
    iter_chat_stream,
)
from retrieve import DEFAULT_CHUNKS, DEFAULT_COLLECTION, DEFAULT_DB, DEFAULT_MODEL

# ---------------------------------------------------------------- 固定配置
# 产品化：接口、模型、检索参数全部固定，不对用户暴露
API_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"
TEMPERATURE = 0.3

TOP_K = 5
NEIGHBORS = 1
GATE = 0.65
COOLDOWN_SECONDS = 2.0

HISTORY_TURNS = 3

# ---------------------------------------------------------------- 文案
MAINTENANCE_TEXT = "系统维护中，请稍后再试。"
REFUSAL_TEXT = "抱歉，《史记》中未记载相关信息。"
ERROR_TEXT = "系统开小差了，请稍后再试。"

EXAMPLES = [
    ("鸿门宴", "鸿门宴上项庄做了什么"),
    ("焚书坑儒", "秦始皇焚书坑儒的具体经过"),
    ("指鹿为马", "赵高指鹿为马是怎么回事"),
    ("李斯之死", "李斯最后的下场是什么"),
    ("司马迁写史记", "司马迁为什么要写《史记》"),
]

ABOUT_TEXT = """**《史记》** 是中国第一部纪传体通史，西汉司马迁撰，
记载了从黄帝到汉武帝约三千年的历史，共一百三十篇：
本纪十二、表十、书八、世家三十、列传七十。

本项目把全文做成可检索的知识库：先在本地从原文中找出最相关的段落，
再让 AI 依据这些段落作答，因此每句结论都能回溯到卷、篇和原文。
"""

USAGE_TEXT = """1. 直接在下方输入问题，例如「鸿门宴」「商鞅变法」
2. 回答先用白话讲清来龙去脉，再引一两句原文作证
3. 点击「检索依据」可以核对 AI 参考的原文段落
4. 《史记》没有记载的，会明确回复「未记载」，不会编造
"""


# ---------------------------------------------------------------- 资源


@st.cache_resource(show_spinner="正在准备《史记》知识库…")
def get_retriever():
    namespace = argparse.Namespace(
        chunks=DEFAULT_CHUNKS,
        db=DEFAULT_DB,
        collection=DEFAULT_COLLECTION,
        model=DEFAULT_MODEL,
        device="cpu",
        dense_only=False,
        sparse_only=False,
        dense_k=30,
        sparse_k=30,
        expand_dense=False,
        neighbors=NEIGHBORS,
        enable_rerank=False,
        reranker_model="BAAI/bge-reranker-base",
    )
    return build_retriever(namespace)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("cooldown_until", 0.0)


def retrieval_query(question: str, previous_messages: list[dict]) -> str:
    """短追问（「那扶苏呢」）拼上一轮问题；两段式检索在下面判断是否采用。"""
    if len(question) > 15:
        return question
    previous = next(
        (item["content"] for item in reversed(previous_messages) if item["role"] == "user"), ""
    )
    if not previous or previous == question:
        return question
    return f"{previous} {question}"


def retrieve(question: str, previous_messages: list[dict], retriever):
    """先单独检索；置信度不足时才合并上一轮问题再试一次。"""
    primary, primary_meta = retriever.search(question, TOP_K)
    similarity, bm25_score = confidence_of(primary) if primary else (0.0, 0.0)
    confident = similarity >= GATE or bm25_score > 0

    merged = retrieval_query(question, previous_messages)
    if confident or merged == question:
        return question, primary, primary_meta

    merged_results, merged_meta = retriever.search(merged, TOP_K)
    merged_sim, merged_bm25 = confidence_of(merged_results) if merged_results else (0.0, 0.0)
    if merged_bm25 > bm25_score or merged_sim > similarity + 0.02:
        return merged, merged_results, merged_meta
    return question, primary, primary_meta


# ---------------------------------------------------------------- 渲染


def render_sources(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"检索依据 · {len(citations)} 段原文"):
        for item in citations:
            tag = "上下文" if item["is_neighbor"] else "命中"
            st.markdown(
                f"**[{item['index']}] 《史记·{item['volume']}·{item['chapter']}》**"
                f"（段{item['para_start']}-{item['para_end']}，{tag}）"
            )
            st.caption(item["text"][:400] + ("…" if len(item["text"]) > 400 else ""))


def render_tech_details(tech: dict) -> None:
    """开发者信息，默认折叠；普通用户不展开就看不到。"""
    if not tech:
        return
    with st.expander("检索详情（开发者）"):
        st.json(tech)


def answer_stream(question: str, context: str, history: list[dict], api_key: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-HISTORY_TURNS * 2:])
    messages.append({
        "role": "user",
        "content": USER_TEMPLATE.format(context=context, question=question),
    })
    return st.write_stream(
        iter_chat_stream(
            messages,
            api_key=api_key,
            base_url=API_BASE_URL,
            model=LLM_MODEL,
            temperature=TEMPERATURE,
        )
    )


# ---------------------------------------------------------------- 对话


def handle_question(question: str, retriever, api_key: str) -> None:
    previous_messages = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question, "citations": []})
    with st.chat_message("user"):
        st.markdown(question)

    retriever.args.neighbors = NEIGHBORS
    query, results, meta = retrieve(question, previous_messages, retriever)
    similarity, bm25_score = confidence_of(results) if results else (0.0, 0.0)
    tech = {
        "检索词": query,
        "dense_similarity": round(similarity, 4),
        "bm25_score": round(bm25_score, 2),
        "gate": GATE,
        "chunks": len(results),
        "expansions": meta.get("expansions", []),
    }

    with st.chat_message("assistant"):
        # 检索置信度不足：本地拒答，不调用 API
        if not results or (similarity < GATE and bm25_score <= 0):
            st.markdown(REFUSAL_TEXT)
            render_tech_details(tech)
            _remember(question, REFUSAL_TEXT, [], tech)
            return

        context, citations = build_context(results)
        try:
            with st.spinner("正在查阅《史记》…"):
                answer = answer_stream(question, context, st.session_state.history, api_key)
        except SystemExit as exc:
            print(f"[qa] DeepSeek 调用失败: {exc}", file=sys.stderr)
            st.markdown(ERROR_TEXT)
            tech["error"] = str(exc)[:300]
            render_tech_details(tech)
            _remember(question, ERROR_TEXT, [], tech)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[qa] 生成失败: {exc}", file=sys.stderr)
            st.markdown(ERROR_TEXT)
            tech["error"] = str(exc)[:300]
            render_tech_details(tech)
            _remember(question, ERROR_TEXT, [], tech)
            return

        render_sources(citations)
        render_tech_details(tech)

    _remember(question, answer, citations, tech)


def _remember(question: str, answer: str, citations: list[dict], tech: dict) -> None:
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "citations": citations, "tech": tech,
    })
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------- 页面


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 关于《史记》")
        st.markdown(ABOUT_TEXT)
        st.divider()
        st.markdown("### 使用说明")
        st.markdown(USAGE_TEXT)
        st.divider()
        st.markdown("### 数据范围")
        st.caption("130 篇 · 约 62.5 万字 · 本纪 / 表 / 书 / 世家 / 列传")
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.session_state.cooldown_until = 0.0
            st.rerun()
        st.caption("回答由 AI 依据《史记》原文生成，请以引用原文为准。")


def main() -> None:
    init_state()
    st.set_page_config(
        page_title="《史记》问答",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    api_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        # 不给用户任何配置指引，只提示维护中；运维信息写日志
        print("[app] DEEPSEEK_API_KEY 未配置，拒绝服务", file=sys.stderr)
        st.title("📜 《史记》问答")
        st.info(MAINTENANCE_TEXT)
        st.stop()

    st.title("📜 《史记》问答")
    st.caption("白话讲解 · 原文佐证 · 出处可查")

    render_sidebar()

    retriever = get_retriever()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("citations"):
                    render_sources(message["citations"])
                if message.get("tech"):
                    render_tech_details(message["tech"])

    if not st.session_state.messages:
        st.markdown("**可以这样问：**")
        columns = st.columns(3)
        for index, (label, question) in enumerate(EXAMPLES):
            if columns[index % 3].button(label, use_container_width=True):
                st.session_state.pending_question = question
                st.session_state.cooldown_until = time.time() + COOLDOWN_SECONDS
                st.rerun()

    cooling = time.time() < st.session_state.cooldown_until
    prompt = st.chat_input("问一个关于《史记》的问题…", disabled=cooling)
    if cooling:
        st.caption("请稍候片刻，正在准备回答…")

    # 收到新问题：先进入冷却并立即重绘，让输入框在生成期间保持禁用
    if prompt:
        st.session_state.pending_question = prompt
        st.session_state.cooldown_until = time.time() + COOLDOWN_SECONDS
        st.rerun()

    question = st.session_state.pending_question
    if not question:
        return

    st.session_state.pending_question = None
    handle_question(question, retriever, api_key)

    # 冷却结束后重绘，恢复输入
    remaining = st.session_state.cooldown_until - time.time()
    if remaining > 0:
        time.sleep(remaining)
    st.rerun()


if __name__ == "__main__":
    main()
