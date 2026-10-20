#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《史记》问答 Web 界面（Streamlit）。

启动:
    streamlit run app.py

特性:
    - 多轮对话：保留最近几轮上下文，支持「那扶苏后来怎样了」这类追问
    - 流式输出：st.write_stream 逐字渲染
    - 检索依据可展开核对，标注卷/篇/段
    - 置信度不足直接拒答，不消耗 API

配置:
    DEEPSEEK_API_KEY 环境变量，或在侧边栏临时填写（不落盘）
"""

from __future__ import annotations

import argparse
import os

import streamlit as st

from qa import (
    DEFAULT_BASE_URL,
    DEFAULT_GATE,
    DEFAULT_LLM_MODEL,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    build_context,
    build_retriever,
    confidence_of,
    iter_chat_stream,
)
from retrieve import DEFAULT_CHUNKS, DEFAULT_COLLECTION, DEFAULT_DB, DEFAULT_MODEL

EXAMPLES = [
    ("焚书坑儒", "秦始皇焚书坑儒的具体经过"),
    ("鸿门宴", "鸿门宴上项庄做了什么"),
    ("司马迁写史记", "司马迁为什么要写《史记》"),
    ("李斯之死", "李斯最后的下场是什么"),
    ("指鹿为马", "赵高指鹿为马是怎么回事"),
]

HISTORY_TURNS = 3  # 送给大模型的历史轮数

# 面向普通用户的拒答文案：不暴露任何阈值/分数
UI_REFUSAL = "抱歉，《史记》中未记载相关信息。"


st.set_page_config(
    page_title="《史记》问答",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="加载检索模型与索引…")
def get_retriever(chunks: str, db: str, collection: str, model: str, device: str):
    namespace = argparse.Namespace(
        chunks=chunks,
        db=db,
        collection=collection,
        model=model,
        device=device,
        dense_only=False,
        sparse_only=False,
        dense_k=30,
        sparse_k=30,
        expand_dense=False,
        neighbors=0,
        enable_rerank=False,
        reranker_model="BAAI/bge-reranker-base",
    )
    return build_retriever(namespace)


def init_state() -> None:
    st.session_state.setdefault("messages", [])   # 仅用于展示
    st.session_state.setdefault("history", [])    # 送给模型的 role/content
    st.session_state.setdefault("pending", None)


def retrieve_with_history(question: str, messages: list[dict], retriever, cfg: dict):
    """两段式检索：先单独检索，只有置信度不足时才合并上一轮问题。

    直接合并并不总是更好：问「那扶苏后来怎样了」单独检索能命中《蒙恬列传》，
    拼上前文的「秦始皇焚书坑儒」后，jieba 反而把「扶苏」切成「扶/苏」匹配到《苏秦列传》。
    """
    primary, primary_meta = retriever.search(question, cfg["top_k"])
    similarity, bm25_score = confidence_of(primary) if primary else (0.0, 0.0)
    confident = similarity >= cfg["gate"] or bm25_score > 0

    previous = next(
        (item["content"] for item in reversed(messages) if item["role"] == "user"), ""
    )
    if confident or not previous or len(question) > 15:
        return question, primary, primary_meta

    merged_query = f"{previous} {question}"
    merged, merged_meta = retriever.search(merged_query, cfg["top_k"])
    merged_sim, merged_bm25 = confidence_of(merged) if merged else (0.0, 0.0)
    if merged_bm25 > bm25_score or merged_sim > similarity + 0.02:
        return merged_query, merged, merged_meta
    return question, primary, primary_meta


def render_sources(citations: list[dict], label: str = "检索依据（可核对原文）") -> None:
    if not citations:
        return
    with st.expander(f"{label} · {len(citations)} 块"):
        for item in citations:
            tag = "上下文补充" if item["is_neighbor"] else "命中"
            st.markdown(
                f"**[{item['index']}] 《史记·{item['volume']}·{item['chapter']}》"
                f"（段{item['para_start']}-{item['para_end']}，{tag}）**"
            )
            st.caption(item["text"][:400] + ("…" if len(item["text"]) > 400 else ""))


def render_tech_details(tech: dict) -> None:
    """技术细节统一折叠，普通用户默认看不到阈值/分数。"""
    if not tech:
        return
    with st.expander("检索详情（开发者）"):
        st.json(tech)


def answer_stream(question: str, context: str, history: list[dict], cfg: dict):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-HISTORY_TURNS * 2:])
    messages.append({
        "role": "user",
        "content": USER_TEMPLATE.format(context=context, question=question),
    })
    return st.write_stream(
        iter_chat_stream(
            messages,
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            model=cfg["model"],
            temperature=cfg["temperature"],
        )
    )


def handle_question(question: str, retriever, cfg: dict) -> None:
    # 必须在写入本轮消息之前取上一轮问题
    previous_messages = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question, "citations": []})
    with st.chat_message("user"):
        st.markdown(question)
    retriever.args.neighbors = cfg["neighbors"]
    query, results, meta = retrieve_with_history(question, previous_messages, retriever, cfg)
    similarity, bm25_score = confidence_of(results) if results else (0.0, 0.0)

    tech = {
        "检索词": query,
        "dense_similarity": round(similarity, 4),
        "bm25_score": round(bm25_score, 2),
        "gate": cfg["gate"],
        "chunks": len(results),
        "expansions": meta.get("expansions", []),
    }

    with st.chat_message("assistant"):
        if not results or (similarity < cfg["gate"] and bm25_score <= 0):
            # 面向普通用户：只说结果，不暴露阈值和分数
            st.markdown(UI_REFUSAL)
            render_tech_details(tech)
            st.session_state.messages.append({
                "role": "assistant", "content": UI_REFUSAL, "citations": [], "tech": tech,
            })
            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "assistant", "content": UI_REFUSAL})
            return

        context, citations = build_context(results)
        try:
            with st.spinner("DeepSeek 生成中…"):
                answer = answer_stream(question, context, st.session_state.history, cfg)
        except SystemExit as exc:  # 接口/网络错误
            st.error(str(exc))
            st.session_state.messages.append({
                "role": "assistant", "content": f"调用失败：{exc}", "citations": citations,
            })
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"生成失败：{exc}")
            return

        render_sources(citations)
        render_tech_details(tech)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "citations": citations, "tech": tech,
    })
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer})


def main() -> None:
    init_state()

    st.title("📜 《史记》问答")
    st.caption("检索增强问答 · 回答带原文引用与出处 · 查不到会明确说「未见记载」")

    with st.sidebar:
        st.subheader("模型配置")
        api_key = st.text_input(
            "DeepSeek API Key",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            type="password",
            help="也可用 setx DEEPSEEK_API_KEY 配置，这里填写后仅保存在会话内存",
        )
        model = st.text_input("模型", value=os.getenv("DEEPSEEK_MODEL", DEFAULT_LLM_MODEL))
        base_url = st.text_input("接口地址", value=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
        temperature = st.slider("温度", 0.0, 1.0, 0.3, 0.1)

        st.divider()
        st.subheader("检索配置")
        top_k = st.slider("直接命中块数", 3, 8, 5)
        neighbors = st.slider("前后邻接块", 0, 2, 1, help="每块前后各取 N 块，防止断章取义")
        gate = st.slider("拒答阈值（dense）", 0.50, 0.80, DEFAULT_GATE, 0.01)

        st.divider()
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()
        st.caption("语料：《史记》130 篇 · 2082 块 · 检索在本地 CPU 运行")

    cfg = {
        "api_key": api_key.strip(),
        "model": model.strip() or DEFAULT_LLM_MODEL,
        "base_url": base_url.strip() or DEFAULT_BASE_URL,
        "temperature": temperature,
        "top_k": top_k,
        "neighbors": neighbors,
        "gate": gate,
    }

    retriever = None
    if cfg["api_key"] or st.session_state.messages:
        retriever = get_retriever(DEFAULT_CHUNKS, DEFAULT_DB, DEFAULT_COLLECTION, DEFAULT_MODEL, "cpu")

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
                st.session_state.pending = question
                st.rerun()

    prompt = st.chat_input("问一个关于《史记》的问题…")
    if st.session_state.pending:
        prompt = st.session_state.pending
        st.session_state.pending = None

    if not prompt:
        return
    if not cfg["api_key"]:
        st.warning(
            "请先在侧边栏填写 DeepSeek API Key，或执行：\n\n"
            "`setx DEEPSEEK_API_KEY \"sk-...\"`（新开终端生效）"
        )
        return

    if retriever is None:
        retriever = get_retriever(DEFAULT_CHUNKS, DEFAULT_DB, DEFAULT_COLLECTION, DEFAULT_MODEL, "cpu")

    handle_question(prompt, retriever, cfg)


if __name__ == "__main__":
    main()
