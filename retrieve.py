#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检索层：Dense（Chroma）+ BM25 + RRF 融合，纯 CPU 极速版。

Rerank 默认关闭，只有显式加 --enable-rerank 才会加载 1.1GB 的 bge-reranker-base。

用法:
    python retrieve.py "秦始皇焚书坑儒"
    python retrieve.py "项羽为什么失败" --top-k 5 --explain
    python retrieve.py "刘邦" --volume 8            # 只在卷八里找
    python retrieve.py --enable-rerank "李斯的下场"   # 需要时再开重排
    python retrieve.py --json "鸿门宴"              # 输出 JSON 供上层调用
    python retrieve.py                             # 交互模式

为什么需要 BM25:
    《史记》原文用「阬术士」而非「坑儒」，用「焚诗书」而非「焚书」。
    纯向量检索对这类现代词/文言词错位不敏感，BM25 + 异体字归一能兜住字面召回。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHUNKS = "data/chunks.jsonl"
DEFAULT_DB = "data/chroma"
DEFAULT_COLLECTION = "shiji_chunks"
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER = "BAAI/bge-reranker-base"

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
# 实测：Dense 单独 Recall@5=0.80，BM25 单独 0.40，等权 RRF 会互相拖累。
# 因此 Dense 为主、BM25 仅作补位（k=10, w_sparse=0.2）。
RRF_K = 10
RRF_WEIGHT_DENSE = 1.0
RRF_WEIGHT_SPARSE = 0.2
DENSE_K = 30
SPARSE_K = 30
TOP_K = 5

# ---------------------------------------------------------------- 字面归一
# 主库用字 → 检索用字。两边都归一后再匹配，避免「阬/坑」这类错位。
NORMALIZE_MAP = str.maketrans({
    "阬": "坑", "馀": "余", "於": "于", "適": "适", "彊": "强", "後": "后",
    "谿": "溪", "沈": "沉", "翦": "剪", "闇": "暗", "夥": "伙", "叩": "扣",
    "謫": "谪", "脩": "修", "邺": "邺", "鄴": "邺",
})

# 现代提问词 → 原文用词。只扩展 BM25 查询，不改语料。
QUERY_EXPANSIONS = {
    "焚书": ["焚诗书", "烧诗书", "诗书百家语", "烧之"],
    "坑儒": ["坑术士", "阬术士", "术士", "诸生", "四百六十"],
    "秦始皇": ["始皇", "嬴政", "秦始皇帝"],
    "始皇": ["秦始皇", "嬴政"],
    "刘邦": ["汉高祖", "沛公", "高祖"],
    "汉高祖": ["刘邦", "沛公"],
    "项羽": ["项籍", "西楚霸王"],
    "吕后": ["吕太后", "高后"],
    "司马迁": ["太史公", "太史公曰"],
    "商鞅": ["卫鞅", "公孙鞅", "商君"],
    "荆轲": ["荆卿"],
    "陈胜": ["陈涉", "陈王"],
    "陈涉": ["陈胜"],
    "匈奴": ["胡", "单于"],
    # 成语/典故：现代说法与原文用词差异很大，补上原文关键词
    "鸿门宴": ["鸿门", "项庄", "樊哙", "沛公"],
    "完璧归赵": ["和氏璧", "蔺相如", "赵王"],
    "负荆请罪": ["廉颇", "蔺相如", "肉袒"],
    "卧薪尝胆": ["句践", "勾践", "会稽"],
    "纸上谈兵": ["赵括", "长平"],
    "指鹿为马": ["赵高", "鹿", "马"],
    "破釜沉舟": ["项羽", "钜鹿", "沉船"],
    "四面楚歌": ["项羽", "楚歌", "垓下"],
    "霸王别姬": ["虞姬", "垓下", "项羽"],
    "围魏救赵": ["孙膑", "庞涓", "桂陵"],
    "退避三舍": ["晋文公", "重耳"],
    "图穷匕见": ["荆轲", "匕首", "秦王"],
    "毛遂自荐": ["毛遂", "平原君"],
    "狡兔三窟": ["冯驩", "孟尝君"],
    "一鸣惊人": ["楚庄王", "不鸣"],
    # 结局类问法：原文不会直说「下场」，而是「腰斩」「五刑」「自杀」等
    "下场": ["腰斩", "五刑", "诛", "自杀", "死"],
    "结局": ["腰斩", "五刑", "诛", "自杀", "死"],
    "怎么死": ["诛", "自杀", "死"],
    # 典故补全
    "月下追韩信": ["信亡", "追之", "何闻"],
    "追韩信": ["信亡", "追之"],
    "写史记": ["发愤", "郁结", "述往事"],
    "为什么写": ["发愤", "郁结", "述往事"],
    "变法": ["变法之令", "什伍", "连坐", "耕战"],
}


def normalize(text: str) -> str:
    return text.translate(NORMALIZE_MAP)


_PUNCT = re.compile(r"[《》〈〉“”‘’\"'（）()【】\[\]·、，。！？；：\s]")


def expand_query(query: str) -> tuple[str, list[str]]:
    """返回 (扩展后的查询串, 实际用到的同义词)。

    匹配时先去掉书名号等标点，避免「写《史记》」匹配不到「写史记」。
    """
    compact = _PUNCT.sub("", query)
    additions: list[str] = []
    for key, values in QUERY_EXPANSIONS.items():
        if key in compact:
            for value in values:
                if value not in additions and value not in query:
                    additions.append(value)
    expanded = " ".join([query, *additions]) if additions else query
    return expanded, additions


# ---------------------------------------------------------------- 数据


@dataclass
class Chunk:
    id: str
    volume_no: int
    volume: str
    chapter: str
    chapter_type: str
    chunk_index: int
    para_start: int
    para_end: int
    text: str
    embed_text: str

    @property
    def loc(self) -> str:
        return f"{self.volume}·{self.chapter} 段{self.para_start}-{self.para_end}"


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise SystemExit(f"找不到 {path}，请先运行 preprocess.py")
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            chunks.append(Chunk(
                id=item["id"],
                volume_no=item["volume_no"],
                volume=item["volume"],
                chapter=item["chapter"],
                chapter_type=item["chapter_type"],
                chunk_index=item["chunk_index"],
                para_start=item["para_start"],
                para_end=item["para_end"],
                text=item["text"],
                embed_text=item["embed_text"],
            ))
    if not chunks:
        raise SystemExit(f"{path} 是空的")
    return chunks


# ---------------------------------------------------------------- 分词 / BM25

try:
    import jieba

    jieba.setLogLevel(60)
except ImportError:  # pragma: no cover
    jieba = None

_ASCII_TOKEN = re.compile(r"[a-z0-9][a-z0-9_+#.\-]*")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    lowered = normalize(text).lower()
    tokens = _ASCII_TOKEN.findall(lowered)
    for run in _CJK_RUN.findall(lowered):
        if jieba is not None:
            tokens.extend(w for w in jieba.lcut_for_search(run) if w.strip())
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
            if len(run) == 1:
                tokens.append(run)
    return tokens


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)
        self.doc_len = [len(tokens) for tokens in corpus]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 1.0
        self.tf = [Counter(tokens) for tokens in corpus]
        df: Counter[str] = Counter()
        for counter in self.tf:
            df.update(counter.keys())
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, counter in enumerate(self.tf):
                freq = counter.get(term)
                if not freq:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[index] / self.avgdl)
                scores[index] += idf * freq * (self.k1 + 1) / (freq + norm)
        return scores


# ---------------------------------------------------------------- Dense


class DenseRetriever:
    def __init__(self, db_dir: Path, collection: str, model_name: str, device: str) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit("缺少依赖，请先执行：pip install -r requirements.txt") from exc

        # 复用 index.py 记录的实际模型路径（ModelScope 缓存），避免再去连 HuggingFace
        manifest = db_dir / "manifest.json"
        if model_name == DEFAULT_MODEL and manifest.exists():
            try:
                cached_path = json.loads(manifest.read_text(encoding="utf-8")).get("model_path")
                if cached_path and Path(cached_path).exists():
                    model_name = cached_path
            except json.JSONDecodeError:
                pass

        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512
        self.client = chromadb.PersistentClient(
            path=str(db_dir), settings=Settings(anonymized_telemetry=False)
        )
        try:
            self.collection = self.client.get_collection(collection)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"读不到集合 {collection}，请先运行 python index.py（错误：{exc}）"
            ) from exc

    def search(self, query: str, top_k: int, volume: int | None = None) -> list[tuple[str, float]]:
        vector = self.model.encode(
            [QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
        )
        result = self.collection.query(
            query_embeddings=vector.tolist(),
            n_results=top_k,
            where={"volume_no": volume} if volume is not None else None,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [(doc_id, 1 - distance) for doc_id, distance in zip(ids, distances)]


# ---------------------------------------------------------------- Rerank（可选）


class Reranker:
    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import CrossEncoder

        print(f"[rerank] 加载 {model_name}（首次会下载约 1.1GB）…", file=sys.stderr)
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(self, query: str, candidates: list[tuple[float, Chunk]], top_k: int) -> list[tuple[float, Chunk]]:
        if not candidates:
            return []
        pairs = [(query, chunk.text) for _, chunk in candidates]
        scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)
        combined = [(float(score), chunk) for score, (_, chunk) in zip(scores, candidates)]
        combined.sort(key=lambda item: item[0], reverse=True)
        return combined[:top_k]


# ---------------------------------------------------------------- 融合


def rrf_fuse(
    dense_hits: list[tuple[str, float]],
    sparse_hits: list[tuple[str, float]],
    chunks_by_id: dict[str, Chunk],
    top_k: int,
) -> list[tuple[float, Chunk, dict]]:
    dense_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_hits, start=1)}
    sparse_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(sparse_hits, start=1)}
    dense_score = dict(dense_hits)
    sparse_score = dict(sparse_hits)

    fused: dict[str, float] = {}
    for doc_id, rank in dense_rank.items():
        fused[doc_id] = fused.get(doc_id, 0.0) + RRF_WEIGHT_DENSE / (RRF_K + rank)
    for doc_id, rank in sparse_rank.items():
        fused[doc_id] = fused.get(doc_id, 0.0) + RRF_WEIGHT_SPARSE / (RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
    results: list[tuple[float, Chunk, dict]] = []
    for doc_id, score in ordered:
        chunk = chunks_by_id.get(doc_id)
        if chunk is None:
            continue
        results.append((score, chunk, {
            "dense_rank": dense_rank.get(doc_id),
            "dense_sim": round(dense_score.get(doc_id, 0.0), 4),
            "bm25_rank": sparse_rank.get(doc_id),
            "bm25_score": round(sparse_score.get(doc_id, 0.0), 3),
        }))
    return results


# ---------------------------------------------------------------- 主检索


class Retriever:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.chunks = load_chunks(Path(args.chunks))
        self.chunks_by_id = {chunk.id: chunk for chunk in self.chunks}

        started = time.time()
        self.bm25 = None
        if not args.dense_only:
            self.bm25 = BM25([tokenize(chunk.text) for chunk in self.chunks])
        self.bm25_build_time = time.time() - started

        self.dense = None
        if not args.sparse_only:
            self.dense = DenseRetriever(
                Path(args.db), args.collection, args.model, args.device
            )

        self.reranker = None
        if args.enable_rerank:
            self.reranker = Reranker(args.reranker_model, args.device)

    def search(self, query: str, top_k: int, volume: int | None = None) -> tuple[list[tuple[float, Chunk, dict]], dict]:
        expanded, additions = expand_query(query)
        dense_hits: list[tuple[str, float]] = []
        sparse_hits: list[tuple[str, float]] = []

        if self.dense is not None:
            search_query = expanded if self.args.expand_dense else query
            dense_hits = self.dense.search(search_query, self.args.dense_k, volume)

        if self.bm25 is not None:
            scores = self.bm25.scores(tokenize(expanded))
            ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
            sparse_hits = [
                (self.chunks[i].id, scores[i]) for i in ranked[: self.args.sparse_k] if scores[i] > 0
            ]

        pool = max(top_k * 4, top_k)
        fused = rrf_fuse(dense_hits, sparse_hits, self.chunks_by_id, pool)
        if volume is not None:
            fused = [item for item in fused if item[1].volume_no == volume]

        if self.reranker is not None:
            fused = self.reranker.rerank(expanded, [(score, chunk) for score, chunk, _ in fused], top_k)
            fused = [(score, chunk, {"reranked": True}) for score, chunk in fused]
        else:
            fused = fused[:top_k]

        meta = {"expanded_query": expanded, "expansions": additions}
        if self.args.neighbors:
            fused = self._add_neighbors(fused, self.args.neighbors)
        return fused, meta

    def _add_neighbors(self, results, count: int):
        """把命中块前后各 count 块附在后面，避免断章取义；不跨篇。"""
        index = {(chunk.volume_no, chunk.chunk_index): chunk for chunk in self.chunks}
        seen = {chunk.id for _, chunk, _ in results}
        extra = []
        for score, chunk, _ in results:
            for step in range(1, count + 1):
                for delta in (-step, step):
                    candidate = index.get((chunk.volume_no, chunk.chunk_index + delta))
                    if (candidate and candidate.chapter == chunk.chapter
                            and candidate.id not in seen):
                        seen.add(candidate.id)
                        extra.append((score * 0.5, candidate, {"neighbor_of": chunk.id}))
        return results + extra


# ---------------------------------------------------------------- 输出


def print_results(results, meta: dict, explain: bool, snippet: int = 110) -> None:
    if meta.get("expansions"):
        print(f"查询扩展: +{'、'.join(meta['expansions'])}")
    if not results:
        print("未检索到相关内容。")
        return
    for rank, (score, chunk, info) in enumerate(results, start=1):
        if info.get("reranked"):
            header = f"rerank {score:.3f}"
        elif info.get("neighbor_of"):
            header = f"邻接块 ← {info['neighbor_of']}"
        else:
            header = f"RRF {score:.4f}"
        print(f"\n[{rank}] {chunk.loc}  {header}")
        if explain and not info.get("neighbor_of"):
            print(f"     dense #{info.get('dense_rank')} (sim {info.get('dense_sim')}) | "
                  f"bm25 #{info.get('bm25_rank')} (score {info.get('bm25_score')})")
        print(f"     {chunk.text[:snippet]}…")


def to_json(results, meta: dict) -> str:
    payload = {
        "query": meta.get("query"),
        "expanded_query": meta.get("expanded_query"),
        "expansions": meta.get("expansions", []),
        "results": [
            {
                "id": chunk.id,
                "volume_no": chunk.volume_no,
                "volume": chunk.volume,
                "chapter": chunk.chapter,
                "chapter_type": chunk.chapter_type,
                "para_start": chunk.para_start,
                "para_end": chunk.para_end,
                "score": round(score, 6),
                **info,
                "text": chunk.text,
            }
            for score, chunk, info in results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="《史记》混合检索（Dense + BM25 + RRF）")
    parser.add_argument("question", nargs="*", help="问题；留空进入交互模式")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding 模型（本地路径或名字）")
    parser.add_argument("--device", default="cpu", help="默认 cpu")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--dense-k", type=int, default=DENSE_K)
    parser.add_argument("--sparse-k", type=int, default=SPARSE_K)
    parser.add_argument("--volume", type=int, default=None, help="限定卷号，如 6")
    parser.add_argument("--dense-only", action="store_true", help="只跑向量检索")
    parser.add_argument("--sparse-only", action="store_true", help="只跑 BM25")
    parser.add_argument("--expand-dense", action="store_true",
                        help="把同义词也拼进向量查询（实测会降低排序质量，默认不开）")
    parser.add_argument("--neighbors", type=int, default=0, help="附上命中块的后续 N 个邻接块")
    parser.add_argument("--enable-rerank", action="store_true", help="开启重排（1.1GB 模型，默认关闭）")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    parser.add_argument("--explain", action="store_true", help="显示 dense/bm25 排名")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    started = time.time()
    retriever = Retriever(args)
    ready = time.time() - started

    def run(question: str) -> None:
        begin = time.time()
        results, meta = retriever.search(question, args.top_k, args.volume)
        elapsed = (time.time() - begin) * 1000
        meta["query"] = question
        if args.json:
            print(to_json(results, meta))
        else:
            print_results(results, meta, args.explain)
            print(f"\n检索耗时 {elapsed:.0f} ms（索引就绪 {ready:.1f}s："
                  f"BM25 构建 {retriever.bm25_build_time:.1f}s）")

    question = " ".join(args.question).strip()
    if question:
        run(question)
        return 0

    print(f"已加载 {len(retriever.chunks)} 块（就绪 {ready:.1f}s）。输入问题回车检索，/exit 退出。")
    while True:
        try:
            line = input("问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"/exit", "/quit", "exit", "quit"}:
            return 0
        run(line)


if __name__ == "__main__":
    raise SystemExit(main())
