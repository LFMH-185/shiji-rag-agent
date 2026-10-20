#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 chunks.jsonl 向量化并写入 Chroma。

模型: BAAI/bge-small-zh-v1.5（512 维，CPU 可跑）
存储: data/chroma（PersistentClient，cosine 距离）

用法:
    python index.py                       # 建库（已是最新则跳过）
    python index.py --rebuild             # 强制重建
    python index.py --search "焚书坑儒"    # 建库后做一次检索自测
    python index.py --limit 100           # 只索引前 100 块，快速验证链路
    python index.py --model-path D:/models/bge-small-zh-v1.5

设计说明:
    - 用 embed_text（带「《史记》卷X·篇名：」前缀）做向量，document 存干净的 text
    - 元数据保留卷号/篇名/篇型/段落/字符偏移，供检索时过滤与引用
    - manifest.json 记录 chunks 指纹，chunks.jsonl 没变就不重复编码
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

DEFAULT_CHUNKS = "data/chunks.jsonl"
DEFAULT_DB = "data/chroma"
DEFAULT_COLLECTION = "shiji_chunks"
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_BATCH = 32

# bge-zh 系列的查询指令前缀，只加在 query 侧
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

METADATA_FIELDS = (
    "volume_no", "volume", "chapter", "chapter_type",
    "chunk_index", "para_start", "para_end",
    "char_start", "char_end", "is_taishigong",
)


# ---------------------------------------------------------------- 数据


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"找不到 {path}，请先运行 python preprocess.py")
    chunks: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    if not chunks:
        raise SystemExit(f"{path} 是空的")
    return chunks


def fingerprint(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(db_dir: Path) -> Path:
    return db_dir / "manifest.json"


def read_manifest(db_dir: Path) -> dict:
    path = manifest_path(db_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_manifest(db_dir: Path, payload: dict) -> None:
    db_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(db_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------- 模型


def resolve_model(model_name: str, model_path: str | None) -> str:
    """优先用本地路径；否则依次尝试 HuggingFace 缓存、ModelScope 下载。"""
    if model_path:
        candidate = Path(model_path)
        if not candidate.exists():
            raise SystemExit(f"--model-path 不存在：{candidate}")
        return str(candidate)

    local = Path("models") / model_name.split("/")[-1]
    if local.exists():
        return str(local)

    errors: list[str] = []
    try:
        from modelscope import snapshot_download  # 国内可达

        print(f"本地没有模型，尝试从 ModelScope 下载 {model_name} …")
        downloaded = snapshot_download(model_name)
        print(f"下载完成: {downloaded}")
        return downloaded
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ModelScope: {exc}")

    try:
        from sentence_transformers import SentenceTransformer

        print(f"尝试 HuggingFace 直连 {model_name} …")
        SentenceTransformer(model_name)
        return model_name
    except Exception as exc:  # noqa: BLE001
        errors.append(f"HuggingFace: {exc}")

    raise SystemExit(
        "模型获取失败，请手动下载后重试：\n"
        f"  python -c \"from modelscope import snapshot_download; print(snapshot_download('{model_name}'))\"\n"
        "  python index.py --model-path <上面输出的目录>\n"
        "详细错误：\n  - " + "\n  - ".join(errors)
    )


def load_embedder(model_ref: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖，请先安装：\n"
            "  pip install torch sentence-transformers chromadb modelscope"
        ) from exc

    print(f"加载模型: {model_ref}（device={device}）")
    model = SentenceTransformer(model_ref, device=device)
    model.max_seq_length = 512
    return model


def pick_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------- 入库


def open_collection(db_dir: Path, name: str, rebuild: bool):
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise SystemExit("缺少 chromadb，请执行：pip install chromadb") from exc

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(db_dir), settings=Settings(anonymized_telemetry=False)
    )
    if rebuild:
        try:
            client.delete_collection(name)
            print(f"已删除旧集合 {name}")
        except Exception:  # noqa: BLE001 - 集合不存在时忽略
            pass
    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def to_metadata(chunk: dict) -> dict:
    return {key: chunk[key] for key in METADATA_FIELDS if key in chunk}


def index_chunks(collection, chunks: list[dict], embedder, batch_size: int) -> int:
    total = len(chunks)
    started = time.time()
    done = 0
    for offset in range(0, total, batch_size):
        batch = chunks[offset:offset + batch_size]
        texts = [chunk["embed_text"] for chunk in batch]
        vectors = embedder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            embeddings=vectors.tolist(),
            documents=[chunk["text"] for chunk in batch],
            metadatas=[to_metadata(chunk) for chunk in batch],
        )
        done += len(batch)
        elapsed = time.time() - started
        speed = done / elapsed if elapsed else 0
        eta = (total - done) / speed if speed else 0
        print(f"\r  进度 {done}/{total} ({done / total:5.1%})  {speed:5.1f} 块/秒  剩余约 {eta:4.0f}s",
              end="", flush=True)
    print()
    return total


# ---------------------------------------------------------------- 自测


def smoke_search(collection, embedder, query: str, top_k: int) -> None:
    vector = embedder.encode(
        [QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
    )
    result = collection.query(
        query_embeddings=vector.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    print(f"\n检索自测: {query}")
    for rank, (doc_id, document, meta, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        similarity = 1 - distance
        print(f"  [{rank}] {meta['volume']}·{meta['chapter']} 相似度 {similarity:.3f}  ({doc_id})")
        print(f"      {document[:60]}…")


# ---------------------------------------------------------------- 入口


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="《史记》文本块向量化入库")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS, help="chunks.jsonl 路径")
    parser.add_argument("--db", default=DEFAULT_DB, help="Chroma 持久化目录")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="集合名")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Embedding 模型")
    parser.add_argument("--model-path", default=None, help="本地模型目录（离线时用）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="每批编码条数")
    parser.add_argument("--rebuild", action="store_true", help="删掉旧集合重建")
    parser.add_argument("--limit", type=int, default=0, help="只索引前 N 块（调试用）")
    parser.add_argument("--search", default=None, help="建库后检索一句话做自测")
    parser.add_argument("--top-k", type=int, default=3, help="自测返回条数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    chunks_path = Path(args.chunks)
    db_dir = Path(args.db)

    all_chunks = load_chunks(chunks_path)
    if args.limit:
        all_chunks = all_chunks[: args.limit]
    chunks_sha = fingerprint(chunks_path)
    manifest = read_manifest(db_dir)
    up_to_date = (
        not args.rebuild
        and not args.limit
        and manifest.get("chunks_sha1") == chunks_sha
        and manifest.get("chunks") == len(all_chunks)
        and manifest.get("model") == args.model_name
    )

    if up_to_date:
        print(f"索引已是最新：{len(all_chunks)} 块，模型 {args.model_name}")
        print(f"如需强制重建：python index.py --rebuild")
        if args.search:
            cached_path = manifest.get("model_path")
            model_ref = cached_path if cached_path and Path(cached_path).exists() else args.model_name
            embedder = load_embedder(resolve_model(model_ref, args.model_path), pick_device(args.device))
            collection = open_collection(db_dir, args.collection, rebuild=False)
            smoke_search(collection, embedder, args.search, args.top_k)
        return 0

    device = pick_device(args.device)
    model_ref = resolve_model(args.model_name, args.model_path)
    embedder = load_embedder(model_ref, device)
    dimension = embedder.get_sentence_embedding_dimension()

    # 走到这里说明内容有变（或显式 --rebuild），整库重建最稳妥：
    # 旧集合的向量维度/模型可能不同，混写会出问题
    collection = open_collection(db_dir, args.collection, rebuild=True)
    print(f"集合      : {args.collection}（cosine）")
    print(f"待索引    : {len(all_chunks)} 块 × {dimension} 维")
    print(f"数据库    : {db_dir}")

    index_chunks(collection, all_chunks, embedder, args.batch_size)

    write_manifest(db_dir, {
        "collection": args.collection,
        "model": args.model_name,
        "model_path": model_ref,
        "dimension": dimension,
        "chunks": len(all_chunks),
        "chunks_sha1": chunks_sha,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"完成      : 集合内 {collection.count()} 条向量")

    if args.search:
        smoke_search(collection, embedder, args.search, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
