"""คำสั่ง `campusai search "คำถาม"`: แสดงผลค้นหาของ vector, BM25 และ hybrid เทียบกัน (ไว้ debug)"""

import argparse

from campusai import config
from campusai.retrieval import bm25, vector_store
from campusai.retrieval.hybrid import Retriever, extract_clause_refs
from campusai.retrieval.indexer import load_chunks
from campusai.retrieval.vector_store import SearchResult


def _print_results(title: str, results: list[SearchResult]) -> None:
    print(f"\n== {title} ==")
    if not results:
        print("  (ไม่พบผลลัพธ์)")
    for rank, r in enumerate(results, start=1):
        preview = r.text.replace("\n", " ")[:60]
        print(f"  {rank}. [{r.score:.4f}] {r.doc} {r.clause or 'เกริ่นนำ'} | {preview}")


def run_search(args: argparse.Namespace) -> int:
    if not config.CHUNKS_PATH.exists():
        print(
            f"[campusai] ไม่พบ {config.CHUNKS_PATH} (รัน `campusai ingest` และ `campusai index` ก่อน)"
        )
        return 1

    retriever = Retriever(
        client=vector_store.get_client(), bm25_index=bm25.build_index(load_chunks())
    )
    refs = extract_clause_refs(args.query)
    print(f"คำถาม: {args.query}")
    print(f"เลขข้อที่ระบุในคำถาม: {', '.join(refs) if refs else '(ไม่มี)'}")

    _print_results("Vector (cosine)", retriever.vector(args.query, args.top_k))
    _print_results("BM25 (keyword)", retriever.keyword(args.query, args.top_k))
    _print_results("Hybrid (RRF)", retriever.hybrid(args.query, args.top_k))
    return 0
