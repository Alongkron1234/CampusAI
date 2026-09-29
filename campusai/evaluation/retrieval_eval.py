"""วัดผลการค้นหา (Hit@k, MRR) ด้วยชุดคำถาม eval/questions.jsonl

เทียบ 3 แบบ: vector อย่างเดียว, BM25 อย่างเดียว, และ hybrid
ไม่นับคำถามประเภท unanswerable (ไม่มีเฉลยให้เทียบ เป็นงานของการวัดคำตอบใน Issue #8)

นับอันดับแบบ "ไม่ซ้ำข้อ": ข้อยาวที่ถูกซอยเป็นหลาย chunk (เช่น ข้อ 10 มี 18 ชิ้น) นับเป็นอันดับเดียว
ไม่งั้นข้อยาวข้อเดียวจะกินหลายอันดับ ทำให้ Hit@k ดูแย่เกินจริง
"""

import argparse
import json
from collections.abc import Callable
from datetime import datetime

from campusai import config
from campusai.retrieval.vector_store import SearchResult

KS = (1, 3, 5)
SEARCH_DEPTH = 20  # ดึงกี่อันดับต่อคำถาม (ลึกพอให้คำนวณ MRR ได้ แม้เฉลยอยู่ลึก)


def first_hit_rank(
    results: list[SearchResult], expected: set[tuple[str, str | None]]
) -> int | None:
    """อันดับแรก (นับแบบไม่ซ้ำข้อ เริ่มที่ 1) ที่เจอข้อใดข้อหนึ่งในเฉลย ไม่เจอคืน None"""
    seen: list[tuple[str, str | None]] = []
    for r in results:
        key = (r.doc, r.clause)
        if key in seen:
            continue
        seen.append(key)
        if key in expected:
            return len(seen)
    return None


def summarize(ranks: list[int | None]) -> dict[str, float]:
    """แปลงอันดับที่เจอเฉลยของแต่ละคำถาม เป็น Hit@k และ MRR"""
    n = len(ranks)
    metrics = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) for k in KS}
    metrics["mrr"] = sum(1 / r for r in ranks if r is not None) / n if n else 0.0
    metrics["n"] = n
    return metrics


def evaluate(
    questions: list[dict], search_fn: Callable[[str, int], list[SearchResult]]
) -> dict[str, dict[str, float]]:
    """รันคำถามทุกข้อผ่าน search_fn แล้วสรุปผลแยกตามประเภท และรวมทั้งหมด ("all")"""
    ranks_by_type: dict[str, list[int | None]] = {}
    for q in questions:
        if q["type"] == "unanswerable":
            continue
        expected = {(e["doc"], e["clause"]) for e in q["expected"]}
        rank = first_hit_rank(search_fn(q["question"], SEARCH_DEPTH), expected)
        ranks_by_type.setdefault(q["type"], []).append(rank)

    report = {qtype: summarize(ranks) for qtype, ranks in sorted(ranks_by_type.items())}
    report["all"] = summarize([r for ranks in ranks_by_type.values() for r in ranks])
    return report


def format_report(reports: dict[str, dict[str, dict[str, float]]]) -> str:
    """ทำตาราง markdown เทียบทุกวิธี แสดงจำนวนข้อดิบคู่เปอร์เซ็นต์ (ชุดวัดเล็ก ห้ามดูแค่ %)"""
    lines = ["| วิธี | ประเภท | n | " + " | ".join(f"Hit@{k}" for k in KS) + " | MRR |"]
    lines.append("|" + "---|" * (len(KS) + 4))
    for method, report in reports.items():
        for qtype, m in report.items():
            n = int(m["n"])
            hits = " | ".join(f"{int(m[f'hit@{k}'])}/{n} ({m[f'hit@{k}'] / n:.0%})" for k in KS)
            lines.append(f"| {method} | {qtype} | {n} | {hits} | {m['mrr']:.3f} |")
    return "\n".join(lines)


def run_eval_retrieval(args: argparse.Namespace) -> int:
    from campusai.evaluation.validate_questions import load_eval_questions
    from campusai.retrieval import bm25, vector_store
    from campusai.retrieval.hybrid import Retriever
    from campusai.retrieval.indexer import load_chunks

    try:
        questions = load_eval_questions(args.questions, allow_test_set=args.allow_test_set)
    except ValueError as exc:
        print(f"[campusai] {exc}")
        return 1
    if args.limit:
        questions = questions[: args.limit]

    if not config.CHUNKS_PATH.exists():
        print(
            f"[campusai] ไม่พบ {config.CHUNKS_PATH} (รัน `campusai ingest` และ `campusai index` ก่อน)"
        )
        return 1

    retriever = Retriever(
        client=vector_store.get_client(), bm25_index=bm25.build_index(load_chunks())
    )
    reports = {
        "vector": evaluate(questions, retriever.vector),
        "bm25": evaluate(questions, retriever.keyword),
        "hybrid": evaluate(questions, retriever.hybrid),
    }
    table = format_report(reports)
    print(table)

    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = config.EVAL_RESULTS_DIR / f"retrieval-{stamp}.md"
    md_path.write_text(f"# Retrieval eval {stamp}\n\n{table}\n", encoding="utf-8")
    json_path = config.EVAL_RESULTS_DIR / f"retrieval-{stamp}.json"
    json_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nบันทึกผลไว้ที่ {md_path}")
    return 0
