"""คำสั่ง `campusai ask "คำถาม"` และ `campusai chat` (ถามต่อเนื่อง โหลดโมเดล/index ครั้งเดียว)"""

import argparse

from campusai import config
from campusai.generation.answer import Answer, answer_question, source_label
from campusai.retrieval import bm25, vector_store
from campusai.retrieval.hybrid import Retriever
from campusai.retrieval.indexer import load_chunks

CHAT_HELP = "พิมพ์คำถามได้เลย | /sources ดูข้อความต้นฉบับของคำตอบล่าสุด | /exit ออก"


def _load_retriever() -> Retriever | None:
    if not config.CHUNKS_PATH.exists():
        print(
            f"[campusai] ไม่พบ {config.CHUNKS_PATH} (รัน `campusai ingest` และ `campusai index` ก่อน)"
        )
        return None
    if config.LLM_BACKEND == "gemini" and not config.GEMINI_API_KEY:
        print("[campusai] ยังไม่ได้ตั้งค่า GEMINI_API_KEY ใน .env")
        return None
    return Retriever(client=vector_store.get_client(), bm25_index=bm25.build_index(load_chunks()))


def format_answer(answer: Answer) -> str:
    lines = [answer.text]
    if answer.sources:
        lines.append("\nแหล่งอ้างอิง:")
        # เลข [n] ต้องตรงกับที่คำตอบใช้ จึงหาเลขเดิมจาก contexts ไม่นับใหม่
        for r in answer.sources:
            lines.append(f"  [{answer.contexts.index(r) + 1}] {source_label(r)}")
    return "\n".join(lines)


def format_sources(answer: Answer) -> str:
    if not answer.sources:
        return "(คำตอบล่าสุดไม่ได้อ้างอิงเอกสาร)"
    blocks = [
        f"[{answer.contexts.index(r) + 1}] {source_label(r)}\n{r.text.strip()}"
        for r in answer.sources
    ]
    return "\n\n".join(blocks)


def run_ask(args: argparse.Namespace) -> int:
    retriever = _load_retriever()
    if retriever is None:
        return 1
    try:
        answer = answer_question(args.question, retriever, top_k=args.top_k)
    except Exception as exc:  # noqa: BLE001 - แสดงข้อความสั้น ๆ แทน traceback ยาว
        print(f"[campusai] เรียก LLM ไม่สำเร็จ: {exc}")
        return 1
    print(format_answer(answer))
    return 0


def run_chat(args: argparse.Namespace) -> int:
    retriever = _load_retriever()
    if retriever is None:
        return 1

    print(f"CampusAI chat — {CHAT_HELP}")
    last: Answer | None = None
    while True:
        try:
            line = input("\nคำถาม> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/sources":
            print(format_sources(last) if last else "(ยังไม่มีคำตอบ)")
            continue
        if line.startswith("/"):
            print(CHAT_HELP)
            continue

        # แต่ละคำถามค้นหาใหม่แยกกัน ยังไม่จำบริบทคำถามก่อนหน้า
        try:
            last = answer_question(line, retriever, top_k=args.top_k)
        except Exception as exc:  # noqa: BLE001 - แจ้ง error แล้วถามต่อได้ ไม่ให้หลุดออกจาก chat
            print(f"[campusai] เรียก LLM ไม่สำเร็จ: {exc}")
            continue
        print(format_answer(last))
