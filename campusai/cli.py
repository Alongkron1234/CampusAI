"""จุดรวมคำสั่งของ CampusAI

คำสั่งส่วนใหญ่ยังไม่ทำงานจริงในขั้นนี้ (Issue #1: วางโครงโปรเจกต์เท่านั้น)
แต่ละคำสั่งจะถูกเติม logic จริงใน Issue ถัดไปตามที่ระบุไว้ใน docs/PLAN.md
"""

import argparse
import sys
from pathlib import Path

from campusai.ingest.check import add_check_args, run_check
from campusai.ingest.pipeline import add_ingest_args, run_ingest


def _run_index(args: argparse.Namespace) -> int:
    # import ตอนเรียกจริง ไม่ import ไว้บนสุดของไฟล์ เพราะ indexer ดึง torch/sentence-transformers
    # ซึ่งโหลดช้า ไม่อยากให้คำสั่งอื่น (check, ingest, --help) ต้องรอไปด้วย
    from campusai.retrieval.indexer import run_index

    return run_index(args)


def _run_serve(args: argparse.Namespace) -> int:
    from campusai.api.app import run_serve

    return run_serve(args)


def _run_remove(args: argparse.Namespace) -> int:
    from campusai.retrieval.indexer import run_remove

    return run_remove(args)


def _run_search(args: argparse.Namespace) -> int:
    from campusai.retrieval.search_cli import run_search

    return run_search(args)


def _run_ask(args: argparse.Namespace) -> int:
    from campusai.generation.ask_cli import run_ask

    return run_ask(args)


def _run_chat(args: argparse.Namespace) -> int:
    from campusai.generation.ask_cli import run_chat

    return run_chat(args)


def _add_top_k_arg(parser: argparse.ArgumentParser) -> None:
    # ค่าเดียวกับ generation.answer.TOP_K_CONTEXT (ไม่ import ตรง ๆ เพื่อให้ --help ไม่ต้องโหลด genai)
    parser.add_argument(
        "-k", "--top-k", type=int, default=5, help="จำนวน chunk ที่ส่งให้ LLM (ค่าเริ่มต้น 5)"
    )


def _run_eval(args: argparse.Namespace) -> int:
    if args.target == "retrieval":
        from campusai.evaluation.retrieval_eval import run_eval_retrieval

        return run_eval_retrieval(args)
    from campusai.evaluation.answer_eval import run_eval_answers

    return run_eval_answers(args)


def _not_implemented(name: str):
    def _run(args: argparse.Namespace) -> int:
        print(f"[campusai] คำสั่ง '{name}' ยังไม่ได้ implement (ดู docs/PLAN.md)")
        return 1

    return _run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="campusai",
        description="CampusAI: ระบบถาม-ตอบระเบียบมหาวิทยาลัยภาษาไทยแบบ RAG",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands = {
        "check": "ตรวจ PDF รายหน้าว่าต้อง OCR หรือไม่ (Issue #2)",
        "ingest": "แปลง PDF -> chunks.jsonl (Issue #2, #3)",
        "index": "สร้าง vector index + BM25 index จาก chunks.jsonl (Issue #5)",
        "remove": "ลบเอกสารออกจาก chunks.jsonl และ index (Issue #10a)",
        "serve": "เปิด Web API (FastAPI) สำหรับหน้าเว็บ (Issue #10b)",
        "search": "ค้นหา chunk ที่เกี่ยวข้องกับคำถาม (Issue #6)",
        "ask": "ถามคำถามครั้งเดียวแล้วรับคำตอบ (Issue #7)",
        "chat": "โหมดถาม-ตอบต่อเนื่องใน terminal (Issue #7)",
        "eval": "รันชุดวัดผล retrieval หรือคำตอบ (Issue #6, #8)",
    }

    for name, help_text in commands.items():
        sub = subparsers.add_parser(name, help=help_text)
        if name == "check":
            add_check_args(sub)
            sub.set_defaults(func=run_check)
        elif name == "ingest":
            add_ingest_args(sub)
            sub.set_defaults(func=run_ingest)
        elif name == "index":
            sub.add_argument(
                "--only", nargs="+", metavar="ไฟล์",
                help="index ใหม่เฉพาะเอกสารที่ระบุ (ไม่ใส่ = สร้างใหม่ทั้งหมด)",
            )
            sub.set_defaults(func=_run_index)
        elif name == "serve":
            sub.add_argument("--host", default="127.0.0.1", help="ค่าเริ่มต้น 127.0.0.1 (เครื่องนี้เท่านั้น)")
            sub.add_argument("--port", type=int, default=8000)
            sub.set_defaults(func=_run_serve)
        elif name == "remove":
            sub.add_argument("docs", nargs="+", metavar="ไฟล์", help="ชื่อไฟล์ .pdf ที่จะลบ")
            sub.add_argument(
                "--delete-pdf", action="store_true", help="ลบไฟล์ PDF ต้นฉบับใน data/raw ด้วย"
            )
            sub.set_defaults(func=_run_remove)
        elif name == "search":
            sub.add_argument("query", help="คำถามที่จะค้นหา")
            sub.add_argument("-k", "--top-k", type=int, default=5, help="จำนวนผลต่อวิธี (ค่าเริ่มต้น 5)")
            sub.set_defaults(func=_run_search)
        elif name == "ask":
            sub.add_argument("question", help="คำถาม")
            _add_top_k_arg(sub)
            sub.set_defaults(func=_run_ask)
        elif name == "chat":
            _add_top_k_arg(sub)
            sub.set_defaults(func=_run_chat)
        elif name == "eval":
            sub.add_argument("target", choices=["retrieval", "answers"], help="สิ่งที่จะวัดผล")
            sub.add_argument(
                "--questions", type=Path, default=None,
                help="ไฟล์ชุดคำถาม (ค่าเริ่มต้น eval/questions.jsonl = dev set)",
            )
            sub.add_argument(
                "--allow-test-set", action="store_true",
                help="ยอมให้ใช้ test set ที่ล็อกไว้ (ใช้ครั้งเดียวหลัง Issue #9 เท่านั้น)",
            )
            sub.add_argument("--limit", type=int, default=0, help="วัดแค่ N ข้อแรก (ไว้ลองระบบ)")
            sub.set_defaults(func=_run_eval)
        else:
            sub.set_defaults(func=_not_implemented(name))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
