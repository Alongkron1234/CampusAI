"""จุดรวมคำสั่งของ CampusAI

คำสั่งส่วนใหญ่ยังไม่ทำงานจริงในขั้นนี้ (Issue #1: วางโครงโปรเจกต์เท่านั้น)
แต่ละคำสั่งจะถูกเติม logic จริงใน Issue ถัดไปตามที่ระบุไว้ใน docs/PLAN.md
"""

import argparse
import sys


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
        "search": "ค้นหา chunk ที่เกี่ยวข้องกับคำถาม (Issue #6)",
        "ask": "ถามคำถามครั้งเดียวแล้วรับคำตอบ (Issue #7)",
        "chat": "โหมดถาม-ตอบต่อเนื่องใน terminal (Issue #7)",
        "eval": "รันชุดวัดผล retrieval หรือคำตอบ (Issue #6, #8)",
    }

    for name, help_text in commands.items():
        sub = subparsers.add_parser(name, help=help_text)
        sub.set_defaults(func=_not_implemented(name))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
