"""ตรวจสอบไฟล์ eval/questions.jsonl ว่ารูปแบบถูกต้องและอ้างอิง doc/clause ที่มีอยู่จริง

ใช้ "เอกสาร + เลขข้อ" อ้างอิงคำตอบ (ไม่ใช้ chunk id ตรง ๆ) เพื่อให้ชุดคำถามนี้ยังใช้ได้
แม้วันหลังจะไปแก้ chunker.py ทำให้วิธีตัด chunk เปลี่ยน
"""

import json
from pathlib import Path

from campusai import config

REQUIRED_TYPES = {"direct", "semantic", "multi", "unanswerable"}
REQUIRED_FIELDS = ("id", "question", "type", "expected", "reference_answer")
MIN_PER_TYPE = 10


def load_questions(path: Path = config.EVAL_QUESTIONS_PATH) -> list[dict]:
    """อ่าน eval/questions.jsonl ทีละบรรทัด คืนเป็น list ของ dict"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_eval_questions(path: Path | None, allow_test_set: bool = False) -> list[dict]:
    """โหลดชุดคำถามสำหรับคำสั่ง eval (ค่าเริ่มต้น = dev set) และกันการเผลอใช้ test set

    test set ต้องวัดครั้งเดียวหลังปรับจูนเสร็จ ถ้าเปิดดูผลระหว่างปรับจูน ตัวเลขจะ overfit เหมือน dev set
    """
    path = path or config.EVAL_QUESTIONS_PATH
    if path.resolve() == config.EVAL_TEST_QUESTIONS_PATH.resolve() and not allow_test_set:
        raise ValueError(
            "test set ถูกล็อกไว้ใช้วัดครั้งเดียวหลัง Issue #9 ถ้าตั้งใจจะวัดจริงให้ใส่ --allow-test-set"
        )
    return load_questions(path)


def load_known_clauses(chunks_path: Path = config.CHUNKS_PATH) -> set[tuple[str, str | None]]:
    """คืนเซ็ตของ (doc, clause) ที่มีอยู่จริงใน chunks.jsonl ไว้เทียบว่าคำถามอ้างอิงถูกไหม"""
    known: set[tuple[str, str | None]] = set()
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            known.add((row["doc"], row["clause"]))
    return known


def validate(questions: list[dict], known_clauses: set[tuple[str, str | None]]) -> list[str]:
    """ตรวจ list คำถามทั้งหมด คืน list ของข้อความ error (list ว่าง = ผ่านหมด)"""
    errors: list[str] = []
    seen_ids: set[str] = set()
    type_counts: dict[str, int] = dict.fromkeys(REQUIRED_TYPES, 0)

    for i, q in enumerate(questions):
        prefix = f"บรรทัด {i + 1} (id={q.get('id', '?')})"

        for field in REQUIRED_FIELDS:
            if field not in q:
                errors.append(f"{prefix}: ขาด field '{field}'")

        qid = q.get("id")
        if qid in seen_ids:
            errors.append(f"{prefix}: id ซ้ำกับข้อก่อนหน้า")
        elif qid:
            seen_ids.add(qid)

        qtype = q.get("type")
        if qtype not in REQUIRED_TYPES:
            errors.append(f"{prefix}: type {qtype!r} ไม่รู้จัก (ต้องเป็นหนึ่งใน {sorted(REQUIRED_TYPES)})")
        else:
            type_counts[qtype] += 1

        expected = q.get("expected", [])
        if qtype == "unanswerable":
            if expected:
                errors.append(
                    f"{prefix}: type 'unanswerable' ต้องมี expected เป็น [] "
                    f"แต่มี {len(expected)} รายการ"
                )
        elif not expected:
            errors.append(f"{prefix}: type '{qtype}' ต้องมี expected อย่างน้อย 1 รายการ")

        for entry in expected:
            doc = entry.get("doc")
            clause = entry.get("clause")
            if (doc, clause) not in known_clauses:
                errors.append(
                    f"{prefix}: อ้างอิง doc={doc!r} clause={clause!r} ไม่มีอยู่จริงใน chunks.jsonl"
                )

    for qtype in sorted(REQUIRED_TYPES):
        if type_counts[qtype] < MIN_PER_TYPE:
            errors.append(
                f"ประเภท '{qtype}' มีแค่ {type_counts[qtype]} ข้อ ต้องมีอย่างน้อย {MIN_PER_TYPE} ข้อ"
            )

    return errors
