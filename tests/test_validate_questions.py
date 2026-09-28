import pytest

from campusai import config
from campusai.evaluation.validate_questions import MIN_PER_TYPE, validate

KNOWN = {
    ("doc.pdf", "ข้อ 1"),
    ("doc.pdf", "ข้อ 2"),
    ("doc.pdf", None),  # เกริ่นนำ
}


def _valid_question(**overrides) -> dict:
    base = {
        "id": "q1",
        "question": "ข้อ 1 ว่าด้วยอะไร",
        "type": "direct",
        "expected": [{"doc": "doc.pdf", "clause": "ข้อ 1"}],
        "reference_answer": "...",
    }
    base.update(overrides)
    return base


def test_valid_question_passes():
    # ส่งคำถามแค่ 1 ข้อ จะโดนเงื่อนไข "ต้องมีอย่างน้อย 10 ข้อ/ประเภท" ด้วยเสมอ
    # เทสนี้เช็คเฉพาะว่าตัวคำถามเองไม่มี error อื่นปน (กรอง error เรื่องจำนวนออก)
    errors = validate([_valid_question()], KNOWN)
    other_errors = [e for e in errors if "ต้องมีอย่างน้อย" not in e]
    assert other_errors == []


def test_missing_field_is_reported():
    q = _valid_question()
    del q["reference_answer"]
    errors = validate([q], KNOWN)
    assert any("reference_answer" in e for e in errors)


def test_duplicate_id_is_reported():
    errors = validate([_valid_question(id="q1"), _valid_question(id="q1")], KNOWN)
    assert any("id ซ้ำ" in e for e in errors)


def test_unknown_type_is_reported():
    errors = validate([_valid_question(type="ambiguous")], KNOWN)
    assert any("ไม่รู้จัก" in e for e in errors)


def test_unanswerable_with_expected_is_reported():
    q = _valid_question(type="unanswerable", expected=[{"doc": "doc.pdf", "clause": "ข้อ 1"}])
    errors = validate([q], KNOWN)
    assert any("unanswerable" in e and "expected" in e for e in errors)


def test_unanswerable_with_empty_expected_passes():
    q = _valid_question(type="unanswerable", expected=[])
    errors = validate([q], KNOWN)
    other_errors = [e for e in errors if "ต้องมีอย่างน้อย" not in e]
    assert other_errors == []


def test_non_unanswerable_without_expected_is_reported():
    q = _valid_question(expected=[])
    errors = validate([q], KNOWN)
    assert any("ต้องมี expected อย่างน้อย 1 รายการ" in e for e in errors)


def test_reference_to_unknown_clause_is_reported():
    q = _valid_question(expected=[{"doc": "doc.pdf", "clause": "ข้อ 999"}])
    errors = validate([q], KNOWN)
    assert any("ไม่มีอยู่จริง" in e for e in errors)


def test_multi_type_can_reference_several_clauses():
    q = _valid_question(
        type="multi",
        expected=[{"doc": "doc.pdf", "clause": "ข้อ 1"}, {"doc": "doc.pdf", "clause": "ข้อ 2"}],
    )
    errors = validate([q], KNOWN)
    other_errors = [e for e in errors if "ต้องมีอย่างน้อย" not in e]
    assert other_errors == []


def test_reports_when_a_type_has_too_few_questions():
    questions = [_valid_question(id=f"q{i}") for i in range(MIN_PER_TYPE - 1)]
    errors = validate(questions, KNOWN)
    assert any("direct" in e and "ต้องมีอย่างน้อย" in e for e in errors)


def test_real_eval_questions_file_is_valid():
    # ไฟล์จริง eval/questions.jsonl commit ขึ้น git แต่ data/processed/chunks.jsonl ไม่ commit
    # (สร้างจาก PDF ที่ไม่ได้อยู่ใน repo) ดังนั้นถ้าเครื่องนี้ยังไม่เคย ingest ให้ข้ามเทสนี้ไป
    if not config.EVAL_QUESTIONS_PATH.exists() or not config.CHUNKS_PATH.exists():
        pytest.skip(
            "ต้องมี eval/questions.jsonl และ data/processed/chunks.jsonl (รัน `campusai ingest` ก่อน)"
        )

    from campusai.evaluation.validate_questions import load_known_clauses, load_questions

    questions = load_questions()
    known = load_known_clauses()
    errors = validate(questions, known)
    assert errors == [], "\n" + "\n".join(errors)
