import json

from campusai import config
from campusai.cli import main
from campusai.ingest import pipeline
from campusai.ingest.chunker import Chunk
from campusai.ingest.pdf_text import PageResult


def _fake_pdf(tmp_path, name="doc.pdf"):
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4 fake")
    return path


def test_join_pages_in_order_sorts_by_page_number():
    page_texts = {2: "หน้าสอง", 1: "หน้าหนึ่ง", 3: "หน้าสาม"}
    result = pipeline._join_pages_in_order(page_texts)
    assert result == "หน้าหนึ่ง\n\nหน้าสอง\n\nหน้าสาม"


def test_ingest_single_file_merges_plain_text_and_ocr_result(tmp_path, monkeypatch):
    pdf_path = _fake_pdf(tmp_path)

    fake_pages = [
        PageResult(
            page_number=1,
            text="ข้อ 1 เนื้อหาปกติจาก pdfplumber",
            needs_ocr=False,
            reason="ข้อความใช้ได้",
        ),
        PageResult(page_number=2, text="", needs_ocr=True, reason="สั้นเกินไป"),
    ]
    monkeypatch.setattr(pipeline, "extract_pages", lambda path: fake_pages)
    monkeypatch.setattr(
        pipeline,
        "ocr_pages",
        lambda path, page_numbers, force=False, on_progress=None: {
            2: "ข้อ 2 เนื้อหาจาก OCR"
        },
    )

    chunks = pipeline.ingest_single_file(pdf_path)
    all_text = " ".join(c.text for c in chunks)

    assert "เนื้อหาปกติจาก pdfplumber" in all_text
    assert "เนื้อหาจาก OCR" in all_text


def test_ingest_single_file_only_calls_ocr_for_flagged_pages(tmp_path, monkeypatch):
    pdf_path = _fake_pdf(tmp_path)
    fake_pages = [
        PageResult(page_number=1, text="ข้อ 1 ปกติ", needs_ocr=False, reason="ข้อความใช้ได้"),
        PageResult(page_number=2, text="ข้อ 2 ปกติ", needs_ocr=False, reason="ข้อความใช้ได้"),
    ]
    monkeypatch.setattr(pipeline, "extract_pages", lambda path: fake_pages)

    received_page_numbers = []

    def _spy_ocr_pages(path, page_numbers, force=False, on_progress=None):
        received_page_numbers.append(page_numbers)
        return {}

    monkeypatch.setattr(pipeline, "ocr_pages", _spy_ocr_pages)

    chunks = pipeline.ingest_single_file(pdf_path)

    # ไม่มีหน้าไหน needs_ocr=True เลย -> ต้องส่ง list ว่างเข้า ocr_pages (ไม่มีหน้าไหนถูกส่งไป OCR จริง)
    assert received_page_numbers == [[]]
    assert len(chunks) == 2


def test_write_chunks_is_idempotent_and_preserves_other_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "chunks.jsonl")

    doc_a_v1 = [Chunk(id="a_1", doc="a.pdf", clause="ข้อ 1", chapter=None, text="เก่า")]
    doc_b = [Chunk(id="b_1", doc="b.pdf", clause="ข้อ 1", chapter=None, text="ไม่แตะ")]
    pipeline.write_chunks({"a.pdf": doc_a_v1, "b.pdf": doc_b})

    # ingest "a.pdf" ใหม่ด้วยเนื้อหาต่าง (เช่น PDF ถูกแทนที่) — ต้องแทนที่ของเก่า ไม่ซ้ำซ้อน
    doc_a_v2 = [Chunk(id="a_1", doc="a.pdf", clause="ข้อ 1", chapter=None, text="ใหม่")]
    pipeline.write_chunks({"a.pdf": doc_a_v2})

    saved_text = config.CHUNKS_PATH.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in saved_text.splitlines()]
    by_doc = {row["doc"]: row for row in rows}

    assert len(rows) == 2  # ไม่ใช่ 3 (ไม่ซ้ำซ้อน)
    assert by_doc["a.pdf"]["text"] == "ใหม่"
    assert by_doc["b.pdf"]["text"] == "ไม่แตะ"  # ไฟล์ที่ไม่ได้ ingest รอบนี้ยังอยู่ครบ


def test_write_debug_markdown_includes_clause_and_chapter(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    chunks = [
        Chunk(id="doc_1", doc="doc.pdf", clause="ข้อ 1", chapter="หมวด 1 บททั่วไป", text="เนื้อหา")
    ]

    out_path = pipeline.write_debug_markdown("doc.pdf", chunks)
    content = out_path.read_text(encoding="utf-8")

    assert "## ข้อ 1" in content
    assert "หมวด 1 บททั่วไป" in content
    assert "เนื้อหา" in content


def test_run_ingest_returns_error_when_no_pdf_found(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)

    exit_code = main(["ingest"])

    assert exit_code == 1
    assert "ไม่พบไฟล์ PDF" in capsys.readouterr().out


def test_run_ingest_only_flag_filters_to_selected_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "processed" / "chunks.jsonl")

    _fake_pdf(tmp_path, "keep.pdf")
    _fake_pdf(tmp_path, "skip.pdf")

    processed_docs = []

    def _fake_ingest_single_file(pdf_path, force_ocr=False, verbose=False):
        processed_docs.append(pdf_path.name)
        return [Chunk(id="x", doc=pdf_path.name, clause=None, chapter=None, text="เนื้อหา")]

    monkeypatch.setattr(pipeline, "ingest_single_file", _fake_ingest_single_file)

    exit_code = main(["ingest", "--only", "keep.pdf"])

    assert exit_code == 0
    assert processed_docs == ["keep.pdf"]
