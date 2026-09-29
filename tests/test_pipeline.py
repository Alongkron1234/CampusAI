import json

from campusai import config
from campusai.cli import main
from campusai.ingest import pipeline, registry
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
        lambda path, page_numbers, force=False, on_progress=None, skip_failed=False: {
            2: "ข้อ 2 เนื้อหาจาก OCR"
        },
    )

    result = pipeline.ingest_single_file(pdf_path)
    all_text = " ".join(c.text for c in result.chunks)

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

    def _spy_ocr_pages(path, page_numbers, force=False, on_progress=None, skip_failed=False):
        received_page_numbers.append(page_numbers)
        return {}

    monkeypatch.setattr(pipeline, "ocr_pages", _spy_ocr_pages)

    result = pipeline.ingest_single_file(pdf_path)

    # ไม่มีหน้าไหน needs_ocr=True เลย -> ต้องส่ง list ว่างเข้า ocr_pages (ไม่มีหน้าไหนถูกส่งไป OCR จริง)
    assert received_page_numbers == [[]]
    assert len(result.chunks) == 2


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

    def _fake_ingest_single_file(pdf_path, force_ocr=False, verbose=False, on_progress=None):
        processed_docs.append(pdf_path.name)
        chunk = Chunk(id="x", doc=pdf_path.name, clause=None, chapter=None, text="เนื้อหา")
        return pipeline.IngestResult(doc=pdf_path.name, chunks=[chunk], pages=1)

    monkeypatch.setattr(pipeline, "ingest_single_file", _fake_ingest_single_file)

    exit_code = main(["ingest", "--only", "keep.pdf"])

    assert exit_code == 0
    assert processed_docs == ["keep.pdf"]


# ----- Issue #10a: ข้ามหน้าที่พัง, บันทึกทีละไฟล์, ทะเบียน, ลบเอกสาร -----


def _use_tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "processed" / "chunks.jsonl")


def _fake_result(pdf_path, force_ocr=False, verbose=False, on_progress=None):
    chunk = Chunk(id=f"{pdf_path.stem}_1", doc=pdf_path.name, clause="ข้อ 1", chapter=None,
                  text="เนื้อหา")
    return pipeline.IngestResult(doc=pdf_path.name, chunks=[chunk], pages=1)


def test_failed_ocr_page_is_skipped_and_reported(tmp_path, monkeypatch):
    pdf_path = _fake_pdf(tmp_path)
    fake_pages = [
        PageResult(page_number=1, text="", needs_ocr=True, reason="สแกน"),
        PageResult(page_number=2, text="", needs_ocr=True, reason="สแกน"),
    ]
    monkeypatch.setattr(pipeline, "extract_pages", lambda path: fake_pages)
    received = {}

    def _ocr(path, page_numbers, force=False, on_progress=None, skip_failed=False):
        received["skip_failed"] = skip_failed
        return {1: "ข้อ 1 ได้จาก OCR"}  # หน้า 2 OCR ไม่สำเร็จ

    monkeypatch.setattr(pipeline, "ocr_pages", _ocr)
    result = pipeline.ingest_single_file(pdf_path)

    assert received["skip_failed"] is True
    assert result.failed_pages == [2]
    assert result.ocr_pages == [1, 2]
    assert "ได้จาก OCR" in result.chunks[0].text


def test_ingest_reports_progress_stages(tmp_path, monkeypatch):
    pdf_path = _fake_pdf(tmp_path)
    monkeypatch.setattr(
        pipeline, "extract_pages",
        lambda path: [PageResult(page_number=1, text="", needs_ocr=True, reason="สแกน")],
    )

    def _ocr(path, page_numbers, force=False, on_progress=None, skip_failed=False):
        on_progress(1, 1, False)
        return {1: "ข้อ 1 ก"}

    monkeypatch.setattr(pipeline, "ocr_pages", _ocr)
    events = []
    pipeline.ingest_single_file(pdf_path, on_progress=lambda *e: events.append(e))

    assert ("ocr", 1, 1) in events
    assert [e[0] for e in events][0] == "extract"
    assert events[-1] == ("chunk", 1, 1)


def test_process_file_writes_chunks_and_registry(tmp_path, monkeypatch):
    _use_tmp_dirs(tmp_path, monkeypatch)
    pdf_path = _fake_pdf(tmp_path, "a.pdf")
    monkeypatch.setattr(pipeline, "ingest_single_file", _fake_result)

    pipeline.process_file(pdf_path)

    assert json.loads(config.CHUNKS_PATH.read_text("utf-8"))["doc"] == "a.pdf"
    record = registry.load_registry()["a.pdf"]
    assert record.chunks == 1
    assert registry.find_by_hash(record.sha256) == "a.pdf"


def test_one_broken_file_does_not_stop_others(tmp_path, monkeypatch, capsys):
    _use_tmp_dirs(tmp_path, monkeypatch)
    _fake_pdf(tmp_path, "bad.pdf")
    _fake_pdf(tmp_path, "good.pdf")

    def _ingest(pdf_path, *args, **kwargs):
        if pdf_path.name == "bad.pdf":
            raise RuntimeError("PDF เสีย")
        return _fake_result(pdf_path)

    monkeypatch.setattr(pipeline, "ingest_single_file", _ingest)

    assert main(["ingest"]) == 1  # มีไฟล์พัง ต้องแจ้งว่าไม่สำเร็จทั้งหมด
    assert "bad.pdf" in capsys.readouterr().out
    lines = config.CHUNKS_PATH.read_text("utf-8").splitlines()
    saved_docs = {json.loads(line)["doc"] for line in lines}
    assert saved_docs == {"good.pdf"}  # ไฟล์ที่ดียังถูกบันทึก


def test_remove_document_clears_chunks_markdown_and_registry(tmp_path, monkeypatch):
    _use_tmp_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "ingest_single_file", _fake_result)
    for name in ("a.pdf", "b.pdf"):
        pipeline.process_file(_fake_pdf(tmp_path, name))

    pipeline.remove_document("a.pdf")

    rows = [json.loads(line) for line in config.CHUNKS_PATH.read_text("utf-8").splitlines()]
    assert {r["doc"] for r in rows} == {"b.pdf"}
    assert not (config.PROCESSED_DIR / "a.md").exists()
    assert set(registry.load_registry()) == {"b.pdf"}
    assert (tmp_path / "a.pdf").exists()  # ไม่ลบ PDF ถ้าไม่สั่ง

    pipeline.remove_document("b.pdf", delete_pdf=True)
    assert not (tmp_path / "b.pdf").exists()
