from pathlib import Path

from campusai import config
from campusai.ingest import ocr


def _make_fake_pdf(tmp_path: Path, content: bytes = b"%PDF-1.4 fake content") -> Path:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(content)
    return pdf_path


def test_file_hash_is_deterministic(tmp_path):
    pdf_path = _make_fake_pdf(tmp_path)
    assert ocr.file_hash(pdf_path) == ocr.file_hash(pdf_path)


def test_file_hash_changes_when_content_changes(tmp_path):
    pdf_a = _make_fake_pdf(tmp_path, b"aaa")
    pdf_b = tmp_path / "other.pdf"
    pdf_b.write_bytes(b"bbb")
    assert ocr.file_hash(pdf_a) != ocr.file_hash(pdf_b)


def test_cache_path_includes_hash_model_and_page(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "OCR_MODEL", "scb10x/typhoon-ocr1.5-3b")
    pdf_path = _make_fake_pdf(tmp_path)

    path = ocr.cache_path_for(pdf_path, 3)

    assert path.parent.name == "scb10x_typhoon-ocr1.5-3b"  # "/" ถูกแทนที่แล้ว
    assert path.parent.parent.name == ocr.file_hash(pdf_path)
    assert path.name == "3.md"


def test_ocr_page_uses_cache_without_calling_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path / "cache")
    pdf_path = _make_fake_pdf(tmp_path)

    cache_path = ocr.cache_path_for(pdf_path, 1)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("ข้อความจาก cache เดิม", encoding="utf-8")

    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError("ไม่ควรเรียกโมเดลซ้ำเมื่อมี cache อยู่แล้ว")

    monkeypatch.setattr(ocr, "_call_model", _should_not_be_called)

    result = ocr.ocr_page(pdf_path, 1)

    assert result == "ข้อความจาก cache เดิม"


def test_ocr_page_calls_model_and_writes_cache_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path / "cache")
    pdf_path = _make_fake_pdf(tmp_path)

    monkeypatch.setattr(ocr, "_call_model", lambda pdf, page: f"ผล OCR หน้า {page}")

    result = ocr.ocr_page(pdf_path, 2)

    assert result == "ผล OCR หน้า 2"
    cache_path = ocr.cache_path_for(pdf_path, 2)
    assert cache_path.read_text(encoding="utf-8") == "ผล OCR หน้า 2"


def test_ocr_page_force_ignores_existing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path / "cache")
    pdf_path = _make_fake_pdf(tmp_path)

    cache_path = ocr.cache_path_for(pdf_path, 1)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("ของเก่า", encoding="utf-8")

    monkeypatch.setattr(ocr, "_call_model", lambda pdf, page: "ของใหม่")

    result = ocr.ocr_page(pdf_path, 1, force=True)

    assert result == "ของใหม่"
    assert cache_path.read_text(encoding="utf-8") == "ของใหม่"


def test_retry_succeeds_after_transient_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ocr, "RETRY_BASE_DELAY_SECONDS", 0)  # ไม่ต้องรอจริงตอน test
    monkeypatch.setattr(ocr.time, "sleep", lambda _seconds: None)

    pdf_path = _make_fake_pdf(tmp_path)

    call_count = {"n": 0}

    def _flaky(pdf, page):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError("Ollama ไม่ตอบ")
        return "สำเร็จหลัง retry"

    monkeypatch.setattr(ocr, "_call_model", _flaky)

    result = ocr.ocr_page(pdf_path, 1)

    assert result == "สำเร็จหลัง retry"
    assert call_count["n"] == 3


def test_retry_raises_after_exhausting_all_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ocr, "RETRY_BASE_DELAY_SECONDS", 0)
    monkeypatch.setattr(ocr, "MAX_RETRIES", 2)
    monkeypatch.setattr(ocr.time, "sleep", lambda _seconds: None)

    pdf_path = _make_fake_pdf(tmp_path)

    def _always_fails(pdf, page):
        raise ConnectionError("Ollama ล่ม")

    monkeypatch.setattr(ocr, "_call_model", _always_fails)

    try:
        ocr.ocr_page(pdf_path, 1)
        assert False, "ควร raise RuntimeError เมื่อ retry ครบแล้วยังไม่สำเร็จ"
    except RuntimeError as exc:
        assert "ล้มเหลวครบ 2 ครั้ง" in str(exc)


def test_ocr_pages_reports_progress_and_cache_status(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path / "cache")
    pdf_path = _make_fake_pdf(tmp_path)

    # หน้า 1 มี cache อยู่แล้ว, หน้า 2 ยังไม่มี (ต้องเรียกโมเดล)
    cache_path_1 = ocr.cache_path_for(pdf_path, 1)
    cache_path_1.parent.mkdir(parents=True)
    cache_path_1.write_text("cache หน้า 1", encoding="utf-8")

    monkeypatch.setattr(ocr, "_call_model", lambda pdf, page: f"OCR หน้า {page}")

    progress_log = []
    result = ocr.ocr_pages(
        pdf_path,
        [1, 2],
        on_progress=lambda i, total, cached: progress_log.append((i, total, cached)),
    )

    assert result == {1: "cache หน้า 1", 2: "OCR หน้า 2"}
    assert progress_log == [(1, 2, True), (2, 2, False)]
