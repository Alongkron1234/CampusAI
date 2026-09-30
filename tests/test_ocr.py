from pathlib import Path

import pytest
import requests

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


def test_cache_path_includes_backend_model_and_page(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "OCR_BACKEND", "typhoon")
    monkeypatch.setattr(config, "OCR_MODEL", "scb10x/typhoon-ocr1.5-3b")
    pdf_path = _make_fake_pdf(tmp_path)

    path = ocr.cache_path_for(pdf_path, 3)

    assert path.parent.name == "typhoon_scb10x_typhoon-ocr1.5-3b"  # "/" ถูกแทนที่แล้ว
    assert path.parent.parent.name == ocr.file_hash(pdf_path)
    assert path.name == "3.md"


def test_cache_path_uses_separate_folder_per_backend(tmp_path, monkeypatch):
    # สลับ OCR_BACKEND ต้องได้ path cache คนละอันกัน ไม่เอา cache backend อื่นมาใช้ปนกัน
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path / "cache")
    pdf_path = _make_fake_pdf(tmp_path)

    monkeypatch.setattr(config, "OCR_BACKEND", "typhoon")
    typhoon_path = ocr.cache_path_for(pdf_path, 1)

    monkeypatch.setattr(config, "OCR_BACKEND", "gemini")
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-3.8-flash")
    gemini_path = ocr.cache_path_for(pdf_path, 1)

    assert typhoon_path != gemini_path
    assert gemini_path.parent.name == "gemini_gemini-3.8-flash"


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


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json_data


def test_call_model_once_typhoon_returns_text_and_not_truncated_on_stop(monkeypatch, tmp_path):
    pdf_path = _make_fake_pdf(tmp_path)
    monkeypatch.setattr(ocr, "render_pdf_to_base64png", lambda *a, **k: "ZmFrZQ==")

    captured_payload = {}

    def _fake_post(url, json, timeout):
        captured_payload.update(json)
        return _FakeResponse({"message": {"content": "ข้อความ OCR"}, "done_reason": "stop"})

    monkeypatch.setattr(ocr.requests, "post", _fake_post)

    text, truncated = ocr._call_model_once_typhoon(pdf_path, 1, num_ctx=8192)

    assert text == "ข้อความ OCR"
    assert truncated is False
    assert captured_payload["options"]["num_ctx"] == 8192


def test_call_model_once_typhoon_reports_truncated_when_done_reason_is_length(
    monkeypatch, tmp_path
):
    pdf_path = _make_fake_pdf(tmp_path)
    monkeypatch.setattr(ocr, "render_pdf_to_base64png", lambda *a, **k: "ZmFrZQ==")
    monkeypatch.setattr(
        ocr.requests,
        "post",
        lambda url, json, timeout: _FakeResponse(
            {"message": {"content": "ข้อความไม่ครบ"}, "done_reason": "length"}
        ),
    )

    text, truncated = ocr._call_model_once_typhoon(pdf_path, 1, num_ctx=4096)

    assert text == "ข้อความไม่ครบ"
    assert truncated is True


def test_call_model_typhoon_escalates_num_ctx_until_not_truncated(monkeypatch, tmp_path):
    pdf_path = _make_fake_pdf(tmp_path)
    monkeypatch.setattr(config, "OCR_NUM_CTX_LEVELS", [1000, 2000, 4000])

    calls = []

    def _fake_call_once(path, page, num_ctx):
        calls.append(num_ctx)
        if num_ctx < 2000:
            return "ตัดกลางคัน", True
        return "ข้อความเต็ม", False

    monkeypatch.setattr(ocr, "_call_model_once_typhoon", _fake_call_once)

    result = ocr._call_model_typhoon(pdf_path, 1)

    assert result == "ข้อความเต็ม"
    assert calls == [1000, 2000]  # หยุดทันทีที่ไม่ถูกตัดแล้ว ไม่ต้องลองระดับ 4000


def test_call_model_typhoon_returns_best_effort_when_still_truncated_at_max_level(
    monkeypatch, tmp_path
):
    pdf_path = _make_fake_pdf(tmp_path)
    monkeypatch.setattr(config, "OCR_NUM_CTX_LEVELS", [1000, 2000])
    monkeypatch.setattr(
        ocr, "_call_model_once_typhoon", lambda path, page, num_ctx: ("ยังไม่ครบ", True)
    )

    result = ocr._call_model_typhoon(pdf_path, 1)

    assert result == "ยังไม่ครบ"  # ลองทุกระดับแล้วยังตัดอยู่ ก็คืนอันที่ดีที่สุดที่มี


def test_call_model_dispatches_to_typhoon_by_default(monkeypatch, tmp_path):
    pdf_path = _make_fake_pdf(tmp_path)
    monkeypatch.setattr(config, "OCR_BACKEND", "typhoon")
    monkeypatch.setattr(ocr, "_call_model_typhoon", lambda path, page: "จาก typhoon")
    monkeypatch.setattr(
        ocr, "_call_model_gemini", lambda path, page: (_ for _ in ()).throw(AssertionError())
    )

    assert ocr._call_model(pdf_path, 1) == "จาก typhoon"


def test_call_model_dispatches_to_gemini_when_configured(monkeypatch, tmp_path):
    pdf_path = _make_fake_pdf(tmp_path)
    monkeypatch.setattr(config, "OCR_BACKEND", "gemini")
    monkeypatch.setattr(ocr, "_call_model_gemini", lambda path, page: "จาก gemini")
    monkeypatch.setattr(
        ocr, "_call_model_typhoon", lambda path, page: (_ for _ in ()).throw(AssertionError())
    )

    assert ocr._call_model(pdf_path, 1) == "จาก gemini"


def test_call_model_gemini_raises_clear_error_when_api_key_missing(monkeypatch, tmp_path):
    pdf_path = _make_fake_pdf(tmp_path)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")

    try:
        ocr._call_model_gemini(pdf_path, 1)
        assert False, "ควร raise RuntimeError เมื่อไม่มี GEMINI_API_KEY"
    except RuntimeError as exc:
        assert "GEMINI_API_KEY" in str(exc)


def test_call_model_gemini_sends_image_and_prompt_to_genai_client(monkeypatch, tmp_path):
    pdf_path = _make_fake_pdf(tmp_path)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr(ocr, "render_pdf_to_base64png", lambda *a, **k: "ZmFrZQ==")

    captured = {}

    class _FakeModels:
        def generate_content(self, *, model, contents):
            captured["model"] = model
            captured["contents"] = contents

            class _Resp:
                text = "ข้อความจาก Gemini"

            return _Resp()

    class _FakeClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.models = _FakeModels()

    monkeypatch.setattr(ocr.genai, "Client", _FakeClient)

    result = ocr._call_model_gemini(pdf_path, 1)

    assert result == "ข้อความจาก Gemini"
    assert captured["api_key"] == "fake-key"
    assert captured["model"] == "gemini-3.8-flash"
    assert len(captured["contents"]) == 2


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


def _page_2_always_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ocr, "MAX_RETRIES", 1)
    monkeypatch.setattr(ocr.time, "sleep", lambda _seconds: None)

    def _call(pdf, page):
        if page == 2:
            raise ConnectionError("Ollama ไม่ตอบ")
        return f"หน้า {page}"

    monkeypatch.setattr(ocr, "_call_model", _call)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    return pdf_path


def test_ocr_pages_skip_failed_continues_to_next_pages(tmp_path, monkeypatch):
    pdf_path = _page_2_always_fails(tmp_path, monkeypatch)
    results = ocr.ocr_pages(pdf_path, [1, 2, 3], skip_failed=True)
    assert results == {1: "หน้า 1", 3: "หน้า 3"}


def test_ocr_pages_raises_on_failure_by_default(tmp_path, monkeypatch):
    pdf_path = _page_2_always_fails(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError):
        ocr.ocr_pages(pdf_path, [1, 2, 3])
