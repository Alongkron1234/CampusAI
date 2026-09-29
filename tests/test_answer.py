import pytest
from google.genai import errors as genai_errors

from campusai.generation import answer
from campusai.generation.ask_cli import format_answer, format_sources
from campusai.retrieval.vector_store import SearchResult


def _r(chunk_id: str, clause: str = "ข้อ 1", text: str = "เนื้อหา") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id, doc="a.pdf", clause=clause, chapter="หมวด 1 ทั่วไป", text=text, score=0.0
    )


@pytest.fixture(autouse=True)
def _no_real_gemini(monkeypatch):
    def _fail(prompt, model):
        raise AssertionError("test ต้อง mock _call_gemini ห้ามเรียก Gemini จริง")

    def _fail_http(*args, **kwargs):
        raise AssertionError("test ต้อง mock requests.post ห้ามเรียก Ollama จริง")

    monkeypatch.setattr(answer, "_call_gemini", _fail)
    monkeypatch.setattr(answer.requests, "post", _fail_http)
    monkeypatch.setattr(answer.time, "sleep", lambda s: None)
    monkeypatch.setattr(answer.config, "LLM_BACKEND", "gemini")
    monkeypatch.setattr(answer.config, "GEMINI_MODEL", "main-model")
    monkeypatch.setattr(answer.config, "GEMINI_FALLBACK_MODEL", "")


def _mock_gemini(monkeypatch, *replies):
    """ตั้งให้ _call_gemini คืน/โยน replies ทีละตัว และเก็บรุ่นที่ถูกเรียกไว้ตรวจ"""
    queue = list(replies)
    prompts: list[str] = []

    def _fake(prompt, model):
        prompts.append(model)
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(answer, "_call_gemini", _fake)
    return prompts


def _api_error(code: int, message: str = "x") -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"message": message, "status": "x"}})


def test_daily_quota_is_not_retried_but_falls_back(monkeypatch):
    monkeypatch.setattr(answer.config, "GEMINI_FALLBACK_MODEL", "backup-model")
    daily = _api_error(429, "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    models = _mock_gemini(monkeypatch, daily, "ตอบ [1]")
    assert answer.generate_answer("q", [_r("1")]).text == "ตอบ [1]"
    assert models == ["main-model", "backup-model"]


# ----- prompt -----


def test_build_context_numbers_chunks_with_source_label():
    context = answer.build_context([_r("1", "ข้อ 5", "ห้ามพนัน"), _r("2", "ข้อ 9", "ห้ามสูบบุหรี่")])
    assert "[1] (a.pdf, ข้อ 5, หมวด 1 ทั่วไป)\nห้ามพนัน" in context
    assert "[2] (a.pdf, ข้อ 9, หมวด 1 ทั่วไป)\nห้ามสูบบุหรี่" in context


def test_source_label_skips_missing_parts():
    r = SearchResult(chunk_id="x", doc="a.pdf", clause=None, chapter=None, text="", score=0.0)
    assert answer.source_label(r) == "a.pdf"


def test_build_prompt_contains_question_and_context():
    prompt = answer.build_prompt("ลาพักได้กี่เทอม", [_r("1", text="ลาพักได้ไม่เกิน 2 ภาค")])
    assert "ลาพักได้ไม่เกิน 2 ภาค" in prompt
    assert prompt.rstrip().endswith("คำถาม: ลาพักได้กี่เทอม")


# ----- citations -----


def test_extract_citations_unique_in_order():
    assert answer.extract_citations("ก [2] ข [1][2] ค [2]", 3) == [2, 1]


def test_extract_citations_drops_numbers_out_of_range():
    assert answer.extract_citations("ก [0] ข [4] ค [3]", 3) == [3]


# ----- generate_answer -----


def test_no_results_refuses_without_calling_llm():
    result = answer.generate_answer("อะไรก็ได้", [])  # autouse fixture จะ fail ถ้าเรียก Gemini
    assert result.text == answer.REFUSAL
    assert result.refused


def test_generate_answer_keeps_only_cited_sources(monkeypatch):
    _mock_gemini(monkeypatch, "ลาพักได้ไม่เกิน 2 ภาค [2]")
    contexts = [_r("1", "ข้อ 1"), _r("2", "ข้อ 20"), _r("3", "ข้อ 30")]
    result = answer.generate_answer("ลาพักได้กี่เทอม", contexts)
    assert [s.clause for s in result.sources] == ["ข้อ 20"]
    assert result.contexts == contexts
    assert not result.refused


def test_generate_answer_refusal_from_llm(monkeypatch):
    _mock_gemini(monkeypatch, answer.REFUSAL)
    result = answer.generate_answer("ค่าหอเท่าไหร่", [_r("1")])
    assert result.refused
    assert result.sources == []


def test_retries_on_overload_then_succeeds(monkeypatch):
    prompts = _mock_gemini(monkeypatch, _api_error(503), _api_error(429), "คำตอบ [1]")
    result = answer.generate_answer("q", [_r("1")])
    assert result.text == "คำตอบ [1]"
    assert len(prompts) == 3


def test_does_not_retry_client_errors(monkeypatch):
    prompts = _mock_gemini(monkeypatch, _api_error(400), "ไม่ควรถึงตรงนี้")
    with pytest.raises(genai_errors.APIError):
        answer.generate_answer("q", [_r("1")])
    assert len(prompts) == 1


def test_gives_up_after_max_retries(monkeypatch):
    prompts = _mock_gemini(monkeypatch, *[_api_error(503)] * answer.MAX_RETRIES)
    with pytest.raises(genai_errors.APIError):
        answer.generate_answer("q", [_r("1")])
    assert len(prompts) == answer.MAX_RETRIES


def test_switches_to_fallback_model_when_main_keeps_failing(monkeypatch):
    monkeypatch.setattr(answer.config, "GEMINI_FALLBACK_MODEL", "backup-model")
    models = _mock_gemini(monkeypatch, *[_api_error(503)] * answer.MAX_RETRIES, "ตอบ [1]")
    result = answer.generate_answer("q", [_r("1")])
    assert result.text == "ตอบ [1]"
    assert models == ["main-model"] * answer.MAX_RETRIES + ["backup-model"]


def test_no_fallback_for_client_errors(monkeypatch):
    monkeypatch.setattr(answer.config, "GEMINI_FALLBACK_MODEL", "backup-model")
    models = _mock_gemini(monkeypatch, _api_error(400))
    with pytest.raises(genai_errors.APIError):
        answer.generate_answer("q", [_r("1")])
    assert models == ["main-model"]


def test_sources_sorted_by_citation_number(monkeypatch):
    _mock_gemini(monkeypatch, "ก [3] ข [1]")
    result = answer.generate_answer("q", [_r("1", "ข้อ 1"), _r("2", "ข้อ 2"), _r("3", "ข้อ 3")])
    assert [s.clause for s in result.sources] == ["ข้อ 1", "ข้อ 3"]


# ----- Ollama backend -----


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._content}, "done_reason": "stop"}


def test_ollama_backend_sends_system_prompt_and_num_ctx(monkeypatch):
    monkeypatch.setattr(answer.config, "LLM_BACKEND", "ollama")
    sent = {}

    def _post(url, json, timeout):
        sent.update(url=url, payload=json)
        return _FakeResponse("ลาได้ 2 ภาค [1]")

    monkeypatch.setattr(answer.requests, "post", _post)
    result = answer.generate_answer("ลาพักได้กี่เทอม", [_r("1", "ข้อ 20")])

    assert result.text == "ลาได้ 2 ภาค [1]"
    assert sent["url"].endswith("/api/chat")
    assert sent["payload"]["messages"][0] == {"role": "system", "content": answer.SYSTEM_PROMPT}
    assert sent["payload"]["options"]["num_ctx"] == answer.config.OLLAMA_LLM_NUM_CTX


def test_ollama_backend_strips_think_block(monkeypatch):
    monkeypatch.setattr(answer.config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(
        answer.requests, "post", lambda *a, **k: _FakeResponse("<think>คิด [2]</think>\nตอบ [1]")
    )
    result = answer.generate_answer("q", [_r("1"), _r("2")])
    assert result.text == "ตอบ [1]"
    assert len(result.sources) == 1  # [2] ในส่วน think ต้องไม่นับเป็นแหล่งอ้างอิง


def test_answer_question_uses_hybrid_top_k(monkeypatch):
    _mock_gemini(monkeypatch, "ตอบ [1]")
    calls = []

    class _Retriever:
        def hybrid(self, query, top_k):
            calls.append((query, top_k))
            return [_r("1")]

    answer.answer_question("คำถาม", _Retriever(), top_k=3)
    assert calls == [("คำถาม", 3)]


# ----- การแสดงผล (ask_cli) -----


def test_format_answer_keeps_original_citation_numbers():
    contexts = [_r("1", "ข้อ 1"), _r("2", "ข้อ 20")]
    result = answer.Answer(text="ตอบ [2]", sources=[contexts[1]], contexts=contexts)
    out = format_answer(result)
    assert "[2] a.pdf, ข้อ 20" in out
    assert "[1]" not in out


def test_format_sources_shows_full_text_or_message():
    contexts = [_r("1", "ข้อ 20", "ข้อความเต็มของข้อ 20")]
    assert "ข้อความเต็มของข้อ 20" in format_sources(
        answer.Answer(text="ตอบ [1]", sources=contexts, contexts=contexts)
    )
    assert "ไม่ได้อ้างอิง" in format_sources(answer.Answer(text=answer.REFUSAL))
