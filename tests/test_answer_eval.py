import pytest

from campusai.evaluation import answer_eval
from campusai.evaluation.answer_eval import EvalRow
from campusai.generation import answer as gen
from campusai.retrieval.vector_store import SearchResult


def _r(clause: str, doc: str = "a.pdf") -> SearchResult:
    return SearchResult(chunk_id=clause, doc=doc, clause=clause, chapter=None, text="t", score=0.0)


class _Retriever:
    def __init__(self, results):
        self.results = results

    def hybrid(self, query, top_k):
        return self.results[:top_k]


@pytest.fixture
def llm(monkeypatch, tmp_path):
    """แทน LLM จริงด้วยตารางคำตอบ: ตอบคำถามด้วย replies["answer"] และกรรมการด้วย replies["judge"]"""
    monkeypatch.setattr(answer_eval.config, "ANSWER_EVAL_CACHE_DIR", tmp_path)
    replies = {
        "answer": "ตอบ [1]",
        "judge": '{"correctness": 4, "faithfulness": 5, "reason": "ok"}',
    }
    calls: list[str] = []

    def _fake(prompt, system=gen.SYSTEM_PROMPT):
        kind = "judge" if system == answer_eval.JUDGE_SYSTEM_PROMPT else "answer"
        calls.append(kind)
        return replies[kind]

    monkeypatch.setattr(gen, "call_llm", _fake)
    return replies, calls


def _q(qtype="semantic", clause="ข้อ 1"):
    expected = [] if qtype == "unanswerable" else [{"doc": "a.pdf", "clause": clause}]
    return {"id": "q1", "type": qtype, "question": "ถาม", "expected": expected,
            "reference_answer": "เฉลย"}


# ----- parse_judgement -----


def test_parse_judgement_reads_json_inside_extra_text():
    j = answer_eval.parse_judgement('ผลคือ\n{"correctness": 3, "faithfulness": 5, "reason": "ขาด"}')
    assert (j.correctness, j.faithfulness, j.reason) == (3, 5, "ขาด")


@pytest.mark.parametrize(
    "text",
    ["ไม่มี json", '{"correctness": 9, "faithfulness": 5}', '{"correctness": "x"}', "{bad json}"],
)
def test_parse_judgement_rejects_bad_output(text):
    assert answer_eval.parse_judgement(text) is None


# ----- evaluate_question -----


def test_answered_question_gets_judged_and_citations_recorded(llm):
    row = answer_eval.evaluate_question(_q(), _Retriever([_r("ข้อ 1"), _r("ข้อ 2")]))
    assert not row.refused
    assert row.cited == [("a.pdf", "ข้อ 1")]
    assert (row.correctness, row.faithfulness) == (4, 5)


def test_false_refusal_scores_correctness_1_without_judge(llm):
    replies, calls = llm
    replies["answer"] = gen.REFUSAL
    row = answer_eval.evaluate_question(_q(), _Retriever([_r("ข้อ 1")]))
    assert row.refused and row.correctness == 1
    assert calls == ["answer"]


def test_unanswerable_is_not_judged(llm):
    _, calls = llm
    row = answer_eval.evaluate_question(_q("unanswerable"), _Retriever([_r("ข้อ 1")]))
    assert row.correctness is None
    assert calls == ["answer"]


def test_no_retrieval_results_refuses_without_llm(llm):
    _, calls = llm
    row = answer_eval.evaluate_question(_q("unanswerable"), _Retriever([]))
    assert row.refused
    assert calls == []


def test_bad_judge_output_is_recorded_not_crashing(llm):
    replies, _ = llm
    replies["judge"] = "ขอโทษ ให้คะแนนไม่ได้"
    row = answer_eval.evaluate_question(_q(), _Retriever([_r("ข้อ 1")]))
    assert row.correctness is None
    assert row.errors


def test_rerun_uses_cache_instead_of_calling_llm(llm):
    _, calls = llm
    retriever = _Retriever([_r("ข้อ 1")])
    answer_eval.evaluate_question(_q(), retriever)
    answer_eval.evaluate_question(_q(), retriever)
    assert calls == ["answer", "judge"]  # รอบสองมาจาก cache ทั้งหมด


def test_cache_invalidated_when_retrieved_context_changes(llm):
    _, calls = llm
    answer_eval.evaluate_question(_q(), _Retriever([_r("ข้อ 1")]))
    answer_eval.evaluate_question(_q(), _Retriever([_r("ข้อ 2"), _r("ข้อ 1")]))
    assert calls.count("answer") == 2


# ----- summarize -----


def _row(qtype, refused, cited=(), expected=(), correctness=None, faithfulness=None):
    return EvalRow(
        id="x", type=qtype, question="q", answer="a", refused=refused, cited=list(cited),
        expected=list(expected), correctness=correctness, faithfulness=faithfulness,
    )


def test_summarize_refusal_and_citation_metrics():
    e1, e2 = ("a.pdf", "ข้อ 1"), ("a.pdf", "ข้อ 2")
    rows = [
        _row("semantic", False, cited=[e1, e2], expected=[e1], correctness=5, faithfulness=5),
        _row("semantic", False, cited=[e2], expected=[e1], correctness=3, faithfulness=4),
        _row("semantic", True, expected=[e1], correctness=1),  # ปฏิเสธทั้งที่มีคำตอบ
        _row("unanswerable", True),  # ปฏิเสธถูก
        _row("unanswerable", False),  # แต่งคำตอบ
    ]
    s = answer_eval.summarize(rows)
    assert s["refusal_accuracy"] == pytest.approx(3 / 5)
    assert (s["correct_refusals"], s["hallucinated_answers"], s["false_refusals"]) == (1, 1, 1)
    assert s["citation_hit"] == pytest.approx(1 / 2)
    assert s["citation_precision"] == pytest.approx(1 / 3)
    assert s["correctness"] == pytest.approx((5 + 3 + 1) / 3)
    assert s["faithfulness"] == pytest.approx(4.5)


def test_format_report_lists_failures():
    rows = [
        _row("semantic", False, correctness=5, faithfulness=5),
        _row("unanswerable", False),
    ]
    meta = {"stamp": "s", "llm": "ollama:m", "questions": "q", "chunks_sha": "abc",
            "elapsed_min": 1.0}
    report = answer_eval.format_report(rows, meta)
    assert "ข้อที่ตอบผิด (1 ข้อ)" in report
    assert "ollama:m" in report and "abc" in report
