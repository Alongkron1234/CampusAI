import pytest

from campusai.evaluation import retrieval_eval
from campusai.retrieval.vector_store import SearchResult


def _r(chunk_id: str, doc: str, clause: str) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, doc=doc, clause=clause, chapter=None, text="", score=0.0)


def test_first_hit_rank_found_and_missing():
    results = [_r("1", "a", "ข้อ 1"), _r("2", "a", "ข้อ 2")]
    assert retrieval_eval.first_hit_rank(results, {("a", "ข้อ 2")}) == 2
    assert retrieval_eval.first_hit_rank(results, {("a", "ข้อ 9")}) is None


def test_first_hit_rank_counts_split_clause_once():
    # ข้อ 10 ถูกซอยเป็น 3 chunk ต้องนับเป็นอันดับเดียว เฉลย ข้อ 3 จึงอยู่อันดับ 2 ไม่ใช่ 4
    results = [
        _r("10a", "a", "ข้อ 10"),
        _r("10b", "a", "ข้อ 10"),
        _r("10c", "a", "ข้อ 10"),
        _r("3", "a", "ข้อ 3"),
    ]
    assert retrieval_eval.first_hit_rank(results, {("a", "ข้อ 3")}) == 2


def test_first_hit_rank_distinguishes_documents():
    results = [_r("x", "discipline.pdf", "ข้อ 6"), _r("y", "grad.pdf", "ข้อ 6")]
    assert retrieval_eval.first_hit_rank(results, {("grad.pdf", "ข้อ 6")}) == 2


def test_summarize_hits_and_mrr():
    m = retrieval_eval.summarize([1, 2, None, 4])
    assert m["hit@1"] == 1
    assert m["hit@3"] == 2
    assert m["hit@5"] == 3
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert m["n"] == 4


def test_evaluate_skips_unanswerable_and_groups_by_type():
    questions = [
        {"question": "q1", "type": "direct", "expected": [{"doc": "a", "clause": "ข้อ 1"}]},
        {"question": "q2", "type": "semantic", "expected": [{"doc": "a", "clause": "ข้อ 2"}]},
        {"question": "q3", "type": "unanswerable", "expected": []},
    ]
    fake_results = {
        "q1": [_r("1", "a", "ข้อ 1")],
        "q2": [_r("9", "a", "ข้อ 9"), _r("2", "a", "ข้อ 2")],
    }

    def _search(query, top_k):
        assert query != "q3", "ไม่ควรรันคำถาม unanswerable"
        return fake_results[query]

    report = retrieval_eval.evaluate(questions, _search)

    assert set(report) == {"direct", "semantic", "all"}
    assert report["direct"]["hit@1"] == 1
    assert report["semantic"]["hit@1"] == 0
    assert report["semantic"]["mrr"] == pytest.approx(0.5)
    assert report["all"]["n"] == 2


def test_evaluate_multi_type_counts_hit_if_any_expected_clause_found():
    questions = [
        {
            "question": "q",
            "type": "multi",
            "expected": [{"doc": "a", "clause": "ข้อ 8"}, {"doc": "a", "clause": "ข้อ 15"}],
        }
    ]
    report = retrieval_eval.evaluate(questions, lambda q, k: [_r("15", "a", "ข้อ 15")])
    assert report["multi"]["hit@1"] == 1


def test_format_report_shows_raw_counts():
    reports = {"vector": {"all": retrieval_eval.summarize([1, None])}}
    table = retrieval_eval.format_report(reports)
    assert "1/2 (50%)" in table
