import pytest

from campusai.ingest.chunker import Chunk
from campusai.retrieval import hybrid
from campusai.retrieval.vector_store import SearchResult


def _r(chunk_id: str, doc: str = "a.pdf", clause: str | None = "ข้อ 1") -> SearchResult:
    return SearchResult(chunk_id=chunk_id, doc=doc, clause=clause, chapter=None, text="", score=0.0)


def _c(chunk_id: str, doc: str = "a.pdf", clause: str | None = "ข้อ 1") -> Chunk:
    return Chunk(id=chunk_id, doc=doc, clause=clause, chapter=None, text="")


# ----- extract_clause_refs -----


def test_extract_clause_refs_arabic_and_thai_digits():
    assert hybrid.extract_clause_refs("ข้อ 15 ว่าอย่างไร") == ["ข้อ 15"]
    assert hybrid.extract_clause_refs("ข้อ ๑๕ ว่าอย่างไร") == ["ข้อ 15"]
    assert hybrid.extract_clause_refs("ข้อ15 ว่าอย่างไร") == ["ข้อ 15"]


def test_extract_clause_refs_sub_item_maps_to_parent_clause():
    assert hybrid.extract_clause_refs("ตามข้อ 17.2.5 ต้องทำอย่างไร") == ["ข้อ 17"]


def test_extract_clause_refs_multiple_unique_in_order():
    assert hybrid.extract_clause_refs("ข้อ 8 กับข้อ 15 และข้อ 8 อีกครั้ง") == ["ข้อ 8", "ข้อ 15"]


def test_extract_clause_refs_supports_matra_and_none():
    assert hybrid.extract_clause_refs("มาตรา 18 ว่าอย่างไร") == ["มาตรา 18"]
    assert hybrid.extract_clause_refs("แต่งกายผิดระเบียบไหม") == []


# ----- rrf_fuse -----


def test_rrf_fuse_rewards_agreement_between_rankings():
    a = [_r("x"), _r("y")]
    b = [_r("y"), _r("z")]
    fused = hybrid.rrf_fuse([a, b])
    assert fused[0].chunk_id == "y"  # อยู่ในทั้งสองรายการ ชนะอันดับ 1 ของรายการเดียว
    assert {r.chunk_id for r in fused} == {"x", "y", "z"}  # ที่เจอแค่รายการเดียวยังอยู่


def test_rrf_fuse_score_formula():
    fused = hybrid.rrf_fuse([[_r("x")], [_r("y"), _r("x")]], k=60)
    scores = {r.chunk_id: r.score for r in fused}
    assert scores["x"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["y"] == pytest.approx(1 / 61)


def test_rrf_fuse_respects_weights():
    a = [_r("x")]
    b = [_r("y")]
    fused = hybrid.rrf_fuse([a, b], weights=[1.0, 0.5])
    assert [r.chunk_id for r in fused] == ["x", "y"]


def test_rrf_fuse_zero_weight_ranking_has_no_influence():
    fused = hybrid.rrf_fuse([[_r("x"), _r("y")], [_r("y")]], weights=[1.0, 0.0])
    assert [r.chunk_id for r in fused] == ["x", "y"]


# ----- clause_match_ranking -----


def test_clause_match_ranking_empty_without_refs():
    assert hybrid.clause_match_ranking([], [_r("x")], [_c("x")]) == []


def test_clause_match_ranking_adds_chunk_no_retriever_found():
    chunks = [_c("c6", clause="ข้อ 6"), _c("c7", clause="ข้อ 7")]
    ranking = hybrid.clause_match_ranking(["ข้อ 6"], candidates=[], all_chunks=chunks)
    assert [r.chunk_id for r in ranking] == ["c6"]


def test_clause_match_ranking_matches_inserted_clause_variant():
    chunks = [_c("c5", clause="ข้อ 5"), _c("c51", clause="ข้อ 5/1")]
    ranking = hybrid.clause_match_ranking(["ข้อ 5"], candidates=[], all_chunks=chunks)
    assert {r.chunk_id for r in ranking} == {"c5", "c51"}


def test_clause_match_ranking_prefers_doc_most_candidates_point_to():
    # ทั้งสองไฟล์มีข้อ 6 แต่ผลค้นหาส่วนใหญ่ชี้ไปที่ grad.pdf จึงต้องได้ข้อ 6 ของ grad.pdf ก่อน
    candidates = [
        _r("d6", doc="discipline.pdf", clause="ข้อ 6"),
        _r("g1", doc="grad.pdf", clause="ข้อ 1"),
        _r("g2", doc="grad.pdf", clause="ข้อ 2"),
        _r("g3", doc="grad.pdf", clause="ข้อ 3"),
    ]
    chunks = [
        _c("d6", doc="discipline.pdf", clause="ข้อ 6"),
        _c("g6", doc="grad.pdf", clause="ข้อ 6"),
    ]
    ranking = hybrid.clause_match_ranking(["ข้อ 6"], candidates, chunks)
    assert [r.chunk_id for r in ranking] == ["g6", "d6"]


# ----- Retriever.hybrid (mock ระบบค้นหาทั้งสอง ไม่ต้องใช้โมเดล/Qdrant จริง) -----


class _StubRetriever(hybrid.Retriever):
    def __init__(self, vector_hits, keyword_hits, chunks):
        self._vector_hits = vector_hits
        self._keyword_hits = keyword_hits
        self._chunks = chunks

    @property
    def chunks(self):
        return self._chunks

    def vector(self, query, top_k=5):
        return self._vector_hits[:top_k]

    def keyword(self, query, top_k=5):
        return self._keyword_hits[:top_k]


def test_hybrid_boosts_clause_mentioned_in_query():
    chunks = [_c("c1", clause="ข้อ 1"), _c("c9", clause="ข้อ 9")]
    retriever = _StubRetriever([_r("c1", clause="ข้อ 1"), _r("c9", clause="ข้อ 9")], [], chunks)
    assert retriever.hybrid("ข้อ 9 ว่าด้วยอะไร", top_k=2)[0].chunk_id == "c9"


def test_hybrid_without_clause_ref_follows_vector_order():
    chunks = [_c("c1"), _c("c2", clause="ข้อ 2")]
    retriever = _StubRetriever(
        [_r("c1"), _r("c2", clause="ข้อ 2")], [_r("c2", clause="ข้อ 2")], chunks
    )
    assert [r.chunk_id for r in retriever.hybrid("แต่งกาย", top_k=2)] == ["c1", "c2"]


def test_hybrid_respects_top_k():
    chunks = [_c(f"c{i}", clause=f"ข้อ {i}") for i in range(5)]
    hits = [_r(f"c{i}", clause=f"ข้อ {i}") for i in range(5)]
    assert len(_StubRetriever(hits, [], chunks).hybrid("คำถาม", top_k=3)) == 3
