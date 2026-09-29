import pytest

from campusai.ingest.chunker import Chunk
from campusai.retrieval import bm25


def _chunk(chunk_id: str, clause: str | None, text: str) -> Chunk:
    return Chunk(id=chunk_id, doc="doc.pdf", clause=clause, chapter="หมวด 1", text=text)


@pytest.fixture
def index():
    chunks = [
        _chunk("c13", "ข้อ 13", "ข้อ 13 นักศึกษาต้องแต่งกายสุภาพ เรียบร้อย ตามระเบียบของมหาวิทยาลัย"),
        _chunk("c20", "ข้อ 20", "ข้อ 20 นักศึกษาต้องไม่สูบบุหรี่ หรือบุหรี่ไฟฟ้า ภายในมหาวิทยาลัย"),
        _chunk("c15", "ข้อ 15", "ข้อ 15 ปริญญาโท ให้ใช้เวลาศึกษาไม่เกิน 5 ปีการศึกษา"),
    ]
    return bm25.build_index(chunks)


def test_tokenize_splits_thai_words():
    tokens = bm25.tokenize("นักศึกษาต้องแต่งกายสุภาพ")
    assert "นักศึกษา" in tokens
    assert "แต่งกาย" in tokens


def test_tokenize_drops_whitespace_and_punctuation():
    tokens = bm25.tokenize("ข้อ 15 | ปริญญาโท  ,  ")
    assert "|" not in tokens
    assert "," not in tokens
    assert all(t.strip() for t in tokens)
    assert "15" in tokens


def test_tokenize_lowercases_english():
    assert "code" in bm25.tokenize("Code of Honor")


def test_tokenize_keeps_thai_words_with_vowel_marks():
    # สระ/วรรณยุกต์ไทยไม่นับเป็น alnum ต้องไม่ทำให้คำไทยถูกทิ้ง
    assert bm25.tokenize("แต่งกาย") != []


def test_build_index_rejects_empty_chunks():
    with pytest.raises(ValueError):
        bm25.build_index([])


def test_search_finds_chunk_by_specific_word(index):
    results = bm25.search(index, "บุหรี่")
    assert results[0].chunk_id == "c20"


def test_search_finds_chunk_by_clause_number(index):
    results = bm25.search(index, "ข้อ 15")
    assert results[0].chunk_id == "c15"


def test_search_returns_full_metadata(index):
    result = bm25.search(index, "แต่งกาย")[0]
    assert result.chunk_id == "c13"
    assert result.doc == "doc.pdf"
    assert result.clause == "ข้อ 13"
    assert result.chapter == "หมวด 1"
    assert "แต่งกาย" in result.text
    assert result.score > 0


def test_search_excludes_chunks_with_zero_score(index):
    results = bm25.search(index, "บุหรี่")
    assert [r.chunk_id for r in results] == ["c20"]  # อีก 2 chunk ไม่มีคำนี้เลย ไม่ควรโผล่


@pytest.fixture
def exam_index():
    # BM25 ให้น้ำหนักติดลบกับคำที่อยู่เกินครึ่งของเอกสาร (ทำให้คะแนน ≤ 0 แล้วถูกตัดทิ้ง)
    # ชุดนี้จึงมี 6 chunk และคำว่า "สอบ" อยู่แค่ 3 chunk (ไม่เกินครึ่ง) เพื่อให้เทสอันดับมีความหมาย
    chunks = [
        _chunk("e1", "ข้อ 1", "ทุจริตในการสอบ สอบ สอบ ถือเป็นความผิดร้ายแรง"),
        _chunk("e2", "ข้อ 2", "ผู้ใดทุจริตในการสอบ ให้พักการศึกษา"),
        _chunk("e3", "ข้อ 3", "ตักเตือนด้วยวาจาหรือเป็นลายลักษณ์อักษร"),
        _chunk("e4", "ข้อ 4", "บำเพ็ญประโยชน์ต่อมหาวิทยาลัย"),
        _chunk("e5", "ข้อ 5", "ให้พ้นสภาพการเป็นนักศึกษา"),
        _chunk("e6", "ข้อ 6", "ภาษาอังกฤษตามที่หลักสูตรกำหนด"),
    ]
    return bm25.build_index(chunks)


def test_search_misses_word_hidden_inside_compound_word():
    # ข้อจำกัดจริงของ BM25 กับภาษาไทย: "สอบผ่าน" ถูกตัดเป็นคำเดียว จึงไม่ตรงกับคำถาม "สอบ"
    # (เป็นเหตุผลที่ต้องใช้คู่กับ vector search ซึ่งเข้าใจความหมาย ไม่ผูกกับการตัดคำ)
    chunks = [_chunk(f"x{i}", None, f"หมวดที่ {i} เรื่องอื่น") for i in range(4)]
    chunks.append(_chunk("hit", "ข้อ 9", "สอบผ่านภาษาอังกฤษตามที่หลักสูตรกำหนด"))
    index = bm25.build_index(chunks)

    assert bm25.search(index, "สอบ") == []


def test_search_results_sorted_by_score_descending(exam_index):
    results = bm25.search(exam_index, "สอบ")
    assert len(results) >= 2  # ต้องมีผลจริง ไม่งั้นเทสนี้ผ่านแบบไม่มีความหมาย
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk_id == "e1"  # เจอคำว่า "สอบ" บ่อยที่สุด


def test_search_respects_top_k(exam_index):
    assert len(bm25.search(exam_index, "สอบ", top_k=10)) >= 2
    assert len(bm25.search(exam_index, "สอบ", top_k=1)) == 1


def test_search_drops_words_that_appear_in_most_chunks(index):
    # "นักศึกษา" อยู่ใน 2 จาก 3 chunk: IDF ติดลบ คะแนนไม่เป็นบวก จึงไม่ถูกนับเป็นผลค้นหา
    assert bm25.search(index, "นักศึกษา") == []


def test_search_with_no_usable_query_tokens_returns_empty(index):
    assert bm25.search(index, "   ...  ") == []


def test_search_with_no_matching_word_returns_empty(index):
    assert bm25.search(index, "ซอฟต์แวร์") == []
