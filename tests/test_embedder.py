from campusai.ingest.chunker import Chunk
from campusai.retrieval import embedder


class _FakeModel:
    """โมเดลปลอมไว้ใช้ mock แทน bge-m3 จริง (ไม่ต้องโหลดโมเดลจริงตอน test)"""

    def __init__(self):
        self.encode_calls: list[list[str]] = []

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        self.encode_calls.append(list(texts))
        # คืน vector ปลอม 3 มิติ ที่ขึ้นกับความยาวข้อความ (deterministic ไว้เช็คผลได้)
        import numpy as np

        return np.array([[float(len(t)), 0.0, 1.0] for t in texts])


def test_chunk_to_text_includes_doc_chapter_clause_and_text():
    chunk = Chunk(
        id="x",
        doc="ruleG2568.pdf",
        clause="ข้อ 15",
        chapter="หมวด 4 การรับเข้าศึกษา",
        text="เนื้อหา",
    )
    result = embedder.chunk_to_text(chunk)

    assert "ruleG2568.pdf" in result
    assert "หมวด 4 การรับเข้าศึกษา" in result
    assert "ข้อ 15" in result
    assert "เนื้อหา" in result


def test_chunk_to_text_skips_missing_chapter_and_clause():
    # chunk เกริ่นนำไม่มีทั้ง chapter และ clause (เป็น None)
    chunk = Chunk(id="x", doc="doc.pdf", clause=None, chapter=None, text="เกริ่นนำ")
    result = embedder.chunk_to_text(chunk)

    assert result == "doc.pdf | เกริ่นนำ"


def test_embed_texts_returns_empty_list_for_empty_input():
    assert embedder.embed_texts([]) == []


def test_embed_texts_calls_model_and_converts_to_list(monkeypatch):
    fake_model = _FakeModel()
    monkeypatch.setattr(embedder, "get_model", lambda: fake_model)

    texts = ["สวัสดี", "ลาก่อน"]
    result = embedder.embed_texts(texts)

    assert fake_model.encode_calls == [texts]
    assert result == [[float(len(t)), 0.0, 1.0] for t in texts]
    assert isinstance(result, list)
    assert isinstance(result[0], list)  # แปลงจาก numpy array เป็น list ธรรมดาแล้ว


def test_embed_query_returns_single_vector_not_wrapped_in_list(monkeypatch):
    fake_model = _FakeModel()
    monkeypatch.setattr(embedder, "get_model", lambda: fake_model)

    query = "คำถาม"
    result = embedder.embed_query(query)

    assert result == [float(len(query)), 0.0, 1.0]  # ไม่ใช่ [[...]] (แกะออกจาก list ชั้นนอกแล้ว)


def test_get_model_loads_only_once(monkeypatch):
    monkeypatch.setattr(embedder, "_model", None)  # เริ่มจากยังไม่โหลด
    call_count = {"n": 0}

    def _fake_constructor(model_name, device):
        call_count["n"] += 1
        return _FakeModel()

    monkeypatch.setattr(embedder, "SentenceTransformer", _fake_constructor)
    monkeypatch.setattr(embedder, "_select_device", lambda: "cpu")

    first = embedder.get_model()
    second = embedder.get_model()

    assert call_count["n"] == 1  # เรียก constructor แค่ครั้งเดียว
    assert first is second  # ได้ object เดิมกลับมา ไม่ได้สร้างใหม่


def test_select_device_prefers_mps_when_available(monkeypatch):
    monkeypatch.setattr(embedder.torch.backends.mps, "is_available", lambda: True)
    assert embedder._select_device() == "mps"


def test_select_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(embedder.torch.backends.mps, "is_available", lambda: False)
    assert embedder._select_device() == "cpu"
