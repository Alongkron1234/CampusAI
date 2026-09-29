import pytest
from qdrant_client import QdrantClient

from campusai import config
from campusai.ingest.chunker import Chunk
from campusai.retrieval import vector_store
from campusai.retrieval.embedder import EMBEDDING_DIM


def _unit_vector(index: int) -> list[float]:
    """vector 1024 มิติที่มีค่า 1.0 แค่ตำแหน่งเดียว (ตั้งฉากกัน: คนละตำแหน่ง = ไม่เกี่ยวกันเลย)"""
    vector = [0.0] * EMBEDDING_DIM
    vector[index] = 1.0
    return vector


def _chunk(chunk_id: str, clause: str | None = "ข้อ 1", text: str = "เนื้อหา") -> Chunk:
    return Chunk(id=chunk_id, doc="doc.pdf", clause=clause, chapter="หมวด 1", text=text)


@pytest.fixture
def client():
    # Qdrant แบบ in-memory: ไม่แตะดิสก์ ไม่ต่อเน็ต หายไปเองเมื่อ test จบ
    return QdrantClient(":memory:")


def test_ensure_collection_creates_when_missing(client):
    assert not client.collection_exists(config.QDRANT_COLLECTION)
    vector_store.ensure_collection(client)
    assert client.collection_exists(config.QDRANT_COLLECTION)


def test_ensure_collection_keeps_existing_data_by_default(client):
    vector_store.ensure_collection(client)
    vector_store.index_chunks(client, [_chunk("a")], [_unit_vector(0)])

    vector_store.ensure_collection(client)  # เรียกซ้ำ ไม่ได้สั่ง recreate

    assert client.count(config.QDRANT_COLLECTION).count == 1


def test_ensure_collection_recreate_wipes_old_data(client):
    vector_store.ensure_collection(client)
    vector_store.index_chunks(client, [_chunk("a")], [_unit_vector(0)])

    vector_store.ensure_collection(client, recreate=True)

    assert client.count(config.QDRANT_COLLECTION).count == 0


def test_chunk_point_id_is_deterministic_and_distinct():
    assert vector_store.chunk_point_id("x") == vector_store.chunk_point_id("x")
    assert vector_store.chunk_point_id("x") != vector_store.chunk_point_id("y")


def test_index_chunks_rejects_mismatched_lengths(client):
    vector_store.ensure_collection(client)
    with pytest.raises(ValueError):
        vector_store.index_chunks(client, [_chunk("a"), _chunk("b")], [_unit_vector(0)])


def test_index_twice_does_not_duplicate(client):
    vector_store.ensure_collection(client)
    chunks = [_chunk("a"), _chunk("b")]
    vectors = [_unit_vector(0), _unit_vector(1)]

    vector_store.index_chunks(client, chunks, vectors)
    vector_store.index_chunks(client, chunks, vectors)

    assert client.count(config.QDRANT_COLLECTION).count == 2


def test_search_returns_closest_chunk_first_with_metadata(client):
    vector_store.ensure_collection(client)
    chunks = [
        _chunk("near", clause="ข้อ 13", text="แต่งกายสุภาพ"),
        _chunk("far", clause="ข้อ 20", text="บุหรี่"),
    ]
    vector_store.index_chunks(client, chunks, [_unit_vector(0), _unit_vector(1)])

    results = vector_store.search(client, _unit_vector(0), top_k=2)

    assert [r.chunk_id for r in results] == ["near", "far"]
    assert results[0].clause == "ข้อ 13"
    assert results[0].text == "แต่งกายสุภาพ"
    assert results[0].doc == "doc.pdf"
    assert results[0].chapter == "หมวด 1"
    assert results[0].score == pytest.approx(1.0)  # vector เหมือนกันเป๊ะ
    assert results[1].score == pytest.approx(0.0)  # vector ตั้งฉากกัน


def test_search_respects_top_k(client):
    vector_store.ensure_collection(client)
    chunks = [_chunk(f"c{i}") for i in range(4)]
    vector_store.index_chunks(client, chunks, [_unit_vector(i) for i in range(4)])

    assert len(vector_store.search(client, _unit_vector(0), top_k=2)) == 2


def test_search_keeps_none_clause_for_preamble_chunk(client):
    vector_store.ensure_collection(client)
    vector_store.index_chunks(client, [_chunk("pre", clause=None)], [_unit_vector(0)])

    results = vector_store.search(client, _unit_vector(0), top_k=1)

    assert results[0].clause is None


def test_get_client_local_mode_uses_configured_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "QDRANT_MODE", "local")
    monkeypatch.setattr(config, "QDRANT_PATH", tmp_path / "qdrant")

    local_client = vector_store.get_client()

    assert (tmp_path / "qdrant").exists()
    local_client.close()


def test_get_client_cloud_mode_requires_url_and_key(monkeypatch):
    monkeypatch.setattr(config, "QDRANT_MODE", "cloud")
    monkeypatch.setattr(config, "QDRANT_URL", "")
    monkeypatch.setattr(config, "QDRANT_API_KEY", "")

    with pytest.raises(RuntimeError, match="QDRANT_URL"):
        vector_store.get_client()
