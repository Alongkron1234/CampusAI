import json

import pytest
from qdrant_client import QdrantClient

from campusai import config
from campusai.cli import main
from campusai.retrieval import indexer
from campusai.retrieval.embedder import EMBEDDING_DIM

ROWS = [
    {"id": "a", "doc": "d.pdf", "clause": "ข้อ 1", "chapter": "หมวด 1", "text": "แต่งกายสุภาพ"},
    {"id": "b", "doc": "d.pdf", "clause": "ข้อ 2", "chapter": "หมวด 1", "text": "ห้ามสูบบุหรี่"},
    {"id": "c", "doc": "d.pdf", "clause": None, "chapter": None, "text": "เกริ่นนำ"},
]


def _write_chunks(path, rows=ROWS):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")


def _fake_embed(texts):
    # vector ปลอมขนาดจริง (1024 มิติ) คนละตำแหน่งต่อ chunk ไม่ต้องโหลดโมเดลจริง
    vectors = []
    for i, _ in enumerate(texts):
        vector = [0.0] * EMBEDDING_DIM
        vector[i] = 1.0
        vectors.append(vector)
    return vectors


@pytest.fixture(autouse=True)
def _never_touch_real_model_or_store(monkeypatch):
    # กันเทสในไฟล์นี้หลุดไปโหลด bge-m3 จริงหรือเขียน data/qdrant จริงโดยไม่ตั้งใจ
    # (เคยหลุดมาแล้วเพราะ config ถูก patch แต่ค่า default ของ load_chunks ยังชี้ไฟล์จริง)
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("เทสนี้ไม่ควรเรียกโมเดล/Qdrant จริง")

    monkeypatch.setattr(indexer, "embed_texts", _forbidden)
    monkeypatch.setattr(indexer.vector_store, "get_client", _forbidden)


@pytest.fixture
def memory_client(monkeypatch):
    client = QdrantClient(":memory:")
    monkeypatch.setattr(indexer.vector_store, "get_client", lambda: client)
    monkeypatch.setattr(indexer, "embed_texts", _fake_embed)
    return client


def test_load_chunks_parses_jsonl_into_chunk_objects(tmp_path):
    path = tmp_path / "chunks.jsonl"
    _write_chunks(path)

    chunks = indexer.load_chunks(path)

    assert [c.id for c in chunks] == ["a", "b", "c"]
    assert chunks[0].clause == "ข้อ 1"
    assert chunks[2].clause is None  # ค่า null ใน JSON กลายเป็น None


def test_run_index_fails_clearly_when_chunks_file_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "missing.jsonl")

    assert main(["index"]) == 1
    assert "campusai ingest" in capsys.readouterr().out


def test_run_index_fails_when_chunks_file_is_empty(tmp_path, monkeypatch, capsys):
    path = tmp_path / "chunks.jsonl"
    path.write_text("", "utf-8")
    monkeypatch.setattr(config, "CHUNKS_PATH", path)

    assert main(["index"]) == 1
    assert "ไม่มี chunk" in capsys.readouterr().out


def test_run_index_stores_every_chunk_in_qdrant(tmp_path, monkeypatch, memory_client, capsys):
    path = tmp_path / "chunks.jsonl"
    _write_chunks(path)
    monkeypatch.setattr(config, "CHUNKS_PATH", path)

    assert main(["index"]) == 0

    assert memory_client.count(config.QDRANT_COLLECTION).count == 3
    output = capsys.readouterr().out
    assert "3 chunk" in output


def test_run_index_twice_does_not_duplicate(tmp_path, monkeypatch, memory_client):
    path = tmp_path / "chunks.jsonl"
    _write_chunks(path)
    monkeypatch.setattr(config, "CHUNKS_PATH", path)

    main(["index"])
    main(["index"])

    assert memory_client.count(config.QDRANT_COLLECTION).count == 3


def test_run_index_drops_chunks_removed_from_source(tmp_path, monkeypatch, memory_client):
    path = tmp_path / "chunks.jsonl"
    monkeypatch.setattr(config, "CHUNKS_PATH", path)

    _write_chunks(path, ROWS)
    main(["index"])
    _write_chunks(path, ROWS[:1])  # ingest ใหม่แล้วเหลือ chunk เดียว
    main(["index"])

    # recreate=True ต้องล้างของเก่า ไม่ให้ chunk ที่หายไปแล้วยังค้างใน index
    assert memory_client.count(config.QDRANT_COLLECTION).count == 1
