from campusai import config
from campusai.ingest import registry


def test_save_load_find_and_remove(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    assert registry.load_registry() == {}

    registry.save_record(registry.DocRecord(doc="a.pdf", sha256="aaa", pages=3, failed_pages=[2]))
    registry.save_record(registry.DocRecord(doc="b.pdf", sha256="bbb", pages=1))

    loaded = registry.load_registry()
    assert loaded["a.pdf"].failed_pages == [2]
    assert registry.find_by_hash("bbb") == "b.pdf"
    assert registry.find_by_hash("zzz") is None

    registry.remove_record("a.pdf")
    registry.remove_record("not-there.pdf")  # ลบของที่ไม่มีต้องไม่ error
    assert set(registry.load_registry()) == {"b.pdf"}


def test_save_replaces_existing_record(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    registry.save_record(registry.DocRecord(doc="a.pdf", sha256="old", pages=1))
    registry.save_record(registry.DocRecord(doc="a.pdf", sha256="new", pages=2))
    assert registry.load_registry()["a.pdf"].sha256 == "new"
