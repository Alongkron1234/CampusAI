import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from campusai import config
from campusai.api import service as service_mod
from campusai.api.app import create_app
from campusai.api.jobs import DONE, FAILED, QUEUED, RUNNING, JobStore
from campusai.generation import answer as gen
from campusai.ingest import pipeline
from campusai.ingest.chunker import Chunk
from campusai.retrieval import hybrid, indexer
from campusai.retrieval.embedder import EMBEDDING_DIM

PDF = b"%PDF-1.4 fake"


def _vec(i: int = 0) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[i % EMBEDDING_DIM] = 1.0
    return v


def _fake_ingest(pdf_path, force_ocr=False, verbose=False, on_progress=None):
    if on_progress:
        on_progress("ocr", 1, 1)
    chunks = [
        Chunk(id=f"{pdf_path.stem}_1", doc=pdf_path.name, clause="ข้อ 1", chapter=None,
              text="ห้ามสูบบุหรี่ในมหาวิทยาลัย"),
        Chunk(id=f"{pdf_path.stem}_2", doc=pdf_path.name, clause="ข้อ 2", chapter=None,
              text="ต้องแต่งกายสุภาพ"),
    ]
    return pipeline.IngestResult(doc=pdf_path.name, chunks=chunks, pages=1, ocr_pages=[1])


@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "processed" / "chunks.jsonl")
    monkeypatch.setattr(indexer, "embed_texts", lambda texts: [_vec(i) for i in range(len(texts))])
    monkeypatch.setattr(hybrid, "embed_query", lambda q: _vec(0))
    monkeypatch.setattr(pipeline, "ingest_single_file", _fake_ingest)
    monkeypatch.setattr(gen, "call_llm", lambda prompt, system=gen.SYSTEM_PROMPT: "ห้ามสูบ [1]")

    service = service_mod.CampusService(
        client=QdrantClient(":memory:"), jobs=JobStore(":memory:"), start_worker=False
    )
    yield service
    service.stop()


@pytest.fixture
def api(svc):
    with TestClient(create_app(service_factory=lambda: svc)) as client:
        yield client


def _upload(api, name="rule.pdf", data=PDF):
    return api.post("/api/documents", files={"file": (name, data, "application/pdf")})


def _upload_and_run(api, svc, name="rule.pdf"):
    job = _upload(api, name).json()
    svc.run_job(job["id"])  # แทน worker thread: รันงานทันทีแบบ deterministic
    return job


# ----- safe_filename -----


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ระเบียบ 2568.pdf", "ระเบียบ_2568.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\x\\rule.PDF", "rule.pdf"),
    ],
)
def test_safe_filename(raw, expected):
    assert service_mod.safe_filename(raw) == expected


@pytest.mark.parametrize("raw", ["notes.txt", ".pdf", "noext"])
def test_safe_filename_rejects_non_pdf(raw):
    with pytest.raises(service_mod.UploadError):
        service_mod.safe_filename(raw)


# ----- endpoints -----


def test_health(api):
    body = api.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm"]


def test_ask_without_documents_refuses(api):
    body = api.post("/api/ask", json={"question": "สูบบุหรี่ได้ไหม"}).json()
    assert body["refused"] is True
    assert body["sources"] == []


def test_ask_validates_input(api):
    assert api.post("/api/ask", json={"question": ""}).status_code == 422
    assert api.post("/api/ask", json={"question": "q", "top_k": 99}).status_code == 422


def test_upload_queues_job_and_returns_immediately(api, svc):
    res = _upload(api)
    assert res.status_code == 202
    job = res.json()
    assert job["status"] == QUEUED and job["doc"] == "rule.pdf"
    assert (config.RAW_DIR / "rule.pdf").read_bytes() == PDF
    assert api.get("/api/documents").json()[0]["job_status"] == QUEUED


def test_job_runs_to_done_then_document_is_searchable(api, svc):
    job = _upload_and_run(api, svc)

    done = api.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == DONE
    assert "2 chunk" in done["message"]

    docs = api.get("/api/documents").json()
    assert docs[0]["doc"] == "rule.pdf" and docs[0]["chunks"] == 2
    assert docs[0]["job_status"] is None

    body = api.post("/api/ask", json={"question": "สูบบุหรี่ได้ไหม"}).json()
    assert body["refused"] is False
    assert body["sources"][0]["n"] == 1
    assert body["sources"][0]["doc"] == "rule.pdf"


def test_upload_rejects_non_pdf_content(api):
    res = _upload(api, data=b"hello")
    assert res.status_code == 400


def test_upload_rejects_same_content_under_another_name(api, svc):
    _upload_and_run(api, svc, "a.pdf")
    res = _upload(api, "b.pdf")
    assert res.status_code == 409
    assert "a.pdf" in res.json()["detail"]


def test_upload_rejects_while_same_doc_in_queue(api):
    _upload(api, "a.pdf")
    assert _upload(api, "a.pdf", data=PDF + b" v2").status_code == 409


def test_failed_job_records_reason(api, svc, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("OCR พัง")

    monkeypatch.setattr(pipeline, "ingest_single_file", _boom)
    job = _upload_and_run(api, svc)
    failed = api.get(f"/api/jobs/{job['id']}").json()
    assert failed["status"] == FAILED
    assert "OCR พัง" in failed["message"]


def test_delete_document(api, svc):
    _upload_and_run(api, svc)
    assert api.delete("/api/documents/rule.pdf").status_code == 200
    assert api.get("/api/documents").json() == []
    assert not (config.RAW_DIR / "rule.pdf").exists()
    assert api.post("/api/ask", json={"question": "สูบบุหรี่"}).json()["refused"] is True


def test_delete_unknown_or_busy_document(api):
    assert api.delete("/api/documents/nope.pdf").status_code == 404
    _upload(api, "busy.pdf")
    assert api.delete("/api/documents/busy.pdf").status_code == 409


def test_unknown_job_is_404(api):
    assert api.get("/api/jobs/does-not-exist").status_code == 404


def test_ask_llm_failure_returns_502(api, svc, monkeypatch):
    _upload_and_run(api, svc)

    def _down(prompt, system=gen.SYSTEM_PROMPT):
        raise ConnectionError("Ollama ไม่ตอบ")

    monkeypatch.setattr(gen, "call_llm", _down)
    assert api.post("/api/ask", json={"question": "สูบบุหรี่"}).status_code == 502


# ----- JobStore -----


def test_jobstore_recovers_running_jobs_after_restart(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    a = store.create("a.pdf")
    b = store.create("b.pdf")
    store.update(a.id, status=RUNNING)
    store.update(b.id, status=DONE)

    reopened = JobStore(tmp_path / "jobs.db")  # เหมือนปิดแล้วเปิดเซิร์ฟเวอร์ใหม่
    pending = reopened.recover_after_restart()
    assert [j.id for j in pending] == [a.id]
    assert reopened.get(b.id).status == DONE


def test_jobstore_rejects_unknown_columns():
    store = JobStore(":memory:")
    job = store.create("a.pdf")
    with pytest.raises(ValueError):
        store.update(job.id, doc="evil")
