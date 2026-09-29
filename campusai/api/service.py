"""ตรรกะของเว็บ API แยกจากตัว HTTP (app.py) ให้ test ได้โดยไม่ต้องยิง request

รวม 3 งาน:
- ถาม-ตอบ: ค้นหา (hybrid) + ให้ LLM ตอบ
- อัปโหลดเอกสาร: บันทึก PDF แล้วเข้าคิว ให้ worker thread ทำ ingest + index ทีละงานเบื้องหลัง
- จัดการเอกสาร: ดูรายการ, ลบ

ทุกอย่างอยู่ใน process เดียว เพราะ Qdrant แบบ local เปิดได้ทีละ process (เคยเจอตอน chat ค้างไว้)
ข้อมูลที่ใช้ร่วมกัน (Qdrant, chunks.jsonl, BM25) คุมด้วย lock ตัวเดียว ช่วง OCR ที่นานไม่ถือ lock
จึงถามคำถามได้ระหว่างที่งาน OCR กำลังทำ
"""

import hashlib
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from campusai import config
from campusai.api.jobs import DONE, FAILED, QUEUED, RUNNING, Job, JobStore
from campusai.generation import answer as gen
from campusai.ingest import pipeline, registry
from campusai.retrieval import bm25, indexer
from campusai.retrieval.hybrid import Retriever

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


class UploadError(ValueError):
    """ไฟล์ที่อัปโหลดใช้ไม่ได้ (ไม่ใช่ PDF, ใหญ่เกิน, ชื่อไฟล์ผิด)"""


class ConflictError(ValueError):
    """ขัดกับสถานะปัจจุบัน (ไฟล์ซ้ำ, เอกสารกำลังถูก ingest อยู่)"""


@dataclass
class DocumentInfo:
    doc: str
    chunks: int
    pages: int | None = None
    failed_pages: list[int] | None = None
    ingested_at: str | None = None
    job_status: str | None = None  # งานล่าสุดของเอกสารนี้ที่ยังไม่เสร็จ (queued/running)


def safe_filename(name: str) -> str:
    """เอาเฉพาะชื่อไฟล์ (กัน path เช่น ../../x.pdf) เปลี่ยนอักขระแปลกเป็น _ และบังคับนามสกุล .pdf

    เก็บตัวอักษรไทยไว้ เพราะชื่อไฟล์ระเบียบมักเป็นภาษาไทย
    """
    base = Path(name.replace("\\", "/")).name
    stem, _, ext = base.rpartition(".")
    if ext.lower() != "pdf" or not stem:
        raise UploadError("รองรับเฉพาะไฟล์ .pdf")
    # \w ของ Python ไม่นับสระบน/ล่างและวรรณยุกต์ไทย (เป็น combining mark) จึงต้องใส่ช่วงอักษรไทยเพิ่ม
    stem = re.sub(r"[^\w\-฀-๿]+", "_", stem).strip("_")
    if not stem:
        raise UploadError("ชื่อไฟล์ใช้ไม่ได้")
    return f"{stem}.pdf"


class CampusService:
    def __init__(self, client: QdrantClient, jobs: JobStore, start_worker: bool = True):
        self.client = client
        self.jobs = jobs
        self._lock = threading.RLock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self.retriever: Retriever | None = None
        self.reload_index()

        for job in self.jobs.recover_after_restart():
            self._queue.put(job.id)
        self._worker: threading.Thread | None = None
        if start_worker:
            self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="ingest")
            self._worker.start()

    # ----- ถาม-ตอบ -----

    def reload_index(self) -> None:
        """สร้าง BM25 ใหม่จาก chunks.jsonl (เรียกหลังเพิ่ม/ลบเอกสาร) ไม่มีเอกสารเลย = retriever None"""
        with self._lock:
            chunks = indexer.load_chunks() if config.CHUNKS_PATH.exists() else []
            self.retriever = (
                Retriever(client=self.client, bm25_index=bm25.build_index(chunks))
                if chunks else None
            )

    def ask(self, question: str, top_k: int = gen.TOP_K_CONTEXT) -> gen.Answer:
        with self._lock:  # ถือ lock แค่ช่วงค้นหา ช่วง LLM ตอบ (นาน) ไม่ถือ
            results = self.retriever.hybrid(question, top_k=top_k) if self.retriever else []
        return gen.generate_answer(question, results)

    # ----- เอกสาร -----

    def list_documents(self) -> list[DocumentInfo]:
        """เอกสารทั้งหมด: จากทะเบียน + เอกสารที่อยู่ใน chunks.jsonl แต่ยังไม่มีในทะเบียน
        (ingest ก่อนมีทะเบียน) + ไฟล์ที่อัปโหลดแล้วแต่ยังรอคิว/กำลังทำ"""
        records = registry.load_registry()
        counts: dict[str, int] = {}
        if self.retriever:
            for chunk in self.retriever.chunks:
                counts[chunk.doc] = counts.get(chunk.doc, 0) + 1
        active = {j.doc: j.status for j in self.jobs.list(200) if j.status in (QUEUED, RUNNING)}

        docs: dict[str, DocumentInfo] = {}
        for name in set(records) | set(counts) | set(active):
            r = records.get(name)
            docs[name] = DocumentInfo(
                doc=name,
                chunks=counts.get(name, 0),
                pages=r.pages if r else None,
                failed_pages=r.failed_pages if r else None,
                ingested_at=r.ingested_at if r else None,
                job_status=active.get(name),
            )
        return sorted(docs.values(), key=lambda d: d.doc)

    def _active_job_for(self, doc: str) -> Job | None:
        for status in (QUEUED, RUNNING):
            for job in self.jobs.with_status(status):
                if job.doc == doc:
                    return job
        return None

    def submit_upload(self, filename: str, data: bytes) -> Job:
        """ตรวจไฟล์ บันทึกลง data/raw แล้วเข้าคิว ingest (คืนทันที ไม่รอ OCR)

        ชื่อซ้ำแต่เนื้อไฟล์ต่าง = อัปเดตเอกสารเดิม (ingest ใหม่ทับ)
        เนื้อไฟล์ซ้ำกับเอกสารที่มีอยู่ (ไม่ว่าชื่ออะไร) = ปฏิเสธ
        """
        name = safe_filename(filename)
        if len(data) > MAX_UPLOAD_BYTES:
            raise UploadError(f"ไฟล์ใหญ่เกิน {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        if not data.startswith(b"%PDF"):
            raise UploadError("ไฟล์นี้ไม่ใช่ PDF")

        duplicate = registry.find_by_hash(hashlib.sha256(data).hexdigest())
        if duplicate:
            raise ConflictError(f"มีเอกสารเนื้อหาเดียวกันนี้อยู่แล้ว: {duplicate}")
        if self._active_job_for(name):
            raise ConflictError(f"{name} กำลังอยู่ในคิวหรือกำลัง ingest อยู่")

        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        (config.RAW_DIR / name).write_bytes(data)
        job = self.jobs.create(name)
        self._queue.put(job.id)
        return job

    def delete_document(self, doc: str) -> None:
        """ลบเอกสารออกทุกที่ รวมไฟล์ PDF (ไม่งั้นรอบหน้าที่ ingest ทั้งหมดจะกลับมาอีก)"""
        if self._active_job_for(doc):
            raise ConflictError(f"{doc} กำลังอยู่ในคิวหรือกำลัง ingest อยู่ ลบไม่ได้")
        known = {d.doc for d in self.list_documents()}
        if doc not in known:
            raise KeyError(doc)
        with self._lock:
            indexer.remove_documents(self.client, [doc], delete_pdf=True)
        self.reload_index()

    # ----- งานเบื้องหลัง -----

    def run_job(self, job_id: str) -> None:
        """ingest + index เอกสาร 1 ไฟล์ พร้อมอัปเดตความคืบหน้าลง JobStore"""
        job = self.jobs.get(job_id)
        if job is None or job.status != QUEUED:
            return
        self.jobs.update(job_id, status=RUNNING, stage="extract", done=0, total=0, message="")

        def _progress(stage: str, done: int, total: int) -> None:
            self.jobs.update(job_id, stage=stage, done=done, total=total)

        try:
            pdf_path = config.RAW_DIR / job.doc
            result = pipeline.ingest_single_file(pdf_path, on_progress=_progress)  # นาน ไม่ถือ lock
            _progress("index", 0, len(result.chunks))
            with self._lock:
                pipeline.save_result(pdf_path, result)
                indexer.index_documents(self.client, indexer.load_chunks(), {job.doc})
            self.reload_index()
        except Exception as exc:  # noqa: BLE001 - งานพังต้องบันทึกสาเหตุไว้ให้ผู้ใช้เห็น ไม่ให้ worker ตาย
            self.jobs.update(job_id, status=FAILED, message=str(exc) or type(exc).__name__)
            return

        message = f"{len(result.chunks)} chunk"
        if result.failed_pages:
            message += f", OCR ไม่สำเร็จ {len(result.failed_pages)} หน้า: {result.failed_pages}"
        total = len(result.chunks)
        self.jobs.update(job_id, status=DONE, stage="index", done=total, total=total,
                         message=message)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.run_job(job_id)

    def stop(self) -> None:
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=5)
