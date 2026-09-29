"""เก็บสถานะงาน ingest ที่รันเบื้องหลัง ลง SQLite (data/jobs.db)

ใช้ SQLite แทนการเก็บในหน่วยความจำ เพื่อให้ปิด/เปิดเซิร์ฟเวอร์ใหม่แล้วยังเห็นประวัติงาน
และงานที่ยังไม่ได้ทำ (queued) ถูกหยิบมาทำต่อได้

สถานะของงาน: queued -> running -> done | failed
ระหว่าง running มี stage/done/total บอกความคืบหน้า เช่น stage="ocr", done=12, total=30
"""

import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"
_UPDATABLE = {"status", "stage", "done", "total", "message"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    doc TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT '',
    done INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


@dataclass
class Job:
    id: str
    doc: str
    status: str
    stage: str
    done: int
    total: int
    message: str  # error ถ้า failed หรือสรุปผลถ้า done (เช่น "52 chunk, OCR ไม่สำเร็จ 1 หน้า")
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class JobStore:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # เรียกจากหลาย thread (request ของ API + worker) จึงปิด check_same_thread แล้วคุมด้วย lock เอง
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def create(self, doc: str) -> Job:
        now = _now()
        job = Job(str(uuid.uuid4()), doc, QUEUED, "", 0, 0, "", now, now)
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs VALUES (:id, :doc, :status, :stage, :done, :total, :message, "
                ":created_at, :updated_at)",
                job.to_dict(),
            )
            self._conn.commit()
        return job

    def update(self, job_id: str, **fields) -> None:
        # ชื่อคอลัมน์ถูกต่อเข้า SQL ตรง ๆ จึงรับเฉพาะชื่อที่รู้จัก (ค่าส่งผ่าน parameter ปลอดภัยอยู่แล้ว)
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"แก้คอลัมน์นี้ไม่ได้: {sorted(unknown)}")
        fields["updated_at"] = _now()
        columns = ", ".join(f"{name} = :{name}" for name in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {columns} WHERE id = :id", {**fields, "id": job_id}
            )
            self._conn.commit()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job(**dict(row)) if row else None

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Job(**dict(r)) for r in rows]

    def with_status(self, status: str) -> list[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at, rowid", (status,)
            ).fetchall()
        return [Job(**dict(r)) for r in rows]

    def recover_after_restart(self) -> list[Job]:
        """เรียกตอนเปิดเซิร์ฟเวอร์: งานที่ค้าง running (เซิร์ฟเวอร์ถูกปิดกลางทาง) ให้กลับไปรอคิวใหม่

        ทำซ้ำได้ปลอดภัยเพราะ OCR มี cache หน้าที่ทำไปแล้วไม่ต้องทำใหม่
        คืนงานที่รอคิวทั้งหมดตามลำดับ ไว้ให้ worker หยิบไปทำ
        """
        for job in self.with_status(RUNNING):
            self.update(job.id, status=QUEUED, message="เซิร์ฟเวอร์ถูกปิดระหว่างทำ เริ่มใหม่")
        return self.with_status(QUEUED)
