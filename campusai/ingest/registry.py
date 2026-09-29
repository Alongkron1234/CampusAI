"""ทะเบียนเอกสารที่ ingest แล้ว เก็บที่ data/processed/documents.json

1 รายการต่อ 1 ไฟล์ PDF: แฮชเนื้อไฟล์, จำนวนหน้า, หน้าที่ OCR, หน้าที่ OCR ไม่สำเร็จ, จำนวน chunk
ใช้ 2 งาน:
- กันอัปโหลดไฟล์ซ้ำ: เทียบแฮชเนื้อไฟล์ (ชื่อไฟล์ต่างแต่เนื้อเดียวกันก็จับได้)
- แสดงรายการเอกสารและสถานะบนเว็บ (Issue #10)
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime

from campusai import config

# อ้างผ่าน PROCESSED_DIR ตอนเรียกจริง (ไม่ตั้งเป็นค่าคงที่) test ที่ patch PROCESSED_DIR ไปโฟลเดอร์ชั่วคราว
# จะได้ไม่เผลอเขียนทับทะเบียนจริง
DOCUMENTS_FILENAME = "documents.json"


def _path():
    return config.PROCESSED_DIR / DOCUMENTS_FILENAME


@dataclass
class DocRecord:
    doc: str  # ชื่อไฟล์ เช่น "ruleG2568.pdf"
    sha256: str
    pages: int
    ocr_pages: list[int] = field(default_factory=list)
    failed_pages: list[int] = field(default_factory=list)  # OCR ไม่สำเร็จ เนื้อหาหน้านี้หายไป
    chunks: int = 0
    ingested_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def load_registry() -> dict[str, DocRecord]:
    if not _path().exists():
        return {}
    rows = json.loads(_path().read_text(encoding="utf-8"))
    return {row["doc"]: DocRecord(**row) for row in rows}


def _save(registry: dict[str, DocRecord]) -> None:
    _path().parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in sorted(registry.values(), key=lambda r: r.doc)]
    # เขียนไฟล์ชั่วคราวแล้วค่อยแทนที่ ถ้าโปรแกรมตายกลางทาง ไฟล์เดิมยังไม่พัง
    tmp = _path().with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_path())


def save_record(record: DocRecord) -> None:
    registry = load_registry()
    registry[record.doc] = record
    _save(registry)


def remove_record(doc: str) -> None:
    registry = load_registry()
    if registry.pop(doc, None) is not None:
        _save(registry)


def find_by_hash(sha256: str) -> str | None:
    """ชื่อเอกสารที่มีเนื้อไฟล์เดียวกันนี้อยู่แล้ว (ไม่มีคืน None)"""
    for record in load_registry().values():
        if record.sha256 == sha256:
            return record.doc
    return None
