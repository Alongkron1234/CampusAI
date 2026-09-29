"""ร้อยทุกขั้นตอนของ ingestion เข้าด้วยกัน: PDF -> ตรวจ+OCR -> clean -> chunk -> เขียนไฟล์

นี่คือคำสั่ง `campusai ingest` ที่ผู้ใช้จริงจะเรียก ส่วนโมดูลอื่น (pdf_text, ocr, clean,
chunker) เป็นแค่ "ชิ้นส่วน" ที่ไฟล์นี้เอามาประกอบร่างเป็นคำสั่งเดียว
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from campusai import config
from campusai.ingest import registry
from campusai.ingest.chunker import Chunk, chunk_document
from campusai.ingest.clean import clean_text
from campusai.ingest.ocr import file_hash, ocr_pages
from campusai.ingest.pdf_text import extract_pages

# chunk ที่สั้นกว่านี้น่าสงสัยว่า chunker อาจตัดผิด (เช่นตัดจากบรรทัดว่าง ๆ)
SHORT_CHUNK_WARNING_CHARS = 15

# callback รายงานความคืบหน้า (ขั้นตอน, ทำไปแล้ว, ทั้งหมด) เช่น ("ocr", 12, 30)
# ขั้นตอน: "extract" ดึงข้อความ, "ocr" OCR ทีละหน้า, "chunk" clean + ตัด chunk
# ให้ API (Issue #10b) เอาไปแสดงบนเว็บได้ว่างานไปถึงไหนแล้ว
ProgressFn = Callable[[str, int, int], None]


@dataclass
class IngestResult:
    doc: str
    chunks: list[Chunk]
    pages: int = 0
    ocr_pages: list[int] = field(default_factory=list)
    failed_pages: list[int] = field(default_factory=list)  # OCR ไม่สำเร็จ ถูกข้ามไป


def _join_pages_in_order(page_texts: dict[int, str]) -> str:
    """รวมข้อความแต่ละหน้าเป็นเอกสารเดียว เรียงตามเลขหน้าจากน้อยไปมาก"""
    ordered_pages = sorted(page_texts.items())
    return "\n\n".join(text for _page_number, text in ordered_pages)


def ingest_single_file(
    pdf_path: Path,
    force_ocr: bool = False,
    verbose: bool = False,
    on_progress: ProgressFn | None = None,
) -> IngestResult:
    """แปลง PDF 1 ไฟล์ให้กลายเป็น list ของ Chunk ที่พร้อมใช้งาน

    ขั้นตอน: ตรวจรายหน้า -> OCR เฉพาะหน้าที่จำเป็น -> รวมข้อความ -> ทำความสะอาด -> ตัด chunk

    หน้าที่ OCR ไม่สำเร็จ (retry ครบแล้ว) จะถูกข้ามและบันทึกไว้ใน failed_pages แทนที่จะหยุดทั้งไฟล์
    ไม่เอาข้อความจาก pdfplumber มาแทน เพราะหน้าที่ต้อง OCR คือหน้าที่ข้อความจาก pdfplumber ใช้ไม่ได้
    """
    report = on_progress or (lambda stage, done, total: None)

    report("extract", 0, 1)
    pages = extract_pages(pdf_path)
    ocr_page_numbers = [p.page_number for p in pages if p.needs_ocr]
    report("extract", 1, 1)

    def _on_ocr_progress(done: int, total: int, was_cached: bool) -> None:
        report("ocr", done, total)
        if verbose:
            source = "cache" if was_cached else "OCR"
            print(f"    [{done}/{total}] หน้า OCR เสร็จ ({source})")

    ocr_results = ocr_pages(
        pdf_path, ocr_page_numbers, force=force_ocr, on_progress=_on_ocr_progress,
        skip_failed=True,
    )
    failed_pages = [n for n in ocr_page_numbers if n not in ocr_results]

    # ประกอบข้อความแต่ละหน้า: หน้าที่ไม่ต้อง OCR ใช้ข้อความจาก pdfplumber เดิม
    # หน้าที่ต้อง OCR ใช้ผลจาก Typhoon OCR แทน (หน้าที่ OCR ไม่สำเร็จไม่มีข้อความ)
    page_texts = {
        page.page_number: (ocr_results.get(page.page_number, "") if page.needs_ocr else page.text)
        for page in pages
    }

    report("chunk", 0, 1)
    raw_text = _join_pages_in_order(page_texts)
    cleaned = clean_text(raw_text)
    chunks = chunk_document(cleaned, doc_name=pdf_path.name)
    report("chunk", 1, 1)

    return IngestResult(
        doc=pdf_path.name,
        chunks=chunks,
        pages=len(pages),
        ocr_pages=ocr_page_numbers,
        failed_pages=failed_pages,
    )


def process_file(
    pdf_path: Path,
    force_ocr: bool = False,
    verbose: bool = False,
    on_progress: ProgressFn | None = None,
) -> IngestResult:
    """ingest ไฟล์เดียวแล้วบันทึกผลทันที (chunks.jsonl + .md + ทะเบียนเอกสาร)

    บันทึกทีละไฟล์ ถ้ารันหลายไฟล์แล้วพังกลางทาง ไฟล์ที่เสร็จไปแล้วไม่หาย
    ยังไม่ index ให้ ผู้เรียกต้องสั่ง index ต่อเอง (CLI: `campusai index --only`, API: ทำให้อัตโนมัติ)
    """
    result = ingest_single_file(pdf_path, force_ocr, verbose, on_progress)
    write_chunks({result.doc: result.chunks})
    write_debug_markdown(result.doc, result.chunks)
    registry.save_record(
        registry.DocRecord(
            doc=result.doc,
            sha256=file_hash(pdf_path),
            pages=result.pages,
            ocr_pages=result.ocr_pages,
            failed_pages=result.failed_pages,
            chunks=len(result.chunks),
        )
    )
    return result


def remove_document(doc_name: str, delete_pdf: bool = False) -> None:
    """ลบเอกสารออกจาก chunks.jsonl, ไฟล์ .md และทะเบียน (ลบใน Qdrant แยกที่ indexer)

    ไม่ลบ PDF ต้นฉบับถ้าไม่สั่ง เพราะลบแล้วกู้คืนไม่ได้ แต่ถ้าไม่ลบ `campusai ingest` รอบหน้า
    (ที่ไม่ใส่ --only) จะเอาไฟล์นี้กลับเข้ามาใหม่
    """
    write_chunks({doc_name: []})  # แทนที่ chunk ของเอกสารนี้ด้วยรายการว่าง = ลบ
    (config.PROCESSED_DIR / f"{doc_name.rsplit('.', 1)[0]}.md").unlink(missing_ok=True)
    registry.remove_record(doc_name)
    if delete_pdf:
        (config.RAW_DIR / doc_name).unlink(missing_ok=True)


def _load_existing_chunks() -> list[dict]:
    """อ่าน chunks.jsonl เดิม (ถ้ามี) เป็น list ของ dict ทีละบรรทัด"""
    if not config.CHUNKS_PATH.exists():
        return []
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_chunks(new_chunks_by_doc: dict[str, list[Chunk]]) -> None:
    """เขียน chunks.jsonl ใหม่ทั้งไฟล์ โดยแทนที่เฉพาะ chunk ของไฟล์ที่เพิ่ง ingest รอบนี้

    ทำให้รัน ingest ซ้ำกับไฟล์เดิมแล้วไม่เกิด chunk ซ้ำซ้อน (idempotent) และไฟล์อื่นที่ไม่ได้
    แตะในรอบนี้ยังคงอยู่ครบเหมือนเดิม
    """
    existing = _load_existing_chunks()
    reprocessed_docs = set(new_chunks_by_doc.keys())

    kept = [row for row in existing if row["doc"] not in reprocessed_docs]
    new_rows = [asdict(chunk) for chunks in new_chunks_by_doc.values() for chunk in chunks]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for row in kept + new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_debug_markdown(doc_name: str, chunks: list[Chunk]) -> Path:
    """เขียนไฟล์ .md แสดง chunk ทั้งหมดของเอกสารนี้ ไว้เปิดตรวจด้วยตาว่าตัดถูกไหม"""
    stem = doc_name.rsplit(".", 1)[0]
    out_path = config.PROCESSED_DIR / f"{stem}.md"

    lines = [f"# {doc_name}", ""]
    for chunk in chunks:
        heading = chunk.clause or "เกริ่นนำ"
        lines.append(f"## {heading}  ({chunk.id})")
        if chunk.chapter:
            lines.append(f"*{chunk.chapter}*")
        lines.append("")
        lines.append(chunk.text)
        lines.append("")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def run_ingest(args: argparse.Namespace) -> int:
    pdf_files = sorted(config.RAW_DIR.glob("*.pdf"))

    if args.only:
        pdf_files = [p for p in pdf_files if p.name in set(args.only)]

    if not pdf_files:
        print(f"[campusai] ไม่พบไฟล์ PDF ที่จะ ingest ใน {config.RAW_DIR}")
        return 1

    total_chunks = 0
    failed_files: list[str] = []

    for pdf_path in pdf_files:
        print(f"\n{pdf_path.name}")
        try:
            result = process_file(pdf_path, force_ocr=args.force_ocr, verbose=args.verbose)
        except Exception as exc:  # noqa: BLE001 - ไฟล์เดียวพังไม่ควรทำให้ไฟล์อื่นไม่ได้ทำ
            print(f"  ✗ ingest ไม่สำเร็จ: {exc}")
            failed_files.append(pdf_path.name)
            continue

        total_chunks += len(result.chunks)
        print(f"  ได้ {len(result.chunks)} chunk")
        if result.failed_pages:
            print(f"  ⚠ OCR ไม่สำเร็จ {len(result.failed_pages)} หน้า (ข้ามไป เนื้อหาหน้าเหล่านี้หาย): "
                  f"{result.failed_pages} รันใหม่ภายหลังเพื่อ OCR หน้าที่ขาด")
        short_chunks = [c for c in result.chunks if len(c.text) < SHORT_CHUNK_WARNING_CHARS]
        if short_chunks:
            print(f"  ⚠ พบ {len(short_chunks)} chunk ที่สั้นผิดปกติ (อาจตัดผิด): "
                  f"{[c.id for c in short_chunks]}")

    done = len(pdf_files) - len(failed_files)
    print(f"\nสำเร็จ {done}/{len(pdf_files)} ไฟล์, {total_chunks} chunk -> {config.CHUNKS_PATH}")
    if failed_files:
        print(f"ไฟล์ที่ไม่สำเร็จ: {failed_files}")
        return 1
    return 0


def add_ingest_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--force-ocr", action="store_true", help="OCR ใหม่ทุกหน้าแม้มี cache อยู่แล้ว"
    )
    sub.add_argument(
        "--only", nargs="+", metavar="ไฟล์", help="ingest เฉพาะไฟล์ที่ระบุ (ชื่อไฟล์ .pdf)"
    )
    sub.add_argument("-v", "--verbose", action="store_true", help="แสดงความคืบหน้าการ OCR รายหน้า")
