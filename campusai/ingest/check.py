"""คำสั่ง `campusai check`: สแกน PDF ทุกไฟล์ใน data/raw/ แล้วสรุปว่าหน้าไหนต้อง OCR"""

import argparse
from pathlib import Path

from campusai import config
from campusai.ingest.pdf_text import extract_pages


def _check_one_file(pdf_path: Path, verbose: bool) -> tuple[int, int]:
    """ตรวจ 1 ไฟล์ คืนค่า (จำนวนหน้าทั้งหมด, จำนวนหน้าที่ต้อง OCR)"""
    pages = extract_pages(pdf_path)
    ocr_pages = [p for p in pages if p.needs_ocr]

    print(f"\n{pdf_path.name}")
    print(f"  ทั้งหมด {len(pages)} หน้า, ต้อง OCR {len(ocr_pages)} หน้า")

    if verbose:
        for page in pages:
            status = "OCR" if page.needs_ocr else "OK "
            print(f"    หน้า {page.page_number:>3} [{status}] {page.reason}")

    return len(pages), len(ocr_pages)


def run_check(args: argparse.Namespace) -> int:
    pdf_files = sorted(config.RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"[campusai] ไม่พบไฟล์ PDF ใน {config.RAW_DIR}")
        return 1

    total_pages = 0
    total_ocr_pages = 0
    for pdf_path in pdf_files:
        pages, ocr_pages = _check_one_file(pdf_path, verbose=args.verbose)
        total_pages += pages
        total_ocr_pages += ocr_pages

    print(f"\nรวม {len(pdf_files)} ไฟล์, {total_pages} หน้า, ต้อง OCR {total_ocr_pages} หน้า")
    return 0


def add_check_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "-v", "--verbose", action="store_true", help="แสดงผลตรวจรายหน้า ไม่ใช่แค่สรุปรวม"
    )
