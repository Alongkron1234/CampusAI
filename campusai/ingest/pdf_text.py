from dataclasses import dataclass
from pathlib import Path

import pdfplumber

# สัดส่วนอักษรไทยขั้นต่ำที่ถือว่า "ข้อความดี" สำหรับเอกสารภาษาไทย
# ต่ำกว่านี้ถือว่าฟอนต์เพี้ยน (เช่น discipline2566.pdf ที่ได้ ASCII ล้วน)
MIN_THAI_RATIO = 0.15

# ข้อความที่สั้นกว่านี้ (ต่อหน้า) ถือว่าน่าจะเป็นภาพสแกนหรือหน้าที่ดึงไม่ได้
MIN_TEXT_LENGTH = 20

# ช่วงอักขระ Private Use Area ที่ฟอนต์ไทยรุ่นเก่ามัก map ผิดมาลง
PUA_RANGE = (0xF700, 0xF8FF)

# ช่วงอักขระไทยตามมาตรฐาน Unicode (U+0E00-U+0E7F)
THAI_RANGE = (0x0E00, 0x0E7F)


@dataclass
class PageResult:
    """ผลตรวจ 1 หน้า"""

    page_number: int  # เริ่มที่ 1
    text: str  # ข้อความที่ pdfplumber ดึงได้ (อาจเป็นข้อความเพี้ยนก็ได้)
    needs_ocr: bool
    reason: str  # เหตุผลที่ต้อง/ไม่ต้อง OCR เอาไว้แสดงผลและ debug


def _thai_ratio(text: str) -> float:
    """สัดส่วนอักขระไทย เทียบกับอักขระที่ไม่ใช่ช่องว่าง/เครื่องหมายวรรคตอนทั้งหมด"""
    letters = [ch for ch in text if not ch.isspace() and ch.isprintable()]
    if not letters:
        return 0.0
    thai_count = sum(1 for ch in letters if THAI_RANGE[0] <= ord(ch) <= THAI_RANGE[1])
    return thai_count / len(letters)


def needs_ocr(text: str) -> tuple[bool, str]:
    """ตัดสินว่าข้อความที่ดึงได้จาก 1 หน้า ต้องส่งไป OCR แทนหรือไม่

    Returns:
        (ต้อง OCR หรือไม่, เหตุผล)
    """
    stripped = text.strip()

    if len(stripped) < MIN_TEXT_LENGTH:
        return True, f"ข้อความสั้นเกินไป ({len(stripped)} ตัวอักษร) น่าจะเป็นภาพสแกน"

    if "(cid:" in stripped:
        return True, "พบ (cid:...) ฟอนต์ไม่มีตาราง unicode"

    pua_count = sum(1 for ch in stripped if PUA_RANGE[0] <= ord(ch) <= PUA_RANGE[1])
    if pua_count > 0:
        return True, f"พบอักขระ Private Use Area {pua_count} ตัว (ฟอนต์ไทยรุ่นเก่า)"

    ratio = _thai_ratio(stripped)
    if ratio < MIN_THAI_RATIO:
        return True, f"สัดส่วนอักษรไทยต่ำเกินไป ({ratio:.1%}) ฟอนต์น่าจะเพี้ยน"

    return False, "ข้อความใช้ได้"


def extract_pages(pdf_path: Path) -> list[PageResult]:
    """ดึงข้อความทุกหน้าของ PDF ด้วย pdfplumber แล้วตรวจแต่ละหน้าไปด้วยเลย"""
    results: list[PageResult] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            ocr_needed, reason = needs_ocr(text)
            results.append(
                PageResult(page_number=i, text=text, needs_ocr=ocr_needed, reason=reason)
            )
    return results
