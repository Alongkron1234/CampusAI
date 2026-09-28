"""ทำความสะอาดข้อความไทยก่อนนำไปตัด chunk

ข้อความที่รับเข้ามาอาจมาจาก 2 แหล่ง:
  1. pdfplumber (หน้าที่ needs_ocr=False) - ข้อความมักสะอาดอยู่แล้ว แต่บางทีสระ/วรรณยุกต์
     สลับลำดับกัน (ปัญหาการ decompose ตัวอักษร)
  2. Typhoon OCR (หน้าที่ needs_ocr=True) - ข้อความดีมาก แต่มีคำบรรยายรูปภาพที่โมเดลใส่มาให้
     เช่น "(Handwritten signature)" เครื่องหมายคำพูดที่ไม่สม่ำเสมอ (" ผสมกับ " ")
     และที่สำคัญ: ส่วนที่เป็น "รายการนิยามศัพท์" (เช่น ข้อ 4 ในระเบียบนี้...) มักถูกแปลง
     เป็นตาราง HTML <table><tr><td>...</td></tr></table> ทำให้หัวข้อ "ข้อ N" ที่อยู่ในเซลล์
     แรกไม่ได้อยู่ต้นบรรทัดอีกต่อไป ต้องแปลงตารางกลับเป็นข้อความปกติก่อนส่งไปตัด chunk

ต้องรันฟังก์ชันเหล่านี้ก่อนส่งไปตัด chunk เสมอ เพราะ chunker.py ใช้ regex หาหัวข้อ
"ข้อ N" / "มาตรา N" ถ้าข้อความยังเพี้ยนอยู่ regex อาจหาไม่เจอ
"""

import re

# สระ-วรรณยุกต์ไทยที่เป็น combining character (ไม่มีตัวเองยืนได้ ต้องเกาะกับพยัญชนะ)
# ถ้าลำดับสลับกัน (เช่น วรรณยุกต์มาก่อนสระบน) จะเรนเดอร์ผิดตำแหน่งแม้ตัวอักษรจะถูกทุกตัว
_SARA_AM = "ำ"  # ำ
_NIKHAHIT = "ํ"  # ํ (ใช้ประกอบกับ า กลายเป็น ำ)
_SARA_AA = "า"  # า

# แท็กคำบรรยายรูปภาพที่ Typhoon OCR v1.5 ใส่มาแทนรูปที่ไม่ใช่ข้อความ
# เช่น "(Handwritten signature)", "(University logo)", "(Signature)"
_FIGURE_PLACEHOLDER_RE = re.compile(r"^\s*\([A-Za-z][A-Za-z ]*\)\s*$", re.MULTILINE)

# บรรทัดที่เป็นเลขหน้าล้วน ๆ เช่น "- 3 -", "หน้า 3", "3/10"
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(-\s*\d+\s*-|หน้า\s*\d+|\d+\s*/\s*\d+)\s*$", re.MULTILINE
)

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

_QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})

_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# ตาราง HTML ที่ Typhoon OCR v1.5 สร้างขึ้นสำหรับเนื้อหาที่ดูเป็นตาราง (เช่น รายการนิยามศัพท์)
_TABLE_RE = re.compile(r"<table>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)


def fix_sara_am(text: str) -> str:
    """แก้ 'ํ' + 'า'  ให้กลายเป็น 'ำ' ตัวเดียว

    ปัญหานี้เกิดกับข้อความที่ผ่านการ decompose unicode มา เช่น "นํา" ที่สระอำกับอาแยกกัน ควรเป็น "นำ" รวมกัน
    """
    return text.replace(_NIKHAHIT + _SARA_AA, _SARA_AM)


def thai_digits_to_arabic(text: str) -> str:
    """แปลงเลขไทย (๐-๙) เป็นเลขอารบิก (0-9) ทั้งหมด"""
    return text.translate(_THAI_DIGITS)


def normalize_quotes(text: str) -> str:
    """แปลงเครื่องหมายคำพูดโค้ง (" " ' ') ให้เป็นแบบตรง (" ') ทั้งหมด

    เอกสารจริงบางฉบับพิมพ์ผสมกัน เช่น `"มหาวิทยาลัย”` เปิดตรง ปิดโค้ง
    ทำให้ค้นหาหรือเทียบข้อความแบบ exact match พลาดถ้าไม่ normalize ก่อน
    """
    return text.translate(_QUOTE_MAP)


def flatten_html_tables(text: str) -> str:
    """แปลงตาราง HTML ของ Typhoon OCR กลับเป็นข้อความบรรทัดปกติ

    ต้องทำก่อนตัด chunk เสมอ เพราะ chunker.py หาหัวข้อ "ข้อ N" จากบรรทัดที่ขึ้นต้นด้วย
    คำนั้นจริง ๆ แต่ถ้าหัวข้อถูกฝังอยู่ใน <td>ข้อ 4 ...</td> มันจะไม่อยู่ต้นบรรทัดอีกต่อไป
    วิธีแก้: แตกแต่ละแถว (<tr>) เป็น 1 บรรทัด แล้วต่อเนื้อหาแต่ละคอลัมน์ (<td>) ด้วยช่องว่าง
    (ตัดเซลล์ว่างทิ้ง เพราะ Typhoon มักใส่ <td></td> ว่างมาคู่กับหัวข้อเสมอ)
    """

    def _flatten_one_table(match: re.Match) -> str:
        table_html = match.group(1)
        lines = []
        for row_match in _ROW_RE.finditer(table_html):
            cells = [cell.strip() for cell in _CELL_RE.findall(row_match.group(1))]
            cells = [cell for cell in cells if cell]
            if cells:
                lines.append(" ".join(cells))
        return "\n" + "\n".join(lines) + "\n"

    return _TABLE_RE.sub(_flatten_one_table, text)


def remove_figure_placeholders(text: str) -> str:
    """ลบบรรทัดที่เป็นคำบรรยายรูปภาพจาก Typhoon OCR เช่น '(Handwritten signature)'

    รูปแบบคือทั้งบรรทัดมีแค่วงเล็บครอบคำภาษาอังกฤษ ไม่ใช่เนื้อหาของระเบียบจริง
    """
    return _FIGURE_PLACEHOLDER_RE.sub("", text)


def remove_page_numbers(text: str) -> str:
    """ลบบรรทัดที่เป็นเลขหน้าล้วน ๆ เช่น '- 3 -', 'หน้า 3', '3/10'"""
    return _PAGE_NUMBER_RE.sub("", text)


def collapse_blank_lines(text: str) -> str:
    """ยุบบรรทัดว่างติดกันเกิน 2 บรรทัดให้เหลือ 2 (1 บรรทัดว่างคั่นย่อหน้า)"""
    return _MULTI_BLANK_RE.sub("\n\n", text)


def strip_trailing_whitespace(text: str) -> str:
    """ตัดช่องว่าง/แท็บท้ายบรรทัดออกทุกบรรทัด (ไม่แตะช่องว่างต้นบรรทัด เช่น การเยื้องข้อย่อย)"""
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines)


def clean_text(text: str) -> str:
    """รันทุกขั้นตอนทำความสะอาดตามลำดับที่ถูกต้อง แล้วคืนข้อความที่พร้อมตัด chunk

    ลำดับมีผล: ต้องแก้ตัวอักษร (sara am, เลขไทย, คำพูด) และแตกตาราง HTML ก่อนลบบรรทัด
    เพราะการลบบรรทัดใช้ regex ที่คาดหวังอักขระที่ถูกต้องและอยู่ในบรรทัดเดี่ยว ๆ แล้ว
    """
    text = fix_sara_am(text)
    text = thai_digits_to_arabic(text)
    text = normalize_quotes(text)
    text = flatten_html_tables(text)
    text = remove_figure_placeholders(text)
    text = remove_page_numbers(text)
    text = strip_trailing_whitespace(text)
    text = collapse_blank_lines(text)
    return text.strip()
