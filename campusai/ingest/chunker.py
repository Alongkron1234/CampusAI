"""แบ่งเอกสารที่ทำความสะอาดแล้ว (จาก clean.py) ออกเป็น chunk ตามโครงสร้าง "ข้อ/มาตรา"

แนวคิดหลัก: 1 ข้อ/มาตรา = 1 chunk (ยกเว้นถ้ายาวเกินจะซอยต่อ) เพื่อให้แต่ละ chunk
มีความหมายสมบูรณ์ในตัวเองและอ้างอิงได้ชัดว่าเป็นข้อไหน ถ้าเอกสารไม่มีโครงสร้างนี้เลย
จะ fallback ไปตัดตามจำนวนตัวอักษรแทน (ไม่ใช้ "คำ" เพราะภาษาไทยไม่มีช่องว่างคั่นคำ)
"""

import re
from dataclasses import dataclass

# หัวข้อ "ข้อ N" หรือ "มาตรา N" ต้องอยู่ต้นบรรทัดเท่านั้น (กันจับประโยคอ้างอิงกลางบรรทัด
# เช่น "อาศัยอำนาจตามความในมาตรา 18 (2) แห่ง..." ซึ่งไม่ได้ขึ้นต้นบรรทัดด้วยคำว่า "มาตรา")
# รองรับเลขซ้อนแบบ "ข้อ 5/1" (ข้อที่แทรกเพิ่มทีหลัง)
_CLAUSE_RE = re.compile(r"^\s*(ข้อ|มาตรา)\s+(\d+(?:/\d+)?)\s*", re.MULTILINE)

# หัวข้อ "หมวด N ชื่อหมวด" หรือ "ส่วนที่ N ชื่อส่วน" ใช้ตามรอยว่า chunk นี้อยู่หมวดไหน
_CHAPTER_RE = re.compile(r"^\s*(หมวด|ส่วนที่)\s+(\d+)\s*(.*)$", re.MULTILINE)

MAX_CHUNK_CHARS = 1200  # ข้อที่ยาวเกินนี้จะถูกซอยต่อ
OVERLAP_CHARS = 200  # ส่วนซ้อนทับระหว่างชิ้นที่ซอย กันบริบทขาดตอนตรงรอยตัด

# ค่าสำหรับ fallback เมื่อเอกสารไม่มีโครงสร้างข้อ/มาตราเลย
FALLBACK_CHUNK_CHARS = 1000
FALLBACK_OVERLAP_CHARS = 200


@dataclass
class Chunk:
    """1 หน่วยเนื้อหาที่พร้อมนำไปทำ embedding และค้นหาต่อ"""

    id: str
    doc: str  # ชื่อไฟล์ต้นฉบับ เช่น "ruleG2568.pdf"
    clause: str | None  # เช่น "ข้อ 15" หรือ None ถ้าไม่มีโครงสร้าง/เป็นส่วนเกริ่นนำ
    chapter: str | None  # เช่น "หมวด 2 ระบบการจัดการศึกษา" หรือ None ถ้าไม่มี
    text: str


def _leading_int(number_text: str) -> int:
    """เอาเลขหลักหน้าสุดจาก '5/1' -> 5 หรือจาก '5' -> 5 ไว้เทียบลำดับข้อ"""
    return int(number_text.split("/")[0])


def _doc_slug(doc_name: str) -> str:
    """ตัดนามสกุลไฟล์และแทนอักขระที่ใช้เป็นชื่อไฟล์/id ไม่ได้ด้วย _"""
    stem = doc_name.rsplit(".", 1)[0]
    return re.sub(r"[^\w]+", "_", stem, flags=re.UNICODE)


def _clause_slug(clause_label: str) -> str:
    """'ข้อ 5/1' -> 'ข้อ_5_1' ไว้ใช้เป็นส่วนหนึ่งของ id"""
    return re.sub(r"[^\w]+", "_", clause_label, flags=re.UNICODE)


def _split_long_text(text: str, max_len: int, overlap: int) -> list[str]:
    """ซอยข้อความยาวเป็นชิ้น ๆ ตามจำนวนตัวอักษร โดยให้แต่ละชิ้นซ้อนทับกัน `overlap` ตัวอักษร

    ใช้ตัวอักษรแทนคำ เพราะภาษาไทยไม่มีช่องว่างคั่นคำที่ใช้อ้างอิงได้แน่นอน
    """
    if len(text) <= max_len:
        return [text]

    step = max_len - overlap
    parts = []
    start = 0
    while start < len(text):
        end = start + max_len
        parts.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return parts


def _chapter_label_at(chapter_matches: list[re.Match], position: int) -> str | None:
    """หาว่า ณ ตำแหน่ง `position` ในเอกสาร กำลังอยู่ภายใต้หมวดไหน

    คือหมวดล่าสุดที่ตำแหน่งเริ่มต้นของมันมาก่อน `position` (ไล่หาตัวสุดท้ายที่ผ่านมาแล้ว)
    """
    current = None
    for match in chapter_matches:
        if match.start() > position:
            break
        marker, number, title = match.group(1), match.group(2), match.group(3).strip()
        current = f"{marker} {number} {title}".strip()
    return current


def _make_chunks_for_clause(
    doc_name: str, clause_label: str | None, chapter: str | None, text: str
) -> list[Chunk]:
    """สร้าง Chunk จากเนื้อหา 1 ข้อ ถ้ายาวเกิน MAX_CHUNK_CHARS จะซอยเป็นหลาย Chunk"""
    doc_slug = _doc_slug(doc_name)
    label_slug = _clause_slug(clause_label) if clause_label else "เกริ่นนำ"

    parts = _split_long_text(text, MAX_CHUNK_CHARS, OVERLAP_CHARS)

    if len(parts) == 1:
        chunk_id = f"{doc_slug}_{label_slug}"
        return [Chunk(id=chunk_id, doc=doc_name, clause=clause_label, chapter=chapter, text=text)]

    return [
        Chunk(
            id=f"{doc_slug}_{label_slug}_{i + 1}",
            doc=doc_name,
            clause=clause_label,
            chapter=chapter,
            text=part,
        )
        for i, part in enumerate(parts)
    ]


def _accept_clause_boundaries(text: str) -> list[tuple[int, str]]:
    """หาตำแหน่งหัวข้อ "ข้อ/มาตรา N" ที่ควรใช้เป็นจุดตัด chunk จริง

    กันบรรทัดอ้างอิงลวงที่บังเอิญขึ้นต้นบรรทัดด้วยคำว่า "ข้อ N" แต่จริง ๆ เป็นการอ้างถึง
    ข้อที่ผ่านมาแล้ว (เลขย้อนหลัง) ด้วยการรับเฉพาะหัวข้อที่เลขไม่ลดลงจากข้อก่อนหน้าเท่านั้น
    """
    accepted: list[tuple[int, str]] = []
    last_num: int | None = None

    for match in _CLAUSE_RE.finditer(text):
        marker, number_text = match.group(1), match.group(2)
        num = _leading_int(number_text)

        if last_num is not None and num < last_num:
            continue  # เลขย้อนหลัง = บรรทัดอ้างอิงลวง ไม่ใช่หัวข้อใหม่จริง

        accepted.append((match.start(), f"{marker} {number_text}"))
        last_num = num

    return accepted


def chunk_document(text: str, doc_name: str) -> list[Chunk]:
    """แบ่งเอกสาร (ข้อความที่ clean_text() แล้ว) เป็น chunk ตามข้อ/มาตรา

    ถ้าไม่พบโครงสร้างข้อ/มาตราเลย จะ fallback ไปตัดตามจำนวนตัวอักษรแทน
    """
    boundaries = _accept_clause_boundaries(text)

    if not boundaries:
        return _chunk_fallback(text, doc_name)

    chapter_matches = list(_CHAPTER_RE.finditer(text))
    chunks: list[Chunk] = []

    # เนื้อหาก่อนหัวข้อแรก (คำนำ/เหตุผลในการออกระเบียบ) เก็บไว้เป็น chunk ต่างหากถ้ามีเนื้อหาจริง
    # ต้องตัดบรรทัดหัวข้อ "หมวด N" ออกก่อนเช็ค เพราะบางเอกสารมีแค่บรรทัดหมวดอยู่ก่อนข้อแรก
    # โดยไม่มีเนื้อหาเกริ่นนำจริง ๆ (ไม่งั้นจะได้ chunk เปล่า ๆ ที่มีแต่บรรทัดหมวด)
    preamble_raw = text[: boundaries[0][0]]
    preamble = _CHAPTER_RE.sub("", preamble_raw).strip()
    if preamble:
        chapter = _chapter_label_at(chapter_matches, 0)
        chunks.extend(_make_chunks_for_clause(doc_name, None, chapter, preamble))

    for i, (start, clause_label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        clause_text = text[start:end].strip()
        chapter = _chapter_label_at(chapter_matches, start)
        chunks.extend(_make_chunks_for_clause(doc_name, clause_label, chapter, clause_text))

    return chunks


def _chunk_fallback(text: str, doc_name: str) -> list[Chunk]:
    """ตัดตามจำนวนตัวอักษรล้วน ๆ สำหรับเอกสารที่ไม่มีโครงสร้างข้อ/มาตรา"""
    doc_slug = _doc_slug(doc_name)
    parts = _split_long_text(text, FALLBACK_CHUNK_CHARS, FALLBACK_OVERLAP_CHARS)
    return [
        Chunk(id=f"{doc_slug}_chunk{i + 1}", doc=doc_name, clause=None, chapter=None, text=part)
        for i, part in enumerate(parts)
    ]
