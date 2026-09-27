import hashlib
import time
from collections.abc import Callable
from pathlib import Path

from typhoon_ocr import ocr_document

from campusai import config

MAX_RETRIES = 5
RETRY_BASE_DELAY_SECONDS = 2  # รอ 2, 4, 8, 16, 32 วินาที (exponential backoff)


def file_hash(pdf_path: Path) -> str:
    """แฮช sha256 ของเนื้อไฟล์ PDF ใช้เป็นชื่อโฟลเดอร์ cache

    ใช้แฮชเนื้อไฟล์แทนชื่อไฟล์ เพราะถ้าไฟล์ถูกแก้ไข (เช่นแทนที่ด้วยสแกนใหม่)
    แฮชจะเปลี่ยน ทำให้ไม่ไปใช้ cache เก่าที่ไม่ตรงกับเนื้อไฟล์ปัจจุบันโดยไม่รู้ตัว
    """
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _model_dir_name() -> str:
    """แปลงชื่อโมเดล เช่น 'scb10x/typhoon-ocr1.5-3b' -> 'scb10x_typhoon-ocr1.5-3b'

    กัน "/" ในชื่อโมเดลไปกลายเป็นการสร้างโฟลเดอร์ย่อยโดยไม่ตั้งใจ
    """
    return config.OCR_MODEL.replace("/", "_")


def cache_path_for(pdf_path: Path, page_number: int) -> Path:
    """ตำแหน่งไฟล์ cache ของหน้านั้น ๆ

    รูปแบบ: data/cache/ocr/<hash ของไฟล์>/<ชื่อโมเดล>/<เลขหน้า>.md
    แยกตามชื่อโมเดลด้วย เผื่อวันหลังเปลี่ยนไปใช้โมเดล OCR ตัวอื่น
    จะได้ไม่เอา cache ของโมเดลเก่ามาใช้ปนกัน
    """
    return config.OCR_CACHE_DIR / file_hash(pdf_path) / _model_dir_name() / f"{page_number}.md"


def _call_model(pdf_path: Path, page_number: int) -> str:
    """เรียก Typhoon OCR จริง 1 ครั้ง (ไม่มี retry) แยกออกมาต่างหากเพื่อให้ mock ใน test ได้ง่าย"""
    return ocr_document(
        pdf_or_image_path=str(pdf_path),
        page_num=page_number,
        base_url=config.OLLAMA_BASE_URL,
        api_key="ollama",  # Ollama ไม่เช็ค key จริง แต่ openai client บังคับต้องส่งค่ามา
        model=config.OCR_MODEL,
    )


def _call_model_with_retry(pdf_path: Path, page_number: int) -> str:
    """เรียก _call_model พร้อม retry แบบเว้นระยะนานขึ้นเรื่อย ๆ เมื่อเจอ error"""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _call_model(pdf_path, page_number)
        except Exception as exc:  # noqa: BLE001 - ต้องดักทุก error จาก network/model แล้ว retry
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                print(
                    f"    [ocr] หน้า {page_number} ล้มเหลว (ครั้งที่ {attempt}/{MAX_RETRIES}): "
                    f"{exc} — รอ {delay}s แล้วลองใหม่"
                )
                time.sleep(delay)
    raise RuntimeError(
        f"OCR หน้า {page_number} ล้มเหลวครบ {MAX_RETRIES} ครั้ง: {last_error}"
    ) from last_error


def ocr_page(pdf_path: Path, page_number: int, force: bool = False) -> str:
    """OCR หน้าเดียว โดยเช็ค cache ก่อนเสมอ

    Returns:
        ข้อความ markdown ที่ OCR ได้ของหน้านั้น
    """
    cache_path = cache_path_for(pdf_path, page_number)

    if cache_path.exists() and not force:
        return cache_path.read_text(encoding="utf-8")

    text = _call_model_with_retry(pdf_path, page_number)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")

    return text


def ocr_pages(
    pdf_path: Path,
    page_numbers: list[int],
    force: bool = False,
    on_progress: Callable[[int, int, bool], None] | None = None,
) -> dict[int, str]:
    """OCR หลายหน้าของไฟล์เดียวกัน ทีละหน้า

    Args:
        page_numbers: เลขหน้าที่ต้อง OCR (ได้จาก needs_ocr() ใน pdf_text.py)
        force: True = OCR ใหม่ทุกหน้าแม้มี cache อยู่แล้ว
        on_progress: callback (หน้าที่เท่าไร, ทั้งหมดกี่หน้า, ใช้ cache หรือเรียกโมเดลจริง)
                     เรียกทุกครั้งที่ทำหน้าหนึ่งเสร็จ ไว้แสดงความคืบหน้า

    Returns:
        dict ที่ key คือเลขหน้า, value คือข้อความ markdown ที่ได้
    """
    results: dict[int, str] = {}
    total = len(page_numbers)

    for i, page_number in enumerate(page_numbers, start=1):
        was_cached = cache_path_for(pdf_path, page_number).exists() and not force
        results[page_number] = ocr_page(pdf_path, page_number, force=force)

        if on_progress:
            on_progress(i, total, was_cached)

    return results
