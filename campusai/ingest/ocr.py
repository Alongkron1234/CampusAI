import base64
import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import requests
from google import genai
from google.genai import types as genai_types
from typhoon_ocr.ocr_utils import get_prompt, render_pdf_to_base64png

from campusai import config

MAX_RETRIES = 5
RETRY_BASE_DELAY_SECONDS = 2  # รอ 2, 4, 8, 16, 32 วินาที (exponential backoff)

OCR_REQUEST_TIMEOUT_SECONDS = 300

# ใช้ prompt เดียวกับ Typhoon ไม่ว่าจะเลือก backend ไหน เพื่อให้ output ออกมารูปแบบเดียวกัน
# (แท็ก <table>, <page_number>, <figure> ฯลฯ) clean.py จะได้ไม่ต้องรู้เลยว่าข้อความมาจากไหน
#
# เพิ่มบรรทัดห้ามใส่ markdown header (#, ##, ###) เอง เพราะทดสอบจริงพบว่า Gemini ชอบใส่
# "### หมวด 2 ..." นำหน้าหัวข้อหมวด (Typhoon ไม่ทำแบบนี้อยู่แล้ว) ทำให้ chunker.py หา
# หัวข้อ "หมวด N" ไม่เจอ (regex บังคับให้ต้องอยู่ต้นบรรทัดพอดี ไม่มี "#" นำหน้า)
_OCR_PROMPT = get_prompt("v1.5")(figure_language="Thai") + (
    "\n\n- Do not add markdown headers (#, ##, ###, etc.) to any text, including section or "
    "chapter titles. Keep every line as plain text exactly as it appears in the document."
)


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
    """ชื่อโฟลเดอร์ cache ที่ผูกกับทั้ง backend และชื่อโมเดลที่ใช้จริง

    เช่น "typhoon_scb10x_typhoon-ocr1.5-3b" หรือ "gemini_gemini-3.8-flash"
    ต้องผูกกับ backend ด้วย (ไม่ใช่แค่ชื่อโมเดล) เพราะสลับ OCR_BACKEND ใน .env
    ไม่ควรไปใช้ cache ของ backend อื่นปนกัน (คุณภาพและรูปแบบผลลัพธ์อาจต่างกัน)
    """
    model_name = config.GEMINI_MODEL if config.OCR_BACKEND == "gemini" else config.OCR_MODEL
    return f"{config.OCR_BACKEND}_{model_name}".replace("/", "_")


def cache_path_for(pdf_path: Path, page_number: int) -> Path:
    """ตำแหน่งไฟล์ cache ของหน้านั้น ๆ

    รูปแบบ: data/cache/ocr/<hash ของไฟล์>/<ชื่อโมเดล>/<เลขหน้า>.md
    แยกตามชื่อโมเดลด้วย เผื่อวันหลังเปลี่ยนไปใช้โมเดล OCR ตัวอื่น
    จะได้ไม่เอา cache ของโมเดลเก่ามาใช้ปนกัน
    """
    return config.OCR_CACHE_DIR / file_hash(pdf_path) / _model_dir_name() / f"{page_number}.md"


def _call_model_once_typhoon(pdf_path: Path, page_number: int, num_ctx: int) -> tuple[str, bool]:
    """เรียก Typhoon OCR 1 ครั้งด้วยค่า num_ctx ที่กำหนด ผ่าน Ollama native API (/api/chat)

    ต้องเรียกผ่าน native API แทนที่จะผ่าน endpoint แบบ OpenAI-compatible (ที่ package
    typhoon-ocr ใช้เป็นค่าเริ่มต้น) เพราะมีแค่ native API เท่านั้นที่รับค่า num_ctx จริง
    ถ้าใช้ endpoint แบบ OpenAI-compatible, Ollama จะ default num_ctx (ขนาด context รวม
    prompt+ภาพ+คำตอบ) ไว้แค่ 4096 token เสมอ ไม่ว่าจะส่ง max_tokens เท่าไรก็ตาม ทำให้หน้าที่
    เนื้อหาแน่น (ภาพกิน token เยอะ) ถูกตัดคำตอบกลางคันโดยไม่มีสัญญาณเตือนอะไรเลย

    Returns:
        (ข้อความที่ได้, ถูกตัดกลางคันหรือไม่ [True = โดนตัด ควรลองใหม่ด้วย num_ctx ที่สูงขึ้น])
    """
    image_base64 = render_pdf_to_base64png(
        str(pdf_path), page_number, target_longest_image_dim=1800
    )

    payload = {
        "model": config.OCR_MODEL,
        "messages": [{"role": "user", "content": _OCR_PROMPT, "images": [image_base64]}],
        "options": {"num_ctx": num_ctx, "temperature": 0.1, "repeat_penalty": 1.1},
        "stream": False,
    }
    response = requests.post(
        f"{config.OLLAMA_NATIVE_URL}/api/chat", json=payload, timeout=OCR_REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    data = response.json()

    text = data["message"]["content"]
    was_truncated = data.get("done_reason") != "stop"
    return text, was_truncated


def _call_model_typhoon(pdf_path: Path, page_number: int) -> str:
    """เรียก Typhoon OCR (บนเครื่องผ่าน Ollama) โดยไล่เพิ่ม num_ctx อัตโนมัติถ้าคำตอบถูกตัดกลางคัน

    เริ่มที่ค่าแรกใน OCR_NUM_CTX_LEVELS (ค่า default ของ Ollama) เสมอ เพื่อให้หน้าส่วนใหญ่
    ที่ไม่มีปัญหายังเร็วเท่าเดิม มีแค่หน้าที่เนื้อหาแน่นจริง ๆ เท่านั้นที่จะโดนลองซ้ำด้วยค่าที่สูงขึ้น
    """
    text = ""
    for num_ctx in config.OCR_NUM_CTX_LEVELS:
        text, was_truncated = _call_model_once_typhoon(pdf_path, page_number, num_ctx)
        if not was_truncated:
            return text
        print(f"    [ocr] หน้า {page_number} เนื้อหาถูกตัด (num_ctx={num_ctx}) ลองเพิ่ม context...")

    # ลองสูงสุดในลิสต์แล้วยังโดนตัดอยู่ ยอมคืนผลลัพธ์เท่าที่ได้ (ดีกว่าไม่ได้อะไรเลย)
    # แต่แจ้งเตือนไว้ชัดเจนเผื่อต้องปรับ OCR_NUM_CTX_LEVELS ให้สูงขึ้นอีก
    print(f"    [ocr] ⚠ หน้า {page_number} ยังถูกตัดอยู่แม้ใช้ num_ctx สูงสุด "
          f"({config.OCR_NUM_CTX_LEVELS[-1]}) เนื้อหาอาจไม่ครบ")
    return text


def _call_model_gemini(pdf_path: Path, page_number: int) -> str:
    """เรียก Gemini ให้ OCR หน้าเดียว เร็วกว่า Typhoon มาก แต่มี free tier จำกัด/วัน

    ใช้ prompt เดียวกับ Typhoon (_OCR_PROMPT) เพื่อให้ output ออกมาในรูปแบบเดียวกัน
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError("OCR_BACKEND=gemini แต่ยังไม่ได้ตั้งค่า GEMINI_API_KEY ใน .env")

    image_base64 = render_pdf_to_base64png(
        str(pdf_path), page_number, target_longest_image_dim=1800
    )
    image_bytes = base64.b64decode(image_base64)

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[
            genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            _OCR_PROMPT,
        ],
    )
    return response.text


def _call_model(pdf_path: Path, page_number: int) -> str:
    """เรียกโมเดล OCR จริง 1 ครั้ง (ไม่มี retry เรื่อง network) เลือก backend ตาม config.OCR_BACKEND

    แยกออกมาต่างหากเพื่อให้ mock ใน test ได้ง่าย
    """
    if config.OCR_BACKEND == "gemini":
        return _call_model_gemini(pdf_path, page_number)
    return _call_model_typhoon(pdf_path, page_number)


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
    skip_failed: bool = False,
) -> dict[int, str]:
    """OCR หลายหน้าของไฟล์เดียวกัน ทีละหน้า

    Args:
        page_numbers: เลขหน้าที่ต้อง OCR (ได้จาก needs_ocr() ใน pdf_text.py)
        force: True = OCR ใหม่ทุกหน้าแม้มี cache อยู่แล้ว
        on_progress: callback (หน้าที่เท่าไร, ทั้งหมดกี่หน้า, ใช้ cache หรือเรียกโมเดลจริง)
                     เรียกทุกครั้งที่ทำหน้าหนึ่งเสร็จ ไว้แสดงความคืบหน้า
        skip_failed: True = หน้าที่ OCR ไม่สำเร็จ (retry ครบแล้ว) ให้ข้ามไปทำหน้าถัดไป
                     แทนที่จะหยุดทั้งไฟล์ หน้าที่ข้ามจะไม่อยู่ใน dict ที่คืน (ผู้เรียกเช็คเองได้)

    Returns:
        dict ที่ key คือเลขหน้า, value คือข้อความ markdown ที่ได้
    """
    results: dict[int, str] = {}
    total = len(page_numbers)

    for i, page_number in enumerate(page_numbers, start=1):
        was_cached = cache_path_for(pdf_path, page_number).exists() and not force
        try:
            results[page_number] = ocr_page(pdf_path, page_number, force=force)
        except RuntimeError as exc:
            if not skip_failed:
                raise
            print(f"    [ocr] ⚠ ข้ามหน้า {page_number}: {exc}")

        if on_progress:
            on_progress(i, total, was_cached)

    return results
