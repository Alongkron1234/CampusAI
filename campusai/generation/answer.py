"""สร้างคำตอบจาก chunk ที่ค้นเจอ พร้อมอ้างอิงแหล่งที่มา [n]

ลำดับงาน: ค้นหา (hybrid) -> จัด context เป็น [1] [2] ... -> ให้ LLM ตอบจาก context เท่านั้น
-> ดึงเลข [n] ที่คำตอบอ้างจริง -> คืนเฉพาะแหล่งที่ถูกอ้าง

LLM เลือกได้ผ่าน config.LLM_BACKEND: "gemini" (API) หรือ "ollama" (บนเครื่อง)
"""

import re
import time
from dataclasses import dataclass, field

import requests
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from campusai import config
from campusai.retrieval.vector_store import SearchResult

REFUSAL = "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่มี"
TOP_K_CONTEXT = 5  # ส่ง chunk ให้ LLM กี่ชิ้น (ปรับได้ใน Issue #9)
TEMPERATURE = 0.1  # ต่ำ ให้ตอบตามเอกสาร ไม่แต่งเติม

MAX_RETRIES = 5
RETRY_BASE_DELAY_SECONDS = 2  # รอ 2, 4, 8, 16 วินาที
# 429 = ติด rate limit, 5xx = ฝั่ง Gemini มีปัญหาชั่วคราว (เคยเจอ 503 overload บ่อยตอน OCR)
# ส่วน 400/401/403/404 (key ผิด, ชื่อรุ่นผิด) ลองใหม่ก็ไม่หาย จึงไม่ retry
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

SYSTEM_PROMPT = f"""คุณคือผู้ช่วยตอบคำถามเกี่ยวกับระเบียบและข้อบังคับของมหาวิทยาลัย

กฎที่ต้องทำตาม:
1. ตอบจาก "เอกสารอ้างอิง" ที่ให้มาเท่านั้น ห้ามใช้ความรู้อื่นหรือเดา
2. ใส่เลขอ้างอิง เช่น [1] หรือ [1][3] ท้ายทุกประโยคที่เป็นข้อเท็จจริงจากเอกสาร
3. ถ้าเอกสารอ้างอิงไม่มีข้อมูลที่ตอบคำถามได้เลย ให้ตอบว่า "{REFUSAL}" เท่านั้น ไม่ต้องใส่เลขอ้างอิง
4. ถ้ามีข้อมูลเพียงบางส่วน ให้ตอบส่วนที่มี แล้วบอกชัดเจนว่าส่วนใดที่เอกสารไม่ได้ระบุไว้
5. ระบุเลขข้อของระเบียบในคำตอบเมื่อทำได้ (เช่น "ตามข้อ 20") และตอบเป็นภาษาไทย กระชับ ตรงประเด็น"""

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    sources: list[SearchResult] = field(default_factory=list)  # แหล่งที่คำตอบอ้างจริง
    contexts: list[SearchResult] = field(default_factory=list)  # ทุกชิ้นที่ส่งให้ LLM

    @property
    def refused(self) -> bool:
        return REFUSAL in self.text and not self.sources


def source_label(result: SearchResult) -> str:
    """"ruleG2568.pdf, ข้อ 20, หมวด 3 ..." ไว้ใช้ทั้งใน context และตอนแสดงแหล่งอ้างอิง"""
    return ", ".join(part for part in (result.doc, result.clause, result.chapter) if part)


def build_context(results: list[SearchResult]) -> str:
    """จัด chunk เป็นบล็อก [n] (เอกสาร, ข้อ, หมวด) ตามด้วยเนื้อหา"""
    return "\n\n".join(
        f"[{i}] ({source_label(r)})\n{r.text.strip()}" for i, r in enumerate(results, start=1)
    )


def build_prompt(question: str, results: list[SearchResult]) -> str:
    return f"เอกสารอ้างอิง:\n\n{build_context(results)}\n\nคำถาม: {question}"


def extract_citations(text: str, n_contexts: int) -> list[int]:
    """ดึงเลข [n] ที่อ้างในคำตอบ ไม่ซ้ำ เรียงตามที่พบ ตัดเลขที่ไม่มีอยู่จริง (LLM อาจอ้างมั่ว)"""
    cited: list[int] = []
    for number in _CITATION_RE.findall(text):
        n = int(number)
        if 1 <= n <= n_contexts and n not in cited:
            cited.append(n)
    return cited


def _call_gemini(prompt: str, model: str) -> str:
    """เรียก Gemini 1 ครั้ง (ไม่มี retry) แยกออกมาให้ mock ใน test ได้ง่าย"""
    if not config.GEMINI_API_KEY:
        raise RuntimeError("ยังไม่ได้ตั้งค่า GEMINI_API_KEY ใน .env")
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            # ไม่ได้ใช้ function calling ปิดไว้ (ไม่งั้น SDK พิมพ์คำเตือนทุกครั้งที่เรียก)
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return response.text or ""


def _is_daily_quota(exc: genai_errors.APIError) -> bool:
    """429 แบบโควตารายวันหมด (free tier ~20 ครั้ง/วัน/รุ่น) รอกี่วินาทีก็ไม่หาย ต่างจาก 429 รายนาที"""
    return exc.code == 429 and "PerDay" in str(exc)


def _call_model_with_retry(prompt: str, model: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _call_gemini(prompt, model)
        except genai_errors.APIError as exc:
            # retry แต่ละครั้งก็นับโควตา จึงไม่ retry กรณีที่ลองใหม่แล้วไม่มีทางหาย
            if exc.code not in RETRYABLE_STATUS or _is_daily_quota(exc) or attempt == MAX_RETRIES:
                raise
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            print(f"[campusai] {model} ตอบ {exc.code} (ครั้งที่ {attempt}) รอ {delay}s แล้วลองใหม่")
            time.sleep(delay)
    raise AssertionError("unreachable")


def _call_gemini_with_retry(prompt: str) -> str:
    """retry รุ่นหลัก ถ้ายังล่ม (429/5xx) และตั้ง GEMINI_FALLBACK_MODEL ไว้ ให้ลองรุ่นสำรองต่อ

    ทดสอบจริงพบว่ารุ่นหลักตอบ 503 (โหลดเต็ม) ติดกันหลายนาทีได้ ซึ่ง retry รุ่นเดิมไม่ช่วย
    """
    try:
        return _call_model_with_retry(prompt, config.GEMINI_MODEL)
    except genai_errors.APIError as exc:
        if exc.code not in RETRYABLE_STATUS or not config.GEMINI_FALLBACK_MODEL:
            raise
        print(f"[campusai] {config.GEMINI_MODEL} ไม่พร้อม เปลี่ยนไปใช้ {config.GEMINI_FALLBACK_MODEL}")
        return _call_model_with_retry(prompt, config.GEMINI_FALLBACK_MODEL)


# บางโมเดล (เช่น qwen3) พิมพ์ขั้นตอนคิดใน <think>...</think> ก่อนคำตอบ ต้องตัดออก
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _call_ollama(prompt: str) -> str:
    """เรียก LLM บนเครื่องผ่าน Ollama native API (/api/chat เพราะรับ num_ctx ได้จริง)

    ไม่ต้อง retry แบบ Gemini เพราะไม่มีโควตาหรือ server โหลดเต็ม ถ้าล้มแปลว่า Ollama ไม่ได้เปิด
    หรือยังไม่ได้ pull โมเดล ซึ่งลองซ้ำก็ไม่หาย
    """
    payload = {
        "model": config.OLLAMA_LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": TEMPERATURE, "num_ctx": config.OLLAMA_LLM_NUM_CTX},
        "stream": False,
    }
    response = requests.post(
        f"{config.OLLAMA_NATIVE_URL}/api/chat",
        json=payload,
        timeout=config.OLLAMA_LLM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return _THINK_RE.sub("", response.json()["message"]["content"])


def call_llm(prompt: str) -> str:
    """เรียก LLM ตาม config.LLM_BACKEND"""
    if config.LLM_BACKEND == "ollama":
        return _call_ollama(prompt)
    return _call_gemini_with_retry(prompt)


def generate_answer(question: str, results: list[SearchResult]) -> Answer:
    """ให้ LLM ตอบจาก results ถ้าไม่มี results เลย ปฏิเสธทันทีโดยไม่เรียก LLM"""
    if not results:
        return Answer(text=REFUSAL)

    text = call_llm(build_prompt(question, results)).strip()
    cited = sorted(extract_citations(text, len(results)))
    return Answer(text=text, sources=[results[n - 1] for n in cited], contexts=results)


def answer_question(question: str, retriever, top_k: int = TOP_K_CONTEXT) -> Answer:
    """ค้นหาด้วย hybrid แล้วสร้างคำตอบ (retriever คือ campusai.retrieval.hybrid.Retriever)"""
    return generate_answer(question, retriever.hybrid(question, top_k=top_k))
