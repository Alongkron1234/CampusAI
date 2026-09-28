"""อ่านค่าตั้งค่าทั้งหมดจาก .env ให้ทุกโมดูลเรียกใช้ผ่านที่นี่ที่เดียว"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ----- Paths -----
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OCR_CACHE_DIR = DATA_DIR / "cache" / "ocr"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

EVAL_DIR = ROOT / "eval"
EVAL_QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# ----- OCR -----
# เลือก backend ได้ 2 แบบ ผ่าน .env (เหมือนที่ QDRANT_MODE เลือก local/cloud ได้):
#   "typhoon" (ค่าเริ่มต้น) - รันบนเครื่องผ่าน Ollama ฟรี ไม่มี limit แต่ช้ากว่า (~30-90s/หน้า)
#   "gemini"  - เร็วกว่ามาก (มักไม่ถึง 10s/หน้า) แต่มี free tier จำกัด/วัน และข้อมูลออกจากเครื่อง
# ทั้งสอง backend ใช้ cache + retry ชุดเดียวกัน (ocr_page/ocr_pages) จึงทนต่อการติด limit
# กลางทางได้เหมือนกัน (หยุดแล้วรันต่อได้ ไม่เสียงานที่ทำไปแล้ว)
OCR_BACKEND = os.getenv("OCR_BACKEND", "typhoon")  # "typhoon" | "gemini"

# --- Typhoon OCR ผ่าน Ollama (รันบนเครื่อง) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# Ollama native API (/api/chat) ใช้ path เดียวกันแต่ไม่มี "/v1" ต่อท้าย
# ต้องเรียกผ่าน native API เพราะ endpoint แบบ OpenAI-compatible ไม่รับค่า num_ctx จริง
# (max_tokens ที่ส่งไปทาง OpenAI-compat ควบคุมได้แค่ความยาวคำตอบ ไม่ใช่ context window รวม
#  ซึ่ง Ollama default ไว้แค่ 4096 token ทำให้หน้าที่เนื้อหาแน่นถูกตัดกลางคัน)
OLLAMA_NATIVE_URL = OLLAMA_BASE_URL.removesuffix("/v1")
OCR_MODEL = os.getenv("OCR_MODEL", "scb10x/typhoon-ocr1.5-3b")
# ไล่เพิ่มขนาด context เฉพาะตอนเจอว่าคำตอบถูกตัดกลางคัน (done_reason != "stop")
#
# เริ่มที่ 8192 ตรง ๆ (ไม่เริ่มที่ 4096 เดิม) — เคยลองเริ่มที่ 4096 ก่อนแล้วคิดว่าจะเร็วกว่า
# แต่ทดสอบจริงกับเอกสารกฎระเบียบ (ruleG2568.pdf) พบว่าเนื้อหาแน่นสม่ำเสมอเกือบทุกหน้า
# ไม่ใช่แค่บางหน้า ทำให้เกือบทุกหน้าโดนตัดที่ 4096 แล้วต้อง escalate อยู่ดี กลายเป็นเสียเวลา
# "ลองผิดก่อน" ทุกหน้าโดยเปล่าประโยชน์ (รวมช้ากว่าเริ่มที่ 8192 ตรง ๆ เกือบเท่าตัว: ~66 นาที
# เทียบกับ ~37 นาที สำหรับเอกสาร 30 หน้า) เก็บ escalation ไว้เป็นตาข่ายกันเหนียวสำหรับ
# เอกสารที่เนื้อหาแน่นกว่านี้อีก ไม่ใช่เพื่อประหยัดเวลาให้หน้าทั่วไป
OCR_NUM_CTX_LEVELS = [8192, 16384, 32768]

# ----- Embedding (รันบนเครื่อง) -----
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# ----- Vector DB: Qdrant (local หรือ cloud) -----
QDRANT_MODE = os.getenv("QDRANT_MODE", "local")  # "local" | "cloud"
QDRANT_PATH = ROOT / os.getenv("QDRANT_PATH", "data/qdrant")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = "campusai"

# ----- Generation (Gemini API) -----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
