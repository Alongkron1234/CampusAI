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

# ----- OCR (Typhoon OCR ผ่าน Ollama, รันบนเครื่อง) -----
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OCR_MODEL = os.getenv("OCR_MODEL", "scb10x/typhoon-ocr1.5-3b")

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
