import torch
from sentence_transformers import SentenceTransformer

from campusai import config
from campusai.ingest.chunker import Chunk

EMBEDDING_DIM = 1024  # ขนาด vector ที่ bge-m3 สร้างออกมา ใช้ตอนสร้าง Qdrant collection

_model: SentenceTransformer | None = None  # โหลดครั้งเดียว (lazy singleton) เพราะโหลดโมเดลช้า


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _silence_model_loading() -> None:
    """ปิดคำเตือน HF_TOKEN และแถบ "Loading weights" ที่ขึ้นทุกครั้งที่โหลดโมเดล (ไม่มีผลต่อการทำงาน)"""
    from huggingface_hub.utils import logging as hf_logging
    from transformers.utils import logging as tf_logging

    hf_logging.set_verbosity_error()
    tf_logging.set_verbosity_error()
    tf_logging.disable_progress_bar()


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _silence_model_loading()
        _model = SentenceTransformer(config.EMBED_MODEL, device=_select_device())
    return _model

# ประกอบข้อความสำหรับ embed จาก Chunk: ชื่อเอกสาร + หมวด + เลขข้อ + เนื้อหา
def chunk_to_text(chunk: Chunk) -> str:
    parts = [chunk.doc]
    if chunk.chapter:
        parts.append(chunk.chapter)
    if chunk.clause:
        parts.append(chunk.clause)
    parts.append(chunk.text)
    return " | ".join(parts)

# แปลง list ของข้อความเป็น list ของ vector (normalize แล้ว พร้อมใช้ cosine similarity)
def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()

# ใช้ตอน search แปลงตำถามเป็น vector เพื่อใช้ค้นหาใน Qdrant
def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
