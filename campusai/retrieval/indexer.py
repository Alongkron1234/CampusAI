"""คำสั่ง `campusai index`: อ่าน chunks.jsonl แล้วสร้าง index สำหรับค้นหา

  - Vector index: embed ทุก chunk ด้วย bge-m3 แล้วเก็บลง Qdrant (ลบของเก่าแล้วสร้างใหม่ทั้งหมด)
  - BM25 index: สร้างสดในหน่วยความจำตอนค้นหาอยู่แล้ว ที่นี่แค่สร้างทดลองเพื่อยืนยันว่าพร้อมใช้
"""

import argparse
import json
from pathlib import Path

from campusai import config
from campusai.ingest.chunker import Chunk
from campusai.retrieval import bm25, vector_store
from campusai.retrieval.embedder import chunk_to_text, embed_texts


def load_chunks(path: Path | None = None) -> list[Chunk]:
    """อ่าน chunks.jsonl ทีละบรรทัด คืนเป็น list ของ Chunk

    ไม่ใส่ config.CHUNKS_PATH เป็นค่า default ตรง ๆ ในลายเซ็นฟังก์ชัน เพราะค่า default ถูกกำหนด
    ตอน import ไฟล์ครั้งเดียว ถ้าแก้ config ทีหลัง (เช่นใน test) จะไม่มีผล อ่านตอนเรียกจริงแทน
    """
    with open(path or config.CHUNKS_PATH, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def run_index(args: argparse.Namespace) -> int:
    if not config.CHUNKS_PATH.exists():
        print(f"[campusai] ไม่พบ {config.CHUNKS_PATH} (รัน `campusai ingest` ก่อน)")
        return 1

    chunks = load_chunks()
    if not chunks:
        print(f"[campusai] {config.CHUNKS_PATH} ไม่มี chunk เลย (รัน `campusai ingest` ก่อน)")
        return 1

    print(f"อ่านได้ {len(chunks)} chunk จาก {config.CHUNKS_PATH}")

    print("กำลัง embed ด้วย bge-m3 (ครั้งแรกจะโหลดโมเดลก่อน)...")
    vectors = embed_texts([chunk_to_text(chunk) for chunk in chunks])

    client = vector_store.get_client()
    vector_store.ensure_collection(client, recreate=True)
    vector_store.index_chunks(client, chunks, vectors)
    stored = client.count(config.QDRANT_COLLECTION).count
    print(f"Vector index: เก็บลง Qdrant ({config.QDRANT_MODE}) แล้ว {stored} chunk")

    bm25.build_index(chunks)
    print(f"BM25 index: พร้อมใช้ {len(chunks)} chunk (สร้างสดตอนค้นหา ไม่เก็บลงไฟล์)")
    return 0
