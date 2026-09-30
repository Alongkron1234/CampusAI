"""คำสั่ง `campusai index`: อ่าน chunks.jsonl แล้วสร้าง index สำหรับค้นหา

  - Vector index: embed ทุก chunk ด้วย bge-m3 แล้วเก็บลง Qdrant (ลบของเก่าแล้วสร้างใหม่ทั้งหมด)
    หรือ `--only ไฟล์.pdf` index ใหม่เฉพาะเอกสารนั้น (เอกสารอื่นไม่แตะ)
  - BM25 index: สร้างสดในหน่วยความจำตอนค้นหาอยู่แล้ว ที่นี่แค่สร้างทดลองเพื่อยืนยันว่าพร้อมใช้
"""

import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from campusai import config
from campusai.ingest.chunker import Chunk
from campusai.ingest.pipeline import remove_document
from campusai.retrieval import bm25, vector_store
from campusai.retrieval.embedder import chunk_to_text, embed_texts


def load_chunks(path: Path | None = None) -> list[Chunk]:
    """อ่าน chunks.jsonl ทีละบรรทัด คืนเป็น list ของ Chunk

    ไม่ใส่ config.CHUNKS_PATH เป็นค่า default ตรง ๆ ในลายเซ็นฟังก์ชัน เพราะค่า default ถูกกำหนด
    ตอน import ไฟล์ครั้งเดียว ถ้าแก้ config ทีหลัง (เช่นใน test) จะไม่มีผล อ่านตอนเรียกจริงแทน
    """
    with open(path or config.CHUNKS_PATH, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def index_documents(client: QdrantClient, chunks: list[Chunk], docs: set[str]) -> int:
    """index ใหม่เฉพาะเอกสารใน docs (ลบ vector เก่าของเอกสารนั้นก่อน) เอกสารอื่นไม่แตะ

    ใช้ตอนเพิ่ม/แก้เอกสารทีละไฟล์ ไม่ต้อง embed ทุกเอกสารใหม่หมด
    เอกสารที่อยู่ใน docs แต่ไม่มี chunk แล้ว (ถูกลบ) จะถูกลบออกจาก index อย่างเดียว
    """
    vector_store.ensure_collection(client)
    for doc in docs:
        vector_store.delete_doc(client, doc)

    selected = [c for c in chunks if c.doc in docs]
    if selected:
        vectors = embed_texts([chunk_to_text(c) for c in selected])
        vector_store.index_chunks(client, selected, vectors)
    return len(selected)


def remove_documents(client: QdrantClient, docs: list[str], delete_pdf: bool = False) -> None:
    """ลบเอกสารออกทุกที่: chunks.jsonl, .md, ทะเบียน และ vector ใน Qdrant"""
    for doc in docs:
        remove_document(doc, delete_pdf=delete_pdf)
        vector_store.delete_doc(client, doc)


def run_remove(args: argparse.Namespace) -> int:
    remove_documents(vector_store.get_client(), args.docs, delete_pdf=args.delete_pdf)
    print(f"ลบแล้ว: {', '.join(args.docs)}" + (" (รวมไฟล์ PDF)" if args.delete_pdf else ""))
    return 0


def run_index(args: argparse.Namespace) -> int:
    if not config.CHUNKS_PATH.exists():
        print(f"[campusai] ไม่พบ {config.CHUNKS_PATH} (รัน `campusai ingest` ก่อน)")
        return 1

    chunks = load_chunks()
    if not chunks:
        print(f"[campusai] {config.CHUNKS_PATH} ไม่มี chunk เลย (รัน `campusai ingest` ก่อน)")
        return 1

    print(f"อ่านได้ {len(chunks)} chunk จาก {config.CHUNKS_PATH}")
    client = vector_store.get_client()

    if args.only:
        print(f"กำลัง index เฉพาะ {', '.join(args.only)} ด้วย bge-m3...")
        count = index_documents(client, chunks, set(args.only))
        print(f"Vector index: index ใหม่ {count} chunk")
    else:
        print("กำลัง embed ทุก chunk ด้วย bge-m3 (ครั้งแรกจะโหลดโมเดลก่อน)...")
        vectors = embed_texts([chunk_to_text(chunk) for chunk in chunks])
        vector_store.ensure_collection(client, recreate=True)
        vector_store.index_chunks(client, chunks, vectors)

    stored = client.count(config.QDRANT_COLLECTION).count
    print(f"Vector index: เก็บลง Qdrant ({config.QDRANT_MODE}) แล้ว {stored} chunk")

    bm25.build_index(chunks)
    print(f"BM25 index: พร้อมใช้ {len(chunks)} chunk (สร้างสดตอนค้นหา ไม่เก็บลงไฟล์)")
    return 0
