"""เก็บ vector ของ chunk ลง Qdrant และค้นหา chunk ที่ความหมายใกล้คำถามที่สุด

สลับโหมดได้ผ่าน QDRANT_MODE ใน .env:
  local (ค่าเริ่มต้น) - เก็บเป็นไฟล์ที่ QDRANT_PATH ไม่ต้องมี key
  cloud               - ใช้ Qdrant Cloud ต้องตั้ง QDRANT_URL และ QDRANT_API_KEY

ฟังก์ชันทุกตัวรับ client เข้ามาตรง ๆ (ไม่สร้างเองข้างใน) เพื่อให้ test ส่ง Qdrant แบบ
in-memory เข้ามาแทนได้ ไม่ต้องแตะไฟล์หรือเน็ตจริง
"""

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from campusai import config
from campusai.ingest.chunker import Chunk
from campusai.retrieval.embedder import EMBEDDING_DIM


@dataclass
class SearchResult:
    """ผลการค้นหา 1 รายการ: chunk ที่เจอ พร้อมคะแนนความใกล้เคียง (cosine similarity)"""

    chunk_id: str
    doc: str
    clause: str | None
    chapter: str | None
    text: str
    score: float


def get_client() -> QdrantClient:
    """สร้างตัวเชื่อม Qdrant ตามโหมดใน config.QDRANT_MODE"""
    if config.QDRANT_MODE == "cloud":
        if not config.QDRANT_URL or not config.QDRANT_API_KEY:
            raise RuntimeError(
                "QDRANT_MODE=cloud แต่ยังไม่ได้ตั้งค่า QDRANT_URL / QDRANT_API_KEY ใน .env"
            )
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.QDRANT_PATH))

# เหมือนเป็นตู้เก็บ vector ของ chunk ทั้งหมดใน Qdrant (เหมือน table ใน database)
def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    name = config.QDRANT_COLLECTION
    exists = client.collection_exists(name)

    if exists and recreate:
        client.delete_collection(name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

# แปลง chunk id เป็น UUID ที่ Qdrant รับได้
def chunk_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

# เก็บ chunk พร้อม vector ลง Qdrant (chunks กับ vectors ต้องเรียงตรงกันทีละตัว)
def index_chunks(client: QdrantClient, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    if len(chunks) != len(vectors):
        raise ValueError(f"จำนวน chunk ({len(chunks)}) ไม่เท่ากับจำนวน vector ({len(vectors)})")

    points = [
        PointStruct(
            id=chunk_point_id(chunk.id),
            vector=vector,
            payload={
                "chunk_id": chunk.id,
                "doc": chunk.doc,
                "clause": chunk.clause,
                "chapter": chunk.chapter,
                "text": chunk.text,
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)

def delete_doc(client: QdrantClient, doc: str) -> None:
    """ลบ vector ทุกชิ้นของเอกสารนี้ (กรองจาก payload "doc")

    ใช้ตอน index เอกสารเดิมซ้ำ: ต้องลบของเก่าก่อน เพราะถ้าเอกสารใหม่มี chunk น้อยกว่าเดิม
    upsert อย่างเดียวจะทับแค่ id ที่ซ้ำ chunk เก่าที่เกินมาจะค้างอยู่
    """
    if not client.collection_exists(config.QDRANT_COLLECTION):
        return
    client.delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc", match=MatchValue(value=doc))])
        ),
    )


# หา chunk ที่ใกล้ query_vector ที่สุด top_k อันดับ เรียงจากใกล้ไปไกล (ส่งคำถามไปได้ ผลกลัยมา)
def search(client: QdrantClient, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
    response = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        SearchResult(
            chunk_id=point.payload["chunk_id"],
            doc=point.payload["doc"],
            clause=point.payload["clause"],
            chapter=point.payload["chapter"],
            text=point.payload["text"],
            score=point.score,
        )
        for point in response.points
    ]
