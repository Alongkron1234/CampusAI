"""ค้นหา chunk แบบ keyword (BM25) คู่กับ vector search

BM25 ให้คะแนนตามคำที่ตรงกับคำถาม เก่งเรื่องเลขข้อและคำเฉพาะ (ที่ vector search พลาดง่าย)
แต่ไม่เข้าใจความหมาย คำพ้องหรือคำที่สะกดต่างกันจะไม่ตรงกันเลย

index สร้างสดในหน่วยความจำจาก chunk ทุกครั้งที่ใช้ ไม่เก็บลงไฟล์ (chunk มีแค่หลักร้อย
สร้างใช้เวลาไม่ถึงวินาที และไม่มีปัญหา index ล้าสมัยหลัง ingest ใหม่)
"""

from dataclasses import dataclass

from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from campusai.ingest.chunker import Chunk
from campusai.retrieval.embedder import chunk_to_text
from campusai.retrieval.vector_store import SearchResult


@dataclass
class Bm25Index:
    """ตัว index BM25 พร้อม chunk ต้นฉบับ (เรียงตรงกับลำดับที่ BM25 ให้คะแนน)"""

    bm25: BM25Okapi
    chunks: list[Chunk]

# ตัดคำไทยแล้วกรองช่องว่างและเครื่องหมายวรรคตอนทิ้ง
def tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text, engine="newmm")
    return [t.lower() for t in tokens if any(ch.isalnum() for ch in t)]

# ตัดคำทุก chunk (ข้อความชุดเดียวกับที่ embed เอกสาร+หมวด+ข้อ+เนื้อหา) แล้วสร้าง index
def build_index(chunks: list[Chunk]) -> Bm25Index:
    if not chunks:
        raise ValueError("ไม่มี chunk ให้สร้าง BM25 index (รัน `campusai ingest` ก่อน)")

    tokenized = [tokenize(chunk_to_text(chunk)) for chunk in chunks]
    return Bm25Index(bm25=BM25Okapi(tokenized), chunks=chunks)

# หา chunk ที่มีคำตรงกับคำถามมากที่สุด top_k อันดับ เรียงจากคะแนนสูงไปต่ำ
def search(index: Bm25Index, query: str, top_k: int = 5) -> list[SearchResult]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = index.bm25.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked[:top_k]:
        if scores[i] <= 0:
            break  # เรียงมากไปน้อยแล้ว ที่เหลือคะแนน 0 หมด
        chunk = index.chunks[i]
        results.append(
            SearchResult(
                chunk_id=chunk.id,
                doc=chunk.doc,
                clause=chunk.clause,
                chapter=chunk.chapter,
                text=chunk.text,
                score=float(scores[i]),
            )
        )
    return results
