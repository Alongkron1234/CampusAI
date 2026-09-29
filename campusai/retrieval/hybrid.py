"""รวมผลค้นหาจาก vector search และ BM25 เข้าด้วยกัน (hybrid search)

รวมด้วย Reciprocal Rank Fusion (RRF): ใช้ "อันดับ" ไม่ใช้ "คะแนน" เพราะคะแนน cosine (0-1)
กับคะแนน BM25 (ไม่มีเพดาน) เทียบกันตรง ๆ ไม่ได้

นอกจากนี้ ถ้าคำถามระบุเลขข้อ (เช่น "ข้อ 15", "ข้อ ๑๕", "ข้อ 17.2.5") จะเพิ่มผลการจัดอันดับชุดที่ 3
คือ "chunk ที่เลขข้อตรงกับที่ถาม" เข้าไปรวมใน RRF ด้วย เพราะวัดแล้วพบว่าทั้ง vector และ BM25
จับเลขข้อได้ไม่แม่น (เลขไม่มีความหมายเชิงภาษา และคำว่า "ข้อ" อยู่เกือบทุก chunk)
"""

import re
from dataclasses import dataclass, replace

from qdrant_client import QdrantClient

from campusai.ingest.chunker import Chunk
from campusai.retrieval import bm25, vector_store
from campusai.retrieval.embedder import embed_query
from campusai.retrieval.vector_store import SearchResult

RRF_K = 60  # ค่ามาตรฐานของ RRF กันไม่ให้อันดับต้น ๆ ได้เปรียบเกินจริง
CANDIDATES_PER_RETRIEVER = 20  # ดึงจากแต่ละระบบกี่อันดับก่อนเอามารวม

# น้ำหนักของแต่ละระบบใน RRF เลือกจากการวัดด้วยชุดคำถาม 44 ข้อ (campusai eval retrieval):
#   BM25 น้ำหนักเท่า vector (1.0)   -> MRR 0.825  (แย่กว่า vector อย่างเดียว 0.875)
#   BM25 0.3 / 0.1                   -> MRR 0.838 / 0.854 (ยังแย่กว่า)
#   BM25 0 + ดันตามเลขข้อ            -> MRR 0.884  (ดีที่สุด)
# BM25 ช่วยเฉพาะคำถามที่ระบุเลขข้อ ซึ่งการดันตามเลขข้อทำได้ดีกว่าอยู่แล้ว แต่ดึงคำถามเชิงความหมาย
# ลงมาก จึงปิดไว้ (0) แต่ยังเก็บโค้ดไว้ ถ้าเอกสาร/คำถามเปลี่ยนให้วัดใหม่แล้วปรับค่าตรงนี้
# ข้อควรระวัง: ชุดวัดเล็กและเขียนโดยคนออกแบบระบบเอง ค่านี้อาจ overfit กับชุดคำถามนี้
VECTOR_WEIGHT = 1.0
BM25_WEIGHT = 0.0
CLAUSE_WEIGHT = 1.0

# "ข้อ 15", "ข้อ ๑๕", "มาตรา 18", "ข้อ 17.2.5" (ข้อย่อย 17.2.5 เป็นส่วนหนึ่งของข้อ 17)
_CLAUSE_REF_RE = re.compile(r"(ข้อ|มาตรา)\s*([0-9๐-๙]+)")
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def extract_clause_refs(query: str) -> list[str]:
    """ดึงเลขข้อที่คำถามพูดถึง คืนเป็นป้ายชื่อแบบเดียวกับ Chunk.clause เช่น ["ข้อ 15"]

    เรียงตามลำดับที่เจอในคำถาม ไม่ซ้ำ
    """
    refs: list[str] = []
    for marker, number in _CLAUSE_REF_RE.findall(query):
        label = f"{marker} {int(number.translate(_THAI_DIGITS))}"
        if label not in refs:
            refs.append(label)
    return refs


def _clause_base(clause: str | None) -> str | None:
    """ "ข้อ 5/1" -> "ข้อ 5" ให้คำถาม "ข้อ 5" ตรงกับข้อที่แทรกเพิ่มทีหลังด้วย"""
    if clause is None:
        return None
    return clause.split("/")[0]


def rrf_fuse(
    rankings: list[list[SearchResult]], weights: list[float] | None = None, k: int = RRF_K
) -> list[SearchResult]:
    """รวมหลายรายการผลค้นหาด้วย RRF คืน list ใหม่เรียงตามคะแนนรวม (score = คะแนน RRF)

    คะแนนของ chunk = ผลรวมของ weight / (k + อันดับ) จากทุกรายการที่เจอ chunk นั้น
    chunk ที่หลายระบบเห็นตรงกันจึงลอยขึ้นบน ส่วน chunk ที่เจอแค่ระบบเดียวยังถูกนับอยู่
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    first_seen: dict[str, SearchResult] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + weight / (k + rank)
            first_seen.setdefault(result.chunk_id, result)

    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [replace(first_seen[cid], score=scores[cid]) for cid in ordered]


def clause_match_ranking(
    refs: list[str], candidates: list[SearchResult], all_chunks: list[Chunk]
) -> list[SearchResult]:
    """สร้างผลการจัดอันดับชุดที่ 3: chunk ที่เลขข้อตรงกับที่คำถามระบุ

    กรณีหลายเอกสารมีข้อเลขเดียวกัน (เช่น "ข้อ 6" มีทั้งสองไฟล์) ให้เอกสารที่ผลค้นหาชี้ไปมากที่สุด
    ขึ้นก่อน (คะแนนเอกสาร = ผลรวม 1/(k+อันดับ) ของ chunk จากเอกสารนั้นใน candidates)
    เช่น "ข้อ 6 ของระเบียบการศึกษาบัณฑิตศึกษา" ผลค้นหาส่วนใหญ่อยู่ในระเบียบบัณฑิตศึกษา จึงเลือกข้อ 6
    ของไฟล์นั้น ไม่ใช่ข้อ 6 ของข้อบังคับวินัย ภายในเอกสารเดียวกันยังคงลำดับเดิม
    """
    if not refs:
        return []
    wanted = set(refs)

    doc_score: dict[str, float] = {}
    for rank, r in enumerate(candidates, start=1):
        doc_score[r.doc] = doc_score.get(r.doc, 0.0) + 1 / (RRF_K + rank)

    ranking = [r for r in candidates if _clause_base(r.clause) in wanted]
    seen = {r.chunk_id for r in ranking}
    for chunk in all_chunks:
        if chunk.id not in seen and _clause_base(chunk.clause) in wanted:
            ranking.append(
                SearchResult(
                    chunk_id=chunk.id,
                    doc=chunk.doc,
                    clause=chunk.clause,
                    chapter=chunk.chapter,
                    text=chunk.text,
                    score=0.0,
                )
            )
            seen.add(chunk.id)

    # sorted เป็น stable sort: เอกสารคะแนนเท่ากันหรือ chunk ในเอกสารเดียวกันยังคงลำดับเดิม
    return sorted(ranking, key=lambda r: doc_score.get(r.doc, 0.0), reverse=True)


@dataclass
class Retriever:
    """รวมระบบค้นหาทั้งหมดไว้ที่เดียว: vector, BM25 และ hybrid"""

    client: QdrantClient
    bm25_index: bm25.Bm25Index

    @property
    def chunks(self) -> list[Chunk]:
        return self.bm25_index.chunks

    def vector(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return vector_store.search(self.client, embed_query(query), top_k=top_k)

    def keyword(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return bm25.search(self.bm25_index, query, top_k=top_k)

    def hybrid(self, query: str, top_k: int = 5) -> list[SearchResult]:
        rankings = [self.vector(query, top_k=CANDIDATES_PER_RETRIEVER)]
        weights = [VECTOR_WEIGHT]
        if BM25_WEIGHT > 0:
            rankings.append(self.keyword(query, top_k=CANDIDATES_PER_RETRIEVER))
            weights.append(BM25_WEIGHT)

        refs = extract_clause_refs(query)
        if refs:
            fused_without_clause = rrf_fuse(rankings, weights)
            rankings.append(clause_match_ranking(refs, fused_without_clause, self.chunks))
            weights.append(CLAUSE_WEIGHT)

        return rrf_fuse(rankings, weights)[:top_k]
