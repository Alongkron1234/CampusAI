"""Web API ของ CampusAI (FastAPI) เปิดด้วย `campusai serve`

Endpoint ทั้งหมดอยู่ใต้ /api:
  GET    /api/health                 เช็คว่าเซิร์ฟเวอร์ทำงาน + ใช้ LLM อะไร
  POST   /api/ask                    ถามคำถาม -> คำตอบ + แหล่งอ้างอิง
  GET    /api/documents              รายการเอกสาร + สถานะ
  POST   /api/documents              อัปโหลด PDF -> เข้าคิว ingest (ตอบ 202 ทันที ไม่รอ OCR)
  DELETE /api/documents/{doc}        ลบเอกสาร
  GET    /api/jobs, /api/jobs/{id}   ติดตามสถานะงาน ingest (หน้าเว็บเรียกซ้ำเป็นระยะ)

ไม่มีระบบ login ตั้งใจให้รันบนเครื่องตัวเอง (ค่าเริ่มต้นฟังแค่ 127.0.0.1)
"""

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from campusai import config
from campusai.api.jobs import JobStore
from campusai.api.service import CampusService, ConflictError, UploadError
from campusai.generation.answer import TOP_K_CONTEXT, source_label

# หน้าเว็บ Next.js ตอนพัฒนารันที่ port 3000 คนละ origin กับ API จึงต้องอนุญาต CORS
FRONTEND_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=TOP_K_CONTEXT, ge=1, le=20)


class Source(BaseModel):
    n: int  # เลขเดียวกับ [n] ในคำตอบ
    doc: str
    clause: str | None
    chapter: str | None
    label: str
    text: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]


def default_service() -> CampusService:
    from campusai.retrieval import vector_store

    return CampusService(
        client=vector_store.get_client(), jobs=JobStore(config.DATA_DIR / "jobs.db")
    )


def create_app(service_factory: Callable[[], CampusService] = default_service) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service_factory()
        yield
        app.state.service.stop()

    app = FastAPI(title="CampusAI API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=FRONTEND_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def service() -> CampusService:
        return app.state.service

    @app.get("/api/health")
    def health():
        model = config.OLLAMA_LLM_MODEL if config.LLM_BACKEND == "ollama" else config.GEMINI_MODEL
        return {"status": "ok", "llm": f"{config.LLM_BACKEND}:{model}"}

    # ใช้ def ธรรมดา (ไม่ใช่ async) เพราะงานข้างในเป็นแบบ blocking (โมเดล, ไฟล์)
    # FastAPI จะรันใน thread pool ให้เอง request อื่นไม่ต้องรอ
    @app.post("/api/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        try:
            result = service().ask(req.question, top_k=req.top_k)
        except Exception as exc:  # noqa: BLE001 - LLM ล่ม/ติดโควตา ส่งข้อความให้หน้าเว็บแสดง
            raise HTTPException(status_code=502, detail=f"เรียก LLM ไม่สำเร็จ: {exc}") from exc
        sources = [
            Source(
                n=result.contexts.index(r) + 1,
                doc=r.doc,
                clause=r.clause,
                chapter=r.chapter,
                label=source_label(r),
                text=r.text,
            )
            for r in result.sources
        ]
        return AskResponse(answer=result.text, refused=result.refused, sources=sources)

    @app.get("/api/documents")
    def list_documents():
        return [asdict(d) for d in service().list_documents()]

    @app.post("/api/documents", status_code=202)
    def upload_document(file: UploadFile = File(...)):
        data = file.file.read()
        try:
            job = service().submit_upload(file.filename or "", data)
        except UploadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job.to_dict()

    @app.delete("/api/documents/{doc}")
    def delete_document(doc: str):
        try:
            service().delete_document(doc)
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"ไม่พบเอกสาร {doc}") from exc
        return {"deleted": doc}

    @app.get("/api/jobs")
    def list_jobs():
        return [j.to_dict() for j in service().jobs.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = service().jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ไม่พบงานนี้")
        return job.to_dict()

    return app


def run_serve(args) -> int:
    import uvicorn

    print(f"[campusai] API: http://{args.host}:{args.port}/docs (Ctrl+C เพื่อหยุด)")
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0
