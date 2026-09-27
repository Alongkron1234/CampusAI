# CampusAI

ระบบถาม-ตอบเกี่ยวกับระเบียบและข้อบังคับของมหาวิทยาลัยภาษาไทย โดยใช้สถาปัตยกรรม RAG
(Retrieval-Augmented Generation) — OCR และ embedding รันบนเครื่องทั้งหมด ไม่ติด rate limit
เหมือนเวอร์ชันก่อนหน้า ส่วน Gemini ใช้เฉพาะตอนสร้างคำตอบสุดท้าย

รายละเอียดสถาปัตยกรรมและแผนพัฒนาทั้งหมดอยู่ที่ [docs/PLAN.md](docs/PLAN.md)

> **สถานะ:** อยู่ระหว่างพัฒนา (Issue #1: วางโครงโปรเจกต์) — คำสั่งส่วนใหญ่ยังไม่ทำงานจริง

## ติดตั้ง

### 1. เครื่องมือระบบ (macOS)

```bash
brew install poppler ollama
brew services start ollama
ollama pull scb10x/typhoon-ocr1.5-3b
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # dependency พื้นฐาน (เบา, Issue #1)
pip install -r requirements-ocr.txt    # + อ่าน PDF / OCR (Issue #2, #3)
pip install -r requirements-ml.txt     # + embedding/search/Gemini (หนัก, ใช้ตั้งแต่ Issue #5)
```

### 3. ตั้งค่า

```bash
cp .env.example .env
# ใส่ GEMINI_API_KEY อย่างน้อย 1 ค่า (ใช้ตอนสร้างคำตอบเท่านั้น)
```

## การใช้งาน (คำสั่งเป้าหมาย — ดูสถานะจริงใน docs/PLAN.md)

```bash
python -m campusai --help

python -m campusai check              # ตรวจว่า PDF หน้าไหนต้อง OCR
python -m campusai ingest             # PDF -> chunks.jsonl
python -m campusai index              # สร้าง vector + BM25 index
python -m campusai ask "ข้อ 15 ว่าด้วยอะไร"
python -m campusai chat               # โหมดถาม-ตอบต่อเนื่อง
```

## พัฒนา

```bash
pytest       # รัน unit test
ruff check . # lint
```

## โครงสร้างโปรเจกต์

ดูรายละเอียดใน [docs/PLAN.md](docs/PLAN.md) หัวข้อ 1.8
