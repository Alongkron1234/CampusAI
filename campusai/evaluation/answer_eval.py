"""วัดคุณภาพคำตอบ (`campusai eval answers`) ด้วยชุดคำถาม eval/questions.jsonl

ตัววัด 4 ตัว:
- Refusal accuracy (นับด้วยโค้ด): คำถาม unanswerable ต้องถูกปฏิเสธ คำถามที่มีคำตอบต้องไม่ถูกปฏิเสธ
- Citation (นับด้วยโค้ด): แหล่งที่คำตอบอ้าง [n] ตรงกับข้อในเฉลยไหม
- Correctness 1-5 (LLM เป็นกรรมการ): เนื้อหาตรงกับ reference_answer แค่ไหน
- Faithfulness 1-5 (LLM เป็นกรรมการ): ทุกข้อความในคำตอบมีอยู่ในเอกสารที่ส่งให้จริงไหม (ไม่แต่งเพิ่ม)

ทุกคำตอบและทุกคะแนน cache ลงดิสก์ทีละข้อ หยุดกลางทาง (Ctrl+C / ติดโควตา) แล้วรันใหม่จะทำต่อ
จากข้อที่ค้าง key ของ cache คือ prompt ทั้งก้อน (รวมเอกสารที่ค้นได้) + ชื่อโมเดล ถ้าเปลี่ยนวิธีค้นหา
โมเดล หรือ prompt ผลเก่าจะไม่ถูกใช้ซ้ำเอง
"""

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from campusai import config
from campusai.generation import answer as gen
from campusai.retrieval.vector_store import SearchResult

JUDGE_SYSTEM_PROMPT = """คุณคือกรรมการตรวจคำตอบของระบบถาม-ตอบระเบียบมหาวิทยาลัย ให้คะแนน 2 ด้าน ด้านละ 1-5

correctness: เทียบ "คำตอบของระบบ" กับ "คำตอบอ้างอิง"
  5 = ครบทุกประเด็นสำคัญและถูกต้อง
  4 = ถูกต้อง ขาดรายละเอียดเล็กน้อย
  3 = ถูกบางส่วน ขาดประเด็นสำคัญ
  2 = ส่วนใหญ่ผิดหรือไม่ตอบสิ่งที่ถาม
  1 = ผิดทั้งหมด ขัดแย้งกับคำตอบอ้างอิง หรือปฏิเสธทั้งที่มีคำตอบ
faithfulness: ทุกข้อความในคำตอบของระบบมีอยู่ใน "เอกสารที่ระบบได้รับ" จริงหรือไม่ (ไม่สนว่าถูกตามเฉลยไหม)
  5 = ทุกข้อความมีในเอกสาร
  3 = มีบางข้อความที่เอกสารไม่ได้ระบุ
  1 = ส่วนใหญ่แต่งขึ้นเอง ไม่มีในเอกสาร

ตอบเป็น JSON บรรทัดเดียวเท่านั้น ห้ามมีข้อความอื่น:
{"correctness": <1-5>, "faithfulness": <1-5>, "reason": "<เหตุผลสั้น ๆ ภาษาไทย>"}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Judgement:
    correctness: int
    faithfulness: int
    reason: str


@dataclass
class EvalRow:
    id: str
    type: str
    question: str
    answer: str
    refused: bool
    cited: list[tuple[str, str | None]]
    expected: list[tuple[str, str | None]]
    correctness: int | None = None
    faithfulness: int | None = None
    judge_reason: str = ""
    errors: list[str] = field(default_factory=list)


# ----- cache -----


def _key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def llm_tag() -> str:
    """ชื่อ backend + โมเดล ใช้เป็นส่วนหนึ่งของ cache key และบันทึกลงผล"""
    model = config.OLLAMA_LLM_MODEL if config.LLM_BACKEND == "ollama" else config.GEMINI_MODEL
    return f"{config.LLM_BACKEND}:{model}"


def cached_llm(kind: str, system: str, prompt: str, cache_dir: Path | None = None) -> str:
    """เรียก LLM ผ่าน cache: เคยถาม prompt นี้กับโมเดลนี้แล้วจะคืนผลเดิมโดยไม่เรียกใหม่"""
    cache_dir = cache_dir or config.ANSWER_EVAL_CACHE_DIR
    path = cache_dir / kind / f"{_key(llm_tag(), system, prompt)}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    text = gen.call_llm(prompt, system=system)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


# ----- กรรมการ -----


def build_judge_prompt(question: str, reference: str, contexts: list[SearchResult], answer: str):
    return (
        f"คำถาม: {question}\n\n"
        f"คำตอบอ้างอิง: {reference}\n\n"
        f"เอกสารที่ระบบได้รับ:\n\n{gen.build_context(contexts)}\n\n"
        f"คำตอบของระบบ: {answer}"
    )


def parse_judgement(text: str) -> Judgement | None:
    """ดึง JSON คะแนนจากคำตอบของกรรมการ รูปแบบผิดหรือคะแนนนอกช่วง 1-5 คืน None"""
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        scores = int(data["correctness"]), int(data["faithfulness"])
    except (ValueError, KeyError, TypeError):
        return None
    if not all(1 <= s <= 5 for s in scores):
        return None
    return Judgement(*scores, reason=str(data.get("reason", "")))


# ----- รันทีละข้อ -----


def evaluate_question(q: dict, retriever, top_k: int = gen.TOP_K_CONTEXT) -> EvalRow:
    contexts = retriever.hybrid(q["question"], top_k=top_k)
    if contexts:
        text = cached_llm(
            "answers", gen.SYSTEM_PROMPT, gen.build_prompt(q["question"], contexts)
        ).strip()
        cited = [contexts[n - 1] for n in sorted(gen.extract_citations(text, len(contexts)))]
    else:
        text, cited = gen.REFUSAL, []
    result = gen.Answer(text=text, sources=cited, contexts=contexts)

    row = EvalRow(
        id=q["id"],
        type=q["type"],
        question=q["question"],
        answer=text,
        refused=result.refused,
        cited=[(r.doc, r.clause) for r in cited],
        expected=[(e["doc"], e["clause"]) for e in q["expected"]],
    )

    # กรรมการตรวจเฉพาะข้อที่มีเฉลยและระบบตอบ ข้อที่ปฏิเสธทั้งที่มีคำตอบ = ผิด (correctness 1) ไม่ต้องเรียก
    if q["type"] == "unanswerable":
        return row
    if row.refused:
        row.correctness = 1
        return row

    judge_prompt = build_judge_prompt(q["question"], q["reference_answer"], contexts, text)
    judgement = parse_judgement(cached_llm("judge", JUDGE_SYSTEM_PROMPT, judge_prompt))
    if judgement is None:
        row.errors.append("กรรมการตอบรูปแบบผิด อ่านคะแนนไม่ได้")
    else:
        row.correctness = judgement.correctness
        row.faithfulness = judgement.faithfulness
        row.judge_reason = judgement.reason
    return row


# ----- สรุปผล -----


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[EvalRow]) -> dict:
    answerable = [r for r in rows if r.type != "unanswerable"]
    unanswerable = [r for r in rows if r.type == "unanswerable"]
    answered = [r for r in answerable if not r.refused]
    cited_total = sum(len(r.cited) for r in answered)
    cited_correct = sum(1 for r in answered for c in r.cited if c in r.expected)

    return {
        "n": len(rows),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        # การตัดสินใจ "ตอบ/ไม่ตอบ" ถูกกี่ข้อจากทั้งหมด
        "refusal_accuracy": (
            sum(1 for r in unanswerable if r.refused) + len(answered)
        ) / len(rows) if rows else None,
        "correct_refusals": sum(1 for r in unanswerable if r.refused),
        "hallucinated_answers": sum(1 for r in unanswerable if not r.refused),
        "false_refusals": len(answerable) - len(answered),
        # คำตอบที่อ้างอย่างน้อย 1 ข้อที่อยู่ในเฉลย / คำตอบทั้งหมดที่ไม่ได้ปฏิเสธ
        "citation_hit": (
            sum(1 for r in answered if set(r.cited) & set(r.expected)) / len(answered)
            if answered else None
        ),
        # แหล่งที่อ้างทั้งหมด อยู่ในเฉลยกี่ % (เฉลยระบุข้อหลักเท่านั้น ตัวเลขนี้จึงต่ำกว่าความจริงได้)
        "citation_precision": cited_correct / cited_total if cited_total else None,
        "correctness": _mean([r.correctness for r in answerable if r.correctness is not None]),
        "faithfulness": _mean([r.faithfulness for r in answered if r.faithfulness is not None]),
        "judge_errors": sum(1 for r in rows if r.errors),
    }


def summarize_by_type(rows: list[EvalRow]) -> dict[str, dict]:
    types = sorted({r.type for r in rows})
    return {t: summarize([r for r in rows if r.type == t]) for t in types}


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:.0%}" if pct else f"{value:.2f}"


def format_report(rows: list[EvalRow], meta: dict) -> str:
    s = summarize(rows)
    lines = [
        f"# Answer eval {meta['stamp']}",
        "",
        f"- LLM (ตอบ + กรรมการ): `{meta['llm']}`",
        f"- ชุดคำถาม: `{meta['questions']}` ({s['n']} ข้อ: มีคำตอบ {s['n_answerable']}, "
        f"ไม่มีคำตอบ {s['n_unanswerable']})",
        f"- ชุดเอกสาร (sha256 ของ chunks.jsonl): `{meta['chunks_sha']}`",
        f"- เวลา: {meta['elapsed_min']:.1f} นาที",
        "",
        "| ตัววัด | ค่า | รายละเอียด |",
        "|---|---|---|",
        f"| Refusal accuracy | {_fmt(s['refusal_accuracy'], True)} | ปฏิเสธถูก "
        f"{s['correct_refusals']}/{s['n_unanswerable']}, แต่งคำตอบ {s['hallucinated_answers']}, "
        f"ปฏิเสธทั้งที่มีคำตอบ {s['false_refusals']}/{s['n_answerable']} |",
        f"| Citation hit | {_fmt(s['citation_hit'], True)} | คำตอบที่อ้างข้อในเฉลยอย่างน้อย 1 ข้อ |",
        f"| Citation precision | {_fmt(s['citation_precision'], True)} | "
        "แหล่งที่อ้างอยู่ในเฉลย (เฉลยแคบ ค่านี้ต่ำกว่าความจริงได้) |",
        f"| Correctness (1-5) | {_fmt(s['correctness'])} | เทียบกับ reference_answer |",
        f"| Faithfulness (1-5) | {_fmt(s['faithfulness'])} | ไม่แต่งเพิ่มจากเอกสาร |",
        f"| กรรมการตอบผิดรูปแบบ | {s['judge_errors']} | ข้อที่อ่านคะแนนไม่ได้ ไม่นับในค่าเฉลี่ย |",
        "",
        "## แยกตามประเภท",
        "",
        "| ประเภท | n | Correctness | Faithfulness | Citation hit | Refusal accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for qtype, t in summarize_by_type(rows).items():
        lines.append(
            f"| {qtype} | {t['n']} | {_fmt(t['correctness'])} | {_fmt(t['faithfulness'])} | "
            f"{_fmt(t['citation_hit'], True)} | {_fmt(t['refusal_accuracy'], True)} |"
        )

    failures = [
        r
        for r in rows
        if (r.type == "unanswerable" and not r.refused)
        or (r.type != "unanswerable" and (r.refused or (r.correctness or 5) <= 2))
        or r.errors
    ]
    lines += ["", f"## ข้อที่ตอบผิด ({len(failures)} ข้อ)", ""]
    for r in failures:
        lines += [
            f"### {r.id} ({r.type}) {r.question}",
            f"- คำตอบ: {r.answer}",
            f"- อ้าง: {r.cited or '-'} | เฉลย: {r.expected or '-'}",
            f"- correctness={r.correctness} faithfulness={r.faithfulness} {r.judge_reason}",
        ]
        lines += [f"- ⚠ {e}" for e in r.errors]
        lines.append("")
    return "\n".join(lines)


# ----- คำสั่ง -----


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def run_eval_answers(args: argparse.Namespace) -> int:
    from campusai.evaluation.validate_questions import load_eval_questions
    from campusai.generation.ask_cli import _load_retriever

    try:
        questions = load_eval_questions(args.questions, allow_test_set=args.allow_test_set)
    except ValueError as exc:
        print(f"[campusai] {exc}")
        return 1
    if args.limit:
        questions = questions[: args.limit]

    retriever = _load_retriever()
    if retriever is None:
        return 1

    print(f"[campusai] วัดคำตอบ {len(questions)} ข้อ ด้วย {llm_tag()} (มี cache จะข้ามข้อที่ทำแล้ว)")
    start = time.time()
    rows: list[EvalRow] = []
    for i, q in enumerate(questions, start=1):
        try:
            row = evaluate_question(q, retriever)
        except Exception as exc:  # noqa: BLE001 - ข้อที่พัง (เช่นติดโควตา) บันทึกไว้ รันใหม่จะทำต่อ
            print(f"[campusai] {q['id']} ล้มเหลว: {exc}")
            print("[campusai] หยุดไว้ก่อน รันคำสั่งเดิมอีกครั้งจะทำต่อจากข้อนี้ (ข้อก่อนหน้าอยู่ใน cache)")
            return 1
        rows.append(row)
        status = "ปฏิเสธ" if row.refused else f"correctness={row.correctness}"
        print(f"  [{i}/{len(questions)}] {q['id']} {status} ({time.time() - start:.0f}s)")

    meta = {
        "stamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "llm": llm_tag(),
        "questions": str(args.questions or config.EVAL_QUESTIONS_PATH.relative_to(config.ROOT)),
        "chunks_sha": _file_sha(config.CHUNKS_PATH),
        "elapsed_min": (time.time() - start) / 60,
    }
    report = format_report(rows, meta)
    print("\n" + report.split("\n## ข้อที่ตอบผิด")[0])

    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = config.EVAL_RESULTS_DIR / f"answers-{meta['stamp']}.md"
    md_path.write_text(report + "\n", encoding="utf-8")
    json_path = md_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {"meta": meta, "summary": summarize(rows), "by_type": summarize_by_type(rows),
             "rows": [asdict(r) for r in rows]},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nบันทึกผลไว้ที่ {md_path}")
    return 0
