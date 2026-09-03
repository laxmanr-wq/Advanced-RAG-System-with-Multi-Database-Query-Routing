"""
100-question accuracy benchmark for the RAG Agent using the IFRS module PDF
(9781513563602-mod01.pdf — "Model Financial Statements" for central banks).

Method:
  1. Parse the PDF with the app's production DocumentParser.
  2. Ingest the chunks into the in-memory Qdrant 'financial' collection
     (topical fit for IFRS/financial-statement content).
  3. Generate 100 factual question/reference-answer pairs from the document
     text via Groq (cached in evaluation/artifacts/ for reuse across runs).
  4. For every question: real router -> retriever -> grounded generator ->
     faithfulness evaluator -> LLM-as-judge correctness (1.0 / 0.5 / 0.0).
  5. Write JSON results + a markdown report into evaluation/artifacts/.

Usage:
  uv run python evaluate_pdf_module.py --label run1
  uv run python evaluate_pdf_module.py --label run2 --regen-questions
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in evaluation/)
sys.path.insert(0, str(ROOT))  # keep `rag_agent` importable when run directly
load_dotenv(ROOT / ".env")

import os

from openai import OpenAI

from rag_agent.databases import add_documents, build_databases
from rag_agent.evaluator import FaithfulnessEvaluator
from rag_agent.parser import DocumentParser
from rag_agent.pipeline import MODEL, RAG_SYSTEM
from rag_agent.postprocess import strip_citations
from rag_agent.retriever import retrieve
from rag_agent.router import route_query

# -- Config --------------------------------------------------------------------

PDF_NAME = "9781513563602-mod01.pdf"
ARTIFACTS = ROOT / "evaluation" / "artifacts"
QUESTIONS_FILE = ARTIFACTS / "mod01_questions.json"


def resolve_pdf_path(cli_arg: str | None = None) -> Path:
    """Locate the benchmark PDF: --pdf arg > EVAL_PDF env > ./data > ~/Downloads."""
    candidates: list[Path] = []
    if cli_arg:
        candidates.append(Path(cli_arg).expanduser())
    env = os.getenv("EVAL_PDF", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [
        ROOT / "data" / PDF_NAME,
        Path.home() / "Downloads" / PDF_NAME,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Benchmark PDF not found. Tried: {[str(c) for c in candidates]}. "
        f"Copy it to ./data/{PDF_NAME} or pass --pdf <path>."
    )


# Resolved lazily in main() so importing this module (evaluate_ragas.py)
# never crashes when the PDF lives outside the default locations - callers
# pass --pdf / EVAL_PDF instead.
PDF_PATH: Path | None = None

COLLECTION = "financial"     # IFRS / financial statements content
TARGET_QUESTIONS = 100
TOP_K = 8

# Q32/Q33 reference answers were verified NOT to exist in the source document
# (QA-generator hallucinations); they are excluded from accuracy aggregates.
DEFECTIVE_QIDS = {32, 33}

QA_WINDOW_CHARS = 3500       # text window fed per QA-generation call
QA_WINDOW_STRIDE = 2600      # overlap between windows for coverage
QA_PER_WINDOW = 12           # questions requested per window
CALL_SLEEP = 1.5             # pacing between Groq calls (free-tier RPM/TPD limits)


@dataclass
class QAResult:
    qid: int
    question: str
    reference: str
    routed_db: str
    routing_reason: str
    routed_correctly: bool          # did the router pick the collection holding the answer?
    searched_db: str                # collection actually searched (routed, or oracle override)
    docs_found: int
    top_score: float
    answer: str
    groundedness: float
    ground_label: str
    judge_score: float
    judge_verdict: str
    judge_reason: str


def _qa_result_from_row(row: dict) -> QAResult:
    """Rebuild a QAResult from saved JSON, tolerating older files that predate
    the routed_correctly / searched_db fields."""
    row = {k: v for k, v in row.items() if k in QAResult.__dataclass_fields__}
    row.setdefault("routed_correctly", row.get("routed_db") == COLLECTION)
    row.setdefault("searched_db", COLLECTION)
    return QAResult(**row)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def llm_call(fn, attempts: int = 14):
    """Call Groq with exponential backoff on rate-limit / transient errors."""
    delay = 2.0
    last_err: Exception | None = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            transient = any(
                k in msg for k in ("429", "rate", "limit", "500", "502", "503", "timeout", "connection", "overloaded")
            )
            if not transient:
                raise
            # Honor Groq's "try again in Xs" hint when present (TPD resets)
            wait = delay
            import re as _re
            m = _re.search(r"try again in (\d+)m([\d.]+)s", msg)
            if m:
                wait = int(m.group(1)) * 60 + float(m.group(2)) + 5
            else:
                m = _re.search(r"try again in ([\d.]+)s", msg)
                if m:
                    wait = float(m.group(1)) + 2
            log(f"  transient error ({type(e).__name__}), retrying in {wait:.0f}s ...")
            time.sleep(wait)
            delay = min(delay * 2, 32.0)
    raise RuntimeError(f"LLM call failed after {attempts} attempts: {last_err}")


def chat_json(client: OpenAI, prompt: str, temperature: float) -> dict:
    resp = llm_call(
        lambda: client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
    )
    return json.loads(resp.choices[0].message.content)


# -- Ingestion ------------------------------------------------------------------

def parse_and_ingest(client, embeddings, groq_client: OpenAI) -> tuple[list[str], str]:
    log(f"Parsing PDF: {PDF_PATH.name} ({PDF_PATH.stat().st_size/1e3:.0f} KB) ...")
    t0 = time.time()
    chunks, engine = DocumentParser.parse_file(str(PDF_PATH), PDF_PATH.name, groq_client=groq_client)
    log(f"  -> {len(chunks)} chunks via {engine} in {time.time()-t0:.1f}s")

    t_emb = time.time()
    ingested = add_documents(client, embeddings, COLLECTION, chunks)
    log(f"  Ingested {ingested} chunks into '{COLLECTION}' in {time.time()-t_emb:.1f}s")
    if ingested != len(chunks):
        log(f"  NOTE: ingestion consolidated {len(chunks)} parser chunks -> {ingested} indexed chunks")
    return chunks, engine


# -- Question generation ----------------------------------------------------------

def build_windows(chunks: list[str]) -> list[str]:
    full_text = "\n\n".join(chunks)
    windows = []
    start = 0
    while start < len(full_text):
        end = min(start + QA_WINDOW_CHARS, len(full_text))
        window = full_text[start:end]
        if len(window.strip()) > 200:
            windows.append(window)
        if end == len(full_text):
            break
        start += QA_WINDOW_STRIDE
    return windows


def generate_questions(groq_client: OpenAI, chunks: list[str]) -> list[dict]:
    if QUESTIONS_FILE.exists():
        pairs = json.loads(QUESTIONS_FILE.read_text())
        log(f"Reusing cached question set: {len(pairs)} pairs from {QUESTIONS_FILE.name}")
        return pairs

    log(f"Generating {TARGET_QUESTIONS} reference QA pairs from document text ...")
    windows = build_windows(chunks)
    pairs: list[dict] = []
    seen: set[str] = set()

    for wi, window in enumerate(windows, 1):
        if len(pairs) >= TARGET_QUESTIONS:
            break
        prompt = (
            "You are building a factual QA benchmark from an excerpt of a central-bank "
            "IFRS guide (Model Financial Statements module).\n"
            f"Generate exactly {QA_PER_WINDOW} specific, diverse, factual question/answer "
            "pairs answerable ONLY from the text below.\n"
            "Vary the question types: definitions, numeric values/figures, accounting "
            "treatments, recognition/measurement rules, disclosure requirements, "
            "line items, notes, and policy details.\n"
            "Each reference answer must be a short, precise, verifiable statement taken "
            "directly from the text (keep numbers exact).\n"
            "Do not ask about page numbers, layout, or 'the table above'.\n"
            'Respond with ONLY JSON: {"pairs":[{"question":"...","reference_answer":"..."}]}\n\n'
            f"TEXT EXCERPT:\n{window}"
        )
        try:
            data = chat_json(groq_client, prompt, temperature=0.4)
        except Exception as e:  # noqa: BLE001
            log(f"  window {wi} failed: {e}")
            continue
        added = 0
        for p in data.get("pairs", []):
            q, ref = str(p.get("question", "")).strip(), str(p.get("reference_answer", "")).strip()
            key = " ".join(q.lower().split())
            if q and ref and key not in seen:
                seen.add(key)
                pairs.append({"question": q, "reference_answer": ref})
                added += 1
        log(f"  window {wi}/{len(windows)}: +{added} (total {len(pairs)})")
        time.sleep(CALL_SLEEP)

    pairs = pairs[:TARGET_QUESTIONS]
    ARTIFACTS.mkdir(exist_ok=True)
    QUESTIONS_FILE.write_text(json.dumps(pairs, indent=2))
    log(f"Saved {len(pairs)} QA pairs -> {QUESTIONS_FILE}")
    return pairs


# -- Answer generation & judging ----------------------------------------------------

def generate_answer(groq_client: OpenAI, query: str, docs, gen_model: str | None = None) -> str:
    context_str = "\n\n".join(f"[{i+1}] {d.text}" for i, d in enumerate(docs))
    resp = llm_call(
        lambda: groq_client.chat.completions.create(
            model=gen_model or MODEL,
            messages=[
                {"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
            ],
            temperature=0.2,
        )
    )
    return strip_citations(resp.choices[0].message.content or "")


def judge_answer(
    groq_client: OpenAI,
    question: str,
    reference: str,
    answer: str,
    judge_model: str | None = None,
) -> tuple[float, str, str]:
    # Truncate to keep judging cheap against strict token quotas; answers are short by design
    answer_t = answer[:600]
    reference_t = reference[:300]
    prompt = (
        "You are grading a RAG system's answer against a reference answer taken from "
        "the source document.\n"
        "Score 1.0 if the answer is factually correct and covers the reference "
        "(matching numbers/values exactly when the reference contains them).\n"
        "Score 0.5 if partially correct or missing a key detail.\n"
        "Score 0.0 if incorrect, contradicts the reference, is off-topic, or says it "
        "cannot answer.\n"
        "Accounting notation: parentheses mean negative, so '(123)' equals '-123'; "
        "treat these as matching.\n"
        'Respond ONLY as JSON: {"score":1.0,"verdict":"correct|partial|incorrect","reason":"short"}\n\n'
        f"QUESTION: {question}\n\nREFERENCE ANSWER: {reference_t}\n\nSYSTEM ANSWER: {answer_t}"
    )
    resp = llm_call(
        lambda: groq_client.chat.completions.create(
            model=judge_model or MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
    )
    data = json.loads(resp.choices[0].message.content)
    return float(data.get("score", 0.0)), str(data.get("verdict", "?")), str(data.get("reason", ""))


# -- Main evaluation loop --------------------------------------------------------------

def run_evaluation(
    groq_client: OpenAI,
    client,
    embeddings,
    qa_pairs: list[dict],
    label: str,
    judge_model: str | None = None,
    gen_model: str | None = None,
    routing_source: dict[int, QAResult] | None = None,
    oracle_collection: bool = False,
) -> list[QAResult]:
    results_file = ARTIFACTS / f"mod01_results_{label}.json"
    done: dict[int, QAResult] = {}
    if results_file.exists():
        for row in json.loads(results_file.read_text()):
            done[row["qid"]] = _qa_result_from_row(row)
        log(f"Resuming: {len(done)} results already in {results_file.name}")

    results: list[QAResult] = []
    for idx, pair in enumerate(qa_pairs, 1):
        if idx in done:
            results.append(done[idx])
            continue

        q = pair["question"]
        ref = pair["reference_answer"]

        if routing_source and idx in routing_source:
            # Router is temperature=0 (deterministic): reuse saved decision to save tokens
            routing_db = routing_source[idx].routed_db
            routing_reason = routing_source[idx].routing_reason + " (reused)"
        else:
            routing = route_query(groq_client, q)
            routing_db, routing_reason = routing.database, routing.reasoning
            time.sleep(CALL_SLEEP)

        # Search the collection the ROUTER chose - this is what the live app does.
        # --oracle-collection forces the known-correct collection instead, which
        # isolates retrieval+generation quality from routing errors.
        routed_correctly = routing_db == COLLECTION
        searched_db = COLLECTION if oracle_collection else routing_db
        docs = retrieve(q, searched_db, client, embeddings, top_k=TOP_K)

        if docs:
            answer = generate_answer(groq_client, q, docs, gen_model=gen_model)
        else:
            answer = "No relevant documents were retrieved from the knowledge base."
        time.sleep(CALL_SLEEP)

        ev = FaithfulnessEvaluator.evaluate(answer, [d.text for d in docs])
        try:
            jscore, jverdict, jreason = judge_answer(
                groq_client, q, ref, answer, judge_model=judge_model
            )
        except Exception as e:  # noqa: BLE001
            jscore, jverdict, jreason = 0.0, "error", str(e)
        time.sleep(CALL_SLEEP)

        res = QAResult(
            qid=idx, question=q, reference=ref,
            routed_db=routing_db, routing_reason=routing_reason,
            routed_correctly=routed_correctly, searched_db=searched_db,
            docs_found=len(docs), top_score=round(docs[0].score, 4) if docs else 0.0,
            answer=answer, groundedness=ev.groundedness_score, ground_label=ev.status_label,
            judge_score=jscore, judge_verdict=jverdict, judge_reason=jreason,
        )
        results.append(res)
        mark = {1.0: "OK", 0.5: "~", 0.0: "X"}.get(jscore, "?")
        log(f"  [{mark}] Q{idx:>3}/{len(qa_pairs)} (docs={len(docs)}, top={res.top_score:.3f}) {q[:65]}")

        # checkpoint after every question
        results_file.write_text(json.dumps([asdict(r) for r in results], indent=2))

    return results


# -- Report ---------------------------------------------------------------------------

def write_report(
    results: list[QAResult],
    engine: str,
    total_chunks: int,
    ingested: int,
    label: str,
    judge_model: str | None = None,
    gen_model: str | None = None,
) -> None:
    valid = [r for r in results if r.qid not in DEFECTIVE_QIDS]
    scores = [r.judge_score for r in valid]
    acc = statistics.mean(scores) if scores else 0.0
    correct = sum(1 for r in valid if r.judge_score == 1.0)
    partial = sum(1 for r in valid if r.judge_score == 0.5)
    incorrect = sum(1 for r in valid if r.judge_score == 0.0)
    retrieval_miss = [r for r in valid if r.docs_found == 0]
    grounded = statistics.mean(r.groundedness for r in valid) if valid else 0.0
    top_scores = [r.top_score for r in valid if r.docs_found > 0]

    lines: list[str] = []
    w = lines.append
    w("# RAG Agent — 100-Question Accuracy Benchmark (IFRS Model Financial Statements)")
    w("")
    w(f"_Run label: `{label}` · Generated: {time.strftime('%Y-%m-%d %H:%M')}_  ")
    w(f"_Model: `{gen_model or MODEL}` · Judge: `{judge_model or MODEL}` · Embeddings: `BAAI/bge-small-en-v1.5` (+ BM25 hybrid) · Retriever top-k: {TOP_K}_")
    w("")
    w("## Executive Summary")
    w("")
    w(f"- **Overall accuracy (LLM-judge mean): {acc*100:.1f}%** (excluding {len(DEFECTIVE_QIDS)} defective reference questions: {sorted(DEFECTIVE_QIDS)})")
    w(f"- Correct: **{correct}** · Partial: **{partial}** · Incorrect: **{incorrect}** (of {len(valid)} valid questions)")
    w(f"- Retrieval hit rate: **{len(valid)-len(retrieval_miss)}/{len(valid)}** · Mean groundedness: **{grounded*100:.0f}%**")

    # Routing accuracy - the feature this project is named after. Reported
    # separately so it is never hidden behind an oracle-collection retrieval.
    routed_ok = sum(1 for r in valid if getattr(r, "routed_correctly", False))
    searched = {getattr(r, "searched_db", COLLECTION) for r in valid}
    mode = "oracle collection (routing bypassed)" if searched == {COLLECTION} and routed_ok < len(valid) else "routed end-to-end"
    w(f"- **Routing accuracy: {routed_ok}/{len(valid)} ({routed_ok/len(valid)*100:.1f}%)** "
      f"— queries sent to `{COLLECTION}`, the collection holding the answers")
    w(f"- Retrieval mode: **{mode}**")
    if top_scores:
        w(f"- Top-doc similarity: min `{min(top_scores):.3f}` · median `{statistics.median(top_scores):.3f}` · max `{max(top_scores):.3f}`")
    w(f"- Parser engine: **{engine}** · parser chunks: **{total_chunks}** · indexed chunks: **{ingested}** (collection `{COLLECTION}`)")
    w("")

    # Failure analysis
    fails = [r for r in valid if r.judge_score < 1.0]
    w("## Failure Analysis")
    w("")
    if not fails:
        w("No failures — all questions answered correctly.")
    else:
        no_docs = [r for r in fails if r.docs_found == 0]
        low_score = [r for r in fails if 0 < r.docs_found and r.top_score < 0.5]
        gen_err = [r for r in fails if r.docs_found > 0 and r.top_score >= 0.5]
        w(f"- Retrieval misses (0 docs): **{len(no_docs)}**")
        w(f"- Weak retrieval (top score < 0.5): **{len(low_score)}**")
        w(f"- Retrieval OK but answer wrong/partial (generation/judge issues): **{len(gen_err)}**")
        w("")
        for r in fails:
            w(f"### Q{r.qid}. {r.question}")
            w(f"- Reference: {r.reference}")
            w(f"- Answer: {r.answer.strip()}")
            w(f"- Verdict: `{r.judge_verdict}` ({r.judge_score:.1f}) — {r.judge_reason}")
            w(f"- Retrieval: {r.docs_found} docs, top `{r.top_score:.3f}` · routed `{r.routed_db}` · groundedness `{r.groundedness:.2f}`")
            w("")

    # Full table
    w("## Full Results")
    w("")
    w("| # | Question | Verdict | Score | Docs | Top | Grounded |")
    w("|---|---|---|---|---|---|---|")
    for r in results:
        q_short = r.question[:60] + ("..." if len(r.question) > 60 else "")
        w(f"| {r.qid} | {q_short} | {r.judge_verdict} | {r.judge_score:.1f} | {r.docs_found} | {r.top_score:.3f} | {r.groundedness:.2f} |")
    w("")

    out = ARTIFACTS / "EVALUATION_REPORT_MOD01.md"
    out.write_text("\n".join(lines))
    log(f"Report written: {out}")
    print("\n" + "=" * 60)
    print(f"ACCURACY: {acc*100:.1f}%  ({correct} correct / {partial} partial / {incorrect} incorrect)")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run1", help="run label for result files")
    ap.add_argument("--regen-questions", action="store_true", help="ignore cached question set")
    ap.add_argument("--judge-model", default=None, help="cheaper model for LLM-judging (quota saving)")
    ap.add_argument("--gen-model", default=None, help="model for answer generation (default: pipeline MODEL)")
    ap.add_argument("--reuse-routing-from", default=None, help="results JSON whose routing decisions are reused (quota saving)")
    ap.add_argument("--pdf", default=None, help="path to the benchmark PDF (default: auto-resolved)")
    ap.add_argument(
        "--oracle-collection",
        action="store_true",
        help="retrieve from the known-correct collection instead of the routed one "
             "(isolates retrieval+generation from routing errors; default is the "
             "honest end-to-end path the app actually runs)",
    )
    args = ap.parse_args()

    global PDF_PATH
    PDF_PATH = resolve_pdf_path(args.pdf)

    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        log("ERROR: GROQ_API_KEY not set (put it in .env). Aborting.")
        sys.exit(1)
    if args.regen_questions and QUESTIONS_FILE.exists():
        QUESTIONS_FILE.unlink()

    routing_source: dict[int, QAResult] | None = None
    if args.reuse_routing_from:
        src = ARTIFACTS / f"mod01_results_{args.reuse_routing_from}.json"
        routing_source = {
            row["qid"]: _qa_result_from_row(row) for row in json.loads(src.read_text())
        }
        log(f"Reusing routing decisions from {src.name} ({len(routing_source)} entries)")

    groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)

    log("Building in-memory Qdrant + FastEmbed embeddings ...")
    client, embeddings = build_databases()

    chunks, engine = parse_and_ingest(client, embeddings, groq_client)
    if not chunks:
        log("ERROR: no chunks extracted from PDF.")
        sys.exit(1)
    ingested = client.get_collection(COLLECTION).points_count or 0

    qa_pairs = generate_questions(groq_client, chunks)
    if len(qa_pairs) < TARGET_QUESTIONS:
        log(f"WARNING: only {len(qa_pairs)} unique questions generated (target {TARGET_QUESTIONS}).")

    results = run_evaluation(
        groq_client, client, embeddings, qa_pairs, args.label,
        judge_model=args.judge_model, gen_model=args.gen_model,
        routing_source=routing_source,
        oracle_collection=args.oracle_collection,
    )
    write_report(results, engine, len(chunks), ingested, args.label, judge_model=args.judge_model, gen_model=args.gen_model)


if __name__ == "__main__":
    main()
