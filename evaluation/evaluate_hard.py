"""
HARD 100-question benchmark for the RAG Agent on the IFRS module PDF.

Companion to evaluate_pdf_module.py, but adversarial:
  * 50 standalone hard questions across 5 categories:
      numeric_trap   - exact figures, parenthesized negatives, look-alike lines
      multi_hop      - year-on-year deltas / reconciliations across statements
      disambiguation - same line name in two sections (derivatives, gold, securities)
      guidance_policy- IAS 1 / IAS 7 rules and rationale questions
      unanswerable   - facts NOT in the document (a grounded refusal = correct)
  * 10 follow-up sessions x 5 turns = 50 multi-turn questions that exercise
    ConversationMemory (elliptical references: "that total", "and in 2018?").

Each question runs through the REAL production run_pipeline (guardrails ->
router -> hybrid retrieval -> grounded generator -> faithfulness evaluator),
then an LLM judge scores 1.0 / 0.5 / 0.0. Session judges see the prior turns
so elliptical follow-ups are graded in context.

Results checkpoint every question (resumable). Report -> evaluation/artifacts/EVALUATION_HARD.md

Usage:
  RAG_MODEL=openai/gpt-oss-20b uv run python evaluation/evaluate_hard.py --label hard_8b
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling harness imports
load_dotenv(ROOT / ".env")

import os

from openai import OpenAI

from evaluate_pdf_module import COLLECTION, llm_call, log, resolve_pdf_path
from openai import RateLimitError

from rag_agent.databases import add_documents, build_databases
from rag_agent.memory import ConversationMemory
from rag_agent.parser import DocumentParser
from rag_agent.pipeline import MODEL, run_pipeline
from rag_agent.quota import chat_with_quota_retry, quota_wait_seconds

ARTIFACTS = ROOT / "evaluation" / "artifacts"
QUESTIONS_FILE = ARTIFACTS / "mod01_hard_questions.json"
CALL_SLEEP = 1.2


@dataclass
class HardResult:
    qid: int                 # global 1..100
    session: int             # 0 = standalone, else session id
    turn: int                # 1-based within session (0 for standalone)
    category: str
    question: str
    reference: str
    routed_db: str
    used_fallback: bool
    docs_found: int
    top_score: float
    answer: str
    groundedness: float
    judge_score: float
    judge_verdict: str
    judge_reason: str


def judge_hard(
    groq_client: OpenAI,
    question: str,
    reference: str,
    answer: str,
    history: str,
    judge_model: str | None,
) -> tuple[float, str, str]:
    prompt = (
        "You are grading a RAG system's answer about a central-bank IFRS model "
        "financial-statements document against a reference answer.\n"
        "Score 1.0 if factually correct and covering the reference (numbers must "
        "match exactly when the reference contains them).\n"
        "Score 0.5 if partially correct / missing a key detail.\n"
        "Score 0.0 if incorrect, contradicts the reference, or fabricates.\n"
        "Special rule: if the REFERENCE starts with 'UNANSWERABLE', the correct "
        "behaviour is to refuse or say the document does not contain that "
        "information - score a grounded refusal 1.0 and any fabricated answer 0.0.\n"
        "Accounting notation: parentheses mean negative, '(123)' equals '-123'.\n"
        "Computed deltas are correct if the arithmetic on the stated figures is right.\n"
        "If CONVERSATION HISTORY is given, the question is a follow-up: resolve "
        "references like 'it', 'that total' from the history before grading.\n"
        'Respond ONLY as JSON: {"score":1.0,"verdict":"correct|partial|incorrect|refused_correct","reason":"short"}\n\n'
        f"CONVERSATION HISTORY:\n{history or '(none)'}\n\n"
        f"QUESTION: {question}\n\nREFERENCE ANSWER: {reference[:400]}\n\nSYSTEM ANSWER: {answer[:700]}"
    )
    resp = chat_with_quota_retry(
        groq_client,
        model=judge_model or MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    data = json.loads(resp.choices[0].message.content)
    return float(data.get("score", 0.0)), str(data.get("verdict", "?")), str(data.get("reason", ""))


def run_hard_benchmark(
    groq_client: OpenAI,
    client,
    embeddings,
    spec: dict,
    label: str,
    judge_model: str | None,
) -> list[HardResult]:
    results_file = ARTIFACTS / f"mod01_hard_results_{label}.json"
    done: dict[int, HardResult] = {}
    if results_file.exists():
        for row in json.loads(results_file.read_text()):
            done[row["qid"]] = HardResult(**row)
        log(f"Resuming: {len(done)} results already in {results_file.name}")

    results: list[HardResult] = []
    qid = 0

    def process(q: str, ref: str, category: str, session: int, turn: int,
                memory: ConversationMemory | None, history: str) -> None:
        nonlocal qid
        qid += 1
        cur = qid
        if cur in done:
            results.append(done[cur])
            return
        t0 = time.time()
        # run_pipeline's own quota-wait caps at 240s; long TPD windows are
        # waited out here instead of crashing the whole run (60 x 950s ~ 16h).
        for _attempt in range(60):
            try:
                res = run_pipeline(q, client, embeddings, groq_client, memory=memory)
                break
            except RateLimitError as e:
                wait = quota_wait_seconds(str(e))
                if wait is None:
                    raise
                wait = min(wait + 5, 950.0)
                log(f"  quota 429 on Q{cur}, waiting {wait:.0f}s ...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Q{cur}: quota window never opened")
        time.sleep(CALL_SLEEP)
        try:
            jscore, jverdict, jreason = judge_hard(
                groq_client, q, ref, res.answer, history, judge_model
            )
        except RateLimitError as e:
            # long judge-side quota window: wait it out once, then retry
            wait = quota_wait_seconds(str(e))
            if wait is not None:
                log(f"  judge quota 429 on Q{cur}, waiting {wait:.0f}s ...")
                time.sleep(min(wait + 5, 950.0))
                jscore, jverdict, jreason = judge_hard(
                    groq_client, q, ref, res.answer, history, judge_model
                )
            else:
                jscore, jverdict, jreason = 0.0, "error", str(e)
        except Exception as e:  # noqa: BLE001
            jscore, jverdict, jreason = 0.0, "error", str(e)
        time.sleep(CALL_SLEEP)
        row = HardResult(
            qid=cur, session=session, turn=turn, category=category,
            question=q, reference=ref, routed_db=res.routing.database,
            used_fallback=res.used_fallback, docs_found=len(res.docs),
            top_score=round(res.docs[0].score, 4) if res.docs else 0.0,
            answer=res.answer, groundedness=res.evaluation.groundedness_score,
            judge_score=jscore, judge_verdict=jverdict, judge_reason=jreason,
        )
        results.append(row)
        mark = {1.0: "OK", 0.5: "~", 0.0: "X"}.get(jscore, "?")
        where = f"S{session}T{turn}" if session else "solo "
        log(f"  [{mark}] {where} Q{cur:>3} ({category}, docs={row.docs_found}, "
            f"{time.time()-t0:.0f}s) {q[:58]}")
        results_file.write_text(json.dumps([asdict(r) for r in results], indent=2))

    # -- standalone --
    log(f"Running {len(spec['standalone'])} standalone hard questions ...")
    for item in spec["standalone"]:
        process(item["question"], item["reference_answer"], item["category"],
                0, 0, None, "")

    # -- follow-up sessions --
    for sess in spec["sessions"]:
        log(f"Session {sess['id']}: {sess['topic']} ({len(sess['turns'])} turns) ...")
        memory = ConversationMemory(max_turns=5)
        history_lines: list[str] = []
        for ti, turn in enumerate(sess["turns"], 1):
            history = "\n".join(history_lines)
            process(turn["question"], turn["reference_answer"], "followup",
                    sess["id"], ti, memory, history)
            last = results[-1]
            history_lines.append(f"User: {turn['question']}")
            history_lines.append(f"Assistant: {last.answer[:400]}")

    return results


def write_hard_report(results: list[HardResult], label: str, gen_model: str,
                      judge_model: str | None, chunks: int, ingested: int) -> None:
    scores = [r.judge_score for r in results]
    acc = statistics.mean(scores) if scores else 0.0
    correct = sum(1 for r in results if r.judge_score == 1.0)
    partial = sum(1 for r in results if r.judge_score == 0.5)
    incorrect = sum(1 for r in results if r.judge_score == 0.0)

    def acc_of(rs: list[HardResult]) -> tuple[float, int]:
        return (statistics.mean([r.judge_score for r in rs]), len(rs)) if rs else (0.0, 0)

    cats = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)

    solo = [r for r in results if r.session == 0]
    sess = [r for r in results if r.session > 0]
    solo_acc, solo_n = acc_of(solo)
    sess_acc, sess_n = acc_of(sess)

    lines: list[str] = []
    w = lines.append
    w("# RAG Agent — HARD 100-Question Benchmark (Adversarial + Follow-ups)")
    w("")
    w(f"_Run label: `{label}` · Generated: {time.strftime('%Y-%m-%d %H:%M')}_  ")
    w(f"_Generator: `{gen_model}` · Judge: `{judge_model or gen_model}` · "
      f"Pipeline: guardrails → router → hybrid retrieval (top-8) → grounded generator → faithfulness evaluator · "
      f"Indexed: **{ingested}** chunks (parser emitted {chunks})_")
    w("")
    w("## Executive Summary")
    w("")
    w(f"- **Overall accuracy: {acc*100:.1f}%** over **100 questions** "
      f"(50 standalone hard + 50 follow-up turns in 10 sessions)")
    w(f"- Correct: **{correct}** · Partial: **{partial}** · Incorrect: **{incorrect}**")
    w(f"- Standalone hard: **{solo_acc*100:.1f}%** ({solo_n}) · Follow-up sessions: **{sess_acc*100:.1f}%** ({sess_n})")
    grounded = statistics.mean(r.groundedness for r in results) if results else 0.0
    w(f"- Mean groundedness (local faithfulness evaluator): **{grounded*100:.0f}%** · "
      f"Routing target: `{COLLECTION}` DB hit on all answered questions unless noted below")
    w("")
    w("## Accuracy by Difficulty Category")
    w("")
    w("| Category | Questions | Accuracy |")
    w("|---|---|---|")
    for cat in ("numeric_trap", "multi_hop", "disambiguation", "guidance_policy", "unanswerable", "followup"):
        if cat in cats:
            a, n = acc_of(cats[cat])
            w(f"| `{cat}` | {n} | **{a*100:.1f}%** |")
    w("")
    w("## Follow-up Memory Retention (accuracy by turn position)")
    w("")
    w("| Turn | Accuracy | Note |")
    w("|---|---|---|")
    for t in range(1, 6):
        rs = [r for r in sess if r.turn == t]
        a, n = acc_of(rs)
        note = "self-contained" if t == 1 else ("elliptical references require memory" if t >= 3 else "context-dependent")
        w(f"| Turn {t} | **{a*100:.1f}%** ({n}) | {note} |")
    w("")

    fails = [r for r in results if r.judge_score < 1.0]
    w("## Failure Analysis")
    w("")
    if not fails:
        w("No failures — all 100 questions answered correctly.")
    else:
        for r in fails:
            where = f"Session {r.session}, turn {r.turn}" if r.session else "Standalone"
            w(f"### Q{r.qid} ({where}, `{r.category}`). {r.question}")
            w(f"- Reference: {r.reference}")
            w(f"- Answer: {r.answer.strip()[:400]}")
            w(f"- Verdict: `{r.judge_verdict}` ({r.judge_score:.1f}) — {r.judge_reason}")
            w(f"- Retrieval: {r.docs_found} docs, top `{r.top_score:.3f}` · routed `{r.routed_db}` · "
              f"fallback `{r.used_fallback}` · groundedness `{r.groundedness:.2f}`")
            w("")

    w("## Full Results")
    w("")
    w("| # | Where | Category | Question | Verdict | Score | Docs | Top |")
    w("|---|---|---|---|---|---|---|---|")
    for r in results:
        where = f"S{r.session}.{r.turn}" if r.session else "—"
        q_short = r.question[:55] + ("..." if len(r.question) > 55 else "")
        w(f"| {r.qid} | {where} | {r.category} | {q_short} | {r.judge_verdict} | "
          f"{r.judge_score:.1f} | {r.docs_found} | {r.top_score:.3f} |")
    w("")

    out = ARTIFACTS / "EVALUATION_HARD.md"
    out.write_text("\n".join(lines))
    log(f"Report written: {out}")
    print("\n" + "=" * 64)
    print(f"HARD BENCHMARK ACCURACY: {acc*100:.1f}%  "
          f"({correct} correct / {partial} partial / {incorrect} incorrect)")
    print(f"  standalone {solo_acc*100:.1f}% · follow-ups {sess_acc*100:.1f}%")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="hard1")
    ap.add_argument("--judge-model", default=None,
                    help="model for judging (default: RAG_MODEL / pipeline MODEL)")
    ap.add_argument("--pdf", default=None)
    args = ap.parse_args()

    if not os.getenv("GROQ_API_KEY", "").strip():
        log("ERROR: GROQ_API_KEY not set. Aborting."); sys.exit(1)

    spec = json.loads(QUESTIONS_FILE.read_text())
    n_solo = len(spec["standalone"])
    n_sess = sum(len(s["turns"]) for s in spec["sessions"])
    log(f"Loaded hard benchmark: {n_solo} standalone + {n_sess} session turns = {n_solo + n_sess} questions")

    pdf_path = resolve_pdf_path(args.pdf)
    groq_client = OpenAI(base_url="https://api.groq.com/openai/v1",
                         api_key=os.getenv("GROQ_API_KEY"))

    log("Building in-memory Qdrant + FastEmbed embeddings ...")
    client, embeddings = build_databases()

    log(f"Parsing PDF: {pdf_path.name} ...")
    t0 = time.time()
    chunks, engine = DocumentParser.parse_file(str(pdf_path), pdf_path.name)
    log(f"  -> {len(chunks)} chunks via {engine} in {time.time()-t0:.1f}s")
    ingested = add_documents(client, embeddings, COLLECTION, chunks)
    log(f"  Ingested {ingested} chunks into '{COLLECTION}'")

    gen_model = os.getenv("RAG_MODEL", MODEL)
    log(f"Generator/router model: {gen_model} · Judge: {args.judge_model or gen_model}")

    results = run_hard_benchmark(groq_client, client, embeddings, spec,
                                 args.label, args.judge_model)
    write_hard_report(results, args.label, gen_model, args.judge_model,
                      len(chunks), ingested)


if __name__ == "__main__":
    main()
