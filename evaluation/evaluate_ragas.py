"""
RAGAS evaluation over a saved benchmark results file.

Reconstructs retrieval contexts deterministically (zero generation cost) and
scores the standard RAGAS metrics with a cheap Groq judge:
  faithfulness, answer_relevancy, context_precision, context_recall,
  answer_correctness.

Usage:
  uv run python evaluate_ragas.py --source run4_full_improved \
      --judge-model openai/gpt-oss-20b
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in evaluation/)
sys.path.insert(0, str(ROOT))  # keep `rag_agent` importable when run directly
load_dotenv(ROOT / ".env")

ARTIFACTS = ROOT / "evaluation" / "artifacts"
DEFECTIVE_QIDS = {32, 33}
TOP_K = 8


# RAGAS fires many judge calls in parallel; the Groq free tier needs pacing.
# timeout: Groq queues requests under load - give each request 15 minutes.
# max_retries=1: ragas' tenacity random backoff burns attempts before a
# rolling TPD window frees up; the quota-aware wrapper below does the waiting.
# timeout is a wall-clock cap per metric job INCLUDING quota waits (the
# wrapper sleeps inside the job); Groq's "try again in" hints can exceed 25m,
# so it must be generous. max_wait only bounds tenacity's own backoff.
RAGAS_RUN_CONFIG_KWARGS = {"max_workers": 1, "timeout": 3000, "max_wait": 60, "max_retries": 1}


def _patch_ragas_backoff(judge_llm) -> None:
    """Wrap the judge's agenerate_text so Groq's 'try again in XmYs' hint is
    honored exactly. ragas' own tenacity backoff is random and gives up long
    before a rolling TPD window frees up, so every job would die on 429."""
    import asyncio
    import re

    orig = judge_llm.agenerate_text

    async def agenerate_text(*args, **kwargs):
        for _ in range(60):  # up to ~hours of rolling-quota waits
            try:
                return await orig(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                m = re.search(r"try again in (\d+)m([\d.]+)s", str(e))
                if m is None:
                    m = re.search(r"try again in ([\d.]+)s", str(e))
                    if m is None:
                        raise
                    wait = float(m.group(1)) + 5
                else:
                    wait = int(m.group(1)) * 60 + float(m.group(2)) + 5
                print(f"[ragas] quota wait {wait:.0f}s ...", flush=True)
                await asyncio.sleep(wait)
        return await orig(*args, **kwargs)

    judge_llm.agenerate_text = agenerate_text


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="results label to evaluate")
    ap.add_argument("--judge-model", default="openai/gpt-oss-20b")
    ap.add_argument("--limit", type=int, default=0, help="evaluate first N only (0 = all)")
    ap.add_argument("--pdf", default=None)
    args = ap.parse_args()

    from datasets import Dataset
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    from evaluate_pdf_module import COLLECTION, resolve_pdf_path
    from langchain_core.embeddings import Embeddings as LCEmbeddings

    from rag_agent.databases import add_documents, build_databases
    from rag_agent.parser import DocumentParser
    from rag_agent.retriever import retrieve

    # -- Wrappers ---------------------------------------------------------------
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model=args.judge_model,
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY"),
            # Groq queues requests under load; the openai SDK default (600s)
            # is what produced the TimeoutError job failures.
            timeout=1200,
            max_retries=0,  # the quota-aware wrapper below owns retries
        )
    )
    _patch_ragas_backoff(judge_llm)

    client, fast_emb = build_databases()

    class _FastEmbed(LCEmbeddings):
        def embed_documents(self, texts):
            return fast_emb.embed_documents(texts)

        def embed_query(self, text):
            return fast_emb.embed_query(text)

    emb_wrapper = LangchainEmbeddingsWrapper(_FastEmbed())

    # -- Context reconstruction ---------------------------------------------------
    pdf_path = resolve_pdf_path(args.pdf)
    log(f"Parsing + ingesting {pdf_path.name} ...")
    chunks, engine = DocumentParser.parse_file(str(pdf_path), pdf_path.name)
    if not chunks:
        raise SystemExit(
            f"ERROR: parser returned 0 chunks for {pdf_path.name} "
            "(missing dependency like pypdf?) - contexts would be empty."
        )
    add_documents(client, fast_emb, COLLECTION, chunks)
    log(f"  {len(chunks)} parser chunks indexed via hybrid retriever")

    rows = json.loads((ARTIFACTS / f"mod01_results_{args.source}.json").read_text())
    rows = [r for r in rows if r["qid"] not in DEFECTIVE_QIDS]
    if args.limit:
        rows = rows[: args.limit]

    data = {"user_input": [], "retrieved_contexts": [], "response": [], "reference": []}
    for r in rows:
        docs = retrieve(r["question"], COLLECTION, client, fast_emb, top_k=TOP_K)
        data["user_input"].append(r["question"])
        data["retrieved_contexts"].append([d.text for d in docs])
        data["response"].append(r["answer"])
        data["reference"].append(r["reference"])
    log(f"Evaluating {len(rows)} questions with RAGAS (judge: {args.judge_model}) ...")

    dataset = Dataset.from_dict(data)
    report = evaluate(
        dataset,
        llm=judge_llm,
        embeddings=emb_wrapper,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
        run_config=RunConfig(**RAGAS_RUN_CONFIG_KWARGS),
    )

    df = report.to_pandas()
    metric_cols = [c for c in df.columns if c not in ("user_input", "retrieved_contexts", "response", "reference")]
    print("=" * 64)
    print(f"RAGAS METRICS ({args.source}, judge={args.judge_model}, n={len(df)})")
    print("=" * 64)
    scores = []
    for c in metric_cols:
        vals = [v for v in df[c].tolist() if v == v]  # drop NaN rows
        mean = statistics.mean(vals) if vals else float("nan")
        scores.append(mean)
        print(f"  {c:>22}: {mean:.4f}  (n={len(vals)})")
    valid = [s for s in scores if s == s]
    print(f"  {'MEAN':>22}: {statistics.mean(valid):.4f}" if valid else "  MEAN: all NaN")

    out = ARTIFACTS / f"mod01_ragas_{args.source}.json"
    payload = {
        c: statistics.mean([v for v in df[c].tolist() if v == v])
        if any(v == v for v in df[c].tolist()) else None
        for c in metric_cols
    }
    payload["n_questions"] = len(df)
    payload["per_question"] = df.to_dict(orient="records")
    out.write_text(json.dumps(payload, indent=2, default=str))
    log(f"Saved -> {out}")


if __name__ == "__main__":
    main()
