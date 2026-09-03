"""
Zero-API smoke evaluation for CI: exercises ingestion transforms, hybrid
retrieval wiring, reranking, and post-processing guards with synthetic data.
Fails the build if any regression breaks the retrieval/grounding contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.databases import add_documents, build_databases
from rag_agent.parser import consolidate_chunks, serialize_tables_batch
from rag_agent.postprocess import extract_numbers, numbers_grounded, strip_citations
from rag_agent.reranker import rerank
from rag_agent.retriever import RetrievedDoc, retrieve

TABLE_CHUNKS = [
    "Statement of Financial Position *As at December 31*",
    "**Note 2019 2018 ASSETS**",
    "| Monetary gold | 6 | 108,000 | 94,000 |",
    "|---|---|---|---|",
    "| Non-monetary gold | 11 | 16,672 | 4,437 |",
    "The statements are presented in thousands of local currency.",
]


def check(name: str, cond: bool) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main() -> int:
    ok = True

    # 1. Consolidation merges fragments
    cons = consolidate_chunks(TABLE_CHUNKS, max_chars=1100)
    ok &= check("consolidation merges fragments", len(cons) < len(TABLE_CHUNKS))

    # 2. Table serialization labels values with years
    ser = serialize_tables_batch(cons)
    joined = "\n".join(ser)
    ok &= check(
        "serialization adds year-labeled figures",
        "Non-monetary gold: 16,672 (2019); 4,437 (2018)" in joined,
    )

    # 3. Post-processing guards
    ok &= check("citation stripping", strip_citations("Answer [3] here [12].") == "Answer here.")
    ok &= check(
        "full-width citation stripping",
        strip_citations("It has 32 GB of RAM 【1†L2-L4】.") == "It has 32 GB of RAM.",
    )
    ok &= check("number extraction", extract_numbers("value is (10,208,747)") == {"10208747"})
    ok &= check(
        "numeric grounding accepts context numbers",
        numbers_grounded("Gold is 108,000.", ["Monetary gold | 6 | 108,000 | 94,000 |"]),
    )
    ok &= check(
        "numeric grounding rejects invented numbers",
        not numbers_grounded("Gold is 999,999.", ["Monetary gold | 6 | 108,000 |"]),
    )

    # 4. Ingestion + hybrid retrieval end-to-end (in-memory, no API)
    client, embeddings = build_databases()
    n = add_documents(client, embeddings, "financial", TABLE_CHUNKS)
    ok &= check("ingestion returns documents", n > 0)
    docs = retrieve("What is the value of non-monetary gold?", "financial", client, embeddings, top_k=4)
    ctx = " ".join(d.text for d in docs)
    ok &= check("retrieval finds the target value", "16,672" in ctx.replace(",", "") or "16672" in ctx.replace(",", ""))

    # 5. Reranker re-orders without dropping
    fake = [
        RetrievedDoc(text="unrelated weather report", score=0.9, source="x"),
        RetrievedDoc(text="non-monetary gold value 16,672 (2019)", score=0.6, source="x"),
    ]
    ranked = rerank("What is the value of non-monetary gold?", fake, top_k=2)
    ok &= check("reranker lifts relevant doc to rank 1", "16,672" in ranked[0].text)

    print("SMOKE EVAL:", "ALL PASSED" if ok else "FAILURES DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
