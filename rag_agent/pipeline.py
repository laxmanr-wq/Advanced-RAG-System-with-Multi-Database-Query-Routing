"""
RAG Agent with Database Routing - Production AI System Pipeline.

Orchestrates: Input Guardrail -> Contextualize -> Route -> Retrieve/Grade/Rewrite
-> Generate -> Faithfulness Evaluation -> Telemetry Trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI
from qdrant_client import QdrantClient

from .contextualize import contextualize_query
from .embeddings import FastEmbeddings
from .evaluator import EvaluationResult, FaithfulnessEvaluator
from .fallback import run_fallback
from .grading import grade_documents
from .guardrails import GuardrailResult, InputGuardrail, OutputGuardrail
from .llm import get_active_model
from .memory import ConversationMemory
from .postprocess import numbers_grounded, strip_citations
from .query_rewrite import rewrite_query
from .quota import chat_with_quota_retry
from .retriever import RetrievedDoc, retrieve
from .router import RoutingDecision, route_query
from .telemetry import ExecutionTrace, Timer

MODEL = get_active_model()  # re-exported for the evaluation harnesses

# Adaptive retrieval: how many retrieve -> grade -> rewrite cycles to allow
# before giving up on local context and handing off to the web fallback.
MAX_RETRIEVAL_ATTEMPTS = 2

RAG_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "Start with the direct answer, be concise, and cite specific details from the context. "
    "Reproduce numbers, dates, years, units, and names EXACTLY as written in the context; "
    "never round, convert, or estimate. Keep amounts in the context's notation: parentheses "
    "around a number mean it is negative, e.g. (123) = -123. "
    "When a table has multiple columns (e.g. different years), report ONLY the value from the "
    "column the question asks about. If the question does not name a year, use the most recent "
    "year shown (the first data column) and state that year. Give the single requested value; "
    "do not list values for several years unless asked. "
    "Never include bracketed source markers like [1] or [3] in your answer. "
    "If the context does not contain enough information, say so clearly."
)


@dataclass
class PipelineResult:
    """The final answer plus routing, retrieval, evaluation, and telemetry metadata."""

    answer: str
    routing: RoutingDecision
    docs: list[RetrievedDoc] = field(default_factory=list)
    used_fallback: bool = False
    retrieval_attempts: int = 1
    contextualized_query: str | None = None
    rewritten_query: str | None = None
    guardrail: GuardrailResult = field(
        default_factory=lambda: GuardrailResult(passed=True, reason="Passed")
    )
    evaluation: EvaluationResult = field(
        default_factory=lambda: EvaluationResult(
            groundedness_score=1.0, is_faithful=True, status_label="Grounded"
        )
    )
    trace: ExecutionTrace = field(default_factory=ExecutionTrace)


def build_pipeline(groq_api_key: str) -> tuple[QdrantClient, FastEmbeddings, OpenAI]:
    """
    Initialise and return all pipeline components.
    Call once at startup and reuse across queries.
    Returns: (qdrant_client, embeddings, groq_client)
    """
    from .databases import build_databases

    client, embeddings = build_databases()
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
    )
    return client, embeddings, groq_client


def run_pipeline(
    query: str,
    client: QdrantClient,
    embeddings: FastEmbeddings,
    groq_client: OpenAI,
    memory: ConversationMemory | None = None,
) -> PipelineResult:
    """
    Full Production AI System pipeline:
      1. Guardrail Check (Input Safety)
      2. History-Aware Query Contextualization
      3. Query Classification & Routing
      4. Adaptive Retrieval (retrieve -> grade -> rewrite -> retry)
      5. Grounded LLM Answer Generation (or Web Fallback)
      6. Faithfulness Evaluation & Output Guardrail
      7. Telemetry & Execution Trace Collection
    """
    trace = ExecutionTrace()

    # Step 1: Input Guardrails
    with Timer() as t_guard:
        guard_input = InputGuardrail.validate(query)
    trace.add_step("Input Guardrail", t_guard.elapsed_ms, guard_input.reason)

    if not guard_input.passed:
        trace.calculate_total_latency()
        return PipelineResult(
            answer=f"⚠️ Safety Guardrail Alert: {guard_input.reason}",
            routing=RoutingDecision(database="support", reasoning="Blocked by Input Guardrail"),
            used_fallback=True,
            guardrail=guard_input,
            evaluation=EvaluationResult(
                groundedness_score=0.0, is_faithful=False, status_label="Rejected by Policy"
            ),
            trace=trace,
        )

    # Step 2: Contextualize follow-ups against conversation history.
    # Routing and retrieval need a standalone question; "and in 2018?" carries
    # no topical signal on its own. The ORIGINAL query is still what gets
    # answered - only the routing/search query is resolved here.
    search_query = query
    if memory and memory.messages:
        with Timer() as t_ctx:
            search_query = contextualize_query(groq_client, query, memory)
        trace.add_step(
            "Query Contextualizer",
            t_ctx.elapsed_ms,
            f'Resolved to: "{search_query}"' if search_query != query else "Already standalone",
        )

    # Step 3: Route
    with Timer() as t_route:
        routing = route_query(groq_client, search_query)
    trace.add_step("Query Router", t_route.elapsed_ms, f"Routed to '{routing.database}' DB")

    # Step 4: Retrieve, with adaptive grading + query rewriting.
    # Mirrors the agentic RAG pattern: retrieve -> grade -> (rewrite -> retry)*
    # -> generate. A rewrite/retry is only attempted when the routed collection
    # DID return candidates but they graded as off-topic; an empty result set
    # already means nothing in that collection clears the score threshold, so
    # retrying there is skipped in favor of the web fallback.
    current_query = search_query
    attempt = 0
    rewritten_query: str | None = None
    docs: list[RetrievedDoc] = []
    grading = None

    while True:
        attempt += 1
        with Timer() as t_ret:
            docs = retrieve(current_query, routing.database, client, embeddings)
        trace.add_step(
            "Vector Retriever",
            t_ret.elapsed_ms,
            f"Attempt {attempt}: retrieved {len(docs)} docs from '{routing.database}'"
            + (f' for rewritten query "{current_query}"' if attempt > 1 else ""),
        )

        if not docs:
            break

        with Timer() as t_grade:
            grading = grade_documents(groq_client, current_query, [d.text for d in docs])
        trace.add_step(
            "Document Grader",
            t_grade.elapsed_ms,
            f"{'Relevant' if grading.is_relevant else 'Not relevant'}: {grading.reasoning}",
        )

        if grading.is_relevant or attempt >= MAX_RETRIEVAL_ATTEMPTS:
            break

        with Timer() as t_rewrite:
            current_query = rewrite_query(groq_client, search_query)
        rewritten_query = current_query
        trace.add_step("Query Rewriter", t_rewrite.elapsed_ms, f'Rewrote to: "{rewritten_query}"')

    context_is_grounded = bool(docs) and grading is not None and grading.is_relevant

    # Step 5: Generation / Fallback
    effective_model = get_active_model()
    if context_is_grounded:
        context_str = "\n\n".join(f"[{i+1}] {doc.text}" for i, doc in enumerate(docs))

        # Memory inclusion if provided
        history = memory.format_history_context() if memory else ""
        full_user_prompt = (
            f"Conversation History:\n{history}\n\nContext:\n{context_str}\n\nQuestion: {query}"
            if history
            else f"Context:\n{context_str}\n\nQuestion: {query}"
        )

        with Timer() as t_gen:
            response = chat_with_quota_retry(
                groq_client,
                model=effective_model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM},
                    {"role": "user", "content": full_user_prompt},
                ],
                temperature=0.2,
            )
            # Strip source markers first: their digits (e.g. 【1†L2-L4】) would
            # otherwise count as ungrounded numbers and trigger a needless retry.
            answer = strip_citations(response.choices[0].message.content or "")
            # Deterministic anti-hallucination guard: if the answer contains a
            # number that appears nowhere in the retrieved context, regenerate once.
            doc_texts_gen = [d.text for d in docs]
            if not numbers_grounded(answer, doc_texts_gen):
                response = chat_with_quota_retry(
                    groq_client,
                    model=effective_model,
                    messages=[
                        {"role": "system", "content": RAG_SYSTEM},
                        {"role": "user", "content": full_user_prompt},
                        {"role": "assistant", "content": answer},
                        {"role": "user", "content": "Some numbers in your answer do not appear in the context. Answer again using ONLY numbers found in the context."},
                    ],
                    temperature=0.2,
                )
                answer = response.choices[0].message.content or answer
            answer = strip_citations(answer)
        trace.add_step("LLM Grounded Generator", t_gen.elapsed_ms, f"Model: {effective_model}")
        used_fallback = False
    else:
        with Timer() as t_fall:
            # Search the resolved question: "and in 2018?" is useless as a web query.
            # Strip source markers here too - the fallback model emits them just
            # like the grounded generator does.
            answer = strip_citations(run_fallback(groq_client, search_query))
        fallback_reason = (
            "no relevant local documents after retrieval + rewrite" if docs else "no local documents cleared the score threshold"
        )
        trace.add_step("Web Fallback Agent", t_fall.elapsed_ms, f"Searched DuckDuckGo ({fallback_reason})")
        used_fallback = True

    # Step 6: Output Guardrail & Faithfulness Evaluation
    with Timer() as t_eval:
        guard_output = OutputGuardrail.validate(answer)
        doc_texts = [d.text for d in docs] if context_is_grounded else []
        evaluation = FaithfulnessEvaluator.evaluate(answer, doc_texts)
    trace.add_step(
        "Faithfulness Evaluator",
        t_eval.elapsed_ms,
        f"Groundedness Score: {evaluation.groundedness_score}",
    )

    # Step 7: Update Memory
    if memory:
        memory.add_user_message(query)
        memory.add_assistant_message(answer)

    trace.calculate_total_latency()

    return PipelineResult(
        answer=answer,
        routing=routing,
        docs=docs if context_is_grounded else [],
        used_fallback=used_fallback,
        retrieval_attempts=attempt,
        contextualized_query=search_query if search_query != query else None,
        rewritten_query=rewritten_query,
        guardrail=guard_output,
        evaluation=evaluation,
        trace=trace,
    )
