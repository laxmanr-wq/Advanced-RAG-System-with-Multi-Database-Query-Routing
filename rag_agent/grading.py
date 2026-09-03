"""
RAG Agent with Database Routing - Document relevance grader.

Ported from the agentic RAG pattern (guardrail -> retrieve -> grade -> rewrite
-> generate): after retrieval, an LLM checks whether the retrieved context
actually helps answer the query before generation proceeds. A "no" routes the
pipeline to query rewriting + a second retrieval attempt instead of grounding
an answer in off-topic context.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import OpenAI

from .llm import get_active_model
from .quota import chat_with_quota_retry

GRADE_INSTRUCTIONS = """\
You are a grader assessing whether retrieved context is relevant enough to answer a user's question.

Retrieved Context:
{context}

User Question: {question}

Grade "yes" if the context contains information that helps answer the question, even partially.
Grade "no" only if the context is clearly about a different topic and cannot help at all.

Respond with ONLY a JSON object: {{"relevant": "yes" or "no", "reasoning": "<one sentence>"}}
"""


@dataclass
class GradingResult:
    """Outcome of grading retrieved context against a query."""

    is_relevant: bool
    reasoning: str


def grade_documents(client: OpenAI, query: str, context_docs: list[str]) -> GradingResult:
    """Grade whether retrieved context is relevant to the query.

    Fails open (assumes relevant) if the LLM call or JSON parsing fails, so a
    transient grader hiccup never blocks generation on context that may be fine.
    """
    if not context_docs:
        return GradingResult(is_relevant=False, reasoning="No context was retrieved.")

    context = "\n\n".join(context_docs)[:6000]

    try:
        response = chat_with_quota_retry(
            client,
            model=get_active_model(),
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise relevance grader. Respond with ONLY a JSON object, no other text.",
                },
                {"role": "user", "content": GRADE_INSTRUCTIONS.format(context=context, question=query)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
    except Exception:
        return GradingResult(is_relevant=True, reasoning="Grading skipped (LLM call failed).")

    text = re.sub(r"```(?:json)?\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL).strip()
    try:
        data = json.loads(text)
        relevant = str(data.get("relevant", "yes")).strip().lower() == "yes"
        reasoning = str(data.get("reasoning", "")).strip() or ("Relevant" if relevant else "Not relevant")
        return GradingResult(is_relevant=relevant, reasoning=reasoning)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return GradingResult(is_relevant=True, reasoning="Grading skipped (could not parse grader response).")
