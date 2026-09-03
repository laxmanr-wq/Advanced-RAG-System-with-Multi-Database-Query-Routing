"""
RAG Agent with Database Routing - Adaptive query rewriting.

Ported from the agentic RAG pattern's rewrite_query node: invoked when the
document grader judges the first retrieval irrelevant. Reformulates the query
into more specific, keyword-rich terms before retrying retrieval against the
same routed collection.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from .llm import get_active_model
from .quota import chat_with_quota_retry

REWRITE_INSTRUCTIONS = """\
The following question did not retrieve relevant results from the knowledge base:

Question: {question}

Rewrite it as a more specific, keyword-rich question that is more likely to match
relevant document chunks. Keep the same intent and do not invent new facts.

Respond with ONLY a JSON object: {{"rewritten_query": "<improved question>"}}
"""


def rewrite_query(client: OpenAI, original_query: str) -> str:
    """Reformulate a query for a second retrieval attempt.

    Falls back to simple keyword expansion if the LLM call or parsing fails.
    """
    try:
        response = chat_with_quota_retry(
            client,
            model=get_active_model(),
            messages=[
                {
                    "role": "system",
                    "content": "You rewrite search queries to improve document retrieval. Respond with ONLY a JSON object.",
                },
                {"role": "user", "content": REWRITE_INSTRUCTIONS.format(question=original_query)},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"```(?:json)?\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL).strip()
        data = json.loads(text)
        rewritten = str(data.get("rewritten_query", "")).strip()
        if rewritten:
            return rewritten
    except Exception:
        pass
    return f"{original_query} details information"
