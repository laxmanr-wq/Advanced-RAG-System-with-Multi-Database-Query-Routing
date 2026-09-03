"""
RAG Agent with Database Routing - History-aware query contextualization.

Follow-up questions are elliptical ("and in 2018?", "what two components make
up that total?"). Routing and retrieval see only the raw string, so the router
misfires and the embedding carries almost no topical signal. This step resolves
references against conversation history and produces a standalone question that
routing and retrieval can actually use.

The original question is still what gets answered - only the search/routing
query is rewritten.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from .llm import get_active_model
from .memory import ConversationMemory
from .quota import chat_with_quota_retry

CONTEXTUALIZE_INSTRUCTIONS = """\
Given a conversation history and a follow-up question, rewrite the follow-up as a
standalone question that can be understood without the history.

Resolve every pronoun and reference ("it", "that total", "the same component",
"those two") into the explicit entity it refers to. Preserve the original intent,
wording, and any years or figures exactly. Do not answer the question, and do not
add facts that are not present in the history.

If the question is already standalone, return it unchanged.

Conversation history:
{history}

Follow-up question: {question}

Respond with ONLY a JSON object: {{"standalone_query": "<rewritten question>"}}
"""


def contextualize_query(
    client: OpenAI,
    query: str,
    memory: ConversationMemory | None,
) -> str:
    """Rewrite a follow-up into a standalone question using conversation history.

    Returns the query unchanged when there is no history to resolve against, or
    when the LLM call/parse fails (fail open - never block the pipeline).
    """
    if not memory or not memory.messages or not query.strip():
        return query

    history = memory.format_history_context()
    if not history:
        return query

    try:
        response = chat_with_quota_retry(
            client,
            model=get_active_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite follow-up questions into standalone questions for "
                        "document retrieval. Respond with ONLY a JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": CONTEXTUALIZE_INSTRUCTIONS.format(
                        history=history[:4000], question=query
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"```(?:json)?\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL).strip()
        standalone = str(json.loads(text).get("standalone_query", "")).strip()
        if standalone:
            return standalone
    except Exception:
        pass
    return query
