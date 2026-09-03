"""
RAG Agent with Database Routing - router.

Uses Groq with openai/gpt-oss-120b to classify each query into one of
three databases and return a structured RoutingDecision.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from .llm import get_active_model
from .quota import chat_with_quota_retry

_DB_OPTIONS = ("products", "support", "financial")

# Keywords used as a fallback when the model does not return valid JSON
_KEYWORDS: dict[str, list[str]] = {
    "financial": [
        "price", "pricing", "plan", "billing", "invoice", "payment",
        "revenue", "contract", "discount", "tax", "vat", "cost", "fee",
        "subscription", "refund", "charge",
    ],
    "support": [
        "reset", "password", "account", "locked", "cancel", "troubleshoot",
        "slow", "error", "help", "contact", "return", "policy", "issue",
        "invite", "2fa", "two-factor", "authentication", "export",
    ],
}


class RoutingDecision(BaseModel):
    """Holds which database a query was routed to and why."""

    database: Literal["products", "support", "financial"]
    reasoning: str


ROUTING_INSTRUCTIONS = """\
You are a query routing agent. Classify the user query into exactly one of three databases:

- products: Questions about product features, specifications, availability,
  hardware, software tools, or what a product does.

- support: Questions about account issues, troubleshooting, how-to guides,
  policies (returns, refunds, cancellation), contacting support, or fixing a problem.

- financial: Questions about pricing, plans, billing, invoices, payments,
  revenue reports, contracts, discounts, or taxes.

You MUST respond with ONLY a JSON object and nothing else. No explanation, no markdown.
Example: {"database": "support", "reasoning": "User asks about password reset"}
"""


def _keyword_fallback(query: str) -> RoutingDecision:
    """Classify by keyword matching when the model returns non-JSON."""
    lower = query.lower()
    for db in ("financial", "support"):  # products is the default
        if any(kw in lower for kw in _KEYWORDS[db]):
            return RoutingDecision(
                database=db,
                reasoning="Keyword match fallback (model did not return JSON).",
            )
    return RoutingDecision(
        database="products",
        reasoning="Keyword fallback default.",
    )


def build_router(groq_api_key: str) -> OpenAI:
    """Return a Groq OpenAI client for routing."""
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)


def route_query(client: OpenAI, query: str) -> RoutingDecision:
    """Classify a query and return the routing decision."""
    if not query or not query.strip():
        return _keyword_fallback(query)

    try:

        response = chat_with_quota_retry(
            client,
            model=get_active_model(),
            messages=[
                {"role": "system", "content": ROUTING_INSTRUCTIONS},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
    except Exception:
        return _keyword_fallback(query)

    # Strip markdown code fences if wrapped
    text = re.sub(r"```(?:json)?\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL).strip()

    if not text:
        return _keyword_fallback(query)

    try:
        data = json.loads(text)
        if data.get("database") in _DB_OPTIONS:
            return RoutingDecision(**data)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Try to find a JSON object embedded in prose
    match = re.search(r'\{[^{}]*"database"\s*:[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if data.get("database") in _DB_OPTIONS:
                return RoutingDecision(**data)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return _keyword_fallback(query)
