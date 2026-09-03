"""
RAG Agent with Database Routing - fallback agent.

Activated when the Qdrant retriever finds no relevant documents.
Searches the web with DuckDuckGo and answers via Groq openai/gpt-oss-120b.
"""

from __future__ import annotations


from ddgs import DDGS
from openai import OpenAI

from .llm import get_active_model
from .quota import chat_with_quota_retry


def run_fallback(client: OpenAI, query: str) -> str:
    """Search the web and generate an answer using Groq."""
    try:
        results = list(DDGS().text(query, max_results=5))
    except Exception:
        results = []

    if not results:
        context = "No web results found."
    else:
        context = "\n\n".join(
            f"{r.get('title', '')}: {r.get('body', '')}" for r in results
        )

    response = chat_with_quota_retry(
        client,
        model=get_active_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using the web search results provided. "
                    "Be concise and factual. Cite key details from the results."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nSearch results:\n{context}",
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
