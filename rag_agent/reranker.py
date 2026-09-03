"""
Second-stage reranking for hybrid retrieval results.

Zero-dependency lexical reranker: token + bigram overlap with extra weight
for exact numbers. Reranking only re-orders candidates; it never drops
documents, so recall is preserved.
"""

from __future__ import annotations

import re

from .postprocess import extract_numbers

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9,.()-]*")


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def lexical_score(query_tokens: list[str], query_bigrams: set, text: str) -> float:
    """0..1 overlap of query terms (and numbers) with the document text."""
    if not query_tokens:
        return 0.0
    doc_tokens = set(_tokens(text))
    doc_text = text.lower()
    unigram = sum(1 for t in query_tokens if t in doc_tokens) / len(query_tokens)
    doc_bigrams = _bigrams(_tokens(text))
    bigram = (
        sum(1 for b in query_bigrams if b in doc_bigrams) / len(query_bigrams)
        if query_bigrams else 0.0
    )
    score = 0.6 * unigram + 0.4 * bigram
    # Distinctive numbers in the query should match exactly (financial
    # lookups). Years (1900-2099) appear in nearly every financial chunk,
    # so they are excluded to avoid noisy boosts.
    qnums = {n for n in extract_numbers(" ".join(query_tokens))
             if len(n) >= 4 and not re.fullmatch(r"(?:19|20)\d{2}", n)}
    if qnums:
        hit = sum(1 for n in qnums if n in doc_text.replace(",", ""))
        score = 0.5 * score + 0.5 * (hit / len(qnums))
    return score


def rerank(query: str, docs: list, top_k: int, dense_weight: float = 0.4) -> list:
    """
    Re-order candidate docs by a blend of normalized dense score and lexical
    relevance. Never drops docs. Lexical gets the larger share because fused
    RRF scores cluster tightly, while exact term/number overlap best
    separates financial lookups.
    """
    if len(docs) <= 1:
        return docs[:top_k]

    qt = _tokens(query)
    qb = _bigrams(qt)
    max_dense = max(d.score for d in docs) or 1.0

    scored = []
    for d in docs:
        lex = lexical_score(qt, qb, d.text)
        combined = dense_weight * (d.score / max_dense) + (1 - dense_weight) * lex
        scored.append((combined, d.score, d))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [d for _, _, d in scored[:top_k]]
