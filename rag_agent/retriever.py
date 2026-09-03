"""
RAG Agent with Database Routing - Qdrant hybrid retriever.

Embeds the query and searches the routed collection with dense vectors,
optionally fused with BM25 sparse vectors (Reciprocal Rank Fusion) so exact
keyword/number matches rank higher alongside semantic matches.
Returns an empty list when no document is found, which signals the pipeline
to trigger the fallback agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from .databases import SCORE_THRESHOLD, SPARSE_VECTOR_NAME
from .embeddings import FastEmbeddings
from .reranker import rerank

RRF_K = 60  # standard RRF smoothing constant


@dataclass
class RetrievedDoc:
    """A single document retrieved from a Qdrant collection, with its similarity score."""

    text: str
    score: float
    source: str


def retrieve(
    query: str,
    collection: str,
    client: QdrantClient,
    embeddings: FastEmbeddings,
    top_k: int = 8,
) -> list[RetrievedDoc]:
    """
    Embed the query and search the specified Qdrant collection.
    Dense results must clear SCORE_THRESHOLD; sparse BM25 results (when
    available) are fused via Reciprocal Rank Fusion to improve ranking of
    exact-match content. An empty list means the fallback agent should be used.
    """
    query_vector = embeddings.embed_query(query)

    dense_response = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k * 2,
        score_threshold=SCORE_THRESHOLD,
    )
    dense_hits = list(dense_response.points)

    sparse_hits: list = []
    try:
        if getattr(embeddings, "has_sparse", False):
            sparse_query = embeddings.embed_query_sparse(query)
            if sparse_query:
                indices, values = sparse_query
                from qdrant_client.models import SparseVector

                sparse_response = client.query_points(
                    collection_name=collection,
                    query=SparseVector(indices=indices, values=values),
                    using=SPARSE_VECTOR_NAME,
                    limit=top_k * 2,
                )
                sparse_hits = list(sparse_response.points)
    except Exception:
        sparse_hits = []

    if not dense_hits and not sparse_hits:
        return []

    # Reciprocal Rank Fusion across the two ranked lists
    def hit_key(hit):
        return (hit.payload.get("text", ""), hit.payload.get("source", collection))

    rrf_scores: dict = {}
    primary: dict = {}
    dense_scores: dict = {}
    sparse_scores: dict = {}
    for rank, hit in enumerate(dense_hits, 1):
        key = hit_key(hit)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        dense_scores[key] = hit.score
        primary.setdefault(key, hit)
    for rank, hit in enumerate(sparse_hits, 1):
        key = hit_key(hit)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        sparse_scores[key] = hit.score
        primary.setdefault(key, hit)

    # BM25 scores live on a different scale than cosine similarities; keep
    # them separate and rescale sparse-only hits to 0..1 so they never
    # distort the dense tie-breaker or the reranker's score normalization.
    max_sparse = max(sparse_scores.values(), default=0.0) or 1.0

    def doc_score(key):
        if key in dense_scores:
            return dense_scores[key]
        return sparse_scores.get(key, 0.0) / max_sparse

    fused_keys = sorted(
        rrf_scores,
        key=lambda k: (rrf_scores[k], doc_score(k)),
        reverse=True,
    )[:top_k]

    candidates = [
        RetrievedDoc(
            text=primary[key].payload.get("text", ""),
            score=doc_score(key),
            source=primary[key].payload.get("source", collection),
        )
        for key in fused_keys
    ]
    # Second stage: reorder by lexical/cross-encoder relevance so the most
    # on-point chunks reach the generator first (never drops documents).
    return rerank(query, candidates, top_k)
