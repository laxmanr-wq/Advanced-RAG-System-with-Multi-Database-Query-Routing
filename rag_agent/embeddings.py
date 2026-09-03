"""
RAG Agent with Database Routing - Local FastEmbed embeddings.

Uses FastEmbed (BAAI/bge-small-en-v1.5) locally without requiring any API key.
Optionally loads a local BM25 sparse model for hybrid retrieval.
"""

from __future__ import annotations

from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"

# FastEmbed defaults to batch_size=256. ONNX pads every sequence in a batch to
# the longest one, and consolidated chunks vary widely in length (~900 chars
# average, 3,700+ at the tail), so large batches spend most of their compute on
# padding. Measured on a 285-chunk sample: batch 8 ran ~2x faster than batch 32.
EMBED_BATCH_SIZE = 8


class FastEmbeddings:
    """Wrapper around local FastEmbed dense model (+ optional BM25 sparse model)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model = TextEmbedding(model_name=model_name)
        self._sparse = None
        try:
            from fastembed import SparseTextEmbedding

            self._sparse = SparseTextEmbedding(model_name=SPARSE_MODEL)
        except Exception:
            self._sparse = None

    @property
    def has_sparse(self) -> bool:
        return self._sparse is not None

    def embed_documents(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
        """Embed a list of documents (index time)."""
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=batch_size)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (query time)."""
        return next(self.model.embed([text])).tolist()

    def embed_documents_sparse(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """BM25 sparse vectors as (indices, values) pairs; [] when unavailable."""
        if not self._sparse:
            return []
        return [
            (e.indices.tolist(), e.values.tolist())
            for e in self._sparse.embed(texts)
        ]

    def embed_query_sparse(self, text: str) -> tuple[list[int], list[float]] | None:
        """BM25 sparse vector for a query, or None when unavailable/empty."""
        if not self._sparse:
            return None
        try:
            e = next(self._sparse.embed([text]))
        except StopIteration:
            return None
        indices, values = e.indices.tolist(), e.values.tolist()
        return (indices, values) if indices else None
