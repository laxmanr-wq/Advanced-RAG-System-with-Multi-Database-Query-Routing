"""
RAG Agent with Database Routing - Qdrant database setup.

Three Qdrant collections are created on startup: in-memory by default, or
persisted to disk when QDRANT_PATH is set (collections are created only if
missing, so restarts keep previously ingested documents).
Documents are added by the user through the UI.
"""

from __future__ import annotations

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from .embeddings import FastEmbeddings
from .parser import consolidate_chunks, serialize_tables_batch

VECTOR_SIZE = 384  # BAAI/bge-small-en-v1.5 output dimension
SPARSE_VECTOR_NAME = "bm25"  # Qdrant/bm25 sparse vectors for hybrid retrieval
COLLECTIONS = ["products", "support", "financial"]
SCORE_THRESHOLD = 0.5


def build_databases(storage_path: str | None = None) -> tuple[QdrantClient, FastEmbeddings]:
    """Create Qdrant collections (dense + BM25 sparse vectors).

    Storage location: `storage_path` arg > QDRANT_PATH env > in-memory.
    With disk storage, existing collections are reused (idempotent startup).
    """
    path = storage_path or os.getenv("QDRANT_PATH", "").strip()
    client = QdrantClient(path=path) if path else QdrantClient(":memory:")
    embeddings = FastEmbeddings()

    sparse_config = (
        {SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams())}
        if embeddings.has_sparse
        else None
    )

    for collection_name in COLLECTIONS:
        if client.collection_exists(collection_name):
            continue
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            sparse_vectors_config=sparse_config,
        )

    return client, embeddings


def add_documents(
    client: QdrantClient,
    embeddings: FastEmbeddings,
    collection_name: str,
    texts: list[str],
) -> int:
    """Embed and insert a list of text chunks into the given collection.

    Consecutive tiny chunks are consolidated first so that table rows keep
    their header/column context inside a single indexed document.
    Returns the number of documents successfully added.
    """
    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        return 0

    texts = consolidate_chunks(texts)
    # Re-state table rows as 'item: value (year)' so column context survives
    texts = serialize_tables_batch(texts)

    vectors = embeddings.embed_documents(texts)
    sparse_vectors = embeddings.embed_documents_sparse(texts)

    points = []
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        if sparse_vectors and i < len(sparse_vectors):
            indices, values = sparse_vectors[i]
            vector_payload = {
                "": vector,
                SPARSE_VECTOR_NAME: SparseVector(indices=indices, values=values),
            }
        else:
            vector_payload = vector
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector_payload,
                payload={"text": text, "source": collection_name},
            )
        )
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def doc_count(client: QdrantClient, collection_name: str) -> int:
    """Return the number of vectors stored in a collection."""
    info = client.get_collection(collection_name)
    return info.points_count or 0


def reset_databases(client: QdrantClient, embeddings: FastEmbeddings) -> None:
    """Drop and recreate all collections (empty schema, same vector config).

    Used when switching project workspaces: the shared Qdrant store is
    cleared and re-populated from the target project's saved chunks.
    """
    sparse_config = (
        {SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams())}
        if embeddings.has_sparse
        else None
    )
    for collection_name in COLLECTIONS:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            sparse_vectors_config=sparse_config,
        )
