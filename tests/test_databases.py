"""Tests for Qdrant setup: in-memory vs persistent storage modes."""

from __future__ import annotations

from rag_agent.databases import add_documents, build_databases, doc_count

TEXTS = [
    "Non-monetary gold holdings were 16,672 in 2019 and 4,437 in 2018.",
    "Monetary gold was reported at 108,000.",
]


def test_build_databases_in_memory():
    client, embeddings = build_databases()
    for name in ("products", "support", "financial"):
        assert client.collection_exists(name)
    assert embeddings is not None


def test_build_databases_persistent_roundtrip(tmp_path):
    """Data ingested under a storage path survives a client rebuild."""
    client, embeddings = build_databases(storage_path=str(tmp_path))
    added = add_documents(client, embeddings, "financial", TEXTS)
    assert added > 0
    del client

    client2, embeddings2 = build_databases(storage_path=str(tmp_path))
    assert doc_count(client2, "financial") == added
    # Idempotent rebuild must not wipe or duplicate existing collections
    assert doc_count(client2, "financial") == added


def test_build_databases_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path))
    client, _ = build_databases()
    assert client.collection_exists("financial")
