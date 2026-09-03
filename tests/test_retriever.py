"""Unit tests for rag_agent.retriever."""

from unittest.mock import MagicMock

from rag_agent.retriever import RetrievedDoc, retrieve


def test_retriever_returns_matching_documents():
    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 384

    hit1 = MagicMock()
    hit1.score = 0.85
    hit1.payload = {"text": "TechPro X1 features 16GB RAM.", "source": "products"}

    response = MagicMock()
    response.points = [hit1]
    mock_client.query_points.return_value = response

    docs = retrieve("specs of TechPro", "products", mock_client, mock_embeddings)

    assert len(docs) == 1
    assert isinstance(docs[0], RetrievedDoc)
    assert docs[0].text == "TechPro X1 features 16GB RAM."
    assert docs[0].score == 0.85
    assert docs[0].source == "products"


def test_retriever_empty_when_no_points_above_threshold():
    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 384

    response = MagicMock()
    response.points = []
    mock_client.query_points.return_value = response

    docs = retrieve("unrelated query", "support", mock_client, mock_embeddings)

    assert docs == []
