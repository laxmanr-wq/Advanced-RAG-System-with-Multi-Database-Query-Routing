"""Unit tests for rag_agent.query_rewrite."""

from unittest.mock import MagicMock

from rag_agent.query_rewrite import rewrite_query


def _mock_client(content: str) -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    client = MagicMock()
    client.chat.completions.create.return_value = mock_response
    return client


def test_rewrite_query_returns_llm_reformulation():
    client = _mock_client('{"rewritten_query": "What is the exact list price of the TechPro X1?"}')
    result = rewrite_query(client, "How much is the X1?")
    assert result == "What is the exact list price of the TechPro X1?"


def test_rewrite_query_falls_back_on_unparseable_response():
    client = _mock_client("not json at all")
    result = rewrite_query(client, "How much is the X1?")
    assert result.startswith("How much is the X1?")


def test_rewrite_query_falls_back_on_llm_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network error")
    result = rewrite_query(client, "How much is the X1?")
    assert result.startswith("How much is the X1?")


def test_rewrite_query_falls_back_on_empty_rewrite():
    client = _mock_client('{"rewritten_query": ""}')
    result = rewrite_query(client, "How much is the X1?")
    assert result.startswith("How much is the X1?")
