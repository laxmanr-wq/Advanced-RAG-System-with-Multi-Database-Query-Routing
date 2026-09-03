"""Unit tests for rag_agent.router."""

from unittest.mock import MagicMock

from rag_agent.router import RoutingDecision, _keyword_fallback, route_query


def test_keyword_fallback_financial():
    decision = _keyword_fallback("What are the pricing options?")
    assert decision.database == "financial"
    assert "Keyword match fallback" in decision.reasoning


def test_keyword_fallback_support():
    decision = _keyword_fallback("How do I reset my password?")
    assert decision.database == "support"
    assert "Keyword match fallback" in decision.reasoning


def test_keyword_fallback_default_products():
    decision = _keyword_fallback("Tell me about TechPro specs")
    assert decision.database == "products"


def test_route_query_empty_string():
    mock_client = MagicMock()
    decision = route_query(mock_client, "")
    assert decision.database == "products"
    # Should not call Groq OpenAI API for empty query
    mock_client.chat.completions.create.assert_not_called()


def test_route_query_valid_json_response():
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = '{"database": "financial", "reasoning": "User asks about pricing"}'
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    decision = route_query(mock_client, "What is the cost of enterprise plan?")
    assert isinstance(decision, RoutingDecision)
    assert decision.database == "financial"
    assert decision.reasoning == "User asks about pricing"


def test_route_query_invalid_json_triggers_fallback():
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Invalid non-JSON response from LLM"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    decision = route_query(mock_client, "How to reset 2fa authentication")
    assert decision.database == "support"
