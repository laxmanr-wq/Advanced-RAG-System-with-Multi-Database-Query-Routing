"""Unit tests for rag_agent.grading."""

from unittest.mock import MagicMock

from rag_agent.grading import GradingResult, grade_documents


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


def test_grade_documents_no_context_is_not_relevant():
    client = MagicMock()
    result = grade_documents(client, "What is the price?", [])
    assert isinstance(result, GradingResult)
    assert result.is_relevant is False
    client.chat.completions.create.assert_not_called()


def test_grade_documents_relevant_json_response():
    client = _mock_client('{"relevant": "yes", "reasoning": "Directly answers the question."}')
    result = grade_documents(client, "What is the price?", ["The price is $500."])
    assert result.is_relevant is True
    assert "Directly answers" in result.reasoning


def test_grade_documents_not_relevant_json_response():
    client = _mock_client('{"relevant": "no", "reasoning": "Talks about a different topic."}')
    result = grade_documents(client, "What is the price?", ["Our return policy is 30 days."])
    assert result.is_relevant is False


def test_grade_documents_fails_open_on_unparseable_response():
    client = _mock_client("Sorry, I cannot help with JSON right now.")
    result = grade_documents(client, "What is the price?", ["The price is $500."])
    assert result.is_relevant is True
    assert "skipped" in result.reasoning.lower()


def test_grade_documents_fails_open_on_llm_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network error")
    result = grade_documents(client, "What is the price?", ["The price is $500."])
    assert result.is_relevant is True
