"""Unit tests for rag_agent.contextualize."""

from unittest.mock import MagicMock

from rag_agent.contextualize import contextualize_query
from rag_agent.memory import ConversationMemory


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


def _memory_with_history() -> ConversationMemory:
    memory = ConversationMemory()
    memory.add_user_message("What was total equity at 31 December 2019?")
    memory.add_assistant_message("Total equity was 1,234,567 at 31 December 2019.")
    return memory


def test_no_memory_skips_the_llm_entirely():
    client = MagicMock()
    result = contextualize_query(client, "What is total equity?", None)
    assert result == "What is total equity?"
    client.chat.completions.create.assert_not_called()


def test_empty_memory_skips_the_llm_entirely():
    client = MagicMock()
    result = contextualize_query(client, "What is total equity?", ConversationMemory())
    assert result == "What is total equity?"
    client.chat.completions.create.assert_not_called()


def test_resolves_elliptical_followup():
    client = _mock_client(
        '{"standalone_query": "What two components make up total equity at 31 December 2019?"}'
    )
    result = contextualize_query(client, "What two components make up that total?", _memory_with_history())
    assert result == "What two components make up total equity at 31 December 2019?"


def test_falls_back_to_original_on_unparseable_response():
    client = _mock_client("I cannot produce JSON right now.")
    result = contextualize_query(client, "And in 2018?", _memory_with_history())
    assert result == "And in 2018?"


def test_falls_back_to_original_on_llm_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network down")
    result = contextualize_query(client, "And in 2018?", _memory_with_history())
    assert result == "And in 2018?"


def test_falls_back_to_original_on_empty_rewrite():
    client = _mock_client('{"standalone_query": "   "}')
    result = contextualize_query(client, "And in 2018?", _memory_with_history())
    assert result == "And in 2018?"


def test_blank_query_is_returned_unchanged():
    client = MagicMock()
    result = contextualize_query(client, "   ", _memory_with_history())
    assert result == "   "
    client.chat.completions.create.assert_not_called()
