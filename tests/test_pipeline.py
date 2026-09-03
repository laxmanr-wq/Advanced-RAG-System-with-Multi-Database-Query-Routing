"""Integration tests for rag_agent.pipeline orchestration."""

from unittest.mock import MagicMock, patch

from rag_agent.grading import GradingResult
from rag_agent.pipeline import PipelineResult, run_pipeline
from rag_agent.retriever import RetrievedDoc
from rag_agent.router import RoutingDecision


def _mock_chat_response(content: str) -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@patch("rag_agent.pipeline.route_query")
@patch("rag_agent.pipeline.retrieve")
def test_run_pipeline_rag_path(mock_retrieve, mock_route):
    mock_route.return_value = RoutingDecision(
        database="products", reasoning="Query asks about product features"
    )
    mock_retrieve.return_value = [
        RetrievedDoc(text="TechPro X1 cost $999.", score=0.9, source="products")
    ]

    mock_groq = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "The TechPro X1 laptop costs $999."
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_groq.chat.completions.create.return_value = mock_response

    mock_qdrant = MagicMock()
    mock_embeddings = MagicMock()

    result = run_pipeline("How much is TechPro X1?", mock_qdrant, mock_embeddings, mock_groq)

    assert isinstance(result, PipelineResult)
    assert result.used_fallback is False
    assert result.answer == "The TechPro X1 laptop costs $999."
    assert len(result.docs) == 1
    assert result.routing.database == "products"


@patch("rag_agent.pipeline.route_query")
@patch("rag_agent.pipeline.retrieve")
@patch("rag_agent.pipeline.run_fallback")
def test_run_pipeline_fallback_path(mock_fallback, mock_retrieve, mock_route):
    mock_route.return_value = RoutingDecision(
        database="support", reasoning="Query asks about password reset"
    )
    mock_retrieve.return_value = []  # No docs found above score threshold
    mock_fallback.return_value = "Follow these web search instructions to reset password."

    mock_groq = MagicMock()
    mock_qdrant = MagicMock()
    mock_embeddings = MagicMock()

    result = run_pipeline("Password reset steps", mock_qdrant, mock_embeddings, mock_groq)

    assert isinstance(result, PipelineResult)
    assert result.used_fallback is True
    assert result.answer == "Follow these web search instructions to reset password."
    assert result.docs == []
    mock_fallback.assert_called_once_with(mock_groq, "Password reset steps")


@patch("rag_agent.pipeline.route_query")
@patch("rag_agent.pipeline.retrieve")
@patch("rag_agent.pipeline.grade_documents")
@patch("rag_agent.pipeline.rewrite_query")
def test_run_pipeline_rewrites_query_after_irrelevant_grade(
    mock_rewrite, mock_grade, mock_retrieve, mock_route
):
    """First retrieval graded irrelevant -> rewrite -> second retrieval graded relevant -> generate."""
    mock_route.return_value = RoutingDecision(database="products", reasoning="Ambiguous product query")

    off_topic_doc = RetrievedDoc(text="Our return policy lasts 30 days.", score=0.6, source="products")
    on_topic_doc = RetrievedDoc(text="TechPro X1 costs $500.", score=0.9, source="products")
    mock_retrieve.side_effect = [[off_topic_doc], [on_topic_doc]]

    mock_grade.side_effect = [
        GradingResult(is_relevant=False, reasoning="Talks about returns, not pricing."),
        GradingResult(is_relevant=True, reasoning="Directly answers the price question."),
    ]
    mock_rewrite.return_value = "What is the exact price of the TechPro X1 laptop?"

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = _mock_chat_response(
        "The TechPro X1 costs $500."
    )
    mock_qdrant = MagicMock()
    mock_embeddings = MagicMock()

    result = run_pipeline("How much is the X1?", mock_qdrant, mock_embeddings, mock_groq)

    assert result.used_fallback is False
    assert result.retrieval_attempts == 2
    assert result.rewritten_query == "What is the exact price of the TechPro X1 laptop?"
    assert result.docs == [on_topic_doc]
    assert mock_retrieve.call_args_list[0].args[0] == "How much is the X1?"
    assert mock_retrieve.call_args_list[1].args[0] == "What is the exact price of the TechPro X1 laptop?"
    mock_rewrite.assert_called_once_with(mock_groq, "How much is the X1?")


@patch("rag_agent.pipeline.contextualize_query")
@patch("rag_agent.pipeline.route_query")
@patch("rag_agent.pipeline.retrieve")
@patch("rag_agent.pipeline.grade_documents")
def test_run_pipeline_routes_and_retrieves_on_the_contextualized_query(
    mock_grade, mock_retrieve, mock_route, mock_ctx
):
    """A follow-up must be resolved BEFORE routing and retrieval see it."""
    from rag_agent.memory import ConversationMemory

    memory = ConversationMemory()
    memory.add_user_message("What was total equity in 2019?")
    memory.add_assistant_message("Total equity was 1,234,567.")

    standalone = "What two components make up total equity in 2019?"
    mock_ctx.return_value = standalone
    mock_route.return_value = RoutingDecision(database="financial", reasoning="equity question")
    mock_retrieve.return_value = [
        RetrievedDoc(text="Equity comprises capital 900,000 and reserves 334,567.", score=0.9, source="financial")
    ]
    mock_grade.return_value = GradingResult(is_relevant=True, reasoning="Lists both components.")

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = _mock_chat_response(
        "Capital of 900,000 and reserves of 334,567."
    )

    result = run_pipeline(
        "What two components make up that total?",
        MagicMock(),
        MagicMock(),
        mock_groq,
        memory=memory,
    )

    # the router and retriever both see the resolved question, not the ellipsis
    assert mock_route.call_args.args[1] == standalone
    assert mock_retrieve.call_args.args[0] == standalone
    assert result.contextualized_query == standalone
    assert result.used_fallback is False
    assert any(s.step_name == "Query Contextualizer" for s in result.trace.steps)


@patch("rag_agent.pipeline.route_query")
@patch("rag_agent.pipeline.retrieve")
@patch("rag_agent.pipeline.run_fallback")
def test_run_pipeline_strips_citations_from_web_fallback(mock_fallback, mock_retrieve, mock_route):
    """The fallback model emits source markers too; they must not reach the user.

    Observed live: "total net sales of $383.3 billion for fiscal 2024【1†L1-L3】".
    """
    mock_route.return_value = RoutingDecision(database="financial", reasoning="finance query")
    mock_retrieve.return_value = []
    mock_fallback.return_value = "Net sales were $383.3 billion in fiscal 2024【1†L1-L3】."

    result = run_pipeline("What were net sales?", MagicMock(), MagicMock(), MagicMock())

    assert result.used_fallback is True
    assert "【" not in result.answer
    assert result.answer == "Net sales were $383.3 billion in fiscal 2024."


@patch("rag_agent.pipeline.contextualize_query")
@patch("rag_agent.pipeline.route_query")
@patch("rag_agent.pipeline.retrieve")
def test_run_pipeline_skips_contextualization_without_history(mock_retrieve, mock_route, mock_ctx):
    """First turn has no history, so no contextualization call and no extra latency."""
    mock_route.return_value = RoutingDecision(database="products", reasoning="product query")
    mock_retrieve.return_value = []

    with patch("rag_agent.pipeline.run_fallback", return_value="web answer"):
        result = run_pipeline("What is the X1 price?", MagicMock(), MagicMock(), MagicMock())

    mock_ctx.assert_not_called()
    assert result.contextualized_query is None
    assert not any(s.step_name == "Query Contextualizer" for s in result.trace.steps)


@patch("rag_agent.pipeline.route_query")
@patch("rag_agent.pipeline.retrieve")
@patch("rag_agent.pipeline.grade_documents")
@patch("rag_agent.pipeline.rewrite_query")
@patch("rag_agent.pipeline.run_fallback")
def test_run_pipeline_falls_back_after_exhausting_retrieval_attempts(
    mock_fallback, mock_rewrite, mock_grade, mock_retrieve, mock_route
):
    """Every attempt grades irrelevant -> exhausts MAX_RETRIEVAL_ATTEMPTS -> web fallback."""
    mock_route.return_value = RoutingDecision(database="support", reasoning="Unclear support query")

    off_topic_doc = RetrievedDoc(text="Our office hours are 9-5.", score=0.55, source="support")
    mock_retrieve.return_value = [off_topic_doc]
    mock_grade.return_value = GradingResult(is_relevant=False, reasoning="Never on topic.")
    mock_rewrite.return_value = "rewritten support question"
    mock_fallback.return_value = "Here is what the web says."

    mock_groq = MagicMock()
    mock_qdrant = MagicMock()
    mock_embeddings = MagicMock()

    result = run_pipeline("Obscure support question", mock_qdrant, mock_embeddings, mock_groq)

    assert result.used_fallback is True
    assert result.retrieval_attempts == 2
    assert result.docs == []
    assert result.answer == "Here is what the web says."
    mock_rewrite.assert_called_once()
    mock_fallback.assert_called_once_with(mock_groq, "Obscure support question")
