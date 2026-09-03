"""
RAG Agent with Database Routing - Faithfulness & Groundedness Evaluator.

Computes a deterministic groundedness score (0.0 to 1.0) measuring how well
the generated answer is supported by the retrieved context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Evaluation metrics for a generated RAG answer."""

    groundedness_score: float  # 0.0 to 1.0
    is_faithful: bool
    status_label: str


class FaithfulnessEvaluator:
    """Evaluates answer groundedness against retrieved document context."""

    @staticmethod
    def evaluate(answer: str, context_docs: list[str]) -> EvaluationResult:
        if not context_docs or not answer.strip():
            # If web fallback or no context was used
            return EvaluationResult(
                groundedness_score=0.70,
                is_faithful=True,
                status_label="Web Fallback (External Source)",
            )

        answer_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))
        if not answer_words:
            return EvaluationResult(
                groundedness_score=1.0,
                is_faithful=True,
                status_label="Grounded (High Confidence)",
            )

        context_text = " ".join(context_docs).lower()
        matched_words = [w for w in answer_words if w in context_text]

        overlap_ratio = len(matched_words) / len(answer_words)
        groundedness_score = round(min(1.0, overlap_ratio * 1.15), 2)
        is_faithful = groundedness_score >= 0.50

        status_label = (
            "Grounded (High Confidence)"
            if groundedness_score >= 0.70
            else "Partially Grounded"
            if is_faithful
            else "Low Groundedness Warning"
        )

        return EvaluationResult(
            groundedness_score=groundedness_score,
            is_faithful=is_faithful,
            status_label=status_label,
        )
