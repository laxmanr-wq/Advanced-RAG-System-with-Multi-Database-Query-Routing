"""
RAG Agent with Database Routing - Guardrails & Safety Policies.

Enforces input validation, prompt injection detection, and output verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Common prompt injection pattern signatures
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above)\s+(instructions|prompts)",
    r"system\s+prompt",
    r"reveal\s+(your\s+)?instructions",
    r"you\s+are\s+now\s+DAN",
    r"override\s+safety",
]


@dataclass
class GuardrailResult:
    """Outcome of a guardrail policy check."""

    passed: bool
    reason: str = ""


class InputGuardrail:
    """Validates user queries before submitting to the system."""

    @staticmethod
    def validate(query: str) -> GuardrailResult:
        query_strip = query.strip()
        if not query_strip:
            return GuardrailResult(passed=False, reason="Query is empty.")

        # Check for prompt injection signatures
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, query_strip, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    reason="Potential prompt injection detected (policy violation).",
                )

        return GuardrailResult(passed=True, reason="Input check passed.")


class OutputGuardrail:
    """Validates generated outputs before sending to user."""

    @staticmethod
    def validate(answer: str) -> GuardrailResult:
        if not answer or not answer.strip():
            return GuardrailResult(passed=False, reason="Generated output is empty.")
        return GuardrailResult(passed=True, reason="Output check passed.")
