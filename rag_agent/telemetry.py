"""
RAG Agent with Database Routing - Telemetry & Execution Tracing.

Captures per-phase latency breakdown, execution steps, and token estimations
for full system observability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StepTrace:
    step_name: str
    latency_ms: float
    details: str = ""


@dataclass
class ExecutionTrace:
    """Full execution trace for a single query pipeline run."""

    steps: list[StepTrace] = field(default_factory=list)
    total_latency_ms: float = 0.0
    estimated_tokens: int = 0

    def add_step(self, step_name: str, latency_ms: float, details: str = "") -> None:
        self.steps.append(StepTrace(step_name=step_name, latency_ms=latency_ms, details=details))

    def calculate_total_latency(self) -> float:

        self.total_latency_ms = sum(s.latency_ms for s in self.steps)
        return self.total_latency_ms


class Timer:
    """Utility timer context manager."""

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end = time.perf_counter()
        self.elapsed_ms = (self.end - self.start) * 1000.0
