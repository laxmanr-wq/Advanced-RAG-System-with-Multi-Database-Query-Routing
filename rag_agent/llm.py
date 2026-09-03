"""
RAG Agent with Database Routing - LLM configuration.

Groq via its OpenAI-compatible endpoint powers routing, generation, document
grading, query rewriting, the web fallback, and vision OCR. RAG_MODEL
overrides the default text model (used for quota management during long
evaluation runs).
"""

from __future__ import annotations

import os

from openai import OpenAI

BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Groq retired the LLaMA 3.2 vision models; no vision-capable model is
# currently served. Image uploads therefore skip the vision-LLM pass and go
# straight to LiteParse OCR. Re-add model ids here if Groq ships one again.
VISION_MODELS: list[str] = []


def get_active_model() -> str:
    """Text model used for routing, generation, grading, and rewriting."""
    return os.getenv("RAG_MODEL", "").strip() or DEFAULT_MODEL


def build_client(api_key: str) -> OpenAI:
    """Build the Groq OpenAI-compatible client."""
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def vision_models_for() -> list[str]:
    """Vision-capable models to try for image OCR."""
    return VISION_MODELS
