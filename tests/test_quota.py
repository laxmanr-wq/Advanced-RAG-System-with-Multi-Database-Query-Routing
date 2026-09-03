"""Tests for the Groq quota-wait hint parser."""

from rag_agent.quota import quota_wait_seconds


def test_parses_minutes_and_seconds_hint():
    msg = "Rate limit reached ... Please try again in 4m22.656s. Need more tokens?"
    assert quota_wait_seconds(msg) == 262.656


def test_parses_minutes_and_seconds_with_space():
    msg = "Please try again in 2m 5s."
    assert quota_wait_seconds(msg) == 125.0


def test_parses_seconds_only_hint():
    msg = "Please try again in 47.5s."
    assert quota_wait_seconds(msg) == 47.5


def test_returns_none_without_hint():
    assert quota_wait_seconds("Some unrelated error message") is None
