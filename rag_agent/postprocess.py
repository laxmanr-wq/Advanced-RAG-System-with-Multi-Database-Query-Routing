"""
Post-processing guards for generated answers.

Two deterministic, LLM-free checks applied after generation:
  * strip_citations      - remove bracketed source markers like [1] / [3]
  * numbers_grounded     - verify every number in the answer appears in the
                           retrieved context (anti-hallucination guard)
"""

from __future__ import annotations

import re

# Bracketed citation markers, e.g. [1], [3], [12]
_CITATION_RE = re.compile(r"\[\d+\]")

# Full-width citation markers emitted by GPT-OSS models, e.g. 【1†L2-L4】.
# Restricted to spans carrying a digit or the dagger separator so ordinary
# full-width bracket usage is left alone.
_FW_CITATION_RE = re.compile(r"【[^】]*?(?:†|\d)[^】]*?】")

# Any number with optional thousand separators / decimals / sign.
# The decimal part requires digits after the dot so a sentence-ending
# period ("108,000.") is not swallowed into the number. A leading hyphen
# only counts as a sign when not preceded by a digit, so ranges like
# "2018-2019" do not yield a phantom "-2019".
_NUMBER_RE = re.compile(r"(?<![\d.])-?\(?\d[\d,]*(?:\.\d+)?\)?")


def strip_citations(text: str) -> str:
    """Remove [n] / 【n†...】 source markers and tidy up leftover whitespace."""
    cleaned = _CITATION_RE.sub("", text)
    cleaned = _FW_CITATION_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)  # no space before punctuation
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def extract_numbers(text: str) -> set[str]:
    """Extract normalized numbers (commas removed, parentheses stripped)."""
    found: set[str] = set()
    for m in _NUMBER_RE.finditer(text):
        n = m.group(0).replace(",", "").strip("()")
        if n and any(ch.isdigit() for ch in n):
            found.add(n)
    return found


def _canonical(num: str) -> str:
    """Sign/parenthesis-insensitive form: '(123)', '-123' and '123' agree."""
    return num.replace(",", "").replace("-", "").strip("()")


def numbers_grounded(answer: str, context_texts: list[str]) -> bool:
    """
    True if every number in the answer can be found in the context.
    Parentheses notation (123) and -123 are treated as equivalent.
    Short integers (1-2 digits) that appear as part of larger context
    numbers are allowed (e.g. answer '2019' vs context '2019').
    """
    if not context_texts:
        return True
    ctx = "".join(context_texts).replace(",", "").replace("-", "")
    ctx = ctx.replace("(", "").replace(")", "")
    for num in extract_numbers(answer):
        if _canonical(num) not in ctx:
            return False
    return True
