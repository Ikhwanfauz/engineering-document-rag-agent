"""Deterministic detection of basic prompt-injection attempts."""

from __future__ import annotations

import re


class PromptInjectionError(ValueError):
    """Raised when a user question contains a suspected injection attempt."""


PROMPT_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget)\b.{0,80}"
            r"\b(?:previous|prior|above|system|developer)\b.{0,40}"
            r"\b(?:instruction|prompt|rule)s?\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "security_bypass",
        re.compile(
            r"\b(?:override|bypass|disable|circumvent)\b.{0,60}"
            r"\b(?:system|developer|safety|security|guardrail)\b.{0,40}"
            r"\b(?:instruction|prompt|rule|filter)s?\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "prompt_extraction",
        re.compile(
            r"\b(?:reveal|show|print|repeat|expose|display)\b.{0,60}"
            r"\b(?:system|developer)\b.{0,20}"
            r"\b(?:prompt|message|instruction)s?\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"\b(?:you are now|act as|pretend to be|assume the role of)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "instruction_replacement",
        re.compile(
            r"\b(?:follow|obey|execute)\b.{0,50}"
            r"\b(?:these|the following|my new)\b.{0,30}"
            r"\b(?:instruction|command|rule)s?\b.{0,40}"
            r"\b(?:instead|over the previous|above all)\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "forged_role_marker",
        re.compile(
            r"(?:<\s*/?\s*(?:system|developer|assistant)\s*>|"
            r"\[\s*(?:system|developer|assistant)\s*\])",
            flags=re.IGNORECASE,
        ),
    ),
)


def detect_prompt_injection(text: str) -> tuple[str, ...]:
    """Return the names of injection patterns detected in text."""
    if not text.strip():
        return ()

    return tuple(
        pattern_name
        for pattern_name, pattern in PROMPT_INJECTION_PATTERNS
        if pattern.search(text)
    )


def contains_prompt_injection(text: str) -> bool:
    """Return whether text matches any basic injection pattern."""
    return bool(detect_prompt_injection(text))


def validate_question_guardrail(question: str) -> None:
    """Reject a user question containing a suspected injection attempt."""
    detected_patterns = detect_prompt_injection(question)

    if detected_patterns:
        raise PromptInjectionError(
            "Question contains a suspected prompt-injection instruction"
        )


def remove_suspicious_document_texts(
    texts: tuple[str, ...],
) -> tuple[str, ...]:
    """Return document texts that do not contain suspected injection attempts."""
    return tuple(
        text
        for text in texts
        if not contains_prompt_injection(text)
    )