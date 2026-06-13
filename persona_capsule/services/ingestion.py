"""Message ingestion, redaction, and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{2,4}\)|\d{2,4})[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"@[A-Za-z0-9_]{2,32}")


@dataclass
class IngestionResult:
    messages: list[str]
    redaction_count: int
    warnings: list[str]


def split_messages(raw_text: str) -> list[str]:
    chunks = re.split(r"\n\s*\n|\n(?=\S)", raw_text.strip())
    cleaned = [re.sub(r"\s+", " ", c.strip()) for c in chunks if c.strip()]
    return cleaned


def redact_text(text: str) -> tuple[str, int]:
    count = 0
    for pattern in (EMAIL_RE, PHONE_RE, URL_RE, HANDLE_RE):
        matches = pattern.findall(text)
        count += len(matches)
        text = pattern.sub("[redacted]", text)
    return text, count


def ingest_messages(raw_text: str, min_messages: int = 5) -> IngestionResult:
    messages = split_messages(raw_text)
    warnings: list[str] = []
    if len(messages) < min_messages:
        warnings.append(
            f"Only {len(messages)} messages found; at least {min_messages} recommended for stable steering."
        )
    redacted: list[str] = []
    total_redactions = 0
    for msg in messages:
        text, n = redact_text(msg)
        redacted.append(text)
        total_redactions += n
    unique = list(dict.fromkeys(redacted))
    if len(unique) < len(redacted):
        warnings.append("Duplicate messages were removed.")
    if total_redactions:
        warnings.append(f"Redacted {total_redactions} sensitive token(s).")
    return IngestionResult(messages=unique, redaction_count=total_redactions, warnings=warnings)
