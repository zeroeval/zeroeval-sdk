from __future__ import annotations

import hashlib


def _normalize_newlines(text: str) -> str:
    # Convert CRLF and CR to LF
    if "\r" not in text:
        return text
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strip_trailing_whitespace(text: str) -> str:
    # Remove trailing whitespace on each line
    return "\n".join(line.rstrip() for line in text.split("\n"))


def normalize_prompt_text(text: str) -> str:
    """
    Normalize prompt content prior to hashing.

    Rules:
    - Convert CRLF/CR to LF
    - Strip trailing whitespace on each line
    - Strip leading/trailing whitespace overall
    - Do not modify {{variable}} tokens
    """
    if not isinstance(text, str):
        text = str(text)
    normalized = _normalize_newlines(text)
    normalized = _strip_trailing_whitespace(normalized)
    normalized = normalized.strip()
    return normalized


def sha256_hex(text: str) -> str:
    """Return lowercase hex SHA-256 of the normalized text."""
    normalized = normalize_prompt_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


