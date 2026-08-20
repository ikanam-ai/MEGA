from __future__ import annotations

MAX_CHARS = 1500


def parse(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [text[:MAX_CHARS].strip()]


def parse_batch(texts: list[str]) -> list[list[str]]:
    return [parse(t) for t in texts]
