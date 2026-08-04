"""Quote grounding against canonical source chunks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from ke.contracts import Evidence
from ke.ids import sha256_text


class GroundingError(ValueError):
    """Raised when exact source evidence cannot be verified."""


def quote_sha256(quote: str) -> str:
    return sha256_text(quote)


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, Mapping):
        for key in ("text", "markdown", "content"):
            if isinstance(chunk.get(key), str):
                return chunk[key]
    for key in ("text", "markdown", "content"):
        value = getattr(chunk, key, None)
        if isinstance(value, str):
            return value
    raise TypeError("chunk must expose text, markdown, or content")


def _collapsed(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def verify_evidence(
    evidence: Evidence,
    chunk: Any,
    *,
    allow_whitespace_normalization: bool = False,
) -> Evidence:
    """Return a verified copy if the quotation occurs in its attributed chunk.

    Exact substring matching is the default.  The optional whitespace mode is
    useful for converters that wrap lines, but the stored quotation and its hash
    are never rewritten.
    """

    text = _chunk_text(chunk)
    found = evidence.quote in text
    if not found and allow_whitespace_normalization:
        found = _collapsed(evidence.quote) in _collapsed(text)
    if not found:
        raise GroundingError(
            f"quote for {evidence.source_id}/{evidence.chunk_id} was not found in the chunk"
        )

    expected_hash = quote_sha256(evidence.quote)
    if evidence.quote_sha256 and evidence.quote_sha256 != expected_hash:
        raise GroundingError("quote_sha256 does not match the exact stored quotation")
    return evidence.model_copy(update={"quote_sha256": expected_hash, "verified": True})


def ground_evidence(
    evidence_items: Iterable[Evidence],
    chunks: Mapping[str, Any],
    *,
    strict: bool = True,
    allow_whitespace_normalization: bool = False,
) -> list[Evidence]:
    """Verify evidence and optionally omit unsupported quotations."""

    grounded: list[Evidence] = []
    for evidence in evidence_items:
        chunk = chunks.get(evidence.chunk_id)
        if chunk is None:
            if strict:
                raise GroundingError(f"unknown evidence chunk: {evidence.chunk_id}")
            continue
        try:
            grounded.append(
                verify_evidence(
                    evidence,
                    chunk,
                    allow_whitespace_normalization=allow_whitespace_normalization,
                )
            )
        except GroundingError:
            if strict:
                raise
    return grounded


def assert_fully_grounded(items: Iterable[Any]) -> None:
    """Reject canonical objects with absent or unverified evidence."""

    for item in items:
        evidence = getattr(item, "evidence", None)
        item_id = getattr(item, "id", type(item).__name__)
        if not evidence or any(not citation.verified for citation in evidence):
            raise GroundingError(f"canonical object {item_id!r} is not fully grounded")
