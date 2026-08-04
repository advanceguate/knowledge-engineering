"""Deterministic identifiers and content fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def slugify(value: str, *, fallback: str = "item", max_length: int = 64) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)[:max_length].rstrip("-")
    return slug or fallback


def canonical_json(value: Any) -> bytes:
    """Serialize a value consistently for content-addressed identifiers."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def source_id(title: str, digest: str) -> str:
    """Return ``<slug>-<sha256[:12]>`` as required by the artifact contract."""

    return f"{slugify(title)}-{digest[:12].lower()}"


def stable_id(prefix: str, label: str, payload: Any | None = None, *, size: int = 12) -> str:
    """Assign a human-readable ID after resolution using label and semantic content."""

    identity = {"label": label, "payload": payload if payload is not None else label}
    return f"{slugify(prefix)}:{slugify(label)}-{content_hash(identity)[:size]}"


def ontology_id(title: str, source_ids: list[str]) -> str:
    return stable_id("ontology", title, sorted(source_ids))


def frame_id(name: str, objective: str, ontology_ids: list[str]) -> str:
    return stable_id("frame", name, {"objective": objective, "ontology_ids": sorted(ontology_ids)})
