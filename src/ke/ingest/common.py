"""Deterministic helpers for document ingestion."""

from __future__ import annotations

import json
import mimetypes
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ke import ids as shared_ids

from .models import SourceFingerprint

CANONICAL_SCHEMA_NAME = "KECanonicalDocument"
CANONICAL_SCHEMA_VERSION = "1.0"

_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".epub": "application/epub+zip",
    ".html": "text/html",
    ".htm": "text/html",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".text": "text/plain",
}
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    return shared_ids.sha256_file(path, block_size=block_size)


def sha256_bytes(value: bytes) -> str:
    return shared_ids.sha256_bytes(value)


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def slugify(value: str, *, default: str = "source") -> str:
    """Create the stable ASCII slug used in source IDs."""

    return shared_ids.slugify(value, fallback=default)


def media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _MEDIA_TYPES:
        return _MEDIA_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def fingerprint_source(path: str | Path, *, title: str | None = None) -> SourceFingerprint:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"Source must be a file: {source}")

    resolved = source.resolve()
    digest = sha256_file(resolved)
    resolved_title = (title or resolved.stem).strip()
    if not resolved_title:
        resolved_title = resolved.stem or "Source"
    slug = slugify(resolved_title)
    extension = resolved.suffix.lower()
    return SourceFingerprint(
        path=resolved,
        source_id=shared_ids.source_id(resolved_title, digest),
        slug=slug,
        title=resolved_title,
        extension=extension,
        media_type=media_type_for(resolved),
        content_sha256=digest,
        size_bytes=resolved.stat().st_size,
    )


def read_text_source(path: Path) -> str:
    """Read a text source without normalizing its line endings or trailing newline."""

    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    # Text ingestion is preferable to a hard failure for legacy books.  The
    # replacement characters remain visible and therefore auditable.
    return raw.decode("utf-8", errors="replace")


def markdown_blocks(markdown: str) -> list[dict[str, Any]]:
    """Split Markdown into ordered, heading-aware canonical blocks.

    This is a fallback for formats without Docling layout objects.  It keeps
    fenced code and Markdown tables together so later chunking does not tear
    their structure apart.
    """

    lines = markdown.splitlines()
    blocks: list[dict[str, Any]] = []
    headings: list[str] = []
    buffer: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush(label: str | None = None) -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip("\n")
        buffer = []
        if not text.strip():
            return
        detected = label or _classify_markdown_block(text)
        blocks.append(
            {
                "label": detected,
                "text": text,
                "heading_path": list(headings),
                "page_start": None,
                "page_end": None,
            }
        )

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not in_fence:
                flush()
                in_fence = True
                fence_marker = marker
            buffer.append(line)
            if in_fence and marker == fence_marker and len(buffer) > 1:
                in_fence = False
                flush("code")
            continue

        if in_fence:
            buffer.append(line)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            label = heading.group(2).strip()
            headings[level - 1 :] = [label]
            blocks.append(
                {
                    "label": "section_header",
                    "text": label,
                    "level": level,
                    "heading_path": list(headings),
                    "page_start": None,
                    "page_end": None,
                }
            )
            continue

        if not line.strip():
            flush()
            continue
        buffer.append(line)

    flush("code" if in_fence else None)
    return blocks


def _classify_markdown_block(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and _TABLE_SEPARATOR_RE.match(lines[1]):
        return "table"
    first = lines[0].lstrip() if lines else ""
    if first.startswith(("- ", "* ", "+ ")) or re.match(r"\d+[.)]\s", first):
        return "list_item"
    return "text"


def canonical_document_from_markdown(
    markdown: str,
    *,
    title: str,
    filename: str,
    media_type: str,
    content_sha256: str,
    converter: str,
    supplied_blocks: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a Docling-shaped canonical document for non-Docling inputs."""

    source_blocks = (
        list(supplied_blocks) if supplied_blocks is not None else markdown_blocks(markdown)
    )
    texts: list[dict[str, Any]] = []
    body_children: list[dict[str, str]] = []
    for index, block in enumerate(source_blocks):
        ref = f"#/texts/{index}"
        page_start = _positive_int_or_none(block.get("page_start"))
        page_end = _positive_int_or_none(block.get("page_end")) or page_start
        prov: list[dict[str, Any]] = []
        if page_start is not None:
            prov.append({"page_no": page_start})
            if page_end is not None and page_end != page_start:
                prov.append({"page_no": page_end})
        item: dict[str, Any] = {
            "self_ref": ref,
            "parent": {"$ref": "#/body"},
            "children": [],
            "label": str(block.get("label") or "text"),
            "text": str(block.get("text") or ""),
            "orig": str(block.get("text") or ""),
            "prov": prov,
            "heading_path": [str(value) for value in block.get("heading_path", [])],
        }
        if block.get("level") is not None:
            item["level"] = int(block["level"])
        for optional_key in ("chapter", "href"):
            if block.get(optional_key) is not None:
                item[optional_key] = block[optional_key]
        texts.append(item)
        body_children.append({"$ref": ref})

    return {
        "schema_name": CANONICAL_SCHEMA_NAME,
        "version": CANONICAL_SCHEMA_VERSION,
        "name": title,
        "origin": {
            "filename": filename,
            "mimetype": media_type,
            "binary_hash": content_sha256,
        },
        "body": {"self_ref": "#/body", "children": body_children},
        "furniture": {"self_ref": "#/furniture", "children": []},
        "texts": texts,
        "tables": [],
        "pictures": [],
        "key_value_items": [],
        "form_items": [],
        "pages": {},
        "ke": {"converter": converter, "canonical_schema_version": CANONICAL_SCHEMA_VERSION},
    }


def json_dumps(value: Any, *, indent: int | None = 2) -> str:
    """Canonical JSON used by cache keys and on-disk artifacts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        default=_json_default,
    )


def jsonable(value: Any) -> Any:
    """Normalize operator metadata into stable JSON-compatible values."""

    return json.loads(json_dumps(value, indent=None))


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 1 else None


__all__ = [
    "CANONICAL_SCHEMA_NAME",
    "CANONICAL_SCHEMA_VERSION",
    "canonical_document_from_markdown",
    "fingerprint_source",
    "json_dumps",
    "jsonable",
    "markdown_blocks",
    "media_type_for",
    "read_text_source",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "slugify",
]
