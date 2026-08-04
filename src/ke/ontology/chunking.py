"""Layout-aware, provenance-preserving document chunking."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHUNK_SCHEMA_VERSION = "1.0"
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_HEADING_LABELS = {"section_header", "title", "heading", "subtitle"}
_TABLE_LABELS = {"table", "document_index"}


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A stable evidence unit with source and layout provenance."""

    chunk_id: str
    source_id: str
    ordinal: int
    text: str
    heading_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0
    content_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHUNK_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "ordinal": self.ordinal,
            "text": self.text,
            "heading_path": list(self.heading_path),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "token_count": self.token_count,
            "content_sha256": self.content_sha256,
            "metadata": dict(self.metadata),
        }

    def model_dump(self, **_: Any) -> dict[str, Any]:
        """Pydantic-style compatibility for generic artifact writers."""

        return self.to_dict()

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(slots=True)
class _Block:
    text: str
    heading_path: list[str]
    pages: list[int]
    label: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_document(
    document: Any = None,
    *,
    source_id: str | None = None,
    markdown: str | None = None,
    max_tokens: int = 1800,
    overlap_tokens: int = 150,
    preserve_headings: bool = True,
    preserve_tables: bool = True,
    chunker: Any | None = None,
    prefer_hybrid: bool = True,
) -> list[DocumentChunk]:
    """Chunk a canonical/Docling document while retaining exact locations.

    ``document`` may be a Docling document, exported JSON dictionary, canonical
    JSON path, source-bundle directory, or ``ConvertedDocument``.  If a live
    Docling document is available, ``HybridChunker`` is preferred; otherwise a
    deterministic local layout chunker is used.
    """

    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    canonical, docling_document, markdown_text, derived_source_id = _coerce_document(
        document, markdown=markdown
    )
    resolved_source_id = source_id or derived_source_id
    if not resolved_source_id:
        raise ValueError("source_id is required when it cannot be derived from a bundle")

    active_chunker = chunker
    if (
        prefer_hybrid
        and active_chunker is None
        and docling_document is not None
        and _looks_like_docling_document(docling_document)
    ):
        active_chunker = _make_hybrid_chunker(max_tokens=max_tokens)
    if prefer_hybrid and active_chunker is not None:
        target_document = docling_document if docling_document is not None else canonical
        try:
            chunks = _chunk_with_hybrid(active_chunker, target_document, resolved_source_id)
        except Exception:
            if chunker is not None:
                raise
        else:
            if chunks:
                return chunks

    blocks = _blocks_from_canonical(canonical) if canonical else []
    if not blocks and markdown_text is not None:
        blocks = _blocks_from_markdown(markdown_text)
    if not blocks:
        return []
    return _chunk_blocks(
        blocks,
        source_id=resolved_source_id,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        preserve_headings=preserve_headings,
        preserve_tables=preserve_tables,
    )


def construct_provenance_chunks(*args: Any, **kwargs: Any) -> list[DocumentChunk]:
    """Named pipeline-step alias for :func:`chunk_document`."""

    return chunk_document(*args, **kwargs)


def write_chunks_jsonl(chunks: Iterable[DocumentChunk], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(
                json.dumps(
                    chunk.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")
    return output


def load_chunks_jsonl(path_or_bundle: Any) -> list[DocumentChunk]:
    """Load chunks from a JSONL path, source-bundle path, or SourceBundle."""

    if hasattr(path_or_bundle, "chunks_path"):
        path = Path(path_or_bundle.chunks_path)
    else:
        path = Path(path_or_bundle)
        if path.is_dir():
            path = path / "chunks.jsonl"
    chunks: list[DocumentChunk] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid chunk JSON at {path}:{line_number}: {exc}") from exc
            chunks.append(_chunk_from_dict(value, path=path, line_number=line_number))
    return chunks


def _chunk_from_dict(value: Any, *, path: Path, line_number: int) -> DocumentChunk:
    if not isinstance(value, dict):
        raise ValueError(f"Chunk at {path}:{line_number} is not a JSON object")
    required = ("chunk_id", "source_id", "ordinal", "text")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"Chunk at {path}:{line_number} is missing {', '.join(missing)}")
    text = str(value["text"])
    digest = str(value.get("content_sha256") or _text_sha256(text))
    return DocumentChunk(
        chunk_id=str(value["chunk_id"]),
        source_id=str(value["source_id"]),
        ordinal=int(value["ordinal"]),
        text=text,
        heading_path=[str(item) for item in value.get("heading_path", [])],
        page_start=_page_number(value.get("page_start")),
        page_end=_page_number(value.get("page_end")),
        token_count=int(value.get("token_count") or _token_count(text)),
        content_sha256=digest,
        metadata=dict(value.get("metadata") or {}),
    )


def _coerce_document(
    document: Any, *, markdown: str | None
) -> tuple[dict[str, Any] | None, Any | None, str | None, str | None]:
    canonical: dict[str, Any] | None = None
    docling_document: Any | None = None
    derived_source_id: str | None = None
    markdown_text = markdown

    if document is None:
        return canonical, docling_document, markdown_text, derived_source_id

    if hasattr(document, "canonical_path") and hasattr(document, "source_id"):
        canonical_path = Path(document.canonical_path)
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        docling_document = _load_docling_document(canonical_path, canonical)
        markdown_path = getattr(document, "markdown_path", None)
        if markdown_text is None and markdown_path is not None:
            markdown_text = Path(markdown_path).read_text(encoding="utf-8")
        derived_source_id = str(document.source_id)
        return canonical, docling_document, markdown_text, derived_source_id

    if hasattr(document, "canonical_document"):
        value = document.canonical_document
        canonical = value if isinstance(value, dict) else None
        docling_document = getattr(document, "docling_document", None)
        if markdown_text is None:
            markdown_text = getattr(document, "markdown", None)
        return canonical, docling_document, markdown_text, derived_source_id

    if isinstance(document, Mapping):
        canonical = dict(document)
        derived_source_id = _source_id_from_canonical(canonical)
        return canonical, None, markdown_text, derived_source_id

    if isinstance(document, (str, Path)):
        path = Path(document)
        if path.is_dir():
            manifest_path = path / "manifest.json"
            canonical_path = path / "document.docling.json"
            markdown_path = path / "document.md"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                derived_source_id = manifest.get("source_id")
            if canonical_path.is_file():
                canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
                docling_document = _load_docling_document(canonical_path, canonical)
            if markdown_text is None and markdown_path.is_file():
                markdown_text = markdown_path.read_text(encoding="utf-8")
            return canonical, docling_document, markdown_text, derived_source_id
        if path.suffix.lower() == ".json":
            canonical = json.loads(path.read_text(encoding="utf-8"))
            derived_source_id = _source_id_from_canonical(canonical)
            docling_document = _load_docling_document(path, canonical)
        else:
            if markdown_text is None:
                markdown_text = path.read_text(encoding="utf-8")
        return canonical, docling_document, markdown_text, derived_source_id

    # Treat any other object as a live Docling document.
    docling_document = document
    exporter = getattr(document, "export_to_dict", None)
    if callable(exporter):
        try:
            value = exporter()
            canonical = value if isinstance(value, dict) else None
        except Exception:
            pass
    return canonical, docling_document, markdown_text, derived_source_id


def _source_id_from_canonical(canonical: dict[str, Any]) -> str | None:
    ke = canonical.get("ke")
    if isinstance(ke, dict) and ke.get("source_id"):
        return str(ke["source_id"])
    return None


def _load_docling_document(path: Path, canonical: dict[str, Any]) -> Any | None:
    if canonical.get("schema_name") != "DoclingDocument":
        return None
    try:
        from docling_core.types.doc import DoclingDocument
    except ImportError:
        return None
    loader = getattr(DoclingDocument, "load_from_json", None)
    if callable(loader):
        for call in (
            lambda: loader(path),
            lambda: loader(filename=path),
            lambda: loader(path.read_text(encoding="utf-8")),
        ):
            try:
                return call()
            except Exception:
                pass
    validator = getattr(DoclingDocument, "model_validate", None)
    if callable(validator):
        try:
            return validator(canonical)
        except Exception:
            return None
    return None


def _make_hybrid_chunker(*, max_tokens: int) -> Any | None:
    try:
        from docling.chunking import HybridChunker
    except ImportError:
        return None
    for factory in (
        lambda: _hybrid_chunker_with_openai_tokens(HybridChunker, max_tokens),
        lambda: HybridChunker(max_tokens=max_tokens),
        lambda: HybridChunker(),
    ):
        try:
            return factory()
        except Exception:
            continue
    return None


def _hybrid_chunker_with_openai_tokens(chunker_type: Any, max_tokens: int) -> Any:
    import tiktoken
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=max_tokens,
    )
    return chunker_type(tokenizer=tokenizer)


def _looks_like_docling_document(document: Any) -> bool:
    module_name = type(document).__module__
    return module_name.startswith("docling_core.") or (
        callable(getattr(document, "iterate_items", None))
        and callable(getattr(document, "export_to_dict", None))
    )


def _chunk_with_hybrid(chunker: Any, document: Any, source_id: str) -> list[DocumentChunk]:
    producer = getattr(chunker, "chunk", None)
    if not callable(producer):
        if callable(chunker):
            producer = chunker
        else:
            raise TypeError("chunker must be callable or expose chunk(document)")

    generated: Iterable[Any] | None = None
    errors: list[TypeError] = []
    for invocation in (
        lambda: producer(dl_doc=document),
        lambda: producer(document=document),
        lambda: producer(document),
    ):
        try:
            generated = invocation()
            break
        except TypeError as exc:
            errors.append(exc)
    if generated is None:
        raise errors[-1] if errors else RuntimeError("HybridChunker returned no iterable")

    output: list[DocumentChunk] = []
    for raw_chunk in generated:
        text = _hybrid_chunk_text(chunker, raw_chunk)
        if not text.strip():
            continue
        headings, pages, metadata = _hybrid_metadata(raw_chunk)
        actual_ordinal = len(output) + 1
        output.append(
            _make_chunk(
                source_id=source_id,
                ordinal=actual_ordinal,
                text=text,
                heading_path=headings,
                pages=pages,
                metadata={"chunker": "docling_hybrid", **metadata},
            )
        )
    return output


def _hybrid_chunk_text(chunker: Any, chunk: Any) -> str:
    serializer = getattr(chunker, "serialize", None)
    if callable(serializer):
        for invocation in (lambda: serializer(chunk=chunk), lambda: serializer(chunk)):
            try:
                value = invocation()
            except (TypeError, AttributeError):
                continue
            if isinstance(value, str):
                return value
    if isinstance(chunk, Mapping):
        value = chunk.get("text")
    else:
        value = getattr(chunk, "text", None)
    return str(value or "")


def _hybrid_metadata(chunk: Any) -> tuple[list[str], list[int], dict[str, Any]]:
    meta = chunk.get("meta") if isinstance(chunk, Mapping) else getattr(chunk, "meta", None)
    headings_value = (
        meta.get("headings", []) if isinstance(meta, Mapping) else getattr(meta, "headings", [])
    )
    headings = [str(value) for value in headings_value or [] if str(value).strip()]
    doc_items = (
        meta.get("doc_items", [])
        if isinstance(meta, Mapping)
        else getattr(meta, "doc_items", [])
        if meta is not None
        else []
    )
    pages: list[int] = []
    labels: list[str] = []
    for item in doc_items or []:
        label = item.get("label") if isinstance(item, Mapping) else getattr(item, "label", None)
        if label is not None:
            labels.append(str(getattr(label, "value", label)))
        prov = item.get("prov", []) if isinstance(item, Mapping) else getattr(item, "prov", [])
        pages.extend(_pages_from_prov(prov))
    metadata: dict[str, Any] = {}
    if labels:
        metadata["labels"] = _ordered_unique(labels)
    return headings, _ordered_unique(pages), metadata


def _blocks_from_canonical(canonical: dict[str, Any]) -> list[_Block]:
    ordered_items = list(_ordered_canonical_items(canonical))
    blocks: list[_Block] = []
    heading_stack: list[str] = []
    for item in ordered_items:
        label_value = item.get("label", "text")
        label = str(getattr(label_value, "value", label_value)).lower()
        text = _item_text(item, label)
        if not text.strip():
            continue
        supplied_headings = item.get("heading_path")
        if isinstance(supplied_headings, Sequence) and not isinstance(supplied_headings, str):
            headings = [str(value) for value in supplied_headings if str(value).strip()]
        else:
            headings = list(heading_stack)

        if label in _HEADING_LABELS:
            level = _heading_level(item, default=len(heading_stack) + 1)
            heading_stack[level - 1 :] = [text.strip()]
            headings = list(heading_stack)

        pages = _pages_from_prov(item.get("prov", []))
        metadata: dict[str, Any] = {}
        for key in ("chapter", "href"):
            if item.get(key) is not None:
                metadata[key] = item[key]
        blocks.append(
            _Block(
                text=text,
                heading_path=headings,
                pages=pages,
                label=label,
                metadata=metadata,
            )
        )
    return blocks


def _ordered_canonical_items(canonical: dict[str, Any]) -> Iterator[dict[str, Any]]:
    body = canonical.get("body")
    seen_refs: set[str] = set()

    def walk(value: Any) -> Iterator[dict[str, Any]]:
        if not isinstance(value, dict):
            return
        reference = value.get("$ref")
        if isinstance(reference, str):
            if reference in seen_refs:
                return
            seen_refs.add(reference)
            resolved = _resolve_json_pointer(canonical, reference)
            if isinstance(resolved, dict):
                yield from walk(resolved)
            return
        label = value.get("label")
        if label is not None or "text" in value or "data" in value:
            yield value
        for child in value.get("children", []) or []:
            yield from walk(child)

    if isinstance(body, dict):
        yield from walk(body)
        if seen_refs:
            return
    for collection_name in ("texts", "tables", "pictures", "key_value_items", "form_items"):
        for item in canonical.get(collection_name, []) or []:
            if isinstance(item, dict):
                yield item


def _resolve_json_pointer(document: dict[str, Any], reference: str) -> Any:
    if not reference.startswith("#/"):
        return None
    current: Any = document
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        try:
            current = current[int(part)] if isinstance(current, list) else current[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    return current


def _item_text(item: dict[str, Any], label: str) -> str:
    for key in ("text", "orig", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if label in _TABLE_LABELS or "data" in item:
        return _table_to_markdown(item.get("data"))
    captions = item.get("captions")
    if isinstance(captions, list):
        return "\n".join(str(value) for value in captions if str(value).strip())
    return ""


def _table_to_markdown(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    grid = data.get("grid")
    if isinstance(grid, list) and grid:
        rows = [[_cell_text(cell) for cell in row] for row in grid if isinstance(row, list)]
    else:
        cells = data.get("table_cells") or data.get("cells") or []
        if not isinstance(cells, list) or not cells:
            return ""
        max_row = max((_cell_index(cell, "start_row_offset_idx") for cell in cells), default=-1)
        max_col = max((_cell_index(cell, "start_col_offset_idx") for cell in cells), default=-1)
        if max_row < 0 or max_col < 0:
            return "\n".join(_cell_text(cell) for cell in cells if _cell_text(cell))
        rows = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        for cell in cells:
            row = _cell_index(cell, "start_row_offset_idx")
            column = _cell_index(cell, "start_col_offset_idx")
            if row >= 0 and column >= 0:
                rows[row][column] = _cell_text(cell)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    rendered = [
        "| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |" for row in normalized
    ]
    rendered.insert(1, "| " + " | ".join("---" for _ in range(width)) + " |")
    return "\n".join(rendered)


def _cell_index(cell: Any, key: str) -> int:
    if not isinstance(cell, Mapping):
        return -1
    try:
        return int(cell.get(key, -1))
    except (TypeError, ValueError):
        return -1


def _cell_text(cell: Any) -> str:
    if isinstance(cell, Mapping):
        for key in ("text", "content", "value"):
            if cell.get(key) is not None:
                return str(cell[key])
    return str(cell) if cell is not None else ""


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _blocks_from_markdown(markdown: str) -> list[_Block]:
    # Import lazily to avoid an ontology -> ingest import cycle at module load.
    from ke.ingest.common import markdown_blocks

    return [
        _Block(
            text=str(block.get("text") or ""),
            heading_path=[str(value) for value in block.get("heading_path", [])],
            pages=[],
            label=str(block.get("label") or "text"),
        )
        for block in markdown_blocks(markdown)
        if str(block.get("text") or "").strip()
    ]


def _chunk_blocks(
    blocks: list[_Block],
    *,
    source_id: str,
    max_tokens: int,
    overlap_tokens: int,
    preserve_headings: bool,
    preserve_tables: bool,
) -> list[DocumentChunk]:
    groups: list[list[_Block]] = []
    current: list[_Block] = []
    current_heading: tuple[str, ...] | None = None
    for block in blocks:
        heading_key = tuple(block.heading_path) if preserve_headings else ()
        if current and heading_key != current_heading:
            groups.append(current)
            current = []
        current_heading = heading_key
        current.append(block)
    if current:
        groups.append(current)

    output: list[DocumentChunk] = []
    for group in groups:
        pieces: list[_Block] = []
        for block in group:
            if preserve_tables and block.label in _TABLE_LABELS:
                pieces.append(block)
            else:
                pieces.extend(_split_block(block, max_tokens))
        pending: list[_Block] = []
        pending_tokens = 0
        for piece in pieces:
            piece_tokens = _token_count(piece.text)
            if pending and pending_tokens + piece_tokens > max_tokens:
                output.append(_emit_blocks(pending, source_id, len(output) + 1))
                pending = _overlap_tail(pending, overlap_tokens)
                pending_tokens = sum(_token_count(item.text) for item in pending)
                room = max_tokens - piece_tokens
                if room < pending_tokens:
                    pending = _overlap_tail(pending, max(0, room))
                    pending_tokens = sum(_token_count(item.text) for item in pending)
            if piece_tokens > max_tokens and preserve_tables and piece.label in _TABLE_LABELS:
                if pending:
                    output.append(_emit_blocks(pending, source_id, len(output) + 1))
                    pending = []
                    pending_tokens = 0
                output.append(_emit_blocks([piece], source_id, len(output) + 1))
                continue
            pending.append(piece)
            pending_tokens += piece_tokens
        if pending:
            output.append(_emit_blocks(pending, source_id, len(output) + 1))
    return output


def _split_block(block: _Block, max_tokens: int) -> list[_Block]:
    matches = list(_TOKEN_RE.finditer(block.text))
    if len(matches) <= max_tokens:
        return [block]
    pieces: list[_Block] = []
    for start_index in range(0, len(matches), max_tokens):
        end_index = min(start_index + max_tokens, len(matches))
        char_start = 0 if start_index == 0 else matches[start_index].start()
        char_end = len(block.text) if end_index == len(matches) else matches[end_index].start()
        text = block.text[char_start:char_end].strip()
        if text:
            pieces.append(
                _Block(
                    text=text,
                    heading_path=list(block.heading_path),
                    pages=list(block.pages),
                    label=block.label,
                    metadata=dict(block.metadata),
                )
            )
    return pieces


def _overlap_tail(blocks: list[_Block], budget: int) -> list[_Block]:
    if budget <= 0 or not blocks:
        return []
    text = "\n\n".join(block.text for block in blocks)
    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return []
    start = matches[max(0, len(matches) - budget)].start()
    tail = text[start:].strip()
    if not tail:
        return []
    pages = _ordered_unique(page for block in blocks for page in block.pages)
    metadata = _merge_metadata(block.metadata for block in blocks)
    metadata["overlap"] = True
    return [
        _Block(
            text=tail,
            heading_path=list(blocks[-1].heading_path),
            pages=pages,
            label="overlap",
            metadata=metadata,
        )
    ]


def _emit_blocks(blocks: list[_Block], source_id: str, ordinal: int) -> DocumentChunk:
    text = "\n\n".join(block.text for block in blocks).strip()
    pages = _ordered_unique(page for block in blocks for page in block.pages)
    headings = list(blocks[-1].heading_path) if blocks else []
    metadata = _merge_metadata(block.metadata for block in blocks)
    labels = _ordered_unique(block.label for block in blocks)
    if labels:
        metadata["labels"] = labels
    metadata.setdefault("chunker", "ke_layout")
    return _make_chunk(
        source_id=source_id,
        ordinal=ordinal,
        text=text,
        heading_path=headings,
        pages=pages,
        metadata=metadata,
    )


def _make_chunk(
    *,
    source_id: str,
    ordinal: int,
    text: str,
    heading_path: list[str],
    pages: list[int],
    metadata: dict[str, Any],
) -> DocumentChunk:
    page_start = min(pages) if pages else None
    page_end = max(pages) if pages else None
    return DocumentChunk(
        chunk_id=f"{source_id}:chunk:{ordinal:06d}",
        source_id=source_id,
        ordinal=ordinal,
        text=text,
        heading_path=heading_path,
        page_start=page_start,
        page_end=page_end,
        token_count=_token_count(text),
        content_sha256=_text_sha256(text),
        metadata=metadata,
    )


def _merge_metadata(values: Iterable[dict[str, Any]]) -> dict[str, Any]:
    collected: dict[str, list[Any]] = {}
    for value in values:
        for key, item in value.items():
            collected.setdefault(key, [])
            if isinstance(item, list):
                collected[key].extend(item)
            else:
                collected[key].append(item)
    output: dict[str, Any] = {}
    for key, items in collected.items():
        unique = _ordered_unique(items)
        output[key] = unique[0] if len(unique) == 1 else unique
    return output


def _pages_from_prov(prov: Any) -> list[int]:
    if not isinstance(prov, (list, tuple)):
        return []
    pages: list[int] = []
    for entry in prov:
        value = (
            entry.get("page_no") if isinstance(entry, Mapping) else getattr(entry, "page_no", None)
        )
        page = _page_number(value)
        if page is not None:
            pages.append(page)
    return _ordered_unique(pages)


def _page_number(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    # Docling currently uses one-based page numbers.  A zero occasionally
    # appears in hand-built fixtures; normalize it to the contract's >=1 range.
    return number if number >= 1 else 1 if number == 0 else None


def _heading_level(item: dict[str, Any], *, default: int) -> int:
    try:
        value = int(item.get("level", default))
    except (TypeError, ValueError):
        value = default
    return min(6, max(1, value))


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = json.dumps(value, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            output.append(value)
    return output


__all__ = [
    "CHUNK_SCHEMA_VERSION",
    "DocumentChunk",
    "chunk_document",
    "construct_provenance_chunks",
    "load_chunks_jsonl",
    "write_chunks_jsonl",
]
