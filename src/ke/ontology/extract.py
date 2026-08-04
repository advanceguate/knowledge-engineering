"""Chunk-local, source-grounded ontology extraction.

The models in this module are deliberately *draft* models: concepts refer to
one another by labels and do not receive canonical identifiers until entity
resolution has completed.  This keeps model-generated identifiers out of the
canonical ontology.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ke.contracts import (
    DraftAxiom,
    DraftMentalModel,
    DraftRelation,
    DraftTerm,
    Evidence,
    ExtractionFragment,
    TermKind,
)
from ke.ids import content_hash
from ke.resource_paths import read_resource_text

# Backward-friendly name local to the ontology package.  The persisted schema
# remains the single shared contract from :mod:`ke.contracts`.
OntologyFragment = ExtractionFragment


@dataclass(frozen=True)
class ChunkView:
    source_id: str
    chunk_id: str
    text: str
    heading_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None


def _read_value(value: object, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def coerce_chunk(chunk: object, *, source_id: str | None = None) -> ChunkView:
    """Adapt canonical chunks, dictionaries, or simple test doubles."""

    text = _read_value(chunk, "text", "markdown", "content", "body", default="")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("chunk must provide non-empty text/markdown/content")
    resolved_source = source_id or _read_value(chunk, "source_id", default="unknown-source")
    chunk_id = _read_value(chunk, "chunk_id", "id", default=None)
    if not chunk_id:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        chunk_id = f"chunk-{digest}"
    headings = _read_value(chunk, "heading_path", "headings", default=[]) or []
    page_start = _read_value(chunk, "page_start", "page", default=None)
    page_end = _read_value(chunk, "page_end", default=page_start)
    return ChunkView(
        source_id=str(resolved_source),
        chunk_id=str(chunk_id),
        text=text,
        heading_path=tuple(str(item) for item in headings),
        page_start=int(page_start) if page_start is not None else None,
        page_end=int(page_end) if page_end is not None else None,
    )


def quote_sha256(quote: str) -> str:
    return hashlib.sha256(quote.encode("utf-8")).hexdigest()


def make_evidence(chunk: ChunkView, quote: str) -> Evidence:
    """Create evidence and verify that the exact quotation occurs in the chunk."""

    return Evidence(
        source_id=chunk.source_id,
        chunk_id=chunk.chunk_id,
        heading_path=list(chunk.heading_path),
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        quote=quote,
        quote_sha256=quote_sha256(quote),
        verified=quote in chunk.text,
    )


def _sentences(text: str) -> list[str]:
    prose = re.sub(r"(?m)^#{1,6}\s+.*$", "", text)
    if not prose:
        return []
    return [
        piece.strip() for piece in re.split(r"(?<=[.!?])(?:[ \t]+|\r?\n+)", prose) if piece.strip()
    ]


def _without_layout_heading_prefix(chunk: ChunkView) -> str:
    """Remove converter-emitted heading lines already represented in metadata."""

    if not chunk.heading_path:
        return chunk.text
    expected = {" ".join(value.split()).casefold() for value in chunk.heading_path}
    lines = chunk.text.splitlines(keepends=True)
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        visible = re.sub(r"^#{1,6}\s+", "", lines[cursor].strip())
        if " ".join(visible.split()).casefold() not in expected:
            break
        cursor += 1
    return "".join(lines[cursor:])


_DEFINITION_PATTERNS = (
    re.compile(
        r"^(?:[Tt]he\s+)?(?P<label>[A-Z][\w'’\-]*(?:\s+[A-Z\w][\w'’\-]*){0,5})\s+"
        r"(?:is|are|means|refers to|denotes)\s+(?P<definition>.+?)[.!?]?$"
    ),
    re.compile(r"^[\"“](?P<label>[^\"”]{2,80})[\"”]\s+(?:means|refers to)\s+(?P<definition>.+)$"),
)

_RELATION_PATTERN = re.compile(
    r"^(?P<subject>[A-Z][\w'’\-]*(?:\s+[A-Z\w][\w'’\-]*){0,4})\s+"
    r"(?P<predicate>causes|enables|requires|contains|influences|precedes|produces|prevents)\s+"
    r"(?P<object>(?:the\s+)?[A-Za-z][^,.;:!?]{1,80})",
    re.IGNORECASE,
)


def offline_extract(chunk: object, *, source_id: str | None = None) -> OntologyFragment:
    """Conservative deterministic extraction for offline use and tests.

    It intentionally extracts only surface-supported definitions, relations,
    and visibly named models.  A configured model gateway can provide richer
    interpretation while passing through the same grounding step.
    """

    view = coerce_chunk(chunk, source_id=source_id)
    sentences = _sentences(_without_layout_heading_prefix(view))
    terms: list[DraftTerm] = []
    relations: list[DraftRelation] = []
    axioms: list[DraftAxiom] = []
    mental_models: list[DraftMentalModel] = []
    known_labels: set[str] = set()

    def add_term(label: str, definition: str, sentence: str, kind: TermKind = "concept") -> None:
        label = " ".join(label.split()).strip(" .,:;\"'`()[]{}")
        key = label.casefold()
        if not label or key in known_labels:
            return
        known_labels.add(key)
        terms.append(
            DraftTerm(
                label=label,
                kind=kind,
                definition=definition.strip().rstrip(".!?"),
                evidence=[make_evidence(view, sentence)],
                confidence=0.66,
            )
        )

    for sentence_quote in sentences[:80]:
        sentence = re.sub(r"\s+", " ", sentence_quote).strip()
        for pattern in _DEFINITION_PATTERNS:
            match = pattern.match(sentence)
            if match:
                label = match.group("label")
                definition = match.group("definition")
                add_term(label, definition, sentence_quote)
                axioms.append(
                    DraftAxiom(
                        statement=sentence,
                        modality="definitional",
                        epistemic_status="asserted",
                        evidence=[make_evidence(view, sentence_quote)],
                        confidence=0.64,
                    )
                )
                break

        relation_match = _RELATION_PATTERN.match(sentence)
        if relation_match:
            subject = relation_match.group("subject").strip()
            object_label = re.sub(
                r"^the\s+", "", relation_match.group("object"), flags=re.I
            ).strip()
            predicate = relation_match.group("predicate").casefold()
            add_term(subject, "", sentence_quote)
            add_term(object_label, "", sentence_quote)
            relations.append(
                DraftRelation(
                    subject_label=subject,
                    predicate=predicate,
                    object_label=object_label,
                    evidence=[make_evidence(view, sentence_quote)],
                    confidence=0.62,
                )
            )
            axioms.append(
                DraftAxiom(
                    statement=sentence,
                    modality="causal"
                    if predicate in {"causes", "produces", "prevents"}
                    else "descriptive",
                    epistemic_status="asserted",
                    evidence=[make_evidence(view, sentence_quote)],
                    confidence=0.61,
                )
            )

        model_match = re.search(
            r"\b(?:The\s+)?(?P<name>(?:[A-Z][\w'’\-]*\s+){0,5}"
            r"(?:Model|Framework|Method|Process))\b",
            sentence,
        )
        if model_match:
            name = model_match.group("name")
            mental_models.append(
                DraftMentalModel(
                    name=name,
                    purpose=sentence.rstrip(".!?"),
                    evidence=[make_evidence(view, sentence_quote)],
                    confidence=0.58,
                )
            )

    summary = " ".join(sentences[:2])
    return OntologyFragment(
        source_id=view.source_id,
        chunk_id=view.chunk_id,
        section_summary=summary,
        terms=terms,
        relations=relations,
        axioms=axioms,
        mental_models=mental_models,
    )


def load_map_prompt() -> str:
    return read_resource_text("prompts", "ontology_map.md")


def render_map_prompt(chunk: ChunkView, template: str | None = None) -> str:
    instructions = template if template is not None else load_map_prompt()
    metadata = json.dumps(
        {
            "source_id": chunk.source_id,
            "chunk_id": chunk.chunk_id,
            "heading_path": list(chunk.heading_path),
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
        },
        ensure_ascii=False,
    )
    return f"{instructions}\n\nCHUNK METADATA\n{metadata}\n\nCHUNK TEXT\n{chunk.text}"


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _supported_kwargs(callable_: Callable[..., Any], candidates: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return candidates
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return candidates
    return {key: value for key, value in candidates.items() if key in signature.parameters}


async def _invoke_gateway(
    gateway: object,
    prompt: str,
    *,
    model: str | None = None,
    cache_key: str | None = None,
) -> Any:
    """Call common structured/fake gateway interfaces without provider coupling."""

    candidates = {
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "response_model": OntologyFragment,
        "output_type": OntologyFragment,
        "result_type": OntologyFragment,
        "schema": OntologyFragment,
        "task": "ontology_map",
        "model": model,
        "cache_key": cache_key,
    }
    for name in (
        "extract_ontology",
        "complete_structured",
        "generate_structured",
        "structured",
        "complete",
        "generate",
        "run",
    ):
        method = getattr(gateway, name, None)
        if method is None:
            continue
        kwargs = _supported_kwargs(method, candidates)
        try:
            if kwargs:
                return await _maybe_await(method(**kwargs))
            return await _maybe_await(method(prompt))
        except TypeError as exc:
            # Some intentionally tiny fakes expose positional-only callables.
            try:
                return await _maybe_await(method(prompt, OntologyFragment))
            except TypeError:
                raise exc from None
    if callable(gateway):
        method = gateway
        kwargs = _supported_kwargs(method, candidates)
        return await _maybe_await(method(**kwargs) if kwargs else method(prompt))
    raise TypeError("model gateway has no supported structured-generation method")


def _unwrap_gateway_result(result: Any) -> Any:
    for name in ("output", "data", "result", "parsed"):
        if hasattr(result, name):
            return getattr(result, name)
    if isinstance(result, str):
        return json.loads(result)
    return result


def _normalize_evidence(item: object, chunk: ChunkView) -> Evidence:
    raw = item.model_dump() if isinstance(item, BaseModel) else dict(item)  # type: ignore[arg-type]
    quote = str(raw.get("quote", ""))
    if not quote:
        raise ValueError("extracted evidence is missing an exact quotation")
    return Evidence(
        source_id=chunk.source_id,
        chunk_id=chunk.chunk_id,
        heading_path=list(chunk.heading_path),
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        quote=quote,
        quote_sha256=quote_sha256(quote),
        verified=quote in chunk.text,
    )


def ground_fragment(
    fragment: OntologyFragment | Mapping[str, Any],
    chunk: object,
    *,
    strict: bool = True,
) -> OntologyFragment:
    """Recompute all evidence hashes and verification flags from source text.

    Unsupported candidate objects are always omitted from the canonical path;
    ``strict`` remains an API-compatible policy marker. The raw model response
    should still be saved by the pipeline before this operation.
    """

    view = coerce_chunk(chunk)
    parsed = (
        fragment
        if isinstance(fragment, OntologyFragment)
        else OntologyFragment.model_validate(fragment)
    )
    data = parsed.model_dump()
    data["source_id"] = view.source_id
    data["chunk_id"] = view.chunk_id
    for collection in ("terms", "relations", "axioms", "mental_models"):
        grounded_items: list[dict[str, Any]] = []
        for item in data[collection]:
            try:
                evidence = [_normalize_evidence(ev, view) for ev in item.get("evidence", [])]
            except (TypeError, ValueError):
                evidence = []
            # Unsupported candidates are quarantined for both strict and
            # permissive runs. ``strict=False`` controls pipeline policy, not
            # whether unverified model content may influence canonical fields.
            evidence = [ev for ev in evidence if ev.verified]
            if not evidence:
                continue
            item["evidence"] = [ev.model_dump(mode="json") for ev in evidence]
            grounded_items.append(item)
        data[collection] = grounded_items
    return OntologyFragment.model_validate(data)


async def extract_chunk(
    chunk: object,
    gateway: object | None = None,
    *,
    source_id: str | None = None,
    prompt_template: str | None = None,
    strict_grounding: bool = True,
    model: str | None = None,
    cache_key: str | None = None,
    ground_evidence: bool = True,
) -> OntologyFragment:
    """Extract one chunk with a gateway, or conservatively offline."""

    view = coerce_chunk(chunk, source_id=source_id)
    if gateway is None:
        return offline_extract(view)
    prompt = render_map_prompt(view, prompt_template)
    raw = _unwrap_gateway_result(
        await _invoke_gateway(gateway, prompt, model=model, cache_key=cache_key)
    )
    if isinstance(raw, OntologyFragment):
        parsed = raw
    else:
        payload = dict(raw)
        payload.setdefault("source_id", view.source_id)
        payload.setdefault("chunk_id", view.chunk_id)
        parsed = OntologyFragment.model_validate(payload)
    parsed = parsed.model_copy(update={"source_id": view.source_id, "chunk_id": view.chunk_id})
    return ground_fragment(parsed, view, strict=strict_grounding) if ground_evidence else parsed


async def extract_chunks(
    chunks: Sequence[object] | Iterable[object],
    gateway: object | None = None,
    *,
    source_id: str | None = None,
    concurrency: int = 6,
    strict_grounding: bool = True,
    model: str | None = None,
    ground_evidence: bool = True,
) -> list[OntologyFragment]:
    """Map extraction concurrently while returning deterministic chunk order."""

    materialized = list(chunks)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    map_prompt = load_map_prompt() if gateway is not None else None
    schema_hash = content_hash(OntologyFragment.model_json_schema())

    async def one(index: int, chunk: object) -> tuple[int, OntologyFragment]:
        async with semaphore:
            view = coerce_chunk(chunk, source_id=source_id)
            rendered_prompt = render_map_prompt(view, map_prompt) if map_prompt is not None else ""
            return (
                index,
                await extract_chunk(
                    view,
                    gateway,
                    source_id=source_id,
                    prompt_template=map_prompt,
                    strict_grounding=strict_grounding,
                    model=model,
                    ground_evidence=ground_evidence,
                    cache_key=content_hash(
                        {
                            "task": "ontology-map-v1",
                            "model": model,
                            "prompt_sha256": quote_sha256(rendered_prompt),
                            "schema_sha256": schema_hash,
                        }
                    ),
                ),
            )

    results = await asyncio.gather(*(one(i, chunk) for i, chunk in enumerate(materialized)))
    return [fragment for _, fragment in sorted(results, key=lambda pair: pair[0])]


def extract_chunk_sync(*args: Any, **kwargs: Any) -> OntologyFragment:
    """Synchronous convenience wrapper (must not run inside an event loop)."""

    return asyncio.run(extract_chunk(*args, **kwargs))


__all__ = [
    "ChunkView",
    "DraftAxiom",
    "DraftMentalModel",
    "DraftRelation",
    "DraftTerm",
    "OntologyFragment",
    "coerce_chunk",
    "extract_chunk",
    "extract_chunk_sync",
    "extract_chunks",
    "ground_fragment",
    "make_evidence",
    "offline_extract",
    "quote_sha256",
]
