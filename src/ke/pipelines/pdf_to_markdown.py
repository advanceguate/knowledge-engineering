"""PDF-to-Markdown source pipeline and optional LangGraph facade."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from ke.ingest.common import fingerprint_source
from ke.ingest.docling import convert_with_docling
from ke.ingest.models import SourceBundle
from ke.ingest.router import ingest_source
from ke.ontology.chunking import load_chunks_jsonl


class PDFToMarkdownState(TypedDict, total=False):
    source_path: str
    path: str
    workspace: str
    output_dir: str
    title: str
    metadata: dict[str, Any]
    settings: Any
    converter: Any
    force: bool
    conversion_options: dict[str, Any]
    chunking_options: dict[str, Any]
    fingerprint: dict[str, Any]
    bundle: SourceBundle
    source_id: str
    manifest: dict[str, Any]
    chunk_count: int
    cache_hit: bool


def pdf_to_markdown(
    source_path: str | Path,
    *,
    workspace: str | Path | None = None,
    output_dir: str | Path | None = None,
    settings: Any | None = None,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    source_metadata: dict[str, Any] | None = None,
    conversion_options: dict[str, Any] | None = None,
    chunking_options: dict[str, Any] | None = None,
    converter: Any | None = None,
    force: bool = False,
) -> SourceBundle:
    """Convert and publish one PDF as a canonical source bundle."""

    path = Path(source_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"pdf_to_markdown requires a .pdf source, got {path.name!r}")
    if workspace is not None and output_dir is not None:
        raise ValueError("Use either workspace or output_dir, not both")
    return ingest_source(
        path,
        workspace=workspace if workspace is not None else output_dir,
        settings=settings,
        title=title,
        metadata=metadata,
        source_metadata=source_metadata,
        conversion_options=conversion_options,
        chunking_options=chunking_options,
        converter=converter,
        force=force,
    )


def export_markdown_and_canonical_json(bundle: SourceBundle) -> SourceBundle:
    """Validate the two required conversion exports and return ``bundle``."""

    if not bundle.markdown_path.is_file():
        raise FileNotFoundError(f"Missing Markdown export: {bundle.markdown_path}")
    if not bundle.canonical_path.is_file():
        raise FileNotFoundError(f"Missing canonical Docling JSON: {bundle.canonical_path}")
    try:
        value = json.loads(bundle.canonical_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid canonical Docling JSON: {bundle.canonical_path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Canonical Docling JSON is not an object: {bundle.canonical_path}")
    return bundle


def construct_provenance_chunks(bundle: SourceBundle) -> SourceBundle:
    """Validate the published provenance chunks and return ``bundle``."""

    chunks = load_chunks_jsonl(bundle)
    for chunk in chunks:
        if chunk.source_id != bundle.source_id:
            raise ValueError(
                f"Chunk {chunk.chunk_id!r} belongs to {chunk.source_id!r}, "
                f"expected {bundle.source_id!r}"
            )
    return bundle


def publish_source_bundle(bundle: SourceBundle) -> SourceBundle:
    """Validate the final manifest boundary and return the published bundle."""

    if not bundle.manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {bundle.manifest_path}")
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_id") != bundle.source_id:
        raise ValueError("Source manifest ID does not match the published directory")
    return bundle


def build_pdf_to_markdown_graph() -> Any:
    """Compile the five named pipeline stages as a LangGraph graph."""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is required to compile the PDF pipeline; call pdf_to_markdown() "
            "directly for the dependency-light execution path."
        ) from exc

    workflow = StateGraph(PDFToMarkdownState)
    workflow.add_node("fingerprint_source", _fingerprint_node)
    workflow.add_node("convert_with_docling", _convert_node)
    workflow.add_node("export_markdown_and_canonical_json", _export_node)
    workflow.add_node("construct_provenance_chunks", _chunks_node)
    workflow.add_node("publish_source_bundle", _publish_node)
    workflow.add_edge(START, "fingerprint_source")
    workflow.add_edge("fingerprint_source", "convert_with_docling")
    workflow.add_edge("convert_with_docling", "export_markdown_and_canonical_json")
    workflow.add_edge("export_markdown_and_canonical_json", "construct_provenance_chunks")
    workflow.add_edge("construct_provenance_chunks", "publish_source_bundle")
    workflow.add_edge("publish_source_bundle", END)
    return workflow.compile()


def build_graph() -> Any:
    """Compatibility alias for :func:`build_pdf_to_markdown_graph`."""

    return build_pdf_to_markdown_graph()


def _state_path(state: PDFToMarkdownState) -> str:
    value = state.get("source_path") or state.get("path")
    if not value:
        raise ValueError("PDF pipeline state requires source_path")
    return str(value)


def _fingerprint_node(state: PDFToMarkdownState) -> dict[str, Any]:
    fingerprint = fingerprint_source(_state_path(state), title=state.get("title"))
    if fingerprint.extension != ".pdf":
        raise ValueError(f"PDF pipeline requires a .pdf source, got {fingerprint.path.name!r}")
    return {"fingerprint": fingerprint.to_dict()}


def _convert_node(state: PDFToMarkdownState) -> dict[str, Any]:
    workspace = state.get("workspace")
    output_dir = state.get("output_dir")
    bundle = pdf_to_markdown(
        _state_path(state),
        workspace=workspace,
        output_dir=output_dir,
        settings=state.get("settings"),
        title=state.get("title"),
        metadata=state.get("metadata"),
        conversion_options=state.get("conversion_options"),
        chunking_options=state.get("chunking_options"),
        converter=state.get("converter"),
        force=bool(state.get("force", False)),
    )
    return {"bundle": bundle, "source_id": bundle.source_id, "cache_hit": bundle.cache_hit}


def _export_node(state: PDFToMarkdownState) -> dict[str, Any]:
    export_markdown_and_canonical_json(state["bundle"])
    return {}


def _chunks_node(state: PDFToMarkdownState) -> dict[str, Any]:
    bundle = construct_provenance_chunks(state["bundle"])
    return {"chunk_count": len(load_chunks_jsonl(bundle))}


def _publish_node(state: PDFToMarkdownState) -> dict[str, Any]:
    bundle = publish_source_bundle(state["bundle"])
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    return {"manifest": manifest}


run_pdf_to_markdown = pdf_to_markdown


__all__ = [
    "PDFToMarkdownState",
    "build_graph",
    "build_pdf_to_markdown_graph",
    "construct_provenance_chunks",
    "convert_with_docling",
    "export_markdown_and_canonical_json",
    "fingerprint_source",
    "pdf_to_markdown",
    "publish_source_bundle",
    "run_pdf_to_markdown",
]
