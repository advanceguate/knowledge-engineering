"""LangGraph pipeline for source-grounded domain ontology extraction."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict
from rdflib import Graph

from ke.artifact_store import ArtifactStore
from ke.contracts import DomainOntology, ExtractionFragment
from ke.ids import content_hash, sha256_file, sha256_text, slugify
from ke.ingest.models import SourceBundle
from ke.llm import ModelGateway
from ke.ontology.chunking import DocumentChunk, chunk_document, load_chunks_jsonl
from ke.ontology.export import ExportResult, export_ontology
from ke.ontology.extract import extract_chunks, ground_fragment
from ke.ontology.reduce import ReductionBudget, canonicalize_ids, hierarchical_reduce
from ke.ontology.resolve import ResolutionResult, resolve_entities_async
from ke.resource_paths import read_resource_text
from ke.semantic.rdf import ontology_to_graph
from ke.semantic.validate import OntologyValidationError, ValidationReport, validate_graph
from ke.settings import Settings, load_settings

PIPELINE_IMPLEMENTATION_VERSION = "1.2"
SourceInput = str | Path | SourceBundle


class SourceContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source_id: str
    source_dir: Path
    chunks_path: Path
    canonical_path: Path
    markdown_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    title: str
    content_sha256: str
    chunks_sha256: str


class BookProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    source_id: str
    chunk_count: int
    heading_counts: dict[str, int]
    page_start: int | None = None
    page_end: int | None = None


class DomainOntologyBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ontology_id: str
    source_ids: list[str]
    output_dir: str
    ontology: DomainOntology
    validation: ValidationReport
    fragment_count: int
    cache_hit: bool = False


class DomainOntologyState(TypedDict):
    source_input: SourceInput
    title_override: NotRequired[str | None]
    context: NotRequired[SourceContext]
    profile: NotRequired[BookProfile]
    chunks: NotRequired[list[DocumentChunk]]
    raw_fragments: NotRequired[list[ExtractionFragment]]
    grounded_fragments: NotRequired[list[ExtractionFragment]]
    resolution: NotRequired[ResolutionResult]
    ontology: NotRequired[DomainOntology]
    rdf_graph: NotRequired[Graph]
    validation: NotRequired[ValidationReport]
    bundle: NotRequired[DomainOntologyBundle]
    pipeline_cache_key: NotRequired[str]


def _safe_source_artifact_path(source_dir: Path, relative_name: object) -> Path:
    """Resolve a manifest artifact without permitting traversal or symlinks."""

    if not isinstance(relative_name, str) or not relative_name.strip() or "\\" in relative_name:
        raise ValueError(f"unsafe source artifact path: {relative_name!r}")
    relative = Path(relative_name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe source artifact path: {relative_name!r}")
    candidate = source_dir / relative
    cursor = source_dir
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"source artifact path must not contain symlinks: {relative_name!r}")
    resolved_root = source_dir.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"source artifact path escapes bundle: {relative_name!r}") from exc
    if not resolved.is_file():
        raise ValueError(f"source manifest artifact is missing: {relative_name!r}")
    return resolved


def _verify_source_manifest_artifacts(
    source_dir: Path,
    manifest: dict[str, Any],
    *,
    chunks_path: Path,
    canonical_path: Path,
    markdown_path: Path,
) -> None:
    """Verify the canonical ingest artifacts before any cache lookup or extraction."""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("source manifest must declare artifact digests")
    required = {"source", "markdown", "canonical_document", "chunks"}
    missing = sorted(required - set(artifacts))
    if missing:
        raise ValueError(f"source manifest is missing artifact records: {missing}")

    expected_paths = {
        "markdown": markdown_path,
        "canonical_document": canonical_path,
        "chunks": chunks_path,
    }
    verified: dict[str, tuple[Path, str]] = {}
    for name in sorted(required):
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"source manifest artifact {name!r} must be an object")
        relative_name = record.get("path")
        expected_digest = record.get("sha256")
        if not isinstance(expected_digest, str) or not expected_digest.strip():
            raise ValueError(f"source manifest artifact {name!r} has no SHA-256 digest")
        artifact_path = _safe_source_artifact_path(source_dir, relative_name)
        expected_path = expected_paths.get(name)
        if expected_path is not None and artifact_path != expected_path.resolve():
            raise ValueError(
                f"source manifest artifact {name!r} does not bind to {expected_path.name!r}"
            )
        actual_digest = sha256_file(artifact_path)
        if actual_digest != expected_digest:
            raise ValueError(f"source manifest artifact digest mismatch for {name!r}")
        verified[name] = (artifact_path, actual_digest)

    source_record_path = str(artifacts["source"].get("path"))
    file_records = manifest.get("files")
    source_metadata = manifest.get("source")
    manifest_source_path = (
        file_records.get("source") if isinstance(file_records, dict) else None
    ) or (source_metadata.get("stored_path") if isinstance(source_metadata, dict) else None)
    if manifest_source_path is not None and source_record_path != manifest_source_path:
        raise ValueError("source manifest artifact 'source' does not bind to the stored source")
    source_digest = verified["source"][1]
    declared_content_digest = manifest.get("content_sha256")
    declared_source_digest = (
        source_metadata.get("sha256") if isinstance(source_metadata, dict) else None
    )
    for label, declared in (
        ("content_sha256", declared_content_digest),
        ("source.sha256", declared_source_digest),
    ):
        if declared is not None and declared != source_digest:
            raise ValueError(f"source manifest {label} does not match source artifact digest")

    asset_records = artifacts.get("assets", [])
    if not isinstance(asset_records, list):
        raise ValueError("source manifest artifact 'assets' must be a list")
    for index, record in enumerate(asset_records):
        if not isinstance(record, dict):
            raise ValueError(f"source manifest asset record {index} must be an object")
        expected_digest = record.get("sha256")
        if not isinstance(expected_digest, str) or not expected_digest.strip():
            raise ValueError(f"source manifest asset record {index} has no SHA-256 digest")
        artifact_path = _safe_source_artifact_path(source_dir, record.get("path"))
        if sha256_file(artifact_path) != expected_digest:
            raise ValueError(f"source manifest artifact digest mismatch for asset {index}")


def load_source_bundle(source: SourceInput, *, title: str | None = None) -> SourceContext:
    """Load and validate the local source-bundle boundary."""

    if isinstance(source, SourceBundle):
        expected_source_id = source.source_id
        source_dir = source.source_dir
        chunks_path = source.chunks_path
        canonical_path = source.canonical_path
        markdown_path = source.markdown_path
        manifest_path = source.manifest_path
    else:
        expected_source_id = None
        path = Path(source)
        source_dir = path if path.is_dir() else path.parent
        chunks_path = path if path.name == "chunks.jsonl" else source_dir / "chunks.jsonl"
        canonical_path = source_dir / "document.docling.json"
        markdown_path = source_dir / "document.md"
        manifest_path = source_dir / "manifest.json"
    for required in (chunks_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"source bundle is missing {required.name}: {required}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"source manifest must contain an object: {manifest_path}")
    _verify_source_manifest_artifacts(
        source_dir,
        manifest,
        chunks_path=chunks_path,
        canonical_path=canonical_path,
        markdown_path=markdown_path,
    )
    source_id = str(manifest.get("source_id") or source_dir.name)
    if not source_id.strip():
        raise ValueError("source manifest source_id must not be blank")
    if expected_source_id is not None and source_id != expected_source_id:
        raise ValueError(
            f"source bundle ID {expected_source_id!r} does not match manifest {source_id!r}"
        )
    resolved_title = str(title or manifest.get("title") or source_id)
    source_metadata = manifest.get("source")
    source_digest = source_metadata.get("sha256") if isinstance(source_metadata, dict) else None
    content_digest = str(manifest.get("content_sha256") or source_digest or content_hash(manifest))
    return SourceContext(
        source_id=source_id,
        source_dir=source_dir,
        chunks_path=chunks_path,
        canonical_path=canonical_path,
        markdown_path=markdown_path,
        manifest_path=manifest_path,
        manifest=manifest,
        title=resolved_title,
        content_sha256=content_digest,
        chunks_sha256=sha256_file(chunks_path),
    )


def _load_validated_chunks(context: SourceContext) -> list[DocumentChunk]:
    chunks = load_chunks_jsonl(context.chunks_path)
    mismatched = [chunk.chunk_id for chunk in chunks if chunk.source_id != context.source_id]
    if mismatched:
        raise ValueError(
            f"source manifest ID {context.source_id!r} does not match chunk source_id "
            f"for chunks: {mismatched}"
        )
    return chunks


def profile_book_structure(
    context: SourceContext,
    chunks: Sequence[DocumentChunk],
) -> BookProfile:
    headings = Counter(
        heading for chunk in chunks for heading in chunk.heading_path if heading.strip()
    )
    pages = [
        page for chunk in chunks for page in (chunk.page_start, chunk.page_end) if page is not None
    ]
    return BookProfile(
        title=context.title,
        source_id=context.source_id,
        chunk_count=len(chunks),
        heading_counts=dict(sorted(headings.items())),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
    )


def _gateway_fingerprint(gateway: ModelGateway | None, settings: Settings) -> dict[str, Any]:
    if gateway is None:
        return {"mode": "offline"}
    gateway_type = type(gateway)
    explicit = getattr(gateway, "cache_identity", None)
    if callable(explicit):
        explicit = explicit()
    if explicit is not None:
        identity: Any = explicit
    elif default_model := getattr(gateway, "default_model", None):
        identity = {"default_model": str(default_model)}
    else:
        # Generic/fake gateway instances can carry different scripted or custom
        # semantics even when they share a Python class.
        identity = {"instance": f"0x{id(gateway):x}"}
    return {
        "mode": "live",
        "class": f"{gateway_type.__module__}.{gateway_type.__qualname__}",
        "identity": identity,
        "extract_model": settings.extract_model,
        "resolve_model": settings.resolve_model,
    }


def _pipeline_dependency_fingerprints() -> dict[str, dict[str, str]]:
    return {
        "prompts": {
            name: sha256_text(read_resource_text("prompts", name))
            for name in (
                "ontology_map.md",
                "ontology_reduce.md",
                "concept_resolution.md",
            )
        },
        "schemas": {
            "domain_ontology": content_hash(DomainOntology.model_json_schema()),
            "extraction_fragment": content_hash(ExtractionFragment.model_json_schema()),
            "reduction_budget": content_hash(ReductionBudget.model_json_schema()),
            "resolution_result": content_hash(ResolutionResult.model_json_schema()),
        },
    }


def _pipeline_key(
    context: SourceContext,
    settings: Settings,
    gateway: ModelGateway | None,
) -> str:
    return content_hash(
        {
            "implementation": PIPELINE_IMPLEMENTATION_VERSION,
            "source_id": context.source_id,
            "title": context.title,
            "source_content_sha256": context.content_sha256,
            "chunks_sha256": context.chunks_sha256,
            "extract_model": settings.extract_model,
            "resolve_model": settings.resolve_model,
            "chunking_settings": settings.chunking.model_dump(mode="json"),
            "ontology_settings": settings.ontology.model_dump(mode="json"),
            "gateway": _gateway_fingerprint(gateway, settings),
            "dependencies": _pipeline_dependency_fingerprints(),
        }
    )


_REQUIRED_ONTOLOGY_ARTIFACTS = {
    "abstract.md",
    "axioms.yaml",
    "book-profile.json",
    "conflicts.yaml",
    "entities-thesaurus.yaml",
    "mental-models.yaml",
    "ontology.json",
    "ontology.jsonld",
    "ontology.ttl",
    "resolution.json",
    "validation.json",
}


def _artifact_digests_are_valid(
    output_dir: Path,
    artifacts: object,
    *,
    fragment_count: int,
) -> bool:
    if not isinstance(artifacts, dict):
        return False
    relative_names = set(artifacts)
    if not _REQUIRED_ONTOLOGY_ARTIFACTS <= relative_names:
        return False
    fragment_names = sorted(name for name in relative_names if name.startswith("fragments/"))
    if len(fragment_names) != fragment_count or any(
        not name.endswith(".json") for name in fragment_names
    ):
        return False
    resolved_output = output_dir.resolve()
    for relative_name, expected_digest in artifacts.items():
        if not isinstance(relative_name, str) or not isinstance(expected_digest, str):
            return False
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            return False
        artifact_path = (output_dir / relative_path).resolve()
        if resolved_output not in artifact_path.parents or not artifact_path.is_file():
            return False
        if sha256_file(artifact_path) != expected_digest:
            return False
    return True


def _cached_bundle(
    context: SourceContext,
    store: ArtifactStore,
    cache_key: str,
) -> DomainOntologyBundle | None:
    try:
        cache = store.get_cached_json("domain-ontology", cache_key)
        if not isinstance(cache, dict):
            return None
        ontology_id = cache.get("ontology_id")
        manifest_digest = cache.get("manifest_sha256")
        if not isinstance(ontology_id, str) or not isinstance(manifest_digest, str):
            return None
        output_dir = store.ontology_dir(ontology_id)
        ontology_path = output_dir / "ontology.json"
        validation_path = output_dir / "validation.json"
        manifest_path = output_dir / "manifest.json"
        if not all(path.is_file() for path in (ontology_path, validation_path, manifest_path)):
            return None
        if sha256_file(manifest_path) != manifest_digest:
            return None
        manifest = store.read_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("pipeline_cache_key") != cache_key:
            return None
        fragment_count = int(manifest.get("fragment_count", -1))
        if fragment_count < 0 or not _artifact_digests_are_valid(
            output_dir,
            manifest.get("artifacts"),
            fragment_count=fragment_count,
        ):
            return None
        ontology = DomainOntology.model_validate(store.read_json(ontology_path))
        if ontology.ontology_id != ontology_id or context.source_id not in ontology.source_ids:
            return None
        validation = ValidationReport.model_validate(store.read_json(validation_path))
        if not validation.conforms:
            return None
        return DomainOntologyBundle(
            ontology_id=ontology.ontology_id,
            source_ids=ontology.source_ids,
            output_dir=str(output_dir),
            ontology=ontology,
            validation=validation,
            fragment_count=fragment_count,
            cache_hit=True,
        )
    except (OSError, TypeError, ValueError, KeyError, AttributeError):
        return None


def build_domain_ontology_graph(
    *,
    store: ArtifactStore,
    settings: Settings,
    gateway: ModelGateway | None = None,
) -> Any:
    """Compile the extraction graph with explicit grounding and validation gates."""

    async def load_node(state: DomainOntologyState) -> dict[str, Any]:
        context = load_source_bundle(state["source_input"], title=state.get("title_override"))
        return {
            "context": context,
            "pipeline_cache_key": _pipeline_key(context, settings, gateway),
        }

    async def profile_node(state: DomainOntologyState) -> dict[str, Any]:
        chunks = _load_validated_chunks(state["context"])
        return {"profile": profile_book_structure(state["context"], chunks)}

    async def chunk_node(state: DomainOntologyState) -> dict[str, Any]:
        context = state["context"]
        chunks = _load_validated_chunks(context)
        if not chunks:
            chunks = chunk_document(
                context.source_dir,
                source_id=context.source_id,
                max_tokens=settings.chunking.max_tokens,
                overlap_tokens=settings.chunking.overlap_tokens,
                preserve_headings=settings.chunking.preserve_headings,
                preserve_tables=settings.chunking.preserve_tables,
            )
        if not chunks:
            raise ValueError("source bundle produced no extractable chunks")
        return {"chunks": chunks}

    async def map_node(state: DomainOntologyState) -> dict[str, Any]:
        fragments = await extract_chunks(
            state["chunks"],
            gateway,
            source_id=state["context"].source_id,
            concurrency=settings.llm_concurrency,
            strict_grounding=False,
            model=settings.extract_model,
            ground_evidence=False,
        )
        return {"raw_fragments": fragments}

    async def ground_node(state: DomainOntologyState) -> dict[str, Any]:
        if len(state["raw_fragments"]) != len(state["chunks"]):
            raise ValueError("map extraction did not return exactly one fragment per chunk")
        grounded = [
            ground_fragment(
                fragment,
                chunk,
                strict=settings.ontology.strict_grounding,
            )
            for fragment, chunk in zip(state["raw_fragments"], state["chunks"], strict=True)
        ]
        return {"grounded_fragments": grounded}

    async def resolve_node(state: DomainOntologyState) -> dict[str, Any]:
        try:
            resolution = await resolve_entities_async(
                state["grounded_fragments"],
                gateway=gateway
                if settings.ontology.entity_resolution.llm_adjudicate_ambiguous_pairs
                else None,
                model=settings.resolve_model,
                lexical_threshold=settings.ontology.entity_resolution.lexical_threshold,
            )
        except LookupError:
            # A deliberately minimal FakeModelGateway may script map outputs but
            # no optional ambiguity adjudications. Preserve ambiguity in that case.
            resolution = await resolve_entities_async(
                state["grounded_fragments"],
                lexical_threshold=settings.ontology.entity_resolution.lexical_threshold,
            )
        return {"resolution": resolution}

    async def reduce_node(state: DomainOntologyState) -> dict[str, Any]:
        ontology = hierarchical_reduce(
            state["grounded_fragments"],
            title=state["context"].title,
            source_ids=[state["context"].source_id],
            resolution=state["resolution"],
            minimum_confidence=settings.ontology.minimum_confidence,
            budget=ReductionBudget.model_validate(
                settings.ontology.salience_budget.model_dump(mode="json")
            ),
            batch_size=settings.ontology.reducer_batch_size,
        )
        return {"ontology": ontology}

    async def canonicalize_node(state: DomainOntologyState) -> dict[str, Any]:
        return {"ontology": canonicalize_ids(state["ontology"])}

    async def rdf_node(state: DomainOntologyState) -> dict[str, Any]:
        return {"rdf_graph": ontology_to_graph(state["ontology"])}

    async def validate_node(state: DomainOntologyState) -> dict[str, Any]:
        report = validate_graph(state["rdf_graph"])
        if not report.conforms:
            raise OntologyValidationError(report)
        return {"validation": report}

    async def publish_node(state: DomainOntologyState) -> dict[str, Any]:
        ontology = state["ontology"]
        output_dir = store.ontology_dir(ontology.ontology_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        fragments_dir = output_dir / "fragments"
        fragments_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.resolve() not in fragments_dir.resolve().parents:
            raise ValueError("ontology fragments directory escapes the ontology bundle")
        for stale_fragment in fragments_dir.glob("*.json"):
            stale_fragment.unlink()
        fragment_paths: list[Path] = []
        for index, fragment in enumerate(state["raw_fragments"], start=1):
            safe_name = slugify(fragment.chunk_id, max_length=48)
            fragment_paths.append(
                store.write_json(
                    fragments_dir / f"{index:05d}-{safe_name}.json",
                    fragment,
                )
            )
        store.write_json(output_dir / "resolution.json", state["resolution"])
        store.write_json(output_dir / "book-profile.json", state["profile"])
        exported: ExportResult = export_ontology(
            ontology,
            output_dir,
            store=store,
            require_conformance=True,
        )
        artifact_paths = [
            *(output_dir / name for name in sorted(_REQUIRED_ONTOLOGY_ARTIFACTS)),
            *fragment_paths,
        ]
        artifacts = {
            path.relative_to(output_dir).as_posix(): sha256_file(path) for path in artifact_paths
        }
        manifest = {
            "schema_version": "1.0",
            "ontology_id": ontology.ontology_id,
            "source_ids": ontology.source_ids,
            "source_content_sha256": state["context"].content_sha256,
            "chunks_sha256": state["context"].chunks_sha256,
            "title": state["context"].title,
            "pipeline_cache_key": state["pipeline_cache_key"],
            "fragment_count": len(state["raw_fragments"]),
            "canonical_counts": {
                "terms": len(ontology.terms),
                "relations": len(ontology.relations),
                "axioms": len(ontology.axioms),
                "mental_models": len(ontology.mental_models),
                "conflicts": len(ontology.conflicts),
            },
            "validation_conforms": exported.validation.conforms,
            "artifacts": artifacts,
        }
        manifest_path = store.write_json(output_dir / "manifest.json", manifest)
        store.put_cached_json(
            "domain-ontology",
            state["pipeline_cache_key"],
            {
                "ontology_id": ontology.ontology_id,
                "manifest_sha256": sha256_file(manifest_path),
            },
        )
        return {
            "bundle": DomainOntologyBundle(
                ontology_id=ontology.ontology_id,
                source_ids=ontology.source_ids,
                output_dir=str(output_dir),
                ontology=ontology,
                validation=exported.validation,
                fragment_count=len(state["raw_fragments"]),
                cache_hit=False,
            )
        }

    graph = StateGraph(DomainOntologyState)
    graph.add_node("load_source_bundle", load_node)
    graph.add_node("profile_book_structure", profile_node)
    graph.add_node("chunk_document", chunk_node)
    graph.add_node("map_extract_chunks", map_node)
    graph.add_node("ground_quotes", ground_node)
    graph.add_node("resolve_entities", resolve_node)
    graph.add_node("hierarchical_reduce", reduce_node)
    graph.add_node("canonicalize_ids", canonicalize_node)
    graph.add_node("export_rdf", rdf_node)
    graph.add_node("shacl_validate", validate_node)
    graph.add_node("publish_ontology_bundle", publish_node)
    graph.add_edge(START, "load_source_bundle")
    graph.add_edge("load_source_bundle", "profile_book_structure")
    graph.add_edge("profile_book_structure", "chunk_document")
    graph.add_edge("chunk_document", "map_extract_chunks")
    graph.add_edge("map_extract_chunks", "ground_quotes")
    graph.add_edge("ground_quotes", "resolve_entities")
    graph.add_edge("resolve_entities", "hierarchical_reduce")
    graph.add_edge("hierarchical_reduce", "canonicalize_ids")
    graph.add_edge("canonicalize_ids", "export_rdf")
    graph.add_edge("export_rdf", "shacl_validate")
    graph.add_edge("shacl_validate", "publish_ontology_bundle")
    graph.add_edge("publish_ontology_bundle", END)
    return graph.compile()


async def run_domain_ontology(
    source_bundle: SourceInput,
    *,
    store: ArtifactStore | None = None,
    settings: Settings | None = None,
    gateway: ModelGateway | None = None,
    title: str | None = None,
    force: bool = False,
) -> DomainOntologyBundle:
    """Run extraction and atomically publish a validated ontology bundle."""

    selected_settings = settings or load_settings()
    selected_store = store or ArtifactStore(selected_settings.workspace)
    context = load_source_bundle(source_bundle, title=title)
    _load_validated_chunks(context)
    cache_key = _pipeline_key(context, selected_settings, gateway)
    if not force:
        cached = _cached_bundle(context, selected_store, cache_key)
        if cached is not None:
            return cached
    graph = build_domain_ontology_graph(
        store=selected_store,
        settings=selected_settings,
        gateway=gateway,
    )
    state = await graph.ainvoke({"source_input": source_bundle, "title_override": title})
    return state["bundle"]


def run_domain_ontology_sync(*args: Any, **kwargs: Any) -> DomainOntologyBundle:
    """Synchronous convenience wrapper for CLI commands and scripts."""

    return asyncio.run(run_domain_ontology(*args, **kwargs))


extract_domain_ontology = run_domain_ontology
build_graph = build_domain_ontology_graph


__all__ = [
    "BookProfile",
    "DomainOntologyBundle",
    "DomainOntologyState",
    "SourceContext",
    "build_domain_ontology_graph",
    "build_graph",
    "extract_domain_ontology",
    "load_source_bundle",
    "profile_book_structure",
    "run_domain_ontology",
    "run_domain_ontology_sync",
]
