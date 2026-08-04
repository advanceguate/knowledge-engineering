"""Format router and canonical source-bundle publisher."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ke.artifact_store import ArtifactStore
from ke.ontology.chunking import chunk_document, write_chunks_jsonl

from .common import (
    canonical_document_from_markdown,
    fingerprint_source,
    json_dumps,
    jsonable,
    read_text_source,
    sha256_file,
)
from .docling import convert_with_docling
from .epub import convert_epub
from .models import ConvertedDocument, SourceBundle, SourceFingerprint

INGEST_SCHEMA_VERSION = "1.0"
INGEST_IMPLEMENTATION_VERSION = "1"

DOCLING_EXTENSIONS = frozenset({".pdf", ".html", ".htm", ".docx"})
EPUB_EXTENSIONS = frozenset({".epub"})
PASSTHROUGH_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".text"})
SUPPORTED_EXTENSIONS = DOCLING_EXTENSIONS | EPUB_EXTENSIONS | PASSTHROUGH_EXTENSIONS


class UnsupportedSourceFormatError(ValueError):
    """Raised when no canonical converter is registered for an extension."""


def route_source(path: str | Path) -> str:
    """Return the canonical converter backend for ``path``."""

    extension = Path(path).suffix.lower()
    if extension in DOCLING_EXTENSIONS:
        return "docling"
    if extension in EPUB_EXTENSIONS:
        return "ebooklib"
    if extension in PASSTHROUGH_EXTENSIONS:
        return "pass_through"
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise UnsupportedSourceFormatError(
        f"Unsupported source format {extension or '<none>'!r}; supported extensions: {supported}"
    )


def ingest_source(
    source_path: str | Path,
    *,
    workspace: str | Path | None = None,
    settings: Any | None = None,
    title: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    conversion_options: Mapping[str, Any] | None = None,
    chunking_options: Mapping[str, Any] | None = None,
    converter: Any | None = None,
    epub_reader: Any | None = None,
    force: bool = False,
) -> SourceBundle:
    """Ingest any supported book format into a canonical local source bundle.

    Conversion output is cached by content and conversion settings.  A valid,
    unchanged bundle returns immediately without constructing or invoking a
    converter; this behavior also works with an injected fake converter.
    """

    if metadata is not None and source_metadata is not None:
        raise ValueError("Use either metadata or source_metadata, not both")
    operator_metadata = jsonable(dict(metadata or source_metadata or {}))
    fingerprint = fingerprint_source(source_path, title=title)
    backend = route_source(fingerprint.path)
    workspace_path = _resolve_workspace(workspace, settings)
    conversion_config = _conversion_config(settings, conversion_options)
    chunking_config = _chunking_config(settings, chunking_options)

    conversion_key = _cache_key(
        {
            "implementation": INGEST_IMPLEMENTATION_VERSION,
            "kind": "conversion",
            "content_sha256": fingerprint.content_sha256,
            "extension": fingerprint.extension,
            "title": fingerprint.title,
            "backend": backend,
            "configuration": conversion_config,
        }
    )
    bundle_key = _cache_key(
        {
            "implementation": INGEST_IMPLEMENTATION_VERSION,
            "kind": "source_bundle",
            "conversion_key": conversion_key,
            "source_id": fingerprint.source_id,
            "metadata": operator_metadata,
            "chunking": chunking_config,
        }
    )

    managed_store = ArtifactStore(workspace_path)
    sources_root = managed_store.sources
    cache_root = managed_store.cache / "ingest"
    if cache_root.is_symlink():
        raise ValueError(f"managed artifact directory cannot be a symlink: {cache_root}")
    target_dir = managed_store.source_dir(fingerprint.source_id)
    conversion_cache_dir = cache_root / conversion_key
    if conversion_cache_dir.is_symlink():
        raise ValueError(f"managed artifact directory cannot be a symlink: {conversion_cache_dir}")
    sources_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    if not force and _valid_bundle(target_dir, fingerprint, bundle_key):
        return _bundle_from_dir(target_dir, fingerprint, cache_hit=True)

    fresh_conversion: ConvertedDocument | None = None
    conversion_cache_hit = not force and _valid_conversion_cache(
        conversion_cache_dir, fingerprint, conversion_key
    )
    if not conversion_cache_hit:
        fresh_conversion = _convert_source(
            fingerprint,
            backend=backend,
            conversion_config=conversion_config,
            converter=converter,
            epub_reader=epub_reader,
        )
        _stamp_canonical_document(fresh_conversion.canonical_document, fingerprint)
        _publish_conversion_cache(
            conversion_cache_dir,
            fresh_conversion,
            fingerprint=fingerprint,
            cache_key=conversion_key,
            configuration=conversion_config,
        )

    stage = Path(tempfile.mkdtemp(prefix=f".{fingerprint.source_id}.", dir=sources_root))
    try:
        _copy_conversion_cache(conversion_cache_dir, stage)
        stored_source = stage / _stored_source_name(fingerprint)
        shutil.copyfile(fingerprint.path, stored_source)

        canonical_path = stage / "document.docling.json"
        markdown_path = stage / "document.md"
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        markdown_text = markdown_path.read_text(encoding="utf-8")
        chunk_input: Any = fresh_conversion if fresh_conversion is not None else canonical
        chunks = chunk_document(
            chunk_input,
            source_id=fingerprint.source_id,
            markdown=markdown_text,
            max_tokens=chunking_config["max_tokens"],
            overlap_tokens=chunking_config["overlap_tokens"],
            preserve_headings=chunking_config["preserve_headings"],
            preserve_tables=chunking_config["preserve_tables"],
            prefer_hybrid=chunking_config["strategy"] == "docling_hybrid",
        )
        chunks_path = write_chunks_jsonl(chunks, stage / "chunks.jsonl")
        manifest = _build_manifest(
            stage,
            fingerprint=fingerprint,
            stored_source=stored_source,
            backend=backend,
            conversion=fresh_conversion,
            conversion_configuration=conversion_config,
            chunking_configuration=chunking_config,
            chunks=chunks,
            metadata=operator_metadata,
            conversion_key=conversion_key,
            bundle_key=bundle_key,
        )
        (stage / "manifest.json").write_text(
            json_dumps(manifest) + "\n", encoding="utf-8", newline="\n"
        )
        if not chunks_path.is_file():  # Defensive assertion before atomic publication.
            raise RuntimeError("Chunk artifact was not written")
        _replace_generated_directory(stage, target_dir)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise

    return _bundle_from_dir(
        target_dir,
        fingerprint,
        cache_hit=conversion_cache_hit,
    )


def ingest(*args: Any, **kwargs: Any) -> SourceBundle:
    """Short alias for CLI/application callers."""

    return ingest_source(*args, **kwargs)


def _resolve_workspace(workspace: str | Path | None, settings: Any | None) -> Path:
    if workspace is not None:
        return Path(workspace).expanduser().resolve()
    if settings is not None and getattr(settings, "workspace", None) is not None:
        return Path(settings.workspace).expanduser().resolve()
    configured = os.environ.get("KE_WORKSPACE", "workspace")
    return Path(configured).expanduser().resolve()


def _conversion_config(settings: Any | None, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "backend": "docling",
        "ocr": "auto",
        "table_structure": "accurate",
        "export_markdown": True,
        "export_docling_json": True,
        "export_images": True,
    }
    section = getattr(settings, "conversion", None) if settings is not None else None
    defaults.update(_section_dict(section))
    defaults.update(dict(overrides or {}))
    return jsonable(defaults)


def _chunking_config(settings: Any | None, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "strategy": "docling_hybrid",
        "max_tokens": 1800,
        "overlap_tokens": 150,
        "preserve_headings": True,
        "preserve_tables": True,
    }
    section = getattr(settings, "chunking", None) if settings is not None else None
    defaults.update(_section_dict(section))
    defaults.update(dict(overrides or {}))
    normalized = jsonable(defaults)
    normalized["max_tokens"] = int(normalized["max_tokens"])
    normalized["overlap_tokens"] = int(normalized["overlap_tokens"])
    normalized["preserve_headings"] = bool(normalized["preserve_headings"])
    normalized["preserve_tables"] = bool(normalized["preserve_tables"])
    normalized["strategy"] = str(normalized["strategy"])
    if normalized["max_tokens"] < 1:
        raise ValueError("chunking max_tokens must be at least 1")
    if not 0 <= normalized["overlap_tokens"] < normalized["max_tokens"]:
        raise ValueError("chunking overlap_tokens must be >= 0 and smaller than max_tokens")
    return normalized


def _section_dict(section: Any) -> dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, Mapping):
        return dict(section)
    dumper = getattr(section, "model_dump", None)
    if callable(dumper):
        return dict(dumper(mode="json"))
    return {
        name: getattr(section, name)
        for name in dir(section)
        if not name.startswith("_") and not callable(getattr(section, name))
    }


def _convert_source(
    fingerprint: SourceFingerprint,
    *,
    backend: str,
    conversion_config: dict[str, Any],
    converter: Any | None,
    epub_reader: Any | None,
) -> ConvertedDocument:
    if backend == "docling":
        return convert_with_docling(
            fingerprint.path,
            converter=converter,
            ocr=conversion_config.get("ocr", "auto"),
            table_structure=str(conversion_config.get("table_structure", "accurate")),
            export_images=bool(conversion_config.get("export_images", True)),
        )
    if backend == "ebooklib":
        return convert_epub(
            fingerprint.path,
            title=fingerprint.title,
            content_sha256=fingerprint.content_sha256,
            reader=epub_reader,
        )
    if backend == "pass_through":
        markdown = read_text_source(fingerprint.path)
        canonical = canonical_document_from_markdown(
            markdown,
            title=fingerprint.title,
            filename=fingerprint.path.name,
            media_type=fingerprint.media_type,
            content_sha256=fingerprint.content_sha256,
            converter="pass_through",
        )
        return ConvertedDocument(
            markdown=markdown,
            canonical_document=canonical,
            backend="pass_through",
            backend_version=INGEST_IMPLEMENTATION_VERSION,
        )
    raise AssertionError(f"Unhandled ingestion backend: {backend}")


def _stamp_canonical_document(canonical: dict[str, Any], fingerprint: SourceFingerprint) -> None:
    canonical.setdefault("name", fingerprint.title)
    origin = canonical.setdefault("origin", {})
    if isinstance(origin, dict):
        origin.setdefault("filename", fingerprint.path.name)
        origin.setdefault("mimetype", fingerprint.media_type)
        origin.setdefault("binary_hash", fingerprint.content_sha256)
    # Keep genuine Docling JSON loadable by ``DoclingDocument.load_from_json``.
    # Source identity belongs in manifest.json because Docling's model rejects
    # unknown top-level fields.  The local canonical schema permits ``ke``.
    if canonical.get("schema_name") != "DoclingDocument":
        ke_metadata = canonical.setdefault("ke", {})
        if isinstance(ke_metadata, dict):
            ke_metadata["source_id"] = fingerprint.source_id
            ke_metadata["source_content_sha256"] = fingerprint.content_sha256


def _cache_key(value: Any) -> str:
    return hashlib.sha256(json_dumps(value, indent=None).encode("utf-8")).hexdigest()


def _publish_conversion_cache(
    target: Path,
    conversion: ConvertedDocument,
    *,
    fingerprint: SourceFingerprint,
    cache_key: str,
    configuration: dict[str, Any],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{cache_key}.", dir=target.parent))
    try:
        (stage / "document.md").write_text(conversion.markdown, encoding="utf-8", newline="")
        (stage / "document.docling.json").write_text(
            json_dumps(conversion.canonical_document) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        assets_dir = stage / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for relative_name, value in sorted(conversion.assets.items()):
            destination = _safe_asset_destination(assets_dir, relative_name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(bytes(value))
        cache_manifest = {
            "schema_version": INGEST_SCHEMA_VERSION,
            "cache_key": cache_key,
            "content_sha256": fingerprint.content_sha256,
            "source_id": fingerprint.source_id,
            "title": fingerprint.title,
            "backend": conversion.backend,
            "backend_version": conversion.backend_version,
            "configuration": configuration,
            "artifacts": _artifact_digests(stage, include_cache_manifest=False),
        }
        (stage / "cache-manifest.json").write_text(
            json_dumps(cache_manifest) + "\n", encoding="utf-8", newline="\n"
        )
        _replace_generated_directory(stage, target)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _valid_conversion_cache(
    directory: Path, fingerprint: SourceFingerprint, cache_key: str
) -> bool:
    manifest_path = directory / "cache-manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        manifest.get("cache_key") != cache_key
        or manifest.get("content_sha256") != fingerprint.content_sha256
        or manifest.get("source_id") != fingerprint.source_id
    ):
        return False
    return _verify_artifact_digests(directory, manifest.get("artifacts"))


def _copy_conversion_cache(cache_dir: Path, destination: Path) -> None:
    for filename in ("document.md", "document.docling.json"):
        shutil.copyfile(cache_dir / filename, destination / filename)
    source_assets = cache_dir / "assets"
    target_assets = destination / "assets"
    if source_assets.is_dir():
        shutil.copytree(source_assets, target_assets)
    else:
        target_assets.mkdir(parents=True, exist_ok=True)


def _build_manifest(
    stage: Path,
    *,
    fingerprint: SourceFingerprint,
    stored_source: Path,
    backend: str,
    conversion: ConvertedDocument | None,
    conversion_configuration: dict[str, Any],
    chunking_configuration: dict[str, Any],
    chunks: list[Any],
    metadata: dict[str, Any],
    conversion_key: str,
    bundle_key: str,
) -> dict[str, Any]:
    asset_records = [
        {"path": path.relative_to(stage).as_posix(), "sha256": sha256_file(path)}
        for path in sorted((stage / "assets").rglob("*"))
        if path.is_file()
    ]
    artifact_records = {
        "source": _artifact_record(stage, stored_source),
        "markdown": _artifact_record(stage, stage / "document.md"),
        "canonical_document": _artifact_record(stage, stage / "document.docling.json"),
        "chunks": _artifact_record(stage, stage / "chunks.jsonl"),
        "assets": asset_records,
    }
    converter_version = conversion.backend_version if conversion is not None else None
    if converter_version is None:
        try:
            cache_manifest = json.loads(
                (
                    stage.parent.parent
                    / "cache"
                    / "ingest"
                    / conversion_key
                    / "cache-manifest.json"
                ).read_text(encoding="utf-8")
            )
            converter_version = cache_manifest.get("backend_version")
        except (OSError, json.JSONDecodeError):
            converter_version = None
    return {
        "schema_version": INGEST_SCHEMA_VERSION,
        "source_id": fingerprint.source_id,
        "title": fingerprint.title,
        "content_sha256": fingerprint.content_sha256,
        "source": {
            "filename": fingerprint.path.name,
            "stored_path": stored_source.relative_to(stage).as_posix(),
            "extension": fingerprint.extension,
            "media_type": fingerprint.media_type,
            "size_bytes": fingerprint.size_bytes,
            "sha256": fingerprint.content_sha256,
        },
        "metadata": metadata,
        "conversion": {
            "backend": backend,
            "backend_version": converter_version,
            "configuration": conversion_configuration,
        },
        "chunking": {
            **chunking_configuration,
            "chunk_count": len(chunks),
            "token_count": sum(int(chunk.token_count) for chunk in chunks),
        },
        "cache": {"key": bundle_key, "conversion_key": conversion_key},
        "files": {
            "source": stored_source.relative_to(stage).as_posix(),
            "markdown": "document.md",
            "canonical_document": "document.docling.json",
            "chunks": "chunks.jsonl",
            "assets": "assets",
        },
        "artifacts": artifact_records,
    }


def _artifact_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _valid_bundle(directory: Path, fingerprint: SourceFingerprint, bundle_key: str) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        manifest.get("source_id") != fingerprint.source_id
        or manifest.get("content_sha256") != fingerprint.content_sha256
        or (manifest.get("cache") or {}).get("key") != bundle_key
    ):
        return False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    records: list[dict[str, Any]] = []
    for key in ("source", "markdown", "canonical_document", "chunks"):
        record = artifacts.get(key)
        if not isinstance(record, dict):
            return False
        records.append(record)
    assets = artifacts.get("assets", [])
    if not isinstance(assets, list):
        return False
    records.extend(record for record in assets if isinstance(record, dict))
    return _verify_records(directory, records)


def _artifact_digests(root: Path, *, include_cache_manifest: bool) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if not include_cache_manifest and relative == "cache-manifest.json":
            continue
        output[relative] = sha256_file(path)
    return output


def _verify_artifact_digests(root: Path, artifacts: Any) -> bool:
    if not isinstance(artifacts, dict):
        return False
    required = {"document.md", "document.docling.json"}
    if not required.issubset(artifacts):
        return False
    for relative, expected in artifacts.items():
        path = _safe_relative_path(root, str(relative))
        if path is None or not path.is_file() or sha256_file(path) != expected:
            return False
    return (root / "assets").is_dir()


def _verify_records(root: Path, records: list[dict[str, Any]]) -> bool:
    for record in records:
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            return False
        path = _safe_relative_path(root, relative)
        if path is None or not path.is_file() or sha256_file(path) != expected:
            return False
    return (root / "assets").is_dir()


def _safe_asset_destination(root: Path, relative_name: str) -> Path:
    path = _safe_relative_path(root, str(relative_name).replace("\\", "/"))
    if path is None or path == root:
        raise ValueError(f"Unsafe converted asset path: {relative_name!r}")
    return path


def _safe_relative_path(root: Path, relative_name: str) -> Path | None:
    candidate = Path(relative_name)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def _replace_generated_directory(stage: Path, target: Path) -> None:
    if not target.exists():
        os.replace(stage, target)
        return

    backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.backup.", dir=target.parent))
    backup.rmdir()
    os.replace(target, backup)
    try:
        os.replace(stage, target)
    except BaseException:
        os.replace(backup, target)
        raise
    else:
        shutil.rmtree(backup)


def _stored_source_name(fingerprint: SourceFingerprint) -> str:
    return f"source{fingerprint.extension or '.bin'}"


def _bundle_from_dir(
    directory: Path, fingerprint: SourceFingerprint, *, cache_hit: bool
) -> SourceBundle:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    stored_relative = manifest.get("source", {}).get("stored_path") or _stored_source_name(
        fingerprint
    )
    return SourceBundle(
        source_id=fingerprint.source_id,
        source_dir=directory,
        source_path=directory / stored_relative,
        markdown_path=directory / "document.md",
        canonical_path=directory / "document.docling.json",
        chunks_path=directory / "chunks.jsonl",
        manifest_path=directory / "manifest.json",
        assets_dir=directory / "assets",
        content_sha256=fingerprint.content_sha256,
        cache_hit=cache_hit,
    )


__all__ = [
    "DOCLING_EXTENSIONS",
    "EPUB_EXTENSIONS",
    "INGEST_SCHEMA_VERSION",
    "PASSTHROUGH_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    "SourceBundle",
    "UnsupportedSourceFormatError",
    "fingerprint_source",
    "ingest",
    "ingest_source",
    "route_source",
]
