"""Small, dependency-free value objects shared by the ingestion modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """Identity and basic file facts for an input document.

    The content digest, rather than a path or modification time, is the source of
    identity.  This makes the identifier stable when a source is moved.
    """

    path: Path
    source_id: str
    slug: str
    title: str
    extension: str
    media_type: str
    content_sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "slug": self.slug,
            "title": self.title,
            "filename": self.path.name,
            "extension": self.extension,
            "media_type": self.media_type,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ConvertedDocument:
    """Normalized output from any source-format converter."""

    markdown: str
    canonical_document: dict[str, Any]
    backend: str
    backend_version: str | None = None
    assets: dict[str, bytes] = field(default_factory=dict)
    # Kept only for immediate HybridChunker use.  It is intentionally not
    # serialized into the source bundle or cache.
    docling_document: Any | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class SourceBundle:
    """Paths and identity for a published source bundle."""

    source_id: str
    source_dir: Path
    source_path: Path
    markdown_path: Path
    canonical_path: Path
    chunks_path: Path
    manifest_path: Path
    assets_dir: Path
    content_sha256: str
    cache_hit: bool = False

    @property
    def bundle_dir(self) -> Path:
        """Compatibility alias used by callers that prefer ``bundle_dir``."""

        return self.source_dir

    @property
    def document_path(self) -> Path:
        """Compatibility alias for the canonical document JSON path."""

        return self.canonical_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_dir": str(self.source_dir),
            "source_path": str(self.source_path),
            "markdown_path": str(self.markdown_path),
            "canonical_path": str(self.canonical_path),
            "chunks_path": str(self.chunks_path),
            "manifest_path": str(self.manifest_path),
            "assets_dir": str(self.assets_dir),
            "content_sha256": self.content_sha256,
            "cache_hit": self.cache_hit,
        }
