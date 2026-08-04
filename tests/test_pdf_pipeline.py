from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ke.ingest.router import ingest_source, route_source
from ke.ontology.chunking import load_chunks_jsonl
from ke.pipelines.pdf_to_markdown import pdf_to_markdown


class _FakeDocument:
    def export_to_markdown(self) -> str:
        return "# A heading\n\nGrounded statement on page two.\n"

    def export_to_dict(self) -> dict[str, object]:
        return {
            "schema_name": "DoclingDocument",
            "version": "1.0",
            "body": {
                "children": [
                    {"$ref": "#/texts/0"},
                    {"$ref": "#/texts/1"},
                ]
            },
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "label": "section_header",
                    "level": 1,
                    "text": "A heading",
                    "prov": [{"page_no": 1}],
                },
                {
                    "self_ref": "#/texts/1",
                    "label": "text",
                    "text": "Grounded statement on page two.",
                    "prov": [{"page_no": 2}],
                },
            ],
            "tables": [],
            "pictures": [],
        }


class _FakeResult:
    document = _FakeDocument()
    status = "success"


class _FakeConverter:
    def __init__(self) -> None:
        self.calls = 0

    def convert(self, _: str) -> _FakeResult:
        self.calls += 1
        return _FakeResult()


def test_pdf_pipeline_publishes_provenance_and_reuses_bundle_cache(tmp_path: Path) -> None:
    source = tmp_path / "Example Book.pdf"
    source.write_bytes(b"fake-pdf-content")
    converter = _FakeConverter()
    workspace = tmp_path / "workspace"

    first = pdf_to_markdown(
        source,
        workspace=workspace,
        converter=converter,
        chunking_options={"strategy": "ke_layout", "max_tokens": 32, "overlap_tokens": 4},
    )

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert first.source_id == f"example-book-{digest[:12]}"
    assert first.source_path.name == "source.pdf"
    assert first.markdown_path.read_text(encoding="utf-8").startswith("# A heading")
    assert first.canonical_path.is_file()
    assert first.assets_dir.is_dir()
    assert converter.calls == 1

    chunks = load_chunks_jsonl(first)
    assert chunks
    assert chunks[0].heading_path == ["A heading"]
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert all(chunk.source_id == first.source_id for chunk in chunks)

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["content_sha256"] == digest
    assert manifest["source"]["sha256"] == digest
    assert manifest["chunking"]["chunk_count"] == len(chunks)

    second = pdf_to_markdown(
        source,
        workspace=workspace,
        converter=converter,
        chunking_options={"strategy": "ke_layout", "max_tokens": 32, "overlap_tokens": 4},
    )
    assert second.cache_hit is True
    assert converter.calls == 1


def test_markdown_is_passed_through_exactly_without_docling(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    original = "# Notes\r\n\r\nExact source text.\r\n"
    source.write_bytes(original.encode("utf-8"))

    bundle = ingest_source(
        source,
        workspace=tmp_path / "workspace",
        chunking_options={"max_tokens": 24, "overlap_tokens": 2},
    )

    assert route_source(source) == "pass_through"
    assert bundle.markdown_path.read_bytes() == source.read_bytes()
    assert load_chunks_jsonl(bundle)[0].heading_path == ["Notes"]


def test_pdf_pipeline_rejects_non_pdf(tmp_path: Path) -> None:
    source = tmp_path / "book.md"
    source.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError, match=r"requires a \.pdf"):
        pdf_to_markdown(source, workspace=tmp_path / "workspace")


def test_ingestion_rejects_symlinked_managed_directories(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n\nExact source text.\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "sources").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        ingest_source(source, workspace=workspace)

    assert not any(outside.iterdir())
