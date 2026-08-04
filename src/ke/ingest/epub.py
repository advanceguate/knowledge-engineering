"""EPUB ingestion through EbookLib with chapter-order provenance."""

from __future__ import annotations

import hashlib
import posixpath
import re
from collections.abc import Iterable
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from .common import canonical_document_from_markdown, markdown_blocks, sha256_file
from .models import ConvertedDocument


class EbookLibUnavailableError(RuntimeError):
    """Raised when EPUB conversion is requested without EbookLib."""


class EpubConversionError(RuntimeError):
    """Raised when an EPUB has no readable document content."""


def convert_epub(
    source_path: str | Path,
    *,
    title: str | None = None,
    content_sha256: str | None = None,
    reader: Any | None = None,
) -> ConvertedDocument:
    """Convert an EPUB to Markdown and the local canonical document shape.

    Spine order is used whenever available.  Chapter and href values are copied
    onto each canonical block so downstream evidence can still identify an EPUB
    location even though an EPUB has no stable page numbers.
    """

    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source does not exist: {path}")

    read_epub, item_document, item_image = _resolve_reader(reader)
    try:
        book = read_epub(str(path))
    except Exception as exc:
        raise EpubConversionError(f"EbookLib failed to read {path.name}: {exc}") from exc

    resolved_title = (title or _book_title(book) or path.stem).strip()
    digest = content_sha256 or sha256_file(path)
    image_assets, image_names = _collect_image_assets(book, item_image)

    chapters: list[str] = []
    canonical_blocks: list[dict[str, Any]] = []
    for chapter_index, item in enumerate(_ordered_documents(book, item_document), start=1):
        href = _item_name(item) or f"chapter-{chapter_index:04d}.xhtml"
        raw_html = _item_html(item)
        if not raw_html.strip():
            continue
        chapter_markdown, chapter_title = _html_to_markdown(
            raw_html,
            href=href,
            image_names=image_names,
        )
        if not chapter_markdown.strip():
            continue
        if not re.search(r"(?m)^#{1,6}\s+", chapter_markdown):
            heading = chapter_title or PurePosixPath(href).stem.replace("-", " ").strip()
            if heading:
                chapter_markdown = f"## {heading}\n\n{chapter_markdown}"
        chapter_markdown = _normalize_generated_markdown(chapter_markdown)
        chapters.append(chapter_markdown)
        for block in markdown_blocks(chapter_markdown):
            block["chapter"] = chapter_index
            block["href"] = href
            canonical_blocks.append(block)

    if not chapters:
        raise EpubConversionError(f"EPUB contains no readable spine documents: {path.name}")

    markdown = "\n\n".join(chapters).strip() + "\n"
    canonical = canonical_document_from_markdown(
        markdown,
        title=resolved_title,
        filename=path.name,
        media_type="application/epub+zip",
        content_sha256=digest,
        converter="ebooklib",
        supplied_blocks=canonical_blocks,
    )
    canonical["ke"].update(
        {
            "converter_version": _package_version("EbookLib"),
            "chapter_count": len(chapters),
            "location_semantics": "spine chapter/href; EPUB page numbers are intentionally unset",
        }
    )
    return ConvertedDocument(
        markdown=markdown,
        canonical_document=canonical,
        backend="ebooklib",
        backend_version=_package_version("EbookLib"),
        assets=image_assets,
    )


def _resolve_reader(reader: Any | None) -> tuple[Any, int, int]:
    if reader is not None:
        if callable(reader):
            read_epub = reader
        elif callable(getattr(reader, "read_epub", None)):
            read_epub = reader.read_epub
        else:
            raise TypeError("reader must be callable or expose read_epub(path)")
        # EbookLib's stable item type values.  Tests can inject a reader without
        # importing the optional package.
        return read_epub, 9, 1
    try:
        from ebooklib import ITEM_DOCUMENT, ITEM_IMAGE, epub
    except ImportError as exc:
        raise EbookLibUnavailableError(
            "EbookLib is required for EPUB ingestion. Install the project dependencies "
            "or inject a reader for tests."
        ) from exc
    return epub.read_epub, ITEM_DOCUMENT, ITEM_IMAGE


def _book_title(book: Any) -> str | None:
    getter = getattr(book, "get_metadata", None)
    if not callable(getter):
        return None
    try:
        values = getter("DC", "title")
    except Exception:
        return None
    if not values:
        return None
    value = values[0]
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    return str(value).strip() or None


def _ordered_documents(book: Any, item_document: int) -> Iterable[Any]:
    seen: set[str] = set()
    get_by_id = getattr(book, "get_item_with_id", None)
    for spine_entry in getattr(book, "spine", []) or []:
        item_id = spine_entry[0] if isinstance(spine_entry, (tuple, list)) else spine_entry
        if not callable(get_by_id):
            break
        item = get_by_id(item_id)
        if item is None or not _is_item_type(item, item_document) or not _is_chapter_item(item):
            continue
        identity = _item_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        yield item

    getter = getattr(book, "get_items_of_type", None)
    if callable(getter):
        candidates = getter(item_document)
    else:
        candidates = getattr(book, "items", []) or []
    for item in candidates:
        if not _is_item_type(item, item_document) or not _is_chapter_item(item):
            continue
        identity = _item_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        yield item


def _collect_image_assets(book: Any, item_image: int) -> tuple[dict[str, bytes], set[str]]:
    getter = getattr(book, "get_items_of_type", None)
    candidates = getter(item_image) if callable(getter) else getattr(book, "items", []) or []
    assets: dict[str, bytes] = {}
    known_names: set[str] = set()
    for item in candidates:
        if not _is_item_type(item, item_image):
            continue
        name = _safe_epub_path(_item_name(item))
        content_getter = getattr(item, "get_content", None)
        if not name or not callable(content_getter):
            continue
        try:
            content = bytes(content_getter())
        except Exception:
            continue
        # Avoid silent overwrites in malformed EPUBs while remaining stable.
        target_name = name
        if target_name in assets and assets[target_name] != content:
            stem = PurePosixPath(name).stem
            suffix = PurePosixPath(name).suffix
            parent = PurePosixPath(name).parent
            short_hash = hashlib.sha256(content).hexdigest()[:8]
            target_name = (parent / f"{stem}-{short_hash}{suffix}").as_posix()
        assets[target_name] = content
        known_names.add(name)
    return assets, known_names


def _html_to_markdown(
    raw_html: bytes | str,
    *,
    href: str,
    image_names: set[str],
) -> tuple[str, str | None]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise EbookLibUnavailableError(
            "BeautifulSoup is required for EPUB chapter parsing."
        ) from exc

    soup = BeautifulSoup(raw_html, "html.parser")
    for unwanted in soup(["script", "style"]):
        unwanted.decompose()
    for image in soup.find_all("img"):
        source = image.get("src")
        resolved = _resolve_epub_reference(href, source) if source else None
        if resolved and resolved in image_names:
            image["src"] = f"assets/{resolved}"

    heading = soup.find(["h1", "h2", "title"])
    chapter_title = heading.get_text(" ", strip=True) if heading else None
    try:
        from markdownify import markdownify

        markdown = markdownify(str(soup.body or soup), heading_style="ATX")
    except ImportError:
        # EbookLib support remains functional in constrained installs, albeit
        # with plain text rather than rich Markdown.
        markdown = soup.get_text("\n", strip=True)
    return markdown, chapter_title


def _item_html(item: Any) -> bytes | str:
    body_getter = getattr(item, "get_body_content", None)
    if callable(body_getter):
        try:
            return body_getter()
        except Exception:
            pass
    content_getter = getattr(item, "get_content", None)
    return content_getter() if callable(content_getter) else b""


def _item_name(item: Any) -> str:
    for attribute in ("file_name", "href", "name"):
        value = getattr(item, attribute, None)
        if value:
            return str(value)
    getter = getattr(item, "get_name", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            return ""
    return ""


def _item_identity(item: Any) -> str:
    getter = getattr(item, "get_id", None)
    if callable(getter):
        try:
            value = getter()
            if value:
                return str(value)
        except Exception:
            pass
    return _item_name(item) or f"object:{id(item)}"


def _is_item_type(item: Any, expected: int) -> bool:
    getter = getattr(item, "get_type", None)
    if not callable(getter):
        return True
    try:
        return getter() == expected
    except Exception:
        return False


def _is_chapter_item(item: Any) -> bool:
    checker = getattr(item, "is_chapter", None)
    if not callable(checker):
        return True
    try:
        return bool(checker())
    except Exception:
        return True


def _resolve_epub_reference(chapter_href: str, reference: str) -> str | None:
    parsed = urlsplit(reference)
    if parsed.scheme or not parsed.path:
        return None
    path = unquote(parsed.path)
    if path.startswith("/"):
        resolved = path.lstrip("/")
    else:
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(chapter_href), path))
    return _safe_epub_path(resolved)


def _safe_epub_path(value: str) -> str | None:
    normalized = posixpath.normpath(str(value).replace("\\", "/")).lstrip("/")
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        return None
    return PurePosixPath(normalized).as_posix()


def _normalize_generated_markdown(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


__all__ = [
    "EbookLibUnavailableError",
    "EpubConversionError",
    "convert_epub",
]
