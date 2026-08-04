"""Lazy Docling adapter for PDF, HTML, and DOCX ingestion.

Docling is deliberately imported only when a conversion is requested.  This
keeps unit tests and Markdown-only workflows usable on machines where Docling's
native/model dependencies have not been installed yet.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from .common import jsonable
from .models import ConvertedDocument


class DoclingUnavailableError(RuntimeError):
    """Raised when a Docling-backed format is requested without Docling."""


class DoclingConversionError(RuntimeError):
    """Raised when Docling cannot produce a usable canonical document."""


def convert_with_docling(
    source_path: str | Path,
    *,
    converter: Any | None = None,
    ocr: str | bool = "auto",
    table_structure: str = "accurate",
    export_images: bool = True,
    assets_dir: str | Path | None = None,
) -> ConvertedDocument:
    """Convert a PDF, HTML, or DOCX file into Markdown and Docling JSON.

    ``converter`` is an explicit injection seam for tests and for applications
    that maintain a pre-warmed ``DocumentConverter``.  It may expose ``convert``
    or be directly callable.
    """

    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source does not exist: {path}")

    active_converter = converter or _build_default_converter(
        path,
        ocr=ocr,
        table_structure=table_structure,
        export_images=export_images,
    )
    try:
        if hasattr(active_converter, "convert"):
            result = active_converter.convert(str(path))
        elif callable(active_converter):
            result = active_converter(str(path))
        else:
            raise TypeError("converter must be callable or expose convert(path)")
    except (DoclingUnavailableError, DoclingConversionError):
        raise
    except Exception as exc:  # Docling has several backend-specific exception types.
        raise DoclingConversionError(f"Docling failed to convert {path.name}: {exc}") from exc

    status = getattr(result, "status", None)
    status_text = str(getattr(status, "value", status) or "").lower()
    if status_text in {"failure", "failed", "error"}:
        detail = getattr(result, "errors", None) or "conversion status was failure"
        raise DoclingConversionError(f"Docling failed to convert {path.name}: {detail}")

    document = _result_document(result)
    markdown = _result_markdown(result, document)
    canonical = _result_canonical(result, document)
    if not isinstance(canonical, dict):
        raise DoclingConversionError(f"Docling returned no canonical JSON document for {path.name}")

    asset_bytes: dict[str, bytes] = {}
    supplied_assets = _mapping_value(result, "assets")
    if isinstance(supplied_assets, Mapping):
        for name, value in supplied_assets.items():
            normalized = _safe_asset_name(str(name))
            if normalized and isinstance(value, (bytes, bytearray, memoryview)):
                asset_bytes[normalized] = bytes(value)
    if export_images and document is not None:
        asset_bytes.update(_extract_picture_assets(document))

    if assets_dir is not None and asset_bytes:
        target = Path(assets_dir)
        target.mkdir(parents=True, exist_ok=True)
        for relative_name, value in asset_bytes.items():
            destination = target / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(value)

    canonical = jsonable(canonical)

    return ConvertedDocument(
        markdown=markdown,
        canonical_document=canonical,
        backend="docling",
        backend_version=_package_version("docling"),
        assets=asset_bytes,
        docling_document=document,
    )


def _build_default_converter(
    source_path: Path,
    *,
    ocr: str | bool,
    table_structure: str,
    export_images: bool,
) -> Any:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise DoclingUnavailableError(
            "Docling is required for PDF, HTML, and DOCX ingestion. "
            "Install the project dependencies or inject a converter for tests."
        ) from exc

    if source_path.suffix.lower() != ".pdf":
        return DocumentConverter()

    # Keep the integration compatible across the supported Docling 2.x range.
    # If an older/minimal build lacks configuration classes, the default
    # converter is still fully functional and preferable to failing import.
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = _ocr_enabled(ocr)
        options.do_table_structure = str(table_structure).lower() not in {
            "off",
            "false",
            "none",
        }
        if hasattr(options, "generate_page_images"):
            options.generate_page_images = bool(export_images)
        if hasattr(options, "generate_picture_images"):
            options.generate_picture_images = bool(export_images)
        if options.do_table_structure and hasattr(options, "table_structure_options"):
            requested_mode = str(table_structure).upper()
            mode = getattr(TableFormerMode, requested_mode, None)
            if mode is None and requested_mode == "ACCURATE":
                mode = getattr(TableFormerMode, "ACCURATE", None)
            if mode is not None:
                options.table_structure_options.mode = mode
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        return DocumentConverter()


def _ocr_enabled(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"off", "false", "never", "none", "0"}


def _result_document(result: Any) -> Any | None:
    if isinstance(result, Mapping):
        return result.get("document")
    return getattr(result, "document", None)


def _result_markdown(result: Any, document: Any | None) -> str:
    supplied = _mapping_value(result, "markdown")
    if isinstance(supplied, str):
        return supplied
    if document is not None:
        exporter = getattr(document, "export_to_markdown", None)
        if callable(exporter):
            try:
                value = exporter()
            except Exception as exc:
                raise DoclingConversionError(f"Could not export Docling Markdown: {exc}") from exc
            if isinstance(value, str):
                return value
    raise DoclingConversionError("Docling returned no Markdown export")


def _result_canonical(result: Any, document: Any | None) -> dict[str, Any] | None:
    for key in ("canonical_document", "document_json", "docling_json"):
        supplied = _mapping_value(result, key)
        parsed = _coerce_json_object(supplied)
        if parsed is not None:
            return parsed

    if document is None:
        return None
    for method_name in ("export_to_dict", "model_dump", "dict"):
        method = getattr(document, method_name, None)
        if not callable(method):
            continue
        try:
            value = method(mode="json") if method_name == "model_dump" else method()
        except TypeError:
            value = method()
        except Exception:
            continue
        parsed = _coerce_json_object(value)
        if parsed is not None:
            return parsed

    exporter = getattr(document, "export_to_json", None)
    if callable(exporter):
        try:
            return _coerce_json_object(exporter())
        except Exception:
            return None
    return None


def _coerce_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _mapping_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _extract_picture_assets(document: Any) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    pictures = getattr(document, "pictures", None)
    if not pictures:
        return assets
    for index, picture in enumerate(pictures, start=1):
        getter = getattr(picture, "get_image", None)
        if not callable(getter):
            continue
        try:
            image = getter(document)
            if image is None:
                continue
            output = io.BytesIO()
            image.save(output, format="PNG")
            assets[f"picture-{index:04d}.png"] = output.getvalue()
        except Exception:
            # Image generation is optional even when the canonical conversion
            # succeeds (for example, some Docling backends omit page images).
            continue
    return assets


def _safe_asset_name(value: str) -> str | None:
    candidate = Path(value.replace("\\", "/"))
    clean_parts = [part for part in candidate.parts if part not in {"", ".", "..", "/"}]
    if not clean_parts:
        return None
    return Path(*clean_parts).as_posix()


def _package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


__all__ = [
    "DoclingConversionError",
    "DoclingUnavailableError",
    "convert_with_docling",
]
