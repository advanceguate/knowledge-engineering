"""Public ingestion API."""

from .docling import DoclingConversionError, DoclingUnavailableError, convert_with_docling
from .epub import EbookLibUnavailableError, EpubConversionError, convert_epub
from .models import ConvertedDocument, SourceBundle, SourceFingerprint
from .router import (
    SUPPORTED_EXTENSIONS,
    UnsupportedSourceFormatError,
    fingerprint_source,
    ingest,
    ingest_source,
    route_source,
)

__all__ = [
    "ConvertedDocument",
    "DoclingConversionError",
    "DoclingUnavailableError",
    "EbookLibUnavailableError",
    "EpubConversionError",
    "SUPPORTED_EXTENSIONS",
    "SourceBundle",
    "SourceFingerprint",
    "UnsupportedSourceFormatError",
    "convert_epub",
    "convert_with_docling",
    "fingerprint_source",
    "ingest",
    "ingest_source",
    "route_source",
]
