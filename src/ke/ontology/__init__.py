"""Source-grounded domain ontology extraction and publication."""

from ke.ontology.export import ExportResult, export_ontology, export_rdf
from ke.ontology.extract import (
    ChunkView,
    OntologyFragment,
    extract_chunk,
    extract_chunk_sync,
    extract_chunks,
    ground_fragment,
    offline_extract,
)
from ke.ontology.reduce import (
    ReductionBudget,
    canonicalize_ids,
    hierarchical_reduce,
    reduce_fragments,
)
from ke.ontology.resolve import (
    EntityCluster,
    ResolutionDecision,
    ResolutionResult,
    normalize_label,
    resolve_entities,
)

__all__ = [
    "ChunkView",
    "EntityCluster",
    "ExportResult",
    "OntologyFragment",
    "ReductionBudget",
    "ResolutionDecision",
    "ResolutionResult",
    "canonicalize_ids",
    "export_ontology",
    "export_rdf",
    "extract_chunk",
    "extract_chunk_sync",
    "extract_chunks",
    "ground_fragment",
    "hierarchical_reduce",
    "normalize_label",
    "offline_extract",
    "reduce_fragments",
    "resolve_entities",
]
