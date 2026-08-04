"""Data contracts used by the cross-ontology synthesis subsystem.

The canonical source and frame contracts live in :mod:`ke.contracts`.  The
models in this module describe artifacts that only exist while integrating
multiple canonical ontologies.  They intentionally reference canonical IDs
instead of copying or replacing source claims.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ke.contracts import AgentFrame


class SynthesisModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


MappingKind = Literal[
    "exact_equivalent",
    "broader",
    "narrower",
    "related_or_analogous",
    "superficially_similar_but_distinct",
    "conflicting",
]


class ConceptMapping(SynthesisModel):
    """A source-preserving relationship between concepts in two ontologies."""

    id: str
    left_ontology_id: str
    left_term_id: str
    right_ontology_id: str
    right_term_id: str
    kind: MappingKind
    rationale: str
    confidence: float = Field(ge=0, le=1)
    support_ids: list[str] = Field(default_factory=list)


class Complementarity(SynthesisModel):
    """A useful combination that does not assert equivalence."""

    id: str
    ontology_ids: list[str] = Field(min_length=2)
    claim_ids: list[str] = Field(min_length=2)
    description: str


class AlignmentReport(SynthesisModel):
    schema_version: str = "1.0"
    ontology_ids: list[str]
    mappings: list[ConceptMapping] = Field(default_factory=list)
    complementarities: list[Complementarity] = Field(default_factory=list)
    method: str = "deterministic lexical alignment"


class IntegratedClaim(SynthesisModel):
    """A claim authored by synthesis, kept separate from source claims."""

    id: str
    statement: str
    source_claim_ids: list[str] = Field(min_length=1)
    derivation: str
    confidence: float = Field(ge=0, le=1)


class IntegratedOntology(SynthesisModel):
    """JSON-friendly representation of a provenance-preserving RDF dataset."""

    schema_version: str = "1.0"
    dataset_id: str
    ontology_ids: list[str]
    source_graphs: dict[str, dict[str, Any]]
    integration_graph_id: str
    mappings: list[ConceptMapping] = Field(default_factory=list)
    complementarities: list[Complementarity] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    synthesized_claims: list[IntegratedClaim] = Field(default_factory=list)


FindingSeverity = Literal["error", "warning", "info"]


class CriticFinding(SynthesisModel):
    code: str
    severity: FindingSeverity
    message: str
    object_id: str | None = None
    repairable: bool = False


class CriticPass(SynthesisModel):
    approved: bool
    findings: list[CriticFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def approval_matches_findings(self) -> CriticPass:
        if self.approved and any(finding.severity == "error" for finding in self.findings):
            raise ValueError("a critic pass with error findings cannot be approved")
        return self


class ReviewReport(SynthesisModel):
    schema_version: str = "1.0"
    approved: bool
    repairs_attempted: int = Field(default=0, ge=0, le=1)
    initial: CriticPass
    final: CriticPass


class StanceLedgerEntry(SynthesisModel):
    frame_object_id: str
    kind: Literal["belief", "desire", "intention"]
    origin: str
    parent_desire_ids: list[str] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    support_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    note: str


class StanceLedger(SynthesisModel):
    schema_version: str = "1.0"
    frame_id: str
    stance_is_instrumental: bool = True
    sentience_claim: bool = False
    entries: list[StanceLedgerEntry]


class SynthesisBundle(SynthesisModel):
    """Return value for a completed intentional synthesis run."""

    frame_id: str
    synthesis_key: str
    cache_hit: bool = False
    output_dir: str
    alignment: AlignmentReport
    integrated: IntegratedOntology
    review: ReviewReport
    frame: AgentFrame
