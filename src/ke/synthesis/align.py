"""Cross-ontology concept alignment without source graph collapse."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher
from itertools import combinations

from ke.contracts import DomainOntology, Term
from ke.llm import ModelGateway
from ke.resource_paths import read_resource_text

from .models import AlignmentReport, Complementarity, ConceptMapping, MappingKind

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_NEGATION_WORDS = {"no", "not", "never", "without", "cannot"}
_OPPOSED_WORD_PAIRS = {
    frozenset({"increase", "decrease"}),
    frozenset({"increases", "decreases"}),
    frozenset({"improve", "harm"}),
    frozenset({"improves", "harms"}),
    frozenset({"enable", "prevent"}),
    frozenset({"enables", "prevents"}),
    frozenset({"effective", "ineffective"}),
    frozenset({"beneficial", "detrimental"}),
}


def normalize_label(value: str) -> str:
    """Normalize a label for candidate generation, never for identity assignment."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_TOKEN_RE.findall(normalized))


def _stable_id(prefix: str, *parts: str) -> str:
    material = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(material).hexdigest()[:16]}"


def _label_forms(term: Term) -> set[str]:
    return {
        normalized
        for label in [term.preferred_label, *term.aliases]
        if (normalized := normalize_label(label))
    }


def _definition_similarity(left: Term, right: Term) -> float:
    return SequenceMatcher(
        None,
        normalize_label(left.definition),
        normalize_label(right.definition),
    ).ratio()


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(normalize_label(left).split())
    right_tokens = set(normalize_label(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _definitions_conflict(left: str, right: str) -> bool:
    left_tokens = set(normalize_label(left).split())
    right_tokens = set(normalize_label(right).split())
    left_negated = bool(left_tokens & _NEGATION_WORDS)
    right_negated = bool(right_tokens & _NEGATION_WORDS)
    left_content = left_tokens - _NEGATION_WORDS
    right_content = right_tokens - _NEGATION_WORDS
    overlap = (
        len(left_content & right_content) / len(left_content | right_content)
        if left_content | right_content
        else 0.0
    )
    opposed = any(
        bool(left_hits := pair & left_tokens)
        and bool(right_hits := pair & right_tokens)
        and left_hits.isdisjoint(right_hits)
        for pair in _OPPOSED_WORD_PAIRS
    )
    return overlap >= 0.50 and (left_negated != right_negated or opposed)


def _strict_label_contains(container: str, contained: str) -> bool:
    container_tokens = normalize_label(container).split()
    contained_tokens = normalize_label(contained).split()
    if len(container_tokens) <= len(contained_tokens) or not contained_tokens:
        return False
    width = len(contained_tokens)
    return any(
        container_tokens[index : index + width] == contained_tokens
        for index in range(len(container_tokens) - width + 1)
    )


def _definitions_compatible(left: Term, right: Term) -> bool:
    return not _definitions_conflict(left.definition, right.definition) and (
        _token_overlap(left.definition, right.definition) >= 0.35
        or _definition_similarity(left, right) >= 0.55
    )


def _classify_pair(left: Term, right: Term) -> tuple[MappingKind, str, float] | None:
    """Classify only high-signal lexical candidates.

    Lexical equality generates candidates, but definitions adjudicate them.  In
    particular, equal surface labels with divergent definitions are explicitly
    retained as distinct instead of being silently merged.
    """

    left_forms = _label_forms(left)
    right_forms = _label_forms(right)
    shared_forms = left_forms & right_forms
    definition_similarity = _definition_similarity(left, right)

    if shared_forms:
        canonical_equal = normalize_label(left.preferred_label) == normalize_label(
            right.preferred_label
        )
        if canonical_equal and _definitions_conflict(left.definition, right.definition):
            return (
                "conflicting",
                "The preferred labels match, but the source definitions make opposed claims.",
                0.9,
            )
        if definition_similarity >= 0.64:
            return (
                "exact_equivalent",
                "Shared normalized label or alias with materially compatible definitions.",
                min(0.99, 0.72 + (definition_similarity * 0.27)),
            )
        if canonical_equal and definition_similarity < 0.42:
            return (
                "superficially_similar_but_distinct",
                "The preferred labels match, but the source definitions materially differ.",
                min(0.95, 0.70 + ((1 - definition_similarity) * 0.20)),
            )
        return (
            "related_or_analogous",
            "A label or alias overlaps, but the definitions do not justify equivalence.",
            0.62,
        )

    label_overlap = _token_overlap(left.preferred_label, right.preferred_label)
    definition_overlap = _token_overlap(left.definition, right.definition)
    if _definitions_compatible(left, right):
        if _strict_label_contains(left.preferred_label, right.preferred_label):
            return (
                "narrower",
                "The left preferred label strictly specializes the right label and the source "
                "definitions are compatible.",
                0.82,
            )
        if _strict_label_contains(right.preferred_label, left.preferred_label):
            return (
                "broader",
                "The right preferred label strictly specializes the left label and the source "
                "definitions are compatible.",
                0.82,
            )
    if label_overlap >= 0.66 and definition_overlap >= 0.30:
        return (
            "related_or_analogous",
            "The labels and source definitions overlap enough to warrant a related mapping.",
            min(0.88, 0.55 + (label_overlap * 0.18) + (definition_overlap * 0.15)),
        )

    return None


def _coerce_gateway_mappings(
    gateway: object,
    ontologies: list[DomainOntology],
) -> list[ConceptMapping] | None:
    """Use an explicit alignment hook when a caller supplies one.

    Keeping the hook narrow makes fake gateways straightforward while leaving
    provider-specific prompting in the shared LLM layer.
    """

    align = getattr(gateway, "align_ontologies", None)
    if not callable(align):
        return None
    result = align(ontologies)
    if isinstance(result, AlignmentReport):
        return result.mappings
    if isinstance(result, dict):
        result = result.get("mappings", [])
    if isinstance(result, Iterable):
        return [
            item if isinstance(item, ConceptMapping) else ConceptMapping.model_validate(item)
            for item in result
        ]
    raise TypeError("align_ontologies gateway hook must return mappings or an AlignmentReport")


def align_ontologies(
    ontologies: Iterable[DomainOntology],
    *,
    gateway: object | None = None,
) -> AlignmentReport:
    """Align concepts across ontologies while retaining their canonical IDs."""

    ontology_list = list(ontologies)
    if len(ontology_list) < 1:
        raise ValueError("At least one ontology is required for synthesis")

    gateway_mappings = (
        _coerce_gateway_mappings(gateway, ontology_list) if gateway is not None else None
    )
    mappings: list[ConceptMapping]
    if gateway_mappings is not None:
        mappings = gateway_mappings
        method = "configured model gateway with canonical reference validation"
    else:
        mappings = []
        for left_ontology, right_ontology in combinations(ontology_list, 2):
            for left_term in left_ontology.terms:
                for right_term in right_ontology.terms:
                    classification = _classify_pair(left_term, right_term)
                    if classification is None:
                        continue
                    kind, rationale, confidence = classification
                    mappings.append(
                        ConceptMapping(
                            id=_stable_id(
                                "mapping",
                                left_ontology.ontology_id,
                                left_term.id,
                                right_ontology.ontology_id,
                                right_term.id,
                                kind,
                            ),
                            left_ontology_id=left_ontology.ontology_id,
                            left_term_id=left_term.id,
                            right_ontology_id=right_ontology.ontology_id,
                            right_term_id=right_term.id,
                            kind=kind,
                            rationale=rationale,
                            confidence=confidence,
                            support_ids=[left_term.id, right_term.id],
                        )
                    )
        method = "deterministic lexical candidate generation and definition comparison"

    _validate_mapping_references(ontology_list, mappings)
    mappings.sort(key=lambda item: item.id)
    complementarities = [
        Complementarity(
            id=_stable_id("complementarity", mapping.id),
            ontology_ids=[mapping.left_ontology_id, mapping.right_ontology_id],
            claim_ids=[mapping.left_term_id, mapping.right_term_id],
            description=(
                "The concepts are potentially useful together but remain source-distinct: "
                f"{mapping.rationale}"
            ),
        )
        for mapping in mappings
        if mapping.kind == "related_or_analogous"
    ]
    return AlignmentReport(
        ontology_ids=[ontology.ontology_id for ontology in ontology_list],
        mappings=mappings,
        complementarities=complementarities,
        method=method,
    )


def _validate_mapping_references(
    ontologies: list[DomainOntology],
    mappings: list[ConceptMapping],
) -> None:
    terms_by_ontology = {
        ontology.ontology_id: {term.id for term in ontology.terms} for ontology in ontologies
    }
    for mapping in mappings:
        if mapping.left_term_id not in terms_by_ontology.get(mapping.left_ontology_id, set()):
            raise ValueError(f"Dangling left term reference in mapping {mapping.id}")
        if mapping.right_term_id not in terms_by_ontology.get(mapping.right_ontology_id, set()):
            raise ValueError(f"Dangling right term reference in mapping {mapping.id}")
        if mapping.left_ontology_id == mapping.right_ontology_id:
            raise ValueError(f"Mapping {mapping.id} must connect separate source ontologies")


async def align_ontologies_with_gateway(
    ontologies: Iterable[DomainOntology],
    *,
    gateway: ModelGateway | None,
    model: str | None = None,
) -> AlignmentReport:
    """Use the structured model gateway when provided, with strict reference checks."""

    ontology_list = list(ontologies)
    if gateway is None:
        return align_ontologies(ontology_list)
    if not ontology_list:
        raise ValueError("At least one ontology is required for synthesis")
    instructions = read_resource_text("prompts", "ontology_alignment.md")
    payload = {
        "source_ontologies": [
            {
                "ontology_id": ontology.ontology_id,
                "title": ontology.title,
                "terms": [term.model_dump(mode="json") for term in ontology.terms],
                "axioms": [axiom.model_dump(mode="json") for axiom in ontology.axioms],
            }
            for ontology in ontology_list
        ]
    }
    report = await gateway.structured(
        json.dumps(payload, ensure_ascii=False, indent=2),
        AlignmentReport,
        model=model,
        system_prompt=instructions,
    )
    if set(report.ontology_ids) != {ontology.ontology_id for ontology in ontology_list}:
        raise ValueError("Model alignment omitted or invented an ontology ID")
    _validate_mapping_references(ontology_list, report.mappings)
    known_claim_ids = {term.id for ontology in ontology_list for term in ontology.terms}
    known_ontology_ids = {ontology.ontology_id for ontology in ontology_list}
    for complementarity in report.complementarities:
        if not set(complementarity.ontology_ids) <= known_ontology_ids:
            raise ValueError(f"Complementarity {complementarity.id} invented an ontology reference")
        if not set(complementarity.claim_ids) <= known_claim_ids:
            raise ValueError(f"Complementarity {complementarity.id} has a dangling claim reference")
    return report
