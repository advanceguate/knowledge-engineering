"""Conflict-preserving integration and named-graph serialization."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from itertools import combinations
from typing import Any, Literal
from urllib.parse import quote

from ke.contracts import AgentFrame, Axiom, ClaimConflict, DomainOntology, Evidence
from ke.semantic.rdf import DEFAULT_BASE_URI, ontology_to_graph

from .models import AlignmentReport, IntegratedClaim, IntegratedOntology

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "when",
    "with",
}
_NEGATIONS = {"no", "not", "never", "without", "cannot", "shouldnt", "mustnt"}
_POLAR_TERMS: dict[str, tuple[str, int]] = {
    "improve": ("effect", 1),
    "improves": ("effect", 1),
    "improved": ("effect", 1),
    "enhance": ("effect", 1),
    "enhances": ("effect", 1),
    "support": ("effect", 1),
    "supports": ("effect", 1),
    "enable": ("effect", 1),
    "enables": ("effect", 1),
    "promote": ("effect", 1),
    "promotes": ("effect", 1),
    "effective": ("effect", 1),
    "beneficial": ("effect", 1),
    "harm": ("effect", -1),
    "harms": ("effect", -1),
    "worsen": ("effect", -1),
    "worsens": ("effect", -1),
    "undermine": ("effect", -1),
    "undermines": ("effect", -1),
    "inhibit": ("effect", -1),
    "inhibits": ("effect", -1),
    "prevent": ("effect", -1),
    "prevents": ("effect", -1),
    "ineffective": ("effect", -1),
    "detrimental": ("effect", -1),
    "increase": ("quantity", 1),
    "increases": ("quantity", 1),
    "raise": ("quantity", 1),
    "raises": ("quantity", 1),
    "decrease": ("quantity", -1),
    "decreases": ("quantity", -1),
    "reduce": ("quantity", -1),
    "reduces": ("quantity", -1),
    "lower": ("quantity", -1),
    "lowers": ("quantity", -1),
}


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _statement_signature(statement: str) -> tuple[set[str], dict[str, int], bool]:
    tokens = _WORD_RE.findall(statement.casefold().replace("'", ""))
    negated = any(token in _NEGATIONS for token in tokens)
    dimensions: dict[str, int] = {}
    content: set[str] = set()
    for token in tokens:
        if token in _STOP_WORDS or token in _NEGATIONS:
            continue
        if token in _POLAR_TERMS:
            dimension, polarity = _POLAR_TERMS[token]
            dimensions[dimension] = -polarity if negated else polarity
            continue
        content.add(_light_stem(token))
    return content, dimensions, negated


def _light_stem(token: str) -> str:
    for suffix in ("ization", "ation", "ments", "ment", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def _axioms_conflict(left: Axiom, right: Axiom) -> bool:
    left_content, left_dimensions, left_negated = _statement_signature(left.statement)
    right_content, right_dimensions, right_negated = _statement_signature(right.statement)
    common = left_content & right_content
    union = left_content | right_content
    overlap = len(common) / len(union) if union else 0.0
    same_dimension_opposed = any(
        dimension in right_dimensions and polarity == -right_dimensions[dimension]
        for dimension, polarity in left_dimensions.items()
    )
    direct_negation = left_negated != right_negated
    return len(common) >= 2 and overlap >= 0.40 and (same_dimension_opposed or direct_negation)


def _conflict_kind(
    left: Axiom, right: Axiom
) -> Literal["empirical", "causal", "normative", "methodological"]:
    modalities = {left.modality, right.modality}
    if "normative" in modalities:
        return "normative"
    if "methodological" in modalities or "heuristic" in modalities:
        return "methodological"
    if "causal" in modalities:
        return "causal"
    return "empirical"


def identify_conflicts(
    ontologies: Iterable[DomainOntology],
    alignment: AlignmentReport | None = None,
) -> list[ClaimConflict]:
    """Return all source-declared and detectable cross-source conflicts.

    Source-declared conflicts are copied verbatim.  Cross-source detection is
    intentionally conservative: it requires substantial topic overlap plus an
    explicit polarity or negation mismatch.
    """

    ontology_list = list(ontologies)
    conflicts: list[ClaimConflict] = []
    seen: set[str] = set()

    for ontology in ontology_list:
        for conflict in ontology.conflicts:
            candidate = conflict.model_copy(deep=True)
            if candidate.id in seen:
                candidate.id = _stable_id("conflict", ontology.ontology_id, candidate.id)
            seen.add(candidate.id)
            conflicts.append(candidate)

    for left_ontology, right_ontology in combinations(ontology_list, 2):
        for left_axiom in left_ontology.axioms:
            for right_axiom in right_ontology.axioms:
                if not _axioms_conflict(left_axiom, right_axiom):
                    continue
                conflict_id = _stable_id(
                    "conflict",
                    left_ontology.ontology_id,
                    left_axiom.id,
                    right_ontology.ontology_id,
                    right_axiom.id,
                )
                if conflict_id in seen:
                    continue
                seen.add(conflict_id)
                conflicts.append(
                    ClaimConflict(
                        id=conflict_id,
                        claim_ids=[left_axiom.id, right_axiom.id],
                        kind=_conflict_kind(left_axiom, right_axiom),
                        description=(
                            f"{left_ontology.title} and {right_ontology.title} make "
                            "materially opposed claims about an overlapping topic."
                        ),
                        resolution="unresolved",
                    )
                )

    if alignment is not None:
        for mapping in alignment.mappings:
            if mapping.kind != "conflicting":
                continue
            conflict_id = _stable_id("conflict", mapping.id)
            if conflict_id in seen:
                continue
            seen.add(conflict_id)
            conflicts.append(
                ClaimConflict(
                    id=conflict_id,
                    claim_ids=[mapping.left_term_id, mapping.right_term_id],
                    kind="terminological",
                    description=mapping.rationale,
                    resolution="unresolved",
                )
            )

    conflicts.sort(key=lambda item: item.id)
    return conflicts


def build_integrated_named_graph(
    ontologies: Iterable[DomainOntology],
    alignment: AlignmentReport,
    conflicts: Iterable[ClaimConflict] | None = None,
) -> IntegratedOntology:
    """Build an immutable source-graph set plus a separate integration graph."""

    ontology_list = list(ontologies)
    ontology_ids = [ontology.ontology_id for ontology in ontology_list]
    if ontology_ids != alignment.ontology_ids:
        if set(ontology_ids) != set(alignment.ontology_ids):
            raise ValueError("Alignment ontology IDs do not match integration inputs")
    dataset_id = _stable_id("integrated", *sorted(ontology_ids))
    source_graphs = {
        source_graph_id(ontology.ontology_id): ontology.model_dump(mode="json")
        for ontology in ontology_list
    }
    conflict_list = (
        list(conflicts) if conflicts is not None else identify_conflicts(ontology_list, alignment)
    )
    return IntegratedOntology(
        dataset_id=dataset_id,
        ontology_ids=ontology_ids,
        source_graphs=source_graphs,
        integration_graph_id=f"urn:ke:graph:integration:{dataset_id}",
        mappings=alignment.mappings,
        complementarities=alignment.complementarities,
        conflicts=[conflict.model_dump(mode="json") for conflict in conflict_list],
        synthesized_claims=[],
    )


def source_graph_id(ontology_id: str) -> str:
    return f"urn:ke:graph:source:{ontology_id}"


def materialize_synthesized_beliefs(
    integrated: IntegratedOntology,
    frame: AgentFrame,
) -> IntegratedOntology:
    """Materialize derived frame beliefs in the separate integration graph."""

    source_claim_ids = set(_claim_locations(integrated))
    existing = {claim.id: claim for claim in integrated.synthesized_claims}
    for belief in frame.beliefs:
        if belief.origin != "derived" and belief.status != "synthesized":
            continue
        if belief.id in source_claim_ids:
            raise ValueError(
                f"Synthesized belief {belief.id!r} collides with a canonical source claim ID"
            )
        dangling = set(belief.support_ids) - source_claim_ids
        if not belief.support_ids or dangling:
            raise ValueError(
                f"Synthesized belief {belief.id!r} has invalid source supports: {dangling}"
            )
        existing[belief.id] = IntegratedClaim(
            id=belief.id,
            statement=belief.proposition,
            source_claim_ids=belief.support_ids,
            derivation=(
                f"Materialized from AgentFrame belief {belief.id}; origin={belief.origin}; "
                f"status={belief.status}."
            ),
            confidence=belief.confidence,
        )
    return integrated.model_copy(
        update={"synthesized_claims": sorted(existing.values(), key=lambda item: item.id)}
    )


def _source_base_uri(ontology_id: str) -> str:
    return f"{DEFAULT_BASE_URI}integrated/{quote(ontology_id, safe='')}/"


def _scoped_claim_uri(ontology_id: str, claim_id: str) -> str:
    return f"{_source_base_uri(ontology_id)}resource/{quote(claim_id, safe='')}"


def _claim_locations(integrated: IntegratedOntology) -> dict[str, list[tuple[str, str]]]:
    locations: dict[str, list[tuple[str, str]]] = {}
    for ontology in integrated.source_graphs.values():
        ontology_id = ontology["ontology_id"]
        for key in ("terms", "relations", "axioms", "mental_models"):
            for item in ontology.get(key, []):
                locations.setdefault(item["id"], []).append(
                    (ontology_id, _scoped_claim_uri(ontology_id, item["id"]))
                )
    return locations


def _integrated_rdf_dataset(integrated: IntegratedOntology) -> Any:
    try:
        from rdflib import RDF, Dataset, Literal, Namespace, URIRef
        from rdflib.namespace import DCTERMS, SKOS, XSD
    except ImportError as exc:  # pragma: no cover - dependency is declared by the package
        raise RuntimeError("rdflib is required to serialize the integrated ontology") from exc

    dataset = Dataset()
    ke = Namespace(DEFAULT_BASE_URI)
    dataset.bind("ke", ke)
    dataset.bind("skos", SKOS)
    dataset.bind("dcterms", DCTERMS)
    locations = _claim_locations(integrated)

    for graph_id, ontology_data in integrated.source_graphs.items():
        ontology = DomainOntology.model_validate(ontology_data)
        source_graph = ontology_to_graph(
            ontology,
            base_uri=_source_base_uri(ontology.ontology_id),
        )
        target_graph = dataset.graph(URIRef(graph_id))
        for triple in source_graph:
            target_graph.add(triple)

    graph = dataset.graph(URIRef(integrated.integration_graph_id))
    integration_base = f"{DEFAULT_BASE_URI}integration/{quote(integrated.dataset_id, safe='')}/"
    for mapping in integrated.mappings:
        ref = URIRef(f"{integration_base}mapping/{quote(mapping.id, safe='')}")
        left_ref = URIRef(_scoped_claim_uri(mapping.left_ontology_id, mapping.left_term_id))
        right_ref = URIRef(_scoped_claim_uri(mapping.right_ontology_id, mapping.right_term_id))
        graph.add((ref, RDF.type, ke.ConceptMapping))
        graph.add((ref, DCTERMS.identifier, Literal(mapping.id)))
        graph.add((ref, ke.mappingKind, Literal(mapping.kind)))
        graph.add((ref, ke.leftConcept, left_ref))
        graph.add((ref, ke.rightConcept, right_ref))
        graph.add((ref, ke.leftOntologyId, Literal(mapping.left_ontology_id)))
        graph.add((ref, ke.rightOntologyId, Literal(mapping.right_ontology_id)))
        graph.add((ref, DCTERMS.description, Literal(mapping.rationale)))
        graph.add((ref, ke.confidence, Literal(mapping.confidence, datatype=XSD.decimal)))
        for support_id in mapping.support_ids:
            for _, support_uri in locations.get(support_id, []):
                graph.add((ref, ke.supportClaim, URIRef(support_uri)))
        mapping_predicate = {
            "exact_equivalent": SKOS.exactMatch,
            # Mapping kind describes the left concept relative to the right.
            # SKOS broadMatch points from a narrower subject to its broader
            # external concept, while narrowMatch points in the inverse direction.
            "broader": SKOS.narrowMatch,
            "narrower": SKOS.broadMatch,
            "related_or_analogous": SKOS.relatedMatch,
        }.get(mapping.kind)
        if mapping_predicate is not None:
            graph.add((left_ref, mapping_predicate, right_ref))

    for conflict in integrated.conflicts:
        ref = URIRef(f"{integration_base}conflict/{quote(conflict['id'], safe='')}")
        graph.add((ref, RDF.type, ke.ClaimConflict))
        graph.add((ref, DCTERMS.identifier, Literal(conflict["id"])))
        graph.add((ref, DCTERMS.description, Literal(conflict["description"])))
        graph.add((ref, ke.conflictKind, Literal(conflict["kind"])))
        graph.add((ref, ke.resolution, Literal(conflict["resolution"])))
        for claim_id in conflict["claim_ids"]:
            graph.add((ref, ke.conflictingClaimId, Literal(claim_id)))
            for _, claim_uri in locations.get(claim_id, []):
                graph.add((ref, ke.conflictingClaim, URIRef(claim_uri)))

    for item in integrated.complementarities:
        ref = URIRef(f"{integration_base}complementarity/{quote(item.id, safe='')}")
        graph.add((ref, RDF.type, ke.Complementarity))
        graph.add((ref, DCTERMS.identifier, Literal(item.id)))
        graph.add((ref, DCTERMS.description, Literal(item.description)))
        for ontology_id in item.ontology_ids:
            graph.add((ref, ke.ontologyId, Literal(ontology_id)))
        for claim_id in item.claim_ids:
            for _, claim_uri in locations.get(claim_id, []):
                graph.add((ref, ke.sourceClaim, URIRef(claim_uri)))

    for claim in integrated.synthesized_claims:
        ref = URIRef(f"{integration_base}claim/{quote(claim.id, safe='')}")
        graph.add((ref, RDF.type, ke.SynthesizedClaim))
        graph.add((ref, DCTERMS.identifier, Literal(claim.id)))
        graph.add((ref, ke.statement, Literal(claim.statement)))
        graph.add((ref, ke.derivation, Literal(claim.derivation)))
        graph.add((ref, ke.confidence, Literal(claim.confidence, datatype=XSD.decimal)))
        for claim_id in claim.source_claim_ids:
            graph.add((ref, ke.sourceClaimId, Literal(claim_id)))
            for _, claim_uri in locations.get(claim_id, []):
                graph.add((ref, ke.sourceClaim, URIRef(claim_uri)))
    return dataset


def integrated_to_jsonld(integrated: IntegratedOntology) -> dict[str, Any]:
    """Serialize the same scoped named-graph dataset used for TriG."""

    context = {
        "ke": DEFAULT_BASE_URI,
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "prov": "http://www.w3.org/ns/prov#",
        "dcterms": "http://purl.org/dc/terms/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
    }
    serialized = _integrated_rdf_dataset(integrated).serialize(
        format="json-ld",
        context=context,
        auto_compact=True,
    )
    parsed = json.loads(serialized.decode("utf-8") if isinstance(serialized, bytes) else serialized)
    if isinstance(parsed, list):
        payload: dict[str, Any] = {"@context": context, "@graph": parsed}
    else:
        payload = parsed
        payload.setdefault("@context", context)
    payload["@id"] = f"urn:ke:dataset:{integrated.dataset_id}"
    payload["@type"] = "ke:IntegratedOntologyDataset"
    payload["ke:canonicalSourceGraphs"] = integrated.source_graphs
    return payload


def _legacy_integrated_to_jsonld(integrated: IntegratedOntology) -> dict[str, Any]:
    """Serialize the integrated dataset as JSON-LD with explicit named graphs."""

    named_graphs: list[dict[str, Any]] = []
    for graph_id, ontology_data in integrated.source_graphs.items():
        named_graphs.append(
            {
                "@id": graph_id,
                "@type": "ke:SourceGraph",
                "ke:ontologyId": ontology_data["ontology_id"],
                "@graph": _ontology_jsonld_nodes(ontology_data),
            }
        )

    integration_nodes: list[dict[str, Any]] = []
    integration_nodes.extend(
        _mapping_jsonld(mapping.model_dump(mode="json")) for mapping in integrated.mappings
    )
    integration_nodes.extend(
        {
            "@id": f"urn:ke:complementarity:{item.id}",
            "@type": "ke:Complementarity",
            "ke:sourceClaim": [f"urn:ke:claim:{claim_id}" for claim_id in item.claim_ids],
            "dcterms:description": item.description,
        }
        for item in integrated.complementarities
    )
    integration_nodes.extend(_conflict_jsonld(conflict) for conflict in integrated.conflicts)
    integration_nodes.extend(
        {
            "@id": f"urn:ke:claim:{claim.id}",
            "@type": "ke:SynthesizedClaim",
            "ke:statement": claim.statement,
            "ke:sourceClaim": [f"urn:ke:claim:{item}" for item in claim.source_claim_ids],
            "ke:derivation": claim.derivation,
            "ke:confidence": claim.confidence,
        }
        for claim in integrated.synthesized_claims
    )
    named_graphs.append(
        {
            "@id": integrated.integration_graph_id,
            "@type": "ke:IntegrationGraph",
            "@graph": integration_nodes,
        }
    )
    return {
        "@context": {
            "ke": DEFAULT_BASE_URI,
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "prov": "http://www.w3.org/ns/prov#",
            "dcterms": "http://purl.org/dc/terms/",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        },
        "@id": f"urn:ke:dataset:{integrated.dataset_id}",
        "@type": "ke:IntegratedOntologyDataset",
        "@graph": named_graphs,
        # This lossless canonical view makes the local runtime independent of an
        # RDF query engine while the @graph structure above retains RDF semantics.
        "ke:canonicalSourceGraphs": integrated.source_graphs,
    }


def _ontology_jsonld_nodes(ontology: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = [
        {
            "@id": f"urn:ke:ontology:{ontology['ontology_id']}",
            "@type": "ke:DomainOntology",
            "dcterms:title": ontology["title"],
            "dcterms:abstract": ontology["abstract"],
            "prov:wasDerivedFrom": [f"urn:ke:source:{item}" for item in ontology["source_ids"]],
        }
    ]
    for term in ontology.get("terms", []):
        node = {
            "@id": f"urn:ke:claim:{term['id']}",
            "@type": ["skos:Concept", "ke:SourceClaim"],
            "skos:prefLabel": term["preferred_label"],
            "skos:altLabel": term.get("aliases", []),
            "dcterms:description": term["definition"],
            "ke:kind": term["kind"],
            "ke:confidence": term["confidence"],
            "skos:broader": [f"urn:ke:claim:{item}" for item in term.get("broader_ids", [])],
            "skos:narrower": [f"urn:ke:claim:{item}" for item in term.get("narrower_ids", [])],
            "skos:related": [f"urn:ke:claim:{item}" for item in term.get("related_ids", [])],
        }
        nodes.extend(_attach_evidence(node, term.get("evidence", [])))
    for relation in ontology.get("relations", []):
        node = {
            "@id": f"urn:ke:claim:{relation['id']}",
            "@type": ["ke:Relation", "ke:SourceClaim"],
            "ke:subject": f"urn:ke:claim:{relation['subject_id']}",
            "ke:predicate": relation["predicate"],
            "ke:object": f"urn:ke:claim:{relation['object_id']}",
            "ke:qualifier": relation.get("qualifiers", {}),
            "ke:confidence": relation["confidence"],
        }
        nodes.extend(_attach_evidence(node, relation.get("evidence", [])))
    for axiom in ontology.get("axioms", []):
        node = {
            "@id": f"urn:ke:claim:{axiom['id']}",
            "@type": ["ke:Axiom", "ke:SourceClaim"],
            "ke:statement": axiom["statement"],
            "ke:modality": axiom["modality"],
            "ke:epistemicStatus": axiom["epistemic_status"],
            "ke:attribution": axiom.get("attribution", "author"),
            "ke:condition": axiom.get("conditions", []),
            "ke:exception": axiom.get("exceptions", []),
            "ke:formalism": axiom.get("formalism", "none"),
            "ke:confidence": axiom["confidence"],
        }
        if axiom.get("formal_expression"):
            node["ke:formalExpression"] = axiom["formal_expression"]
        nodes.extend(_attach_evidence(node, axiom.get("evidence", [])))
    for model in ontology.get("mental_models", []):
        node = {
            "@id": f"urn:ke:claim:{model['id']}",
            "@type": ["ke:MentalModel", "ke:SourceClaim"],
            "dcterms:title": model["name"],
            "ke:purpose": model["purpose"],
            "ke:applicableWhen": model.get("applicable_when", []),
            "ke:assumption": model.get("assumptions", []),
            "ke:variable": model.get("variables", []),
            "ke:mechanism": model.get("mechanism", []),
            "ke:procedure": model.get("procedure", []),
            "ke:prediction": model.get("predictions", []),
            "ke:failureMode": model.get("failure_modes", []),
            "ke:confidence": model["confidence"],
        }
        nodes.extend(_attach_evidence(node, model.get("evidence", [])))
    return nodes


def _attach_evidence(node: dict[str, Any], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_nodes: list[dict[str, Any]] = []
    refs: list[str] = []
    for item in evidence:
        evidence_id = _evidence_id(item)
        ref = f"urn:ke:evidence:{evidence_id}"
        refs.append(ref)
        evidence_nodes.append(
            {
                "@id": ref,
                "@type": "prov:Entity",
                "prov:wasDerivedFrom": f"urn:ke:source:{item['source_id']}",
                "ke:chunkId": item["chunk_id"],
                "ke:headingPath": item.get("heading_path", []),
                "ke:pageStart": item.get("page_start"),
                "ke:pageEnd": item.get("page_end"),
                "ke:quote": item["quote"],
                "ke:quoteSha256": item["quote_sha256"],
                "ke:verified": item["verified"],
            }
        )
    node["ke:evidence"] = refs
    return [node, *evidence_nodes]


def _evidence_id(evidence: dict[str, Any] | Evidence) -> str:
    if isinstance(evidence, Evidence):
        evidence = evidence.model_dump(mode="json")
    return _stable_id(
        "evidence",
        str(evidence["source_id"]),
        str(evidence["chunk_id"]),
        str(evidence["quote_sha256"]),
    )


def _mapping_jsonld(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        "@id": f"urn:ke:mapping:{mapping['id']}",
        "@type": "ke:ConceptMapping",
        "ke:mappingKind": mapping["kind"],
        "ke:leftConcept": f"urn:ke:claim:{mapping['left_term_id']}",
        "ke:rightConcept": f"urn:ke:claim:{mapping['right_term_id']}",
        "ke:rationale": mapping["rationale"],
        "ke:confidence": mapping["confidence"],
    }


def _conflict_jsonld(conflict: dict[str, Any]) -> dict[str, Any]:
    return {
        "@id": f"urn:ke:conflict:{conflict['id']}",
        "@type": "ke:ClaimConflict",
        "ke:conflictingClaim": [f"urn:ke:claim:{item}" for item in conflict["claim_ids"]],
        "ke:conflictKind": conflict["kind"],
        "dcterms:description": conflict["description"],
        "ke:resolution": conflict["resolution"],
    }


def integrated_to_trig(integrated: IntegratedOntology) -> str:
    """Serialize the scoped RDF dataset as TriG."""

    serialized = _integrated_rdf_dataset(integrated).serialize(format="trig")
    return serialized.decode("utf-8") if isinstance(serialized, bytes) else serialized


def _legacy_integrated_to_trig(integrated: IntegratedOntology) -> str:
    """Serialize named source and integration graphs as RDF TriG."""

    try:
        from rdflib import RDF, Dataset, Literal, Namespace, URIRef
        from rdflib.namespace import DCTERMS, PROV, SKOS, XSD
    except ImportError as exc:  # pragma: no cover - dependency is declared by the package
        raise RuntimeError("rdflib is required to serialize the integrated ontology") from exc

    dataset = Dataset()
    ke = Namespace(DEFAULT_BASE_URI)
    dataset.bind("ke", ke)
    dataset.bind("skos", SKOS)
    dataset.bind("prov", PROV)
    dataset.bind("dcterms", DCTERMS)

    for graph_id, ontology in integrated.source_graphs.items():
        graph = dataset.graph(URIRef(graph_id))
        ontology_ref = URIRef(f"urn:ke:ontology:{ontology['ontology_id']}")
        graph.add((ontology_ref, RDF.type, ke.DomainOntology))
        graph.add((ontology_ref, DCTERMS.title, Literal(ontology["title"])))
        graph.add((ontology_ref, DCTERMS.abstract, Literal(ontology["abstract"])))
        for source_id in ontology["source_ids"]:
            graph.add((ontology_ref, PROV.wasDerivedFrom, URIRef(f"urn:ke:source:{source_id}")))
        for support_id in ontology.get("abstract_support_ids", []):
            graph.add((ontology_ref, ke.abstractSupport, URIRef(f"urn:ke:claim:{support_id}")))
        for scope_item in ontology.get("scope_in", []):
            graph.add((ontology_ref, ke.scopeIn, Literal(scope_item)))
        for scope_item in ontology.get("scope_out", []):
            graph.add((ontology_ref, ke.scopeOut, Literal(scope_item)))
        for term in ontology.get("terms", []):
            ref = URIRef(f"urn:ke:claim:{term['id']}")
            graph.add((ref, RDF.type, SKOS.Concept))
            graph.add((ref, RDF.type, ke.SourceClaim))
            graph.add((ref, SKOS.prefLabel, Literal(term["preferred_label"])))
            graph.add((ref, DCTERMS.description, Literal(term["definition"])))
            graph.add((ref, ke.kind, Literal(term["kind"])))
            graph.add((ref, ke.confidence, Literal(term["confidence"], datatype=XSD.double)))
            for alias in term.get("aliases", []):
                graph.add((ref, SKOS.altLabel, Literal(alias)))
            for predicate, key in (
                (SKOS.broader, "broader_ids"),
                (SKOS.narrower, "narrower_ids"),
                (SKOS.related, "related_ids"),
            ):
                for target in term.get(key, []):
                    graph.add((ref, predicate, URIRef(f"urn:ke:claim:{target}")))
            _add_rdf_evidence(
                graph, ref, term.get("evidence", []), ke, PROV, RDF, XSD, Literal, URIRef
            )
        for relation in ontology.get("relations", []):
            ref = URIRef(f"urn:ke:claim:{relation['id']}")
            graph.add((ref, RDF.type, ke.Relation))
            graph.add((ref, RDF.type, ke.SourceClaim))
            graph.add((ref, ke.subject, URIRef(f"urn:ke:claim:{relation['subject_id']}")))
            graph.add((ref, ke.predicate, Literal(relation["predicate"])))
            graph.add((ref, ke.object, URIRef(f"urn:ke:claim:{relation['object_id']}")))
            graph.add(
                (
                    ref,
                    ke.qualifiers,
                    Literal(json.dumps(relation.get("qualifiers", {}), sort_keys=True)),
                )
            )
            graph.add((ref, ke.confidence, Literal(relation["confidence"], datatype=XSD.double)))
            _add_rdf_evidence(
                graph, ref, relation.get("evidence", []), ke, PROV, RDF, XSD, Literal, URIRef
            )
        for axiom in ontology.get("axioms", []):
            ref = URIRef(f"urn:ke:claim:{axiom['id']}")
            graph.add((ref, RDF.type, ke.Axiom))
            graph.add((ref, RDF.type, ke.SourceClaim))
            graph.add((ref, ke.statement, Literal(axiom["statement"])))
            graph.add((ref, ke.modality, Literal(axiom["modality"])))
            graph.add((ref, ke.epistemicStatus, Literal(axiom["epistemic_status"])))
            graph.add((ref, ke.attribution, Literal(axiom.get("attribution", "author"))))
            graph.add((ref, ke.formalism, Literal(axiom.get("formalism", "none"))))
            if axiom.get("formal_expression"):
                graph.add((ref, ke.formalExpression, Literal(axiom["formal_expression"])))
            for condition in axiom.get("conditions", []):
                graph.add((ref, ke.condition, Literal(condition)))
            for exception in axiom.get("exceptions", []):
                graph.add((ref, ke.exception, Literal(exception)))
            graph.add((ref, ke.confidence, Literal(axiom["confidence"], datatype=XSD.double)))
            _add_rdf_evidence(
                graph, ref, axiom.get("evidence", []), ke, PROV, RDF, XSD, Literal, URIRef
            )
        for model in ontology.get("mental_models", []):
            ref = URIRef(f"urn:ke:claim:{model['id']}")
            graph.add((ref, RDF.type, ke.MentalModel))
            graph.add((ref, RDF.type, ke.SourceClaim))
            graph.add((ref, DCTERMS.title, Literal(model["name"])))
            graph.add((ref, ke.purpose, Literal(model["purpose"])))
            graph.add((ref, ke.confidence, Literal(model["confidence"], datatype=XSD.double)))
            for predicate, field in (
                (ke.applicableWhen, "applicable_when"),
                (ke.assumption, "assumptions"),
                (ke.variable, "variables"),
                (ke.mechanism, "mechanism"),
                (ke.procedure, "procedure"),
                (ke.prediction, "predictions"),
                (ke.failureMode, "failure_modes"),
            ):
                for value in model.get(field, []):
                    graph.add((ref, predicate, Literal(value)))
            _add_rdf_evidence(
                graph, ref, model.get("evidence", []), ke, PROV, RDF, XSD, Literal, URIRef
            )

    integration_graph = dataset.graph(URIRef(integrated.integration_graph_id))
    for mapping in integrated.mappings:
        ref = URIRef(f"urn:ke:mapping:{mapping.id}")
        integration_graph.add((ref, RDF.type, ke.ConceptMapping))
        integration_graph.add((ref, ke.mappingKind, Literal(mapping.kind)))
        integration_graph.add((ref, ke.leftConcept, URIRef(f"urn:ke:claim:{mapping.left_term_id}")))
        integration_graph.add(
            (ref, ke.rightConcept, URIRef(f"urn:ke:claim:{mapping.right_term_id}"))
        )
        integration_graph.add((ref, DCTERMS.description, Literal(mapping.rationale)))
        integration_graph.add(
            (ref, ke.confidence, Literal(mapping.confidence, datatype=XSD.double))
        )
        mapping_predicate = {
            "exact_equivalent": SKOS.exactMatch,
            "broader": SKOS.narrowMatch,
            "narrower": SKOS.broadMatch,
            "related_or_analogous": SKOS.relatedMatch,
        }.get(mapping.kind)
        if mapping_predicate is not None:
            integration_graph.add(
                (
                    URIRef(f"urn:ke:claim:{mapping.left_term_id}"),
                    mapping_predicate,
                    URIRef(f"urn:ke:claim:{mapping.right_term_id}"),
                )
            )
    for conflict in integrated.conflicts:
        ref = URIRef(f"urn:ke:conflict:{conflict['id']}")
        integration_graph.add((ref, RDF.type, ke.ClaimConflict))
        integration_graph.add((ref, DCTERMS.description, Literal(conflict["description"])))
        integration_graph.add((ref, ke.conflictKind, Literal(conflict["kind"])))
        integration_graph.add((ref, ke.resolution, Literal(conflict["resolution"])))
        for claim_id in conflict["claim_ids"]:
            integration_graph.add((ref, ke.conflictingClaim, URIRef(f"urn:ke:claim:{claim_id}")))
    for item in integrated.complementarities:
        ref = URIRef(f"urn:ke:complementarity:{item.id}")
        integration_graph.add((ref, RDF.type, ke.Complementarity))
        integration_graph.add((ref, DCTERMS.description, Literal(item.description)))
        for claim_id in item.claim_ids:
            integration_graph.add((ref, ke.sourceClaim, URIRef(f"urn:ke:claim:{claim_id}")))
    for claim in integrated.synthesized_claims:
        ref = URIRef(f"urn:ke:claim:{claim.id}")
        integration_graph.add((ref, RDF.type, ke.SynthesizedClaim))
        integration_graph.add((ref, ke.statement, Literal(claim.statement)))
        integration_graph.add((ref, ke.derivation, Literal(claim.derivation)))
        integration_graph.add((ref, ke.confidence, Literal(claim.confidence, datatype=XSD.double)))
        for claim_id in claim.source_claim_ids:
            integration_graph.add((ref, ke.sourceClaim, URIRef(f"urn:ke:claim:{claim_id}")))

    serialized = dataset.serialize(format="trig")
    return serialized.decode("utf-8") if isinstance(serialized, bytes) else serialized


def _add_rdf_evidence(
    graph: Any,
    claim_ref: Any,
    evidence: list[dict[str, Any]],
    ke: Any,
    prov: Any,
    rdf: Any,
    xsd: Any,
    literal: Any,
    uri_ref: Any,
) -> None:
    for item in evidence:
        evidence_ref = uri_ref(f"urn:ke:evidence:{_evidence_id(item)}")
        graph.add((claim_ref, ke.evidence, evidence_ref))
        graph.add((evidence_ref, rdf.type, prov.Entity))
        graph.add(
            (evidence_ref, prov.wasDerivedFrom, uri_ref(f"urn:ke:source:{item['source_id']}"))
        )
        graph.add((evidence_ref, ke.chunkId, literal(item["chunk_id"])))
        graph.add(
            (
                evidence_ref,
                ke.headingPath,
                literal(json.dumps(item.get("heading_path", []), ensure_ascii=False)),
            )
        )
        if item.get("page_start") is not None:
            graph.add(
                (
                    evidence_ref,
                    ke.pageStart,
                    literal(item["page_start"], datatype=xsd.integer),
                )
            )
        if item.get("page_end") is not None:
            graph.add(
                (
                    evidence_ref,
                    ke.pageEnd,
                    literal(item["page_end"], datatype=xsd.integer),
                )
            )
        graph.add((evidence_ref, ke.quote, literal(item["quote"])))
        graph.add((evidence_ref, ke.quoteSha256, literal(item["quote_sha256"])))
        graph.add(
            (evidence_ref, ke.verified, literal(bool(item["verified"]), datatype=xsd.boolean))
        )


def dumps_jsonld(integrated: IntegratedOntology) -> str:
    return json.dumps(integrated_to_jsonld(integrated), ensure_ascii=False, indent=2) + "\n"
