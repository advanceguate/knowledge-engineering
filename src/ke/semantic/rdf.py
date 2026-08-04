"""RDF/SKOS/PROV representation of canonical domain ontologies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from urllib.parse import quote

from rdflib import RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, PROV, SKOS, XSD

from ke.contracts import DomainOntology, Evidence
from ke.ids import slugify

DEFAULT_BASE_URI = "https://w3id.org/knowledge-engineering/"
KE = Namespace(DEFAULT_BASE_URI)


def _base(value: str) -> str:
    return value if value.endswith(("/", "#")) else value + "/"


def _resource(base_uri: str, identifier: str) -> URIRef:
    return URIRef(f"{_base(base_uri)}resource/{quote(identifier, safe='')}")


def _source(base_uri: str, source_id: str) -> URIRef:
    return URIRef(f"{_base(base_uri)}source/{quote(source_id, safe='')}")


def _chunk(base_uri: str, source_id: str, chunk_id: str) -> URIRef:
    return URIRef(
        f"{_base(base_uri)}source/{quote(source_id, safe='')}/chunk/{quote(chunk_id, safe='')}"
    )


def _predicate(base_uri: str, predicate: str) -> URIRef:
    if predicate.startswith(("http://", "https://", "urn:")):
        return URIRef(predicate)
    return URIRef(f"{_base(base_uri)}relation/{quote(slugify(predicate), safe='')}")


def _evidence_uri(base_uri: str, evidence: Evidence) -> URIRef:
    signature = "\0".join(
        (evidence.source_id, evidence.chunk_id, evidence.quote_sha256, evidence.quote)
    )
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    return URIRef(f"{_base(base_uri)}evidence/{digest}")


def _bind(graph: Graph) -> Namespace:
    # The KE vocabulary is stable even when callers choose a different base
    # for instance resources.
    namespace = KE
    graph.bind("ke", KE)
    graph.bind("skos", SKOS)
    graph.bind("prov", PROV)
    graph.bind("owl", OWL)
    graph.bind("dcterms", DCTERMS)
    graph.bind("rdfs", RDFS)
    graph.bind("xsd", XSD)
    return namespace


def _add_string_values(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
    values: Iterable[str],
) -> None:
    for value in values:
        graph.add((subject, predicate, Literal(value)))


def _add_evidence(
    graph: Graph,
    subject: URIRef,
    evidence_items: Iterable[Evidence],
    namespace: Namespace,
    base_uri: str,
) -> None:
    for evidence in evidence_items:
        evidence_uri = _evidence_uri(base_uri, evidence)
        source_uri = _source(base_uri, evidence.source_id)
        chunk_uri = _chunk(base_uri, evidence.source_id, evidence.chunk_id)
        graph.add((subject, PROV.wasDerivedFrom, evidence_uri))
        graph.add((evidence_uri, RDF.type, namespace.Evidence))
        graph.add((evidence_uri, RDF.type, PROV.Entity))
        graph.add((evidence_uri, namespace.sourceId, Literal(evidence.source_id)))
        graph.add((evidence_uri, namespace.chunkId, Literal(evidence.chunk_id)))
        graph.add((evidence_uri, namespace.quote, Literal(evidence.quote)))
        graph.add((evidence_uri, namespace.quoteSha256, Literal(evidence.quote_sha256)))
        graph.add(
            (evidence_uri, namespace.verified, Literal(evidence.verified, datatype=XSD.boolean))
        )
        if evidence.heading_path:
            graph.add(
                (
                    evidence_uri,
                    namespace.headingPath,
                    Literal(
                        json.dumps(evidence.heading_path, ensure_ascii=False), datatype=RDF.JSON
                    ),
                )
            )
        if evidence.page_start is not None:
            graph.add(
                (
                    evidence_uri,
                    namespace.pageStart,
                    Literal(evidence.page_start, datatype=XSD.positiveInteger),
                )
            )
        if evidence.page_end is not None:
            graph.add(
                (
                    evidence_uri,
                    namespace.pageEnd,
                    Literal(evidence.page_end, datatype=XSD.positiveInteger),
                )
            )
        graph.add((evidence_uri, PROV.wasDerivedFrom, chunk_uri))
        graph.add((chunk_uri, RDF.type, PROV.Entity))
        graph.add((chunk_uri, DCTERMS.identifier, Literal(evidence.chunk_id)))
        graph.add((chunk_uri, PROV.wasDerivedFrom, source_uri))
        graph.add((source_uri, RDF.type, PROV.Entity))
        graph.add((source_uri, DCTERMS.identifier, Literal(evidence.source_id)))


def ontology_to_graph(
    ontology: DomainOntology,
    *,
    base_uri: str = DEFAULT_BASE_URI,
) -> Graph:
    """Map a validated ontology to SKOS plus evidence-rich custom resources."""

    if not isinstance(ontology, DomainOntology):
        ontology = DomainOntology.model_validate(ontology)
    graph = Graph(
        identifier=URIRef(f"{_base(base_uri)}graph/{quote(ontology.ontology_id, safe='')}")
    )
    namespace = _bind(graph)
    ontology_uri = URIRef(f"{_base(base_uri)}ontology/{quote(ontology.ontology_id, safe='')}")
    graph.add((ontology_uri, RDF.type, OWL.Ontology))
    graph.add((ontology_uri, DCTERMS.identifier, Literal(ontology.ontology_id)))
    graph.add((ontology_uri, DCTERMS.title, Literal(ontology.title)))
    graph.add((ontology_uri, DCTERMS.abstract, Literal(ontology.abstract)))
    graph.add((ontology_uri, OWL.versionInfo, Literal(ontology.schema_version)))
    for source_id in ontology.source_ids:
        source_uri = _source(base_uri, source_id)
        graph.add((ontology_uri, PROV.wasDerivedFrom, source_uri))
        graph.add((source_uri, RDF.type, PROV.Entity))
        graph.add((source_uri, DCTERMS.identifier, Literal(source_id)))

    resource_by_id = {
        item.id: _resource(base_uri, item.id)
        for item in [
            *ontology.terms,
            *ontology.relations,
            *ontology.axioms,
            *ontology.mental_models,
            *ontology.conflicts,
        ]
    }

    for term in ontology.terms:
        uri = resource_by_id[term.id]
        graph.add((ontology_uri, namespace.hasTerm, uri))
        graph.add((uri, RDF.type, SKOS.Concept))
        graph.add((uri, DCTERMS.identifier, Literal(term.id)))
        graph.add((uri, SKOS.prefLabel, Literal(term.preferred_label)))
        graph.add((uri, SKOS.definition, Literal(term.definition)))
        graph.add((uri, namespace.termKind, Literal(term.kind)))
        graph.add((uri, namespace.confidence, Literal(term.confidence, datatype=XSD.decimal)))
        _add_string_values(graph, uri, SKOS.altLabel, term.aliases)
        for broader_id in term.broader_ids:
            graph.add((uri, SKOS.broader, resource_by_id[broader_id]))
        for narrower_id in term.narrower_ids:
            graph.add((uri, SKOS.narrower, resource_by_id[narrower_id]))
        for related_id in term.related_ids:
            graph.add((uri, SKOS.related, resource_by_id[related_id]))
        _add_evidence(graph, uri, term.evidence, namespace, base_uri)

    for relation in ontology.relations:
        uri = resource_by_id[relation.id]
        predicate_uri = _predicate(base_uri, relation.predicate)
        subject_uri = resource_by_id[relation.subject_id]
        object_uri = resource_by_id[relation.object_id]
        graph.add((ontology_uri, namespace.hasRelation, uri))
        graph.add((uri, RDF.type, namespace.Relation))
        graph.add((uri, RDF.type, RDF.Statement))
        graph.add((uri, DCTERMS.identifier, Literal(relation.id)))
        graph.add((uri, RDF.subject, subject_uri))
        graph.add((uri, RDF.predicate, predicate_uri))
        graph.add((uri, RDF.object, object_uri))
        graph.add((uri, namespace.predicateLabel, Literal(relation.predicate)))
        graph.add((uri, namespace.confidence, Literal(relation.confidence, datatype=XSD.decimal)))
        graph.add((subject_uri, predicate_uri, object_uri))
        graph.add((predicate_uri, RDF.type, RDF.Property))
        graph.add((predicate_uri, RDFS.label, Literal(relation.predicate)))
        for key, value in sorted(relation.qualifiers.items()):
            qualifier = BNode()
            graph.add((uri, namespace.qualifier, qualifier))
            graph.add((qualifier, namespace.qualifierKey, Literal(key)))
            graph.add((qualifier, namespace.qualifierValue, Literal(value)))
        _add_evidence(graph, uri, relation.evidence, namespace, base_uri)

    for axiom in ontology.axioms:
        uri = resource_by_id[axiom.id]
        graph.add((ontology_uri, namespace.hasAxiom, uri))
        graph.add((uri, RDF.type, namespace.Axiom))
        graph.add((uri, DCTERMS.identifier, Literal(axiom.id)))
        graph.add((uri, namespace.statement, Literal(axiom.statement)))
        graph.add((uri, namespace.modality, Literal(axiom.modality)))
        graph.add((uri, namespace.epistemicStatus, Literal(axiom.epistemic_status)))
        graph.add((uri, namespace.attribution, Literal(axiom.attribution)))
        graph.add((uri, namespace.formalism, Literal(axiom.formalism)))
        graph.add((uri, namespace.confidence, Literal(axiom.confidence, datatype=XSD.decimal)))
        _add_string_values(graph, uri, namespace.condition, axiom.conditions)
        _add_string_values(graph, uri, namespace.exception, axiom.exceptions)
        if axiom.formal_expression:
            graph.add((uri, namespace.formalExpression, Literal(axiom.formal_expression)))
        _add_evidence(graph, uri, axiom.evidence, namespace, base_uri)

    model_fields = (
        ("applicable_when", "applicableWhen"),
        ("assumptions", "assumption"),
        ("variables", "variable"),
        ("mechanism", "mechanismStep"),
        ("procedure", "procedureStep"),
        ("predictions", "prediction"),
        ("failure_modes", "failureMode"),
    )
    for model in ontology.mental_models:
        uri = resource_by_id[model.id]
        graph.add((ontology_uri, namespace.hasMentalModel, uri))
        graph.add((uri, RDF.type, namespace.MentalModel))
        graph.add((uri, DCTERMS.identifier, Literal(model.id)))
        graph.add((uri, RDFS.label, Literal(model.name)))
        graph.add((uri, namespace.purpose, Literal(model.purpose)))
        graph.add((uri, namespace.confidence, Literal(model.confidence, datatype=XSD.decimal)))
        for field, predicate_name in model_fields:
            _add_string_values(graph, uri, namespace[predicate_name], getattr(model, field))
        _add_evidence(graph, uri, model.evidence, namespace, base_uri)

    for conflict in ontology.conflicts:
        uri = resource_by_id[conflict.id]
        graph.add((ontology_uri, namespace.hasConflict, uri))
        graph.add((uri, RDF.type, namespace.ClaimConflict))
        graph.add((uri, DCTERMS.identifier, Literal(conflict.id)))
        graph.add((uri, DCTERMS.description, Literal(conflict.description)))
        graph.add((uri, namespace.conflictKind, Literal(conflict.kind)))
        graph.add((uri, namespace.resolution, Literal(conflict.resolution)))
        for claim_id in conflict.claim_ids:
            graph.add((uri, namespace.conflictingClaim, resource_by_id[claim_id]))

    for support_id in ontology.abstract_support_ids:
        graph.add((ontology_uri, namespace.abstractSupport, resource_by_id[support_id]))
    _add_string_values(graph, ontology_uri, namespace.scopeIn, ontology.scope_in)
    _add_string_values(graph, ontology_uri, namespace.scopeOut, ontology.scope_out)
    _add_string_values(graph, ontology_uri, namespace.openQuestion, ontology.open_questions)
    return graph


def serialize_graph(graph: Graph, *, format: str = "turtle") -> str:
    serialized = graph.serialize(format=format)
    return serialized.decode("utf-8") if isinstance(serialized, bytes) else serialized


def ontology_to_turtle(ontology: DomainOntology, *, base_uri: str = DEFAULT_BASE_URI) -> str:
    return serialize_graph(ontology_to_graph(ontology, base_uri=base_uri), format="turtle")


def ontology_to_jsonld(ontology: DomainOntology, *, base_uri: str = DEFAULT_BASE_URI) -> str:
    return serialize_graph(ontology_to_graph(ontology, base_uri=base_uri), format="json-ld")


# Descriptive aliases used by callers and older fixtures.
domain_ontology_to_graph = ontology_to_graph
ontology_to_rdf = ontology_to_graph


__all__ = [
    "DEFAULT_BASE_URI",
    "KE",
    "domain_ontology_to_graph",
    "ontology_to_graph",
    "ontology_to_jsonld",
    "ontology_to_rdf",
    "ontology_to_turtle",
    "serialize_graph",
]
