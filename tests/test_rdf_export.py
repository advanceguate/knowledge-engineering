from __future__ import annotations

from rdflib import RDF, Graph
from rdflib.namespace import PROV, SKOS

from ke.contracts import Axiom, DomainOntology, Evidence, Term
from ke.grounding import quote_sha256
from ke.ontology.export import export_ontology
from ke.semantic.rdf import KE, ontology_to_graph
from ke.semantic.validate import validate_graph


def ontology_fixture() -> DomainOntology:
    quote = "A market coordinates exchange."
    evidence = Evidence(
        source_id="book-aabbccddeeff",
        chunk_id="chunk-1",
        quote=quote,
        quote_sha256=quote_sha256(quote),
        verified=True,
    )
    term = Term(
        id="term:market-aabbccddeeff",
        preferred_label="Market",
        aliases=["market system"],
        kind="system",
        definition="a system that coordinates exchange",
        evidence=[evidence],
        confidence=0.9,
    )
    axiom = Axiom(
        id="axiom:market-coordinates-aabbccddeeff",
        statement=quote,
        modality="descriptive",
        epistemic_status="asserted",
        evidence=[evidence],
        confidence=0.9,
    )
    return DomainOntology(
        ontology_id="ontology:markets-aabbccddeeff",
        title="Markets",
        source_ids=["book-aabbccddeeff"],
        abstract=quote,
        abstract_support_ids=[axiom.id],
        terms=[term],
        relations=[],
        axioms=[axiom],
        mental_models=[],
    )


def test_rdf_contains_skos_and_prov_and_passes_shacl(tmp_path):
    ontology = ontology_fixture()
    graph = ontology_to_graph(ontology)
    assert any(graph.subjects(RDF.type, SKOS.Concept))
    evidence_nodes = list(graph.subjects(RDF.type, KE.Evidence))
    assert len(evidence_nodes) == 1
    assert list(graph.objects(evidence_nodes[0], PROV.wasDerivedFrom))
    assert validate_graph(graph).conforms

    result = export_ontology(ontology, tmp_path / "ontology")
    parsed_turtle = Graph().parse(result.ontology_ttl, format="turtle")
    parsed_jsonld = Graph().parse(result.ontology_jsonld, format="json-ld")
    assert len(parsed_turtle) == len(graph)
    assert len(parsed_jsonld) == len(graph)
    assert result.validation.conforms
