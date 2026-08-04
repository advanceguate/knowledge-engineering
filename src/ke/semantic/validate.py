"""SHACL and structural validation for ontology RDF artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from rdflib import RDF, Graph, Namespace

from ke.contracts import DomainOntology
from ke.resource_paths import resource_path
from ke.semantic.rdf import ontology_to_graph

SH = Namespace("http://www.w3.org/ns/shacl#")


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    focus_node: str = ""
    path: str = ""
    message: str
    severity: str = "Violation"
    source_shape: str = ""


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conforms: bool
    engine: str = "pyshacl"
    violations: list[ValidationIssue] = Field(default_factory=list)
    results_text: str = ""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OntologyValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(report.results_text or "ontology RDF does not conform to SHACL")


def default_shapes_path() -> Path:
    return resource_path("shapes", "domain_ontology.shacl.ttl")


def _as_graph(value: Graph | str | Path, *, format: str | None = None) -> Graph:
    if isinstance(value, Graph):
        return value
    graph = Graph()
    path = Path(value)
    graph.parse(path, format=format)
    return graph


def _compact(value: Any) -> str:
    return "" if value is None else str(value)


def _issues(report_graph: Graph) -> list[ValidationIssue]:
    output: list[ValidationIssue] = []
    for result in report_graph.subjects(RDF.type, SH.ValidationResult):
        messages = list(report_graph.objects(result, SH.resultMessage))
        output.append(
            ValidationIssue(
                focus_node=_compact(report_graph.value(result, SH.focusNode)),
                path=_compact(report_graph.value(result, SH.resultPath)),
                message="; ".join(str(message) for message in messages)
                or "SHACL constraint violation",
                severity=_compact(report_graph.value(result, SH.resultSeverity)).rsplit("#", 1)[-1]
                or "Violation",
                source_shape=_compact(report_graph.value(result, SH.sourceShape)),
            )
        )
    return sorted(
        output,
        key=lambda issue: (issue.severity, issue.focus_node, issue.path, issue.message),
    )


def _fallback_validate(graph: Graph) -> ValidationReport:
    """Minimal fail-closed checks if pySHACL is not installed."""

    namespace = Namespace("https://w3id.org/knowledge-engineering/")
    classes = (
        Namespace("http://www.w3.org/2004/02/skos/core#").Concept,
        namespace.Relation,
        namespace.Axiom,
        namespace.MentalModel,
    )
    prov = Namespace("http://www.w3.org/ns/prov#")
    issues: list[ValidationIssue] = []
    for class_ in classes:
        for subject in graph.subjects(RDF.type, class_):
            if not list(graph.objects(subject, prov.wasDerivedFrom)):
                issues.append(
                    ValidationIssue(
                        focus_node=str(subject),
                        path=str(prov.wasDerivedFrom),
                        message="Every canonical knowledge object needs source evidence.",
                    )
                )
    for evidence in graph.subjects(RDF.type, namespace.Evidence):
        if not any(bool(value.toPython()) for value in graph.objects(evidence, namespace.verified)):
            issues.append(
                ValidationIssue(
                    focus_node=str(evidence),
                    path=str(namespace.verified),
                    message="Canonical evidence must be verified.",
                )
            )
    return ValidationReport(
        conforms=not issues,
        engine="structural-fallback",
        violations=issues,
        results_text="Fallback structural validation passed."
        if not issues
        else "Fallback structural validation failed.",
    )


def validate_graph(
    graph: Graph | str | Path,
    *,
    shapes_path: str | Path | None = None,
    data_format: str | None = None,
    inference: str = "rdfs",
    raise_on_error: bool = False,
) -> ValidationReport:
    """Validate RDF and return a JSON-serializable report."""

    data_graph = _as_graph(graph, format=data_format)
    shape_path = Path(shapes_path) if shapes_path else default_shapes_path()
    if not shape_path.exists():
        raise FileNotFoundError(f"SHACL shapes not found: {shape_path}")
    try:
        from pyshacl import validate as pyshacl_validate
    except ImportError:  # pragma: no cover - dependencies normally include pySHACL
        report = _fallback_validate(data_graph)
    else:
        conforms, report_graph, results_text = pyshacl_validate(
            data_graph,
            shacl_graph=str(shape_path),
            inference=inference,
            abort_on_first=False,
            allow_infos=True,
            allow_warnings=True,
            advanced=True,
        )
        parsed_report = report_graph if isinstance(report_graph, Graph) else Graph()
        report = ValidationReport(
            conforms=bool(conforms),
            engine="pyshacl",
            violations=_issues(parsed_report),
            results_text=str(results_text),
        )
    if raise_on_error and not report.conforms:
        raise OntologyValidationError(report)
    return report


def validate_ontology(
    ontology: DomainOntology | dict[str, Any],
    *,
    shapes_path: str | Path | None = None,
    raise_on_error: bool = False,
) -> ValidationReport:
    canonical = (
        ontology
        if isinstance(ontology, DomainOntology)
        else DomainOntology.model_validate(ontology)
    )
    return validate_graph(
        ontology_to_graph(canonical),
        shapes_path=shapes_path,
        raise_on_error=raise_on_error,
    )


validate_ontology_graph = validate_graph
validate_rdf = validate_graph


__all__ = [
    "OntologyValidationError",
    "ValidationIssue",
    "ValidationReport",
    "default_shapes_path",
    "validate_graph",
    "validate_ontology",
    "validate_ontology_graph",
    "validate_rdf",
]
