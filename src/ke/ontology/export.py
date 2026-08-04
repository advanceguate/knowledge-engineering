"""Publish canonical ontology bundles and semantic serializations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ke.artifact_store import ArtifactStore
from ke.contracts import DomainOntology
from ke.semantic.rdf import DEFAULT_BASE_URI, ontology_to_graph, serialize_graph
from ke.semantic.validate import OntologyValidationError, ValidationReport, validate_graph


class ExportResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    output_dir: Path
    ontology_json: Path
    ontology_ttl: Path
    ontology_jsonld: Path
    validation_json: Path
    validation: ValidationReport

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def as_dict(self) -> dict[str, str]:
        return {
            "output_dir": str(self.output_dir),
            "ontology_json": str(self.ontology_json),
            "ontology_ttl": str(self.ontology_ttl),
            "ontology_jsonld": str(self.ontology_jsonld),
            "validation_json": str(self.validation_json),
        }


def _writer(store: ArtifactStore | None, output_dir: Path) -> ArtifactStore:
    # ArtifactStore write methods accept explicit paths inside their managed
    # root. Direct exports therefore manage the destination's parent.
    return store or ArtifactStore(output_dir.parent)


def export_ontology(
    ontology: DomainOntology | dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    store: ArtifactStore | None = None,
    base_uri: str = DEFAULT_BASE_URI,
    shapes_path: str | Path | None = None,
    require_conformance: bool = True,
) -> ExportResult:
    """Write the complete local ontology bundle and validate its RDF."""

    canonical = (
        ontology
        if isinstance(ontology, DomainOntology)
        else DomainOntology.model_validate(ontology)
    )
    if output_dir is None:
        store = store or ArtifactStore()
        destination = store.ontology_dir(canonical.ontology_id)
    else:
        destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    writer = _writer(store, destination)

    graph = ontology_to_graph(canonical, base_uri=base_uri)
    report = validate_graph(graph, shapes_path=shapes_path)
    if require_conformance and not report.conforms:
        raise OntologyValidationError(report)

    ontology_json = writer.write_json(destination / "ontology.json", canonical)
    ontology_ttl = writer.write_text(
        destination / "ontology.ttl", serialize_graph(graph, format="turtle")
    )
    ontology_jsonld = writer.write_text(
        destination / "ontology.jsonld", serialize_graph(graph, format="json-ld")
    )
    validation_json = writer.write_json(destination / "validation.json", report)
    writer.write_text(
        destination / "abstract.md",
        f"# {canonical.title}\n\n{canonical.abstract}\n\n"
        + "## Support\n\n"
        + "\n".join(f"- `{identifier}`" for identifier in canonical.abstract_support_ids)
        + "\n",
    )
    writer.write_yaml(
        destination / "entities-thesaurus.yaml",
        [term.model_dump(mode="json") for term in canonical.terms],
    )
    writer.write_yaml(
        destination / "axioms.yaml",
        [axiom.model_dump(mode="json") for axiom in canonical.axioms],
    )
    writer.write_yaml(
        destination / "mental-models.yaml",
        [model.model_dump(mode="json") for model in canonical.mental_models],
    )
    writer.write_yaml(
        destination / "conflicts.yaml",
        [conflict.model_dump(mode="json") for conflict in canonical.conflicts],
    )
    return ExportResult(
        output_dir=destination,
        ontology_json=ontology_json,
        ontology_ttl=ontology_ttl,
        ontology_jsonld=ontology_jsonld,
        validation_json=validation_json,
        validation=report,
    )


def export_rdf(
    ontology: DomainOntology | dict[str, Any],
    destination: str | Path,
    *,
    format: str = "turtle",
    base_uri: str = DEFAULT_BASE_URI,
) -> Path:
    """Write a single RDF serialization (useful for integration callers)."""

    canonical = (
        ontology
        if isinstance(ontology, DomainOntology)
        else DomainOntology.model_validate(ontology)
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    ArtifactStore(path.parent).write_text(
        path,
        serialize_graph(ontology_to_graph(canonical, base_uri=base_uri), format=format),
    )
    return path


publish_ontology_bundle = export_ontology


__all__ = ["ExportResult", "export_ontology", "export_rdf", "publish_ontology_bundle"]
