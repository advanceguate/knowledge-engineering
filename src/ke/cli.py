"""Command-line interface for the three local-first pipelines."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from ke.artifact_store import ArtifactStore
from ke.contracts import DomainOntology
from ke.llm import PydanticAIGateway
from ke.settings import Settings

app = typer.Typer(
    name="ke",
    help="Source-grounded ontology extraction and intentional-stance synthesis.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


def _settings(config: Path | None, workspace: Path | None) -> Settings:
    selected = Settings.load(config)
    if workspace is not None:
        selected = selected.model_copy(update={"workspace": workspace})
    return selected


def _gateway(
    settings: Settings,
    store: ArtifactStore,
    *,
    offline: bool,
) -> PydanticAIGateway | None:
    if offline:
        return None
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    return PydanticAIGateway(
        settings.extract_model,
        store=store,
        concurrency=settings.llm_concurrency,
        openai_api_key=api_key,
    )


def _print(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    console.print_json(data=value, default=str)


def _ontology_summary(bundle: Any) -> dict[str, Any]:
    ontology = bundle.ontology
    return {
        "ontology_id": bundle.ontology_id,
        "output_dir": bundle.output_dir,
        "source_ids": bundle.source_ids,
        "cache_hit": bundle.cache_hit,
        "validation_conforms": bundle.validation.conforms,
        "counts": {
            "terms": len(ontology.terms),
            "relations": len(ontology.relations),
            "axioms": len(ontology.axioms),
            "mental_models": len(ontology.mental_models),
            "conflicts": len(ontology.conflicts),
        },
    }


def _synthesis_summary(bundle: Any) -> dict[str, Any]:
    return {
        "frame_id": bundle.frame_id,
        "synthesis_key": bundle.synthesis_key,
        "cache_hit": bundle.cache_hit,
        "output_dir": bundle.output_dir,
        "ontology_ids": bundle.integrated.ontology_ids,
        "preserved_conflicts": len(bundle.integrated.conflicts),
        "review_approved": bundle.review.approved,
        "repairs_attempted": bundle.review.repairs_attempted,
    }


@app.command("pdf-to-markdown")
def pdf_to_markdown_command(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    title: Annotated[str | None, typer.Option("--title")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Convert a PDF into Markdown, canonical Docling JSON, and provenance chunks."""

    from ke.pipelines.pdf_to_markdown import pdf_to_markdown

    settings = _settings(config, workspace)
    bundle = pdf_to_markdown(
        source,
        settings=settings,
        workspace=settings.workspace,
        title=title,
        force=force,
    )
    _print(bundle)


@app.command("ingest")
def ingest_command(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    title: Annotated[str | None, typer.Option("--title")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Ingest PDF, EPUB, HTML, DOCX, Markdown, or plain text."""

    from ke.ingest import ingest_source

    settings = _settings(config, workspace)
    bundle = ingest_source(
        source,
        settings=settings,
        workspace=settings.workspace,
        title=title,
        force=force,
    )
    _print(bundle)


@app.command("extract")
def extract_command(
    source_bundle: Annotated[Path, typer.Argument(exists=True)],
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    title: Annotated[str | None, typer.Option("--title")] = None,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Use conservative local extraction instead of a model provider.",
        ),
    ] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Extract, ground, reduce, export, and validate one domain ontology."""

    from ke.pipelines.domain_ontology import run_domain_ontology_sync

    settings = _settings(config, workspace)
    store = ArtifactStore(settings.workspace)
    bundle = run_domain_ontology_sync(
        source_bundle,
        store=store,
        settings=settings,
        gateway=_gateway(settings, store, offline=offline),
        title=title,
        force=force,
    )
    _print(_ontology_summary(bundle))


@app.command("synthesize")
def synthesize_command(
    ontologies: Annotated[list[Path], typer.Argument(exists=True)],
    name: Annotated[str, typer.Option("--name")],
    objective: Annotated[str, typer.Option("--objective")],
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Use deterministic local alignment and frame composition.",
        ),
    ] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Align ontologies and compose a conflict-preserving intentional frame."""

    from ke.pipelines.intentional_synthesis import synthesize_ontologies

    settings = _settings(config, workspace)
    store = ArtifactStore(settings.workspace)
    bundle = asyncio.run(
        synthesize_ontologies(
            ontologies,
            name=name,
            objective=objective,
            store=store,
            settings=settings,
            gateway=_gateway(settings, store, offline=offline),
            force=force,
        )
    )
    _print(_synthesis_summary(bundle))


@app.command("build")
def build_command(
    sources: Annotated[list[Path], typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    name: Annotated[str, typer.Option("--name")],
    objective: Annotated[str, typer.Option("--objective")],
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Run extraction and synthesis without provider calls.",
        ),
    ] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Run ingestion, ontology extraction, and intentional synthesis end to end."""

    from ke.ingest import ingest_source
    from ke.pipelines.domain_ontology import run_domain_ontology
    from ke.pipelines.intentional_synthesis import synthesize_ontologies

    settings = _settings(config, workspace)
    store = ArtifactStore(settings.workspace)
    gateway = _gateway(settings, store, offline=offline)

    async def run() -> Any:
        ontology_inputs: list[DomainOntology] = []
        for source in sources:
            source_bundle = ingest_source(
                source,
                settings=settings,
                workspace=settings.workspace,
                force=force,
            )
            ontology_bundle = await run_domain_ontology(
                source_bundle,
                store=store,
                settings=settings,
                gateway=gateway,
                force=force,
            )
            ontology_inputs.append(ontology_bundle.ontology)
        return await synthesize_ontologies(
            ontology_inputs,
            name=name,
            objective=objective,
            store=store,
            settings=settings,
            gateway=gateway,
            force=force,
        )

    _print(_synthesis_summary(asyncio.run(run())))


@app.command("validate")
def validate_command(
    artifact: Annotated[Path, typer.Argument(exists=True)],
    shapes: Annotated[Path | None, typer.Option("--shapes")] = None,
) -> None:
    """Validate an ontology JSON bundle or an RDF serialization."""

    from ke.semantic.validate import validate_graph, validate_ontology

    path = artifact / "ontology.json" if artifact.is_dir() else artifact
    if path.name == "ontology.json" or path.suffix.lower() == ".json":
        ontology = DomainOntology.model_validate_json(path.read_text(encoding="utf-8"))
        report = validate_ontology(ontology, shapes_path=shapes)
    else:
        data_format = {
            ".ttl": "turtle",
            ".jsonld": "json-ld",
            ".rdf": "xml",
            ".nt": "nt",
        }.get(path.suffix.lower())
        report = validate_graph(path, shapes_path=shapes, data_format=data_format)
    _print(report)
    if not report.conforms:
        raise typer.Exit(code=1)


@app.command("ask")
def ask_command(
    frame: Annotated[Path, typer.Argument(exists=True)],
    question: Annotated[str, typer.Argument()],
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Retrieve evidence without generating a model synthesis.",
        ),
    ] = False,
) -> None:
    """Query a published frame through the optional local runtime."""

    from ke.runtime.agent import IntentionalRuntime

    settings = _settings(config, workspace)
    store = ArtifactStore(settings.workspace)
    bundle_dir = frame.parent if frame.is_file() else frame
    runtime = IntentionalRuntime(
        bundle_dir,
        gateway=_gateway(settings, store, offline=offline),
        model=settings.synthesis_model,
    )
    _print(asyncio.run(runtime.ask(question)))


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
