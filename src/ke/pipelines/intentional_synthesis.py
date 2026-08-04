"""LangGraph pipeline for conflict-preserving intentional-stance synthesis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph

from ke.artifact_store import ArtifactStore
from ke.contracts import AgentFrame, ClaimConflict, DomainOntology
from ke.ids import content_hash, sha256_file, sha256_text
from ke.llm import ModelGateway
from ke.resource_paths import read_resource_text
from ke.settings import Settings, load_settings
from ke.synthesis.align import align_ontologies_with_gateway
from ke.synthesis.critic import (
    critique_frame,
    critique_frame_with_gateway,
    finalize_repair_review,
    repair_frame,
)
from ke.synthesis.frame import compose_bdi_frame_with_gateway
from ke.synthesis.integrate import (
    build_integrated_named_graph,
    identify_conflicts,
    integrated_to_jsonld,
    integrated_to_trig,
    materialize_synthesized_beliefs,
)
from ke.synthesis.models import (
    AlignmentReport,
    CriticPass,
    IntegratedOntology,
    ReviewReport,
    SynthesisBundle,
)
from ke.synthesis.render import (
    build_stance_ledger,
    build_tool_manifest,
    compile_system_prompt,
    render_reasoning_policy,
)

OntologyInput = str | Path | DomainOntology
SYNTHESIS_PIPELINE_VERSION = "2.0"
SYNTHESIS_CACHE_NAMESPACE = "intentional-synthesis-v2"
SYNTHESIS_ARTIFACTS = {
    "alignment.yaml",
    "conflicts.yaml",
    "integrated-ontology.jsonld",
    "integrated-ontology.trig",
    "agent-frame.yaml",
    "stance-ledger.json",
    "reasoning-policy.md",
    "system-prompt.md",
    "tool-manifest.json",
    "review.json",
}
SYNTHESIS_PROMPTS = (
    "ontology_alignment.md",
    "frame_compose.md",
    "frame_critic.md",
    "system_prompt.jinja2",
)


class SynthesisReviewRejected(RuntimeError):
    """Raised before publication when the bounded critic rejects a frame."""

    def __init__(self, review: ReviewReport) -> None:
        self.review = review
        codes = sorted(
            {finding.code for finding in review.final.findings if finding.severity == "error"}
        )
        super().__init__(f"Synthesis frame failed final critic review: {codes}")


class SynthesisState(TypedDict):
    ontology_inputs: Sequence[OntologyInput]
    name: str
    objective: str
    synthesis_key: NotRequired[str]
    ontologies: NotRequired[list[DomainOntology]]
    alignment: NotRequired[AlignmentReport]
    conflicts: NotRequired[list[ClaimConflict]]
    integrated: NotRequired[IntegratedOntology]
    frame: NotRequired[AgentFrame]
    initial_review: NotRequired[CriticPass]
    review: NotRequired[ReviewReport]
    system_prompt: NotRequired[str]
    reasoning_policy_markdown: NotRequired[str]
    bundle: NotRequired[SynthesisBundle]


def build_intentional_synthesis_graph(
    *,
    store: ArtifactStore,
    settings: Settings,
    gateway: ModelGateway | None = None,
) -> Any:
    """Build the bounded synthesis graph.

    The only conditional cycle is a single edge through ``repair_frame``. The
    repair node always exits to compilation after a fresh deterministic review;
    there is no autonomous revision loop.
    """

    async def load_ontologies_node(state: SynthesisState) -> dict[str, Any]:
        ontologies = load_ontologies(state["ontology_inputs"])
        synthesis_key = state.get("synthesis_key") or synthesis_cache_key(
            ontologies,
            name=state["name"],
            objective=state["objective"],
            settings=settings,
            gateway=gateway,
        )
        return {"ontologies": ontologies, "synthesis_key": synthesis_key}

    async def align_concepts_node(state: SynthesisState) -> dict[str, Any]:
        alignment = await align_ontologies_with_gateway(
            state["ontologies"],
            gateway=gateway,
            model=settings.synthesis_model,
        )
        return {"alignment": alignment}

    async def identify_conflicts_node(state: SynthesisState) -> dict[str, Any]:
        return {
            "conflicts": identify_conflicts(
                state["ontologies"],
                state["alignment"],
            )
        }

    async def integrate_node(state: SynthesisState) -> dict[str, Any]:
        integrated = build_integrated_named_graph(
            state["ontologies"],
            state["alignment"],
            state["conflicts"],
        )
        dataset_id = f"{integrated.dataset_id}-s{state['synthesis_key'][:12]}"
        return {
            "integrated": integrated.model_copy(
                update={
                    "dataset_id": dataset_id,
                    "integration_graph_id": f"urn:ke:graph:integration:{dataset_id}",
                }
            )
        }

    async def compose_frame_node(state: SynthesisState) -> dict[str, Any]:
        frame = await compose_bdi_frame_with_gateway(
            state["ontologies"],
            name=state["name"],
            objective=state["objective"],
            conflicts=state["conflicts"],
            integrated=state["integrated"],
            gateway=gateway,
            model=settings.synthesis_model,
        )
        frame = frame.model_copy(
            update={"frame_id": _versioned_frame_id(frame.frame_id, state["synthesis_key"])}
        )
        return {"frame": frame}

    async def critique_node(state: SynthesisState) -> dict[str, Any]:
        initial = await critique_frame_with_gateway(
            state["frame"],
            state["ontologies"],
            conflicts=state["conflicts"],
            gateway=gateway,
            model=settings.critic_model,
        )
        if initial.approved or settings.synthesis.critic_repairs == 0:
            return {
                "initial_review": initial,
                "review": ReviewReport(
                    approved=initial.approved,
                    repairs_attempted=0,
                    initial=initial,
                    final=initial,
                ),
            }
        return {"initial_review": initial}

    def route_after_critic(state: SynthesisState) -> str:
        if state["initial_review"].approved or settings.synthesis.critic_repairs == 0:
            return "compile_system_prompt"
        return "conditional_single_repair"

    async def repair_node(state: SynthesisState) -> dict[str, Any]:
        repaired = repair_frame(
            state["frame"],
            state["initial_review"],
            state["ontologies"],
            conflicts=state["conflicts"],
        )
        final = critique_frame(
            repaired,
            state["ontologies"],
            conflicts=state["conflicts"],
        )
        final = finalize_repair_review(state["initial_review"], final)
        return {
            "frame": repaired,
            "review": ReviewReport(
                approved=final.approved,
                repairs_attempted=1,
                initial=state["initial_review"],
                final=final,
            ),
        }

    async def compile_node(state: SynthesisState) -> dict[str, Any]:
        if not state["review"].approved:
            raise SynthesisReviewRejected(state["review"])
        integrated = materialize_synthesized_beliefs(state["integrated"], state["frame"])
        return {
            "integrated": integrated,
            "system_prompt": compile_system_prompt(
                state["frame"],
                ontologies=state["ontologies"],
                conflicts=state["conflicts"],
            ),
            "reasoning_policy_markdown": render_reasoning_policy(state["frame"]),
        }

    async def publish_node(state: SynthesisState) -> dict[str, Any]:
        output_dir = publish_synthesis_bundle(
            store=store,
            alignment=state["alignment"],
            conflicts=state["conflicts"],
            integrated=state["integrated"],
            frame=state["frame"],
            review=state["review"],
            system_prompt=state["system_prompt"],
            reasoning_policy_markdown=state["reasoning_policy_markdown"],
        )
        return {
            "bundle": SynthesisBundle(
                frame_id=state["frame"].frame_id,
                synthesis_key=state["synthesis_key"],
                cache_hit=False,
                output_dir=str(output_dir),
                alignment=state["alignment"],
                integrated=state["integrated"],
                review=state["review"],
                frame=state["frame"],
            )
        }

    graph = StateGraph(SynthesisState)
    graph.add_node("load_ontologies", load_ontologies_node)
    graph.add_node("align_concepts", align_concepts_node)
    graph.add_node("identify_conflicts_and_complementarities", identify_conflicts_node)
    graph.add_node("build_integrated_named_graph", integrate_node)
    graph.add_node("compose_bdi_frame", compose_frame_node)
    graph.add_node("critique_frame", critique_node)
    graph.add_node("conditional_single_repair", repair_node)
    graph.add_node("compile_system_prompt", compile_node)
    graph.add_node("publish_synthesis_bundle", publish_node)
    graph.add_edge(START, "load_ontologies")
    graph.add_edge("load_ontologies", "align_concepts")
    graph.add_edge("align_concepts", "identify_conflicts_and_complementarities")
    graph.add_edge("identify_conflicts_and_complementarities", "build_integrated_named_graph")
    graph.add_edge("build_integrated_named_graph", "compose_bdi_frame")
    graph.add_edge("compose_bdi_frame", "critique_frame")
    graph.add_conditional_edges(
        "critique_frame",
        route_after_critic,
        {
            "conditional_single_repair": "conditional_single_repair",
            "compile_system_prompt": "compile_system_prompt",
        },
    )
    graph.add_edge("conditional_single_repair", "compile_system_prompt")
    graph.add_edge("compile_system_prompt", "publish_synthesis_bundle")
    graph.add_edge("publish_synthesis_bundle", END)
    return graph.compile()


async def synthesize_ontologies(
    ontology_inputs: Sequence[OntologyInput],
    *,
    name: str,
    objective: str,
    store: ArtifactStore | None = None,
    settings: Settings | None = None,
    gateway: ModelGateway | None = None,
    force: bool = False,
) -> SynthesisBundle:
    """Run intentional synthesis and atomically publish all required artifacts."""

    if not objective or not objective.strip():
        raise ValueError("An explicit non-empty --objective is required for synthesis")
    if not name or not name.strip():
        raise ValueError("A non-empty frame name is required for synthesis")
    if not ontology_inputs:
        raise ValueError("At least one ontology input is required for synthesis")
    selected_settings = settings or load_settings()
    selected_store = store or ArtifactStore(selected_settings.workspace)
    ontologies = load_ontologies(ontology_inputs)
    synthesis_key = synthesis_cache_key(
        ontologies,
        name=name.strip(),
        objective=objective.strip(),
        settings=selected_settings,
        gateway=gateway,
    )
    if not force:
        cached = _load_cached_bundle(selected_store, synthesis_key)
        if cached is not None:
            return cached
    graph = build_intentional_synthesis_graph(
        store=selected_store,
        settings=selected_settings,
        gateway=gateway,
    )
    state = await graph.ainvoke(
        {
            "ontology_inputs": ontologies,
            "name": name.strip(),
            "objective": objective.strip(),
            "synthesis_key": synthesis_key,
        }
    )
    bundle = state["bundle"]
    _write_cache_record(selected_store, bundle)
    return bundle


synthesize = synthesize_ontologies


def load_ontologies(inputs: Sequence[OntologyInput]) -> list[DomainOntology]:
    ontologies: list[DomainOntology] = []
    for item in inputs:
        if isinstance(item, DomainOntology):
            ontology = item
        else:
            path = Path(item)
            if path.is_dir():
                path = path / "ontology.json"
            if not path.is_file():
                raise FileNotFoundError(f"Ontology input does not exist: {path}")
            ontology = DomainOntology.model_validate_json(path.read_text(encoding="utf-8"))
        ontologies.append(ontology)
    ids = [ontology.ontology_id for ontology in ontologies]
    if len(ids) != len(set(ids)):
        raise ValueError("Synthesis inputs must have unique ontology IDs")
    return ontologies


def publish_synthesis_bundle(
    *,
    store: ArtifactStore,
    alignment: AlignmentReport,
    conflicts: list[ClaimConflict],
    integrated: IntegratedOntology,
    frame: AgentFrame,
    review: ReviewReport,
    system_prompt: str,
    reasoning_policy_markdown: str,
) -> Path:
    """Write the complete synthesis artifact contract atomically file-by-file."""

    if not review.approved:
        raise SynthesisReviewRejected(review)
    output_dir = store.synthesis_dir(frame.frame_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    store.write_yaml(output_dir / "alignment.yaml", alignment)
    store.write_yaml(
        output_dir / "conflicts.yaml",
        [conflict.model_dump(mode="json") for conflict in conflicts],
    )
    store.write_json(
        output_dir / "integrated-ontology.jsonld",
        integrated_to_jsonld(integrated),
    )
    store.write_text(
        output_dir / "integrated-ontology.trig",
        integrated_to_trig(integrated),
    )
    store.write_yaml(output_dir / "agent-frame.yaml", frame)
    store.write_json(output_dir / "stance-ledger.json", build_stance_ledger(frame))
    store.write_text(output_dir / "reasoning-policy.md", reasoning_policy_markdown)
    store.write_text(output_dir / "system-prompt.md", system_prompt)
    store.write_json(output_dir / "tool-manifest.json", build_tool_manifest())
    store.write_json(output_dir / "review.json", review)
    return output_dir


def synthesis_cache_key(
    ontologies: Sequence[DomainOntology],
    *,
    name: str,
    objective: str,
    settings: Settings,
    gateway: ModelGateway | None,
) -> str:
    """Hash every semantic and model input that can affect synthesis output."""

    payload = {
        "pipeline_version": SYNTHESIS_PIPELINE_VERSION,
        "ontologies": [
            {
                "ontology_id": ontology.ontology_id,
                "content_hash": content_hash(ontology),
            }
            for ontology in ontologies
        ],
        "name": name.strip(),
        "objective": objective.strip(),
        "gateway": _gateway_cache_identity(gateway),
        "synthesis_settings": settings.synthesis.model_dump(mode="json"),
        "runtime_settings": settings.runtime.model_dump(mode="json"),
        "models": {
            "synthesis": settings.synthesis_model,
            "critic": settings.critic_model,
        },
        "prompt_versions": {
            prompt_name: sha256_text(read_resource_text("prompts", prompt_name))
            for prompt_name in SYNTHESIS_PROMPTS
        },
        "schema_versions": {
            model.__name__: content_hash(model.model_json_schema())
            for model in (
                DomainOntology,
                AgentFrame,
                AlignmentReport,
                CriticPass,
                IntegratedOntology,
                ReviewReport,
            )
        },
    }
    return content_hash(payload)


def _gateway_cache_identity(gateway: ModelGateway | None) -> dict[str, Any]:
    if gateway is None:
        return {"mode": "offline", "identity": "deterministic-local"}
    gateway_type = f"{type(gateway).__module__}.{type(gateway).__qualname__}"
    explicit = getattr(gateway, "cache_identity", None)
    if callable(explicit):
        explicit = explicit()
    identity: dict[str, Any] = {"mode": "live", "type": gateway_type}
    if explicit is not None:
        identity["identity"] = explicit
    elif default_model := getattr(gateway, "default_model", None):
        identity["identity"] = {"default_model": str(default_model)}
    else:
        # Generic/fake gateways cannot be assumed semantically interchangeable.
        # A stable instance marker permits safe repeated calls in one process.
        identity["identity"] = {"instance": f"0x{id(gateway):x}"}
    return identity


def _versioned_frame_id(base_frame_id: str, synthesis_key: str) -> str:
    suffix = f"-s{synthesis_key[:12]}"
    return base_frame_id if base_frame_id.endswith(suffix) else f"{base_frame_id}{suffix}"


def _load_cached_bundle(
    store: ArtifactStore,
    synthesis_key: str,
) -> SynthesisBundle | None:
    try:
        record = store.get_cached_json(SYNTHESIS_CACHE_NAMESPACE, synthesis_key)
        if not isinstance(record, dict):
            return None
        if (
            record.get("schema_version") != SYNTHESIS_PIPELINE_VERSION
            or record.get("synthesis_key") != synthesis_key
        ):
            return None
        bundle = SynthesisBundle.model_validate(record.get("bundle"))
        if (
            bundle.synthesis_key != synthesis_key
            or not bundle.review.approved
            or bundle.frame_id != bundle.frame.frame_id
        ):
            return None
        output_dir = store.synthesis_dir(bundle.frame_id)
        artifact_hashes = record.get("artifact_hashes")
        if not isinstance(artifact_hashes, dict):
            return None
        for filename in SYNTHESIS_ARTIFACTS:
            path = output_dir / filename
            expected_hash = artifact_hashes.get(filename)
            if not path.is_file() or not isinstance(expected_hash, str):
                return None
            if sha256_file(path) != expected_hash:
                return None
        on_disk_frame = AgentFrame.model_validate(
            yaml.safe_load((output_dir / "agent-frame.yaml").read_text(encoding="utf-8"))
        )
        on_disk_alignment = AlignmentReport.model_validate(
            yaml.safe_load((output_dir / "alignment.yaml").read_text(encoding="utf-8"))
        )
        on_disk_review = ReviewReport.model_validate(
            json.loads((output_dir / "review.json").read_text(encoding="utf-8"))
        )
        json.loads((output_dir / "integrated-ontology.jsonld").read_text(encoding="utf-8"))
        json.loads((output_dir / "stance-ledger.json").read_text(encoding="utf-8"))
        json.loads((output_dir / "tool-manifest.json").read_text(encoding="utf-8"))
        conflicts = yaml.safe_load((output_dir / "conflicts.yaml").read_text(encoding="utf-8"))
        if not isinstance(conflicts, list):
            return None
        if (
            content_hash(on_disk_frame) != content_hash(bundle.frame)
            or content_hash(on_disk_alignment) != content_hash(bundle.alignment)
            or content_hash(on_disk_review) != content_hash(bundle.review)
        ):
            return None
        if not (output_dir / "system-prompt.md").read_text(encoding="utf-8").strip():
            return None
        if not (output_dir / "reasoning-policy.md").read_text(encoding="utf-8").strip():
            return None
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return None
    return bundle.model_copy(update={"output_dir": str(output_dir), "cache_hit": True})


def _write_cache_record(store: ArtifactStore, bundle: SynthesisBundle) -> None:
    if not bundle.review.approved:
        return
    output_dir = Path(bundle.output_dir)
    artifact_hashes = {
        filename: sha256_file(output_dir / filename) for filename in SYNTHESIS_ARTIFACTS
    }
    cache_bundle = bundle.model_copy(update={"cache_hit": False})
    store.put_cached_json(
        SYNTHESIS_CACHE_NAMESPACE,
        bundle.synthesis_key,
        {
            "schema_version": SYNTHESIS_PIPELINE_VERSION,
            "synthesis_key": bundle.synthesis_key,
            "frame_id": bundle.frame_id,
            "artifact_hashes": artifact_hashes,
            "bundle": cache_bundle.model_dump(mode="json"),
        },
    )
