"""Render runtime-facing prompt, reasoning policy, ledger, and tool manifest."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined

from ke.contracts import AgentFrame, ClaimConflict, DomainOntology
from ke.resource_paths import resource_path

from .models import StanceLedger, StanceLedgerEntry


def compile_system_prompt(
    frame: AgentFrame,
    *,
    ontologies: Iterable[DomainOntology] = (),
    conflicts: Iterable[ClaimConflict] = (),
    template_path: str | Path | None = None,
) -> str:
    """Compile the frame into a citation-enforcing runtime system prompt."""

    ontology_list = list(ontologies)
    conflict_list = list(conflicts)
    selected_template = (
        Path(template_path)
        if template_path is not None
        else resource_path("prompts", "system_prompt.jinja2")
    )
    environment = Environment(
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.from_string(selected_template.read_text(encoding="utf-8"))
    rendered = template.render(
        frame=frame,
        ontologies=ontology_list,
        conflicts=conflict_list,
        operator_desire=next(
            desire for desire in frame.desires if desire.origin == "operator_objective"
        ),
        source_norm_desires=[desire for desire in frame.desires if desire.origin == "source_norm"],
        synthesized_subgoals=[
            desire for desire in frame.desires if desire.origin == "synthesized_subgoal"
        ],
    ).strip()
    return rendered + "\n"


def render_reasoning_policy(frame: AgentFrame) -> str:
    policy = frame.reasoning_policy
    lines = [
        f"# Reasoning policy: {frame.name}",
        "",
        "The BDI frame is an instrumental behavioral description. It is not a claim of "
        "consciousness or sentience.",
        "",
        "## Deliberation cycle",
        "",
    ]
    lines.extend(f"{index}. {step}" for index, step in enumerate(policy.deliberation_cycle, 1))
    lines.extend(
        [
            "",
            "## Model selection",
            "",
            policy.model_selection_policy,
            "",
            "## Evidence",
            "",
            policy.evidence_policy,
            "",
            "## Conflicts",
            "",
            policy.conflict_policy,
            "",
            "## Uncertainty",
            "",
            policy.uncertainty_policy,
            "",
            "## Answer contract",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in policy.answer_contract)
    return "\n".join(lines).strip() + "\n"


def build_stance_ledger(frame: AgentFrame) -> StanceLedger:
    """Create an auditable ledger of every BDI ascription and its origin."""

    entries: list[StanceLedgerEntry] = []
    for belief in frame.beliefs:
        evidence_ids = [
            _evidence_id(
                evidence.source_id,
                evidence.chunk_id,
                evidence.quote_sha256,
            )
            for evidence in belief.evidence
        ]
        entries.append(
            StanceLedgerEntry(
                frame_object_id=belief.id,
                kind="belief",
                origin=belief.origin,
                support_ids=belief.support_ids,
                evidence_ids=evidence_ids,
                note=(
                    "Source-attributed candidate knowledge."
                    if belief.origin == "source"
                    else "Explicitly marked non-source ascription."
                ),
            )
        )
    for desire in frame.desires:
        if desire.origin == "operator_objective":
            note = "The explicit operator objective; the sole adopted top-level goal."
        elif desire.origin == "source_norm":
            note = "A source-attributed norm recorded for analysis, not silently adopted."
        else:
            note = "A synthesized procedural subgoal for pursuing the operator objective safely."
        entries.append(
            StanceLedgerEntry(
                frame_object_id=desire.id,
                kind="desire",
                origin=desire.origin,
                parent_desire_ids=desire.parent_desire_ids,
                source_claim_ids=desire.source_claim_ids,
                support_ids=[*desire.parent_desire_ids, *desire.source_claim_ids],
                note=note,
            )
        )
    for intention in frame.intentions:
        entries.append(
            StanceLedgerEntry(
                frame_object_id=intention.id,
                kind="intention",
                origin="instrumental_policy",
                support_ids=[*intention.belief_ids, *intention.desire_ids],
                note=(
                    "A trigger-action policy linked to grounded beliefs and explicit desires; "
                    "not a claim about an inner mental state."
                ),
            )
        )
    return StanceLedger(frame_id=frame.frame_id, entries=entries)


def build_tool_manifest() -> dict[str, Any]:
    """Return the stable local runtime tool contract."""

    return {
        "schema_version": "1.0",
        "transport": "local_python",
        "tools": [
            {
                "name": "lookup_concept",
                "description": "Look up a canonical concept by ID without merging source graphs.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "trace_claim",
                "description": "Trace a claim to its source graph and exact verified evidence.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "retrieve_evidence",
                "description": "Lexically retrieve verified evidence from selected sources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "source_ids": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "default": None,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_conflicts",
                "description": "List preserved conflicts that involve a concept or claim ID.",
                "parameters": {
                    "type": "object",
                    "properties": {"concept_id": {"type": "string"}},
                    "required": ["concept_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_mental_model",
                "description": "Retrieve a source-grounded mental model by canonical ID.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
        ],
    }


def _evidence_id(source_id: str, chunk_id: str, quote_sha256: str) -> str:
    digest = hashlib.sha256(
        "\x1f".join([source_id, chunk_id, quote_sha256]).encode("utf-8")
    ).hexdigest()[:16]
    return f"evidence-{digest}"
