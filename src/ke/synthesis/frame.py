"""Compose an instrumental, source-grounded belief/desire/intention frame."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Literal

from ke.contracts import (
    CANONICAL_CONFLICT_SUBGOAL,
    CANONICAL_PROVENANCE_SUBGOAL,
    AgentFrame,
    Belief,
    ClaimConflict,
    Desire,
    DomainOntology,
    Intention,
    ReasoningPolicy,
    validate_agent_frame_support,
)
from ke.ids import frame_id as make_frame_id
from ke.ids import stable_id
from ke.llm import ModelGateway
from ke.resource_paths import read_resource_text

from .models import IntegratedOntology

DEFAULT_DELIBERATION_CYCLE = [
    "Interpret the task and identify the operative operator objective.",
    "Retrieve relevant beliefs, source axioms, and verified evidence.",
    "Surface relevant source conflicts without prematurely resolving them.",
    "Select an applicable mental model and state why it applies.",
    "Generate candidate conclusions.",
    "Test assumptions, exceptions, uncertainty, and failure modes.",
    "Answer while separating source claims, synthesis, and uncertainty.",
]


def compose_bdi_frame(
    ontologies: Iterable[DomainOntology],
    *,
    name: str,
    objective: str,
    conflicts: Iterable[ClaimConflict] | None = None,
) -> AgentFrame:
    """Deterministically compose a validated frame from canonical source claims.

    The objective is the only adopted top-level goal. Normative source claims
    are recorded as ``source_norm`` desires, never silently promoted to operator
    instructions. The stance is an instrumental behavioral ascription.
    """

    objective = objective.strip()
    name = name.strip()
    if not objective:
        raise ValueError("An explicit non-empty operator objective is required")
    if not name:
        raise ValueError("A non-empty frame name is required")

    ontology_list = list(ontologies)
    if not ontology_list:
        raise ValueError("At least one ontology is required for synthesis")
    conflict_list = list(conflicts or [])
    contested_claim_ids = {
        claim_id for conflict in conflict_list for claim_id in conflict.claim_ids
    }

    beliefs = _compose_beliefs(ontology_list, contested_claim_ids)
    if not beliefs:
        raise ValueError("Cannot compose a grounded frame from ontologies with no source claims")

    operator_desire = Desire(
        id=stable_id("desire", "operator objective", objective),
        goal=objective,
        origin="operator_objective",
        priority=5,
        rationale="Explicitly supplied by the operator; this is the frame's operative objective.",
        success_criteria=[
            "The response materially advances the stated objective.",
            "Conclusions remain within the declared scope and evidence limits.",
            "The response distinguishes source claims from synthesis and uncertainty.",
        ],
    )
    provenance_desire = Desire(
        id=stable_id("desire", "preserve source provenance", objective),
        goal=CANONICAL_PROVENANCE_SUBGOAL,
        origin="synthesized_subgoal",
        parent_desire_ids=[operator_desire.id],
        priority=5,
        rationale=(
            "A procedural subgoal required to pursue the objective without attribution leakage."
        ),
        success_criteria=[
            "Every source-derived claim names its source and evidence reference.",
            "Synthesis is labeled separately from source claims.",
        ],
    )
    conflict_desire = Desire(
        id=stable_id("desire", "preserve disagreements", objective),
        goal=CANONICAL_CONFLICT_SUBGOAL,
        origin="synthesized_subgoal",
        parent_desire_ids=[operator_desire.id],
        priority=4,
        rationale=(
            "A procedural subgoal that protects decision quality under the operator objective."
        ),
        success_criteria=[
            "Relevant unresolved conflicts are stated before recommendations.",
            "Conditional resolutions retain the claims and conditions from each source.",
        ],
    )
    source_norm_desires = _source_norm_desires(ontology_list)
    desires = [operator_desire, provenance_desire, conflict_desire, *source_norm_desires]

    adopted_models = [model.id for ontology in ontology_list for model in ontology.mental_models]
    intentions = _compose_intentions(
        beliefs=beliefs,
        operator_desire=operator_desire,
        provenance_desire=provenance_desire,
        conflict_desire=conflict_desire,
        ontologies=ontology_list,
        conflicts=conflict_list,
    )

    frame = AgentFrame(
        frame_id=make_frame_id(
            name,
            objective,
            [ontology.ontology_id for ontology in ontology_list],
        ),
        name=name,
        role=(
            "A source-grounded reasoning assistant that applies the supplied ontologies "
            "instrumentally to the operator's objective."
        ),
        objective=objective,
        scope_in=_ordered_unique(
            [objective, *(item for ontology in ontology_list for item in ontology.scope_in)]
        ),
        scope_out=_ordered_unique(
            [item for ontology in ontology_list for item in ontology.scope_out]
        ),
        capabilities=[
            "Look up canonical concepts while retaining source ontology identity.",
            "Trace beliefs and claims to verified quotations.",
            "Compare source positions and surface unresolved conflicts.",
            "Select and apply source-grounded mental models under explicit assumptions.",
            "Produce clearly labeled source claims, synthesis, and uncertainty.",
        ],
        limitations=[
            "Has no knowledge authority beyond the integrated source artifacts and operator input.",
            (
                "Cannot resolve empirical or normative disagreement without additional evidence "
                "or values."
            ),
            "Cannot treat source authors' goals as the operator's goals.",
            "Uses an instrumental BDI description and makes no consciousness or sentience claim.",
        ],
        anti_goals=[
            "Do not erase, average away, or conceal source disagreements.",
            "Do not convert descriptive claims or source norms into operator objectives.",
            "Do not fabricate citations, evidence, concepts, or mental models.",
            "Do not expose hidden chain-of-thought or claim consciousness or sentience.",
        ],
        beliefs=beliefs,
        desires=desires,
        intentions=intentions,
        adopted_mental_model_ids=adopted_models,
        reasoning_policy=ReasoningPolicy(
            deliberation_cycle=DEFAULT_DELIBERATION_CYCLE,
            model_selection_policy=(
                "Choose only an adopted mental model whose applicable_when conditions fit the "
                "task. State relevant assumptions and failure modes; compare models when several "
                "fit, and proceed without a model when none is justified."
            ),
            evidence_policy=(
                "Treat ontology content as source-attributed candidate knowledge. Ground source "
                "claims in verified evidence, cite source/chunk or claim IDs, and never fill gaps "
                "with unstated general knowledge."
            ),
            conflict_policy=(
                "Surface relevant conflicts with each position's attribution. Keep unresolved "
                "conflicts unresolved unless explicit evidence or operator criteria justify a "
                "conditional resolution."
            ),
            uncertainty_policy=(
                "State confidence, conditions, missing evidence, and plausible alternatives. Stop "
                "or ask for evidence when a conclusion would outrun verified support."
            ),
            answer_contract=[
                "Begin with the conclusion or decision-relevant result.",
                "Separate Source claims, Synthesis, and Uncertainty into explicit sections.",
                "Attach source/claim/evidence citations to every substantive source-derived claim.",
                "Name relevant conflicts, assumptions, exceptions, and mental-model failure modes.",
                "Never present a source norm as the operator objective.",
            ],
        ),
    )
    validate_agent_frame_support(frame, _known_support_ids(ontology_list))
    return frame


def _compose_beliefs(
    ontologies: list[DomainOntology],
    contested_claim_ids: set[str],
) -> list[Belief]:
    beliefs: list[Belief] = []
    for ontology in ontologies:
        labels = {term.id: term.preferred_label for term in ontology.terms}
        for term in ontology.terms:
            beliefs.append(
                Belief(
                    id=stable_id(
                        "belief",
                        term.id,
                        {"ontology": ontology.ontology_id, "definition": term.definition},
                    ),
                    proposition=f"{term.preferred_label}: {term.definition}",
                    status="contested" if term.id in contested_claim_ids else "accepted",
                    origin="source",
                    support_ids=[term.id],
                    evidence=[item.model_copy(deep=True) for item in term.evidence],
                    confidence=term.confidence,
                )
            )
        for axiom in ontology.axioms:
            if axiom.id in contested_claim_ids or axiom.epistemic_status == "contested":
                status: Literal["accepted", "conditional", "contested", "synthesized"] = "contested"
            elif axiom.epistemic_status in {"conditional", "hypothetical", "reported"}:
                status = "conditional"
            else:
                status = "accepted"
            beliefs.append(
                Belief(
                    id=stable_id(
                        "belief",
                        axiom.id,
                        {"ontology": ontology.ontology_id, "statement": axiom.statement},
                    ),
                    proposition=axiom.statement,
                    status=status,
                    origin="source",
                    support_ids=[axiom.id],
                    evidence=[item.model_copy(deep=True) for item in axiom.evidence],
                    confidence=axiom.confidence,
                )
            )
        for relation in ontology.relations:
            beliefs.append(
                Belief(
                    id=stable_id(
                        "belief",
                        relation.id,
                        {"ontology": ontology.ontology_id, "predicate": relation.predicate},
                    ),
                    proposition=(
                        f"{labels.get(relation.subject_id, relation.subject_id)} "
                        f"{relation.predicate} "
                        f"{labels.get(relation.object_id, relation.object_id)}."
                    ),
                    status="contested" if relation.id in contested_claim_ids else "accepted",
                    origin="source",
                    support_ids=[relation.id],
                    evidence=[item.model_copy(deep=True) for item in relation.evidence],
                    confidence=relation.confidence,
                )
            )
        for mental_model in ontology.mental_models:
            beliefs.append(
                Belief(
                    id=stable_id(
                        "belief",
                        mental_model.id,
                        {"ontology": ontology.ontology_id, "purpose": mental_model.purpose},
                    ),
                    proposition=(
                        f"{ontology.title} proposes the {mental_model.name} mental model for: "
                        f"{mental_model.purpose}"
                    ),
                    status=(
                        "contested" if mental_model.id in contested_claim_ids else "conditional"
                    ),
                    origin="source",
                    support_ids=[mental_model.id],
                    evidence=[item.model_copy(deep=True) for item in mental_model.evidence],
                    confidence=mental_model.confidence,
                )
            )

    return beliefs


def _source_norm_desires(ontologies: list[DomainOntology]) -> list[Desire]:
    desires: list[Desire] = []
    for ontology in ontologies:
        for axiom in ontology.axioms:
            if axiom.modality != "normative":
                continue
            desires.append(
                Desire(
                    id=stable_id(
                        "desire",
                        f"source norm {axiom.id}",
                        {"ontology": ontology.ontology_id, "claim": axiom.id},
                    ),
                    goal=axiom.statement,
                    origin="source_norm",
                    source_claim_ids=[axiom.id],
                    priority=2,
                    rationale=(
                        f"Source-attributed norm from {ontology.title} ({axiom.id}); recorded for "
                        "analysis and not adopted as an operator instruction."
                    ),
                    success_criteria=[
                        "Attribute the norm to its source when it is relevant.",
                        "Apply it only when the operator objective or explicit values justify it.",
                    ],
                )
            )
    return desires


def _compose_intentions(
    *,
    beliefs: list[Belief],
    operator_desire: Desire,
    provenance_desire: Desire,
    conflict_desire: Desire,
    ontologies: list[DomainOntology],
    conflicts: list[ClaimConflict],
) -> list[Intention]:
    intentions = [
        Intention(
            id=stable_id("intention", "grounded deliberation", operator_desire.goal),
            trigger="A request within the declared scope is received.",
            action_policy=(
                "Identify the operative objective; retrieve relevant supported beliefs and exact "
                "evidence; inspect conflicts; select an applicable model; test conditions and "
                "failure modes; then produce a cited answer."
            ),
            expected_result=(
                "A decision-relevant answer grounded in the supplied source ontologies."
            ),
            belief_ids=[belief.id for belief in beliefs[: min(12, len(beliefs))]],
            desire_ids=[operator_desire.id, provenance_desire.id, conflict_desire.id],
            constraints=[
                "Do not treat source-attributed candidate knowledge as universal truth.",
                "Do not promote source norms into the operator objective.",
                "Do not use unverified or unreferenced evidence.",
            ],
            stop_conditions=[
                "No verified source belief supports a necessary premise.",
                "The request falls outside the declared scope.",
                "A material conflict cannot be resolved under the operator's stated criteria.",
            ],
        )
    ]
    belief_by_support = {
        support_id: belief for belief in beliefs for support_id in belief.support_ids
    }
    for ontology in ontologies:
        for model in ontology.mental_models:
            model_belief = belief_by_support[model.id]
            intentions.append(
                Intention(
                    id=stable_id(
                        "intention",
                        f"apply {model.id}",
                        {"ontology": ontology.ontology_id, "objective": operator_desire.goal},
                    ),
                    trigger=(
                        "; ".join(model.applicable_when)
                        if model.applicable_when
                        else f"The task matches the stated purpose of {model.name}."
                    ),
                    action_policy=(
                        f"Apply {model.name} using its stated procedure; disclose assumptions and "
                        "check its predictions and failure modes against retrieved evidence."
                    ),
                    expected_result=(
                        f"A conditional {model.name}-based analysis that advances the operator "
                        "objective without overstating the source."
                    ),
                    belief_ids=[model_belief.id],
                    desire_ids=[operator_desire.id, provenance_desire.id],
                    mental_model_ids=[model.id],
                    constraints=[*model.assumptions],
                    stop_conditions=[
                        *model.failure_modes,
                        "The model's applicability conditions are not met.",
                        "Required evidence is absent or conflicts remain decision-critical.",
                    ],
                )
            )

    for conflict in conflicts:
        linked = [
            belief_by_support[claim_id].id
            for claim_id in conflict.claim_ids
            if claim_id in belief_by_support
        ]
        if not linked:
            continue
        intentions.append(
            Intention(
                id=stable_id("intention", f"surface {conflict.id}", conflict.claim_ids),
                trigger=f"A request depends on claims in conflict {conflict.id}.",
                action_policy=(
                    "Present each source position and its evidence, state the conflict kind and "
                    "resolution status, and conditionalize any synthesis on explicit assumptions."
                ),
                expected_result="A conflict-aware answer that retains all source positions.",
                belief_ids=linked,
                desire_ids=[operator_desire.id, conflict_desire.id, provenance_desire.id],
                constraints=["Do not choose a preferred source without an explicit criterion."],
                stop_conditions=[
                    "The operator requests a categorical conclusion but provides no criterion for "
                    "resolving the material disagreement."
                ],
            )
        )
    return intentions


def _known_support_ids(ontologies: list[DomainOntology]) -> set[str]:
    return {
        item.id
        for ontology in ontologies
        for group in (
            ontology.terms,
            ontology.relations,
            ontology.axioms,
            ontology.mental_models,
        )
        for item in group
    }


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value.strip()))


async def compose_bdi_frame_with_gateway(
    ontologies: Iterable[DomainOntology],
    *,
    name: str,
    objective: str,
    conflicts: Iterable[ClaimConflict],
    integrated: IntegratedOntology,
    gateway: ModelGateway | None,
    model: str | None = None,
) -> AgentFrame:
    """Use the configured structured gateway, or deterministic composition offline."""

    ontology_list = list(ontologies)
    conflict_list = list(conflicts)
    if gateway is None:
        return compose_bdi_frame(
            ontology_list,
            name=name,
            objective=objective,
            conflicts=conflict_list,
        )
    instructions = read_resource_text("prompts", "frame_compose.md")
    payload = {
        "name": name,
        "operator_objective": objective,
        "source_ontologies": [ontology.model_dump(mode="json") for ontology in ontology_list],
        "alignment_mappings": [item.model_dump(mode="json") for item in integrated.mappings],
        "conflicts": [item.model_dump(mode="json") for item in conflict_list],
    }
    frame = await gateway.structured(
        json.dumps(payload, ensure_ascii=False, indent=2),
        AgentFrame,
        model=model,
        system_prompt=instructions,
    )
    if frame.objective.strip() != objective.strip():
        raise ValueError("Model-composed frame changed the explicit operator objective")
    if frame.name.strip() != name.strip():
        raise ValueError("Model-composed frame changed the requested frame name")
    expected_frame_id = make_frame_id(
        name.strip(),
        objective.strip(),
        [ontology.ontology_id for ontology in ontology_list],
    )
    if frame.frame_id != expected_frame_id:
        frame = frame.model_copy(update={"frame_id": expected_frame_id})
    validate_agent_frame_support(frame, _known_support_ids(ontology_list))
    return frame
