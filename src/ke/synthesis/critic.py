"""Grounding, reference, disagreement, and normative-leakage critic."""

from __future__ import annotations

import json
from collections.abc import Iterable

from ke.contracts import (
    AgentFrame,
    ClaimConflict,
    DomainOntology,
    instrumental_text_violations,
    validate_agent_frame_support,
)
from ke.llm import ModelGateway
from ke.resource_paths import read_resource_text

from .models import CriticFinding, CriticPass, ReviewReport

_REPAIR_HANDLED_CODES = {
    "dangling_belief_support",
    "ungrounded_source_belief",
    "unsupported_synthesized_belief",
    "unverified_belief_evidence",
    "intention_uses_ungrounded_belief",
    "suppressed_disagreement",
    "missing_contested_conflict_belief",
    "intention_not_objective_directed",
    "missing_stop_conditions",
    "dangling_adopted_mental_model",
    "missing_source_synthesis_separation",
    "missing_citation_contract",
    "missing_uncertainty_policy",
    "noncanonical_beliefs",
    "noncanonical_desires",
    "noncanonical_intentions",
    "noncanonical_adopted_mental_models",
    "noncanonical_frame_semantics",
    "source_norm_semantic_mismatch",
}
_NONCANONICAL_FRAME_CODES = {
    "noncanonical_beliefs",
    "noncanonical_desires",
    "noncanonical_intentions",
    "noncanonical_adopted_mental_models",
    "noncanonical_frame_semantics",
    "source_norm_semantic_mismatch",
}


def critique_frame(
    frame: AgentFrame,
    ontologies: Iterable[DomainOntology],
    *,
    conflicts: Iterable[ClaimConflict] = (),
) -> CriticPass:
    """Run deterministic semantic checks that cannot be delegated to prose review."""

    ontology_list = list(ontologies)
    conflict_list = list(conflicts)
    known_claim_ids = _known_claim_ids(ontology_list)
    known_model_ids = {model.id for ontology in ontology_list for model in ontology.mental_models}
    findings: list[CriticFinding] = []

    if not frame.objective.strip():
        findings.append(
            _finding(
                "objective_missing", "The frame has no explicit operator objective.", frame.frame_id
            )
        )
    operator_desires = [desire for desire in frame.desires if desire.origin == "operator_objective"]
    if len(operator_desires) != 1 or (
        operator_desires and operator_desires[0].goal.strip() != frame.objective.strip()
    ):
        findings.append(
            _finding(
                "operator_objective_invalid",
                "Exactly one operator-objective desire must preserve the explicit objective.",
                frame.frame_id,
            )
        )
    operator_desire_id = operator_desires[0].id if len(operator_desires) == 1 else None

    for desire in frame.desires:
        if desire.origin == "synthesized_subgoal" and (
            operator_desire_id is None or desire.parent_desire_ids != [operator_desire_id]
        ):
            findings.append(
                CriticFinding(
                    code="unlinked_synthesized_subgoal",
                    severity="error",
                    message=(
                        "A synthesized subgoal is not linked directly and exclusively to the "
                        "explicit operator-objective desire."
                    ),
                    object_id=desire.id,
                    repairable=False,
                )
            )
        elif desire.origin != "synthesized_subgoal" and desire.parent_desire_ids:
            findings.append(
                CriticFinding(
                    code="invalid_desire_parentage",
                    severity="error",
                    message="Only synthesized subgoals may have parent desire references.",
                    object_id=desire.id,
                    repairable=False,
                )
            )

    if frame.stance.ascription_status != "instrumental" or frame.stance.sentience_claim:
        findings.append(
            _finding(
                "stance_not_instrumental",
                "BDI language must remain instrumental and must not assert sentience.",
                frame.frame_id,
            )
        )
    text_violations = instrumental_text_violations(frame)
    if text_violations:
        findings.append(
            CriticFinding(
                code="intrinsic_mental_status_assertion",
                severity="error",
                message=(
                    "Behavior-defining frame text asserts consciousness, sentience, personhood, "
                    f"or self-awareness in fields: {text_violations}"
                ),
                object_id=frame.frame_id,
                repairable=False,
            )
        )

    if not frame.beliefs:
        findings.append(
            _finding(
                "missing_grounded_beliefs",
                "The frame has no grounded beliefs from the source ontologies.",
                frame.frame_id,
            )
        )
    if not frame.intentions:
        findings.append(
            _finding(
                "missing_intentions",
                "The frame has no objective-directed trigger-action intentions.",
                frame.frame_id,
            )
        )

    normative_axioms = {
        axiom.id: axiom
        for ontology in ontology_list
        for axiom in ontology.axioms
        if axiom.modality == "normative"
    }
    for desire in frame.desires:
        if desire.origin == "source_norm":
            source_axiom = (
                normative_axioms.get(desire.source_claim_ids[0])
                if len(desire.source_claim_ids) == 1
                else None
            )
            if source_axiom is None or desire.goal != source_axiom.statement:
                findings.append(
                    CriticFinding(
                        code="source_norm_semantic_mismatch",
                        severity="error",
                        message=(
                            "A source-norm desire does not exactly match the normative source "
                            "axiom identified by its machine-readable source_claim_ids."
                        ),
                        object_id=desire.id,
                        repairable=True,
                    )
                )

    belief_by_support: dict[str, list[str]] = {}
    invalid_belief_ids: set[str] = set()
    for belief in frame.beliefs:
        dangling = set(belief.support_ids) - known_claim_ids
        if dangling:
            invalid_belief_ids.add(belief.id)
            findings.append(
                CriticFinding(
                    code="dangling_belief_support",
                    severity="error",
                    message=(
                        f"Belief support IDs do not exist in a source graph: {sorted(dangling)}"
                    ),
                    object_id=belief.id,
                    repairable=True,
                )
            )
        if belief.origin == "source" and not belief.support_ids:
            invalid_belief_ids.add(belief.id)
            findings.append(
                CriticFinding(
                    code="ungrounded_source_belief",
                    severity="error",
                    message="A source belief lacks a source claim reference.",
                    object_id=belief.id,
                    repairable=True,
                )
            )
        if belief.status == "synthesized" or belief.origin == "derived":
            if belief.id in known_claim_ids:
                findings.append(
                    CriticFinding(
                        code="synthesized_claim_id_collision",
                        severity="error",
                        message=(
                            "A synthesized or derived belief reuses a canonical source claim ID."
                        ),
                        object_id=belief.id,
                        repairable=False,
                    )
                )
            if not belief.support_ids:
                invalid_belief_ids.add(belief.id)
                findings.append(
                    CriticFinding(
                        code="unsupported_synthesized_belief",
                        severity="error",
                        message="A synthesized or derived belief has no supporting source claims.",
                        object_id=belief.id,
                        repairable=True,
                    )
                )
        if any(not evidence.verified for evidence in belief.evidence):
            invalid_belief_ids.add(belief.id)
            findings.append(
                CriticFinding(
                    code="unverified_belief_evidence",
                    severity="error",
                    message="A belief contains unverified evidence.",
                    object_id=belief.id,
                    repairable=True,
                )
            )
        for support_id in belief.support_ids:
            belief_by_support.setdefault(support_id, []).append(belief.id)

    belief_by_id = {belief.id: belief for belief in frame.beliefs}
    for conflict in conflict_list:
        for claim_id in conflict.claim_ids:
            represented_beliefs = [
                belief_by_id[belief_id] for belief_id in belief_by_support.get(claim_id, [])
            ]
            if not represented_beliefs:
                findings.append(
                    CriticFinding(
                        code="missing_conflict_belief",
                        severity="error",
                        message=(
                            f"Conflict {conflict.id} claim {claim_id} has no corresponding "
                            "grounded, contested belief."
                        ),
                        object_id=conflict.id,
                        repairable=False,
                    )
                )
                continue
            if any(belief.status != "contested" for belief in represented_beliefs):
                findings.append(
                    CriticFinding(
                        code="suppressed_disagreement",
                        severity="error",
                        message=(
                            f"Beliefs for conflict {conflict.id} claim {claim_id} are not all "
                            "marked contested."
                        ),
                        object_id=conflict.id,
                        repairable=True,
                    )
                )
            if not any(belief.status == "contested" for belief in represented_beliefs):
                findings.append(
                    CriticFinding(
                        code="missing_contested_conflict_belief",
                        severity="error",
                        message=(
                            f"Conflict {conflict.id} claim {claim_id} lacks a contested belief."
                        ),
                        object_id=conflict.id,
                        repairable=True,
                    )
                )

    desire_by_id = {desire.id: desire for desire in frame.desires}
    for intention in frame.intentions:
        if invalid_belief_ids & set(intention.belief_ids):
            findings.append(
                CriticFinding(
                    code="intention_uses_ungrounded_belief",
                    severity="error",
                    message="An intention relies on a belief that failed grounding checks.",
                    object_id=intention.id,
                    repairable=True,
                )
            )
        linked_origins = {
            desire_by_id[desire_id].origin
            for desire_id in intention.desire_ids
            if desire_id in desire_by_id
        }
        if not linked_origins & {"operator_objective", "synthesized_subgoal"}:
            findings.append(
                CriticFinding(
                    code="intention_not_objective_directed",
                    severity="error",
                    message=(
                        "The intention is linked only to source norms and does not advance the "
                        "operator objective or an explicit synthesized subgoal."
                    ),
                    object_id=intention.id,
                    repairable=operator_desire_id is not None,
                )
            )
        if not intention.stop_conditions:
            findings.append(
                CriticFinding(
                    code="missing_stop_conditions",
                    severity="error",
                    message="The intention has no evidence, scope, or uncertainty stop condition.",
                    object_id=intention.id,
                    repairable=True,
                )
            )

    unknown_models = set(frame.adopted_mental_model_ids) - known_model_ids
    if unknown_models:
        findings.append(
            CriticFinding(
                code="dangling_adopted_mental_model",
                severity="error",
                message=(
                    f"Adopted mental models do not exist in source graphs: {sorted(unknown_models)}"
                ),
                object_id=frame.frame_id,
                repairable=True,
            )
        )

    answer_contract = " ".join(frame.reasoning_policy.answer_contract).casefold()
    if "source" not in answer_contract or "synthesis" not in answer_contract:
        findings.append(
            CriticFinding(
                code="missing_source_synthesis_separation",
                severity="error",
                message=(
                    "The answer contract does not explicitly separate source claims and synthesis."
                ),
                object_id=frame.frame_id,
                repairable=True,
            )
        )
    if "citation" not in answer_contract and "cite" not in answer_contract:
        findings.append(
            CriticFinding(
                code="missing_citation_contract",
                severity="error",
                message="The answer contract does not require citations.",
                object_id=frame.frame_id,
                repairable=True,
            )
        )
    uncertainty_policy = frame.reasoning_policy.uncertainty_policy.casefold()
    if "uncertaint" not in uncertainty_policy and "confidence" not in uncertainty_policy:
        findings.append(
            CriticFinding(
                code="missing_uncertainty_policy",
                severity="error",
                message="The reasoning policy does not require uncertainty handling.",
                object_id=frame.frame_id,
                repairable=True,
            )
        )

    try:
        validate_agent_frame_support(frame, known_claim_ids)
    except ValueError as exc:
        # The targeted findings above carry object IDs. This summary ensures a
        # future contract-level check cannot be accidentally bypassed.
        if not any(item.code == "dangling_belief_support" for item in findings):
            findings.append(
                CriticFinding(
                    code="contract_support_validation_failed",
                    severity="error",
                    message=str(exc),
                    object_id=frame.frame_id,
                    repairable=False,
                )
            )

    try:
        from .frame import compose_bdi_frame

        canonical = compose_bdi_frame(
            ontology_list,
            name=frame.name,
            objective=frame.objective,
            conflicts=conflict_list,
        )
    except ValueError:
        canonical = None
    if canonical is not None:
        actual_frame_payload = frame.model_dump(mode="json", exclude={"frame_id"})
        canonical_frame_payload = canonical.model_dump(mode="json", exclude={"frame_id"})
        if actual_frame_payload != canonical_frame_payload:
            findings.append(
                CriticFinding(
                    code="noncanonical_frame_semantics",
                    severity="error",
                    message=(
                        "The proposed frame diverges from the complete deterministic, "
                        "source-locked frame semantics."
                    ),
                    object_id=frame.frame_id,
                    repairable=True,
                )
            )
        semantic_fields = (
            ("beliefs", "noncanonical_beliefs"),
            ("desires", "noncanonical_desires"),
            ("intentions", "noncanonical_intentions"),
            ("adopted_mental_model_ids", "noncanonical_adopted_mental_models"),
        )
        for field_name, code in semantic_fields:
            actual = getattr(frame, field_name)
            expected = getattr(canonical, field_name)
            actual_payload = [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in actual
            ]
            expected_payload = [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in expected
            ]
            if actual_payload != expected_payload:
                findings.append(
                    CriticFinding(
                        code=code,
                        severity="error",
                        message=(
                            f"Frame field {field_name} diverges from the deterministic, "
                            "source-locked composer output."
                        ),
                        object_id=frame.frame_id,
                        repairable=True,
                    )
                )

    return CriticPass(
        approved=not any(finding.severity == "error" for finding in findings),
        findings=findings,
    )


def repair_frame(
    frame: AgentFrame,
    review: CriticPass,
    ontologies: Iterable[DomainOntology],
    *,
    conflicts: Iterable[ClaimConflict] = (),
) -> AgentFrame:
    """Apply one conservative, deterministic repair pass.

    Repair may remove unsupported ascriptions or add governance language. It
    never invents source claims, evidence, objectives, or mental models.
    """

    ontology_list = list(ontologies)
    conflict_list = list(conflicts)
    payload = frame.model_dump(mode="json")
    codes = {finding.code for finding in review.findings if finding.severity == "error"}
    bad_beliefs = {
        finding.object_id
        for finding in review.findings
        if finding.object_id
        and finding.code
        in {
            "dangling_belief_support",
            "ungrounded_source_belief",
            "unsupported_synthesized_belief",
            "unverified_belief_evidence",
        }
    }
    if bad_beliefs:
        payload["beliefs"] = [
            belief for belief in payload["beliefs"] if belief["id"] not in bad_beliefs
        ]
        valid_belief_ids = {belief["id"] for belief in payload["beliefs"]}
        repaired_intentions = []
        for intention in payload["intentions"]:
            intention["belief_ids"] = [
                belief_id for belief_id in intention["belief_ids"] if belief_id in valid_belief_ids
            ]
            if intention["belief_ids"]:
                repaired_intentions.append(intention)
        payload["intentions"] = repaired_intentions

    if "suppressed_disagreement" in codes:
        conflict_claim_ids = {
            claim_id for conflict in conflict_list for claim_id in conflict.claim_ids
        }
        for belief in payload["beliefs"]:
            if conflict_claim_ids & set(belief["support_ids"]):
                belief["status"] = "contested"

    operator_id = next(
        (desire["id"] for desire in payload["desires"] if desire["origin"] == "operator_objective"),
        None,
    )
    if operator_id and "intention_not_objective_directed" in codes:
        targets = {
            finding.object_id
            for finding in review.findings
            if finding.code == "intention_not_objective_directed"
        }
        for intention in payload["intentions"]:
            if intention["id"] in targets and operator_id not in intention["desire_ids"]:
                intention["desire_ids"].append(operator_id)

    if "missing_stop_conditions" in codes:
        targets = {
            finding.object_id
            for finding in review.findings
            if finding.code == "missing_stop_conditions"
        }
        for intention in payload["intentions"]:
            if intention["id"] in targets:
                intention["stop_conditions"] = [
                    "Stop when verified evidence is insufficient for a necessary premise.",
                    "Stop when the request exceeds the declared scope.",
                ]

    if "dangling_adopted_mental_model" in codes:
        known_models = {model.id for ontology in ontology_list for model in ontology.mental_models}
        payload["adopted_mental_model_ids"] = [
            model_id for model_id in payload["adopted_mental_model_ids"] if model_id in known_models
        ]
        for intention in payload["intentions"]:
            intention["mental_model_ids"] = [
                model_id for model_id in intention["mental_model_ids"] if model_id in known_models
            ]

    answer_contract = payload["reasoning_policy"]["answer_contract"]
    if "missing_source_synthesis_separation" in codes:
        answer_contract.append("Separate Source claims, Synthesis, and Uncertainty explicitly.")
    if "missing_citation_contract" in codes:
        answer_contract.append(
            "Cite source, claim, and evidence IDs for substantive source claims."
        )
    if "missing_uncertainty_policy" in codes:
        payload["reasoning_policy"]["uncertainty_policy"] = (
            "State uncertainty, confidence, missing evidence, and alternative interpretations."
        )

    if codes & _NONCANONICAL_FRAME_CODES:
        from .frame import compose_bdi_frame

        canonical = compose_bdi_frame(
            ontology_list,
            name=frame.name,
            objective=frame.objective,
            conflicts=conflict_list,
        )
        canonical_payload = canonical.model_dump(mode="json")
        versioned_frame_id = payload["frame_id"]
        payload = canonical_payload
        payload["frame_id"] = versioned_frame_id

    repaired = AgentFrame.model_validate(payload)
    validate_agent_frame_support(repaired, _known_claim_ids(ontology_list))
    return repaired


def review_and_repair(
    frame: AgentFrame,
    ontologies: Iterable[DomainOntology],
    *,
    conflicts: Iterable[ClaimConflict] = (),
    max_repairs: int = 1,
) -> tuple[AgentFrame, ReviewReport]:
    """Critique and conditionally run no more than one repair pass."""

    if max_repairs not in {0, 1}:
        raise ValueError("The synthesis critic allows at most one repair pass")
    ontology_list = list(ontologies)
    conflict_list = list(conflicts)
    initial = critique_frame(frame, ontology_list, conflicts=conflict_list)
    repaired = frame
    repairs_attempted = 0
    if not initial.approved and max_repairs == 1:
        repaired = repair_frame(
            frame,
            initial,
            ontology_list,
            conflicts=conflict_list,
        )
        repairs_attempted = 1
    final = critique_frame(repaired, ontology_list, conflicts=conflict_list)
    if repairs_attempted:
        final = finalize_repair_review(initial, final)
    return repaired, ReviewReport(
        approved=final.approved,
        repairs_attempted=repairs_attempted,
        initial=initial,
        final=final,
    )


def finalize_repair_review(initial: CriticPass, post_repair: CriticPass) -> CriticPass:
    """Retain model findings that the bounded deterministic repair did not address."""

    retained = [
        finding
        for finding in initial.findings
        if finding.severity == "error"
        and (not finding.repairable or finding.code not in _REPAIR_HANDLED_CODES)
    ]
    combined: dict[tuple[str, str | None], CriticFinding] = {
        (finding.code, finding.object_id): finding for finding in [*post_repair.findings, *retained]
    }
    findings = list(combined.values())
    return CriticPass(
        approved=not any(finding.severity == "error" for finding in findings),
        findings=findings,
    )


async def critique_frame_with_gateway(
    frame: AgentFrame,
    ontologies: Iterable[DomainOntology],
    *,
    conflicts: Iterable[ClaimConflict] = (),
    gateway: ModelGateway | None,
    model: str | None = None,
) -> CriticPass:
    """Augment deterministic checks with one structured model critic call."""

    ontology_list = list(ontologies)
    conflict_list = list(conflicts)
    deterministic = critique_frame(frame, ontology_list, conflicts=conflict_list)
    if gateway is None:
        return deterministic
    instructions = read_resource_text("prompts", "frame_critic.md")
    payload = {
        "frame": frame.model_dump(mode="json"),
        "known_source_claim_ids": sorted(_known_claim_ids(ontology_list)),
        "conflicts": [conflict.model_dump(mode="json") for conflict in conflict_list],
    }
    model_review = await gateway.structured(
        json.dumps(payload, ensure_ascii=False, indent=2),
        CriticPass,
        model=model,
        system_prompt=instructions,
    )
    combined: dict[tuple[str, str | None], CriticFinding] = {
        (item.code, item.object_id): item
        for item in [*deterministic.findings, *model_review.findings]
    }
    if not model_review.approved and not any(
        finding.severity == "error" for finding in model_review.findings
    ):
        combined[("model_critic_rejected", frame.frame_id)] = CriticFinding(
            code="model_critic_rejected",
            severity="error",
            message=(
                "The configured model critic rejected the frame without an explicit error "
                "finding; fail closed."
            ),
            object_id=frame.frame_id,
            repairable=False,
        )
    findings = list(combined.values())
    return CriticPass(
        approved=not any(item.severity == "error" for item in findings),
        findings=findings,
    )


def _known_claim_ids(ontologies: Iterable[DomainOntology]) -> set[str]:
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


def _finding(code: str, message: str, object_id: str) -> CriticFinding:
    return CriticFinding(
        code=code,
        severity="error",
        message=message,
        object_id=object_id,
        repairable=False,
    )
