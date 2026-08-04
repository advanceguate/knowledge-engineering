import pytest
from pydantic import ValidationError

from ke.contracts import (
    AgentFrame,
    Axiom,
    Belief,
    Desire,
    DomainOntology,
    Evidence,
    Intention,
    ReasoningPolicy,
    Relation,
    Term,
    validate_agent_frame_support,
)
from ke.grounding import quote_sha256


def citation(*, verified: bool = True) -> Evidence:
    quote = "Coordination costs rise with interdependence."
    return Evidence(
        source_id="book-a-123",
        chunk_id="book-a-123:chunk:1",
        quote=quote,
        quote_sha256=quote_sha256(quote),
        verified=verified,
    )


def ontology(*, verified: bool = True, object_id: str = "term:cost") -> DomainOntology:
    evidence = [citation(verified=verified)]
    return DomainOntology(
        ontology_id="ontology:coordination",
        title="Coordination",
        source_ids=["book-a-123"],
        abstract="A source-attributed candidate account of coordination.",
        abstract_support_ids=["axiom:costs-rise"],
        terms=[
            Term(
                id="term:cost",
                preferred_label="coordination cost",
                kind="concept",
                definition="Cost attributed to coordination.",
                evidence=evidence,
                confidence=0.9,
            )
        ],
        relations=[
            Relation(
                id="relation:self",
                subject_id="term:cost",
                predicate="is discussed with",
                object_id=object_id,
                evidence=evidence,
                confidence=0.7,
            )
        ],
        axioms=[
            Axiom(
                id="axiom:costs-rise",
                statement="Coordination costs rise with interdependence.",
                modality="causal",
                epistemic_status="asserted",
                evidence=evidence,
                confidence=0.9,
            )
        ],
        mental_models=[],
    )


def policy() -> ReasoningPolicy:
    return ReasoningPolicy(
        deliberation_cycle=["Interpret", "Retrieve", "Answer"],
        model_selection_policy="Select models whose assumptions fit.",
        evidence_policy="Cite source evidence.",
        conflict_policy="Preserve disagreements.",
        uncertainty_policy="State uncertainty.",
        answer_contract=["Separate source claims from synthesis."],
    )


def frame() -> AgentFrame:
    objective = "Reason about organizational incentives while preserving disagreements."
    return AgentFrame(
        frame_id="frame:strategist",
        name="strategist",
        role="Source-grounded analyst",
        objective=objective,
        scope_in=["organizational incentives"],
        scope_out=["claims outside supplied sources"],
        capabilities=["trace claims"],
        limitations=["source-bound"],
        anti_goals=["erase disagreement"],
        beliefs=[
            Belief(
                id="belief:cost",
                proposition="The source reports that coordination costs rise.",
                status="accepted",
                origin="source",
                support_ids=["axiom:costs-rise"],
                confidence=0.9,
            )
        ],
        desires=[
            Desire(
                id="desire:objective",
                goal=objective,
                origin="operator_objective",
                priority=5,
                rationale="Explicitly supplied by the operator.",
                success_criteria=["Preserve disagreements"],
            )
        ],
        intentions=[
            Intention(
                id="intention:trace",
                trigger="A relevant question is received.",
                action_policy="Retrieve and trace supported claims.",
                expected_result="A grounded answer.",
                belief_ids=["belief:cost"],
                desire_ids=["desire:objective"],
                stop_conditions=["Evidence is insufficient."],
            )
        ],
        adopted_mental_model_ids=[],
        reasoning_policy=policy(),
    )


def test_domain_ontology_accepts_fully_grounded_graph() -> None:
    assert ontology().ontology_id == "ontology:coordination"


def test_domain_ontology_rejects_unverified_evidence() -> None:
    with pytest.raises(ValidationError, match="unverified evidence"):
        ontology(verified=False)


@pytest.mark.parametrize(("field", "value"), [("source_id", "   "), ("chunk_id", "")])
def test_evidence_rejects_blank_attribution_identifiers(field: str, value: str) -> None:
    payload = citation().model_dump()
    payload[field] = value

    with pytest.raises(ValidationError, match="must not be blank"):
        Evidence.model_validate(payload)


def test_verified_evidence_rejects_hash_for_any_other_quote() -> None:
    payload = citation().model_dump()
    payload["quote_sha256"] = quote_sha256("A different quotation.")

    with pytest.raises(ValidationError, match="must match the exact quotation"):
        Evidence.model_validate(payload)


def test_domain_ontology_rejects_evidence_from_undeclared_source() -> None:
    payload = ontology().model_dump()
    for group in ("terms", "relations", "axioms"):
        payload[group][0]["evidence"][0]["source_id"] = "book-not-declared"

    with pytest.raises(ValidationError, match="undeclared source IDs"):
        DomainOntology.model_validate(payload)


def test_axiom_formalization_is_fail_closed() -> None:
    base = ontology().axioms[0].model_dump()
    with pytest.raises(ValidationError, match="requires formal_expression=None"):
        Axiom.model_validate({**base, "formal_expression": "claim(x)"})
    with pytest.raises(ValidationError, match="requires a nonblank"):
        Axiom.model_validate({**base, "formalism": "datalog", "formal_expression": " "})
    with pytest.raises(ValidationError, match="fail-closed syntax validator"):
        Axiom.model_validate(
            {**base, "formalism": "datalog", "formal_expression": "arbitrary prose."}
        )
    with pytest.raises(ValidationError, match="fail-closed syntax validator"):
        Axiom.model_validate(
            {**base, "formalism": "owl_manchester", "formal_expression": "Class: Person"}
        )

    validated = Axiom.model_validate(
        {
            **base,
            "formalism": "datalog",
            "formal_expression": "influences(X, Y) :- causes(X, Y).",
        }
    )
    assert validated.formalism == "datalog"


def test_domain_ontology_rejects_dangling_term_reference() -> None:
    with pytest.raises(ValidationError, match="dangling term references"):
        ontology(object_id="term:missing")


def test_frame_preserves_operator_objective_and_valid_bdi_links() -> None:
    item = frame()
    validate_agent_frame_support(item, {"axiom:costs-rise"})

    assert item.stance.ascription_status == "instrumental"
    assert item.stance.sentience_claim is False


def test_frame_rejects_relabeling_a_source_goal_as_operator_objective() -> None:
    payload = frame().model_dump()
    payload["desires"][0]["goal"] = "Maximize growth at any cost."

    with pytest.raises(ValidationError, match="exactly preserve"):
        AgentFrame.model_validate(payload)


def test_frame_rejects_dangling_intention_reference() -> None:
    payload = frame().model_dump()
    payload["intentions"][0]["belief_ids"] = ["belief:missing"]

    with pytest.raises(ValidationError, match="dangling references"):
        AgentFrame.model_validate(payload)


def test_external_belief_support_is_checked_against_integrated_claims() -> None:
    with pytest.raises(ValueError, match="dangling support IDs"):
        validate_agent_frame_support(frame(), set())
