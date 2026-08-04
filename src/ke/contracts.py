"""Canonical, source-grounded data contracts.

Draft extraction objects intentionally use labels.  Canonical identifiers only
appear after entity resolution, which keeps model output from becoming identity.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContractModel(BaseModel):
    """Strict base model used for persisted artifacts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Evidence(ContractModel):
    source_id: str
    chunk_id: str
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    quote: str = Field(min_length=1)
    quote_sha256: str
    verified: bool = False

    @field_validator("source_id", "chunk_id")
    @classmethod
    def attributed_identifiers_are_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence source_id and chunk_id must not be blank")
        return value

    @model_validator(mode="after")
    def page_range_is_ordered(self) -> Evidence:
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must be greater than or equal to page_start")
        if self.verified:
            expected = hashlib.sha256(self.quote.encode("utf-8")).hexdigest()
            if self.quote_sha256 != expected:
                raise ValueError("verified evidence quote_sha256 must match the exact quotation")
        return self


_DATALOG_TERM = (
    r"(?:[A-Z_][A-Za-z0-9_]*|[a-z][A-Za-z0-9_]*|-?(?:0|[1-9][0-9]*)"
    r'(?:\.[0-9]+)?|"(?:[^"\\]|\\.)*")'
)
_DATALOG_ATOM = rf"[a-z][A-Za-z0-9_]*\(\s*{_DATALOG_TERM}(?:\s*,\s*{_DATALOG_TERM})*\s*\)"
_DATALOG_RULE_RE = re.compile(
    rf"\s*{_DATALOG_ATOM}(?:\s*:-\s*(?:not\s+)?{_DATALOG_ATOM}"
    rf"(?:\s*,\s*(?:not\s+)?{_DATALOG_ATOM})*)?\s*\.\s*"
)
_DATALOG_ATOM_RE = re.compile(_DATALOG_ATOM)
_DATALOG_VARIABLE_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9_]*|_[A-Za-z0-9_]*)\b")


def _is_valid_datalog_rule(expression: str) -> bool:
    """Validate a deliberately small, function-free Datalog rule subset."""

    if _DATALOG_RULE_RE.fullmatch(expression) is None:
        return False
    atoms = list(_DATALOG_ATOM_RE.finditer(expression))
    if not atoms:
        return False
    head_variables = set(_DATALOG_VARIABLE_RE.findall(atoms[0].group(0)))
    if ":-" not in expression:
        return not head_variables
    positive_body_variables: set[str] = set()
    negative_body_variables: set[str] = set()
    for atom in atoms[1:]:
        prefix = expression[max(0, atom.start() - 5) : atom.start()]
        variables = set(_DATALOG_VARIABLE_RE.findall(atom.group(0)))
        if re.search(r"not\s+$", prefix):
            negative_body_variables.update(variables)
        else:
            positive_body_variables.update(variables)
    return head_variables <= positive_body_variables and (
        negative_body_variables <= positive_body_variables
    )


class Term(ContractModel):
    id: str
    preferred_label: str
    aliases: list[str] = Field(default_factory=list)
    kind: Literal[
        "entity",
        "concept",
        "process",
        "role",
        "value",
        "artifact",
        "system",
        "property",
        "event",
    ]
    definition: str
    broader_ids: list[str] = Field(default_factory=list)
    narrower_ids: list[str] = Field(default_factory=list)
    related_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class Relation(ContractModel):
    id: str
    subject_id: str
    predicate: str
    object_id: str
    qualifiers: dict[str, str] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class Axiom(ContractModel):
    id: str
    statement: str
    modality: Literal[
        "definitional",
        "descriptive",
        "causal",
        "normative",
        "heuristic",
        "methodological",
    ]
    epistemic_status: Literal[
        "asserted", "reported", "hypothetical", "contested", "conditional", "derived"
    ]
    attribution: str = "author"
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    formalism: Literal["none", "fol", "datalog", "owl_manchester"] = "none"
    formal_expression: str | None = None
    evidence: list[Evidence] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def formal_expression_matches_validated_formalism(self) -> Axiom:
        expression = self.formal_expression
        if self.formalism == "none":
            if expression is not None:
                raise ValueError("formalism='none' requires formal_expression=None")
            return self
        if expression is None or not expression.strip():
            raise ValueError("a non-none formalism requires a nonblank formal_expression")
        if self.formalism == "datalog" and _is_valid_datalog_rule(expression):
            return self
        raise ValueError(
            f"formalism {self.formalism!r} is not supported by a fail-closed syntax validator"
        )


class MentalModel(ContractModel):
    id: str
    name: str
    purpose: str
    applicable_when: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    mechanism: list[str] = Field(default_factory=list)
    procedure: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class ClaimConflict(ContractModel):
    id: str
    claim_ids: list[str] = Field(min_length=2)
    kind: Literal["terminological", "empirical", "causal", "normative", "methodological", "scope"]
    description: str
    resolution: Literal["unresolved", "conditionalized", "source-preferred", "compatible"] = (
        "unresolved"
    )


class DomainOntology(ContractModel):
    schema_version: str = "1.0"
    ontology_id: str
    title: str
    source_ids: list[str]
    abstract: str
    abstract_support_ids: list[str]
    scope_in: list[str] = Field(default_factory=list)
    scope_out: list[str] = Field(default_factory=list)
    terms: list[Term]
    relations: list[Relation]
    axioms: list[Axiom]
    mental_models: list[MentalModel]
    conflicts: list[ClaimConflict] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_canonical_graph(self) -> DomainOntology:
        groups = [self.terms, self.relations, self.axioms, self.mental_models, self.conflicts]
        ids = [item.id for group in groups for item in group]
        if len(ids) != len(set(ids)):
            raise ValueError("canonical ontology IDs must be globally unique")

        term_ids = {term.id for term in self.terms}
        for term in self.terms:
            refs = set(term.broader_ids + term.narrower_ids + term.related_ids)
            dangling = refs - term_ids
            if dangling:
                raise ValueError(f"term {term.id!r} has dangling term references: {dangling}")
            if term.id in refs:
                raise ValueError(f"term {term.id!r} cannot refer to itself")
        for relation in self.relations:
            dangling = {relation.subject_id, relation.object_id} - term_ids
            if dangling:
                raise ValueError(
                    f"relation {relation.id!r} has dangling term references: {dangling}"
                )

        object_ids = set(ids) - {conflict.id for conflict in self.conflicts}
        dangling_abstract = set(self.abstract_support_ids) - object_ids
        if dangling_abstract:
            raise ValueError(f"abstract has dangling support IDs: {dangling_abstract}")
        for conflict in self.conflicts:
            dangling = set(conflict.claim_ids) - object_ids
            if dangling:
                raise ValueError(f"conflict {conflict.id!r} has dangling claim IDs: {dangling}")

        grounded = [*self.terms, *self.relations, *self.axioms, *self.mental_models]
        declared_sources = set(self.source_ids)
        for item in grounded:
            if not item.evidence or any(not evidence.verified for evidence in item.evidence):
                raise ValueError(f"canonical object {item.id!r} contains unverified evidence")
            foreign_sources = {
                evidence.source_id
                for evidence in item.evidence
                if evidence.source_id not in declared_sources
            }
            if foreign_sources:
                raise ValueError(
                    f"canonical object {item.id!r} cites undeclared source IDs: {foreign_sources}"
                )
        return self


class StanceSpec(ContractModel):
    mode: Literal["intentional"] = "intentional"
    tradition: Literal["Dennettian intentional stance"] = "Dennettian intentional stance"
    ascription_status: Literal["instrumental"] = "instrumental"
    sentience_claim: bool = False
    rationality_assumption: Literal[
        "Boundedly rational relative to declared beliefs, goals, available evidence, "
        "tools, and constraints."
    ] = (
        "Boundedly rational relative to declared beliefs, goals, "
        "available evidence, tools, and constraints."
    )

    @model_validator(mode="after")
    def reject_sentience_claim(self) -> StanceSpec:
        if self.sentience_claim:
            raise ValueError("the intentional stance must not assert sentience")
        return self


class Belief(ContractModel):
    id: str
    proposition: str
    status: Literal["accepted", "conditional", "contested", "synthesized"]
    origin: Literal["source", "derived", "operator"]
    support_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def source_beliefs_are_supported(self) -> Belief:
        if self.origin == "source" and not (self.support_ids or self.evidence):
            raise ValueError("source beliefs require evidence or supporting source claims")
        if self.evidence and any(not evidence.verified for evidence in self.evidence):
            raise ValueError("belief evidence must be verified")
        return self


CANONICAL_PROVENANCE_SUBGOAL = (
    "Preserve source attribution and cite verified evidence for substantive claims."
)
CANONICAL_CONFLICT_SUBGOAL = (
    "Surface material source disagreements instead of silently harmonizing them."
)
CANONICAL_SYNTHESIZED_SUBGOALS = frozenset(
    {CANONICAL_PROVENANCE_SUBGOAL, CANONICAL_CONFLICT_SUBGOAL}
)


class Desire(ContractModel):
    id: str
    goal: str
    origin: Literal["operator_objective", "source_norm", "synthesized_subgoal"]
    parent_desire_ids: list[str] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    priority: int = Field(ge=1, le=5)
    rationale: str
    success_criteria: list[str]


class Intention(ContractModel):
    id: str
    trigger: str
    action_policy: str
    expected_result: str
    belief_ids: list[str]
    desire_ids: list[str]
    mental_model_ids: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)


class ReasoningPolicy(ContractModel):
    deliberation_cycle: list[str]
    model_selection_policy: str
    evidence_policy: str
    conflict_policy: str
    uncertainty_policy: str
    answer_contract: list[str]


class AgentFrame(ContractModel):
    schema_version: str = "1.0"
    frame_id: str
    name: str
    role: str
    objective: str
    stance: StanceSpec = Field(default_factory=StanceSpec)
    scope_in: list[str]
    scope_out: list[str]
    capabilities: list[str]
    limitations: list[str]
    anti_goals: list[str]
    beliefs: list[Belief]
    desires: list[Desire]
    intentions: list[Intention]
    adopted_mental_model_ids: list[str]
    reasoning_policy: ReasoningPolicy

    @model_validator(mode="after")
    def validate_bdi_references(self) -> AgentFrame:
        groups = [self.beliefs, self.desires, self.intentions]
        ids = [item.id for group in groups for item in group]
        if len(ids) != len(set(ids)):
            raise ValueError("belief, desire, and intention IDs must be globally unique")

        belief_ids = {belief.id for belief in self.beliefs}
        desire_ids = {desire.id for desire in self.desires}
        mental_model_ids = set(self.adopted_mental_model_ids)
        if len(mental_model_ids) != len(self.adopted_mental_model_ids):
            raise ValueError("adopted mental-model IDs must be unique")

        for intention in self.intentions:
            if not intention.belief_ids or not intention.desire_ids:
                raise ValueError(f"intention {intention.id!r} must link to a belief and a desire")
            dangling_beliefs = set(intention.belief_ids) - belief_ids
            dangling_desires = set(intention.desire_ids) - desire_ids
            dangling_models = set(intention.mental_model_ids) - mental_model_ids
            if dangling_beliefs or dangling_desires or dangling_models:
                raise ValueError(
                    f"intention {intention.id!r} has dangling references: "
                    f"beliefs={dangling_beliefs}, desires={dangling_desires}, "
                    f"mental_models={dangling_models}"
                )

        operator_desires = [
            desire for desire in self.desires if desire.origin == "operator_objective"
        ]
        if len(operator_desires) != 1:
            raise ValueError(
                "an agent frame requires exactly one explicit operator objective desire"
            )
        if operator_desires[0].goal.strip() != self.objective.strip():
            raise ValueError("the operator-objective desire must exactly preserve the objective")
        operator_desire_id = operator_desires[0].id
        for desire in self.desires:
            if desire.origin == "synthesized_subgoal":
                if desire.parent_desire_ids != [operator_desire_id]:
                    raise ValueError(
                        f"synthesized subgoal {desire.id!r} must link directly to the "
                        "operator-objective desire"
                    )
                if desire.goal not in CANONICAL_SYNTHESIZED_SUBGOALS:
                    raise ValueError(
                        f"synthesized subgoal {desire.id!r} is not an approved deterministic "
                        "procedural subgoal"
                    )
                if desire.source_claim_ids:
                    raise ValueError("synthesized subgoals cannot masquerade as source claims")
            elif desire.parent_desire_ids:
                raise ValueError(
                    f"non-subgoal desire {desire.id!r} cannot have parent desire references"
                )
            elif desire.origin == "source_norm" and len(desire.source_claim_ids) != 1:
                raise ValueError("source-norm desires require exactly one source claim reference")
            elif desire.origin == "operator_objective" and desire.source_claim_ids:
                raise ValueError("the operator objective cannot masquerade as a source claim")
        mental_status_violations = instrumental_text_violations(self)
        if mental_status_violations:
            raise ValueError(
                "frame text must not assert consciousness, sentience, personhood, or "
                f"self-awareness: {mental_status_violations}"
            )
        return self


_MENTAL_STATUS_PATTERN = re.compile(
    r"\b(conscious(?:ness)?|sentien(?:t|ce)|person(?:hood)?|self[- ]aware(?:ness)?|"
    r"subjective experience|inner experience)\b",
    re.IGNORECASE,
)
_STATUS_CLAUSE_SPLIT = re.compile(r"[.;\n]+")
_SAFE_STATUS_DENIALS = (
    re.compile(r"\bno\b(?!\s+doubt\b).{0,60}$", re.IGNORECASE),
    re.compile(r"\bwithout\b.{0,50}$", re.IGNORECASE),
    re.compile(
        r"\b(?:do|does|must|should)\s+not\b.{0,80}\b(?:claim|assert|imply|present|"
        r"describe)\b.{0,35}$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcannot\b.{0,60}\b(?:claim|assert|imply|present|describe)\b.{0,35}$",
        re.IGNORECASE,
    ),
    re.compile(r"\bnot\s+(?:a\s+(?:claim|assertion|statement)\s+of\s+)?$", re.IGNORECASE),
    re.compile(
        r"\b(?:analy[sz]e|study|discuss|research|evaluate|explain|compare|reason\s+about)\b"
        r".{0,60}$",
        re.IGNORECASE,
    ),
)


def _asserts_mental_status(text: str) -> bool:
    for clause in _STATUS_CLAUSE_SPLIT.split(text):
        for match in _MENTAL_STATUS_PATTERN.finditer(clause):
            prefix = clause[max(0, match.start() - 100) : match.start()]
            if re.search(r"\bnot\s+only\b", prefix, re.IGNORECASE):
                return True
            if not any(pattern.search(prefix) for pattern in _SAFE_STATUS_DENIALS):
                return True
    return False


def instrumental_text_violations(frame: AgentFrame) -> list[str]:
    """Return behavior-defining fields that assert an intrinsic mental status."""

    policy = frame.reasoning_policy
    fields: dict[str, list[str]] = {
        "name": [frame.name],
        "role": [frame.role],
        "capabilities": frame.capabilities,
        "limitations": frame.limitations,
        "anti_goals": frame.anti_goals,
        "reasoning_policy.deliberation_cycle": policy.deliberation_cycle,
        "reasoning_policy.model_selection_policy": [policy.model_selection_policy],
        "reasoning_policy.evidence_policy": [policy.evidence_policy],
        "reasoning_policy.conflict_policy": [policy.conflict_policy],
        "reasoning_policy.uncertainty_policy": [policy.uncertainty_policy],
        "reasoning_policy.answer_contract": policy.answer_contract,
    }
    for desire in frame.desires:
        if desire.origin != "source_norm":
            fields[f"desire:{desire.id}"] = [
                desire.goal,
                desire.rationale,
                *desire.success_criteria,
            ]
    for intention in frame.intentions:
        fields[f"intention:{intention.id}"] = [
            intention.trigger,
            intention.action_policy,
            intention.expected_result,
            *intention.constraints,
            *intention.stop_conditions,
        ]
    return [
        field_name
        for field_name, values in fields.items()
        if any(_asserts_mental_status(value) for value in values)
    ]


# Draft schemas: labels are resolved to canonical IDs only after extraction.
TermKind = Literal[
    "entity", "concept", "process", "role", "value", "artifact", "system", "property", "event"
]


class DraftTerm(ContractModel):
    label: str
    aliases: list[str] = Field(default_factory=list)
    kind: TermKind = "concept"
    definition: str
    broader_labels: list[str] = Field(default_factory=list)
    narrower_labels: list[str] = Field(default_factory=list)
    related_labels: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class DraftRelation(ContractModel):
    subject_label: str
    predicate: str
    object_label: str
    qualifiers: dict[str, str] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class DraftAxiom(ContractModel):
    statement: str
    modality: Literal[
        "definitional", "descriptive", "causal", "normative", "heuristic", "methodological"
    ]
    epistemic_status: Literal[
        "asserted", "reported", "hypothetical", "contested", "conditional", "derived"
    ]
    attribution: str = "author"
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class DraftMentalModel(ContractModel):
    name: str
    purpose: str
    applicable_when: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    mechanism: list[str] = Field(default_factory=list)
    procedure: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class ExtractionFragment(ContractModel):
    source_id: str
    chunk_id: str
    section_summary: str
    terms: list[DraftTerm] = Field(default_factory=list)
    relations: list[DraftRelation] = Field(default_factory=list)
    axioms: list[DraftAxiom] = Field(default_factory=list)
    mental_models: list[DraftMentalModel] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


def validate_agent_frame_support(frame: AgentFrame, known_support_ids: set[str]) -> None:
    """Validate links from frame beliefs into an integrated ontology."""

    for belief in frame.beliefs:
        dangling = set(belief.support_ids) - known_support_ids
        if dangling:
            raise ValueError(f"belief {belief.id!r} has dangling support IDs: {dangling}")
        if belief.origin == "source" and not (belief.evidence or belief.support_ids):
            raise ValueError(f"source belief {belief.id!r} has no source support")
