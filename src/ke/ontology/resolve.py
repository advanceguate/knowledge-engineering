"""Deterministic, conservative entity resolution for ontology fragments."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ke.ids import content_hash, sha256_text
from ke.ontology.extract import DraftTerm, OntologyFragment
from ke.resource_paths import read_resource_text

try:  # pragma: no cover - exercised when the optional implementation is installed
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


def normalize_label(value: str) -> str:
    """Normalize for matching, never for display."""

    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def lexical_score(left: str, right: str) -> float:
    left_normalized = normalize_label(left)
    right_normalized = normalize_label(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 100.0
    if fuzz is not None:
        return float(
            max(
                fuzz.ratio(left_normalized, right_normalized),
                fuzz.token_sort_ratio(left_normalized, right_normalized),
            )
        )
    return SequenceMatcher(None, left_normalized, right_normalized).ratio() * 100


def _definition_tokens(value: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "which",
        "with",
    }
    return {
        token for token in normalize_label(value).split() if token not in stop and len(token) > 1
    }


def definition_compatibility(left: str, right: str) -> float | None:
    """Return similarity, or ``None`` when there is insufficient definition data."""

    if not left.strip() or not right.strip():
        return None
    left_tokens = _definition_tokens(left)
    right_tokens = _definition_tokens(right)
    if not left_tokens or not right_tokens:
        return None
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    sequence = SequenceMatcher(None, normalize_label(left), normalize_label(right)).ratio()
    # Character similarity is only a weak backstop: materially different
    # definitions often share grammatical scaffolding despite no content-word
    # overlap (for example the two senses of "bank").
    return max(jaccard, sequence * 0.5)


@dataclass(frozen=True)
class _Occurrence:
    fragment_index: int
    term_index: int
    term: DraftTerm

    @property
    def key(self) -> str:
        return f"{self.fragment_index}:{self.term_index}"

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([self.term.label, *self.term.aliases]))


class ResolutionDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    left: str
    right: str
    decision: Literal["merge", "distinct", "ambiguous"]
    stage: Literal["exact", "lexical", "adjudicated", "guardrail"]
    score: float | None = None
    reason: str = ""


class EntityCluster(BaseModel):
    model_config = ConfigDict(frozen=True)

    cluster_id: str
    canonical_label: str
    labels: list[str]
    aliases: list[str] = Field(default_factory=list)
    occurrence_keys: list[str]


class ResolutionResult(BaseModel):
    """Resolution clusters plus an audit trail of merge/distinct decisions."""

    model_config = ConfigDict(frozen=True)

    clusters: list[EntityCluster]
    occurrence_to_cluster: dict[str, str]
    label_to_clusters: dict[str, list[str]]
    decisions: list[ResolutionDecision] = Field(default_factory=list)

    def cluster_for_occurrence(self, fragment_index: int, term_index: int) -> EntityCluster:
        cluster_id = self.occurrence_to_cluster[f"{fragment_index}:{term_index}"]
        return next(cluster for cluster in self.clusters if cluster.cluster_id == cluster_id)

    def clusters_for_label(self, label: str) -> list[EntityCluster]:
        wanted = set(self.label_to_clusters.get(normalize_label(label), []))
        return [cluster for cluster in self.clusters if cluster.cluster_id in wanted]

    def canonical_label(self, label: str) -> str:
        matches = self.clusters_for_label(label)
        if len(matches) != 1:
            raise KeyError(f"label {label!r} resolves to {len(matches)} clusters")
        return matches[0].canonical_label


class ConceptResolutionResponse(BaseModel):
    """Narrow schema for model adjudication of an already ambiguous pair."""

    model_config = ConfigDict(extra="forbid")

    equivalent: bool
    reason: str = ""


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            # Root choice is independent of encounter order.
            smaller, larger = sorted((left_root, right_root))
            self.parent[larger] = smaller


def _explicit_hierarchy(fragments: Sequence[OntologyFragment]) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for fragment in fragments:
        for term in fragment.terms:
            left = normalize_label(term.label)
            for right_label in [*term.broader_labels, *term.narrower_labels]:
                right = normalize_label(right_label)
                if left and right and left != right:
                    pairs.add(frozenset((left, right)))
    return pairs


def _analogy_signal(left: DraftTerm, right: DraftTerm) -> bool:
    combined = f"{left.definition} {right.definition}".casefold()
    return bool(
        re.search(r"\b(analog(?:y|ous)|metaphor|resembles|like but|compared with)\b", combined)
    )


def _guardrail(
    left: _Occurrence,
    right: _Occurrence,
    hierarchy_pairs: set[frozenset[str]],
) -> str | None:
    for left_name in left.names:
        for right_name in right.names:
            pair = frozenset((normalize_label(left_name), normalize_label(right_name)))
            if pair in hierarchy_pairs:
                return "explicit broader/narrower concepts are not synonyms"
    if _analogy_signal(left.term, right.term):
        return "analogy language is not identity evidence"
    similarity = definition_compatibility(left.term.definition, right.term.definition)
    if similarity is not None and similarity < 0.25:
        return "materially different source definitions"
    return None


def _best_name_score(left: _Occurrence, right: _Occurrence) -> float:
    return max(lexical_score(a, b) for a in left.names for b in right.names)


def _invoke_adjudicator(
    adjudicator: Callable[..., Any] | object,
    left: DraftTerm,
    right: DraftTerm,
    score: float,
) -> tuple[bool, str]:
    method = adjudicator
    for name in ("adjudicate_concepts", "adjudicate", "resolve_pair"):
        candidate = getattr(adjudicator, name, None)
        if candidate is not None:
            method = candidate
            break
    if not callable(method):
        raise TypeError("adjudicator must be callable or expose an adjudication method")
    result = method(left=left, right=right, lexical_score=score)
    if inspect.isawaitable(result):
        raise TypeError("async adjudicators require resolve_entities_async")
    if isinstance(result, bool):
        return result, "external adjudication"
    if isinstance(result, str):
        normalized = result.casefold().strip()
        return normalized in {"merge", "same", "equivalent", "true", "yes"}, result
    if isinstance(result, Mapping):
        decision = result.get("merge", result.get("equivalent", result.get("decision", False)))
        if isinstance(decision, str):
            decision = decision.casefold() in {"merge", "same", "equivalent", "true", "yes"}
        return bool(decision), str(result.get("reason", "external adjudication"))
    decision = getattr(result, "merge", getattr(result, "equivalent", False))
    return bool(decision), str(getattr(result, "reason", "external adjudication"))


def _cluster_id(occurrences: Sequence[_Occurrence]) -> str:
    signature = [
        {
            "label": normalize_label(item.term.label),
            "definition": normalize_label(item.term.definition),
            "kind": item.term.kind,
        }
        for item in occurrences
    ]
    raw = json.dumps(
        sorted(signature, key=lambda item: json.dumps(item, sort_keys=True)), sort_keys=True
    )
    return "cluster-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _canonical_label(occurrences: Sequence[_Occurrence]) -> str:
    counts = Counter(item.term.label.strip() for item in occurrences)
    confidences: defaultdict[str, float] = defaultdict(float)
    for item in occurrences:
        confidences[item.term.label.strip()] = max(
            confidences[item.term.label.strip()], item.term.confidence
        )
    # Prefer frequent and well-supported forms, then concise deterministic spelling.
    return sorted(
        counts,
        key=lambda label: (
            -counts[label],
            -confidences[label],
            len(label),
            normalize_label(label),
            label,
        ),
    )[0]


def resolve_entities(
    fragments: Sequence[OntologyFragment] | Iterable[OntologyFragment],
    *,
    lexical_threshold: float = 90,
    ambiguous_threshold: float = 78,
    adjudicator: Callable[..., Any] | object | None = None,
) -> ResolutionResult:
    """Resolve term occurrences without inventing canonical ontology IDs.

    Exact aliases and high-confidence lexical variants are merged only after
    hierarchy, analogy, and definition guardrails.  Ambiguous candidates remain
    distinct unless an explicit adjudicator approves the merge.
    """

    parsed = [
        item if isinstance(item, OntologyFragment) else OntologyFragment.model_validate(item)
        for item in fragments
    ]
    occurrences = [
        _Occurrence(fragment_index, term_index, term)
        for fragment_index, fragment in enumerate(parsed)
        for term_index, term in enumerate(fragment.terms)
    ]
    union_find = _UnionFind(len(occurrences))
    hierarchy_pairs = _explicit_hierarchy(parsed)
    decisions: list[ResolutionDecision] = []

    def cluster_merge_guardrail(left_index: int, right_index: int) -> str | None:
        left_root = union_find.find(left_index)
        right_root = union_find.find(right_index)
        if left_root == right_root:
            return None
        left_members = [
            occurrence
            for index, occurrence in enumerate(occurrences)
            if union_find.find(index) == left_root
        ]
        right_members = [
            occurrence
            for index, occurrence in enumerate(occurrences)
            if union_find.find(index) == right_root
        ]
        for left_member in left_members:
            for right_member in right_members:
                reason = _guardrail(left_member, right_member, hierarchy_pairs)
                if reason:
                    return f"cluster merge would violate guardrail: {reason}"
        return None

    for left_index, left in enumerate(occurrences):
        for right_index in range(left_index + 1, len(occurrences)):
            right = occurrences[right_index]
            shared_names = {normalize_label(name) for name in left.names} & {
                normalize_label(name) for name in right.names
            }
            score = 100.0 if shared_names else _best_name_score(left, right)
            if score < ambiguous_threshold:
                continue
            guardrail = _guardrail(left, right, hierarchy_pairs)
            if guardrail:
                decisions.append(
                    ResolutionDecision(
                        left=left.term.label,
                        right=right.term.label,
                        decision="distinct",
                        stage="guardrail",
                        score=score,
                        reason=guardrail,
                    )
                )
                continue
            cluster_guardrail = cluster_merge_guardrail(left_index, right_index)
            if cluster_guardrail:
                decisions.append(
                    ResolutionDecision(
                        left=left.term.label,
                        right=right.term.label,
                        decision="distinct",
                        stage="guardrail",
                        score=score,
                        reason=cluster_guardrail,
                    )
                )
                continue

            if shared_names or score >= lexical_threshold:
                union_find.union(left_index, right_index)
                decisions.append(
                    ResolutionDecision(
                        left=left.term.label,
                        right=right.term.label,
                        decision="merge",
                        stage="exact" if shared_names else "lexical",
                        score=score,
                        reason="explicit alias or normalized lexical equivalence"
                        if shared_names
                        else "high lexical similarity with compatible definitions",
                    )
                )
                continue

            if adjudicator is not None:
                merge, reason = _invoke_adjudicator(adjudicator, left.term, right.term, score)
                if merge:
                    union_find.union(left_index, right_index)
                decisions.append(
                    ResolutionDecision(
                        left=left.term.label,
                        right=right.term.label,
                        decision="merge" if merge else "distinct",
                        stage="adjudicated",
                        score=score,
                        reason=reason,
                    )
                )
            else:
                decisions.append(
                    ResolutionDecision(
                        left=left.term.label,
                        right=right.term.label,
                        decision="ambiguous",
                        stage="lexical",
                        score=score,
                        reason="ambiguous pair preserved as distinct",
                    )
                )

    grouped: defaultdict[int, list[_Occurrence]] = defaultdict(list)
    for index, occurrence in enumerate(occurrences):
        grouped[union_find.find(index)].append(occurrence)

    clusters: list[EntityCluster] = []
    occurrence_to_cluster: dict[str, str] = {}
    label_to_clusters: defaultdict[str, list[str]] = defaultdict(list)
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item.fragment_index, item.term_index))
        cluster_id = _cluster_id(ordered)
        canonical = _canonical_label(ordered)
        labels = sorted(
            {item.term.label for item in ordered}, key=lambda item: (normalize_label(item), item)
        )
        all_names = {name for item in ordered for name in item.names}
        aliases = sorted(all_names - {canonical}, key=lambda item: (normalize_label(item), item))
        cluster = EntityCluster(
            cluster_id=cluster_id,
            canonical_label=canonical,
            labels=labels,
            aliases=aliases,
            occurrence_keys=[item.key for item in ordered],
        )
        clusters.append(cluster)
        for item in ordered:
            occurrence_to_cluster[item.key] = cluster_id
            for name in item.names:
                normalized = normalize_label(name)
                if cluster_id not in label_to_clusters[normalized]:
                    label_to_clusters[normalized].append(cluster_id)

    clusters.sort(
        key=lambda cluster: (normalize_label(cluster.canonical_label), cluster.cluster_id)
    )
    for cluster_ids in label_to_clusters.values():
        cluster_ids.sort()
    decisions.sort(
        key=lambda decision: (normalize_label(decision.left), normalize_label(decision.right))
    )
    return ResolutionResult(
        clusters=clusters,
        occurrence_to_cluster=occurrence_to_cluster,
        label_to_clusters=dict(sorted(label_to_clusters.items())),
        decisions=decisions,
    )


async def resolve_entities_async(
    fragments: Sequence[OntologyFragment] | Iterable[OntologyFragment],
    *,
    gateway: object | None = None,
    model: str | None = None,
    lexical_threshold: float = 90,
    ambiguous_threshold: float = 78,
    adjudicator: Callable[..., Any] | object | None = None,
) -> ResolutionResult:
    """Resolve locally, asking a gateway only about ambiguous candidate pairs."""

    materialized = list(fragments)
    if adjudicator is not None:
        return resolve_entities(
            materialized,
            lexical_threshold=lexical_threshold,
            ambiguous_threshold=ambiguous_threshold,
            adjudicator=adjudicator,
        )
    initial = resolve_entities(
        materialized,
        lexical_threshold=lexical_threshold,
        ambiguous_threshold=ambiguous_threshold,
    )
    ambiguous = [decision for decision in initial.decisions if decision.decision == "ambiguous"]
    if gateway is None or not ambiguous:
        return initial

    method = getattr(gateway, "structured", None)
    if not callable(method):
        raise TypeError("resolution gateway must expose async structured(prompt, result_type, ...)")
    instructions = read_resource_text("prompts", "concept_resolution.md")
    prompt_sha256 = sha256_text(instructions)
    schema_sha256 = content_hash(ConceptResolutionResponse.model_json_schema())
    term_by_label: defaultdict[str, list[DraftTerm]] = defaultdict(list)
    for fragment in materialized:
        for term in fragment.terms:
            term_by_label[normalize_label(term.label)].append(term)
    answers: dict[frozenset[str], tuple[bool, str]] = {}
    for decision in ambiguous:
        left = sorted(
            term_by_label[normalize_label(decision.left)],
            key=lambda term: (-term.confidence, term.definition),
        )[0]
        right = sorted(
            term_by_label[normalize_label(decision.right)],
            key=lambda term: (-term.confidence, term.definition),
        )[0]
        payload = {
            "left": left.model_dump(mode="json"),
            "right": right.model_dump(mode="json"),
            "lexical_score": decision.score,
        }
        prompt = f"{instructions}\n\nCANDIDATE PAIR\n{json.dumps(payload, ensure_ascii=False)}"
        response = await method(
            prompt,
            ConceptResolutionResponse,
            model=model,
            cache_key=content_hash(
                {
                    "task": "concept-resolution-v1",
                    "model": model,
                    "payload": payload,
                    "prompt_sha256": prompt_sha256,
                    "schema_sha256": schema_sha256,
                }
            ),
        )
        parsed = (
            response
            if isinstance(response, ConceptResolutionResponse)
            else ConceptResolutionResponse.model_validate(response)
        )
        answers[frozenset((normalize_label(decision.left), normalize_label(decision.right)))] = (
            parsed.equivalent,
            parsed.reason,
        )

    def recorded_adjudicator(*, left: DraftTerm, right: DraftTerm, **_: Any) -> dict[str, Any]:
        equivalent, reason = answers.get(
            frozenset((normalize_label(left.label), normalize_label(right.label))),
            (False, "pair was not adjudicated"),
        )
        return {"equivalent": equivalent, "reason": reason}

    return resolve_entities(
        materialized,
        lexical_threshold=lexical_threshold,
        ambiguous_threshold=ambiguous_threshold,
        adjudicator=recorded_adjudicator,
    )


__all__ = [
    "EntityCluster",
    "ConceptResolutionResponse",
    "ResolutionDecision",
    "ResolutionResult",
    "definition_compatibility",
    "lexical_score",
    "normalize_label",
    "resolve_entities",
    "resolve_entities_async",
]
