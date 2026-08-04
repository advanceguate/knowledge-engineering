"""Evidence-preserving reduction and post-resolution canonicalization."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ke.contracts import (
    Axiom,
    ClaimConflict,
    DomainOntology,
    DraftAxiom,
    DraftMentalModel,
    DraftRelation,
    DraftTerm,
    Evidence,
    ExtractionFragment,
    MentalModel,
    Relation,
    Term,
)
from ke.ids import ontology_id as make_ontology_id
from ke.ids import stable_id
from ke.ontology.resolve import EntityCluster, ResolutionResult, normalize_label, resolve_entities


class ReductionBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    terms: int = Field(default=250, ge=1)
    relations: int = Field(default=400, ge=1)
    axioms: int = Field(default=200, ge=1)
    mental_models: int = Field(default=60, ge=1)


def _unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = " ".join(value.split())
        key = normalize_label(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def _evidence_key(evidence: Evidence) -> tuple[str, str, str, int, int]:
    return (
        evidence.source_id,
        evidence.chunk_id,
        evidence.quote_sha256,
        evidence.page_start or 0,
        evidence.page_end or 0,
    )


def _merge_evidence(values: Iterable[Evidence], *, require_verified: bool = True) -> list[Evidence]:
    unique: dict[tuple[str, str, str, int, int], Evidence] = {}
    for evidence in values:
        if require_verified and not evidence.verified:
            continue
        unique.setdefault(_evidence_key(evidence), evidence)
    return [unique[key] for key in sorted(unique)]


def _quarantine_unverified_candidates(
    fragments: Sequence[ExtractionFragment],
) -> list[ExtractionFragment]:
    """Remove unsupported objects before resolution or canonical selection."""

    sanitized: list[ExtractionFragment] = []
    for fragment in fragments:
        payload = fragment.model_dump(mode="json")
        for collection in ("terms", "relations", "axioms", "mental_models"):
            retained: list[dict[str, Any]] = []
            for item in payload[collection]:
                evidence = [citation for citation in item["evidence"] if citation["verified"]]
                if not evidence:
                    continue
                item["evidence"] = evidence
                retained.append(item)
            payload[collection] = retained
        # Section summaries have no evidence field and therefore cannot enter
        # the canonical abstract. They remain available in raw fragments.
        payload["section_summary"] = ""
        sanitized.append(ExtractionFragment.model_validate(payload))
    return sanitized


def _salience(item: Any) -> tuple[float, int, str]:
    evidence = getattr(item, "evidence", ())
    confidence = float(getattr(item, "confidence", 0))
    label = getattr(item, "preferred_label", None) or getattr(item, "name", None)
    label = label or getattr(item, "statement", None) or getattr(item, "predicate", "")
    return (-confidence, -len(evidence), normalize_label(str(label)))


def _best_definition(terms: Sequence[Any]) -> str:
    candidates = [term for term in terms if term.definition.strip()]
    if not candidates:
        return ""
    winner = sorted(
        candidates,
        key=lambda term: (
            -term.confidence,
            -len(term.evidence),
            -len(term.definition),
            term.definition,
        ),
    )[0]
    return " ".join(winner.definition.split())


def _best_kind(terms: Sequence[Any]) -> str:
    weighted: defaultdict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for term in terms:
        weighted[term.kind] += term.confidence
        counts[term.kind] += 1
    return sorted(weighted, key=lambda kind: (-weighted[kind], -counts[kind], kind))[0]


def _cluster_lookup(result: ResolutionResult) -> dict[str, EntityCluster]:
    return {cluster.cluster_id: cluster for cluster in result.clusters}


def _resolve_reference(
    label: str,
    fragment_index: int,
    fragments: Sequence[ExtractionFragment],
    result: ResolutionResult,
) -> str | None:
    wanted = normalize_label(label)
    local: set[str] = set()
    for term_index, term in enumerate(fragments[fragment_index].terms):
        names = [term.label, *term.aliases]
        if wanted in {normalize_label(name) for name in names}:
            cluster_id = result.occurrence_to_cluster.get(f"{fragment_index}:{term_index}")
            if cluster_id:
                local.add(cluster_id)
    if len(local) == 1:
        return next(iter(local))
    global_matches = result.label_to_clusters.get(wanted, [])
    if len(global_matches) == 1:
        return global_matches[0]
    return None


def _build_terms(
    fragments: Sequence[ExtractionFragment],
    resolution: ResolutionResult,
    *,
    minimum_confidence: float,
    budget: int,
) -> tuple[list[Term], dict[str, str]]:
    grouped: defaultdict[str, list[tuple[int, int, Any]]] = defaultdict(list)
    for fragment_index, fragment in enumerate(fragments):
        for term_index, term in enumerate(fragment.terms):
            cluster_id = resolution.occurrence_to_cluster[f"{fragment_index}:{term_index}"]
            grouped[cluster_id].append((fragment_index, term_index, term))

    clusters = _cluster_lookup(resolution)
    candidates: list[tuple[Term, str, list[tuple[int, int, Any]]]] = []
    for cluster_id, occurrences in grouped.items():
        draft_terms = [item[2] for item in occurrences]
        confidence = max(term.confidence for term in draft_terms)
        evidence = _merge_evidence(citation for term in draft_terms for citation in term.evidence)
        if confidence < minimum_confidence or not evidence:
            continue
        cluster = clusters[cluster_id]
        preferred_label = cluster.canonical_label
        aliases = _unique_strings(
            [
                *cluster.aliases,
                *(alias for term in draft_terms for alias in term.aliases),
                *(
                    term.label
                    for term in draft_terms
                    if term.label != preferred_label and term.label != cluster_id
                ),
            ]
        )
        aliases = [
            alias for alias in aliases if normalize_label(alias) != normalize_label(preferred_label)
        ]
        definition = _best_definition(draft_terms)
        kind = _best_kind(draft_terms)
        identifier = stable_id(
            "term",
            preferred_label,
            {
                "label": normalize_label(preferred_label),
                "kind": kind,
                "definition": normalize_label(definition),
                "aliases": sorted(normalize_label(alias) for alias in aliases),
            },
        )
        candidates.append(
            (
                Term(
                    id=identifier,
                    preferred_label=preferred_label,
                    aliases=aliases,
                    kind=kind,
                    definition=definition,
                    evidence=evidence,
                    confidence=confidence,
                ),
                cluster_id,
                occurrences,
            )
        )

    candidates.sort(key=lambda item: (_salience(item[0]), item[0].id))
    candidates = candidates[:budget]
    cluster_to_term = {cluster_id: term.id for term, cluster_id, _ in candidates}

    link_sets: dict[str, dict[str, set[str]]] = {
        term.id: {"broader": set(), "narrower": set(), "related": set()}
        for term, _, _ in candidates
    }
    for term, _cluster_id, occurrences in candidates:
        for fragment_index, _, draft in occurrences:
            for field, relation_name, inverse in (
                (draft.broader_labels, "broader", "narrower"),
                (draft.narrower_labels, "narrower", "broader"),
                (draft.related_labels, "related", "related"),
            ):
                for label in field:
                    target_cluster = _resolve_reference(
                        label, fragment_index, fragments, resolution
                    )
                    target_id = cluster_to_term.get(target_cluster or "")
                    if target_id and target_id != term.id:
                        link_sets[term.id][relation_name].add(target_id)
                        link_sets[target_id][inverse].add(term.id)

    terms = [
        term.model_copy(
            update={
                "broader_ids": sorted(link_sets[term.id]["broader"]),
                "narrower_ids": sorted(link_sets[term.id]["narrower"]),
                "related_ids": sorted(link_sets[term.id]["related"]),
            }
        )
        for term, _, _ in candidates
    ]
    terms.sort(key=lambda term: (normalize_label(term.preferred_label), term.id))
    return terms, cluster_to_term


def _build_relations(
    fragments: Sequence[ExtractionFragment],
    resolution: ResolutionResult,
    cluster_to_term: dict[str, str],
    *,
    minimum_confidence: float,
    budget: int,
) -> list[Relation]:
    grouped: defaultdict[tuple[str, str, str, tuple[tuple[str, str], ...]], list[Any]] = (
        defaultdict(list)
    )
    for fragment_index, fragment in enumerate(fragments):
        for relation in fragment.relations:
            subject_cluster = _resolve_reference(
                relation.subject_label, fragment_index, fragments, resolution
            )
            object_cluster = _resolve_reference(
                relation.object_label, fragment_index, fragments, resolution
            )
            subject_id = cluster_to_term.get(subject_cluster or "")
            object_id = cluster_to_term.get(object_cluster or "")
            if not subject_id or not object_id:
                continue
            key = (
                subject_id,
                normalize_label(relation.predicate),
                object_id,
                tuple(sorted((str(k), str(v)) for k, v in relation.qualifiers.items())),
            )
            grouped[key].append(relation)

    output: list[Relation] = []
    for (subject_id, normalized_predicate, object_id, qualifier_items), drafts in grouped.items():
        confidence = max(item.confidence for item in drafts)
        evidence = _merge_evidence(citation for item in drafts for citation in item.evidence)
        if confidence < minimum_confidence or not evidence:
            continue
        predicate = sorted(
            {item.predicate.strip() for item in drafts},
            key=lambda value: (len(value), value.casefold(), value),
        )[0]
        payload = {
            "subject_id": subject_id,
            "predicate": normalized_predicate,
            "object_id": object_id,
            "qualifiers": qualifier_items,
        }
        output.append(
            Relation(
                id=stable_id("relation", predicate, payload),
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
                qualifiers=dict(qualifier_items),
                evidence=evidence,
                confidence=confidence,
            )
        )
    output.sort(key=lambda item: (_salience(item), item.id))
    return sorted(output[:budget], key=lambda item: item.id)


def _build_axioms(
    fragments: Sequence[ExtractionFragment],
    *,
    minimum_confidence: float,
    budget: int,
) -> list[Axiom]:
    grouped: defaultdict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for fragment in fragments:
        for axiom in fragment.axioms:
            key = (
                normalize_label(axiom.statement),
                axiom.modality,
                axiom.epistemic_status,
                normalize_label(axiom.attribution),
                tuple(normalize_label(value) for value in axiom.conditions),
                tuple(normalize_label(value) for value in axiom.exceptions),
            )
            grouped[key].append(axiom)

    output: list[Axiom] = []
    for key, drafts in grouped.items():
        confidence = max(item.confidence for item in drafts)
        evidence = _merge_evidence(citation for item in drafts for citation in item.evidence)
        if confidence < minimum_confidence or not evidence:
            continue
        winner = sorted(
            drafts,
            key=lambda item: (-item.confidence, -len(item.evidence), item.statement),
        )[0]
        payload = {
            "statement": key[0],
            "modality": winner.modality,
            "epistemic_status": winner.epistemic_status,
            "attribution": key[3],
            "conditions": key[4],
            "exceptions": key[5],
        }
        output.append(
            Axiom(
                id=stable_id("axiom", winner.statement, payload),
                statement=winner.statement,
                modality=winner.modality,
                epistemic_status=winner.epistemic_status,
                attribution=winner.attribution,
                conditions=_unique_strings(value for item in drafts for value in item.conditions),
                exceptions=_unique_strings(value for item in drafts for value in item.exceptions),
                evidence=evidence,
                confidence=confidence,
            )
        )
    output.sort(key=lambda item: (_salience(item), item.id))
    return sorted(output[:budget], key=lambda item: item.id)


def _build_mental_models(
    fragments: Sequence[ExtractionFragment],
    *,
    minimum_confidence: float,
    budget: int,
) -> list[MentalModel]:
    grouped: defaultdict[str, list[Any]] = defaultdict(list)
    for fragment in fragments:
        for model in fragment.mental_models:
            grouped[normalize_label(model.name)].append(model)

    output: list[MentalModel] = []
    list_fields = (
        "applicable_when",
        "assumptions",
        "variables",
        "mechanism",
        "procedure",
        "predictions",
        "failure_modes",
    )
    for normalized_name, drafts in grouped.items():
        confidence = max(item.confidence for item in drafts)
        evidence = _merge_evidence(citation for item in drafts for citation in item.evidence)
        if confidence < minimum_confidence or not evidence:
            continue
        winner = sorted(
            drafts,
            key=lambda item: (-item.confidence, -len(item.evidence), item.name),
        )[0]
        values = {
            field: _unique_strings(value for item in drafts for value in getattr(item, field))
            for field in list_fields
        }
        purposes = [item for item in drafts if item.purpose.strip()]
        purpose = (
            sorted(purposes, key=lambda item: (-item.confidence, -len(item.purpose), item.purpose))[
                0
            ].purpose
            if purposes
            else ""
        )
        payload = {"name": normalized_name, "purpose": normalize_label(purpose), **values}
        output.append(
            MentalModel(
                id=stable_id("mental-model", winner.name, payload),
                name=winner.name,
                purpose=purpose,
                evidence=evidence,
                confidence=confidence,
                **values,
            )
        )
    output.sort(key=lambda item: (_salience(item), item.id))
    return sorted(output[:budget], key=lambda item: item.id)


def _contradiction_signature(statement: str) -> tuple[str, bool]:
    normalized = normalize_label(statement)
    tokens = normalized.split()
    negated = any(token in {"no", "not", "never", "cannot"} for token in tokens)
    base = []
    for token in tokens:
        if token in {"no", "not", "never", "cannot", "do", "does", "did"}:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "is", "us")):
            token = token[:-1]
        base.append(token)
    return " ".join(base), negated


def _detect_conflicts(axioms: Sequence[Axiom]) -> list[ClaimConflict]:
    grouped: defaultdict[str, dict[bool, list[Axiom]]] = defaultdict(lambda: defaultdict(list))
    for axiom in axioms:
        signature, negated = _contradiction_signature(axiom.statement)
        grouped[signature][negated].append(axiom)
    conflicts: list[ClaimConflict] = []
    for signature, polarities in grouped.items():
        if not polarities[False] or not polarities[True]:
            continue
        claim_ids = sorted({item.id for items in polarities.values() for item in items})
        kind = (
            "causal"
            if any(item.modality == "causal" for items in polarities.values() for item in items)
            else "empirical"
        )
        conflicts.append(
            ClaimConflict(
                id=stable_id("conflict", signature, claim_ids),
                claim_ids=claim_ids,
                kind=kind,
                description="Source-attributed claims express opposite polarities.",
                resolution="unresolved",
            )
        )
    return sorted(conflicts, key=lambda item: item.id)


def _canonical_abstract(
    terms: Sequence[Term],
    axioms: Sequence[Axiom],
    mental_models: Sequence[MentalModel],
    *,
    limit: int = 8,
) -> tuple[str, list[str]]:
    candidates: list[tuple[Any, str]] = []
    candidates.extend((axiom, axiom.statement.strip()) for axiom in axioms)
    candidates.extend(
        (
            term,
            (
                f"{term.preferred_label}: {term.definition.strip()}"
                if term.definition.strip()
                else term.preferred_label
            ),
        )
        for term in terms
    )
    candidates.extend(
        (
            model,
            f"{model.name}: {model.purpose.strip()}" if model.purpose.strip() else model.name,
        )
        for model in mental_models
    )
    selected = [
        (item, text)
        for item, text in sorted(candidates, key=lambda pair: (_salience(pair[0]), pair[0].id))
        if text
    ][:limit]
    return "\n\n".join(text for _, text in selected), [item.id for item, _ in selected]


def _term_claim_key(term: DraftTerm) -> tuple[str, str, str]:
    return normalize_label(term.label), term.kind, normalize_label(term.definition)


def _relation_claim_key(relation: DraftRelation) -> tuple[Any, ...]:
    return (
        normalize_label(relation.subject_label),
        normalize_label(relation.predicate),
        normalize_label(relation.object_label),
        tuple(sorted((str(key), str(value)) for key, value in relation.qualifiers.items())),
    )


def _axiom_claim_key(axiom: DraftAxiom) -> tuple[Any, ...]:
    return (
        normalize_label(axiom.statement),
        axiom.modality,
        axiom.epistemic_status,
        normalize_label(axiom.attribution),
        tuple(normalize_label(value) for value in axiom.conditions),
        tuple(normalize_label(value) for value in axiom.exceptions),
    )


def _mental_model_claim_key(model: DraftMentalModel) -> tuple[Any, ...]:
    return (
        normalize_label(model.name),
        normalize_label(model.purpose),
        *(
            tuple(normalize_label(value) for value in getattr(model, field))
            for field in (
                "applicable_when",
                "assumptions",
                "variables",
                "mechanism",
                "procedure",
                "predictions",
                "failure_modes",
            )
        ),
    )


def _merge_fragment_batch(batch: Sequence[ExtractionFragment]) -> ExtractionFragment:
    """Associatively merge one bounded batch without inventing claim content."""

    if not batch:
        raise ValueError("a reduction batch must contain at least one fragment")

    term_groups: defaultdict[tuple[str, str, str], list[DraftTerm]] = defaultdict(list)
    relation_groups: defaultdict[tuple[Any, ...], list[DraftRelation]] = defaultdict(list)
    axiom_groups: defaultdict[tuple[Any, ...], list[DraftAxiom]] = defaultdict(list)
    model_groups: defaultdict[tuple[Any, ...], list[DraftMentalModel]] = defaultdict(list)
    for fragment in batch:
        for term in fragment.terms:
            term_groups[_term_claim_key(term)].append(term)
        for relation in fragment.relations:
            relation_groups[_relation_claim_key(relation)].append(relation)
        for axiom in fragment.axioms:
            axiom_groups[_axiom_claim_key(axiom)].append(axiom)
        for model in fragment.mental_models:
            model_groups[_mental_model_claim_key(model)].append(model)

    terms: list[DraftTerm] = []
    for _key, candidates in sorted(term_groups.items(), key=lambda pair: pair[0]):
        winner = sorted(
            candidates,
            key=lambda item: (-item.confidence, -len(item.evidence), item.label),
        )[0]
        terms.append(
            winner.model_copy(
                update={
                    "aliases": _unique_strings(
                        alias for item in candidates for alias in item.aliases
                    ),
                    "broader_labels": _unique_strings(
                        label for item in candidates for label in item.broader_labels
                    ),
                    "narrower_labels": _unique_strings(
                        label for item in candidates for label in item.narrower_labels
                    ),
                    "related_labels": _unique_strings(
                        label for item in candidates for label in item.related_labels
                    ),
                    "evidence": _merge_evidence(
                        citation for item in candidates for citation in item.evidence
                    ),
                    "confidence": max(item.confidence for item in candidates),
                }
            )
        )

    relations = [
        sorted(items, key=lambda item: (-item.confidence, item.predicate))[0].model_copy(
            update={
                "evidence": _merge_evidence(
                    citation for item in items for citation in item.evidence
                ),
                "confidence": max(item.confidence for item in items),
            }
        )
        for _, items in sorted(relation_groups.items(), key=lambda pair: pair[0])
    ]
    axioms = [
        sorted(items, key=lambda item: (-item.confidence, item.statement))[0].model_copy(
            update={
                "evidence": _merge_evidence(
                    citation for item in items for citation in item.evidence
                ),
                "confidence": max(item.confidence for item in items),
            }
        )
        for _, items in sorted(axiom_groups.items(), key=lambda pair: pair[0])
    ]
    mental_models = [
        sorted(items, key=lambda item: (-item.confidence, item.name))[0].model_copy(
            update={
                "evidence": _merge_evidence(
                    citation for item in items for citation in item.evidence
                ),
                "confidence": max(item.confidence for item in items),
            }
        )
        for _, items in sorted(model_groups.items(), key=lambda pair: pair[0])
    ]

    source_values = sorted({fragment.source_id for fragment in batch})
    source_id = source_values[0] if len(source_values) == 1 else "reduction-batch"
    child_ids = sorted(fragment.chunk_id for fragment in batch)
    fragment = ExtractionFragment(
        source_id=source_id,
        chunk_id=stable_id("reduction-batch", "fragments", child_ids),
        section_summary="",
        terms=terms,
        relations=relations,
        axioms=axioms,
        mental_models=mental_models,
        open_questions=_unique_strings(
            question for item in batch for question in item.open_questions
        ),
    )

    # Fail closed if a future merger edit emits a semantic claim core that was
    # absent from every input in this batch.
    checks = (
        (
            {_term_claim_key(item) for value in batch for item in value.terms},
            fragment.terms,
            _term_claim_key,
        ),
        (
            {_relation_claim_key(item) for value in batch for item in value.relations},
            fragment.relations,
            _relation_claim_key,
        ),
        (
            {_axiom_claim_key(item) for value in batch for item in value.axioms},
            fragment.axioms,
            _axiom_claim_key,
        ),
        (
            {_mental_model_claim_key(item) for value in batch for item in value.mental_models},
            fragment.mental_models,
            _mental_model_claim_key,
        ),
    )
    for allowed, output, key_function in checks:
        if any(key_function(item) not in allowed for item in output):
            raise ValueError("hierarchical reducer attempted to introduce a new domain claim")
    return fragment


def _merge_fragment_hierarchy(
    fragments: Sequence[ExtractionFragment],
    *,
    batch_size: int,
) -> ExtractionFragment:
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    level = list(fragments)
    if not level:
        raise ValueError("at least one extraction fragment is required")
    while len(level) > 1:
        level = [
            _merge_fragment_batch(level[start : start + batch_size])
            for start in range(0, len(level), batch_size)
        ]
    return level[0]


def _resolution_matches_term_layout(
    fragments: Sequence[ExtractionFragment],
    resolution: ResolutionResult,
) -> bool:
    cluster_ids = {cluster.cluster_id for cluster in resolution.clusters}
    expected_keys = {
        f"{fragment_index}:{term_index}"
        for fragment_index, fragment in enumerate(fragments)
        for term_index, _term in enumerate(fragment.terms)
    }
    return expected_keys <= set(resolution.occurrence_to_cluster) and all(
        resolution.occurrence_to_cluster[key] in cluster_ids for key in expected_keys
    )


def _bind_fragments_to_resolution(
    fragments: Sequence[ExtractionFragment],
    resolution: ResolutionResult,
) -> list[ExtractionFragment]:
    """Bind draft labels to resolved clusters before lossy hierarchical merging.

    Opaque cluster IDs are temporary internal labels. They preserve supplied
    adjudications and fragment-local homonym references while exact candidates
    are combined across bounded reduction batches.
    """

    bound: list[ExtractionFragment] = []
    for fragment_index, fragment in enumerate(fragments):

        def bind_reference(label: str, *, _fragment_index: int = fragment_index) -> str:
            return _resolve_reference(label, _fragment_index, fragments, resolution) or label

        terms: list[DraftTerm] = []
        for term_index, term in enumerate(fragment.terms):
            cluster_id = resolution.occurrence_to_cluster[f"{fragment_index}:{term_index}"]
            terms.append(
                term.model_copy(
                    update={
                        "label": cluster_id,
                        "aliases": _unique_strings([term.label, *term.aliases]),
                        "broader_labels": _unique_strings(
                            bind_reference(label) for label in term.broader_labels
                        ),
                        "narrower_labels": _unique_strings(
                            bind_reference(label) for label in term.narrower_labels
                        ),
                        "related_labels": _unique_strings(
                            bind_reference(label) for label in term.related_labels
                        ),
                    }
                )
            )
        relations = [
            relation.model_copy(
                update={
                    "subject_label": bind_reference(relation.subject_label),
                    "object_label": bind_reference(relation.object_label),
                }
            )
            for relation in fragment.relations
        ]
        bound.append(fragment.model_copy(update={"terms": terms, "relations": relations}))
    return bound


def _project_resolution_to_merged_fragment(
    merged: ExtractionFragment,
    resolution: ResolutionResult,
) -> ResolutionResult:
    """Project original occurrence clusters onto one hierarchy output fragment."""

    original_clusters = _cluster_lookup(resolution)
    occurrence_to_cluster: dict[str, str] = {}
    occurrence_keys: defaultdict[str, list[str]] = defaultdict(list)
    for term_index, term in enumerate(merged.terms):
        cluster_id = term.label
        if cluster_id not in original_clusters:
            raise ValueError("hierarchical reducer lost a resolved term-cluster binding")
        occurrence_key = f"0:{term_index}"
        occurrence_to_cluster[occurrence_key] = cluster_id
        occurrence_keys[cluster_id].append(occurrence_key)

    active_cluster_ids = set(occurrence_keys)
    clusters = [
        cluster.model_copy(update={"occurrence_keys": occurrence_keys[cluster.cluster_id]})
        for cluster in resolution.clusters
        if cluster.cluster_id in active_cluster_ids
    ]
    label_to_clusters = {
        label: sorted(cluster_id for cluster_id in cluster_ids if cluster_id in active_cluster_ids)
        for label, cluster_ids in resolution.label_to_clusters.items()
        if any(cluster_id in active_cluster_ids for cluster_id in cluster_ids)
    }
    for cluster_id in active_cluster_ids:
        label_to_clusters[normalize_label(cluster_id)] = [cluster_id]
    return ResolutionResult(
        clusters=clusters,
        occurrence_to_cluster=occurrence_to_cluster,
        label_to_clusters=dict(sorted(label_to_clusters.items())),
        decisions=resolution.decisions,
    )


def reduce_fragments(
    fragments: Sequence[ExtractionFragment] | Iterable[ExtractionFragment],
    *,
    title: str,
    source_ids: Sequence[str] | None = None,
    resolution: ResolutionResult | None = None,
    ontology_id: str | None = None,
    minimum_confidence: float = 0.55,
    budget: ReductionBudget | Any | None = None,
    scope_in: Sequence[str] = (),
    scope_out: Sequence[str] = (),
) -> DomainOntology:
    """Reduce grounded fragments without adding claims absent from them."""

    raw_materialized = [
        item if isinstance(item, ExtractionFragment) else ExtractionFragment.model_validate(item)
        for item in fragments
    ]
    if not raw_materialized:
        raise ValueError("at least one extraction fragment is required")
    materialized = _quarantine_unverified_candidates(raw_materialized)
    resolved_source_ids = sorted(
        set(source_ids or [fragment.source_id for fragment in raw_materialized])
    )
    selected_budget = (
        budget
        if isinstance(budget, ReductionBudget)
        else ReductionBudget.model_validate(
            budget.model_dump() if hasattr(budget, "model_dump") else (budget or {})
        )
    )
    term_layout_unchanged = all(
        len(raw.terms) == len(clean.terms)
        for raw, clean in zip(raw_materialized, materialized, strict=True)
    )
    resolution = (
        resolution
        if resolution is not None
        and term_layout_unchanged
        and _resolution_matches_term_layout(materialized, resolution)
        else resolve_entities(materialized)
    )
    terms, cluster_to_term = _build_terms(
        materialized,
        resolution,
        minimum_confidence=minimum_confidence,
        budget=selected_budget.terms,
    )
    relations = _build_relations(
        materialized,
        resolution,
        cluster_to_term,
        minimum_confidence=minimum_confidence,
        budget=selected_budget.relations,
    )
    axioms = _build_axioms(
        materialized,
        minimum_confidence=minimum_confidence,
        budget=selected_budget.axioms,
    )
    mental_models = _build_mental_models(
        materialized,
        minimum_confidence=minimum_confidence,
        budget=selected_budget.mental_models,
    )
    abstract, support_ids = _canonical_abstract(terms, axioms, mental_models)
    verified_quotes = [
        evidence.quote
        for fragment in materialized
        for collection in (
            fragment.terms,
            fragment.relations,
            fragment.axioms,
            fragment.mental_models,
        )
        for item in collection
        for evidence in item.evidence
    ]
    return DomainOntology(
        ontology_id=ontology_id or make_ontology_id(title, resolved_source_ids),
        title=title,
        source_ids=resolved_source_ids,
        abstract=abstract,
        abstract_support_ids=support_ids,
        scope_in=_unique_strings(scope_in),
        scope_out=_unique_strings(scope_out),
        terms=terms,
        relations=relations,
        axioms=axioms,
        mental_models=mental_models,
        conflicts=_detect_conflicts(axioms),
        open_questions=_unique_strings(
            question
            for fragment in materialized
            for question in fragment.open_questions
            if any(question in quote for quote in verified_quotes)
        ),
    )


def hierarchical_reduce(
    fragments: Sequence[ExtractionFragment] | Iterable[ExtractionFragment],
    **kwargs: Any,
) -> DomainOntology:
    """Deterministic hierarchical reduction entry point.

    Exact candidate duplicates are merged through bounded, associative batches
    before the final global resolution/reduction pass. No reducer-generated
    claim is admitted.
    """

    kwargs.pop("gateway", None)
    kwargs.pop("model", None)
    batch_size = int(kwargs.pop("batch_size", 8))
    supplied_resolution = kwargs.pop("resolution", None)
    materialized = [
        item if isinstance(item, ExtractionFragment) else ExtractionFragment.model_validate(item)
        for item in fragments
    ]
    source_ids = kwargs.get("source_ids") or sorted(
        {fragment.source_id for fragment in materialized}
    )
    sanitized = _quarantine_unverified_candidates(materialized)
    term_layout_unchanged = all(
        len(raw.terms) == len(clean.terms)
        for raw, clean in zip(materialized, sanitized, strict=True)
    )
    resolution = (
        supplied_resolution
        if supplied_resolution is not None
        and term_layout_unchanged
        and _resolution_matches_term_layout(sanitized, supplied_resolution)
        else resolve_entities(sanitized)
    )
    bound = _bind_fragments_to_resolution(sanitized, resolution)
    merged = _merge_fragment_hierarchy(bound, batch_size=batch_size)
    projected_resolution = _project_resolution_to_merged_fragment(merged, resolution)
    kwargs["source_ids"] = source_ids
    kwargs["resolution"] = projected_resolution
    return reduce_fragments([merged], **kwargs)


def canonicalize_ids(ontology: DomainOntology) -> DomainOntology:
    """Replace provisional IDs with deterministic post-resolution identifiers."""

    def evidence_identity(item: Any) -> list[tuple[str, str, str]]:
        return sorted(
            (evidence.source_id, evidence.chunk_id, evidence.quote_sha256)
            for evidence in item.evidence
        )

    term_id_map = {
        term.id: stable_id(
            "term",
            term.preferred_label,
            {
                "label": normalize_label(term.preferred_label),
                "kind": term.kind,
                "definition": normalize_label(term.definition),
                "aliases": sorted(normalize_label(alias) for alias in term.aliases),
                "evidence": evidence_identity(term),
            },
        )
        for term in ontology.terms
    }
    terms = [
        term.model_copy(
            update={
                "id": term_id_map[term.id],
                "broader_ids": sorted(term_id_map[item] for item in term.broader_ids),
                "narrower_ids": sorted(term_id_map[item] for item in term.narrower_ids),
                "related_ids": sorted(term_id_map[item] for item in term.related_ids),
            }
        )
        for term in ontology.terms
    ]
    object_id_map: dict[str, str] = dict(term_id_map)
    relations: list[Relation] = []
    for relation in ontology.relations:
        subject_id = term_id_map[relation.subject_id]
        object_id = term_id_map[relation.object_id]
        identifier = stable_id(
            "relation",
            relation.predicate,
            {
                "subject_id": subject_id,
                "predicate": normalize_label(relation.predicate),
                "object_id": object_id,
                "qualifiers": relation.qualifiers,
            },
        )
        object_id_map[relation.id] = identifier
        relations.append(
            relation.model_copy(
                update={"id": identifier, "subject_id": subject_id, "object_id": object_id}
            )
        )
    axioms: list[Axiom] = []
    for axiom in ontology.axioms:
        identifier = stable_id(
            "axiom",
            axiom.statement,
            {
                "statement": normalize_label(axiom.statement),
                "modality": axiom.modality,
                "epistemic_status": axiom.epistemic_status,
                "attribution": normalize_label(axiom.attribution),
                "conditions": [normalize_label(value) for value in axiom.conditions],
                "exceptions": [normalize_label(value) for value in axiom.exceptions],
                "formalism": axiom.formalism,
                "formal_expression": axiom.formal_expression,
                "evidence": evidence_identity(axiom),
            },
        )
        object_id_map[axiom.id] = identifier
        axioms.append(axiom.model_copy(update={"id": identifier}))
    mental_models: list[MentalModel] = []
    for model in ontology.mental_models:
        identifier = stable_id(
            "mental-model",
            model.name,
            {
                "name": normalize_label(model.name),
                "purpose": normalize_label(model.purpose),
                "applicable_when": model.applicable_when,
                "assumptions": model.assumptions,
                "variables": model.variables,
                "mechanism": model.mechanism,
                "procedure": model.procedure,
                "predictions": model.predictions,
                "failure_modes": model.failure_modes,
                "evidence": evidence_identity(model),
            },
        )
        object_id_map[model.id] = identifier
        mental_models.append(model.model_copy(update={"id": identifier}))
    conflicts = [
        conflict.model_copy(
            update={
                "id": stable_id(
                    "conflict",
                    conflict.description,
                    sorted(object_id_map[item] for item in conflict.claim_ids),
                ),
                "claim_ids": sorted(object_id_map[item] for item in conflict.claim_ids),
            }
        )
        for conflict in ontology.conflicts
    ]
    updated = ontology.model_copy(
        update={
            "terms": sorted(terms, key=lambda item: item.id),
            "relations": sorted(relations, key=lambda item: item.id),
            "axioms": sorted(axioms, key=lambda item: item.id),
            "mental_models": sorted(mental_models, key=lambda item: item.id),
            "conflicts": sorted(conflicts, key=lambda item: item.id),
            "abstract_support_ids": [object_id_map[item] for item in ontology.abstract_support_ids],
        }
    )
    return DomainOntology.model_validate(updated.model_dump(mode="json"))


__all__ = [
    "ReductionBudget",
    "canonicalize_ids",
    "hierarchical_reduce",
    "reduce_fragments",
]
