"""Read-only, local-first tools over a published synthesis bundle."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class KnowledgeTools:
    """Filesystem-backed implementations of the synthesis tool manifest.

    Source graphs are indexed separately. If a canonical ID appears in more than
    one graph, results return every match and mark the lookup ambiguous instead of
    silently collapsing provenance.
    """

    def __init__(self, bundle_dir: str | Path) -> None:
        self.bundle_dir = Path(bundle_dir)
        if not self.bundle_dir.is_dir():
            raise FileNotFoundError(f"Synthesis bundle does not exist: {self.bundle_dir}")
        integrated_path = self.bundle_dir / "integrated-ontology.jsonld"
        if not integrated_path.is_file():
            raise FileNotFoundError(f"Missing integrated ontology: {integrated_path}")
        integrated = json.loads(integrated_path.read_text(encoding="utf-8"))
        self.source_graphs: dict[str, dict[str, Any]] = integrated.get(
            "ke:canonicalSourceGraphs", {}
        )
        if not self.source_graphs:
            raise ValueError("Integrated ontology has no canonical source named graphs")
        conflicts_path = self.bundle_dir / "conflicts.yaml"
        self.conflicts: list[dict[str, Any]] = (
            yaml.safe_load(conflicts_path.read_text(encoding="utf-8")) or []
            if conflicts_path.exists()
            else []
        )
        self._claims: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._evidence: list[dict[str, Any]] = []
        self._build_indexes()

    def _build_indexes(self) -> None:
        seen_evidence: dict[tuple[str, str, str], dict[str, Any]] = {}
        for graph_id, ontology in self.source_graphs.items():
            common = {
                "graph_id": graph_id,
                "ontology_id": ontology["ontology_id"],
                "ontology_title": ontology["title"],
            }
            for object_type, key in (
                ("term", "terms"),
                ("relation", "relations"),
                ("axiom", "axioms"),
                ("mental_model", "mental_models"),
            ):
                for item in ontology.get(key, []):
                    record = {**common, "object_type": object_type, "claim": item}
                    self._claims[item["id"]].append(record)
                    for evidence in item.get("evidence", []):
                        identity = (
                            evidence["source_id"],
                            evidence["chunk_id"],
                            evidence["quote_sha256"],
                        )
                        indexed = seen_evidence.setdefault(
                            identity,
                            {
                                **evidence,
                                "claim_ids": [],
                                "ontology_ids": [],
                                "graph_ids": [],
                            },
                        )
                        if item["id"] not in indexed["claim_ids"]:
                            indexed["claim_ids"].append(item["id"])
                        if ontology["ontology_id"] not in indexed["ontology_ids"]:
                            indexed["ontology_ids"].append(ontology["ontology_id"])
                        if graph_id not in indexed["graph_ids"]:
                            indexed["graph_ids"].append(graph_id)
        self._evidence = list(seen_evidence.values())

    def lookup_concept(self, id: str) -> dict[str, Any] | None:
        matches = [
            _public_record(record)
            for record in self._claims.get(id, [])
            if record["object_type"] == "term"
        ]
        return _one_or_ambiguous(id, matches)

    def trace_claim(self, id: str) -> dict[str, Any] | None:
        matches: list[dict[str, Any]] = []
        for record in self._claims.get(id, []):
            public = _public_record(record)
            public["evidence"] = [
                evidence
                for evidence in record["claim"].get("evidence", [])
                if evidence.get("verified") is True
            ]
            matches.append(public)
        return _one_or_ambiguous(id, matches)

    def retrieve_evidence(
        self,
        query: str,
        source_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Evidence query must be non-empty")
        requested_sources = set(source_ids or [])
        query_tokens = set(_tokens(query))
        results: list[tuple[float, dict[str, Any]]] = []
        for evidence in self._evidence:
            if evidence.get("verified") is not True:
                continue
            if requested_sources and evidence["source_id"] not in requested_sources:
                continue
            searchable = " ".join([evidence.get("quote", ""), *evidence.get("heading_path", [])])
            evidence_tokens = set(_tokens(searchable))
            shared = query_tokens & evidence_tokens
            if not shared:
                continue
            score = len(shared) / max(1, len(query_tokens))
            phrase_bonus = 0.25 if query.casefold() in searchable.casefold() else 0.0
            result = {**evidence, "score": round(min(1.0, score + phrase_bonus), 4)}
            results.append((result["score"], result))
        results.sort(
            key=lambda item: (
                -item[0],
                item[1]["source_id"],
                item[1]["chunk_id"],
                item[1]["quote_sha256"],
            )
        )
        return [result for _, result in results]

    def list_conflicts(self, concept_id: str) -> list[dict[str, Any]]:
        related_claim_ids = {concept_id}
        for records in self._claims.values():
            for record in records:
                claim = record["claim"]
                if record["object_type"] == "relation" and concept_id in {
                    claim.get("subject_id"),
                    claim.get("object_id"),
                }:
                    related_claim_ids.add(claim["id"])
                if record["object_type"] == "term" and concept_id in {
                    *claim.get("broader_ids", []),
                    *claim.get("narrower_ids", []),
                    *claim.get("related_ids", []),
                }:
                    related_claim_ids.add(claim["id"])
        return [
            conflict
            for conflict in self.conflicts
            if related_claim_ids & set(conflict.get("claim_ids", []))
        ]

    def get_mental_model(self, id: str) -> dict[str, Any] | None:
        matches = [
            _public_record(record)
            for record in self._claims.get(id, [])
            if record["object_type"] == "mental_model"
        ]
        return _one_or_ambiguous(id, matches)


def load_tools(bundle_dir: str | Path) -> KnowledgeTools:
    return KnowledgeTools(bundle_dir)


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "graph_id": record["graph_id"],
        "ontology_id": record["ontology_id"],
        "ontology_title": record["ontology_title"],
        "object_type": record["object_type"],
        **record["claim"],
    }


def _one_or_ambiguous(id: str, matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return {"id": id, "ambiguous_across_source_graphs": True, "matches": matches}


def _tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(value.casefold())
