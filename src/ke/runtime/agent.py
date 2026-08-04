"""Minimal optional runtime over a published intentional frame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from ke.contracts import AgentFrame
from ke.llm import ModelGateway

from .tools import KnowledgeTools


class RuntimeResponse(BaseModel):
    """Public answer shape that keeps source claims and synthesis separate."""

    source_claims: list[str] = Field(default_factory=list)
    synthesis: str
    conflicts_and_uncertainty: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class IntentionalRuntime:
    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        gateway: ModelGateway | None = None,
        model: str | None = None,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.tools = KnowledgeTools(self.bundle_dir)
        self.gateway = gateway
        self.model = model
        frame_path = self.bundle_dir / "agent-frame.yaml"
        prompt_path = self.bundle_dir / "system-prompt.md"
        if not frame_path.exists() or not prompt_path.exists():
            raise FileNotFoundError("Bundle must contain agent-frame.yaml and system-prompt.md")
        self.frame = AgentFrame.model_validate(
            yaml.safe_load(frame_path.read_text(encoding="utf-8"))
        )
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

    async def ask(self, question: str) -> RuntimeResponse:
        """Answer with retrieved bundle evidence, optionally using a model gateway."""

        if not question or not question.strip():
            raise ValueError("Question must be non-empty")
        evidence = self.tools.retrieve_evidence(question)
        relevant_conflicts: list[dict[str, object]] = []
        seen_conflicts: set[str] = set()
        for item in evidence:
            for claim_id in item["claim_ids"]:
                for conflict in self.tools.list_conflicts(claim_id):
                    if conflict["id"] not in seen_conflicts:
                        seen_conflicts.add(conflict["id"])
                        relevant_conflicts.append(conflict)
        if self.gateway is None:
            return _offline_response(evidence, relevant_conflicts)
        payload = {
            "question": question,
            "retrieved_verified_evidence": evidence[:20],
            "relevant_conflicts": relevant_conflicts,
            "instruction": (
                "Return source claims, synthesis, conflicts/uncertainty, and citations separately. "
                "Do not cite or infer claims beyond the supplied evidence."
            ),
        }
        return await self.gateway.structured(
            json.dumps(payload, ensure_ascii=False, indent=2),
            RuntimeResponse,
            model=self.model,
            system_prompt=self.system_prompt,
        )


def load_runtime(
    bundle_dir: str | Path,
    *,
    gateway: ModelGateway | None = None,
    model: str | None = None,
) -> IntentionalRuntime:
    return IntentionalRuntime(bundle_dir, gateway=gateway, model=model)


def _offline_response(
    evidence: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> RuntimeResponse:
    source_claims = [
        f"{item['quote']} [source:{item['source_id']}; chunk:{item['chunk_id']}]"
        for item in evidence[:10]
    ]
    citations = [
        (
            f"source:{item['source_id']}; chunk:{item['chunk_id']}; "
            f"claims:{','.join(item['claim_ids'])}"
        )
        for item in evidence[:10]
    ]
    uncertainty = [
        f"Conflict {item['id']} remains {item['resolution']}: {item['description']}"
        for item in conflicts
    ]
    if not evidence:
        uncertainty.append("No verified bundle evidence matched the question.")
    return RuntimeResponse(
        source_claims=source_claims,
        synthesis=(
            "No model gateway is configured; the local runtime retrieved source evidence but did "
            "not generate an additional synthesis."
        ),
        conflicts_and_uncertainty=uncertainty,
        citations=citations,
    )
