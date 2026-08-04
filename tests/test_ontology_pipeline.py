from __future__ import annotations

import json

import pytest

import ke.ontology.reduce as reduce_module
import ke.ontology.resolve as resolve_module
from ke.artifact_store import ArtifactStore
from ke.contracts import (
    DraftAxiom,
    DraftMentalModel,
    DraftRelation,
    DraftTerm,
    Evidence,
    ExtractionFragment,
)
from ke.grounding import quote_sha256
from ke.ids import sha256_file
from ke.llm import FakeModelGateway
from ke.ontology.chunking import DocumentChunk
from ke.ontology.extract import extract_chunks, ground_fragment, offline_extract
from ke.ontology.reduce import hierarchical_reduce, reduce_fragments
from ke.ontology.resolve import resolve_entities, resolve_entities_async
from ke.pipelines.domain_ontology import run_domain_ontology
from ke.settings import Settings


def write_source_bundle(
    root,
    *,
    source_id: str = "book-aabbccddeeff",
    chunk_source_id: str | None = None,
    text: str = "Market is a coordination system.",
):
    root.mkdir()
    source_path = root / "source.txt"
    markdown_path = root / "document.md"
    canonical_path = root / "document.docling.json"
    chunks_path = root / "chunks.jsonl"
    source_path.write_text(text, encoding="utf-8")
    markdown_path.write_text(text, encoding="utf-8")
    canonical_path.write_text(json.dumps({"text": text}), encoding="utf-8")
    chunks_path.write_text(
        json.dumps(
            {
                "chunk_id": "chunk-1",
                "source_id": chunk_source_id or source_id,
                "ordinal": 1,
                "text": text,
                "heading_path": ["Markets"],
                "page_start": 2,
                "page_end": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_digest = sha256_file(source_path)
    artifacts = {
        "source": {"path": source_path.name, "sha256": source_digest},
        "markdown": {"path": markdown_path.name, "sha256": sha256_file(markdown_path)},
        "canonical_document": {
            "path": canonical_path.name,
            "sha256": sha256_file(canonical_path),
        },
        "chunks": {"path": chunks_path.name, "sha256": sha256_file(chunks_path)},
        "assets": [],
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_id": source_id,
                "title": "Small Book",
                "content_sha256": source_digest,
                "source": {
                    "stored_path": source_path.name,
                    "sha256": source_digest,
                },
                "files": {
                    "source": source_path.name,
                    "markdown": markdown_path.name,
                    "canonical_document": canonical_path.name,
                    "chunks": chunks_path.name,
                },
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_offline_extraction_strips_layout_heading_and_keeps_exact_quotes():
    text = (
        "Coordination\n\n"
        "Coordination Cost is the effort required to align work. "
        "The Alignment Model is a framework for comparing that effort."
    )
    fragment = offline_extract(
        DocumentChunk(
            chunk_id="chunk-heading",
            source_id="book-heading",
            ordinal=1,
            text=text,
            heading_path=["Coordination"],
        )
    )

    assert fragment.terms[0].label == "Coordination Cost"
    assert fragment.terms[0].evidence[0].quote == (
        "Coordination Cost is the effort required to align work."
    )
    assert fragment.terms[0].evidence[0].verified
    assert [model.name for model in fragment.mental_models] == ["Alignment Model"]
    assert fragment.mental_models[0].evidence[0].quote in text


@pytest.mark.asyncio
async def test_pipeline_grounds_fragments_exports_and_caches(tmp_path):
    source_id = "small-book-aabbccddeeff"
    quote = "Markets coordinate exchange."
    source_dir = write_source_bundle(tmp_path / "source", source_id=source_id, text=quote)
    evidence = Evidence(
        source_id="model-invented-source",
        chunk_id="model-invented-chunk",
        heading_path=["Fabricated Heading"],
        page_start=999,
        page_end=999,
        quote=quote,
        quote_sha256=quote_sha256(quote),
        verified=False,
    )
    fragment = ExtractionFragment(
        source_id="model-invented-source",
        chunk_id="model-invented-chunk",
        section_summary=quote,
        terms=[
            DraftTerm(
                label="Market",
                aliases=["markets"],
                kind="system",
                definition="a system that coordinates exchange",
                evidence=[evidence],
                confidence=0.9,
            ),
            DraftTerm(
                label="Exchange",
                kind="process",
                definition="transfer between parties",
                evidence=[evidence],
                confidence=0.8,
            ),
        ],
        relations=[
            DraftRelation(
                subject_label="Market",
                predicate="coordinates",
                object_label="Exchange",
                evidence=[evidence],
                confidence=0.9,
            )
        ],
        axioms=[
            DraftAxiom(
                statement=quote,
                modality="descriptive",
                epistemic_status="asserted",
                evidence=[evidence],
                confidence=0.9,
            )
        ],
    )
    gateway = FakeModelGateway(responses=[fragment])
    gateway.cache_identity = "pipeline-grounding-fixture"
    settings = Settings(workspace=tmp_path / "workspace")
    store = ArtifactStore(settings.workspace)

    first = await run_domain_ontology(
        source_dir,
        settings=settings,
        store=store,
        gateway=gateway,
    )
    assert first.validation.conforms
    assert not first.cache_hit
    assert len(first.ontology.terms) == 2
    assert all(
        evidence.verified
        for item in [
            *first.ontology.terms,
            *first.ontology.relations,
            *first.ontology.axioms,
        ]
        for evidence in item.evidence
    )
    assert {
        (evidence.source_id, evidence.chunk_id)
        for item in [
            *first.ontology.terms,
            *first.ontology.relations,
            *first.ontology.axioms,
        ]
        for evidence in item.evidence
    } == {(source_id, "chunk-1")}
    assert {
        (tuple(evidence.heading_path), evidence.page_start, evidence.page_end)
        for item in [
            *first.ontology.terms,
            *first.ontology.relations,
            *first.ontology.axioms,
        ]
        for evidence in item.evidence
    } == {(("Markets",), 2, 2)}
    output = store.ontology_dir(first.ontology_id)
    for filename in (
        "ontology.json",
        "ontology.ttl",
        "ontology.jsonld",
        "validation.json",
        "entities-thesaurus.yaml",
        "axioms.yaml",
        "mental-models.yaml",
        "conflicts.yaml",
        "resolution.json",
        "book-profile.json",
    ):
        assert (output / filename).is_file()
    raw = next((output / "fragments").glob("*.json"))
    raw_payload = json.loads(raw.read_text(encoding="utf-8"))
    assert raw_payload["source_id"] == source_id
    assert raw_payload["chunk_id"] == "chunk-1"
    assert raw_payload["terms"][0]["evidence"][0]["verified"] is False

    second = await run_domain_ontology(
        source_dir,
        settings=settings,
        store=store,
        gateway=gateway,
    )
    assert second.cache_hit
    assert second.ontology_id == first.ontology_id
    assert len(gateway.calls) == 1

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "abstract.md",
        "axioms.yaml",
        "book-profile.json",
        "conflicts.yaml",
        "entities-thesaurus.yaml",
        "mental-models.yaml",
        "ontology.json",
        "ontology.jsonld",
        "ontology.ttl",
        "resolution.json",
        "validation.json",
    }
    assert required <= set(manifest["artifacts"])
    assert len([name for name in manifest["artifacts"] if name.startswith("fragments/")]) == 1

    (output / "entities-thesaurus.yaml").write_text("corrupt", encoding="utf-8")
    recovery_gateway = FakeModelGateway(responses=[fragment])
    recovery_gateway.cache_identity = "pipeline-grounding-fixture"
    recovered = await run_domain_ontology(
        source_dir,
        settings=settings,
        store=store,
        gateway=recovery_gateway,
    )
    assert not recovered.cache_hit
    assert len(recovery_gateway.calls) == 1

    refreshed_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    cache_path = store.cache_path("domain-ontology", refreshed_manifest["pipeline_cache_key"])
    cache_path.write_text("{corrupt", encoding="utf-8")
    corrupt_cache_gateway = FakeModelGateway(responses=[fragment])
    corrupt_cache_gateway.cache_identity = "pipeline-grounding-fixture"
    recovered_again = await run_domain_ontology(
        source_dir,
        settings=settings,
        store=store,
        gateway=corrupt_cache_gateway,
    )
    assert not recovered_again.cache_hit
    assert len(corrupt_cache_gateway.calls) == 1


def test_resolution_does_not_bridge_materially_different_definitions():
    quote = "The word bank is used in several contexts."
    evidence = Evidence(
        source_id="book",
        chunk_id="chunk",
        quote=quote,
        quote_sha256=quote_sha256(quote),
        verified=True,
    )
    fragment = ExtractionFragment(
        source_id="book",
        chunk_id="chunk",
        section_summary=quote,
        terms=[
            DraftTerm(
                label="Bank",
                kind="system",
                definition="a regulated financial institution handling deposited money",
                evidence=[evidence],
                confidence=0.9,
            ),
            DraftTerm(
                label="Bank",
                kind="concept",
                definition="",
                evidence=[evidence],
                confidence=0.6,
            ),
            DraftTerm(
                label="Bank",
                kind="entity",
                definition="sloping land bordering a flowing river",
                evidence=[evidence],
                confidence=0.9,
            ),
        ],
    )

    resolution = resolve_entities([fragment])

    assert len(resolution.clusters) == 2
    assert any(
        decision.stage == "guardrail" and decision.decision == "distinct"
        for decision in resolution.decisions
    )


def test_reduction_preserves_direct_contradictions_as_conflicts():
    positive = "Incentives increase effort."
    negative = "Incentives do not increase effort."

    def evidence(quote: str) -> Evidence:
        return Evidence(
            source_id="book",
            chunk_id="chunk",
            quote=quote,
            quote_sha256=quote_sha256(quote),
            verified=True,
        )

    fragment = ExtractionFragment(
        source_id="book",
        chunk_id="chunk",
        section_summary=f"{positive} {negative}",
        axioms=[
            DraftAxiom(
                statement=positive,
                modality="causal",
                epistemic_status="asserted",
                evidence=[evidence(positive)],
                confidence=0.9,
            ),
            DraftAxiom(
                statement=negative,
                modality="causal",
                epistemic_status="contested",
                evidence=[evidence(negative)],
                confidence=0.9,
            ),
        ],
    )

    ontology = reduce_fragments([fragment], title="Incentives")

    assert len(ontology.conflicts) == 1
    assert ontology.conflicts[0].kind == "causal"
    assert set(ontology.conflicts[0].claim_ids) == {axiom.id for axiom in ontology.axioms}


@pytest.mark.asyncio
async def test_pipeline_rejects_manifest_chunk_source_mismatch(tmp_path):
    source_dir = write_source_bundle(
        tmp_path / "mismatch",
        source_id="manifest-source",
        chunk_source_id="different-source",
    )

    with pytest.raises(ValueError, match="does not match chunk source_id"):
        await run_domain_ontology(
            source_dir,
            settings=Settings(workspace=tmp_path / "workspace"),
            gateway=None,
        )


@pytest.mark.asyncio
async def test_pipeline_rejects_chunks_tampered_after_manifest_publication(tmp_path):
    source_dir = write_source_bundle(tmp_path / "tampered")
    chunks_path = source_dir / "chunks.jsonl"
    chunks_path.write_text(chunks_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact digest mismatch for 'chunks'"):
        await run_domain_ontology(
            source_dir,
            settings=Settings(workspace=tmp_path / "workspace"),
            gateway=None,
        )


@pytest.mark.asyncio
async def test_cache_key_separates_titles_and_chunking_settings(tmp_path):
    source_dir = write_source_bundle(tmp_path / "source")
    store = ArtifactStore(tmp_path / "workspace")
    settings = Settings(workspace=store.root)

    first = await run_domain_ontology(
        source_dir,
        title="First title",
        settings=settings,
        store=store,
        gateway=None,
    )
    cached = await run_domain_ontology(
        source_dir,
        title="First title",
        settings=settings,
        store=store,
        gateway=None,
    )
    retitled = await run_domain_ontology(
        source_dir,
        title="Second title",
        settings=settings,
        store=store,
        gateway=None,
    )
    different_chunking = Settings(
        workspace=store.root,
        chunking={"max_tokens": 1700, "overlap_tokens": 150},
    )
    rechunked_key = await run_domain_ontology(
        source_dir,
        title="Second title",
        settings=different_chunking,
        store=store,
        gateway=None,
    )

    assert not first.cache_hit
    assert cached.cache_hit
    assert not retitled.cache_hit
    assert first.ontology_id != retitled.ontology_id
    assert not rechunked_key.cache_hit


@pytest.mark.asyncio
async def test_cache_key_separates_offline_and_live_extraction(tmp_path):
    text = "Market is a coordination system."
    source_dir = write_source_bundle(tmp_path / "source", text=text)
    store = ArtifactStore(tmp_path / "workspace")
    settings = Settings(workspace=store.root)

    offline = await run_domain_ontology(
        source_dir,
        settings=settings,
        store=store,
        gateway=None,
    )
    evidence = Evidence(
        source_id="book-aabbccddeeff",
        chunk_id="chunk-1",
        quote=text,
        quote_sha256=quote_sha256(text),
        verified=False,
    )
    live_fragment = ExtractionFragment(
        source_id="book-aabbccddeeff",
        chunk_id="chunk-1",
        section_summary="",
        terms=[
            DraftTerm(
                label="Coordination System",
                kind="system",
                definition="a system for coordinating participants",
                evidence=[evidence],
                confidence=0.9,
            )
        ],
    )
    gateway = FakeModelGateway(responses=[live_fragment])
    live = await run_domain_ontology(
        source_dir,
        settings=settings,
        store=store,
        gateway=gateway,
    )
    cached_live = await run_domain_ontology(
        source_dir,
        settings=settings,
        store=store,
        gateway=gateway,
    )
    separate_gateway = FakeModelGateway(responses=[live_fragment])
    separate_live = await run_domain_ontology(
        source_dir,
        settings=settings,
        store=store,
        gateway=separate_gateway,
    )

    assert not offline.cache_hit
    assert not live.cache_hit
    assert cached_live.cache_hit
    assert not separate_live.cache_hit
    assert len(gateway.calls) == 1
    assert len(separate_gateway.calls) == 1


@pytest.mark.asyncio
async def test_map_cache_key_tracks_full_chunk_metadata():
    fragment = ExtractionFragment(
        source_id="book",
        chunk_id="chunk",
        section_summary="",
    )
    first_gateway = FakeModelGateway(responses=[fragment])
    second_gateway = FakeModelGateway(responses=[fragment])
    common = {
        "chunk_id": "chunk",
        "source_id": "book",
        "ordinal": 1,
        "text": "The same chunk text.",
    }
    await extract_chunks(
        [DocumentChunk(**common, heading_path=["First"], page_start=1, page_end=1)],
        first_gateway,
        ground_evidence=False,
    )
    await extract_chunks(
        [DocumentChunk(**common, heading_path=["Second"], page_start=2, page_end=3)],
        second_gateway,
        ground_evidence=False,
    )

    assert first_gateway.calls[0]["cache_key"] != second_gateway.calls[0]["cache_key"]


@pytest.mark.asyncio
async def test_concept_resolution_cache_key_tracks_prompt(monkeypatch):
    quote = "Market coordination is discussed with market coordinatoin."
    evidence = Evidence(
        source_id="book",
        chunk_id="chunk",
        quote=quote,
        quote_sha256=quote_sha256(quote),
        verified=True,
    )
    fragment = ExtractionFragment(
        source_id="book",
        chunk_id="chunk",
        section_summary="",
        terms=[
            DraftTerm(
                label="Market coordination",
                kind="process",
                definition="coordination among market participants",
                evidence=[evidence],
                confidence=0.8,
            ),
            DraftTerm(
                label="Market coordinatoin",
                kind="process",
                definition="coordination among market participants",
                evidence=[evidence],
                confidence=0.8,
            ),
        ],
    )

    monkeypatch.setattr(resolve_module, "read_resource_text", lambda *_: "prompt version one")
    first_gateway = FakeModelGateway(responses=[{"equivalent": False, "reason": "distinct"}])
    await resolve_entities_async(
        [fragment],
        gateway=first_gateway,
        model="test:model",
        lexical_threshold=100,
    )
    monkeypatch.setattr(resolve_module, "read_resource_text", lambda *_: "prompt version two")
    second_gateway = FakeModelGateway(responses=[{"equivalent": False, "reason": "distinct"}])
    await resolve_entities_async(
        [fragment],
        gateway=second_gateway,
        model="test:model",
        lexical_threshold=100,
    )

    assert first_gateway.calls[0]["cache_key"] != second_gateway.calls[0]["cache_key"]


def test_grounding_overwrites_fabricated_provenance_and_quarantines_unsupported():
    supported_quote = "Supported concept is defined here."
    chunk = DocumentChunk(
        chunk_id="canonical-chunk",
        source_id="canonical-source",
        ordinal=1,
        text=supported_quote,
        heading_path=["Canonical heading"],
        page_start=4,
        page_end=5,
    )
    fabricated = Evidence(
        source_id="fabricated-source",
        chunk_id="fabricated-chunk",
        heading_path=["Fabricated heading"],
        page_start=999,
        page_end=999,
        quote=supported_quote,
        quote_sha256="not-yet-grounded",
        verified=False,
    )
    unsupported = fabricated.model_copy(
        update={"quote": "This quotation is absent.", "quote_sha256": "wrong"}
    )
    fragment = ExtractionFragment(
        source_id="fabricated-source",
        chunk_id="fabricated-chunk",
        section_summary="A fabricated summary.",
        terms=[
            DraftTerm(
                label="Supported concept",
                kind="concept",
                definition="defined here",
                evidence=[fabricated],
                confidence=0.7,
            ),
            DraftTerm(
                label="Poison concept",
                kind="value",
                definition="unsupported poison",
                evidence=[unsupported],
                confidence=1.0,
            ),
        ],
    )

    grounded = ground_fragment(fragment, chunk, strict=False)

    assert [term.label for term in grounded.terms] == ["Supported concept"]
    evidence = grounded.terms[0].evidence[0]
    assert evidence.source_id == "canonical-source"
    assert evidence.chunk_id == "canonical-chunk"
    assert evidence.heading_path == ["Canonical heading"]
    assert (evidence.page_start, evidence.page_end) == (4, 5)


def test_unverified_draft_and_fake_summary_cannot_poison_canonical_ontology():
    quote = "Market is a supported coordination concept."
    verified = Evidence(
        source_id="book",
        chunk_id="chunk",
        quote=quote,
        quote_sha256=quote_sha256(quote),
        verified=True,
    )
    unverified = Evidence(
        source_id="book",
        chunk_id="chunk",
        quote="Unsupported poison.",
        quote_sha256="model-invented-hash",
        verified=False,
    )
    fragment = ExtractionFragment(
        source_id="book",
        chunk_id="chunk",
        section_summary="FAKE SUMMARY: maximize an unsupported objective.",
        terms=[
            DraftTerm(
                label="Market",
                kind="concept",
                definition="a supported coordination concept",
                evidence=[verified],
                confidence=0.6,
            ),
            DraftTerm(
                label="Market",
                kind="value",
                definition="an unsupported normative command",
                evidence=[unverified],
                confidence=1.0,
            ),
        ],
        mental_models=[
            DraftMentalModel(
                name="Poison Model",
                purpose="Introduce unsupported fields",
                evidence=[unverified],
                confidence=1.0,
            )
        ],
    )

    ontology = reduce_fragments([fragment], title="Grounded")

    assert len(ontology.terms) == 1
    assert ontology.terms[0].kind == "concept"
    assert ontology.terms[0].definition == "a supported coordination concept"
    assert ontology.terms[0].confidence == 0.6
    assert ontology.mental_models == []
    assert "FAKE SUMMARY" not in ontology.abstract
    assert ontology.abstract_support_ids == [ontology.terms[0].id]
    assert ontology.terms[0].definition in ontology.abstract


def test_direct_bank_homonyms_remain_distinct():
    quote = "Bank has distinct source definitions."
    evidence = Evidence(
        source_id="book",
        chunk_id="chunk",
        quote=quote,
        quote_sha256=quote_sha256(quote),
        verified=True,
    )
    fragment = ExtractionFragment(
        source_id="book",
        chunk_id="chunk",
        section_summary="",
        terms=[
            DraftTerm(
                label="Bank",
                kind="system",
                definition="a regulated institution that holds deposited money",
                evidence=[evidence],
                confidence=0.9,
            ),
            DraftTerm(
                label="Bank",
                kind="entity",
                definition="land that borders the edge of a flowing river",
                evidence=[evidence],
                confidence=0.9,
            ),
        ],
    )

    assert len(resolve_entities([fragment]).clusters) == 2


def test_hierarchical_reducer_obeys_bounded_batch_size(monkeypatch):
    fragments: list[ExtractionFragment] = []
    for index in range(5):
        quote = f"Concept {index} is supported."
        evidence = Evidence(
            source_id="book",
            chunk_id=f"chunk-{index}",
            quote=quote,
            quote_sha256=quote_sha256(quote),
            verified=True,
        )
        fragments.append(
            ExtractionFragment(
                source_id="book",
                chunk_id=f"chunk-{index}",
                section_summary="fabricated and ignored",
                terms=[
                    DraftTerm(
                        label=f"Concept {index}",
                        kind="concept",
                        definition=f"supported concept {index}",
                        evidence=[evidence],
                        confidence=0.8,
                    )
                ],
            )
        )

    observed_batch_sizes: list[int] = []
    original = reduce_module._merge_fragment_batch

    def observe(batch):
        observed_batch_sizes.append(len(batch))
        return original(batch)

    monkeypatch.setattr(reduce_module, "_merge_fragment_batch", observe)
    ontology = hierarchical_reduce(
        fragments,
        title="Concepts",
        source_ids=["book"],
        batch_size=2,
    )

    assert observed_batch_sizes
    assert max(observed_batch_sizes) == 2
    assert {term.preferred_label for term in ontology.terms} == {
        f"Concept {index}" for index in range(5)
    }
    with pytest.raises(ValueError, match="at least 2"):
        hierarchical_reduce(fragments, title="Concepts", batch_size=1)


def test_hierarchical_reducer_preserves_supplied_multi_fragment_adjudication():
    fragments: list[ExtractionFragment] = []
    for index, label in enumerate(("Alpha", "Beta")):
        quote = f"{label} denotes the shared coordination mechanism."
        evidence = Evidence(
            source_id="book",
            chunk_id=f"chunk-{index}",
            quote=quote,
            quote_sha256=quote_sha256(quote),
            verified=True,
        )
        fragments.append(
            ExtractionFragment(
                source_id="book",
                chunk_id=f"chunk-{index}",
                section_summary="",
                terms=[
                    DraftTerm(
                        label=label,
                        kind="concept",
                        definition="the shared coordination mechanism",
                        evidence=[evidence],
                        confidence=0.8,
                    )
                ],
            )
        )
    supplied = resolve_entities(
        fragments,
        lexical_threshold=100,
        ambiguous_threshold=0,
        adjudicator=lambda **_: True,
    )

    deterministic = hierarchical_reduce(fragments, title="Mechanisms", batch_size=2)
    adjudicated = hierarchical_reduce(
        fragments,
        title="Mechanisms",
        resolution=supplied,
        batch_size=2,
    )

    assert len(deterministic.terms) == 2
    assert len(adjudicated.terms) == 1
    assert set(adjudicated.terms[0].aliases) | {adjudicated.terms[0].preferred_label} == {
        "Alpha",
        "Beta",
    }


def test_hierarchical_reducer_preserves_fragment_local_homonym_relations():
    def make_fragment(
        chunk_id: str,
        *,
        bank_definition: str,
        object_label: str,
        predicate: str,
    ) -> ExtractionFragment:
        quote = f"Bank {predicate} {object_label}."
        evidence = Evidence(
            source_id="book",
            chunk_id=chunk_id,
            quote=quote,
            quote_sha256=quote_sha256(quote),
            verified=True,
        )
        return ExtractionFragment(
            source_id="book",
            chunk_id=chunk_id,
            section_summary="",
            terms=[
                DraftTerm(
                    label="Bank",
                    kind="system" if object_label == "Deposit" else "entity",
                    definition=bank_definition,
                    evidence=[evidence],
                    confidence=0.9,
                ),
                DraftTerm(
                    label=object_label,
                    kind="concept",
                    definition=f"the {object_label.casefold()} concept",
                    evidence=[evidence],
                    confidence=0.8,
                ),
            ],
            relations=[
                DraftRelation(
                    subject_label="Bank",
                    predicate=predicate,
                    object_label=object_label,
                    evidence=[evidence],
                    confidence=0.9,
                )
            ],
        )

    fragments = [
        make_fragment(
            "finance",
            bank_definition="a regulated institution that holds deposited money",
            object_label="Deposit",
            predicate="accepts",
        ),
        make_fragment(
            "river",
            bank_definition="land that borders the edge of a flowing river",
            object_label="Water",
            predicate="borders",
        ),
    ]
    ontology = hierarchical_reduce(
        fragments,
        title="Banks",
        resolution=resolve_entities(fragments),
        batch_size=2,
    )

    bank_terms = [term for term in ontology.terms if term.preferred_label == "Bank"]
    assert len(bank_terms) == 2
    bank_id_by_domain = {
        "finance" if "money" in term.definition else "river": term.id for term in bank_terms
    }
    relation_by_predicate = {relation.predicate: relation for relation in ontology.relations}
    assert relation_by_predicate["accepts"].subject_id == bank_id_by_domain["finance"]
    assert relation_by_predicate["borders"].subject_id == bank_id_by_domain["river"]
