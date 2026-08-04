# Knowledge Engineering

`knowledge-engineering` is a local-first Python package for turning books and
documents into source-attributed candidate ontologies, then composing several
ontologies into an intentional-stance reasoning frame.

The package treats every extracted term, relation, axiom, and mental model as a
claim attributed to evidence—not as universal truth. Agent goals always preserve
an explicit operator objective. Beliefs, desires, and intentions are instrumental
behavioral ascriptions and do not imply consciousness or sentience.

## Quick start

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
cp .env.example .env
uv run ke --help
```

The default artifact store is `workspace/`. See `ke --help` for ingestion,
extraction, synthesis, validation, end-to-end build, and optional runtime commands.

## Pipelines

The package exposes three bounded LangGraph workflows:

1. **Source ingestion** fingerprints a PDF, EPUB, HTML, DOCX, Markdown, or text
   source; converts it to Markdown and canonical JSON; creates layout-aware
   provenance chunks; and publishes a content-addressed source bundle.
2. **Domain extraction** maps each chunk into structured candidate knowledge,
   verifies every quotation, resolves entities, reduces fragments hierarchically,
   assigns deterministic IDs, and publishes JSON, RDF/SKOS/PROV, and SHACL results.
3. **Intentional synthesis** aligns source ontologies without collapsing their
   named graphs, preserves conflicts, composes a source-grounded BDI frame from an
   explicit objective, runs one bounded critic/repair pass, and compiles a prompt.

```bash
# Convert a PDF, or ingest any supported format
uv run ke pdf-to-markdown books/domain-book.pdf
uv run ke ingest books/domain-book.epub

# Extract and validate a source ontology
uv run ke extract workspace/sources/<source-id>
uv run ke validate workspace/ontologies/<ontology-id>

# Compose multiple ontologies; the objective is mandatory
uv run ke synthesize \
  workspace/ontologies/<ontology-a>/ontology.json \
  workspace/ontologies/<ontology-b>/ontology.json \
  --name organizational-reasoner \
  --objective "Reason about organizational incentives while preserving disagreements"

# Run ingestion, extraction, and synthesis together
uv run ke build books/book-a.pdf books/book-b.epub \
  --name cross-domain-reasoner \
  --objective "Apply the sources to product and organizational strategy"
```

Model identifiers are configured through `.env`. `openai:` and
`openai-responses:` identifiers are routed explicitly through the OpenAI Responses
API; an explicit `openai-chat:` identifier retains legacy Chat Completions routing,
and other provider identifiers retain PydanticAI's provider routing. The default
split uses GPT-5.6 Terra for extraction and resolution, and GPT-5.6 Sol for
synthesis and critique. Tests and offline workflows use the same structured
gateway contract with deterministic fake or conservative local implementations.
Add `--offline` to `extract`, `synthesize`, `build`, or `ask` to run without
provider credentials.

`OPENAI_API_KEY` is loaded from `.env` as a masked secret and passed directly to
the OpenAI provider. It is not included in artifact identities or model caches,
and Responses API server-side storage is explicitly disabled.

## Artifact layout

Published artifacts use deterministic identities and a local content-addressed
cache by default. Cache reads verify required artifact digests and fail closed on
missing, corrupt, or mismatched files:

```text
workspace/
├── sources/<source-id>/
├── ontologies/<ontology-id>/
├── syntheses/<frame-id>/
├── cache/
└── runs/
```

Each source object remains attributable to an exact verified quotation. Source
ontologies stay in separate named graphs during synthesis; mappings, conflicts,
and genuinely synthesized claims live in an integration graph.

## Semantic guarantees

- Extracted objects are **source-attributed candidate knowledge**, not universal
  truth.
- Canonical ontologies reject unverified evidence, duplicate IDs, and dangling
  references.
- An agent frame must contain exactly one explicit operator-objective desire, and
  intentions must link to valid beliefs and desires.
- Source norms stay marked as source norms. They are never silently promoted to
  the operator objective.
- BDI vocabulary is an instrumental behavioral ascription. The generated stance
  makes no claim of consciousness or sentience.
- The compiled reasoning policy requires citations, preserves conflicts, and
  separates source claims, synthesis, and uncertainty.

## Development

```bash
uv run pytest
uv run ruff check src tests
```
