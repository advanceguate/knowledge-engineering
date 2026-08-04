# Chunk-local ontology map extraction

Extract source-attributed candidate knowledge from the supplied chunk into the requested
structured schema. The result is a map fragment, not a statement of universal truth.

Rules:

1. Use only the supplied chunk. Do not add facts, definitions, implications, or terminology from
   memory or general knowledge.
2. Locate exact supporting quotations before interpreting them. Every term, relation, axiom, and
   mental model must carry at least one verbatim quotation from this chunk.
3. Keep `source_id` and `chunk_id` exactly as supplied in CHUNK METADATA. Copy heading and page
   metadata into evidence where available. Leave `verified` false; the pipeline verifies quotes.
4. Use labels, never invented canonical IDs. Identity is decided after extraction.
5. Distinguish the author's assertions from reported speech, quotations, hypotheses, conditions,
   and positions the author criticizes. Record that distinction in `epistemic_status` and
   `attribution`.
6. Classify normative language as normative; do not turn descriptive claims into prescriptions.
7. Relations may be explicit or strongly implied by the supplied words, but the quotation must
   support the precise direction and predicate. Omit weak guesses.
8. A mental model must be operational: it should state a purpose and, where the text supplies them,
   applicability, assumptions, mechanism, procedure, predictions, and failure modes.
9. Confidence measures extraction and quotation support quality, not metaphysical truth or your
   agreement with the source.
10. Omit unsupported objects rather than filling gaps. Preserve open questions as questions.

The section summary must be an extractive or close paraphrase supported by the chunk. Do not infer
an operator objective, agent desire, or intention from descriptive source material.
