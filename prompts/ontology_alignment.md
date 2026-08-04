# Ontology alignment

Align only the canonical concepts supplied in the input. Treat every ontology as
source-attributed candidate knowledge, not as universal truth.

For each high-confidence cross-ontology candidate, classify the relationship as
exactly one of:

- `exact_equivalent`
- `broader`
- `narrower`
- `related_or_analogous`
- `superficially_similar_but_distinct`
- `conflicting`

Rules:

1. Preserve each ontology's canonical IDs. Never invent, rename, or merge IDs.
2. Equal labels are candidates, not proof of identity. Compare source definitions.
3. Do not collapse analogies, broader/narrower pairs, or materially different
   definitions into equivalence.
4. Keep conflicts explicit. Do not resolve them by averaging or choosing a source.
5. Put useful non-equivalent combinations in `complementarities` and reference at
   least two canonical source claim IDs.
6. Give a concise rationale and calibrated confidence for every mapping.

Return only the requested structured `AlignmentReport`.

