# Grounded ontology reduction

Reduce the supplied source-attributed fragments into a clearer candidate ontology without adding
domain claims.

Allowed operations:

- Merge genuine duplicates and exact aliases.
- Select a canonical display label from labels already present.
- Combine and deduplicate verified evidence.
- Retain explicit broader, narrower, and related links.
- Select salient supported objects under the stated budget.
- Summarize only by extractive combination of supported fragment summaries.

Prohibited operations:

- Introducing a term, relation, axiom, mental model, condition, exception, or conclusion that is
  absent from the fragments.
- Treating an analogy as an identity or a hierarchy relation as synonymy.
- Discarding disagreement by averaging or silently choosing one source claim.
- Changing attribution or epistemic status.
- Inventing canonical IDs. IDs are assigned deterministically after resolution.
- Returning an object without verified evidence.

When evidence supports materially different definitions or opposing claims, retain both and expose
the unresolved conflict. Raw fragments remain artifacts even when a candidate is omitted from the
curated salience budget.
