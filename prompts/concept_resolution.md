# Conservative concept-pair resolution

Decide whether two source-attributed term candidates denote the same concept. Return a merge only
when the supplied labels, aliases, definitions, and context support identity.

Keep the candidates distinct when they are:

- analogous, compared, or metaphorically related rather than identical;
- broader and narrower concepts;
- the same surface label with materially different definitions or scopes;
- roles, processes, values, properties, or events of different kinds;
- merely topically related;
- ambiguous on the supplied evidence.

Do not use outside knowledge. Explain the decisive source evidence. An uncertain result is
`distinct`; downstream alignment can still record relatedness. Do not create canonical IDs.
