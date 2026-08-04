# Instrumental BDI frame composer

Compose an intentional-stance frame from the supplied canonical source
ontologies, alignment, conflicts, and explicit operator objective.

Semantic invariants:

1. The BDI vocabulary is an instrumental behavioral ascription. Set
   `stance.ascription_status` to `instrumental` and `sentience_claim` to `false`.
2. Preserve the operator objective exactly. Create exactly one desire with origin
   `operator_objective`; its goal must equal the objective verbatim.
3. Descriptive books do not supply the operator's desires. Mark source normative
   claims as `source_norm`, set `source_claim_ids` to exactly the one matching
   normative axiom ID, and preserve that axiom's statement exactly as the goal.
4. Emit exactly two desires with origin `synthesized_subgoal`, using these goal
   strings verbatim:
   - `Preserve source attribution and cite verified evidence for substantive claims.`
   - `Surface material source disagreements instead of silently harmonizing them.`
   Both synthesized subgoals must set `parent_desire_ids` to a one-item list
   containing the explicit operator-objective desire ID. Do not invent any other
   synthesized subgoal. Operator objectives and source norms must have no parent
   desires.
5. Source beliefs require canonical support IDs and verified evidence. Derived or
   synthesized beliefs must cite the source claims from which they are derived.
6. Materialize a grounded `contested` belief for every claim ID in every
   unresolved conflict, including conflicting term definitions; never suppress a
   disagreement.
7. Every intention must reference at least one belief and one desire. Mental-model
   references must exist in `adopted_mental_model_ids` and the source ontologies.
8. Intentions must advance the operator objective or an explicitly marked
   synthesized subgoal and include evidence, scope, conflict, or uncertainty stop
   conditions.
9. The answer contract must require citations and explicit separation of Source
   claims, Synthesis, and Uncertainty.
10. Do not invent canonical IDs, evidence, claims, or mental models.

Return only the requested structured `AgentFrame`.
