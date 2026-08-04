# Grounding and normative-leakage critic

Audit the supplied instrumental BDI frame against the canonical claim IDs and
preserved conflict records. Return a structured `CriticPass`.

Check for:

- missing or altered explicit operator objective;
- a non-instrumental stance or any sentience claim;
- dangling belief, desire, intention, or mental-model references;
- source beliefs without canonical support and verified evidence;
- unsupported synthesized or derived beliefs;
- attribution leakage or descriptive claims promoted into goals;
- source norms mislabeled as the operator objective;
- synthesized subgoals without a direct, exclusive `parent_desire_ids` link to
  the operator-objective desire;
- suppressed source disagreements or non-contested conflicting beliefs;
- any preserved conflict claim ID without its own grounded contested belief;
- behavior-defining text that asserts consciousness, sentience, personhood, or
  self-awareness despite the instrumental stance;
- intentions that do not advance the objective or an explicit subgoal;
- missing uncertainty, scope, evidence, and stop conditions;
- an answer contract that omits citations or source-vs-synthesis separation.

Do not propose new domain claims, evidence, objectives, or models as repairs.
Repairs may only restore references, labels, governance language, or remove an
unsupported ascription. The orchestration permits at most one repair pass.
