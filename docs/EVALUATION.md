# Evaluation

Methodology, gold set provenance, and per-phase results.

> **Status: pending.** The harness lands in Phase 3. This file is deliberately
> created empty of results rather than pre-populated — no number appears here
> before it has been measured.

---

## Principles

1. **One frozen control.** The `baseline` profile (dense retrieval, no
   reranking, no corrective loop) is fixed after Phase 2 and never edited. Every
   reported delta is measured against it, so all numbers stay comparable.
2. **Ablate independently.** Each enhancement is measured on its own, not only
   in combination. A stack that improves overall while containing a component
   that hurts is a stack that has not been understood.
3. **Report what failed.** Changes that did not help, or made things worse, are
   recorded below with a hypothesis. A documented negative result is evidence
   the evaluation is real.
4. **Cache aggressively.** Grading calls are cached by prompt hash so re-running
   after a code change costs approximately nothing, and the full suite is never
   wired to run on every commit.

---

## Gold set

Target composition, 60 questions:

| Type | Count | What it tests |
|---|---|---|
| Factual lookup (single section) | 24 (40%) | Baseline competence; regression canary |
| Multi-hop (2+ sections) | 18 (30%) | What decomposition and hybrid retrieval must move |
| Unanswerable but plausible | 12 (20%) | **Abstention.** The differentiating capability |
| Ambiguous / false presupposition | 6 (10%) | Graceful degradation |

### Provenance — stated plainly

Answerable questions are **drafted by `gpt-4o-mini` from sampled chunks, then
verified by hand.** Each is checked for three things: that it reads naturally,
that the cited chunk genuinely answers it, and that it is *not* answerable from
a model's parametric knowledge without retrieval. That third check is what stops
the evaluation from silently measuring nothing.

Unanswerable questions are **written by hand.** An LLM asked to produce
unanswerable questions reliably produces obviously out-of-domain ones, which
would make abstention look easy and the resulting metric meaningless. These must
be biology-shaped and plausible.

This is standard practice and is documented rather than described as
"hand-curated", which would be inaccurate.

---

## Metrics

RAGAS: faithfulness, answer relevancy, context precision, context recall.

RAGAS moved metrics to `ragas.metrics.collections` in v0.4; the legacy
`ragas.metrics` path is deprecated and removed in 1.0. All contact with that API
is confined to `eval/metrics.py` so version churn touches exactly one file.

Abstention is scored separately from RAGAS, since correctly refusing to answer
is not something a faithfulness score captures:

- **Abstention precision** — of the questions the system declined, how many were
  genuinely unanswerable.
- **Abstention recall** — of the genuinely unanswerable questions, how many it
  declined.
- **False abstention rate** — answerable questions wrongly refused. The cost of
  over-abstaining, which a recall-only view hides.

---

## Results

_Pending Phase 3._

## What did not work

_Pending. This section is expected to be non-empty._
