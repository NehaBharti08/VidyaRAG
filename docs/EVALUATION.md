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

## Corpus statistics

Measured by the Phase 1a ingestion run. Both titles are CC BY 4.0 first
editions; see [ATTRIBUTION.md](../ATTRIBUTION.md) for why the edition matters.

| | Biology | Anatomy and Physiology |
|---|---|---|
| PDF pages | 1,480 | 1,420 |
| Pages ingested | 1,273 (86%) | 1,289 (91%) |
| Chunks | 1,651 | 1,701 |
| Tokens/chunk, mean / median | 461 / 495 | 467 / 494 |
| Chunks with a printed page number | 100% | 100% |
| Chunks with a section | 98.1% | 98.4% |
| Chunks spanning a page break | 59.7% | 58.6% |
| Download size | 279.4 MB | 135.4 MB |
| SHA-256 | `723dec671a4d…` | `16cb34457cba…` |

**Total: 3,352 chunks** — comfortably inside `qdrant-client` local mode's ~20k
point advisory, so the embedded index that ships with the demo stays fast.

### What was excluded, and why

Roughly 10% of pages are dropped before chunking. Each exclusion removes text
that would compete in retrieval while answering nothing:

- **Front matter, indices, appendices, references** — dropped via the PDF
  outline.
- **`Solutions`** — OpenStax's name for the answer key. The single most
  important exclusion: it is phrased in the same vocabulary as the questions a
  student would ask, so it ranks well and contains only bare answers.
- **End-of-chapter question sets** (`REVIEW QUESTIONS`, `CRITICAL THINKING
  QUESTIONS`, and continuation pages of multiple-choice options). These get no
  outline entry of their own — they fall under the trailing `Glossary` heading —
  so they are detected from the text. Removing them cut 314 chunks (8.6%).
- **Chapter summaries and key terms are deliberately kept.** Summaries are
  condensed explanation and key terms are definitions; both answer real
  questions.

### Extraction quality gate

The Phase 1a gate was 20 randomly sampled chunks inspected for clean text and
correct metadata, plus automated contamination checks across all 3,352 chunks.
All checks returned zero: no non-breaking spaces (the PDF outline is full of
them), no leaked running heads or footer boilerplate, no control characters, no
empty chunks.

Two findings drove real fixes rather than being noted and ignored:

1. **Biology's outline contains a `Blank Page` entry pointing backwards** to
   page 6, roughly 1,450 pages before its neighbour. Processed in document
   order it made the final range swallow the entire book. Outline entries are
   now sorted by page.
2. **Printed page numbers are not declared in the PDF.** `get_label()` returns
   empty on every page, but the number a student would look up is printed in
   the running head — which was already being stripped. It is now recovered
   from there, so a citation points at printed page 121 rather than PDF page
   133. The offset is consistent (12 for Biology, 10 for Anatomy and
   Physiology) but is read per page rather than assumed.

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
