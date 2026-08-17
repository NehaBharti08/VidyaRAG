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
| Chunks | 1,765 | 1,843 |
| Chunks with a printed page number | 100% | 100% |
| Chunks with a section | ~98% | ~98% |
| Download size | 279.4 MB | 135.4 MB |
| SHA-256 | `723dec671a4d…` | `16cb34457cba…` |

**Total: 3,608 chunks**, all 3,608 indexed — comfortably inside `qdrant-client`
local mode's ~20k point advisory, so the embedded index that ships with the demo
stays fast.

### Index

| | |
|---|---|
| Embedding model | `BAAI/bge-base-en-v1.5` (fastembed, ONNX, CPU) |
| Dimensions | 768, cosine |
| Points | 3,608 |
| Index size on disk | ~34 MB |
| Build time | 3,464 s (~58 min, CPU only) |
| Cost | **$0** — no API key is needed to build the index |

Chunk sizing is measured with **the embedding model's own tokeniser**, not
tiktoken. `bge-base-en-v1.5` truncates hard at 512 tokens and counts BERT
WordPiece, roughly 7% more than `cl100k_base` on the same prose — so chunks
sized to 512 tiktoken tokens would have arrived as ~548 model tokens and had
their tails silently dropped at embed time. Text still shown to the reader,
still named in the citation, contributing nothing to the vector that retrieved
it. Measured after the fix: **maximum 487 tokens, zero chunks over the limit.**

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
  so they are detected from the text. Removing them cut ~8.6% of chunks.
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

Answerable questions are **drafted by the generation model from sampled chunks, then
verified by hand.** Each is checked for three things: that it reads naturally,
that the cited chunk genuinely answers it, and that it is *not* answerable from
a model's parametric knowledge without retrieval. That third check is what stops
the evaluation from silently measuring nothing.

Unanswerable questions are **machine-proposed, mechanically verified against the
corpus, then approved by hand** — recorded as `llm_drafted_retrieval_verified`,
never as `human_written`.

The original plan was to author these by hand, for a good reason: an LLM asked
plainly for questions a corpus cannot answer produces obviously out-of-domain
ones. "What is the capital of France?" is refused trivially, and an abstention
score built on such questions measures nothing.

The objection is to *unverified* generation, though, not to generation as such.
So a candidate here has to clear two independent checks before a person ever
sees it:

1. **In domain**, measured as retrieval similarity against the real index. The
   cutoff is not chosen by feel — it is the 10th percentile of the top-1 scores
   of the drafted answerable questions, which are in domain by construction,
   having been written *from* corpus passages. Calibrated value: **0.714**.
2. **Genuinely absent**, judged by a grader reading the passages that similarity
   actually retrieved.

The useful case is exactly a question that scores high on the first and fails
the second: topically adjacent, plausibly in scope, and not in the book. Neither
check alone finds those, and the out-of-domain failure mode is eliminated by the
first one rather than by hoping the prompt was good enough.

Measured acceptance rate: **roughly a third of candidates**. A filter that
accepted nearly everything would not be filtering, so the rate is reported and
the tool warns above 90%.

### A third check the first two missed

The two checks above catch *topical* triviality. They are blind to *stylistic*
triviality, and the first full run demonstrated the difference painfully: of
twelve accepted questions, **eleven contained the word "exact"** and nine opened
"What is the exact...". Two were near-duplicates of each other.

Every one passed both checks legitimately. They were in domain and genuinely
absent from the corpus. They were also worthless, because a system could learn
"phrasing like *exact atomic-level crystal structure of X* → refuse" and score
perfect abstention without doing any groundedness reasoning at all — precisely
the failure the verification was built to prevent, arriving through a door
nobody was watching.

Two additions fixed it:

- **Shape rotation.** Eight question forms — a quantitative value, a named
  mechanism, a clinical detail, a landmark experiment, a cross-species
  comparison, a developmental timing, an evolutionary origin, a disease basis —
  cycled by index so the set cannot collapse onto one template. Indexed rather
  than randomised, so a seed still reproduces its run exactly.
- **Near-duplicate rejection** at cosine ≥ 0.88 against already-accepted
  questions, evaluated *before* the grader call so a duplicate costs no quota.

The prompt also now states that a question which telegraphs its own
unanswerability through stiff phrasing is useless, because it can be refused on
style alone without reading the textbook.

**The general lesson is worth keeping:** an automated filter validates what it
was told to look for, and is perfectly happy for everything it was not told
about to go wrong. The acceptance count looked healthy in both runs. Only
reading the questions revealed that one set was unusable.

A person still approves every one, because the grader is a language model and
can be wrong. What changed is the size of that job — approving twelve verified
candidates instead of authoring twelve from scratch. That trade is worth making
on its own terms: **a review small enough to actually happen is worth more than
a more rigorous one that gets skipped**, and an unreviewed gold set is worth
nothing at all.

### What is deliberately *not* checked

Triage never drops a question because retrieval failed to find its gold chunk.
That would be measuring the system with the instrument meant to calibrate it:
removing the questions the pipeline currently misses leaves a gold set the
baseline already succeeds on, and every later "improvement" would then be scored
against a target quietly moved to meet it. A gold chunk ranked nowhere is a
result, not a defect.

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
