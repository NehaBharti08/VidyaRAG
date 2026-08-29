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

## Run validity

A run is only quotable if it actually answered the questions. Above a **10%
failure rate the harness reports no metrics at all** — not a warning beside the
numbers, but the numbers withheld.

That rule was written after a specific incident, recorded here because the
failure mode is easy to walk into and hard to notice.

The first full baseline attempt lost **39 of 58 questions** to Gemini quota
exhaustion. It printed a clean table:

| Metric | Reported |
|---|---:|
| Faithfulness | 0.949 |
| Context recall | 0.941 |
| Hit rate | 0.941 |

Those numbers are excellent, and worthless. The gold set is ordered factual →
multi-hop → unanswerable, so the run died partway through and the survivors
were **17 factual, 0 multi-hop, and 2 unanswerable**. Faithfulness read 0.949
because every hard question was missing — the score was high *as a direct
consequence of the failure*.

The lesson generalises past this project: **a partial run is not a noisier
measurement of the intended task, it is a confident measurement of an easier
one.** Nothing about the output looked wrong. There was a `WARN` line beneath
the table, and a table is far easier to copy than a warning is to heed.

Two changes followed:

- Metrics are withheld entirely when the failure rate exceeds the threshold,
  and the report leads with which *categories* were lost, since losing every
  multi-hop question is not a smaller version of losing a tenth of each.
- Generated answers are now cached by configuration and question, so a run
  stopped by a daily quota resumes rather than restarting. Previously the 19
  successful answers were discarded and the next attempt spent the same quota
  recomputing them — which on a free tier is the difference between a benchmark
  completable across two days and one not completable at all.

## Measurement noise, and which deltas are claimable

Running the **same profile twice on the same gold set** produced this:

| Metric | Run 1 | Run 2 | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.954 | 0.948 | −0.006 |
| Answer relevancy | 0.798 | 0.755 | **−0.043** |
| Context precision | 0.732 | 0.732 | 0.000 |
| Context recall | 0.938 | 0.938 | 0.000 |
| Recall @k | 0.967 | 0.967 | 0.000 |
| Recall @context | 0.880 | 0.880 | 0.000 |
| MRR | 0.770 | 0.770 | 0.000 |

The split is not random, and it is the useful part. **Every metric whose inputs
exclude the generated answer is bit-identical. Every metric that reads the
answer moved.**

- Retrieval metrics are pure functions of chunk ids. Embedding and search are
  deterministic, so these cannot drift.
- Context precision and context recall are judged against the question,
  the retrieved passages, and the reference answer — never the generated one.
  Same inputs, cached grading, identical scores.
- Faithfulness and answer relevancy read the generated answer, and the
  generated answer is not stable.

### `temperature: 0.0` is not determinism

Verified directly rather than assumed: three identical calls to
`gemini-3.5-flash-lite` at temperature 0.0 with the same prompt returned three
different texts (315, 300 and 307 characters). Temperature zero makes sampling
greedy; it does not make a served model reproducible.

### What this means for every delta reported below

**A change smaller than the noise floor is not a result.** On this setup the
observed floor is roughly ±0.006 on faithfulness and ±0.04 on answer relevancy,
from two samples — enough to bound the order of magnitude, not enough to call a
confidence interval.

The concrete consequence: the first rerank ablation showed answer relevancy
−0.013 against baseline, and that was nearly written up as "reranking slightly
hurts relevancy". It is three times *smaller* than the baseline's own run-to-run
spread. It says nothing.

So the ablations below distinguish two kinds of claim:

- **Deterministic metrics** — retrieval, context precision, context recall.
  A delta here is real, because a re-run reproduces it exactly.
- **Answer-dependent metrics** — faithfulness, answer relevancy. Only
  differences comfortably above ±0.05 are treated as signal; smaller ones are
  reported as "within noise" rather than as small effects.

Answers are cached per profile, so a *given* result file is exactly
reproducible. The noise is irreducible only *between* profiles, which is
precisely where the comparisons live.

---

## Results

### Baseline — `20260818T075845Z`

Dense retrieval, no reranking, no corrective loop. 58 questions, **0 failures**,
0 grader errors. `goldset_v1.jsonl` sha256 `258cb6f9b1a2ab04`.

| RAGAS metric | Score |
|---|---:|
| Faithfulness | 0.954 |
| Answer relevancy | 0.798 |
| Context precision | 0.732 |
| Context recall | 0.938 |

Averaged over the 46 answerable questions the system attempted.

| Retrieval metric | Score |
|---|---:|
| Hit rate @k | 0.978 |
| Recall @k | 0.967 |
| Recall @context | 0.880 |
| MRR | 0.770 |

| Abstention | Value |
|---|---:|
| Unanswerable questions | 12 |
| …correctly refused | **0** |
| Abstention recall | **0.000** |
| False abstention rate | 0.000 |

| Cost & latency | |
|---|---:|
| Mean latency | 2,485 ms |
| List price per query | $0.00027 |
| Actual spend | $0 |

### What this baseline says

**Abstention recall is 0.000.** The baseline answered all twelve unanswerable
questions rather than refusing any of them. That is the expected behaviour of a
pipeline with no corrective loop — nothing in it can decline — and it is the
single most useful number in this table, because it makes the project's
headline claim falsifiable. Phase 5 either moves it or it does not.

Note that abstention *precision* is undefined rather than zero: the system never
refused anything, so there is nothing to compute a precision over. Reporting it
as 0.0 would imply refusals were made and were wrong.

**Retrieval already finds the right passage; the prompt often does not get it.**
Recall @k is 0.967 but recall @context is 0.880 — an 8.7-point gap. The gold
chunk is in the retrieved pool for essentially every question, and is then
ranked out of the top 5 before generation for roughly one question in eight.
That gap is precisely what reranking exists to close, so Phase 4 has a
well-defined target rather than a hope.

**Context precision (0.732) is the weakest generation-side metric** while
faithfulness is 0.954. The model is being scrupulous with the evidence it is
given, and a quarter of that evidence is not relevant. This is consistent with
the recall gap above: the context window is being padded with near-misses.

**Faithfulness at 0.954 leaves little headroom**, which is worth saying plainly
before Phase 4 starts. Improvements will have to show up in context precision,
the recall gap, and abstention — not in faithfulness, where there is barely a
twentieth of the scale left to win.

### Phase 4a — cross-encoder reranking

`Xenova/ms-marco-MiniLM-L-6-v2` via fastembed, over the same 20-candidate pool.
One flag differs from `baseline`. 58 questions, 0 failures, both runs.

| Metric | Baseline | Rerank | Δ | |
|---|---:|---:|---:|---|
| Recall @k | 0.967 | 0.967 | 0.000 | *sanity check* |
| Hit rate @k | 0.978 | 0.978 | 0.000 | *sanity check* |
| **MRR** | 0.770 | 0.830 | **+0.060** | real |
| **Recall @context** | 0.880 | 0.913 | **+0.033** | real |
| **Context precision** | 0.732 | 0.792 | **+0.060** | real |
| Context recall | 0.938 | 0.946 | +0.008 | real, small |
| Faithfulness | 0.948 | 0.950 | +0.002 | within noise |
| Answer relevancy | 0.755 | 0.770 | +0.015 | within noise |
| Mean latency | 1,091 ms | 6,856 ms | **+5,765 ms** | |

The first two rows are the ablation's own control. Reranking reorders the pool
without changing it, so pool-level recall and hit rate **must not move**. They
did not, to three decimals. If they had, something other than the reranker had
changed and nothing below would be attributable.

**The reranker demonstrably did work**, which is a separate question from
whether the metric moved:

| | |
|---|---:|
| Chunks reordered per query (of 20) | 17.7 |
| Queries where the top result changed | 55% |
| Queries where ≥1 chunk was promoted into the prompt | 95% |

Recorded because a metric moving is not evidence that the component credited for
it did anything. A reranker that never altered the top 5 could not be the reason
context precision improved, and only this table would show that.

**The gap it was built to close narrowed but did not shut.** Recall @k 0.967
against recall @context 0.880 was an 8.7-point gap; it is now 5.4 points. About
38% of the loss recovered.

#### Where it fails, and why that is the interesting part

Split by question type, the aggregate hides a reversal:

| Question type | Baseline | Rerank | Δ |
|---|---:|---:|---:|
| Factual | 0.893 | 0.964 | **+0.071** |
| Multi-hop | 0.861 | 0.833 | **−0.028** |

**Reranking helps factual questions substantially and makes multi-hop questions
slightly worse.** The aggregate improvement is entirely carried by the factual
slice, which is 28 of the 46 answerable questions.

The mechanism is not mysterious. A cross-encoder scores each passage
independently against the query and has no notion of what the other selected
passages contain. For a factual question there is one right passage and pushing
it up is exactly correct. A multi-hop question needs two *complementary*
passages, and independent scoring promotes whatever most resembles the query —
which tends to be several near-duplicates of the strongest match, crowding out
the second passage the hop actually requires. Precision improves; coverage does
not.

This is the argument for query decomposition rather than a bigger reranker: the
failure is a diversity problem, and a better pointwise scorer cannot fix a
pointwise objective.

#### The cost, stated plainly

Mean latency goes from 1.1 s to 6.9 s — a **6.3× regression**, about 5.8 s of
CPU cross-encoder work over 20 passages. Token cost is unchanged
($0.00026 → $0.00027) because reranking is local.

That is a real price for +0.06 MRR, and it is the number that decides whether
this ships in the deployed demo. Options not yet measured: rerank a shorter pool
(top 10 rather than 20), or a smaller ONNX model. Both are cheaper than
accepting seven-second answers.

### Phase 4b — query decomposition

Built to fix the multi-hop weakness Phase 4a exposed. It made it worse.

| Metric | baseline | decompose | Δ |
|---|---:|---:|---:|
| Recall @k | 0.967 | 0.957 | −0.011 |
| Hit rate @k | 0.978 | 0.957 | −0.022 |
| **Recall @context** | 0.880 | 0.826 | **−0.054** |
| Context precision | 0.732 | 0.690 | −0.042 |
| Context recall | 0.938 | 0.873 | −0.065 |
| MRR | 0.770 | 0.769 | −0.001 |
| Answer relevancy | 0.755 | 0.709 | −0.046 |
| Mean latency | 1,091 ms | 2,635 ms | +1,544 ms |

Split by question type, on the metric it was built to move:

| Recall @context | baseline | decompose |
|---|---:|---:|
| Factual | 0.893 | 0.857 |
| Multi-hop | 0.861 | **0.778** |
| Multi-hop, split only | 0.864 | **0.727** (−0.136) |

#### The ablation has a built-in control

Decomposition declines to split questions it judges atomic, so each run contains
its own control group. Fire rates: **61% of multi-hop, 75% of unanswerable, 21%
of factual**.

The questions it declined scored **bit-identically to baseline** — +0.000 on
every metric, n=29. Every point of damage is attributable to the 17 questions
that were actually split, and none of it to noise or drift.

#### The mechanism is not the obvious one

The natural hypothesis is that fusion drops gold passages out of the candidate
pool. **It does not.** Across all split questions, exactly **one** gold chunk was
lost from the pool and 26 were kept. Recall @k on split questions fell 0.029
while recall @context fell 0.147.

The passage is still retrieved. Reciprocal Rank Fusion ranks it out of the prompt.

RRF combines lists by rank agreement: a chunk that several sub-questions retrieve
outranks one that only a single sub-question found. `decompose.py` argued for
that in as many words — *"a chunk both hops agree on is more likely to be the
bridge between them."*

**The data says the opposite.** For a genuine two-hop question, the passage each
hop needs is by construction retrieved by *that hop only*, so it earns one RRF
contribution. Generic passages that both sub-questions surface earn two, and win.
Consensus selects for the unspecific, and fusion systematically demotes exactly
the passages decomposition exists to find.

That is not a tuning problem. Raising *k*, splitting differently, or retrieving
more per hop all leave the ranking rule that causes it untouched. A fix would
have to abandon rank agreement — reserving prompt slots per sub-question, so each
hop is guaranteed representation regardless of consensus.

**Rejected.** Worse on every deterministic metric including its target, for 2.4×
latency and an extra LLM call per query.

### What Phase 4 ships

`baseline + rerank`.

**Hybrid retrieval was planned for this phase and is not built.** It needs the
corpus re-indexed with sparse vectors, and the rerank ablation already shows
first-stage recall is not the bottleneck: the gold passage is in the pool for
96.7% of questions, and hit rate is 0.978. Spending an index migration to raise a
number already near ceiling — while recall @context sits at 0.913 — would be
optimising the wrong stage. That is a decision from the measurements, not a
shortcut around them.

### Phase 5 — the corrective self-check loop

The number this phase existed to move:

| Abstention | baseline | rerank | corrective |
|---|---:|---:|---:|
| Unanswerable questions | 12 | 12 | 12 |
| …correctly refused | 0 | 0 | **12** |
| **Recall** | 0.000 | 0.000 | **1.000** |
| Precision | — | — | 0.800 |
| F1 | — | — | 0.889 |
| False abstention rate | 0.000 | 0.000 | 0.065 |

**Every unanswerable question is now refused, and none was refused by accident
of a keyword.** Recall alone would be trivial to fake — a system that refused
everything scores 1.000 — so precision and the false abstention rate are
reported beside it. Fifteen questions were refused: the twelve that should have
been, and three that should not.

#### The quality gains are not real, and saying so matters

Read naively, the headline table looks like the loop improved answers:

| | rerank | corrective | apparent Δ |
|---|---:|---:|---:|
| Answer relevancy | 0.770 | 0.844 | +0.074 |
| Faithfulness | 0.950 | 0.959 | +0.009 |

It did not. Abstentions are excluded from RAGAS grading — a refusal has no
faithfulness to measure — so `graded_samples` falls from 46 to 43, and the
three questions removed are among the worst-answered in the set. Comparing the
two profiles on **only the 43 questions both actually answered**:

| | rerank | corrective | real Δ |
|---|---:|---:|---:|
| Answer relevancy | 0.807 | 0.844 | +0.037 |
| Faithfulness | 0.950 | 0.959 | +0.010 |
| Context precision | 0.825 | 0.819 | −0.006 |
| Context recall | 0.981 | 0.981 | 0.000 |

**Half the apparent relevancy gain is the denominator shrinking, and what
remains sits inside the ±0.04 noise floor.** The corrective loop does not make
answers better. It removes answers that should not have been given.

That is worth stating plainly because the naive comparison was available, it
flattered the work, and nothing in the harness would have objected to it.

#### The three "false" abstentions are not false alarms

The three answerable questions it refused — all multi-hop — scored like this
under `rerank`, which answered them:

| | score on those 3 |
|---|---:|
| Faithfulness | 0.952 |
| **Answer relevancy** | **0.238** |
| Context precision | 0.317 |
| **Context recall** | **0.444** |

Against 0.981 context recall across the run. **Retrieval genuinely failed on
those three questions**, and the generator — looking at passages that did not
contain the answer — said so. The loop converted that into an explicit
abstention.

So the 6.5% false abstention rate is not a trigger-happy guard. It is three
retrieval failures the system declined to paper over. Under `rerank` those same
questions produced *well-grounded answers to the wrong material*: faithfulness
0.952 on context that had 0.444 recall. Faithful to the passages, useless to
the student. That failure mode is invisible to faithfulness alone, and it is
exactly what abstention is for.

#### The retry path barely runs

| | |
|---|---:|
| Loop fired | 16 / 58 |
| Abstained | 15 / 58 |
| Needed a second attempt | **1 / 58** |

Of 58 questions, 57 finished in one attempt. **The retry mechanism — query
reformulation from failed claims, re-retrieval, regeneration — fired exactly
once.** In practice this is an abstention gate, not a corrective loop.

The honest reading is that `max_attempts=2` is not currently earning its
complexity, and the claim-level grading that justifies it is being used almost
entirely for its refusal signal rather than its retry signal. Claim decomposition
still pays for itself — it is what produces a defensible groundedness score —
but the loop that consumes it is doing one job, not two.

Left in rather than removed, because one sample is not enough to conclude the
retry never helps, and the cost of an unused branch is latency on a single
query. Recorded here so the next person does not assume it is load-bearing.

#### What it costs

| | baseline | rerank | corrective |
|---|---:|---:|---:|
| Mean latency | 1,091 ms | 6,856 ms | **22,031 ms** |
| Cost per query (list price) | $0.00026 | $0.00027 | $0.00027 |

**20× baseline latency**, driven by the grading call on every query plus
reranking. Actual spend is unchanged because grading runs on the free tier; the
price is entirely time. For a study assistant answering one question at a time
that is defensible. For anything interactive it would not be.

## What did not work

_Pending. This section is expected to be non-empty._
