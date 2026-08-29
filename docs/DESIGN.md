# Design

Why each choice was made, and what was rejected. Written as the decisions are
made rather than reconstructed afterwards, so the reasoning reflects what was
actually known at the time.

---

## Decision log

### Vector store: one client, three targets

`src/vidyarag/store/client.py` is the only module that knows how Qdrant is
reached. Everything downstream receives a `QdrantClient` and cannot tell
whether it is backed by an in-process index, a local container, or Qdrant
Cloud.

| Target | Used for |
|---|---|
| `embedded` (`:memory:`) | Tests — real client class, no server, no network |
| `embedded` (on-disk path) | The published demo |
| `server` | Local development via docker-compose |
| `cloud` | Development and evaluation runs |

**Why the demo ships an embedded index rather than pointing at Qdrant Cloud.**
Free Qdrant Cloud clusters are suspended after roughly one week of inactivity
and deleted after four. This project is a portfolio artifact whose demo link
must survive months of neglect and still work when someone clicks it. A managed
cluster that deletes itself is the wrong dependency for that requirement. An
embedded index has no such failure mode and no network hop.

*Trade-off:* embedded mode uses a linear scan and warns past ~20,000 points.
The corpus is expected near 6,000 chunks, so this is comfortable — but adding
titles is not free, and crossing that threshold means moving the demo to a
server-backed target.

*Rejected:* keeping Qdrant Cloud in the deployed path with a scheduled job
pinging it to prevent idle suspension. It works, but it makes the demo's
liveness depend on a cron job continuing to run correctly and indefinitely —
more moving parts protecting a weaker guarantee.

### Configuration: environment vs. profile

Deployment concerns (secrets, endpoints, log format) come from the environment.
Pipeline behaviour (chunk size, top-k, thresholds, feature flags) comes from
committed YAML profiles.

The split exists for evaluation. A profile is a reproducible, diffable, citable
description of one pipeline variant, so a row in a results table can be traced
to an exact configuration. Secrets must never end up in something committed,
and a benchmark must never depend on an undeclared local environment variable.

Pydantic models use `extra="forbid"`, so a typo in a profile key raises at load
time. The failure mode this prevents is specific and nasty: a misspelled key is
silently ignored, the run completes, and the resulting number is attributed to
a configuration that was never actually applied.

### Models: local embeddings, hosted generation

Embeddings run **locally** (`BAAI/bge-base-en-v1.5` via fastembed: ONNX, CPU,
768-dim, no torch). Generation and grading go to **Gemini**
(`gemini-3.5-flash-lite` / `gemini-3.1-flash-lite`).

*Why split them:* embedding is a fixed, mechanical transformation where a small
open model is close enough to a hosted one to be worth the trade; answering a
question and grading its groundedness are not. Spending the external dependency
only where it buys something keeps the important property intact — **retrieval
has no runtime dependency on any third party.** The deployed demo already ships
a self-contained index so a suspended cloud cluster cannot break it months
later; local embeddings close the last hole in that argument, because
retrieving a passage now requires no account, no key, and no network.

It also makes ingestion runnable by anyone who clones the repo, with no
credentials at all — which is the difference between a reproducible project and
one that merely claims to be.

*Models are pinned, never aliased.* `gemini-flash-latest` would silently change
model underneath a benchmark and make every reported delta incomparable. Note
also that `gemini-2.5-*` returns 404 for accounts created after ~mid-2026
("no longer available to new users"), so the 3.x line is what a new key can
actually call — worth knowing before copying a tutorial that predates it.

*Generator and grader are different models, deliberately.* A model asked to
judge whether its own output is grounded rates it favourably, which would
inflate faithfulness precisely where this project claims to measure it
honestly. The separation costs nothing and removes the objection.

*Why the lite tier for generation.* `gemini-3.5-flash` is the stronger model and
was the original choice, but its free tier allows **20 requests per day** —
measured, not inferred: three calls spaced 65 seconds apart all returned 429,
so it is a daily cap rather than a per-minute one. One 60-question evaluation
needs ~60 generation calls, which is three days of quota for a single run, and
Phase 4's ablations need several runs. The lite models served every call in the
same session. Choosing a slightly weaker model that can actually be *measured*
beats a stronger one that can only be run once a week; an unmeasured improvement
is indistinguishable from no improvement.

*The tokeniser is not interchangeable.* `bge-base-en-v1.5` truncates at 512
tokens and counts BERT WordPiece, roughly 7% more than `cl100k_base` on the same
prose. Chunks sized with tiktoken would run ~548 model tokens and have their
tails silently dropped at embed time — text still shown to the reader and still
named in the citation, contributing nothing to the vector that retrieved it. So
chunking measures with the embedding model's own tokeniser.

*Rejected: OpenAI (`gpt-4o-mini` + `text-embedding-3-small`).* The original
plan, and a fine stack, but it requires funded billing. The free path above
costs nothing to run and produces a *stronger* deployment story, so the paid
one was buying convenience rather than capability.

### Reranker: ONNX cross-encoder, not a torch one

`Xenova/ms-marco-MiniLM-L-6-v2` via fastembed — 80 MB, ONNX runtime, **no torch
dependency**.

*Why:* torch would add roughly 2 GB to the deployed image for a model that runs
for ~130 ms per batch on CPU either way. Avoiding it keeps the container near
400 MB and makes the app deployable on constrained free tiers.

*To be measured (Phase 4):* `BAAI/bge-reranker-base` (1.04 GB, MIT) scores
better on public benchmarks. The ablation reports the delta **on this corpus**.
If it is under a point, shipping the small model is the correct call and the
number justifies it.

*Rejected:* `jinaai/jina-reranker-v2-base-multilingual`. Strong model, but
CC-BY-NC-4.0 — a non-commercial license is wrong for an MIT-licensed repo that
others may build on.

### Corpus: Biology (1st ed) + Anatomy and Physiology (1st ed)

Both CC BY 4.0, verified per title *and per edition* against OpenStax's content
API (see [ATTRIBUTION.md](../ATTRIBUTION.md)).

*Rejected: Biology 2e, Microbiology, Concepts of Biology, A&P 2e.* All four are
CC BY-**NC-SA** 4.0, not CC BY. OpenStax relicensed much of its catalog on the
second edition, so the license cannot be inferred from the title — only the
first editions of *Biology* and *Anatomy and Physiology* remain CC BY, and they
are therefore the complete usable biology corpus rather than a preference among
many. A NonCommercial + ShareAlike corpus would restrict downstream reuse of an
MIT-licensed repo, which is the same objection that rules out the Jina reranker
above; applying it to a model but not to the corpus would be incoherent.

The two chosen titles are complementary — general biology against human organ
systems — with overlap confined to *Biology*'s animal-systems unit. That matters
for measurement, not just coverage: heavily redundant titles make
context-precision ambiguous, because a chunk scores as "irrelevant" only when a
near-duplicate from the other book outranks it, and corpus redundancy then shows
up as retrieval error. The shallow seam that does exist is what makes genuine
cross-title multi-hop questions possible rather than contrived.

### Evaluation: RAGAS, and two upstream bugs

RAGAS is the metric suite, but making it run against Gemini took two
workarounds. Both are pinned and commented at the point of use, because a
future reader will otherwise see an inexplicable dependency and remove it.

**1. `import ragas` fails outright.** ragas 0.4.x imports
`langchain_community.chat_models.vertexai`, which langchain-community deleted
in 0.4.0 (ragas [#2741](https://github.com/vibrantlabsai/ragas/issues/2741),
[#2745](https://github.com/vibrantlabsai/ragas/issues/2745)). Downgrading ragas
does not help — 0.3.9 fails identically, because the break is on the langchain
side. Pinning `langchain-community<0.4` fixes it while keeping the modern
`ragas.metrics.collections` API.

**2. Gemini cannot drive the metrics as shipped.** RAGAS decides whether an LLM
client is async by looking for `chat.completions.create` — an OpenAI shape. A
`google.genai.Client` has no such attribute, so RAGAS concludes "synchronous",
while the collections metrics only expose an async path. Every call raises
`Cannot use agenerate() with a synchronous client`.

The fix is Google's own OpenAI-compatibility endpoint: an `AsyncOpenAI` client
pointed at `generativelanguage.googleapis.com` satisfies the detection and
still calls Gemini. **The `openai` package is a client library here, not a model
provider** — no OpenAI account or spend is involved.

*Rejected: writing the metrics by hand.* Tempting after two bugs, and it would
have removed both. But "we implemented our own faithfulness metric" is far
weaker evidence than a standard one, precisely because a hand-rolled metric is
unfalsifiable by a reader who does not know its internals.

### Evaluation: what is measured without a model

RAGAS context precision and recall are LLM-judged against a reference answer,
so a low score is ambiguous — retrieval may have missed the passage, or the
grader may have disagreed about relevance. Ordinary ranking metrics computed
from chunk ids (recall@k, hit rate, MRR) have no such ambiguity, cost nothing,
never rate-limit, and are exactly reproducible. Both are reported. When they
disagree, the deterministic one says where the fault actually is.

Recall is reported twice: over the whole candidate pool, and over only the
chunks that reached the prompt. A gold chunk retrieved at rank 18 but cut
before generation is a retrieval success and a pipeline failure at once, and
one number cannot show both. Closing that gap is what Phase 4 reranking is for.

Abstention is scored separately from RAGAS, because faithfulness cannot express
"there was no answer to give, and saying so was correct". Precision is always
reported beside the **false abstention rate**: a system that refused everything
would score perfect abstention precision, and only that second column reveals it.

### Gold set: what a model may draft, and what it may not

Factual and multi-hop questions are drafted by a model from sampled passages,
then verified by hand.

Unanswerable questions were originally going to be written by hand outright,
for a real reason: asked to produce questions a corpus cannot answer, a model
reliably produces obviously out-of-domain ones — refusing those is trivial, so
the abstention metric they generate would be meaningless. These are the
questions that prove the differentiating capability, so they are the ones worth
being strict about.

**That reasoning objects to unverified generation, not to generation.** So they
are now machine-proposed and then *mechanically checked* before a person sees
them: a candidate must be in domain, measured as retrieval similarity against
the real index, and judged unanswered by a grader reading the passages that
similarity retrieved. The out-of-domain failure mode is eliminated by the first
check rather than by trusting the prompt. Roughly a third of candidates survive.

A person still approves each one. The change is the size of that job — approving
twelve verified candidates rather than authoring twelve — and a review small
enough to actually happen is worth more than a stricter one that gets skipped.
Provenance is recorded as `llm_drafted_retrieval_verified`, never `human_written`.

Every draft is also asked whether it could be answered from general knowledge
without the passage, and those are dropped. **Roughly 80% of single-passage
candidates are rejected by this check** on an introductory biology corpus,
which is high enough to be worth stating: much of an intro textbook genuinely
is general knowledge, and a gold set full of such questions would show a
healthy score for a system whose retrieval was completely broken.

*Multi-hop pairing is semantic, not random.* The first implementation paired
two randomly sampled chunks, and the output was unusable — asked to connect
skeletal muscle tone to plant water potential, the model produced a question
joining them on "both rely on continuous processes". That is a non sequitur
with a question mark. Pairing each seed passage with its nearest neighbour from
a *different section* yields pairs that share a topic but not a location, which
is what a real multi-hop question needs.

### Formatting: black only

ruff lints; black formats. `ruff-format` is deliberately disabled — the two
disagree on edge cases and would rewrite each other's output on every commit.

---

### Corrective loop: claim-level grading, and one job not two

The grader decomposes a draft answer into atomic claims and scores each against
the retrieved passages, rather than asking "is this answer grounded?" once.

*Why claim level.* An answer-level verdict is unactionable. "Not grounded" tells
the loop nothing about *what* to re-retrieve, so the only available retry is
running the same query again and hoping. Claim-level scoring names the
unsupported claim, and that claim becomes the reformulated query. The retry has
a target.

*Trade-off, and it is real:* an extra call and roughly double the grader tokens
per attempt, and a large part of why the shipped configuration runs at 22 s per
query against the baseline's 1.1 s. Grading is on the free tier, so the cost is
time rather than money — acceptable for a study assistant answering one question
at a time, and not acceptable for anything interactive.

*What the measurement then showed.* The retry path fired **once in 58 queries**.
57 questions finished in a single attempt. In practice the loop is an abstention
gate, not a corrective loop, and `max_attempts=2` is not currently earning its
complexity.

Claim decomposition still pays for itself — it is what produces a defensible
groundedness score and what makes the refusal decision principled. But it is
being used for one of its two jobs. Left in rather than removed, because one
observation is not enough to conclude the retry never helps and an unused branch
costs latency on a single query; recorded so the next reader does not assume it
is load-bearing.

### Abstention: thresholds, and why they were not swept

Accept at a groundedness score >= 0.8, abstain below 0.5, retry between.

These are the values the plan proposed, and they are **not tuned**. A sweep would
mean several full evaluation runs, each around 50 minutes against a rate-limited
free tier, to choose between thresholds on a 58-question set where the twelve
unanswerable questions are the only ones that discriminate. Fitting two decision
boundaries to twelve examples produces a number that looks tuned and generalises
no better than the untuned one — and it would consume the gold set as a
development set, leaving nothing held out.

Measured at the default: abstention recall 1.000, precision 0.800, F1 0.889,
false abstention 0.065. **Recall is at ceiling, so the only direction tuning
could move is precision** — and the three imprecise refusals turn out not to be
threshold errors at all. Those questions had context recall 0.444 against 0.981
across the run: retrieval genuinely failed and the system declined rather than
answering from passages that did not contain the answer. No threshold fixes a
retrieval failure; raising the bar would have produced a confident wrong answer
instead of a refusal.

Documented as untuned rather than presented as chosen. "Default value, measured,
and here is why the obvious tuning would not help" is a defensible position;
implying a sweep that never happened is not.

### Guardrails: two surfaces, one of them not actually under threat

*User input* is genuinely untrusted and is **blocked**, before retrieval. A
question containing "ignore your previous instructions" is not a biology question
with an unfortunate phrase in it, so sanitising it would mean doing an attacker's
editing for them. Screening first also means a blocked question costs nothing —
no embedding, no search, no generation — which matters on a quota an attacker
could otherwise burn for free. Measured: `guard_input=0ms`, and no retrieve stage
in the trace at all.

*Retrieved context* is the attack most RAG demos miss, and **not a live threat to
this system**. The corpus is two OpenStax PDFs fetched over HTTPS and verified by
SHA-256 at ingest; nothing user-supplied reaches the index. The guard exists
because that is a fact about today's configuration rather than a guarantee, and
one added after the first bad document is added too late.

Poisoned passages are **quarantined, not refused**. A bad chunk is a property of
one document, not of the student's question, and the other four passages usually
still answer it. Refusing would let anyone able to write one document deny
service to every question that retrieves it.

*The design constraint was false positives, not recall.* A textbook is full of
imperative prose, and a guard that fires on ordinary pedagogy suppresses correct
answers and trains its maintainer to ignore the alarm. An earlier revision
matched three real chunks — the chapter headings "THE CARDIOVASCULAR SYSTEM:
BLOOD", "...THE HEART" and "...BLOOD VESSELS AND CIRCULATION", where PDF
line-wrapping put `SYSTEM:` at a line start.

So a bare role marker is no longer the signal: it must be followed by directive
language on the same line. An injection that merely asserts a fact under a
`SYSTEM:` label is not caught; one that issues an order is, and the order is what
makes it dangerous. Measured after the change: **0 false positives across all
3,608 chunks**, 5/5 context injections caught, 8/8 input attacks blocked, 0/8
legitimate questions blocked.

### Deployment: an embedded index, and why the demo ships its own data

The published Space carries the built 35 MB index rather than talking to a hosted
vector store.

Free Qdrant Cloud clusters are suspended after roughly a week of inactivity and
deleted after four. A portfolio demo is idle most of the time and is clicked
months after it was last touched — precisely the access pattern that guarantees
the backing store is gone by the time anyone looks. An embedded index has no such
failure mode, no network hop, and no account to expire.

*Trade-off:* embedded mode uses a linear scan and warns past ~20,000 points. At
3,608 chunks this is comfortable, but adding titles is not free, and crossing
that threshold means moving the demo to a server-backed target.

*Rejected:* keeping Qdrant Cloud in the deployed path with a scheduled job
pinging it to prevent idle suspension. It works, and it makes the demo's liveness
depend on a cron job continuing to run correctly and indefinitely — more moving
parts defending a weaker guarantee.

### Docker: no torch, and CI proves it

The image is built without torch. Embeddings and reranking both run through
fastembed on ONNX Runtime, keeping it near 400 MB rather than the ~2.5 GB a torch
stack needs — the difference between something that deploys on a free tier and
something that does not.

The `docker` workflow **fails the build above 900 MB**. That decision is
load-bearing enough to be defended by CI rather than by whoever remembers it,
since the natural way to lose it is a dependency quietly pulling torch back in.

Building in CI is not optional here: Docker Desktop is not installed on the
development machine, and two Dockerfile bugs surfaced only there — a missing
`LICENSE` at build time, and an editable install pointing at a path that does not
exist in the runtime stage. Neither could reproduce locally, where both are
always present.

---

## Known gaps

Stated rather than left for a reader to notice.

- **Hybrid retrieval was planned and is not built.** It needs the corpus
  re-indexed with sparse vectors, and the rerank ablation showed first-stage
  recall is not the bottleneck: the gold passage is in the pool for 96.7% of
  questions. Optimising it would raise a number already near ceiling.
- **Abstention thresholds are untuned defaults**, for the reason above.
- **The retry path is effectively unexercised** — one firing in 58 queries.
- **The noise floor rests on two samples.** Enough to bound the order of
  magnitude, not enough for a confidence interval, and the text says so wherever
  a delta is called noise.
- **58 questions is a small gold set.** Every per-type figure rests on 12–28
  questions, so the deltas are indicative rather than statistically significant.
