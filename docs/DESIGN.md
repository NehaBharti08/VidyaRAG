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
(`gemini-3.5-flash` / `gemini-3.5-flash-lite`).

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

### Formatting: black only

ruff lints; black formats. `ruff-format` is deliberately disabled — the two
disagree on edge cases and would rewrite each other's output on every commit.

---

## Pending

Recorded as each phase lands: chunking strategy and the structure extraction
approach (Phase 1), prompt and citation format (Phase 2), retrieval ablations
(Phase 4), grader prompt design and abstention threshold tuning (Phase 5),
injection threat model (Phase 6).
