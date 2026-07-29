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

### Corpus: Biology 2e + Microbiology

Both CC BY 4.0, verified per title (see [ATTRIBUTION.md](../ATTRIBUTION.md)).

*Rejected: Concepts of Biology.* Also CC BY 4.0 and an obvious third candidate,
but it overlaps Biology 2e substantially. Indexing both would let the same fact
be retrieved from two titles, which makes context-precision scores ambiguous —
a chunk would score as "irrelevant" only because a near-duplicate outranked it.
Corpus redundancy would have been measured as retrieval error.

Biology 2e and Microbiology are complementary, which also makes genuine
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
