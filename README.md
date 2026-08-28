<div align="center">

# VidyaRAG

**A study assistant that checks its own work — and admits when the textbook doesn't have the answer.**

Agentic, self-correcting RAG over open-license OpenStax biology textbooks, with
page-level citations, a claim-level groundedness check, and an evaluation
harness that measures whether any of it actually helped.

[![CI](https://github.com/NehaBharti08/VidyaRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/NehaBharti08/VidyaRAG/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Corpus: CC BY 4.0](https://img.shields.io/badge/corpus-CC%20BY%204.0-lightgrey.svg)](ATTRIBUTION.md)

</div>

> **Status: Phase 4 — retrieval quality.** Ingestion, baseline pipeline,
> evaluation harness and the retrieval ablations are done and measured. The
> corrective self-check (Phase 5) is next, and it is the one that has to move
> abstention off 0.000. Sections marked _pending_ are filled in as the work
> behind them lands; no number appears here before it has been measured.

---

## The problem

Ask a general-purpose chatbot a textbook question and you get a fluent answer
with no way to tell whether it came from the book, from the model's memory, or
from nowhere at all. For a student revising for an exam, a confident wrong
answer is worse than no answer.

VidyaRAG is built around one idea: **a RAG system should be able to prove its
answer is in the source, and abstain when it isn't.**

- **Page-level citations.** Every claim resolves to `Biology, §7.3, p.214` —
  a real page in a real section, verified by a test that fails if a citation
  points nowhere.
- **A corrective self-check.** A grader scores each atomic claim in the draft
  answer against the retrieved evidence. Unsupported claims trigger a bounded
  re-retrieval; if the evidence still isn't there, the system says so instead of
  inventing it.
- **Numbers, not adjectives.** Every enhancement is ablated independently
  against a frozen baseline on a 58-question gold set. Results that didn't help
  are reported too — query decomposition is in the table below because it made
  things worse.

---

## Live demo

_Pending — deploying to Hugging Face Spaces._

---

## Architecture

_Pending (Phase 7) — Mermaid diagram of the ingestion and query paths._

---

## Results

Every configuration is measured against the same 58-question gold set. `baseline`
is a frozen dense-retrieval control, never edited after Phase 2, so all deltas
are directly comparable.

**58 questions, 0 failures.** Each component ablated separately against the
frozen baseline.

| | baseline | + rerank | + decompose |
|---|---:|---:|---:|
| Faithfulness | 0.948 | 0.950 | 0.951 |
| Answer relevancy | 0.755 | 0.770 | 0.709 |
| Context precision | 0.732 | **0.792** | 0.690 |
| Context recall | 0.938 | 0.946 | 0.873 |
| Recall @k | 0.967 | 0.967 | 0.957 |
| Recall @context | 0.880 | **0.913** | 0.826 |
| MRR | 0.770 | **0.830** | 0.769 |
| Abstention recall | 0.000 | 0.000 | 0.000 |
| Mean latency | 1,091 ms | 6,856 ms | 2,635 ms |
| Cost/query (list price) | $0.00026 | $0.00027 | $0.00026 |

**Reranking ships.** MRR +0.060 and recall @context +0.033, at 6.3× latency. Recall @k
and hit rate are unchanged to three decimals — that is the ablation's own control,
since reordering a pool must not change what is in it.

**Decomposition does not.** It was built to fix multi-hop retrieval and made it
worse: recall @context on split multi-hop questions fell from 0.864 to 0.727. The
cause is not what I expected — only *one* gold passage was lost from the candidate
pool, so fusion is not failing to retrieve. Reciprocal Rank Fusion ranks by
agreement between sub-questions, and for a genuine two-hop question the passage
each hop needs is retrieved by that hop alone, while generic passages both hops
surface score twice and win. **Consensus selects for the unspecific.** Full
analysis in [docs/EVALUATION.md](docs/EVALUATION.md).

**Abstention recall is 0.000 everywhere.** No component in Phase 4 can decline to
answer; that is Phase 5's job, and this is the number it has to move.

Full methodology, gold-set provenance, and the failures that shaped the harness
are in [docs/EVALUATION.md](docs/EVALUATION.md). Every run is a committed JSON
file under [`eval/results/`](eval/results/); a number here that cannot be traced
to one of those files is not evidence.

---

## Quickstart

Requires Python 3.11+. No Docker, no vector-store account, and **no API key to
build the index** — embeddings run locally on CPU and the default configuration
uses an in-process vector store. A free Gemini key is needed only to *answer*
questions.

```bash
git clone https://github.com/NehaBharti08/VidyaRAG.git
cd VidyaRAG

# Install uv if you don't have it: https://docs.astral.sh/uv/
uv sync

cp .env.example .env
# Optional for ingestion; needed to answer questions.
# Free key, no credit card: https://aistudio.google.com/apikey

uv run vidyarag health          # validate config and store connectivity
uv run vidyarag download        # fetch the OpenStax PDFs (~415 MB, resumable)
uv run vidyarag ingest          # parse, chunk, embed, index (~1 hr, offline)

# Answering needs a free Gemini key in .env
uv run vidyarag ask "How does facilitated diffusion move glucose into a cell?"
uv run uvicorn vidyarag.api.main:app     # then POST /v1/query
```

A real answer, abridged:

```
Because glucose is both large and polar, it cannot cross the cell membrane's
lipid bilayer through simple diffusion [1, 2]. Instead it enters via
facilitated diffusion, moving down its concentration gradient with the help of
selective carrier proteins [1, 3, 5]...

[1] Anatomy and Physiology, 3.1. The Cell Membrane, p.92 (CC BY 4.0)
[3] Biology, 5.2. Passive Transport, p.147 (CC BY 4.0)

10437ms [retrieve=3163ms generate=7274ms] 2553+227 tok
```

Page numbers are the **printed** ones, so they can be checked against a paper
copy — not the PDF page index, which differs by 12 in Biology.

`health` validates configuration, resolves the pipeline profile, and checks
vector store connectivity. It exits non-zero on failure, so CI and the
container healthcheck both use it.

```bash
uv run vidyarag config --profile baseline   # inspect a pipeline variant
uv run pytest                                # run the test suite
```

```bash
uv run vidyarag eval --profile rerank        # run an ablation against the gold set
uv run vidyarag report                       # compare committed runs
```

---

## How it is put together

| Concern | Choice | Why |
|---|---|---|
| Orchestration | LlamaIndex | Structure-aware node metadata survives retrieval, which is what makes citations real rather than decorative. |
| Vector store | Qdrant | One `QdrantClient` API covers in-process, local server, and cloud — see [`store/client.py`](src/vidyarag/store/client.py). |
| Embeddings | `BAAI/bge-base-en-v1.5` via fastembed | ONNX on CPU, **no torch, no API key**. Retrieval therefore has no external dependency at all — the published demo cannot be broken by an expired account. |
| Generation | `gemini-3.5-flash-lite` (grading: `gemini-3.1-flash-lite`) | Pinned, never aliased: `-latest` would change model underneath a benchmark and make every reported delta incomparable. Generator and grader are deliberately different models — a model grading its own output rates it favourably. |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` via fastembed | ONNX, 80 MB, **no torch** — keeps the deployed image near 400 MB instead of ~2.5 GB. |
| Evaluation | RAGAS | Faithfulness, answer relevancy, context precision, context recall. |

Full rationale, including the alternatives rejected and why, is in
[docs/DESIGN.md](docs/DESIGN.md). Evaluation methodology and per-phase deltas
are in [docs/EVALUATION.md](docs/EVALUATION.md).

### Configuration model

Deployment concerns (secrets, endpoints) live in the environment. Pipeline
behaviour lives in committed YAML profiles under [`config/profiles/`](config/profiles/),
so a benchmark result can be traced to an exact, diffable configuration.
Unknown keys are rejected at load time — a typo fails the run instead of
silently invalidating it.

---

## Limitations

_Expanded as the system is measured. Known now:_

- The corpus is two introductory biology textbooks. Questions outside that
  scope should be abstained on, not answered — that behaviour is measured, not
  assumed.
- Answers are generated by a language model and can be wrong even when
  well-grounded. This is a study aid, not a reference.
- The embedded index mode uses a linear scan and is appropriate for this
  corpus size (~6k chunks), not for arbitrary scale.

---

## Corpus & licensing

The code is MIT licensed. The textbooks are **not** — they are published by
OpenStax under CC BY 4.0, and that license travels with every retrieved
passage and generated citation.

See [ATTRIBUTION.md](ATTRIBUTION.md) for per-title licensing, required
attribution, and why these particular titles were chosen.

> Download Biology for free at https://openstax.org/details/books/biology
> Download Anatomy and Physiology for free at https://openstax.org/details/books/anatomy-and-physiology

VidyaRAG is an independent student project and is not affiliated with or
endorsed by OpenStax or Rice University.
