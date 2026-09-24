# VidyaRAG Study Report — outline (adapted to what the repository actually contains)

Snapshot analysed: repository NehaBharti08/VidyaRAG at commit fa5164b (package version 1.2.1), 2026-09-14.
Evidence notes: scratchpad/analysis/01..08 (+ rejudge results).

Conventions: default prose = Verified (with file/line or run-file reference); Inferred and Recommended always in labelled boxes; "Not determinable from the repository." used verbatim where applicable. Findings discovered during this analysis in red "Finding" boxes.

No chapters for: frontend frameworks (only a Gradio Blocks UI), databases (Qdrant vector store only), model training (none — pretrained models + prompting), authentication (none), GPU (none), microservices (none).

## Front matter
- Title page; How to read this report (evidence labels, notation, what "the repository" means, snapshot); Table of contents.

## Part I — Understanding the project
1. The project in plain language
2. The ideas you need first (from first principles): LLM, hallucination, tokens/tokeniser, embeddings, cosine similarity, vector database, RAG, chunking, bi-encoder vs cross-encoder, reranking, reciprocal rank fusion, groundedness/faithfulness, abstention, prompt injection, evaluation metrics (recall@k, MRR, precision/recall/F1), LLM-as-judge.
3. Architecture: ingestion time, query time, evaluation time, deployment; component catalogue (what/why/input/output/talks to/where).

## Part II — How the code works
4. Repository structure (tree + every important file)
5. Development environment and technologies (Python 3.10–3.12, uv, hatchling, Qdrant, fastembed/ONNX Runtime, BGE, MiniLM cross-encoder, Gemini/google-genai, OpenAI-compatible endpoint, RAGAS, FastAPI/Uvicorn, Gradio, Pydantic/pydantic-settings, PyMuPDF, httpx, Typer/Rich, structlog, pytest/respx, ruff/black/mypy, pre-commit/gitleaks, Docker, GitHub Actions, Hugging Face Spaces ZeroGPU)
6. Dependencies (runtime/dev/eval; core vs incidental; pins and why)
7. Configuration system (Settings vs PipelineConfig; profiles; deep merge; extra=forbid; config-dir resolution)
8. Ingestion: PDFs to searchable index (corpus registry, download: idempotent/resumable/atomic; parse: outline structure map, header/footer bands, column detection, cleaning, end-matter; chunk: sentence splitting, greedy packing with overlap, token counting; embed; store: uuid5 idempotent ids, payload, named vector; orchestrator; data structures)
9. Retrieval: dense search, cross-encoder reranking, query decomposition + RRF (math, code, params, alternatives)
10. Answer generation and citation validation
11. The corrective self-check loop (grader, claim verdicts, score, policy table, reformulation, loop; refusal flag)
12. Prompt-injection guardrails (threat model, regex families, input vs context, false-positive engineering, reproduction)
13. Observability: tracing, cost accounting and logging (QueryTrace, stages, pricing formula, structlog)
14. Interfaces: HTTP API, command-line tool, Gradio demo
15. A question's journey: end-to-end execution trace (real numbers from live runs)
16. Master flows: data flow, execution flow, startup flow, failure flow

## Part III — Evidence
17. The evaluation harness (gold set: schema, provenance, drafting, verification; runner two phases; caches; rate limiter; RAGAS + Gemini workarounds; deterministic retrieval metrics; abstention metrics; validity rule; reports)
18. Experiments and results (baseline, rerank, decompose, corrective, guarded; per-type; noise; interpretation; corrected numbers from this analysis)
19. Testing (strategy; every test file; coverage by module; what is not tested)
20. Performance (latency evidence incl. cold start and double counting; reranker cost; tokens; memory/image/index size; bottlenecks)
21. Security (implemented / not implemented / recommended)

## Part IV — Shipping
22. Build, packaging and deployment (Dockerfile, compose, CI workflows, HF Space deploy script, runtime env)
23. Complete local setup on a fresh machine
24. Reproducing every workflow

## Part V — Critical analysis
25. Problems met during development and how they were fixed (from git history + code comments)
26. What this analysis found: contradictions and latent bugs (judge truncation bug; latency double counting; token undercount; ignored book_slug; stale docs; etc.)
27. Design decisions and trade-offs
28. Limitations
29. Improvements (prioritised; complexity)
30. Rebuilding VidyaRAG yourself (stages)

## Part VI — Interview preparation
31. Explaining the project (30 s, 1 min, 3 min, deep technical)
32. "Why did you use this?" (+ why not the alternative)
33. Interview question bank (by category, with answers)
34. What I must know before putting this project on my resume (checklist + most important questions)

## Appendices
A. Glossary
B. Command reference
C. Configuration and environment variable reference
D. File and function map
E. Data tables (gold set composition, per-profile per-type metrics, corrective decisions)
F. Domain primer: the biology behind the demo questions (general knowledge, flagged as not from the repo)
G. Self-audit against the brief
