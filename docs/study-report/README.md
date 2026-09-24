# VidyaRAG: a technical study

**[Read the PDF](VidyaRAG_Study_Report.pdf)** — 70 pages.

A full analysis of this repository, written from the code rather than from the
documentation: what every part does, how it was measured, how it ships, where
it is weak, what its evaluation got wrong about itself, and how to explain all
of it in an interview.

It began as an audit of commit `fa5164b`. That audit found 32 defects, one of
which invalidated the project's headline result, and the fixes that followed
(PR #20) are why the report can state corrected numbers as current fact.
Chapter 15 tells that story.

## How to read it

Every statement carries one label, never mixed inside a sentence:

| Label | Means |
|---|---|
| **Verified** | Read out of the code, a config file, a test, a committed result file or git history. |
| **Measured** | Produced by running the system while writing the report; not in the repository. |
| **Inferred** | Reasoned from the code, but neither stated nor measured. |
| **Recommended** | Not implemented. A suggestion, marked as one. |

Claims about the code cite the file they come from, for example
`[src/vidyarag/pipeline.py]`.

## Structure

| Chapters | Covers |
|---|---|
| 1–3 | The project in plain language, the concepts with their maths, the architecture |
| 4–11 | Stack and configuration, ingestion, retrieval, generation, the self-check, guardrails, observability and interfaces, one question traced end to end |
| 12–14 | Evaluation and the corrected results; testing, performance and security; shipping and reproducibility |
| 15–16 | What went wrong and what was fixed, all 32 findings with status; trade-offs, limitations, next steps |
| 17 | Interview preparation: four-length explanations, "why X" answers, a question bank, résumé claims |
| A–B | Glossary; commands, configuration and key files |

## How it was checked

Every file reference resolves to a real file, and every cell of the results
table recomputes from the committed run files in `eval/results/`. The longer
257-page audit this was condensed from is in git history at commit `41ebf4d`.

## Building it

Needs XeLaTeX (for `fontspec`); [Tectonic](https://tectonic-typesetting.github.io/)
fetches what it needs on first run:

```bash
cd docs/study-report
tectonic -X compile --outdir build main.tex
```

Copy `build/main.pdf` over `VidyaRAG_Study_Report.pdf` to update the published
version.
