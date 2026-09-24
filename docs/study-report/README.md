# VidyaRAG: a complete technical study

**[Read the PDF](VidyaRAG_Study_Report.pdf)** — 257 pages.

A full analysis of this repository, written from the code rather than from the
documentation: what every part does, how it was measured, how it is deployed,
where it is weak, and what the evaluation got wrong about itself.

It is the document that found the measurement bug described in the README's
Results section. Appendix H records the corrections that followed.

## How to read it

Every statement carries one label, and they are never mixed inside a sentence:

| Label | Means |
|---|---|
| **Verified** | Read directly out of the code, a config file, a test or a committed result file. |
| **Measured in this analysis** | Produced by running the system during the analysis, on the machine and date stated. |
| **Inferred** | Reasoned from the code, but not stated anywhere and not measured. |
| **Recommended** | Not implemented. A suggestion, marked as one. |

Claims about a file cite it as `[src/vidyarag/pipeline.py:155-171]`, at commit
`fa5164b`. Line numbers drift; the file and symbol names are the stable part.

**Start with "How to Read This Report", then Appendix H.** The report describes
commit `fa5164b`, and its headline evaluation numbers were later corrected —
Appendix H carries the corrected figures and the status of all 32 findings, so
nothing in Part III gets quoted after it has been retracted.

## Structure

| Part | Chapters | What it covers |
|---|---|---|
| I | 1–3 | The project in plain language, the concepts, the architecture |
| II | 4–16 | Repository, environment, dependencies, config, ingestion, retrieval, generation, the self-check, guardrails, observability, interfaces, a real end-to-end trace, master flows |
| III | 17–21 | Evaluation harness, results, testing, performance, security |
| IV | 22–24 | Deployment, local setup, reproducibility |
| V | 25–30 | Development history, the findings register, trade-offs, limitations, improvements, a rebuilding guide |
| VI | 31–34 | Explaining the project at four depths, design rationale, 68 interview questions, and a claims audit |
| Appendices | A–H | Glossary, commands, config reference, file and function map, data tables, biology background, self-audit, and the fixes applied afterwards |

## Building it

Needs XeLaTeX (for `fontspec`). [Tectonic](https://tectonic-typesetting.github.io/)
fetches what it needs on first run:

```bash
cd docs/study-report
tectonic -X compile --outdir build main.tex
```

Any XeLaTeX toolchain works too; run it twice so the cross-references settle.
The committed `VidyaRAG_Study_Report.pdf` is the output of that command — copy
a rebuilt `build/main.pdf` over it to update the published version.

`OUTLINE.md` is the working plan the chapters were written against.
