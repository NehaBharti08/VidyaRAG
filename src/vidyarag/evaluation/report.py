"""Render an evaluation run as markdown.

The report is written to be read by someone who is sceptical. It leads with the
configuration that produced the numbers, shows how many samples each average is
actually over, and prints failures rather than hiding them behind a mean.

Deltas against the frozen baseline are shown with their direction, and never
described as an improvement without the number beside them.
"""

from __future__ import annotations

from vidyarag.evaluation.metrics import METRIC_NAMES
from vidyarag.evaluation.runner import EvalRun

_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}

_RETRIEVAL_LABELS = {
    "hit_rate": "Hit rate @k",
    "recall_at_k": "Recall @k",
    "context_recall_at_context": "Recall @context",
    "reciprocal_rank": "MRR",
}


def _fmt(value: float | int | None, places: int = 3) -> str:
    """Format a metric, distinguishing "not measured" from zero."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.{places}f}"


def _one_line(message: str, limit: int = 140) -> str:
    """Collapse an exception to one readable line.

    Instructor wraps retry failures in a multi-line XML-ish blob; pasted
    verbatim it buries the actual cause under a screen of markup.
    """
    collapsed = " ".join(message.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def _delta(current: float | None, baseline: float | None) -> str:
    """Signed change against the baseline, or an em dash when incomparable."""
    if current is None or baseline is None:
        return "—"
    change = current - baseline
    if abs(change) < 5e-4:
        return "±0.000"
    return f"{change:+.3f}"


def render_report(run: EvalRun, baseline: EvalRun | None = None) -> str:
    """Render a full markdown report for one run.

    Args:
        run: The run to report.
        baseline: The frozen baseline to compare against. Omitted when the run
            *is* the baseline.
    """
    lines: list[str] = []
    add = lines.append

    add(f"# Evaluation: `{run.profile}`")
    add("")
    add(f"**Run** `{run.run_id}` · {run.created_at}")
    add("")

    if not run.is_valid:
        # Lead with this, and omit every metric below. A number that must not
        # be used should not be sitting in a tidy table waiting to be copied
        # into a README by someone who skimmed past a warning.
        by_type = run.failures_by_type()
        add("## ⛔ INVALID RUN — no metrics reported")
        add("")
        add(
            f"**{len(run.failed)} of {len(run.samples)} questions "
            f"({run.failure_rate:.0%}) never produced an answer**, so any average "
            "would describe whichever subset happened to survive."
        )
        add("")
        add("Questions lost, by type:")
        add("")
        add("| Type | Failed |")
        add("|---|---:|")
        for kind, count in sorted(by_type.items()):
            add(f"| {kind} | {count} |")
        add("")
        add(
            "That distribution is the reason metrics are withheld rather than "
            "merely flagged. Failures are not spread evenly: the gold set is "
            "ordered by type, so an interrupted run loses whole categories. "
            "Averaging over what remains produces a confident measurement of an "
            "easier task, not a noisier measurement of the intended one."
        )
        add("")
        if run.failed:
            add(f"First failure: `{_one_line(str(run.failed[0].error))}`")
            add("")
        add("Re-run when the cause is resolved. Cached answers make it cheap.")
        add("")
        return "\n".join(lines) + "\n"

    # --- Provenance --------------------------------------------------------
    add("## Configuration")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Profile | `{run.profile}` |")
    add(f"| Generation model | `{run.generation_model}` |")
    add(f"| Grader model | `{run.grader_model}` |")
    add(f"| Embedding model | `{run.embedding_model}` |")
    retrieval = run.config.get("retrieval", {})
    add(
        f"| Retrieval | top_k {retrieval.get('top_k_retrieve')} → "
        f"context {retrieval.get('top_k_context')} |"
    )
    add(
        f"| Hybrid / rerank / decompose | "
        f"{retrieval.get('use_hybrid')} / {retrieval.get('use_reranker')} / "
        f"{retrieval.get('use_decomposition')} |"
    )
    add(f"| Corrective loop | {run.config.get('corrective', {}).get('enabled')} |")
    add(f"| Gold set | `{run.goldset_path}` (sha256 `{run.goldset_sha256}`) |")
    counts = ", ".join(f"{k}={v}" for k, v in run.goldset_counts.items() if v)
    add(f"| Composition | {counts} |")
    add("")

    if baseline is not None and baseline.goldset_sha256 != run.goldset_sha256:
        add(
            "> **Warning.** This run and the baseline used different gold sets "
            f"(`{run.goldset_sha256}` vs `{baseline.goldset_sha256}`). "
            "The deltas below are not comparable."
        )
        add("")

    # --- Generation quality ------------------------------------------------
    graded = int(run.aggregates.get("graded_samples") or 0)
    add("## RAGAS metrics")
    add("")
    add(
        f"Averaged over **{graded} graded samples** — answerable questions the "
        "system attempted. Refusals and unanswerable questions are excluded, "
        "since a refusal has no faithfulness to measure; they are scored under "
        "Abstention below."
    )
    add("")

    if baseline is not None:
        add("| Metric | Baseline | This run | Δ |")
        add("|---|---:|---:|---:|")
        for name in METRIC_NAMES:
            add(
                f"| {_METRIC_LABELS[name]} | {_fmt(baseline.aggregates.get(name))} "
                f"| {_fmt(run.aggregates.get(name))} "
                f"| {_delta(run.aggregates.get(name), baseline.aggregates.get(name))} |"
            )
    else:
        add("| Metric | Score |")
        add("|---|---:|")
        for name in METRIC_NAMES:
            add(f"| {_METRIC_LABELS[name]} | {_fmt(run.aggregates.get(name))} |")
    add("")

    # --- Retrieval ---------------------------------------------------------
    add("## Retrieval quality")
    add("")
    add(
        "Computed from chunk ids against the gold set. No model is involved, so "
        "these are exactly reproducible and cannot be distorted by a grader "
        "disagreeing about relevance."
    )
    add("")
    if baseline is not None:
        add("| Metric | Baseline | This run | Δ |")
        add("|---|---:|---:|---:|")
        for key, label in _RETRIEVAL_LABELS.items():
            add(
                f"| {label} | {_fmt(baseline.retrieval_aggregates.get(key))} "
                f"| {_fmt(run.retrieval_aggregates.get(key))} "
                f"| {_delta(run.retrieval_aggregates.get(key), baseline.retrieval_aggregates.get(key))} |"
            )
    else:
        add("| Metric | Score |")
        add("|---|---:|")
        for key, label in _RETRIEVAL_LABELS.items():
            add(f"| {label} | {_fmt(run.retrieval_aggregates.get(key))} |")
    add("")
    add(
        "*Recall @k* covers the whole candidate pool; *Recall @context* covers "
        "only the chunks that reached the prompt. The gap between them is what "
        "reranking exists to close."
    )
    add("")

    # --- Abstention --------------------------------------------------------
    a = run.abstention
    add("## Abstention")
    add("")
    add("| Metric | Value |")
    add("|---|---:|")
    add(f"| Unanswerable questions | {a.get('unanswerable_total')} |")
    add(f"| …correctly refused | {a.get('unanswerable_abstained')} |")
    add(f"| Abstention recall | {_fmt(a.get('recall'))} |")
    add(f"| Abstention precision | {_fmt(a.get('precision'))} |")
    add(f"| Abstention F1 | {_fmt(a.get('f1'))} |")
    add(f"| False abstention rate | {_fmt(a.get('false_abstention_rate'))} |")
    add("")
    add(
        "False abstention rate is reported beside precision deliberately: a "
        "system that refused everything would score perfect precision, and only "
        "this column would reveal it."
    )
    add("")

    # --- Cost and latency --------------------------------------------------
    t = run.totals
    add("## Cost and latency")
    add("")
    add("| | |")
    add("|---|---:|")
    add(f"| Questions | {int(t.get('questions', 0))} |")
    add(f"| Failed | {int(t.get('failed', 0))} |")
    add(f"| Mean latency | {t.get('mean_latency_ms', 0):.0f} ms |")
    add(f"| Input tokens | {int(t.get('input_tokens', 0)):,} |")
    add(f"| Output tokens | {int(t.get('output_tokens', 0)):,} |")
    add(f"| List-price cost | ${t.get('list_price_usd', 0):.4f} |")
    questions = t.get("questions", 0) or 1
    add(f"| List price per query | ${t.get('list_price_usd', 0) / questions:.5f} |")
    add("")
    add(
        "Actual spend is **$0** — generation runs on Gemini's free tier and "
        "embeddings run locally. The figures above are what this traffic would "
        "cost at published rates, which is the number that matters for judging "
        "whether the design is economical to run for real."
    )
    add("")

    # --- Failures ----------------------------------------------------------
    failures = [s for s in run.samples if s.error]
    grader_errors = [s for s in run.samples if s.ragas_errors]
    if failures or grader_errors:
        add("## Failures")
        add("")
        for sample in failures:
            add(f"- **{sample.id}** answering failed — `{sample.error}`")
        for sample in grader_errors:
            for metric, message in sample.ragas_errors.items():
                add(f"- **{sample.id}** `{metric}` — `{_one_line(message)}`")
        add("")
        if any("429" in m for s in grader_errors for m in s.ragas_errors.values()):
            add(
                "> Rate-limit failures mean the affected metrics were **not measured**. "
                "They are excluded from the averages above rather than counted as zero, "
                "but a run with many of them is averaging over fewer samples than it "
                "appears to. Re-run with a lower `--rate` before trusting it."
            )
            add("")

    return "\n".join(lines)


def render_comparison(runs: list[EvalRun]) -> str:
    """Render one table comparing several profiles.

    This is the table the README carries. Profiles appear in the order given,
    so the baseline should come first.
    """
    if not runs:
        return "_No runs to compare._"

    lines: list[str] = []
    add = lines.append

    header = "| Metric | " + " | ".join(f"`{r.profile}`" for r in runs) + " |"
    add(header)
    add("|---" * (len(runs) + 1) + "|")

    for name in METRIC_NAMES:
        values = " | ".join(_fmt(r.aggregates.get(name)) for r in runs)
        add(f"| {_METRIC_LABELS[name]} | {values} |")
    for key, label in _RETRIEVAL_LABELS.items():
        values = " | ".join(_fmt(r.retrieval_aggregates.get(key)) for r in runs)
        add(f"| {label} | {values} |")
    add(
        "| Abstention recall | " + " | ".join(_fmt(r.abstention.get("recall")) for r in runs) + " |"
    )
    add(
        "| False abstention | "
        + " | ".join(_fmt(r.abstention.get("false_abstention_rate")) for r in runs)
        + " |"
    )
    add(
        "| List price / query | "
        + " | ".join(
            f"${(r.totals.get('list_price_usd', 0) / (r.totals.get('questions') or 1)):.5f}"
            for r in runs
        )
        + " |"
    )
    return "\n".join(lines)
