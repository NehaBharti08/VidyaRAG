"""Command-line entry point.

Kept deliberately thin: commands resolve configuration, call into the package,
and print. Any logic worth testing belongs in a module, not here.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from vidyarag import __version__
from vidyarag.ingest import CORPUS, download_corpus, get_book
from vidyarag.ingest.pipeline import ingest as run_ingest
from vidyarag.llm.provider import get_gemini_client
from vidyarag.pipeline import Pipeline
from vidyarag.settings import Settings, load_pipeline_config
from vidyarag.store import build_client, describe_target

app = typer.Typer(
    name="vidyarag",
    help="Agentic, self-correcting RAG study assistant over OpenStax textbooks.",
    no_args_is_help=True,
    add_completion=False,
)
# Windows consoles default to cp1252, and redirecting output to a file makes
# that the encoding Python writes with. Rich's spinner glyphs are outside
# cp1252, so `vidyarag eval > log.txt` on Windows died with a UnicodeEncodeError
# after fifty minutes of grading -- the work survived only because it was
# cached. Forcing UTF-8 costs nothing and removes a whole class of failure that
# only ever appears when output is being captured, which is exactly when a long
# run is least likely to be watched.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        with contextlib.suppress(ValueError, OSError):
            _reconfigure(encoding="utf-8", errors="replace")

console = Console()


def _progress_columns() -> tuple[Any, ...]:
    """Progress columns, with the animated spinner dropped when not on a TTY.

    A spinner redrawn into a log file is noise at best; it was also the source
    of the encoding crash above. Piped output gets the same information without
    the animation.
    """
    head: tuple[Any, ...] = (SpinnerColumn(),) if console.is_terminal else ()
    return (
        *head,
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"vidyarag {__version__}")


@app.command()
def config(
    profile: str = typer.Option(None, "--profile", "-p", help="Profile name to inspect."),
) -> None:
    """Show the resolved pipeline configuration for a profile."""
    settings = Settings()
    name = profile or settings.profile
    cfg = load_pipeline_config(name)

    table = Table(title=f"Pipeline profile: {cfg.name}", show_header=True, header_style="bold")
    table.add_column("Setting")
    table.add_column("Value")

    table.add_row("description", cfg.description or "-")
    table.add_row("generation_model", cfg.generation_model)
    table.add_row("embedding_model", cfg.embedding_model)
    table.add_row(
        "chunk_size / overlap", f"{cfg.chunking.chunk_size} / {cfg.chunking.chunk_overlap}"
    )
    table.add_row(
        "top_k retrieve -> context",
        f"{cfg.retrieval.top_k_retrieve} -> {cfg.retrieval.top_k_context}",
    )
    table.add_row("hybrid", str(cfg.retrieval.use_hybrid))
    table.add_row("reranker", cfg.retrieval.reranker_model if cfg.retrieval.use_reranker else "off")
    table.add_row("decomposition", str(cfg.retrieval.use_decomposition))
    table.add_row("corrective loop", str(cfg.corrective.enabled))
    if cfg.corrective.enabled:
        table.add_row(
            "  thresholds",
            f"accept>={cfg.corrective.accept_threshold} "
            f"abstain<{cfg.corrective.abstain_threshold} "
            f"max_attempts={cfg.corrective.max_attempts}",
        )
    console.print(table)


@app.command()
def download(
    book: str = typer.Option(None, "--book", "-b", help="Fetch only this slug."),
    force: bool = typer.Option(False, "--force", help="Re-download even if already present."),
) -> None:
    """Fetch the source corpus and write data/raw/manifest.json.

    Safe to re-run: existing files are checksummed and skipped, and an
    interrupted transfer resumes rather than restarting.
    """
    try:
        books = (get_book(book),) if book else CORPUS
    except KeyError as exc:
        console.print(f"[red]FAIL[/red]  {exc}")
        raise typer.Exit(code=1) from exc

    columns = (
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
    )
    with Progress(*columns, console=console) as bar:
        tasks: dict[str, TaskID] = {}

        def on_progress(slug: str, done: int, total: int) -> None:
            if slug not in tasks:
                tasks[slug] = bar.add_task(slug, total=total or None)
            bar.update(tasks[slug], completed=done, total=total or None)

        manifest = download_corpus(books=books, force=force, progress=on_progress)

    table = Table(title="Corpus", show_header=True, header_style="bold")
    for column in ("Title", "Pages", "Size", "SHA-256", "License"):
        table.add_column(column)
    for record in manifest.books:
        table.add_row(
            f"{record.title} ({record.edition})",
            f"{record.page_count:,}",
            f"{record.size_mb:,.0f} MB",
            record.sha256[:12],
            record.license_name,
        )
    console.print(table)
    console.print(f"[green]OK[/green]    manifest written for {len(manifest.books)} book(s)")


@app.command()
def ingest(
    profile: str = typer.Option(None, "--profile", "-p", help="Profile to take chunking from."),
    recreate: bool = typer.Option(False, "--recreate", help="Drop the collection and re-embed."),
    batch_size: int = typer.Option(64, "--batch-size", help="Chunks per embed/upsert batch."),
) -> None:
    """Build the search index from downloaded PDFs.

    Runs entirely offline -- embeddings are local, so no API key is required.
    Safe to interrupt: a later run resumes from whatever was already written.
    """
    settings = Settings()
    cfg = load_pipeline_config(profile or settings.profile)

    console.print(f"[bold]collection[/bold]  {settings.qdrant_collection}")
    console.print(f"[bold]target    [/bold]  {describe_target(settings)}")
    console.print(f"[bold]embedder  [/bold]  {cfg.embedding_model} ({cfg.embedding_dim}d, local)")

    client = build_client(settings)
    columns = (
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console) as bar:
        tasks: dict[str, TaskID] = {}

        def on_progress(stage: str, done: int, total: int) -> None:
            if stage not in tasks:
                tasks[stage] = bar.add_task(stage, total=total or None)
            bar.update(tasks[stage], completed=done, total=total or None)

        report = run_ingest(
            client,
            collection=settings.qdrant_collection,
            embedding_model=cfg.embedding_model,
            embedding_dim=cfg.embedding_dim,
            chunk_size=cfg.chunking.chunk_size,
            chunk_overlap=cfg.chunking.chunk_overlap,
            batch_size=batch_size,
            recreate=recreate,
            progress=on_progress,
        )

    table = Table(title="Ingest", show_header=True, header_style="bold")
    for column in ("Book", "Pages", "Chunks", "Embedded", "Skipped"):
        table.add_column(column)
    for book in report.books:
        table.add_row(
            book.title,
            f"{book.pages_parsed:,}",
            f"{book.chunks_created:,}",
            f"{book.chunks_embedded:,}",
            f"{book.chunks_skipped:,}",
        )
    console.print(table)
    console.print(
        f"[green]OK[/green]    {report.points_in_collection:,} points in "
        f"'{report.collection}' in {report.duration_seconds:.0f}s"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    profile: str = typer.Option(None, "--profile", "-p", help="Pipeline profile to use."),
    show_context: bool = typer.Option(False, "--context", help="Print the passages used."),
) -> None:
    """Answer a question from the indexed corpus."""
    settings = Settings()
    cfg = load_pipeline_config(profile or settings.profile)
    pipeline = Pipeline(settings, cfg)

    try:
        with console.status("thinking..."):
            result = pipeline.answer(question)
    except ValueError as exc:
        console.print(f"[red]FAIL[/red]  {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        pipeline.close()

    console.print()
    console.print(result.text)

    if result.citations:
        console.print("\n[bold]Sources[/bold]")
        for citation in result.citations:
            console.print(
                f"  [{citation.marker}] {citation.citation}  [dim]({citation.license_name})[/dim]"
            )
    else:
        console.print(
            "\n[yellow]WARN[/yellow]  no citation resolved - treat this answer as unsupported"
        )

    if show_context:
        console.print("\n[bold]Context[/bold]")
        for index, chunk in enumerate(result.retrieved, start=1):
            console.print(f"  [{index}] {chunk.score:.3f}  {chunk.citation}")
            console.print(f"      [dim]{' '.join(chunk.text.split())[:160]}...[/dim]")

    console.print(f"\n[dim]{result.trace.summary()}[/dim]")


@app.command()
def health() -> None:
    """Verify configuration and vector store connectivity.

    Exits non-zero on failure so CI and the container healthcheck can rely on it.
    """
    try:
        settings = Settings()
    # Broad catches throughout this command are deliberate: `health` exists to
    # turn any startup failure into a legible message and a non-zero exit, not
    # to propagate a traceback. Each is re-raised as typer.Exit, so ruff's
    # BLE001 does not flag them.
    except Exception as exc:
        console.print(f"[red]FAIL[/red]  configuration invalid: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]OK[/green]    settings loaded (profile={settings.profile})")
    console.print(f"[green]OK[/green]    qdrant target: {describe_target(settings)}")

    try:
        load_pipeline_config(settings.profile)
        console.print("[green]OK[/green]    pipeline config valid")
    except Exception as exc:
        console.print(f"[red]FAIL[/red]  pipeline config: {exc}")
        raise typer.Exit(code=1) from exc

    try:
        client = build_client(settings)
        collections = [c.name for c in client.get_collections().collections]
    except Exception as exc:
        console.print(f"[red]FAIL[/red]  qdrant unreachable: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]OK[/green]    qdrant reachable ({len(collections)} collection(s))")

    if settings.qdrant_collection in collections:
        count = client.count(settings.qdrant_collection).count
        console.print(
            f"[green]OK[/green]    collection '{settings.qdrant_collection}': {count} points"
        )
    else:
        console.print(
            f"[yellow]WARN[/yellow]  collection '{settings.qdrant_collection}' not found "
            "- run ingestion first"
        )


# ---------------------------------------------------------------------------
# Gold set
# ---------------------------------------------------------------------------

goldset_app = typer.Typer(
    name="goldset",
    help="Build and validate the evaluation gold set.",
    no_args_is_help=True,
)
app.add_typer(goldset_app)


@goldset_app.command("draft")
def goldset_draft(
    factual: int = typer.Option(28, "--factual", help="Single-passage questions to draft."),
    multi_hop: int = typer.Option(20, "--multi-hop", help="Two-passage questions to draft."),
    unanswerable: int = typer.Option(12, "--unanswerable", help="Blank stubs to emit."),
    out: Path = typer.Option(
        Path("eval/goldset/drafts.jsonl"), "--out", help="Where to write candidates."
    ),
    seed: int = typer.Option(20260813, "--seed", help="Sampling seed, for reproducibility."),
    oversample: int = typer.Option(
        6, "--oversample", help="Candidates sampled per question wanted."
    ),
) -> None:
    """Draft gold-set candidates from the indexed corpus.

    Produces CANDIDATES, not a gold set. Every draft needs human review before
    it is worth anything, and the unanswerable questions are emitted as blank
    stubs because a model cannot write them usefully.
    """
    from vidyarag.evaluation import draft as drafting

    settings = Settings()
    cfg = load_pipeline_config(settings.profile)
    client = build_client(settings)

    with console.status("reading corpus..."):
        chunks = drafting.scroll_chunks(client, settings.qdrant_collection)

    if not chunks:
        console.print("[red]FAIL[/red]  collection is empty - run `vidyarag ingest` first")
        raise typer.Exit(code=1)
    console.print(f"[green]OK[/green]    {len(chunks):,} chunks available")

    llm = get_gemini_client(settings.google_api_key.get_secret_value())
    # Two independent pools, each heavily oversampled. Roughly half of all
    # drafts are dropped for being answerable from general knowledge -- that
    # rejection rate is the parametric-knowledge check working, not a fault --
    # and drawing both types from one pool let the factual pass consume it all
    # and starve multi-hop entirely.
    factual_pool = drafting.sample_chunks(chunks, factual * oversample, seed=seed)
    multihop_pool = drafting.sample_chunks(chunks, multi_hop * oversample, seed=seed + 1)

    drafts: list[dict[str, Any]] = []
    columns = (
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console) as bar:
        task = bar.add_task("factual", total=factual)
        for candidate in factual_pool:
            if sum(1 for d in drafts if d["type"] == "factual") >= factual:
                break
            drafted = drafting.draft_factual(llm, cfg.grader_model, candidate)
            if drafted:
                drafts.append(drafted)
                bar.update(task, completed=sum(1 for d in drafts if d["type"] == "factual"))

        task = bar.add_task("multi-hop", total=multi_hop)
        # Each seed is paired with its nearest neighbour from another section
        # rather than with whatever came next in a shuffled list. Random pairs
        # yield questions that join unrelated passages under a token "both
        # involve..." framing, which is not multi-hop reasoning.
        for seed_chunk in multihop_pool:
            if sum(1 for d in drafts if d["type"] == "multi_hop") >= multi_hop:
                break
            partner = drafting.find_related_chunk(
                client,
                settings.qdrant_collection,
                seed=seed_chunk,
                embedding_model=cfg.embedding_model,
            )
            if partner is None:
                continue
            drafted = drafting.draft_multihop(llm, cfg.grader_model, seed_chunk, partner)
            if drafted:
                drafts.append(drafted)
                bar.update(task, completed=sum(1 for d in drafts if d["type"] == "multi_hop"))

    client.close()
    drafting.assign_ids(drafts)
    records = drafts + drafting.unanswerable_stubs(unanswerable)
    written = drafting.write_jsonl(records, out, header=drafting.UNANSWERABLE_STUB_HEADER)
    review = drafting.write_review_sheet(drafts, out.with_name("REVIEW.md"))

    table = Table(title="Drafted", show_header=True, header_style="bold")
    table.add_column("Type")
    table.add_column("Count", justify="right")
    for kind in ("factual", "multi_hop"):
        table.add_row(kind, str(sum(1 for d in drafts if d["type"] == kind)))
    table.add_row("unanswerable (stubs)", str(unanswerable))
    console.print(table)
    console.print(f"[green]OK[/green]    candidates -> {written}")
    console.print(f"[green]OK[/green]    review sheet -> {review}")
    console.print(
        "\n[yellow]NOT A GOLD SET YET.[/yellow] Review every question, write the "
        f"{unanswerable} unanswerable ones by hand, then rename to "
        "[bold]goldset_v1.jsonl[/bold]."
    )


@goldset_app.command("check")
def goldset_check(
    path: Path = typer.Option(None, "--path", help="Gold set to validate."),
) -> None:
    """Validate a gold set and show its composition."""
    from vidyarag.evaluation.goldset import load_goldset, summarise_goldset

    try:
        questions = load_goldset(path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]FAIL[/red]  {exc}")
        raise typer.Exit(code=1) from exc

    counts = summarise_goldset(questions)
    table = Table(title=f"Gold set ({len(questions)} questions)", header_style="bold")
    table.add_column("Type")
    table.add_column("Count", justify="right")
    table.add_column("Share", justify="right")
    for kind, count in counts.items():
        share = f"{count / len(questions):.0%}" if questions else "-"
        table.add_row(kind, str(count), share)
    console.print(table)

    todo = [q for q in questions if q.question.startswith("TODO")]
    if todo:
        console.print(
            f"[yellow]WARN[/yellow]  {len(todo)} unwritten stub(s) remain - "
            "results measured now would be meaningless"
        )
    else:
        console.print("[green]OK[/green]    gold set is valid")


@goldset_app.command("unanswerable")
def goldset_unanswerable(
    drafts: Path = typer.Option(
        Path("eval/goldset/drafts.jsonl"), "--drafts", help="Draft file to fill stubs in."
    ),
    wanted: int = typer.Option(12, "--wanted", help="Accepted candidates to produce."),
    attempts: int = typer.Option(40, "--attempts", help="Maximum candidates to try."),
    seed: int = typer.Option(20260816, "--seed", help="Sampling seed, for reproducibility."),
    write: bool = typer.Option(
        False, "--write", help="Replace the TODO stubs in --drafts with accepted candidates."
    ),
) -> None:
    """Propose unanswerable questions and verify them against the corpus.

    A candidate is accepted only if it is topically in domain -- judged by
    retrieval similarity against the real index -- AND the passages that
    similarity retrieves do not answer it. That combination is the hard case
    abstention has to handle; either check alone admits questions that make the
    metric look good for the wrong reason.

    Output is still candidates. Read them before trusting them.
    """
    import json

    from vidyarag.evaluation import draft as drafting
    from vidyarag.evaluation import verify
    from vidyarag.evaluation.goldset import GoldQuestion

    settings = Settings()
    cfg = load_pipeline_config(settings.profile)
    client = build_client(settings)

    with console.status("reading corpus..."):
        chunks = drafting.scroll_chunks(client, settings.qdrant_collection)
    if not chunks:
        console.print("[red]FAIL[/red]  collection is empty - run `vidyarag ingest` first")
        raise typer.Exit(code=1)

    # Calibrate the in-domain cutoff from questions already known to be in
    # domain, rather than picking a number that feels right.
    existing = [
        GoldQuestion.model_validate(json.loads(line))
        for line in drafts.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("#", "//"))
    ]
    answerable = [q for q in existing if q.is_answerable]
    if not answerable:
        console.print("[red]FAIL[/red]  no answerable questions to calibrate against")
        raise typer.Exit(code=1)

    with console.status("calibrating in-domain threshold..."):
        threshold = verify.in_domain_threshold(
            client,
            answerable,
            collection=settings.qdrant_collection,
            embedding_model=cfg.embedding_model,
        )
    console.print(
        f"[green]OK[/green]    in-domain cutoff [bold]{threshold:.3f}[/bold] "
        f"(10th percentile of {len(answerable)} known in-domain questions)"
    )

    llm = get_gemini_client(settings.google_api_key.get_secret_value())
    seeds = drafting.sample_chunks(chunks, attempts, seed=seed)

    def show(check: verify.UnanswerableCheck) -> None:
        mark = "[green]accept[/green]" if check.accepted else "[dim]reject[/dim]"
        console.print(f"  {mark}  {check.question[:88]}")
        if not check.accepted:
            console.print(f"          [dim]{check.verdict}[/dim]")

    try:
        checks = verify.propose_unanswerable(
            llm,
            client,
            seeds,
            draft_model=cfg.generation_model,
            grader_model=cfg.grader_model,
            collection=settings.qdrant_collection,
            embedding_model=cfg.embedding_model,
            threshold=threshold,
            wanted=wanted,
            on_result=show,
        )
    except verify.ProposalAborted as exc:
        console.print(f"\n[red]FAIL[/red]  {exc}")
        console.print(
            "[dim]Gemini's free tier has a daily request cap as well as a per-minute "
            "one. A daily cap does not recover within a retry - wait for the reset "
            "or use a different key.[/dim]"
        )
        raise typer.Exit(code=1) from exc

    accepted = [c for c in checks if c.accepted]
    console.print(
        f"\n[green]OK[/green]    accepted [bold]{len(accepted)}[/bold] of {len(checks)} candidates"
    )
    if len(checks) and len(accepted) / len(checks) > 0.9:
        console.print(
            "[yellow]WARN[/yellow]  acceptance rate above 90% - the checks may be too "
            "permissive to be filtering anything"
        )

    if not write:
        console.print("\n[dim]Re-run with --write to replace the TODO stubs.[/dim]")
        return

    lines = drafts.read_text(encoding="utf-8").splitlines()
    queue = list(accepted)
    out: list[str] = []
    replaced = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            out.append(line)
            continue
        record = json.loads(stripped)
        if record.get("type") == "unanswerable" and str(record.get("question", "")).startswith(
            "TODO"
        ):
            if not queue:
                out.append(line)
                continue
            check = queue.pop(0)
            record["question"] = check.question
            record["provenance"] = "llm_drafted_retrieval_verified"
            record["notes"] = (
                f"APPROVE OR DELETE. Adjacent to: {check.topic}. "
                f"Why absent: {check.rationale} "
                f"Verified: top similarity {check.top_score:.3f} >= {threshold:.3f}; "
                f"grader says corpus does not answer it ({check.grader_reason})"
            )
            replaced += 1
            out.append(json.dumps(record, ensure_ascii=False))
        else:
            out.append(line)

    drafts.write_text("\n".join(out) + "\n", encoding="utf-8")
    console.print(f"[green]OK[/green]    replaced {replaced} stub(s) in {drafts}")


@goldset_app.command("triage")
def goldset_triage(
    drafts: Path = typer.Option(
        Path("eval/goldset/drafts.jsonl"), "--drafts", help="Draft file to triage."
    ),
) -> None:
    """Flag drafted questions whose gold passage does not support the answer.

    Checks data quality only. It deliberately does NOT check whether retrieval
    finds the gold chunk: dropping questions the pipeline currently misses would
    leave a gold set the baseline already succeeds on, and every later
    improvement would be measured against a target moved to meet it.
    """
    import json

    from vidyarag.evaluation import draft as drafting
    from vidyarag.evaluation.goldset import GoldQuestion

    settings = Settings()
    cfg = load_pipeline_config(settings.profile)
    client = build_client(settings)

    questions = [
        GoldQuestion.model_validate(json.loads(line))
        for line in drafts.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("#", "//"))
    ]
    answerable = [q for q in questions if q.is_answerable]
    if not answerable:
        console.print("[yellow]WARN[/yellow]  nothing to triage")
        return

    with console.status("reading corpus..."):
        chunks = drafting.scroll_chunks(client, settings.qdrant_collection)
    lookup = {c.chunk_id: (c.citation, c.text) for c in chunks}

    llm = get_gemini_client(settings.google_api_key.get_secret_value())

    def show(finding: Any) -> None:
        if finding.needs_attention:
            console.print(f"  [yellow]flag[/yellow]  {finding.id}  {finding.question[:76]}")
            console.print(f"          [dim]{finding.reason[:110]}[/dim]")

    from vidyarag.evaluation import verify

    findings = verify.triage_answerable(
        llm,
        answerable,
        lookup,
        grader_model=cfg.grader_model,
        on_result=show,
    )

    flagged = [f for f in findings if f.needs_attention]
    console.print(
        f"\n[green]OK[/green]    {len(findings) - len(flagged)} of {len(findings)} "
        "questions supported by their gold passage"
    )
    if flagged:
        console.print(
            f"[yellow]WARN[/yellow]  {len(flagged)} need a look - "
            "delete rather than repair anything doubtful"
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@app.command("eval")
def evaluate(
    profile: str = typer.Option(None, "--profile", "-p", help="Profile to evaluate."),
    goldset: Path = typer.Option(None, "--goldset", help="Gold set JSONL."),
    limit: int = typer.Option(None, "--limit", help="Only the first N questions (smoke run)."),
    concurrency: int = typer.Option(3, "--concurrency", help="Simultaneous grading requests."),
    rate: float = typer.Option(
        6.0, "--rate", help="Graded metrics per minute. Lower this if you see 429s."
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached grader responses."),
    compare: str = typer.Option(
        "baseline", "--compare", help="Profile to diff against. Empty to skip."
    ),
) -> None:
    """Run a profile against the gold set and write a versioned result file."""
    from vidyarag.evaluation.report import render_report
    from vidyarag.evaluation.runner import latest_run, run_evaluation

    settings = Settings()
    name = profile or settings.profile
    console.print(f"[bold]profile   [/bold]  {name}")
    console.print(f"[bold]target    [/bold]  {describe_target(settings)}")

    columns = _progress_columns()
    try:
        with Progress(*columns, console=console) as bar:
            answering = bar.add_task("answering", total=limit)
            grading = bar.add_task("grading", total=limit)

            run = run_evaluation(
                profile=name,
                goldset_path=goldset,
                limit=limit,
                concurrency=concurrency,
                scores_per_minute=rate,
                use_cache=not no_cache,
                settings=settings,
                on_answered=lambda _: bar.advance(answering),
                on_graded=lambda _: bar.advance(grading),
            )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]FAIL[/red]  {exc}")
        raise typer.Exit(code=1) from exc

    written = run.save()
    baseline = latest_run(compare) if compare and compare != run.profile else None
    report_path = written.with_suffix(".md")
    report_path.write_text(render_report(run, baseline), encoding="utf-8")

    if not run.is_valid:
        # Withhold the table entirely. Printing scores beside a warning invites
        # someone to read the scores and skip the warning, and these scores are
        # not a weaker version of the real ones -- they describe an easier task.
        console.print(
            f"\n[red]INVALID RUN[/red]  {len(run.failed)} of {len(run.samples)} "
            f"questions ({run.failure_rate:.0%}) never produced an answer."
        )
        for kind, count in sorted(run.failures_by_type().items()):
            console.print(f"              lost {count:>3} {kind}")
        console.print(
            "\n[dim]Metrics are withheld, not merely flagged. The gold set is ordered\n"
            "by type, so an interrupted run loses whole categories and the average\n"
            "over what remains measures an easier task.[/dim]"
        )
        if run.failed:
            first = " ".join(str(run.failed[0].error).split())[:150]
            console.print(f"\n[dim]first failure: {first}[/dim]")
        console.print(f"\n[green]OK[/green]    results -> {written}")
        console.print(f"[green]OK[/green]    report  -> {report_path}")
        console.print(
            "\n[dim]Answers are cached, so re-running once the cause is resolved\n"
            "only pays for the questions that failed.[/dim]"
        )
        raise typer.Exit(code=1)

    table = Table(title=f"{run.profile} · {run.run_id}", header_style="bold")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    for key, value in run.aggregates.items():
        if key == "graded_samples":
            continue
        table.add_row(key, "—" if value is None else f"{value:.3f}")
    for key, value in run.retrieval_aggregates.items():
        table.add_row(f"[dim]{key}[/dim]", "—" if value is None else f"{value:.3f}")
    recall = run.abstention.get("recall")
    table.add_row("abstention recall", "—" if recall is None else f"{recall:.3f}")
    console.print(table)

    failed = int(run.totals.get("failed", 0))
    if failed:
        console.print(f"[yellow]WARN[/yellow]  {failed} question(s) failed to answer")
    console.print(f"[green]OK[/green]    results -> {written}")
    console.print(f"[green]OK[/green]    report  -> {report_path}")


@app.command()
def report(
    profiles: list[str] = typer.Argument(None, help="Profiles to compare, baseline first."),
) -> None:
    """Print a comparison table across the latest run of each profile."""
    from vidyarag.evaluation.report import render_comparison
    from vidyarag.evaluation.runner import latest_run

    names = profiles or ["baseline"]
    runs = []
    for name in names:
        run = latest_run(name)
        if run is None:
            console.print(f"[yellow]WARN[/yellow]  no results for profile {name!r}")
            continue
        runs.append(run)

    if not runs:
        console.print("[red]FAIL[/red]  nothing to report - run `vidyarag eval` first")
        raise typer.Exit(code=1)
    console.print(render_comparison(runs))


if __name__ == "__main__":  # pragma: no cover
    app()
