"""Command-line entry point.

Kept deliberately thin: commands resolve configuration, call into the package,
and print. Any logic worth testing belongs in a module, not here.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
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
from vidyarag.pipeline import Pipeline
from vidyarag.settings import Settings, load_pipeline_config
from vidyarag.store import build_client, describe_target

app = typer.Typer(
    name="vidyarag",
    help="Agentic, self-correcting RAG study assistant over OpenStax textbooks.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


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


if __name__ == "__main__":  # pragma: no cover
    app()
