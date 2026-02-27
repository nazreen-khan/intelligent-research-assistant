from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from ira.ingest.chunk_runner import chunk_all
from ira.ingest.processor import process_all
from ira.ingest.runner import run_ingest
from ira.settings import settings

console = Console()

app = typer.Typer(help="Intelligent Research Assistant CLI (ingest/index/query/eval)")

# ── Sub-command groups ────────────────────────────────────────────────────────
ingest_app = typer.Typer(no_args_is_help=True, help="Ingest pipeline: fetch/process/chunk")
index_app = typer.Typer(no_args_is_help=True, help="Vector index: build/query/info")

app.add_typer(ingest_app, name="ingest")
app.add_typer(index_app, name="index")


# ═══════════════════════════════════════════════════════════════════════════════
# INGEST COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@ingest_app.command("fetch")
def ingest_fetch(
    seed: Path = typer.Option(..., "--seed", exists=True, readable=True),
    force: bool = typer.Option(False, "--force"),
    limit: int = typer.Option(0, "--limit", help="0 = no limit"),
):
    """Fetch raw documents from arXiv/docs/GitHub into data/raw/."""
    data_dir = Path(settings.IRA_DATA_DIR)
    manifest = data_dir / "provenance" / "manifest.jsonl"
    run_ingest(
        seed_path=seed,
        data_dir=data_dir,
        manifest_path=manifest,
        github_token=settings.GITHUB_TOKEN,
        force=force,
        limit=limit,
    )
    typer.echo(f"Done. Raw: {data_dir / 'raw'} | Manifest: {manifest}")


@ingest_app.command("process")
def process_cmd(
    raw: Path = typer.Option(Path("data/raw"), "--raw", help="Raw docs root"),
    out: Path = typer.Option(Path("data/processed"), "--out", help="Processed docs root"),
    force: bool = typer.Option(False, "--force"),
    limit: int | None = typer.Option(None, "--limit"),
    only_kind: str | None = typer.Option(None, "--only-kind", help="docs|pdf|github"),
    keep_pdf_page_breaks: bool = typer.Option(True, "--keep-pdf-page-breaks/--no-keep-pdf-page-breaks"),
):
    """Parse raw documents into cleaned Markdown in data/processed/."""
    results = process_all(
        raw_root=raw,
        out_root=out,
        force=force,
        limit=limit,
        only_kind=only_kind,
        keep_pdf_page_breaks=keep_pdf_page_breaks,
    )
    ok = sum(1 for r in results if r.ok)
    fail = sum(1 for r in results if not r.ok)
    typer.echo(f"Processed: {len(results)} | ok={ok} fail={fail}")
    if fail:
        for r in results:
            if not r.ok:
                typer.echo(f"  FAIL {r.doc_id}: {r.error}")


@ingest_app.command("chunk")
def chunk_cmd(
    processed: Path = typer.Option(Path("data/processed"), "--processed"),
    out: Path = typer.Option(Path("data/chunks"), "--out"),
    force: bool = typer.Option(False, "--force"),
    limit: int | None = typer.Option(None, "--limit"),
):
    """Split processed documents into parent/child chunks in data/chunks/."""
    results = chunk_all(processed_root=processed, out_root=out, force=force, limit=limit)
    ok = sum(1 for r in results if r.ok)
    fail = sum(1 for r in results if not r.ok)
    total_parents = sum(r.parent_count for r in results if r.ok)
    total_children = sum(r.child_count for r in results if r.ok)
    typer.echo(f"Chunked: {len(results)} docs | ok={ok} fail={fail}")
    typer.echo(f"Total parents={total_parents} | Total children={total_children}")
    if fail:
        for r in results:
            if not r.ok:
                typer.echo(f"  FAIL {r.doc_id}: {r.error}")


# ═══════════════════════════════════════════════════════════════════════════════
# INDEX COMMANDS  (Day 5)
# ═══════════════════════════════════════════════════════════════════════════════

@index_app.command("build")
def index_build(
    chunks: Path = typer.Option(Path("data/chunks"), "--chunks", help="Chunks root directory"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Only index this specific doc_id"),
    embed_batch: int = typer.Option(settings.EMBED_BATCH_SIZE, "--embed-batch", help="Embedding batch size"),
    upsert_batch: int = typer.Option(256, "--upsert-batch", help="Qdrant upsert batch size"),
    force_reindex: bool = typer.Option(False, "--force-reindex", help="Delete and rebuild collection from scratch"),
):
    """
    Build or update the Qdrant vector index from data/chunks/.

    Runs incremental by default — skips chunks that haven't changed.
    Use --force-reindex to drop and rebuild from scratch.

    Examples:
        uv run python -m ira index build
        uv run python -m ira index build --doc-id arxiv_2205.14135v1
        uv run python -m ira index build --force-reindex
    """
    from ira.retrieval.embedder import get_embedder
    from ira.retrieval.index_runner import build_index
    from ira.retrieval.qdrant_index import get_index

    rprint(f"[bold cyan]Building index[/bold cyan] from: {chunks}")
    rprint(f"  Model   : [green]{settings.EMBED_MODEL}[/green]")
    rprint(f"  Qdrant  : [green]{settings.QDRANT_MODE}[/green] → {settings.QDRANT_PATH if settings.QDRANT_MODE == 'embedded' else settings.QDRANT_URL}")
    rprint(f"  Cache   : [green]{'enabled' if settings.caching_enabled else 'disabled'}[/green]")
    if doc_id:
        rprint(f"  Filter  : doc_id=[yellow]{doc_id}[/yellow]")

    embedder = get_embedder(batch_size=embed_batch)
    index = get_index(vector_dim=embedder.embedding_dim)

    if force_reindex:
        rprint("[yellow]--force-reindex: dropping existing collection...[/yellow]")
        try:
            index.delete_collection()
        except Exception:
            pass  # may not exist yet

    result = build_index(
        chunks_root=chunks,
        doc_id=doc_id,
        embed_batch_size=embed_batch,
        upsert_batch_size=upsert_batch,
        embedder=embedder,
        index=index,
    )

    # ── Print results table ───────────────────────────────────────────────────
    table = Table(title="Index Build Results", show_lines=True)
    table.add_column("Doc ID", style="cyan", no_wrap=False)
    table.add_column("Chunks", justify="right")
    table.add_column("Upserted", justify="right", style="green")
    table.add_column("Skipped", justify="right", style="yellow")
    table.add_column("Status")

    for doc in result.docs:
        status = "[green]✓ ok[/green]" if doc.ok else f"[red]✗ {doc.error}[/red]"
        table.add_row(
            doc.doc_id,
            str(doc.chunks_read),
            str(doc.upserted),
            str(doc.skipped),
            status,
        )

    console.print(table)

    info = result.collection_info
    rprint(f"\n[bold]Collection:[/bold] {info.get('name')}")
    rprint(f"  Points  : [green]{info.get('points_count', 'N/A')}[/green]")
    rprint(f"  Vectors : [green]{info.get('vectors_count', 'N/A')}[/green]")
    rprint(f"  Status  : {info.get('status', 'N/A')}")

    # FIX 3: explicit close to suppress Windows "sys.meta_path is None" warning
    index.close()

    if result.failed_docs:
        rprint(f"\n[red]Failed docs: {result.failed_docs}[/red]")
        raise typer.Exit(code=1)

    rprint(f"\n[bold green]✓ Done.[/bold green] Total upserted={result.total_upserted} skipped={result.total_skipped}")


@index_app.command("query")
def index_query(
    q: str = typer.Option(..., "--q", help="Query text"),
    top_k: int = typer.Option(5, "--top-k", help="Number of results"),
    score_threshold: float = typer.Option(0.0, "--score-threshold", help="Minimum similarity score"),
    show_text: bool = typer.Option(True, "--show-text/--no-text", help="Show text preview"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """
    Query the vector index and return top-K matching chunks.

    Examples:
        uv run python -m ira index query --q "FlashAttention IO-awareness"
        uv run python -m ira index query --q "KV cache quantization" --top-k 10
        uv run python -m ira index query --q "speculative decoding" --json
    """
    from ira.retrieval.index_runner import query_index

    results = query_index(
        query_text=q,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    if json_output:
        typer.echo(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if not results:
        rprint("[yellow]No results found.[/yellow]")
        return  # index will be closed when process exits naturally

    rprint(f"\n[bold cyan]Query:[/bold cyan] {q!r}  [dim](top_k={top_k})[/dim]\n")

    table = Table(show_lines=True)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Doc ID", style="cyan", no_wrap=False, max_width=35)
    table.add_column("Section", no_wrap=False, max_width=40)
    table.add_column("Flags", width=6)
    if show_text:
        table.add_column("Text preview", max_width=60)

    for r in results:
        flags = ""
        if r.get("is_code"):
            flags += "💻"
        if r.get("is_table"):
            flags += "📊"

        row = [
            str(r["rank"]),
            str(r["score"]),
            r["doc_id"],
            r["section"],
            flags,
        ]
        if show_text:
            row.append(r["text_preview"])

        table.add_row(*row)

    console.print(table)

    # FIX 3: explicit close to suppress Windows shutdown warning
    from ira.retrieval.qdrant_index import _default_index
    if _default_index is not None:
        _default_index.close()


@index_app.command("info")
def index_info():
    """Show collection statistics."""
    from ira.retrieval.embedder import get_embedder
    from ira.retrieval.qdrant_index import get_index

    embedder = get_embedder()
    index = get_index(vector_dim=embedder.embedding_dim)
    info = index.collection_info()

    rprint("\n[bold cyan]Qdrant Collection Info[/bold cyan]")
    for k, v in info.items():
        rprint(f"  {k}: [green]{v}[/green]")

    cache_stats = embedder.cache_stats()
    rprint("\n[bold cyan]Embedding Cache[/bold cyan]")
    for k, v in cache_stats.items():
        rprint(f"  {k}: [green]{v}[/green]")


# ═══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL STUBS (forward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

@app.command()
def query(q: str):
    """[Day 9+] Full agent query pipeline."""
    rprint(f"[yellow]TODO:[/yellow] agent pipeline coming Day 9+. You asked: {q!r}")
    rprint("[dim]For now, use: uv run python -m ira index query --q '...'[/dim]")
    raise typer.Exit(code=0)


@app.command()
def eval():
    """[Day 13] Evaluation harness."""
    rprint("[yellow]TODO:[/yellow] eval harness coming Day 13.")
    raise typer.Exit(code=0)