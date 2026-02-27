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
index_app  = typer.Typer(no_args_is_help=True, help="Indexes: dense (Qdrant) + keyword (BM25)")

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
# INDEX COMMANDS — Dense / Qdrant  (Day 5)
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
    Build or update the Qdrant dense vector index from data/chunks/.

    Incremental by default — skips unchanged chunks.
    Use --force-reindex to drop and rebuild from scratch.

    Examples:
        uv run python -m ira index build
        uv run python -m ira index build --doc-id arxiv_2205.14135v1
        uv run python -m ira index build --force-reindex
    """
    from ira.retrieval.embedder import get_embedder
    from ira.retrieval.index_runner import build_index
    from ira.retrieval.qdrant_index import get_index

    rprint(f"[bold cyan]Building dense index[/bold cyan] from: {chunks}")
    rprint(f"  Model  : [green]{settings.EMBED_MODEL}[/green]")
    rprint(f"  Qdrant : [green]{settings.QDRANT_MODE}[/green] → "
           f"{settings.QDRANT_PATH if settings.QDRANT_MODE == 'embedded' else settings.QDRANT_URL}")
    rprint(f"  Cache  : [green]{'enabled' if settings.caching_enabled else 'disabled'}[/green]")
    if doc_id:
        rprint(f"  Filter : doc_id=[yellow]{doc_id}[/yellow]")

    embedder = get_embedder(batch_size=embed_batch)
    index    = get_index(vector_dim=embedder.embedding_dim)

    if force_reindex:
        rprint("[yellow]--force-reindex: dropping existing collection...[/yellow]")
        try:
            index.delete_collection()
        except Exception:
            pass

    result = build_index(
        chunks_root=chunks,
        doc_id=doc_id,
        embed_batch_size=embed_batch,
        upsert_batch_size=upsert_batch,
        embedder=embedder,
        index=index,
    )

    table = Table(title="Dense Index Build", show_lines=True)
    table.add_column("Doc ID", style="cyan", max_width=40)
    table.add_column("Chunks",    justify="right")
    table.add_column("Upserted",  justify="right", style="green")
    table.add_column("Skipped",   justify="right", style="yellow")
    table.add_column("Status")
    for doc in result.docs:
        status = "[green]✓[/green]" if doc.ok else f"[red]✗ {doc.error}[/red]"
        table.add_row(doc.doc_id, str(doc.chunks_read), str(doc.upserted), str(doc.skipped), status)
    console.print(table)

    info = result.collection_info
    rprint(f"\n[bold]Collection:[/bold] {info.get('name')}")
    rprint(f"  Points  : [green]{info.get('points_count', 'N/A')}[/green]")
    rprint(f"  Vectors : [green]{info.get('vectors_count', 'N/A')}[/green]")

    index.close()   # avoid Windows sys.meta_path shutdown warning

    if result.failed_docs:
        rprint(f"\n[red]Failed: {result.failed_docs}[/red]")
        raise typer.Exit(code=1)

    rprint(f"\n[bold green]✓ Done.[/bold green] "
           f"upserted={result.total_upserted} skipped={result.total_skipped}")


@index_app.command("query")
def index_query(
    q: str = typer.Option(..., "--q"),
    top_k: int = typer.Option(5, "--top-k"),
    score_threshold: float = typer.Option(0.0, "--score-threshold"),
    show_text: bool = typer.Option(True, "--show-text/--no-text"),
    json_output: bool = typer.Option(False, "--json"),
):
    """
    Query the dense vector index (semantic search).

    Examples:
        uv run python -m ira index query --q "FlashAttention IO-awareness"
        uv run python -m ira index query --q "KV cache quantization" --top-k 10
        uv run python -m ira index query --q "speculative decoding" --json
    """
    from ira.retrieval.index_runner import query_index

    results = query_index(query_text=q, top_k=top_k, score_threshold=score_threshold)

    if json_output:
        typer.echo(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if not results:
        rprint("[yellow]No results found.[/yellow]")
        return

    rprint(f"\n[bold cyan]Dense query:[/bold cyan] {q!r}  [dim](top_k={top_k})[/dim]\n")
    table = Table(show_lines=True)
    table.add_column("#",      justify="right", style="dim", width=3)
    table.add_column("Score",  justify="right", width=6)
    table.add_column("Doc ID", style="cyan", max_width=35)
    table.add_column("Section", max_width=40)
    table.add_column("Flags",  width=4)
    if show_text:
        table.add_column("Text preview", max_width=60)

    for r in results:
        flags = ("💻" if r.get("is_code") else "") + ("📊" if r.get("is_table") else "")
        row   = [str(r["rank"]), str(r["score"]), r["doc_id"], r["section"], flags]
        if show_text:
            row.append(r["text_preview"])
        table.add_row(*row)

    console.print(table)

    from ira.retrieval.qdrant_index import _default_index
    if _default_index is not None:
        _default_index.close()


@index_app.command("info")
def index_info():
    """Show dense (Qdrant) collection and embedding cache statistics."""
    from ira.retrieval.embedder import get_embedder
    from ira.retrieval.qdrant_index import get_index

    embedder = get_embedder()
    index    = get_index(vector_dim=embedder.embedding_dim)

    rprint("\n[bold cyan]Qdrant Collection[/bold cyan]")
    for k, v in index.collection_info().items():
        rprint(f"  {k}: [green]{v}[/green]")

    rprint("\n[bold cyan]Embedding Cache[/bold cyan]")
    for k, v in embedder.cache_stats().items():
        rprint(f"  {k}: [green]{v}[/green]")

    index.close()


# ═══════════════════════════════════════════════════════════════════════════════
# INDEX COMMANDS — BM25 / Keyword  (Day 6)
# ═══════════════════════════════════════════════════════════════════════════════

@index_app.command("bm25-build")
def bm25_build(
    chunks: Path = typer.Option(Path("data/chunks"), "--chunks", help="Chunks root directory"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Only index this doc_id"),
    batch_size: int = typer.Option(500, "--batch-size", help="SQLite executemany batch size"),
    force_rebuild: bool = typer.Option(
        False, "--force-rebuild",
        help="Delete existing DB and rebuild from scratch",
    ),
):
    """
    Build or update the BM25 keyword index from data/chunks/.

    Uses SQLite FTS5 with a technical-aware tokenizer that preserves
    terms like FlashAttention-2, FP8, H100, bge-small-en-v1.5 as single tokens.

    Incremental by default — skips chunks whose text hasn't changed.

    Examples:
        uv run python -m ira index bm25-build
        uv run python -m ira index bm25-build --doc-id arxiv_2205.14135v1
        uv run python -m ira index bm25-build --force-rebuild
    """
    from ira.retrieval.bm25_index import BM25Index
    from ira.retrieval.bm25_runner import build_bm25_index

    db_path = settings.bm25_db_path
    rprint(f"[bold cyan]Building BM25 index[/bold cyan] from: {chunks}")
    rprint(f"  DB path : [green]{db_path}[/green]")
    if doc_id:
        rprint(f"  Filter  : doc_id=[yellow]{doc_id}[/yellow]")

    if force_rebuild and db_path.exists():
        rprint("[yellow]--force-rebuild: deleting existing BM25 database...[/yellow]")
        db_path.unlink()

    bm25_idx = BM25Index(db_path)

    result = build_bm25_index(
        chunks_root=chunks,
        doc_id=doc_id,
        batch_size=batch_size,
        bm25_index=bm25_idx,
    )

    # Results table
    table = Table(title="BM25 Index Build", show_lines=True)
    table.add_column("Doc ID",   style="cyan", max_width=40)
    table.add_column("Chunks",   justify="right")
    table.add_column("Inserted", justify="right", style="green")
    table.add_column("Updated",  justify="right", style="blue")
    table.add_column("Skipped",  justify="right", style="yellow")
    table.add_column("Status")

    for doc in result.docs:
        status = "[green]✓[/green]" if doc.ok else f"[red]✗ {doc.error}[/red]"
        table.add_row(
            doc.doc_id,
            str(doc.chunks_read),
            str(doc.inserted),
            str(doc.updated),
            str(doc.skipped),
            status,
        )
    console.print(table)

    stats = result.index_stats
    rprint(f"\n[bold]BM25 Index Stats[/bold]")
    rprint(f"  Chunks  : [green]{stats.get('chunks', 'N/A')}[/green]")
    rprint(f"  Docs    : [green]{stats.get('docs', 'N/A')}[/green]")
    rprint(f"  DB size : [green]{stats.get('db_size_mb', 'N/A')} MB[/green]")

    bm25_idx.close()

    if result.failed_docs:
        rprint(f"\n[red]Failed: {result.failed_docs}[/red]")
        raise typer.Exit(code=1)

    rprint(
        f"\n[bold green]✓ Done.[/bold green] "
        f"inserted={result.total_inserted} "
        f"updated={result.total_updated} "
        f"skipped={result.total_skipped}"
    )


@index_app.command("bm25-query")
def bm25_query(
    q: str = typer.Option(..., "--q", help="Keyword query"),
    top_n: int = typer.Option(10, "--top-n", help="Number of results"),
    show_text: bool = typer.Option(True, "--show-text/--no-text"),
    json_output: bool = typer.Option(False, "--json"),
    show_tokens: bool = typer.Option(False, "--show-tokens", help="Print how query is tokenized"),
):
    """
    Query the BM25 keyword index.

    Technical terms like FlashAttention-2, FP8, H100 are matched exactly.
    Multi-word queries require ALL terms to appear (implicit AND).

    Examples:
        uv run python -m ira index bm25-query --q "FlashAttention-2"
        uv run python -m ira index bm25-query --q "FP8 quantization H100"
        uv run python -m ira index bm25-query --q "KV cache" --top-n 20
        uv run python -m ira index bm25-query --q "bge-small-en-v1.5" --show-tokens
    """
    from ira.retrieval.bm25_index import tokenize, tokenize_for_query
    from ira.retrieval.bm25_runner import query_bm25

    if show_tokens:
        tokens = tokenize(q)
        fts_q  = tokenize_for_query(q)
        rprint(f"\n[bold]Tokenization of:[/bold] {q!r}")
        rprint(f"  Tokens    : [cyan]{tokens}[/cyan]")
        rprint(f"  FTS5 MATCH: [cyan]{fts_q}[/cyan]\n")

    results = query_bm25(query_text=q, top_n=top_n)

    if json_output:
        typer.echo(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if not results:
        rprint("[yellow]No results found.[/yellow]")
        rprint("[dim]Tip: try --show-tokens to see how your query was tokenized.[/dim]")
        return

    rprint(f"\n[bold cyan]BM25 query:[/bold cyan] {q!r}  [dim](top_n={top_n})[/dim]\n")

    table = Table(show_lines=True)
    table.add_column("#",          justify="right", style="dim", width=3)
    table.add_column("BM25 Score", justify="right", width=8)
    table.add_column("Doc ID",     style="cyan", max_width=35)
    table.add_column("Section",    max_width=40)
    table.add_column("Flags",      width=4)
    if show_text:
        table.add_column("Text preview", max_width=60)

    for r in results:
        flags = ("💻" if r.get("is_code") else "") + ("📊" if r.get("is_table") else "")
        row   = [
            str(r["rank"]),
            str(r["bm25_score"]),
            r["doc_id"],
            r["section"],
            flags,
        ]
        if show_text:
            row.append(r["text_preview"])
        table.add_row(*row)

    console.print(table)

    # Clean shutdown
    from ira.retrieval.bm25_index import _default_bm25
    if _default_bm25 is not None:
        _default_bm25.close()


@index_app.command("bm25-info")
def bm25_info():
    """Show BM25 index statistics and tokenizer behaviour."""
    from ira.retrieval.bm25_index import BM25Index, tokenize

    db_path = settings.bm25_db_path
    idx = BM25Index(db_path)

    rprint("\n[bold cyan]BM25 Index Stats[/bold cyan]")
    for k, v in idx.stats().items():
        rprint(f"  {k}: [green]{v}[/green]")

    rprint("\n[bold cyan]Tokenizer Samples[/bold cyan]")
    samples = [
        "FlashAttention-2 achieves 3x speedup",
        "FP8 quantization on H100",
        "BAAI/bge-small-en-v1.5 384-dim embeddings",
        "KV cache paging with PagedAttention",
        "LoRA rank=8 with alpha=16",
    ]
    for s in samples:
        rprint(f"  [dim]{s!r}[/dim]")
        rprint(f"    → [cyan]{tokenize(s)}[/cyan]")

    idx.close()


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