from __future__ import annotations

import typer
from rich import print as rprint
from pathlib import Path
from typing import Optional

from ira.ingest.processor import process_all
from ira.ingest.chunk_runner import chunk_all
from ira.settings import settings
from ira.ingest.runner import run_ingest

app = typer.Typer(help="Intelligent Research Assistant CLI (ingest/query/eval)")
# app = typer.Typer(no_args_is_help=True)
ingest_app = typer.Typer(no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("fetch")
def ingest_fetch(
    seed: Path = typer.Option(..., "--seed", exists=True, readable=True),
    force: bool = typer.Option(False, "--force"),
    limit: int = typer.Option(0, "--limit", help="0 = no limit"),
):
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
    force: bool = typer.Option(False, "--force", help="Re-process even if output exists"),
    limit: int | None = typer.Option(None, "--limit", help="Max number of docs to process"),
    only_kind: str | None = typer.Option(None, "--only-kind", help="Filter: docs|pdf|github"),
    keep_pdf_page_breaks: bool = typer.Option(True, "--keep-pdf-page-breaks/--no-keep-pdf-page-breaks"),
):
    # uv run python -m ira ingest process --raw data/raw --out data/processed
    # uv run python -m ira ingest process --raw data/raw_v2 --only-kind pdf --out data/processed_v2 --limit 1
    # uv run python -m ira ingest process --raw data/raw --only-kind pdf --out data/processed_v2
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
                typer.echo(f" - {r.doc_id}: {r.error}")

@ingest_app.command("chunk")
def chunk_cmd(
    processed: Path = typer.Option(Path("data/processed"), "--processed", help="Processed docs root"),
    out: Path = typer.Option(Path("data/chunks"), "--out", help="Chunks output root"),
    force: bool = typer.Option(False, "--force", help="Re-chunk even if output exists"),
    limit: int | None = typer.Option(None, "--limit", help="Max number of docs to chunk"),
):
    # uv run python -m ira ingest chunk --processed data/processed --out data/chunks
    # uv run python -m ira ingest chunk --processed data/processed --out data/chunks --force
    # uv run python -m ira ingest chunk --processed data/processed --out data/chunks --limit 1
    results = chunk_all(
        processed_root=processed,
        out_root=out,
        force=force,
        limit=limit,
    )

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


@app.command()
def ingest():
    rprint("[yellow]TODO:[/yellow] ingest pipeline will be implemented in Day 2+")
    raise typer.Exit(code=0)

@app.command()
def query(q: str):
    rprint(f"[yellow]TODO:[/yellow] query agent will be implemented later. You asked: {q!r}")
    raise typer.Exit(code=0)

@app.command()
def eval():
    rprint("[yellow]TODO:[/yellow] eval harness will be implemented later")
    raise typer.Exit(code=0)