from __future__ import annotations

import typer
from rich import print as rprint
from pathlib import Path
from typing import Optional


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
