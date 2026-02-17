from __future__ import annotations

import typer
from rich import print as rprint

app = typer.Typer(help="Intelligent Research Assistant CLI (ingest/query/eval)")

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
