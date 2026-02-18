# File: tests/test_processor.py

from __future__ import annotations

import json
from pathlib import Path

import ira.ingest.processor as proc
from ira.ingest.processor import process_one_doc


def test_processor_routes_docs(monkeypatch, tmp_path: Path):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "processed"
    d = raw_root / "docs_1"
    d.mkdir(parents=True)

    (d / "meta.json").write_text(json.dumps({"kind": "docs", "seed": {"title": "T"}}), encoding="utf-8")
    (d / "page.html").write_text("<html><body><main><h1>X</h1></main></body></html>", encoding="utf-8")

    monkeypatch.setattr(proc, "parse_html_doc", lambda raw_dir: ("# T\n\n## X\n", {"doc_id": raw_dir.name, "parser": {"name": "parse_html_to_md"}}))

    res = process_one_doc(d, out_root, force=True)
    assert res.ok is True
    assert res.kind == "docs"
    assert (out_root / "docs_1" / "content.md").exists()
    assert (out_root / "docs_1" / "meta.json").exists()


def test_processor_routes_pdf(monkeypatch, tmp_path: Path):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "processed"
    d = raw_root / "pdf_1"
    d.mkdir(parents=True)

    (d / "meta.json").write_text(json.dumps({"kind": "arxiv", "seed": {"title": "P"}}), encoding="utf-8")
    (d / "paper.pdf").write_bytes(b"%PDF-1.4\n%fake\n")

    monkeypatch.setattr(proc, "parse_pdf_doc", lambda raw_dir, keep_page_breaks=True: ("# P\n\n## 1 Intro\n", {"doc_id": raw_dir.name, "parser": {"name": "parse_pdf_to_md"}}))

    res = process_one_doc(d, out_root, force=True)
    assert res.ok is True
    assert res.kind == "pdf"


def test_processor_routes_github(monkeypatch, tmp_path: Path):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "processed"
    d = raw_root / "gh_1"
    d.mkdir(parents=True)

    (d / "meta.json").write_text(json.dumps({"kind": "github", "seed": {"title": "G"}}), encoding="utf-8")
    (d / "README.md").write_text("# Hello\n", encoding="utf-8")

    monkeypatch.setattr(proc, "parse_github_doc", lambda raw_dir: ("# G\n\n## File: README.md\n# Hello\n", {"doc_id": raw_dir.name, "parser": {"name": "parse_code_snippets"}}))

    res = process_one_doc(d, out_root, force=True)
    assert res.ok is True
    assert res.kind == "github"
