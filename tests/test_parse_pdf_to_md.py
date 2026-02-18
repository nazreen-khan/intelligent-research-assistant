# File: tests/test_parse_pdf_to_md.py

from __future__ import annotations

import json
from pathlib import Path

import ira.ingest.parse_pdf_to_md as pdfmod
from ira.ingest.parse_pdf_to_md import parse_pdf_doc


def test_parse_pdf_adds_header_and_fixes_hyphenation(monkeypatch, tmp_path: Path):
    raw_dir = tmp_path / "arxiv_test"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "meta.json").write_text(
        json.dumps(
            {
                "kind": "arxiv",
                "seed": {"title": "Test Paper", "url": "https://arxiv.org/abs/1234.5678"},
                "source": {"pdf_url": "https://arxiv.org/pdf/1234.5678.pdf", "version": "v1"},
            }
        ),
        encoding="utf-8",
    )

    # Create a dummy PDF file (content is not used because we monkeypatch extract_text)
    (raw_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n%fake\n")

    fake_text = (
        "arXiv preprint\n"
        "1 Introduction\n"
        "quantiza-\n"
        " tion improves speed.\n"
        "\f"
        "arXiv preprint\n"
        "2 Methods\n"
        "More text here.\n"
    )

    monkeypatch.setattr(pdfmod, "extract_text", lambda _: fake_text)

    md, meta = parse_pdf_doc(raw_dir, keep_page_breaks=True)

    assert md.startswith("# Test Paper")
    assert "**Doc ID:** arxiv_test" in md
    assert "**Source:** https://arxiv.org/pdf/1234.5678.pdf" in md
    assert "## 1 Introduction" in md
    assert "## 2 Methods" in md
    # hyphenation fixed by normalization
    assert "quantization improves speed" in md.replace("\n", " ")
    # page break marker included
    assert "\n---\n" in md

    assert meta["doc_id"] == "arxiv_test"
    assert meta["parser"]["name"] == "parse_pdf_to_md"
    assert meta["parser"]["method"] == "pdfminer"
    assert meta["pages"] == 2
    # repeated header/footer removed (at least the repeated line)
    assert meta["boilerplate"]["candidates"] >= 1


def test_parse_pdf_without_page_breaks(monkeypatch, tmp_path: Path):
    raw_dir = tmp_path / "arxiv_nopb"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "meta.json").write_text(
        json.dumps({"kind": "arxiv", "seed": {"title": "No PB"}}),
        encoding="utf-8",
    )
    (raw_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n%fake\n")

    monkeypatch.setattr(pdfmod, "extract_text", lambda _: "1 Intro\nHello\n\f2 Next\nWorld\n")

    md, meta = parse_pdf_doc(raw_dir, keep_page_breaks=False)
    assert "\n---\n" not in md
    assert meta["keep_page_breaks"] is False
