from __future__ import annotations

import json
from pathlib import Path

from ira.ingest.parse_html_to_md import parse_html_doc


def test_parse_html_preserves_headings_lists_code_and_links(tmp_path: Path):
    raw_dir = tmp_path / "docs_test"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "meta.json").write_text(
        json.dumps(
            {
                "kind": "docs",
                "seed": {"title": "Test Docs Page", "url": "https://example.com/docs"},
                "source": {"resolved_url": "https://example.com/docs", "version": "2026-02-18"},
                "artifacts": {"page.html": {"sha256": "x", "bytes": 999}},
            }
        ),
        encoding="utf-8",
    )

    (raw_dir / "page.html").write_text(
        """
        <html>
          <head><title>Ignored Title</title></head>
          <body>
            <header>nav stuff</header>
            <main>
              <h1>Heading One</h1>
              <p>See <a href="https://example.com/x">this link</a>.</p>
              <h2>Section</h2>
              <ul><li>Item A</li><li>Item B</li></ul>
              <pre><code class="language-python">def f():
    return 1
</code></pre>
            </main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    md, meta = parse_html_doc(raw_dir)

    assert md.startswith("# Test Docs Page")
    assert "## Section" in md or "# Heading One" in md  # at least one heading survives
    assert "- Item A" in md
    assert "```python" in md
    assert "def f():" in md
    assert "[this link](https://example.com/x)" in md

    assert meta["doc_id"] == "docs_test"
    assert meta["source_type"] == "docs"
    assert meta["source_url"] == "https://example.com/docs"
    assert meta["used_fallback_md"] is False


def test_parse_html_tables_best_effort(tmp_path: Path):
    raw_dir = tmp_path / "docs_table"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "meta.json").write_text(
        json.dumps({"kind": "docs", "seed": {"title": "Table Page"}}),
        encoding="utf-8",
    )

    (raw_dir / "page.html").write_text(
        """
        <html><body><main>
          <h1>Table Section</h1>
          <table>
            <tr><th>Model</th><th>Speed</th></tr>
            <tr><td>A</td><td>10</td></tr>
            <tr><td>B</td><td>20</td></tr>
          </table>
        </main></body></html>
        """,
        encoding="utf-8",
    )

    md, _ = parse_html_doc(raw_dir)
    assert "| Model | Speed |" in md
    assert "| --- | --- |" in md
    assert "| A | 10 |" in md
