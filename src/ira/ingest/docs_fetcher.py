from __future__ import annotations

from pathlib import Path
from typing import Any

import html2text
import httpx

from ira.ingest.provenance import now_utc_iso, sha256_file, write_json
from ira.ingest.seeds import DocsSeed


def fetch_docs(seed: DocsSeed, *, out_dir: Path, client: httpx.Client) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    r = client.get(str(seed.url))
    r.raise_for_status()

    html_path = out_dir / "page.html"
    md_path = out_dir / "page.md"

    html_path.write_bytes(r.content)

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    md = h.handle(r.text)
    md_path.write_text(md, encoding="utf-8")

    meta: dict[str, Any] = {
        "kind": "docs",
        "retrieved_at": now_utc_iso(),
        "source": {
            "url": str(seed.url),
            "resolved_url": str(r.url),
            "version": seed.version,
            "http": {
                "etag": r.headers.get("etag"),
                "last_modified": r.headers.get("last-modified"),
            },
        },
        "seed": seed.model_dump(mode='json'),
        "artifacts": {
            "page.html": {"sha256": sha256_file(html_path), "bytes": html_path.stat().st_size},
            "page.md": {"sha256": sha256_file(md_path), "bytes": md_path.stat().st_size},
        },
    }

    write_json(out_dir / "meta.json", meta)
    return meta
