from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx

from ira.ingest.provenance import now_utc_iso, sha256_file, write_json
from ira.ingest.seeds import ArxivSeed


ARXIV_API = "http://export.arxiv.org/api/query?id_list={id}"


def _pdf_url(arxiv_id: str, version: int) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}v{version}.pdf"


def fetch_arxiv(seed: ArxivSeed, *, out_dir: Path, client: httpx.Client) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_url = _pdf_url(seed.arxiv_id, seed.arxiv_version)
    pdf_path = out_dir / "paper.pdf"

    with client.stream("GET", pdf_url) as r:
        r.raise_for_status()
        with pdf_path.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)

    # metadata via arXiv API (reproducible snapshot stored)
    meta_xml = client.get(ARXIV_API.format(id=seed.arxiv_id))
    meta_xml.raise_for_status()
    xml_path = out_dir / "arxiv_api.xml"
    xml_path.write_bytes(meta_xml.content)

    # parse key fields (best-effort)
    parsed: dict[str, Any] = {"title": seed.title}
    try:
        root = ET.fromstring(meta_xml.content)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = root.find("a:entry", ns)
        if entry is not None:
            parsed["api_title"] = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            parsed["published"] = entry.findtext("a:published", default="", namespaces=ns)
            parsed["updated"] = entry.findtext("a:updated", default="", namespaces=ns)
            parsed["authors"] = [
                (e.findtext("a:name", default="", namespaces=ns) or "").strip()
                for e in entry.findall("a:author", ns)
            ]
    except Exception:
        parsed["parse_error"] = True

    meta = {
        "kind": "arxiv",
        "retrieved_at": now_utc_iso(),
        "source": {
            "arxiv_id": seed.arxiv_id,
            "arxiv_version": seed.arxiv_version,
            "pdf_url": pdf_url,
        },
        "seed": seed.model_dump(mode='json'),
        "artifacts": {
            "paper.pdf": {"sha256": sha256_file(pdf_path), "bytes": pdf_path.stat().st_size},
            "arxiv_api.xml": {"sha256": sha256_file(xml_path), "bytes": xml_path.stat().st_size},
        },
        "parsed": parsed,
    }

    write_json(out_dir / "meta.json", meta)
    return meta
