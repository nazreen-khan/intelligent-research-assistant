from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx

from ira.ingest.arxiv_downloader import fetch_arxiv
from ira.ingest.docs_fetcher import fetch_docs
from ira.ingest.doc_id import doc_id_for_arxiv, doc_id_for_docs, doc_id_for_github
from ira.ingest.github_fetcher import fetch_github, resolve_ref_to_commit
from ira.ingest.provenance import append_jsonl, now_utc_iso
from ira.ingest.seeds import ArxivSeed, DocsSeed, GithubSeed, SeedItem, load_seed_items


def run_ingest(
    *,
    seed_path: Path,
    data_dir: Path,
    manifest_path: Path,
    github_token: Optional[str] = None,
    force: bool = False,
    limit: int = 0,
) -> None:
    items = load_seed_items(seed_path)
    if limit and limit > 0:
        items = items[:limit]

    raw_root = data_dir / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for num, seed in enumerate(items):
            print(f"Ingesting {num}/{len(items)}: {seed}")
            if isinstance(seed, ArxivSeed):
                doc_id = doc_id_for_arxiv(seed)
                out_dir = raw_root / doc_id
                if (out_dir / "meta.json").exists() and not force:
                    continue
                meta = fetch_arxiv(seed, out_dir=out_dir, client=client)
                append_jsonl(manifest_path, {
                    "schema_version": 1,
                    "doc_id": doc_id,
                    "kind": "arxiv",
                    "source_url": meta["source"]["pdf_url"],
                    "retrieved_at": meta["retrieved_at"],
                    "version": f"{seed.arxiv_id}v{seed.arxiv_version}",
                    "artifacts": meta["artifacts"],
                    "tags": seed.tags,
                })

            elif isinstance(seed, DocsSeed):
                doc_id = doc_id_for_docs(seed)
                out_dir = raw_root / doc_id
                if (out_dir / "meta.json").exists() and not force:
                    continue
                meta = fetch_docs(seed, out_dir=out_dir, client=client)
                append_jsonl(manifest_path, {
                    "schema_version": 1,
                    "doc_id": doc_id,
                    "kind": "docs",
                    "source_url": meta["source"]["url"],
                    "resolved_url": meta["source"]["resolved_url"],
                    "retrieved_at": meta["retrieved_at"],
                    "version": seed.version,
                    "http": meta["source"]["http"],
                    "artifacts": meta["artifacts"],
                    "tags": seed.tags,
                })

            elif isinstance(seed, GithubSeed):
                # resolve first so doc_id is commit-stable
                resolved = resolve_ref_to_commit(seed, client=client, github_token=github_token)
                doc_id = doc_id_for_github(resolved.owner, resolved.name, resolved.commit, resolved.path, seed.title)
                out_dir = raw_root / doc_id
                if (out_dir / "meta.json").exists() and not force:
                    continue
                meta = fetch_github(seed, out_dir=out_dir, client=client, github_token=github_token)
                append_jsonl(manifest_path, {
                    "schema_version": 1,
                    "doc_id": doc_id,
                    "kind": "github",
                    "source_url": meta["source"]["raw_url"],
                    "retrieved_at": meta["retrieved_at"],
                    "repo": meta["source"]["repo"],
                    "ref": meta["source"]["ref"],
                    "resolved_commit": meta["source"]["resolved_commit"],
                    "path": meta["source"]["path"],
                    "artifacts": meta["artifacts"],
                    "tags": seed.tags,
                })
            else:
                raise TypeError(f"Unhandled seed type: {type(seed)}")
