from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import httpx

from ira.ingest.doc_id import ResolvedGithub
from ira.ingest.provenance import now_utc_iso, sha256_file, write_json
from ira.ingest.seeds import GithubSeed


def _split_repo(repo: str) -> tuple[str, str]:
    if "/" not in repo:
        raise ValueError(f"Invalid repo '{repo}', expected 'owner/name'")
    owner, name = repo.split("/", 1)
    return owner, name


def resolve_ref_to_commit(seed: GithubSeed, *, client: httpx.Client, github_token: Optional[str]) -> ResolvedGithub:
    owner, name = _split_repo(seed.repo)

    headers = {}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    # GitHub API resolves tags/branches/short-sha into a commit object
    api = f"https://api.github.com/repos/{owner}/{name}/commits/{seed.ref}"
    r = client.get(api, headers=headers)
    r.raise_for_status()
    sha = r.json()["sha"]

    raw_url = f"https://raw.githubusercontent.com/{owner}/{name}/{sha}/{seed.path}"
    return ResolvedGithub(owner=owner, name=name, commit=sha, path=seed.path, raw_url=raw_url)


def fetch_github(seed: GithubSeed, *, out_dir: Path, client: httpx.Client, github_token: Optional[str]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_ref_to_commit(seed, client=client, github_token=github_token)

    r = client.get(resolved.raw_url)
    r.raise_for_status()

    basename = Path(seed.path).name or "file.txt"
    file_path = out_dir / basename
    file_path.write_bytes(r.content)

    meta: dict[str, Any] = {
        "kind": "github",
        "retrieved_at": now_utc_iso(),
        "source": {
            "repo": seed.repo,
            "ref": seed.ref,
            "resolved_commit": resolved.commit,
            "path": seed.path,
            "raw_url": resolved.raw_url,
        },
        "seed": seed.model_dump(mode="json"),
        "artifacts": {
            basename: {"sha256": sha256_file(file_path), "bytes": file_path.stat().st_size},
        },
    }

    write_json(out_dir / "meta.json", meta)
    return meta
