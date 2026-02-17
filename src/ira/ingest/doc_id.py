from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ira.ingest.seeds import ArxivSeed, DocsSeed, GithubSeed


_slug_re = re.compile(r"[^a-z0-9]+", re.IGNORECASE)


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = _slug_re.sub("-", s)
    return s.strip("-")[:60] or "doc"


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def doc_id_for_arxiv(seed: ArxivSeed) -> str:
    # deterministic, human-readable
    return f"arxiv_{seed.arxiv_id}v{seed.arxiv_version}"


def doc_id_for_docs(seed: DocsSeed) -> str:
    key = f"{seed.url}|{seed.version}"
    return f"docs_{_slug(seed.url.host or 'site')}_{_sha1(key)[:12]}"


def doc_id_for_github(owner: str, name: str, commit: str, path: str, title: str) -> str:
    key = f"{owner}/{name}@{commit}:{path}"
    return f"gh_{_slug(owner)}_{_slug(name)}_{_sha1(key)[:12]}"


@dataclass(frozen=True)
class ResolvedGithub:
    owner: str
    name: str
    commit: str
    path: str
    raw_url: str
