from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, HttpUrl


class ArxivSeed(BaseModel):
    kind: Literal["arxiv"] = "arxiv"
    arxiv_id: str  # e.g. "2307.08691"
    arxiv_version: int = 1
    title: str
    tags: list[str] = Field(default_factory=list)


class DocsSeed(BaseModel):
    kind: Literal["docs"] = "docs"
    url: HttpUrl
    version: str  # snapshot label, e.g. "2026-02-17"
    title: str
    tags: list[str] = Field(default_factory=list)


class GithubSeed(BaseModel):
    kind: Literal["github"] = "github"
    repo: str  # "owner/name"
    ref: str   # commit sha OR tag/branch (will be resolved & recorded)
    path: str  # path inside repo
    title: str
    tags: list[str] = Field(default_factory=list)


SeedItem = Annotated[Union[ArxivSeed, DocsSeed, GithubSeed], Field(discriminator="kind")]


def load_seed_items(seed_path: Path) -> list[SeedItem]:
    items: list[SeedItem] = []
    with seed_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
                items.append(SeedItem.__get_pydantic_core_schema__(None, None))  # type: ignore
            except Exception:
                # fallback: pydantic validate in a clean way
                from pydantic import TypeAdapter
                adapter = TypeAdapter(SeedItem)
                items.append(adapter.validate_python(json.loads(line)))
    return items
