# File: src/ira/retrieval/parent_store.py
"""
Day 7 — Parent document store.

Each doc_id directory in data/chunks/ contains a parents.jsonl file with
the full parent section text (~1500 tokens). Child chunks reference their
parent via parent_id.

This module loads those files lazily and caches them in RAM so that the
hybrid retriever can expand child matches to full parent context without
re-reading disk on every query.

Parent record shape (from chunker.py Day 3):
    {
        "parent_id":   "arxiv_2205.14135v1::parent::12",
        "doc_id":      "arxiv_2205.14135v1",
        "title":       "FlashAttention: Fast Memory-Efficient...",
        "url":         "https://arxiv.org/abs/2205.14135",
        "section":     "3. FlashAttention",
        "text":        "<full parent section text ~1500 tokens>",
        "token_count": 1487,
        "child_ids":   ["..::chunk::24", "..::chunk::25", "..::chunk::26"]
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ParentStore:
    """
    In-memory cache of parent documents, loaded from data/chunks/*/parents.jsonl.

    Loading is lazy per doc_id and happens on first access.
    The full cache is built via load_all() before serving queries.
    """

    def __init__(self, chunks_root: str | Path) -> None:
        self.chunks_root = Path(chunks_root)
        # parent_id → parent record dict
        self._cache: dict[str, dict[str, Any]] = {}
        self._loaded_docs: set[str] = set()

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_all(self) -> dict[str, int]:
        """
        Load all parents.jsonl files under chunks_root into RAM.
        Returns {doc_id: parent_count} for logging.
        """
        counts: dict[str, int] = {}
        if not self.chunks_root.exists():
            logger.warning("chunks_root does not exist: %s", self.chunks_root)
            return counts

        for doc_dir in sorted(self.chunks_root.iterdir()):
            if not doc_dir.is_dir():
                continue
            parents_file = doc_dir / "parents.jsonl"
            if not parents_file.exists():
                continue
            n = self._load_doc(doc_dir.name, parents_file)
            counts[doc_dir.name] = n

        logger.info(
            "ParentStore loaded %d docs, %d parents total: %s",
            len(counts),
            sum(counts.values()),
            counts,
        )
        return counts

    def load_doc(self, doc_id: str) -> int:
        """Lazily load a single doc's parents. Returns count loaded."""
        if doc_id in self._loaded_docs:
            return 0
        parents_file = self.chunks_root / doc_id / "parents.jsonl"
        if not parents_file.exists():
            logger.warning("No parents.jsonl for doc_id=%s", doc_id)
            return 0
        return self._load_doc(doc_id, parents_file)

    def _load_doc(self, doc_id: str, parents_file: Path) -> int:
        count = 0
        with parents_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    pid = record.get("parent_id")
                    if pid:
                        self._cache[pid] = record
                        count += 1
                    else:
                        logger.warning(
                            "parents.jsonl %s line %d: missing parent_id",
                            parents_file, line_no,
                        )
                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON in %s line %d: %s", parents_file, line_no, e)
        self._loaded_docs.add(doc_id)
        logger.debug("Loaded %d parents from %s", count, doc_id)
        return count

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, parent_id: str, doc_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        """
        Fetch a parent record by parent_id.
        If not in cache and doc_id provided, attempts a lazy load first.
        """
        if parent_id in self._cache:
            return self._cache[parent_id]
        if doc_id and doc_id not in self._loaded_docs:
            self.load_doc(doc_id)
            return self._cache.get(parent_id)
        return None

    def get_text(
        self,
        parent_id: str,
        doc_id: Optional[str] = None,
        max_chars: int = 6000,
    ) -> str:
        """Return parent text, truncated to max_chars. Returns '' if not found."""
        record = self.get(parent_id, doc_id=doc_id)
        if record is None:
            return ""
        text = record.get("text", "")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…[truncated]"
        return text

    def get_many(self, parent_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch multiple parents. Missing ones are silently omitted."""
        return {pid: self._cache[pid] for pid in parent_ids if pid in self._cache}

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def loaded_docs(self) -> set[str]:
        return set(self._loaded_docs)

    def stats(self) -> dict[str, Any]:
        doc_counts: dict[str, int] = {}
        for record in self._cache.values():
            did = record.get("doc_id", "unknown")
            doc_counts[did] = doc_counts.get(did, 0) + 1
        return {
            "total_parents": self.size,
            "docs_loaded": len(self._loaded_docs),
            "parents_per_doc": doc_counts,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_default_store: Optional[ParentStore] = None


def get_parent_store(chunks_root: Optional[str | Path] = None) -> ParentStore:
    """Return the module-level singleton ParentStore, loading all docs on first call."""
    global _default_store
    if _default_store is None:
        if chunks_root is None:
            from ira.settings import settings
            chunks_root = settings.data_dir / "chunks"
        store = ParentStore(chunks_root)
        store.load_all()
        _default_store = store
    return _default_store


def reset_parent_store() -> None:
    """Reset singleton — for use in tests."""
    global _default_store
    _default_store = None