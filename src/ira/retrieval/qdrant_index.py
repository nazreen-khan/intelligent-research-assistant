# File: src/ira/retrieval/qdrant_index.py
"""
Day 5 — Qdrant collection management: create, upsert, deduplicate, query.

Fixes applied (v1.1):
  FIX 1 — Removed `Must` import: qdrant-client >= 1.7 removed the Must/Should
           wrapper classes. Pass conditions list directly to Filter(must=[...]).
  FIX 2 — collection_info() now reads name from self.collection_name instead of
           the collection object, which avoids returning None before first query.
  FIX 3 — Explicit client.close() on QdrantIndex.close() to avoid the
           "sys.meta_path is None" Windows shutdown warning from __del__.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Payload field names (single source of truth) ──────────────────────────────

class PayloadField:
    CHUNK_ID = "chunk_id"
    PARENT_ID = "parent_id"
    DOC_ID = "doc_id"
    TITLE = "title"
    URL = "url"
    SECTION = "section"
    TEXT = "text"
    TOKEN_COUNT = "token_count"
    CHUNK_INDEX = "chunk_index"
    PARENT_CHUNK_INDEX = "parent_chunk_index"
    IS_TABLE = "is_table"
    IS_CODE = "is_code"
    SOURCE_TYPE = "source_type"
    CONTENT_SHA = "content_sha"


def _content_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Qdrant accepts UUID strings as point IDs directly."""
    return chunk_id


# ── Search result dataclass ────────────────────────────────────────────────────

@dataclass
class SearchResult:
    chunk_id: str
    parent_id: str
    doc_id: str
    title: str
    url: Optional[str]
    section: str
    text: str
    score: float
    token_count: int
    is_table: bool
    is_code: bool
    source_type: str


# ── QdrantIndex class ──────────────────────────────────────────────────────────

class QdrantIndex:
    """
    Wraps qdrant-client with production patterns:
      - Embedded or server mode (configured via settings)
      - Idempotent collection creation
      - Batched upsert with SHA-based dedup
      - Vector similarity search with payload filtering
    """

    def __init__(
        self,
        *,
        mode: str = "embedded",
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "ira_chunks",
        vector_dim: int = 384,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        on_disk_payload: bool = False,
    ) -> None:
        self.mode = mode
        self.path = path
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.on_disk_payload = on_disk_payload
        self._client = None

    # ── Client ────────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError(
                "qdrant-client is required. Run: uv add qdrant-client"
            ) from e

        if self.mode == "embedded":
            if not self.path:
                raise ValueError("QDRANT_PATH must be set for embedded mode")
            import os
            os.makedirs(self.path, exist_ok=True)
            logger.info("Qdrant embedded mode: path=%s", self.path)
            self._client = QdrantClient(path=self.path)
        else:
            logger.info("Qdrant server mode: url=%s", self.url)
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30,
            )

        return self._client

    def close(self) -> None:
        """
        Explicitly close the Qdrant client.

        FIX 3: Call this at the end of CLI commands to avoid the Windows warning:
        "sys.meta_path is None, Python is likely shutting down"
        which fires when the GC runs __del__ after interpreter teardown.
        Explicit close() before exit sidesteps that entirely.
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # ── Collection management ─────────────────────────────────────────────────

    def ensure_collection(self) -> bool:
        """
        Create the collection if it doesn't exist.
        Returns True if created, False if already existed.
        Idempotent — safe to call on every startup.
        """
        from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

        client = self._get_client()
        existing = [c.name for c in client.get_collections().collections]

        if self.collection_name in existing:
            logger.info("Collection '%s' already exists", self.collection_name)
            return False

        logger.info(
            "Creating collection '%s' dim=%d hnsw_m=%d ef=%d",
            self.collection_name, self.vector_dim,
            self.hnsw_m, self.hnsw_ef_construct,
        )
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_dim,
                distance=Distance.COSINE,
                on_disk=False,
            ),
            hnsw_config=HnswConfigDiff(
                m=self.hnsw_m,
                ef_construct=self.hnsw_ef_construct,
                full_scan_threshold=10_000,
            ),
            on_disk_payload=self.on_disk_payload,
        )

        self._create_payload_indexes()
        logger.info("Collection '%s' created successfully", self.collection_name)
        return True

    def _create_payload_indexes(self) -> None:
        """Create keyword indexes on frequently-filtered payload fields."""
        from qdrant_client.models import PayloadSchemaType

        client = self._get_client()
        for field, schema_type in [
            (PayloadField.DOC_ID, PayloadSchemaType.KEYWORD),
            (PayloadField.SOURCE_TYPE, PayloadSchemaType.KEYWORD),
            (PayloadField.IS_TABLE, PayloadSchemaType.BOOL),
            (PayloadField.IS_CODE, PayloadSchemaType.BOOL),
        ]:
            try:
                client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema_type,
                )
            except Exception as e:
                logger.debug("Payload index %s: %s", field, e)

    def collection_info(self) -> dict[str, Any]:
        """
        Return basic stats about the collection.

        FIX 2: uses self.collection_name directly so 'name' is never None.
        """
        client = self._get_client()
        try:
            info = client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,          # FIX 2: always correct
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
                "vector_dim": self.vector_dim,
                "mode": self.mode,
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "error": str(e),
            }

    def delete_collection(self) -> None:
        """Delete the collection entirely (use with care)."""
        client = self._get_client()
        client.delete_collection(self.collection_name)
        logger.info("Deleted collection '%s'", self.collection_name)

    # ── Upsert with dedup ─────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        vectors: np.ndarray,
        batch_size: int = 256,
    ) -> dict[str, int]:
        """
        Upsert chunks with their embedding vectors.

        Dedup: skip any chunk where chunk_id already exists AND content_sha matches.
        Returns: {"upserted": N, "skipped": N, "total": N}
        """
        from qdrant_client.models import PointStruct

        assert len(chunks) == len(vectors), "chunks and vectors must be same length"

        if not chunks:
            return {"upserted": 0, "skipped": 0, "total": 0}

        client = self._get_client()

        chunk_shas = [_content_sha(c.get("text", "")) for c in chunks]
        point_ids = [_chunk_id_to_point_id(c["chunk_id"]) for c in chunks]

        # Fetch existing points to check SHA dedup
        existing_map: dict[str, str] = {}
        fetch_batch = 512
        for i in range(0, len(point_ids), fetch_batch):
            batch_ids = point_ids[i: i + fetch_batch]
            try:
                results = client.retrieve(
                    collection_name=self.collection_name,
                    ids=batch_ids,
                    with_payload=[PayloadField.CONTENT_SHA],
                    with_vectors=False,
                )
                for r in results:
                    existing_map[str(r.id)] = (r.payload or {}).get(PayloadField.CONTENT_SHA, "")
            except Exception as e:
                logger.warning("Dedup fetch failed (will upsert all): %s", e)

        # Decide which chunks need upserting
        to_upsert_indices = []
        skipped = 0
        for i, (chunk, sha, pid) in enumerate(zip(chunks, chunk_shas, point_ids)):
            if existing_map.get(pid, "") == sha:
                skipped += 1
            else:
                to_upsert_indices.append(i)

        logger.info("Upsert plan: %d new/changed, %d skipped", len(to_upsert_indices), skipped)

        # Build PointStructs
        points: list[PointStruct] = []
        for i in to_upsert_indices:
            chunk = chunks[i]
            points.append(PointStruct(
                id=_chunk_id_to_point_id(chunk["chunk_id"]),
                vector=vectors[i].tolist(),
                payload={
                    PayloadField.CHUNK_ID:           chunk.get("chunk_id", ""),
                    PayloadField.PARENT_ID:          chunk.get("parent_id", ""),
                    PayloadField.DOC_ID:             chunk.get("doc_id", ""),
                    PayloadField.TITLE:              chunk.get("title", ""),
                    PayloadField.URL:                chunk.get("url"),
                    PayloadField.SECTION:            chunk.get("section", ""),
                    PayloadField.TEXT:               chunk.get("text", ""),
                    PayloadField.TOKEN_COUNT:        chunk.get("token_count", 0),
                    PayloadField.CHUNK_INDEX:        chunk.get("chunk_index", 0),
                    PayloadField.PARENT_CHUNK_INDEX: chunk.get("parent_chunk_index", 0),
                    PayloadField.IS_TABLE:           chunk.get("is_table", False),
                    PayloadField.IS_CODE:            chunk.get("is_code", False),
                    PayloadField.SOURCE_TYPE:        chunk.get("source_type", "internal"),
                    PayloadField.CONTENT_SHA:        chunk_shas[i],
                },
            ))

        total_upserted = 0
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start: batch_start + batch_size]
            client.upsert(collection_name=self.collection_name, points=batch, wait=True)
            total_upserted += len(batch)
            logger.debug("Upserted offset=%d size=%d", batch_start, len(batch))

        logger.info(
            "Upsert complete: upserted=%d skipped=%d total=%d",
            total_upserted, skipped, len(chunks),
        )
        return {"upserted": total_upserted, "skipped": skipped, "total": len(chunks)}

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_doc_ids: Optional[list[str]] = None,
        filter_source_type: Optional[str] = None,
        exclude_code: bool = False,
        exclude_tables: bool = False,
    ) -> list[SearchResult]:
        """
        Search for top-K nearest chunks.

        FIX 1: Filter(must=[...]) — no Must() wrapper, correct for qdrant-client >= 1.7.
        FIX 4: client.search() was removed in qdrant-client >= 1.10.
               Replaced with client.query_points() which is the stable API going forward.
               Response is QueryResponse; iterate response.points (list of ScoredPoint).
        """
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        client = self._get_client()

        # Build conditions list — passed directly to Filter(must=[...])
        conditions = []

        if filter_doc_ids:
            conditions.append(FieldCondition(
                key=PayloadField.DOC_ID,
                match=MatchAny(any=filter_doc_ids),
            ))

        if filter_source_type:
            conditions.append(FieldCondition(
                key=PayloadField.SOURCE_TYPE,
                match=MatchValue(value=filter_source_type),
            ))

        if exclude_code:
            conditions.append(FieldCondition(
                key=PayloadField.IS_CODE,
                match=MatchValue(value=False),
            ))

        if exclude_tables:
            conditions.append(FieldCondition(
                key=PayloadField.IS_TABLE,
                match=MatchValue(value=False),
            ))

        query_filter = Filter(must=conditions) if conditions else None

        # FIX 4: query_points() replaced search() in qdrant-client >= 1.10
        # query=  accepts a plain Python list (the vector)
        # Returns QueryResponse; .points is list[ScoredPoint]
        response = client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold if score_threshold > 0 else None,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        return [
            SearchResult(
                chunk_id=p.get(PayloadField.CHUNK_ID, ""),
                parent_id=p.get(PayloadField.PARENT_ID, ""),
                doc_id=p.get(PayloadField.DOC_ID, ""),
                title=p.get(PayloadField.TITLE, ""),
                url=p.get(PayloadField.URL),
                section=p.get(PayloadField.SECTION, ""),
                text=p.get(PayloadField.TEXT, ""),
                score=r.score,
                token_count=p.get(PayloadField.TOKEN_COUNT, 0),
                is_table=p.get(PayloadField.IS_TABLE, False),
                is_code=p.get(PayloadField.IS_CODE, False),
                source_type=p.get(PayloadField.SOURCE_TYPE, "internal"),
            )
            for r in response.points          # .points replaces iterating response directly
            for p in [(r.payload or {})]
        ]

    def search_by_text(
        self,
        query: str,
        embedder,
        top_k: int = 10,
        **kwargs,
    ) -> list[SearchResult]:
        """Convenience: embed query text then search."""
        query_vec = embedder.embed_query(query)
        return self.search(query_vec, top_k=top_k, **kwargs)


# ── Module-level singleton (lazy) ─────────────────────────────────────────────

_default_index: Optional[QdrantIndex] = None


def get_index(vector_dim: Optional[int] = None) -> QdrantIndex:
    """Return the module-level singleton QdrantIndex, built from settings."""
    global _default_index

    if _default_index is None:
        from ira.settings import settings
        _default_index = QdrantIndex(
            mode=settings.QDRANT_MODE,
            path=str(settings.qdrant_path) if settings.QDRANT_MODE == "embedded" else None,
            url=settings.QDRANT_URL if settings.QDRANT_MODE == "server" else None,
            api_key=settings.QDRANT_API_KEY,
            collection_name=settings.QDRANT_COLLECTION,
            vector_dim=vector_dim or 384,
            hnsw_m=settings.QDRANT_HNSW_M,
            hnsw_ef_construct=settings.QDRANT_HNSW_EF_CONSTRUCT,
            on_disk_payload=settings.QDRANT_ON_DISK_PAYLOAD,
        )

    return _default_index


def reset_index() -> None:
    """Reset the singleton (useful in tests)."""
    global _default_index
    _default_index = None