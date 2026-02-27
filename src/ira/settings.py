from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Core ──────────────────────────────────────────────────────────────────
    app_env: str = "dev"
    log_level: str = "INFO"
    service_name: str = "intelligent-research-assistant"

    # ── Paths ─────────────────────────────────────────────────────────────────
    IRA_DATA_DIR: str = "data"
    GITHUB_TOKEN: str | None = None

    # ── Day 5: Embedding model ────────────────────────────────────────────────
    # BAAI/bge-small-en-v1.5  — 33M params, 384-dim, ~130 MB, CPU-friendly
    # Upgrade path: swap to BAAI/bge-base-en-v1.5 (768-dim) with one env var
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBED_BATCH_SIZE: int = 64          # chunks per forward pass
    EMBED_MAX_LENGTH: int = 512         # tokens; bge-small context window
    EMBED_DEVICE: str = "cpu"           # "cpu" | "cuda" | "mps"
    EMBED_NORMALIZE: bool = True        # L2-normalise → cosine sim = dot product

    # Disk cache for embeddings — SHA256(text) → numpy file
    # Set to empty string "" to disable caching
    EMBED_CACHE_DIR: str = "data/cache/embeddings"

    # ── Day 5: Qdrant ─────────────────────────────────────────────────────────
    # QDRANT_MODE:
    #   "embedded"  — runs in-process, persists to QDRANT_PATH (zero setup)
    #   "server"    — connects to a running Qdrant server (Docker / Cloud)
    QDRANT_MODE: Literal["embedded", "server"] = "embedded"

    # Used only when QDRANT_MODE="embedded"
    QDRANT_PATH: str = "data/qdrant"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None   # set for Qdrant Cloud

    # Collection name — single collection for all chunks
    QDRANT_COLLECTION: str = "ira_chunks"

    # HNSW index parameters (production defaults; good for <1M vectors)
    QDRANT_HNSW_M: int = 16
    QDRANT_HNSW_EF_CONSTRUCT: int = 100
    QDRANT_ON_DISK_PAYLOAD: bool = False   # True → saves RAM for large corpora

    # ── Day 6: BM25 / SQLite FTS5 ─────────────────────────────────────────────
    # Single SQLite file holding the FTS5 full-text index + chunk metadata.
    # Uses SQLite's built-in BM25 ranking via the `bm25()` auxiliary function.
    BM25_DB_PATH: str = "data/bm25/bm25_index.db"

    # Max results returned from BM25 before hybrid merge (Day 7 will fuse these
    # with the dense results — keep this larger than your final top_k)
    BM25_TOP_N: int = 20

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def data_dir(self) -> Path:
        return Path(self.IRA_DATA_DIR)

    @property
    def embed_cache_dir(self) -> Path | None:
        if not self.EMBED_CACHE_DIR:
            return None
        return Path(self.EMBED_CACHE_DIR)

    @property
    def qdrant_path(self) -> Path:
        return Path(self.QDRANT_PATH)

    @property
    def bm25_db_path(self) -> Path:
        return Path(self.BM25_DB_PATH)

    @property
    def caching_enabled(self) -> bool:
        return bool(self.EMBED_CACHE_DIR)


settings = Settings()