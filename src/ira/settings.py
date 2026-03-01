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
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBED_BATCH_SIZE: int = 64
    EMBED_MAX_LENGTH: int = 512
    EMBED_DEVICE: str = "cpu"
    EMBED_NORMALIZE: bool = True
    EMBED_CACHE_DIR: str = "data/cache/embeddings"

    # ── Day 5: Qdrant ─────────────────────────────────────────────────────────
    QDRANT_MODE: Literal["embedded", "server"] = "embedded"
    QDRANT_PATH: str = "data/qdrant"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "ira_chunks"
    QDRANT_HNSW_M: int = 16
    QDRANT_HNSW_EF_CONSTRUCT: int = 100
    QDRANT_ON_DISK_PAYLOAD: bool = False

    # ── Day 6: BM25 / SQLite FTS5 ─────────────────────────────────────────────
    BM25_DB_PATH: str = "data/bm25/bm25_index.db"
    BM25_TOP_N: int = 20

    # ── Day 8: Cross-encoder reranker ─────────────────────────────────────────
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_DEVICE: str = "cpu"
    RERANKER_BATCH_SIZE: int = 16
    RERANKER_TOP_N: int = 20
    RERANKER_KEEP_K: int = 5
    RERANKER_CACHE_DIR: str = "data/cache/rerank"

    # ── Day 9: LLM ────────────────────────────────────────────────────────────
    LLM_PROVIDER: Literal["anthropic", "openai"] = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_API_KEY: str = ""
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.1

    # ── Day 9: Agent ──────────────────────────────────────────────────────────
    AGENT_MAX_RETRIES: int = 2
    AGENT_WEAK_SCORE_THRESHOLD: float = 0.3
    AGENT_MIN_EVIDENCE_PACKS: int = 2

    # ── Day 10: Web search ────────────────────────────────────────────────────
    # Provider: "tavily" | "exa" | "mock"
    # "mock" uses data/fixtures/web_mock.jsonl — no API key needed (CI/tests)
    # "tavily" free tier: 1000 searches/month — recommended for dev
    # Switch provider with one env var — zero code changes.
    WEB_SEARCH_PROVIDER: Literal["tavily", "exa", "mock"] = "mock"
    WEB_SEARCH_API_KEY: str = ""            # set in .env — TAVILY_API_KEY or EXA_API_KEY
    WEB_SEARCH_MAX_RESULTS: int = 5         # results returned per query
    WEB_SEARCH_MAX_AGE_DAYS: int = 90       # skip results older than this (0 = no filter)

    # TTL disk cache — same query within this window hits disk, not the API
    # Set to "" to disable caching (not recommended — burns API quota fast)
    WEB_SEARCH_CACHE_DIR: str = "data/cache/web"
    WEB_SEARCH_CACHE_TTL_HOURS: int = 24

    # Domain allowlist — comma-separated, empty string = allow all domains
    # Example: "arxiv.org,huggingface.co,github.com,docs.nvidia.com"
    # Tighten this in production to keep results on-topic
    WEB_SEARCH_DOMAIN_ALLOWLIST: str = ""

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def reranker_cache_dir(self) -> Path | None:
        if not self.RERANKER_CACHE_DIR:
            return None
        return Path(self.RERANKER_CACHE_DIR)

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

    @property
    def web_search_cache_dir(self) -> Path | None:
        if not self.WEB_SEARCH_CACHE_DIR:
            return None
        return Path(self.WEB_SEARCH_CACHE_DIR)

    @property
    def web_domain_allowlist(self) -> list[str]:
        """Returns parsed domain allowlist. Empty list = allow all."""
        if not self.WEB_SEARCH_DOMAIN_ALLOWLIST.strip():
            return []
        return [d.strip() for d in self.WEB_SEARCH_DOMAIN_ALLOWLIST.split(",") if d.strip()]


settings = Settings()