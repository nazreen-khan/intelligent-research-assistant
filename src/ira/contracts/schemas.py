from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ── Day 11: Self-check models ─────────────────────────────────────────────────

class CitationIssue(BaseModel):
    """A single problem found by the self_check node for one citation."""
    citation_id: str                          # e.g. "2"
    claim_fragment: str                       # first 120 chars of the flagged claim
    issue: Literal[
        "unsupported",          # claim not grounded in cited source
        "numeric_mismatch",     # a number in the claim is absent from cited text
        "citation_id_not_found" # [N] references a non-existent evidence pack
    ]
    severity: Literal["warn", "error"] = "warn"


class SelfCheckResult(BaseModel):
    """
    Verification summary produced by the self_check node.
    Attached to every AnswerResponse once Day 11 is live.
    """
    passed: bool                                              # True = all checks clean
    checks_run: int = 0                                       # total checks attempted
    checks_passed: int = 0
    checks_failed: int = 0
    citation_issues: list[CitationIssue] = Field(default_factory=list)
    uncited_sentences: int = 0     # factual sentences with no [N] marker
    numeric_failures: int = 0      # numbers in claims absent from cited text
    coverage_score: float = 1.0    # 0.0–1.0; fraction of checks that passed


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    top_k: int = Field(default=5, ge=1, le=20)
    use_web: Optional[bool] = Field(default=None, description="Force web search on/off if set")
    filters: dict[str, Any] = Field(default_factory=dict)


class EvidenceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source_type: Literal["internal", "web"]
    url: Optional[str] = None
    section: Optional[str] = None
    text: str
    score: float = 0.0


class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    label: str
    url: Optional[str] = None


class AgentTraceStep(BaseModel):
    step: int
    node: str
    tool: Optional[str] = None
    status: Literal["ok", "error", "skipped"] = "ok"
    latency_ms: int = 0
    detail: dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    request_id: str
    final_answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    trace: list[AgentTraceStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    self_check: Optional[SelfCheckResult] = None   # Day 11: populated by self_check node
    schema_version: int = 1