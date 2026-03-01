from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ── Day 12: Citation models ───────────────────────────────────────────────────

class CitationRecord(BaseModel):
    """
    Structured citation built from one evidence pack.
    Used internally by citation_mapper — serialised to plain dicts for AgentState.
    """
    citation_id: str                                      # "1", "2", "3" ...
    chunk_id: str                                         # UUID from the index
    doc_id: str                                           # e.g. "arxiv_2205.14135v1"
    doc_title: str                                        # full document title
    url: Optional[str] = None                             # canonical source URL
    section: str = "N/A"                                  # section heading from chunk metadata
    source_type: Literal["internal", "web"] = "internal"
    footnote_label: str = ""                              # "[N] Title — Section (url)"
    position: int = 0                                     # 1-based position in evidence list


class CitationValidationResult(BaseModel):
    """
    Structural citation audit produced by citation_mapper.validate().
    Checks: orphan IDs, citation density. Never blocks the response.
    """
    passed: bool
    total_citations: int = 0              # unique [N] markers found in answer
    valid_citations: int = 0              # [N] that map to a real CitationRecord
    orphan_ids: list[str] = Field(default_factory=list)          # [N] with no CitationRecord
    unused_citation_ids: list[str] = Field(default_factory=list) # records never referenced in answer
    density_ok: bool = True
    density_score: float = 1.0            # cited_sources / total_sources (0.0-1.0)
    min_density_threshold: float = 0.4   # default 40%
    issues: list[str] = Field(default_factory=list)


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
    coverage_score: float = 1.0    # 0.0-1.0; fraction of checks that passed


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
    answer_markdown: str                                              # Day 12: renamed from final_answer
    citations: list[Citation] = Field(default_factory=list)
    used_sources: list[EvidenceChunk] = Field(default_factory=list)  # Day 12: cited subset of evidence
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    citation_validation: Optional[CitationValidationResult] = None   # Day 12: structural audit
    citation_density: float = 0.0                                     # Day 12: cited/retrieved ratio
    trace: list[AgentTraceStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    self_check: Optional[SelfCheckResult] = None
    schema_version: int = 2                                           # Day 12: bumped from 1

    @property
    def final_answer(self) -> str:
        """Backward-compat alias — code using .final_answer still works."""
        return self.answer_markdown