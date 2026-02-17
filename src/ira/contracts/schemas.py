from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


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
    schema_version: int = 1
