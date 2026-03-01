"""
agent/runner.py — Public entry point for the IRA agent.

Usage:
    from ira.agent.runner import run_agent
    response = run_agent(query="How does FlashAttention reduce HBM reads?")

Flow:
    1. Policy gate  — domain + injection check (fast, no LLM)
    2. make_initial_state — build typed AgentState dict
    3. graph.invoke(state) — run the full LangGraph workflow
    4. Map final state → AnswerResponse Pydantic schema
    5. Return AnswerResponse to caller (FastAPI / CLI)

Design:
    - run_agent() never raises — all errors surface as warnings in AnswerResponse
    - request_id generated here if not provided (FastAPI will pass its own)
    - Structured logging at every major step
"""

from __future__ import annotations

import logging
import time
import uuid

from ira.agent.graph import get_graph
from ira.agent.state import AgentState, make_initial_state
from ira.contracts.schemas import (
    AgentTraceStep,
    AnswerResponse,
    Citation,
    CitationIssue,
    EvidenceChunk,
    QueryRequest,
    SelfCheckResult,
)
from ira.policy import check_policy
from ira.observability.logging import log_event

logger = logging.getLogger(__name__)


def run_agent(
    query: str,
    *,
    request_id: str | None = None,
    top_k: int = 5,
    use_web: bool | None = None,
) -> AnswerResponse:
    """
    Run the full IRA agent pipeline for a single query.

    Args:
        query:      Raw user question string.
        request_id: Correlation ID. Auto-generated (UUID4) if not provided.
        top_k:      Number of evidence packs to retrieve (passed to reranker).
        use_web:    Override web search flag. None = use policy default (False).

    Returns:
        AnswerResponse — fully populated Pydantic model.
        Never raises; errors appear in response.warnings.
    """
    rid = request_id or str(uuid.uuid4())
    t_start = time.time()

    log_event(
        logger,
        event="run_agent.start",
        request_id=rid,
        query=query[:120],
    )

    # ── 1. Policy gate ────────────────────────────────────────────────────
    policy = check_policy(query)
    if not policy.allowed:
        log_event(
            logger,
            event="run_agent.blocked",
            request_id=rid,
            check=policy.check,
            reason=policy.reason,
        )
        return AnswerResponse(
            request_id=rid,
            final_answer=f"Query blocked: {policy.reason}",
            citations=[],
            evidence=[],
            trace=[
                AgentTraceStep(
                    step=0,
                    node="policy_gate",
                    status="error",
                    latency_ms=0,
                    detail={"check": policy.check, "reason": policy.reason},
                )
            ],
            warnings=[f"Policy gate blocked this query ({policy.check})."],
        )

    # ── 2. Build initial state ────────────────────────────────────────────
    _use_web = use_web if use_web is not None else False
    state = make_initial_state(
        query=query,
        request_id=rid,
        use_web=_use_web,
    )

    # ── 3. Run LangGraph ──────────────────────────────────────────────────
    try:
        graph = get_graph()
        final_state: AgentState = graph.invoke(state)
    except Exception as exc:
        logger.error("graph.invoke failed: %s", exc, exc_info=True)
        return AnswerResponse(
            request_id=rid,
            final_answer=f"Agent error: {exc}",
            citations=[],
            evidence=[],
            trace=[
                AgentTraceStep(
                    step=0,
                    node="graph",
                    status="error",
                    latency_ms=int((time.time() - t_start) * 1000),
                    detail={"error": str(exc)},
                )
            ],
            warnings=[f"Agent execution error: {exc}"],
        )

    # ── 4. Map state → AnswerResponse ─────────────────────────────────────
    response = _build_response(rid, final_state, t_start)

    elapsed_ms = int((time.time() - t_start) * 1000)
    log_event(
        logger,
        event="run_agent.complete",
        request_id=rid,
        intent=final_state.get("intent"),
        grade=final_state.get("context_grade"),
        retries=final_state.get("retries"),
        evidence_count=len(final_state.get("evidence_packs", [])),
        warnings=len(response.warnings),
        elapsed_ms=elapsed_ms,
    )

    return response


def run_agent_from_request(req: QueryRequest) -> AnswerResponse:
    """
    Convenience wrapper that accepts a QueryRequest Pydantic model.
    Used by the FastAPI POST /query endpoint (Day 14).
    """
    return run_agent(
        query=req.query,
        top_k=req.top_k,
        use_web=req.use_web,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_response(
    request_id: str,
    state: AgentState,
    t_start: float,
) -> AnswerResponse:
    """Map final AgentState fields to AnswerResponse schema."""

    # ── Citations ─────────────────────────────────────────────────────────
    citations = [
        Citation(
            citation_id=c["citation_id"],
            chunk_id=c["chunk_id"],
            label=c["label"],
            url=c.get("url"),
        )
        for c in state.get("citations", [])
    ]

    # ── Evidence chunks (child_text for concise display) ──────────────────
    evidence = [
        EvidenceChunk(
            chunk_id=p["chunk_id"],
            doc_id=p["doc_id"],
            title=p["title"],
            source_type="internal",
            url=p.get("url"),
            section=p.get("section"),
            text=p.get("child_text", ""),
            score=float(p.get("reranker_score") or p.get("rrf_score") or 0.0),
        )
        for p in state.get("evidence_packs", [])
    ]

    # ── Trace ─────────────────────────────────────────────────────────────
    # Step 1: intent classification result
    # Step 2: context grade result
    # Step 3+: one entry per tool_call
    trace: list[AgentTraceStep] = [
        AgentTraceStep(
            step=1,
            node="analyze_intent",
            status="ok",
            latency_ms=0,
            detail={"intent": state.get("intent", "?")},
        ),
        AgentTraceStep(
            step=2,
            node="grade_context",
            status="ok",
            latency_ms=0,
            detail={
                "grade": state.get("context_grade", "?"),
                "retries": state.get("retries", 0),
            },
        ),
    ]

    for i, tc in enumerate(state.get("tool_calls", []), start=3):
        trace.append(
            AgentTraceStep(
                step=i,
                node=tc["tool"],
                status="ok",
                latency_ms=tc.get("ms", 0),
                detail={
                    "input": tc.get("input", "")[:120],
                    "result_count": tc.get("result_count", 0),
                },
            )
        )

    return AnswerResponse(
        request_id=request_id,
        final_answer=state.get("answer_draft", ""),
        citations=citations,
        evidence=evidence,
        trace=trace,
        warnings=state.get("warnings", []),
        self_check=_build_self_check_result(state.get("self_check_result", {})),
    )


def _build_self_check_result(raw: dict) -> SelfCheckResult | None:
    """
    Convert the plain dict written by the self_check node into a typed
    SelfCheckResult Pydantic model.

    Returns None if the dict is empty (self_check node has not run yet —
    e.g. in error paths that short-circuit before the graph runs).
    Returns a SelfCheckResult with defaults when the node ran but produced
    minimal output (e.g. the skip path for empty answers).
    """
    if not raw:
        return None

    citation_issues = [
        CitationIssue(
            citation_id=issue["citation_id"],
            claim_fragment=issue["claim_fragment"],
            issue=issue["issue"],
            severity=issue.get("severity", "warn"),
        )
        for issue in raw.get("citation_issues", [])
    ]

    return SelfCheckResult(
        passed=raw.get("passed", True),
        checks_run=raw.get("checks_run", 0),
        checks_passed=raw.get("checks_passed", 0),
        checks_failed=raw.get("checks_failed", 0),
        citation_issues=citation_issues,
        uncited_sentences=raw.get("uncited_sentences", 0),
        numeric_failures=raw.get("numeric_failures", 0),
        coverage_score=raw.get("coverage_score", 1.0),
    )