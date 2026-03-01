"""
agent/state.py — AgentState definition for the LangGraph workflow.

All fields use plain Python types (str, list, dict, int) so that:
  - LangGraph can checkpoint/serialize the state without custom serializers
  - Tests can construct state with simple dict literals
  - No circular imports (Pydantic schemas imported only in runner.py)

Field groups:
  INPUT       — set once by the caller, never mutated by nodes
  ROUTING     — set by analyze_intent, read by route
  RETRIEVAL   — accumulated by retrieve_internal / search_web
  CONTROL     — loop counters and grading signals
  OUTPUT      — populated by synthesize_answer + self_check (Day 11)
"""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── INPUT ─────────────────────────────────────────────────────────────────
    request_id: str          # correlation ID propagated from caller
    query: str               # original user query (never mutated)
    use_web: bool            # True = web search allowed; False = internal only

    # ── ROUTING ───────────────────────────────────────────────────────────────
    # Set by analyze_intent node.
    # "internal"  → go straight to retrieve_internal
    # "web"       → go straight to search_web
    # "hybrid"    → retrieve_internal first, then search_web if still weak
    intent: str              # "internal" | "web" | "hybrid"

    # ── RETRIEVAL ─────────────────────────────────────────────────────────────
    # subqueries: populated by decompose_query when context is weak.
    # Empty list = use original query as-is.
    subqueries: list[str]

    # evidence_packs: list of serialised EvidencePack dicts.
    # Each dict mirrors EvidencePack fields:
    #   chunk_id, doc_id, title, url, section,
    #   child_text, parent_text, rrf_score, reranker_score, in_both
    # Accumulated across retries (deduped by chunk_id in retrieve_internal).
    evidence_packs: list[dict[str, Any]]

    # ── CONTROL ───────────────────────────────────────────────────────────────
    # context_grade: set by grade_context node after each retrieval pass.
    # "strong" → proceed to synthesize_answer
    # "weak"   → trigger decompose_query (if retries < max)
    # "empty"  → no evidence at all (skip to synthesize with warning)
    context_grade: str       # "strong" | "weak" | "empty"

    retries: int             # incremented by decompose_query; guards loop exit

    # tool_calls: append-only trace of every tool invocation.
    # Each entry: {"tool": str, "input": str, "result_count": int, "ms": int}
    tool_calls: list[dict[str, Any]]

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    # answer_draft: raw markdown string from synthesize_answer.
    answer_draft: str

    # citations: list of citation dicts built during synthesis.
    # Each dict: {"citation_id": str, "chunk_id": str, "label": str, "url": str}
    citations: list[dict[str, Any]]

    # warnings: human-readable notes appended by any node.
    # Examples: "context grade: weak after 2 retries", "web search stubbed"
    warnings: list[str]

    # ── DAY 11: SELF-CHECK ────────────────────────────────────────────────────
    # self_check_result: verification summary written by the self_check node.
    # Shape mirrors SelfCheckResult Pydantic model (plain dict for LangGraph compat):
    #   passed, checks_run, checks_passed, checks_failed,
    #   citation_issues, uncited_sentences, numeric_failures, coverage_score
    # Empty dict {} until self_check node runs.
    self_check_result: dict[str, Any]


def make_initial_state(
    query: str,
    request_id: str,
    use_web: bool = False,
) -> AgentState:
    """
    Build a fully-initialised AgentState for a new request.
    All optional fields set to safe defaults so nodes never KeyError.

    Args:
        query:      Raw user query string.
        request_id: Correlation ID (pass from FastAPI request or generate).
        use_web:    Whether the agent may call the web search tool.

    Returns:
        AgentState dict ready to pass to graph.invoke().
    """
    return AgentState(
        request_id=request_id,
        query=query,
        use_web=use_web,
        intent="internal",
        subqueries=[],
        evidence_packs=[],
        context_grade="empty",
        retries=0,
        tool_calls=[],
        answer_draft="",
        citations=[],
        warnings=[],
        self_check_result={},   # Day 11: populated by self_check node
    )