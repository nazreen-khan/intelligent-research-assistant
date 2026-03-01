"""
agent/graph.py — LangGraph state machine.

Graph topology:
    START
      └─► analyze_intent
            └─► route
                  ├─► retrieve_internal   (intent == "internal")
                  ├─► search_web          (intent == "web")
                  └─► retrieve_internal   (intent == "hybrid" — web added after grading)
                        └─► grade_context
                              ├─► synthesize_answer   (grade == "strong")
                              ├─► decompose_query     (grade == "weak" AND retries < max)
                              │     └─► retrieve_internal  (loop back)
                              ├─► search_web          (grade == "weak" AND use_web AND retries >= max)
                              │     └─► synthesize_answer
                              └─► synthesize_answer   (grade == "empty" OR retries >= max)
                                    └─► self_check    (Day 11 — ALL paths go through self_check)
                                          └─► END

Key design decisions:
  - route() node is a no-op; conditional edge reads state["intent"] directly.
  - grade_context conditional edge encodes the full loop/exit logic.
  - MAX_RETRIES guard prevents infinite decompose→retrieve cycles.
  - search_web is always available as a fallback even when intent=="internal"
    if use_web=True and context stays weak after max retries.
  - self_check (Day 11) sits between synthesize_answer and END on all paths.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from ira.agent.state import AgentState
from ira.agent.nodes import (
    analyze_intent,
    decompose_query,
    grade_context,
    retrieve_internal,
    route,
    search_web,
    self_check,
    synthesize_answer,
)
from ira.settings import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge functions
# (return the NAME of the next node as a string)
# ─────────────────────────────────────────────────────────────────────────────

def _route_by_intent(state: AgentState) -> str:
    """After route node: dispatch based on classified intent."""
    intent = state["intent"]
    if intent == "web":
        return "search_web"
    # "internal" and "hybrid" both start with internal retrieval
    return "retrieve_internal"


def _route_by_grade(state: AgentState) -> str:
    """
    After grade_context: decide whether to loop, fallback, or synthesize.

    Logic:
      strong                          → synthesize_answer
      weak/empty + retries < max      → decompose_query  (loop)
      weak/empty + retries >= max
            + use_web == True         → search_web       (web fallback)
      weak/empty + retries >= max
            + use_web == False        → synthesize_answer (degrade gracefully)
    """
    grade = state["context_grade"]
    retries = state["retries"]
    use_web = state["use_web"]
    max_retries = settings.AGENT_MAX_RETRIES

    if grade == "strong":
        return "synthesize_answer"

    # grade is "weak" or "empty"
    if retries < max_retries:
        return "decompose_query"

    # Exhausted retries
    if use_web:
        return "search_web"

    return "synthesize_answer"


def _after_web_search(state: AgentState) -> str:
    """
    After search_web: always go to synthesize_answer.
    (Web results — real or stub — are the last retrieval step.)
    """
    return "synthesize_answer"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Returns a compiled graph ready for graph.invoke(state).
    Call this once and reuse — compilation is not free.
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("analyze_intent", analyze_intent)
    builder.add_node("route", route)
    builder.add_node("retrieve_internal", retrieve_internal)
    builder.add_node("search_web", search_web)
    builder.add_node("grade_context", grade_context)
    builder.add_node("decompose_query", decompose_query)
    builder.add_node("synthesize_answer", synthesize_answer)
    builder.add_node("self_check", self_check)   # Day 11

    # ── Entry point ───────────────────────────────────────────────────────
    builder.add_edge(START, "analyze_intent")
    builder.add_edge("analyze_intent", "route")

    # ── Intent routing (after route node) ────────────────────────────────
    builder.add_conditional_edges(
        "route",
        _route_by_intent,
        {
            "retrieve_internal": "retrieve_internal",
            "search_web": "search_web",
        },
    )

    # ── After internal retrieval → always grade ───────────────────────────
    builder.add_edge("retrieve_internal", "grade_context")

    # ── Grade routing (core loop logic) ──────────────────────────────────
    builder.add_conditional_edges(
        "grade_context",
        _route_by_grade,
        {
            "synthesize_answer": "synthesize_answer",
            "decompose_query": "decompose_query",
            "search_web": "search_web",
        },
    )

    # ── Decompose loops back to retrieval ─────────────────────────────────
    builder.add_edge("decompose_query", "retrieve_internal")

    # ── Web search → synthesize ───────────────────────────────────────────
    builder.add_conditional_edges(
        "search_web",
        _after_web_search,
        {"synthesize_answer": "synthesize_answer"},
    )

    # ── synthesize_answer → self_check → END  (Day 11: all paths verified) ──
    builder.add_edge("synthesize_answer", "self_check")
    builder.add_edge("self_check", END)

    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_graph: StateGraph | None = None


def get_graph() -> StateGraph:
    """
    Return the compiled graph singleton.
    Builds once on first call; subsequent calls return cached instance.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
        logger.debug("LangGraph compiled and cached")
    return _graph


def reset_graph() -> None:
    """Force rebuild on next get_graph() call. Used in tests."""
    global _graph
    _graph = None