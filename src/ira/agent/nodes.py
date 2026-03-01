"""
agent/nodes.py — All 7 LangGraph node functions.

Each node signature:  (state: AgentState) -> dict
The returned dict is MERGED into AgentState by LangGraph — only return
the fields your node actually changes.

Node responsibilities:
  analyze_intent   → set intent ("internal" | "web" | "hybrid")
  route            → pure routing — no state change (returns {})
  retrieve_internal→ call hybrid retriever + reranker, populate evidence_packs
  search_web       → call web search tool, append results to evidence_packs
  grade_context    → set context_grade ("strong" | "weak" | "empty")
  decompose_query  → LLM call: break query into sub-questions, increment retries
  synthesize_answer→ LLM call: generate answer + citations from evidence

Patterns followed (same as rest of codebase):
  - Lazy imports for heavy objects
  - log_event() for structured logging
  - Never raise inside a node — append to warnings and degrade gracefully
  - All timing captured in tool_calls trace
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ira.agent.state import AgentState
from ira.agent import llm as llm_module
from ira.settings import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ms(start: float) -> int:
    """Elapsed milliseconds since start (time.time())."""
    return int((time.time() - start) * 1000)


def _tool_entry(tool: str, input_: str, result_count: int, ms: int) -> dict[str, Any]:
    return {"tool": tool, "input": input_, "result_count": result_count, "ms": ms}


def _pack_to_dict(pack: Any) -> dict[str, Any]:
    """Convert an EvidencePack object to a plain dict for state storage."""
    return {
        "chunk_id":       pack.chunk_id,
        "doc_id":         pack.doc_id,
        "title":          pack.title,
        "url":            pack.url,
        "section":        pack.section,
        "child_text":     pack.child_text,
        "parent_text":    pack.parent_text,
        "rrf_score":      pack.rrf_score,
        "reranker_score": pack.reranker_score,
        "in_both":        pack.in_both,
        "source_type":    "internal",
    }


def _dedup_packs(
    existing: list[dict[str, Any]],
    new_packs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge new packs into existing, deduplicating by chunk_id."""
    seen = {p["chunk_id"] for p in existing}
    merged = list(existing)
    for p in new_packs:
        if p["chunk_id"] not in seen:
            merged.append(p)
            seen.add(p["chunk_id"])
    return merged


def _format_evidence_for_llm(evidence_packs: list[dict[str, Any]]) -> str:
    """
    Format evidence packs into a numbered context block for LLM prompts.
    Uses parent_text (full section ~1500 tokens) for richer context.
    """
    if not evidence_packs:
        return "No evidence available."

    blocks: list[str] = []
    for i, pack in enumerate(evidence_packs, 1):
        section = pack.get("section") or "—"
        source_tag = f" [{pack.get('source_type', 'internal').upper()}]"
        blocks.append(
            f"[{i}]{source_tag} Source: {pack['title']} | Section: {section}\n"
            f"URL: {pack.get('url', 'N/A')}\n"
            f"chunk_id: {pack['chunk_id']}\n\n"
            f"{pack['parent_text']}"
        )
    return "\n\n---\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: analyze_intent
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_SYSTEM = (
    "You are a query router for a research assistant specialising in LLM efficiency. "
    "Classify the user query into exactly one of these intents:\n"
    "  internal — answerable from a corpus of LLM efficiency papers and docs\n"
    "  web      — requires latest/real-time info (e.g. 'latest vLLM release', 'current SOTA')\n"
    "  hybrid   — needs both corpus knowledge AND latest web info\n\n"
    "Reply with ONLY one word: internal, web, or hybrid."
)


def analyze_intent(state: AgentState) -> dict:
    """
    Classify query intent using a fast LLM call.
    Falls back to 'internal' on any error.
    """
    query = state["query"]
    logger.debug("analyze_intent", extra={"request_id": state["request_id"], "query": query})

    try:
        raw = llm_module.call(
            prompt=f"Query: {query}",
            system=_INTENT_SYSTEM,
            max_tokens=5,
            temperature=0.0,
        ).strip().lower()

        intent = raw if raw in ("internal", "web", "hybrid") else "internal"
    except Exception as exc:
        logger.warning("analyze_intent failed, defaulting to internal: %s", exc)
        intent = "internal"

    logger.info(
        "intent classified",
        extra={"request_id": state["request_id"], "intent": intent},
    )
    return {"intent": intent}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: route
# ─────────────────────────────────────────────────────────────────────────────

def route(state: AgentState) -> dict:
    """
    Pure routing node — no state mutation.
    The conditional edge in graph.py reads state['intent'] directly.
    This node exists so the graph has an explicit named routing step in traces.
    """
    logger.debug(
        "route",
        extra={"request_id": state["request_id"], "intent": state["intent"]},
    )
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: retrieve_internal
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_internal(state: AgentState) -> dict:
    """
    Call hybrid retriever + reranker for each query/subquery.
    Deduplicates results against already-collected evidence_packs.
    Records tool_calls trace entry.
    """
    from ira.retrieval.rerank_runner import query_with_rerank  # lazy import

    queries_to_run: list[str] = state["subqueries"] if state["subqueries"] else [state["query"]]

    new_packs: list[dict[str, Any]] = []
    tool_entries: list[dict[str, Any]] = []

    for q in queries_to_run:
        t0 = time.time()
        try:
            result = query_with_rerank(
                query_text=q,
                keep_k=settings.RERANKER_KEEP_K,
            )
            packs_as_dicts = [_pack_to_dict(p) for p in result.evidence_packs]
            new_packs.extend(packs_as_dicts)
            tool_entries.append(
                _tool_entry("retrieve_internal", q, len(packs_as_dicts), _ms(t0))
            )
            logger.info(
                "retrieve_internal",
                extra={
                    "request_id": state["request_id"],
                    "query": q,
                    "packs": len(packs_as_dicts),
                    "ms": _ms(t0),
                },
            )
        except Exception as exc:
            logger.error("retrieve_internal error: %s", exc, exc_info=True)
            tool_entries.append(_tool_entry("retrieve_internal", q, 0, _ms(t0)))

    merged = _dedup_packs(state["evidence_packs"], new_packs)

    return {
        "evidence_packs": merged,
        "tool_calls": state["tool_calls"] + tool_entries,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: search_web  (Day 10 — real implementation)
# ─────────────────────────────────────────────────────────────────────────────

def search_web(state: AgentState) -> dict:
    """
    Call the web search tool and append results to evidence_packs.

    Uses WEB_SEARCH_PROVIDER from settings:
      "mock"   — loads fixtures, no API key needed (default, CI-safe)
      "tavily" — Tavily Search API
      "exa"    — Exa Search API

    Web results are converted to evidence_pack dicts (source_type="web")
    and deduped by chunk_id before merging — synthesize_answer handles
    them identically to internal packs.
    """
    from ira.agent.web_search_tool import search_web_tool, web_results_to_evidence_packs  # lazy

    query = state["query"]
    t0 = time.time()

    logger.info(
        "search_web",
        extra={
            "request_id": state["request_id"],
            "query": query[:80],
            "provider": settings.WEB_SEARCH_PROVIDER,
        },
    )

    warnings = list(state["warnings"])

    try:
        web_results = search_web_tool(
            query=query,
            max_results=settings.WEB_SEARCH_MAX_RESULTS,
        )
    except Exception as exc:
        logger.error("search_web error: %s", exc, exc_info=True)
        warnings.append(f"Web search failed: {exc}")
        return {
            "tool_calls": state["tool_calls"] + [
                _tool_entry("search_web", query, 0, _ms(t0))
            ],
            "warnings": warnings,
        }

    if not web_results:
        warnings.append(f"Web search returned no results for: {query[:80]}")

    # Convert to evidence_pack dicts and merge
    new_packs = web_results_to_evidence_packs(web_results)
    merged = _dedup_packs(state["evidence_packs"], new_packs)

    elapsed = _ms(t0)
    logger.info(
        "search_web complete",
        extra={
            "request_id": state["request_id"],
            "results": len(web_results),
            "provider": settings.WEB_SEARCH_PROVIDER,
            "ms": elapsed,
        },
    )

    return {
        "evidence_packs": merged,
        "tool_calls": state["tool_calls"] + [
            _tool_entry("search_web", query, len(web_results), elapsed)
        ],
        "warnings": warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: grade_context
# ─────────────────────────────────────────────────────────────────────────────

def grade_context(state: AgentState) -> dict:
    """
    Score the collected evidence and assign context_grade.

    Rules (no LLM — deterministic, free):
      "empty"  → no evidence packs at all
      "weak"   → fewer than AGENT_MIN_EVIDENCE_PACKS, OR
                 top reranker_score < AGENT_WEAK_SCORE_THRESHOLD
      "strong" → everything else
    """
    packs = state["evidence_packs"]

    if not packs:
        grade = "empty"
    else:
        top_score = max(
            (p.get("reranker_score") or p.get("rrf_score") or 0.0 for p in packs),
            default=0.0,
        )
        too_few = len(packs) < settings.AGENT_MIN_EVIDENCE_PACKS
        low_score = top_score < settings.AGENT_WEAK_SCORE_THRESHOLD

        grade = "weak" if (too_few or low_score) else "strong"

    logger.info(
        "grade_context",
        extra={
            "request_id": state["request_id"],
            "grade": grade,
            "pack_count": len(packs),
            "retries": state["retries"],
        },
    )
    return {"context_grade": grade}


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: decompose_query
# ─────────────────────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = (
    "You are a research query decomposer. "
    "Break the given question into 2 or 3 focused sub-questions that together "
    "cover the full answer. Each sub-question must be self-contained and specific.\n\n"
    "Reply with ONLY a Python list of strings, no explanation. Example:\n"
    '["What is X?", "How does Y work?", "What are the tradeoffs of Z?"]'
)


def decompose_query(state: AgentState) -> dict:
    """
    Break the original query into 2-3 sub-questions via LLM.
    Increments retries counter to prevent infinite loops.
    Falls back to [original_query] on parse failure.
    """
    query = state["query"]
    logger.info(
        "decompose_query",
        extra={"request_id": state["request_id"], "retry": state["retries"] + 1},
    )

    try:
        raw = llm_module.call(
            prompt=f"Question to decompose: {query}",
            system=_DECOMPOSE_SYSTEM,
            max_tokens=200,
            temperature=0.2,
        ).strip()

        import ast
        subqueries: list[str] = ast.literal_eval(raw)
        if not isinstance(subqueries, list) or not subqueries:
            raise ValueError("not a non-empty list")
        subqueries = [str(s).strip() for s in subqueries if str(s).strip()]

    except Exception as exc:
        logger.warning("decompose_query parse failed (%s), using original query", exc)
        subqueries = [query]

    logger.info(
        "subqueries",
        extra={"request_id": state["request_id"], "subqueries": subqueries},
    )
    return {
        "subqueries": subqueries,
        "retries": state["retries"] + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 7: synthesize_answer
# ─────────────────────────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = (
    "You are a precise research assistant specialising in LLM efficiency, "
    "optimisation, and evaluation. "
    "Answer the question using ONLY the provided evidence. "
    "For every factual claim, cite the source using its number [1], [2], etc. "
    "If the evidence does not fully answer the question, say so explicitly. "
    "Be concise and technical. Use markdown formatting."
)


def synthesize_answer(state: AgentState) -> dict:
    """
    Generate the final answer from collected evidence using the LLM.
    Builds citations list from evidence_packs (chunk_id → label + url).
    Appends a warning if context_grade was not 'strong'.
    """
    packs = state["evidence_packs"]
    query = state["query"]
    warnings = list(state["warnings"])

    if not packs:
        answer = (
            "I could not find sufficient evidence in the corpus to answer this question. "
            "Try rephrasing or enabling web search."
        )
        warnings.append("No evidence packs available for synthesis.")
        return {"answer_draft": answer, "citations": [], "warnings": warnings}

    if state["context_grade"] in ("weak", "empty"):
        warnings.append(
            f"Context grade was '{state['context_grade']}' after "
            f"{state['retries']} retry(ies). Answer may be incomplete."
        )

    evidence_text = _format_evidence_for_llm(packs)
    prompt = (
        f"Question: {query}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Answer (cite sources as [1], [2], etc.):"
    )

    t0 = time.time()
    try:
        answer_draft = llm_module.call(
            prompt=prompt,
            system=_SYNTHESIS_SYSTEM,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    except Exception as exc:
        logger.error("synthesize_answer LLM call failed: %s", exc, exc_info=True)
        answer_draft = f"Synthesis failed due to an error: {exc}"
        warnings.append(f"LLM synthesis error: {exc}")

    logger.info(
        "synthesize_answer",
        extra={
            "request_id": state["request_id"],
            "ms": _ms(t0),
            "evidence_count": len(packs),
        },
    )

    # Build citations — tag web sources clearly in the label
    citations: list[dict[str, Any]] = []
    for i, pack in enumerate(packs, 1):
        source_tag = " [WEB]" if pack.get("source_type") == "web" else ""
        citations.append({
            "citation_id": str(i),
            "chunk_id":    pack["chunk_id"],
            "label":       f"[{i}]{source_tag} {pack['title']} — {pack.get('section') or 'N/A'}",
            "url":         pack.get("url"),
        })

    return {
        "answer_draft": answer_draft,
        "citations":    citations,
        "warnings":     warnings,
    }