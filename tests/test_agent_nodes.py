"""
tests/test_agent_nodes.py — Unit tests for all 7 agent nodes.

Strategy:
- Every test mocks the LLM (no real API calls)
- retrieve_internal mocks query_with_rerank (no real index needed)
- Each node is tested in isolation with a hand-crafted AgentState
- Tests run fast (<1s total) and work offline

Run:
    uv run pytest tests/test_agent_nodes.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ira.agent.state import make_initial_state


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _base_state(**overrides) -> dict:
    """Build a minimal AgentState dict, with optional field overrides."""
    s = make_initial_state(
        query="How does FlashAttention reduce HBM reads?",
        request_id="test-req-001",
        use_web=False,
    )
    s.update(overrides)
    return s


def _fake_pack(chunk_id: str = "chunk-001", reranker_score: float = 0.8) -> MagicMock:
    """Return a mock EvidencePack object matching the fields _pack_to_dict expects."""
    pack = MagicMock()
    pack.chunk_id = chunk_id
    pack.doc_id = "arxiv_test"
    pack.title = "Test Paper"
    pack.url = "https://arxiv.org/test"
    pack.section = "3.1 Algorithm"
    pack.child_text = "FlashAttention uses tiling to reduce HBM reads."
    pack.parent_text = "Full section text about FlashAttention tiling algorithm."
    pack.rrf_score = 0.05
    pack.reranker_score = reranker_score
    pack.in_both = True
    return pack


def _fake_rerank_result(n_packs: int = 2) -> MagicMock:
    """Return a mock RetrievalResult with n fake EvidencePacks."""
    result = MagicMock()
    result.evidence_packs = [
        _fake_pack(f"chunk-{i:03d}", reranker_score=0.9 - i * 0.1)
        for i in range(n_packs)
    ]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: analyze_intent
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeIntent:

    def test_returns_internal_for_corpus_query(self):
        from ira.agent.nodes import analyze_intent
        with patch("ira.agent.llm.call", return_value="internal"):
            result = analyze_intent(_base_state())
        assert result["intent"] == "internal"

    def test_returns_web_for_latest_query(self):
        from ira.agent.nodes import analyze_intent
        with patch("ira.agent.llm.call", return_value="web"):
            result = analyze_intent(_base_state(query="Latest vLLM benchmark 2025"))
        assert result["intent"] == "web"

    def test_returns_hybrid_when_llm_says_hybrid(self):
        from ira.agent.nodes import analyze_intent
        with patch("ira.agent.llm.call", return_value="hybrid"):
            result = analyze_intent(_base_state())
        assert result["intent"] == "hybrid"

    def test_defaults_to_internal_on_invalid_llm_output(self):
        from ira.agent.nodes import analyze_intent
        with patch("ira.agent.llm.call", return_value="UNKNOWN_VALUE"):
            result = analyze_intent(_base_state())
        assert result["intent"] == "internal"

    def test_defaults_to_internal_on_llm_exception(self):
        from ira.agent.nodes import analyze_intent
        with patch("ira.agent.llm.call", side_effect=RuntimeError("API error")):
            result = analyze_intent(_base_state())
        assert result["intent"] == "internal"

    def test_only_returns_intent_key(self):
        from ira.agent.nodes import analyze_intent
        with patch("ira.agent.llm.call", return_value="internal"):
            result = analyze_intent(_base_state())
        assert set(result.keys()) == {"intent"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: route
# ─────────────────────────────────────────────────────────────────────────────

class TestRoute:

    def test_returns_empty_dict(self):
        from ira.agent.nodes import route
        result = route(_base_state(intent="internal"))
        assert result == {}

    def test_does_not_mutate_state(self):
        from ira.agent.nodes import route
        state = _base_state(intent="web")
        route(state)
        assert state["intent"] == "web"  # unchanged


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: retrieve_internal
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieveInternal:

    def test_populates_evidence_packs(self):
        from ira.agent.nodes import retrieve_internal
        with patch("ira.retrieval.rerank_runner.query_with_rerank",
                   return_value=_fake_rerank_result(2)):
            result = retrieve_internal(_base_state())
        assert len(result["evidence_packs"]) == 2

    def test_uses_subqueries_when_set(self):
        from ira.agent.nodes import retrieve_internal
        calls = []
        def mock_rerank(query_text, **kwargs):
            calls.append(query_text)
            return _fake_rerank_result(1)

        state = _base_state(subqueries=["sub Q1", "sub Q2"])
        with patch("ira.retrieval.rerank_runner.query_with_rerank", side_effect=mock_rerank):
            retrieve_internal(state)
        assert calls == ["sub Q1", "sub Q2"]

    def test_deduplicates_packs_across_subqueries(self):
        from ira.agent.nodes import retrieve_internal
        # Both subqueries return same chunk_id — should deduplicate to 1
        with patch("ira.retrieval.rerank_runner.query_with_rerank",
                   return_value=_fake_rerank_result(1)):
            state = _base_state(subqueries=["Q1", "Q2"])
            result = retrieve_internal(state)
        assert len(result["evidence_packs"]) == 1

    def test_appends_tool_call_entry(self):
        from ira.agent.nodes import retrieve_internal
        with patch("ira.retrieval.rerank_runner.query_with_rerank",
                   return_value=_fake_rerank_result(2)):
            result = retrieve_internal(_base_state())
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool"] == "retrieve_internal"
        assert result["tool_calls"][0]["result_count"] == 2

    def test_handles_retriever_exception_gracefully(self):
        from ira.agent.nodes import retrieve_internal
        with patch("ira.retrieval.rerank_runner.query_with_rerank",
                   side_effect=RuntimeError("index error")):
            result = retrieve_internal(_base_state())
        # Should not raise — returns empty packs + error tool entry
        assert result["evidence_packs"] == []
        assert result["tool_calls"][0]["result_count"] == 0

    def test_merges_new_packs_with_existing(self):
        from ira.agent.nodes import retrieve_internal
        existing_pack = {
            "chunk_id": "existing-001", "doc_id": "d", "title": "t",
            "url": None, "section": None, "child_text": "", "parent_text": "",
            "rrf_score": 0.1, "reranker_score": 0.5, "in_both": False,
        }
        state = _base_state(evidence_packs=[existing_pack])
        with patch("ira.retrieval.rerank_runner.query_with_rerank",
                   return_value=_fake_rerank_result(2)):
            result = retrieve_internal(state)
        # 1 existing + 2 new (different chunk_ids) = 3
        assert len(result["evidence_packs"]) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: search_web
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchWeb:
    # Tests updated for Day 10: search_web is now real (uses mock provider by default)

    def test_appends_tool_call_entry(self):
        from ira.agent.nodes import search_web
        result = search_web(_base_state())
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool"] == "search_web"

    def test_populates_evidence_packs(self):
        from ira.agent.nodes import search_web
        # Mock provider returns 5 fixture results
        result = search_web(_base_state())
        assert "evidence_packs" in result
        assert len(result["evidence_packs"]) > 0

    def test_evidence_packs_have_web_source_type(self):
        from ira.agent.nodes import search_web
        result = search_web(_base_state())
        for pack in result["evidence_packs"]:
            assert pack["source_type"] == "web"

    def test_preserves_existing_warnings(self):
        from ira.agent.nodes import search_web
        state = _base_state(warnings=["prior warning"])
        result = search_web(state)
        assert "prior warning" in result["warnings"]

    def test_appends_warning_on_empty_results(self):
        from ira.agent.nodes import search_web
        from unittest.mock import patch
        with patch("ira.agent.web_search_tool.search_web_tool", return_value=[]):
            result = search_web(_base_state())
        assert any("no results" in w.lower() for w in result["warnings"])

    def test_handles_exception_gracefully(self):
        from ira.agent.nodes import search_web
        from unittest.mock import patch
        with patch("ira.agent.web_search_tool.search_web_tool",
                   side_effect=RuntimeError("API down")):
            result = search_web(_base_state())
        assert result["tool_calls"][0]["result_count"] == 0
        assert any("failed" in w.lower() for w in result["warnings"])


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: grade_context
# ─────────────────────────────────────────────────────────────────────────────

class TestGradeContext:

    def _pack_dict(self, chunk_id: str, reranker_score: float) -> dict:
        return {
            "chunk_id": chunk_id, "doc_id": "d", "title": "t",
            "url": None, "section": None, "child_text": "", "parent_text": "",
            "rrf_score": 0.05, "reranker_score": reranker_score, "in_both": True,
        }

    def test_empty_when_no_packs(self):
        from ira.agent.nodes import grade_context
        result = grade_context(_base_state(evidence_packs=[]))
        assert result["context_grade"] == "empty"

    def test_strong_when_high_score_and_sufficient_packs(self):
        from ira.agent.nodes import grade_context
        packs = [self._pack_dict(f"c{i}", 0.8) for i in range(3)]
        result = grade_context(_base_state(evidence_packs=packs))
        assert result["context_grade"] == "strong"

    def test_weak_when_score_below_threshold(self):
        from ira.agent.nodes import grade_context
        # reranker_score=0.1 is below AGENT_WEAK_SCORE_THRESHOLD=0.3
        packs = [self._pack_dict(f"c{i}", 0.1) for i in range(3)]
        result = grade_context(_base_state(evidence_packs=packs))
        assert result["context_grade"] == "weak"

    def test_weak_when_too_few_packs(self):
        from ira.agent.nodes import grade_context
        # Only 1 pack (below AGENT_MIN_EVIDENCE_PACKS=2)
        packs = [self._pack_dict("c0", 0.9)]
        result = grade_context(_base_state(evidence_packs=packs))
        assert result["context_grade"] == "weak"

    def test_only_returns_context_grade_key(self):
        from ira.agent.nodes import grade_context
        result = grade_context(_base_state(evidence_packs=[]))
        assert set(result.keys()) == {"context_grade"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: decompose_query
# ─────────────────────────────────────────────────────────────────────────────

class TestDecomposeQuery:

    def test_returns_list_of_subqueries(self):
        from ira.agent.nodes import decompose_query
        llm_output = '["What is tiling?", "How does SRAM help?", "What is HBM?"]'
        with patch("ira.agent.llm.call", return_value=llm_output):
            result = decompose_query(_base_state())
        assert result["subqueries"] == ["What is tiling?", "How does SRAM help?", "What is HBM?"]

    def test_increments_retries(self):
        from ira.agent.nodes import decompose_query
        with patch("ira.agent.llm.call", return_value='["Q1", "Q2"]'):
            result = decompose_query(_base_state(retries=0))
        assert result["retries"] == 1

    def test_accumulates_retries_across_calls(self):
        from ira.agent.nodes import decompose_query
        with patch("ira.agent.llm.call", return_value='["Q1"]'):
            result = decompose_query(_base_state(retries=1))
        assert result["retries"] == 2

    def test_falls_back_to_original_query_on_bad_llm_output(self):
        from ira.agent.nodes import decompose_query
        with patch("ira.agent.llm.call", return_value="not a list at all"):
            result = decompose_query(_base_state(query="original query"))
        assert result["subqueries"] == ["original query"]

    def test_falls_back_on_llm_exception(self):
        from ira.agent.nodes import decompose_query
        with patch("ira.agent.llm.call", side_effect=RuntimeError("API down")):
            result = decompose_query(_base_state(query="original query"))
        assert result["subqueries"] == ["original query"]

    def test_filters_empty_strings_from_output(self):
        from ira.agent.nodes import decompose_query
        with patch("ira.agent.llm.call", return_value='["Q1", "", "Q2"]'):
            result = decompose_query(_base_state())
        assert "" not in result["subqueries"]
        assert len(result["subqueries"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Node 7: synthesize_answer
# ─────────────────────────────────────────────────────────────────────────────

class TestSynthesizeAnswer:

    def _strong_state(self) -> dict:
        packs = [
            {
                "chunk_id": f"c{i}", "doc_id": "arxiv_test", "title": "Test Paper",
                "url": "https://arxiv.org/test", "section": f"Section {i}",
                "child_text": f"Child text {i}", "parent_text": f"Full parent context {i}",
                "rrf_score": 0.05, "reranker_score": 0.8, "in_both": True,
            }
            for i in range(3)
        ]
        return _base_state(evidence_packs=packs, context_grade="strong")

    def test_returns_answer_draft(self):
        from ira.agent.nodes import synthesize_answer
        with patch("ira.agent.llm.call", return_value="FlashAttention uses tiling [1]."):
            result = synthesize_answer(self._strong_state())
        assert result["answer_draft"] == "FlashAttention uses tiling [1]."

    def test_builds_citations_from_packs(self):
        from ira.agent.nodes import synthesize_answer
        with patch("ira.agent.llm.call", return_value="Answer [1][2][3]."):
            result = synthesize_answer(self._strong_state())
        assert len(result["citations"]) == 3
        assert result["citations"][0]["citation_id"] == "1"
        assert result["citations"][0]["chunk_id"] == "c0"

    def test_citation_label_contains_title_and_section(self):
        from ira.agent.nodes import synthesize_answer
        with patch("ira.agent.llm.call", return_value="Answer."):
            result = synthesize_answer(self._strong_state())
        label = result["citations"][0]["label"]
        assert "Test Paper" in label
        assert "Section 0" in label

    def test_handles_empty_evidence_gracefully(self):
        from ira.agent.nodes import synthesize_answer
        state = _base_state(evidence_packs=[], context_grade="empty")
        result = synthesize_answer(state)
        assert result["answer_draft"] != ""
        assert len(result["warnings"]) >= 1

    def test_adds_warning_for_weak_context(self):
        from ira.agent.nodes import synthesize_answer
        state = self._strong_state()
        state["context_grade"] = "weak"
        state["retries"] = 2
        with patch("ira.agent.llm.call", return_value="Partial answer."):
            result = synthesize_answer(state)
        assert any("weak" in w for w in result["warnings"])

    def test_handles_llm_exception_gracefully(self):
        from ira.agent.nodes import synthesize_answer
        with patch("ira.agent.llm.call", side_effect=RuntimeError("timeout")):
            result = synthesize_answer(self._strong_state())
        assert "error" in result["answer_draft"].lower() or "failed" in result["answer_draft"].lower()
        assert len(result["warnings"]) >= 1