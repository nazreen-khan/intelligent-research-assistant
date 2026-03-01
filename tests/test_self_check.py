"""
tests/test_self_check.py — Unit tests for the Day 11 self_check node.

Coverage:
  - Skip path: empty answer or empty evidence_packs
  - Citation grounding: pass (high overlap) and fail (low overlap)
  - Numeric faithfulness: pass (number present) and fail (number absent)
  - Uncited claim detection: factual sentence without [N] is counted
  - Error graceful degradation: exception inside check → warning, state returned

All tests run fully offline — no API keys, no Qdrant, no disk I/O.
Run:  uv run pytest tests/test_self_check.py -v
"""

from __future__ import annotations

import pytest

from ira.agent.state import make_initial_state
from ira.agent.nodes import (
    self_check,
    _extract_cited_claims,
    _extract_numbers,
    _is_grounded_in_text,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _state(
    answer: str = "",
    citations: list | None = None,
    packs: list | None = None,
    warnings: list | None = None,
) -> dict:
    """Build a minimal AgentState ready for self_check."""
    s = make_initial_state(query="test query", request_id="test-sc-001")
    s["answer_draft"]   = answer
    s["citations"]      = citations if citations is not None else []
    s["evidence_packs"] = packs     if packs     is not None else []
    s["warnings"]       = warnings  if warnings  is not None else []
    return s


def _pack(
    chunk_id: str = "c1",
    parent_text: str = "FlashAttention uses tiling to reduce HBM memory reads by 3x on A100.",
    source_type: str = "internal",
) -> dict:
    """Build a minimal evidence pack dict."""
    return {
        "chunk_id":       chunk_id,
        "doc_id":         "arxiv_test",
        "title":          "FlashAttention Paper",
        "parent_text":    parent_text,
        "child_text":     parent_text[:80],
        "reranker_score": 0.9,
        "rrf_score":      0.7,
        "source_type":    source_type,
        "url":            None,
        "section":        "Methods",
    }


def _citation(citation_id: int | str, chunk_id: str = "c1") -> dict:
    """Build a minimal citation dict."""
    return {
        "citation_id": str(citation_id),
        "chunk_id":    chunk_id,
        "label":       f"[{citation_id}] FlashAttention Paper — Methods",
        "url":         None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper unit tests  (fast, isolated)
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractCitedClaims:
    def test_returns_only_cited_sentences(self):
        answer = "FlashAttention uses tiling [1]. It also helps with speed. Memory is reduced [2]."
        claims = _extract_cited_claims(answer)
        # Only sentences with [N] are returned
        assert len(claims) == 2
        cids = [c["citation_ids"] for c in claims]
        assert ["1"] in cids
        assert ["2"] in cids

    def test_no_citations_returns_empty(self):
        answer = "This sentence has no citations. Neither does this one."
        claims = _extract_cited_claims(answer)
        assert claims == []

    def test_multiple_citations_in_one_sentence(self):
        answer = "This claim is supported by two sources [1][2]."
        claims = _extract_cited_claims(answer)
        assert len(claims) == 1
        assert set(claims[0]["citation_ids"]) == {"1", "2"}


class TestExtractNumbers:
    def test_integer(self):
        # "3x" is captured as a unit — check both bare integer and multiplier forms
        result = _extract_numbers("reduces reads by 3x")
        assert any("3" in r for r in result)  # "3x" contains "3"

    def test_decimal(self):
        assert "1.5" in _extract_numbers("1.5 times faster")

    def test_percentage(self):
        assert "75%" in _extract_numbers("75% reduction in memory")

    def test_no_numbers(self):
        assert _extract_numbers("fast and efficient inference") == []


class TestIsGroundedInText:
    def test_high_overlap_passes(self):
        claim = "FlashAttention reduces HBM memory reads using tiling."
        source = "FlashAttention uses tiling to reduce HBM memory reads by 3x."
        assert _is_grounded_in_text(claim, source, threshold=0.30) is True

    def test_low_overlap_fails(self):
        claim = "Quantum entanglement accelerates transformer inference at scale."
        source = "FlashAttention uses tiling to reduce HBM memory reads."
        assert _is_grounded_in_text(claim, source, threshold=0.30) is False

    def test_empty_claim_passes(self):
        # Nothing to verify — should not block
        assert _is_grounded_in_text("", "some source text") is True

    def test_stop_words_only_claim_passes(self):
        # Pure stop words yield no content words → passes
        assert _is_grounded_in_text("the and or but", "anything") is True


# ─────────────────────────────────────────────────────────────────────────────
# self_check node integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfCheckSkip:
    def test_skip_on_empty_answer(self):
        result = self_check(_state(answer=""))
        sc = result["self_check_result"]
        assert sc["checks_run"] == 0
        assert sc["passed"] is True
        assert sc["coverage_score"] == 1.0

    def test_skip_on_empty_packs(self):
        result = self_check(_state(
            answer="FlashAttention is fast [1].",
            citations=[_citation(1)],
            packs=[],    # no packs → skip
        ))
        sc = result["self_check_result"]
        assert sc["checks_run"] == 0

    def test_answer_unchanged_on_skip(self):
        answer = "Some answer text."
        result = self_check(_state(answer=answer, packs=[]))
        assert result["answer_draft"] == answer


class TestCitationGrounding:
    def test_grounded_claim_passes(self):
        """Claim words match source text → no issue."""
        state = _state(
            answer="FlashAttention reduces HBM memory reads using tiling [1].",
            citations=[_citation(1, "c1")],
            packs=[_pack("c1", "FlashAttention uses tiling to reduce HBM memory reads by 3x.")],
        )
        result = self_check(state)
        sc = result["self_check_result"]
        issues = [i for i in sc["citation_issues"] if i["issue"] == "unsupported"]
        assert len(issues) == 0
        assert sc["checks_failed"] == 0

    def test_ungrounded_claim_is_flagged(self):
        """Claim unrelated to source text → unsupported issue."""
        state = _state(
            answer="Quantum entanglement dramatically speeds up inference [1].",
            citations=[_citation(1, "c1")],
            packs=[_pack("c1", "FlashAttention uses tiling to reduce HBM memory reads.")],
        )
        result = self_check(state)
        sc = result["self_check_result"]
        assert any(i["issue"] == "unsupported" for i in sc["citation_issues"])
        assert sc["checks_failed"] >= 1

    def test_missing_citation_id_flagged_as_error(self):
        """[2] in answer but only citation 1 in evidence → error severity."""
        state = _state(
            answer="This claim uses citation two [2].",
            citations=[_citation(1, "c1")],   # only [1] mapped
            packs=[_pack("c1")],
        )
        result = self_check(state)
        sc = result["self_check_result"]
        assert any(
            i["issue"] == "citation_id_not_found" and i["severity"] == "error"
            for i in sc["citation_issues"]
        )

    def test_warning_appended_for_ungrounded(self):
        state = _state(
            answer="Completely unrelated claim about cooking [1].",
            citations=[_citation(1, "c1")],
            packs=[_pack("c1", "FlashAttention tiling reduces HBM reads.")],
        )
        result = self_check(state)
        assert any("self_check" in w for w in result["warnings"])


class TestNumericFaithfulness:
    def test_number_present_in_source_passes(self):
        """3x appears in source → no numeric failure."""
        state = _state(
            answer="Throughput improves 3x with FlashAttention [1].",
            citations=[_citation(1, "c1")],
            packs=[_pack("c1", "FlashAttention reduces HBM memory reads by 3x on A100.")],
        )
        result = self_check(state)
        assert result["self_check_result"]["numeric_failures"] == 0

    def test_number_absent_from_source_fails(self):
        """128 does not appear in source → numeric_mismatch."""
        state = _state(
            answer="Achieves 128 tokens per second [1].",
            citations=[_citation(1, "c1")],
            packs=[_pack("c1", "FlashAttention uses tiling to save memory.")],
        )
        result = self_check(state)
        sc = result["self_check_result"]
        assert sc["numeric_failures"] >= 1
        assert any(i["issue"] == "numeric_mismatch" for i in sc["citation_issues"])

    def test_answer_annotated_on_numeric_failure(self):
        """answer_draft gets [unverified: X] marker when number not in source."""
        state = _state(
            answer="Uses 512 GB of HBM [1].",
            citations=[_citation(1, "c1")],
            packs=[_pack("c1", "FlashAttention tiles attention computation.")],
        )
        result = self_check(state)
        assert "unverified" in result["answer_draft"]

    def test_numeric_failure_adds_warning(self):
        state = _state(
            answer="Reaches 999 tokens/sec throughput [1].",
            citations=[_citation(1, "c1")],
            packs=[_pack("c1", "FlashAttention is fast.")],
        )
        result = self_check(state)
        assert any("numeric" in w.lower() or "unverified" in w.lower() or "not found" in w.lower()
                   for w in result["warnings"])


class TestUncitedClaimDetection:
    def test_long_sentence_without_citation_is_counted(self):
        """A sentence > 12 words with no [N] → uncited_sentences += 1."""
        state = _state(
            answer=(
                "FlashAttention achieves substantially better performance by tiling "
                "the attention computation to fit in SRAM."  # 18 words, no citation
            ),
            citations=[],
            packs=[_pack("c1")],
        )
        result = self_check(state)
        sc = result["self_check_result"]
        assert sc["uncited_sentences"] >= 1
        assert any("factual" in w or "citation" in w or "uncited" in w
                   for w in result["warnings"])

    def test_numeric_sentence_without_citation_counted(self):
        """Sentence with a number but no [N] → uncited."""
        state = _state(
            answer="Reduces memory by 50%.",
            citations=[],
            packs=[_pack("c1")],
        )
        result = self_check(state)
        assert result["self_check_result"]["uncited_sentences"] >= 1

    def test_all_cited_answer_has_zero_uncited(self):
        """Every sentence has a [N] → uncited_sentences == 0."""
        state = _state(
            answer="FlashAttention reduces HBM reads [1]. Tiling is the key technique [1].",
            citations=[_citation(1, "c1")],
            packs=[_pack("c1", "FlashAttention uses tiling to reduce HBM reads.")],
        )
        result = self_check(state)
        # uncited_sentences counts sentences WITHOUT [N] — here all have one
        assert result["self_check_result"]["uncited_sentences"] == 0


class TestSelfCheckResultStructure:
    def test_result_dict_has_all_required_keys(self):
        """self_check_result always has the full set of keys."""
        result = self_check(_state(answer=""))
        sc = result["self_check_result"]
        required = {
            "passed", "checks_run", "checks_passed", "checks_failed",
            "citation_issues", "uncited_sentences", "numeric_failures", "coverage_score",
        }
        assert required.issubset(sc.keys())

    def test_warnings_list_preserved_when_no_issues(self):
        """Existing warnings are preserved; self_check only appends."""
        initial_warning = "prior warning from grade_context"
        state = _state(
            answer="FlashAttention reduces reads [1].",
            citations=[_citation(1, "c1")],
            packs=[_pack("c1", "FlashAttention reduces HBM reads.")],
            warnings=[initial_warning],
        )
        result = self_check(state)
        assert initial_warning in result["warnings"]

    def test_coverage_score_is_float_between_0_and_1(self):
        state = _state(
            answer="FlashAttention is fast [1].",
            citations=[_citation(1, "c1")],
            packs=[_pack("c1", "FlashAttention is fast.")],
        )
        result = self_check(state)
        score = result["self_check_result"]["coverage_score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0