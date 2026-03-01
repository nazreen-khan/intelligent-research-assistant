"""
tests/test_citation_mapper.py — Unit tests for Day 12 citation_mapper module.

Coverage:
  TestBuildCitations      — footnote format, web tag, URL handling, missing fields
  TestGetUsedCitations    — cited vs uncited, empty answer, empty citations
  TestValidateOrphans     — [N] in answer with no CitationRecord
  TestValidateDensity     — below-threshold density warning
  TestValidateClean       — fully valid answer passes with no issues
  TestAnswerResponseSchema — answer_markdown, used_sources, schema_version=2

All tests fully offline — no API keys, no Qdrant, no disk I/O.
Run:  uv run pytest tests/test_citation_mapper.py -v
"""

from __future__ import annotations

import pytest

from ira.agent.citation_mapper import (
    build_citations,
    citation_records_to_state_dicts,
    get_used_citations,
    validate,
)
from ira.contracts.schemas import (
    AnswerResponse,
    CitationRecord,
    CitationValidationResult,
    EvidenceChunk,
    Citation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _pack(
    chunk_id: str = "c1",
    doc_id: str = "arxiv_123",
    title: str = "FlashAttention Paper",
    url: str | None = "https://arxiv.org/abs/1",
    section: str = "Methods",
    source_type: str = "internal",
) -> dict:
    return {
        "chunk_id":    chunk_id,
        "doc_id":      doc_id,
        "title":       title,
        "url":         url,
        "section":     section,
        "source_type": source_type,
        "child_text":  "Some chunk text.",
        "parent_text": "Some parent text.",
        "rrf_score":   0.7,
        "reranker_score": 0.9,
        "in_both":     True,
    }


def _three_packs() -> list[dict]:
    return [
        _pack("c1", "arxiv_1", "FlashAttention", "https://arxiv.org/1", "Methods"),
        _pack("c2", "arxiv_2", "vLLM Paper",     None,                  "Design"),
        _pack("c3", "web_1",   "Blog Post",      "https://blog.com",    "N/A", "web"),
    ]


def _three_records() -> list[CitationRecord]:
    return build_citations(_three_packs())


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildCitations
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCitations:
    def test_returns_one_record_per_pack(self):
        packs = _three_packs()
        records = build_citations(packs)
        assert len(records) == 3

    def test_citation_ids_are_1_based_strings(self):
        records = build_citations(_three_packs())
        assert [r.citation_id for r in records] == ["1", "2", "3"]

    def test_position_matches_citation_id(self):
        records = build_citations(_three_packs())
        for r in records:
            assert r.position == int(r.citation_id)

    def test_chunk_id_preserved(self):
        records = build_citations(_three_packs())
        assert records[0].chunk_id == "c1"
        assert records[1].chunk_id == "c2"

    def test_footnote_includes_url_when_present(self):
        records = build_citations(_three_packs())
        assert "https://arxiv.org/1" in records[0].footnote_label

    def test_footnote_no_url_when_absent(self):
        records = build_citations(_three_packs())
        # Pack 2 has no URL
        assert "https" not in records[1].footnote_label
        assert "vLLM Paper" in records[1].footnote_label

    def test_web_source_gets_web_tag(self):
        records = build_citations(_three_packs())
        assert "[WEB]" in records[2].footnote_label

    def test_internal_source_no_web_tag(self):
        records = build_citations(_three_packs())
        assert "[WEB]" not in records[0].footnote_label

    def test_footnote_format_is_canonical(self):
        # [N] Title — Section (url)
        records = build_citations([_pack("c1", title="MyDoc", section="Intro", url="https://x.com")])
        assert records[0].footnote_label == "[1] MyDoc — Intro (https://x.com)"

    def test_missing_section_defaults_to_na(self):
        pack = _pack(section="")
        pack["section"] = ""
        records = build_citations([pack])
        assert records[0].section == "N/A"

    def test_missing_title_falls_back_to_doc_id(self):
        pack = _pack()
        pack["title"] = ""
        records = build_citations([pack])
        assert records[0].doc_title == pack["doc_id"]

    def test_empty_packs_returns_empty_list(self):
        assert build_citations([]) == []

    def test_serialise_to_state_dicts(self):
        records = build_citations(_three_packs())
        dicts = citation_records_to_state_dicts(records)
        assert len(dicts) == 3
        assert all(isinstance(d, dict) for d in dicts)
        assert dicts[0]["citation_id"] == "1"
        assert dicts[0]["chunk_id"] == "c1"
        # Verify roundtrip: dict → CitationRecord
        rt = CitationRecord(**dicts[0])
        assert rt.citation_id == records[0].citation_id
        assert rt.footnote_label == records[0].footnote_label


# ─────────────────────────────────────────────────────────────────────────────
# TestGetUsedCitations
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUsedCitations:
    def test_returns_only_cited_records(self):
        records = _three_records()
        answer  = "FlashAttention uses tiling [1]. vLLM uses paging [2]."
        used    = get_used_citations(answer, records)
        assert [r.citation_id for r in used] == ["1", "2"]

    def test_uncited_record_excluded(self):
        records = _three_records()
        answer  = "Only citing one source [1]."
        used    = get_used_citations(answer, records)
        assert len(used) == 1
        assert used[0].citation_id == "1"

    def test_empty_answer_returns_empty(self):
        assert get_used_citations("", _three_records()) == []

    def test_empty_citations_returns_empty(self):
        assert get_used_citations("Some answer [1].", []) == []

    def test_no_citation_markers_returns_empty(self):
        records = _three_records()
        used    = get_used_citations("This answer has no citation markers.", records)
        assert used == []

    def test_duplicate_markers_deduplicated(self):
        records = _three_records()
        answer  = "First mention [1]. Another claim [1]. Third [1]."
        used    = get_used_citations(answer, records)
        assert len(used) == 1

    def test_preserves_citation_order(self):
        records = _three_records()
        answer  = "Claims [3][1][2]."
        used    = get_used_citations(answer, records)
        # Should be in records order (1, 2, 3), not answer order
        assert [r.citation_id for r in used] == ["1", "2", "3"]


# ─────────────────────────────────────────────────────────────────────────────
# TestValidateOrphans
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateOrphans:
    def test_orphan_id_causes_failure(self):
        records = _three_records()           # IDs: 1, 2, 3
        answer  = "Uses citation five [5]."  # [5] doesn't exist
        result  = validate(answer, records)
        assert result.passed is False
        assert "5" in result.orphan_ids

    def test_orphan_id_appears_in_issues(self):
        records = _three_records()
        result  = validate("Claim [9].", records)
        assert any("9" in issue for issue in result.issues)

    def test_no_orphans_when_all_ids_valid(self):
        records = _three_records()
        result  = validate("FlashAttention [1]. vLLM [2]. Blog [3].", records)
        assert result.orphan_ids == []

    def test_orphan_does_not_count_as_valid_citation(self):
        records = _three_records()
        result  = validate("Uses [99].", records)
        assert result.valid_citations == 0
        assert result.total_citations == 1


# ─────────────────────────────────────────────────────────────────────────────
# TestValidateDensity
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateDensity:
    def test_low_density_causes_failure(self):
        # 5 packs retrieved, only [1] cited = 20% < 40% threshold
        packs   = [_pack(f"c{i}", f"doc_{i}", f"Doc {i}") for i in range(1, 6)]
        records = build_citations(packs)
        result  = validate("Only citing one source [1].", records)
        assert result.density_ok is False
        assert result.density_score < 0.4

    def test_density_issue_in_issues_list(self):
        packs   = [_pack(f"c{i}", f"doc_{i}", f"Doc {i}") for i in range(1, 6)]
        records = build_citations(packs)
        result  = validate("Only [1].", records)
        assert any("density" in issue.lower() for issue in result.issues)

    def test_high_density_passes(self):
        records = _three_records()
        # Cite all 3 of 3 = 100%
        result  = validate("Claims [1][2][3].", records)
        assert result.density_ok is True
        assert result.density_score == 1.0

    def test_meets_threshold_exactly_passes(self):
        # 5 packs, cite 2 = 40% == threshold
        packs   = [_pack(f"c{i}", f"doc_{i}", f"Doc {i}") for i in range(1, 6)]
        records = build_citations(packs)
        result  = validate("Claims [1][2].", records, min_density=0.4)
        assert result.density_ok is True

    def test_unused_ids_recorded_but_not_failure(self):
        records = _three_records()
        # Only cite [1]; [2] and [3] are unused but that's normal
        result  = validate("Only [1].", records)
        assert "2" in result.unused_citation_ids
        assert "3" in result.unused_citation_ids
        # density failure but NOT an orphan issue
        assert result.orphan_ids == []

    def test_custom_min_density_respected(self):
        records = _three_records()
        # Cite 1 of 3 = 33%. With threshold 0.2 → passes. With 0.5 → fails.
        result_pass = validate("Only [1].", records, min_density=0.2)
        result_fail = validate("Only [1].", records, min_density=0.5)
        assert result_pass.density_ok is True
        assert result_fail.density_ok is False

    def test_empty_answer_no_density_warning(self):
        # Answer with no citations at all → density=0 but we don't warn
        # (Day 11 self_check handles uncited detection)
        records = _three_records()
        result  = validate("This answer has no citation markers.", records)
        # density=0 < 0.4 but cited_ids is empty so no density issue appended
        assert not any("density" in issue.lower() for issue in result.issues)


# ─────────────────────────────────────────────────────────────────────────────
# TestValidateClean
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateClean:
    def test_all_cited_answer_passes(self):
        records = _three_records()
        result  = validate("A [1]. B [2]. C [3].", records)
        assert result.passed is True
        assert result.issues == []

    def test_empty_packs_trivially_valid(self):
        result = validate("Some answer with no packs.", [])
        assert result.passed is True
        assert result.total_citations == 0

    def test_result_fields_complete(self):
        records = _three_records()
        result  = validate("A [1]. B [2].", records)
        required = {
            "passed", "total_citations", "valid_citations",
            "orphan_ids", "unused_citation_ids", "density_ok",
            "density_score", "min_density_threshold", "issues",
        }
        assert required.issubset(result.model_fields.keys())

    def test_density_score_is_float_0_to_1(self):
        records = _three_records()
        result  = validate("A [1].", records)
        assert isinstance(result.density_score, float)
        assert 0.0 <= result.density_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TestAnswerResponseSchema  (Day 12 contract changes)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerResponseSchema:
    def _make_response(self, **kwargs) -> AnswerResponse:
        defaults = dict(
            request_id      = "test-001",
            answer_markdown = "FlashAttention uses tiling [1].",
        )
        defaults.update(kwargs)
        return AnswerResponse(**defaults)

    def test_answer_markdown_field_exists(self):
        r = self._make_response()
        assert r.answer_markdown == "FlashAttention uses tiling [1]."

    def test_final_answer_alias_works(self):
        """Backward-compat: .final_answer still returns the same value."""
        r = self._make_response()
        assert r.final_answer == r.answer_markdown

    def test_schema_version_is_2(self):
        r = self._make_response()
        assert r.schema_version == 2

    def test_used_sources_defaults_to_empty(self):
        r = self._make_response()
        assert r.used_sources == []

    def test_citation_validation_defaults_to_none(self):
        r = self._make_response()
        assert r.citation_validation is None

    def test_citation_density_defaults_to_zero(self):
        r = self._make_response()
        assert r.citation_density == 0.0

    def test_used_sources_accepts_evidence_chunks(self):
        chunk = EvidenceChunk(
            chunk_id="c1", doc_id="d1", title="Test",
            source_type="internal", text="text", score=0.9
        )
        r = self._make_response(used_sources=[chunk])
        assert len(r.used_sources) == 1
        assert r.used_sources[0].chunk_id == "c1"

    def test_citation_validation_accepts_result(self):
        cv = CitationValidationResult(passed=True, density_score=1.0)
        r  = self._make_response(citation_validation=cv)
        assert r.citation_validation.passed is True

    def test_full_response_roundtrip(self):
        """Build a complete AnswerResponse and verify model_dump() is JSON-safe."""
        import json
        cit = Citation(citation_id="1", chunk_id="c1", label="[1] Test", url=None)
        ev  = EvidenceChunk(chunk_id="c1", doc_id="d1", title="T", source_type="internal", text="t")
        cv  = CitationValidationResult(passed=True, density_score=1.0, total_citations=1)
        r   = self._make_response(
            citations=[cit], evidence=[ev], used_sources=[ev],
            citation_validation=cv, citation_density=1.0,
        )
        dumped = r.model_dump()
        # Must be JSON-serialisable
        json_str = json.dumps(dumped)
        assert "answer_markdown" in json_str
        assert "schema_version" in json_str