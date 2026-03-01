from ira.contracts.schemas import (
    QueryRequest, EvidenceChunk, Citation, AnswerResponse, AgentTraceStep,
    CitationRecord, CitationValidationResult,
)


def test_contract_models_instantiation():
    req = QueryRequest(query="What is KV cache?")
    chunk = EvidenceChunk(
        chunk_id="c1",
        doc_id="d1",
        title="Paper",
        source_type="internal",
        text="KV cache is ...",
        score=0.9,
    )
    cit = Citation(citation_id="1", chunk_id="c1", label="[1] Paper", url="https://example.com")
    step = AgentTraceStep(step=1, node="analyze_intent", status="ok", latency_ms=10, detail={})
    resp = AnswerResponse(
        request_id="r1",
        answer_markdown="Answer",   # Day 12: renamed from final_answer
        citations=[cit],
        evidence=[chunk],
        trace=[step],
        warnings=[],
    )
    assert resp.schema_version == 2                       # Day 12: bumped
    assert resp.answer_markdown == "Answer"
    assert resp.final_answer == "Answer"                  # backward-compat alias
    assert resp.used_sources == []                        # Day 12: new field
    assert resp.citation_validation is None               # Day 12: new field
    assert resp.citation_density == 0.0                   # Day 12: new field


def test_citation_record_model():
    """Day 12: CitationRecord Pydantic model."""
    rec = CitationRecord(
        citation_id="1",
        chunk_id="c1",
        doc_id="arxiv_123",
        doc_title="FlashAttention Paper",
        url="https://arxiv.org/abs/1",
        section="Methods",
        source_type="internal",
        footnote_label="[1] FlashAttention Paper — Methods (https://arxiv.org/abs/1)",
        position=1,
    )
    assert rec.citation_id == "1"
    assert rec.source_type == "internal"
    assert "FlashAttention" in rec.footnote_label


def test_citation_validation_result_model():
    """Day 12: CitationValidationResult Pydantic model."""
    result = CitationValidationResult(
        passed=True,
        total_citations=3,
        valid_citations=3,
        density_score=1.0,
    )
    assert result.passed is True
    assert result.orphan_ids == []
    assert result.unused_citation_ids == []
    assert result.issues == []

    failed = CitationValidationResult(
        passed=False,
        orphan_ids=["5"],
        issues=["Orphan citation ID [5] found"],
        density_score=0.2,
        density_ok=False,
    )
    assert failed.passed is False
    assert "5" in failed.orphan_ids