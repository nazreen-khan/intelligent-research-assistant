from ira.contracts.schemas import (
    QueryRequest, EvidenceChunk, Citation, AnswerResponse, AgentTraceStep
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
        final_answer="Answer",
        citations=[cit],
        evidence=[chunk],
        trace=[step],
        warnings=[],
    )
    assert resp.schema_version == 1
