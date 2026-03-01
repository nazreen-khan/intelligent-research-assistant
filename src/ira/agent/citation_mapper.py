"""
agent/citation_mapper.py — Central citation ownership module.

Owns the complete lifecycle of chunk_id → structured CitationRecord:
  1. build_citations()              evidence_packs → list[CitationRecord]
  2. get_used_citations()           parse [N] markers → cited CitationRecords only
  3. validate()                     structural integrity check → CitationValidationResult
  4. citation_records_to_state_dicts()  CitationRecord → plain dicts for AgentState

Design principles:
  - Zero LLM calls — all operations are deterministic and free
  - Never raises — all errors return safe defaults
  - Single source of truth for footnote label format
  - Called by synthesize_answer (build) and runner._build_response (validate)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ira.contracts.schemas import CitationRecord, CitationValidationResult

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Minimum fraction of retrieved sources that should be cited in the answer.
# Below this → density warning added to response.warnings[].
# 0.4 = at least 40% of retrieved chunks should appear as [N] in the answer.
_DEFAULT_MIN_DENSITY: float = 0.4


# ── Public API ────────────────────────────────────────────────────────────────

def build_citations(packs: list[dict[str, Any]]) -> list[CitationRecord]:
    """
    Convert evidence_pack dicts to a typed list of CitationRecord.

    Position is 1-based, matching the [1], [2]... [N] markers the LLM
    will use in the synthesized answer. Call this BEFORE building the
    LLM prompt so the evidence block and citation numbering stay aligned.

    Args:
        packs: list of evidence_pack dicts from AgentState["evidence_packs"].
               Each dict must have "chunk_id". Other fields degrade gracefully
               to safe defaults if missing.

    Returns:
        list[CitationRecord], one record per pack, in the same order.
    """
    records: list[CitationRecord] = []

    for i, pack in enumerate(packs, 1):
        source_type = pack.get("source_type", "internal")
        source_tag  = " [WEB]" if source_type == "web" else ""
        section     = (pack.get("section") or "N/A").strip()
        doc_title   = (pack.get("title") or pack.get("doc_id") or "Unknown Source").strip()
        url         = pack.get("url")
        chunk_id    = pack.get("chunk_id", f"unknown-{i}")
        doc_id      = pack.get("doc_id", "")

        # Canonical footnote label — single format used everywhere in the system
        if url:
            footnote = f"[{i}]{source_tag} {doc_title} — {section} ({url})"
        else:
            footnote = f"[{i}]{source_tag} {doc_title} — {section}"

        records.append(CitationRecord(
            citation_id    = str(i),
            chunk_id       = chunk_id,
            doc_id         = doc_id,
            doc_title      = doc_title,
            url            = url,
            section        = section,
            source_type    = source_type,
            footnote_label = footnote,
            position       = i,
        ))

    return records


def get_used_citations(
    answer_text: str,
    citations: list[CitationRecord],
) -> list[CitationRecord]:
    """
    Return only the CitationRecords that are actually referenced in answer_text.

    Parses [N] markers in the answer, looks them up in citations by citation_id,
    and returns the matching records in their original order (lowest ID first).

    Args:
        answer_text: The synthesized answer markdown string.
        citations:   Full list of CitationRecords built by build_citations().

    Returns:
        Subset of citations whose citation_id appears in answer_text.
        Empty list if answer_text is empty or no [N] markers found.
    """
    if not answer_text or not citations:
        return []

    cited_ids = set(_CITATION_RE.findall(answer_text))
    return [c for c in citations if c.citation_id in cited_ids]


def validate(
    answer_text: str,
    citations: list[CitationRecord],
    min_density: float = _DEFAULT_MIN_DENSITY,
) -> CitationValidationResult:
    """
    Run all structural integrity checks on answer + citation list.

    Checks (all deterministic, zero LLM cost):
      1. Orphan IDs — [N] appears in answer but no CitationRecord has that ID
      2. Citation density — cited_count / total_count >= min_density

    Unused IDs (CitationRecord exists but not cited) are recorded but do NOT
    cause passed=False — retrieving more context than needed is normal behaviour.

    Args:
        answer_text: Synthesized answer markdown string (answer_draft from state).
        citations:   CitationRecord list from build_citations().
        min_density: Minimum fraction of retrieved sources that must be cited.

    Returns:
        CitationValidationResult — never raises; always returns a valid result.
    """
    # Handle edge cases gracefully
    if not citations:
        # No evidence → no citations expected → trivially valid
        return CitationValidationResult(
            passed               = True,
            total_citations      = 0,
            valid_citations      = 0,
            density_ok           = True,
            density_score        = 1.0,
            min_density_threshold = min_density,
        )

    cited_ids = set(_CITATION_RE.findall(answer_text or ""))
    known_ids = {c.citation_id for c in citations}

    # [N] appears in answer but has no matching CitationRecord
    orphan_ids = sorted(cited_ids - known_ids, key=lambda x: int(x) if x.isdigit() else 0)

    # CitationRecord exists but is never referenced in the answer
    unused_ids = sorted(known_ids - cited_ids, key=lambda x: int(x) if x.isdigit() else 0)

    # Density: what fraction of retrieved sources were actually cited?
    density       = len(cited_ids) / len(citations)
    density_ok    = density >= min_density

    issues: list[str] = []
    if orphan_ids:
        issues.append(
            f"Orphan citation IDs referenced in answer but not in evidence: {orphan_ids}"
        )
    if not density_ok and cited_ids:
        # Only warn about low density when SOME citations exist.
        # If the answer has zero citations that's caught by Day 11 uncited detection.
        issues.append(
            f"Citation density {density:.0%} is below threshold {min_density:.0%} "
            f"({len(cited_ids)} of {len(citations)} sources cited)"
        )

    return CitationValidationResult(
        passed                = len(issues) == 0,
        total_citations       = len(cited_ids),
        valid_citations       = len(cited_ids - set(orphan_ids)),
        orphan_ids            = orphan_ids,
        unused_citation_ids   = unused_ids,
        density_ok            = density_ok,
        density_score         = round(density, 3),
        min_density_threshold = min_density,
        issues                = issues,
    )


def citation_records_to_state_dicts(records: list[CitationRecord]) -> list[dict[str, Any]]:
    """
    Serialise CitationRecord list to plain dicts for AgentState storage.

    LangGraph requires all state values to be JSON-serialisable plain types.
    CitationRecord is a Pydantic model so it must be converted before being
    stored in AgentState["citations"].

    runner._build_response reverses this with CitationRecord(**d).
    """
    dicts = []
    for r in records:
        d = r.model_dump()
        # Add "label" alias so existing code that reads citations[N]["label"] still works.
        # footnote_label is the canonical field; label is a backward-compat convenience copy.
        d["label"] = d.get("footnote_label", "")
        dicts.append(d)
    return dicts