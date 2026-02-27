# File: tests/test_chunker.py
"""
Day 4 — QA tests for the hybrid-semantic chunker.
Updated to match REAL data patterns discovered from actual processed docs.

Real patterns observed:
  - arxiv: LlamaParse emits ALL headings as flat # (h1), numbered like "2.1 Hardware"
  - docs:  markdownify preserves Sphinx anchors: '## Heading[#](#id "Link")'
  - github: README has `# comment` lines inside fenced bash code blocks
"""

from __future__ import annotations

import uuid
from typing import List

import pytest
import tiktoken

from ira.ingest.chunker import (
    CHILD_MAX_TOKENS,
    ENCODING_NAME,
    ChildChunk,
    ParentChunk,
    _clean_heading_text,
    _find_headings,
    _iter_blocks,
    _mask_fenced_code,
    chunk_document,
    count_tokens,
    parse_markdown_sections,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def _make_doc(content: str):
    return chunk_document(
        doc_id="test_doc",
        title="Test Document",
        url="https://example.com/test",
        content_md=content,
    )


def _parent_ids(parents: List[ParentChunk]) -> set:
    return {p.parent_id for p in parents}


# ---------------------------------------------------------------------------
# Fixtures matching REAL content.md patterns
# ---------------------------------------------------------------------------

# Mirrors actual arxiv_2205.14135v1/content.md — ALL headings are flat H1
ARXIV_DOC = """\
# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré

# Abstract

Transformers are slow and memory-hungry on long sequences.

# 1 Introduction

Transformer models have emerged as the most widely used architecture.
Many approximate attention methods have aimed to reduce compute requirements.

# 2 Background

We provide some background on the performance characteristics of deep learning.

# 2.1 Hardware Performance

We focus here on GPUs. Performance on other hardware accelerators are similar.

# GPU Memory Hierarchy

The GPU memory hierarchy comprises multiple forms of memory of different sizes.

# 2.2  Standard Attention Implementation

Given input sequences Q K V where N is the sequence length.

S = QK
P = softmax S
O = PV

# 3  FlashAttention: Algorithm, Analysis, and Extensions

We present FlashAttention, an IO-aware exact attention algorithm.

# 4 Experiments

We evaluate FlashAttention on BERT, GPT-2, and long-range arena tasks.

| Model   | Speedup | Memory reduction |
| ------- | ------- | ---------------- |
| BERT    | 1.15x   | 35%              |
| GPT-2   | 3.0x    | 5x               |

# 5 Limitations and Future Directions

FlashAttention currently only supports exact attention on GPUs.
"""

# Mirrors actual docs_docs-nvidia-com_039b8a08f0ed/content.md — has Sphinx anchors
DOCS_DOC = """\
# TensorRT: Working with Quantized Types
**Source:** https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
**Version:** 2026-02-17

# Working with Quantized Types[#](#working-with-quantized-types "Link to this heading")

## Introduction to Quantization[#](#introduction-to-quantization "Link to this heading")

TensorRT enables high-performance inference by supporting quantization.

**Supported Data Types**:
- INT8 (signed 8-bit integer)
- FP8 (8-bit floating point)

### Quantization Workflows[#](#quantization-workflows "Link to this heading")

TensorRT supports both post-training quantization (PTQ) and QAT workflows.

### Explicit vs Implicit Quantization[#](#explicit-vs-implicit-quantization "Link to this heading")

In explicitly quantized networks, Q/DQ layers control quantization.

| | Implicit (Deprecated) | Explicit |
| --- | --- | --- |
| Supported types | INT8 | INT8, FP8, INT4, FP4 |

## Quantization Schemes[#](#quantization-schemes "Link to this heading")

TensorRT supports per-tensor, per-channel, and block-wise scaling.
"""

# Mirrors actual gh_flashinfer-ai_flashinfer/content.md — has # comments inside code blocks
GITHUB_DOC = """\
# FlashInfer README
**Doc ID:** gh_flashinfer-ai_flashinfer_922ba3e27710
**Source:** https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/README.md

## File: README.md
**Lines:** 1-257

## Why FlashInfer?

FlashInfer is a library for inference that delivers state-of-the-art performance.

## Getting Started

### Installation

Install the package with pip.

```bash
pip install flashinfer-python flashinfer-cubin
# JIT cache (replace cu129 with your CUDA version)
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129
```

### Verify Installation

```bash
flashinfer show-config
# Verify installation and view configuration
```

### Basic Usage

```python
# Single decode attention
import torch
from flashinfer import single_decode_with_kv_cache
output = single_decode_with_kv_cache(q, k, v)
```

After installing, you can import flashinfer in your Python scripts.

## GPU Support

| Architecture | Compute Capability | Example GPUs |
|--------------|-------------------|------|
| Hopper | SM 9.0 | H100, H200 |
| Blackwell | SM 10.0 | B200 |
"""


# ---------------------------------------------------------------------------
# TEST GROUP 1 — FIX 2: Anchor noise stripping
# ---------------------------------------------------------------------------

class TestAnchorNoiseStripping:
    def test_sphinx_anchor_stripped(self):
        raw = 'Introduction to Quantization[#](#introduction-to-quantization "Link to this heading")'
        assert _clean_heading_text(raw) == "Introduction to Quantization"

    def test_short_anchor_stripped(self):
        raw = "My Section[#](#my-section)"
        assert _clean_heading_text(raw) == "My Section"

    def test_pilcrow_anchor_stripped(self):
        raw = "Working with Quantized Types[¶](#working-with-quantized-types)"
        assert _clean_heading_text(raw) == "Working with Quantized Types"

    def test_clean_heading_unchanged(self):
        raw = "2 Background"
        assert _clean_heading_text(raw) == "2 Background"

    def test_docs_headings_are_clean_in_sections(self):
        sections = parse_markdown_sections(DOCS_DOC)
        for s in sections:
            assert "[#]" not in s.heading, f"Anchor not stripped from: {s.heading!r}"
            assert "Link to this heading" not in s.heading

    def test_docs_section_paths_are_clean(self):
        sections = parse_markdown_sections(DOCS_DOC)
        for s in sections:
            assert "[#]" not in s.section_path
            assert "Link to this heading" not in s.section_path


# ---------------------------------------------------------------------------
# TEST GROUP 2 — FIX 3: Code-block fence masking
# ---------------------------------------------------------------------------

class TestFenceMasking:
    def test_comment_lines_inside_fence_are_masked(self):
        content = "## Section\n\n```bash\n# this is a comment\npip install foo\n```\n"
        masked = _mask_fenced_code(content)
        # Interior comment line should be spaces, not #
        lines = masked.split("\n")
        interior = [l for l in lines if "this is a comment" in l]
        assert not interior, "Interior comment line was not masked"

    def test_fence_open_close_lines_preserved(self):
        content = "```bash\n# comment\n```\n"
        masked = _mask_fenced_code(content)
        lines = masked.split("\n")
        assert lines[0].strip() == "```bash"
        assert lines[2].strip() == "```"

    def test_github_code_comments_not_detected_as_headings(self):
        headings = _find_headings(GITHUB_DOC)
        heading_texts = [h[3] for h in headings]
        # These are bash/python comments inside code blocks — must NOT appear as headings
        assert "JIT cache (replace cu129 with your CUDA version)" not in heading_texts
        assert "Single decode attention" not in heading_texts
        assert "Verify installation and view configuration" not in heading_texts

    def test_real_headings_still_detected_after_masking(self):
        headings = _find_headings(GITHUB_DOC)
        heading_texts = [h[3] for h in headings]
        assert "Why FlashInfer?" in heading_texts
        assert "Getting Started" in heading_texts
        assert "GPU Support" in heading_texts

    def test_github_chunk_count_reasonable(self):
        parents, children = _make_doc(GITHUB_DOC)
        # With 3 code blocks + tables + prose, should get a reasonable number
        assert len(children) >= 5
        # And code children should match our 3 code blocks
        code_children = [c for c in children if c.is_code]
        assert len(code_children) == 3


# ---------------------------------------------------------------------------
# TEST GROUP 3 — FIX 1: Flat-H1 arXiv section nesting
# ---------------------------------------------------------------------------

class TestFlatH1Nesting:
    def test_numbered_subsections_nested_under_parent(self):
        sections = parse_markdown_sections(ARXIV_DOC)
        paths = [s.section_path for s in sections]
        # "2.1 Hardware Performance" must appear NESTED under "2 Background"
        assert any("2 Background" in p and "2.1 Hardware" in p for p in paths), (
            f"Expected '2 Background > 2.1 Hardware...' path. Got: {paths}"
        )

    def test_2_2_nested_under_2_background(self):
        sections = parse_markdown_sections(ARXIV_DOC)
        paths = [s.section_path for s in sections]
        # 2.2 has level 2; GPU Memory Hierarchy also inherits level 2.
        # When the stack processes 2.2, it pops GPU Memory Hierarchy (same level),
        # then pops nothing else (2 Background is level 1), so 2.2 nests under
        # '2 Background'. Verify 2.2 is NOT a top-level standalone section.
        path_22 = next((p for p in paths if "2.2" in p), None)
        assert path_22 is not None, f"No path containing 2.2 found. Got: {paths}"
        # Must contain the parent '2 Background' — not be a standalone section
        assert "2 Background" in path_22, (
            f"2.2 should nest under '2 Background', got: {path_22!r}"
        )

    def test_section_3_is_top_level(self):
        sections = parse_markdown_sections(ARXIV_DOC)
        section_3 = next(
            (s for s in sections if s.heading.startswith("3") and "FlashAttention" in s.heading),
            None
        )
        assert section_3 is not None
        assert ">" not in section_3.section_path, (
            f"Section 3 should be top-level, got: {section_3.section_path!r}"
        )

    def test_logical_levels_assigned_from_numbering(self):
        sections = parse_markdown_sections(ARXIV_DOC)
        level_map = {s.heading: s.level for s in sections}
        # "1 Introduction" → level 1
        intro = next((v for k, v in level_map.items() if k.startswith("1 ")), None)
        assert intro == 1, f"Expected level 1 for '1 Introduction', got {intro}"
        # "2.1 Hardware Performance" → level 2
        hw = next((v for k, v in level_map.items() if "2.1" in k), None)
        assert hw == 2, f"Expected level 2 for '2.1 Hardware', got {hw}"

    def test_abstract_and_title_are_level_1(self):
        sections = parse_markdown_sections(ARXIV_DOC)
        abstract = next((s for s in sections if s.heading == "Abstract"), None)
        assert abstract is not None
        assert abstract.level == 1


# ---------------------------------------------------------------------------
# TEST GROUP 4 — Tables are atomic
# ---------------------------------------------------------------------------

class TestTablesAtomic:
    def test_arxiv_table_is_single_child(self):
        _, children = _make_doc(ARXIV_DOC)
        table_children = [c for c in children if c.is_table]
        assert len(table_children) == 1

    def test_table_contains_all_rows(self):
        _, children = _make_doc(ARXIV_DOC)
        t = next(c for c in children if c.is_table)
        assert "BERT" in t.text
        assert "GPT-2" in t.text

    def test_docs_table_is_single_child(self):
        _, children = _make_doc(DOCS_DOC)
        table_children = [c for c in children if c.is_table]
        assert len(table_children) >= 1

    def test_github_gpu_table_is_atomic(self):
        _, children = _make_doc(GITHUB_DOC)
        table_children = [c for c in children if c.is_table]
        assert len(table_children) == 1
        assert "Hopper" in table_children[0].text
        assert "Blackwell" in table_children[0].text


# ---------------------------------------------------------------------------
# TEST GROUP 5 — Code blocks are atomic
# ---------------------------------------------------------------------------

class TestCodeBlocksAtomic:
    def test_github_has_three_code_blocks(self):
        _, children = _make_doc(GITHUB_DOC)
        assert len([c for c in children if c.is_code]) == 3

    def test_bash_code_block_intact(self):
        _, children = _make_doc(GITHUB_DOC)
        bash_blocks = [c for c in children if c.is_code and "pip install" in c.text]
        assert bash_blocks, "bash install block not found"
        # The JIT cache comment should be INSIDE the code chunk, not a heading
        assert "JIT cache" in bash_blocks[0].text

    def test_python_code_block_intact(self):
        _, children = _make_doc(GITHUB_DOC)
        py_blocks = [c for c in children if c.is_code and "single_decode" in c.text]
        assert py_blocks
        assert "import torch" in py_blocks[0].text


# ---------------------------------------------------------------------------
# TEST GROUP 6 — parent_id linkage
# ---------------------------------------------------------------------------

class TestParentIdLinkage:
    def test_every_child_parent_id_exists_arxiv(self):
        parents, children = _make_doc(ARXIV_DOC)
        valid = _parent_ids(parents)
        for c in children:
            assert c.parent_id in valid

    def test_every_child_parent_id_exists_docs(self):
        parents, children = _make_doc(DOCS_DOC)
        valid = _parent_ids(parents)
        for c in children:
            assert c.parent_id in valid

    def test_every_child_parent_id_exists_github(self):
        parents, children = _make_doc(GITHUB_DOC)
        valid = _parent_ids(parents)
        for c in children:
            assert c.parent_id in valid

    def test_chunk_ids_unique_across_all_docs(self):
        _, c1 = _make_doc(ARXIV_DOC)
        _, c2 = _make_doc(DOCS_DOC)
        _, c3 = _make_doc(GITHUB_DOC)
        all_ids = [c.chunk_id for c in c1 + c2 + c3]
        assert len(all_ids) == len(set(all_ids))

    def test_children_section_matches_parent(self):
        parents, children = _make_doc(ARXIV_DOC)
        parent_map = {p.parent_id: p.section for p in parents}
        for c in children:
            assert c.section == parent_map[c.parent_id]


# ---------------------------------------------------------------------------
# TEST GROUP 7 — Token counts
# ---------------------------------------------------------------------------

class TestTokenCounts:
    def test_child_token_count_matches_tiktoken(self):
        enc = _enc()
        _, children = _make_doc(DOCS_DOC)
        for child in children:
            actual = len(enc.encode(child.text, disallowed_special=()))
            assert child.token_count == actual, (
                f"token_count={child.token_count} actual={actual} for {child.chunk_id}"
            )

    def test_no_paragraph_child_exceeds_max_tokens(self):
        long_text = "attention mechanism " * 200   # ~600 tokens
        content = f"# Doc\n\n## Big Section\n\n{long_text}\n\n## Short\n\nBrief.\n"
        _, children = _make_doc(content)
        for c in children:
            if c.is_code or c.is_table:
                continue
            assert c.token_count <= CHILD_MAX_TOKENS + 5, (
                f"Paragraph chunk {c.token_count} > {CHILD_MAX_TOKENS}"
            )

    def test_char_span_is_positive_width(self):
        _, children = _make_doc(ARXIV_DOC)
        for c in children:
            assert c.char_span[1] > c.char_span[0], (
                f"Zero-width span on {c.chunk_id}"
            )


# ---------------------------------------------------------------------------
# TEST GROUP 8 — No cross-heading chunks
# ---------------------------------------------------------------------------

class TestNoCrossHeadingChunks:
    def test_children_have_same_section_as_parent(self):
        parents, children = _make_doc(ARXIV_DOC)
        parent_sections = {p.parent_id: p.section for p in parents}
        for c in children:
            assert c.section == parent_sections[c.parent_id]

    def test_hardware_and_attention_sections_are_distinct(self):
        sections = parse_markdown_sections(ARXIV_DOC)
        paths = [s.section_path for s in sections]
        hw = [p for p in paths if "2.1" in p]
        attn = [p for p in paths if "2.2" in p]
        assert hw and attn
        assert hw[0] != attn[0]

    def test_docs_nested_sections_are_distinct(self):
        sections = parse_markdown_sections(DOCS_DOC)
        paths = [s.section_path for s in sections]
        workflows = [p for p in paths if "Workflows" in p]
        explicit = [p for p in paths if "Explicit" in p]
        assert workflows and explicit
        assert workflows[0] != explicit[0]


# ---------------------------------------------------------------------------
# TEST GROUP 9 — Internal helpers
# ---------------------------------------------------------------------------

class TestIterBlocks:
    def test_fenced_code_detected(self):
        body = "Some text.\n\n```python\nx = 1\n```\n\nMore text."
        blocks = list(_iter_blocks(body))
        assert any(b.kind == "code" for b in blocks)

    def test_table_detected(self):
        body = "Intro.\n\n| A | B |\n| - | - |\n| 1 | 2 |\n\nOutro."
        blocks = list(_iter_blocks(body))
        assert any(b.kind == "table" for b in blocks)

    def test_comment_inside_fence_stays_in_code_block(self):
        body = "```bash\n# this is a comment\npip install foo\n```"
        blocks = list(_iter_blocks(body))
        assert len(blocks) == 1
        assert blocks[0].kind == "code"
        assert "this is a comment" in blocks[0].text

    def test_prose_after_code_is_paragraph(self):
        body = "```bash\npip install foo\n```\n\nAfter the code block."
        blocks = list(_iter_blocks(body))
        kinds = [b.kind for b in blocks]
        assert "code" in kinds
        assert "paragraph" in kinds