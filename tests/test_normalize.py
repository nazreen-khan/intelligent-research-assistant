from ira.ingest.normalize import normalize_text, normalize_markdown


def test_normalize_text_fixes_hyphenation_and_soft_hyphen():
    raw = "quantiza-\n tion and hy\u00adphen"
    out = normalize_text(raw)
    assert "quantization" in out
    assert "hyphen" in out
    assert "\u00ad" not in out


def test_normalize_markdown_preserves_code_blocks():
    md = """# Title

    Text with quantiza-
    tion.

    ```python
    x = "quantiza-\\n tion"
    """
    out = normalize_markdown(md)

    # Outside code block is fixed
    assert "quantization" in out

    # Inside code block should remain unchanged
    assert 'x = "quantiza-\\n tion"' in out