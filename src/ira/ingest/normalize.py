from __future__ import annotations

import re
import unicodedata
from typing import Iterable


# Common Unicode ligatures found in PDFs
_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

# Zero-width / invisible chars that often pollute extracted text
_INVISIBLES = [
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # BOM
]

_SOFT_HYPHEN = "\u00ad"


def _normalize_newlines(text: str) -> str:
    # Normalize Windows/Mac newlines -> \n
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _fix_hyphenation_linebreaks(text: str) -> str:
    """
    Fix classic PDF hyphenation:
        "quantiza-\n tion" -> "quantization"
        "quantiza-\n    tion" -> "quantization"
    Only merges when both sides look like letters/numbers to reduce false merges.
    """
    return re.sub(r"([0-9A-Za-z])-\n[ \t]*([0-9A-Za-z])", r"\1\2", text)



def _collapse_blank_lines(text: str, max_consecutive: int = 2) -> str:
    if max_consecutive < 2:
        max_consecutive = 2
    # Convert 3+ newlines into exactly 2 newlines
    return re.sub(r"\n{3,}", "\n" * max_consecutive, text)


def normalize_text(text: str) -> str:
    """
    Normalize plain extracted text (PDF text, raw HTML text, etc.).
    Deterministic + safe for retrieval.
    """
    if not text:
        return ""

    text = _normalize_newlines(text)

    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)

    # Remove invisibles + soft hyphen
    for ch in _INVISIBLES:
        text = text.replace(ch, "")
    text = text.replace(_SOFT_HYPHEN, "")

    # Replace ligatures
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)

    # Fix hyphenation line breaks
    text = _fix_hyphenation_linebreaks(text)

    # Normalize whitespace
    text = _strip_trailing_whitespace(text)
    text = _collapse_blank_lines(text, max_consecutive=2)

    return text.strip()


def _iter_markdown_blocks(md: str) -> Iterable[tuple[bool, str]]:
    """
    Yield (is_code_block, block_text) by scanning fenced code blocks.
    Supports ``` and ~~~ fences. Keeps code blocks untouched.
    """
    lines = md.split("\n")
    buf: list[str] = []
    in_code = False
    fence = None  # "```" or "~~~"

    def flush():
        nonlocal buf
        if buf:
            yield (in_code, "\n".join(buf))
            buf = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Fence start/end detection (must be at line start ignoring whitespace)
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = "```" if stripped.startswith("```") else "~~~"
            if not in_code:
                # flush non-code content
                yield from flush()
                in_code = True
                fence = marker
                buf.append(line)
            else:
                # only close if matches opening fence type
                if fence == marker:
                    buf.append(line)
                    yield from flush()
                    in_code = False
                    fence = None
                else:
                    buf.append(line)
            i += 1
            continue

        buf.append(line)
        i += 1

    # Flush remaining
    yield from flush()


def normalize_markdown(md: str) -> str:
    """
    Normalize Markdown while preserving fenced code blocks exactly.
    Applies text normalization only outside code blocks.
    """
    if not md:
        return ""

    md = _normalize_newlines(md)

    parts: list[str] = []
    for is_code, block in _iter_markdown_blocks(md):
        if is_code:
            # Keep code blocks as-is (except newline normalization already done)
            parts.append(block)
        else:
            parts.append(normalize_text(block))

    # Re-join, then lightly clean any extra blank lines created by strip()
    out = "\n\n".join(p for p in parts if p != "")
    out = _collapse_blank_lines(out, max_consecutive=2)
    return out.strip()
