# from __future__ import annotations

# import json
# from pathlib import Path

# from ira.ingest.parse_code_snippets import parse_code_doc


# def test_parse_code_doc_single_file(tmp_path: Path):
#     raw_dir = tmp_path / "gh_test_doc"
#     raw_dir.mkdir(parents=True, exist_ok=True)

#     (raw_dir / "meta.json").write_text(
#         json.dumps(
#             {
#                 "doc_id": "gh_test_doc",
#                 "title": "Test Repo File",
#                 "source_type": "github",
#                 "repo": "https://github.com/example/repo",
#                 "commit": "abc123",
#                 "path": "src/example.py",
#                 "source_url": "https://github.com/example/repo/blob/abc123/src/example.py",
#             }
#         ),
#         encoding="utf-8",
#     )

#     (raw_dir / "example.py").write_text(
#         "def hello():\n"
#         "    return 'hi'\n",
#         encoding="utf-8",
#     )

#     md, meta = parse_code_doc(raw_dir)

#     assert md.startswith("# Test Repo File")
#     assert "## File: example.py" in md
#     assert "**Lines:** 1-2" in md
#     assert "```python" in md
#     assert "def hello():" in md

#     assert meta["doc_id"] == "gh_test_doc"
#     assert meta["source_type"] == "github"
#     assert meta["repo"] == "https://github.com/example/repo"
#     assert meta["commit"] == "abc123"
#     assert meta["extracted_files"][0]["path"] == "example.py"
#     assert meta["extracted_files"][0]["line_end"] == 2


# def test_parse_code_doc_multiple_files(tmp_path: Path):
#     raw_dir = tmp_path / "gh_multi"
#     raw_dir.mkdir(parents=True, exist_ok=True)

#     (raw_dir / "meta.json").write_text(
#         json.dumps({"doc_id": "gh_multi", "title": "Multi File Doc", "source_type": "github"}),
#         encoding="utf-8",
#     )

#     (raw_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
#     (raw_dir / "b.cpp").write_text("int main(){return 0;}\n", encoding="utf-8")

#     md, meta = parse_code_doc(raw_dir)

#     assert "## File: a.py" in md
#     assert "```python" in md
#     assert "## File: b.cpp" in md
#     assert "```cpp" in md

#     paths = [f["path"] for f in meta["extracted_files"]]
#     assert "a.py" in paths and "b.cpp" in paths

# File: tests/test_parse_code_snippets.py

from __future__ import annotations

import json
from pathlib import Path

from ira.ingest.parse_code_snippets import parse_github_doc


def test_parse_github_doc_markdown_readme(tmp_path: Path):
    raw_dir = tmp_path / "gh_unsloth_readme"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "meta.json").write_text(
        json.dumps(
            {
                "kind": "github",
                "source": {
                    "repo": "unslothai/unsloth",
                    "ref": "main",
                    "resolved_commit": "b0361918",
                    "path": "README.md",
                    "raw_url": "https://raw.githubusercontent.com/unslothai/unsloth/b0361918/README.md",
                },
                "seed": {"title": "Unsloth README", "tags": ["unsloth", "finetuning"]},
                "artifacts": {"README.md": {"sha256": "x" * 64, "bytes": 10}},
            }
        ),
        encoding="utf-8",
    )

    (raw_dir / "README.md").write_text(
        "# Unsloth\n\nHello world.\n\n```python\nx=1\n```\n",
        encoding="utf-8",
    )

    md, meta = parse_github_doc(raw_dir)

    assert md.startswith("# Unsloth README")
    assert "**Repo:** unslothai/unsloth@b0361918" in md
    assert "**Path:** README.md" in md
    assert "## File: README.md" in md
    assert "# Unsloth" in md  # original markdown preserved
    assert "```python" in md  # fenced blocks preserved

    assert meta["source_type"] == "github"
    assert meta["repo"] == "unslothai/unsloth"
    assert meta["commit"] == "b0361918"
    assert meta["path"] == "README.md"
    assert meta["extracted_files"][0]["path"] == "README.md"


def test_parse_github_doc_code_file_fenced(tmp_path: Path):
    raw_dir = tmp_path / "gh_code"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "meta.json").write_text(
        json.dumps(
            {
                "kind": "github",
                "source": {
                    "repo": "huggingface/peft",
                    "ref": "main",
                    "resolved_commit": "f6a7e678",
                    "path": "src/foo.py",
                    "raw_url": "https://raw.githubusercontent.com/huggingface/peft/f6a7e678/src/foo.py",
                },
                "seed": {"title": "PEFT Foo", "tags": ["peft"]},
                "artifacts": {"foo.py": {"sha256": "y" * 64, "bytes": 10}},
            }
        ),
        encoding="utf-8",
    )

    (raw_dir / "foo.py").write_text("def hello():\n    return 1\n", encoding="utf-8")

    md, meta = parse_github_doc(raw_dir)

    assert "## File: foo.py" in md
    assert "**Lines:** 1-2" in md
    assert "```python" in md
    assert "def hello():" in md

    assert meta["repo"] == "huggingface/peft"
    assert meta["commit"] == "f6a7e678"
    assert len(meta["extracted_files"]) == 1
    assert meta["extracted_files"][0]["line_end"] == 2
