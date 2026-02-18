# File: src/ira/ingest/parse_code_snippets.py

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ira.ingest.normalize import normalize_markdown


LANG_BY_EXT = {
    ".py": "python",
    ".ipynb": "json",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cu": "cpp",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".sh": "bash",
    ".ps1": "powershell",
}


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _read_text_any(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _pick(meta: Dict[str, Any], paths: List[List[str]]) -> Optional[str]:
    for path in paths:
        cur: Any = meta
        ok = True
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok and isinstance(cur, str) and cur.strip():
            return cur.strip()
    return None


def _detect_language(path: Path) -> str:
    return LANG_BY_EXT.get(path.suffix.lower(), "text")


def _list_candidate_files(raw_dir: Path) -> List[Path]:
    # Everything except meta.json
    files = []
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.lower() == "meta.json":
            continue
        files.append(p)
    return sorted(files, key=lambda x: x.as_posix())


def _resolve_target_files(raw_dir: Path, raw_meta: Dict[str, Any]) -> List[Path]:
    """
    Prefer the file specified by meta.source.path.
    If not found, fallback to artifacts keys, then to "all files".
    """
    candidates = _list_candidate_files(raw_dir)
    if not candidates:
        return []

    # 1) Prefer meta.source.path
    source_path = _pick(raw_meta, [["source", "path"]])
    if source_path:
        direct = raw_dir / source_path
        if direct.exists():
            return [direct]
        # common ingestion stores only basename at root; try matching basename
        base = Path(source_path).name
        for p in candidates:
            if p.name == base:
                return [p]

    # 2) Prefer artifacts keys (if present)
    artifacts = raw_meta.get("artifacts")
    if isinstance(artifacts, dict) and artifacts:
        artifact_keys = [k for k in artifacts.keys() if isinstance(k, str)]
        resolved: List[Path] = []
        for k in artifact_keys:
            direct = raw_dir / k
            if direct.exists():
                resolved.append(direct)
                continue
            base = Path(k).name
            for p in candidates:
                if p.name == base:
                    resolved.append(p)
                    break
        if resolved:
            # keep stable order, remove duplicates
            seen = set()
            out = []
            for p in resolved:
                if p.as_posix() not in seen:
                    seen.add(p.as_posix())
                    out.append(p)
            return out

    # 3) Fallback: include all files
    return candidates


@dataclass(frozen=True)
class ExtractedFile:
    path: str
    language: str
    line_start: int
    line_end: int
    sha256: str
    artifact_sha256: Optional[str] = None


def parse_github_doc(raw_doc_dir: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Parse a GitHub snapshot folder (data/raw/<doc_id>/) into Markdown + processed meta.

    Supports your meta schema:
      kind: "github"
      source: {repo, ref, resolved_commit, path, raw_url}
      seed: {title, tags}
      artifacts: {<filename>: {sha256, bytes}}
    """
    meta_path = raw_doc_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {raw_doc_dir}")

    raw_meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))

    doc_id = raw_doc_dir.name
    title = _pick(raw_meta, [["seed", "title"], ["title"]]) or doc_id

    repo = _pick(raw_meta, [["source", "repo"], ["seed", "repo"]])
    ref = _pick(raw_meta, [["source", "ref"], ["seed", "ref"]])
    commit = _pick(raw_meta, [["source", "resolved_commit"]])
    source_path = _pick(raw_meta, [["source", "path"], ["seed", "path"]])
    raw_url = _pick(raw_meta, [["source", "raw_url"], ["source", "url"], ["seed", "url"]])

    tags = raw_meta.get("seed", {}).get("tags")
    if not isinstance(tags, list):
        tags = []

    artifacts = raw_meta.get("artifacts")
    artifact_sha_by_name: Dict[str, str] = {}
    if isinstance(artifacts, dict):
        for k, v in artifacts.items():
            if isinstance(k, str) and isinstance(v, dict):
                sha = v.get("sha256")
                if isinstance(sha, str) and sha:
                    artifact_sha_by_name[Path(k).name] = sha

    files = _resolve_target_files(raw_doc_dir, raw_meta)
    if not files:
        raise RuntimeError(f"No files found to parse in {raw_doc_dir}")

    extracted: List[ExtractedFile] = []
    md_parts: List[str] = []

    # Canonical header for stable citations
    md_parts.append(f"# {title}")
    md_parts.append(f"**Doc ID:** {doc_id}")
    if raw_url:
        md_parts.append(f"**Source:** {raw_url}")
    if repo:
        repo_line = repo
        if commit:
            repo_line += f"@{commit}"
        elif ref:
            repo_line += f"@{ref}"
        md_parts.append(f"**Repo:** {repo_line}")
    if source_path:
        md_parts.append(f"**Path:** {source_path}")
    if tags:
        md_parts.append(f"**Tags:** {', '.join(str(t) for t in tags)}")
    md_parts.append("")

    for p in files:
        rel = p.relative_to(raw_doc_dir).as_posix()
        lang = _detect_language(p)
        text = _read_text_any(p)
        # line_end = max(1, text.count("\n") + 1)
        line_end = max(1, len(text.splitlines()))

        sha = _sha256_bytes(p.read_bytes())
        artifact_sha = artifact_sha_by_name.get(p.name)

        extracted.append(
            ExtractedFile(
                path=rel,
                language=lang,
                line_start=1,
                line_end=line_end,
                sha256=sha,
                artifact_sha256=artifact_sha,
            )
        )

        md_parts.append(f"## File: {rel}")
        md_parts.append(f"**Lines:** 1-{line_end}")
        md_parts.append(f"**SHA256:** {sha}")
        if artifact_sha and artifact_sha != sha:
            md_parts.append(f"**Artifact SHA256 (meta):** {artifact_sha}")
        md_parts.append("")

        if p.suffix.lower() == ".md":
            # Keep markdown content as markdown (no code fences)
            md_parts.append(text.rstrip("\n"))
            md_parts.append("")
        else:
            # Wrap non-md files in fenced code blocks
            md_parts.append(f"```{lang}")
            md_parts.append(text.rstrip("\n"))
            md_parts.append("```")
            md_parts.append("")

    markdown = "\n".join(md_parts)
    markdown = normalize_markdown(markdown)

    # content richness metric
    content_non_ws_chars = len(re.sub(r"\s+", "", markdown))

    processed_meta: Dict[str, Any] = {
        "doc_id": doc_id,
        "title": title,
        "source_type": "github",
        "source_url": raw_url,
        "repo": repo,
        "ref": ref,
        "commit": commit,
        "path": source_path,
        "tags": tags,
        "parser": {"name": "parse_code_snippets", "version": 2},
        "content_non_ws_chars": content_non_ws_chars,
        "extracted_files": [
            {
                "path": e.path,
                "language": e.language,
                "line_start": e.line_start,
                "line_end": e.line_end,
                "sha256": e.sha256,
                "artifact_sha256": e.artifact_sha256,
            }
            for e in extracted
        ],
        "raw_meta": raw_meta,
    }

    return markdown, processed_meta
