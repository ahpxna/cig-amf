#!/usr/bin/env python3
"""Content-address the runnable source tree for confirmatory provenance.

Generated results, caches, environments, VCS metadata, external checkouts and
local notes are excluded.  The digest is stable for a pristine ZIP+patch tree
and is rechecked after the experiment to detect source mutation mid-run.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git", "results", "__pycache__", ".pytest_cache", ".mypy_cache",
    "cig-env", "venv", "env", "external_envs", "docs",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".so"}


def source_tree_sha256(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append((rel.as_posix(), path))
    for rel, path in sorted(files):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    print(source_tree_sha256())
