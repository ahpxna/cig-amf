"""Build a reproducible source manifest for archive or Git confirmatory runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

try:
    from exp_common import ROOT
except ModuleNotFoundError:
    from scripts.exp_common import ROOT

EXCLUDED_DIRS = {".git", "results", "__pycache__", ".pytest_cache"}
EXCLUDED_FILES = {"SOURCE_MANIFEST.json"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_entries(root):
    entries = []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for name in sorted(files):
            if name in EXCLUDED_FILES or name.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(current, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            entries.append({"path": rel, "sha256": sha256_file(path)})
    return entries


def tree_hash(entries):
    digest = hashlib.sha256()
    for item in entries:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_commit(root):
    try:
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "SOURCE_MANIFEST.json"))
    args = parser.parse_args(argv)
    entries = source_entries(ROOT)
    payload = {
        "manifest_protocol": "cig_amf_source_manifest_v1",
        "git_commit": git_commit(ROOT),
        "source_tree_sha256": tree_hash(entries),
        "requirements_sha256": (
            sha256_file(os.path.join(ROOT, "requirements.txt"))
            if os.path.isfile(os.path.join(ROOT, "requirements.txt")) else None
        ),
        "file_count": len(entries),
        "files": entries,
    }
    with open(os.path.abspath(args.out), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(os.path.abspath(args.out))


if __name__ == "__main__":
    main()
