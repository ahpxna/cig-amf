"""Inspect the single external-repository root and verify pinned revisions."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.external.registry import SPECS, external_root, repo_path, registry_payload
from envs.external.runtime import runtime_metadata


def _git_head(path: Path):
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _expected_revisions():
    path = Path(__file__).resolve().parent / "external_env_revisions.tsv"
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        name, revision, url = line.split("\t")
        rows[name] = {"revision": revision, "url": url}
    return rows


def status_payload():
    expected = _expected_revisions()
    payload = {"root": str(external_root()), "runtime": runtime_metadata(), "repositories": {}}
    for key, spec in SPECS.items():
        path = repo_path(key)
        head = _git_head(path)
        pin = expected.get(spec.repo_dir, {})
        payload["repositories"][key] = {
            "repo_dir": spec.repo_dir,
            "path": str(path),
            "exists": path.exists(),
            "head": head,
            "expected_revision": pin.get("revision"),
            "revision_match": bool(head and head == pin.get("revision")),
            "capabilities": vars(spec.capabilities),
        }
    payload["all_pins_match"] = all(
        item["revision_match"] for item in payload["repositories"].values()
    )
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    payload = status_payload()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["all_pins_match"] else 2)


if __name__ == "__main__":
    main()
