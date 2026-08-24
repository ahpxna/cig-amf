"""Isolation helpers for pinned external benchmark runtimes.

The CIG-AMF main environment and the pinned external benchmarks intentionally
have incompatible dependency constraints (notably Flatland's ``numpy<2``).
External benchmark commands therefore run in a managed Python 3.10-3.12 venv
under ``external_envs/runtime`` and never silently fall back to the main venv.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = ROOT / "external_envs"
RUNTIME_DIRNAME = "runtime"
READY_MARKER = ".cig-external-runtime-ready"


def external_root() -> Path:
    root = Path(os.environ.get("CIG_EXTERNAL_ENVS_DIR", DEFAULT_EXTERNAL_ROOT)).resolve()
    return root.parent if root.name == "repos" else root


def runtime_dir() -> Path:
    return external_root() / RUNTIME_DIRNAME


def runtime_python() -> Path:
    name = "python.exe" if os.name == "nt" else "python"
    subdir = "Scripts" if os.name == "nt" else "bin"
    return runtime_dir() / subdir / name


def runtime_ready() -> bool:
    return runtime_python().is_file() and (runtime_dir() / READY_MARKER).is_file()


def external_python_supported(version: Tuple[int, int]) -> bool:
    major, minor = int(version[0]), int(version[1])
    return major == 3 and 10 <= minor <= 12


def setup_hint() -> str:
    return (
        "external benchmark runtime is not ready.\n"
        "Create it with Python 3.10-3.12 (Python 3.12 recommended):\n"
        "  CIG_EXTERNAL_PYTHON=/path/to/python3.12 "
        "bash scripts/setup_external_envs.sh --install --recreate-runtime\n"
        f"Expected managed runtime: {runtime_dir()}"
    )


def maybe_reexec_in_external_runtime(*, require_ready: bool = False) -> None:
    """Re-exec this command in the managed external runtime when available.

    ``require_ready=True`` is the correct mode for any command that imports or
    constructs an external benchmark.  It prevents accidental execution inside
    the main CIG environment, which otherwise yields misleading missing-package
    errors and can trigger incompatible dependency resolution.
    """
    active = os.environ.get("CIG_EXTERNAL_RUNTIME_ACTIVE") == "1"
    if active:
        if require_ready and not runtime_ready():
            raise SystemExit(setup_hint())
        return

    if not runtime_ready():
        if require_ready:
            raise SystemExit(setup_hint())
        return

    target = runtime_python()
    try:
        same = target.resolve() == Path(sys.executable).resolve()
    except OSError:
        same = False
    if same:
        os.environ["CIG_EXTERNAL_RUNTIME_ACTIVE"] = "1"
        os.environ["CIG_EXTERNAL_RUNTIME_PYTHON"] = str(target)
        return

    env = os.environ.copy()
    env["CIG_EXTERNAL_RUNTIME_ACTIVE"] = "1"
    env["CIG_EXTERNAL_RUNTIME_PYTHON"] = str(target)
    os.execve(str(target), [str(target), *sys.argv], env)


def _next_legacy_path(path: Path) -> Path:
    base = path.with_name(path.name + ".legacy-file")
    if not base.exists():
        return base
    for idx in range(1, 10000):
        candidate = path.with_name(path.name + f".legacy-file.{idx}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate a legacy path for {path}")


def ensure_directory(path: Path) -> Optional[Path]:
    """Ensure ``path`` is a directory while preserving a colliding legacy file."""
    path = Path(path)
    migrated = None
    if path.exists() and not path.is_dir():
        migrated = _next_legacy_path(path)
        path.replace(migrated)
    path.mkdir(parents=True, exist_ok=True)
    return migrated


def ensure_output_file_parent(path: Path) -> Optional[Path]:
    return ensure_directory(Path(path).parent)


def runtime_metadata() -> dict:
    metadata_path = runtime_dir() / "runtime.json"
    payload = {
        "path": str(runtime_dir()),
        "python": str(runtime_python()),
        "ready": runtime_ready(),
    }
    if metadata_path.is_file():
        try:
            payload["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload["metadata_error"] = "invalid runtime.json"
    return payload
