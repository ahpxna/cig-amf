"""External-runtime isolation and filesystem helpers.

Pinned benchmark repositories can have dependency constraints that intentionally
conflict with the main CIG-AMF research environment.  In particular, the pinned
Flatland revision requires ``numpy<2``.  Keep those dependencies in a managed
runtime under ``external_envs/runtime`` and re-exec external commands there.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable, Optional, Tuple


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
    """Return whether the shared external runtime is in the supported ABI band.

    NumPy 1.26.x, required by the pinned Flatland ``numpy<2`` contract, has a
    reliable wheel path through Python 3.12.  We therefore fail early on newer
    interpreters rather than attempting a source build inside experiment setup.
    """
    major, minor = int(version[0]), int(version[1])
    return major == 3 and 10 <= minor <= 12


def maybe_reexec_in_external_runtime() -> None:
    """Re-exec the current external command in the managed runtime if ready."""
    if os.environ.get("CIG_EXTERNAL_RUNTIME_ACTIVE") == "1":
        return
    target = runtime_python()
    if not runtime_ready():
        return
    try:
        same = target.resolve() == Path(sys.executable).resolve()
    except OSError:
        same = False
    if same:
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
    """Ensure ``path`` is a directory, preserving any legacy file collision.

    Older external-suite invocations accepted a JSON *file* such as
    ``results/external_flatland``.  Newer commands use that same stem as a
    directory.  Preserve the old file under a deterministic ``.legacy-file``
    name instead of crashing with ``FileExistsError`` or deleting evidence.
    """
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
