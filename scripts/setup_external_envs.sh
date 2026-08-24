#!/usr/bin/env bash
# Manage all pinned external benchmark repositories under one canonical root.
# Optional installation is isolated from the main CIG environment because the
# pinned Flatland revision requires numpy<2.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTERNAL_ROOT="${CIG_EXTERNAL_ENVS_DIR:-$ROOT/external_envs}"
if [ "$(basename "$EXTERNAL_ROOT")" = "repos" ]; then
  EXTERNAL_ROOT="$(dirname "$EXTERNAL_ROOT")"
fi
DEST="$EXTERNAL_ROOT/repos"
RUNTIME="$EXTERNAL_ROOT/runtime"
INSTALL=0
RECREATE=0

usage() {
  cat >&2 <<EOF
usage: $0 [--install] [--recreate-runtime]

--install            create/update the isolated external runtime and install all
                     pinned benchmark packages there
--recreate-runtime   delete only the managed external_envs/runtime venv first

Interpreter selection for --install:
  CIG_EXTERNAL_PYTHON=/path/to/python3.12 $0 --install
Otherwise python3.12, python3.11, then python3.10 are auto-detected.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --install) INSTALL=1 ;;
    --recreate-runtime) RECREATE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
  shift
done
mkdir -p "$DEST"

clone_pinned() {
  local name="$1" revision="$2" url="$3"
  local legacy="$EXTERNAL_ROOT/$name" target="$DEST/$name"
  if [ ! -e "$target" ] && [ -d "$legacy/.git" ]; then
    mv "$legacy" "$target"
  fi
  if [ -d "$target/.git" ]; then
    git -C "$target" fetch --depth 1 origin "$revision"
  else
    git clone --no-checkout "$url" "$target"
  fi
  git -C "$target" checkout --detach "$revision"
  local actual
  actual="$(git -C "$target" rev-parse HEAD)"
  if [ "$actual" != "$revision" ]; then
    echo "revision mismatch for $name: expected $revision got $actual" >&2
    exit 3
  fi
  printf '%s\t%s\t%s\n' "$name" "$actual" "$target"
}

while IFS=$'\t' read -r name revision url; do
  [ -z "$name" ] || clone_pinned "$name" "$revision" "$url"
done < "$ROOT/scripts/external_env_revisions.tsv"

choose_external_python() {
  local candidates=()
  if [ -n "${CIG_EXTERNAL_PYTHON:-}" ]; then
    candidates+=("$CIG_EXTERNAL_PYTHON")
  else
    candidates+=(python3.12 python3.11 python3.10)
  fi
  local candidate resolved version ok
  for candidate in "${candidates[@]}"; do
    if ! resolved="$(command -v "$candidate" 2>/dev/null)"; then
      continue
    fi
    version="$($resolved -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    ok="$($resolved -c 'import sys; print(int(sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12))')"
    if [ "$ok" = "1" ]; then
      printf '%s\n' "$resolved"
      return 0
    fi
    echo "[external-runtime] skipping $resolved (Python $version; need 3.10-3.12)" >&2
  done
  return 1
}

if [ "$INSTALL" -eq 1 ]; then
  if ! BASE_PYTHON="$(choose_external_python)"; then
    CURRENT="$(python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || true)"
    cat >&2 <<EOF
ERROR: no compatible external-runtime interpreter found.
Current project interpreter: ${CURRENT:-unknown}
Pinned Flatland requires numpy<2; do not install it into the Python 3.14 CIG env.

On Homebrew macOS, install Python 3.12 and retry:
  brew install python@3.12
  CIG_EXTERNAL_PYTHON="\$(brew --prefix python@3.12)/bin/python3.12" \\
    bash scripts/setup_external_envs.sh --install --recreate-runtime
EOF
    exit 4
  fi

  if [ "$RECREATE" -eq 1 ] && [ -d "$RUNTIME" ]; then
    rm -rf "$RUNTIME"
  fi
  if [ ! -x "$RUNTIME/bin/python" ]; then
    "$BASE_PYTHON" -m venv "$RUNTIME"
  fi
  RUNTIME_PY="$RUNTIME/bin/python"
  RUNTIME_VERSION="$($RUNTIME_PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if ! "$RUNTIME_PY" -c 'import sys; raise SystemExit(0 if sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12 else 1)'; then
    echo "managed runtime uses incompatible Python $RUNTIME_VERSION; rerun with --recreate-runtime" >&2
    exit 5
  fi

  rm -f "$RUNTIME/.cig-external-runtime-ready"
  "$RUNTIME_PY" -m pip install --upgrade pip setuptools wheel
  "$RUNTIME_PY" -m pip install -r "$ROOT/requirements-external.txt"

  FAILED_INSTALLS=""
  INSTALLED=""
  while IFS=$'\t' read -r name revision url; do
    [ -z "$name" ] && continue
    target="$DEST/$name"
    echo "[external-runtime] installing $name into $RUNTIME"
    if "$RUNTIME_PY" -m pip install -e "$target"; then
      INSTALLED="$INSTALLED $name"
    else
      FAILED_INSTALLS="$FAILED_INSTALLS $name"
      echo "[external-runtime] install failed for $name; continuing so other benchmarks remain usable" >&2
    fi
  done < "$ROOT/scripts/external_env_revisions.tsv"

  CIG_RUNTIME_DIR="$RUNTIME" CIG_INSTALLED="$INSTALLED" CIG_FAILED_INSTALLS="$FAILED_INSTALLS" "$RUNTIME_PY" - <<'PY'
import importlib, json, os, pathlib, platform, sys
module_by_repo = {
    "flatland-rl": "flatland",
    "robotic-warehouse": "rware",
    "CybORG": "CybORG",
    "CityFlow": "cityflow",
}
imports = {}
for repo, module in module_by_repo.items():
    try:
        importlib.import_module(module)
        imports[repo] = {"ok": True, "module": module}
    except Exception as exc:
        imports[repo] = {"ok": False, "module": module, "error": f"{type(exc).__name__}: {exc}"}
path = pathlib.Path(os.environ["CIG_RUNTIME_DIR"]) / "runtime.json"
payload = {
    "python": sys.executable,
    "python_version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "purpose": "isolated pinned external benchmark runtime",
    "installed_attempts": os.environ.get("CIG_INSTALLED", "").split(),
    "failed_install_attempts": os.environ.get("CIG_FAILED_INSTALLS", "").split(),
    "imports": imports,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  touch "$RUNTIME/.cig-external-runtime-ready"
  echo "[external-runtime] READY: $RUNTIME_PY ($RUNTIME_VERSION)"
  if [ -n "$FAILED_INSTALLS" ]; then
    echo "[external-runtime] WARNING: failed installs:$FAILED_INSTALLS" >&2
  fi
fi

python "$ROOT/scripts/external_env_manager.py" status --json "$EXTERNAL_ROOT/manifest.json"
if [ "$INSTALL" -eq 1 ] && [ -n "${FAILED_INSTALLS:-}" ]; then
  exit 6
fi
