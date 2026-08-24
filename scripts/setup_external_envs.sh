#!/usr/bin/env bash
# Clone/migrate pinned external benchmarks into one canonical repository root and
# optionally install them into an isolated Python 3.10-3.12 runtime.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTERNAL_ROOT="${CIG_EXTERNAL_ENVS_DIR:-$ROOT/external_envs}"
if [ "$(basename "$EXTERNAL_ROOT")" = "repos" ]; then
  EXTERNAL_ROOT="$(dirname "$EXTERNAL_ROOT")"
fi
DEST="$EXTERNAL_ROOT/repos"
LEGACY_ARCHIVE="$EXTERNAL_ROOT/legacy_repos"
RUNTIME="$EXTERNAL_ROOT/runtime"
INSTALL=0
RECREATE=0

usage() {
  cat >&2 <<EOF
usage: $0 [--install] [--recreate-runtime]

--install            install pinned benchmark packages into the isolated
                     external_envs/runtime environment
--recreate-runtime   recreate only the managed external runtime

For installation use Python 3.10-3.12 (3.12 recommended):
  CIG_EXTERNAL_PYTHON=/path/to/python3.12 $0 --install --recreate-runtime
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

archive_dirty_legacy() {
  name="$1"
  legacy="$2"
  mkdir -p "$LEGACY_ARCHIVE"
  stamp="$(date +%Y%m%d-%H%M%S)"
  target="$LEGACY_ARCHIVE/${name}-${stamp}"
  suffix=0
  while [ -e "$target" ]; do
    suffix=$((suffix + 1))
    target="$LEGACY_ARCHIVE/${name}-${stamp}-${suffix}"
  done
  mv "$legacy" "$target"
  echo "[external-repos] preserved dirty/ divergent legacy checkout: $target" >&2
}

reconcile_legacy() {
  name="$1"
  legacy="$EXTERNAL_ROOT/$name"
  target="$DEST/$name"
  [ -d "$legacy" ] || return 0
  [ "$legacy" != "$target" ] || return 0

  if [ ! -e "$target" ]; then
    mv "$legacy" "$target"
    echo "[external-repos] migrated $name -> $target" >&2
    return 0
  fi

  # Both paths exist. Remove only an unmodified duplicate at the same commit;
  # otherwise preserve it under legacy_repos instead of losing user changes.
  if [ -d "$legacy/.git" ] && [ -d "$target/.git" ]; then
    legacy_head="$(git -C "$legacy" rev-parse HEAD 2>/dev/null || true)"
    target_head="$(git -C "$target" rev-parse HEAD 2>/dev/null || true)"
    legacy_dirty="$(git -C "$legacy" status --porcelain 2>/dev/null || true)"
    if [ -n "$legacy_head" ] && [ "$legacy_head" = "$target_head" ] && [ -z "$legacy_dirty" ]; then
      rm -rf "$legacy"
      echo "[external-repos] removed duplicate clean legacy checkout: $legacy" >&2
      return 0
    fi
  fi
  archive_dirty_legacy "$name" "$legacy"
}

clone_pinned() {
  name="$1"; revision="$2"; url="$3"
  reconcile_legacy "$name"
  target="$DEST/$name"
  if [ -d "$target/.git" ]; then
    git -C "$target" fetch --depth 1 origin "$revision"
  elif [ -e "$target" ]; then
    echo "ERROR: canonical path exists but is not a git checkout: $target" >&2
    exit 3
  else
    git clone --no-checkout "$url" "$target"
  fi
  git -C "$target" checkout --detach "$revision"
  actual="$(git -C "$target" rev-parse HEAD)"
  if [ "$actual" != "$revision" ]; then
    echo "revision mismatch for $name: expected $revision got $actual" >&2
    exit 3
  fi
  printf '%s\t%s\t%s\n' "$name" "$actual" "$target"
}

while IFS="$(printf '\t')" read -r name revision url; do
  [ -z "$name" ] || clone_pinned "$name" "$revision" "$url"
done < "$ROOT/scripts/external_env_revisions.tsv"

choose_external_python() {
  if [ -n "${CIG_EXTERNAL_PYTHON:-}" ]; then
    candidates="$CIG_EXTERNAL_PYTHON"
  else
    candidates="python3.12 python3.11 python3.10"
  fi
  for candidate in $candidates; do
    resolved="$(command -v "$candidate" 2>/dev/null || true)"
    [ -n "$resolved" ] || continue
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
  BASE_PYTHON="$(choose_external_python || true)"
  if [ -z "$BASE_PYTHON" ]; then
    CURRENT="$(python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || true)"
    cat >&2 <<EOF
ERROR: no compatible external-runtime interpreter found.
Current project interpreter: ${CURRENT:-unknown}
Pinned Flatland requires numpy<2; do not install external repos into the main Python 3.14 environment.

Homebrew macOS:
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
  if ! "$RUNTIME_PY" -c 'import sys; raise SystemExit(0 if sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12 else 1)'; then
    echo "managed runtime has an unsupported Python; rerun with --recreate-runtime" >&2
    exit 5
  fi

  rm -f "$RUNTIME/.cig-external-runtime-ready"
  "$RUNTIME_PY" -m pip install --upgrade pip 'setuptools<81' wheel
  "$RUNTIME_PY" -m pip install -r "$ROOT/requirements-external.txt"

  INSTALLED=""
  FAILED_INSTALLS=""
  while IFS="$(printf '\t')" read -r name revision url; do
    [ -z "$name" ] && continue
    target="$DEST/$name"
    echo "[external-runtime] installing $name into $RUNTIME"
    if "$RUNTIME_PY" -m pip install -e "$target"; then
      INSTALLED="$INSTALLED $name"
    else
      FAILED_INSTALLS="$FAILED_INSTALLS $name"
      echo "[external-runtime] install failed for $name; other benchmarks remain usable" >&2
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
        imports[repo] = {
            "ok": False,
            "module": module,
            "error": f"{type(exc).__name__}: {exc}",
        }
payload = {
    "python": sys.executable,
    "python_version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "purpose": "isolated pinned external benchmark runtime",
    "installed_attempts": os.environ.get("CIG_INSTALLED", "").split(),
    "failed_install_attempts": os.environ.get("CIG_FAILED_INSTALLS", "").split(),
    "imports": imports,
}
path = pathlib.Path(os.environ["CIG_RUNTIME_DIR"]) / "runtime.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  touch "$RUNTIME/.cig-external-runtime-ready"
  echo "[external-runtime] runtime created: $RUNTIME_PY"
  if [ -n "$FAILED_INSTALLS" ]; then
    echo "[external-runtime] WARNING: failed installs:$FAILED_INSTALLS" >&2
  fi
fi

python "$ROOT/scripts/external_env_manager.py" status --json "$EXTERNAL_ROOT/manifest.json"
if [ "$INSTALL" -eq 1 ] && [ -n "${FAILED_INSTALLS:-}" ]; then
  exit 6
fi
