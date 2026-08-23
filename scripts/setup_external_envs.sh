#!/usr/bin/env bash
# Clone pinned external benchmark sources. Installation is intentionally
# separate so a frozen CIG-AMF environment stays reproducible.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${CIG_EXTERNAL_ENVS_DIR:-$ROOT/external_envs}"
mkdir -p "$DEST"

clone_pinned() {
  local name="$1"
  local revision="$2"
  local url="$3"
  if [ -d "$DEST/$name/.git" ]; then
    git -C "$DEST/$name" fetch --depth 1 origin "$revision"
  else
    git clone --no-checkout "$url" "$DEST/$name"
  fi
  git -C "$DEST/$name" checkout --detach "$revision"
  printf '%s\t%s\n' "$name" "$(git -C "$DEST/$name" rev-parse HEAD)"
}

while IFS=$'\t' read -r name revision url; do
  [ -z "$name" ] || clone_pinned "$name" "$revision" "$url"
done < "$ROOT/scripts/external_env_revisions.tsv"
