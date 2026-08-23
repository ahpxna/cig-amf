#!/usr/bin/env bash
# Clone pinned external benchmark sources.  Installation is intentionally
# separate so a frozen CIG-AMF environment stays reproducible.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${CIG_EXTERNAL_ENVS_DIR:-$ROOT/external_envs}"
mkdir -p "$DEST"

clone_if_missing() {
  local name="$1"
  local url="$2"
  if [ -d "$DEST/$name/.git" ]; then
    echo "present: $name ($(git -C "$DEST/$name" rev-parse --short HEAD))"
  else
    git clone --depth 1 "$url" "$DEST/$name"
  fi
}

clone_if_missing flatland-rl https://github.com/flatland-association/flatland-rl.git
clone_if_missing robotic-warehouse https://github.com/semitable/robotic-warehouse.git
clone_if_missing CybORG https://github.com/cage-challenge/CybORG.git
clone_if_missing CityFlow https://github.com/cityflow-project/CityFlow.git

