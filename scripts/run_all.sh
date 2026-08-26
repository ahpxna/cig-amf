#!/usr/bin/env bash
# Execute the complete CIG-AMF verification manifest in an isolated run root.
#
#   bash scripts/run_all.sh
#   bash scripts/run_all.sh --quick
#   bash scripts/run_all.sh --claims-only
#   bash scripts/run_all.sh --with-legacy-h3   # optional diagnostic only
#   bash scripts/run_all.sh --run-id 20260821_confirmatory
#
# Confirmatory mode uses eight paired seeds for every hypothesis panel.
# Runtime settings can be overridden without editing this file:
#
#   CIG_H1_SEEDS="0 1 2 3 4 5 6 7" CIG_RUN_SEEDS="0 1 2 3 4" bash scripts/run_all.sh
#   CIG_H2_EPISODES=400 CIG_H2_PRETRAIN_EPISODES=60 CIG_H3_EPISODES=200 bash scripts/run_all.sh
#
# Exit codes distinguish execution from evidence:
#   0: all commands completed and all hypothesis gates were supported
#   2: an operational/protocol failure made the run invalid
#   3: the run was complete, but at least one scientific claim was unsupported
set -uo pipefail

CIG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CIG_ROOT"

if [ -x "$CIG_ROOT/cig-env/bin/python" ]; then
  CIG_PYTHON="$CIG_ROOT/cig-env/bin/python"
else
  CIG_PYTHON="${CIG_PYTHON:-python3}"
fi

CIG_QUICK=0
CIG_CLAIMS_ONLY=0
CIG_RUN_LEGACY_H3="${CIG_RUN_LEGACY_H3:-0}"
CIG_RUN_ID=""
CIG_DEVICE="${CIG_DEVICE:-cpu}"
# A frozen oracle-only threshold artifact is mandatory for confirmatory H1.
# Generate it with scripts/calibrate_h1_oracle_thresholds.py on development
# oracle states that are disjoint from the confirmatory seed set.
CIG_H1_THRESHOLD_CALIBRATION="${CIG_H1_THRESHOLD_CALIBRATION:-}"
# H1 support and CUSUM calibration are independent development-only artifacts.
# Build H1 rows with collect_h1_oracle_support.py, certify them with
# validate_h1_oracle_support.py, then freeze thresholds with
# calibrate_h1_oracle_thresholds.py.  They are required before any
# confirmatory estimator/tracking claim.
CIG_H1_ORACLE_SUPPORT="${CIG_H1_ORACLE_SUPPORT:-}"
CIG_CUSUM_CALIBRATION="${CIG_CUSUM_CALIBRATION:-}"
# Optional full-profile second-benchmark artifact for G8.
CIG_EXTERNAL_GATE_ROOT="${CIG_EXTERNAL_GATE_ROOT:-}"
# Paper-B scaling measures the candidate-restricted edge implementation. This
# is explicit in the run manifest; it never licenses an O(N) claim unless the
# adapter also reports subquadratic candidate construction.
CIG_PAPER_B_CANDIDATE_MAX_DEGREE="${CIG_PAPER_B_CANDIDATE_MAX_DEGREE:-8}"
CIG_PAPER_B_CANDIDATE_RECALL_STATES="${CIG_PAPER_B_CANDIDATE_RECALL_STATES:-}"
CIG_PAPER_B_CANDIDATE_RECALL_HORIZON="${CIG_PAPER_B_CANDIDATE_RECALL_HORIZON:-8}"
CIG_PAPER_B_CANDIDATE_RECALL_TRIALS="${CIG_PAPER_B_CANDIDATE_RECALL_TRIALS:-2}"
CIG_PAPER_B_CANDIDATE_RECALL_MIN="${CIG_PAPER_B_CANDIDATE_RECALL_MIN:-0.80}"
CIG_PAPER_B_CANDIDATE_RECALL_STABILITY_MIN="${CIG_PAPER_B_CANDIDATE_RECALL_STABILITY_MIN:-0.80}"
CIG_PAPER_B_CANDIDATE_RECALL_STABLE_FRACTION_MIN="${CIG_PAPER_B_CANDIDATE_RECALL_STABLE_FRACTION_MIN:-0.80}"
CIG_PAPER_B_MAX_REL_REWARD_DROP="${CIG_PAPER_B_MAX_REL_REWARD_DROP:-0.10}"
CIG_PAPER_B_MAX_REL_LOGIT_INCREASE="${CIG_PAPER_B_MAX_REL_LOGIT_INCREASE:-0.25}"
CIG_PAPER_B_MAX_REL_VALUE_INCREASE="${CIG_PAPER_B_MAX_REL_VALUE_INCREASE:-0.25}"
CIG_PAPER_B_MAX_ACTION_DROP="${CIG_PAPER_B_MAX_ACTION_DROP:-0.05}"
# If the caller does not provide development artifacts, confirmatory mode
# generates them automatically on source-hash-scoped, disjoint development seeds.
CIG_DEV_H1_SEED_TEXT="${CIG_DEV_H1_SEEDS:-901 902 903 904 905 906 907 908}"
CIG_DEV_CUSUM_SEED_TEXT="${CIG_DEV_CUSUM_SEEDS:-1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040}"
CIG_DEV_H1_STATES_PER_SEED="${CIG_DEV_H1_STATES_PER_SEED:-100}"
CIG_DEV_CUSUM_EPISODES="${CIG_DEV_CUSUM_EPISODES:-200}"
CIG_DEV_CUSUM_PRETRAIN_EPISODES="${CIG_DEV_CUSUM_PRETRAIN_EPISODES:-60}"

usage() {
  sed -n '2,18p' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --quick)
      CIG_QUICK=1
      shift
      ;;
    --claims-only)
      CIG_CLAIMS_ONLY=1
      shift
      ;;
    --with-legacy-h3)
      CIG_RUN_LEGACY_H3=1
      shift
      ;;
    --run-id)
      if [ "$#" -lt 2 ]; then
        echo "--run-id requires a value" >&2
        exit 2
      fi
      CIG_RUN_ID="$2"
      shift 2
      ;;
    --device)
      if [ "$#" -lt 2 ]; then
        echo "--device requires a value" >&2
        exit 2
      fi
      CIG_DEVICE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$CIG_RUN_ID" ]; then
  CIG_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
fi
case "$CIG_RUN_ID" in
  *[!A-Za-z0-9._-]*|"")
    echo "Invalid run id: use only letters, digits, dot, underscore, or hyphen." >&2
    exit 2
    ;;
esac

if [ "$CIG_QUICK" -eq 1 ]; then
  CIG_PAPER_B_CANDIDATE_RECALL_STATES="${CIG_PAPER_B_CANDIDATE_RECALL_STATES:-1}"
  CIG_DEFAULT_H1_SEEDS="0"
  CIG_DEFAULT_H23_SEEDS="0"
  CIG_DEFAULT_GATE_SEEDS="0"
  CIG_H2_EPISODES="${CIG_H2_EPISODES:-60}"
  CIG_H3_EPISODES="${CIG_H3_EPISODES:-60}"
  CIG_LATENCY_TRAIN_EPISODES="${CIG_LATENCY_TRAIN_EPISODES:-2}"
  CIG_PAPER_B_SELECTOR_STATES="${CIG_PAPER_B_SELECTOR_STATES:-2}"
  CIG_PAPER_B_CORE_BUDGETS="${CIG_PAPER_B_CORE_BUDGETS:-2}"
  CIG_MODE="quick"
else
  CIG_PAPER_B_CANDIDATE_RECALL_STATES="${CIG_PAPER_B_CANDIDATE_RECALL_STATES:-4}"
  # Untouched confirmatory seeds.  Development seeds remain opt-in through
  # CIG_H1_SEEDS/CIG_RUN_SEEDS and are recorded as non-confirmatory metadata.
  CIG_DEFAULT_H1_SEEDS="101 102 103 104 105 106 107 108"
  CIG_DEFAULT_H23_SEEDS="201 202 203 204 205 206 207 208"
  # G0-G4 are preregistered feasibility gates, not a preview of the
  # confirmatory H1/H2 seeds. Keep their seed set disjoint.
  CIG_DEFAULT_GATE_SEEDS="301 302 303 304 305"
  CIG_H2_EPISODES="${CIG_H2_EPISODES:-400}"
  CIG_H3_EPISODES="${CIG_H3_EPISODES:-200}"
  CIG_LATENCY_TRAIN_EPISODES="${CIG_LATENCY_TRAIN_EPISODES:-200}"
  CIG_PAPER_B_SELECTOR_STATES="${CIG_PAPER_B_SELECTOR_STATES:-8}"
  CIG_PAPER_B_CORE_BUDGETS="${CIG_PAPER_B_CORE_BUDGETS:-2 3 4 5}"
  CIG_MODE="confirmatory"
fi

CIG_H23_SEED_TEXT="${CIG_RUN_SEEDS:-$CIG_DEFAULT_H23_SEEDS}"
CIG_H1_SEED_TEXT="${CIG_H1_SEEDS:-${CIG_RUN_SEEDS:-$CIG_DEFAULT_H1_SEEDS}}"
CIG_GATE_SEED_TEXT="${CIG_GATE_SEEDS:-$CIG_DEFAULT_GATE_SEEDS}"
CIG_LATENCY_ORACLE_STATES="${CIG_LATENCY_ORACLE_STATES:-12}"
CIG_LATENCY_ORACLE_TRIALS="${CIG_LATENCY_ORACLE_TRIALS:-2}"
CIG_H2_PRETRAIN_EPISODES="${CIG_H2_PRETRAIN_EPISODES:-60}"
CIG_PAPER_B_SCALING_AGENTS="${CIG_PAPER_B_SCALING_AGENTS:-12 24 48}"
if [ "$CIG_QUICK" -eq 1 ]; then
  CIG_GATE_N_AGENTS="${CIG_GATE_N_AGENTS:-6}"
  CIG_GATE_ORACLE_STATES="${CIG_GATE_ORACLE_STATES:-1}"
  CIG_GATE_H1_TRAIN_EPISODES="${CIG_GATE_H1_TRAIN_EPISODES:-2}"
  CIG_GATE_H1_STATES="${CIG_GATE_H1_STATES:-2}"
  CIG_GATE_G3_STATES="${CIG_GATE_G3_STATES:-1}"
  CIG_GATE_ALLOCATION_EPISODES="${CIG_GATE_ALLOCATION_EPISODES:-2}"
  CIG_GATE_ALLOCATION_MAX_STEPS="${CIG_GATE_ALLOCATION_MAX_STEPS:-8}"
  CIG_GATE_ALLOCATION_FINAL_WINDOW="${CIG_GATE_ALLOCATION_FINAL_WINDOW:-2}"
else
  CIG_GATE_N_AGENTS="${CIG_GATE_N_AGENTS:-24}"
  CIG_GATE_ORACLE_STATES="${CIG_GATE_ORACLE_STATES:-4}"
  CIG_GATE_H1_TRAIN_EPISODES="${CIG_GATE_H1_TRAIN_EPISODES:-30}"
  CIG_GATE_H1_STATES="${CIG_GATE_H1_STATES:-16}"
  CIG_GATE_G3_STATES="${CIG_GATE_G3_STATES:-3}"
  CIG_GATE_ALLOCATION_EPISODES="${CIG_GATE_ALLOCATION_EPISODES:-40}"
  CIG_GATE_ALLOCATION_MAX_STEPS="${CIG_GATE_ALLOCATION_MAX_STEPS:-30}"
  CIG_GATE_ALLOCATION_FINAL_WINDOW="${CIG_GATE_ALLOCATION_FINAL_WINDOW:-10}"
fi
read -r -a CIG_H1_SEED_ARRAY <<< "$CIG_H1_SEED_TEXT"
read -r -a CIG_H23_SEED_ARRAY <<< "$CIG_H23_SEED_TEXT"
read -r -a CIG_GATE_SEED_ARRAY <<< "$CIG_GATE_SEED_TEXT"
read -r -a CIG_DEV_H1_SEED_ARRAY <<< "$CIG_DEV_H1_SEED_TEXT"
read -r -a CIG_DEV_CUSUM_SEED_ARRAY <<< "$CIG_DEV_CUSUM_SEED_TEXT"
if [ "${#CIG_H1_SEED_ARRAY[@]}" -eq 0 ] || [ "${#CIG_H23_SEED_ARRAY[@]}" -eq 0 ] || [ "${#CIG_GATE_SEED_ARRAY[@]}" -eq 0 ]; then
  echo "CIG_RUN_SEEDS resolved to an empty seed list." >&2
  exit 2
fi

case "$CIG_RUN_LEGACY_H3" in
  0|1) ;;
  *)
    echo "CIG_RUN_LEGACY_H3 must be 0 or 1." >&2
    exit 2
    ;;
esac

CIG_SOURCE_TREE_HASH="$($CIG_PYTHON scripts/source_tree_hash.py)"
if [ -z "$CIG_SOURCE_TREE_HASH" ]; then
  echo "Could not compute confirmatory source-tree hash." >&2
  exit 2
fi
if [ "$CIG_MODE" = "confirmatory" ] && [ "${CIG_ALLOW_DIRTY_CONFIRMATORY:-0}" != "1" ]; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse --verify HEAD >/dev/null 2>&1 || {
      echo "Git worktree exists but has no valid HEAD." >&2; exit 2;
    }
    if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
      echo "Confirmatory mode requires a clean Git worktree when Git metadata is present." >&2
      exit 2
    fi
  else
    echo "INFO: no Git metadata found; using frozen content-addressed source SHA-256 $CIG_SOURCE_TREE_HASH"
  fi
fi
for CIG_SEED in "${CIG_H1_SEED_ARRAY[@]}" "${CIG_H23_SEED_ARRAY[@]}" "${CIG_GATE_SEED_ARRAY[@]}"; do
  case "$CIG_SEED" in
    *[!0-9]*|"")
      echo "Invalid seed '$CIG_SEED'; seeds must be non-negative integers." >&2
      exit 2
      ;;
  esac
done

CIG_SEEN_SEEDS=" "
for CIG_SEED in "${CIG_H1_SEED_ARRAY[@]}"; do
  case "$CIG_SEEN_SEEDS" in
    *" $CIG_SEED "*)
      echo "Duplicate H1 seed '$CIG_SEED' is pseudo-replication." >&2
      exit 2
      ;;
  esac
  CIG_SEEN_SEEDS="$CIG_SEEN_SEEDS$CIG_SEED "
done
CIG_SEEN_SEEDS=" "
for CIG_SEED in "${CIG_H23_SEED_ARRAY[@]}"; do
  case "$CIG_SEEN_SEEDS" in
    *" $CIG_SEED "*)
      echo "Duplicate H2/H3 seed '$CIG_SEED' is pseudo-replication." >&2
      exit 2
      ;;
  esac
  CIG_SEEN_SEEDS="$CIG_SEEN_SEEDS$CIG_SEED "
done

CIG_SEEN_GATE_SEEDS=" "
for CIG_SEED in "${CIG_GATE_SEED_ARRAY[@]}"; do
  case "$CIG_SEEN_GATE_SEEDS" in
    *" $CIG_SEED "*)
      echo "Duplicate scientific-precheck seed '$CIG_SEED' is pseudo-replication." >&2
      exit 2
      ;;
  esac
  CIG_SEEN_GATE_SEEDS="$CIG_SEEN_GATE_SEEDS$CIG_SEED "
  case " ${CIG_H1_SEED_ARRAY[*]} ${CIG_H23_SEED_ARRAY[*]} " in
    *" $CIG_SEED "*)
      if [ "$CIG_MODE" = "confirmatory" ]; then
        echo "Scientific-precheck seed '$CIG_SEED' overlaps a confirmatory H1/H2 seed." >&2
        exit 2
      fi
      ;;
  esac
done

if [ "$CIG_MODE" = "confirmatory" ]; then
  # Development seeds matter only for artifacts that will actually be generated
  # in this invocation.  Pre-supplied frozen artifacts must not be rejected
  # merely because the caller did not also provide unused development seeds.
  CIG_NEED_AUTO_H1_DEV=0
  CIG_NEED_AUTO_CUSUM_DEV=0
  if [ -z "$CIG_H1_THRESHOLD_CALIBRATION" ] && [ -z "$CIG_H1_ORACLE_SUPPORT" ]; then
    CIG_NEED_AUTO_H1_DEV=1
  fi
  if [ -z "$CIG_CUSUM_CALIBRATION" ]; then
    CIG_NEED_AUTO_CUSUM_DEV=1
  fi
  CIG_SEEN_DEV_SEEDS=" "
  if [ "$CIG_NEED_AUTO_H1_DEV" -eq 1 ]; then
    if [ "${#CIG_DEV_H1_SEED_ARRAY[@]}" -lt 5 ]; then
      echo "Automatic H1 development calibration requires at least 5 disjoint seeds." >&2; exit 2
    fi
    for CIG_DEV_SEED in "${CIG_DEV_H1_SEED_ARRAY[@]}"; do
      case "$CIG_DEV_SEED" in *[!0-9]*|"") echo "H1 development seeds must be non-negative integers." >&2; exit 2;; esac
      case "$CIG_SEEN_DEV_SEEDS" in *" $CIG_DEV_SEED "*) echo "Duplicate development seed $CIG_DEV_SEED is pseudo-replication." >&2; exit 2;; esac
      case " ${CIG_H1_SEED_ARRAY[*]} ${CIG_H23_SEED_ARRAY[*]} ${CIG_GATE_SEED_ARRAY[*]} " in
        *" $CIG_DEV_SEED "*) echo "Development seed $CIG_DEV_SEED overlaps confirmatory/precheck seeds." >&2; exit 2;;
      esac
      CIG_SEEN_DEV_SEEDS="$CIG_SEEN_DEV_SEEDS$CIG_DEV_SEED "
    done
  fi
  if [ "$CIG_NEED_AUTO_CUSUM_DEV" -eq 1 ]; then
    if [ "${#CIG_DEV_CUSUM_SEED_ARRAY[@]}" -lt 40 ]; then
      echo "Automatic CUSUM calibration requires at least 40 independent development trajectories/seeds." >&2; exit 2
    fi
    for CIG_DEV_SEED in "${CIG_DEV_CUSUM_SEED_ARRAY[@]}"; do
      case "$CIG_DEV_SEED" in *[!0-9]*|"") echo "CUSUM development seeds must be non-negative integers." >&2; exit 2;; esac
      case "$CIG_SEEN_DEV_SEEDS" in *" $CIG_DEV_SEED "*) echo "Duplicate/reused development seed $CIG_DEV_SEED is pseudo-replication." >&2; exit 2;; esac
      case " ${CIG_H1_SEED_ARRAY[*]} ${CIG_H23_SEED_ARRAY[*]} ${CIG_GATE_SEED_ARRAY[*]} " in
        *" $CIG_DEV_SEED "*) echo "Development seed $CIG_DEV_SEED overlaps confirmatory/precheck seeds." >&2; exit 2;;
      esac
      CIG_SEEN_DEV_SEEDS="$CIG_SEEN_DEV_SEEDS$CIG_DEV_SEED "
    done
  fi
fi

for CIG_EPISODE_BUDGET in "$CIG_H2_EPISODES" "$CIG_H2_PRETRAIN_EPISODES" "$CIG_H3_EPISODES" \
  "$CIG_LATENCY_ORACLE_STATES" "$CIG_LATENCY_ORACLE_TRIALS" \
  "$CIG_LATENCY_TRAIN_EPISODES" "$CIG_GATE_ALLOCATION_EPISODES" \
  "$CIG_GATE_ALLOCATION_MAX_STEPS" "$CIG_GATE_ALLOCATION_FINAL_WINDOW"; do
  case "$CIG_EPISODE_BUDGET" in
    *[!0-9]*|""|0)
      echo "Episode budgets must be positive integers." >&2
      exit 2
      ;;
  esac
done
case "$CIG_PAPER_B_SELECTOR_STATES" in
  *[!0-9]*|""|0)
    echo "CIG_PAPER_B_SELECTOR_STATES must be a positive integer." >&2
    exit 2
    ;;
esac
case "$CIG_PAPER_B_CANDIDATE_MAX_DEGREE" in
  *[!0-9]*|""|0)
    echo "CIG_PAPER_B_CANDIDATE_MAX_DEGREE must be a positive integer." >&2
    exit 2
    ;;
esac
for CIG_CANDIDATE_RECALL_BUDGET in \
  "$CIG_PAPER_B_CANDIDATE_RECALL_STATES" \
  "$CIG_PAPER_B_CANDIDATE_RECALL_HORIZON" \
  "$CIG_PAPER_B_CANDIDATE_RECALL_TRIALS"; do
  case "$CIG_CANDIDATE_RECALL_BUDGET" in
    *[!0-9]*|""|0)
      echo "Paper-B candidate-recall budgets must be positive integers." >&2
      exit 2
      ;;
  esac
done
case "$CIG_PAPER_B_CANDIDATE_RECALL_MIN" in
  0|0.*|1|1.0|1.00|1.000) ;;
  *)
    echo "CIG_PAPER_B_CANDIDATE_RECALL_MIN must lie in [0,1]." >&2
    exit 2
    ;;
esac
for CIG_AGENT_COUNT in $CIG_PAPER_B_SCALING_AGENTS; do
  case "$CIG_AGENT_COUNT" in
    *[!0-9]*|""|0|1)
      echo "CIG_PAPER_B_SCALING_AGENTS must contain integers greater than one." >&2
      exit 2
      ;;
  esac
done
CIG_SEEN_CORE_BUDGETS=" "
for CIG_CORE_BUDGET in $CIG_PAPER_B_CORE_BUDGETS; do
  case "$CIG_CORE_BUDGET" in
    *[!0-9]*|""|0)
      echo "CIG_PAPER_B_CORE_BUDGETS must contain positive integers." >&2
      exit 2
      ;;
  esac
  case "$CIG_SEEN_CORE_BUDGETS" in
    *" $CIG_CORE_BUDGET "*)
      echo "CIG_PAPER_B_CORE_BUDGETS contains duplicate budget $CIG_CORE_BUDGET." >&2
      exit 2
      ;;
  esac
  CIG_SEEN_CORE_BUDGETS="$CIG_SEEN_CORE_BUDGETS$CIG_CORE_BUDGET "
done

if [ "$CIG_QUICK" -eq 0 ]; then
  # Supply both H1 development artifacts explicitly, or neither.  When neither
  # is supplied, build a source-hash-scoped development calibration exactly once.
  if { [ -n "$CIG_H1_THRESHOLD_CALIBRATION" ] && [ -z "$CIG_H1_ORACLE_SUPPORT" ]; } || \
     { [ -z "$CIG_H1_THRESHOLD_CALIBRATION" ] && [ -n "$CIG_H1_ORACLE_SUPPORT" ]; }; then
    echo "Provide both CIG_H1_THRESHOLD_CALIBRATION and CIG_H1_ORACLE_SUPPORT, or neither." >&2
    exit 2
  fi
  CIG_DEV_DIR="${CIG_DEV_DIR:-$CIG_ROOT/results/development_calibration/$CIG_SOURCE_TREE_HASH}"
  mkdir -p "$CIG_DEV_DIR"
  if [ -z "$CIG_H1_THRESHOLD_CALIBRATION" ]; then
    CIG_H1_DEV_ROOT="$CIG_DEV_DIR/h1"
    CIG_H1_PAIR_ROWS="$CIG_H1_DEV_ROOT/tiny_oracle_pair_rows.csv"
    CIG_H1_THRESHOLD_CALIBRATION="$CIG_DEV_DIR/h1_thresholds.json"
    CIG_H1_ORACLE_SUPPORT="$CIG_DEV_DIR/h1_oracle_support_validated.json"
    if [ ! -s "$CIG_H1_THRESHOLD_CALIBRATION" ] || [ ! -s "$CIG_H1_ORACLE_SUPPORT" ]; then
      echo "=== automatic development calibration: H1 oracle support/thresholds ==="
      rm -rf "$CIG_H1_DEV_ROOT"
      "$CIG_PYTHON" scripts/collect_h1_oracle_support.py \
        --seeds "${CIG_DEV_H1_SEED_ARRAY[@]}" \
        --states-per-seed "$CIG_DEV_H1_STATES_PER_SEED" \
        --h1-eval-uniform-mass 0.10 --out-root "$CIG_H1_DEV_ROOT" || exit 2
      "$CIG_PYTHON" scripts/calibrate_h1_oracle_thresholds.py \
        --oracle-pair-rows "$CIG_H1_PAIR_ROWS" \
        --capacity-min-effect 0.01 --direction-min-effect 0.005 \
        --h1-target-policy-mode scripted_uniform_mixture \
        --h1-eval-uniform-mass 0.10 --out "$CIG_H1_THRESHOLD_CALIBRATION" || exit 2
      "$CIG_PYTHON" scripts/validate_h1_oracle_support.py \
        --oracle-pair-rows "$CIG_H1_PAIR_ROWS" \
        --capacity-threshold 0.01 --direction-threshold 0.005 \
        --h1-target-policy-mode scripted_uniform_mixture \
        --h1-eval-uniform-mass 0.10 --out "$CIG_H1_ORACLE_SUPPORT" || exit 2
    fi
  fi
  if [ ! -f "$CIG_H1_THRESHOLD_CALIBRATION" ] || [ ! -f "$CIG_H1_ORACLE_SUPPORT" ]; then
    echo "H1 development calibration artifacts are missing." >&2; exit 2
  fi
  if [ -z "$CIG_CUSUM_CALIBRATION" ]; then
    CIG_CUSUM_NULL="$CIG_DEV_DIR/cusum_no_change.json"
    CIG_CUSUM_CALIBRATION="$CIG_DEV_DIR/cusum_calibration.json"
    if [ ! -s "$CIG_CUSUM_CALIBRATION" ]; then
      echo "=== automatic development calibration: CUSUM no-change threshold ==="
      "$CIG_PYTHON" scripts/collect_cusum_no_change.py \
        --seeds "${CIG_DEV_CUSUM_SEED_ARRAY[@]}" \
        --episodes "$CIG_DEV_CUSUM_EPISODES" \
        --pretrain-episodes "$CIG_DEV_CUSUM_PRETRAIN_EPISODES" \
        --max-steps 30 --eval-every 1 --device "$CIG_DEVICE" \
        --out "$CIG_CUSUM_NULL" || exit 2
      "$CIG_PYTHON" scripts/calibrate_cusum_threshold.py \
        --no-change-z-json "$CIG_CUSUM_NULL" --allowance 0.5 \
        --false-alarm-target 0.05 --min-trajectories 40 \
        --development-seeds "${CIG_DEV_CUSUM_SEED_ARRAY[@]}" \
        --out "$CIG_CUSUM_CALIBRATION" || exit 2
    fi
  fi
  if [ ! -f "$CIG_CUSUM_CALIBRATION" ]; then
    echo "CUSUM calibration artifact is missing." >&2; exit 2
  fi
  if [ "${#CIG_H1_SEED_ARRAY[@]}" -lt 8 ]; then
    echo "Confirmatory H1 requires at least 8 unique paired seeds." >&2
    exit 2
  fi
  if [ "${#CIG_H23_SEED_ARRAY[@]}" -lt 8 ]; then
    echo "Confirmatory H2/H3 require at least 8 unique paired seeds." >&2
    exit 2
  fi
  if [ "${#CIG_GATE_SEED_ARRAY[@]}" -lt 3 ]; then
    echo "Confirmatory G0-G4 prechecks require at least 3 disjoint gate seeds." >&2
    exit 2
  fi
  if [ "$CIG_H2_EPISODES" -lt 400 ] || [ "$CIG_H3_EPISODES" -lt 200 ]; then
    echo "Confirmatory budgets require H2>=400 and H3>=200 episodes." >&2
    echo "Use --quick for reduced protocol-path checks." >&2
    exit 2
  fi
  if [ "$CIG_LATENCY_TRAIN_EPISODES" -lt 200 ]; then
    echo "Confirmatory latency calibration requires at least 200 training episodes." >&2
    exit 2
  fi
fi

CIG_RUN_DIR="$CIG_ROOT/results/runs/$CIG_RUN_ID"
CIG_LOG_DIR="$CIG_RUN_DIR/logs"
if [ -e "$CIG_RUN_DIR" ]; then
  echo "Run directory already exists: $CIG_RUN_DIR" >&2
  echo "Choose a new --run-id. Existing artifacts are never reused." >&2
  exit 2
fi
mkdir -p "$CIG_LOG_DIR"

CIG_STATUS_TSV="$CIG_RUN_DIR/command_status.tsv"
printf 'category\tlabel\texit_code\telapsed_seconds\tlog\n' > "$CIG_STATUS_TSV"

{
  printf 'run_id=%s\n' "$CIG_RUN_ID"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'mode=%s\n' "$CIG_MODE"
  printf 'python=%s\n' "$CIG_PYTHON"
  printf 'device=%s\n' "$CIG_DEVICE"
  printf 'source_tree_sha256=%s\n' "$CIG_SOURCE_TREE_HASH"
  printf 'profile=%s\n' "$([ "$CIG_CLAIMS_ONLY" -eq 1 ] && printf claims-only || printf complete)"
  printf 'legacy_h3=%s\n' "$CIG_RUN_LEGACY_H3"
  printf 'h1_seeds=%s\n' "${CIG_H1_SEED_ARRAY[*]}"
  printf 'h2_h3_seeds=%s\n' "${CIG_H23_SEED_ARRAY[*]}"
  printf 'scientific_precheck_seeds=%s\n' "${CIG_GATE_SEED_ARRAY[*]}"
  printf 'h2_episodes=%s\n' "$CIG_H2_EPISODES"
  printf 'h2_pretrain_episodes=%s\n' "$CIG_H2_PRETRAIN_EPISODES"
  printf 'h3_episodes=%s\n' "$CIG_H3_EPISODES"
  printf 'latency_oracle_states=%s\n' "$CIG_LATENCY_ORACLE_STATES"
  printf 'latency_oracle_trials=%s\n' "$CIG_LATENCY_ORACLE_TRIALS"
  printf 'latency_train_episodes=%s\n' "$CIG_LATENCY_TRAIN_EPISODES"
  printf 'paper_b_selector_states=%s\n' "$CIG_PAPER_B_SELECTOR_STATES"
  printf 'paper_b_scaling_agents=%s\n' "$CIG_PAPER_B_SCALING_AGENTS"
  printf 'paper_b_candidate_max_degree=%s\n' "$CIG_PAPER_B_CANDIDATE_MAX_DEGREE"
  printf 'paper_b_candidate_recall_states=%s\n' "$CIG_PAPER_B_CANDIDATE_RECALL_STATES"
  printf 'paper_b_candidate_recall_horizon=%s\n' "$CIG_PAPER_B_CANDIDATE_RECALL_HORIZON"
  printf 'paper_b_candidate_recall_trials=%s\n' "$CIG_PAPER_B_CANDIDATE_RECALL_TRIALS"
  printf 'paper_b_candidate_recall_min=%s\n' "$CIG_PAPER_B_CANDIDATE_RECALL_MIN"
  printf 'paper_b_candidate_recall_stability_min=%s\n' "$CIG_PAPER_B_CANDIDATE_RECALL_STABILITY_MIN"
  printf 'paper_b_candidate_recall_stable_fraction_min=%s\n' "$CIG_PAPER_B_CANDIDATE_RECALL_STABLE_FRACTION_MIN"
  printf 'paper_b_core_budgets=%s\n' "$CIG_PAPER_B_CORE_BUDGETS"
  printf 'h1_threshold_calibration=%s\n' "$CIG_H1_THRESHOLD_CALIBRATION"
  printf 'h1_oracle_support=%s\n' "$CIG_H1_ORACLE_SUPPORT"
  printf 'cusum_calibration=%s\n' "$CIG_CUSUM_CALIBRATION"
  printf 'external_gate_root=%s\n' "$CIG_EXTERNAL_GATE_ROOT"
  printf 'gate_n_agents=%s\n' "$CIG_GATE_N_AGENTS"
  printf 'gate_oracle_states=%s\n' "$CIG_GATE_ORACLE_STATES"
  printf 'gate_h1_train_episodes=%s\n' "$CIG_GATE_H1_TRAIN_EPISODES"
  printf 'gate_h1_states=%s\n' "$CIG_GATE_H1_STATES"
  printf 'gate_g3_states=%s\n' "$CIG_GATE_G3_STATES"
  printf 'gate_allocation_episodes=%s\n' "$CIG_GATE_ALLOCATION_EPISODES"
  printf 'gate_allocation_max_steps=%s\n' "$CIG_GATE_ALLOCATION_MAX_STEPS"
  printf 'gate_allocation_final_window=%s\n' "$CIG_GATE_ALLOCATION_FINAL_WINDOW"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  git status --porcelain --untracked-files=normal 2>/dev/null > "$CIG_RUN_DIR/git_status_porcelain.txt"
  if [ ! -s "$CIG_RUN_DIR/git_status_porcelain.txt" ]; then
    printf 'git_worktree=clean\n'
  else
    printf 'git_worktree=dirty\n'
  fi
} > "$CIG_RUN_DIR/run_metadata.txt"

CIG_OPERATIONAL_FAILURES=()
CIG_CLAIM_FAILURES=()
CIG_SMOKE_ONLY=0
CIG_PASSES=0
CIG_STARTED_AT=$SECONDS

run_logged() {
  local category="$1"
  local label="$2"
  local log_name="$3"
  shift 3
  local command_started=$SECONDS

  echo
  echo "======================================================================"
  echo "RUN: $label"
  echo "CATEGORY: $category"
  echo "LOG: $CIG_LOG_DIR/$log_name"
  echo "CMD: $*"
  echo "======================================================================"

  "$@" 2>&1 | tee "$CIG_LOG_DIR/$log_name"
  local command_status=${PIPESTATUS[0]}
  local command_elapsed=$((SECONDS - command_started))
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$category" "$label" "$command_status" "$command_elapsed" \
    "logs/$log_name" >> "$CIG_STATUS_TSV"

  if [ "$command_status" -eq 0 ]; then
    CIG_PASSES=$((CIG_PASSES + 1))
    echo "EXECUTION PASS: $label"
  else
    CIG_OPERATIONAL_FAILURES+=("$label (exit $command_status)")
    echo "FAIL: $label (exit $command_status)" >&2
  fi
  return 0
}

run_paper_validation() {
  local label="$1"
  local log_name="$2"
  shift 2
  local command_started=$SECONDS
  "$@" 2>&1 | tee "$CIG_LOG_DIR/$log_name"
  local command_status=${PIPESTATUS[0]}
  local command_elapsed=$((SECONDS - command_started))
  printf 'claim\t%s\t%s\t%s\t%s\n' \
    "$label" "$command_status" "$command_elapsed" "logs/$log_name" \
    >> "$CIG_STATUS_TSV"
  case "$command_status" in
    0)
      CIG_PASSES=$((CIG_PASSES + 1))
      ;;
    10)
      CIG_CLAIM_FAILURES+=("$label was NOT_SUPPORTED")
      ;;
    11)
      CIG_PASSES=$((CIG_PASSES + 1))
      CIG_SMOKE_ONLY=1
      ;;
    *)
      CIG_OPERATIONAL_FAILURES+=("$label invalid (exit $command_status)")
      ;;
  esac
}

finish_metadata() {
  local final_state="$1"
  {
    printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'elapsed_seconds=%s\n' "$((SECONDS - CIG_STARTED_AT))"
    printf 'state=%s\n' "$final_state"
    printf 'passed_commands=%s\n' "$CIG_PASSES"
    printf 'operational_failures=%s\n' "${#CIG_OPERATIONAL_FAILURES[@]}"
    printf 'claim_failures=%s\n' "${#CIG_CLAIM_FAILURES[@]}"
  } > "$CIG_RUN_DIR/run_finished.txt"
}

echo "Run root: $CIG_RUN_DIR"
echo "Python: $CIG_PYTHON"
echo "Mode: $CIG_MODE"
echo "Profile: $([ "$CIG_CLAIMS_ONLY" -eq 1 ] && printf claims-only || printf complete)"
echo "Device: $CIG_DEVICE"
echo "H1 seeds: ${CIG_H1_SEED_ARRAY[*]}"
echo "H2/H3 seeds: ${CIG_H23_SEED_ARRAY[*]}"
echo "G0-G4 precheck seeds: ${CIG_GATE_SEED_ARRAY[*]}"
CIG_H1_VARIANT_COUNT=$("$CIG_PYTHON" -c 'from scripts.run_h1_calibration import VARIANTS; print(len(VARIANTS))')
CIG_H2_MODEL_COUNT=$("$CIG_PYTHON" -c 'from scripts.run_h2_selectivity import MODELS; print(len(MODELS))')
CIG_H3_VARIANT_COUNT=$("$CIG_PYTHON" -c 'from scripts.run_h3_slots import VARIANTS; print(len(VARIANTS))' 2>/dev/null || printf 6)
CIG_PB_ALLOC_VARIANT_COUNT=$("$CIG_PYTHON" -c 'from scripts.run_paper_b_allocation import VARIANTS; print(len(VARIANTS))')
CIG_PB_PAIR_VARIANT_COUNT=$("$CIG_PYTHON" -c 'from scripts.run_paper_b_pair_latent import VARIANTS; print(len(VARIANTS))')
CIG_PB_PERIPH_VARIANT_COUNT=$("$CIG_PYTHON" -c 'from scripts.run_paper_b_periphery import VARIANTS; print(len(VARIANTS))')
echo "Planned H1 attempts: $((CIG_H1_VARIANT_COUNT * ${#CIG_H1_SEED_ARRAY[@]}))"
echo "Planned H2 episodes: $((CIG_H2_MODEL_COUNT * ${#CIG_H23_SEED_ARRAY[@]} * (4 * CIG_H2_EPISODES + CIG_H2_PRETRAIN_EPISODES)))"
if [ "$CIG_RUN_LEGACY_H3" -eq 1 ]; then
  echo "Planned legacy H3 episodes: $((CIG_H3_VARIANT_COUNT * ${#CIG_H23_SEED_ARRAY[@]} * CIG_H3_EPISODES))"
else
  echo "Planned legacy H3 episodes: 0 (diagnostic disabled)"
fi
echo "Planned Paper-B allocation episodes: $((CIG_PB_ALLOC_VARIANT_COUNT * ${#CIG_H23_SEED_ARRAY[@]} * CIG_H3_EPISODES + ${#CIG_H23_SEED_ARRAY[@]} * CIG_H2_PRETRAIN_EPISODES))"
echo "Planned Paper-B adaptive-budget episodes: $((6 * ${#CIG_H23_SEED_ARRAY[@]} * CIG_H3_EPISODES))"
echo "Planned Paper-B pair-latent episodes: $((CIG_PB_PAIR_VARIANT_COUNT * ${#CIG_H23_SEED_ARRAY[@]} * CIG_H3_EPISODES))"
echo "Planned Paper-B periphery episodes: $((CIG_PB_PERIPH_VARIANT_COUNT * ${#CIG_H23_SEED_ARRAY[@]} * CIG_H3_EPISODES))"
echo "Latency oracle: ${CIG_LATENCY_ORACLE_STATES} states x ${CIG_LATENCY_ORACLE_TRIALS} CRN trials"
echo "Learned latency training after oracle pass: ${CIG_LATENCY_TRAIN_EPISODES} episodes"

# ---------------------------------------------------------------------------
# 0. Preflight: failures here invalidate the code path before long experiments.
# ---------------------------------------------------------------------------
echo "=== [0/9] Preflight and mechanism checks ==="
run_logged "preflight" "Python syntax compilation" "00_py_compile.log" \
  "$CIG_PYTHON" -m compileall -q \
  envs models runners scripts training utils \
  run_experiment.py run_main_exp.py run_oracle.py run_step_0.py run_step_1.py \
  smoke_test.py test_bc_loss_control.py
run_logged "preflight" "Mechanism smoke suite" "01_smoke_test.log" \
  "$CIG_PYTHON" smoke_test.py
run_logged "preflight" "Behaviour-cloning loss controls" "02_bc_loss_control.log" \
  "$CIG_PYTHON" test_bc_loss_control.py

if [ -d "$CIG_ROOT/tests" ]; then
  run_logged "preflight" "Protocol regression tests" "03_protocol_tests.log" \
    "$CIG_PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
fi

if [ "${#CIG_OPERATIONAL_FAILURES[@]}" -gt 0 ]; then
  echo
  echo "Preflight failed; long experiments were not started." >&2
  printf '  - %s\n' "${CIG_OPERATIONAL_FAILURES[@]}" >&2
  finish_metadata "INVALID_PREFLIGHT"
  echo "Artifacts: $CIG_RUN_DIR"
  exit 2
fi

# ---------------------------------------------------------------------------
# 1. Focused environment invariants and diagnostics.
# ---------------------------------------------------------------------------
echo "=== [1/9] Focused environment checks ==="
run_logged "diagnostic" "Frenet coordinate check (C1)" "10_env_test_c1.log" \
  "$CIG_PYTHON" envs/test_c1.py
run_logged "diagnostic" "P2/P4 boundary guards" "11_env_p2_p4_guards.log" \
  "$CIG_PYTHON" envs/test_p2_p4_guards.py
run_logged "diagnostic" "Tiny-oracle signed-Phi acceptance" "12_env_p0_oracle_check.log" \
  "$CIG_PYTHON" envs/_p0_oracle_check.py
run_logged "diagnostic" "Congestion-channel activation" "13_env_congestion.log" \
  "$CIG_PYTHON" envs/_diag_congestion.py
run_logged "diagnostic" "Phi/W-star horizon and unit check" "14_env_phi_wstar_horizon.log" \
  "$CIG_PYTHON" envs/code_test.py

# ---------------------------------------------------------------------------
# 2. Environment acceptance gates.
# ---------------------------------------------------------------------------
echo "=== [2/9] Environment acceptance gates ==="
run_logged "environment_gate" "All-flags environment audit" "20_env_audit.log" \
  "$CIG_PYTHON" envs/env_audit.py \
  --json-out "$CIG_RUN_DIR/env_audit.json"
run_logged "environment_gate" "Cumulative staged environment audit" "21_env_audit_staged.log" \
  "$CIG_PYTHON" envs/env_audit_staged.py \
  --json-out "$CIG_RUN_DIR/env_audit_staged.json"

if [ "${#CIG_OPERATIONAL_FAILURES[@]}" -gt 0 ]; then
  echo
  echo "Environment prerequisites failed; scientific experiments were not started." >&2
  printf '  - %s\n' "${CIG_OPERATIONAL_FAILURES[@]}" >&2
  finish_metadata "INVALID_ENVIRONMENT"
  echo "Artifacts: $CIG_RUN_DIR"
  exit 2
fi

# ---------------------------------------------------------------------------
# 2b. Retained latency falsification/recovery gate. The learned direct-lag
# calibration runs only after the all-action oracle establishes an identifiable
# delayed mechanism. A failure remains a reported negative H3 result; it does
# not remove latency from the method after observing the gate.
# ---------------------------------------------------------------------------
run_logged "diagnostic" "Lag-specific oracle latency gate" "22_latency_oracle.log" \
  "$CIG_PYTHON" scripts/run_latency_oracle.py \
  --states "$CIG_LATENCY_ORACLE_STATES" \
  --trials "$CIG_LATENCY_ORACLE_TRIALS" \
  --json-out "$CIG_RUN_DIR/latency_oracle.json"

if "$CIG_PYTHON" -c \
  'import json,sys; sys.exit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("gate_pass") else 1)' \
  "$CIG_RUN_DIR/latency_oracle.json"; then
  run_logged "diagnostic" "Learned direct-lag latency calibration" "23_latency_learned.log" \
    "$CIG_PYTHON" scripts/run_latency_calibration.py \
    --seeds "${CIG_H23_SEED_ARRAY[@]}" \
    --train-episodes "$CIG_LATENCY_TRAIN_EPISODES" \
    --states "$CIG_LATENCY_ORACLE_STATES" \
    --trials "$CIG_LATENCY_ORACLE_TRIALS" \
    --device "$CIG_DEVICE" \
    --json-out "$CIG_RUN_DIR/latency_calibration.json"
  if "$CIG_PYTHON" -c \
    'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if p.get("gate_pass") else 1)' \
    "$CIG_RUN_DIR/latency_calibration.json"; then
    echo "SCIENTIFIC GATE PASS: learned direct-lag latency"
  else
    echo "SCIENTIFIC GATE FAIL: learned direct-lag latency (execution succeeded; latency claim remains unsupported)."
  fi
else
  echo "SCIENTIFIC GATE FAIL: oracle latency mechanism was not established; retained latency-recovery claim is unsupported."
fi

# ---------------------------------------------------------------------------
# 3. Structure-value experiments.
# ---------------------------------------------------------------------------
echo "=== [3/9] Structure-value experiments ==="
run_logged "environment_gate" "True zero-cost oracle-core Experiment 0" "34_run_oracle.log" \
  "$CIG_PYTHON" run_oracle.py \
  --json-out "$CIG_RUN_DIR/experiment0_gate.json"

if [ "${#CIG_OPERATIONAL_FAILURES[@]}" -gt 0 ]; then
  echo
  echo "Experiment 0 prerequisite failed; H1/H2/H3 were not started." >&2
  printf '  - %s\n' "${CIG_OPERATIONAL_FAILURES[@]}" >&2
  finish_metadata "INVALID_EXPERIMENT0"
  echo "Artifacts: $CIG_RUN_DIR"
  exit 2
fi

CIG_PRECHECK_ARGS=(
  --experiment0 "$CIG_RUN_DIR/experiment0_gate.json"
  --seeds "${CIG_GATE_SEED_ARRAY[@]}"
  --protocol-mode "$CIG_MODE"
  --device "$CIG_DEVICE"
  --n-agents "$CIG_GATE_N_AGENTS"
  --oracle-states "$CIG_GATE_ORACLE_STATES"
  --h1-train-episodes "$CIG_GATE_H1_TRAIN_EPISODES"
  --h1-states "$CIG_GATE_H1_STATES"
  --g3-states "$CIG_GATE_G3_STATES"
  --allocation-episodes "$CIG_GATE_ALLOCATION_EPISODES"
  --allocation-max-steps "$CIG_GATE_ALLOCATION_MAX_STEPS"
  --allocation-final-window "$CIG_GATE_ALLOCATION_FINAL_WINDOW"
  --out "$CIG_RUN_DIR/scientific_prechecks_g0_g4.json"
  --work-root "$CIG_RUN_DIR/scientific_prechecks_work"
)
if [ "$CIG_QUICK" -eq 0 ]; then
  CIG_PRECHECK_ARGS+=(--threshold-calibration "$CIG_H1_THRESHOLD_CALIBRATION")
fi
run_logged "scientific_precheck" "G0-G4 oracle/forced-only scientific prechecks" "35_scientific_prechecks.log" \
  "$CIG_PYTHON" scripts/run_scientific_prechecks.py "${CIG_PRECHECK_ARGS[@]}"
if [ "${#CIG_OPERATIONAL_FAILURES[@]}" -gt 0 ]; then
  finish_metadata "INVALID_SCIENTIFIC_PRECHECK"
  exit 2
fi
if [ "$CIG_MODE" = "confirmatory" ] && ! "$CIG_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if p.get("core_prechecks_pass") else 1)' \
  "$CIG_RUN_DIR/scientific_prechecks_g0_g4.json"; then
  echo "SCIENTIFIC PRECHECK FAIL: at least one of G0-G4 failed; expensive learned panels were not started." >&2
  finish_metadata "COMPLETE_PRECHECK_NOT_SUPPORTED"
  exit 3
fi

if [ "$CIG_QUICK" -eq 0 ] && [ "$CIG_CLAIMS_ONLY" -eq 0 ]; then
  run_logged "structure_value" "Tier 0 oracle structure regret" "30_structure_value_tier0.log" \
    "$CIG_PYTHON" envs/structure_value_tier0.py
  run_logged "structure_value" "Tier 0 cumulative staged structure regret" "31_structure_value_tier0_staged.log" \
    "$CIG_PYTHON" envs/structure_value_tier0_staged.py
  run_logged "structure_value" "Tier 1 predictive value gap" "32_structure_value_tier1.log" \
    "$CIG_PYTHON" envs/structure_value_tier1.py
  run_logged "structure_value" "Tier 2 learned structure value" "33_structure_value_tier2.log" \
    "$CIG_PYTHON" envs/structure_value_tier2.py
else
  echo "SKIP: redundant long structure-value tiers in quick/claims-only profile."
fi

# ---------------------------------------------------------------------------
# 4. Stage entry points.
# ---------------------------------------------------------------------------
echo "=== [4/9] Step entry points ==="
if [ "$CIG_QUICK" -eq 0 ] && [ "$CIG_CLAIMS_ONLY" -eq 0 ]; then
  run_logged "stage" "Step 0 learned structure-sensitivity comparison" "40_run_step_0.log" \
    "$CIG_PYTHON" run_step_0.py
  run_logged "stage" "Step 1 learned-stage diagnostic" "41_run_step_1.log" \
    "$CIG_PYTHON" run_step_1.py
else
  echo "SKIP: hard-coded long step entry points in quick/claims-only profile."
fi

# ---------------------------------------------------------------------------
# 5. Main runner integration and legacy comparison.
# ---------------------------------------------------------------------------
echo "=== [5/9] Main runner integration ==="
run_logged "integration" "Main CLI smoke: behavioural task" "50_main_cli_behavioral_smoke.log" \
  "$CIG_PYTHON" run_experiment.py \
  --task behavioral \
  --models PureMeanField,Final-CIGAMF \
  --episodes 2 \
  --max_steps 12 \
  --eval_every 1 \
  --seed 0 \
  --device "$CIG_DEVICE" \
  --smoke \
  --result_dir "$CIG_RUN_DIR/integration_smoke"
run_logged "integration" "Main CLI smoke: tiny-oracle task" "51_main_cli_tiny_smoke.log" \
  "$CIG_PYTHON" run_experiment.py \
  --task tiny \
  --tiny_states 3 \
  --tiny_horizon 1 \
  --tiny_proxy_train_episodes 2 \
  --seed 0 \
  --device "$CIG_DEVICE" \
  --smoke \
  --result_dir "$CIG_RUN_DIR/integration_smoke"

if [ "$CIG_QUICK" -eq 0 ] && [ "$CIG_CLAIMS_ONLY" -eq 0 ]; then
  run_logged "legacy" "Legacy three-model comparison" "52_run_main_exp.log" \
    "$CIG_PYTHON" run_main_exp.py
fi

# ---------------------------------------------------------------------------
# 6. Paper hypotheses and faithful ablations.
# ---------------------------------------------------------------------------
echo "=== [6/9] H1/H2/H3 experiments ==="
CIG_H1_ARGS=(--quiet)
if [ "$CIG_QUICK" -eq 1 ]; then
  CIG_H1_ARGS+=(--allow-development-thresholds)
else
  CIG_H1_ARGS+=(--threshold-calibration "$CIG_H1_THRESHOLD_CALIBRATION" --oracle-support "$CIG_H1_ORACLE_SUPPORT")
fi
run_logged "hypothesis" "H1 one-step causal identification and calibration" "60_h1.log" \
  "$CIG_PYTHON" scripts/run_h1_calibration.py \
  --seeds "${CIG_H1_SEED_ARRAY[@]}" \
  --device "$CIG_DEVICE" \
  --out-root "$CIG_RUN_DIR/h1" \
  "${CIG_H1_ARGS[@]}"
run_logged "hypothesis" "Paper-A H2 factorial structural/behavioural selectivity" "61_h2.log" \
  "$CIG_PYTHON" scripts/run_h2_selectivity.py \
    --seeds "${CIG_H23_SEED_ARRAY[@]}" \
    --episodes "$CIG_H2_EPISODES" \
    --pretrain-episodes "$CIG_H2_PRETRAIN_EPISODES" \
    --cusum-calibration "$CIG_CUSUM_CALIBRATION" \
  --device "$CIG_DEVICE" \
  --out-root "$CIG_RUN_DIR/h2"
if [ "$CIG_RUN_LEGACY_H3" -eq 1 ]; then
  run_logged "ablation" "Legacy end-to-end slot-system ablations" "62_h3.log" \
    "$CIG_PYTHON" scripts/run_h3_slots.py \
    --seeds "${CIG_H23_SEED_ARRAY[@]}" \
    --episodes "$CIG_H3_EPISODES" \
    --device "$CIG_DEVICE" \
    --out-root "$CIG_RUN_DIR/h3"
else
  echo "SKIP: legacy H3 diagnostic is not a Paper-A/Paper-B claim gate."
fi
run_logged "allocation" "Paper-B selector isolation and end-to-end allocation" "63_paper_b_allocation.log" \
  "$CIG_PYTHON" scripts/run_paper_b_allocation.py \
  --seeds "${CIG_H23_SEED_ARRAY[@]}" \
  --episodes "$CIG_H3_EPISODES" \
  --pretrain-episodes "$CIG_H2_PRETRAIN_EPISODES" \
  --selector-states "$CIG_PAPER_B_SELECTOR_STATES" \
  --device "$CIG_DEVICE" \
  --out-root "$CIG_RUN_DIR/paper_b_allocation"
run_logged "allocation" "Paper-B entropy-adaptive budget at matched fixed cost" "64_paper_b_adaptive_budget.log" \
  "$CIG_PYTHON" scripts/run_paper_b_adaptive_budget.py \
  --seeds "${CIG_H23_SEED_ARRAY[@]}" \
  --episodes "$CIG_H3_EPISODES" \
  --k-min 2 --k-max 5 \
  --device "$CIG_DEVICE" \
  --out-root "$CIG_RUN_DIR/paper_b_adaptive_budget"
run_logged "representation" "Paper-B pair-latent ablations under fixed core" "65_paper_b_pair_latent.log" \
  "$CIG_PYTHON" scripts/run_paper_b_pair_latent.py \
  --seeds "${CIG_H23_SEED_ARRAY[@]}" \
  --episodes "$CIG_H3_EPISODES" \
  --device "$CIG_DEVICE" \
  --out-root "$CIG_RUN_DIR/paper_b_pair_latent"
run_logged "representation" "Paper-B peripheral encoders under fixed core" "66_paper_b_periphery.log" \
  "$CIG_PYTHON" scripts/run_paper_b_periphery.py \
  --seeds "${CIG_H23_SEED_ARRAY[@]}" \
  --episodes "$CIG_H3_EPISODES" \
  --device "$CIG_DEVICE" \
  --out-root "$CIG_RUN_DIR/paper_b_periphery"
run_logged "scalability" "Paper-B reward-compute-memory scaling" "67_paper_b_scaling.log" \
  "$CIG_PYTHON" scripts/run_paper_b_scaling.py \
  --seeds "${CIG_H23_SEED_ARRAY[@]}" \
  --agent-counts $CIG_PAPER_B_SCALING_AGENTS \
  --core-budgets $CIG_PAPER_B_CORE_BUDGETS \
  --candidate-max-degree "$CIG_PAPER_B_CANDIDATE_MAX_DEGREE" \
  --candidate-recall-states "$CIG_PAPER_B_CANDIDATE_RECALL_STATES" \
  --candidate-recall-horizon "$CIG_PAPER_B_CANDIDATE_RECALL_HORIZON" \
  --candidate-recall-trials "$CIG_PAPER_B_CANDIDATE_RECALL_TRIALS" \
  --candidate-recall-min "$CIG_PAPER_B_CANDIDATE_RECALL_MIN" \
  --candidate-recall-stability-min "$CIG_PAPER_B_CANDIDATE_RECALL_STABILITY_MIN" \
  --candidate-recall-stable-fraction-min "$CIG_PAPER_B_CANDIDATE_RECALL_STABLE_FRACTION_MIN" \
  --candidate-max-relative-reward-drop "$CIG_PAPER_B_MAX_REL_REWARD_DROP" \
  --candidate-max-relative-logit-error-increase "$CIG_PAPER_B_MAX_REL_LOGIT_INCREASE" \
  --candidate-max-relative-value-error-increase "$CIG_PAPER_B_MAX_REL_VALUE_INCREASE" \
  --candidate-max-action-agreement-drop "$CIG_PAPER_B_MAX_ACTION_DROP" \
  --episodes "$CIG_H3_EPISODES" \
  --device "$CIG_DEVICE" \
  --out-root "$CIG_RUN_DIR/paper_b_scaling"

# ---------------------------------------------------------------------------
# 7. Diagnostic figure.
# ---------------------------------------------------------------------------
echo "=== [7/9] Diagnostic figure ==="
if [ "$CIG_CLAIMS_ONLY" -eq 0 ]; then
  run_logged "diagnostic" "W-star timeline" "70_w_star_timeline.log" \
    "$CIG_PYTHON" scripts/plot_w_star_timeline.py \
    --steps 20 \
    --horizon 8 \
    --seed 123 \
    --tag "$CIG_RUN_ID" \
    --out-dir "$CIG_RUN_DIR/figures"
else
  echo "SKIP: W-star figure in claims-only profile."
fi

# ---------------------------------------------------------------------------
# 8. Run-scoped aggregation and scientific gates.
# ---------------------------------------------------------------------------
echo "=== [8/9] Aggregation and claim validation ==="
CIG_COLLECT_ARGS=(
  --run-root "$CIG_RUN_DIR"
  --expected-h1-seeds "${CIG_H1_SEED_ARRAY[@]}"
  --expected-h2-seeds "${CIG_H23_SEED_ARRAY[@]}"
)
if [ "$CIG_RUN_LEGACY_H3" -eq 1 ]; then
  CIG_COLLECT_ARGS+=(--expected-h3-seeds "${CIG_H23_SEED_ARRAY[@]}")
fi
run_logged "report" "Aggregate experiment results" "71_collect_results.log" \
  "$CIG_PYTHON" scripts/collect_results.py "${CIG_COLLECT_ARGS[@]}"
run_paper_validation "Validate Paper A" "72_validate_paper_a.log" \
  "$CIG_PYTHON" scripts/validate_paper_a.py \
  --run-root "$CIG_RUN_DIR" \
  --expected-h1-seeds "${CIG_H1_SEED_ARRAY[@]}" \
  --expected-h2-seeds "${CIG_H23_SEED_ARRAY[@]}" \
  --protocol-mode "$CIG_MODE"
run_paper_validation "Validate Paper B" "73_validate_paper_b.log" \
  "$CIG_PYTHON" scripts/validate_paper_b.py \
  --run-root "$CIG_RUN_DIR" \
  --expected-seeds "${CIG_H23_SEED_ARRAY[@]}" \
  --protocol-mode "$CIG_MODE"
CIG_GATE_VALIDATION_ARGS=(
  --run-root "$CIG_RUN_DIR"
  --prechecks "$CIG_RUN_DIR/scientific_prechecks_g0_g4.json"
  --protocol-mode "$CIG_MODE"
  --out "$CIG_RUN_DIR/scientific_gate_status.json"
)
if [ -n "$CIG_EXTERNAL_GATE_ROOT" ]; then
  CIG_GATE_VALIDATION_ARGS+=(--external-root "$CIG_EXTERNAL_GATE_ROOT")
fi
run_paper_validation "Validate scientific gate ladder G0-G9" "74_validate_scientific_gates.log" \
  "$CIG_PYTHON" scripts/validate_scientific_gates.py "${CIG_GATE_VALIDATION_ARGS[@]}"

# ---------------------------------------------------------------------------
# 9. Final manifest summary.
# ---------------------------------------------------------------------------
CIG_SOURCE_TREE_HASH_END="$($CIG_PYTHON scripts/source_tree_hash.py)"
if [ "$CIG_SOURCE_TREE_HASH_END" != "$CIG_SOURCE_TREE_HASH" ]; then
  CIG_OPERATIONAL_FAILURES+=("source tree mutated during run")
  echo "FAIL: source tree SHA-256 changed during the run." >&2
fi
printf 'source_tree_sha256_start=%s\nsource_tree_sha256_end=%s\n' \
  "$CIG_SOURCE_TREE_HASH" "$CIG_SOURCE_TREE_HASH_END" \
  > "$CIG_RUN_DIR/source_tree_integrity.txt"
echo
echo "=== [9/9] Final summary ==="
echo "Passed commands: $CIG_PASSES"
echo "Operational/protocol failures: ${#CIG_OPERATIONAL_FAILURES[@]}"
echo "Unsupported-claim reports: ${#CIG_CLAIM_FAILURES[@]}"
echo "Elapsed seconds: $((SECONDS - CIG_STARTED_AT))"
echo "Run artifacts: $CIG_RUN_DIR"
echo "Command manifest: $CIG_STATUS_TSV"
echo "Summary table: $CIG_RUN_DIR/summary_tables.md"
echo "Paper-A claim report: $CIG_RUN_DIR/paper_a_claim_status.json"
echo "Paper-B claim report: $CIG_RUN_DIR/paper_b_claim_status.json"
echo "Scientific gate ladder: $CIG_RUN_DIR/scientific_gate_status.json"

if [ "${#CIG_OPERATIONAL_FAILURES[@]}" -gt 0 ]; then
  printf '  - %s\n' "${CIG_OPERATIONAL_FAILURES[@]}"
  finish_metadata "INVALID"
  exit 2
fi

if [ "${#CIG_CLAIM_FAILURES[@]}" -gt 0 ]; then
  printf '  - %s\n' "${CIG_CLAIM_FAILURES[@]}"
  finish_metadata "COMPLETE_NOT_SUPPORTED"
  exit 3
fi

if [ "$CIG_SMOKE_ONLY" -eq 1 ]; then
  finish_metadata "COMPLETE_SMOKE_ONLY"
  echo "All selected smoke paths completed; no scientific claim was adjudicated."
  exit 0
fi

finish_metadata "COMPLETE_SUPPORTED"
echo "All selected checks completed and all valid claim gates were supported."
