#!/usr/bin/env bash
# Execute the complete CIG-AMF verification manifest in an isolated run root.
#
#   bash scripts/run_all.sh
#   bash scripts/run_all.sh --quick
#   bash scripts/run_all.sh --claims-only
#   bash scripts/run_all.sh --run-id 20260821_confirmatory
#
# Confirmatory mode uses eight paired H1 seeds and five paired H2/H3 seeds.
# Runtime settings can be overridden without editing this file:
#
#   CIG_H1_SEEDS="0 1 2 3 4 5 6 7" CIG_RUN_SEEDS="0 1 2 3 4" bash scripts/run_all.sh
#   CIG_H2_EPISODES=400 CIG_H3_EPISODES=200 bash scripts/run_all.sh
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
CIG_RUN_ID=""
CIG_DEVICE="${CIG_DEVICE:-cpu}"

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
  CIG_DEFAULT_H1_SEEDS="0"
  CIG_DEFAULT_H23_SEEDS="0"
  CIG_H2_EPISODES="${CIG_H2_EPISODES:-60}"
  CIG_H3_EPISODES="${CIG_H3_EPISODES:-60}"
  CIG_MODE="quick"
else
  CIG_DEFAULT_H1_SEEDS="0 1 2 3 4 5 6 7"
  CIG_DEFAULT_H23_SEEDS="0 1 2 3 4"
  CIG_H2_EPISODES="${CIG_H2_EPISODES:-400}"
  CIG_H3_EPISODES="${CIG_H3_EPISODES:-200}"
  CIG_MODE="confirmatory"
fi

CIG_H23_SEED_TEXT="${CIG_RUN_SEEDS:-$CIG_DEFAULT_H23_SEEDS}"
CIG_H1_SEED_TEXT="${CIG_H1_SEEDS:-${CIG_RUN_SEEDS:-$CIG_DEFAULT_H1_SEEDS}}"
read -r -a CIG_H1_SEED_ARRAY <<< "$CIG_H1_SEED_TEXT"
read -r -a CIG_H23_SEED_ARRAY <<< "$CIG_H23_SEED_TEXT"
if [ "${#CIG_H1_SEED_ARRAY[@]}" -eq 0 ] || [ "${#CIG_H23_SEED_ARRAY[@]}" -eq 0 ]; then
  echo "CIG_RUN_SEEDS resolved to an empty seed list." >&2
  exit 2
fi
for CIG_SEED in "${CIG_H1_SEED_ARRAY[@]}" "${CIG_H23_SEED_ARRAY[@]}"; do
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

for CIG_EPISODE_BUDGET in "$CIG_H2_EPISODES" "$CIG_H3_EPISODES"; do
  case "$CIG_EPISODE_BUDGET" in
    *[!0-9]*|""|0)
      echo "Episode budgets must be positive integers." >&2
      exit 2
      ;;
  esac
done

if [ "$CIG_QUICK" -eq 0 ]; then
  if [ "${#CIG_H1_SEED_ARRAY[@]}" -lt 8 ]; then
    echo "Confirmatory H1 requires at least 8 unique paired seeds." >&2
    exit 2
  fi
  if [ "${#CIG_H23_SEED_ARRAY[@]}" -lt 5 ]; then
    echo "Confirmatory H2/H3 require at least 5 unique paired seeds." >&2
    exit 2
  fi
  if [ "$CIG_H2_EPISODES" -lt 400 ] || [ "$CIG_H3_EPISODES" -lt 200 ]; then
    echo "Confirmatory budgets require H2>=400 and H3>=200 episodes." >&2
    echo "Use --quick for reduced protocol-path checks." >&2
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
  printf 'profile=%s\n' "$([ "$CIG_CLAIMS_ONLY" -eq 1 ] && printf claims-only || printf complete)"
  printf 'h1_seeds=%s\n' "${CIG_H1_SEED_ARRAY[*]}"
  printf 'h2_h3_seeds=%s\n' "${CIG_H23_SEED_ARRAY[*]}"
  printf 'h2_episodes=%s\n' "$CIG_H2_EPISODES"
  printf 'h3_episodes=%s\n' "$CIG_H3_EPISODES"
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
    echo "PASS: $label"
  else
    CIG_OPERATIONAL_FAILURES+=("$label (exit $command_status)")
    echo "FAIL: $label (exit $command_status)" >&2
  fi
  return 0
}

run_claim_validation() {
  local log_name="72_validate_claims.log"
  local command_started=$SECONDS

  echo
  echo "======================================================================"
  echo "RUN: Validate H1/H2/H3 scientific claims"
  echo "CATEGORY: claim"
  echo "LOG: $CIG_LOG_DIR/$log_name"
  echo "======================================================================"

  "$CIG_PYTHON" scripts/validate_claims.py \
    --run-root "$CIG_RUN_DIR" \
    --expected-h1-seeds "${CIG_H1_SEED_ARRAY[@]}" \
    --expected-h2-seeds "${CIG_H23_SEED_ARRAY[@]}" \
    --expected-h3-seeds "${CIG_H23_SEED_ARRAY[@]}" \
    --protocol-mode "$CIG_MODE" \
    2>&1 | tee "$CIG_LOG_DIR/$log_name"
  local command_status=${PIPESTATUS[0]}
  local command_elapsed=$((SECONDS - command_started))
  printf 'claim\t%s\t%s\t%s\t%s\n' \
    "Validate H1/H2/H3 scientific claims" "$command_status" \
    "$command_elapsed" "logs/$log_name" >> "$CIG_STATUS_TSV"

  case "$command_status" in
    0)
      CIG_PASSES=$((CIG_PASSES + 1))
      echo "SUPPORTED: all valid hypothesis gates passed"
      ;;
    10)
      CIG_CLAIM_FAILURES+=("At least one hypothesis was NOT_SUPPORTED")
      echo "NOT_SUPPORTED: see claim_status.md" >&2
      ;;
    11)
      CIG_PASSES=$((CIG_PASSES + 1))
      CIG_SMOKE_ONLY=1
      echo "SMOKE_ONLY: protocol paths completed; claims were not adjudicated"
      ;;
    *)
      CIG_OPERATIONAL_FAILURES+=("Claim validation invalid (exit $command_status)")
      echo "INVALID: claim report could not validate a complete protocol" >&2
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
echo "Planned H1 attempts: $((7 * ${#CIG_H1_SEED_ARRAY[@]}))"
echo "Planned H2 episodes: $((3 * 2 * ${#CIG_H23_SEED_ARRAY[@]} * CIG_H2_EPISODES))"
echo "Planned H3 episodes: $((6 * ${#CIG_H23_SEED_ARRAY[@]} * CIG_H3_EPISODES))"

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
run_logged "hypothesis" "H1 one-step causal identification and calibration" "60_h1.log" \
  "$CIG_PYTHON" scripts/run_h1_calibration.py \
  --seeds "${CIG_H1_SEED_ARRAY[@]}" \
  --device "$CIG_DEVICE" \
  --out-root "$CIG_RUN_DIR/h1" \
  --quiet
run_logged "hypothesis" "H2 structural selectivity and recovery" "61_h2.log" \
  "$CIG_PYTHON" scripts/run_h2_selectivity.py \
  --seeds "${CIG_H23_SEED_ARRAY[@]}" \
  --episodes "$CIG_H2_EPISODES" \
  --device "$CIG_DEVICE" \
  --out-root "$CIG_RUN_DIR/h2"
run_logged "hypothesis" "H3 specialization, capacity, and no-memory controls" "62_h3.log" \
  "$CIG_PYTHON" scripts/run_h3_slots.py \
  --seeds "${CIG_H23_SEED_ARRAY[@]}" \
  --episodes "$CIG_H3_EPISODES" \
  --device "$CIG_DEVICE" \
  --out-root "$CIG_RUN_DIR/h3"

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
run_logged "report" "Aggregate experiment results" "71_collect_results.log" \
  "$CIG_PYTHON" scripts/collect_results.py \
  --run-root "$CIG_RUN_DIR" \
  --expected-h1-seeds "${CIG_H1_SEED_ARRAY[@]}" \
  --expected-h2-seeds "${CIG_H23_SEED_ARRAY[@]}" \
  --expected-h3-seeds "${CIG_H23_SEED_ARRAY[@]}"
run_claim_validation

# ---------------------------------------------------------------------------
# 9. Final manifest summary.
# ---------------------------------------------------------------------------
echo
echo "=== [9/9] Final summary ==="
echo "Passed commands: $CIG_PASSES"
echo "Operational/protocol failures: ${#CIG_OPERATIONAL_FAILURES[@]}"
echo "Unsupported-claim reports: ${#CIG_CLAIM_FAILURES[@]}"
echo "Elapsed seconds: $((SECONDS - CIG_STARTED_AT))"
echo "Run artifacts: $CIG_RUN_DIR"
echo "Command manifest: $CIG_STATUS_TSV"
echo "Summary table: $CIG_RUN_DIR/summary_tables.md"
echo "Claim report: $CIG_RUN_DIR/claim_status.md"

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
