#!/usr/bin/env bash
# Run the complete CIG-AMF validation and experiment manifest.
#
#   bash scripts/run_all.sh          # every diagnostic and experiment
#   bash scripts/run_all.sh --quick  # reduced seeds/episodes; skips long runs
#
# Every command receives a separate log under results/logs/run_all/. A failed
# command is recorded without aborting the remaining manifest, so one early
# regression cannot hide later failures. The script exits nonzero after the
# final summary if any command failed.
set -uo pipefail

CIG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CIG_ROOT"

CIG_LOG_DIR="$CIG_ROOT/results/logs/run_all"
mkdir -p "$CIG_LOG_DIR"

if [ -x "$CIG_ROOT/cig-env/bin/python" ]; then
  CIG_PYTHON="$CIG_ROOT/cig-env/bin/python"
else
  CIG_PYTHON="${CIG_PYTHON:-python3}"
fi

CIG_QUICK=0
for arg in "$@"; do
  case "$arg" in
    --quick)
      CIG_QUICK=1
      ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [ "$CIG_QUICK" -eq 1 ]; then
  CIG_SEEDS=(0)
  CIG_H2_EPISODES=60
  CIG_H3_EPISODES=60
else
  CIG_SEEDS=(0 1 2)
  CIG_H2_EPISODES=400
  CIG_H3_EPISODES=200
fi

CIG_FAILURES=()
CIG_PASSES=0
CIG_STARTED_AT=$SECONDS

run_logged() {
  local label="$1"
  local log_name="$2"
  shift 2

  echo
  echo "======================================================================"
  echo "RUN: $label"
  echo "LOG: results/logs/run_all/$log_name"
  echo "CMD: $*"
  echo "======================================================================"

  "$@" 2>&1 | tee "$CIG_LOG_DIR/$log_name"
  local command_status=${PIPESTATUS[0]}

  if [ "$command_status" -eq 0 ]; then
    CIG_PASSES=$((CIG_PASSES + 1))
    echo "PASS: $label"
  else
    CIG_FAILURES+=("$label (exit $command_status)")
    echo "FAIL: $label (exit $command_status)" >&2
  fi
}

echo "Python: $CIG_PYTHON"
if [ "$CIG_QUICK" -eq 1 ]; then
  echo "Mode: quick"
else
  echo "Mode: full"
fi

# ---------------------------------------------------------------------------
# 0. Static and local mechanism checks
# ---------------------------------------------------------------------------
echo "=== [0/8] Static, smoke, and mechanism checks ==="
run_logged "Python syntax compilation" "00_py_compile.log" \
  "$CIG_PYTHON" -m compileall -q \
  envs models runners scripts training utils \
  run_experiment.py run_main_exp.py run_oracle.py run_step_0.py run_step_1.py \
  smoke_test.py test_bc_loss_control.py
run_logged "Mechanism smoke suite" "01_smoke_test.log" \
  "$CIG_PYTHON" smoke_test.py
run_logged "Behaviour-cloning loss controls" "02_bc_loss_control.log" \
  "$CIG_PYTHON" test_bc_loss_control.py

# ---------------------------------------------------------------------------
# 1. Focused environment invariants and diagnostics
# ---------------------------------------------------------------------------
echo "=== [1/8] Focused environment checks ==="
run_logged "Frenet coordinate check (C1)" "10_env_test_c1.log" \
  "$CIG_PYTHON" envs/test_c1.py
run_logged "P2/P4 boundary guards" "11_env_p2_p4_guards.log" \
  "$CIG_PYTHON" envs/test_p2_p4_guards.py
run_logged "Tiny-oracle signed-Phi acceptance" "12_env_p0_oracle_check.log" \
  "$CIG_PYTHON" envs/_p0_oracle_check.py
run_logged "Congestion-channel activation diagnostic" "13_env_congestion.log" \
  "$CIG_PYTHON" envs/_diag_congestion.py
run_logged "Phi/W* horizon and unit diagnostic" "14_env_phi_wstar_horizon.log" \
  "$CIG_PYTHON" envs/code_test.py

# ---------------------------------------------------------------------------
# 2. Environment acceptance gates
# ---------------------------------------------------------------------------
echo "=== [2/8] Environment acceptance gates ==="
run_logged "All-flags environment audit" "20_env_audit.log" \
  "$CIG_PYTHON" envs/env_audit.py
run_logged "Cumulative staged environment audit" "21_env_audit_staged.log" \
  "$CIG_PYTHON" envs/env_audit_staged.py

# ---------------------------------------------------------------------------
# 3. Structure-value experiments
# ---------------------------------------------------------------------------
echo "=== [3/8] Structure-value experiments ==="
if [ "$CIG_QUICK" -eq 0 ]; then
  run_logged "Tier 0 oracle structure regret" "30_structure_value_tier0.log" \
    "$CIG_PYTHON" envs/structure_value_tier0.py
  run_logged "Tier 0 cumulative staged structure regret" "31_structure_value_tier0_staged.log" \
    "$CIG_PYTHON" envs/structure_value_tier0_staged.py
  run_logged "Tier 1 predictive value gap" "32_structure_value_tier1.log" \
    "$CIG_PYTHON" envs/structure_value_tier1.py
  run_logged "Tier 2 learned structure value" "33_structure_value_tier2.log" \
    "$CIG_PYTHON" envs/structure_value_tier2.py
  run_logged "True zero-cost oracle-core Experiment 0" "34_run_oracle.log" \
    "$CIG_PYTHON" run_oracle.py
else
  echo "SKIP: long structure-value tiers and run_oracle.py in --quick mode."
fi

# ---------------------------------------------------------------------------
# 4. Stage entry points requested by the debugging protocol
# ---------------------------------------------------------------------------
echo "=== [4/8] Step entry points ==="
if [ "$CIG_QUICK" -eq 0 ]; then
  run_logged "Step 0 learned structure-sensitivity comparison" "40_run_step_0.log" \
    "$CIG_PYTHON" run_step_0.py
  run_logged "Step 1 learned-stage diagnostic" "41_run_step_1.log" \
    "$CIG_PYTHON" run_step_1.py
else
  echo "SKIP: run_step_0.py and run_step_1.py in --quick mode (hard-coded long runs)."
fi

# ---------------------------------------------------------------------------
# 5. Main runner integration and legacy comparison
# ---------------------------------------------------------------------------
echo "=== [5/8] Main runner integration ==="
run_logged "Main CLI smoke: behavioral task" "50_main_cli_behavioral_smoke.log" \
  "$CIG_PYTHON" run_experiment.py \
  --task behavioral \
  --models PureMeanField,Final-CIGAMF \
  --episodes 2 \
  --max_steps 12 \
  --eval_every 1 \
  --seed 0 \
  --device cpu \
  --smoke \
  --result_dir results/run_all_smoke
run_logged "Main CLI smoke: tiny-oracle task" "51_main_cli_tiny_smoke.log" \
  "$CIG_PYTHON" run_experiment.py \
  --task tiny \
  --tiny_states 3 \
  --tiny_proxy_train_episodes 2 \
  --seed 0 \
  --device cpu \
  --smoke \
  --result_dir results/run_all_smoke

if [ "$CIG_QUICK" -eq 0 ]; then
  run_logged "Legacy three-model comparison" "52_run_main_exp.log" \
    "$CIG_PYTHON" run_main_exp.py
fi

# ---------------------------------------------------------------------------
# 6. Paper hypotheses and ablations
# ---------------------------------------------------------------------------
echo "=== [6/8] H1/H2/H3 experiments ==="
run_logged "H1 causal identification and calibration" "60_h1.log" \
  "$CIG_PYTHON" scripts/run_h1_calibration.py \
  --seeds "${CIG_SEEDS[@]}" \
  --device cpu
run_logged "H2 structural selectivity and recovery" "61_h2.log" \
  "$CIG_PYTHON" scripts/run_h2_selectivity.py \
  --seeds "${CIG_SEEDS[@]}" \
  --episodes "$CIG_H2_EPISODES" \
  --device cpu
run_logged "H3 slot specialization and capacity ablations" "62_h3.log" \
  "$CIG_PYTHON" scripts/run_h3_slots.py \
  --seeds "${CIG_SEEDS[@]}" \
  --episodes "$CIG_H3_EPISODES" \
  --device cpu

# ---------------------------------------------------------------------------
# 7. Figures and aggregation
# ---------------------------------------------------------------------------
echo "=== [7/8] Diagnostic figures and result aggregation ==="
run_logged "W* timeline diagnostic" "70_w_star_timeline.log" \
  "$CIG_PYTHON" scripts/plot_w_star_timeline.py \
  --steps 20 \
  --horizon 8 \
  --seed 123 \
  --tag run_all
run_logged "Aggregate experiment results" "71_collect_results.log" \
  "$CIG_PYTHON" scripts/collect_results.py

# ---------------------------------------------------------------------------
# 8. Final manifest summary
# ---------------------------------------------------------------------------
echo
echo "=== [8/8] Final summary ==="
echo "Passed commands: $CIG_PASSES"
echo "Failed commands: ${#CIG_FAILURES[@]}"
echo "Elapsed seconds: $((SECONDS - CIG_STARTED_AT))"
echo "Logs: $CIG_LOG_DIR"
echo "Summary table: $CIG_ROOT/results/summary_tables.md"

if [ "${#CIG_FAILURES[@]}" -gt 0 ]; then
  printf '  - %s\n' "${CIG_FAILURES[@]}"
  exit 1
fi

echo "All selected checks completed successfully."
