#!/usr/bin/env bash
# Chạy toàn bộ pipeline nghiệm thu qua đêm. Log vào results/logs/.
#   bash scripts/run_all.sh            # đầy đủ (3 seed)
#   bash scripts/run_all.sh --quick    # 1 seed, ít episode (kiểm tra pipeline)
set -u
cd "$(dirname "$0")/.."
source cig-env/bin/activate 2>/dev/null || true
mkdir -p results/logs

if [ "${1:-}" = "--quick" ]; then
  SEEDS="0"; H2_EPS=60; H3_EPS=60
else
  SEEDS="0 1 2"; H2_EPS=400; H3_EPS=200
fi

echo "=== [0/4] Env gates ==="
( cd envs && python test_p2_p4_guards.py && python env_audit.py ) \
  2>&1 | tee results/logs/env_audit.log
python test_bc_loss_control.py 2>&1 | tee results/logs/bc_test.log
python smoke_test.py           2>&1 | tee results/logs/smoke.log

echo "=== [1/4] H1 ==="
python scripts/run_h1_calibration.py --seeds $SEEDS 2>&1 | tee results/logs/h1.log
echo "=== [2/4] H2 ==="
python scripts/run_h2_selectivity.py --seeds $SEEDS --episodes $H2_EPS 2>&1 | tee results/logs/h2.log
echo "=== [3/4] H3 ==="
python scripts/run_h3_slots.py --seeds $SEEDS --episodes $H3_EPS 2>&1 | tee results/logs/h3.log
echo "=== [4/4] Collect ==="
python scripts/collect_results.py 2>&1 | tee results/logs/collect.log
echo "DONE -> results/summary_tables.md"
