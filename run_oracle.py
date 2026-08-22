"""Run the zero-cost true-oracle structure-value prerequisite (Experiment 0)."""

import argparse
import json
import math
import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
ENVS_DIR = os.path.join(ROOT, "envs")
if ENVS_DIR not in sys.path:
    sys.path.append(ENVS_DIR)

from omni_arena import OmniArena  # noqa: E402
from structure_value_tier0 import run_tier0  # noqa: E402


def main(json_out=None):
    print("=" * 72)
    print("TRUE ORACLE-CORE (ZERO COST): STRUCTURE-VALUE PREREQUISITE")
    print("=" * 72)

    env = OmniArena(
        n_agents=24,
        grid_size=24,
        n_zones=4,
        enable_conditional_gates=True,
        enable_latency_ladder=True,
        enable_congestion=True,
        enable_structural_shift=True,
    )
    metrics, _ = run_tier0(
        env=env,
        n_states=20,
        steps_between=10,
        k_core=3,
        horizon=8,
        seed=123,
    )

    print("\nRaw metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  - {key:<25}: {value:.6f}")
        else:
            print(f"  - {key:<25}: {value}")

    frac_changed = float(metrics.get("frac_action_changed", 0.0))
    norm_std = float(metrics.get("norm_by_std", float("nan")))
    checks = [
        {
            "name": "oracle structure changes the selected action",
            "value": frac_changed,
            "threshold": "> 0.05",
            "passed": frac_changed > 0.05,
        },
        {
            "name": "oracle structure has material normalized value",
            "value": norm_std,
            "threshold": "finite and > 0.05",
            "passed": math.isfinite(norm_std) and norm_std > 0.05,
        },
    ]
    gate_pass = all(row["passed"] for row in checks)

    print("\nExperiment 0 gate:")
    for row in checks:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"  [{status}] {row['name']}: {row['value']:.6f} "
            f"({row['threshold']})"
        )
    print(f"Required Experiment 0 gate: {'PASS' if gate_pass else 'FAIL'}")

    payload = {
        "experiment": "true_oracle_structure_value",
        "required_gate_pass": gate_pass,
        "checks": checks,
        "metrics": metrics,
    }
    if json_out:
        output_path = os.path.abspath(json_out)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=float)
            handle.write("\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=None)
    cli_args = parser.parse_args()
    result = main(json_out=cli_args.json_out)
    raise SystemExit(0 if result["required_gate_pass"] else 2)
