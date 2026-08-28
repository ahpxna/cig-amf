"""Controlled CPU micro-experiment for the Paper-07/09 D0 gate.

This is a deterministic synthetic falsification gate, not evidence from the
learned CIG-AMF policy.  Quick mode can only produce SMOKE_ONLY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from models.disturbance_contracts import (
    DisturbanceRegime, PairedDisturbanceRecord, adjudicate_d0,
)
from scripts.scientific_gate_common import atomic_json

PROTOCOL_VERSION = "p11_d0_controlled_microgate_v1"
ARMS = {"localized": np.array([1.0, 0.0]), "distributed": np.array([0.5, 0.5])}


def _sha(value) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def collect(replicates: int):
    rows, records = [], []
    transition = np.array([[0.72, 0.22], [0.08, 0.51]])
    response = np.array([0.8, -0.35])
    for replicate in range(replicates):
        noise = np.random.default_rng(911 + replicate).normal(0.0, 0.015, (4, 2))
        for regime in DisturbanceRegime:
            for arm, intervention in ARMS.items():
                baseline = np.zeros(2)
                treated = intervention.copy()
                policy_shift = 0.0
                for step in range(4):
                    baseline = transition @ baseline + noise[step]
                    if regime == DisturbanceRegime.RESET:
                        treated = baseline.copy()
                    else:
                        treated = transition @ treated + noise[step]
                        if regime == DisturbanceRegime.LIVE_LEARNING:
                            policy_shift += 0.08 * abs(float(response @ treated))
                            treated = treated + 0.03 * response * policy_shift
                record = PairedDisturbanceRecord(
                    regime=regime, arm=arm, replicate=replicate,
                    target_key="fixed_two_state_target_v1",
                    immediate_cost=float(np.abs(intervention).sum()),
                    future_state_distance=float(np.linalg.norm(treated - baseline)),
                    future_response_shift=abs(float(response @ (treated - baseline))),
                    future_policy_distance=float(policy_shift),
                )
                rows.append(record)
                item = dict(record.__dict__)
                item["regime"] = regime.value
                item.update({
                    "action_applied": True,
                    "realized_action_sha256": _sha(intervention),
                    "common_noise_sha256": _sha(noise),
                    "paired_baseline_verified": True,
                })
                records.append(item)
    return rows, records


def run(*, mode: str, replicates: int):
    if replicates < 2:
        raise ValueError("at least two paired replicates are required")
    if mode == "confirmatory" and replicates < 30:
        raise ValueError("confirmatory mode requires at least 30 paired replicates")
    rows, records = collect(replicates)
    gate = adjudicate_d0(
        rows,
        metric_thresholds={
            "future_state_distance": 0.05,
            "future_response_shift": 0.02,
            "future_policy_distance": 0.01,
        },
        minimum_arm_spreads={
            "future_state_distance": 0.005,
            "future_response_shift": 0.005,
            "future_policy_distance": 0.001,
        },
        immediate_cost_tolerance=0.0,
        minimum_replicates=2 if mode == "quick" else 30,
    )
    status = "SMOKE_ONLY" if mode == "quick" else gate.status
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "collection_complete": True,
        "d0_status": status,
        "raw_gate_status": gate.status,
        "evidence_class": "DETERMINISTIC_SYNTHETIC_GATE_ONLY",
        "reasons": list(gate.reasons),
        "records": records,
        "regime_metric_means": gate.regime_metric_means,
        "arm_metric_means": gate.arm_metric_means,
        "manifest": {
            "replicates": replicates,
            "regimes": [item.value for item in DisturbanceRegime],
            "arms": sorted(ARMS),
            "fixed_target": "fixed_two_state_target_v1",
            "matched_immediate_cost": True,
            "common_random_numbers": True,
            "learned_cig_amf_policy_used": False,
            "limitation": "does not establish the Paper-07/09 causal theorem",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "confirmatory"), default="quick")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        payload = run(mode=args.mode, replicates=args.replicates)
    except Exception as exc:
        payload = {
            "protocol_version": PROTOCOL_VERSION, "mode": args.mode,
            "collection_complete": False, "d0_status": "FAILED",
            "evidence_class": "NO_EVIDENCE", "reasons": [str(exc)], "records": [],
        }
        atomic_json(Path(args.out), payload)
        raise
    atomic_json(Path(args.out), payload)
    print(json.dumps({key: payload[key] for key in (
        "mode", "collection_complete", "d0_status", "raw_gate_status"
    )}, indent=2))
    return payload


if __name__ == "__main__":
    main()
