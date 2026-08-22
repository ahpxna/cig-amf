#!/usr/bin/env python3
"""Oracle gate for the optional lag-specific causal latency contribution.

The gate measures a one-time action intervention followed by the same fixed
reference continuation policy in both branches.  It does not train a latency
head and does not make a paper claim.  A failed gate therefore records that
latency must remain absent from the learned signature rather than turning a
diagnostic failure into an execution failure.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.omni_arena import OmniArena


ROLE_ORDER = ("blocker", "gatekeeper", "relay", "controller")


def _finite_median(values):
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(valid)) if valid else None


def _profile_summary(rows, min_response_mass):
    by_role = {role: [] for role in ROLE_ORDER}
    for row in rows:
        by_role[row["role"]].append(row)

    role_summary = {}
    usable_roles = []
    for role, role_rows in by_role.items():
        active = [row for row in role_rows if row["response_mass"] >= min_response_mass]
        summary = {
            "n_states": len(role_rows),
            "n_active": len(active),
            "response_mass_median": _finite_median([row["response_mass"] for row in role_rows]),
            "onset_lag_median": _finite_median([row["onset_lag"] for row in active]),
            "peak_lag_median": _finite_median([row["peak_lag"] for row in active]),
            "centre_of_mass_lag_median": _finite_median(
                [row["centre_of_mass_lag"] for row in active]
            ),
        }
        summary["usable"] = bool(
            summary["n_active"] >= max(2, int(np.ceil(0.5 * max(1, len(role_rows)))))
            and summary["centre_of_mass_lag_median"] is not None
        )
        if summary["usable"]:
            usable_roles.append(role)
        role_summary[role] = summary

    centres = [
        role_summary[role]["centre_of_mass_lag_median"]
        for role in usable_roles
    ]
    peaks = [role_summary[role]["peak_lag_median"] for role in usable_roles]
    centre_spread = float(max(centres) - min(centres)) if len(centres) >= 2 else 0.0
    peak_spread = float(max(peaks) - min(peaks)) if len(peaks) >= 2 else 0.0

    expected_centres = [
        role_summary[role]["centre_of_mass_lag_median"]
        for role in ROLE_ORDER
    ]
    all_roles_usable = len(usable_roles) == len(ROLE_ORDER)
    expected_order = bool(
        all_roles_usable
        and all(
            expected_centres[index] <= expected_centres[index + 1] + 0.25
            for index in range(len(expected_centres) - 1)
        )
    )

    # The environment declares a blocker -> gatekeeper -> relay -> controller
    # latency ladder.  A generic three-role spread could pass even when the
    # declared relay/controller tiers are absent, which would not support that
    # contribution.  Require measurable responses for every declared tier,
    # its coarse ordering, and a non-trivial total spread.  This remains an
    # oracle/environment criterion, not a learned proxy metric.
    gate_pass = bool(
        all_roles_usable
        and expected_order
        and (centre_spread >= 0.50 or peak_spread >= 1.0)
    )
    reasons = []
    if not all_roles_usable:
        reasons.append("not every declared latency tier has stable response mass")
    if all_roles_usable and not expected_order:
        reasons.append("declared blocker-to-controller latency order is not recovered")
    if centre_spread < 0.50 and peak_spread < 1.0:
        reasons.append("lag summaries lack the required role separation")

    return {
        "gate_pass": gate_pass,
        "gate_reasons": reasons,
        "usable_roles": usable_roles,
        "expected_role_order": list(ROLE_ORDER),
        "expected_order_pass": expected_order,
        "centre_of_mass_spread": centre_spread,
        "peak_lag_spread": peak_spread,
        "role_summary": role_summary,
    }


def run_gate(
    seed=0,
    n_states=12,
    horizon=8,
    n_trials=2,
    min_response_mass=0.02,
):
    env = OmniArena(
        n_agents=24,
        n_zones=4,
        max_steps=60,
        phase_length=6,
        causal_horizon=int(horizon),
        seed=int(seed),
        mode="cooperative",
        enable_latency_ladder=True,
    )
    # The sampled states are all safely inside the same episode regime.  The
    # oracle itself rejects a boundary-crossing H-step window.
    bank = env.sample_state_bank(
        n_states=int(n_states),
        burn_in=3,
        bank_seed=int(seed) + 7103,
    )
    rows = []
    zone = 0
    role_agents = env.zone_role_agents[zone]
    ego = int(role_agents[env.ROLE_COLLECTOR])

    for state_index, state in enumerate(bank):
        for role in ROLE_ORDER:
            source = int(role_agents[role])
            candidates = []
            for action in range(env.N_ACTIONS):
                env.restore_state(state)
                profile = env.compute_oracle_lag_response_from_current_state(
                    ego_id=ego,
                    agent_j=source,
                    intervention_action=action,
                    horizon=int(horizon),
                    n_trials=int(n_trials),
                    forced_step=0,
                    crn_seed=(int(seed) + 1) * 1000003 + state_index * 97 + source * 11 + action,
                )
                candidates.append((profile["response_mass"], action, profile))

            _, action, profile = max(candidates, key=lambda item: item[0])
            rows.append(
                {
                    "state_index": int(state_index),
                    "role": role,
                    "ego": ego,
                    "source": source,
                    "selected_action": int(action),
                    "per_lag_response": [float(x) for x in profile["per_lag_response"]],
                    "discounted_response": float(profile["discounted_response"]),
                    "response_mass": float(profile["response_mass"]),
                    "onset_lag": profile["onset_lag"],
                    "peak_lag": profile["peak_lag"],
                    "centre_of_mass_lag": profile["centre_of_mass_lag"],
                }
            )

    summary = _profile_summary(rows, min_response_mass=float(min_response_mass))
    return {
        "protocol_version": "latency_oracle_impulse_v1",
        "continuation_policy": "OmniArena.scripted_policy",
        "seed": int(seed),
        "n_states": int(n_states),
        "horizon": int(horizon),
        "n_trials": int(n_trials),
        "min_response_mass": float(min_response_mass),
        "rows": rows,
        **summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--min-response-mass", type=float, default=0.02)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    result = run_gate(
        seed=args.seed,
        n_states=args.states,
        horizon=args.horizon,
        n_trials=args.trials,
        min_response_mass=args.min_response_mass,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    console = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(console, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
