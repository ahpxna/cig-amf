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
from envs.causal_adapter import resolve_env_adapter


ROLE_ORDER = ("blocker", "gatekeeper", "relay", "controller")


def _finite_median(values):
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(valid)) if valid else None


def _rank_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    def average_ranks(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return ranks

    rank_x = average_ranks(x)
    rank_y = average_ranks(y)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


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

    active_rows = [row for row in rows if row["response_mass"] >= min_response_mass]
    true_delays = [row["true_delay"] for row in active_rows]
    estimated_delays = [row["centre_of_mass_lag"] for row in active_rows]
    valid_pairs = [
        (truth, estimate)
        for truth, estimate in zip(true_delays, estimated_delays)
        if estimate is not None and np.isfinite(estimate)
    ]
    delay_rank = _rank_correlation(
        [item[0] for item in valid_pairs], [item[1] for item in valid_pairs]
    )
    delay_mae = (
        float(np.mean([abs(item[0] - item[1]) for item in valid_pairs]))
        if valid_pairs else float("nan")
    )
    active_fraction = float(len(active_rows) / max(1, len(rows)))
    # The primary gate uses pair-randomized mechanism delay, independent of
    # role. Role ordering remains an interpretability diagnostic only.
    gate_pass = bool(
        active_fraction >= 0.5
        and len(valid_pairs) >= 8
        and delay_rank >= 0.70
        and np.isfinite(delay_mae)
        and delay_mae <= 1.5
    )
    reasons = []
    if active_fraction < 0.5:
        reasons.append("fewer than half of randomized-delay pairs have stable response mass")
    if len(valid_pairs) < 8:
        reasons.append("too few active pair-state latency estimates")
    if delay_rank < 0.70:
        reasons.append("capacity-spectrum latency does not rank randomized delay")
    if not np.isfinite(delay_mae) or delay_mae > 1.5:
        reasons.append("capacity-spectrum latency error exceeds the oracle gate")

    return {
        "gate_pass": gate_pass,
        "gate_reasons": reasons,
        "usable_roles": usable_roles,
        "expected_role_order": list(ROLE_ORDER),
        "expected_order_pass": expected_order,
        "centre_of_mass_spread": centre_spread,
        "peak_lag_spread": peak_spread,
        "role_summary": role_summary,
        "randomized_delay_rank_correlation": delay_rank,
        "randomized_delay_mae": delay_mae,
        "active_pair_state_fraction": active_fraction,
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
        enable_sgtp_delays=True,
    )
    # The sampled states are all safely inside the same episode regime.  The
    # oracle itself rejects a boundary-crossing H-step window.
    bank = env.sample_state_bank(
        n_states=int(n_states),
        burn_in=3,
        bank_seed=int(seed) + 7103,
        min_remaining_steps=int(horizon),
    )
    rows = []
    zone = 0
    role_agents = env.zone_role_agents[zone]
    ego = int(role_agents[env.ROLE_COLLECTOR])

    for state_index, state in enumerate(bank):
        for role in ROLE_ORDER:
            source = int(role_agents[role])
            action_profiles = []
            adapter = resolve_env_adapter(env)
            valid_actions = np.flatnonzero(adapter.valid_action_mask(source))
            for action in valid_actions:
                env.restore_state(state)
                profile = env.compute_oracle_lag_response_from_current_state(
                    ego_id=ego,
                    agent_j=source,
                    intervention_action=action,
                    horizon=int(horizon),
                    n_trials=int(n_trials),
                    forced_step=0,
                    crn_seed=(int(seed) + 1) * 1000003 + state_index * 97 + source * 11,
                )
                action_profiles.append(
                    np.asarray(profile["per_lag_response"], dtype=np.float64)
                )

            response_surface = np.stack(action_profiles, axis=0)  # [A,H]
            capacity_spectrum = (
                np.max(response_surface, axis=0)
                - np.min(response_surface, axis=0)
            )
            response_mass = float(np.sum(np.abs(capacity_spectrum)))
            peak = float(np.max(capacity_spectrum)) if capacity_spectrum.size else 0.0
            active_threshold = max(1e-8, 0.05 * peak)
            active_lags = np.flatnonzero(capacity_spectrum > active_threshold)
            onset_lag = int(active_lags[0]) if active_lags.size else None
            peak_lag = int(np.argmax(capacity_spectrum)) if peak > 0.0 else None
            centre_of_mass_lag = (
                float(
                    np.dot(
                        np.arange(capacity_spectrum.size, dtype=np.float64),
                        capacity_spectrum,
                    )
                    / response_mass
                )
                if response_mass > 0.0
                else None
            )
            rows.append(
                {
                    "state_index": int(state_index),
                    "role": role,
                    "ego": ego,
                    "source": source,
                    "true_delay": int(env.sgtp_delay_by_pair.get((source, ego), 0)),
                    "action_lag_response": response_surface.tolist(),
                    "capacity_lag_spectrum": capacity_spectrum.tolist(),
                    "response_mass": response_mass,
                    "onset_lag": onset_lag,
                    "peak_lag": peak_lag,
                    "centre_of_mass_lag": centre_of_mass_lag,
                }
            )

    summary = _profile_summary(rows, min_response_mass=float(min_response_mass))
    return {
        "protocol_version": "latency_oracle_capacity_spectrum_v2",
        "continuation_policy": "OmniArena.scripted_policy",
        "seed": int(seed),
        "n_states": int(n_states),
        "horizon": int(horizon),
        "n_trials": int(n_trials),
        "action_count": int(env.get_action_dim()),
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
