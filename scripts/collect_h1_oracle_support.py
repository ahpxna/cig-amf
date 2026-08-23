"""Collect a development-only, estimator-free H1 oracle-support bank.

This command records only clone-state oracle response surfaces under the
prespecified fixed ``pi_eval`` policy.  It is intentionally separate from
the epsilon/estimator sweep so threshold and identifiability decisions cannot
depend on learned predictions or on a selected estimator variant.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import tempfile

import numpy as np

try:
    from exp_common import ROOT, ensure_dir
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, ensure_dir

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.tiny_oracle_resource_flow_v1 import TinyOracleResourceFlowV1
import run_experiment as RE


def _atomic_csv(path, rows):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    fd, temporary = tempfile.mkstemp(prefix=".h1-oracle-", suffix=".csv", dir=directory)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".h1-oracle-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _pi_eval(env, action, uniform_mass):
    n_actions = int(env.get_action_dim())
    policy = np.full((n_actions,), float(uniform_mass) / float(n_actions))
    policy[int(action)] += 1.0 - float(uniform_mass)
    return policy


def _rows_for_state(env, state, state_idx, seed, uniform_mass):
    env.restore_state(copy.deepcopy(state))
    factual_actions = [int(value) for value in env.default_joint_action()]
    pi_rows = [_pi_eval(env, action, uniform_mass) for action in factual_actions]
    _, factual_rewards, _, _ = env.step(list(factual_actions))
    env.restore_state(copy.deepcopy(state))
    step = {
        "env_snapshot_before_step": copy.deepcopy(state),
        "actions": factual_actions,
        "rewards": [float(value) for value in factual_rewards],
        "policy_probs": pi_rows,
    }
    candidate_actions = RE._tiny_candidate_intervention_actions(env)
    records = []
    for ego in env.get_supported_egos():
        neighbors = [target for target in range(env.n_agents) if target != ego]
        oracle, replay_error = RE._h1_one_step_oracle_scores(
            env, step, ego, neighbors, candidate_actions, pi_rows
        )
        if replay_error > 1e-9:
            raise RuntimeError("oracle support collector failed factual replay")
        for target in neighbors:
            q_values = np.asarray(oracle["q"][target], dtype=np.float64)
            valid_count = int(q_values.size)
            uniform = np.full((valid_count,), 1.0 / max(1, valid_count))
            target_pi = pi_rows[target]
            records.append({
                "seed": int(seed),
                "state_idx": int(state_idx),
                "ego_id": int(ego),
                "neighbor_id": int(target),
                "oracle_score": float(oracle["capacity"][target]),
                "oracle_signed": float(oracle["direction"][target]),
                "oracle_q_nonconstant": int(np.ptp(q_values) > 1e-12),
                "oracle_q_distinct_levels": int(np.unique(np.round(q_values, 12)).size),
                "valid_action_count": valid_count,
                "target_policy_l1_to_uniform": float(
                    np.sum(np.abs(target_pi - uniform))
                ),
                "oracle_only": 1,
            })
    return records


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--states-per-seed", type=int, default=100)
    parser.add_argument("--burn-in", type=int, default=3)
    parser.add_argument("--h1-eval-uniform-mass", type=float, default=0.10)
    parser.add_argument("--out-root", default=os.path.join(ROOT, "results", "h1_oracle_support"))
    args = parser.parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be non-empty and unique")
    if args.states_per_seed <= 0 or args.burn_in < 0:
        parser.error("states-per-seed must be positive and burn-in non-negative")
    if not 0.0 < args.h1_eval_uniform_mass < 1.0:
        parser.error("h1-eval-uniform-mass must be in (0, 1)")

    out_root = ensure_dir(os.path.abspath(args.out_root))
    rows = []
    for seed in args.seeds:
        env = TinyOracleResourceFlowV1(seed=int(seed))
        bank = env.sample_state_bank(
            n_states=int(args.states_per_seed),
            burn_in=int(args.burn_in),
            bank_seed=int(seed) + 70123,
        )
        for state_idx, state in enumerate(bank):
            rows.extend(_rows_for_state(
                env, state, state_idx, seed, args.h1_eval_uniform_mass
            ))

    csv_path = os.path.join(out_root, "tiny_oracle_pair_rows.csv")
    metadata_path = os.path.join(out_root, "oracle_support_manifest.json")
    _atomic_csv(csv_path, rows)
    _atomic_json(metadata_path, {
        "protocol": "h1_oracle_support_collection_v1",
        "oracle_only": True,
        "development_seeds": [int(seed) for seed in args.seeds],
        "states_per_seed": int(args.states_per_seed),
        "burn_in": int(args.burn_in),
        "h1_target_policy_mode": "scripted_uniform_mixture",
        "h1_eval_uniform_mass": float(args.h1_eval_uniform_mass),
        "pair_rows": int(len(rows)),
        "pair_rows_csv": csv_path,
    })
    print(json.dumps({"pair_rows_csv": csv_path, "manifest": metadata_path}, indent=2))


if __name__ == "__main__":
    main()
