"""Train and gate the learned direct-lag capacity spectrum after oracle pass."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.omni_arena import OmniArena
from models.structural_proxy import build_pair_feat
import run_experiment as RE
try:
    from run_latency_oracle import _rank_correlation
except ModuleNotFoundError:
    from scripts.run_latency_oracle import _rank_correlation


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".latency-calibration-", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=float)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _centre(spectrum):
    values = np.clip(np.asarray(spectrum, dtype=np.float64), 0.0, None)
    total = float(values.sum())
    if total <= 1e-12:
        return None
    return float(np.dot(np.arange(values.size), values) / total)


def _oracle_spectrum(env, state, ego, source, horizon, trials, seed):
    profiles = []
    for action in range(env.N_ACTIONS):
        env.restore_state(state)
        result = env.compute_oracle_lag_response_from_current_state(
            ego_id=int(ego), agent_j=int(source), intervention_action=int(action),
            horizon=int(horizon), n_trials=int(trials), crn_seed=int(seed),
        )
        profiles.append(np.asarray(result["per_lag_response"], dtype=np.float64))
    surface = np.stack(profiles, axis=0)
    return np.max(surface, axis=0) - np.min(surface, axis=0)


def run(seed, train_episodes, states, horizon, trials, device):
    RE.set_global_seed(seed)
    env = OmniArena(
        n_agents=24, n_zones=4, max_steps=60, phase_length=40,
        causal_horizon=int(horizon), seed=int(seed), mode="cooperative",
        enable_latency_ladder=True, enable_sgtp_delays=True,
    )
    cfg = RE.default_cfg()
    cfg.update({
        "seed": int(seed), "causal_horizon": int(horizon),
        "proxy_n_horizons": int(horizon), "k0_warmup": 0,
        "proxy_use_doubly_robust": False,
    })
    runner = RE.make_runner("Final-CIGAMF", env, cfg, device)
    runner.run(n_episodes=int(train_episodes), eval_every=max(1, int(train_episodes)))
    bank = env.sample_state_bank(
        n_states=int(states), burn_in=3, bank_seed=int(seed) + 9901
    )
    role_agents = env.zone_role_agents[0]
    ego = int(role_agents[env.ROLE_COLLECTOR])
    sources = [
        int(role_agents[role])
        for role in (
            env.ROLE_BLOCKER, env.ROLE_GATEKEEPER,
            env.ROLE_RELAY, env.ROLE_CONTROLLER,
        )
    ]
    rows = []
    old_mode = runner.proxy.effect_mode
    runner.proxy.effect_mode = "range"
    try:
        for state_index, state in enumerate(bank):
            env.restore_state(state)
            obs_all = env._get_obs_all()
            actions = [int(env.scripted_policy(agent)) for agent in range(env.n_agents)]
            belief_items = runner._build_belief_items_for_ego(ego)
            belief_summary = runner._belief_summary_np_from_items(belief_items)
            contexts = [
                runner._raw_proxy_context_excluding(
                    ego, source, current_actions=actions
                )
                for source in sources
            ]
            out = runner.proxy.score_batch_full(
                obs_i_batch=[env.get_obs_of_ego(obs_all, ego) for _ in sources],
                action_i_batch=[actions[ego] for _ in sources],
                observed_action_j_batch=[actions[source] for source in sources],
                z_core_excl_j_batch=[context[0] for context in contexts],
                m_periph_excl_j_batch=[context[1] for context in contexts],
                belief_summary_batch=[belief_summary for _ in sources],
                pair_feat_batch=[
                    build_pair_feat(
                        env.positions, env.agent_zone, env.grid_size, env.n_zones,
                        ego, source, agent_role=env.agent_role,
                    )
                    for source in sources
                ],
            )
            for source_index, source in enumerate(sources):
                oracle = _oracle_spectrum(
                    env, state, ego, source, horizon, trials,
                    seed=(int(seed) + 1) * 1000003 + state_index * 101 + source * 17,
                )
                learned = np.asarray(out["c_lag_mu"][source_index], dtype=np.float64)
                rows.append({
                    "seed": int(seed),
                    "state_index": int(state_index),
                    "ego": ego,
                    "source": source,
                    "true_delay": int(env.sgtp_delay_by_pair[(source, ego)]),
                    "oracle_spectrum": oracle.tolist(),
                    "learned_spectrum": learned.tolist(),
                    "oracle_center": _centre(oracle),
                    "learned_center": _centre(learned),
                })
    finally:
        runner.proxy.effect_mode = old_mode

    valid = [
        row for row in rows
        if row["oracle_center"] is not None and row["learned_center"] is not None
    ]
    truth = [row["true_delay"] for row in valid]
    learned_centres = [row["learned_center"] for row in valid]
    oracle_centres = [row["oracle_center"] for row in valid]
    delay_rank = _rank_correlation(truth, learned_centres)
    delay_mae = (
        float(np.mean(np.abs(np.asarray(truth) - np.asarray(learned_centres))))
        if valid else float("nan")
    )
    oracle_alignment = _rank_correlation(oracle_centres, learned_centres)
    zero_baseline_mae = (
        float(np.mean(np.abs(np.asarray(truth, dtype=np.float64))))
        if valid else float("nan")
    )
    terminal_baseline_mae = (
        float(np.mean(np.abs(
            np.asarray(truth, dtype=np.float64) - float(horizon - 1)
        )))
        if valid else float("nan")
    )
    gate_pass = bool(
        len(valid) >= 8 and delay_rank >= 0.50 and delay_mae <= 2.0
        and oracle_alignment >= 0.50
    )
    return {
        "protocol_version": "learned_latency_capacity_spectrum_v2",
        "gate_pass": gate_pass,
        "seed": int(seed),
        "train_episodes": int(train_episodes),
        "n_states": int(states),
        "horizon": int(horizon),
        "n_trials": int(trials),
        "evaluation_bank_seed": int(seed) + 9901,
        "evaluation_split": "held_out_state_bank",
        "n_valid": len(valid),
        "learned_delay_rank_correlation": delay_rank,
        "learned_delay_mae": delay_mae,
        "learned_oracle_center_rank_correlation": oracle_alignment,
        "zero_lag_baseline_mae": zero_baseline_mae,
        "terminal_lag_baseline_mae": terminal_baseline_mae,
        "learned_beats_zero_lag_baseline": bool(
            np.isfinite(delay_mae) and delay_mae < zero_baseline_mae
        ),
        "learned_beats_terminal_lag_baseline": bool(
            np.isfinite(delay_mae) and delay_mae < terminal_baseline_mae
        ),
        "rows": rows,
    }


def _bootstrap_mean_ci(values, seed=7193, n_bootstrap=10000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float("nan"), float("nan"), float("nan")]
    if values.size == 1:
        value = float(values[0])
        return [value, value, value]
    rng = np.random.RandomState(int(seed))
    draws = rng.choice(values, size=(int(n_bootstrap), values.size), replace=True)
    means = draws.mean(axis=1)
    return [
        float(np.mean(values)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def run_many(seeds, train_episodes, states, horizon, trials, device):
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("latency seeds must be a non-empty unique list")
    per_seed = [
        run(seed, train_episodes, states, horizon, trials, device)
        for seed in seeds
    ]
    metrics = {
        "learned_delay_rank_correlation": [
            result["learned_delay_rank_correlation"] for result in per_seed
        ],
        "learned_delay_mae": [result["learned_delay_mae"] for result in per_seed],
        "learned_oracle_center_rank_correlation": [
            result["learned_oracle_center_rank_correlation"] for result in per_seed
        ],
        "zero_lag_baseline_mae": [
            result["zero_lag_baseline_mae"] for result in per_seed
        ],
        "terminal_lag_baseline_mae": [
            result["terminal_lag_baseline_mae"] for result in per_seed
        ],
    }
    aggregate = {
        key: _bootstrap_mean_ci(values, seed=7193 + index)
        for index, (key, values) in enumerate(metrics.items())
    }
    pass_fraction = float(np.mean([result["gate_pass"] for result in per_seed]))
    learned_mae = aggregate["learned_delay_mae"][0]
    gate_pass = bool(
        len(seeds) >= 5
        and pass_fraction >= 0.8
        and aggregate["learned_delay_rank_correlation"][0] >= 0.50
        and aggregate["learned_oracle_center_rank_correlation"][0] >= 0.50
        and learned_mae <= 2.0
        and learned_mae < aggregate["zero_lag_baseline_mae"][0]
        and learned_mae < aggregate["terminal_lag_baseline_mae"][0]
    )
    rows = [row for result in per_seed for row in result["rows"]]
    return {
        "protocol_version": "learned_latency_capacity_spectrum_multiseed_v1",
        "gate_pass": gate_pass,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "minimum_confirmatory_seeds": 5,
        "per_seed_gate_pass_fraction": pass_fraction,
        "train_episodes_per_seed": int(train_episodes),
        "n_states_per_seed": int(states),
        "horizon": int(horizon),
        "n_trials": int(trials),
        "n_valid": int(sum(result["n_valid"] for result in per_seed)),
        "evaluation_split": "independent_held_out_state_bank_per_seed",
        "learned_delay_rank_correlation": aggregate[
            "learned_delay_rank_correlation"
        ][0],
        "learned_delay_mae": learned_mae,
        "learned_oracle_center_rank_correlation": aggregate[
            "learned_oracle_center_rank_correlation"
        ][0],
        "zero_lag_baseline_mae": aggregate["zero_lag_baseline_mae"][0],
        "terminal_lag_baseline_mae": aggregate["terminal_lag_baseline_mae"][0],
        "bootstrap_mean_ci95": aggregate,
        "per_seed": [
            {key: value for key, value in result.items() if key != "rows"}
            for result in per_seed
        ],
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(argv)
    if min(args.train_episodes, args.states, args.horizon, args.trials) <= 0:
        parser.error("all budgets must be positive")
    result = run_many(
        args.seeds, args.train_episodes, args.states, args.horizon,
        args.trials, args.device,
    )
    _atomic_json(os.path.abspath(args.json_out), result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
