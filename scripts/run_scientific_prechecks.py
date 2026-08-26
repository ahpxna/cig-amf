"""Run oracle/forced-only G0--G4 prechecks before expensive paper experiments.

These gates deliberately separate environment/estimand viability from learned
end-to-end performance.  Confirmatory mode is fail-closed; quick mode executes
the same wiring but reports SMOKE_ONLY rather than scientific support.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_experiment as RE
from scripts import run_h1_calibration as H1
from scripts import run_h2_selectivity as H2
from scripts import run_paper_b_allocation as PB
from scripts.exp_common import make_args
from envs.causal_adapter import resolve_env_adapter
from scripts.scientific_gate_common import (
    GATE_SCHEMA_VERSION, atomic_json, bootstrap_mean_ci, gate_record, load_json,
    wilson_interval,
)

PROTOCOL_VERSION = "scientific_prechecks_g0_g4_v3_disagreement_capture"
TRUE_NULL_C_TOLERANCE = 1e-10


def _mean(values):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _oracle_allocation_value_seed(
    *, seed, device, n_agents, episodes, max_steps, k, horizon, trials,
    core_refresh_every, final_window,
):
    """Run matched oracle-C, oracle-|D|, random-core and mean-field outcomes.

    The three core variants share the same policy/value architecture, parameter
    initialization seed, refresh cadence, and exact core budget.  PureMeanField
    is the separately preregistered structural-blind control.  This makes G0/G4
    outcome gates rather than score-capture tautologies.
    """
    rewards = {}
    models = ("OracleCore", "OracleAbsDCore", "RandomCore", "PureMeanField")
    for model in models:
        RE.set_global_seed(int(seed))
        env = RE.make_main_env(
            task_mode="S0B0", n_agents=int(n_agents), max_steps=int(max_steps),
            phase_length=max(40, int(episodes) + 5), seed=int(seed),
            structural_factor=False, behavioral_factor=False,
        )
        cfg = RE.default_cfg()
        cfg.update({
            "seed": int(seed),
            "seed_core_top_k": int(k),
            "causal_horizon": int(horizon),
            "oracle_n_trials": int(trials),
            "core_refresh_every": int(core_refresh_every),
            "freeze_policy_learning": False,
        })
        runner = RE.make_runner(model, env, cfg, device)
        history = runner.run(
            n_episodes=int(episodes),
            eval_every=max(1, min(5, int(episodes))),
        )
        reward_history = [
            float(value) for value in history.get("mean_reward", [])
            if math.isfinite(float(value))
        ]
        if not reward_history:
            raise RuntimeError(f"{model} produced no finite allocation-gate reward")
        window = max(1, min(int(final_window), len(reward_history)))
        rewards[model] = float(np.mean(reward_history[-window:]))
    return {
        "seed": int(seed),
        "k": int(k),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "final_window": int(final_window),
        "oracle_core_refresh_every": int(core_refresh_every),
        "reward": rewards,
        "oracle_C_minus_random": float(rewards["OracleCore"] - rewards["RandomCore"]),
        "oracle_C_minus_mean_field": float(rewards["OracleCore"] - rewards["PureMeanField"]),
        "oracle_C_minus_absD": float(rewards["OracleCore"] - rewards["OracleAbsDCore"]),
    }


def _oracle_c_d_bank(*, seed, n_agents, n_states, k, horizon, trials, discount):
    """Compute matched-k oracle C*/|D*| diagnostics without N-fold ego replay.

    Each forced action rollout already returns every ego's response.  Reusing
    that vector makes the precheck O(states * sources * actions) rather than
    O(states * egos * sources * actions), while preserving the exact C/D
    response surface and common-random-number pairing.
    """
    env = RE.make_main_env(
        task_mode="S0B0", n_agents=int(n_agents), max_steps=max(30, int(horizon) + 8),
        phase_length=40, seed=int(seed), structural_factor=False, behavioral_factor=False,
    )
    bank = env.sample_state_bank(
        n_states=int(n_states), burn_in=3, bank_seed=int(seed) + 8849,
        min_remaining_steps=int(horizon),
    )
    adapter = resolve_env_adapter(env)
    random_advantages = []
    c_vs_d_advantages = []
    disagreements = 0
    comparisons = 0
    for state_index, state in enumerate(bank):
        capacities = {ego: {} for ego in range(int(env.n_agents))}
        directions = {ego: {} for ego in range(int(env.n_agents))}
        for source in range(int(env.n_agents)):
            env.restore_state(copy.deepcopy(state))
            env.set_behaviour_override("cooperative")
            valid_actions = np.flatnonzero(adapter.valid_action_mask(source))
            if valid_actions.size < 2:
                continue
            pi = np.asarray(env.scripted_policy_distribution(source), dtype=np.float64)[valid_actions]
            pi = pi / np.clip(pi.sum(), 1e-12, None)
            q = np.full(valid_actions.size, 1.0 / valid_actions.size, dtype=np.float64)
            vectors = []
            crn_seed = (int(seed) + 1) * 1000003 + state_index * 1009 + source * 9176
            for action in valid_actions:
                env.restore_state(copy.deepcopy(state))
                env.set_behaviour_override("cooperative")
                vectors.append(np.asarray(
                    env.compute_oracle_influence_all_egos_from_current_state(
                        agent_j=int(source), intervention_action=int(action),
                        horizon=int(horizon), n_trials=int(trials), forced_step=0,
                        crn_seed=int(crn_seed),
                    ), dtype=np.float64,
                ))
            surface = np.stack(vectors, axis=0)  # [A, ego]
            c_vec = np.max(surface, axis=0) - np.min(surface, axis=0)
            d_vec = np.sum((pi - q)[:, None] * surface, axis=0)
            for ego in range(int(env.n_agents)):
                if ego == source:
                    continue
                capacities[ego][source] = max(0.0, float(c_vec[ego]))
                directions[ego][source] = float(d_vec[ego])

        for ego in sorted(capacities):
            c_row = capacities[ego]
            d_row = {int(j): abs(float(directions[ego][j])) for j in c_row}
            if len(c_row) < int(k):
                continue
            c_top = PB._top_k(c_row, ego, int(k))
            d_top = PB._top_k(d_row, ego, int(k))
            c_best = float(sum(c_row[j] for j in c_top))
            random_expected = float(int(k) * np.mean(list(c_row.values())))
            scale = max(abs(c_best), 1e-12)
            random_advantages.append((c_best - random_expected) / scale)
            comparisons += 1
            if c_top != d_top:
                disagreements += 1
                d_capture = float(sum(c_row[j] for j in d_top))
                c_vs_d_advantages.append((c_best - d_capture) / scale)
    if comparisons == 0:
        raise RuntimeError("oracle precheck produced no matched-k comparisons")
    return {
        "seed": int(seed),
        "random_advantage_mean": _mean(random_advantages),
        "c_vs_absd_advantage_mean": _mean(c_vs_d_advantages),
        # Defined only on C/D-disagreement cases. This is the direct selector
        # quality endpoint; end-to-end reward remains its practical outcome.
        "c_capture_advantage_on_disagreement_mean": _mean(c_vs_d_advantages),
        "disagreement_fraction": float(disagreements / comparisons),
        "comparison_count": int(comparisons),
        "disagreement_count": int(disagreements),
    }


def _forced_only_h1(*, seed, device, out_root, episodes, states, max_steps, epsilon, thresholds):
    cfg = H1.build_h1_config(
        {"proxy_use_doubly_robust": False, "eps": float(epsilon)}, int(seed), thresholds,
    )
    cfg["proxy_forced_only_training"] = True
    cfg["forcer_anneal_to"] = float(epsilon)
    cfg["freeze_policy_learning"] = True
    args = make_args(
        seed=int(seed), device=device, result_dir=str(out_root), tiny_horizon=1,
        tiny_proxy_train_episodes=max(1, int(episodes)), tiny_states=max(1, int(states)),
        max_steps=max(1, int(max_steps)),
    )
    args.h1_exact_protocol = True
    args.h1_diagnostic_only = True
    seed_root = Path(out_root) / f"seed_{int(seed)}"
    summary = RE.run_tiny_task(
        args, cfg, device, out_dir=str(seed_root), run_label=f"g1_forced_only_seed{seed}",
    )
    pair_path = seed_root / "tiny_oracle_pair_rows.csv"
    with pair_path.open(newline="", encoding="utf-8") as handle:
        pair_rows = list(csv.DictReader(handle))
    prediction_threshold = float(cfg.get("h1_capacity_prediction_threshold", 0.01))
    true_null = [
        row for row in pair_rows
        if abs(float(row["oracle_score"])) <= TRUE_NULL_C_TOLERANCE
    ]
    false_positives = sum(
        float(row["learned_score"]) > prediction_threshold for row in true_null
    )
    coverage = summary.get("action_coverage", {})
    forced_actions_seen = int(coverage.get("forced_actions_seen", 0))
    proxy_action_count = int(summary.get("proxy_action_count", 0))
    return {
        "seed": int(seed),
        "q_spearman": float(summary.get("q_within_state_action_spearman_mean", float("nan"))),
        "q_normalized_rmse": float(summary.get("q_normalized_rmse_mean", float("nan"))),
        "q_nonconstant_surfaces": float(summary.get("q_nonconstant_surface_count_mean", 0.0)),
        "true_null_C_tolerance": TRUE_NULL_C_TOLERANCE,
        "true_null_C_count": int(len(true_null)),
        "true_null_C_false_positives": int(false_positives),
        "true_null_C_fpr": (
            float(false_positives / len(true_null)) if true_null else float("nan")
        ),
        "capacity_prediction_threshold": prediction_threshold,
        "forced_action_coverage_gate_pass": bool(
            proxy_action_count > 0 and forced_actions_seen == proxy_action_count
        ),
        "forced_actions_seen": forced_actions_seen,
        "proxy_action_count": proxy_action_count,
        "proxy_forced_only_training": bool(summary.get("proxy_forced_only_training", False)),
        "realised_forcing_rate": float(summary.get("realised_forcing_rate", 0.0)),
        "n_forced_proxy_samples": int(summary.get("n_forced_proxy_samples", 0)),
        "protocol_gate_pass": bool(summary.get("protocol_gate_pass", False)),
    }


def _g3_seed(*, seed, n_agents, states, horizon, trials, discount):
    env = RE.make_main_env(
        task_mode="S0B0", n_agents=int(n_agents), max_steps=max(30, int(horizon) + 8),
        phase_length=40, seed=int(seed), structural_factor=False, behavioral_factor=False,
    )
    cells = {
        mode: H2._fixed_estimand_panel(
            env, structural_factor=structural, behavioral_factor=behavioral,
            seed=int(seed), n_states=int(states), horizon=int(horizon),
            discount=float(discount), n_trials=int(trials),
        )
        for mode, (structural, behavioral) in H2.FACTORIAL_CELLS.items()
    }
    if not all(bool(row.get("applicable")) for row in cells.values()):
        raise RuntimeError("fixed-rho oracle panel is not applicable in every factorial cell")
    c = {mode: float(row["capacity_mean"]) for mode, row in cells.items()}
    d = {mode: float(row["direction_abs_mean"]) for mode, row in cells.items()}
    beta_sc = 0.5 * ((c["S1B0"] - c["S0B0"]) + (c["S1B1"] - c["S0B1"]))
    beta_bc = 0.5 * ((c["S0B1"] - c["S0B0"]) + (c["S1B1"] - c["S1B0"]))
    beta_bd = 0.5 * ((d["S0B1"] - d["S0B0"]) + (d["S1B1"] - d["S1B0"]))
    return {
        "seed": int(seed),
        "beta_structural_C": float(beta_sc),
        "beta_behavioral_C": float(beta_bc),
        "beta_behavioral_absD": float(beta_bd),
        "C_separation": float(beta_sc - abs(beta_bc)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment0", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--protocol-mode", choices=("quick", "confirmatory"), default="confirmatory")
    ap.add_argument("--threshold-calibration")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-root", default=None)
    ap.add_argument("--n-agents", type=int, default=24)
    ap.add_argument("--oracle-states", type=int, default=4)
    ap.add_argument("--oracle-horizon", type=int, default=4)
    ap.add_argument("--oracle-trials", type=int, default=1)
    ap.add_argument("--core-k", type=int, default=3)
    ap.add_argument("--allocation-episodes", type=int, default=40)
    ap.add_argument("--allocation-max-steps", type=int, default=30)
    ap.add_argument("--allocation-core-refresh-every", type=int, default=5)
    ap.add_argument("--allocation-final-window", type=int, default=10)
    ap.add_argument("--h1-train-episodes", type=int, default=30)
    ap.add_argument("--h1-states", type=int, default=16)
    ap.add_argument("--h1-max-steps", type=int, default=30)
    ap.add_argument("--forced-epsilon", type=float, default=0.20)
    ap.add_argument("--g3-states", type=int, default=3)
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--min-c-d-disagreement", type=float, default=0.10)
    ap.add_argument("--min-c-d-disagreement-comparisons", type=int, default=5)
    ap.add_argument("--min-null-pairs", type=int, default=20)
    args = ap.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("--seeds must be unique")
    if args.protocol_mode == "confirmatory" and len(args.seeds) < int(args.min_seeds):
        ap.error("confirmatory prechecks require at least --min-seeds unique seeds")
    if args.protocol_mode == "confirmatory" and not args.threshold_calibration:
        ap.error("confirmatory G1/G2 require --threshold-calibration")
    if args.core_k <= 0 or args.core_k >= args.n_agents:
        ap.error("--core-k must be between 1 and n-agents-1")
    if args.min_c_d_disagreement_comparisons <= 0:
        ap.error("--min-c-d-disagreement-comparisons must be positive")
    if (
        args.allocation_episodes <= 0 or args.allocation_max_steps <= 0
        or args.allocation_core_refresh_every <= 0 or args.allocation_final_window <= 0
    ):
        ap.error("allocation-gate budgets must be positive integers")

    exp0 = load_json(args.experiment0, "Experiment 0")
    exp0_pass = bool(exp0.get("required_gate_pass", False))
    thresholds = None
    threshold_fingerprint = None
    if args.threshold_calibration:
        calibration_path = os.path.abspath(args.threshold_calibration)
        thresholds, threshold_fingerprint = H1._load_threshold_calibration(calibration_path)
        with open(calibration_path, encoding="utf-8") as handle:
            calibration_payload = json.load(handle)
        development_seeds = H1._development_seed_set(
            calibration_payload, "H1 threshold calibration"
        )
        overlap = development_seeds.intersection(int(seed) for seed in args.seeds)
        if overlap and args.protocol_mode == "confirmatory":
            ap.error(
                "G0-G4 precheck seeds overlap H1 threshold-development seeds: "
                + ", ".join(str(seed) for seed in sorted(overlap))
            )

    work_root = Path(args.work_root or (str(Path(args.out).with_suffix("")) + "_work"))
    work_root.mkdir(parents=True, exist_ok=True)

    oracle_rows = [
        _oracle_c_d_bank(
            seed=seed, n_agents=args.n_agents, n_states=args.oracle_states,
            k=args.core_k, horizon=args.oracle_horizon, trials=args.oracle_trials,
            discount=0.97,
        ) for seed in args.seeds
    ]
    allocation_rows = [
        _oracle_allocation_value_seed(
            seed=seed, device=args.device, n_agents=args.n_agents,
            episodes=args.allocation_episodes, max_steps=args.allocation_max_steps,
            k=args.core_k, horizon=args.oracle_horizon, trials=args.oracle_trials,
            core_refresh_every=args.allocation_core_refresh_every,
            final_window=args.allocation_final_window,
        ) for seed in args.seeds
    ]
    h1_rows = [
        _forced_only_h1(
            seed=seed, device=args.device, out_root=work_root / "forced_only_h1",
            episodes=args.h1_train_episodes, states=args.h1_states,
            max_steps=args.h1_max_steps, epsilon=args.forced_epsilon,
            thresholds=thresholds,
        ) for seed in args.seeds
    ]
    g3_rows = [
        _g3_seed(
            seed=seed, n_agents=args.n_agents, states=args.g3_states,
            horizon=args.oracle_horizon, trials=args.oracle_trials, discount=0.97,
        ) for seed in args.seeds
    ]

    scientific = args.protocol_mode == "confirmatory"
    c_vs_random = [row["oracle_C_minus_random"] for row in allocation_rows]
    c_vs_mean_field = [row["oracle_C_minus_mean_field"] for row in allocation_rows]
    c_random_ci = bootstrap_mean_ci(c_vs_random, seed=4100)
    c_mean_field_ci = bootstrap_mean_ci(c_vs_mean_field, seed=4101)
    g0_pass = bool(
        exp0_pass and scientific
        and math.isfinite(c_random_ci[0]) and c_random_ci[0] > 0.0
        and math.isfinite(c_mean_field_ci[0]) and c_mean_field_ci[0] > 0.0
    )
    g0 = gate_record(
        "G0", g0_pass, required=True,
        metrics={
            "experiment0_structure_value_gate_pass": exp0_pass,
            "oracle_C_minus_random_reward_by_seed": c_vs_random,
            "oracle_C_minus_random_reward_ci95": c_random_ci,
            "oracle_C_minus_PureMeanField_reward_by_seed": c_vs_mean_field,
            "oracle_C_minus_PureMeanField_reward_ci95": c_mean_field_ci,
            "matched_core_k": int(args.core_k),
            "allocation_episodes": int(args.allocation_episodes),
            "allocation_final_window": int(args.allocation_final_window),
            "oracle_core_refresh_every": int(args.allocation_core_refresh_every),
            "oracle_scope": "periodically_refreshed_operational_oracle",
        },
        rule=(
            "the periodically refreshed operational oracle C*-core must beat a matched-k RandomCore and the structural-blind "
            "PureMeanField control on paired final-window reward, with positive seed-level "
            "bootstrap lower bounds; Experiment0 must independently establish non-zero "
            "structure value"
        ),
        failure_action="redesign environment or stop Paper-B allocation claim",
    )

    q_spearman = [row["q_spearman"] for row in h1_rows]
    q_rmse = [row["q_normalized_rmse"] for row in h1_rows]
    q_ci = bootstrap_mean_ci(q_spearman, seed=4200)
    g1_pass = bool(
        scientific and all(row["protocol_gate_pass"] for row in h1_rows)
        and all(row["proxy_forced_only_training"] for row in h1_rows)
        and all(row["forced_action_coverage_gate_pass"] for row in h1_rows)
        and all(row["realised_forcing_rate"] > 0.0 for row in h1_rows)
        and all(row["n_forced_proxy_samples"] > 0 for row in h1_rows)
        and _mean(q_spearman) >= H1.MIN_Q_SPEARMAN
        and q_ci[0] > 0.0
        and _mean(q_rmse) <= H1.MAX_Q_NORMALIZED_RMSE
        and sum(row["q_nonconstant_surfaces"] for row in h1_rows) > 0
    )
    g1 = gate_record(
        "G1", g1_pass, required=True,
        metrics={
            "forced_only": True, "epsilon": float(args.forced_epsilon),
            "q_spearman_by_seed": q_spearman, "q_spearman_mean": _mean(q_spearman),
            "q_spearman_ci95": q_ci, "q_normalized_rmse_mean": _mean(q_rmse),
            "forced_only_training_all_seeds": all(row["proxy_forced_only_training"] for row in h1_rows),
            "forced_action_coverage_all_seeds": all(row["forced_action_coverage_gate_pass"] for row in h1_rows),
            "forced_actions_seen_by_seed": [row["forced_actions_seen"] for row in h1_rows],
            "forced_proxy_samples_by_seed": [row["n_forced_proxy_samples"] for row in h1_rows],
            "forcing_positive_all_seeds": all(row["realised_forcing_rate"] > 0 for row in h1_rows),
        },
        rule=f"randomized/forced-only Q recovery requires a forced-only nuisance sampler, forced-subset support for every action head, mean within-state rank >= {H1.MIN_Q_SPEARMAN}, positive lower CI, and normalized RMSE <= {H1.MAX_Q_NORMALIZED_RMSE}",
        failure_action="fix Q estimator/support before tuning C, D, routing, or downstream architecture",
    )

    true_null_count = int(sum(row["true_null_C_count"] for row in h1_rows))
    true_null_fp = int(sum(row["true_null_C_false_positives"] for row in h1_rows))
    true_null_fpr = (
        float(true_null_fp / true_null_count) if true_null_count > 0 else float("nan")
    )
    null_ci = wilson_interval(true_null_fp, true_null_count)
    g2_pass = bool(
        scientific and true_null_count >= int(args.min_null_pairs)
        and math.isfinite(null_ci[1]) and null_ci[1] <= H1.MAX_CAPACITY_NULL_FPR
    )
    g2 = gate_record(
        "G2", g2_pass, required=True,
        metrics={
            "true_null_C_tolerance": TRUE_NULL_C_TOLERANCE,
            "true_null_C_false_positives": true_null_fp,
            "true_null_C_pair_count": true_null_count,
            "true_null_C_fpr": true_null_fpr,
            "true_null_C_fpr_wilson95": null_ci,
            "capacity_prediction_threshold_by_seed": [
                row["capacity_prediction_threshold"] for row in h1_rows
            ],
            "max_fpr": H1.MAX_CAPACITY_NULL_FPR,
            "min_null_pairs": int(args.min_null_pairs),
        },
        rule=(
            "among held-out pairs with oracle C* numerically equal to zero, the 95% "
            "Wilson upper bound of Pr(C-hat > frozen prediction threshold) must not "
            "exceed the preregistered null-FPR limit"
        ),
        failure_action="fix max-min estimator bias/noise before testing real structural capacity",
    )

    c_sep = [row["C_separation"] for row in g3_rows]
    b_d = [row["beta_behavioral_absD"] for row in g3_rows]
    c_sep_ci = bootstrap_mean_ci(c_sep, seed=4400)
    b_d_ci = bootstrap_mean_ci(b_d, seed=4401)
    g3_pass = bool(scientific and c_sep_ci[0] > 0.0 and b_d_ci[0] > 0.0)
    g3 = gate_record(
        "G3", g3_pass, required=True,
        metrics={
            "seed_rows": g3_rows,
            "beta_structural_C_minus_abs_beta_behavioral_C_ci95": c_sep_ci,
            "beta_behavioral_absD_ci95": b_d_ci,
        },
        rule="under fixed rho, structural manipulation must move C more than behavioral manipulation in magnitude, while behavioral manipulation must move |D| positively",
        failure_action="redefine the structural-C claim before learned estimation",
    )

    disagreement = [row["disagreement_fraction"] for row in oracle_rows]
    disagreement_count = int(sum(row["disagreement_count"] for row in oracle_rows))
    c_capture = [row["c_capture_advantage_on_disagreement_mean"] for row in oracle_rows]
    c_capture_ci = bootstrap_mean_ci(c_capture, seed=4501)
    c_vs_d_reward = [row["oracle_C_minus_absD"] for row in allocation_rows]
    c_vs_d_reward_ci = bootstrap_mean_ci(c_vs_d_reward, seed=4500)
    g4_pass = bool(
        scientific and _mean(disagreement) >= float(args.min_c_d_disagreement)
        and disagreement_count >= int(args.min_c_d_disagreement_comparisons)
        and math.isfinite(c_capture_ci[0]) and c_capture_ci[0] > 0.0
        and math.isfinite(c_vs_d_reward_ci[0]) and c_vs_d_reward_ci[0] > 0.0
    )
    g4 = gate_record(
        "G4", g4_pass, required=True,
        metrics={
            "oracle_C_minus_oracle_absD_reward_by_seed": c_vs_d_reward,
            "reward_ci95": c_vs_d_reward_ci,
            "disagreement_fraction_by_seed": disagreement,
            "disagreement_fraction_mean": _mean(disagreement),
            "min_disagreement": float(args.min_c_d_disagreement),
            "disagreement_comparison_count": disagreement_count,
            "min_disagreement_comparisons": int(args.min_c_d_disagreement_comparisons),
            "C_capture_advantage_on_disagreement_by_seed": c_capture,
            "C_capture_advantage_on_disagreement_ci95": c_capture_ci,
            "matched_core_k": int(args.core_k),
        },
        rule=(
            "on oracle-disagreement states and matched k, C*-top-k must capture "
            "more oracle C* than |D*|-top-k with a positive paired lower CI; the "
            "operational oracle-C runner must also deliver a positive paired "
            "final-window reward advantage over the oracle-|D*| runner"
        ),
        failure_action=(
            "do not claim C and D have distinct allocation roles; redefine the "
            "benchmark/selector claim before learned Paper-B experiments"
        ),
    )

    gates = {record["gate"]: record for record in (g0, g1, g2, g3, g4)}
    core_pass = bool(scientific and all(record["passed"] for record in gates.values()))
    payload = {
        "schema_version": GATE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_mode": args.protocol_mode,
        "seeds": [int(seed) for seed in args.seeds],
        "threshold_calibration_fingerprint": threshold_fingerprint,
        "gates": gates,
        "core_prechecks_pass": core_pass,
        "overall_status": "PASS" if core_pass else ("SMOKE_ONLY" if not scientific else "FAIL"),
        "oracle_rows": oracle_rows,
        "oracle_allocation_rows": allocation_rows,
        "forced_only_h1_rows": h1_rows,
        "fixed_rho_rows": g3_rows,
    }
    atomic_json(args.out, payload)
    print(json.dumps({
        "overall_status": payload["overall_status"],
        "core_prechecks_pass": core_pass,
        "gate_pass": {name: row["passed"] for name, row in gates.items()},
        "out": os.path.abspath(args.out),
    }, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
