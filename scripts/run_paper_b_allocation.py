"""Run Paper-B selector-isolation and end-to-end allocation panels.

The central test is deliberately narrow: ``C`` and ``|D|`` select the same
number of explicit neighbours under the same seed, architecture, and training
budget.  C is the structural-capacity selector; |D| is the behavioural
direction confounder.  This runner does not treat the result as a causal
identification test—that remains Paper A/H1.
"""

import argparse
import copy
import csv
import json
import os
import tempfile

import numpy as np

try:
    from exp_common import ROOT, ensure_dir
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, ensure_dir

import run_experiment as RE
try:
    from run_h2_selectivity import (
        _capture_frozen_learning_checkpoint,
        _restore_frozen_learning_checkpoint,
    )
except ModuleNotFoundError:
    from scripts.run_h2_selectivity import (
        _capture_frozen_learning_checkpoint,
        _restore_frozen_learning_checkpoint,
    )


VARIANTS = {
    "C-Core": "structural_capacity",
    "AbsD-Core": "behavioral_direction",
    "Random-Core": "random",
    "Correlation-Core": "observational_correlation",
    "Oracle-C-Core": "oracle_capacity",
    "Full-Explicit": "full_explicit",
}


def _atomic_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".tmp-paper-b-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _mean(values):
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _cfg(seed, core_budget, selector="structural_capacity"):
    cfg = RE.default_cfg()
    cfg.update({
        "seed": int(seed),
        "k0_warmup": 0,
        "core_selection_mode": str(selector),
        "belief_adaptive_k": False,
        "min_core_size": int(core_budget),
        "belief_adaptive_k_min": int(core_budget),
        "max_core_size": int(core_budget),
        "periph_require_full_signature": True,
        "periph_allow_legacy_items": False,
    })
    return cfg


def _make_env(seed, phase_length=40):
    return RE.make_main_env(
        task_mode="behavioral_drift",
        n_agents=24,
        max_steps=30,
        phase_length=int(phase_length),
        seed=int(seed),
    )


def _prepare_variant_runner(variant, seed, core_budget, device, checkpoint):
    selector = VARIANTS[variant]
    cfg = _cfg(seed, core_budget, selector=selector)
    env = _make_env(seed)
    runner = RE.make_runner("Final-CIGAMF", env, cfg, device)
    _restore_frozen_learning_checkpoint(runner, checkpoint)
    if selector == "full_explicit":
        for belief in runner.belief_modules.values():
            belief.max_core_size = len(belief.neighbor_ids)
            belief.set_fixed_core(belief.neighbor_ids)
    return runner


def _top_k(scores, ego, core_budget):
    candidates = [int(j) for j in scores if int(j) != int(ego)]
    return set(sorted(
        candidates, key=lambda j: float(scores[j]), reverse=True
    )[:int(core_budget)])


def _oracle_capacity_for_state(
    env, state, horizon, discount, trials, seed
):
    """Compute clone-state all-action C* for every directed pair."""
    outer = env.clone_state()
    result = {}
    try:
        for ego in range(int(env.n_agents)):
            row = {}
            for source in range(int(env.n_agents)):
                if source == ego:
                    continue
                pair_seed = (
                    (int(seed) + 1) * 1000003
                    + int(ego) * 1009 + int(source) * 9176
                )
                action_values = []
                for action in range(int(env.get_action_dim())):
                    env.restore_state(copy.deepcopy(state))
                    env.set_behaviour_override("cooperative")
                    response = env.compute_oracle_lag_response_from_current_state(
                        ego_id=ego,
                        agent_j=source,
                        intervention_action=action,
                        horizon=int(horizon),
                        n_trials=int(trials),
                        forced_step=0,
                        continuation_policy=env.scripted_policy,
                        crn_seed=pair_seed,
                        discount=float(discount),
                    )
                    action_values.append(float(response["discounted_response"]))
                row[source] = float(max(action_values) - min(action_values))
            result[ego] = row
    finally:
        env.restore_state(outer)
    return result


def _mean_oracle_capacity(oracle_bank):
    if not oracle_bank:
        raise ValueError("oracle capacity bank is empty")
    output = {}
    for ego in oracle_bank[0]:
        output[ego] = {
            source: float(np.mean([
                bank[ego][source] for bank in oracle_bank
            ]))
            for source in oracle_bank[0][ego]
        }
    return output


def _f1(predicted, truth):
    denom = len(predicted) + len(truth)
    return 2.0 * len(predicted & truth) / denom if denom else 1.0


def _capacity_ndcg_at_k(predicted, oracle_scores, core_budget):
    gains = sorted(
        (max(0.0, float(oracle_scores[j])) for j in predicted), reverse=True
    )[:int(core_budget)]
    ideal = sorted(
        (max(0.0, float(value)) for value in oracle_scores.values()),
        reverse=True,
    )[:int(core_budget)]
    dcg = sum(gain / np.log2(index + 2.0) for index, gain in enumerate(gains))
    idcg = sum(gain / np.log2(index + 2.0) for index, gain in enumerate(ideal))
    return float(dcg / idcg) if idcg > 1e-12 else 1.0


def _pretrain_checkpoint(seed, episodes, core_budget, device):
    RE.set_global_seed(seed)
    cfg = _cfg(seed, core_budget)
    env = _make_env(seed, phase_length=max(100000, int(episodes) + 1))
    runner = RE.make_runner("Final-CIGAMF", env, cfg, device)
    runner.run(n_episodes=int(episodes), eval_every=max(1, int(episodes)))
    checkpoint = _capture_frozen_learning_checkpoint(runner)
    checkpoint["episodes"] = int(episodes)
    return checkpoint


def _isolation_rows(
    variants, seed, core_budget, device, checkpoint, state_bank, oracle_bank
):
    records = {variant: [] for variant in variants}
    for variant in variants:
        runner = _prepare_variant_runner(
            variant, seed, core_budget, device, checkpoint
        )
        runner.cfg["freeze_policy_learning"] = True
        runner.cfg["freeze_representation_state"] = True
        for state_index, (state, oracle_scores) in enumerate(
            zip(state_bank, oracle_bank)
        ):
            _restore_frozen_learning_checkpoint(runner, checkpoint)
            runner.env.restore_state(copy.deepcopy(state))
            runner.env.set_behaviour_override("cooperative")
            runner.oracle_capacity_scores_by_ego = copy.deepcopy(oracle_scores)
            if VARIANTS[variant] == "full_explicit":
                for belief in runner.belief_modules.values():
                    belief.max_core_size = len(belief.neighbor_ids)
                    belief.set_fixed_core(belief.neighbor_ids)
            obs_all = runner.env._get_obs_all()
            action_dict, cache = runner._select_actions_population(obs_all)
            actions = [int(action_dict[agent]) for agent in range(runner.n_agents)]
            runner._score_all_pairs_and_update_beliefs(
                obs_all=obs_all,
                actions=actions,
                behaviour_probs=cache.get("behaviour_probs"),
                policy_probs=cache.get("policy_probs"),
            )
            for ego, belief in runner.belief_modules.items():
                predicted = set(belief.get_core_set())
                truth = _top_k(oracle_scores[int(ego)], ego, core_budget)
                records[variant].append({
                    "key": (int(state_index), int(ego)),
                    "predicted": predicted,
                    "truth": truth,
                    "f1": _f1(predicted, truth),
                    "ndcg": _capacity_ndcg_at_k(
                        predicted, oracle_scores[int(ego)], core_budget
                    ),
                    "core_size": len(predicted),
                })

    disagreement_keys = set()
    if "C-Core" in records and "AbsD-Core" in records:
        c_by_key = {row["key"]: row["predicted"] for row in records["C-Core"]}
        d_by_key = {row["key"]: row["predicted"] for row in records["AbsD-Core"]}
        disagreement_keys = {
            key for key in c_by_key if c_by_key[key] != d_by_key[key]
        }
    total_keys = len(records.get("C-Core", next(iter(records.values()), [])))
    rows = []
    for variant in variants:
        values = records[variant]
        selector = VARIANTS[variant]
        core_sizes = [row["core_size"] for row in values]
        if selector != "full_explicit" and any(
            size != int(core_budget) for size in core_sizes
        ):
            raise RuntimeError(
                f"{variant}/seed={seed} violated isolation budget: {core_sizes}"
            )
        disagreement = [
            row["f1"] for row in values if row["key"] in disagreement_keys
        ]
        rows.append({
            "panel": "selector_isolation",
            "variant": variant,
            "selector": selector,
            "seed": int(seed),
            "episodes": 0,
            "pretrain_episodes": int(checkpoint["episodes"]),
            "checkpoint_sha256": checkpoint["sha256"],
            "core_budget": int(core_budget),
            "matched_budget": int(selector != "full_explicit"),
            "strict_5d_signature": True,
            "selector_state_count": int(len(state_bank)),
            "selector_pair_state_count": int(len(values)),
            "oracle_reference": "clone_state_all_action_fixed_rho_C",
            "mean_core_size": _mean(core_sizes),
            "selector_oracle_f1": _mean([row["f1"] for row in values]),
            "selector_oracle_ndcg_at_k": _mean([row["ndcg"] for row in values]),
            "disagreement_pair_state_count": int(len(disagreement_keys)),
            "disagreement_pair_state_fraction": float(
                len(disagreement_keys) / max(1, total_keys)
            ),
            "disagreement_selector_oracle_f1": _mean(disagreement),
            "mean_reward": float("nan"),
            "final_reward": float("nan"),
            "mean_f1": float("nan"),
            "throughput_total": float("nan"),
        })
    return rows


def _end_to_end_row(
    variant, seed, episodes, core_budget, device, checkpoint,
    mean_oracle_capacity,
):
    runner = _prepare_variant_runner(
        variant, seed, core_budget, device, checkpoint
    )
    if VARIANTS[variant] == "oracle_capacity":
        runner.oracle_capacity_scores_by_ego = copy.deepcopy(mean_oracle_capacity)
    history = runner.run(n_episodes=int(episodes), eval_every=10)
    core_sizes = [len(module.get_core_set()) for module in runner.belief_modules.values()]
    selector = VARIANTS[variant]
    if selector != "full_explicit" and any(size != int(core_budget) for size in core_sizes):
        raise RuntimeError(
            f"{variant}/seed={seed} violated matched budget: {core_sizes}"
        )
    return {
        "panel": "end_to_end",
        "variant": variant,
        "selector": selector,
        "seed": int(seed),
        "episodes": int(episodes),
        "pretrain_episodes": int(checkpoint["episodes"]),
        "checkpoint_sha256": checkpoint["sha256"],
        "core_budget": int(core_budget),
        "matched_budget": int(selector != "full_explicit"),
        "strict_5d_signature": True,
        "mean_core_size": _mean(history.get("mean_core_size", [])),
        "selector_oracle_f1": float("nan"),
        "oracle_reference": (
            "clone_state_bank_mean_C"
            if VARIANTS[variant] == "oracle_capacity" else "not_applicable"
        ),
        "mean_reward": _mean(history.get("mean_reward", [])),
        "final_reward": float(history.get("mean_reward", [float("nan")])[-1]),
        "mean_f1": _mean(history.get("mean_f1", [])),
        "throughput_total": _mean(history.get("throughput_total_agent_steps_per_sec", [])),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--pretrain-episodes", type=int, default=60)
    parser.add_argument("--core-budget", type=int, default=3)
    parser.add_argument("--selector-states", type=int, default=8)
    parser.add_argument("--oracle-trials", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None)
    parser.add_argument("--out-root", default=os.path.join(ROOT, "results", "paper_b_allocation"))
    args = parser.parse_args(argv)
    if (
        args.episodes <= 0 or args.pretrain_episodes <= 0
        or args.selector_states <= 0 or args.oracle_trials <= 0
        or args.core_budget <= 0 or not args.seeds
    ):
        parser.error(
            "episodes, pretrain episodes, core budget, and seeds must be positive"
        )
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    out_root = ensure_dir(os.path.abspath(args.out_root))
    rows = []
    selected_variants = args.variants or list(VARIANTS)
    for seed in args.seeds:
        checkpoint = _pretrain_checkpoint(
            seed, args.pretrain_episodes, args.core_budget, args.device
        )
        reference_runner = _prepare_variant_runner(
            "C-Core", seed, args.core_budget, args.device, checkpoint
        )
        reference_runner.env.set_behaviour_override("cooperative")
        bank_seed = int(seed) + 88001
        state_bank = reference_runner.env.sample_state_bank(
            n_states=int(args.selector_states), burn_in=3, bank_seed=bank_seed
        )
        horizon = int(reference_runner.cfg["causal_horizon"])
        discount = float(reference_runner.cfg["discount"])
        oracle_bank = [
            _oracle_capacity_for_state(
                reference_runner.env,
                state,
                horizon=horizon,
                discount=discount,
                trials=args.oracle_trials,
                seed=bank_seed + state_index * 100003,
            )
            for state_index, state in enumerate(state_bank)
        ]
        mean_oracle_capacity = _mean_oracle_capacity(oracle_bank)
        rows.extend(_isolation_rows(
            selected_variants,
            seed,
            args.core_budget,
            args.device,
            checkpoint,
            state_bank,
            oracle_bank,
        ))
        for variant in selected_variants:
            rows.append(_end_to_end_row(
                variant, seed, args.episodes, args.core_budget, args.device,
                checkpoint, mean_oracle_capacity,
            ))
    path = os.path.join(out_root, "summary_paper_b_allocation.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(os.path.join(out_root, "manifest.json"), {
        "experiment": "paper_b_c_vs_absd_allocation",
        "complete": True,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "pretrain_episodes": args.pretrain_episodes,
        "core_budget": args.core_budget,
        "selector_states": args.selector_states,
        "oracle_trials": args.oracle_trials,
        "selector_bank_seed_rule": "training_seed_plus_88001",
        "selector_bank_held_out": True,
        "variants": {name: VARIANTS[name] for name in selected_variants},
        "panels": ["selector_isolation", "end_to_end"],
        "summary": path,
    })
    print(path)


if __name__ == "__main__":
    main()
