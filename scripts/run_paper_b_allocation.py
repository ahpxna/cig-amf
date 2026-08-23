"""Run Paper-B selector-isolation and end-to-end allocation panels.

The central test is deliberately narrow: ``C`` and ``|D|`` select the same
number of explicit neighbours under the same seed, architecture, and training
budget.  C is the structural-capacity selector; |D| is the behavioural
direction confounder.  This runner does not treat the result as a causal
identification test—that remains Paper A/H1.
"""

import argparse
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


def _oracle_core(env, ego, core_budget):
    row = getattr(env, "gt_influence_by_ego", {}).get(int(ego), {})
    ranked = sorted(
        (int(j) for j in row if int(j) != int(ego)),
        key=lambda j: abs(float(row[j])),
        reverse=True,
    )
    return set(ranked[:int(core_budget)])


def _selector_f1(runner, core_budget):
    scores = []
    for ego, belief in runner.belief_modules.items():
        predicted = belief.get_core_set()
        truth = _oracle_core(runner.env, ego, core_budget)
        denom = len(predicted) + len(truth)
        scores.append(2.0 * len(predicted & truth) / denom if denom else 1.0)
    return _mean(scores)


def _pretrain_checkpoint(seed, episodes, core_budget, device):
    RE.set_global_seed(seed)
    cfg = _cfg(seed, core_budget)
    env = _make_env(seed, phase_length=max(100000, int(episodes) + 1))
    runner = RE.make_runner("Final-CIGAMF", env, cfg, device)
    runner.run(n_episodes=int(episodes), eval_every=max(1, int(episodes)))
    checkpoint = _capture_frozen_learning_checkpoint(runner)
    checkpoint["episodes"] = int(episodes)
    return checkpoint


def _isolation_row(variant, seed, core_budget, device, checkpoint):
    runner = _prepare_variant_runner(
        variant, seed, core_budget, device, checkpoint
    )
    runner.cfg["freeze_policy_learning"] = True
    runner.cfg["freeze_representation_state"] = True
    obs_all = runner.env._get_obs_all()
    action_dict, cache = runner._select_actions_population(obs_all)
    actions = [int(action_dict[agent]) for agent in range(runner.n_agents)]
    runner._score_all_pairs_and_update_beliefs(
        obs_all=obs_all,
        actions=actions,
        behaviour_probs=cache.get("behaviour_probs"),
        policy_probs=cache.get("policy_probs"),
    )
    core_sizes = [
        len(module.get_core_set()) for module in runner.belief_modules.values()
    ]
    selector = VARIANTS[variant]
    if selector != "full_explicit" and any(
        size != int(core_budget) for size in core_sizes
    ):
        raise RuntimeError(
            f"{variant}/seed={seed} violated isolation budget: {core_sizes}"
        )
    return {
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
        "mean_core_size": _mean(core_sizes),
        "selector_oracle_f1": _selector_f1(runner, core_budget),
        "mean_reward": float("nan"),
        "final_reward": float("nan"),
        "mean_f1": float("nan"),
        "throughput_total": float("nan"),
    }


def _end_to_end_row(variant, seed, episodes, core_budget, device, checkpoint):
    runner = _prepare_variant_runner(
        variant, seed, core_budget, device, checkpoint
    )
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
        "selector_oracle_f1": _selector_f1(runner, core_budget),
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None)
    parser.add_argument("--out-root", default=os.path.join(ROOT, "results", "paper_b_allocation"))
    args = parser.parse_args(argv)
    if (
        args.episodes <= 0 or args.pretrain_episodes <= 0
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
        for variant in selected_variants:
            rows.append(_isolation_row(
                variant, seed, args.core_budget, args.device, checkpoint
            ))
            rows.append(_end_to_end_row(
                variant, seed, args.episodes, args.core_budget, args.device,
                checkpoint,
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
        "variants": {name: VARIANTS[name] for name in selected_variants},
        "panels": ["selector_isolation", "end_to_end"],
        "summary": path,
    })
    print(path)


if __name__ == "__main__":
    main()
