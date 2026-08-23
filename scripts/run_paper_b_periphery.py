"""Dedicated Paper-B peripheral encoders under a shared oracle-fixed core."""

import argparse
import copy
import csv
import json
import os
import tempfile

import numpy as np

try:
    from exp_common import ROOT, ensure_dir
    from run_h2_selectivity import (
        _capture_frozen_learning_checkpoint,
        _restore_frozen_learning_checkpoint,
    )
    from run_paper_b_allocation import (
        _mean_oracle_capacity, _oracle_capacity_for_state,
    )
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, ensure_dir
    from scripts.run_h2_selectivity import (
        _capture_frozen_learning_checkpoint,
        _restore_frozen_learning_checkpoint,
    )
    from scripts.run_paper_b_allocation import (
        _mean_oracle_capacity, _oracle_capacity_for_state,
    )

import run_experiment as RE
from runners.h3_ablation_runner import H3NoMultiMemoryRunner


VARIANTS = {
    "Semantic-Free": {"runner": "multi", "num_memory_slots": 6},
    "Semantic-Only": {
        "runner": "multi", "num_memory_slots": 4,
        "periph_semantic_mass": 1.0,
    },
    "Unconstrained": {
        "runner": "multi", "num_memory_slots": 6,
        "periph_routing_mode": "unconstrained",
    },
    "No-Aux": {
        "runner": "multi", "num_memory_slots": 6,
        "periph_lb_coeff": 0.0, "periph_orth_coeff": 0.0,
    },
    "Single-Mean": {"runner": "single", "num_memory_slots": 6},
}


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(
        prefix=".tmp-periphery-", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _mean(values):
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _cfg(seed, core_budget):
    cfg = RE.default_cfg()
    cfg.update({
        "seed": int(seed),
        "k0_warmup": 0,
        "slow_ratio": 1.0,
        "core_selection_mode": "oracle_capacity",
        "belief_adaptive_k": False,
        "min_core_size": int(core_budget),
        "belief_adaptive_k_min": int(core_budget),
        "max_core_size": int(core_budget),
        "periph_require_full_signature": True,
        "periph_allow_legacy_items": False,
        "periph_use_uniform_mix": False,
    })
    return cfg


def _env(seed):
    return RE.make_main_env(
        task_mode="behavioral_drift", n_agents=24, max_steps=30,
        phase_length=40, seed=int(seed),
    )


def _common_checkpoint(seed, core_budget, device):
    RE.set_global_seed(seed)
    runner = RE.make_runner(
        "Final-CIGAMF", _env(seed), _cfg(seed, core_budget), device
    )
    return _capture_frozen_learning_checkpoint(runner)


def _restore_shared_state(runner, checkpoint):
    shared = copy.deepcopy(checkpoint)
    shared["state"].pop("periph_module", None)
    shared["optimizer_state"].pop("policy_optim", None)
    _restore_frozen_learning_checkpoint(runner, shared)


def _seed_oracle_core(runner, core_budget, table):
    runner.oracle_capacity_scores_by_ego = copy.deepcopy(table)
    for ego, belief in runner.belief_modules.items():
        row = table.get(int(ego), {})
        ranked = sorted(
            belief.neighbor_ids,
            key=lambda neighbor: float(row.get(int(neighbor), 0.0)),
            reverse=True,
        )
        belief.set_fixed_core(ranked[:int(core_budget)])


def _oracle_capacity_table(seed, checkpoint, core_budget, device, n_states=2):
    runner = RE.make_runner(
        "Final-CIGAMF", _env(seed), _cfg(seed, core_budget), device
    )
    _restore_frozen_learning_checkpoint(runner, checkpoint)
    runner.env.set_behaviour_override("cooperative")
    bank_seed = int(seed) + 92001
    bank = runner.env.sample_state_bank(
        n_states=int(n_states), burn_in=3, bank_seed=bank_seed
    )
    oracle_bank = [
        _oracle_capacity_for_state(
            runner.env, state,
            horizon=int(runner.cfg["causal_horizon"]),
            discount=float(runner.cfg["discount"]),
            trials=1,
            seed=bank_seed + index * 100003,
        )
        for index, state in enumerate(bank)
    ]
    return _mean_oracle_capacity(oracle_bank)


def _run_variant(
    name, seed, episodes, core_budget, device, checkpoint, oracle_capacity
):
    RE.set_global_seed(seed)
    spec = VARIANTS[name]
    cfg = _cfg(seed, core_budget)
    cfg.update({key: value for key, value in spec.items() if key != "runner"})
    env = _env(seed)
    runner = (
        H3NoMultiMemoryRunner(env, cfg, device=device)
        if spec["runner"] == "single"
        else RE.make_runner("Final-CIGAMF", env, cfg, device)
    )
    _restore_shared_state(runner, checkpoint)
    _seed_oracle_core(runner, core_budget, oracle_capacity)
    history = runner.run(n_episodes=int(episodes), eval_every=10)
    core_sizes = [
        len(module.get_core_set()) for module in runner.belief_modules.values()
    ]
    if any(size != int(core_budget) for size in core_sizes):
        raise RuntimeError(f"{name}/seed={seed} violated fixed oracle core")
    diagnostics = (
        runner.periph_module.get_slot_diagnostics()
        if callable(getattr(runner.periph_module, "get_slot_diagnostics", None))
        else {}
    )
    return {
        "variant": name,
        "seed": int(seed),
        "episodes": int(episodes),
        "core_budget": int(core_budget),
        "core_contract": "oracle_fixed_equal_budget",
        "checkpoint_sha256": checkpoint["sha256"],
        "mean_core_size": _mean(core_sizes),
        "mean_reward": _mean(history.get("mean_reward", [])),
        "mean_f1": _mean(history.get("mean_f1", [])),
        "throughput_total": _mean(
            history.get("throughput_total_agent_steps_per_sec", [])
        ),
        "usage_entropy_ratio": float(
            diagnostics.get("usage_entropy_ratio", -1.0)
        ),
        "slot_cos_offdiag": float(
            diagnostics.get("mean_offdiag_cosine", -1.0)
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--core-budget", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None)
    parser.add_argument(
        "--out-root", default=os.path.join(ROOT, "results", "paper_b_periphery")
    )
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.core_budget <= 0 or not args.seeds:
        parser.error("episodes, core budget, and seeds must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    out_root = ensure_dir(os.path.abspath(args.out_root))
    rows = []
    selected_variants = args.variants or list(VARIANTS)
    for seed in args.seeds:
        checkpoint = _common_checkpoint(seed, args.core_budget, args.device)
        oracle_capacity = _oracle_capacity_table(
            seed, checkpoint, args.core_budget, args.device
        )
        for name in selected_variants:
            rows.append(_run_variant(
                name, seed, args.episodes, args.core_budget, args.device,
                checkpoint, oracle_capacity,
            ))
    summary_path = os.path.join(out_root, "summary_paper_b_periphery.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(os.path.join(out_root, "manifest.json"), {
        "experiment": "paper_b_fixed_core_periphery",
        "complete": True,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "core_budget": args.core_budget,
        "variants": list(selected_variants),
        "summary": summary_path,
    })
    print(summary_path)


if __name__ == "__main__":
    main()
