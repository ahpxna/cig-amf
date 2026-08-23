"""Run the Paper-B core-allocation comparison at an identical core budget.

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


VARIANTS = {
    "C-Core": "structural_capacity",
    "AbsD-Core": "behavioral_direction",
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


def _row(variant, seed, episodes, core_budget, device):
    RE.set_global_seed(seed)
    cfg = RE.default_cfg()
    cfg.update({
        "seed": int(seed),
        "core_selection_mode": VARIANTS[variant],
        # The allocation estimand needs matched capacity. Adaptive-k is tested
        # separately; it must not alter C-vs-|D| membership cardinality here.
        "belief_adaptive_k": False,
        "min_core_size": int(core_budget),
        "belief_adaptive_k_min": int(core_budget),
        "max_core_size": int(core_budget),
        "periph_require_full_signature": True,
        "periph_allow_legacy_items": False,
    })
    env = RE.make_main_env(
        task_mode="behavioral_drift",
        n_agents=24,
        max_steps=30,
        phase_length=40,
        seed=seed,
    )
    runner = RE.make_runner("Final-CIGAMF", env, cfg, device)
    history = runner.run(n_episodes=int(episodes), eval_every=10)
    core_sizes = [len(module.get_core_set()) for module in runner.belief_modules.values()]
    if any(size != int(core_budget) for size in core_sizes):
        raise RuntimeError(
            f"{variant}/seed={seed} violated matched budget: {core_sizes}"
        )
    return {
        "variant": variant,
        "selector": VARIANTS[variant],
        "seed": int(seed),
        "episodes": int(episodes),
        "core_budget": int(core_budget),
        "strict_5d_signature": True,
        "mean_core_size": _mean(history.get("mean_core_size", [])),
        "mean_reward": _mean(history.get("mean_reward", [])),
        "final_reward": float(history.get("mean_reward", [float("nan")])[-1]),
        "mean_f1": _mean(history.get("mean_f1", [])),
        "throughput_total": _mean(history.get("throughput_total_agent_steps_per_sec", [])),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--core-budget", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-root", default=os.path.join(ROOT, "results", "paper_b_allocation"))
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.core_budget <= 0 or not args.seeds:
        parser.error("episodes, core budget, and at least one seed must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    out_root = ensure_dir(os.path.abspath(args.out_root))
    rows = [
        _row(variant, seed, args.episodes, args.core_budget, args.device)
        for seed in args.seeds
        for variant in VARIANTS
    ]
    path = os.path.join(out_root, "summary_paper_b_allocation.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(os.path.join(out_root, "manifest.json"), {
        "experiment": "paper_b_c_vs_absd_allocation",
        "complete": True,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "core_budget": args.core_budget,
        "variants": VARIANTS,
        "summary": path,
    })
    print(path)


if __name__ == "__main__":
    main()
