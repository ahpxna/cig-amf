"""Paper-B entropy-adaptive core-budget panel.

Adaptive k is compared with every fixed integer k in [k_min, k_max].  A
single fixed-k comparator is selected from pooled *representation-cost* data
(K_t/N), never reward, then reused for every seed.  This preserves paired
inference and prevents the matched comparator itself from changing by seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile

import numpy as np

try:
    from exp_common import ROOT, ensure_dir
    from run_paper_b_allocation import _decision_fidelity, _decision_probe
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, ensure_dir
    from scripts.run_paper_b_allocation import _decision_fidelity, _decision_probe

import run_experiment as RE


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".paper-b-budget-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _mean(values):
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _make_runner(
    seed, device, *, adaptive, k_min, k_max, n_agents=24, full_explicit=False
):
    RE.set_global_seed(int(seed))
    cfg = RE.default_cfg()
    cfg.update({
        "seed": int(seed),
        "k0_warmup": 0,
        "slow_ratio": 1.0,
        "min_core_size": int(k_min if adaptive else k_max),
        "belief_adaptive_k_min": int(k_min if adaptive else k_max),
        "max_core_size": int(k_max),
        "belief_adaptive_k": bool(adaptive),
        "periph_require_full_signature": True,
        "periph_allow_legacy_items": False,
        "strict_causal_profile": True,
        "semantic_router_frozen": True,
        "causal_horizon": 1,
        "proxy_n_horizons": 1,
    })
    if full_explicit:
        full_k = int(n_agents) - 1
        cfg.update({
            "core_selection_mode": "full_explicit",
            "belief_adaptive_k": False,
            "min_core_size": full_k,
            "belief_adaptive_k_min": full_k,
            "max_core_size": full_k,
            "seed_core_top_k": full_k,
        })
    env = RE.make_main_env(
        task_mode="behavioral_drift", n_agents=int(n_agents), max_steps=30,
        phase_length=40, seed=int(seed),
    )
    return RE.make_runner("Final-CIGAMF", env, cfg, device)


def _run(seed, episodes, device, variant, k_min, k_max, n_agents=24):
    if variant == "Adaptive-K":
        runner = _make_runner(
            seed, device, adaptive=True, k_min=k_min, k_max=k_max,
            n_agents=n_agents
        )
    elif variant == "Full-Explicit":
        runner = _make_runner(
            seed, device, adaptive=False, k_min=k_max, k_max=k_max,
            n_agents=n_agents, full_explicit=True,
        )
    else:
        fixed_k = int(variant.rsplit("-", 1)[1])
        runner = _make_runner(
            seed, device, adaptive=False, k_min=fixed_k, k_max=fixed_k,
            n_agents=n_agents
        )
    history = runner.run(n_episodes=int(episodes), eval_every=10)
    saturation = [belief.get_saturation_stats() for belief in runner.belief_modules.values()]
    row = {
        "variant": variant,
        "seed": int(seed),
        "episodes": int(episodes),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "mean_reward": _mean(history.get("mean_reward", [])),
        "final_reward": float(history.get("mean_reward", [float("nan")])[-1]),
        "mean_core_size": _mean(history.get("mean_core_size", [])),
        "mean_K_t": _mean(history.get("K_t", [])),
        "mean_core_cost_per_ego": (
            _mean(history.get("K_t", [])) / float(max(1, int(n_agents)))
        ),
        "core_size_variance": float(np.var(history.get("mean_core_size", [0.0]))),
        "mean_hit_min_rate": _mean(item["hit_min_rate"] for item in saturation),
        "mean_hit_max_rate": _mean(item["hit_max_rate"] for item in saturation),
        "mean_effective_max_k": _mean(item["effective_max_k"] for item in saturation),
        "throughput_total": _mean(history.get("throughput_total_agent_steps_per_sec", [])),
        "matched_to_adaptive": 0,
        "mean_core_cost_gap_to_adaptive": float("nan"),
    }
    probe = _decision_probe(runner, n_states=4, seed=int(seed) + 91009)
    return row, probe


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--agent-count", type=int, default=24)
    parser.add_argument(
        "--out-root", default=os.path.join(ROOT, "results", "paper_b_adaptive_budget")
    )
    args = parser.parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be non-empty and unique")
    if (
        args.episodes <= 0 or args.k_min < 1 or args.k_max < args.k_min
        or args.agent_count <= args.k_max
    ):
        parser.error("invalid episode or adaptive-budget bounds")

    variants = ["Adaptive-K"] + [
        f"Fixed-K-{k}" for k in range(int(args.k_min), int(args.k_max) + 1)
    ] + ["Full-Explicit"]
    results_by_seed = {}
    for seed in args.seeds:
        results_by_seed[int(seed)] = {
            variant: _run(
                seed, args.episodes, args.device, variant, args.k_min, args.k_max,
                n_agents=args.agent_count
            )
            for variant in variants
        }

    fixed_names = [name for name in variants if name.startswith("Fixed-K-")]
    adaptive_cost = _mean(
        results_by_seed[int(seed)]["Adaptive-K"][0]["mean_core_cost_per_ego"]
        for seed in args.seeds
    )
    fixed_costs = {
        name: _mean(
            results_by_seed[int(seed)][name][0]["mean_core_cost_per_ego"]
            for seed in args.seeds
        )
        for name in fixed_names
    }
    matched = min(
        fixed_names, key=lambda name: abs(float(fixed_costs[name]) - adaptive_cost)
    )

    rows = []
    for seed in args.seeds:
        results = results_by_seed[int(seed)]
        reference = results["Full-Explicit"][1]
        adaptive_seed_cost = float(
            results["Adaptive-K"][0]["mean_core_cost_per_ego"]
        )
        for variant, (row, probe) in results.items():
            row.update(_decision_fidelity(probe, reference))
            row["decision_fidelity_reference"] = "Full-Explicit"
            row["matched_fixed_variant"] = matched
            row["pooled_adaptive_core_cost_per_ego"] = float(adaptive_cost)
            if variant == matched:
                row["matched_to_adaptive"] = 1
            if variant.startswith("Fixed-K-"):
                row["mean_core_cost_gap_to_adaptive"] = abs(
                    float(row["mean_core_cost_per_ego"]) - adaptive_seed_cost
                )
            rows.append(row)

    out_root = ensure_dir(os.path.abspath(args.out_root))
    summary = os.path.join(out_root, "summary_paper_b_adaptive_budget.csv")
    with open(summary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(os.path.join(out_root, "manifest.json"), {
        "experiment": "paper_b_entropy_adaptive_budget",
        "complete": True,
        "seeds": [int(seed) for seed in args.seeds],
        "episodes": int(args.episodes),
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
        "agent_count": int(args.agent_count),
        "variants": variants,
        "matching_rule": "single fixed k nearest pooled adaptive mean K_t/N cost; selected without reward",
        "matched_fixed_variant": matched,
        "pooled_adaptive_core_cost_per_ego": float(adaptive_cost),
        "summary": summary,
    })
    print(summary)


if __name__ == "__main__":
    main()
