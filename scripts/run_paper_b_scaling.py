"""Paper-B matched-budget runtime and memory scaling panel.

This harness reports, rather than assumes, the reward/throughput/memory Pareto
trade-off for explicit, semantic-memory, and single-mean representations at
several population sizes. It is separate from the fixed-24-agent fidelity
panels so scale is an explicit experimental axis.
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
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, ensure_dir

import run_experiment as RE
from runners.h3_ablation_runner import H3NoMultiMemoryRunner


VARIANTS = (
    "PureMeanField", "Attention-Mean", "Full-Explicit", "Semantic-Free",
    "Single-Mean",
)


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(
        prefix=".paper-b-scale-", suffix=".json", dir=directory
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
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _model_bytes(runner):
    if not hasattr(runner, "pair_rel_module"):
        modules = (runner.policy_value, runner.actor, runner.critic)
        return int(sum(
            parameter.numel() * parameter.element_size()
            for module in modules for parameter in module.parameters()
        ))
    modules = (runner.policy_value, runner.pair_rel_module.full_encoder,
               runner.periph_module, runner.belief_summary_builder)
    parameters = sum(
        parameter.numel() * parameter.element_size()
        for module in modules for parameter in module.parameters()
    )
    # Measure actual state allocation after the run, not an old all-pair
    # full-state formula.  Shadow is maintained for candidates; full z only
    # exists for selected core pairs.
    stats = runner.pair_rel_module.get_debug_stats()
    state = int(stats["full_state_bytes"] + stats["shadow_state_bytes"])
    return int(parameters + state)


def _runner(variant, n_agents, seed, device, core_budget):
    cfg = RE.default_cfg()
    cfg.update({
        "seed": int(seed), "k0_warmup": 0, "slow_ratio": 1.0,
        "belief_adaptive_k": False, "min_core_size": int(core_budget),
        "belief_adaptive_k_min": int(core_budget),
        "max_core_size": int(core_budget), "periph_require_full_signature": True,
        "periph_allow_legacy_items": False,
    })
    env = RE.make_main_env(
        task_mode="behavioral_drift", n_agents=int(n_agents), max_steps=30,
        phase_length=40, seed=int(seed),
    )
    if variant == "PureMeanField":
        return RE.make_runner("PureMeanField", env, cfg, device)
    if variant in {"Single-Mean", "Attention-Mean"}:
        if variant == "Attention-Mean":
            cfg["periph_beta_mode"] = "attention"
        return H3NoMultiMemoryRunner(env, cfg, device=device)
    runner = RE.make_runner("Final-CIGAMF", env, cfg, device)
    if variant == "Full-Explicit":
        runner.cfg["core_selection_mode"] = "full_explicit"
        for belief in runner.belief_modules.values():
            belief.max_core_size = len(belief.neighbor_ids)
            belief.set_fixed_core(belief.neighbor_ids)
        runner.pair_rel_module.reconcile_core_sets(
            {ego: belief.get_core_set() for ego, belief in runner.belief_modules.items()}
        )
    return runner


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--agent-counts", type=int, nargs="+", default=[12, 24, 48])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--core-budget", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-root", default=os.path.join(ROOT, "results", "paper_b_scaling"))
    args = parser.parse_args(argv)
    if not args.seeds or any(value <= 1 for value in args.agent_counts):
        parser.error("seeds and agent counts greater than one are required")
    rows = []
    for seed in args.seeds:
        for n_agents in args.agent_counts:
            for variant in VARIANTS:
                runner = _runner(
                    variant, n_agents, seed, args.device,
                    min(args.core_budget, int(n_agents) - 1),
                )
                history = runner.run(n_episodes=int(args.episodes), eval_every=10)
                rows.append({
                    "variant": variant,
                    "seed": int(seed),
                    "n_agents": int(n_agents),
                    "episodes": int(args.episodes),
                    "mean_reward": _mean(history.get("mean_reward", [])),
                    "throughput_total": _mean(
                        history.get("throughput_total_agent_steps_per_sec", [])
                    ),
                    "representation_memory_bytes": _model_bytes(runner),
                })
    out_root = ensure_dir(os.path.abspath(args.out_root))
    summary = os.path.join(out_root, "summary_paper_b_scaling.csv")
    with open(summary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(os.path.join(out_root, "manifest.json"), {
        "experiment": "paper_b_runtime_memory_scaling",
        "complete": True, "seeds": args.seeds, "agent_counts": args.agent_counts,
        "episodes": args.episodes, "variants": list(VARIANTS), "summary": summary,
    })
    print(summary)


if __name__ == "__main__":
    main()
