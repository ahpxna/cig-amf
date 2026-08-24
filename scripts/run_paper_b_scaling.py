"""Paper-B matched-budget runtime and memory scaling panel.

This harness reports, rather than assumes, the reward/throughput/memory Pareto
trade-off for explicit, semantic-memory, and single-mean representations at
several population sizes. It is separate from the fixed-24-agent fidelity
panels so scale is an explicit experimental axis.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import tempfile
import time

import numpy as np
import torch

try:
    from exp_common import ROOT, ensure_dir
    from run_paper_b_periphery import _memory_accounting
    from run_paper_b_allocation import _decision_fidelity
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, ensure_dir
    from scripts.run_paper_b_periphery import _memory_accounting
    from scripts.run_paper_b_allocation import _decision_fidelity

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


def _decision_probe_with_latency(runner, n_states, seed):
    bank = runner.env.sample_state_bank(
        n_states=int(n_states), burn_in=3, bank_seed=int(seed)
    )
    outer = runner.env.clone_state()
    logits, values, actions, latencies = [], [], [], []
    try:
        for state in bank:
            runner.env.restore_state(copy.deepcopy(state))
            obs = runner.env._get_obs_all()
            started = time.perf_counter()
            result = runner._select_actions_population(obs)
            latencies.append(time.perf_counter() - started)
            if (
                isinstance(result, tuple) and len(result) == 2
                and isinstance(result[1], dict)
            ):
                selected, cache = result
                logits_arr = np.asarray(cache["policy_logits"], dtype=np.float64)
                values_arr = np.asarray(
                    [cache["value_cache"][ego] for ego in range(runner.n_agents)],
                    dtype=np.float64,
                )
                actions_arr = np.asarray(
                    [selected[ego] for ego in range(runner.n_agents)], dtype=np.int64
                )
            elif isinstance(result, tuple) and len(result) == 3:
                # PureMeanField keeps its historical action-selection interface.
                # Reconstruct the same masked policy logits for a comparable
                # decision-fidelity probe without changing the training API.
                selected, values_np, mean_field = result
                obs_batch = np.stack([
                    runner.env_adapter.observation(obs, ego)
                    for ego in range(runner.n_agents)
                ])
                with torch.no_grad():
                    obs_t = torch.as_tensor(
                        obs_batch, dtype=torch.float32, device=runner.device
                    )
                    mf_t = torch.as_tensor(
                        np.stack(mean_field), dtype=torch.float32, device=runner.device
                    )
                    raw_logits, _ = runner._forward(obs_t, mf_t)
                    valid = np.stack([
                        runner.env_adapter.valid_action_mask(agent)
                        for agent in range(runner.n_agents)
                    ])
                    raw_logits = raw_logits.masked_fill(
                        ~torch.as_tensor(valid, dtype=torch.bool, device=raw_logits.device),
                        -torch.inf,
                    )
                    logits_arr = raw_logits.detach().cpu().numpy().astype(np.float64)
                values_arr = np.asarray(values_np, dtype=np.float64)
                actions_arr = np.asarray(selected, dtype=np.int64)
            else:
                raise RuntimeError("unsupported scaling decision-probe interface")
            logits.append(logits_arr)
            values.append(values_arr)
            actions.append(actions_arr)
    finally:
        runner.env.restore_state(outer)
    latency = np.asarray(latencies, dtype=np.float64) * 1000.0
    return {
        "logits": np.stack(logits, axis=0),
        "values": np.stack(values, axis=0),
        "actions": np.stack(actions, axis=0),
    }, {
        "inference_latency_mean_ms": float(np.mean(latency)),
        "inference_latency_p50_ms": float(np.percentile(latency, 50)),
        "inference_latency_p95_ms": float(np.percentile(latency, 95)),
    }


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
            group = {}
            probe_seed = int(seed) + int(n_agents) * 1009 + 92009
            for variant in VARIANTS:
                runner = _runner(
                    variant, n_agents, seed, args.device,
                    min(args.core_budget, int(n_agents) - 1),
                )
                history = runner.run(n_episodes=int(args.episodes), eval_every=10)
                memory = _memory_accounting(runner)
                probe, latency = _decision_probe_with_latency(
                    runner, n_states=4, seed=probe_seed
                )
                group[variant] = ({
                    "variant": variant,
                    "seed": int(seed),
                    "n_agents": int(n_agents),
                    "episodes": int(args.episodes),
                    "mean_reward": _mean(history.get("mean_reward", [])),
                    "reward_per_agent": _mean(history.get("reward_per_agent", [])),
                    "throughput_total": _mean(
                        history.get("throughput_total_agent_steps_per_sec", [])
                    ),
                    **latency,
                    **memory,
                }, probe)
            reference = group["Full-Explicit"][1]
            for variant in VARIANTS:
                row, probe = group[variant]
                row.update(_decision_fidelity(probe, reference))
                row["decision_fidelity_reference"] = "Full-Explicit"
                row["decision_probe_state_count"] = 4
                rows.append(row)
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
