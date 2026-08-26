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
import random

import numpy as np
import torch

try:
    from exp_common import ROOT, ensure_dir
    from run_paper_b_periphery import _memory_accounting
    from run_paper_b_allocation import (
        _decision_fidelity, _oracle_capacity_for_state,
    )
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, ensure_dir
    from scripts.run_paper_b_periphery import _memory_accounting
    from scripts.run_paper_b_allocation import (
        _decision_fidelity, _oracle_capacity_for_state,
    )

import run_experiment as RE
from envs.causal_adapter import bounded_candidate_neighbors, resolve_env_adapter
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


def _runner(
    variant, n_agents, seed, device, core_budget, candidate_max_degree=None
):
    # Every variant in a matched seed/population cell must start from the exact
    # same parameter RNG state.  Without this reset, sequential construction
    # confounds architecture differences with different random initialisation.
    RE.set_global_seed(int(seed))
    cfg = RE.default_cfg()
    cfg.update({
        "seed": int(seed), "k0_warmup": 0, "slow_ratio": 1.0,
        "belief_adaptive_k": False, "min_core_size": int(core_budget),
        "belief_adaptive_k_min": int(core_budget),
        "max_core_size": int(core_budget), "periph_require_full_signature": True,
        "periph_allow_legacy_items": False,
        "strict_causal_profile": True,
        "semantic_router_frozen": True,
    })
    # The Paper-B scalability panel measures bounded candidate-edge execution,
    # not an all-pairs system with a small full-core only.  Full-Explicit is
    # deliberately exempt because it is the dense upper reference.  Direct
    # helper calls default to a candidate budget equal to the explicit budget;
    # the CLI still passes its prespecified d_max explicitly.
    if candidate_max_degree is None:
        candidate_max_degree = int(core_budget)
    if variant != "Full-Explicit":
        cfg["candidate_max_degree"] = int(candidate_max_degree)
    if variant == "Full-Explicit":
        # Configure the reference before construction so the runner's
        # episode-zero invariant activates every pair immediately.
        cfg.update({
            "core_selection_mode": "full_explicit",
            "belief_adaptive_k": False,
            "min_core_size": int(n_agents) - 1,
            "belief_adaptive_k_min": int(n_agents) - 1,
            "max_core_size": int(n_agents) - 1,
            "seed_core_top_k": int(n_agents) - 1,
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
    return RE.make_runner("Final-CIGAMF", env, cfg, device)


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
            np_state = np.random.get_state()
            py_state = random.getstate()
            torch_state = torch.get_rng_state()
            cuda_state = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            forcer = getattr(runner, "forcer", None)
            forcer_state = (
                copy.deepcopy(forcer.state_dict())
                if forcer is not None and callable(getattr(forcer, "state_dict", None))
                else None
            )
            started = time.perf_counter()
            result = runner._select_actions_population(obs)
            latencies.append(time.perf_counter() - started)
            # The scaling probe is observational.  It may time the complete
            # action-selection path, but it must not advance forcing counters or
            # RNG streams used by the subsequent run.
            np.random.set_state(np_state)
            random.setstate(py_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            if forcer_state is not None:
                forcer.load_state_dict(forcer_state)
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
                actions_arr = np.argmax(logits_arr, axis=-1).astype(np.int64)
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
                actions_arr = np.argmax(logits_arr, axis=-1).astype(np.int64)
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
        "inference_latency_protocol": "side_effect_restored_full_action_selection",
    }


def _candidate_oracle_recall_at_degree(
    env, *, n_states, max_degree, horizon, discount, trials, seed,
):
    """Certify whether the fixed candidate provider retains oracle top-C edges.

    Candidate construction is deliberately evaluated independently of learned
    allocation.  For each bank state and ego, the unrestricted clone-state
    oracle ranks all positive-capacity neighbours; recall is measured on the
    top ``min(d_max, n_positive)`` set.  States with no causal edge are not
    treated as automatic successes because they provide no pruning evidence.
    """
    if int(n_states) <= 0 or int(max_degree) <= 0:
        raise ValueError("candidate oracle recall requires positive state and degree budgets")
    adapter = resolve_env_adapter(env)
    outer = env.clone_state()
    hits = 0
    comparisons = 0
    evaluated_ego_states = 0
    try:
        bank = env.sample_state_bank(
            n_states=int(n_states), burn_in=3,
            bank_seed=int(seed), min_remaining_steps=int(horizon),
        )
        for state_index, state in enumerate(bank):
            capacities = _oracle_capacity_for_state(
                env, state, horizon=int(horizon), discount=float(discount),
                trials=int(trials),
                seed=int(seed) + 100003 * int(state_index),
            )
            env.restore_state(copy.deepcopy(state))
            for ego, scores in capacities.items():
                oracle_rank = [
                    int(target) for target, capacity in sorted(
                        scores.items(), key=lambda item: (-float(item[1]), int(item[0]))
                    )
                    if float(capacity) > 0.0
                ][:int(max_degree)]
                if not oracle_rank:
                    continue
                candidate_ids = set(bounded_candidate_neighbors(
                    adapter, int(ego), int(max_degree),
                ))
                hits += sum(target in candidate_ids for target in oracle_rank)
                comparisons += len(oracle_rank)
                evaluated_ego_states += 1
    finally:
        env.restore_state(outer)
    recall = float(hits / comparisons) if comparisons else float("nan")
    return {
        "candidate_oracle_recall_at_degree": recall,
        "candidate_oracle_recall_hits": int(hits),
        "candidate_oracle_recall_comparisons": int(comparisons),
        "candidate_oracle_recall_evaluated_ego_states": int(evaluated_ego_states),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--agent-counts", type=int, nargs="+", default=[12, 24, 48])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--core-budget", type=int, default=3)
    parser.add_argument(
        "--candidate-max-degree", type=int, default=8,
        help=(
            "Frozen policy-independent candidate degree for edge-scaling "
            "variants. Full-Explicit remains dense by definition."
        ),
    )
    parser.add_argument(
        "--candidate-recall-states", type=int, default=4,
        help="Oracle-only state-bank size for the top-C candidate-recall gate.",
    )
    parser.add_argument(
        "--candidate-recall-horizon", type=int, default=1,
        help="One-step clone-state capacity horizon used by the candidate-recall oracle.",
    )
    parser.add_argument(
        "--candidate-recall-trials", type=int, default=2,
        help="Common-random-number trials per oracle action for candidate recall.",
    )
    parser.add_argument(
        "--candidate-recall-min", type=float, default=0.80,
        help="Frozen minimum oracle top-C recall@d_max required for a scaling claim.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-root", default=os.path.join(ROOT, "results", "paper_b_scaling"))
    args = parser.parse_args(argv)
    if not args.seeds or any(value <= 1 for value in args.agent_counts):
        parser.error("seeds and agent counts greater than one are required")
    if (
        args.candidate_max_degree <= 0
        or args.candidate_recall_states <= 0
        or args.candidate_recall_horizon != 1
        or args.candidate_recall_trials <= 0
        or not 0.0 <= args.candidate_recall_min <= 1.0
    ):
        parser.error("candidate recall requires horizon=1, positive budgets, and minimum in [0, 1]")
    rows = []
    for seed in args.seeds:
        for n_agents in args.agent_counts:
            group = {}
            probe_seed = int(seed) + int(n_agents) * 1009 + 92009
            degree = min(int(args.candidate_max_degree), int(n_agents) - 1)
            # This uses a fresh, policy-independent environment instance so
            # candidate recall cannot be affected by any variant's learned
            # allocation or trajectory.  It is a prerequisite, not an
            # end-to-end score.
            oracle_env = RE.make_main_env(
                task_mode="behavioral_drift", n_agents=int(n_agents),
                max_steps=max(30, int(args.candidate_recall_horizon) + 8),
                phase_length=40, seed=int(seed),
            )
            oracle_cfg = RE.default_cfg()
            candidate_gate = _candidate_oracle_recall_at_degree(
                oracle_env,
                n_states=int(args.candidate_recall_states),
                max_degree=degree,
                horizon=int(args.candidate_recall_horizon),
                discount=float(oracle_cfg["discount"]),
                trials=int(args.candidate_recall_trials),
                seed=probe_seed + 700001,
            )
            for variant in VARIANTS:
                runner = _runner(
                    variant, n_agents, seed, args.device,
                    min(args.core_budget, int(n_agents) - 1),
                    degree,
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
                    "candidate_max_degree_protocol": int(
                        degree
                    ),
                    "candidate_recall_applicable": int(
                        variant not in {"Full-Explicit", "PureMeanField"}
                    ),
                    **(
                        candidate_gate
                        if variant not in {"Full-Explicit", "PureMeanField"}
                        else {
                            "candidate_oracle_recall_at_degree": float("nan"),
                            "candidate_oracle_recall_hits": 0,
                            "candidate_oracle_recall_comparisons": 0,
                            "candidate_oracle_recall_evaluated_ego_states": 0,
                        }
                    ),
                    "measured_edge_count": int(
                        history.get("measured_edge_count", [0])[-1]
                    ) if history.get("measured_edge_count") else 0,
                    "E_t": int(history.get("E_t", [0])[-1])
                    if history.get("E_t") else 0,
                    "K_t": int(history.get("K_t", [0])[-1])
                    if history.get("K_t") else 0,
                    "mean_candidate_degree": _mean(history.get("d_bar", [])),
                    "candidate_construction_subquadratic": bool(
                        history.get("candidate_construction_subquadratic", [False])[-1]
                    ) if history.get("candidate_construction_subquadratic") else False,
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
        "episodes": args.episodes,
        "candidate_max_degree": int(args.candidate_max_degree),
        "candidate_policy": "adapter-owned, policy-independent weak-prior candidate set",
        "candidate_oracle_recall": {
            "minimum": float(args.candidate_recall_min),
            "states": int(args.candidate_recall_states),
            "horizon": int(args.candidate_recall_horizon),
            "trials": int(args.candidate_recall_trials),
            "definition": "unrestricted oracle positive-C top-min(d_max,n_positive) recall",
        },
        "full_explicit_reference_is_dense": True,
        "variants": list(VARIANTS), "summary": summary,
    })
    print(summary)


if __name__ == "__main__":
    main()
