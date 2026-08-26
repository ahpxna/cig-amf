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
import inspect
import threading

try:
    import psutil
except ImportError:  # confirmatory validation fails closed when unavailable
    psutil = None

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
from envs.causal_adapter import build_dynamic_candidate_map, resolve_env_adapter
from runners.h3_ablation_runner import H3NoMultiMemoryRunner
from utils.paper_contracts import PAPER_B_SELECTOR_ORACLE_HORIZON


VARIANTS = (
    "PureMeanField", "Attention-Mean", "Full-Explicit",
    "Semantic-Free", "Semantic-Free-Unrestricted", "Single-Mean",
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
    variant, n_agents, seed, device, core_budget, candidate_max_degree=None,
    candidate_cell_width=4.0, candidate_stencil_radius=1, candidate_radius=None,
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
    # Dynamic Semantic-Free is the bounded candidate treatment.  The
    # otherwise identical Semantic-Free-Unrestricted arm is deliberately
    # all-pairs, allowing H4 to isolate candidate pruning from architecture.
    bounded_variant = variant not in {
        "Full-Explicit", "Semantic-Free-Unrestricted", "PureMeanField"
    }
    if bounded_variant:
        if candidate_max_degree is None:
            candidate_max_degree = int(core_budget)
        cfg["candidate_max_degree"] = int(candidate_max_degree)
        cfg["candidate_refresh_interval"] = 1
        cfg["candidate_cell_width"] = float(candidate_cell_width)
        cfg["candidate_stencil_radius"] = int(candidate_stencil_radius)
        cfg["candidate_radius"] = candidate_radius
    else:
        cfg["candidate_max_degree"] = None
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


def _set_periphery_diagnostics(runner, enabled):
    module = getattr(runner, "periph_module", None)
    setter = getattr(module, "set_diagnostics_enabled", None)
    if callable(setter):
        return setter(bool(enabled))
    return None


def _select_for_probe(runner, obs):
    """Deterministic policy inference without forcing or training-cache work."""
    selector = runner._select_actions_population
    parameters = inspect.signature(selector).parameters
    kwargs = {}
    if "apply_forcing" in parameters:
        kwargs["apply_forcing"] = False
    if "collect_training_cache" in parameters:
        kwargs["collect_training_cache"] = False
    if "force_candidate_refresh" in parameters:
        # State-bank probes restore different environment states without
        # advancing the runner interaction counter.  Force a candidate refresh
        # so every probed state uses Gamma(O_t), not the first bank state's set.
        kwargs["force_candidate_refresh"] = True
    return selector(obs, **kwargs)


def _sync_device(runner):
    device = getattr(runner, "device", None)
    if torch.cuda.is_available() and device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _decision_probe_with_latency(runner, n_states, seed):
    bank = runner.env.sample_state_bank(
        n_states=int(n_states), burn_in=3, bank_seed=int(seed)
    )
    outer = runner.env.clone_state()
    logits, values, actions, latencies = [], [], [], []
    previous_diag = _set_periphery_diagnostics(runner, False)
    try:
        for state in bank:
            runner.env.restore_state(copy.deepcopy(state))
            obs = runner.env._get_obs_all()
            # No RNG/forcing snapshot is necessary on Final-CIGAMF because the
            # scaling path is deterministic and apply_forcing=False.  Keep RNG
            # restoration for legacy/PureMeanField interfaces that may sample.
            np_state = np.random.get_state()
            py_state = random.getstate()
            torch_state = torch.get_rng_state()
            cuda_state = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            _sync_device(runner)
            started = time.perf_counter()
            result = _select_for_probe(runner, obs)
            _sync_device(runner)
            latencies.append(time.perf_counter() - started)
            np.random.set_state(np_state)
            random.setstate(py_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
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
        if previous_diag is not None:
            _set_periphery_diagnostics(runner, previous_diag)
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
        "inference_latency_protocol": (
            "deterministic_policy_inference_without_training_cache_sampling_or_epsilon_forcing"
        ),
    }


def _runtime_memory_probe(runner, n_states, seed):
    """Measure process/GPU peak memory on a separate deterministic probe.

    CPU RSS is sampled in a dedicated thread and is therefore intentionally
    not mixed with the latency timing panel. GPU allocator peaks use PyTorch's
    native peak counters. Analytic parameter/state storage remains a distinct
    quantity reported by `_memory_accounting`.
    """
    if psutil is None:
        raise RuntimeError(
            "Paper-B confirmatory runtime-memory scaling requires psutil; "
            "install the pinned requirements before running this panel"
        )
    bank = runner.env.sample_state_bank(
        n_states=int(n_states), burn_in=3, bank_seed=int(seed)
    )
    outer = runner.env.clone_state()
    process = psutil.Process(os.getpid())
    baseline_rss = int(process.memory_info().rss)
    peak_rss = [baseline_rss]
    stop = threading.Event()

    def sample_rss():
        while not stop.is_set():
            try:
                peak_rss[0] = max(peak_rss[0], int(process.memory_info().rss))
            except Exception:
                pass
            stop.wait(0.001)

    cuda_device = getattr(runner, "device", None)
    use_cuda = (
        torch.cuda.is_available()
        and cuda_device is not None
        and str(cuda_device).startswith("cuda")
    )
    previous_diag = _set_periphery_diagnostics(runner, False)
    sampler = threading.Thread(target=sample_rss, daemon=True)
    if use_cuda:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        cuda_baseline_alloc = int(torch.cuda.memory_allocated(cuda_device))
        cuda_baseline_reserved = int(torch.cuda.memory_reserved(cuda_device))
    else:
        cuda_baseline_alloc = 0
        cuda_baseline_reserved = 0
    sampler.start()
    try:
        for state in bank:
            runner.env.restore_state(copy.deepcopy(state))
            obs = runner.env._get_obs_all()
            _select_for_probe(runner, obs)
        _sync_device(runner)
        peak_rss[0] = max(peak_rss[0], int(process.memory_info().rss))
        if use_cuda:
            peak_cuda_alloc = int(torch.cuda.max_memory_allocated(cuda_device))
            peak_cuda_reserved = int(torch.cuda.max_memory_reserved(cuda_device))
        else:
            peak_cuda_alloc = 0
            peak_cuda_reserved = 0
    finally:
        stop.set()
        sampler.join(timeout=1.0)
        if previous_diag is not None:
            _set_periphery_diagnostics(runner, previous_diag)
        runner.env.restore_state(outer)
    return {
        "runtime_peak_process_rss_bytes": int(peak_rss[0]),
        "runtime_peak_process_rss_delta_bytes": int(max(0, peak_rss[0] - baseline_rss)),
        "runtime_peak_cuda_allocated_bytes": int(peak_cuda_alloc),
        "runtime_peak_cuda_allocated_delta_bytes": int(
            max(0, peak_cuda_alloc - cuda_baseline_alloc)
        ),
        "runtime_peak_cuda_reserved_bytes": int(peak_cuda_reserved),
        "runtime_peak_cuda_reserved_delta_bytes": int(
            max(0, peak_cuda_reserved - cuda_baseline_reserved)
        ),
        "runtime_memory_protocol": (
            "separate_deterministic_probe_sampled_process_rss_and_torch_cuda_peaks"
        ),
        "runtime_memory_state_count": int(len(bank)),
    }


def _top_positive(scores, max_degree):
    return [
        int(target) for target, capacity in sorted(
            scores.items(), key=lambda item: (-float(item[1]), int(item[0]))
        ) if float(capacity) > 0.0
    ][:int(max_degree)]


def _jaccard(left, right):
    left, right = set(left), set(right)
    if not left and not right:
        return 1.0
    return float(len(left & right) / max(1, len(left | right)))


def _candidate_oracle_recall_at_degree(
    env, *, n_states, max_degree, target_k, horizon, discount, trials, seed,
    cell_width=4.0, stencil_radius=1, radius=None, stability_min=0.80,
):
    """Oracle recall on only independently stable top-C ego-state rankings."""
    if (
        int(n_states) <= 0 or int(max_degree) <= 0 or int(target_k) <= 0
        or int(target_k) > int(max_degree) or int(trials) <= 0
    ):
        raise ValueError(
            "candidate oracle recall requires 0 < target_k <= max_degree "
            "and positive state/trial budgets"
        )
    if not 0.0 <= float(stability_min) <= 1.0:
        raise ValueError("candidate oracle stability threshold must be in [0,1]")
    adapter = resolve_env_adapter(env)
    outer = env.clone_state()
    hits = comparisons = evaluated = unresolved = 0
    jaccards = []
    try:
        bank = env.sample_state_bank(
            n_states=int(n_states), burn_in=3, bank_seed=int(seed),
            min_remaining_steps=int(horizon),
        )
        for state_index, state in enumerate(bank):
            # Independent CRN replicas estimate whether top-C truth itself is
            # stable enough to support a binary recall judgement.
            cap_a = _oracle_capacity_for_state(
                env, state, horizon=int(horizon), discount=float(discount),
                trials=int(trials), seed=int(seed) + 100003 * state_index + 11,
            )
            cap_b = _oracle_capacity_for_state(
                env, state, horizon=int(horizon), discount=float(discount),
                trials=int(trials), seed=int(seed) + 100003 * state_index + 7919,
            )
            env.restore_state(copy.deepcopy(state))
            candidate_map, _telemetry = build_dynamic_candidate_map(
                adapter, int(max_degree), cell_width=float(cell_width),
                stencil_radius=int(stencil_radius), radius=radius,
            )
            for ego in sorted(set(cap_a) & set(cap_b)):
                rank_a = _top_positive(cap_a[ego], target_k)
                rank_b = _top_positive(cap_b[ego], target_k)
                if not rank_a and not rank_b:
                    continue
                jac = _jaccard(rank_a, rank_b)
                jaccards.append(jac)
                if jac + 1e-12 < float(stability_min):
                    unresolved += 1
                    continue
                # Once stable, average the two independent oracle estimates and
                # rank that combined surface for the actual recall target.
                targets = set(cap_a[ego]) | set(cap_b[ego])
                mean_scores = {
                    int(j): 0.5 * (float(cap_a[ego].get(j, 0.0)) + float(cap_b[ego].get(j, 0.0)))
                    for j in targets
                }
                oracle_rank = _top_positive(mean_scores, target_k)
                if not oracle_rank:
                    continue
                candidate_ids = set(candidate_map[int(ego)])
                hits += sum(target in candidate_ids for target in oracle_rank)
                comparisons += len(oracle_rank)
                evaluated += 1
    finally:
        env.restore_state(outer)
    recall = float(hits / comparisons) if comparisons else float("nan")
    return {
        "candidate_oracle_recall_at_degree": recall,
        "candidate_oracle_recall_hits": int(hits),
        "candidate_oracle_recall_comparisons": int(comparisons),
        "candidate_oracle_recall_evaluated_ego_states": int(evaluated),
        "candidate_oracle_unresolved_ego_states": int(unresolved),
        "candidate_oracle_stable_fraction": float(
            evaluated / max(1, evaluated + unresolved)
        ),
        "candidate_oracle_ranking_jaccard_mean": (
            float(np.mean(jaccards)) if jaccards else float("nan")
        ),
        "candidate_oracle_ranking_jaccard_min": (
            float(np.min(jaccards)) if jaccards else float("nan")
        ),
        "candidate_oracle_stability_min": float(stability_min),
        "candidate_oracle_target_k": int(target_k),
        "candidate_oracle_replicates": 2,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--agent-counts", type=int, nargs="+", default=[12, 24, 48])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--core-budgets", type=int, nargs="+", default=[2, 3, 4, 5],
        help="Prespecified core-budget sweep for the H3b reward-compute Pareto panel.",
    )
    parser.add_argument(
        "--core-budget", type=int, default=None,
        help=argparse.SUPPRESS,
    )
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
        "--candidate-recall-horizon", type=int,
        default=int(PAPER_B_SELECTOR_ORACLE_HORIZON),
        help=(
            "One-step structural horizon used by the confirmatory Paper-B "
            "selector/candidate-recall oracle. Delayed-path validation is a "
            "separate H>1 latency panel."
        ),
    )
    parser.add_argument("--candidate-cell-width", type=float, default=4.0)
    parser.add_argument("--candidate-stencil-radius", type=int, default=1)
    parser.add_argument("--candidate-radius", type=float, default=None)
    parser.add_argument(
        "--candidate-max-cell-occupancy", type=int, default=16,
        help="Frozen maximum cell occupancy allowed for the strict linear-provider label.",
    )
    parser.add_argument(
        "--candidate-recall-trials", type=int, default=2,
        help="Common-random-number trials per oracle action for candidate recall.",
    )
    parser.add_argument(
        "--candidate-recall-min", type=float, default=0.80,
        help="Frozen minimum oracle top-C recall@d_max required for a scaling claim.",
    )
    parser.add_argument(
        "--candidate-recall-stability-min", type=float, default=0.80,
        help="Frozen minimum top-C Jaccard between independent oracle replicas.",
    )
    parser.add_argument(
        "--candidate-recall-stable-fraction-min", type=float, default=0.80,
        help="Frozen minimum fraction of informative ego-states with stable oracle top-C ranking.",
    )
    parser.add_argument("--candidate-max-relative-reward-drop", type=float, default=0.10)
    parser.add_argument("--candidate-max-relative-logit-error-increase", type=float, default=0.25)
    parser.add_argument("--candidate-max-relative-value-error-increase", type=float, default=0.25)
    parser.add_argument("--candidate-max-action-agreement-drop", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-root", default=os.path.join(ROOT, "results", "paper_b_scaling"))
    args = parser.parse_args(argv)
    if args.core_budget is not None:
        args.core_budgets = [int(args.core_budget)]
    if not args.seeds or any(value <= 1 for value in args.agent_counts):
        parser.error("seeds and agent counts greater than one are required")
    if (
        args.candidate_max_degree <= 0
        or args.candidate_recall_states <= 0
        or args.candidate_recall_horizon <= 0
        or args.candidate_recall_trials <= 0
        or args.candidate_cell_width <= 0.0
        or args.candidate_stencil_radius < 0
        or args.candidate_max_cell_occupancy <= 0
        or not 0.0 <= args.candidate_recall_min <= 1.0
        or not 0.0 <= args.candidate_recall_stability_min <= 1.0
        or not 0.0 <= args.candidate_recall_stable_fraction_min <= 1.0
        or not args.core_budgets or any(k <= 0 for k in args.core_budgets)
        or min(
            args.candidate_max_relative_reward_drop,
            args.candidate_max_relative_logit_error_increase,
            args.candidate_max_relative_value_error_increase,
            args.candidate_max_action_agreement_drop,
        ) < 0.0
    ):
        parser.error("invalid scaling/candidate gate configuration")
    rows = []
    for seed in args.seeds:
        for n_agents in args.agent_counts:
            probe_seed = int(seed) + int(n_agents) * 1009 + 92009
            degree = min(int(args.candidate_max_degree), int(n_agents) - 1)
            oracle_env = RE.make_main_env(
                task_mode="behavioral_drift", n_agents=int(n_agents),
                max_steps=max(30, int(args.candidate_recall_horizon) + 8),
                phase_length=40, seed=int(seed),
            )
            oracle_cfg = RE.default_cfg()
            # Candidate recall protects the largest explicit allocation budget
            # actually tested, not every d_max slot.  The candidate interface
            # has size d_max; its causal duty is to retain the top-k* relations
            # the allocator could promote, with k*=max tested core budget.
            target_k = min(
                degree, max(min(int(k), int(n_agents) - 1) for k in args.core_budgets)
            )
            candidate_gate = _candidate_oracle_recall_at_degree(
                oracle_env,
                n_states=int(args.candidate_recall_states),
                max_degree=degree,
                target_k=target_k,
                horizon=int(args.candidate_recall_horizon),
                discount=float(oracle_cfg["discount"]),
                trials=int(args.candidate_recall_trials),
                seed=probe_seed + 700001,
                cell_width=float(args.candidate_cell_width),
                stencil_radius=int(args.candidate_stencil_radius),
                radius=args.candidate_radius,
                stability_min=float(args.candidate_recall_stability_min),
            )
            for requested_budget in args.core_budgets:
                core_budget = min(int(requested_budget), int(n_agents) - 1)
                group = {}
                for variant in VARIANTS:
                    runner = _runner(
                        variant, n_agents, seed, args.device, core_budget, degree,
                        float(args.candidate_cell_width),
                        int(args.candidate_stencil_radius), args.candidate_radius,
                    )
                    history = runner.run(n_episodes=int(args.episodes), eval_every=10)
                    memory = _memory_accounting(runner)
                    probe, latency = _decision_probe_with_latency(
                        runner, n_states=4, seed=probe_seed + 31 * core_budget
                    )
                    runtime_memory = _runtime_memory_probe(
                        runner, n_states=4, seed=probe_seed + 17 + 31 * core_budget
                    )
                    bounded = variant not in {
                        "Full-Explicit", "PureMeanField", "Semantic-Free-Unrestricted"
                    }
                    state_counts = {
                        "shadow_pair_count_final": int(history.get("shadow_pair_count", [0])[-1]) if history.get("shadow_pair_count") else 0,
                        "shadow_pair_count_max": max(history.get("shadow_pair_count", [0])),
                        "full_state_pair_count_final": int(history.get("full_state_pair_count", [0])[-1]) if history.get("full_state_pair_count") else 0,
                        "full_state_pair_count_max": max(history.get("full_state_pair_count", [0])),
                        "belief_pair_count_final": int(history.get("belief_pair_count", [0])[-1]) if history.get("belief_pair_count") else 0,
                        "belief_pair_count_max": max(history.get("belief_pair_count", [0])),
                        "signature_pair_count_final": int(history.get("signature_pair_count", [0])[-1]) if history.get("signature_pair_count") else 0,
                        "signature_pair_count_max": max(history.get("signature_pair_count", [0])),
                        "degree_limited_ego_count_mean": _mean(history.get("degree_limited_ego_count", [])),
                    }
                    group[variant] = ({
                        "variant": variant,
                        "seed": int(seed),
                        "n_agents": int(n_agents),
                        "core_budget": int(core_budget),
                        "requested_core_budget": int(requested_budget),
                        "episodes": int(args.episodes),
                        "candidate_max_degree_protocol": int(degree),
                        "candidate_recall_applicable": int(bounded),
                        **(candidate_gate if bounded else {
                            "candidate_oracle_recall_at_degree": float("nan"),
                            "candidate_oracle_recall_hits": 0,
                            "candidate_oracle_recall_comparisons": 0,
                            "candidate_oracle_recall_evaluated_ego_states": 0,
                            "candidate_oracle_unresolved_ego_states": 0,
                            "candidate_oracle_stable_fraction": float("nan"),
                            "candidate_oracle_ranking_jaccard_mean": float("nan"),
                            "candidate_oracle_ranking_jaccard_min": float("nan"),
                            "candidate_oracle_stability_min": float(args.candidate_recall_stability_min),
                            "candidate_oracle_target_k": int(target_k),
                            "candidate_oracle_replicates": 2,
                        }),
                        "measured_edge_count": int(history.get("measured_edge_count", [0])[-1]) if history.get("measured_edge_count") else 0,
                        "E_t": int(history.get("E_t", [0])[-1]) if history.get("E_t") else 0,
                        "K_t": int(history.get("K_t", [0])[-1]) if history.get("K_t") else 0,
                        "E_t_max": int(max(history.get("E_t", [0]))),
                        "K_t_max": int(max(history.get("K_t", [0]))),
                        "mean_candidate_degree": _mean(history.get("d_bar", [])),
                        "candidate_construction_subquadratic": bool(history.get("candidate_construction_subquadratic", [False])[-1]) if history.get("candidate_construction_subquadratic") else False,
                        "candidate_construction_linear_candidate": bool(history.get("candidate_construction_linear_candidate", [False])[-1]) if history.get("candidate_construction_linear_candidate") else False,
                        "feature_snapshot_subquadratic": bool(history.get("feature_snapshot_subquadratic", [False])[-1]) if history.get("feature_snapshot_subquadratic") else False,
                        "feature_snapshot_linear_candidate": bool(history.get("feature_snapshot_linear_candidate", [False])[-1]) if history.get("feature_snapshot_linear_candidate") else False,
                        "candidate_refresh_ms": _mean(history.get("candidate_refresh_ms", [])),
                        "candidate_provider_work_units": _mean(history.get("candidate_provider_work_units", [])),
                        "candidate_cell_occupancy_mean": _mean(history.get("candidate_cell_occupancy_mean", [])),
                        "candidate_cell_occupancy_max": max(history.get("candidate_cell_occupancy_max", [-1])),
                        "candidate_churn": _mean(history.get("candidate_churn", [])),
                        "candidate_added_pairs": _mean(history.get("candidate_added_pairs", [])),
                        "candidate_removed_pairs": _mean(history.get("candidate_removed_pairs", [])),
                        "candidate_last_hash": str(history.get("candidate_map_hash", [""])[-1]) if history.get("candidate_map_hash") else "",
                        "candidate_provider": str(history.get("candidate_provider", ["unknown"])[-1]) if history.get("candidate_provider") else "unknown",
                        "mean_reward": _mean(history.get("mean_reward", [])),
                        "reward_per_agent": _mean(history.get("reward_per_agent", [])),
                        "throughput_total": _mean(history.get("throughput_total_agent_steps_per_sec", [])),
                        **state_counts, **latency, **memory, **runtime_memory,
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
        "core_budgets": [int(k) for k in args.core_budgets],
        "candidate_max_degree": int(args.candidate_max_degree),
        "candidate_policy": "dynamic adapter-owned policy-independent pre-measurement candidate set",
        "candidate_update_protocol": "refresh_before_measurement_each_step",
        "candidate_cell_width": float(args.candidate_cell_width),
        "candidate_stencil_radius": int(args.candidate_stencil_radius),
        "candidate_radius": args.candidate_radius,
        "candidate_max_cell_occupancy": int(args.candidate_max_cell_occupancy),
        "candidate_oracle_recall": {
            "minimum": float(args.candidate_recall_min),
            "states": int(args.candidate_recall_states),
            "horizon": int(args.candidate_recall_horizon),
            "trials": int(args.candidate_recall_trials),
            "independent_replicates": 2,
            "target_k_rule": "max tested explicit core budget clipped to d_max and N-1",
            "stability_min": float(args.candidate_recall_stability_min),
            "stable_fraction_min": float(args.candidate_recall_stable_fraction_min),
            "definition": "dynamic-state candidate recall of oracle top-k* C, where k* is the largest tested explicit core budget, using only ego-state rankings stable across independent CRN replicas and the same runtime provider",
        },
        "candidate_degradation_gate": {
            "max_relative_reward_drop": float(args.candidate_max_relative_reward_drop),
            "max_relative_logit_error_increase": float(args.candidate_max_relative_logit_error_increase),
            "max_relative_value_error_increase": float(args.candidate_max_relative_value_error_increase),
            "max_action_agreement_drop": float(args.candidate_max_action_agreement_drop),
            "reward_denominator_floor": 1.0,
            "error_denominator_floor": 0.1,
            "reference_variant": "Semantic-Free-Unrestricted",
        },
        "full_explicit_reference_is_dense": True,
        "inference_latency_protocol": (
            "deterministic_policy_inference_without_training_cache_sampling_or_epsilon_forcing"
        ),
        "runtime_memory_protocol": (
            "separate_deterministic_probe_sampled_process_rss_and_torch_cuda_peaks"
        ),
        "analytic_memory_protocol": (
            "analytic_trainable_parameters_plus_persistent_representation_state"
        ),
        "variants": list(VARIANTS), "summary": summary,
    })
    print(summary)


if __name__ == "__main__":
    main()
