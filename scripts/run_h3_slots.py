"""Run H3: signature routing, slot collapse, and adaptive capacity.

The experiment isolates one mechanism at a time:

* ``Full-CIGAMF``: tracker-derived 5D signatures, semantic plus free slots,
  both auxiliary losses, adaptive cardinality, and no uniform mixing.
* ``Scalar-Only``: only signed influence is retained from the signature.
* ``Unconstrained-NoSemantic``: a learned softmax routes all slots.
* ``No-AuxLoss``: semantic routing without load-balance or orthogonality loss.
* ``Fixed-Cardinality``: min and max core size are both k_max.
* ``NoMultiMemory-SingleMean``: the existing single aggregate baseline.

``runner.run`` is invoked exactly once per variant. Repeated chunked calls
restart its local episode index, repeat calibration episode numbers, and make
the previous H3 protocol invalid.

Outputs default to ``results/h3`` and may be isolated with ``--out-root``.
Each attempt writes metadata, per-evaluation JSONL, per-seed summaries, a CSV,
and a hypothesis-gate JSON. Scientific gate failure is recorded as data; only
an incomplete or internally inconsistent run exits nonzero.
"""

import argparse
import csv
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch

try:
    from exp_common import ROOT, append_jsonl, ensure_dir, save_json
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, append_jsonl, ensure_dir, save_json

import run_experiment as RE
from runners.final_runner import FinalCIGAMFRunner
from runners.h3_ablation_runner import H3NoMultiMemoryRunner


BASE_H3_CONFIG = {
    "num_memory_slots": 6,
    "periph_use_uniform_mix": False,
    "periph_uniform_mix": 0.0,
    "periph_routing_mode": "semantic",
    "periph_signature_mode": "full",
    "periph_require_full_signature": True,
    "periph_allow_legacy_items": False,
    # Paper-B arms use the retained typed profile boundary.  Individual
    # ablations may change the declared representation mechanism, but may not
    # reactivate signed/legacy/uniform compatibility paths.
    "strict_causal_profile": True,
    "semantic_router_frozen": True,
}


VARIANTS = [
    {
        "name": "Full-CIGAMF",
        "runner_model": "Final-CIGAMF",
        "overrides": {},
    },
    {
        "name": "Scalar-Only",
        "runner_model": "Final-CIGAMF",
        "overrides": {"periph_signature_mode": "scalar"},
    },
    {
        "name": "Unconstrained-NoSemantic",
        "runner_model": "Final-CIGAMF",
        "overrides": {"periph_routing_mode": "unconstrained"},
    },
    {
        "name": "No-AuxLoss",
        "runner_model": "Final-CIGAMF",
        "overrides": {
            "periph_lb_coeff": 0.0,
            "periph_orth_coeff": 0.0,
        },
    },
    {
        "name": "Fixed-Cardinality",
        "runner_model": "Final-CIGAMF",
        "overrides": {
            "belief_adaptive_k": False,
            "min_core_size": 4,
            "belief_adaptive_k_min": 4,
            "max_core_size": 4,
        },
    },
    {
        "name": "NoMultiMemory-SingleMean",
        "runner_model": "H3FinalSingleMean",
        "overrides": {},
    },
]


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _finite_mean(values):
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def _finite_std(values):
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.std(values)) if values else float("nan")


def _history_value(history, key, index):
    values = history.get(key, [])
    return float(values[index]) if index < len(values) else float("nan")


def _resolve_out_root(value):
    if not value:
        return os.path.join(ROOT, "results", "h3")
    return value if os.path.isabs(value) else os.path.join(ROOT, value)


def _parameter_bytes(model):
    """Count unique tensor parameters for nn.Modules or runner containers."""
    modules = (
        [model]
        if isinstance(model, torch.nn.Module)
        else [value for value in vars(model).values() if isinstance(value, torch.nn.Module)]
    )
    seen = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            total += parameter.numel() * parameter.element_size()
    return int(total)


def _make_variant_runner(spec, env, cfg, device):
    if spec["runner_model"] == "H3FinalSingleMean":
        return H3NoMultiMemoryRunner(env, cfg, device=device)
    return RE.make_runner(spec["runner_model"], env, cfg, device)


def _build_heldout_probes(seed, n_episodes):
    """Create a common, independently seeded state set for every H3 arm."""
    env = RE.make_main_env(
        task_mode="behavioral_drift",
        n_agents=24,
        max_steps=30,
        phase_length=40,
        seed=int(seed) + 100_003,
    )
    rng = np.random.RandomState(int(seed) + 200_003)
    probes = []
    for probe_episode in range(int(n_episodes)):
        obs_all = env.reset()
        done = False
        step = 0
        while not done:
            probes.append(
                {
                    "probe_episode": int(probe_episode),
                    "step": int(step),
                    "obs_all": [np.asarray(obs).copy() for obs in obs_all],
                    "env_snapshot": env.clone_state(),
                }
            )
            actions = rng.randint(
                0, int(env.get_action_dim()), size=int(env.n_agents)
            ).tolist()
            obs_all, _, done, _ = env.step(actions)
            step += 1
    if not probes:
        raise RuntimeError("H3 held-out probe set is empty")
    return probes


def _device_sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _predict_final_on_probes(runner, probes):
    """Evaluate one trained Final-family runner without sampling or updates."""
    module = runner.periph_module
    reset = getattr(module, "reset_slot_diagnostics", None)
    if reset is None:
        reset = getattr(module, "reset_diagnostics", None)
    if callable(reset):
        reset()

    original_state = runner.env.clone_state()
    logits_rows = []
    value_rows = []
    _device_sync(runner.device)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for probe in probes:
                runner.env.restore_state(probe["env_snapshot"])
                obs_batch = []
                core_batch = []
                periph_batch = []
                belief_batch = []
                for ego in range(runner.n_agents):
                    belief_items = runner._build_belief_items_for_ego(ego)
                    periph_inputs = runner._build_periph_inputs_for_ego(ego)
                    obs_batch.append(
                        runner.env.get_obs_of_ego(probe["obs_all"], ego)
                    )
                    core_batch.append(runner._core_summary_for_ego(ego))
                    periph_batch.append(
                        runner._periph_summary_np_from_inputs(periph_inputs)
                    )
                    belief_batch.append(
                        runner._belief_summary_np_from_items(belief_items)
                    )

                logits, values = runner.policy_value(
                    torch.as_tensor(
                        np.stack(obs_batch), dtype=torch.float32,
                        device=runner.device,
                    ),
                    torch.as_tensor(
                        np.stack(core_batch), dtype=torch.float32,
                        device=runner.device,
                    ),
                    torch.as_tensor(
                        np.stack(periph_batch), dtype=torch.float32,
                        device=runner.device,
                    ),
                    torch.as_tensor(
                        np.stack(belief_batch), dtype=torch.float32,
                        device=runner.device,
                    ),
                )
                logits_rows.append(logits.detach().cpu().numpy())
                value_rows.append(values.detach().cpu().numpy())
        _device_sync(runner.device)
    finally:
        runner.env.restore_state(original_state)
    elapsed = time.perf_counter() - started
    return {
        "logits": np.concatenate(logits_rows, axis=0),
        "values": np.concatenate(value_rows, axis=0).reshape(-1),
        "elapsed_seconds": float(elapsed),
    }


def _tracker_support(runner, min_observations=2):
    counts = [
        runner.sig_tracker.get_n_observations(ego, neighbor)
        for ego in range(runner.n_agents)
        for neighbor in runner._candidate_ids(ego)
    ]
    observed = [count >= int(min_observations) for count in counts]
    return {
        "signature_observed_pair_fraction": float(np.mean(observed)),
        "signature_min_pair_observations": int(min(counts)) if counts else 0,
        "signature_median_pair_observations": float(np.median(counts)) if counts else 0.0,
        "signature_support_minimum_required": int(min_observations),
        "signature_empirical_support_valid": bool(observed)
        and float(np.mean(observed)) >= 0.95,
    }


def _validate_variant_config(spec, cfg):
    name = spec["name"]
    errors = []

    if bool(cfg.get("periph_use_uniform_mix", False)):
        errors.append("uniform mixing must be disabled in every valid H3 arm")
    if int(cfg.get("num_memory_slots", 0)) != 6:
        errors.append("H3 requires four semantic plus two free slots")
    if not bool(cfg.get("periph_require_full_signature", False)):
        errors.append("H3 must fail rather than silently use legacy signatures")

    routing = cfg.get("periph_routing_mode")
    signature = cfg.get("periph_signature_mode")
    if name == "Scalar-Only" and signature != "scalar":
        errors.append("Scalar-Only must use the explicitly declared scalar signature")
    if name != "Scalar-Only" and signature != "full":
        errors.append(f"{name} must use the full 5D signature")
    if name == "Unconstrained-NoSemantic" and routing != "unconstrained":
        errors.append("NoSemantic must use a learned unconstrained router")
    if name != "Unconstrained-NoSemantic" and routing != "semantic":
        errors.append(f"{name} must retain semantic routing")
    if name == "No-AuxLoss":
        if float(cfg.get("periph_lb_coeff", -1.0)) != 0.0:
            errors.append("No-AuxLoss must set load-balancing coefficient to zero")
        if float(cfg.get("periph_orth_coeff", -1.0)) != 0.0:
            errors.append("No-AuxLoss must set orthogonality coefficient to zero")
    if name == "Fixed-Cardinality":
        if bool(cfg.get("belief_adaptive_k", True)):
            errors.append("Fixed-Cardinality must disable adaptive k")
        if int(cfg.get("min_core_size", -1)) != int(cfg.get("max_core_size", -2)):
            errors.append("Fixed-Cardinality must set min_core_size=max_core_size")
    if int(cfg.get("belief_adaptive_k_min", -1)) != int(cfg.get("min_core_size", -2)):
        errors.append("adaptive and fixed core minima must be identical")

    if errors:
        raise ValueError(f"Invalid H3 config for {name}: " + "; ".join(errors))


def _configure_and_validate_runner(spec, runner, cfg):
    if spec["runner_model"] == "H3FinalSingleMean":
        if not isinstance(runner, FinalCIGAMFRunner):
            raise AssertionError(
                "Single-mean H3 arm must inherit FinalCIGAMFRunner"
            )
        if bool(getattr(runner, "use_multi_memory", True)):
            raise AssertionError("Single-mean arm still executes multi-memory")
        if getattr(runner, "ablation_contract", None) != (
            "peripheral_multislot_to_single_mean_only"
        ):
            raise AssertionError("Single-mean arm has no exact ablation contract")
        module = runner.periph_module
        checks = {
            "single mean module": bool(getattr(module, "is_single_mean", False)),
            "signature mode": module.signature_mode
            == cfg["periph_signature_mode"],
            "strict signature": bool(module.require_full_signature),
            "legacy disabled": not bool(module.allow_legacy_items),
            "forcer retained": hasattr(runner, "forcer"),
            "DR proxy retained": bool(runner.proxy.use_doubly_robust),
            "signature tracker retained": hasattr(runner, "sig_tracker"),
        }
        failed = [label for label, ok in checks.items() if not ok]
        if failed:
            raise AssertionError(
                "Single-mean arm is not a faithful Final ablation: "
                + ", ".join(failed)
            )
        return

    module = runner.periph_module
    module.set_ablation_modes(
        routing_mode=cfg["periph_routing_mode"],
        signature_mode=cfg["periph_signature_mode"],
        require_full_signature=cfg["periph_require_full_signature"],
    )
    module.allow_legacy_items = bool(cfg["periph_allow_legacy_items"])

    checks = {
        "slot count": int(module.num_slots) == int(cfg["num_memory_slots"]),
        "routing mode": module.routing_mode == cfg["periph_routing_mode"],
        "signature mode": module.signature_mode == cfg["periph_signature_mode"],
        "strict signature": bool(module.require_full_signature),
        "legacy disabled": not bool(module.allow_legacy_items),
        "uniform mix disabled": not bool(module.use_uniform_mix),
        "load-balance coefficient": float(module.lb_coeff)
        == float(cfg["periph_lb_coeff"]),
        "orthogonality coefficient": float(module.orth_coeff)
        == float(cfg["periph_orth_coeff"]),
    }
    failed = [label for label, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(
            f"Runner construction ignored H3 config for {spec['name']}: "
            + ", ".join(failed)
        )


def _slot_summary(runner):
    support = _tracker_support(runner)
    if not bool(getattr(runner, "use_multi_memory", True)):
        diag = runner.periph_module.get_input_diagnostics()
        full_fraction = float(diag.get("signature_full_fraction", float("nan")))
        input_valid = bool(
            diag.get("signature_source") == "full_profile"
            and np.isfinite(full_fraction)
            and abs(full_fraction - 1.0) <= 1e-12
            and bool(diag.get("require_full_signature", False))
        )
        return {
            "slot_diagnostics_applicable": False,
            "signature_protocol_valid": bool(
                input_valid and support["signature_empirical_support_valid"]
            ),
            **diag,
            **support,
        }

    diag = runner.periph_module.get_slot_diagnostics()
    full_fraction = float(diag.get("signature_full_fraction", float("nan")))
    input_protocol_valid = bool(
        diag.get("signature_source") == "full_profile"
        and np.isfinite(full_fraction)
        and abs(full_fraction - 1.0) <= 1e-12
        and bool(diag.get("require_full_signature", False))
    )
    if not input_protocol_valid:
        raise RuntimeError(
            "H3 did not consume tracker-derived 5D signatures: "
            f"source={diag.get('signature_source')!r}, "
            f"full_fraction={full_fraction}"
        )

    return {
        "slot_diagnostics_applicable": True,
        "signature_protocol_valid": bool(
            input_protocol_valid and support["signature_empirical_support_valid"]
        ),
        **diag,
        **support,
    }


def run_variant(
    spec,
    seed,
    episodes,
    eval_every,
    device,
    out_root,
    probes,
):
    name = spec["name"]
    out_dir = ensure_dir(os.path.join(out_root, f"{name}_seed{seed}"))
    jsonl = os.path.join(out_dir, "eval.jsonl")
    with open(jsonl, "w", encoding="utf-8"):
        pass

    RE.set_global_seed(seed)
    cfg = RE.default_cfg()
    cfg.update(BASE_H3_CONFIG)
    cfg.update(spec["overrides"])
    cfg["seed"] = int(seed)
    _validate_variant_config(spec, cfg)

    env = RE.make_main_env(
        task_mode="behavioral_drift",
        n_agents=24,
        max_steps=30,
        phase_length=40,
        seed=seed,
    )
    runner = _make_variant_runner(spec, env, cfg, device)
    _configure_and_validate_runner(spec, runner, cfg)

    # One continuous run preserves global episode semantics and performs each
    # warm-up/calibration event exactly once.
    history = runner.run(n_episodes=episodes, eval_every=eval_every)
    history = history if isinstance(history, dict) else getattr(runner, "history", {})

    expected_evals = len(range(0, int(episodes), int(eval_every)))
    eval_episodes = [int(v) for v in history.get("episodes", [])]
    if len(eval_episodes) != expected_evals:
        raise RuntimeError(
            f"{name} seed {seed} produced {len(eval_episodes)} evaluations; "
            f"expected {expected_evals}"
        )

    warmup = int(cfg.get("k0_warmup", 30))
    post_indices = []
    for index, episode in enumerate(eval_episodes):
        post_warmup = bool(episode >= warmup)
        if post_warmup:
            post_indices.append(index)
        append_jsonl(
            jsonl,
            {
                "record_type": "evaluation",
                "variant": name,
                "runner_model": spec["runner_model"],
                "seed": int(seed),
                "episode": int(episode),
                "post_warmup": post_warmup,
                "mean_core_size": _history_value(
                    history, "mean_core_size", index
                ),
                "mean_reward": _history_value(history, "mean_reward", index),
                "f1": _history_value(history, "mean_f1", index),
                "throughput_rollout": _history_value(
                    history, "throughput_agent_steps_per_sec", index
                ),
                "throughput_total": _history_value(
                    history, "throughput_total_agent_steps_per_sec", index
                ),
            },
        )

    probe_predictions = _predict_final_on_probes(runner, probes)
    heldout_state_count = int(probe_predictions["values"].size)
    heldout_prediction_seconds = float(probe_predictions["elapsed_seconds"])
    heldout_states_per_second = heldout_state_count / max(
        heldout_prediction_seconds, 1e-12
    )
    slot = _slot_summary(runner)
    append_jsonl(
        jsonl,
        {
            "record_type": "final_slot_diagnostics",
            "variant": name,
            "seed": int(seed),
            **slot,
        },
    )

    def post_values(key):
        return [_history_value(history, key, i) for i in post_indices]

    core_sizes = post_values("mean_core_size")
    rewards = post_values("mean_reward")
    f1_values = post_values("mean_f1")
    rollout_tp = post_values("throughput_agent_steps_per_sec")
    total_tp = post_values("throughput_total_agent_steps_per_sec")
    kmax = float(cfg.get("max_core_size", 4))
    valid_core_sizes = [v for v in core_sizes if np.isfinite(v)]

    summary = {
        "variant": name,
        "runner_model": spec["runner_model"],
        "seed": int(seed),
        "episodes": int(episodes),
        "eval_every": int(eval_every),
        "expected_evaluations": int(expected_evals),
        "completed_evaluations": int(len(eval_episodes)),
        "post_warmup_evaluations": int(len(post_indices)),
        "run_complete": True,
        "mean_core_size": _finite_mean(core_sizes),
        "core_size_std": _finite_std(core_sizes),
        "frac_k_at_kmax": (
            float(np.mean([v >= kmax - 1e-6 for v in valid_core_sizes]))
            if valid_core_sizes else float("nan")
        ),
        "mean_reward": _finite_mean(rewards),
        "mean_f1": _finite_mean(f1_values),
        "final_f1": (
            float(f1_values[-1]) if f1_values else float("nan")
        ),
        "throughput_rollout": _finite_mean(rollout_tp),
        "throughput_total": _finite_mean(total_tp),
        "parameter_bytes": _parameter_bytes(runner),
        "heldout_prediction_seconds": heldout_prediction_seconds,
        "heldout_agent_states_per_second": float(heldout_states_per_second),
        "heldout_agent_state_count": heldout_state_count,
        # Backward-compatible H3 collector fields.
        "usage_entropy_ratio": float(
            slot.get("usage_entropy_ratio", float("nan"))
        ),
        "hard_usage_entropy_ratio": float(
            slot.get("hard_usage_entropy_ratio", float("nan"))
        ),
        "assignment_mutual_info_ratio": float(
            slot.get("assignment_mutual_info_ratio", float("nan"))
        ),
        "slot_cos_offdiag": float(
            slot.get("mean_offdiag_cosine", float("nan"))
        ),
        "semantic_role_nmi": float(
            slot.get("semantic_role_nmi", float("nan"))
        ),
        "signature_centroid_distance": float(
            slot.get("mean_signature_centroid_distance", float("nan"))
        ),
        "monopoly_collapse": bool(slot.get("monopoly_collapse", False)),
        "diffuse_assignment_collapse": bool(
            slot.get("diffuse_assignment_collapse", False)
        ),
        "uniform_content_collapse": bool(
            slot.get("uniform_content_collapse", False)
        ),
        "collapse_detected": bool(slot.get("collapse_detected", False)),
        "signature_protocol_valid": bool(
            slot.get("signature_protocol_valid", False)
        ),
        "signature_observed_pair_fraction": float(
            slot.get("signature_observed_pair_fraction", float("nan"))
        ),
        "signature_min_pair_observations": int(
            slot.get("signature_min_pair_observations", 0)
        ),
        "signature_median_pair_observations": float(
            slot.get("signature_median_pair_observations", 0.0)
        ),
        "signature_empirical_support_valid": bool(
            slot.get("signature_empirical_support_valid", False)
        ),
        "cfg_num_memory_slots": int(cfg["num_memory_slots"]),
        "cfg_routing_mode": str(cfg["periph_routing_mode"]),
        "cfg_signature_mode": str(cfg["periph_signature_mode"]),
        "cfg_lb_coeff": float(cfg["periph_lb_coeff"]),
        "cfg_orth_coeff": float(cfg["periph_orth_coeff"]),
        "cfg_adaptive_k": bool(cfg["belief_adaptive_k"]),
        "cfg_min_core_size": int(cfg["min_core_size"]),
        "cfg_max_core_size": int(cfg["max_core_size"]),
    }
    save_json(os.path.join(out_dir, "summary.json"), summary)

    print(
        f"[H3 {name} seed={seed}] reward={summary['mean_reward']:.3f} "
        f"F1={summary['mean_f1']:.3f} k={summary['mean_core_size']:.2f} "
        f"hard-H={summary['hard_usage_entropy_ratio']:.3f} "
        f"MI={summary['assignment_mutual_info_ratio']:.3f} "
        f"slot-cos={summary['slot_cos_offdiag']:.3f} "
        f"collapse={summary['collapse_detected']}"
    )
    return summary


def _paired_seed_differences(rows, treatment, control, metric):
    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], {})[int(row["seed"])] = row
    treatment_rows = by_variant.get(treatment, {})
    control_rows = by_variant.get(control, {})
    seeds = sorted(set(treatment_rows) & set(control_rows))
    values = []
    for seed in seeds:
        a = float(treatment_rows[seed].get(metric, float("nan")))
        b = float(control_rows[seed].get(metric, float("nan")))
        if np.isfinite(a) and np.isfinite(b):
            values.append(a - b)
    return values


def _paired_gate(rows, treatment, control, metric, lower_is_better):
    differences = _paired_seed_differences(rows, treatment, control, metric)
    finite = np.asarray(
        [value for value in differences if np.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "metric": metric,
            "contrast": f"{treatment} minus {control}",
            "lower_is_better": bool(lower_is_better),
            "paired_seed_count": 0,
            "paired_differences": [],
            "mean_difference": float("nan"),
            "bootstrap_ci95": [float("nan"), float("nan")],
            "supported": False,
        }
    rng_seed = 7_301 + sum(ord(char) for char in f"{treatment}{control}{metric}")
    rng = np.random.RandomState(rng_seed)
    samples = rng.choice(finite, size=(20_000, finite.size), replace=True).mean(axis=1)
    lower, upper = np.percentile(samples, [2.5, 97.5])
    # Three paired seeds are the minimum for a directional experiment. The
    # overnight confirmatory harness uses five; fewer rows remain smoke-only.
    supported = bool(
        finite.size >= 3
        and (float(upper) < 0.0 if lower_is_better else float(lower) > 0.0)
    )
    return {
        "metric": metric,
        "contrast": f"{treatment} minus {control}",
        "lower_is_better": bool(lower_is_better),
        "paired_seed_count": int(finite.size),
        "paired_differences": finite.tolist(),
        "mean_difference": float(np.mean(finite)),
        "bootstrap_ci95": [float(lower), float(upper)],
        "supported": supported,
    }


def _build_hypothesis_gate(rows):
    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)

    full = by_variant.get("Full-CIGAMF", [])
    scalar = by_variant.get("Scalar-Only", [])
    unconstrained = by_variant.get("Unconstrained-NoSemantic", [])
    no_aux = by_variant.get("No-AuxLoss", [])
    fixed = by_variant.get("Fixed-Cardinality", [])
    single = by_variant.get("NoMultiMemory-SingleMean", [])

    arm_protocol = {
        name: bool(arm_rows) and all(
            row["run_complete"]
            and row["signature_protocol_valid"]
            and row["signature_empirical_support_valid"]
            for row in arm_rows
        )
        for name, arm_rows in by_variant.items()
    }

    full_protocol = bool(arm_protocol.get("Full-CIGAMF", False))
    full_anti_collapse = bool(full) and all(
        not row["collapse_detected"]
        and row["hard_usage_entropy_ratio"] >= 0.50
        and row["assignment_mutual_info_ratio"] >= 0.10
        and row["slot_cos_offdiag"] <= 0.95
        for row in full
    )
    fixed_cardinality_valid = bool(fixed) and all(
        row["frac_k_at_kmax"] >= 0.99 and row["core_size_std"] <= 1e-6
        for row in fixed
    )

    full_collapse_rate = _finite_mean(
        [float(row["collapse_detected"]) for row in full]
    )
    unconstrained_collapse_rate = _finite_mean(
        [float(row["collapse_detected"]) for row in unconstrained]
    )
    no_aux_collapse_rate = _finite_mean(
        [float(row["collapse_detected"]) for row in no_aux]
    )

    scalar_outcome = _paired_gate(
        rows, "Full-CIGAMF", "Scalar-Only", "mean_f1", False
    )
    unconstrained_outcome = _paired_gate(
        rows,
        "Full-CIGAMF",
        "Unconstrained-NoSemantic",
        "mean_f1",
        False,
    )
    no_aux_outcome = _paired_gate(
        rows, "Full-CIGAMF", "No-AuxLoss", "mean_f1", False
    )
    fixed_outcome = _paired_gate(
        rows,
        "Full-CIGAMF",
        "Fixed-Cardinality",
        "mean_f1",
        False,
    )
    single_outcome = _paired_gate(
        rows,
        "Full-CIGAMF",
        "NoMultiMemory-SingleMean",
        "mean_f1",
        False,
    )
    single_reward = _paired_gate(
        rows,
        "Full-CIGAMF",
        "NoMultiMemory-SingleMean",
        "mean_reward",
        False,
    )
    single_throughput_advantage = _paired_gate(
        rows,
        "NoMultiMemory-SingleMean",
        "Full-CIGAMF",
        "heldout_agent_states_per_second",
        False,
    )

    full_core_variation = _finite_mean(
        [row["core_size_std"] for row in full]
    )
    fixed_core_variation = _finite_mean(
        [row["core_size_std"] for row in fixed]
    )
    full_hit_cap = _finite_mean([row["frac_k_at_kmax"] for row in full])
    fixed_hit_cap = _finite_mean([row["frac_k_at_kmax"] for row in fixed])
    adaptive_capacity_active = bool(
        fixed_cardinality_valid
        and np.isfinite(full_core_variation)
        and np.isfinite(fixed_core_variation)
        and np.isfinite(full_hit_cap)
        and np.isfinite(fixed_hit_cap)
        and full_core_variation > fixed_core_variation + 1e-6
        and full_hit_cap < fixed_hit_cap - 1e-6
    )

    scalar_gate = bool(
        full_protocol
        and arm_protocol.get("Scalar-Only", False)
        and scalar_outcome["supported"]
    )
    unconstrained_gate = bool(
        full_protocol
        and arm_protocol.get("Unconstrained-NoSemantic", False)
        and full_anti_collapse
        and unconstrained_outcome["supported"]
        and np.isfinite(unconstrained_collapse_rate)
        and unconstrained_collapse_rate > full_collapse_rate
    )
    no_aux_gate = bool(
        full_protocol
        and arm_protocol.get("No-AuxLoss", False)
        and full_anti_collapse
        and no_aux_outcome["supported"]
        and np.isfinite(no_aux_collapse_rate)
        and no_aux_collapse_rate > full_collapse_rate
    )
    fixed_gate = bool(
        full_protocol
        and arm_protocol.get("Fixed-Cardinality", False)
        and adaptive_capacity_active
        and fixed_outcome["supported"]
    )
    single_outcome_gate = bool(
        full_protocol
        and arm_protocol.get("NoMultiMemory-SingleMean", False)
        and single_outcome["supported"]
    )
    specialization_supported = bool(
        scalar_gate and unconstrained_gate and no_aux_gate
    )
    capacity_supported = bool(fixed_gate)
    decision_value_supported = bool(single_outcome_gate)
    h3_supported = bool(
        specialization_supported
        and capacity_supported
        and decision_value_supported
    )
    f1_ci = single_outcome["bootstrap_ci95"]
    single_mean_should_replace = bool(
        single_outcome["paired_seed_count"] >= 5
        and np.isfinite(float(f1_ci[1]))
        and float(f1_ci[1]) <= 0.01
        and single_throughput_advantage["supported"]
    )

    all_variants_present = all(
        bool(by_variant.get(spec["name"])) for spec in VARIANTS
    )
    protocol_complete = bool(rows) and all_variants_present and all(
        arm_protocol.get(spec["name"], False) for spec in VARIANTS
    )
    if not protocol_complete or not full_protocol:
        status = "INVALID_PROTOCOL"
    elif h3_supported:
        status = "SUPPORTED"
    else:
        status = "NOT_SUPPORTED"

    return {
        "schema_version": 4,
        "hypothesis": "H3",
        "protocol_complete": protocol_complete,
        "full_signature_protocol_valid": full_protocol,
        "arm_protocol_valid": {
            spec["name"]: bool(arm_protocol.get(spec["name"], False))
            for spec in VARIANTS
        },
        "full_anti_collapse_gate": full_anti_collapse,
        "diagnostics_source": "common_heldout_probe_after_reset",
        "gates": {
            "full_vs_scalar_signature": {
                "supported": scalar_gate,
                "criterion": "Full has higher matched-run post-warm-up F1 with paired 95% bootstrap CI above zero.",
                "primary_outcome": scalar_outcome,
            },
            "full_vs_unconstrained_routing": {
                "supported": unconstrained_gate,
                "criterion": "Full is non-collapsed, has higher matched-run F1, and the unconstrained arm has a higher collapse rate.",
                "primary_outcome": unconstrained_outcome,
                "full_collapse_rate": full_collapse_rate,
                "control_collapse_rate": unconstrained_collapse_rate,
            },
            "full_vs_no_auxiliary_losses": {
                "supported": no_aux_gate,
                "criterion": "Full is non-collapsed, has higher matched-run F1, and NoAux has a higher collapse rate.",
                "primary_outcome": no_aux_outcome,
                "full_collapse_rate": full_collapse_rate,
                "control_collapse_rate": no_aux_collapse_rate,
            },
            "adaptive_vs_fixed_cardinality": {
                "supported": fixed_gate,
                "criterion": "The fixed arm is constant, Full varies below the cap, and Full has higher matched-run F1.",
                "primary_outcome": fixed_outcome,
                "fixed_control_valid": fixed_cardinality_valid,
                "adaptive_capacity_active": adaptive_capacity_active,
                "full_core_size_std": full_core_variation,
                "fixed_core_size_std": fixed_core_variation,
                "full_fraction_at_cap": full_hit_cap,
                "fixed_fraction_at_cap": fixed_hit_cap,
            },
            "full_vs_single_mean_decision_cost": {
                "supported": single_outcome_gate,
                "criterion": "Full has higher matched-run post-warm-up F1 than the faithful single-mean ablation. Paired reward and held-out throughput are reported as separate trade-offs.",
                "primary_outcome": single_outcome,
                "reward": single_reward,
                "single_mean_throughput_advantage": single_throughput_advantage,
                "outcome_supported": single_outcome_gate,
            },
        },
        "specialization_claim_supported": specialization_supported,
        "capacity_claim_supported": capacity_supported,
        "decision_value_claim_supported": decision_value_supported,
        "h3_claim_supported": h3_supported,
        "h3_claim_status": status,
        "go_no_go": {
            "replace_multi_memory_with_single_mean": single_mean_should_replace,
            "criterion": (
                "With at least five paired seeds, the upper 95% CI for Full "
                "minus SingleMean F1 is at most 0.01 and SingleMean is faster "
                "on common held-out states."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument(
        "--probe-episodes",
        type=int,
        default=3,
        help="Independent held-out episodes shared by every H3 arm.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Absolute path or repository-relative H3 output directory.",
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.eval_every <= 0 or args.probe_episodes <= 0:
        parser.error("--episodes, --eval_every, and --probe-episodes must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    out_root = ensure_dir(_resolve_out_root(args.out_root))
    expected = [
        {"variant": spec["name"], "seed": int(seed)}
        for seed in args.seeds
        for spec in VARIANTS
    ]
    attempt = {
        "schema_version": 4,
        "hypothesis": "H3",
        "status": "running",
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "seeds": [int(seed) for seed in args.seeds],
        "episodes": int(args.episodes),
        "eval_every": int(args.eval_every),
        "probe_episodes": int(args.probe_episodes),
        "device": str(args.device),
        "expected_runs": expected,
        "completed_runs": [],
        "failed_run": None,
    }
    attempt_path = os.path.join(out_root, "attempt.json")
    save_json(attempt_path, attempt)

    rows = []
    current = None
    try:
        for seed in args.seeds:
            probes = _build_heldout_probes(seed, args.probe_episodes)
            for spec in VARIANTS:
                current = {"variant": spec["name"], "seed": int(seed)}
                print(
                    f"\n########## H3 {spec['name']} | seed {seed} ##########"
                )
                rows.append(
                    run_variant(
                        spec,
                        seed,
                        args.episodes,
                        args.eval_every,
                        args.device,
                        out_root,
                        probes,
                    )
                )
                attempt["completed_runs"].append(current)
                save_json(attempt_path, attempt)

        keys = sorted({key for row in rows for key in row})
        csv_path = os.path.join(out_root, "summary_h3.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

        gate = _build_hypothesis_gate(rows)
        save_json(os.path.join(out_root, "hypothesis_gate.json"), gate)

        attempt["status"] = "complete"
        attempt["completed_at_utc"] = _utc_now()
        attempt["summary_csv"] = csv_path
        attempt["hypothesis_gate"] = gate
        save_json(attempt_path, attempt)

        print(f"\n[H3] saved {csv_path}")
        print(
            "[H3] specialization claim: "
            f"{'PASS' if gate['specialization_claim_supported'] else 'FAIL/NULL'}"
        )
        print(
            "[H3] faithful single-mean decision-value claim: "
            f"{'PASS' if gate['decision_value_claim_supported'] else 'FAIL/NULL'}"
        )
    except Exception as exc:
        attempt["status"] = "failed"
        attempt["completed_at_utc"] = _utc_now()
        attempt["failed_run"] = current
        attempt["error_type"] = type(exc).__name__
        attempt["error"] = str(exc)
        save_json(attempt_path, attempt)
        raise


if __name__ == "__main__":
    main()
