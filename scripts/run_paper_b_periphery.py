"""Dedicated Paper-B peripheral encoders under a shared oracle-fixed core."""

import argparse
import copy
import csv
import hashlib
import pickle
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
        _decision_probe, _mean_oracle_capacity, _oracle_capacity_for_state,
        _policy_context_for_state,
    )
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, ensure_dir
    from scripts.run_h2_selectivity import (
        _capture_frozen_learning_checkpoint,
        _restore_frozen_learning_checkpoint,
    )
    from scripts.run_paper_b_allocation import (
        _decision_probe, _mean_oracle_capacity, _oracle_capacity_for_state,
        _policy_context_for_state,
    )

import run_experiment as RE
from runners.h3_ablation_runner import H3NoMultiMemoryRunner
from models.peripheral_memory import PeripheralMultiMemory
from models.single_mean_memory import SingleMeanPeripheral
try:
    from representation_isolation import (
        collect_teacher_trajectory, terminal_states, train_periphery_on_teacher_history,
        teacher_history_hashes, replay_pair_history, collect_aligned_reference_targets,
        probe_periphery_on_teacher_history, reference_targets_as_probe, _reference_target_hash,
    )
except ModuleNotFoundError:
    from scripts.representation_isolation import (
        collect_teacher_trajectory, terminal_states, train_periphery_on_teacher_history,
        teacher_history_hashes, replay_pair_history, collect_aligned_reference_targets,
        probe_periphery_on_teacher_history, reference_targets_as_probe, _reference_target_hash,
    )


VARIANTS = {
    "Full-Explicit": {"runner": "full", "num_memory_slots": 4},
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
    # This is the fidelity control required by the Paper-B memory-budget
    # claim.  Its internal width is selected before construction to match the
    # semantic module's trainable peripheral parameter budget within ±5%; the
    # output width stays fixed, so downstream policy inputs are unchanged.
    "Single-Mean-Matched": {
        "runner": "single", "num_memory_slots": 6,
        "match_semantic_memory": True,
    },
    "Attention-Mean": {
        "runner": "single", "num_memory_slots": 6,
        "periph_beta_mode": "attention",
    },
    "AbsD-Pooling": {
        "runner": "multi", "num_memory_slots": 6,
        "periph_beta_mode": "abs_direction",
    },
}

_MATCHED_DIMENSION_CACHE = {}


def _decision_fidelity(probe, reference):
    return {
        "policy_logit_l2_to_full_explicit": float(np.mean(
            (probe["logits"] - reference["logits"]) ** 2
        ) ** 0.5),
        "value_mae_to_full_explicit": float(np.mean(np.abs(
            probe["values"] - reference["values"]
        ))),
        "action_agreement_to_full_explicit": float(np.mean(
            probe["actions"] == reference["actions"]
        )),
    }


def _probe_state_bank(runner, bank):
    outer = runner.env.clone_state()
    logits, values, actions = [], [], []
    try:
        for state in bank:
            runner.env.restore_state(copy.deepcopy(state))
            selected, cache = runner._select_actions_population(runner.env._get_obs_all())
            logits.append(np.asarray(cache["policy_logits"], dtype=np.float64))
            values.append(np.asarray(
                [cache["value_cache"][agent] for agent in range(runner.n_agents)],
                dtype=np.float64,
            ))
            actions.append(np.argmax(
                np.asarray(cache["policy_logits"], dtype=np.float64), axis=-1
            ).astype(np.int64))
    finally:
        runner.env.restore_state(outer)
    return {"logits": np.stack(logits), "values": np.stack(values), "actions": np.stack(actions)}


def _unique_parameter_bytes(*modules):
    """Count shared parameters once across a runner's trainable modules."""
    seen, total = set(), 0
    for module in modules:
        if module is None or not callable(getattr(module, "parameters", None)):
            continue
        for parameter in module.parameters():
            marker = id(parameter)
            if marker in seen:
                continue
            seen.add(marker)
            total += parameter.numel() * parameter.element_size()
    return int(total)


def _runner_trainable_parameter_bytes(runner):
    modules = [
        getattr(runner, "policy_value", None),
        getattr(runner, "actor", None),
        getattr(runner, "critic", None),
        getattr(runner, "backbone", None),
        getattr(runner, "periph_module", None),
        getattr(runner, "belief_summary_builder", None),
        getattr(runner, "single_periph_proj", None),
        getattr(runner, "heads", None),
    ]
    pair = getattr(runner, "pair_rel_module", None)
    if pair is not None:
        modules.extend([
            getattr(pair, "full_encoder", None),
            getattr(pair, "shadow_encoder", None),
            getattr(pair, "shadow_to_full", None),
            getattr(pair, "bc_head", None),
        ])
    proxy = getattr(runner, "proxy", None)
    if proxy is not None and not bool(getattr(proxy, "use_vmap_ensemble", False)):
        modules.extend(getattr(proxy, "models", ()))
    total = _unique_parameter_bytes(*modules)
    # The vectorised ensemble keeps trainable tensors in functional stacks,
    # not nn.Module.parameters().  Count those tensors once as well.
    if bool(getattr(proxy, "use_vmap_ensemble", False)):
        total += sum(
            tensor.numel() * tensor.element_size()
            for tensor in getattr(proxy, "_stacked_params", {}).values()
        )
    return int(total)


def _persistent_representation_state_bytes(runner):
    """Persistent pair state is distinct from trainable parameter storage."""
    pair = getattr(runner, "pair_rel_module", None)
    if pair is None:
        return 0
    state_stats = pair.get_debug_stats()
    return int(
        state_stats.get("full_state_bytes", 0)
        + state_stats.get("shadow_state_bytes", 0)
        + state_stats.get("pooled_state_bytes", 0)
    )


def _memory_accounting(runner):
    trainable = _runner_trainable_parameter_bytes(runner)
    persistent = _persistent_representation_state_bytes(runner)
    return {
        "trainable_parameter_bytes": int(trainable),
        "persistent_representation_state_bytes": int(persistent),
        # Analytic storage accounting only.  This intentionally excludes
        # optimizer state, replay, temporary tensors and allocator peaks; the
        # scaling panel records runtime peak memory separately.
        "representation_memory_bytes": int(trainable + persistent),
        "representation_memory_accounting_protocol": (
            "analytic_trainable_parameters_plus_persistent_representation_state"
        ),
        "representation_memory_is_runtime_peak": 0,
    }


def _semantic_peripheral_parameter_bytes(cfg, action_dim):
    module = PeripheralMultiMemory(
        action_dim=int(action_dim),
        memory_dim=int(cfg["periph_memory_dim"]),
        out_dim=int(cfg["periph_dim"]),
        num_slots=6,
        routing_mode="semantic",
        signature_mode=cfg.get("periph_signature_mode", "full"),
        require_full_signature=cfg.get("periph_require_full_signature", True),
        allow_legacy_items=cfg.get("periph_allow_legacy_items", False),
        use_uniform_mix=cfg.get("periph_use_uniform_mix", False),
    )
    return _unique_parameter_bytes(module)


def _matched_single_mean_dimensions(cfg, action_dim):
    """Choose a one-mean encoder whose peripheral parameter budget matches.

    The search is deterministic and architecture-local; it never uses
    outcomes or oracle labels.  It therefore defines the control before any
    experimental data are collected.
    """
    cache_key = (
        int(action_dim), int(cfg["periph_memory_dim"]), int(cfg["periph_dim"]),
        str(cfg.get("periph_signature_mode", "full")),
        bool(cfg.get("periph_require_full_signature", True)),
        bool(cfg.get("periph_allow_legacy_items", False)),
    )
    cached = _MATCHED_DIMENSION_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    target = _semantic_peripheral_parameter_bytes(cfg, action_dim)
    best = None
    for memory_dim in range(8, 257, 4):
        for item_hidden in range(16, 257, 8):
            candidate = SingleMeanPeripheral(
                action_dim=int(action_dim), memory_dim=memory_dim,
                out_dim=int(cfg["periph_dim"]), item_hidden=item_hidden,
                signature_mode=cfg.get("periph_signature_mode", "full"),
                require_full_signature=cfg.get("periph_require_full_signature", True),
                allow_legacy_items=cfg.get("periph_allow_legacy_items", False),
            )
            size = _unique_parameter_bytes(candidate)
            error = abs(size - target)
            key = (error, memory_dim * item_hidden, memory_dim, item_hidden)
            if best is None or key < best[0]:
                best = (key, memory_dim, item_hidden, size)
    _, memory_dim, item_hidden, actual = best
    relative_error = abs(actual - target) / max(target, 1)
    if relative_error > 0.05:
        raise RuntimeError(
            "could not construct Single-Mean-Matched within the required "
            f"±5% peripheral parameter budget (target={target}, actual={actual})"
        )
    result = {
        "single_mean_memory_dim": int(memory_dim),
        "single_mean_item_hidden": int(item_hidden),
        "matched_periph_target_parameter_bytes": int(target),
        "matched_periph_parameter_bytes": int(actual),
        "matched_periph_relative_error": float(relative_error),
    }
    _MATCHED_DIMENSION_CACHE[cache_key] = dict(result)
    return result


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


def _state_bank_sha256(bank):
    if not bank:
        raise ValueError("fidelity state bank is empty")
    return hashlib.sha256(
        pickle.dumps(bank, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


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
        "strict_causal_profile": True,
        "semantic_router_frozen": True,
        # The oracle selector is fixed, while proxy/signature/profile learning
        # must remain live.  Freezing graph updates would make the semantic
        # panel scientifically vacuous by suppressing all typed profile work.
        "freeze_graph_updates": False,
        "force_graph_update_every_episode": True,
        "causal_horizon": 1,
        "proxy_n_horizons": 1,
    })
    return cfg


def _env(seed):
    return RE.make_main_env(
        task_mode="behavioral_drift", n_agents=24, max_steps=30,
        phase_length=40, seed=int(seed),
    )


def _common_pretrain_cfg(seed, core_budget):
    """Train the shared proxy before any clone-oracle table exists."""
    cfg = _cfg(seed, core_budget)
    cfg["core_selection_mode"] = "structural_capacity"
    return cfg


def _common_checkpoint(seed, core_budget, device, pretrain_episodes=60):
    RE.set_global_seed(seed)
    runner = RE.make_runner(
        "Final-CIGAMF", _env(seed), _common_pretrain_cfg(seed, core_budget), device
    )
    runner.run(n_episodes=int(pretrain_episodes), eval_every=max(1, int(pretrain_episodes)))
    checkpoint = _capture_frozen_learning_checkpoint(runner)
    checkpoint["pretrain_episodes"] = int(pretrain_episodes)
    return checkpoint


def _restore_shared_state(runner, checkpoint):
    shared = copy.deepcopy(checkpoint)
    shared["state"].pop("periph_module", None)
    shared["optimizer_state"].pop("policy_optim", None)
    _restore_frozen_learning_checkpoint(runner, shared)


def _seed_oracle_core(runner, core_budget, table, full_explicit=False):
    runner.oracle_capacity_scores_by_ego = copy.deepcopy(table)
    for ego, belief in runner.belief_modules.items():
        row = table.get(int(ego), {})
        ranked = sorted(
            belief.neighbor_ids,
            key=lambda neighbor: float(row.get(int(neighbor), 0.0)),
            reverse=True,
        )
        target = list(
            belief.neighbor_ids if full_explicit else ranked[:int(core_budget)]
        )
        # Restoring the common checkpoint also restores its former belief
        # limits.  Without resetting these limits, ``set_fixed_core`` silently
        # truncates a Full-Explicit arm back to the checkpoint's k-core.
        target_size = len(target)
        belief.adaptive_k = False
        belief._configured_min_core_size = target_size
        belief._configured_max_core_size = target_size
        belief.min_core_size = target_size
        belief.max_core_size = target_size
        belief.adaptive_k_min = target_size
        belief.set_fixed_core(target)
        if set(belief.get_core_set()) != set(target):
            raise RuntimeError(
                f"failed to seed exact fixed core for ego={ego}: "
                f"expected={sorted(target)}, actual={sorted(belief.get_core_set())}"
            )
    runner.pair_rel_module.reconcile_core_sets(
        {ego: belief.get_core_set() for ego, belief in runner.belief_modules.items()}
    )


def _oracle_capacity_table(seed, checkpoint, core_budget, device, n_states=2):
    runner = RE.make_runner(
        "Final-CIGAMF", _env(seed), _cfg(seed, core_budget), device
    )
    _restore_frozen_learning_checkpoint(runner, checkpoint)
    runner.env.set_behaviour_override("cooperative")
    bank_seed = int(seed) + 92001
    bank = runner.env.sample_state_bank(
        n_states=int(n_states), burn_in=3, bank_seed=bank_seed,
        min_remaining_steps=int(runner.cfg["causal_horizon"]),
    )
    oracle_bank = []
    for index, state in enumerate(bank):
        policy_context = _policy_context_for_state(runner, state)
        oracle_bank.append(_oracle_capacity_for_state(
            runner.env, state,
            horizon=int(runner.cfg["causal_horizon"]),
            discount=float(runner.cfg["discount"]),
            trials=8,
            seed=bank_seed + index * 100003,
            target_policy_probs=policy_context["policy_probs"],
        ))
    return _mean_oracle_capacity(oracle_bank)


def _run_variant(
    name, seed, episodes, core_budget, device, checkpoint, oracle_capacity
):
    RE.set_global_seed(seed)
    spec = VARIANTS[name]
    cfg = _cfg(seed, core_budget)
    cfg.update({key: value for key, value in spec.items() if key != "runner"})
    env = _env(seed)
    matched_budget = {}
    if bool(spec.get("match_semantic_memory", False)):
        matched_budget = _matched_single_mean_dimensions(
            cfg, action_dim=env.get_action_dim(),
        )
        cfg.update(matched_budget)
    if spec["runner"] == "full":
        cfg["core_selection_mode"] = "full_explicit"
    runner = (
        H3NoMultiMemoryRunner(env, cfg, device=device)
        if spec["runner"] == "single"
        else RE.make_runner("Final-CIGAMF", env, cfg, device)
    )
    _restore_shared_state(runner, checkpoint)
    _seed_oracle_core(
        runner, core_budget, oracle_capacity,
        full_explicit=(spec["runner"] == "full"),
    )
    history = runner.run(n_episodes=int(episodes), eval_every=10)
    core_sizes = [
        len(module.get_core_set()) for module in runner.belief_modules.values()
    ]
    expected_size = (
        runner.n_agents - 1 if spec["runner"] == "full" else int(core_budget)
    )
    if any(size != expected_size for size in core_sizes):
        raise RuntimeError(f"{name}/seed={seed} violated fixed oracle core")
    diagnostics = (
        runner.periph_module.get_slot_diagnostics()
        if callable(getattr(runner.periph_module, "get_slot_diagnostics", None))
        else {}
    )
    row = {
        "panel": "end_to_end_reward",
        "variant": name,
        "seed": int(seed),
        "episodes": int(episodes),
        "pretrain_episodes": int(checkpoint.get("pretrain_episodes", 0)),
        "core_budget": int(core_budget),
        "core_contract": (
            "full_explicit_reference" if spec["runner"] == "full"
            else "oracle_fixed_equal_budget"
        ),
        "checkpoint_sha256": checkpoint["sha256"],
        "mean_core_size": _mean(core_sizes),
        "mean_reward": _mean(history.get("mean_reward", [])),
        "mean_f1": _mean(history.get("mean_f1", [])),
        "throughput_total": _mean(
            history.get("throughput_total_agent_steps_per_sec", [])
        ),
        **_memory_accounting(runner),
        **matched_budget,
        "usage_entropy_ratio": float(
            diagnostics.get("usage_entropy_ratio", -1.0)
        ),
        "slot_cos_offdiag": float(
            diagnostics.get("mean_offdiag_cosine", -1.0)
        ),
    }
    return row, _decision_probe(runner, n_states=4, seed=int(seed) + 94009)


def _representation_isolation_probe(
    name, seed, core_budget, device, checkpoint, oracle_capacity, traces,
    reference_targets,
):
    """Train only peripheral representation against frozen Full-Explicit targets."""
    spec = VARIANTS[name]
    if spec["runner"] == "full":
        # The reference is not a trainable periphery arm.  Returning the exact
        # target history prevents a vacuous empty-periphery "training" pass.
        return reference_targets_as_probe(reference_targets)
    cfg = _cfg(seed, core_budget)
    cfg.update({key: value for key, value in spec.items() if key != "runner"})
    cfg["freeze_downstream_policy_value"] = True
    cfg["freeze_belief_summary_learning"] = True
    cfg["freeze_policy_learning"] = False
    cfg["freeze_graph_updates"] = True
    env = _env(seed)
    if bool(spec.get("match_semantic_memory", False)):
        cfg.update(_matched_single_mean_dimensions(cfg, action_dim=env.get_action_dim()))
    runner = (
        H3NoMultiMemoryRunner(env, cfg, device=device)
        if spec["runner"] == "single"
        else RE.make_runner("Final-CIGAMF", env, cfg, device)
    )
    _restore_shared_state(runner, checkpoint)
    _seed_oracle_core(runner, core_budget, oracle_capacity, full_explicit=False)
    train_periphery_on_teacher_history(runner, traces, reference_targets)
    # Probe exactly the same immutable pre-action rows that supplied the
    # Full-Explicit targets.  No forcing, sampling, V-trace, or state-bank
    # reconstruction enters the fidelity metric.
    return probe_periphery_on_teacher_history(runner, traces)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--pretrain-episodes", type=int, default=60)
    parser.add_argument("--core-budget", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None)
    parser.add_argument(
        "--out-root", default=os.path.join(ROOT, "results", "paper_b_periphery")
    )
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.pretrain_episodes <= 0 or args.core_budget <= 0 or not args.seeds:
        parser.error("episodes, core budget, and seeds must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    out_root = ensure_dir(os.path.abspath(args.out_root))
    rows = []
    selected_variants = args.variants or list(VARIANTS)
    for seed in args.seeds:
        checkpoint = _common_checkpoint(
            seed, args.core_budget, args.device, args.pretrain_episodes
        )
        oracle_capacity = _oracle_capacity_table(
            seed, checkpoint, args.core_budget, args.device
        )
        # The common history teacher must retain a real periphery.  A
        # Full-Explicit teacher has P_i=empty and therefore produces empty
        # periph_inputs_cache, making every "periphery-only" training arm
        # scientifically vacuous.  Use the same oracle-fixed k-core for the
        # immutable teacher history; Full-Explicit remains only the decision
        # fidelity reference.
        teacher_cfg = _cfg(seed, args.core_budget)
        teacher_cfg["core_selection_mode"] = "oracle_capacity"
        teacher_cfg["freeze_policy_learning"] = True
        teacher = RE.make_runner("Final-CIGAMF", _env(seed), teacher_cfg, args.device)
        _restore_shared_state(teacher, checkpoint)
        _seed_oracle_core(teacher, args.core_budget, oracle_capacity, full_explicit=False)
        teacher_traces = collect_teacher_trajectory(teacher, max(1, int(args.episodes)))
        provenance = teacher_history_hashes(teacher_traces)
        if int(provenance["teacher_peripheral_item_count"]) <= 0:
            raise RuntimeError(
                "Paper-B periphery isolation teacher produced zero peripheral items"
            )
        # Build the decision target from a true Full-Explicit reference on the
        # same factual pre-action history.  The reference is collected once per
        # seed and then hashed; every representation arm consumes exactly this
        # immutable target tensor.
        reference_cfg = _cfg(seed, args.core_budget)
        reference_cfg.update({
            "core_selection_mode": "full_explicit",
            "freeze_policy_learning": True,
            "freeze_graph_updates": True,
        })
        reference_runner = RE.make_runner(
            "Final-CIGAMF", _env(seed), reference_cfg, args.device
        )
        _restore_shared_state(reference_runner, checkpoint)
        _seed_oracle_core(
            reference_runner, args.core_budget, oracle_capacity, full_explicit=True
        )
        reference_targets = collect_aligned_reference_targets(
            reference_runner, teacher_traces
        )
        reference_target_hash = _reference_target_hash(reference_targets)
        variants = {}
        for name in selected_variants:
            variants[name] = _run_variant(
                name, seed, args.episodes, args.core_budget, args.device,
                checkpoint, oracle_capacity,
            )
        isolation_probes = {
            name: _representation_isolation_probe(
                name, seed, args.core_budget, args.device, checkpoint,
                oracle_capacity, teacher_traces, reference_targets,
            )
            for name in selected_variants
        }
        reference = reference_targets_as_probe(reference_targets)
        for name, (row, probe) in variants.items():
            if reference is None:
                row.update({
                    "policy_logit_l2_to_full_explicit": float("nan"),
                    "value_mae_to_full_explicit": float("nan"),
                    "action_agreement_to_full_explicit": float("nan"),
                })
                row["decision_fidelity_reference"] = "not_collected"
            else:
                row.update(_decision_fidelity(isolation_probes[name], reference))
                row["decision_fidelity_reference"] = "Full-Explicit"
                row["decision_fidelity_protocol"] = (
                    "common_checkpoint_full_explicit_distillation_on_immutable_pre_action_history"
                )
                row["decision_fidelity_history_steps"] = int(sum(
                    len(trace) for trace in teacher_traces
                ))
            row.update(provenance)
            row["full_explicit_target_sha256"] = reference_target_hash
            row["decision_fidelity_downstream_checkpoint_sha256"] = str(
                checkpoint["sha256"]
            )
            rows.append(row)
    summary_path = os.path.join(out_root, "summary_paper_b_periphery.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_path, "rb") as handle:
        summary_sha256 = hashlib.sha256(handle.read()).hexdigest()
    _atomic_json(os.path.join(out_root, "manifest.json"), {
        "experiment": "paper_b_fixed_core_periphery",
        "complete": True,
        # This runner is an engineering/representation-isolation protocol.  It
        # does not instantiate the information-query oracle required by the
        # current Paper-08 proposal.  Keep these fields explicit and false so
        # a completed plumbing run cannot be mistaken for scientific support.
        "scientific_scope": "periphery_representation_isolation_only",
        "information_channel_gate_pass": False,
        "same_q_opposite_information_gate_pass": False,
        "information_oracle_sha256": None,
        "scientific_grid_ready": False,
        "scientific_limitation": (
            "no information-query oracle or same-Q/opposite-information "
            "intervention is implemented in this runner"
        ),
        "seeds": args.seeds,
        "episodes": args.episodes,
        "pretrain_episodes": args.pretrain_episodes,
        "core_budget": args.core_budget,
        "variants": list(selected_variants),
        "isolation_teacher": "oracle_fixed_k_core_common_history",
        "isolation_teacher_requires_nonempty_periphery": True,
        "isolation_objective": "KL_full_to_variant_plus_value_MSE_plus_periphery_aux",
        "isolation_reference": "aligned_full_explicit_pre_action_targets",
        "memory_budget_control": {
            "variant": "Single-Mean-Matched",
            "target": "Semantic-Free trainable peripheral parameter bytes",
            "tolerance": 0.05,
            "selection": "deterministic architecture-only width search",
        },
        "summary_row_count": int(len(rows)),
        "summary_sha256": summary_sha256,
        "summary": summary_path,
    })
    print(summary_path)


if __name__ == "__main__":
    main()
