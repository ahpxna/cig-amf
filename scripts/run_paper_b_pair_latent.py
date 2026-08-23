"""Dedicated Paper-B pair-latent ablation under an oracle-fixed core."""

import argparse
import copy
import csv
import json
import os
import tempfile

import numpy as np
import torch

try:
    from exp_common import ROOT, ensure_dir
    from run_h2_selectivity import (
        _capture_frozen_learning_checkpoint,
        _restore_frozen_learning_checkpoint,
    )
    from run_paper_b_allocation import (
        _decision_probe, _mean_oracle_capacity, _oracle_capacity_for_state,
    )
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, ensure_dir
    from scripts.run_h2_selectivity import (
        _capture_frozen_learning_checkpoint,
        _restore_frozen_learning_checkpoint,
    )
    from scripts.run_paper_b_allocation import (
        _decision_probe, _mean_oracle_capacity, _oracle_capacity_for_state,
    )

import run_experiment as RE
from models.ego_conditioned_latent import pair_specificity_score


VARIANTS = {
    "Full-Explicit-Reference": {
        "pair_state_mode": "recurrent", "heads_w_influence": 1.0,
        "heads_w_contrastive": 1.0, "full_explicit_reference": True,
    },
    "Shared-Neighbor-State": {"pair_state_mode": "aggregate", "heads_w_influence": 0.0,
                  "heads_w_contrastive": 0.0},
    "Pooled-Neighbor": {"pair_state_mode": "pooled", "heads_w_influence": 0.0,
                  "heads_w_contrastive": 0.0},
    "Explicit-FF-BC": {"pair_state_mode": "feedforward", "heads_w_influence": 0.0,
                       "heads_w_contrastive": 0.0},
    "Recurrent-BC": {"pair_state_mode": "recurrent", "heads_w_influence": 0.0,
                     "heads_w_contrastive": 0.0},
    "Recurrent-BC-CD": {"pair_state_mode": "recurrent", "heads_w_influence": 1.0,
                        "heads_w_contrastive": 0.0, "promotion_panel": True},
    "Recurrent-BC-CD-NoWarmStart": {
        "pair_state_mode": "recurrent", "heads_w_influence": 1.0,
        "heads_w_contrastive": 0.0, "pair_warm_start": False,
        "promotion_panel": True,
    },
    "Recurrent-BC-CD-Contrastive": {
        "pair_state_mode": "recurrent", "heads_w_influence": 1.0,
        "heads_w_contrastive": 1.0,
    },
}


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(
        prefix=".tmp-pair-latent-", suffix=".json", dir=directory
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


def _base_cfg(seed, core_budget):
    cfg = RE.default_cfg()
    cfg.update({
        "seed": int(seed),
        "k0_warmup": 0,
        "core_selection_mode": "oracle_capacity",
        "belief_adaptive_k": False,
        "min_core_size": int(core_budget),
        "belief_adaptive_k_min": int(core_budget),
        "max_core_size": int(core_budget),
        "periph_require_full_signature": True,
        "periph_allow_legacy_items": False,
        # Oracle-fixed representation panels must not silently reselect the
        # core on a scheduler tick.  Pair state and BC learning remain active.
        "freeze_graph_updates": True,
    })
    return cfg


def _env(seed):
    return RE.make_main_env(
        task_mode="behavioral_drift", n_agents=24, max_steps=30,
        phase_length=40, seed=int(seed),
    )


def _initial_checkpoint(seed, core_budget, device):
    RE.set_global_seed(seed)
    runner = RE.make_runner(
        "Final-CIGAMF", _env(seed), _base_cfg(seed, core_budget), device
    )
    checkpoint = _capture_frozen_learning_checkpoint(runner)
    return checkpoint


def _cd_retrieval_mae(runner):
    latents, targets = [], []
    for ego, belief in runner.belief_modules.items():
        for neighbor in belief.neighbor_ids:
            latents.append(runner.pair_rel_module.get_pair_latent(ego, neighbor))
            targets.append([
                belief.debiased_mu(neighbor),
                runner.sig_tracker.get_signature(ego, neighbor)[1],
            ])
    if not latents:
        return float("nan")
    target = np.asarray(targets, dtype=np.float32)
    target_norm = (
        target - runner.pair_rel_module.cd_norm_mean.reshape(1, 2)
    ) / runner.pair_rel_module.cd_norm_std.reshape(1, 2)
    with torch.no_grad():
        prediction = runner.heads.cd_head(torch.as_tensor(
            np.asarray(latents), dtype=torch.float32, device=runner.device
        )).cpu().numpy()
    return float(np.mean(np.abs(prediction - target_norm)))


def _latent_profile_geometry(runner):
    latents, profiles = [], []
    for ego, belief in runner.belief_modules.items():
        for neighbor in belief.neighbor_ids:
            latents.append(runner.pair_rel_module.get_pair_latent(ego, neighbor))
            profiles.append([
                belief.debiased_mu(neighbor),
                runner.sig_tracker.get_signature(ego, neighbor)[1],
            ])
    if len(latents) < 3:
        return float("nan")
    z = np.asarray(latents, dtype=np.float64)
    p = np.asarray(profiles, dtype=np.float64)
    p = (p - runner.pair_rel_module.cd_norm_mean.reshape(1, 2)) / (
        runner.pair_rel_module.cd_norm_std.reshape(1, 2) + 1e-8
    )
    upper = np.triu_indices(len(z), k=1)
    dz = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)[upper]
    dp = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)[upper]
    return float(RE.safe_spearman(dz, dp)[0])


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
    """Evaluate a runner on an already sampled, shared clone-state bank."""
    outer = runner.env.clone_state()
    logits, values, actions = [], [], []
    try:
        for state in bank:
            runner.env.restore_state(copy.deepcopy(state))
            obs = runner.env._get_obs_all()
            selected, cache = runner._select_actions_population(obs)
            logits.append(np.asarray(cache["policy_logits"], dtype=np.float64))
            values.append(np.asarray(
                [cache["value_cache"][ego] for ego in range(runner.n_agents)],
                dtype=np.float64,
            ))
            actions.append(np.asarray(
                [selected[ego] for ego in range(runner.n_agents)], dtype=np.int64,
            ))
    finally:
        runner.env.restore_state(outer)
    return {
        "logits": np.stack(logits, axis=0),
        "values": np.stack(values, axis=0),
        "actions": np.stack(actions, axis=0),
    }


def _post_promotion_bc_loss(history):
    promoted = list(history.get("promoted", []))
    losses = list(history.get("bc_loss", []))
    values = [float(loss) for loss, count in zip(losses, promoted) if int(count) > 0]
    # Keep the CSV finite when a fixed-core run never promotes a pair. The
    # event count is reported separately and the validator marks that outcome
    # unsupported rather than treating it as a malformed artifact.
    return _mean(values) if values else 0.0


def _oracle_capacity_table(seed, checkpoint, core_budget, device, n_states=2):
    runner = RE.make_runner(
        "Final-CIGAMF", _env(seed), _base_cfg(seed, core_budget), device
    )
    _restore_frozen_learning_checkpoint(runner, checkpoint)
    runner.env.set_behaviour_override("cooperative")
    bank_seed = int(seed) + 91001
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


def _set_oracle_fixed_core(runner, oracle_capacity, core_budget, *, full=False):
    """Install a deterministic capacity-ranked core allocation."""
    mapping = {}
    for ego, belief in runner.belief_modules.items():
        if full:
            belief.max_core_size = len(belief.neighbor_ids)
        ranked = sorted(
            belief.neighbor_ids,
            key=lambda neighbor: float(
                oracle_capacity.get(int(ego), {}).get(int(neighbor), 0.0)
            ),
            reverse=True,
        )
        mapping[int(ego)] = set(
            belief.neighbor_ids if full else ranked[:int(core_budget)]
        )
        belief.set_fixed_core(mapping[int(ego)])
    runner.pair_rel_module.reconcile_core_sets(
        mapping, warm_start=bool(runner.cfg.get("pair_warm_start", True))
    )
    return mapping


def _promotion_mapping(runner, oracle_capacity, core_budget):
    """Swap one explicit pair per ego for a known non-core candidate.

    This is an explicit intervention panel, not a side effect of adaptive
    selection.  Every valid ego contributes exactly one promotion whenever a
    non-core candidate exists.
    """
    mapping, events = {}, []
    for ego, belief in runner.belief_modules.items():
        current = set(belief.get_core_set())
        ranked = sorted(
            belief.neighbor_ids,
            key=lambda neighbor: float(
                oracle_capacity.get(int(ego), {}).get(int(neighbor), 0.0)
            ),
            reverse=True,
        )
        incoming = next((j for j in ranked if j not in current), None)
        if incoming is None or not current:
            mapping[int(ego)] = current
            continue
        outgoing = min(
            current,
            key=lambda neighbor: float(
                oracle_capacity.get(int(ego), {}).get(int(neighbor), 0.0)
            ),
        )
        updated = set(current)
        updated.remove(outgoing)
        updated.add(incoming)
        if len(updated) != int(core_budget):
            raise RuntimeError("promotion panel changed the fixed core budget")
        mapping[int(ego)] = updated
        events.append((int(ego), int(outgoing), int(incoming)))
    return mapping, events


def _full_explicit_reference_from_checkpoint(seed, core_budget, device, checkpoint):
    cfg = _base_cfg(seed, core_budget)
    cfg.update({
        "pair_state_mode": "recurrent",
        "freeze_policy_learning": True,
        "freeze_graph_updates": True,
    })
    runner = RE.make_runner("Final-CIGAMF", _env(seed), cfg, device)
    _restore_frozen_learning_checkpoint(runner, checkpoint)
    _set_oracle_fixed_core(runner, {}, core_budget, full=True)
    return runner


def _promotion_panel(
    runner, seed, episodes, core_budget, device, oracle_capacity,
    initial_checkpoint,
):
    """Evaluate shadow warm-start on a prespecified core promotion.

    The policy is frozen for the transient.  Only pair state and BC learning
    evolve, and every post-promotion measurement is compared with a
    Full-Explicit reference on the same clone-state bank.
    """
    transient_steps = max(1, min(10, int(episodes) // 4))
    pre_steps = max(0, int(episodes) - transient_steps)
    runner.cfg["freeze_policy_learning"] = True
    runner.cfg["freeze_pair_bc_learning"] = False
    runner.cfg["freeze_graph_updates"] = True
    # The reference starts from the same checkpoint and runs throughout the
    # pre-promotion interval with every pair explicit.  It is therefore a
    # genuine full-model trajectory, not a post-hoc expansion of a sparse
    # state dictionary at the intervention boundary.
    reference = _full_explicit_reference_from_checkpoint(
        seed, core_budget, device, initial_checkpoint
    )
    reference.cfg["freeze_pair_bc_learning"] = True
    reference.run(n_episodes=pre_steps, eval_every=10)
    # Reference execution advances process RNG state; restore the candidate
    # branch point before its matched pre-promotion trajectory.
    _restore_frozen_learning_checkpoint(runner, initial_checkpoint)
    runner.oracle_capacity_scores_by_ego = oracle_capacity
    runner.pair_rel_module.state_mode = "recurrent"
    _set_oracle_fixed_core(runner, oracle_capacity, core_budget, full=False)
    runner.cfg["freeze_policy_learning"] = True
    runner.cfg["freeze_pair_bc_learning"] = False
    runner.cfg["freeze_graph_updates"] = True
    pre_history = runner.run(n_episodes=pre_steps, eval_every=10)
    bank_seed = int(seed) + 95021
    bank = runner.env.sample_state_bank(n_states=4, burn_in=3, bank_seed=bank_seed)
    reference_probe = _probe_state_bank(reference, bank)
    mapping, events = _promotion_mapping(runner, oracle_capacity, core_budget)
    for ego, core in mapping.items():
        runner.belief_modules[ego].set_fixed_core(core)
    runner.pair_rel_module.reconcile_core_sets(
        mapping, warm_start=bool(runner.cfg.get("pair_warm_start", True)),
    )
    if events and not any(
        (ego, incoming) in runner.pair_rel_module.active_core_pairs
        for ego, _outgoing, incoming in events
    ):
        raise RuntimeError("promotion panel failed to allocate promoted full states")
    series, history = [], {key: list(value) for key, value in pre_history.items()}
    for _ in range(transient_steps):
        step_history = runner.run(n_episodes=1, eval_every=1)
        for key, values in step_history.items():
            history.setdefault(key, []).extend(values)
        fidelity = _decision_fidelity(_probe_state_bank(runner, bank), reference_probe)
        fidelity["bc_loss"] = float(_mean(step_history.get("bc_loss", [])))
        series.append(fidelity)
    def auc(key):
        return _mean([row[key] for row in series])
    return history, {
        "promotion_event_count": int(len(events)),
        "promotion_episode": int(pre_steps),
        "promotion_transient_horizon": int(transient_steps),
        "promotion_logit_error_auc": auc("policy_logit_l2_to_full_explicit"),
        "promotion_value_error_auc": auc("value_mae_to_full_explicit"),
        "promotion_action_agreement_auc": auc("action_agreement_to_full_explicit"),
        "promotion_bc_loss_auc": auc("bc_loss"),
        "promotion_reference": "full_explicit_frozen_clone_state_bank",
    }


def _run_variant(
    name, seed, episodes, core_budget, device, checkpoint, oracle_capacity
):
    RE.set_global_seed(seed)
    cfg = _base_cfg(seed, core_budget)
    cfg.update(VARIANTS[name])
    if bool(cfg.get("full_explicit_reference", False)):
        cfg["core_selection_mode"] = "full_explicit"
    runner = RE.make_runner("Final-CIGAMF", _env(seed), cfg, device)
    _restore_frozen_learning_checkpoint(runner, checkpoint)
    runner.oracle_capacity_scores_by_ego = oracle_capacity
    runner.pair_rel_module.state_mode = cfg["pair_state_mode"]
    _set_oracle_fixed_core(
        runner, oracle_capacity, core_budget,
        full=bool(cfg.get("full_explicit_reference", False)),
    )
    promotion = {}
    if bool(cfg.get("promotion_panel", False)):
        history, promotion = _promotion_panel(
            runner, seed, episodes, core_budget, device, oracle_capacity,
            checkpoint,
        )
    else:
        history = runner.run(n_episodes=int(episodes), eval_every=10)
    specificity = pair_specificity_score(runner.pair_rel_module, runner.n_agents)
    core_sizes = [
        len(module.get_core_set()) for module in runner.belief_modules.values()
    ]
    expected_size = (
        runner.n_agents - 1 if bool(cfg.get("full_explicit_reference", False))
        else int(core_budget)
    )
    if any(size != expected_size for size in core_sizes):
        raise RuntimeError(f"{name}/seed={seed} violated fixed oracle core")
    row = {
        "variant": name,
        "seed": int(seed),
        "episodes": int(episodes),
        "core_budget": int(core_budget),
        "core_contract": (
            "full_explicit_reference"
            if bool(cfg.get("full_explicit_reference", False))
            else "oracle_fixed_equal_budget"
        ),
        "checkpoint_sha256": checkpoint["sha256"],
        "pair_state_mode": cfg["pair_state_mode"],
        "w_cd": float(cfg["heads_w_influence"]),
        "w_contrastive": float(cfg["heads_w_contrastive"]),
        "pair_warm_start": bool(cfg.get("pair_warm_start", True)),
        "mean_reward": _mean(history.get("mean_reward", [])),
        "mean_f1": _mean(history.get("mean_f1", [])),
        "throughput_total": _mean(
            history.get("throughput_total_agent_steps_per_sec", [])
        ),
        "pair_specificity_ratio": float(specificity["specificity_ratio"]),
        "latent_profile_distance_spearman": _latent_profile_geometry(runner),
        "cd_retrieval_mae": _cd_retrieval_mae(runner),
        "promotion_event_count": int(promotion.get(
            "promotion_event_count", sum(history.get("promoted", []))
        )),
        "post_promotion_bc_loss": float(promotion.get(
            "promotion_bc_loss_auc", _post_promotion_bc_loss(history)
        )),
        **promotion,
    }
    return row, _decision_probe(runner, n_states=4, seed=int(seed) + 93011)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--core-budget", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None)
    parser.add_argument(
        "--out-root", default=os.path.join(ROOT, "results", "paper_b_pair_latent")
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
        checkpoint = _initial_checkpoint(seed, args.core_budget, args.device)
        oracle_capacity = _oracle_capacity_table(
            seed, checkpoint, args.core_budget, args.device
        )
        variants = {}
        for name in selected_variants:
            variants[name] = _run_variant(
                name, seed, args.episodes, args.core_budget, args.device,
                checkpoint, oracle_capacity,
            )
        reference_name = "Full-Explicit-Reference"
        reference = variants.get(reference_name, (None, None))[1]
        for name, (row, probe) in variants.items():
            if reference is None:
                row.update({
                    "policy_logit_l2_to_full_explicit": float("nan"),
                    "value_mae_to_full_explicit": float("nan"),
                    "action_agreement_to_full_explicit": float("nan"),
                })
                row["decision_fidelity_reference"] = "not_collected"
            else:
                row.update(_decision_fidelity(probe, reference))
                row["decision_fidelity_reference"] = reference_name
            rows.append(row)
    summary_path = os.path.join(out_root, "summary_paper_b_pair_latent.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(os.path.join(out_root, "manifest.json"), {
        "experiment": "paper_b_pair_latent",
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
