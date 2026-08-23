"""Dedicated Paper-B pair-latent ablation under an oracle-fixed core."""

import argparse
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
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, ensure_dir
    from scripts.run_h2_selectivity import (
        _capture_frozen_learning_checkpoint,
        _restore_frozen_learning_checkpoint,
    )

import run_experiment as RE
from models.ego_conditioned_latent import pair_specificity_score


VARIANTS = {
    "Aggregate": {"pair_state_mode": "aggregate", "heads_w_influence": 0.0,
                  "heads_w_contrastive": 0.0},
    "Explicit-FF-BC": {"pair_state_mode": "feedforward", "heads_w_influence": 0.0,
                       "heads_w_contrastive": 0.0},
    "Recurrent-BC": {"pair_state_mode": "recurrent", "heads_w_influence": 0.0,
                     "heads_w_contrastive": 0.0},
    "Recurrent-BC-CD": {"pair_state_mode": "recurrent", "heads_w_influence": 1.0,
                        "heads_w_contrastive": 0.0},
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


def _run_variant(name, seed, episodes, core_budget, device, checkpoint):
    RE.set_global_seed(seed)
    cfg = _base_cfg(seed, core_budget)
    cfg.update(VARIANTS[name])
    runner = RE.make_runner("Final-CIGAMF", _env(seed), cfg, device)
    _restore_frozen_learning_checkpoint(runner, checkpoint)
    runner.pair_rel_module.state_mode = cfg["pair_state_mode"]
    history = runner.run(n_episodes=int(episodes), eval_every=10)
    specificity = pair_specificity_score(runner.pair_rel_module, runner.n_agents)
    core_sizes = [
        len(module.get_core_set()) for module in runner.belief_modules.values()
    ]
    if any(size != int(core_budget) for size in core_sizes):
        raise RuntimeError(f"{name}/seed={seed} violated fixed oracle core")
    return {
        "variant": name,
        "seed": int(seed),
        "episodes": int(episodes),
        "core_budget": int(core_budget),
        "core_contract": "oracle_fixed_equal_budget",
        "checkpoint_sha256": checkpoint["sha256"],
        "pair_state_mode": cfg["pair_state_mode"],
        "w_cd": float(cfg["heads_w_influence"]),
        "w_contrastive": float(cfg["heads_w_contrastive"]),
        "mean_reward": _mean(history.get("mean_reward", [])),
        "mean_f1": _mean(history.get("mean_f1", [])),
        "throughput_total": _mean(
            history.get("throughput_total_agent_steps_per_sec", [])
        ),
        "pair_specificity_ratio": float(specificity["specificity_ratio"]),
        "cd_retrieval_mae": _cd_retrieval_mae(runner),
    }


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
        for name in selected_variants:
            rows.append(_run_variant(
                name, seed, args.episodes, args.core_budget, args.device,
                checkpoint,
            ))
    summary_path = os.path.join(out_root, "summary_paper_b_pair_latent.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
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
