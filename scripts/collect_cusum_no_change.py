"""Collect frozen-witness standardized residuals under a no-change regime.

The output is accepted only by ``calibrate_cusum_threshold.py``.  Both
structural and behavioural environment factors are disabled, and the artifact
contains raw standardized residual z trajectories rather than CUSUM values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

try:
    from exp_common import ROOT, ensure_dir
except ModuleNotFoundError:
    from scripts.exp_common import ROOT, ensure_dir

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_experiment as RE
try:
    from h2_cusum_contract import build_h2_cusum_contract, contract_hash, COLLECTION_PROTOCOL
    from run_h2_selectivity import (
        H2_EVALUATION_EGO_ROLES, H2_MANIPULATED_NEIGHBOR_ROLES,
        _pretrain_common_checkpoint, _restore_frozen_learning_checkpoint,
    )
except ModuleNotFoundError:
    from scripts.h2_cusum_contract import (
        build_h2_cusum_contract, contract_hash, COLLECTION_PROTOCOL,
    )
    from scripts.run_h2_selectivity import (
        H2_EVALUATION_EGO_ROLES, H2_MANIPULATED_NEIGHBOR_ROLES,
        _pretrain_common_checkpoint, _restore_frozen_learning_checkpoint,
    )


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".cusum-no-change-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _config(seed):
    cfg = RE.default_cfg()
    cfg.update({
        "seed": int(seed),
        "behavioral_adapter_lambda": 0.0,
        "behavioral_adapter_only_in_behavioral_drift": False,
        "behavioral_adapter_target_roles": list(H2_MANIPULATED_NEIGHBOR_ROLES),
        "freeze_policy_learning": True,
        "freeze_representation_state": True,
    })
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--pretrain-episodes", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be non-empty and unique")
    if (
        args.episodes <= 0 or args.pretrain_episodes <= 0
        or args.max_steps <= 0 or args.eval_every <= 0
    ):
        parser.error("episode, pretrain, step, and evaluation budgets must be positive")

    sequences, contract_hashes, checkpoint_hashes = [], [], {}
    for seed in args.seeds:
        cfg = _config(seed)
        contract = build_h2_cusum_contract(
            cfg, n_agents=24, max_steps=int(args.max_steps),
            pretrain_episodes=int(args.pretrain_episodes),
            evaluation_roles=H2_EVALUATION_EGO_ROLES,
            manipulated_roles=H2_MANIPULATED_NEIGHBOR_ROLES,
        )
        contract_hashes.append(contract_hash(contract))
        checkpoint = _pretrain_common_checkpoint(
            "Final-CIGAMF", int(seed), int(args.pretrain_episodes), args.device,
            cusum_calibration=None,
        )
        checkpoint_hashes[str(int(seed))] = str(checkpoint["sha256"])
        env = RE.make_main_env(
            task_mode="no_change",
            n_agents=24,
            max_steps=int(args.max_steps),
            phase_length=max(int(args.episodes) + 1, 100000),
            seed=int(seed),
            structural_factor=False,
            behavioral_factor=False,
        )
        runner = RE.make_runner("Final-CIGAMF", env, cfg, args.device)
        _restore_frozen_learning_checkpoint(runner, checkpoint)
        if hasattr(runner, "drift") and not runner.drift.is_monitoring_ready():
            raise RuntimeError("restored H2 null witness is not monitoring-ready")
        history = runner.run(
            n_episodes=int(args.episodes), eval_every=int(args.eval_every)
        )
        raw_z = history.get("scheduler_residual_ewma", [])
        ready = history.get("drift_monitoring_ready", [])
        sequence = [float(z) for z, is_ready in zip(raw_z, ready) if int(is_ready)]
        if not sequence:
            raise RuntimeError(
                f"no monitoring-ready residuals were collected for seed {seed}; "
                "increase --episodes or adjust the frozen-witness warmup"
            )
        sequences.append(sequence)

    if len(set(contract_hashes)) != 1:
        raise RuntimeError("no-change residual runs used inconsistent H2 null contracts")
    _atomic_json(args.out, {
        "protocol": COLLECTION_PROTOCOL,
        "no_change_only": True,
        "development_seeds": [int(seed) for seed in args.seeds],
        "pretrain_episodes": int(args.pretrain_episodes),
        "checkpoint_sha256_by_seed": checkpoint_hashes,
        "reference_config_hash": contract_hashes[0],
        "contract": contract,
        "monitoring_horizon": int(max(len(sequence) for sequence in sequences)),
        "z_sequences": sequences,
    })
    print(json.dumps({"out": os.path.abspath(args.out)}, indent=2))


if __name__ == "__main__":
    main()
