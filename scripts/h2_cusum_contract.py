"""Canonical Page-CUSUM null-calibration contract for Paper A/H2.

The contract is deliberately seed-independent: each development seed owns its
own common pretraining checkpoint, while all checkpoints must be produced by
the same monitored protocol.  The calibration artifact binds to this exact
contract and separately records the per-seed checkpoint digests.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

CONTRACT_PROTOCOL = "h2_cusum_null_contract_v3_residual_statistic"
COLLECTION_PROTOCOL = "cusum_no_change_residual_collection_v3_fixed_horizon"
CALIBRATION_PROTOCOL = "page_cusum_no_change_v3_provenance"
MIN_NO_CHANGE_TRAJECTORIES = 40


def build_h2_cusum_contract(
    cfg,
    *,
    n_agents: int,
    max_steps: int,
    pretrain_episodes: int,
    evaluation_roles,
    manipulated_roles,
):
    """Return the seed-independent configuration that defines the H2 null."""
    return {
        "contract_protocol": CONTRACT_PROTOCOL,
        "model": "Final-CIGAMF",
        "environment_family": "OmniArena",
        "null_task_mode": "no_change",
        "n_agents": int(n_agents),
        "max_steps": int(max_steps),
        "pretrain_episodes": int(pretrain_episodes),
        "causal_horizon": int(cfg["causal_horizon"]),
        "proxy_n_horizons": int(cfg["proxy_n_horizons"]),
        "discount": float(cfg["discount"]),
        "freeze_policy_learning": bool(cfg.get("freeze_policy_learning", False)),
        "freeze_representation_state": bool(
            cfg.get("freeze_representation_state", False)
        ),
        "behavioral_adapter_lambda": float(
            cfg.get("behavioral_adapter_lambda", 0.0)
        ),
        "behavioral_adapter_only_in_behavioral_drift": bool(
            cfg.get("behavioral_adapter_only_in_behavioral_drift", False)
        ),
        "behavioral_adapter_target_roles": sorted(
            str(role) for role in (cfg.get("behavioral_adapter_target_roles") or [])
        ),
        "evaluation_ego_roles": sorted(str(role) for role in evaluation_roles),
        "manipulated_neighbor_roles": sorted(str(role) for role in manipulated_roles),
        "drift_warmup_batches": int(cfg["drift_warmup_batches"]),
        "drift_train_batches": int(cfg["drift_train_batches"]),
        "drift_batch_size": int(cfg.get("drift_batch_size", 256)),
        "drift_window": int(cfg.get("drift_window", 20)),
        "drift_monitoring_statistic": "newest_complete_batch_mean_abs_discounted_return_residual",
        "drift_recalibrate_after": int(cfg["drift_recalibrate_after"]),
        "proxy_effect_mode": str(cfg["proxy_effect_mode"]),
        "proxy_use_belief_input": bool(cfg.get("proxy_use_belief_input", False)),
    }


def contract_hash(contract) -> str:
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value) -> bool:
    value = str(value or "").strip().lower()
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_calibration_artifact(
    calibration,
    *,
    expected_reference_config_hash: str | None = None,
    min_trajectories: int = MIN_NO_CHANGE_TRAJECTORIES,
    verify_source: bool = True,
):
    """Fail closed on malformed, stale, or protocol-mismatched calibration."""
    if not isinstance(calibration, dict):
        raise ValueError("CUSUM calibration must be a JSON object")
    required = {
        "calibration_protocol",
        "no_change_only",
        "source",
        "source_protocol",
        "cusum_allowance",
        "cusum_threshold",
        "target_false_alarm_rate",
        "observed_false_alarm_rate",
        "development_seeds",
        "reference_config_hash",
        "n_no_change_trajectories",
        "monitoring_horizon",
        "source_checkpoint_sha256_by_seed",
    }
    missing = sorted(required.difference(calibration))
    if missing:
        raise ValueError("CUSUM calibration is incomplete; missing " + ", ".join(missing))
    if calibration["calibration_protocol"] != CALIBRATION_PROTOCOL:
        raise ValueError("CUSUM calibration protocol is incompatible")
    if calibration["source_protocol"] != COLLECTION_PROTOCOL:
        raise ValueError("CUSUM source protocol is incompatible")
    if calibration["no_change_only"] is not True:
        raise ValueError("CUSUM calibration must be no-change-only")

    allowance = float(calibration["cusum_allowance"])
    threshold = float(calibration["cusum_threshold"])
    target = float(calibration["target_false_alarm_rate"])
    observed = float(calibration["observed_false_alarm_rate"])
    if not math.isfinite(allowance) or allowance < 0.0:
        raise ValueError("CUSUM allowance must be finite and non-negative")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("CUSUM threshold must be finite and strictly positive")
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("CUSUM false-alarm target must lie in (0, 1)")
    if not math.isfinite(observed) or not 0.0 <= observed <= 1.0:
        raise ValueError("CUSUM observed false-alarm rate must lie in [0, 1]")

    seeds = calibration["development_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or len({int(seed) for seed in seeds}) != len(seeds)
    ):
        raise ValueError("CUSUM development seeds must be non-empty and unique")
    checkpoint_hashes = calibration["source_checkpoint_sha256_by_seed"]
    if (
        not isinstance(checkpoint_hashes, dict)
        or set(checkpoint_hashes) != {str(int(seed)) for seed in seeds}
        or not all(_valid_sha256(value) for value in checkpoint_hashes.values())
    ):
        raise ValueError(
            "CUSUM calibration must preserve one valid source checkpoint SHA-256 per development seed"
        )

    n_trajectories = int(calibration["n_no_change_trajectories"])
    if n_trajectories < int(min_trajectories):
        raise ValueError(
            f"CUSUM calibration requires at least {int(min_trajectories)} "
            f"no-change trajectories; got {n_trajectories}"
        )
    if int(calibration["monitoring_horizon"]) <= 0:
        raise ValueError("CUSUM monitoring horizon must be positive")

    reference_hash = str(calibration["reference_config_hash"]).strip().lower()
    if not _valid_sha256(reference_hash):
        raise ValueError("CUSUM reference_config_hash is not a SHA-256 digest")
    if (
        expected_reference_config_hash is not None
        and reference_hash != str(expected_reference_config_hash).strip().lower()
    ):
        raise ValueError("CUSUM calibration does not match the current H2 null contract")

    source = calibration["source"]
    if not isinstance(source, dict) or not _valid_sha256(source.get("sha256")):
        raise ValueError("CUSUM calibration source must contain a SHA-256 digest")
    if verify_source:
        path = os.path.abspath(str(source.get("path") or ""))
        if not os.path.isfile(path):
            raise ValueError(f"CUSUM calibration source is missing: {path}")
        if _sha256_file(path) != str(source["sha256"]).strip().lower():
            raise ValueError("CUSUM calibration source SHA-256 mismatch")

    maxima = calibration.get("maxima", [])
    if maxima and not np.isfinite(np.asarray(maxima, dtype=np.float64)).all():
        raise ValueError("CUSUM calibration maxima contain non-finite values")
    return calibration
