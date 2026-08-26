"""Freeze Page-CUSUM parameters from no-change development trajectories.

Input is a JSON list of standardized residual trajectories produced under a
fixed no-change regime.  The script never accepts structural-change rows, so
the resulting threshold cannot be tuned on detection performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.drift_probe import DriftDetector
try:
    from h2_cusum_contract import (
        CALIBRATION_PROTOCOL, COLLECTION_PROTOCOL, MIN_NO_CHANGE_TRAJECTORIES,
    )
except ModuleNotFoundError:
    from scripts.h2_cusum_contract import (
        CALIBRATION_PROTOCOL, COLLECTION_PROTOCOL, MIN_NO_CHANGE_TRAJECTORIES,
    )


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".cusum-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-change-z-json", required=True)
    parser.add_argument("--allowance", type=float, required=True)
    parser.add_argument("--false-alarm-target", type=float, default=0.05)
    parser.add_argument(
        "--min-trajectories", type=int, default=MIN_NO_CHANGE_TRAJECTORIES,
        help="Minimum independent no-change trajectories required for calibration.",
    )
    parser.add_argument(
        "--development-seeds", type=int, nargs="+", required=True,
        help="No-change development seeds; must be disjoint from confirmatory seeds.",
    )
    parser.add_argument(
        "--reference-config-hash", default=None,
        help="Optional assertion against the no-change artifact configuration hash.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if not args.development_seeds or len(set(args.development_seeds)) != len(args.development_seeds):
        parser.error("--development-seeds must be non-empty and unique")
    if args.min_trajectories <= 0:
        parser.error("--min-trajectories must be positive")
    if args.allowance < 0.0:
        parser.error("--allowance must be non-negative")
    if not 0.0 < args.false_alarm_target < 1.0:
        parser.error("--false-alarm-target must be in (0, 1)")
    with open(args.no_change_z_json, encoding="utf-8") as handle:
        source = json.load(handle)
    if (
        not isinstance(source, dict)
        or source.get("no_change_only") is not True
        or source.get("protocol") != COLLECTION_PROTOCOL
    ):
        raise ValueError(
            "CUSUM calibration input must be a no-change-only JSON artifact "
            "with no_change_only=true"
        )
    sequences = source.get("z_sequences")
    source_config_hash = str(source.get("reference_config_hash", "")).strip()
    if not source_config_hash:
        raise ValueError("no-change artifact omits reference_config_hash")
    if (
        args.reference_config_hash is not None
        and str(args.reference_config_hash).strip() != source_config_hash
    ):
        raise ValueError(
            "--reference-config-hash does not match the no-change residual artifact"
        )
    source_seeds = [int(seed) for seed in source.get("development_seeds", [])]
    if source_seeds != [int(seed) for seed in args.development_seeds]:
        raise ValueError(
            "--development-seeds must exactly match the no-change residual artifact"
        )
    if not isinstance(sequences, list):
        raise ValueError("no-change JSON must contain z_sequences")
    if len(sequences) < int(args.min_trajectories):
        raise ValueError(
            f"CUSUM calibration requires at least {int(args.min_trajectories)} "
            f"independent no-change trajectories; got {len(sequences)}"
        )
    calibration = DriftDetector.calibrate_cusum_from_no_change(
        sequences, args.allowance, args.false_alarm_target
    )
    digest = hashlib.sha256(open(args.no_change_z_json, "rb").read()).hexdigest()
    payload = {
        "calibration_protocol": CALIBRATION_PROTOCOL,
        "no_change_only": True,
        "source": {"path": os.path.abspath(args.no_change_z_json), "sha256": digest},
        "source_protocol": source.get("protocol", "unspecified"),
        "development_seeds": source_seeds,
        "source_checkpoint_sha256_by_seed": source.get(
            "checkpoint_sha256_by_seed", {}
        ),
        "reference_config_hash": source_config_hash,
        **calibration,
    }
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
