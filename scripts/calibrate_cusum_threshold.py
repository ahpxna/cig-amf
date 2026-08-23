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

from models.drift_probe import DriftDetector


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
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    with open(args.no_change_z_json, encoding="utf-8") as handle:
        source = json.load(handle)
    sequences = source.get("z_sequences", source) if isinstance(source, dict) else source
    if not isinstance(sequences, list):
        raise ValueError("no-change JSON must be a list or contain z_sequences")
    calibration = DriftDetector.calibrate_cusum_from_no_change(
        sequences, args.allowance, args.false_alarm_target
    )
    digest = hashlib.sha256(open(args.no_change_z_json, "rb").read()).hexdigest()
    payload = {
        "calibration_protocol": "page_cusum_no_change_v1",
        "no_change_only": True,
        "source": {"path": os.path.abspath(args.no_change_z_json), "sha256": digest},
        **calibration,
    }
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
