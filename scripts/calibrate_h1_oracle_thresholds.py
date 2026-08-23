"""Freeze H1 active/null thresholds from oracle columns on a development split.

This tool deliberately reads only ``oracle_score`` and ``oracle_signed`` from
H1 pair-row CSV files.  It cannot inspect learned Q/C/D estimates, preventing
threshold selection from adapting to estimator performance.  The resulting
JSON is an input to ``run_h1_calibration.py`` and its content hash becomes part
of every confirmatory attempt fingerprint.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import tempfile

import numpy as np


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".h1-thresholds-", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _expand_paths(patterns):
    paths = []
    for pattern in patterns:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded if expanded else [pattern])
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("missing oracle pair-row files: " + ", ".join(missing))
    return sorted(set(os.path.abspath(path) for path in paths))


def _read_oracle_values(paths):
    capacity, direction, seeds = [], [], set()
    sources = []
    for path in paths:
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        sources.append({"path": path, "sha256": digest})
        with open(path, newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            fields = set(rows.fieldnames or ())
            required = {"oracle_score", "oracle_signed", "seed"}
            if not required.issubset(fields):
                raise ValueError(
                    f"{path} lacks oracle-only H1 columns: "
                    f"expected {sorted(required)}, got {sorted(fields)}"
                )
            for row in rows:
                c = float(row["oracle_score"])
                d = float(row["oracle_signed"])
                if not (math.isfinite(c) and math.isfinite(d)):
                    raise ValueError(f"non-finite oracle value in {path}")
                capacity.append(c)
                direction.append(d)
                seeds.add(int(row["seed"]))
    if not capacity:
        raise ValueError("oracle calibration received no pair rows")
    return np.asarray(capacity), np.asarray(direction), sorted(seeds), sources


def _support(values, threshold):
    active = np.abs(values) > float(threshold)
    return {
        "active_count": int(active.sum()),
        "total_count": int(values.size),
        "active_fraction": float(active.mean()),
        "null_count": int((~active).sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle-pair-rows", nargs="+", required=True,
        help="Development-only tiny_oracle_pair_rows.csv path(s) or glob(s).",
    )
    parser.add_argument("--capacity-min-effect", type=float, required=True)
    parser.add_argument("--direction-min-effect", type=float, required=True)
    parser.add_argument("--min-active-count", type=int, default=30)
    parser.add_argument("--min-active-fraction", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.capacity_min_effect < 0 or args.direction_min_effect < 0:
        parser.error("minimum effects must be non-negative")
    if args.min_active_count < 2 or not 0.0 < args.min_active_fraction < 1.0:
        parser.error("invalid active-support requirements")

    paths = _expand_paths(args.oracle_pair_rows)
    capacity, direction, seeds, sources = _read_oracle_values(paths)
    capacity_support = _support(capacity, args.capacity_min_effect)
    direction_support = _support(direction, args.direction_min_effect)
    for name, support in (("capacity", capacity_support), ("direction", direction_support)):
        if (
            support["active_count"] < args.min_active_count
            or support["active_fraction"] < args.min_active_fraction
        ):
            raise ValueError(
                f"{name} threshold leaves inadequate oracle support: {support}; "
                "choose a physically justified lower minimum effect or collect "
                "more development oracle states before freezing."
            )

    payload = {
        "calibration_protocol": "h1_oracle_only_thresholds_v1",
        "oracle_only": True,
        "development_seeds": seeds,
        "sources": sources,
        "capacity_active_threshold": float(args.capacity_min_effect),
        "capacity_prediction_threshold": float(args.capacity_min_effect),
        "direction_active_threshold": float(args.direction_min_effect),
        "direction_prediction_threshold": float(args.direction_min_effect),
        "support": {"capacity": capacity_support, "direction": direction_support},
        "rule": (
            "Thresholds equal prespecified physically meaningful minimum effects; "
            "the oracle-only development sample must retain the recorded minimum "
            "active count and fraction."
        ),
    }
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
