"""Certify that an oracle-only H1 development bank can identify Q, C, and D.

This gate is intentionally evaluated before fitting or reading any learned
estimator result.  A failed gate means the benchmark is not informative for a
claim; it is never recorded as an estimator failure.
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
    fd, temporary = tempfile.mkstemp(prefix=".h1-support-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _paths(patterns):
    values = []
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        values.extend(found if found else [pattern])
    values = sorted(set(os.path.abspath(value) for value in values))
    missing = [value for value in values if not os.path.isfile(value)]
    if missing:
        raise FileNotFoundError("missing H1 oracle row files: " + ", ".join(missing))
    return values


def _read(paths):
    capacities, directions, q_nonconstant, q_levels = [], [], [], []
    sources = []
    for path in paths:
        sources.append({"path": path, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()})
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"oracle_score", "oracle_signed", "oracle_q_nonconstant", "oracle_q_distinct_levels"}
            if not required.issubset(set(reader.fieldnames or ())):
                raise ValueError(f"{path} lacks support columns: {sorted(required)}")
            for row in reader:
                capacities.append(float(row["oracle_score"]))
                directions.append(float(row["oracle_signed"]))
                q_nonconstant.append(int(row["oracle_q_nonconstant"]))
                q_levels.append(int(row["oracle_q_distinct_levels"]))
    array_c = np.asarray(capacities, dtype=np.float64)
    array_d = np.asarray(directions, dtype=np.float64)
    if not array_c.size or not np.isfinite(array_c).all() or not np.isfinite(array_d).all():
        raise ValueError("oracle support data is empty or non-finite")
    return array_c, array_d, np.asarray(q_nonconstant), np.asarray(q_levels), sources


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-pair-rows", nargs="+", required=True)
    parser.add_argument("--capacity-threshold", type=float, required=True)
    parser.add_argument("--direction-threshold", type=float, required=True)
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--min-fraction", type=float, default=0.05)
    parser.add_argument("--min-capacity-levels", type=int, default=3)
    parser.add_argument("--h1-target-policy-mode", required=True)
    parser.add_argument("--h1-eval-uniform-mass", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    paths = _paths(args.oracle_pair_rows)
    c, d, q_nonconstant, q_levels, sources = _read(paths)
    active_c = c > float(args.capacity_threshold)
    active_d = np.abs(d) > float(args.direction_threshold)
    c_levels = np.unique(np.round(c[active_c], 12))
    checks = {
        "nonconstant_q_surfaces": int(q_nonconstant.sum()) >= args.min_count,
        "oracle_null_capacity_support": int((~active_c).sum()) >= args.min_count,
        "active_capacity_support": int(active_c.sum()) >= args.min_count and float(active_c.mean()) >= args.min_fraction,
        "capacity_rank_variation": int(c_levels.size) >= args.min_capacity_levels,
        "active_direction_support": int(active_d.sum()) >= args.min_count and float(active_d.mean()) >= args.min_fraction,
        "positive_direction_support": int((d > args.direction_threshold).sum()) >= max(2, args.min_count // 10),
        "negative_direction_support": int((d < -args.direction_threshold).sum()) >= max(2, args.min_count // 10),
        "pi_q_separation": (
            args.h1_target_policy_mode == "scripted_uniform_mixture"
            and 0.0 < float(args.h1_eval_uniform_mass) < 1.0
        ),
    }
    payload = {
        "protocol": "h1_oracle_support_v1",
        "oracle_only": True,
        "sources": sources,
        "thresholds": {"capacity": args.capacity_threshold, "direction": args.direction_threshold},
        "h1_target_policy_mode": args.h1_target_policy_mode,
        "h1_eval_uniform_mass": args.h1_eval_uniform_mass,
        "counts": {
            "total_pairs": int(c.size), "nonconstant_q": int(q_nonconstant.sum()),
            "capacity_null": int((~active_c).sum()), "capacity_active": int(active_c.sum()),
            "capacity_active_levels": c_levels.tolist(),
            "direction_active": int(active_d.sum()),
            "direction_positive": int((d > args.direction_threshold).sum()),
            "direction_negative": int((d < -args.direction_threshold).sum()),
            "max_q_distinct_action_levels": int(np.max(q_levels)) if q_levels.size else 0,
        },
        "checks": checks,
        "benchmark_identifiable": bool(all(checks.values())),
        "decision": "PASS" if all(checks.values()) else "BENCHMARK_NOT_IDENTIFIABLE",
    }
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
