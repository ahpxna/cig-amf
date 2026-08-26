"""Shared fail-closed helpers for the G0--G9 scientific gate ladder."""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

GATE_SCHEMA_VERSION = 2


def json_safe(value):
    """Convert gate payloads to strict JSON without hiding missing evidence.

    A null/undefined metric is a normal outcome for a failed scientific gate
    (for example, no non-constant oracle Q surface). Serialising it as JSON
    ``null`` preserves that distinction; emitting IEEE NaN turns a scientific
    failure into an operational crash.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def finite(values):
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def bootstrap_mean_ci(values, *, seed=1729, n_bootstrap=5000):
    vals = np.asarray(finite(values), dtype=np.float64)
    if vals.size == 0:
        return [float("nan"), float("nan")]
    if vals.size == 1:
        value = float(vals[0])
        return [value, value]
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, vals.size, size=(int(n_bootstrap), vals.size))
    means = vals[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def wilson_interval(successes, trials, z=1.959963984540054):
    """Two-sided Wilson score interval for a binomial proportion.

    Gate G2/G7 use Bernoulli false-positive/false-alarm events.  Bootstrapping
    already-aggregated per-seed rates gives the wrong sampling unit and can
    become overconfident when seed denominators differ, so those gates use the
    exact event counts and this bounded score interval instead.
    """
    n = int(trials)
    k = int(successes)
    if n <= 0 or k < 0 or k > n:
        return [float("nan"), float("nan")]
    p = float(k / n)
    z2 = float(z) ** 2
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (float(z) / denom) * math.sqrt(
        p * (1.0 - p) / n + z2 / (4.0 * n * n)
    )
    return [max(0.0, centre - half), min(1.0, centre + half)]


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".gate-", suffix=".json.tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, default=float, allow_nan=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_json(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} artifact is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} artifact must be a JSON object")
    return payload


def load_csv(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} summary is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{label} summary is empty")
    return rows


def gate_record(gate, passed, *, required, metrics=None, rule="", failure_action=""):
    return {
        "gate": str(gate),
        "passed": bool(passed),
        "required": bool(required),
        "metrics": metrics or {},
        "rule": str(rule),
        "failure_action": str(failure_action),
    }
