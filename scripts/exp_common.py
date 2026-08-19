"""Helper for scripts/run_h*.py — do not run directly."""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


def make_args(**over):
    """Create a fully populated argparse.Namespace of run_experiment.parse_args()."""
    import run_experiment as RE
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        args = RE.parse_args()
    finally:
        sys.argv = argv
    for k, v in over.items():
        setattr(args, k, v)
    return args


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=float) + "\n")


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def w_matrix(runner, n_agents):
    """W(t): matrix mu_bar debiased from belief modules (Eq 33 uses it)."""
    W = np.zeros((n_agents, n_agents), dtype=np.float64)
    for ego, mod in runner.belief_modules.items():
        try:
            for j, st in mod.get_state_dict().items():
                W[int(ego), int(j)] = float(st.get("mu_bar", 0.0))
        except Exception:
            pass
    return W


def delta_norm(W_prev, W_new, eps=1e-8, min_ref=1e-3):
    """Eq (33): ||W(t)-W(t-1)||_F / (||W(t-1)||_F + eps).

    [FIX-9] During warm-up, belief is still at prior μ=0 so ||W(t-1)||_F ≈ 0 and
    the denominator drops to eps=1e-8 => Δ blows up to millions (log H2 true:
    delta_behav = 9.08e6). That is not "belief strong change" but division by near-zero — exactly the class of error encountered in T5/T6/structure_value.

    Return NaN when the reference matrix has no content (||W_prev|| < min_ref):
    Δ at that point is undefined, and NaN will be excluded from statistics instead of
    pulling the mean. The paper (§D, Eq 33) also clearly states the purpose of normalization is
    to make the quantity dimensionless — it only makes sense when there is a scale to normalize against."""
    ref = float(np.linalg.norm(W_prev))
    if ref < min_ref:
        return float("nan")
    return float(np.linalg.norm(W_new - W_prev) / (ref + eps))


def mean_over_egos(runner, method, key=None):
    vals = []
    for mod in runner.belief_modules.values():
        try:
            v = getattr(mod, method)()
            if key is not None:
                v = v.get(key, np.nan)
            vals.append(float(v))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float("nan")


def last(hist, key, default=float("nan")):
    v = hist.get(key, [])
    return float(v[-1]) if len(v) else default
