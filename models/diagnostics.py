"""
diagnostics.py — three diagnostic experiments still missing from the paper.

=============================================================================
[D1] SELECTIVE RESPONSIVENESS — the paper's headline figure
=============================================================================
The paper's strongest conceptual contribution is separating non-stationarity
into structural (which agents influence which) and behavioural (how agents
currently act) tiers. Standard MARL treatments, including Albrecht et al.
(MIT Press, 2024), treat non-stationarity as one block, making this separation
a substantive distinction.

The paper has not yet measured that distinction directly. Exp. 1
(behavioural drift) and Exp. 2 (structural shift) run separately and are not
contrasted against one another.

The required headline figure is a two-line plot with "belief change" on the
shared vertical axis:
    - under BEHAVIOURAL DRIFT with fixed structure, belief should remain FLAT;
    - under a true STRUCTURAL SHIFT, belief should JUMP;
    - a correlation-based attention baseline should jump under BOTH.
This figure would summarize the complete central claim.

THEORETICAL BASIS (from Pieroth, ICML 2024, Theorem 5.11):
    TIM/SIM is continuous in the policy parameter theta. Smooth behavioural
    drift in theta therefore produces only smooth changes in influence. A
    discontinuous jump in the influence matrix must consequently indicate a
    structural shift. This published theorem provides the mathematical basis
    for separating the two tiers.

=============================================================================
[D2] STRUCTURE SENSITIVITY — experiment required before all others
=============================================================================
The paper's result table reports:
    Pure Mean Field (ignores ALL structure)       : -0.211
    Full Explicit Local (models EVERYTHING)       : -0.196
    difference                                    :  0.015

If complete structural knowledge is worth only 1.5% reward, the maximum
benefit available to any structural method is limited to that range. The
benchmark then provides almost no reward for solving the intended problem.

structure_sensitivity_test() measures this ceiling directly with an
oracle-core baseline. If the ceiling is too low, the environment must be
corrected before the algorithm.

=============================================================================
[D3] PROXY CALIBRATION — evidence for "Causal" in the paper title
=============================================================================
The title uses "Causal" without a validating experiment. Exp. 3 is already
designed in the paper but has no results.

ESTIMAND-ALIGNMENT WARNING (defect identified in v1):
    the v1 proxy computes  mean_a |f(a) - f(a_obs)|   (ALWAYS >= 0)
    the environment oracle computes mean_a (R(a) - R_base)    (SIGNED)
These quantities cannot agree, so Exp. 3 cannot pass even with a correct
algorithm. The function below must use effect_mode="signed_oracle_matched".
=============================================================================
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =========================================================================
# [D1] Selective responsiveness
# =========================================================================

def influence_matrix_from_beliefs(
    belief_modules: Dict[int, object],
    n_agents: int,
    signed: bool = True,
) -> np.ndarray:
    """
    Assemble all ego beliefs into one influence matrix.

    Returns:
        np.ndarray [n_agents, n_agents]
        W[i, j] = influence of j on i, with zeros on the diagonal.
    """
    W = np.zeros((int(n_agents), int(n_agents)), dtype=np.float64)

    for i, mod in belief_modules.items():
        for j in getattr(mod, "neighbor_ids", []):
            if int(i) == int(j):
                continue

            v = float(mod.mu_bar.get(int(j), 0.0))
            W[int(i), int(j)] = v if signed else abs(v)

    return W


def matrix_change(
    W_prev: np.ndarray,
    W_curr: np.ndarray,
    normalise: bool = True,
) -> float:
    """
    Change between two influence matrices under the Frobenius norm.

        d = ||W_curr - W_prev||_F / (||W_prev||_F + eps)

    This metric improves on v1's "temporal variance":
      - v1 reported ~1e-8 and interpreted it as stability. A value of 1e-8 is
        effectively zero: the belief did not move. That indicates freezing,
        not stability.
      - Normalization by ||W_prev|| makes the quantity dimensionless and
        comparable across methods, avoiding artificially small values caused
        only by small mu values.
      - Measuring the entire matrix captures structural changes in who
        influences whom, not just the magnitude of individual edges.
    """
    diff = float(np.linalg.norm(W_curr - W_prev, ord="fro"))

    if not normalise:
        return diff

    base = float(np.linalg.norm(W_prev, ord="fro"))

    return diff / (base + 1e-8)


class SelectiveResponsivenessTracker:
    """
    Track belief responsiveness for each type of non-stationarity.

    Usage:
        tracker = SelectiveResponsivenessTracker()

        # At each evaluation in both environment modes:
        W = influence_matrix_from_beliefs(belief_modules, n_agents)
        tracker.record(condition="behavioural_drift", episode=ep, W=W)
        # ... run separately ...
        tracker.record(condition="structural_shift", episode=ep, W=W)

        # At completion:
        result = tracker.compute_selectivity()
        curves = tracker.get_curves()   # Data for the headline figure.

    EXPECTED BEHAVIOUR FOR A CORRECT METHOD:
        mean_change[behavioural_drift] LOW  (fixed structure: remain stable)
        mean_change[structural_shift]  HIGH (changed structure: respond)
        selectivity_ratio = structural / behavioural  >> 1

    A correlation-based attention baseline should have a ratio near 1 because
    it cannot distinguish the two types. This is the proposition to test.
    """

    def __init__(self, shift_episode: Optional[int] = None):
        self.shift_episode = shift_episode
        self._history: Dict[str, List[Tuple[int, np.ndarray]]] = {}

    def record(self, condition: str, episode: int, W: np.ndarray):
        self._history.setdefault(str(condition), []).append(
            (int(episode), np.asarray(W, dtype=np.float64).copy())
        )

    def get_curves(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Returns:
            {condition: {"episodes": [...], "change": [...]}}

        `change[t]` is the change in W from evaluation t-1 to t. These values
        form the headline figure.
        """
        out = {}

        for cond, entries in self._history.items():
            entries = sorted(entries, key=lambda x: x[0])

            eps_list, changes = [], []

            for t in range(1, len(entries)):
                ep_prev, W_prev = entries[t - 1]
                ep_curr, W_curr = entries[t]

                eps_list.append(int(ep_curr))
                changes.append(float(matrix_change(W_prev, W_curr)))

            out[cond] = {"episodes": eps_list, "change": changes}

        return out

    def compute_selectivity(
        self,
        behavioural_key: str = "behavioural_drift",
        structural_key: str = "structural_shift",
        post_shift_only: bool = True,
    ) -> Dict[str, float]:
        """
        Single statistic summarizing the paper's central claim.

            selectivity_ratio = mean_change(structural) / mean_change(behavioural)

        >> 1  : the method distinguishes the two tiers as claimed
        ~ 1   : no distinction; the central claim is unsupported
        < 1   : inverted response, worse than no adaptation
        """
        curves = self.get_curves()

        def _mean(key: str) -> float:
            if key not in curves or len(curves[key]["change"]) == 0:
                return float("nan")

            eps = np.asarray(curves[key]["episodes"])
            ch = np.asarray(curves[key]["change"])

            if (
                post_shift_only
                and self.shift_episode is not None
                and key == structural_key
            ):
                mask = eps >= int(self.shift_episode)
                if np.any(mask):
                    ch = ch[mask]

            return float(np.mean(ch))

        b = _mean(behavioural_key)
        s = _mean(structural_key)

        ratio = (
            float(s / b)
            if (np.isfinite(b) and abs(b) > 1e-12 and np.isfinite(s))
            else float("nan")
        )

        return {
            "mean_change_behavioural": b,
            "mean_change_structural": s,
            "selectivity_ratio": ratio,
            "interpretation": (
                "GOOD: the two tiers are separated" if (np.isfinite(ratio) and ratio > 1.5)
                else "WEAK: tier separation is unclear" if (np.isfinite(ratio) and ratio > 1.0)
                else "FAIL: no tier separation or reversed response"
            ),
        }


# =========================================================================
# [D2] Structure sensitivity
# =========================================================================

def structure_sensitivity_test(
    run_fn: Callable[[str], float],
    conditions: Sequence[str] = ("pure_mean_field", "oracle_core", "full_explicit"),
    n_seeds: int = 3,
) -> Dict[str, object]:
    """
    This experiment must run before all others.

    Measure the benefit ceiling of structural knowledge in this environment.

    Args:
        run_fn:
            Function receiving a condition name and returning mean reward.
            "oracle_core" must receive the correct core at zero cost without
            learning; it is the theoretical upper bound for every structural
            method.
        n_seeds:
            Number of seeds per condition.

    Returns:
        Dictionary containing per-condition rewards and structure_value.

    INTERPRETATION:
        structure_value = reward(oracle_core) - reward(pure_mean_field)

        If structure_value is small, for example <5% of the reward scale, no
        algorithm can recover a meaningful gain. Correct the environment by
        increasing the influence contrast between bottleneck and ordinary
        agents, narrowing lanes, raising congestion penalties, or reducing
        bypass routes before optimizing the algorithm.

        If structure_value is large, the environment is adequate and the
        remaining gap is algorithmic.
    """
    # [FIX-5] Pass (condition, seed) when run_fn accepts two parameters so both
    # conditions use the same seed list for a paired comparison. The old form
    # passed only condition and allowed the caller to sample a seed, causing
    # branches to use different seeds and mixing between-seed variance into
    # the measured difference.
    import inspect
    try:
        takes_seed = len(inspect.signature(run_fn).parameters) >= 2
    except (TypeError, ValueError):
        takes_seed = False

    seeds = list(range(int(n_seeds)))
    results = {c: [] for c in conditions}
    for cond in conditions:
        for s in seeds:
            results[cond].append(
                float(run_fn(cond, s) if takes_seed else run_fn(cond))
            )

    summary = {
        c: {
            "mean": float(np.mean(v)) if v else float("nan"),
            "std": float(np.std(v)) if v else float("nan"),
            "n": len(v),
            "values": [float(x) for x in v],
        }
        for c, v in results.items()
    }

    # [FIX-5] The old implementation hard-coded "oracle_core". After
    # run_step_0.py renamed the condition to "explicit_local_learned", that
    # branch never ran, leaving structure_value as NaN while still printing a
    # definitive environment warning. Baseline and treatment are now selected
    # by position rather than name.
    conds = list(conditions)
    base_name = "pure_mean_field" if "pure_mean_field" in summary else conds[0]
    treat_name = next((c for c in conds if c != base_name), None)

    val = float("nan")
    ci_lo = ci_hi = float("nan")
    significant = False
    if treat_name is not None:
        a = np.asarray(results[treat_name], dtype=float)
        b = np.asarray(results[base_name], dtype=float)
        if a.size and b.size:
            val = float(a.mean() - b.mean())
            # [FIX-5] Bootstrap CI: the debug specification requires an
            # inconclusive result when the interval contains zero instead of a
            # definitive verdict from a point estimate within noise.
            rng = np.random.RandomState(12345)
            diffs = [
                rng.choice(a, a.size, replace=True).mean()
                - rng.choice(b, b.size, replace=True).mean()
                for _ in range(5000)
            ]
            ci_lo, ci_hi = (float(np.percentile(diffs, 2.5)),
                            float(np.percentile(diffs, 97.5)))
            significant = bool(ci_lo > 0.0 or ci_hi < 0.0)

    if not np.isfinite(val):
        verdict = "NOT MEASURABLE: no valid condition is available."
    elif not significant:
        verdict = (
            f"INCONCLUSIVE: CI95 = [{ci_lo:.3f}, {ci_hi:.3f}] contains zero; "
            "there is insufficient evidence to determine structural sensitivity. "
            "Increase the seed count before changing the environment."
        )
    elif val > 0:
        verdict = (
            f"The environment is structurally sensitive "
            f"(CI95 = [{ci_lo:.3f}, {ci_hi:.3f}]); continue algorithm development."
        )
    else:
        verdict = (
            "WARNING: the treatment is significantly worse than the baseline "
            f"(CI95 = [{ci_lo:.3f}, {ci_hi:.3f}])."
        )

    return {
        "per_condition": summary,
        "baseline_condition": base_name,
        "treatment_condition": treat_name,
        "structure_value": val,
        "ci95": [ci_lo, ci_hi],
        "significant": significant,
        "verdict": verdict,
        # This is not Experiment 0's zero-cost oracle core unless run_fn
        # actually supplies that treatment.
        "note": (
            "structure_value is Experiment 0 only when the treatment is a "
            "zero-cost oracle core with no training. A learned treatment such "
            "as FullExplicitLocal is a baseline comparison, not an upper bound."
        ),
    }


# =========================================================================
# [D3] Proxy calibration
# =========================================================================

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation in NumPy, without a SciPy dependency."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.size < 2:
        return float("nan")

    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)

        # Resolve ties using their mean rank.
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts), dtype=np.float64)
        np.add.at(sums, inv, ranks)

        return (sums / counts)[inv]

    rx, ry = _rank(x), _rank(y)

    sx, sy = np.std(rx), np.std(ry)

    if sx < 1e-12 or sy < 1e-12:
        return float("nan")

    return float(np.mean((rx - np.mean(rx)) * (ry - np.mean(ry))) / (sx * sy))


def proxy_calibration_report(
    proxy_scores: np.ndarray,
    oracle_effects: np.ndarray,
    proxy_scores_baseline: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Exp. 3 — compare the proxy with the true interventional oracle.

    Args:
        proxy_scores:
            [N] proxy scores. These must use
            effect_mode="signed_oracle_matched" to align estimands; see the
            warning at the top of the file.
        oracle_effects:
            [N] true signed intervention effects from
            OracleInterventionSampler.signed_effect().
        proxy_scores_baseline:
            Optional [N] scores from unsigned Pieroth-style
            effect_mode="range". Better performance from the signed variant
            directly supports the claimed novelty.

    Returns:
        Dictionary with bias, rank correlation, sign agreement, and related metrics.

    INTERPRETATION THRESHOLDS:
        stable spearman > 0.3 across seeds supports structural ranking and the
        term "Causal"; spearman near zero makes the proxy unusable and requires
        renaming; sign_agreement > 0.7 supports reliable signed effects and the
        beneficial/harmful slots.
    """
    p = np.asarray(proxy_scores, dtype=np.float64).reshape(-1)
    o = np.asarray(oracle_effects, dtype=np.float64).reshape(-1)

    n = int(min(p.size, o.size))

    if n < 2:
        return {"n": n, "error": "insufficient samples"}

    p, o = p[:n], o[:n]

    bias = float(np.mean(p - o))
    mae = float(np.mean(np.abs(p - o)))

    pearson = float("nan")
    if np.std(p) > 1e-12 and np.std(o) > 1e-12:
        pearson = float(np.corrcoef(p, o)[0, 1])

    spearman = _spearman(p, o)

    # Compute sign agreement only where the oracle is genuinely nonzero.
    mask = np.abs(o) > 1e-8
    sign_agreement = (
        float(np.mean(np.sign(p[mask]) == np.sign(o[mask])))
        if np.any(mask)
        else float("nan")
    )

    out = {
        "n": n,
        "bias": bias,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
        "sign_agreement": sign_agreement,
        "proxy_std": float(np.std(p)),
        "oracle_std": float(np.std(o)),
    }

    if proxy_scores_baseline is not None:
        b = np.asarray(proxy_scores_baseline, dtype=np.float64).reshape(-1)[:n]

        # Compare the unsigned baseline with |oracle| for a fair estimand match.
        out["baseline_spearman_vs_abs_oracle"] = _spearman(b, np.abs(o))
        out["signed_spearman_vs_signed_oracle"] = spearman
        out["signed_beats_unsigned"] = bool(
            np.isfinite(spearman)
            and np.isfinite(out["baseline_spearman_vs_abs_oracle"])
            and abs(spearman) > abs(out["baseline_spearman_vs_abs_oracle"])
        )

    out["verdict"] = (
        "PASS: proxy quality is sufficient for structural ranking"
        if np.isfinite(spearman) and spearman > 0.3
        else "WEAK: correlation is low; more intervention samples are needed"
        if np.isfinite(spearman) and spearman > 0.1
        else "FAIL: proxy is not correlated with true interventions"
    )

    return out


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval. The paper currently contains no statistical
    significance tests.

    This is required because the result table reports:
        Final CIG-AMF  -0.199 +- 0.018
        Pure MF        -0.211 +- 0.008
    The difference is 0.012 while the standard deviation is 0.018 over five
    seeds. This is not a significant difference, so claims such as "moved into
    the strongest reward group" are not yet supported.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)

    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}

    rng = np.random.RandomState(int(seed))
    means = np.array([
        np.mean(rng.choice(v, size=v.size, replace=True))
        for _ in range(int(n_boot))
    ])

    return {
        "mean": float(np.mean(v)),
        "lo": float(np.percentile(means, 100.0 * alpha / 2.0)),
        "hi": float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0))),
        "n": int(v.size),
    }


def compare_two_methods(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Compare two methods by bootstrapping their difference.

    Returns:
        prob_a_better: probability that A exceeds B under the bootstrap distribution.
        significant: True when the confidence interval of the difference excludes zero.

    Report this statistic instead of merely stating that A is higher than B.
    """
    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)

    if a.size == 0 or b.size == 0:
        return {"error": "empty array"}

    rng = np.random.RandomState(int(seed))

    diffs = np.array([
        np.mean(rng.choice(a, a.size, replace=True))
        - np.mean(rng.choice(b, b.size, replace=True))
        for _ in range(int(n_boot))
    ])

    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))

    return {
        "mean_diff": float(np.mean(a) - np.mean(b)),
        "ci_lo": lo,
        "ci_hi": hi,
        "prob_a_better": float(np.mean(diffs > 0.0)),
        "significant": bool(lo > 0.0 or hi < 0.0),
    }
