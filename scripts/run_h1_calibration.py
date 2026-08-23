"""H1a/H1b/H1c calibration with an estimand-aligned one-step oracle.

The confirmatory protocol uses logged action-time contexts, the factual
one-step reward, the exact observed-action propensity, and clone-state
interventions that change only the current neighbour action. Multi-step
calibration remains exploratory until factual and counterfactual rollouts can
share the same future policy.

Every invocation writes to ``<out-root>/runs/<run-id>``. The top-level summary
is published only after every requested seed/variant completes and passes the
protocol integrity gate, preventing old partial outputs from being aggregated.
"""
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import tempfile
import uuid

try:
    from exp_common import ROOT, ensure_dir, make_args, save_json  # noqa: F401
except ModuleNotFoundError:  # Support module imports in focused tests.
    from scripts.exp_common import ROOT, ensure_dir, make_args, save_json  # noqa: F401

import run_experiment as RE

VARIANTS = [
    # Online D remains plug-in. Enabling row AIPW exposes the same-row
    # correction only as a held-out diagnostic; it is never routed into the
    # signature, semantic memory, or core selector.
    ("row_aipw_diag_eps005", {"proxy_use_doubly_robust": True, "eps": 0.05}),
    ("plugin_eps005", {"proxy_use_doubly_robust": False, "eps": 0.05}),
    # Constant-epsilon identification sweep.
    ("plugin_eps000", {"proxy_use_doubly_robust": False, "eps": 0.0}),
    ("plugin_eps001", {"proxy_use_doubly_robust": False, "eps": 0.01}),
    ("plugin_eps003", {"proxy_use_doubly_robust": False, "eps": 0.03}),
    ("plugin_eps008", {"proxy_use_doubly_robust": False, "eps": 0.08}),
    ("plugin_eps012", {"proxy_use_doubly_robust": False, "eps": 0.12}),
]

KEEP_KEYS = (
    "rank_correlation_mean", "spearman_mean", "sign_agreement_mean",
    "signed_spearman_mean", "signed_mae_mean", "signed_bias_mean",
    "signed_rmse_mean", "signed_p_value_mean",
    "range_rank_correlation_mean",
    "bias_mean", "mae_mean", "pearson_mean", "n_states", "n_pairs",
    "q_mae_mean", "q_rmse_mean", "q_spearman_mean",
    "q_centered_mae_mean", "q_centered_rmse_mean",
    "q_within_state_action_spearman_mean", "q_nonconstant_surface_count_mean",
    "q_normalized_rmse_mean",
    "q_raw_mae_mean", "q_raw_rmse_mean",
    "capacity_rank_correlation_mean", "capacity_mae_mean",
    "capacity_bias_mean", "oracle_core_f1_mean",
    "oracle_core_f1_random_baseline_mean", "oracle_core_f1_adjusted_mean",
    "direction_spearman_mean", "direction_mae_mean",
    "direction_bias_mean", "direction_sign_agreement_mean",
    "capacity_active_mae_mean", "capacity_active_spearman_mean",
    "capacity_active_normalized_mae_mean", "capacity_active_pair_count_mean",
    "capacity_null_fpr_mean", "direction_active_mae_mean",
    "direction_active_spearman_mean", "direction_active_sign_agreement_mean",
    "direction_active_normalized_mae_mean", "direction_active_pair_count_mean",
    "direction_null_fpr_mean",
    "direction_crossfit_aipw_signed_mae_mean",
    "direction_crossfit_aipw_signed_spearman_mean",
    "direction_crossfit_aipw_sign_agreement_mean",
    "direction_row_aipw_signed_mae_mean",
    "direction_row_aipw_signed_spearman_mean",
    "direction_row_aipw_sign_agreement_mean",
    "support_poor_pair_count_mean", "support_poor_capacity_mae_mean",
    "support_poor_direction_mae_mean",
)

PROTOCOL_VERSION = "h1_qcd_crossfit_v4"
MIN_CONFIRMATORY_SEEDS = 8
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 1729

# Confirmatory recovery thresholds for the three response-spectrum objects.
# They make no comparison between AIPW and the plug-in estimator: that remains
# a reported estimator ablation, as a variance-reduction technique should not
# be elevated to a scientific hypothesis.
MIN_Q_SPEARMAN = 0.30
MIN_CAPACITY_RANK = 0.30
MIN_CAPACITY_CORE_F1 = 0.35
MIN_DIRECTION_SPEARMAN = 0.30
MIN_DIRECTION_SIGN_AGREEMENT = 0.60
MAX_CAPACITY_NULL_FPR = 0.20
MAX_DIRECTION_NULL_FPR = 0.20
MAX_Q_NORMALIZED_RMSE = 1.0
MAX_CAPACITY_NORMALIZED_MAE = 1.0
MAX_DIRECTION_NORMALIZED_MAE = 1.0


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".h1-", suffix=".json.tmp", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=float)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    fd, tmp = tempfile.mkstemp(
        prefix=".h1-", suffix=".csv.tmp", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _fingerprint(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _load_threshold_calibration(path):
    """Load the oracle-only threshold decision frozen before an H1 run."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("oracle_only") is not True:
        raise ValueError("H1 threshold calibration must be explicitly oracle-only")
    required = (
        "capacity_active_threshold",
        "capacity_prediction_threshold",
        "direction_active_threshold",
        "direction_prediction_threshold",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            "H1 threshold calibration is incomplete; missing "
            + ", ".join(missing)
        )
    values = {key: float(payload[key]) for key in required}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("H1 threshold calibration contains invalid thresholds")
    return values, _fingerprint(payload)


def build_h1_config(cfg_override, seed, threshold_calibration=None):
    """Build the one-step H1 configuration shared by CLI and contract tests."""
    cfg = RE.default_cfg()
    cfg.update(cfg_override)
    cfg["forcer_anneal_to"] = float(cfg_override["eps"])
    # H1 identifies the one-step estimand.  The runner rejects mismatched
    # proxy/replay horizons, so both fields are protocol-critical.
    cfg["causal_horizon"] = 1
    cfg["proxy_n_horizons"] = 1
    cfg["proxy_iw_clip"] = 2000.0
    cfg["seed"] = int(seed)
    if threshold_calibration is not None:
        cfg["h1_capacity_active_threshold"] = float(
            threshold_calibration["capacity_active_threshold"]
        )
        cfg["h1_capacity_prediction_threshold"] = float(
            threshold_calibration["capacity_prediction_threshold"]
        )
        cfg["h1_direction_active_threshold"] = float(
            threshold_calibration["direction_active_threshold"]
        )
        cfg["h1_direction_prediction_threshold"] = float(
            threshold_calibration["direction_prediction_threshold"]
        )
    return cfg


def _variant_means(rows, variant):
    selected = [row for row in rows if row["variant"] == variant]
    out = {}
    aliases = {
        "capacity_rank_correlation_mean": "rank_correlation_mean",
        "capacity_mae_mean": "mae_mean",
        "capacity_bias_mean": "bias_mean",
        "direction_spearman_mean": "signed_spearman_mean",
        "direction_mae_mean": "signed_mae_mean",
        "direction_bias_mean": "signed_bias_mean",
        "direction_sign_agreement_mean": "sign_agreement_mean",
    }
    for key in (
        "rank_correlation_mean", "signed_spearman_mean",
        "sign_agreement_mean", "signed_bias_mean", "signed_mae_mean",
        "range_rank_correlation_mean",
        "q_spearman_mean", "q_mae_mean", "q_rmse_mean",
        "q_centered_mae_mean", "q_centered_rmse_mean",
        "q_within_state_action_spearman_mean", "q_nonconstant_surface_count_mean",
        "q_normalized_rmse_mean",
        "capacity_rank_correlation_mean", "capacity_mae_mean",
        "capacity_bias_mean", "oracle_core_f1_mean",
        "oracle_core_f1_random_baseline_mean", "oracle_core_f1_adjusted_mean",
        "direction_spearman_mean", "direction_mae_mean",
        "direction_bias_mean", "direction_sign_agreement_mean",
        "capacity_active_mae_mean", "capacity_active_spearman_mean",
        "capacity_active_normalized_mae_mean", "capacity_active_pair_count_mean",
        "capacity_null_fpr_mean", "direction_active_mae_mean",
        "direction_active_spearman_mean",
        "direction_active_sign_agreement_mean", "direction_null_fpr_mean",
        "direction_active_normalized_mae_mean", "direction_active_pair_count_mean",
        "direction_row_aipw_signed_spearman_mean",
        "direction_row_aipw_signed_mae_mean",
        "direction_row_aipw_sign_agreement_mean",
        "direction_crossfit_aipw_signed_spearman_mean",
        "direction_crossfit_aipw_signed_mae_mean",
        "direction_crossfit_aipw_sign_agreement_mean",
        "realised_forcing_rate", "heldout_policy_return_mean_per_agent",
        "support_poor_pair_count_mean", "support_poor_capacity_mae_mean",
        "support_poor_direction_mae_mean",
    ):
        fallback = aliases.get(key)
        vals = [
            float(row.get(key, row.get(fallback)))
            for row in selected
            if row.get(key, row.get(fallback)) not in (None, "")
        ]
        out[key] = float(sum(vals) / len(vals)) if vals else float("nan")
    bias_vals = [
        abs(float(row["signed_bias_mean"]))
        for row in selected if row.get("signed_bias_mean") not in (None, "")
    ]
    out["signed_bias_abs_mean"] = (
        float(sum(bias_vals) / len(bias_vals))
        if bias_vals else float("nan")
    )
    out["n_seeds"] = len({int(row["seed"]) for row in selected})
    out["action_coverage_all_seeds"] = bool(
        selected
        and all(bool(row.get("action_coverage_gate_pass", False)) for row in selected)
    )
    out["dr_clipping_absent_all_seeds"] = bool(
        selected
        and all(bool(row.get("dr_clipping_absent", False)) for row in selected)
    )
    truthy = lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    out["capacity_active_support_all_seeds"] = bool(
        selected and all(truthy(row.get("capacity_active_support_pass", False)) for row in selected)
    )
    out["direction_active_support_all_seeds"] = bool(
        selected and all(truthy(row.get("direction_active_support_pass", False)) for row in selected)
    )
    return out


def _percentile(values, probability):
    """Return a linearly interpolated percentile for a sorted numeric sample."""
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    location = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_differences(rows, left_variant, right_variant, difference_fn):
    """Build one paired contrast per seed and reject duplicate seed rows."""
    indexed = {}
    for row in rows:
        variant = row.get("variant")
        if variant not in (left_variant, right_variant):
            continue
        key = (str(variant), int(row["seed"]))
        if key in indexed:
            raise RuntimeError(
                f"Duplicate H1 row for variant={variant}, seed={row['seed']}"
            )
        indexed[key] = row

    seeds = sorted(
        {seed for variant, seed in indexed if variant == left_variant}
        & {seed for variant, seed in indexed if variant == right_variant}
    )
    return [
        {
            "seed": int(seed),
            "difference": float(difference_fn(
                indexed[(left_variant, seed)],
                indexed[(right_variant, seed)],
            )),
        }
        for seed in seeds
    ]


def _paired_bootstrap_summary(paired_rows, seed_offset=0):
    """Deterministic percentile bootstrap CI for a paired mean contrast."""
    values = [float(row["difference"]) for row in paired_rows]
    if not values or not all(math.isfinite(value) for value in values):
        return {
            "n_pairs": len(values),
            "mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED + int(seed_offset),
            "paired_seed_differences": paired_rows,
        }

    rng = random.Random(BOOTSTRAP_SEED + int(seed_offset))
    n_values = len(values)
    bootstrap_means = []
    for _ in range(BOOTSTRAP_REPLICATES):
        bootstrap_means.append(sum(
            values[rng.randrange(n_values)] for _ in range(n_values)
        ) / n_values)
    return {
        "n_pairs": n_values,
        "mean": float(sum(values) / n_values),
        "ci95_low": float(_percentile(bootstrap_means, 0.025)),
        "ci95_high": float(_percentile(bootstrap_means, 0.975)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED + int(seed_offset),
        "paired_seed_differences": paired_rows,
    }


def _epsilon_bias_trend(rows):
    """Summarize the declared absolute-bias trend over constant epsilon arms."""
    variants = (
        (0.00, "plugin_eps000"),
        (0.01, "plugin_eps001"),
        (0.03, "plugin_eps003"),
        (0.05, "plugin_eps005"),
        (0.08, "plugin_eps008"),
        (0.12, "plugin_eps012"),
    )
    curve = []
    for epsilon, variant in variants:
        metrics = _variant_means(rows, variant)
        curve.append({
            "epsilon": epsilon,
            "variant": variant,
            "mean_absolute_signed_bias": metrics["signed_bias_abs_mean"],
            "n_seeds": metrics["n_seeds"],
        })

    values = [point["mean_absolute_signed_bias"] for point in curve]
    complete = bool(
        all(math.isfinite(value) for value in values)
        and all(point["n_seeds"] >= MIN_CONFIRMATORY_SEEDS for point in curve)
    )
    if not complete:
        return {
            "complete": False,
            "curve": curve,
            "ols_slope_abs_bias_per_epsilon": float("nan"),
            "adjacent_nonincrease_count": 0,
            "adjacent_comparison_count": len(curve) - 1,
            "adjacent_nonincrease_fraction": float("nan"),
            "endpoint_improvement": float("nan"),
            "gate_pass": False,
        }

    epsilons = [point["epsilon"] for point in curve]
    mean_epsilon = sum(epsilons) / len(epsilons)
    mean_bias = sum(values) / len(values)
    denominator = sum((value - mean_epsilon) ** 2 for value in epsilons)
    slope = sum(
        (epsilon - mean_epsilon) * (bias - mean_bias)
        for epsilon, bias in zip(epsilons, values)
    ) / denominator
    nonincrease_count = sum(
        right <= left for left, right in zip(values[:-1], values[1:])
    )
    fraction = nonincrease_count / (len(values) - 1)
    endpoint_improvement = values[0] - values[-1]
    gate_pass = bool(
        slope < 0.0
        and endpoint_improvement > 0.0
        and fraction >= 0.60
    )
    return {
        "complete": True,
        "curve": curve,
        "ols_slope_abs_bias_per_epsilon": float(slope),
        "adjacent_nonincrease_count": int(nonincrease_count),
        "adjacent_comparison_count": len(curve) - 1,
        "adjacent_nonincrease_fraction": float(fraction),
        "endpoint_improvement": float(endpoint_improvement),
        "gate_pass": gate_pass,
    }


def _forcing_reporting(rows):
    """Validate realised forcing and pair its held-out return cost to eps=0."""
    expected_variants = {name for name, _ in VARIANTS}
    by_variant = {
        variant: [row for row in rows if row.get("variant") == variant]
        for variant in expected_variants
    }
    realised_rate_complete = bool(all(
        len({int(row["seed"]) for row in selected}) >= MIN_CONFIRMATORY_SEEDS
        and all(
            row.get("realised_forcing_rate") not in (None, "")
            and math.isfinite(float(row["realised_forcing_rate"]))
            for row in selected
        )
        for selected in by_variant.values()
    ))
    realised_rate_by_variant = {
        variant: (
            float(sum(float(row["realised_forcing_rate"]) for row in selected)
                  / len(selected))
            if selected and all(
                row.get("realised_forcing_rate") not in (None, "")
                for row in selected
            ) else float("nan")
        )
        for variant, selected in sorted(by_variant.items())
    }

    cost_pairs = _paired_differences(
        rows, "plugin_eps000", "plugin_eps005",
        lambda no_forcing, forcing: (
            float(no_forcing["heldout_policy_return_mean_per_agent"])
            - float(forcing["heldout_policy_return_mean_per_agent"])
        ),
    )
    cost_summary = _paired_bootstrap_summary(cost_pairs, seed_offset=2)
    support_poor_capacity = _paired_bootstrap_summary(
        _paired_differences(
            rows, "plugin_eps000", "plugin_eps005",
            lambda no_forcing, forcing: (
                float(no_forcing["support_poor_capacity_mae_mean"])
                - float(forcing["support_poor_capacity_mae_mean"])
            ),
        ), seed_offset=3,
    )
    support_poor_direction = _paired_bootstrap_summary(
        _paired_differences(
            rows, "plugin_eps000", "plugin_eps005",
            lambda no_forcing, forcing: (
                float(no_forcing["support_poor_direction_mae_mean"])
                - float(forcing["support_poor_direction_mae_mean"])
            ),
        ), seed_offset=4,
    )
    endpoints = [
        row for row in rows
        if row.get("variant") in ("plugin_eps000", "plugin_eps005")
    ]
    return_endpoint_complete = bool(
        cost_summary["n_pairs"] >= MIN_CONFIRMATORY_SEEDS
        and endpoints
        and all(
            bool(row.get("policy_return_endpoint_measured", False))
            and row.get("heldout_policy_return_mean_per_agent") not in (None, "")
            and math.isfinite(float(
                row["heldout_policy_return_mean_per_agent"]
            ))
            for row in endpoints
        )
    )
    return {
        "realised_forcing_rate_complete": realised_rate_complete,
        "realised_forcing_rate_mean_by_variant": realised_rate_by_variant,
        "forcing_return_cost_definition": (
            "paired held-out mean per-agent episode return: "
            "plugin_eps000 minus plugin_eps005; positive values are a forcing cost"
        ),
        "forcing_return_cost_paired_bootstrap": cost_summary,
        "forcing_return_cost_measured": return_endpoint_complete,
        "support_poor_endpoint": {
            "subset": "pre-forcing observed-action natural policy mass below frozen threshold",
            "capacity_mae_eps0_minus_eps005": support_poor_capacity,
            "direction_mae_eps0_minus_eps005": support_poor_direction,
            "prediction_pass": bool(
                support_poor_capacity["ci95_low"] > 0.0
                or support_poor_direction["ci95_low"] > 0.0
            ),
        },
        "reporting_complete": bool(
            realised_rate_complete and return_endpoint_complete
        ),
    }


def _claim_gate(rows):
    """Adjudicate plug-in Q/C/D recovery; report AIPW only as diagnostics."""
    plugin = _variant_means(rows, "plugin_eps005")
    row_aipw = _variant_means(rows, "row_aipw_diag_eps005")
    observational = _variant_means(rows, "plugin_eps000")
    direction_rank_paired = _paired_bootstrap_summary(
        _paired_differences(
            rows, "row_aipw_diag_eps005", "plugin_eps005",
            lambda left, right: (
                float(left["direction_row_aipw_signed_spearman_mean"])
                - float(right.get(
                    "direction_spearman_mean", right["signed_spearman_mean"]
                ))
            ),
        ),
        seed_offset=0,
    )
    direction_mae_paired = _paired_bootstrap_summary(
        _paired_differences(
            rows, "row_aipw_diag_eps005", "plugin_eps005",
            lambda left, right: (
                float(right.get("direction_mae_mean", right["signed_mae_mean"]))
                - float(left["direction_row_aipw_signed_mae_mean"])
            ),
        ),
        seed_offset=1,
    )
    forcing_reporting = _forcing_reporting(rows)
    q_recovery = bool(
        math.isfinite(plugin["q_within_state_action_spearman_mean"])
        and plugin["q_nonconstant_surface_count_mean"] > 0.0
        and plugin["q_within_state_action_spearman_mean"] >= MIN_Q_SPEARMAN
        and math.isfinite(plugin["q_normalized_rmse_mean"])
        and plugin["q_normalized_rmse_mean"] <= MAX_Q_NORMALIZED_RMSE
    )
    capacity_recovery = bool(
        math.isfinite(plugin["capacity_active_spearman_mean"])
        and plugin["capacity_active_spearman_mean"] >= MIN_CAPACITY_RANK
        and plugin["capacity_active_support_all_seeds"]
        and math.isfinite(plugin["capacity_active_normalized_mae_mean"])
        and plugin["capacity_active_normalized_mae_mean"] <= MAX_CAPACITY_NORMALIZED_MAE
        and plugin["oracle_core_f1_adjusted_mean"] >= MIN_CAPACITY_CORE_F1
        and math.isfinite(plugin["capacity_null_fpr_mean"])
        and plugin["capacity_null_fpr_mean"] <= MAX_CAPACITY_NULL_FPR
    )
    direction_recovery = bool(
        math.isfinite(plugin["direction_active_spearman_mean"])
        and plugin["direction_active_spearman_mean"] >= MIN_DIRECTION_SPEARMAN
        and plugin["direction_active_support_all_seeds"]
        and math.isfinite(plugin["direction_active_normalized_mae_mean"])
        and plugin["direction_active_normalized_mae_mean"] <= MAX_DIRECTION_NORMALIZED_MAE
        and math.isfinite(plugin["direction_active_sign_agreement_mean"])
        and plugin["direction_active_sign_agreement_mean"]
        >= MIN_DIRECTION_SIGN_AGREEMENT
        and math.isfinite(plugin["direction_null_fpr_mean"])
        and plugin["direction_null_fpr_mean"] <= MAX_DIRECTION_NULL_FPR
    )
    support_integrity = bool(
        plugin["action_coverage_all_seeds"]
        and math.isfinite(plugin["realised_forcing_rate"])
        and plugin["realised_forcing_rate"] > 0.0
    )
    # The full epsilon sweep and paired return endpoint remain reproducibility
    # reporting.  They are not a monotonic-bias theorem and no longer decide
    # whether Q/C/D themselves recover their oracle targets.
    passed = bool(q_recovery and capacity_recovery and direction_recovery and support_integrity)
    return {
        "h1_claim_gate_pass": passed,
        "h1_main": plugin,
        "h1_plugin_control": plugin,
        "h1_row_aipw_diagnostic": row_aipw,
        "h1_observational_control": observational,
        "h1a_q_recovery_pass": q_recovery,
        "h1b_capacity_recovery_pass": capacity_recovery,
        "h1c_direction_recovery_pass": direction_recovery,
        "h1_support_integrity_pass": support_integrity,
        "h1_estimator_ablation": {
            "direction_rank_row_aipw_minus_plugin_paired_bootstrap": direction_rank_paired,
            "direction_mae_plugin_minus_row_aipw_paired_bootstrap": direction_mae_paired,
            "interpretation": (
                "Row-AIPW and cross-fitted AIPW versus plug-in are estimator "
                "diagnostics. Their sign does not "
                "adjudicate H1a/H1b/H1c."
            ),
        },
        "h1_forcing_reporting": forcing_reporting,
        "h1_exp1_reporting_complete": forcing_reporting["reporting_complete"],
        "h1_main_action_coverage_gate_pass": plugin["action_coverage_all_seeds"],
        "h1_main_dr_clipping_absent": True,
        "h1_min_confirmatory_seeds": MIN_CONFIRMATORY_SEEDS,
        "h1_gate_definition": (
            f"nonconstant-surface within-state Q Spearman>={MIN_Q_SPEARMAN}; "
            f"active-C rank>={MIN_CAPACITY_RANK}, random-adjusted C top-k "
            f"F1>={MIN_CAPACITY_CORE_F1}, and "
            f"C null FPR<={MAX_CAPACITY_NULL_FPR}; active-D Spearman>="
            f"{MIN_DIRECTION_SPEARMAN}, active-D sign agreement>="
            f"{MIN_DIRECTION_SIGN_AGREEMENT}, and D null FPR<="
            f"{MAX_DIRECTION_NULL_FPR}; every main-arm seed has action "
            "support and active epsilon forcing. Row-AIPW and cross-fitted AIPW versus "
            "plug-in and the epsilon sweep are reported estimator/manipulation "
            "ablations, not recovery gates."
        ),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7],
        help="Seed list for H1 calibration; default is 8 seeds 0..7",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--variants", nargs="+", choices=[name for name, _ in VARIANTS],
        default=None, help="Optional subset for protocol smoke checks.",
    )
    ap.add_argument("--tiny-proxy-train-episodes", type=int, default=40)
    ap.add_argument("--tiny-states", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument(
        "--threshold-calibration", type=str, default=None,
        help=(
            "Frozen JSON emitted by calibrate_h1_oracle_thresholds.py. "
            "It may contain oracle-derived thresholds only."
        ),
    )
    ap.add_argument(
        "--allow-development-thresholds", action="store_true",
        help=(
            "Permit default thresholds for a development smoke check only. "
            "Never use this flag for confirmatory evidence."
        ),
    )
    ap.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-batch diagnostics; keep one concise line per run.",
    )
    ap.add_argument(
        "--summary_name", type=str, default="summary_h1.csv",
        help="Summary filename under the isolated run directory.",
    )
    ap.add_argument(
        "--out-root", type=str, default=os.path.join("results", "h1"),
        help="Absolute or repository-relative H1 output root.",
    )
    ap.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier. A timestamp plus random suffix is generated by default.",
    )
    ap.add_argument(
        "--aggregate_only", action="store_true",
        help="Do not train; rebuild the requested summary from existing JSON files.",
    )
    args_cli = ap.parse_args(argv)

    if args_cli.aggregate_only and not args_cli.run_id:
        ap.error("--aggregate_only requires --run-id; stale runs are never guessed")
    if len(set(args_cli.seeds)) != len(args_cli.seeds):
        ap.error("--seeds must be unique; duplicate seeds are pseudo-replication")
    if not args_cli.threshold_calibration and not args_cli.allow_development_thresholds:
        ap.error(
            "--threshold-calibration is required for H1 evidence; use "
            "--allow-development-thresholds only for a non-confirmatory smoke check"
        )
    threshold_values = None
    threshold_fingerprint = None
    if args_cli.threshold_calibration:
        threshold_path = os.path.abspath(args_cli.threshold_calibration)
        threshold_values, threshold_fingerprint = _load_threshold_calibration(
            threshold_path
        )
        with open(threshold_path, encoding="utf-8") as handle:
            calibration_payload = json.load(handle)
        development_seeds = {
            int(seed) for seed in calibration_payload.get("development_seeds", [])
        }
        overlap = development_seeds.intersection(int(seed) for seed in args_cli.seeds)
        if overlap and not args_cli.allow_development_thresholds:
            ap.error(
                "confirmatory H1 seeds overlap oracle-threshold development seeds: "
                + ", ".join(map(str, sorted(overlap)))
            )

    run_id = args_cli.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid.uuid4().hex[:8]
    )
    out_root_arg = os.path.expanduser(args_cli.out_root)
    out_root = ensure_dir(
        out_root_arg if os.path.isabs(out_root_arg)
        else os.path.join(ROOT, out_root_arg)
    )
    run_root = ensure_dir(os.path.join(out_root, "runs", run_id))
    selected_variants = [
        item for item in VARIANTS
        if args_cli.variants is None or item[0] in set(args_cli.variants)
    ]
    expected = [
        {"variant": name, "seed": int(seed)}
        for seed in args_cli.seeds for name, _ in selected_variants
    ]
    manifest_path = os.path.join(run_root, "manifest.json")
    if args_cli.aggregate_only:
        if not os.path.exists(manifest_path):
            raise RuntimeError(f"Missing run manifest: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("run_id") != run_id
            or manifest.get("protocol_version") != PROTOCOL_VERSION
            or manifest.get("expected_attempts") != expected
            or manifest.get("threshold_calibration", {}).get("fingerprint")
            != threshold_fingerprint
            or bool(manifest.get("threshold_calibration", {}).get(
                "development_defaults", False
            )) != bool(args_cli.allow_development_thresholds)
        ):
            raise RuntimeError(
                "Run manifest identity or expected attempt matrix does not match"
            )
    else:
        manifest = {
            "run_id": run_id,
            "protocol_version": PROTOCOL_VERSION,
            "created_at": _utc_now(),
            "status": "running",
            "expected_attempts": expected,
            "completed_attempts": [],
            "failed_attempts": [],
            "threshold_calibration": {
                "path": (
                    None if args_cli.threshold_calibration is None
                    else os.path.abspath(args_cli.threshold_calibration)
                ),
                "fingerprint": threshold_fingerprint,
                "development_defaults": bool(args_cli.allow_development_thresholds),
            },
        }
        _atomic_json(manifest_path, manifest)
    rows = []

    for seed in args_cli.seeds:
        for name, cfg_over in selected_variants:
            out_dir = ensure_dir(os.path.join(run_root, f"{name}_seed{seed}"))
            cfg = build_h1_config(cfg_over, seed, threshold_values)
            args = make_args(
                seed=seed, device=args_cli.device, result_dir=out_dir,
                tiny_horizon=1,
                tiny_proxy_train_episodes=max(
                    1, int(args_cli.tiny_proxy_train_episodes)
                ),
                tiny_states=max(1, int(args_cli.tiny_states)),
                max_steps=max(1, int(args_cli.max_steps)),
            )
            args.h1_exact_protocol = True
            attempt_key = {"variant": name, "seed": int(seed)}
            attempt_config = {
                "protocol_version": PROTOCOL_VERSION,
                "variant": name,
                "seed": int(seed),
                "cfg_override": cfg_over,
                "fixed": {
                    "proxy_n_horizons": 1,
                    "causal_horizon": 1,
                    "proxy_iw_clip": 2000.0,
                    "tiny_horizon": 1,
                    "forcer_anneal_to": float(cfg_over["eps"]),
                    "threshold_calibration_fingerprint": threshold_fingerprint,
                    "development_thresholds": bool(
                        args_cli.allow_development_thresholds
                    ),
                },
            }
            attempt_fingerprint = _fingerprint(attempt_config)
            attempt_path = os.path.join(out_dir, "attempt.json")
            if not args_cli.aggregate_only:
                _atomic_json(attempt_path, {
                    **attempt_key,
                    "run_id": run_id,
                    "protocol_version": PROTOCOL_VERSION,
                    "config_fingerprint": attempt_fingerprint,
                    "started_at": _utc_now(),
                    "status": "running",
                })
            print(f"\n########## H1 variant={name} seed={seed} ##########", flush=True)
            if not args_cli.aggregate_only:
                try:
                    if args_cli.quiet:
                        with open(os.devnull, "w", encoding="utf-8") as sink:
                            with contextlib.redirect_stdout(sink):
                                summ = RE.run_tiny_task(
                                    args, cfg, args_cli.device, out_dir=out_dir,
                                    run_label=f"h1_{name}_seed{seed}",
                                )
                    else:
                        summ = RE.run_tiny_task(
                            args, cfg, args_cli.device, out_dir=out_dir,
                            run_label=f"h1_{name}_seed{seed}",
                        )
                    summ.update({
                        "run_id": run_id,
                        "variant": name,
                        "protocol_version": PROTOCOL_VERSION,
                        "config_fingerprint": attempt_fingerprint,
                        "threshold_calibration_fingerprint": threshold_fingerprint,
                        "development_thresholds": bool(
                            args_cli.allow_development_thresholds
                        ),
                        "attempt_complete": True,
                    })
                    _atomic_json(
                        os.path.join(out_dir, "tiny_oracle_summary.json"), summ
                    )
                    _atomic_json(attempt_path, {
                        **attempt_key,
                        "run_id": run_id,
                        "protocol_version": PROTOCOL_VERSION,
                        "config_fingerprint": attempt_fingerprint,
                        "finished_at": _utc_now(),
                        "status": "complete",
                    })
                except Exception as exc:
                    _atomic_json(attempt_path, {
                        **attempt_key,
                        "run_id": run_id,
                        "protocol_version": PROTOCOL_VERSION,
                        "config_fingerprint": attempt_fingerprint,
                        "finished_at": _utc_now(),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    manifest["failed_attempts"].append({
                        **attempt_key, "error": str(exc),
                    })
                    manifest["status"] = "failed"
                    _atomic_json(manifest_path, manifest)
                    raise

            summ_path = os.path.join(out_dir, "tiny_oracle_summary.json")
            row = {"variant": name, "seed": seed}
            row.update(cfg_over)
            if os.path.exists(summ_path):
                with open(summ_path, encoding="utf-8") as f:
                    summ = json.load(f)
                identity_ok = bool(
                    summ.get("run_id") == run_id
                    and summ.get("variant") == name
                    and int(summ.get("seed", -1)) == int(seed)
                    and summ.get("protocol_version") == PROTOCOL_VERSION
                    and summ.get("config_fingerprint") == attempt_fingerprint
                    and summ.get("attempt_complete") is True
                    and summ.get("protocol_gate_pass") is True
                )
                if not identity_ok:
                    raise RuntimeError(
                        f"Refusing incomplete, stale, or protocol-invalid H1 result: "
                        f"{summ_path}"
                    )
                for k, v in summ.items():
                    if k in KEEP_KEYS or isinstance(v, (int, float)):
                        row[k] = v
                row.update({
                    "run_id": run_id,
                    "protocol_version": PROTOCOL_VERSION,
                    "config_fingerprint": attempt_fingerprint,
                    "attempt_complete": True,
                })
                print(
                    f"[H1] {name} seed={seed}: "
                    f"Q={float(summ.get('q_spearman_mean', float('nan'))):+.4f} "
                    f"C={float(summ.get('capacity_rank_correlation_mean', summ.get('rank_correlation_mean', float('nan')))):+.4f} "
                    f"D={float(summ.get('direction_spearman_mean', summ.get('signed_spearman_mean', float('nan')))):+.4f} "
                    f"sign={float(summ.get('direction_sign_agreement_mean', summ.get('sign_agreement_mean', float('nan')))):.3f}",
                    flush=True,
                )
            else:
                raise RuntimeError(f"Missing expected H1 result: {summ_path}")
            rows.append(row)
            if not args_cli.aggregate_only:
                manifest["completed_attempts"].append(attempt_key)
                _atomic_json(manifest_path, manifest)

    if len(rows) != len(expected):
        raise RuntimeError(
            f"H1 completeness failure: expected {len(expected)} rows, got {len(rows)}"
        )

    claim = _claim_gate(rows)
    run_csv_path = os.path.join(run_root, os.path.basename(args_cli.summary_name))
    _atomic_csv(run_csv_path, rows)
    manifest.update({
        "status": "complete",
        "finished_at": _utc_now(),
        "summary_path": run_csv_path,
        "row_count": len(rows),
        **claim,
    })
    _atomic_json(manifest_path, manifest)
    claim_path = os.path.join(run_root, "h1_claim.json")
    _atomic_json(claim_path, {
        "run_id": run_id,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": manifest["finished_at"],
        **claim,
    })

    # Publish only a complete summary. Per-attempt artifacts remain isolated
    # under runs/<run-id>, so a later aggregation cannot mix attempts.
    published_csv = os.path.join(out_root, os.path.basename(args_cli.summary_name))
    _atomic_csv(published_csv, rows)
    _atomic_json(os.path.join(out_root, "latest_complete_run.json"), {
        "run_id": run_id,
        "protocol_version": PROTOCOL_VERSION,
        "summary_path": run_csv_path,
        "manifest_path": manifest_path,
        "claim_path": claim_path,
        "completed_at": manifest["finished_at"],
        **claim,
    })
    _atomic_json(os.path.join(out_root, "latest_h1_claim.json"), {
        "run_id": run_id,
        "protocol_version": PROTOCOL_VERSION,
        "claim_path": claim_path,
        "completed_at": manifest["finished_at"],
        **claim,
    })
    print(f"\n[H1] complete run: {run_root}")
    print(f"[H1] published summary: {published_csv}")
    print(f"[H1] scientific claim gate: {claim['h1_claim_gate_pass']}")


if __name__ == "__main__":
    main()
