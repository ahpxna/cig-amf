"""H1 confirmatory calibration with an estimand-aligned one-step oracle.

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
    # Main paired comparison. Both variants train the same plug-in nuisance
    # model; only held-out scoring changes.
    ("dr_eps005", {"proxy_use_doubly_robust": True, "eps": 0.05}),
    ("plugin_eps005", {"proxy_use_doubly_robust": False, "eps": 0.05}),
    # Constant-epsilon identification sweep.
    ("dr_eps000", {"proxy_use_doubly_robust": True, "eps": 0.0}),
    ("dr_eps001", {"proxy_use_doubly_robust": True, "eps": 0.01}),
    ("dr_eps003", {"proxy_use_doubly_robust": True, "eps": 0.03}),
    ("dr_eps008", {"proxy_use_doubly_robust": True, "eps": 0.08}),
    ("dr_eps012", {"proxy_use_doubly_robust": True, "eps": 0.12}),
]

KEEP_KEYS = (
    "rank_correlation_mean", "spearman_mean", "sign_agreement_mean",
    "signed_spearman_mean", "signed_mae_mean", "signed_bias_mean",
    "signed_rmse_mean", "signed_p_value_mean",
    "range_rank_correlation_mean",
    "bias_mean", "mae_mean", "pearson_mean", "n_states", "n_pairs",
)

PROTOCOL_VERSION = "h1_exact_v1"
MIN_CONFIRMATORY_SEEDS = 8
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 1729


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


def _variant_means(rows, variant):
    selected = [row for row in rows if row["variant"] == variant]
    out = {}
    for key in (
        "rank_correlation_mean", "signed_spearman_mean",
        "sign_agreement_mean", "signed_bias_mean", "signed_mae_mean",
        "range_rank_correlation_mean",
        "realised_forcing_rate", "heldout_policy_return_mean_per_agent",
    ):
        vals = [float(row[key]) for row in selected if row.get(key) not in (None, "")]
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
        (0.00, "dr_eps000"),
        (0.01, "dr_eps001"),
        (0.03, "dr_eps003"),
        (0.05, "dr_eps005"),
        (0.08, "dr_eps008"),
        (0.12, "dr_eps012"),
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
        rows, "dr_eps000", "dr_eps005",
        lambda no_forcing, forcing: (
            float(no_forcing["heldout_policy_return_mean_per_agent"])
            - float(forcing["heldout_policy_return_mean_per_agent"])
        ),
    )
    cost_summary = _paired_bootstrap_summary(cost_pairs, seed_offset=2)
    endpoints = [
        row for row in rows
        if row.get("variant") in ("dr_eps000", "dr_eps005")
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
            "dr_eps000 minus dr_eps005; positive values are a forcing cost"
        ),
        "forcing_return_cost_paired_bootstrap": cost_summary,
        "forcing_return_cost_measured": return_endpoint_complete,
        "reporting_complete": bool(
            realised_rate_complete and return_endpoint_complete
        ),
    }


def _claim_gate(rows):
    """Return the preregistered H1 outcome without changing process status."""
    dr = _variant_means(rows, "dr_eps005")
    plugin = _variant_means(rows, "plugin_eps005")
    observational = _variant_means(rows, "dr_eps000")
    signed_rank_paired = _paired_bootstrap_summary(
        _paired_differences(
            rows, "dr_eps005", "plugin_eps005",
            lambda left, right: (
                float(left["signed_spearman_mean"])
                - float(right["signed_spearman_mean"])
            ),
        ),
        seed_offset=0,
    )
    absolute_bias_paired = _paired_bootstrap_summary(
        _paired_differences(
            rows, "dr_eps005", "plugin_eps005",
            lambda left, right: (
                abs(float(right["signed_bias_mean"]))
                - abs(float(left["signed_bias_mean"]))
            ),
        ),
        seed_offset=1,
    )
    epsilon_trend = _epsilon_bias_trend(rows)
    forcing_reporting = _forcing_reporting(rows)
    paired_uncertainty_pass = bool(
        signed_rank_paired["n_pairs"] >= MIN_CONFIRMATORY_SEEDS
        and absolute_bias_paired["n_pairs"] >= MIN_CONFIRMATORY_SEEDS
        and signed_rank_paired["ci95_low"] > 0.0
        and absolute_bias_paired["ci95_low"] > 0.0
    )
    passed = bool(
        dr["signed_spearman_mean"] > plugin["signed_spearman_mean"]
        and dr["signed_spearman_mean"] > observational["signed_spearman_mean"]
        and dr["signed_spearman_mean"] > dr["range_rank_correlation_mean"]
        and dr["sign_agreement_mean"] >= 0.75
        and dr["signed_bias_abs_mean"] < plugin["signed_bias_abs_mean"]
        and dr["action_coverage_all_seeds"]
        and dr["dr_clipping_absent_all_seeds"]
        and paired_uncertainty_pass
        and epsilon_trend["gate_pass"]
        and forcing_reporting["reporting_complete"]
    )
    return {
        "h1_claim_gate_pass": passed,
        "h1_main": dr,
        "h1_plugin_control": plugin,
        "h1_observational_control": observational,
        "h1_signed_rank_dr_minus_plugin_paired_bootstrap": signed_rank_paired,
        "h1_absolute_bias_plugin_minus_dr_paired_bootstrap": absolute_bias_paired,
        "h1_paired_uncertainty_gate_pass": paired_uncertainty_pass,
        "h1_epsilon_absolute_bias_trend": epsilon_trend,
        "h1_forcing_reporting": forcing_reporting,
        "h1_exp1_reporting_complete": forcing_reporting["reporting_complete"],
        "h1_main_action_coverage_gate_pass": dr["action_coverage_all_seeds"],
        "h1_main_dr_clipping_absent": dr["dr_clipping_absent_all_seeds"],
        "h1_min_confirmatory_seeds": MIN_CONFIRMATORY_SEEDS,
        "h1_gate_definition": (
            "signed_rank(dr_eps005)>signed_rank(plugin_eps005), "
            "signed_rank(dr_eps005)>signed_rank(dr_eps000), "
            "signed_rank(dr_eps005)>unsigned_range_rank(dr_eps005), "
            "sign_agreement(dr_eps005)>=0.75, mean_seed_abs_signed_bias"
            "(dr_eps005)<mean_seed_abs_signed_bias(plugin_eps005), all action "
            "heads covered in every main-arm seed, no importance-weight "
            "clipping in the main arm, paired 95% bootstrap lower bounds above "
            "zero for signed-rank gain and absolute-bias improvement, and decreasing "
            "absolute bias over the constant-"
            "epsilon sweep (negative OLS slope, positive endpoint improvement, "
            "and >=60% adjacent non-increases). Full Exp1 support also requires "
            "realised forcing rates and a paired eps=0 held-out return-cost "
            "endpoint for every confirmatory seed"
        ),
    }


def main():
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
    ap.add_argument("--tiny-states", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=30)
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
    args_cli = ap.parse_args()

    if args_cli.aggregate_only and not args_cli.run_id:
        ap.error("--aggregate_only requires --run-id; stale runs are never guessed")
    if len(set(args_cli.seeds)) != len(args_cli.seeds):
        ap.error("--seeds must be unique; duplicate seeds are pseudo-replication")

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
        }
        _atomic_json(manifest_path, manifest)
    rows = []

    for seed in args_cli.seeds:
        for name, cfg_over in selected_variants:
            out_dir = ensure_dir(os.path.join(run_root, f"{name}_seed{seed}"))
            cfg = RE.default_cfg()
            cfg.update(cfg_over)
            # The epsilon sweep must remain constant. Previously every variant
            # annealed toward 0.05, so eps=0 was not an observational control.
            cfg["forcer_anneal_to"] = float(cfg_over["eps"])
            cfg["proxy_n_horizons"] = 1
            # The smallest declared positive exploration rate is 0.01, whose
            # exact marginal propensity floor is eps/13 and inverse is 1300.
            # A 2000 ceiling therefore leaves every randomized arm untruncated
            # while retaining a finite safety guard for the eps=0 control.
            cfg["proxy_iw_clip"] = 2000.0
            cfg["seed"] = int(seed)
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
                    "proxy_iw_clip": 2000.0,
                    "tiny_horizon": 1,
                    "forcer_anneal_to": float(cfg_over["eps"]),
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
                    f"rank={float(summ.get('rank_correlation_mean', float('nan'))):+.4f} "
                    f"signed={float(summ.get('signed_spearman_mean', float('nan'))):+.4f} "
                    f"sign={float(summ.get('sign_agreement_mean', float('nan'))):.3f}",
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
