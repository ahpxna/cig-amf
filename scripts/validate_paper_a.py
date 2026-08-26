"""Validate Paper-A Q/C/D recovery, selectivity, tracking, and optional latency."""

import argparse
import json
import math
import os
import sys
import tempfile

try:
    import collect_results as CR
    import validate_claims as VC
except ModuleNotFoundError:
    from scripts import collect_results as CR
    from scripts import validate_claims as VC


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".paper-a-", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=float)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate(run_root, h1_seeds, h2_seeds, protocol_mode):
    h1_rows, _, _ = CR._load_h1_complete_rows(os.path.join(run_root, "h1"))
    h2_rows, _ = CR._load_h2_complete_rows(os.path.join(run_root, "h2"))
    if {int(row["seed"]) for row in h1_rows} != set(h1_seeds):
        raise CR.ResultValidationError("Paper-A H1 seed matrix mismatch")
    if {int(row["seed"]) for row in h2_rows} != set(h2_seeds):
        raise CR.ResultValidationError("Paper-A H2 seed matrix mismatch")
    latency_path = os.path.join(run_root, "latency_oracle.json")
    oracle_latency = None
    latency_gate_reason = None
    if os.path.exists(latency_path):
        with open(latency_path, encoding="utf-8") as handle:
            oracle_latency = json.load(handle)
    else:
        latency_gate_reason = "oracle latency artifact is absent"
    oracle_pass = bool(oracle_latency and oracle_latency.get("gate_pass", False))
    learned_latency = None
    if oracle_pass:
        learned_path = os.path.join(run_root, "latency_calibration.json")
        if not os.path.exists(learned_path):
            latency_gate_reason = "oracle passed but learned latency calibration is absent"
        else:
            with open(learned_path, encoding="utf-8") as handle:
                learned_latency = json.load(handle)
            learned_seeds = {int(seed) for seed in learned_latency.get("seeds", [])}
            if protocol_mode == "confirmatory" and learned_seeds != set(h2_seeds):
                raise CR.ResultValidationError(
                    "learned latency seed matrix does not match the confirmatory "
                    f"Paper-A seeds: {sorted(learned_seeds)} != {sorted(h2_seeds)}"
                )
    elif oracle_latency is not None and not oracle_pass:
        latency_gate_reason = "oracle latency mechanism gate did not pass"
    learned_pass = bool(learned_latency and learned_latency.get("gate_pass"))
    if learned_latency is not None and not learned_pass:
        latency_gate_reason = "learned latency calibration did not pass"
    latency_status = {
        "status": "SUPPORTED" if learned_pass else "GATED_OUT",
        "supported": learned_pass,
        "optional": True,
        "gate_reason": None if learned_pass else latency_gate_reason,
        "oracle_artifact_present": oracle_latency is not None,
        "oracle_gate_pass": oracle_pass,
        "learned_gate_pass": learned_pass,
        "oracle_metrics": None if oracle_latency is None else {
            key: oracle_latency.get(key)
            for key in (
                "randomized_delay_rank_correlation",
                "randomized_delay_mae",
                "active_pair_state_fraction",
            )
        },
        "learned_metrics": None if learned_latency is None else {
            key: learned_latency.get(key)
            for key in (
                "learned_delay_rank_correlation",
                "learned_delay_mae",
                "learned_oracle_center_rank_correlation",
                "zero_lag_baseline_mae",
                "terminal_lag_baseline_mae",
                "n_seeds",
                "per_seed_gate_pass_fraction",
                "n_valid",
            )
        },
    }
    if protocol_mode == "quick":
        return {
            "paper": "A",
            "overall_status": "SMOKE_ONLY",
            "H1": {"status": "SMOKE_ONLY", "supported": False},
            "H2": {"status": "SMOKE_ONLY", "supported": False},
            "H3a_latency": latency_status,
            "H3b_tracking": {"status": "SMOKE_ONLY", "supported": False},
        }, VC.EXIT_SMOKE_ONLY
    h1 = VC._h1_status(h1_rows)
    h2 = VC._h2_status(h2_rows, h1_supported=h1["supported"])
    # H3b's prespecified primary comparator is the otherwise matched tracker
    # without change-triggered re-estimation.  Other tracker variants are
    # mechanism diagnostics/trade-offs and must not turn a valid primary
    # comparison into a false rejection.
    required_tracking_models = {"Final-CIGAMF", "NoDetector"}
    observed_tracking_models = {str(row.get("model")) for row in h2_rows}
    missing_tracking_models = sorted(required_tracking_models - observed_tracking_models)
    tracking_deltas = {}
    if not missing_tracking_models:
        final_by_seed = {int(row["seed"]): row for row in h2_rows if row.get("model") == "Final-CIGAMF"}
        for control in ("NoDetector",):
            control_by_seed = {int(row["seed"]): row for row in h2_rows if row.get("model") == control}
            if set(final_by_seed) != set(control_by_seed):
                raise CR.ResultValidationError(
                    f"H3b unpaired recovery rows for {control}"
                )
            final_latency = [float(final_by_seed[seed]["recovery_latency"])
                             for seed in sorted(final_by_seed)]
            control_latency = [float(control_by_seed[seed]["recovery_latency"])
                               for seed in sorted(final_by_seed)]
            # A non-recovering Final arm is unfavorable evidence. A
            # non-recovering control is right-censored, not malformed: it
            # supports the intended ordering when Final recovered.
            if any(value < 0.0 for value in final_latency):
                tracking_deltas[control] = {
                    "control_minus_final": [],
                    "ci95": [float("nan"), float("nan")],
                    "final_nonrecovery": True,
                    "control_right_censored_count": int(sum(value < 0.0 for value in control_latency)),
                }
                continue
            censor_horizon = max(
                1.0,
                max(float(row.get("episodes", 1)) / max(1.0, float(row.get("eval_every", 1)))
                    for row in control_by_seed.values()),
            )
            deltas = [
                (censor_horizon if right < 0.0 else right) - left
                for left, right in zip(final_latency, control_latency)
            ]
            tracking_deltas[control] = {
                "control_minus_final": deltas,
                "ci95": VC._bootstrap_mean_ci(
                    deltas, seed=5200 + len(tracking_deltas)
                ),
                "final_nonrecovery": False,
                "control_right_censored_count": int(sum(value < 0.0 for value in control_latency)),
                "control_censor_horizon_intervals": censor_horizon,
            }
    final_rows = [
        row for row in h2_rows if row.get("model") == "Final-CIGAMF"
    ]
    false_alarm_targets = [
        float(row.get("cusum_false_alarm_target", float("nan")))
        for row in final_rows
    ]
    false_alarm_windows = [
        int(row.get("behavioral_false_alarm_window_count", -1))
        for row in final_rows
    ]
    monitoring_windows = [
        int(row.get("behavioral_monitoring_window_count", -1))
        for row in final_rows
    ]
    false_alarm_reported = bool(
        final_rows
        and all(window > 0 for window in monitoring_windows)
        and all(0 <= alarm <= window for alarm, window in zip(false_alarm_windows, monitoring_windows))
    )
    pooled_false_alarm_rate = (
        float(sum(false_alarm_windows) / sum(monitoring_windows))
        if false_alarm_reported and sum(monitoring_windows) > 0
        else float("nan")
    )
    one_frozen_target = bool(
        false_alarm_targets
        and all(math.isfinite(target) and 0.0 < target < 1.0 for target in false_alarm_targets)
        and max(false_alarm_targets) - min(false_alarm_targets) <= 1e-12
    )
    false_alarm_calibrated = bool(
        false_alarm_reported and one_frozen_target
        and pooled_false_alarm_rate <= false_alarm_targets[0]
    )
    tracking_conditions = {
        f"final_recovers_faster_than_{control}": metrics["ci95"][0] > 0.0
        for control, metrics in tracking_deltas.items()
    }
    tracking_conditions["behavioral_false_alarm_control_reported"] = false_alarm_reported
    tracking_conditions["behavioral_false_alarm_at_or_below_calibrated_target"] = false_alarm_calibrated
    tracking_supported = bool(
        not missing_tracking_models
        and tracking_conditions
        and all(tracking_conditions.values())
    )
    tracking_status = {
        "status": "SUPPORTED" if tracking_supported else (
            "NOT_EVALUATED" if missing_tracking_models else "NOT_SUPPORTED"
        ),
        "supported": tracking_supported,
        "required_comparators": sorted(required_tracking_models),
        "observed_comparators": sorted(observed_tracking_models),
        "missing_comparators": missing_tracking_models,
        "conditions": tracking_conditions,
        "paired_recovery_latency": tracking_deltas,
        "behavioral_false_alarm": {
            "monitoring_windows_by_seed": monitoring_windows,
            "false_alarm_windows_by_seed": false_alarm_windows,
            "pooled_window_rate": pooled_false_alarm_rate,
            "frozen_target": (false_alarm_targets[0] if one_frozen_target else float("nan")),
        },
        "rule": (
            "H3b requires faster recovery than matched NoDetector while "
            "meeting the frozen no-change behavioural false-alarm target. "
            "Control non-recovery is right-censored; Final non-recovery fails. Fixed-rate, fast, "
            "no-uncertainty, and no-two-timescale arms are diagnostics."
        ),
    }
    h3_latency_supported = bool(latency_status["supported"])
    h3_tracking_supported = bool(tracking_status["supported"])

    # The gate ladder treats both latency and the online CUSUM/tracking trigger
    # as modular contributions.  H1 Q/C/D recovery plus H2 structural/behavioural
    # separation define the Paper-A causal core.  A failed optional mechanism
    # must shrink the submitted claim set rather than incorrectly falsify the
    # already-supported causal estimands.
    core_supported = bool(h1["supported"] and h2["supported"])
    all_modules_supported = bool(
        core_supported and h3_latency_supported and h3_tracking_supported
    )
    if core_supported and h3_latency_supported and h3_tracking_supported:
        overall_status = "SUPPORTED_WITH_LATENCY_AND_TRACKING"
    elif core_supported and h3_tracking_supported:
        overall_status = "SUPPORTED_LATENCY_GATED_OUT"
    elif core_supported and h3_latency_supported:
        overall_status = "SUPPORTED_TRACKING_GATED_OUT"
    elif core_supported:
        overall_status = "SUPPORTED_OPTIONAL_MODULES_GATED_OUT"
    else:
        overall_status = "NOT_SUPPORTED"

    return {
        "paper": "A",
        "overall_status": overall_status,
        "H1": h1,
        "H2": h2,
        "H3a_latency": latency_status,
        "H3b_tracking": tracking_status,
        "submitted_claim_set_supported": core_supported,
        "full_hypothesis_set_supported": all_modules_supported,
        "all_optional_modules_supported": all_modules_supported,
        "latency": latency_status,
        "latency_policy": (
            "H3a latency is separately gated and optional. Oracle or learned "
            "latency failure gates out only the latency contribution."
        ),
        "tracking_policy": (
            "H3b CUSUM/tracking is separately gated and optional. Failure of "
            "the frozen behavioural false-alarm or recovery gate removes the "
            "trigger/tracking contribution without rewriting H1/H2."
        ),
    }, VC.EXIT_SUPPORTED if core_supported else VC.EXIT_UNSUPPORTED


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-h1-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--expected-h2-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--protocol-mode", choices=("quick", "confirmatory"),
                        default="confirmatory")
    args = parser.parse_args(argv)
    output = os.path.join(os.path.abspath(args.run_root), "paper_a_claim_status.json")
    try:
        report, code = validate(
            os.path.abspath(args.run_root), args.expected_h1_seeds,
            args.expected_h2_seeds, args.protocol_mode,
        )
    except Exception as exc:
        report = {
            "paper": "A", "overall_status": "INVALID",
            "error_type": type(exc).__name__, "error": str(exc),
        }
        code = VC.EXIT_INVALID
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, default=float))
    return code


if __name__ == "__main__":
    sys.exit(main())
