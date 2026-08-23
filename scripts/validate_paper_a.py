"""Validate Paper-A Q/C/D recovery, selectivity, and each H3 contribution."""

import argparse
import json
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
    with open(latency_path, encoding="utf-8") as handle:
        oracle_latency = json.load(handle)
    oracle_pass = bool(oracle_latency.get("gate_pass", False))
    learned_latency = None
    if oracle_pass:
        learned_path = os.path.join(run_root, "latency_calibration.json")
        if not os.path.exists(learned_path):
            raise CR.ResultValidationError(
                "oracle latency gate passed but learned calibration is missing"
            )
        with open(learned_path, encoding="utf-8") as handle:
            learned_latency = json.load(handle)
        learned_seeds = {int(seed) for seed in learned_latency.get("seeds", [])}
        if protocol_mode == "confirmatory" and learned_seeds != set(h2_seeds):
            raise CR.ResultValidationError(
                "learned latency seed matrix does not match the confirmatory "
                f"Paper-A seeds: {sorted(learned_seeds)} != {sorted(h2_seeds)}"
            )
    learned_pass = bool(learned_latency and learned_latency.get("gate_pass"))
    latency_status = {
        "status": "SUPPORTED" if learned_pass else "GATED_OUT",
        "supported": learned_pass,
        "optional": True,
        "oracle_gate_pass": oracle_pass,
        "learned_gate_pass": learned_pass,
        "oracle_metrics": {
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
            final_latency = [
                float(final_by_seed[seed]["recovery_latency"])
                for seed in sorted(final_by_seed)
            ]
            control_latency = [
                float(control_by_seed[seed]["recovery_latency"])
                for seed in sorted(final_by_seed)
            ]
            if any(value < 0.0 for value in final_latency + control_latency):
                raise CR.ResultValidationError(
                    f"H3b recovery failed or was undefined for {control}"
                )
            deltas = [right - left for left, right in zip(final_latency, control_latency)]
            tracking_deltas[control] = {
                "control_minus_final": deltas,
                "ci95": VC._bootstrap_mean_ci(
                    deltas, seed=5200 + len(tracking_deltas)
                ),
            }
    final_false_alarm = [
        float(row.get("behavioral_false_trigger_rate", float("nan")))
        for row in h2_rows if row.get("model") == "Final-CIGAMF"
    ]
    false_alarm_reported = bool(
        final_false_alarm and all(value >= 0.0 for value in final_false_alarm)
    )
    tracking_conditions = {
        f"final_recovers_faster_than_{control}": metrics["ci95"][0] > 0.0
        for control, metrics in tracking_deltas.items()
    }
    tracking_conditions["behavioral_false_alarm_control_reported"] = false_alarm_reported
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
        "rule": (
            "H3b requires faster recovery than matched NoDetector while "
            "reporting behavioural-only false alarms. Fixed-rate, fast, "
            "no-uncertainty, and no-two-timescale arms are diagnostics."
        ),
    }
    h3_latency_supported = bool(latency_status["supported"])
    h3_tracking_supported = bool(tracking_status["supported"])
    supported = bool(
        h1["supported"] and h2["supported"]
        and h3_latency_supported and h3_tracking_supported
    )
    if supported:
        overall_status = "SUPPORTED"
    elif h1["supported"] and h2["supported"]:
        overall_status = "CORE_SUPPORTED_H3_INCOMPLETE"
    else:
        overall_status = "NOT_SUPPORTED"
    return {
        "paper": "A",
        "overall_status": overall_status,
        "H1": h1,
        "H2": h2,
        "H3a_latency": latency_status,
        "H3b_tracking": tracking_status,
        "full_hypothesis_set_supported": supported,
        # Compatibility alias; it is not used to infer full Paper-A support.
        "latency": latency_status,
        "latency_policy": (
            "H3a is separately gated. An oracle or learned failure does not "
            "invalidate Q/C/D, but it prevents a full Paper-A supported label."
        ),
    }, VC.EXIT_SUPPORTED if supported else VC.EXIT_UNSUPPORTED


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
