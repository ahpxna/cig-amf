"""Validate Paper-A Q/C/D recovery, factorial selectivity, and gated latency."""

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
            "latency": latency_status,
        }, VC.EXIT_SMOKE_ONLY
    h1 = VC._h1_status(h1_rows)
    h2 = VC._h2_status(h2_rows, h1_supported=h1["supported"])
    supported = bool(h1["supported"] and h2["supported"])
    return {
        "paper": "A",
        "overall_status": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "H1": h1,
        "H2": h2,
        "latency": latency_status,
        "latency_policy": (
            "Optional contribution; both oracle and learned gates must pass, "
            "and either failure leaves Q/C/D unaffected."
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
