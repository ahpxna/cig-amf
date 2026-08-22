"""Validate scientific support separately from experiment completion.

Exit status:
    0   complete confirmatory protocol; H1, H2, and H3 are supported
    10  complete confirmatory protocol; at least one claim is not supported
    11  smoke protocol completed; scientific claims were not evaluated
    20  incomplete, stale, malformed, or otherwise invalid protocol

The validator consumes only the isolated run root supplied by ``run_all.sh``.
It reuses the collector's completeness checks before applying scientific
comparisons, so a process exit code or a finite CSV alone can never count as a
claim pass.
"""

import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np

try:
    import collect_results as CR
except ModuleNotFoundError:
    from scripts import collect_results as CR

try:
    import run_h1_calibration as H1
except ModuleNotFoundError:
    from scripts import run_h1_calibration as H1


EXIT_SUPPORTED = 0
EXIT_UNSUPPORTED = 10
EXIT_SMOKE_ONLY = 11
EXIT_INVALID = 20

MIN_H1_CONFIRMATORY_SEEDS = H1.MIN_CONFIRMATORY_SEEDS
MIN_H23_CONFIRMATORY_SEEDS = 5
MIN_H2_CONFIRMATORY_EPISODES = 400
MIN_H3_CONFIRMATORY_EPISODES = 200
H2_MATCHED_WINDOW_INTERVALS = 2
H2_MIN_FINAL_SR = 3.0
H2_CORRELATION_SR_RANGE = (0.5, 2.0)


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".claim-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=float)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_text(path, content):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".claim-", suffix=".md", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _boolean(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(rows, metric):
    values = [_number(row.get(metric)) for row in rows]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else float("nan")


def _by_group(rows, key):
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: int(row["seed"]))
    return grouped


def _paired_difference(rows_a, rows_b, metric_a, metric_b=None, transform=None):
    metric_b = metric_a if metric_b is None else metric_b
    by_seed_a = {int(row["seed"]): row for row in rows_a}
    by_seed_b = {int(row["seed"]): row for row in rows_b}
    if set(by_seed_a) != set(by_seed_b):
        raise CR.ResultValidationError("paired comparison has mismatched seeds")
    differences = []
    for seed in sorted(by_seed_a):
        a = _number(by_seed_a[seed].get(metric_a))
        b = _number(by_seed_b[seed].get(metric_b))
        if a is None or b is None:
            raise CR.ResultValidationError(
                f"paired comparison has non-finite value at seed {seed}"
            )
        if transform is not None:
            a, b = transform(a), transform(b)
        differences.append(float(a - b))
    return differences


def _paired_derived_difference(rows_a, rows_b, value_fn):
    """Return paired ``value_fn(a)-value_fn(b)`` values in seed order."""
    by_seed_a = {int(row["seed"]): row for row in rows_a}
    by_seed_b = {int(row["seed"]): row for row in rows_b}
    if set(by_seed_a) != set(by_seed_b):
        raise CR.ResultValidationError("paired comparison has mismatched seeds")
    differences = []
    for seed in sorted(by_seed_a):
        a = _number(value_fn(by_seed_a[seed]))
        b = _number(value_fn(by_seed_b[seed]))
        if a is None or b is None:
            raise CR.ResultValidationError(
                f"paired derived comparison has non-finite value at seed {seed}"
            )
        differences.append(float(a - b))
    return differences


def _nonnegative_integer(row, key):
    value = _number(row.get(key))
    if value is None or value < 0.0 or not float(value).is_integer():
        raise CR.ResultValidationError(
            f"H2 {row.get('model')}/seed{row.get('seed')} has invalid {key}"
        )
    return int(value)


def _h2_coverage(row, key):
    shifts = _nonnegative_integer(row, "n_shift_events")
    covered = _nonnegative_integer(row, key)
    if shifts <= 0 or covered > shifts:
        raise CR.ResultValidationError(
            f"H2 {row.get('model')}/seed{row.get('seed')} has impossible "
            f"{key}/n_shift_events counts"
        )
    return float(covered) / float(shifts)


def _h2_joint_recovery_trigger_coverage(row):
    return min(
        _h2_coverage(row, "n_recovered_shifts"),
        _h2_coverage(row, "n_shift_with_trigger"),
    )


def _bootstrap_mean_ci(values, seed, n_bootstrap=10000):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return [float("nan"), float("nan")]
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, values.size, size=(int(n_bootstrap), values.size))
    means = values[indices].mean(axis=1)
    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def _h1_status(rows):
    claim = H1._claim_gate(rows)
    main = claim["h1_main"]
    forcing = claim["h1_forcing_reporting"]
    alignment_protocol = all(
        _boolean(
            row.get(
                "alignment_protocol_gate_pass",
                row.get("protocol_gate_pass", False),
            )
        )
        and int(float(row.get("confirmatory_horizon", -1))) == 1
        for row in rows
    )
    conditions = {
        "alignment_protocol_valid": alignment_protocol,
        "H1a_response_surface_Q_recovery": bool(claim["h1a_q_recovery_pass"]),
        "H1b_structural_capacity_C_recovery": bool(
            claim["h1b_capacity_recovery_pass"]
        ),
        "H1c_directional_effect_D_recovery": bool(
            claim["h1c_direction_recovery_pass"]
        ),
        "intervention_support_integrity": bool(
            claim["h1_support_integrity_pass"]
        ),
        "forcing_return_cost_reported": bool(forcing["reporting_complete"]),
    }
    supported = bool(claim["h1_claim_gate_pass"] and all(conditions.values()))
    return {
        "status": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "supported": supported,
        "conditions": conditions,
        "metrics": {
            "q_spearman": float(main["q_spearman_mean"]),
            "capacity_rank_correlation": float(
                main["capacity_rank_correlation_mean"]
            ),
            "capacity_core_f1": float(main["oracle_core_f1_mean"]),
            "direction_spearman": float(main["direction_spearman_mean"]),
            "direction_sign_agreement": float(
                main["direction_sign_agreement_mean"]
            ),
            "estimator_ablation": claim["h1_estimator_ablation"],
            "forcing_reporting": forcing,
        },
        "scope_note": (
            "This gate validates the one-step V_pi-minus-V_uniform contrast. "
            "It does not validate the previous realised-action or multi-step target."
        ),
    }


def _h2_status(rows, h1_supported):
    grouped = _by_group(rows, "model")
    final = grouped["Final-CIGAMF"]
    correlation = grouped["CorrelationMeanField"]
    no_two = grouped["NoTwoTimescale"]

    sr_difference = _paired_difference(final, correlation, "SR_C")
    sr_ci = _bootstrap_mean_ci(sr_difference, seed=2201)
    final_sr = _mean(final, "SR_C")
    correlation_sr = _mean(correlation, "SR_C")
    no_two_sr = _mean(no_two, "SR_C")

    required_rows = final + correlation + no_two
    evaluable = all(
        _boolean(row.get("claim_evaluable", False))
        and int(float(row.get("n_shift_events", 0))) > 0
        and int(float(row.get("episodes", 0))) >= MIN_H2_CONFIRMATORY_EPISODES
        for row in required_rows
    )
    matched_windows = all(
        int(float(row.get("n_complete_structural_windows", 0)))
        == int(float(row.get("n_complete_behavioral_windows", -1)))
        and int(float(row.get("n_complete_structural_windows", 0))) > 0
        for row in required_rows
    )
    full_recovery_coverage = all(
        _h2_joint_recovery_trigger_coverage(row) == 1.0
        for row in final
    )
    directional_manipulation = all(
        _boolean(row.get("direction_manipulation_pass", False))
        for row in final
    )
    no_two_coverage_delta = _paired_derived_difference(
        final, no_two, _h2_joint_recovery_trigger_coverage
    )
    no_two_coverage_ci = _bootstrap_mean_ci(no_two_coverage_delta, seed=2202)
    paired_latency_differences = []
    final_by_seed = {int(row["seed"]): row for row in final}
    no_two_by_seed = {int(row["seed"]): row for row in no_two}
    for seed in sorted(final_by_seed):
        final_row = final_by_seed[seed]
        no_two_row = no_two_by_seed[seed]
        if (
            _h2_joint_recovery_trigger_coverage(final_row) == 1.0
            and _h2_joint_recovery_trigger_coverage(no_two_row) == 1.0
        ):
            no_two_latency = _number(no_two_row.get("recovery_latency"))
            final_latency = _number(final_row.get("recovery_latency"))
            if no_two_latency is None or final_latency is None:
                raise CR.ResultValidationError(
                    "H2 scheduler latency is non-finite on a fully recovered pair"
                )
            paired_latency_differences.append(no_two_latency - final_latency)
    no_two_latency_ci = _bootstrap_mean_ci(
        paired_latency_differences, seed=2203
    )
    scheduler_effect = bool(
        no_two_coverage_ci[0] > 0.0
        or (
            len(paired_latency_differences) == len(final)
            and no_two_latency_ci[0] > 0.0
        )
    )
    low, high = H2_CORRELATION_SR_RANGE
    mechanism_conditions = {
        "all_matrix_runs_evaluable": evaluable,
        "matched_windows_complete": matched_windows,
        "final_sr_at_least_3": final_sr >= H2_MIN_FINAL_SR,
        "correlation_baseline_sr_near_one": low <= correlation_sr <= high,
        "paired_final_minus_correlation_sr_ci_above_zero": sr_ci[0] > 0.0,
        "final_recovers_and_triggers_for_every_structural_shift": full_recovery_coverage,
        "two_timescale_scheduler_beats_every_episode_control": scheduler_effect,
        "executed_behavioral_policy_moves_direction_D": directional_manipulation,
    }
    mechanism_supported = all(mechanism_conditions.values())
    supported = bool(mechanism_supported and h1_supported)
    if not h1_supported:
        status = "BLOCKED_BY_H1"
    else:
        status = "SUPPORTED" if supported else "NOT_SUPPORTED"
    return {
        "status": status,
        "supported": supported,
        "mechanism_supported": mechanism_supported,
        "depends_on_h1": True,
        "conditions": {
            **mechanism_conditions,
            "validated_structural_input_from_h1": bool(h1_supported),
        },
        "metrics": {
            "final_sr_mean": final_sr,
            "correlation_sr_mean": correlation_sr,
            "no_two_timescale_sr_mean": no_two_sr,
            "paired_final_minus_correlation_sr": sr_difference,
            "paired_final_minus_correlation_sr_ci95": sr_ci,
            "final_recovery_latency_mean": _mean(final, "recovery_latency"),
            "final_joint_recovery_trigger_coverage": [
                _h2_joint_recovery_trigger_coverage(row) for row in final
            ],
            "paired_final_minus_no_two_coverage": no_two_coverage_delta,
            "paired_final_minus_no_two_coverage_ci95": no_two_coverage_ci,
            "paired_no_two_minus_final_recovery_latency": (
                paired_latency_differences
            ),
            "paired_no_two_minus_final_recovery_latency_ci95": no_two_latency_ci,
            "final_direction_behavioral_delta_mean": _mean(final, "direction_behav"),
        },
    }


def _h3_status(rows, attempt, h1_supported):
    grouped = _by_group(rows, "variant")
    full = grouped["Full-CIGAMF"]
    single = grouped["NoMultiMemory-SingleMean"]
    gate = attempt.get("hypothesis_gate", {})

    reward_difference = _paired_difference(full, single, "mean_reward")
    throughput_difference = _paired_difference(full, single, "throughput_total")
    reward_ci = _bootstrap_mean_ci(reward_difference, seed=3301)
    throughput_ci = _bootstrap_mean_ci(throughput_difference, seed=3302)
    specialization = bool(gate.get("specialization_claim_supported", False))
    capacity = bool(gate.get("capacity_claim_supported", False))
    decision_value = bool(gate.get("decision_value_claim_supported", False))
    confirmatory_budget = all(
        int(float(row.get("episodes", 0))) >= MIN_H3_CONFIRMATORY_EPISODES
        for row in rows
    )

    # Independent critics are not a decision ground truth: their value scales
    # are arbitrary. The H3 endpoint is therefore matched-seed post-warm-up
    # F1, with reward and common-probe throughput reported as trade-offs.
    full_supported = bool(
        specialization
        and capacity
        and decision_value
        and confirmatory_budget
        and h1_supported
    )
    if not h1_supported:
        status = "BLOCKED_BY_H1"
    else:
        status = "SUPPORTED" if full_supported else "NOT_SUPPORTED"

    remove_multi_memory = bool(
        float(np.mean(reward_difference)) <= 0.0
        and float(np.mean(throughput_difference)) < 0.0
    )
    return {
        "status": status,
        "supported": full_supported,
        "specialization_mechanism_supported": specialization,
        "capacity_mechanism_supported": capacity,
        "decision_value_claim_supported": decision_value,
        "depends_on_h1": True,
        "conditions": {
            "validated_signature_input_from_h1": bool(h1_supported),
            "specialization_gate": specialization,
            "adaptive_capacity_gate": capacity,
            "decision_value_gate": decision_value,
            "at_least_200_episodes_per_arm": confirmatory_budget,
        },
        "metrics": {
            "full_hard_usage_entropy_mean": _mean(
                full, "hard_usage_entropy_ratio"
            ),
            "full_assignment_mutual_info_mean": _mean(
                full, "assignment_mutual_info_ratio"
            ),
            "full_slot_cosine_mean": _mean(full, "slot_cos_offdiag"),
            "paired_reward_full_minus_single_mean": reward_difference,
            "paired_reward_difference_ci95": reward_ci,
            "paired_throughput_full_minus_single_mean": throughput_difference,
            "paired_throughput_difference_ci95": throughput_ci,
        },
        "go_no_go": {
            "remove_multi_memory_if_tied_or_worse_and_slower": bool(
                gate.get("go_no_go", {}).get(
                    "replace_multi_memory_with_single_mean",
                    remove_multi_memory,
                )
            ),
            "preferred_replacement": (
                "deterministic signed semantic moment bank with one Deep Sets residual"
                if remove_multi_memory or not specialization
                else "retain provisionally; repeat the matched-outcome protocol"
            ),
        },
    }


def _render_markdown(report):
    lines = [
        "# CIG-AMF claim validation\n",
        f"Overall status: **{report['overall_status']}**\n",
        "| Hypothesis | Status | Supported |\n",
        "|---|---:|---:|\n",
    ]
    for name in ("H1", "H2", "H3"):
        item = report["hypotheses"][name]
        lines.append(
            f"| {name} | {item['status']} | "
            f"{'yes' if item.get('supported') else 'no'} |\n"
        )

    for name in ("H1", "H2", "H3"):
        item = report["hypotheses"][name]
        lines.append(f"\n## {name}\n\n")
        for condition, passed in item.get("conditions", {}).items():
            lines.append(f"- [{'x' if passed else ' '}] `{condition}`\n")
        lines.append("\n```json\n")
        lines.append(json.dumps(item.get("metrics", {}), indent=2, default=float))
        lines.append("\n```\n")

    lines.extend([
        "\n## Interpretation constraints\n\n",
        "- H1 is confirmatory only for the aligned one-step stochastic-policy contrast.\n",
        "- H2 is conditional on H1 because its scheduler consumes the learned influence matrix.\n",
        "- H3 uses matched-seed post-warm-up F1 as the decision endpoint; reward and throughput remain separate trade-offs.\n",
        "- A completed command is never treated as a supported claim without the gates above.\n",
    ])
    return "".join(lines)


def validate(
    run_root,
    expected_seeds=None,
    protocol_mode="confirmatory",
    expected_h1_seeds=None,
    expected_h2_seeds=None,
    expected_h3_seeds=None,
):
    # This call re-runs completeness, identity, checksum, finite-value, and
    # exact seed-matrix validation before any scientific comparison.
    expected_h1_seeds = (
        expected_h1_seeds if expected_h1_seeds is not None else expected_seeds
    )
    expected_h2_seeds = (
        expected_h2_seeds if expected_h2_seeds is not None else expected_seeds
    )
    expected_h3_seeds = (
        expected_h3_seeds if expected_h3_seeds is not None else expected_seeds
    )
    if any(value is None for value in (
        expected_h1_seeds, expected_h2_seeds, expected_h3_seeds
    )):
        raise CR.ResultValidationError("expected seed lists are required")
    CR.collect(
        run_root,
        expected_h1_seeds=expected_h1_seeds,
        expected_h2_seeds=expected_h2_seeds,
        expected_h3_seeds=expected_h3_seeds,
    )
    h1_rows, _, _ = CR._load_h1_complete_rows(os.path.join(run_root, "h1"))
    h2_rows, _ = CR._load_h2_complete_rows(os.path.join(run_root, "h2"))
    h3_rows, h3_attempt = CR._load_h3_complete_rows(os.path.join(run_root, "h3"))

    if protocol_mode == "quick":
        hypotheses = {
            name: {
                "status": "SMOKE_ONLY",
                "supported": False,
                "conditions": {},
                "metrics": {},
            }
            for name in ("H1", "H2", "H3")
        }
        return {
            "overall_status": "SMOKE_ONLY",
            "protocol_mode": protocol_mode,
            "expected_seeds": {
                "h1": list(expected_h1_seeds),
                "h2": list(expected_h2_seeds),
                "h3": list(expected_h3_seeds),
            },
            "hypotheses": hypotheses,
        }, EXIT_SMOKE_ONLY

    if len(expected_h1_seeds) < MIN_H1_CONFIRMATORY_SEEDS:
        raise CR.ResultValidationError(
            "confirmatory H1 requires at least "
            f"{MIN_H1_CONFIRMATORY_SEEDS} paired seeds"
        )
    if (
        len(expected_h2_seeds) < MIN_H23_CONFIRMATORY_SEEDS
        or len(expected_h3_seeds) < MIN_H23_CONFIRMATORY_SEEDS
    ):
        raise CR.ResultValidationError(
            "confirmatory H2/H3 require at least "
            f"{MIN_H23_CONFIRMATORY_SEEDS} paired seeds"
        )

    h1 = _h1_status(h1_rows)
    h2 = _h2_status(h2_rows, h1_supported=h1["supported"])
    h3 = _h3_status(
        h3_rows, h3_attempt, h1_supported=h1["supported"]
    )
    hypotheses = {"H1": h1, "H2": h2, "H3": h3}
    all_supported = all(item["supported"] for item in hypotheses.values())
    return {
        "overall_status": "SUPPORTED" if all_supported else "NOT_SUPPORTED",
        "protocol_mode": protocol_mode,
        "expected_seeds": {
            "h1": list(expected_h1_seeds),
            "h2": list(expected_h2_seeds),
            "h3": list(expected_h3_seeds),
        },
        "hypotheses": hypotheses,
    }, EXIT_SUPPORTED if all_supported else EXIT_UNSUPPORTED


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--expected-h1-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--expected-h2-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--expected-h3-seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--protocol-mode",
        choices=("confirmatory", "quick"),
        default="confirmatory",
    )
    args = parser.parse_args(argv)
    run_root = os.path.abspath(args.run_root)
    json_path = os.path.join(run_root, "claim_status.json")
    markdown_path = os.path.join(run_root, "claim_status.md")

    try:
        report, exit_status = validate(
            run_root,
            expected_seeds=args.expected_seeds,
            protocol_mode=args.protocol_mode,
            expected_h1_seeds=args.expected_h1_seeds,
            expected_h2_seeds=args.expected_h2_seeds,
            expected_h3_seeds=args.expected_h3_seeds,
        )
    except (
        CR.ResultValidationError,
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as exc:
        fallback = args.expected_seeds
        report = {
            "overall_status": "INVALID",
            "protocol_mode": args.protocol_mode,
            "expected_seeds": {
                "h1": list(args.expected_h1_seeds or fallback or []),
                "h2": list(args.expected_h2_seeds or fallback or []),
                "h3": list(args.expected_h3_seeds or fallback or []),
            },
            "error_type": type(exc).__name__,
            "error": str(exc),
            "hypotheses": {},
        }
        _atomic_json(json_path, report)
        _atomic_text(
            markdown_path,
            "# CIG-AMF claim validation\n\n"
            f"Overall status: **INVALID**\n\n{type(exc).__name__}: {exc}\n",
        )
        print(f"[claims][INVALID] {exc}")
        return EXIT_INVALID

    _atomic_json(json_path, report)
    _atomic_text(markdown_path, _render_markdown(report))
    print(_render_markdown(report))
    print(f"[claims] JSON: {json_path}")
    print(f"[claims] Markdown: {markdown_path}")
    return exit_status


if __name__ == "__main__":
    sys.exit(main())
