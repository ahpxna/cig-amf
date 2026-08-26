"""Legacy combined H1/H2/H3 claim validator.

Final paper decisions are made by ``validate_paper_a.py`` and
``validate_paper_b.py``. This module is retained for historical/diagnostic
compatibility and for shared helper functions used by those validators.

Exit status for this legacy combined validator:
    0   complete confirmatory protocol; H1, H2, and legacy H3 are supported
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
MIN_H23_CONFIRMATORY_SEEDS = 8
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
        "H1d_forcing_improves_support_poor_endpoint": bool(
            forcing["support_poor_endpoint"]["prediction_pass"]
        ),
    }
    supported = bool(claim["h1_claim_gate_pass"] and all(conditions.values()))
    return {
        "status": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "supported": supported,
        "conditions": conditions,
        "metrics": {
            "q_centered_mae": float(main["q_centered_mae_mean"]),
            "q_centered_rmse": float(main["q_centered_rmse_mean"]),
            "q_within_state_action_spearman_nonconstant": float(
                main["q_within_state_action_spearman_mean"]
            ),
            "q_nonconstant_surface_count": float(
                main["q_nonconstant_surface_count_mean"]
            ),
            "q_normalized_rmse": float(main["q_normalized_rmse_mean"]),
            "capacity_rank_correlation": float(
                main["capacity_rank_correlation_mean"]
            ),
            "capacity_core_f1": float(main["oracle_core_f1_mean"]),
            "capacity_core_f1_random_baseline": float(
                main["oracle_core_f1_random_baseline_mean"]
            ),
            "capacity_core_f1_adjusted": float(
                main["oracle_core_f1_adjusted_mean"]
            ),
            "capacity_active_spearman": float(
                main["capacity_active_spearman_mean"]
            ),
            "capacity_active_pair_count": float(
                main["capacity_active_pair_count_mean"]
            ),
            "capacity_active_normalized_mae": float(
                main["capacity_active_normalized_mae_mean"]
            ),
            "capacity_null_fpr": float(main["capacity_null_fpr_mean"]),
            "direction_spearman": float(main["direction_spearman_mean"]),
            "direction_sign_agreement": float(
                main["direction_sign_agreement_mean"]
            ),
            "direction_active_spearman": float(
                main["direction_active_spearman_mean"]
            ),
            "direction_active_pair_count": float(
                main["direction_active_pair_count_mean"]
            ),
            "direction_active_normalized_mae": float(
                main["direction_active_normalized_mae_mean"]
            ),
            "direction_active_sign_agreement": float(
                main["direction_active_sign_agreement_mean"]
            ),
            "direction_null_fpr": float(main["direction_null_fpr_mean"]),
            "estimator_ablation": claim["h1_estimator_ablation"],
            "forcing_reporting": forcing,
        },
        "scope_note": (
            "This gate validates the one-step V_pi-minus-V_uniform contrast. "
            "It does not validate the previous realised-action or multi-step target."
        ),
    }


def _h2_status(rows, h1_supported):
    """Validate the revised 2x2 structural/behavioural factorial."""
    grouped = _by_group(rows, "model")
    final = grouped["Final-CIGAMF"]
    correlation = grouped["CorrelationMeanField"]
    required_rows = final + correlation
    evaluable = all(
        _boolean(row.get("claim_evaluable", False))
        and int(float(row.get("episodes", 0))) >= MIN_H2_CONFIRMATORY_EPISODES
        and _boolean(row.get("policy_learning_frozen", False))
        and _boolean(row.get("representation_learning_frozen", False))
        and _boolean(row.get("frozen_representation_unchanged", False))
        for row in required_rows
    )

    def values(source, key):
        result = [_number(row.get(key)) for row in source]
        if any(value is None for value in result):
            raise CR.ResultValidationError(f"H2 has non-finite {key}")
        return [float(value) for value in result]

    capacity_struct = values(final, "capacity_beta_structural")
    capacity_behav = values(final, "capacity_beta_behavioral")
    direction_struct = values(final, "direction_beta_structural")
    direction_behav = values(final, "direction_beta_behavioral")
    capacity_selectivity = [
        structural - abs(behavioural)
        for structural, behavioural in zip(capacity_struct, capacity_behav)
    ]
    direction_selectivity = [
        behavioural - abs(structural)
        for behavioural, structural in zip(direction_behav, direction_struct)
    ]
    capacity_struct_ci = _bootstrap_mean_ci(capacity_struct, seed=2201)
    capacity_selectivity_ci = _bootstrap_mean_ci(capacity_selectivity, seed=2202)
    direction_selectivity_ci = _bootstrap_mean_ci(direction_selectivity, seed=2203)

    estimand_capacity_struct = values(final, "estimand_capacity_beta_structural")
    estimand_capacity_behav = values(final, "estimand_capacity_beta_behavioral")
    estimand_direction_behav = values(final, "estimand_direction_beta_behavioral")
    estimand_capacity_selectivity = [
        structural - abs(behavioural)
        for structural, behavioural in zip(
            estimand_capacity_struct, estimand_capacity_behav
        )
    ]
    estimand_capacity_selectivity_ci = _bootstrap_mean_ci(
        estimand_capacity_selectivity, seed=2204
    )
    estimand_direction_behav_ci = _bootstrap_mean_ci(
        estimand_direction_behav, seed=2205
    )

    correlation_capacity_struct = values(correlation, "capacity_beta_structural")
    correlation_capacity_behav = values(correlation, "capacity_beta_behavioral")
    correlation_selectivity = [
        structural - abs(behavioural)
        for structural, behavioural in zip(
            correlation_capacity_struct, correlation_capacity_behav
        )
    ]
    correlation_selectivity_delta = [
        final_value - association_value
        for final_value, association_value in zip(
            capacity_selectivity, correlation_selectivity
        )
    ]
    correlation_selectivity_ci = _bootstrap_mean_ci(
        correlation_selectivity_delta, seed=2206
    )
    behavioral_false_alarm_window_rate = values(
        final, "behavioral_false_alarm_window_rate"
    )
    behavioral_false_alarm_targets = values(final, "cusum_false_alarm_target")

    mechanism_conditions = {
        "factorial_protocol_evaluable_and_frozen": evaluable,
        "capacity_structural_main_effect_positive": capacity_struct_ci[0] > 0.0,
        "capacity_prefers_structural_over_behavioral_change": (
            capacity_selectivity_ci[0] > 0.0
        ),
        "direction_prefers_behavioral_over_structural_change": (
            direction_selectivity_ci[0] > 0.0
        ),
        "fixed_estimand_capacity_is_structurally_selective": (
            estimand_capacity_selectivity_ci[0] > 0.0
        ),
        "fixed_estimand_direction_moves_under_behavioral_intervention": (
            estimand_direction_behav_ci[0] > 0.0
        ),
        "executed_behavioral_policy_moves_direction_D": all(
            _boolean(row.get("direction_manipulation_pass", False))
            for row in final
        ),
        "behavioral_only_false_alarm_at_or_below_target": bool(
            behavioral_false_alarm_window_rate
            and behavioral_false_alarm_targets
            and len(behavioral_false_alarm_window_rate) == len(behavioral_false_alarm_targets)
            and all(
                0.0 <= rate <= target
                for rate, target in zip(
                    behavioral_false_alarm_window_rate,
                    behavioral_false_alarm_targets,
                )
            )
        ),
        "causal_capacity_is_more_structurally_selective_than_correlation": (
            correlation_selectivity_ci[0] > 0.0
        ),
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
            "capacity_beta_structural": capacity_struct,
            "capacity_beta_structural_ci95": capacity_struct_ci,
            "capacity_structural_minus_abs_behavioral": capacity_selectivity,
            "capacity_selectivity_ci95": capacity_selectivity_ci,
            "direction_behavioral_minus_abs_structural": direction_selectivity,
            "direction_selectivity_ci95": direction_selectivity_ci,
            "fixed_estimand_capacity_selectivity_ci95": (
                estimand_capacity_selectivity_ci
            ),
            "fixed_estimand_direction_behavioral_ci95": estimand_direction_behav_ci,
            "behavioral_only_false_alarm_window_rate": behavioral_false_alarm_window_rate,
            "final_recovery_latency_mean": _mean(final, "recovery_latency"),
            "correlation_capacity_structural_minus_abs_behavioral": (
                correlation_selectivity
            ),
            "causal_minus_correlation_capacity_selectivity_ci95": (
                correlation_selectivity_ci
            ),
            "correlation_capacity_beta_structural_mean": _mean(
                correlation, "capacity_beta_structural"
            ),
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
        "- Legacy H3 uses matched-seed post-warm-up F1 as a diagnostic endpoint; it is not a Paper-A/Paper-B gate.\n",
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
