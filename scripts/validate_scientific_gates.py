"""Adjudicate the G0--G9 scientific gate ladder from run-scoped artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scientific_gate_common import (
    GATE_SCHEMA_VERSION, atomic_json, bootstrap_mean_ci, gate_record, load_csv, load_json,
    wilson_interval,
)
from scripts.run_latency_oracle import PROTOCOL_VERSION as LATENCY_ORACLE_PROTOCOL

PROTOCOL_VERSION = "scientific_gate_ladder_g0_g9_v3_provenance_and_total_memory"
EXIT_SUPPORTED = 0
EXIT_UNSUPPORTED = 10
EXIT_INVALID = 20
EXIT_SMOKE_ONLY = 11


def _boolish(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "supported"}


def _sha256(value):
    value = str(value or "").strip().lower()
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _h1_capacity_gate(run_root, expected_threshold_fingerprint=None):
    h1_root = Path(run_root) / "h1"
    latest = load_json(h1_root / "latest_h1_claim.json", "H1 latest claim")
    claim_path = latest.get("claim_path")
    if not claim_path:
        raise ValueError("H1 latest claim does not identify its run-scoped claim artifact")
    if not _within(claim_path, h1_root):
        raise ValueError("H1 claim path escapes the current run's h1 artifact root")
    claim = load_json(claim_path, "H1 run-scoped claim")
    run_id = str(latest.get("run_id", ""))
    claim_run_id = str(claim.get("run_id", ""))
    manifest = load_json(Path(claim_path).parent / "manifest.json", "H1 run manifest")
    provenance_valid = bool(
        run_id and run_id == claim_run_id == str(manifest.get("run_id", ""))
        and manifest.get("status") == "complete"
        and manifest.get("protocol_version") == claim.get("protocol_version")
        and not bool(manifest.get("threshold_calibration", {}).get("development_defaults", True))
        and (
            expected_threshold_fingerprint is None
            or manifest.get("threshold_calibration", {}).get("fingerprint")
            == expected_threshold_fingerprint
        )
    )
    passed = bool(provenance_valid and claim.get("h1b_capacity_recovery_pass", False))
    main = claim.get("h1_main", {})
    return gate_record(
        "G5", passed, required=True,
        metrics={
            "h1_protocol_version": claim.get("protocol_version"),
            "run_id": run_id,
            "provenance_valid": provenance_valid,
            "threshold_fingerprint_match": (
                expected_threshold_fingerprint is None
                or manifest.get("threshold_calibration", {}).get("fingerprint")
                == expected_threshold_fingerprint
            ),
            "capacity_recovery_pass": passed,
            "capacity_active_spearman_mean": main.get("capacity_active_spearman_mean"),
            "capacity_active_normalized_mae_mean": main.get("capacity_active_normalized_mae_mean"),
            "oracle_core_f1_adjusted_mean": main.get("oracle_core_f1_adjusted_mean"),
            "capacity_null_fpr_mean": main.get("capacity_null_fpr_mean"),
        },
        rule="learned C-hat must recover held-out oracle C* after oracle viability gates pass",
        failure_action="classify the learned estimator as the bottleneck; do not tune Paper-B downstream selectors",
    )


def _latency_gate(run_root):
    path = Path(run_root) / "latency_oracle.json"
    if not path.is_file():
        return gate_record(
            "G6", False, required=True,
            metrics={"artifact_present": False},
            rule="lag-specific oracle must test the retained latency estimand",
            failure_action="report the retained latency-recovery claim as unsupported",
        )
    payload = load_json(path, "latency oracle")
    protocol_ok = payload.get("protocol_version") == LATENCY_ORACLE_PROTOCOL
    passed = bool(protocol_ok and payload.get("gate_pass", False))
    return gate_record(
        "G6", passed, required=True,
        metrics={
            "artifact_present": True,
            "gate_pass": bool(payload.get("gate_pass", False)),
            "protocol_version": payload.get("protocol_version"),
            "protocol_match": protocol_ok,
            "latency_onset_fraction": payload.get("latency_onset_fraction"),
            "latency_onset_rule": payload.get("latency_onset_rule"),
        },
        rule="lag-specific oracle latency gate must pass to support latency recovery",
        failure_action="retain the latency estimand but report latency recovery as unsupported",
    )


def _cusum_gate(run_root, min_windows):
    rows = load_csv(Path(run_root) / "h2" / "summary_h2.csv", "H2 summary")
    selected = [row for row in rows if row.get("model") == "Final-CIGAMF"]
    if not selected:
        raise ValueError("H2 summary contains no Final-CIGAMF rows")
    windows_by_seed, alarms_by_seed, targets, references = [], [], [], []
    for row in selected:
        windows = int(row.get("behavioral_monitoring_window_count", -1))
        alarms = int(row.get("behavioral_false_alarm_window_count", -1))
        rate = float(row.get("behavioral_false_alarm_window_rate", "nan"))
        target = float(row.get("cusum_false_alarm_target", "nan"))
        horizon = int(row.get("cusum_monitoring_horizon", 0))
        reference = str(row.get("cusum_calibration_reference_config_hash", "")).strip()
        if (
            windows <= 0 or alarms < 0 or alarms > windows or horizon <= 0
            or not math.isfinite(rate) or not math.isfinite(target)
            or not (0.0 < target < 1.0)
            or abs(rate - alarms / windows) > 1e-9 or not _sha256(reference)
        ):
            raise ValueError(
                "H2 Final-CIGAMF row contains invalid window-matched CUSUM false-alarm fields"
            )
        windows_by_seed.append(windows)
        alarms_by_seed.append(alarms)
        targets.append(target)
        references.append(reference.lower())
    if max(targets) - min(targets) > 1e-12:
        raise ValueError("H2 rows do not share one frozen CUSUM false-alarm target")
    if len(set(references)) != 1:
        raise ValueError("H2 rows do not share one frozen CUSUM calibration reference")
    total_windows = int(sum(windows_by_seed))
    total_alarms = int(sum(alarms_by_seed))
    pooled_rate = float(total_alarms / total_windows) if total_windows else float("nan")
    interval = wilson_interval(total_alarms, total_windows)
    target = float(targets[0])
    passed = bool(
        total_windows >= int(min_windows)
        and all(value > 0 for value in windows_by_seed)
        and math.isfinite(pooled_rate) and pooled_rate <= target
        and math.isfinite(interval[1]) and interval[1] <= target
    )
    return gate_record(
        "G7", passed, required=False,
        metrics={
            "monitoring_windows_by_seed": windows_by_seed,
            "false_alarm_windows_by_seed": alarms_by_seed,
            "monitoring_window_count": total_windows,
            "false_alarm_window_count": total_alarms,
            "behavioral_false_alarm_window_rate": pooled_rate,
            "behavioral_false_alarm_wilson95": interval,
            "frozen_target": target,
            "calibration_reference_config_hash": references[0],
            "min_monitoring_windows": int(min_windows),
        },
        rule=(
            "on behavior-only/no-structural-change runs, false alarms are counted "
            "over detector-ready non-overlapping windows matched to the calibration "
            "horizon; the pooled observed window rate must stay at or below the "
            "single frozen no-change target, its 95% Wilson upper bound must stay "
            "at or below that target, and every confirmatory seed must expose at "
            "least one monitoring window"
        ),
        failure_action="delete/gate out the trigger and structural-tracking contribution",
    )


def _external_gate(external_root, min_seeds, min_episodes):
    if not external_root:
        return gate_record(
            "G8", False, required=False,
            metrics={"external_artifact_present": False},
            rule="a pinned second benchmark must show active response support and paired Final-CIGAMF reward advantage over PureMeanField",
            failure_action="scope claims explicitly to the custom/primary domain only",
        )
    root = Path(external_root)
    manifest = load_json(root / "manifest.json", "external training manifest")
    rows = load_csv(root / "summary_external_training.csv", "external training")
    if manifest.get("profile") != "full":
        return gate_record(
            "G8", False, required=False,
            metrics={"reason": "external profile is not full", "profile": manifest.get("profile")},
            rule="only full-profile paired external runs count as generalisation evidence",
            failure_action="scope claims explicitly to the custom/primary domain only",
        )
    by_model = {}
    for row in rows:
        seed = int(row["seed"])
        model = row.get("model")
        if seed in by_model.setdefault(model, {}):
            raise ValueError(f"duplicate external row for model={model} seed={seed}")
        by_model[model][seed] = row
    final = by_model.get("Final-CIGAMF", {})
    pure = by_model.get("PureMeanField", {})
    common = sorted(set(final).intersection(pure))
    manifest_seeds = sorted(int(seed) for seed in manifest.get("seeds", []))
    exact_pairing = bool(common and common == manifest_seeds)
    deltas = [float(final[s]["mean_reward"]) - float(pure[s]["mean_reward"]) for s in common]
    if not all(math.isfinite(value) for value in deltas):
        raise ValueError("external reward comparison contains non-finite values")
    ci = bootstrap_mean_ci(deltas, seed=4800)
    support = manifest.get("h1_support_by_seed", {})
    support_ready = bool(
        exact_pairing
        and set(support) == {str(seed) for seed in common}
        and all(bool(support[str(seed)].get("signal_ready", False)) for seed in common)
    )
    episode_budget_ok = bool(
        int(manifest.get("episodes", 0)) >= int(min_episodes)
        and all(
            int(final[seed].get("episodes", 0)) >= int(min_episodes)
            and int(pure[seed].get("episodes", 0)) >= int(min_episodes)
            for seed in common
        )
    )
    step_cap_ok = bool(
        all(
            int(row.get("max_steps_effective", -1))
            == int(row.get("max_steps_requested", -2))
            for model_rows in (final, pure) for row in model_rows.values()
        )
    )
    config_match = bool(
        all(
            str(final[seed].get("config_fingerprint_sha256", ""))
            == str(pure[seed].get("config_fingerprint_sha256", ""))
            and bool(str(final[seed].get("config_fingerprint_sha256", "")))
            for seed in common
        )
    )
    passed = bool(
        manifest.get("provenance_complete") is True
        and manifest.get("source_git_clean") is True
        and manifest.get("external_pin_match") is True
        and manifest.get("paired_generalization_models_present") is True
        and exact_pairing
        and len(common) >= int(min_seeds)
        and episode_budget_ok and step_cap_ok and config_match
        and support_ready
        and math.isfinite(ci[0]) and ci[0] > 0.0
    )
    return gate_record(
        "G8", passed, required=False,
        metrics={
            "environment": manifest.get("environment"),
            "external_pin_match": manifest.get("external_pin_match"),
            "provenance_complete": manifest.get("provenance_complete"),
            "source_git_clean": manifest.get("source_git_clean"),
            "paired_seed_count": len(common),
            "min_paired_seeds": int(min_seeds),
            "exact_manifest_seed_pairing": exact_pairing,
            "minimum_episodes": int(min_episodes),
            "episode_budget_ok": episode_budget_ok,
            "effective_max_steps_match_requested": step_cap_ok,
            "paired_config_fingerprint_match": config_match,
            "Final_minus_PureMeanField_reward_by_seed": deltas,
            "reward_advantage_ci95": ci,
            "active_h1_support_all_seeds": support_ready,
        },
        rule="second benchmark must be pinned/provenanced, use exact paired full-profile seeds/configs with the preregistered minimum training budget and effective step cap, exhibit active action-response support on every seed, and show a positive paired reward advantage over PureMeanField",
        failure_action="scope claims explicitly to the custom/primary domain only",
    )


def _pareto_gate(run_root, g7_passed, core_gates_passed):
    paper_a = load_json(Path(run_root) / "paper_a_claim_status.json", "Paper-A claim report")
    paper_b = load_json(Path(run_root) / "paper_b_claim_status.json", "Paper-B claim report")
    a_core = bool(paper_a.get("submitted_claim_set_supported", False))
    b_core = str(paper_b.get("overall_status", "")) == "SUPPORTED"
    conditions = paper_b.get("conditions", {})
    ci = paper_b.get("paired_ci95", {})
    nondominated = bool(conditions.get("semantic_memory_extends_reward_compute_pareto_frontier", False))

    def lower(name):
        value = ci.get(name, [float("nan"), float("nan")])
        try:
            return float(value[0])
        except Exception:
            return float("nan")

    reward_low = lower("scaling_semantic_minus_pure_mean_field_reward")
    throughput_low = lower("scaling_semantic_minus_full_throughput")
    memory_low = lower("scaling_full_minus_semantic_memory_bytes")
    peak_memory_low = lower("scaling_full_minus_semantic_peak_runtime_memory_bytes")
    passed = bool(
        core_gates_passed and a_core and b_core and g7_passed and nondominated
        and reward_low > 0.0 and throughput_low > 0.0
        and memory_low > 0.0 and peak_memory_low > 0.0
    )
    return gate_record(
        "G9", passed, required=False,
        metrics={
            "G0_G5_core_gates_supported": bool(core_gates_passed),
            "paper_A_core_supported": a_core,
            "paper_B_selective_representation_supported": b_core,
            "G7_trigger_supported": bool(g7_passed),
            "semantic_nondominated": nondominated,
            "reward_vs_PureMeanField_ci95_low": reward_low,
            "throughput_vs_FullExplicit_ci95_low": throughput_low,
            "memory_saving_vs_FullExplicit_ci95_low": memory_low,
            "peak_runtime_memory_saving_vs_FullExplicit_ci95_low": peak_memory_low,
        },
        rule="the scalable-system claim requires the G0-G5 causal/allocation core, supported Paper A and Paper B claim sets, the retained G7 trigger, and a reward-throughput-total-memory Pareto improvement over structural-blind/full-explicit controls; total memory includes a separately measured peak runtime endpoint",
        failure_action="drop the 'scalable system' claim and retain only the selective-representation architecture claim",
    )


def validate(
    run_root, prechecks, external_root, protocol_mode, min_external_seeds,
    min_cusum_windows=72, min_external_episodes=50,
):
    early = load_json(prechecks, "G0-G4 precheck")
    if early.get("protocol_version") != "scientific_prechecks_g0_g4_v3_disagreement_capture":
        raise ValueError("G0-G4 artifact uses an incompatible protocol")
    if protocol_mode == "quick":
        return {
            "schema_version": GATE_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_mode": protocol_mode,
            "overall_status": "SMOKE_ONLY",
            "gates": early.get("gates", {}),
        }, EXIT_SMOKE_ONLY
    early_gates = early.get("gates")
    if not isinstance(early_gates, dict) or set(early_gates) != {"G0", "G1", "G2", "G3", "G4"}:
        raise ValueError("G0-G4 artifact is incomplete")

    gates = dict(early_gates)
    gates["G5"] = _h1_capacity_gate(
        run_root, early.get("threshold_calibration_fingerprint")
    )
    gates["G6"] = _latency_gate(run_root)
    gates["G7"] = _cusum_gate(run_root, min_cusum_windows)
    gates["G8"] = _external_gate(
        external_root, min_external_seeds, min_external_episodes
    )
    core_before_g9 = all(
        bool(gates[name]["passed"]) for name in ("G0", "G1", "G2", "G3", "G4", "G5")
    )
    gates["G9"] = _pareto_gate(
        run_root, gates["G7"]["passed"], core_before_g9
    )

    required_core = ["G0", "G1", "G2", "G3", "G4", "G5"]
    core_pass = all(bool(gates[name]["passed"]) for name in required_core)
    paper_a_full_pass = bool(core_pass and gates["G6"]["passed"] and gates["G7"]["passed"])
    scope = {
        "paper_A_QCD_selectivity": "SUPPORTED" if core_pass else "NOT_SUPPORTED",
        "paper_A_full_claim_set": "SUPPORTED" if paper_a_full_pass else "NOT_SUPPORTED",
        "latency": "SUPPORTED" if gates["G6"]["passed"] else "RETAINED_BUT_RECOVERY_UNSUPPORTED",
        "trigger_tracking": "KEEP" if gates["G7"]["passed"] else "DELETE_OR_GATE_OUT",
        "generalisation": "SECOND_BENCHMARK" if gates["G8"]["passed"] else "CUSTOM_DOMAIN_ONLY",
        "system_claim": "SCALABLE_SYSTEM" if gates["G9"]["passed"] else "SELECTIVE_REPRESENTATION_ARCHITECTURE_ONLY",
    }
    overall = "SUPPORTED" if paper_a_full_pass else "NOT_SUPPORTED"
    payload = {
        "schema_version": GATE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_mode": protocol_mode,
        "overall_status": overall,
        "core_gates_pass": core_pass,
        "required_core_gates": required_core,
        "gates": gates,
        "claim_scope": scope,
        "interpretation": (
            "G0-G5 adjudicate the causal/allocation core. G6 latency and G7 "
            "tracking are retained Paper-A H3 gates; failure leaves H1/H2 "
            "reportable but prevents full submitted Paper-A support. G8-G9 "
            "scope generalisation and scalable-system claims."
        ),
    }
    return payload, EXIT_SUPPORTED if paper_a_full_pass else EXIT_UNSUPPORTED


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--prechecks", required=True)
    ap.add_argument("--external-root", default=None)
    ap.add_argument("--min-external-seeds", type=int, default=5)
    ap.add_argument("--min-external-episodes", type=int, default=50)
    ap.add_argument("--min-cusum-windows", type=int, default=72)
    ap.add_argument("--protocol-mode", choices=("quick", "confirmatory"), default="confirmatory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = args.out or str(Path(args.run_root) / "scientific_gate_status.json")
    try:
        payload, code = validate(
            os.path.abspath(args.run_root), os.path.abspath(args.prechecks),
            None if args.external_root is None else os.path.abspath(args.external_root),
            args.protocol_mode, args.min_external_seeds,
            args.min_cusum_windows, args.min_external_episodes,
        )
    except Exception as exc:
        payload = {
            "schema_version": GATE_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "overall_status": "INVALID",
            "error_type": type(exc).__name__, "error": str(exc),
        }
        code = EXIT_INVALID
    atomic_json(out, payload)
    print(json.dumps(payload, indent=2, default=float))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
