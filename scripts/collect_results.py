"""Validate and aggregate fresh H1/H2/H3 summaries.

Default paths remain ``results/h1``, ``results/h2``, and ``results/h3``.
Passing ``--run-root`` instead reads the three experiment directories below
that isolated root and writes ``summary_tables.md`` beside them.

The collector rejects partial seed matrices, duplicate rows, and non-finite
claim metrics. H2 additionally requires the complete-attempt manifest written
by ``run_h2_selectivity.py``; a failed/in-progress attempt can therefore never
fall back to an older CSV.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics as st
import tempfile

try:
    from exp_common import ROOT
except ModuleNotFoundError:  # Support ``import scripts.collect_results`` in tests.
    from scripts.exp_common import ROOT


H1_VARIANTS = {
    "dr_eps003",
    "plugin_eps005",
    "dr_eps000",
    "dr_eps001",
    "dr_eps005",
    "dr_eps008",
    "dr_eps012",
}
H3_VARIANTS = {
    "Full-CIGAMF",
    "Scalar-Only",
    "Unconstrained-NoSemantic",
    "No-AuxLoss",
    "Fixed-Cardinality",
    "NoMultiMemory-SingleMean",
}
H2_PROTOCOL_VERSION = "h2_matched_change_v2"

SPECS = {
    "h1": {
        "title": "H1 — RQ1: causal identification and calibration",
        "filename": "summary_h1.csv",
        "group_key": "variant",
        "expected_groups": H1_VARIANTS,
        "metrics": [
            "rank_correlation_mean",
            "signed_spearman_mean",
            "sign_agreement_mean",
            "bias_mean",
            "mae_mean",
        ],
        "required_finite": [
            "rank_correlation_mean",
            "signed_spearman_mean",
            "sign_agreement_mean",
            "bias_mean",
            "mae_mean",
        ],
    },
    "h2": {
        "title": "H2 — RQ2: tier separation and selectivity (Eq. 33)",
        "filename": "summary_h2.csv",
        "group_key": "model",
        "expected_groups": None,  # Read from the complete H2 manifest.
        "metrics": [
            "delta_behav",
            "delta_background_structural_run",
            "delta_background_behavioral_run",
            "delta_struct",
            "SR_cross_run",
            "recovery_latency",
            "n_shift_events",
            "n_shift_with_trigger",
            "final_f1_struct",
        ],
        "required_finite": [
            "delta_behav",
            "delta_background_structural_run",
            "delta_background_behavioral_run",
            "delta_struct",
            "SR_cross_run",
            "recovery_latency",
            "n_shift_events",
            "n_shift_with_trigger",
            "final_f1_struct",
        ],
    },
    "h3": {
        "title": "H3 — RQ3: slot collapse and capacity allocation",
        "filename": "summary_h3.csv",
        "group_key": "variant",
        "expected_groups": H3_VARIANTS,
        "metrics": [
            "hard_usage_entropy_ratio",
            "assignment_mutual_info_ratio",
            "slot_cos_offdiag",
            "mean_core_size",
            "frac_k_at_kmax",
            "mean_reward",
            "throughput_total",
        ],
        "required_finite": [
            "mean_core_size",
            "frac_k_at_kmax",
            "mean_reward",
            "mean_f1",
            "throughput_total",
        ],
    },
}


class ResultValidationError(RuntimeError):
    pass


def _fnum(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_csv(path):
    if not os.path.exists(path):
        raise ResultValidationError(f"missing summary: {path}")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ResultValidationError(f"empty summary: {path}")
    return rows


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_h2_complete_rows(h2_dir):
    """Load only the immutable summary named by a complete H2 attempt."""
    marker_path = os.path.join(h2_dir, "latest_attempt.json")
    if not os.path.exists(marker_path):
        raise ResultValidationError(
            f"H2 completeness marker is missing: {marker_path}"
        )
    try:
        with open(marker_path, encoding="utf-8") as f:
            marker = json.load(f)
    except (OSError, ValueError) as exc:
        raise ResultValidationError(f"invalid H2 completeness marker: {exc}") from exc

    if marker.get("status") != "complete":
        raise ResultValidationError(
            "latest H2 attempt is not complete: "
            f"run_id={marker.get('run_id')} status={marker.get('status')}"
        )
    if marker.get("protocol_version") != H2_PROTOCOL_VERSION:
        raise ResultValidationError(
            "H2 marker does not use matched structural/behavioral change windows"
        )
    if not bool(marker.get("required_comparator_present", False)):
        raise ResultValidationError(
            "H2 is partial: required CorrelationMeanField comparator is absent"
        )
    if not bool(marker.get("required_claim_models_present", False)):
        raise ResultValidationError(
            "H2 is partial: Final-CIGAMF, CorrelationMeanField, and "
            "NoTwoTimescale are all required"
        )
    def attempt_keys(items):
        try:
            return {
                (str(item["model"]), str(item["mode"]), int(item["seed"]))
                for item in items
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ResultValidationError("H2 attempt matrix is malformed") from exc

    expected_items = marker.get("expected_attempts", [])
    completed_items = marker.get("completed_attempts", [])
    expected_attempts = attempt_keys(expected_items)
    completed_attempts = attempt_keys(completed_items)
    declared_attempts = {
        (str(model), mode, int(seed))
        for model in marker.get("models", [])
        for mode in ("behavioral_drift", "structural_shift")
        for seed in marker.get("seeds", [])
    }
    if (
        not expected_attempts
        or expected_attempts != declared_attempts
        or completed_attempts != expected_attempts
        or len(expected_items) != len(expected_attempts)
        or len(completed_items) != len(completed_attempts)
        or marker.get("failed_attempt") is not None
    ):
        raise ResultValidationError("H2 per-mode attempt matrix is incomplete")
    summary_ref = marker.get("summary_path")
    if not summary_ref:
        raise ResultValidationError("complete H2 marker has no summary_path")
    summary_path = (
        summary_ref
        if os.path.isabs(summary_ref)
        else os.path.join(ROOT, summary_ref)
    )
    rows = _read_csv(os.path.abspath(summary_path))

    expected_hash = marker.get("summary_sha256")
    if not expected_hash or _sha256(summary_path) != expected_hash:
        raise ResultValidationError("H2 summary checksum does not match its manifest")
    expected_count = int(marker.get("expected_summary_rows", -1))
    if len(rows) != expected_count or len(rows) != int(marker.get("summary_rows", -1)):
        raise ResultValidationError(
            f"H2 row count mismatch: expected {expected_count}, found {len(rows)}"
        )
    run_id = str(marker.get("run_id", ""))
    if not run_id or any(str(row.get("run_id", "")) != run_id for row in rows):
        raise ResultValidationError("H2 rows do not all belong to the published run_id")
    if any(row.get("protocol_version") != H2_PROTOCOL_VERSION for row in rows):
        raise ResultValidationError("H2 rows do not use the published protocol version")

    required = {"Final-CIGAMF", "CorrelationMeanField", "NoTwoTimescale"}
    required_rows = [row for row in rows if row.get("model") in required]
    for row in required_rows:
        if str(row.get("claim_evaluable", "")).lower() not in {"1", "true"}:
            raise ResultValidationError(
                f"H2 required arm is not evaluable: {row.get('model')}/seed{row.get('seed')}"
            )
        try:
            n_structural = int(row.get("n_complete_structural_windows", -1))
            n_behavioral = int(row.get("n_complete_behavioral_windows", -1))
        except (TypeError, ValueError) as exc:
            raise ResultValidationError("H2 matched-window counts are malformed") from exc
        if n_structural <= 0 or n_structural != n_behavioral:
            raise ResultValidationError(
                "H2 structural and behavioral post-change windows are not matched"
            )
        if row.get("model") == "NoTwoTimescale" and (
            row.get("runner_class") != "NoTwoTimescaleRunner"
            or row.get("ablation_contract")
            != "scheduler_only_graph_update_every_episode"
        ):
            raise ResultValidationError(
                "H2 NoTwoTimescale is not the faithful Final scheduler-only ablation"
            )
    return rows, marker


def _resolve_metadata_path(path_value):
    if not path_value:
        raise ResultValidationError("completeness metadata contains an empty path")
    return os.path.abspath(
        path_value if os.path.isabs(path_value) else os.path.join(ROOT, path_value)
    )


def _load_h1_complete_rows(h1_dir):
    pointer_path = os.path.join(h1_dir, "latest_complete_run.json")
    if not os.path.exists(pointer_path):
        raise ResultValidationError(
            f"H1 complete-run pointer is missing: {pointer_path}"
        )
    try:
        with open(pointer_path, encoding="utf-8") as f:
            pointer = json.load(f)
    except (OSError, ValueError) as exc:
        raise ResultValidationError(f"invalid H1 complete-run pointer: {exc}") from exc
    run_id = str(pointer.get("run_id", ""))
    if not run_id or pointer.get("protocol_version") != "h1_exact_v1":
        raise ResultValidationError("H1 pointer is not an h1_exact_v1 run")

    manifest_path = _resolve_metadata_path(pointer.get("manifest_path"))
    summary_path = _resolve_metadata_path(pointer.get("summary_path"))
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        raise ResultValidationError(f"invalid H1 manifest: {exc}") from exc
    if (
        manifest.get("status") != "complete"
        or str(manifest.get("run_id", "")) != run_id
        or manifest.get("protocol_version") != "h1_exact_v1"
        or manifest.get("failed_attempts")
    ):
        raise ResultValidationError("H1 manifest is incomplete, failed, or mismatched")
    if _resolve_metadata_path(manifest.get("summary_path")) != summary_path:
        raise ResultValidationError("H1 pointer and manifest name different summaries")

    def attempts(items):
        return {
            (str(item["variant"]), int(item["seed"]))
            for item in items
        }

    expected = attempts(manifest.get("expected_attempts", []))
    completed = attempts(manifest.get("completed_attempts", []))
    if not expected or completed != expected:
        raise ResultValidationError(
            f"H1 attempt matrix is incomplete: missing={sorted(expected - completed)}"
        )
    rows = _read_csv(summary_path)
    if len(rows) != int(manifest.get("row_count", -1)) or len(rows) != len(expected):
        raise ResultValidationError("H1 summary row count does not match its manifest")
    actual = {(str(row.get("variant")), int(row.get("seed"))) for row in rows}
    if actual != expected:
        raise ResultValidationError("H1 summary rows do not match expected attempts")
    for row in rows:
        if (
            str(row.get("run_id", "")) != run_id
            or row.get("protocol_version") != "h1_exact_v1"
            or str(row.get("attempt_complete", "")).lower() not in {"true", "1"}
            or not str(row.get("config_fingerprint", ""))
        ):
            raise ResultValidationError("H1 summary contains stale or incomplete rows")
    return rows, pointer, manifest


def _load_h3_complete_rows(h3_dir):
    attempt_path = os.path.join(h3_dir, "attempt.json")
    if not os.path.exists(attempt_path):
        raise ResultValidationError(
            f"H3 completeness marker is missing: {attempt_path}"
        )
    try:
        with open(attempt_path, encoding="utf-8") as f:
            attempt = json.load(f)
    except (OSError, ValueError) as exc:
        raise ResultValidationError(f"invalid H3 attempt metadata: {exc}") from exc
    if attempt.get("status") != "complete" or attempt.get("failed_run") is not None:
        raise ResultValidationError(
            "latest H3 attempt is not complete: "
            f"status={attempt.get('status')} failed_run={attempt.get('failed_run')}"
        )
    expected = {
        (str(item["variant"]), int(item["seed"]))
        for item in attempt.get("expected_runs", [])
    }
    completed = {
        (str(item["variant"]), int(item["seed"]))
        for item in attempt.get("completed_runs", [])
    }
    if not expected or completed != expected:
        raise ResultValidationError(
            f"H3 run matrix is incomplete: missing={sorted(expected - completed)}"
        )
    rows = _read_csv(os.path.join(h3_dir, "summary_h3.csv"))
    actual = {(str(row.get("variant")), int(row.get("seed"))) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ResultValidationError("H3 summary rows do not match attempt metadata")
    return rows, attempt


def _validate_seed_matrix(name, rows, group_key, expected_groups, expected_seeds):
    seen = []
    for row in rows:
        group = row.get(group_key)
        try:
            seed = int(row.get("seed", ""))
        except (TypeError, ValueError) as exc:
            raise ResultValidationError(f"{name}: row has invalid seed: {row}") from exc
        seen.append((group, seed))

    if len(seen) != len(set(seen)):
        raise ResultValidationError(f"{name}: duplicate group/seed rows detected")
    actual_groups = {group for group, _ in seen}
    if expected_groups is not None and actual_groups != set(expected_groups):
        raise ResultValidationError(
            f"{name}: expected groups {sorted(expected_groups)}, found {sorted(actual_groups)}"
        )

    groups = set(expected_groups) if expected_groups is not None else actual_groups
    if expected_seeds is None:
        seed_sets = {
            group: {seed for row_group, seed in seen if row_group == group}
            for group in groups
        }
        distinct = {tuple(sorted(values)) for values in seed_sets.values()}
        if len(distinct) != 1:
            raise ResultValidationError(f"{name}: groups contain inconsistent seed sets")
        return sorted(next(iter(seed_sets.values()))) if seed_sets else []

    expected_pairs = {
        (group, int(seed)) for group in groups for seed in expected_seeds
    }
    if set(seen) != expected_pairs:
        missing = sorted(expected_pairs - set(seen))
        extra = sorted(set(seen) - expected_pairs)
        raise ResultValidationError(
            f"{name}: incomplete seed matrix; missing={missing}, extra={extra}"
        )
    return list(expected_seeds)


def _validate_finite(name, rows, metrics):
    for row in rows:
        # PureMeanField can be requested explicitly for H2, but Eq. 33 is
        # undefined without W. Only that declared non-applicable case may have
        # empty/non-finite W metrics.
        if name == "h2" and str(row.get("not_applicable_reason", "")) == (
            "runner_has_no_influence_matrix_W"
        ):
            continue
        missing = [metric for metric in metrics if _fnum(row.get(metric)) is None]
        if missing:
            identity = row.get("variant", row.get("model", "?"))
            raise ResultValidationError(
                f"{name}: non-finite required metrics for {identity}/seed{row.get('seed')}: "
                + ", ".join(missing)
            )
        if name == "h3" and row.get("variant") != "NoMultiMemory-SingleMean":
            slot_metrics = (
                "hard_usage_entropy_ratio",
                "assignment_mutual_info_ratio",
                "slot_cos_offdiag",
            )
            missing_slot = [
                metric for metric in slot_metrics if _fnum(row.get(metric)) is None
            ]
            if missing_slot:
                raise ResultValidationError(
                    f"h3: non-finite slot metrics for {row.get('variant')}/"
                    f"seed{row.get('seed')}: " + ", ".join(missing_slot)
                )


def _render_table(rows, group_key, metrics):
    groups = {}
    for row in rows:
        groups.setdefault(row.get(group_key, "?"), []).append(row)

    out = [
        "| " + group_key + " | n | " + " | ".join(metrics) + " |",
        "|" + "---|" * (len(metrics) + 2),
    ]
    for group in sorted(groups):
        group_rows = groups[group]
        cells = []
        for metric in metrics:
            values = [
                number
                for number in (_fnum(row.get(metric)) for row in group_rows)
                if number is not None
            ]
            if not values:
                cells.append("N/A")
            elif len(values) == 1:
                cells.append(f"{values[0]:.3f}")
            else:
                cells.append(f"{st.mean(values):.3f} ± {st.pstdev(values):.3f}")
        out.append(f"| {group} | {len(group_rows)} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _atomic_write_text(path, text):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-collect-", suffix=".md", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def collect(
    run_root,
    expected_seeds=None,
    expected_h1_seeds=None,
    expected_h2_seeds=None,
    expected_h3_seeds=None,
):
    run_root = os.path.abspath(run_root)
    expected_by_name = {
        "h1": expected_h1_seeds if expected_h1_seeds is not None else expected_seeds,
        "h2": expected_h2_seeds if expected_h2_seeds is not None else expected_seeds,
        "h3": expected_h3_seeds if expected_h3_seeds is not None else expected_seeds,
    }
    loaded = {}
    h2_marker = None
    for name, spec in SPECS.items():
        experiment_dir = os.path.join(run_root, name)
        expected_for_name = expected_by_name[name]
        if name == "h1":
            rows, h1_pointer, h1_manifest = _load_h1_complete_rows(experiment_dir)
            expected_groups = {
                str(item["variant"])
                for item in h1_manifest.get("expected_attempts", [])
            }
            if expected_groups != H1_VARIANTS:
                raise ResultValidationError(
                    "H1 is partial: expected all preregistered variants; "
                    f"found {sorted(expected_groups)}"
                )
            marker_seeds = sorted({
                int(item["seed"])
                for item in h1_manifest.get("expected_attempts", [])
            })
            if expected_for_name is not None and marker_seeds != sorted(expected_for_name):
                raise ResultValidationError(
                    f"h1: manifest seeds {marker_seeds} do not match "
                    f"expected seeds {list(expected_for_name)}"
                )
        elif name == "h2":
            rows, h2_marker = _load_h2_complete_rows(experiment_dir)
            expected_groups = set(h2_marker.get("models", []))
            marker_seeds = [int(seed) for seed in h2_marker.get("seeds", [])]
            if expected_for_name is not None and marker_seeds != list(expected_for_name):
                raise ResultValidationError(
                    f"h2: manifest seeds {marker_seeds} do not match "
                    f"expected seeds {list(expected_for_name)}"
                )
        elif name == "h3":
            rows, h3_attempt = _load_h3_complete_rows(experiment_dir)
            expected_groups = {
                str(item["variant"])
                for item in h3_attempt.get("expected_runs", [])
            }
            if expected_groups != H3_VARIANTS:
                raise ResultValidationError(
                    "H3 is partial: expected the complete ablation matrix; "
                    f"found {sorted(expected_groups)}"
                )
            marker_seeds = [int(seed) for seed in h3_attempt.get("seeds", [])]
            if expected_for_name is not None and marker_seeds != list(expected_for_name):
                raise ResultValidationError(
                    f"h3: attempt seeds {marker_seeds} do not match "
                    f"expected seeds {list(expected_for_name)}"
                )
        _validate_seed_matrix(
            name,
            rows,
            spec["group_key"],
            expected_groups,
            expected_for_name,
        )
        _validate_finite(name, rows, spec["required_finite"])
        loaded[name] = rows

    markdown = ["# CIG-AMF validated experiment summary (mean ± population SD)\n"]
    for name, spec in SPECS.items():
        markdown.append(f"\n## {spec['title']}\n\n")
        markdown.append(
            _render_table(loaded[name], spec["group_key"], spec["metrics"])
        )

    markdown.append("\n## Claim gates\n\n")
    markdown.append(
        "- H1 passes only if the DR condition improves ranking over plug-in and "
        "the observational control, with sign agreement at least 0.75.\n"
    )
    markdown.append(
        "- H2 passes only if Final-CIGAMF has SR clearly above 1, the "
        "CorrelationMeanField observational comparator is near 1, and recovery "
        "is observed with matched triggers. NoTwoTimescale isolates the scheduler; "
        "PureMeanField cannot define Eq. 33 because it has no W matrix.\n"
    )
    markdown.append(
        "- H3 passes only if slot entropy remains high while off-diagonal slot "
        "cosine is low, semantic/auxiliary ablations degrade, and the "
        "Fixed-Cardinality control is "
        "verified as constant rather than adaptive. The decision endpoint is "
        "matched-seed post-warm-up F1 against the faithful single-mean "
        "ablation; reward and common-probe throughput are reported separately.\n"
    )
    text = "".join(markdown)
    output_path = os.path.join(run_root, "summary_tables.md")
    _atomic_write_text(output_path, text)
    return output_path, text


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        default=os.path.join(ROOT, "results"),
        help=(
            "Root containing h1/, h2/, and h3/ (absolute or relative to the "
            "repository root). Default: results/"
        ),
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Require the exact listed seed matrix in every hypothesis summary",
    )
    parser.add_argument("--expected-h1-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--expected-h2-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--expected-h3-seeds", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)
    run_root = (
        args.run_root
        if os.path.isabs(args.run_root)
        else os.path.join(ROOT, args.run_root)
    )
    try:
        output_path, text = collect(
            run_root,
            expected_seeds=args.expected_seeds,
            expected_h1_seeds=args.expected_h1_seeds,
            expected_h2_seeds=args.expected_h2_seeds,
            expected_h3_seeds=args.expected_h3_seeds,
        )
    except ResultValidationError as exc:
        parser.exit(2, f"[collect][INVALID] {exc}\n")
    print(text)
    print(f"[collect] saved {output_path}")
    return 0


if __name__ == "__main__":
    main()
