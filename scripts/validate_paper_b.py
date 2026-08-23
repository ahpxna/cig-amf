"""Validate Paper-B allocation, pair-latent, and fixed-core periphery panels."""

import argparse
import csv
import json
import math
import os
import sys
import tempfile

try:
    import validate_claims as VC
except ModuleNotFoundError:
    from scripts import validate_claims as VC


EXPECTED_ALLOCATION = {
    "C-Core", "AbsD-Core", "Random-Core", "Correlation-Core",
    "Oracle-C-Core", "Full-Explicit",
}
EXPECTED_PAIR = {
    "Aggregate", "Explicit-FF-BC", "Recurrent-BC", "Recurrent-BC-CD",
    "Recurrent-BC-CD-Contrastive",
}
EXPECTED_PERIPHERY = {
    "Semantic-Free", "Semantic-Only", "Unconstrained", "No-Aux", "Single-Mean",
}


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".paper-b-", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=float)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_panel(root, directory, filename, expected_variants, expected_seeds):
    panel_root = os.path.join(root, directory)
    with open(os.path.join(panel_root, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("complete"):
        raise ValueError(f"{directory} manifest is incomplete")
    with open(os.path.join(panel_root, filename), newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    variants = {row["variant"] for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    if variants != set(expected_variants) or seeds != set(expected_seeds):
        raise ValueError(
            f"{directory} matrix mismatch: variants={sorted(variants)}, seeds={sorted(seeds)}"
        )
    return rows, manifest


def _paired(rows, treatment, control, metric, panel=None):
    selected = [
        row for row in rows
        if panel is None or row.get("panel") == panel
    ]
    by_variant = {}
    for row in selected:
        by_variant.setdefault(row["variant"], {})[int(row["seed"])] = row
    left = by_variant[treatment]
    right = by_variant[control]
    if set(left) != set(right):
        raise ValueError(f"unpaired {treatment} versus {control}")
    output = []
    for seed in sorted(left):
        a = float(left[seed][metric])
        b = float(right[seed][metric])
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError(f"non-finite {metric} at seed {seed}")
        output.append(a - b)
    return output


def validate(run_root, expected_seeds, protocol_mode):
    allocation, _ = _load_panel(
        run_root, "paper_b_allocation", "summary_paper_b_allocation.csv",
        EXPECTED_ALLOCATION, expected_seeds,
    )
    pair_rows, _ = _load_panel(
        run_root, "paper_b_pair_latent", "summary_paper_b_pair_latent.csv",
        EXPECTED_PAIR, expected_seeds,
    )
    periphery_rows, _ = _load_panel(
        run_root, "paper_b_periphery", "summary_paper_b_periphery.csv",
        EXPECTED_PERIPHERY, expected_seeds,
    )
    if protocol_mode == "quick":
        return {
            "paper": "B", "overall_status": "SMOKE_ONLY",
            "panels_complete": True,
        }, VC.EXIT_SMOKE_ONLY

    c_vs_d = _paired(
        allocation, "C-Core", "AbsD-Core", "selector_oracle_f1",
        panel="selector_isolation",
    )
    oracle_vs_random = _paired(
        allocation, "Oracle-C-Core", "Random-Core", "selector_oracle_f1",
        panel="selector_isolation",
    )
    c_reward_vs_d = _paired(
        allocation, "C-Core", "AbsD-Core", "mean_reward", panel="end_to_end"
    )
    pair_retrieval = [
        -value for value in _paired(
            pair_rows, "Recurrent-BC-CD-Contrastive", "Recurrent-BC",
            "cd_retrieval_mae",
        )
    ]
    periphery_reward = _paired(
        periphery_rows, "Semantic-Free", "Single-Mean", "mean_reward"
    )
    metrics = {
        "c_core_minus_absd_selector_f1": c_vs_d,
        "oracle_minus_random_selector_f1": oracle_vs_random,
        "c_core_minus_absd_reward": c_reward_vs_d,
        "bc_minus_full_cd_retrieval_mae": pair_retrieval,
        "semantic_free_minus_single_mean_reward": periphery_reward,
    }
    cis = {
        key: VC._bootstrap_mean_ci(value, seed=4100 + index)
        for index, (key, value) in enumerate(metrics.items())
    }
    conditions = {
        "C_selector_beats_absD_at_equal_budget": cis[
            "c_core_minus_absd_selector_f1"
        ][0] > 0.0,
        "oracle_selector_beats_random_at_equal_budget": cis[
            "oracle_minus_random_selector_f1"
        ][0] > 0.0,
        "C_allocation_improves_end_to_end_reward_over_absD": cis[
            "c_core_minus_absd_reward"
        ][0] > 0.0,
        "CD_contrastive_latent_improves_profile_retrieval": cis[
            "bc_minus_full_cd_retrieval_mae"
        ][0] > 0.0,
        "semantic_free_memory_improves_reward_over_single_mean": cis[
            "semantic_free_minus_single_mean_reward"
        ][0] > 0.0,
    }
    supported = all(conditions.values())
    return {
        "paper": "B",
        "overall_status": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "conditions": conditions,
        "paired_metrics": metrics,
        "paired_ci95": cis,
    }, VC.EXIT_SUPPORTED if supported else VC.EXIT_UNSUPPORTED


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--protocol-mode", choices=("quick", "confirmatory"),
                        default="confirmatory")
    args = parser.parse_args(argv)
    output = os.path.join(os.path.abspath(args.run_root), "paper_b_claim_status.json")
    try:
        report, code = validate(
            os.path.abspath(args.run_root), args.expected_seeds,
            args.protocol_mode,
        )
    except Exception as exc:
        report = {
            "paper": "B", "overall_status": "INVALID",
            "error_type": type(exc).__name__, "error": str(exc),
        }
        code = VC.EXIT_INVALID
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, default=float))
    return code


if __name__ == "__main__":
    sys.exit(main())
