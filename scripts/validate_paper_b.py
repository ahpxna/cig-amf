"""Validate Paper-B allocation, pair-latent, and fixed-core periphery panels."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile

try:
    import validate_claims as VC
    import run_paper_b_allocation as PB_ALLOCATION
    import run_paper_b_pair_latent as PB_PAIR
    import run_paper_b_periphery as PB_PERIPHERY
    import run_paper_b_scaling as PB_SCALING
except ModuleNotFoundError:
    from scripts import validate_claims as VC
    from scripts import run_paper_b_allocation as PB_ALLOCATION
    from scripts import run_paper_b_pair_latent as PB_PAIR
    from scripts import run_paper_b_periphery as PB_PERIPHERY
    from scripts import run_paper_b_scaling as PB_SCALING
from utils.paper_contracts import (
    PAPER_B_CANDIDATE_RECALL_PROTOCOL_VERSION,
    PAPER_B_PROMOTION_WINDOW_STEPS,
    PAPER_B_SELECTOR_ORACLE_HORIZON,
)


# Derive matrix contracts from the producing runners so a producer update
# cannot leave the final validator silently pinned to a stale variant set.
EXPECTED_ALLOCATION = set(PB_ALLOCATION.VARIANTS)
EXPECTED_PAIR = set(PB_PAIR.VARIANTS)
EXPECTED_PERIPHERY = set(PB_PERIPHERY.VARIANTS)
EXPECTED_SCALING = set(PB_SCALING.VARIANTS)


def _validate_candidate_recall_protocol_horizon(protocol):
    """Reject artifacts whose oracle estimand differs from Paper B's H=1.

    This check is deliberately performed while loading the scaling artifact,
    before any aggregate gate can consume its rows.  Launchers already freeze
    the horizon, but artifacts are an independent trust boundary: a stale or
    modified manifest must not be adjudicated under the one-step Paper-B
    allocation claim.
    """
    if not isinstance(protocol, dict):
        raise ValueError("Paper-B candidate-recall protocol must be a JSON object")
    raw_horizon = protocol.get("horizon")
    if isinstance(raw_horizon, bool):
        raise ValueError("Paper-B candidate-recall horizon must be an integer")
    try:
        numeric_horizon = float(raw_horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Paper-B candidate-recall protocol omits an integer horizon"
        ) from exc
    if not math.isfinite(numeric_horizon) or not numeric_horizon.is_integer():
        raise ValueError("Paper-B candidate-recall horizon must be an integer")
    observed_horizon = int(numeric_horizon)
    expected_horizon = int(PAPER_B_SELECTOR_ORACLE_HORIZON)
    if observed_horizon != expected_horizon:
        raise ValueError(
            "Paper-B candidate-recall horizon contract mismatch: "
            f"expected {expected_horizon}, received {observed_horizon}"
        )
    return observed_horizon


def _validate_candidate_recall_protocol_version(protocol, *, required=False):
    """Validate the candidate-recall protocol-version trust boundary.

    Inspection/quick mode may read historical H=1 artifacts which predate the
    version field.  Confirmatory adjudication must be fail-closed: matching a
    numeric horizon alone does not establish the same candidate-oracle
    sampling, CRN, or action-mask semantics.
    """
    if not isinstance(protocol, dict):
        raise ValueError("Paper-B candidate-recall protocol must be a JSON object")
    if "protocol_version" not in protocol:
        if required:
            raise ValueError(
                "Paper-B confirmatory candidate-recall protocol omits "
                "protocol_version"
            )
        return None
    observed = protocol["protocol_version"]
    if observed != PAPER_B_CANDIDATE_RECALL_PROTOCOL_VERSION:
        raise ValueError(
            "Paper-B candidate-recall protocol version mismatch: "
            f"expected {PAPER_B_CANDIDATE_RECALL_PROTOCOL_VERSION!r}, "
            f"received {observed!r}"
        )
    return observed


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


def _require_exact_matrix(rows, expected_cells, key_fn, label):
    """Fail closed unless every logical confirmatory cell occurs exactly once."""
    keys = [key_fn(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} contains duplicate logical cells")
    expected = set(expected_cells)
    observed = set(keys)
    if observed != expected:
        missing = len(expected - observed)
        unexpected = len(observed - expected)
        raise ValueError(
            f"{label} matrix mismatch: missing={missing}, unexpected={unexpected}"
        )
    if len(rows) != len(expected):
        raise ValueError(f"{label} row-count mismatch")


def _require_unique_cells(rows, key_fn, label):
    """Reject duplicates even when the full expected matrix is data-dependent."""
    keys = [key_fn(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} contains duplicate logical cells")


def _validate_summary_binding(manifest, summary_path, rows, label, *, required):
    """Bind a confirmatory manifest to the exact CSV bytes it adjudicates."""
    if not required:
        return
    expected_count = manifest.get("summary_row_count")
    expected_hash = str(manifest.get("summary_sha256", "")).strip()
    try:
        count_matches = int(expected_count) == len(rows)
    except (TypeError, ValueError):
        count_matches = False
    if not count_matches or len(expected_hash) != 64:
        raise ValueError(f"{label} manifest omits a valid summary binding")
    with open(summary_path, "rb") as handle:
        observed_hash = hashlib.sha256(handle.read()).hexdigest()
    if observed_hash != expected_hash:
        raise ValueError(f"{label} summary SHA-256 does not match its manifest")


def _read_panel_artifact(root, directory, filename, *, protocol_mode="quick"):
    """Read one completed panel and bind its manifest to its summary bytes.

    This is deliberately schema-agnostic: callers own the definition of a
    logical experiment cell.  Fixed panels and population/budget sweeps have
    different Cartesian grids and must not share an implicit `(variant, seed)`
    assumption.
    """
    panel_root = os.path.join(root, directory)
    with open(os.path.join(panel_root, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("complete"):
        raise ValueError(f"{directory} manifest is incomplete")
    summary_path = os.path.join(panel_root, filename)
    with open(summary_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _validate_summary_binding(
        manifest, summary_path, rows, directory,
        required=(protocol_mode == "confirmatory"),
    )
    return rows, manifest


def _load_fixed_panel(
    root, directory, filename, expected_variants, expected_seeds,
    expected_panels=None, protocol_mode="quick",
):
    """Load a panel whose frozen cells are `(variant, seed)`-shaped."""
    rows, manifest = _read_panel_artifact(
        root, directory, filename, protocol_mode=protocol_mode,
    )
    if expected_panels is None:
        expected = {
            (variant, int(seed))
            for variant in expected_variants for seed in expected_seeds
        }
        _require_exact_matrix(
            rows, expected,
            lambda row: (row["variant"], int(row["seed"])),
            directory,
        )
    else:
        expected = {
            (panel, variant, int(seed))
            for panel in expected_panels
            for variant in expected_variants
            for seed in expected_seeds
        }
        _require_exact_matrix(
            rows, expected,
            lambda row: (row.get("panel", ""), row["variant"], int(row["seed"])),
            directory,
        )
    return rows, manifest


def _unique_int_axis(values, label, *, minimum=1):
    """Validate one manifest-owned integer experiment axis without coercion loss."""
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list")
    output = []
    for raw in values:
        if isinstance(raw, bool):
            raise ValueError(f"{label} must contain integers")
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must contain integers") from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{label} must contain integers")
        value = int(numeric)
        if value < int(minimum):
            raise ValueError(f"{label} values must be >= {minimum}")
        output.append(value)
    if len(output) != len(set(output)):
        raise ValueError(f"{label} contains duplicates")
    return tuple(output)


def _validate_scaling_matrix(rows, manifest, expected_seeds):
    """Enforce the manifest-owned 4-D scaling grid and clipping contract."""
    expected_seed_set = {int(seed) for seed in expected_seeds}
    if not expected_seed_set:
        raise ValueError("scaling expected seed contract is empty")
    manifest_seeds = _unique_int_axis(
        manifest.get("seeds"), "scaling seeds", minimum=0,
    )
    if set(manifest_seeds) != expected_seed_set:
        raise ValueError("scaling manifest seed contract mismatch")
    agent_counts = _unique_int_axis(
        manifest.get("agent_counts"), "scaling agent_counts", minimum=2,
    )
    budgets = _unique_int_axis(
        manifest.get("core_budgets"), "scaling core_budgets", minimum=1,
    )
    manifest_variants = manifest.get("variants")
    if (
        not isinstance(manifest_variants, list)
        or len(manifest_variants) != len(set(manifest_variants))
        or set(manifest_variants) != set(EXPECTED_SCALING)
    ):
        raise ValueError("scaling manifest variant contract mismatch")

    for n_agents in agent_counts:
        effective = tuple(min(int(budget), int(n_agents) - 1) for budget in budgets)
        if len(effective) != len(set(effective)):
            raise ValueError(
                "scaling core-budget contract collapses after clipping "
                f"at N={n_agents}"
            )

    for row in rows:
        try:
            seed = int(row["seed"])
            n_agents = int(row["n_agents"])
            requested = int(row["requested_core_budget"])
            effective = int(row["core_budget"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("scaling row has an invalid cell schema") from exc
        variant = row.get("variant")
        if seed not in expected_seed_set or n_agents not in agent_counts:
            raise ValueError("scaling row lies outside the frozen seed/population axes")
        if requested not in budgets or variant not in set(EXPECTED_SCALING):
            raise ValueError("scaling row lies outside the frozen budget/variant axes")
        expected_effective = min(requested, n_agents - 1)
        if effective != expected_effective:
            raise ValueError(
                "scaling row core_budget does not match the frozen clipping rule"
            )

    expected = {
        (int(seed), int(n_agents), min(int(budget), int(n_agents) - 1), variant)
        for seed in expected_seed_set
        for n_agents in agent_counts
        for budget in budgets
        for variant in EXPECTED_SCALING
    }
    _require_exact_matrix(
        rows,
        expected,
        lambda row: (
            int(row["seed"]), int(row["n_agents"]),
            int(row["core_budget"]), row["variant"],
        ),
        "scaling seed x population x core-budget x variant",
    )


def _load_scaling(root, expected_seeds, protocol_mode="quick"):
    rows, manifest = _read_panel_artifact(
        root, "paper_b_scaling", "summary_paper_b_scaling.csv",
        protocol_mode=protocol_mode,
    )
    _validate_scaling_matrix(rows, manifest, expected_seeds)
    degree = int(manifest.get("candidate_max_degree", 0))
    if degree <= 0:
        raise ValueError("scaling manifest omits a positive candidate_max_degree")
    if manifest.get("candidate_policy") != (
        "dynamic adapter-owned policy-independent pre-measurement candidate set"
    ):
        raise ValueError("scaling manifest must declare the dynamic pre-measurement candidate policy")
    if manifest.get("candidate_update_protocol") != "refresh_before_measurement_each_step":
        raise ValueError("scaling manifest must refresh dynamic candidates before each measurement step")
    if int(manifest.get("candidate_max_cell_occupancy", 0)) <= 0:
        raise ValueError("scaling manifest omits a frozen positive cell-occupancy gate")
    recall = manifest.get("candidate_oracle_recall")
    if not isinstance(recall, dict):
        raise ValueError("scaling manifest omits the oracle candidate-recall protocol")
    _validate_candidate_recall_protocol_horizon(recall)
    _validate_candidate_recall_protocol_version(
        recall, required=(protocol_mode == "confirmatory")
    )
    minimum = float(recall.get("minimum", float("nan")))
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise ValueError("scaling manifest has an invalid candidate-recall minimum")
    for key in ("states", "trials", "independent_replicates"):
        if int(recall.get(key, 0)) <= 0:
            raise ValueError(f"scaling candidate-recall protocol omits positive {key}")
    if str(recall.get("target_k_rule", "")).strip() != (
        "max tested explicit core budget clipped to d_max and N-1"
    ):
        raise ValueError("scaling candidate-recall protocol omits the frozen top-k* rule")
    stability_min = float(recall.get("stability_min", float("nan")))
    if not math.isfinite(stability_min) or not 0.0 <= stability_min <= 1.0:
        raise ValueError("scaling manifest omits a valid oracle ranking-stability gate")
    stable_fraction_min = float(recall.get("stable_fraction_min", float("nan")))
    if not math.isfinite(stable_fraction_min) or not 0.0 <= stable_fraction_min <= 1.0:
        raise ValueError("scaling manifest omits a valid stable-ranking fraction gate")
    degradation = manifest.get("candidate_degradation_gate")
    if not isinstance(degradation, dict) or degradation.get("reference_variant") != "Semantic-Free-Unrestricted":
        raise ValueError("scaling manifest omits the frozen dynamic-vs-unrestricted gate")
    for key in (
        "max_relative_reward_drop", "max_relative_logit_error_increase",
        "max_relative_value_error_increase", "max_action_agreement_drop",
        "reward_denominator_floor", "error_denominator_floor",
    ):
        value = float(degradation.get(key, float("nan")))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid frozen candidate degradation threshold: {key}")
    return rows, manifest


def _mean_finite(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        raise ValueError("expected at least one finite value")
    return sum(values) / float(len(values))


def _load_adaptive_budget(root, expected_seeds, protocol_mode="quick"):
    rows, manifest = _read_panel_artifact(
        root, "paper_b_adaptive_budget", "summary_paper_b_adaptive_budget.csv",
        protocol_mode=protocol_mode,
    )
    _require_unique_cells(
        rows, lambda row: (row["variant"], int(row["seed"])),
        "adaptive-budget",
    )
    seeds = {int(row["seed"]) for row in rows}
    if seeds != set(expected_seeds):
        raise ValueError("adaptive-budget seed matrix mismatch")
    matched_names = set()
    manifest_matched = str(manifest.get("matched_fixed_variant", "")).strip()
    if not manifest_matched.startswith("Fixed-K-"):
        raise ValueError("adaptive-budget manifest omits the pooled matched fixed-k variant")
    matching_rule = str(manifest.get("matching_rule", ""))
    if "disjoint pilot" not in matching_rule or "no reward" not in matching_rule:
        raise ValueError(
            "adaptive-budget comparator must be frozen on disjoint pilot cost only"
        )
    matching_seeds = [int(seed) for seed in manifest.get("matching_seeds", [])]
    if (
        not matching_seeds
        or len(set(matching_seeds)) != len(matching_seeds)
        or set(matching_seeds) & set(expected_seeds)
        or not bool(manifest.get("matching_seed_disjoint", False))
    ):
        raise ValueError(
            "adaptive-budget matching seeds must be non-empty, unique, and disjoint "
            "from confirmatory seeds"
        )
    pilot_cost = float(manifest.get("pilot_adaptive_core_cost_per_ego", float("nan")))
    if not math.isfinite(pilot_cost):
        raise ValueError("adaptive-budget manifest omits finite pilot matching cost")
    for seed in expected_seeds:
        subset = [row for row in rows if int(row["seed"]) == int(seed)]
        names = {row["variant"] for row in subset}
        if "Adaptive-K" not in names or "Full-Explicit" not in names:
            raise ValueError("adaptive-budget panel omits required variants")
        fixed = [row for row in subset if row["variant"].startswith("Fixed-K-")]
        matched = [row for row in fixed if int(row.get("matched_to_adaptive", 0)) == 1]
        if not fixed or len(matched) != 1:
            raise ValueError("adaptive-budget panel must identify one matched fixed-k row")
        matched_names.add(matched[0]["variant"])
        if matched[0]["variant"] != manifest_matched:
            raise ValueError("adaptive-budget matched fixed k changed across seeds")
        if not math.isfinite(float(matched[0].get("mean_core_cost_gap_to_adaptive", float("nan")))):
            raise ValueError("adaptive-budget row omits realised cost gap")
        if not math.isfinite(float(matched[0].get("mean_K_t", float("nan")))):
            raise ValueError("adaptive-budget matching must report K_t cost")
    if matched_names != {manifest_matched}:
        raise ValueError("adaptive-budget panel does not use one pooled fixed-k comparator")
    adaptive_cost = _mean_finite([
        float(row["mean_core_cost_per_ego"])
        for row in rows if row["variant"] == "Adaptive-K"
    ])
    matched_cost = _mean_finite([
        float(row["mean_core_cost_per_ego"])
        for row in rows if row["variant"] == manifest_matched
    ])
    realised_gap = abs(float(adaptive_cost) - float(matched_cost))
    if realised_gap > 0.5 + 1e-9:
        raise ValueError(
            "pilot-frozen adaptive comparator is not matched in pooled confirmatory cost"
        )
    manifest["realised_confirmatory_cost_gap"] = float(realised_gap)
    saturation_max = float(manifest.get("max_boundary_saturation_fraction", float("nan")))
    if not math.isfinite(saturation_max) or not 0.0 <= saturation_max < 1.0:
        raise ValueError("adaptive-budget manifest omits frozen saturation threshold")
    for row in rows:
        if row["variant"] == "Adaptive-K":
            rate = float(row.get("boundary_saturation_rate", float("nan")))
            if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
                raise ValueError("Adaptive-K row omits boundary_saturation_rate")
    return rows, manifest


def _paired(rows, treatment, control, metric, panel=None):
    selected = [
        row for row in rows
        if panel is None or row.get("panel") == panel
    ]
    _require_unique_cells(
        selected,
        lambda row: (row.get("panel", ""), row["variant"], int(row["seed"])),
        "paired comparison",
    )
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
    allocation, allocation_manifest = _load_fixed_panel(
        run_root, "paper_b_allocation", "summary_paper_b_allocation.csv",
        EXPECTED_ALLOCATION, expected_seeds,
        expected_panels={"selector_isolation", "end_to_end"},
        protocol_mode=protocol_mode,
    )
    pair_rows, _ = _load_fixed_panel(
        run_root, "paper_b_pair_latent", "summary_paper_b_pair_latent.csv",
        EXPECTED_PAIR, expected_seeds, protocol_mode=protocol_mode,
    )
    periphery_rows, _ = _load_fixed_panel(
        run_root, "paper_b_periphery", "summary_paper_b_periphery.csv",
        EXPECTED_PERIPHERY, expected_seeds, protocol_mode=protocol_mode,
    )
    scaling_rows, scaling_manifest = _load_scaling(
        run_root, expected_seeds, protocol_mode=protocol_mode
    )
    adaptive_rows, adaptive_manifest = _load_adaptive_budget(
        run_root, expected_seeds, protocol_mode=protocol_mode
    )
    if protocol_mode == "quick":
        return {
            "paper": "B", "overall_status": "SMOKE_ONLY",
            "panels_complete": True,
        }, VC.EXIT_SMOKE_ONLY
    pair_isolation_protocol = "common_checkpoint_frozen_downstream_teacher_forced_history"
    periphery_isolation_protocol = (
        "common_checkpoint_full_explicit_distillation_on_immutable_pre_action_history"
    )
    for row in pair_rows:
        if row.get("decision_fidelity_reference") == "not_collected":
            raise ValueError("Paper-B pair fidelity reference is missing")
        if row.get("decision_fidelity_protocol") != pair_isolation_protocol:
            raise ValueError(
                "Paper-B pair fidelity must use a shared checkpoint, frozen "
                "downstream policy/value, and teacher-forced pair history"
            )
    for row in periphery_rows:
        if row.get("decision_fidelity_reference") == "not_collected":
            raise ValueError("Paper-B periphery fidelity reference is missing")
        if row.get("decision_fidelity_protocol") != periphery_isolation_protocol:
            raise ValueError(
                "Paper-B periphery fidelity must use frozen Full-Explicit "
                "distillation targets on one immutable pre-action history"
            )
    # Pair representation and allocation are isolation experiments too. Bind
    # every arm to the same immutable teacher/state/target/downstream content,
    # rather than trusting a shared seed or checkpoint label alone.
    pair_provenance = {}
    for row in pair_rows:
        if int(float(row.get("decision_fidelity_history_steps", 0))) <= 0:
            raise ValueError("pair fidelity history is empty")
        for key in (
            "teacher_trace_sha256",
            "teacher_action_history_sha256",
            "decision_fidelity_state_bank_sha256",
            "full_explicit_target_sha256",
            "decision_fidelity_downstream_checkpoint_sha256",
        ):
            if len(str(row.get(key, "")).strip()) != 64:
                raise ValueError(f"pair row omits valid {key}")
        seed = int(float(row["seed"]))
        fingerprint = tuple(str(row[key]) for key in (
            "teacher_trace_sha256",
            "teacher_action_history_sha256",
            "decision_fidelity_state_bank_sha256",
            "full_explicit_target_sha256",
            "decision_fidelity_downstream_checkpoint_sha256",
        ))
        previous = pair_provenance.setdefault(seed, fingerprint)
        if previous != fingerprint:
            raise ValueError(
                "pair variants within a seed did not share identical teacher/state provenance"
            )
    allocation_provenance = {}
    for row in allocation:
        if row.get("panel") != "selector_isolation":
            continue
        for key in (
            "selector_state_bank_sha256",
            "selector_teacher_context_sha256",
            "selector_oracle_target_sha256",
            "full_explicit_decision_target_sha256",
            "decision_fidelity_downstream_checkpoint_sha256",
        ):
            if len(str(row.get(key, "")).strip()) != 64:
                raise ValueError(f"allocation selector row omits valid {key}")
        seed = int(float(row["seed"]))
        fingerprint = tuple(str(row[key]) for key in (
            "selector_state_bank_sha256",
            "selector_teacher_context_sha256",
            "selector_oracle_target_sha256",
            "full_explicit_decision_target_sha256",
            "decision_fidelity_downstream_checkpoint_sha256",
        ))
        previous = allocation_provenance.setdefault(seed, fingerprint)
        if previous != fingerprint:
            raise ValueError(
                "allocation variants within a seed did not share identical "
                "state/context/oracle provenance"
            )
    # Periphery isolation must be backed by one immutable, non-empty
    # peripheral teacher history and one common fidelity state bank.  A protocol
    # label alone is not evidence that variants consumed identical data.
    periphery_provenance = {}
    for row in periphery_rows:
        if int(float(row.get("decision_fidelity_history_steps", 0))) <= 0:
            raise ValueError("periphery fidelity history is empty")
        if int(float(row.get("teacher_peripheral_item_count", 0))) <= 0:
            raise ValueError("periphery isolation teacher contains no peripheral items")
        for key in (
            "teacher_trace_sha256",
            "teacher_action_history_sha256",
            "full_explicit_target_sha256",
            "decision_fidelity_downstream_checkpoint_sha256",
        ):
            value = str(row.get(key, "")).strip()
            if len(value) != 64:
                raise ValueError(f"periphery row omits valid {key}")
        seed = int(float(row["seed"]))
        fingerprint = tuple(str(row[key]) for key in (
            "teacher_trace_sha256",
            "teacher_action_history_sha256",
            "full_explicit_target_sha256",
            "decision_fidelity_downstream_checkpoint_sha256",
        ))
        previous = periphery_provenance.setdefault(seed, fingerprint)
        if previous != fingerprint:
            raise ValueError(
                "periphery variants within a seed did not share identical teacher/state provenance"
            )

    oracle_replicates = int(allocation_manifest.get("oracle_replicates", 0))
    oracle_stability_min = float(
        allocation_manifest.get("oracle_stability_min", float("nan"))
    )
    if oracle_replicates < 2:
        raise ValueError("allocation manifest omits independent oracle replicas")
    if not math.isfinite(oracle_stability_min) or not 0.0 <= oracle_stability_min <= 1.0:
        raise ValueError("allocation manifest has an invalid oracle stability threshold")
    for row in allocation:
        if row.get("panel") != "selector_isolation":
            continue
        for key in ("oracle_min_c_topk_jaccard", "oracle_min_absd_topk_jaccard"):
            value = float(row.get(key, float("nan")))
            if not math.isfinite(value) or value + 1e-12 < oracle_stability_min:
                raise ValueError(
                    f"allocation selector bank failed frozen oracle stability gate: {key}={value}"
                )

    for row in allocation:
        if row.get("panel") == "selector_isolation" and row.get(
            "decision_fidelity_protocol"
        ) != "common_frozen_policy_context_selector_then_forward":
            raise ValueError("allocation fidelity must be computed after selector commit")
        if row.get("panel") == "selector_isolation" and row.get("checkpoint_role") != (
            "selector_neutral_full_explicit_pretrain"
        ):
            raise ValueError("selector isolation must use neutral full-explicit pretraining")
        if row.get("panel") == "end_to_end" and row.get("checkpoint_role") != (
            "common_untrained_initialization"
        ):
            raise ValueError("end-to-end selector runs must start from common untrained weights")

    c_vs_d = _paired(
        allocation, "C-Core", "AbsD-Core", "selector_oracle_f1",
        panel="selector_isolation",
    )
    c_vs_d_disagreement = _paired(
        allocation, "C-Core", "AbsD-Core",
        "disagreement_selector_oracle_f1", panel="selector_isolation",
    )
    c_vs_d_ndcg = _paired(
        allocation, "C-Core", "AbsD-Core", "selector_oracle_ndcg_at_k",
        panel="selector_isolation",
    )
    c_vs_attention = _paired(
        allocation, "C-Core", "Attention-Core", "selector_oracle_f1",
        panel="selector_isolation",
    )
    c_vs_random = _paired(
        allocation, "C-Core", "Random-Core", "selector_oracle_f1",
        panel="selector_isolation",
    )
    c_vs_correlation = _paired(
        allocation, "C-Core", "Correlation-Core", "selector_oracle_f1",
        panel="selector_isolation",
    )
    c_vs_weak_prior = _paired(
        allocation, "C-Core", "WeakPrior-Core", "selector_oracle_f1",
        panel="selector_isolation",
    )
    c_disagreement_selector_by_comparator = {}
    for comparator in (
        "AbsD-Core", "Attention-Core", "Random-Core", "Correlation-Core",
        "WeakPrior-Core",
    ):
        c_disagreement_selector_by_comparator[comparator] = _paired(
            allocation, "C-Core", comparator,
            "disagreement_selector_oracle_f1", panel="selector_isolation",
        )
    oracle_vs_random = _paired(
        allocation, "Oracle-C-Core", "Random-Core", "selector_oracle_f1",
        panel="selector_isolation",
    )
    c_reward_vs_d = _paired(
        allocation, "C-Core", "AbsD-Core", "mean_reward", panel="end_to_end"
    )
    c_logit_vs_d = [-value for value in _paired(
        allocation, "C-Core", "AbsD-Core",
        "policy_logit_l2_to_full_explicit", panel="selector_isolation",
    )]
    c_value_vs_d = [-value for value in _paired(
        allocation, "C-Core", "AbsD-Core",
        "value_mae_to_full_explicit", panel="selector_isolation",
    )]
    c_action_vs_d = _paired(
        allocation, "C-Core", "AbsD-Core",
        "action_agreement_to_full_explicit", panel="selector_isolation",
    )
    allocation_fidelity_comparators = (
        "AbsD-Core", "Attention-Core", "Random-Core", "Correlation-Core",
        "WeakPrior-Core",
    )
    c_fidelity_by_comparator = {}
    c_disagreement_fidelity_by_comparator = {}
    for comparator in allocation_fidelity_comparators:
        c_fidelity_by_comparator[comparator] = {
            "logit": [-value for value in _paired(
                allocation, "C-Core", comparator,
                "policy_logit_l2_to_full_explicit", panel="selector_isolation",
            )],
            "value": [-value for value in _paired(
                allocation, "C-Core", comparator,
                "value_mae_to_full_explicit", panel="selector_isolation",
            )],
            "action": _paired(
                allocation, "C-Core", comparator,
                "action_agreement_to_full_explicit", panel="selector_isolation",
            ),
        }
        c_disagreement_fidelity_by_comparator[comparator] = {
            "logit": [-value for value in _paired(
                allocation, "C-Core", comparator,
                "disagreement_policy_logit_l2_to_full_explicit",
                panel="selector_isolation",
            )],
            "value": [-value for value in _paired(
                allocation, "C-Core", comparator,
                "disagreement_value_mae_to_full_explicit",
                panel="selector_isolation",
            )],
            "action": _paired(
                allocation, "C-Core", comparator,
                "disagreement_action_agreement_to_full_explicit",
                panel="selector_isolation",
            ),
        }
    pair_retrieval = [
        -value for value in _paired(
            pair_rows, "Recurrent-BC-CD-Contrastive", "Recurrent-BC",
            "cd_retrieval_mae",
        )
    ]
    pair_geometry = _paired(
        pair_rows, "Recurrent-BC-CD-Contrastive", "Recurrent-BC",
        "latent_profile_distance_spearman",
    )
    pair_logit = [-value for value in _paired(
        pair_rows, "Recurrent-BC-CD-Contrastive", "Recurrent-BC",
        "policy_logit_l2_to_full_explicit",
    )]
    pair_value = [-value for value in _paired(
        pair_rows, "Recurrent-BC-CD-Contrastive", "Recurrent-BC",
        "value_mae_to_full_explicit",
    )]
    pair_action = _paired(
        pair_rows, "Recurrent-BC-CD-Contrastive", "Recurrent-BC",
        "action_agreement_to_full_explicit",
    )
    pair_structural_baselines = (
        "Shared-Neighbor-State", "Pooled-Neighbor", "Explicit-FF-BC",
    )
    pair_fidelity_by_baseline = {}
    for baseline in pair_structural_baselines:
        pair_fidelity_by_baseline[baseline] = {
            "logit": [-value for value in _paired(
                pair_rows, "Recurrent-BC-CD-Contrastive", baseline,
                "policy_logit_l2_to_full_explicit",
            )],
            "value": [-value for value in _paired(
                pair_rows, "Recurrent-BC-CD-Contrastive", baseline,
                "value_mae_to_full_explicit",
            )],
            "action": _paired(
                pair_rows, "Recurrent-BC-CD-Contrastive", baseline,
                "action_agreement_to_full_explicit",
            ),
        }
    warm_start_transient = [-value for value in _paired(
        pair_rows, "Recurrent-BC-CD", "Recurrent-BC-CD-NoWarmStart",
        "post_promotion_bc_loss",
    )]
    warm_start_logit = [-value for value in _paired(
        pair_rows, "Recurrent-BC-CD", "Recurrent-BC-CD-NoWarmStart",
        "promotion_logit_error_auc",
    )]
    warm_start_policy_kl = [-value for value in _paired(
        pair_rows, "Recurrent-BC-CD", "Recurrent-BC-CD-NoWarmStart",
        "promotion_policy_kl_auc",
    )]
    warm_start_value = [-value for value in _paired(
        pair_rows, "Recurrent-BC-CD", "Recurrent-BC-CD-NoWarmStart",
        "promotion_value_error_auc",
    )]
    warm_start_action = _paired(
        pair_rows, "Recurrent-BC-CD", "Recurrent-BC-CD-NoWarmStart",
        "promotion_action_agreement_auc",
    )
    warm_start_events = [
        min(
            int(left["promotion_event_count"]), int(right["promotion_event_count"])
        )
        for left, right in zip(
            sorted(
                (row for row in pair_rows if row["variant"] == "Recurrent-BC-CD"),
                key=lambda row: int(row["seed"]),
            ),
            sorted(
                (row for row in pair_rows if row["variant"] == "Recurrent-BC-CD-NoWarmStart"),
                key=lambda row: int(row["seed"]),
            ),
        )
    ]
    periphery_reward = _paired(
        periphery_rows, "Semantic-Free", "Single-Mean-Matched", "mean_reward"
    )
    periphery_logit = [-value for value in _paired(
        periphery_rows, "Semantic-Free", "Single-Mean-Matched",
        "policy_logit_l2_to_full_explicit",
    )]
    periphery_value = [-value for value in _paired(
        periphery_rows, "Semantic-Free", "Single-Mean-Matched",
        "value_mae_to_full_explicit",
    )]
    periphery_action = _paired(
        periphery_rows, "Semantic-Free", "Single-Mean-Matched",
        "action_agreement_to_full_explicit",
    )
    periphery_parameter_budget_delta = _paired(
        periphery_rows, "Semantic-Free", "Single-Mean-Matched",
        "trainable_parameter_bytes",
    )
    semantic_vs_unconstrained = _paired(
        periphery_rows, "Semantic-Free", "Unconstrained", "mean_reward"
    )
    semantic_vs_unconstrained_logit = [-value for value in _paired(
        periphery_rows, "Semantic-Free", "Unconstrained",
        "policy_logit_l2_to_full_explicit",
    )]
    semantic_vs_unconstrained_value = [-value for value in _paired(
        periphery_rows, "Semantic-Free", "Unconstrained",
        "value_mae_to_full_explicit",
    )]
    semantic_vs_unconstrained_action = _paired(
        periphery_rows, "Semantic-Free", "Unconstrained",
        "action_agreement_to_full_explicit",
    )
    capacity_vs_absd_pooling = _paired(
        periphery_rows, "Semantic-Free", "AbsD-Pooling", "mean_reward"
    )
    semantic_vs_attention_mean = _paired(
        periphery_rows, "Semantic-Free", "Attention-Mean", "mean_reward"
    )
    agent_counts = sorted({int(row["n_agents"]) for row in scaling_rows})
    core_budgets = sorted({int(row["core_budget"]) for row in scaling_rows})
    scaling_pareto = []
    semantic_frontier_flags = []
    candidate_reward_degradation = []
    candidate_logit_degradation = []
    candidate_value_degradation = []
    candidate_action_degradation = []
    candidate_degradation_pass_flags = []
    degradation_gate = scaling_manifest["candidate_degradation_gate"]
    reward_tol = float(degradation_gate["max_relative_reward_drop"])
    logit_tol = float(degradation_gate["max_relative_logit_error_increase"])
    value_tol = float(degradation_gate["max_relative_value_error_increase"])
    action_tol = float(degradation_gate["max_action_agreement_drop"])
    reward_floor = float(degradation_gate["reward_denominator_floor"])
    error_floor = float(degradation_gate["error_denominator_floor"])

    # H3b is genuinely across core budgets. For each matched seed/population
    # cell, determine whether at least one Semantic-Free budget contributes a
    # non-dominated reward/throughput/memory operating point.
    for n_agents in agent_counts:
        for seed in expected_seeds:
            cell = [
                row for row in scaling_rows
                if int(row["n_agents"]) == int(n_agents)
                and int(row["seed"]) == int(seed)
            ]
            points = []
            for row in cell:
                points.append({
                    "variant": row["variant"],
                    "core_budget": int(row["core_budget"]),
                    "reward": float(row["mean_reward"]),
                    "throughput": float(row["throughput_total"]),
                    "memory": float(row["representation_memory_bytes"]),
                    "row": row,
                })
            semantic_points = [p for p in points if p["variant"] == "Semantic-Free"]
            if len(semantic_points) != len(core_budgets):
                raise ValueError(
                    f"Semantic-Free core-budget sweep incomplete at N={n_agents}, seed={seed}"
                )
            any_nondominated = False
            for subject in semantic_points:
                dominated = any(
                    other is not subject
                    and other["reward"] >= subject["reward"]
                    and other["throughput"] >= subject["throughput"]
                    and other["memory"] <= subject["memory"]
                    and (
                        other["reward"] > subject["reward"]
                        or other["throughput"] > subject["throughput"]
                        or other["memory"] < subject["memory"]
                    )
                    for other in points
                )
                any_nondominated = any_nondominated or (not dominated)
                budget = subject["core_budget"]
                same_budget = {
                    p["variant"]: p for p in points if p["core_budget"] == budget
                }
                required = {
                    "Single-Mean", "Full-Explicit", "PureMeanField",
                    "Semantic-Free-Unrestricted",
                }
                if not required.issubset(same_budget):
                    raise ValueError("scaling budget cell omits required comparators")
                scaling_pareto.append({
                    "n_agents": n_agents,
                    "seed": seed,
                    "core_budget": budget,
                    "reward_vs_single": subject["reward"] - same_budget["Single-Mean"]["reward"],
                    "throughput_vs_full": subject["throughput"] - same_budget["Full-Explicit"]["throughput"],
                    "memory_vs_full": same_budget["Full-Explicit"]["memory"] - subject["memory"],
                    "reward_vs_pure_mean_field": subject["reward"] - same_budget["PureMeanField"]["reward"],
                    "nondominated": 0.0 if dominated else 1.0,
                })

                # H4 compares dynamic candidate restriction against the exact
                # same Semantic-Free architecture with unrestricted pair scan.
                dyn = subject["row"]
                unres = same_budget["Semantic-Free-Unrestricted"]["row"]
                reward_drop = max(
                    0.0, float(unres["mean_reward"]) - float(dyn["mean_reward"])
                ) / max(reward_floor, abs(float(unres["mean_reward"])))
                logit_inc = max(
                    0.0,
                    float(dyn["policy_logit_l2_to_full_explicit"])
                    - float(unres["policy_logit_l2_to_full_explicit"]),
                ) / max(error_floor, abs(float(unres["policy_logit_l2_to_full_explicit"])))
                value_inc = max(
                    0.0,
                    float(dyn["value_mae_to_full_explicit"])
                    - float(unres["value_mae_to_full_explicit"]),
                ) / max(error_floor, abs(float(unres["value_mae_to_full_explicit"])))
                action_drop = max(
                    0.0,
                    float(unres["action_agreement_to_full_explicit"])
                    - float(dyn["action_agreement_to_full_explicit"]),
                )
                candidate_reward_degradation.append(reward_drop)
                candidate_logit_degradation.append(logit_inc)
                candidate_value_degradation.append(value_inc)
                candidate_action_degradation.append(action_drop)
                candidate_degradation_pass_flags.append(float(
                    reward_drop <= reward_tol + 1e-12
                    and logit_inc <= logit_tol + 1e-12
                    and value_inc <= value_tol + 1e-12
                    and action_drop <= action_tol + 1e-12
                ))
            semantic_frontier_flags.append(1.0 if any_nondominated else 0.0)

    pareto_reward = [row["reward_vs_single"] for row in scaling_pareto]
    pareto_throughput = [row["throughput_vs_full"] for row in scaling_pareto]
    pareto_memory = [row["memory_vs_full"] for row in scaling_pareto]
    pareto_pure_mean = [row["reward_vs_pure_mean_field"] for row in scaling_pareto]
    adaptive_reward_vs_matched, adaptive_logit_vs_matched = [], []
    adaptive_value_vs_matched, adaptive_action_vs_matched = [], []
    for seed in expected_seeds:
        subset = [row for row in adaptive_rows if int(row["seed"]) == int(seed)]
        adaptive = next(row for row in subset if row["variant"] == "Adaptive-K")
        matched = next(
            row for row in subset
            if row["variant"].startswith("Fixed-K-")
            and int(row.get("matched_to_adaptive", 0)) == 1
        )
        adaptive_reward_vs_matched.append(
            float(adaptive["mean_reward"]) - float(matched["mean_reward"])
        )
        adaptive_logit_vs_matched.append(
            float(matched["policy_logit_l2_to_full_explicit"])
            - float(adaptive["policy_logit_l2_to_full_explicit"])
        )
        adaptive_value_vs_matched.append(
            float(matched["value_mae_to_full_explicit"])
            - float(adaptive["value_mae_to_full_explicit"])
        )
        adaptive_action_vs_matched.append(
            float(adaptive["action_agreement_to_full_explicit"])
            - float(matched["action_agreement_to_full_explicit"])
        )
    # Scaling rows must include the paper-promised fidelity and latency metrics.
    bounded_edge_accounting_valid = True
    candidate_oracle_recall_valid = True
    candidate_construction_subquadratic_valid = True
    candidate_construction_linear_valid = True
    feature_snapshot_linear_valid = True
    candidate_provider_contract_valid = True
    candidate_occupancy_gate_valid = True
    candidate_provider_work_bound_valid = True
    candidate_state_eviction_valid = True
    candidate_oracle_stability_valid = True
    candidate_recall_protocol = scaling_manifest["candidate_oracle_recall"]
    candidate_recall_minimum = float(candidate_recall_protocol["minimum"])
    _validate_candidate_recall_protocol_horizon(candidate_recall_protocol)
    _validate_candidate_recall_protocol_version(candidate_recall_protocol)
    if int(candidate_recall_protocol.get("trials", 0)) < 2:
        raise ValueError("Paper-B candidate recall requires repeated CRN trials")
    if int(candidate_recall_protocol.get("independent_replicates", 0)) < 2:
        raise ValueError("Paper-B candidate recall requires independent oracle ranking replicas")
    candidate_stability_minimum = float(candidate_recall_protocol.get("stability_min", float("nan")))
    candidate_stable_fraction_minimum = float(
        candidate_recall_protocol.get("stable_fraction_min", float("nan"))
    )
    if not math.isfinite(candidate_stability_minimum):
        raise ValueError("Paper-B candidate recall omits ranking stability threshold")
    for row in scaling_rows:
        for key in (
            "policy_logit_l2_to_full_explicit", "value_mae_to_full_explicit",
            "action_agreement_to_full_explicit", "inference_latency_p50_ms",
            "inference_latency_p95_ms",
        ):
            if not math.isfinite(float(row.get(key, float("nan")))):
                raise ValueError(f"scaling row omits finite {key}")
        if row.get("inference_latency_protocol") != (
            "deterministic_policy_inference_without_training_cache_sampling_or_epsilon_forcing"
        ):
            raise ValueError(
                "scaling latency must measure deterministic policy inference without forcing"
            )
        if row.get("representation_memory_accounting_protocol") != (
            "analytic_trainable_parameters_plus_persistent_representation_state"
        ):
            raise ValueError("scaling row mislabels analytic representation memory")
        if int(float(row.get("representation_memory_is_runtime_peak", 1))) != 0:
            raise ValueError("analytic representation memory must not be labelled runtime peak")
        if row.get("runtime_memory_protocol") != (
            "separate_deterministic_probe_sampled_process_rss_and_torch_cuda_peaks"
        ):
            raise ValueError("scaling row omits the separate runtime-memory protocol")
        for key in (
            "runtime_peak_process_rss_bytes",
            "runtime_peak_process_rss_delta_bytes",
        ):
            value = float(row.get(key, float("nan")))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"scaling row omits valid {key}")
        variant = row["variant"]
        n_agents = int(row["n_agents"])
        if variant in {"Full-Explicit", "Semantic-Free-Unrestricted"}:
            # Dense references are never evidence for candidate-restricted
            # edge scaling. Semantic-Free-Unrestricted is the H4 pruning control.
            continue
        if variant == "PureMeanField":
            # Mean field has no directed pair-state allocation.
            continue
        d_max = int(row.get("candidate_max_degree_protocol", 0))
        edges = int(row.get("measured_edge_count", -1))
        e_t = int(row.get("E_t", -1))
        k_t = int(row.get("K_t", -1))
        if e_t != edges:
            raise ValueError("scaling E_t disagrees with measured_edge_count")
        if k_t < 0 or k_t > e_t:
            raise ValueError("scaling K_t must satisfy 0 <= K_t <= E_t")
        if d_max <= 0 or edges < 0 or edges > n_agents * d_max:
            bounded_edge_accounting_valid = False
        e_max = int(float(row.get("E_t_max", -1)))
        k_max_seen = int(float(row.get("K_t_max", -1)))
        shadow_final = int(float(row.get("shadow_pair_count_final", -1)))
        belief_final = int(float(row.get("belief_pair_count_final", -1)))
        full_final = int(float(row.get("full_state_pair_count_final", -1)))
        sig_final = int(float(row.get("signature_pair_count_final", -1)))
        shadow_max = int(float(row.get("shadow_pair_count_max", -1)))
        belief_max = int(float(row.get("belief_pair_count_max", -1)))
        full_max = int(float(row.get("full_state_pair_count_max", -1)))
        sig_max = int(float(row.get("signature_pair_count_max", -1)))
        if not (
            e_max >= e_t >= 0 and k_max_seen >= k_t >= 0
            and shadow_final == e_t and belief_final == e_t
            and 0 <= full_final <= k_t and 0 <= sig_final <= e_t
            and 0 <= shadow_max <= e_max and 0 <= belief_max <= e_max
            and 0 <= full_max <= k_max_seen and 0 <= sig_max <= e_max
        ):
            candidate_state_eviction_valid = False
        recall = float(row.get("candidate_oracle_recall_at_degree", float("nan")))
        comparisons = int(row.get("candidate_oracle_recall_comparisons", 0))
        evaluated_stable = int(row.get("candidate_oracle_recall_evaluated_ego_states", 0))
        jaccard_mean = float(row.get("candidate_oracle_ranking_jaccard_mean", float("nan")))
        stable_fraction = float(row.get("candidate_oracle_stable_fraction", float("nan")))
        unresolved = int(row.get("candidate_oracle_unresolved_ego_states", -1))
        target_k = int(row.get("candidate_oracle_target_k", 0))
        expected_target_k = min(
            d_max, max(min(int(k), n_agents - 1) for k in core_budgets)
        )
        if target_k != expected_target_k or target_k <= 0 or target_k > d_max:
            candidate_oracle_recall_valid = False
        if (
            int(row.get("candidate_recall_applicable", 0)) != 1
            or not math.isfinite(recall)
            or comparisons <= 0
            or evaluated_stable <= 0
            or recall < candidate_recall_minimum
        ):
            candidate_oracle_recall_valid = False
        if (
            not math.isfinite(jaccard_mean)
            or not math.isfinite(stable_fraction)
            or stable_fraction + 1e-12 < candidate_stable_fraction_minimum
            or unresolved < 0
            or int(row.get("candidate_oracle_replicates", 0)) < 2
        ):
            candidate_oracle_stability_valid = False
        construction_flag = str(
            row.get("candidate_construction_subquadratic", "false")
        ).strip().lower()
        if construction_flag not in {"1", "true", "yes"}:
            candidate_construction_subquadratic_valid = False
        linear_flag = str(
            row.get("candidate_construction_linear_candidate", "false")
        ).strip().lower()
        if linear_flag not in {"1", "true", "yes"}:
            candidate_construction_linear_valid = False
        feature_linear_flag = str(
            row.get("feature_snapshot_linear_candidate", "false")
        ).strip().lower()
        if feature_linear_flag not in {"1", "true", "yes"}:
            feature_snapshot_linear_valid = False
        provider = str(row.get("candidate_provider", "")).strip()
        if provider != "omni_linked_cell_dynamic":
            candidate_provider_contract_valid = False
        occupancy = int(float(row.get("candidate_cell_occupancy_max", -1)))
        occupancy_cap = int(scaling_manifest.get("candidate_max_cell_occupancy", 0))
        if occupancy < 0 or occupancy_cap <= 0 or occupancy > occupancy_cap:
            candidate_occupancy_gate_valid = False
        provider_work = float(row.get("candidate_provider_work_units", float("nan")))
        stencil = int(scaling_manifest.get("candidate_stencil_radius", -1))
        if stencil < 0 or not math.isfinite(provider_work) or provider_work < 0.0:
            candidate_provider_work_bound_valid = False
        else:
            # Linked-cell refresh: one population insertion per agent plus at
            # most one candidate check per occupant in each cell of the fixed
            # stencil.  A frozen occupancy cap therefore yields a linear
            # certificate for the measured provider path.
            stencil_cells = (2 * stencil + 1) ** 2
            structural_linear_bound = n_agents * (1 + stencil_cells * occupancy_cap)
            if provider_work > structural_linear_bound + 1e-9:
                candidate_provider_work_bound_valid = False
        if not str(row.get("candidate_last_hash", "")).strip():
            candidate_provider_contract_valid = False
        if len(str(row.get("candidate_history_sha256", "")).strip()) != 64:
            candidate_provider_contract_valid = False
        for key in (
            "candidate_refresh_ms", "candidate_churn", "candidate_added_pairs",
            "candidate_removed_pairs", "candidate_cell_occupancy_mean",
        ):
            if not math.isfinite(float(row.get(key, float("nan")))):
                candidate_provider_contract_valid = False

    metrics = {
        "c_core_minus_absd_selector_f1": c_vs_d,
        "c_core_minus_absd_disagreement_f1": c_vs_d_disagreement,
        "c_core_minus_absd_selector_ndcg": c_vs_d_ndcg,
        "c_core_minus_attention_selector_f1": c_vs_attention,
        "c_core_minus_random_selector_f1": c_vs_random,
        "c_core_minus_correlation_selector_f1": c_vs_correlation,
        "c_core_minus_weak_prior_selector_f1": c_vs_weak_prior,
        **{
            "c_core_minus_"
            f"{comparator.lower().replace('-core', '').replace('-', '_')}"
            "_disagreement_selector_f1": values
            for comparator, values in c_disagreement_selector_by_comparator.items()
        },
        "oracle_minus_random_selector_f1": oracle_vs_random,
        "c_core_minus_absd_reward": c_reward_vs_d,
        "c_core_minus_absd_logit_fidelity_error": c_logit_vs_d,
        "c_core_minus_absd_value_fidelity_error": c_value_vs_d,
        "c_core_minus_absd_action_agreement": c_action_vs_d,
        "bc_minus_full_cd_retrieval_mae": pair_retrieval,
        "full_cd_minus_bc_latent_profile_geometry": pair_geometry,
        "full_cd_minus_bc_logit_fidelity_error": pair_logit,
        "full_cd_minus_bc_value_fidelity_error": pair_value,
        "full_cd_minus_bc_action_agreement": pair_action,
        "warm_start_minus_no_warm_post_promotion_bc_loss": warm_start_transient,
        "warm_start_minus_no_warm_promotion_logit_error": warm_start_logit,
        "warm_start_minus_no_warm_promotion_policy_kl": warm_start_policy_kl,
        "warm_start_minus_no_warm_promotion_value_error": warm_start_value,
        "warm_start_minus_no_warm_promotion_action_agreement": warm_start_action,
        "semantic_free_minus_single_mean_reward": periphery_reward,
        "semantic_free_minus_single_mean_logit_fidelity_error": periphery_logit,
        "semantic_free_minus_single_mean_value_fidelity_error": periphery_value,
        "semantic_free_minus_single_mean_action_agreement": periphery_action,
        "semantic_free_minus_matched_single_mean_parameter_bytes": periphery_parameter_budget_delta,
        "semantic_free_minus_unconstrained_reward": semantic_vs_unconstrained,
        "capacity_pooling_minus_absD_pooling_reward": capacity_vs_absd_pooling,
        "semantic_free_minus_attention_mean_reward": semantic_vs_attention_mean,
        "scaling_semantic_minus_single_reward": pareto_reward,
        "scaling_semantic_minus_full_throughput": pareto_throughput,
        "scaling_full_minus_semantic_memory_bytes": pareto_memory,
        "scaling_semantic_minus_pure_mean_field_reward": pareto_pure_mean,
        "scaling_semantic_pareto_nondominated": semantic_frontier_flags,
        "adaptive_minus_matched_fixed_reward": adaptive_reward_vs_matched,
        "adaptive_minus_matched_fixed_logit_fidelity": adaptive_logit_vs_matched,
        "adaptive_minus_matched_fixed_value_fidelity": adaptive_value_vs_matched,
        "adaptive_minus_matched_fixed_action_agreement": adaptive_action_vs_matched,
    }
    for comparator, values in c_fidelity_by_comparator.items():
        tag = comparator.lower().replace("-core", "").replace("-", "_")
        metrics[f"c_core_minus_{tag}_logit_fidelity_error"] = values["logit"]
        metrics[f"c_core_minus_{tag}_value_fidelity_error"] = values["value"]
        metrics[f"c_core_minus_{tag}_action_agreement"] = values["action"]
    for comparator, values in c_disagreement_fidelity_by_comparator.items():
        tag = comparator.lower().replace("-core", "").replace("-", "_")
        metrics[f"c_core_minus_{tag}_disagreement_logit_fidelity_error"] = values["logit"]
        metrics[f"c_core_minus_{tag}_disagreement_value_fidelity_error"] = values["value"]
        metrics[f"c_core_minus_{tag}_disagreement_action_agreement"] = values["action"]
    for baseline, values in pair_fidelity_by_baseline.items():
        tag = baseline.lower().replace("-", "_")
        metrics[f"full_pair_minus_{tag}_logit_fidelity_error"] = values["logit"]
        metrics[f"full_pair_minus_{tag}_value_fidelity_error"] = values["value"]
        metrics[f"full_pair_minus_{tag}_action_agreement"] = values["action"]
    metrics.update({
        "semantic_free_minus_unconstrained_logit_fidelity_error": semantic_vs_unconstrained_logit,
        "semantic_free_minus_unconstrained_value_fidelity_error": semantic_vs_unconstrained_value,
        "semantic_free_minus_unconstrained_action_agreement": semantic_vs_unconstrained_action,
        "candidate_relative_reward_degradation": candidate_reward_degradation,
        "candidate_relative_logit_error_increase": candidate_logit_degradation,
        "candidate_relative_value_error_increase": candidate_value_degradation,
        "candidate_action_agreement_drop": candidate_action_degradation,
        "candidate_degradation_pass": candidate_degradation_pass_flags,
        "adaptive_boundary_saturation_rate": [
            float(row["boundary_saturation_rate"])
            for row in adaptive_rows if row["variant"] == "Adaptive-K"
        ],
    })
    cis = {
        key: VC._bootstrap_mean_ci(value, seed=4100 + index)
        for index, (key, value) in enumerate(metrics.items())
    }
    c_disagreement_fidelity_all = all(
        cis[f"c_core_minus_{comp.lower().replace('-core','').replace('-','_')}_disagreement_logit_fidelity_error"][0] > 0.0
        and cis[f"c_core_minus_{comp.lower().replace('-core','').replace('-','_')}_disagreement_value_fidelity_error"][0] > 0.0
        and cis[f"c_core_minus_{comp.lower().replace('-core','').replace('-','_')}_disagreement_action_agreement"][0] > 0.0
        for comp in allocation_fidelity_comparators
    )
    for row in pair_rows:
        if row["variant"] not in {
            "Recurrent-BC-CD", "Recurrent-BC-CD-NoWarmStart",
        }:
            continue
        if int(float(row.get("promotion_transient_horizon", -1))) != int(
            PAPER_B_PROMOTION_WINDOW_STEPS
        ):
            raise ValueError("pair promotion artifact violates frozen Paper-B W")
        if int(float(row.get("promotion_window_contract", -1))) != int(
            PAPER_B_PROMOTION_WINDOW_STEPS
        ):
            raise ValueError("pair promotion artifact omits frozen W contract")
        if int(float(row.get("promotion_total_teacher_steps", 0))) < int(
            PAPER_B_PROMOTION_WINDOW_STEPS
        ):
            raise ValueError("pair promotion artifact has fewer than W post-promotion steps")
    pair_fidelity_all = all(
        cis[f"full_pair_minus_{base.lower().replace('-','_')}_logit_fidelity_error"][0] > 0.0
        and cis[f"full_pair_minus_{base.lower().replace('-','_')}_value_fidelity_error"][0] > 0.0
        and cis[f"full_pair_minus_{base.lower().replace('-','_')}_action_agreement"][0] > 0.0
        for base in pair_structural_baselines
    )
    adaptive_saturation_limit = float(
        adaptive_manifest["max_boundary_saturation_fraction"]
    )
    adaptive_saturation_ok = all(
        float(row["boundary_saturation_rate"]) < adaptive_saturation_limit + 1e-12
        for row in adaptive_rows if row["variant"] == "Adaptive-K"
    )
    candidate_degradation_ok = bool(
        candidate_degradation_pass_flags
        and all(value > 0.5 for value in candidate_degradation_pass_flags)
    )
    primary_conditions = {
        "C_selector_beats_absD_when_profiles_disagree": cis[
            "c_core_minus_absd_disagreement_f1"
        ][0] > 0.0,
        **{
            "C_selector_beats_"
            f"{comparator.lower().replace('-core', '').replace('-', '_')}"
            "_when_profiles_disagree": cis[
                "c_core_minus_"
                f"{comparator.lower().replace('-core', '').replace('-', '_')}"
                "_disagreement_selector_f1"
            ][0] > 0.0
            for comparator in (
                "Attention-Core", "Random-Core", "Correlation-Core",
                "WeakPrior-Core",
            )
        },
        "H1a_C_improves_full_explicit_decision_fidelity_over_all_stated_comparators": bool(
            c_disagreement_fidelity_all
        ),
        "CD_contrastive_latent_improves_profile_retrieval": cis[
            "bc_minus_full_cd_retrieval_mae"
        ][0] > 0.0,
        "CD_contrastive_latent_aligns_with_profile_distance": cis[
            "full_cd_minus_bc_latent_profile_geometry"
        ][0] > 0.0,
        "CD_contrastive_latent_improves_decision_fidelity": cis[
            "full_cd_minus_bc_logit_fidelity_error"
        ][0] > 0.0 and cis["full_cd_minus_bc_action_agreement"][0] > 0.0,
        "CD_contrastive_latent_improves_value_fidelity": cis[
            "full_cd_minus_bc_value_fidelity_error"
        ][0] > 0.0,
        "H2a_full_pair_representation_beats_shared_pooled_and_feedforward": bool(
            pair_fidelity_all
        ),
        "shadow_warm_start_reduces_post_promotion_transient": cis[
            "warm_start_minus_no_warm_promotion_policy_kl"
        ][0] > 0.0 and cis[
            "warm_start_minus_no_warm_promotion_value_error"
        ][0] > 0.0 and all(count > 0 for count in warm_start_events),
        "semantic_free_memory_improves_reward_over_single_mean": cis[
            "semantic_free_minus_single_mean_reward"
        ][0] > 0.0,
        "semantic_free_memory_improves_policy_fidelity": cis[
            "semantic_free_minus_single_mean_logit_fidelity_error"
        ][0] > 0.0 and cis[
            "semantic_free_minus_single_mean_action_agreement"
        ][0] > 0.0,
        "semantic_free_memory_improves_value_fidelity": cis[
            "semantic_free_minus_single_mean_value_fidelity_error"
        ][0] > 0.0,
        "semantic_free_matches_single_mean_trainable_budget": max(
            abs(value) for value in periphery_parameter_budget_delta
        ) <= max(
            1.0,
            0.05 * min(
                float(row["trainable_parameter_bytes"])
                for row in periphery_rows
                if row["variant"] == "Semantic-Free"
            )
        ),
        "semantic_routing_beats_unconstrained_soft_slots": (
            cis["semantic_free_minus_unconstrained_logit_fidelity_error"][0] > 0.0
            and cis["semantic_free_minus_unconstrained_value_fidelity_error"][0] > 0.0
            and cis["semantic_free_minus_unconstrained_action_agreement"][0] > 0.0
        ),
        "semantic_memory_extends_reward_compute_pareto_frontier": (
            cis["scaling_semantic_pareto_nondominated"][0] > 0.5
        ),
        "adaptive_budget_cost_match_is_valid": (
            float(adaptive_manifest.get("realised_confirmatory_cost_gap", float("inf")))
            <= 0.5 + 1e-9
        ),
        "adaptive_budget_does_not_saturate_at_boundaries_on_most_updates": bool(
            adaptive_saturation_ok
        ),
        "H1b_adaptive_budget_improves_decision_fidelity_at_matched_compute": (
            cis["adaptive_minus_matched_fixed_logit_fidelity"][0] > 0.0
            and cis["adaptive_minus_matched_fixed_value_fidelity"][0] > 0.0
            and cis["adaptive_minus_matched_fixed_action_agreement"][0] > 0.0
        ),
        "candidate_restricted_edge_accounting_is_valid": bool(
            bounded_edge_accounting_valid
        ),
        "candidate_restricted_oracle_recall_is_valid": bool(
            candidate_oracle_recall_valid
        ),
        "candidate_oracle_ranking_stability_is_valid": bool(
            candidate_oracle_stability_valid
        ),
        "candidate_pair_state_eviction_is_O_Et": bool(candidate_state_eviction_valid),
        "candidate_reward_and_fidelity_degradation_is_within_frozen_bounds": bool(
            candidate_degradation_ok
        ),
        "dynamic_candidate_provider_contract_is_valid": bool(
            candidate_provider_contract_valid
        ),
        "dynamic_candidate_occupancy_gate_is_valid": bool(
            candidate_occupancy_gate_valid
        ),
        "dynamic_candidate_provider_work_bound_is_valid": bool(
            candidate_provider_work_bound_valid
        ),
        # Bounded measured edges alone are not a population-linear claim: a
        # dense candidate constructor remains quadratic.  This condition is
        # deliberately separate so the report can retain honest O(E) results
        # without licensing the stronger O(N) wording.
        "population_linear_scaling_claim_is_eligible": bool(
            bounded_edge_accounting_valid
            and candidate_oracle_recall_valid
            and candidate_oracle_stability_valid
            and candidate_state_eviction_valid
            and candidate_degradation_ok
            and candidate_construction_linear_valid
            and feature_snapshot_linear_valid
            and candidate_provider_contract_valid
            and candidate_occupancy_gate_valid
            and candidate_provider_work_bound_valid
        ),
    }
    secondary_predictions = {
        "C_selector_beats_absD_at_equal_budget": cis[
            "c_core_minus_absd_selector_f1"
        ][0] > 0.0,
        "C_selector_beats_attention_at_equal_budget": cis[
            "c_core_minus_attention_selector_f1"
        ][0] > 0.0,
        "C_selector_beats_random_at_equal_budget": cis[
            "c_core_minus_random_selector_f1"
        ][0] > 0.0,
        "C_selector_beats_correlation_at_equal_budget": cis[
            "c_core_minus_correlation_selector_f1"
        ][0] > 0.0,
        "C_selector_beats_weak_prior_at_equal_budget": cis[
            "c_core_minus_weak_prior_selector_f1"
        ][0] > 0.0,
        "oracle_selector_beats_random_at_equal_budget": cis[
            "oracle_minus_random_selector_f1"
        ][0] > 0.0,
        "C_allocation_improves_end_to_end_reward_over_absD": cis[
            "c_core_minus_absd_reward"
        ][0] > 0.0,
        "C_allocation_improves_policy_fidelity_over_absD": cis[
            "c_core_minus_absd_logit_fidelity_error"
        ][0] > 0.0 and cis["c_core_minus_absd_action_agreement"][0] > 0.0,
        "C_allocation_improves_value_fidelity_over_absD": cis[
            "c_core_minus_absd_value_fidelity_error"
        ][0] > 0.0,
        "capacity_pooling_beats_absD_weighted_pooling": cis[
            "capacity_pooling_minus_absD_pooling_reward"
        ][0] > 0.0,
        "semantic_memory_beats_attention_weighted_aggregate": cis[
            "semantic_free_minus_attention_mean_reward"
        ][0] > 0.0,
        "semantic_memory_beats_pure_mean_field_on_scaling_panel": cis[
            "scaling_semantic_minus_pure_mean_field_reward"
        ][0] > 0.0,
    }
    conditions = {**primary_conditions, **secondary_predictions}
    supported = all(primary_conditions.values())
    return {
        "paper": "B",
        "overall_status": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "conditions": conditions,
        "primary_conditions": primary_conditions,
        "secondary_mechanism_predictions": secondary_predictions,
        "paired_metrics": metrics,
        "paired_ci95": cis,
        "warm_start_promotion_events": warm_start_events,
        "scaling_agent_counts": agent_counts,
        "scaling_core_budgets": core_budgets,
        "scaling_manifest": scaling_manifest,
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
