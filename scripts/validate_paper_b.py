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


# Derive matrix contracts from the producing runners so a producer update
# cannot leave the final validator silently pinned to a stale variant set.
EXPECTED_ALLOCATION = set(PB_ALLOCATION.VARIANTS)
EXPECTED_PAIR = set(PB_PAIR.VARIANTS)
EXPECTED_PERIPHERY = set(PB_PERIPHERY.VARIANTS)
EXPECTED_SCALING = set(PB_SCALING.VARIANTS)


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


def _load_scaling(root, expected_seeds):
    rows, manifest = _load_panel(
        root, "paper_b_scaling", "summary_paper_b_scaling.csv",
        EXPECTED_SCALING, expected_seeds,
    )
    degree = int(manifest.get("candidate_max_degree", 0))
    if degree <= 0:
        raise ValueError("scaling manifest omits a positive candidate_max_degree")
    if not manifest.get("candidate_policy"):
        raise ValueError("scaling manifest omits the frozen candidate policy")
    recall = manifest.get("candidate_oracle_recall")
    if not isinstance(recall, dict):
        raise ValueError("scaling manifest omits the oracle candidate-recall protocol")
    minimum = float(recall.get("minimum", float("nan")))
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise ValueError("scaling manifest has an invalid candidate-recall minimum")
    for key in ("states", "horizon", "trials"):
        if int(recall.get(key, 0)) <= 0:
            raise ValueError(f"scaling candidate-recall protocol omits positive {key}")
    return rows, manifest


def _mean_finite(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        raise ValueError("expected at least one finite value")
    return sum(values) / float(len(values))


def _load_adaptive_budget(root, expected_seeds):
    panel_root = os.path.join(root, "paper_b_adaptive_budget")
    with open(os.path.join(panel_root, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("complete"):
        raise ValueError("paper_b_adaptive_budget manifest is incomplete")
    with open(
        os.path.join(panel_root, "summary_paper_b_adaptive_budget.csv"),
        newline="", encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
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
    allocation, allocation_manifest = _load_panel(
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
    scaling_rows, scaling_manifest = _load_scaling(run_root, expected_seeds)
    adaptive_rows, adaptive_manifest = _load_adaptive_budget(run_root, expected_seeds)
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
    warm_start_transient = [-value for value in _paired(
        pair_rows, "Recurrent-BC-CD", "Recurrent-BC-CD-NoWarmStart",
        "post_promotion_bc_loss",
    )]
    warm_start_logit = [-value for value in _paired(
        pair_rows, "Recurrent-BC-CD", "Recurrent-BC-CD-NoWarmStart",
        "promotion_logit_error_auc",
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
    capacity_vs_absd_pooling = _paired(
        periphery_rows, "Semantic-Free", "AbsD-Pooling", "mean_reward"
    )
    semantic_vs_attention_mean = _paired(
        periphery_rows, "Semantic-Free", "Attention-Mean", "mean_reward"
    )
    agent_counts = sorted({int(row["n_agents"]) for row in scaling_rows})
    scaling_pareto = []
    semantic_frontier_flags = []
    for n_agents in agent_counts:
        by_variant = {}
        for row in scaling_rows:
            if int(row["n_agents"]) != n_agents:
                continue
            by_variant.setdefault(row["variant"], {})[int(row["seed"])] = row
        if set(by_variant) != EXPECTED_SCALING:
            raise ValueError(f"scaling variants missing at N={n_agents}")
        semantic = by_variant["Semantic-Free"]
        single = by_variant["Single-Mean"]
        full = by_variant["Full-Explicit"]
        pure = by_variant["PureMeanField"]
        if set(semantic) != set(single) or set(semantic) != set(full) or set(semantic) != set(pure):
            raise ValueError(f"scaling seeds are unpaired at N={n_agents}")
        for seed in sorted(semantic):
            points = {
                variant: {
                    "reward": float(values[seed]["mean_reward"]),
                    "throughput": float(values[seed]["throughput_total"]),
                    "memory": float(values[seed]["representation_memory_bytes"]),
                }
                for variant, values in by_variant.items()
            }
            subject = points["Semantic-Free"]
            dominated = any(
                other != "Semantic-Free"
                and candidate["reward"] >= subject["reward"]
                and candidate["throughput"] >= subject["throughput"]
                and candidate["memory"] <= subject["memory"]
                and (
                    candidate["reward"] > subject["reward"]
                    or candidate["throughput"] > subject["throughput"]
                    or candidate["memory"] < subject["memory"]
                )
                for other, candidate in points.items()
            )
            semantic_frontier_flags.append(0.0 if dominated else 1.0)
            scaling_pareto.append({
                "n_agents": n_agents,
                "seed": seed,
                "reward_vs_single": float(semantic[seed]["mean_reward"]) - float(single[seed]["mean_reward"]),
                "throughput_vs_full": float(semantic[seed]["throughput_total"]) - float(full[seed]["throughput_total"]),
                "memory_vs_full": float(full[seed]["representation_memory_bytes"]) - float(semantic[seed]["representation_memory_bytes"]),
                "reward_vs_pure_mean_field": float(semantic[seed]["mean_reward"]) - float(pure[seed]["mean_reward"]),
            })
    pareto_reward = [row["reward_vs_single"] for row in scaling_pareto]
    pareto_throughput = [row["throughput_vs_full"] for row in scaling_pareto]
    pareto_memory = [row["memory_vs_full"] for row in scaling_pareto]
    pareto_pure_mean = [row["reward_vs_pure_mean_field"] for row in scaling_pareto]
    adaptive_reward_vs_matched, adaptive_logit_vs_matched = [], []
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
    # Scaling rows must include the paper-promised fidelity and latency metrics.
    bounded_edge_accounting_valid = True
    candidate_oracle_recall_valid = True
    candidate_construction_subquadratic_valid = True
    candidate_recall_protocol = scaling_manifest["candidate_oracle_recall"]
    candidate_recall_minimum = float(candidate_recall_protocol["minimum"])
    if int(candidate_recall_protocol.get("horizon", -1)) != 1:
        raise ValueError(
            "Paper-B candidate recall must use the one-step capacity oracle"
        )
    if int(candidate_recall_protocol.get("trials", 0)) < 2:
        raise ValueError("Paper-B candidate recall requires repeated CRN trials")
    for row in scaling_rows:
        for key in (
            "policy_logit_l2_to_full_explicit", "value_mae_to_full_explicit",
            "action_agreement_to_full_explicit", "inference_latency_p50_ms",
            "inference_latency_p95_ms",
        ):
            if not math.isfinite(float(row.get(key, float("nan")))):
                raise ValueError(f"scaling row omits finite {key}")
        if row.get("inference_latency_protocol") != (
            "deterministic_policy_inference_without_sampling_or_epsilon_forcing"
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
        if variant == "Full-Explicit":
            # This is intentionally the dense upper reference, never evidence
            # for candidate-restricted edge scaling.
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
        recall = float(row.get("candidate_oracle_recall_at_degree", float("nan")))
        comparisons = int(row.get("candidate_oracle_recall_comparisons", 0))
        if (
            int(row.get("candidate_recall_applicable", 0)) != 1
            or not math.isfinite(recall)
            or comparisons <= 0
            or recall < candidate_recall_minimum
        ):
            candidate_oracle_recall_valid = False
        construction_flag = str(
            row.get("candidate_construction_subquadratic", "false")
        ).strip().lower()
        if construction_flag not in {"1", "true", "yes"}:
            candidate_construction_subquadratic_valid = False

    metrics = {
        "c_core_minus_absd_selector_f1": c_vs_d,
        "c_core_minus_absd_disagreement_f1": c_vs_d_disagreement,
        "c_core_minus_absd_selector_ndcg": c_vs_d_ndcg,
        "c_core_minus_attention_selector_f1": c_vs_attention,
        "c_core_minus_random_selector_f1": c_vs_random,
        "c_core_minus_correlation_selector_f1": c_vs_correlation,
        "c_core_minus_weak_prior_selector_f1": c_vs_weak_prior,
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
    }
    cis = {
        key: VC._bootstrap_mean_ci(value, seed=4100 + index)
        for index, (key, value) in enumerate(metrics.items())
    }
    primary_conditions = {
        "C_selector_beats_absD_at_equal_budget": cis[
            "c_core_minus_absd_selector_f1"
        ][0] > 0.0,
        "C_selector_beats_absD_when_profiles_disagree": cis[
            "c_core_minus_absd_disagreement_f1"
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
        "shadow_warm_start_reduces_post_promotion_transient": cis[
            "warm_start_minus_no_warm_post_promotion_bc_loss"
        ][0] > 0.0 and cis[
            "warm_start_minus_no_warm_promotion_logit_error"
        ][0] > 0.0 and cis[
            "warm_start_minus_no_warm_promotion_value_error"
        ][0] > 0.0 and cis[
            "warm_start_minus_no_warm_promotion_action_agreement"
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
        "semantic_routing_beats_unconstrained_soft_slots": cis[
            "semantic_free_minus_unconstrained_reward"
        ][0] > 0.0,
        "semantic_memory_extends_reward_compute_pareto_frontier": (
            cis["scaling_semantic_pareto_nondominated"][0] > 0.5
        ),
        "adaptive_budget_cost_match_is_valid": (
            float(adaptive_manifest.get("realised_confirmatory_cost_gap", float("inf")))
            <= 0.5 + 1e-9
        ),
        "candidate_restricted_edge_accounting_is_valid": bool(
            bounded_edge_accounting_valid
        ),
        "candidate_restricted_oracle_recall_is_valid": bool(
            candidate_oracle_recall_valid
        ),
        # Bounded measured edges alone are not a population-linear claim: a
        # dense candidate constructor remains quadratic.  This condition is
        # deliberately separate so the report can retain honest O(E) results
        # without licensing the stronger O(N) wording.
        "population_linear_scaling_claim_is_eligible": bool(
            bounded_edge_accounting_valid
            and candidate_oracle_recall_valid
            and candidate_construction_subquadratic_valid
        ),
    }
    secondary_predictions = {
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
