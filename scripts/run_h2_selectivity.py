"""Run H2/RQ2 selectivity and structural-recovery experiments.

Each model/seed is evaluated in two matched environments:

* ``behavioral_drift`` changes policy execution while keeping the structural
  influence mechanism fixed.
* ``structural_shift`` changes the influence mechanism at phase boundaries.

For runners with an explicit influence matrix, Eq. 33 is measured as
``||W_t - W_(t-1)||_F / ||W_(t-1)||_F``. ``CorrelationMeanField`` is the
required observational comparator. ``PureMeanField`` may be requested as an
extra reward control, but its selectivity ratio is explicitly not applicable
because it has no W matrix.

The numerator and denominator use the same number of evaluation intervals,
starting with the interval that straddles their respective structural and
behavioral change events. Ordinary between-change adaptation is reported
separately as background drift.

Artifacts are written under a unique run directory. ``latest_attempt.json``
is switched to ``complete`` only after every requested model/seed/mode has
finished and the summary checksum has been recorded. This prevents a failed
attempt from silently publishing a summary left by an older run.
"""
import argparse
import csv
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone

import numpy as np

try:
    from exp_common import ROOT, append_jsonl, delta_norm, ensure_dir, last, make_args, w_matrix
except ModuleNotFoundError:  # Support ``import scripts.run_h2_selectivity`` in tests.
    from scripts.exp_common import (
        ROOT,
        append_jsonl,
        delta_norm,
        ensure_dir,
        last,
        make_args,
        w_matrix,
    )

import run_experiment as RE


# Eq. 33 requires an explicit influence matrix W. CorrelationMeanField is the
# required observational comparator; NoTwoTimescale isolates scheduling only.
# PureMeanField may still be requested explicitly as a reward control, but it
# cannot define Eq. 33.
MODELS = ["Final-CIGAMF", "CorrelationMeanField", "NoTwoTimescale"]
MODES = ["behavioral_drift", "structural_shift"]
PROTOCOL_VERSION = "h2_matched_change_v2"
CHANGE_WINDOW_EVAL_INTERVALS = 2


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _new_run_id():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _atomic_write_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-h2-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=float)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_write_csv(path, rows):
    if not rows:
        raise ValueError("cannot publish an empty H2 summary")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    fd, tmp = tempfile.mkstemp(prefix=".tmp-h2-", suffix=".csv", dir=directory)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_copy(source, destination):
    directory = os.path.dirname(os.path.abspath(destination))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-h2-", suffix=".csv", dir=directory)
    try:
        with open(source, "rb") as src, os.fdopen(fd, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        os.replace(tmp, destination)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phi_fingerprint(env):
    try:
        return tuple(
            (
                int(ego),
                tuple(
                    sorted(
                        (int(key), round(float(value), 9))
                        for key, value in row.items()
                    )
                ),
            )
            for ego, row in sorted(env.gt_influence_by_ego.items())
        )
    except Exception:
        return None


def _mean_finite(values):
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _runner_influence_matrix(runner, n_agents):
    if hasattr(runner, "get_influence_matrix"):
        matrix = np.asarray(runner.get_influence_matrix(), dtype=np.float64)
        expected = (int(n_agents), int(n_agents))
        if matrix.shape != expected:
            raise RuntimeError(
                f"runner influence matrix has shape {matrix.shape}; expected {expected}"
            )
        return matrix.copy()
    if hasattr(runner, "belief_modules"):
        return w_matrix(runner, n_agents)
    return None


def _recovery_statistics(
    eval_records,
    shift_episodes,
    trigger_episodes,
    causal_horizon_steps,
    max_steps,
    eval_every,
):
    """Compute non-negative, unit-consistent recovery latency per shift.

    ``causal_horizon_steps`` is measured in environment steps, whereas H2
    latency is measured in evaluation intervals of training episodes. The
    conversion therefore divides by both ``max_steps`` and ``eval_every``.
    The corrected latency is clamped at zero because an observation delay
    cannot make recovery occur before the shift.
    """
    delay_intervals = float(causal_horizon_steps) / float(
        max(1, int(max_steps)) * max(1, int(eval_every))
    )
    shifts = sorted({int(ep) for ep in shift_episodes})
    triggers = sorted({int(ep) for ep in trigger_episodes})
    details = []

    for index, shift_ep in enumerate(shifts):
        next_shift = shifts[index + 1] if index + 1 < len(shifts) else None
        pre = [
            float(row["f1"])
            for row in eval_records
            if int(row["episode"]) < shift_ep and np.isfinite(row["f1"])
        ]
        baseline = float(np.mean(pre[-3:])) if pre else float("nan")
        target = 0.9 * baseline if np.isfinite(baseline) else float("nan")

        post = [
            row
            for row in eval_records
            if int(row["episode"]) >= shift_ep
            and (next_shift is None or int(row["episode"]) < next_shift)
            and np.isfinite(row["f1"])
        ]
        recovered_ep = None
        if np.isfinite(target):
            for row in post:
                if float(row["f1"]) >= target:
                    recovered_ep = int(row["episode"])
                    break

        matching_triggers = [
            ep
            for ep in triggers
            if ep >= shift_ep and (next_shift is None or ep < next_shift)
        ]
        trigger_ep = matching_triggers[0] if matching_triggers else None

        raw = (
            float(recovered_ep - shift_ep) / float(max(1, int(eval_every)))
            if recovered_ep is not None
            else float("nan")
        )
        corrected = max(0.0, raw - delay_intervals) if np.isfinite(raw) else float("nan")
        details.append({
            "shift_episode": shift_ep,
            "pre_shift_f1": baseline,
            "target_f1": target,
            "recovered_episode": recovered_ep,
            "trigger_episode": trigger_ep,
            "raw_latency_intervals": raw,
            "corrected_latency_intervals": corrected,
        })

    raw_values = [item["raw_latency_intervals"] for item in details]
    corrected_values = [item["corrected_latency_intervals"] for item in details]
    # A completed shift with no recovery is a finite scientific result, not a
    # missing value. Use -1 so artifact validators can distinguish it from an
    # experiment that never observed a shift at all.
    raw_mean = _mean_finite(raw_values)
    corrected_mean = _mean_finite(corrected_values)
    if shifts and not np.isfinite(raw_mean):
        raw_mean = -1.0
    if shifts and not np.isfinite(corrected_mean):
        corrected_mean = -1.0
    return {
        "trigger_delay_intervals": delay_intervals,
        "recovery_latency_raw_intervals": raw_mean,
        "recovery_latency_intervals": corrected_mean,
        "n_recovered_shifts": sum(
            int(np.isfinite(item["corrected_latency_intervals"])) for item in details
        ),
        "n_shift_with_trigger": sum(item["trigger_episode"] is not None for item in details),
        "recovery_by_shift": details,
    }


def _validate_episode_events(events, episodes):
    numbers = [int(event["episode"]) for event in events]
    expected = list(range(1, int(episodes) + 1))
    if numbers != expected:
        raise RuntimeError(
            "runner episode event stream is incomplete or non-monotonic: "
            f"expected 1..{episodes}, got {numbers[:3]}...{numbers[-3:]}"
        )


def _matched_change_interval_mask(finite_rows, change_episodes, n_intervals):
    """Select N delta intervals beginning with the interval straddling change.

    Each row is ``(interval_start, interval_end, delta)``. Endpoint-only masks
    are biased here: behavioral change 40 lies in W(40)-W(30), whereas
    structural change 41 lies in W(50)-W(40). Selecting the straddling interval
    in both arms gives them identical relative lag. Incomplete terminal windows
    are reported but excluded from the estimand.
    """
    n_intervals = int(n_intervals)
    if n_intervals <= 0:
        raise ValueError("n_intervals must be positive")
    mask = np.zeros(len(finite_rows), dtype=bool)
    changes = sorted({int(ep) for ep in change_episodes})
    windows = []
    for index, change_episode in enumerate(changes):
        next_change = changes[index + 1] if index + 1 < len(changes) else None
        transition = next((
            row_index
            for row_index, (start_episode, end_episode, _) in enumerate(finite_rows)
            if int(start_episode) < change_episode <= int(end_episode)
        ), None)
        candidates = []
        if transition is not None:
            candidates = [
                row_index
                for row_index in range(transition, len(finite_rows))
                if (
                    next_change is None
                    or int(finite_rows[row_index][1]) < next_change
                )
            ][:n_intervals]
        complete = len(candidates) == n_intervals
        if complete:
            mask[candidates] = True
        windows.append({
            "change_episode": change_episode,
            "delta_intervals": [
                [int(finite_rows[i][0]), int(finite_rows[i][1])]
                for i in candidates
            ],
            "complete": complete,
        })
    return mask, windows


def run_one(model, mode, seed, episodes, eval_every, device, out_root, run_id):
    out_dir = os.path.join(out_root, f"{model}_{mode}_seed{seed}")
    os.mkdir(out_dir)
    jsonl = os.path.join(out_dir, "eval.jsonl")
    with open(jsonl, "x", encoding="utf-8"):
        pass

    RE.set_global_seed(seed)
    cfg = RE.default_cfg()
    cfg["seed"] = seed
    make_args(seed=seed, device=device)  # Validate the shared CLI defaults.

    max_steps = 30
    phase_length = 40
    env = RE.make_main_env(
        task_mode=mode,
        n_agents=24,
        max_steps=max_steps,
        phase_length=phase_length,
        seed=seed,
    )
    runner = RE.make_runner(model, env, cfg, device)

    causal_horizon = int(cfg.get("causal_horizon", 8))
    previous_w = _runner_influence_matrix(runner, env.n_agents)
    has_influence_matrix = previous_w is not None
    previous_phi = _phi_fingerprint(env)
    deltas = []
    previous_delta_episode = 0
    eval_records = []
    shift_episodes = []
    behavioral_shift_episodes = []
    trigger_episodes = []
    event_cursor = 0

    while int(getattr(runner, "episodes_completed", 0)) < episodes:
        completed_before = int(getattr(runner, "episodes_completed", 0))
        chunk_size = min(eval_every, episodes - completed_before)
        runner.run(n_episodes=chunk_size, eval_every=eval_every)
        completed = int(getattr(runner, "episodes_completed", 0))
        if completed != completed_before + chunk_size:
            raise RuntimeError(
                f"{model}/{mode}/seed{seed}: runner completed {completed} episodes; "
                f"expected {completed_before + chunk_size}"
            )

        all_events = list(getattr(runner, "episode_events", []))
        new_events = all_events[event_cursor:]
        event_cursor = len(all_events)

        chunk_shifts = [
            int(event["episode"])
            for event in new_events
            if int(event.get("structural_shift", 0))
        ]
        chunk_triggers = [
            int(event["episode"])
            for event in new_events
            if int(event.get("triggered", 0))
        ]
        chunk_behavioral_shifts = [
            int(event["episode"])
            for event in new_events
            if int(event.get("behavioral_shift", 0))
        ]

        # Compatibility fallback for a future runner that has not yet adopted
        # the per-episode event interface. Current H2 runners use exact events.
        phi_now = _phi_fingerprint(env)
        if not new_events and previous_phi is not None and phi_now is not None:
            if phi_now != previous_phi:
                chunk_shifts.append(completed)
        previous_phi = phi_now

        shift_episodes.extend(chunk_shifts)
        behavioral_shift_episodes.extend(chunk_behavioral_shifts)
        trigger_episodes.extend(chunk_triggers)

        delta = float("nan")
        if has_influence_matrix:
            current_w = _runner_influence_matrix(runner, env.n_agents)
            delta = delta_norm(previous_w, current_w)
            previous_w = current_w

        history = getattr(runner, "history", {})
        row = {
            "run_id": run_id,
            "episode": completed,
            "delta": delta,
            "delta_interval_start": previous_delta_episode,
            "delta_interval_end": completed,
            "n_shift_events": len(chunk_shifts),
            "shift_episodes": chunk_shifts,
            "is_shift_window": int(bool(chunk_shifts)),
            "n_behavioral_shift_events": len(chunk_behavioral_shifts),
            "behavioral_shift_episodes": chunk_behavioral_shifts,
            "n_triggers": len(chunk_triggers),
            "trigger_episodes": chunk_triggers,
            "triggered": int(bool(chunk_triggers)),
            "f1": last(history, "mean_f1"),
            "reward": last(history, "mean_reward"),
            "core_size": last(history, "mean_core_size"),
            "tier_separation_ratio": float(
                getattr(env, "tier_separation_ratio", lambda: float("nan"))()
            ),
            "n_close_coupling_pairs": int(
                env.close_coupling_pairs()
                if hasattr(env, "close_coupling_pairs")
                else -1
            ),
        }
        append_jsonl(jsonl, row)
        eval_records.append(row)
        deltas.append((previous_delta_episode, completed, delta))
        previous_delta_episode = completed
        print(
            f"[H2 {model}/{mode} s{seed}] ep={completed} delta={delta:.4f} "
            f"f1={row['f1']:.3f} structural={chunk_shifts} "
            f"behavioral={chunk_behavioral_shifts} triggers={chunk_triggers}"
        )

    events = list(getattr(runner, "episode_events", []))
    if events:
        _validate_episode_events(events, episodes)
    if not eval_records or int(eval_records[-1]["episode"]) != episodes:
        raise RuntimeError(f"{model}/{mode}/seed{seed}: missing terminal evaluation")

    finite_rows = [
        (start_ep, end_ep, delta)
        for start_ep, end_ep, delta in deltas
        if np.isfinite(delta)
    ]
    structural_mask, structural_windows = _matched_change_interval_mask(
        finite_rows, shift_episodes, CHANGE_WINDOW_EVAL_INTERVALS
    )
    behavioral_mask, behavioral_windows = _matched_change_interval_mask(
        finite_rows, behavioral_shift_episodes, CHANGE_WINDOW_EVAL_INTERVALS
    )
    delta_values = np.asarray(
        [delta for _, _, delta in finite_rows], dtype=np.float64
    )
    delta_struct = (
        float(np.mean(delta_values[structural_mask]))
        if structural_mask.any()
        else float("nan")
    )
    delta_behav = (
        float(np.mean(delta_values[behavioral_mask]))
        if behavioral_mask.any()
        else float("nan")
    )
    background_mask = ~(structural_mask | behavioral_mask)
    delta_background = (
        float(np.mean(delta_values[background_mask]))
        if background_mask.any()
        else float("nan")
    )
    selectivity_ratio = (
        float(delta_struct / delta_background)
        if np.isfinite(delta_struct)
        and np.isfinite(delta_background)
        and delta_background > 1e-6
        else float("nan")
    )

    recovery = _recovery_statistics(
        eval_records=eval_records,
        shift_episodes=shift_episodes,
        trigger_episodes=trigger_episodes,
        causal_horizon_steps=causal_horizon,
        max_steps=max_steps,
        eval_every=eval_every,
    )
    metric_applicable = bool(has_influence_matrix)
    summary = {
        "run_id": run_id,
        "protocol_version": PROTOCOL_VERSION,
        "complete": True,
        "model": model,
        "runner_class": type(runner).__name__,
        "ablation_contract": getattr(runner, "ablation_contract", ""),
        "mode": mode,
        "seed": seed,
        "episodes": episodes,
        "episodes_completed": int(getattr(runner, "episodes_completed", 0)),
        "eval_every": eval_every,
        "n_eval_points": len(eval_records),
        "expected_eval_points": int(np.ceil(episodes / eval_every)),
        "metric_applicable": metric_applicable,
        "not_applicable_reason": "" if metric_applicable else "runner_has_no_influence_matrix_W",
        "delta_mean_struct": delta_struct,
        "delta_mean_behav": delta_behav,
        "delta_mean_background": delta_background,
        "selectivity_ratio": selectivity_ratio,
        "change_window_eval_intervals": CHANGE_WINDOW_EVAL_INTERVALS,
        "structural_change_windows": structural_windows,
        "behavioral_change_windows": behavioral_windows,
        "n_complete_structural_windows": sum(
            int(window["complete"]) for window in structural_windows
        ),
        "n_complete_behavioral_windows": sum(
            int(window["complete"]) for window in behavioral_windows
        ),
        "n_shift_events": len(set(shift_episodes)),
        "shift_episodes": sorted(set(shift_episodes)),
        "n_behavioral_shift_events": len(set(behavioral_shift_episodes)),
        "behavioral_shift_episodes": sorted(set(behavioral_shift_episodes)),
        "n_triggers": len(set(trigger_episodes)),
        "trigger_episodes": sorted(set(trigger_episodes)),
        "final_f1": last(getattr(runner, "history", {}), "mean_f1"),
        "final_reward": last(getattr(runner, "history", {}), "mean_reward"),
        "association_window_steps": getattr(
            runner, "association_window_steps", None
        ),
        "association_min_steps": getattr(
            runner, "association_min_steps", None
        ),
        "association_min_action_support": getattr(
            runner, "association_min_action_support", None
        ),
        "association_statistic": getattr(runner, "association_statistic", None),
        "association_signed": getattr(runner, "association_signed", None),
    }
    summary.update(recovery)
    _atomic_write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def _validate_complete_rows(rows, models, seeds, run_id):
    expected = {(str(model), int(seed)) for seed in seeds for model in models}
    actual = {(str(row["model"]), int(row["seed"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError(
            "H2 summary is incomplete: "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )
    for row in rows:
        if (
            row.get("run_id") != run_id
            or row.get("protocol_version") != PROTOCOL_VERSION
            or int(row.get("complete", 0)) != 1
        ):
            raise RuntimeError("H2 row does not belong to the complete current run")


def _resolve_out_root(value):
    if value is None:
        return os.path.join(ROOT, "results", "h2")
    return os.path.abspath(value if os.path.isabs(value) else os.path.join(ROOT, value))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=MODELS,
        help=(
            "Models with an explicit W matrix (default: Final-CIGAMF, "
            "CorrelationMeanField, and the faithful Final scheduler-only "
            "NoTwoTimescale ablation). PureMeanField may be "
            "requested as a non-applicable "
            "reward control, but it cannot define Eq. 33 selectivity."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out-root",
        default=None,
        help="H2 artifact root (absolute or relative to the repository root)",
    )
    parser.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.eval_every <= 0:
        parser.error("--eval_every must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")

    out_root = ensure_dir(_resolve_out_root(args.out_root))
    runs_root = ensure_dir(os.path.join(out_root, "runs"))
    run_id = args.run_id or _new_run_id()
    run_dir = os.path.join(runs_root, run_id)
    os.mkdir(run_dir)
    latest_attempt_path = os.path.join(out_root, "latest_attempt.json")
    started_at = _utc_now()
    attempt = {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "experiment": "h2_selectivity",
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "episodes": args.episodes,
        "eval_every": args.eval_every,
        "change_window_eval_intervals": CHANGE_WINDOW_EVAL_INTERVALS,
        "models": list(args.models),
        "seeds": list(args.seeds),
        "required_comparator": "CorrelationMeanField",
        "required_comparator_present": "CorrelationMeanField" in args.models,
        "required_claim_models": list(MODELS),
        "required_claim_models_present": set(MODELS).issubset(set(args.models)),
        "expected_attempts": [
            {"model": model, "mode": mode, "seed": int(seed)}
            for seed in args.seeds
            for model in args.models
            for mode in MODES
        ],
        "completed_attempts": [],
        "failed_attempt": None,
        "expected_summary_rows": len(args.models) * len(args.seeds),
        "run_dir": os.path.relpath(run_dir, ROOT),
    }
    _atomic_write_json(latest_attempt_path, attempt)

    try:
        rows = []
        for seed in args.seeds:
            for model in args.models:
                per_mode = {}
                for mode in MODES:
                    per_mode[mode] = run_one(
                        model=model,
                        mode=mode,
                        seed=seed,
                        episodes=args.episodes,
                        eval_every=args.eval_every,
                        device=args.device,
                        out_root=run_dir,
                        run_id=run_id,
                    )
                    attempt["completed_attempts"].append({
                        "model": model,
                        "mode": mode,
                        "seed": int(seed),
                    })
                    _atomic_write_json(latest_attempt_path, attempt)

                structural = per_mode["structural_shift"]
                behavioral = per_mode["behavioral_drift"]
                delta_struct = structural["delta_mean_struct"]
                delta_behav = behavioral["delta_mean_behav"]
                sr_cross = (
                    float(delta_struct / delta_behav)
                    if np.isfinite(delta_struct)
                    and np.isfinite(delta_behav)
                    and delta_behav > 1e-6
                    else float("nan")
                )
                claim_evaluable = bool(
                    structural["metric_applicable"]
                    and behavioral["metric_applicable"]
                    and structural["n_shift_events"] > 0
                    and behavioral["n_behavioral_shift_events"] > 0
                    and structural["n_complete_structural_windows"]
                    == behavioral["n_complete_behavioral_windows"]
                    and structural["n_complete_structural_windows"] > 0
                    and np.isfinite(sr_cross)
                )
                rows.append({
                    "run_id": run_id,
                    "protocol_version": PROTOCOL_VERSION,
                    "complete": 1,
                    "model": model,
                    "runner_class": structural["runner_class"],
                    "ablation_contract": structural["ablation_contract"],
                    "seed": seed,
                    "episodes": args.episodes,
                    "eval_every": args.eval_every,
                    "claim_evaluable": int(claim_evaluable),
                    "required_comparator_present": int(
                        "CorrelationMeanField" in args.models
                    ),
                    "not_applicable_reason": (
                        ""
                        if structural["metric_applicable"]
                        else structural["not_applicable_reason"]
                    ),
                    "delta_struct": delta_struct,
                    "delta_behav": delta_behav,
                    "delta_background_structural_run": structural[
                        "delta_mean_background"
                    ],
                    "delta_background_behavioral_run": behavioral[
                        "delta_mean_background"
                    ],
                    "SR_cross_run": sr_cross,
                    "SR_within_structural": structural["selectivity_ratio"],
                    "recovery_latency": structural["recovery_latency_intervals"],
                    "recovery_latency_raw": structural["recovery_latency_raw_intervals"],
                    "trigger_delay_intervals": structural["trigger_delay_intervals"],
                    "n_shift_events": structural["n_shift_events"],
                    "n_behavioral_shift_events": behavioral[
                        "n_behavioral_shift_events"
                    ],
                    "n_complete_structural_windows": structural[
                        "n_complete_structural_windows"
                    ],
                    "n_complete_behavioral_windows": behavioral[
                        "n_complete_behavioral_windows"
                    ],
                    "n_recovered_shifts": structural["n_recovered_shifts"],
                    "n_shift_with_trigger": structural["n_shift_with_trigger"],
                    "n_triggers": structural["n_triggers"],
                    "final_f1_struct": structural["final_f1"],
                    "association_window_steps": structural[
                        "association_window_steps"
                    ],
                    "association_min_steps": structural[
                        "association_min_steps"
                    ],
                    "association_min_action_support": structural[
                        "association_min_action_support"
                    ],
                    "association_statistic": structural[
                        "association_statistic"
                    ],
                    "association_signed": structural["association_signed"],
                })

        _validate_complete_rows(rows, args.models, args.seeds, run_id)
        if attempt["completed_attempts"] != attempt["expected_attempts"]:
            raise RuntimeError("H2 per-mode attempt matrix is incomplete")
        run_summary = os.path.join(run_dir, "summary_h2.csv")
        _atomic_write_csv(run_summary, rows)
        summary_hash = _sha256(run_summary)
        completed_at = _utc_now()
        manifest = {
            **attempt,
            "status": "complete",
            "completed_at": completed_at,
            "summary_path": os.path.relpath(run_summary, ROOT),
            "summary_sha256": summary_hash,
            "summary_rows": len(rows),
        }
        _atomic_write_json(os.path.join(run_dir, "manifest.json"), manifest)

        # Backward-compatible convenience path. The collector still validates
        # latest_attempt.json and the immutable run summary before accepting it.
        _atomic_copy(run_summary, os.path.join(out_root, "summary_h2.csv"))
        _atomic_write_json(latest_attempt_path, manifest)
        print(f"\n[H2] complete run {run_id}: {run_summary}")
        print(
            "[H2] Artifact matrix is complete. Scientific gates remain data "
            "outcomes and are evaluated by the validated collector."
        )
        return 0
    except BaseException as exc:
        failed = {
            **attempt,
            "status": "failed",
            "failed_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_attempt": (
                attempt["expected_attempts"][len(attempt["completed_attempts"])]
                if len(attempt["completed_attempts"])
                < len(attempt["expected_attempts"])
                else None
            ),
        }
        try:
            _atomic_write_json(os.path.join(run_dir, "manifest.json"), failed)
            _atomic_write_json(latest_attempt_path, failed)
        finally:
            raise


if __name__ == "__main__":
    main()
