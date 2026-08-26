"""Run H2/RQ2 selectivity and structural-recovery experiments.

Each model/seed is evaluated in four matched factorial cells crossing a
structural mechanism change with a neighbour-policy change.

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
import copy
import csv
import hashlib
import json
import os
import pickle
import random
import tempfile
import uuid
from datetime import datetime, timezone

import numpy as np
import torch

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
from envs.causal_adapter import resolve_env_adapter
try:
    from h2_cusum_contract import (
        build_h2_cusum_contract, contract_hash, validate_calibration_artifact,
    )
except ModuleNotFoundError:
    from scripts.h2_cusum_contract import (
        build_h2_cusum_contract, contract_hash, validate_calibration_artifact,
    )


# Eq. 33 requires an explicit influence matrix W. CorrelationMeanField is the
# required observational comparator; NoTwoTimescale isolates scheduling only.
# PureMeanField may still be requested explicitly as a reward control, but it
# cannot define Eq. 33.
MODELS = [
    "Final-CIGAMF", "CorrelationMeanField", "NoTwoTimescale",
    "FixedRateTracker", "NoDetector", "NoUncertainty", "FastTracker",
]
FACTORIAL_CELLS = {
    "S0B0": (False, False),
    "S0B1": (False, True),
    "S1B0": (True, False),
    "S1B1": (True, True),
}
MODES = list(FACTORIAL_CELLS)
PROTOCOL_VERSION = "h2_factorial_frozen_policy_v6"
CHANGE_WINDOW_EVAL_INTERVALS = 2
RECOVERY_F1_VALID_FLOOR = 0.50
H2_EVALUATION_EGO_ROLES = ("collector",)
H2_MANIPULATED_NEIGHBOR_ROLES = (
    "gatekeeper", "relay", "blocker", "controller", "drifter",
)


def _apply_tracker_control(model, cfg):
    """Map tracking-control labels to isolated Final-CIGAMF mechanisms."""
    model = str(model)
    if model == "FixedRateTracker":
        cfg["force_graph_update_every_episode"] = True
        cfg["disable_drift_detector"] = True
    elif model == "NoDetector":
        cfg["disable_drift_detector"] = True
    elif model == "NoUncertainty":
        cfg["belief_uncertainty_scale"] = 0.0
    elif model == "FastTracker":
        cfg["force_graph_update_every_episode"] = True
        cfg["slow_ratio"] = 1.0
    return cfg


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


def _false_alarm_window_stats(events, *, start_episode, monitoring_horizon):
    """Count non-overlapping detector-ready windows and false alarms.

    CUSUM calibration controls the probability that a *monitoring trajectory*
    of a frozen horizon crosses threshold.  Comparing that target with
    ``n_triggers / episodes`` mixes two different denominators.  H2 therefore
    reconstructs independent, non-overlapping post-intervention windows of the
    same calibrated horizon, resetting a partial window whenever the detector
    is not monitoring-ready (e.g. reference recalibration after a trigger).
    """
    horizon = int(monitoring_horizon or 0)
    if horizon <= 0:
        return {
            "monitoring_window_count": 0,
            "false_alarm_window_count": 0,
            "false_alarm_window_rate": float("nan"),
        }
    selected = sorted(
        (event for event in events if int(event.get("episode", -1)) >= int(start_episode)),
        key=lambda event: int(event.get("episode", -1)),
    )
    current = []
    windows = 0
    false_alarms = 0
    previous_episode = None
    for event in selected:
        episode = int(event.get("episode", -1))
        ready = bool(event.get("drift_monitoring_ready", 0))
        contiguous = previous_episode is None or episode == previous_episode + 1
        previous_episode = episode
        if not ready or not contiguous:
            current = []
            if not ready:
                continue
        current.append(event)
        if len(current) == horizon:
            windows += 1
            false_alarms += int(any(bool(row.get("triggered", 0)) for row in current))
            current = []
    return {
        "monitoring_window_count": int(windows),
        "false_alarm_window_count": int(false_alarms),
        "false_alarm_window_rate": (
            float(false_alarms / windows) if windows > 0 else float("nan")
        ),
    }


def _fixed_estimand_panel(
    env,
    structural_factor,
    behavioral_factor,
    seed,
    n_states=8,
    horizon=8,
    discount=0.97,
    n_trials=8,
    core_budget=3,
):
    """Evaluate C and D on matched cloned states with fixed continuation."""
    if not all(
        callable(getattr(env, name, None))
        for name in ("clone_state", "restore_state", "sample_state_bank")
    ):
        return {"applicable": False, "capacity_mean": float("nan"), "direction_abs_mean": float("nan")}
    outer = env.clone_state()
    saved_override = getattr(env, "_behaviour_override", None)
    rows_c, rows_d = [], []
    capacity_by_ego = {}
    try:
        env.set_behaviour_override("cooperative")
        bank = env.sample_state_bank(
            n_states=int(n_states), burn_in=2, bank_seed=int(seed) + 3401,
            min_remaining_steps=int(horizon),
        )
        for raw_state in bank:
            state = copy.deepcopy(raw_state)
            env.restore_state(state)
            apply_oracle = getattr(env, "apply_factorial_intervention_for_oracle", None)
            if not callable(apply_oracle):
                raise RuntimeError(
                    "H2 fixed estimand panel requires the canonical oracle "
                    "factorial-intervention adapter"
                )
            apply_oracle(
                structural=bool(structural_factor),
                behavioral=False,
                behavior_mode="selfish",
            )
            state = env.clone_state()
            egos = _agents_with_roles(env, H2_EVALUATION_EGO_ROLES)
            targets = _agents_with_roles(env, H2_MANIPULATED_NEIGHBOR_ROLES)
            for ego in egos:
                same_zone_targets = [
                    target for target in targets
                    if target != ego and env.agent_zone[target] == env.agent_zone[ego]
                ]
                for target in same_zone_targets:
                    q_values = []
                    env.restore_state(state)
                    env.set_behaviour_override(
                        "selfish" if behavioral_factor else "cooperative"
                    )
                    pi = np.asarray(
                        env.scripted_policy_distribution(target), dtype=np.float64
                    )
                    valid_mask = resolve_env_adapter(env).valid_action_mask(target)
                    pi = np.where(valid_mask, pi, 0.0)
                    pi = pi / np.clip(pi.sum(), 1e-12, None)
                    valid_actions = np.flatnonzero(valid_mask)
                    for action in valid_actions:
                        env.restore_state(state)
                        env.set_behaviour_override("cooperative")
                        response = env.compute_oracle_lag_response_from_current_state(
                            ego_id=int(ego),
                            agent_j=int(target),
                            intervention_action=int(action),
                            horizon=int(horizon),
                            n_trials=int(n_trials),
                            forced_step=0,
                            continuation_policy=env.scripted_policy,
                            crn_seed=(int(seed) + 1) * 1000003
                            + len(rows_c) * 101,
                            discount=float(discount),
                        )
                        # Responses are relative to one shared reference
                        # rollout. The additive reference cancels in both C
                        # and D because the D contrast weights sum to zero.
                        q_values.append(float(response["discounted_response"]))
                    q_values = np.asarray(q_values, dtype=np.float64)
                    uniform = np.full(q_values.size, 1.0 / q_values.size)
                    rows_c.append(float(np.max(q_values) - np.min(q_values)))
                    rows_d.append(float(np.dot(pi - uniform, q_values)))
                    capacity_by_ego.setdefault(int(ego), {}).setdefault(
                        int(target), []
                    ).append(rows_c[-1])
        capacity_mean_by_ego = {
            int(ego): {
                int(target): _mean_finite(values)
                for target, values in targets.items()
            }
            for ego, targets in capacity_by_ego.items()
        }
        oracle_core_by_ego = {
            int(ego): [
                int(target)
                for target, _ in sorted(
                    targets.items(), key=lambda item: item[1], reverse=True
                )[:int(core_budget)]
            ]
            for ego, targets in capacity_mean_by_ego.items()
        }
        return {
            "applicable": bool(rows_c),
            "n_pair_states": int(len(rows_c)),
            "capacity_mean": _mean_finite(rows_c),
            "direction_abs_mean": _mean_finite(np.abs(rows_d)),
            "continuation_regime": "cooperative_fixed_after_intervention",
            "horizon": int(horizon),
            "discount": float(discount),
            "n_trials": int(n_trials),
            "core_budget": int(core_budget),
            "capacity_mean_by_ego": capacity_mean_by_ego,
            "oracle_core_by_ego": oracle_core_by_ego,
        }
    finally:
        env.set_behaviour_override(saved_override)
        env.restore_state(outer)


def _runner_influence_matrix(runner, n_agents, ego_ids=None):
    if hasattr(runner, "get_influence_matrix"):
        matrix = np.asarray(runner.get_influence_matrix(), dtype=np.float64)
        expected = (int(n_agents), int(n_agents))
        if matrix.shape != expected:
            raise RuntimeError(
                f"runner influence matrix has shape {matrix.shape}; expected {expected}"
            )
        return matrix.copy() if ego_ids is None else matrix[np.asarray(ego_ids, dtype=int)].copy()
    if hasattr(runner, "belief_modules"):
        return w_matrix(runner, n_agents)
    return None


def _runner_direction_matrix(runner, n_agents, ego_ids=None):
    getter = getattr(runner, "get_direction_matrix", None)
    if not callable(getter):
        return None
    matrix = np.asarray(getter(), dtype=np.float64)
    expected = (int(n_agents), int(n_agents))
    if matrix.shape != expected:
        raise RuntimeError(
            f"runner direction matrix has shape {matrix.shape}; expected {expected}"
        )
    return matrix.copy() if ego_ids is None else matrix[np.asarray(ego_ids, dtype=int)].copy()


def _oracle_capacity_core_f1(runner, oracle_core_by_ego, ego_ids):
    """Score the learned core against all-action capacity C*, not role/Phi."""
    beliefs = getattr(runner, "belief_modules", None)
    if beliefs is None:
        return float("nan")
    scores = []
    for ego in ego_ids:
        truth = oracle_core_by_ego.get(int(ego), [])
        if not truth or int(ego) not in beliefs:
            continue
        predicted = set(beliefs[int(ego)].get_core_set())
        truth = set(int(agent) for agent in truth)
        denominator = len(predicted) + len(truth)
        scores.append(
            2.0 * len(predicted & truth) / denominator if denominator else 0.0
        )
    return _mean_finite(scores)


def _agents_with_roles(env, roles):
    """Resolve a protocol-specified public role subset without oracle labels."""
    wanted = {str(role) for role in roles}
    table = getattr(env, "agent_role", {})
    selected = []
    for agent in range(int(env.n_agents)):
        try:
            if str(table[agent]) in wanted:
                selected.append(agent)
        except (KeyError, IndexError, TypeError):
            continue
    if not selected:
        raise RuntimeError(f"H2 role subset {sorted(wanted)} resolved to no agents")
    return selected


_FROZEN_COMPONENT_PATHS = (
    "policy_value", "actor", "critic", "backbone", "periph_module",
    "belief_summary_builder", "single_periph_proj", "heads",
    "pair_rel_module.full_encoder", "pair_rel_module.shadow_encoder",
    "pair_rel_module.shadow_to_full", "pair_rel_module.bc_head",
)


def _resolve_component(root, path):
    value = root
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _frozen_representation_digest(runner):
    """Hash only policy and downstream representation parameters."""
    digest = hashlib.sha256()
    found = 0
    for path in _FROZEN_COMPONENT_PATHS:
        module = _resolve_component(runner, path)
        if module is None or not callable(getattr(module, "state_dict", None)):
            continue
        found += 1
        digest.update(path.encode("utf-8"))
        # The scientific freeze contract concerns trainable parameters.
        # Diagnostic EMA buffers (for example slot-usage counters) may update
        # during read-only forward passes and must not be mistaken for learning.
        for key, value in sorted(module.named_parameters()):
            digest.update(key.encode("utf-8"))
            digest.update(np.asarray(value.detach().cpu()).tobytes())
    if found == 0:
        raise RuntimeError("H2 found no policy or representation parameters to freeze")
    return digest.hexdigest()


def _capture_frozen_learning_checkpoint(runner):
    """Capture the scientific branch point, not only neural weights."""
    state = {}
    for path in _FROZEN_COMPONENT_PATHS:
        module = _resolve_component(runner, path)
        if module is not None and callable(getattr(module, "state_dict", None)):
            state[path] = copy.deepcopy(module.state_dict())
    if not state:
        raise RuntimeError("H2 could not capture any policy/representation modules")
    digest = hashlib.sha256()
    for path in sorted(state):
        digest.update(path.encode("utf-8"))
        for key, value in sorted(state[path].items()):
            digest.update(key.encode("utf-8"))
            digest.update(np.asarray(value.detach().cpu()).tobytes())
    proxy = getattr(runner, "proxy", None)
    proxy_state = None
    if proxy is not None:
        if bool(getattr(proxy, "use_vmap_ensemble", False)):
            proxy_state = {
                "kind": "vmap",
                "params": {
                    key: value.detach().clone()
                    for key, value in proxy._stacked_params.items()
                },
                "buffers": {
                    key: value.detach().clone()
                    for key, value in proxy._stacked_buffers.items()
                },
            }
        elif hasattr(proxy, "models"):
            proxy_state = {
                "kind": "models",
                "models": [copy.deepcopy(model.state_dict()) for model in proxy.models],
            }
        if proxy_state is not None:
            digest.update(b"proxy")
            for group in ("params", "buffers"):
                for key, value in sorted(proxy_state.get(group, {}).items()):
                    digest.update(key.encode("utf-8"))
                    digest.update(np.asarray(value.detach().cpu()).tobytes())
            for model_state in proxy_state.get("models", []):
                for key, value in sorted(model_state.items()):
                    digest.update(key.encode("utf-8"))
                    digest.update(np.asarray(value.detach().cpu()).tobytes())
    optimizer_state = {}
    for path in ("policy_optim", "heads_optim", "pair_rel_module.optim"):
        optimizer = _resolve_component(runner, path)
        if optimizer is not None:
            optimizer_state[path] = copy.deepcopy(optimizer.state_dict())
    if proxy is not None:
        if hasattr(proxy, "optim"):
            optimizer_state["proxy.optim"] = copy.deepcopy(proxy.optim.state_dict())
        elif hasattr(proxy, "optims"):
            optimizer_state["proxy.optims"] = [
                copy.deepcopy(optim.state_dict()) for optim in proxy.optims
            ]

    runtime_state = {}
    # These counters index causal replay/profile labels.  Restoring only
    # tensors while leaving them advanced would make matched H2 arms differ
    # in target age/version even with identical model state.
    runtime_state["runner_causal_clock"] = {
        "interaction_step": int(getattr(runner, "_interaction_step", 0)),
        "policy_version": int(getattr(runner, "_policy_version", 0)),
        "episodes_completed": int(getattr(runner, "episodes_completed", 0)),
        "profile_update_step": copy.deepcopy(
            getattr(runner, "_profile_update_step", {})
        ),
    }
    for name in (
        "belief_modules", "sig_tracker", "scheduler", "drift", "matdet", "recip"
    ):
        if hasattr(runner, name):
            runtime_state[name] = copy.deepcopy(getattr(runner, name))
    pair_module = getattr(runner, "pair_rel_module", None)
    if pair_module is not None:
        runtime_state["pair_full_states"] = {
            key: value.detach().clone() for key, value in pair_module.full_states.items()
        }
        runtime_state["pair_shadow_states"] = {
            key: value.detach().clone() for key, value in pair_module.shadow_states.items()
        }
        runtime_state["pair_active_core_pairs"] = set(pair_module.active_core_pairs)
        runtime_state["pair_pooled_states"] = {
            key: value.detach().clone() for key, value in pair_module.pooled_states.items()
        }
        runtime_state["pair_state_mode"] = str(pair_module.state_mode)
        runtime_state["pair_bc_buffer"] = copy.deepcopy(pair_module.bc_buffer)
        runtime_state["pair_cd_norm_mean"] = pair_module.cd_norm_mean.copy()
        runtime_state["pair_cd_norm_std"] = pair_module.cd_norm_std.copy()
        runtime_state["pair_cd_normalization_frozen"] = bool(
            pair_module.cd_normalization_frozen
        )
    if proxy is not None:
        runtime_state["proxy_buffer"] = copy.deepcopy(proxy.buffer)
        runtime_state["proxy_counters"] = {
            "n_interventional_samples": int(proxy.n_interventional_samples),
            "total_dr_applied_calls": int(proxy.total_dr_applied_calls),
            "total_dr_applied_rows": int(proxy.total_dr_applied_rows),
            "total_dr_clipped_rows": int(proxy.total_dr_clipped_rows),
        }
    forcer = getattr(runner, "forcer", None)
    if forcer is not None and callable(getattr(forcer, "state_dict", None)):
        runtime_state["forcer_state"] = copy.deepcopy(forcer.state_dict())
    env_state = (
        copy.deepcopy(runner.env.clone_state())
        if callable(getattr(runner.env, "clone_state", None))
        else None
    )
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }
    digest.update(
        pickle.dumps(
            {
                "optimizer_state": optimizer_state,
                "runtime_state": runtime_state,
                "env_state": env_state,
                "rng_state": rng_state,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )
    return {
        "state": state,
        "proxy_state": proxy_state,
        "optimizer_state": optimizer_state,
        "runtime_state": runtime_state,
        "env_state": env_state,
        "rng_state": rng_state,
        "sha256": digest.hexdigest(),
    }


def _restore_frozen_learning_checkpoint(runner, checkpoint):
    for path, state in checkpoint["state"].items():
        module = _resolve_component(runner, path)
        if module is None:
            raise RuntimeError(f"H2 checkpoint component is missing: {path}")
        module.load_state_dict(state)
    proxy_state = checkpoint.get("proxy_state")
    proxy = getattr(runner, "proxy", None)
    if proxy_state is not None and proxy is None:
        raise RuntimeError("H2 checkpoint has a proxy but the runner does not")
    if proxy_state is not None and proxy_state["kind"] == "vmap":
        with torch.no_grad():
            for key, value in proxy_state["params"].items():
                proxy._stacked_params[key].copy_(value)
            for key, value in proxy_state["buffers"].items():
                proxy._stacked_buffers[key].copy_(value)
    elif proxy_state is not None and proxy_state["kind"] == "models":
        if len(proxy.models) != len(proxy_state["models"]):
            raise RuntimeError("H2 proxy ensemble cardinality differs from checkpoint")
        for model, state in zip(proxy.models, proxy_state["models"]):
            model.load_state_dict(state)
    elif proxy_state is not None:
        raise RuntimeError(f"unknown H2 proxy checkpoint kind: {proxy_state['kind']}")

    for path, state in checkpoint.get("optimizer_state", {}).items():
        if path == "proxy.optims":
            for optim, optim_state in zip(proxy.optims, state):
                optim.load_state_dict(optim_state)
            continue
        optimizer = _resolve_component(runner, path)
        if optimizer is None:
            raise RuntimeError(f"H2 optimizer component is missing: {path}")
        optimizer.load_state_dict(state)

    runtime = checkpoint.get("runtime_state", {})
    causal_clock = runtime.get("runner_causal_clock")
    if causal_clock is not None:
        runner._interaction_step = int(causal_clock["interaction_step"])
        runner._policy_version = int(causal_clock["policy_version"])
        runner.episodes_completed = int(causal_clock["episodes_completed"])
        runner._profile_update_step = copy.deepcopy(
            causal_clock["profile_update_step"]
        )
    for name in (
        "belief_modules", "sig_tracker", "scheduler", "drift", "matdet", "recip"
    ):
        if name in runtime:
            setattr(runner, name, copy.deepcopy(runtime[name]))
    pair_module = getattr(runner, "pair_rel_module", None)
    if pair_module is not None and "pair_full_states" in runtime:
        pair_module.full_states = {
            key: value.detach().clone().to(runner.device)
            for key, value in runtime["pair_full_states"].items()
        }
        pair_module.shadow_states = {
            key: value.detach().clone().to(runner.device)
            for key, value in runtime["pair_shadow_states"].items()
        }
        pair_module.active_core_pairs = set(
            runtime.get("pair_active_core_pairs", pair_module.full_states.keys())
        )
        pair_module.pooled_states = {
            key: value.detach().clone().to(runner.device)
            for key, value in runtime.get("pair_pooled_states", {}).items()
        }
        pair_module.state_mode = str(runtime.get("pair_state_mode", pair_module.state_mode))
        if pair_module.state_mode != "pooled":
            full_keys = set(pair_module.full_states)
            if full_keys != set(pair_module.active_core_pairs):
                raise RuntimeError(
                    "checkpoint pair allocation mismatch: full-state keys and "
                    "active-core pairs must be identical"
                )
        pair_module.bc_buffer = copy.deepcopy(runtime["pair_bc_buffer"])
        pair_module.cd_norm_mean = runtime["pair_cd_norm_mean"].copy()
        pair_module.cd_norm_std = runtime["pair_cd_norm_std"].copy()
        pair_module.cd_normalization_frozen = bool(
            runtime["pair_cd_normalization_frozen"]
        )
    if proxy is not None and "proxy_buffer" in runtime:
        proxy.buffer = copy.deepcopy(runtime["proxy_buffer"])
        for key, value in runtime.get("proxy_counters", {}).items():
            setattr(proxy, key, int(value))
    forcer = getattr(runner, "forcer", None)
    if forcer is not None and "forcer_state" in runtime:
        forcer.load_state_dict(copy.deepcopy(runtime["forcer_state"]))
    if checkpoint.get("env_state") is not None:
        runner.env.restore_state(copy.deepcopy(checkpoint["env_state"]))
    rng = checkpoint.get("rng_state")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])


def _configure_h2_monitoring_cfg(cfg):
    """Apply the seed-independent H2 arm settings used by null calibration."""
    cfg["behavioral_adapter_lambda"] = 0.0
    cfg["behavioral_adapter_only_in_behavioral_drift"] = False
    cfg["behavioral_adapter_target_roles"] = list(H2_MANIPULATED_NEIGHBOR_ROLES)
    cfg["freeze_policy_learning"] = True
    cfg["freeze_representation_state"] = True
    return cfg


def _h2_cusum_reference_hash(cfg, pretrain_episodes, max_steps=30):
    contract = build_h2_cusum_contract(
        cfg, n_agents=24, max_steps=int(max_steps),
        pretrain_episodes=int(pretrain_episodes),
        evaluation_roles=H2_EVALUATION_EGO_ROLES,
        manipulated_roles=H2_MANIPULATED_NEIGHBOR_ROLES,
    )
    return contract_hash(contract)


def _apply_cusum_calibration(
    cfg, calibration, *, expected_reference_config_hash=None
):
    if calibration is None:
        return
    validate_calibration_artifact(
        calibration,
        expected_reference_config_hash=expected_reference_config_hash,
    )
    cfg["drift_cusum_allowance"] = float(calibration["cusum_allowance"])
    cfg["drift_cusum_threshold"] = float(calibration["cusum_threshold"])
    # Compatibility for components that have not yet consumed the explicit
    # CUSUM configuration name.
    cfg["z_threshold"] = float(calibration["cusum_threshold"])
    cfg["cusum_false_alarm_target"] = float(calibration["target_false_alarm_rate"])
    cfg["cusum_monitoring_horizon"] = int(calibration["monitoring_horizon"])
    cfg["cusum_calibration_reference_config_hash"] = str(
        calibration["reference_config_hash"]
    )


def _pretrain_common_checkpoint(model, seed, episodes, device, cusum_calibration=None):
    """Train a neutral common policy checkpoint before both H2 arms."""
    RE.set_global_seed(seed)
    cfg = RE.default_cfg()
    cfg["seed"] = int(seed)
    _apply_cusum_calibration(cfg, cusum_calibration)
    _apply_tracker_control(model, cfg)
    cfg["behavioral_adapter_lambda"] = 0.0
    cfg["freeze_policy_learning"] = False
    # Hold the environment in its initial regime during shared pretraining.
    env = RE.make_main_env(
        task_mode="behavioral_drift",
        n_agents=24,
        max_steps=30,
        phase_length=max(100000, int(episodes) + 1),
        seed=seed,
    )
    runner = RE.make_runner(model, env, cfg, device)
    runner.run(n_episodes=int(episodes), eval_every=max(1, int(episodes)))
    drift = getattr(runner, "drift", None)
    proxy = getattr(runner, "proxy", None)
    if drift is not None:
        if proxy is None:
            raise RuntimeError("H2 drift witness requires proxy replay")
        drift.prepare_for_monitoring(
            proxy.buffer,
            episode=int(episodes),
            reference_batches=max(20, int(getattr(drift, "window", 20))),
        )
        if not drift.is_monitoring_ready():
            raise RuntimeError("H2 drift witness is not monitoring-ready")
    if hasattr(runner, "pair_rel_module"):
        runner.pair_rel_module.fit_cd_normalization(min_samples=1)
    checkpoint = _capture_frozen_learning_checkpoint(runner)
    return {
        **checkpoint,
        "episodes": int(episodes),
        "model": str(model),
        "seed": int(seed),
    }


def _recovery_statistics(
    eval_records,
    shift_episodes,
    trigger_episodes,
    causal_horizon_steps,
    max_steps,
    eval_every,
    valid_f1_floor=RECOVERY_F1_VALID_FLOOR,
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
            float(row.get("oracle_capacity_f1", float("nan")))
            for row in eval_records
            if int(row["episode"]) < shift_ep
            and np.isfinite(row.get("oracle_capacity_f1", float("nan")))
        ]
        baseline = float(np.mean(pre[-3:])) if pre else float("nan")
        baseline_valid = bool(
            np.isfinite(baseline) and baseline >= float(valid_f1_floor)
        )
        target = (
            max(0.9 * baseline, float(valid_f1_floor))
            if baseline_valid
            else float("nan")
        )

        post = [
            row
            for row in eval_records
            if int(row["episode"]) >= shift_ep
            and (next_shift is None or int(row["episode"]) < next_shift)
            and np.isfinite(row.get("oracle_capacity_f1", float("nan")))
        ]
        recovered_ep = None
        if np.isfinite(target):
            for row in post:
                if float(row["oracle_capacity_f1"]) >= target:
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
            "pre_shift_valid": int(baseline_valid),
            "invalid_reason": "" if baseline_valid else "invalid_pre_shift_structure",
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
        "recovery_f1_valid_floor": float(valid_f1_floor),
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


def run_one(
    model, mode, seed, episodes, eval_every, device, out_root, run_id,
    frozen_checkpoint=None,
    cached_pre_estimand_panel=None,
    cached_estimand_panel=None,
    cusum_calibration=None,
):
    out_dir = os.path.join(out_root, f"{model}_{mode}_seed{seed}")
    os.mkdir(out_dir)
    jsonl = os.path.join(out_dir, "eval.jsonl")
    with open(jsonl, "x", encoding="utf-8"):
        pass

    RE.set_global_seed(seed)
    cfg = RE.default_cfg()
    cfg["seed"] = seed
    # The four factorial arms are confirmatory Paper-A tracking evidence.
    # Set this before runner construction so the runner rejects legacy
    # profiles, uniform-memory compatibility mixing, capped forcing, and an
    # unproven CUSUM threshold.
    cfg["confirmatory"] = True
    cfg["strict_causal_profile"] = True
    cfg["semantic_router_frozen"] = True
    _apply_tracker_control(model, cfg)
    _configure_h2_monitoring_cfg(cfg)
    expected_cusum_hash = _h2_cusum_reference_hash(
        cfg, int(frozen_checkpoint.get("episodes", 0)) if frozen_checkpoint else 0,
        max_steps=30,
    )
    _apply_cusum_calibration(
        cfg, cusum_calibration,
        expected_reference_config_hash=(
            expected_cusum_hash if model == "Final-CIGAMF" else None
        ),
    )
    if mode not in FACTORIAL_CELLS:
        raise ValueError(f"Unknown H2 factorial cell: {mode!r}")
    structural_factor, behavioral_factor = FACTORIAL_CELLS[mode]
    # The old behavioural arm changed only environment metadata while every
    # learned runner continued sampling its own policy.  This intervention is
    # the executed policy: pi_tilde=(1-lambda)pi+lambda*pi_scripted.  It is
    # disabled in the structural arm so the two perturbations remain separate.
    # The adapter is activated only at the common exogenous intervention;
    # all four branches are identical before that point.
    make_args(seed=seed, device=device)  # Validate the shared CLI defaults.

    max_steps = 30
    phase_length = 40
    env = RE.make_main_env(
        task_mode=mode,
        n_agents=24,
        max_steps=max_steps,
        phase_length=phase_length,
        seed=seed,
        structural_factor=False,
        behavioral_factor=False,
    )
    runner = RE.make_runner(model, env, cfg, device)
    if frozen_checkpoint is None:
        raise RuntimeError("H2 requires a common pretraining checkpoint")
    _restore_frozen_learning_checkpoint(runner, frozen_checkpoint)
    if hasattr(runner, "drift") and not runner.drift.is_monitoring_ready():
        raise RuntimeError(
            f"{model}/{mode}/seed{seed}: drift witness is not monitoring-ready"
        )
    frozen_representation_sha256_before = _frozen_representation_digest(runner)

    env.set_behaviour_override("cooperative")
    intervention_after_episodes = int(
        min(40, max(0, int(episodes) // 2))
    )
    intervention_episode = intervention_after_episodes + 1
    pre_estimand_panel = copy.deepcopy(cached_pre_estimand_panel)
    if pre_estimand_panel is None:
        pre_estimand_panel = _fixed_estimand_panel(
            env,
            structural_factor=False,
            behavioral_factor=False,
            seed=seed,
            horizon=int(cfg["causal_horizon"]),
            discount=float(cfg["discount"]),
            core_budget=int(cfg.get("max_core_size", 3)),
        )
    estimand_panel = copy.deepcopy(cached_estimand_panel)
    if estimand_panel is None:
        estimand_panel = _fixed_estimand_panel(
            env,
            structural_factor=structural_factor,
            behavioral_factor=behavioral_factor,
            seed=seed,
            horizon=int(cfg["causal_horizon"]),
            discount=float(cfg["discount"]),
            core_budget=int(cfg.get("max_core_size", 3)),
        )

    evaluation_egos = _agents_with_roles(env, H2_EVALUATION_EGO_ROLES)
    manipulated_agents = _agents_with_roles(env, H2_MANIPULATED_NEIGHBOR_ROLES)

    causal_horizon = int(cfg.get("causal_horizon", 8))
    previous_w = _runner_influence_matrix(runner, env.n_agents, evaluation_egos)
    has_influence_matrix = previous_w is not None
    previous_d = _runner_direction_matrix(runner, env.n_agents, evaluation_egos)
    has_direction_matrix = previous_d is not None
    previous_phi = _phi_fingerprint(env)
    deltas = []
    direction_deltas = []
    previous_delta_episode = 0
    eval_records = []
    shift_episodes = []
    behavioral_shift_episodes = []
    trigger_episodes = []
    event_cursor = 0
    intervention_scheduled = False

    while int(getattr(runner, "episodes_completed", 0)) < episodes:
        completed_before = int(getattr(runner, "episodes_completed", 0))
        if (
            not intervention_scheduled
            and completed_before == intervention_after_episodes
        ):
            env.schedule_factorial_intervention(
                structural=structural_factor,
                behavioral=behavioral_factor,
                behavior_mode="selfish",
            )
            runner.cfg["behavioral_adapter_lambda"] = (
                1.0 if behavioral_factor else 0.0
            )
            intervention_scheduled = True
        if completed_before < intervention_after_episodes:
            chunk_size = min(
                eval_every,
                intervention_after_episodes - completed_before,
                episodes - completed_before,
            )
        else:
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
        adapter_rows = [
            event for event in new_events
            if int(event.get("behavioral_adapter_active", 0))
        ]
        adapter_kl = _mean_finite([
            event.get("behavioral_adapter_kl", float("nan"))
            for event in adapter_rows
        ])
        adapter_tv = _mean_finite([
            event.get("behavioral_adapter_tv", float("nan"))
            for event in adapter_rows
        ])
        adapter_action_freq_tv = _mean_finite([
            event.get("behavioral_adapter_action_freq_tv", float("nan"))
            for event in adapter_rows
        ])
        adapter_target_tv = _mean_finite([
            event.get("behavioral_adapter_target_tv", float("nan"))
            for event in adapter_rows
        ])
        adapter_non_target_tv = _mean_finite([
            event.get("behavioral_adapter_non_target_tv", float("nan"))
            for event in adapter_rows
        ])
        adapter_target_count = max(
            [int(event.get("behavioral_adapter_target_count", 0)) for event in adapter_rows]
            or [0]
        )

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
            current_w = _runner_influence_matrix(runner, env.n_agents, evaluation_egos)
            delta = delta_norm(previous_w, current_w)
            previous_w = current_w

        direction_delta = float("nan")
        if has_direction_matrix:
            current_d = _runner_direction_matrix(runner, env.n_agents, evaluation_egos)
            direction_delta = delta_norm(previous_d, current_d)
            previous_d = current_d

        history = getattr(runner, "history", {})
        oracle_core_panel = (
            estimand_panel if completed >= intervention_episode
            else pre_estimand_panel
        )
        oracle_capacity_f1 = _oracle_capacity_core_f1(
            runner,
            oracle_core_panel.get("oracle_core_by_ego", {}),
            evaluation_egos,
        )
        row = {
            "run_id": run_id,
            "episode": completed,
            "delta": delta,
            "direction_delta": direction_delta,
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
            "n_behavioral_adapter_events": len(adapter_rows),
            "behavioral_adapter_kl": adapter_kl,
            "behavioral_adapter_tv": adapter_tv,
            "behavioral_adapter_action_freq_tv": adapter_action_freq_tv,
            "behavioral_adapter_target_tv": adapter_target_tv,
            "behavioral_adapter_non_target_tv": adapter_non_target_tv,
            "behavioral_adapter_target_count": adapter_target_count,
            "f1": last(history, "mean_f1"),
            "oracle_capacity_f1": oracle_capacity_f1,
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
        direction_deltas.append((previous_delta_episode, completed, direction_delta))
        previous_delta_episode = completed
        print(
            f"[H2 {model}/{mode} s{seed}] ep={completed} delta={delta:.4f} "
            f"C*f1={row['oracle_capacity_f1']:.3f} structural={chunk_shifts} "
            f"behavioral={chunk_behavioral_shifts} triggers={chunk_triggers}"
        )

    events = list(getattr(runner, "episode_events", []))
    if events:
        _validate_episode_events(events, episodes)
    if not eval_records or int(eval_records[-1]["episode"]) != episodes:
        raise RuntimeError(f"{model}/{mode}/seed{seed}: missing terminal evaluation")
    frozen_representation_sha256_after = _frozen_representation_digest(runner)
    if frozen_representation_sha256_after != frozen_representation_sha256_before:
        raise RuntimeError(
            f"{model}/{mode}/seed{seed}: frozen policy/representation state changed"
        )

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
    finite_direction_rows = [
        (start_ep, end_ep, delta)
        for start_ep, end_ep, delta in direction_deltas
        if np.isfinite(delta)
    ]
    direction_values = np.asarray(
        [delta for _, _, delta in finite_direction_rows], dtype=np.float64
    )
    common_mask, common_windows = _matched_change_interval_mask(
        finite_rows, [intervention_episode], CHANGE_WINDOW_EVAL_INTERVALS
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
    # Direction D is a separate behavioural object.  Use the same interval
    # masks only when C and D have identical complete evaluation coverage.
    direction_struct = (
        float(np.mean(direction_values[structural_mask]))
        if len(direction_values) == len(delta_values) and structural_mask.any()
        else float("nan")
    )
    direction_behav = (
        float(np.mean(direction_values[behavioral_mask]))
        if len(direction_values) == len(delta_values) and behavioral_mask.any()
        else float("nan")
    )
    analysis_mask = common_mask
    capacity_factorial_outcome = (
        float(np.mean(delta_values[analysis_mask]))
        if analysis_mask.any()
        else float("nan")
    )
    direction_factorial_outcome = (
        float(np.mean(direction_values[analysis_mask]))
        if len(direction_values) == len(delta_values) and analysis_mask.any()
        else float("nan")
    )
    # This is a within-run background-drift diagnostic.  It is deliberately
    # not the paper's selectivity ratio: the latter compares the same C
    # estimator across the matched structural and behavioural interventions.
    background_drift_ratio = (
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
    false_alarm_windows = _false_alarm_window_stats(
        events,
        start_episode=(intervention_episode if behavioral_factor and not structural_factor else 1),
        monitoring_horizon=int(cfg.get("cusum_monitoring_horizon", 0)),
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
        "structural_factor": int(structural_factor),
        "behavioral_factor": int(behavioral_factor),
        "seed": seed,
        "episodes": episodes,
        "pretrain_episodes": int(frozen_checkpoint["episodes"]),
        "frozen_checkpoint_sha256": str(frozen_checkpoint["sha256"]),
        "policy_learning_frozen": True,
        "representation_learning_frozen": True,
        "frozen_representation_sha256_before": frozen_representation_sha256_before,
        "frozen_representation_sha256_after": frozen_representation_sha256_after,
        "evaluation_ego_roles": list(H2_EVALUATION_EGO_ROLES),
        "evaluation_ego_count": len(evaluation_egos),
        "manipulated_neighbor_roles": list(H2_MANIPULATED_NEIGHBOR_ROLES),
        "manipulated_neighbor_count": len(manipulated_agents),
        "episodes_completed": int(getattr(runner, "episodes_completed", 0)),
        "eval_every": eval_every,
        "n_eval_points": len(eval_records),
        "expected_eval_points": int(np.ceil(episodes / eval_every)),
        "metric_applicable": metric_applicable,
        "not_applicable_reason": "" if metric_applicable else "runner_has_no_influence_matrix_W",
        "delta_mean_struct": delta_struct,
        "delta_mean_behav": delta_behav,
        "delta_mean_background": delta_background,
        "direction_metric_applicable": bool(has_direction_matrix),
        "direction_mean_struct": direction_struct,
        "direction_mean_behav": direction_behav,
        "capacity_factorial_outcome": capacity_factorial_outcome,
        "direction_factorial_outcome": direction_factorial_outcome,
        "intervention_episode": int(intervention_episode),
        "common_response_windows": common_windows,
        "n_complete_common_windows": sum(
            int(window["complete"]) for window in common_windows
        ),
        "estimand_panel": estimand_panel,
        "pre_intervention_estimand_panel": pre_estimand_panel,
        "recovery_ground_truth": "fixed_state_bank_all_action_capacity_C_star",
        "estimand_capacity_mean": estimand_panel["capacity_mean"],
        "estimand_direction_abs_mean": estimand_panel["direction_abs_mean"],
        "background_drift_ratio": background_drift_ratio,
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
        "behavioral_adapter_required": int(behavioral_factor),
        "behavioral_adapter_event_count": int(sum(
            int(row["n_behavioral_adapter_events"]) for row in eval_records
        )),
        "behavioral_adapter_kl": _mean_finite([
            row["behavioral_adapter_kl"] for row in eval_records
        ]),
        "behavioral_adapter_tv": _mean_finite([
            row["behavioral_adapter_tv"] for row in eval_records
        ]),
        "behavioral_adapter_action_freq_tv": _mean_finite([
            row["behavioral_adapter_action_freq_tv"] for row in eval_records
        ]),
        "behavioral_adapter_target_tv": _mean_finite([
            row["behavioral_adapter_target_tv"] for row in eval_records
        ]),
        "behavioral_adapter_non_target_tv": _mean_finite([
            row["behavioral_adapter_non_target_tv"] for row in eval_records
        ]),
        "behavioral_adapter_target_count": max(
            [int(row["behavioral_adapter_target_count"]) for row in eval_records] or [0]
        ),
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
        "cusum_false_alarm_target": float(
            cfg.get("cusum_false_alarm_target", float("nan"))
        ),
        "cusum_monitoring_horizon": int(cfg.get("cusum_monitoring_horizon", 0)),
        "cusum_calibration_reference_config_hash": str(
            cfg.get("cusum_calibration_reference_config_hash", "")
        ),
        **false_alarm_windows,
        "cusum_allowance": float(cfg.get("drift_cusum_allowance", float("nan"))),
        "cusum_threshold": float(cfg.get(
            "drift_cusum_threshold", cfg.get("z_threshold", float("nan"))
        )),
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


def _cached_estimand_panels(seed, run_dir):
    """Compute model-independent H2 oracle panels once per seed.

    The cloned-state all-action panel depends on the environment seed and the
    factorial cell, not on a learned model. Reusing the serialized panel keeps
    every comparator on exactly the same oracle states and removes repeated
    rollout work from the model loop.
    """
    cfg = RE.default_cfg()
    cache_path = os.path.join(run_dir, f"oracle_panels_seed{int(seed)}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if (
            payload.get("protocol_version") == PROTOCOL_VERSION
            and int(payload.get("seed", -1)) == int(seed)
            and int(payload.get("horizon", -1)) == int(cfg["causal_horizon"])
            and np.isclose(float(payload.get("discount", -1.0)), float(cfg["discount"]))
            and int(payload.get("core_budget", -1)) == int(cfg.get("max_core_size", 3))
        ):
            def _restore_integer_keys(panel):
                panel = copy.deepcopy(panel)
                panel["capacity_mean_by_ego"] = {
                    int(ego): {
                        int(target): float(value)
                        for target, value in targets.items()
                    }
                    for ego, targets in panel.get(
                        "capacity_mean_by_ego", {}
                    ).items()
                }
                panel["oracle_core_by_ego"] = {
                    int(ego): [int(target) for target in targets]
                    for ego, targets in panel.get(
                        "oracle_core_by_ego", {}
                    ).items()
                }
                return panel

            return (
                _restore_integer_keys(payload["pre"]),
                {
                    mode: _restore_integer_keys(panel)
                    for mode, panel in payload["cells"].items()
                },
            )

    env = RE.make_main_env(
        task_mode="S0B0",
        n_agents=24,
        max_steps=30,
        phase_length=40,
        seed=int(seed),
        structural_factor=False,
        behavioral_factor=False,
    )
    common = dict(
        env=env,
        seed=int(seed),
        horizon=int(cfg["causal_horizon"]),
        discount=float(cfg["discount"]),
        core_budget=int(cfg.get("max_core_size", 3)),
    )
    pre = _fixed_estimand_panel(
        structural_factor=False, behavioral_factor=False, **common
    )
    cells = {"S0B0": copy.deepcopy(pre)}
    cells.update({
        mode: _fixed_estimand_panel(
            structural_factor=structural,
            behavioral_factor=behavioral,
            **common,
        )
        for mode, (structural, behavioral) in FACTORIAL_CELLS.items()
        if mode != "S0B0"
    })
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "horizon": int(cfg["causal_horizon"]),
        "discount": float(cfg["discount"]),
        "core_budget": int(cfg.get("max_core_size", 3)),
        "pre": pre,
        "cells": cells,
    }
    _atomic_write_json(cache_path, payload)
    return pre, cells


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument(
        "--pretrain-episodes", type=int, default=60,
        help="Neutral common-policy pretraining before both H2 arms.",
    )
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
        "--cusum-calibration",
        help="Frozen no-change Page-CUSUM calibration artifact.",
    )
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
    if args.pretrain_episodes <= 0:
        parser.error("--pretrain-episodes must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    cusum_calibration = None
    if args.cusum_calibration:
        with open(args.cusum_calibration, encoding="utf-8") as handle:
            cusum_calibration = json.load(handle)
        contract_cfg = RE.default_cfg()
        _configure_h2_monitoring_cfg(contract_cfg)
        expected_hash = _h2_cusum_reference_hash(
            contract_cfg, args.pretrain_episodes, max_steps=30
        )
        _apply_cusum_calibration(
            contract_cfg, cusum_calibration,
            expected_reference_config_hash=expected_hash,
        )
        overlap = set(int(seed) for seed in args.seeds).intersection(
            int(seed) for seed in cusum_calibration["development_seeds"]
        )
        if overlap:
            parser.error(
                "H2/H3 seeds overlap CUSUM no-change development seeds: "
                + ", ".join(str(seed) for seed in sorted(overlap))
            )

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
        "pretrain_episodes": args.pretrain_episodes,
        "policy_learning_frozen_during_arms": True,
        "cusum_calibration": None if cusum_calibration is None else {
            key: cusum_calibration[key] for key in (
                "calibration_protocol", "cusum_allowance", "cusum_threshold",
                "target_false_alarm_rate", "observed_false_alarm_rate",
                "monitoring_horizon", "development_seeds",
                "reference_config_hash",
            ) if key in cusum_calibration
        },
        "evaluation_ego_roles": list(H2_EVALUATION_EGO_ROLES),
        "manipulated_neighbor_roles": list(H2_MANIPULATED_NEIGHBOR_ROLES),
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
            pre_estimand_panel, estimand_panels = _cached_estimand_panels(
                seed=seed, run_dir=run_dir
            )
            for model in args.models:
                checkpoint = _pretrain_common_checkpoint(
                    model=model,
                    seed=seed,
                    episodes=args.pretrain_episodes,
                    device=args.device,
                    cusum_calibration=cusum_calibration,
                )
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
                        frozen_checkpoint=checkpoint,
                        cached_pre_estimand_panel=pre_estimand_panel,
                        cached_estimand_panel=estimand_panels[mode],
                        cusum_calibration=cusum_calibration,
                    )
                    attempt["completed_attempts"].append({
                        "model": model,
                        "mode": mode,
                        "seed": int(seed),
                    })
                    _atomic_write_json(latest_attempt_path, attempt)

                control = per_mode["S0B0"]
                behavioral = per_mode["S0B1"]
                structural = per_mode["S1B0"]
                combined = per_mode["S1B1"]
                y_c = {
                    cell: per_mode[cell]["capacity_factorial_outcome"]
                    for cell in FACTORIAL_CELLS
                }
                y_d = {
                    cell: per_mode[cell]["direction_factorial_outcome"]
                    for cell in FACTORIAL_CELLS
                }
                y_ce = {
                    cell: per_mode[cell]["estimand_capacity_mean"]
                    for cell in FACTORIAL_CELLS
                }
                y_de = {
                    cell: per_mode[cell]["estimand_direction_abs_mean"]
                    for cell in FACTORIAL_CELLS
                }
                capacity_beta_structural = 0.5 * (
                    (y_c["S1B0"] - y_c["S0B0"])
                    + (y_c["S1B1"] - y_c["S0B1"])
                )
                capacity_beta_behavioral = 0.5 * (
                    (y_c["S0B1"] - y_c["S0B0"])
                    + (y_c["S1B1"] - y_c["S1B0"])
                )
                capacity_beta_interaction = (
                    y_c["S1B1"] - y_c["S1B0"]
                    - y_c["S0B1"] + y_c["S0B0"]
                )
                direction_beta_structural = 0.5 * (
                    (y_d["S1B0"] - y_d["S0B0"])
                    + (y_d["S1B1"] - y_d["S0B1"])
                )
                direction_beta_behavioral = 0.5 * (
                    (y_d["S0B1"] - y_d["S0B0"])
                    + (y_d["S1B1"] - y_d["S1B0"])
                )
                direction_beta_interaction = (
                    y_d["S1B1"] - y_d["S1B0"]
                    - y_d["S0B1"] + y_d["S0B0"]
                )
                estimand_capacity_beta_structural = 0.5 * (
                    (y_ce["S1B0"] - y_ce["S0B0"])
                    + (y_ce["S1B1"] - y_ce["S0B1"])
                )
                estimand_capacity_beta_behavioral = 0.5 * (
                    (y_ce["S0B1"] - y_ce["S0B0"])
                    + (y_ce["S1B1"] - y_ce["S1B0"])
                )
                estimand_direction_beta_behavioral = 0.5 * (
                    (y_de["S0B1"] - y_de["S0B0"])
                    + (y_de["S1B1"] - y_de["S1B0"])
                )
                delta_struct = y_c["S1B0"] - y_c["S0B0"]
                delta_behav = y_c["S0B1"] - y_c["S0B0"]
                sr_cross = (
                    float(delta_struct / delta_behav)
                    if np.isfinite(delta_struct)
                    and np.isfinite(delta_behav)
                    and delta_behav > 1e-6
                    else float("nan")
                )
                claim_evaluable = bool(
                    all(per_mode[cell]["metric_applicable"] for cell in FACTORIAL_CELLS)
                    and structural["n_shift_events"] > 0
                    and behavioral["n_behavioral_shift_events"] > 0
                    and combined["n_shift_events"] > 0
                    and combined["n_behavioral_shift_events"] > 0
                    and control["n_shift_events"] == 0
                    and control["n_behavioral_shift_events"] == 0
                    and len({
                        int(per_mode[cell]["intervention_episode"])
                        for cell in FACTORIAL_CELLS
                    }) == 1
                    and all(
                        per_mode[cell]["n_complete_common_windows"] > 0
                        for cell in FACTORIAL_CELLS
                    )
                    and len({
                        json.dumps(
                            per_mode[cell]["common_response_windows"],
                            sort_keys=True,
                        )
                        for cell in FACTORIAL_CELLS
                    }) == 1
                    and np.isfinite(sr_cross)
                    and behavioral["behavioral_adapter_event_count"] > 0
                    and np.isfinite(behavioral["behavioral_adapter_kl"])
                    and np.isfinite(behavioral["behavioral_adapter_tv"])
                    and behavioral["behavioral_adapter_tv"] > 1e-6
                    and behavioral["behavioral_adapter_target_count"] > 0
                    and np.isfinite(behavioral["behavioral_adapter_target_tv"])
                    and behavioral["behavioral_adapter_target_tv"] > 1e-6
                    and np.isfinite(behavioral["behavioral_adapter_non_target_tv"])
                    and behavioral["behavioral_adapter_non_target_tv"] <= 1e-9
                    and all(np.isfinite(value) for value in y_ce.values())
                    and all(np.isfinite(value) for value in y_de.values())
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
                    "pretrain_episodes": checkpoint["episodes"],
                    "frozen_checkpoint_sha256": checkpoint["sha256"],
                    "policy_learning_frozen": 1,
                    "representation_learning_frozen": 1,
                    "frozen_representation_unchanged": int(all(
                        per_mode[cell]["frozen_representation_sha256_before"]
                        == per_mode[cell]["frozen_representation_sha256_after"]
                        for cell in FACTORIAL_CELLS
                    )),
                    "evaluation_ego_roles": ";".join(H2_EVALUATION_EGO_ROLES),
                    "evaluation_ego_count": structural["evaluation_ego_count"],
                    "manipulated_neighbor_roles": ";".join(H2_MANIPULATED_NEIGHBOR_ROLES),
                    "manipulated_neighbor_count": structural["manipulated_neighbor_count"],
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
                    "capacity_beta_structural": capacity_beta_structural,
                    "capacity_beta_behavioral": capacity_beta_behavioral,
                    "capacity_beta_interaction": capacity_beta_interaction,
                    "direction_beta_structural": direction_beta_structural,
                    "direction_beta_behavioral": direction_beta_behavioral,
                    "direction_beta_interaction": direction_beta_interaction,
                    "estimand_capacity_beta_structural": estimand_capacity_beta_structural,
                    "estimand_capacity_beta_behavioral": estimand_capacity_beta_behavioral,
                    "estimand_direction_beta_behavioral": estimand_direction_beta_behavioral,
                    "capacity_control_floor": y_c["S0B0"],
                    "capacity_s0b1": y_c["S0B1"],
                    "capacity_s1b0": y_c["S1B0"],
                    "capacity_s1b1": y_c["S1B1"],
                    "direction_s0b0": y_d["S0B0"],
                    "direction_s0b1": y_d["S0B1"],
                    "direction_s1b0": y_d["S1B0"],
                    "direction_s1b1": y_d["S1B1"],
                    "delta_background_structural_run": structural[
                        "delta_mean_background"
                    ],
                    "delta_background_behavioral_run": behavioral[
                        "delta_mean_background"
                    ],
                    "direction_struct": structural["direction_mean_struct"],
                    "direction_behav": behavioral["direction_mean_behav"],
                    "direction_manipulation_pass": int(
                        model in {"Final-CIGAMF", "NoTwoTimescale"}
                        and np.isfinite(behavioral["direction_mean_behav"])
                        and behavioral["direction_mean_behav"] > 1e-6
                    ),
                    # Paper A's primary selectivity endpoint:
                    # SR_C = mean(delta C | structural) /
                    #        mean(delta C | behavioural).
                    "SR_C": sr_cross,
                    # Compatibility names retain the two diagnostics without
                    # confusing either one with the paper endpoint.
                    "SR_cross_run_legacy": sr_cross,
                    "background_drift_ratio_structural": structural[
                        "background_drift_ratio"
                    ],
                    "recovery_latency": structural["recovery_latency_intervals"],
                    "recovery_latency_raw": structural["recovery_latency_raw_intervals"],
                    "recovery_ground_truth": structural["recovery_ground_truth"],
                    "trigger_delay_intervals": structural["trigger_delay_intervals"],
                    "n_shift_events": structural["n_shift_events"],
                    "n_behavioral_shift_events": behavioral[
                        "n_behavioral_shift_events"
                    ],
                    "behavioral_adapter_event_count": behavioral[
                        "behavioral_adapter_event_count"
                    ],
                    "behavioral_adapter_kl": behavioral["behavioral_adapter_kl"],
                    "behavioral_adapter_tv": behavioral["behavioral_adapter_tv"],
                    "behavioral_adapter_action_freq_tv": behavioral[
                        "behavioral_adapter_action_freq_tv"
                    ],
                    "behavioral_adapter_target_tv": behavioral[
                        "behavioral_adapter_target_tv"
                    ],
                    "behavioral_adapter_non_target_tv": behavioral[
                        "behavioral_adapter_non_target_tv"
                    ],
                    "behavioral_adapter_target_count": behavioral[
                        "behavioral_adapter_target_count"
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
                    "n_triggers_behavioral": behavioral["n_triggers"],
                    "n_triggers_control": control["n_triggers"],
                    "n_triggers_combined": combined["n_triggers"],
                    # Legacy per-episode quantity is retained only as a
                    # diagnostic.  The scientific false-alarm endpoint uses
                    # detector-ready windows matched to the calibration horizon.
                    "behavioral_false_trigger_rate_legacy_per_episode": float(
                        behavioral["n_triggers"] / max(1, args.episodes)
                    ),
                    "behavioral_false_alarm_window_rate": behavioral[
                        "false_alarm_window_rate"
                    ],
                    "behavioral_false_alarm_window_count": behavioral[
                        "false_alarm_window_count"
                    ],
                    "behavioral_monitoring_window_count": behavioral[
                        "monitoring_window_count"
                    ],
                    "cusum_monitoring_horizon": behavioral[
                        "cusum_monitoring_horizon"
                    ],
                    "cusum_calibration_reference_config_hash": behavioral[
                        "cusum_calibration_reference_config_hash"
                    ],
                    "cusum_false_alarm_target": structural["cusum_false_alarm_target"],
                    "cusum_allowance": structural["cusum_allowance"],
                    "cusum_threshold": structural["cusum_threshold"],
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
