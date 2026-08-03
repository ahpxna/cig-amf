import argparse
import gc
import importlib
import inspect
import os
import pkgutil
import random
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Runner imports
# ============================================================

# ============================================================
# Runner imports
# ============================================================

try:
    from runners.baseline_runner import (
        PureMeanFieldRunner,
        FullExplicitLocalRunner,
        NoBeliefRunner,
        NoMultiMemoryRunner,
        NoTwoTimescaleRunner,
    )
except ModuleNotFoundError:
    from runners.baseline_runner import (
        PureMeanFieldRunner,
        FullExplicitLocalRunner,
        NoBeliefRunner,
        NoMultiMemoryRunner,
        NoTwoTimescaleRunner,
    )

try:
    from runners.final_runner import FinalCIGAMFRunner
except ModuleNotFoundError:
    from runners.final_runner import FinalCIGAMFRunner


# ============================================================
# Utility imports
# ============================================================

try:
    from utils.io_utils import (
        ensure_dir,
        save_csv,
        save_history_csv,
        save_json,
        plot_histories,
    )
except ModuleNotFoundError:
    from utils.io_utils import (
        ensure_dir,
        save_csv,
        save_history_csv,
        save_json,
        plot_histories,
    )

try:
    from utils.metrics import (
        oracle_calibration,
        oracle_core_f1_from_scores,
        recovery_latency,
    )
except ModuleNotFoundError:
    from utils.metrics import (
        oracle_calibration,
        oracle_core_f1_from_scores,
        recovery_latency,
    )


# ============================================================
# Reproducibility / memory cleanup
# ============================================================

def set_global_seed(seed):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_runtime_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Config
# ============================================================

def default_cfg():
    return {
        "discount": 0.97,
        "causal_horizon": 8,

        "core_dim": 64,
        "periph_dim": 64,
        "belief_dim": 64,
        "policy_hidden": 160,

        "shadow_dim": 24,

        "num_memory_slots": 4,
        "periph_memory_dim": 32,

        "belief_top_k": 4,
        "belief_pooled_hidden": 32,

        "n_ensemble": 6,
        "proxy_lr": 5e-4,
        "proxy_buffer_size": 100000,
        "proxy_grad_clip": 1.0,
        "proxy_train_steps": 16,
        "proxy_batch_size": 256,
        "proxy_holdout_size": 128,

        "core_lr": 5e-4,
        "bc_buffer_size": 200000,
        "bc_grad_clip": 1.0,
        "bc_train_steps": 8,
        "bc_batch_size": 256,

        "policy_lr": 7e-4,

        "k0_warmup": 30,
        "slow_ratio": 0.15,
        "accel_factor": 4.0,
        "accel_duration": 8,
        # require_both/inflation_t_reset: xem TwoTimescaleScheduler, default
        # gốc trong training/scheduler.py là False/1.
        "require_both": False,
        "inflation_t_reset": 1,

        "belief_lambda_0": 0.08,
        "belief_uncertainty_scale": 2.0,
        "belief_tau": 0.005,
        "belief_tau_in": 0.55,
        "belief_tau_out": 0.35,
        "seed_core_top_k": 3,
        "min_core_size": 2,
        "max_core_size": 4,
        "sigma_floor": 0.08,
        # ---- belief_layer v2, default = giá trị khuyến nghị gốc ----
        "belief_core_rule": "lcb",
        "belief_kappa": 1.0,
        "belief_alpha_decay": 0.7,
        # [P-6 FINAL DEBUG] Bật Eq (17) — adaptive core budget theo entropy.
        # False = capacity allocation là hằng số => claim RQ3 (paper còn yêu
        # cầu báo cáo "fraction of updates at which k_i binds at k_max")
        # không thể kiểm chứng. Code đã có sẵn trong belief_layer.py:322-360.
        "belief_adaptive_k": True,
        "belief_adaptive_k_min": 1,
        "belief_signed_balance": 0.5,

        "periph_mu_floor": 0.008,
        "periph_beta_floor": 0.03,
        "periph_uniform_mix": 0.35,
        "periph_use_uniform_mix": True,
        "periph_lb_coeff": 0.5,
        "belief_priority_mu_floor": 0.01,
        "shadow_loss_weight": 0.25,
        "graph_score_steps": 8,

        # ---- structural_proxy v2, default = giá trị gốc trong
        # structural_proxy.py ----
        "proxy_n_horizons": 3,
        "proxy_effect_mode": "signed_aristocrat",
        "proxy_use_doubly_robust": True,
        "proxy_iw_clip": 10.0,
        "proxy_bootstrap_ratio": 0.8,
        "proxy_use_belief_input": False,
        "proxy_ensemble_dropout": 0.0,
        "seed": 0,

        # ---- final_runner.py: sig_tracker / forcer / heads / drift /
        # matdet / recip, default = giá trị gốc đang hard-code trong
        # runner (xem models/{influence_signature,intervention,
        # ego_conditioned_latent,drift_probe,reciprocity}.py) ----
        "sig_tracker_window": 30,

        "eps": 0.03,
        "forcer_max_forced_per_step": 2,
        "forcer_anneal_to": 0.01,
        "forcer_anneal_episodes": 60,

        "heads_lr": 5e-4,
        "heads_w_contrastive": 0.3,
        "heads_w_influence": 1.0,

        "drift_n_horizons": 3,
        "drift_warmup_batches": 200,
        "drift_recalibrate_after": 15,
        "drift_train_batches": 5,

        "matdet_window": 20,

        "recip_min_causal_samples": 20,
    }


def smoke_cfg():
    cfg = default_cfg()

    cfg.update(
        {
            "core_dim": 32,
            "periph_dim": 32,
            "belief_dim": 32,
            "policy_hidden": 96,

            "shadow_dim": 8,

            "periph_memory_dim": 24,
            "belief_pooled_hidden": 16,

            "n_ensemble": 2,
            "proxy_train_steps": 2,
            "proxy_batch_size": 64,
            "proxy_holdout_size": 32,

            "bc_train_steps": 2,
            "bc_batch_size": 64,

            "k0_warmup": 5,
            "slow_ratio": 0.25,
        }
    )

    return cfg


# ============================================================
# Environment resolution
# ============================================================

MAIN_ENV_CLASS_NAMES = [
    "AdaptiveResourceFlowArena",
    "AdaptiveResourceFlowArenaV3",
    "ResourceFlowArena",
    "AdaptiveResourceFlowEnv",
    "ResourceFlowEnv",
    "ResourceFlowWorld",
    "AdaptiveResourceFlowWorld",
]

TINY_ENV_CLASS_NAMES = [
    "TinyOracleResourceFlowV1",
    "TinyOracleResourceFlowEnv",
    "TinyOracleResourceFlowArena",
    "TinyResourceFlowOracle",
    "TinyResourceFlowEnv",
    "TinyOracleEnv",
]

MAIN_ENV_MODULE_CANDIDATES = [
    "envs.adaptive_resource_flow_arena_v3",
    "envs.adaptive_resource_flow_arena",
    "envs.resource_flow_arena",
    "envs.resource_flow",
    "envs.main_env",
    "envs.env",
]

TINY_ENV_MODULE_CANDIDATES = [
    "envs.tiny_oracle_resource_flow_v1",
    "envs.tiny_oracle_resource_flow",
    "envs.tiny_oracle",
    "envs.tiny_oracle_env",
    "envs.tiny_resource_flow",
    "envs.oracle_env",
]

MAIN_REQUIRED_METHODS = [
    "reset",
    "step",
    "clone_state",
    "restore_state",
    "get_obs_dim",
    "get_action_dim",
    "get_obs_of_ego",
]

TINY_REQUIRED_METHODS = [
    "reset",
    "step",
    "sample_state_bank",
    "restore_state",
    "clone_state",
    "get_supported_egos",
    "scripted_policy",
    "compute_oracle_influence_from_current_state",
    "rollout_from_current_state",
    "get_obs_of_ego",
    "get_obs_dim",
    "get_action_dim",
]

RUNNER_REQUIRED_INSTANCE_ATTRS = [
    "positions",
    "agent_zone",
    "agent_role",
    "last_actions",
]


def _project_root():
    return Path(__file__).resolve().parents[1]


def _safe_import_module(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _class_has_methods(cls, required_methods):
    for name in required_methods:
        if not hasattr(cls, name):
            return False
    return True


def _module_class_by_names(module, class_names):
    if module is None:
        return None

    for class_name in class_names:
        if hasattr(module, class_name):
            obj = getattr(module, class_name)

            if inspect.isclass(obj):
                return obj

    return None


def _scan_env_package(class_names, required_methods=None):
    required_methods = required_methods or []

    try:
        import envs as envs_pkg
    except Exception:
        return None

    package_path = getattr(envs_pkg, "__path__", None)

    if package_path is None:
        return None

    for _, modname, _ in pkgutil.walk_packages(package_path, prefix="envs."):
        module = _safe_import_module(modname)

        if module is None:
            continue

        cls = _module_class_by_names(module, class_names)

        if cls is not None:
            return cls

    for _, modname, _ in pkgutil.walk_packages(package_path, prefix="envs."):
        module = _safe_import_module(modname)

        if module is None:
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue

            if required_methods and _class_has_methods(obj, required_methods):
                return obj

    return None


def _resolve_main_env_class():
    for module_name in MAIN_ENV_MODULE_CANDIDATES:
        module = _safe_import_module(module_name)
        cls = _module_class_by_names(module, MAIN_ENV_CLASS_NAMES)

        if cls is not None:
            return cls

    cls = _scan_env_package(
        MAIN_ENV_CLASS_NAMES,
        required_methods=MAIN_REQUIRED_METHODS,
    )

    if cls is not None:
        return cls

    raise ImportError(
        "Cannot resolve main environment class.\n"
        "Expected module is usually:\n"
        "  envs.adaptive_resource_flow_arena_v3.AdaptiveResourceFlowArena\n"
        f"Required methods: {MAIN_REQUIRED_METHODS}"
    )


def _resolve_tiny_env_class():
    for module_name in TINY_ENV_MODULE_CANDIDATES:
        module = _safe_import_module(module_name)
        cls = _module_class_by_names(module, TINY_ENV_CLASS_NAMES)

        if cls is not None:
            return cls

    cls = _scan_env_package(
        TINY_ENV_CLASS_NAMES,
        required_methods=TINY_REQUIRED_METHODS,
    )

    if cls is not None:
        return cls

    raise ImportError(
        "Cannot resolve tiny oracle environment class.\n"
        "Expected module is usually:\n"
        "  envs.tiny_oracle_resource_flow_v1.TinyOracleResourceFlowV1\n"
        f"Required methods: {TINY_REQUIRED_METHODS}"
    )


def _constructor_kwargs_for_signature(cls, candidate_kwargs):
    try:
        sig = inspect.signature(cls)
    except Exception:
        return dict(candidate_kwargs)

    has_var_kwargs = False

    for _, p in sig.parameters.items():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kwargs = True
            break

    if has_var_kwargs:
        return dict(candidate_kwargs)

    accepted = {}

    for k, v in candidate_kwargs.items():
        if k in sig.parameters:
            accepted[k] = v

    return accepted


def _instantiate_with_fallbacks(cls, candidate_kwargs):
    filtered = _constructor_kwargs_for_signature(cls, candidate_kwargs)

    try:
        return cls(**filtered)
    except TypeError:
        pass

    try:
        return cls(**candidate_kwargs)
    except TypeError:
        pass

    return cls()


def _set_if_present(env, name, value):
    if hasattr(env, name):
        try:
            setattr(env, name, value)
        except Exception:
            pass


def _call_seed_if_present(env, seed):
    if hasattr(env, "set_seed"):
        try:
            env.set_seed(int(seed))
            return
        except TypeError:
            pass

    if hasattr(env, "seed"):
        try:
            env.seed(int(seed))
            return
        except TypeError:
            pass


def _ensure_env_aliases(env):
    if not hasattr(env, "n_agents") and hasattr(env, "num_agents"):
        try:
            env.n_agents = int(env.num_agents)
        except Exception:
            pass

    if not hasattr(env, "last_actions") and hasattr(env, "last_action"):
        try:
            env.last_actions = env.last_action
        except Exception:
            pass

    if hasattr(env, "n_agents") and not hasattr(env, "last_actions"):
        try:
            env.last_actions = [0 for _ in range(int(env.n_agents))]
        except Exception:
            pass

    return env


def _validate_main_env(env):
    env = _ensure_env_aliases(env)

    missing = []

    for name in MAIN_REQUIRED_METHODS:
        if not hasattr(env, name):
            missing.append(name)

    for name in RUNNER_REQUIRED_INSTANCE_ATTRS:
        if not hasattr(env, name):
            missing.append(name)

    if not hasattr(env, "n_agents"):
        missing.append("n_agents")

    if len(missing) > 0:
        raise AttributeError(
            "Main env is missing required attributes/methods used by runners: "
            + ", ".join(missing)
        )

    if not hasattr(env, "mode"):
        try:
            env.mode = "behavioral_drift"
        except Exception:
            pass

    return env


def _validate_tiny_env(env):
    env = _ensure_env_aliases(env)

    missing = []

    for name in TINY_REQUIRED_METHODS:
        if not hasattr(env, name):
            missing.append(name)

    for name in RUNNER_REQUIRED_INSTANCE_ATTRS:
        if not hasattr(env, name):
            missing.append(name)

    if not hasattr(env, "n_agents"):
        missing.append("n_agents")

    if len(missing) > 0:
        raise AttributeError(
            "Tiny env is missing required attributes/methods used by tiny calibration and FinalCIGAMFRunner: "
            + ", ".join(missing)
        )

    return env


def make_main_env(task_mode, n_agents, max_steps, phase_length, seed):
    EnvCls = _resolve_main_env_class()

    kwargs = {
        "mode": task_mode,
        "task_mode": task_mode,
        "n_agents": int(n_agents),
        "num_agents": int(n_agents),
        "max_steps": int(max_steps),
        "episode_length": int(max_steps),
        "phase_length": int(phase_length),
        "seed": int(seed),
        "diagnostic_core_k": 4,
        "max_core_size": 4,
        "resample_agent_layout_each_reset": False,
        "resample_hidden_rules_each_reset": False,
    }

    env = _instantiate_with_fallbacks(EnvCls, kwargs)

    if hasattr(env, "set_mode"):
        try:
            env.set_mode(task_mode)
        except TypeError:
            pass

    _set_if_present(env, "mode", task_mode)
    _set_if_present(env, "max_steps", int(max_steps))
    _set_if_present(env, "phase_length", int(phase_length))
    _set_if_present(env, "diagnostic_core_k", 4)
    _set_if_present(env, "max_core_size", 4)
    _set_if_present(env, "resample_agent_layout_each_reset", False)
    _set_if_present(env, "resample_hidden_rules_each_reset", False)

    _call_seed_if_present(env, seed)

    env = _validate_main_env(env)

    return env


def make_tiny_env(seed, max_steps=None, phase_length=None):
    EnvCls = _resolve_tiny_env_class()

    kwargs = {
        "seed": int(seed),
    }

    if max_steps is not None:
        kwargs["max_steps"] = int(max_steps)
        kwargs["horizon"] = int(max_steps)

    if phase_length is not None:
        kwargs["phase_length"] = int(phase_length)

    env = _instantiate_with_fallbacks(EnvCls, kwargs)

    _call_seed_if_present(env, seed)

    if max_steps is not None:
        _set_if_present(env, "max_steps", int(max_steps))
        _set_if_present(env, "horizon", int(max_steps))

    if phase_length is not None:
        _set_if_present(env, "phase_length", int(phase_length))

    env = _validate_tiny_env(env)

    return env


# ============================================================
# Runner factory
# ============================================================

def make_runner(model_name, env, cfg, device):
    model_name = str(model_name).strip()

    aliases = {
        "PureMeanField": "PureMeanField",
        "pure": "PureMeanField",
        "meanfield": "PureMeanField",
        "MeanField": "PureMeanField",

        "FullExplicitLocal": "FullExplicitLocal",
        "explicit": "FullExplicitLocal",
        "FullExplicit": "FullExplicitLocal",

        "Final-CIGAMF": "Final-CIGAMF",
        "Final": "Final-CIGAMF",
        "CIGAMF": "Final-CIGAMF",
        "CIG-AMF": "Final-CIGAMF",

        "NoBelief": "NoBelief",
        "NoMultiMemory": "NoMultiMemory",
        "NoTwoTimescale": "NoTwoTimescale",
    }

    canonical = aliases.get(model_name, model_name)

    if canonical == "PureMeanField":
        return PureMeanFieldRunner(env, cfg, device=device)

    if canonical == "FullExplicitLocal":
        return FullExplicitLocalRunner(env, cfg, device=device)

    if canonical == "Final-CIGAMF":
        return FinalCIGAMFRunner(env, cfg, device=device)

    if canonical == "NoBelief":
        return NoBeliefRunner(env, cfg, device=device)

    if canonical == "NoMultiMemory":
        return NoMultiMemoryRunner(env, cfg, device=device)

    if canonical == "NoTwoTimescale":
        return NoTwoTimescaleRunner(env, cfg, device=device)

    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# Population tasks
# ============================================================

def _task_to_env_mode(task_name):
    task_name = str(task_name)

    if task_name == "behavioral":
        return "behavioral_drift"

    if task_name == "structural":
        return "structural_shift"

    if task_name == "scalability":
        return "behavioral_drift"

    return task_name


def _eval_every_from_episodes(episodes, override=None):
    if override is not None:
        return max(1, int(override))

    episodes = int(episodes)
    return max(1, episodes // 8)


def _safe_model_filename(model_name):
    return (
        str(model_name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _history_summary_row(hist, base_row):
    row = dict(base_row)

    if hist is None:
        return row

    for key, values in hist.items():
        if not isinstance(values, list):
            continue

        if len(values) == 0:
            continue

        row[f"last_{key}"] = values[-1]

        numeric_vals = []

        for v in values:
            try:
                numeric_vals.append(float(v))
            except Exception:
                pass

        if len(numeric_vals) > 0:
            row[f"mean_{key}"] = float(np.mean(numeric_vals))
            row[f"std_{key}"] = float(np.std(numeric_vals))
            row[f"min_{key}"] = float(np.min(numeric_vals))
            row[f"max_{key}"] = float(np.max(numeric_vals))

    return row


def _save_history_outputs(task_dir, task_name, mode, model_name, hist, args):
    model_safe = _safe_model_filename(model_name)

    out_csv = os.path.join(task_dir, f"{model_safe}.csv")
    out_json = os.path.join(task_dir, f"{model_safe}.json")

    save_history_csv(
        hist,
        out_csv,
        extra={
            "task": task_name,
            "mode": mode,
            "model": model_name,
            "seed": int(args.seed),
            "n_agents": int(args.n_agents),
            "episodes_total": int(args.episodes),
            "max_steps": int(args.max_steps),
            "phase_length": int(args.phase_length),
        },
    )

    save_json(hist, out_json)


def run_population_task(task_name, args, cfg, model_names, device):
    mode = _task_to_env_mode(task_name)
    histories = {}

    task_dir = os.path.join(
        args.result_dir,
        f"{task_name}_seed_{args.seed}_N{args.n_agents}_E{args.episodes}",
    )
    ensure_dir(task_dir)

    metadata = {
        "task": task_name,
        "mode": mode,
        "seed": int(args.seed),
        "n_agents": int(args.n_agents),
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "phase_length": int(args.phase_length),
        "models": list(model_names),
        "device": str(device),
        "cfg": cfg,
    }

    save_json(metadata, os.path.join(task_dir, "metadata.json"))

    for model_name in model_names:
        print("")
        print(f"=== {mode} | {model_name} | seed={args.seed} ===")

        set_global_seed(int(args.seed))

        env = make_main_env(
            task_mode=mode,
            n_agents=args.n_agents,
            max_steps=args.max_steps,
            phase_length=args.phase_length,
            seed=args.seed,
        )

        runner = make_runner(model_name, env, cfg, device=device)

        history = runner.run(
            n_episodes=int(args.episodes),
            eval_every=_eval_every_from_episodes(args.episodes, args.eval_every),
        )

        histories[model_name] = history

        _save_history_outputs(
            task_dir=task_dir,
            task_name=task_name,
            mode=mode,
            model_name=model_name,
            hist=history,
            args=args,
        )

        del runner
        del env
        cleanup_runtime_memory()

    metrics_to_plot = [
        "mean_reward",
        "reward_per_agent",
        "mean_f1",
        "mean_uncertainty",
        "mean_core_size",
        "mean_core_switches",
        "mean_temporal_var",
        "proxy_loss",
        "bc_loss",
        "policy_loss",
        "throughput_agent_steps_per_sec",
        "proxy_buffer_size",
        "mean_mu",
        "max_p",
        "triggered",
        "trigger_count",
        "stage",
    ]

    for metric in metrics_to_plot:
        plot_histories(
            histories,
            metric=metric,
            title=f"{task_name} | {metric}",
            save_path=os.path.join(task_dir, f"{metric}.png"),
        )

    summary_rows = []

    for model_name, hist in histories.items():
        base_row = {
            "task": task_name,
            "mode": mode,
            "model": model_name,
            "seed": int(args.seed),
            "n_agents": int(args.n_agents),
            "episodes_total": int(args.episodes),
            "max_steps": int(args.max_steps),
            "phase_length": int(args.phase_length),
        }

        summary_rows.append(_history_summary_row(hist, base_row))

    save_csv(summary_rows, os.path.join(task_dir, "summary.csv"))

    if task_name == "structural":
        latency_rows = []
        eval_every = _eval_every_from_episodes(args.episodes, args.eval_every)
        shift_episode = int(args.phase_length)
        shift_eval_index = int(round(shift_episode / max(1, eval_every)))

        for model_name, hist in histories.items():
            if "episodes" not in hist or "mean_reward" not in hist:
                continue

            latency, baseline, target = recovery_latency(
                hist,
                shift_eval_index=shift_eval_index,
                metric_key="mean_reward",
                threshold=0.90,
                lookback=2,
                horizon=6,
            )

            latency_rows.append(
                {
                    "task": task_name,
                    "mode": mode,
                    "model": model_name,
                    "seed": int(args.seed),
                    "shift_episode": int(shift_episode),
                    "shift_eval_index": int(shift_eval_index),
                    "recovery_latency": int(latency),
                    "baseline": "" if baseline is None else float(baseline),
                    "target": "" if target is None else float(target),
                }
            )

        save_csv(latency_rows, os.path.join(task_dir, "recovery_latency.csv"))

    print("")
    print(f"Saved results to: {task_dir}")

    return histories


# ============================================================
# Scalability task
# ============================================================

def run_scalability_task(args, cfg, model_names, device):
    n_values = [
        int(x.strip())
        for x in str(args.scalability_agents).split(",")
        if x.strip()
    ]

    root_dir = os.path.join(
        args.result_dir,
        f"scalability_seed_{args.seed}_E{args.episodes}",
    )
    ensure_dir(root_dir)

    all_histories = {}
    scalability_rows = []

    for n in n_values:
        local_args = argparse.Namespace(**vars(args))
        local_args.n_agents = int(n)
        local_args.result_dir = root_dir

        histories = run_population_task(
            task_name="scalability",
            args=local_args,
            cfg=cfg,
            model_names=model_names,
            device=device,
        )

        for model_name, hist in histories.items():
            key = f"{model_name}_N{n}"
            all_histories[key] = hist

            base_row = {
                "task": "scalability",
                "mode": _task_to_env_mode("scalability"),
                "model": model_name,
                "seed": int(args.seed),
                "n_agents": int(n),
                "episodes_total": int(args.episodes),
                "max_steps": int(args.max_steps),
                "phase_length": int(args.phase_length),
            }

            scalability_rows.append(_history_summary_row(hist, base_row))

        cleanup_runtime_memory()

    for metric in [
        "mean_reward",
        "reward_per_agent",
        "throughput_agent_steps_per_sec",
        "mean_core_size",
        "mean_uncertainty",
        "mean_f1",
        "proxy_buffer_size",
    ]:
        plot_histories(
            all_histories,
            metric=metric,
            title=f"scalability | {metric}",
            save_path=os.path.join(root_dir, f"scalability_{metric}.png"),
        )

    save_csv(scalability_rows, os.path.join(root_dir, "scalability_summary.csv"))
    save_json(all_histories, os.path.join(root_dir, "scalability_histories.json"))

    return all_histories


# ============================================================
# Tiny oracle task
# ============================================================

def _tiny_candidate_intervention_actions(tiny_env):
    out = []

    preferred = [
        "STAY",
        "SIGNAL_A",
        "SIGNAL_B",
        "TOGGLE_LANE",
        "PICK",
        "DROP_TO_BUFFER",
        "PROCESS",
        "DELIVER",
    ]

    for name in preferred:
        if hasattr(tiny_env, name):
            out.append(int(getattr(tiny_env, name)))

    if len(out) == 0:
        out = list(range(int(tiny_env.get_action_dim())))

    seen = set()
    unique = []

    for x in out:
        if x not in seen:
            seen.add(x)
            unique.append(int(x))

    return unique


def _get_obs_all_from_env(env):
    if hasattr(env, "_get_obs_all"):
        return env._get_obs_all()

    if hasattr(env, "get_all_obs"):
        return env.get_all_obs()

    return [
        env.get_obs_of_ego(None, ego)
        for ego in range(int(env.n_agents))
    ]


def _call_oracle_influence(tiny_env, ego, j, action, horizon, discount):
    try:
        return tiny_env.compute_oracle_influence_from_current_state(
            ego_id=int(ego),
            agent_j=int(j),
            intervention_action=int(action),
            horizon=int(horizon),
            n_trials=3,
            discount=float(discount),
        )
    except TypeError:
        pass

    try:
        return tiny_env.compute_oracle_influence_from_current_state(
            ego_id=int(ego),
            agent_j=int(j),
            intervention_action=int(action),
            horizon=int(horizon),
            n_trials=3,
        )
    except TypeError:
        pass

    try:
        return tiny_env.compute_oracle_influence_from_current_state(
            int(ego),
            int(j),
            int(action),
            int(horizon),
        )
    except TypeError:
        pass

    return tiny_env.compute_oracle_influence_from_current_state(
        int(ego),
        int(j),
        int(action),
    )


def _compute_tiny_oracle_scores(tiny_env, state, ego, neighbor_ids, cfg, tiny_horizon):
    oracle_scores = {}
    candidate_actions = _tiny_candidate_intervention_actions(tiny_env)

    for j in neighbor_ids:
        vals = []

        for action in candidate_actions:
            tiny_env.restore_state(state)

            score = _call_oracle_influence(
                tiny_env=tiny_env,
                ego=ego,
                j=j,
                action=action,
                horizon=int(tiny_horizon),
                discount=float(cfg.get("discount", 0.95)),
            )

            vals.append(abs(float(score)))

        oracle_scores[int(j)] = float(np.mean(vals)) if len(vals) > 0 else 0.0

    tiny_env.restore_state(state)

    return oracle_scores


def _tiny_train_cfg_from_base(cfg):
    tiny_cfg = dict(cfg)

    tiny_cfg["k0_warmup"] = int(min(tiny_cfg.get("k0_warmup", 20), 2))
    tiny_cfg["slow_ratio"] = float(max(tiny_cfg.get("slow_ratio", 0.05), 1.0))
    tiny_cfg["proxy_train_steps"] = int(max(tiny_cfg.get("proxy_train_steps", 4), 8))
    tiny_cfg["proxy_batch_size"] = int(max(16, min(tiny_cfg.get("proxy_batch_size", 256), 128)))
    tiny_cfg["proxy_holdout_size"] = int(max(0, min(tiny_cfg.get("proxy_holdout_size", 64), 64)))
    tiny_cfg["bc_train_steps"] = int(max(1, min(tiny_cfg.get("bc_train_steps", 2), 2)))
    tiny_cfg["bc_batch_size"] = int(max(16, min(tiny_cfg.get("bc_batch_size", 128), 128)))
    tiny_cfg["seed_core_top_k"] = int(max(1, min(tiny_cfg.get("seed_core_top_k", 2), 2)))
    tiny_cfg["min_core_size"] = int(max(1, min(tiny_cfg.get("min_core_size", 1), 2)))
    tiny_cfg["max_core_size"] = int(max(tiny_cfg["min_core_size"], min(tiny_cfg.get("max_core_size", 4), 4)))

    return tiny_cfg
def _push_proxy_replay_compat(runner, trajectory):
    """
    Compatibility helper.

    Newer FinalCIGAMFRunner has:
        runner.push_proxy_replay(trajectory)

    Some current local versions do not. In that case, use the replay_builder
    directly so tiny-oracle calibration still trains the learned proxy from
    real population-wide trajectory samples.
    """
    if hasattr(runner, "push_proxy_replay"):
        return runner.push_proxy_replay(trajectory)

    if (
        hasattr(runner, "replay_builder")
        and hasattr(runner, "proxy")
        and hasattr(runner, "env")
    ):
        return runner.replay_builder.push_trajectory_to_proxy(
            trajectory=trajectory,
            proxy_ensemble=runner.proxy,
            env=runner.env,
        )

    raise AttributeError(
        "Runner has neither push_proxy_replay() nor replay_builder/proxy/env. "
        "Cannot push trajectory into local counterfactual proxy replay."
    )

def _train_tiny_runner_for_proxy(tiny_env, tiny_cfg, args, device):
    runner = FinalCIGAMFRunner(
        env=tiny_env,
        cfg=tiny_cfg,
        device=device,
    )

    if hasattr(runner.scheduler, "force_learned_stage"):
        runner.scheduler.force_learned_stage()

    train_episodes = int(max(1, getattr(args, "tiny_proxy_train_episodes", 8)))

    for ep in range(train_episodes):
        trajectory, episode_reward, runtime = runner.collect_episode()

        _push_proxy_replay_compat(runner, trajectory)

        runner.update_policy(trajectory)

        runner.pair_rel_module.train_bc(
            n_steps=tiny_cfg["bc_train_steps"],
            batch_size=tiny_cfg["bc_batch_size"],
        )

        runner.proxy.train_step(
            n_steps=tiny_cfg["proxy_train_steps"],
            batch_size=tiny_cfg["proxy_batch_size"],
            holdout_size=tiny_cfg.get("proxy_holdout_size", 0),
        )

        if len(trajectory) > 0:
            last = trajectory[-1]

            runner._score_all_pairs_and_update_beliefs(
                obs_all=last["obs_all"],
                actions=last["actions"],
            )

    return runner
def _core_context_excluding_all_compat(runner, ego):
    """
    Return dict {j: Z_i^{-j}} compatible with both new and old FinalCIGAMFRunner.
    """
    ego = int(ego)

    if hasattr(runner, "_core_context_excluding_all_for_ego"):
        return runner._core_context_excluding_all_for_ego(ego)

    out = {}

    for j in range(int(runner.n_agents)):
        if j == ego:
            continue

        if hasattr(runner, "_core_context_excluding"):
            out[int(j)] = runner._core_context_excluding(ego, j)
        else:
            core_set = runner.belief_modules[ego].get_core_set()
            reduced = [x for x in core_set if x != j]
            out[int(j)] = runner.pair_rel_module.get_core_summary(ego, reduced)

    return out


def _periph_context_excluding_all_compat(runner, ego):
    """
    Return dict {j: M_i^{-j}} compatible with both new and old FinalCIGAMFRunner.
    """
    ego = int(ego)

    if hasattr(runner, "_periph_context_excluding_all_for_ego"):
        return runner._periph_context_excluding_all_for_ego(ego)

    out = {}

    for j in range(int(runner.n_agents)):
        if j == ego:
            continue

        if hasattr(runner, "_periph_context_excluding"):
            out[int(j)] = runner._periph_context_excluding(ego, j)
        else:
            belief_mod = runner.belief_modules[ego]
            periph_ids = sorted(list(belief_mod.get_peripheral_set() - {j}))
            belief_state = belief_mod.get_state_dict()

            inputs = runner.periph_module.build_inputs(
                ego_id=ego,
                peripheral_ids=periph_ids,
                env=runner.env,
                belief_state=belief_state,
                prev_core_set=belief_mod.prev_core_set,
            )

            out[int(j)] = runner._periph_summary_np_from_inputs(inputs)

    return out

def _score_learned_proxy_for_state(runner, tiny_env, state, ego, neighbor_ids):
    tiny_env.restore_state(state)

    obs_all = _get_obs_all_from_env(tiny_env)

    if hasattr(tiny_env, "last_actions"):
        current_actions = list(tiny_env.last_actions)
    else:
        current_actions = [
            int(tiny_env.scripted_policy(a))
            for a in range(int(tiny_env.n_agents))
        ]

    belief_items = runner._build_belief_items_for_ego(int(ego))
    belief_summary = runner._belief_summary_np_from_items(belief_items)

    z_excluding = _core_context_excluding_all_compat(
        runner=runner,
        ego=int(ego),
    )

    m_excluding = _periph_context_excluding_all_compat(
        runner=runner,
        ego=int(ego),
    )

    obs_i = tiny_env.get_obs_of_ego(obs_all, int(ego))
    action_i = int(current_actions[int(ego)])

    learned_scores = {}
    sigmas = {}

    for j in neighbor_ids:
        j = int(j)
        action_j = int(current_actions[j])

        mu, sigma = runner.proxy.score_pair(
            obs_i=obs_i,
            action_i=action_i,
            observed_action_j=action_j,
            z_core_excl_j=z_excluding[j],
            m_periph_excl_j=m_excluding[j],
            belief_summary=belief_summary,
        )

        learned_scores[j] = float(abs(mu))
        sigmas[j] = float(sigma)

    tiny_env.restore_state(state)

    return learned_scores, sigmas


def run_tiny_task(args, cfg, device, out_dir=None, run_label="standalone"):
    """
    Post-hoc tiny-oracle calibration/evaluation protocol.

    Important: this function trains/evaluates a tiny-environment calibration
    runner and compares its learned proxy scores against controlled
    intervention effects. It does not inject oracle labels into the
    main-environment Final-CIGAMF training loop.
    """
    print("")
    print(f"=== tiny oracle calibration | seed={args.seed} | label={run_label} ===")

    set_global_seed(int(args.seed))

    tiny_env = make_tiny_env(
        seed=int(args.seed),
        max_steps=int(args.max_steps),
        phase_length=int(args.phase_length),
    )

    tiny_cfg = _tiny_train_cfg_from_base(cfg)

    if out_dir is None:
        out_dir = os.path.join(
            args.result_dir,
            f"tiny_seed_{args.seed}",
        )

    ensure_dir(out_dir)

    runner = _train_tiny_runner_for_proxy(
        tiny_env=tiny_env,
        tiny_cfg=tiny_cfg,
        args=args,
        device=device,
    )

    try:
        state_bank = tiny_env.sample_state_bank(
            n_states=int(args.tiny_states),
            burn_in=int(args.tiny_burn_in),
        )
    except TypeError:
        state_bank = tiny_env.sample_state_bank(
            int(args.tiny_states),
            int(args.tiny_burn_in),
        )

    rows = []
    aggregate_rows = []

    for state_idx, state in enumerate(state_bank):
        tiny_env.restore_state(state)

        if hasattr(tiny_env, "get_supported_egos"):
            egos = list(tiny_env.get_supported_egos())
        else:
            egos = list(range(int(tiny_env.n_agents)))

        for ego in egos:
            ego = int(ego)
            neighbor_ids = [j for j in range(int(tiny_env.n_agents)) if j != ego]

            learned_scores, sigmas = _score_learned_proxy_for_state(
                runner=runner,
                tiny_env=tiny_env,
                state=state,
                ego=ego,
                neighbor_ids=neighbor_ids,
            )

            oracle_scores = _compute_tiny_oracle_scores(
                tiny_env=tiny_env,
                state=state,
                ego=ego,
                neighbor_ids=neighbor_ids,
                cfg=tiny_cfg,
                tiny_horizon=int(args.tiny_horizon),
            )

            cal = oracle_calibration(
                learned_scores=learned_scores,
                oracle_scores=oracle_scores,
                neighbor_ids=neighbor_ids,
            )

            top_k = int(max(1, min(tiny_cfg.get("max_core_size", 4), len(neighbor_ids))))

            core_f1 = oracle_core_f1_from_scores(
                learned_scores=learned_scores,
                oracle_scores=oracle_scores,
                neighbor_ids=neighbor_ids,
                top_k=top_k,
            )

            aggregate_row = {
                "seed": int(args.seed),
                "state_idx": int(state_idx),
                "ego_id": int(ego),
                "top_k": int(top_k),
                "oracle_core_f1": float(core_f1),
            }
            aggregate_row.update(cal)
            aggregate_rows.append(aggregate_row)

            for j in neighbor_ids:
                rows.append(
                    {
                        "seed": int(args.seed),
                        "state_idx": int(state_idx),
                        "ego_id": int(ego),
                        "neighbor_id": int(j),
                        "learned_score": float(learned_scores.get(j, 0.0)),
                        "oracle_score": float(oracle_scores.get(j, 0.0)),
                        "abs_error": float(abs(learned_scores.get(j, 0.0) - oracle_scores.get(j, 0.0))),
                        "proxy_sigma": float(sigmas.get(j, 0.0)),
                    }
                )

    save_csv(rows, os.path.join(out_dir, "tiny_oracle_pair_rows.csv"))
    save_csv(aggregate_rows, os.path.join(out_dir, "tiny_oracle_calibration_by_state.csv"))

    summary = {
        "seed": int(args.seed),
        "run_label": str(run_label),
        "calibration_scope": "posthoc_tiny_oracle_evaluation",
        "uses_oracle_for_training": False,
        "main_training_integration": "evaluation_pipeline_only_not_policy_or_proxy_supervision",
        "n_pair_rows": int(len(rows)),
        "n_state_ego_rows": int(len(aggregate_rows)),
        "proxy_buffer_size": int(runner.proxy.get_buffer_size()),
        "tiny_proxy_train_episodes": int(args.tiny_proxy_train_episodes),
        "note": (
            "learned_score is computed from a tiny-environment calibration "
            "FinalCIGAMFRunner.proxy.score_pair on the same sampled tiny-oracle "
            "state and pair. Oracle intervention scores are used only for "
            "post-hoc evaluation/calibration metrics, not as training labels for "
            "the main-environment runner."
        ),
    }

    numeric_keys = [
        "bias",
        "variance",
        "mae",
        "rmse",
        "rank_correlation",
        "p_value",
        "constant_case",
        "oracle_core_f1",
    ]

    for key in numeric_keys:
        vals = []

        for row in aggregate_rows:
            if key in row and row[key] is not None:
                try:
                    vals.append(float(row[key]))
                except Exception:
                    pass

        if len(vals) > 0:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
            summary[f"{key}_min"] = float(np.min(vals))
            summary[f"{key}_max"] = float(np.max(vals))
        else:
            summary[f"{key}_mean"] = 0.0
            summary[f"{key}_std"] = 0.0
            summary[f"{key}_min"] = 0.0
            summary[f"{key}_max"] = 0.0

    save_json(summary, os.path.join(out_dir, "tiny_oracle_summary.json"))

    print("")
    print("=== tiny oracle calibration summary ===")
    print(f"pair_rows={summary['n_pair_rows']}")
    print(f"state_ego_rows={summary['n_state_ego_rows']}")
    print(f"proxy_buffer_size={summary['proxy_buffer_size']}")
    print(f"rank_correlation_mean={summary.get('rank_correlation_mean', 0.0):.4f}")
    print(f"oracle_core_f1_mean={summary.get('oracle_core_f1_mean', 0.0):.4f}")
    print(f"mae_mean={summary.get('mae_mean', 0.0):.4f}")
    print(f"Saved tiny oracle results to: {out_dir}")

    del runner
    del tiny_env
    cleanup_runtime_memory()

    return summary


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "behavioral", "structural", "scalability", "tiny"],
    )

    parser.add_argument(
        "--models",
        type=str,
        default="PureMeanField,FullExplicitLocal,Final-CIGAMF,NoBelief,NoMultiMemory,NoTwoTimescale",
    )

    parser.add_argument("--n_agents", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--phase_length", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--result_dir", type=str, default="results")

    parser.add_argument(
        "--scalability_agents",
        type=str,
        default="8,16,24,48",
    )

    parser.add_argument("--tiny_states", type=int, default=8)
    parser.add_argument("--tiny_burn_in", type=int, default=4)
    parser.add_argument("--tiny_horizon", type=int, default=3)
    parser.add_argument("--tiny_proxy_train_episodes", type=int, default=8)
    parser.add_argument(
        "--with_tiny_calibration",
        action="store_true",
        help=(
            "After a population task, also run the post-hoc tiny-oracle "
            "calibration/evaluation protocol and save it under that task result directory. "
            "This does not use oracle labels in main training."
        ),
    )

    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def _resolve_device(device_arg):
    device_arg = str(device_arg).lower()

    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    return device_arg


def _print_startup(args, model_names, device, cfg):
    print("")
    print("=== CIG-AMF experiment runner ===")
    print(f"task={args.task}")
    print(f"models={model_names}")
    print(f"seed={args.seed}")
    print(f"n_agents={args.n_agents}")
    print(f"episodes={args.episodes}")
    print(f"max_steps={args.max_steps}")
    print(f"phase_length={args.phase_length}")
    print(f"result_dir={args.result_dir}")
    print(f"device={device}")
    print(f"smoke={bool(args.smoke)}")
    print("")

    print("=== Env resolver targets ===")
    print("main expected: envs.adaptive_resource_flow_arena_v3.AdaptiveResourceFlowArena")
    print("tiny expected: envs.tiny_oracle_resource_flow_v1.TinyOracleResourceFlowV1")
    print("")



def _maybe_run_tiny_calibration_after_population(task_name, args, cfg, device):
    """
    Optionally attach tiny-oracle calibration results to a population task.

    This integrates tiny-oracle into the experiment output pipeline, not into
    the main training loop. Oracle intervention scores are used only for
    post-hoc calibration/evaluation metrics and are never fed back into the
    main policy, proxy, belief, scheduler, or environment updates.
    """
    if not bool(getattr(args, "with_tiny_calibration", False)):
        return

    task_name = str(task_name)

    if task_name not in {"behavioral", "structural"}:
        print(
            f"[INFO] Skipping tiny-oracle calibration for task={task_name}. "
            "Tiny calibration is attached only to behavioral/structural tasks."
        )
        return

    task_dir = os.path.join(
        args.result_dir,
        f"{task_name}_seed_{args.seed}_N{args.n_agents}_E{args.episodes}",
    )

    calib_dir = os.path.join(task_dir, "tiny_oracle_calibration")

    run_tiny_task(
        args=args,
        cfg=cfg,
        device=device,
        out_dir=calib_dir,
        run_label=f"{task_name}_posthoc",
    )


def main():
    args = parse_args()

    set_global_seed(int(args.seed))

    cfg = smoke_cfg() if args.smoke else default_cfg()

    if args.smoke:
        args.episodes = min(int(args.episodes), 12)
        args.max_steps = min(int(args.max_steps), 12)
        args.n_agents = min(int(args.n_agents), 8)
        args.tiny_states = min(int(args.tiny_states), 3)
        args.tiny_proxy_train_episodes = min(int(args.tiny_proxy_train_episodes), 2)

    device = _resolve_device(args.device)

    ensure_dir(args.result_dir)

    model_names = [
        m.strip()
        for m in str(args.models).split(",")
        if m.strip()
    ]

    _print_startup(args, model_names, device, cfg)

    if args.task == "behavioral":
        run_population_task(
            task_name="behavioral",
            args=args,
            cfg=cfg,
            model_names=model_names,
            device=device,
        )
        _maybe_run_tiny_calibration_after_population("behavioral", args, cfg, device)

    elif args.task == "structural":
        run_population_task(
            task_name="structural",
            args=args,
            cfg=cfg,
            model_names=model_names,
            device=device,
        )
        _maybe_run_tiny_calibration_after_population("structural", args, cfg, device)

    elif args.task == "scalability":
        run_scalability_task(
            args=args,
            cfg=cfg,
            model_names=model_names,
            device=device,
        )

    elif args.task == "tiny":
        run_tiny_task(
            args=args,
            cfg=cfg,
            device=device,
        )

    elif args.task == "all":
        run_population_task(
            task_name="behavioral",
            args=args,
            cfg=cfg,
            model_names=model_names,
            device=device,
        )
        _maybe_run_tiny_calibration_after_population("behavioral", args, cfg, device)

        run_population_task(
            task_name="structural",
            args=args,
            cfg=cfg,
            model_names=model_names,
            device=device,
        )
        _maybe_run_tiny_calibration_after_population("structural", args, cfg, device)

        run_scalability_task(
            args=args,
            cfg=cfg,
            model_names=model_names,
            device=device,
        )

        run_tiny_task(
            args=args,
            cfg=cfg,
            device=device,
        )

    cleanup_runtime_memory()

    print("")
    print("Selected tasks finished.")
    print(f"Check results under: {args.result_dir}")


if __name__ == "__main__":
    main()