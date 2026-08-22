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
        CorrelationMeanFieldRunner,
        FullExplicitLocalRunner,
        NoBeliefRunner,
        NoMultiMemoryRunner,
        OracleCoreRunner,      # [FIX-O3]
        RandomCoreRunner,      # [FIX-O3]
    )
except ModuleNotFoundError:
    from runners.baseline_runner import (
        PureMeanFieldRunner,
        CorrelationMeanFieldRunner,
        FullExplicitLocalRunner,
        NoBeliefRunner,
        NoMultiMemoryRunner,
        OracleCoreRunner,      # [FIX-O3]
        RandomCoreRunner,      # [FIX-O3]
    )

from models.structural_proxy import build_pair_feat  # [FIX-X1]

try:
    from runners.final_runner import FinalCIGAMFRunner, NoTwoTimescaleRunner
except ModuleNotFoundError:
    from runners.final_runner import FinalCIGAMFRunner, NoTwoTimescaleRunner


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
        safe_spearman,
    )
except ModuleNotFoundError:
    from utils.metrics import (
        oracle_calibration,
        oracle_core_f1_from_scores,
        recovery_latency,
        safe_spearman,
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

        # Four fixed semantic roles plus two learned residual slots, matching
        # Eq. 23.  A total of four silently removed every free slot.
        "num_memory_slots": 6,
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
        # See TwoTimescaleScheduler for require_both/inflation_t_reset. The
        # original defaults in training/scheduler.py are False and 1.
        "require_both": False,
        "inflation_t_reset": 1,

        "belief_lambda_0": 0.08,
        "belief_uncertainty_scale": 2.0,
        "belief_tau": 0.005,
        "belief_tau_in": 0.55,
        "belief_tau_out": 0.35,
        "seed_core_top_k": 3,
        # Fixed oracle truth cardinality.  Evaluation never adapts the target
        # core size to a model's predicted core, which would inflate F1.
        "ground_truth_core_k": 3,
        "min_core_size": 2,
        "max_core_size": 4,
        "sigma_floor": 0.08,
        # belief_layer v2: retain the original recommended defaults.
        "belief_core_rule": "lcb",
        "belief_kappa": 1.0,
        "belief_alpha_decay": 0.7,
        "belief_sigma_alpha_max": 1.0,
        # [P-6 FINAL DEBUG] Enable Equation 17: an entropy-adaptive core budget.
        # False makes capacity allocation constant, leaving the RQ3 claim
        # untestable. The paper also requires reporting the "fraction of updates
        # at which k_i binds at k_max." The implementation already exists in
        # belief_layer.py:322–360.
        "belief_adaptive_k": True,
        "belief_adaptive_k_min": 1,
        "belief_signed_balance": 0.5,

        "periph_mu_floor": 0.015,
        "periph_beta_floor": 0.05,
        # Uniform mixing recreates the global mean inside every slot and is
        # therefore reserved for a legacy-collapse ablation, never Full.
        "periph_uniform_mix": 0.0,
        "periph_use_uniform_mix": False,
        "periph_routing_mode": "semantic",
        "periph_signature_mode": "full",
        # Main runners may opt into strictness once the signature tracker is
        # plumbed into build_inputs. H3 overrides this to True and fails fast
        # if it would otherwise test the legacy derived vector.
        "periph_require_full_signature": False,
        "periph_allow_legacy_items": True,
        # Switch load balancing now applies only to trainable exchangeable
        # slots. The previous four-slot configuration contained only fixed
        # semantic gates, so changing this coefficient could not change router
        # gradients; the apparent 1.2-versus-0.05 tuning result was therefore
        # not evidence about load balancing. Use the published Switch scale.
        "periph_lb_coeff": 1e-2,
        # [FIX-2] Expose orth_coeff through configuration so the No-AuxLoss
        # ablation disables BOTH Equation 27 components, not only load balancing.
        "periph_orth_coeff": 1e-2,
        "belief_priority_mu_floor": 0.01,
        "shadow_loss_weight": 0.25,
        "graph_score_steps": 8,

        # structural_proxy v2 defaults match structural_proxy.py.
        "proxy_n_horizons": 3,
        # Stable H1 estimand: value of the logged neighbour policy relative to
        # a uniform reference policy. The former realised-action/range target
        # changed with the sampled action and could not be ranked consistently.
        "proxy_effect_mode": "signed_policy_contrast",
        "proxy_use_doubly_robust": True,
        "proxy_iw_clip": 10.0,
        "proxy_bootstrap_ratio": 0.8,
        "proxy_use_belief_input": False,
        # [FIX-X1] x_ij dimensionality in Equation 8. The default five features
        # are drow, dcol, distance, same_zone, and zone_diff. Set this to zero
        # to reproduce the old configuration, where f_theta is blind to
        # neighbour identity, for the H1 ablation.
        "proxy_pair_feat_dim": 5,
        "proxy_ensemble_dropout": 0.0,
        # Controlled H2 behavioural manipulation.  The active policy becomes
        # (1-lambda)*pi_learned + lambda*pi_scripted before epsilon forcing.
        # Default zero preserves ordinary training; the H2 protocol sets this
        # to one in its behavioural arm and verifies KL/TV diagnostics.
        "behavioral_adapter_lambda": 0.0,
        "behavioral_adapter_only_in_behavioral_drift": True,
        "seed": 0,

        # final_runner.py sig_tracker/forcer/heads/drift/matdet/recip defaults
        # match the runner's original hard-coded values. See
        # models/{influence_signature,intervention,ego_conditioned_latent,
        # drift_probe,reciprocity}.py.
        "sig_tracker_window": 30,
        "semantic_calibration_every": 25,

        # [EPS-ANNEAL] 0.03 -> 0.05. A1 showed that ONLY eps=0.05 separated
        # from noise: Spearman +0.074, positive for 8/8 seeds, and surviving
        # Bonferroni correction.
        "eps": 0.05,
        # [FIX-P1] 2 -> None. The cap makes the ACTUAL forcing probability
        # eps_eff ~ 0.0282 differ by ~6% from the 0.03 recorded in propensity.
        # DR then loses unbiasedness and the "b_j known exactly" claim becomes
        # false. With eps=0.03 and n=24, E[#forced] = 0.72 per step, so the cap
        # is almost never reached and removing it costs essentially nothing.
        # See the detailed warning in intervention.py.
        "forcer_max_forced_per_step": None,
        # [EPS-ANNEAL] 0.01 -> 0.05, meaning NO annealing.
        # Evidence: min_head_frac decreased MONOTONICALLY by episode:
        #     ep14: 0.099   ep28: 0.05   ep42: 0.026   ep56: 0.014
        # exactly tracking eps annealing from 0.03 to 0.01 over 60 episodes.
        # forced_frac fell with it from 0.13 to 0.02. Because min_head_frac is
        # lower-bounded by forced_frac/|A|, rare-action heads were starved of
        # data late in the run, exactly when the best estimate was needed.
        # Epsilon forcing is the ONLY IDENTIFICATION SOURCE; annealing it away
        # removes that source. The paper commits to measuring and reporting its
        # behavioural-noise cost, not to minimizing that cost.
        "forcer_anneal_to": 0.05,
        "forcer_anneal_episodes": 60,

        "heads_lr": 5e-4,
        "heads_w_contrastive": 1,
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
    # ==================================================================
    # [ENV-RESOLVE] THE MOST SERIOUS DEFECT FOUND IN THE PROJECT.
    #
    # This list PREVIOUSLY placed AdaptiveResourceFlowArena first, so
    # make_main_env() always constructed envs/adaptive_resource_flow_arena_v3.py,
    # NOT OmniArena. Consequences:
    #
    #   * run_step_1, run_step_0, run_h2, and run_h3 all pass through
    #     make_main_env and therefore ran on arena_v3.
    #   * env_audit.py and env_audit_staged.py import OmniArena DIRECTLY and
    #     therefore measured OmniArena. Audit and training consequently
    #     REFERRED TO TWO DIFFERENT ENVIRONMENTS throughout the project. This
    #     explains why env_audit passed while reward did not improve and why
    #     cross-diagnostics between them contradicted each other.
    #   * ALL C1/C2/C3 work—Frenet (s,d), continuous SGTP Phi, zone asymmetry,
    #     relay gate, and remeasured Phi—lived in omni_arena.py and therefore
    #     NEVER affected a single training step.
    #
    # CONCLUSIVE EVIDENCE: arena_v3.get_action_dim() returns 13, whereas
    # OmniArena.N_ACTIONS = 6. The [VERIFY-F1] log printed hist_action_forced
    # with EXACTLY 13 entries. Observed min_head_frac of 0.001–0.005 also
    # matches the forced_frac/13 ~ 0.006 lower bound exactly. This was NOT the
    # suspected action-label defect; the real |A| was 13 rather than 6.
    #
    # OmniArena has been verified to implement all seven methods in
    # MAIN_REQUIRED_METHODS.
    # ==================================================================
    "OmniArena",
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
    "envs.omni_arena",          # [ENV-RESOLVE] See MAIN_ENV_CLASS_NAMES.
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

    # ------------------------------------------------------------------
    # [EXP5-ZONES] n_zones MUST scale with N.
    # omni_arena only enforces `assert n_agents >= 5 * n_zones`, and n_zones
    # defaults to the constructor constant 4; no formula derives it from
    # n_agents. Consequently, the Experiment 5 sweep over N in {8..96} always
    # ran with EXACTLY four zones. At N=96 this gives 24 agents per zone, of
    # which only five have functional roles and the other 19 are ROLE_DRIFTER.
    # Exp5 was therefore measuring "more drifters," NOT "more structure";
    # the zone-symmetry-breaking patch cannot repair this design error.
    # Preserve approximately six agents per zone: N=24 -> 4 zones and
    # N=96 -> 16 zones. Because zone_param_rng uses the same seed, the first
    # four zones at N=96 reproduce the four draws at N=24. N=96 can therefore
    # be read as "N=24 plus 12 new zones," not a completely different system.
    # grid_size must also grow with sqrt(n_zones), or cell_h contracts and the
    # path_len assertion in _init_zone_layout fires.
    # ------------------------------------------------------------------
    n_zones = max(1, int(n_agents) // 6)
    grid_size = max(24, int(np.ceil(12 * np.sqrt(max(1, n_zones)))))

    kwargs = {
        "mode": task_mode,
        "task_mode": task_mode,
        "n_agents": int(n_agents),
        "n_zones": int(n_zones),
        "grid_size": int(grid_size),
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

    # [ENV-RESOLVE] Print the concrete environment class. The defect above
    # persisted because NO LOG LINE identified which environment was running.
    print(f"[ENV-RESOLVE] main env = {EnvCls.__module__}.{EnvCls.__name__} "
          f"(n_agents={n_agents}, n_zones={n_zones}, grid={grid_size})")

    env = _instantiate_with_fallbacks(EnvCls, kwargs)

    try:
        print(f"[ENV-RESOLVE] action_dim={env.get_action_dim()} "
              f"obs_dim={env.get_obs_dim()}")
    except Exception:
        pass

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

        "CorrelationMeanField": "CorrelationMeanField",
        "correlation_meanfield": "CorrelationMeanField",
        "association": "CorrelationMeanField",

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

        # [FIX-O3] OracleCoreRunner and RandomCoreRunner had long existed in
        # baseline_runner.py but were ABSENT from this alias table. make_runner()
        # could never construct them, so run_step_0.py used FullExplicitLocal as
        # a substitute and the Experiment 0 gate had never been measured
        # correctly. Register the actual runners here.
        "OracleCore": "OracleCore",
        "oracle_core": "OracleCore",
        "RandomCore": "RandomCore",
        "random_core": "RandomCore",
    }

    canonical = aliases.get(model_name, model_name)

    if canonical == "PureMeanField":
        return PureMeanFieldRunner(env, cfg, device=device)

    if canonical == "CorrelationMeanField":
        return CorrelationMeanFieldRunner(env, cfg, device=device)

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

    if canonical == "OracleCore":
        return OracleCoreRunner(env, cfg, device=device)

    if canonical == "RandomCore":
        return RandomCoreRunner(env, cfg, device=device)

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
    """Return the complete discrete action support used by the proxy.

    Earlier calibration averaged eight hand-picked actions in the oracle while
    ``signed_oracle_matched`` averaged all thirteen proxy heads.  Correlation
    between those two quantities cannot validate the estimator.  The tiny
    environment exposes a finite action space, so exact support matching is
    both cheaper and clearer than maintaining a partial allow-list.
    """
    return list(range(int(tiny_env.get_action_dim())))


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


def _oracle_profile_value(score, key, default=None):
    if isinstance(score, dict):
        if key in score:
            return float(score[key])
        if default is not None:
            return float(default)
    return float(score)


def _compute_tiny_oracle_scores(tiny_env, state, ego, neighbor_ids, cfg, tiny_horizon):
    signed_scores = {}
    magnitude_scores = {}
    range_scores = {}
    candidate_actions = _tiny_candidate_intervention_actions(tiny_env)

    for j in neighbor_ids:
        signed_vals = []
        magnitude_vals = []
        range_vals = []

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

            signed = _oracle_profile_value(score, "signed")
            signed_vals.append(signed)
            magnitude_vals.append(abs(signed))
            range_vals.append(_oracle_profile_value(score, "range", default=abs(signed)))

        signed_scores[int(j)] = (
            float(np.mean(signed_vals)) if len(signed_vals) > 0 else 0.0
        )
        magnitude_scores[int(j)] = (
            float(np.mean(magnitude_vals)) if len(magnitude_vals) > 0 else 0.0
        )
        range_scores[int(j)] = (
            float(np.mean(range_vals)) if len(range_vals) > 0 else 0.0
        )

    tiny_env.restore_state(state)

    return {
        "signed": signed_scores,
        "magnitude": magnitude_scores,
        "range": range_scores,
    }


def _signed_calibration(proxy_signed, oracle_signed, neighbor_ids):
    ids = list(neighbor_ids)
    if len(ids) == 0:
        return {
            "signed_bias": 0.0,
            "signed_mae": 0.0,
            "signed_rmse": 0.0,
            "signed_spearman": 0.0,
            "signed_p_value": 1.0,
            "signed_constant_case": 1,
            "sign_agreement": 0.0,
        }

    proxy = np.array([float(proxy_signed.get(j, 0.0)) for j in ids], dtype=np.float32)
    oracle = np.array([float(oracle_signed.get(j, 0.0)) for j in ids], dtype=np.float32)

    diff = proxy - oracle
    rho, p_value, constant_case = safe_spearman(proxy, oracle)

    nonzero = np.abs(oracle) > 1e-8
    sign_agreement = (
        float(np.mean(np.sign(proxy[nonzero]) == np.sign(oracle[nonzero])))
        if np.any(nonzero)
        else 0.0
    )

    return {
        "signed_bias": float(np.mean(diff)),
        "signed_mae": float(np.mean(np.abs(diff))),
        "signed_rmse": float(np.sqrt(np.mean(diff ** 2))),
        "signed_spearman": float(rho),
        "signed_p_value": float(p_value),
        "signed_constant_case": int(constant_case),
        "sign_agreement": float(sign_agreement),
    }


def _response_surface_calibration(proxy_q, oracle_q, neighbor_ids):
    """Calibrate the primitive action-response surface Q before C/D.

    The flattened comparison keeps every (neighbour, candidate-action) value;
    reporting only C or D can hide an action-head failure whose aggregation
    happens to look plausible.
    """
    learned, oracle = [], []
    for neighbor in neighbor_ids:
        p = np.asarray(proxy_q.get(int(neighbor), []), dtype=np.float64).reshape(-1)
        q = np.asarray(oracle_q.get(int(neighbor), []), dtype=np.float64).reshape(-1)
        if p.shape != q.shape or p.size == 0:
            raise ValueError(
                f"Q calibration action support mismatch for neighbor={neighbor}: "
                f"{p.shape} vs {q.shape}"
            )
        learned.extend(p.tolist())
        oracle.extend(q.tolist())
    if not learned:
        return {"q_mae": 0.0, "q_rmse": 0.0, "q_spearman": 0.0, "q_constant_case": 1}
    learned = np.asarray(learned, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    rho, _p, constant = safe_spearman(learned, oracle)
    diff = learned - oracle
    return {
        "q_mae": float(np.mean(np.abs(diff))),
        "q_rmse": float(np.sqrt(np.mean(diff ** 2))),
        "q_spearman": float(rho),
        "q_constant_case": int(constant),
    }


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

    # H1 compares plug-in and AIPW scoring on the same nuisance learner and
    # data-generating process. Letting DR-modified beliefs alter later actions
    # would make the ablation compare two training distributions rather than
    # two estimators. Keep online structural scoring plug-in during nuisance
    # training, then restore the requested evaluation mode below.
    h1_requested_dr = bool(runner.proxy.use_doubly_robust)
    if bool(getattr(args, "h1_exact_protocol", False)):
        runner.proxy.use_doubly_robust = False

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

            # Compute H-step returns for this trajectory to provide
            # observed_returns into score_batch for possible DR correction.
            try:
                h_returns = runner.replay_builder.build_h_step_returns(trajectory, runner.n_agents)
                observed_returns = h_returns[-1]
            except Exception:
                observed_returns = None

            behaviour_probs = last.get("behaviour_probs") if isinstance(last, dict) else None
            # ``signed_policy_contrast`` is defined for the policy that
            # generated this action-time state.  The training phase keeps the
            # plug-in estimator active, but still supplies the logged policy
            # probabilities so the structural target remains well-defined.
            policy_probs = last.get("policy_probs") if isinstance(last, dict) else None

            runner._score_all_pairs_and_update_beliefs(
                obs_all=last["obs_all"],
                actions=last["actions"],
                observed_returns=observed_returns,
                behaviour_probs=behaviour_probs,
                policy_probs=policy_probs,
            )

    if bool(getattr(args, "h1_exact_protocol", False)):
        runner.proxy.use_doubly_robust = h1_requested_dr
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
    neighbor_ids = [int(j) for j in neighbor_ids]

    obs_all = _get_obs_all_from_env(tiny_env)

    if hasattr(tiny_env, "scripted_policy"):
        current_actions = [
            int(tiny_env.scripted_policy(a))
            for a in range(int(tiny_env.n_agents))
        ]
    elif hasattr(tiny_env, "last_actions"):
        current_actions = list(tiny_env.last_actions)
    else:
        current_actions = [0 for _ in range(int(tiny_env.n_agents))]

    belief_items = runner._build_belief_items_for_ego(int(ego))
    belief_summary = runner._belief_summary_np_from_items(belief_items)

    if hasattr(runner, "_raw_proxy_context_excluding"):
        raw_excluding = {
            int(j): runner._raw_proxy_context_excluding(int(ego), int(j))
            for j in neighbor_ids if int(j) != int(ego)
        }
        z_excluding = {j: raw_excluding[j][0] for j in raw_excluding}
        m_excluding = {j: raw_excluding[j][1] for j in raw_excluding}
    else:
        z_excluding = _core_context_excluding_all_compat(runner=runner, ego=int(ego))
        m_excluding = _periph_context_excluding_all_compat(runner=runner, ego=int(ego))

    obs_i = tiny_env.get_obs_of_ego(obs_all, int(ego))
    action_i = int(current_actions[int(ego)])

    learned_signed = {}
    learned_magnitude = {}
    learned_range = {}
    sigmas = {}

    old_effect_mode = getattr(runner.proxy, "effect_mode", None)
    try:
        runner.proxy.effect_mode = "signed_oracle_matched"

        out = runner.proxy.score_batch_full(
            obs_i_batch=[obs_i for _ in neighbor_ids],
            action_i_batch=[action_i for _ in neighbor_ids],
            observed_action_j_batch=[int(current_actions[j]) for j in neighbor_ids],
            z_core_excl_j_batch=[z_excluding[j] for j in neighbor_ids],
            m_periph_excl_j_batch=[m_excluding[j] for j in neighbor_ids],
            belief_summary_batch=[belief_summary for _ in neighbor_ids],
            # [FIX-X1] The H1 calibration path MUST also provide x_ij; otherwise
            # the proxy raises when pair_feat_dim > 0. This is the exact path
            # that measures Spearman correlation against the oracle, so omitting
            # x_ij here removes it at the most important measurement site.
            pair_feat_batch=[
                build_pair_feat(
                    tiny_env.positions, tiny_env.agent_zone,
                    getattr(tiny_env, "grid_size", 1),
                    getattr(tiny_env, "n_zones", 1), int(ego), int(j),
                )
                for j in neighbor_ids
            ],
        )

        for k, j in enumerate(neighbor_ids):
            mu = float(out["mu"][k])
            learned_signed[j] = mu
            learned_magnitude[j] = abs(mu)
            learned_range[j] = float(out["mu_range"][k])
            sigmas[j] = float(out["sigma"][k])
    finally:
        if old_effect_mode is not None:
            runner.proxy.effect_mode = old_effect_mode

    tiny_env.restore_state(state)

    return {
        "signed": learned_signed,
        "magnitude": learned_magnitude,
        "range": learned_range,
    }, sigmas


def _score_h1_logged_step(runner, tiny_env, step, ego, neighbor_ids):
    """Score one factual action-time cache with all AIPW inputs present.

    H1 previously restored only the environment and then constructed
    ``Z^{-j}``, ``M^{-j}``, and belief features from the runner's unrelated
    end-of-training state.  It also omitted the observed return and propensity,
    silently turning every DR-labelled evaluation into plug-in scoring.  The
    trajectory cache is the only source that keeps observation, action,
    relational context, reward, and propensity aligned at the same time.
    """
    required = (
        "obs_all", "actions", "rewards", "proxy_context_excluding",
        "belief_summary_cache", "geom_snapshot",
        "behaviour_probs", "policy_probs", "env_snapshot_before_step",
    )
    missing = [name for name in required if step.get(name) is None]
    if missing:
        raise RuntimeError(
            "H1 exact protocol requires action-time trajectory fields; missing "
            + ", ".join(missing)
        )

    ego = int(ego)
    neighbor_ids = [int(j) for j in neighbor_ids]
    actions = [int(a) for a in step["actions"]]
    geom = step["geom_snapshot"]
    observed_return = float(step["rewards"][ego])

    behaviour_obs = []
    policy_rows = []
    for j in neighbor_ids:
        action_j = actions[j]
        behaviour_obs.append(float(step["behaviour_probs"][j][action_j]))
        policy_rows.append(np.asarray(step["policy_probs"][j], dtype=np.float32))

    old_effect_mode = getattr(runner.proxy, "effect_mode", None)
    try:
        runner.proxy.effect_mode = "signed_policy_contrast"
        out = runner.proxy.score_batch_full(
            obs_i_batch=[
                tiny_env.get_obs_of_ego(step["obs_all"], ego)
                for _ in neighbor_ids
            ],
            action_i_batch=[actions[ego] for _ in neighbor_ids],
            observed_action_j_batch=[actions[j] for j in neighbor_ids],
            z_core_excl_j_batch=[
                step["proxy_context_excluding"][ego][j][0] for j in neighbor_ids
            ],
            m_periph_excl_j_batch=[
                step["proxy_context_excluding"][ego][j][1] for j in neighbor_ids
            ],
            belief_summary_batch=[
                step["belief_summary_cache"][ego] for _ in neighbor_ids
            ],
            policy_probs_j_batch=policy_rows,
            observed_returns_batch=[observed_return for _ in neighbor_ids],
            behaviour_probs_obs_batch=behaviour_obs,
            pair_feat_batch=[
                build_pair_feat(
                    geom["positions"], geom["agent_zone"], geom["grid_size"],
                    geom["n_zones"], ego, j,
                )
                for j in neighbor_ids
            ],
        )
    finally:
        if old_effect_mode is not None:
            runner.proxy.effect_mode = old_effect_mode

    signed = {
        j: float(out.get("d_mu", out["mu"])[k])
        for k, j in enumerate(neighbor_ids)
    }
    capacity = {
        j: float(out.get("c_mu", out["mu_range"])[k])
        for k, j in enumerate(neighbor_ids)
    }
    response_surface = {
        j: [float(value) for value in out["q_mu"][k]]
        for k, j in enumerate(neighbor_ids)
    }
    learned = {
        "direction": signed,
        "capacity": capacity,
        "q": response_surface,
        "signed": signed,
        "magnitude": {j: abs(value) for j, value in signed.items()},
        "range": capacity,
    }
    sigmas = {
        j: float(out.get("d_sigma", out["sigma"])[k])
        for k, j in enumerate(neighbor_ids)
    }
    metadata = {
        "dr_applied": bool(out.get("dr_applied", False)),
        "dr_applied_rows": int(out.get("dr_applied_rows", 0)),
        "dr_clipped_rows": int(out.get("dr_clipped_rows", 0)),
        "dr_raw_inverse_max": float(out.get("dr_raw_inverse_max", 0.0)),
        "dr_correction_mean": float(np.mean(out["dr_correction"])),
        "dr_weight_abs_mean": float(np.mean(np.abs(out["dr_weight"]))),
        "propensity_min": float(np.min(behaviour_obs)),
        "propensity_max": float(np.max(behaviour_obs)),
    }
    return learned, sigmas, metadata


def _h1_one_step_oracle_scores(tiny_env, step, ego, neighbor_ids,
                               candidate_actions):
    """Exact one-step V_pi minus V_uniform controlled intervention.

    All other agents retain their factual action.  The intervention changes
    only neighbour ``j`` at the current step, and each candidate is evaluated
    from the identical cloned RNG/environment state. Candidate outcomes are
    weighted by the logged target policy pi and contrasted with a uniform
    intervention q. With H=1 this matches the proxy target exactly and avoids
    the previous mismatch where the oracle repeated the forced action for H
    steps under a scripted future policy.
    """
    state = step["env_snapshot_before_step"]
    factual_actions = [int(a) for a in step["actions"]]
    ego = int(ego)

    tiny_env.restore_state(state)
    _, factual_rewards, _, _ = tiny_env.step(list(factual_actions))
    factual_reward = float(factual_rewards[ego])
    logged_reward = float(step["rewards"][ego])
    replay_error = abs(factual_reward - logged_reward)

    signed_scores = {}
    magnitude_scores = {}
    range_scores = {}
    response_surfaces = {}

    for j in neighbor_ids:
        candidate_returns = []
        for action in candidate_actions:
            intervened = list(factual_actions)
            intervened[int(j)] = int(action)
            tiny_env.restore_state(state)
            _, rewards, _, _ = tiny_env.step(intervened)
            candidate_returns.append(float(rewards[ego]))

        candidate_returns = np.asarray(candidate_returns, dtype=np.float64)
        response_surfaces[int(j)] = candidate_returns.tolist()
        policy = np.asarray(step["policy_probs"][int(j)], dtype=np.float64)
        policy = policy / np.clip(policy.sum(), 1e-12, None)
        if policy.shape != candidate_returns.shape:
            raise RuntimeError(
                "H1 oracle action support does not match the logged policy: "
                f"{candidate_returns.shape} versus {policy.shape}"
            )
        signed = float(
            np.dot(policy, candidate_returns) - np.mean(candidate_returns)
        )
        signed_scores[int(j)] = signed
        magnitude_scores[int(j)] = abs(signed)
        range_scores[int(j)] = float(
            np.max(candidate_returns) - np.min(candidate_returns)
        )

    tiny_env.restore_state(state)
    return {
        "direction": signed_scores,
        "capacity": range_scores,
        "q": response_surfaces,
        "signed": signed_scores,
        "magnitude": magnitude_scores,
        "range": range_scores,
    }, float(replay_error)


def _collect_h1_eval_steps(runner, n_states):
    """Collect held-out factual steps and policy-return endpoints.

    Episodes collected here are never added to proxy replay or used for an
    optimizer update. Their mean per-agent return is therefore a held-out
    endpoint that can be paired with the eps=0 arm to report the realised
    return cost of forcing.
    """
    requested = int(max(1, n_states))
    selected = []
    episode_returns_mean_per_agent = []
    attempts = 0

    while len(selected) < requested and attempts < 8:
        attempts += 1
        trajectory, episode_reward, _ = runner.collect_episode()
        if not trajectory:
            continue
        episode_returns_mean_per_agent.append(float(np.mean(
            np.asarray(episode_reward, dtype=np.float64)
        )))

        remaining = requested - len(selected)
        take = min(remaining, len(trajectory))
        indices = np.linspace(0, len(trajectory) - 1, num=take, dtype=int)
        for idx in dict.fromkeys(indices.tolist()):
            selected.append(trajectory[int(idx)])
            if len(selected) >= requested:
                break

    if len(selected) != requested:
        raise RuntimeError(
            f"H1 requested {requested} held-out states but collected "
            f"{len(selected)}"
        )
    return selected, {
        "heldout_policy_return_mean_per_agent": float(np.mean(
            episode_returns_mean_per_agent
        )),
        "heldout_policy_return_std_per_agent": float(np.std(
            episode_returns_mean_per_agent
        )),
        "heldout_policy_return_n_episodes": int(
            len(episode_returns_mean_per_agent)
        ),
        "policy_return_endpoint_measured": bool(
            episode_returns_mean_per_agent
        ),
    }


def _evaluate_h1_exact_protocol(runner, tiny_env, args, tiny_cfg):
    """Evaluate the confirmatory, estimand-aligned H1 protocol (H=1)."""
    if int(args.tiny_horizon) != 1 or int(runner.proxy.n_horizons) != 1:
        raise RuntimeError(
            "Confirmatory H1 requires tiny_horizon=1 and proxy_n_horizons=1; "
            f"received {args.tiny_horizon} and {runner.proxy.n_horizons}. "
            "Multi-step policy rollouts are exploratory until a common-policy "
            "clone protocol is implemented."
        )

    candidate_actions = _tiny_candidate_intervention_actions(tiny_env)
    runner.proxy.candidate_actions = list(candidate_actions)
    eval_steps, policy_return_metadata = _collect_h1_eval_steps(
        runner, int(args.tiny_states)
    )

    rows = []
    aggregate_rows = []
    dr_rows = 0
    dr_calls = 0
    dr_clipped_rows = 0
    dr_raw_inverse_maxes = []
    replay_errors = []
    propensity_mins = []
    propensity_maxes = []
    correction_means = []

    for state_idx, step in enumerate(eval_steps):
        if hasattr(tiny_env, "get_supported_egos"):
            egos = list(tiny_env.get_supported_egos())
        else:
            egos = list(range(int(tiny_env.n_agents)))

        for ego in egos:
            ego = int(ego)
            neighbor_ids = [
                j for j in range(int(tiny_env.n_agents)) if j != ego
            ]
            learned_scores, sigmas, score_meta = _score_h1_logged_step(
                runner, tiny_env, step, ego, neighbor_ids,
            )
            oracle_scores, replay_error = _h1_one_step_oracle_scores(
                tiny_env, step, ego, neighbor_ids, candidate_actions,
            )
            replay_errors.append(replay_error)
            propensity_mins.append(score_meta["propensity_min"])
            propensity_maxes.append(score_meta["propensity_max"])
            correction_means.append(score_meta["dr_correction_mean"])
            dr_rows += int(score_meta["dr_applied_rows"])
            dr_calls += int(score_meta["dr_applied"])
            dr_clipped_rows += int(score_meta["dr_clipped_rows"])
            dr_raw_inverse_maxes.append(score_meta["dr_raw_inverse_max"])

            cal = oracle_calibration(
                learned_scores=learned_scores["capacity"],
                oracle_scores=oracle_scores["capacity"],
                neighbor_ids=neighbor_ids,
            )
            signed_cal = _signed_calibration(
                proxy_signed=learned_scores["direction"],
                oracle_signed=oracle_scores["direction"],
                neighbor_ids=neighbor_ids,
            )
            q_cal = _response_surface_calibration(
                learned_scores["q"], oracle_scores["q"], neighbor_ids,
            )
            top_k = int(max(
                1, min(tiny_cfg.get("max_core_size", 4), len(neighbor_ids))
            ))
            core_f1 = oracle_core_f1_from_scores(
                learned_scores=learned_scores["capacity"],
                oracle_scores=oracle_scores["capacity"],
                neighbor_ids=neighbor_ids,
                top_k=top_k,
            )

            aggregate_row = {
                "seed": int(args.seed),
                "state_idx": int(state_idx),
                "ego_id": ego,
                "top_k": top_k,
                "oracle_core_f1": float(core_f1),
                "dr_applied": int(score_meta["dr_applied"]),
                "dr_applied_rows": int(score_meta["dr_applied_rows"]),
                "dr_clipped_rows": int(score_meta["dr_clipped_rows"]),
                "dr_raw_inverse_max": float(score_meta["dr_raw_inverse_max"]),
                "dr_correction_mean": score_meta["dr_correction_mean"],
                "replay_consistency_abs_error": replay_error,
            }
            aggregate_row.update(cal)
            aggregate_row.update(signed_cal)
            aggregate_row.update(q_cal)
            # Explicit Paper-A names.  The historical fields above remain for
            # compatibility, but H1a/H1b/H1c must not require readers to infer
            # which scalar is Q, C, or D.
            aggregate_row.update({
                "capacity_rank_correlation": cal["rank_correlation"],
                "capacity_mae": cal["mae"],
                "capacity_bias": cal["bias"],
                "direction_spearman": signed_cal["signed_spearman"],
                "direction_mae": signed_cal["signed_mae"],
                "direction_bias": signed_cal["signed_bias"],
                "direction_sign_agreement": signed_cal["sign_agreement"],
            })
            # Retain the historical column names for collectors while their
            # meaning is now explicitly C recovery (the action-response
            # range), not a separate realised-action baseline.
            aggregate_row.update({f"range_{k}": v for k, v in cal.items()})
            aggregate_rows.append(aggregate_row)

            for j in neighbor_ids:
                rows.append({
                    "seed": int(args.seed),
                    "state_idx": int(state_idx),
                    "ego_id": ego,
                    "neighbor_id": int(j),
                    "learned_score": float(learned_scores["capacity"][j]),
                    "oracle_score": float(oracle_scores["capacity"][j]),
                    "learned_signed": float(learned_scores["direction"][j]),
                    "oracle_signed": float(oracle_scores["direction"][j]),
                    "learned_range": float(learned_scores["capacity"][j]),
                    "oracle_range": float(oracle_scores["capacity"][j]),
                    "abs_error": float(abs(
                        learned_scores["capacity"][j]
                        - oracle_scores["capacity"][j]
                    )),
                    "signed_error": float(
                        learned_scores["signed"][j]
                        - oracle_scores["signed"][j]
                    ),
                    "proxy_sigma": float(sigmas[j]),
                    "dr_applied": int(score_meta["dr_applied"]),
                })

    dr_requested = bool(runner.proxy.use_doubly_robust)
    coverage = runner.proxy.get_action_coverage_diagnostics()
    forcing_stats = runner.forcer.get_stats()
    metadata = {
        "protocol_name": "logged_one_step_policy_contrast_v1",
        "protocol_match": True,
        "confirmatory_horizon": 1,
        "exploratory_h3_reported": False,
        "context_source": "action_time_trajectory_cache",
        "oracle_baseline": "uniform_action_policy",
        "oracle_intervention": "V_pi_minus_V_uniform_current_action",
        "factual_replay_role": "integrity_check_and_AIPW_observed_outcome",
        "nuisance_training_score_mode": "plugin_fixed_across_ablation",
        "candidate_actions": list(candidate_actions),
        "candidate_action_count": int(len(candidate_actions)),
        "proxy_action_count": int(runner.proxy.action_dim),
        "dr_requested": dr_requested,
        "dr_applied_calls_eval": int(dr_calls),
        "dr_applied_rows_eval": int(dr_rows),
        "dr_clipped_rows_eval": int(dr_clipped_rows),
        "dr_clipping_fraction_eval": float(
            dr_clipped_rows / max(1, dr_rows)
        ),
        "dr_raw_inverse_max_eval": float(
            max(dr_raw_inverse_maxes, default=0.0)
        ),
        "proxy_iw_clip": float(runner.proxy.iw_clip),
        "dr_exercised": bool((not dr_requested) or dr_rows > 0),
        "dr_clipping_absent": bool((not dr_requested) or dr_clipped_rows == 0),
        "replay_consistency_max_abs_error": float(max(replay_errors, default=0.0)),
        "propensity_min_eval": float(min(propensity_mins, default=float("nan"))),
        "propensity_max_eval": float(max(propensity_maxes, default=float("nan"))),
        "dr_correction_mean_eval": float(np.mean(correction_means)) if correction_means else 0.0,
        "action_coverage": coverage,
        "actions_seen": int(coverage["actions_seen"]),
        "min_action_fraction": float(coverage["min_action_fraction"]),
        "forced_actions_seen": int(coverage["forced_actions_seen"]),
        "min_forced_action_fraction": float(
            coverage["min_forced_action_fraction"]
        ),
        "n_forced_proxy_samples": int(coverage["n_forced_samples"]),
        "forcing_stats": forcing_stats,
        "realised_forcing_rate": float(
            forcing_stats["realised_forcing_rate"]
        ),
        "forcing_total_agent_steps": int(
            forcing_stats["total_agent_steps"]
        ),
        "forcing_total_forced": int(forcing_stats["total_forced"]),
        **policy_return_metadata,
    }
    metadata["alignment_protocol_gate_pass"] = bool(
        metadata["protocol_match"]
        and metadata["candidate_action_count"] == metadata["proxy_action_count"]
        and metadata["dr_exercised"]
        and metadata["replay_consistency_max_abs_error"] <= 1e-6
        and metadata["propensity_min_eval"] > 0.0
    )
    metadata["action_coverage_gate_pass"] = bool(
        coverage["actions_seen"] == runner.proxy.action_dim
    )
    # Backward-compatible process gate. Empirical coverage is scientific
    # evidence, not a protocol-integrity condition: in particular, eps=0 is
    # intentionally allowed to expose missing action support and must still be
    # aggregatable as the observational control.
    metadata["protocol_gate_pass"] = metadata["alignment_protocol_gate_pass"]
    return rows, aggregate_rows, metadata


def _evaluate_legacy_tiny_protocol(runner, tiny_env, args, tiny_cfg):
    """Retain the historical state-bank diagnostic for non-H1 CLI callers.

    This path is explicitly marked as non-confirmatory because its restored
    environment state is not accompanied by matching relational caches or an
    observed outcome/propensity for the AIPW correction.
    """
    try:
        state_bank = tiny_env.sample_state_bank(
            n_states=int(args.tiny_states), burn_in=int(args.tiny_burn_in),
        )
    except TypeError:
        state_bank = tiny_env.sample_state_bank(
            int(args.tiny_states), int(args.tiny_burn_in),
        )

    rows = []
    aggregate_rows = []
    for state_idx, state in enumerate(state_bank):
        tiny_env.restore_state(state)
        egos = (
            list(tiny_env.get_supported_egos())
            if hasattr(tiny_env, "get_supported_egos")
            else list(range(int(tiny_env.n_agents)))
        )
        for ego in egos:
            ego = int(ego)
            neighbor_ids = [j for j in range(int(tiny_env.n_agents)) if j != ego]
            learned_scores, sigmas = _score_learned_proxy_for_state(
                runner, tiny_env, state, ego, neighbor_ids,
            )
            oracle_scores = _compute_tiny_oracle_scores(
                tiny_env, state, ego, neighbor_ids, tiny_cfg,
                int(args.tiny_horizon),
            )
            cal = oracle_calibration(
                learned_scores["magnitude"], oracle_scores["magnitude"],
                neighbor_ids,
            )
            signed_cal = _signed_calibration(
                learned_scores["signed"], oracle_scores["signed"], neighbor_ids,
            )
            range_cal = oracle_calibration(
                learned_scores["range"], oracle_scores["range"], neighbor_ids,
            )
            top_k = int(max(
                1, min(tiny_cfg.get("max_core_size", 4), len(neighbor_ids))
            ))
            core_f1 = oracle_core_f1_from_scores(
                learned_scores["magnitude"], oracle_scores["magnitude"],
                neighbor_ids, top_k,
            )
            aggregate_row = {
                "seed": int(args.seed), "state_idx": int(state_idx),
                "ego_id": ego, "top_k": top_k,
                "oracle_core_f1": float(core_f1),
            }
            aggregate_row.update(cal)
            aggregate_row.update(signed_cal)
            aggregate_row.update({f"range_{k}": v for k, v in range_cal.items()})
            aggregate_rows.append(aggregate_row)

            for j in neighbor_ids:
                rows.append({
                    "seed": int(args.seed), "state_idx": int(state_idx),
                    "ego_id": ego, "neighbor_id": int(j),
                    "learned_score": float(learned_scores["magnitude"].get(j, 0.0)),
                    "oracle_score": float(oracle_scores["magnitude"].get(j, 0.0)),
                    "learned_signed": float(learned_scores["signed"].get(j, 0.0)),
                    "oracle_signed": float(oracle_scores["signed"].get(j, 0.0)),
                    "learned_range": float(learned_scores["range"].get(j, 0.0)),
                    "oracle_range": float(oracle_scores["range"].get(j, 0.0)),
                    "abs_error": float(abs(
                        learned_scores["magnitude"].get(j, 0.0)
                        - oracle_scores["magnitude"].get(j, 0.0)
                    )),
                    "signed_error": float(
                        learned_scores["signed"].get(j, 0.0)
                        - oracle_scores["signed"].get(j, 0.0)
                    ),
                    "proxy_sigma": float(sigmas.get(j, 0.0)),
                })

    metadata = {
        "protocol_name": "legacy_scripted_state_bank",
        "protocol_match": False,
        "alignment_protocol_gate_pass": False,
        "action_coverage_gate_pass": False,
        "protocol_gate_pass": False,
        "confirmatory_horizon": None,
        "dr_requested": bool(runner.proxy.use_doubly_robust),
        "dr_applied_calls_eval": 0,
        "dr_applied_rows_eval": 0,
        "dr_exercised": not bool(runner.proxy.use_doubly_robust),
        "protocol_warning": (
            "Legacy diagnostic only: action-time contexts and AIPW inputs are "
            "not aligned. Do not use these metrics for H1."
        ),
        "action_coverage": runner.proxy.get_action_coverage_diagnostics(),
    }
    return rows, aggregate_rows, metadata


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

    if bool(getattr(args, "h1_exact_protocol", False)):
        rows, aggregate_rows, protocol_metadata = _evaluate_h1_exact_protocol(
            runner, tiny_env, args, tiny_cfg,
        )
    else:
        rows, aggregate_rows, protocol_metadata = _evaluate_legacy_tiny_protocol(
            runner, tiny_env, args, tiny_cfg,
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
            "Oracle intervention scores are used only for held-out post-hoc "
            "evaluation, never as policy or proxy training labels. Confirmatory "
            "H1 additionally requires protocol_match=true and "
            "protocol_gate_pass=true."
        ),
    }
    summary.update(protocol_metadata)

    numeric_keys = [
        "bias",
        "variance",
        "mae",
        "rmse",
        "rank_correlation",
        "p_value",
        "constant_case",
        "oracle_core_f1",
        "signed_bias",
        "signed_mae",
        "signed_rmse",
        "signed_spearman",
        "signed_p_value",
        "signed_constant_case",
        "sign_agreement",
        "range_rank_correlation",
        "q_mae",
        "q_rmse",
        "q_spearman",
        "capacity_rank_correlation",
        "capacity_mae",
        "capacity_bias",
        "direction_spearman",
        "direction_mae",
        "direction_bias",
        "direction_sign_agreement",
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
    print(f"signed_spearman_mean={summary.get('signed_spearman_mean', 0.0):.4f}")
    print(f"sign_agreement_mean={summary.get('sign_agreement_mean', 0.0):.4f}")
    print(f"oracle_core_f1_mean={summary.get('oracle_core_f1_mean', 0.0):.4f}")
    print(f"mae_mean={summary.get('mae_mean', 0.0):.4f}")
    print(f"protocol={summary.get('protocol_name', 'unknown')}")
    print(f"protocol_gate_pass={summary.get('protocol_gate_pass', False)}")
    print(f"dr_applied_rows_eval={summary.get('dr_applied_rows_eval', 0)}")
    print(f"Saved tiny oracle results to: {out_dir}")

    if bool(getattr(args, "h1_exact_protocol", False)) and not bool(
        summary.get("protocol_gate_pass", False)
    ):
        raise RuntimeError(
            "H1 protocol gate failed; the saved metrics are diagnostic only and "
            "must not be aggregated as confirmatory evidence."
        )

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
    parser.add_argument("--tiny_proxy_train_episodes", type=int, default=40)
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
