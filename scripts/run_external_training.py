"""Run matched project models on one normalized external benchmark.

Protocol v5 separates *training* from the model-comparison estimand used by
G8.  Final-CIGAMF deliberately executes epsilon-forced actions while training
to acquire causal support; PureMeanField does not.  Training return is thus a
useful diagnostic but not a fair learned-policy comparator.  After training,
every model is evaluated on fresh paired environment seeds with policy
learning, representation-training updates, and epsilon forcing disabled while
deployment-time recurrent inference remains active.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.external.runtime import ensure_directory, maybe_reexec_in_external_runtime
from utils.paper_contracts import (
    EXTERNAL_EVAL_SEED_OFFSET,
    EXTERNAL_G8_MIN_EVAL_EPISODES,
    EXTERNAL_GENERALIZATION_PROTOCOL_VERSION,
)

ENVIRONMENTS = ("cityflow", "cyborg", "flatland", "rware")
MODELS = ("Final-CIGAMF", "PureMeanField", "FullExplicitLocal")


def _finite_mean(values):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def _finite_std(values):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(vals, ddof=0)) if vals else None


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_head(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_is_clean(path):
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        text=True, capture_output=True, check=False,
    )
    return bool(result.returncode == 0 and not result.stdout.strip())


def _expected_external_revision(repo_path):
    revisions = ROOT / "scripts" / "external_env_revisions.tsv"
    if not revisions.is_file():
        return None
    repo_name = Path(repo_path).name
    for raw in revisions.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] == repo_name:
            return parts[1]
    return None


def _cfg_for(RE, profile, seed, env, max_steps):
    """Return the frozen external training configuration.

    ``full`` now really means the paper/default training configuration.  The
    old harness silently forced ``k0_warmup=0`` and shortened both causal
    horizons even in full mode; those smoke conveniences are retained only in
    ``quick``.  Population-size constraints remain environment dependent.
    """
    cfg = RE.default_cfg() if profile == "full" else RE.smoke_cfg()
    cfg["seed"] = int(seed)

    if profile == "quick":
        quick_horizon = min(4, max(1, int(max_steps) // 4))
        cfg.update({
            "k0_warmup": 0,
            "causal_horizon": quick_horizon,
            "proxy_n_horizons": quick_horizon,
        })

    cfg.update({
        "max_core_size": min(3, max(1, env.n_agents - 1)),
        "min_core_size": 1,
        "seed_core_top_k": min(3, max(1, env.n_agents - 1)),
        "belief_adaptive_k_min": 1,
        "periph_require_full_signature": True,
        "periph_allow_legacy_items": False,
    })
    return cfg


def _series_value(history, key, index, default=None):
    values = history.get(key, []) if isinstance(history, dict) else []
    if index < len(values):
        value = values[index]
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return default


def _forcing_fraction(runner):
    forcer = getattr(runner, "forcer", None)
    if forcer is None:
        return 0.0
    eligible = int(getattr(forcer, "total_eligible", 0) or 0)
    forced = int(getattr(forcer, "total_forced", 0) or 0)
    return float(forced) / float(eligible) if eligible > 0 else 0.0


def _forcer_snapshot(forcer):
    if forcer is None:
        return None
    state_dict = getattr(forcer, "state_dict", None)
    if callable(state_dict):
        return {
            "kind": "state_dict",
            "state": copy.deepcopy(state_dict()),
            "last_execution_records": copy.deepcopy(
                getattr(forcer, "last_execution_records", ())
            ),
        }
    eps_per_agent = getattr(forcer, "_eps_per_agent", None)
    if isinstance(eps_per_agent, np.ndarray):
        eps_per_agent = eps_per_agent.copy()
    return {
        "kind": "attributes",
        "eps_initial": float(getattr(forcer, "eps_initial", 0.0)),
        "eps": float(getattr(forcer, "eps", 0.0)),
        "anneal_to": getattr(forcer, "anneal_to", None),
        "episode": int(getattr(forcer, "episode", 0)),
        "_eps_per_agent": eps_per_agent,
        "total_steps": int(getattr(forcer, "total_steps", 0)),
        "total_forced": int(getattr(forcer, "total_forced", 0)),
        "total_eligible": int(getattr(forcer, "total_eligible", 0)),
        "last_execution_records": getattr(forcer, "last_execution_records", ()),
    }


def _set_forcer_off(forcer):
    if forcer is None:
        return
    forcer.eps_initial = 0.0
    forcer.eps = 0.0
    forcer.anneal_to = 0.0
    forcer._eps_per_agent = None


def _restore_forcer(forcer, snapshot):
    if forcer is None or snapshot is None:
        return
    if snapshot.get("kind") == "state_dict":
        load_state_dict = getattr(forcer, "load_state_dict", None)
        if not callable(load_state_dict):
            raise RuntimeError("forcer snapshot requires load_state_dict()")
        load_state_dict(copy.deepcopy(snapshot["state"]))
        forcer.last_execution_records = copy.deepcopy(
            snapshot.get("last_execution_records", ())
        )
        return
    for key, value in snapshot.items():
        if key == "kind":
            continue
        setattr(forcer, key, value)


def _evaluation_seed(train_seed, eval_index):
    # Deterministic, prespecified and disjoint from ordinary small training
    # seeds.  The exact rule is written into the manifest and independently
    # checked by G8.
    return int(EXTERNAL_EVAL_SEED_OFFSET + int(train_seed) * 10_000 + int(eval_index))


def _clone_tensor_mapping(mapping):
    out = {}
    for key, value in dict(mapping or {}).items():
        if hasattr(value, "detach") and callable(value.detach):
            out[key] = value.detach().clone()
        else:
            out[key] = copy.deepcopy(value)
    return out


def _capture_external_inference_state(runner):
    """Capture mutable deployment state that a rollout may advance.

    Frozen-policy evaluation freezes *learning*, not recurrent inference.  A
    recurrent policy is therefore allowed to update z_ij/s_ij within an
    episode, but every fresh evaluation episode must start from the same
    trained checkpoint.  This snapshot intentionally excludes environment
    state because every evaluation episode gets a freshly constructed env.
    """
    state = {
        "interaction_step": copy.deepcopy(getattr(runner, "_interaction_step", None)),
        "attrs": {},
        "pair": None,
    }
    for name in (
        "belief_modules",
        "sig_tracker",
        "candidate_neighbors_by_ego",
        "candidate_epoch",
        "candidate_map_hash",
        "measured_edge_count",
        "candidate_construction_subquadratic",
        "candidate_construction_linear_candidate",
        "feature_snapshot_subquadratic",
        "feature_snapshot_linear_candidate",
        "_last_candidate_refresh_step",
        "_candidate_telemetry",
        "_candidate_refresh_totals",
        "_profile_update_step",
    ):
        if hasattr(runner, name):
            state["attrs"][name] = copy.deepcopy(getattr(runner, name))

    pair = getattr(runner, "pair_rel_module", None)
    if pair is not None:
        state["pair"] = {
            "full_states": _clone_tensor_mapping(getattr(pair, "full_states", {})),
            "shadow_states": _clone_tensor_mapping(getattr(pair, "shadow_states", {})),
            "pooled_states": _clone_tensor_mapping(getattr(pair, "pooled_states", {})),
            "active_core_pairs": set(getattr(pair, "active_core_pairs", set())),
            "candidate_neighbors_by_ego": copy.deepcopy(
                getattr(pair, "candidate_neighbors_by_ego", {})
            ),
        }
    return state


def _restore_external_inference_state(runner, state):
    if state.get("interaction_step") is not None and hasattr(runner, "_interaction_step"):
        runner._interaction_step = copy.deepcopy(state["interaction_step"])
    for name, value in state.get("attrs", {}).items():
        setattr(runner, name, copy.deepcopy(value))

    pair_state = state.get("pair")
    pair = getattr(runner, "pair_rel_module", None)
    if pair is not None and pair_state is not None:
        device = getattr(runner, "device", None)

        def _restore_map(mapping):
            out = _clone_tensor_mapping(mapping)
            if device is not None:
                for key, value in list(out.items()):
                    if hasattr(value, "to") and callable(value.to):
                        out[key] = value.to(device)
            return out

        pair.full_states = _restore_map(pair_state["full_states"])
        pair.shadow_states = _restore_map(pair_state["shadow_states"])
        pair.pooled_states = _restore_map(pair_state["pooled_states"])
        pair.active_core_pairs = set(pair_state["active_core_pairs"])
        pair.candidate_neighbors_by_ego = copy.deepcopy(
            pair_state["candidate_neighbors_by_ego"]
        )
        # The external evaluator freezes representation-learning state, so the
        # potentially large BC replay buffer is guaranteed read-only.  Do not
        # clone it once per fresh eval episode: that would turn a 20-episode
        # evaluation into repeated O(buffer_size) memory copies for no change
        # in policy semantics.

    reset_caches = getattr(runner, "_reset_exclusion_caches", None)
    if callable(reset_caches):
        reset_caches()


def _evaluate_frozen_policy(
    *, runner, model, train_seed, eval_episodes, environment, agent_count,
    max_steps, config_path, config_fingerprint, RE, build_environment,
    require_panel, resolve_env_adapter,
):
    """Evaluate a trained runner without learning or intervention noise.

    Evaluation calls ``collect_episode`` directly: no policy optimizer,
    replay/proxy push, graph update, scheduler step, or semantic recalibration
    is executed.  Final-CIGAMF keeps recurrent deployment inference active but
    freezes representation-training state and hard-disables epsilon forcing.
    Every fresh episode restores the same trained inference checkpoint before
    entering its paired environment seed.
    """
    original_env = getattr(runner, "env", None)
    original_adapter = getattr(runner, "env_adapter", None)
    cfg = getattr(runner, "cfg", None)
    if not isinstance(cfg, dict):
        raise TypeError("external runner must expose a mutable cfg dictionary")

    sentinel = object()
    old_freeze_rep = cfg.get("freeze_representation_state", sentinel)
    old_freeze_rep_learning = cfg.get(
        "freeze_representation_learning_state", sentinel
    )
    # Recurrent z/s filtering is part of deployment inference and must stay
    # active.  Only representation *training-data* updates are frozen.
    cfg["freeze_representation_state"] = False
    cfg["freeze_representation_learning_state"] = True

    original_interaction_step = getattr(runner, "_interaction_step", sentinel)
    forcer = getattr(runner, "forcer", None)
    forcer_state = _forcer_snapshot(forcer)
    _set_forcer_off(forcer)
    inference_state = _capture_external_inference_state(runner)

    rows = []
    try:
        base_interaction_step = (
            int(original_interaction_step)
            if original_interaction_step is not sentinel else None
        )
        for eval_index in range(int(eval_episodes)):
            _restore_external_inference_state(runner, inference_state)
            eval_seed = _evaluation_seed(train_seed, eval_index)
            RE.set_global_seed(eval_seed)
            eval_env = build_environment(
                environment,
                seed=eval_seed,
                n_agents=agent_count,
                max_steps=max_steps,
                config_path=config_path,
            )
            require_panel(eval_env, "training")
            if int(eval_env.n_agents) != int(runner.n_agents):
                raise RuntimeError("external evaluation changed population size")
            if int(eval_env.get_action_dim()) != int(runner.action_dim):
                raise RuntimeError("external evaluation changed action dimension")
            if int(eval_env.get_obs_dim()) != int(runner.obs_dim):
                raise RuntimeError("external evaluation changed observation dimension")

            runner.env = eval_env
            runner.env_adapter = resolve_env_adapter(
                eval_env, action_dim=int(runner.action_dim)
            )
            if base_interaction_step is not None:
                runner._interaction_step = int(base_interaction_step)

            # collect_episode is rollout-only.  Any forcing observed here is a
            # protocol violation, not a benign diagnostic.
            trajectory, episode_reward, runtime = runner.collect_episode()
            forced_count = 0
            for step in trajectory:
                mask = np.asarray(step.get("forced_mask", []), dtype=bool)
                forced_count += int(mask.sum()) if mask.size else 0
            if forced_count != 0:
                raise RuntimeError(
                    "frozen external evaluation executed epsilon-forced actions"
                )

            reward = float(np.mean(np.asarray(episode_reward, dtype=np.float64)))
            rows.append({
                "environment": environment,
                "model": model,
                "train_seed": int(train_seed),
                "eval_index": int(eval_index),
                "eval_seed": int(eval_seed),
                "eval_reward": reward,
                "episode_steps": int(len(trajectory)),
                "max_steps_requested": int(max_steps),
                "max_steps_effective": int(getattr(eval_env, "max_steps", max_steps)),
                "forcing_count": 0,
                "rollout_seconds": float(runtime),
                "config_fingerprint_sha256": str(config_fingerprint),
            })
    finally:
        _restore_external_inference_state(runner, inference_state)
        runner.env = original_env
        runner.env_adapter = original_adapter
        if old_freeze_rep is sentinel:
            cfg.pop("freeze_representation_state", None)
        else:
            cfg["freeze_representation_state"] = old_freeze_rep
        if old_freeze_rep_learning is sentinel:
            cfg.pop("freeze_representation_learning_state", None)
        else:
            cfg["freeze_representation_learning_state"] = old_freeze_rep_learning
        if original_interaction_step is not sentinel:
            runner._interaction_step = original_interaction_step
        _restore_forcer(forcer, forcer_state)

    return rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty external artifact: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["Final-CIGAMF"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--agent-count", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--config-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--profile", choices=("full", "quick"), default="full",
        help="full uses paper configurations; quick is wiring-only smoke",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    maybe_reexec_in_external_runtime(require_ready=True)

    import run_experiment as RE
    from envs.causal_adapter import resolve_env_adapter
    from envs.external.registry import build_environment, repo_path
    from envs.external.runtime import runtime_metadata
    from envs.external_contract import require_panel
    from scripts.run_external_suite import _h1_smoke

    if args.episodes <= 0 or args.max_steps <= 0 or not args.seeds:
        parser.error("episodes, max-steps and seeds must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if len(set(args.models)) != len(args.models):
        parser.error("models must be unique")

    eval_episodes = (
        int(args.eval_episodes)
        if args.eval_episodes is not None
        else (EXTERNAL_G8_MIN_EVAL_EPISODES if args.profile == "full" else 2)
    )
    if eval_episodes <= 0:
        parser.error("eval-episodes must be positive")

    migrated = ensure_directory(args.out)
    if migrated is not None:
        print(f"[EXTERNAL-OUT] preserved legacy output file as {migrated}", file=sys.stderr)

    rows = []
    training_episode_rows = []
    evaluation_rows = []
    seed_config_fingerprints = {}
    h1_support_by_seed = {}

    for seed in args.seeds:
        # Diagnostic support is measured on its own fresh environment so model
        # training cannot create or suppress the benchmark's H1 support.
        support_env = build_environment(
            args.environment, seed=seed, n_agents=args.agent_count,
            max_steps=args.max_steps, config_path=args.config_path,
        )
        require_panel(support_env, "training")
        if support_env.capabilities.supports("h1"):
            require_panel(support_env, "h1")
            h1_support_by_seed[str(int(seed))] = _h1_smoke(support_env)
        else:
            h1_support_by_seed[str(int(seed))] = {
                "interface_ready": False,
                "signal_ready": False,
                "status": "H1_ORACLE_UNAVAILABLE",
                "reason": "adapter declares training-only capability",
            }
        del support_env

        for model in args.models:
            # Every seed/model cell is reconstructed from the same benchmark
            # seed. Reusing a mutated env would confound model order with state.
            RE.set_global_seed(int(seed))
            env = build_environment(
                args.environment, seed=seed, n_agents=args.agent_count,
                max_steps=args.max_steps, config_path=args.config_path,
            )
            require_panel(env, "training")
            cfg = _cfg_for(RE, args.profile, seed, env, args.max_steps)
            fp = _fingerprint(cfg)
            seed_config_fingerprints[f"{int(seed)}:{model}"] = fp
            runner = RE.make_runner(model, env, cfg, args.device)

            started = time.perf_counter()
            history = runner.run(n_episodes=args.episodes, eval_every=1)
            elapsed = time.perf_counter() - started
            reward_history = history.get("mean_reward", [])
            if len(reward_history) != int(args.episodes):
                raise RuntimeError(
                    f"{model}/seed={seed} did not retain one training reward per episode"
                )

            for index, reward in enumerate(reward_history):
                training_episode_rows.append({
                    "environment": args.environment,
                    "model": model,
                    "seed": int(seed),
                    "episode": int(index + 1),
                    "training_reward": float(reward),
                    "stage": _series_value(history, "stage", index),
                    "mean_core_size": _series_value(history, "mean_core_size", index),
                    "policy_loss": _series_value(history, "policy_loss", index),
                    "proxy_loss": _series_value(history, "proxy_loss", index),
                    "config_fingerprint_sha256": fp,
                })

            model_eval_rows = _evaluate_frozen_policy(
                runner=runner,
                model=model,
                train_seed=seed,
                eval_episodes=eval_episodes,
                environment=args.environment,
                agent_count=args.agent_count,
                max_steps=args.max_steps,
                config_path=args.config_path,
                config_fingerprint=fp,
                RE=RE,
                build_environment=build_environment,
                require_panel=require_panel,
                resolve_env_adapter=resolve_env_adapter,
            )
            evaluation_rows.extend(model_eval_rows)
            eval_rewards = [float(item["eval_reward"]) for item in model_eval_rows]

            rows.append({
                "environment": args.environment,
                "model": model,
                "profile": args.profile,
                "seed": int(seed),
                "episodes": int(args.episodes),
                "eval_episodes": int(eval_episodes),
                "n_agents": int(env.n_agents),
                "action_dim": int(env.get_action_dim()),
                "obs_dim": int(env.get_obs_dim()),
                "max_steps_requested": int(args.max_steps),
                "max_steps_effective": int(getattr(env, "max_steps", args.max_steps)),
                "mean_reward": _finite_mean(reward_history),
                "final_reward": float(reward_history[-1]) if reward_history else float("nan"),
                "eval_mean_reward": _finite_mean(eval_rewards),
                "eval_reward_std": _finite_std(eval_rewards),
                "training_forcing_fraction": _forcing_fraction(runner),
                "mean_core_size": _finite_mean(history.get("mean_core_size", [])),
                "throughput_total": _finite_mean(history.get("throughput_total_agent_steps_per_sec", [])),
                "elapsed_seconds": elapsed,
                "config_fingerprint_sha256": fp,
            })

    csv_path = args.out / "summary_external_training.csv"
    episode_path = args.out / "external_training_episodes.csv"
    evaluation_path = args.out / "external_frozen_evaluation.csv"
    _write_csv(csv_path, rows)
    _write_csv(episode_path, training_episode_rows)
    _write_csv(evaluation_path, evaluation_rows)

    source_head = _git_head(ROOT)
    external_repo = repo_path(args.environment)
    external_head = _git_head(external_repo)
    external_expected = _expected_external_revision(external_repo)
    external_pin_match = bool(
        external_head and external_expected and external_head == external_expected
    )
    config_path = Path(args.config_path).resolve() if args.config_path else None
    config_hash = (
        _sha256_file(config_path)
        if config_path is not None and config_path.is_file() else None
    )
    paired_models_present = {"Final-CIGAMF", "PureMeanField"}.issubset(set(args.models))
    source_clean = _git_is_clean(ROOT)

    manifest = {
        "protocol_version": EXTERNAL_GENERALIZATION_PROTOCOL_VERSION,
        "experiment": "external_matched_training_v5_recurrent_frozen_policy_eval",
        "environment": args.environment,
        "models": list(args.models),
        "seeds": [int(seed) for seed in args.seeds],
        "episodes": int(args.episodes),
        "eval_episodes": int(eval_episodes),
        "evaluation_mode": (
            "fresh_seed_frozen_policy_no_learning_no_forcing_recurrent_inference"
        ),
        "evaluation_recurrent_inference_active": True,
        "evaluation_representation_learning_state_frozen": True,
        "evaluation_representation_state_frozen": False,
        "evaluation_seed_offset": int(EXTERNAL_EVAL_SEED_OFFSET),
        "evaluation_seed_rule": "offset + train_seed*10000 + eval_index",
        "agent_count_requested": int(args.agent_count),
        "max_steps_requested": int(args.max_steps),
        "profile": args.profile,
        "device": args.device,
        "summary": str(csv_path),
        "summary_row_count": int(len(rows)),
        "summary_sha256": _sha256_file(csv_path),
        "training_episode_artifact": str(episode_path),
        "training_episode_row_count": int(len(training_episode_rows)),
        "training_episode_sha256": _sha256_file(episode_path),
        "evaluation_artifact": str(evaluation_path),
        "evaluation_row_count": int(len(evaluation_rows)),
        "evaluation_sha256": _sha256_file(evaluation_path),
        "source_git_head": source_head,
        "source_git_clean": source_clean,
        "external_repo": str(external_repo),
        "external_repo_head": external_head,
        "external_repo_expected_revision": external_expected,
        "external_pin_match": external_pin_match,
        "external_runtime": runtime_metadata(),
        "config_path": str(config_path) if config_path else None,
        "config_sha256": config_hash,
        "seed_model_config_fingerprints_sha256": seed_config_fingerprints,
        "h1_support_by_seed": h1_support_by_seed,
        "paired_generalization_models_present": paired_models_present,
        "paper_evidence_scope": (
            "frozen_policy_reward_generalization_candidate" if all(
                bool(item.get("signal_ready", False))
                for item in h1_support_by_seed.values()
            ) else "architecture_generalization_only"
        ),
        "not_an_h2_or_latency_claim": True,
        "not_an_external_allocation_selector_claim": True,
        "provenance_complete": bool(
            source_head and source_clean and external_pin_match
            and all(seed_config_fingerprints.values())
            and (config_path is None or bool(config_hash))
            and len(_sha256_file(csv_path)) == 64
            and len(_sha256_file(episode_path)) == 64
            and len(_sha256_file(evaluation_path)) == 64
        ),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(csv_path)


if __name__ == "__main__":
    main()
