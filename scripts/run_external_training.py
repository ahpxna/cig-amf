"""Run matched project models on one normalized external benchmark."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, subprocess, time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from envs.external.runtime import ensure_directory, maybe_reexec_in_external_runtime

ENVIRONMENTS = ("cityflow", "cyborg", "flatland", "rware")
MODELS = ("Final-CIGAMF", "PureMeanField", "FullExplicitLocal")


def _finite_mean(values):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


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
    cfg = RE.default_cfg() if profile == "full" else RE.smoke_cfg()
    cfg.update({
        "seed": int(seed), "k0_warmup": 0,
        "causal_horizon": min(4, max(1, int(max_steps) // 4)),
        "proxy_n_horizons": min(4, max(1, int(max_steps) // 4)),
        "max_core_size": min(3, max(1, env.n_agents - 1)),
        "min_core_size": 1,
        "seed_core_top_k": min(3, max(1, env.n_agents - 1)),
        "belief_adaptive_k_min": 1,
        "periph_require_full_signature": True,
        "periph_allow_legacy_items": False,
    })
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["Final-CIGAMF"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes", type=int, default=5)
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
    migrated = ensure_directory(args.out)
    if migrated is not None:
        print(f"[EXTERNAL-OUT] preserved legacy output file as {migrated}", file=sys.stderr)

    rows = []
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
            # A training-only adapter may validate architecture wiring, but it
            # cannot contribute causal-response evidence to G8.
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
            # seed.  Reusing a mutated env would confound model order with state.
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
            rows.append({
                "environment": args.environment,
                "model": model,
                "profile": args.profile,
                "seed": int(seed),
                "episodes": int(args.episodes),
                "n_agents": int(env.n_agents),
                "action_dim": int(env.get_action_dim()),
                "obs_dim": int(env.get_obs_dim()),
                "max_steps_requested": int(args.max_steps),
                "max_steps_effective": int(getattr(env, "max_steps", args.max_steps)),
                "mean_reward": _finite_mean(reward_history),
                "final_reward": float(reward_history[-1]) if reward_history else float("nan"),
                "mean_core_size": _finite_mean(history.get("mean_core_size", [])),
                "throughput_total": _finite_mean(history.get("throughput_total_agent_steps_per_sec", [])),
                "elapsed_seconds": elapsed,
                "config_fingerprint_sha256": fp,
            })

    csv_path = args.out / "summary_external_training.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    source_head = _git_head(ROOT)
    external_repo = repo_path(args.environment)
    external_head = _git_head(external_repo)
    external_expected = _expected_external_revision(external_repo)
    external_pin_match = bool(
        external_head and external_expected and external_head == external_expected
    )
    config_path = Path(args.config_path).resolve() if args.config_path else None
    paired_models_present = {"Final-CIGAMF", "PureMeanField"}.issubset(set(args.models))
    manifest = {
        "experiment": "external_matched_training_v3_generalization_gate",
        "environment": args.environment,
        "models": list(args.models),
        "seeds": [int(seed) for seed in args.seeds],
        "episodes": int(args.episodes),
        "agent_count_requested": int(args.agent_count),
        "max_steps_requested": int(args.max_steps),
        "profile": args.profile,
        "device": args.device,
        "summary": str(csv_path),
        "source_git_head": source_head,
        "source_git_clean": _git_is_clean(ROOT),
        "external_repo": str(external_repo),
        "external_repo_head": external_head,
        "external_repo_expected_revision": external_expected,
        "external_pin_match": external_pin_match,
        "external_runtime": runtime_metadata(),
        "config_path": str(config_path) if config_path else None,
        "config_sha256": _sha256_file(config_path) if config_path is not None and config_path.is_file() else None,
        "seed_model_config_fingerprints_sha256": seed_config_fingerprints,
        "h1_support_by_seed": h1_support_by_seed,
        "paired_generalization_models_present": paired_models_present,
        "paper_evidence_scope": (
            "causal_generalization_candidate" if all(
                bool(item.get("signal_ready", False))
                for item in h1_support_by_seed.values()
            ) else "architecture_generalization_only"
        ),
        "not_an_h2_or_latency_claim": True,
        "provenance_complete": bool(
            source_head and _git_is_clean(ROOT) and external_pin_match
            and all(seed_config_fingerprints.values())
            and (config_path is None or (config_path.is_file() and _sha256_file(config_path)))
        ),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(csv_path)


if __name__ == "__main__":
    main()
