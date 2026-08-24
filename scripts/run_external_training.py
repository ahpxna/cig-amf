"""Run actual Final-CIGAMF training on one normalized external benchmark."""
from __future__ import annotations
import argparse, csv, json, math, os, time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from envs.external.runtime import ensure_directory, maybe_reexec_in_external_runtime

ENVIRONMENTS = ("cityflow", "cyborg", "flatland", "rware")


def _finite_mean(values):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--agent-count", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--config-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--profile", choices=("full", "quick"), default="full",
        help="full uses the paper architecture; quick uses smoke_cfg only for wiring checks",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    maybe_reexec_in_external_runtime(require_ready=True)

    # Heavy project/external dependencies are imported only after the managed
    # external-runtime handoff. This prevents a lean/main interpreter from
    # failing on torch/numpy before it has a chance to re-exec.
    import numpy as np
    import run_experiment as RE
    from envs.external.registry import build_environment
    from envs.external_contract import require_panel

    if args.episodes <= 0 or args.max_steps <= 0 or not args.seeds:
        parser.error("episodes, max-steps and seeds must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    migrated = ensure_directory(args.out)
    if migrated is not None:
        print(f"[EXTERNAL-OUT] preserved legacy output file as {migrated}", file=sys.stderr)
    rows = []
    for seed in args.seeds:
        RE.set_global_seed(int(seed))
        env = build_environment(
            args.environment, seed=seed, n_agents=args.agent_count,
            max_steps=args.max_steps, config_path=args.config_path,
        )
        require_panel(env, "training")
        cfg = RE.default_cfg() if args.profile == "full" else RE.smoke_cfg()
        cfg.update({
            "seed": int(seed), "k0_warmup": 0,
            "causal_horizon": min(4, max(1, args.max_steps // 4)),
            "proxy_n_horizons": min(4, max(1, args.max_steps // 4)),
            "max_core_size": min(3, max(1, env.n_agents - 1)),
            "min_core_size": 1,
            "seed_core_top_k": min(3, max(1, env.n_agents - 1)),
            "belief_adaptive_k_min": 1,
            "periph_require_full_signature": True,
            "periph_allow_legacy_items": False,
        })
        runner = RE.make_runner("Final-CIGAMF", env, cfg, args.device)
        started = time.perf_counter()
        history = runner.run(n_episodes=args.episodes, eval_every=1)
        elapsed = time.perf_counter() - started
        rows.append({
            "environment": args.environment,
            "profile": args.profile,
            "seed": int(seed),
            "episodes": int(args.episodes),
            "n_agents": int(env.n_agents),
            "action_dim": int(env.get_action_dim()),
            "obs_dim": int(env.get_obs_dim()),
            "mean_reward": _finite_mean(history.get("mean_reward", [])),
            "final_reward": float(history.get("mean_reward", [float("nan")])[-1]),
            "mean_core_size": _finite_mean(history.get("mean_core_size", [])),
            "throughput_total": _finite_mean(history.get("throughput_total_agent_steps_per_sec", [])),
            "elapsed_seconds": elapsed,
        })
    csv_path = args.out / "summary_external_training.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        "experiment": "external_final_cigamf_training_v1",
        "environment": args.environment,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "profile": args.profile,
        "summary": str(csv_path),
        "paper_evidence_scope": "architecture_generalization_only",
        "not_an_h1_h2_or_latency_claim": True,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(csv_path)


if __name__ == "__main__":
    main()
