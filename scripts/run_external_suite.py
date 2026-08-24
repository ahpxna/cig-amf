"""Executable capability audit for pinned external CIG-AMF benchmarks.

The suite validates both declarative capabilities and the methods/runtime path
required by the selected panel.  It does not silently promote a smoke test into
paper evidence: training smoke is architecture validation only, and H2/latency
remain blocked unless the adapter declares those scientific contracts.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.external.runtime import ensure_output_file_parent, maybe_reexec_in_external_runtime

import numpy as np

from envs.external.registry import SPECS, build_environment, repo_path
from envs.external_contract import require_panel


def _static_panel_support(environment, panel):
    spec = SPECS[environment]
    cap = spec.capabilities
    if not cap.supports(panel):
        return False, "capability flag is disabled"
    cls_methods = {
        "training": ("clone_state", "restore_state", "reset", "step", "valid_action_mask"),
        "h1": ("clone_state", "restore_state", "fixed_continuation_policy", "oracle_lag_response"),
        "h2": ("clone_state", "restore_state", "apply_structural_intervention", "apply_behavioural_intervention"),
        "latency": ("clone_state", "restore_state", "fixed_continuation_policy", "oracle_lag_response"),
    }
    # The runtime object is authoritative; static method checks are performed
    # after construction because some wrappers delegate methods through a base.
    return True, ""


def _runtime_smoke(env):
    observations = env.reset(seed=123)
    if len(observations) != int(env.n_agents):
        raise RuntimeError("reset did not return one observation per agent")
    masks = [np.asarray(env.valid_action_mask(i), dtype=bool) for i in range(env.n_agents)]
    if any(mask.shape != (env.get_action_dim(),) or not np.any(mask) for mask in masks):
        raise RuntimeError("invalid state-dependent action mask")
    before = env.clone_state()
    actions = [int(np.flatnonzero(mask)[0]) for mask in masks]
    next_obs, rewards, done, info = env.step(actions)
    if len(next_obs) != env.n_agents or len(rewards) != env.n_agents:
        raise RuntimeError("step did not return a population transition")
    env.restore_state(copy.deepcopy(before))
    restored = env._get_obs_all()
    if len(restored) != env.n_agents:
        raise RuntimeError("clone/restore did not restore a population state")
    return {
        "n_agents": int(env.n_agents),
        "action_dim": int(env.get_action_dim()),
        "obs_dim": int(env.get_obs_dim()),
        "one_step_done": bool(done),
        "reward_mean": float(np.mean(np.asarray(rewards, dtype=np.float64))),
    }


def _h1_smoke(env):
    if env.n_agents < 2:
        raise RuntimeError("H1 smoke requires at least two agents")
    env.reset(seed=321)
    source = 1
    mask = np.asarray(env.valid_action_mask(source), dtype=bool)
    valid = np.flatnonzero(mask)
    if valid.size < 2:
        raise RuntimeError("H1 smoke requires a source with at least two valid actions")
    response = env.oracle_lag_response(
        ego_id=0, agent_j=source, intervention_action=int(valid[-1]),
        horizon=2, n_trials=1, forced_step=0,
        continuation_policy=env.fixed_continuation_policy, discount=0.95,
    )
    required = {"per_lag_response", "discounted_response", "response_mass"}
    if not required.issubset(response):
        raise RuntimeError("oracle_lag_response returned an incomplete record")
    return {
        "discounted_response": float(response["discounted_response"]),
        "response_mass": float(response["response_mass"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=sorted(SPECS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panels", nargs="+", default=["training", "h1", "h2", "latency"])
    parser.add_argument("--agent-count", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--config-path")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.manifest_only:
        maybe_reexec_in_external_runtime(require_ready=True)

    decisions = {panel: _static_panel_support(args.environment, panel) for panel in args.panels}
    runtime = {}
    construction_error = None
    env = None
    if not args.manifest_only and any(ok for ok, _ in decisions.values()):
        try:
            env = build_environment(
                args.environment, seed=0, n_agents=args.agent_count,
                max_steps=args.max_steps, config_path=args.config_path,
            )
            if decisions.get("training", (False,))[0]:
                require_panel(env, "training")
                runtime["training"] = _runtime_smoke(env)
            if decisions.get("h1", (False,))[0]:
                require_panel(env, "h1")
                runtime["h1"] = _h1_smoke(env)
        except Exception as exc:
            construction_error = f"{type(exc).__name__}: {exc}"
            # Construction/runtime failure invalidates all panels that would
            # depend on this adapter instance; fail closed.
            for panel, (ok, _) in list(decisions.items()):
                if ok:
                    decisions[panel] = (False, "runtime adapter check failed: " + construction_error)

    cap = SPECS[args.environment].capabilities
    declared_panels = [panel for panel, (ok, _) in decisions.items() if ok]
    repository_exists = repo_path(args.environment).exists()
    if args.manifest_only:
        # Manifest mode reports declared support only.  It must not call a
        # panel runtime-runnable when the pinned repository has not been
        # constructed and smoke-tested in this process.
        runnable_panels = []
        operational_status = "not_executed"
    else:
        # Only panels actually exercised in this process are called runnable.
        # A future adapter may *declare* H2/latency support, but those scientific
        # panels still require their dedicated runner before becoming evidence.
        runnable_panels = [panel for panel in declared_panels if panel in runtime]
        operational_status = "failed" if construction_error else "verified"
    payload = {
        "environment": args.environment,
        "repository": str(repo_path(args.environment)),
        "repository_exists": repository_exists,
        "capabilities": vars(cap),
        "requested_panels": args.panels,
        "declared_supported_panels": declared_panels,
        "runnable_panels": runnable_panels,
        "blocked_panels": [panel for panel, (ok, _) in decisions.items() if not ok],
        "blocked_reasons": {panel: reason for panel, (ok, reason) in decisions.items() if not ok},
        "runtime_smoke": runtime,
        "construction_error": construction_error,
        "operational_status": operational_status,
        "manifest_only": bool(args.manifest_only),
        "rule": "blocked panels must not generate paper evidence",
        "training_scope": "architecture generalization; not causal identification",
    }
    migrated = ensure_output_file_parent(args.out)
    if migrated is not None:
        print(f"[EXTERNAL-OUT] preserved legacy output file as {migrated}", file=sys.stderr)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    # A blocked scientific panel is not an operational failure. Runtime adapter
    # failures are operational and return nonzero so experiment automation can
    # stop before treating the environment as ready.
    raise SystemExit(2 if construction_error else 0)


if __name__ == "__main__":
    main()
