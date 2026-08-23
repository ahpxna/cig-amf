"""Shared teacher-forced histories for Paper-B representation fidelity.

Fidelity is an architectural quantity only when every representation observes
the same state/action history and the downstream policy/value weights are
held fixed.  End-to-end reward experiments intentionally remain separate.
"""

from __future__ import annotations

import copy

import numpy as np


def collect_teacher_trajectory(runner, episodes: int):
    """Collect immutable common state/action trajectories from one teacher."""
    traces = []
    for _ in range(int(episodes)):
        trajectory, _reward, _runtime = runner.collect_episode()
        traces.append(copy.deepcopy(trajectory))
    return traces


def replay_pair_history(runner, traces, *, train_bc: bool, bc_steps: int):
    """Teacher-force actions through pair state without policy optimisation.

    Pair modules see exactly the same pre-step environment snapshot and action
    sequence in each arm.  Only BC/CD auxiliary updates are optional; policy
    and value weights are never updated by this replay.
    """
    module = runner.pair_rel_module
    if bool(runner.cfg.get("freeze_representation_state", False)):
        raise ValueError("matched-history replay requires mutable representation state")
    for trajectory in traces:
        previous = None
        for step in trajectory:
            before = copy.deepcopy(step["env_snapshot_before_step"])
            after = copy.deepcopy(step["env_snapshot_after_step"])
            obs = step["obs_all"]
            actions = [int(value) for value in step["actions"]]
            h_before = module.clone_full_states_np()
            if previous is not None:
                runner.env_adapter.restore_state(previous["before"])
                module.add_bc_transition(
                    observations={agent: previous["obs"][agent] for agent in range(runner.n_agents)},
                    actions={agent: previous["actions"][agent] for agent in range(runner.n_agents)},
                    next_actions={agent: actions[agent] for agent in range(runner.n_agents)},
                    env=runner.env,
                    h_prev_snapshot=previous["h_before"],
                    cd_target_fn=lambda ego, neighbour: np.asarray([
                        runner.belief_modules[ego].debiased_mu(neighbour),
                        runner.sig_tracker.get_signature(ego, neighbour)[1],
                    ], dtype=np.float32),
                )
            runner.env_adapter.restore_state(before)
            module.step_population(obs_all=obs, actions=actions, env=runner.env)
            runner.env_adapter.restore_state(after)
            previous = {
                "before": before,
                "obs": copy.deepcopy(obs),
                "actions": actions,
                "h_before": h_before,
            }
        if train_bc and not bool(runner.cfg.get("freeze_pair_bc_learning", False)):
            module.train_bc(n_steps=int(bc_steps))


def terminal_states(traces, n_states: int):
    """Select a deterministic terminal-state bank from a common history."""
    states = [copy.deepcopy(step["env_snapshot_after_step"])
              for trajectory in traces for step in trajectory]
    if not states:
        raise ValueError("teacher history is empty")
    count = max(1, min(int(n_states), len(states)))
    if count == len(states):
        return states
    indices = np.linspace(0, len(states) - 1, num=count, dtype=int)
    return [states[int(index)] for index in indices]


def train_periphery_on_teacher_history(runner, traces):
    """Optimise representation-only modules against a fixed downstream net."""
    if not bool(runner.cfg.get("freeze_downstream_policy_value", False)):
        raise ValueError("representation isolation requires frozen downstream policy/value")
    if bool(runner.cfg.get("freeze_policy_learning", False)):
        raise ValueError("representation isolation must allow representation gradients")
    metrics = []
    for trajectory in traces:
        metrics.append(runner.update_policy(trajectory))
    return metrics
