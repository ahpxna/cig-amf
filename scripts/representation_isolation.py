"""Shared teacher-forced histories for Paper-B representation fidelity.

Fidelity is an architectural quantity only when every representation observes
the same state/action history and the downstream policy/value weights are
held fixed.  End-to-end reward experiments intentionally remain separate.
"""

from __future__ import annotations

import copy
import hashlib
import pickle

import numpy as np

from models.influence_signature import CausalPairSignal


def collect_teacher_trajectory(runner, episodes: int):
    """Collect immutable common state/action trajectories from one teacher."""
    traces = []
    for _ in range(int(episodes)):
        trajectory, _reward, _runtime = runner.collect_episode()
        traces.append(copy.deepcopy(trajectory))
    return traces




def teacher_history_hashes(traces):
    """Stable provenance hashes for the immutable teacher history.

    The full trace hash protects state/cache provenance; the action hash makes
    it cheap for validators to ensure every representation arm saw identical
    factual actions even if auxiliary cache formats evolve.
    """
    if not traces or not any(traces):
        raise ValueError("teacher history is empty")
    full = hashlib.sha256(
        pickle.dumps(traces, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()
    actions = [
        tuple(int(a) for a in step.get("actions", ()))
        for trajectory in traces for step in trajectory
    ]
    action_hash = hashlib.sha256(
        pickle.dumps(actions, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()
    peripheral_items = sum(
        sum(len(items) for items in step.get("periph_inputs_cache", {}).values())
        for trajectory in traces for step in trajectory
    )
    return {
        "teacher_trace_sha256": full,
        "teacher_action_history_sha256": action_hash,
        "teacher_history_steps": int(len(actions)),
        "teacher_peripheral_item_count": int(peripheral_items),
    }


def replay_pair_history(runner, traces, *, train_bc: bool, bc_steps: int):
    """Teacher-force actions through pair state without policy optimisation.

    Pair modules see exactly the same pre-step environment snapshot and action
    sequence in each arm.  Only BC/CD auxiliary updates are optional; policy
    and value weights are never updated by this replay.
    """
    module = runner.pair_rel_module
    if bool(runner.cfg.get("freeze_representation_state", False)):
        raise ValueError("matched-history replay requires mutable representation state")
    def cached_cd_target(step, ego, neighbour):
        """Read the action-time typed C/D profile cached with the history.

        The isolation panel must not synthesize a current slow-belief label
        after replaying a different runner.  The cached signal is the profile
        available at the same pre-action state as the representation sample;
        its timestamp is preserved and its age is zero by construction.
        """
        signal = (
            step.get("pair_signal_cache", {})
            .get(int(ego), {})
            .get(int(neighbour))
        )
        if not isinstance(signal, CausalPairSignal):
            return {
                "target": None,
                "age": None,
                "timestamp": None,
                "valid": False,
            }
        return {
            "target": signal.allocator_profile[:2],
            "age": 0,
            "timestamp": int(signal.timestamp),
            "valid": bool(signal.support_valid),
        }

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
                next_pre_forcing = step.get("pre_forcing_actions", actions)
                next_masks = step.get("valid_action_masks")
                module.add_bc_transition(
                    observations={agent: previous["obs"][agent] for agent in range(runner.n_agents)},
                    actions={agent: previous["actions"][agent] for agent in range(runner.n_agents)},
                    next_actions={
                        agent: int(next_pre_forcing[agent])
                        for agent in range(runner.n_agents)
                    },
                    next_forced_mask=step.get("forced_mask"),
                    next_valid_action_masks=next_masks,
                    next_actions_are_pre_forcing=(
                        "pre_forcing_actions" in step
                    ),
                    env=runner.env,
                    h_prev_snapshot=previous["h_before"],
                    cd_target_fn=lambda ego, neighbour, source=previous: (
                        cached_cd_target(source, ego, neighbour)
                    ),
                )
            runner.env_adapter.restore_state(before)
            module.step_population(obs_all=obs, actions=actions, env=runner.env)
            runner.env_adapter.restore_state(after)
            previous = {
                "before": before,
                "obs": copy.deepcopy(obs),
                "actions": actions,
                "h_before": h_before,
                "pair_signal_cache": copy.deepcopy(
                    step.get("pair_signal_cache", {})
                ),
            }
    if train_bc and not bool(runner.cfg.get("freeze_pair_bc_learning", False)):
        # Paper B freezes the C/D normalizer on the common pre-confirmatory
        # history before applying L_CD/L_con.  Fitting after the complete
        # shared replay avoids a variant-specific first-trajectory scale.
        if not bool(module.cd_normalization_frozen):
            module.fit_cd_normalization(
                min_samples=int(runner.cfg.get("cd_normalization_min_samples", 32))
            )
        module.train_bc(
            n_steps=int(bc_steps),
            batch_size=int(runner.cfg.get("bc_batch_size", 256)),
            heads=runner.heads,
            heads_optim=runner.heads_optim,
            w_influence=float(runner.cfg.get("heads_w_influence", 1.0)),
            w_contrastive=float(runner.cfg.get("heads_w_contrastive", 0.3)),
        )


def probe_trace_terminals(runner, traces, probe_fn):
    """Replay common factual history and probe only representation-aligned states.

    A cloned environment snapshot is not sufficient for a recurrent model: pair
    states and other representation memory must correspond to the same point in
    history.  This helper advances the pair state with the teacher actions and
    evaluates only the final post-step state of each replayed trace.
    """
    probes = []
    for trace in traces:
        if not trace:
            continue
        replay_pair_history(runner, [trace], train_bc=False, bc_steps=0)
        state = copy.deepcopy(trace[-1]["env_snapshot_after_step"])
        probes.append(probe_fn(runner, [state]))
    if not probes:
        raise ValueError("cannot probe an empty teacher history")
    keys = ("logits", "values", "actions")
    return {key: np.concatenate([probe[key] for probe in probes], axis=0) for key in keys}


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
    if not bool(runner.cfg.get("freeze_belief_summary_learning", False)):
        raise ValueError(
            "representation isolation requires a frozen common belief summary"
        )
    if bool(runner.cfg.get("freeze_policy_learning", False)):
        raise ValueError("representation isolation must allow representation gradients")
    metrics = []
    for trajectory in traces:
        metrics.append(runner.update_policy(trajectory))
    return metrics
