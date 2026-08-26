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
import torch
import torch.nn.functional as F

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


def _reference_target_hash(reference_targets):
    """Stable hash of the exact Full-Explicit logits/value targets."""
    if not reference_targets or not any(reference_targets):
        raise ValueError("reference target history is empty")
    canonical = []
    for trajectory in reference_targets:
        row = []
        for target in trajectory:
            row.append((
                np.asarray(target["logits"], dtype=np.float32),
                np.asarray(target["values"], dtype=np.float32),
                np.asarray(target["valid_action_masks"], dtype=bool),
            ))
        canonical.append(row)
    return hashlib.sha256(
        pickle.dumps(canonical, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _teacher_step_tensors(runner, step):
    """Build the common non-peripheral inputs for one immutable teacher step."""
    obs = torch.as_tensor(
        np.stack([
            runner.env_adapter.observation(step["obs_all"], ego)
            for ego in range(runner.n_agents)
        ]),
        dtype=torch.float32, device=runner.device,
    )
    core = torch.as_tensor(
        np.stack([step["core_summary_cache"][ego] for ego in range(runner.n_agents)]),
        dtype=torch.float32, device=runner.device,
    )
    belief = torch.stack([
        runner._belief_summary_tensor_from_items(step["belief_items_cache"][ego])
        for ego in range(runner.n_agents)
    ], dim=0)
    valid = np.asarray(step.get("valid_action_masks"), dtype=bool)
    if valid.shape != (runner.n_agents, runner.action_dim) or np.any(valid.sum(axis=1) == 0):
        raise ValueError("teacher history contains malformed valid-action masks")
    return obs, core, belief, valid


def collect_aligned_reference_targets(runner, traces):
    """Collect Full-Explicit targets on the exact factual pre-action history.

    The reference runner starts from the same frozen checkpoint as the
    representation variants, promotes every neighbour to the explicit tier, and
    is then teacher-forced through the immutable factual action sequence.  The
    target at each row is computed *before* the factual action is applied, so the
    recurrent pair state and environment state refer to the same timestep.
    Forcing/sampling is never invoked on this path.
    """
    if str(runner.cfg.get("core_selection_mode", "")).strip().lower() != "full_explicit":
        raise ValueError("reference target collection requires a Full-Explicit runner")
    targets = []
    for trajectory in traces:
        trajectory_targets = []
        for step in trajectory:
            before = copy.deepcopy(step["env_snapshot_before_step"])
            after = copy.deepcopy(step["env_snapshot_after_step"])
            runner.env_adapter.restore_state(before)
            obs = torch.as_tensor(
                np.stack([
                    runner.env_adapter.observation(step["obs_all"], ego)
                    for ego in range(runner.n_agents)
                ]),
                dtype=torch.float32, device=runner.device,
            )
            core = torch.as_tensor(
                np.stack([runner._core_summary_for_ego(ego) for ego in range(runner.n_agents)]),
                dtype=torch.float32, device=runner.device,
            )
            # Hold the belief projection common across the isolation arms.  The
            # Full-Explicit target differs only in the explicit relational
            # representation, not in a separately evolving belief encoder.
            belief = torch.stack([
                runner._belief_summary_tensor_from_items(step["belief_items_cache"][ego])
                for ego in range(runner.n_agents)
            ], dim=0)
            periph = torch.as_tensor(
                np.stack([
                    runner._periph_summary_np_from_inputs([])
                    for _ego in range(runner.n_agents)
                ]),
                dtype=torch.float32, device=runner.device,
            )
            valid = np.asarray(step.get("valid_action_masks"), dtype=bool)
            if valid.shape != (runner.n_agents, runner.action_dim) or np.any(valid.sum(axis=1) == 0):
                raise ValueError("reference history contains malformed valid-action masks")
            with torch.no_grad():
                logits, values = runner.policy_value(
                    obs, core, periph, belief, valid_action_mask=valid
                )
            trajectory_targets.append({
                "logits": logits.detach().cpu().numpy().astype(np.float32),
                "values": values.detach().cpu().numpy().astype(np.float32),
                "valid_action_masks": valid.copy(),
            })
            actions = [int(value) for value in step["actions"]]
            runner.pair_rel_module.step_population(
                obs_all=step["obs_all"], actions=actions, env=runner.env
            )
            runner.env_adapter.restore_state(after)
        targets.append(trajectory_targets)
    return targets


def _masked_policy_kl(student_logits, teacher_logits, valid_mask):
    """KL(teacher||student) over state-valid actions only, without 0*inf NaNs."""
    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=student_logits.device)
    if valid.shape != student_logits.shape or teacher_logits.shape != student_logits.shape:
        raise ValueError("distillation logits/mask shape mismatch")
    if not bool(valid.any(dim=1).all()):
        raise ValueError("distillation mask contains an empty valid-action row")
    student_masked = student_logits.masked_fill(~valid, -torch.inf)
    teacher_masked = teacher_logits.masked_fill(~valid, -torch.inf)
    teacher_prob = torch.softmax(teacher_masked, dim=-1)
    student_logp = torch.log_softmax(student_masked, dim=-1)
    teacher_logp = torch.where(
        valid, torch.log(torch.clamp(teacher_prob, min=1e-12)), torch.zeros_like(teacher_prob)
    )
    terms = torch.where(
        valid, teacher_prob * (teacher_logp - student_logp), torch.zeros_like(teacher_prob)
    )
    return terms.sum(dim=-1).mean()


def train_periphery_on_teacher_history(runner, traces, reference_targets):
    """Train only peripheral representation by Full-Explicit distillation.

    This intentionally does not call ``runner.update_policy``.  That routine
    consumes teacher-cached V-trace/value targets while differentiating through
    the current representation arm, creating a hybrid objective that is neither
    clean off-policy RL nor a controlled representation test.
    """
    if not bool(runner.cfg.get("freeze_downstream_policy_value", False)):
        raise ValueError("representation isolation requires frozen downstream policy/value")
    if not bool(runner.cfg.get("freeze_belief_summary_learning", False)):
        raise ValueError("representation isolation requires a frozen common belief summary")
    if len(traces) != len(reference_targets):
        raise ValueError("teacher/reference trajectory count mismatch")
    for parameter in runner.policy_value.parameters():
        if parameter.requires_grad:
            raise ValueError("downstream policy/value parameters must be frozen")
    for parameter in runner.belief_summary_builder.parameters():
        if parameter.requires_grad:
            raise ValueError("belief-summary parameters must be frozen")
    trainable = [p for p in runner.periph_module.parameters() if p.requires_grad]
    if not trainable:
        return []
    optim = torch.optim.Adam(
        trainable, lr=float(runner.cfg.get("representation_distill_lr", runner.cfg.get("policy_lr", 1e-3)))
    )
    lambda_pi = float(runner.cfg.get("representation_distill_policy_coeff", 1.0))
    lambda_v = float(runner.cfg.get("representation_distill_value_coeff", 1.0))
    lambda_aux = float(runner.cfg.get("representation_distill_aux_coeff", 1.0))
    metrics = []
    for trajectory, target_trajectory in zip(traces, reference_targets):
        if len(trajectory) != len(target_trajectory):
            raise ValueError("teacher/reference timestep count mismatch")
        total = None
        policy_total = None
        value_total = None
        aux_total = None
        count = 0
        for step, target in zip(trajectory, target_trajectory):
            obs, core, belief, valid = _teacher_step_tensors(runner, step)
            periph_rows, aux_rows = [], []
            for ego in range(runner.n_agents):
                out = runner._periph_full_from_inputs(step["periph_inputs_cache"][ego])
                periph_rows.append(out["memory"])
                aux_rows.append(out["aux_loss"])
            periph = torch.stack(periph_rows, dim=0)
            logits, values = runner.policy_value(
                obs, core, periph, belief, valid_action_mask=valid
            )
            teacher_logits = torch.as_tensor(
                target["logits"], dtype=torch.float32, device=runner.device
            )
            teacher_values = torch.as_tensor(
                target["values"], dtype=torch.float32, device=runner.device
            )
            pi_loss = _masked_policy_kl(logits, teacher_logits, valid)
            v_loss = F.mse_loss(values, teacher_values)
            aux_loss = torch.stack([
                item if torch.is_tensor(item) else torch.as_tensor(item, dtype=torch.float32, device=runner.device)
                for item in aux_rows
            ]).mean() if aux_rows else torch.zeros((), device=runner.device)
            loss = lambda_pi * pi_loss + lambda_v * v_loss + lambda_aux * aux_loss
            total = loss if total is None else total + loss
            policy_total = pi_loss if policy_total is None else policy_total + pi_loss
            value_total = v_loss if value_total is None else value_total + v_loss
            aux_total = aux_loss if aux_total is None else aux_total + aux_loss
            count += 1
        if count <= 0:
            continue
        optim.zero_grad(set_to_none=True)
        loss = total / float(count)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable, float(runner.cfg.get("policy_grad_clip", 0.5))
        )
        optim.step()
        metrics.append({
            "loss": float(loss.detach().cpu()),
            "policy_kl": float((policy_total / float(count)).detach().cpu()),
            "value_mse": float((value_total / float(count)).detach().cpu()),
            "aux_loss": float((aux_total / float(count)).detach().cpu()),
        })
    return metrics


def probe_periphery_on_teacher_history(runner, traces):
    """Numerically side-effect-free probe on immutable pre-action rows."""
    logits, values, actions = [], [], []
    module = runner.periph_module
    setter = getattr(module, "set_diagnostics_enabled", None)
    previous_diag = setter(False) if callable(setter) else None
    try:
        for trajectory in traces:
            for step in trajectory:
                obs, core, belief, valid = _teacher_step_tensors(runner, step)
                periph = torch.stack([
                    runner._periph_summary_tensor_from_inputs(
                        step["periph_inputs_cache"][ego]
                    )
                    for ego in range(runner.n_agents)
                ], dim=0)
                with torch.no_grad():
                    step_logits, step_values = runner.policy_value(
                        obs, core, periph, belief, valid_action_mask=valid
                    )
                arr = step_logits.detach().cpu().numpy().astype(np.float64)
                logits.append(arr)
                values.append(step_values.detach().cpu().numpy().astype(np.float64))
                actions.append(np.argmax(arr, axis=-1).astype(np.int64))
    finally:
        if callable(setter) and previous_diag is not None:
            setter(previous_diag)
    if not logits:
        raise ValueError("cannot probe an empty teacher history")
    return {
        "logits": np.stack(logits, axis=0),
        "values": np.stack(values, axis=0),
        "actions": np.stack(actions, axis=0),
    }


def reference_targets_as_probe(reference_targets):
    logits, values, actions = [], [], []
    for trajectory in reference_targets:
        for target in trajectory:
            arr = np.asarray(target["logits"], dtype=np.float64)
            logits.append(arr)
            values.append(np.asarray(target["values"], dtype=np.float64))
            actions.append(np.argmax(arr, axis=-1).astype(np.int64))
    if not logits:
        raise ValueError("reference target history is empty")
    return {
        "logits": np.stack(logits, axis=0),
        "values": np.stack(values, axis=0),
        "actions": np.stack(actions, axis=0),
    }
