"""Shared utilities for executable external CIG-AMF adapters."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

import numpy as np

from envs.external_contract import BenchmarkCapabilities, flatten_observation
from utils.latency_protocol import LATENCY_ONSET_ABS_FLOOR, LATENCY_ONSET_FRACTION


class ExternalCIGAdapter:
    """Adapter consumed by FinalCIGAMFRunner for external population envs."""

    def __init__(self, env):
        self.env = env

    @property
    def n_agents(self): return int(self.env.n_agents)
    @property
    def max_action_dim(self): return int(self.env.get_action_dim())
    @property
    def obs_dim(self): return int(self.env.get_obs_dim())
    @property
    def relation_feature_dim(self): return 6
    @property
    def pair_feature_dim(self): return 6
    @property
    def context_item_dim(self): return 6 + self.max_action_dim + 4
    def reset(self): return self.env.reset()
    def step(self, actions): return self.env.step(actions)
    def observation(self, observations, agent):
        return np.asarray(observations[int(agent)], dtype=np.float32)
    def valid_action_mask(self, agent): return self.env.valid_action_mask(agent)
    def relation_features(self, ego, target): return self.env.relation_features(ego, target)
    def pair_features(self, ego, target): return self.relation_features(ego, target)
    def compact_relation_features(self, ego, target, width):
        raw = np.asarray(self.relation_features(ego, target), dtype=np.float32).reshape(-1)
        out = np.zeros((int(width),), dtype=np.float32)
        out[: min(out.size, raw.size)] = raw[: min(out.size, raw.size)]
        return out
    def neighbour_features(self, ego, neighbour, action):
        onehot = np.zeros((self.max_action_dim,), dtype=np.float32)
        action = int(action)
        if action < 0 or action >= self.max_action_dim:
            raise ValueError("neighbour action is outside the adapter action space")
        onehot[action] = 1.0
        extra = np.asarray(self.env.mechanism_features(neighbour), dtype=np.float32).reshape(-1)
        padded = np.zeros((4,), dtype=np.float32)
        padded[: min(4, extra.size)] = extra[:4]
        return np.concatenate([self.relation_features(ego, neighbour), onehot, padded])
    def context_key(self, ego, neighbour):
        return tuple(np.round(self.relation_features(ego, neighbour), 2).tolist())
    def weak_prior_score(self, ego, neighbour): return float(self.env.weak_prior_score(ego, neighbour))
    def feature_snapshot(self): return self.env.feature_snapshot()
    def pair_features_from_snapshot(self, snapshot, ego, target):
        return self.env.pair_features_from_snapshot(snapshot, ego, target)
    def clone_state(self): return self.env.clone_state()
    def restore_state(self, state): return self.env.restore_state(state)
    def fixed_continuation_policy(self, agent): return self.env.fixed_continuation_policy(agent)


class ExternalPopulationMixin:
    """Small normalized population API plus a generic action-intervention oracle.

    Environment wrappers supply reset/step/masks/relation features and clone/restore.
    The generic oracle compares one forced action with the wrapper's fixed
    continuation action under the exact same cloned state. It is suitable for
    H1 action-response measurements, but does not imply a structural H2 oracle
    or a ground-truth latency mechanism.
    """

    capabilities: BenchmarkCapabilities

    def _finish_init(self):
        self.causal_adapter = ExternalCIGAdapter(self)
        self.last_actions = [0] * int(self.n_agents)
        self._behaviour_override = "cooperative"

    def get_action_dim(self): return int(self.max_action_dim)
    def get_obs_dim(self): return int(self.obs_dim)
    def get_obs_of_ego(self, observations, ego):
        return np.asarray(observations[int(ego)], dtype=np.float32)
    def _get_obs_all(self): return self._obs
    def set_behaviour_override(self, mode): self._behaviour_override = str(mode)
    def scripted_policy_distribution(self, agent):
        mask = np.asarray(self.valid_action_mask(int(agent)), dtype=bool)
        if mask.shape != (self.max_action_dim,) or not np.any(mask):
            raise RuntimeError("external adapter returned an invalid action mask")
        probs = mask.astype(np.float64)
        if self._behaviour_override == "selfish":
            preferred = int(self.selfish_action(int(agent)))
            if 0 <= preferred < mask.size and mask[preferred]:
                probs[:] = 0.0
                probs[preferred] = 1.0
        return probs / probs.sum()
    def scripted_policy(self, agent):
        probs = self.scripted_policy_distribution(agent)
        return int(np.argmax(probs))
    def selfish_action(self, agent): return self.fixed_continuation_policy(agent)
    def fixed_continuation_policy(self, agent):
        mask = np.asarray(self.valid_action_mask(int(agent)), dtype=bool)
        return int(np.flatnonzero(mask)[0])
    def mechanism_features(self, neighbour):
        mask = np.asarray(self.valid_action_mask(int(neighbour)), dtype=bool)
        return np.asarray([
            float(np.count_nonzero(mask)) / max(1.0, float(mask.size)),
            float(self.last_actions[int(neighbour)]) / max(1.0, float(self.max_action_dim - 1)),
            1.0,
            0.0,
        ], dtype=np.float32)
    def weak_prior_score(self, ego, neighbour):
        rel = np.asarray(self.relation_features(ego, neighbour), dtype=np.float64)
        return float(1.0 / (1.0 + np.linalg.norm(rel)))
    def candidate_neighbors(self, ego, max_degree):
        ids = [j for j in range(int(self.n_agents)) if int(j) != int(ego)]
        ids.sort(key=lambda j: (-self.weak_prior_score(ego, j), int(j)))
        return ids[: int(max_degree)]
    def feature_snapshot(self):
        return {"relations": np.stack([
            np.stack([
                np.zeros(6, dtype=np.float32) if i == j else self.relation_features(i, j)
                for j in range(self.n_agents)
            ], axis=0) for i in range(self.n_agents)
        ], axis=0)}
    def pair_features_from_snapshot(self, snapshot, ego, target):
        return np.asarray(snapshot["relations"][int(ego), int(target)], dtype=np.float32)

    def _rollout_reward_sequence(
        self, forced, horizon, continuation_policy, *, return_mask=False
    ):
        rewards_by_lag = []
        valid = []
        done = False
        for lag in range(int(horizon)):
            actions = [int(continuation_policy(i)) for i in range(self.n_agents)]
            if forced is not None and lag == int(forced[2]):
                actions[int(forced[0])] = int(forced[1])
            _, rewards, done, _ = self.step(actions)
            rewards_by_lag.append(np.asarray(rewards, dtype=np.float64))
            valid.append(True)
            if done:
                # A wrapper cannot infer whether this is an absorbing terminal
                # or a time-limit truncation.  Never turn an unobserved tail
                # into observed zero rewards; callers receive a censoring mask.
                for _ in range(lag + 1, int(horizon)):
                    rewards_by_lag.append(np.zeros((self.n_agents,), dtype=np.float64))
                    valid.append(False)
                break
        values = np.stack(rewards_by_lag, axis=0)
        if return_mask:
            return values, np.asarray(valid, dtype=bool)
        return values

    def _copy_cloned_state(self, state):
        return copy.deepcopy(state)

    def compute_oracle_lag_response_from_current_state(
        self, ego_id, agent_j, intervention_action, horizon=8, n_trials=1,
        forced_step=0, continuation_policy=None, crn_seed=None, discount=0.95,
    ):
        del crn_seed  # determinism comes from clone/restore for these wrappers.
        if int(n_trials) != 1:
            raise ValueError(
                "generic external clone-state oracle is deterministic; "
                "n_trials must be 1 to avoid pseudo-replication"
            )
        horizon = int(horizon)
        forced_step = int(forced_step)
        if not 0 <= forced_step < horizon:
            raise ValueError("forced_step must lie in [0, horizon)")
        if continuation_policy is None:
            continuation_policy = self.fixed_continuation_policy
        snapshot = self.clone_state()
        response = np.zeros((horizon,), dtype=np.float64)
        try:
            for _ in range(1):
                self.restore_state(self._copy_cloned_state(snapshot))
                base, base_mask = self._rollout_reward_sequence(
                    None, horizon, continuation_policy, return_mask=True
                )
                self.restore_state(self._copy_cloned_state(snapshot))
                alt, alt_mask = self._rollout_reward_sequence(
                    (int(agent_j), int(intervention_action), forced_step),
                    horizon, continuation_policy, return_mask=True
                )
                base = base[:, int(ego_id)]
                alt = alt[:, int(ego_id)]
                valid_mask = base_mask & alt_mask
                response += np.where(valid_mask, alt - base, 0.0)
        finally:
            self.restore_state(snapshot)
        mass = np.abs(response)
        total_mass = float(mass.sum())
        if total_mass <= 1e-12:
            onset = peak = centre = None
        else:
            onset_threshold = max(
                LATENCY_ONSET_ABS_FLOOR,
                LATENCY_ONSET_FRACTION * float(mass.max()),
            )
            onset = int(np.flatnonzero(mass >= onset_threshold)[0])
            peak = int(np.argmax(mass))
            centre = float(np.dot(np.arange(horizon, dtype=np.float64), mass) / total_mass)
        weights = float(discount) ** np.arange(horizon, dtype=np.float64)
        return {
            "per_lag_response": response,
            "discounted_response": float(np.dot(weights, response)),
            "onset_lag": onset,
            "peak_lag": peak,
            "centre_of_mass_lag": centre,
            "response_mass": total_mass,
            "horizon": horizon,
            "forced_step": forced_step,
            "lag_valid_mask": valid_mask.astype(bool),
            "horizon_complete": bool(np.all(valid_mask)),
        }

    # external_contract.require_panel looks for this shorter alias.
    def oracle_lag_response(self, *args, **kwargs):
        return self.compute_oracle_lag_response_from_current_state(*args, **kwargs)


def pad_observations(values: Sequence[Any], width: int) -> list[np.ndarray]:
    return [flatten_observation(value, int(width)) for value in values]
