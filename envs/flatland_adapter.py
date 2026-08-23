"""Flatland population wrapper and CIG-AMF causal adapter.

The wrapper normalizes Flatland's dictionary API into the population API used
by the runner.  It exposes observable rail-conflict relations and state-aware
action masks.  Clone-state interventions use deep snapshots, so H1/latency
are intentionally configured for small oracle state banks.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

import numpy as np

from envs.external_contract import BenchmarkCapabilities, flatten_observation


class FlatlandCIGAdapter:
    def __init__(self, wrapper):
        self.env = wrapper

    @property
    def n_agents(self): return self.env.n_agents
    @property
    def max_action_dim(self): return 5
    @property
    def obs_dim(self): return self.env.obs_dim
    @property
    def relation_feature_dim(self): return 5
    @property
    def pair_feature_dim(self): return 5
    @property
    def context_item_dim(self): return 5 + self.max_action_dim + 4
    def reset(self): return self.env.reset()
    def step(self, actions): return self.env.step(actions)
    def observation(self, observations, agent): return np.asarray(observations[int(agent)], dtype=np.float32)
    def valid_action_mask(self, agent): return self.env.valid_action_mask(agent)
    def relation_features(self, ego, target): return self.env.relation_features(ego, target)
    def pair_features(self, ego, target): return self.relation_features(ego, target)
    def neighbour_features(self, ego, neighbour, action):
        onehot = np.zeros(5, dtype=np.float32); onehot[int(action)] = 1.0
        agent = self.env.rail_env.agents[int(neighbour)]
        active = float(getattr(agent, "position", None) is not None)
        malfunction = float(getattr(getattr(agent, "malfunction_handler", None), "malfunction_down_counter", 0) > 0)
        speed = float(getattr(getattr(agent, "speed_counter", None), "is_cell_exit", lambda: False)())
        return np.concatenate([self.relation_features(ego, neighbour), onehot,
                               np.asarray([active, malfunction, speed, 0.0], dtype=np.float32)])
    def context_key(self, ego, neighbour):
        rel = self.relation_features(ego, neighbour)
        return tuple(np.round(rel, 2).tolist())
    def weak_prior_score(self, ego, neighbour):
        return float(1.0 / (1.0 + max(0.0, self.relation_features(ego, neighbour)[0])))
    def feature_snapshot(self):
        return {"relations": np.stack([[self.relation_features(i, j) if i != j else np.zeros(5)
                                         for j in range(self.n_agents)] for i in range(self.n_agents)])}
    def pair_features_from_snapshot(self, snapshot, ego, target):
        return np.asarray(snapshot["relations"][int(ego), int(target)], dtype=np.float32)
    def clone_state(self): return self.env.clone_state()
    def restore_state(self, state): return self.env.restore_state(state)
    def fixed_continuation_policy(self, agent): return self.env.fixed_continuation_policy(agent)


class FlatlandCIGEnvironment:
    # Only training support is advertised until rail-specific intervention,
    # fixed-rho oracle, and structural-disruption methods are implemented.
    capabilities = BenchmarkCapabilities(
        "Flatland", True, True, True, True, False, False, False
    )

    def __init__(self, rail_env, observation_width: int = 256):
        self.rail_env = rail_env
        self.n_agents = int(getattr(rail_env, "number_of_agents", len(rail_env.agents)))
        self.obs_dim = int(observation_width)
        self.causal_adapter = FlatlandCIGAdapter(self)
        self._obs = [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.n_agents)]
        self.last_actions = [0 for _ in range(self.n_agents)]
        self._behaviour_override = "cooperative"

    def get_action_dim(self): return 5
    def get_obs_dim(self): return self.obs_dim
    def get_obs_of_ego(self, observations, ego): return self._obs[int(ego)] if observations is None else observations[int(ego)]
    def _get_obs_all(self): return self._obs
    def set_behaviour_override(self, mode): self._behaviour_override = str(mode)
    def scripted_policy_distribution(self, agent):
        mask = self.valid_action_mask(agent)
        out = mask.astype(np.float32)
        if self._behaviour_override == "selfish" and mask[0]:
            out.fill(0.0); out[0] = 1.0
        return out / np.clip(out.sum(), 1.0, None)
    def scripted_policy(self, agent):
        return int(np.argmax(self.scripted_policy_distribution(agent)))
    def _normalise_obs(self, observations):
        return [flatten_observation(observations.get(i), self.obs_dim) for i in range(self.n_agents)]
    def reset(self, seed=None):
        if seed is None:
            result = self.rail_env.reset()
        else:
            parameters = inspect.signature(self.rail_env.reset).parameters
            if "random_seed" in parameters:
                result = self.rail_env.reset(random_seed=seed)
            elif "seed" in parameters:
                result = self.rail_env.reset(seed=seed)
            else:
                raise RuntimeError("unsupported Flatland reset API: no random_seed/seed parameter")
        observations = result[0] if isinstance(result, tuple) else result
        self._obs = self._normalise_obs(observations)
        return self._obs
    def step(self, actions):
        result = self.rail_env.step({i: int(actions[i]) for i in range(self.n_agents)})
        if len(result) == 4:
            observations, rewards, dones, info = result
            done = bool(dones.get("__all__", all(dones.values())))
        else:
            observations, rewards, terminated, truncated, info = result
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        self.last_actions = [int(a) for a in actions]
        self._obs = self._normalise_obs(observations)
        return self._obs, [float(rewards.get(i, 0.0)) for i in range(self.n_agents)], done, dict(info)
    def valid_action_mask(self, agent):
        mask = np.zeros(5, dtype=bool)
        obj = self.rail_env.agents[int(agent)]
        required = type(self.rail_env).action_required(
            obj.state, obj.speed_counter.is_cell_exit(obj.speed_counter.max_speed)
        )
        if not required:
            mask[0] = True
            return mask
        position = getattr(obj, "position", None)
        direction = getattr(obj, "direction", 0)
        if position is None:
            mask[0] = True
            return mask
        for action in range(5):
            try:
                allowed = self.rail_env.rail.apply_action_independent(action, (position, direction)) is not None
            except Exception:
                allowed = True
            mask[action] = bool(allowed)
        return mask
    def relation_features(self, ego, neighbour):
        a, b = self.rail_env.agents[int(ego)], self.rail_env.agents[int(neighbour)]
        pa, pb = getattr(a, "position", None), getattr(b, "position", None)
        if pa is None or pb is None: return np.zeros(5, dtype=np.float32)
        distance = abs(float(pa[0] - pb[0])) + abs(float(pa[1] - pb[1]))
        same_cell = float(tuple(pa) == tuple(pb))
        same_target = float(getattr(a, "target", None) == getattr(b, "target", None))
        return np.asarray([distance, same_cell, same_target, float(pa[0] - pb[0]), float(pa[1] - pb[1])], dtype=np.float32)
    def clone_state(self):
        clone_from = getattr(type(self.rail_env), "clone_from", None)
        clone = clone_from(self.rail_env) if callable(clone_from) else copy.deepcopy(self.rail_env)
        return clone, copy.deepcopy(self._obs), list(self.last_actions), self._behaviour_override
    def restore_state(self, state):
        self.rail_env, self._obs, self.last_actions, self._behaviour_override = state
    def fixed_continuation_policy(self, agent):
        mask = self.valid_action_mask(agent)
        return int(2 if mask[2] else np.flatnonzero(mask)[0])
