"""CybORG PettingZoo-parallel adapter for CIG-AMF."""
from __future__ import annotations
import copy
import numpy as np

from envs.external_contract import BenchmarkCapabilities
from envs.external.common import ExternalPopulationMixin, pad_observations


class CybORGCIGEnvironment(ExternalPopulationMixin):
    capabilities = BenchmarkCapabilities("CybORG", True, True, True, True, False, False, False)

    def __init__(self, wrapped_env, observation_width=256):
        self.raw_env = wrapped_env
        self.agent_names = list(getattr(wrapped_env, "possible_agents", []))
        if not self.agent_names:
            self.agent_names = list(getattr(wrapped_env, "agents", []))
        if not self.agent_names:
            raise RuntimeError("CybORG wrapper exposes no controllable agents")
        self.n_agents = len(self.agent_names)
        dims = [int(wrapped_env.action_space(name).n) for name in self.agent_names]
        self.max_action_dim = max(dims)
        self.obs_dim = int(observation_width)
        self._obs = [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.n_agents)]
        self._finish_init()

    def _ordered_obs(self, mapping):
        return pad_observations([mapping.get(name, np.zeros(1)) for name in self.agent_names], self.obs_dim)

    def reset(self, seed=None):
        # The pinned PettingZooParallelWrapper accepts ``seed`` but does not
        # forward it to CybORG.reset(). Seed the wrapped CybORG object
        # explicitly before invoking the wrapper reset so repeated experiment
        # seeds are actually reproducible.
        if seed is not None:
            base = getattr(self.raw_env, "env", None)
            setter = getattr(base, "set_seed", None)
            if callable(setter):
                setter(int(seed))
        try:
            observations = self.raw_env.reset(seed=seed)
        except TypeError:
            observations = self.raw_env.reset()
        if isinstance(observations, tuple):
            observations = observations[0]
        self._obs = self._ordered_obs(observations)
        self.last_actions = [0] * self.n_agents
        return self._obs

    def step(self, actions):
        action_dict = {}
        for idx, name in enumerate(self.agent_names):
            mask = self.valid_action_mask(idx)
            action = int(actions[idx])
            if not (0 <= action < mask.size and mask[action]):
                action = int(np.flatnonzero(mask)[0])
            action_dict[name] = action
        result = self.raw_env.step(action_dict)
        if len(result) == 4:
            observations, rewards, dones, info = result
            done = bool(all(bool(dones.get(name, False)) for name in self.agent_names))
        else:
            observations, rewards, terminated, truncated, info = result
            done = bool(all(bool(terminated.get(name, False) or truncated.get(name, False)) for name in self.agent_names))
        self._obs = self._ordered_obs(observations)
        self.last_actions = [int(action_dict[name]) for name in self.agent_names]
        reward_list = [float(rewards.get(name, 0.0)) for name in self.agent_names]
        return self._obs, reward_list, done, dict(info)

    def valid_action_mask(self, agent):
        name = self.agent_names[int(agent)]
        n = int(self.raw_env.action_space(name).n)
        mask = np.zeros((self.max_action_dim,), dtype=bool)
        mask[:n] = True
        return mask

    def relation_features(self, ego, neighbour):
        oi = np.asarray(self._obs[int(ego)], dtype=np.float64)
        oj = np.asarray(self._obs[int(neighbour)], dtype=np.float64)
        ni, nj = np.linalg.norm(oi), np.linalg.norm(oj)
        cosine = float(np.dot(oi, oj) / max(1e-12, ni * nj))
        l1 = float(np.mean(np.abs(oi - oj)))
        l2 = float(np.sqrt(np.mean((oi - oj) ** 2)))
        overlap = float(np.mean((oi != 0) & (oj != 0)))
        same_action_dim = float(self.valid_action_mask(ego).sum() == self.valid_action_mask(neighbour).sum())
        active = 1.0
        return np.asarray([cosine, l1, l2, overlap, same_action_dim, active], dtype=np.float32)

    def clone_state(self):
        return (copy.deepcopy(self.raw_env), copy.deepcopy(self._obs), list(self.last_actions), self._behaviour_override)

    def restore_state(self, state):
        self.raw_env, self._obs, self.last_actions, self._behaviour_override = copy.deepcopy(state)

    def fixed_continuation_policy(self, agent):
        # PettingZooParallelWrapper maps action 0 to Sleep in this pinned CybORG family.
        mask = self.valid_action_mask(agent)
        return int(0 if mask[0] else np.flatnonzero(mask)[0])


def make_cyborg_environment(seed=0, observation_width=256, **_):
    # The pinned CybORG wrapper still references the removed NumPy alias
    # ``np.int``. Keep the compatibility shim local to this optional adapter.
    if not hasattr(np, "int"):
        np.int = int
    from CybORG import CybORG
    from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
    from CybORG.Agents.Wrappers.PettingZooParallelWrapper import PettingZooParallelWrapper
    scenario = DroneSwarmScenarioGenerator()
    raw = PettingZooParallelWrapper(CybORG(scenario_generator=scenario, environment="sim"))
    env = CybORGCIGEnvironment(raw, observation_width=observation_width)
    env.reset(seed=int(seed))
    return env
