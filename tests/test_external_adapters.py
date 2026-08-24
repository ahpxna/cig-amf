"""Dependency-free contract tests for optional external wrappers."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import numpy as np

from envs.external.rware import RWARECIGEnvironment
from envs.external.cyborg import CybORGCIGEnvironment
from envs.external.cityflow import CityFlowCIGEnvironment
from envs.external_contract import require_panel, flatten_observation


class _Dir:
    def __init__(self, value): self.value = value
    def __eq__(self, other): return isinstance(other, _Dir) and self.value == other.value


class _RWAREDouble:
    n_agents = 3
    grid_size = (8, 8)
    def __init__(self):
        self.agents = [
            SimpleNamespace(x=i, y=i + 1, dir=_Dir(i % 4), carrying_shelf=None)
            for i in range(self.n_agents)
        ]
        self.t = 0
    def reset(self, seed=None):
        self.t = 0
        return tuple(np.asarray([i, 0.0], dtype=np.float32) for i in range(self.n_agents)), {}
    def step(self, actions):
        self.t += 1
        obs = tuple(np.asarray([i, self.t], dtype=np.float32) for i in range(self.n_agents))
        return obs, [float(a) for a in actions], self.t >= 4, False, {}


class _Discrete:
    def __init__(self, n): self.n = n


class _CyborgDouble:
    possible_agents = ["blue_agent_0", "blue_agent_1", "blue_agent_2"]
    agents = possible_agents
    def __init__(self): self.t = 0
    def action_space(self, name): return _Discrete(4 if name.endswith("0") else 3)
    def reset(self, seed=None):
        self.t = 0
        return {name: np.asarray([i, 0.0]) for i, name in enumerate(self.possible_agents)}
    def step(self, actions):
        self.t += 1
        obs = {name: np.asarray([i, self.t]) for i, name in enumerate(self.possible_agents)}
        rewards = {name: float(actions[name]) for name in self.possible_agents}
        dones = {name: self.t >= 4 for name in self.possible_agents}
        return obs, rewards, dones, {}


class _Archive:
    def __init__(self, state): self.state = copy.deepcopy(state)


class _CityEngineDouble:
    def __init__(self): self.t = 0; self.phases = {}
    def get_lane_vehicle_count(self): return {"r0_0": 2, "r1_0": 1}
    def get_lane_waiting_vehicle_count(self): return {"r0_0": self.t, "r1_0": 1}
    def get_current_time(self): return float(self.t)
    def set_random_seed(self, seed): self.seed = seed
    def reset(self, seed=False): self.t = 0; self.phases = {}
    def set_tl_phase(self, iid, action): self.phases[iid] = int(action)
    def next_step(self): self.t += 1
    def snapshot(self): return _Archive((self.t, self.phases))
    def load(self, archive): self.t, self.phases = copy.deepcopy(archive.state)


class ExternalAdapterTests(unittest.TestCase):
    def _exercise(self, env):
        require_panel(env, "training")
        require_panel(env, "h1")
        obs = env.reset(seed=3)
        self.assertEqual(len(obs), env.n_agents)
        snap = env.clone_state()
        actions = [int(np.flatnonzero(env.valid_action_mask(i))[0]) for i in range(env.n_agents)]
        env.step(actions)
        env.restore_state(snap)
        response = env.oracle_lag_response(
            ego_id=0, agent_j=1, intervention_action=int(np.flatnonzero(env.valid_action_mask(1))[-1]),
            horizon=2, n_trials=1, continuation_policy=env.fixed_continuation_policy,
        )
        self.assertEqual(np.asarray(response["per_lag_response"]).shape, (2,))
        self.assertTrue(np.isfinite(response["discounted_response"]))


    def test_optional_none_observation_pads_to_finite_zeros(self):
        value = flatten_observation(None, width=5)
        np.testing.assert_array_equal(value, np.zeros(5, dtype=np.float32))
        self.assertTrue(np.isfinite(value).all())

    def test_nonfinite_external_observation_is_sanitized(self):
        value = flatten_observation([1.0, float("nan"), float("inf"), -float("inf")], width=6)
        self.assertTrue(np.isfinite(value).all())
        np.testing.assert_array_equal(
            value, np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )

    def test_final_runner_executes_through_normalized_external_adapter(self):
        # This is intentionally dependency-free: it protects the interface
        # between external adapters and the real Final-CIGAMF runner, rather
        # than merely testing reset/step in isolation.
        import run_experiment as RE

        RE.set_global_seed(0)
        env = RWARECIGEnvironment(_RWAREDouble(), observation_width=8)
        cfg = RE.smoke_cfg()
        cfg.update({
            "seed": 0,
            "k0_warmup": 0,
            "causal_horizon": 1,
            "proxy_n_horizons": 1,
            "max_core_size": 1,
            "min_core_size": 1,
            "seed_core_top_k": 1,
            "belief_adaptive_k_min": 1,
            "periph_require_full_signature": True,
            "periph_allow_legacy_items": False,
        })
        runner = RE.make_runner("Final-CIGAMF", env, cfg, "cpu")
        history = runner.run(n_episodes=1, eval_every=1)
        self.assertEqual(len(history["mean_reward"]), 1)
        self.assertTrue(np.isfinite(float(history["mean_reward"][0])))

    def test_rware_contract(self):
        self._exercise(RWARECIGEnvironment(_RWAREDouble(), observation_width=8))

    def test_cyborg_contract(self):
        self._exercise(CybORGCIGEnvironment(_CyborgDouble(), observation_width=8))


    def test_cityflow_records_applied_not_invalid_padded_action(self):
        import json, tempfile
        roadnet = {
            "intersections": [
                {"id": "i0", "virtual": False, "roads": ["r0"], "trafficLight": {"lightphases": [{}, {}]}},
                {"id": "i1", "virtual": False, "roads": ["r1"], "trafficLight": {"lightphases": [{}, {}, {}]}},
            ],
            "roads": [{"id": "x", "startIntersection": "i0", "endIntersection": "i1"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/roadnet.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(roadnet, handle)
            env = CityFlowCIGEnvironment(
                _CityEngineDouble(), path, max_steps=5, observation_width=8
            )
            env.reset(seed=1)
            env.step([2, 2])  # phase 2 is padded/invalid for i0 but valid for i1
            self.assertEqual(env.last_actions, [0, 2])
            self.assertEqual(env.raw_env.phases, {"i0": 0, "i1": 2})

    def test_cityflow_contract(self):
        import json, tempfile
        roadnet = {
            "intersections": [
                {"id": "i0", "virtual": False, "roads": ["r0"], "trafficLight": {"lightphases": [{}, {}]}},
                {"id": "i1", "virtual": False, "roads": ["r1"], "trafficLight": {"lightphases": [{}, {}, {}]}},
            ],
            "roads": [{"id": "x", "startIntersection": "i0", "endIntersection": "i1"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/roadnet.json"
            with open(path, "w", encoding="utf-8") as handle: json.dump(roadnet, handle)
            self._exercise(CityFlowCIGEnvironment(_CityEngineDouble(), path, max_steps=5, observation_width=8))


if __name__ == "__main__":
    unittest.main()
