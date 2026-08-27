"""Regression tests for external v4 train/evaluate separation."""
from __future__ import annotations

import unittest
import copy
import pickle

import numpy as np

from scripts.run_external_training import (
    _cfg_for,
    _evaluate_frozen_policy,
)
from utils.paper_contracts import (
    EXTERNAL_EVAL_SEED_OFFSET,
    EXTERNAL_GENERALIZATION_PROTOCOL_VERSION,
)


class _CfgRE:
    @staticmethod
    def default_cfg():
        return {
            "k0_warmup": 30,
            "causal_horizon": 8,
            "proxy_n_horizons": 8,
        }

    @staticmethod
    def smoke_cfg():
        return {
            "k0_warmup": 5,
            "causal_horizon": 2,
            "proxy_n_horizons": 2,
        }

    @staticmethod
    def set_global_seed(seed):
        np.random.seed(int(seed))


class _Env:
    n_agents = 2
    max_steps = 5

    def get_action_dim(self): return 2
    def get_obs_dim(self): return 3


class _Forcer:
    def __init__(self):
        self.eps_initial = 0.05
        self.eps = 0.05
        self.anneal_to = 0.05
        self.episode = 7
        self._eps_per_agent = np.asarray([0.02, 0.08])
        self.total_steps = 10
        self.total_forced = 2
        self.total_eligible = 20
        self.last_execution_records = ("sentinel",)
        self.rng = np.random.RandomState(123)

    def state_dict(self):
        return {
            "eps_initial": self.eps_initial,
            "eps": self.eps,
            "anneal_to": self.anneal_to,
            "episode": self.episode,
            "eps_per_agent": None if self._eps_per_agent is None else self._eps_per_agent.copy(),
            "total_steps": self.total_steps,
            "total_forced": self.total_forced,
            "total_eligible": self.total_eligible,
            "rng_state": self.rng.get_state(),
        }

    def load_state_dict(self, state):
        self.eps_initial = float(state["eps_initial"])
        self.eps = float(state["eps"])
        self.anneal_to = state["anneal_to"]
        self.episode = int(state["episode"])
        values = state["eps_per_agent"]
        self._eps_per_agent = None if values is None else np.asarray(values).copy()
        self.total_steps = int(state["total_steps"])
        self.total_forced = int(state["total_forced"])
        self.total_eligible = int(state["total_eligible"])
        self.rng.set_state(state["rng_state"])


class _PairModule:
    def __init__(self):
        self.full_states = {}
        self.shadow_states = {(0, 1): np.asarray([4.0], dtype=np.float32)}
        self.pooled_states = {}
        self.active_core_pairs = set()
        self.candidate_neighbors_by_ego = {0: (1,), 1: (0,)}
        self.bc_buffer = []


class _Runner:
    n_agents = 2
    action_dim = 2
    obs_dim = 3

    def __init__(self):
        self.env = _Env()
        self.env_adapter = object()
        self.cfg = {}
        self.forcer = _Forcer()
        self.pair_rel_module = _PairModule()
        self._interaction_step = 11
        self.calls = 0
        self.start_states = []

    def collect_episode(self):
        self.calls += 1
        assert self.cfg["freeze_representation_state"] is False
        assert self.cfg["freeze_representation_learning_state"] is True
        assert self.forcer.eps == 0.0
        assert self.forcer.eps_initial == 0.0
        # Real EpsilonForcedActionController.apply() consumes its private RNG
        # even at eps=0, so frozen evaluation must restore that RNG exactly.
        self.forcer.rng.rand(2)
        self.forcer.total_steps += 2
        self.forcer.total_eligible += 2
        recurrent_state = float(self.pair_rel_module.shadow_states[(0, 1)][0])
        self.start_states.append(recurrent_state)
        # Stand in for deployment-time recurrent filtering.  The external
        # evaluator must allow this within an episode but reset it before the
        # next fresh paired episode.
        self.pair_rel_module.shadow_states[(0, 1)][0] += 3.0
        # Frozen external evaluation must never carry a forced action.
        trajectory = [
            {"forced_mask": np.asarray([False, False], dtype=bool)}
            for _ in range(3)
        ]
        return trajectory, np.asarray([1.0, 3.0]), 0.01


class ExternalGeneralizationProtocolTests(unittest.TestCase):
    def test_protocol_version_marks_recurrent_inference_semantics(self):
        self.assertIn("v5", EXTERNAL_GENERALIZATION_PROTOCOL_VERSION)
        self.assertIn("recurrent", EXTERNAL_GENERALIZATION_PROTOCOL_VERSION)

    def test_full_profile_preserves_paper_warmup_and_horizons(self):
        env = _Env()
        full = _cfg_for(_CfgRE, "full", 4, env, 60)
        self.assertEqual(full["k0_warmup"], 30)
        self.assertEqual(full["causal_horizon"], 8)
        self.assertEqual(full["proxy_n_horizons"], 8)

        quick = _cfg_for(_CfgRE, "quick", 4, env, 60)
        self.assertEqual(quick["k0_warmup"], 0)
        self.assertEqual(quick["causal_horizon"], 4)
        self.assertEqual(quick["proxy_n_horizons"], 4)

    def test_frozen_evaluation_disables_forcing_and_restores_runner_state(self):
        runner = _Runner()
        original_adapter = runner.env_adapter
        forcer_before = pickle.dumps(
            copy.deepcopy(runner.forcer.state_dict()),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        rows = _evaluate_frozen_policy(
            runner=runner,
            model="Final-CIGAMF",
            train_seed=3,
            eval_episodes=2,
            environment="cyborg",
            agent_count=2,
            max_steps=5,
            config_path=None,
            config_fingerprint="abc",
            RE=_CfgRE,
            build_environment=lambda *args, **kwargs: _Env(),
            require_panel=lambda env, panel: None,
            resolve_env_adapter=lambda env, action_dim=None: object(),
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["eval_seed"], EXTERNAL_EVAL_SEED_OFFSET + 30_000)
        self.assertEqual(rows[1]["eval_seed"], EXTERNAL_EVAL_SEED_OFFSET + 30_001)
        self.assertEqual(rows[0]["eval_reward"], 2.0)
        self.assertEqual(rows[0]["forcing_count"], 0)

        self.assertNotIn("freeze_representation_state", runner.cfg)
        self.assertNotIn("freeze_representation_learning_state", runner.cfg)
        self.assertEqual(runner.forcer.eps, 0.05)
        self.assertEqual(runner.forcer.eps_initial, 0.05)
        self.assertEqual(runner.forcer.episode, 7)
        np.testing.assert_allclose(runner.forcer._eps_per_agent, [0.02, 0.08])
        self.assertEqual(
            forcer_before,
            pickle.dumps(
                runner.forcer.state_dict(), protocol=pickle.HIGHEST_PROTOCOL
            ),
        )
        self.assertEqual(runner.forcer.last_execution_records, ("sentinel",))
        self.assertEqual(runner._interaction_step, 11)
        self.assertEqual(runner.start_states, [4.0, 4.0])
        self.assertEqual(float(runner.pair_rel_module.shadow_states[(0, 1)][0]), 4.0)
        self.assertIs(runner.env_adapter, original_adapter)


if __name__ == "__main__":
    unittest.main()
