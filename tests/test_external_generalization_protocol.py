"""Regression tests for external v4 train/evaluate separation."""
from __future__ import annotations

import unittest

import numpy as np

from scripts.run_external_training import (
    _cfg_for,
    _evaluate_frozen_policy,
)
from utils.paper_contracts import EXTERNAL_EVAL_SEED_OFFSET


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


class _Runner:
    n_agents = 2
    action_dim = 2
    obs_dim = 3

    def __init__(self):
        self.env = _Env()
        self.env_adapter = object()
        self.cfg = {}
        self.forcer = _Forcer()
        self._interaction_step = 11
        self.calls = 0

    def collect_episode(self):
        self.calls += 1
        assert self.cfg["freeze_representation_state"] is True
        assert self.forcer.eps == 0.0
        assert self.forcer.eps_initial == 0.0
        # Frozen external evaluation must never carry a forced action.
        trajectory = [
            {"forced_mask": np.asarray([False, False], dtype=bool)}
            for _ in range(3)
        ]
        return trajectory, np.asarray([1.0, 3.0]), 0.01


class ExternalGeneralizationProtocolTests(unittest.TestCase):
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
        self.assertEqual(runner.forcer.eps, 0.05)
        self.assertEqual(runner.forcer.eps_initial, 0.05)
        self.assertEqual(runner.forcer.episode, 7)
        np.testing.assert_allclose(runner.forcer._eps_per_agent, [0.02, 0.08])
        self.assertEqual(runner._interaction_step, 11)
        self.assertIs(runner.env_adapter, original_adapter)


if __name__ == "__main__":
    unittest.main()
