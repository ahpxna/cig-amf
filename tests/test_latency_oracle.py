"""Regression tests for the gated lag-specific oracle measurement."""
import copy
import unittest

import numpy as np

from envs.omni_arena import OmniArena


class LatencyOracleTests(unittest.TestCase):
    def _env(self):
        return OmniArena(
            n_agents=24,
            n_zones=4,
            max_steps=30,
            phase_length=6,
            causal_horizon=4,
            seed=17,
            mode="cooperative",
        )

    def test_lag_oracle_returns_impulse_profile_and_restores_state(self):
        env = self._env()
        before = copy.deepcopy(env.clone_state())
        roles = env.zone_role_agents[0]
        result = env.compute_oracle_lag_response_from_current_state(
            ego_id=roles[env.ROLE_COLLECTOR],
            agent_j=roles[env.ROLE_BLOCKER],
            intervention_action=env.OPEN,
            horizon=4,
            n_trials=2,
            crn_seed=71,
        )
        self.assertEqual(result["per_lag_response"].shape, (4,))
        self.assertTrue(np.all(np.isfinite(result["per_lag_response"])))
        self.assertTrue(np.isfinite(result["discounted_response"]))
        after = env.clone_state()
        self.assertEqual(after["positions"], before["positions"])
        self.assertEqual(after["last_actions"], before["last_actions"])
        self.assertEqual(after["t"], before["t"])
        np.testing.assert_array_equal(after["rng_state"][1], before["rng_state"][1])

    def test_lag_oracle_rejects_boundary_crossing_window(self):
        env = self._env()
        for _ in range(28):
            actions = [env.scripted_policy(agent) for agent in range(env.n_agents)]
            env.step(actions)
        roles = env.zone_role_agents[0]
        with self.assertRaises(AssertionError):
            env.compute_oracle_lag_response_from_current_state(
                ego_id=roles[env.ROLE_COLLECTOR],
                agent_j=roles[env.ROLE_BLOCKER],
                intervention_action=env.STAY,
                horizon=4,
            )


if __name__ == "__main__":
    unittest.main()
