"""Regression tests for the gated lag-specific oracle measurement."""
import copy
import unittest

import numpy as np

from envs.omni_arena import OmniArena
from scripts.run_latency_oracle import _rank_correlation, run_gate


class LatencyOracleTests(unittest.TestCase):
    def test_rank_correlation_uses_average_ranks_for_tied_delays(self):
        truth = [0, 0, 2, 2, 4, 4]
        estimate = [0.1, 0.2, 2.1, 2.2, 4.1, 4.2]
        base = _rank_correlation(truth, estimate)
        permutation = [1, 0, 3, 2, 5, 4]
        permuted = _rank_correlation(
            [truth[index] for index in permutation],
            [estimate[index] for index in permutation],
        )
        self.assertAlmostEqual(base, permuted)

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

    def test_all_action_capacity_spectrum_recovers_randomized_delay(self):
        result = run_gate(seed=0, n_states=3, horizon=8, n_trials=1)
        self.assertTrue(result["gate_pass"])
        self.assertGreaterEqual(result["randomized_delay_rank_correlation"], 0.70)
        self.assertLessEqual(result["randomized_delay_mae"], 1.5)
        for row in result["rows"]:
            self.assertEqual(len(row["capacity_lag_spectrum"]), 8)
            self.assertEqual(
                len(row["action_lag_response"]), result["action_count"]
            )


if __name__ == "__main__":
    unittest.main()
